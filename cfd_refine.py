"""CFD + FEA refinement endpoints for the Lumexa backend (drop next to main.py, hooked in with apply_hook.py).

  POST /cfd-analyze-async   build123d script + flow + supports + criteria -> job_id   (poll the existing GET /job/{id})
  GET  /cfd-health          state of the CFD service and of the CalculiX service (incl. CFD-load support)
  GET  /cfd-selftest        runs the CFD service's known-answer test

One job does:  script -> B-rep -> STL -> OpenFOAM (steady RANS) -> wall pressure + shear -> conservative transfer onto the
FE mesh -> CalculiX static analysis -> ONE combined verdict (structure + fluid + coupling assessment) with concrete
refinement hints for the agent. The result carries the fields the Supabase runner already reads
(refinement.passed_quality_gate, analytical_fea.*, analysis_report_md, generated_stl_base64) plus `cfd` and `combined`.
"""
import asyncio
import base64
import json
import math
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from fastapi import BackgroundTasks, Form, HTTPException

CFD_SERVICE_URL = os.environ.get("CFD_SERVICE_URL", "").strip().rstrip("/")
CFD_API_KEY = os.environ.get("CFD_API_KEY", "").strip()
CFD_TIMEOUT_S = float(os.environ.get("CFD_TIMEOUT_S", "1700"))
ASSUMPTIONS = [
    "The part is a free body in a uniform stream; the support structure and any wall it is mounted on are not part of the flow. Fixed faces are wetted in the CFD, their share of the load goes straight into the support.",
    "Steady RANS (k-omega SST, wall functions): mean loads only. Vortex shedding, buffeting and other unsteady effects are not captured.",
    "Loads are applied statically (pressure + wall shear, conservative transfer); no dynamic amplification factor unless load_scale is set.",
    "One-way coupling: the flow does not react to the part's deformation (see combined.coupling for when that is acceptable).",
]
FSI_WEAK_COUPLING_LIMIT = float(os.environ.get("CFD_WEAK_COUPLING_LIMIT", "0.01"))   # deflection / part size


# ------------------------------------------------------------------------------------------------ HTTP to the CFD service
def cfd_call(multipart, path, fields=None, files=None, timeout=60):
    headers = {"X-API-KEY": CFD_API_KEY, "ngrok-skip-browser-warning": "1", "User-Agent": "LumexaBackend/1.0"}
    data = None
    if fields is not None or files is not None:
        data, headers["Content-Type"] = multipart(fields or {}, files or {})
    req = urllib.request.Request(CFD_SERVICE_URL + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"CFD service HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}")
    except Exception as e:
        raise RuntimeError(f"CFD service unreachable ({type(e).__name__}: {str(e)[:160]})")
    try:
        return json.loads(raw)
    except ValueError:
        raise RuntimeError(f"CFD service returned non-JSON: {raw[:200]!r}")


# ------------------------------------------------------------------------------------------------ judgement
def _check(name, value, limit, kind):
    """kind 'max': value must be <= limit; 'min': value must be >= limit."""
    ok = value is not None and (value <= limit if kind == "max" else value >= limit)
    return {"name": name, "value": None if value is None else round(float(value), 4), "limit": limit, "rule": "<=" if kind == "max" else ">=", "pass": bool(ok)}


def evaluate(cfd_s, fem, mat, criteria, L_part_mm):
    vm, disp = float(fem["vm_mpa"]), float(fem["disp_mm"])
    sy = float(mat["yield_strength_mpa"])
    sf = sy / vm if vm > 1e-9 else 999.0
    c = criteria or {}
    checks = [_check("safety_factor", sf, float(c.get("min_safety_factor", 2.0)), "min")]
    if c.get("max_drag_n") is not None:
        checks.append(_check("drag_n", cfd_s["drag_n"], float(c["max_drag_n"]), "max"))
    if c.get("max_cd") is not None:
        checks.append(_check("Cd", cfd_s["Cd"], float(c["max_cd"]), "max"))
    if c.get("max_deflection_mm") is not None:
        checks.append(_check("deflection_mm", disp, float(c["max_deflection_mm"]), "max"))
    if c.get("max_stress_mpa") is not None:
        checks.append(_check("von_mises_mpa", vm, float(c["max_stress_mpa"]), "max"))
    if c.get("max_lift_n") is not None:
        checks.append(_check("lift_n", abs(cfd_s["lift_n"]), float(c["max_lift_n"]), "max"))
    drag_stable = cfd_s.get("drag_change_last_write_pct") is not None and cfd_s["drag_change_last_write_pct"] <= 2.0
    cfd_ok = bool(cfd_s.get("converged")) or drag_stable
    defl_ratio = disp / max(L_part_mm, 1e-9)
    coupling = {"deflection_over_part_size": round(defl_ratio, 5), "limit": FSI_WEAK_COUPLING_LIMIT,
                "one_way_valid": defl_ratio < FSI_WEAK_COUPLING_LIMIT,
                "note": ("deflection is small compared with the part: the fluid loads would barely change if the part deformed, so one-way coupling is accurate"
                         if defl_ratio < FSI_WEAK_COUPLING_LIMIT else
                         "the part deflects enough to change the flow: one-way loads are only a first estimate, run two-way FSI (preCICE) for the final answer")}
    return checks, sf, cfd_ok, coupling


def hints(cfd_s, fem, sf, checks, flow, criteria, cfd_ok, coupling, mesh):
    h = []
    fails = {c["name"] for c in checks if not c["pass"]}
    vm = float(fem["vm_mpa"])
    hot = fem.get("hotspot_xyz_mm")
    if not cfd_ok:
        h.append("CFD not converged: results are not trustworthy yet. Raise cfd.max_iterations, lower cfd.relax (e.g. 0.7) or use a simpler shape before judging the design.")
    if "safety_factor" in fails:
        h.append(f"Safety factor {sf:.2f} is below target; the stress peak (von Mises {vm:.1f} MPa) is at {[round(v, 1) for v in hot] if hot else 'n/a'} mm. "
                 "Add a fillet there (radius >= 1/10 of the local thickness), thicken the section, or move the support closer to the loaded region.")
    if "drag_n" in fails or "Cd" in fails:
        pd, vd, dr = cfd_s["drag_pressure_n"], cfd_s["drag_viscous_n"], max(abs(cfd_s["drag_n"]), 1e-12)
        if abs(pd) >= 0.6 * dr:
            h.append(f"Drag is pressure-dominated ({100 * pd / dr:.0f}%, Cd {cfd_s['Cd']:.2f}): round or chamfer the windward edges, taper the rear (boat-tail) and reduce the frontal area ({cfd_s['A_ref_m2'] * 1e6:.0f} mm2).")
        else:
            h.append(f"Drag has a large friction share ({100 * vd / dr:.0f}%): reduce wetted area and avoid long flat surfaces parallel to the flow.")
    if "deflection_mm" in fails:
        h.append("Deflection above the limit: increase the second moment of area in the load direction (taller section, ribs), or shorten the unsupported length.")
    if abs(cfd_s.get("Cl") or 0) > 0.3:
        h.append(f"Large side force (lift {cfd_s['lift_n']:.3g} N, Cl {cfd_s['Cl']:.2f}): the pressure field is asymmetric; check for unintended camber or an off-axis support, or make the part symmetric about the flow axis.")
    if cfd_s.get("cp_max", 0) > 0.9 and "safety_factor" in fails:
        h.append(f"Stagnation pressure ({cfd_s['p_max_pa']:.0f} Pa at {[round(v, 1) for v in cfd_s['p_max_at_mm']]} mm) loads the windward face directly; a rounded nose lowers it.")
    if not coupling["one_way_valid"]:
        h.append(f"Deflection is {100 * coupling['deflection_over_part_size']:.1f}% of the part size: loads will change with deformation, so treat results as a first estimate (two-way FSI recommended).")
    if not fails and cfd_ok and sf > 3.0 * float(criteria.get("min_safety_factor", 2.0)):
        h.append(f"All checks pass with a large margin (SF {sf:.1f}): material can be removed, e.g. thinner walls or pockets in low-stress regions, to save weight.")
    if mesh and not mesh.get("ok"):
        h.append("The CFD mesh has failed quality checks (sharp internal corners or very thin features): simplify tiny details for CFD.")
    return h


def report_md(name, flow_info, cfd_s, fem, sf, checks, coupling, warnings, hint_list, passed, mat_key):
    L = ["# CFD + structural analysis: " + name, "", f"**Verdict: {'PASS' if passed else 'FAIL'}**", "",
         "## Flow", f"- {flow_info['U']} m/s, {'air' if flow_info['rho'] < 10 else 'water'}, Re = {flow_info['Re']:.3g}, turbulence: {flow_info['turbulence']}, "
         f"frontal area {flow_info['A_ref_m2'] * 1e6:.0f} mm2",
         f"- Drag **{cfd_s['drag_n']:.4g} N** (pressure {cfd_s['drag_pressure_n']:.4g}, friction {cfd_s['drag_viscous_n']:.4g}), Cd {cfd_s['Cd']:.3f}",
         f"- Lift {cfd_s['lift_n']:.4g} N, side {cfd_s['side_n']:.4g} N; Cp max {cfd_s['cp_max']:.2f} / min {cfd_s['cp_min']:.2f}",
         f"- Converged: {cfd_s.get('converged')} after {cfd_s.get('iterations')} iterations (drag change over the last write: {cfd_s.get('drag_change_last_write_pct')}%)",
         "", "## Structure (CalculiX, loaded by the CFD wall pressure and shear)",
         f"- Material {mat_key}; max von Mises **{float(fem['vm_mpa']):.2f} MPa**; safety factor **{sf:.2f}**; max displacement {float(fem['disp_mm']):.4f} mm",
         "", "## Checks", "", "| check | value | rule | limit | result |", "|---|---|---|---|---|"]
    L += [f"| {c['name']} | {c['value']} | {c['rule']} | {c['limit']} | {'pass' if c['pass'] else 'FAIL'} |" for c in checks]
    L += ["", "## Coupling", f"- {coupling['note']} (deflection / part size = {coupling['deflection_over_part_size']:.4f})"]
    if warnings:
        L += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    if hint_list:
        L += ["", "## Refinement hints"] + [f"- {x}" for x in hint_list]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ the pipeline
def analyze(ctx, p):
    """Blocking. Returns the job result dict. Raises RuntimeError for infrastructure problems (the job then ends in 'error')."""
    TP, MATERIALS = ctx["TP"], ctx["MATERIALS"]
    progress = ctx["progress"]
    name = p.get("part_name") or "part"
    mat_key = p.get("material") or "aluminum_6061"
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    t0 = time.time()
    base = {"part_name": name, "fea_method": "cfd_coupled_calculix", "cfd_used": True, "calculix_used": True, "simscale_used": False,
            "lumexa_cfd_version": "1"}

    def fail(reason, design=True, **extra):
        return {**base, "refinement": {"passed_quality_gate": False, "final_reasons": [reason], "mode": "cfd_fea"}, **extra}

    progress(stage="building the part")
    obj, err = ctx["execute"](p["script"])
    if err or obj is None:
        return fail(f"script error: {str(err)[:400]}", execution_error=str(err))
    try:
        descs = TP.describe_faces(obj)
        sels = p["fixed"] if isinstance(p["fixed"], list) else [p["fixed"]]
        fixed_idx = sorted({i for n, s in enumerate(sels) for i in TP.select_faces(descs, s, f"fixed[{n}]")})
    except Exception as e:
        return fail(f"supports: {e}", execution_error=None, analysis_error=str(e))
    lo, hi = TP.part_bbox(descs)
    L_mm = float(max(hi[k] - lo[k] for k in range(3)))

    tmp = tempfile.mkdtemp(prefix="cfdjob_")
    try:
        stl_path, step_path = os.path.join(tmp, "part.stl"), os.path.join(tmp, "part.step")
        ctx["write_stl"](obj, stl_path, 0.02, 0.1)
        ctx["write_step"](obj, step_path)
        stl_bytes, step_bytes = open(stl_path, "rb").read(), open(step_path, "rb").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- CFD
    progress(stage="waking the CFD service")
    last = None
    for _ in range(12):
        try:
            h = cfd_call(ctx["multipart"], "/health", timeout=40)
            last = None
            break
        except RuntimeError as e:
            last = e
            time.sleep(8)
    if last is not None:
        raise RuntimeError(str(last))
    if not h.get("ok"):
        raise RuntimeError(f"CFD service is up but OpenFOAM is not ready: {json.dumps(h)[:300]}")
    flow = dict(p.get("flow") or {})
    flow["return_surface"] = True
    flow.setdefault("units", "mm")
    progress(stage="CFD (OpenFOAM)")
    resp = cfd_call(ctx["multipart"], "/solve", fields={"spec": json.dumps(flow)}, files={"stl": ("part.stl", stl_bytes)}, timeout=CFD_TIMEOUT_S)
    if not resp.get("ok"):
        msg = f"CFD [{resp.get('stage')}]: {resp.get('reason')}"
        if resp.get("design_related") or resp.get("stage") in ("geometry", "mesh", "features", "setup"):
            return fail(msg, analysis_error=msg, cfd_diag={"stage": resp.get("stage"), "design_related": bool(resp.get("design_related"))})
        raise RuntimeError(msg)
    cfd = resp["result"]
    cfd_s = cfd["summary"]

    # ---- FEA with the CFD loads
    def fdesc(i):
        d = descs[i]
        return {"center": [float(v) for v in d["center"]], "area": float(d["area"]), "bbox_min": [float(v) for v in d["bbox_min"]],
                "bbox_max": [float(v) for v in d["bbox_max"]], "geom": d["geom"]}
    case = {"material": {"E_mpa": float(mat["youngs_modulus_gpa"]) * 1000.0, "nu": float(mat["poissons_ratio"]), "name": mat_key},
            "bbox_mm": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]}, "min_thickness_mm": TP.min_thickness(descs),
            "analysis": "static", "faces": {"fixed": [fdesc(i) for i in fixed_idx], "loads": []},
            "cfd_load": {"surface": cfd["surface"], "scale": float(p.get("load_scale", 1.0))}}
    progress(stage="structural analysis (CalculiX)")
    try:
        r2 = ctx["ccx_call"]("/solve", fields={"case": json.dumps(case)}, files={"step": ("part.step", step_bytes)}, timeout=ctx["ccx_timeout"])
    except RuntimeError as e:
        raise RuntimeError(str(e))
    if not r2.get("ok"):
        msg = f"CalculiX [{r2.get('stage')}]: {r2.get('reason')}"
        if r2.get("design_related"):
            return fail(msg, analysis_error=msg, cfd={"summary": cfd_s})
        raise RuntimeError(msg)
    fem = r2["result"]
    xfer = (fem.get("bc_check") or {}).get("cfd_load")
    if not xfer:
        raise RuntimeError("the CalculiX service is an old version without CFD-load support: update analysis_service (pressure_load.py + app.py) and restart it")

    # ---- judgement
    criteria = dict(p.get("criteria") or {})
    checks, sf, cfd_ok, coupling = evaluate(cfd_s, fem, mat, criteria, L_mm)
    warnings = list(cfd.get("warnings", []))
    if xfer["net_error_pct"] > 2.0:
        warnings.append(f"load transfer lost {xfer['net_error_pct']}% of the net force: check the supports and the mesh")
    if any(abs(v) > 0.2 * max(abs(x) for x in xfer["cfd_net_n"] + [1e-12]) for v in xfer["lost_on_fixed_n"]):
        warnings.append("more than 20% of the fluid force acts directly on the fixed faces (it goes straight into the support)")
    reasons = [f"{c['name']} {c['value']} violates {c['rule']} {c['limit']}" for c in checks if not c["pass"]]
    if not cfd_ok:
        reasons.append("CFD is not converged (drag still changing): results are not reliable")
    passed = not reasons
    hint_list = hints(cfd_s, fem, sf, checks, flow, criteria, cfd_ok, coupling, cfd.get("mesh"))
    fem_light = {k: fem.get(k) for k in ("vm_mpa", "disp_mm", "hotspot_xyz_mm", "mesh", "timings") if k in fem}
    md = report_md(name, cfd["flow"], cfd_s, fem, sf, checks, coupling, warnings, hint_list, passed, mat_key)
    return {**base,
            "refinement": {"passed_quality_gate": passed, "final_reasons": reasons, "mode": "cfd_fea"},
            "analytical_fea": {"method": "cfd_coupled_calculix", "safety_factor": round(sf, 4),
                               "stress": {"von_mises_mpa": round(float(fem["vm_mpa"]), 4)}, "deflection_mm": round(float(fem["disp_mm"]), 5)},
            "combined": {"verdict": "pass" if passed else "fail", "assumptions": ASSUMPTIONS, "checks": checks, "coupling": coupling, "hints": hint_list, "warnings": warnings,
                         "structure": {"von_mises_mpa": round(float(fem["vm_mpa"]), 3), "safety_factor": round(sf, 3),
                                       "deflection_mm": round(float(fem["disp_mm"]), 5), "hotspot_mm": fem.get("hotspot_xyz_mm")},
                         "fluid": {k: cfd_s.get(k) for k in ("drag_n", "Cd", "lift_n", "side_n", "Cl", "cp_max", "cp_min", "p_max_pa", "p_max_at_mm",
                                                              "converged", "iterations", "drag_change_last_write_pct")},
                         "load_transfer": xfer},
            "cfd": {"summary": cfd_s, "flow": cfd["flow"], "mesh": cfd["mesh"], "versions": cfd.get("versions"), "timings": resp.get("timings")},
            "fea": fem_light,
            "required_tests_status": None, "execution_error": None, "analysis_error": None,
            "analysis_report_md": md, "generated_stl_base64": base64.b64encode(stl_bytes).decode(),
            "timings": {"total_s": round(time.time() - t0, 1)}}


# ------------------------------------------------------------------------------------------------ routes
def install(app, ctx):
    """ctx: JOB_STORE, execute, write_step, write_stl, ccx_call, multipart, MATERIALS, TP, sanitize, max_jobs, job_id_var, progress, ccx_timeout"""
    JOB_STORE, san = ctx["JOB_STORE"], ctx["sanitize"]

    @app.post("/cfd-analyze-async")
    async def cfd_analyze_async(background_tasks: BackgroundTasks,
                                script: str = Form(...), part_name: str = Form("part"), material: str = Form("aluminum_6061"),
                                flow: str = Form(...), fixed: str = Form(...), criteria: str = Form("{}"), load_scale: float = Form(1.0)):
        if not CFD_SERVICE_URL:
            raise HTTPException(503, "CFD_SERVICE_URL is not set on this backend")
        try:
            flow_d, fixed_d, crit_d = json.loads(flow), json.loads(fixed), json.loads(criteria or "{}")
        except ValueError as e:
            raise HTTPException(400, f"flow, fixed and criteria must be JSON: {e}")
        if not isinstance(flow_d, dict) or flow_d.get("velocity_ms") is None:
            raise HTTPException(400, "flow needs at least {\"velocity_ms\": ..}")
        running = [j for j, v in JOB_STORE.items() if v.get("status") == "running" and time.time() - v.get("created", 0) < 5400]
        if len(running) >= ctx["max_jobs"]:
            raise HTTPException(409, f"{len(running)} job(s) already running (job {running[0]}). Jobs run one at a time on this server.")
        job_id = str(uuid.uuid4())
        JOB_STORE[job_id] = {"status": "running", "created": time.time(), "filename": f"cfd:{part_name}"[:80]}
        params = {"script": script, "part_name": part_name, "material": material, "flow": flow_d, "fixed": fixed_d, "criteria": crit_d, "load_scale": load_scale}

        async def run_job():
            ctx["job_id_var"].set(job_id)
            try:
                result = await asyncio.to_thread(analyze, ctx, params)
                JOB_STORE[job_id] = {"status": "complete", "result": result, "created": time.time()}
            except Exception as e:
                JOB_STORE[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}", "created": time.time()}

        background_tasks.add_task(run_job)
        return {"job_id": job_id, "status": "running", "message": "CFD + FEA started. Poll GET /job/{job_id}.",
                "estimated_time": "1-10 minutes depending on the CFD quality level"}

    @app.get("/cfd-health")
    @san
    def cfd_health():
        out = {"cfd_service_url_set": bool(CFD_SERVICE_URL)}
        try:
            out["cfd"] = cfd_call(ctx["multipart"], "/health", timeout=30) if CFD_SERVICE_URL else None
        except RuntimeError as e:
            out["cfd_error"] = str(e)
        try:
            h = ctx["ccx_call"]("/health", timeout=30)
            out["calculix"] = {"ok": h.get("ok"), "version": h.get("version"), "ccx": h.get("ccx")}
        except RuntimeError as e:
            out["calculix_error"] = str(e)
        out["ok"] = bool((out.get("cfd") or {}).get("ok") and (out.get("calculix") or {}).get("ok"))
        if not out["ok"]:                       # the Supabase tick() submits a CFD job only on HTTP 200 from here
            raise HTTPException(503, detail=out)
        return out

    @app.get("/cfd-selftest")
    @san
    def cfd_selftest():
        if not CFD_SERVICE_URL:
            raise HTTPException(503, "CFD_SERVICE_URL is not set")
        try:
            return cfd_call(ctx["multipart"], "/selftest", timeout=300)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
