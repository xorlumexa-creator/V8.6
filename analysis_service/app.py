"""Lumexa solver service: STEP + load case in -> CalculiX linear-static result out.

Pipeline: gmsh (STEP import, tet10 mesh) -> match faces by position/area/bbox -> .inp -> ccx -> .frd -> metrics.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000      Env: CCX_API_KEY (optional shared secret), CCX_BIN (default ccx),
CCX_THREADS (forced to 1 unless CCX_ALLOW_MULTITHREAD is set), CCX_TIMEOUT_S (default 600), MAX_NODES (default 12000)."""
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import fem_core as FC

app = FastAPI(title="Lumexa CalculiX solver")


@app.exception_handler(Exception)
async def _crash_handler(request, exc):
    """Never answer a bare 'Internal Server Error': return the error and the tail of the traceback as JSON, so a
    problem can be read straight from curl without digging through the server's terminal."""
    import traceback
    tail = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(status_code=500, content={
        "ok": False, "stage": "crash", "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
        "trace_tail": "".join(tail)[-1800:]})
CCX_BIN = os.environ.get("CCX_BIN", "ccx")
# CalculiX's SPOOLES solver gives random results on several threads (seen: buckling factors off by 100x, a 6% equilibrium
# error in a plain static solve). These solves take seconds, so run on ONE thread unless multi-thread is explicitly allowed.
CCX_THREADS = os.environ.get("CCX_THREADS", "1") if os.environ.get("CCX_ALLOW_MULTITHREAD") else "1"
CCX_TIMEOUT_S = float(os.environ.get("CCX_TIMEOUT_S", "600"))
MAX_NODES = int(os.environ.get("MAX_NODES", "12000"))
API_KEY = os.environ.get("CCX_API_KEY", "").strip()
_lock = threading.Lock()          # one solve at a time: keeps memory predictable on small machines


def _gmsh_init(gmsh):
    """FastAPI runs these endpoints in worker threads, but gmsh.initialize() installs a Ctrl-C handler by default,
    which Python only allows in the main thread ("signal only works in main thread"). interruptible=False skips it;
    older gmsh versions without that argument get the plain call."""
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()


class SolveError(Exception):
    def __init__(self, stage, msg, design_related=False):
        super().__init__(msg)
        self.stage, self.design_related = stage, design_related


def _auth(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "bad or missing X-API-KEY")


def _gmsh():
    try:
        import gmsh
        return gmsh
    except Exception as e:                      # pragma: no cover - depends on the host
        raise SolveError("setup", f"gmsh python module not available: {e}")


def _ccx_version():
    try:
        r = subprocess.run([CCX_BIN, "-v"], capture_output=True, text=True, timeout=20)
        txt = (r.stdout + r.stderr).strip().splitlines()
        return next((l.strip() for l in txt if "ersion" in l), txt[0] if txt else "unknown")
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


# ---------------------------------------------------------------------------------------------- the solve
def _run_buckling(workdir, node_xyz, elem_ids, conn, fixed_nodes, forces, mat, sc, modes, accuracy, threads, env, tag="a"):
    """Linear buckling in its own CalculiX run: a deck with ONE *BUCKLE step whose load is the reference load (scaled to a
    ~1 N total, which keeps CalculiX from skipping modes). Factors are scaled back to the ACTUAL load. CalculiX's eigenvalue
    solver is known to return random values with SPOOLES on more than 1-2 threads, so every thread setting is forced to
    `threads` (default 1)."""
    t = str(int(threads))
    env = dict(env, OMP_NUM_THREADS=t, CCX_NPROC_EQUATION_SOLVER=t, CCX_NPROC_RESULTS=t, NUMBER_OF_CPUS=t,
               OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t)
    job = f"buck{tag}"
    FC.write_inp(os.path.join(workdir, job + ".inp"), np.array(list(node_xyz)), np.array(list(node_xyz.values())), elem_ids,
                 FC.tet10_gmsh_to_ccx(conn), set(fixed_nodes), forces, float(mat["E_mpa"]), float(mat["nu"]),
                 analysis="buckling", buckle_modes=modes, buckle_scale=sc, buckle_accuracy=accuracy)
    try:
        r = subprocess.run([CCX_BIN, job], cwd=workdir, capture_output=True, text=True, env=env, timeout=CCX_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SolveError("solve", f"CalculiX buckling run exceeded {CCX_TIMEOUT_S:.0f}s")
    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
    dat_p, frd_p = os.path.join(workdir, job + ".dat"), os.path.join(workdir, job + ".frd")
    dat = open(dat_p, errors="ignore").read() if os.path.exists(dat_p) else ""
    raw = FC.parse_buckling_factors(dat)
    if r.returncode != 0 or not raw:
        raise SolveError("results", f"CalculiX buckling run gave no buckling factors (exit {r.returncode}): "
                                    f"{' | '.join(tail)[:500]}")
    facs = [f * sc for f in raw]
    pos = [f for f in facs if f > 0]
    bk = {"factors": [round(f, 4) for f in facs[:5]], "factor": round(min(pos), 4) if pos else None,
          "n_modes": len(facs), "solver_load_scale": sc, "raw_factors": [round(f, 5) for f in raw[:5]]}
    blocks = []
    if os.path.exists(frd_p):
        blocks = FC.parse_frd(open(frd_p, errors="ignore").read()).get("DISP_BLOCKS") or []
    stats = []
    for b in blocks[:4]:
        if b:
            mg = np.linalg.norm(np.array(list(b.values()), float), axis=1)
            stats.append({"nodes": len(b), "max_disp": float(mg.max())})
    if blocks and blocks[0]:
        ids = sorted(blocks[0])
        mg = np.linalg.norm(np.array([blocks[0][i] for i in ids], float), axis=1)
        if mg.max() > 0:
            bk["mode1_peak_xyz_mm"] = [float(c) for c in node_xyz[ids[int(np.argmax(mg))]]]
    bk["debug"] = {"mode_blocks": stats, "ccx_stdout_tail": tail}
    return bk


def _buckling_checked(workdir, node_xyz, elem_ids, conn, fixed_nodes, forces, mat, bscale, modes, accuracy, threads, env):
    """Two buckling runs at load scales 10x apart. A correct linear buckling factor (per actual load) does not depend on the
    size of the reference load; the CalculiX failure modes (random values, a skipped first mode) do. A mismatch marks the
    result as suspect so it can never be passed off as a verdict."""
    a = _run_buckling(workdir, node_xyz, elem_ids, conn, fixed_nodes, forces, mat, bscale, modes, accuracy, threads, env, "a")
    b = _run_buckling(workdir, node_xyz, elem_ids, conn, fixed_nodes, forces, mat, bscale * 0.1, modes, accuracy, threads, env, "b")
    fa, fb = a.get("factor"), b.get("factor")
    agree = bool(fa and fb and abs(fa - fb) <= 0.03 * max(fa, fb))
    a["check"] = {"factor_load_1N": fa, "factor_load_0.1N": fb, "agree_within_3pct": agree}
    if not agree:
        a["suspect"] = True
        a["suspect_reason"] = (f"CalculiX buckling is not repeatable (factor {fa} with a 1 N reference load vs {fb} with 0.1 N); "
                               "a real buckling factor does not depend on the reference load")
    return a


def solve_case(step_path, case, workdir, gmsh=None):
    """Returns (result_dict, timings). Raises SolveError(stage, msg, design_related)."""
    gmsh = gmsh or _gmsh()
    t0 = time.time()
    tm = {}
    mat = case["material"]
    analysis = case.get("analysis", "static")
    if analysis not in ("static", "buckling", "thermal"):
        raise SolveError("setup", f"unknown analysis {analysis!r} (static | buckling | thermal)")
    th_in = case.get("thermal") or {}
    if analysis == "buckling" and not (case.get("faces") or {}).get("loads"):
        raise SolveError("setup", "a buckling case needs at least one load")
    if analysis == "thermal" and not all(k in th_in for k in ("k_w_mk", "alpha_per_c", "t_ref_c")):
        raise SolveError("setup", "a thermal case needs thermal.k_w_mk, thermal.alpha_per_c and thermal.t_ref_c")
    exp_lo, exp_hi = np.asarray(case["bbox_mm"]["min"], float), np.asarray(case["bbox_mm"]["max"], float)
    exp_ext = float(np.linalg.norm(exp_hi - exp_lo))
    _gmsh_init(gmsh)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        try:
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
        except Exception as e:
            raise SolveError("geometry", f"gmsh could not read the STEP file: {e}", design_related=True)
        vols = gmsh.model.getEntities(3)
        if not vols:
            raise SolveError("geometry", "the STEP file contains no solid (only surfaces/curves)", design_related=True)
        bb = gmsh.model.getBoundingBox(-1, -1)
        g_ext = float(np.linalg.norm(np.array(bb[3:6]) - np.array(bb[0:3])))
        k = FC.unit_scale(g_ext, exp_ext)
        if k != 1.0:
            gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, k, k, k)
            gmsh.model.occ.synchronize()
        tm["geometry_s"] = round(time.time() - t0, 2)

        surfaces = []
        for dim, tag in gmsh.model.getEntities(2):
            b = gmsh.model.getBoundingBox(2, tag)
            surfaces.append({"tag": tag, "center": list(gmsh.model.occ.getCenterOfMass(2, tag)),
                             "area": float(gmsh.model.occ.getMass(2, tag)),
                             "bbox_min": list(b[0:3]), "bbox_max": list(b[3:6])})
        faces = case["faces"]
        try:
            fixed_m = FC.match_faces(faces["fixed"], surfaces, exp_ext)
            load_m = [FC.match_faces(l["faces"], surfaces, exp_ext) for l in faces.get("loads", [])]
            temp_m = [FC.match_faces(t["faces"], surfaces, exp_ext) for t in faces.get("temps", [])]
            flux_m = [FC.match_faces(f["faces"], surfaces, exp_ext) for f in faces.get("fluxes", [])]
        except ValueError as e:
            raise SolveError("face_match", str(e))
        _tags = lambda ms: sorted({t for m in ms for t in m["tags"]})
        fixed_tags = _tags(fixed_m)
        load_tags = [_tags(lm) for lm in load_m]
        temp_tags = [_tags(tm) for tm in temp_m]
        flux_tags = [_tags(fm) for fm in flux_m]
        if not fixed_tags:
            raise SolveError("setup", "no fixed face: the part must be held somewhere or the stiffness matrix is singular")
        if set(fixed_tags) & {t for ts in load_tags for t in ts}:
            raise SolveError("face_match", "a fixed face and a loaded face resolved to the same gmsh surface")
        if {t for ts in flux_tags for t in ts} & {t for ts in temp_tags for t in ts}:
            raise SolveError("face_match", "a heat-input face and a temperature face resolved to the same gmsh surface")

        # ---- mesh (tet10), coarsened if it would exceed the node budget
        lc = FC.choose_lc(exp_ext, case.get("min_thickness_mm"))
        max_nodes = int(case.get("max_nodes") or MAX_NODES)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        # a mesh a little over the budget is fine (the estimate is rough and RAM is not the limit at this size);
        # beyond that, coarsen the element size AND the curvature refinement that small fillets/holes trigger
        hard_max = int(max_nodes * 1.3)
        curv = 12
        n_nodes = 0
        for attempt in range(6):
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
            gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
            gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.2)
            try:
                gmsh.model.mesh.generate(3)
            except Exception as e:
                raise SolveError("mesh", f"gmsh meshing failed: {e}", design_related=True)
            n_nodes = len(gmsh.model.mesh.getNodes(-1, -1, True, False)[0])
            if n_nodes <= hard_max:
                break
            lc *= min(1.6, (n_nodes / max_nodes) ** (1 / 3) * 1.1)
            curv = max(4, curv - 2)
        if n_nodes > hard_max:
            raise SolveError("mesh", f"mesh still has {n_nodes} nodes (limit {max_nodes}) at element size {lc:.2f} mm "
                                     "after coarsening: the geometry has many tiny features (small fillets, holes or "
                                     "ribs) - simplify them or raise MAX_NODES", design_related=True)
        tm["mesh_s"] = round(time.time() - t0, 2)

        ntags, ncoords, _ = gmsh.model.mesh.getNodes(-1, -1, True, False)
        coords = np.asarray(ncoords, float).reshape(-1, 3)
        node_xyz = {int(t): coords[i] for i, t in enumerate(ntags)}
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        tets = [(t, n, e) for t, n, e in zip(etypes, etags, enodes) if int(t) == 11]
        if not tets:
            raise SolveError("mesh", f"no 10-node tetrahedra in the mesh (element types {list(etypes)})")
        elem_ids = np.concatenate([np.asarray(e, int) for _, e, _ in tets])
        conn = np.concatenate([np.asarray(n, int).reshape(-1, 10) for _, _, n in tets])

        def surf_nodes(tag):
            return {int(n) for n in gmsh.model.mesh.getNodes(2, tag, True, False)[0]}

        def surf_tris(tag):
            ty, _, nn = gmsh.model.mesh.getElements(2, tag)
            rows = []
            for t, n in zip(ty, nn):
                width = {9: 6, 2: 3}.get(int(t))
                if width:
                    rows.append(np.asarray(n, int).reshape(-1, width))
            return np.concatenate(rows) if rows else np.zeros((0, 6), int)

        fixed_nodes = set().union(*[surf_nodes(t) for t in fixed_tags])
        forces, load_nodes, force_by_load, total = {}, {}, {}, {}
        for l, tags in zip(faces.get("loads", []), load_tags):
            w = {}
            for tg in tags:
                for n, v in FC.tri_node_weights(surf_tris(tg), node_xyz).items():
                    w[n] = w.get(n, 0.0) + v
            fx = FC.distribute_force(w, l["force_xyz"])
            for n, f in fx.items():
                forces[n] = tuple(a + b for a, b in zip(forces.get(n, (0, 0, 0)), f))
            load_nodes[l["id"]], force_by_load[l["id"]] = list(w), l["force_xyz"]
            total[l["id"]] = [round(sum(forces[n][i] for n in w), 4) for i in range(3)]
        temp_bc, cflux, temp_conflicts = {}, {}, 0
        if analysis == "thermal":
            by_node = {}
            for t, tags in zip(faces.get("temps", []), temp_tags):
                for tg in tags:
                    for n in surf_nodes(tg):
                        by_node.setdefault(n, []).append(float(t["temperature_c"]))
            temp_conflicts = sum(1 for v in by_node.values() if max(v) - min(v) > 1e-6)
            temp_bc = {n: sum(v) / len(v) for n, v in by_node.items()}      # an edge node shared by two faces: mean
            if not temp_bc:
                raise SolveError("setup", "a thermal case needs at least one temperature face")
            for f, tags in zip(faces.get("fluxes", []), flux_tags):
                w = {}
                for tg in tags:
                    for n, v in FC.tri_node_weights(surf_tris(tg), node_xyz).items():
                        w[n] = w.get(n, 0.0) + v
                for n, fx in FC.distribute_force(w, [float(f["power_w"]) * 1000.0, 0.0, 0.0]).items():   # W -> mW
                    cflux[n] = cflux.get(n, 0.0) + fx[0]
        tm["setup_s"] = round(time.time() - t0, 2)
    finally:
        gmsh.finalize()

    # ---- CalculiX
    # buckling: the reference load inside the *BUCKLE step is scaled to a ~1 N total; factors are scaled back below
    fsum = sum(float(np.linalg.norm(l["force_xyz"])) for l in faces.get("loads", []))
    bscale = 1.0 / fsum if (analysis == "buckling" and fsum > 0) else 1.0
    inp = os.path.join(workdir, "job.inp")
    FC.write_inp(inp, np.array(list(node_xyz)), np.array(list(node_xyz.values())), elem_ids,
                 FC.tet10_gmsh_to_ccx(conn), fixed_nodes, forces, float(mat["E_mpa"]), float(mat["nu"]),
                 analysis=("static" if analysis == "buckling" else analysis),
                 thermal=({"k_w_mk": th_in["k_w_mk"], "alpha_per_c": th_in["alpha_per_c"], "t_ref_c": th_in["t_ref_c"],
                           "temps": temp_bc, "cflux_mw": cflux} if analysis == "thermal" else None))
    env = dict(os.environ, OMP_NUM_THREADS=CCX_THREADS, CCX_NPROC_EQUATION_SOLVER=CCX_THREADS)
    try:
        r = subprocess.run([CCX_BIN, "job"], cwd=workdir, capture_output=True, text=True, env=env, timeout=CCX_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SolveError("solve", f"CalculiX exceeded {CCX_TIMEOUT_S:.0f}s")
    except FileNotFoundError:
        raise SolveError("setup", f"CalculiX binary '{CCX_BIN}' not found on this machine")
    frd = os.path.join(workdir, "job.frd")
    if r.returncode != 0 or not os.path.exists(frd):
        tail = " | ".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        raise SolveError("solve", f"CalculiX failed (exit {r.returncode}): {tail[:600]}")
    tm["solve_s"] = round(time.time() - t0, 2)

    res = FC.parse_frd(open(frd, errors="ignore").read())
    dat_path = os.path.join(workdir, "job.dat")
    dat_text = open(dat_path, errors="ignore").read() if os.path.exists(dat_path) else ""
    try:
        out = FC.analyse(node_xyz, res["DISP"], res["STRESS"], temps=(res["NDTEMP"] if analysis == "thermal" else None))
    except ValueError as e:
        raise SolveError("results", str(e))
    if analysis == "buckling":
        out["buckling"] = _buckling_checked(workdir, node_xyz, elem_ids, conn, fixed_nodes, forces, mat, bscale,
                                            int(case.get("buckle_modes") or 3), case.get("buckle_accuracy"),
                                            int(case.get("buckle_threads") or 1), env)
    bc = FC.bc_sanity(res["DISP"], fixed_nodes, load_nodes, force_by_load)
    if analysis == "thermal":
        ts = FC.thermal_sanity(res["NDTEMP"], temp_bc, bool(cflux))
        ts["bc_conflicting_nodes"] = temp_conflicts
        bc["thermal"] = ts
        if not ts["ok"]:
            bc["ok"] = False
            bc["reason"] = ts["reason"]
    bc["face_matching"] = {"fixed": fixed_m, "loads": load_m}
    bc["applied_force_n"] = total
    dat_total = FC.parse_dat_totals(dat_text) if dat_text else None
    eq = FC.equilibrium(res.get("FORC"), force_by_load, total=dat_total)
    if eq is not None:
        eq["source"] = "dat_totals" if dat_total is not None else "frd_forc"
        rf = res.get("FORC") or {}
        if rf:                                              # what the .frd RF block says (diagnostic only)
            arr = np.array(list(rf.values()), float)
            eq["frd_forc"] = {"nodes": int(len(rf)), "sum": [round(float(v), 3) for v in arr.sum(axis=0)],
                              "max_abs": [round(float(v), 3) for v in np.abs(arr).max(axis=0)]}
        bc["equilibrium"] = eq
        if eq["error_pct"] > 3.0:
            bc["ok"] = False
            bc["reason"] = (f"reaction forces do not balance the applied loads (error {eq['error_pct']}%): "
                            "a load or a constraint was not applied as intended")
    if case.get("probe_mm"):                       # stress / temperature / displacement at the nodes nearest to given points
        ids_arr = np.array(list(node_xyz))
        pts_arr = np.array(list(node_xyz.values()), float)
        probes = []
        for p_ in case["probe_mm"]:
            k_ = int(np.argmin(np.linalg.norm(pts_arr - np.asarray(p_, float), axis=1)))
            nid = int(ids_arr[k_])
            s6, tt, d3 = res["STRESS"].get(nid), (res.get("NDTEMP") or {}).get(nid), res["DISP"].get(nid)
            probes.append({"at_mm": [float(c) for c in p_], "node_mm": [round(float(c), 3) for c in pts_arr[k_]],
                           "stress_mpa": [round(float(v), 3) for v in s6] if s6 else None,
                           "von_mises_mpa": round(float(FC.vm_from_tensor(s6)[0]), 3) if s6 else None,
                           "t_c": round(float(tt[0]), 3) if tt else None,
                           "disp_mm": [round(float(v), 5) for v in d3] if d3 else None})
        out["probes"] = probes
    out.update({"analysis": analysis, "bc_check": bc, "mesh": {"nodes": n_nodes, "elements": int(len(elem_ids)), "element_size_mm": round(lc, 3),
                                         "unit_scale_applied": k},
                "versions": {"ccx": _ccx_version()}})
    tm["total_s"] = round(time.time() - t0, 2)
    return out, tm


# ---------------------------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    info = {"ok": True, "ccx_bin": CCX_BIN, "ccx": _ccx_version(), "threads": CCX_THREADS, "max_nodes": MAX_NODES,
            "auth_required": bool(API_KEY)}
    try:
        import gmsh
        info["gmsh"] = gmsh.__version__
    except Exception as e:
        info["gmsh"] = f"unavailable ({e})"
        info["ok"] = False
    if "unavailable" in str(info["ccx"]):
        info["ok"] = False
    return info


@app.post("/solve")
def solve(step: UploadFile = File(...), case: str = Form(...), x_api_key: str = Header(default="")):
    _auth(x_api_key)
    try:
        case_d = json.loads(case)
    except ValueError:
        raise HTTPException(400, "case must be JSON")
    work = tempfile.mkdtemp(prefix="ccx_")
    try:
        path = os.path.join(work, "part.step")
        with open(path, "wb") as fh:
            shutil.copyfileobj(step.file, fh)
        with _lock:
            try:
                out, tm = solve_case(path, case_d, work)
                return {"ok": True, "result": out, "timings": tm}
            except SolveError as e:
                return {"ok": False, "stage": e.stage, "reason": str(e), "design_related": e.design_related}
            except Exception as e:
                return {"ok": False, "stage": "internal", "reason": f"{type(e).__name__}: {e}", "design_related": False}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/selftest")
def selftest(x_api_key: str = Header(default="")):
    """Aluminium cantilever 100 x 20 x 10 mm, 100 N down at the free end, compared with beam theory."""
    _auth(x_api_key)
    gmsh = _gmsh()
    work = tempfile.mkdtemp(prefix="ccx_self_")
    try:
        _gmsh_init(gmsh)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("box")
            gmsh.model.occ.addBox(0, 0, 0, 100, 20, 10)
            gmsh.model.occ.synchronize()
            step = os.path.join(work, "box.step")
            gmsh.write(step)
        finally:
            gmsh.finalize()
        E, nu, F, L, b, h = 70000.0, 0.33, 100.0, 100.0, 20.0, 10.0
        case = {"material": {"E_mpa": E, "nu": nu}, "bbox_mm": {"min": [0, 0, 0], "max": [L, b, h]},
                "faces": {"fixed": [{"center": [0, b / 2, h / 2], "area": b * h, "bbox_min": [0, 0, 0], "bbox_max": [0, b, h]}],
                          "loads": [{"id": "tip", "force_xyz": [0, 0, -F],
                                     "faces": [{"center": [L, b / 2, h / 2], "area": b * h,
                                                "bbox_min": [L, 0, 0], "bbox_max": [L, b, h]}]}]},
                "min_thickness_mm": h}
        with _lock:
            try:
                out, tm = solve_case(step, case, work)
            except SolveError as e:
                return {"ok": False, "stage": e.stage, "reason": str(e)}
        I = b * h ** 3 / 12.0
        th_defl, th_sig = F * L ** 3 / (3 * E * I), F * L * (h / 2) / I
        return {"ok": True, "theory": {"deflection_mm": round(th_defl, 4), "max_bending_stress_mpa": round(th_sig, 3)},
                "fe": {"deflection_mm": round(out["disp_mm"], 4), "von_mises_mpa": round(out["vm_mpa"], 3),
                       "hotspot_mm": [round(v, 1) for v in out["hotspot_xyz_mm"]]},
                "ratio": {"deflection": round(out["disp_mm"] / th_defl, 3), "stress": round(out["vm_mpa"] / th_sig, 3)},
                "expect": "deflection ratio ~0.95-1.03; stress ratio ~1.0-1.3 (peak sits at the fixed edge)",
                "bc_check": out["bc_check"], "mesh": out["mesh"], "timings": tm}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _box_step(gmsh, work, dx, dy, dz):
    _gmsh_init(gmsh)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        step = os.path.join(work, "box.step")
        gmsh.write(step)
        return step
    finally:
        gmsh.finalize()


def _x_face(x, b, h):
    return {"center": [x, b / 2, h / 2], "area": b * h, "bbox_min": [x, 0, 0], "bbox_max": [x, b, h]}


@app.get("/selftest-buckling")
def selftest_buckling(x_api_key: str = Header(default="")):
    """Clamped-free square column 10 x 10 x 200 mm, 100 N compression, against Euler: Pcr = pi^2 E I / (2 L)^2."""
    _auth(x_api_key)
    gmsh = _gmsh()
    work = tempfile.mkdtemp(prefix="ccx_selfb_")
    try:
        E, nu, F, L, b, h = 70000.0, 0.33, 100.0, 200.0, 10.0, 10.0
        step = _box_step(gmsh, work, L, b, h)
        case = {"analysis": "buckling", "buckle_modes": 3, "material": {"E_mpa": E, "nu": nu},
                "bbox_mm": {"min": [0, 0, 0], "max": [L, b, h]},
                "faces": {"fixed": [_x_face(0, b, h)],
                          "loads": [{"id": "axial", "force_xyz": [-F, 0, 0], "faces": [_x_face(L, b, h)]}]},
                "min_thickness_mm": 6.0}
        I = b * h ** 3 / 12.0
        theory = math.pi ** 2 * E * I / (2.0 * L) ** 2 / F
        variants, last = [], {}
        for name, extra in (("1thread", {"buckle_threads": 1}),
                            ("1thread_acc1e-4", {"buckle_threads": 1, "buckle_accuracy": 1e-4}),
                            ("2threads_acc1e-4", {"buckle_threads": 2, "buckle_accuracy": 1e-4})):
            with _lock:
                try:
                    out, tm = solve_case(step, {**case, **extra}, work)
                except SolveError as e:
                    variants.append({"variant": name, "ok": False, "stage": e.stage, "reason": str(e)[:400]})
                    continue
            bk = out.get("buckling") or {}
            fe = bk.get("factor")
            last = {"bc_check": out["bc_check"], "mesh": out["mesh"], "timings": tm}
            variants.append({"variant": name, "factors": bk.get("factors"), "raw_factors": bk.get("raw_factors"),
                             "ratio_fe_over_theory": round(fe / theory, 3) if fe else None,
                             "mode1_peak_xyz_mm": bk.get("mode1_peak_xyz_mm"), "check": bk.get("check"),
                             "suspect": bk.get("suspect", False), "debug": bk.get("debug")})
        good = [v for v in variants if v.get("ratio_fe_over_theory") and 0.9 <= v["ratio_fe_over_theory"] <= 1.15
                and not v.get("suspect")]
        return {"ok": bool(good), "theory_buckling_factor": round(theory, 3),
                "best_variant": good[0]["variant"] if good else None, "variants": variants,
                "expect": "ratio ~0.95-1.10, first two factors nearly equal (a degenerate pair), mode-1 peak at the free end x~200",
                **last}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/selftest-thermal")
def selftest_thermal(x_api_key: str = Header(default="")):
    """Two closed-form checks on an aluminium bar 100 x 10 x 10 mm.
    (1) conduction: 100 C at one end, 0 C at the other -> T = 50 C at mid-length (linear profile).
    (2) thermal stress: both ends clamped, uniform 120 C, stress-free at 20 C. With Poisson's ratio 0 the axes decouple
        and the axial stress is exactly -E * alpha * dT = -165.2 MPa everywhere (tight check). With nu = 0.33 the clamps'
        lateral restraint stretches the end zones, the total length cannot change, and the middle compresses a little
        more (about +3%), so that case only has to land within 6%."""
    _auth(x_api_key)
    gmsh = _gmsh()
    work = tempfile.mkdtemp(prefix="ccx_selft_")
    try:
        E, nu, L, b, h, alpha = 70000.0, 0.33, 100.0, 10.0, 10.0, 23.6e-6
        step = _box_step(gmsh, work, L, b, h)
        base = {"analysis": "thermal", "material": {"E_mpa": E, "nu": nu},
                "thermal": {"k_w_mk": 167.0, "alpha_per_c": alpha, "t_ref_c": 20.0},
                "bbox_mm": {"min": [0, 0, 0], "max": [L, b, h]}, "min_thickness_mm": 6.0,
                "probe_mm": [[L / 2, b / 2, h / 2]]}
        cases = {
            "conduction": {**base, "faces": {"fixed": [_x_face(0, b, h)], "loads": [], "temps": [
                {"id": "hot", "temperature_c": 100.0, "faces": [_x_face(0, b, h)]},
                {"id": "cold", "temperature_c": 0.0, "faces": [_x_face(L, b, h)]}]}},
            "restrained_expansion": {**base, "faces": {"fixed": [_x_face(0, b, h), _x_face(L, b, h)], "loads": [], "temps": [
                {"id": "a", "temperature_c": 120.0, "faces": [_x_face(0, b, h)]},
                {"id": "b", "temperature_c": 120.0, "faces": [_x_face(L, b, h)]}]}}}
        cases["restrained_expansion_nu0"] = {**cases["restrained_expansion"], "material": {"E_mpa": E, "nu": 0.0}}
        res, last = {}, {}
        for name, c in cases.items():
            with _lock:
                try:
                    res[name], tm = solve_case(step, c, work)
                except SolveError as e:
                    return {"ok": False, "case": name, "stage": e.stage, "reason": str(e)}
            last = {"mesh": res[name]["mesh"], "timings": tm}
        p1, p2 = res["conduction"]["probes"][0], res["restrained_expansion"]["probes"][0]
        p0 = res["restrained_expansion_nu0"]["probes"][0]
        theory = -E * alpha * 100.0
        sxx = p2["stress_mpa"][0] if p2.get("stress_mpa") else None
        sxx0 = p0["stress_mpa"][0] if p0.get("stress_mpa") else None
        checks = {
            "conduction_T_at_mid_c": {"fe": p1["t_c"], "theory": 50.0,
                                      "pass": p1["t_c"] is not None and abs(p1["t_c"] - 50.0) < 0.6},
            "restrained_axial_stress_at_mid_nu0_mpa": {"fe": sxx0, "theory": round(theory, 2),
                                                       "ratio": round(sxx0 / theory, 4) if sxx0 else None,
                                                       "pass": bool(sxx0 and abs(sxx0 / theory - 1.0) < 0.015)},
            "restrained_axial_stress_at_mid_nu033_mpa": {"fe": sxx, "theory_ignoring_end_effect": round(theory, 2),
                                                         "ratio": round(sxx / theory, 3) if sxx else None,
                                                         "pass": bool(sxx and abs(sxx / theory - 1.0) < 0.06)},
            "restrained_lateral_stress_at_mid_mpa": {"fe": (p2["stress_mpa"][1:3] if p2.get("stress_mpa") else None),
                                                     "theory": [0.0, 0.0],
                                                     "pass": bool(p2.get("stress_mpa") and max(abs(v) for v in p2["stress_mpa"][1:3]) < 8.0)},
            "restrained_temperature_uniform_120c": {"t_range_c": [res["restrained_expansion"].get("t_min_c"),
                                                                  res["restrained_expansion"].get("t_max_c")],
                                                    "pass": all(abs((res["restrained_expansion"].get(k) or 0) - 120.0) < 0.6
                                                                for k in ("t_min_c", "t_max_c"))},
            "boundary_checks_ok": {"pass": all(r["bc_check"].get("ok") for r in res.values())}}
        checks["conduction_T_at_mid_c"]["pass"] = bool(p1["t_c"] is not None and abs(p1["t_c"] - 50.0) < 0.6)
        return {"ok": all(c["pass"] for c in checks.values()), "checks": checks,
                "note": "Closed-form answers; the 3 mm mesh and nodal stress averaging allow a few percent.", **last}
    finally:
        shutil.rmtree(work, ignore_errors=True)
