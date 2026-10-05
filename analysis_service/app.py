"""Lumexa solver service: STEP + load case in -> CalculiX linear-static result out.

Pipeline: gmsh (STEP import, tet10 mesh) -> match faces by position/area/bbox -> .inp -> ccx -> .frd -> metrics.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000      Env: CCX_API_KEY (optional shared secret), CCX_BIN (default ccx),
CCX_THREADS (default 2), CCX_TIMEOUT_S (default 600), MAX_NODES (default 12000)."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

import fem_core as FC

app = FastAPI(title="Lumexa CalculiX solver")
CCX_BIN = os.environ.get("CCX_BIN", "ccx")
CCX_THREADS = os.environ.get("CCX_THREADS", "2")
CCX_TIMEOUT_S = float(os.environ.get("CCX_TIMEOUT_S", "600"))
MAX_NODES = int(os.environ.get("MAX_NODES", "12000"))
API_KEY = os.environ.get("CCX_API_KEY", "").strip()
_lock = threading.Lock()          # one solve at a time: keeps memory predictable on small machines


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
def solve_case(step_path, case, workdir, gmsh=None):
    """Returns (result_dict, timings). Raises SolveError(stage, msg, design_related)."""
    gmsh = gmsh or _gmsh()
    t0 = time.time()
    tm = {}
    mat = case["material"]
    exp_lo, exp_hi = np.asarray(case["bbox_mm"]["min"], float), np.asarray(case["bbox_mm"]["max"], float)
    exp_ext = float(np.linalg.norm(exp_hi - exp_lo))
    gmsh.initialize()
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
            load_m = [FC.match_faces(l["faces"], surfaces, exp_ext) for l in faces["loads"]]
        except ValueError as e:
            raise SolveError("face_match", str(e))
        fixed_tags = sorted({t for m in fixed_m for t in m["tags"]})
        load_tags = [sorted({t for m in lm for t in m["tags"]}) for lm in load_m]
        if set(fixed_tags) & {t for ts in load_tags for t in ts}:
            raise SolveError("face_match", "a fixed face and a loaded face resolved to the same gmsh surface")

        # ---- mesh (tet10), coarsened if it would exceed the node budget
        lc = FC.choose_lc(exp_ext, case.get("min_thickness_mm"))
        max_nodes = int(case.get("max_nodes") or MAX_NODES)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        n_nodes = 0
        for attempt in range(4):
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
            gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.2)
            try:
                gmsh.model.mesh.generate(3)
            except Exception as e:
                raise SolveError("mesh", f"gmsh meshing failed: {e}", design_related=True)
            n_nodes = len(gmsh.model.mesh.getNodes(-1, -1, True, False)[0])
            if n_nodes <= max_nodes:
                break
            lc *= min(1.6, (n_nodes / max_nodes) ** (1 / 3) * 1.1)
        if n_nodes > max_nodes:
            raise SolveError("mesh", f"mesh still has {n_nodes} nodes (> {max_nodes}) at element size {lc:.2f} mm: "
                                     "part too large/complex for this machine", design_related=False)
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
        for l, tags in zip(faces["loads"], load_tags):
            w = {}
            for tg in tags:
                for n, v in FC.tri_node_weights(surf_tris(tg), node_xyz).items():
                    w[n] = w.get(n, 0.0) + v
            fx = FC.distribute_force(w, l["force_xyz"])
            for n, f in fx.items():
                forces[n] = tuple(a + b for a, b in zip(forces.get(n, (0, 0, 0)), f))
            load_nodes[l["id"]], force_by_load[l["id"]] = list(w), l["force_xyz"]
            total[l["id"]] = [round(sum(forces[n][i] for n in w), 4) for i in range(3)]
        tm["setup_s"] = round(time.time() - t0, 2)
    finally:
        gmsh.finalize()

    # ---- CalculiX
    inp = os.path.join(workdir, "job.inp")
    FC.write_inp(inp, np.array(list(node_xyz)), np.array(list(node_xyz.values())), elem_ids,
                 FC.tet10_gmsh_to_ccx(conn), fixed_nodes, forces, float(mat["E_mpa"]), float(mat["nu"]))
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
    try:
        out = FC.analyse(node_xyz, res["DISP"], res["STRESS"])
    except ValueError as e:
        raise SolveError("results", str(e))
    bc = FC.bc_sanity(res["DISP"], fixed_nodes, load_nodes, force_by_load)
    bc["face_matching"] = {"fixed": fixed_m, "loads": load_m}
    bc["applied_force_n"] = total
    eq = FC.equilibrium(res.get("FORC"), force_by_load)
    if eq is not None:
        bc["equilibrium"] = eq
        if eq["error_pct"] > 3.0:
            bc["ok"] = False
            bc["reason"] = (f"reaction forces do not balance the applied loads (error {eq['error_pct']}%): "
                            "a load or a constraint was not applied as intended")
    out.update({"bc_check": bc, "mesh": {"nodes": n_nodes, "elements": int(len(elem_ids)), "element_size_mm": round(lc, 3),
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
        gmsh.initialize()
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
