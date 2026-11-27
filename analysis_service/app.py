"""Lumexa CFD service: steady external-flow CFD (OpenFOAM) behind a small HTTP API.

  GET  /health        OpenFOAM version, tools found, limits, preCICE status
  GET  /selftest      known-answer check (laminar channel vs. analytic Poiseuille pressure drop)
  POST /solve         multipart: stl (millimetres, closed solid) + spec (JSON) -> forces, coefficients, surface loads
  GET  /fsi-status    whether the preCICE two-way coupling stack is installed
  POST /fsi-run       experimental two-way FSI (OpenFOAM <-> CalculiX through preCICE)

Same envelope as analysis_service:  {"ok": true, "result": ..., "timings": ...}  or
{"ok": false, "stage": ..., "reason": ..., "design_related": bool}. Run: uvicorn app:app --port 8002
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import cfd_selftest as ST
import foam_case as FCASE
import foam_post as FPOST
import fsi_precice as FSI

VERSION = "cfd-v1"
API_KEY = os.environ.get("CFD_API_KEY", "")
BUDGET_S = float(os.environ.get("CFD_BUDGET_S", "1500"))              # wall-time cap for one /solve
MAX_TRIS = int(os.environ.get("CFD_MAX_TRIANGLES", "400000"))
MAX_UPLOAD_MB = int(os.environ.get("CFD_MAX_UPLOAD_MB", "60"))
MAX_SURFACE_FACES = int(os.environ.get("CFD_MAX_SURFACE_FACES", "150000"))
MAX_PROCS = max(1, int(os.environ.get("CFD_MAX_PROCS", str(os.cpu_count() or 1))))        # MPI ranks per solve, capped to the machine
DEFAULT_PROCS = max(1, min(int(os.environ.get("CFD_DEFAULT_PROCS", "1")), MAX_PROCS))

app = FastAPI(title="Lumexa CFD service", version=VERSION)
_lock = threading.Lock()


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"ok": False, "stage": "internal", "design_related": False,
                                                  "reason": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})


class CfdError(Exception):
    def __init__(self, stage, msg, design_related=False):
        super().__init__(msg)
        self.stage, self.design_related = stage, design_related


# ------------------------------------------------------------------------------------------------ OpenFOAM plumbing
def _find_bashrc():
    env = os.environ.get("FOAM_BASHRC")
    cands = [env] if env else []
    cands += ["/usr/share/openfoam/etc/bashrc"] + sorted(glob.glob("/usr/lib/openfoam/openfoam*/etc/bashrc"), reverse=True) \
        + sorted(glob.glob("/opt/openfoam*/etc/bashrc"), reverse=True)
    return next((c for c in cands if c and os.path.exists(c)), None)


BASHRC = _find_bashrc()
_ver_cache = {}


def sh(cwd, cmd, log, timeout):
    """Run a command with the OpenFOAM environment, output to cwd/log. Returns (returncode, timed_out)."""
    if not BASHRC:
        raise CfdError("setup", "OpenFOAM not found (set FOAM_BASHRC to its etc/bashrc)")
    p = subprocess.Popen(["bash", "-c", f'source "{BASHRC}" >/dev/null 2>&1; {cmd} > "{log}" 2>&1'], cwd=cwd, start_new_session=True)
    try:
        return p.wait(timeout=max(5.0, timeout)), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        return -9, True


def foam_version():
    if "v" not in _ver_cache:
        out = subprocess.run(["bash", "-c", f'source "{BASHRC}" >/dev/null 2>&1; echo $WM_PROJECT_VERSION'], capture_output=True, text=True).stdout.strip() if BASHRC else ""
        digits = "".join(c for c in out if c.isdigit())
        _ver_cache["v"] = (out, int(digits) if digits else 0)
    return _ver_cache["v"]


def _tail(cwd, log, n=6):
    try:
        ls = [l for l in open(os.path.join(cwd, log), errors="ignore").read().splitlines() if l.strip()]
        return " | ".join(ls[-n:])[:700]
    except OSError:
        return "(no log)"


def _auth(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "bad or missing x-api-key")


def stl_report(tris):
    ext = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
    q = np.round(tris.reshape(-1, 3) / max(ext * 1e-7, 1e-12)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e[:, 0] * (int(inv.max()) + 1) + e[:, 1], return_counts=True)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    return {"triangles": int(len(tris)), "edges": int(len(counts)), "open_edges": int((counts == 1).sum()),
            "non_manifold_edges": int((counts > 2).sum()), "degenerate": int((area < 1e-12 * ext * ext).sum()), "extent": ext}


# ------------------------------------------------------------------------------------------------ the solve
def run_cfd(stl_bytes, spec):
    t0 = time.time()
    tm, warn = {}, []
    spec = dict(spec or {})
    spec["n_procs"] = max(1, min(int(spec.get("n_procs") or DEFAULT_PROCS), MAX_PROCS))
    left = lambda: BUDGET_S - (time.time() - t0)
    try:
        tris = FCASE.read_stl(stl_bytes)
    except ValueError as e:
        raise CfdError("geometry", str(e), design_related=True)
    if len(tris) > MAX_TRIS:
        raise CfdError("geometry", f"{len(tris)} triangles (limit {MAX_TRIS}): export the STL with a coarser tolerance", design_related=False)
    rep = stl_report(tris)
    if rep["open_edges"] or rep["non_manifold_edges"]:
        frac = (rep["open_edges"] + rep["non_manifold_edges"]) / max(rep["edges"], 1)
        msg = f"the STL is not a closed solid ({rep['open_edges']} open and {rep['non_manifold_edges']} non-manifold edges)"
        if frac > 0.002:
            raise CfdError("geometry", msg + ": CFD needs a watertight body", design_related=True)
        warn.append(msg + "; small leaks can make snappyHexMesh fail")
    if rep["degenerate"]:
        warn.append(f"{rep['degenerate']} zero-area triangles in the STL")

    ver_s, ver = foam_version()
    work = tempfile.mkdtemp(prefix="cfd_")
    try:
        try:
            info = FCASE.build_case(work, tris, spec, ver)
        except ValueError as e:
            raise CfdError("setup", str(e))
        if info["blockage"] > 0.10:
            warn.append(f"blockage {info['blockage']*100:.0f}% of the tunnel cross-section: forces are over-predicted")
        tm["setup_s"] = round(time.time() - t0, 2)

        # ---- mesh
        mt = time.time()
        for cmd, log, cap, stage in (("surfaceFeatureExtract", "log.sfe", 120, "features"), ("blockMesh", "log.blockMesh", 120, "mesh")):
            rc, to = sh(work, cmd, log, min(cap, left()))
            if rc != 0:
                raise CfdError(stage, f"{cmd} failed: {_tail(work, log)}", design_related=(stage == "features"))
        rc, to = sh(work, "snappyHexMesh -overwrite", "log.snappy", min(left() * 0.45, 1100))
        if to:
            raise CfdError("mesh", "snappyHexMesh exceeded its time budget: use quality 'coarse' or a simpler part")
        bnd = FPOST.read_boundary(work) if os.path.exists(os.path.join(work, "constant/polyMesh/boundary")) else {}
        if rc != 0 or "part" not in bnd or bnd["part"]["nFaces"] == 0:
            raise CfdError("mesh", f"snappyHexMesh produced no wall on the part: {_tail(work, 'log.snappy')}", design_related=True)
        sh(work, "checkMesh", "log.checkMesh", min(180, max(left(), 5)))
        ck = open(os.path.join(work, "log.checkMesh"), errors="ignore").read()
        import re
        cells = re.search(r"cells:\s+(\d+)", ck)
        mesh = {"cells": int(cells.group(1)) if cells else None, "wall_faces": bnd["part"]["nFaces"],
                "ok": "Mesh OK" in ck, "failed_checks": (int(re.search(r"Failed (\d+) mesh checks", ck).group(1)) if re.search(r"Failed (\d+) mesh checks", ck) else 0),
                "max_non_orthogonality": float(m.group(1)) if (m := re.search(r"non-orthogonality Max:\s*([\d.]+)", ck)) else None,
                "max_skewness": float(m.group(1)) if (m := re.search(r"Max skewness = ([\d.]+)", ck)) else None,
                "quality": info["quality"], "layers": info["layers"]}
        if not mesh["ok"]:
            warn.append(f"checkMesh reported {mesh['failed_checks']} failed checks: forces may be less accurate")
        tm["mesh_s"] = round(time.time() - mt, 2)

        # ---- solve
        st = time.time()
        n = info["n_procs"]
        timed_out = False
        if n > 1:
            sh(work, "decomposePar -force", "log.decomposePar", 300)
            flags = os.environ.get("CFD_MPI_FLAGS", "--allow-run-as-root --oversubscribe" if os.geteuid() == 0 else "--oversubscribe")
            rc, timed_out = sh(work, f"mpirun {flags} -np {n} simpleFoam -parallel", "log.simpleFoam", left() - 90)
            sh(work, "reconstructPar -latestTime", "log.reconstructPar", 300)
        else:
            rc, timed_out = sh(work, "simpleFoam", "log.simpleFoam", left() - 30)
        log_txt = open(os.path.join(work, "log.simpleFoam"), errors="ignore").read()
        conv = FPOST.parse_log(log_txt)
        tm["solve_s"] = round(time.time() - st, 2)
        if conv["diverged"] or (rc != 0 and not timed_out):
            raise CfdError("solve", f"OpenFOAM solver {'diverged' if conv['diverged'] else 'failed'} (exit {rc}): {_tail(work, 'log.simpleFoam')}")
        if timed_out:
            warn.append(f"time budget reached after {conv['iterations']} iterations: results are from the last written time and may not be converged")

        # ---- results
        rt = time.time()
        tname = FPOST.latest_time(work)
        if tname in (None, "0"):
            raise CfdError("results", "the solver wrote no result time (stopped before the first write)")
        summary, surf = FPOST.wall_loads(work, info, "part", tname)
        times = sorted((float(d), d) for d in os.listdir(work) if d.replace(".", "", 1).isdigit() and float(d) > 0)
        drag_change = None
        if len(times) >= 2:
            try:
                prev, _ = FPOST.wall_loads(work, info, "part", times[-2][1])
                if abs(summary["drag_n"]) > 1e-12:
                    drag_change = abs(summary["drag_n"] - prev["drag_n"]) / abs(summary["drag_n"]) * 100.0
            except Exception:
                pass
        res = conv["final_residuals"]
        fail_to_converge = not conv["converged"]
        if fail_to_converge and (drag_change is None or drag_change > 2.0):
            warn.append("not converged: drag still changing" + (f" ({drag_change:.1f}% between the last two writes)" if drag_change is not None else "")
                        + "; raise max_iterations or lower 'relax'")
        yp = summary.get("y_plus")
        if info["turbulence"] != "laminar" and yp and not (yp["fraction_30_300"] > 0.4):
            warn.append(f"wall-function y+ is mostly outside 30-300 (mean {yp['mean']:.0f}): viscous forces and separation are only indicative; try another quality")
        if abs(summary["drag_pressure_n"]) > 0 and abs(summary["drag_viscous_n"]) > 0.6 * abs(summary["drag_n"]) and info["turbulence"] != "laminar":
            warn.append("friction dominates the drag: a wall-function mesh without prism layers is not reliable here (set layers: true)")
        summary.update({"converged": conv["converged"], "iterations": conv["iterations"], "final_residuals": res,
                        "drag_change_last_write_pct": None if drag_change is None else round(drag_change, 3), "timed_out": bool(timed_out)})
        if len(surf) > MAX_SURFACE_FACES:
            warn.append(f"surface map has {len(surf)} faces; not returned (limit {MAX_SURFACE_FACES})")
            surface = None
        else:
            surface = FPOST.encode_surface(surf) if spec.get("return_surface", True) else None
        tm["post_s"] = round(time.time() - rt, 2)
        tm["total_s"] = round(time.time() - t0, 2)
        info_out = {k: v for k, v in info.items() if k not in ("tunnel_min_m", "tunnel_max_m")}
        return {"summary": summary, "flow": info_out, "mesh": mesh, "geometry": {k: rep[k] for k in ("triangles", "open_edges", "non_manifold_edges")},
                "warnings": warn, "surface": surface, "versions": {"openfoam": ver_s, "service": VERSION}}, tm
    finally:
        if not os.environ.get("CFD_KEEP_CASES"):
            shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    ver_s, ver = foam_version()
    tools = {t: bool(shutil.which(t)) for t in ("blockMesh", "snappyHexMesh", "surfaceFeatureExtract", "simpleFoam", "checkMesh", "decomposePar", "mpirun")}
    ok = bool(BASHRC) and all(tools[t] for t in ("blockMesh", "snappyHexMesh", "surfaceFeatureExtract", "simpleFoam", "checkMesh"))
    return {"ok": ok, "service": VERSION, "openfoam": ver_s or None, "bashrc": BASHRC, "tools": tools, "budget_s": BUDGET_S,
            "max_triangles": MAX_TRIS, "max_procs": MAX_PROCS, "default_procs": DEFAULT_PROCS, "auth_required": bool(API_KEY), "cpu_count": os.cpu_count(), "quality_levels": list(FCASE.QUALITY),
            "fluids": list(FCASE.FLUIDS), "precice": FSI.status()}


@app.post("/solve")
def solve(stl: UploadFile = File(...), spec: str = Form(...), x_api_key: str = Header(default="")):
    _auth(x_api_key)
    try:
        spec_d = json.loads(spec)
    except ValueError:
        raise HTTPException(400, "spec must be JSON")
    data = stl.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"STL larger than {MAX_UPLOAD_MB} MB")
    if not _lock.acquire(timeout=BUDGET_S):
        return {"ok": False, "stage": "busy", "reason": "another CFD run is in progress", "design_related": False}
    try:
        out, tm = run_cfd(data, spec_d)
        return {"ok": True, "result": out, "timings": tm}
    except CfdError as e:
        return {"ok": False, "stage": e.stage, "reason": str(e), "design_related": e.design_related}
    finally:
        _lock.release()


@app.get("/selftest")
def selftest(x_api_key: str = Header(default="")):
    _auth(x_api_key)
    if not BASHRC:
        return {"ok": False, "reason": "OpenFOAM not found"}
    with _lock:
        return ST.channel_selftest(sh, foam_version()[1])


@app.get("/fsi-status")
def fsi_status():
    return FSI.status()


@app.post("/fsi-run")
def fsi_run(stl: UploadFile = File(...), spec: str = Form(...), x_api_key: str = Header(default="")):
    _auth(x_api_key)
    st = FSI.status()
    if not st["ready"]:
        return JSONResponse(status_code=501, content={"ok": False, "stage": "setup", "design_related": False,
                                                      "reason": "preCICE two-way coupling is not installed on this server", "missing": st["missing"],
                                                      "how": "run build_precice_stack.sh (see README), then restart the service"})
    return JSONResponse(status_code=501, content={"ok": False, "stage": "setup", "design_related": False,
                                                  "reason": "FSI runner is scaffolding only in this release; use /solve + one-way coupling"})
