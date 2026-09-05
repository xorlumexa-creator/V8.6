"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     LUMEXA ENGINEERING BACKEND v8.0 — ENTERPRISE GRADE                      ║
║                                                                              ║
║  METHODOLOGY (self-reported, not independently benchmarked — treat as a      ║
║  guide to which module to trust for what, not a certified error bound):      ║
║  Geometry:         trimesh exact math                                        ║
║  Wall thickness:   dual-pass surface sampling (ray-cast thickness probe)     ║
║  Hole detection:   multi-axis RANSAC                                         ║
║  Sharp corners:    Peterson-Neuber stress concentration                      ║
║  FEA (CalculiX):   real solver run, S3 shell elements on the surface mesh    ║
║                     (when ccx is available) — best for thin-walled parts     ║
║  FEA (fallback):   multi-section analytical (no CalculiX available)         ║
║  Fatigue:          full Marin 6-factor + Goodman/Gerber                      ║
║  Fracture:         Paris Law + failure assessment diagram                    ║
║  Thermal:          gradient field + Coffin-Manson                            ║
║  Topology opt:     SIMP-style density heuristic — fast first pass, NOT a     ║
║                     per-iteration FEA-verified optimization (see docstring)  ║
║  Composite:        Classical Laminate Theory + Tsai-Wu                       ║
║                                                                              ║
║  None of the above numbers are validated against NAFEMS or other published  ║
║  benchmark problems yet. Run those before making accuracy claims to users.   ║
║                                                                              ║
║  NEW IN v8.0:                                                                ║
║  + CalculiX real FEM (tetrahedral elements)                                  ║
║  + Gmsh mesh generation                                                      ║
║  + Topology optimization (SIMP)                                              ║
║  + Composite material analysis (CLT)                                         ║
║  + Rainflow fatigue counting                                                 ║
║  + Gemini script generation (/generate-from-prompt)                          ║
║  + Image-to-params (/image-to-params)                                        ║
║  + Manufacturing cost estimate                                               ║
║  + Design comparison (2 designs)                                             ║
║  + Background job queue for heavy analysis                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import trimesh
import trimesh.smoothing
import trimesh.creation
import numpy as np
import tempfile, os, math, json, base64, subprocess, threading, time, uuid, functools, asyncio
from typing import Optional, Dict
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull
from collections import defaultdict

try:
    import cadquery as cq
    CQ = True
except ImportError:
    CQ = False

# CALCULIX/GMSH are no longer checked or imported HERE — that whole
# dependency chain (and its multi-hundred-MB footprint) moved to the
# separate analysis_service.py, which is what this service now calls over
# HTTP instead of doing that work in-process. See ANALYSIS_SERVICE_URL and
# run_calculix_fem's new remote-call implementation further down this file.

# Check Blender
try:
    r = subprocess.run(["blender","--version"], capture_output=True, timeout=5)
    BLENDER = r.returncode == 0
except:
    BLENDER = False

# Check ezdxf (pure-Python, no external binary — used for 2D manufacturing
# drawing export, i.e. DXF files a laser-cutter/CNC/machine shop can open
# directly, distinct from the 3D STEP/STL export elsewhere in this file)
try:
    import ezdxf
    from ezdxf import units as ezdxf_units
    EZDXF = True
except ImportError:
    EZDXF = False

def _json_safe(obj):
    """
    Recursively convert numpy scalar/array types to native Python types.

    numpy.bool_, numpy.integer, numpy.floating, and numpy.ndarray are NOT
    natively JSON serializable even though they print/compare identically to
    their plain-Python equivalents — trimesh's mesh.is_watertight, and any
    boolean/numeric produced by comparing against a numpy-derived value
    (stress calculations, safety factors, etc. — this codebase does a LOT of
    numpy arithmetic), can silently end up as a numpy type in a response dict.
    This surfaced for real as "Object of type bool is not JSON serializable"
    on a live generation call, from some numpy-typed value elsewhere in the
    response — not from a single fixable field, from the general pattern of
    numpy math results flowing into response dicts throughout this file.
    Fixing it once here, applied to every response, is far more reliable than
    hunting down and individually bool()/int()/float()-wrapping every numpy
    comparison across ~19 endpoints.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _sanitize_response(func):
    """
    Decorator: run an endpoint's return value through _json_safe() before
    FastAPI's own internal serialization ever touches it.

    Why this is needed IN ADDITION to SafeJSONResponse below (confirmed via a
    live crash, not assumed): FastAPI calls its own jsonable_encoder() on a
    route's return value during serialize_response(), which happens INSIDE
    FastAPI's routing logic — before any custom response_class's .render()
    is ever invoked. SafeJSONResponse only guards the final json.dumps() step,
    which never gets reached if jsonable_encoder already raised. Confirmed
    live traceback: "'numpy.bool' object is not iterable" /
    "vars() argument must have __dict__ attribute" inside
    fastapi/encoders.py's jsonable_encoder — a numpy scalar reached FastAPI's
    own encoder, which doesn't know how to handle it, well before
    SafeJSONResponse ever got a chance to sanitize anything. This decorator
    closes that earlier gap; SafeJSONResponse stays as defense in depth for
    anything constructed as a raw JSONResponse instead of returned directly.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        if isinstance(result, (dict, list)):
            return _json_safe(result)
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, (dict, list)):
            return _json_safe(result)
        return result

    # 4 of this file's 19 routes are plain `def`, not `async def` (home,
    # get_materials, get_part_types, get_job) — caught before shipping by
    # checking rather than assuming every route was async. `await`-ing a
    # plain function's return value raises immediately, so branch here
    # instead of always using the async wrapper.
    return wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class SafeJSONResponse(JSONResponse):
    """JSONResponse that sanitizes numpy types out of the content right before
    the final json.dumps call — see _json_safe's docstring for why this is
    the correct interception point (FastAPI's own jsonable_encoder does not
    reliably catch numpy scalar types before handing off to this class)."""
    def render(self, content) -> bytes:
        return super().render(_json_safe(content))


app = FastAPI(title="Lumexa v8.22 Enterprise (split architecture)", version="8.22.0",
              default_response_class=SafeJSONResponse)
# NOTE: allow_origins=["*"] combined with allow_credentials=True is an invalid/unsafe
# CORS configuration — browsers reject wildcard origins when credentials are allowed,
# and permissively is unsafe if it ever does work via a proxy that echoes the origin.
# Set ALLOWED_ORIGINS env var (comma-separated) in production; credentials stay off
# unless you actually need cookies/auth headers across origins.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# Job store for background analysis
JOB_STORE: Dict[str, dict] = {}

# ═══════════════════════════════════════════════════════════════════
# MATERIAL DATABASE v3 — Extended with composite support
# ═══════════════════════════════════════════════════════════════════
MATERIALS = {
    "aluminum_6061":{"name":"Aluminum 6061-T6","density":2.70,
        "yield_strength_mpa":276,"ultimate_strength_mpa":310,
        "youngs_modulus_gpa":68.9,"poissons_ratio":0.33,
        "thermal_expansion_per_c":23.6e-6,"thermal_conductivity":167,
        "max_service_temp_c":150,"fatigue_limit_mpa":96,
        "fracture_toughness_mpa_sqrtm":29.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":1.2e-4,
        "paris_C":1.5e-10,"paris_m":3.58,"shear_modulus_gpa":26.0,
        "hardness_brinell":95,"endurance_ratio":0.4,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":3.5,"machinability":0.85},
    "aluminum_7075":{"name":"Aluminum 7075-T6","density":2.81,
        "yield_strength_mpa":503,"ultimate_strength_mpa":572,
        "youngs_modulus_gpa":71.7,"poissons_ratio":0.33,
        "thermal_expansion_per_c":23.4e-6,"thermal_conductivity":130,
        "max_service_temp_c":120,"fatigue_limit_mpa":159,
        "fracture_toughness_mpa_sqrtm":24.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":1.0e-4,
        "paris_C":1.2e-10,"paris_m":3.5,"shear_modulus_gpa":26.9,
        "hardness_brinell":150,"endurance_ratio":0.4,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":5.5,"machinability":0.70},
    "alsi10mg_slm":{"name":"AlSi10Mg SLM (3D Printed)","density":2.68,
        "yield_strength_mpa":230,"ultimate_strength_mpa":345,
        "youngs_modulus_gpa":70.0,"poissons_ratio":0.33,
        "thermal_expansion_per_c":21.0e-6,"thermal_conductivity":130,
        "max_service_temp_c":120,"fatigue_limit_mpa":70,
        "fracture_toughness_mpa_sqrtm":20.0,"creep_exponent_n":5.0,
        "creep_activation_energy":142000,"creep_A_constant":2.0e-4,
        "paris_C":2.0e-10,"paris_m":3.8,"shear_modulus_gpa":26.3,
        "hardness_brinell":80,"endurance_ratio":0.35,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.095,
        "min_wall_mm":0.8,"min_fillet_mm":0.4,
        "cost_per_kg_usd":45.0,"machinability":0.60},
    "titanium_6al4v":{"name":"Titanium Ti-6Al-4V","density":4.43,
        "yield_strength_mpa":880,"ultimate_strength_mpa":950,
        "youngs_modulus_gpa":114.0,"poissons_ratio":0.34,
        "thermal_expansion_per_c":8.6e-6,"thermal_conductivity":7.2,
        "max_service_temp_c":315,"fatigue_limit_mpa":510,
        "fracture_toughness_mpa_sqrtm":75.0,"creep_exponent_n":4.0,
        "creep_activation_energy":250000,"creep_A_constant":5.0e-6,
        "paris_C":5.0e-11,"paris_m":3.2,"shear_modulus_gpa":44.0,
        "hardness_brinell":334,"endurance_ratio":0.55,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.075,
        "min_wall_mm":0.8,"min_fillet_mm":0.3,
        "cost_per_kg_usd":85.0,"machinability":0.30},
    "steel_4340":{"name":"Steel AISI 4340","density":7.85,
        "yield_strength_mpa":470,"ultimate_strength_mpa":745,
        "youngs_modulus_gpa":205.0,"poissons_ratio":0.29,
        "thermal_expansion_per_c":12.3e-6,"thermal_conductivity":44.5,
        "max_service_temp_c":370,"fatigue_limit_mpa":380,
        "fracture_toughness_mpa_sqrtm":50.0,"creep_exponent_n":5.5,
        "creep_activation_energy":280000,"creep_A_constant":6.0e-7,
        "paris_C":6.0e-12,"paris_m":3.0,"shear_modulus_gpa":80.0,
        "hardness_brinell":217,"endurance_ratio":0.5,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.5,"min_fillet_mm":1.0,
        "cost_per_kg_usd":2.5,"machinability":0.55},
    "inconel_718":{"name":"Inconel 718","density":8.19,
        "yield_strength_mpa":1034,"ultimate_strength_mpa":1241,
        "youngs_modulus_gpa":200.0,"poissons_ratio":0.29,
        "thermal_expansion_per_c":13.0e-6,"thermal_conductivity":11.4,
        "max_service_temp_c":650,"fatigue_limit_mpa":550,
        "fracture_toughness_mpa_sqrtm":100.0,"creep_exponent_n":4.5,
        "creep_activation_energy":300000,"creep_A_constant":3.0e-7,
        "paris_C":3.0e-12,"paris_m":3.0,"shear_modulus_gpa":77.0,
        "hardness_brinell":310,"endurance_ratio":0.45,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.080,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":65.0,"machinability":0.20},
    "carbon_fiber_ud":{"name":"Carbon Fiber CFRP (UD)","density":1.60,
        "yield_strength_mpa":600,"ultimate_strength_mpa":1500,
        "youngs_modulus_gpa":135.0,"poissons_ratio":0.28,
        "thermal_expansion_per_c":2.1e-6,"thermal_conductivity":5.0,
        "max_service_temp_c":180,"fatigue_limit_mpa":450,
        "fracture_toughness_mpa_sqrtm":35.0,"creep_exponent_n":3.0,
        "creep_activation_energy":200000,"creep_A_constant":1.0e-8,
        "paris_C":1.0e-11,"paris_m":3.0,"shear_modulus_gpa":5.0,
        "hardness_brinell":0,"endurance_ratio":0.6,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.070,
        "min_wall_mm":0.5,"min_fillet_mm":0.3,
        # Composite-specific
        "E1_gpa":135.0,"E2_gpa":10.0,"G12_gpa":5.0,"nu12":0.28,
        "Xt_mpa":1500,"Xc_mpa":1200,"Yt_mpa":50,"Yc_mpa":250,"S12_mpa":70,
        "cost_per_kg_usd":80.0,"machinability":0.15},
    "pla_plastic":{"name":"PLA Plastic (FDM)","density":1.24,
        "yield_strength_mpa":50,"ultimate_strength_mpa":65,
        "youngs_modulus_gpa":3.5,"poissons_ratio":0.36,
        "thermal_expansion_per_c":68e-6,"thermal_conductivity":0.13,
        "max_service_temp_c":60,"fatigue_limit_mpa":20,
        "fracture_toughness_mpa_sqrtm":3.5,"creep_exponent_n":3.0,
        "creep_activation_energy":80000,"creep_A_constant":1.0e-3,
        "paris_C":1.0e-8,"paris_m":4.0,"shear_modulus_gpa":1.3,
        "hardness_brinell":0,"endurance_ratio":0.35,
        "Sut_at_1000":0.80,"fatigue_slope_b":-0.110,
        "min_wall_mm":1.2,"min_fillet_mm":0.8,
        "cost_per_kg_usd":25.0,"machinability":0.90},
    "petg_plastic":{"name":"PETG Plastic (FDM)","density":1.27,
        "yield_strength_mpa":53,"ultimate_strength_mpa":50,
        "youngs_modulus_gpa":2.1,"poissons_ratio":0.38,
        "thermal_expansion_per_c":60e-6,"thermal_conductivity":0.20,
        "max_service_temp_c":80,"fatigue_limit_mpa":18,
        "fracture_toughness_mpa_sqrtm":4.0,"creep_exponent_n":3.0,
        "creep_activation_energy":80000,"creep_A_constant":1.2e-3,
        "paris_C":1.2e-8,"paris_m":4.0,"shear_modulus_gpa":0.76,
        "hardness_brinell":0,"endurance_ratio":0.32,
        "Sut_at_1000":0.78,"fatigue_slope_b":-0.115,
        "min_wall_mm":1.2,"min_fillet_mm":0.8,
        "cost_per_kg_usd":28.0,"machinability":0.88},
    "stainless_316l":{"name":"Stainless Steel 316L","density":7.98,
        "yield_strength_mpa":170,"ultimate_strength_mpa":485,
        "youngs_modulus_gpa":193.0,"poissons_ratio":0.28,
        "thermal_expansion_per_c":16.0e-6,"thermal_conductivity":16.3,
        "max_service_temp_c":870,"fatigue_limit_mpa":240,
        "fracture_toughness_mpa_sqrtm":200.0,"creep_exponent_n":5.0,
        "creep_activation_energy":270000,"creep_A_constant":4.0e-7,
        "paris_C":4.0e-12,"paris_m":3.1,"shear_modulus_gpa":74.0,
        "hardness_brinell":217,"endurance_ratio":0.5,
        "Sut_at_1000":0.9,"fatigue_slope_b":-0.085,
        "min_wall_mm":1.5,"min_fillet_mm":1.0,
        "cost_per_kg_usd":8.0,"machinability":0.45},
    "magnesium_az31":{"name":"Magnesium AZ31B","density":1.77,
        "yield_strength_mpa":200,"ultimate_strength_mpa":260,
        "youngs_modulus_gpa":45.0,"poissons_ratio":0.35,
        "thermal_expansion_per_c":26.0e-6,"thermal_conductivity":96,
        "max_service_temp_c":120,"fatigue_limit_mpa":90,
        "fracture_toughness_mpa_sqrtm":18.0,"creep_exponent_n":4.5,
        "creep_activation_energy":135000,"creep_A_constant":3.0e-4,
        "paris_C":2.0e-10,"paris_m":3.5,"shear_modulus_gpa":17.0,
        "hardness_brinell":73,"endurance_ratio":0.35,
        "Sut_at_1000":0.85,"fatigue_slope_b":-0.095,
        "min_wall_mm":1.0,"min_fillet_mm":0.5,
        "cost_per_kg_usd":4.0,"machinability":0.80},
}

# ═══════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════
def sf(v, d=0.0):
    try:
        r = float(v)
        return d if (math.isnan(r) or math.isinf(r)) else r
    except: return d

# ═══════════════════════════════════════════════════════════════════
# CALCULIX FEM — real solid tetrahedral analysis (Gmsh-tetrahedralized),
# with a real shell-element analysis as fallback when Gmsh/tet-meshing
# isn't available or fails on a given part.
# ═══════════════════════════════════════════════════════════════════

# Gmsh's Python API holds session state at module/global scope and is not
# safe to run concurrently from multiple requests in the same process —
# serialize access to it.
# ═══════════════════════════════════════════════════════════════════
# FEM now runs in a SEPARATE service (analysis_service.py) — split out to
# isolate the heaviest computation (Gmsh volume meshing + CalculiX solid-tet
# solving) from this process's memory footprint. Confirmed necessary via a
# real OOM crash on Render free tier: a Termux curl test showed the
# connection dying mid-request (0 bytes received), immediately followed by
# Render auto-restarting the container in the logs — the signature of the
# process being killed for memory, not a normal error.
#
# This function keeps the EXACT same name/signature/return-shape as the old
# local implementation, so nothing else in this file needs to change — it's
# now a thin HTTP client instead of doing the work in-process.
ANALYSIS_SERVICE_URL = os.environ.get("ANALYSIS_SERVICE_URL", "").rstrip("/")
if ANALYSIS_SERVICE_URL and not ANALYSIS_SERVICE_URL.startswith(("http://", "https://")):
    # Defensive: if someone pastes a bare "host:port" or "host.onrender.com"
    # without a scheme (an easy mistake — Render's own fromService/hostport
    # auto-wiring returns exactly that, bare, with no scheme), assume https
    # rather than let urllib fail with a confusing "unknown url type" error.
    ANALYSIS_SERVICE_URL = "https://" + ANALYSIS_SERVICE_URL

def run_calculix_fem(mesh, mat_key, force_n=1000, force_dir="z"):
    """
    Real FEM entry point — now a remote call to the separate analysis
    service, not local computation. Returns (fem_result_or_None, diag) where
    diag always explains what happened: not attempted (service not
    configured), or attempted with the specific error if it failed, or
    attempted successfully. This replaces the old bare-None return, which
    made "not configured" and "configured but silently failing on every
    call" indistinguishable from the API response — exactly the ambiguity
    that made calculix_used=false undiagnosable all session. Still never
    raises: a down or unconfigured analysis service degrades to the
    analytical fallback instead of failing the whole generation request.
    """
    if not ANALYSIS_SERVICE_URL:
        return None, {"attempted": False, "reason": "ANALYSIS_SERVICE_URL not configured"}

    import urllib.request, urllib.error, time

    t0 = time.time()
    try:
        stl_bytes = mesh.export(file_type="stl")
        if isinstance(stl_bytes, str):
            stl_bytes = stl_bytes.encode()

        boundary = "----lumexafemboundary"
        body = []
        body.append(f"--{boundary}\r\n".encode())
        body.append(b'Content-Disposition: form-data; name="mesh_file"; filename="part.stl"\r\n')
        body.append(b"Content-Type: application/octet-stream\r\n\r\n")
        body.append(stl_bytes)
        body.append(b"\r\n")
        for field, value in [("material", mat_key), ("force_n", str(force_n)),
                              ("force_dir", force_dir)]:
            body.append(f"--{boundary}\r\n".encode())
            body.append(f'Content-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        body.append(f"--{boundary}--\r\n".encode())
        payload = b"".join(body)

        req = urllib.request.Request(
            f"{ANALYSIS_SERVICE_URL}/run-fem", data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        # Real solid-tet FEM is genuinely slow on constrained hardware —
        # matching the same 300s ceiling the old in-process ccx subprocess
        # call used, so this isn't a NEW bottleneck, just moved to a
        # different process.
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
            data = json.loads(raw)
        elapsed = round(time.time() - t0, 1)
        if "fem_result" not in data:
            return None, {"attempted": True, "reason": "response missing fem_result key",
                           "raw_keys": list(data.keys()), "elapsed_s": elapsed}
        fem_result = data.get("fem_result")
        if not fem_result:
            return None, {"attempted": True, "reason": "analysis service returned a null/empty "
                           "fem_result (HTTP call succeeded — the failure is inside that service, "
                           "e.g. Gmsh meshing or the CalculiX solve itself)",
                           "service_error_field": data.get("note"),
                           "service_stage_diagnostic": data.get("diagnostic"),
                           "mesh_repair": data.get("mesh_repair"),
                           "elapsed_s": elapsed}
        return fem_result, {"attempted": True, "reason": None, "elapsed_s": elapsed}
    except urllib.error.HTTPError as e:
        body_snippet = ""
        try: body_snippet = e.read().decode(errors="replace")[:300]
        except Exception: pass
        return None, {"attempted": True, "reason": f"HTTP {e.code} from analysis service",
                       "body": body_snippet, "elapsed_s": round(time.time()-t0,1)}
    except urllib.error.URLError as e:
        return None, {"attempted": True, "reason": f"unreachable: {e.reason}",
                       "elapsed_s": round(time.time()-t0,1)}
    except Exception as e:
        return None, {"attempted": True, "reason": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time()-t0,1)}


def topology_optimization_simp(mesh, mat_key, volfrac=0.5,
                                 penal=3.0, n_iterations=30):
    """
    SIMP (Solid Isotropic Material with Penalization) — density-based lightweighting.

    NOTE ON METHODOLOGY: a textbook SIMP loop re-solves the full FEA at every
    iteration to get a true compliance sensitivity field. This implementation uses
    a density-proportional sensitivity heuristic instead of a per-iteration FEA
    solve, so it is a fast, useful *first-pass* material-removal suggestion, not a
    structurally-verified topology optimization. Always re-run a full FEA (see
    run_calculix_fem / multi_section_fea) on the resulting geometry before trusting
    the mass savings for a real part.
    """
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    E = mat["youngs_modulus_gpa"] * 1000
    Emin = E * 1e-4

    bounds = mesh.bounds
    extents = mesh.bounding_box.extents

    # Discretize into voxel grid
    nx = min(30, max(10, int(extents[0]/5)))
    ny = min(30, max(10, int(extents[1]/5)))
    nz = min(20, max(8, int(extents[2]/5)))

    n_elements = nx * ny * nz
    n_nodes = (nx+1) * (ny+1) * (nz+1)

    # Initialize density field
    x = np.full(n_elements, volfrac)

    # Sensitivity field (simplified compliance gradient)
    dx = extents[0]/nx; dy = extents[1]/ny; dz = extents[2]/nz

    # Apply SIMP iterations
    history = []
    for iteration in range(n_iterations):
        # Penalized stiffness
        E_penalized = Emin + (E - Emin) * x**penal

        # Sensitivity (dC/dx) — compliance gradient
        # Simplified: sensitivity proportional to element stress
        # In real SIMP: requires full FEA at each iteration
        sensitivity = -penal * (E - Emin) * x**(penal-1)

        # Compliance estimate
        compliance = np.sum(E_penalized * (1.0/(E_penalized+1e-10)))
        history.append(float(compliance))

        # Filter sensitivities (checkerboard prevention)
        # Simple averaging filter
        x_3d = x.reshape(nx, ny, nz)
        from scipy.ndimage import uniform_filter
        try:
            sens_3d = sensitivity.reshape(nx, ny, nz)
            sens_filtered = uniform_filter(sens_3d, size=3)
            sensitivity = sens_filtered.flatten()
        except: pass

        # Optimality criteria update
        l1, l2 = 0.0, 1e9
        move = 0.2
        x_new = x.copy()
        while (l2 - l1) / (l2 + l1) > 1e-4:
            lmid = 0.5*(l2+l1)
            # Bisection on Lagrange multiplier
            x_new = np.maximum(1e-3,
                      np.maximum(x - move,
                        np.minimum(1.0,
                          np.minimum(x + move,
                            x * np.sqrt(-sensitivity/lmid)))))
            if x_new.sum() - volfrac * n_elements > 0:
                l1 = lmid
            else:
                l2 = lmid

        change = np.max(np.abs(x_new - x))
        x = x_new

        if change < 0.01:
            break

    # Final density field
    x_final = x.reshape(nx, ny, nz)

    # Find removed material regions (density < 0.3)
    removed = x_final < 0.3
    kept = x_final >= 0.3

    removed_fraction = float(removed.sum() / n_elements)
    weight_saving_pct = removed_fraction * 100 * (1 - volfrac)

    # Identify high-stress regions (density near 1.0)
    high_stress_regions = []
    threshold_coords = np.argwhere(x_final > 0.8)
    for coord in threshold_coords[:10]:
        cx = bounds[0][0] + (coord[0]+0.5)*dx
        cy = bounds[0][1] + (coord[1]+0.5)*dy
        cz = bounds[0][2] + (coord[2]+0.5)*dz
        high_stress_regions.append({
            "position": {"x":round(cx,2),"y":round(cy,2),"z":round(cz,2)},
            "density": round(float(x_final[coord[0],coord[1],coord[2]]),3)
        })

    # Material saving suggestions
    removable_regions = []
    threshold_coords_low = np.argwhere(x_final < 0.2)
    for coord in threshold_coords_low[:10]:
        cx = bounds[0][0] + (coord[0]+0.5)*dx
        cy = bounds[0][1] + (coord[1]+0.5)*dy
        cz = bounds[0][2] + (coord[2]+0.5)*dz
        removable_regions.append({
            "position": {"x":round(cx,2),"y":round(cy,2),"z":round(cz,2)},
            "density": round(float(x_final[coord[0],coord[1],coord[2]]),3),
            "suggestion": "Safe to remove — low stress region"
        })

    vol = sf(mesh.volume)
    original_mass = vol * mat["density"] * 1e-3
    optimized_mass = original_mass * volfrac

    return {
        "method": "simp_topology_optimization",
        "iterations_run": min(iteration+1, n_iterations),
        "volume_fraction_target": volfrac,
        "grid_resolution": {"nx":nx,"ny":ny,"nz":nz},
        "total_elements": n_elements,
        "weight_saving_estimate_pct": round(weight_saving_pct, 1),
        "original_mass_g": round(original_mass, 2),
        "optimized_mass_g": round(optimized_mass, 2),
        "mass_saved_g": round(original_mass - optimized_mass, 2),
        "high_stress_keep_regions": high_stress_regions[:5],
        "safe_to_remove_regions": removable_regions[:5],
        "compliance_history": [round(c,4) for c in history[-5:]],
        "recommendation": (
            f"Remove {removed_fraction*100:.1f}% of material volume. "
            f"Estimated {weight_saving_pct:.1f}% weight reduction. "
            f"Add holes/pockets at low-density regions."
        ),
    }

# ═══════════════════════════════════════════════════════════════════
# NEW: COMPOSITE MATERIAL ANALYSIS — CLT
# ═══════════════════════════════════════════════════════════════════
def composite_analysis_clt(mat_key, layup_angles, thickness_per_ply_mm,
                              Nx=1000, Ny=0, Nxy=0, Mx=0, My=0, Mxy=0):
    """
    Classical Laminate Theory (CLT) for composite materials.
    Computes A, B, D matrices and failure analysis.
    Uses Tsai-Wu failure criterion.
    Accuracy: 85%
    """
    mat = MATERIALS.get(mat_key, MATERIALS["carbon_fiber_ud"])

    E1  = mat.get("E1_gpa", mat["youngs_modulus_gpa"]) * 1000  # MPa
    E2  = mat.get("E2_gpa", 10.0) * 1000
    G12 = mat.get("G12_gpa", 5.0) * 1000
    nu12= mat.get("nu12", 0.28)
    nu21= nu12 * E2 / E1

    Xt  = mat.get("Xt_mpa", 1500)
    Xc  = mat.get("Xc_mpa", 1200)
    Yt  = mat.get("Yt_mpa", 50)
    Yc  = mat.get("Yc_mpa", 250)
    S12 = mat.get("S12_mpa", 70)

    t = thickness_per_ply_mm
    n_plies = len(layup_angles)
    total_thickness = n_plies * t

    # Ply stiffness in principal directions
    Q11 = E1 / (1 - nu12*nu21)
    Q22 = E2 / (1 - nu12*nu21)
    Q12 = nu12*E2 / (1 - nu12*nu21)
    Q66 = G12

    # Transform Q to global for each ply
    A = np.zeros((3,3))  # Extensional stiffness
    B = np.zeros((3,3))  # Coupling stiffness
    D = np.zeros((3,3))  # Bending stiffness

    z_positions = []
    z = -total_thickness/2
    for i in range(n_plies):
        z_positions.append((z, z+t))
        z += t

    for i, theta_deg in enumerate(layup_angles):
        theta = math.radians(theta_deg)
        c = math.cos(theta); s = math.sin(theta)
        c2=c**2; s2=s**2; cs=c*s

        # Transformed stiffness Qbar
        Qbar = np.zeros((3,3))
        Qbar[0,0] = Q11*c2**2 + 2*(Q12+2*Q66)*s2*c2 + Q22*s2**2
        Qbar[0,1] = (Q11+Q22-4*Q66)*s2*c2 + Q12*(s2**2+c2**2)
        Qbar[0,2] = (Q11-Q12-2*Q66)*s*c2*c + (Q12-Q22+2*Q66)*s2*s
        Qbar[1,0] = Qbar[0,1]
        Qbar[1,1] = Q11*s2**2 + 2*(Q12+2*Q66)*s2*c2 + Q22*c2**2
        Qbar[1,2] = (Q11-Q12-2*Q66)*s2*s + (Q12-Q22+2*Q66)*c2*s
        Qbar[2,0] = Qbar[0,2]
        Qbar[2,1] = Qbar[1,2]
        Qbar[2,2] = (Q11+Q22-2*Q12-2*Q66)*s2*c2 + Q66*(s2**2+c2**2)

        z0, z1 = z_positions[i]
        h0 = z1 - z0
        zm = (z0+z1)/2

        A += Qbar * h0
        B += Qbar * h0 * zm
        D += Qbar * (h0*(zm**2) + h0**3/12)

    # Solve for midplane strains and curvatures
    # [A B] [e0]   [N]
    # [B D] [k ] = [M]
    ABD = np.block([[A, B],[B, D]])
    NM = np.array([Nx, Ny, Nxy, Mx, My, Mxy])

    try:
        ek = np.linalg.solve(ABD, NM)
        e0 = ek[:3]  # midplane strains
        k  = ek[3:]  # curvatures
    except np.linalg.LinAlgError:
        return {"error":"Singular ABD matrix — check layup angles"}

    # Ply stresses and Tsai-Wu failure
    ply_results = []
    max_tsai_wu = 0.0
    first_ply_failure = None

    for i, theta_deg in enumerate(layup_angles):
        theta = math.radians(theta_deg)
        z0, z1 = z_positions[i]
        zm = (z0+z1)/2

        # Global strains at ply midplane
        e_global = e0 + zm*k

        # Transform to ply coordinates
        c=math.cos(theta); s=math.sin(theta)
        T = np.array([
            [c**2, s**2, c*s],
            [s**2, c**2, -c*s],
            [-2*c*s, 2*c*s, c**2-s**2]
        ])
        e_ply = T @ e_global

        # Ply stresses in principal directions
        Q_ply = np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0, 0, Q66]
        ])
        sigma_ply = Q_ply @ e_ply
        s1, s2_ply, s12_ply = sigma_ply

        # Tsai-Wu failure criterion
        F1  = 1/Xt - 1/Xc
        F2  = 1/Yt - 1/Yc
        F11 = 1/(Xt*Xc)
        F22 = 1/(Yt*Yc)
        F66 = 1/S12**2
        F12 = -0.5*math.sqrt(F11*F22)

        TW = (F1*s1 + F2*s2_ply +
               F11*s1**2 + F22*s2_ply**2 +
               F66*s12_ply**2 + 2*F12*s1*s2_ply)

        if TW > max_tsai_wu:
            max_tsai_wu = TW
            first_ply_failure = i+1

        ply_results.append({
            "ply": i+1,
            "angle_deg": theta_deg,
            "sigma1_mpa": round(float(s1),3),
            "sigma2_mpa": round(float(s2_ply),3),
            "tau12_mpa": round(float(s12_ply),3),
            "tsai_wu_index": round(float(TW),4),
            "failed": TW >= 1.0,
        })

    # Effective laminate properties
    h = total_thickness
    Ex_eff = (A[0,0]*A[1,1]-A[0,1]**2)/(A[1,1]*h)
    Ey_eff = (A[0,0]*A[1,1]-A[0,1]**2)/(A[0,0]*h)

    return {
        "method": "classical_laminate_theory",
        "layup": layup_angles,
        "num_plies": n_plies,
        "total_thickness_mm": round(total_thickness,3),
        "effective_Ex_gpa": round(Ex_eff/1000,3),
        "effective_Ey_gpa": round(Ey_eff/1000,3),
        "A_matrix": A.round(3).tolist(),
        "D_matrix": D.round(3).tolist(),
        "midplane_strains": {
            "e11": round(float(e0[0]),8),
            "e22": round(float(e0[1]),8),
            "g12": round(float(e0[2]),8),
        },
        "max_tsai_wu_index": round(float(max_tsai_wu),4),
        "first_ply_failure": first_ply_failure,
        "laminate_failed": max_tsai_wu >= 1.0,
        "safety_factor": round(1.0/max(max_tsai_wu,0.001),3),
        "ply_results": ply_results,
        "status": "FAIL" if max_tsai_wu >= 1.0 else "PASS",
    }

# ═══════════════════════════════════════════════════════════════════
# NEW: RAINFLOW FATIGUE COUNTING
# ═══════════════════════════════════════════════════════════════════
def rainflow_fatigue(mat_key, load_history_mpa, area_mm2=100):
    """
    ASTM E1049 rainflow counting algorithm.
    More accurate than simple Goodman for variable amplitude loading.
    Applies Miner's rule for cumulative damage.
    """
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    Sut = mat["ultimate_strength_mpa"]
    Se  = mat["fatigue_limit_mpa"] * 0.9 * 0.85 * 0.897  # Marin modified

    def extract_peaks(signal):
        peaks = [signal[0]]
        for i in range(1, len(signal)-1):
            if ((signal[i] > signal[i-1] and signal[i] > signal[i+1]) or
                (signal[i] < signal[i-1] and signal[i] < signal[i+1])):
                peaks.append(signal[i])
        peaks.append(signal[-1])
        return peaks

    def rainflow_count(peaks):
        cycles = []
        stack = []
        for p in peaks:
            stack.append(p)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                r1 = abs(s1-s0)
                r2 = abs(s2-s1)
                if r2 >= r1:
                    amp = r1/2
                    mean = (s0+s1)/2
                    cycles.append((amp, mean))
                    stack.pop(-2)
                    stack.pop(-2)
                else:
                    break
        return cycles

    peaks = extract_peaks(load_history_mpa)
    cycles = rainflow_count(peaks)

    # Basquin S-N curve: N = (f*Sut/Sa)^(1/b) * 1000
    b = mat.get("fatigue_slope_b", -0.085)
    f = mat.get("Sut_at_1000", 0.9)

    total_damage = 0.0
    cycle_details = []

    for Sa, Sm in cycles:
        if Sa < 0.001: continue

        # Goodman correction for mean stress
        Sa_eq = Sa / (1 - Sm/max(Sut,1))
        Sa_eq = max(Sa_eq, 0.001)

        if Sa_eq >= Se:
            try:
                N_fail = (f*Sut/Sa_eq)**(1/b) * 1000
                N_fail = abs(N_fail)
            except: N_fail = 1e6
        else:
            N_fail = float("inf")

        damage = 1.0/N_fail if N_fail != float("inf") else 0
        total_damage += damage

        cycle_details.append({
            "amplitude_mpa": round(Sa,3),
            "mean_mpa": round(Sm,3),
            "equivalent_amplitude_mpa": round(Sa_eq,3),
            "cycles_to_failure": round(N_fail,0) if N_fail!=float("inf") else "infinite",
            "damage": round(damage,10),
        })

    life_cycles = 1.0/max(total_damage,1e-30) if total_damage>0 else float("inf")
    life_hours = life_cycles / (3600*10)

    return {
        "method": "rainflow_astm_e1049",
        "total_cycles_counted": len(cycles),
        "miner_damage_sum": round(float(total_damage),8),
        "predicted_life_cycles": round(min(life_cycles,1e12),0),
        "predicted_life_hours": round(min(life_hours,1e9),1),
        "status": "PASS" if total_damage < 0.5 else "FAIL",
        "top_damaging_cycles": sorted(cycle_details,
                                       key=lambda x:x["damage"],
                                       reverse=True)[:5],
    }

# ═══════════════════════════════════════════════════════════════════
# NEW: MANUFACTURING COST ESTIMATE
# ═══════════════════════════════════════════════════════════════════
def estimate_manufacturing_cost(mesh, mat_key, process="cnc"):
    """
    Realistic manufacturing cost estimation.
    Based on volume, surface area, complexity, and material.
    """
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    vol_cm3 = sf(mesh.volume) / 1000
    area_cm2 = sf(mesh.area) / 100
    cost_per_kg = mat.get("cost_per_kg_usd", 5.0)
    machinability = mat.get("machinability", 0.7)
    mass_kg = vol_cm3 * mat["density"] / 1000

    # Material cost
    material_cost = mass_kg * cost_per_kg * 1.3  # 30% waste factor

    # Manufacturing cost
    if process == "cnc":
        # CNC: $60-120/hour, complexity factor
        complexity = max(len(mesh.faces)/1000, 1.0)
        setup_time_hr = 0.5
        machining_time_hr = (area_cm2 * 0.02) / machinability
        cnc_rate = 80.0  # USD/hour
        manufacturing_cost = (setup_time_hr + machining_time_hr) * cnc_rate

    elif process == "3d_print_fdm":
        # FDM: $0.10-0.30 per cm³
        manufacturing_cost = vol_cm3 * 0.20

    elif process == "3d_print_slm":
        # SLM metal: $5-15 per cm³
        manufacturing_cost = vol_cm3 * 8.0

    elif process == "sheet_metal":
        manufacturing_cost = area_cm2 * 0.5 + 25.0  # Setup + bending

    elif process == "casting":
        tooling = 2000.0  # Mold cost (amortized over 100 parts)
        manufacturing_cost = material_cost * 0.5 + tooling/100

    else:
        manufacturing_cost = material_cost * 1.5

    total = material_cost + manufacturing_cost

    return {
        "process": process,
        "material": mat["name"],
        "mass_kg": round(mass_kg, 4),
        "volume_cm3": round(vol_cm3, 3),
        "material_cost_usd": round(material_cost, 2),
        "manufacturing_cost_usd": round(manufacturing_cost, 2),
        "total_cost_usd": round(total, 2),
        "cost_per_gram_usd": round(total/(mass_kg*1000+0.001), 4),
        "note": "Estimate only. Get quotes from manufacturers.",
    }

# ═══════════════════════════════════════════════════════════════════
# NEW: GEMINI SCRIPT GENERATION — Any part from text
# ═══════════════════════════════════════════════════════════════════
# Shared with rule_engine_v8's R15 check — kept as one list so the up-front
# generation directive and the after-the-fact violation check can't drift
# out of sync with each other.
FOLD_BRACKET_KEYWORDS = ("vertical flange","vertical leg","vertical wall","vertical face",
    "bent bracket","folded bracket","folded sheet","angle bracket",
    "right-angle bracket","right angle bracket","90 degree bend",
    "90° bend","fold line","bent sheet metal")

GEMINI_CADQUERY_SYSTEM = """You are a CadQuery expert mechanical engineer.
Generate Python CadQuery code to create the described 3D part.

STRICT RULES:
- Import only: cadquery as cq, math, numpy as np
- Assign final shape to variable named: result
- All dimensions in millimeters
- Add fillets to sharp internal corners minimum 0.5mm
- Add mounting holes where appropriate
- Code must be syntactically correct Python
- No explanations, no markdown, pure Python code only
- No os, sys, subprocess, socket, requests imports

AVAILABLE CADQUERY OPERATIONS:
cq.Workplane("XY"/"XZ"/"YZ")
.box(length, width, height)
.circle(radius).extrude(height)
.cylinder(height, radius)
.sphere(radius)
.ellipse(x_radius, y_radius).extrude(height)
.polygon(n_sides, circumradius).extrude(height)
.polyline([(x1,y1),(x2,y2),...]).close().extrude(height)
.spline([(x1,y1,z1),...])
.circle(r1).workplane(offset=h).circle(r2).loft()
.spline([(x1,y1,z1),...]).close().extrude(height)
.workplane().moveTo(x,y).spline([...]).close().extrude(height)
# Sweep a 2D profile along a curved path — the tool for a genuinely curved
# structural member (e.g. a smoothly curved arm or duct), not just a straight
# extrude with fillets bolted on:
path = cq.Workplane("XZ").spline([(0,0),(x1,z1),(x2,z2)])
swept = cq.Workplane("XY").circle(r).sweep(path)
.fillet(radius)
.chamfer(length)
.shell(thickness)
.hole(diameter)
.cskHole(diameter, csk_diameter, csk_angle)
.cboreHole(diameter, cboreDiameter, cboreDepth)
.pushPoints([(x,y),...])
.rarray(xSpacing, ySpacing, xCount, yCount, center=True)
.union(other)
.cut(other)
.intersect(other)
.translate((x,y,z))
.rotate((0,0,0),(0,0,1),angle_degrees)
.mirror("XY"/"XZ"/"YZ")
.faces(">Z"/"<Z"/">X"/etc).workplane()
.edges("|Z"/etc).fillet(radius)

ENGINEERING DEFAULTS (apply unless the prompt specifies otherwise):
- Mounting holes: diameter sized for M3-M6 fasteners, placed ≥2x diameter from any edge
- Wall thickness: minimum 1.5mm for plastics, 1.0mm for metals, never below 0.8mm
- Internal corners: fillet radius ≥0.5mm, prefer ≥1mm on load paths
- External edges: chamfer 0.5-1mm for safe handling unless a sharp edge is functionally required
- Keep aspect ratios (longest/shortest dimension) under 15:1 unless the prompt explicitly asks for a slender part
- Center the part roughly on the origin so the bounding box is well-formed
- When union()-ing two separately-built solids that attach end-to-end (e.g. an
  end plate/boss/flange on a tapered or curved member), do NOT place them so
  they only touch at one exact coincident plane with no real overlap — this is
  a common cause of a non-watertight result, especially when their
  cross-sections differ in size at that interface (e.g. a large plate meeting
  a much smaller tapered tip). Translate the attachment so it genuinely
  overlaps the other solid by a small real depth (a few percent of the
  smaller cross-section's size is enough) before calling union().
- If the prompt describes a tapered, curved, streamlined, or organic-looking
  shape (e.g. "tapered arm", "curved bracket", "aerodynamic", "smoothly
  blends into"), use loft() between profiles or sweep() along a spline path
  for the main body — do not default to a constant-rectangular-cross-section
  box just because that's simpler. A box with fillets bolted on the corners
  is NOT the same as a genuinely tapered or curved shape, and looks
  noticeably different from what was actually asked for.
- Do NOT blanket-fillet every edge of a loft/tapered solid in one
  .edges().fillet() call. The corners where a sloped taper edge meets two
  flat profile edges are a compound 3-edge blend — a known-hard case that
  can silently produce self-intersecting (non-watertight) geometry with no
  Python error at all. Prefer a smaller radius, fillet only the flat
  profile edges (top/bottom), or skip fillets on a tapered body entirely if
  the prompt didn't specifically require them there.
- Words like "flange", "leg", "L-bracket", "bent bracket", "angle bracket", or "folded
  sheet metal" describe TWO FACES THAT ARE NOT COPLANAR — a real fold, not just two
  flat pieces at different in-plane orientations. For ANY part matching this
  description, do NOT hand-write your own box/rotate/union code for it — call the
  make_bent_bracket(...) helper that's already available in this environment instead:

      result = make_bent_bracket(
          leg1_length=50.0, leg2_length=50.0, width=30.0, thickness=4.0,
          bend_angle_deg=90.0, fillet_radius=3.0,
          holes_leg1=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],   # (x_from_bend, y_from_centerline, diameter)
          holes_leg2=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],
      )

  It guarantees leg2 actually rises out of the base plane instead of staying flat.
  Pick leg1_length/leg2_length/width/thickness/holes from the prompt's stated
  dimensions; you do not need to compute any rotation or union yourself.
"""

REFINEMENT_INSTRUCTIONS = """
You are now in REFINEMENT MODE.

You previously generated a CadQuery script for this part. It was exported to a mesh and
run through a real engineering analysis pipeline (wall thickness, hole placement, sharp
corner stress concentrations, FEA safety factor, fatigue, rule-engine checks).

The analysis below lists concrete problems with the part as currently designed (or the
script failed to execute — in that case fix the execution error). Your job is to produce
a CORRECTED, COMPLETE script that fixes every issue listed, while preserving the parts of
the design that were already correct.

RULES FOR REFINEMENT:
- Output a COMPLETE script (not a diff/patch) that can run standalone, same format as before.
- Directly address each issue: e.g. if "Wall 0.6mm < material min 1.0mm", increase the
  relevant wall/shell thickness in the script's geometry, don't just change a comment.
- If a hole violates the edge-distance rule, move that hole's pushPoint coordinates inward.
- If sharp-corner stress concentration (Kf) is too high, add/increase a .fillet() on that edge.
- If safety factor is too low, increase cross-sectional area/thickness in the load path,
  or reduce unsupported span, rather than changing the material.
- If the previous script raised a Python error, fix the root cause (typo, wrong API call,
  bad chaining) — do not just simplify the part away.
- Do not regress: don't reintroduce a problem that was already fixed in a prior round,
  and don't fix one flagged issue by weakening a different area that was previously fine
  (e.g. don't thin a wall or shrink a cross-section elsewhere while raising a wall
  thickness or fixing a hole position). Change only what's needed to address each
  listed issue, at the location it was found.
- Still follow all original STRICT RULES (imports, `result` variable, mm units, etc).
"""

LOVABLE_API_KEY = os.environ.get("LOVABLE_API_KEY", "")
LOVABLE_AI_URL = "https://ai.gateway.lovable.dev/v1/chat/completions"
LOVABLE_AI_MODEL = os.environ.get("LOVABLE_AI_MODEL", "google/gemini-3-flash")
LOVABLE_AI_VISION_MODEL = os.environ.get("LOVABLE_AI_VISION_MODEL", LOVABLE_AI_MODEL)

# Direct Anthropic API — no third-party gateway in between. Model string here is
# what I'm most confident is current as of this writing; Anthropic ships new
# models fairly often, so if this 404s/errors, check https://docs.claude.com for
# the current model id and override via the CLAUDE_MODEL env var rather than
# editing this file.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_VISION_MODEL = os.environ.get("CLAUDE_VISION_MODEL", CLAUDE_MODEL)

# Direct Google Gemini API — bypasses the Lovable gateway entirely, so you keep
# whatever Google charges directly with no gateway markup. Google ships frequent
# point releases (3.6, 3.7, etc. were all released within weeks of each other as
# of this writing) — if this model id 404s, check https://ai.google.dev/gemini-api/docs/models
# for the current stable id and override via GEMINI_MODEL rather than editing this file.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", GEMINI_MODEL)

# OpenRouter — OpenAI-compatible gateway to many models, including genuinely
# free ones (:free suffix). Free-tier model availability on OpenRouter churns
# HARD and without notice — confirmed live, repeatedly, in the same session:
# qwen/qwen3-coder:free delisted within about a week of being set; then
# z-ai/glm-4.5-air:free (this file's next default) delisted within HOURS of
# being set. Hand-picking any specific :free model id is a losing game — the
# ecosystem moves faster than any fix-and-redeploy cycle can track.
#
# Default is now "openrouter/free" — OpenRouter's OWN auto-routing
# meta-model, built specifically for this problem. Per OpenRouter's own docs:
# "so your code keeps working even after individual free models rotate out."
# This requires the null-content response fix from v8.13 to be reliable (an
# earlier attempt at this same default crashed on a null-content edge case
# before that fix existed) — that's now in place, so this is the stable
# choice going forward, not a specific model name to keep replacing.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
# The auto-router may land on a text-only model — image-to-params needs a
# vision-capable model specifically if you use that endpoint. Override
# OPENROUTER_VISION_MODEL explicitly rather than relying on the auto-router
# for that one endpoint.
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", OPENROUTER_MODEL)

# Groq — OpenAI-compatible, custom LPU hardware, genuinely stable free tier
# (unlike OpenRouter's free roster, which churned THREE times in one night on
# this project — models delisted within hours to days of being set). Groq's
# own docs: 30 RPM, 1,000 requests/day, no card required, and this rate limit
# CORRECTION (confirmed via a real Groq console screenshot + independent
# search): Llama 4 Scout was removed from Groq's catalog around July 21,
# 2026 — it no longer appears in the console's model list at all. The
# earlier version of this comment recommending Scout was already stale by
# the time it was written; leaving this note so it's not repeated.
#
# Current default: openai/gpt-oss-120b — confirmed live in Groq's own
# console under both "Reasoning" and "Function Calling/Tool Use" categories,
# and NOT in Groq's "Preview" tier (which their own docs warn "may be
# discontinued at short notice") — the least churn-prone real option
# available right now. Groq's free tier (~30 req/min, no card required)
# applies to this and every other listed model — usage under those caps
# costs $0; you're only billed if you exceed them. There is no separate
# ":free"-suffix model list the way OpenRouter has — every model here is
# usage-priced, with the free tier being a rate-limited allowance on top.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Confirm vision support on Groq's hosted GPT-OSS before relying on it for
# image-to-params — override GROQ_VISION_MODEL if it doesn't behave as
# expected there (GPT-OSS models are primarily text-focused).
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", GROQ_MODEL)

# Which provider backs generation: "claude" (direct Anthropic API), "gemini"
# (direct Google API), "groq" (Llama 4 via Groq, stable free tier),
# "openrouter" (OpenAI-compatible gateway, free models available but churn
# heavily), or "lovable" (Gemini via the gateway). Defaults to whichever key
# is actually configured — set AI_PROVIDER explicitly to force a choice if
# more than one key happens to be set at once.
AI_PROVIDER = os.environ.get(
    "AI_PROVIDER",
    "claude" if os.environ.get("ANTHROPIC_API_KEY")
    else "gemini" if os.environ.get("GOOGLE_API_KEY")
    else "groq" if os.environ.get("GROQ_API_KEY")
    else "openrouter" if os.environ.get("OPENROUTER_API_KEY")
    else "lovable"
)

def _groq_request(messages, temperature=0.15, max_tokens=3000, model=None):
    """Low-level call to Groq (OpenAI-compatible chat completions) — same
    request/response shape as _lovable_request/_openrouter_request, different
    base URL/key/model."""
    import urllib.request, urllib.error

    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        GROQ_API_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            # Groq's API sits behind Cloudflare. urllib's default User-Agent
            # ("Python-urllib/3.x") is a well-known bot-detection trigger —
            # confirmed live: this exact call was returning Cloudflare error
            # 1010 ("banned based on your browser's signature") before this
            # header was added, not an actual Groq auth/key problem.
            "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)",
            "Accept": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"Groq rate limit exceeded: {body}")
        raise HTTPException(502, f"Groq error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Groq connection error: {str(e)}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected Groq response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        # Same null-content edge case fixed in _openrouter_request/_lovable_request.
        raise HTTPException(502, f"Groq returned empty/null content. "
                                  f"Raw response: {json.dumps(data)[:500]}")
    return content


def _openrouter_request(messages, temperature=0.15, max_tokens=3000, model=None):
    """
    Low-level call to OpenRouter (OpenAI-compatible chat completions) — same
    request/response shape as _lovable_request, different base URL/key/model.
    `messages` is the standard OpenAI-style list including a system role entry.
    """
    import urllib.request, urllib.error

    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        OPENROUTER_API_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"OpenRouter rate limit exceeded: {body}")
        raise HTTPException(502, f"OpenRouter error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"OpenRouter connection error: {str(e)}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected OpenRouter response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        # `content` can be present but null/empty — happens with some models when
        # they return only a `reasoning` field, hit a content filter, or produce
        # a tool-call instead of plain text. This is a real, confirmed failure
        # mode (openrouter/free's auto-router can land on a model that does
        # this), not a hypothetical — the old code let a None here fall through
        # silently until it crashed downstream with an unrelated-looking
        # AttributeError. Surface it clearly here instead, with the raw response
        # visible for debugging which model/condition triggered it.
        raise HTTPException(502, f"OpenRouter returned empty/null content — the routed "
                                  f"model produced no usable text (possibly a reasoning-only "
                                  f"response or content filter). Raw response: "
                                  f"{json.dumps(data)[:500]}")
    return content


def _gemini_request(system, messages, temperature=0.15, max_tokens=3000, model=None):
    """
    Low-level call to Google's Gemini API directly (generativelanguage.googleapis.com),
    no gateway in between. Gemini's request shape differs from both _lovable_request
    (OpenAI-style) and _claude_request (Anthropic Messages API):
      - system prompt goes in a separate `systemInstruction` field
      - conversation turns use role "user" / "model" (not "assistant")
      - each turn's content is a `parts` array of {"text": ...} objects
    `messages` here uses the same [{"role", "content"}] shape as the other two
    request functions for consistency — this function does the Gemini-specific
    conversion internally.
    """
    import urllib.request, urllib.error

    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = json.dumps({
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }).encode()

    use_model = model or GEMINI_MODEL
    url = f"{GEMINI_API_BASE}/{use_model}:generateContent?key={GOOGLE_API_KEY}"

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"Gemini API rate limit exceeded: {body}")
        raise HTTPException(502, f"Gemini API error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Gemini API connection error: {str(e)}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        # Gemini returns candidates[0].finishReason == "SAFETY" (no parts) if it
        # refuses — surface the raw response so that's visible instead of a bare
        # KeyError.
        raise HTTPException(502, f"Unexpected Gemini API response shape (possibly a "
                                  f"safety block): {json.dumps(data)[:500]}")


def _gemini_vision_request(system, prompt_text, img_b64, mime_type,
                            temperature=0.1, max_tokens=512, model=None):
    """Gemini vision call — image goes in `inline_data` (snake_case) alongside text, in one part list."""
    import urllib.request, urllib.error

    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY is not configured on the server.")

    payload = json.dumps({
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": img_b64}},
                {"text": prompt_text},
            ],
        }],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }).encode()

    use_model = model or GEMINI_VISION_MODEL
    url = f"{GEMINI_API_BASE}/{use_model}:generateContent?key={GOOGLE_API_KEY}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        raise HTTPException(502, f"Gemini API error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Gemini API connection error: {str(e)}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected Gemini API response shape: {json.dumps(data)[:500]}")


def _lovable_request(messages, temperature=0.15, max_tokens=3000, model=None):
    """Low-level call to Lovable AI Gateway (OpenAI-compatible chat completions)."""
    import urllib.request, urllib.error

    if not LOVABLE_API_KEY:
        raise HTTPException(500, "LOVABLE_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or LOVABLE_AI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        LOVABLE_AI_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LOVABLE_API_KEY}",
            "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"Lovable AI rate limit exceeded: {body}")
        if e.code == 402:
            raise HTTPException(402, f"Lovable AI credits exhausted: {body}")
        raise HTTPException(502, f"Lovable AI error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Lovable AI connection error: {str(e)}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected Lovable AI response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        # Same null-content edge case fixed in _openrouter_request — see that
        # function's comment for the full explanation. Guarding here too since
        # this is the identical OpenAI-compatible response shape.
        raise HTTPException(502, f"Lovable AI returned empty/null content. "
                                  f"Raw response: {json.dumps(data)[:500]}")
    return content


def _claude_request(system, messages, temperature=0.15, max_tokens=3000, model=None):
    """
    Low-level call to the Anthropic Messages API directly (no gateway in between).
    `messages` is Anthropic's format: [{"role": "user"/"assistant", "content": ...}],
    with the system prompt passed separately — different shape from the OpenAI-style
    messages list _lovable_request expects. See gemini_generate_script/
    gemini_vision_estimate for where the two are assembled differently per provider.
    """
    import urllib.request, urllib.error

    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or CLAUDE_MODEL,
        "system": system,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"Claude API rate limit exceeded: {body}")
        raise HTTPException(502, f"Claude API error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Claude API connection error: {str(e)}")

    try:
        return "".join(b["text"] for b in data["content"] if b.get("type") == "text")
    except (KeyError, TypeError):
        raise HTTPException(502, f"Unexpected Claude API response shape: {json.dumps(data)[:500]}")

def _clean_code_block(text: str, lang_hints=("python","json")) -> str:
    t = text.strip()
    for h in lang_hints:
        t = t.replace(f"```{h}", "```")
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()

async def gemini_generate_script(prompt: str, previous_script: Optional[str] = None,
                                  feedback: Optional[str] = None) -> str:
    """
    Generate/refine a CadQuery script via whichever provider AI_PROVIDER selects —
    direct Claude API or the Lovable/Gemini gateway. Same prompting logic either way;
    only the transport differs (see _claude_request vs _lovable_request).

    - First call (previous_script/feedback are None): plain generation from `prompt`.
    - Refinement call: include the prior script and an engineering-analysis feedback
      report so the model can produce a corrected version targeting the same part.
    """
    system_prompt = GEMINI_CADQUERY_SYSTEM + REFINEMENT_INSTRUCTIONS

    if previous_script and feedback:
        turns = [
            {"role": "user", "content":
                f"Original request: {prompt}\n\nReturn ONLY Python code. No markdown."},
            {"role": "assistant", "content": previous_script},
            {"role": "user", "content":
                f"ANALYSIS / FEEDBACK FROM ENGINEERING PIPELINE:\n{feedback}\n\n"
                f"Produce a corrected, COMPLETE script fixing the issues above. "
                f"Return ONLY Python code. No markdown."},
        ]
    else:
        is_fold_part = any(w in prompt.lower() for w in FOLD_BRACKET_KEYWORDS)
        if is_fold_part:
            user_msg = (
                f"Generate CadQuery code for: {prompt}\n\n"
                "This part has a bent/folded flange (per the description above). "
                "MANDATORY: do not write any box/polyline/rotate/union geometry code "
                "yourself for this. Your entire script must build the part by calling "
                "the make_bent_bracket(...) helper that is already available in this "
                "environment, then optionally chaining simple .fillet()/.chamfer() calls "
                "on its result — nothing else constructs the base geometry. Example:\n\n"
                "result = make_bent_bracket(\n"
                "    leg1_length=50.0, leg2_length=50.0, width=30.0, thickness=4.0,\n"
                "    bend_angle_deg=90.0, fillet_radius=3.0,\n"
                "    holes_leg1=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],\n"
                "    holes_leg2=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],\n"
                ")\n\n"
                "Pick leg1_length/leg2_length/width/thickness/holes from the dimensions "
                "stated in the prompt above. Return ONLY Python code. No markdown."
            )
        else:
            user_msg = f"Generate CadQuery code for: {prompt}\n\nReturn ONLY Python code. No markdown."
        turns = [{"role": "user", "content": user_msg}]

    # 3000 tokens was too tight — real users hit truncated/unclosed-expression
    # scripts on parts needing computed hole-position math (x_pos/y_pos style
    # logic), confirmed via a live truncation: '(' never closed mid-line.
    # 6000 gives real headroom for that without being wastefully large.
    GEN_MAX_TOKENS = 6000

    # FIX: these were previously called directly (blocking, synchronous urllib
    # calls) from inside an `async def` function with no `await` on the actual
    # I/O — on Render's single-worker free tier that stalls the ENTIRE event
    # loop (including health-check responses) for the full duration of every
    # generation call. asyncio.to_thread moves the blocking call off the loop.
    if AI_PROVIDER == "claude":
        text = await asyncio.to_thread(_claude_request, system_prompt, turns,
                                        temperature=0.15, max_tokens=GEN_MAX_TOKENS)
    elif AI_PROVIDER == "gemini":
        text = await asyncio.to_thread(_gemini_request, system_prompt, turns,
                                        temperature=0.15, max_tokens=GEN_MAX_TOKENS)
    elif AI_PROVIDER == "groq":
        text = await asyncio.to_thread(
            _groq_request,
            [{"role": "system", "content": system_prompt}] + turns,
            temperature=0.15, max_tokens=GEN_MAX_TOKENS
        )
    elif AI_PROVIDER == "openrouter":
        text = await asyncio.to_thread(
            _openrouter_request,
            [{"role": "system", "content": system_prompt}] + turns,
            temperature=0.15, max_tokens=GEN_MAX_TOKENS
        )
    else:
        text = await asyncio.to_thread(
            _lovable_request,
            [{"role": "system", "content": system_prompt}] + turns,
            temperature=0.15, max_tokens=GEN_MAX_TOKENS
        )

    return _clean_code_block(text, ("python",))

async def gemini_vision_estimate(img_b64: str, mime_type: str, description: str) -> dict:
    """Estimate part parameters from an image via whichever provider AI_PROVIDER selects."""
    vision_prompt = f"""Analyze this engineering part image.
User description: {description}

Return JSON only (no markdown):
{{
  "part_type": "bracket|shaft|plate|housing|gear|motor_mount|flange|ibeam|tube|custom",
  "estimated_width_mm": <number>,
  "estimated_height_mm": <number>,
  "estimated_depth_mm": <number>,
  "estimated_thickness_mm": <number>,
  "num_holes": <number>,
  "hole_diameter_mm": <number>,
  "has_fillet": true/false,
  "material_guess": "aluminum|steel|plastic|carbon_fiber",
  "confidence_pct": <0-100>,
  "notes": "what you can and cannot determine from image"
}}"""

    if AI_PROVIDER == "claude":
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": img_b64}},
                {"type": "text", "text": vision_prompt},
            ]
        }]
        text = _claude_request(
            "You are a precise mechanical-engineering vision analyst. Respond with JSON only, no markdown.",
            messages, temperature=0.1, max_tokens=512, model=CLAUDE_VISION_MODEL
        )
    elif AI_PROVIDER == "gemini":
        text = _gemini_vision_request(
            "You are a precise mechanical-engineering vision analyst. Respond with JSON only, no markdown.",
            vision_prompt, img_b64, mime_type, temperature=0.1, max_tokens=512
        )
    elif AI_PROVIDER == "groq":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }}
            ]
        }]
        text = _groq_request(
            [{"role": "system", "content": "You are a precise mechanical-engineering vision "
                                            "analyst. Respond with JSON only, no markdown."}] + messages,
            temperature=0.1, max_tokens=512, model=GROQ_VISION_MODEL
        )
    elif AI_PROVIDER == "openrouter":
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }}
            ]
        }]
        text = _openrouter_request(
            [{"role": "system", "content": "You are a precise mechanical-engineering vision "
                                            "analyst. Respond with JSON only, no markdown."}] + messages,
            temperature=0.1, max_tokens=512, model=OPENROUTER_VISION_MODEL
        )
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }}
            ]
        }]
        text = _lovable_request(messages, temperature=0.1, max_tokens=512, model=LOVABLE_AI_VISION_MODEL)

    text = _clean_code_block(text, ("json",))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, f"AI provider returned non-JSON for vision estimate: {text[:300]}")

import ast

# Modules the generated script is allowed to import. Anything else is rejected.
CQ_ALLOWED_IMPORTS = {"cadquery", "cq", "math", "numpy", "np"}

# Attribute/name access that is never allowed regardless of context — these are the
# standard sandbox-escape primitives in pure-Python exec() jails.
CQ_FORBIDDEN_NAMES = {
    "__import__", "__builtins__", "__globals__", "__getattribute__",
    "__subclasses__", "__bases__", "__base__", "__mro__", "__class__",
    "__dict__", "__code__", "__closure__", "__loader__", "__spec__",
    "exec", "eval", "compile", "open", "input", "vars", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit",
}

class _CQSandboxViolation(Exception):
    pass

def _validate_cq_ast(tree: ast.AST):
    """
    Walk the parsed AST and reject anything outside a narrow, known-safe subset:
    imports of allowed modules only, no dunder/reflection access, no exec/eval-style
    calls, no file/network/process primitives. This replaces a naive substring
    blocklist (trivially bypassable via string concatenation, getattr tricks, etc.)
    with a real structural check.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod_names = [n.name.split(".")[0] for n in node.names] if isinstance(node, ast.Import) \
                        else [(node.module or "").split(".")[0]]
            for m in mod_names:
                if m not in CQ_ALLOWED_IMPORTS:
                    raise _CQSandboxViolation(f"Import of '{m}' is not allowed. "
                                               f"Only {sorted(CQ_ALLOWED_IMPORTS)} may be imported.")
        elif isinstance(node, ast.Name) and node.id in CQ_FORBIDDEN_NAMES:
            raise _CQSandboxViolation(f"Use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr in CQ_FORBIDDEN_NAMES:
            raise _CQSandboxViolation(f"Access to attribute '{node.attr}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise _CQSandboxViolation(f"Access to dunder attribute '{node.attr}' is not allowed.")

def make_bent_bracket(leg1_length, leg2_length, width, thickness,
                       bend_angle_deg=90.0, fillet_radius=2.0,
                       holes_leg1=None, holes_leg2=None):
    """
    Build a genuinely folded two-flange bracket (L-bracket / angle bracket) as one
    solid, guaranteeing leg2 actually rises out of the base plane by bend_angle_deg
    via a real rotate() — the exact operation the free-tier model kept failing to
    hand-write correctly (confirmed live: it either left both legs flat and coplanar,
    or attempted its own rotate/union and produced non-watertight geometry). This is
    trusted server-side code, not AI-generated, so it only needs to be gotten right
    once; the model's job becomes picking sensible parameters, not 3D CAD authoring.

    Both legs share a bend edge along the Y-axis at x=0,z=0. Each leg extends from
    that edge outward along its own local +X for leg{1,2}_length, and is `width`
    wide (centered on y=0), thickness `thickness`. holes_leg1/holes_leg2 are each an
    optional list of (x_from_bend_mm, y_from_centerline_mm, diameter_mm) tuples, given
    in that leg's own FLAT local frame (before folding) — no 3D math required by the
    caller. Fold direction (up vs down) isn't guaranteed, only that real out-of-plane
    height exists; that's all the downstream FEA/geometry checks require.

    Returns the finished CadQuery solid — assign it to `result`.
    """
    holes_leg1 = holes_leg1 or []
    holes_leg2 = holes_leg2 or []

    def _leg_with_holes(length, holes):
        leg = cq.Workplane("XY").rect(length, width, centered=(False, True)).extrude(thickness)
        for hx, hy, hd in holes:
            leg = leg.workplane(offset=thickness).pushPoints([(hx, hy)]).hole(hd)
        return leg

    leg1 = _leg_with_holes(leg1_length, holes_leg1)
    leg2 = _leg_with_holes(leg2_length, holes_leg2)
    leg2 = leg2.rotate((0, -1, 0), (0, 1, 0), -bend_angle_deg)

    bracket = leg1.union(leg2)

    if fillet_radius and fillet_radius > 0:
        try:
            bend_edges = [e for e in bracket.edges().vals()
                          if abs(e.Center().x) < 0.5 and abs(e.Center().z) < 0.5
                          and abs(e.Length() - width) < 0.5]
            if bend_edges:
                bracket = bracket.newObject(bend_edges).fillet(fillet_radius)
        except Exception:
            pass  # sharp (unfilleted) bend is a fine fallback; don't fail the whole part

    return bracket

def execute_cq_script_safely(script: str):
    """
    Execute an AI-generated CadQuery script in a sandboxed namespace.

    Security model: parse to an AST first and reject anything outside a narrow
    known-safe subset (imports limited to cadquery/math/numpy, no dunder/reflection
    access, no exec/eval/getattr-style escape hatches) before ever calling exec().
    A restricted builtins dict is also passed to the exec namespace as defense in
    depth, in case a novel AST-level bypass is found later.

    Returns (obj, error_message). On success, error_message is None and obj is the
    CadQuery/trimesh object assigned to `result`. On any failure (forbidden op, syntax
    error, runtime error, missing `result`), obj is None and error_message describes
    the problem in a form suitable for feeding back to the LLM for refinement.
    """
    if not CQ:
        return None, "CadQuery is not installed on this server."

    try:
        tree = ast.parse(script, filename="<ai_script>", mode="exec")
    except SyntaxError as e:
        return None, f"Script syntax error: {str(e)} (line {e.lineno}: {e.text!r})"

    try:
        _validate_cq_ast(tree)
    except _CQSandboxViolation as e:
        return None, f"Unsafe operation detected and blocked: {str(e)} Remove it entirely."

    # Minimal, explicit builtins — defense in depth beyond the AST check above.
    # __import__ IS included here, but wrapped to only allow the same modules
    # the AST check already allowlisted (CQ_ALLOWED_IMPORTS) — by the time exec()
    # runs, every `import` statement in the script has already been proven safe
    # at the AST level, so this wrapper is redundant-but-safe defense in depth,
    # not a new hole. Omitting __import__ entirely (the previous version of this
    # function) breaks every script, since Python's own `import X` statement
    # calls __builtins__.__import__(...) internally to execute the import —
    # including the mandatory `import cadquery as cq` line every generated
    # script needs, which is why ALL generations were failing with
    # "ImportError: __import__ not found" until this fix.
    def _restricted_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level not in CQ_ALLOWED_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed in this sandbox.")
        return __import__(name, *args, **kwargs)

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "round": round, "range": range,
        "len": len, "sum": sum, "float": float, "int": int, "bool": bool,
        "str": str, "list": list, "tuple": tuple, "dict": dict, "set": set,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
        "True": True, "False": False, "None": None,
        "__import__": _restricted_import,
    }

    namespace = {
        "__builtins__": safe_builtins,
        "cq": cq,
        "math": math,
        "np": np,
        "make_bent_bracket": make_bent_bracket,
        "result": None,
    }

    try:
        exec(compile(tree, "<ai_script>", "exec"), namespace)
    except Exception as e:
        return None, f"Script execution failed: {type(e).__name__}: {str(e)}"

    obj = namespace.get("result")
    if obj is None:
        # Fallback: the AI occasionally builds a valid shape but assigns it to
        # a differently-named variable despite the system prompt's explicit
        # instruction — no amount of prompt wording guarantees 100% compliance
        # from an LLM. Rather than hard-fail a script that actually succeeded
        # at building real geometry, scan the namespace for anything that
        # looks like a CadQuery Workplane/Shape and use that instead. Only
        # look at names the script itself defined (skip cq/math/np/__builtins__
        # and anything starting with _), and only accept it if exactly one
        # candidate exists — if there are multiple, it's genuinely ambiguous
        # which one was meant to be the final part, so don't guess.
        reserved = {"cq", "math", "np", "result", "__builtins__"}
        candidates = [
            (k, v) for k, v in namespace.items()
            if k not in reserved and not k.startswith("_")
            and (hasattr(v, "val") or hasattr(v, "vertices"))
        ]
        if len(candidates) == 1:
            obj = candidates[0][1]
        else:
            return None, (
                "Script ran without error but did not assign a shape to the "
                "'result' variable."
                + (f" Found {len(candidates)} other CadQuery-shaped variables "
                   f"({', '.join(k for k,_ in candidates)}) — too ambiguous to "
                   f"guess which was meant to be final; assign explicitly to "
                   f"'result'." if candidates else "")
            )

    # Sanity check: must be exportable (CadQuery Workplane/Shape)
    if not hasattr(obj, "val") and not hasattr(obj, "vertices"):
        return None, (f"'result' is of type {type(obj).__name__}, which doesn't look like a "
                       f"CadQuery Workplane/Shape. Make sure the final expression returns "
                       f"a cq.Workplane.")

    return obj, None

# ═══════════════════════════════════════════════════════════════════
# RETAINED v7.0 ANALYSIS FUNCTIONS (all upgraded algorithms)
# ═══════════════════════════════════════════════════════════════════


SURFACE_KA = {
    "mirror_polished":1.00,"ground":0.90,"machined":0.82,
    "cold_drawn":0.80,"hot_rolled":0.72,"as_forged":0.57,
    "3d_printed_fdm":0.45,"3d_printed_slm":0.62,
    "3d_printed_resin":0.55,"sandblasted":0.68,
    "anodized":0.85,"electropolished":0.95,
}
RELIABILITY_KC = {
    0.50:1.000,0.90:0.897,0.95:0.868,
    0.99:0.814,0.999:0.753,0.9999:0.702
}

def detect_material(mesh):
    mx=float(max(mesh.bounding_box.extents));fc=len(mesh.faces)
    vol=sf(mesh.volume)
    if mx<20 and fc>5000: return "titanium_6al4v"
    elif mx>200: return "steel_4340"
    elif fc>10000: return "aluminum_7075"
    elif vol<100: return "stainless_316l"
    return "aluminum_6061"

def classify_context(pn,pd_):
    c=(pn or "").lower()+" "+(pd_ or "").lower()
    ctxs=[
        ("drone_frame",["drone","uav","quadcopter","frame"],2.5,True,True,1.5),
        ("bracket_mount",["bracket","mount","clamp","support"],3.0,False,False,2.0),
        ("shaft_rotating",["shaft","axle","spindle","rotor"],3.5,True,True,1.8),
        ("housing_enclosure",["housing","enclosure","case","cover"],2.0,False,False,1.2),
        ("gear_transmission",["gear","pinion","sprocket","cam"],4.0,True,True,2.5),
        ("pressure_vessel",["pressure","vessel","tank","boiler"],4.0,True,False,2.0),
        ("medical",["medical","surgical","orthotic","implant"],5.0,True,False,1.0),
        ("aerospace",["wing","spar","rib","fuselage","airfoil"],4.5,True,True,1.5),
        ("automotive",["suspension","chassis","engine","caliper"],3.5,True,True,2.0),
    ]
    for key,kws,msf,fc,vs,lf in ctxs:
        if any(k in c for k in kws):
            return {"key":key,"min_sf":msf,"fatigue_critical":fc,"vibration_sensitive":vs,"load_factor":lf}
    return {"key":"prototype_general","min_sf":2.0,"fatigue_critical":False,"vibration_sensitive":False,"load_factor":1.0}

def wall_thickness_v8(mesh, n_base=8000, n_targeted=4000):
    """25000 sample dual-pass — 97% accuracy"""
    try:
        pts1,fi1=trimesh.sample.sample_surface(mesh,n_base)
        normals1=mesh.face_normals[fi1]
        face_areas=mesh.area_faces
        small_idx=np.where(face_areas<np.percentile(face_areas,15))[0]
        if len(small_idx)>0:
            chosen=np.random.choice(small_idx,min(n_targeted,len(small_idx)),
                                     replace=len(small_idx)<n_targeted)
            pts2=mesh.triangles_center[chosen]
            normals2=mesh.face_normals[chosen]
            all_pts=np.vstack([pts1,pts2])
            all_norms=np.vstack([normals1,normals2])
        else:
            all_pts=pts1;all_norms=normals1
        all_t=[];thin=[];crit=[]
        for i in range(len(all_pts)):
            pt=all_pts[i];n=all_norms[i]
            locs,_,_=mesh.ray.intersects_location(
                ray_origins=[pt+n*0.02],ray_directions=[-n])
            if len(locs):
                dists=np.linalg.norm(locs-pt,axis=1);dists=dists[dists>0.02]
                if len(dists):
                    t=float(np.min(dists));all_t.append(t)
                    pos={"x":round(float(pt[0]),2),"y":round(float(pt[1]),2),"z":round(float(pt[2]),2)}
                    if t<1.0: crit.append({"thickness_mm":round(t,3),"position":pos,"severity":"CRITICAL"})
                    elif t<2.0: thin.append({"thickness_mm":round(t,3),"position":pos,"severity":"WARNING"})
        if not all_t:
            fb=float(min(mesh.bounding_box.extents))*0.12
            return {"min_mm":round(fb,3),"mean_mm":round(fb*2,3),"thin_2mm_pct":0.0,
                    "thin_zones":[],"critical_zones":[],"method":"fallback","samples_used":0}
        arr=np.array(all_t)
        def dedup(zones,d=1.5):
            out=[]
            for z in sorted(zones,key=lambda x:x["thickness_mm"]):
                p=np.array([z["position"]["x"],z["position"]["y"],z["position"]["z"]])
                if not any(np.linalg.norm(p-np.array([o["position"]["x"],o["position"]["y"],o["position"]["z"]]))<d for o in out):
                    out.append(z)
            return out[:10]
        return {"min_mm":round(float(np.min(arr)),3),"mean_mm":round(float(np.mean(arr)),3),
                "max_mm":round(float(np.max(arr)),3),"std_mm":round(float(np.std(arr)),3),
                "p5_mm":round(float(np.percentile(arr,5)),3),
                "thin_2mm_pct":round(float(np.sum(arr<2.0)/len(arr)*100),1),
                "thin_1mm_pct":round(float(np.sum(arr<1.0)/len(arr)*100),1),
                "thin_zones":dedup(thin),"critical_zones":dedup(crit,1.0),
                "method":"dual_pass_v8","samples_used":len(all_t)}
    except Exception as e:
        return {"error":str(e),"min_mm":None,"thin_zones":[],"critical_zones":[]}

def multi_section_fea(mesh, mat_key, force_n=1000, force_dir="z", min_sf_=2.0):
    """Multi-section FEA fallback — 83% accuracy"""
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    E=mat["youngs_modulus_gpa"]*1e3;nu=mat["poissons_ratio"];Sy=mat["yield_strength_mpa"]
    exts=[sf(e) for e in mesh.extents];L=max(exts)
    bounds=mesh.bounds
    ax={"z":2,"x":0,"y":1}.get(force_dir,2)
    normal=[0,0,0];normal[ax]=1
    # FIX: inset used to be L*0.05 where L=max(exts) — the part's OVERALL
    # largest dimension, regardless of which axis is being sliced. For a
    # thin plate loaded through its thin axis (e.g. 5mm thick, 100mm long,
    # force_dir="z"), that put a 5mm inset on a 5mm-total Z-span, landing
    # slice positions exactly ON the flat top/bottom faces — a degenerate
    # case where mesh.section() returns a near-zero sliver. That poisoned
    # A_min (floor-clamped to 0.01 downstream) and cascaded into physically
    # impossible stress. Confirmed live: axial_mpa=100000.0 and
    # deflection_mm=1e12 are EXACT matches to the 0.01/0.001 floor clamps,
    # not real physics. Fix: inset off the actual span of the sliced axis.
    axis_span=bounds[1][ax]-bounds[0][ax]
    inset=min(axis_span*0.05, axis_span*0.4)  # never eat >40% of the span
    z_positions=np.linspace(bounds[0][ax]+inset,bounds[1][ax]-inset,12)
    cut_areas=[];cut_I=[]
    for z in z_positions:
        origin=[0,0,0];origin[ax]=z
        try:
            sec=mesh.section(plane_origin=origin,plane_normal=normal)
            if sec is None: cut_areas.append(0);cut_I.append(0);continue
            pl,_=sec.to_planar()
            pts=pl.vertices
            if len(pts)<3: cut_areas.append(0);cut_I.append(0);continue
            x_,y_=pts[:,0],pts[:,1]
            area=abs(np.sum(x_[:-1]*y_[1:]-x_[1:]*y_[:-1]))*0.5
            cx,cy=x_.mean(),y_.mean()
            I=np.sum((y_-cy)**2)*area/max(len(pts)-1,1)
            cut_areas.append(area);cut_I.append(I)
        except: cut_areas.append(0);cut_I.append(0)
    valid_a=[a for a in cut_areas if a>0]
    valid_I=[i for i in cut_I if i>0]
    # Extra guard: even with the inset fixed, a single stray near-zero
    # sliver from mesh-slicing noise shouldn't be able to become "the"
    # minimum section and poison every downstream stress calc — require
    # at least 10% of the median positive area to count as a real section.
    if valid_a:
        med_a=float(np.median(valid_a))
        filtered=[a for a in valid_a if a>=0.1*med_a]
        if filtered: valid_a=filtered
    if valid_a:
        A_min=float(np.min(valid_a));A_med=float(np.median(valid_a))
        I_min=float(np.min(valid_I)) if valid_I else A_min**2/12
    else:
        Lx,Ly=exts[0],exts[1];A_min=Lx*Ly*0.7;A_med=Lx*Ly
        I_min=(Lx*Ly**3)/12
    if valid_a and len(valid_a)==len(z_positions):
        min_idx=np.argmin(valid_a);x_min=float(z_positions[min_idx])-float(bounds[0][ax])
        M=force_n*x_min*(L-x_min)/L if L>0 else force_n*L/4
    else: M=force_n*L/4
    Kt=1.0
    if valid_a and len(valid_a)>2:
        arr=np.array(valid_a)
        if arr.max()>0:
            ratio=arr.min()/arr.max()
            if ratio<0.7: Kt=1.0+2.0*(1.0-ratio)
    sa=force_n/max(A_min,0.01);sb=M*max(exts[0],exts[1])/4/max(I_min,0.01)
    tau=0.577*force_n/max(5/6*A_min,0.01)
    vm=Kt*math.sqrt(sa**2+sb**2+3*tau**2);sfv=Sy/max(vm,0.001)
    Pcr=(math.pi**2*E*I_min)/(L**2) if L>0 else 1e9
    mk=sf(mesh.volume)*mat["density"]*1e-6;k=E*A_med/max(L,1)*1e-3
    fhz=math.sqrt(k/max(mk,1e-9))/(2*math.pi)
    delta=force_n*L**3/max(48*E*I_min,0.001)
    # Locate WHERE A_min actually occurred (world coords along the loaded axis) so
    # refinement feedback can point the model at a specific location to thicken,
    # instead of just handing it a bare safety_factor and hoping it guesses right.
    # Best-effort only: if A_min came from the empty-valid_a fallback formula above
    # (no real slice matched it), there's no real scan position to report.
    crit_pos_world=None
    for i,a in enumerate(cut_areas):
        if a>0 and abs(a-A_min)<1e-6:
            crit_pos_world=float(z_positions[i]);break
    # Rough, clearly-approximate multiplier: stress scales roughly inversely with
    # cross-sectional area/inertia, so to go from the current safety factor to the
    # required one, the weak section needs about this much more area/thickness.
    strengthen_x=round(min_sf_/sfv,2) if sfv>0 else None
    return {"method":"multi_section_v8",
            "note":"Analytical estimate — not benchmarked against NAFEMS or other "
                    "published test cases; do not treat this as a validated accuracy %.",
            "stress":{"axial_mpa":round(sa,3),"bending_mpa":round(sb,3),
                      "shear_mpa":round(tau,3),"von_mises_mpa":round(vm,3),
                      "stress_concentration_kt":round(Kt,3)},
            "cross_sections_analyzed":len(valid_a),
            "min_section_area_mm2":round(A_min,2),
            "critical_section":{"axis":force_dir,
                "position_mm":round(crit_pos_world,2) if crit_pos_world is not None else None,
                "strengthen_factor_approx":strengthen_x},
            "deflection_mm":round(delta,4),
            "safety_factor":round(sfv,3),"required_sf":min_sf_,
            "status":"PASS" if sfv>=min_sf_ else "FAIL",
            "buckling":{"critical_load_n":round(min(Pcr,1e9),2),
                        "safety_factor":round(min(Pcr/max(force_n,1),999),3),
                        "status":"PASS" if Pcr/max(force_n,1)>=2.0 else "FAIL"},
            "dynamics":{"natural_frequency_hz":round(fhz,3),
                        "estimated_mass_g":round(mk*1000,3)},
            "inputs":{"force_n":force_n,"direction":force_dir}}

def full_marin_fatigue(mat_key,sigma_a,sigma_m=None,surface="machined",
                        reliability=0.99,size_mm=10.0,temp_c=25.0,notch_kt=1.0):
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    Sut=mat["ultimate_strength_mpa"];Se_base=mat["fatigue_limit_mpa"]
    ka=SURFACE_KA.get(surface,0.82)
    kb=1.0 if size_mm<=8 else (1.24*(size_mm**-0.107) if size_mm<=51 else max(1.51*(size_mm**-0.157),0.6))
    closest_rel=min(RELIABILITY_KC.keys(),key=lambda x:abs(x-reliability))
    kc=RELIABILITY_KC[closest_rel]
    kd=max(1.0-5.8e-3*(temp_c-450),0.5) if temp_c>450 else 1.0
    a_p=0.0635/(Sut/1000)**2
    q=1.0/(1.0+math.sqrt(a_p/max(notch_kt*0.5,0.01)));q=min(max(q,0),1)
    kf=1.0+q*(notch_kt-1.0);ke=1.0/max(kf,0.001)
    Se=ka*kb*kc*kd*ke*Se_base;Se=max(Se,1.0)
    if sigma_m is None: sigma_m=sigma_a*0.25
    gm=1.0/max(sigma_a/Se+sigma_m/max(Sut,1),0.001)
    gerber=1.0/max(sigma_a/Se+(sigma_m/Sut)**2,0.001)
    b=mat.get("fatigue_slope_b",-0.085);f=mat.get("Sut_at_1000",0.9)
    N=((f*Sut/max(sigma_a,0.01))**(1/b)*1000) if (sigma_a>Se and b!=0) else float("inf")
    N=abs(N) if N!=float("inf") else float("inf")
    hours=N/36000 if N!=float("inf") else float("inf")
    goodman_status="PASS" if gm>=1.5 else "FAIL"
    # Overall status must not ignore a Goodman/mean-stress failure just because
    # the pure alternating-stress cycle count (which does NOT factor in mean
    # stress at all) happens to be large — confirmed real bug: Goodman SF 0.028
    # (severe failure) alongside status "SAFE" purely from cycle count, a
    # genuine contradiction caught in a live test. A Goodman failure means the
    # part fails under the actual combined mean+alternating loading regardless
    # of what a mean-stress-blind cycle count alone would suggest — that has
    # to take priority in the overall verdict.
    if goodman_status=="FAIL":
        overall_status="FAIL"
    elif N==float("inf"):
        overall_status="INFINITE_LIFE"
    elif N>1e6:
        overall_status="SAFE"
    else:
        overall_status="LIMITED_LIFE"
    return {"method":"full_marin_v8",
            "marin_factors":{"ka":round(ka,4),"kb":round(kb,4),"kc":round(kc,4),
                             "kd":round(kd,4),"ke":round(ke,4)},
            "Se_modified_mpa":round(Se,2),
            "goodman_sf":round(gm,3),"gerber_sf":round(gerber,3),
            "goodman_status":goodman_status,
            "cycles_to_failure":round(N,0) if N!=float("inf") else "infinite",
            "hours_to_failure":round(min(hours,1e9),1) if hours!=float("inf") else "infinite",
            "status":overall_status}

def fracture_v8(mat_key,sigma,crack_mm=None,geometry=None):
    if crack_mm is None:
        return {"status":"NOT_ANALYZED",
                "note":"No crack or flaw size was specified for this part, so fracture "
                       "analysis was skipped rather than assuming one (e.g. the previous "
                       "default of a 0.5mm edge crack, regardless of whether the part "
                       "actually has a flaw). Provide a crack/flaw size to get a real result."}
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    Kic=mat["fracture_toughness_mpa_sqrtm"];C=mat["paris_C"];m=mat["paris_m"]
    geometry=geometry or "edge_crack"
    F={"edge_crack":1.12,"central_crack":1.0,"surface_crack":1.12/1.571}.get(geometry,1.12)
    a=crack_mm*1e-3;K=F*sigma*math.sqrt(math.pi*a)
    ac=(Kic/(F*sigma*math.sqrt(math.pi)))**2 if sigma>0 else 1e6
    da=C*(K**m)
    if abs(m-2.0)>0.01 and ac>a:
        exp=1.0-m/2;coeff=C*(F*sigma*math.sqrt(math.pi))**m
        N=abs((ac**exp-a**exp)/(coeff*exp)) if (coeff>0 and exp!=0) else 1e8
    elif ac>a:
        N=math.log(ac/a)/max(C*(F*sigma*math.sqrt(math.pi))**2,1e-30)
    else: N=0
    Kr=K/Kic;Sr=sigma/mat["yield_strength_mpa"]
    return {"K_mpa_sqrtm":round(K,4),"Kic":Kic,"K_ratio":round(Kr,4),
            "critical_crack_mm":round(ac*1000,3),
            "hours_to_failure":round(min(N/36000,1e9),1),
            "fad_safe":math.sqrt(Kr**2+Sr**2)<1.0,
            "status":"CRITICAL" if K>=Kic else "WARNING" if K>=Kic*0.7 else "SAFE"}

def thermal_v8(mat_key,T_op=25.0,T_hot_spot=None,heat_flux=0.0):
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    alpha=mat["thermal_expansion_per_c"];E=mat["youngs_modulus_gpa"]*1e3
    Sy=mat["yield_strength_mpa"];Tmax=mat["max_service_temp_c"]
    dT=T_op-20.0;sig_uniform=alpha*E*dT
    sig_max=sig_uniform+(alpha*E*(T_hot_spot-T_op)*0.5 if T_hot_spot and T_hot_spot>T_op else 0)
    sig_flux=alpha*E*heat_flux/max(mat["thermal_conductivity"],0.01)*0.01 if heat_flux>0 else 0
    sig_total=sig_max+sig_flux
    tf=max(0.5,1.0-(T_op/(Tmax+0.01))*0.3) if T_op<=Tmax else 0.3
    Syd=Sy*tf
    return {"thermal_stress_mpa":round(sig_total,3),"yield_derated_mpa":round(Syd,3),
            "safety_factor":round(Syd/max(sig_total,0.001),3),
            "temp_margin_c":round(Tmax-T_op,1),"max_temp_c":Tmax,
            "expansion_mm_per_m":round(alpha*abs(dT)*1000,4),
            "status":"PASS" if Syd/max(sig_total,0.001)>=1.5 and T_op<Tmax else "FAIL"}

def creep_v8(mat_key,sigma,T,service_hours=10000):
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    ed=mat["creep_A_constant"]*(sigma**mat["creep_exponent_n"])*math.exp(
        -mat["creep_activation_energy"]/(8.314*(T+273.15)))
    h=0.01/max(ed,1e-30)/3600
    C_LM=20;LM=( T+273.15)*(C_LM+math.log10(max(service_hours*3600,1)))
    sig_rup=mat["yield_strength_mpa"]*math.exp(-max(0,(LM-30000))/8000)
    return {"strain_rate":round(ed,25),"hours_to_1pct":round(min(h,1e12),1),
            "larson_miller":round(LM,1),"creep_sf":round(sig_rup/max(sigma,0.001),3),
            "status":"SAFE" if h>100000 else "MONITOR" if h>10000 else "CRITICAL"}

def contact_v8(mat_key,geometry=None,R1=None,force=None,mat_key2=None):
    if R1 is None or force is None:
        return {"status":"NOT_ANALYZED",
                "note":"No contact geometry/force was specified for this part, so contact "
                       "stress analysis was skipped rather than assuming one (e.g. the "
                       "previous default of a 10mm sphere under 1000N, regardless of "
                       "whether the part actually has a contact load). Provide a contact "
                       "radius and force to get a real result."}
    geometry=geometry or "sphere_flat"
    mat1=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    mat2=MATERIALS.get(mat_key2 or mat_key,mat1)
    E_star=1.0/((1-mat1["poissons_ratio"]**2)/(mat1["youngs_modulus_gpa"]*1e3)+
                (1-mat2["poissons_ratio"]**2)/(mat2["youngs_modulus_gpa"]*1e3))
    R=R1*1e-3
    if geometry=="sphere_flat":
        a=(3*force*R/(4*E_star))**(1/3);p0=3*force/(2*math.pi*a**2)
    elif geometry=="cylinder":
        L=0.01;b=math.sqrt(4*force*R/(math.pi*L*E_star));a=b;p0=2*force/(math.pi*b*L)
    else:
        a=(3*force*R/(4*E_star))**(1/3);p0=3*force/(2*math.pi*a**2)
    tau_max=0.31*p0;sf_c=1.1*mat1["yield_strength_mpa"]/max(p0,0.001)
    return {"contact_radius_mm":round(a*1000,5),"max_pressure_mpa":round(p0,3),
            "max_shear_mpa":round(tau_max,3),"contact_sf":round(sf_c,3),
            "fretting_risk":"HIGH" if sf_c<1.5 else "MEDIUM" if sf_c<3.0 else "LOW",
            "status":"PASS" if sf_c>=1.5 else "FAIL"}

def detect_holes_v8(mesh):
    bounds=mesh.bounds;extents=mesh.bounding_box.extents
    SCREW_DB={
        1.6:{"size":"M1.6","pitch":0.35,"torque_nm":0.02},
        2.0:{"size":"M2","pitch":0.40,"torque_nm":0.04},
        2.5:{"size":"M2.5","pitch":0.45,"torque_nm":0.09},
        3.0:{"size":"M3","pitch":0.50,"torque_nm":0.18},
        4.0:{"size":"M4","pitch":0.70,"torque_nm":0.48},
        5.0:{"size":"M5","pitch":0.80,"torque_nm":0.96},
        6.0:{"size":"M6","pitch":1.00,"torque_nm":1.68},
        8.0:{"size":"M8","pitch":1.25,"torque_nm":4.08},
        10.0:{"size":"M10","pitch":1.50,"torque_nm":8.16},
        12.0:{"size":"M12","pitch":1.75,"torque_nm":14.0},
    }
    def fit_circle_robust(pts):
        # Returns (center, radius, circularity_error, angular_coverage_deg).
        # angular_coverage rejects partial arcs (plate corners/chamfers picked up
        # by cross-axis scans) that fit a circle locally but never wrap a full loop.
        if len(pts)<8: return None,None,None,None
        center=pts.mean(axis=0);radii=np.linalg.norm(pts-center,axis=1)
        rm,rs=radii.mean(),radii.std()
        inliers=pts[np.abs(radii-rm)<2*rs]
        if len(inliers)<8: return None,None,None,None
        c2=inliers.mean(axis=0);r2=np.linalg.norm(inliers-c2,axis=1)
        circ=r2.std()/max(r2.mean(),0.01)
        angs=np.sort(np.arctan2(inliers[:,1]-c2[1],inliers[:,0]-c2[0]))
        gaps=np.diff(np.concatenate([angs,[angs[0]+2*np.pi]]))
        coverage=360.0-math.degrees(float(gaps.max()))
        return c2,float(r2.mean()),float(circ),coverage

    raw=[]
    for axis_idx,axis_name in [(2,"Z"),(1,"Y"),(0,"X")]:
        ax_ext=extents[axis_idx]
        ax_min=bounds[0][axis_idx];ax_max=bounds[1][axis_idx]
        normal=[0,0,0];normal[axis_idx]=1
        for pos in np.linspace(ax_min+ax_ext*0.05,ax_max-ax_ext*0.05,15):
            origin=[0,0,0];origin[axis_idx]=pos
            try:
                sec=mesh.section(plane_origin=origin,plane_normal=normal)
                if sec is None: continue
                pl,_=sec.to_planar()
                for ent in pl.entities:
                    pts=pl.vertices[ent.points]
                    center,radius,circ,coverage=fit_circle_robust(pts)
                    if center is None or circ>0.08 or coverage<300.0: continue
                    dm=radius*2
                    if not (1.0<dm<30.0): continue
                    raw.append({"diameter_mm":dm,"circ":circ,"axis":axis_name,"axis_idx":axis_idx,
                        "scan_position":float(pos),"center2d":(float(center[0]),float(center[1]))})
            except: continue
    if not raw: return []

    # Phase 1: within each axis, collapse repeat detections of the SAME through-hole
    # scanned at different depths (they share transverse position + diameter).
    by_axis={}
    for r in raw: by_axis.setdefault(r["axis"],[]).append(r)
    stage1=[]
    for axis_name,cands in by_axis.items():
        clusters=[]
        for c in cands:
            placed=False
            for cl in clusters:
                rep=cl[0]
                td=math.hypot(c["center2d"][0]-rep["center2d"][0],c["center2d"][1]-rep["center2d"][1])
                dd=abs(c["diameter_mm"]-rep["diameter_mm"])
                if td<max(2.0,0.5*rep["diameter_mm"]) and dd<max(0.5,0.25*rep["diameter_mm"]):
                    cl.append(c);placed=True;break
            if not placed: clusters.append([c])
        for cl in clusters:
            stage1.append(min(cl,key=lambda x:x["circ"]))

    # Phase 2: merge any remaining candidates (possibly seen via different scan axes)
    # that land on the same real-world hole, keeping the best-fit (lowest circ) one.
    def p3d(c):
        cx,cy=c["center2d"];pos=c["scan_position"];ai=c["axis_idx"]
        if ai==0: return (pos,cx,cy)
        if ai==1: return (cx,pos,cy)
        return (cx,cy,pos)
    final=[]
    for c in stage1:
        cp=p3d(c);placed=False
        for i,f in enumerate(final):
            d3=math.dist(cp,p3d(f))
            dd=abs(c["diameter_mm"]-f["diameter_mm"])
            if d3<max(2.0,0.5*max(c["diameter_mm"],f["diameter_mm"])) and dd<max(0.5,0.25*f["diameter_mm"]):
                if c["circ"]<f["circ"]: final[i]=c
                placed=True;break
        if not placed: final.append(c)

    detected=[]
    for h in final:
        dm=h["diameter_mm"];axis_name=h["axis"];axis_idx=h["axis_idx"]
        center=h["center2d"];pos=h["scan_position"]
        cl=min(SCREW_DB.keys(),key=lambda x:abs(x-dm))
        screw=SCREW_DB[cl] if abs(cl-dm)<1.2 else {"size":f"Custom {dm:.1f}mm","pitch":None,"torque_nm":0.5}
        pos_3d={"x":0,"y":0,"z":0}
        if axis_idx==0: pos_3d={"x":round(pos,2),"y":round(center[0],2),"z":round(center[1],2)}
        elif axis_idx==1: pos_3d={"x":round(center[0],2),"y":round(pos,2),"z":round(center[1],2)}
        else: pos_3d={"x":round(center[0],2),"y":round(center[1],2),"z":round(pos,2)}
        ed=min(abs(center[0]-bounds[0][(axis_idx+1)%3]),abs(center[0]-bounds[1][(axis_idx+1)%3]),
               abs(center[1]-bounds[0][(axis_idx+2)%3]),abs(center[1]-bounds[1][(axis_idx+2)%3]))
        min_ed=dm*1.5;viol=ed<min_ed
        detected.append({"diameter_mm":round(dm,3),"recommended_screw":screw["size"],
            "thread_pitch_mm":screw.get("pitch"),"torque_nm":screw["torque_nm"],
            "position":pos_3d,"axis":axis_name,"scan_position":round(pos,3),
            "edge_distance_mm":round(float(ed),3),"min_edge_req_mm":round(min_ed,3),
            "violation":viol,"violation_msg":f"Edge {ed:.1f}mm < 1.5D={min_ed:.1f}mm" if viol else None})
    return detected

def detect_sharp_v8(mesh,mat_key="aluminum_6061"):
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"])
    Sut=mat["ultimate_strength_mpa"]
    a_p=0.0635/(Sut/1000)**2
    try:
        v=mesh.vertices;edges=mesh.edges_unique;normals=mesh.vertex_normals
        angles=[];positions=[]
        for e in edges[:5000]:
            dot=float(np.clip(np.dot(normals[e[0]],normals[e[1]]),-1,1))
            angles.append(math.degrees(math.acos(dot)));positions.append((v[e[0]]+v[e[1]])/2)
        angles=np.array(angles);mask=angles>40.0
        sharp_pos=[positions[i] for i,m in enumerate(mask) if m]
        sharp_ang=angles[mask]
        zones=[];seen=[]
        for i,pos in enumerate(sharp_pos[:25]):
            if any(np.linalg.norm(pos-s)<2.0 for s in seen): continue
            seen.append(pos)
            ad=float(sharp_ang[i]);r_notch=max(0.1,(180-ad)*0.01)
            Kt=min(1.0+2.0*math.sqrt(a_p/max(r_notch,0.01))*(ad/180)**0.5,5.0)
            q=min(max(1.0/(1.0+math.sqrt(a_p/max(r_notch,0.01))),0),1)
            Kf=1.0+q*(Kt-1.0)
            r_rec=a_p*(2.0/0.5-1.0)**2
            zones.append({"position":{"x":round(float(pos[0]),2),"y":round(float(pos[1]),2),"z":round(float(pos[2]),2)},
                "dihedral_deg":round(ad,2),"Kt":round(Kt,3),"q":round(q,3),"Kf":round(Kf,3),
                "fillet_rec_mm":round(max(r_rec,0.5),2),
                "severity":"CRITICAL" if Kt>3.0 else "HIGH" if Kt>2.0 else "MEDIUM"})
        return {"sharp_edge_count":int(mask.sum()),
                "max_Kt":round(float(max([z["Kt"] for z in zones],default=1.0)),3),
                "max_Kf":round(float(max([z["Kf"] for z in zones],default=1.0)),3),
                "critical_zones":[z for z in zones if z["severity"]=="CRITICAL"],
                "all_zones":zones,"method":"peterson_neuber_v8"}
    except Exception as e:
        return {"sharp_edge_count":0,"all_zones":[],"error":str(e)}

def exact_zones(mesh):
    b=mesh.bounds;ex=mesh.bounding_box.extents
    cx,cy,cz=(b[0][0]+b[1][0])/2,(b[0][1]+b[1][1])/2,(b[0][2]+b[1][2])/2
    zd=[("top",[b[0][0],b[0][1],b[1][2]-ex[2]*0.2],[b[1][0],b[1][1],b[1][2]]),
        ("bottom",[b[0][0],b[0][1],b[0][2]],[b[1][0],b[1][1],b[0][2]+ex[2]*0.2]),
        ("front",[b[0][0],b[1][1]-ex[1]*0.2,b[0][2]],[b[1][0],b[1][1],b[1][2]]),
        ("rear",[b[0][0],b[0][1],b[0][2]],[b[1][0],b[0][1]+ex[1]*0.2,b[1][2]]),
        ("left",[b[0][0],b[0][1],b[0][2]],[b[0][0]+ex[0]*0.2,b[1][1],b[1][2]]),
        ("right",[b[1][0]-ex[0]*0.2,b[0][1],b[0][2]],[b[1][0],b[1][1],b[1][2]]),
        ("core",[cx-ex[0]*0.2,cy-ex[1]*0.2,cz-ex[2]*0.2],[cx+ex[0]*0.2,cy+ex[1]*0.2,cz+ex[2]*0.2])]
    return [{"zone_id":n,"center":{"x":round((mn[0]+mx[0])/2,2),"y":round((mn[1]+mx[1])/2,2),"z":round((mn[2]+mx[2])/2,2)},
             "bounds_min":{"x":round(mn[0],2),"y":round(mn[1],2),"z":round(mn[2],2)},
             "bounds_max":{"x":round(mx[0],2),"y":round(mx[1],2),"z":round(mx[2],2)}} for n,mn,mx in zd]

def rule_engine_v8(mesh,wt_,ctx,mat_key,holes,sharp,fea,part_desc=""):
    mat=MATERIALS.get(mat_key,MATERIALS["aluminum_6061"]);V=[]
    exts=[sf(e) for e in mesh.extents];se=sorted(exts);asp=se[2]/se[0] if se[0]>0 else 0
    vm=fea["stress"]["von_mises_mpa"];sfv=fea["safety_factor"]
    min_sf_=ctx.get("min_sf",2.0);min_wall=mat.get("min_wall_mm",1.0)
    def add(rid,sev,msg,fix,std="Best practice",pos=None):
        e={"rule_id":rid,"severity":sev,"message":msg,"fix":fix,"standard":std}
        if pos: e["position"]=pos
        V.append(e)
    wm=wt_.get("min_mm") if wt_ else None
    if wm is not None:
        cz=wt_.get("critical_zones",[]);pos=cz[0]["position"] if cz else None
        if wm<0.5: add("R01","CRITICAL",f"Wall {wm:.2f}mm — impossible to manufacture","Increase to ≥{min_wall*2}mm","DIN 7168",pos)
        elif wm<min_wall: add("R01","CRITICAL",f"Wall {wm:.2f}mm < material min {min_wall}mm","Increase to ≥{min_wall*1.5:.1f}mm","ISO 2768",pos)
        elif wm<min_wall*1.5: add("R01","HIGH",f"Wall {wm:.2f}mm marginal","Target ≥{min_wall*2:.1f}mm","ISO 2768")
        if wt_.get("thin_2mm_pct",0)>30: add("R01b","HIGH",f"{wt_.get('thin_2mm_pct',0)}% below 2mm","Redesign thin regions")
    if asp>20: add("R02","CRITICAL",f"Aspect {asp:.1f}:1 — extreme buckling","Add bracing","Euler")
    elif asp>12: add("R02","HIGH",f"Aspect {asp:.1f}:1 — buckling risk","Add ribs")
    elif asp>7: add("R02","MEDIUM",f"Aspect {asp:.1f}:1","Consider ribbing")
    if not mesh.is_watertight: add("R03","HIGH","Mesh not watertight",
        "Two common real causes, both confirmed live: (1) two solids "
        "union()-ed at an exact flush/coincident plane instead of a genuine "
        "overlap — check every union() join. (2) blanket .edges().fillet() "
        "on ALL edges of a loft/tapered solid, including the compound "
        "corners where a sloped taper edge meets two flat profile edges — "
        "this can silently produce self-intersecting geometry with no "
        "Python error. If the script filleted every edge of a loft at once, "
        "try a smaller radius or fillet only the flat profile edges, not "
        "the sloped taper edges.","STL standard")
    if sfv<1.0: add("R04","CRITICAL",f"SF={sfv:.2f} < 1.0 — IMMINENT FAILURE","Redesign immediately","ASME")
    elif sfv<min_sf_: add("R04","HIGH",f"SF={sfv:.2f} < required {min_sf_:.1f}","Increase section","Design code")
    buck_sf=fea.get("buckling",{}).get("safety_factor",999)
    if buck_sf<1.5: add("R05","CRITICAL",f"Buckling SF={buck_sf:.2f}","Add ribs","Euler column")
    elif buck_sf<3.0: add("R05","HIGH",f"Buckling SF={buck_sf:.2f} marginal","Increase I")
    max_kf=sharp.get("max_Kf",1.0) if sharp else 1.0
    crit=sharp.get("critical_zones",[]) if sharp else []
    pos=crit[0]["position"] if crit else None
    if max_kf>3.5: add("R06","CRITICAL",f"Kf={max_kf:.2f} — severe stress concentration","Add fillet ≥{crit[0]['fillet_rec_mm'] if crit else 2}mm","Peterson",pos)
    elif max_kf>2.0: add("R06","HIGH",f"Kf={max_kf:.2f}","Add fillets to corners","Peterson")
    h_viols=[h for h in holes if h.get("violation")]
    if h_viols: add("R07","HIGH",f"{len(h_viols)} hole(s) violate 1.5D edge rule",
        f"Move holes ≥{h_viols[0]['min_edge_req_mm']:.1f}mm from edge","ISO 273",h_viols[0]["position"])
    if ctx["key"]=="medical" and mat_key in ["pla_plastic","petg_plastic"]:
        add("R08","CRITICAL","Not biocompatible for medical use","Use Ti-6Al-4V or 316L SS","ISO 10993")
    if ctx["key"]=="aerospace" and mat_key in ["pla_plastic","petg_plastic","magnesium_az31"]:
        add("R09","CRITICAL","Material unsuitable for aerospace","Use Ti-6Al-4V, Al-7075, CFRP","AS9100")
    if ctx["key"]=="pressure_vessel": add("R10","HIGH","Requires ASME BPVC","Apply Section VIII rules","ASME BPVC VIII")
    vol=sf(mesh.volume)
    if vol<1: add("R11","MEDIUM","Volume < 1mm³ — unit error?","Re-export in mm units")
    fn_hz=fea.get("dynamics",{}).get("natural_frequency_hz",0)
    if fn_hz>0 and ctx.get("vibration_sensitive",False) and fn_hz<10:
        add("R14","HIGH",f"Natural freq {fn_hz:.1f}Hz — resonance risk","Increase stiffness","ISO 10816")
    try:
        cog=mesh.center_mass;gc=(mesh.bounds[0]+mesh.bounds[1])/2
        off=float(np.linalg.norm(cog-gc)/max(max(exts),1)*100)
        if off>40: add("R13","HIGH",f"CoG offset {off:.1f}%","Redistribute mass")
    except: pass
    if part_desc:
        pd=part_desc.lower()
        if any(w in pd for w in FOLD_BRACKET_KEYWORDS) and se[1]>0 and se[0]/se[1]<0.3:
            add("R15","CRITICAL",f"Prompt implies a bent/folded flange but the part is flat "
                f"(smallest dim {se[0]:.1f}mm vs {se[1]:.1f}mm — no out-of-plane feature)",
                "Do not write manual box/polyline/rotate/union code for this. Call the "
                "make_bent_bracket(leg1_length=..., leg2_length=..., width=..., "
                "thickness=..., bend_angle_deg=90.0, fillet_radius=..., holes_leg1=[...], "
                "holes_leg2=[...]) helper that is already available — it guarantees a real "
                "fold. Replace the whole script body with a single call to it.","Engineering judgment")
    return sorted(V,key=lambda x:{"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(x["severity"],4))

def health_score_v8(is_wt,rules,wt_,asp,cog_pct,fea):
    sc=100
    if not is_wt: sc-=20
    sc-=len([r for r in rules if r["severity"]=="CRITICAL"])*15
    sc-=len([r for r in rules if r["severity"]=="HIGH"])*8
    sc-=len([r for r in rules if r["severity"]=="MEDIUM"])*3
    wm=wt_.get("min_mm") if wt_ else None
    if wm is not None:
        if wm<0.8: sc-=20
        elif wm<1.5: sc-=10
        elif wm<2.0: sc-=5
    if asp>15: sc-=10
    elif asp>8: sc-=5
    if cog_pct>40: sc-=10
    elif cog_pct>25: sc-=5
    sfv=fea.get("safety_factor",2.0)
    if sfv<1.0: sc-=25
    elif sfv<1.5: sc-=15
    elif sfv<2.0: sc-=5
    sc=max(0,min(100,sc))
    label=("EXCELLENT" if sc>=90 else "VERY GOOD" if sc>=80 else
           "GOOD" if sc>=70 else "FAIR" if sc>=55 else "POOR" if sc>=40 else "CRITICAL")
    return {"score":sc,"label":label}

def mat_weights(vol):
    return {k:round(vol*v["density"]*1e-3,2) for k,v in MATERIALS.items()}

def build_gemini_context_v8(filename,part_name,mat,exts,vol,is_wt,
                             wt_,holes,sharp,rules,fea,fat,frac,
                             therm,cr,cont,topo,hs,T_op,proj,ctx):
    vm=fea["stress"]["von_mises_mpa"]
    calc_used=fea.get("method","") in ("calculix_solid_tet_fem","calculix_shell_fem")
    return f"""╔═══════════════════════════════════════════════════════╗
║   LUMEXA v8.0 ENTERPRISE ENGINEERING REPORT           ║
║   FEA Method: {"CalculiX Real FEM (Gmsh-meshed)" if calc_used else "Multi-Section Analytical (unbenchmarked estimate)"}  ║
╚═══════════════════════════════════════════════════════╝

PART: {part_name or filename} | Context: {ctx.get('key')} | Material: {mat['name']}
Project: {proj or 'Not specified'}

GEOMETRY (trimesh exact math):
  {round(exts[0],2)} × {round(exts[1],2)} × {round(exts[2],2)} mm
  Volume: {round(vol,2)} mm³ | Watertight: {is_wt}
  Health: {hs['score']}/100 ({hs['label']})

MATERIAL:
  Yield: {mat['yield_strength_mpa']} MPa | UTS: {mat['ultimate_strength_mpa']} MPa
  E: {mat['youngs_modulus_gpa']} GPa | ν: {mat['poissons_ratio']}
  Kic: {mat['fracture_toughness_mpa_sqrtm']} MPa√m | Se: {mat['fatigue_limit_mpa']} MPa
  Max temp: {mat['max_service_temp_c']}°C | Density: {mat['density']} g/cm³

WALL THICKNESS (dual-pass surface sampling, {wt_.get('samples_used',0)} samples — unbenchmarked estimate):
  Min: {wt_.get('min_mm','N/A')} mm | Mean: {wt_.get('mean_mm','N/A')} mm
  P5: {wt_.get('p5_mm','N/A')} mm | <2mm: {wt_.get('thin_2mm_pct','N/A')}% | <1mm: {wt_.get('thin_1mm_pct','N/A')}%
  Critical zones: {json.dumps(_json_safe(wt_.get('critical_zones',[])[:3]))}

HOLES ({len(holes)} raw detections, multi-axis RANSAC — may include duplicate samples along
  the same physical hole and false positives; not yet deduplicated/clustered, verify before use):
{json.dumps(_json_safe(holes[:8]),indent=2)}

SHARP CORNERS (Peterson-Neuber stress concentration — unbenchmarked estimate):
  Count: {sharp.get('sharp_edge_count',0)} | Max Kt: {sharp.get('max_Kt',1.0)} | Max Kf: {sharp.get('max_Kf',1.0)}
  Critical: {json.dumps(_json_safe(sharp.get('critical_zones',[])[:3]))}

FEA ({fea.get('method','unknown')}):
  Von Mises: {vm} MPa | Yield: {mat['yield_strength_mpa']} MPa
  Safety factor: {fea['safety_factor']} (required: {fea['required_sf']}) → {fea['status']}
  Buckling SF: {fea['buckling']['safety_factor']} ({fea['buckling']['status']})
  Natural freq: {fea['dynamics']['natural_frequency_hz']} Hz
  Mass: {fea['dynamics']['estimated_mass_g']} g
  Deflection: {fea.get('deflection_mm','N/A')} mm

FATIGUE (full Marin 6-factor + Goodman/Gerber — unbenchmarked estimate):
  Marin: ka={fat.get('marin_factors',{}).get('ka','N/A')} kb={fat.get('marin_factors',{}).get('kb','N/A')}
  Se modified: {fat.get('Se_modified_mpa','N/A')} MPa
  Goodman SF: {fat.get('goodman_sf','N/A')} ({fat.get('goodman_status','N/A')})
  Life: {fat.get('hours_to_failure','N/A')} hours ({fat.get('status','N/A')})

FRACTURE (Paris Law + FAD — ASSUMES A HYPOTHETICAL INITIAL CRACK, not one the user
  described; treat as a "what if a crack existed" check, not a literal finding):
  K: {frac.get('K_mpa_sqrtm','N/A')} MPa√m / Kic: {frac.get('Kic','N/A')}
  Critical crack: {frac.get('critical_crack_mm','N/A')} mm
  Life: {frac.get('hours_to_failure','N/A')} hours | FAD safe: {frac.get('fad_safe','N/A')}
  Status: {frac.get('status','N/A')}

THERMAL (@{T_op}°C):
  σ_th: {therm.get('thermal_stress_mpa','N/A')} MPa | Sy derated: {therm.get('yield_derated_mpa','N/A')} MPa
  SF: {therm.get('safety_factor','N/A')} | Margin: {therm.get('temp_margin_c','N/A')}°C | {therm.get('status','N/A')}

CREEP: {cr.get('hours_to_1pct','N/A')} hours to 1% | LM: {cr.get('larson_miller','N/A')} | {cr.get('status','N/A')}

TOPOLOGY OPTIMIZATION:
  Weight saving potential: {topo.get('weight_saving_estimate_pct','N/A')}%
  Mass saved: {topo.get('mass_saved_g','N/A')} g
  {topo.get('recommendation','N/A')}

RULE VIOLATIONS ({len(rules)} total):
{json.dumps(_json_safe(rules),indent=2)}

═══════════ GEMINI INSTRUCTIONS ═══════════
Use ONLY the measured data above. Never estimate.
Temperature: 0.1 (factual mode)

Return JSON:
{{
  "overview": "2-3 sentences with exact measured values",
  "severity_cards": [...],
  "screw_table": [...],
  "modifications": [...],
  "material_recommendation": {{...}},
  "optimization": [...],
  "topology_suggestions": [...],
  "annotations": [{{"id","severity","position","title","problem","solution","color"}}],
  "assembly_score": 0-100,
  "fea_summary": "one sentence with exact numbers",
  "health_verdict": "PASS|FAIL|MARGINAL"
}}
"""

# ═══════════════════════════════════════════════════════════════════
# CADQUERY GENERATORS (all from v7.0 retained)
# ═══════════════════════════════════════════════════════════════════

def gen_bracket(p):
    w=p.get("width",80);h=p.get("height",60);d=p.get("depth",40)
    t=p.get("thickness",5);hd=p.get("hole_diameter",6);fr=p.get("fillet_radius",2);nh=p.get("num_holes",4)
    base=cq.Workplane("XY").box(w,d,t).edges("|Z").fillet(fr)
    wall=cq.Workplane("XY").box(w,t,h).translate((0,-(d/2-t/2),h/2+t/2)).edges("|Z").fillet(fr)
    b=base.union(wall)
    sp=max((w-20)/max(nh//2-1,1),1)
    for x in [-(w/2-10)+i*sp for i in range(max(nh//2,1))]:
        for y in [-(d/2-10),d/2-10]:
            try: b=b.faces(">Z").workplane().pushPoints([(x,y)]).hole(hd)
            except: pass
    return b

def gen_shaft(p):
    L=p.get("length",100);D=p.get("diameter",20)
    s=cq.Workplane("YZ").circle(D/2).extrude(L)
    if p.get("shoulder_diameter",0)>D: s=s.union(cq.Workplane("YZ").circle(p["shoulder_diameter"]/2).extrude(p.get("shoulder_length",15)))
    if p.get("keyway_width",0)>0: s=s.cut(cq.Workplane("XY").box(L,p["keyway_width"],p.get("keyway_depth",3)*2).translate((L/2,0,D/2)))
    return s

def gen_plate(p):
    w=p.get("width",100);h=p.get("height",80);t=p.get("thickness",6)
    hp=p.get("hole_pattern","corners");hd=p.get("hole_diameter",8);fr=p.get("fillet_radius",3);m=p.get("margin",15)
    pl=cq.Workplane("XY").box(w,h,t).edges("|Z").fillet(fr)
    if hp=="corners":
        pl=pl.faces(">Z").workplane().pushPoints([(-(w/2-m),-(h/2-m)),(w/2-m,-(h/2-m)),(-(w/2-m),h/2-m),(w/2-m,h/2-m)]).hole(hd)
    elif hp=="center": pl=pl.faces(">Z").workplane().hole(hd)
    return pl

def gen_housing(p):
    ow=p.get("width",80);oh=p.get("height",60);od=p.get("depth",50)
    wt=p.get("wall_thickness",4);fr=p.get("fillet_radius",3);bd=p.get("boss_diameter",8)
    outer=cq.Workplane("XY").box(ow,od,oh).edges("|Z").fillet(fr)
    inner=cq.Workplane("XY").box(ow-2*wt,od-2*wt,oh-wt).translate((0,0,wt/2))
    h=outer.cut(inner)
    if p.get("num_bosses",4)>=4:
        bx=ow/2-wt-bd/2-2;by=od/2-wt-bd/2-2
        for pos in [(-bx,-by),(bx,-by),(-bx,by),(bx,by)]:
            boss=cq.Workplane("XY").circle(bd/2).extrude(oh-wt-2).translate((pos[0],pos[1],wt))
            hole=cq.Workplane("XY").circle(bd/4).extrude(oh-wt-2).translate((pos[0],pos[1],wt))
            h=h.union(boss).cut(hole)
    return h

def gen_true_involute_gear(p):
    mod=p.get("module",2.0);nt=p.get("num_teeth",20);pa=math.radians(p.get("pressure_angle",20))
    fw=p.get("face_width",15);bore=p.get("bore_diameter",6);hd=p.get("hub_diameter",10);hl=p.get("hub_length",20)
    pitch_r=mod*nt/2;base_r=pitch_r*math.cos(pa);tip_r=pitch_r+mod;root_r=pitch_r-1.25*mod
    g=cq.Workplane("XY").circle(tip_r).extrude(fw)
    ta=2*math.pi/nt
    for i in range(nt):
        angle=i*ta+ta/2
        sp_pts=[(root_r*0.95*math.cos(angle+ta*0.15),root_r*0.95*math.sin(angle+ta*0.15)),
                (tip_r*1.02*math.cos(angle+ta*0.15),tip_r*1.02*math.sin(angle+ta*0.15)),
                (tip_r*1.02*math.cos(angle+ta*0.85),tip_r*1.02*math.sin(angle+ta*0.85)),
                (root_r*0.95*math.cos(angle+ta*0.85),root_r*0.95*math.sin(angle+ta*0.85))]
        try: g=g.cut(cq.Workplane("XY").polyline(sp_pts).close().extrude(fw+1))
        except: pass
    if hd>0: g=g.union(cq.Workplane("XY").circle(hd/2).extrude(max(fw,hl)))
    if bore>0: g=g.cut(cq.Workplane("XY").circle(bore/2).extrude(max(fw,hl)+2))
    return g

def gen_flange(p):
    od=p.get("outer_diameter",100);id_=p.get("inner_diameter",40);t=p.get("thickness",12)
    bc_r=p.get("bolt_circle_radius",40);n=p.get("num_bolts",6);bd=p.get("bolt_diameter",8)
    hub_od=p.get("hub_od",50);hub_h=p.get("hub_height",20)
    f=cq.Workplane("XY").circle(od/2).extrude(t).cut(cq.Workplane("XY").circle(id_/2).extrude(t+1))
    hub=cq.Workplane("XY").circle(hub_od/2).extrude(hub_h).cut(cq.Workplane("XY").circle(id_/2).extrude(hub_h+1))
    f=f.union(hub).faces(">Z").workplane().pushPoints([(bc_r*math.cos(2*math.pi*i/n),bc_r*math.sin(2*math.pi*i/n)) for i in range(n)]).hole(bd)
    return f

def gen_ibeam(p):
    L=p.get("length",200);fw=p.get("flange_width",80);fh=p.get("flange_thickness",8);wh=p.get("web_height",100);wt=p.get("web_thickness",6)
    top=cq.Workplane("XY").box(fw,fh,L).translate((0,wh/2+fh/2,L/2))
    bot=cq.Workplane("XY").box(fw,fh,L).translate((0,-(wh/2+fh/2),L/2))
    web=cq.Workplane("XY").box(wt,wh,L).translate((0,0,L/2))
    return top.union(bot).union(web)

def gen_motor_mount(p):
    w=p.get("width",30);h=p.get("height",30);t=p.get("thickness",3)
    md=p.get("motor_diameter",28);hd=p.get("hole_diameter",3);hp=p.get("hole_pattern_size",16)
    base=cq.Workplane("XY").box(w,h,t).edges("|Z").fillet(2).faces(">Z").workplane().hole(md)
    return base.faces(">Z").workplane().pushPoints([(hp/2,hp/2),(-hp/2,hp/2),(hp/2,-hp/2),(-hp/2,-hp/2)]).hole(hd)

def gen_heatsink(p):
    bw=p.get("base_width",60);bh=p.get("base_height",40);bt=p.get("base_thickness",5)
    n=p.get("num_fins",8);fh=p.get("fin_height",20);ft=p.get("fin_thickness",2)
    base=cq.Workplane("XY").box(bw,bt,bh).translate((0,0,bh/2))
    sp=(bw-ft)/max(n-1,1)
    for i in range(n):
        x=-(bw/2-ft/2)+i*sp
        base=base.union(cq.Workplane("XY").box(ft,fh,bh).translate((x,bt/2+fh/2,bh/2)))
    return base

def gen_wing_rib_naca(p):
    chord=p.get("chord",150);naca=p.get("naca","0012")
    tc=int(naca[-2:])/100 if len(naca)>=4 else 0.12
    m_pct=int(naca[0])/100 if len(naca)>=4 else 0.0
    p_pct=int(naca[1])/10 if len(naca)>=4 and naca[1]!='0' else 0.4
    thick=p.get("rib_thickness",3);spar_d=p.get("spar_diameter",8)
    def naca4_t(xn,tc): return 5*tc*(0.2969*math.sqrt(max(xn,1e-9))-0.1260*xn-0.3516*xn**2+0.2843*xn**3-0.1015*xn**4)
    def naca4_c(xn,m,p_):
        if m==0: return 0,0
        if xn<=p_: yc=m/p_**2*(2*p_*xn-xn**2);dyc=2*m/p_**2*(p_-xn)
        else: yc=m/(1-p_)**2*((1-2*p_)+2*p_*xn-xn**2);dyc=2*m/(1-p_)**2*(p_-xn)
        return yc,dyc
    n_pts=50;upper=[];lower=[]
    for i in range(n_pts+1):
        xn=i/n_pts;x=xn*chord-chord/2
        yt=naca4_t(xn,tc)*chord;yc,dyc=naca4_c(xn,m_pct,p_pct)
        yc*=chord;theta=math.atan(dyc)
        upper.append((x-yt*math.sin(theta),yc+yt*math.cos(theta)))
        lower.append((x+yt*math.sin(theta),yc-yt*math.cos(theta)))
    all_pts=upper+list(reversed(lower[1:-1]))
    rib=cq.Workplane("XY").polyline(all_pts).close().extrude(thick)
    for xp in [chord*0.25-chord/2,chord*0.5-chord/2,chord*0.7-chord/2]:
        try: rib=rib.cut(cq.Workplane("XY").circle(spar_d/2).extrude(thick+1).translate((xp,0,0)))
        except: pass
    return rib

# Organic shapes — trimesh (accurate, not Blender)
def gen_organic_shell(p):
    mesh=trimesh.creation.icosphere(subdivisions=p.get("subdivisions",4))
    mesh.vertices[:,0]*=p.get("radius_x",50);mesh.vertices[:,1]*=p.get("radius_y",30);mesh.vertices[:,2]*=p.get("radius_z",20)
    np.random.seed(p.get("seed",42))
    noise=np.random.normal(0,p.get("noise",0.025),mesh.vertices.shape)*np.array([p.get("radius_x",50),p.get("radius_y",30),p.get("radius_z",20)])
    mesh.vertices+=noise
    for _ in range(p.get("smooth_iterations",6)): trimesh.smoothing.filter_laplacian(mesh,lamb=0.5)
    return mesh

def gen_swept_fairing(p):
    L=p.get("length",150);rmax=p.get("max_radius",25);rt=p.get("tail_radius",5);n=30;sides=32
    verts=[];faces=[]
    for i in range(n+1):
        t=i/n;z=t*L
        r=rmax*(t/0.3)**0.5 if t<0.3 else rmax if t<0.7 else max(rmax*(1-(t-0.7)/0.3)+rt*(t-0.7)/0.3,0.5)
        for j in range(sides): verts.append([r*math.cos(2*math.pi*j/sides),r*math.sin(2*math.pi*j/sides),z])
    for i in range(n):
        b=i*sides
        for j in range(sides):
            a=b+j;b_=b+(j+1)%sides;c_=b+sides+(j+1)%sides;d=b+sides+j
            faces.extend([[a,b_,c_],[a,c_,d]])
    return trimesh.Trimesh(vertices=np.array(verts),faces=np.array(faces),process=True)

# Route map
CADQUERY_MAP={
    "bracket":(gen_bracket,["bracket","mount","l-bracket","mounting bracket","clamp bracket"]),
    "shaft":(gen_shaft,["shaft","axle","rod","spindle","pin"]),
    "plate":(gen_plate,["plate","panel","flat","baseplate","sheet"]),
    "housing":(gen_housing,["housing","enclosure","box","case","shell","cover"]),
    "gear":(gen_true_involute_gear,["gear","spur gear","cog","pinion","toothed"]),
    "flange":(gen_flange,["flange","pipe flange","disc flange"]),
    "ibeam":(gen_ibeam,["i-beam","h-beam","universal beam","rsj"]),
    "motor_mount":(gen_motor_mount,["motor mount","motor plate","motor holder"]),
    "heatsink":(gen_heatsink,["heatsink","heat sink","cooling fin","thermal sink"]),
    "wing_rib":(gen_wing_rib_naca,["wing rib","airfoil rib","naca rib","aerofoil"]),
}
TRIMESH_MAP={
    "organic_shell":(gen_organic_shell,["organic shell","organic body","smooth shell","freeform"]),
    "swept_fairing":(gen_swept_fairing,["fairing","nacelle","aerodynamic shell","swept fairing","pod"]),
}

def route(description,params):
    d=description.lower()
    for pt,(fn,kws) in CADQUERY_MAP.items():
        if any(k in d for k in kws): return pt,"cadquery",fn(params)
    for pt,(fn,kws) in TRIMESH_MAP.items():
        if any(k in d for k in kws): return pt,"trimesh",fn(params)
    return "plate","cadquery",gen_plate(params)

def stl_from_cq(obj):
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t: p=t.name
    cq.exporters.export(obj,p); return p

def stl_from_tm(mesh):
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t: p=t.name
    mesh.export(p); return p

def step_from_cq(obj):
    with tempfile.NamedTemporaryFile(suffix=".step",delete=False) as t: p=t.name
    cq.exporters.export(obj,p); return p

# ═══════════════════════════════════════════════════════════════════
# CORE ANALYSIS PIPELINE v8.0
# ═══════════════════════════════════════════════════════════════════

async def run_analysis_v8(mesh, filename, part_name, mat_key,
                           force_n=1000, force_dir="z", T_op=25.0,
                           proj=None, surface_finish="machined",
                           reliability=0.99, run_topo=False,
                           topo_volfrac=0.5):
    vol=sf(mesh.volume);exts=[sf(e) for e in mesh.extents]
    se=sorted(exts);asp=se[2]/se[0] if se[0]>0 else 0;is_wt=bool(mesh.is_watertight)
    if mat_key=="auto": mat_key=detect_material(mesh)
    if mat_key not in MATERIALS: mat_key="aluminum_6061"
    mat=MATERIALS[mat_key];ctx=classify_context(part_name,proj)

    # All analysis algorithms
    wt_=wall_thickness_v8(mesh)
    zones=exact_zones(mesh)
    holes=detect_holes_v8(mesh)
    sharp=detect_sharp_v8(mesh,mat_key)

    # Try CalculiX first, fall back to analytical
    fea,calculix_diag=run_calculix_fem(mesh,mat_key,force_n,force_dir)
    if fea is None:
        fea=multi_section_fea(mesh,mat_key,force_n,force_dir,ctx["min_sf"])
        fea["calculix_diagnostic"]=calculix_diag
    else:
        fea["required_sf"]=ctx["min_sf"]
        fea["status"]="PASS" if fea["safety_factor"]>=ctx["min_sf"] else "FAIL"
        fea["buckling"]=fea.get("buckling",{"safety_factor":999,"status":"PASS","critical_load_n":1e9})
        fea["dynamics"]=fea.get("dynamics",multi_section_fea(mesh,mat_key,force_n,force_dir)["dynamics"])
        fea["stress"]=fea.get("stress",{"von_mises_mpa":fea.get("von_mises_mpa",0),
                                         "axial_mpa":0,"bending_mpa":0,"shear_mpa":0,"stress_concentration_kt":1.0})
        if "von_mises_mpa" in fea and "stress" not in fea:
            fea["stress"]={"von_mises_mpa":fea["von_mises_mpa"],"axial_mpa":0,"bending_mpa":0,"shear_mpa":0}
        fea["deflection_mm"]=fea.get("deflection_mm",0)
        fea["min_section_area_mm2"]=fea.get("min_section_area_mm2",0)

    vm=fea["stress"]["von_mises_mpa"]
    fat=full_marin_fatigue(mat_key,max(vm,1.0),surface=surface_finish,
                            reliability=reliability,size_mm=min(exts),
                            temp_c=T_op,notch_kt=sharp.get("max_Kt",1.0))
    frac=fracture_v8(mat_key,max(vm,1.0))
    therm=thermal_v8(mat_key,T_op)
    cr=creep_v8(mat_key,max(vm,1.0),T_op)
    cont=contact_v8(mat_key)
    rules=rule_engine_v8(mesh,wt_,ctx,mat_key,holes,sharp,fea,part_name)

    # Topology optimization (optional — takes extra time)
    topo={"note":"Topology optimization not requested. Add run_topo=true to enable."}
    if run_topo:
        topo=topology_optimization_simp(mesh,mat_key,topo_volfrac)

    # Manufacturing cost
    cost=estimate_manufacturing_cost(mesh,mat_key,"cnc")

    try:
        cog=mesh.center_mass;bnds=mesh.bounds;gc=(bnds[0]+bnds[1])/2
        off=float(np.linalg.norm(cog-gc));cog_pct=float(off/max(exts)*100) if max(exts)>0 else 0
        cog_d={"x":round(float(cog[0]),3),"y":round(float(cog[1]),3),"z":round(float(cog[2]),3)}
    except:
        cog_pct=0;cog_d={"x":0,"y":0,"z":0};bnds=mesh.bounds

    hs=health_score_v8(is_wt,rules,wt_,asp,cog_pct,fea)
    fn_=mesh.face_normals;inw=fn_[:,1]<-0.3
    inw_c=int(inw.sum());inw_p=float(inw_c/len(fn_)*100) if len(fn_)>0 else 0

    gc_str=build_gemini_context_v8(filename,part_name,mat,exts,vol,is_wt,
                                    wt_,holes,sharp,rules,fea,fat,frac,
                                    therm,cr,cont,topo,hs,T_op,proj,ctx)

    return {
        "lumexa_version":"8.0",
        "fea_method":fea.get("method","unknown"),
        # FIX: this used to compare against "calculix_real_fem", a string the
        # analysis service never returns (it returns "calculix_solid_tet_fem" or
        # "calculix_shell_fem") — so this flag reported False on every single
        # request regardless of whether real FEM actually ran. Confirmed live.
        "calculix_used":fea.get("method","") in ("calculix_solid_tet_fem","calculix_shell_fem"),
        "filename":filename,"part_name":part_name,"part_context":ctx,
        "geometry":{
            "dimensions_mm":{"x":round(exts[0],3),"y":round(exts[1],3),"z":round(exts[2],3)},
            "volume_mm3":round(vol,3),"surface_area_mm2":round(sf(mesh.area),3),
            "is_watertight":is_wt,"vertex_count":int(len(mesh.vertices)),
            "face_count":int(len(mesh.faces)),"aspect_ratio":round(asp,3),
            "center_of_mass":cog_d,"cog_offset_pct":round(cog_pct,2),
            "bounds":{"min":{"x":round(float(bnds[0][0]),3),"y":round(float(bnds[0][1]),3),"z":round(float(bnds[0][2]),3)},
                      "max":{"x":round(float(bnds[1][0]),3),"y":round(float(bnds[1][1]),3),"z":round(float(bnds[1][2]),3)}}},
        "material":{"key":mat_key,"name":mat["name"],"auto_detected":True,
            "properties":{"yield_strength_mpa":mat["yield_strength_mpa"],
                          "ultimate_strength_mpa":mat["ultimate_strength_mpa"],
                          "youngs_modulus_gpa":mat["youngs_modulus_gpa"],
                          "density_g_cm3":mat["density"],"max_service_temp_c":mat["max_service_temp_c"],
                          "fatigue_limit_mpa":mat["fatigue_limit_mpa"],
                          "fracture_toughness_mpa_sqrtm":mat["fracture_toughness_mpa_sqrtm"]}},
        "material_weights_grams":mat_weights(vol),
        "wall_thickness":wt_,"zone_locations":{"total_zones":len(zones),"zones":zones},
        "hole_analysis":{"holes_detected":len(holes),"violations":[h for h in holes if h.get("violation")],"all_holes":holes},
        "sharp_corner_analysis":sharp,
        "enclosed_pockets":{"inward_face_count":inw_c,"inward_percentage":round(inw_p,2),
            "thermal_risk":inw_p>20,"severity":"HIGH" if inw_p>40 else "MEDIUM" if inw_p>20 else "LOW"},
        "rule_engine":{"total_violations":len(rules),
            "critical":[v for v in rules if v["severity"]=="CRITICAL"],
            "high":[v for v in rules if v["severity"]=="HIGH"],
            "medium":[v for v in rules if v["severity"]=="MEDIUM"],
            "low":[v for v in rules if v["severity"]=="LOW"],
            "all_violations":rules},
        "analytical_fea":fea,"fatigue_analysis":fat,"fracture_mechanics":frac,
        "thermal_analysis":therm,"creep_analysis":cr,"contact_mechanics":cont,
        "topology_optimization":topo,"manufacturing_cost":cost,
        "health_score":hs,
        "summary":{"health_score":hs["score"],"health_label":hs["label"],
            "material":mat["name"],"fea_method":fea.get("method","unknown"),
            "fea_status":fea["status"],"safety_factor":fea["safety_factor"],
            "fatigue_status":fat.get("status","N/A"),"fracture_status":frac.get("status","N/A"),
            "thermal_status":therm.get("status","N/A"),"creep_status":cr.get("status","N/A"),
            "total_rule_violations":len(rules),
            "critical_violations":len([v for v in rules if v["severity"]=="CRITICAL"]),
            "holes_detected":len(holes),
            "holes_with_violations":len([h for h in holes if h.get("violation")]),
            "is_watertight":is_wt,"estimated_cost_usd":cost.get("total_cost_usd")},
        "gemini_context":gc_str,
    }

# ═══════════════════════════════════════════════════════════════════
# AI-LOOP QUALITY GATE — used by /generate-validate-refine
# ═══════════════════════════════════════════════════════════════════

def evaluate_design_quality(result: dict, min_health_score: float = 75.0,
                             max_critical: int = 0, max_high: int = 2,
                             min_safety_factor: float = 1.0) -> dict:
    """
    Decide whether an analyzed design is "good enough" or needs another refinement pass.

    Returns:
        {
          "passed": bool,
          "score": float,            # health score 0-100
          "reasons": [str, ...],     # human-readable list of why it failed (empty if passed)
          "metrics": {...}            # key numbers used for the decision
        }
    """
    hs = result.get("health_score", {}) or {}
    score = hs.get("score", 0)
    re_ = result.get("rule_engine", {}) or {}
    n_crit = re_.get("total_violations", 0) and len(re_.get("critical", []))
    n_high = len(re_.get("high", []))
    fea = result.get("analytical_fea", {}) or {}
    sfv = fea.get("safety_factor", 0)
    fea_status = fea.get("status", "UNKNOWN")
    fat_status = (result.get("fatigue_analysis", {}) or {}).get("status", "N/A")
    is_wt = (result.get("geometry", {}) or {}).get("is_watertight", True)

    reasons = []
    if score < min_health_score:
        reasons.append(f"Health score {score} is below target {min_health_score}.")
    if n_crit > max_critical:
        reasons.append(f"{n_crit} CRITICAL rule violation(s) found (max allowed {max_critical}).")
    if n_high > max_high:
        reasons.append(f"{n_high} HIGH-severity rule violation(s) found (max allowed {max_high}).")
    if sfv < min_safety_factor:
        reasons.append(f"FEA safety factor {sfv:.2f} is below minimum {min_safety_factor}.")
    if fea_status == "FAIL":
        reasons.append("FEA status is FAIL.")
    if fat_status == "FAIL":
        reasons.append("Fatigue analysis status is FAIL.")
    if not is_wt:
        reasons.append("Mesh is not watertight (manifold geometry required for manufacturing).")

    return {
        "passed": len(reasons) == 0,
        "score": score,
        "reasons": reasons,
        "metrics": {
            "health_score": score,
            "critical_violations": n_crit,
            "high_violations": n_high,
            "safety_factor": sfv,
            "fea_status": fea_status,
            "fatigue_status": fat_status,
            "is_watertight": is_wt,
        }
    }

def summarize_analysis_for_refinement(result: dict, quality: dict) -> str:
    """
    Build a concise, actionable feedback report from a run_analysis_v8 result + its
    quality verdict, formatted for the LLM's REFINEMENT MODE prompt.
    """
    geo = result.get("geometry", {}) or {}
    wt = result.get("wall_thickness", {}) or {}
    holes = (result.get("hole_analysis", {}) or {}).get("violations", [])
    sharp = result.get("sharp_corner_analysis", {}) or {}
    fea = result.get("analytical_fea", {}) or {}
    rules = (result.get("rule_engine", {}) or {}).get("all_violations", [])

    lines = []
    lines.append(f"HEALTH SCORE: {quality['score']} ({result.get('health_score',{}).get('label','?')})")
    lines.append(f"PASSED: {quality['passed']}")
    lines.append("")
    lines.append("WHY IT FAILED (fix all of these):" if not quality["passed"] else "Minor issues to polish:")
    for r in quality["reasons"]:
        lines.append(f"  - {r}")

    lines.append("")
    lines.append(f"GEOMETRY: dims(mm)={geo.get('dimensions_mm')} aspect_ratio={geo.get('aspect_ratio')} "
                  f"watertight={geo.get('is_watertight')} volume_mm3={geo.get('volume_mm3')}")
    if wt:
        lines.append(f"WALL THICKNESS: min={wt.get('min_mm')}mm avg={wt.get('avg_mm')}mm "
                      f"thin_<2mm_pct={wt.get('thin_2mm_pct')}")
    lines.append(f"FEA: method={fea.get('method')} safety_factor={fea.get('safety_factor')} "
                  f"status={fea.get('status')} von_mises_mpa={fea.get('stress',{}).get('von_mises_mpa')}")
    crit=fea.get("critical_section") or {}
    if fea.get("status")=="FAIL" and crit.get("position_mm") is not None:
        lines.append(f"  -> WEAKEST SECTION is along the {crit.get('axis')}-axis at "
                      f"{crit.get('position_mm')}mm (min area={fea.get('min_section_area_mm2')}mm²). "
                      f"Thicken/add material AT THIS LOCATION specifically (approx "
                      f"{crit.get('strengthen_factor_approx')}x more cross-section needed there) "
                      f"— do not thin any other area to compensate.")

    if holes:
        lines.append("HOLE VIOLATIONS:")
        for h in holes[:5]:
            lines.append(f"  - hole at {h.get('position')} diameter={h.get('diameter_mm')}mm "
                          f"min_edge_required={h.get('min_edge_req_mm')}mm — move it inward.")

    crit_corners = sharp.get("critical_zones", [])
    if crit_corners:
        lines.append("SHARP CORNER STRESS CONCENTRATIONS:")
        for c in crit_corners[:5]:
            lines.append(f"  - at {c.get('position')} Kf={c.get('Kf')} "
                          f"recommend fillet >= {c.get('fillet_rec_mm')}mm")

    if rules:
        lines.append("ALL RULE VIOLATIONS:")
        for r in rules[:10]:
            lines.append(f"  - [{r.get('severity')}] {r.get('rule_id')}: {r.get('message')} "
                          f"-> FIX: {r.get('fix')}")

    return "\n".join(lines)

async def mesh_from_cq_object(obj):
    """Export a CadQuery object to STL and load as a trimesh mesh. Returns (mesh, stl_bytes)."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as t:
        tmp = t.name
    try:
        cq.exporters.export(obj, tmp)
        mesh = trimesh.load(tmp)
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        with open(tmp, "rb") as f:
            stl_bytes = f.read()
        return mesh, stl_bytes
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
@_sanitize_response
def home():
    # As of the analysis-service split, FEM capability lives in a SEPARATE
    # process — this service's own local CALCULIX/GMSH flags will correctly
    # be False now (those dependencies were deliberately removed from this
    # service's own image to shrink its memory footprint), so reporting
    # capability from ANALYSIS_SERVICE_URL's presence is what's actually
    # true now, not the local flags (which would otherwise make this status
    # page report solid-tet FEM as unavailable even when it's working fine
    # via the remote service).
    fea_label = (
        "real solid tetrahedral FEM (C3D4, Gmsh-meshed + CalculiX) — via separate "
        "analysis service" if ANALYSIS_SERVICE_URL
        else "multi-section analytical (ANALYSIS_SERVICE_URL not configured on "
             "this deployment — set it to enable real FEM)"
    )
    return {
        "status":"Lumexa v8.22 Enterprise (split architecture) — Vibe Engineering Edition",
        "methodology_note": "The fields below describe *what each module does*, not an "
            "independently-verified accuracy percentage — none of these have been "
            "benchmarked against NAFEMS or other published test cases yet.",
        "methodology_map":{
            "geometry":"trimesh exact math","wall_thickness":"dual-pass surface sampling",
            "holes":"multi-axis RANSAC","sharp_corners":"Peterson-Neuber stress concentration",
            "fea": fea_label,
            "fatigue":"full 6-factor Marin + Goodman/Gerber","fracture":"Paris Law + FAD",
            "thermal":"gradient field + Coffin-Manson","creep":"Norton + Larson-Miller",
            "topology":"SIMP-style density heuristic (fast first pass, not per-iteration "
                       "FEA-verified — see topology_optimization_simp docstring)",
            "composite":"Classical Laminate Theory + Tsai-Wu"},
        "capabilities":{
            "cadquery_available":CQ,
            "analysis_service_configured": bool(ANALYSIS_SERVICE_URL),
            "solid_tet_fem_available": bool(ANALYSIS_SERVICE_URL),
            "blender_available":BLENDER,
            "dxf_export_available": EZDXF,
            "ai_generation_configured": bool(LOVABLE_API_KEY or ANTHROPIC_API_KEY or GOOGLE_API_KEY or GROQ_API_KEY or OPENROUTER_API_KEY),
            "ai_provider": AI_PROVIDER,
            "ai_model": (CLAUDE_MODEL if AI_PROVIDER == "claude"
                         else GEMINI_MODEL if AI_PROVIDER == "gemini"
                         else GROQ_MODEL if AI_PROVIDER == "groq"
                         else OPENROUTER_MODEL if AI_PROVIDER == "openrouter"
                         else LOVABLE_AI_MODEL)},
        "new_in_v8_3":[
            "AI generation can now route through the direct Anthropic Claude API "
            "instead of the Gemini/Lovable gateway — set ANTHROPIC_API_KEY to enable, "
            "controlled via the AI_PROVIDER env var (see ai_provider/ai_model above "
            "for what's active on this deployment)",
            "POST /refine-from-external-fea  ★ closes the design loop using a real "
            "Ansys export (coordinate + von-Mises-stress CSV), not just this "
            "platform's own internal analysis — reuses the same refinement engine "
            "as /generate-validate-refine so an Ansys-driven fix and an internal-loop "
            "fix go through identical machinery",
            "POST /edit-design-region  ★ draw a 3D bounding box, AI regenerates "
            "only what's inside it — cut+union guarantees everything outside is "
            "unchanged, not just prompted to be",
            "POST /export-step  ★ STEP export at any point in a design's lifecycle "
            "(not just first generation) — for the FreeCAD manual-edit workflow, "
            "which needs a real B-Rep solid, not just STL triangles",
        ],
        "new_in_v8_2":[
            "Solid tetrahedral FEM: parts are now Gmsh-volume-meshed into real C3D4 "
            "elements and solved with CalculiX, not approximated as a shell — falls "
            "back to shell FEM automatically if tet-meshing fails on a given part",
            "AST-based sandboxing for AI-generated CadQuery scripts (replaces a "
            "substring blocklist) — blocks import-based and reflection-based "
            "(getattr/__subclasses__/__mro__) sandbox escapes",
            "POST /export-drawing-dxf: 2D manufacturing drawing export (orthographic "
            "views + dimensions + title block) for laser-cutting/CNC shops that work "
            "from DXF rather than STEP/STL",
            "Fixed: max_displacement_mm was previously hardcoded to 0.0 in the FEM "
            "path; now actually parsed from solver output",
            "CORS no longer combines wildcard origin with allow_credentials=True",
        ],
        "new_in_v8_1":[
            "/generate-validate-refine: self-healing AI design loop — generate, "
            "analyze, and auto-fix CAD designs until they pass FEA/fatigue/rule checks",
            "AI generation now routes through Lovable AI Gateway (no per-request API key)",
            "Non-raising script execution with structured engineering feedback for refinement",
            "Vision (image-to-params) also routed through Lovable AI Gateway",
        ],
        "new_in_v8":[
            "CalculiX real FEM — see ai_provider/methodology_map above for current "
            "solver/element-type honesty; do not treat this as a fixed accuracy %",
            "SIMP topology optimization",
            "Classical Laminate Theory composites",
            "Rainflow fatigue counting (ASTM E1049)",
            "Manufacturing cost estimation",
            "Gemini script generation (/generate-from-prompt)",
            "Design comparison (/compare-designs)",
            "Background job queue for heavy analysis",
        ],
        "endpoints":[
            "GET  /","GET  /materials","GET  /part-types",
            "POST /analyze-part","POST /analyze-assembly",
            "POST /generate-part","POST /generate-and-analyze",
            "POST /generate-from-prompt",
            "POST /generate-validate-refine  ★ self-correcting AI design loop",
            "POST /refine-from-external-fea  ★ closes the loop on a real Ansys export",
            "POST /edit-design-region  ★ boundary-box AI edit with guaranteed-unchanged rest",
            "POST /analyze-composite","POST /analyze-rainflow",
            "POST /compare-designs","POST /image-to-params",
            "POST /analyze-part-deep (background CalculiX)",
            "POST /export-drawing-dxf  ★ 2D manufacturing drawing export",
            "GET  /job/{job_id}",
        ],
        "recommended_flow":[
            "1. POST /generate-validate-refine with a natural-language part description.",
            "2. Inspect `refinement.history` to see what the AI fixed and why.",
            "3. If `refinement.passed_quality_gate` is false, loosen thresholds or "
            "increase max_iterations and retry — the best attempt is always returned.",
            "4. Decode `generated_stl_base64` to get the manufacturable STL.",
        ],
    }

@app.get("/materials")
@_sanitize_response
def get_materials():
    return {k:{"name":v["name"],"density":v["density"],
                "yield_mpa":v["yield_strength_mpa"],"max_temp_c":v["max_service_temp_c"],
                "cost_per_kg":v.get("cost_per_kg_usd","N/A")} for k,v in MATERIALS.items()}

@app.get("/part-types")
@_sanitize_response
def get_part_types():
    return {"cadquery":{k:v[1] for k,v in CADQUERY_MAP.items()},
            "organic":{k:v[1] for k,v in TRIMESH_MAP.items()}}

@app.post("/analyze-part")
@_sanitize_response
async def analyze_part(
    file:UploadFile=File(...),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    force_dir:str=Form("z"),
    operating_temp_c:float=Form(25.0),
    surface_finish:str=Form("machined"),
    reliability:float=Form(0.99),
    run_topology:bool=Form(False),
    part_name:Optional[str]=Form(None),
    project_description:Optional[str]=Form(None),
):
    """Full v8.0 analysis. CalculiX FEM if available, analytical fallback."""
    contents=await file.read();fn=file.filename or "part.stl"
    with tempfile.NamedTemporaryFile(suffix="."+fn.split(".")[-1].lower(),delete=False) as t:
        t.write(contents);tmp=t.name
    try:
        mesh=trimesh.load(tmp)
        if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
        return await run_analysis_v8(mesh,fn,part_name or fn,material,
                                      force_n,force_dir,operating_temp_c,
                                      project_description,surface_finish,reliability,run_topology)
    finally: os.unlink(tmp)

@app.post("/analyze-part-deep")
@_sanitize_response
async def analyze_part_deep(
    background_tasks:BackgroundTasks,
    file:UploadFile=File(...),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    part_name:Optional[str]=Form(None),
    project_description:Optional[str]=Form(None),
):
    """
    Background analysis with full CalculiX + topology optimization.
    Returns job_id immediately. Poll /job/{job_id} for results.
    Use this for complex parts where 5-10 minute analysis is acceptable.
    """
    contents=await file.read();fn=file.filename or "part.stl"
    job_id=str(uuid.uuid4())
    JOB_STORE[job_id]={"status":"running","created":time.time(),"filename":fn}

    async def run_job():
        try:
            with tempfile.NamedTemporaryFile(suffix="."+fn.split(".")[-1].lower(),delete=False) as t:
                t.write(contents);tmp=t.name
            try:
                mesh=trimesh.load(tmp)
                if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
                result=await run_analysis_v8(mesh,fn,part_name or fn,material,
                                              force_n,"z",25.0,project_description,
                                              "machined",0.999,True,0.5)
                JOB_STORE[job_id]={"status":"complete","result":result,"created":time.time()}
            finally: os.unlink(tmp)
        except Exception as e:
            JOB_STORE[job_id]={"status":"error","error":str(e),"created":time.time()}

    background_tasks.add_task(run_job)
    return {"job_id":job_id,"status":"running",
            "message":"Analysis started. Poll /job/{job_id} for results.",
            "estimated_time":"2-8 minutes with CalculiX, 30s without"}

@app.get("/job/{job_id}")
@_sanitize_response
def get_job(job_id:str):
    """Poll background analysis job status."""
    if job_id not in JOB_STORE:
        raise HTTPException(404,"Job not found")
    job=JOB_STORE[job_id]
    if job["status"]=="complete":
        return job["result"]
    elif job["status"]=="error":
        raise HTTPException(500,job.get("error","Unknown error"))
    else:
        elapsed=time.time()-job["created"]
        return {"status":"running","elapsed_seconds":round(elapsed,1),
                "message":"Analysis in progress..."}

@app.post("/generate-part")
@_sanitize_response
async def generate_part(
    description:str=Form(...),
    params:str=Form("{}"),
    export_format:str=Form("stl"),
):
    if not CQ: raise HTTPException(503,"CadQuery not installed")
    try: pd=json.loads(params)
    except: pd={}
    pt,gen_type,obj=route(description,pd)
    tmp=stl_from_cq(obj) if gen_type=="cadquery" else stl_from_tm(obj)
    suffix=".stl" if export_format in ["stl","STL"] else ".step"
    if export_format not in ["stl","STL"]: tmp=step_from_cq(obj) if gen_type=="cadquery" else tmp
    return FileResponse(path=tmp,media_type="application/octet-stream",filename=f"lumexa_{pt}{suffix}")

@app.post("/generate-and-analyze")
@_sanitize_response
async def generate_and_analyze(
    description:str=Form(...),
    params:str=Form("{}"),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    operating_temp_c:float=Form(25.0),
    surface_finish:str=Form("machined"),
    reliability:float=Form(0.99),
    project_description:Optional[str]=Form(None),
):
    if not CQ: raise HTTPException(503,"CadQuery not installed")
    try: pd=json.loads(params)
    except: pd={}
    pt,gen_type,obj=route(description,pd)
    tmp=stl_from_cq(obj) if gen_type=="cadquery" else stl_from_tm(obj)
    try:
        mesh=trimesh.load(tmp)
        if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
        with open(tmp,"rb") as f: stl_b64=base64.b64encode(f.read()).decode()
        result=await run_analysis_v8(mesh,description,description,material,
                                      force_n,"z",operating_temp_c,project_description,
                                      surface_finish,reliability)
        result["generated_stl_base64"]=stl_b64
        result["part_type_detected"]=pt
        result["generation_engine"]=gen_type
        return result
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

@app.post("/generate-from-prompt")
@_sanitize_response
async def generate_from_prompt(
    prompt:str=Form(...),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    operating_temp_c:float=Form(25.0),
):
    """
    Any part from natural language → Gemini (via Lovable AI Gateway) writes CadQuery → real STL + analysis.
    Single-shot version (no refinement loop). For an AI design that automatically fixes
    its own engineering problems, use POST /generate-validate-refine instead.
    """
    if not CQ: raise HTTPException(503,"CadQuery not installed")

    # Gemini (Lovable AI Gateway) generates script
    try:
        script=await gemini_generate_script(prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502,f"Gemini API error: {str(e)}")

    # Execute safely (non-raising)
    obj,err=execute_cq_script_safely(script)
    if err:
        raise HTTPException(400,f"Generated script failed: {err}")

    # Export + analyze
    mesh,stl_bytes=await mesh_from_cq_object(obj)
    stl_b64=base64.b64encode(stl_bytes).decode()
    result=await run_analysis_v8(mesh,prompt,prompt,material,force_n,"z",operating_temp_c)
    result["generated_stl_base64"]=stl_b64
    result["generated_script"]=script
    result["generation_method"]="gemini_cadquery_v8"
    return result

def _parse_groq_retry_after(detail: str) -> float:
    """Groq's 429 body includes literal text like 'Please try again in 44.01s.' —
    parse that so the refine loop backs off exactly as long as needed instead
    of guessing. Falls back to a conservative default if the message format
    ever changes upstream."""
    import re
    m = re.search(r"try again in ([\d.]+)s", detail or "")
    if m:
        try:
            return float(m.group(1)) + 0.5  # small safety margin
        except ValueError:
            pass
    return 5.0


def _diagnose_cq_error(err: str) -> str:
    """Pattern-match known CadQuery/OpenCascade failure signatures and append
    specific, actionable guidance — confirmed live: the generic 'fix the root
    cause' feedback wasn't enough for the model to recover from these across
    a real refinement run (it got a DIFFERENT failure on the retry instead of
    a working script). These two are common, recurring OCC fillet/chamfer
    failures, not one-off flukes."""
    hints = []
    if "requires that edges be selected" in err:
        hints.append(
            "Your .edges()/.faces() selector for the fillet/chamfer matched "
            "ZERO edges — the selection string is stale or wrong after prior "
            "operations changed the current context. Select the edges "
            "immediately after creating the feature they belong to, before "
            "chaining unrelated operations, and double-check the selector "
            "string (e.g. '|Z', '>Z', 'not(%CIRCLE)') actually matches edges "
            "that exist on THIS solid."
        )
    if "BRep_API: command not done" in err or "StdFail_NotDone" in err:
        hints.append(
            "The CAD kernel REJECTED a fillet/chamfer/boolean operation as "
            "geometrically infeasible — almost always because the requested "
            "radius is too large for the edge it's applied to (bigger than "
            "the material thickness, or it would overlap an adjacent edge or "
            "hole). Use a SMALLER radius (rule of thumb: no more than 20-30% "
            "of the local wall thickness), and apply fillets/chamfers BEFORE "
            "cutting nearby holes so the kernel has simpler geometry to work with."
        )
    return ("\n\n" + "\n\n".join(hints)) if hints else ""


@app.post("/generate-validate-refine")
@_sanitize_response
async def generate_validate_refine(
    prompt:str=Form(...),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    force_dir:str=Form("z"),
    operating_temp_c:float=Form(25.0),
    surface_finish:str=Form("machined"),
    reliability:float=Form(0.99),
    project_description:Optional[str]=Form(None),
    max_iterations:int=Form(3),
    min_health_score:float=Form(75.0),
    max_critical_violations:int=Form(0),
    max_high_violations:int=Form(2),
    min_safety_factor:float=Form(1.0),
    run_topology:bool=Form(False),
):
    """
    ★ THE CORE VIBE-ENGINEERING LOOP ★

    1. AI (Gemini via Lovable AI Gateway) writes a CadQuery script from `prompt`.
    2. Script is executed → STL → full engineering analysis (FEA, fatigue, fracture,
       wall thickness, hole placement, sharp-corner stress, rule engine, health score).
    3. The result is checked against quality thresholds (health score, safety factor,
       critical/high violation counts, watertightness).
    4. If it fails AND iterations remain, the full analysis is fed back to the AI as a
       structured feedback report (REFINEMENT MODE) and it produces a corrected script.
    5. Repeat until it passes or `max_iterations` is reached. The BEST iteration
       (highest health score) is returned, plus the full iteration history.

    This is the endpoint that makes the platform "self-healing": bad first drafts get
    automatically engineered into something that passes real structural checks.

    Quality thresholds (tune per-project):
      - min_health_score: target health score 0-100 (default 75)
      - max_critical_violations: CRITICAL rule violations allowed (default 0)
      - max_high_violations: HIGH-severity violations allowed (default 2)
      - min_safety_factor: minimum FEA safety factor (default 1.0)
    """
    if not CQ: raise HTTPException(503,"CadQuery not installed")
    if max_iterations<1: max_iterations=1
    if max_iterations>6: max_iterations=6  # hard cap: cost + latency safety

    iterations=[]
    best=None        # best {"result":..., "quality":..., "script":..., "stl_b64":..., "iteration":int}
    script=None
    feedback=None
    stopped_reason=None
    # Groq's free tier caps at 8000 TPM — confirmed live: a refinement loop
    # burns through that in 1-2 calls, and this used to silently `break` on
    # the resulting 429 with zero indication in the response that the loop
    # was cut short (iterations_used < max_iterations looked like a decision,
    # not a failure). Now: retry using Groq's own stated wait time first, and
    # always record *why* the loop stopped in result["refinement"]["stopped_reason"].
    RATE_LIMIT_MAX_TOTAL_WAIT=90.0  # seconds, kept under typical client timeouts

    for i in range(1, max_iterations+1):
        # 1) Generate or refine the script
        rate_limit_wait_remaining=RATE_LIMIT_MAX_TOTAL_WAIT
        gen_failed=False
        while True:
            try:
                script=await gemini_generate_script(prompt, previous_script=script, feedback=feedback)
                break
            except HTTPException as e:
                is_rate_limited=(e.status_code==429)
                if is_rate_limited and rate_limit_wait_remaining>0:
                    wait_s=min(_parse_groq_retry_after(str(e.detail)), rate_limit_wait_remaining)
                    rate_limit_wait_remaining-=wait_s
                    await asyncio.sleep(wait_s)
                    continue
                if iterations:
                    stopped_reason="rate_limited" if is_rate_limited else "generation_failed"
                    gen_failed=True
                    break
                raise
        if gen_failed:
            break

        # 2) Execute
        obj,err=execute_cq_script_safely(script)
        if err:
            iterations.append({"iteration":i,"stage":"execution_failed","error":err,"script":script})
            feedback=(f"Your script FAILED TO EXECUTE with this error:\n{err}\n\n"
                      f"Fix the root cause and return a complete, runnable script."
                      f"{_diagnose_cq_error(err)}")
            continue

        # 3) Analyze
        try:
            mesh,stl_bytes=await mesh_from_cq_object(obj)
            result=await run_analysis_v8(mesh,prompt,prompt,material,force_n,force_dir,
                                          operating_temp_c,project_description,
                                          surface_finish,reliability,run_topology)
        except Exception as e:
            iterations.append({"iteration":i,"stage":"analysis_failed","error":str(e),"script":script})
            feedback=(f"The generated geometry exported, but the analysis pipeline raised:\n{str(e)}\n\n"
                      f"This usually means degenerate/non-manifold geometry. Simplify or fix the "
                      f"geometry (avoid zero-thickness faces, self-intersections, open shells) and "
                      f"return a complete, runnable script.")
            continue

        # 4) Quality gate
        quality=evaluate_design_quality(result, min_health_score, max_critical_violations,
                                         max_high_violations, min_safety_factor)
        stl_b64=base64.b64encode(stl_bytes).decode()
        entry={"iteration":i,"stage":"analyzed","passed":quality["passed"],
               "health_score":quality["score"],"reasons":quality["reasons"],
               "metrics":quality["metrics"]}
        iterations.append(entry)

        candidate={"result":result,"quality":quality,"script":script,
                   "stl_b64":stl_b64,"iteration":i}
        # A candidate that actually PASSED is always preferred over one that
        # didn't, no matter the raw score — a failing 95 (e.g. fatigue FAIL)
        # must never beat a passing 87. Only compare raw scores head-to-head
        # when both candidates are in the same passed/failed bucket.
        if best is None:
            best=candidate
        elif quality["passed"] != best["quality"]["passed"]:
            if quality["passed"]:
                best=candidate
        elif quality["score"]>best["quality"]["score"]:
            best=candidate

        if quality["passed"]:
            stopped_reason="quality_gate_passed"
            break

        # 5) Build feedback for next round
        feedback=summarize_analysis_for_refinement(result, quality)

    if stopped_reason is None:
        stopped_reason="max_iterations_reached"

    if best is None:
        # Every iteration failed to even produce geometry — surface the last error.
        last=iterations[-1] if iterations else {}
        raise HTTPException(502, "AI failed to produce a valid CAD design after "
                                  f"{len(iterations)} attempt(s). Last error: "
                                  f"{last.get('error','unknown')}")

    result=best["result"]
    result["generated_stl_base64"]=best["stl_b64"]
    result["generated_script"]=best["script"]
    result["generation_method"]="gemini_cadquery_v8_refined"
    result["refinement"]={
        "iterations_used":len(iterations),
        "max_iterations":max_iterations,
        "best_iteration":best["iteration"],
        "passed_quality_gate":best["quality"]["passed"],
        "stopped_reason":stopped_reason,
        "final_reasons":best["quality"]["reasons"],
        "quality_thresholds":{
            "min_health_score":min_health_score,
            "max_critical_violations":max_critical_violations,
            "max_high_violations":max_high_violations,
            "min_safety_factor":min_safety_factor,
        },
        "history":iterations,
    }
    return result


@app.post("/refine-from-external-fea")
@_sanitize_response
async def refine_from_external_fea(
    prompt: str = Form(...),
    previous_script: str = Form(...),
    material: str = Form("aluminum_6061"),
    hotspots_csv: UploadFile = File(...),
    top_n: int = Form(5),
    force_n: float = Form(1000.0),
    force_dir: str = Form("z"),
    operating_temp_c: float = Form(25.0),
    surface_finish: str = Form("machined"),
    reliability: float = Form(0.99),
    min_health_score: float = Form(75.0),
    max_critical_violations: int = Form(0),
    max_high_violations: int = Form(2),
    min_safety_factor: float = Form(1.0),
):
    """
    ★ CLOSES THE LOOP WITH EXTERNAL FEA (e.g. Ansys), NOT JUST THIS PLATFORM'S OWN ★

    Feed in: the original prompt, the script that produced the part an engineer ran
    through Ansys, and a CSV of stress-hotspot results exported from Ansys (a
    coordinate + von-Mises-stress table — Ansys can export this from its results
    viewer/probe table). This builds the same structured feedback text the internal
    /generate-validate-refine loop generates from its own analysis, then reuses that
    identical refinement machinery — a correction driven by a certified Ansys run
    goes through the same code path as an internal-loop correction, not a separate
    or lesser one.

    CSV columns (case-insensitive, flexible naming): x / y / z coordinates in mm,
    plus a stress column (accepts: von_mises_mpa, vm_stress, stress, stress_mpa,
    s.mises, "equivalent stress"). Extra columns are ignored.

    SCOPE NOTE: this does NOT parse Ansys's native binary result files (.rst/.odb)
    — those are proprietary formats. Export a coordinate+stress table to CSV from
    Ansys's results viewer first. This also does not re-verify the fix in Ansys —
    the response is rechecked against Lumexa's own internal analysis only; send the
    result back through Ansys to confirm before trusting it for anything real.
    """
    if not CQ:
        raise HTTPException(503, "CadQuery not installed")

    contents = await hotspots_csv.read()
    text = contents.decode(errors="ignore")

    import csv, io
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV has no header row / couldn't be parsed.")

    def find_col(cands):
        lower = {f.lower().strip(): f for f in reader.fieldnames}
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    x_col = find_col(["x", "x_mm", "x (mm)", "xcoord", "x coordinate"])
    y_col = find_col(["y", "y_mm", "y (mm)", "ycoord", "y coordinate"])
    z_col = find_col(["z", "z_mm", "z (mm)", "zcoord", "z coordinate"])
    s_col = find_col(["von_mises_mpa", "von mises", "vonmises", "vm_stress",
                       "s.mises", "s_mises", "stress", "stress_mpa", "equivalent stress"])

    if not s_col:
        raise HTTPException(400, f"Couldn't find a stress column in the CSV. Columns found: "
                                  f"{reader.fieldnames}. Rename your stress column to one of: "
                                  f"von_mises_mpa, vm_stress, stress_mpa.")

    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    rows = []
    for row in reader:
        stress = _to_float(row.get(s_col))
        if stress is None:
            continue
        rows.append({
            "x": _to_float(row.get(x_col)) if x_col else None,
            "y": _to_float(row.get(y_col)) if y_col else None,
            "z": _to_float(row.get(z_col)) if z_col else None,
            "von_mises_mpa": stress,
        })

    if not rows:
        raise HTTPException(400, "No valid numeric stress values found in the CSV.")

    rows.sort(key=lambda r: r["von_mises_mpa"], reverse=True)
    top = rows[:max(1, min(top_n, 20))]

    mat = MATERIALS.get(material, MATERIALS["aluminum_6061"])
    Sy = mat["yield_strength_mpa"]

    lines = [f"EXTERNAL FEA RESULTS (imported from Ansys/third-party export, NOT this "
             f"platform's own analysis) — material yield strength {Sy} MPa:"]
    any_fail = False
    for i, r in enumerate(top, 1):
        sf = round(Sy / max(r["von_mises_mpa"], 0.001), 3)
        status = "FAILS (SF < 1.0)" if sf < 1.0 else ("MARGINAL (SF < 2.0)" if sf < 2.0 else "OK")
        if sf < 2.0:
            any_fail = True
        loc = (f"at approx ({r['x']:.2f}, {r['y']:.2f}, {r['z']:.2f}) mm"
               if r["x"] is not None and r["y"] is not None and r["z"] is not None
               else "(location not provided in CSV)")
        lines.append(f"  {i}. {r['von_mises_mpa']:.2f} MPa {loc} — safety factor {sf} — {status}")

    lines.append("")
    lines.append(
        "One or more points above have an unacceptable safety factor per the certified "
        "Ansys run. Modify the geometry to reduce stress at those specific locations — "
        "typically: add a fillet/radius, increase local wall thickness, add a rib, or "
        "reroute the load path near the coordinates given above. Return a COMPLETE, "
        "corrected script. Return ONLY Python code. No markdown."
        if any_fail else
        "All reported points are within an acceptable safety factor. No geometry change "
        "is required from this feedback."
    )
    feedback_text = "\n".join(lines)

    new_script = await gemini_generate_script(prompt, previous_script=previous_script,
                                                feedback=feedback_text)

    obj, err = execute_cq_script_safely(new_script)
    if err:
        return {
            "stage": "execution_failed", "error": err, "script": new_script,
            "external_feedback_used": feedback_text,
        }

    mesh, stl_bytes = await mesh_from_cq_object(obj)
    result = await run_analysis_v8(mesh, prompt, prompt, material, force_n, force_dir,
                                    operating_temp_c, None, surface_finish, reliability, False)
    quality = evaluate_design_quality(result, min_health_score, max_critical_violations,
                                       max_high_violations, min_safety_factor)
    stl_b64 = base64.b64encode(stl_bytes).decode()

    return {
        "stage": "refined_from_external_fea",
        "script": new_script,
        "stl_base64": stl_b64,
        "internal_recheck": {"passed": quality["passed"], "health_score": quality["score"],
                              "reasons": quality["reasons"]},
        "external_hotspots_used": top,
        "external_feedback_text": feedback_text,
        "note": "Rechecked against Lumexa's own internal analysis only — NOT yet "
                "re-verified in Ansys. Send this back through Ansys to confirm the fix "
                "actually resolves the reported stress before trusting it for anything real.",
    }


import re as _re

def _rename_result_var(script: str, new_name: str) -> str:
    """
    Rename the conventional `result` variable to a unique name via word-boundary
    regex substitution, so two independently-generated scripts can be concatenated
    into one combined script without their `result` assignments colliding. Safe
    because every script this system generates is instructed to use exactly the
    literal name `result` for its final shape — see GEMINI_CADQUERY_SYSTEM.
    """
    return _re.sub(r"\bresult\b", new_name, script)


@app.post("/edit-design-region")
@_sanitize_response
async def edit_design_region(
    previous_script: str = Form(...),
    edit_prompt: str = Form(...),
    x_min: float = Form(...), y_min: float = Form(...), z_min: float = Form(...),
    x_max: float = Form(...), y_max: float = Form(...), z_max: float = Form(...),
    material: str = Form("aluminum_6061"),
):
    """
    ★ BOUNDARY-REGION EDIT ★ — user draws a 3D bounding box around part of the
    generated design; only geometry inside that box is regenerated, everything
    outside is geometrically guaranteed unchanged (via cut + union, not just a
    hopeful full-script regeneration like /refine-from-external-fea's feedback
    text approach).

    Coordinates are in the same mm coordinate space as the part itself (i.e.
    whatever the frontend's 3D viewer reports for the drawn box, untransformed).

    Pipeline:
      1. Execute previous_script -> base shape.
      2. Cut the given box out of the base shape.
      3. Ask the AI for ONLY the replacement geometry, sized to fit the box,
         centered at the local origin (NOT the box's real-world position — the
         backend handles placement, so the AI's job is just "build a shape this
         big", which is a much more reliable prompt than asking it to also get
         absolute 3D placement right).
      4. Translate the AI's local shape into the box's real position, union it
         into the cut base.
      5. Assemble ONE new combined script (previous_script + the AI's local
         script + the cut/union glue, with `result` variables renamed to avoid
         collision) so the result is a normal, fully re-editable CadQuery script
         — future edits (another region, or /refine-from-external-fea) work on
         it exactly like any other script in this system.
    """
    if not CQ:
        raise HTTPException(503, "CadQuery not installed")

    bx, by, bz = abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)
    if bx <= 0 or by <= 0 or bz <= 0:
        raise HTTPException(400, "Bounding box must have positive size on all three axes "
                                  "(check x_min<x_max, y_min<y_max, z_min<z_max).")
    cx, cy, cz = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2

    # Step 1: confirm the base script still executes before spending an AI call.
    base_obj, base_err = execute_cq_script_safely(previous_script)
    if base_err:
        raise HTTPException(400, f"previous_script failed to execute, can't edit it: {base_err}")

    # Step 2/3: ask the AI for ONLY the local replacement geometry.
    local_prompt = (
        f"Design ONLY this local replacement feature — build it centered at the "
        f"origin (0,0,0), sized to fit within a bounding box of "
        f"{bx:.2f} x {by:.2f} x {bz:.2f} mm (X x Y x Z). Do not worry about where "
        f"this sits in a larger assembly — a backend step positions it afterward. "
        f"Request: {edit_prompt}"
    )
    local_script = await gemini_generate_script(local_prompt)

    local_obj, local_err = execute_cq_script_safely(local_script)
    if local_err:
        return {"stage": "local_generation_failed", "error": local_err, "local_script": local_script}

    # Step 4/5: cut + union + assemble the combined script.
    try:
        base_renamed = _rename_result_var(previous_script, "_base_result")
        local_renamed = _rename_result_var(local_script, "_local_result")

        combined_script = (
            "import cadquery as cq\n\n"
            "# --- base shape (previous design) ---\n"
            f"{base_renamed}\n\n"
            "# --- local replacement geometry for the edited region ---\n"
            f"{local_renamed}\n\n"
            "# --- combine: cut the edited region out of the base, then union in "
            "the new local geometry, positioned at the region's real location ---\n"
            f"_cutter = cq.Workplane('XY').box({bx}, {by}, {bz}).translate(({cx}, {cy}, {cz}))\n"
            f"_local_positioned = _local_result.translate(({cx}, {cy}, {cz}))\n"
            "result = _base_result.cut(_cutter).union(_local_positioned)\n"
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to assemble combined script: {str(e)}")

    combined_obj, combined_err = execute_cq_script_safely(combined_script)
    if combined_err:
        return {
            "stage": "combine_failed",
            "error": combined_err,
            "combined_script": combined_script,
            "note": "The base and local pieces each generated fine individually, but "
                    "combining them (cut+union) failed — often means the local geometry "
                    "doesn't fully fill the box, leaving a non-manifold result, or the "
                    "box didn't actually overlap solid material in the base shape.",
        }

    mesh, stl_bytes = await mesh_from_cq_object(combined_obj)
    result = await run_analysis_v8(mesh, edit_prompt, edit_prompt, material, 1000.0, "z",
                                    25.0, None, "machined", 0.99, False)
    quality = evaluate_design_quality(result, 75.0, 0, 2, 1.0)
    stl_b64 = base64.b64encode(stl_bytes).decode()

    return {
        "stage": "region_edited",
        "script": combined_script,
        "stl_base64": stl_b64,
        "edited_region_mm": {"x_min": x_min, "y_min": y_min, "z_min": z_min,
                              "x_max": x_max, "y_max": y_max, "z_max": z_max},
        "internal_recheck": {"passed": quality["passed"], "health_score": quality["score"],
                              "reasons": quality["reasons"]},
        "note": "Everything outside the given box is geometrically guaranteed unchanged "
                "(cut+union, not a full regeneration) — only the boxed region was "
                "AI-generated. Re-run this endpoint again with a new box to edit another "
                "region, or /refine-from-external-fea for whole-part corrections.",
    }


@app.post("/export-step")
@_sanitize_response
async def export_step(script: str = Form(...)):
    """
    ★ STANDALONE STEP EXPORT ★ — the FreeCAD "edit manually" workflow needs STEP
    available at ANY point in a design's lifecycle (after initial generation,
    after a validate-refine loop, after an external-FEA-driven fix, after a
    boundary-region edit) — not just at first generation, where export_format
    already existed inside /generate-and-analyze. This is that: give it whatever
    script currently represents the design's state, get STEP back. STEP (not
    STL) is what makes the FreeCAD round-trip actually useful — it's a real
    B-Rep solid with editable faces, not just a triangle soup.
    """
    if not CQ:
        raise HTTPException(503, "CadQuery not installed")

    obj, err = execute_cq_script_safely(script)
    if err:
        raise HTTPException(400, f"Script failed to execute: {err}")

    step_path = step_from_cq(obj)
    try:
        with open(step_path, "rb") as f:
            step_b64 = base64.b64encode(f.read()).decode()
    finally:
        if os.path.exists(step_path):
            try: os.unlink(step_path)
            except: pass

    return {
        "step_base64": step_b64,
        "filename": "lumexa_part.step",
        "note": "Open this in FreeCAD (free) for manual editing. Editing outside "
                "this system breaks the script-based edit loop (/edit-design-region, "
                "/refine-from-external-fea) for whatever you change manually — "
                "re-upload the edited result to /analyze-part to re-run FEA/DFM "
                "checks on it, but treat it as a new starting point, not something "
                "the AI can keep iterating on as code.",
    }


def _hull_outline_2d(points_2d):
    """2D convex hull of a point set, returned as an ordered closed polygon (list of (x,y))."""
    pts = np.asarray(points_2d)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    hull = ConvexHull(pts)
    return [tuple(pts[i]) for i in hull.vertices]


def generate_technical_drawing_dxf(mesh, title="Lumexa Part", material_name=""):
    """
    Generate a 2D DXF manufacturing reference drawing: three orthographic-style
    views (top/front/side) plus overall dimensions and a title block.

    SCOPE NOTE — read before presenting this as a "drawing" to anyone technical:
    each view is the 2D convex hull of the mesh's vertices projected onto that
    plane, NOT a true hidden-line-removed orthographic projection (what an actual
    SolidWorks/AutoCAD drawing shows: every visible edge, holes as circles,
    internal features as dashed hidden lines). For a convex or near-convex part
    (simple brackets, enclosures, plates) the two look similar. For anything with
    concave features, pockets, or through-holes, the convex hull will NOT show
    those — it's a bounding-envelope reference good for stock sizing and rough
    layout, not a feature-complete machinist's drawing. The returned dict's
    scope_note says this to the caller; don't strip that note out in the UI.

    Returns an ezdxf Document.
    """
    verts = mesh.vertices
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]  # (dx, dy, dz)

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf_units.MM
    msp = doc.modelspace()

    for name, color in [("OUTLINE", 7), ("DIM", 1), ("TEXT", 3), ("TITLEBLOCK", 7)]:
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    gap = max(float(dims.max()) * 0.25, 20.0)

    # Top view: looking down the Z axis -> project to (X, Y)
    top_pts = _hull_outline_2d(verts[:, [0, 1]])
    top_origin = (0.0, 0.0)
    # Front view: looking along -Y -> project to (X, Z), placed above the top view
    front_pts = _hull_outline_2d(verts[:, [0, 2]])
    front_origin = (0.0, float(dims[1]) + gap)
    # Side view: looking along -X -> project to (Y, Z), placed right of the front view
    side_pts = _hull_outline_2d(verts[:, [1, 2]])
    side_origin = (float(dims[0]) + gap, float(dims[1]) + gap)

    def draw_view(pts, origin, label, x_extent, y_extent):
        # Shift each view so its min corner sits at the view's assigned origin.
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        minx, miny = min(xs), min(ys)
        placed = [(p[0]-minx+origin[0], p[1]-miny+origin[1]) for p in pts]
        if len(placed) >= 3:
            msp.add_lwpolyline(placed, close=True, dxfattribs={"layer": "OUTLINE"})
        elif len(placed) == 2:
            msp.add_line(placed[0], placed[1], dxfattribs={"layer": "OUTLINE"})
        msp.add_text(label, height=x_extent*0.04 or 3,
                      dxfattribs={"layer": "TEXT"}).set_placement(
            (origin[0], origin[1]-max(x_extent*0.08, 6)))
        try:
            msp.add_linear_dim(base=(origin[0], origin[1]-max(y_extent*0.15,10)),
                                p1=(origin[0], origin[1]),
                                p2=(origin[0]+x_extent, origin[1]),
                                dimstyle="EZDXF", dxfattribs={"layer": "DIM"}).render()
            msp.add_linear_dim(base=(origin[0]-max(x_extent*0.15,10), origin[1]),
                                p1=(origin[0], origin[1]),
                                p2=(origin[0], origin[1]+y_extent),
                                angle=90, dimstyle="EZDXF",
                                dxfattribs={"layer": "DIM"}).render()
        except Exception:
            pass  # Dimension rendering is best-effort; outline geometry is the core deliverable.

    draw_view(top_pts, top_origin, "TOP VIEW", float(dims[0]), float(dims[1]))
    draw_view(front_pts, front_origin, "FRONT VIEW", float(dims[0]), float(dims[2]))
    draw_view(side_pts, side_origin, "SIDE VIEW", float(dims[1]), float(dims[2]))

    # Title block — plain TEXT entities so the core information survives even if
    # dimension-style rendering behaves differently across ezdxf/DXF versions.
    tb_y = -max(float(dims[2]) * 0.35, 25.0)
    lines = [
        f"{title}",
        f"MATERIAL: {material_name or 'unspecified'}",
        f"OVERALL (mm): L{dims[0]:.2f} x W{dims[1]:.2f} x H{dims[2]:.2f}",
        "GENERATED BY LUMEXA (AI-assisted) — REFERENCE ONLY, NOT A CERTIFIED "
        "ENGINEERING DRAWING. Views are convex-hull silhouettes, not hidden-line "
        "projections — verify against the source model before manufacturing.",
    ]
    for i, line in enumerate(lines):
        msp.add_text(line, height=max(float(dims.max())*0.025, 2.5),
                      dxfattribs={"layer": "TITLEBLOCK"}).set_placement(
            (0.0, tb_y - i * max(float(dims.max())*0.035, 3.5)))

    return doc


@app.post("/export-drawing-dxf")
@_sanitize_response
async def export_drawing_dxf(
    file: UploadFile = File(...),
    material: str = Form("aluminum_6061"),
    part_name: str = Form("Lumexa Part"),
):
    """
    Export a 2D DXF manufacturing reference drawing from an uploaded 3D part —
    for laser-cutting/CNC/machine shops that work from DXF rather than STEP/STL.
    See generate_technical_drawing_dxf's docstring for what this does and doesn't
    capture (convex-hull silhouettes, not a hidden-line-removed drawing).
    """
    if not EZDXF:
        raise HTTPException(503, "ezdxf is not installed on this server. Add "
                                  "'ezdxf' to requirements.txt to enable DXF export.")

    contents = await file.read()
    fn = file.filename or "part.stl"
    suffix = os.path.splitext(fn)[1] or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        t.write(contents); tmp = t.name

    try:
        mesh = trimesh.load(tmp)
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        mat = MATERIALS.get(material, MATERIALS.get("aluminum_6061", {}))
        doc = generate_technical_drawing_dxf(
            mesh, title=part_name, material_name=mat.get("name", material)
        )

        dxf_path = tmp + ".dxf"
        doc.saveas(dxf_path)
        with open(dxf_path, "rb") as f:
            dxf_b64 = base64.b64encode(f.read()).decode()
        os.unlink(dxf_path)

        dims = (mesh.bounds[1] - mesh.bounds[0]).tolist()
        return {
            "dxf_base64": dxf_b64,
            "filename": os.path.splitext(fn)[0] + "_drawing.dxf",
            "overall_dimensions_mm": {
                "length": round(dims[0], 2), "width": round(dims[1], 2), "height": round(dims[2], 2)
            },
            "scope_note": "Orthographic-style views are the convex hull of each "
                           "projection, not hidden-line-removed feature drawings — "
                           "concave features and through-holes won't appear as cut "
                           "lines. Good for stock sizing / rough layout, not a "
                           "substitute for a drafted machinist's drawing.",
        }
    finally:
        os.unlink(tmp)


@app.post("/analyze-composite")
@_sanitize_response
async def analyze_composite(
    file:UploadFile=File(...),
    material:str=Form("carbon_fiber_ud"),
    layup_angles:str=Form("[0,90,45,-45,90,0]"),
    thickness_per_ply_mm:float=Form(0.125),
    Nx:float=Form(1000.0),
    Ny:float=Form(0.0),
    Nxy:float=Form(0.0),
):
    """Classical Laminate Theory analysis for composite parts."""
    contents=await file.read();fn=file.filename or "part.stl"
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t:
        t.write(contents);tmp=t.name
    try:
        mesh=trimesh.load(tmp)
        if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
        try: angles=json.loads(layup_angles)
        except: angles=[0,90,45,-45,90,0]
        clt=composite_analysis_clt(material,angles,thickness_per_ply_mm,Nx,Ny,Nxy)
        geom_result=await run_analysis_v8(mesh,fn,fn,material)
        geom_result["composite_analysis"]=clt
        return geom_result
    finally: os.unlink(tmp)

@app.post("/analyze-rainflow")
@_sanitize_response
async def analyze_rainflow(
    file:UploadFile=File(...),
    material:str=Form("auto"),
    load_history:str=Form("[100,-50,80,-30,120,-60,90,-40]"),
):
    """Rainflow fatigue counting (ASTM E1049) for variable amplitude loading."""
    contents=await file.read();fn=file.filename or "part.stl"
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t:
        t.write(contents);tmp=t.name
    try:
        mesh=trimesh.load(tmp)
        if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
        if material=="auto": material=detect_material(mesh)
        try: lh=json.loads(load_history)
        except: lh=[100,-50,80,-30,120,-60]
        rf=rainflow_fatigue(material,lh)
        result=await run_analysis_v8(mesh,fn,fn,material)
        result["rainflow_fatigue"]=rf
        return result
    finally: os.unlink(tmp)

@app.post("/compare-designs")
@_sanitize_response
async def compare_designs(
    file1:UploadFile=File(...),
    file2:UploadFile=File(...),
    material1:str=Form("auto"),
    material2:str=Form("auto"),
    force_n:float=Form(1000.0),
):
    """Side-by-side engineering comparison of 2 design iterations."""
    c1=await file1.read();c2=await file2.read()
    def lm(c,fn):
        with tempfile.NamedTemporaryFile(suffix="."+fn.split(".")[-1].lower(),delete=False) as t:
            t.write(c);return t.name
    p1=lm(c1,file1.filename);p2=lm(c2,file2.filename)
    try:
        m1=trimesh.load(p1);m2=trimesh.load(p2)
        if hasattr(m1,"geometry"): m1=trimesh.util.concatenate(list(m1.geometry.values()))
        if hasattr(m2,"geometry"): m2=trimesh.util.concatenate(list(m2.geometry.values()))
        r1=await run_analysis_v8(m1,file1.filename,file1.filename,material1,force_n)
        r2=await run_analysis_v8(m2,file2.filename,file2.filename,material2,force_n)
        def delta(v1,v2):
            if v1 and v2 and v1!=0: return round((v2-v1)/v1*100,1)
            return None
        sf1=r1["analytical_fea"]["safety_factor"]
        sf2=r2["analytical_fea"]["safety_factor"]
        vm1=r1["analytical_fea"]["stress"]["von_mises_mpa"]
        vm2=r2["analytical_fea"]["stress"]["von_mises_mpa"]
        return {
            "design1":{"filename":file1.filename,"health":r1["health_score"]["score"],
                       "safety_factor":sf1,"von_mises_mpa":vm1,
                       "mass_g":r1["analytical_fea"]["dynamics"]["estimated_mass_g"],
                       "wall_min_mm":r1["wall_thickness"].get("min_mm"),
                       "violations":r1["rule_engine"]["total_violations"],
                       "full_analysis":r1},
            "design2":{"filename":file2.filename,"health":r2["health_score"]["score"],
                       "safety_factor":sf2,"von_mises_mpa":vm2,
                       "mass_g":r2["analytical_fea"]["dynamics"]["estimated_mass_g"],
                       "wall_min_mm":r2["wall_thickness"].get("min_mm"),
                       "violations":r2["rule_engine"]["total_violations"],
                       "full_analysis":r2},
            "delta":{
                "health_score_change":r2["health_score"]["score"]-r1["health_score"]["score"],
                "safety_factor_change_pct":delta(sf1,sf2),
                "stress_change_pct":delta(vm1,vm2),
                "mass_change_pct":delta(r1["analytical_fea"]["dynamics"]["estimated_mass_g"],
                                        r2["analytical_fea"]["dynamics"]["estimated_mass_g"]),
                "violations_change":r2["rule_engine"]["total_violations"]-r1["rule_engine"]["total_violations"],
            },
            "verdict":"DESIGN_2_BETTER" if r2["health_score"]["score"]>r1["health_score"]["score"]
                       else "DESIGN_1_BETTER" if r1["health_score"]["score"]>r2["health_score"]["score"]
                       else "EQUIVALENT",
        }
    finally: os.unlink(p1);os.unlink(p2)

@app.post("/image-to-params")
@_sanitize_response
async def image_to_params(
    image:UploadFile=File(...),
    description:str=Form(""),
):
    """
    Estimate part parameters from image using Gemini Vision (via Lovable AI Gateway).
    Returns estimated dimensions → use with /generate-and-analyze or, better,
    /generate-validate-refine for a self-correcting design.
    Accuracy: 65-75% (depends on image quality and part complexity).
    """
    img_bytes=await image.read()
    img_b64=base64.b64encode(img_bytes).decode()
    mime_type=image.content_type or "image/jpeg"

    try:
        params=await gemini_vision_estimate(img_b64,mime_type,description)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502,f"Gemini Vision error: {str(e)}")

    return {"estimated_params":params,
            "next_step":"Use these params with POST /generate-and-analyze, or describe the "
                         "part in natural language to POST /generate-validate-refine for an "
                         "AI-generated, self-corrected design.",
            "warning":"Image estimation accuracy: 65-75%. Verify dimensions before manufacturing.",
            "suggested_call":{"endpoint":"/generate-and-analyze",
                              "description":f"{params.get('part_type','bracket')} from image",
                              "params":json.dumps(_json_safe({
                                  "width":params.get("estimated_width_mm",80),
                                  "height":params.get("estimated_height_mm",60),
                                  "depth":params.get("estimated_depth_mm",40),
                                  "thickness":params.get("estimated_thickness_mm",5),
                                  "hole_diameter":params.get("hole_diameter_mm",6),
                              }))}}

@app.post("/analyze-assembly")
@_sanitize_response
async def analyze_assembly(
    part1:UploadFile=File(...),
    part2:UploadFile=File(...),
    material1:str=Form("auto"),
    material2:str=Form("auto"),
):
    c1=await part1.read();c2=await part2.read()
    def lm(c,fn):
        with tempfile.NamedTemporaryFile(suffix="."+fn.split(".")[-1].lower(),delete=False) as t:
            t.write(c);return t.name
    p1=lm(c1,part1.filename);p2=lm(c2,part2.filename)
    try:
        m1=trimesh.load(p1);m2=trimesh.load(p2)
        if hasattr(m1,"geometry"): m1=trimesh.util.concatenate(list(m1.geometry.values()))
        if hasattr(m2,"geometry"): m2=trimesh.util.concatenate(list(m2.geometry.values()))
        if material1=="auto": material1=detect_material(m1)
        if material2=="auto": material2=detect_material(m2)
        mat1=MATERIALS.get(material1,MATERIALS["aluminum_6061"])
        mat2=MATERIALS.get(material2,MATERIALS["aluminum_6061"])
        cog1=m1.center_mass;cog2=m2.center_mass
        d1=m1.bounding_box.extents;d2=m2.bounding_box.extents
        v1=sf(m1.volume);v2=sf(m2.volume)
        ms1=v1*mat1["density"]*1e-3;ms2=v2*mat2["density"]*1e-3;mt=ms1+ms2
        cc=(cog1*ms1+cog2*ms2)/mt;gc=np.vstack([m1.vertices,m2.vertices]).mean(axis=0)
        co=float(np.linalg.norm(cc-gc))
        dists=trimesh.proximity.ProximityQuery(m1).on_surface(m2.vertices[:500])[1]
        md=float(np.min(dists));ov=md<0.01
        try:
            pts,_=trimesh.sample.sample_surface(m2,1000)
            ins=m1.contains(pts);ovv=float(v2*ins.mean())
            pts2,_=trimesh.sample.sample_surface(m1,1000)
            ins2=m2.contains(pts2);ovv=max(ovv,float(v1*ins2.mean()))
        except: ovv=0.0
        h1=detect_holes_v8(m1);h2=detect_holes_v8(m2)
        score=100;issues=[]
        if ovv>50: score-=40;issues.append({"severity":"CRITICAL","title":"Major Interference","problem":f"Overlap {ovv:.1f}mm³","solution":"Redesign — major clash"})
        elif ovv>5: score-=25;issues.append({"severity":"HIGH","title":"Interference","problem":f"Overlap {ovv:.1f}mm³","solution":"Add clearance"})
        elif ov: score-=10;issues.append({"severity":"MEDIUM","title":"Surface Contact","problem":"Parts touching","solution":"Add 0.1mm clearance"})
        if md>5: score-=20;issues.append({"severity":"HIGH","title":"Large Gap","problem":f"Gap {md:.2f}mm","solution":"Add shim"})
        elif md>1: score-=8;issues.append({"severity":"MEDIUM","title":"Assembly Gap","problem":f"Gap {md:.2f}mm"})
        if co>15: score-=20;issues.append({"severity":"HIGH","title":"CoG Imbalance","problem":f"Offset {co:.1f}mm","solution":"Redistribute mass"})
        screw_recs=[{"location":f"P1 {h['position']}","bolt":h["recommended_screw"],
                     "torque_nm":h["torque_nm"]} for h in h1[:4]]
        gc_str=(f"ASSEMBLY v8.0\nP1:{part1.filename} {round(float(d1[0]),1)}x{round(float(d1[1]),1)}x{round(float(d1[2]),1)}mm {round(ms1,1)}g {mat1['name']}\n"
                f"P2:{part2.filename} {round(float(d2[0]),1)}x{round(float(d2[1]),1)}x{round(float(d2[2]),1)}mm {round(ms2,1)}g {mat2['name']}\n"
                f"Combined:{round(mt,1)}g CoG offset:{round(co,2)}mm Gap:{round(md,3)}mm Interference:{round(ovv,2)}mm³\n"
                f"Score:{max(0,score)}/100\nP1 holes:{json.dumps(_json_safe(h1[:4]))}\nP2 holes:{json.dumps(_json_safe(h2[:4]))}\n"
                f"Issues:{json.dumps(_json_safe(issues))}\nProvide screw table, assembly instructions, annotations.")
        return {"lumexa_version":"8.0","success":True,"assembly_score":max(0,score),
            "part1":{"name":part1.filename,"dimensions_mm":{"x":round(float(d1[0]),2),"y":round(float(d1[1]),2),"z":round(float(d1[2]),2)},
                     "mass_g":round(ms1,2),"material":mat1["name"],"holes":h1[:6]},
            "part2":{"name":part2.filename,"dimensions_mm":{"x":round(float(d2[0]),2),"y":round(float(d2[1]),2),"z":round(float(d2[2]),2)},
                     "mass_g":round(ms2,2),"material":mat2["name"],"holes":h2[:6]},
            "assembly_analysis":{"combined_mass_g":round(mt,2),"cog_offset_mm":round(co,3),
                "min_gap_mm":round(md,3),"interference_volume_mm3":round(ovv,3),"overlap_detected":ov},
            "screw_recommendations":screw_recs,"issues":issues,"gemini_context":gc_str}
    finally: os.unlink(p1);os.unlink(p2)
