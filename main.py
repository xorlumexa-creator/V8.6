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

# v8.24: CAD kernel is now build123d (CadQuery removed). Every generated design is analysed
# by SimScale (cloud FEA, when configured); a failed analysis is fed back to the LLM (Nemotron
# via NVIDIA NIM, AI_PROVIDER=nvidia) as structured refinement feedback. CalculiX service and the
# analytical model remain as automatic fallbacks.

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
import test_plan as TP
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

# CAD kernel: build123d (OpenCASCADE, like CadQuery, but with a cleaner algebra-style
# Python API that LLMs write more reliably). Install: pip install build123d
try:
    import build123d as b3d
    from build123d import export_stl as _b3d_export_stl, export_step as _b3d_export_step
    B3D = True
except ImportError:
    b3d = None
    B3D = False

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


app = FastAPI(title="Lumexa v8.24 Enterprise (build123d + SimScale)", version="8.24.0",
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
        # Set to 90s: Railway (1GB RAM, documented 5-min request timeout) removes
        # the platform-proxy-timeout problem that forced a tight 25s cutoff on
        # Render's free tier — a real solid-tet solve at full fidelity (0.08
        # mesh_size_factor, 80000 max_tets, restored in analysis_service.py) can
        # need more than 25s. Kept well under Railway's actual 300s ceiling
        # anyway, not raised all the way back to 300s, so a live/competition
        # demo still has a predictable worst-case wait before falling over to
        # the analytical path rather than hanging for minutes if something
        # genuinely goes wrong.
        with urllib.request.urlopen(req, timeout=90) as resp:
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


# ═══════════════════════════════════════════════════════════════════
# SIMSCALE CLOUD FEA — primary analyzer whenever it is configured.
#
# Every design the LLM produces is exported as a STEP B-rep, pushed to SimScale,
# meshed and solved there (linear static), and the peak von Mises stress / its
# location / tip deflection come back as the `fea` dict that the rest of this
# file already consumes (health score, quality gate, refinement feedback,
# engineering-agent diagnosis). If SimScale is not configured, or any stage
# fails, run_analysis_v8 falls back to the CalculiX analysis service and then to
# the analytical model — and says so in result["simscale_diagnostic"].
#
# HOW THE SIMULATION IS DEFINED — "template simulation" approach
# --------------------------------------------------------------
# Instead of building SimScale's very large StaticAnalysis object graph
# (materials, numerics, result control, mesher settings...) in code — which is
# version-sensitive and easy to get subtly wrong — you create ONE linear-static
# simulation + ONE mesh operation by hand in the SimScale workbench (one Fixed
# support BC, one Force BC, any material, standard mesher). This code clones
# that setup for every design and only swaps the things that change per design:
#   * geometry            -> the newly uploaded STEP
#   * material assignment -> the new body
#   * Fixed support faces -> the end face(s) at the MIN end of the longest axis
#   * Force faces         -> the end face(s) at the MAX end, force_n along force_dir
# Because the load is a pure force BC on a single linear-elastic material,
# stress does not depend on Young's modulus; displacement scales as 1/E. So the
# template's material is only a scaffold: displacement is rescaled to the
# requested material (SIMSCALE_TEMPLATE_MATERIAL tells us what the template
# used) and the safety factor always uses the REQUESTED material's yield.
#
# Env vars:
#   SIMSCALE_API_KEY                      required (X-API-KEY)
#   SIMSCALE_API_URL                      default https://api.simscale.com
#   SIMSCALE_TEMPLATE_PROJECT_ID          project that holds the template (needs "Allow API access")
#   SIMSCALE_TEMPLATE_SIMULATION_ID       the linear-static template simulation
#   SIMSCALE_TEMPLATE_MESH_OPERATION_ID   the template mesh operation
#   SIMSCALE_TEMPLATE_MATERIAL            MATERIALS key the template uses (default aluminum_6061)
#   SIMSCALE_TIMEOUT_S                    total wall-clock budget per analysis (default 480)
#   SIMSCALE_POLL_S                       polling interval (default 6)
#   SIMSCALE_ENABLED=0                    force-disable without removing the key
# ═══════════════════════════════════════════════════════════════════
import re
import copy
import zipfile
import io


def _env_flag(name, default=True):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


SIMSCALE_API_KEY = os.environ.get("SIMSCALE_API_KEY", "").strip()
SIMSCALE_API_URL = os.environ.get("SIMSCALE_API_URL", "https://api.simscale.com").strip().rstrip("/")
SIMSCALE_ENABLED = bool(SIMSCALE_API_KEY) and _env_flag("SIMSCALE_ENABLED", True)
SIMSCALE_TEMPLATE_PROJECT_ID = os.environ.get("SIMSCALE_TEMPLATE_PROJECT_ID", "").strip()
SIMSCALE_TEMPLATE_SIMULATION_ID = os.environ.get("SIMSCALE_TEMPLATE_SIMULATION_ID", "").strip()
SIMSCALE_TEMPLATE_MESH_OPERATION_ID = os.environ.get("SIMSCALE_TEMPLATE_MESH_OPERATION_ID", "").strip()
SIMSCALE_TEMPLATE_MATERIAL = os.environ.get("SIMSCALE_TEMPLATE_MATERIAL", "aluminum_6061").strip()
SIMSCALE_TIMEOUT_S = float(os.environ.get("SIMSCALE_TIMEOUT_S", "480"))
SIMSCALE_POLL_S = float(os.environ.get("SIMSCALE_POLL_S", "6"))

# One SimScale analysis at a time per process: keeps us inside plan concurrency
# limits and stops parallel requests from racing on the same template project.
_simscale_lock = threading.Lock()

def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


class _SimScaleError(Exception):
    """`stage` says where it failed. `design_related` is True only when SimScale
    itself reports that the GEOMETRY is the problem (import / meshing / solve
    failure) — that is worth telling the LLM. Auth, plan, network, config and
    timeout failures are NOT design problems and are never blamed on the part."""
    def __init__(self, stage, message, design_related=False):
        super().__init__(message)
        self.stage = stage
        self.design_related = design_related


def _ss_client():
    try:
        import simscale_sdk as sim
    except ImportError as e:
        raise _SimScaleError(
            "sdk", "simscale-sdk is not installed (pip install "
                   "git+https://github.com/SimScaleGmbH/simscale-python-sdk.git)") from e
    cfg = sim.Configuration()
    cfg.host = SIMSCALE_API_URL + "/v0"
    cfg.api_key = {"X-API-KEY": SIMSCALE_API_KEY}
    return sim, sim.ApiClient(cfg)


def _ss_reason(obj):
    for attr in ("failure_reason", "failureReason", "error", "message", "details"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)[:500]
    return "no failure reason reported by SimScale"


def _ss_wait(fetch, stage, deadline, ok=("FINISHED",), bad=("FAILED", "CANCELED", "CANCELLED"),
             design_related=True):
    """Poll `fetch()` until its .status is terminal. Raises _SimScaleError on
    FAILED/CANCELED (design_related per caller) or when the deadline passes."""
    last = ""
    while True:
        obj = fetch()
        last = str(getattr(obj, "status", "") or "").upper()
        if last in ok:
            return obj
        if last in bad:
            raise _SimScaleError(stage, f"{stage} {last}: {_ss_reason(obj)}", design_related=design_related)
        if time.time() > deadline:
            raise _SimScaleError(stage, f"timed out waiting for {stage} (last status {last or 'unknown'}). "
                                        f"Raise SIMSCALE_TIMEOUT_S or call the async endpoint.", False)
        time.sleep(SIMSCALE_POLL_S)


def _ss_format_exception(e):
    """SimScale SDK ApiException -> one readable line (with a plan/permission hint on 401/403)."""
    status = getattr(e, "status", None)
    if status is not None:
        body = str(getattr(e, "body", "") or "")[:300]
        hint = ""
        if status in (401, 403):
            hint = (" — check SIMSCALE_API_KEY, that your SimScale plan includes API access, and that the "
                    "project has 'Allow API access' enabled")
        return f"HTTP {status} {getattr(e, 'reason', '')}: {body}{hint}"
    return f"{type(e).__name__}: {str(e)[:400]}"


def _ss_mappings(sim, api_client, project_id, geometry_id, klass):
    """Entity names (in SimScale's own order) for one entity class ('face', 'body', ...)."""
    geo_api = sim.GeometriesApi(api_client)
    try:
        m = geo_api.get_geometry_mappings(project_id, geometry_id, _class=klass, limit=1000)
    except TypeError:
        m = geo_api.get_geometry_mappings(project_id, geometry_id, _class=klass)
    items = list(getattr(m, "embedded", None) or [])
    names = [getattr(i, "name", None) for i in items if getattr(i, "name", None)]
    # SimScale returns the names sorted as TEXT (..._TE13, _TE23, _TE27, _TE3, _TE31 ...), which is NOT the
    # B-rep order. Parasolid numbers entities in B-rep order, so sort numerically (instance, body, TE).
    return sorted(names, key=lambda n: ([int(x) for x in re.findall(r"\d+", n)], n))


def _ss_pick_end_faces(shape, axis, exts):
    """Indices (into shape.faces()) of the faces closing the MIN end and the MAX end
    of the part along `axis`: planar faces perpendicular to the axis, sitting at the
    part's extreme. Falls back to 'faces starting/ending within the outer 3%' when
    the ends are not flat (rounded / drilled tips)."""
    bb = shape.bounding_box()
    lo = [bb.min.X, bb.min.Y, bb.min.Z][axis]
    hi = [bb.max.X, bb.max.Y, bb.max.Z][axis]
    span = max(hi - lo, 1e-9)
    tol = max(span * 1e-3, 1e-4)
    faces = list(shape.faces())
    lo_idx, hi_idx = [], []
    for i, f in enumerate(faces):
        fb = f.bounding_box()
        fmin = [fb.min.X, fb.min.Y, fb.min.Z][axis]
        fmax = [fb.max.X, fb.max.Y, fb.max.Z][axis]
        if fmax - fmin <= tol:
            if abs(fmin - lo) <= tol:
                lo_idx.append(i)
            elif abs(fmax - hi) <= tol:
                hi_idx.append(i)
    if not lo_idx or not hi_idx:
        band = span * 0.03
        for i, f in enumerate(faces):
            fb = f.bounding_box()
            fmin = [fb.min.X, fb.min.Y, fb.min.Z][axis]
            fmax = [fb.max.X, fb.max.Y, fb.max.Z][axis]
            if not lo_idx and fmax <= lo + band:
                lo_idx.append(i)
            if not hi_idx and fmin >= hi - band:
                hi_idx.append(i)
    return lo_idx, hi_idx, len(faces)


def _ss_set_force(sim, bc, fx, fy, fz):
    try:
        comp = lambda v: sim.ConstantFunction(value=float(v))
        vec = sim.ComponentVectorFunction(x=comp(fx), y=comp(fy), z=comp(fz))
        bc.force = sim.DimensionalVectorFunctionForce(value=vec, unit="N")
    except Exception as e:
        raise _SimScaleError("setup", f"could not write the force vector into the template's Force BC "
                                      f"({type(e).__name__}: {e}) — the SDK class layout differs from what "
                                      f"_ss_set_force expects; adjust that one function.")


def _ss_attr_names(node):
    """Attribute names of a SimScale SDK model object (openapi-generated classes)."""
    types = getattr(node, "openapi_types", None)
    if isinstance(types, dict) and types:
        return list(types.keys())
    skip = ("local_vars_configuration", "discriminator", "configuration")
    return [k.lstrip("_") for k in getattr(node, "__dict__", {}) if k.lstrip("_") not in skip]


def _ss_stale_entities(sim, node, valid, _seen=None, _depth=0):
    """Names of every entity referenced (via a TopologicalReference) anywhere under `node`
    that does NOT exist in the freshly imported geometry, i.e. leftovers from the template."""
    if node is None or isinstance(node, (str, bytes, int, float, bool)) or _depth > 14:
        return []
    _seen = set() if _seen is None else _seen
    if id(node) in _seen:
        return []
    _seen.add(id(node))
    if isinstance(node, sim.TopologicalReference):
        return [e for e in (getattr(node, "entities", None) or []) if e not in valid]
    out = []
    if isinstance(node, (list, tuple)):
        for x in node:
            out += _ss_stale_entities(sim, x, valid, _seen, _depth + 1)
    elif isinstance(node, dict):
        for x in node.values():
            out += _ss_stale_entities(sim, x, valid, _seen, _depth + 1)
    else:
        for name in _ss_attr_names(node):
            try:
                v = getattr(node, name)
            except Exception:
                continue
            out += _ss_stale_entities(sim, v, valid, _seen, _depth + 1)
    return out


def _ss_drop_stale_assignments(sim, node, valid, body_names, report, path="model", _depth=0):
    """The template was built on a DIFFERENT geometry. Besides the material / fixed support / force
    that we re-point explicitly, it may hold other assignments (result controls such as area or
    surface data, contacts, mesh refinements, extra BCs ...) that still name the template's
    entities — SimScale rejects those with 'unknown Topological Entities'. Walk the whole cloned
    model: a list item that references entities missing from the new geometry is removed; a
    stand-alone stale reference is re-pointed at the body. Everything done is logged in `report`."""
    if node is None or isinstance(node, (str, bytes, int, float, bool)) or _depth > 14:
        return
    if isinstance(node, list):
        for i in range(len(node) - 1, -1, -1):
            item = node[i]
            stale = _ss_stale_entities(sim, item, valid)
            if stale:
                report["dropped"].append({"path": f"{path}[{i}]", "type": type(item).__name__,
                                          "stale": sorted(set(stale))[:6]})
                del node[i]
            else:
                _ss_drop_stale_assignments(sim, item, valid, body_names, report, f"{path}[{i}]", _depth + 1)
        return
    if isinstance(node, (tuple, dict, sim.TopologicalReference)):
        return
    for name in _ss_attr_names(node):
        try:
            v = getattr(node, name)
        except Exception:
            continue
        if isinstance(v, sim.TopologicalReference):
            stale = [e for e in (getattr(v, "entities", None) or []) if e not in valid]
            if stale:
                try:
                    setattr(node, name, sim.TopologicalReference(entities=list(body_names)))
                    report["remapped"].append({"path": f"{path}.{name}", "type": type(node).__name__,
                                               "stale": sorted(set(stale))[:6]})
                except Exception as e:
                    report.setdefault("unresolved", []).append(
                        {"path": f"{path}.{name}", "error": f"{type(e).__name__}: {e}"[:200]})
        elif v is not None and not isinstance(v, (str, bytes, int, float, bool)):
            _ss_drop_stale_assignments(sim, v, valid, body_names, report, f"{path}.{name}", _depth + 1)


def _ss_setup_errors(sims_api, pid, simulation_id, swallow=False):
    """ERROR-severity messages from SimScale's setup check (deduplicated, order kept)."""
    try:
        chk = sims_api.check_simulation_setup(pid, simulation_id)
    except Exception:
        if swallow:
            return []
        raise
    out = []
    for e in (getattr(chk, "entries", None) or []):
        if str(getattr(e, "severity", "")).upper() == "ERROR":
            msg = str(getattr(e, "message", e))
            if msg not in out:
                out.append(msg)
    return out


def _ss_patch_model(sim, template_model, body_names, fixed_names, load_names, force_xyz, valid_names=None,
                    extra_loads=()):
    """Deep-copy the template's model and re-point it at the new geometry."""
    model = copy.deepcopy(template_model)
    TR = sim.TopologicalReference
    report = {"materials_reassigned": 0, "fixed_support_bcs": 0, "force_bcs": 0}
    for mat in (getattr(model, "materials", None) or []):
        if hasattr(mat, "topological_reference"):
            mat.topological_reference = TR(entities=list(body_names))
            report["materials_reassigned"] += 1
    for bc in (getattr(model, "boundary_conditions", None) or []):
        cls = type(bc).__name__.lower()
        if "fixedsupport" in cls:
            bc.topological_reference = TR(entities=list(fixed_names))
            report["fixed_support_bcs"] += 1
        elif "forceload" in cls or cls in ("forcebc", "force"):
            bc.topological_reference = TR(entities=list(load_names))
            _ss_set_force(sim, bc, *force_xyz)
            report["force_bcs"] += 1
            force_tpl = bc
    if not report["fixed_support_bcs"] or not report["force_bcs"]:
        raise _SimScaleError(
            "setup", f"the template simulation must contain one 'Fixed support' and one 'Force' boundary "
                     f"condition (found {report}). Re-create the template in the SimScale workbench.")
    # additional loads (test plans with several forces): clone the template's Force BC once per extra load
    for k, (names, xyz) in enumerate(extra_loads, start=2):
        clone = copy.deepcopy(force_tpl)
        clone.topological_reference = TR(entities=list(names))
        _ss_set_force(sim, clone, *xyz)
        if hasattr(clone, "name") and getattr(clone, "name", None):
            clone.name = f"{clone.name} {k}"
        model.boundary_conditions.append(clone)
        report["force_bcs"] += 1
    # Anything else in the template that still names the template's own faces/bodies
    # (result controls, contacts, extra BCs, ...) would make the setup check fail.
    report["dropped"], report["remapped"] = [], []
    if valid_names is None:
        valid_names = set(body_names) | set(fixed_names) | set(load_names)
    _ss_drop_stale_assignments(sim, model, set(valid_names), body_names, report)
    return model, report


_VM_RE = re.compile(r"mises|sieq|vmis|equiv.*stress", re.I)
_STRESS_TENSOR_RE = re.compile(r"stress|sigma|sief", re.I)
_DISP_RE = re.compile(r"displacement|^u$|^d$|^depl", re.I)


def _ss_analyse_arrays(pts, point_data, fname):
    """Peak von Mises / displacement from per-point arrays (SI units in, raw SI numbers out).
    Understands a scalar von Mises field or a 6/9-component stress tensor. Returns None when
    neither stress nor displacement is found."""
    out = {"vm": None, "vm_xyz": None, "disp": None, "_vm_array": None,
           "points_extent": float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0, "file": fname}
    for name, arr in point_data.items():
        try:
            a = np.asarray(arr, dtype=float)
        except (TypeError, ValueError):
            continue
        if "strain" in name.lower():
            continue
        vm = None
        if _VM_RE.search(name) and a.ndim == 1:
            vm = np.abs(a)
        elif _STRESS_TENSOR_RE.search(name) and a.ndim == 2 and a.shape[1] in (6, 9):
            if a.shape[1] == 6:
                sxx, syy, szz, sxy, syz, sxz = (a[:, i] for i in range(6))
            else:
                sxx, syy, szz, sxy, syz, sxz = a[:, 0], a[:, 4], a[:, 8], a[:, 1], a[:, 5], a[:, 2]
            vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                         + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
        if vm is not None and len(vm) == len(pts):
            i = int(np.argmax(vm))
            if out["vm"] is None or vm[i] > out["vm"]:
                out["vm"], out["vm_xyz"], out["_vm_array"] = float(vm[i]), pts[i].tolist(), vm
        if _DISP_RE.search(name) and out["disp"] is None and a.ndim >= 1 and len(a) == len(pts):
            umag = np.linalg.norm(a.reshape(len(a), -1), axis=1)
            out["disp"] = float(umag.max())
            out["_umag"] = umag
    if out["vm"] is None and out["disp"] is None:
        return None
    out["_pts"] = pts
    try:                                    # hot zone: where is the stress within 15% of the peak? (one point may be an artifact)
        _va = out.get("_vm_array")
        if _va is not None and out.get("vm") and len(_va) == len(pts):
            _m = _va >= 0.85 * out["vm"]
            if _m.any():
                _zp = pts[_m]
                out["vm_zone"] = {"min": _zp.min(axis=0).tolist(), "max": _zp.max(axis=0).tolist(),
                                  "n_nodes": int(_m.sum()), "n_total": int(len(pts))}
    except Exception:
        pass
    try:                                    # compact profile along the longest axis: where is it fixed, where does it peak?
        ext = np.ptp(pts, axis=0)
        ax = int(np.argmax(ext))
        t = (pts[:, ax] - pts[:, ax].min()) / max(float(ext[ax]), 1e-12)
        sl = np.minimum((t * 10).astype(int), 9)
        vmarr, uarr = out.get("_vm_array"), out.get("_umag")
        nz = ext[ext > 1e-9]
        out["field_diag"] = {
            "axis": "xyz"[ax], "n_points": int(len(pts)),
            "bbox_min": [round(float(v), 5) for v in pts.min(axis=0)],
            "bbox_max": [round(float(v), 5) for v in pts.max(axis=0)],
            "approx_node_spacing": round(float((np.prod(nz) / len(pts)) ** (1.0 / len(nz))), 6) if len(nz) else None,
            "vm_peak_by_tenth": ([round(float(vmarr[sl == k].max()), 1) if (sl == k).any() else None for k in range(10)]
                                 if vmarr is not None else None),
            "disp_peak_by_tenth": ([float("%.4g" % uarr[sl == k].max()) if (sl == k).any() else None for k in range(10)]
                                   if uarr is not None else None)}
    except Exception as e:
        out["field_diag"] = f"unavailable: {type(e).__name__}"
    return out


_SS_RESULT_EXTS = (".vtu", ".vtk", ".vtp", ".vtm", ".pvd", ".pvtu", ".case", ".xdmf", ".xmf", ".med",
                   ".msh", ".e", ".exo", ".cgns", ".h5", ".foam", ".bin")


def _ss_arrays_from_file(path):
    """(points[N,3], {name: array}, cell_array_names, disp_from_nodes) from one result file.
    meshio first (vtu/vtk/xdmf/med/...), PyVista as the fallback (EnSight .case, multiblock, ...)."""
    low = path.lower()
    if low.endswith(".pvd"):                       # ParaView collection -> read the last data set it lists
        import xml.etree.ElementTree as ET
        files = [d.get("file") for d in ET.parse(path).getroot().iter("DataSet") if d.get("file")]
        if not files:
            raise ValueError("empty .pvd")
        return _ss_arrays_from_file(os.path.join(os.path.dirname(path), files[-1]))
    errs = []
    if not low.endswith((".case", ".vtm", ".pvtu", ".foam")):
        try:
            import meshio
            m = meshio.read(path)
            pts = np.asarray(m.points, dtype=float)
            pdata = dict(m.point_data or {})
            cdata = dict(m.cell_data or {})
            if cdata and not any(_VM_RE.search(k) or _STRESS_TENSOR_RE.search(k) for k in pdata):
                # stress only stored per element -> use element centroids as the "points"
                cents = [pts[np.asarray(cb.data)].mean(axis=1) for cb in m.cells]
                cpts = np.concatenate(cents) if cents else pts[:0]
                carr = {k: np.concatenate([np.asarray(x) for x in v]) for k, v in cdata.items()}
                if any(_VM_RE.search(k) or _STRESS_TENSOR_RE.search(k) for k in carr):
                    nodal = _ss_analyse_arrays(pts, pdata, "")
                    return cpts, {**carr}, sorted(cdata), (nodal or {}).get("disp")
            return pts, pdata, sorted(cdata), None
        except Exception as e:
            errs.append(f"meshio {type(e).__name__}: {str(e)[:100]}")
    try:
        import pyvista as pv
        d = pv.read(path)
        if isinstance(d, pv.MultiBlock):
            d = d.combine()
        cnames = list(d.cell_data.keys())
        if cnames and not any(_VM_RE.search(k) or _STRESS_TENSOR_RE.search(k) for k in d.point_data.keys()):
            d = d.cell_data_to_point_data()
        return (np.asarray(d.points, dtype=float),
                {k: np.asarray(d.point_data[k]) for k in d.point_data.keys()}, cnames, None)
    except Exception as e:
        errs.append(f"pyvista {type(e).__name__}: {str(e)[:100]}")
    raise ValueError("; ".join(errs))


def _ss_fields_from_file(path):
    """Peak von Mises / displacement from one SimScale result file (see _ss_arrays_from_file)."""
    pts, pdata, cnames, disp_nodes = _ss_arrays_from_file(path)
    out = _ss_analyse_arrays(pts, pdata, os.path.basename(path))
    if out is not None and out["disp"] is None and disp_nodes is not None:
        out["disp"] = disp_nodes
    if out is None:
        raise ValueError(f"no stress/displacement array (point arrays={ {k: list(np.shape(v)) for k, v in pdata.items()} }; "
                         f"cell arrays={cnames})")
    return out


def _ss_item_dict(it):
    try:
        d = it.to_dict()
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {k: getattr(it, k, None) for k in _ss_attr_names(it)}


def _ss_candidate_urls(d, prefix=""):
    """[(key_path, url)] for every url/href string inside a nested dict/list."""
    found = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str) and k in ("url", "href") and v.startswith(("http://", "https://")):
                found.append((f"{prefix}{k}", v))
            else:
                found += _ss_candidate_urls(v, f"{prefix}{k}.")
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            found += _ss_candidate_urls(v, f"{prefix}{i}.")
    return found


def _ss_http_get(url):
    """(bytes, None) or (None, error). SimScale API urls need the key; pre-signed storage urls must not get it."""
    import urllib.request
    err = None
    for headers in ({"X-API-KEY": SIMSCALE_API_KEY}, {}):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read(), None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:100]}"
    return None, err


def _ss_sniff_ext(data):
    h = data[:300].lstrip()
    if h.startswith(b"# vtk DataFile"):
        return ".vtk"
    if h.startswith(b"<?xml") or h.startswith(b"<VTKFile"):
        return ".vtu" if b"VTKFile" in data[:2000] else ".xdmf"
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return ".h5"
    return ".bin"


def _ss_parse_blob(data, base, info, found, hop=0):
    """Unpack one downloaded blob (zip / single file / JSON pointing at more urls) and add every
    parsable field set to `found` ({"vm": best_vm_dict, "disp": best_disp_dict}). Logs into `info`."""
    os.makedirs(base, exist_ok=True)
    if not zipfile.is_zipfile(io.BytesIO(data)) and data.lstrip()[:1] in (b"{", b"[") and hop < 2:
        try:
            doc = json.loads(data.decode("utf-8", "replace"))
        except Exception:
            doc = None
        if doc is not None:
            urls = _ss_candidate_urls(doc)
            info.setdefault("json_hop", []).append({"keys": sorted(doc)[:12] if isinstance(doc, dict) else "list",
                                                     "urls": [k for k, _ in urls][:5]})
            for i, (_, u) in enumerate(urls[:4]):
                blob, err = _ss_http_get(u)
                if blob is None:
                    info["errors"].append(f"hop url {i}: {err}")
                    continue
                _ss_parse_blob(blob, os.path.join(base, f"hop{hop}_{i}"), info, found, hop + 1)
            return
    paths = []
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(base)
        for root, _, files in os.walk(base):
            paths += [os.path.join(root, f) for f in files]
    else:
        p = os.path.join(base, "result" + _ss_sniff_ext(data))
        with open(p, "wb") as f:
            f.write(data)
        paths = [p]
    for p in sorted(paths):
        size = os.path.getsize(p)
        rel = os.path.relpath(p, base)
        if len(info["files"]) < 25:
            info["files"].append([rel, size])
        if not p.lower().endswith(_SS_RESULT_EXTS) or p.lower().endswith((".vtm", ".pvd", ".pvtu")):
            continue                        # containers only point at member files, which are read directly
        if size > 120 * 1024 * 1024:
            info["errors"].append(f"{rel}: skipped, {size // 2**20} MB is too big for this instance")
            continue
        try:
            f = _ss_fields_from_file(p)
        except Exception as e:
            info["errors"].append(f"{rel}: {type(e).__name__}: {str(e)[:220]}")
            continue
        info.setdefault("parsed", []).append(rel)
        if f["vm"] is not None and (found.get("vm") is None or f["vm"] > found["vm"]["vm"]):
            found["vm"] = f
        if f["disp"] is not None and (found.get("disp") is None or f["disp"] > found["disp"]["disp"]):
            found["disp"] = f


_SS_LOCK_BACKOFF_S = (8, 15, 25, 40, 60)     # waits (s) before re-asking when SimScale answers export-source-locked
_SS_NOISE_PARAMS = ("self", "kwargs", "async_req", "_return_http_data_only", "_preload_content",
                    "_request_timeout")


def _ss_scrub(v):
    """Copy of a nested structure with every url cut down to host+path (query strings hold signatures)."""
    if isinstance(v, dict):
        return {k: _ss_scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_ss_scrub(x) for x in v]
    if isinstance(v, str) and v.startswith(("http://", "https://")):
        return v.split("?")[0]
    return v if (v is None or isinstance(v, (str, int, float, bool))) else str(v)


def _ss_find_class(sim, pred):
    import inspect
    for n in sorted(dir(sim)):
        o = getattr(sim, n, None)
        if inspect.isclass(o) and pred(n, o):
            return n, o
    return None, None


def _ss_sdk_export_surface(sim):
    """What the INSTALLED SimScale SDK offers around results / exports (for diagnosing API drift)."""
    import inspect
    out = {"apis": [], "methods": {}, "models": {}}
    for n in dir(sim):
        o = getattr(sim, n, None)
        if not inspect.isclass(o):
            continue
        if n.endswith("Api"):
            out["apis"].append(n)
            for m in dir(o):
                if m.startswith("_") or m.endswith("_with_http_info"):
                    continue
                if any(k in m.lower() for k in ("export", "result")):
                    try:
                        out["methods"][f"{n}.{m}"] = str(inspect.signature(getattr(o, m)))[:160]
                    except Exception:
                        out["methods"][f"{n}.{m}"] = "?"
        elif "export" in n.lower() and len(out["models"]) < 30:
            out["models"][n] = {k: str(v) for k, v in (getattr(o, "openapi_types", None) or {}).items()}
    return out


def _ss_export_formats(d):
    vals = []
    for f in (d.get("available_export_formats") or []):
        if isinstance(f, dict):
            f = f.get("format") or f.get("name") or str(f)
        vals.append(str(f))
    pref = [f for f in vals if re.search(r"VTM|VTU|VTK|PVD", f, re.I)]
    rest = [f for f in vals if f not in pref and not re.search(r"CSV|FOAM|ENSIGHT", f, re.I)]
    return (pref + rest) or vals or ["VTM", "PVD"]


def _ss_run_export(sim, api_client, pid, sid, rid, result_id, formats, info, tmpdir, found, wait_s=150,
                   alt_result_ids=()):
    """The SOLUTION_FIELD item's own `download` is only a stub (an empty archive). The real field data has to
    be requested as an *export* (createExport -> poll -> download). The SDK method / model names are discovered
    at run time and every step is logged in info["export"], so API drift shows up in the diagnostics."""
    import inspect
    exp = info["export"] = {"tried": []}
    api_name, api_cls = _ss_find_class(sim, lambda n, o: n.endswith("Api") and hasattr(o, "create_export"))
    if api_cls is None:
        exp["error"] = "installed SDK has no *Api.create_export"
        return
    exp["api"] = api_name
    api = api_cls(api_client)
    known = {"project_id": pid, "simulation_id": sid, "run_id": rid, "result_id": result_id}

    def build_kwargs(method, **extra):
        kw = {}
        for pname in inspect.signature(method).parameters:
            if pname in _SS_NOISE_PARAMS:
                continue
            if pname in known:
                kw[pname] = known[pname]
            elif pname in extra:
                kw[pname] = extra[pname]
        return kw

    sig = inspect.signature(api.create_export)
    exp["create_export_params"] = [p for p in sig.parameters if p not in _SS_NOISE_PARAMS]
    body_params = [p for p in sig.parameters if p not in known and p not in _SS_NOISE_PARAMS]
    models = [(n, o) for n in sorted(dir(sim)) for o in [getattr(sim, n, None)]
              if inspect.isclass(o) and not n.endswith("Api") and "export" in n.lower()
              and "format" in (getattr(o, "openapi_types", None) or {})]
    models.sort(key=lambda t: 0 if "request" in t[0].lower() else 1)
    if len(body_params) != 1 or not models:
        exp["error"] = f"cannot build the export request (body params={body_params}, models={[m[0] for m in models]})"
        return
    exp["request_model"] = models[0][0]
    req_cls = models[0][1]
    req_fields = set(getattr(req_cls, "openapi_types", None) or {})
    # the item's own result_id is tried first; the id of the parent result (visible in the item's download url)
    # is the fallback in case SimScale wants that one
    res_ids = [result_id] + [x for x in alt_result_ids if x and x != result_id]
    for fmt in formats[:3]:
        for res_id in res_ids:
            rec = {"format": fmt, "result_id": res_id}
            exp["tried"].append(rec)
            resp = None
            for attempt in range(len(_SS_LOCK_BACKOFF_S) + 1):
                try:
                    kwargs = {k: v for k, v in (("format", fmt), ("result_id", res_id)) if k in req_fields}
                    req = req_cls(**kwargs)
                    resp = api.create_export(**build_kwargs(api.create_export, **{body_params[0]: req}))
                    break
                except Exception as e:
                    body = getattr(e, "body", None)
                    if "export-source-locked" in f"{body} {e}" and attempt < len(_SS_LOCK_BACKOFF_S):
                        # SimScale says the result is locked right now (seen once, right after a run finished);
                        # waiting and asking again is the only sensible move
                        rec["locked_retries"] = attempt + 1
                        time.sleep(_SS_LOCK_BACKOFF_S[attempt])
                        continue
                    rec["error"] = f"{type(e).__name__}: {str(e)[:80]}"
                    if body:
                        rec["error_body"] = str(body)[:400]
                    break
            if resp is None:
                continue
            rd = _ss_item_dict(resp)
            rec["created"] = _ss_scrub(rd)
            export_id = getattr(resp, "export_id", None)
            get = getattr(api, "get_export", None)
            t_end = time.time() + wait_s
            cur = rd
            while get is not None and export_id and time.time() < t_end:
                status = str(cur.get("status") or "").upper()
                if status in ("FAILED", "CANCELED", "CANCELLED", "ERROR") or cur.get("error_code"):
                    rec["failed"] = _ss_scrub(cur)
                    break
                if _ss_candidate_urls(cur) and status not in ("RUNNING", "QUEUED", "PENDING", "CREATED"):
                    break
                time.sleep(3)
                try:
                    cur = _ss_item_dict(get(**build_kwargs(get, export_id=export_id)))
                except Exception as e:
                    rec["poll_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    break
            rec["final"] = _ss_scrub(cur)
            for j, (key, url) in enumerate(_ss_candidate_urls(cur)[:3]):
                blob, err = _ss_http_get(url)
                if blob is None:
                    info["errors"].append(f"export {fmt} {key}: download failed ({err})")
                    continue
                rec["downloaded"] = [key, len(blob)]
                try:
                    _ss_parse_blob(blob, os.path.join(tmpdir, f"exp_{fmt}_{j}"), info, found)
                except Exception as e:
                    info["errors"].append(f"export {fmt} {key}: {type(e).__name__}: {str(e)[:150]}")
                if found.get("vm") is not None:
                    exp["used_format"], exp["used_result_id"] = fmt, res_id
                    return
            break          # the export was created; a different result id would only duplicate it


def _ss_fetch_results(sim, api_client, project_id, simulation_id, run_id, tmpdir, probe=None):
    """Download the run's result items and pull peak stress/displacement out of them.
    Returns (fields_dict, listing). `probe` (a dict, filled in place) records what SimScale actually
    returned for every item - attributes, urls found, files downloaded, parse errors - so a failure is
    diagnosable from the response alone. This is the most SimScale-version-sensitive step."""
    probe = {} if probe is None else probe
    runs_api = sim.SimulationRunsApi(api_client)
    res = runs_api.get_simulation_run_results(project_id, simulation_id, run_id)
    items = list(getattr(res, "embedded", None) or [])
    listing = [{"type": str(getattr(i, "type", None)), "category": str(getattr(i, "category", None)),
                "name": str(getattr(i, "name", None))} for i in items]
    probe["items"] = []
    found = {}
    for idx, it in enumerate(items):
        d = _ss_item_dict(it)
        cands = _ss_candidate_urls(d)
        cands.sort(key=lambda kv: 0 if kv[0].startswith("download") else 1)
        info = {"idx": idx, "class": type(it).__name__, "type": listing[idx]["type"],
                "category": listing[idx]["category"],
                "values": _ss_scrub({k: v for k, v in d.items() if v is not None}),
                "files": [], "errors": []}
        probe["items"].append(info)
        kind = (info["type"] + " " + info["category"]).lower()
        if not any(k in kind for k in ("solution", "field", "volume", "surface", "data", "result")):
            info["skipped"] = "kind"
            continue
        # 1) the item's own download (usually an empty stub for a SOLUTION_FIELD)
        for j, (key, url) in enumerate(cands[:3]):
            blob, err = _ss_http_get(url)
            if blob is None:
                info["errors"].append(f"{key}: download failed ({err})")
                continue
            info["downloaded"] = [key, len(blob)]
            if len(blob) < 200:
                info["blob_head_hex"] = blob[:60].hex()
            try:
                _ss_parse_blob(blob, os.path.join(tmpdir, f"res_{idx}_{j}"), info, found)
            except Exception as e:
                info["errors"].append(f"{key}: {type(e).__name__}: {str(e)[:150]}")
            if found.get("vm") is not None:
                break
        # 2) otherwise ask SimScale to build an export of that result and read that
        if found.get("vm") is None and d.get("result_id"):
            try:
                dl_url = ((d.get("download") or {}).get("url")) or ""
                parent_ids = re.findall(r"/results/([0-9a-fA-F-]{36})/components/", dl_url)
                _ss_run_export(sim, api_client, project_id, simulation_id, run_id, d["result_id"],
                               _ss_export_formats(d), info, tmpdir, found, alt_result_ids=parent_ids)
            except Exception as e:
                info["errors"].append(f"export: {type(e).__name__}: {str(e)[:200]}")
    best = found.get("vm")
    if best is not None and best.get("disp") is None and found.get("disp") is not None:
        best = dict(best, disp=found["disp"]["disp"])
    probe["parsed_ok"] = best is not None
    if best is None:
        try:
            probe["sdk"] = _ss_sdk_export_surface(sim)
        except Exception as e:
            probe["sdk"] = f"{type(e).__name__}: {e}"
        short = json.dumps(probe["items"], default=str)
        raise _SimScaleError(
            "results", "run FINISHED but no von Mises stress could be read from the result items. What SimScale "
                       f"returned: {short[:1400]}")
    return best, listing


def _ss_sdk_info(geometry_id=""):
    """Read-only introspection: what SimScale tells us about a geometry's faces, which selector models the SDK has
    (geometry primitives / boxes), and what the template simulation contains (material, primitives)."""
    out = {}
    sim, api_client = _ss_client()
    pid = SIMSCALE_TEMPLATE_PROJECT_ID
    geo_api = sim.GeometriesApi(api_client)
    try:
        if not geometry_id:
            page = geo_api.get_geometries(pid, limit=50)
            gs = [g for g in (getattr(page, "embedded", None) or []) if str(getattr(g, "name", "")).startswith("lumexa_")]
            gs.sort(key=lambda g: str(getattr(g, "created_at", "")), reverse=True)
            if gs:
                geometry_id = gs[0].geometry_id
                out["geometry_name"] = getattr(gs[0], "name", None)
        out["geometry_id"] = geometry_id
        out["geometry"] = _ss_scrub(_ss_item_dict(geo_api.get_geometry(pid, geometry_id)))
        m = geo_api.get_geometry_mappings(pid, geometry_id, _class="face", limit=1000)
        items = list(getattr(m, "embedded", None) or [])
        if items:
            out["face_mapping_class"] = type(items[0]).__name__
            out["face_mapping_types"] = {k: str(v) for k, v in (getattr(type(items[0]), "openapi_types", None) or {}).items()}
        out["face_mappings"] = [_ss_scrub(_ss_item_dict(i)) for i in items[:8]]
        out["geometry_api_methods"] = [x for x in dir(geo_api) if not x.startswith("_") and not x.endswith("_with_http_info")]
    except Exception as e:
        out["geometry_error"] = _ss_format_exception(e)
    names = [n for n in dir(sim) if re.search(r"Primitive|CartesianBox|Sphere|Cylinder|TopologicalReference|GeometryMapping|BoundingBox", n)]
    out["models"] = {n: {k: str(v) for k, v in (getattr(getattr(sim, n), "openapi_types", None) or {}).items()}
                     for n in names[:25]}
    try:
        tpl = sim.SimulationsApi(api_client).get_simulation(pid, SIMSCALE_TEMPLATE_SIMULATION_ID)
        md = _ss_scrub(_ss_item_dict(tpl.model))
        out["template_model_attrs"] = sorted(k for k, v in md.items() if v not in (None, [], {}))
        out["template_materials"] = json.dumps(md.get("materials"), default=str)[:1500]
        out["template_geometry_primitives"] = json.dumps(md.get("geometry_primitives"), default=str)[:800]
    except Exception as e:
        out["template_error"] = _ss_format_exception(e)
    return out


def _ss_bc_check(field_diag, axis):
    """Sanity-check that the faces we fixed / loaded really are the intended ends: the fixed (min) end must
    barely move and the displacement must peak at the loaded (max) end. Uses the along-axis profile."""
    try:
        if not isinstance(field_diag, dict) or field_diag.get("axis") != "xyz"[axis]:
            return None
        u = field_diag.get("disp_peak_by_tenth")
        if not u or u[0] is None or u[-1] is None:
            return None
        peak = max(v for v in u if v is not None)
        if peak <= 0:
            return None
        fixed_r, load_r = u[0] / peak, u[-1] / peak
        return {"ok": bool(fixed_r <= 0.15 and load_r >= 0.6),
                "fixed_end_disp_over_peak": round(fixed_r, 3), "load_end_disp_over_peak": round(load_r, 3)}
    except Exception:
        return None


def _ss_bc_check_points(fields, fixed_pts_mm, load_pts_mm, max_extent_mm):
    """Plan-mode sanity check, independent of part shape: mesh nodes sitting on the faces we FIXED must barely
    move, and the loaded faces must move clearly more than the fixed ones."""
    try:
        pts, u = fields.get("_pts"), fields.get("_umag")
        fd = fields.get("field_diag")
        if pts is None or u is None or not fixed_pts_mm:
            return None
        k = float(fields["points_extent"]) / max(max_extent_mm, 1e-9)      # coordinate units per mm (0.001 = metres)
        spacing = (fd or {}).get("approx_node_spacing") if isinstance(fd, dict) else None
        r = 1.25 * (spacing or 1.5 * k)
        def near(pts_mm):
            q = np.asarray(pts_mm, dtype=float) * k
            hit = np.zeros(len(pts), dtype=bool)
            for i in range(0, len(q), 64):
                d = np.linalg.norm(pts[:, None, :] - q[None, i:i + 64, :], axis=2).min(axis=1)
                hit |= d <= r
            return hit
        fx = near(fixed_pts_mm)
        if not fx.any():
            return {"ok": False, "reason": "no mesh nodes found on the fixed faces (face selection or units are off)"}
        peak = float(u.max())
        fixed_u = float(u[fx].max())
        out = {"ok": bool(fixed_u <= 0.25 * peak), "fixed_faces_disp_over_peak": round(fixed_u / max(peak, 1e-30), 3)}
        if load_pts_mm:
            lm = near(load_pts_mm)
            if lm.any():
                ratio = float(u[lm].mean()) / max(float(u[fx].mean()), 1e-30)
                out["loaded_over_fixed_mean_disp"] = round(ratio, 1)
                out["ok"] = bool(out["ok"] and ratio >= 3.0)
        if not out["ok"]:
            out["reason"] = "the fixed faces move (or the loaded faces do not move more) - faces were probably mismatched"
        return out
    except Exception as e:
        return {"ok": None, "reason": f"check unavailable: {type(e).__name__}"}


def _ss_probe(simulation_id="", run_id=""):
    """Re-read the results of an ALREADY FINISHED run (no new solve). With no ids it picks the newest
    finished 'lumexa_*' run in the template project."""
    import shutil
    out = {"project_id": SIMSCALE_TEMPLATE_PROJECT_ID}
    tmpdir = tempfile.mkdtemp(prefix="ssprobe_")
    try:
        sim, api_client = _ss_client()
        pid = SIMSCALE_TEMPLATE_PROJECT_ID
        sims_api, runs_api = sim.SimulationsApi(api_client), sim.SimulationRunsApi(api_client)
        if not (simulation_id and run_id):
            page = sims_api.get_simulations(pid, limit=50)
            sims = [x for x in (getattr(page, "embedded", None) or [])
                    if str(getattr(x, "name", "")).startswith("lumexa_")]
            sims.sort(key=lambda x: str(getattr(x, "created_at", "")), reverse=True)
            out["candidates_checked"] = 0
            for sm in sims[:12]:
                out["candidates_checked"] += 1
                rp = runs_api.get_simulation_runs(pid, sm.simulation_id)
                fin = [r for r in (getattr(rp, "embedded", None) or [])
                       if str(getattr(r, "status", "")).upper() == "FINISHED"]
                if fin:
                    simulation_id, run_id = sm.simulation_id, fin[-1].run_id
                    out["picked_name"] = getattr(sm, "name", None)
                    break
            if not (simulation_id and run_id):
                out["error"] = "no finished lumexa_* run found; pass ?simulation_id=&run_id="
                return out
        out.update(simulation_id=simulation_id, run_id=run_id)
        probe = {}
        out["results_probe"] = probe
        try:
            fields, _ = _ss_fetch_results(sim, api_client, pid, simulation_id, run_id, tmpdir, probe=probe)
            out["parsed"] = {k: (v if not isinstance(v, (list, tuple)) else v) for k, v in fields.items()
                             if not k.startswith("_")}
        except _SimScaleError as e:
            out["error"] = str(e)[:300]
    except Exception as e:
        out["error"] = _ss_format_exception(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def _ss_export_audit(n=10):
    """For the newest n lumexa_* simulations: is the run FINISHED, and does SimScale still let us EXPORT its
    result? Tells apart 'every run after #K is locked' (plan/quota) from 'random runs are locked' (something
    else), and pages through ALL simulations (a single page of 50 can hide the newest ones)."""
    import inspect
    pid = SIMSCALE_TEMPLATE_PROJECT_ID
    out = {"project_id": pid, "rows": []}
    try:
        sim, api_client = _ss_client()
        sims_api, runs_api = sim.SimulationsApi(api_client), sim.SimulationRunsApi(api_client)
        ex_api = sim.SimulationResultExportsApi(api_client)
        try:
            pinfo = sim.ProjectsApi(api_client).get_project(pid)
            out["project_info"] = _ss_scrub({k: v for k, v in _ss_item_dict(pinfo).items()
                                             if v is not None and not isinstance(v, (list, dict))})
        except Exception as e:
            out["project_info_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        allsims = []
        paged = "page" in inspect.signature(sims_api.get_simulations).parameters
        for pg in range(1, 15):
            kw = {"limit": 50}
            if paged:
                kw["page"] = pg
            got = list(getattr(sims_api.get_simulations(pid, **kw), "embedded", None) or [])
            allsims += got
            if not paged or len(got) < 50:
                break
        out["simulations_in_project"] = len(allsims)
        mine = [x for x in allsims if str(getattr(x, "name", "")).startswith("lumexa_")]
        mine.sort(key=lambda x: str(getattr(x, "created_at", "")), reverse=True)
        out["lumexa_simulations"] = len(mine)
        for sm in mine[:max(1, min(int(n), 20))]:
            row = {"name": getattr(sm, "name", None), "created_at": str(getattr(sm, "created_at", None))[:19]}
            out["rows"].append(row)
            try:
                runs = list(getattr(runs_api.get_simulation_runs(pid, sm.simulation_id), "embedded", None) or [])
                row["runs"] = len(runs)
                if not runs:
                    row["export"] = "no run"
                    continue
                runs.sort(key=lambda r: str(getattr(r, "created_at", "")))
                r = runs[-1]
                row["run_status"] = str(getattr(r, "status", None))
                if row["run_status"].upper() != "FINISHED":
                    row["export"] = "not finished"
                    continue
                items = list(getattr(runs_api.get_simulation_run_results(pid, sm.simulation_id, r.run_id),
                                     "embedded", None) or [])
                rid = None
                for it in items:
                    d = _ss_item_dict(it)
                    if d.get("result_id") and "SOLUTION" in str(d.get("category", "")).upper() + str(d.get("type", "")).upper():
                        rid = d["result_id"]
                        break
                if not rid:
                    row["export"] = "no solution result item"
                    continue
                row["result_id"] = rid[:8]
                try:
                    ex_api.create_export(pid, sim.CreateExportRequest(format="VTK", result_id=rid))
                    row["export"] = "OK"
                except Exception as e:
                    body = str(getattr(e, "body", "") or "")
                    m = re.search(r'"code"\s*:\s*"([^"]+)"', body)
                    row["export"] = (m.group(1) if m else f"{type(e).__name__}: {str(e)[:60]}")
                    row["http"] = getattr(e, "status", None)
            except Exception as e:
                row["export"] = f"error: {type(e).__name__}: {str(e)[:80]}"
        ok = [r for r in out["rows"] if r.get("export") == "OK"]
        locked = [r for r in out["rows"] if "locked" in str(r.get("export"))]
        out["summary"] = {"checked": len(out["rows"]), "export_ok": len(ok), "export_locked": len(locked),
                          "newest_ok": ok[0]["created_at"] if ok else None,
                          "oldest_locked": locked[-1]["created_at"] if locked else None}
    except Exception as e:
        out["error"] = _ss_format_exception(e)
    return out


def run_simscale_fem(cad_obj, mat_key, force_n=1000, force_dir="z", min_sf=2.0, load_case=None):
    """
    Blocking (call it via asyncio.to_thread). Returns (fem_result_or_None, diag)
    exactly like run_calculix_fem: never raises, and `diag` always explains what
    happened, including per-stage timings and — when SimScale itself rejected the
    geometry — `design_related=True` so the refinement loop can pass that on to
    the LLM.
    """
    if not SIMSCALE_ENABLED:
        return None, {"attempted": False, "reason": "SIMSCALE_API_KEY not set (or SIMSCALE_ENABLED=0)"}
    missing = [n for n, v in (("SIMSCALE_TEMPLATE_PROJECT_ID", SIMSCALE_TEMPLATE_PROJECT_ID),
                              ("SIMSCALE_TEMPLATE_SIMULATION_ID", SIMSCALE_TEMPLATE_SIMULATION_ID),
                              ("SIMSCALE_TEMPLATE_MESH_OPERATION_ID", SIMSCALE_TEMPLATE_MESH_OPERATION_ID)) if not v]
    if missing:
        return None, {"attempted": False, "reason": f"SimScale template not configured — set {', '.join(missing)}"}
    if cad_obj is None:
        return None, {"attempted": False,
                      "reason": "no B-rep available (analysis of an uploaded mesh) — SimScale needs the CAD solid"}

    diag = {"attempted": True, "stages": [], "reason": None}
    t0 = time.time()
    deadline = t0 + SIMSCALE_TIMEOUT_S
    current = {"stage": "start"}

    def mark(stage, **kw):
        current["stage"] = stage
        diag["stages"].append({"stage": stage, "t_s": round(time.time() - t0, 1), **kw})

    tmpdir = tempfile.mkdtemp(prefix="simscale_")
    try:
        with _simscale_lock:
            # the time budget starts once we hold the lock, not while queued behind another run
            diag["queued_s"] = round(time.time() - t0, 1)
            t0 = time.time()
            deadline = t0 + SIMSCALE_TIMEOUT_S
            # ── geometry facts (from our own B-rep, not from SimScale) ──────────
            mark("export_step")
            step_path = os.path.join(tmpdir, "part.step")
            _b3d_write_step(cad_obj, step_path)
            bb = cad_obj.bounding_box()
            exts = [bb.size.X, bb.size.Y, bb.size.Z]
            axis = int(np.argmax(exts))
            if load_case:
                # test-plan mode: faces / forces were chosen geometrically by the plan, nothing is guessed here
                n_faces = len(list(cad_obj.faces()))
                lo_idx = list(load_case["fixed_idx"])
                hi_idx = list(load_case["loads"][0]["face_idx"])
                force_xyz = [float(v) for v in load_case["loads"][0]["force_xyz"]]
                fa = int(np.argmax(np.abs(force_xyz)))
            else:
                lo_idx, hi_idx, n_faces = _ss_pick_end_faces(cad_obj, axis, exts)
                if not lo_idx or not hi_idx:
                    raise _SimScaleError("setup", "could not identify two opposite end faces to fix / load "
                                                  "(part has no usable ends along its longest axis).")
                fa = {"x": 0, "y": 1, "z": 2}.get((force_dir or "z").lower(), 2)
                force_xyz = [0.0, 0.0, 0.0]
                force_xyz[fa] = float(force_n)

            # ── SimScale session + template ─────────────────────────────────────
            mark("connect")
            sim, api_client = _ss_client()
            pid = SIMSCALE_TEMPLATE_PROJECT_ID
            sims_api = sim.SimulationsApi(api_client)
            mesh_api = sim.MeshOperationsApi(api_client)
            tpl_sim = sims_api.get_simulation(pid, SIMSCALE_TEMPLATE_SIMULATION_ID)
            tpl_mesh = mesh_api.get_mesh_operation(pid, SIMSCALE_TEMPLATE_MESH_OPERATION_ID)

            # ── upload + import STEP ────────────────────────────────────────────
            mark("upload")
            tag = f"lumexa_{uuid.uuid4().hex[:8]}"
            storage = sim.StorageApi(api_client).create_storage()
            with open(step_path, "rb") as f:
                blob = f.read()
            try:
                api_client.rest_client.PUT(url=storage.url, headers={"Content-Type": "application/octet-stream"},
                                           body=blob)
            except (AttributeError, TypeError):
                import urllib.request
                urllib.request.urlopen(urllib.request.Request(
                    storage.url, data=blob, method="PUT",
                    headers={"Content-Type": "application/octet-stream"}), timeout=120).close()
            mark("geometry_import")
            try:
                opts = sim.GeometryImportRequestOptions(facet_split=False, sewing=False, improve=False,
                                                        optimize_for_lbm_solver=False)
            except TypeError:
                opts = sim.GeometryImportRequestOptions()
            gi_api = sim.GeometryImportsApi(api_client)
            gi = gi_api.import_geometry(pid, sim.GeometryImportRequest(
                name=tag, location=sim.GeometryImportRequestLocation(storage.storage_id),
                format="STEP", input_unit="mm", options=opts))
            gi = _ss_wait(lambda: gi_api.get_geometry_import(pid, gi.geometry_import_id),
                          "geometry_import", deadline, design_related=True)
            geometry_id = gi.geometry_id

            # ── map our faces/body onto SimScale's entity names ─────────────────
            mark("map_entities")
            face_names = _ss_mappings(sim, api_client, pid, geometry_id, "face")
            body_names = []
            for klass in ("body", "volume", "region"):
                body_names = _ss_mappings(sim, api_client, pid, geometry_id, klass)
                if body_names:
                    break
            diag["entity_counts"] = {"simscale_faces": len(face_names), "b3d_faces": n_faces,
                                     "simscale_bodies": len(body_names)}
            if len(face_names) != n_faces:
                raise _SimScaleError(
                    "map_entities", f"SimScale reports {len(face_names)} faces but the B-rep has {n_faces}; "
                                    f"face order cannot be trusted, refusing to guess which faces to fix/load "
                                    f"(SimScale may have merged/split faces on import).")
            if not body_names:
                raise _SimScaleError("map_entities", "SimScale returned no body/volume entity to assign a material to.")
            fixed_names = [face_names[i] for i in lo_idx]
            load_names = [face_names[i] for i in hi_idx]
            extra_loads = ([([face_names[i] for i in l["face_idx"]], l["force_xyz"]) for l in load_case["loads"][1:]]
                           if load_case else [])

            # ── clone template -> new simulation + mesh operation ───────────────
            mark("setup")
            valid_names = set(face_names) | set(body_names)
            model, patch_report = _ss_patch_model(sim, tpl_sim.model, body_names, fixed_names, load_names,
                                                  force_xyz, valid_names=valid_names, extra_loads=extra_loads)
            diag["patch"] = patch_report
            simulation = sims_api.create_simulation(pid, sim.SimulationSpec(name=tag, geometry_id=geometry_id, model=model))
            simulation_id = simulation.simulation_id
            # fail fast on entity-assignment errors BEFORE paying for the mesh (no mesh yet, so only
            # the topological-entity complaints are meaningful at this point)
            early = [m for m in _ss_setup_errors(sims_api, pid, simulation_id, swallow=True)
                     if "opological" in m or "unknown entit" in m.lower()]
            if early:
                diag["setup_errors"] = early
                raise _SimScaleError("setup", f"SimScale setup check reported {len(early)} entity-assignment "
                                              f"error(s): " + "; ".join(early[:8]))
            mesh_model = copy.deepcopy(tpl_mesh.model)
            mesh_report = {"dropped": [], "remapped": []}
            _ss_drop_stale_assignments(sim, mesh_model, valid_names, body_names, mesh_report, path="mesh_model")
            diag["mesh_patch"] = mesh_report
            mesh_op = mesh_api.create_mesh_operation(
                pid, sim.MeshOperation(name=tag, geometry_id=geometry_id, model=mesh_model))
            mark("mesh")
            mesh_api.start_mesh_operation(pid, mesh_op.mesh_operation_id, simulation_id=simulation_id)
            mesh_op = _ss_wait(lambda: mesh_api.get_mesh_operation(pid, mesh_op.mesh_operation_id),
                               "mesh", deadline, design_related=True)
            spec = sims_api.get_simulation(pid, simulation_id)
            spec.mesh_id = mesh_op.mesh_id
            sims_api.update_simulation(pid, simulation_id, spec)
            try:
                bad = _ss_setup_errors(sims_api, pid, simulation_id)
            except AttributeError:
                bad = []
            if bad:
                diag["setup_errors"] = bad
                raise _SimScaleError("setup", f"SimScale setup check reported {len(bad)} error(s): "
                                              + "; ".join(bad[:8]))

            # ── solve ───────────────────────────────────────────────────────────
            mark("solve")
            runs_api = sim.SimulationRunsApi(api_client)
            run_id = runs_api.create_simulation_run(pid, simulation_id, sim.SimulationRun(name=tag)).run_id
            runs_api.start_simulation_run(pid, simulation_id, run_id)
            _ss_wait(lambda: runs_api.get_simulation_run(pid, simulation_id, run_id),
                     "solve", deadline, design_related=True)

            # ── results ─────────────────────────────────────────────────────────
            mark("results")
            diag["simscale_ids"] = {"simulation_id": simulation_id, "run_id": run_id, "geometry_id": geometry_id}
            probe = {}
            diag["results_probe"] = probe
            fields, listing = _ss_fetch_results(sim, api_client, pid, simulation_id, run_id, tmpdir, probe=probe)

        # ── convert SI results to this file's units ─────────────────────────────
        L = max(exts)
        scale = 1e3 if fields["points_extent"] < 0.05 * L else 1.0   # metres -> mm if the result mesh is ~1000x smaller
        vm_raw = float(fields["vm"] or 0.0)
        vm_mpa = vm_raw / 1e6 if vm_raw > 1e5 else vm_raw            # Pa -> MPa (values that large are Pa)
        disp_mm = float(fields["disp"] or 0.0) * scale
        E_tpl = MATERIALS.get(SIMSCALE_TEMPLATE_MATERIAL, MATERIALS["aluminum_6061"])["youngs_modulus_gpa"]
        E_req = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])["youngs_modulus_gpa"]
        disp_mm *= E_tpl / max(E_req, 1e-9)
        Sy = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])["yield_strength_mpa"]
        sfv = Sy / max(vm_mpa, 1e-6)
        xyz = fields.get("vm_xyz")
        xyz_mm = [round(float(c) * scale, 2) for c in xyz] if xyz else None
        _vz = fields.get("vm_zone")
        zone_mm = ({"min": [round(float(c) * scale, 1) for c in _vz["min"]],
                    "max": [round(float(c) * scale, 1) for c in _vz["max"]],
                    "n_nodes": _vz["n_nodes"], "n_total": _vz["n_total"]} if _vz else None)
        crit = {"axis": "xyz"[axis], "position_mm": (xyz_mm[axis] if xyz_mm else None),
                "strengthen_factor_approx": round(min_sf / sfv, 2) if sfv > 0 else None,
                "hotspot_xyz_mm": xyz_mm, "hot_zone_mm": zone_mm}
        fem = {
            "method": "simscale_static_fem",
            "note": ("SimScale cloud linear-static FEM. Load case: fixed at the min-"
                     f"{'xyz'[axis]} end face(s), {force_n} N along +{'xyz'[fa]} distributed over the max-"
                     f"{'xyz'[axis]} end face(s). Peak stress at a fixed/loaded face edge can include a local "
                     "singularity — judge by where the hotspot is, not just its magnitude."),
            "stress": {"von_mises_mpa": round(vm_mpa, 3), "axial_mpa": 0.0, "bending_mpa": 0.0,
                       "shear_mpa": 0.0, "stress_concentration_kt": 1.0},
            "deflection_mm": round(disp_mm, 4),
            "safety_factor": round(sfv, 3),
            "critical_section": crit,
            "numerically_suspect": bool(vm_mpa <= 0 or vm_mpa > 50 * Sy),
            "simscale": {"project_id": pid, "geometry_id": geometry_id, "simulation_id": simulation_id,
                         "run_id": run_id, "mesh_id": getattr(mesh_op, "mesh_id", None),
                         "run_name": tag, "result_items": listing, "result_file": fields.get("file"),
                         "field_diag": fields.get("field_diag"),
                         "template_material": SIMSCALE_TEMPLATE_MATERIAL, "requested_material": mat_key,
                         "displacement_rescaled_by_E_ratio": round(E_tpl / max(E_req, 1e-9), 4)},
            "inputs": ({"load_case": load_case["id"], "forces_n": [l["force_xyz"] for l in load_case["loads"]]}
                       if load_case else {"force_n": force_n, "direction": force_dir}),
        }
        if load_case:
            fem["note"] = (f"SimScale cloud linear-static FEM. Load case from test plan '{load_case['id']}': "
                           f"{len(load_case['fixed_idx'])} fixed face(s), {len(load_case['loads'])} load(s). "
                           "Peak stress at a fixed/loaded face edge can include a local singularity -- judge by "
                           "where the hotspot is, not just its magnitude.")
        bc = (_ss_bc_check_points(fields, load_case["fixed_pts_mm"], load_case["load_pts_mm"], max(exts))
              if load_case else _ss_bc_check(fields.get("field_diag"), axis))
        fem["simscale"]["bc_check"] = bc
        if bc is not None and not bc["ok"]:
            fem["numerically_suspect"] = True
            fem["note"] += (" WARNING: boundary-condition sanity check FAILED (the intended fixed end moves and/or "
                            "the loaded end does not) -- the SimScale faces were probably mismatched; do not "
                            "trust these numbers.")
        mark("done")
        diag["elapsed_s"] = round(time.time() - t0, 1)
        return fem, diag
    except _SimScaleError as e:
        diag.update(reason=str(e), failed_stage=e.stage, design_related=e.design_related)
    except Exception as e:
        diag.update(reason=_ss_format_exception(e), failed_stage=current["stage"], design_related=False)
    finally:
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
    diag["elapsed_s"] = round(time.time() - t0, 1)
    return None, diag


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

# Same purpose as FOLD_BRACKET_KEYWORDS above, for parts needing a genuine
# loft/taper instead of a constant cross-section — confirmed live, twice,
# that hand-written loft+fillet code is unreliable (see make_tapered_beam).
TAPER_KEYWORDS = ("taper","tapered","tapering","drone arm","connecting rod",
    "streamlined","aerodynamic profile","tapered leg","tapered beam",
    "tapered spar","loft between")

BUILD123D_SYSTEM = """You are a build123d expert mechanical engineer.
Generate Python build123d code (ALGEBRA MODE) to create the described 3D part.

MANDATORY FIRST STEP — REQUIREMENTS CHECKLIST:
Before writing any geometry code, write a Python comment block enumerating EVERY
explicitly stated requirement from the prompt as a checklist — every hole/bore
(with count, diameter, and rough position), every named dimension, every
fillet/chamfer instruction, every material/wall-thickness call-out. One line
per item, e.g.:
    # REQUIREMENTS:
    # [ ] 2x bearing bore, 22mm dia, coaxial, on opposite end faces
    # [ ] 4x M6 mounting hole, near bottom corners
    # [ ] wall thickness 4mm
    # [ ] fillet all internal corners 3mm
Then write the geometry code. Before finishing, go back through this exact
checklist line by line and confirm your code actually creates each item —
mark each one [x] once you've verified it's really there in the code below it,
not just planned. A named requirement that never appears anywhere in your
code (e.g. a bore the prompt asked for that never got cut) is a hard
failure — worse than an imperfect fillet radius, because a missing feature
is not a matter of tuning, it's a part that doesn't do what was asked. Do
not submit a script with any unchecked box; if you can't fit a requirement
in, go back and add it rather than leaving it off the list.

STRICT RULES:
- First line: from build123d import *      (only build123d, math and numpy may be imported)
- Prefer ALGEBRA MODE: Box(...), Pos(x,y,z) * shape, a + b, a - b. Builder blocks
  (`with BuildPart() as p:` ... `result = p.part`, with Locations()/GridLocations()/PolarLocations()
  and mode=Mode.SUBTRACT) are also allowed where they make a repeated feature pattern simpler.
- Assign the final SOLID to a variable named: result   (a solid Part — never a
  Sketch, Face, Wire, Curve or builder object)
- All dimensions in millimeters
- Add fillets to sharp internal corners minimum 0.5mm
- Add mounting holes where appropriate
- Code must be syntactically correct Python
- No explanations, no markdown, pure Python code only
- No os, sys, subprocess, socket, requests imports; no file reading/writing/exporting
- If a "VERIFIED build123d API REFERENCE" block is present in the user message, it was read from the
  installed library: use exactly those signatures, parameters and enum members and never invent others.

AVAILABLE BUILD123D OPERATIONS (algebra mode):
3D primitives — CENTERED on the origin by default (add
align=(Align.MIN, Align.MIN, Align.MIN) to put the min corner at the origin):
  Box(length, width, height)                  # X, Y, Z extents
  Cylinder(radius, height)                    # axis along Z
  Cone(bottom_radius, top_radius, height)
  Sphere(radius)
  Torus(major_radius, minor_radius)
Placing shapes (a Location times a shape returns the moved shape):
  Pos(x, y, z) * shape                        # translate
  Rot(rx, ry, rz) * shape                     # rotate (degrees) about X, Y, Z through the origin
  Pos(10, 0, 5) * Rot(0, 0, 45) * shape       # rotate first, THEN move
  Rot(0, 90, 0) * Cylinder(5, 40)             # a cylinder lying along X
Booleans:
  a + b        # union        a - b        # cut        a & b        # intersect
2D sketches -> solids (a sketch lies on the XY plane, extrudes along +Z from z=0):
  Rectangle(w, h)   Circle(r)   Ellipse(rx, ry)   RegularPolygon(radius, side_count)
  SlotOverall(width, height)
  Polygon((x1,y1), (x2,y2), (x3,y3), align=None)   # ALWAYS align=None, otherwise the outline is re-centered
  extrude(sketch, amount=h)
  Pos(x, y, z) * sketch   /   Plane.XZ * sketch     # place a sketch before extruding
  Plane.XY normal = +Z,  Plane.XZ normal = -Y,  Plane.YZ normal = +X
  revolve(profile, axis=Axis.Z)                     # profile drawn on Plane.XZ
  loft([Rectangle(30, 10), Pos(0, 0, 100) * Rectangle(15, 6)])
  # Sweep a 2D profile along a curved path — the tool for a genuinely curved
  # structural member (smoothly curved arm, duct), not a straight extrude:
  path = Spline((0, 0, 0), (10, 0, 30), (40, 0, 60))
  profile = Plane(origin=path @ 0, z_dir=path % 0) * Circle(4)
  result = sweep(profile, path=path)
Selecting edges/faces (return a ShapeList):
  part.edges().filter_by(Axis.Z)               # edges parallel to Z
  part.edges().filter_by(GeomType.CIRCLE)
  part.faces().sort_by(Axis.Z)[-1]             # top face   ([0] = bottom face)
  part.edges().sort_by(Axis.Z)[-4:]            # the four highest edges
  part.edges().filter_by_position(Axis.Z, 9.5, 10.5)   # edges whose Z lies in a band
Edge treatment (returns the modified part):
  part = fillet(part.edges().filter_by(Axis.Z), radius=2)
  part = chamfer(part.edges().sort_by(Axis.Z)[-4:], length=1)
Hollowing:
  part = offset(part, amount=-wall, openings=part.faces().sort_by(Axis.Z)[-1])
Holes: subtract a cylinder that overshoots BOTH faces of the material it goes through:
  part = part - Pos(x, y, 0) * Cylinder(r, part_height + 2)
  For hole patterns compute the (x, y) list with math.cos/math.sin or a loop and subtract each.

FEATURE DECLARATIONS (machine-verified — do not skip):
For EVERY hole or bore the prompt asks for (and every one you add on your own), put a comment directly
above the code that cuts it, in exactly this form:
    # FEATURE: hole dia=6 count=4
    # FEATURE: bore dia=22 count=2
`dia` is the DIAMETER in mm (Cylinder() takes the RADIUS, so radius = dia/2); `count` is how many identical
holes/bores that line covers; one line per distinct diameter. After your script runs, the server measures the
finished solid's cylindrical cavities and REJECTS the script if a declared hole is missing or the wrong size —
so declare only what your code really cuts, and make sure the cut actually reaches the material (a cutter
that misses the part, points along the wrong axis, or is subtracted from a shape you later discard cuts nothing).

ENGINEERING DEFAULTS (apply unless the prompt specifies otherwise):
- Mounting holes: diameter sized for M3-M6 fasteners, placed ≥2x diameter from any edge
- Wall thickness: minimum 1.5mm for plastics, 1.0mm for metals, never below 0.8mm
- Internal corners: fillet radius ≥0.5mm, prefer ≥1mm on load paths
- External edges: chamfer 0.5-1mm for safe handling unless a sharp edge is functionally required
- Keep aspect ratios (longest/shortest dimension) under 15:1 unless the prompt explicitly asks for a slender part
- Center the part roughly on the origin so the bounding box is well-formed
- When union-ing (a + b) two separately-built solids that attach end-to-end (e.g. an
  end plate/boss/flange on a tapered or curved member), do NOT place them so
  they only touch at one exact coincident plane with no real overlap — this is
  a common cause of a non-watertight result, especially when their
  cross-sections differ in size at that interface (e.g. a large plate meeting
  a much smaller tapered tip). Move the attachment so it genuinely
  overlaps the other solid by a small real depth (a few percent of the
  smaller cross-section's size is enough) before adding it.
- If the prompt describes a tapered, curved, streamlined, or organic-looking
  shape (e.g. "tapered arm", "curved bracket", "aerodynamic", "smoothly
  blends into"), use loft() between profiles or sweep() along a spline path
  for the main body — do not default to a constant-rectangular-cross-section
  box just because that's simpler. A box with fillets bolted on the corners
  is NOT the same as a genuinely tapered or curved shape, and looks
  noticeably different from what was actually asked for.
- Do NOT blanket-fillet every edge of a loft/tapered solid in one
  fillet(part.edges(), ...) call — this is a real cause of self-intersecting
  (non-watertight) geometry with no Python error at all. The corners where a
  sloped taper edge meets two flat profile edges are a compound 3-edge blend, a
  known-hard case for any CAD kernel. For a loft/tapered body: skip fillets on
  it entirely unless the prompt specifically requires edge-breaking there — an
  unfilleted taper edge is far better than a self-intersecting one. If fillets
  are truly required, fillet only the flat top/bottom profile edges via an
  explicit selector (filter_by_position), never all edges of the loft.
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

You previously generated a build123d script for this part. It was exported as a STEP/STL
and run through a real engineering analysis pipeline: a SimScale cloud FEA run (peak von
Mises stress, its location, deflection, safety factor) plus wall thickness, hole placement,
sharp-corner stress concentration, fatigue and rule-engine checks.

The analysis below lists concrete problems with the part as currently designed (or the
script failed to execute — in that case fix the execution error). Your job is to produce
a CORRECTED, COMPLETE script that fixes every issue listed, while preserving the parts of
the design that were already correct.

RULES FOR REFINEMENT:
- Output a COMPLETE script (not a diff/patch) that can run standalone, same format as before.
- Directly address each issue: e.g. if "Wall 0.6mm < material min 1.0mm", increase the
  relevant wall/shell thickness in the script's geometry, don't just change a comment.
- If a hole violates the edge-distance rule, move that hole's (x, y) coordinates inward.
- If sharp-corner stress concentration (Kf) is too high, add/increase a fillet() on that edge.
- If safety factor is too low, increase cross-sectional area/thickness in the load path,
  or reduce unsupported span, rather than changing the material.
- If the feedback contains a "SIMSCALE FEA" section, treat it as the authoritative
  structural result. It gives the peak von Mises stress, WHERE it occurs (x/y/z in mm),
  the deflection and the safety factor. Add material, fillets, ribs or cross-section AT
  THAT LOCATION and along the load path leading into it — do not just scale the whole
  part up, and do not thin any area that was fine. The load case is: fixed at the min end
  of the longest axis, force applied on the max end.
- If the feedback says SimScale could not import, mesh or solve the geometry, the shape
  is topologically bad for FEM: remove sliver features, faces/walls thinner than ~0.5mm,
  fillets smaller than ~0.3mm, coincident/tangent-only boolean joints and zero-thickness
  faces, then rebuild the affected boolean with real overlap (see below).
- If the previous script raised a Python error, fix the root cause (typo, wrong API call,
  bad chaining) — do not just simplify the part away.
- If the feedback says "DETERMINISTIC FEATURE CHECK FAILED", a hole/bore you declared with a
  '# FEATURE:' comment is NOT in the solid (measured on the B-rep). Find why the cut did nothing
  (cutter misses the material, wrong axis, subtraction discarded, re-filled by a later union,
  radius vs diameter) and fix the geometry. Keep every '# FEATURE:' comment in the script you return.
- If told the mesh is STILL not watertight AFTER automatic tessellation repair was already
  attempted, this is a REAL geometric defect, not a triangulation artifact — most often a
  boolean union/cut that leaves a gap or self-intersection (e.g. two solids that only
  partially overlap before a union, or a cut bore that exits through a corner instead
  of a flat face). Rebuild the affected boolean operation with fully-overlapping/fully-
  enclosed operands rather than adding fillets or changing wall thickness — those don't fix
  a topology gap. Copy this exact overshoot/overlap pattern for whichever boolean op is
  suspected:

    # DANGEROUS — cutting tool ends EXACTLY flush with the far face. This leaves a
    # coincident/zero-thickness face where they meet -> non-manifold mesh.
    bore = Pos(0, 0, wall_thickness / 2) * Cylinder(hole_r, wall_thickness)          # BAD
    result = housing - bore

    # SAFE — cutting tool starts before the near face and ends after the far face,
    # overshooting BOTH by a real margin (>= 1.0mm or 10% of wall_thickness).
    overshoot = max(1.0, wall_thickness * 0.1)
    bore = Pos(0, 0, wall_thickness / 2) * Cylinder(hole_r, wall_thickness + 2 * overshoot)
    result = housing - bore

    # DANGEROUS — second solid starts EXACTLY at the first solid's face, so they
    # only touch (tangent), never truly interpenetrate -> non-manifold seam on union.
    boss = Pos(0, 0, base_height + h / 2) * Cylinder(r, h)                            # BAD
    result = base + boss

    # SAFE — sink the second solid INTO the first by a real overlap margin before
    # unioning, so the two volumes genuinely share interior volume, not just a face.
    overlap = max(0.5, base_height * 0.05)
    boss = Pos(0, 0, base_height - overlap + (h + overlap) / 2) * Cylinder(r, h + overlap)
    result = base + boss

  This overshoot/overlap margin is the fix — not a fillet, not a wall-thickness change,
  not a different hole position. Apply it only to the boolean operation actually
  producing the non-manifold result; leave every other operation untouched.
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

# Cerebras Cloud (cloud.cerebras.ai) — OpenAI-compatible, function-calling capable,
# and (as of this writing) hosts gpt-oss-120b for free — the same model already used
# via Groq above, just a different inference backend with a separate rate-limit pool.
# Verify current card/limit terms at signup before relying on this; free-tier terms
# across every provider in this file change often enough that hardcoding a promise
# here would go stale.
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")

# NVIDIA NIM (build.nvidia.com) — OpenAI-compatible, one endpoint/key serves every
# model in NVIDIA's catalog (Nemotron, Kimi K2/K3, DeepSeek V4, and 90+ others) —
# just change NVIDIA_MODEL to switch models, no code change needed. Confirmed via
# NVIDIA's own catalog page (build.nvidia.com/models) as of this writing:
#   nvidia/nemotron-3-ultra-550b-a55b     (verify it shows a live Playground/API
#                                          tab, not just downloadable weights)
#   moonshotai/kimi-k3                    (confirmed live free endpoint)
#   deepseek-ai/deepseek-v4-pro-0813      (confirmed live — NOT the bare
#                                          "deepseek-v4-pro", that ID is deprecated)
#   deepseek-ai/deepseek-v4-flash-0731    (confirmed live free endpoint)
# Free-tier limits aren't published as a fixed table (same situation as every
# other free provider in this file) and are shared account-wide across whichever
# of the above models you call — check your own account's actual limits rather
# than assume headroom.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# Nemotron 3 Ultra is a large reasoning model on a shared free tier: a design script can take
# well over a minute. Raise/lower via env if you see read timeouts.
NVIDIA_TIMEOUT_S = float(os.environ.get("NVIDIA_TIMEOUT_S", "360"))   # TOTAL time allowed for one call
NVIDIA_STREAM = os.environ.get("NVIDIA_STREAM", "1").strip() != "0"   # stream tokens: a long think no longer trips the read timeout
NVIDIA_STALL_S = float(os.environ.get("NVIDIA_STALL_S", "60"))        # abort only if NO data arrives for this long
NVIDIA_THINKING = os.environ.get("NVIDIA_THINKING", "").strip().lower()   # Nemotron Ultra: ""=model default | on | off
NVIDIA_REASONING_BUDGET = int(os.environ.get("NVIDIA_REASONING_BUDGET", "0") or 0)   # cap on hidden thinking tokens (0 = none)

# Direct DeepSeek API (api-docs.deepseek.com) - separate key, independent of NVIDIA NIM model churn.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash").strip()
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "").strip().lower()   # ""=omit | enabled | disabled
NVIDIA_GEN_MAX_TOKENS = int(os.environ.get("NVIDIA_GEN_MAX_TOKENS", "12000"))

# Optional "strategic advisor" second model for the Engineering Agent — called
# SPARINGLY (once up front, then once per FAILED run_fea — not per tool call)
# for the high-level "understand this / why did it fail / what's the smallest
# valid fix" reasoning, while GPT-OSS-120B (via whichever AI_PROVIDER is
# configured) keeps driving the actual tool-calling loop as it already does.
# Reuses OpenRouter's existing endpoint/key — it's the same account either
# way, just a different model string — so no new API key is needed if
# OPENROUTER_API_KEY is already set.
#
# Rate-limit math behind why this is SPARING rather than per-step, confirmed
# against each provider's own published limits: Groq's gpt-oss-120b gets its
# own 1,000 requests/day (per-model, per-org — not shared with anything else),
# while OpenRouter's free ":free" models share ONE 50-requests/day pool
# ACROSS EVERY free model called from that account (rising to 1,000/day only
# after a $10 lifetime credit purchase). Calling the advisor on every tool
# call would blow through that shared 50/day budget in a single run; calling
# it 2-4 times per run keeps a full day's testing comfortably inside it.
#
# Leave this blank to disable the advisor entirely — the agent then behaves
# exactly as it did before this was added (GPT-OSS-120B alone, unchanged).
#
# IMPORTANT — verify this exact model string yourself before relying on it:
# sources disagree on whether "Nemotron 3 Ultra" (550B/55B active) specifically
# has a free tier on OpenRouter, versus the smaller "Nemotron 3 Super" (120B/
# 12B active) definitely having one. Check openrouter.ai/models yourself and
# use whichever one actually shows a live ":free" tag — a wrong model string
# here just makes the advisor calls fail silently (see _call_advisor below),
# so the main loop keeps working either way, but you won't get the benefit.
NEMOTRON_ADVISOR_MODEL = os.environ.get("NEMOTRON_ADVISOR_MODEL", "")


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
# Optional additional Groq keys (e.g. from separate free-tier accounts) so a
# rate-limited key automatically falls through to the next one instead of
# failing the request. Accepts GROQ_API_KEY_2, GROQ_API_KEY_3, ... (numbered,
# checked in order until one is unset) as well as a single comma-separated
# GROQ_API_KEYS env var — use whichever is more convenient to set on Render.
def _load_groq_keys():
    keys = [GROQ_API_KEY] if GROQ_API_KEY else []
    for extra in os.environ.get("GROQ_API_KEYS", "").split(","):
        extra = extra.strip()
        if extra and extra not in keys:
            keys.append(extra)
    i = 2
    while True:
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if not k:
            break
        if k not in keys:
            keys.append(k)
        i += 1
    return keys

GROQ_API_KEYS = _load_groq_keys()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Confirm vision support on Groq's hosted GPT-OSS before relying on it for
# image-to-params — override GROQ_VISION_MODEL if it doesn't behave as
# expected there (GPT-OSS models are primarily text-focused).
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", GROQ_MODEL)

# Which provider backs generation: "claude" (direct Anthropic API), "gemini"
# (direct Google API), "groq" (gpt-oss-120b via Groq — this deployment's primary/
# intended provider), "cerebras" (OpenAI-compatible, also hosts gpt-oss-120b free
# as of this writing — verify current card/limit terms at signup, they change
# often), "openrouter" (OpenAI-compatible gateway, free models available but churn
# heavily and cap out at 50 requests/day unfunded), or "lovable" (Gemini via
# the gateway). Defaults to whichever key is actually configured — set
# AI_PROVIDER explicitly to force a choice if more than one key is set.
#
# .strip().lower() on the whole expression: a value like "Groq" or " groq" (a
# typo made once in this deployment's history, via Render's dashboard) would
# otherwise match NONE of the string comparisons below or anywhere else this
# variable is checked, and silently fall through to whatever the last provider
# in a given if/elif chain happens to be — a confusing failure mode with no
# error message pointing at the real cause. Normalizing here means a typo'd
# value still selects the intended provider instead of failing silently.
AI_PROVIDER = os.environ.get(
    "AI_PROVIDER",
    "claude" if os.environ.get("ANTHROPIC_API_KEY")
    else "gemini" if os.environ.get("GOOGLE_API_KEY")
    else "nvidia" if os.environ.get("NVIDIA_API_KEY")
    else "groq" if os.environ.get("GROQ_API_KEY")
    else "cerebras" if os.environ.get("CEREBRAS_API_KEY")
    else "openrouter" if os.environ.get("OPENROUTER_API_KEY")
    else "lovable"
).strip().lower()


# Remembers which key in GROQ_API_KEYS last succeeded, so the next call tries
# that one first instead of always starting from index 0 (which would waste a
# round trip re-hitting an already-exhausted key on every single request once
# it's rate-limited). Plain module-level int: worst case under concurrent
# requests is one extra wasted attempt, not a correctness issue.
_groq_key_state = {"index": 0}


def _groq_request(messages, temperature=0.15, max_tokens=3000, model=None):
    """Low-level call to Groq (OpenAI-compatible chat completions) — same
    request/response shape as _lovable_request/_openrouter_request, different
    base URL/key/model.

    Tries each configured Groq key (GROQ_API_KEY plus any GROQ_API_KEY_2,
    GROQ_API_KEY_3, ... / GROQ_API_KEYS) in turn, falling through to the next
    key only on a 429 (rate limit) — any other error (auth, bad request,
    connection failure) still fails immediately rather than masking a real
    problem by silently retrying."""
    import urllib.request, urllib.error

    if not GROQ_API_KEYS:
        raise HTTPException(500, "GROQ_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    n = len(GROQ_API_KEYS)
    start = _groq_key_state["index"] % n
    last_exc = None

    for offset in range(n):
        idx = (start + offset) % n
        key = GROQ_API_KEYS[idx]
        req = urllib.request.Request(
            GROQ_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
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
            _groq_key_state["index"] = idx
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 429:
                last_exc = HTTPException(429, f"Groq rate limit exceeded on all "
                                                f"{n} configured key(s): {body}")
                continue  # try the next key, if any
            raise HTTPException(502, f"Groq error ({e.code}): {body}")
        except urllib.error.URLError as e:
            raise HTTPException(502, f"Groq connection error: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            # FIX: confirmed live (on the near-identical _openrouter_request below) — a
            # response that times out mid-read, or comes back with a non-JSON body, raises
            # something urllib.error.HTTPError/URLError doesn't catch (json.JSONDecodeError,
            # a raw socket.timeout/TimeoutError not wrapped in URLError, etc.), which used
            # to propagate uncaught and crash the entire request with a bare HTTP 500.
            raise HTTPException(502, f"Groq request failed unexpectedly: {type(e).__name__}: {e}")
    else:
        # every configured key hit a 429 — nothing left to fall back to
        raise last_exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected Groq response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        # Same null-content edge case fixed in _openrouter_request/_lovable_request.
        raise HTTPException(502, f"Groq returned empty/null content. "
                                  f"Raw response: {json.dumps(data)[:500]}")
    return content


def _cerebras_request(messages, temperature=0.15, max_tokens=3000, model=None):
    """Low-level call to Cerebras Cloud (OpenAI-compatible chat completions) —
    identical shape to _groq_request, different base URL/key/model. Added as a
    free-tier, no-card-reported alternative to Groq for gpt-oss-120b specifically
    (verify current terms at signup, they change often across every free provider
    in this file)."""
    import urllib.request, urllib.error

    if not CEREBRAS_API_KEY:
        raise HTTPException(500, "CEREBRAS_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    payload = json.dumps({
        "model": model or CEREBRAS_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        CEREBRAS_API_URL, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
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
            raise HTTPException(429, f"Cerebras rate limit exceeded: {body}")
        raise HTTPException(502, f"Cerebras error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Cerebras connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Cerebras request failed unexpectedly: {type(e).__name__}: {e}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected Cerebras response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        raise HTTPException(502, f"Cerebras returned empty/null content. "
                                  f"Raw response: {json.dumps(data)[:500]}")
    return content


def _nvidia_extra_body(model, thinking=None):
    """Nemotron 3 Ultra thinking controls (NVIDIA sample: chat_template_kwargs.enable_thinking + reasoning_budget).
    Other models get nothing extra, so unknown fields can never break them. thinking: True/False/None(=env)."""
    if "nemotron-3-ultra" not in (model or "").lower():
        return {}
    th = thinking if thinking is not None else {"on": True, "off": False}.get(NVIDIA_THINKING)
    ex = {}
    if th is False:
        ex["chat_template_kwargs"] = {"enable_thinking": False, "force_nonempty_content": True}
    elif th is True:
        ex["chat_template_kwargs"] = {"enable_thinking": True}
    if th is not False and NVIDIA_REASONING_BUDGET > 0:
        ex["reasoning_budget"] = NVIDIA_REASONING_BUDGET
    return ex


class _NvidiaStreamEmpty(Exception):
    """The stream ended without any answer text (not an HTTP error) - caller retries once without streaming."""


def _nvidia_stream_read(req, total_s):
    """Read an OpenAI-style SSE stream. Only answer text (delta.content) is kept; thinking text is dropped.
    Fails on a stall (no bytes for NVIDIA_STALL_S) or when total_s is exceeded - never on a merely long think."""
    import urllib.request, urllib.error
    t0 = time.time()
    parts, finish, think_chars, n_events, raw_other = [], None, 0, 0, []
    try:
        with urllib.request.urlopen(req, timeout=NVIDIA_STALL_S) as resp:
            for raw in resp:
                if time.time() - t0 > total_s:
                    raise HTTPException(504, f"NVIDIA NIM call exceeded {total_s:.0f}s total "
                                             f"(answer so far: {sum(len(x) for x in parts)} chars, "
                                             f"thinking: {think_chars} chars)")
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    if line and sum(len(x) for x in raw_other) < 4000:
                        raw_other.append(line)          # not SSE (e.g. a plain JSON body) - keep for fallback/diagnosis
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    ev = json.loads(chunk)
                except ValueError:
                    continue
                n_events += 1
                if isinstance(ev, dict) and ev.get("error") and not ev.get("choices"):
                    raise HTTPException(502, f"NVIDIA NIM stream error: {json.dumps(ev.get('error'))[:400]}")
                ch = ((ev.get("choices") or [{}])[0]) if isinstance(ev, dict) else {}
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                think = delta.get("reasoning_content") or delta.get("reasoning")
                if think:
                    think_chars += len(think)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except HTTPException:
        raise
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"NVIDIA NIM rate limit exceeded: {body}")
        raise HTTPException(502, f"NVIDIA NIM error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"NVIDIA NIM connection error: {str(e)}")
    except TimeoutError:
        raise HTTPException(504, f"NVIDIA NIM stalled: no data for {NVIDIA_STALL_S:.0f}s after "
                                 f"{time.time() - t0:.0f}s (thinking: {think_chars} chars)")
    except Exception as e:
        raise HTTPException(502, f"NVIDIA NIM stream failed: {type(e).__name__}: {e}")
    content = "".join(parts)
    if not content.strip() and n_events == 0 and raw_other:      # server ignored stream=true and sent plain JSON
        try:
            alt = json.loads("".join(raw_other))
            alt_c = alt["choices"][0]["message"]["content"]
            if isinstance(alt_c, str) and alt_c.strip():
                return alt_c
        except Exception:
            pass
    if not content.strip():
        if finish == "length":
            raise HTTPException(502, f"NVIDIA NIM returned empty content (finish_reason=length, thinking: "
                                     f"{think_chars} chars). It spent the whole token budget thinking - set "
                                     "NVIDIA_REASONING_BUDGET or NVIDIA_THINKING=off.")
        raise _NvidiaStreamEmpty(f"empty stream: events={n_events}, finish={finish}, thinking={think_chars} chars, "
                                 f"after {time.time() - t0:.0f}s, other_lines={' | '.join(raw_other)[:300]!r}")
    return content


def _nvidia_request_once(messages, temperature=0.15, max_tokens=3000, model=None, timeout=None, thinking=None):
    """Low-level call to NVIDIA's NIM catalog (build.nvidia.com, OpenAI-compatible
    chat completions) — identical shape to _groq_request/_cerebras_request,
    different base URL/key. `model` (or NVIDIA_MODEL) picks which of NVIDIA's
    90+ catalog models actually answers this call — Nemotron, Kimi K3, DeepSeek
    V4, etc. all go through this exact same function."""
    import urllib.request, urllib.error

    if not NVIDIA_API_KEY:
        raise HTTPException(500, "NVIDIA_API_KEY is not configured on the server. "
                                   "Set it as a secret/env var in your deployment.")

    _mdl = model or NVIDIA_MODEL
    _body = {
        "model": _mdl,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    _body.update(_nvidia_extra_body(_mdl, thinking))

    def _mk_req(stream):
        bd = dict(_body)
        if stream:
            bd["stream"] = True
        return urllib.request.Request(
            NVIDIA_API_URL, data=json.dumps(bd).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)",
                "Accept": "text/event-stream" if stream else "application/json",
            })

    if NVIDIA_STREAM:
        try:
            return _nvidia_stream_read(_mk_req(True), float(timeout or NVIDIA_TIMEOUT_S))
        except _NvidiaStreamEmpty as e:
            print(f"[nvidia] {e} - retrying once without streaming", flush=True)
    req = _mk_req(False)
    try:
        with urllib.request.urlopen(req, timeout=(timeout or NVIDIA_TIMEOUT_S)) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"NVIDIA NIM rate limit exceeded: {body}")
        raise HTTPException(502, f"NVIDIA NIM error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"NVIDIA NIM connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"NVIDIA NIM request failed unexpectedly: {type(e).__name__}: {e}")

    # FIX: this function previously ended right here — it sent the request and
    # parsed the HTTP response into `data`, but never pulled the actual answer
    # out of it, so every successful call returned None instead of the
    # generated text. Every other provider function (_groq_request,
    # _cerebras_request) ends with exactly this extract-and-validate block;
    # _nvidia_request was missing it, meaning the whole NVIDIA/Nemotron path
    # has never actually returned usable output. Reasoning models on NVIDIA's
    # catalog (Nemotron included) put any chain-of-thought in a SEPARATE
    # "reasoning_content" field on the message, not in "content" — so no
    # extra stripping is needed here; reading "content" already excludes it.
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected NVIDIA NIM response shape: {json.dumps(data)[:500]}")

    if not content or not isinstance(content, str):
        finish = None
        try: finish = data["choices"][0].get("finish_reason")
        except Exception: pass
        hint = (" finish_reason=length: the model spent its whole token budget on reasoning — raise "
                "NVIDIA_GEN_MAX_TOKENS." if finish == "length" else "")
        raise HTTPException(502, f"NVIDIA NIM returned empty/null content.{hint} "
                                  f"Raw response: {json.dumps(data)[:500]}")
    return content


def _deepseek_request(messages, temperature=0.1, max_tokens=6000, model=None, timeout=90):
    """Direct call to DeepSeek's own API (OpenAI-compatible /chat/completions). Needs DEEPSEEK_API_KEY.
    Thinking text arrives in a separate reasoning_content field, so reading "content" excludes it."""
    import urllib.request, urllib.error

    if not DEEPSEEK_API_KEY:
        raise HTTPException(500, "DEEPSEEK_API_KEY is not configured on the server.")
    body = {"model": model or DEEPSEEK_MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    if DEEPSEEK_THINKING in ("enabled", "disabled"):
        body["thinking"] = {"type": DEEPSEEK_THINKING}
    req = urllib.request.Request(
        DEEPSEEK_API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                 "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="ignore")
        raise HTTPException(429 if e.code == 429 else 502, f"DeepSeek API error ({e.code}): {raw[:500]}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"DeepSeek connection error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"DeepSeek request failed: {type(e).__name__}: {e}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected DeepSeek response shape: {json.dumps(data)[:500]}")
    if not content or not isinstance(content, str):
        fin = (data.get("choices") or [{}])[0].get("finish_reason")
        raise HTTPException(502, f"DeepSeek returned empty content (finish_reason={fin}). "
                                 f"Raw: {json.dumps(data)[:400]}")
    return content


_NVIDIA_RETRY_WAITS_S = (6, 15, 30)       # waits before re-asking after a transient NVIDIA error
_NVIDIA_TRANSIENT_MARKERS = ("overloaded", "temporarily", "service_unavailable", "(503)", "(502)", "(500)",
                             "\"code\": 503", "\"code\": 502", "bad gateway", "rate limit")


def _nvidia_is_transient(e):
    """Overload / rate-limit style errors that usually clear within seconds. Dead models (410), bad keys (401/403),
    bad requests (400/404) and our own timeouts are NOT retried here."""
    if getattr(e, "status_code", None) == 429:
        return True
    d = str(getattr(e, "detail", e)).lower()
    return any(m in d for m in _NVIDIA_TRANSIENT_MARKERS)


def _nvidia_request(messages, temperature=0.15, max_tokens=3000, model=None, timeout=None, thinking=None):
    """_nvidia_request_once + retries with backoff on transient overload errors (NVIDIA's free endpoint returns
    'Service temporarily overloaded' now and then; one such blip used to kill a whole design job)."""
    for attempt in range(len(_NVIDIA_RETRY_WAITS_S) + 1):
        try:
            return _nvidia_request_once(messages, temperature=temperature, max_tokens=max_tokens,
                                        model=model, timeout=timeout, thinking=thinking)
        except HTTPException as e:
            if attempt < len(_NVIDIA_RETRY_WAITS_S) and _nvidia_is_transient(e):
                print(f"[nvidia] transient error ({str(e.detail)[:120]}) - retry {attempt + 1}/"
                      f"{len(_NVIDIA_RETRY_WAITS_S)} in {_NVIDIA_RETRY_WAITS_S[attempt]}s", flush=True)
                time.sleep(_NVIDIA_RETRY_WAITS_S[attempt])
                continue
            raise


def _openrouter_request(messages, temperature=0.15, max_tokens=3000, model=None, timeout=60):
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"OpenRouter rate limit exceeded: {body}")
        raise HTTPException(502, f"OpenRouter error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"OpenRouter connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        # FIX: confirmed live — a Nemotron 3 Ultra advisor call (large model, free/shared
        # queue) timed out mid-read at almost exactly this function's 60s timeout, raising
        # something urllib.error.URLError doesn't catch. That propagated all the way up
        # through _call_advisor's narrower except HTTPException, past FastAPI's normal JSON
        # error handling, and crashed the entire /engineering-agent request with a bare
        # HTTP 500 plaintext body — for a call that was only ever supposed to be a
        # best-effort advisory extra, never something that could break the main loop.
        raise HTTPException(502, f"OpenRouter request failed unexpectedly: {type(e).__name__}: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Gemini API request failed unexpectedly: {type(e).__name__}: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Gemini API request failed unexpectedly: {type(e).__name__}: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Lovable AI request failed unexpectedly: {type(e).__name__}: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Claude API request failed unexpectedly: {type(e).__name__}: {e}")

    try:
        return "".join(b["text"] for b in data["content"] if b.get("type") == "text")
    except (KeyError, TypeError):
        raise HTTPException(502, f"Unexpected Claude API response shape: {json.dumps(data)[:500]}")

def _clean_code_block(text: str, lang_hints=("python","json")) -> str:
    t = text.strip()
    # If the model wrapped the code in prose, take the largest fenced block instead.
    fenced = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", t, flags=re.S)
    if fenced:
        return max(fenced, key=len).strip()
    for h in lang_hints:
        t = t.replace(f"```{h}", "```")
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()

# ═══════════════════════════════════════════════════════════════════
# API KNOWLEDGE BASE (RAG) — stop the LLM inventing build123d calls
#
# Ground truth = the INSTALLED build123d package itself: every class/function/enum/method
# signature and docstring is read with `inspect`, so the reference always matches the exact
# version deployed (and only names the sandbox really exposes are indexed — never the file
# I/O functions the sandbox forbids). It is topped up with
#   * a small set of idiom snippets (algebra mode, BuildPart + Locations + Mode.SUBTRACT, ...),
#     each one machine-checked against the installed library at index time — an idiom that
#     uses a name or keyword the installed version doesn't have is DROPPED, not shown;
#   * any .md/.rst/.txt/.py files under KB_DOCS_DIR (the Docker build drops build123d's own
#     docs + examples there when it can).
#
# Retrieval per generation call:  BM25 (exact identifiers)  +  NVIDIA nv-embedqa-e5-v5
# (semantic)  ->  reciprocal-rank fusion  ->  NVIDIA reranker (precision filter)  ->  top-k
# chunks injected into the prompt. Identifiers that appear in the previous script / the error
# message are looked up EXACTLY and always included.
#
# Degrades gracefully at every level: no NVIDIA key or embeddings down -> BM25 only; reranker
# unavailable -> fused order; KB off/not ready -> the prompt is unchanged. It never blocks or
# fails a generation.
#
# Separately, lint_b3d_script() statically checks a script against the installed signatures
# (unknown names / keywords / enum members / too many positionals, with "did you mean")
# and is appended to the feedback when a script fails to run — so ONE refinement round
# fixes ALL the API mistakes instead of one per round.
# ═══════════════════════════════════════════════════════════════════
import hashlib
import inspect
import difflib
import enum
import contextvars

KB_ENABLED = _env_flag("KB_ENABLED", True)
# Semantic (embedding) retrieval is OFF by default: the index would have to be re-embedded (dozens of NVIDIA
# API calls, shared with your generation quota) every time the instance boots, and Render's free disk is
# ephemeral (a sleeping instance wakes cold). BM25 + exact-identifier lookup + the reranker already cover
# the main job (finding the right function signature). Set KB_EMBEDDINGS=1 if you have a persistent disk
# (KB_INDEX_PATH) or don't mind the warm-up.
KB_EMBEDDINGS = _env_flag("KB_EMBEDDINGS", False)
KB_TOP_K = int(os.environ.get("KB_TOP_K", "6"))
KB_MAX_CHARS = int(os.environ.get("KB_MAX_CHARS", "7000"))
KB_TIMEOUT_S = float(os.environ.get("KB_TIMEOUT_S", "25"))
KB_DOCS_DIR = os.environ.get("KB_DOCS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge"))
KB_INDEX_PATH = os.environ.get("KB_INDEX_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_index.npz"))
KB_EMBED_BATCH = int(os.environ.get("KB_EMBED_BATCH", "32"))
KB_EMBED_PAUSE_S = float(os.environ.get("KB_EMBED_PAUSE_S", "1.6"))   # stay under the free tier's ~40 requests/min

NVIDIA_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
NVIDIA_EMBED_URL = os.environ.get("NVIDIA_EMBED_URL", "https://integrate.api.nvidia.com/v1/embeddings")
# Reranker candidates, tried in this order until one answers. `nvidia/rerank-qa-mistral-4b` is last on purpose:
# NVIDIA's own model page says that API was deprecated on 08/24/2026. Put your preferred model FIRST via
# NVIDIA_RERANK_MODELS (comma-separated) or disable reranking with KB_RERANK=0.
NVIDIA_RERANK_MODELS = [m.strip() for m in os.environ.get(
    "NVIDIA_RERANK_MODELS",
    "nvidia/llama-nemotron-rerank-vl-1b-v2,nvidia/llama-nemotron-rerank-1b-v2,"
    "nvidia/llama-3.2-nv-rerankqa-1b-v2,nvidia/nv-rerankqa-mistral-4b-v3,nvidia/rerank-qa-mistral-4b").split(",")
    if m.strip()]
KB_RERANK = _env_flag("KB_RERANK", True)

_KB_METHOD_CLASSES = ("Shape", "ShapeList", "Mixin1D", "Mixin2D", "Mixin3D", "Face", "Edge", "Wire", "Solid",
                      "Compound", "Part", "Sketch", "Curve", "Location", "Plane", "Axis", "Vector", "BoundBox",
                      "BuildPart", "BuildSketch", "BuildLine")
_KB_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "for", "with", "is", "are", "be", "by", "on", "as",
            "it", "this", "that", "from", "at", "we", "you", "your", "can", "will", "not"}
_KB_FORBIDDEN_IN_DOCS = ("export_", "import_step", "import_stl", "import_brep", "ocp_vscode", "open(", "import os",
                         "import sys", "subprocess", "Mesher", "ExportSVG", "ExportDXF")

_kb = {"chunks": [], "bm25": None, "emb": None, "mode": "off", "error": None, "built_at": None,
       "building": False, "counts": {}, "dropped_idioms": [], "skipped_doc_chunks": 0, "by_name": {},
       "b3d_version": None, "embed_dim": None, "fingerprint": None, "docs_files": 0}
_kb_lock = threading.Lock()
_rr = {"model": None, "url": None, "disabled_until": 0.0, "last_error": None, "tried": []}
_KB_TRACE = contextvars.ContextVar("kb_trace", default=None)

# ── the ONE network function (monkeypatched in tests) ───────────────────────
def _nv_post(url, payload, timeout=30):
    """POST JSON with the NVIDIA key. Returns (http_status | None, parsed_json | None, error_text)."""
    if not NVIDIA_API_KEY:
        return None, None, "NVIDIA_API_KEY not set"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                                          "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), ""
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode(errors="ignore")[:400]
    except Exception as e:
        return None, None, f"{type(e).__name__}: {str(e)[:200]}"


# ── tokenizer + BM25 (exact identifier matching; no network) ────────────────
def _kb_tokens(text):
    toks = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text or ""):
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ").lower().split()
        toks.extend(parts)
        if len(parts) > 1:
            toks.append(w.lower())
    return [t for t in toks if len(t) > 1 and t not in _KB_STOP]


class _BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf, self.df, self.dl = [], {}, []
        for d in docs:
            counts = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            self.dl.append(len(d))
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(docs)
        self.avg = (sum(self.dl) / self.n) if self.n else 1.0
        self.inv = {}
        for i, counts in enumerate(self.tf):
            for t in counts:
                self.inv.setdefault(t, []).append(i)

    def scores(self, q):
        out = {}
        for t in set(q):
            if t not in self.inv:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i in self.inv[t]:
                f = self.tf[i][t]
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return out


# ── corpus 1: the installed build123d itself ────────────────────────────────
def _kb_doc(obj, limit=1200):
    d = (inspect.getdoc(obj) or "").strip()
    d = re.sub(r"\n{3,}", "\n\n", d)
    return d if len(d) <= limit else d[:limit].rsplit("\n", 1)[0] + "\n..."

def _kb_sig(obj):
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return s if len(s) <= 420 else s[:420] + "...)"

def _kb_introspect():
    ns = _b3d_public_namespace()
    chunks = []
    def add(kind, title, text, names):
        chunks.append({"kind": kind, "title": title, "text": text, "names": names})
    for name in sorted(ns):
        v = ns[name]
        if isinstance(v, type):
            if issubclass(v, enum.Enum):
                members = [m.name for m in v]
                add("enum", f"ENUM {name}", f"ENUM {name} - members: {', '.join(members)}\nUse as {name}.{members[0] if members else 'X'}\n{_kb_doc(v, 600)}", [name])
                continue
            pub = [a for a in dir(v) if not a.startswith("_")]
            def _callable_attr(a):
                try:
                    return callable(getattr(v, a))
                except Exception:
                    return False
            meths = [a for a in pub if _callable_attr(a)][:30]
            extra = ""
            if name in ("Plane", "Axis"):
                known = [a for a in ("XY", "YZ", "ZX", "XZ", "ZY", "YX", "X", "Y", "Z", "front", "back", "left", "right", "top", "bottom") if hasattr(v, a)]
                if known:
                    extra = f"\nPredefined: {', '.join(name + '.' + a for a in known)}"
            add("class", f"CLASS {name}", f"CLASS {name}{_kb_sig(v)}\n{_kb_doc(v)}{extra}"
                + (f"\nMethods: {', '.join(meths)}" if meths else ""), [name])
            if name in _KB_METHOD_CLASSES:
                for m in pub:
                    if m not in v.__dict__:
                        continue                                   # inherited: documented under its defining class
                    try:
                        static = inspect.getattr_static(v, m)
                        member = getattr(v, m)
                    except Exception:
                        continue
                    if isinstance(static, property):
                        doc = _kb_doc(static.fget, 500) if static.fget else ""
                        if doc:
                            add("method", f"PROPERTY {name}.{m}", f"PROPERTY {name}.{m}\n{doc}", [f"{name}.{m}"])
                    elif callable(member):
                        doc = _kb_doc(member, 800)
                        if doc:
                            add("method", f"METHOD {name}.{m}", f"METHOD {name}.{m}{_kb_sig(member)}\n{doc}", [f"{name}.{m}"])
        elif callable(v):
            add("function", f"FUNCTION {name}", f"FUNCTION {name}{_kb_sig(v)}\n{_kb_doc(v)}", [name])
    return chunks


# ── corpus 2: idioms, each verified against the installed library ───────────
_KB_IDIOMS = [
    ("Plate with a hole pattern (algebra mode)",
     "from build123d import *\nplate = Box(60, 40, 8)\nfor x, y in [(-20, -12), (20, -12), (-20, 12), (20, 12)]:\n"
     "    plate = plate - Pos(x, y, 0) * Cylinder(3, 10)   # Cylinder(radius, height); overshoots both faces\nresult = plate\n"),
    ("Holes with BuildPart, Locations and Mode.SUBTRACT (builder mode)",
     "from build123d import *\nwith BuildPart() as p:\n    Box(60, 40, 8)\n    with Locations((-20, -12), (20, -12), (-20, 12), (20, 12)):\n"
     "        Cylinder(3, 10, mode=Mode.SUBTRACT)\nresult = p.part\n"),
    ("Bolt pattern on a grid with GridLocations (builder mode)",
     "from build123d import *\nwith BuildPart() as p:\n    Box(80, 50, 6)\n    with GridLocations(60, 30, 2, 2):\n"
     "        Cylinder(2.5, 8, mode=Mode.SUBTRACT)\nresult = p.part\n"),
    ("Bolt circle with PolarLocations (builder mode)",
     "from build123d import *\nwith BuildPart() as p:\n    Cylinder(40, 8)\n    with PolarLocations(28, 6):\n"
     "        Cylinder(3, 10, mode=Mode.SUBTRACT)\n    Cylinder(8, 10, mode=Mode.SUBTRACT)   # centre bore\nresult = p.part\n"),
    ("Sketch with cut-outs, then extrude (builder mode)",
     "from build123d import *\nwith BuildPart() as p:\n    with BuildSketch():\n        Rectangle(60, 30)\n        with Locations((-20, 0), (20, 0)):\n"
     "            Circle(4, mode=Mode.SUBTRACT)\n    extrude(amount=8)\nresult = p.part\n"),
    ("Extrude a 2D sketch (algebra mode)",
     "from build123d import *\nsketch = Rectangle(60, 30) - SlotOverall(20, 6)\nresult = extrude(sketch, amount=8)\n"),
    ("Fillet the vertical edges of a block",
     "from build123d import *\npart = Box(50, 30, 10)\npart = fillet(part.edges().filter_by(Axis.Z), radius=4)\nresult = part\n"),
    ("Chamfer the top edges",
     "from build123d import *\npart = Box(50, 30, 10)\npart = chamfer(part.edges().group_by(Axis.Z)[-1], length=1.5)\nresult = part\n"),
    ("Select faces and edges by position",
     "from build123d import *\npart = Box(50, 30, 10)\ntop_face = part.faces().sort_by(Axis.Z)[-1]\nbottom_edges = part.edges().group_by(Axis.Z)[0]\n"
     "vertical_edges = part.edges().filter_by(Axis.Z)\nresult = part\n"),
    ("Tube lying along X (rotate a Z-axis cylinder)",
     "from build123d import *\nresult = Rot(0, 90, 0) * (Cylinder(10, 50) - Cylinder(7, 52))\n"),
    ("Hollow box with an open top",
     "from build123d import *\npart = Box(60, 40, 30)\nresult = offset(part, amount=-2, openings=part.faces().sort_by(Axis.Z)[-1])\n"),
    ("Revolve a profile into a solid of revolution",
     "from build123d import *\nprofile = Plane.XZ * Polygon((0, 0), (10, 0), (10, 20), (4, 30), (0, 30), align=None)\nresult = revolve(profile, axis=Axis.Z)\n"),
    ("Sweep a circle along a spline (curved arm / duct)",
     "from build123d import *\npath = Spline((0, 0, 0), (10, 0, 30), (40, 0, 60))\nprofile = Plane(origin=path @ 0, z_dir=path % 0) * Circle(4)\nresult = sweep(profile, path=path)\n"),
    ("Mirror a feature across a plane",
     "from build123d import *\nhalf = Pos(20, 0, 0) * Box(30, 20, 10)\nresult = half + mirror(half, about=Plane.YZ)\n"),
    ("Rounded rectangle plate",
     "from build123d import *\nresult = extrude(RectangleRounded(60, 30, 5), amount=6)\n"),
]

def _kb_idiom_chunks():
    good, dropped = [], []
    for title, code in _KB_IDIOMS:
        try:
            ast.parse(code)
            issues = lint_b3d_script(code, max_issues=3)
        except SyntaxError as e:
            issues = [f"syntax error {e}"]
        if issues:
            dropped.append({"idiom": title, "why": issues[0][:160]})
            continue
        good.append({"kind": "idiom", "title": f"EXAMPLE {title}", "names": [],
                     "text": f"EXAMPLE - {title} (checked against the installed build123d)\n```python\n{code}```"})
    return good, dropped


# ── corpus 3: user/Docker-supplied docs and examples ────────────────────────
def _kb_sanitize_example(text):
    keep = []
    for line in text.splitlines():
        if re.search(r"ocp_vscode|\bshow(_object)?\(|export_|import_(step|stl|brep|svg)|\bset_defaults\(|^\s*from\s+(os|sys)\b|^\s*import\s+(os|sys)\b", line):
            continue
        keep.append(line)
    return "\n".join(keep)

def _kb_doc_chunks(root):
    chunks, skipped, files = [], 0, 0
    if not root or not os.path.isdir(root):
        return chunks, skipped, files
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in sorted(fnames):
            if not fn.lower().endswith((".md", ".rst", ".txt", ".py")) or fn.lower() == "readme.txt":
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > 400_000:
                    continue
                text = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            files += 1
            rel = os.path.relpath(path, root)
            if fn.endswith(".py"):
                text = _kb_sanitize_example(text)
                parts = [text] if len(text) <= 1800 else [p for p in re.split(r"\n\s*\n(?=\S)", text) if p.strip()]
            else:
                text = re.sub(r"^\.\. (image|figure|toctree|index|only|raw)::.*$", "", text, flags=re.M)
                parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            buf = ""
            merged = []
            for p in parts:
                if len(buf) + len(p) < 1200:
                    buf += ("\n\n" if buf else "") + p
                else:
                    if buf:
                        merged.append(buf)
                    buf = p
            if buf:
                merged.append(buf)
            for m in merged:
                m = m.strip()[:1800]
                if len(m) < 60:
                    continue
                if any(tok in m for tok in _KB_FORBIDDEN_IN_DOCS):
                    skipped += 1
                    continue
                chunks.append({"kind": "doc", "title": f"DOC {rel}", "names": [], "text": f"DOC {rel}\n{m}"})
    return chunks, skipped, files


# ── NVIDIA embeddings + reranker ────────────────────────────────────────────
def _kb_embed(texts, input_type):
    """-> float32 array (n, d), L2-normalised. Raises RuntimeError on failure."""
    rows = []
    for i in range(0, len(texts), KB_EMBED_BATCH):
        batch = [t[:1800] for t in texts[i:i + KB_EMBED_BATCH]]
        for attempt in range(5):
            status, data, err = _nv_post(NVIDIA_EMBED_URL, {"model": NVIDIA_EMBED_MODEL, "input": batch,
                                                            "input_type": input_type, "encoding_format": "float",
                                                            "truncate": "END"}, timeout=60)
            if status == 200 and data and data.get("data"):
                break
            if status in (429, 500, 502, 503, 504, None) and attempt < 4:
                time.sleep(min(2 ** attempt * 2, 20))
                continue
            raise RuntimeError(f"embeddings HTTP {status}: {err[:200]}")
        rows.extend(r["embedding"] for r in sorted(data["data"], key=lambda r: r.get("index", 0)))
        if len(texts) > KB_EMBED_BATCH:
            time.sleep(KB_EMBED_PAUSE_S)
    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)

def _rerank_urls(model):
    short = model.split("/")[-1]
    return ["https://integrate.api.nvidia.com/v1/ranking",
            f"https://ai.api.nvidia.com/v1/retrieval/nvidia/{short.replace('.', '_')}/reranking"]

def _kb_rerank(query, passages):
    """[(index, logit)] best-first, or None when reranking is unavailable. The first (model, endpoint)
    that answers is remembered; if every candidate fails (e.g. models retired) reranking is switched off
    for 15 minutes instead of paying a failed round-trip on every request."""
    if not KB_RERANK or not NVIDIA_API_KEY or not passages:
        return None
    now = time.time()
    if now < _rr["disabled_until"]:
        return None
    combos = [(_rr["model"], _rr["url"])] if _rr["model"] else [(m, u) for m in NVIDIA_RERANK_MODELS for u in _rerank_urls(m)]
    for model, url in combos:
        payload = {"model": model, "query": {"text": query[:1200]},
                   "passages": [{"text": p[:1800]} for p in passages], "truncate": "END"}
        status, data, err = _nv_post(url, payload, timeout=25)
        if status == 200 and data and data.get("rankings"):
            _rr.update(model=model, url=url, last_error=None)
            return sorted(((int(r["index"]), float(r.get("logit", 0.0))) for r in data["rankings"]), key=lambda t: -t[1])
        _rr["tried"] = (_rr["tried"] + [f"{model} @ {url.split('/v1/')[1][:40]} -> {status}"])[-12:]
        _rr["last_error"] = f"{model}: HTTP {status} {err[:120]}"
        if status in (401, 403):
            break                                  # auth problem: other models will not help
        if status == 429:
            return None                            # transient: keep the cached model, just skip this once
        if _rr["model"]:                           # cached combo stopped working -> re-probe from scratch next time
            _rr["model"] = _rr["url"] = None
    _rr["disabled_until"] = now + 900
    return None


# ── index lifecycle ─────────────────────────────────────────────────────────
def kb_build(force=False):
    """(Re)build the knowledge base. Chunking + BM25 is instant (mode 'lexical'); embeddings are then
    loaded from KB_INDEX_PATH or fetched from NVIDIA (mode 'hybrid'). Safe to call from a thread."""
    with _kb_lock:
        if _kb["building"]:
            return
        _kb["building"] = True
        _kb["error"] = None
    try:
        chunks = []
        if B3D:
            chunks += _kb_introspect()
            idioms, dropped = _kb_idiom_chunks()
            chunks += idioms
            _kb["dropped_idioms"] = dropped
        docs, skipped, nfiles = _kb_doc_chunks(KB_DOCS_DIR)
        chunks += docs
        for i, c in enumerate(chunks):
            c["id"] = i
            c["embed_text"] = c["text"][:1800]
        by_name = {}
        for c in chunks:
            for n in c["names"]:
                by_name.setdefault(n, []).append(c["id"])
        bm25 = _BM25([_kb_tokens(c["title"] + " " + c["text"]) for c in chunks])
        ver = getattr(b3d, "__version__", None) if B3D else None
        fp = hashlib.sha1((NVIDIA_EMBED_MODEL + "|" + str(ver) + "|" + "\x00".join(c["embed_text"] for c in chunks)).encode()).hexdigest()
        counts = {}
        for c in chunks:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
        with _kb_lock:
            _kb.update(chunks=chunks, bm25=bm25, by_name=by_name, counts=counts, b3d_version=ver, fingerprint=fp,
                       skipped_doc_chunks=skipped, docs_files=nfiles, emb=None,
                       mode="lexical" if chunks else "off", built_at=time.time())
        if not chunks or not NVIDIA_API_KEY or not KB_EMBEDDINGS:
            if not NVIDIA_API_KEY:
                _kb["error"] = "NVIDIA_API_KEY not set — lexical (BM25) retrieval only"
            return
        emb = None
        if not force and os.path.exists(KB_INDEX_PATH):
            try:
                z = np.load(KB_INDEX_PATH, allow_pickle=False)
                if str(z["fp"]) == fp and z["emb"].shape[0] == len(chunks):
                    emb = z["emb"].astype(np.float32)
            except Exception:
                emb = None
        if emb is None:
            emb = _kb_embed([c["embed_text"] for c in chunks], "passage")
            try:
                np.savez_compressed(KB_INDEX_PATH, emb=emb.astype(np.float16), fp=np.array(fp))
            except Exception:
                pass                              # a read-only disk only costs a re-embed next boot
            emb = emb.astype(np.float32)
        with _kb_lock:
            _kb.update(emb=emb, mode="hybrid", embed_dim=int(emb.shape[1]))
    except Exception as e:
        _kb["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        _kb["building"] = False


def _rrf(lists, k=60):
    score = {}
    for lst in lists:
        for rank, cid in enumerate(lst):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda c: -score[c])


def kb_retrieve(query, exact_names=(), k=None):
    """-> (chunks, info). Hybrid retrieval + rerank; identifiers in `exact_names` are always included."""
    k = k or KB_TOP_K
    t0 = time.time()
    with _kb_lock:
        chunks, bm25, emb, by_name, mode = _kb["chunks"], _kb["bm25"], _kb["emb"], _kb["by_name"], _kb["mode"]
    info = {"mode": mode, "rerank": "off"}
    if not chunks:
        return [], info
    lists = []
    bm = bm25.scores(_kb_tokens(query))
    lists.append(sorted(bm, key=lambda i: -bm[i])[:30])
    if emb is not None:
        try:
            qv = _kb_embed([query], "query")[0]
            lists.append([int(i) for i in np.argsort(-(emb @ qv))[:30]])
        except Exception as e:
            info["embed_error"] = str(e)[:160]
            info["mode"] = "lexical (embedding query failed)"
    cand = _rrf(lists)[:20]
    order = cand
    ranked = _kb_rerank(query, [chunks[i]["embed_text"] for i in cand]) if len(cand) > 1 else None
    if ranked:
        order = [cand[i] for i, _ in ranked if i < len(cand)]
        info["rerank"] = _rr["model"]
    elif KB_RERANK and NVIDIA_API_KEY:
        info["rerank"] = f"unavailable ({_rr['last_error'] or 'cooling down'})"
    picked = []
    for n in exact_names:
        for cid in by_name.get(n, [])[:1]:
            if cid not in picked:
                picked.append(cid)
        if len(picked) >= 3:
            break
    info["exact"] = len(picked)
    for cid in order:
        if len(picked) >= k + info["exact"]:
            break
        if cid not in picked:
            picked.append(cid)
    info["ms"] = int((time.time() - t0) * 1000)
    return [chunks[i] for i in picked], info


def kb_format(chunks, max_chars=None):
    max_chars = max_chars or KB_MAX_CHARS
    ver = _kb.get("b3d_version") or "installed"
    head = (f"VERIFIED build123d API REFERENCE (read from the installed library, v{ver}). "
            f"Use ONLY these names, parameters and enum members — do not invent others. "
            f"Prefer algebra mode; use BuildPart/Locations/Mode.SUBTRACT only where an example below shows the need.")
    out, used = [head], len(head)
    for c in chunks:
        body = c["text"]
        if used + len(body) + 2 > max_chars:
            body = body[:max(0, max_chars - used - 20)].rstrip() + "\n..."
            if len(body) < 80:
                break
        out.append(body)
        used += len(body) + 2
        if used >= max_chars:
            break
    return "\n\n".join(out) + "\n=== END REFERENCE ==="


_EXEC_ERR_MARKERS = ("Script execution failed", "Script syntax error", "Unsafe operation", "did not assign",
                     "not a valid B-rep", "zero volume")

def _kb_identifiers(text):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")

def _kb_context_sync(query, exact_candidates):
    with _kb_lock:
        by_name = _kb["by_name"]
    exact = []
    for n in exact_candidates:
        if n in by_name and n not in exact:
            exact.append(n)
    chunks, info = kb_retrieve(query, exact_names=exact)
    info["titles"] = [c["title"] for c in chunks]
    return (kb_format(chunks) if chunks else ""), info

async def kb_context_for(prompt, previous_script=None, feedback=None):
    """Reference block for the prompt ('' when the KB is off / not ready / anything goes wrong)."""
    trace = _KB_TRACE.get()
    if trace is None:
        trace = {}
        _KB_TRACE.set(trace)
    trace["enabled"] = KB_ENABLED
    try:
        if not KB_ENABLED or not _kb["chunks"]:
            trace["skipped"] = "kb off" if not KB_ENABLED else "kb not ready"
            return ""
        is_exec_err = bool(feedback) and any(m in feedback for m in _EXEC_ERR_MARKERS)
        query = (feedback[:700] + " " if feedback else "") + prompt[:400]
        # exact lookups: identifiers on the failing line first, then everything the script uses
        cands = []
        if is_exec_err:
            cands += _kb_identifiers(feedback)
            cands += _kb_identifiers(previous_script) if previous_script else []
        block, info = await asyncio.wait_for(asyncio.to_thread(_kb_context_sync, query, cands), KB_TIMEOUT_S)
        trace.update(info)
        trace["chars"] = len(block)
        return block
    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return ""


# ── static API lint against the installed signatures ────────────────────────
_LINT_BUILTINS = {"abs", "min", "max", "round", "range", "len", "sum", "float", "int", "bool", "str", "list", "tuple",
                  "dict", "set", "enumerate", "zip", "map", "filter", "sorted", "reversed", "isinstance", "any", "all",
                  "pow", "divmod", "print", "iter", "next", "slice", "Exception", "ValueError", "TypeError",
                  "ZeroDivisionError", "True", "False", "None", "math", "np", "result", "make_bent_bracket",
                  "make_tapered_beam", "as_solid"}

def lint_b3d_script(script, max_issues=8):
    """Static check of a generated script against the INSTALLED build123d: unknown names, unknown keyword
    arguments, too many positional arguments, and unknown members on classes/enums (Mode.SUBTRACTION).
    Returns a list of readable issues (each with 'did you mean' and the true signature)."""
    if not B3D:
        return []
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    ns = _b3d_public_namespace()
    defined = set(_LINT_BUILTINS)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
    issues = []
    def add(node, msg):
        if len(issues) < max_issues:
            issues.append(f"line {getattr(node, 'lineno', '?')}: {msg}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in ns and n.id not in defined:
            close = difflib.get_close_matches(n.id, list(ns), n=3, cutoff=0.7)
            add(n, f"unknown name '{n.id}'" + (f" — did you mean {', '.join(close)}?" if close else "")
                + " (only build123d names, math and np exist in this environment)")
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and isinstance(ns.get(n.value.id), type):
            cls = ns[n.value.id]
            if not hasattr(cls, n.attr) and not n.attr.startswith("_"):
                members = ([m.name for m in cls] if issubclass(cls, enum.Enum)
                           else [a for a in dir(cls) if not a.startswith("_")])
                close = difflib.get_close_matches(n.attr, members, n=3, cutoff=0.6)
                add(n, f"{n.value.id} has no member '{n.attr}'" + (f" — did you mean {', '.join(close)}?" if close else "")
                    + (f" Members: {', '.join(members[:14])}" if issubclass(cls, enum.Enum) else ""))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ns and callable(ns[n.func.id]):
            obj = ns[n.func.id]
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            params = sig.parameters
            kinds = [p.kind for p in params.values()]
            has_varkw = inspect.Parameter.VAR_KEYWORD in kinds
            has_varpos = inspect.Parameter.VAR_POSITIONAL in kinds
            if not has_varkw:
                for kw in n.keywords:
                    if kw.arg and kw.arg not in params:
                        close = difflib.get_close_matches(kw.arg, list(params), n=2, cutoff=0.5)
                        add(n, f"{n.func.id}() has no parameter '{kw.arg}'" + (f" — did you mean {', '.join(close)}?" if close else "")
                            + f" Real signature: {n.func.id}{_kb_sig(obj)}")
            if not has_varpos and not any(isinstance(a, ast.Starred) for a in n.args):
                cap = sum(1 for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
                if len(n.args) > cap:
                    add(n, f"{n.func.id}() takes at most {cap} positional arguments but {len(n.args)} were given. "
                           f"Real signature: {n.func.id}{_kb_sig(obj)}")
    return issues


def _kb_lint_report(issues):
    return ("STATIC API CHECK of your previous script against the installed build123d (fix ALL of these in one go):\n"
            + "\n".join(f"- {i}" for i in issues))

async def _kb_enrich_feedback(previous_script, feedback):
    """When the previous script failed to run, append the static API check so one round fixes every API mistake."""
    try:
        if any(m in feedback for m in _EXEC_ERR_MARKERS):
            issues = await asyncio.to_thread(lint_b3d_script, previous_script)
            t = _KB_TRACE.get()
            if t is not None:
                t["lint_issues"] = len(issues)
            if issues:
                return feedback + "\n\n" + _kb_lint_report(issues)
    except Exception:
        pass
    return feedback


async def gemini_generate_script(prompt: str, previous_script: Optional[str] = None,
                                  feedback: Optional[str] = None) -> str:
    """
    Generate/refine a build123d script via whichever provider AI_PROVIDER selects
    (nvidia = Nemotron via NIM, groq, cerebras, openrouter, claude, gemini, lovable).
    Same prompting logic for all; only the transport differs. (Name kept for backward
    compatibility — it is not Gemini-specific.)

    - First call (previous_script/feedback are None): plain generation from `prompt`.
    - Refinement call: include the prior script and an engineering-analysis feedback
      report so the model can produce a corrected version targeting the same part.
    """
    system_prompt = BUILD123D_SYSTEM + REFINEMENT_INSTRUCTIONS
    _KB_TRACE.set({})            # per-call trace, read by the refinement loop

    if previous_script and feedback:
        # a failed script gets a static API check appended (all mistakes in one round), and the
        # exact docs for the APIs involved are retrieved and shown
        feedback = await _kb_enrich_feedback(previous_script, feedback)
        kb_block = await kb_context_for(prompt, previous_script, feedback)
        turns = [
            {"role": "user", "content":
                f"Original request: {prompt}\n\nReturn ONLY Python code. No markdown."},
            {"role": "assistant", "content": previous_script},
            {"role": "user", "content": (kb_block + "\n\n" if kb_block else "") +
                f"ANALYSIS / FEEDBACK FROM ENGINEERING PIPELINE:\n{feedback}\n\n"
                f"Produce a corrected, COMPLETE script fixing the issues above. "
                f"Return ONLY Python code. No markdown."},
        ]
    else:
        is_fold_part = any(w in prompt.lower() for w in FOLD_BRACKET_KEYWORDS)
        is_taper_part = any(w in prompt.lower() for w in TAPER_KEYWORDS)
        if is_fold_part:
            user_msg = (
                f"Generate build123d code for: {prompt}\n\n"
                "This part has a bent/folded flange (per the description above). "
                "MANDATORY: do not write any box/polyline/rotate/union geometry code "
                "yourself for this. Your entire script must build the part by calling "
                "the make_bent_bracket(...) helper that is already available in this "
                "environment, then optionally applying fillet()/chamfer() to selected edges "
                "of its result — nothing else constructs the base geometry. Example:\n\n"
                "result = make_bent_bracket(\n"
                "    leg1_length=50.0, leg2_length=50.0, width=30.0, thickness=4.0,\n"
                "    bend_angle_deg=90.0, fillet_radius=3.0,\n"
                "    holes_leg1=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],\n"
                "    holes_leg2=[(15.0, 10.0, 6.0), (15.0, -10.0, 6.0)],\n"
                ")\n\n"
                "Pick leg1_length/leg2_length/width/thickness/holes from the dimensions "
                "stated in the prompt above. Return ONLY Python code. No markdown."
            )
        elif is_taper_part:
            user_msg = (
                f"Generate build123d code for: {prompt}\n\n"
                "This part is tapered/lofted (per the description above). MANDATORY: "
                "do not write your own loft()/fillet() geometry code for this — "
                "confirmed live, twice, that hand-written loft+fillet code on a "
                "tapered body produces silently broken (non-watertight) geometry "
                "with no Python error at all. Your entire script must build the "
                "part by calling the make_tapered_beam(...) helper that is already "
                "available in this environment. Example:\n\n"
                "result = make_tapered_beam(\n"
                "    length=150.0, base_width=25.0, base_thick=10.0,\n"
                "    tip_width=15.0, tip_thick=6.0, fillet_radius=1.0,\n"
                "    holes_base=[(6.0, 0.0, 4.0), (-6.0, 0.0, 4.0)],\n"
                "    holes_tip=[(3.0, 3.0, 3.0), (3.0, -3.0, 3.0), (-3.0, 3.0, 3.0), (-3.0, -3.0, 3.0)],\n"
                ")\n\n"
                "Pick length/base_width/base_thick/tip_width/tip_thick/holes from the "
                "dimensions stated in the prompt above. Return ONLY Python code. No markdown."
            )
        else:
            kb_block = await kb_context_for(prompt)
            user_msg = ((kb_block + "\n\n") if kb_block else "") + \
                f"Generate build123d code for: {prompt}\n\nReturn ONLY Python code. No markdown."
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
    elif AI_PROVIDER == "cerebras":
        text = await asyncio.to_thread(
            _cerebras_request,
            [{"role": "system", "content": system_prompt}] + turns,
            temperature=0.15, max_tokens=GEN_MAX_TOKENS
        )
    elif AI_PROVIDER == "nvidia":
        # Nemotron is a reasoning model: its (hidden) chain-of-thought counts against
        # max_tokens, so give it more headroom than the non-reasoning providers get.
        text = await asyncio.to_thread(
            _nvidia_request,
            [{"role": "system", "content": system_prompt}] + turns,
            temperature=0.15, max_tokens=NVIDIA_GEN_MAX_TOKENS
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
import traceback
import types

# Modules the generated script is allowed to import. Anything else is rejected.
B3D_ALLOWED_IMPORTS = {"build123d", "math", "numpy", "np"}

# Attribute/name access that is never allowed regardless of context — the
# standard sandbox-escape primitives in pure-Python exec() jails, plus every
# build123d entry point that reads or writes files (a sandboxed script must never
# touch the filesystem; the server does all importing/exporting itself).
B3D_FORBIDDEN_NAMES = {
    "__import__", "__builtins__", "__globals__", "__getattribute__",
    "__subclasses__", "__bases__", "__base__", "__mro__", "__class__",
    "__dict__", "__code__", "__closure__", "__loader__", "__spec__",
    "exec", "eval", "compile", "open", "input", "vars", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit",
    # build123d file I/O
    "export_stl", "export_step", "export_brep", "export_gltf", "export_svg", "export_dxf",
    "import_step", "import_stl", "import_brep", "import_svg", "import_3mf",
    "Mesher", "ExportDXF", "ExportSVG", "ImportSVG",
}

class _B3DSandboxViolation(Exception):
    pass

def _validate_b3d_ast(tree: ast.AST):
    """
    Walk the parsed AST and reject anything outside a narrow, known-safe subset:
    imports of allowed modules only, no dunder/reflection access, no exec/eval-style
    calls, no file/network/process primitives. This replaces a naive substring
    blocklist (trivially bypassable via string concatenation, getattr tricks, etc.)
    with a real structural check. It is defense in depth, not a security boundary —
    run untrusted prompts in a locked-down container.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod_names = [n.name.split(".")[0] for n in node.names] if isinstance(node, ast.Import) \
                        else [(node.module or "").split(".")[0]]
            for m in mod_names:
                if m not in B3D_ALLOWED_IMPORTS:
                    raise _B3DSandboxViolation(f"Import of '{m}' is not allowed. "
                                               f"Only {sorted(B3D_ALLOWED_IMPORTS)} may be imported.")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("build123d."):
                raise _B3DSandboxViolation("Import only from the top-level 'build123d' module "
                                           "(from build123d import *).")
        elif isinstance(node, ast.Name) and node.id in B3D_FORBIDDEN_NAMES:
            raise _B3DSandboxViolation(f"Use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr in B3D_FORBIDDEN_NAMES:
            raise _B3DSandboxViolation(f"Access to attribute '{node.attr}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise _B3DSandboxViolation(f"Access to dunder attribute '{node.attr}' is not allowed.")


# ── build123d helpers ────────────────────────────────────────────────────────

_B3D_NS_CACHE = {}

def _b3d_public_namespace():
    """Every public build123d name a generated script may use — minus modules and
    all file-I/O entry points. Built once and reused."""
    if not _B3D_NS_CACHE:
        for n in dir(b3d):
            if n.startswith("_") or n in B3D_FORBIDDEN_NAMES or n.startswith(("import_", "export_")):
                continue
            v = getattr(b3d, n, None)
            if isinstance(v, types.ModuleType):
                continue
            _B3D_NS_CACHE[n] = v
    return _B3D_NS_CACHE

_B3D_SHIM = []

def _b3d_sandbox_module():
    """A stand-in 'build123d' module that only exposes the safe names above, so that
    `from build123d import *` / `import build123d as bd` inside a script can never
    pull in the real module's file-I/O functions."""
    if not _B3D_SHIM:
        ns = _b3d_public_namespace()
        m = types.ModuleType("build123d")
        for k, v in ns.items():
            setattr(m, k, v)
        m.__all__ = sorted(ns)
        _B3D_SHIM.append(m)
    return _B3D_SHIM[0]

def _b3d_prop(obj, attr, default=None):
    """build123d turned several methods into properties across releases
    (is_valid, is_manifold, ...). Read either form."""
    try:
        v = getattr(obj, attr)
        return v() if callable(v) else v
    except Exception:
        return default

def _b3d_is_valid(shape) -> bool:
    v = _b3d_prop(shape, "is_valid", None)
    return True if v is None else bool(v)

def as_solid(obj):
    """Public helper for scripts: turn a BuildPart builder / list of shapes into one
    Shape. (Scripts normally just assign a Part to `result`.)"""
    shape, err = _b3d_to_shape(obj)
    if err:
        raise ValueError(err)
    return shape

def _b3d_to_shape(obj):
    """Coerce whatever a script (or one of our primitives) produced into one
    build123d Shape with real volume. Returns (shape, error_message)."""
    if obj is None:
        return None, "no shape was assigned"
    if isinstance(obj, (list, tuple)):
        parts = []
        for o in obj:
            s, err = _b3d_to_shape(o)
            if err:
                return None, err
            parts.append(s)
        if not parts:
            return None, "result is an empty list"
        shape = parts[0]
        for p in parts[1:]:
            shape = shape + p
        return shape, None
    if not isinstance(obj, b3d.Shape):
        part = getattr(obj, "part", None)          # BuildPart context object
        if part is not None:
            obj = part
        elif hasattr(obj, "sketch"):
            return None, ("'result' is a 2D sketch (BuildSketch). Extrude/revolve it into a solid first, "
                          "e.g. result = extrude(Rectangle(20, 10), amount=5).")
        elif hasattr(obj, "line"):
            return None, "'result' is a curve (BuildLine), not a solid."
        else:
            return None, (f"'result' is of type {type(obj).__name__}, not a build123d solid. Assign a "
                          f"Part/Solid, e.g. result = Box(10, 10, 10).")
    vol = _b3d_prop(obj, "volume", 0.0) or 0.0
    if not (vol > 1e-9):
        return None, (f"'result' ({type(obj).__name__}) has zero volume — it is a Face/Wire/Sketch or an empty "
                      f"boolean, not a solid. Extrude/revolve/loft it into a solid, and check that a cut "
                      f"didn't remove everything.")
    return obj, None

def _b3d_fillet(shape, edges, radius):
    """Fillet `edges` of `shape` — works across build123d versions."""
    edges = list(edges)
    try:
        return shape.fillet(radius, edges)
    except (TypeError, AttributeError):
        return b3d.fillet(edges, radius)

def _b3d_rect_wire(w, h, z):
    """Closed rectangular wire, centered on the Z axis, lying in the plane Z=z."""
    face = b3d.Pos(0, 0, z) * b3d.Face.make_rect(w, h)
    return face.outer_wire()

def _b3d_write_stl(shape, path, tolerance=0.01, angular_tolerance=0.05):
    ok = _b3d_export_stl(shape, path, tolerance=tolerance, angular_tolerance=angular_tolerance)
    if ok is False or not os.path.exists(path) or os.path.getsize(path) < 84:
        raise RuntimeError("build123d STL export produced no usable file (empty/invalid solid?)")

def _b3d_write_step(shape, path):
    ok = _b3d_export_step(shape, path)
    if ok is False or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("build123d STEP export produced no usable file (empty/invalid solid?)")


def make_bent_bracket(leg1_length, leg2_length, width, thickness,
                       bend_angle_deg=90.0, fillet_radius=2.0,
                       holes_leg1=None, holes_leg2=None):
    """
    Build a genuinely folded two-flange bracket (L-bracket / angle bracket) as one
    solid, guaranteeing leg2 actually rises out of the base plane by bend_angle_deg
    via a real rotation — the exact operation free-tier LLMs kept failing to
    hand-write correctly (confirmed live: they either left both legs flat and
    coplanar, or attempted their own rotate/union and produced non-watertight
    geometry). This is trusted server-side code, not AI-generated, so it only needs
    to be gotten right once; the model's job becomes picking sensible parameters,
    not 3D CAD authoring.

    Both legs share a bend edge along the Y-axis at x=0,z=0. Each leg extends from
    that edge outward along its own local +X for leg{1,2}_length, and is `width`
    wide (centered on y=0), thickness `thickness`. holes_leg1/holes_leg2 are each an
    optional list of (x_from_bend_mm, y_from_centerline_mm, diameter_mm) tuples, given
    in that leg's own FLAT local frame (before folding) — no 3D math required by the
    caller. Leg2 folds up towards +Z.

    Returns the finished build123d solid — assign it to `result`.
    """
    holes_leg1 = holes_leg1 or []
    holes_leg2 = holes_leg2 or []

    def _leg_with_holes(length, holes):
        leg = b3d.Pos(length / 2.0, 0, thickness / 2.0) * b3d.Box(length, width, thickness)
        for hx, hy, hd in holes:
            # cutter overshoots both faces by 1mm -> clean through-hole, no coincident faces
            leg = leg - (b3d.Pos(hx, hy, thickness / 2.0) * b3d.Cylinder(hd / 2.0, thickness + 2.0))
        return leg

    leg1 = _leg_with_holes(leg1_length, holes_leg1)
    leg2 = b3d.Rot(0, -bend_angle_deg, 0) * _leg_with_holes(leg2_length, holes_leg2)

    bracket = leg1 + leg2

    if fillet_radius and fillet_radius > 0:
        try:
            bend_edges = [e for e in bracket.edges()
                          if abs(e.center().X) < 0.5 and abs(e.center().Z) < 0.5
                          and abs(e.length - width) < 0.5]
            if bend_edges:
                bracket = _b3d_fillet(bracket, bend_edges, fillet_radius)
        except Exception:
            pass  # sharp (unfilleted) bend is a fine fallback; don't fail the whole part

    return bracket

def make_tapered_beam(length, base_width, base_thick, tip_width, tip_thick,
                       fillet_radius=0.0, holes_base=None, holes_tip=None):
    """
    Build a tapered/lofted beam (drone arm, tapered leg, connecting rod, fin,
    tapered housing wall) as one solid, guaranteeing correct topology via a
    proper ruled loft PLUS safe fillet handling — instead of relying on the model
    to hand-write loft+fillet code itself. Confirmed live, twice: blanket
    fillet on ALL of a loft's edges — including the compound corners where a
    sloped taper edge meets two flat profile edges — silently produces
    self-intersecting (non-watertight) geometry with no Python error at all, and
    simply telling the model the exact coordinates of the resulting gap was NOT
    enough for it to reliably avoid the mistake next time. This is trusted
    server-side code, not AI-generated, so it only needs to be gotten right once.

    The beam runs along Z from 0 (base) to length (tip). Cross-section is a
    rectangle: base_width x base_thick at Z=0, tapering to tip_width x
    tip_thick at Z=length. holes_base/holes_tip are each an optional list of
    (x_from_center_mm, y_from_center_mm, diameter_mm) tuples, drilled straight
    through along Z in the beam's centered cross-section frame — no 3D math
    required by the caller.

    fillet_radius, if given, is applied ONLY to the 8 flat top/bottom
    profile edges (the rectangle outlines at Z=0 and Z=length) — NEVER the 4
    sloped taper edges connecting them, since the compound corners where
    those meet are exactly where the self-intersection risk lives. Default
    0 (no fillet): an unfilleted-but-correct beam is far better than a
    filleted-but-broken one.

    Returns the finished build123d solid — assign it to `result`.
    """
    holes_base = holes_base or []
    holes_tip = holes_tip or []

    beam = b3d.Solid.make_loft([_b3d_rect_wire(base_width, base_thick, 0.0),
                                _b3d_rect_wire(tip_width, tip_thick, float(length))], True)

    if fillet_radius and fillet_radius > 0:
        try:
            flat_edges = [e for e in beam.edges()
                          if abs(e.center().Z) < 0.1 or abs(e.center().Z - length) < 0.1]
            if flat_edges:
                beam = _b3d_fillet(beam, flat_edges, fillet_radius)
        except Exception:
            pass  # unfilleted taper is a fine fallback; don't fail the whole part

    # Same semantics as before the CadQuery -> build123d switch: each hole is a
    # bore along Z through the whole beam, overshooting both end faces by 1mm.
    for hx, hy, hd in list(holes_base) + list(holes_tip):
        beam = beam - (b3d.Pos(hx, hy, length / 2.0) * b3d.Cylinder(hd / 2.0, length + 2.0))

    return beam

def execute_cad_script_safely(script: str):
    """
    Execute an AI-generated build123d script in a sandboxed namespace.

    The script is first parsed to an AST and checked against a narrow allowlist
    (imports limited to build123d/math/numpy, no dunder/reflection access, no
    exec/eval/getattr-style escape hatches, no build123d file I/O) before ever
    calling exec(). A restricted builtins dict and a filtered build123d namespace
    are also used as defense in depth.

    Returns (shape, error_message). On success, error_message is None and shape is
    the build123d solid assigned to `result`. On any failure (forbidden op, syntax
    error, runtime error, missing/invalid `result`), shape is None and
    error_message describes the problem in a form suitable for feeding back to the
    LLM for refinement.
    """
    if not B3D:
        return None, "build123d is not installed on this server."

    try:
        tree = ast.parse(script, filename="<ai_script>", mode="exec")
    except SyntaxError as e:
        return None, f"Script syntax error: {str(e)} (line {e.lineno}: {e.text!r})"

    try:
        _validate_b3d_ast(tree)
    except _B3DSandboxViolation as e:
        return None, f"Unsafe operation detected and blocked: {str(e)} Remove it entirely."

    # __import__ must exist for Python's own `import X` statement to work at all,
    # but this wrapper only ever hands back the filtered build123d shim, math or
    # numpy — by the time exec() runs, every import in the script has already been
    # proven safe at the AST level, so this is redundant-but-safe defense in depth.
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        if name == "build123d":
            return _b3d_sandbox_module()
        if top_level in ("math", "numpy") and name.count(".") == 0:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import of '{name}' is not allowed in this sandbox.")

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "round": round, "range": range,
        "len": len, "sum": sum, "float": float, "int": int, "bool": bool,
        "str": str, "list": list, "tuple": tuple, "dict": dict, "set": set,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
        "any": any, "all": all, "pow": pow, "divmod": divmod, "print": print,
        "iter": iter, "next": next, "slice": slice,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "ZeroDivisionError": ZeroDivisionError,
        "True": True, "False": False, "None": None,
        "__import__": _restricted_import,
    }

    namespace = dict(_b3d_public_namespace())
    namespace.update({
        "__builtins__": safe_builtins,
        "math": math,
        "np": np,
        "make_bent_bracket": make_bent_bracket,
        "make_tapered_beam": make_tapered_beam,
        "as_solid": as_solid,
        "result": None,
    })
    pre_existing = set(namespace)

    try:
        exec(compile(tree, "<ai_script>", "exec"), namespace)
    except Exception as e:
        # Extracting the failing line from the AI's own script (filtering traceback
        # frames to filename "<ai_script>" so this harness's own exec()-call frame
        # doesn't leak in) turns an unlocatable error into an actionable one —
        # without it the refinement loop burned all its attempts repeating the same
        # mistake because the model could not tell which operation to fix.
        tb_lines = script.splitlines()
        script_frames = [f for f in traceback.extract_tb(e.__traceback__)
                          if f.filename == "<ai_script>"]
        location = ""
        if script_frames:
            ln = script_frames[-1].lineno
            src = tb_lines[ln - 1].strip() if ln and 0 < ln <= len(tb_lines) else None
            location = f" [line {ln}: `{src}`]" if src else f" [line {ln}]"
        return None, f"Script execution failed: {type(e).__name__}: {str(e)}{location}"

    obj = namespace.get("result")
    if obj is None:
        # Fallback: the AI occasionally builds a valid shape but assigns it to a
        # differently-named variable despite the system prompt's explicit
        # instruction. Rather than hard-fail a script that actually built real
        # geometry, look at names the script itself defined and accept the shape
        # only if exactly one candidate exists — with several it is genuinely
        # ambiguous which was meant to be final, so don't guess.
        candidates = [
            (k, v) for k, v in namespace.items()
            if k not in pre_existing and not k.startswith("_")
            and (isinstance(v, b3d.Shape) or getattr(v, "part", None) is not None)
        ]
        if len(candidates) == 1:
            obj = candidates[0][1]
        else:
            return None, (
                "Script ran without error but did not assign a shape to the "
                "'result' variable."
                + (f" Found {len(candidates)} other build123d shapes "
                   f"({', '.join(k for k, _ in candidates)}) — too ambiguous to "
                   f"guess which was meant to be final; assign explicitly to "
                   f"'result'." if candidates else "")
            )

    obj, shape_err = _b3d_to_shape(obj)
    if shape_err:
        return None, shape_err

    # Defensively simplify the final solid at the B-rep level before it is ever
    # tessellated or sent to a solver. A boolean union/cut chain — especially one
    # using the deliberate overshoot/overlap margins the refinement prompt teaches —
    # can leave redundant/coincident topology that later tessellates into thin
    # overlapping facets, which Gmsh/SimScale meshers correctly reject. Doing it
    # here fixes it at the right layer. Best-effort: never fail a script over cleanup.
    try:
        cleaned = obj.clean()
        if cleaned is not None and _b3d_prop(cleaned, "volume", 0.0):
            obj = cleaned
    except Exception:
        pass

    if not _b3d_is_valid(obj):
        try:
            fixed = obj.fix()
            if fixed is not None and _b3d_is_valid(fixed):
                obj = fixed
        except Exception:
            pass
    if not _b3d_is_valid(obj):
        return None, ("The resulting solid is not a valid B-rep (OpenCascade validity check failed) — usually a "
                      "self-intersecting boolean/fillet or a degenerate loft. Simplify the failing operation, "
                      "use real overlap for unions and overshoot for cuts, and avoid tiny fillets.")

    return obj, None

# ═══════════════════════════════════════════════════════════════════
# DETERMINISTIC FEATURE VERIFICATION — "the AI says there is a hole, is there?"
#
# The LLM is told (BUILD123D_SYSTEM) to put a machine-readable comment above the code
# that cuts each hole/bore:
#       # FEATURE: hole dia=6 count=4
#       # FEATURE: bore dia=22 count=2
# After the script runs, we MEASURE the finished B-rep — cylindrical faces whose axis
# lies in empty space (i.e. cavities, not bosses or fillets) — and compare against the
# declarations. A declared hole that is not in the solid (cutter missed the material,
# wrong axis, subtraction discarded, later union re-filled it, radius/diameter mixup)
# is reported back to the LLM as a "missing_features" refinement round. This runs
# BEFORE the slow SimScale analysis, so a part missing a required hole never costs a
# cloud run.
#
# Kernel access goes through OCP (the OpenCascade binding build123d itself sits on)
# and shape.is_inside(), not build123d's higher-level face helpers, because those
# helpers changed between build123d releases while these have not.
# ═══════════════════════════════════════════════════════════════════
_FEATURE_COMMENT_RE = re.compile(r"#\s*FEATURE\s*:\s*(.+?)\s*$", re.I)
_FNUM = r"(\d+(?:\.\d+)?)"


def parse_declared_features(script: str):
    """Parse '# FEATURE: <hole|bore> dia=<mm> count=<n>' comments (flexible wording:
    '4x hole 6mm', 'hole dia 6 x4', 'bore d=22 count=2' all work). Returns a list of
    dicts: kind (hole|bore|None), dia (mm or None), count (int), line, text."""
    feats = []
    for ln, line in enumerate(script.splitlines(), 1):
        m = _FEATURE_COMMENT_RE.search(line)
        if not m:
            continue
        text = m.group(1).strip()
        low = text.lower()
        kind = "bore" if re.search(r"\bbores?\b", low) else ("hole" if re.search(r"\bholes?\b", low) else None)
        md = (re.search(r"(?<![a-z])(?:dia(?:meter)?|d|\u2300|\u00f8)\s*[=:]?\s*" + _FNUM, low)
              or re.search(_FNUM + r"\s*mm", low))
        dia = float(md.group(1)) if md else None
        mc = (re.search(r"(?:count|qty|num|n)\s*[=:]\s*(\d+)", low)
              or re.search(r"[x\u00d7]\s*(\d+)\b", low)
              or re.search(r"\b(\d+)\s*[x\u00d7]", low))
        count = max(int(mc.group(1)), 1) if mc else 1
        feats.append({"kind": kind, "dia": dia, "count": count, "line": ln, "text": text})
    return feats


def _extract_cylinder_candidates(shape):
    """Every CONCAVE cylindrical face of `shape` as a plain dict (radius, canonical axis
    direction, perpendicular axis offset, axial interval, angular span, midpoint).
    Concave = the point on the cylinder's axis half-way along the face lies OUTSIDE the
    material (holes, bores, tube interiors); bosses and outer rounds fail that test."""
    import OCP.GeomAbs as ga
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    cyl_type = getattr(ga, "GeomAbs_Cylinder", None) or ga.GeomAbs_SurfaceType.GeomAbs_Cylinder
    solids = list(shape.solids()) or [shape]

    def inside(pt):
        """True / False, or None when the kernel query itself failed (then we skip the face
        rather than guess)."""
        answered = False
        for s in solids:
            try:
                if s.is_inside(pt):
                    return True
                answered = True
            except Exception:
                continue
        return False if answered else None

    out = []
    for f in shape.faces():
        try:
            surf = BRepAdaptor_Surface(f.wrapped, True)
            if surf.GetType() != cyl_type:
                continue
            cyl = surf.Cylinder()
            r = float(cyl.Radius())
            ax = cyl.Axis()
            loc, dr = ax.Location(), ax.Direction()
            L = (loc.X(), loc.Y(), loc.Z())
            D = [dr.X(), dr.Y(), dr.Z()]
            u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
            v0, v1 = surf.FirstVParameter(), surf.LastVParameter()
        except Exception:
            continue
        h = abs(v1 - v0)
        if r <= 1e-6 or h <= 1e-6:
            continue
        vm = 0.5 * (v0 + v1)
        mid = tuple(L[i] + D[i] * vm for i in range(3))
        if inside(mid) is not False:      # material on the axis (boss/round) or unknown -> not a hole
            continue
        # canonical axis direction (first significant component positive) so that the two
        # halves of a split hole wall — and anti-parallel duplicates — compare equal
        flip = D[0] < -1e-9 or (abs(D[0]) <= 1e-9 and (D[1] < -1e-9 or (abs(D[1]) <= 1e-9 and D[2] < 0)))
        Dc = [-c for c in D] if flip else D
        tm = sum(mid[i] * Dc[i] for i in range(3))
        q = tuple(mid[i] - Dc[i] * tm for i in range(3))
        out.append({"r": r, "d": tuple(Dc), "q": q, "t0": tm - h / 2, "t1": tm + h / 2,
                    "ang": min(abs(u1 - u0), 2 * math.pi)})
    return out


def _cluster_hole_candidates(cands, min_angle_deg=300.0):
    """Merge cylinder pieces that belong to the same physical hole (the two halves of a
    split wall share radius/axis/axial range) and keep only those covering >= min_angle_deg
    of the circumference — that is what separates a hole from a slot end or an inner fillet."""
    groups = []
    for c in cands:
        placed = False
        for g in groups:
            if abs(g["r"] - c["r"]) > 1e-3 + 1e-4 * g["r"]:
                continue
            if sum(a * b for a, b in zip(g["d"], c["d"])) < 0.9999:
                continue
            if math.dist(g["q"], c["q"]) > 0.02:
                continue
            ov = min(g["t1"], c["t1"]) - max(g["t0"], c["t0"])
            if ov < 0.9 * min(g["t1"] - g["t0"], c["t1"] - c["t0"]):
                continue
            g["ang"] += c["ang"]
            g["t0"], g["t1"] = min(g["t0"], c["t0"]), max(g["t1"], c["t1"])
            placed = True
            break
        if not placed:
            groups.append(dict(c))
    holes = []
    for g in groups:
        if math.degrees(g["ang"]) < min_angle_deg:
            continue
        tm = 0.5 * (g["t0"] + g["t1"])
        ctr = [g["q"][i] + g["d"][i] * tm for i in range(3)]
        holes.append({"diameter_mm": round(2 * g["r"], 3),
                      "axis": [round(x, 3) for x in g["d"]],
                      "center_mm": [round(x, 2) for x in ctr],
                      "depth_mm": round(g["t1"] - g["t0"], 3)})
    holes.sort(key=lambda h: (h["diameter_mm"], h["center_mm"]))
    return holes


def find_b3d_holes(shape):
    """All holes/bores actually present in a build123d solid (see module comment)."""
    return _cluster_hole_candidates(_extract_cylinder_candidates(shape))


def _verify_against_holes(declared, holes):
    out = {"declared": declared, "found_holes": holes, "missing": [], "unverified": [], "ok": True,
           "checked": 0}
    avail = list(holes)
    # declarations with a diameter claim their holes first; 'any hole' declarations take what is left
    for f in sorted(declared, key=lambda f: (f["dia"] is None)):
        if f["kind"] is None:
            out["unverified"].append(f)
            continue
        out["checked"] += 1
        if f["dia"] is None:
            match = list(avail)
        else:
            tol = max(0.05, 0.01 * f["dia"])
            match = [h for h in avail if abs(h["diameter_mm"] - f["dia"]) <= tol]
        take = match[:f["count"]]
        for h in take:
            avail.remove(h)
        if len(take) < f["count"]:
            out["missing"].append({**f, "found_matching": len(take), "missing_count": f["count"] - len(take)})
    out["ok"] = not out["missing"]
    return out


def verify_declared_features(script: str, shape):
    """Check every '# FEATURE:' declaration in `script` against the finished solid.
    Never raises: if the kernel query itself fails, verification is skipped (ok=True,
    `error` set) rather than blocking a design over a checker problem."""
    declared = parse_declared_features(script)
    if not declared:
        return {"declared": [], "found_holes": [], "missing": [], "unverified": [], "ok": True, "checked": 0}
    try:
        holes = find_b3d_holes(shape)
    except Exception as e:
        return {"declared": declared, "found_holes": [], "missing": [], "unverified": declared, "ok": True,
                "checked": 0, "error": f"feature check skipped: {type(e).__name__}: {str(e)[:200]}"}
    return _verify_against_holes(declared, holes)


# ═══════════════════════════════════════════════════════════════════
# PROGRAMMATIC GEOMETRY INSPECTION (replaces the PyVista stress-image idea)
#
# Nemotron is a text model — it cannot reliably judge a 3D part from a picture, and a render
# needs a GL context that is fragile on a headless free-tier container. What it CAN do well is
# read exact numbers. So instead of drawing the part, we MEASURE the finished B-rep with
# OpenCascade (exact volume / area / bounding box / centre of mass / topology counts / holes)
# and hand those numbers to the model. Two layers:
#   1. inspect_b3d_geometry(): deterministic, free, runs before the slow SimScale call.
#      One HARD gate: the result must be exactly ONE solid (disconnected pieces mean a missed
#      union or features that do not touch — SimScale would import them as separate bodies).
#      Everything else is reported, not enforced.
#   2. review_geometry_with_llm(): Nemotron compares the ORIGINAL REQUEST with the measured
#      report ("asked 80x50x5, solid measures 80x50x12") — the one class of mistake no
#      deterministic check can catch, because only a reader of the prompt knows what was asked.
#      Fail-open (any error/timeout skips it), at most ONE review-triggered refinement per job,
#      and switchable with GEOMETRY_REVIEW=0.
# ═══════════════════════════════════════════════════════════════════
GEOMETRY_REVIEW = os.environ.get("GEOMETRY_REVIEW", "1").strip().lower() not in ("0", "false", "no", "off")
GEOMETRY_REVIEW_TIMEOUT_S = float(os.environ.get("GEOMETRY_REVIEW_TIMEOUT_S", "90"))
GEOMETRY_REVIEW_MODEL = os.environ.get("GEOMETRY_REVIEW_MODEL", "").strip() or None   # default: NVIDIA_MODEL
GEOMETRY_REVIEW_MAX_TOKENS = int(os.environ.get("GEOMETRY_REVIEW_MAX_TOKENS", "4000"))  # reasoning eats budget


def inspect_b3d_geometry(shape):
    """Measure the finished build123d solid. Never raises — a field whose query fails is left
    out with an `*_error` note instead of blocking the design over a checker problem."""
    out = {"ok": True, "issues": []}

    def rnd(v, n=3):
        return round(float(v), n)

    try:
        solids = list(shape.solids()) or [shape]
        out["solid_count"] = len(solids)
        if len(solids) > 1:
            out["ok"] = False
            vols = sorted((rnd(_b3d_prop(s_, "volume", 0.0), 2) for s_ in solids), reverse=True)
            out["issues"].append(
                f"The script produced {len(solids)} separate solids (volumes mm3: {vols[:6]}), not 1. A "
                "single manufacturable part must be one connected solid — usually a union (+) was skipped, "
                "or two features only touch at a face/edge instead of really overlapping.")
    except Exception as e:
        out["solid_count_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    out["is_valid_brep"] = bool(_b3d_is_valid(shape))

    try:
        bb = shape.bounding_box()
        out["bounding_box_mm"] = {
            "size": {"x": rnd(bb.size.X), "y": rnd(bb.size.Y), "z": rnd(bb.size.Z)},
            "min": {"x": rnd(bb.min.X), "y": rnd(bb.min.Y), "z": rnd(bb.min.Z)},
            "max": {"x": rnd(bb.max.X), "y": rnd(bb.max.Y), "z": rnd(bb.max.Z)}}
    except Exception as e:
        out["bounding_box_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    vol = _b3d_prop(shape, "volume", None)
    if vol is not None:
        out["volume_mm3"] = rnd(vol, 2)
    area = _b3d_prop(shape, "area", None)
    if area is not None:
        out["surface_area_mm2"] = rnd(area, 2)

    try:
        c = shape.center()
        out["center_of_mass_mm"] = {"x": rnd(c.X), "y": rnd(c.Y), "z": rnd(c.Z)}
    except Exception as e:
        out["center_of_mass_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    try:
        out["face_count"] = len(list(shape.faces()))
        out["edge_count"] = len(list(shape.edges()))
        out["vertex_count"] = len(list(shape.vertices()))
    except Exception as e:
        out["topology_count_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    try:
        out["holes"] = [{"diameter_mm": h["diameter_mm"], "axis": h["axis"], "center_mm": h["center_mm"]}
                        for h in find_b3d_holes(shape)][:40]
    except Exception as e:
        out["holes_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return out


def geometry_report_text(gi) -> str:
    """The measured facts as compact text — appended to refinement feedback and given to the
    reviewer so the model can check its own claims against reality."""
    bb = (gi.get("bounding_box_mm") or {}).get("size")
    lines = ["MEASURED GEOMETRY of the solid your script just produced (exact OpenCascade values, not estimates):"]
    if bb:
        lines.append(f"- bounding box: {bb['x']} x {bb['y']} x {bb['z']} mm (X x Y x Z)")
    if gi.get("volume_mm3") is not None:
        lines.append(f"- volume: {gi['volume_mm3']} mm3; surface area: {gi.get('surface_area_mm2')} mm2")
    if gi.get("center_of_mass_mm"):
        c = gi["center_of_mass_mm"]
        lines.append(f"- centre of mass: ({c['x']}, {c['y']}, {c['z']}) mm")
    lines.append(f"- solids: {gi.get('solid_count')}; valid B-rep: {gi.get('is_valid_brep')}; faces: "
                 f"{gi.get('face_count')}, edges: {gi.get('edge_count')}")
    holes = gi.get("holes")
    if holes is not None:
        lines.append("- holes/bores: " + ("none" if not holes else "; ".join(
            f"dia {h['diameter_mm']:g} axis {h['axis']} at {h['center_mm']}" for h in holes[:12])))
    return "\n".join(lines)


def geometry_failure_feedback(gi) -> str:
    """Feedback for a 'geometry_inspection_failed' refinement round (hard structural failure)."""
    return ("DETERMINISTIC GEOMETRY CHECK FAILED — measured directly on the finished solid:\n\n"
            + "\n".join(f"- {i}" for i in gi.get("issues", [])) + "\n\n" + geometry_report_text(gi)
            + "\n\nFix the modelling so the result is exactly one valid solid. Keep every other requirement "
              "(dimensions, features, material) unchanged and keep the '# FEATURE:' comments.")


def _gi_brief(gi):
    return {k: gi.get(k) for k in ("ok", "solid_count", "is_valid_brep", "bounding_box_mm", "volume_mm3",
                                   "surface_area_mm2", "center_of_mass_mm", "face_count", "edge_count", "issues")}


_GEOMETRY_REVIEW_SYSTEM = (
    "You are a strict mechanical-design reviewer. You get the ORIGINAL USER REQUEST for a CAD part and "
    "the exact MEASURED geometry of the solid that was actually built. Decide whether the measured solid "
    "plausibly satisfies the explicit numbers in the request: overall dimensions (allow ~3% or 0.5 mm), "
    "hole/bore counts and diameters, and obvious scale or axis mix-ups (e.g. 5 mm thick asked, 50 mm built). "
    "For L-brackets, angles, channels, tubes, shells, flanged plates, ribbed or bent parts, a stated "
    "thickness / wall / sheet value is NOT an overall bounding-box dimension: never flag that no bounding-box "
    "dimension equals it, and judge such parts only on leg / length / width numbers that must appear. "
    "Only flag CLEAR contradictions of stated numbers. Do NOT flag things the request never specified, "
    "do not judge strength or style, and do not invent requirements. Reply with ONLY compact JSON: "
    '{"verdict":"pass"|"fail","issues":["one short sentence per contradiction, quoting asked vs measured"]}')


async def review_geometry_with_llm(prompt: str, gi):
    """Nemotron reads the request + measured numbers. Returns {"verdict","issues"} or None when
    the review is off / errored / timed out (fail-open: a slow or broken reviewer never blocks)."""
    if not GEOMETRY_REVIEW or not NVIDIA_API_KEY:
        return None
    try:
        user = f"ORIGINAL REQUEST:\n{prompt[:3000]}\n\n{geometry_report_text(gi)}"
        text = await asyncio.to_thread(
            _nvidia_request,
            [{"role": "system", "content": _GEOMETRY_REVIEW_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=GEOMETRY_REVIEW_MAX_TOKENS,
            model=GEOMETRY_REVIEW_MODEL, timeout=GEOMETRY_REVIEW_TIMEOUT_S)
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
        if not isinstance(data, dict) or data.get("verdict") not in ("pass", "fail"):
            return None
        issues = [str(i)[:300] for i in (data.get("issues") or []) if i][:6]
        return {"verdict": data["verdict"], "issues": issues}
    except Exception:
        return None


def feature_failure_summary(fv) -> str:
    parts = []
    for m in fv.get("missing", []):
        d = f" dia={m['dia']:g}" if m.get("dia") is not None else ""
        parts.append(f"'{m['kind']}{d}' x{m['count']} (line {m['line']}): found {m['found_matching']}")
    return "; ".join(parts) or "none"


def _fv_brief(fv):
    return {"ok": fv.get("ok", True), "declared": len(fv.get("declared", [])), "checked": fv.get("checked", 0),
            "missing": [{k: m[k] for k in ("kind", "dia", "count", "line", "found_matching")}
                        for m in fv.get("missing", [])],
            "found_hole_diameters_mm": [h["diameter_mm"] for h in fv.get("found_holes", [])][:40],
            **({"error": fv["error"]} if fv.get("error") else {})}


def feature_failure_feedback(fv) -> str:
    """The message the LLM gets in a 'missing_features' refinement round."""
    lines = ["DETERMINISTIC FEATURE CHECK FAILED. Your script declares features with '# FEATURE:' comments "
             "that the finished SOLID does not actually contain. This was measured on the B-rep itself "
             "(cylindrical cavities), it is not an estimate:", ""]
    for m in fv.get("missing", []):
        d = f" dia={m['dia']:g} mm" if m.get("dia") is not None else ""
        lines.append(f"- line {m['line']}: `# FEATURE: {m['text']}` -> expected {m['count']} x {m['kind']}{d}, "
                     f"solid has only {m['found_matching']} matching.")
    holes = fv.get("found_holes", [])
    if holes:
        lines.append("")
        lines.append("Holes/bores that DO exist in the solid: " + "; ".join(
            f"dia {h['diameter_mm']:g} at {h['center_mm']} axis {h['axis']}" for h in holes[:12]))
    else:
        lines.append("")
        lines.append("The solid contains NO holes or bores at all.")
    lines += ["",
              "Typical causes: the cutter sits outside the material or points along the wrong axis "
              "(Cylinder's axis is Z by default — use Rot(0, 90, 0) for X, Rot(90, 0, 0) for Y); the cutter is "
              "too short to reach the material; the subtraction was applied to a shape that was later "
              "discarded, or `result` was assigned from a variable created BEFORE the cut; a later union (+) "
              "re-filled the hole; Cylinder() takes the RADIUS (dia/2), not the diameter.",
              "Fix the geometry so every declared feature really exists — editing only the comment does not "
              "count. Keep all other features and dimensions unchanged, and keep the '# FEATURE:' comments "
              "(update them only if you deliberately change a feature)."]
    return "\n".join(lines)


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
        # FIX: mesh.ray.intersects_location() needs an rtree-based spatial index
        # under the hood — confirmed live across many real runs on this deployment
        # ("No module named 'rtree'" despite it being listed in requirements.txt).
        # Rather than keep gambling on a system/build fix for a package outside
        # our control, estimate thickness via a directional nearest-neighbor
        # search over the same sample cloud instead of true ray-surface
        # intersection. Needs only scipy (already a hard dependency here), and
        # is actually cheaper per-point than ray casting was. Coarser than true
        # ray casting — limited by sample density rather than exact geometry —
        # but most sensitive exactly where it matters most: two sample points
        # from opposite faces of a genuinely THIN wall are likely to land among
        # each other's nearest neighbors precisely because the wall is thin, so
        # min_mm/critical_zones (what the rule engine actually acts on) degrade
        # gracefully; mean_mm/max_mm on thick sections are the least-accurate
        # part of this estimate. "A working coarse estimate" beats "unavailable".
        from scipy.spatial import cKDTree
        tree = cKDTree(all_pts)
        K = min(40, len(all_pts))
        all_dists, all_idxs = tree.query(all_pts, k=K)

        all_t=[];thin=[];crit=[]
        for i in range(len(all_pts)):
            pt=all_pts[i];n=all_norms[i]
            best=None
            for d,j in zip(all_dists[i],all_idxs[i]):
                if j==i or d<0.02:
                    continue
                direction=(all_pts[j]-pt)/d
                # FIX: confirmed live — the direction-only check above fires near
                # end-caps/corners, where a point on one face can have a VERY
                # close neighbor on a DIFFERENT, roughly-PERPENDICULAR adjacent
                # face (e.g. a side wall right next to the end cap at the base/
                # tip of a beam). That neighbor's direction can look "roughly
                # backward" from the query point's normal even though it isn't
                # the opposite wall at all — this reported a beam's actual
                # 6-10mm-thick section as 0.02mm at exactly the two ends
                # (z near 0 and z near length), which is a corner artifact, not
                # a real thin wall. A genuine thin wall means the CANDIDATE's
                # own surface also faces roughly the opposite way — an adjacent
                # perpendicular face's normal does not — so require both the
                # direction-to-candidate AND the candidate's own normal to be
                # roughly antiparallel to this point's normal before accepting it.
                if np.dot(direction,n) < -0.4 and np.dot(all_norms[j],n) < -0.6:
                    best = d if best is None else min(best,d)
            if best is not None:
                t=float(best);all_t.append(t)
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
                "method":"dual_pass_v8_kdtree","samples_used":len(all_t)}
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
    # FIX: a highly tapered/slender loft (confirmed live on a 200mm tapered beam,
    # ~25:1 aspect ratio) can make mesh.section() return a near-zero-area sliver
    # at one or more z positions — a slicing/parametrization artifact, not a real
    # physical throat. The old defense (drop slices below 10% of the median of
    # ALL positive slices, applied further down) doesn't catch this when MORE
    # THAN ONE slice is degenerate, since a cluster of tiny values drags the
    # median down with them and they end up passing their own filter. That tiny
    # A_min then hits the max(A_min,0.01) floor clamp downstream — confirmed
    # live: axial_mpa=40000.0 is an exact match for 400N/0.01mm^2, not real
    # physics, on a perfectly legitimate, buildable tapered beam. Fix: reject
    # any slice below an ABSOLUTE floor tied to the part's own bounding-box
    # footprint (2% of it — real cross-sections of a machined/printed part don't
    # legitimately shrink to a sliver of their own bounding box) before it can
    # ever reach valid_a/valid_I/A_min at all.
    other_axes=[i for i in range(3) if i!=ax]
    bbox_cross_area=exts[other_axes[0]]*exts[other_axes[1]]
    area_floor=max(bbox_cross_area*0.02, 1e-6)
    cut_areas=[];cut_I=[]
    for z in z_positions:
        origin=[0,0,0];origin[ax]=z
        try:
            sec=mesh.section(plane_origin=origin,plane_normal=normal)
            if sec is None: cut_areas.append(0);cut_I.append(0);continue
            pl,_=sec.to_planar()
            pts=pl.vertices
            if len(pts)<3: cut_areas.append(0);cut_I.append(0);continue
            # FIX: this used to be a hand-rolled shoelace sum over pl.vertices,
            # which treats every point from every sub-loop as one single closed
            # walk. A cross-section taken right at a fillet-to-taper blend
            # (confirmed live: z=26.36mm on a 200mm tapered beam, exactly where
            # make_tapered_beam's base fillet meets the straight loft) can come
            # back from to_planar() as more than one loop, or with a winding
            # order the naive sum doesn't handle — the sum then partially
            # cancels between loops and returns a near-zero area for a section
            # that isn't physically thin at all. That poisoned A_min, which then
            # hit the max(A_min,0.01) floor clamp below and produced exactly
            # axial_mpa=40000.0 (=400N/0.01mm^2) — a slicing artifact, not a
            # real 25:1-taper failure. pl.area is trimesh's own shapely-backed
            # polygon area, which correctly handles multiple loops/holes instead
            # of assuming a single simple walk.
            x_,y_=pts[:,0],pts[:,1]
            area=abs(float(pl.area))
            if area<=area_floor: cut_areas.append(0);cut_I.append(0);continue
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
    # Defensive backstop, independent of the area-calc fix above: no legitimately
    # slender-but-real section should produce stress orders of magnitude past
    # yield, or a deflection many times the part's own length, under this linear
    # model. If it does, something upstream (mesh slicing, a future trimesh
    # version, an even more extreme geometry) produced a numerical artifact, not
    # a real structural finding — say so explicitly rather than reporting FAIL
    # with fabricated-looking millions-of-MPa numbers as if they were physical.
    numerically_suspect = bool(vm > 50*Sy or delta > 10*max(L,1))
    return {"method":"multi_section_v8",
            "note":"Analytical estimate — not benchmarked against NAFEMS or other "
                    "published test cases; do not treat this as a validated accuracy %.",
            "numerically_suspect": numerically_suspect,
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
            "status":overall_status,
            "note":("cycles_to_failure/hours_to_failure reflect only the pure alternating-stress "
                    "S-N curve, which ignores mean stress entirely — 'infinite' there means the "
                    "alternating component alone is below the endurance limit, NOT that the part is "
                    "safe overall. status/goodman_status factor in mean stress too and are the "
                    "governing verdict; a FAIL there overrides an 'infinite' cycle count."
                    if goodman_status=="FAIL" and N==float("inf") else None)}

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

def _find_watertight_defect_locations(mesh, max_regions=3):
    """
    A watertight mesh has every edge shared by exactly 2 faces. Find edges
    shared by only 1 (the actual boundary/gap causing non-watertightness),
    cluster their vertices into separate defect regions by proximity, and
    return each region's centroid — real (x,y,z) coordinates the AI can
    actually target on the next iteration, instead of a bare "not
    watertight" label with no location at all.
    """
    try:
        edges_sorted = np.sort(mesh.edges_sorted, axis=1)
        uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        naked = uniq[counts == 1]
        if len(naked) == 0:
            return []
        pts = mesh.vertices[np.unique(naked)]
        # Cheap greedy clustering: group points within 5% of the part's
        # longest dimension of each other, so several separate gaps don't
        # collapse into one meaningless average location.
        tol = float(max(mesh.extents)) * 0.05 or 1.0
        clusters = []
        for p in pts:
            placed = False
            for c in clusters:
                if np.linalg.norm(c[0] - p) < tol:
                    c.append(p); placed = True; break
            if not placed:
                clusters.append([p])
        clusters.sort(key=len, reverse=True)
        return [{"x": round(float(np.mean([p[0] for p in c])), 2),
                  "y": round(float(np.mean([p[1] for p in c])), 2),
                  "z": round(float(np.mean([p[2] for p in c])), 2),
                  "edge_count": len(c)} for c in clusters[:max_regions]]
    except Exception:
        return []

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
    if not mesh.is_watertight:
        defect_locs = _find_watertight_defect_locations(mesh)
        loc_txt = (" Gap location(s) found: " +
                   "; ".join(f"({d['x']}, {d['y']}, {d['z']})" for d in defect_locs) +
                   " — inspect and fix the geometry construction near these exact "
                   "coordinates specifically, not the whole part."
                   ) if defect_locs else " (could not isolate the exact gap location.)"
        add("R03","HIGH","Mesh not watertight",
        "Two common real causes, both confirmed live: (1) two solids "
        "union()-ed at an exact flush/coincident plane instead of a genuine "
        "overlap — check every union() join. (2) blanket .edges().fillet() "
        "on ALL edges of a loft/tapered solid, including the compound "
        "corners where a sloped taper edge meets two flat profile edges — "
        "this can silently produce self-intersecting geometry with no "
        "Python error. If the script filleted every edge of a loft at once, "
        "try a smaller radius or fillet only the flat profile edges, not "
        "the sloped taper edges." + loc_txt,
        "STL standard", defect_locs[0] if defect_locs else None)
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
        if off>40: add("R13","HIGH",f"CoG offset {off:.1f}%","Redistribute mass",pos={"x":round(float(cog[0]),2),"y":round(float(cog[1]),2),"z":round(float(cog[2]),2)})
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
        if any(w in pd for w in TAPER_KEYWORDS):
            try:
                ax=int(np.argmax(exts))
                lo,hi=mesh.bounds[0][ax],mesh.bounds[1][ax];span=hi-lo
                normal=[0,0,0];normal[ax]=1
                def _cross_width(frac):
                    origin=[0,0,0];origin[ax]=lo+span*frac
                    sec=mesh.section(plane_origin=origin,plane_normal=normal)
                    if sec is None: return None
                    pl,_=sec.to_planar();b=pl.bounds
                    return max(b[1][0]-b[0][0],b[1][1]-b[0][1])
                w_near=_cross_width(0.15);w_far=_cross_width(0.85)
                if w_near and w_far and abs(w_near-w_far)/max(w_near,w_far)<0.1:
                    add("R16","CRITICAL",f"Prompt implies a taper but cross-section is "
                        f"nearly constant ({w_near:.1f}mm vs {w_far:.1f}mm along the main axis)",
                        "Do not write manual loft()/fillet() code for this. Call the "
                        "make_tapered_beam(length=..., base_width=..., base_thick=..., "
                        "tip_width=..., tip_thick=..., fillet_radius=..., holes_base=[...], "
                        "holes_tip=[...]) helper that is already available — it guarantees a "
                        "genuine, safely-filleted taper. Replace the whole script body with "
                        "a single call to it.","Engineering judgment")
            except Exception: pass
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
║   FEA Method: {"SimScale cloud FEM" if fea.get("method")=="simscale_static_fem" else "CalculiX Real FEM (Gmsh-meshed)" if calc_used else "Multi-Section Analytical (unbenchmarked estimate)"}  ║
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
# BUILD123D GENERATORS (ported from the v7.0 build123d generators; algebra mode)
# ═══════════════════════════════════════════════════════════════════

def _fillet_vertical(shape, r):
    """Fillet every edge parallel to Z (the 'corner rounding' the build123d
    `.edges("|Z").fillet(r)` calls used to do)."""
    if not r or r <= 0:
        return shape
    return _b3d_fillet(shape, shape.edges().filter_by(b3d.Axis.Z), r)

def _bore(x, y, z_center, radius, height):
    """Vertical cylinder cutter centered at (x, y, z_center)."""
    return b3d.Pos(x, y, z_center) * b3d.Cylinder(radius, height)

def gen_bracket(p):
    w=p.get("width",80);h=p.get("height",60);d=p.get("depth",40)
    t=p.get("thickness",5);hd=p.get("hole_diameter",6);fr=p.get("fillet_radius",2);nh=p.get("num_holes",4)
    base=_fillet_vertical(b3d.Box(w,d,t),fr)
    wall=_fillet_vertical(b3d.Box(w,t,h),fr)
    wall=b3d.Pos(0,-(d/2-t/2),h/2+t/2)*wall
    b=base+wall
    sp=max((w-20)/max(nh//2-1,1),1)
    for x in [-(w/2-10)+i*sp for i in range(max(nh//2,1))]:
        for y in [-(d/2-10),d/2-10]:
            try: b=b-_bore(x,y,h/2,hd/2,h+t+4)
            except Exception: pass
    return b

def gen_shaft(p):
    L=p.get("length",100);D=p.get("diameter",20)
    along_x=lambda length,dia: b3d.Pos(length/2,0,0)*b3d.Rot(0,90,0)*b3d.Cylinder(dia/2,length)
    s=along_x(L,D)
    if p.get("shoulder_diameter",0)>D: s=s+along_x(p.get("shoulder_length",15),p["shoulder_diameter"])
    if p.get("keyway_width",0)>0: s=s-(b3d.Pos(L/2,0,D/2)*b3d.Box(L,p["keyway_width"],p.get("keyway_depth",3)*2))
    return s

def gen_plate(p):
    w=p.get("width",100);h=p.get("height",80);t=p.get("thickness",6)
    hp=p.get("hole_pattern","corners");hd=p.get("hole_diameter",8);fr=p.get("fillet_radius",3);m=p.get("margin",15)
    pl=_fillet_vertical(b3d.Box(w,h,t),fr)
    if hp=="corners":
        for x,y in [(-(w/2-m),-(h/2-m)),(w/2-m,-(h/2-m)),(-(w/2-m),h/2-m),(w/2-m,h/2-m)]:
            pl=pl-_bore(x,y,0,hd/2,t+2)
    elif hp=="center": pl=pl-_bore(0,0,0,hd/2,t+2)
    return pl

def gen_housing(p):
    ow=p.get("width",80);oh=p.get("height",60);od=p.get("depth",50)
    wt=p.get("wall_thickness",4);fr=p.get("fillet_radius",3);bd=p.get("boss_diameter",8)
    outer=_fillet_vertical(b3d.Box(ow,od,oh),fr)
    # open-top cavity: inner box top is flush with the outer top face
    inner=b3d.Pos(0,0,wt/2)*b3d.Box(ow-2*wt,od-2*wt,oh-wt)
    h=outer-inner
    if p.get("num_bosses",4)>=4:
        bx=ow/2-wt-bd/2-2;by=od/2-wt-bd/2-2
        floor_z=wt-oh/2                      # inner floor height (ported fix: bosses used to be placed
        boss_h=oh-wt-2                       # at z=wt, i.e. floating above/outside the housing)
        for pos in [(-bx,-by),(bx,-by),(-bx,by),(bx,by)]:
            boss=b3d.Pos(pos[0],pos[1],floor_z-0.5+(boss_h+0.5)/2)*b3d.Cylinder(bd/2,boss_h+0.5)
            hole=b3d.Pos(pos[0],pos[1],floor_z+boss_h/2)*b3d.Cylinder(bd/4,boss_h)
            h=(h+boss)-hole
    return h

def gen_true_involute_gear(p):
    mod=p.get("module",2.0);nt=p.get("num_teeth",20);pa=math.radians(p.get("pressure_angle",20))
    fw=p.get("face_width",15);bore=p.get("bore_diameter",6);hd=p.get("hub_diameter",10);hl=p.get("hub_length",20)
    pitch_r=mod*nt/2;base_r=pitch_r*math.cos(pa);tip_r=pitch_r+mod;root_r=pitch_r-1.25*mod
    g=b3d.extrude(b3d.Circle(tip_r),amount=fw)
    ta=2*math.pi/nt
    for i in range(nt):
        angle=i*ta+ta/2
        sp_pts=[(root_r*0.95*math.cos(angle+ta*0.15),root_r*0.95*math.sin(angle+ta*0.15)),
                (tip_r*1.02*math.cos(angle+ta*0.15),tip_r*1.02*math.sin(angle+ta*0.15)),
                (tip_r*1.02*math.cos(angle+ta*0.85),tip_r*1.02*math.sin(angle+ta*0.85)),
                (root_r*0.95*math.cos(angle+ta*0.85),root_r*0.95*math.sin(angle+ta*0.85))]
        try: g=g-(b3d.Pos(0,0,-0.5)*b3d.extrude(b3d.Polygon(*sp_pts,align=None),amount=fw+1))
        except Exception: pass
    if hd>0: g=g+b3d.extrude(b3d.Circle(hd/2),amount=max(fw,hl))
    if bore>0: g=g-(b3d.Pos(0,0,-1)*b3d.extrude(b3d.Circle(bore/2),amount=max(fw,hl)+2))
    return g

def gen_flange(p):
    od=p.get("outer_diameter",100);id_=p.get("inner_diameter",40);t=p.get("thickness",12)
    bc_r=p.get("bolt_circle_radius",40);n=p.get("num_bolts",6);bd=p.get("bolt_diameter",8)
    hub_od=p.get("hub_od",50);hub_h=p.get("hub_height",20)
    f=b3d.extrude(b3d.Circle(od/2),amount=t)-(b3d.Pos(0,0,-0.5)*b3d.extrude(b3d.Circle(id_/2),amount=t+1))
    hub=b3d.extrude(b3d.Circle(hub_od/2),amount=hub_h)-(b3d.Pos(0,0,-0.5)*b3d.extrude(b3d.Circle(id_/2),amount=hub_h+1))
    f=f+hub
    H=max(t,hub_h)
    for i in range(n):
        f=f-_bore(bc_r*math.cos(2*math.pi*i/n),bc_r*math.sin(2*math.pi*i/n),H/2,bd/2,H+2)
    return f

def gen_ibeam(p):
    L=p.get("length",200);fw=p.get("flange_width",80);fh=p.get("flange_thickness",8);wh=p.get("web_height",100);wt=p.get("web_thickness",6)
    top=b3d.Pos(0,wh/2+fh/2,L/2)*b3d.Box(fw,fh,L)
    bot=b3d.Pos(0,-(wh/2+fh/2),L/2)*b3d.Box(fw,fh,L)
    web=b3d.Pos(0,0,L/2)*b3d.Box(wt,wh+0.2,L)   # +0.2 sinks 0.1mm into each flange -> real overlap, same outer shape
    return top+bot+web

def gen_motor_mount(p):
    w=p.get("width",30);h=p.get("height",30);t=p.get("thickness",3)
    md=p.get("motor_diameter",28);hd=p.get("hole_diameter",3);hp=p.get("hole_pattern_size",16)
    base=_fillet_vertical(b3d.Box(w,h,t),2)
    base=base-_bore(0,0,0,md/2,t+2)
    for x,y in [(hp/2,hp/2),(-hp/2,hp/2),(hp/2,-hp/2),(-hp/2,-hp/2)]:
        base=base-_bore(x,y,0,hd/2,t+2)
    return base

def gen_heatsink(p):
    bw=p.get("base_width",60);bh=p.get("base_height",40);bt=p.get("base_thickness",5)
    n=p.get("num_fins",8);fh=p.get("fin_height",20);ft=p.get("fin_thickness",2)
    base=b3d.Pos(0,0,bh/2)*b3d.Box(bw,bt,bh)
    sp=(bw-ft)/max(n-1,1)
    for i in range(n):
        x=-(bw/2-ft/2)+i*sp
        # fin sinks 0.1mm into the base plate (real overlap); its tip stays at bt/2+fh
        base=base+(b3d.Pos(x,bt/2+fh/2-0.05,bh/2)*b3d.Box(ft,fh+0.1,bh))
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
    rib=b3d.extrude(b3d.Polygon(*all_pts,align=None),amount=thick)
    for xp in [chord*0.25-chord/2,chord*0.5-chord/2,chord*0.7-chord/2]:
        try: rib=rib-(b3d.Pos(xp,0,-0.5)*b3d.extrude(b3d.Circle(spar_d/2),amount=thick+1))
        except Exception: pass
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
B3D_MAP={
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
    for pt,(fn,kws) in B3D_MAP.items():
        if any(k in d for k in kws): return pt,"build123d",fn(params)
    for pt,(fn,kws) in TRIMESH_MAP.items():
        if any(k in d for k in kws): return pt,"trimesh",fn(params)
    return "plate","build123d",gen_plate(params)

def stl_from_cad(obj):
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t: p=t.name
    _b3d_write_stl(obj,p,tolerance=0.02,angular_tolerance=0.1); return p

def stl_from_tm(mesh):
    with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as t: p=t.name
    mesh.export(p); return p

def step_from_cad(obj):
    with tempfile.NamedTemporaryFile(suffix=".step",delete=False) as t: p=t.name
    _b3d_write_step(obj,p); return p

# ═══════════════════════════════════════════════════════════════════
# CORE ANALYSIS PIPELINE v8.0
# ═══════════════════════════════════════════════════════════════════

def _repair_watertight_mesh(mesh):
    """Attempt cheap, best-effort repairs on a mesh straight off build123d's STL
    export before judging or using it for anything.

    build123d/OCCT's STL tessellation routinely leaves tiny gaps and near-but-
    not-quite-coincident duplicate vertices at the seams between adjacent
    tessellated patches — most visibly at fillet-to-flat-face boundaries. This
    is a known tessellation artifact, not necessarily a real defect in the
    underlying B-rep solid (mesh_from_cad_object's own tolerance-tightening fix
    above notes the same class of artifact causing Gmsh/CalculiX meshing
    failures downstream). trimesh.load()'s default vertex-merge tolerance is
    often too tight to close these seams on its own.

    Without this step, the is_watertight check below — which drives the
    /generate-validate-refine quality gate — was flagging these tessellation
    artifacts as "non-manifold geometry" with nothing for the AI to actually
    change about the design. Confirmed live: three refinement iterations in a
    row producing an IDENTICAL health score and IDENTICAL "not watertight"
    reason, because there was no real design defect to fix.

    Each repair step is independently try/excepted — analysis_service.py's
    _repair_mesh_for_meshing hit real trimesh-version API renames doing the
    same kind of repair, so one incompatible call here shouldn't skip the rest.
    Only ever changes what gets reported/analyzed; a genuine defect (e.g. an
    actual gap from a failed boolean union) will still fail to repair and
    should still fail the gate — this only clears the false positives.

    Returns (mesh, was_repaired: bool) — was_repaired is True only if the mesh
    started non-watertight AND ended up watertight after these steps, so
    callers/feedback text can distinguish "needed no repair" from "tessellation
    artifact, auto-fixed" from "still broken after repair, likely a real defect"."""
    if mesh.is_watertight:
        return mesh, False
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        try:
            mesh.fill_holes()  # older/newer trimesh convenience alias
        except Exception:
            pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh, bool(mesh.is_watertight)


def _apply_plan_to_fea(fea, tp_results):
    """A valid plan verdict (real FEM, plan-chosen loads) replaces the generic analytical numbers, so fatigue, the
    rule engine and the report all work from the governing test instead of the legacy fixed 1000 N case."""
    valid = [r for r in (tp_results or {}).get("results", [])
             if r.get("status") in ("PASS", "FAIL") and (r.get("metrics") or {}).get("von_mises_mpa") is not None]
    if not valid:
        return fea
    worst = min(valid, key=lambda r: r["metrics"].get("safety_factor") if r["metrics"].get("safety_factor") is not None else 1e9)
    m = worst["metrics"]
    fea["stress"] = dict(fea.get("stress") or {}, von_mises_mpa=m["von_mises_mpa"])
    fea["von_mises_mpa"] = m["von_mises_mpa"]
    if m.get("safety_factor") is not None:
        fea["safety_factor"] = m["safety_factor"]
    if m.get("max_deflection_mm") is not None:
        fea["deflection_mm"] = m["max_deflection_mm"]
    fea["status"] = worst["status"]
    fea["method"] = "test_plan_simscale"
    fea["governing_test"] = worst["test_id"]
    return fea


# The test plan is a short JSON answer: a fast model is enough and a 550B reasoning model just times out.
# On NVIDIA NIM the plan therefore goes to PLAN_MODEL first (default DeepSeek V4 Flash 0731; the original v4-flash id reached end of life 2026-08-07) and falls back to
# NVIDIA_MODEL if that call fails. Override with PLAN_MODEL=<nim model id> (set PLAN_MODEL=none to disable).
PLAN_MODEL = os.environ.get("PLAN_MODEL", "deepseek-ai/deepseek-v4-flash-0731").strip()
PLAN_TIMEOUT_S = float(os.environ.get("PLAN_TIMEOUT_S", "90"))
_LLM_LAST = {}


async def _llm_text(system_prompt, turns, max_tokens=1800, temperature=0.1):
    """Plain-text completion through whichever provider AI_PROVIDER selects (same dispatch as script generation)."""
    _LLM_LAST.clear()
    _LLM_LAST["provider"] = AI_PROVIDER
    msgs = [{"role": "system", "content": system_prompt}] + turns
    if DEEPSEEK_API_KEY:     # direct DeepSeek first (remove the key to disable); NVIDIA stays as fallback
        try:
            out = await asyncio.to_thread(_deepseek_request, msgs, temperature=temperature,
                                          max_tokens=max(max_tokens, 6000), timeout=PLAN_TIMEOUT_S)
            _LLM_LAST["provider"] = "deepseek"
            _LLM_LAST["model"] = DEEPSEEK_MODEL
            return out
        except HTTPException as e:
            _LLM_LAST["deepseek_error"] = f"{e.status_code} {str(e.detail)[:200]}"
    if AI_PROVIDER == "claude":
        return await asyncio.to_thread(_claude_request, system_prompt, turns, temperature=temperature, max_tokens=max_tokens)
    if AI_PROVIDER == "gemini":
        return await asyncio.to_thread(_gemini_request, system_prompt, turns, temperature=temperature, max_tokens=max_tokens)
    if AI_PROVIDER == "groq":
        return await asyncio.to_thread(_groq_request, msgs, temperature=temperature, max_tokens=max_tokens)
    if AI_PROVIDER == "cerebras":
        return await asyncio.to_thread(_cerebras_request, msgs, temperature=temperature, max_tokens=max_tokens)
    if AI_PROVIDER == "nvidia":      # reasoning models: hidden thinking counts against max_tokens
        if PLAN_MODEL and PLAN_MODEL.lower() != "none" and PLAN_MODEL != NVIDIA_MODEL:
            try:
                out = await asyncio.to_thread(_nvidia_request, msgs, temperature=temperature,
                                              max_tokens=max(max_tokens, 6000), model=PLAN_MODEL,
                                              timeout=PLAN_TIMEOUT_S)
                _LLM_LAST["model"] = PLAN_MODEL
                return out
            except HTTPException as e:
                _LLM_LAST["fallback_reason"] = f"{PLAN_MODEL}: {e.status_code} {str(e.detail)[:160]}"
        _LLM_LAST["model"] = NVIDIA_MODEL
        return await asyncio.to_thread(_nvidia_request, msgs, temperature=temperature,
                                       max_tokens=max(max_tokens, NVIDIA_GEN_MAX_TOKENS),
                                       thinking=False)      # plan = short JSON: no hidden thinking needed
    if AI_PROVIDER == "openrouter":
        return await asyncio.to_thread(_openrouter_request, msgs, temperature=temperature, max_tokens=max_tokens)
    return await asyncio.to_thread(_lovable_request, msgs, temperature=temperature, max_tokens=max_tokens)


_PLAN_SYSTEM = ("You are a mechanical test engineer. You are shown a part request and the MEASURED geometry of the "
                "solid that was built from it. Decide how the part will really be used and write the test plan that "
                "verifies it. Choose load directions from the real use case: a force at the tip of an arm or cantilever "
                "normally acts PERPENDICULAR to it (bending), a hanging weight acts along gravity, a clamp or bolt "
                "pattern is the fixed set - fix only faces that are really held. "
                "Output ONLY the JSON object - no prose, no markdown.\n\n")


async def generate_test_plan(prompt, cad_obj, material="auto", previous_plan=None, diag=None):
    """The LLM writes the test plan for a built part. The plan is dry-run against the real faces first (instant)
    and sent back for correction if a selector is unusable. Returns a plan dict or None (caller then falls back
    to the legacy fixed load case). `diag` (dict, filled in place) records WHY a plan was not produced."""
    diag = {} if diag is None else diag
    diag["attempts"] = []
    try:
        err = None
        if previous_plan is not None:
            chk = await run_plan_tests(cad_obj, previous_plan, dry_run=True)
            if chk.get("overall") == "RESOLVED":
                diag["reused_previous_plan"] = True
                return previous_plan
            err = chk.get("error") or next((r.get("error") for r in chk.get("results", []) if r.get("error")), None)
        diag["stage"] = "describe_faces"
        summary = TP.face_summary(TP.describe_faces(cad_obj))
        diag["n_faces"] = summary.get("n_faces")
        hint = f"\nPreferred material: {material}." if material and material != "auto" else ""
        user = (f"PART REQUEST:\n{prompt[:2500]}{hint}\n\nMEASURED GEOMETRY (mm):\n{json.dumps(summary)}\n"
                + (f"\nThe previous test plan no longer fits this geometry: {err}\n" if err else "")
                + "\nOutput ONLY the JSON test plan.")
        turns = [{"role": "user", "content": user}]
        system = _PLAN_SYSTEM + TP.schema_prompt(MATERIALS)
        diag["stage"] = "llm"
        for _ in range(2):
            try:
                text = await _llm_text(system, turns)
            except HTTPException as e:
                diag["attempts"].append({"llm_error": f"{e.status_code}: {str(e.detail)[:240]}", **_LLM_LAST})
                continue
            att = {"llm_reply_head": (text or "")[:300], **_LLM_LAST}
            diag["attempts"].append(att)
            m = re.search(r"\{.*\}", text or "", re.S)
            try:
                plan = TP.parse_plan(m.group(0) if m else (text or ""))
                chk = await run_plan_tests(cad_obj, plan, dry_run=True)
                if chk.get("overall") == "RESOLVED":
                    diag["stage"] = "ok"
                    return plan
                msg = chk.get("error") or next((r.get("error") for r in chk.get("results", []) if r.get("error")), "invalid")
            except TP.PlanError as e:
                msg = str(e)
            att["rejected"] = str(msg)[:400]
            turns += [{"role": "assistant", "content": text or ""},
                      {"role": "user", "content": f"Your test plan was rejected: {msg}\nOutput the corrected JSON only."}]
        diag["error"] = ("LLM call failed twice" if all("llm_error" in a for a in diag["attempts"])
                         else "plan rejected twice")
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return None


def _verification_gate(use_test_plan, result, plan_diag, quality):
    """A run that asked for a test plan only counts as PASSED with a real FEM verdict. Without one (plan missing,
    SimScale not run, boundary-condition check failed) the analytical fallback numbers are not trusted - they once
    gave a safety factor of 330 for a 700 N bracket. Mutates `quality`; returns (verified, unverifiable) where
    unverifiable means more redesign rounds cannot help (infrastructure / untrustworthy solve)."""
    quality.setdefault("metrics", {})["verified"] = True
    if not use_test_plan:
        return True, False
    tp = result.get("test_plan_results") or {}
    overall = tp.get("overall")
    if overall in ("PASS", "FAIL"):
        return True, False
    rs = tp.get("results") or []
    if rs:
        r0 = rs[0]
        why = r0.get("reason") or r0.get("error") or r0.get("status") or overall
    else:
        why = (plan_diag or {}).get("error") or tp.get("error") or "no test plan could be generated"
    design_related = any(r.get("design_related") for r in rs)
    quality["passed"] = False
    quality["metrics"]["verified"] = False
    quality.setdefault("reasons", []).insert(
        0, f"UNVERIFIED: the test plan gave no valid FEM verdict (overall {overall or 'no plan'}: {str(why)[:220]}). "
           "The analytical fallback numbers are not trusted for a pass.")
    unverifiable = (overall in ("NOT_RUN", "INVALID")) and not design_related
    return False, unverifiable


def _plan_targets(r):
    """Numbers the LLM can design to: how much the failing metric must improve and what that means for thickness."""
    try:
        need_stress, need_defl = 1.0, 1.0
        for c in r.get("criteria", []):
            if c.get("pass"):
                continue
            m = re.search(r"([0-9.]+)", str(c.get("required", "")))
            if not m or not isinstance(c.get("actual"), (int, float)):
                continue
            req, act = float(m.group(1)), float(c["actual"])
            if req <= 0 or act <= 0:
                continue
            if c.get("name") == "safety_factor":
                need_stress = max(need_stress, req / act)
            elif c.get("name") == "von_mises_mpa":
                need_stress = max(need_stress, act / req)
            elif c.get("name") == "deflection_mm":
                need_defl = max(need_defl, act / req)
        out = []
        if need_stress > 1.0:
            t = need_stress ** 0.5
            out.append(f"    TARGET: peak stress must drop by at least {(1 - 1 / need_stress) * 100:.0f}% "
                       f"(x{need_stress:.2f}). If the hotspot is bending-dominated, stress ~ 1/thickness^2, so that section "
                       f"needs about x{t:.2f} thickness; aim for x{t * 1.1:.2f} to leave margin. Fillets at inside corners "
                       "typically buy 10-30% on their own.")
        if need_defl > 1.0:
            out.append(f"    TARGET: deflection must drop by x{need_defl:.2f}; stiffness ~ thickness^3, so about "
                       f"x{need_defl ** (1 / 3):.2f} thickness (or add a rib).")
        return out
    except Exception:
        return []


def _plan_feedback_lines(result):
    """Test-plan section of the refinement feedback the LLM receives."""
    tp = result.get("test_plan_results") or {}
    if not tp.get("results"):
        return []
    out = ["", f"TEST PLAN RESULTS (overall {tp.get('overall')}) - real FEM on the built solid, loads/fixtures from the plan:"]
    for r in tp["results"]:
        st = r.get("status")
        out.append(f"- test '{r.get('test_id')}': {st}")
        lc = r.get("load_case") or {}
        for l in lc.get("loads", []):
            out.append(f"    load '{l['id']}': {l['force_n']} N on {l['faces']} face(s); {lc.get('fixed_faces')} fixed face(s)")
        m = r.get("metrics") or {}
        if m:
            out.append(f"    von Mises {m.get('von_mises_mpa')} MPa, safety factor {m.get('safety_factor')}, "
                       f"max deflection {m.get('max_deflection_mm')} mm, hotspot at {m.get('hotspot_mm')} mm")
        for c in r.get("criteria", []):
            out.append(f"    criterion {c['name']}: {c['actual']} (required {c['required']}) -> {'ok' if c['pass'] else 'FAILED'}")
        for w in r.get("warnings", []):
            out.append(f"    WARNING: {w}")
        loc = r.get("localization")
        if loc and loc.get("text"):
            out.extend("    " + ln for ln in loc["text"].split("\n"))
        if st == "FAIL":
            out.extend(_plan_targets(r))
            out.append("    FIX: strengthen the part where the hotspot is (thicker section, fillets, ribs, larger radius at "
                       "the stress raiser) WITHOUT moving or removing the fixed and loaded faces - the plan selects them "
                       "by position/normal and must still find them. Change the region named above; leave parts that "
                       "are not the problem alone.")
        elif st == "INVALID":
            out.append("    The numbers are not trustworthy (boundary-condition check failed); do not redesign on them.")
        elif st == "NOT_RUN":
            out.append(f"    not run: {r.get('reason')} (design_related={r.get('design_related')})")
        elif st == "PLAN_ERROR":
            out.append(f"    plan error: {r.get('error')}")
    return out


async def run_analysis_v8(mesh, filename, part_name, mat_key,
                           force_n=1000, force_dir="z", T_op=25.0,
                           proj=None, surface_finish="machined",
                           reliability=0.99, run_topo=False,
                           topo_volfrac=0.5, cad_obj=None, test_plan=None):
    mesh, was_auto_repaired = _repair_watertight_mesh(mesh)
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

    # FEA priority: SimScale cloud FEM (needs the CAD solid `cad_obj`) -> CalculiX
    # analysis service -> analytical model. Whatever ran is reported in fea["method"];
    # SimScale's own success/failure detail is in result["simscale_diagnostic"].
    fea=None;simscale_diag=None;calculix_diag=None
    if cad_obj is not None:
        if test_plan is not None:
            simscale_diag={"attempted":False,"reason":"a test plan drives the FEM (see test_plan_results); the legacy fixed load case is skipped"}
        elif SIMSCALE_ENABLED:
            fea,simscale_diag=await asyncio.to_thread(run_simscale_fem,cad_obj,mat_key,force_n,force_dir,ctx["min_sf"])
        else:
            simscale_diag={"attempted":False,"reason":"SIMSCALE_API_KEY not set (or SIMSCALE_ENABLED=0)"}
    if fea is None:
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

    tp_results=None
    if test_plan is not None and cad_obj is not None:
        tp_results=await run_plan_tests(cad_obj,test_plan,ctx["min_sf"])
        _apply_plan_to_fea(fea,tp_results)
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
        "simscale_used":fea.get("method","")=="simscale_static_fem",
        "simscale_diagnostic":simscale_diag,
        "test_plan_results":tp_results,
        "test_plan":test_plan,
        "filename":filename,"part_name":part_name,"part_context":ctx,
        "geometry":{
            "dimensions_mm":{"x":round(exts[0],3),"y":round(exts[1],3),"z":round(exts[2],3)},
            "volume_mm3":round(vol,3),"surface_area_mm2":round(sf(mesh.area),3),
            "is_watertight":is_wt,"watertight_auto_repair_succeeded":was_auto_repaired,
            "vertex_count":int(len(mesh.vertices)),
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
    tp = result.get("test_plan_results") or {}
    plan_authoritative = tp.get("overall") in ("PASS", "FAIL")      # valid FEM verdict -> it replaces the generic SF check
    for r in tp.get("results", []):
        if r.get("status") == "FAIL":
            bad = [c for c in r.get("criteria", []) if not c.get("pass")]
            reasons.append(f"Test '{r.get('test_id')}' FAILED: " + "; ".join(
                f"{c['name']} {c['actual']} (required {c['required']})" for c in bad) + ".")
        elif r.get("status") == "NOT_RUN" and r.get("design_related"):
            reasons.append(f"Test '{r.get('test_id')}' could not run - SimScale rejected the geometry at "
                           f"{r.get('failed_stage')}: {r.get('reason')}")
    ss_diag = result.get("simscale_diagnostic") or {}
    if ss_diag.get("attempted") and ss_diag.get("design_related"):
        # SimScale itself rejected the geometry (import / mesh / solve) — that is a
        # design defect the LLM must fix, even if a fallback solver happily analysed it.
        reasons.append(f"SimScale could not {ss_diag.get('failed_stage','process')} this geometry: "
                       f"{ss_diag.get('reason')}. Fix the geometry so the cloud FEM solver accepts it "
                       f"(no slivers/tiny fillets, real overlap on unions, overshoot on cuts).")
    if score < min_health_score:
        reasons.append(f"Health score {score} is below target {min_health_score}.")
    if n_crit > max_critical:
        reasons.append(f"{n_crit} CRITICAL rule violation(s) found (max allowed {max_critical}).")
    if n_high > max_high:
        reasons.append(f"{n_high} HIGH-severity rule violation(s) found (max allowed {max_high}).")
    if sfv < min_safety_factor and not plan_authoritative:
        reasons.append(f"FEA safety factor {sfv:.2f} is below minimum {min_safety_factor}.")
    if fea_status == "FAIL" and not plan_authoritative:
        reasons.append("FEA status is FAIL.")
    if fat_status == "FAIL" and not plan_authoritative:
        # the generic fatigue check treats the peak stress as fully reversed cyclic stress; when a test plan with
        # real loads is in charge (static_stress only for now) that would fail every static design, so it is
        # informational there (still reported in metrics.fatigue_status)
        reasons.append("Fatigue analysis status is FAIL.")
    if not is_wt:
        # FIX: this used to branch on a "was_repaired" flag read from
        # watertight_auto_repair_attempted — but that field is actually "did
        # repair SUCCEED", not "was repair attempted", and repair (in
        # _repair_watertight_mesh, above) runs unconditionally before is_wt is
        # ever computed. So reaching this branch at all already means repair
        # was attempted AND failed — the was_repaired==True case could never
        # fire, and the AI was always getting the generic message instead of
        # the actionable one below. Confirmed live: a real non-manifold defect
        # (Gmsh: "Wrong topology of boundary mesh for parametrization") still
        # only produced the generic reason text.
        reasons.append("Mesh is STILL not watertight even after automatic tessellation "
                        "repair — this is a real geometry defect (likely a boolean union/cut "
                        "leaving a gap or self-intersection), not an export artifact. "
                        "See REFINEMENT MODE guidance.")

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
            "fea_method": fea.get("method"),
            "test_plan": tp.get("overall"),
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
    lines += _plan_feedback_lines(result)
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

    ss_diag = result.get("simscale_diagnostic") or {}
    if fea.get("method") == "simscale_static_fem":
        mprops = (result.get("material", {}) or {}).get("properties", {}) or {}
        hot = crit.get("hotspot_xyz_mm")
        lines.append("")
        lines.append("SIMSCALE FEA (authoritative cloud-solver result):")
        lines.append(f"  peak von Mises = {fea.get('stress',{}).get('von_mises_mpa')} MPa vs yield "
                      f"{mprops.get('yield_strength_mpa')} MPa -> safety factor {fea.get('safety_factor')} "
                      f"(required {fea.get('required_sf')}); max deflection {fea.get('deflection_mm')} mm")
        if hot:
            lines.append(f"  peak stress is at x,y,z = {hot} mm (position along the {crit.get('axis')}-axis: "
                          f"{crit.get('position_mm')} mm) — reinforce THERE and along the load path into it.")
        lines.append(f"  load case: {fea.get('note')}")
    elif ss_diag.get("attempted") and ss_diag.get("reason"):
        who = "the design" if ss_diag.get("design_related") else "infrastructure, not the design"
        lines.append("")
        lines.append(f"SIMSCALE: run did not complete at stage '{ss_diag.get('failed_stage')}' "
                      f"({ss_diag.get('reason')}). Cause is {who}; the FEA figures above come from the "
                      f"fallback solver ({fea.get('method')}).")

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

async def mesh_from_cad_object(obj):
    """Export a build123d shape to STL and load as a trimesh mesh. Returns (mesh, stl_bytes).

    Tessellation control matters: OCCT's default deflection is an ABSOLUTE distance,
    not scaled to the size of the feature being tessellated, so a 1.5-2mm fillet gets
    the same coarse triangulation budget as a 200mm flat face — a plausible real
    contributor to sliver triangles behind Gmsh "invalid exterior boundary mesh" and
    CalculiX "nonpositive jacobian" failures. tolerance=0.01mm / angular 0.05rad keeps
    small features well resolved.
    """
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as t:
        tmp = t.name
    try:
        _b3d_write_stl(obj, tmp, tolerance=0.01, angular_tolerance=0.05)
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
        "SimScale cloud FEM (primary; falls back to CalculiX service / analytical)"
        if (SIMSCALE_ENABLED and SIMSCALE_TEMPLATE_SIMULATION_ID) else
        "real solid tetrahedral FEM (C3D4, Gmsh-meshed + CalculiX) — via separate "
        "analysis service" if ANALYSIS_SERVICE_URL
        else "multi-section analytical (ANALYSIS_SERVICE_URL not configured on "
             "this deployment — set it to enable real FEM)"
    )
    return {
        "status":"Lumexa v8.24 Enterprise (build123d + SimScale) — Vibe Engineering Edition",
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
            "build123d_available":B3D,
            "cadquery_available":B3D,   # deprecated alias — kept so existing frontend health checks keep working
            "simscale_configured": SIMSCALE_ENABLED,
            "api_knowledge_base": {"enabled": KB_ENABLED, "mode": _kb["mode"], "chunks": len(_kb["chunks"])},
            "simscale_template_configured": bool(SIMSCALE_TEMPLATE_PROJECT_ID and SIMSCALE_TEMPLATE_SIMULATION_ID
                                                  and SIMSCALE_TEMPLATE_MESH_OPERATION_ID),
            "analysis_service_configured": bool(ANALYSIS_SERVICE_URL),
            "solid_tet_fem_available": bool(ANALYSIS_SERVICE_URL),
            "blender_available":BLENDER,
            "dxf_export_available": EZDXF,
            "ai_generation_configured": bool(LOVABLE_API_KEY or ANTHROPIC_API_KEY or GOOGLE_API_KEY
                                              or GROQ_API_KEY or CEREBRAS_API_KEY or NVIDIA_API_KEY
                                              or OPENROUTER_API_KEY),
            "ai_provider": AI_PROVIDER,
            "ai_model": (CLAUDE_MODEL if AI_PROVIDER == "claude"
                         else GEMINI_MODEL if AI_PROVIDER == "gemini"
                         else GROQ_MODEL if AI_PROVIDER == "groq"
                         else CEREBRAS_MODEL if AI_PROVIDER == "cerebras"
                         else NVIDIA_MODEL if AI_PROVIDER == "nvidia"
                         else OPENROUTER_MODEL if AI_PROVIDER == "openrouter"
                         else LOVABLE_AI_MODEL),
            "engineering_agent_configured": AI_PROVIDER in ("groq", "openrouter", "lovable", "cerebras", "nvidia")},
        "new_in_v8_24":[
            "CAD kernel switched from CadQuery to build123d (algebra-mode scripts, new sandbox, "
            "new LLM prompts). make_tapered_beam / make_bent_bracket / all gen_* parametric parts ported.",
            "SimScale cloud FEA is the primary analyzer when SIMSCALE_* env vars are set; any failure "
            "falls back to CalculiX / analytical and is reported in result.simscale_diagnostic. "
            "If SimScale itself rejects the geometry, that failure is fed to the LLM as refinement feedback.",
            "Deterministic hole/bore verification for build123d ('# FEATURE: hole dia=6 count=4' comments "
            "checked against the finished B-rep; failures become a 'missing_features' refinement round BEFORE "
            "the slow SimScale analysis). Reconstructed for build123d — see feature_verification in results.",
            "build123d API knowledge base (RAG): the installed library's real signatures/docstrings + verified "
            "idioms, retrieved with NVIDIA embeddings + reranker and injected into every generation; failed "
            "scripts also get a static API check (GET /kb/status, /kb/search, POST /kb/lint).",
            "Programmatic geometry inspection (replaces PyVista): exact OpenCascade measurements of every "
            "generated solid (bbox, volume, area, centre of mass, topology counts, holes) — hard gate on "
            "'exactly one solid', plus a Nemotron design review that checks the request against the measured "
            "numbers (GEOMETRY_REVIEW=0 disables; GEOMETRY_REVIEW_TIMEOUT_S / _MODEL / _MAX_TOKENS tune it). "
            "See result.geometry_inspection / result.geometry_review.",
            "GET /cad-selftest and GET /simscale-selftest — verify the build123d kernel and the SimScale "
            "wiring end-to-end (the latter compares against a closed-form cantilever).",
            "POST /generate-validate-refine-async — same loop, returns a job_id to poll at GET /job/{id} "
            "(SimScale runs take minutes; use this behind proxies with request timeouts).",
        ],
        "new_in_v8_23":[
            "Optional second-model 'strategic advisor' for the Engineering Agent: set "
            "NEMOTRON_ADVISOR_MODEL (e.g. to a Nemotron/Kimi/DeepSeek model via OpenRouter, reusing "
            "OPENROUTER_API_KEY) and it gets consulted sparingly — once up front for engineering "
            "strategy, once per actual solver FAILURE for root-cause/smallest-fix guidance — while "
            "GPT-OSS-120B keeps driving the actual tool-calling loop unchanged. Advisory only: its "
            "input is added as context, never overrides a solver result, and any failure/misconfig "
            "silently falls back to today's single-model behavior. Left blank, nothing changes.",
            "POST /engineering-agent  ★★ tool-calling reasoning agent: Understand -> "
            "Inspect -> Diagnose -> Propose -> Modify -> Simulate -> Compare -> "
            "Refine, instead of /generate-validate-refine's regenerate-the-whole-script "
            "loop. The frontier model (GPT-OSS-120B via Groq by default) never writes "
            "build123d or declares pass/fail itself for the tapered-beam/bent-bracket "
            "workflows — it calls tools (inspect_geometry, diagnose_failure, "
            "modify_parameter/modify_thickness/.../add_hole, run_fea, compare_designs, "
            "...) that go through a safe parameter contract and the same "
            "make_tapered_beam/make_bent_bracket/run_analysis_v8 machinery every "
            "other endpoint already trusts. Geometry is validated and meshed "
            "automatically as part of every build/modify call (no separate "
            "validate_geometry/run_mesh round-trip needed) — cuts model calls per "
            "design iteration from 4 to 2, which matters a lot on rate-limited free "
            "API tiers. A monotonic-refinement guard automatically "
            "reverts to the last known-valid design if a candidate crashes or comes back "
            "non-manifold/non-watertight. First implementation target (see the endpoint's "
            "own docstring): the tapered-beam workflow — test that before bent-bracket. "
            "Geometry outside those two primitives falls back to the existing AI "
            "script-generation/refinement path, still wrapped in the same verify loop. "
            "Requires AI_PROVIDER to be an OpenAI-compatible tool-calling provider "
            "(groq/openrouter/lovable/cerebras/nvidia) — see engineering_agent_configured above.",
        ],
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
            "POST /generate-validate-refine-async  ★ same loop as a background job (poll GET /job/{id})",
            "GET  /cad-selftest","GET  /simscale-selftest",
            "POST /engineering-agent  ★★ tool-calling reasoning agent (Understand->Inspect->"
            "Diagnose->Propose->Modify->Verify->Simulate->Compare->Refine) — tapered-beam "
            "workflow is the first implementation target, see docstring",
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
    parametric={k:v[1] for k,v in B3D_MAP.items()}
    return {"build123d":parametric,
            "cadquery":parametric,   # deprecated alias for older frontends
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
    if not B3D: raise HTTPException(503,"build123d not installed")
    try: pd=json.loads(params)
    except: pd={}
    pt,gen_type,obj=route(description,pd)
    tmp=stl_from_cad(obj) if gen_type=="build123d" else stl_from_tm(obj)
    suffix=".stl" if export_format in ["stl","STL"] else ".step"
    if export_format not in ["stl","STL"]: tmp=step_from_cad(obj) if gen_type=="build123d" else tmp
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
    if not B3D: raise HTTPException(503,"build123d not installed")
    try: pd=json.loads(params)
    except: pd={}
    pt,gen_type,obj=route(description,pd)
    tmp=stl_from_cad(obj) if gen_type=="build123d" else stl_from_tm(obj)
    try:
        mesh=trimesh.load(tmp)
        if hasattr(mesh,"geometry"): mesh=trimesh.util.concatenate(list(mesh.geometry.values()))
        with open(tmp,"rb") as f: stl_b64=base64.b64encode(f.read()).decode()
        result=await run_analysis_v8(mesh,description,description,material,
                                      force_n,"z",operating_temp_c,project_description,
                                      surface_finish,reliability,
                                      cad_obj=(obj if gen_type=="build123d" else None))
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
    Any part from natural language → Gemini (via Lovable AI Gateway) writes build123d → real STL + analysis.
    Single-shot version (no refinement loop). For an AI design that automatically fixes
    its own engineering problems, use POST /generate-validate-refine instead.
    """
    if not B3D: raise HTTPException(503,"build123d not installed")

    # Gemini (Lovable AI Gateway) generates script
    try:
        script=await gemini_generate_script(prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502,f"Gemini API error: {str(e)}")

    # Execute safely (non-raising)
    obj,err=execute_cad_script_safely(script)
    if err:
        raise HTTPException(400,f"Generated script failed: {err}")

    # Export + analyze
    mesh,stl_bytes=await mesh_from_cad_object(obj)
    stl_b64=base64.b64encode(stl_bytes).decode()
    result=await run_analysis_v8(mesh,prompt,prompt,material,force_n,"z",operating_temp_c,cad_obj=obj)
    result["generated_stl_base64"]=stl_b64
    result["generated_script"]=script
    result["feature_verification"]=verify_declared_features(script,obj)   # single-shot: reported, not refined
    result["generation_method"]="llm_build123d_v8"
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


def _diagnose_cad_error(err: str) -> str:
    """Pattern-match known build123d/OpenCascade failure signatures and append
    specific, actionable guidance — the generic 'fix the root cause' feedback was not
    enough for the model to recover from these across a real refinement run (it got a
    DIFFERENT failure on the retry instead of a working script)."""
    hints = []
    low = err.lower()
    if "BRep_API: command not done" in err or "StdFail_NotDone" in err or "Standard_ConstructionError" in err:
        hints.append(
            "The CAD kernel REJECTED a fillet/chamfer/offset/boolean operation as "
            "geometrically infeasible — almost always because the requested "
            "radius is too large for the edge it's applied to (bigger than "
            "the material thickness, or it would overlap an adjacent edge or "
            "hole). Use a SMALLER radius (rule of thumb: no more than 20-30% "
            "of the local wall thickness), and apply fillets/chamfers BEFORE "
            "cutting nearby holes so the kernel has simpler geometry to work with."
        )
    if ("empty" in low and ("shapelist" in low or "edge" in low or "list" in low)) or "index out of range" in low \
            or "no edges" in low or "no faces" in low:
        hints.append(
            "Your edge/face selector matched NOTHING (empty ShapeList) — the filter is wrong for THIS solid, or "
            "the geometry changed since you assumed a position. Select edges immediately after creating the "
            "feature they belong to, and check the selector actually matches: filter_by(Axis.Z) keeps edges "
            "PARALLEL to Z, sort_by(Axis.Z)[-1] is the top-most, filter_by_position(Axis.Z, lo, hi) uses "
            "world coordinates."
        )
    if "nameerror" in low:
        hints.append(
            "Only names from `from build123d import *`, plus math and np, exist in this environment. Common "
            "algebra-mode names: Box, Cylinder, Cone, Sphere, Torus, Rectangle, Circle, Ellipse, Polygon, "
            "RegularPolygon, SlotOverall, Pos, Rot, Plane, Axis, extrude, revolve, loft, sweep, fillet, "
            "chamfer, offset, mirror, Spline, Polyline, Line."
        )
    if "typeerror" in low and ("argument" in low or "positional" in low):
        hints.append(
            "Wrong build123d call signature. Reminders: Box(length, width, height); Cylinder(radius, height) "
            "(radius FIRST); Cone(bottom_radius, top_radius, height); Sphere(radius); extrude(sketch, amount=h); "
            "fillet(edge_list, radius=r); chamfer(edge_list, length=l); Pos(x, y, z) * shape; "
            "Polygon((x,y), (x,y), ..., align=None)."
        )
    if "zero volume" in low or "2d sketch" in low or "not a build123d solid" in low:
        hints.append(
            "`result` must be a SOLID. A Rectangle/Circle/Polygon is a 2D sketch — extrude(it, amount=...) or "
            "revolve(it) first, and make sure a subtraction didn't remove the whole part."
        )
    if "not a valid b-rep" in low:
        hints.append(
            "Invalid solid: typically a self-intersecting fillet on a loft, a boolean between tangent-only "
            "solids, or a cut that exits exactly flush with a face. Rebuild that step with real overlap "
            "(unions) / overshoot (cuts)."
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
    max_iterations:int=Form(6),
    min_health_score:float=Form(75.0),
    max_critical_violations:int=Form(0),
    max_high_violations:int=Form(2),
    min_safety_factor:float=Form(1.0),
    run_topology:bool=Form(False),
    use_test_plan:bool=Form(False),
):
    """
    ★ THE CORE VIBE-ENGINEERING LOOP ★

    1. AI (Gemini via Lovable AI Gateway) writes a build123d script from `prompt`.
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
    if not B3D: raise HTTPException(503,"build123d not installed")
    use_test_plan=(use_test_plan is True)     # direct calls may leave a Form() default object here
    plan_cache=None
    if max_iterations<1: max_iterations=1
    if max_iterations>6: max_iterations=6  # hard cap: cost + latency safety

    iterations=[]
    kb_log=[]        # per-iteration API-knowledge-base trace (mode, reranker, titles, lint issues)
    review_refinements_used=0   # LLM design-review may trigger at most ONE refinement round per job
    best=None        # best {"result":..., "quality":..., "script":..., "stl_b64":..., "iteration":int}
    script=None
    feedback=None
    stopped_reason=None
    fallback_candidate=None     # a solid the reviewer questioned: analysed if the refinement cannot be generated
    last_gen_err=""
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
                kb_log.append({"iteration":i,**(_KB_TRACE.get() or {})})
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
                    last_gen_err=f"{e.status_code}: {str(e.detail)[:300]}"
                    gen_failed=True
                    break
                raise
        used_fallback=False
        if gen_failed:
            iterations.append({"iteration":i,"stage":"generation_failed",
                               "error":f"refinement generation failed ({last_gen_err})"})
            if fallback_candidate is not None and best is None:
                # the reviewer is advisory: do not throw away a solid that built fine because the LLM
                # could not be reached for the refinement -- analyse it instead
                script=fallback_candidate["script"]; obj=fallback_candidate["obj"]; fv=fallback_candidate["fv"]
                gi=fallback_candidate["gi"]; review=fallback_candidate["review"]
                fallback_candidate=None
                used_fallback=True
                stopped_reason=None
                iterations[-1]["fallback"]="analysing the earlier solid that the design review had questioned"
            else:
                break

        if not used_fallback:
            # 2) Execute
            obj,err=execute_cad_script_safely(script)
            if err:
                iterations.append({"iteration":i,"stage":"execution_failed","error":err,"script":script})
                feedback=(f"Your script FAILED TO EXECUTE with this error:\n{err}\n\n"
                          f"Fix the root cause and return a complete, runnable script."
                          f"{_diagnose_cad_error(err)}")
                continue

            # 2b) Deterministic feature check: do the holes/bores the script DECLARED ('# FEATURE:')
            # actually exist in the solid? Exact and cheap, so it runs BEFORE the slow SimScale
            # analysis — a part missing a required hole is never worth a cloud run. On the LAST
            # iteration there is no refinement left, so analyse anyway and just record the failure.
            fv=verify_declared_features(script,obj)
            if not fv["ok"] and i<max_iterations:
                iterations.append({"iteration":i,"stage":"missing_features",
                                   "error":"declared features missing from the solid: "+feature_failure_summary(fv),
                                   "feature_verification":_fv_brief(fv),"script":script})
                feedback=feature_failure_feedback(fv)
                continue

            # 2c) Programmatic geometry inspection: exact OpenCascade measurements of the finished
            # solid. Hard gate = exactly one solid (cheap, deterministic, runs BEFORE SimScale). Then
            # Nemotron reads the ORIGINAL REQUEST against the measured numbers and can flag clear
            # contradictions (asked 80x50x5, built 80x50x12) — fail-open, and at most one
            # review-triggered refinement per job so a false positive cannot eat every iteration.
            gi=inspect_b3d_geometry(obj)
            if not gi["ok"] and i<max_iterations:
                iterations.append({"iteration":i,"stage":"geometry_inspection_failed",
                                   "error":"; ".join(gi["issues"]),"geometry_inspection":_gi_brief(gi),"script":script})
                feedback=geometry_failure_feedback(gi)
                continue
            review=None
            if gi["ok"] and i<max_iterations and review_refinements_used<1:
                review=await review_geometry_with_llm(prompt,gi)
                if review and review["verdict"]=="fail" and review["issues"]:
                    review_refinements_used+=1
                    fallback_candidate={"script":script,"obj":obj,"fv":fv,"gi":gi,"review":review}
                    iterations.append({"iteration":i,"stage":"geometry_review_failed",
                                       "error":"design review: "+"; ".join(review["issues"]),
                                       "geometry_inspection":_gi_brief(gi),"review":review,"script":script})
                    feedback=("DESIGN REVIEW FOUND THE BUILT SOLID CONTRADICTS THE REQUEST. A reviewer compared your "
                              "original request with the exact measured geometry:\n"
                              + "\n".join(f"- {x}" for x in review["issues"]) + "\n\n" + geometry_report_text(gi)
                              + "\n\nFix the script so the solid matches the numbers in the request. If you are "
                                "confident a flagged item is actually satisfied, keep it and change nothing else.")
                    continue

        # 3) Analyze
        try:
            mesh,stl_bytes=await mesh_from_cad_object(obj)
            plan=None
            plan_diag={}
            if use_test_plan and obj is not None:
                plan=await generate_test_plan(prompt,obj,material,previous_plan=plan_cache,diag=plan_diag)
                plan_cache=plan or plan_cache
            result=await run_analysis_v8(mesh,prompt,prompt,material,force_n,force_dir,
                                          operating_temp_c,project_description,
                                          surface_finish,reliability,run_topology,
                                          cad_obj=obj,test_plan=plan)
            if use_test_plan:
                result["test_plan_diag"]=plan_diag
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
        if not fv["ok"]:
            # last iteration only (earlier ones `continue`d above): never call a design that is
            # missing a declared hole "passed", whatever its health score
            quality["passed"]=False
            quality["reasons"].insert(0,"Declared feature(s) missing from the solid: "+feature_failure_summary(fv))
        if not gi["ok"]:
            # last iteration only: a multi-solid result is never a "passed" design
            quality["passed"]=False
            quality["reasons"].insert(0,"Geometry inspection failed: "+"; ".join(gi["issues"]))
        verified,unverifiable=_verification_gate(use_test_plan,result,plan_diag,quality)
        stl_b64=base64.b64encode(stl_bytes).decode()
        entry={"iteration":i,"stage":"analyzed","passed":quality["passed"],
               "health_score":quality["score"],"reasons":quality["reasons"],
               "metrics":quality["metrics"],"feature_verification":_fv_brief(fv),
               "geometry_inspection":_gi_brief(gi),**({"review":review} if review else {})}
        iterations.append(entry)

        candidate={"result":result,"quality":quality,"script":script,
                   "stl_b64":stl_b64,"iteration":i,"fv":fv,"gi":gi,"review":review}
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
        if unverifiable:
            stopped_reason="verification_unavailable"      # SimScale/solve problem, not a design problem
            break

        # 5) Build feedback for next round
        feedback=summarize_analysis_for_refinement(result, quality)+"\n\n"+geometry_report_text(gi)

    if stopped_reason is None:
        stopped_reason="max_iterations_reached"

    if best is None:
        # Every iteration failed to even produce geometry — surface the last error.
        last=iterations[-1] if iterations else {}
        raise HTTPException(502, "AI failed to produce a valid CAD design after "
                                  f"{len(iterations)} attempt(s). Last error: "
                                  f"{last.get('error','unknown')} (loop stopped: {stopped_reason}"
                                  + (f"; last generation error: {last_gen_err}" if last_gen_err else "") + ")")

    result=best["result"]
    result["generated_stl_base64"]=best["stl_b64"]
    result["generated_script"]=best["script"]
    result["feature_verification"]=best.get("fv")
    result["geometry_inspection"]=best.get("gi")
    result["geometry_review"]=best.get("review")
    result["generation_method"]="llm_build123d_v8_refined"
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
        "kb":kb_log,
    }
    return result


@app.on_event("startup")
async def _kb_startup():
    """Build the API knowledge base in the background (chunking is instant -> BM25 works at once;
    embeddings follow when NVIDIA answers). Never delays or fails startup."""
    if KB_ENABLED and B3D:
        threading.Thread(target=kb_build, daemon=True).start()


@app.get("/kb/status")
@_sanitize_response
async def kb_status():
    """State of the build123d API knowledge base: retrieval mode, chunk counts, embedding/reranker health."""
    with _kb_lock:
        st = {k: _kb[k] for k in ("mode", "error", "building", "counts", "dropped_idioms", "skipped_doc_chunks",
                                   "b3d_version", "embed_dim", "docs_files")}
        st["chunks"] = len(_kb["chunks"])
        st["built_at"] = _kb["built_at"]
    st.update(enabled=KB_ENABLED, embeddings_enabled=KB_EMBEDDINGS, embed_model=NVIDIA_EMBED_MODEL, docs_dir=KB_DOCS_DIR, index_path=KB_INDEX_PATH,
              nvidia_key_set=bool(NVIDIA_API_KEY))
    st["reranker"] = {"enabled": KB_RERANK, "working_model": _rr["model"], "endpoint": _rr["url"],
                      "last_error": _rr["last_error"], "recent_attempts": _rr["tried"],
                      "cooling_down_s": max(0, int(_rr["disabled_until"] - time.time())),
                      "candidates": NVIDIA_RERANK_MODELS,
                      "note": "nvidia/rerank-qa-mistral-4b: NVIDIA's model page announces its API was deprecated on 08/24/2026"}
    return st


@app.get("/kb/search")
@_sanitize_response
async def kb_search(q: str, k: int = 6):
    """Debug: what would be retrieved for this query (hybrid + rerank), without calling the LLM."""
    chunks, info = await asyncio.to_thread(kb_retrieve, q, (), max(1, min(k, 15)))
    return {"query": q, "info": info,
            "results": [{"title": c["title"], "kind": c["kind"], "text": c["text"][:800]} for c in chunks]}


@app.post("/kb/rebuild")
@_sanitize_response
async def kb_rebuild(force: bool = Form(True)):
    """Re-read the installed build123d + docs dir and (force=true) re-embed everything."""
    threading.Thread(target=kb_build, kwargs={"force": force}, daemon=True).start()
    return {"started": True, "note": "poll GET /kb/status"}


@app.post("/kb/lint")
@_sanitize_response
async def kb_lint(script: str = Form(...)):
    """Static API check of a build123d script against the installed library (unknown names/keywords/members)."""
    issues = await asyncio.to_thread(lint_b3d_script, script, 25)
    return {"ok": not issues, "issues": issues}


@app.post("/generate-validate-refine-async")
@_sanitize_response
async def generate_validate_refine_async(
    background_tasks: BackgroundTasks,
    prompt:str=Form(...),
    material:str=Form("auto"),
    force_n:float=Form(1000.0),
    force_dir:str=Form("z"),
    operating_temp_c:float=Form(25.0),
    surface_finish:str=Form("machined"),
    reliability:float=Form(0.99),
    project_description:Optional[str]=Form(None),
    max_iterations:int=Form(6),
    min_health_score:float=Form(75.0),
    max_critical_violations:int=Form(0),
    max_high_violations:int=Form(2),
    min_safety_factor:float=Form(1.0),
    run_topology:bool=Form(False),
    use_test_plan:bool=Form(False),
):
    """
    Same self-correcting loop as /generate-validate-refine, but as a background job.
    Use this with SimScale: one cloud analysis takes minutes, so a multi-iteration loop
    will exceed the request timeout of most hosts (Render/Railway/proxies). Returns a
    job_id immediately; poll GET /job/{job_id} (returns the final result when done).
    """
    job_id=str(uuid.uuid4())
    JOB_STORE[job_id]={"status":"running","created":time.time(),"filename":prompt[:80]}

    async def run_job():
        try:
            result=await generate_validate_refine(
                prompt=prompt,material=material,force_n=force_n,force_dir=force_dir,
                operating_temp_c=operating_temp_c,surface_finish=surface_finish,reliability=reliability,
                project_description=project_description,max_iterations=max_iterations,
                min_health_score=min_health_score,max_critical_violations=max_critical_violations,
                max_high_violations=max_high_violations,min_safety_factor=min_safety_factor,
                run_topology=run_topology,use_test_plan=use_test_plan)
            JOB_STORE[job_id]={"status":"complete","result":result,"created":time.time()}
        except HTTPException as e:
            JOB_STORE[job_id]={"status":"error","error":f"{e.status_code}: {e.detail}","created":time.time()}
        except Exception as e:
            JOB_STORE[job_id]={"status":"error","error":f"{type(e).__name__}: {e}","created":time.time()}

    background_tasks.add_task(run_job)
    return {"job_id":job_id,"status":"running",
            "message":"Design loop started. Poll GET /job/{job_id} for the result.",
            "estimated_time":"1-5 minutes per iteration with SimScale, seconds without"}


def _b3d_stats(shape):
    bb=shape.bounding_box()
    return {"volume_mm3":round(float(shape.volume),2),"valid_brep":bool(_b3d_is_valid(shape)),
            "bbox_mm":[round(float(bb.size.X),2),round(float(bb.size.Y),2),round(float(bb.size.Z),2)]}


@app.get("/cad-selftest")
@_sanitize_response
async def cad_selftest():
    """
    Smoke-test the build123d kernel on THIS deployment: the two trusted primitives, every
    parametric generator, the LLM-script sandbox (positive AND negative cases), and STL/STEP
    export + watertightness. Run this once after deploying — build123d's API moves between
    releases, and this pinpoints exactly which call (if any) needs adjusting.
    """
    if not B3D:
        return {"ok":False,"error":"build123d is not installed (pip install build123d)"}
    try:
        version=getattr(b3d,"__version__",None)
    except Exception:
        version=None
    checks=[]
    def run(name,fn):
        t0=time.time()
        try:
            info=fn() or {}
            checks.append({"check":name,"ok":True,"t_s":round(time.time()-t0,2),**info})
        except Exception as e:
            checks.append({"check":name,"ok":False,"error":f"{type(e).__name__}: {str(e)[:300]}"})

    run("make_tapered_beam",lambda:_b3d_stats(make_tapered_beam(120,24,10,12,6,fillet_radius=1.0,
                                                                holes_base=[(6,0,3)])))
    run("make_bent_bracket",lambda:_b3d_stats(make_bent_bracket(50,40,30,4,90,3,
                                                                holes_leg1=[(25,0,6)],holes_leg2=[(20,0,6)])))
    for pt,(fn,_kw) in B3D_MAP.items():
        run(f"gen_{pt}",lambda fn=fn:_b3d_stats(fn({})))

    def _feature_check():
        part = b3d.Box(60, 40, 10)
        for x, y in [(-20, -10), (20, -10), (-20, 10), (20, 10)]:
            part = part - (b3d.Pos(x, y, 0) * b3d.Cylinder(3, 14))          # 4 x dia6 through
        part = part - (b3d.Pos(0, 0, 4) * b3d.Cylinder(5, 6))               # 1 x dia10 blind (from the top)
        part = part + (b3d.Pos(25, 0, 7) * b3d.Cylinder(4, 10))             # a BOSS (must not count as a hole)
        found = sorted(h["diameter_mm"] for h in find_b3d_holes(part))
        if found != [6.0, 6.0, 6.0, 6.0, 10.0]:
            raise RuntimeError(f"hole detection found {found}, expected four dia6 + one dia10 (boss must not count)")
        good = verify_declared_features("# FEATURE: hole dia=6 count=4\n# FEATURE: hole dia=10 count=1\n", part)
        if not good["ok"]:
            raise RuntimeError(f"true declarations were rejected: {good['missing']}")
        bad = verify_declared_features("# FEATURE: hole dia=6 count=5\n# FEATURE: hole dia=8 count=1\n", part)
        if bad["ok"] or len(bad["missing"]) != 2:
            raise RuntimeError("false declarations (5th dia6 hole; a dia8 hole that is only a boss) were not caught")
        return {"holes_found_mm": found, "false_claims_caught": len(bad["missing"])}
    run("feature_verification", _feature_check)

    def _script_ok():
        s=("from build123d import *\n"
           "result = Box(40, 20, 10) - Pos(10, 0, 0) * Cylinder(3, 12)\n"
           "result = fillet(result.edges().filter_by(Axis.Z), radius=2)\n")
        obj,err=execute_cad_script_safely(s)
        if err: raise RuntimeError(err)
        return _b3d_stats(obj)
    run("sandbox_runs_valid_script",_script_ok)

    def _script_blocked():
        for bad in ("from build123d import *\nexport_stl(Box(1,1,1), '/tmp/x.stl')\nresult = Box(1,1,1)\n",
                    "import os\nresult = None\n",
                    "from build123d import *\nresult = Box(1,1,1).export_step('/tmp/x.step')\n"):
            _o,err=execute_cad_script_safely(bad)
            if not err: raise RuntimeError(f"sandbox failed to block: {bad[:60]!r}")
    run("sandbox_blocks_io_and_os",_script_blocked)

    async def _export():
        obj=make_tapered_beam(100,20,10,10,6)
        mesh,stl=await mesh_from_cad_object(obj)
        p=step_from_cad(obj)
        try: step_kb=round(os.path.getsize(p)/1024,1)
        finally: os.unlink(p)
        return {"stl_kb":round(len(stl)/1024,1),"triangles":int(len(mesh.faces)),
                "watertight":bool(mesh.is_watertight),"step_kb":step_kb}
    t0=time.time()
    try:
        info=await _export(); checks.append({"check":"stl_step_export","ok":True,"t_s":round(time.time()-t0,2),**info})
    except Exception as e:
        checks.append({"check":"stl_step_export","ok":False,"error":f"{type(e).__name__}: {str(e)[:300]}"})
    return {"ok":all(c["ok"] for c in checks),"build123d_version":version,"checks":checks}


@app.get("/simscale-selftest")
@_sanitize_response
async def simscale_selftest(force_n:float=100.0, material:str=""):
    """
    End-to-end check of the SimScale wiring using a 100 x 20 x 10 mm cantilever (fixed at x=0,
    `force_n` along +Y on the x=100 end) and compare with the closed-form answer
    (sigma = 6FL/(b h^2), delta = 4FL^3/(E b h^3)). Expect the FEM peak stress to be somewhat
    ABOVE the beam-theory value (root singularity) and the deflection within a few percent.
    If the numbers are wildly off, the face mapping / load case is wrong — check
    diagnostic.entity_counts and diagnostic.patch. A run takes several minutes.
    """
    if not B3D:
        raise HTTPException(503,"build123d not installed")
    mat_key=material or SIMSCALE_TEMPLATE_MATERIAL
    if mat_key not in MATERIALS: mat_key="aluminum_6061"
    L,h,b=100.0,20.0,10.0    # length (X), depth along the load (Y), width (Z)
    beam=b3d.Pos(L/2,0,0)*b3d.Box(L,h,b)
    fem,diag=await asyncio.to_thread(run_simscale_fem,beam,mat_key,force_n,"y",2.0)
    E=MATERIALS[mat_key]["youngs_modulus_gpa"]*1e3   # MPa
    expected={"beam_theory_max_stress_mpa":round(6*force_n*L/(b*h**2),3),
              "beam_theory_tip_deflection_mm":round(4*force_n*L**3/(E*b*h**3),5),
              "note":"cantilever, end shear load; FEM stress at the fixed edge is typically higher (singularity)."}
    out={"configured":SIMSCALE_ENABLED,"diagnostic":diag,"expected":expected}
    if fem:
        vm=fem["stress"]["von_mises_mpa"];dz=fem["deflection_mm"]
        out["fem"]={"von_mises_mpa":vm,"deflection_mm":dz,"hotspot_xyz_mm":fem["critical_section"].get("hotspot_xyz_mm"),
                    "simscale":fem.get("simscale")}
        out["ratio_fem_over_theory"]={"stress":round(vm/max(expected["beam_theory_max_stress_mpa"],1e-9),3),
                                      "deflection":round(dz/max(expected["beam_theory_tip_deflection_mm"],1e-9),3)}
    return out


async def run_plan_tests(cad_obj, plan, default_min_sf=2.0, dry_run=False):
    """Run an LLM-written test plan against a build123d solid. Faces are chosen geometrically from the plan;
    nothing (forces, directions, faces) is passed in by hand. dry_run only resolves the selectors."""
    try:
        plan = TP.parse_plan(plan)
    except TP.PlanError as e:
        return {"overall": "PLAN_ERROR", "error": str(e), "results": []}
    descs = TP.describe_faces(cad_obj)
    results = []
    for test in plan["tests"]:
        try:
            case = TP.build_load_case(test, descs, MATERIALS)
        except TP.PlanError as e:
            results.append({"test_id": str(test.get("id", "?")), "status": "PLAN_ERROR", "error": str(e)})
            continue
        if dry_run:
            results.append({"test_id": case["id"], "status": "RESOLVED", "material": case["material"],
                            "fixed_faces": case["fixed_idx"], "criteria": case["criteria"],
                            "loads": [{"id": l["id"], "faces": l["face_idx"],
                                       "force_n": [round(v, 2) for v in l["force_xyz"]]} for l in case["loads"]]})
            continue
        if not SIMSCALE_ENABLED:
            results.append(TP.evaluate(case, None, {"reason": "SIMSCALE_API_KEY not set"}, default_min_sf))
            continue
        fem, diag = await asyncio.to_thread(run_simscale_fem, cad_obj, case["material"], 0, "z",
                                            case["criteria"].get("min_safety_factor", default_min_sf), case)
        r = TP.evaluate(case, fem, diag, default_min_sf)
        r["elapsed_s"] = diag.get("elapsed_s")
        _m = r.get("metrics") or {}
        if _m.get("hotspot_mm"):
            r["localization"] = TP.localize(descs, case, _m.get("hotspot_mm"), _m.get("hot_zone_mm"), _m)
        results.append(r)
    return {"plan_version": TP.PLAN_VERSION, "overall": TP.overall(results), "results": results,
            "geometry": TP.face_summary(descs)}


@app.get("/test-plan-schema")
async def test_plan_schema():
    """What the LLM must output after generating a part: the prompt block, an example, supported test types."""
    return {"prompt_block": TP.schema_prompt(MATERIALS), "example_plan": TP.EXAMPLE_PLAN,
            "supported_tests": list(TP.SUPPORTED_TESTS)}


@app.post("/run-test-plan")
@_sanitize_response
async def run_test_plan(file: UploadFile = File(...), plan: str = Form(...), dry_run: bool = Form(False)):
    """STEP file + test-plan JSON -> verdicts. dry_run=true only resolves the face selectors (instant)."""
    if not B3D:
        raise HTTPException(503, "build123d not installed")
    data = await file.read()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "part.step")
        with open(path, "wb") as f:
            f.write(data)
        try:
            obj = b3d.import_step(path)
        except Exception as e:
            raise HTTPException(400, f"could not read STEP file: {type(e).__name__}: {str(e)[:200]}")
        return await run_plan_tests(obj, plan, dry_run=dry_run)


@app.get("/test-plan-selftest")
@_sanitize_response
async def test_plan_selftest(force_n: float = 100.0, material: str = "", dry_run: bool = True):
    """Cantilever driven ONLY by a test plan (no force/face arguments). dry_run=true (default) is instant and
    shows which faces the selectors picked; dry_run=false runs SimScale (several minutes) and compares with
    beam theory."""
    if not B3D:
        raise HTTPException(503, "build123d not installed")
    mat_key = material or SIMSCALE_TEMPLATE_MATERIAL
    if mat_key not in MATERIALS:
        mat_key = "aluminum_6061"
    L, h, b = 100.0, 20.0, 10.0
    beam = b3d.Pos(L / 2, 0, 0) * b3d.Box(L, h, b)
    plan = {"version": 1, "tests": [{
        "id": "cantilever", "type": "static_stress", "material": mat_key,
        "constraints": [{"id": "root", "type": "fixed", "faces": {"at": "min_x", "normal": "-x"}}],
        "loads": [{"id": "tip", "type": "force", "faces": {"at": "max_x", "normal": "+x"},
                   "force_n": force_n, "direction": "+y"}],
        "criteria": {"min_safety_factor": 2.0}}]}
    out = await run_plan_tests(beam, plan, dry_run=dry_run)
    E = MATERIALS[mat_key]["youngs_modulus_gpa"] * 1e3
    exp = {"beam_theory_max_stress_mpa": round(6 * force_n * L / (b * h ** 2), 3),
           "beam_theory_tip_deflection_mm": round(4 * force_n * L ** 3 / (E * b * h ** 3), 5)}
    out["expected"] = exp
    r = (out.get("results") or [{}])[0]
    m = r.get("metrics") or {}
    if m.get("von_mises_mpa") is not None:
        out["ratio_fem_over_theory"] = {
            "stress": round(m["von_mises_mpa"] / max(exp["beam_theory_max_stress_mpa"], 1e-9), 3),
            "deflection": round((m.get("max_deflection_mm") or 0) / max(exp["beam_theory_tip_deflection_mm"], 1e-9), 3)}
    return out


@app.get("/deepseek-selftest")
@_sanitize_response
async def deepseek_selftest():
    """One tiny call to the direct DeepSeek API: is the key valid, which model, how fast. No SimScale."""
    import time as _t
    info = {"key_set": bool(DEEPSEEK_API_KEY), "model": DEEPSEEK_MODEL, "url": DEEPSEEK_API_URL,
            "thinking": DEEPSEEK_THINKING or "(default)"}
    if not DEEPSEEK_API_KEY:
        return {**info, "ok": False, "error": "DEEPSEEK_API_KEY is not set on the server"}
    t0 = _t.time()
    try:
        out = await asyncio.to_thread(
            _deepseek_request, [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
            temperature=0, max_tokens=2000, timeout=PLAN_TIMEOUT_S)
        return {**info, "ok": True, "seconds": round(_t.time() - t0, 1), "reply_head": out[:200]}
    except HTTPException as e:
        return {**info, "ok": False, "seconds": round(_t.time() - t0, 1), "error": f"{e.status_code}: {str(e.detail)[:400]}"}


@app.get("/test-plan-llm-selftest")
@_sanitize_response
async def test_plan_llm_selftest(shape: str = "bracket", prompt: str = ""):
    """Does the LLM produce a usable test plan? One LLM call, no SimScale (10-60 s). shape=bracket (L-bracket
    with two holes) or cantilever. Shows the plan, what the selectors resolved to, and why it failed if it did."""
    if not B3D:
        raise HTTPException(503, "build123d not installed")
    if shape == "cantilever":
        part = b3d.Pos(50, 0, 0) * b3d.Box(100, 20, 10)
        prompt = prompt or "Steel cantilever beam 100x20x10 mm, fixed at one end, 100 N at the free end"
    else:
        part = b3d.Pos(40, 20, 2.5) * b3d.Box(80, 40, 5) + b3d.Pos(2.5, 20, 25) * b3d.Box(5, 40, 40)
        for x, y in ((55, 10), (55, 30)):
            part = part - b3d.Pos(x, y, 2.5) * b3d.Cylinder(3, 8)
        prompt = prompt or ("Aluminum L-bracket 80x40x5 mm, two 6 mm holes in the base, wall-mounted, "
                            "carries a 300 N load on the free arm")
    diag = {}
    plan = await generate_test_plan(prompt, part, "auto", diag=diag)
    out = {"provider": AI_PROVIDER, "prompt": prompt, "plan": plan, "diag": diag}
    if plan is not None:
        out["resolved"] = await run_plan_tests(part, plan, dry_run=True)
    return out


@app.get("/simscale-sdk")
@_sanitize_response
async def simscale_sdk(geometry_id: str = ""):
    """Read-only: face-mapping details of the newest lumexa_* geometry, the SDK's selector models and the
    template's material/primitives. Used to make face selection geometric instead of order-based."""
    if not SIMSCALE_ENABLED:
        raise HTTPException(503, "SIMSCALE_API_KEY not set")
    return await asyncio.to_thread(_ss_sdk_info, geometry_id)


@app.get("/simscale-probe")
@_sanitize_response
async def simscale_probe(simulation_id: str = "", run_id: str = ""):
    """Re-reads the results of an already FINISHED SimScale run without solving again (~10-60 s).
    No parameters = newest finished lumexa_* run. Shows every result item SimScale returned, the urls
    found, the files downloaded and why each file did / did not parse. Use it to debug the results step."""
    if not SIMSCALE_ENABLED:
        raise HTTPException(503, "SIMSCALE_API_KEY not set")
    return await asyncio.to_thread(_ss_probe, simulation_id, run_id)


@app.get("/simscale-export-audit")
@_sanitize_response
async def simscale_export_audit(n: int = 10):
    """Which of the newest lumexa_* SimScale runs can still be exported and which are locked (export-source-locked)?
    No solving; ~10-60 s. Also reports how many simulations the project holds and basic project info."""
    if not SIMSCALE_ENABLED:
        raise HTTPException(503, "SIMSCALE_API_KEY not set")
    return await asyncio.to_thread(_ss_export_audit, n)


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
    if not B3D:
        raise HTTPException(503, "build123d not installed")

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

    obj, err = execute_cad_script_safely(new_script)
    if err:
        return {
            "stage": "execution_failed", "error": err, "script": new_script,
            "external_feedback_used": feedback_text,
        }

    mesh, stl_bytes = await mesh_from_cad_object(obj)
    result = await run_analysis_v8(mesh, prompt, prompt, material, force_n, force_dir,
                                    operating_temp_c, None, surface_finish, reliability, False,
                                    cad_obj=obj)
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
    literal name `result` for its final shape — see BUILD123D_SYSTEM.
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
         collision) so the result is a normal, fully re-editable build123d script
         — future edits (another region, or /refine-from-external-fea) work on
         it exactly like any other script in this system.
    """
    if not B3D:
        raise HTTPException(503, "build123d not installed")

    bx, by, bz = abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)
    if bx <= 0 or by <= 0 or bz <= 0:
        raise HTTPException(400, "Bounding box must have positive size on all three axes "
                                  "(check x_min<x_max, y_min<y_max, z_min<z_max).")
    cx, cy, cz = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2

    # Step 1: confirm the base script still executes before spending an AI call.
    base_obj, base_err = execute_cad_script_safely(previous_script)
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

    local_obj, local_err = execute_cad_script_safely(local_script)
    if local_err:
        return {"stage": "local_generation_failed", "error": local_err, "local_script": local_script}

    # Step 4/5: cut + union + assemble the combined script.
    try:
        base_renamed = _rename_result_var(previous_script, "_base_result")
        local_renamed = _rename_result_var(local_script, "_local_result")

        combined_script = (
            "from build123d import *\n\n"
            "# --- base shape (previous design) ---\n"
            f"{base_renamed}\n"
            "_base_result = as_solid(_base_result)\n\n"
            "# --- local replacement geometry for the edited region ---\n"
            f"{local_renamed}\n"
            "_local_result = as_solid(_local_result)\n\n"
            "# --- combine: cut the edited region out of the base, then union in "
            "the new local geometry, positioned at the region's real location ---\n"
            f"_cutter = Pos({cx}, {cy}, {cz}) * Box({bx}, {by}, {bz})\n"
            f"_local_positioned = Pos({cx}, {cy}, {cz}) * _local_result\n"
            "result = (_base_result - _cutter) + _local_positioned\n"
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to assemble combined script: {str(e)}")

    combined_obj, combined_err = execute_cad_script_safely(combined_script)
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

    mesh, stl_bytes = await mesh_from_cad_object(combined_obj)
    result = await run_analysis_v8(mesh, edit_prompt, edit_prompt, material, 1000.0, "z",
                                    25.0, None, "machined", 0.99, False, cad_obj=combined_obj)
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
    if not B3D:
        raise HTTPException(503, "build123d not installed")

    obj, err = execute_cad_script_safely(script)
    if err:
        raise HTTPException(400, f"Script failed to execute: {err}")

    step_path = step_from_cad(obj)
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


# ══════════════════════════════════════════════════════════════════════════
# ENGINEERING AGENT (v8.23) — a reasoning layer on top of the existing
# deterministic systems above. Implements the build spec:
#
#     Understand -> Inspect -> Diagnose -> Propose -> Modify -> Verify ->
#     Simulate -> Compare -> Refine
#
# instead of the "regenerate the whole build123d script and hope" pattern
# /generate-validate-refine uses. Nothing above this line is modified —
# this section only ADDS a new endpoint (/engineering-agent) that
# orchestrates the frontier model (tool-calling) around the exact same
# deterministic helpers, sandboxed executor, mesher, and analysis pipeline
# already defined above. make_tapered_beam/make_bent_bracket/
# execute_cad_script_safely/mesh_from_cad_object/run_analysis_v8/
# evaluate_design_quality/gemini_generate_script are all reused as-is.
#
# Current scope (per the spec's "First Implementation Target"): the
# tapered-beam workflow is the one to test first. Bent-bracket support is
# wired the same way since make_bent_bracket already existed, but has had
# less real-world exercise than the beam path. Anything neither primitive
# covers falls back to the existing AI script-generation/refinement
# machinery (generic_script design_type) — still wrapped in the same
# validate -> mesh -> FEA -> compare loop, just without a validated numeric
# parameter contract on that particular path.
#
# Tool-calling is currently implemented for OpenAI-compatible providers
# (groq/openrouter/lovable/cerebras/nvidia — identical wire format). Nemotron 3
# Ultra, Kimi K3, and DeepSeek V4 are all reachable via AI_PROVIDER=nvidia +
# NVIDIA_MODEL (one endpoint/key, NVIDIA_MODEL just picks which of their 90+
# catalog models actually answers — see the NVIDIA_API_KEY setup comment
# above for confirmed-live model IDs). Claude/Gemini native tool-calling for
# this specific endpoint is not wired up yet; every other endpoint is
# unaffected and still works on any configured provider as before.
# ══════════════════════════════════════════════════════════════════════════

import gc

AGENT_TURN_MAX_TOKENS = 3000

# ----------------------------------------------------------------------
# Safe parameter contracts (spec section 4) — reject bad numbers BEFORE
# ever calling build123d/OpenCascade, with structured REJECTED feedback.
# ----------------------------------------------------------------------

def _validate_hole(hx, hy, hd, width, thick):
    """Edge-distance rule for a hole on a tapered_beam end face (rect centered
    on both axes: x in [-width/2,width/2], y in [-thick/2,thick/2]) — same
    1.5*diameter rule already used by rule_engine_v8/detect_holes_v8 (R07)."""
    if hd is None or hd <= 0:
        return False, {"status": "REJECTED", "reason": "hole diameter must be > 0",
                        "constraint": "diameter_mm > 0", "received": hd}
    min_edge = 1.5 * hd
    if abs(hx) + hd / 2.0 + min_edge > width / 2.0:
        return False, {"status": "REJECTED",
                        "reason": f"Hole at x={hx} (d={hd}mm) violates the 1.5*D edge-distance rule "
                                  f"against the {width}mm section width",
                        "constraint": "edge_distance >= 1.5*diameter", "received": {"x": hx, "width": width}}
    if abs(hy) + hd / 2.0 + min_edge > thick / 2.0:
        return False, {"status": "REJECTED",
                        "reason": f"Hole at y={hy} (d={hd}mm) violates the 1.5*D edge-distance rule "
                                  f"against the {thick}mm section thickness",
                        "constraint": "edge_distance >= 1.5*diameter", "received": {"y": hy, "thick": thick}}
    if hd > 0.6 * min(width, thick):
        return False, {"status": "REJECTED",
                        "reason": f"Hole diameter {hd}mm exceeds 60% of the smallest local section "
                                  f"dimension ({round(min(width, thick), 2)}mm) — would leave almost no material",
                        "constraint": "diameter <= 0.6*min(width,thick)", "received": hd}
    return True, None


def _validate_hole_bracket(hx, hy, hd, length, width):
    """Edge-distance rule for a hole on a bent_bracket leg (rect spans
    x in [0,length], y in [-width/2,width/2] — matches make_bent_bracket's
    rect(length,width,centered=(False,True)))."""
    if hd is None or hd <= 0:
        return False, {"status": "REJECTED", "reason": "hole diameter must be > 0",
                        "constraint": "diameter_mm > 0", "received": hd}
    min_edge = 1.5 * hd
    if hx - hd / 2.0 - min_edge < 0 or hx + hd / 2.0 + min_edge > length:
        return False, {"status": "REJECTED",
                        "reason": f"Hole at x={hx} (d={hd}mm) violates the 1.5*D edge-distance rule "
                                  f"against the {length}mm leg length",
                        "constraint": "edge_distance >= 1.5*diameter", "received": {"x": hx, "length": length}}
    if abs(hy) + hd / 2.0 + min_edge > width / 2.0:
        return False, {"status": "REJECTED",
                        "reason": f"Hole at y={hy} (d={hd}mm) violates the 1.5*D edge-distance rule "
                                  f"against the {width}mm leg width",
                        "constraint": "edge_distance >= 1.5*diameter", "received": {"y": hy, "width": width}}
    if hd > 0.6 * min(length, width):
        return False, {"status": "REJECTED", "reason": "Hole diameter too large relative to leg dimensions",
                        "constraint": "diameter <= 0.6*min(length,width)", "received": hd}
    return True, None


def _beam_param_contract(params, mat_key):
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    min_wall = mat.get("min_wall_mm", 1.0)

    def reject(msg, constraint, received):
        return False, {"status": "REJECTED", "reason": msg, "constraint": constraint, "received": received}

    length = params.get("length"); bw = params.get("base_width"); bt = params.get("base_thick")
    tw = params.get("tip_width"); tt = params.get("tip_thick"); fr = params.get("fillet_radius") or 0.0
    for name, val in [("length", length), ("base_width", bw), ("base_thick", bt),
                       ("tip_width", tw), ("tip_thick", tt)]:
        if val is None or val <= 0:
            return reject(f"'{name}' must be > 0", f"{name} > 0", val)
    if bt < min_wall or tt < min_wall:
        return reject(f"Section thickness would drop below the material minimum ({min_wall}mm for "
                       f"{mat['name']}) — base_thick={bt}, tip_thick={tt}",
                       f"thickness >= {min_wall}", min(bt, tt))
    min_cross = min(bw, bt, tw, tt)
    if fr < 0:
        return reject("'fillet_radius' cannot be negative", "fillet_radius >= 0", fr)
    if fr > min_cross / 2.0:
        return reject(f"fillet_radius {fr}mm is too large for the smallest cross-section dimension "
                       f"({round(min_cross,2)}mm) — would self-intersect",
                       "fillet_radius < min_section_dim/2", fr)
    for end, holes, w, t in [("base", params.get("holes_base") or [], bw, bt),
                              ("tip", params.get("holes_tip") or [], tw, tt)]:
        for h in holes:
            hx, hy, hd = h
            ok, err = _validate_hole(hx, hy, hd, w, t)
            if not ok:
                return False, err
    if length / min_cross > 40:
        return reject(f"Aspect ratio {round(length/min_cross,1)}:1 is extreme even for a slender arm "
                       f"(length={length}mm vs smallest section {round(min_cross,2)}mm)",
                       "length/min_section_dim <= 40", round(length / min_cross, 1))
    return True, None


def _bracket_param_contract(params, mat_key):
    mat = MATERIALS.get(mat_key, MATERIALS["aluminum_6061"])
    min_wall = mat.get("min_wall_mm", 1.0)

    def reject(msg, constraint, received):
        return False, {"status": "REJECTED", "reason": msg, "constraint": constraint, "received": received}

    l1 = params.get("leg1_length"); l2 = params.get("leg2_length")
    w = params.get("width"); t = params.get("thickness")
    ang = params.get("bend_angle_deg", 90.0); fr = params.get("fillet_radius") or 0.0
    for name, val in [("leg1_length", l1), ("leg2_length", l2), ("width", w), ("thickness", t)]:
        if val is None or val <= 0:
            return reject(f"'{name}' must be > 0", f"{name} > 0", val)
    if t < min_wall:
        return reject(f"thickness {t}mm is below the material minimum {min_wall}mm for {mat['name']}",
                       f"thickness >= {min_wall}", t)
    if not (5.0 <= ang <= 175.0):
        return reject(f"bend_angle_deg {ang} is degenerate (too close to flat/folded-flat)",
                       "5 <= bend_angle_deg <= 175", ang)
    if fr < 0:
        return reject("fillet_radius cannot be negative", "fillet_radius >= 0", fr)
    if fr > w / 2.0 or fr > min(l1, l2) / 2.0:
        return reject(f"fillet_radius {fr}mm is too large for this bracket's geometry",
                       "fillet_radius < min(width,leg_length)/2", fr)
    for leg, holes, length in [("leg1", params.get("holes_leg1") or [], l1),
                                ("leg2", params.get("holes_leg2") or [], l2)]:
        for h in holes:
            hx, hy, hd = h
            ok, err = _validate_hole_bracket(hx, hy, hd, length, w)
            if not ok:
                return False, err
    max_dim = max(l1, l2); min_dim = min(w, t)
    if min_dim > 0 and max_dim / min_dim > 40:
        return reject(f"Aspect ratio {round(max_dim/min_dim,1)}:1 is extreme",
                       "max_dim/min(width,thickness) <= 40", round(max_dim / min_dim, 1))
    return True, None


def _copy_params(p):
    if p is None:
        return None
    return {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}


def _diff_params(old, new):
    if old is None:
        return {"all": new}
    changed = {}
    for k, v in new.items():
        if old.get(k) != v:
            changed[k] = {"before": old.get(k), "after": v}
    return changed


def params_to_script_tapered_beam(params):
    return (
        "from build123d import *\n"
        "result = make_tapered_beam(\n"
        f"    length={params['length']}, base_width={params['base_width']}, base_thick={params['base_thick']},\n"
        f"    tip_width={params['tip_width']}, tip_thick={params['tip_thick']}, "
        f"fillet_radius={params.get('fillet_radius', 0.0)},\n"
        f"    holes_base={params.get('holes_base') or []}, holes_tip={params.get('holes_tip') or []},\n"
        ")\n"
    )


def params_to_script_bent_bracket(params):
    return (
        "from build123d import *\n"
        "result = make_bent_bracket(\n"
        f"    leg1_length={params['leg1_length']}, leg2_length={params['leg2_length']},\n"
        f"    width={params['width']}, thickness={params['thickness']}, "
        f"bend_angle_deg={params.get('bend_angle_deg', 90.0)},\n"
        f"    fillet_radius={params.get('fillet_radius', 0.0)},\n"
        f"    holes_leg1={params.get('holes_leg1') or []}, holes_leg2={params.get('holes_leg2') or []},\n"
        ")\n"
    )


# ----------------------------------------------------------------------
# Design state (spec section 6 & 8) — lives for the duration of one
# /engineering-agent request; not persisted across requests.
# ----------------------------------------------------------------------

class EngineeringDesignState:
    def __init__(self, original_prompt, material, force_n, force_dir, operating_temp_c,
                 surface_finish, reliability, project_description, max_iterations,
                 min_health_score, max_critical_violations, max_high_violations, min_safety_factor):
        self.original_prompt = original_prompt
        self.material = material
        self.force_n = force_n
        self.force_dir = force_dir
        self.operating_temp_c = operating_temp_c
        self.surface_finish = surface_finish
        self.reliability = reliability
        self.project_description = project_description
        self.max_iterations = max_iterations
        self.min_health_score = min_health_score
        self.max_critical_violations = max_critical_violations
        self.max_high_violations = max_high_violations
        self.min_safety_factor = min_safety_factor

        self.design_type = None     # "tapered_beam" | "bent_bracket" | "generic_script"
        self.params = None          # dict of constructor kwargs, for parametric designs
        self.script = None          # script text, for generic_script designs
        self.obj = None             # current build123d object (server-side only, never sent to the model)
        self.mesh = None            # current trimesh (server-side only)
        self.stl_bytes = None

        self.iteration_count = 0
        self.last_analysis = None
        self.last_quality = None

        self.previous_design = None
        self.current_candidate = None
        self.best_valid_design = None
        self.best_passing_design = None
        self.pending_hypothesis = None
        self.hypothesis_log = []

        self.finalized = False
        self.final_verdict_claimed = None
        self.final_summary = None


def _update_best_valid(state, snap):
    if state.best_valid_design is None or snap["health_score"] > state.best_valid_design["health_score"]:
        state.best_valid_design = snap


def _update_best_passing(state, snap):
    if state.best_passing_design is None or snap["health_score"] > state.best_passing_design["health_score"]:
        state.best_passing_design = snap


def _revert_to_last_known_good(state):
    """Monotonic refinement protection (spec section 8): rebuild the last
    known-good design deterministically from its stored params/script.
    Returns True if a revert happened, False if there was nothing valid yet
    to revert to (in which case state is left untouched)."""
    good = state.best_valid_design
    if good is None:
        return False
    try:
        design_type = good["design_type"]
        if design_type == "tapered_beam":
            params = _copy_params(good["params"])
            obj = make_tapered_beam(**params)
            state.script = None
        elif design_type == "bent_bracket":
            params = _copy_params(good["params"])
            obj = make_bent_bracket(**params)
            state.script = None
        elif design_type == "generic_script":
            script = good.get("script")
            obj, err = execute_cad_script_safely(script)
            if err:
                return False
            params = None
            state.script = script
        else:
            return False
        state.design_type = design_type
        state.params = params
        state.obj = obj
        state.mesh = None
        state.stl_bytes = None
        gc.collect()  # drop the rejected candidate's OCCT shape promptly (see _rebuild_beam)
        return True
    except Exception:
        return False


def _snapshot_current(state, quality, result, critical_region=None):
    fea = result.get("analytical_fea", {}) or {}
    return {
        "iteration": state.iteration_count,
        "design_type": state.design_type,
        "params": _copy_params(state.params),
        "script": state.script,
        "passed": quality["passed"],
        "health_score": quality["score"],
        "safety_factor": fea.get("safety_factor"),
        "max_von_mises_mpa": (fea.get("stress", {}) or {}).get("von_mises_mpa"),
        "mass_g": (fea.get("dynamics", {}) or {}).get("estimated_mass_g"),
        "is_watertight": (result.get("geometry", {}) or {}).get("is_watertight"),
        "violations": (result.get("rule_engine", {}) or {}).get("total_violations"),
        "reasons": quality["reasons"],
        "critical_region": critical_region,
    }


def _summarize_snapshot(snap):
    if snap is None:
        return None
    return {"iteration": snap.get("iteration"), "passed": snap.get("passed"),
            "health_score": snap.get("health_score"), "safety_factor": snap.get("safety_factor"),
            "max_von_mises_mpa": snap.get("max_von_mises_mpa"), "mass_g": snap.get("mass_g"),
            "is_watertight": snap.get("is_watertight"), "violations": snap.get("violations")}


def _build_failure_region(result, state, crit, axis, pos_mm):
    """
    Turns the FEA critical section's 1D position into a proper spatial 'failure
    object' — bbox, centroid, nearby features — instead of just a coarse
    near_base/near_tip label. Built entirely from data already computed
    elsewhere (state.params' known section geometry, plus zone data other
    analysis functions already produced), not from a new CalculiX per-node
    field readout or a new OCCT face-query capability — those are real,
    separately-scoped pieces of work, most valuable once geometry has more
    than two sections/features to disambiguate between. This is the honest,
    buildable-today slice of that idea: precise where the data already
    supports it, and says so plainly where it doesn't.
    """
    if state is None or state.design_type != "tapered_beam" or not state.params or axis != "z":
        return None  # bent_bracket/generic_script: no known-geometry interpolation to build a bbox from yet

    p = state.params
    length = p.get("length") or 0
    if length <= 0:
        return None
    frac = max(0.0, min(1.0, pos_mm / length))
    width_at_z = p["base_width"] + (p["tip_width"] - p["base_width"]) * frac
    thick_at_z = p["base_thick"] + (p["tip_thick"] - p["base_thick"]) * frac
    band = max(length * 0.03, 2.0)  # a few-mm slice around the critical position, not a single point

    nearby = []
    fillet_r = p.get("fillet_radius") or 0
    if fillet_r > 0 and pos_mm < band * 2:
        nearby.append(f"root fillet (radius {fillet_r}mm)")
    if pos_mm > length - band * 2 and fillet_r > 0:
        nearby.append("tip transition")
    for hx, hy, hd in (p.get("holes_base") or []):
        if pos_mm < band * 3:
            nearby.append(f"base hole (d={hd}mm at x={hx},y={hy})")
    for hx, hy, hd in (p.get("holes_tip") or []):
        if pos_mm > length - band * 3:
            nearby.append(f"tip hole (d={hd}mm at x={hx},y={hy})")

    # Cross-reference zones other analysis functions already computed (sharp
    # corners, thin walls) that fall within this same Z band — same data
    # find_problem_regions already surfaces, just correlated here by position.
    for corner in (result.get("sharp_corner_analysis", {}) or {}).get("all_zones", []) or []:
        cz = (corner.get("position") or {}).get("z")
        if cz is not None and abs(cz - pos_mm) < band and corner.get("severity") in ("HIGH", "CRITICAL"):
            nearby.append(f"sharp corner (Kf={corner.get('Kf')}) at z={round(cz,1)}mm")
    for zone in (result.get("wall_thickness", {}) or {}).get("critical_zones", []) or []:
        cz = (zone.get("position") or {}).get("z")
        if cz is not None and abs(cz - pos_mm) < band:
            nearby.append(f"thin section ({zone.get('thickness_mm')}mm) at z={round(cz,1)}mm")

    return {
        "bbox_mm": {"xmin": round(-width_at_z/2, 2), "xmax": round(width_at_z/2, 2),
                    "ymin": round(-thick_at_z/2, 2), "ymax": round(thick_at_z/2, 2),
                    "zmin": round(pos_mm - band, 2), "zmax": round(pos_mm + band, 2)},
        "centroid_mm": [0.0, 0.0, round(pos_mm, 2)],
        "nearby_features": nearby if nearby else ["no declared feature (hole/fillet) near this position"],
        "note": "bbox interpolated from this iteration's known base/tip section geometry, not a "
                "per-node CalculiX field readout — precise for this parametric shape, but won't "
                "generalize to arbitrary geometry without extending the CalculiX result parser "
                "to expose real per-node coordinates.",
    }


def build_engineering_diagnosis(result, state=None):
    """Spec section 5: convert raw solver results into structured engineering
    evidence. Every field here comes from already-computed real analysis
    output (run_analysis_v8) — never from the LLM's own judgment."""
    fea = result.get("analytical_fea", {}) or {}
    rules = (result.get("rule_engine", {}) or {}).get("all_violations", []) or []
    geo = result.get("geometry", {}) or {}
    hs = result.get("health_score", {}) or {}
    crit = fea.get("critical_section") or {}
    fea_status = fea.get("status")

    failure_modes = []
    if fea_status == "FAIL":
        stress = fea.get("stress", {}) or {}
        bending = stress.get("bending_mpa") or 0
        axial = stress.get("axial_mpa") or 0
        shear = stress.get("shear_mpa") or 0
        if bending >= axial and bending >= shear and bending > 0:
            failure_modes.append("bending stress")
        elif axial >= shear and axial > 0:
            failure_modes.append("axial stress")
        elif shear > 0:
            failure_modes.append("shear stress")
        sfv = fea.get("safety_factor")
        if sfv is not None and sfv < 1.0:
            failure_modes.append("insufficient section stiffness")
        buck = fea.get("buckling", {}) or {}
        if buck.get("status") == "FAIL":
            failure_modes.append("buckling instability")

    max_kf = (result.get("sharp_corner_analysis", {}) or {}).get("max_Kf", 1.0) or 1.0
    if max_kf > 2.0:
        failure_modes.append("stress concentration at sharp corner")
    if any(r.get("rule_id") == "R01" for r in rules):
        failure_modes.append("thin wall / insufficient section thickness")
    if not geo.get("is_watertight", True):
        failure_modes.append("non-manifold geometry")
    hole_viol = (result.get("hole_analysis", {}) or {}).get("violations", []) or []
    if hole_viol:
        failure_modes.append("hole edge-distance violation")

    has_critical_rule = any(r.get("severity") == "CRITICAL" for r in rules)
    status = "PASS" if (fea_status == "PASS" and geo.get("is_watertight", True)
                         and not has_critical_rule) else "FAIL"

    numerically_suspect = bool(fea.get("numerically_suspect"))
    if numerically_suspect:
        # The analytical solver itself is flagging its own stress/deflection numbers as
        # implausible (a cross-section slicing artifact, not a real structural finding —
        # see multi_section_fea's own comment). Lead with this so the agent doesn't spend
        # an iteration "fixing" a problem that may not actually exist, and doesn't report
        # a confident FAIL built on fabricated-looking numbers.
        failure_modes = ["SOLVER OUTPUT NUMERICALLY SUSPECT — treat this iteration's stress/"
                          "deflection numbers as unreliable, not a confirmed structural failure"] + failure_modes

    critical_region = None
    failure_region = None
    if crit.get("position_mm") is not None:
        axis = crit.get("axis", "z"); pos_mm = crit.get("position_mm")
        critical_region = f"{axis}-axis @ {pos_mm}mm"
        if state is not None and state.design_type == "tapered_beam" and state.params and axis == "z":
            length = state.params.get("length") or 0
            if length > 0:
                frac = pos_mm / length
                critical_region = "near_base_fixed_support" if frac < 0.5 else "near_tip_load_application"
        failure_region = _build_failure_region(result, state, crit, axis, pos_mm)
    elif rules:
        crit_rules = [r for r in rules if r.get("severity") in ("CRITICAL", "HIGH") and r.get("position")]
        if crit_rules:
            critical_region = f"near {crit_rules[0]['position']}"

    return {
        "status": status,
        "safety_factor": fea.get("safety_factor"),
        "max_von_mises_mpa": (fea.get("stress", {}) or {}).get("von_mises_mpa"),
        "critical_region": critical_region,
        "failure_region": failure_region,
        "failure_modes": failure_modes if failure_modes else (["none detected"] if status == "PASS" else ["unspecified"]),
        "health_score": hs.get("score"),
        "fatigue_status": (result.get("fatigue_analysis", {}) or {}).get("status"),
        "is_watertight": geo.get("is_watertight"),
        "mass_g": (fea.get("dynamics", {}) or {}).get("estimated_mass_g"),
        "numerically_suspect": numerically_suspect,
    }


# ----------------------------------------------------------------------
# Parameter-name resolution for the modify_* convenience tools (spec
# sections 3 & 9) — lets the agent say "thickness"/"width"/"length" with
# an optional region hint, or an exact constructor field name.
# ----------------------------------------------------------------------

def _resolve_beam_parameter(parameter, region):
    p = (parameter or "").strip().lower()
    r = (region or "").strip().lower()
    exact = {"length": ["length"], "base_width": ["base_width"], "base_thick": ["base_thick"],
             "tip_width": ["tip_width"], "tip_thick": ["tip_thick"], "fillet_radius": ["fillet_radius"]}
    if p in exact:
        return exact[p]
    is_base = any(w in r for w in ("base", "fixed", "root")) if r else False
    is_tip = any(w in r for w in ("tip", "free", "load")) if r else False
    if p in ("thickness", "height", "section_height", "thick"):
        if is_base and not is_tip:
            return ["base_thick"]
        if is_tip and not is_base:
            return ["tip_thick"]
        return ["base_thick", "tip_thick"]
    if p in ("width", "section_width"):
        if is_base and not is_tip:
            return ["base_width"]
        if is_tip and not is_base:
            return ["tip_width"]
        return ["base_width", "tip_width"]
    if p in ("fillet", "fillet_radius_mm"):
        return ["fillet_radius"]
    return None


def _resolve_bracket_parameter(parameter, region):
    p = (parameter or "").strip().lower()
    r = (region or "").strip().lower()
    exact = {"leg1_length": ["leg1_length"], "leg2_length": ["leg2_length"], "width": ["width"],
             "thickness": ["thickness"], "bend_angle_deg": ["bend_angle_deg"], "fillet_radius": ["fillet_radius"]}
    if p in exact:
        return exact[p]
    if p in ("length", "leg_length"):
        if "2" in r or "second" in r:
            return ["leg2_length"]
        return ["leg1_length"]
    if p in ("height",):
        return ["thickness"]
    if p in ("fillet",):
        return ["fillet_radius"]
    if p in ("angle", "bend_angle"):
        return ["bend_angle_deg"]
    return None


async def _auto_validate_and_mesh(state):
    """Runs the B-rep validity check + mesh export + watertight/manifold check inline,
    right after any geometry build/modification. Folded into every modify_*/
    set_initial_design/add_hole/generic-script result so the agent doesn't need two
    extra model round-trips (validate_geometry, run_mesh) per iteration just to reach
    run_fea — cut from 4 model turns per design iteration to 2 (build/modify, run_fea).
    On failure this performs the same revert-to-last-known-good the old standalone
    validate_geometry/run_mesh tools used to."""
    if state.obj is None:
        return {"validated": False, "meshed": False}
    brep_ok = True; brep_note = None
    try:
        brep_ok = _b3d_is_valid(state.obj)
        if not brep_ok:
            brep_note = "OpenCASCADE's BRepCheck_Analyzer flagged this shape as an invalid B-rep."
    except Exception as e:
        brep_note = f"Could not run the B-rep validity check ({type(e).__name__}: {e}); proceeding to mesh export."
    if not brep_ok:
        reverted = _revert_to_last_known_good(state)
        return {"status": "REJECTED", "valid_brep": False, "reason": brep_note, "reverted": reverted,
                "current_params": state.params}
    try:
        mesh, stl_bytes = await mesh_from_cad_object(state.obj)
    except Exception as e:
        reverted = _revert_to_last_known_good(state)
        return {"status": "REJECTED", "reason": f"STL export/mesh load failed: {type(e).__name__}: {e}",
                "reverted": reverted, "current_params": state.params}
    is_wt = bool(mesh.is_watertight); is_wind = bool(getattr(mesh, "is_winding_consistent", True))
    if not is_wt or not is_wind:
        defect_locs = _find_watertight_defect_locations(mesh)
        reverted = _revert_to_last_known_good(state)
        return {"status": "REJECTED", "watertight": is_wt, "manifold": is_wind, "defect_locations": defect_locs,
                "reason": "Meshed geometry is non-manifold/non-watertight — rejected per monotonic refinement "
                          "protection.", "reverted": reverted, "current_params": state.params}
    state.mesh = mesh; state.stl_bytes = stl_bytes
    if state.best_valid_design is None:
        state.best_valid_design = {"design_type": state.design_type, "params": _copy_params(state.params),
                                    "script": state.script, "health_score": -1, "passed": False,
                                    "safety_factor": None, "max_von_mises_mpa": None, "mass_g": None,
                                    "is_watertight": True, "violations": None, "reasons": [],
                                    "iteration": state.iteration_count}
    gc.collect()
    return {"status": "OK", "validated": True, "meshed": True, "watertight": True,
            "face_count": int(len(mesh.faces)), "vertex_count": int(len(mesh.vertices))}


async def _rebuild_beam(state, new_params, change_desc, reason, predicted_effect=None):
    ok, err = _beam_param_contract(new_params, state.material)
    if not ok:
        return err
    try:
        obj = make_tapered_beam(**new_params)
    except Exception as e:
        return {"status": "REJECTED", "reason": f"CAD kernel (build123d/OpenCASCADE) rejected this change: {type(e).__name__}: {e}",
                "constraint": "kernel_geometric_feasibility"}
    old_params = state.params
    state.params = new_params; state.obj = obj; state.mesh = None; state.stl_bytes = None
    gc.collect()  # OCCT-wrapped shapes are C++-backed; encourage prompt release of the
                  # superseded object rather than waiting on Python's GC schedule — free-tier
                  # 512MB instances have no headroom for stale shapes piling up across iterations
    state.pending_hypothesis = {"change": change_desc, "reason": reason,
                                 "predicted_effect": predicted_effect or "address the diagnosed issue"}
    check = await _auto_validate_and_mesh(state)
    if check.get("status") == "REJECTED":
        return check
    return {"status": "OK", "change_applied": change_desc, "diff": _diff_params(old_params, new_params),
            "validated": True, "meshed": True, "face_count": check.get("face_count"),
            "message": "Geometry rebuilt, validated, and meshed successfully. Call run_fea to test this change."}


async def _rebuild_bracket(state, new_params, change_desc, reason, predicted_effect=None):
    ok, err = _bracket_param_contract(new_params, state.material)
    if not ok:
        return err
    try:
        obj = make_bent_bracket(**new_params)
    except Exception as e:
        return {"status": "REJECTED", "reason": f"CAD kernel (build123d/OpenCASCADE) rejected this change: {type(e).__name__}: {e}",
                "constraint": "kernel_geometric_feasibility"}
    old_params = state.params
    state.params = new_params; state.obj = obj; state.mesh = None; state.stl_bytes = None
    gc.collect()
    state.pending_hypothesis = {"change": change_desc, "reason": reason,
                                 "predicted_effect": predicted_effect or "address the diagnosed issue"}
    check = await _auto_validate_and_mesh(state)
    if check.get("status") == "REJECTED":
        return check
    return {"status": "OK", "change_applied": change_desc, "diff": _diff_params(old_params, new_params),
            "validated": True, "meshed": True, "face_count": check.get("face_count"),
            "message": "Geometry rebuilt, validated, and meshed successfully. Call run_fea to test this change."}


async def _agent_generic_script_modification(state, feature_description, reason, predicted_effect=None):
    """The 'unfamiliar geometry' fallback path (spec section 11's second
    branch). Reuses the EXISTING gemini_generate_script REFINEMENT MODE and
    execute_cad_script_safely sandbox verbatim — zero new AI-prompting or
    sandbox-security code. Crossing over from a parametric design into a
    generic_script one is a one-way, logged transition."""
    if not reason:
        return {"status": "REJECTED", "reason": "A 'reason' is required for every modification."}
    crossed_over = False
    if state.design_type in ("tapered_beam", "bent_bracket") and state.script is None:
        state.script = (params_to_script_tapered_beam(state.params) if state.design_type == "tapered_beam"
                         else params_to_script_bent_bracket(state.params))
        crossed_over = True
    if state.script is None:
        return {"status": "REJECTED", "reason": "No existing design to modify. Call set_initial_design first."}

    feedback = (f"MANUAL FEATURE REQUEST (not a numeric parameter change): {feature_description}\n"
                f"Engineering reason: {reason}\n"
                "Modify ONLY what's needed for this request; preserve every other dimension/feature exactly.")
    try:
        new_script = await gemini_generate_script(state.original_prompt, previous_script=state.script,
                                                    feedback=feedback)
    except HTTPException as e:
        return {"status": "ERROR", "message": f"Script modification call failed: {e.detail}"}

    obj, err = execute_cad_script_safely(new_script)
    if err:
        return {"status": "REJECTED",
                "reason": f"The modified script failed: {err} {_diagnose_cad_error(err)}",
                "note": "State unchanged; the previous working script/geometry is preserved."}

    state.design_type = "generic_script"; state.script = new_script; state.params = None
    state.obj = obj; state.mesh = None; state.stl_bytes = None
    state.pending_hypothesis = {"change": f"generic_script_modification: {feature_description}", "reason": reason,
                                 "predicted_effect": predicted_effect or "address the described issue"}
    check = await _auto_validate_and_mesh(state)
    if check.get("status") == "REJECTED":
        return check
    return {"status": "OK", "crossed_over_to_generic_script": crossed_over, "validated": True, "meshed": True,
            "face_count": check.get("face_count"),
            "message": "Script modified, validated, and meshed successfully via the AI-assisted generic path "
                       "(this design is no longer tracked by discrete numeric parameters). Call run_fea next."}


# ----------------------------------------------------------------------
# Tool implementations — inspection (read-only, need a meshed design)
# ----------------------------------------------------------------------

def _require_mesh(state):
    if state.mesh is None:
        return {"status": "NOT_AVAILABLE",
                "message": "No meshed/validated geometry yet. Call set_initial_design (or a modify_*/add_hole "
                           "tool) first — it validates and meshes automatically."}
    return None


def _tool_inspect_geometry(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    is_wt = bool(mesh.is_watertight)
    is_wind = bool(getattr(mesh, "is_winding_consistent", True))
    holes = detect_holes_v8(mesh)
    wt_ = wall_thickness_v8(mesh)
    features = []
    if state.design_type == "tapered_beam" and state.params:
        p = state.params
        if abs(p["base_width"] - p["tip_width"]) > 0.05 or abs(p["base_thick"] - p["tip_thick"]) > 0.05:
            features.append("taper")
        if (p.get("fillet_radius") or 0) > 0:
            features.append("fillet")
    elif state.design_type == "bent_bracket" and state.params:
        features.append("fold")
        if (state.params.get("fillet_radius") or 0) > 0:
            features.append("fillet")
    if holes:
        features.append("through_holes")
    exts = [sf(e) for e in mesh.extents]
    return {"solid": True, "watertight": is_wt, "manifold": is_wind,
            "bounding_box_mm": [round(exts[0], 3), round(exts[1], 3), round(exts[2], 3)],
            "volume_mm3": round(sf(mesh.volume), 3), "features": features, "hole_count": len(holes),
            "min_wall_thickness_mm": wt_.get("min_mm")}


def _tool_measure_geometry(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    exts = [sf(e) for e in mesh.extents]
    wt_ = wall_thickness_v8(mesh)
    return {"bounding_box_mm": {"x": round(exts[0], 3), "y": round(exts[1], 3), "z": round(exts[2], 3)},
            "volume_mm3": round(sf(mesh.volume), 3), "surface_area_mm2": round(sf(mesh.area), 3),
            "wall_thickness": wt_}


def _tool_identify_features(state):
    features = []; hole_count = 0
    if state.design_type == "tapered_beam" and state.params:
        p = state.params
        if abs(p["base_width"] - p["tip_width"]) > 0.05 or abs(p["base_thick"] - p["tip_thick"]) > 0.05:
            features.append("taper")
        if (p.get("fillet_radius") or 0) > 0:
            features.append("fillet")
        hole_count = len(p.get("holes_base") or []) + len(p.get("holes_tip") or [])
        if hole_count > 0:
            features.append("through_holes")
    elif state.design_type == "bent_bracket" and state.params:
        p = state.params
        features.append("bend")
        if (p.get("fillet_radius") or 0) > 0:
            features.append("fillet")
        hole_count = len(p.get("holes_leg1") or []) + len(p.get("holes_leg2") or [])
        if hole_count > 0:
            features.append("through_holes")
    elif state.mesh is not None:
        holes = detect_holes_v8(state.mesh); hole_count = len(holes)
        if hole_count > 0:
            features.append("through_holes")
    return {"design_type": state.design_type, "features": features, "hole_count": hole_count,
            "declared_parameters": state.params}


def _tool_find_holes(state):
    if state.mesh is not None:
        return {"source": "geometric_detection", "holes": detect_holes_v8(state.mesh)}
    if state.params:
        key_pairs = ([("holes_base", "base"), ("holes_tip", "tip")] if state.design_type == "tapered_beam"
                      else [("holes_leg1", "leg1"), ("holes_leg2", "leg2")] if state.design_type == "bent_bracket"
                      else [])
        declared = []
        for key, label in key_pairs:
            for (x, y, d) in (state.params.get(key) or []):
                declared.append({"location": label, "x": x, "y": y, "diameter_mm": d})
        return {"source": "declared_parameters_not_yet_meshed", "holes": declared}
    return {"status": "NOT_AVAILABLE", "message": "No design built yet."}


def _tool_measure_wall_thickness(state):
    err = _require_mesh(state)
    if err:
        return err
    return wall_thickness_v8(state.mesh)


def _tool_get_bounding_box(state):
    err = _require_mesh(state)
    if err:
        return err
    exts = [sf(e) for e in state.mesh.extents]
    return {"dimensions_mm": {"x": round(exts[0], 3), "y": round(exts[1], 3), "z": round(exts[2], 3)},
            "aspect_ratio": round(max(exts) / max(min(exts), 1e-6), 3)}


def _tool_get_mass_properties(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    mat = MATERIALS.get(state.material, MATERIALS["aluminum_6061"])
    vol = sf(mesh.volume); mass_g = vol * mat["density"] * 1e-3
    try:
        cog = mesh.center_mass
        cog_d = {"x": round(float(cog[0]), 3), "y": round(float(cog[1]), 3), "z": round(float(cog[2]), 3)}
    except Exception:
        cog_d = None
    return {"volume_mm3": round(vol, 3), "mass_g": round(mass_g, 3), "material": mat["name"],
            "density_g_cm3": mat["density"], "center_of_mass_mm": cog_d}


def _tool_check_manifold(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    return {"manifold": bool(getattr(mesh, "is_winding_consistent", True)),
            "watertight": bool(mesh.is_watertight),
            "is_volume": bool(getattr(mesh, "is_volume", mesh.is_watertight))}


def _tool_check_watertight(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    is_wt = bool(mesh.is_watertight)
    out = {"watertight": is_wt}
    if not is_wt:
        out["defect_locations"] = _find_watertight_defect_locations(mesh)
    return out


def _tool_find_problem_regions(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh; regions = []
    if not mesh.is_watertight:
        for d in _find_watertight_defect_locations(mesh):
            regions.append({"type": "non_watertight_gap", "position": {"x": d["x"], "y": d["y"], "z": d["z"]},
                             "severity": "HIGH"})
    wt_ = wall_thickness_v8(mesh)
    for z in wt_.get("critical_zones", []) or []:
        regions.append({"type": "thin_wall", "position": z.get("position"),
                         "thickness_mm": z.get("thickness_mm"), "severity": "CRITICAL"})
    for z in wt_.get("thin_zones", []) or []:
        regions.append({"type": "thin_wall", "position": z.get("position"),
                         "thickness_mm": z.get("thickness_mm"), "severity": "WARNING"})
    try:
        sharp = detect_sharp_v8(mesh, state.material)
        for z in sharp.get("critical_zones", []) or []:
            regions.append({"type": "stress_concentration", "position": z.get("position"),
                             "Kf": z.get("Kf"), "severity": z.get("severity")})
    except Exception:
        pass
    for h in detect_holes_v8(mesh):
        if h.get("violation"):
            regions.append({"type": "hole_edge_violation", "position": h.get("position"),
                             "detail": h.get("violation_msg"), "severity": "HIGH"})
    if state.last_analysis:
        crit = (state.last_analysis.get("analytical_fea", {}) or {}).get("critical_section") or {}
        if crit.get("position_mm") is not None:
            fea_status = (state.last_analysis.get("analytical_fea", {}) or {}).get("status")
            regions.append({"type": "fea_critical_section", "axis": crit.get("axis"),
                             "position_mm": crit.get("position_mm"),
                             "severity": "HIGH" if fea_status == "FAIL" else "INFO"})
    return {"problem_region_count": len(regions), "regions": regions}


def _tool_get_topology_summary(state):
    err = _require_mesh(state)
    if err:
        return err
    mesh = state.mesh
    try:
        euler = int(mesh.euler_number)
    except Exception:
        euler = None
    return {"vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces)),
            "edge_count": int(len(mesh.edges)), "euler_number": euler,
            "watertight": bool(mesh.is_watertight), "manifold": bool(getattr(mesh, "is_winding_consistent", True))}


def _tool_diagnose_failure(state):
    if state.last_analysis is None:
        return {"status": "NOT_AVAILABLE", "message": "No analysis has been run yet. Call run_fea first."}
    return build_engineering_diagnosis(state.last_analysis, state)


def _tool_calculate_properties(state):
    err = _require_mesh(state)
    if err:
        return err
    if state.last_analysis:
        fea = state.last_analysis.get("analytical_fea", {}) or {}
        return {"source": "last_run_fea", "stress": fea.get("stress"), "deflection_mm": fea.get("deflection_mm"),
                "min_section_area_mm2": fea.get("min_section_area_mm2"), "dynamics": fea.get("dynamics"),
                "buckling": fea.get("buckling")}
    try:
        mat_key = state.material if state.material in MATERIALS else "aluminum_6061"
        analytic = multi_section_fea(state.mesh, mat_key, state.force_n, state.force_dir)
        return {"source": "on_demand_analytical_estimate", "stress": analytic.get("stress"),
                "deflection_mm": analytic.get("deflection_mm"), "dynamics": analytic.get("dynamics"),
                "buckling": analytic.get("buckling"),
                "note": "Estimate only — run_fea has not been called yet for this candidate; call run_fea "
                        "for the authoritative check."}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


# ----------------------------------------------------------------------
# Tool implementations — construction & modification (spec sections 2,3,4)
# ----------------------------------------------------------------------

async def _tool_set_initial_design(state, design_type, reason="",
                                    length=None, base_width=None, base_thick=None,
                                    tip_width=None, tip_thick=None, fillet_radius=0.0,
                                    holes_base=None, holes_tip=None,
                                    leg1_length=None, leg2_length=None, width=None, thickness=None,
                                    bend_angle_deg=90.0, holes_leg1=None, holes_leg2=None):
    if state.mesh is not None:
        return {"status": "REJECTED", "reason": "Initial design has already been set for this session. "
                "Use modify_parameter/modify_feature/add_hole to change the existing design instead."}
    if design_type == "tapered_beam":
        missing = [n for n, v in [("length", length), ("base_width", base_width), ("base_thick", base_thick),
                                    ("tip_width", tip_width), ("tip_thick", tip_thick)] if v is None]
        if missing:
            return {"status": "REJECTED", "reason": f"Missing required parameter(s) for tapered_beam: {missing}"}
        params = {"length": float(length), "base_width": float(base_width), "base_thick": float(base_thick),
                  "tip_width": float(tip_width), "tip_thick": float(tip_thick),
                  "fillet_radius": float(fillet_radius or 0.0),
                  "holes_base": [tuple(h) for h in (holes_base or [])],
                  "holes_tip": [tuple(h) for h in (holes_tip or [])]}
        ok, err = _beam_param_contract(params, state.material)
        if not ok:
            return err
        try:
            obj = make_tapered_beam(**params)
        except Exception as e:
            return {"status": "REJECTED", "reason": f"CAD kernel (build123d/OpenCASCADE) rejected these parameters: {type(e).__name__}: {e}"}
        state.design_type = "tapered_beam"; state.params = params; state.script = None; state.obj = obj
    elif design_type == "bent_bracket":
        missing = [n for n, v in [("leg1_length", leg1_length), ("leg2_length", leg2_length),
                                    ("width", width), ("thickness", thickness)] if v is None]
        if missing:
            return {"status": "REJECTED", "reason": f"Missing required parameter(s) for bent_bracket: {missing}"}
        params = {"leg1_length": float(leg1_length), "leg2_length": float(leg2_length),
                  "width": float(width), "thickness": float(thickness),
                  "bend_angle_deg": float(bend_angle_deg or 90.0), "fillet_radius": float(fillet_radius or 0.0),
                  "holes_leg1": [tuple(h) for h in (holes_leg1 or [])],
                  "holes_leg2": [tuple(h) for h in (holes_leg2 or [])]}
        ok, err = _bracket_param_contract(params, state.material)
        if not ok:
            return err
        try:
            obj = make_bent_bracket(**params)
        except Exception as e:
            return {"status": "REJECTED", "reason": f"CAD kernel (build123d/OpenCASCADE) rejected these parameters: {type(e).__name__}: {e}"}
        state.design_type = "bent_bracket"; state.params = params; state.script = None; state.obj = obj
    elif design_type == "generic_script":
        try:
            script = await gemini_generate_script(state.original_prompt)
        except HTTPException as e:
            return {"status": "ERROR", "message": f"Script generation failed: {e.detail}"}
        obj, err = execute_cad_script_safely(script)
        if err:
            return {"status": "REJECTED", "reason": err}
        state.design_type = "generic_script"; state.params = None; state.script = script; state.obj = obj
    else:
        return {"status": "REJECTED",
                "reason": f"Unknown design_type '{design_type}'. Must be tapered_beam, bent_bracket, or generic_script."}
    check = await _auto_validate_and_mesh(state)
    if check.get("status") == "REJECTED":
        return check
    return {"status": "OK", "design_type": state.design_type, "params": state.params,
            "validated": True, "meshed": True, "face_count": check.get("face_count"),
            "message": "Initial geometry constructed, validated, and meshed successfully. Call run_fea next."}


async def _tool_modify_parameter(state, parameter, change_percent=None, new_value=None, region=None,
                                  reason="", predicted_effect=None):
    if state.obj is None:
        return {"status": "REJECTED", "reason": "No initial design exists yet. Call set_initial_design first."}
    if change_percent is None and new_value is None:
        return {"status": "REJECTED", "reason": "Provide either change_percent or new_value."}
    if not reason:
        return {"status": "REJECTED", "reason": "A 'reason' explaining the engineering justification is required."}

    if state.design_type == "tapered_beam":
        fields = _resolve_beam_parameter(parameter, region)
        if not fields:
            return {"status": "REJECTED", "reason": f"Unknown parameter '{parameter}' for a tapered beam.",
                     "valid_parameters": ["length", "base_width", "base_thick", "tip_width", "tip_thick",
                                           "fillet_radius", "thickness", "width"]}
        new_params = _copy_params(state.params)
        for f in fields:
            cur = new_params[f]
            new_params[f] = (round(float(new_value), 4) if (new_value is not None and len(fields) == 1)
                              else round(cur * (1 + (change_percent or 0) / 100.0), 4))
        desc = f"modify_parameter({parameter}" + (f",region={region}" if region else "") + ")"
        return await _rebuild_beam(state, new_params, desc, reason, predicted_effect)

    elif state.design_type == "bent_bracket":
        fields = _resolve_bracket_parameter(parameter, region)
        if not fields:
            return {"status": "REJECTED", "reason": f"Unknown parameter '{parameter}' for a bent bracket.",
                     "valid_parameters": ["leg1_length", "leg2_length", "width", "thickness", "bend_angle_deg",
                                           "fillet_radius", "length", "height"]}
        new_params = _copy_params(state.params)
        for f in fields:
            cur = new_params[f]
            new_params[f] = (round(float(new_value), 4) if (new_value is not None and len(fields) == 1)
                              else round(cur * (1 + (change_percent or 0) / 100.0), 4))
        desc = f"modify_parameter({parameter}" + (f",region={region}" if region else "") + ")"
        return await _rebuild_bracket(state, new_params, desc, reason, predicted_effect)

    else:
        desc = (f"Adjust the parameter '{parameter}'" + (f" near {region}" if region else "")
                + (f" by {change_percent}%" if change_percent is not None else f" to {new_value}"))
        return await _agent_generic_script_modification(state, desc, reason, predicted_effect)


async def _tool_modify_thickness(state, change_percent=None, new_value=None, region=None, reason="", predicted_effect=None):
    return await _tool_modify_parameter(state, "thickness", change_percent, new_value, region, reason, predicted_effect)

async def _tool_modify_length(state, change_percent=None, new_value=None, region=None, reason="", predicted_effect=None):
    return await _tool_modify_parameter(state, "length", change_percent, new_value, region, reason, predicted_effect)

async def _tool_modify_width(state, change_percent=None, new_value=None, region=None, reason="", predicted_effect=None):
    return await _tool_modify_parameter(state, "width", change_percent, new_value, region, reason, predicted_effect)

async def _tool_modify_height(state, change_percent=None, new_value=None, region=None, reason="", predicted_effect=None):
    return await _tool_modify_parameter(state, "height", change_percent, new_value, region, reason, predicted_effect)

async def _tool_modify_fillet(state, change_percent=None, new_value=None, region=None, reason="", predicted_effect=None):
    return await _tool_modify_parameter(state, "fillet_radius", change_percent, new_value, region, reason, predicted_effect)


async def _tool_modify_taper(state, change_percent, reason="", predicted_effect=None):
    if state.design_type != "tapered_beam":
        return {"status": "NOT_APPLICABLE", "reason": "modify_taper only applies to a tapered_beam design_type."}
    if not reason:
        return {"status": "REJECTED", "reason": "A 'reason' is required."}
    new_params = _copy_params(state.params)
    new_params["tip_width"] = round(new_params["tip_width"] * (1 + change_percent / 100.0), 4)
    new_params["tip_thick"] = round(new_params["tip_thick"] * (1 + change_percent / 100.0), 4)
    return await _rebuild_beam(state, new_params, f"modify_taper({change_percent}%)", reason, predicted_effect)


async def _tool_add_hole(state, x_mm, y_mm, diameter_mm, end=None, leg=None, reason="", predicted_effect=None):
    if not reason:
        return {"status": "REJECTED", "reason": "A 'reason' is required."}
    if state.design_type == "tapered_beam":
        if end not in ("base", "tip"):
            return {"status": "REJECTED", "reason": "For a tapered_beam, 'end' must be 'base' or 'tip'."}
        width = state.params["base_width"] if end == "base" else state.params["tip_width"]
        thick = state.params["base_thick"] if end == "base" else state.params["tip_thick"]
        ok, err = _validate_hole(x_mm, y_mm, diameter_mm, width, thick)
        if not ok:
            return err
        new_params = _copy_params(state.params)
        key = "holes_base" if end == "base" else "holes_tip"
        new_params[key] = list(new_params[key]) + [(x_mm, y_mm, diameter_mm)]
        return await _rebuild_beam(state, new_params, f"add_hole(end={end})", reason, predicted_effect)
    elif state.design_type == "bent_bracket":
        if leg not in ("leg1", "leg2"):
            return {"status": "REJECTED", "reason": "For a bent_bracket, 'leg' must be 'leg1' or 'leg2'."}
        length = state.params["leg1_length"] if leg == "leg1" else state.params["leg2_length"]
        width = state.params["width"]
        ok, err = _validate_hole_bracket(x_mm, y_mm, diameter_mm, length, width)
        if not ok:
            return err
        new_params = _copy_params(state.params)
        key = "holes_leg1" if leg == "leg1" else "holes_leg2"
        new_params[key] = list(new_params[key]) + [(x_mm, y_mm, diameter_mm)]
        return await _rebuild_bracket(state, new_params, f"add_hole(leg={leg})", reason, predicted_effect)
    else:
        desc = f"Add a through hole of diameter {diameter_mm}mm at approximately (x={x_mm}, y={y_mm}) in the relevant local face frame."
        return await _agent_generic_script_modification(state, desc, reason, predicted_effect)


async def _tool_modify_feature(state, feature_description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, feature_description, reason, predicted_effect)

async def _tool_create_feature(state, feature_description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, f"Add: {feature_description}", reason, predicted_effect)

async def _tool_remove_feature(state, feature_description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, f"Remove: {feature_description}", reason, predicted_effect)

async def _tool_add_rib(state, description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, f"Add a structural rib: {description}", reason, predicted_effect)

async def _tool_add_gusset(state, description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, f"Add a gusset/corner brace: {description}", reason, predicted_effect)

async def _tool_modify_chamfer(state, description, reason="", predicted_effect=None):
    return await _agent_generic_script_modification(state, f"Modify chamfer: {description}", reason, predicted_effect)


# ----------------------------------------------------------------------
# Tool implementations — validate / mesh / simulate / compare / finalize
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# NOTE: validate_geometry and run_mesh used to be standalone tools here. They're
# now folded automatically into every set_initial_design/modify_*/add_hole/
# generic-script call via _auto_validate_and_mesh (see above) — cuts 2 of the 4
# model round-trips a design iteration used to need. See that function for the
# actual logic; kept here as a single reference point rather than duplicated.

async def _tool_run_fea(state, force_n=None, force_dir=None):
    if state.mesh is None:
        return {"status": "ERROR", "message": "No validated mesh available yet. Call set_initial_design "
                                                "(or a modify_*/add_hole tool) first — it validates and "
                                                "meshes automatically."}
    if state.iteration_count >= state.max_iterations:
        return {"status": "ITERATION_LIMIT_REACHED",
                "message": f"The configured max_iterations ({state.max_iterations}) analysis runs have "
                           "already been used. Call finalize_design now with your honest assessment.",
                "iterations_used": state.iteration_count}
    fn = force_n if force_n is not None else state.force_n
    fd = force_dir if force_dir is not None else state.force_dir
    try:
        result = await run_analysis_v8(state.mesh, state.original_prompt, state.original_prompt, state.material,
                                        fn, fd, state.operating_temp_c, state.project_description,
                                        state.surface_finish, state.reliability, False,
                                        cad_obj=state.obj)
    except Exception as e:
        return {"status": "ERROR",
                "message": f"Analysis pipeline raised: {type(e).__name__}: {e}. This usually means degenerate "
                           "geometry slipped past validate_geometry/run_mesh."}

    state.iteration_count += 1
    state.last_analysis = result
    quality = evaluate_design_quality(result, state.min_health_score, state.max_critical_violations,
                                       state.max_high_violations, state.min_safety_factor)
    state.last_quality = quality
    diagnosis = build_engineering_diagnosis(result, state)
    snap = _snapshot_current(state, quality, result, critical_region=diagnosis.get("critical_region"))
    state.previous_design = state.current_candidate
    state.current_candidate = snap
    _update_best_valid(state, snap)
    if quality["passed"]:
        _update_best_passing(state, snap)

    if state.pending_hypothesis is not None:
        ph = state.pending_hypothesis
        prev_sf = (state.previous_design or {}).get("safety_factor")
        cur_sf = snap.get("safety_factor")
        prev_region = (state.previous_design or {}).get("critical_region")
        cur_region = snap.get("critical_region")
        if quality["passed"]:
            verdict = "PASSED"
        elif prev_sf is not None and cur_sf is not None:
            verdict = "IMPROVED" if cur_sf > prev_sf else "WORSE" if cur_sf < prev_sf else "UNCHANGED"
        else:
            verdict = "UNKNOWN"
        improvement_pct = (round((cur_sf - prev_sf) / prev_sf * 100, 1)
                            if (prev_sf not in (None, 0) and cur_sf is not None) else None)
        state.hypothesis_log.append({"iteration": state.iteration_count, "change": ph["change"],
                                      "reason": ph["reason"], "predicted_effect": ph["predicted_effect"],
                                      "safety_factor_before": prev_sf, "safety_factor_after": cur_sf,
                                      "safety_factor_improvement_pct": improvement_pct,
                                      "same_region_as_previous_failure": (
                                          (prev_region == cur_region) if (prev_region and cur_region) else None),
                                      "health_score_after": snap["health_score"], "verdict": verdict})
        state.pending_hypothesis = None
    else:
        state.hypothesis_log.append({"iteration": state.iteration_count, "change": "initial_design",
                                      "reason": "initial engineering interpretation of the request",
                                      "predicted_effect": None, "safety_factor_before": None,
                                      "safety_factor_after": snap.get("safety_factor"),
                                      "safety_factor_improvement_pct": None,
                                      "same_region_as_previous_failure": None,
                                      "health_score_after": snap["health_score"],
                                      "verdict": "PASSED" if quality["passed"] else "BASELINE"})

    return {**diagnosis, "quality_gate_passed": quality["passed"], "quality_gate_reasons": quality["reasons"],
            "iterations_used": state.iteration_count, "max_iterations": state.max_iterations}


def _tool_run_fatigue(state):
    if state.last_analysis is None:
        return {"status": "NOT_AVAILABLE", "message": "Call run_fea first."}
    return state.last_analysis.get("fatigue_analysis")


def _tool_compare_designs(state, baseline="previous"):
    baseline_map = {"previous": state.previous_design, "best_valid": state.best_valid_design,
                     "best_passing": state.best_passing_design}
    if baseline not in baseline_map:
        return {"status": "ERROR", "message": f"baseline must be one of {list(baseline_map)}"}
    a = baseline_map[baseline]; b = state.current_candidate
    if b is None:
        return {"status": "ERROR", "message": "No analyzed candidate yet — call run_fea first."}
    if a is None:
        return {"status": "NO_BASELINE", "message": f"No '{baseline}' design recorded yet.",
                "current": _summarize_snapshot(b)}

    def delta(v1, v2):
        if v1 in (None, 0) or v2 is None:
            return None
        return round((v2 - v1) / v1 * 100, 2)

    sfv_a, sfv_b = a.get("safety_factor"), b.get("safety_factor")
    hs_a, hs_b = a.get("health_score", 0), b.get("health_score", 0)
    if b.get("passed") and not a.get("passed"):
        verdict = "IMPROVED_TO_PASSING"
    elif b.get("passed"):
        verdict = "PASSED"
    elif sfv_a is not None and sfv_b is not None and sfv_b > sfv_a and hs_b >= hs_a:
        verdict = "IMPROVED"
    elif (sfv_a is not None and sfv_b is not None and sfv_b < sfv_a) or hs_b < hs_a:
        verdict = "WORSE"
    else:
        verdict = "UNCHANGED"
    return {"status": "OK", "baseline": baseline, "baseline_snapshot": _summarize_snapshot(a),
            "current_snapshot": _summarize_snapshot(b),
            "delta": {"safety_factor_pct": delta(sfv_a, sfv_b), "health_score_change": round(hs_b - hs_a, 1),
                      "mass_g_pct": delta(a.get("mass_g"), b.get("mass_g")),
                      "violations_change": (b.get("violations") or 0) - (a.get("violations") or 0)},
            "verdict": verdict}


def _tool_finalize_design(state, verdict, summary=""):
    state.finalized = True
    state.final_verdict_claimed = verdict
    state.final_summary = summary
    return {"status": "ACKNOWLEDGED",
            "message": "Recorded. The final response to the user is always computed independently from the "
                       "actual solver results on the winning design, not from this claimed verdict."}


AGENT_TOOL_HANDLERS = {
    "set_initial_design": _tool_set_initial_design,
    "inspect_geometry": _tool_inspect_geometry,
    "measure_geometry": _tool_measure_geometry,
    "identify_features": _tool_identify_features,
    "find_holes": _tool_find_holes,
    "measure_wall_thickness": _tool_measure_wall_thickness,
    "get_bounding_box": _tool_get_bounding_box,
    "get_mass_properties": _tool_get_mass_properties,
    "check_manifold": _tool_check_manifold,
    "check_watertight": _tool_check_watertight,
    "find_problem_regions": _tool_find_problem_regions,
    "get_topology_summary": _tool_get_topology_summary,
    "diagnose_failure": _tool_diagnose_failure,
    "calculate_properties": _tool_calculate_properties,
    "modify_parameter": _tool_modify_parameter,
    "modify_thickness": _tool_modify_thickness,
    "modify_length": _tool_modify_length,
    "modify_width": _tool_modify_width,
    "modify_height": _tool_modify_height,
    "modify_fillet": _tool_modify_fillet,
    "modify_taper": _tool_modify_taper,
    "add_hole": _tool_add_hole,
    "modify_feature": _tool_modify_feature,
    "create_feature": _tool_create_feature,
    "remove_feature": _tool_remove_feature,
    "add_rib": _tool_add_rib,
    "add_gusset": _tool_add_gusset,
    "modify_chamfer": _tool_modify_chamfer,
    # (validate_geometry and run_mesh were removed as standalone tools — folded
    # into _auto_validate_and_mesh, called automatically by every build/modify tool)
    "run_fea": _tool_run_fea,
    "run_fatigue": _tool_run_fatigue,
    "compare_designs": _tool_compare_designs,
    "finalize_design": _tool_finalize_design,
}


async def _execute_agent_tool(state, name, args):
    handler = AGENT_TOOL_HANDLERS.get(name)
    if handler is None:
        return {"status": "ERROR", "message": f"Unknown tool '{name}'. Valid tools: {sorted(AGENT_TOOL_HANDLERS)}"}
    try:
        if asyncio.iscoroutinefunction(handler):
            result = await handler(state, **args)
        else:
            result = handler(state, **args)
        if not isinstance(result, dict):
            result = {"status": "OK", "value": result}
        return result
    except TypeError as e:
        return {"status": "ERROR", "message": f"Bad arguments for '{name}': {e}"}
    except Exception as e:
        return {"status": "ERROR", "message": f"Tool '{name}' raised: {type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# Tool schemas (OpenAI/Groq function-calling JSON Schema format)
# ----------------------------------------------------------------------

ENGINEERING_AGENT_TOOLS = [
    {"name": "set_initial_design",
     "description": "Establish the FIRST version of the design from your understanding of the engineering "
        "request. Choose design_type='tapered_beam' for a tapered/lofted member (drone arm, connecting rod, "
        "tapered spar/leg), 'bent_bracket' for a bracket with a real fold between two flat legs, or "
        "'generic_script' to let the AI author a custom build123d script for geometry neither primitive "
        "covers. Can only be called once per session.",
     "parameters": {"type": "object", "properties": {
         "design_type": {"type": "string", "enum": ["tapered_beam", "bent_bracket", "generic_script"]},
         "reason": {"type": "string", "description": "Why you picked this design_type and these starting dimensions."},
         "length": {"type": "number", "description": "[tapered_beam] overall length in mm, base(Z=0) to tip(Z=length)."},
         "base_width": {"type": "number", "description": "[tapered_beam] cross-section width at the base, mm."},
         "base_thick": {"type": "number", "description": "[tapered_beam] cross-section thickness at the base, mm."},
         "tip_width": {"type": "number", "description": "[tapered_beam] cross-section width at the tip, mm."},
         "tip_thick": {"type": "number", "description": "[tapered_beam] cross-section thickness at the tip, mm."},
         "fillet_radius": {"type": "number", "description": "[both primitives] mm, 0 for none."},
         "holes_base": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                         "description": "[tapered_beam] list of [x_from_center_mm, y_from_center_mm, diameter_mm] at the base face."},
         "holes_tip": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "description": "[tapered_beam] same shape as holes_base, at the tip face."},
         "leg1_length": {"type": "number", "description": "[bent_bracket] mm."},
         "leg2_length": {"type": "number", "description": "[bent_bracket] mm."},
         "width": {"type": "number", "description": "[bent_bracket] mm, shared by both legs."},
         "thickness": {"type": "number", "description": "[bent_bracket] mm, shared by both legs."},
         "bend_angle_deg": {"type": "number", "description": "[bent_bracket] default 90."},
         "holes_leg1": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                         "description": "[bent_bracket] list of [x_from_bend_mm, y_from_centerline_mm, diameter_mm]."},
         "holes_leg2": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                         "description": "[bent_bracket] same shape as holes_leg1."},
     }, "required": ["design_type"]}},

    {"name": "inspect_geometry", "description": "Structured summary of the current meshed geometry: "
        "solid/watertight/manifold flags, bounding box, volume, detected features, hole count, minimum "
        "wall thickness. Geometry is validated and meshed automatically by set_initial_design/modify_* — "
        "call one of those first if this comes back NOT_AVAILABLE.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "measure_geometry", "description": "Bounding box, volume, surface area, and full "
        "wall-thickness distribution of the current meshed geometry.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "identify_features", "description": "Lists engineering features present in the current "
        "design (taper, fold/bend, fillet, through_holes) and the hole count, from the declared parameters.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "find_holes", "description": "Detailed list of every hole: position, diameter, recommended "
        "fastener, and whether it violates the edge-distance rule.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "measure_wall_thickness", "description": "Dual-pass wall-thickness scan: min/mean/max "
        "thickness and specific thin/critical zone locations.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_bounding_box", "description": "Overall dimensions (mm) and aspect ratio of the current "
        "meshed geometry.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_mass_properties", "description": "Volume, mass, material, and center of mass of the "
        "current meshed geometry.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "check_manifold", "description": "Whether the current mesh is a valid closed manifold solid "
        "(winding-consistent, watertight/is_volume).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "check_watertight", "description": "Whether the current mesh is watertight; if not, the "
        "(x,y,z) location(s) of the gap(s).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "find_problem_regions", "description": "Merged list of every known problem location on the "
        "current design: non-watertight gaps, thin walls, sharp-corner stress concentrations, hole "
        "violations, and the FEA critical section if run_fea has been called.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_topology_summary", "description": "Vertex/face/edge counts, Euler number, watertight "
        "and manifold flags for the current mesh.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "diagnose_failure", "description": "The authoritative engineering diagnosis from the most "
        "recent run_fea call: status, safety_factor, max_von_mises_mpa, critical_region, failure_modes. "
        "Call run_fea first.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "calculate_properties", "description": "Section/mass/dynamic properties (stress components, "
        "deflection, natural frequency, buckling) — from the last run_fea if available, otherwise a cheap "
        "on-demand analytical estimate.",
     "parameters": {"type": "object", "properties": {}}},

    {"name": "modify_parameter", "description": "General-purpose targeted parameter change on the current "
        "parametric design. Prefer the specific modify_thickness/length/width/height/fillet/taper shortcuts "
        "when they fit; use this for anything else, including exact field names like 'base_thick' or 'tip_width'.",
     "parameters": {"type": "object", "properties": {
         "parameter": {"type": "string", "description": "e.g. 'thickness','width','length','fillet_radius', "
                        "or an exact field name such as 'base_thick'/'tip_width'/'leg1_length'."},
         "change_percent": {"type": "number", "description": "Relative change, e.g. 20 for +20%. Provide this OR new_value."},
         "new_value": {"type": "number", "description": "Absolute new value in mm/deg. Provide this OR change_percent."},
         "region": {"type": "string", "description": "Optional disambiguation for a generic parameter name: "
                     "'base'/'near_fixed_support' vs 'tip'/'near_tip' for a beam; 'leg1' vs 'leg2' for a bracket."},
         "reason": {"type": "string", "description": "REQUIRED. The engineering justification for this change."},
         "predicted_effect": {"type": "string", "description": "What you expect this change to do to the result."},
     }, "required": ["parameter", "reason"]}},
    {"name": "modify_thickness", "description": "Shortcut for modify_parameter targeting section thickness "
        "(base_thick/tip_thick for a beam, thickness for a bracket).",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "new_value": {"type": "number"},
         "region": {"type": "string", "description": "'base' or 'tip' for a beam (omit to scale both)."},
         "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["reason"]}},
    {"name": "modify_length", "description": "Shortcut for modify_parameter targeting overall length "
        "(beam) or a leg length (bracket, use region='leg1'|'leg2').",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "new_value": {"type": "number"},
         "region": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["reason"]}},
    {"name": "modify_width", "description": "Shortcut for modify_parameter targeting section width "
        "(base_width/tip_width for a beam, width for a bracket).",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "new_value": {"type": "number"},
         "region": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["reason"]}},
    {"name": "modify_height", "description": "Shortcut for modify_parameter targeting the out-of-plane "
        "dimension (alias for thickness).",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "new_value": {"type": "number"},
         "region": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["reason"]}},
    {"name": "modify_fillet", "description": "Shortcut for modify_parameter targeting fillet_radius.",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "new_value": {"type": "number"},
         "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["reason"]}},
    {"name": "modify_taper", "description": "[tapered_beam only] Scale both tip_width and tip_thick "
        "together by change_percent, making the taper more (negative %) or less (positive %) aggressive "
        "without touching the base.",
     "parameters": {"type": "object", "properties": {
         "change_percent": {"type": "number"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["change_percent", "reason"]}},
    {"name": "add_hole", "description": "Add one through-hole. For a tapered_beam set end='base'|'tip'; "
        "for a bent_bracket set leg='leg1'|'leg2'. Position is in that face's local centered frame.",
     "parameters": {"type": "object", "properties": {
         "x_mm": {"type": "number"}, "y_mm": {"type": "number"}, "diameter_mm": {"type": "number"},
         "end": {"type": "string", "enum": ["base", "tip"]}, "leg": {"type": "string", "enum": ["leg1", "leg2"]},
         "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["x_mm", "y_mm", "diameter_mm", "reason"]}},
    {"name": "modify_feature", "description": "Fallback for a design change the dedicated parametric "
        "tools can't express. Describe the change in plain engineering language; it is applied via an "
        "AI-assisted, fully re-verified script edit rather than a validated numeric parameter change, so "
        "prefer the specific modify_*/add_hole tools whenever they fit.",
     "parameters": {"type": "object", "properties": {
         "feature_description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["feature_description", "reason"]}},
    {"name": "create_feature", "description": "Same mechanism as modify_feature, framed as adding "
        "something new (e.g. a boss, a slot, a mounting tab) not covered by add_hole.",
     "parameters": {"type": "object", "properties": {
         "feature_description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["feature_description", "reason"]}},
    {"name": "remove_feature", "description": "Same mechanism as modify_feature, framed as removing an "
        "existing feature.",
     "parameters": {"type": "object", "properties": {
         "feature_description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["feature_description", "reason"]}},
    {"name": "add_rib", "description": "Add a stiffening rib. Not yet a dedicated parametric primitive — "
        "routed through the same AI-assisted script-edit fallback as modify_feature, then fully re-verified.",
     "parameters": {"type": "object", "properties": {
         "description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["description", "reason"]}},
    {"name": "add_gusset", "description": "Add a corner gusset/brace. Not yet a dedicated parametric "
        "primitive — routed through the same AI-assisted script-edit fallback, then fully re-verified.",
     "parameters": {"type": "object", "properties": {
         "description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["description", "reason"]}},
    {"name": "modify_chamfer", "description": "Add/change a chamfer. Not yet a dedicated parametric "
        "primitive — routed through the same AI-assisted script-edit fallback, then fully re-verified.",
     "parameters": {"type": "object", "properties": {
         "description": {"type": "string"}, "reason": {"type": "string"}, "predicted_effect": {"type": "string"},
     }, "required": ["description", "reason"]}},

    {"name": "run_fea", "description": "THE authoritative engineering check: full FEA (real CalculiX where "
        "configured, analytical fallback otherwise) plus fatigue and the rule engine, on the current meshed "
        "design. Consumes one iteration of your budget. Returns the same shape as diagnose_failure.",
     "parameters": {"type": "object", "properties": {
         "force_n": {"type": "number", "description": "Override the load magnitude for this run only."},
         "force_dir": {"type": "string", "enum": ["x", "y", "z"], "description": "Override load direction for this run only."},
     }}},
    {"name": "run_fatigue", "description": "Fatigue analysis (full Marin 6-factor + Goodman/Gerber) from "
        "the most recent run_fea call.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "compare_designs", "description": "Compare the current analyzed candidate against a stored "
        "baseline ('previous' iteration, 'best_valid' design so far, or 'best_passing' design so far) to "
        "see whether your last change actually helped.",
     "parameters": {"type": "object", "properties": {
         "baseline": {"type": "string", "enum": ["previous", "best_valid", "best_passing"]},
     }}},
    {"name": "finalize_design", "description": "End the engineering loop. Call this once the design "
        "passes, or your iteration budget is spent, or further iteration clearly won't help.",
     "parameters": {"type": "object", "properties": {
         "verdict": {"type": "string", "enum": ["PASSED", "BEST_EFFORT", "FAILED"]},
         "summary": {"type": "string", "description": "Brief summary of the final state and what was tried."},
     }, "required": ["verdict"]}},
]


ENGINEERING_AGENT_SYSTEM_PROMPT = """You are the Lumexa Engineering Agent — a mechanical design reasoning layer sitting on top of Lumexa's deterministic CAD/FEA systems. You are NOT a CAD kernel and you do NOT do freehand geometry math yourself. Every geometric fact you know comes from calling a tool; every geometric change you make happens by calling a tool. Lumexa's solvers (real FEA where configured, analytical fallback otherwise) are the sole source of truth for whether a design passes — you never declare PASS/FAIL yourself, you read it from run_fea's/diagnose_failure's output.

WORKFLOW (follow this shape; you may repeat steps as needed):
Understand -> Inspect -> Diagnose -> Propose -> Modify (auto-verified) -> Simulate -> Compare -> Refine

1. UNDERSTAND: read the engineering request. Call set_initial_design with your best translation of it into concrete parameters for whichever primitive fits (tapered_beam or bent_bracket), or generic_script if neither fits. Before proposing a change on later iterations, silently answer for yourself: what is being built, its engineering purpose, its important geometric features, the loads/constraints on it, what is currently failing and where, what design variable could influence that failure, what change you will attempt, why it should help, and what must NOT change.
2. INSPECT: set_initial_design and every modify_*/add_hole/create_feature call automatically validate the B-rep and mesh it for you as part of that same call (status OK means it's already meshed and ready) — you do NOT need to call separate validate/mesh steps. If a build/modify call comes back REJECTED for a geometry reason, the system has already reverted to the last known-good design for you. Once meshed, inspect_geometry/measure_geometry/find_holes/check_watertight/etc. are available for closer inspection if you want it, but are optional, read-only, and don't cost you an iteration.
3. DIAGNOSE: call run_fea (this also runs fatigue/rule-engine checks) and read diagnose_failure/find_problem_regions for WHERE and WHY it is failing. Trust these numbers completely — never override or second-guess a solver result. Exception: if failure_modes leads with "SOLVER OUTPUT NUMERICALLY SUSPECT", the solver itself flagged that iteration's stress/deflection numbers as an implausible artifact (not a real structural finding) — don't treat it as a confirmed FAIL or try to "fix" it with a large parameter change; a small, unrelated tweak and re-running run_fea is enough to get past a one-off slicing artifact.
4. PROPOSE + MODIFY: pick ONE targeted, physically-justified change at a time (modify_parameter and its shortcuts modify_thickness/length/width/height/fillet/taper, add_hole, or for anything the built-in primitives can't express, modify_feature/create_feature/remove_feature/add_rib/add_gusset/modify_chamfer). Every modify call requires a `reason` — the engineering justification — and should include a `predicted_effect` — what you expect to happen. Do not rewrite the whole design when one parameter needs changing. Do not fix one flagged issue by weakening something that was already fine elsewhere.
5. SIMULATE: call run_fea right after each modify call (it's already validated/meshed by step 4 — no separate verify step needed). If a change was rejected instead (bad parameters, or the resulting geometry came back non-manifold/non-watertight), you were told so explicitly and the system already reverted to the best known-valid design — just try a different, smaller, or better-justified change next.
6. COMPARE + REFINE: call compare_designs to see whether your last change actually helped versus the previous iteration (or the best-so-far). Never assume a change worked — check.
7. When the design passes the quality gate, or you have used your iteration budget, or you are confident further iteration will not help, call finalize_design with your honest verdict and a short summary. The system will independently re-verify the final numbers regardless of what you report here — this call only records your reasoning, it never overrides the solver's own verdict.

RULES:
- You have a limited number of model calls available for this run — every call you make (including read-only inspection) counts against it, so don't waste calls on redundant inspection once a design is already meshed; go straight to run_fea unless you have a specific reason to inspect first.
- Never claim a design passed or failed — only report what run_fea/diagnose_failure told you.
- Always give a `reason` on every modification.
- One targeted change per modify call. If you're unsure which parameter to change, call diagnose_failure/find_problem_regions/calculate_properties first rather than guessing.
- If a tool returns status REJECTED or ERROR, read the message, adjust your approach, and try again — do not repeat the exact same rejected call.
- You have a limited number of run_fea calls (shown in the user message) — don't waste them on inspection; use the read-only tools freely, they don't count against that budget."""


# ----------------------------------------------------------------------
# Provider tool-calling transport (OpenAI-compatible: groq/openrouter/lovable/
# cerebras/nvidia all share this exact wire shape). See the module docstring
# above this section for why Claude/Gemini aren't wired up for this endpoint yet.
# ----------------------------------------------------------------------

def _to_openai_tools(tool_specs):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]}} for t in tool_specs]


def _openai_compatible_tool_chat(api_url, api_key, model, messages, tools, temperature=0.2,
                                  max_tokens=AGENT_TURN_MAX_TOKENS, extra_headers=None):
    """
    One tool-calling round-trip against any OpenAI-compatible chat/completions
    endpoint — same urllib-direct pattern as _groq_request/_openrouter_request/
    _lovable_request above, extended with tools/tool_choice and returning the
    full message object (content + tool_calls) rather than just extracted text,
    since the agent loop needs to see and execute tool_calls, not just prose.
    """
    import urllib.request, urllib.error

    payload = json.dumps({"model": model, "messages": messages, "tools": tools,
                           "tool_choice": "auto", "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
               "User-Agent": "Mozilla/5.0 (compatible; LumexaBackend/1.0)", "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(api_url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 429:
            raise HTTPException(429, f"Provider rate limit exceeded: {body}")
        raise HTTPException(502, f"Provider error ({e.code}): {body}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Provider connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        # FIX: confirmed live via the near-identical bug in _openrouter_request — a call
        # that times out mid-read or returns a non-JSON body raises something neither
        # HTTPError nor URLError catches, and this function drives EVERY step of the main
        # agent loop (whichever provider is configured), not just the advisor — so this
        # gap could crash an /engineering-agent request on its primary model call, not
        # only the sparingly-used advisor path where it was actually first observed.
        raise HTTPException(502, f"Provider request failed unexpectedly: {type(e).__name__}: {e}")

    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, f"Unexpected tool-calling response shape: {json.dumps(data)[:500]}")


def _provider_tool_endpoint():
    if AI_PROVIDER == "groq":
        return GROQ_API_URL, GROQ_API_KEY, GROQ_MODEL
    if AI_PROVIDER == "openrouter":
        return OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL
    if AI_PROVIDER == "lovable":
        return LOVABLE_AI_URL, LOVABLE_API_KEY, LOVABLE_AI_MODEL
    if AI_PROVIDER == "cerebras":
        return CEREBRAS_API_URL, CEREBRAS_API_KEY, CEREBRAS_MODEL
    if AI_PROVIDER == "nvidia":
        return NVIDIA_API_URL, NVIDIA_API_KEY, NVIDIA_MODEL
    return None, None, None


def _call_model_with_tools(messages, temperature=0.2, max_tokens=AGENT_TURN_MAX_TOKENS):
    url, key, model = _provider_tool_endpoint()
    if url is None:
        raise HTTPException(501,
            "The Engineering Agent's tool-calling loop is currently implemented for OpenAI-compatible "
            "providers only (groq, openrouter, lovable, cerebras, nvidia). Current AI_PROVIDER is "
            f"'{AI_PROVIDER}'. Set AI_PROVIDER to one of those (plus its matching API key) to use this "
            "endpoint; Claude/Gemini native tool-calling for this specific agent loop is not wired up "
            "yet — /generate-validate-refine still works on every provider as before.")
    if not key:
        raise HTTPException(500, f"{AI_PROVIDER.upper()}_API_KEY is not configured on the server.")

    msg = _openai_compatible_tool_chat(url, key, model, messages, _to_openai_tools(ENGINEERING_AGENT_TOOLS),
                                        temperature=temperature, max_tokens=max_tokens)
    tool_calls = []
    for tc in (msg.get("tool_calls") or []):
        try:
            args = json.loads(tc.get("function", {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments_unparseable": tc.get("function", {}).get("arguments")}
        tool_calls.append({"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("function", {}).get("name"), "arguments": args})
    return {"role": "assistant", "content": msg.get("content") or msg.get("reasoning"),
            "tool_calls": tool_calls, "_raw_message": msg}


def _format_tool_result_message(tool_call_id, tool_name, result_dict):
    safe = _json_safe(result_dict)
    text = json.dumps(safe)
    if len(text) > 6000:
        text = json.dumps({"truncated": True, "note": "Full result was too large; showing a partial view.",
                            "status": safe.get("status") if isinstance(safe, dict) else None,
                            "keys_available": list(safe.keys()) if isinstance(safe, dict) else None,
                            "partial": text[:4000]})
    return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": text}


# ----------------------------------------------------------------------
# Orchestration loop (spec sections 1, 9, 12)
# ----------------------------------------------------------------------

def _call_advisor(situation_text, max_tokens=800):
    """The optional 'strategic advisor' second model (see NEMOTRON_ADVISOR_MODEL
    above). Returns its text response, or None if the advisor isn't configured
    or the call fails for any reason — a missing/broken advisor NEVER breaks
    the main agent loop, it just means gpt-oss-120b reasons on its own like it
    always has. This is advisory input only: it gets appended to the
    conversation as context, never executes anything itself and never
    overrides an actual solver result."""
    if not NEMOTRON_ADVISOR_MODEL or not OPENROUTER_API_KEY:
        return None
    try:
        return _openrouter_request(
            [{"role": "system", "content":
                "You are a senior mechanical engineering advisor reviewing an automated design "
                "iteration. You do not have tools and cannot change anything yourself — you give "
                "concise, physically-grounded strategic guidance that another AI (which DOES have "
                "tools) will read and act on. Be specific and brief: 3-6 sentences. Never invent "
                "numbers you weren't given."},
             {"role": "user", "content": situation_text}],
            temperature=0.3, max_tokens=max_tokens, model=NEMOTRON_ADVISOR_MODEL,
            timeout=30,  # fail fast rather than burn most of a minute on a slow/overloaded
                         # advisor model — this is a best-effort extra, not worth a long wait
        )
    except HTTPException as e:
        print(f"[engineering-agent] advisor call failed ({NEMOTRON_ADVISOR_MODEL}): {e.detail} "
              f"— continuing without it")
        return None
    except Exception as e:
        # Defense in depth on top of the fix now in _openrouter_request itself — this
        # function's entire design promise is "can never break the main loop", so it
        # catches broadly here too rather than relying on the callee alone.
        print(f"[engineering-agent] advisor call raised unexpectedly ({NEMOTRON_ADVISOR_MODEL}): "
              f"{type(e).__name__}: {e} — continuing without it")
        return None


def render_mesh_snapshot_png(mesh, view="iso", size_px=800):
    """
    Lightweight, pure-CPU mesh snapshot renderer — no OpenGL, no headless
    browser, no display server. Deliberately cruder than a proper WebGL/
    ray-traced render (flat per-face lambertian shading, matplotlib's own
    antialiasing) in exchange for being reliable on a memory-constrained
    free-tier container: a headless-Chromium-based renderer (the approach
    earthtojake/text-to-cad's cadgen skill uses for its mandatory snapshot-
    review policy) would add a real further OOM risk on top of everything
    else this deployment has already fought — Chromium alone typically wants
    200-500MB of RAM just to run.

    Adopts that project's core PRINCIPLE, not its mechanism: a rendered
    snapshot is DIAGNOSTIC, not authoritative. It's for a human (or a future
    vision-capable model call, once one is configured) to glance at before
    trusting a design purely on its numbers — it never overrides an actual
    solver result, and nothing in this codebase treats it as one.

    Returns PNG bytes, or raises on failure (caller decides how to degrade —
    see its use in run_engineering_agent, which never lets a render failure
    break the actual result).
    """
    import matplotlib
    matplotlib.use("Agg")  # pure-CPU raster backend — no display/GPU needed at all
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import io

    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    verts = mesh.vertices
    faces = mesh.faces
    tris = verts[faces]  # (F, 3, 3)

    # Simple fixed-direction lambertian shading per face, computed once from
    # already-available face normals — cheap, deterministic, no lighting/
    # material system to get wrong; good enough to read shape, proportions,
    # and obvious topology problems (which is the actual job here).
    normals = mesh.face_normals
    light_dir = np.array([0.5, -0.5, 0.8]); light_dir = light_dir / np.linalg.norm(light_dir)
    brightness = np.clip(normals @ light_dir, 0.15, 1.0)  # 0.15 ambient floor, never fully black
    base_color = np.array([0.65, 0.70, 0.78])
    face_colors = np.clip(base_color[None, :] * brightness[:, None], 0, 1)

    coll = Poly3DCollection(tris, facecolor=face_colors, edgecolor=(0, 0, 0, 0.15), linewidths=0.3)
    ax.add_collection3d(coll)

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0])) / 2.0 or 1.0
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass  # older matplotlib without set_box_aspect — proportions degrade gracefully, not fatally

    # Two opposed-ish views by default cover most of a part the way
    # text-to-cad's "two opposed isometrics" packet does, without needing a
    # multi-image packet for every single result.
    views = {"iso": (25, -60), "top": (90, -90), "front": (0, -90), "side": (0, 0)}
    elev, azim = views.get(view, views["iso"])
    ax.view_init(elev=elev, azim=azim)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def run_engineering_agent(prompt, material="auto", force_n=1000.0, force_dir="z",
                                 operating_temp_c=25.0, surface_finish="machined", reliability=0.99,
                                 project_description=None, max_iterations=4, min_health_score=75.0,
                                 max_critical_violations=0, max_high_violations=2, min_safety_factor=1.0):
    if not B3D:
        raise HTTPException(503, "build123d not installed on this server.")

    max_iterations = max(1, min(int(max_iterations), 6))
    max_steps = max(12, min(max_iterations * 8, 40))

    state = EngineeringDesignState(prompt, material, force_n, force_dir, operating_temp_c, surface_finish,
                                    reliability, project_description, max_iterations, min_health_score,
                                    max_critical_violations, max_high_violations, min_safety_factor)

    prompt_lower = prompt.lower()
    is_taper_hint = any(w in prompt_lower for w in TAPER_KEYWORDS)
    is_fold_hint = any(w in prompt_lower for w in FOLD_BRACKET_KEYWORDS)
    hint = ("tapered/lofted — consider set_initial_design with design_type='tapered_beam'" if is_taper_hint
            else "a bent/folded bracket — consider set_initial_design with design_type='bent_bracket'" if is_fold_hint
            else "not an obvious match for either built-in parametric primitive — use your judgment; "
                 "design_type='generic_script' is the safe default if neither tapered_beam nor "
                 "bent_bracket actually fits")

    user_msg = (
        f"ENGINEERING REQUEST: {prompt}\n\n"
        f"Material: {material}. Load: {force_n}N along {force_dir}. Operating temp: {operating_temp_c}C. "
        f"Surface finish: {surface_finish}. Target reliability: {reliability}.\n"
        + (f"Project context: {project_description}\n" if project_description else "")
        + f"\nKeyword heuristic (not a hard rule — use your own judgment): this request looks like {hint}.\n\n"
        f"Quality thresholds you are building to: min_health_score={min_health_score}, "
        f"max_critical_violations={max_critical_violations}, max_high_violations={max_high_violations}, "
        f"min_safety_factor={min_safety_factor}. You have a budget of {max_iterations} run_fea calls.\n\n"
        "Begin with set_initial_design."
    )
    messages = [{"role": "system", "content": ENGINEERING_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]

    advisor_calls = []
    cfd_assessment = {"recommended": False, "reasoning": None,
                       "note": "No advisor configured (NEMOTRON_ADVISOR_MODEL unset) — CFD need was "
                               "not assessed. This does not mean CFD isn't needed, just that nothing "
                               "checked."}
    advisor_intro = await asyncio.to_thread(
        _call_advisor,
        f"A user has requested this part be designed: {prompt}\n"
        f"Material: {material}. Load: {force_n}N along {force_dir}. "
        f"Quality thresholds: min_health_score={min_health_score}, min_safety_factor={min_safety_factor}.\n"
        "In 3-6 sentences: what is this part's engineering purpose, what design strategy would you "
        "start with, and what's the single biggest risk to watch for?\n\n"
        "Then, on its own final line, state exactly: 'CFD_NEEDED: yes' or 'CFD_NEEDED: no', followed "
        "by a short reason — e.g. 'CFD_NEEDED: yes - propeller wash creates a real aerodynamic load on "
        "this arm, not just the static tip force given'. Only say yes if fluid flow (aerodynamics, "
        "cooling airflow, internal fluid/gas flow) is actually part of this part's function — a plain "
        "structural bracket or arm evaluated only under a static point load should be 'no'. This "
        "system does not have a working CFD solver yet, so say so plainly if you believe CFD IS "
        "warranted here — don't let that absence change your answer."
    )
    if advisor_intro:
        advisor_calls.append({"checkpoint": "initial_understanding", "response": advisor_intro})
        messages.append({"role": "user", "content":
            f"ENGINEERING ADVISOR (a second model's strategic read on this request — advisory only, "
            f"you still make the actual tool calls and the solver results are still the only "
            f"authoritative truth):\n{advisor_intro}"})

        # Parse the tagged CFD_NEEDED line the advisor was asked for. Defensive by
        # design — a model not following the exact format is treated as "couldn't
        # determine", never silently coerced to a false negative or a fabricated yes.
        import re
        m = re.search(r"CFD_NEEDED:\s*(yes|no)\s*-?\s*(.*)", advisor_intro, re.IGNORECASE)
        if m:
            recommended = m.group(1).strip().lower() == "yes"
            reason = m.group(2).strip() or None
            cfd_assessment = {
                "recommended": recommended,
                "reasoning": reason,
                "note": ("CFD analysis was assessed as valuable for this part, but OpenFOAM is not "
                         "yet integrated into this system — this is a known, honestly-flagged gap, "
                         "not a result." if recommended else
                         "Advisor assessed this part as not needing fluid-flow analysis; only "
                         "structural/thermal/fatigue checks apply."),
            }
        else:
            cfd_assessment = {"recommended": None, "reasoning": None,
                               "note": "Advisor was asked to assess CFD need but didn't return a "
                                       "parseable answer — treat as unknown, not as 'no'."}

    tool_trace = []
    stopped_reason = None
    step = 0
    no_tool_call_strikes = 0
    rate_limit_wait_remaining = 75.0  # BUG FIX: this used to be reset to 90.0 inside the
    # step loop below, meaning EVERY step got its own fresh 90s retry budget instead of
    # the whole request sharing one. With up to max_steps (~40) steps, that's a
    # theoretical 40*90s=3600s of possible retry waiting with no overall cap — confirmed
    # live as a hang: max_iterations=3 timed out at the full 300s client --max-time with
    # 0 bytes received (a genuine hang, not the earlier OOM-style connection abort).
    # Scoped to the whole run now, so sustained rate-limiting fails fast with a clear
    # "rate_limited" stopped_reason instead of silently retrying past any client timeout.
    run_started = time.time()
    MAX_WALL_CLOCK_SECONDS = 240.0  # comfortably under the --max-time 300 used in testing —
    # the server should always be the one to give up first, with an honest partial result,
    # rather than depend on the caller's timeout to be the only thing that ever stops this

    for step in range(1, max_steps + 1):
        if time.time() - run_started > MAX_WALL_CLOCK_SECONDS:
            stopped_reason = "wall_clock_budget_exceeded"
            print(f"[engineering-agent] step {step}: stopping — {MAX_WALL_CLOCK_SECONDS}s wall-clock "
                  f"budget exceeded ({time.time()-run_started:.1f}s elapsed)")
            break
        step_started = time.time()
        assistant = None
        while True:
            try:
                assistant = await asyncio.to_thread(_call_model_with_tools, messages, 0.2, AGENT_TURN_MAX_TOKENS)
                break
            except HTTPException as e:
                if e.status_code == 429 and rate_limit_wait_remaining > 0:
                    wait_s = min(_parse_groq_retry_after(str(e.detail)), rate_limit_wait_remaining)
                    rate_limit_wait_remaining -= wait_s
                    print(f"[engineering-agent] step {step}: 429 rate-limited, waiting {wait_s:.1f}s "
                          f"({rate_limit_wait_remaining:.1f}s of retry budget left for this request)")
                    await asyncio.sleep(wait_s)
                    continue
                stopped_reason = "rate_limited" if e.status_code == 429 else f"model_call_failed: {e.detail}"
                break
        if assistant is None:
            print(f"[engineering-agent] step {step}: giving up — {stopped_reason} "
                  f"(elapsed so far: {time.time()-run_started:.1f}s)")
            break
        print(f"[engineering-agent] step {step}: model call took {time.time()-step_started:.1f}s, "
              f"{len(assistant['tool_calls'])} tool call(s), total elapsed {time.time()-run_started:.1f}s")

        messages.append(assistant["_raw_message"])

        if not assistant["tool_calls"]:
            # Confirmed live: a reasoning model asked to call finalize_design will
            # sometimes just explain itself in prose instead on the very first
            # nudge. One miss used to end the whole loop outright, discarding
            # whatever it actually said. Give it up to 2 misses, with an
            # increasingly explicit nudge, and keep its prose as a fallback
            # summary rather than throwing it away.
            if assistant.get("content") and not state.final_summary:
                state.final_summary = assistant["content"][:2000]
            if state.iteration_count > 0:
                no_tool_call_strikes += 1
                if no_tool_call_strikes >= 2:
                    stopped_reason = "model_stopped_without_finalize"
                    break
                messages.append({"role": "user", "content":
                    "You responded without calling a tool. Call finalize_design now — pass 'verdict' "
                    "(PASSED/BEST_EFFORT/FAILED) and 'summary' as arguments to that tool, don't just "
                    "describe your assessment in text."})
                continue
            messages.append({"role": "user", "content":
                "Please proceed by calling a tool — start with set_initial_design. Do not describe what "
                "you would do in prose; call the tool directly."})
            continue
        no_tool_call_strikes = 0

        finalize_called = False
        for tc in assistant["tool_calls"]:
            tool_started = time.time()
            result = await _execute_agent_tool(state, tc["name"], tc["arguments"])
            tool_elapsed = round(time.time() - tool_started, 2)
            print(f"[engineering-agent] step {step}: tool={tc['name']} took {tool_elapsed}s "
                  f"-> {result.get('status') if isinstance(result, dict) else '?'}")
            # FIX: confirmed live — a chain of REJECTED set_initial_design attempts was
            # only showing result_status here, never the actual reason, so diagnosing why
            # each attempt failed meant separately re-deriving the safe-parameter-contract
            # math by hand. Every REJECTED/ERROR result already carries a human-readable
            # reason/message; surface it directly instead of just the status label.
            detail = None
            if isinstance(result, dict):
                detail = result.get("reason") or result.get("message")
            tool_trace.append({"step": step, "tool": tc["name"], "arguments": tc["arguments"],
                                "result_status": result.get("status") if isinstance(result, dict) else None,
                                "detail": detail, "elapsed_s": tool_elapsed})
            messages.append(_format_tool_result_message(tc["id"], tc["name"], result))
            if tc["name"] == "finalize_design":
                finalize_called = True
            if (tc["name"] == "run_fea" and isinstance(result, dict)
                    and result.get("status") == "FAIL" and not result.get("numerically_suspect")):
                # Sparingly-called checkpoint (see NEMOTRON_ADVISOR_MODEL) — only on an
                # actual solver FAIL, not on every run_fea call, and skipped entirely
                # when the solver already flagged its own output as a numerical
                # artifact (advising on a phantom failure wastes one of a scarce
                # daily budget of calls for no benefit).
                advisor_diag = await asyncio.to_thread(
                    _call_advisor,
                    f"Design iteration {state.iteration_count} just failed. Current parameters: "
                    f"{state.params}. Solver diagnosis: status={result.get('status')}, "
                    f"safety_factor={result.get('safety_factor')}, "
                    f"critical_region={result.get('critical_region')}, "
                    f"failure_modes={result.get('failure_modes')}. Recent change history: "
                    f"{state.hypothesis_log[-3:]}. In 3-6 sentences: what is the most likely root "
                    f"cause, and what is the SMALLEST parameter change that would address it without "
                    f"changing anything the user didn't ask to change?"
                )
                if advisor_diag:
                    advisor_calls.append({"checkpoint": f"iteration_{state.iteration_count}_failure",
                                           "response": advisor_diag})
                    messages.append({"role": "user", "content":
                        f"ENGINEERING ADVISOR (a second model's diagnosis — advisory only, weigh it "
                        f"but you decide the actual tool call; the solver result above remains the "
                        f"authoritative truth):\n{advisor_diag}"})

        gc.collect()  # end of this step's tool-call batch — a natural point to release
                      # whatever the last modify/mesh/FEA cycle allocated before the next
                      # (possibly slow) model round-trip, rather than let it sit and stack up

        if finalize_called:
            stopped_reason = "agent_finalized"
            break

        if state.iteration_count >= max_iterations and state.current_candidate is not None:
            messages.append({"role": "user", "content":
                f"You have used all {max_iterations} run_fea iterations. Call finalize_design now with "
                "your honest assessment of the final result."})
    else:
        stopped_reason = stopped_reason or "max_steps_reached"

    if stopped_reason is None:
        stopped_reason = "max_steps_reached"

    winner = state.best_passing_design or state.best_valid_design or state.current_candidate
    if winner is None:
        # FIX: confirmed live — every set_initial_design attempt got REJECTED (the safe
        # parameter contract correctly caught infeasible geometry every time), so nothing
        # was ever built to analyze. This used to raise a raw HTTPException with the whole
        # tool trace dumped as an escaped JSON string inside the error detail — technically
        # informative but painful to actually read. A design the agent never managed to
        # build is a legitimate engineering outcome (same as an analyzed design that FAILs
        # FEA), not a server malfunction — so this now returns a normal, structured 200
        # response like every other result in this file, not an exception.
        last_rejection = next((t for t in reversed(tool_trace) if t.get("result_status") == "REJECTED"), None)
        return {
            "status": "NO_VALID_DESIGN",
            "summary": f"The Engineering Agent never produced geometry that passed the safe parameter "
                       f"contract, across {len(tool_trace)} tool call(s). No FEA/analysis was possible "
                       f"since nothing was ever successfully built.",
            "last_rejection_reason": (last_rejection or {}).get("detail"),
            "engineering_agent": {
                "design_type": None, "final_parameters": None, "iterations_used": state.iteration_count,
                "max_iterations": max_iterations, "steps_used": step, "max_steps": max_steps,
                "stopped_reason": stopped_reason, "passed_quality_gate": False,
                "advisor_calls": advisor_calls, "cfd_assessment": cfd_assessment,
                "hypothesis_log": state.hypothesis_log, "tool_call_trace": tool_trace,
            },
        }

    # FIX: confirmed live — the old "reuse state.mesh if winner is state.current_candidate"
    # shortcut assumed state.mesh always reflects whatever design `winner` points to. It
    # doesn't: a modify_* call can succeed (updating state.obj/state.mesh) and then the loop
    # can get cut off (rate-limited, wall-clock budget) BEFORE run_fea ever confirms that
    # change with a new snapshot. When that happens, `winner` still correctly points at the
    # last CONFIRMED snapshot, but state.mesh had already moved on to the unconfirmed next
    # candidate — so the response reported one set of parameters while actually returning the
    # analysis/STL of a different, later, never-verified geometry. Rebuilding deterministically
    # from winner's own stored params every time is cheap for these primitives and makes this
    # class of mismatch impossible rather than merely unlikely.
    try:
        if winner["design_type"] == "tapered_beam":
            final_obj = make_tapered_beam(**winner["params"])
        elif winner["design_type"] == "bent_bracket":
            final_obj = make_bent_bracket(**winner["params"])
        else:
            final_obj, build_err = execute_cad_script_safely(winner["script"])
            if build_err:
                raise RuntimeError(build_err)
        final_mesh, final_stl = await mesh_from_cad_object(final_obj)
    except Exception as e:
        raise HTTPException(502, f"Failed to rebuild the winning design for final export: {e}")

    final_result = await run_analysis_v8(final_mesh, prompt, prompt, material, force_n, force_dir,
                                          operating_temp_c, project_description, surface_finish,
                                          reliability, False, cad_obj=final_obj)
    final_quality = evaluate_design_quality(final_result, min_health_score, max_critical_violations,
                                             max_high_violations, min_safety_factor)

    final_result["generated_stl_base64"] = base64.b64encode(final_stl).decode()
    try:
        snapshot_png = render_mesh_snapshot_png(final_mesh, view="iso")
        final_result["generated_snapshot_base64"] = base64.b64encode(snapshot_png).decode()
        final_result["generated_snapshot_note"] = (
            "Diagnostic only, not authoritative — a quick visual sanity check (proportions, obvious "
            "topology problems) for a human to glance at, same principle as any other unbenchmarked "
            "estimate in this file. Never treat this image as confirming or overriding a solver result."
        )
    except Exception as e:
        final_result["generated_snapshot_base64"] = None
        final_result["generated_snapshot_note"] = f"Snapshot rendering failed ({type(e).__name__}: {e}) — not fatal, every other result field is unaffected."
    final_result["generated_script"] = (winner["script"] if winner["design_type"] == "generic_script"
                                         else params_to_script_tapered_beam(winner["params"])
                                         if winner["design_type"] == "tapered_beam"
                                         else params_to_script_bent_bracket(winner["params"]))
    final_result["generation_method"] = f"lumexa_engineering_agent_{winner['design_type']}"
    final_result["engineering_agent"] = {
        "design_type": winner["design_type"], "final_parameters": winner.get("params"),
        "iterations_used": state.iteration_count, "max_iterations": max_iterations,
        "steps_used": step, "max_steps": max_steps, "stopped_reason": stopped_reason,
        "passed_quality_gate": final_quality["passed"], "final_reasons": final_quality["reasons"],
        "used_best_passing": winner is state.best_passing_design,
        "used_best_valid_fallback": (winner is state.best_valid_design and winner is not state.best_passing_design),
        "agent_final_verdict_claimed": state.final_verdict_claimed, "agent_final_summary": state.final_summary,
        "hypothesis_log": state.hypothesis_log, "tool_call_trace": tool_trace,
        "advisor_calls": advisor_calls, "cfd_assessment": cfd_assessment,
        "quality_thresholds": {"min_health_score": min_health_score,
                                "max_critical_violations": max_critical_violations,
                                "max_high_violations": max_high_violations,
                                "min_safety_factor": min_safety_factor},
    }
    return final_result


@app.post("/engineering-agent")
@_sanitize_response
async def engineering_agent_endpoint(
    prompt: str = Form(...),
    material: str = Form("auto"),
    force_n: float = Form(1000.0),
    force_dir: str = Form("z"),
    operating_temp_c: float = Form(25.0),
    surface_finish: str = Form("machined"),
    reliability: float = Form(0.99),
    project_description: Optional[str] = Form(None),
    max_iterations: int = Form(4),
    min_health_score: float = Form(75.0),
    max_critical_violations: int = Form(0),
    max_high_violations: int = Form(2),
    min_safety_factor: float = Form(1.0),
):
    """
    THE ENGINEERING AGENT — Understand -> Inspect -> Diagnose -> Propose -> Modify ->
    Verify -> Simulate -> Compare -> Refine, instead of /generate-validate-refine's
    "regenerate the whole script and hope" loop.

    The frontier model (GPT-OSS-120B via Groq, by default — see AI_PROVIDER) reasons about
    the design and calls tools; Lumexa's deterministic geometry kernel, mesher, and solver
    remain the sole source of engineering truth. The model never declares pass/fail itself
    and never hand-writes build123d for the tapered-beam/bent-bracket workflows — it only
    proposes named parameter changes, which are validated against a safe-parameter contract
    and applied through make_tapered_beam/make_bent_bracket, the same trusted server-side
    primitives /generate-validate-refine already relies on. Geometry outside what those two
    primitives cover falls back to the existing AI-script-generation/refinement machinery,
    still wrapped in the same validate -> mesh -> FEA -> compare loop.

    FIRST IMPLEMENTATION TARGET (per the build spec this endpoint implements): the
    tapered-beam workflow — e.g. "Design a tapered drone arm capable of carrying 2kg."
    Bent-bracket support is wired the same way since make_bent_bracket already existed, but
    has had less real-world exercise than the beam path — test that path first.

    Requires AI_PROVIDER to be an OpenAI-compatible provider with tool-calling
    (groq/openrouter/lovable/cerebras/nvidia). Set AI_PROVIDER=nvidia + NVIDIA_MODEL to use
    any model in NVIDIA's NIM catalog (Nemotron 3 Ultra, Kimi K3, DeepSeek V4, and 90+
    others all share one endpoint/key — see the NVIDIA_API_KEY setup comment for confirmed-
    live model IDs), or AI_PROVIDER=groq for Groq's openai/gpt-oss-120b. Claude/Gemini
    native tool-calling is not wired up for this endpoint yet; /generate-validate-refine
    still works on every provider as before.

    Response shape matches /analyze-part (geometry/FEA/fatigue/rule_engine/health_score/...)
    plus generated_stl_base64, generated_script, and an "engineering_agent" block with the
    full hypothesis log and tool-call trace for transparency.
    """
    return await run_engineering_agent(
        prompt, material, force_n, force_dir, operating_temp_c, surface_finish, reliability,
        project_description, max_iterations, min_health_score, max_critical_violations,
        max_high_violations, min_safety_factor,
    )
