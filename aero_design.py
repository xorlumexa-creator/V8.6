"""Aerodynamic shape and drag/lift estimates for Lumexa. Pure numpy. This is a design-stage estimator (flat-plate friction with form
factors, lifting-line style induced drag), good for comparing shapes and sizing; it is not CFD. Use OpenFOAM / SU2 for that.

Shapes are build123d scripts generated from numbers, so they are deterministic: a streamlined fuselage (lofted ellipses) and a
tapered wing (lofted NACA sections). X front(-) to rear(+), Y span, Z up, like the rest of the assembly. Units: mm for geometry,
SI for the flight estimate.
"""
from __future__ import annotations

import math

import numpy as np

NU_AIR = 1.46e-5      # kinematic viscosity, m2/s (15 C, sea level)
RHO_AIR = 1.225       # kg/m3


# ------------------------------------------------------------------------------------------------------------ airfoil
def naca4(code="2412", n=41):
    """NACA 4-digit section with a closed trailing edge, chord = 1, cosine spacing. Returns (x_upper, y_upper, x_lower, y_lower)."""
    code = str(code).strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA 4-digit code, e.g. 2412 or 0012")
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    x = (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m == 0 or p == 0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    return x - yt * np.sin(th), yc + yt * np.cos(th), x + yt * np.sin(th), yc - yt * np.cos(th)


def airfoil_outline(code, chord, twist_deg=0.0, n=41):
    """Closed outline (list of (x, z) in mm, leading edge at x=0, trailing edge at x=chord) with twist about the quarter chord."""
    xu, yu, xl, yl = naca4(code, n)
    upper = np.stack([xu, yu], axis=1)[::-1]            # TE -> LE along the top
    lower = np.stack([xl, yl], axis=1)[1:-1]            # LE -> just before TE along the bottom
    pts = np.vstack([upper, lower]) * chord
    a = math.radians(twist_deg)
    c, s = math.cos(a), math.sin(a)
    q = np.array([0.25 * chord, 0.0])
    pts = (pts - q) @ np.array([[c, s], [-s, c]]) + q
    return [(round(float(x), 3), round(float(z), 3)) for x, z in pts]


# ------------------------------------------------------------------------------------------------------------ drag and lift
def reynolds(v, length_m):
    return v * length_m / NU_AIR


def cf_flat_plate(re):
    """Skin friction coefficient of a flat plate: laminar below Re 5e5, turbulent (with the transition correction) above."""
    re = max(float(re), 1e3)
    return 1.328 / math.sqrt(re) if re < 5e5 else 0.455 / (math.log10(re) ** 2.58) - 1700.0 / re


def fuselage_cda(length_mm, dmax_mm, v):
    """Drag area (m2) of a streamlined body: Cf x form factor x wetted area (Raymer's fuselage form factor)."""
    L, D = length_mm / 1000.0, dmax_mm / 1000.0
    lam = L / D
    swet = math.pi * D * L * (1 - 2 / lam) ** (2 / 3) * (1 + 1 / lam ** 2) if lam > 2 else math.pi * D * L
    ff = 1 + 60 / lam ** 3 + lam / 400
    return cf_flat_plate(reynolds(v, L)) * ff * swet, {"fineness": round(lam, 2), "wetted_m2": round(swet, 4), "form_factor": round(ff, 3)}


def wing_geometry(span_mm, root_mm, tip_mm):
    b, cr, ct = span_mm / 1000.0, root_mm / 1000.0, tip_mm / 1000.0
    s = b * (cr + ct) / 2
    lam = ct / cr
    mac = 2 / 3 * cr * (1 + lam + lam ** 2) / (1 + lam)
    return {"S_m2": s, "AR": b * b / s, "MAC_m": mac, "taper": lam}


def estimate(p):
    """p: {mass_g, speed_ms, span_mm, root_chord_mm, tip_chord_mm, thickness_ratio, fuselage_length_mm, fuselage_dia_mm,
           extra_cda_m2?, cl_max?, rho?, sweep_deg?}  ->  a dict of numbers and warnings."""
    w = wing_geometry(p["span_mm"], p["root_chord_mm"], p["tip_chord_mm"])
    v = float(p["speed_ms"])
    rho = float(p.get("rho", RHO_AIR))
    tc = float(p.get("thickness_ratio", 0.12))
    weight = float(p["mass_g"]) / 1000.0 * 9.80665
    sweep = math.radians(float(p.get("sweep_deg", 0.0)))
    q = 0.5 * rho * v * v
    cl = weight / (q * w["S_m2"])
    re_w = reynolds(v, w["MAC_m"])
    ff_w = (1 + 0.6 / 0.3 * tc + 100 * tc ** 4) * (1.34 * (v / 343.0) ** 0.18 * math.cos(sweep) ** 0.28)
    ff_w = max(ff_w, 1 + 2 * tc)                       # at low Mach the Mach term is below 1 and would understate the form factor
    cd0_w = cf_flat_plate(re_w) * ff_w * 2 * (1 + 0.2 * tc)
    cda_f, finfo = fuselage_cda(p["fuselage_length_mm"], p["fuselage_dia_mm"], v)
    cda_x = float(p.get("extra_cda_m2", 0.0))
    cd0 = cd0_w + (cda_f + cda_x) / w["S_m2"]
    e = max(0.6, min(0.95, 1.78 * (1 - 0.045 * w["AR"] ** 0.68) - 0.64))      # Oswald efficiency, straight wing
    cdi = cl * cl / (math.pi * w["AR"] * e)
    cd = cd0 + cdi
    drag = q * w["S_m2"] * cd
    cl_max = float(p.get("cl_max", 1.2))
    out = {
        "wing": {k: round(v_, 4) for k, v_ in w.items()}, "cl_required": round(cl, 3), "reynolds_wing": int(re_w), "oswald_e": round(e, 3),
        "cd0": round(cd0, 5), "cd_induced": round(cdi, 5), "cd_total": round(cd, 5), "l_over_d": round(cl / cd, 2),
        "drag_n": round(drag, 3), "power_required_w": round(drag * v, 1), "fuselage": finfo,
        "stall_speed_ms": round(math.sqrt(2 * weight / (rho * w["S_m2"] * cl_max)), 2), "wing_loading_n_m2": round(weight / w["S_m2"], 1),
        "best_glide_speed_ms": round(math.sqrt(2 * weight / (rho * w["S_m2"] * math.sqrt(cd0 * math.pi * w["AR"] * e))), 2),
        "warnings": [],
    }
    if cl > cl_max * 0.85:
        out["warnings"].append(f"Needs CL {cl:.2f} at {v:g} m/s, close to or above a typical CLmax of {cl_max}: fly faster or add wing area.")
    if re_w < 2e5:
        out["warnings"].append("Wing Reynolds number below 200 000: laminar separation is likely, so real drag will be higher than this estimate.")
    if finfo["fineness"] < 3 or finfo["fineness"] > 8:
        out["warnings"].append(f"Fuselage fineness {finfo['fineness']}: lowest drag for a given volume is around 4 to 6.")
    out["notes"] = ["Estimator only: flat-plate friction x form factor + induced drag. Check the final shape with CFD (OpenFOAM or SU2)."]
    return out


# ------------------------------------------------------------------------------------------------------------ shape scripts
def fuselage_sections(length, dmax, nose_frac=0.28, tail_frac=0.42, n_nose=2.0, n=11, tail_end=0.06):
    """(x, radius) stations from nose (x=-L/2) to tail: ellipsoidal nose of order n_nose, parallel middle, smooth tail cone."""
    R = dmax / 2
    xs = np.linspace(0, 1, n)
    out = []
    for s in xs:
        if s < nose_frac:
            t = s / nose_frac
            r = R * (1 - (1 - t) ** n_nose) ** (1 / n_nose)
        elif s > 1 - tail_frac:
            t = (s - (1 - tail_frac)) / tail_frac
            r = R * (1 - (1 - tail_end) * t ** 1.5)
        else:
            r = R
        out.append((round(float(-length / 2 + s * length), 2), round(float(max(r, 0.5)), 2)))     # plain floats: they are printed into the script
    return out


def fuselage_script(length_mm, dmax_mm, height_mm=None, **kw):
    secs = fuselage_sections(length_mm, dmax_mm, **kw)
    k = (height_mm or dmax_mm) / dmax_mm
    rows = [(x, r, round(r * k, 2)) for x, r in secs]
    return f'''from build123d import *
# Streamlined fuselage: length {length_mm:g} mm, max width {dmax_mm:g} mm, height {height_mm or dmax_mm:g} mm. Nose at -X.
SECTIONS = {rows}
with BuildPart() as fus:
    for x, a, b in SECTIONS:
        with BuildSketch(Plane.YZ.offset(x)):
            Ellipse(a, b)
    loft()
result = fus.part
'''


def wing_script(span_mm, root_mm, tip_mm, naca="2412", sweep_deg=0.0, dihedral_deg=0.0, twist_tip_deg=-2.0, x_quarter_root=0.0, z0=0.0):
    """Full wing (both sides, 5 sections, ruled loft so the taper stays straight). Quarter-chord line swept by sweep_deg."""
    ys = [-span_mm / 2, -span_mm / 4, 0.0, span_mm / 4, span_mm / 2]
    sections = []
    for y in ys:
        f = abs(y) / (span_mm / 2)
        chord = root_mm + (tip_mm - root_mm) * f
        x_le = x_quarter_root - 0.25 * chord + abs(y) * math.tan(math.radians(sweep_deg))
        z = z0 + abs(y) * math.tan(math.radians(dihedral_deg))
        sections.append((airfoil_outline(naca, chord, twist_tip_deg * f), (round(x_le, 3), round(y, 3), round(z, 3))))
    return f'''from build123d import *
# Tapered wing: span {span_mm:g} mm, chord {root_mm:g} -> {tip_mm:g} mm, NACA {naca}, sweep {sweep_deg:g} deg, dihedral {dihedral_deg:g} deg.
SECTIONS = {sections}
with BuildPart() as wing:
    for pts, org in SECTIONS:
        with BuildSketch(Plane(origin=org, x_dir=(1, 0, 0), z_dir=(0, -1, 0))):
            with BuildLine():
                Polyline(*pts, close=True)
            make_face()
    loft(ruled=True)
result = wing.part
'''
