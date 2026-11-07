"""Parametric crankshaft (and its bearing block) for the assembly page.

The geometry is a build123d script generated from numbers computed here, so the result is deterministic and not LLM-written. The
counterweights are sized by physics: the rotating mass of crank pin + webs is balanced about the main axis, and the residual
static unbalance and the couple between throws are reported. Units mm, g (steel by default). Crank axis along X in the part.
"""
from __future__ import annotations

import math

import numpy as np

PHASES = {1: [0], 2: [0, 180], 3: [0, 120, 240], 4: [0, 180, 180, 0], 5: [0, 144, 288, 72, 216], 6: [0, 120, 240, 240, 120, 0]}
DENSITY = {"steel": 7.85, "aluminium": 2.70, "titanium": 4.43, "brass": 8.5}        # g/cm3


def _web_raster(r, rw, rp, res=0.25):
    """Area (mm2) and first moment along the arm (mm3) of one web: disc around the main axis + disc around the pin + the arm
    between them, as a union (found on a grid so the overlaps are not counted twice)."""
    ext = max(rw, r + rp) + 1
    xs = np.arange(-ext, ext, res) + res / 2
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    m = (X ** 2 + Y ** 2 <= rw ** 2) | ((X - r) ** 2 + Y ** 2 <= rp ** 2) | ((X >= 0) & (X <= r) & (abs(Y) <= min(rw, rp)))
    a = float(m.sum()) * res * res
    return a, float((X[m]).sum()) * res * res


def design(p=None):
    q = dict(n_throws=4, stroke_mm=40.0, main_d=20.0, pin_d=18.0, main_len=14.0, pin_len=16.0, web_t=8.0, counterweights=True,
             balance_pct=100.0, material="steel", with_blocks=True, bore_clearance=0.10,
             # engine side (optional): without a piston and rod only the crankshaft's own rotating mass is balanced
             piston_mass_g=0.0, rod_mass_g=0.0, rod_length_mm=0.0, rod_cg_frac=0.30, balance_factor=0.50, rpm_reference=6000.0)
    q.update({k: v for k, v in (p or {}).items() if v is not None})
    n = int(q["n_throws"])
    if not 1 <= n <= 6:
        raise ValueError("n_throws must be 1 to 6")
    if q["stroke_mm"] <= 0 or q["pin_d"] >= q["stroke_mm"] * 2.5 or q["main_d"] <= 0:
        raise ValueError("check stroke and journal diameters")
    phases = list(q.get("phases_deg") or PHASES[n])
    if len(phases) != n:
        raise ValueError("phases_deg needs one angle per throw")
    r = q["stroke_mm"] / 2.0
    dm, dp, ml, pl, tw = q["main_d"], q["pin_d"], q["main_len"], q["pin_len"], q["web_t"]
    rho = DENSITY.get(str(q["material"]).lower().split()[0], 7.85)
    rw, rp = 0.78 * dm, 0.78 * dp                     # web radius around the main axis and around the pin
    a_web, mom_web = _web_raster(r, rw, rp)
    v_pin = math.pi / 4 * dp ** 2 * pl
    v_web = a_web * tw
    r_web = mom_web / a_web
    # Static balance of one throw about the main axis, as a moment in g*mm: the crank's own rotating mass, plus what the rod and
    # piston add. The rod's big-end share rotates with the pin; the piston and the rod's small-end share reciprocate, and the
    # counterweights take the fraction `balance_factor` of those (0.5 is the usual compromise for a single cylinder).
    rho_mm3 = rho / 1000.0                                   # g per mm3
    beta = float(q["balance_factor"])
    m_big = q["rod_mass_g"] * (1.0 - q["rod_cg_frac"])       # rod mass at the big end: m * (L - a) / L, a = CoG distance from the big end
    m_recip = q["piston_mass_g"] + q["rod_mass_g"] * q["rod_cg_frac"]
    own = rho_mm3 * (v_pin * r + 2 * v_web * r_web)          # the crankshaft's own rotating moment, g*mm
    target = q["balance_pct"] / 100.0 * (own + (m_big + beta * m_recip) * r)
    rc = None
    capped = False
    if q["counterweights"]:
        rc = (rw ** 3 + 3.0 * target / (4.0 * tw * rho_mm3)) ** (1.0 / 3.0)
        cap = r + 0.5 * dp + 4.0                     # keep counterweights inside the crank pin's swing circle (+4 mm), where the rods and housing clear
        if rc > cap:
            rc, capped = cap, True
    got = 0.0 if rc is None else rho_mm3 * 4.0 * tw / 3.0 * (rc ** 3 - rw ** 3)      # 2 half-annuli: 2 * tw * (2/3)(rc^3 - rw^3) * rho
    tw_needed = None
    if capped and rc and rc > rw:
        # web thickness at which the capped counterweight does balance: the webs get heavier too, so solve
        #   tw * (4/3 (rc^3 - rw^3) - 2 pct A r_web) = pct (V_pin r + (m_big + beta m_recip) r / rho)
        pct = q["balance_pct"] / 100.0
        den = 4.0 / 3.0 * (rc ** 3 - rw ** 3) - 2 * pct * a_web * r_web
        if den > 0:
            tw_needed = pct * (v_pin * r + (m_big + beta * m_recip) * r / rho_mm3) / den
    full = own + (m_big + beta * m_recip) * r
    resid_pct = 100.0 * (1.0 - got / full) if q["counterweights"] else 100.0

    # layout along the axis, and the vector sum of the throws (force) and its couple about the middle (moment)
    zs, z = [], 0.0
    z += ml
    for _ in range(n):
        zs.append(z + tw + pl / 2)
        z += 2 * tw + pl + ml
    length = z
    rot_m = full - got                                                  # unbalanced rotating moment per throw, g*mm
    ang = np.radians(phases)
    vec = rot_m * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    force = vec.sum(axis=0)
    couple = (vec * (np.array(zs) - length / 2)[:, None]).sum(axis=0)
    mass_g = rho_mm3 * (n * (v_pin + 2 * v_web) + (n + 1) * math.pi / 4 * dm ** 2 * ml
                             + (0 if rc is None else n * 2 * tw * (math.pi * rc ** 2 / 2 - 0.5 * math.pi * min(rc, rw) ** 2)))
    data = {
        "n_throws": n, "phases_deg": phases, "stroke_mm": q["stroke_mm"], "crank_radius_mm": r, "length_mm": round(length, 2),
        "counterweight_radius_mm": None if rc is None else round(rc, 2), "counterweight_capped": capped,
        "static_unbalance_per_throw_pct": round(resid_pct, 1), "net_force_unbalance_g_mm": round(float(np.hypot(*force)), 3),
        "net_couple_unbalance_g_mm2": round(float(np.hypot(*couple)), 3), "mass_g_estimate": round(mass_g, 1), "material": q["material"],
        "z_throw_centres_mm": [round(v, 2) for v in zs],
        "notes": ["Balance figures are analytic (web area by grid, counterweights as half discs). The assembly page measures the real "
                  "centre of gravity of the solid, which should sit on the main axis."]
              + ([f"The counterweight hit its size limit (inside the crank pin's swing circle), so the throw stays partly unbalanced. "
                  f"Making the webs about {tw_needed:.0f} mm thick would balance it at this radius, or raise the stroke or lower the balance %."
                  if tw_needed else "The counterweight hit its size limit: the throw stays partly unbalanced."] if capped else []),
        "web_t_needed_mm": None if tw_needed is None else round(tw_needed, 1),
    }
    if m_recip > 0 or m_big > 0:
        data["engine"] = _engine_balance(phases, zs, length, r, m_recip, beta, q["rod_length_mm"], q["rpm_reference"], rot_m)
    out = {"params": {k: q[k] for k in ("n_throws", "stroke_mm", "main_d", "pin_d", "main_len", "pin_len", "web_t",
                                        "counterweights", "balance_pct", "material", "with_blocks", "bore_clearance",
                                        "piston_mass_g", "rod_mass_g", "rod_length_mm", "rod_cg_frac", "balance_factor")},
           "data": data, "script": _crank_script(phases, r, dm, dp, ml, pl, tw, rw, rp, rc, length, q["balance_pct"], data)}
    if q["with_blocks"]:
        out["block_script"] = _block_script(dm, ml, q["bore_clearance"])
    return out


def _engine_balance(phases, zs, length, r, m_recip, beta, rod_len, rpm, rot_resid_gmm):
    """Inertia forces of the reciprocating mass for an inline engine with all cylinders in one plane.
    With a balance factor beta, one cylinder keeps (1-beta) of its primary force along the bore and gains beta of it at right
    angles; the sum over cylinders depends on the crank phases. Secondary forces (twice crank speed, scaled by r / rod length)
    are not touched by crankshaft counterweights at all: they need balance shafts. Values are amplitudes in g*mm (multiply by
    omega^2 for a force) and, for the reference rpm, in newtons / N*m."""
    th = np.radians(phases)
    zc = np.array(zs) - length / 2.0
    s1 = abs(np.exp(1j * th).sum())
    c1 = abs((zc * np.exp(1j * th)).sum())
    s2 = abs(np.exp(2j * th).sum())
    c2 = abs((zc * np.exp(2j * th)).sum())
    lam = r / rod_len if rod_len > 0 else 0.0
    w2 = (2 * math.pi * rpm / 60.0) ** 2 * 1e-6                  # (g*mm) -> (kg*m) is 1e-6, times omega^2 gives newtons
    prim = m_recip * r * s1
    out = {
        "reciprocating_mass_g": round(m_recip, 2), "balance_factor": beta, "rpm_reference": rpm,
        "primary_force_g_mm": round(float(prim), 2), "primary_force_n": round(float(prim * w2), 2),
        "primary_along_bore_fraction": round(1 - beta, 2), "primary_across_bore_fraction": round(beta, 2),
        "primary_couple_g_mm2": round(float(m_recip * r * c1), 1), "primary_couple_nm": round(float(m_recip * r * c1 * w2 * 1e-3), 3),
        "secondary_force_g_mm": round(float(m_recip * r * lam * s2), 2), "secondary_force_n": round(float(m_recip * r * lam * s2 * w2), 2),
        "secondary_couple_nm": round(float(m_recip * r * lam * c2 * w2 * 1e-3), 3),
        "rotating_residual_g_mm_per_throw": round(float(rot_resid_gmm), 3),
        "notes": ["Primary forces balance when the phase vectors sum to zero (inline-4 flat-plane, inline-6, inline-3 couple aside); a single "
                  "cylinder can only trade force along the bore for force across it (balance factor).",
                  "Secondary forces and couples need balance shafts or a different configuration; counterweights cannot remove them.",
                  "This is an analytic estimate for one plane of cylinders. Dynamic balance of the real part has to be measured on a balancing machine."],
    }
    if rod_len <= 0:
        out["notes"].append("Rod length not given, so secondary forces are not computed.")
    return out


def _crank_script(phases, r, dm, dp, ml, pl, tw, rw, rp, rc, length, pct, data):
    return f'''from build123d import *
import math
# Parametric crankshaft: {len(phases)} throw(s), stroke {2 * r:g} mm, main journal D{dm:g}, crank pin D{dp:g}. Axis along X.
# Counterweights sized for {pct:g}% static balance (analytic unbalance per throw {data["static_unbalance_per_throw_pct"]}%).
PHASES = {list(phases)}
R, DM, DP, ML, PL, TW, RW, RP, RC = {r}, {dm}, {dp}, {ml}, {pl}, {tw}, {rw:.3f}, {rp:.3f}, {rc if rc else 0.0}
L = {length:.3f}


def web(theta, z0):
    a = math.radians(theta)
    w = Pos(0, 0, z0 + TW / 2) * Cylinder(RW, TW)
    w = w + Pos(R * math.cos(a), R * math.sin(a), z0 + TW / 2) * Cylinder(RP, TW)
    w = w + Rot(0, 0, theta) * (Pos(R / 2, 0, z0 + TW / 2) * Box(R, 2 * min(RW, RP), TW))
    if RC > 0:
        half = Cylinder(RC, TW) & (Pos(RC / 2, 0, 0) * Box(RC, 2 * RC, TW))
        w = w + Rot(0, 0, theta + 180) * (Pos(0, 0, z0 + TW / 2) * half)
    return w


crank = Pos(0, 0, ML / 2) * Cylinder(DM / 2, ML)
z = ML
for theta in PHASES:
    a = math.radians(theta)
    crank = crank + web(theta, z)
    crank = crank + Pos(R * math.cos(a), R * math.sin(a), z + TW + PL / 2) * Cylinder(DP / 2, PL)
    crank = crank + web(theta, z + TW + PL)
    z += 2 * TW + PL
    crank = crank + Pos(0, 0, z + ML / 2) * Cylinder(DM / 2, ML)
    z += ML
crank = Pos(0, 0, -L / 2) * crank
result = Rot(0, 90, 0) * crank
'''


def _block_script(dm, ml, clearance):
    wall, base = 8.0, 6.0
    return f'''from build123d import *
# Plain bearing block for the crankshaft main journal (bore = journal + {2 * clearance:g} mm). Bore axis along X through the origin.
DM, BL, CLR, WALL, BASE = {dm}, {ml}, {clearance}, {wall}, {base}
W = DM + 2 * WALL
blk = Box(BL, W, W)
blk = blk + Pos(0, 0, -(W / 2 + BASE / 2)) * Box(BL, W * 1.7, BASE)
blk = blk - Rot(0, 90, 0) * Cylinder(DM / 2 + CLR, BL * 3)
for y in (-W * 0.72, W * 0.72):
    blk = blk - Pos(0, y, -(W / 2 + BASE / 2)) * Cylinder(2.25, BASE * 3)
result = blk
'''
