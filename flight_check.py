"""Flight feasibility of a multirotor / VTOL lift system, from the numbers the assembly already has. Pure numpy.

Motor placement around the CoG (cog_balance.py) is only the start. This adds what a layout also has to satisfy:
  * thrust: thrust-to-weight ratio and the hover throttle it implies,
  * control: how much thrust headroom is left for attitude control, and whether yaw torque cancels,
  * arm strength: cantilever bending stress and tip deflection of the arm under a gust-factored thrust,
  * vibration: the arm's first bending frequency against the rotor's once-per-rev and blade-pass frequencies.
These are design-stage hand calculations (stated with their assumptions), not a flight simulation or a structural analysis.
"""
from __future__ import annotations

import math
import re

import numpy as np

_THRUST_RES = [re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g)\s*(?:max\.?\s*)?(?:static\s*)?thrust", re.I),
               re.compile(r"thrust\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(kg|g)\b", re.I)]
_KV_RES = [re.compile(r"\bKV\s*(\d{2,5})\b", re.I), re.compile(r"\b(\d{2,5})\s*KV\b", re.I)]
_CELLS_RE = re.compile(r"\b(\d{1,2})\s*S\b")
_TUBE_RES = [re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\s*outer\s*diameter\D{0,40}?(\d+(?:[.,]\d+)?)\s*mm\s*inner\s*diameter", re.I),
             re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:mm)?\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*mm", re.I)]

# material -> (Young's modulus Pa, allowable bending stress Pa). Allowables are conservative working values, not datasheet maxima.
MATERIAL = {
    "carbon": (70e9, 250e6), "aluminium": (69e9, 150e6), "steel": (200e9, 250e6),
    "petg": (2.1e9, 20e6), "pla": (3.5e9, 30e6), "abs": (2.2e9, 20e6), "nylon": (1.5e9, 25e6),
}


def _f(s):
    return float(str(s).replace(",", "."))


def parse_thrust_g(text):
    for r in _THRUST_RES:
        m = r.search(str(text or ""))
        if m:
            return _f(m.group(1)) * (1000.0 if m.group(2).lower() == "kg" else 1.0)
    return None


def parse_kv(text):
    for r in _KV_RES:
        m = r.search(str(text or ""))
        if m:
            return int(m.group(1))
    return None


def parse_cells(text):
    m = _CELLS_RE.search(str(text or ""))
    return int(m.group(1)) if m else None


def parse_tube(text):
    """(outer, inner) mm from '16 mm outer diameter x 14 mm inner diameter' or '16 x 14 mm', else None."""
    for r in _TUBE_RES:
        m = r.search(str(text or ""))
        if m and _f(m.group(1)) > _f(m.group(2)) > 0:
            return _f(m.group(1)), _f(m.group(2))
    return None


def material_props(name):
    t = str(name or "").lower()
    for k, v in MATERIAL.items():
        if k in t or (k == "aluminium" and ("alumin" in t or "6061" in t)) or (k == "carbon" and ("cfrp" in t or "cfk" in t)):
            return v
    return MATERIAL["petg"]


def tube_section(od, idia):
    """(second moment of area I in m^4, outer radius c in m) of a round tube given in mm."""
    I = math.pi / 64.0 * ((od / 1000.0) ** 4 - (idia / 1000.0) ** 4)
    return I, od / 2000.0


def arm_check(length_mm, od, idia, material, tip_mass_g, hover_thrust_n, max_thrust_n, beam_mass_g=0.0, rpm_hover=None, rpm_max=None, blades=2,
              gust=1.5):
    """Cantilever arm: bending stress at the root under gust x max thrust, deflection at hover thrust, first bending frequency with
    the motor+prop+ESC as a tip mass, and whether it falls within +-15% of a rotor frequency between hover and max rpm."""
    E, allow = material_props(material)
    I, c = tube_section(od, idia)
    L = length_mm / 1000.0
    moment = gust * max_thrust_n * L
    sigma = moment * c / I
    defl = hover_thrust_n * L ** 3 / (3 * E * I)
    m_eff = tip_mass_g / 1000.0 + 0.2357 * beam_mass_g / 1000.0
    fn = math.sqrt(3 * E * I / (m_eff * L ** 3)) / (2 * math.pi) if m_eff > 0 else float("inf")
    out = {"length_mm": round(length_mm, 1), "section_mm": [od, idia], "material": material,
           "root_stress_mpa": round(sigma / 1e6, 1), "allowable_mpa": round(allow / 1e6, 1), "safety_factor": round(allow / sigma, 2),
           "tip_deflection_mm_at_hover": round(defl * 1000, 2), "first_bending_hz": round(fn, 1)}
    if rpm_hover and rpm_max:
        lo, hi = rpm_hover / 60.0, rpm_max / 60.0
        bands = {"once per rev": (lo, hi), "blade pass": (lo * blades, hi * blades)}
        hit = [k for k, (a, b) in bands.items() if a * 0.85 <= fn <= b * 1.15]
        out["rotor_hz"] = {"once_per_rev": [round(lo, 1), round(hi, 1)], "blade_pass": [round(lo * blades, 1), round(hi * blades, 1)]}
        out["resonance_with"] = hit
    return out


def check(p):
    """p: {mass_g, n_motors, prop_d_mm?, motor_max_thrust_g? | motor_text?, kv? , cells? | battery_text?, spins?:[+1,-1..],
           arm_length_mm?, tube_od_mm?, tube_id_mm? | tube_text?, arm_material?, tip_mass_g?, vehicle?}  ->  {checks:[...], ...}"""
    checks, notes = [], []
    mass = float(p["mass_g"])
    n = int(p["n_motors"])
    W = mass / 1000.0 * 9.80665
    tmax_g = p.get("motor_max_thrust_g") or parse_thrust_g(p.get("motor_text"))
    kv = p.get("kv") or parse_kv(p.get("motor_text"))
    cells = p.get("cells") or parse_cells(p.get("battery_text"))
    out = {"mass_g": mass, "n_motors": n, "weight_n": round(W, 2), "inputs_used": {"motor_max_thrust_g": tmax_g, "kv": kv, "cells": cells}}

    hover_thrust_each = W / n
    rpm_hover = rpm_max = None
    if tmax_g:
        tmax_n = tmax_g / 1000.0 * 9.80665
        twr = n * tmax_n / W
        throttle = math.sqrt(1.0 / twr)                      # thrust ~ rpm^2 ~ throttle^2 (rough)
        out.update({"thrust_to_weight": round(twr, 2), "hover_throttle_pct": round(min(throttle, 1.0) * 100, 0),
                    "attitude_headroom_pct": round(max(0.0, (twr - 1) / twr) * 100, 0)})
        lim = 2.0 if p.get("vehicle") != "vtol" else 1.8
        checks.append({"name": "Thrust-to-weight", "value": f"{twr:.2f}", "ok": twr >= lim,
                       "note": f"A multirotor wants at least about {lim:.1f} for control and climb; hover throttle would be about {throttle * 100:.0f}%."
                               if twr < lim else f"Hover at about {throttle * 100:.0f}% throttle, {out['attitude_headroom_pct']:.0f}% thrust headroom."})
        if n >= 4:
            lost = (n - 1) * tmax_n / W
            checks.append({"name": "One motor out", "value": f"{lost:.2f} x weight left", "ok": (lost >= 1.0) if n >= 6 else None, "info": True,
                           "note": "Quads cannot hold attitude after losing a motor; hex and octo layouts can if the remaining thrust still exceeds the weight."})
    else:
        checks.append({"name": "Thrust-to-weight", "value": "unknown", "ok": None,
                       "note": "Give the motor's maximum static thrust (g) to check thrust, hover throttle and control headroom."})
    if kv and cells:
        rpm_max = kv * cells * 3.7 * 0.85                     # loaded: ~85% of the nominal no-load speed
        rpm_hover = rpm_max * (out.get("hover_throttle_pct", 60) / 100.0)
        out["rpm_estimate"] = {"hover": int(rpm_hover), "max": int(rpm_max)}
    spins = p.get("spins")
    if spins:
        net = int(sum(spins))
        checks.append({"name": "Yaw torque", "value": f"{sum(1 for s in spins if s > 0)} CW / {sum(1 for s in spins if s < 0)} CCW", "ok": net == 0,
                       "note": "Equal numbers of clockwise and counter-clockwise props cancel the reaction torque." if net == 0
                               else "Unequal spin directions leave a net yaw torque that the controller has to fight all the time."})

    L = p.get("arm_length_mm")
    tube = (p.get("tube_od_mm"), p.get("tube_id_mm")) if p.get("tube_od_mm") else parse_tube(p.get("tube_text"))
    if L and tube:
        tip = float(p.get("tip_mass_g") or 0.0)
        a = arm_check(float(L), tube[0], tube[1], p.get("arm_material", "carbon"), tip, hover_thrust_each,
                      (tmax_g / 1000.0 * 9.80665) if tmax_g else hover_thrust_each * 2.0, 0.0, rpm_hover, rpm_max)
        out["arm"] = a
        checks.append({"name": "Arm bending", "value": f"SF {a['safety_factor']} ({a['root_stress_mpa']} of {a['allowable_mpa']} MPa)", "ok": a["safety_factor"] >= 2.0,
                       "note": f"{a['section_mm'][0]} x {a['section_mm'][1]} mm {a['material']} tube, {a['length_mm']} mm long, root stress at 1.5 x max thrust. "
                               f"Tip deflection {a['tip_deflection_mm_at_hover']} mm at hover."})
        if "resonance_with" in a:
            checks.append({"name": "Arm vibration", "value": f"{a['first_bending_hz']} Hz", "ok": not a["resonance_with"],
                           "note": ("Arm bending frequency sits inside the rotor " + " and ".join(a["resonance_with"]) + " range: expect vibration. Shorten or stiffen the arm.")
                                   if a["resonance_with"] else "Clear of the rotor once-per-rev and blade-pass frequencies between hover and full throttle."})
        elif kv is None or cells is None:
            checks.append({"name": "Arm vibration", "value": f"{a['first_bending_hz']} Hz", "ok": None,
                           "note": "Needs the motor KV and battery cell count (e.g. 'KV700', '4S') to compare with rotor frequencies."})
    else:
        notes.append("Arm checks need the arm length and the tube size (e.g. '16 mm outer diameter x 14 mm inner diameter').")
    out["checks"] = checks
    out["notes"] = notes + ["Hand calculations: thrust ~ throttle^2, cantilever beam theory, loaded motor speed ~ 85% of KV x 3.7 V x cells. "
                            "Not a flight simulation or a structural analysis (use FEA and flight tests for those)."]
    graded = [c for c in checks if not c.get("info")]          # "info" checks are advice, they do not decide the verdict
    out["verdict"] = ("fail" if any(c["ok"] is False for c in graded) else "incomplete" if any(c["ok"] is None for c in graded) else "pass")
    return out
