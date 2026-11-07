"""Mass, centre of gravity (CoG) and balance of an assembly. Pure numpy: no CAD kernel and no model involved, so it is fast
enough to rerun on every change and can be tested without OpenCASCADE.

What it knows
  * the mass of every part (volume from the real solid x material density; printed plastics get an infill factor),
  * the mass of every component (read from the datasheet text, e.g. "82 g (104 g with cable)", else a flagged estimate),
  * where each placed component sits (from the packing of the measured cavities), so the CoG moves when a part is rebuilt,
  * for multirotors and VTOLs: where the lift motors must go so that their thrust centroid lies on the CoG, how far the
    battery can slide to correct an offset, and what thrust each motor needs for a given CoG.

Units: mm, g, assembly frame (X front(-) to rear(+), Y sideways, Z up) unless stated otherwise.
"""
from __future__ import annotations

import math
import re

import numpy as np

# keyword -> (density g/cm3, fill factor). Fill < 1 is the effective solid fraction of an FDM print (walls + infill).
_MATERIALS = [
    (r"petg", 1.27, 0.35), (r"\bpla\b|pla[ _+]", 1.24, 0.35), (r"\babs\b", 1.05, 0.35), (r"\basa\b", 1.07, 0.35),
    (r"nylon|\bpa\d*\b|pa12|pa6", 1.14, 0.40), (r"tpu|flex", 1.20, 0.30),
    (r"carbon|cfrp|cfk", 1.60, 1.0), (r"fib(?:re|er)glass|glass", 1.90, 1.0),
    (r"alumin|6061|7075|alsi", 2.70, 1.0), (r"stainless|316|304", 7.98, 1.0), (r"steel|4340|1045", 7.85, 1.0),
    (r"titan", 4.43, 1.0), (r"magnes", 1.77, 1.0), (r"brass|bronze", 8.50, 1.0), (r"balsa", 0.16, 1.0),
    (r"epp|eps|depron|foam", 0.03, 1.0), (r"wood|ply", 0.60, 1.0),
]
DEFAULT_DENSITY = (1.20, 0.35)        # an unknown plastic-like material
ELECTRONICS_G_PER_CM3 = 1.4           # effective density of a board or module, used only when no mass is given

_MASS_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(kg|g)\b", re.I)
_DIAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\s*(?:outer\s*)?diameter", re.I)


def parse_mass_g(text):
    """First mass in a datasheet string: '429.8 g' -> 429.8, '82 g (104 g with cable)' -> 82, '1.2 kg' -> 1200, none -> None."""
    m = _MASS_RE.search(str(text or ""))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * 1000.0 if m.group(2).lower() == "kg" else v


def material_density(text):
    """(density g/cm3, fill factor, recognised) for a material name such as 'PETG', 'Carbon fibre', 'Aluminium 6061'."""
    t = str(text or "").lower()
    for pat, rho, fill in _MATERIALS:
        if re.search(pat, t):
            return rho, fill, True
    return DEFAULT_DENSITY[0], DEFAULT_DENSITY[1], False


def part_mass_g(volume_mm3, material_text, print_fill=None):
    rho, fill, known = material_density(material_text)
    if print_fill is not None and fill < 1.0:
        fill = float(print_fill)
    return float(volume_mm3) / 1000.0 * rho * fill, known


def component_mass_g(name, dims_text, dims=None):
    """(mass in g, source) for one unit. 'datasheet' when the text states it, 'estimate' for a flagged guess, else (None, None)."""
    m = parse_mass_g(dims_text)
    if m is not None:
        return m, "datasheet"
    n = str(name or "").lower()
    if re.search(r"propeller|\bprop\b", n):
        d = _DIAM_RE.search(str(dims_text or ""))
        inch = re.search(r"(\d+(?:\.\d+)?)\s*in\b", n)
        if d:
            return 0.125 * (float(d.group(1).replace(",", ".")) / 25.4) ** 2, "estimate"      # ~18 g for a 12 in prop
        if inch:
            return 0.125 * float(inch.group(1)) ** 2, "estimate"
        return None, None
    if re.search(r"wire|cable|tube|spar|boom", n):
        return None, None                              # needs a length
    if dims and len(dims) == 3:
        return float(np.prod(dims)) / 1000.0 * ELECTRONICS_G_PER_CM3, "estimate"
    return None, None


def role_of(name):
    """What a component does for balance: lift_motor, forward_motor, lift_prop, forward_prop, esc, battery, or None."""
    n = str(name or "").lower()
    fwd = bool(re.search(r"pusher|forward|cruise|tractor|tail", n))
    if re.search(r"motor", n) and not re.search(r"servo", n):
        return "forward_motor" if fwd else "lift_motor"
    if re.search(r"propeller|\bprop\b", n):
        return "forward_prop" if fwd else "lift_prop"
    if re.search(r"\besc\b|speed controller", n):
        return "esc"
    if re.search(r"batter|lipo|li-ion|cell pack", n):
        return "battery"
    return None


def vehicle_kind(text):
    """Guess the vehicle type from a project's category / purpose / name text."""
    t = str(text or "").lower()
    if "vtol" in t:
        return "vtol"
    if re.search(r"drone|quad|multirotor|multi-rotor|copter|hexa|octo", t):
        return "multirotor"
    if re.search(r"plane|wing|glider|aircraft", t):
        return "fixed_wing"
    return "generic"


# ------------------------------------------------------------------------------------------------------------ geometry helpers
def world_point(M, p):
    return (np.asarray(M, float).reshape(4, 4) @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def world_aabb(M, lo, hi):
    pts = np.array([world_point(M, (x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return pts.min(axis=0), pts.max(axis=0)


def cog_of(items):
    """(total mass g, CoG mm) of items that have a mass and a position."""
    use = [i for i in items if i.get("mass_g") and i.get("pos") is not None]
    if not use:
        return 0.0, None
    m = np.array([i["mass_g"] for i in use], float)
    p = np.array([i["pos"] for i in use], float)
    return float(m.sum()), (m[:, None] * p).sum(axis=0) / m.sum()


def inertia_about(items, c):
    """Point-mass estimate of the inertia tensor about c, kg*m2 (every item is a point at its centre of mass)."""
    I = np.zeros((3, 3))
    for i in items:
        if not i.get("mass_g") or i.get("pos") is None:
            continue
        r = (np.asarray(i["pos"], float) - c) / 1000.0
        I += (i["mass_g"] / 1000.0) * ((r @ r) * np.eye(3) - np.outer(r, r))
    return I


# ------------------------------------------------------------------------------------------------------------ motors and thrust
def motor_layout(n, prop_d_mm, style="auto", clearance=0.10):
    """n lift motors on a circle with adjacent props `clearance` (fraction of the prop diameter) apart and alternating spins so
    the yaw torque cancels. Centred on (0, 0)."""
    n = int(max(3, n))
    spacing = float(prop_d_mm) * (1.0 + clearance)
    radius = spacing / (2.0 * math.sin(math.pi / n))
    x_style = n == 4 and style in ("auto", "x")
    theta0 = math.radians(45.0) if x_style else 0.0
    pos = [{"x": round(radius * math.cos(theta0 + 2 * math.pi * k / n), 2),
            "y": round(radius * math.sin(theta0 + 2 * math.pi * k / n), 2), "spin": 1 if k % 2 == 0 else -1} for k in range(n)]
    return {"radius_mm": round(radius, 2), "positions": pos, "spacing_mm": round(spacing, 2), "yaw_balanced": n % 2 == 0,
            "style": "x" if x_style else ("plus" if n == 4 else f"{n}-gon")}


def thrust_shares(xy, cog_xy):
    """Share of the total thrust each motor must give to hover with the CoG at cog_xy: closest to equal shares while the forces
    sum to the weight and their moments about the CoG cancel. A share far from 1/n means the layout is out of balance."""
    xy = np.asarray(xy, float)
    n = len(xy)
    A = np.vstack([np.ones(n), xy[:, 0], xy[:, 1]])
    b = np.array([1.0, cog_xy[0], cog_xy[1]])
    t0 = np.full(n, 1.0 / n)
    return t0 + np.linalg.pinv(A) @ (b - A @ t0)


# ------------------------------------------------------------------------------------------------------------ the analysis
def analyze(inp):
    """inp: {parts:[{id,name,matrix,mass_g?,volume_mm3?,com_local?,bbox_local?,material?}],
             components:[{part,name,mass_each_g?,center_local,travel_world?}]      (one record per placed unit)
             inventory:[{name,dims_text,qty,dims?}], vehicle, prop_d_mm?, layout?, frame_center?, loose_positions?, print_fill?}"""
    warnings = []
    fill = inp.get("print_fill")
    items, unknown, mats = [], [], {}
    for p in inp.get("parts", []):
        M = np.asarray(p["matrix"], float).reshape(4, 4)
        mass, src = p.get("mass_g"), "given"
        if mass is None and p.get("volume_mm3"):
            mass, known = part_mass_g(p["volume_mm3"], p.get("material"), fill)
            src = f"volume x {p.get('material') or 'default'} density"
            if not known:
                warnings.append(f"{p['name']}: material '{p.get('material') or ''}' not recognised, a generic plastic density was used")
        com = p.get("com_local")
        if com is None:
            bb = p.get("bbox_local")
            com = ((np.asarray(bb["min"]) + np.asarray(bb["max"])) / 2).tolist() if bb else [0, 0, 0]
        if mass is None:
            unknown.append(p["name"])
            continue
        items.append({"name": p["name"], "kind": "part", "mass_g": float(mass), "pos": world_point(M, com), "source": src, "part": p["id"]})
        mats[p["id"]] = M
    placed_count = {}
    for c in inp.get("components", []):
        M = mats.get(c["part"])
        if M is None:
            continue
        placed_count[c["name"]] = placed_count.get(c["name"], 0) + 1
        if c.get("mass_each_g") is None:
            if c["name"] not in unknown:
                unknown.append(c["name"])
            continue
        items.append({"name": c["name"], "kind": "component", "mass_g": float(c["mass_each_g"]), "pos": world_point(M, c["center_local"]),
                      "source": c.get("source", "datasheet"), "part": c["part"], "travel": c.get("travel_world"), "role": role_of(c["name"])})

    loose, loose_pos = [], inp.get("loose_positions") or {}
    for it in inp.get("inventory", []):
        left = max(0, int(it.get("qty") or 1) - placed_count.get(it["name"], 0))
        if left <= 0:
            continue
        mass, src = component_mass_g(it["name"], it.get("dims_text", ""), it.get("dims"))
        rec = {"name": it["name"], "qty": left, "mass_each_g": mass, "source": src, "role": role_of(it["name"]), "dims_text": it.get("dims_text", "")}
        if it["name"] in loose_pos:
            rec["pos"] = [float(v) for v in loose_pos[it["name"]]]
        loose.append(rec)
    for rec in loose:
        if "pos" in rec and rec["mass_each_g"]:
            for _ in range(rec["qty"]):
                items.append({"name": rec["name"], "kind": "component", "mass_g": rec["mass_each_g"], "pos": np.array(rec["pos"]),
                              "source": rec["source"], "role": rec["role"], "part": None})
            rec["counted"] = True

    mass, cog = cog_of(items)
    ref = None
    if cog is not None:
        pts = np.array([i["pos"] for i in items])
        ref = np.array(inp["frame_center"], float) if inp.get("frame_center") else (pts.min(axis=0) + pts.max(axis=0)) / 2
    unplaced = [r for r in loose if not r.get("counted")]
    unplaced_mass = sum((r["mass_each_g"] or 0) * r["qty"] for r in unplaced)
    for r in unplaced:
        if r["mass_each_g"] is None and r["name"] not in unknown:
            unknown.append(r["name"])
    r2 = lambda v, k=2: [round(float(x), k) for x in v]
    out = {
        "units": "mm, g, assembly frame", "vehicle": inp.get("vehicle", "generic"),
        "mass": {"counted_g": round(mass, 1), "not_placed_g": round(unplaced_mass, 1), "total_g": round(mass + unplaced_mass, 1)},
        "cog_mm": r2(cog) if cog is not None else None,
        "frame_center_mm": r2(ref) if ref is not None else None,
        "cog_offset_mm": r2(cog - ref) if cog is not None else None,
        "items": sorted([{"name": i["name"], "kind": i["kind"], "mass_g": round(i["mass_g"], 2), "pos": r2(i["pos"], 1), "source": i["source"]}
                         for i in items], key=lambda r: -r["mass_g"])[:40],
        "not_placed": [{"name": r["name"], "qty": r["qty"], "mass_each_g": None if r["mass_each_g"] is None else round(r["mass_each_g"], 1),
                        "source": r["source"], "role": r["role"]} for r in unplaced],
        "mass_unknown": unknown, "warnings": warnings,
        "assumptions": ["Part mass = solid volume x density x a print-fill factor for plastics; component mass is the datasheet value, "
                        "or a flagged estimate."],
    }
    if cog is not None:
        I = inertia_about(items, cog)
        out["inertia_kgm2"] = {"xx": round(float(I[0, 0]), 5), "yy": round(float(I[1, 1]), 5), "zz": round(float(I[2, 2]), 5)}
    if inp.get("vehicle") in ("multirotor", "vtol") and cog is not None:
        out["lift_plan"] = _lift_plan(inp, items, loose, cog, mass)
        out["battery_shift"] = _battery_shift(items, cog, mass, ref)
    if unknown:
        out["warnings"].append("No mass known for: " + ", ".join(unknown) + ". They are not in the CoG.")
    return out


def _lift_plan(inp, items, loose, cog, mass):
    """Where the lift motors go: their centroid must be the CoG of the whole vehicle, which holds exactly when it equals the CoG
    of everything else (a unit = motor + prop + ESC)."""
    lift = [r for r in loose if r["role"] == "lift_motor" and not r.get("counted")]
    if not lift:
        placed = [i for i in items if i.get("role") == "lift_motor"]
        if len(placed) >= 3:
            xy = np.array([i["pos"][:2] for i in placed])
            off = xy.mean(axis=0) - cog[:2]
            return {"status": "placed", "n": len(placed), "centroid_mm": [round(float(v), 2) for v in xy.mean(axis=0)],
                    "offset_from_cog_mm": [round(float(v), 2) for v in off],
                    "thrust_share_pct": [round(float(s) * 100, 1) for s in thrust_shares(xy, cog[:2])],
                    "balanced": bool(np.abs(off).max() < 5.0)}
        return {"status": "no_lift_motors"}
    n = sum(r["qty"] for r in lift)
    motor_m = next((r["mass_each_g"] for r in lift if r["mass_each_g"]), None)
    prop = next((r for r in loose if r["role"] == "lift_prop" and not r.get("counted")), None)
    esc = next((r for r in loose if r["role"] == "esc" and not r.get("counted")), None)
    unit_g = (motor_m or 0.0) + ((prop["mass_each_g"] or 0.0) if prop else 0.0) + ((esc["mass_each_g"] or 0.0) if esc and esc["qty"] >= n else 0.0)
    prop_d = inp.get("prop_d_mm")
    if not prop_d and prop:
        mm = _DIAM_RE.search(prop.get("dims_text", ""))
        inch = re.search(r"(\d+(?:\.\d+)?)\s*in\b", prop["name"], re.I)
        prop_d = float(mm.group(1)) if mm else (float(inch.group(1)) * 25.4 if inch else None)
    prop_d = float(prop_d or 254.0)
    lay = motor_layout(n, prop_d, inp.get("layout") or "auto")
    c = cog[:2]
    plan = {
        "status": "planned", "n": n, "prop_diameter_mm": round(prop_d, 1), "layout": lay["style"], "arm_radius_mm": lay["radius_mm"],
        "yaw_balanced": lay["yaw_balanced"], "unit_mass_g": round(unit_g, 1), "centre_xy_mm": [round(float(v), 2) for v in c],
        "motors": [{"x": round(float(c[0] + p["x"]), 2), "y": round(float(c[1] + p["y"]), 2), "spin": p["spin"]} for p in lay["positions"]],
        "thrust_share_pct": [round(float(s) * 100, 1) for s in thrust_shares(np.array([[c[0] + p["x"], c[1] + p["y"]] for p in lay["positions"]]), c)],
        "rule": "Put the lift-motor circle's centre on the CoG of everything else. Then the total CoG is that same point and every "
                "motor carries an equal share.",
    }
    if motor_m is None:
        plan["warning"] = "motor mass unknown, so the lift units are not in the CoG numbers"
    others = [r for r in loose if not r.get("counted") and r["role"] not in ("lift_motor", "lift_prop", "esc") and r["mass_each_g"]]
    if others:
        plan["assumption"] = ("Not placed yet, assumed to sit on the CoG: " + ", ".join(r["name"] for r in others) +
                              ". A pusher motor at the tail pulls the real CoG aft: give its position to move the plan.")
    return plan


def _battery_shift(items, cog, mass, ref):
    """How far the battery must slide (and whether its cavity allows it) to bring the CoG onto the frame centre, i.e. onto the
    lift-motor centre when the arms are fixed where they attach."""
    bats = [i for i in items if i.get("role") == "battery"]
    if not bats or ref is None:
        return None
    b = max(bats, key=lambda i: i["mass_g"])
    target = np.array(ref[:2], float)
    delta = (target - cog[:2]) * mass / b["mass_g"]
    res = {"battery": b["name"], "battery_g": round(b["mass_g"], 1), "needed_shift_mm": [round(float(v), 1) for v in delta],
           "target_xy_mm": [round(float(v), 2) for v in target]}
    tr = b.get("travel")
    if tr:
        lo, hi = np.array(tr["min"][:2], float), np.array(tr["max"][:2], float)
        got = np.clip(delta, lo, hi)
        res.update({"travel_mm": {"min": [round(float(v), 1) for v in lo], "max": [round(float(v), 1) for v in hi]},
                    "achievable_shift_mm": [round(float(v), 1) for v in got],
                    "residual_cog_offset_mm": [round(float(v), 2) for v in (cog[:2] + got * b["mass_g"] / mass - target)],
                    "fully_corrects": bool(np.abs(delta - got).max() < 1.0)})
    else:
        res["note"] = "the battery's free travel inside its cavity is unknown"
    return res


def brief_text(a):
    """A short paragraph for generation prompts: the numbers a designer of arms / motor mounts / the frame must respect."""
    if not a or not a.get("cog_mm"):
        return ""
    s = [f"BALANCE (measured on the current assembly; assembly frame, mm): counted mass {a['mass']['counted_g']:.0f} g, "
         f"CoG at x={a['cog_mm'][0]:.1f}, y={a['cog_mm'][1]:.1f}, z={a['cog_mm'][2]:.1f}."]
    lp = a.get("lift_plan") or {}
    if lp.get("status") == "planned":
        pts = "; ".join(f"({m['x']:.0f}, {m['y']:.0f})" for m in lp["motors"])
        s.append(f"Place the {lp['n']} lift motors on a circle of radius {lp['arm_radius_mm']:.0f} mm ({lp['layout']} layout, "
                 f"{lp['prop_diameter_mm']:.0f} mm props) centred on x={lp['centre_xy_mm'][0]:.1f}, y={lp['centre_xy_mm'][1]:.1f}: {pts}. "
                 "Mirror the arms about that centre so every motor carries an equal thrust share.")
    bs = a.get("battery_shift")
    if bs and bs.get("achievable_shift_mm"):
        s.append(f"The battery can slide {bs['achievable_shift_mm'][0]:+.0f} mm along X to trim the CoG.")
    return " ".join(s)
