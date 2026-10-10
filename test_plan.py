"""
LLM-driven test plans for the AI-CAD backend.

The LLM looks at the part it generated and writes a JSON *test plan*: which analyses to run, where the
part is held, where and how it is loaded, which material, and what counts as a pass. This module turns
such a plan into a concrete load case on the real B-rep faces (selected GEOMETRICALLY, never by face
index) and turns the solver output back into a compact verdict for the LLM.

No SimScale / build123d imports here, so everything is unit-testable. `describe_faces()` only uses
duck typing on the shape object it is given.

Supported now : test type "static_stress"; constraint "fixed"; loads "force" (vector or magnitude +
                direction) and "pressure" (planar faces only).
Planned       : modal, thermal, CFD, buckling, fatigue (same plan/selector/result format).
"""
import json
import math
import re

PLAN_VERSION = 1
SUPPORTED_TESTS = ("static_stress", "buckling", "thermal_stress")

# Strength retention vs temperature, as a fraction of the room-temperature yield strength. APPROXIMATE handbook trends
# for typical heat-treated conditions - NOT supplier data. Materials without an entry cannot be used in thermal tests.
HOT_YIELD = {
    "aluminum_6061":  [(20, 1.00), (100, 0.95), (150, 0.85), (200, 0.50), (250, 0.20), (300, 0.10)],
    "aluminum_7075":  [(20, 1.00), (100, 0.93), (150, 0.70), (200, 0.35), (250, 0.12), (300, 0.08)],
    "steel_4340":     [(20, 1.00), (200, 0.95), (300, 0.90), (400, 0.85), (500, 0.70), (600, 0.40)],
    "titanium_6al4v": [(20, 1.00), (100, 0.90), (200, 0.80), (300, 0.72), (400, 0.66), (500, 0.60)],
    "inconel_718":    [(20, 1.00), (200, 0.93), (400, 0.88), (600, 0.85), (650, 0.83), (700, 0.60)],
}


def yield_fraction(mat, t_c):
    """(fraction of room-temperature yield at t_c, inside_data_range). Linear interpolation between the knots."""
    pts = HOT_YIELD.get(mat)
    if not pts or t_c is None:
        return 1.0, False
    if t_c <= pts[0][0]:
        return pts[0][1], True
    for (t0, f0), (t1, f1) in zip(pts, pts[1:]):
        if t_c <= t1:
            return f0 + (f1 - f0) * (t_c - t0) / (t1 - t0), True
    return pts[-1][1], False
SIDES = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
SELECTOR_KEYS = ("at", "normal", "shape", "diameter_mm", "area_mm2", "inside_box_mm", "pick", "count")


class PlanError(ValueError):
    """The plan (or one of its face selectors) is unusable. The message is written for the LLM so it can fix it."""


# --------------------------------------------------------------------------------------------------
# text shown to the LLM
# --------------------------------------------------------------------------------------------------
EXAMPLE_PLAN = {
    "version": PLAN_VERSION,
    "tests": [{
        "id": "mount_static",
        "type": "static_stress",
        "material": "aluminum_6061",
        "rationale": "Bracket bolted through two 6 mm holes in its base, 500 N downward on the top face.",
        "constraints": [{"id": "bolts", "type": "fixed", "faces": {"shape": "cylinder", "diameter_mm": [5, 7], "at": "min_z"}}],
        "loads": [{"id": "payload", "type": "force", "faces": {"at": "max_z", "normal": "+z"},
                   "force_n": 500, "direction": "-z"}],
        "criteria": {"min_safety_factor": 2.0, "max_deflection_mm": 0.5},
    }],
}


def schema_prompt(materials):
    """Instruction block to append to the part-generation prompt."""
    return (
        "After the part, output a TEST PLAN as JSON (no prose) describing how this part must be verified.\n"
        "Think about the real use: where is it held, what loads act on it, how big, what must not happen.\n"
        "Never use face indices. Select faces geometrically; all selector keys are optional and AND-combined:\n"
        '  "at": one of min_x,max_x,min_y,max_y,min_z,max_z   (plane: lies on that side of the part bounding box;\n'
        '                                                      cylinder/hole: the hole reaches that side of the part)\n'
        '  "normal": one of +x,-x,+y,-y,+z,-z                (PLANAR faces only - holes have no normal, never use it for them)\n'
        '  "shape": "plane" | "cylinder"                      (cylinder = holes, bores, pins)\n'
        '  "diameter_mm": [min, max]                          (for cylinders, approximate)\n'
        '  Holes: select with shape "cylinder" + diameter_mm (+ optional "at" or "inside_box_mm"); do not add "normal".\n'
        '  "area_mm2": [min, max]\n'
        '  "inside_box_mm": [x0,y0,z0,x1,y1,z1]               (face centre inside this box, mm)\n'
        '  "pick": "largest" | "smallest", "count": n          (applied last)\n'
        'BOLTED / SCREWED parts: fix the bolt holes (shape "cylinder" + diameter_mm), NEVER the whole mounting plane - a fully\n'
        '  fixed underside is far too stiff and hides real stress. Fix a plane only for parts that are clamped, glued or\n'
        '  welded flat, or have no holes there, and say so in the rationale.\n'
        f"Supported test types: {', '.join(SUPPORTED_TESTS)}.  Constraint: type \"fixed\".\n"
        '  "static_stress": strength under the loads (constraints + loads).\n'
        '  "buckling": for slender members in COMPRESSION (columns, struts, long thin ribs, thin walls). Same constraints and\n'
        '     loads as static_stress; the load you give is the reference load and the result is a buckling factor = critical\n'
        '     load / reference load. Criterion "min_buckling_factor" (default 3). Use the real compressive load.\n'
        '  "thermal_stress": ONLY when the request states temperatures or a heat input. Fields: "temperatures": [{"id","faces",\n'
        '     "temperature_c"}] (at least one), optional "heat_inputs": [{"id","faces","power_w"}] (heat INTO the part, W),\n'
        '     optional "reference_temperature_c" (stress-free temperature, default 20), "constraints" (required: the part must\n'
        '     be held), optional mechanical "loads". Use ONLY temperatures and powers stated in the request - never invent\n'
        f"     them. Criteria: \"min_safety_factor\" (judged against the strength at the hot spot's temperature), \"max_temperature_c\",\n"
        f"     \"max_deflection_mm\". Materials with hot-strength data: {', '.join(sorted(HOT_YIELD))}.\n"
        'Loads: type "force" with either "vector_n":[fx,fy,fz] or "force_n" + "direction" (+x..-z),\n'
        '       or type "pressure" with "pressure_mpa" on planar faces (acts into the part).\n'
        f"Materials: {', '.join(sorted(materials))}.\n"
        'Criteria (all optional): "min_safety_factor", "max_deflection_mm", "max_von_mises_mpa", "min_buckling_factor", "max_temperature_c".\n'
        "Use SI-consistent mm / N / MPa. Example:\n" + json.dumps(EXAMPLE_PLAN, indent=1)
    )


# --------------------------------------------------------------------------------------------------
# face descriptors (duck-typed build123d access)
# --------------------------------------------------------------------------------------------------
def _vec(v):
    return [float(v.X), float(v.Y), float(v.Z)]


def _call(obj, name, *a):
    attr = getattr(obj, name)
    return attr(*a) if callable(attr) else attr


def _geom_name(face):
    try:
        return str(_call(face, "geom_type")).split(".")[-1].upper()
    except Exception:
        return "OTHER"


def _cyl_diameter(face, ext):
    try:
        r = _call(face, "radius")
        if r:
            return 2.0 * float(r)
    except Exception:
        pass
    e0, e1, e2 = sorted(ext)
    if e2 > 0 and abs(e1 - e2) <= 0.1 * e2:          # hole / short boss: the two big extents are the diameter
        return 0.5 * (e1 + e2)
    if e1 > 0 and abs(e0 - e1) <= 0.1 * e1:          # long pin: the two small extents are the diameter
        return 0.5 * (e0 + e1)
    return e2


def _sample_points(face, n=40):
    pts = []
    try:
        verts, _tris = face.tessellate(0.5)
        pts = [_vec(v) for v in verts]
    except Exception:
        pass
    if not pts:
        try:
            pts = [_vec(_call(face, "center"))]
        except Exception:
            pts = []
    if len(pts) > n:
        step = len(pts) / float(n)
        pts = [pts[int(i * step)] for i in range(n)]
    return pts


def _full_cyl_center(center, bmin, bmax, area, diameter):
    """A closed cylindrical face (a bore, a pin) reports a centre on its surface in some kernels, which puts bore
    selectors and hole-proximity checks off by one radius. When area ~ pi*d*length the face is a full cylinder and its
    bbox centre is its true axis centre."""
    if not diameter or area <= 0:
        return center
    ext = [bmax[k] - bmin[k] for k in range(3)]
    if any(abs(area - math.pi * diameter * e) <= 0.05 * area for e in ext):
        return [(bmin[k] + bmax[k]) / 2 for k in range(3)]
    return center


def describe_faces(shape):
    """[{idx, area, center, bbox_min, bbox_max, normal|None, geom, diameter|None, samples}] in B-rep order (mm)."""
    out = []
    for i, f in enumerate(list(shape.faces())):
        fb = f.bounding_box()
        bmin, bmax = _vec(fb.min), _vec(fb.max)
        geom = _geom_name(f)
        ext = [bmax[k] - bmin[k] for k in range(3)]
        try:
            center = _vec(_call(f, "center"))
        except Exception:
            center = [(bmin[k] + bmax[k]) / 2 for k in range(3)]
        normal = None
        if "PLANE" in geom:
            try:
                normal = _vec(f.normal_at())
            except Exception:
                normal = None
        try:
            area = float(_call(f, "area"))
        except Exception:
            area = 0.0
        _dia = _cyl_diameter(f, ext) if "CYLINDER" in geom else None
        center = _full_cyl_center(center, bmin, bmax, area, _dia)
        out.append({"idx": i, "area": area, "center": center, "bbox_min": bmin, "bbox_max": bmax,
                    "normal": normal, "geom": "PLANE" if "PLANE" in geom else ("CYLINDER" if "CYLINDER" in geom else "OTHER"),
                    "diameter": _dia,
                    "samples": _sample_points(f)})
    return out


def part_bbox(descs):
    return ([min(d["bbox_min"][k] for d in descs) for k in range(3)],
            [max(d["bbox_max"][k] for d in descs) for k in range(3)])


def min_thickness(descs):
    """Smallest wall thickness we can see: the gap between two parallel, opposite-facing flat faces that overlap.
    Used to size the FEM mesh (at least ~2 element layers through it). None when no such pair exists."""
    pl = [d for d in descs if d["geom"] == "PLANE" and d["normal"]]
    best = None
    for i, a in enumerate(pl):
        for b in pl[i + 1:]:
            if sum(x * y for x, y in zip(a["normal"], b["normal"])) > -0.95:
                continue
            inplane = [k for k in range(3) if abs(a["normal"][k]) < 0.5]
            if not all(min(a["bbox_max"][k], b["bbox_max"][k]) - max(a["bbox_min"][k], b["bbox_min"][k]) > 0.5
                       for k in inplane):
                continue
            t = abs(sum(a["normal"][k] * (b["center"][k] - a["center"][k]) for k in range(3)))
            if t > 0.3 and (best is None or t < best):
                best = t
    return best


def face_summary(descs, limit=24):
    """Compact description of the faces so the LLM can write better selectors (shown on selector errors)."""
    lo, hi = part_bbox(descs)
    rows = []
    for d in sorted(descs, key=lambda d: -d["area"])[:limit]:
        row = {"area_mm2": round(d["area"], 1), "shape": d["geom"].lower(),
               "center_mm": [round(c, 1) for c in d["center"]]}
        if d["normal"]:
            row["normal"] = [round(c, 2) for c in d["normal"]]
        if d["diameter"]:
            row["diameter_mm"] = round(d["diameter"], 1)
        rows.append(row)
    return {"bbox_mm": {"min": [round(v, 2) for v in lo], "max": [round(v, 2) for v in hi]},
            "n_faces": len(descs), "largest_faces": rows}


# --------------------------------------------------------------------------------------------------
# issue localization: turn "hotspot at (x,y,z)" into words the LLM can act on
# --------------------------------------------------------------------------------------------------
_AXN = "xyz"


def _facing(n):
    """'+Z' for an axis-aligned normal, else the rounded vector."""
    k = max(range(3), key=lambda i: abs(n[i]))
    if abs(n[k]) >= 0.94:
        return ("+" if n[k] > 0 else "-") + _AXN[k].upper()
    return "(" + ", ".join(f"{v:.2f}" for v in n) + ")"


def _box_dist(p, d):
    """Distance from point p to the face's bounding box (0 when inside)."""
    return math.sqrt(sum(max(d["bbox_min"][k] - p[k], 0.0, p[k] - d["bbox_max"][k]) ** 2 for k in range(3)))


def _face_label(d, roles):
    if d["geom"] == "CYLINDER" and d.get("diameter"):
        txt = f"hole/bore dia {d['diameter']:.1f} mm at ({d['center'][0]:.1f}, {d['center'][1]:.1f}, {d['center'][2]:.1f})"
    elif d["geom"] == "PLANE" and d["normal"]:
        sp = ", ".join(f"{_AXN[k]} {d['bbox_min'][k]:.0f}-{d['bbox_max'][k]:.0f}" if d["bbox_max"][k] - d["bbox_min"][k] > 0.5
                       else f"{_AXN[k]}={d['bbox_min'][k]:.0f}" for k in range(3))
        txt = f"flat face facing {_facing(d['normal'])} ({sp})"
    else:
        txt = f"curved/other face near ({d['center'][0]:.0f}, {d['center'][1]:.0f}, {d['center'][2]:.0f})"
    r = roles.get(d["idx"])
    return txt + (f" [{r}]" if r else "")


def localize(descs, case, hotspot, zone=None, metrics=None):
    """Describe WHERE the peak stress is, in terms of part features, and what usually causes that. Pure geometry;
    never raises (returns None when it cannot say anything)."""
    try:
        if not hotspot or len(hotspot) != 3 or not descs:
            return None
        p = [float(v) for v in hotspot]
        lo, hi = part_bbox(descs)
        ext = [max(hi[k] - lo[k], 1e-9) for k in range(3)]
        tol = max(1.0, 0.03 * max(ext))
        roles = {}
        for i in (case or {}).get("fixed_idx", []):
            roles[i] = "FIXED"
        for l in (case or {}).get("loads", []):
            for i in l.get("face_idx", []):
                roles[i] = f"LOADED by {l.get('id', 'load')}"

        # faces the hotspot lies on (flat faces: within tol of the plane and inside its box)
        on = []
        for d in descs:
            if d["geom"] == "PLANE" and d["normal"]:
                dist = abs(sum(d["normal"][k] * (p[k] - d["center"][k]) for k in range(3)))
                if dist <= tol and _box_dist(p, d) <= tol:
                    on.append(d)
        # edge / corner: two or more flat faces that are not parallel
        edge = None
        for i, a in enumerate(on):
            for b in on[i + 1:]:
                if abs(sum(a["normal"][k] * b["normal"][k] for k in range(3))) < 0.9:
                    edge = (a, b)
                    break
            if edge:
                break

        # nearest hole and its distance from the hole wall
        holes = [d for d in descs if d["geom"] == "CYLINDER" and d.get("diameter")]
        near_hole = None
        if holes:
            def hd(d):
                return max(0.0, math.dist(p, d["center"]) - d["diameter"] / 2)
            h = min(holes, key=hd)
            near_hole = (h, hd(h))

        def role_dist(tag):
            best = None
            for i, r in roles.items():
                if r.startswith(tag):
                    dd = _box_dist(p, descs[i])
                    if best is None or dd < best[1]:
                        best = (descs[i], dd)
            return best
        d_fixed, d_load = role_dist("FIXED"), role_dist("LOADED")

        # section thickness at the hotspot: distance to the opposite-facing flat face behind it
        thick = None
        for a in on:
            k = max(range(3), key=lambda i: abs(a["normal"][i]))
            for o in descs:
                if o is a or o["geom"] != "PLANE" or not o["normal"]:
                    continue
                if sum(a["normal"][j] * o["normal"][j] for j in range(3)) > -0.9:
                    continue
                if all(o["bbox_min"][j] - tol <= p[j] <= o["bbox_max"][j] + tol for j in range(3) if j != k):
                    t = abs(p[k] - o["center"][k])
                    if t > 0.3 and (thick is None or t < thick):
                        thick = t

        if thick and thick > 0.5 * max(ext):      # that is a length through the part, not a wall thickness
            thick = None

        # where on the part
        pos = []
        for k in range(3):
            f = (p[k] - lo[k]) / ext[k]
            if p[k] - lo[k] <= tol:
                pos.append(f"at the min-{_AXN[k].upper()} side")
            elif hi[k] - p[k] <= tol:
                pos.append(f"at the max-{_AXN[k].upper()} side")
            else:
                pos.append(f"{f * 100:.0f}% along {_AXN[k].upper()}")
        out = {"hotspot_mm": [round(v, 1) for v in p], "position": "; ".join(pos),
               "on_faces": [_face_label(d, roles) for d in on],
               "at_edge_between": [_face_label(edge[0], roles), _face_label(edge[1], roles)] if edge else None,
               "section_thickness_mm": round(thick, 1) if thick else None,
               "nearest_hole": ({"label": _face_label(near_hole[0], roles), "gap_mm": round(near_hole[1], 1)}
                                if near_hole else None),
               "dist_to_fixed_mm": round(d_fixed[1], 1) if d_fixed else None,
               "dist_to_load_mm": round(d_load[1], 1) if d_load else None}
        if zone and zone.get("min") and zone.get("max"):
            out["hot_zone"] = ("stress within 15% of the peak spans " +
                               ", ".join(f"{_AXN[k]} {zone['min'][k]:.0f}-{zone['max'][k]:.0f}" for k in range(3)) +
                               f" mm ({zone.get('n_nodes', '?')} of {zone.get('n_total', '?')} nodes)")

        # likely cause, from where it sits
        hints = []
        t_ref = thick or 5.0
        if near_hole and near_hole[1] <= max(0.6 * near_hole[0]["diameter"], 2.0):
            if roles.get(near_hole[0]["idx"]) == "FIXED":
                hints.append("Peak is at the wall of a FIXED hole: part of it is the rigid-hole constraint, but the real "
                             "fix is more material around the mounting holes (thicker section / boss / pad, larger "
                             "edge distance).")
            else:
                hints.append("Peak is at the edge of a hole: stress concentration (about 2-3x nominal). Add material "
                             "around it (boss/pad/thicker wall), increase edge distance, or move it away from the "
                             "high-stress path.")
        if edge:
            hints.append("Peak sits on an edge/corner between two faces: if it is an inside corner, add a fillet of "
                         f"radius >= {max(2.0, 0.5 * t_ref):.0f} mm there (sharp inside corners multiply stress); if it is an "
                         "outer edge the real driver is section bending - see below.")
        if d_load and d_load[1] <= max(2.0, 0.05 * max(ext)):
            hints.append("Peak is where the load enters: spread the load over a larger pad or thicken locally.")
        elif d_fixed and d_fixed[1] <= max(2.0, 0.05 * max(ext)) and not (near_hole and roles.get(near_hole[0]['idx']) == 'FIXED'):
            hints.append("Peak is at the fixed support: partly a constraint effect; add a fillet or a thicker "
                         "section where the part meets the support.")
        if not hints or (d_fixed and d_load and d_fixed[1] > 2 * t_ref and d_load[1] > 2 * t_ref):
            hints.append(f"Peak is away from holes/supports/load: it is bending of a thin section"
                         + (f" (about {thick:.1f} mm thick here)" if thick else "")
                         + ". Bending stress scales with 1/thickness^2: thicken THIS section, or add a rib/gusset "
                           "along the span between the support and the load.")
        out["likely_cause"] = hints

        lines = [f"WHERE: peak stress at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm = {out['position']}."]
        if on:
            lines.append("  On: " + " AND ".join(out["on_faces"]) + ".")
        if edge:
            lines.append("  It is on the edge/corner between those two faces.")
        if thick:
            lines.append(f"  Section thickness at this spot is about {thick:.1f} mm.")
        if near_hole:
            lines.append(f"  Nearest hole wall is {near_hole[1]:.1f} mm away: {out['nearest_hole']['label']}.")
        if d_fixed:
            lines.append(f"  {d_fixed[1]:.0f} mm from the fixed support.")
        if d_load:
            lines.append(f"  {d_load[1]:.0f} mm from the loaded face.")
        if out.get("hot_zone"):
            lines.append("  " + out["hot_zone"][0].upper() + out["hot_zone"][1:] + ".")
        for h in hints:
            lines.append("  LIKELY CAUSE/FIX: " + h)
        out["text"] = "\n".join(lines)
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------
# face selection
# --------------------------------------------------------------------------------------------------
def _in_range(v, rng):
    return rng[0] <= v <= rng[1]


def select_faces(descs, selector, label="selector"):
    """Indices of the faces matching `selector`. Raises PlanError (LLM-readable) when nothing / garbage matches."""
    if not isinstance(selector, dict) or not selector:
        raise PlanError(f"{label}: faces must be a non-empty object of selector keys {list(SELECTOR_KEYS)}.")
    bad = [k for k in selector if k not in SELECTOR_KEYS]
    if bad:
        raise PlanError(f"{label}: unknown selector key(s) {bad}; allowed: {list(SELECTOR_KEYS)}. "
                        "Face indices are not allowed - select geometrically.")
    lo, hi = part_bbox(descs)
    span = max(hi[k] - lo[k] for k in range(3))
    tol = max(span * 2e-3, 1e-3)
    cand = list(descs)
    want = None

    if "shape" in selector:
        want = str(selector["shape"]).upper().replace("PLANAR", "PLANE").replace("CYLINDRICAL", "CYLINDER")
        if want not in ("PLANE", "CYLINDER"):
            raise PlanError(f"{label}: shape must be 'plane' or 'cylinder'.")
        cand = [d for d in cand if d["geom"] == want]
    if "normal" in selector:
        key = str(selector["normal"]).lower()
        if key not in DIRS:
            raise PlanError(f"{label}: normal must be one of {list(DIRS)}.")
        if want != "CYLINDER":      # a hole/bore has no single face normal - for cylinders "normal" is simply ignored
            t = DIRS[key]
            cand = [d for d in cand if d["normal"] and sum(a * b for a, b in zip(d["normal"], t)) >= math.cos(math.radians(20))]
    if "at" in selector:
        side = str(selector["at"]).lower()
        if side not in SIDES:
            raise PlanError(f"{label}: at must be one of {list(SIDES)}.")
        ax = "xyz".index(side[-1])
        want_hi = side.startswith("max")
        ref = hi[ax] if want_hi else lo[ax]
        def on_side(d):
            if d["geom"] == "CYLINDER":     # a hole is not a thin face: "at" = the hole reaches that side of the part
                return (ref - d["bbox_max"][ax] <= tol) if want_hi else (d["bbox_min"][ax] - ref <= tol)
            thin = d["bbox_max"][ax] - d["bbox_min"][ax] <= tol
            pos = d["bbox_max"][ax] if want_hi else d["bbox_min"][ax]
            return thin and abs(pos - ref) <= tol
        cand = [d for d in cand if on_side(d)]
    if "diameter_mm" in selector:
        rng = selector["diameter_mm"]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise PlanError(f"{label}: diameter_mm must be [min, max].")
        cand = [d for d in cand if d["diameter"] is not None and _in_range(d["diameter"], rng)]
    if "area_mm2" in selector:
        rng = selector["area_mm2"]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise PlanError(f"{label}: area_mm2 must be [min, max].")
        cand = [d for d in cand if _in_range(d["area"], rng)]
    if "inside_box_mm" in selector:
        b = selector["inside_box_mm"]
        if not (isinstance(b, (list, tuple)) and len(b) == 6):
            raise PlanError(f"{label}: inside_box_mm must be [x0,y0,z0,x1,y1,z1].")
        cand = [d for d in cand if all(min(b[k], b[k + 3]) <= d["center"][k] <= max(b[k], b[k + 3]) for k in range(3))]
    if "pick" in selector:
        pick = str(selector["pick"]).lower()
        if pick not in ("largest", "smallest"):
            raise PlanError(f"{label}: pick must be 'largest' or 'smallest'.")
        n = int(selector.get("count", 1) or 1)
        cand = sorted(cand, key=lambda d: d["area"], reverse=(pick == "largest"))[:max(n, 1)]

    if not cand:
        raise PlanError(f"{label}: selector {json.dumps(selector)} matched no faces. Part geometry: "
                        + json.dumps(face_summary(descs, 12)))
    return sorted(d["idx"] for d in cand)


# --------------------------------------------------------------------------------------------------
# plan validation + load case
# --------------------------------------------------------------------------------------------------
def _vec_from_load(load, label):
    if "vector_n" in load:
        v = load["vector_n"]
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise PlanError(f"{label}: vector_n must be [fx, fy, fz] in newtons.")
        return [float(x) for x in v]
    if "force_n" in load:
        d = str(load.get("direction", "")).lower()
        if d not in DIRS:
            raise PlanError(f"{label}: direction must be one of {list(DIRS)} when force_n is used.")
        m = float(load["force_n"])
        return [m * c for c in DIRS[d]]
    raise PlanError(f"{label}: give either vector_n [fx,fy,fz] or force_n + direction.")


def parse_plan(plan):
    """Accept a dict or JSON string (tolerates ```json fences). Returns the dict or raises PlanError."""
    if isinstance(plan, str):
        txt = re.sub(r"^```(?:json)?|```$", "", plan.strip(), flags=re.M).strip()
        try:
            plan = json.loads(txt)
        except Exception as e:
            raise PlanError(f"test plan is not valid JSON: {e}")
    if not isinstance(plan, dict) or not isinstance(plan.get("tests"), list) or not plan["tests"]:
        raise PlanError('test plan must be an object with a non-empty "tests" list.')
    return plan


_BOLT_WORDS = re.compile(r"\b(bolt\w*|screw\w*|fasten\w*|rivet\w*|mounting[- ]holes?|through[- ]holes?|dowel\w*)", re.I)
_CLAMP_WORDS = re.compile(r"\b(clamp\w*|glu\w*|bond\w*|weld\w*|adhesive|fully fixed|rigid(ly)? (mounted|fixed))", re.I)


_FORCE_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(kN|N)(?![\w/\u00b7*])")
_AXIS_RE = re.compile(r"(?:\(\s*([+-])\s*([XYZxyz])\s*\)|([+-])\s*([XYZxyz])\s*(?:direction|axis))")


_TEMP_PATTERNS = (re.compile(r"(-?\d+(?:\.\d+)?)\s*[\u00b0\u00ba]\s*[Cc]"),
                  re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg)\s*(?:C|celsius)", re.I),
                  re.compile(r"(-?\d+(?:\.\d+)?)\s*celsius", re.I),
                  re.compile(r"(-?\d+(?:\.\d+)?)\s*C(?![A-Za-z0-9])"))
_KELVIN_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*K(?![A-Za-z0-9])")
_POWER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(kW|W)(?![A-Za-z0-9/])")


def _stated_temps(text):
    out = []
    for pat in _TEMP_PATTERNS:
        out += [float(m) for m in pat.findall(text or "")]
    out += [float(m) - 273.15 for m in _KELVIN_RE.findall(text or "")]
    return out


def _stated_powers(text):
    out = []
    for num, unit in _POWER_RE.findall(text or ""):
        try:
            out.append(float(num.replace(",", "")) * (1000.0 if unit == "kW" else 1.0))
        except ValueError:
            pass
    return out


def check_plan_against_request(plan, request_text):
    """The LLM plan must apply the force the request states, along the axis it states. Raises PlanError (fed back to
    the LLM) when it does not - a plan with a different load verifies a different part than the one asked for."""
    text = request_text or ""
    stated = []
    for num, unit in _FORCE_RE.findall(text):
        try:
            v = float(num.replace(",", "")) * (1000.0 if unit == "kN" else 1.0)
        except ValueError:
            continue
        if v > 0 and v not in stated:
            stated.append(v)
    dirs = {(m[0] or m[2]) + (m[1] or m[3]).lower() for m in _AXIS_RE.findall(text)}
    if re.search(r"\bdownward\b", text, re.I) and not dirs:
        dirs = {"-z"}
    for t in (plan.get("tests") or []):
        tid = t.get("id", "test")
        mags, axes = [], set()
        for ld in (t.get("loads") or []):
            f = ld.get("force_n")
            if isinstance(f, (list, tuple)) and len(f) == 3:
                m = math.sqrt(sum(float(c) ** 2 for c in f))
                if m > 0:
                    k = max(range(3), key=lambda j: abs(float(f[j])))
                    axes.add(("+" if float(f[k]) > 0 else "-") + "xyz"[k])
                mags.append(m)
            elif f is not None:
                try:
                    mags.append(abs(float(f)))
                except (TypeError, ValueError):
                    continue
                mm = re.search(r"([+-])?\s*([xyz])", str(ld.get("direction", "")).lower())
                if mm:
                    axes.add((mm.group(1) or "+") + mm.group(2))
        if t.get("type") == "thermal_stress":              # every temperature / power must come from the request
            st_t = _stated_temps(text)
            for tp in (t.get("temperatures") or []):
                try:
                    v = float(tp.get("temperature_c"))
                except (TypeError, ValueError):
                    continue
                if not st_t:
                    raise PlanError(f"{tid}: the request states no temperatures, so a thermal_stress test would rest on "
                                    "invented numbers. Remove the thermal test (or state the temperatures in the request).")
                if not any(abs(v - x) <= 1.5 for x in st_t):
                    raise PlanError(f"{tid}: temperature {v:g} C is not in the request (it states "
                                    f"{', '.join(f'{x:g}' for x in sorted(set(st_t)))} C). Use only stated temperatures.")
            st_p = _stated_powers(text)
            for hp in (t.get("heat_inputs") or []):
                try:
                    v = abs(float(hp.get("power_w")))
                except (TypeError, ValueError):
                    continue
                if not st_p or not any(abs(v - x) <= 0.05 * x for x in st_p):
                    raise PlanError(f"{tid}: heat input {v:g} W is not stated in the request"
                                    + (f" (it states {', '.join(f'{x:g}' for x in sorted(set(st_p)))} W)" if st_p else "")
                                    + ". Use only stated powers, or remove the heat input.")
        if stated and mags:
            total = sum(mags)
            ok_vals = stated + ([sum(stated)] if len(stated) > 1 else [])
            if not any(abs(total - v) <= 0.05 * v for v in ok_vals):
                raise PlanError(f"{tid}: the request states a force of {', '.join(f'{v:g}' for v in stated)} N but the plan "
                                f"applies {total:g} N. Use exactly the stated force.")
        if len(dirs) == 1 and axes:
            want = next(iter(dirs))
            if want not in axes:
                raise PlanError(f"{tid}: the request says the load acts along {want} but the plan uses "
                                f"{', '.join(sorted(axes))}. Use the stated direction ({want}).")


def _check_bolt_intent(tid, test, descs, fixed_idx):
    """The rationale says 'bolted' but the constraint clamps a whole plane that has holes in it: the model would be far
    stiffer than the real joint and under-report stress. Raise a PlanError that tells the LLM exactly what to select."""
    text = " ".join([str(test.get("rationale", "")), str(test.get("id", ""))]
                    + [str(c.get("id", "")) for c in (test.get("constraints") or [])])
    if not _BOLT_WORDS.search(text) or _CLAMP_WORDS.search(text):
        return
    fixed = [descs[i] for i in fixed_idx]
    if not fixed or any(d["geom"] != "PLANE" for d in fixed):
        return                                    # already constrains holes (or something non-planar)
    lo, hi = part_bbox(descs)
    span = max(hi[k] - lo[k] for k in range(3))
    tol = max(span * 2e-3, 1e-3)
    holes = []
    for h in descs:
        if h["geom"] != "CYLINDER" or h["diameter"] is None:
            continue
        for d in fixed:
            ax = max(range(3), key=lambda k: abs((d["normal"] or (0, 0, 0))[k]))
            pos = d["bbox_min"][ax]
            if h["bbox_min"][ax] - tol <= pos <= h["bbox_max"][ax] + tol:
                holes.append(h)
                break
    if not holes:
        return                                    # nothing to bolt through on that face - a plane is all we can do
    dia = sorted({round(h["diameter"], 1) for h in holes})
    sides = sorted({s for s in SIDES
                    if any(((h["bbox_max"]["xyz".index(s[-1])] >= hi["xyz".index(s[-1])] - tol) if s.startswith("max")
                            else (h["bbox_min"]["xyz".index(s[-1])] <= lo["xyz".index(s[-1])] + tol)) for h in holes)})
    area = sum(d["area"] for d in fixed)
    raise PlanError(
        f"{tid}: the rationale says the part is bolted/mounted, but the fixed constraint clamps a whole plane "
        f"({area:.0f} mm2) that has {len(holes)} hole(s) in it. That is far stiffer than bolts and hides real stress. "
        f"Fix the bolt holes instead: faces {{\"shape\": \"cylinder\", \"diameter_mm\": [{dia[0] - 0.5:g}, {dia[-1] + 0.5:g}]"
        + (f", \"at\": \"{sides[0]}\"" if len(sides) == 1 else "")
        + "}. If the part really is clamped/glued flat, say 'clamped' in the rationale.")


def build_load_case(test, descs, materials, default_material="aluminum_6061"):
    """Concrete load case for one static_stress test. Raises PlanError for anything unusable."""
    tid = str(test.get("id", "test"))
    if test.get("type") not in SUPPORTED_TESTS:
        raise PlanError(f"{tid}: test type {test.get('type')!r} is not supported yet; supported: {list(SUPPORTED_TESTS)}.")
    mat = test.get("material") or default_material
    if mat not in materials:
        raise PlanError(f"{tid}: unknown material {mat!r}; choose one of {sorted(materials)}.")
    ttype = test.get("type")
    cons = test.get("constraints") or []
    loads = test.get("loads") or []
    temps_in = test.get("temperatures") or []
    heat_in = test.get("heat_inputs") or []
    if ttype in ("static_stress", "buckling"):
        if not cons or not loads:
            raise PlanError(f"{tid}: a {ttype} test needs at least one fixed constraint and one load.")
        if temps_in or heat_in:
            raise PlanError(f"{tid}: temperatures/heat_inputs only belong in a thermal_stress test.")
    else:                                                  # thermal_stress
        if not cons:
            raise PlanError(f"{tid}: a thermal_stress test needs at least one fixed constraint (the part must be held somewhere).")
        if not temps_in:
            raise PlanError(f"{tid}: a thermal_stress test needs at least one entry in \"temperatures\" (heat has to be able to leave).")
        if mat not in HOT_YIELD:
            raise PlanError(f"{tid}: no hot-strength data for material {mat!r}; thermal tests support {sorted(HOT_YIELD)}.")
        mrec = materials[mat]
        if not mrec.get("thermal_conductivity") or not mrec.get("thermal_expansion_per_c"):
            raise PlanError(f"{tid}: material {mat!r} has no thermal conductivity / expansion data.")

    fixed_idx = []
    for c in cons:
        if c.get("type", "fixed") != "fixed":
            raise PlanError(f"{tid}: constraint type {c.get('type')!r} unsupported; only 'fixed'.")
        fixed_idx += select_faces(descs, c.get("faces"), f"{tid}/constraint {c.get('id', '?')}")
    fixed_idx = sorted(set(fixed_idx))
    _check_bolt_intent(tid, test, descs, fixed_idx)

    load_list = []
    for ld in loads:
        lid = f"{tid}/load {ld.get('id', '?')}"
        idx = select_faces(descs, ld.get("faces"), lid)
        kind = ld.get("type", "force")
        if kind == "force":
            vec = _vec_from_load(ld, lid)
        elif kind == "pressure":
            p = float(ld.get("pressure_mpa", 0))
            vec = [0.0, 0.0, 0.0]
            for i in idx:
                d = descs[i]
                if not d["normal"]:
                    raise PlanError(f"{lid}: pressure is only supported on planar faces.")
                for k in range(3):
                    vec[k] += -d["normal"][k] * p * d["area"]
        else:
            raise PlanError(f"{lid}: load type {kind!r} unsupported; use 'force' or 'pressure'.")
        if not any(abs(v) > 1e-9 for v in vec):
            raise PlanError(f"{lid}: resulting force is zero.")
        load_list.append({"id": ld.get("id", "load"), "face_idx": idx, "force_xyz": vec})

    overlap = set(fixed_idx) & {i for l in load_list for i in l["face_idx"]}
    if overlap:
        raise PlanError(f"{tid}: {len(overlap)} face(s) are both fixed and loaded - choose different faces.")

    temp_list, flux_list = [], []
    for t in temps_in:
        lid = f"{tid}/temperature {t.get('id', '?')}"
        try:
            tc = float(t.get("temperature_c"))
        except (TypeError, ValueError):
            raise PlanError(f"{lid}: needs a numeric \"temperature_c\".")
        if not -200.0 <= tc <= 1500.0:
            raise PlanError(f"{lid}: temperature_c {tc:g} is outside -200..1500 C.")
        temp_list.append({"id": t.get("id", "temp"), "face_idx": select_faces(descs, t.get("faces"), lid), "temperature_c": tc})
    for h in heat_in:
        lid = f"{tid}/heat input {h.get('id', '?')}"
        try:
            pw = float(h.get("power_w"))
        except (TypeError, ValueError):
            raise PlanError(f"{lid}: needs a numeric \"power_w\".")
        if pw == 0 or abs(pw) > 1e6:
            raise PlanError(f"{lid}: power_w must be non-zero and below 1 MW.")
        flux_list.append({"id": h.get("id", "heat"), "face_idx": select_faces(descs, h.get("faces"), lid), "power_w": pw})
    if {i for f in flux_list for i in f["face_idx"]} & {i for t in temp_list for i in t["face_idx"]}:
        raise PlanError(f"{tid}: a face is both a heat input and a fixed-temperature face - choose different faces.")

    pts = lambda idxs: [p for i in idxs for p in descs[i]["samples"]]
    crit = test.get("criteria") or {}
    try:
        t_ref = float(test.get("reference_temperature_c", 20.0))
    except (TypeError, ValueError):
        raise PlanError(f"{tid}: reference_temperature_c must be a number.")
    return {"id": tid, "type": ttype, "material": mat, "fixed_idx": fixed_idx, "loads": load_list,
            "temps": temp_list, "fluxes": flux_list, "t_ref_c": t_ref,
            "fixed_pts_mm": pts(fixed_idx), "load_pts_mm": pts([i for l in load_list for i in l["face_idx"]]),
            "criteria": {k: float(v) for k, v in crit.items()
                         if k in ("min_safety_factor", "max_deflection_mm", "max_von_mises_mpa",
                                  "min_buckling_factor", "max_temperature_c")},
            "rationale": str(test.get("rationale", ""))[:300]}


# --------------------------------------------------------------------------------------------------
# result -> verdict for the LLM
# --------------------------------------------------------------------------------------------------
def evaluate(case, fem, diag, default_min_sf=2.0):
    """Compact, LLM-readable verdict. status: PASS | FAIL | INVALID (numbers not trustworthy) | NOT_RUN."""
    ttype = case.get("type", "static_stress")
    lc = {"fixed_faces": len(case["fixed_idx"]),
          "loads": [{"id": l["id"], "faces": len(l["face_idx"]),
                     "force_n": [round(v, 2) for v in l["force_xyz"]]} for l in case["loads"]]}
    if case.get("temps"):
        lc["temperatures"] = [{"id": t["id"], "faces": len(t["face_idx"]), "temperature_c": t["temperature_c"]}
                              for t in case["temps"]]
        lc["heat_inputs"] = [{"id": f["id"], "faces": len(f["face_idx"]), "power_w": f["power_w"]}
                             for f in case.get("fluxes", [])]
        lc["reference_temperature_c"] = case.get("t_ref_c")
    base = {"test_id": case["id"], "type": ttype, "material": case["material"], "load_case": lc}
    if fem is None:
        return {**base, "status": "NOT_RUN", "solver": (diag or {}).get("solver", "simscale"),
                "reason": (diag or {}).get("reason"), "failed_stage": (diag or {}).get("failed_stage"),
                "design_related": bool((diag or {}).get("design_related")),
                "hint": "If design_related is true the geometry itself was rejected; otherwise it is an "
                        "infrastructure problem and the design is not to blame."}
    ss = fem.get("solver_meta") or fem.get("simscale") or {}
    vm = (fem.get("stress") or {}).get("von_mises_mpa")
    sf, defl = fem.get("safety_factor"), fem.get("deflection_mm")
    warnings = []
    bc = ss.get("bc_check")
    if bc is not None and not bc.get("ok"):
        warnings.append("boundary-condition sanity check failed: " + str(bc.get("reason") or bc))
    if fem.get("numerically_suspect"):
        warnings.append("result flagged numerically suspect")
    if fem.get("hot_data_warning"):
        warnings.append(fem["hot_data_warning"])
    crit = dict(case["criteria"])
    bf = fem.get("buckling_factor")
    if ttype == "buckling":
        crit.setdefault("min_buckling_factor", 3.0)        # stress under the reference load is not a pass/fail here
        if bf is None:
            warnings.append("no positive buckling factor: the reference load is tensile or stabilizing - check that the "
                            "load pushes the slender member in compression")
    else:
        crit.setdefault("min_safety_factor", default_min_sf)
    rows = []
    if ttype == "buckling" and bf is not None and "min_buckling_factor" in crit:
        rows.append({"name": "buckling_factor", "required": f">= {crit['min_buckling_factor']}", "actual": bf,
                     "pass": bf >= crit["min_buckling_factor"]})
    if "min_safety_factor" in crit and sf is not None:
        rows.append({"name": "safety_factor", "required": f">= {crit['min_safety_factor']}", "actual": sf,
                     "pass": sf >= crit["min_safety_factor"]})
    if "max_deflection_mm" in crit and defl is not None:
        rows.append({"name": "deflection_mm", "required": f"<= {crit['max_deflection_mm']}", "actual": defl,
                     "pass": defl <= crit["max_deflection_mm"]})
    if "max_von_mises_mpa" in crit and vm is not None:
        rows.append({"name": "von_mises_mpa", "required": f"<= {crit['max_von_mises_mpa']}", "actual": vm,
                     "pass": vm <= crit["max_von_mises_mpa"]})
    tmax = fem.get("t_max_c")
    if "max_temperature_c" in crit and tmax is not None:
        rows.append({"name": "max_temperature_c", "required": f"<= {crit['max_temperature_c']}", "actual": round(tmax, 1),
                     "pass": tmax <= crit["max_temperature_c"]})
    status = "INVALID" if warnings else ("PASS" if rows and all(r["pass"] for r in rows) else "FAIL")
    metrics = {"von_mises_mpa": vm, "safety_factor": sf, "max_deflection_mm": defl,
               "hotspot_mm": (fem.get("critical_section") or {}).get("hotspot_xyz_mm"),
               "hot_zone_mm": (fem.get("critical_section") or {}).get("hot_zone_mm")}
    for k in ("buckling_factor", "buckling_factors", "buckling_mode1_peak_mm", "t_max_c", "t_min_c", "t_at_hotspot_c",
              "yield_at_hotspot_mpa"):
        if fem.get(k) is not None:
            metrics[k] = fem[k]
    out = {**base, "status": status, "solver": fem.get("solver", "simscale"), "metrics": metrics,
           "criteria": rows, "warnings": warnings,
           "mesh": {"nodes": (ss.get("mesh") or {}).get("nodes") or ((ss.get("field_diag") or {}).get("n_points") if isinstance(ss.get("field_diag"), dict) else None)}}
    if fem.get("assumptions"):
        out["assumptions"] = fem["assumptions"]
    return out


def overall(results):
    st = [r["status"] for r in results]
    for s in ("PLAN_ERROR", "INVALID", "NOT_RUN", "FAIL"):
        if s in st:
            return s
    return "RESOLVED" if st and all(x == "RESOLVED" for x in st) else "PASS"
