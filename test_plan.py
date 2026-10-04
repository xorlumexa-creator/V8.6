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
SUPPORTED_TESTS = ("static_stress",)
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
        "rationale": "Bracket bolted through its base, 500 N downward on the top face.",
        "constraints": [{"id": "base", "type": "fixed", "faces": {"at": "min_z", "normal": "-z"}}],
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
        f"Supported test types: {', '.join(SUPPORTED_TESTS)}.  Constraint: type \"fixed\".\n"
        'Loads: type "force" with either "vector_n":[fx,fy,fz] or "force_n" + "direction" (+x..-z),\n'
        '       or type "pressure" with "pressure_mpa" on planar faces (acts into the part).\n'
        f"Materials: {', '.join(sorted(materials))}.\n"
        'Criteria (all optional): "min_safety_factor", "max_deflection_mm", "max_von_mises_mpa".\n'
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
        out.append({"idx": i, "area": area, "center": center, "bbox_min": bmin, "bbox_max": bmax,
                    "normal": normal, "geom": "PLANE" if "PLANE" in geom else ("CYLINDER" if "CYLINDER" in geom else "OTHER"),
                    "diameter": _cyl_diameter(f, ext) if "CYLINDER" in geom else None,
                    "samples": _sample_points(f)})
    return out


def part_bbox(descs):
    return ([min(d["bbox_min"][k] for d in descs) for k in range(3)],
            [max(d["bbox_max"][k] for d in descs) for k in range(3)])


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


def build_load_case(test, descs, materials, default_material="aluminum_6061"):
    """Concrete load case for one static_stress test. Raises PlanError for anything unusable."""
    tid = str(test.get("id", "test"))
    if test.get("type") not in SUPPORTED_TESTS:
        raise PlanError(f"{tid}: test type {test.get('type')!r} is not supported yet; supported: {list(SUPPORTED_TESTS)}.")
    mat = test.get("material") or default_material
    if mat not in materials:
        raise PlanError(f"{tid}: unknown material {mat!r}; choose one of {sorted(materials)}.")
    cons = test.get("constraints") or []
    loads = test.get("loads") or []
    if not cons or not loads:
        raise PlanError(f"{tid}: a static test needs at least one fixed constraint and one load.")

    fixed_idx = []
    for c in cons:
        if c.get("type", "fixed") != "fixed":
            raise PlanError(f"{tid}: constraint type {c.get('type')!r} unsupported; only 'fixed'.")
        fixed_idx += select_faces(descs, c.get("faces"), f"{tid}/constraint {c.get('id', '?')}")
    fixed_idx = sorted(set(fixed_idx))

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

    pts = lambda idxs: [p for i in idxs for p in descs[i]["samples"]]
    crit = test.get("criteria") or {}
    return {"id": tid, "material": mat, "fixed_idx": fixed_idx, "loads": load_list,
            "fixed_pts_mm": pts(fixed_idx), "load_pts_mm": pts([i for l in load_list for i in l["face_idx"]]),
            "criteria": {k: float(v) for k, v in crit.items()
                         if k in ("min_safety_factor", "max_deflection_mm", "max_von_mises_mpa")},
            "rationale": str(test.get("rationale", ""))[:300]}


# --------------------------------------------------------------------------------------------------
# result -> verdict for the LLM
# --------------------------------------------------------------------------------------------------
def evaluate(case, fem, diag, default_min_sf=2.0):
    """Compact, LLM-readable verdict. status: PASS | FAIL | INVALID (numbers not trustworthy) | NOT_RUN."""
    base = {"test_id": case["id"], "type": "static_stress", "material": case["material"],
            "load_case": {"fixed_faces": len(case["fixed_idx"]),
                          "loads": [{"id": l["id"], "faces": len(l["face_idx"]),
                                     "force_n": [round(v, 2) for v in l["force_xyz"]]} for l in case["loads"]]}}
    if fem is None:
        return {**base, "status": "NOT_RUN", "solver": "simscale",
                "reason": (diag or {}).get("reason"), "failed_stage": (diag or {}).get("failed_stage"),
                "design_related": bool((diag or {}).get("design_related")),
                "hint": "If design_related is true the geometry itself was rejected; otherwise it is an "
                        "infrastructure problem and the design is not to blame."}
    ss = fem.get("simscale") or {}
    vm = (fem.get("stress") or {}).get("von_mises_mpa")
    sf, defl = fem.get("safety_factor"), fem.get("deflection_mm")
    warnings = []
    bc = ss.get("bc_check")
    if bc is not None and not bc.get("ok"):
        warnings.append("boundary-condition sanity check failed: " + str(bc.get("reason") or bc))
    if fem.get("numerically_suspect"):
        warnings.append("result flagged numerically suspect")
    crit = dict(case["criteria"])
    crit.setdefault("min_safety_factor", default_min_sf)
    rows = []
    if "min_safety_factor" in crit and sf is not None:
        rows.append({"name": "safety_factor", "required": f">= {crit['min_safety_factor']}", "actual": sf,
                     "pass": sf >= crit["min_safety_factor"]})
    if "max_deflection_mm" in crit and defl is not None:
        rows.append({"name": "deflection_mm", "required": f"<= {crit['max_deflection_mm']}", "actual": defl,
                     "pass": defl <= crit["max_deflection_mm"]})
    if "max_von_mises_mpa" in crit and vm is not None:
        rows.append({"name": "von_mises_mpa", "required": f"<= {crit['max_von_mises_mpa']}", "actual": vm,
                     "pass": vm <= crit["max_von_mises_mpa"]})
    status = "INVALID" if warnings else ("PASS" if rows and all(r["pass"] for r in rows) else "FAIL")
    return {**base, "status": status, "solver": "simscale",
            "metrics": {"von_mises_mpa": vm, "safety_factor": sf, "max_deflection_mm": defl,
                        "hotspot_mm": (fem.get("critical_section") or {}).get("hotspot_xyz_mm"),
                        "hot_zone_mm": (fem.get("critical_section") or {}).get("hot_zone_mm")},
            "criteria": rows, "warnings": warnings,
            "mesh": {"nodes": (ss.get("field_diag") or {}).get("n_points") if isinstance(ss.get("field_diag"), dict) else None}}


def overall(results):
    st = [r["status"] for r in results]
    for s in ("PLAN_ERROR", "INVALID", "NOT_RUN", "FAIL"):
        if s in st:
            return s
    return "RESOLVED" if st and all(x == "RESOLVED" for x in st) else "PASS"
