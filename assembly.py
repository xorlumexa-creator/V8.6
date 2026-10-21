"""Deterministic assembly of generated parts.

An LLM may PROPOSE mates (which feature touches which), but this module decides everything geometric: it solves the
placement, checks that hole patterns really line up, picks standard fasteners, measures the real internal space of every
part and checks that components fit in it. Nothing here asks a model to measure anything.

Pure numpy (+ scipy for the internal-space analysis); the OpenCASCADE-dependent steps live in `features_from_shape` and in
main.py, and are injected, so the logic can be tested without a CAD kernel.
"""
from __future__ import annotations

import itertools
import math
import re

import numpy as np

TOL_POS_MM = 0.3          # how closely two hole patterns must coincide to count as the same pattern
TOL_ANGLE_DEG = 1.0


class AssemblyError(Exception):
    """A mate or part description that cannot be solved. The message is written for the LLM that proposed it."""


# ---------------------------------------------------------------------------------------------------------------------
# small linear algebra
# ---------------------------------------------------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise AssemblyError("zero-length direction")
    return v / n


def rot_axis(axis, ang):
    x, y, z = unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rot_between(a, b):
    """Smallest rotation taking direction a onto direction b."""
    a, b = unit(a), unit(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return rot_axis(np.cross(a, helper), math.pi)
    v = np.cross(a, b)
    s2 = float(np.dot(v, v))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s2)


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_about(point, axis, ang):
    """Rotation by `ang` about the line through `point` with direction `axis`, as a 4x4."""
    R = rot_axis(axis, ang)
    p = np.asarray(point, float)
    return T_from(R, p - R @ p)


def pt(T, p):
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]


def dr(T, d):
    return T[:3, :3] @ np.asarray(d, float)


def plane_basis(n):
    n = unit(n)
    helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(n, helper))
    return u, np.cross(n, u)


# ---------------------------------------------------------------------------------------------------------------------
# metric hardware (ISO 4762 socket head cap screw, ISO 4032 nut, ISO 7089 washer) - standard nominal dimensions
# ---------------------------------------------------------------------------------------------------------------------
#        d    pitch tap_drill clr_fine clr_med head_d head_h  nut_h nut_af  washer(id, od, h)
_FASTENERS = {
    "M2":   (2.0,  0.40, 1.6, 2.2,  2.4,  3.8,  2.0,  1.6, 4.0,  (2.2, 5.0, 0.3)),
    "M2.5": (2.5,  0.45, 2.05, 2.7, 2.9,  4.5,  2.5,  2.0, 5.0,  (2.7, 6.0, 0.5)),
    "M3":   (3.0,  0.50, 2.5, 3.2,  3.4,  5.5,  3.0,  2.4, 5.5,  (3.2, 7.0, 0.5)),
    "M4":   (4.0,  0.70, 3.3, 4.3,  4.5,  7.0,  4.0,  3.2, 7.0,  (4.3, 9.0, 0.8)),
    "M5":   (5.0,  0.80, 4.2, 5.3,  5.5,  8.5,  5.0,  4.7, 8.0,  (5.3, 10.0, 1.0)),
    "M6":   (6.0,  1.00, 5.0, 6.4,  6.6,  10.0, 6.0,  5.2, 10.0, (6.4, 12.0, 1.6)),
    "M8":   (8.0,  1.25, 6.8, 8.4,  9.0,  13.0, 8.0,  6.8, 13.0, (8.4, 16.0, 1.6)),
    "M10":  (10.0, 1.50, 8.5, 10.5, 11.0, 16.0, 10.0, 8.4, 16.0, (10.5, 20.0, 2.0)),
}
_BOLT_LENGTHS = [4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]


def fastener_spec(size):
    s = str(size).upper().replace(" ", "")
    if s not in _FASTENERS:
        raise AssemblyError(f"unknown fastener size {size!r}; supported: {', '.join(_FASTENERS)}")
    d, pitch, tap, cf, cm, hd, hh, nh, naf, w = _FASTENERS[s]
    return {"size": s, "d": d, "pitch": pitch, "tap_drill": tap, "clearance_fine": cf, "clearance_medium": cm,
            "head_d": hd, "head_h": hh, "nut_h": nh, "nut_af": naf,
            "washer_id": w[0], "washer_od": w[1], "washer_h": w[2]}


def pick_bolt_length(required_mm):
    for L in _BOLT_LENGTHS:
        if L >= required_mm - 1e-9:
            return float(L)
    raise AssemblyError(f"no standard bolt is long enough for {required_mm:.1f} mm of stack")


# ---------------------------------------------------------------------------------------------------------------------
# feature selection (features come from features_from_shape: planes P*, cylinders H* (holes) / S* (pins))
# ---------------------------------------------------------------------------------------------------------------------
_AXES = {"x": 0, "y": 1, "z": 2}


def _by_id(features, fid):
    for f in features["planes"] + features["cylinders"]:
        if f["id"].upper() == str(fid).upper():
            return f
    return None


def pick(features, spec, kinds=("plane", "hole", "pin"), label="feature"):
    """Exactly one feature, by id ('P3', 'H2') or by selector {kind, at, normal, diameter_mm, largest}."""
    if isinstance(spec, str):
        f = _by_id(features, spec)
        if not f:
            ids = [p["id"] for p in features["planes"]] + [c["id"] for c in features["cylinders"]]
            raise AssemblyError(f"{label}: no feature with id {spec!r}; this part has {ids}")
        return f
    if not isinstance(spec, dict):
        raise AssemblyError(f"{label}: a feature must be an id string or a selector object")
    if "id" in spec:
        return pick(features, spec["id"], kinds, label)
    kind = spec.get("kind")
    pool = []
    if kind in (None, "plane"):
        pool += [dict(p, _kind="plane") for p in features["planes"]]
    if kind in (None, "hole"):
        pool += [dict(c, _kind="hole") for c in features["cylinders"] if c["kind"] == "hole"]
    if kind in (None, "pin"):
        pool += [dict(c, _kind="pin") for c in features["cylinders"] if c["kind"] == "pin"]
    bb = features["bbox_mm"]
    span = max(1e-6, max(np.array(bb["max"]) - np.array(bb["min"])))
    tol = max(0.02 * span, 0.2)
    out = []
    for f in pool:
        if "normal" in spec and f["_kind"] == "plane":
            m = re.fullmatch(r"([+-]?)([xyz])", str(spec["normal"]).lower().strip())
            if not m:
                raise AssemblyError(f"{label}: normal must look like '+z' or '-x'")
            want = np.zeros(3)
            want[_AXES[m.group(2)]] = -1.0 if m.group(1) == "-" else 1.0
            if float(np.dot(unit(f["normal"]), want)) < 0.98:
                continue
        if "at" in spec:
            m = re.fullmatch(r"(min|max)_([xyz])", str(spec["at"]).lower().strip())
            if not m:
                raise AssemblyError(f"{label}: 'at' must look like 'min_z' or 'max_x'")
            ax = _AXES[m.group(2)]
            target = bb["min"][ax] if m.group(1) == "min" else bb["max"][ax]
            c = f["center"] if f["_kind"] == "plane" else f["center"]
            if abs(c[ax] - target) > tol:
                continue
            if f["_kind"] == "plane" and abs(unit(f["normal"])[ax]) < 0.9:
                continue
        if "diameter_mm" in spec and f["_kind"] != "plane":
            lo, hi = (spec["diameter_mm"] if isinstance(spec["diameter_mm"], (list, tuple))
                      else (spec["diameter_mm"] - 0.3, spec["diameter_mm"] + 0.3))
            if not lo <= f["diameter"] <= hi:
                continue
        out.append(f)
    if not out:
        raise AssemblyError(f"{label}: nothing matches {spec}. Available: planes "
                            f"{[(p['id'], p['normal'], round(p['area'])) for p in features['planes'][:8]]}")
    if len(out) > 1:
        out.sort(key=lambda f: -f.get("area", 0.0))
        if spec.get("largest") or all(f["_kind"] == "plane" for f in out):
            return {k: v for k, v in out[0].items() if k != "_kind"}
        raise AssemblyError(f"{label}: {len(out)} features match {spec} ({[f['id'] for f in out]}); use an id")
    return {k: v for k, v in out[0].items() if k != "_kind"}


def holes_on(features, plane):
    """Holes whose axis is perpendicular to `plane` and whose opening lies on it."""
    n = unit(plane["normal"])
    c = np.asarray(plane["center"], float)
    out = []
    for h in features["cylinders"]:
        if h["kind"] != "hole":
            continue
        if abs(float(np.dot(unit(h["axis"]), n))) < 0.98:
            continue
        ends = [np.asarray(e, float) for e in h["ends"]]
        if min(abs(float(np.dot(e - c, n))) for e in ends) <= 0.25:
            out.append(h)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# hole-pattern registration: after two faces are brought together, find the in-plane rotation and shift that makes the
# hole patterns coincide (or prove that they cannot)
# ---------------------------------------------------------------------------------------------------------------------
def register_holes(a_pts2, b_pts2, tol=TOL_POS_MM):
    """a_pts2, b_pts2: Nx2 arrays. Returns (theta, t(2), matches [(ia, ib)], max_residual). Every B point must land on
    an A point; A may have more."""
    A = np.asarray(a_pts2, float).reshape(-1, 2)
    B = np.asarray(b_pts2, float).reshape(-1, 2)
    if len(B) == 0 or len(A) < len(B):
        raise AssemblyError(f"hole patterns: part A offers {len(A)} hole(s) on the mating face but part B has {len(B)}")

    def rot2(th):
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s], [s, c]])

    angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    if len(B) >= 2:
        for (i, j), (k, l) in itertools.product(itertools.permutations(range(len(A)), 2),
                                                itertools.permutations(range(len(B)), 2)):
            da, db = A[j] - A[i], B[l] - B[k]
            if abs(np.linalg.norm(da) - np.linalg.norm(db)) <= 2 * tol:
                angles.append(math.atan2(da[1], da[0]) - math.atan2(db[1], db[0]))
    best = None
    for th in angles:
        R = rot2(th)
        RB = (R @ B.T).T
        for i in range(len(A)):
            for j in range(len(B)):
                t = A[i] - RB[j]
                P = RB + t
                d = np.linalg.norm(P[:, None, :] - A[None, :, :], axis=2)
                nearest = d.min(axis=1)
                matched = int((nearest <= tol).sum())
                score = (matched, -float(nearest.sum()))
                if best is None or score > best[0]:
                    best = (score, th, t, [(int(d[k].argmin()), k) for k in range(len(B))], float(nearest.max()))
    (matched, _), th, t, pairs, worst = best
    if matched < len(B):
        raise AssemblyError(f"hole patterns do not line up: the best fit still leaves a hole {worst:.2f} mm off "
                            f"({matched} of {len(B)} holes coincide). Change the hole positions or mate other faces.")
    return th, t, pairs, worst


# ---------------------------------------------------------------------------------------------------------------------
# mate solving
# ---------------------------------------------------------------------------------------------------------------------
def _world_plane(F, T):
    return unit(dr(T, F["normal"])), pt(T, F["center"])


def _world_cyl(F, T):
    return unit(dr(T, F["axis"])), pt(T, F["point"]), [pt(T, e) for e in F["ends"]]


def _swap_sides(m):
    """The same mate with sides a and b exchanged (mates are symmetric; this lets the solver always place side b)."""
    s = dict(m)
    s["a"], s["b"] = m["b"], m["a"]
    ax = m.get("axial")
    if isinstance(ax, dict):
        ax = dict(ax)
        if "a_plane" in ax or "b_plane" in ax:
            ax["a_plane"], ax["b_plane"] = ax.get("b_plane"), ax.get("a_plane")
        if ax.get("mode") == "offset" and "offset_mm" in ax:
            ax["offset_mm"] = -float(ax["offset_mm"])
        s["axial"] = ax
    fa = m.get("fasteners")
    if isinstance(fa, dict) and fa.get("head_on") in ("a", "b"):
        fa = dict(fa)
        fa["head_on"] = "b" if fa["head_on"] == "a" else "a"
        if fa.get("tapped_part") in ("a", "b"):
            fa["tapped_part"] = "b" if fa["tapped_part"] == "a" else "a"
        s["fasteners"] = fa
    elif isinstance(fa, dict) and fa.get("tapped_part") in ("a", "b"):
        fa = dict(fa)
        fa["tapped_part"] = "b" if fa["tapped_part"] == "a" else "a"
        s["fasteners"] = fa
    return s


def _solve_plane_mate(m, FA, TA, FB):
    fa = pick(FA, m["a"].get("feature"), ("plane",), f"mate {m['id']} side a")
    fb = pick(FB, m["b"].get("feature"), ("plane",), f"mate {m['id']} side b")
    if "normal" not in fa or "normal" not in fb:
        raise AssemblyError(f"mate {m['id']}: a plane_mate needs two planar faces (got {fa['id']} and {fb['id']})")
    nA, cA = _world_plane(fa, TA)
    nB0 = unit(fb["normal"])
    R = rot_between(nB0, -nA)
    rot = float(m.get("rotate_deg", 0.0) or 0.0)
    if rot:
        R = rot_axis(nA, math.radians(rot)) @ R
    gap = float(m.get("gap_mm", 0.0) or 0.0)
    t = (cA + nA * gap) - R @ np.asarray(fb["center"], float)
    uv = m.get("offset_uv_mm")
    if uv:
        u, v = plane_basis(nA)
        t = t + u * float(uv[0]) + v * float(uv[1])
    T = T_from(R, t)
    info = {"type": "plane_mate", "a_feature": fa["id"], "b_feature": fb["id"], "pairs": [], "explode_dir": nA.tolist()}
    if str(m.get("register", "")).lower() == "holes":
        ha = holes_on(FA, fa)
        hb = holes_on(FB, fb)
        u, v = plane_basis(nA)

        def to2(p):
            q = np.asarray(p, float) - cA
            return [float(q @ u), float(q @ v)]

        a2 = [to2(pt(TA, h["point"])) for h in ha]
        b2 = [to2(pt(T, h["point"])) for h in hb]
        th, t2, pairs, worst = register_holes(a2, b2)
        shift = u * t2[0] + v * t2[1]
        # rotate about the plane normal through cA by th, then shift (the 2-D solution is in the (u, v) frame about cA)
        T = T_from(np.eye(3), shift) @ T_about(cA, nA, th) @ T
        info["pairs"] = [{"a": ha[i]["id"], "b": hb[k]["id"], "a_d": ha[i]["diameter"], "b_d": hb[k]["diameter"],
                          "a_len": ha[i]["length"], "b_len": hb[k]["length"]} for i, k in pairs]
        info["registration_residual_mm"] = round(worst, 3)
        info["_holes"] = {"a": ha, "b": hb, "pairs": pairs, "nA": nA.tolist(), "cA": cA.tolist()}
    return T, info


def _solve_coaxial(m, FA, TA, FB):
    fa = pick(FA, m["a"].get("feature"), ("hole", "pin"), f"mate {m['id']} side a")
    fb = pick(FB, m["b"].get("feature"), ("hole", "pin"), f"mate {m['id']} side b")
    if "axis" not in fa or "axis" not in fb:
        raise AssemblyError(f"mate {m['id']}: a coaxial mate needs two cylindrical features (got {fa['id']} and {fb['id']})")
    aA, pA, endsA = _world_cyl(fa, TA)
    target = -aA if m.get("flip") else aA
    R = rot_between(unit(fb["axis"]), target)
    rot = float(m.get("rotate_deg", 0.0) or 0.0)
    if rot:
        R = rot_axis(aA, math.radians(rot)) @ R
    t = pA - R @ np.asarray(fb["point"], float)
    T = T_from(R, t)
    mid = lambda ends: (np.asarray(ends[0], float) + np.asarray(ends[1], float)) / 2
    midA = mid(endsA)
    midB = pt(T, mid(fb["ends"]))
    s = float(np.dot(midA - midB, aA))                    # default: centre the two cylinders on each other
    ax = m.get("axial") or {}
    mode = ax.get("mode", "center")
    if mode in ("plane_contact", "plane_flush"):
        # contact: the two faces touch (normals opposed). flush: the faces are coplanar and face the same way
        # (a pin end level with a surface).
        pa = pick(FA, ax.get("a_plane"), ("plane",), f"mate {m['id']} axial a_plane")
        pb = pick(FB, ax.get("b_plane"), ("plane",), f"mate {m['id']} axial b_plane")
        nA, cA = _world_plane(pa, TA)
        nB = unit(dr(T, pb["normal"]))
        if abs(float(np.dot(nA, aA))) < 0.9:
            raise AssemblyError(f"mate {m['id']}: the a_plane is not perpendicular to the cylinder axis")
        want = -1.0 if mode == "plane_contact" else 1.0
        if float(np.dot(nA, nB)) * want < 0.98:
            raise AssemblyError(f"mate {m['id']}: after the axes line up the two faces point "
                                f"{'the same way' if mode == 'plane_contact' else 'opposite ways'}, so they cannot "
                                f"{'touch' if mode == 'plane_contact' else 'be flush'}. Use axial mode "
                                f"{'plane_flush' if mode == 'plane_contact' else 'plane_contact'}, set flip=true, or choose "
                                f"the opposite face.")
        cB = pt(T, pb["center"])
        gap = float(ax.get("gap_mm", 0.0) or 0.0)
        s = float(np.dot(cA + nA * gap - cB, aA) / np.dot(nA, aA))
    elif mode == "offset":
        s += float(ax.get("offset_mm", 0.0))
    T = T_from(R, t + aA * s)
    d_a, d_b = fa["diameter"], fb["diameter"]
    info = {"type": "coaxial", "a_feature": fa["id"], "b_feature": fb["id"], "explode_dir": aA.tolist()}
    if {fa["kind"], fb["kind"]} == {"hole", "pin"}:
        hole_d = d_a if fa["kind"] == "hole" else d_b
        pin_d = d_b if fa["kind"] == "hole" else d_a
        c = hole_d - pin_d
        info["fit"] = {"hole_mm": round(hole_d, 3), "pin_mm": round(pin_d, 3), "clearance_mm": round(c, 3),
                       "kind": ("interference (needs force or heat)" if c < -0.05 else
                                "press / transition fit" if c < 0.05 else
                                "close sliding fit" if c <= 0.25 else
                                "free fit" if c <= 0.6 else "loose (sloppy)")}
    return T, info


def solve_assembly(features_by_part, mates, ground):
    """Place every part. features_by_part: id -> features dict. Returns (T_by_part, mate_reports, warnings).
    Mates are processed from the ground outwards; a mate between two already placed parts is checked, not used."""
    ids = list(features_by_part)
    if ground not in ids:
        raise AssemblyError(f"ground part {ground!r} is not in the part list {ids}")
    for m in mates:
        for side in ("a", "b"):
            p = (m.get(side) or {}).get("part")
            if p not in features_by_part:
                raise AssemblyError(f"mate {m.get('id')}: part {p!r} is not in the part list {ids}")
        if m["a"]["part"] == m["b"]["part"]:
            raise AssemblyError(f"mate {m.get('id')}: a part cannot be mated to itself")
    placed = {ground: np.eye(4)}
    reports, warnings = [], []
    pending = list(mates)
    guard = 0
    while pending and guard < 4 * len(mates) + 4:
        guard += 1
        progressed = False
        for m in list(pending):
            a, b = m["a"]["part"], m["b"]["part"]
            if a in placed and b in placed:
                pending.remove(m)
                reports.append(_closure_report(m, features_by_part, placed, warnings))
                progressed = True
            elif a in placed or b in placed:
                mm = m if a in placed else _swap_sides(m)
                parent, child = mm["a"]["part"], mm["b"]["part"]
                mtype = (mm.get("type") or "plane_mate").lower()
                if mtype == "plane_mate":
                    T, info = _solve_plane_mate(mm, features_by_part[parent], placed[parent], features_by_part[child])
                elif mtype == "coaxial":
                    T, info = _solve_coaxial(mm, features_by_part[parent], placed[parent], features_by_part[child])
                else:
                    raise AssemblyError(f"mate {m.get('id')}: unknown type {mtype!r} (plane_mate | coaxial)")
                placed[child] = T
                info.update({"id": m.get("id"), "parent": parent, "child": child, "status": "ok",
                             "gap_mm": float(m.get("gap_mm", 0.0) or 0.0), "_mate": mm})
                reports.append(info)
                pending.remove(m)
                progressed = True
        if not progressed:
            break
    if pending:
        raise AssemblyError("these mates connect parts that are not linked to the ground part '%s': %s"
                            % (ground, ", ".join(str(m.get("id")) for m in pending)))
    loose = [p for p in ids if p not in placed]
    if loose:
        raise AssemblyError(f"parts with no mate to the assembly: {loose}")
    return placed, reports, warnings


def _closure_report(m, FBP, placed, warnings):
    """A mate between two parts that are already placed: it was not needed to place anything, so verify it."""
    a, b = m["a"]["part"], m["b"]["part"]
    rep = {"id": m.get("id"), "type": (m.get("type") or "plane_mate"), "parent": a, "child": b, "status": "ok",
           "closure_check": True}
    try:
        if rep["type"] == "plane_mate":
            fa = pick(FBP[a], m["a"].get("feature"), ("plane",), f"mate {m['id']} side a")
            fb = pick(FBP[b], m["b"].get("feature"), ("plane",), f"mate {m['id']} side b")
            nA, cA = _world_plane(fa, placed[a])
            nB, cB = _world_plane(fb, placed[b])
            ang = math.degrees(math.acos(max(-1, min(1, float(-nA @ nB)))))
            dist = float((cB - cA) @ nA)
            gap = float(m.get("gap_mm", 0.0) or 0.0)
            rep.update({"angle_error_deg": round(ang, 3), "gap_error_mm": round(dist - gap, 3)})
            if ang > TOL_ANGLE_DEG or abs(dist - gap) > 0.2:
                rep["status"] = "warning"
                warnings.append(f"mate {m['id']} closes a loop but the faces are {ang:.1f} deg / {dist - gap:.2f} mm "
                                f"from touching in the solved placement")
        else:
            fa = pick(FBP[a], m["a"].get("feature"), ("hole", "pin"), f"mate {m['id']} side a")
            fb = pick(FBP[b], m["b"].get("feature"), ("hole", "pin"), f"mate {m['id']} side b")
            aA, pA, _ = _world_cyl(fa, placed[a])
            aB, pB, _ = _world_cyl(fb, placed[b])
            off = float(np.linalg.norm(np.cross(pB - pA, aA)))
            rep["axis_offset_mm"] = round(off, 3)
            if off > 0.2 or abs(float(aA @ aB)) < math.cos(math.radians(TOL_ANGLE_DEG)):
                rep["status"] = "warning"
                warnings.append(f"mate {m['id']} closes a loop but the two axes are {off:.2f} mm apart in the placement")
    except AssemblyError as e:
        rep["status"] = "warning"
        warnings.append(str(e))
    return rep


# ---------------------------------------------------------------------------------------------------------------------
# fasteners: one bolt per registered hole pair of a plane_mate that asks for them
# ---------------------------------------------------------------------------------------------------------------------
def plan_fasteners(report, placed, features_by_part):
    """One bolt per registered hole pair of a plane_mate that asks for hardware. Returns (hardware list, notes).

    Geometry along the joint normal n (pointing from the parent, side a, into the child, side b), measured from the joint
    plane: the parent occupies s <= 0 and the child s >= 0. The head sits on one outer face, the nut (or the thread in a
    tapped part) at the other."""
    mm = report.get("_mate") or {}
    fa = mm.get("fasteners")
    if not fa or report.get("type") != "plane_mate" or "_holes" not in report:
        return [], []
    spec = fastener_spec(fa.get("size", "M3"))
    tapped = fa.get("tapped_part") if fa.get("tapped_part") in ("a", "b") else None
    nut = bool(fa.get("nut", True)) and tapped is None
    washer = bool(fa.get("washer", False))
    head_on = fa.get("head_on") if fa.get("head_on") in ("a", "b") else "b"
    if tapped == head_on:                                   # the head always sits on the clearance-hole side
        head_on = "b" if tapped == "a" else "a"
    wh = spec["washer_h"] if washer else 0.0
    H = report["_holes"]
    n = unit(H["nA"])
    cA = np.asarray(H["cA"], float)
    TP_, TC_ = placed[report["parent"]], placed[report["child"]]
    notes, hw = [], []
    for k, (ia, ib) in enumerate(H["pairs"]):
        ha, hb = H["a"][ia], H["b"][ib]
        ea = [pt(TP_, e) for e in ha["ends"]]
        eb = [pt(TC_, e) for e in hb["ends"]]
        sa = sorted(float((e - cA) @ n) for e in ea)         # parent hole, s <= ~0
        sb = sorted(float((e - cA) @ n) for e in eb)         # child hole,  s >= ~0
        outer_a, outer_b = sa[0], sb[1]
        stack = outer_b - outer_a
        mid = (ea[0] + ea[1]) / 2
        p0 = mid - n * float((mid - cA) @ n)                 # the bolt axis, on the joint plane
        if tapped:
            t_len = (sa[1] - sa[0]) if tapped == "a" else (sb[1] - sb[0])
            need = (stack - t_len) + wh + 1.5 * spec["d"]    # head-side part + washer + thread engagement
        elif nut:
            need = stack + wh + (wh + spec["nut_h"] + 1.5 * spec["pitch"])
        else:
            need = stack + wh
        L = pick_bolt_length(need)
        if L > stack + wh + 2.0 and (tapped or not nut):
            notes.append(f"{spec['size']}x{L:g} is {L - stack - wh:.1f} mm longer than the stack; the tip would stick out "
                         f"or bottom out - use a thinner washer/part or a shorter bolt")
        elif L - need > 4.0 + spec["d"]:
            notes.append(f"{spec['size']}x{L:g}: {L - need:.1f} mm longer than the stack needs ({need:.1f} mm)")
        for side, hole in (("a", ha), ("b", hb)):
            dh = hole["diameter"]
            if tapped == side:
                if abs(dh - spec["tap_drill"]) > 0.3:
                    notes.append(f"mate {mm.get('id')}: tapped hole {hole['id']} is {dh:.2f} mm; {spec['size']} needs a "
                                 f"{spec['tap_drill']:.2f} mm tap-drill hole")
            elif dh < spec["d"] + 0.05:
                notes.append(f"mate {mm.get('id')}: hole {hole['id']} ({dh:.2f} mm) is too small for a {spec['size']} bolt; "
                             f"make it at least {spec['clearance_fine']:.1f} mm")
            elif dh > spec["clearance_medium"] + 0.8:
                notes.append(f"mate {mm.get('id')}: hole {hole['id']} ({dh:.2f} mm) is very loose for {spec['size']} "
                             f"(normal clearance {spec['clearance_medium']:.1f} mm)")
        if head_on == "a":
            axis, head_s, far_s = n, outer_a - wh, outer_b
        else:
            axis, head_s, far_s = -n, outer_b + wh, outer_a
        sign = 1.0 if head_on == "a" else -1.0
        item = {"id": f"{mm.get('id')}-b{k + 1}", "joint": mm.get("id"), "type": "bolt", "standard": "ISO 4762",
                "size": spec["size"], "length_mm": L, "head_d": spec["head_d"], "head_h": spec["head_h"],
                "shank_d": spec["d"], "origin": [round(float(x), 4) for x in (p0 + n * head_s)],
                "axis": [round(float(x), 6) for x in axis], "pair": {"a": ha["id"], "b": hb["id"]},
                "nut": None, "washers": None, "tapped_part": tapped}
        if nut:
            item["nut"] = {"standard": "ISO 4032", "h": spec["nut_h"], "af": spec["nut_af"],
                           "origin": [round(float(x), 4) for x in (p0 + n * (far_s + sign * wh))]}
        if washer:
            item["washers"] = {"standard": "ISO 7089", "od": spec["washer_od"], "id": spec["washer_id"], "h": spec["washer_h"],
                               "origins": [[round(float(x), 4) for x in (p0 + n * (outer_a if head_on == "a" else outer_b))]]
                               + ([[round(float(x), 4) for x in (p0 + n * far_s)]] if nut else [])}
        hw.append(item)
    return hw, notes


def bill_of_materials(parts, hardware):
    rows = [{"kind": "part", "name": p.get("name", p["id"]), "id": p["id"], "qty": 1} for p in parts]
    agg = {}
    for h in hardware:
        key = ("bolt", h["size"], h["length_mm"])
        agg[key] = agg.get(key, 0) + 1
        if h.get("nut"):
            agg[("nut", h["size"], None)] = agg.get(("nut", h["size"], None), 0) + 1
        if h.get("washers"):
            agg[("washer", h["size"], None)] = agg.get(("washer", h["size"], None), 0) + h["washers"]["count"]
    names = {"bolt": "Socket head cap screw ISO 4762", "nut": "Hex nut ISO 4032", "washer": "Plain washer ISO 7089"}
    for (kind, size, length), qty in sorted(agg.items(), key=lambda kv: (kv[0][0] != "bolt", kv[0][0], kv[0][1])):
        label = f"{names[kind]} {size}" + (f" x {length:g}" if length else "")
        rows.append({"kind": "hardware", "name": label, "qty": qty})
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# internal space: measure the cavities of the real geometry (no model involved)
# ---------------------------------------------------------------------------------------------------------------------
def voxelize_triangles(tris, pitch=None, target_cells=200_000, max_cells=450_000, pad=1):
    """Solid occupancy of a closed triangle mesh by casting a vertical ray through every grid column and filling between
    entry/exit pairs. Returns (occ[nx,ny,nz] bool, origin(3), pitch, stats)."""
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    size = np.maximum(hi - lo, 1e-6)
    if pitch is None:
        pitch = max(float(np.cbrt(np.prod(size) / target_cells)), float(size.max()) / 200.0)
    pitch = float(pitch)
    while True:
        n = np.ceil(size / pitch).astype(int) + 2 * pad
        if int(n.prod()) <= max_cells:
            break
        pitch *= 1.15
    origin = lo - pad * pitch
    nx, ny, nz = (int(v) for v in n)
    jx, jy = 0.0137 * pitch, 0.0291 * pitch              # keep rays off exact vertices / edges of grid-aligned parts
    cols_i, cols_j, zs = [], [], []
    for tri in tris:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-14:
            continue                                      # vertical triangle: parallel to the rays
        i0 = max(int(math.ceil((min(ax, bx, cx) - jx - origin[0]) / pitch - 0.5)), 0)
        i1 = min(int(math.floor((max(ax, bx, cx) - jx - origin[0]) / pitch - 0.5)), nx - 1)
        j0 = max(int(math.ceil((min(ay, by, cy) - jy - origin[1]) / pitch - 0.5)), 0)
        j1 = min(int(math.floor((max(ay, by, cy) - jy - origin[1]) / pitch - 0.5)), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        xs = origin[0] + (np.arange(i0, i1 + 1) + 0.5) * pitch + jx
        ys = origin[1] + (np.arange(j0, j1 + 1) + 0.5) * pitch + jy
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-12) & (l2 >= -1e-12) & (l3 >= -1e-12)
        if not inside.any():
            continue
        ii, jj = np.nonzero(inside)
        cols_i.append(ii + i0)
        cols_j.append(jj + j0)
        zs.append((l1 * az + l2 * bz + l3 * cz)[inside])
    occ = np.zeros((nx, ny, nz), bool)
    stats = {"odd_columns": 0, "columns_hit": 0}
    if not zs:
        return occ, origin, pitch, stats
    ci, cj, z = np.concatenate(cols_i), np.concatenate(cols_j), np.concatenate(zs)
    key = ci.astype(np.int64) * ny + cj
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    for s0, s1 in zip(starts, ends):
        zz = z[s0:s1]
        zz = zz[np.r_[True, np.diff(zz) > 1e-7]]          # a ray through a shared edge hits two triangles at one z
        stats["columns_hit"] += 1
        if len(zz) % 2:
            stats["odd_columns"] += 1
            zz = zz[:-1]
        i, j = divmod(int(key[s0]), ny)
        for a, b in zip(zz[0::2], zz[1::2]):
            k0 = max(int(math.ceil((a - origin[2]) / pitch - 0.5)), 0)
            k1 = min(int(math.floor((b - origin[2]) / pitch - 0.5)), nz - 1)
            if k1 >= k0:
                occ[i, j, k0:k1 + 1] = True
    return occ, origin, pitch, stats


_DIRS = ["-x", "+x", "-y", "+y", "-z", "+z"]


def _axis_hits(solid):
    """For every voxel: is there solid anywhere before / after it along each axis (exclusive)."""
    out = []
    for ax in range(3):
        c = np.maximum.accumulate(solid, axis=ax)
        neg = np.zeros_like(solid)
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[ax], src[ax] = slice(1, None), slice(0, -1)
        neg[tuple(dst)] = c[tuple(src)]
        cr = np.flip(np.maximum.accumulate(np.flip(solid, axis=ax), axis=ax), axis=ax)
        pos = np.zeros_like(solid)
        dst[ax], src[ax] = slice(0, -1), slice(1, None)
        pos[tuple(dst)] = cr[tuple(src)]
        out.append((neg, pos))
    return out


def _largest_box(mask, seeds=4):
    """Axis-aligned box of True voxels, grown from the deepest points of the region. Returns (lo, hi) inclusive indices."""
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(mask)
    flat = np.argsort(dist, axis=None)[::-1]
    picked = []
    for f in flat[: max(2000, seeds * 50)]:
        p = np.array(np.unravel_index(f, mask.shape))
        if dist[tuple(p)] <= 0:
            break
        if all(np.abs(p - q).max() > 3 for q in picked):
            picked.append(p)
        if len(picked) >= seeds:
            break
    best = None
    for p in picked:
        lo, hi = p.copy(), p.copy()
        changed = True
        while changed:
            changed = False
            for ax in range(3):
                for sgn in (-1, 1):
                    idx = lo[ax] - 1 if sgn < 0 else hi[ax] + 1
                    if idx < 0 or idx >= mask.shape[ax]:
                        continue
                    sl = [slice(lo[k], hi[k] + 1) for k in range(3)]
                    sl[ax] = slice(idx, idx + 1)
                    if mask[tuple(sl)].all():
                        if sgn < 0:
                            lo[ax] -= 1
                        else:
                            hi[ax] += 1
                        changed = True
        vol = int(np.prod(hi - lo + 1))
        if best is None or vol > best[0]:
            best = (vol, lo.copy(), hi.copy())
    return (best[1], best[2]) if best else (None, None)


def analyze_internal_space(tris, pitch=None, min_voxels=8):
    """Cavities of a part: enclosed voids, pockets (open one way), channels / tunnels (open both ends) and troughs. A cavity
    voxel is empty space with solid on BOTH sides along at least one axis and on at least three of the six directions.
    That keeps open corners (an L bracket's inner corner) out and lets through-holes and wire channels in."""
    from scipy import ndimage
    occ, origin, pitch, stats = voxelize_triangles(tris, pitch)
    solid = occ
    hits = _axis_hits(solid)
    both = sum((n & p).astype(np.int8) for n, p in hits)
    one = sum((n ^ p).astype(np.int8) for n, p in hits)
    interior = (~solid) & (both >= 1) & (2 * both + one >= 3)
    labels, ncomp = ndimage.label(interior)
    cavities = []
    cell_mm3 = pitch ** 3
    for lab in range(1, ncomp + 1):
        comp = labels == lab
        count = int(comp.sum())
        if count < min_voxels:
            continue
        idx = np.argwhere(comp)
        lo_i, hi_i = idx.min(axis=0), idx.max(axis=0)
        sub = comp[lo_i[0]:hi_i[0] + 1, lo_i[1]:hi_i[1] + 1, lo_i[2]:hi_i[2] + 1]
        open_dirs = []
        for ax in range(3):
            neg, pos = hits[ax]
            neg_s = neg[lo_i[0]:hi_i[0] + 1, lo_i[1]:hi_i[1] + 1, lo_i[2]:hi_i[2] + 1][sub]
            pos_s = pos[lo_i[0]:hi_i[0] + 1, lo_i[1]:hi_i[1] + 1, lo_i[2]:hi_i[2] + 1][sub]
            if (~neg_s).mean() > 0.5:
                open_dirs.append(_DIRS[2 * ax])
            if (~pos_s).mean() > 0.5:
                open_dirs.append(_DIRS[2 * ax + 1])
        axes_open = {d[1] for d in open_dirs}
        if not open_dirs:
            kind = "enclosed"
        elif len(open_dirs) == 1:
            kind = "pocket"
        elif len(open_dirs) == 2 and len(axes_open) == 1:
            kind = "channel"
        else:
            kind = "trough"
        lo_b, hi_b = _largest_box(comp)
        if lo_b is None:
            continue
        size = (hi_b - lo_b + 1) * pitch
        cavities.append({
            "kind": kind, "volume_mm3": round(count * cell_mm3, 2), "openings": open_dirs,
            "bbox_min": [round(float(v), 3) for v in origin + lo_i * pitch],
            "bbox_max": [round(float(v), 3) for v in origin + (hi_i + 1) * pitch],
            "inscribed": {"min": [round(float(v), 3) for v in origin + lo_b * pitch],
                          "max": [round(float(v), 3) for v in origin + (hi_b + 1) * pitch],
                          "size": [round(float(v), 3) for v in size]},
        })
    cavities.sort(key=lambda c: -c["volume_mm3"])
    for k, c in enumerate(cavities):
        c["id"] = f"C{k + 1}"
    warnings = []
    if stats["columns_hit"] and stats["odd_columns"] / stats["columns_hit"] > 0.02:
        warnings.append(f"{stats['odd_columns']} of {stats['columns_hit']} voxel columns had an odd number of surface hits "
                        f"(the mesh may not be closed); cavity sizes may be unreliable")
    return {"pitch_mm": round(pitch, 4), "grid": list(occ.shape), "solid_volume_mm3": round(float(solid.sum()) * cell_mm3, 2),
            "cavities": cavities, "warnings": warnings,
            "accuracy_note": f"measured on a {pitch:.2f} mm grid: sizes are conservative to about +/-{pitch:.2f} mm per side"}


# ---------------------------------------------------------------------------------------------------------------------
# components: dimensions from datasheet text, and layer / shelf packing into a measured cavity
# ---------------------------------------------------------------------------------------------------------------------
_NUM = r"(\d+(?:\.\d+)?)"
_DIM3 = re.compile(_NUM + r"\s*(?:mm)?\s*[x\u00d7*]\s*" + _NUM + r"\s*(?:mm)?\s*[x\u00d7*]\s*" + _NUM + r"\s*(?:mm)?", re.I)
_DIM2 = re.compile(_NUM + r"\s*(?:mm)?\s*[x\u00d7*]\s*" + _NUM + r"\s*mm", re.I)
_DIA = re.compile(r"(?:[\u00d8\u2300]|dia(?:meter)?\.?\s*)\s*" + _NUM + r"\s*mm", re.I)


def parse_dims_mm(text):
    """(a, b, c) in mm from text such as '45 x 18 x 7 mm', '4.9x3.9x1.75mm' or 'diameter 20 mm x 15 mm', else None."""
    if not text:
        return None
    t = str(text)
    m = _DIM3.search(t)
    if m:
        return tuple(float(g) for g in m.groups())
    d = _DIA.search(t)
    if d:
        dia = float(d.group(1))
        rest = t[d.end():]
        m2 = re.search(r"[x\u00d7*]\s*" + _NUM + r"\s*mm", rest, re.I) or re.search(_NUM + r"\s*mm\s*(?:long|tall|high|length|height)", rest, re.I)
        return (dia, dia, float(m2.group(1))) if m2 else (dia, dia, dia)
    m = _DIM2.search(t)
    if m:
        return (float(m.group(1)), float(m.group(2)), 1.0)
    return None


def _shelf_pack(items2d, W, D, gap):
    """Place rectangles (w, d, ref) on a W x D layer, left to right in shelves. Returns (placed, unplaced)."""
    placed, unplaced = [], []
    shelves = []                                          # [y, depth, x_used]
    for w, d, ref in sorted(items2d, key=lambda t: -t[0] * t[1]):
        done = False
        for orient in ((w, d), (d, w)):
            ww, dd = orient
            for sh in shelves:
                if dd <= sh[1] + 1e-9 and sh[2] + ww <= W + 1e-9:
                    placed.append((sh[2], sh[0], ww, dd, ref))
                    sh[2] += ww + gap
                    done = True
                    break
            if done:
                break
        if done:
            continue
        for orient in ((w, d), (d, w)):
            ww, dd = orient
            y = (shelves[-1][0] + shelves[-1][1] + gap) if shelves else 0.0
            if ww <= W + 1e-9 and y + dd <= D + 1e-9:
                shelves.append([y, dd, ww + gap])
                placed.append((0.0, y, ww, dd, ref))
                done = True
                break
        if not done:
            unplaced.append(ref)
    return placed, unplaced


def pack_components(items, box_size, clearance=1.0, margin=1.0):
    """Stack components in layers inside a box (x, y, z). items: [{'name', 'dims': (a, b, c), 'qty'}]. Tries each box axis as
    the stacking direction; each component lies flat (thinnest side vertical). Returns the best arrangement."""
    flat = []
    for it in items:
        dims = it.get("dims")
        if not dims:
            continue
        for q in range(int(it.get("qty", 1) or 1)):
            flat.append({"name": it.get("name", "item"), "index": q + 1, "dims": tuple(sorted(float(v) for v in dims))})
    if not flat:
        return {"fits": True, "placements": [], "unplaced": [], "note": "no components with known dimensions"}
    best = None
    for h_ax in (2, 0, 1):
        other = [a for a in range(3) if a != h_ax]
        W = box_size[other[0]] - 2 * margin
        D = box_size[other[1]] - 2 * margin
        H = box_size[h_ax] - 2 * margin
        if min(W, D, H) <= 0:
            continue
        layers, unplaced = [], []
        todo = sorted(flat, key=lambda c: -c["dims"][1] * c["dims"][2])
        remaining = list(todo)
        used_h = 0.0
        while remaining:
            # a layer is as tall as the tallest item that opens it; fill it with everything that fits
            first = remaining[0]
            t0 = first["dims"][0]
            if used_h + t0 > H + 1e-9:
                unplaced += remaining
                break
            cand = [c for c in remaining if used_h + c["dims"][0] <= H + 1e-9]
            rects = [(c["dims"][2], c["dims"][1], c) for c in cand]
            placed, left = _shelf_pack(rects, W, D, clearance)
            if not placed:
                unplaced += remaining
                break
            lay_h = max(p[4]["dims"][0] for p in placed)
            layers.append((used_h, lay_h, placed))
            used_h += lay_h + clearance
            ids = {id(p[4]) for p in placed}
            remaining = [c for c in remaining if id(c) not in ids]
        result = {"stack_axis": "xyz"[h_ax], "layers": len(layers), "used_height_mm": round(max(used_h - clearance, 0), 2),
                  "unplaced": [{"name": u["name"], "index": u["index"], "dims": list(u["dims"])} for u in unplaced],
                  "placements": [], "fits": not unplaced}
        for li, (z0, lh, placed) in enumerate(layers):
            for x0, y0, ww, dd, c in placed:
                mn = [0.0, 0.0, 0.0]
                sz = [0.0, 0.0, 0.0]
                mn[other[0]], mn[other[1]], mn[h_ax] = x0 + margin, y0 + margin, z0 + margin
                sz[other[0]], sz[other[1]], sz[h_ax] = ww, dd, c["dims"][0]
                result["placements"].append({"name": c["name"], "index": c["index"], "layer": li + 1,
                                             "min": [round(v, 3) for v in mn],
                                             "size": [round(v, 3) for v in sz]})
        score = (len(unplaced), result["used_height_mm"])
        if best is None or score < best[0]:
            best = (score, result)
    return best[1] if best else {"fits": False, "placements": [], "unplaced": [it for it in flat],
                                 "note": "the cavity is smaller than the margins"}


def required_cavity(items, clearance=1.0, margin=1.0, wire_allowance=0.0):
    """Smallest simple stack that holds the components: the footprint of the biggest one, layers of the rest on top.
    This is the number to give the CAD generator as the internal space a part must provide."""
    flat = []
    for it in items:
        if it.get("dims"):
            flat += [tuple(sorted(float(v) for v in it["dims"]))] * int(it.get("qty", 1) or 1)
    if not flat:
        return None
    fw = max(d[2] for d in flat)
    fd = max(d[1] for d in flat)
    ht = sum(d[0] for d in flat) + clearance * (len(flat) - 1)
    return {"min_inner_size_mm": [round(fw + 2 * margin + wire_allowance, 2), round(fd + 2 * margin, 2), round(ht + 2 * margin, 2)],
            "note": "stacked in layers, thinnest side vertical; the CAD part must contain at least this much measured internal space"}


def check_requirement(cavities, items, clearance=1.0, margin=1.0):
    """Does any cavity (or the cavities together) hold the components? The verdict the generator loop and the UI use."""
    ordered = sorted(cavities, key=lambda c: -c["inscribed"]["size"][0] * c["inscribed"]["size"][1] * c["inscribed"]["size"][2])
    tries = []
    for c in ordered:
        r = pack_components(items, c["inscribed"]["size"], clearance, margin)
        tries.append({"cavity": c["id"], **{k: r[k] for k in ("fits", "stack_axis", "used_height_mm") if k in r}})
        if r["fits"]:
            lo = np.array(c["inscribed"]["min"])
            for p in r["placements"]:
                p["min"] = [round(float(v), 3) for v in (np.array(p["min"]) + lo)]
            return {"fits": True, "cavity": c["id"], "packing": r, "tried": tries}
    need = required_cavity(items, clearance, margin)
    best = ordered[0]["inscribed"]["size"] if ordered else None
    return {"fits": False, "tried": tries, "required_inner_size_mm": need["min_inner_size_mm"] if need else None,
            "largest_measured_box_mm": best,
            "message": ("no cavity at all" if not ordered else
                        f"largest measured usable box is {best[0]:.1f} x {best[1]:.1f} x {best[2]:.1f} mm, the components need about "
                        f"{need['min_inner_size_mm'][0]:.1f} x {need['min_inner_size_mm'][1]:.1f} x {need['min_inner_size_mm'][2]:.1f} mm")}


# ---------------------------------------------------------------------------------------------------------------------
# features of a real part (the only place that needs the CAD kernel; build123d / OCP are imported lazily)
# ---------------------------------------------------------------------------------------------------------------------
def _cyl_axis(face, d):
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        ad = BRepAdaptor_Surface(face.wrapped)
        if ad.GetType() == GeomAbs_Cylinder:
            ax = ad.Cylinder().Axis()
            dd, pp = ax.Direction(), ax.Location()
            return np.array([dd.X(), dd.Y(), dd.Z()]), np.array([pp.X(), pp.Y(), pp.Z()])
    except Exception:
        pass
    ext = np.array(d["bbox_max"], float) - np.array(d["bbox_min"], float)
    L = d["area"] / (math.pi * d["diameter"]) if d.get("diameter") else float(ext.max())
    k = int(np.argmin(np.abs(ext - L)))
    axis = np.zeros(3)
    axis[k] = 1.0
    return axis, (np.array(d["bbox_min"], float) + np.array(d["bbox_max"], float)) / 2


def features_from_shape(shape, TP):
    """Planes (P1..), holes (H1..) and pins (S1..) of a build123d solid, in part coordinates (mm)."""
    descs = TP.describe_faces(shape)
    faces = list(shape.faces())
    bb = shape.bounding_box()
    feats = {"bbox_mm": {"min": [float(v) for v in TP._vec(bb.min)], "max": [float(v) for v in TP._vec(bb.max)]},
             "volume_mm3": round(float(shape.volume), 2), "planes": [], "cylinders": []}
    planes = sorted([d for d in descs if d["geom"] == "PLANE" and d.get("normal")], key=lambda d: -d["area"])
    for k, d in enumerate(planes):
        feats["planes"].append({"id": f"P{k + 1}", "face_idx": d["idx"], "center": [round(float(v), 4) for v in d["center"]],
                                "normal": [round(float(v), 6) for v in unit(d["normal"])], "area": round(float(d["area"]), 3),
                                "bbox_min": [round(float(v), 3) for v in d["bbox_min"]],
                                "bbox_max": [round(float(v), 3) for v in d["bbox_max"]]})
    nh = ns = 0
    for d in descs:
        if d["geom"] != "CYLINDER" or not d.get("diameter"):
            continue
        f = faces[d["idx"]]
        axis, point = _cyl_axis(f, d)
        axis = unit(axis)
        dia = float(d["diameter"])
        proj = []
        try:
            for v in f.vertices():
                proj.append(float((np.array(TP._vec(TP._call(v, "center")), float) - point) @ axis))
        except Exception:
            proj = []
        if len(proj) < 2 or max(proj) - min(proj) < 1e-6:
            corners = [np.array([x, y, z], float) for x in (d["bbox_min"][0], d["bbox_max"][0])
                       for y in (d["bbox_min"][1], d["bbox_max"][1]) for z in (d["bbox_min"][2], d["bbox_max"][2])]
            proj = [float((c - point) @ axis) for c in corners]
        s0, s1 = min(proj), max(proj)
        length = s1 - s0
        if length < 1e-6 or abs(d["area"] - math.pi * dia * length) > 0.12 * d["area"]:
            continue                                      # a fillet or another partial cylinder, not a bore or a pin
        kind = "hole"
        try:
            samples = d.get("samples") or []
            p = np.array(samples[len(samples) // 2], float)
            rad = (p - point) - axis * float((p - point) @ axis)
            rad = unit(rad)
            try:
                from build123d import Vector
                n = np.array(TP._vec(f.normal_at(Vector(*[float(x) for x in p]))), float)
                kind = "pin" if float(n @ rad) > 0 else "hole"
            except Exception:
                inside = shape.is_inside(Vector(*[float(x) for x in (p - rad * 0.05)]))
                kind = "pin" if inside else "hole"
        except Exception:
            kind = "hole"
        if kind == "hole":
            nh += 1
            cid = f"H{nh}"
        else:
            ns += 1
            cid = f"S{ns}"
        e0, e1 = point + axis * s0, point + axis * s1
        host = None
        for pl in feats["planes"]:
            n_ = np.array(pl["normal"])
            if abs(float(axis @ n_)) < 0.98:
                continue
            for e in (e0, e1):
                if abs(float((e - np.array(pl["center"])) @ n_)) <= 0.25 and \
                        all(pl["bbox_min"][i] - 0.5 <= e[i] <= pl["bbox_max"][i] + 0.5 for i in range(3)):
                    host = host or pl["id"]
        feats["cylinders"].append({"id": cid, "kind": kind, "diameter": round(dia, 4), "axis": [round(float(v), 6) for v in axis],
                                   "point": [round(float(v), 4) for v in point],
                                   "center": [round(float(v), 4) for v in (e0 + e1) / 2], "length": round(float(length), 4),
                                   "ends": [[round(float(v), 4) for v in e0], [round(float(v), 4) for v in e1]],
                                   "face_idx": d["idx"], "host_plane": host})
    return feats


def _axis_name(v):
    v = np.asarray(v, float)
    k = int(np.argmax(np.abs(v)))
    if abs(abs(v[k]) - 1.0) < 1e-3:
        return ("+" if v[k] > 0 else "-") + "xyz"[k]
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def feature_summary(F, name):
    """Compact text of what an LLM may refer to: ids, normals, positions. Numbers only - nothing to guess."""
    bb = F["bbox_mm"]
    size = [bb["max"][i] - bb["min"][i] for i in range(3)]
    lines = [f"PART '{name}': bounding box {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, spans "
             f"x {bb['min'][0]:.1f}..{bb['max'][0]:.1f}, y {bb['min'][1]:.1f}..{bb['max'][1]:.1f}, z {bb['min'][2]:.1f}..{bb['max'][2]:.1f}"]
    for p in F["planes"][:10]:
        c = p["center"]
        lines.append(f"  {p['id']}: flat face facing {_axis_name(p['normal'])}, centre ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), "
                     f"area {p['area']:.0f} mm2")
    groups = {}
    for c in F["cylinders"]:
        groups.setdefault((c["kind"], c["host_plane"], round(c["diameter"], 2), round(c["length"], 1)), []).append(c)
    for (kind, host, dia, ln), cs in groups.items():
        pos = " ".join(f"({c['center'][0]:.1f},{c['center'][1]:.1f},{c['center'][2]:.1f})" for c in cs[:8])
        ids = ",".join(c["id"] for c in cs)
        lines.append(f"  {ids}: {len(cs)} {kind}{'s' if len(cs) > 1 else ''} diameter {dia:g} mm, length {ln:g} mm, axis "
                     f"{_axis_name(cs[0]['axis'])}{(', opening on ' + host) if host else ''}, centres {pos}")
    return "\n".join(lines)


MATE_SYSTEM = """You propose how generated parts connect. You only choose WHICH features touch; a solver places the parts, so never
give coordinates or rotations except the optional fields below. Answer with ONE JSON object and nothing else:
{"ground": "<part id that stays fixed>", "mates": [ ... ]}

A mate connects side a (an already-placed part) to side b:
 {"id":"m1","type":"plane_mate","a":{"part":"<id>","feature":"P2"},"b":{"part":"<id>","feature":"P1"},
  "gap_mm":0,"register":"holes","fasteners":{"size":"M3","nut":true,"head_on":"b"}}
   plane_mate: two flat faces touch. register:"holes" aligns the hole patterns on those faces exactly (use it for any bolted
   joint; every hole on the b face must have a matching hole on the a face). optional: gap_mm, rotate_deg (multiples of 90),
   offset_uv_mm [u,v]. fasteners (only with register:"holes"): size M2..M10, nut true/false, washer true/false,
   head_on "a" or "b", or tapped_part "a"/"b" when that part's holes are threaded (then no nut).
 {"id":"m2","type":"coaxial","a":{"part":"<id>","feature":"H1"},"b":{"part":"<id>","feature":"S1"},
  "axial":{"mode":"plane_contact","a_plane":"P1","b_plane":"P2","gap_mm":0}}
   coaxial: a pin/boss (S*) goes into a hole (H*), or two holes line up. axial modes: plane_contact (two faces touch),
   plane_flush (two faces level), offset (offset_mm), or leave out axial to centre them. flip:true reverses the pin.
Rules: use ONLY feature ids that appear in the part descriptions (P* flat faces, H* holes, S* pins), or a selector like
{"kind":"plane","normal":"+z"}. Every part must be connected, directly or through others, to the ground part. A hole
pattern that does not match cannot be bolted - choose different faces instead of inventing holes. Without hardware (snap fit,
press fit, slide-in) use coaxial or plane_mate without "fasteners". Keep gap_mm 0 for touching parts."""


def parse_mates_json(text):
    import json
    t = str(text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise AssemblyError("the answer must be one JSON object")
    try:
        data = json.loads(t[a:b + 1])
    except Exception as e:
        raise AssemblyError(f"the answer is not valid JSON ({e})")
    if not isinstance(data.get("mates"), list) or not data["mates"]:
        raise AssemblyError('the JSON must contain a non-empty "mates" list')
    for i, m in enumerate(data["mates"]):
        m.setdefault("id", f"m{i + 1}")
        if not isinstance(m.get("a"), dict) or not isinstance(m.get("b"), dict):
            raise AssemblyError(f"mate {m['id']}: needs 'a' and 'b' objects with part and feature")
    return data


async def propose_mates(features_by_part, names, hints, llm, max_rounds=3):
    """Ask `llm(system, turns) -> text` for mates, validate them with the real solver, and feed errors back."""
    desc = "\n\n".join(feature_summary(F, names.get(pid, pid)).replace("PART '" + names.get(pid, pid) + "'",
                                                                       f"PART id '{pid}' ({names.get(pid, pid)})")
                       for pid, F in features_by_part.items())
    hint_txt = ""
    if hints:
        hint_txt = "\n\nWHAT THE DESIGNER SAID ABOUT EACH PART (use as a guide, the geometry above is the truth):\n" + \
                   "\n".join(f"- {pid}: {h}" for pid, h in hints.items() if h)
    first = next(iter(features_by_part))
    turns = [{"role": "user", "content": f"Parts:\n{desc}{hint_txt}\n\nThe first part ('{first}') is the default ground. "
                                         "Propose the mates."}]
    log = []
    err = "no answer"
    for rnd in range(1, max_rounds + 1):
        text = await llm(MATE_SYSTEM, turns)
        try:
            data = parse_mates_json(text)
            ground = data.get("ground") or first
            solve_assembly(features_by_part, data["mates"], ground)
            log.append({"round": rnd, "ok": True})
            return data["mates"], ground, log
        except AssemblyError as e:
            err = str(e)
        log.append({"round": rnd, "error": err})
        turns += [{"role": "assistant", "content": str(text)[:3000]},
                  {"role": "user", "content": f"Your assembly was rejected: {err}\nCorrect it and answer with the JSON object only."}]
    raise AssemblyError("no valid assembly plan after %d tries; last problem: %s" % (max_rounds, err))


# ---------------------------------------------------------------------------------------------------------------------
# the result the frontend draws
# ---------------------------------------------------------------------------------------------------------------------
class OccOps:
    """Kernel operations on build123d shapes; every method returns None on failure so one bad boolean never kills a result."""

    @staticmethod
    def apply(shape, T):
        from OCP.gp import gp_Trsf
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from build123d import Shape
        tr = gp_Trsf()
        R, t = T[:3, :3], T[:3, 3]
        tr.SetValues(R[0, 0], R[0, 1], R[0, 2], t[0], R[1, 0], R[1, 1], R[1, 2], t[1], R[2, 0], R[2, 1], R[2, 2], t[2])
        return Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape())

    @staticmethod
    def intersect_volume(a, b):
        try:
            inter = a.intersect(b)
            return float(inter.volume) if inter is not None else 0.0
        except Exception:
            return None

    @staticmethod
    def distance(a, b):
        try:
            return float(a.distance_to(b))
        except Exception:
            return None


def _world_bbox(F, T):
    bb = F["bbox_mm"]
    corners = np.array([[x, y, z] for x in (bb["min"][0], bb["max"][0]) for y in (bb["min"][1], bb["max"][1])
                        for z in (bb["min"][2], bb["max"][2])], float)
    w = (T[:3, :3] @ corners.T).T + T[:3, 3]
    return w.min(axis=0), w.max(axis=0)


def build_result(parts, mates, ground, ops=None, components_by_part=None, clearance=1.0, margin=1.0):
    """parts: [{id, name, features, shape (optional), stl_base64, internal_space}]. Solves the mates, plans hardware, checks
    interference and clearances, and checks that the components assigned to each part fit its measured cavities."""
    F = {p["id"]: p["features"] for p in parts}
    placed, reports, warnings = solve_assembly(F, mates, ground)
    hardware, notes = [], []
    for r in reports:
        hw, nt = plan_fasteners(r, placed, F)
        hardware += hw
        notes += nt
    wb = {pid: _world_bbox(F[pid], placed[pid]) for pid in F}
    world_shape = {}
    if ops is not None:
        for p in parts:
            if p.get("shape") is not None:
                try:
                    world_shape[p["id"]] = ops.apply(p["shape"], placed[p["id"]])
                except Exception as e:
                    warnings.append(f"could not place {p['id']} for interference checks: {e}")
    interference, clearances = [], []
    ids = [p["id"] for p in parts]
    for a, b in itertools.combinations(ids, 2):
        lo = np.maximum(wb[a][0], wb[b][0])
        hi = np.minimum(wb[a][1], wb[b][1])
        gap_box = float(np.max(lo - hi))                  # <= 0: the boxes overlap
        if gap_box > 2.0:
            continue
        sa, sb = world_shape.get(a), world_shape.get(b)
        if ops is None or sa is None or sb is None:
            continue
        vol = ops.intersect_volume(sa, sb) if gap_box <= 0 else 0.0
        if vol is not None and vol > 0.5:
            c = ((lo + hi) / 2).tolist()
            interference.append({"a": a, "b": b, "volume_mm3": round(vol, 2), "near": [round(v, 2) for v in c]})
        else:
            d = ops.distance(sa, sb)
            if d is not None:
                clearances.append({"a": a, "b": b, "min_gap_mm": round(d, 3)})
    packing = {}
    for pid, items in (components_by_part or {}).items():
        part = next((p for p in parts if p["id"] == pid), None)
        if not part:
            continue
        cavs = (part.get("internal_space") or {}).get("cavities") or []
        packing[pid] = check_requirement(cavs, items, clearance, margin) if cavs else \
            {"fits": False, "message": "this part has no measurable internal space", "tried": []}
    out_parts = []
    allb_lo = np.min([wb[i][0] for i in wb], axis=0)
    allb_hi = np.max([wb[i][1] for i in wb], axis=0)
    explode = {}
    for r in reports:
        if r.get("child") and r.get("explode_dir") and not r.get("closure_check"):
            explode[r["child"]] = r["explode_dir"]
    for p in parts:
        T = placed[p["id"]]
        out_parts.append({"id": p["id"], "name": p.get("name", p["id"]), "matrix": [round(float(v), 6) for v in T.reshape(-1)],
                          "bbox_world": {"min": [round(float(v), 3) for v in wb[p["id"]][0]],
                                         "max": [round(float(v), 3) for v in wb[p["id"]][1]]},
                          "stl_base64": p.get("stl_base64"), "internal_space": p.get("internal_space"),
                          "features": {"planes": [{k: v for k, v in f.items() if k in ("id", "center", "normal", "area")}
                                                  for f in F[p["id"]]["planes"][:12]],
                                       "cylinders": [{k: v for k, v in c.items() if k in ("id", "kind", "diameter", "center", "axis", "length")}
                                                     for c in F[p["id"]]["cylinders"]]},
                          "explode_dir": explode.get(p["id"], [0, 0, 1]), "is_ground": p["id"] == ground})
    public_reports = [{k: v for k, v in r.items() if not k.startswith("_")} for r in reports]
    ok = not interference
    return {"ok": ok, "ground": ground, "parts": out_parts, "mates": public_reports, "hardware": hardware,
            "bom": bill_of_materials(parts, hardware), "interference": interference, "clearances": clearances,
            "packing": packing, "warnings": warnings + notes,
            "assembly_bbox": {"min": [round(float(v), 3) for v in allb_lo], "max": [round(float(v), 3) for v in allb_hi]}}
