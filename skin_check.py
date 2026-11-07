"""Checks of a visual skin (a mesh made in Blender or anywhere else) against the engineering geometry. Pure numpy.

Lumexa keeps two representations of a product: the engineering skeleton (build123d / OpenCASCADE, exact, goes to FEA) and the visual skin
(a mesh: organic surfaces, fairings, appearance). The skin is never converted back into CAD. Instead the engineering side states what the
skin must respect (the interfaces: components that must stay enclosed with clearance, motors and props that must stay free, screw heads
that must stay reachable) and this module checks a skin mesh against them:

  * mesh quality: closed, consistently oriented, no degenerate triangles, aspect ratios (is it even usable for meshing?),
  * wall thickness of the shell (rays along the inward normals),
  * keep-out boxes: no triangle may cut through one, the clearance must hold, enclosed ones must really be inside,
  * access: lines or cylinders that must stay open (a propeller disc, a screwdriver path).
A passing skin is a statement about appearance geometry only. Whether anything is strong enough is decided on the skeleton.
"""
from __future__ import annotations

import math
import re
import struct

import numpy as np

import capacity

EPS = 1e-9


# ------------------------------------------------------------------------------------------------------------ reading meshes
def parse_stl(data: bytes) -> np.ndarray:
    """Triangles (n, 3, 3) from a binary or ASCII STL."""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data) and n > 0:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float).copy()
    txt = data.decode("utf-8", "ignore")
    nums = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt)
    if not nums or len(nums) % 3:
        raise ValueError("not a readable STL (neither binary nor ASCII triangles)")
    return np.array(nums, dtype=float).reshape(-1, 3, 3)


def to_stl(tris: np.ndarray) -> bytes:
    t = np.asarray(tris, dtype="<f4")
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    nrm = np.cross(a, b)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, t
    return b"lumexa".ljust(80, b" ") + struct.pack("<I", len(t)) + rec.tobytes()


def _areas(tris):
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


def signed_volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def oriented(tris):
    """The mesh with outward-facing triangles (flipped when its signed volume is negative)."""
    return tris[:, ::-1].copy() if signed_volume(tris) < 0 else tris


# ------------------------------------------------------------------------------------------------------------ quality
def mesh_quality(tris, weld_mm=1e-4):
    """Closedness and triangle quality. 'fem_ready' says the mesh could go to a mesher as a closed shell; it does not say the shape is good."""
    tris = np.asarray(tris, float)
    key = np.round(tris.reshape(-1, 3) / weld_mm).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    area = _areas(tris)
    degenerate = (area < 1e-9) | (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    f = faces[~degenerate]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    _, dcounts = np.unique(directed, axis=0, return_counts=True)
    boundary, nonmanifold, flipped = int((counts == 1).sum()), int((counts > 2).sum()), int((dcounts > 1).sum())
    tri = tris[~degenerate]
    e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1), np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)
    longest = e.max(axis=1)
    altitude = 2 * _areas(tri) / np.maximum(longest, EPS)
    aspect = longest / np.maximum(altitude, EPS)
    watertight = boundary == 0 and nonmanifold == 0 and flipped == 0
    return {
        "triangles": int(len(tris)), "degenerate": int(degenerate.sum()), "boundary_edges": boundary, "non_manifold_edges": nonmanifold,
        "inconsistent_orientation": flipped, "watertight": bool(watertight), "signed_volume_mm3": round(signed_volume(tris), 1),
        "aspect_max": round(float(aspect.max()), 1) if len(aspect) else None, "aspect_p95": round(float(np.percentile(aspect, 95)), 1) if len(aspect) else None,
        "min_edge_mm": round(float(e.min()), 4) if len(e) else None,
        "fem_ready": bool(watertight and not degenerate.any() and len(aspect) and aspect.max() < 60),
    }


# ------------------------------------------------------------------------------------------------------------ ray / box / distance
def _ray_hits(origins, dirs, tris, chunk=None):
    """Nearest positive hit distance (inf when none) of each ray against all triangles, double sided (Moller-Trumbore)."""
    chunk = chunk or capacity.profile()["chunk_entries"]               # ray x triangle pairs per block: bounds the memory
    o, d = np.atleast_2d(origins).astype(float), np.atleast_2d(dirs).astype(float)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    out = np.full(len(o), np.inf)
    step = max(1, chunk // max(len(tris), 1))
    for s in range(0, len(o), step):
        oo, dd = o[s:s + step][:, None, :], d[s:s + step][:, None, :]
        p = np.cross(dd, e2[None])
        det = np.einsum("rtk,rtk->rt", e1[None] * np.ones_like(p), p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = oo - v0[None]
        u = np.einsum("rtk,rtk->rt", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("rtk,rtk->rt", dd * np.ones_like(q), q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-7)
        out[s:s + step] = np.where(hit, t, np.inf).min(axis=1)
    return out


def _count_hits(origin, direction, tris):
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    d = np.asarray(direction, float)
    p = np.cross(d, e2)
    det = np.einsum("tk,tk->t", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = np.asarray(origin, float) - v0
    u = np.einsum("tk,tk->t", tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum("tk,tk->t", e2, q) * inv
    return int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-7)).sum())


def point_inside(point, tris):
    """Inside a closed mesh? Ray parity along a slightly skewed direction (so it avoids edges and vertices)."""
    d = np.array([1.0, 0.01234, 0.00713])
    return _count_hits(point, d / np.linalg.norm(d), tris) % 2 == 1


def enclosed_fraction(point, tris, n_dirs=32):
    """Fraction of rays from `point` that hit the skin (Fibonacci directions). 1.0 means fully surrounded, 0.0 means in the open.
    Parity would fail for a hollow shell (inner and outer surface) and for a skin with a few openings, so this is used for 'enclosed'."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    hits = _ray_hits(np.tile(np.asarray(point, float), (n_dirs, 1)), dirs, tris)
    return float(np.isfinite(hits).mean())


def tris_hit_box(tris, lo, hi):
    """Boolean per triangle: does it overlap the interior of the axis-aligned box (separating axis test)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2 - 1e-6                   # shrink a hair: touching is not cutting through
    v = tris - c
    cand = np.nonzero(np.all(v.min(axis=1) <= h, axis=1) & np.all(v.max(axis=1) >= -h, axis=1))[0]
    hit = np.zeros(len(tris), bool)
    if not len(cand):
        return hit
    v = v[cand]
    e = [v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]]
    n = np.cross(e[0], e[1])
    sep = np.abs(np.einsum("tk,tk->t", n, v[:, 0])) > (np.abs(n) * h).sum(axis=1)
    for i in range(3):
        unit = np.zeros(3)
        unit[i] = 1.0
        for j in range(3):
            ax = np.cross(unit, e[j])
            p = np.einsum("tvk,tk->tv", v, ax)
            r = (np.abs(ax) * h).sum(axis=1)
            sep |= (p.min(axis=1) > r) | (p.max(axis=1) < -r)
    hit[cand] = ~sep
    return hit


def _closest_dist(points, tris, chunk=None):
    """Smallest distance from each point to the triangle soup (exact point-triangle distance, Ericson)."""
    P = np.asarray(points, float)[:, None, :]
    chunk = chunk or max(1, capacity.profile()["chunk_entries"] // 4 // max(len(points), 1))     # ~25 float arrays of points x triangles x 3
    best = np.full(len(points), np.inf)
    for s in range(0, len(tris), chunk):
        T = tris[s:s + chunk]
        a, b, c = T[None, :, 0], T[None, :, 1], T[None, :, 2]
        ab, ac, ap, bp, cp = b - a, c - a, P - a, P - b, P - c
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = np.where(np.abs(va + vb + vc) < EPS, 1.0, va + vb + vc)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]
            ab_pt = a + ab * (d1 / (d1 - d3))[..., None]
            ac_pt = a + ac * (d2 / (d2 - d6))[..., None]
            bc_pt = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0), (d6 >= 0) & (d5 <= d6),
                 (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)]
        cp_ = np.select([x[..., None] for x in conds], [a + 0 * P, b + 0 * P, ab_pt, c + 0 * P, ac_pt, bc_pt], default=inside)
        best = np.minimum(best, np.linalg.norm(P - cp_, axis=-1).min(axis=1))
    return best


def _box_points(lo, hi, n=4):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    g = np.linspace(0, 1, n)
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for side in (lo[ax], hi[ax]):
            for a in g:
                for b in g:
                    p = np.zeros(3)
                    p[ax], p[o[0]], p[o[1]] = side, lo[o[0]] + a * (hi[o[0]] - lo[o[0]]), lo[o[1]] + b * (hi[o[1]] - lo[o[1]])
                    pts.append(p)
    return np.array(pts)


# ------------------------------------------------------------------------------------------------------------ wall thickness
def wall_thickness(tris, samples=150, seed=7, cap_mm=200.0):
    """Local wall thickness at random surface points: a ray from just inside the surface along the inward normal, up to the next surface."""
    rng = np.random.default_rng(seed)
    area = _areas(tris)
    ok = area > 1e-9
    idx = rng.choice(np.nonzero(ok)[0], size=min(samples, int(ok.sum())), p=area[ok] / area[ok].sum())
    t = tris[idx]
    r = rng.random((len(idx), 2))
    flip = r.sum(axis=1) > 1
    r[flip] = 1 - r[flip]
    pts = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)
    d = _ray_hits(pts - nrm * 1e-3, -nrm, tris)
    return np.minimum(d + 1e-3, cap_mm)


# ------------------------------------------------------------------------------------------------------------ the check
class TooLarge(ValueError):
    """The skin has more triangles than this device can check safely. The message says how much to decimate."""
    def __init__(self, n, limit, profile_name):
        self.n, self.limit = n, limit
        super().__init__(f"the skin has {n:,} triangles; this device ({profile_name} memory profile) checks up to {limit:,}. "
                         f"In Blender add a Decimate modifier with ratio {limit * 0.8 / n:.2f} (or use the generated script, which decimates), then export again.")


def check(tris, interfaces, params=None):
    """tris: skin mesh (n,3,3) mm in the assembly frame. interfaces: list of
       {id, kind, mode: 'inside'|'outside', box:{min,max}, clearance_mm?, access?:{origin,axis,length_mm,radius_mm?}}."""
    cap = capacity.profile()
    p = {"min_wall_mm": 1.2, "clearance_mm": 2.0, "wall_samples": cap["wall_samples"]}
    p.update(params or {})
    tris = np.asarray(tris, float)
    if len(tris) > cap["max_skin_triangles"]:
        raise TooLarge(len(tris), cap["max_skin_triangles"], cap["name"])
    tris = oriented(tris)
    tmin, tmax = tris.min(axis=1), tris.max(axis=1)
    q = mesh_quality(tris)
    checks = []

    def add(name, ok, detail, **extra):
        checks.append({"name": name, "ok": ok, "detail": detail, **extra})

    add("Closed surface", q["watertight"], "closed and consistently oriented" if q["watertight"] else
        f"{q['boundary_edges']} open edges, {q['non_manifold_edges']} non-manifold edges, {q['inconsistent_orientation']} flipped edges: fix the mesh before relying on any check that needs a closed shell")
    t = wall_thickness(tris, p["wall_samples"])
    thin = int((t < p["min_wall_mm"]).sum())
    add("Wall thickness", thin == 0, f"thinnest {t.min():.2f} mm (required {p['min_wall_mm']} mm), median {np.median(t):.2f} mm" + (f", {thin} of {len(t)} samples too thin" if thin else ""),
        min_mm=round(float(t.min()), 3), median_mm=round(float(np.median(t)), 3), thin_samples=thin)
    for it in interfaces:
        box = it.get("box")
        if box:
            lo, hi = np.array(box["min"], float), np.array(box["max"], float)
            clr = float(it.get("clearance_mm", p["clearance_mm"]))
            cut = int(tris_hit_box(tris, lo, hi).sum())
            if cut:
                add(f"{it['id']}: keep-out", False, f"{cut} skin triangles cut through this {it.get('kind', 'volume')}", interface=it["id"])
                continue
            # Only triangles within reach of the clearance can matter: skip the rest instead of measuring against the whole mesh.
            limit = max(clr, 0.0) + 0.5
            near = tris[np.all(tmax >= lo - limit, axis=1) & np.all(tmin <= hi + limit, axis=1)]
            if len(near):
                dist = float(_closest_dist(_box_points(lo, hi), near).min())
                ok = dist >= clr
                detail = f"closest skin surface {dist:.2f} mm away (clearance required {clr:g} mm)"
            else:
                dist, ok = limit, True
                detail = f"no skin surface within {limit:.1f} mm (clearance required {clr:g} mm)"
            if it.get("mode", "inside") == "inside":
                frac = enclosed_fraction((lo + hi) / 2, tris)
                if frac < 0.85:
                    add(f"{it['id']}: enclosed", False, f"the component is not inside the skin (only {frac * 100:.0f}% of the directions from it meet the skin)", interface=it["id"])
                    continue
                add(f"{it['id']}: enclosed with clearance", ok, detail + f"; {frac * 100:.0f}% of directions from it meet the skin", interface=it["id"], distance_mm=round(dist, 3))
            else:
                frac = enclosed_fraction((lo + hi) / 2, tris)
                if frac >= 0.85:
                    add(f"{it['id']}: stays outside", False, f"the skin closes around this {it.get('kind', 'volume')} ({frac * 100:.0f}% of the directions from it meet the skin); it has to stay outside", interface=it["id"])
                    continue
                add(f"{it['id']}: clearance", ok, detail, interface=it["id"], distance_mm=round(dist, 3))
        acc = it.get("access")
        if acc:
            o, ax = np.array(acc["origin"], float), np.array(acc["axis"], float)
            ax /= max(np.linalg.norm(ax), EPS)
            L, R = float(acc.get("length_mm", 30.0)), float(acc.get("radius_mm", 0.0))
            u = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
            u /= np.linalg.norm(u)
            w = np.cross(ax, u)
            offs = [np.zeros(3)] + ([u * R, -u * R, w * R, -w * R] if R > 0 else [])
            hits = _ray_hits(np.array([o + off for off in offs]), np.tile(ax, (len(offs), 1)), tris)
            blocked = int((hits < L).sum())
            add(f"{it['id']}: access", blocked == 0, f"clear for {L:g} mm" + (f" (radius {R:g} mm)" if R else "") if blocked == 0 else
                f"the skin blocks {blocked} of {len(offs)} test lines within {L:g} mm (first at {float(hits.min()):.1f} mm)", interface=it["id"])
    graded = [c for c in checks if c["ok"] is not None]
    verdict = "fail" if any(c["ok"] is False for c in graded) else ("incomplete" if any(c["ok"] is None for c in checks) else "pass")
    return {"verdict": verdict, "quality": q, "checks": checks, "params": p,
            "note": "Appearance geometry only. This says whether the skin respects the engineering interfaces; strength is decided on the skeleton (Gmsh + CalculiX)."}
