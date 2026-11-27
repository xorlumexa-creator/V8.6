"""Maps a CFD wall-traction field onto the FE surface mesh as nodal forces (CalculiX *CLOAD).

Input  : the surface payload returned by the CFD service: one row per CFD wall face with
         [cx, cy, cz (mm), area (mm2), tx, ty, tz (MPa, force the fluid exerts on the part per unit area: pressure + shear), p (Pa)].
Method : conservative. The force of every CFD face (t * area) goes to the FE surface triangle closest to it (exact
         point-to-triangle distance) and is split over that triangle's three mid-edge nodes, the same lumping the solver uses
         for face loads. So the total force is conserved by construction, whatever the two meshes look like.
Guards : if the CFD and FE geometry do not coincide (wrong units, shifted part) the median gap is large -> error.
Nodes on the fixed faces are skipped (their share goes straight into the supports) and reported as `lost_on_fixed_n`.
"""
import base64

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:                                    # numpy fallback below
    cKDTree = None


def decode_surface(payload):
    cols = payload["columns"]
    raw = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype="<f4")
    return raw.reshape(int(payload["n"]), len(cols)).astype(float)


def closest_point_on_triangle(P, A, B, C):
    """Vectorised closest point (Ericson, Real-Time Collision Detection 5.1.5). All inputs (n, 3)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        s = va + vb + vc
        s = np.where(np.abs(s) < 1e-30, 1.0, s)
        Q = A + ab * (vb / s)[:, None] + ac * (vc / s)[:, None]                       # inside the face
        t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1.0, (d4 - d3) + (d5 - d6))
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        Q[m] = (B + (C - B) * t[:, None])[m]                                          # edge BC
        t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1.0, d2 - d6)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        Q[m] = (A + ac * t[:, None])[m]                                               # edge AC
        m = (d6 >= 0) & (d5 <= d6)
        Q[m] = C[m]                                                                   # vertex C
        t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1.0, d1 - d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        Q[m] = (A + ab * t[:, None])[m]                                               # edge AB
        m = (d3 >= 0) & (d4 <= d3)
        Q[m] = B[m]                                                                   # vertex B
        m = (d1 <= 0) & (d2 <= 0)
        Q[m] = A[m]                                                                   # vertex A
    return Q


def _knn(points, queries, k):
    k = min(k, len(points))
    if cKDTree is not None:
        _, idx = cKDTree(points).query(queries, k=k)
        return idx.reshape(len(queries), k)
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), 2000):
        d = ((queries[i:i + 2000, None, :] - points[None, :, :]) ** 2).sum(2)
        out[i:i + 2000] = np.argsort(d, axis=1)[:, :k]
    return out


def map_cfd_to_nodes(cfd, tri_arrays, node_xyz, fixed_nodes, bbox_diag_mm):
    """cfd: {'surface': payload, 'scale': 1.0}. tri_arrays: list of (m, 3|6) int arrays of FE surface triangles
    (gmsh node tags). Returns ({node: (fx, fy, fz)}, diagnostics). Raises ValueError with a readable message."""
    S = decode_surface(cfd["surface"])
    scale = float(cfd.get("scale", 1.0))
    if not len(S):
        raise ValueError("the CFD surface is empty")
    corners, dist = [], []
    for arr in tri_arrays:
        arr = np.asarray(arr, int)
        if not len(arr):
            continue
        corners.append(arr[:, :3])
        dist.append(arr[:, 3:6] if arr.shape[1] == 6 else arr[:, :3])
    if not corners:
        raise ValueError("the FE mesh has no surface triangles")
    corners, dist = np.concatenate(corners), np.concatenate(dist)
    xyz = lambda ids: np.array([node_xyz[int(n)] for n in ids.ravel()], float).reshape(ids.shape + (3,))
    A, B, C = (xyz(corners[:, i]) for i in range(3))
    cent = (A + B + C) / 3.0
    P = S[:, 0:3]
    F = S[:, 4:7] * S[:, 3:4] * scale                          # N on the part, per CFD face

    cand = _knn(cent, P, 8)                                    # candidate triangles by centroid, then exact distance
    n, k = cand.shape
    Pk = np.repeat(P[:, None, :], k, axis=1).reshape(-1, 3)
    ci = cand.reshape(-1)
    Q = closest_point_on_triangle(Pk, A[ci], B[ci], C[ci])
    d = np.linalg.norm(Pk - Q, axis=1).reshape(n, k)
    best = cand[np.arange(n), d.argmin(axis=1)]
    gap = d.min(axis=1)

    med, mx = float(np.median(gap)), float(gap.max())
    if med > 0.05 * bbox_diag_mm:
        raise ValueError(f"the CFD surface and the FE mesh do not coincide (median gap {med:.2f} mm, part size {bbox_diag_mm:.0f} mm): "
                         "check that both come from the same geometry in the same units")

    tri_force = np.zeros((len(corners), 3))
    np.add.at(tri_force, best, F)
    forces, lost = {}, np.zeros(3)
    fixed = set(int(x) for x in fixed_nodes)
    for j in np.nonzero(np.abs(tri_force).sum(axis=1) > 0)[0]:
        share = tri_force[j] / 3.0
        for node in dist[j]:
            node = int(node)
            if node in fixed:
                lost += share
                continue
            cur = forces.get(node, (0.0, 0.0, 0.0))
            forces[node] = (cur[0] + share[0], cur[1] + share[1], cur[2] + share[2])
    applied = np.zeros(3)
    for v in forces.values():
        applied += v
    cfd_net = F.sum(axis=0)
    cog = (cent).mean(axis=0)
    M_cfd = np.cross(P - cog, F).sum(axis=0)
    M_fe = sum(np.cross(np.asarray(node_xyz[nd]) - cog, np.asarray(v)) for nd, v in forces.items()) + np.zeros(3)
    nrm = lambda v: float(np.linalg.norm(v))
    diag = {"cfd_faces": int(n), "fe_triangles": int(len(corners)), "loaded_nodes": len(forces),
            "cfd_net_n": [round(float(v), 5) for v in cfd_net], "applied_net_n": [round(float(v), 5) for v in applied],
            "lost_on_fixed_n": [round(float(v), 5) for v in lost],
            "net_error_pct": round(100.0 * nrm(cfd_net - lost - applied) / max(nrm(cfd_net), 1e-12), 4),
            "moment_cfd_nmm": [round(float(v), 3) for v in M_cfd], "moment_applied_nmm": [round(float(v), 3) for v in M_fe],
            "median_gap_mm": round(med, 4), "max_gap_mm": round(mx, 4), "p_max_pa": round(float(S[:, 7].max()), 3), "load_scale": scale,
            "mapping": "conservative nearest-triangle"}
    return forces, diag
