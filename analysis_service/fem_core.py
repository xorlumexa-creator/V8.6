"""Pure-python/numpy building blocks for the CalculiX solver service (no gmsh / ccx imports -> unit-testable).

Units everywhere: mm, N, MPa (so E in MPa, stress in MPa, displacement in mm)."""
import math
import numpy as np


# ---------------------------------------------------------------------------------------------- element helpers
def vm_from_tensor(s):
    """von Mises from rows of (sxx, syy, szz, sxy, syz, szx)."""
    s = np.asarray(s, float).reshape(-1, 6)
    sxx, syy, szz, sxy, syz, szx = s.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def tet10_gmsh_to_ccx(conn):
    """gmsh 10-node tet order -> CalculiX C3D10 order: the last two mid-edge nodes are swapped."""
    c = np.asarray(conn)
    return c[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]


def tri_node_weights(tri_conn, coords):
    """Consistent nodal weights for a uniform traction on a triangulated surface.
    tri6 (gmsh order: 3 corners, then mid-edge 0-1, 1-2, 2-0): corners carry 0, each mid node A/3.
    tri3: each node A/3. Returns {node_id: weight}; weights sum to the surface area."""
    w = {}
    for row in tri_conn:
        row = [int(v) for v in row]
        p0, p1, p2 = (np.asarray(coords[row[k]], float) for k in range(3))
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        nodes = row[3:6] if len(row) >= 6 else row[:3]
        for n in nodes:
            w[n] = w.get(n, 0.0) + area / 3.0
    return w


def distribute_force(weights, force_xyz):
    tot = sum(weights.values())
    if tot <= 0:
        raise ValueError("loaded surface has zero area")
    return {n: tuple(f * w / tot for f in force_xyz) for n, w in weights.items()}


# ---------------------------------------------------------------------------------------------- CalculiX input
def write_inp(path, node_ids, coords, elem_ids, elem_conn_ccx, fixed_nodes, forces, E_mpa, nu):
    """Linear static analysis with C3D10 elements. forces: {node: (fx, fy, fz)} (N)."""
    used = set(int(v) for v in np.asarray(elem_conn_ccx).ravel())
    lines = ["*HEADING", "lumexa static", "*NODE"]
    for nid, xyz in zip(node_ids, coords):
        if int(nid) in used:
            lines.append(f"{int(nid)}, {xyz[0]:.9g}, {xyz[1]:.9g}, {xyz[2]:.9g}")
    lines.append("*ELEMENT, TYPE=C3D10, ELSET=EALL")
    for eid, conn in zip(elem_ids, elem_conn_ccx):
        lines.append(f"{int(eid)}, " + ", ".join(str(int(v)) for v in conn))
    fx_nodes = [int(n) for n in sorted(fixed_nodes) if int(n) in used]
    if fx_nodes:                                           # node set of the fixed faces (for the reaction totals)
        lines.append("*NSET, NSET=NFIX")
        for i in range(0, len(fx_nodes), 10):
            lines.append(", ".join(str(n) for n in fx_nodes[i:i + 10]))
    lines += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{E_mpa:.9g}, {nu:.6g}",
              "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT", "*BOUNDARY"]
    for n in fx_nodes:
        lines.append(f"{n}, 1, 3")
    lines += ["*STEP", "*STATIC", "*CLOAD"]
    for n in sorted(forces):
        if int(n) in used:
            for dof, f in enumerate(forces[n], start=1):
                if abs(f) > 0.0:
                    lines.append(f"{int(n)}, {dof}, {f:.9g}")
    if fx_nodes:                                           # total reaction of the fixed set -> equilibrium check (.dat)
        lines += ["*NODE PRINT, NSET=NFIX, TOTALS=ONLY", "RF"]
    lines += ["*NODE FILE", "U, RF", "*EL FILE", "S", "*END STEP"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_frd(text):
    """Read the nodal DISP and STRESS blocks of a CalculiX .frd (ASCII). Later increments overwrite earlier ones.
    Data lines are ' -1' + I10 node id + n x E12.5."""
    res = {"DISP": {}, "STRESS": {}, "FORC": {}}
    block, ncomp = None, 0
    for line in text.splitlines():
        if line.startswith(" -4"):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else ""
            if name in res:
                block, ncomp = name, (6 if name == "STRESS" else 3)
                res[name] = {}
            else:
                block = None
            continue
        if line.startswith(" -3"):
            block = None
            continue
        if block and line.startswith(" -1"):
            body = line.rstrip("\n")
            try:
                nid = int(body[3:13])
                vals = tuple(float(body[13 + 12 * k:25 + 12 * k]) for k in range(ncomp))
            except ValueError:
                continue
            res[block][nid] = vals
    return res


# ---------------------------------------------------------------------------------------------- results
def analyse(node_xyz, disp, stress, zone_frac=0.85):
    """Peak von Mises (+ where, + the zone within 15 % of it) and peak displacement magnitude."""
    if not stress:
        raise ValueError("no STRESS block in the result file")
    ids = sorted(stress)
    vm = vm_from_tensor([stress[i] for i in ids])
    xyz = np.array([node_xyz[i] for i in ids], float)
    k = int(np.argmax(vm))
    out = {"vm_mpa": float(vm[k]), "hotspot_xyz_mm": [float(c) for c in xyz[k]]}
    m = vm >= zone_frac * vm[k]
    out["hot_zone_mm"] = {"min": xyz[m].min(axis=0).tolist(), "max": xyz[m].max(axis=0).tolist(),
                          "n_nodes": int(m.sum()), "n_total": int(len(ids))}
    if disp:
        dids = sorted(disp)
        mag = np.linalg.norm(np.array([disp[i] for i in dids], float), axis=1)
        j = int(np.argmax(mag))
        out["disp_mm"] = float(mag[j])
        out["disp_xyz_mm"] = [float(c) for c in np.array([node_xyz[dids[j]]], float)[0]]
    else:
        out["disp_mm"] = 0.0
    return out


def bc_sanity(disp, fixed_nodes, load_nodes_by_load, force_by_load, tol_fixed=0.01):
    """Physical checks that need no second solver: fixed nodes did not move, and every load moved its own
    face in the direction it pushes (positive work)."""
    if not disp:
        return {"ok": False, "reason": "no displacement result"}
    mags = {n: float(np.linalg.norm(v)) for n, v in disp.items()}
    peak = max(mags.values()) or 1e-30
    fixed_move = max((mags.get(n, 0.0) for n in fixed_nodes), default=0.0)
    out = {"ok": True, "fixed_max_move_over_peak": round(fixed_move / peak, 6), "loads": []}
    if fixed_move > tol_fixed * peak:
        out["ok"] = False
        out["reason"] = "the fixed faces moved: constraint not applied where expected"
    for lid, nodes in load_nodes_by_load.items():
        f = np.asarray(force_by_load[lid], float)
        mean_u = np.mean([disp[n] for n in nodes if n in disp], axis=0) if any(n in disp for n in nodes) else np.zeros(3)
        work = float(np.dot(mean_u, f))
        out["loads"].append({"id": lid, "work_sign": "positive" if work > 0 else "NOT positive"})
        if work <= 0 and np.linalg.norm(f) > 0:
            out["ok"] = False
            out["reason"] = f"load '{lid}' does not push its face along the force (negative work): wrong face or direction"
    return out


def parse_dat_totals(text):
    """Total reaction force (fx, fy, fz) of the fixed node set from CalculiX's .dat file (*NODE PRINT, TOTALS=ONLY).
    The value sits on the first numeric line after 'total force (fx,fy,fz) ...'. The last one in the file wins."""
    lines = text.splitlines()
    found = None
    for i, ln in enumerate(lines):
        if "total force" in ln.lower():
            for nxt in lines[i + 1:i + 5]:
                parts = nxt.split()
                if len(parts) >= 3:
                    try:
                        found = tuple(float(x) for x in parts[:3])
                        break
                    except ValueError:
                        continue
    return found


def equilibrium(reactions, force_by_load, total=None):
    """Sum of the reaction forces must cancel the applied loads. A big mismatch means a load or a constraint was
    not applied the way we think (the strongest single check there is for an FE setup)."""
    if total is not None:
        R = np.asarray(total, float)
    elif reactions:
        R = np.sum(np.array(list(reactions.values()), float), axis=0)
    else:
        return None
    F = np.sum([np.asarray(f, float) for f in force_by_load.values()], axis=0)
    nf = float(np.linalg.norm(F))
    if nf <= 0:
        return None
    err = float(np.linalg.norm(R + F)) / nf
    return {"error_pct": round(100.0 * err, 3), "applied_n": [round(float(v), 3) for v in F],
            "reaction_n": [round(float(v), 3) for v in R]}


# ---------------------------------------------------------------------------------------------- face matching
def _cost(d, s, extent):
    dc = float(np.linalg.norm(np.asarray(d["center"], float) - np.asarray(s["center"], float))) / extent
    da = abs(float(s["area"]) / max(float(d["area"]), 1e-12) - 1.0)
    db = (float(np.linalg.norm(np.asarray(d["bbox_min"], float) - np.asarray(s["bbox_min"], float))) +
          float(np.linalg.norm(np.asarray(d["bbox_max"], float) - np.asarray(s["bbox_max"], float)))) / extent
    return dc + 0.5 * da + db


def match_faces(descs, surfaces, extent, accept=0.06):
    """Map B-rep faces described by position/area/bbox (from the client's CAD) to gmsh surfaces.
    A face split into several surfaces is matched as a group. Returns one dict per desc:
    {tags:[...], cost, mode:'single'|'group'} or raises ValueError listing the best candidates."""
    out = []
    tol = 1e-3 * extent + 1e-6
    for d in descs:
        ranked = sorted(((_cost(d, s, extent), s) for s in surfaces), key=lambda t: t[0])
        if ranked and ranked[0][0] <= accept:
            out.append({"tags": [ranked[0][1]["tag"]], "cost": round(ranked[0][0], 5), "mode": "single"})
            continue
        lo, hi = np.asarray(d["bbox_min"], float) - tol, np.asarray(d["bbox_max"], float) + tol
        grp = [s for s in surfaces if np.all(np.asarray(s["bbox_min"]) >= lo) and np.all(np.asarray(s["bbox_max"]) <= hi)
               and np.all(np.asarray(s["center"]) >= lo) and np.all(np.asarray(s["center"]) <= hi)]
        if len(grp) > 1 and abs(sum(s["area"] for s in grp) / max(d["area"], 1e-12) - 1.0) < 0.05:
            out.append({"tags": [s["tag"] for s in grp], "cost": 0.0, "mode": "group"})
            continue
        best = [(round(c, 3), s["tag"], [round(v, 1) for v in s["center"]], round(s["area"], 1)) for c, s in ranked[:3]]
        raise ValueError(f"could not match face at {[round(v, 1) for v in d['center']]} area {round(d['area'], 1)} "
                         f"to a gmsh surface (best candidates cost/tag/center/area: {best})")
    return out


def choose_lc(diag_mm, min_thickness_mm=None):
    """Target element size: at least ~2 element layers through the thinnest section, never coarser than diag/12."""
    lc = diag_mm / 20.0
    if min_thickness_mm and min_thickness_mm > 0:
        lc = min(lc, min_thickness_mm / 2.0)
    return float(min(max(lc, diag_mm / 150.0), diag_mm / 12.0))


def unit_scale(gmsh_extent, expected_extent):
    """Factor to bring the gmsh geometry to mm when the STEP units differ (m or inch); 1.0 when sizes agree."""
    if gmsh_extent <= 0 or expected_extent <= 0:
        return 1.0
    r = expected_extent / gmsh_extent
    for k in (1000.0, 0.001, 25.4, 1 / 25.4):
        if abs(r / k - 1.0) < 0.05:
            return k
    return 1.0
