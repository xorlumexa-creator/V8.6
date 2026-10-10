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
def write_inp(path, node_ids, coords, elem_ids, elem_conn_ccx, fixed_nodes, forces, E_mpa, nu,
              analysis="static", buckle_modes=3, thermal=None, buckle_scale=1.0):
    """C3D10 model. analysis: 'static' (linear static) | 'buckling' (the same static step as the reference load, then
    *BUCKLE: the eigenvalues are the factors the load must be multiplied by to buckle the part) | 'thermal' (steady-state
    heat conduction + the thermal stress it causes, plus any mechanical forces).
    forces: {node: (fx, fy, fz)} N.  thermal: {"k_w_mk", "alpha_per_c", "t_ref_c", "temps": {node: degC},
    "cflux_mw": {node: mW}}  (mm-N-s-tonne units: 1 W/(m K) == 1 mW/(mm K), power in mW)."""
    if analysis not in ("static", "buckling", "thermal"):
        raise ValueError(f"unknown analysis {analysis!r}")
    if analysis == "thermal" and not thermal:
        raise ValueError("a thermal analysis needs the thermal data")
    used = set(int(v) for v in np.asarray(elem_conn_ccx).ravel())
    lines = ["*HEADING", f"lumexa {analysis}", "*NODE, NSET=NALL" if analysis == "thermal" else "*NODE"]
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
    lines += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{E_mpa:.9g}, {nu:.6g}"]
    if analysis == "thermal":
        lines += [f"*EXPANSION, ZERO={float(thermal['t_ref_c']):.9g}", f"{float(thermal['alpha_per_c']):.9g}",
                  "*CONDUCTIVITY", f"{float(thermal['k_w_mk']):.9g}"]
    lines += ["*SOLID SECTION, ELSET=EALL, MATERIAL=MAT"]
    if analysis == "thermal":
        lines += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f"NALL, {float(thermal['t_ref_c']):.9g}"]
    lines += ["*BOUNDARY"]
    for n in fx_nodes:
        lines.append(f"{n}, 1, 3")
    force_lines = []
    for n in sorted(forces):
        if int(n) in used:
            for dof, f in enumerate(forces[n], start=1):
                if abs(f) > 0.0:
                    force_lines.append(f"{int(n)}, {dof}, {f:.9g}")
    if analysis == "thermal":
        lines += ["*STEP", "*UNCOUPLED TEMPERATURE-DISPLACEMENT, STEADY STATE", "*BOUNDARY"]
        for n, t in sorted((thermal.get("temps") or {}).items()):
            if int(n) in used:
                lines.append(f"{int(n)}, 11, 11, {float(t):.9g}")
        cf = [(int(n), float(w)) for n, w in sorted((thermal.get("cflux_mw") or {}).items()) if int(n) in used and w]
        if cf:
            lines.append("*CFLUX")
            lines += [f"{n}, 11, {w:.9g}" for n, w in cf]
        if force_lines:
            lines += ["*CLOAD"] + force_lines
    else:
        lines += ["*STEP", "*STATIC", "*CLOAD"] + force_lines
    if fx_nodes:                                           # total reaction of the fixed set -> equilibrium check (.dat)
        lines += ["*NODE PRINT, NSET=NFIX, TOTALS=ONLY", "RF"]
    lines += ["*NODE FILE", "U, NT, RF" if analysis == "thermal" else "U, RF", "*EL FILE", "S", "*END STEP"]
    if analysis == "buckling":
        # CalculiX removes every earlier load at the start of a buckling step and scales ONLY the load written inside it,
        # so the reference load goes here (scaled to a small total, which also avoids CalculiX skipping the first mode
        # when the load is far above the buckling load). Multiply the factors by buckle_scale to get them per actual load.
        sc = float(buckle_scale)
        bl = []
        for n in sorted(forces):
            if int(n) in used:
                for dof, f in enumerate(forces[n], start=1):
                    if abs(f) > 0.0:
                        bl.append(f"{int(n)}, {dof}, {f * sc:.9g}")
        if not bl:
            raise ValueError("a buckling analysis needs a load")
        lines += ["*STEP", "*BUCKLE", f"{int(buckle_modes)}", "*CLOAD"] + bl + ["*NODE FILE", "U", "*END STEP"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_frd(text, first_only=False):
    """Read the nodal DISP, STRESS, FORC (reaction) and NDTEMP (temperature) blocks of a CalculiX .frd (ASCII).
    Default: later increments overwrite earlier ones. first_only=True keeps the FIRST block of each kind (the static
    step of a buckling run) and collects every DISP block, in order, in res['DISP_BLOCKS'] (block 0 = static, block 1 =
    buckling mode 1, ...). Data lines are ' -1' + I10 node id + n x E12.5."""
    comps = {"DISP": 3, "STRESS": 6, "FORC": 3, "NDTEMP": 1}
    res = {"DISP": {}, "STRESS": {}, "FORC": {}, "NDTEMP": {}, "DISP_BLOCKS": []}
    seen = set()
    block, ncomp, tgt = None, 0, None
    for line in text.splitlines():
        if line.startswith(" -4"):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else ""
            if name not in comps:
                block = None
                continue
            dup = name in seen
            seen.add(name)
            block, ncomp = name, comps[name]
            if dup and first_only:
                if name != "DISP":
                    block = None                           # later duplicates (mode stresses, ...) are ignored
                    continue
                tgt = {}                                   # a buckling-mode shape: kept in DISP_BLOCKS only
                res["DISP_BLOCKS"].append(tgt)
            else:
                tgt = {}
                res[name] = tgt
                if name == "DISP":
                    res["DISP_BLOCKS"].append(tgt)
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
            tgt[nid] = vals
    return res


def parse_buckling_factors(text):
    """Buckling factors from CalculiX's .dat ('B U C K L I N G   F A C T O R   O U T P U T' table). Returns a list of
    floats in mode order (may include negatives: the load would have to be reversed). First table wins."""
    import re
    lines = text.splitlines()
    row = re.compile(r"^\s*(\d+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
    for i, ln in enumerate(lines):
        if "bucklingfactoroutput" in re.sub(r"\s+", "", ln.lower()):
            out = []
            for nxt in lines[i + 1:i + 40]:
                m = row.match(nxt)
                if m:
                    out.append(float(m.group(2)))
                elif out:
                    break
            if out:
                return out
    return []


# ---------------------------------------------------------------------------------------------- results
def analyse(node_xyz, disp, stress, zone_frac=0.85, temps=None):
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
    if temps:
        tv = np.array([temps[i][0] for i in sorted(temps)], float)
        out["t_max_c"], out["t_min_c"] = float(tv.max()), float(tv.min())
        h = temps.get(ids[k])
        out["t_at_hotspot_c"] = float(h[0]) if h else None
    return out


def thermal_sanity(temps, target_by_node, has_flux, t_ref_c=None, tol_frac=0.005, tol_abs=0.5):
    """Physical checks on a steady-state thermal solution: every node with a prescribed temperature reached it, and
    (without heat input) no node is hotter or colder than the prescribed extremes (maximum principle)."""
    if not temps:
        return {"ok": False, "reason": "no temperature result"}
    span = max(max(target_by_node.values()) - min(target_by_node.values()), 1.0) if target_by_node else 1.0
    tol = max(tol_abs, tol_frac * span)
    errs = [abs(temps[n][0] - t) for n, t in target_by_node.items() if n in temps]
    out = {"ok": True, "bc_max_err_c": round(max(errs), 4) if errs else None}
    tv = [v[0] for v in temps.values()]
    out["t_range_c"] = [round(min(tv), 3), round(max(tv), 3)]
    if errs and max(errs) > tol:
        out["ok"] = False
        out["reason"] = "prescribed temperatures were not reached: a temperature boundary condition was not applied"
    elif target_by_node and not has_flux:
        lo, hi = min(target_by_node.values()), max(target_by_node.values())
        if min(tv) < lo - tol or max(tv) > hi + tol:
            out["ok"] = False
            out["reason"] = "temperatures outside the prescribed extremes with no heat input: the thermal solve is not trustworthy"
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


def parse_dat_totals(text, first=False):
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
            if first and found is not None:       # buckling runs print the totals again for the mode shapes
                break
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
    # A full cylinder's reported centre can sit on its surface instead of its axis (seen on 47 mm bores). Its bbox
    # centre is always right, so take the better of the two; the bbox and area terms below still have to agree.
    bc_d = (np.asarray(d["bbox_min"], float) + np.asarray(d["bbox_max"], float)) / 2.0
    bc_s = (np.asarray(s["bbox_min"], float) + np.asarray(s["bbox_max"], float)) / 2.0
    dc = min(dc, float(np.linalg.norm(bc_d - bc_s)) / extent)
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
