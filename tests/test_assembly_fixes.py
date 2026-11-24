"""Pure-numpy tests (no CAD kernel) for the assembly fixes: cavity measurement, component scope, hardware planning.
Run:  python -m pytest tests/test_assembly_fixes.py   (or just  python tests/test_assembly_fixes.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import assembly as A  # noqa: E402


# ---------------------------------------------------------------- synthetic voxel parts (same dimensions as the generated scripts)
def _grid(L, W, H, pitch):
    pad = 1
    n = np.ceil(np.array([L, W, H]) / pitch).astype(int) + 2 * pad
    origin = -np.array([L, W, H]) / 2 - pad * pitch
    axes = [origin[i] + (np.arange(n[i]) + .5) * pitch for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return X, Y, Z, origin


def _box(X, Y, Z, c, d):
    return (abs(X - c[0]) <= d[0] / 2) & (abs(Y - c[1]) <= d[1] / 2) & (abs(Z - c[2]) <= d[2] / 2)


def _cyl(X, Y, Z, c, r, h, ax):
    d = [X - c[0], Y - c[1], Z - c[2]]
    a = d.pop(ax)
    return (d[0] ** 2 + d[1] ** 2 <= r * r) & (abs(a) <= h / 2)


def _measure(L, W, H, pitch, build):
    X, Y, Z, origin = _grid(L, W, H, pitch)
    occ = build(X, Y, Z)
    orig = A.voxelize_triangles
    A.voxelize_triangles = lambda tris, pitch=None: (occ, origin, pitch_, {"odd_columns": 0, "columns_hit": 1})
    pitch_ = pitch
    try:
        return A.analyze_internal_space(np.zeros((1, 3, 3)), None)
    finally:
        A.voxelize_triangles = orig


def rear_shell(X, Y, Z):          # the generated "Fuselage rear shell": 290 x 80 x 66, cavity 280 x 72 x 62, many holes
    L, W, H, TS, TF, FL, HD = 290, 80, 66, 4, 4, 5, 3.4
    s = _box(X, Y, Z, (0, 0, 0), (L, W, H))
    s &= ~_box(X, Y, Z, (0, 0, (TF + 1) / 2), (L - 2 * FL, W - 2 * TS, H - TF + 1))
    for y in (-30, 30):
        for z in (-23, 23):
            s &= ~_cyl(X, Y, Z, (-L / 2 + FL / 2, y, z), HD / 2, FL + 4, 0)
    s &= ~_box(X, Y, Z, (-L / 2 + FL / 2, 0, 0), (FL + 4, 24, 12))
    for y in (-20, 20):
        for z in (-20, 20):
            s &= ~_cyl(X, Y, Z, (L / 2 - FL / 2, y, z), HD / 2, FL + 4, 0)
    s &= ~_cyl(X, Y, Z, (L / 2 - FL / 2, 0, 0), 8, FL + 4, 0)
    for x in (-132, 46):
        for y in (-23, 23):
            s &= ~_cyl(X, Y, Z, (x, y, -H / 2 + TF / 2), HD / 2, TF + 4, 2)
    s &= ~_box(X, Y, Z, (24, 0, -H / 2 + TF / 2), (18, 24, TF + 4))
    for side in (-1, 1):
        yc = side * (W / 2 - TS / 2)
        for x in (-50, -12, 12, 50):
            s &= ~_cyl(X, Y, Z, (x, yc, 18), HD / 2, TS + 4, 1)
        for x in (-62, 79):
            s &= ~_cyl(X, Y, Z, (x, yc, -15), HD / 2, TS + 4, 1)
        s &= ~_cyl(X, Y, Z, (-32, yc, 0), 9, TS + 4, 1)
        s &= ~_cyl(X, Y, Z, (-4, yc, 6), 5, TS + 4, 1)
        s &= ~_cyl(X, Y, Z, (16, yc, 6), 5.5, TS + 4, 1)
    return s


def front_shell(X, Y, Z):         # 110 x 80 x 66, open-top tray, cavity 102 x 74 x 63
    L, W, H, T, FL = 110, 80, 66, 3, 5
    s = _box(X, Y, Z, (0, 0, 0), (L, W, H))
    s &= ~_box(X, Y, Z, ((T - FL) / 2, 0, (T + 1) / 2), (L - T - FL, W - 2 * T, H - T + 1))
    return s


def cradle(X, Y, Z):              # 162 x 62 x 48 tray, cavity 156 x 56 x 45, two ears with 4.0 mm insert holes
    L, W, H, T, EAR_T, EAR_OUT = 162, 62, 48, 3, 8, 16
    s = _box(X, Y, Z, (0, 0, 0), (L, W, H))
    s &= ~_box(X, Y, Z, (0, 0, (T + 1) / 2), (L - 2 * T, W - 2 * T, H - T + 1))
    for sx in (-1, 1):
        s |= _box(X, Y, Z, (sx * (L / 2 + (EAR_OUT - 2) / 2), 0, H / 2 - EAR_T / 2), (EAR_OUT + 2, W, EAR_T))
    for sx in (-1, 1):
        for sy in (-1, 1):
            s &= ~_cyl(X, Y, Z, (sx * (L / 2 + EAR_OUT / 2), sy * 23, H / 2 - EAR_T / 2), 2.0, EAR_T + 2, 2)
    return s


def _space(r):
    return [c for c in r["cavities"] if c["role"] == "space"]


def test_long_pocket_is_measured_full_length():
    r = _measure(290, 80, 66, 1.97, rear_shell)
    best = _space(r)[0]["inscribed"]["size"]
    assert best[0] > 270, f"rear shell cavity length {best[0]} (true 280)"     # was 88.7
    assert 70 < best[1] < 76 and 58 < best[2] < 66


def test_front_shell_and_cradle():
    f = _space(_measure(110, 80, 66, 1.5, front_shell))[0]["inscribed"]["size"]
    assert abs(f[0] - 102) < 3 and abs(f[1] - 74) < 3 and abs(f[2] - 63) < 3
    cr = _measure(194, 62, 48, 1.42, cradle)
    c = _space(cr)[0]["inscribed"]["size"]
    assert abs(c[0] - 156) < 3 and abs(c[1] - 56) < 3 and abs(c[2] - 45) < 3
    holes = [x for x in cr["cavities"] if x["role"] == "hole"]
    assert len(holes) == 4                                # the four insert holes are holes, not storage


# ---------------------------------------------------------------- what is packed where
def test_scope_classification():
    k = A.classify_component
    assert k("Hover propeller (12 in)") == "external" and k("Pusher propeller (11 in)") == "external"
    assert k("Silicone power wire (12 AWG)") == "wire" and k("Servo cable") == "wire"
    assert k("Carbon boom tube (16 x 14 mm)") == "external" and k("Carbon wing spar tube (20 x 18 mm)") == "external"
    assert k("Flight controller") == "internal" and k("Battery pack (4S 5200 mAh)") == "internal"
    assert k("Lead-acid battery") == "internal" and k("Wireless module") == "internal"


def test_each_part_only_packs_its_own_components():
    front = [{"name": "Flight controller", "dims": (53.3, 39, 16.2), "qty": 1}, {"name": "GPS", "dims": (34, 28, 11), "qty": 1},
             {"name": "Telemetry radio", "dims": (28, 53, 10.7), "qty": 1}, {"name": "RC receiver", "dims": (13, 11, 7), "qty": 1}]
    cav = _measure(110, 80, 66, 1.5, front_shell)["cavities"]
    assert A.check_requirement(cav, front, 1.0, 1.0)["fits"]
    battery = [{"name": "Battery pack", "dims": (134, 44, 32), "qty": 1}]
    cav = _measure(194, 62, 48, 1.42, cradle)["cavities"]
    assert A.check_requirement(cav, battery, 1.0, 1.0)["fits"]
    everything = front + battery + [{"name": "Hover motor", "dims": (41.8, 41.8, 26.5), "qty": 4}] * 6
    need = A.required_cavity(everything)["min_inner_size_mm"]
    assert need[2] > 300                                  # the old 448 mm "needs": every part was asked to hold it all


# ---------------------------------------------------------------- hardware: the cradle's four screws
def _cyl_feat(i, x, y, z0, z1, d):
    return {"id": f"H{i}", "kind": "hole", "diameter": d, "axis": [0, 0, 1], "point": [x, y, z0], "center": [x, y, (z0 + z1) / 2],
            "length": z1 - z0, "ends": [[x, y, z0], [x, y, z1]], "host_plane": None}


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_cradle_screws_are_planned_without_any_mate_fasteners():
    shell = {"bbox_mm": {"min": [-145, -40, -33], "max": [145, 40, 33]}, "planes": [], "cylinders":
             [_cyl_feat(i + 1, x, y, -33, -29, 3.4) for i, (x, y) in enumerate([(-132, -23), (-132, 23), (46, -23), (46, 23)])]}
    cr = {"bbox_mm": {"min": [-89, -31, -24], "max": [89, 31, 24]}, "planes": [], "cylinders":
          [_cyl_feat(i + 1, x, y, 16, 24, 4.0) for i, (x, y) in enumerate([(-89, -23), (-89, 23), (89, -23), (89, 23)])]}
    T = np.eye(4)
    T[:3, :3] = _rz(180)
    T[:3, 3] = [-43, 0, -57]
    hw, notes = A.audit_joints({"shell": shell, "cradle": cr}, {"shell": np.eye(4), "cradle": T}, [], [],
                               {"shell": "Fuselage rear shell", "cradle": "Battery cradle"})
    assert len(hw) == 4, notes
    assert all(h["size"] == "M3" and h["insert"] and h["tapped_part"] == "b" and h["auto"] for h in hw)
    assert all(h["length_mm"] == 10.0 for h in hw), [h["length_mm"] for h in hw]   # 4 mm floor + 5.7 mm insert
    assert all(h["nut"] is None for h in hw)
    bom = A.bill_of_materials([{"id": "shell", "name": "Shell"}, {"id": "cradle", "name": "Cradle"}], hw)
    names = {r["name"]: r["qty"] for r in bom}
    assert names["Socket head cap screw ISO 4762 M3 x 10"] == 4 and names["Heat-set threaded insert (brass) M3"] == 4


def test_two_clearance_holes_get_bolt_and_nut():
    a = {"bbox_mm": {"min": [0, 0, 0], "max": [10, 10, 5]}, "planes": [],
         "cylinders": [_cyl_feat(1, 5, 5, 0, 5, 3.4)]}
    b = {"bbox_mm": {"min": [0, 0, 5], "max": [10, 10, 10]}, "planes": [],
         "cylinders": [_cyl_feat(1, 5, 5, 5, 10, 3.4)]}
    hw, _ = A.audit_joints({"a": a, "b": b}, {"a": np.eye(4), "b": np.eye(4)}, [], [])
    assert len(hw) == 1 and hw[0]["nut"] and not hw[0]["insert"]
    assert hw[0]["length_mm"] == 16.0                    # 10 mm stack + nut + thread


def test_washer_bom_does_not_crash():
    hw = [{"joint": "m1", "size": "M3", "length_mm": 12.0, "nut": {"h": 2.4}, "washers": {"origins": [[0, 0, 0], [0, 0, 1]]}}]
    rows = A.bill_of_materials([{"id": "a", "name": "A"}], hw)
    assert {r["name"]: r["qty"] for r in rows}["Plain washer ISO 7089 M3"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
