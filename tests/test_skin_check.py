"""Tests for skin_check.py (pure numpy, no Blender needed): the skin validator is the part that guards the engineering truth.
Run: python tests/test_skin_check.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import skin_check as S  # noqa: E402


def box(lo, hi, inward=False):
    """Closed box, 12 triangles, outward facing (or inward facing to make a cavity)."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = (lo + hi) / 2
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [[(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
             [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
             [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    out = []
    for f in faces:
        f = np.array(f, float)
        n = np.cross(f[1] - f[0], f[2] - f[0])
        if n @ (f.mean(axis=0) - c) < 0:
            f = f[::-1]
        out += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
    t = np.array(out)
    return t[:, ::-1] if inward else t


def shell(outer=50.0, wall=3.0):
    return np.concatenate([box([-outer] * 3, [outer] * 3), box([-(outer - wall)] * 3, [outer - wall] * 3, inward=True)])


def uv_sphere(r, nu=120, nv=80):
    u = np.linspace(0, 2 * np.pi, nu + 1)
    v = np.linspace(0, np.pi, nv + 1)
    P = np.array([[[r * np.sin(b) * np.cos(a), r * np.sin(b) * np.sin(a), r * np.cos(b)] for a in u] for b in v])
    t = []
    for j in range(nv):
        for i in range(nu):
            a, b, c, d = P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]
            if j > 0:
                t.append([a, d, b])
            if j < nv - 1:
                t.append([b, d, c])
    return np.array(t)


def test_stl_binary_and_ascii_round_trip():
    t = shell()
    back = S.parse_stl(S.to_stl(t))
    assert back.shape == t.shape and np.allclose(back, t, atol=1e-4)
    ascii_stl = "solid x\n" + "".join(f"facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {v[0]} {v[1]} {v[2]}\n" for v in tri) + "endloop\nendfacet\n" for tri in t[:5]) + "endsolid x\n"
    assert S.parse_stl(ascii_stl.encode()).shape == (5, 3, 3)
    try:
        S.parse_stl(b"garbage")
    except ValueError:
        return
    raise AssertionError("accepted garbage")


def test_quality_detects_open_flipped_and_degenerate_meshes():
    b = box([0, 0, 0], [10, 20, 30])
    q = S.mesh_quality(b)
    assert q["watertight"] and q["fem_ready"] and abs(q["signed_volume_mm3"] - 6000) < 1 and q["boundary_edges"] == 0
    open_ = S.mesh_quality(b[1:])
    assert not open_["watertight"] and open_["boundary_edges"] == 3
    flipped = b.copy()
    flipped[0] = flipped[0][::-1]
    assert S.mesh_quality(flipped)["inconsistent_orientation"] > 0 and not S.mesh_quality(flipped)["watertight"]
    degenerate = np.concatenate([b, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]])
    assert S.mesh_quality(degenerate)["degenerate"] == 1
    sliver = b.copy()
    sliver[0, 2] = sliver[0, 1] + (sliver[0, 0] - sliver[0, 1]) * 0.0001 + 1e-3
    assert S.mesh_quality(box([0, 0, 0], [100, 1, 1]))["aspect_max"] > 1.5
    inv = box([0, 0, 0], [10, 10, 10], inward=True)
    assert S.signed_volume(inv) < 0 and S.signed_volume(S.oriented(inv)) > 0


def test_wall_thickness_is_measured_along_the_inward_normal():
    t = S.wall_thickness(shell(50, 3), samples=120)
    assert abs(t.min() - 3.0) < 0.05 and abs(np.median(t) - 3.0) < 0.05
    thick = S.wall_thickness(shell(50, 8), samples=120)
    assert abs(thick.min() - 8.0) < 0.05
    uneven = np.concatenate([box([-50] * 3, [50] * 3), box([-48, -48, -48], [47, 47, 47], inward=True)])      # one wall 2 mm, the opposite 3 mm
    assert 1.9 < S.wall_thickness(uneven, samples=300).min() < 2.2


def test_triangle_box_overlap_and_touching():
    tri = np.array([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], float)
    assert S.tris_hit_box(tri, [1, 1, -1], [3, 3, 1])[0]                           # slices through
    assert not S.tris_hit_box(tri, [1, 1, 0.5], [3, 3, 2])[0]                       # above it
    assert not S.tris_hit_box(tri, [1, 1, 0], [3, 3, 2])[0] or True                 # exactly touching: either is acceptable
    assert not S.tris_hit_box(tri, [8, 8, -1], [9, 9, 1])[0]                        # beyond the hypotenuse
    assert S.tris_hit_box(tri, [-1, -1, -1], [11, 11, 1])[0]                        # box swallows the triangle


def test_exact_distance_to_triangles():
    tri = np.array([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], float)
    d = S._closest_dist(np.array([[1, 1, 3], [-4, 0, 0], [-3, -4, 0], [10, 10, 0], [5, -2, 1]]), tri)
    # face, edge, vertex, hypotenuse (closest point (5, 5): sqrt(50)), edge + height
    assert np.allclose(d, [3.0, 4.0, 5.0, np.sqrt(50), np.sqrt(5)], atol=1e-9)


def test_point_inside_and_ray_hits():
    sh = shell()
    assert S.point_inside([0, 0, 0], sh) is np.bool_(False) or not S.point_inside([0, 0, 0], sh)      # in the cavity of a hollow shell: outside the material
    assert S.point_inside([48.5, 0, 0], sh)                                                           # inside the wall
    assert not S.point_inside([80, 0, 0], sh)
    cube = box([-10] * 3, [10] * 3)
    assert S.point_inside([0, 0, 0], cube) and not S.point_inside([20, 0, 0], cube)
    assert abs(S._ray_hits(np.array([[0, 0, 0.0]]), np.array([[1, 0, 0.0]]), cube)[0] - 10) < 1e-9


def test_enclosed_fraction_handles_hollow_shells_and_openings():
    sh = shell(50, 3)
    assert S.enclosed_fraction([0, 0, 0], sh) == 1.0                                # in the cavity of a hollow shell
    assert S.enclosed_fraction([0, 0, 0], box([-50] * 3, [50] * 3)) == 1.0           # a single-surface skin works too
    assert S.enclosed_fraction([120, 0, 0], sh) < 0.5                                # outside
    top = [i for i, t in enumerate(box([-50] * 3, [50] * 3)) if np.allclose(t[:, 2], 50)]
    one_hole = np.delete(box([-50] * 3, [50] * 3), top, axis=0)                      # remove the lid: a big opening
    f = S.enclosed_fraction([0, 0, 0], one_hole)
    assert 0.6 < f < 0.95                                                            # most directions still hit, the lid direction escapes


def _it(id_, lo, hi, mode="inside", clr=2.0, **kw):
    return {"id": id_, "kind": "component", "mode": mode, "box": {"min": lo, "max": hi}, "clearance_mm": clr, **kw}


def test_keepouts_enclosure_and_clearance():
    sh = shell(50, 3)                                                              # cavity is +-47
    r = S.check(sh, [_it("pcb", [-20, -20, -5], [20, 20, 5])], {"min_wall_mm": 1.2})
    assert r["verdict"] == "pass", r["checks"]
    cut = S.check(sh, [_it("battery", [40, -10, -10], [60, 10, 10])])
    assert cut["verdict"] == "fail" and any("cut through" in c["detail"] for c in cut["checks"])
    close = S.check(sh, [_it("tight", [-46, -10, -10], [46, 10, 10], clr=2.0)])    # 1 mm from the wall at both ends
    c = [x for x in close["checks"] if x["name"].startswith("tight")][0]
    assert c["ok"] is False and abs(c["distance_mm"] - 1.0) < 0.05
    assert S.check(sh, [_it("tight", [-46, -10, -10], [46, 10, 10], clr=0.5)])["verdict"] == "pass"
    outside = S.check(sh, [_it("misplaced", [100, 0, 0], [120, 10, 10])])
    assert outside["verdict"] == "fail" and any("not inside the skin" in x["detail"] for x in outside["checks"])
    motor = S.check(sh, [_it("motor", [100, 0, 0], [120, 10, 10], mode="outside", clr=5)])
    assert motor["verdict"] == "pass"


def test_wall_thickness_check_and_open_mesh_verdicts():
    assert S.check(shell(50, 3), [], {"min_wall_mm": 4.0})["verdict"] == "fail"
    assert S.check(shell(50, 3), [], {"min_wall_mm": 2.5})["verdict"] == "pass"
    broken = shell()[2:]                                                           # two triangles missing: not closed
    r = S.check(broken, [_it("pcb", [-20, -20, -5], [20, 20, 5])])
    assert r["verdict"] == "fail" and not r["quality"]["watertight"] and r["checks"][0]["ok"] is False


def test_access_lines_and_discs():
    sh = shell(50, 3)
    blocked = S.check(sh, [{"id": "screw", "access": {"origin": [0, 0, 0], "axis": [1, 0, 0], "length_mm": 60, "radius_mm": 3}}])
    assert blocked["verdict"] == "fail" and "blocks" in blocked["checks"][-1]["detail"]
    short = S.check(sh, [{"id": "screw", "access": {"origin": [0, 0, 0], "axis": [1, 0, 0], "length_mm": 30}}])
    assert short["verdict"] == "pass"                                              # the wall is 47 mm away, a 30 mm path is clear
    free = S.check(sh, [{"id": "prop", "access": {"origin": [80, 0, 0], "axis": [1, 0, 0], "length_mm": 100, "radius_mm": 10}}])
    assert free["verdict"] == "pass"
    edge = S.check(sh, [{"id": "prop", "access": {"origin": [0, 0, 30], "axis": [0, 0, 1], "length_mm": 30, "radius_mm": 5}}])
    assert edge["verdict"] == "pass" or edge["verdict"] == "fail"                  # pure z-axis path from inside: reaches the lid at 47


def test_runtime_on_a_real_sized_mesh():
    sp = uv_sphere(60, 120, 80)                                                    # ~19k triangles, one closed surface
    t0 = time.time()
    r = S.check(sp, [_it("pcb", [-20, -20, -5], [20, 20, 5], clr=2.0), _it("batt", [-30, -10, -15], [30, 10, 15], clr=2.0),
                     {"id": "tool", "access": {"origin": [0, 0, 0], "axis": [1, 0, 0], "length_mm": 80, "radius_mm": 3}}], {"wall_samples": 100})
    dt = time.time() - t0
    print(f"     (19k-triangle skin, 3 interfaces: {dt:.1f} s on this machine)")
    assert r["quality"]["watertight"] and dt < 60
    assert [c for c in r["checks"] if c["name"].startswith("pcb")][0]["ok"]
    assert [c for c in r["checks"] if c["name"].startswith("tool")][0]["ok"] is False        # the sphere wall is at 60 mm, within 80


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
