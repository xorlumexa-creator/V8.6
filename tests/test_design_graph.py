"""Tests for design_graph.py (interfaces from the assembly, sync status, Blender script generators) and capacity.py (memory profiles),
and for the skeleton-skin loop end to end on synthetic meshes. Run: python tests/test_design_graph.py"""
import ast
import os
import sys
import time
import tracemalloc
from unittest import mock

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import capacity as CAP  # noqa: E402
import design_graph as DG  # noqa: E402
import skin_check as S  # noqa: E402
from test_skin_check import box, shell, uv_sphere  # noqa: E402


def _result(dx=0.0):
    M = np.eye(4)
    M[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                    # part turned 90 degrees about Z: local +x -> world +y
    M[:3, 3] = [100.0 + dx, 50.0, 10.0]
    return {
        "parts": [{"id": "s", "name": "Fuselage", "matrix": M.ravel().tolist(), "internal_space": {"cavities": [{"id": "C1", "inscribed": {"size": [100, 60, 40]}}]}}],
        "packing": {"s": {"packing": {"placements": [
            {"name": "Flight controller", "index": 0, "min": [0, 0, 0], "size": [50, 40, 10]},
            {"name": "Battery pack", "index": 0, "min": [0, 0, 20], "size": [100, 40, 30]}]}}},
        "hardware": [{"id": "auto1-b1", "joint": "auto1", "size": "M3", "length_mm": 16, "origin": [10, 20, 30], "axis": [0, 0, -1]}],
        "balance": {"lift_plan": {"prop_diameter_mm": 300}},
        "balance_inputs": {"components": [{"part": "s", "name": "Hover motor (MN3508)", "mass_each_g": 82, "center_local": [30, 80, 15]}],
                           "inventory": [{"name": "Hover motor (MN3508)", "dims": [42, 42, 30], "qty": 4}]},
    }


def test_interfaces_are_read_from_the_assembly_in_world_coordinates():
    ifs = {i["id"]: i for i in DG.interfaces_from_assembly(_result(), clearance_mm=2.5)}
    fc = ifs["flight-controller-1"]
    assert fc["mode"] == "inside" and fc["clearance_mm"] == 2.5 and fc["kind"] == "component" and fc["preserve"]
    # local box x 0..50, y 0..40, z 0..10 -> turned 90 deg about Z (x->y, y->-x) and moved by (100, 50, 10)
    assert np.allclose(fc["box"]["min"], [100 - 40, 50, 10]) and np.allclose(fc["box"]["max"], [100, 50 + 50, 20])
    motor = ifs["lift-motor-1"]
    pos = np.array([100 - 80, 50 + 30, 10 + 15])                          # (30, 80, 15) -> (-80, 30, 15) + (100, 50, 10)
    assert motor["mode"] == "outside" and np.allclose((np.array(motor["box"]["min"]) + np.array(motor["box"]["max"])) / 2, pos)
    assert np.allclose(np.array(motor["box"]["max"]) - np.array(motor["box"]["min"]), [42, 42, 30])
    acc = motor["access"]
    assert acc["axis"] == [0.0, 0.0, 1.0] and abs(acc["radius_mm"] - 135.0) < 0.1 and np.allclose(acc["origin"], pos + [0, 0, 15])   # prop disc 0.9 * 300 / 2
    bolt = ifs["fastener-auto1-b1"]
    assert bolt["access"]["axis"] == [-0.0, -0.0, 1.0] or bolt["access"]["axis"] == [0.0, 0.0, 1.0]       # the tool comes from the side opposite the tip
    assert bolt["access"]["origin"] == [10, 20, 30] and "box" not in bolt


def test_geometry_hash_follows_the_skeleton_only():
    a, b = _result(), _result()
    assert DG.geometry_hash(a) == DG.geometry_hash(b)
    assert DG.geometry_hash(a) != DG.geometry_hash(_result(dx=5.0))
    c = _result()
    c["warnings"] = ["irrelevant"]
    c["parts"][0]["name"] = "Renamed"
    assert DG.geometry_hash(c) == DG.geometry_hash(a)                       # only placements, cavities and hardware count


def test_sync_status_walks_through_the_life_of_a_skin():
    res = _result()
    ifs = DG.interfaces_from_assembly(res)
    g = DG.build_graph(res, ifs)
    h = g["skeleton"]["geometry_hash"]
    assert DG.sync_status(g, h) == "no_skin" and "never converted" in g["rule"]
    report = {"verdict": "pass", "quality": {"triangles": 12}}
    g = DG.build_graph(res, ifs, DG.skin_record(b"mesh", report, h))
    assert DG.sync_status(g, h) == "in_sync"
    assert DG.sync_status(g, DG.geometry_hash(_result(dx=5.0))) == "skin_stale"          # the skeleton moved after the skin was validated
    g["skin"]["verdict"] = "fail"
    assert DG.sync_status(g, h) == "skin_fails"


def test_a_skin_built_around_the_interfaces_passes_and_a_tight_one_fails():
    res = _result()
    ifs = DG.interfaces_from_assembly(res, clearance_mm=2.0)
    inside = [i for i in ifs if i["mode"] == "inside"]
    lo = np.min([i["box"]["min"] for i in inside], axis=0)
    hi = np.max([i["box"]["max"] for i in inside], axis=0)
    good = np.concatenate([box(lo - 12, hi + 12), box(lo - 9, hi + 9, inward=True)])         # 3 mm wall, 7 mm clear of the components
    keep = [i for i in ifs if i["kind"] != "motor" and i["kind"] != "fastener"]
    assert S.check(good, keep, {"min_wall_mm": 2.0})["verdict"] == "pass"
    tight = np.concatenate([box(lo - 4, hi + 4), box(lo - 1, hi + 1, inward=True)])           # inner wall 1 mm from the components
    r = S.check(tight, keep, {"min_wall_mm": 2.0})
    assert r["verdict"] == "fail" and any("clearance" in c["name"] or "enclosed" in c["name"] for c in r["checks"] if c["ok"] is False)
    motor = [i for i in ifs if i["kind"] == "motor"][0]
    assert S.check(good, [motor])["verdict"] == "pass"                                               # the motor is outside this skin, as it should be
    huge = np.concatenate([box(lo - 200, hi + 200), box(lo - 197, hi + 197, inward=True)])           # a skin that swallows the motor and its props
    swallowed = S.check(huge, [motor])
    assert swallowed["verdict"] == "fail" and any("stays outside" in c["name"] for c in swallowed["checks"] if c["ok"] is False)


def test_blender_skin_script_is_valid_and_carries_the_interfaces():
    ifs = DG.interfaces_from_assembly(_result())
    sc = DG.blender_skin_script(ifs, wall_mm=2.5, margin_mm=4, bevel_mm=5, subsurf=1, max_tris=30000)
    ast.parse(sc)
    assert "np.float64" not in sc and "MAX_TRIS = " in sc and "30000" in sc and "DECIMATE" in sc
    ns_vars = {}
    tree = ast.parse(sc)
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("INSIDE", "CUTS"):
            ns_vars[n.targets[0].id] = ast.literal_eval(n.value)
    assert len(ns_vars["INSIDE"]) == 2 and len(ns_vars["CUTS"]) == 2                        # two enclosed components; motor prop disc + screw tool
    fake = {k: mock.MagicMock() for k in ("bpy", "bmesh", "mathutils")}
    with mock.patch.dict(sys.modules, fake):
        exec(compile(sc, "<skin>", "exec"), {"__name__": "x"})                              # runs through against a stand-in bpy
    assert fake["bpy"].ops.wm.read_factory_settings.called and fake["bpy"].ops.mesh.primitive_cylinder_add.call_count == 2
    try:
        DG.blender_skin_script([i for i in ifs if i["mode"] != "inside"])
    except ValueError:
        pass
    else:
        raise AssertionError("built a skin around nothing")
    ast.parse(DG.blender_render_script([{"file": "a.stl", "color": [1, 0, 0], "name": "arm"}], "skin.stl", 32))


def test_capacity_profiles():
    with mock.patch.dict(os.environ):
        os.environ.pop("LUMEXA_PROFILE", None)                                 # these asserts are about the automatic choice, whatever the shell has set
        _capacity_profiles_body()


def _capacity_profiles_body():
    tablet = "MemTotal:        3800000 kB\nMemAvailable:    1200000 kB\n"
    assert CAP.profile(CAP.mem_available_mb(tablet), CAP.mem_total_mb(tablet))["name"] == "tight"
    free_tablet = "MemTotal:        3800000 kB\nMemAvailable:    3000000 kB\n"
    assert CAP.profile(CAP.mem_available_mb(free_tablet), CAP.mem_total_mb(free_tablet))["name"] == "normal"          # a 4 GB tablet is never 'roomy'
    assert CAP.profile(9000, 16000)["name"] == "roomy" and CAP.profile(1000, 16000)["name"] == "tight"
    assert CAP.profile(None, None)["name"] == "tight"                                                                  # unreadable -> assume the small device
    old = "MemTotal: 2000000 kB\nMemFree: 500000 kB\nCached: 300000 kB\n"
    assert abs(CAP.mem_available_mb(old) - 800000 / 1024) < 1
    with mock.patch.dict(os.environ, {"LUMEXA_PROFILE": "roomy"}):
        assert CAP.profile(100, 3800)["name"] == "roomy"
    t, n, r = CAP.profile(1000, 3800), CAP.profile(3000, 3800), CAP.profile(9000, 16000)
    assert t["chunk_entries"] < n["chunk_entries"] < r["chunk_entries"] and t["max_skin_triangles"] < r["max_skin_triangles"]
    assert t["stl_tol"][0] > r["stl_tol"][0]                                                                           # coarser tessellation when tight
    assert CAP.stl_triangles_from_size(84 + 50 * 1234) == 1234


def test_big_skins_are_refused_with_advice_instead_of_crashing_the_server():
    old = CAP._PROFILES["tight"]["max_skin_triangles"]
    CAP._PROFILES["tight"]["max_skin_triangles"] = 10
    try:
        with mock.patch.dict(os.environ, {"LUMEXA_PROFILE": "tight"}):
            try:
                S.check(shell(), [])
            except S.TooLarge as e:
                assert "Decimate" in str(e) and "ratio" in str(e) and e.limit == 10
            else:
                raise AssertionError("a mesh over the limit was accepted")
    finally:
        CAP._PROFILES["tight"]["max_skin_triangles"] = old


def test_memory_stays_small_on_the_tight_profile():
    sp = uv_sphere(60, 200, 130)                                                             # ~52k triangles
    ifs = [{"id": "pcb", "kind": "component", "mode": "inside", "box": {"min": [-20, -20, -5], "max": [20, 20, 5]}, "clearance_mm": 2.0},
           {"id": "tool", "access": {"origin": [0, 0, 0], "axis": [1, 0, 0], "length_mm": 80, "radius_mm": 3}}]
    with mock.patch.dict(os.environ, {"LUMEXA_PROFILE": "tight"}):
        tracemalloc.start()
        t0 = time.time()
        S.check(sp, ifs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    print(f"     (52k-triangle skin on the tight profile: peak {peak / 1e6:.0f} MB, {time.time() - t0:.1f} s here)")
    assert peak < 100e6, f"peak {peak / 1e6:.0f} MB is too much for a 4 GB tablet running CalculiX next to it"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
