"""Tests for cog_balance.py (pure numpy). Run: python tests/test_cog_balance.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cog_balance as B  # noqa: E402

I4 = np.eye(4)


def test_mass_parsing():
    assert B.parse_mass_g("39.2 g") == 39.2
    assert B.parse_mass_g("82 g (104 g with cable)") == 82
    assert B.parse_mass_g("1.2 kg pack") == 1200
    assert B.parse_mass_g("5200 mAh, 4S, 134 x 44 x 32 mm") is None            # mAh and mm are not masses
    assert B.parse_mass_g("28 mm x 53 mm, 23.5 g with antenna") == 23.5
    assert B.parse_mass_g("12 AWG") is None


def test_materials():
    assert B.material_density("PETG")[:2] == (1.27, 0.35)
    assert B.material_density("Carbon fibre tube")[:2] == (1.60, 1.0)
    assert B.material_density("Aluminium 6061-T6")[0] == 2.70
    assert B.material_density("unobtainium")[2] is False
    g, _ = B.part_mass_g(100_000, "PETG")            # 100 cm3 of print
    assert abs(g - 100 * 1.27 * 0.35) < 1e-6
    g, _ = B.part_mass_g(100_000, "PETG", print_fill=0.2)
    assert abs(g - 100 * 1.27 * 0.2) < 1e-6


def test_roles_and_vehicle():
    assert B.role_of("Hover motor (MN3508)") == "lift_motor" and B.role_of("Pusher motor") == "forward_motor"
    assert B.role_of("Hover propeller (12 in)") == "lift_prop" and B.role_of("Pusher propeller (11 in)") == "forward_prop"
    assert B.role_of("Battery pack (4S)") == "battery" and B.role_of("ESC 40A") == "esc" and B.role_of("Servo") is None
    assert B.vehicle_kind("VTOL powerline perching drone") == "vtol" and B.vehicle_kind("RC car") == "generic"
    assert B.vehicle_kind("delivery drone") == "multirotor"


def _frame(battery_x, battery_travel=None):
    parts = [{"id": "p", "name": "Frame", "matrix": I4, "mass_g": 400.0, "com_local": [0, 0, 0]}]
    comps = [{"part": "p", "name": "Battery pack", "mass_each_g": 400.0, "center_local": [battery_x, 0, 0], "travel_world": battery_travel}]
    inv = [{"name": "Battery pack", "qty": 1, "dims_text": "400 g"},
           {"name": "Hover motor", "qty": 4, "dims_text": "82 g"},
           {"name": "Hover propeller (12 in)", "qty": 4, "dims_text": "304.8 mm diameter"},
           {"name": "ESC 40A", "qty": 4, "dims_text": "39 g"}]
    return {"parts": parts, "components": comps, "inventory": inv, "vehicle": "multirotor", "frame_center": [0, 0, 0]}


def test_cog_and_unplaced_mass():
    a = B.analyze(_frame(-30))
    assert abs(a["cog_mm"][0] - (-15.0)) < 1e-6                      # 400 g at 0 and 400 g at -30 -> -15
    assert a["mass"]["counted_g"] == 800.0
    names = {r["name"] for r in a["not_placed"]}
    assert names == {"Hover motor", "Hover propeller (12 in)", "ESC 40A"}
    assert a["mass"]["not_placed_g"] > 0 and "inertia_kgm2" in a


def test_lift_plan_centres_motors_on_the_cog():
    a = B.analyze(_frame(-30))
    lp = a["lift_plan"]
    assert lp["status"] == "planned" and lp["n"] == 4
    assert lp["centre_xy_mm"] == [round(a["cog_mm"][0], 2), round(a["cog_mm"][1], 2)]
    xs = np.array([[m["x"], m["y"]] for m in lp["motors"]])
    assert np.allclose(xs.mean(axis=0), lp["centre_xy_mm"], atol=0.02)         # thrust centroid == CoG
    assert all(abs(s - 25.0) < 0.2 for s in lp["thrust_share_pct"])
    d = [math.dist(xs[i], xs[(i + 1) % 4]) for i in range(4)]
    assert min(d) >= 304.8 * 1.10 - 0.5                                       # props clear each other
    assert lp["yaw_balanced"] and sorted(m["spin"] for m in lp["motors"]) == [-1, -1, 1, 1]
    assert lp["unit_mass_g"] > 82                                             # motor + prop + ESC


def test_thrust_shares_react_to_cog_offset():
    lay = B.motor_layout(4, 254)["positions"]
    xy = np.array([[p["x"], p["y"]] for p in lay])
    even = B.thrust_shares(xy, [0, 0])
    assert np.allclose(even, 0.25)
    off = B.thrust_shares(xy, [20, 0])                                        # CoG moved toward +x motors
    assert abs(off.sum() - 1) < 1e-9 and off.max() > 0.25 and off.min() < 0.25
    toward = xy[:, 0] > 0
    assert off[toward].mean() > off[~toward].mean()
    assert abs((off[:, None] * xy).sum(axis=0)[0] - 20) < 1e-9                # moments about the CoG cancel


def test_battery_shift_corrects_the_offset():
    a = B.analyze(_frame(-30, battery_travel={"min": [-60, -20, 0], "max": [60, 20, 0]}))
    bs = a["battery_shift"]
    assert bs["battery"] == "Battery pack"
    # CoG is -15 mm from the frame centre with 800 g counted: sliding the 400 g battery +30 mm moves it to 0
    assert abs(bs["needed_shift_mm"][0] - 30.0) < 0.1
    assert bs["fully_corrects"] and abs(bs["residual_cog_offset_mm"][0]) < 0.2
    tight = B.analyze(_frame(-30, battery_travel={"min": [-5, -5, 0], "max": [5, 5, 0]}))["battery_shift"]
    assert not tight["fully_corrects"] and abs(tight["achievable_shift_mm"][0] - 5) < 0.1
    assert abs(tight["residual_cog_offset_mm"][0]) > 5


def test_placed_motors_are_checked_not_replanned():
    parts = [{"id": "p", "name": "Frame", "matrix": I4, "mass_g": 300.0, "com_local": [0, 0, 0]}]
    lay = B.motor_layout(4, 254)["positions"]
    comps = [{"part": "p", "name": "Hover motor", "mass_each_g": 80.0, "center_local": [q["x"] + 15, q["y"], 0]} for q in lay]
    a = B.analyze({"parts": parts, "components": comps, "inventory": [{"name": "Hover motor", "qty": 4, "dims_text": "80 g"}], "vehicle": "multirotor"})
    lp = a["lift_plan"]
    assert lp["status"] == "placed" and not lp["balanced"]
    # frame (300 g at 0) + motors (320 g centred at +15): CoG = 320*15/620 = 7.74, the motors' centroid is 15 -> 7.26 mm apart
    assert abs(lp["offset_from_cog_mm"][0] - (15 - 320 * 15 / 620)) < 0.05
    shares = lp["thrust_share_pct"]
    r = B.motor_layout(4, 254)["radius_mm"] / math.sqrt(2)                     # |x| of each motor from the centroid
    expected_spread = 2 * 7.26 / (4 * r) * 100                                 # shares differ by 2*offset/(4*|x|)
    assert abs((max(shares) - min(shares)) - expected_spread) < 0.15, (shares, expected_spread)


def test_brief_text_has_numbers():
    t = B.brief_text(B.analyze(_frame(-30)))
    assert "CoG at x=-15.0" in t and "4 lift motors" in t and "radius" in t


def test_cog_uses_each_parts_transform_not_its_local_centre():
    """A part turned 90 degrees about Z and moved: its mass and its components must land where the matrix puts them."""
    R = np.eye(4)
    R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]              # local +x -> world +y
    R[:3, 3] = [100.0, 50.0, 10.0]
    parts = [{"id": "a", "name": "Turned part", "matrix": R, "mass_g": 300.0, "com_local": [20.0, 0.0, 0.0]},
             {"id": "b", "name": "Origin part", "matrix": I4, "mass_g": 100.0, "com_local": [0.0, 0.0, 0.0]}]
    comps = [{"part": "a", "name": "Battery pack", "mass_each_g": 100.0, "center_local": [0.0, 30.0, 5.0]}]    # local +y -> world -x
    a = B.analyze({"parts": parts, "components": comps, "inventory": [], "vehicle": "generic"})
    pos_a = np.array([100.0, 50.0 + 20.0, 10.0])                # (20,0,0) -> (0,20,0) then + (100,50,10)
    pos_c = np.array([100.0 - 30.0, 50.0, 10.0 + 5.0])           # (0,30,5) -> (-30,0,5) then + (100,50,10)
    want = (300 * pos_a + 100 * pos_c + 100 * np.zeros(3)) / 500
    assert np.allclose(a["cog_mm"], want, atol=0.01), (a["cog_mm"], want)
    unturned = B.analyze({"parts": [dict(parts[0], matrix=I4), parts[1]], "components": [], "inventory": [], "vehicle": "generic"})
    assert not np.allclose(unturned["cog_mm"], B.analyze({"parts": parts[:1] + parts[1:], "components": [], "inventory": [], "vehicle": "generic"})["cog_mm"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
