"""Tests for flight_check.py and drone_frame.py, and for how an arm frame's motor mounts feed the balance. Run: python tests/test_flight_frame.py"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import cog_balance as B  # noqa: E402
import drone_frame as D  # noqa: E402
import flight_check as F  # noqa: E402
from test_balance_glue import _lift  # noqa: E402
from test_designs import _run  # noqa: E402


def test_parsers():
    assert F.parse_thrust_g("max thrust 2.1 kg") == 2100 and F.parse_thrust_g("1500 g thrust") == 1500 and F.parse_thrust_g("82 g") is None
    assert F.parse_kv("Hover motor (MN3508 KV700)") == 700 and F.parse_kv("700KV") == 700 and F.parse_kv("brushless") is None
    assert F.parse_cells("Battery pack (4S 5200 mAh)") == 4 and F.parse_cells("5200 mAh") is None
    assert F.parse_tube("16 mm outer diameter × 14 mm inner diameter") == (16.0, 14.0) and F.parse_tube("tube 20 x 18 mm") == (20.0, 18.0)
    assert F.parse_tube("14 x 16 mm") is None                                  # inner larger than outer is not a tube


def test_thrust_and_hover_throttle():
    r = F.check({"mass_g": 2000, "n_motors": 4, "motor_max_thrust_g": 1500})
    assert abs(r["thrust_to_weight"] - 3.0) < 0.01
    assert abs(r["hover_throttle_pct"] - round(math.sqrt(1 / 3.0) * 100)) < 1        # thrust ~ throttle^2
    assert abs(r["attitude_headroom_pct"] - round(2 / 3 * 100)) < 1
    weak = F.check({"mass_g": 2000, "n_motors": 4, "motor_max_thrust_g": 600})
    assert [c for c in weak["checks"] if c["name"] == "Thrust-to-weight"][0]["ok"] is False and weak["verdict"] == "fail"
    unknown = F.check({"mass_g": 2000, "n_motors": 4})
    assert unknown["verdict"] == "incomplete" and "thrust_to_weight" not in unknown


def test_yaw_check_and_info_checks_do_not_decide_the_verdict():
    ok = F.check({"mass_g": 2000, "n_motors": 4, "motor_max_thrust_g": 1500, "spins": [1, -1, 1, -1]})
    assert [c for c in ok["checks"] if c["name"] == "Yaw torque"][0]["ok"] is True
    bad = F.check({"mass_g": 2000, "n_motors": 3, "motor_max_thrust_g": 2000, "spins": [1, -1, 1]})
    assert [c for c in bad["checks"] if c["name"] == "Yaw torque"][0]["ok"] is False
    quad = F.check({"mass_g": 2000, "n_motors": 4, "motor_max_thrust_g": 1500, "spins": [1, -1, 1, -1]})
    out = [c for c in quad["checks"] if c["name"] == "One motor out"][0]
    assert out["info"] and quad["verdict"] == "pass"                            # advice about losing a motor does not fail a quad


def test_arm_beam_numbers():
    a = F.arm_check(200, 16, 14, "carbon", 150, 5.0, 14.7, rpm_hover=5000, rpm_max=8800)
    I = math.pi / 64 * (16 ** 4 - 14 ** 4)                                       # mm^4
    assert abs(a["root_stress_mpa"] - 1.5 * 14.7 * 200 * 8 / I) < 0.2           # M c / I with the 1.5 gust factor
    assert abs(a["safety_factor"] - 250 / a["root_stress_mpa"]) < 0.02
    fn = math.sqrt(3 * 70e9 * (I * 1e-12) / (0.15 * 0.2 ** 3)) / (2 * math.pi)
    assert abs(a["first_bending_hz"] - fn) < 0.2
    assert a["tip_deflection_mm_at_hover"] > 0
    stiff = F.arm_check(200, 20, 18, "carbon", 150, 5.0, 14.7)
    assert stiff["first_bending_hz"] > a["first_bending_hz"] and stiff["root_stress_mpa"] < a["root_stress_mpa"]
    printed = F.arm_check(200, 16, 0.0001, "petg", 150, 5.0, 14.7)
    assert printed["safety_factor"] < a["safety_factor"]


def test_resonance_window():
    hover_hz, max_hz = 85.0, 147.0                                               # 5100 and 8800 rpm
    inside = F.arm_check(200, 16, 14, "carbon", 150, 5, 14.7, rpm_hover=hover_hz * 60, rpm_max=max_hz * 60)
    assert "once per rev" in inside["resonance_with"]
    far = F.arm_check(120, 20, 18, "carbon", 150, 5, 14.7, rpm_hover=hover_hz * 60, rpm_max=max_hz * 60)
    assert far["first_bending_hz"] > 300 and far["resonance_with"] == [] or "blade pass" in far["resonance_with"]


def test_full_check_reads_motor_battery_and_tube_text():
    r = F.check({"mass_g": 2000, "n_motors": 4, "motor_text": "MN3508 KV700, 1500 g thrust", "battery_text": "4S 5200 mAh",
                 "tube_text": "16 mm outer diameter x 14 mm inner diameter", "arm_length_mm": 200, "tip_mass_g": 150, "spins": [1, -1, 1, -1]})
    assert r["inputs_used"] == {"motor_max_thrust_g": 1500.0, "kv": 700, "cells": 4}
    assert r["rpm_estimate"]["max"] == int(700 * 4 * 3.7 * 0.85) and "arm" in r and "resonance_with" in r["arm"]
    assert {c["name"] for c in r["checks"]} >= {"Thrust-to-weight", "Yaw torque", "Arm bending", "Arm vibration"}


# ------------------------------------------------------------------ arm frame
def _plan(cog=(-30.0, 2.0), n=4, d=304.8):
    lay = B.motor_layout(n, d)
    return [{"x": cog[0] + p["x"], "y": cog[1] + p["y"], "spin": p["spin"]} for p in lay["positions"]], cog


def test_arm_frame_sits_at_the_planned_positions():
    motors, cog = _plan()
    r = D.arm_frame(motors, cog)
    ns, _ = _run(r["script"])
    mm = np.array(ns["MOTOR_MOUNTS"])
    assert np.allclose(mm[:, :2].mean(axis=0), 0, atol=0.01)                      # part origin = centre of the motor circle
    assert np.allclose(mm[:, :2] + np.array(cog), [[m["x"], m["y"]] for m in motors], atol=0.01)     # = the planned assembly positions
    assert ns["MOTOR_SPIN"] == [1, -1, 1, -1] and r["spins"] == ns["MOTOR_SPIN"]
    assert all(abs(m[2] - (1.5 + 12.5)) < 1e-6 for m in ns["MOTOR_MOUNTS"])       # plate half-thickness + half the motor height
    assert abs(r["arm_length_mm"] - 237.1) < 0.2


def test_arm_frame_rejects_a_hub_that_collides_with_the_motor_plates():
    motors, cog = _plan(d=60.0)                                                   # tiny props -> motors only ~46 mm from the centre
    try:
        D.arm_frame(motors, cog)
    except ValueError as e:
        assert "too close" in str(e)
    else:
        raise AssertionError("accepted a layout whose plates hit the hub")
    try:
        D.arm_frame(motors[:2], cog)
    except ValueError:
        return
    raise AssertionError("accepted two motors")


def test_mounts_are_read_from_the_script_without_running_it():
    f = _lift("_mounts_from_script")
    motors, cog = _plan()
    r = D.arm_frame(motors, cog)
    got = f(r["script"])
    assert np.allclose(np.array(got["MOTOR_MOUNTS"]), np.array(r["mounts"])) and got["MOTOR_SPIN"] == r["spins"]
    assert f("result = Box(1, 1, 1)") is None and f("def broken(:") is None
    assert f("MOTOR_MOUNTS = __import__('os').getcwd()") is None                  # not a literal: ignored, never evaluated


# ------------------------------------------------------------------ the loop: planned layout -> frame -> balance check on geometry
def _glue_case(frame_dx=0.0, battery_x=-30.0):
    motors, cog = _plan(cog=(0.0, 0.0))
    r = D.arm_frame(motors, cog)
    M = np.eye(4)
    M[0, 3] = frame_dx
    infos = [{"id": "frame", "name": "Arm frame", "mass": {"volume_mm3": 200_000, "com": [0, 0, 0]}, "features": {}, "mounts": _lift("_mounts_from_script")(r["script"])},
             {"id": "body", "name": "Fuselage", "mass": {"volume_mm3": 300_000, "com": [0, 0, 0]}, "features": {}}]
    res = {"parts": [{"id": "frame", "matrix": M.tolist(), "internal_space": {}}, {"id": "body", "matrix": np.eye(4).tolist(), "internal_space": {}}], "packing": {}}
    bopt = {"vehicle": "multirotor", "materials": {"frame": "PETG", "body": "PETG"}, "inventory": [
        {"name": "Hover motor (MN3508 KV700)", "dims_text": "82 g", "qty": 4}, {"name": "Hover propeller (12 in)", "dims_text": "304.8 mm diameter", "qty": 4},
        {"name": "ESC 40A", "dims_text": "39 g", "qty": 4}]}
    inp = _lift("_balance_inputs")(res, infos, bopt)
    return inp, B.analyze(inp)


def test_motor_mounts_become_placed_motors_props_and_escs():
    inp, a = _glue_case()
    names = [c["name"] for c in inp["components"]]
    assert names.count("Hover motor (MN3508 KV700)") == 4 and names.count("Hover propeller (12 in)") == 4 and names.count("ESC 40A") == 4
    assert a["not_placed"] == []                                                   # everything now has a place
    lp = a["lift_plan"]
    assert lp["status"] == "placed" and lp["n"] == 4
    assert lp["balanced"] and all(abs(s - 25.0) < 0.2 for s in lp["thrust_share_pct"])
    assert a["mass"]["counted_g"] > 4 * (82 + 39)                                  # frame + body + 4 units


def test_a_misplaced_frame_shows_up_as_an_offset_and_unequal_thrust():
    _, ok = _glue_case(frame_dx=0.0)
    _, off = _glue_case(frame_dx=40.0)                                             # frame bolted 40 mm further aft than planned
    lp = off["lift_plan"]
    assert lp["status"] == "placed" and not lp["balanced"]
    assert abs(lp["offset_from_cog_mm"][0]) > 5                                    # thrust centroid is no longer on the CoG
    assert max(lp["thrust_share_pct"]) - min(lp["thrust_share_pct"]) > 1.5        # so the motors must work unequally to hover
    assert ok["lift_plan"]["balanced"]
    # the motors are most of the mass, so the CoG follows the frame: 40 mm of misplacement leaves ~7 mm, 100 mm leaves ~17 mm
    _, far = _glue_case(frame_dx=100.0)
    assert 15 < far["lift_plan"]["offset_from_cog_mm"][0] < 19 and max(far["lift_plan"]["thrust_share_pct"]) - min(far["lift_plan"]["thrust_share_pct"]) > 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
