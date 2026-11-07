"""Tests for crank_design.py and aero_design.py (pure numpy). The generated build123d scripts are also run against a stand-in
build123d so typos, wrong variable names and bad loops show up; whether OpenCASCADE accepts the geometry can only be seen with the
real kernel. Run: python tests/test_designs.py"""
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aero_design as A  # noqa: E402
import crank_design as C  # noqa: E402


# ------------------------------------------------------------------ stand-in build123d (records what a script asks for)
class _S:
    def __add__(self, o): return _S()
    __radd__ = __add__
    def __sub__(self, o): return _S()
    def __and__(self, o): return _S()


class _T:
    def __mul__(self, o): return _S() if isinstance(o, _S) else _T()


class _Ctx:
    def __init__(self, *a, **k): self.part = _S()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _stub():
    m = types.ModuleType("build123d")
    calls = {"sections": 0, "polylines": [], "ellipses": [], "lofts": []}
    m.Pos = lambda *a: _T()
    m.Rot = lambda *a: _T()
    m.Box = lambda *a, **k: _S()
    m.Cylinder = lambda *a, **k: _S()
    m.BuildPart, m.BuildLine = _Ctx, _Ctx

    class BuildSketch(_Ctx):
        def __init__(self, plane): calls["sections"] += 1
    m.BuildSketch = BuildSketch

    class _Plane:
        def __init__(self, origin=None, x_dir=None, z_dir=None): assert len(origin) == 3 and len(z_dir) == 3
        def offset(self, d): return self
    class _PlaneNS:
        YZ = _Plane((0, 0, 0), (0, 1, 0), (1, 0, 0))
    m.Plane = type("Plane", (), {"YZ": _PlaneNS.YZ, "__new__": lambda cls, origin=None, x_dir=None, z_dir=None: _Plane(origin, x_dir, z_dir)})
    m.Ellipse = lambda a, b: calls["ellipses"].append((a, b))
    m.Polyline = lambda *pts, close=False: calls["polylines"].append((len(pts), close, all(len(p) == 2 for p in pts)))
    m.make_face = lambda: None
    m.loft = lambda ruled=False: calls["lofts"].append(ruled)
    return m, calls


def _run(script):
    m, calls = _stub()
    sys.modules["build123d"] = m
    ns = {}
    try:
        exec(compile(script, "<generated>", "exec"), ns)
    finally:
        del sys.modules["build123d"]
    assert "result" in ns, "the script must define `result`"
    return ns, calls


# ------------------------------------------------------------------ crankshaft
def test_inline4_is_balanced_and_counterweights_cancel_static_unbalance():
    d = C.design({"n_throws": 4})["data"]
    assert d["phases_deg"] == [0, 180, 180, 0]
    assert d["static_unbalance_per_throw_pct"] == 0.0 and not d["counterweight_capped"]
    assert d["net_force_unbalance_g_mm"] < 1 and d["net_couple_unbalance_g_mm2"] < 1     # flat-plane four: forces and couples cancel
    assert d["counterweight_radius_mm"] > 0.78 * 20                                   # bigger than the web disc


def test_counterweight_grows_with_stroke_and_pin_and_is_capped():
    a = C.design({"stroke_mm": 30})["data"]["counterweight_radius_mm"]
    b = C.design({"stroke_mm": 50})["data"]["counterweight_radius_mm"]
    assert b > a
    huge = C.design({"stroke_mm": 40, "pin_d": 60, "pin_len": 60, "balance_pct": 100})["data"]
    assert huge["counterweight_capped"] and huge["static_unbalance_per_throw_pct"] > 0


def test_three_throw_without_weights_has_a_couple_but_no_force():
    d = C.design({"n_throws": 3, "counterweights": False})["data"]
    assert d["net_force_unbalance_g_mm"] < 1 and d["net_couple_unbalance_g_mm2"] > 1e3
    assert d["counterweight_radius_mm"] is None and d["static_unbalance_per_throw_pct"] == 100.0


def test_crank_inputs_are_validated_and_length_adds_up():
    for bad in ({"n_throws": 0}, {"n_throws": 7}, {"stroke_mm": 0}, {"phases_deg": [0, 90]}):
        try:
            C.design(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    d = C.design({"n_throws": 2, "main_len": 14, "pin_len": 16, "web_t": 8})["data"]
    assert abs(d["length_mm"] - (3 * 14 + 2 * (2 * 8 + 16))) < 1e-6


def test_crank_and_block_scripts_run():
    out = C.design({"n_throws": 3})
    ns, _ = _run(out["script"])
    assert ns["PHASES"] == [0, 120, 240] and ns["RC"] > 0
    _run(out["block_script"])
    assert "block_script" not in C.design({"with_blocks": False})


# ------------------------------------------------------------------ engine side of the crankshaft (reciprocating mass)
def _eng(**kw):
    q = dict(n_throws=4, piston_mass_g=300.0, rod_mass_g=450.0, rod_length_mm=130.0, stroke_mm=40.0, rpm_reference=6000.0)
    q.update(kw)
    return C.design(q)["data"]


def test_engine_balance_matches_the_textbook_for_inline_four():
    d = _eng()
    e = d["engine"]
    m_recip = 300 + 450 * 0.30
    assert abs(e["reciprocating_mass_g"] - m_recip) < 1e-6
    assert e["primary_force_g_mm"] == 0 and e["primary_couple_g_mm2"] == 0                 # flat-plane four: primary balanced
    lam = 20.0 / 130.0
    assert abs(e["secondary_force_g_mm"] - 4 * m_recip * 20.0 * lam) < 0.05                # 4 m r lambda: needs balance shafts
    w2 = (2 * math.pi * 6000 / 60) ** 2
    assert abs(e["secondary_force_n"] - 4 * (m_recip / 1000) * 0.02 * lam * w2) < 0.5      # in newtons at 6000 rpm
    assert any("balance shafts" in n for n in e["notes"]) and any("balancing machine" in n for n in e["notes"])


def test_single_cylinder_trades_force_along_the_bore_for_force_across_it():
    e = _eng(n_throws=1)["engine"]
    assert abs(e["primary_force_g_mm"] - 435 * 20.0) < 0.05                                 # one cylinder: |sum of phase vectors| = 1
    assert e["primary_along_bore_fraction"] == 0.5 and e["primary_across_bore_fraction"] == 0.5
    assert _eng(n_throws=1, balance_factor=0.7)["engine"]["primary_along_bore_fraction"] == 0.3


def test_counterweights_grow_with_reciprocating_mass_and_balance_factor():
    def rc(**kw):
        d = C.design({"n_throws": 1, "web_t": 40, **kw})["data"]
        assert not d["counterweight_capped"]
        return d["counterweight_radius_mm"]
    base = rc()
    light = rc(piston_mass_g=60, rod_mass_g=60, rod_length_mm=120)
    heavy = rc(piston_mass_g=120, rod_mass_g=60, rod_length_mm=120)
    more = rc(piston_mass_g=60, rod_mass_g=60, rod_length_mm=120, balance_factor=0.8)
    assert base < light < heavy and light < more


def test_rod_big_end_share_follows_its_centre_of_gravity():
    near_big = _eng(n_throws=1, rod_cg_frac=0.2)["engine"]["reciprocating_mass_g"]
    near_small = _eng(n_throws=1, rod_cg_frac=0.4)["engine"]["reciprocating_mass_g"]
    assert abs(near_big - (300 + 450 * 0.2)) < 1e-6 and abs(near_small - (300 + 450 * 0.4)) < 1e-6


def test_capped_counterweight_reports_the_web_that_would_fix_it():
    d = _eng()
    assert d["counterweight_capped"] and d["static_unbalance_per_throw_pct"] > 50
    need = d["web_t_needed_mm"]
    assert need and need > 8
    fixed = _eng(web_t=need + 0.5)
    assert not fixed["counterweight_capped"] and fixed["static_unbalance_per_throw_pct"] < 1
    assert "engine" not in C.design({"n_throws": 4})["data"]                               # crank alone: no engine block


# ------------------------------------------------------------------ aerodynamics
def test_airfoil_geometry():
    xu, yu, xl, yl = A.naca4("0012", 81)
    assert abs(float((yu - yl).max()) - 0.12) < 0.002                                # 12 % thick
    assert np.allclose(yu, -yl, atol=1e-9)                                           # symmetric
    assert abs(float(yu[-1] - yl[-1])) < 1e-9                                        # closed trailing edge
    _, yc_u, _, yc_l = A.naca4("2412", 81)
    assert float(((yc_u + yc_l) / 2).max()) > 0.015                                  # cambered
    pts = A.airfoil_outline("2412", 200.0)
    assert min(p[0] for p in pts) < 0.5 and abs(max(p[0] for p in pts) - 200) < 1.0


def _wing(**kw):
    p = dict(mass_g=2000, speed_ms=18, span_mm=1500, root_chord_mm=220, tip_chord_mm=150, thickness_ratio=0.12,
             fuselage_length_mm=900, fuselage_dia_mm=120)
    p.update(kw)
    return A.estimate(p)


def test_estimate_physics_directions():
    base = _wing()
    assert abs(base["wing"]["S_m2"] - 1.5 * (0.22 + 0.15) / 2) < 1e-4
    weight = 2.0 * 9.80665
    assert abs(base["cl_required"] - weight / (0.5 * 1.225 * 18 ** 2 * base["wing"]["S_m2"])) < 0.001
    fast = _wing(speed_ms=30)
    assert fast["cl_required"] < base["cl_required"] and fast["cd_induced"] < base["cd_induced"] and fast["drag_n"] > 0
    assert _wing(mass_g=3000)["stall_speed_ms"] > base["stall_speed_ms"]                # heavier stalls faster
    assert _wing(span_mm=2000)["l_over_d"] > base["l_over_d"]                           # more aspect ratio, better glide
    assert _wing(extra_cda_m2=0.01)["drag_n"] > base["drag_n"]
    assert abs(base["l_over_d"] / (base["cl_required"] / base["cd_total"]) - 1) < 0.01          # outputs are rounded, so compare relatively


def test_estimate_warnings():
    assert any("CLmax" in w for w in _wing(speed_ms=8)["warnings"])
    assert any("Reynolds" in w for w in _wing(speed_ms=6, root_chord_mm=80, tip_chord_mm=60)["warnings"])
    assert any("fineness" in w for w in _wing(fuselage_length_mm=300, fuselage_dia_mm=120)["warnings"])


def test_streamlined_body_has_less_drag_than_a_bluff_one():
    slim, _ = A.fuselage_cda(900, 120, 18)          # fineness 7.5
    fat, _ = A.fuselage_cda(300, 120, 18)           # fineness 2.5
    assert slim / (900 * 120) < fat / (300 * 120) * 3     # per length x width, the short body is far worse
    assert A.cf_flat_plate(1e6) < A.cf_flat_plate(1e5)


def test_shape_scripts_run():
    ns, calls = _run(A.fuselage_script(900, 120, 100))
    secs = ns["SECTIONS"]
    assert len(secs) == 11 and secs[0][0] == -450 and secs[-1][0] == 450
    assert secs[0][1] < 5 and secs[-1][1] < 15 and max(r for _, r, _ in secs) == 60          # pointed nose, slim tail, max radius = D/2
    assert calls["sections"] == 11 and len(calls["ellipses"]) == 11
    ns, calls = _run(A.wing_script(1500, 220, 150, "2412", sweep_deg=5, dihedral_deg=2))
    assert calls["sections"] == 5 and calls["lofts"] == [True]
    assert all(n >= 60 and close and two_d for n, close, two_d in calls["polylines"])
    orgs = [o for _, o in ns["SECTIONS"]]
    assert [o[1] for o in orgs] == [-750, -375, 0, 375, 750]                                  # symmetric span stations
    assert orgs[0][2] == orgs[4][2] > orgs[2][2]                                              # dihedral lifts the tips


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
