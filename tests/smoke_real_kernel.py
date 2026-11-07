"""Run this where the real build123d is installed (your Redmi's Termux, or a PC): it builds every generator in this project with the actual
OpenCASCADE kernel and checks the solids. The other tests use a stand-in for build123d, so only this one shows whether the generated
geometry is valid. It prints PASS / FAIL / WARN per check and exits 1 if anything failed.

    python tests/smoke_real_kernel.py            (from the backend folder)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import build123d as b3
except Exception as e:                                              # pragma: no cover
    print(f"SKIP: build123d is not installed here ({e}). Run this on the machine that runs the backend.")
    sys.exit(0)

import aero_design as AERO
import assembly as ASM
import cog_balance as BAL
import crank_design as CRANK
import drone_frame as FRAME

FAILS = []


def report(ok, name, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"{tag}  {name}  {detail}")


def warn(name, detail):
    print(f"WARN  {name}  {detail}")


def build(script):
    ns = {}
    exec(compile(script, "<generated>", "exec"), ns)
    return ns, ns["result"]


def valid(shape):
    v = getattr(shape, "is_valid", None)
    return bool(v() if callable(v) else v)


def bbox(shape):
    bb = shape.bounding_box()
    return (bb.size.X, bb.size.Y, bb.size.Z), (bb.min.X, bb.min.Y, bb.min.Z)


def check_solid(name, script, expect_size=None, tol=2.0):
    try:
        ns, shape = build(script)
    except Exception as e:
        report(False, f"{name}: script runs", f"{type(e).__name__}: {str(e)[:160]}")
        return None, None
    report(True, f"{name}: script runs")
    report(valid(shape), f"{name}: solid is valid")
    mp = ASM.OccOps.mass_props(shape)
    report(bool(mp and mp["volume_mm3"] > 0), f"{name}: has volume", f"{mp['volume_mm3']:.0f} mm3 via {mp['source']}" if mp else "no mass properties")
    size, _ = bbox(shape)
    if expect_size:
        ok = all(abs(a - b) <= tol for a, b in zip(sorted(size), sorted(expect_size)))
        report(ok, f"{name}: bounding box", f"got {[round(v, 1) for v in size]} expected about {[round(v, 1) for v in expect_size]}")
    return shape, mp


def main():
    print("build123d", getattr(b3, "__version__", "?"))
    # crankshaft: valid, right length, centre of mass on the axis (X), mass near the analytic estimate
    for n in (1, 3, 4):
        d = CRANK.design({"n_throws": n})
        r = d["data"]["crank_radius_mm"]
        shape, mp = check_solid(f"crankshaft {n} throw", d["script"], None)
        if shape is None:
            continue
        size, _ = bbox(shape)
        report(abs(size[0] - d["data"]["length_mm"]) < 1.0, f"crankshaft {n}: length along X", f"{size[0]:.1f} vs {d['data']['length_mm']}")
        if mp:
            off = math.hypot(mp["com"][1], mp["com"][2])
            report(off < 0.03 * r, f"crankshaft {n}: centre of mass on the rotation axis", f"{off:.3f} mm off ({100 * off / r:.2f}% of the crank radius)")
            est = d["data"]["mass_g_estimate"]
            rho = {"steel": 7.85}.get(d["data"]["material"].split()[0].lower(), 7.85)
            real = mp["volume_mm3"] / 1000.0 * rho
            if abs(real - est) / est < 0.06:
                report(True, f"crankshaft {n}: mass vs estimate", f"{real:.0f} g vs {est:.0f} g")
            else:
                warn(f"crankshaft {n}: mass differs from the analytic estimate by more than 6%", f"{real:.0f} g vs {est:.0f} g")
        if n == 3:
            ck, _ = check_solid("crank bearing block", d["block_script"], None)
    # aerodynamic shapes
    check_solid("fuselage", AERO.fuselage_script(900, 120, 100), (900, 120, 100), tol=3)
    wing_span, root = 1500, 220
    shape, _ = check_solid("wing", AERO.wing_script(wing_span, root, 150, "2412", 5, 2), None)
    if shape is not None:
        size, _ = bbox(shape)
        report(abs(size[1] - wing_span) < 5, "wing: span", f"{size[1]:.1f} vs {wing_span}")
        report(root * 0.95 < size[0] < root * 1.35, "wing: chord extent (root chord plus the sweep of the tip)", f"{size[0]:.1f}")
    # arm frame at planned positions, and it must carry the motor mounts
    lay = BAL.motor_layout(4, 304.8)
    motors = [{"x": p["x"], "y": p["y"], "spin": p["spin"]} for p in lay["positions"]]
    fr = FRAME.arm_frame(motors, (0, 0))
    shape, _ = check_solid("arm frame", fr["script"], None)
    if shape is not None:
        size, _ = bbox(shape)
        reach = 2 * (max(math.hypot(m[0], m[1]) for m in fr["mounts"]) + 16)
        report(size[0] > reach * 0.55 and size[1] > reach * 0.55, "arm frame: reaches the motor positions", f"{size[0]:.0f} x {size[1]:.0f} mm")
        try:
            cyl = len([f for f in shape.faces() if str(f.geom_type).split(".")[-1].upper() == "CYLINDER"])
            report(cyl >= 16, "arm frame: has the motor and hub holes", f"{cyl} cylindrical faces")
        except Exception as e:
            warn("arm frame: could not count the holes", f"{type(e).__name__}: {e}")
    print()
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
