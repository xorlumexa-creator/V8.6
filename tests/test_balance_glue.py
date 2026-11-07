"""Checks the glue in main.py that feeds cog_balance from a solved assembly. main.py needs fastapi, so the function is lifted out of
the source and run against a hand-made result with the same shape as the real one. Run: python tests/test_balance_glue.py"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import assembly as ASM  # noqa: E402
import cog_balance as BAL  # noqa: E402


def _lift(name):
    src = open(os.path.join(HERE, "..", "main.py"), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np, "ASM": ASM, "BAL": BAL, "ast": ast}
    exec(compile(ast.Module([node], []), "main.py", "exec"), ns)
    return ns[name]


def _fixture(matrix):
    res = {"parts": [{"id": "s", "matrix": matrix, "internal_space": {"cavities": [
        {"id": "C1", "role": "space", "inscribed": {"min": [-50, -37, -30], "size": [100, 74, 62]}}]}}],
        "packing": {"s": {"cavity": "C1", "packing": {"placements": [
            {"name": "Battery pack", "index": 0, "min": [-10, -10, -30], "size": [34, 20, 62]}]}}}}
    infos = [{"id": "s", "name": "Shell", "mass": {"volume_mm3": 200_000, "com": [0, 0, 0]},
              "features": {"bbox_mm": {"min": [-55, -40, -33], "max": [55, 40, 33]}}}]
    bopt = {"vehicle": "multirotor", "materials": {"s": "PETG"},
            "inventory": [{"name": "Battery pack", "dims_text": "62 x 34 x 20 mm, 429.8 g", "qty": 1},
                          {"name": "Hover motor", "dims_text": "41.8 mm diameter, 82 g (104 g with cable)", "qty": 4}]}
    return res, infos, bopt


def test_inputs_have_real_masses_positions_and_travel():
    f = _lift("_balance_inputs")
    for matrix in (np.eye(4).tolist(), np.eye(4).ravel().tolist()):             # the result's matrix may be nested or flat
        res, infos, bopt = _fixture(matrix)
        inp = f(res, infos, bopt)
        assert inp["parts"][0]["volume_mm3"] == 200_000 and inp["parts"][0]["material"] == "PETG"
        c = inp["components"][0]
        assert c["name"] == "Battery pack" and abs(c["mass_each_g"] - 429.8) < 1e-9 and c["source"] == "datasheet"
        assert np.allclose(c["center_local"], [7, 0, 1])
        assert np.allclose(c["travel_world"]["min"], [-40, -27, 0]) and np.allclose(c["travel_world"]["max"], [26, 27, 0])
        a = BAL.analyze(inp)
        assert a["mass"]["counted_g"] > 429.8                                    # shell + battery
        assert a["not_placed"][0]["name"] == "Hover motor" and a["not_placed"][0]["mass_each_g"] == 82
        assert a["lift_plan"]["status"] == "planned" and a["battery_shift"]["battery"] == "Battery pack"


def test_travel_follows_the_part_rotation():
    f = _lift("_balance_inputs")
    R = np.eye(4)
    R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                              # part turned 90 degrees about Z in the assembly
    res, infos, bopt = _fixture(R.tolist())
    c = f(res, infos, bopt)["components"][0]
    tr = c["travel_world"]
    # the battery slides 56 mm along the part's x, which is now the assembly's y
    assert abs((tr["max"][1] - tr["min"][1]) - (100 - 34)) < 1e-6 and abs((tr["max"][0] - tr["min"][0]) - (74 - 20)) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
