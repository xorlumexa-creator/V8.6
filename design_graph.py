"""The Skeleton-Skin design graph: what the engineering geometry requires of a visual skin, and whether the two are still in sync.

Two representations of one product are kept apart on purpose:
  * the SKELETON: build123d / OpenCASCADE solids, exact, the only thing that goes to FEA (Gmsh -> CalculiX);
  * the SKIN: a mesh (Blender, organic surfaces, fairings, appearance), never converted back into CAD.
They are tied together by INTERFACES, read from the solved assembly (not invented): components that must stay enclosed with clearance,
motors and their propeller discs that must stay free, screw heads that must stay reachable. skin_check.py tests a skin mesh against them.
When the skeleton changes its geometry hash changes and a validated skin becomes 'stale': it has to be checked again.
Also here: generators for the Blender side (a script that builds a skin around the interfaces, and a Cycles render script).
Units: mm, assembly frame.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

import cog_balance as BAL


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "item"


def _m(matrix):
    return np.asarray(matrix, float).reshape(4, 4)


def geometry_hash(result):
    """Short hash of the engineering geometry of an assembly result: part placements, measured spaces and hardware."""
    rows = []
    for p in sorted(result.get("parts", []), key=lambda p: p["id"]):
        cav = [[round(v, 1) for v in c["inscribed"]["size"]] for c in (p.get("internal_space") or {}).get("cavities", []) if c.get("inscribed")]
        rows.append([p["id"], [round(float(v), 2) for v in _m(p["matrix"]).ravel()], cav])
    rows.append(len(result.get("hardware", [])))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def interfaces_from_assembly(result, clearance_mm=2.0, tool_access_mm=30.0, prop_clear_mm=100.0, max_bolts=40):
    """Interfaces the skin must respect, from a solved assembly result (as saved by the assembly page)."""
    parts = {p["id"]: p for p in result.get("parts", [])}
    out, seen = [], {}
    for pid, pk in (result.get("packing") or {}).items():
        placements = ((pk or {}).get("packing") or {}).get("placements") or []
        if pid not in parts:
            continue
        M = _m(parts[pid]["matrix"])
        for pl in placements:
            lo = np.array(pl["min"], float)
            lo_w, hi_w = BAL.world_aabb(M, lo, lo + np.array(pl["size"], float))
            seen[pl["name"]] = seen.get(pl["name"], 0) + 1
            out.append({"id": f"{_slug(pl['name'])}-{seen[pl['name']]}", "kind": "component", "mode": "inside", "preserve": True,
                        "box": {"min": [round(float(v), 2) for v in lo_w], "max": [round(float(v), 2) for v in hi_w]},
                        "clearance_mm": clearance_mm, "source": parts[pid]["name"]})
    bi = result.get("balance_inputs") or {}
    inv = {i["name"]: i for i in bi.get("inventory", [])}
    prop_d = ((result.get("balance") or {}).get("lift_plan") or {}).get("prop_diameter_mm")
    n_motor = 0
    for c in bi.get("components", []):
        if BAL.role_of(c["name"]) != "lift_motor" or c.get("part") not in parts:
            continue
        pos = BAL.world_point(_m(parts[c["part"]]["matrix"]), c["center_local"])
        dims = (inv.get(c["name"]) or {}).get("dims") or []
        dia = float(max(dims[:2])) if len(dims) >= 2 else 42.0
        h = float(dims[2]) if len(dims) >= 3 else 30.0
        n_motor += 1
        r = float(prop_d) / 2 * 0.9 if prop_d else 0.0
        out.append({"id": f"lift-motor-{n_motor}", "kind": "motor", "mode": "outside", "preserve": True,
                    "box": {"min": [round(float(pos[0] - dia / 2), 2), round(float(pos[1] - dia / 2), 2), round(float(pos[2] - h / 2), 2)],
                            "max": [round(float(pos[0] + dia / 2), 2), round(float(pos[1] + dia / 2), 2), round(float(pos[2] + h / 2), 2)]},
                    "clearance_mm": 5.0, "access": {"origin": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2] + h / 2), 2)],
                                                     "axis": [0.0, 0.0, 1.0], "length_mm": prop_clear_mm, "radius_mm": round(r, 1)},
                    "source": "propeller airflow stays open above the motor"})
    for h in (result.get("hardware") or [])[:max_bolts]:
        ax = -np.array(h["axis"], float)                     # the head sits at `origin`, the bolt points along `axis`: the tool comes from the other side
        out.append({"id": f"fastener-{h['id']}", "kind": "fastener", "mode": "outside", "preserve": True,
                    "access": {"origin": [round(float(v), 2) for v in h["origin"]], "axis": [round(float(v), 4) for v in ax],
                               "length_mm": tool_access_mm, "radius_mm": 3.0}, "source": f"{h['size']}x{h['length_mm']} screw head"})
    return out


def build_graph(result, interfaces, skin=None):
    return {"version": 1,
            "skeleton": {"representation": "build123d / OpenCASCADE B-Rep (exact, analysed)", "geometry_hash": geometry_hash(result),
                         "parts": [p["name"] for p in result.get("parts", [])]},
            "skin": skin, "interfaces": interfaces,
            "checks": ["mesh quality", "wall thickness", "keep-out volumes", "enclosure and clearance", "access (props, tools)"],
            "rule": "The skin is appearance only and is never converted to CAD. Only the skeleton goes to FEA."}


def skin_record(mesh_bytes, report, geometry_hash_):
    """What is remembered about a validated skin: which skeleton it was checked against and how it did."""
    return {"representation": "mesh (Blender or other), visual only", "mesh_sha256": hashlib.sha256(mesh_bytes).hexdigest()[:16],
            "validated_against": geometry_hash_, "verdict": report["verdict"], "triangles": report["quality"]["triangles"]}


def sync_status(graph, current_hash):
    """no_skin | in_sync | skin_fails | skin_stale (the skeleton changed after the skin was validated)."""
    sk = (graph or {}).get("skin")
    if not sk:
        return "no_skin"
    if sk.get("validated_against") != current_hash:
        return "skin_stale"
    return "in_sync" if sk.get("verdict") == "pass" else "skin_fails"


# ------------------------------------------------------------------------------------------------------------ Blender side
def _r(v):
    return [round(float(x), 3) for x in v]


def blender_skin_script(interfaces, wall_mm=2.0, margin_mm=3.0, bevel_mm=6.0, subsurf=2, max_tris=40000):
    """A Blender (bpy) script that builds a smooth shell around everything that must be enclosed, with a wall, and cuts openings where
    the interfaces need access. The result is decimated to at most `max_tris` triangles so the file stays small enough for a tablet to check.
    Run:  blender -b -P skin.py -- skin.stl     (1 Blender unit = 1 mm)."""
    inside = [(_r(i["box"]["min"]), _r(i["box"]["max"]), float(i.get("clearance_mm", 2.0))) for i in interfaces if i.get("mode") == "inside" and i.get("box")]
    if not inside:
        raise ValueError("no enclosed components to build a skin around")
    cuts = [(_r(i["access"]["origin"]), _r(i["access"]["axis"]), max(float(i["access"].get("radius_mm", 0)), 1.0), float(i["access"].get("length_mm", 30)))
            for i in interfaces if i.get("access")]
    return f'''import sys
import bpy
import bmesh
from mathutils import Vector

# Skin around the engineering geometry (generated by Lumexa). 1 Blender unit = 1 mm.
# INSIDE: (min, max, clearance) of every component that must stay enclosed. CUTS: (origin, axis, radius, length) openings the interfaces need.
INSIDE = {inside}
CUTS = {cuts}
WALL, MARGIN, BEVEL, SUBSURF, MAX_TRIS = {wall_mm}, {margin_mm}, {bevel_mm}, {subsurf}, {int(max_tris)}
out_path = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "skin.stl"

bpy.ops.wm.read_factory_settings(use_empty=True)
bm = bmesh.new()
for lo, hi, clr in INSIDE:
    pad = clr + MARGIN + WALL
    for x in (lo[0] - pad, hi[0] + pad):
        for y in (lo[1] - pad, hi[1] + pad):
            for z in (lo[2] - pad, hi[2] + pad):
                bm.verts.new(Vector((x, y, z)))
bmesh.ops.convex_hull(bm, input=bm.verts[:])
mesh = bpy.data.meshes.new("SkinMesh")
bm.to_mesh(mesh)
bm.free()
skin = bpy.data.objects.new("Skin", mesh)
bpy.context.scene.collection.objects.link(skin)
bpy.context.view_layer.objects.active = skin
skin.select_set(True)


def apply(mod):
    bpy.ops.object.modifier_apply(modifier=mod.name)


bevel = skin.modifiers.new("Round", "BEVEL")
bevel.width = BEVEL
bevel.segments = 4
apply(bevel)
smooth = skin.modifiers.new("Smooth", "SUBSURF")
smooth.levels = SUBSURF
apply(smooth)
wall = skin.modifiers.new("Wall", "SOLIDIFY")
wall.thickness = WALL
wall.offset = -1.0                                   # grow inward: the outside stays where the hull is
apply(wall)
for origin, axis, radius, length in CUTS:
    d = Vector(axis).normalized()
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=Vector(origin) + d * (length / 2))
    cyl = bpy.context.active_object
    cyl.rotation_mode = "QUATERNION"
    cyl.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    bpy.context.view_layer.objects.active = skin
    cut = skin.modifiers.new("Open", "BOOLEAN")
    cut.operation = "DIFFERENCE"
    cut.object = cyl
    cut.solver = "EXACT"
    apply(cut)
    bpy.data.objects.remove(cyl, do_unlink=True)

tris = sum(len(p.vertices) - 2 for p in skin.data.polygons)
if tris > MAX_TRIS:                                   # keep the file small: the backend that checks it is a 4 GB tablet
    thin = skin.modifiers.new("Budget", "DECIMATE")
    thin.ratio = MAX_TRIS / tris
    apply(thin)
    print("decimated", tris, "->", sum(len(p.vertices) - 2 for p in skin.data.polygons), "triangles")
bpy.ops.object.select_all(action="DESELECT")
skin.select_set(True)
try:
    bpy.ops.wm.stl_export(filepath=out_path, export_selected_objects=True)      # Blender 4.x
except AttributeError:
    bpy.ops.export_mesh.stl(filepath=out_path, use_selection=True)               # Blender 3.x and older
print("wrote", out_path)
'''


def blender_render_script(parts, skin_file="skin.stl", samples=64, resolution=(1600, 1000)):
    """A Cycles render of the skeleton parts and the skin. parts: [{file, color: [r, g, b], name}]. Run: blender -b -P render.py -- out.png"""
    items = [{"file": p["file"], "color": [float(c) for c in p.get("color", [0.55, 0.58, 0.62])][:3], "name": p.get("name", "part")} for p in parts]
    return f'''import sys
import bpy
from mathutils import Vector

# Cycles render of the engineering parts and the skin (generated by Lumexa). 1 Blender unit = 1 mm.
PARTS = {items}
SKIN = {skin_file!r}
SAMPLES, RES = {int(samples)}, {tuple(int(v) for v in resolution)}
out_path = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "render.png"

bpy.ops.wm.read_factory_settings(use_empty=True)


def load(path):
    try:
        bpy.ops.wm.stl_import(filepath=path)             # Blender 4.x
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=path)           # Blender 3.x and older
    return bpy.context.selected_objects[0]


def material(name, color, alpha=1.0, rough=0.35, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    return m


objs = []
for p in PARTS:
    o = load(p["file"])
    o.data.materials.append(material(p["name"], p["color"], metal=0.2))
    objs.append(o)
skin = load(SKIN)
skin.data.materials.append(material("Skin", (0.92, 0.93, 0.95), alpha=0.55, rough=0.2))
objs.append(skin)

lo = Vector((min(min(v[i] for v in (o.matrix_world @ Vector(c) for c in o.bound_box)) for o in objs) for i in range(3)))
hi = Vector((max(max(v[i] for v in (o.matrix_world @ Vector(c) for c in o.bound_box)) for o in objs) for i in range(3)))
centre, size = (lo + hi) / 2, (hi - lo).length
bpy.ops.object.camera_add(location=centre + Vector((0.9, -1.3, 0.7)) * size)
cam = bpy.context.active_object
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam
bpy.ops.object.light_add(type="AREA", location=centre + Vector((-0.6, -0.8, 1.2)) * size)
bpy.context.active_object.data.energy = 3.0 * size * size
bpy.context.active_object.data.size = size
bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.color = (0.05, 0.05, 0.06)

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.render.resolution_x, sc.render.resolution_y = RES
sc.render.filepath = out_path
bpy.ops.render.render(write_still=True)
print("rendered", out_path)
'''
