"""Pure Blender-Python scene builder and headless renderer.

Has zero dependency on ASE or external packages so it runs directly inside
Blender's embedded Python interpreter (both headless CLI and GUI addon).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Quaternion, Vector

try:
    from atomic_blender.elements import (
        COVALENT_RADII,
        METAL_ELEMENTS,
        VIEW_PRESETS,
        resolve_element_color,
        resolve_element_radius,
    )
except ImportError:
    if __package__:
        from .elements import (  # type: ignore[no-redef]
            COVALENT_RADII,
            METAL_ELEMENTS,
            VIEW_PRESETS,
            resolve_element_color,
            resolve_element_radius,
        )
    else:
        _here = str(Path(__file__).resolve().parent)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from elements import (  # type: ignore[import-not-found,no-redef]
            COVALENT_RADII,
            METAL_ELEMENTS,
            VIEW_PRESETS,
            resolve_element_color,
            resolve_element_radius,
        )

COLLECTION_NAME = "AtomicBlender"


def srgb_hex_to_linear_rgba(hex_color: str) -> tuple[float, float, float, float]:
    """Convert an sRGB hex string (``#RRGGBB``) to Blender linear RGBA."""
    h = hex_color.lstrip("#")
    vals = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255.0
        lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        vals.append(lin)
    return (vals[0], vals[1], vals[2], 1.0)


def make_material(
    name: str,
    hex_color: str,
    roughness: float = 0.35,
    specular: float = 0.4,
    metallic: float = 0.0,
    subsurface_weight: float = 0.0,
    subsurface_scale: float = 0.1,
    coat_weight: float = 0.0,
    coat_roughness: float = 0.1,
    ao_factor: float = 0.0,
    ao_distance: float = 2.5,
    alpha: float = 1.0,
) -> bpy.types.Material:
    """Create a Principled BSDF material with optional metallic, SSS, clearcoat, and AO shading."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None and hasattr(mat, "use_nodes"):
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.get("Material Output") or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    rgba = srgb_hex_to_linear_rgba(hex_color)

    if ao_factor > 0.0:
        ao_node = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao_node.samples = 8
        ao_node.inputs["Distance"].default_value = float(ao_distance)
        ao_node.inputs["Color"].default_value = rgba
        try:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = float(ao_factor)
            mix.inputs[6].default_value = rgba
            nt.links.new(ao_node.outputs["Color"], mix.inputs[7])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        except Exception:
            mix_rgb = nt.nodes.new("ShaderNodeMixRGB")
            mix_rgb.blend_type = "MULTIPLY"
            mix_rgb.inputs["Fac"].default_value = float(ao_factor)
            mix_rgb.inputs["Color1"].default_value = rgba
            nt.links.new(ao_node.outputs["Color"], mix_rgb.inputs["Color2"])
            nt.links.new(mix_rgb.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = rgba

    bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(metallic)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(specular)
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(specular)
    if subsurface_weight > 0.0:
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = float(subsurface_weight)
        elif "Subsurface" in bsdf.inputs:
            bsdf.inputs["Subsurface"].default_value = float(subsurface_weight)
        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = float(subsurface_scale)
    if coat_weight > 0.0:
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = float(coat_weight)
        elif "Clearcoat" in bsdf.inputs:
            bsdf.inputs["Clearcoat"].default_value = float(coat_weight)
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = float(coat_roughness)
        elif "Clearcoat Roughness" in bsdf.inputs:
            bsdf.inputs["Clearcoat Roughness"].default_value = float(coat_roughness)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = max(0.0, min(1.0, float(alpha)))
    if float(alpha) < 0.995 and hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND"
        except Exception:
            pass
    return mat


def _element_material_kwargs(elem: str, style: dict[str, Any]) -> dict[str, float]:
    """Resolve material shader parameters for an element (supporting metal/nonmetal profiles)."""
    mat_cfg = style.get("material", {})
    keys = (
        "roughness",
        "specular",
        "metallic",
        "subsurface_weight",
        "subsurface_scale",
        "coat_weight",
        "coat_roughness",
        "ao_factor",
        "ao_distance",
    )
    defaults = {
        "roughness": 0.35,
        "specular": 0.4,
        "metallic": 0.0,
        "subsurface_weight": 0.0,
        "subsurface_scale": 0.1,
        "coat_weight": 0.0,
        "coat_roughness": 0.1,
        "ao_factor": 0.0,
        "ao_distance": 2.5,
    }
    merged = {k: float(mat_cfg.get(k, defaults[k])) for k in keys}
    profile_key = "metal" if elem in METAL_ELEMENTS else "nonmetal"
    profile = mat_cfg.get(profile_key)
    if isinstance(profile, dict):
        for k in keys:
            if k in profile:
                merged[k] = float(profile[k])
    return merged


def _purge_unused_scene_datablocks(
    meshes: set[Any],
    materials: set[Any],
    lights: set[Any],
    cameras: set[Any],
) -> None:
    """Remove unused meshes, materials, lights, and cameras after clearing a rebuilt scene."""
    prefixes = ("proto_", "bond_", "cell_", "poly_", "cellvec_", "mat_", "AtomicSun", "AtomicFill", "AtomicRim", "OrthoCam")
    for block_pool, candidates in (
        (bpy.data.meshes, meshes),
        (bpy.data.lights, lights),
        (bpy.data.cameras, cameras),
        (bpy.data.materials, materials),
    ):
        for db in list(candidates):
            try:
                if getattr(db, "users", 1) == 0:
                    block_pool.remove(db)
            except Exception:
                pass
        for db in list(block_pool):
            try:
                if getattr(db, "users", 1) == 0 and db.name.startswith(prefixes):
                    block_pool.remove(db)
            except Exception:
                pass


def _ensure_collection(scene: bpy.types.Scene, clear_existing: bool) -> bpy.types.Collection:
    """Return the dedicated collection, removing previous objects and zero-user datablocks."""
    col = bpy.data.collections.get(COLLECTION_NAME)
    old_meshes: set[Any] = set()
    old_materials: set[Any] = set()
    old_lights: set[Any] = set()
    old_cameras: set[Any] = set()
    if col is not None and clear_existing:
        for obj in list(col.objects):
            data = getattr(obj, "data", None)
            if data is not None:
                obj_type = getattr(obj, "type", "")
                if obj_type == "MESH":
                    old_meshes.add(data)
                    for mat in getattr(data, "materials", ()):
                        if mat is not None:
                            old_materials.add(mat)
                elif obj_type == "LIGHT":
                    old_lights.add(data)
                elif obj_type == "CAMERA":
                    old_cameras.add(data)
            for slot in getattr(obj, "material_slots", ()):
                if getattr(slot, "material", None) is not None:
                    old_materials.add(slot.material)
            bpy.data.objects.remove(obj, do_unlink=True)
    if clear_existing:
        # Remove Blender's default startup Cube/Light/Camera (which live in 'Collection')
        for default_name in ("Cube", "Light", "Camera"):
            obj = bpy.data.objects.get(default_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
        _purge_unused_scene_datablocks(old_meshes, old_materials, old_lights, old_cameras)
    if col is None:
        col = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(col)
    return col


def _create_cylinder_mesh(
    p0: Vector,
    p1: Vector,
    radius: float,
    mat: bpy.types.Material,
    name: str,
    col: bpy.types.Collection,
    segments: int = 14,
    smooth: bool = True,
) -> bpy.types.Object | None:
    """Create a smooth cylinder connecting ``p0`` and ``p1`` without slow mode switches."""
    dv = p1 - p0
    length = dv.length
    if length < 1e-6:
        return None
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    # Build unit cylinder along Z using bmesh for speed and zero bpy.ops overhead
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=segments,
        radius1=radius,
        radius2=radius,
        depth=length,
    )
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (p0 + p1) * 0.5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = dv.normalized().to_track_quat("Z", "Y")
    col.objects.link(obj)
    return obj


def _create_uv_sphere_mesh(
    name: str,
    radius: float,
    segments: int,
    ring_count: int,
    smooth: bool,
    mat: bpy.types.Material,
) -> bpy.types.Mesh:
    """Create a prototype UV sphere mesh via bmesh for fast instancing."""
    import bmesh

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(
        bm,
        u_segments=segments,
        v_segments=ring_count,
        radius=radius,
    )
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    mesh.materials.append(mat)
    return mesh


def _parse_cutoffs(cutoffs_cfg: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Normalize bond cutoff dictionary keys like ``'C-H'`` into symmetric pairs."""
    table: dict[tuple[str, str], float] = {}
    for key, val in cutoffs_cfg.items():
        parts = [p.strip() for p in str(key).replace("_", "-").split("-") if p.strip()]
        if len(parts) != 2:
            continue
        a, b = parts
        cutoff = float(val)
        table[(a, b)] = cutoff
        table[(b, a)] = cutoff
    return table


def _find_bonds_spatial(
    symbols: list[str],
    positions: list[Vector],
    bond_cfg: dict[str, Any],
    ghost_start_index: int | None = None,
) -> list[tuple[int, int]]:
    """Find bonded atom index pairs using an O(N) spatial hash grid."""
    mode = bond_cfg.get("mode", "table")
    if mode == "none" or not bond_cfg.get("enabled", True):
        return []

    table = _parse_cutoffs(bond_cfg.get("cutoffs", {}))
    tol = float(bond_cfg.get("covalent_tolerance", 1.2))
    pair_cutoffs: dict[tuple[str, str], float] = {}

    unique_elems = sorted(set(symbols))
    max_cut = 0.0
    for ei in unique_elems:
        for ej in unique_elems:
            if mode == "table":
                c = table.get((ei, ej), 0.0)
            elif mode == "auto":
                ri = COVALENT_RADII.get(ei, 0.75)
                rj = COVALENT_RADII.get(ej, 0.75)
                # Avoid metallic bonding clutter in alloys unless explicitly listed in table
                c = table.get((ei, ej), (ri + rj) * tol)
            else:
                c = table.get((ei, ej), 0.0)
            if c > 0.0:
                pair_cutoffs[(ei, ej)] = c
                if c > max_cut:
                    max_cut = c

    if max_cut <= 0.0:
        return []

    cell_size = max(max_cut, 0.5)
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, pos in enumerate(positions):
        key = (
            math.floor(pos.x / cell_size),
            math.floor(pos.y / cell_size),
            math.floor(pos.z / cell_size),
        )
        grid[key].append(idx)

    bonds: list[tuple[int, int]] = []
    for (gx, gy, gz), bucket in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nbr_bucket = grid.get((gx + dx, gy + dy, gz + dz))
                    if not nbr_bucket:
                        continue
                    for i in bucket:
                        ei = symbols[i]
                        pi = positions[i]
                        for j in nbr_bucket:
                            if j <= i:
                                continue
                            if (
                                ghost_start_index is not None
                                and 0 < ghost_start_index < len(symbols)
                                and (i < ghost_start_index) != (j < ghost_start_index)
                            ):
                                continue
                            cut = pair_cutoffs.get((ei, symbols[j]))
                            if cut is None:
                                continue
                            if (positions[j] - pi).length <= cut:
                                bonds.append((i, j))
    return bonds


def compute_cell_corners(origin: Vector, cell_vecs: list[Vector]) -> list[Vector]:
    """Compute the 8 corners of a (possibly triclinic) 3D periodic cell."""
    a, b, c = cell_vecs
    return [
        origin,
        origin + a,
        origin + a + b,
        origin + b,
        origin + c,
        origin + a + c,
        origin + a + b + c,
        origin + b + c,
    ]


def _direction_to_quaternion(view_dir: Vector, up_hint: Vector | None = None) -> Quaternion:
    """Compute Blender camera orientation looking along ``-view_dir`` without pole singularities."""
    forward = (-view_dir).normalized()
    up = (up_hint if up_hint is not None else Vector((0.0, 0.0, 1.0))).normalized()
    if abs(forward.dot(up)) > 0.999:
        up = Vector((0.0, 1.0, 0.0)) if abs(forward.z) > 0.9 else Vector((0.0, 0.0, 1.0))
    right = forward.cross(up).normalized()
    cam_up = right.cross(forward).normalized()
    # Camera local axes in world space: +X = right, +Y = cam_up, +Z = -forward (= view_dir)
    rot_mat = Matrix((right, cam_up, -forward)).transposed()
    return rot_mat.to_quaternion()


def resolve_preset_orientation(
    preset: str,
    cell_vecs: list[Vector] | None = None,
) -> Quaternion:
    """Return camera quaternion for a named view preset (``iso``, ``side-oblique``, ``a/b/c``, ``a*/b*/c*``)."""
    name = preset.lower().strip()
    if name == "iso":
        az, el = math.radians(35.0), math.radians(20.0)
        view_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        return (-view_dir).to_track_quat("-Z", "Y")
    if name == "side-oblique":
        az, el = math.radians(25.0), math.radians(10.0)
        view_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        return (-view_dir).to_track_quat("-Z", "Y")
    a = cell_vecs[0] if cell_vecs and cell_vecs[0].length > 1e-6 else Vector((1.0, 0.0, 0.0))
    b = cell_vecs[1] if cell_vecs and cell_vecs[1].length > 1e-6 else Vector((0.0, 1.0, 0.0))
    c = cell_vecs[2] if cell_vecs and cell_vecs[2].length > 1e-6 else Vector((0.0, 0.0, 1.0))
    if name == "a":
        return _direction_to_quaternion(a.normalized(), c.normalized())
    if name == "b":
        return _direction_to_quaternion(b.normalized(), c.normalized())
    if name == "c":
        up_c = c.cross(a)
        return _direction_to_quaternion(c.normalized(), up_c.normalized() if up_c.length > 1e-6 else b.normalized())
    if name == "a*":
        na = b.cross(c)
        return _direction_to_quaternion((na if na.length > 1e-6 else a).normalized(), c.normalized())
    if name == "b*":
        nb = c.cross(a)
        return _direction_to_quaternion((nb if nb.length > 1e-6 else b).normalized(), c.normalized())
    if name == "c*":
        nc = a.cross(b)
        nc_v = (nc if nc.length > 1e-6 else c).normalized()
        up_nc = nc_v.cross(a)
        return _direction_to_quaternion(nc_v, up_nc.normalized() if up_nc.length > 1e-6 else b.normalized())
    raise ValueError(f"Unknown view preset {preset!r}. Expected one of {VIEW_PRESETS}.")


def compute_framing_points(
    positions: list[Vector],
    radii: list[float],
    has_cell: bool,
    origin: Vector,
    cell_vecs: list[Vector],
    crop_z: tuple[float, float] | None,
) -> tuple[Vector, list[Vector]]:
    """Compute target center and 8 bounding points for orthographic framing."""
    if has_cell:
        a, b, c = cell_vecs
        corners = compute_cell_corners(origin, cell_vecs)
        if crop_z is not None and positions:
            pad_bot, pad_top = crop_z
            zs = [p.z for p in positions]
            zmin = min(zs) - pad_bot
            zmax = max(zs) + pad_top
            base = [origin, origin + a, origin + a + b, origin + b]
            if abs(c.z) > 1e-6:
                t_min = (zmin - origin.z) / c.z
                t_max = (zmax - origin.z) / c.z
                pts_frame = [p + c * t_min for p in base] + [p + c * t_max for p in base]
            else:
                pts_frame = [Vector((p.x, p.y, zmin)) for p in base] + [Vector((p.x, p.y, zmax)) for p in base]
            target = sum(pts_frame, Vector((0.0, 0.0, 0.0))) * (1.0 / len(pts_frame))
            return target, pts_frame
        target = origin + 0.5 * (a + b + c)
        return target, corners

    if not positions:
        return Vector((0.0, 0.0, 0.0)), [Vector((0.0, 0.0, 0.0))]

    xs = [p.x - r for p, r in zip(positions, radii)] + [p.x + r for p, r in zip(positions, radii)]
    ys = [p.y - r for p, r in zip(positions, radii)] + [p.y + r for p, r in zip(positions, radii)]
    zs = [p.z - r for p, r in zip(positions, radii)] + [p.z + r for p, r in zip(positions, radii)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    if crop_z is not None:
        raw_zs = [p.z for p in positions]
        zmin = min(raw_zs) - crop_z[0]
        zmax = max(raw_zs) + crop_z[1]
    target = Vector((0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)))
    pts_frame = [
        Vector((px, py, pz))
        for px in (xmin, xmax)
        for py in (ymin, ymax)
        for pz in (zmin, zmax)
    ]
    return target, pts_frame


def fit_camera_to_points(
    cam_obj: bpy.types.Object,
    rot_quat: Quaternion,
    target: Vector,
    pts_frame: list[Vector],
    aspect: float,
    margin: float = 1.08,
    base_distance: float = 200.0,
    base_clip_end: float = 1000.0,
) -> None:
    """Position camera along ``rot_quat`` looking at ``target`` and fit ``ortho_scale``."""
    view_dir = (rot_quat @ Vector((0.0, 0.0, 1.0))).normalized()
    max_extent = max(((p - target).length for p in pts_frame), default=10.0)
    dist = max(base_distance, max_extent * 4.0)
    cam_obj.location = target + view_dir * dist
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = rot_quat
    cam_obj.data.clip_start = 1.0
    cam_obj.data.clip_end = max(base_clip_end, dist + max_extent * 4.0)

    cam_rot_inv = rot_quat.inverted()
    xs_c, ys_c = [], []
    for p in pts_frame:
        pc = cam_rot_inv @ (p - target)
        xs_c.append(abs(pc.x))
        ys_c.append(abs(pc.y))
    half_w = max(xs_c, default=1.0) * margin
    half_h = max(ys_c, default=1.0) * margin
    cam_obj.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect)


def apply_camera_view(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    view: str | dict[str, Any],
    framing_data: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    fit_override: bool = False,
) -> None:
    """Apply either a named view preset or a loaded ``view.json`` dictionary to ``cam_obj``."""
    cam_cfg = (style or {}).get("camera", {})
    margin = float(cam_cfg.get("margin", 1.08))
    base_dist = float(cam_cfg.get("distance", 200.0))
    base_clip_end = float(cam_cfg.get("clip_end", 1000.0))
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)

    if isinstance(view, str):
        cell_vecs = framing_data.get("cell_vecs") if framing_data else None
        rot_quat = resolve_preset_orientation(view, cell_vecs)
        if framing_data is not None:
            fit_camera_to_points(
                cam_obj=cam_obj,
                rot_quat=rot_quat,
                target=framing_data["target"],
                pts_frame=framing_data["pts_frame"],
                aspect=aspect,
                margin=margin,
                base_distance=base_dist,
                base_clip_end=base_clip_end,
            )
        else:
            cam_obj.rotation_mode = "QUATERNION"
            cam_obj.rotation_quaternion = rot_quat
        return

    # Dict view (from view.json)
    if "resolution" in view and len(view["resolution"]) == 2:
        scene.render.resolution_x = int(view["resolution"][0])
        scene.render.resolution_y = int(view["resolution"][1])
        aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)

    cam_type = str(view.get("camera_type", "ORTHO"))
    cam_obj.data.type = cam_type
    if cam_type == "PERSP" and "lens" in view:
        cam_obj.data.lens = float(view["lens"])

    if "rotation_quaternion" in view:
        q = view["rotation_quaternion"]
        rot_quat = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3]))).normalized()
    elif "preset" in view:
        cell_vecs = framing_data.get("cell_vecs") if framing_data else None
        rot_quat = resolve_preset_orientation(str(view["preset"]), cell_vecs)
    else:
        raise ValueError("view.json must contain 'rotation_quaternion' or 'preset'.")

    fit_to_scene = fit_override or bool(view.get("fit_to_scene", False))
    if fit_to_scene and framing_data is not None:
        fit_camera_to_points(
            cam_obj=cam_obj,
            rot_quat=rot_quat,
            target=framing_data["target"],
            pts_frame=framing_data["pts_frame"],
            aspect=aspect,
            margin=margin,
            base_distance=base_dist,
            base_clip_end=base_clip_end,
        )
        return

    if "location" not in view or "ortho_scale" not in view:
        raise ValueError("Exact view.json requires 'location' and 'ortho_scale'.")
    cam_obj.location = Vector(view["location"])
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = rot_quat
    cam_obj.data.ortho_scale = float(view["ortho_scale"])
    cam_obj.data.clip_start = float(view.get("clip_start", cam_cfg.get("clip_start", 1.0)))
    cam_obj.data.clip_end = float(view.get("clip_end", cam_cfg.get("clip_end", 1000.0)))


def export_camera_view(scene: bpy.types.Scene, cam_obj: bpy.types.Object) -> dict[str, Any]:
    """Serialize the current camera transform, ortho_scale, and render resolution to dict."""
    cam_obj.rotation_mode = "QUATERNION"
    q = cam_obj.rotation_quaternion
    loc = cam_obj.location
    return {
        "schema_version": 1,
        "camera_type": cam_obj.data.type,
        "location": [float(loc.x), float(loc.y), float(loc.z)],
        "rotation_quaternion": [float(q.w), float(q.x), float(q.y), float(q.z)],
        "ortho_scale": float(cam_obj.data.ortho_scale),
        "lens": float(cam_obj.data.lens),
        "clip_start": float(cam_obj.data.clip_start),
        "clip_end": float(cam_obj.data.clip_end),
        "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        "fit_to_scene": False,
    }


def configure_render_and_world(scene: bpy.types.Scene, style: dict[str, Any]) -> None:
    """Apply Cycles device, sampling, denoising, transparency, and world settings."""
    render_cfg = style.get("render", {})
    scene.render.engine = render_cfg.get("engine", "CYCLES")

    if scene.render.engine == "CYCLES":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        gpu_ok = False
        for backend in render_cfg.get("device_priority", ["OPTIX", "CUDA", "CPU"]):
            if backend == "CPU":
                break
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == backend]
                if devs:
                    for d in devs:
                        d.use = True
                    scene.cycles.device = "GPU"
                    gpu_ok = True
                    print(f"Using Cycles backend: {backend}")
                    break
            except Exception:
                pass
        if not gpu_ok:
            scene.cycles.device = "CPU"
            print("Using Cycles backend: CPU")

        scene.cycles.samples = int(render_cfg.get("samples", 64))
        scene.cycles.use_denoising = bool(render_cfg.get("use_denoising", True))

    scene.render.film_transparent = bool(render_cfg.get("film_transparent", True))
    scene.render.image_settings.file_format = render_cfg.get("file_format", "PNG")
    scene.render.image_settings.color_mode = render_cfg.get("color_mode", "RGBA")
    scene.render.image_settings.color_depth = str(render_cfg.get("color_depth", "8"))

    world_cfg = style.get("world", {})
    world = scene.world or bpy.data.worlds.new("AtomicWorld")
    scene.world = world
    if world.node_tree is None and hasattr(world, "use_nodes"):
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        wout = world.node_tree.nodes.get("World Output") or world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs["Background"], wout.inputs["Surface"])
    if bg is not None:
        wcol = world_cfg.get("color", [0.85, 0.85, 0.85, 1.0])
        bg.inputs["Color"].default_value = tuple(float(c) for c in wcol)
        bg.inputs["Strength"].default_value = float(world_cfg.get("strength", 0.6))


def _create_directional_area_light(
    col: bpy.types.Collection,
    name: str,
    target: Vector,
    cfg: dict[str, Any],
    default_energy: float,
    default_size: float,
    default_az: float,
    default_el: float,
    default_dist: float,
) -> None:
    """Create an AREA (or other typed) fill/rim light aimed at ``target``."""
    ltype = str(cfg.get("type", "AREA"))
    ldata = bpy.data.lights.new(name, type=ltype)
    ldata.energy = float(cfg.get("energy", default_energy))
    if ltype == "AREA":
        ldata.size = float(cfg.get("size", default_size))
    lobj = bpy.data.objects.new(name, ldata)
    az = math.radians(float(cfg.get("azimuth_deg", default_az)))
    el = math.radians(float(cfg.get("elevation_deg", default_el)))
    dist = float(cfg.get("distance", default_dist))
    ldir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))).normalized()
    lobj.location = target + ldir * dist
    lobj.rotation_mode = "QUATERNION"
    lobj.rotation_quaternion = (-ldir).to_track_quat("-Z", "Y")
    col.objects.link(lobj)


def setup_lighting(
    col: bpy.types.Collection,
    target: Vector,
    style: dict[str, Any],
) -> None:
    """Create Sun key light, Area fill light, and optional Rim light aimed at ``target``."""
    light_cfg = style.get("lighting", {})
    sun_cfg = light_cfg.get("sun", {})
    sun_data = bpy.data.lights.new("AtomicSun", type="SUN")
    sun_data.energy = float(sun_cfg.get("energy", 3.0))
    sun_obj = bpy.data.objects.new("AtomicSun", sun_data)
    rot_deg = sun_cfg.get("rotation_deg", [35.0, -20.0, 30.0])
    sun_obj.rotation_euler = tuple(math.radians(float(a)) for a in rot_deg)
    col.objects.link(sun_obj)

    _create_directional_area_light(
        col=col,
        name="AtomicFill",
        target=target,
        cfg=light_cfg.get("fill", {}),
        default_energy=4000.0,
        default_size=40.0,
        default_az=-40.0,
        default_el=25.0,
        default_dist=45.0,
    )
    if isinstance(light_cfg.get("rim"), dict):
        _create_directional_area_light(
            col=col,
            name="AtomicRim",
            target=target,
            cfg=light_cfg["rim"],
            default_energy=2500.0,
            default_size=30.0,
            default_az=150.0,
            default_el=35.0,
            default_dist=42.0,
        )


def build_scene(
    structure: dict[str, Any],
    style: dict[str, Any],
    view: str | dict[str, Any] = "iso",
    crop_z: tuple[float, float] | None = None,
    resolution: tuple[int, int] | None = None,
    bonds_mode_override: str | None = None,
    reset_factory: bool = True,
    fit_override: bool = False,
    radius_scale: float = 1.0,
    bond_color_override: str | None = None,
    show_cell_override: bool | None = None,
) -> dict[str, Any]:
    """Build atoms, bonds, cell wireframe, lighting, and camera from neutral structure JSON."""
    if reset_factory:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    col = _ensure_collection(scene, clear_existing=not reset_factory)
    configure_render_and_world(scene, style)

    cam_cfg = style.get("camera", {})
    default_res = cam_cfg.get("default_resolution", [1400, 1400])
    res_x, res_y = resolution if resolution is not None else (int(default_res[0]), int(default_res[1]))
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    symbols: list[str] = list(structure["symbols"])
    positions = [Vector(p) for p in structure["positions"]]
    raw_alphas = structure.get("alphas")
    if isinstance(raw_alphas, list) and len(raw_alphas) == len(symbols):
        alphas = [max(0.02, min(1.0, float(a))) for a in raw_alphas]
    else:
        alphas = [1.0] * len(symbols)
    raw_r_scales = structure.get("radius_scales")
    if isinstance(raw_r_scales, list) and len(raw_r_scales) == len(symbols):
        r_scales = [max(0.15, min(3.0, float(s))) for s in raw_r_scales]
    else:
        r_scales = [1.0] * len(symbols)
    has_cell = bool(structure.get("has_cell", True))
    origin = Vector(structure.get("origin", [0.0, 0.0, 0.0]))
    cell_vecs = [Vector(v) for v in structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])]

    # Atom meshes instanced per (element, alpha, radius_scale, color) for per-atom transparency & sizing/colormaps
    eff_radius_scale = float(style.get("elements", {}).get("radius_scale", 1.0)) * float(radius_scale)
    mesh_cfg = style.get("atom_mesh", {})
    segments = int(mesh_cfg.get("segments", 24))
    ring_count = int(mesh_cfg.get("ring_count", 12))
    smooth = bool(mesh_cfg.get("smooth", True))

    raw_colors = structure.get("colors")
    if isinstance(raw_colors, list) and len(raw_colors) == len(symbols):
        atom_colors = [str(c) for c in raw_colors]
    else:
        atom_colors = [resolve_element_color(s, style) for s in symbols]

    proto_meshes: dict[tuple[str, float, float, str], bpy.types.Mesh] = {}
    elem_radii_cache: dict[str, float] = {}
    elem_materials: dict[tuple[str, float, str], bpy.types.Material] = {}
    for elem, a_val, col_hex in sorted({(sym, round(a, 3), col_h) for sym, a, col_h in zip(symbols, alphas, atom_colors)}):
        mat_kwargs = _element_material_kwargs(elem, style)
        safe_col = col_hex.lstrip("#")
        elem_materials[(elem, a_val, col_hex)] = make_material(
            f"mat_{elem}_{safe_col}_{int(round(a_val * 1000))}",
            col_hex,
            alpha=a_val,
            **mat_kwargs,
        )
    for elem, a_val, r_val, col_hex in sorted(
        {(sym, round(a, 3), round(rs, 3), col_h) for sym, a, rs, col_h in zip(symbols, alphas, r_scales, atom_colors)}
    ):
        if elem not in elem_radii_cache:
            elem_radii_cache[elem] = resolve_element_radius(elem, style) * eff_radius_scale
        rad = elem_radii_cache[elem] * r_val
        mat = elem_materials[(elem, a_val, col_hex)]
        safe_col = col_hex.lstrip("#")
        proto_meshes[(elem, a_val, r_val, col_hex)] = _create_uv_sphere_mesh(
            name=f"proto_{elem}_{safe_col}_{int(round(a_val * 1000))}_{int(round(r_val * 1000))}",
            radius=rad,
            segments=segments,
            ring_count=ring_count,
            smooth=smooth,
            mat=mat,
        )

    for idx, (elem, pos, a_val, r_val, col_hex) in enumerate(zip(symbols, positions, alphas, r_scales, atom_colors)):
        obj = bpy.data.objects.new(
            f"atom_{idx}_{elem}", proto_meshes[(elem, round(a_val, 3), round(r_val, 3), col_hex)]
        )
        obj.location = pos
        col.objects.link(obj)

    # Bonds (single-color or MolecularNodes-style bicolor split cylinders)
    bond_cfg = dict(style.get("bonds", {}))
    if bonds_mode_override is not None and bonds_mode_override != "style":
        bond_cfg["mode"] = bonds_mode_override
        bond_cfg["enabled"] = bonds_mode_override != "none"
    ghost_start = structure.get("ghost_start_index")
    bond_pairs = _find_bonds_spatial(
        symbols,
        positions,
        bond_cfg,
        ghost_start_index=int(ghost_start) if ghost_start is not None else None,
    )
    b_rad = float(bond_cfg.get("radius", 0.12))
    b_seg = int(bond_cfg.get("segments", 14))
    b_smooth = bool(bond_cfg.get("smooth", True))
    if bond_pairs:
        color_mode = (
            bond_color_override
            if bond_color_override and bond_color_override != "style"
            else str(bond_cfg.get("color_mode", "single"))
        )
        if color_mode == "bicolor":
            for i, j in bond_pairs:
                pi, pj = positions[i], positions[j]
                pm = (pi + pj) * 0.5
                _create_cylinder_mesh(
                    pi,
                    pm,
                    b_rad,
                    elem_materials[(symbols[i], round(alphas[i], 3), atom_colors[i])],
                    f"bond_{i}_{j}_a",
                    col,
                    segments=b_seg,
                    smooth=b_smooth,
                )
                _create_cylinder_mesh(
                    pm,
                    pj,
                    b_rad,
                    elem_materials[(symbols[j], round(alphas[j], 3), atom_colors[j])],
                    f"bond_{i}_{j}_b",
                    col,
                    segments=b_seg,
                    smooth=b_smooth,
                )
        else:
            single_bond_mats: dict[float, bpy.types.Material] = {}
            for i, j in bond_pairs:
                b_alpha = round(min(alphas[i], alphas[j]), 3)
                bond_mat = single_bond_mats.get(b_alpha)
                if bond_mat is None:
                    bond_mat = make_material(
                        f"mat_bond_{int(round(b_alpha * 1000))}",
                        str(bond_cfg.get("color", "#9CA3AF")),
                        roughness=float(bond_cfg.get("roughness", 0.45)),
                        specular=float(bond_cfg.get("specular", 0.25)),
                        alpha=b_alpha,
                    )
                    single_bond_mats[b_alpha] = bond_mat
                _create_cylinder_mesh(
                    positions[i],
                    positions[j],
                    b_rad,
                    bond_mat,
                    f"bond_{i}_{j}",
                    col,
                    segments=b_seg,
                    smooth=b_smooth,
                )

    raw_dashed = structure.get("dashed_bonds", [])
    if isinstance(raw_dashed, list) and raw_dashed:
        dashed_hex = str(structure.get("dashed_color", "#38BDF8"))
        dashed_mat = make_material("mat_dashed_contact", dashed_hex, roughness=0.35, specular=0.2, alpha=0.9)
        d_rad = max(0.03, b_rad * 0.52)
        n_dashes = 4
        for pair in raw_dashed:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            i_idx, j_idx = int(pair[0]), int(pair[1])
            if not (0 <= i_idx < len(positions) and 0 <= j_idx < len(positions)):
                continue
            p0, p1 = positions[i_idx], positions[j_idx]
            dv = p1 - p0
            if dv.length < 1e-4:
                continue
            for seg_i in range(n_dashes):
                t0 = (seg_i + 0.12) / n_dashes
                t1 = (seg_i + 0.65) / n_dashes
                _create_cylinder_mesh(
                    p0 + dv * t0,
                    p0 + dv * t1,
                    d_rad,
                    dashed_mat,
                    f"dashed_{i_idx}_{j_idx}_{seg_i}",
                    col,
                    segments=10,
                    smooth=True,
                )

    # Periodic cell wireframe (supports general triclinic vectors)
    cell_cfg = style.get("cell", {})
    cell_enabled = cell_cfg.get("enabled", True) if show_cell_override is None else bool(show_cell_override)
    if has_cell and cell_enabled:
        box_mat = make_material(
            "mat_cell",
            str(cell_cfg.get("color", "#111111")),
            roughness=float(cell_cfg.get("roughness", 0.5)),
            specular=float(cell_cfg.get("specular", 0.1)),
        )
        c_rad = float(cell_cfg.get("radius", 0.06))
        c_seg = int(cell_cfg.get("segments", 14))
        c_smooth = bool(cell_cfg.get("smooth", True))
        corners = compute_cell_corners(origin, cell_vecs)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for idx, (a_idx, b_idx) in enumerate(edges):
            _create_cylinder_mesh(
                corners[a_idx],
                corners[b_idx],
                c_rad,
                box_mat,
                f"cell_{idx}",
                col,
                segments=c_seg,
                smooth=c_smooth,
            )

        if bool(structure.get("show_cell_vectors", False)):
            vec_specs = (
                ("a", cell_vecs[0], "#EF4444"),
                ("b", cell_vecs[1], "#22C55E"),
                ("c", cell_vecs[2], "#3B82F6"),
            )
            import bmesh

            v_rad = max(0.08, c_rad * 1.65)
            for v_lbl, v_vec, v_hex in vec_specs:
                v_len = v_vec.length
                if v_len < 1e-4:
                    continue
                tip_len = min(0.42 * v_len, 2.8)
                v_dir = v_vec.normalized()
                shaft_end = origin + v_dir * max(0.2, tip_len - 0.35)
                cone_tip = origin + v_dir * tip_len
                v_mat = make_material(f"mat_cellvec_{v_lbl}", v_hex, roughness=0.35, specular=0.25)
                _create_cylinder_mesh(origin, shaft_end, v_rad, v_mat, f"cellvec_{v_lbl}_shaft", col, segments=14, smooth=True)
                cone_mesh = bpy.data.meshes.new(f"cellvec_{v_lbl}_head_mesh")
                bm_c = bmesh.new()
                bmesh.ops.create_cone(
                    bm_c,
                    cap_ends=True,
                    cap_tris=False,
                    segments=14,
                    radius1=v_rad * 2.2,
                    radius2=0.0,
                    depth=0.45,
                )
                bm_c.to_mesh(cone_mesh)
                bm_c.free()
                cone_mesh.materials.append(v_mat)
                c_obj = bpy.data.objects.new(f"cellvec_{v_lbl}_head", cone_mesh)
                c_obj.location = (shaft_end + cone_tip) * 0.5
                c_obj.rotation_mode = "QUATERNION"
                c_obj.rotation_quaternion = v_dir.to_track_quat("Z", "Y")
                col.objects.link(c_obj)

    raw_poly = structure.get("polyhedra", [])
    if isinstance(raw_poly, list) and raw_poly:
        poly_mats: dict[tuple[str, float], bpy.types.Material] = {}
        for p_idx, poly in enumerate(raw_poly):
            verts = poly.get("vertices", [])
            faces = poly.get("faces", [])
            if len(verts) < 4 or not faces:
                continue
            p_hex = str(poly.get("color", "#38BDF8"))
            p_alpha = round(max(0.05, min(0.95, float(poly.get("alpha", 0.36)))), 3)
            p_mat = poly_mats.get((p_hex, p_alpha))
            if p_mat is None:
                p_mat = make_material(
                    f"mat_poly_{p_hex.lstrip('#')}_{int(p_alpha * 1000)}",
                    p_hex,
                    roughness=0.25,
                    specular=0.4,
                    alpha=p_alpha,
                )
                poly_mats[(p_hex, p_alpha)] = p_mat
            pmesh = bpy.data.meshes.new(f"poly_{p_idx}_mesh")
            pmesh.from_pydata([tuple(float(x) for x in v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
            pmesh.update()
            pmesh.materials.append(p_mat)
            p_obj = bpy.data.objects.new(f"poly_{p_idx}", pmesh)
            col.objects.link(p_obj)

    # Framing & camera
    radii_list = [elem_radii_cache[s] for s in symbols]
    target, pts_frame = compute_framing_points(
        positions=positions,
        radii=radii_list,
        has_cell=has_cell,
        origin=origin,
        cell_vecs=cell_vecs,
        crop_z=crop_z,
    )
    framing_data = {
        "target": target,
        "pts_frame": pts_frame,
        "cell_vecs": cell_vecs if has_cell else None,
        "n_atoms": len(symbols),
        "n_bonds": len(bond_pairs),
    }

    cam_data = bpy.data.cameras.new("OrthoCam")
    cam_data.type = str(cam_cfg.get("type", "ORTHO"))
    cam_obj = bpy.data.objects.new("OrthoCam", cam_data)
    col.objects.link(cam_obj)
    scene.camera = cam_obj

    apply_camera_view(
        scene=scene,
        cam_obj=cam_obj,
        view=view,
        framing_data=framing_data,
        style=style,
        fit_override=fit_override,
    )
    setup_lighting(col, target=target, style=style)
    return framing_data


def _parse_crop_z(val: str | None) -> tuple[float, float] | None:
    if not val or val.lower() == "none":
        return None
    if val.lower() == "auto":
        return (2.0, 4.0)
    parts = [float(x.strip()) for x in val.split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"Invalid --crop-z value: {val!r}")


def _resolve_view_arg(view_val: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(view_val, dict):
        return view_val
    view_str = str(view_val)
    if view_str.lower() in VIEW_PRESETS:
        return view_str.lower()
    view_path = Path(view_str)
    if view_path.is_file():
        return json.loads(view_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Unknown preset or missing view file: {view_str}")


def _apply_fast_preview_cycles_settings(scene: bpy.types.Scene) -> None:
    """Tune Cycles bounce caps, adaptive sampling, and AA filter for fast crisp previews."""
    if scene.render.engine != "CYCLES":
        return
    c = scene.cycles
    c.max_bounces = 2
    c.diffuse_bounces = 1
    c.glossy_bounces = 1
    c.transmission_bounces = 2
    c.transparent_max_bounces = 4
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.05
    c.filter_width = 1.1
    if hasattr(c, "denoising_prefilter"):
        try:
            c.denoising_prefilter = "FAST"
        except Exception:
            pass


def run_preview_server() -> None:
    """Keep Blender warm and render preview PNGs on demand from JSON-line commands on stdin."""
    current_scene_key: str | None = None
    cached_framing: dict[str, Any] | None = None
    cached_style: dict[str, Any] | None = None
    print("PREVIEW_READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        seq: Any = "?"
        try:
            cmd = json.loads(line)
            action = cmd.get("action")
            if action == "quit":
                break
            if action != "render":
                continue

            seq = cmd["seq"]
            scene_key = str(cmd["scene_key"])
            style = json.loads(Path(cmd["style"]).read_text(encoding="utf-8"))
            if cmd.get("samples") is not None:
                style.setdefault("render", {})["samples"] = int(cmd["samples"])
            # Smooth instanced prototype meshes with minimal overhead
            style.setdefault("atom_mesh", {})
            style["atom_mesh"]["segments"] = 24
            style["atom_mesh"]["ring_count"] = 14
            style.setdefault("bonds", {})["segments"] = 12
            style.setdefault("cell", {})["segments"] = 10
            res = tuple(int(x) for x in cmd["resolution"])
            view_arg = _resolve_view_arg(cmd["view"])

            if scene_key != current_scene_key:
                structure = json.loads(Path(cmd["structure"]).read_text(encoding="utf-8"))
                cached_framing = build_scene(
                    structure=structure,
                    style=style,
                    view=view_arg,
                    crop_z=_parse_crop_z(cmd.get("crop_z")),
                    resolution=res,
                    bonds_mode_override=cmd.get("bonds"),
                    reset_factory=True,
                    fit_override=bool(cmd.get("fit_view", False)),
                    radius_scale=float(cmd.get("radius_scale", 1.0)),
                    bond_color_override=cmd.get("bond_color"),
                    show_cell_override=False if cmd.get("no_cell", False) else None,
                )
                _apply_fast_preview_cycles_settings(bpy.context.scene)
                current_scene_key = scene_key
                cached_style = style
            else:
                scene = bpy.context.scene
                scene.render.resolution_x = res[0]
                scene.render.resolution_y = res[1]
                scene.render.resolution_percentage = 100
                if cmd.get("samples") is not None and scene.render.engine == "CYCLES":
                    scene.cycles.samples = int(cmd["samples"])
                _apply_fast_preview_cycles_settings(scene)
                apply_camera_view(
                    scene=scene,
                    cam_obj=scene.camera,
                    view=view_arg,
                    framing_data=cached_framing,
                    style=cached_style or style,
                    fit_override=bool(cmd.get("fit_view", False)),
                )

            out_path = Path(cmd["out"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(out_path.resolve())
            bpy.ops.render.render(write_still=True)
            print(f"PREVIEW_DONE {seq} {out_path}", flush=True)
        except Exception as exc:
            print(f"PREVIEW_ERR {seq} {exc}", flush=True)


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected '--' followed by render arguments.")
    args_list = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Render neutral structure JSON inside Blender.")
    parser.add_argument("--structure", type=Path, default=None)
    parser.add_argument("--style", type=Path, default=None)
    parser.add_argument("--preview-server", action="store_true", help="Read JSON preview render commands on stdin")
    parser.add_argument("--view", default="iso", help="Preset name or path to view.json")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--save-blend", type=Path, default=None, help="Optional output .blend path")
    parser.add_argument("--crop-z", default=None, help="auto or bot,top margin in Angstrom")
    parser.add_argument("--resolution", nargs=2, type=int, default=None, metavar=("W", "H"))
    parser.add_argument("--samples", type=int, default=None, help="Override Cycles sample count")
    parser.add_argument("--bonds", choices=["table", "auto", "none"], default=None)
    parser.add_argument("--bond-color", choices=["style", "single", "bicolor"], default=None)
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Scale factor for atom sphere radii")
    parser.add_argument("--no-cell", action="store_true", help="Hide periodic cell wireframe")
    parser.add_argument("--fit-view", action="store_true", help="Re-fit view.json orientation to structure")
    args = parser.parse_args(args_list)

    if args.preview_server:
        run_preview_server()
        return

    if args.structure is None or args.style is None:
        parser.error("--structure and --style are required unless --preview-server is set.")

    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    style = json.loads(args.style.read_text(encoding="utf-8"))
    if args.samples is not None:
        style.setdefault("render", {})["samples"] = int(args.samples)

    view_arg = _resolve_view_arg(args.view)

    build_scene(
        structure=structure,
        style=style,
        view=view_arg,
        crop_z=_parse_crop_z(args.crop_z),
        resolution=tuple(args.resolution) if args.resolution else None,
        bonds_mode_override=args.bonds,
        reset_factory=True,
        fit_override=args.fit_view,
        radius_scale=float(args.radius_scale),
        bond_color_override=args.bond_color,
        show_cell_override=False if args.no_cell else None,
    )

    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.resolve()))
        print(f"Saved blend -> {args.save_blend}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(args.out.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"Rendered -> {args.out}")


if __name__ == "__main__":
    main()
