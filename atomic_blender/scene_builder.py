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
        VIEW_PRESETS,
        resolve_element_color,
        resolve_element_radius,
    )
except ImportError:
    if __package__:
        from .elements import (  # type: ignore[no-redef]
            COVALENT_RADII,
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
) -> bpy.types.Material:
    """Create a Principled BSDF material compatible with Blender 3.x–5.x."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None and hasattr(mat, "use_nodes"):
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = srgb_hex_to_linear_rgba(hex_color)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(specular)
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(specular)
    return mat


def _ensure_collection(scene: bpy.types.Scene, clear_existing: bool) -> bpy.types.Collection:
    """Return the dedicated collection, optionally removing any previously built objects."""
    col = bpy.data.collections.get(COLLECTION_NAME)
    if col is not None and clear_existing:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    if clear_existing:
        # Remove Blender's default startup Cube/Light/Camera (which live in 'Collection')
        for default_name in ("Cube", "Light", "Camera"):
            obj = bpy.data.objects.get(default_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
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
    """Return camera quaternion for a named view preset (``iso``, ``side-oblique``, ``a``, ``b``, ``c``)."""
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
        return _direction_to_quaternion(c.normalized(), b.normalized())
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
            pts_frame = [Vector((p.x, p.y, zmin)) for p in base] + [
                Vector((p.x, p.y, zmax)) for p in base
            ]
            cx = sum(p.x for p in base) * 0.25
            cy = sum(p.y for p in base) * 0.25
            target = Vector((cx, cy, 0.5 * (zmin + zmax)))
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
    if bg is not None:
        wcol = world_cfg.get("color", [0.85, 0.85, 0.85, 1.0])
        bg.inputs["Color"].default_value = tuple(float(c) for c in wcol)
        bg.inputs["Strength"].default_value = float(world_cfg.get("strength", 0.6))


def setup_lighting(
    col: bpy.types.Collection,
    target: Vector,
    style: dict[str, Any],
) -> None:
    """Create Sun key light and Area fill light aimed at ``target``."""
    light_cfg = style.get("lighting", {})
    sun_cfg = light_cfg.get("sun", {})
    sun_data = bpy.data.lights.new("AtomicSun", type="SUN")
    sun_data.energy = float(sun_cfg.get("energy", 3.0))
    sun_obj = bpy.data.objects.new("AtomicSun", sun_data)
    rot_deg = sun_cfg.get("rotation_deg", [35.0, -20.0, 30.0])
    sun_obj.rotation_euler = tuple(math.radians(float(a)) for a in rot_deg)
    col.objects.link(sun_obj)

    fill_cfg = light_cfg.get("fill", {})
    fill_type = str(fill_cfg.get("type", "AREA"))
    area_data = bpy.data.lights.new("AtomicFill", type=fill_type)
    area_data.energy = float(fill_cfg.get("energy", 4000.0))
    if fill_type == "AREA":
        area_data.size = float(fill_cfg.get("size", 40.0))
    area_obj = bpy.data.objects.new("AtomicFill", area_data)
    az = math.radians(float(fill_cfg.get("azimuth_deg", -40.0)))
    el = math.radians(float(fill_cfg.get("elevation_deg", 25.0)))
    fill_dist = float(fill_cfg.get("distance", 45.0))
    fill_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))).normalized()
    area_obj.location = target + fill_dir * fill_dist
    area_obj.rotation_mode = "QUATERNION"
    area_obj.rotation_quaternion = (-fill_dir).to_track_quat("-Z", "Y")
    col.objects.link(area_obj)


def build_scene(
    structure: dict[str, Any],
    style: dict[str, Any],
    view: str | dict[str, Any] = "iso",
    crop_z: tuple[float, float] | None = None,
    resolution: tuple[int, int] | None = None,
    bonds_mode_override: str | None = None,
    reset_factory: bool = True,
    fit_override: bool = False,
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
    has_cell = bool(structure.get("has_cell", True))
    origin = Vector(structure.get("origin", [0.0, 0.0, 0.0]))
    cell_vecs = [Vector(v) for v in structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])]

    # Atom meshes instanced per element
    mat_cfg = style.get("material", {})
    roughness = float(mat_cfg.get("roughness", 0.35))
    specular = float(mat_cfg.get("specular", 0.4))
    mesh_cfg = style.get("atom_mesh", {})
    segments = int(mesh_cfg.get("segments", 24))
    ring_count = int(mesh_cfg.get("ring_count", 12))
    smooth = bool(mesh_cfg.get("smooth", True))

    proto_meshes: dict[str, bpy.types.Mesh] = {}
    elem_radii_cache: dict[str, float] = {}
    for elem in sorted(set(symbols)):
        rad = resolve_element_radius(elem, style)
        elem_radii_cache[elem] = rad
        mat = make_material(
            f"mat_{elem}",
            resolve_element_color(elem, style),
            roughness=roughness,
            specular=specular,
        )
        proto_meshes[elem] = _create_uv_sphere_mesh(
            name=f"proto_{elem}",
            radius=rad,
            segments=segments,
            ring_count=ring_count,
            smooth=smooth,
            mat=mat,
        )

    for idx, (elem, pos) in enumerate(zip(symbols, positions)):
        obj = bpy.data.objects.new(f"atom_{idx}_{elem}", proto_meshes[elem])
        obj.location = pos
        col.objects.link(obj)

    # Bonds
    bond_cfg = dict(style.get("bonds", {}))
    if bonds_mode_override is not None:
        bond_cfg["mode"] = bonds_mode_override
        bond_cfg["enabled"] = bonds_mode_override != "none"
    bond_pairs = _find_bonds_spatial(symbols, positions, bond_cfg)
    if bond_pairs:
        bond_mat = make_material(
            "mat_bond",
            str(bond_cfg.get("color", "#9CA3AF")),
            roughness=float(bond_cfg.get("roughness", 0.45)),
            specular=float(bond_cfg.get("specular", 0.25)),
        )
        b_rad = float(bond_cfg.get("radius", 0.12))
        b_seg = int(bond_cfg.get("segments", 14))
        b_smooth = bool(bond_cfg.get("smooth", True))
        for i, j in bond_pairs:
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

    # Periodic cell wireframe (supports general triclinic vectors)
    cell_cfg = style.get("cell", {})
    if has_cell and cell_cfg.get("enabled", True):
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


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected '--' followed by render arguments.")
    args_list = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Render neutral structure JSON inside Blender.")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--view", default="iso", help="Preset name or path to view.json")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--save-blend", type=Path, default=None, help="Optional output .blend path")
    parser.add_argument("--crop-z", default=None, help="auto or bot,top margin in Angstrom")
    parser.add_argument("--resolution", nargs=2, type=int, default=None, metavar=("W", "H"))
    parser.add_argument("--samples", type=int, default=None, help="Override Cycles sample count")
    parser.add_argument("--bonds", choices=["table", "auto", "none"], default=None)
    parser.add_argument("--fit-view", action="store_true", help="Re-fit view.json orientation to structure")
    args = parser.parse_args(args_list)

    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    style = json.loads(args.style.read_text(encoding="utf-8"))
    if args.samples is not None:
        style.setdefault("render", {})["samples"] = int(args.samples)

    view_arg: str | dict[str, Any]
    view_path = Path(args.view)
    if args.view.lower() in VIEW_PRESETS:
        view_arg = args.view.lower()
    elif view_path.is_file():
        view_arg = json.loads(view_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Unknown preset or missing view file: {args.view}")

    build_scene(
        structure=structure,
        style=style,
        view=view_arg,
        crop_z=_parse_crop_z(args.crop_z),
        resolution=tuple(args.resolution) if args.resolution else None,
        bonds_mode_override=args.bonds,
        reset_factory=True,
        fit_override=args.fit_view,
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
