"""Atomic Blender Studio — Interactive N-panel Blender addon.

Uses unique module/operator identifiers (``atomic_blender_studio`` / ``atomic_studio.*``)
to avoid any collision with Blender's built-in ``io_mesh_atomic`` ("Atomic Blender (PDB/XYZ)").
"""

from __future__ import annotations

bl_info = {
    "name": "Atomic Blender Studio (Publication Renderer)",
    "author": "polar2251",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Atomic Studio",
    "description": "Load styled atomic structures, adjust representations, frame interactively, and render.",
    "category": "Import-Export",
}

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, IntVectorProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Quaternion, Vector

_ADDON_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _ADDON_DIR / "config.json"


def _load_addon_config() -> dict[str, str]:
    if _CONFIG_PATH.is_file():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _import_scene_builder():
    if (_ADDON_DIR / "scene_builder.py").is_file():
        from . import scene_builder  # type: ignore
        return scene_builder
    cfg = _load_addon_config()
    repo_root = cfg.get("repo_root")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    repo_candidate = _ADDON_DIR.parent.parent
    if (repo_candidate / "atomic_blender" / "scene_builder.py").is_file():
        if str(repo_candidate) not in sys.path:
            sys.path.insert(0, str(repo_candidate))
    from atomic_blender import scene_builder
    return scene_builder


def _resolve_style_file(style_preset: str, custom_path: str) -> Path | None:
    if style_preset == "CUSTOM":
        p = Path(bpy.path.abspath(custom_path)) if custom_path else None
        return p if p and p.is_file() else None

    fname = f"{style_preset}.json"
    candidates = [
        _ADDON_DIR / "styles" / fname,
        _ADDON_DIR.parent.parent / "styles" / fname,
        _ADDON_DIR.parent.parent / "atomic_blender" / "styles" / fname,
    ]
    cfg = _load_addon_config()
    repo_root = cfg.get("repo_root")
    if repo_root:
        candidates.extend([
            Path(repo_root) / "styles" / fname,
            Path(repo_root) / "atomic_blender" / "styles" / fname,
        ])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _default_style_path() -> str:
    found = _resolve_style_file("default", "")
    return str(found) if found else ""


def _default_python_exe() -> str:
    cfg = _load_addon_config()
    return cfg.get("python_exe", "python")


def _load_structure_any(
    filepath: Path,
    python_exe: str,
    type_map: str,
    replicate: tuple[int, int, int],
    wrap: bool,
) -> dict[str, Any]:
    """Load neutral JSON directly or invoke system Python ASE converter for any format."""
    if filepath.suffix.lower() == ".json" and replicate == (1, 1, 1):
        return json.loads(filepath.read_text(encoding="utf-8"))

    cfg = _load_addon_config()
    repo_root = cfg.get("repo_root") or str(_ADDON_DIR.parent.parent)
    with tempfile.TemporaryDirectory(prefix="atomic_studio_") as tmpdir:
        out_json = Path(tmpdir) / "structure.json"
        cmd = [
            python_exe or "python",
            "-m",
            "atomic_blender.convert",
            str(filepath),
            "-o",
            str(out_json),
            "--replicate",
            str(replicate[0]),
            str(replicate[1]),
            str(replicate[2]),
        ]
        if type_map.strip():
            cmd.extend(["--type-map", type_map.strip()])
        if not wrap:
            cmd.append("--no-wrap")
        env = os.environ.copy()
        if Path(repo_root).is_dir():
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{pp}" if pp else repo_root
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ASE conversion failed (python={python_exe}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return json.loads(out_json.read_text(encoding="utf-8"))


def _store_scene_metadata(
    scene: bpy.types.Scene,
    structure: dict[str, Any],
    framing_data: dict[str, Any],
) -> None:
    """Persist structure JSON and framing bounds on the scene for style/preset updates."""
    scene["atomic_studio_structure"] = json.dumps(structure)
    payload = {
        "target": [float(v) for v in framing_data["target"]],
        "pts_frame": [[float(v) for v in p] for p in framing_data["pts_frame"]],
        "cell_vecs": (
            [[float(v) for v in c] for c in framing_data["cell_vecs"]]
            if framing_data.get("cell_vecs")
            else None
        ),
    }
    scene["atomic_studio_framing"] = json.dumps(payload)

    cell = structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    lengths = [Vector(v).length for v in cell]
    n_atoms = int(framing_data.get("n_atoms", len(structure.get("symbols", []))))
    n_bonds = int(framing_data.get("n_bonds", 0))
    if structure.get("has_cell", True):
        cell_str = f"{lengths[0]:.1f}×{lengths[1]:.1f}×{lengths[2]:.1f} Å"
    else:
        cell_str = "no cell"
    scene["atomic_studio_summary"] = f"{n_atoms} atoms | {n_bonds} bonds | {cell_str}"


def _load_framing_metadata(scene: bpy.types.Scene) -> dict[str, Any] | None:
    raw = scene.get("atomic_studio_framing")
    if not raw:
        return None
    data = json.loads(raw)
    return {
        "target": Vector(data["target"]),
        "pts_frame": [Vector(p) for p in data["pts_frame"]],
        "cell_vecs": [Vector(c) for c in data["cell_vecs"]] if data.get("cell_vecs") else None,
    }


def _build_from_props(
    context: bpy.types.Context,
    structure: dict[str, Any],
    style: dict[str, Any],
    preserve_camera: bool = False,
) -> dict[str, Any]:
    scene = context.scene
    props = scene.atomic_studio
    sb = _import_scene_builder()

    saved_cam = None
    if preserve_camera and scene.camera is not None:
        cam = scene.camera
        cam.rotation_mode = "QUATERNION"
        saved_cam = {
            "type": cam.data.type,
            "location": Vector(cam.location),
            "quat": Quaternion(cam.rotation_quaternion),
            "ortho_scale": float(cam.data.ortho_scale),
            "lens": float(cam.data.lens),
            "clip_start": float(cam.data.clip_start),
            "clip_end": float(cam.data.clip_end),
        }

    style.setdefault("render", {})["samples"] = int(props.samples)
    crop_z = (props.crop_z_bot, props.crop_z_top) if props.crop_z_auto else None
    bonds_override = None if props.bonds_mode == "STYLE" else props.bonds_mode
    bond_color_override = None if props.bond_color == "STYLE" else props.bond_color

    framing = sb.build_scene(
        structure=structure,
        style=style,
        view=props.preset,
        crop_z=crop_z,
        resolution=(int(props.res_x), int(props.res_y)),
        bonds_mode_override=bonds_override,
        reset_factory=False,
        radius_scale=float(props.radius_scale),
        bond_color_override=bond_color_override,
        show_cell_override=bool(props.show_cell),
    )

    if saved_cam is not None and scene.camera is not None:
        cam = scene.camera
        cam.data.type = saved_cam["type"]
        cam.location = saved_cam["location"]
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = saved_cam["quat"]
        cam.data.ortho_scale = saved_cam["ortho_scale"]
        cam.data.lens = saved_cam["lens"]
        cam.data.clip_start = saved_cam["clip_start"]
        cam.data.clip_end = saved_cam["clip_end"]

    _store_scene_metadata(scene, structure, framing)
    return framing


class AtomicStudioProperties(PropertyGroup):
    structure_path: StringProperty(
        name="Structure",
        description="Structure file (.json, .cif, POSCAR, .xyz, LAMMPS .data)",
        subtype="FILE_PATH",
    )
    style_preset: EnumProperty(
        name="Style",
        items=[
            ("default", "Default (KANG-D3)", "Original flat Principled BSDF publication style"),
            ("studio", "Studio (SSS + AO + Rim)", "Subsurface scattering, AO contact shadows, metallic slabs, bicolor bonds"),
            ("matte", "Matte Clay (AO)", "Soft illustrative clay finish with strong AO and even ambient lighting"),
            ("ball_and_stick", "Ball & Stick", "Compact sphere radii with auto covalent bicolor bonds"),
            ("CUSTOM", "Custom JSON...", "Specify a custom style JSON file"),
        ],
        default="default",
    )
    style_path: StringProperty(
        name="Custom JSON",
        description="Path to custom style JSON file",
        subtype="FILE_PATH",
        default=_default_style_path(),
    )
    view_json_path: StringProperty(
        name="View JSON",
        description="Path to save/load view.json",
        subtype="FILE_PATH",
        default="//view.json",
    )
    render_out_path: StringProperty(
        name="Output PNG",
        description="Output PNG file for GUI Render button",
        subtype="FILE_PATH",
        default="//render.png",
    )
    python_exe: StringProperty(
        name="Python (ASE)",
        description="System Python executable with ASE installed (used for non-JSON files)",
        subtype="FILE_PATH",
        default=_default_python_exe(),
    )
    type_map: StringProperty(
        name="Type Map",
        description="Optional LAMMPS type map, e.g. '1:Al,2:C,3:H'",
        default="",
    )
    replicate: IntVectorProperty(
        name="Replicate",
        description="Supercell replication along a, b, c",
        size=3,
        default=(1, 1, 1),
        min=1,
        max=20,
    )
    wrap_atoms: BoolProperty(
        name="Wrap into Cell",
        description="Wrap periodic coordinates into the unit cell",
        default=True,
    )
    crop_z_auto: BoolProperty(
        name="Crop Z (Auto)",
        description="Frame slab structures tightly around atomic Z range",
        default=False,
    )
    crop_z_bot: FloatProperty(name="Z Margin Bottom", default=2.0, min=0.0, max=50.0)
    crop_z_top: FloatProperty(name="Z Margin Top", default=4.0, min=0.0, max=50.0)
    radius_scale: FloatProperty(
        name="Atom Scale",
        description="Multiplier for atomic sphere radii",
        default=1.0,
        min=0.3,
        max=2.0,
    )
    bonds_mode: EnumProperty(
        name="Bonds",
        items=[
            ("STYLE", "From Style", "Use bond mode defined in style JSON"),
            ("table", "Table", "Use style cutoff table"),
            ("auto", "Auto Covalent", "Use covalent radii tolerance"),
            ("none", "None", "Do not draw bonds"),
        ],
        default="STYLE",
    )
    bond_color: EnumProperty(
        name="Bond Color",
        items=[
            ("STYLE", "From Style", "Use bond color mode from style JSON"),
            ("bicolor", "Bicolor", "Split each bond into half-cylinders matching endpoint elements"),
            ("single", "Single Gray", "Uniform bond cylinder color"),
        ],
        default="STYLE",
    )
    show_cell: BoolProperty(
        name="Show Cell Box",
        description="Display the 12-edge periodic cell wireframe",
        default=True,
    )
    preset: EnumProperty(
        name="Preset",
        items=[
            ("iso", "Iso (35°, 20°)", "Isometric oblique view"),
            ("side-oblique", "Side Oblique (25°, 10°)", "Low-elevation side oblique view"),
            ("a", "Axis a", "View along lattice vector a"),
            ("b", "Axis b", "View along lattice vector b"),
            ("c", "Axis c", "View along lattice vector c"),
        ],
        default="iso",
    )
    res_x: IntProperty(name="Width", default=1400, min=64, max=16384)
    res_y: IntProperty(name="Height", default=1400, min=64, max=16384)
    samples: IntProperty(name="Samples", default=64, min=1, max=4096)


class ATOMIC_STUDIO_OT_load_structure(Operator):
    bl_idname = "atomic_studio.load_structure"
    bl_label = "Load Structure"
    bl_description = "Load atomic structure with the selected style and representation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomic_studio
        struct_path = Path(bpy.path.abspath(props.structure_path))
        if not struct_path.is_file():
            self.report({"ERROR"}, f"Structure file not found: {struct_path}")
            return {"CANCELLED"}

        style_file = _resolve_style_file(props.style_preset, props.style_path)
        if style_file is None:
            self.report({"ERROR"}, "Style JSON not found.")
            return {"CANCELLED"}

        try:
            structure = _load_structure_any(
                struct_path,
                python_exe=props.python_exe,
                type_map=props.type_map,
                replicate=tuple(props.replicate),
                wrap=props.wrap_atoms,
            )
            style = json.loads(style_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _build_from_props(context, structure, style, preserve_camera=False)
        self.report({"INFO"}, f"Loaded {len(structure['symbols'])} atoms from {struct_path.name}")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_update_style(Operator):
    bl_idname = "atomic_studio.update_style"
    bl_label = "Update Style / Representation"
    bl_description = "Rebuild loaded structure with the selected style, atom scale, bonds, and cell settings while keeping current camera"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.atomic_studio
        raw_struct = scene.get("atomic_studio_structure")
        if not raw_struct:
            self.report({"ERROR"}, "Load a structure first before updating style.")
            return {"CANCELLED"}

        style_file = _resolve_style_file(props.style_preset, props.style_path)
        if style_file is None:
            self.report({"ERROR"}, "Style JSON not found.")
            return {"CANCELLED"}

        structure = json.loads(raw_struct)
        style = json.loads(style_file.read_text(encoding="utf-8"))
        _build_from_props(context, structure, style, preserve_camera=True)
        self.report({"INFO"}, f"Applied style '{props.style_preset}' (camera preserved)")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_apply_preset(Operator):
    bl_idname = "atomic_studio.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Snap scene camera to the chosen view preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.atomic_studio
        cam = scene.camera
        if cam is None:
            self.report({"ERROR"}, "No active camera in scene.")
            return {"CANCELLED"}
        sb = _import_scene_builder()
        style = {}
        sf = _resolve_style_file(props.style_preset, props.style_path)
        if sf is not None:
            style = json.loads(sf.read_text(encoding="utf-8"))
        scene.render.resolution_x = int(props.res_x)
        scene.render.resolution_y = int(props.res_y)
        framing = _load_framing_metadata(scene)
        sb.apply_camera_view(scene, cam, props.preset, framing_data=framing, style=style)
        self.report({"INFO"}, f"Applied preset '{props.preset}'")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_quick_preset(Operator):
    bl_idname = "atomic_studio.quick_preset"
    bl_label = "Quick View Preset"
    bl_description = "Immediately snap camera to this preset"
    bl_options = {"REGISTER", "UNDO"}

    preset_name: StringProperty(default="iso")

    def execute(self, context):
        context.scene.atomic_studio.preset = self.preset_name
        return bpy.ops.atomic_studio.apply_preset()


class ATOMIC_STUDIO_OT_set_resolution(Operator):
    bl_idname = "atomic_studio.set_resolution"
    bl_label = "Set Resolution"
    bl_description = "Set render width and height"

    width: IntProperty(default=1400)
    height: IntProperty(default=1400)

    def execute(self, context):
        props = context.scene.atomic_studio
        props.res_x = int(self.width)
        props.res_y = int(self.height)
        context.scene.render.resolution_x = int(self.width)
        context.scene.render.resolution_y = int(self.height)
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_camera_to_view(Operator):
    bl_idname = "atomic_studio.camera_to_view"
    bl_label = "Camera to Current View"
    bl_description = "Align active camera (direction, position, and ortho zoom) to the current 3D Viewport"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        space = context.space_data
        if space is None or space.type != "VIEW_3D":
            self.report({"ERROR"}, "Run this from the 3D Viewport sidebar.")
            return {"CANCELLED"}

        rv3d = space.region_3d
        cam = scene.camera
        if cam is None:
            cam_data = bpy.data.cameras.new("OrthoCam")
            cam_data.type = "ORTHO"
            cam = bpy.data.objects.new("OrthoCam", cam_data)
            scene.collection.objects.link(cam)
            scene.camera = cam

        if rv3d.view_perspective == "CAMERA":
            self.report({"INFO"}, "Viewport is already locked to the active camera.")
            return {"FINISHED"}

        cam.matrix_world = rv3d.view_matrix.inverted()
        cam.rotation_mode = "QUATERNION"

        if cam.data.type == "ORTHO":
            sensor_fit = cam.data.sensor_fit
            sensor_size = cam.data.sensor_height if sensor_fit == "VERTICAL" else cam.data.sensor_width
            cam.data.ortho_scale = float(rv3d.view_distance * sensor_size / max(1e-3, space.lens))

        self.report({"INFO"}, f"Aligned camera '{cam.name}' to current viewport (ortho_scale={cam.data.ortho_scale:.2f})")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_save_view(Operator):
    bl_idname = "atomic_studio.save_view"
    bl_label = "Save View"
    bl_description = "Save active camera state and resolution to view.json for headless CLI reuse"

    def execute(self, context):
        scene = context.scene
        props = scene.atomic_studio
        cam = scene.camera
        if cam is None:
            self.report({"ERROR"}, "No active camera in scene.")
            return {"CANCELLED"}

        out_path = Path(bpy.path.abspath(props.view_json_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sb = _import_scene_builder()
        payload = sb.export_camera_view(scene, cam)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.report({"INFO"}, f"Saved view -> {out_path}")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_load_view(Operator):
    bl_idname = "atomic_studio.load_view"
    bl_label = "Load View"
    bl_description = "Apply a saved view.json to the active camera"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.atomic_studio
        cam = scene.camera
        if cam is None:
            self.report({"ERROR"}, "No active camera in scene.")
            return {"CANCELLED"}
        in_path = Path(bpy.path.abspath(props.view_json_path))
        if not in_path.is_file():
            self.report({"ERROR"}, f"View JSON not found: {in_path}")
            return {"CANCELLED"}
        sb = _import_scene_builder()
        view_data = json.loads(in_path.read_text(encoding="utf-8"))
        framing = _load_framing_metadata(scene)
        sb.apply_camera_view(scene, cam, view_data, framing_data=framing)
        props.res_x = int(scene.render.resolution_x)
        props.res_y = int(scene.render.resolution_y)
        self.report({"INFO"}, f"Loaded view <- {in_path.name}")
        return {"FINISHED"}


class ATOMIC_STUDIO_OT_render(Operator):
    bl_idname = "atomic_studio.render"
    bl_label = "Render"
    bl_description = "Render the current camera view to PNG"

    def execute(self, context):
        scene = context.scene
        props = scene.atomic_studio
        if scene.camera is None:
            self.report({"ERROR"}, "No active camera in scene.")
            return {"CANCELLED"}
        scene.render.resolution_x = int(props.res_x)
        scene.render.resolution_y = int(props.res_y)
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = int(props.samples)
        out_path = Path(bpy.path.abspath(props.render_out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        self.report({"INFO"}, f"Rendered -> {out_path}")
        return {"FINISHED"}


class ATOMIC_STUDIO_PT_panel(Panel):
    bl_label = "Atomic Blender Studio"
    bl_idname = "ATOMIC_STUDIO_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Atomic Studio"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.atomic_studio

        summary = scene.get("atomic_studio_summary")
        if summary:
            badge = layout.box()
            badge.label(text=str(summary), icon="INFO")

        # 1. Input Structure
        box_in = layout.box()
        box_in.label(text="Structure Input", icon="IMPORT")
        box_in.prop(props, "structure_path")
        box_in.prop(props, "python_exe")
        box_in.prop(props, "type_map")
        box_in.prop(props, "replicate")
        row = box_in.row(align=True)
        row.prop(props, "wrap_atoms")
        row.prop(props, "crop_z_auto")
        if props.crop_z_auto:
            row_z = box_in.row(align=True)
            row_z.prop(props, "crop_z_bot")
            row_z.prop(props, "crop_z_top")
        box_in.operator("atomic_studio.load_structure", icon="FILE_REFRESH")

        # 2. Style & Representation
        box_st = layout.box()
        box_st.label(text="Style & Representation", icon="SHADING_RENDERED")
        box_st.prop(props, "style_preset")
        if props.style_preset == "CUSTOM":
            box_st.prop(props, "style_path")
        box_st.prop(props, "radius_scale", slider=True)
        box_st.prop(props, "bonds_mode")
        box_st.prop(props, "bond_color")
        box_st.prop(props, "show_cell")
        if scene.get("atomic_studio_structure"):
            box_st.operator("atomic_studio.update_style", icon="MATERIAL")

        # 3. View & Framing
        box_cam = layout.box()
        box_cam.label(text="View & Framing", icon="CAMERA_DATA")
        row_q = box_cam.row(align=True)
        for label, pname in (("Iso", "iso"), ("Side", "side-oblique"), ("a", "a"), ("b", "b"), ("c", "c")):
            op = row_q.operator("atomic_studio.quick_preset", text=label)
            op.preset_name = pname
        box_cam.operator("atomic_studio.camera_to_view", icon="OUTLINER_OB_CAMERA")
        if context.space_data and context.space_data.type == "VIEW_3D":
            box_cam.prop(context.space_data, "lock_camera", text="Lock Camera to View")
        box_cam.prop(props, "view_json_path")
        row_v = box_cam.row(align=True)
        row_v.operator("atomic_studio.save_view", icon="FILE_TICK")
        row_v.operator("atomic_studio.load_view", icon="FILE_FOLDER")

        # 4. Render
        box_ren = layout.box()
        box_ren.label(text="Render", icon="RENDER_STILL")
        row_r = box_ren.row(align=True)
        row_r.prop(props, "res_x")
        row_r.prop(props, "res_y")
        row_rp = box_ren.row(align=True)
        for label, rw, rh in (("700²", 700, 700), ("1400²", 1400, 1400), ("Slab", 1400, 1200)):
            op = row_rp.operator("atomic_studio.set_resolution", text=label)
            op.width = rw
            op.height = rh
        box_ren.prop(props, "samples")
        box_ren.prop(props, "render_out_path")
        box_ren.operator("atomic_studio.render", icon="RENDER_STILL")


_CLASSES = (
    AtomicStudioProperties,
    ATOMIC_STUDIO_OT_load_structure,
    ATOMIC_STUDIO_OT_update_style,
    ATOMIC_STUDIO_OT_apply_preset,
    ATOMIC_STUDIO_OT_quick_preset,
    ATOMIC_STUDIO_OT_set_resolution,
    ATOMIC_STUDIO_OT_camera_to_view,
    ATOMIC_STUDIO_OT_save_view,
    ATOMIC_STUDIO_OT_load_view,
    ATOMIC_STUDIO_OT_render,
    ATOMIC_STUDIO_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.atomic_studio = bpy.props.PointerProperty(type=AtomicStudioProperties)


def unregister():
    if hasattr(bpy.types.Scene, "atomic_studio"):
        del bpy.types.Scene.atomic_studio
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
