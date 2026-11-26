"""Atomic Blender Studio — Interactive N-panel Blender addon.

Uses unique module/operator identifiers (``atomic_blender_studio`` / ``atomic_studio.*``)
to avoid any collision with Blender's built-in ``io_mesh_atomic`` ("Atomic Blender (PDB/XYZ)").
"""

from __future__ import annotations

bl_info = {
    "name": "Atomic Blender Studio (Publication Renderer)",
    "author": "polar2251",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Atomic Studio",
    "description": "Load styled atomic structures, frame interactively, save/reuse view.json, and render.",
    "category": "Import-Export",
}

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, IntVectorProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector

# Locate bundled scene_builder (when installed via `install-addon`) or repo package
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


def _default_style_path() -> str:
    local_style = _ADDON_DIR / "styles" / "default.json"
    if local_style.is_file():
        return str(local_style)
    cfg = _load_addon_config()
    repo_root = cfg.get("repo_root")
    if repo_root:
        p = Path(repo_root) / "styles" / "default.json"
        if p.is_file():
            return str(p)
    repo_style = _ADDON_DIR.parent.parent / "styles" / "default.json"
    if repo_style.is_file():
        return str(repo_style)
    return ""


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
        import os

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


def _store_framing_metadata(scene: bpy.types.Scene, framing_data: dict[str, Any]) -> None:
    """Persist lightweight framing points on the scene so preset switching works anytime."""
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


class AtomicStudioProperties(PropertyGroup):
    structure_path: StringProperty(
        name="Structure",
        description="Structure file (.json, .cif, POSCAR, .xyz, LAMMPS .data)",
        subtype="FILE_PATH",
    )
    style_path: StringProperty(
        name="Style JSON",
        description="Path to style JSON file",
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
    bl_description = "Load atomic structure with the selected style and preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.atomic_studio
        sb = _import_scene_builder()
        struct_path = Path(bpy.path.abspath(props.structure_path))
        if not struct_path.is_file():
            self.report({"ERROR"}, f"Structure file not found: {struct_path}")
            return {"CANCELLED"}

        style_file = Path(bpy.path.abspath(props.style_path)) if props.style_path else Path(_default_style_path())
        if not style_file.is_file():
            self.report({"ERROR"}, f"Style JSON not found: {style_file}")
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

        style.setdefault("render", {})["samples"] = int(props.samples)
        crop_z = (props.crop_z_bot, props.crop_z_top) if props.crop_z_auto else None
        bonds_override = None if props.bonds_mode == "STYLE" else props.bonds_mode

        framing = sb.build_scene(
            structure=structure,
            style=style,
            view=props.preset,
            crop_z=crop_z,
            resolution=(int(props.res_x), int(props.res_y)),
            bonds_mode_override=bonds_override,
            reset_factory=False,
        )
        _store_framing_metadata(context.scene, framing)
        self.report({"INFO"}, f"Loaded {len(structure['symbols'])} atoms from {struct_path.name}")
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
        if props.style_path:
            sp = Path(bpy.path.abspath(props.style_path))
            if sp.is_file():
                style = json.loads(sp.read_text(encoding="utf-8"))
        scene.render.resolution_x = int(props.res_x)
        scene.render.resolution_y = int(props.res_y)
        framing = _load_framing_metadata(scene)
        sb.apply_camera_view(scene, cam, props.preset, framing_data=framing, style=style)
        self.report({"INFO"}, f"Applied preset '{props.preset}'")
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

        # Set exact camera world transform from the 3D Viewport view matrix
        cam.matrix_world = rv3d.view_matrix.inverted()
        cam.rotation_mode = "QUATERNION"

        # Match visible orthographic framing to the current viewport zoom distance
        if cam.data.type == "ORTHO":
            sensor_fit = cam.data.sensor_fit
            sensor_size = cam.data.sensor_height if sensor_fit == "VERTICAL" else cam.data.sensor_width
            # In Blender's VIEW_3D, ortho half-extent scales with view_distance * 36 / (2 * lens)
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
        props = context.scene.atomic_studio

        box_in = layout.box()
        box_in.label(text="Structure & Style", icon="MESH_UVSPHERE")
        box_in.prop(props, "structure_path")
        box_in.prop(props, "style_path")
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
        box_in.prop(props, "bonds_mode")
        box_in.operator("atomic_studio.load_structure", icon="IMPORT")

        box_cam = layout.box()
        box_cam.label(text="View & Framing", icon="CAMERA_DATA")
        row_p = box_cam.row(align=True)
        row_p.prop(props, "preset", text="")
        row_p.operator("atomic_studio.apply_preset", text="Apply")
        box_cam.operator("atomic_studio.camera_to_view", icon="OUTLINER_OB_CAMERA")
        box_cam.prop(props, "view_json_path")
        row_v = box_cam.row(align=True)
        row_v.operator("atomic_studio.save_view", icon="FILE_TICK")
        row_v.operator("atomic_studio.load_view", icon="FILE_FOLDER")

        box_ren = layout.box()
        box_ren.label(text="Render", icon="RENDER_STILL")
        row_r = box_ren.row(align=True)
        row_r.prop(props, "res_x")
        row_r.prop(props, "res_y")
        box_ren.prop(props, "samples")
        box_ren.prop(props, "render_out_path")
        box_ren.operator("atomic_studio.render", icon="RENDER_STILL")


_CLASSES = (
    AtomicStudioProperties,
    ATOMIC_STUDIO_OT_load_structure,
    ATOMIC_STUDIO_OT_apply_preset,
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
