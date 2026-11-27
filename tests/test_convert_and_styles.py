"""Unit and integration tests for atomic_blender (non-Blender and style/CLI checks)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ase import Atoms
from ase.build import bulk
import ase.io
import numpy as np

from atomic_blender.cli import (
    build_parser,
    cmd_install_addon,
    package_dir,
    package_root,
    resolve_style_path,
)
from atomic_blender.convert import atoms_to_neutral_dict, convert_file, parse_type_map
from atomic_blender.elements import (
    METAL_ELEMENTS,
    classify_dropped_file,
    resolve_element_color,
    resolve_element_radius,
)
from atomic_blender.gui import (
    GuiRenderConfig,
    build_render_cli_args,
    inspect_structure_summary,
    validate_gui_config,
)


class TestAtomicBlender(unittest.TestCase):
    def test_default_style_values(self) -> None:
        style_path = resolve_style_path("default")
        style = json.loads(style_path.read_text(encoding="utf-8"))
        self.assertEqual(style["elements"]["colors"]["Al"], "#A9C8E6")
        self.assertEqual(style["elements"]["colors"]["O"], "#D7382E")
        self.assertEqual(style["elements"]["radii"]["H"], 0.30)
        self.assertEqual(style["elements"]["radii"]["C"], 0.55)
        self.assertEqual(style["elements"]["radii"]["Al"], 1.10)
        self.assertEqual(style["elements"]["radii"]["Fe"], 1.05)
        self.assertEqual(style["material"]["roughness"], 0.35)
        self.assertEqual(style["material"]["specular"], 0.4)
        self.assertEqual(style["render"]["samples"], 64)
        self.assertTrue(style["render"]["film_transparent"])

    def test_element_fallback(self) -> None:
        style = json.loads(resolve_style_path("default").read_text(encoding="utf-8"))
        self.assertEqual(resolve_element_color("Pt", style), "#D0D0E0")
        self.assertEqual(resolve_element_radius("Pt", style), 1.05)
        self.assertEqual(resolve_element_radius("Al", style), 1.10)
        self.assertGreater(resolve_element_radius("Si", style), 0.5)

    def test_parse_type_map(self) -> None:
        self.assertEqual(parse_type_map("1:Al,2:C,3:H"), {1: 13, 2: 6, 3: 1})
        self.assertEqual(parse_type_map("O,H"), {1: 8, 2: 1})
        self.assertIsNone(parse_type_map(None))

    def test_triclinic_conversion_and_replication(self) -> None:
        cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.8, 0.0], [0.4, 0.2, 5.0]])
        atoms = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.0, 1.9, 2.5]], cell=cell, pbc=True)
        with tempfile.TemporaryDirectory() as tmp:
            cif_path = Path(tmp) / "tri.cif"
            json_path = Path(tmp) / "tri.json"
            ase.io.write(str(cif_path), atoms)
            payload = convert_file(cif_path, json_path, replicate=(2, 2, 1))
            self.assertEqual(len(payload["symbols"]), 8)
            self.assertTrue(payload["has_cell"])
            self.assertEqual(len(payload["cell"]), 3)
            # Check replicated lattice lengths doubled along a and b
            orig_a = np.linalg.norm(cell[0])
            rep_a = np.linalg.norm(payload["cell"][0])
            self.assertAlmostEqual(rep_a, 2.0 * orig_a, places=4)

    def test_lammps_type_map_override(self) -> None:
        atoms = bulk("Al", "fcc", a=4.05, cubic=True)
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "box.data"
            json_path = Path(tmp) / "box.json"
            ase.io.write(str(data_path), atoms, format="lammps-data")
            payload = convert_file(data_path, json_path, type_map="1:Ni")
            self.assertTrue(all(s == "Ni" for s in payload["symbols"]))

    def test_cli_parser_accepts_render_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "render",
            "in.cif",
            "--style",
            "default",
            "--view",
            "iso",
            "--out",
            "out.png",
            "--save-blend",
            "out.blend",
            "--replicate",
            "2",
            "2",
            "1",
            "--crop-z",
            "auto",
        ])
        self.assertEqual(args.replicate, [2, 2, 1])
        self.assertEqual(args.crop_z, "auto")
        self.assertEqual(args.view, "iso")

    def test_packaged_styles_and_addon_match_root(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        for style_name in ("default.json", "studio.json", "matte.json", "ball_and_stick.json"):
            pkg_style = package_dir() / "styles" / style_name
            root_style = repo_root / "styles" / style_name
            self.assertTrue(pkg_style.is_file(), f"Missing packaged style {style_name}")
            self.assertTrue(root_style.is_file(), f"Missing root style {style_name}")
            self.assertEqual(pkg_style.read_text(encoding="utf-8"), root_style.read_text(encoding="utf-8"))

        pkg_addon = package_dir() / "addon" / "atomic_blender_studio" / "__init__.py"
        root_addon = repo_root / "addon" / "atomic_blender_studio" / "__init__.py"
        self.assertTrue(pkg_addon.is_file())
        self.assertTrue(root_addon.is_file())
        self.assertEqual(pkg_addon.read_text(encoding="utf-8"), root_addon.read_text(encoding="utf-8"))

    def test_metal_classification(self) -> None:
        for sym in ("Al", "Pt", "Cu", "Fe", "Li", "Mg"):
            self.assertIn(sym, METAL_ELEMENTS)
        for sym in ("H", "C", "N", "O", "S", "Cl"):
            self.assertNotIn(sym, METAL_ELEMENTS)

    def test_install_addon_zip_and_dir(self) -> None:
        import zipfile

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "atomic_blender_studio.zip"
            args_zip = parser.parse_args(["install-addon", "--zip", str(zip_path)])
            self.assertEqual(cmd_install_addon(args_zip), 0)
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
            self.assertIn("atomic_blender_studio/__init__.py", names)
            self.assertIn("atomic_blender_studio/scene_builder.py", names)
            self.assertIn("atomic_blender_studio/elements.py", names)
            self.assertIn("atomic_blender_studio/styles/default.json", names)
            self.assertIn("atomic_blender_studio/config.json", names)

            addons_dir = Path(tmp) / "addons"
            args_dir = parser.parse_args([
                "install-addon",
                "--addons-dir",
                str(addons_dir),
                "--no-enable",
                "--blender",
                __file__,
            ])
            self.assertEqual(cmd_install_addon(args_dir), 0)
            self.assertTrue((addons_dir / "atomic_blender_studio" / "__init__.py").is_file())
            self.assertTrue((addons_dir / "atomic_blender_studio" / "styles" / "default.json").is_file())

    def test_gui_cli_subcommand_and_config_builder(self) -> None:
        parser = build_parser()
        gui_args = parser.parse_args(["gui", "example.cif"])
        self.assertEqual(gui_args.command, "gui")
        self.assertEqual(gui_args.input, Path("example.cif"))

        with tempfile.TemporaryDirectory() as tmp:
            struct_file = Path(tmp) / "mol.xyz"
            view_file = Path(tmp) / "view.json"
            struct_file.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
            view_file.write_text("{}", encoding="utf-8")

            cfg = GuiRenderConfig(
                input_path=str(struct_file),
                output_path=str(Path(tmp) / "out.png"),
                style="default",
                view_mode="file",
                view_file=str(view_file),
                fit_view=True,
                type_map="1:Al,2:C",
                replicate=(2, 2, 1),
                wrap=False,
                crop_z_mode="auto",
                bonds="auto",
                bond_color="bicolor",
                radius_scale=0.85,
                show_cell=False,
                resolution=(1200, 900),
                samples=32,
            )
            self.assertEqual(validate_gui_config(cfg), struct_file)
            summary = inspect_structure_summary(str(struct_file))
            self.assertIn("1 atoms", summary)
            cli_args = build_render_cli_args(cfg, save_blend=Path(tmp) / "scene.blend")
            self.assertEqual(cli_args[0], "render")
            self.assertIn("--fit-view", cli_args)
            self.assertIn("--no-wrap", cli_args)
            self.assertIn("--crop-z", cli_args)
            self.assertIn("auto", cli_args)
            self.assertIn("--type-map", cli_args)
            self.assertIn("1:Al,2:C", cli_args)
            self.assertIn("--bonds", cli_args)
            self.assertIn("--bond-color", cli_args)
            self.assertIn("bicolor", cli_args)
            self.assertIn("--radius-scale", cli_args)
            self.assertIn("--no-cell", cli_args)
            self.assertIn("--save-blend", cli_args)
            # Confirm build_parser parses generated CLI args cleanly
            parsed = parser.parse_args(cli_args)
            self.assertEqual(parsed.replicate, [2, 2, 1])
            self.assertEqual(parsed.resolution, [1200, 900])
            self.assertEqual(parsed.samples, 32)
            self.assertEqual(parsed.bond_color, "bicolor")
            self.assertAlmostEqual(parsed.radius_scale, 0.85)
            self.assertTrue(parsed.no_cell)

    def test_classify_dropped_file_and_gui_drop_handler(self) -> None:
        from atomic_blender.gui import AtomicBlenderApp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cif = tmp_path / "bulk.cif"
            cif.write_text("data_test\n", encoding="utf-8")
            struct_json = tmp_path / "struct.json"
            struct_json.write_text(json.dumps({"symbols": ["Cu"], "positions": [[0, 0, 0]]}), encoding="utf-8")
            view_json = tmp_path / "cam_view.json"
            view_json.write_text(json.dumps({"rotation_quaternion": [1, 0, 0, 0], "ortho_scale": 10.0}), encoding="utf-8")
            style_json = tmp_path / "my_style.json"
            style_json.write_text(json.dumps({"elements": {}, "material": {}}), encoding="utf-8")

            self.assertEqual(classify_dropped_file(cif), "structure")
            self.assertEqual(classify_dropped_file(struct_json), "structure")
            self.assertEqual(classify_dropped_file(view_json), "view")
            self.assertEqual(classify_dropped_file(style_json), "style")

            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.handle_dropped_files([str(cif), str(view_json), str(style_json)])
                self.assertEqual(app.var_input.get(), f"{cif.name} ({cif})")
                self.assertEqual(app._collect_config().input_path, str(cif))
                self.assertEqual(app.var_view_mode.get(), "file")
                self.assertEqual(app.var_view_file.get(), str(view_json))
                self.assertEqual(app.var_style.get(), str(style_json))

                # Verify queued drop dispatch inside a real mainloop tick (no WNDPROC GIL crash)
                app._queue.put(("drop", [str(struct_json)]))
                app.after(150, app.destroy)
                app.mainloop()
                self.assertEqual(app.var_input.get(), f"{struct_json.name} ({struct_json})")
            except Exception:
                try:
                    app.destroy()
                except Exception:
                    pass
                raise

    def test_viewport3d_and_interactive_camera_export(self) -> None:
        from atomic_blender.cli import resolve_style_path
        from atomic_blender.gui import AtomicBlenderApp
        from atomic_blender.viewport3d import ViewportScene, orbit_quat, preset_to_quat

        style = json.loads(resolve_style_path("studio").read_text(encoding="utf-8"))
        structure = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [True, True, True],
            "has_cell": True,
            "origin": [0.0, 0.0, 0.0],
        }
        vp = ViewportScene.from_neutral_dict(structure, style)
        self.assertEqual(len(vp.bond_i), 2)
        q_iso = preset_to_quat("iso", vp.cell_vecs)
        vp.fit_camera(q_iso, aspect=1.0)
        q_orb = orbit_quat(vp.quat, 0.25, -0.15)
        vp.quat = q_orb
        vp.zoom(0.8)
        view_dict = vp.to_view_dict(800, 600)
        self.assertEqual(view_dict["camera_type"], "ORTHO")
        self.assertEqual(view_dict["resolution"], [800, 600])
        self.assertFalse(view_dict["fit_to_scene"])
        self.assertAlmostEqual(view_dict["ortho_scale"], vp.ortho_scale)

        app = AtomicBlenderApp()
        app.withdraw()
        try:
            app.canvas.configure(width=600, height=600)
            app.update_idletasks()
            app._structure_cache = structure
            app._rebuild_viewport(reset_camera=True)
            self.assertIsNotNone(app._vp_scene)
            img_before = app._vp_scene.render_image(300, 300, 800, 800, interactive=False)
            self.assertEqual(img_before.size, (300, 300))

            app._vp_scene.orbit(0.4, 0.25)
            app._use_interactive_cam = True
            img_after = app._vp_scene.render_image(300, 300, 800, 800, interactive=False)
            self.assertNotEqual(img_before.tobytes(), img_after.tobytes())

            app._draw_3d_viewport(interactive=False)
            self.assertGreater(len(app.canvas.find_all()), 0)

            # Verify _redraw_preview runs cleanly without AttributeError
            app._redraw_preview()

            eff = app._effective_cfg_for_cli(app._collect_config())
            self.assertEqual(eff.view_mode, "file")
            self.assertTrue(Path(eff.view_file).is_file())
        finally:
            app.destroy()

    def test_h_outline_bonds_and_auto_preview_dispatch(self) -> None:
        from PIL import Image
        import numpy as np
        from atomic_blender.cli import resolve_style_path
        from atomic_blender.gui import AtomicBlenderApp, compute_preview_resolution
        from atomic_blender.viewport3d import ViewportScene

        self.assertEqual(compute_preview_resolution(1400, 1200, max_side=400), (400, 343))
        self.assertEqual(compute_preview_resolution(800, 1400, max_side=400), (229, 400))
        self.assertEqual(compute_preview_resolution(300, 250, max_side=400), (300, 250))

        style = json.loads(resolve_style_path("studio").read_text(encoding="utf-8"))
        structure = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [True, True, True],
            "has_cell": True,
            "origin": [0.0, 0.0, 0.0],
        }
        vp = ViewportScene.from_neutral_dict(structure, style)
        self.assertIn("H", vp._outline_symbols)
        h_sprite = vp._get_sprite("H", 40, 2)
        self.assertIsNotNone(h_sprite)
        arr = np.asarray(h_sprite)
        # Edge pixel near silhouette should be slightly darker than pure center highlight
        center_lum = float(arr[20, 20, :3].mean())
        edge_lum = float(arr[20, 1, :3].mean())
        self.assertLess(edge_lum, center_lum)

        class _RecorderWorker:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def request(self, payload: dict[str, object]) -> None:
                self.requests.append(payload)

            def stop(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            fake_blender = Path(tmp) / "blender_stub.exe"
            fake_blender.write_text("stub", encoding="utf-8")
            fake_png = Path(tmp) / "preview.png"
            Image.new("RGBA", (64, 64), (200, 220, 240, 255)).save(fake_png)

            app = AtomicBlenderApp()
            app.withdraw()
            try:
                recorder = _RecorderWorker()
                app._preview_worker.stop()
                app._preview_worker = recorder  # type: ignore[assignment]
                app.var_blender.set(str(fake_blender))
                app.var_res_w.set(1400)
                app.var_res_h.set(1200)
                app.var_samples.set(64)
                app._structure_cache = structure
                app._structure_version = 1
                app._rebuild_viewport(reset_camera=True, trigger_preview=True)

                self.assertEqual(len(recorder.requests), 1)
                req1 = recorder.requests[0]
                self.assertEqual(req1["resolution"], [640, 549])
                self.assertEqual(req1["samples"], 8)
                seq1 = int(req1["seq"])  # type: ignore[arg-type]

                # Matching preview_done flips canvas cleanly to rendered PNG
                app._queue.put(("preview_done", (seq1, fake_png)))
                app._poll_queue()
                self.assertEqual(app.var_canvas_mode.get(), "png")

                # Simulate drag motion returning immediately to 3D and invalidating old seq
                app._dragging = True
                app._drag_mode = "orbit"
                app._drag_last = (10, 10)
                ev = type("E", (), {"x": 25, "y": 18, "state": 0})()
                app._on_canvas_motion(ev)  # type: ignore[arg-type]
                self.assertEqual(app.var_canvas_mode.get(), "3d")
                self.assertGreater(app._preview_seq, seq1)

                # Stale preview_done for seq1 is ignored while moving / invalidated
                app._queue.put(("preview_done", (seq1, fake_png)))
                app._poll_queue()
                self.assertEqual(app.var_canvas_mode.get(), "3d")

                # Release mouse -> triggers new Blender preview seq
                app._on_canvas_mouse_up(ev)  # type: ignore[arg-type]
                self.assertEqual(len(recorder.requests), 2)
                req2 = recorder.requests[1]
                self.assertIsInstance(req2["view"], dict)
                self.assertEqual(req2["scene_key"], req1["scene_key"])
            finally:
                app.destroy()

    def test_display_types_draggable_axes_and_dark_theme(self) -> None:
        from PIL import Image
        import numpy as np
        from atomic_blender.cli import resolve_style_path
        from atomic_blender.gui import AtomicBlenderApp
        from atomic_blender.viewport3d import render_reference_axes_overlay

        img_with_lbl = Image.new("RGBA", (400, 400), (24, 26, 32, 255))
        cx, cy, hit_r = render_reference_axes_overlay(
            img_with_lbl,
            (1.0, 0.0, 0.0, 0.0),
            axes_pos=(0.25, 0.75),
            show_labels=True,
            dark_backdrop=True,
        )
        self.assertAlmostEqual(cx, 100.0, delta=1.0)
        self.assertAlmostEqual(cy, 300.0, delta=1.0)
        self.assertGreater(hit_r, 30.0)

        img_no_lbl = Image.new("RGBA", (400, 400), (24, 26, 32, 255))
        render_reference_axes_overlay(
            img_no_lbl,
            (1.0, 0.0, 0.0, 0.0),
            axes_pos=(0.25, 0.75),
            show_labels=False,
            dark_backdrop=True,
        )
        # x, y, z tip labels should produce distinct rendered pixels vs labels disabled
        self.assertNotEqual(img_with_lbl.tobytes(), img_no_lbl.tobytes())
        # Axes patch around (100, 300) must differ from pure background
        patch = np.asarray(img_with_lbl)[260:340, 60:140, :3]
        self.assertGreater(float(patch.std()), 5.0)

        style = json.loads(resolve_style_path("studio").read_text(encoding="utf-8"))
        structure = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [True, True, True],
            "has_cell": True,
            "origin": [0.0, 0.0, 0.0],
        }
        app = AtomicBlenderApp()
        app.withdraw()
        try:
            app.var_blender.set("")
            app.canvas.configure(width=600, height=600)
            app.update_idletasks()
            app._structure_cache = structure
            app._structure_version = 1
            app._rebuild_viewport(reset_camera=True, trigger_preview=False)
            assert app._vp_scene is not None

            # Verify default Display Types states
            self.assertTrue(app.var_disp_ball_stick.get())
            self.assertTrue(app.var_show_cell.get())
            self.assertTrue(app.var_show_axes.get())
            self.assertTrue(app.var_axes_labels.get())
            self.assertTrue(app.var_axes_fixed.get())
            self.assertEqual(app._axes_pos, (0.12, 0.86))

            # Switch to Licorice and toggle Atom Labels
            app._on_display_type_clicked("licorice")
            self.assertFalse(app.var_disp_ball_stick.get())
            self.assertTrue(app.var_disp_licorice.get())
            self.assertAlmostEqual(app.var_radius_scale.get(), 0.42, places=2)

            app._on_display_type_clicked("atom_labels")
            self.assertTrue(app.var_show_labels.get())
            self.assertTrue(app._vp_scene.show_labels)

            # Drag the Reference Axes center directly and verify unfixed position
            center = app._axes_canvas_center()
            self.assertIsNotNone(center)
            ax_x, ax_y, _ = center  # type: ignore[misc]
            ev_down = type("E", (), {"x": int(round(ax_x)), "y": int(round(ax_y)), "state": 0})()
            app._on_canvas_lbutton_down(ev_down)  # type: ignore[arg-type]
            self.assertEqual(app._drag_mode, "axes")
            self.assertFalse(app.var_axes_fixed.get())

            ev_move = type("E", (), {"x": 450, "y": 150, "state": 0})()
            app._on_canvas_motion(ev_move)  # type: ignore[arg-type]
            self.assertGreater(app._axes_pos[0], 0.5)
            self.assertLess(app._axes_pos[1], 0.5)

            ev_up = type("E", (), {"x": 450, "y": 150, "state": 0})()
            app._on_canvas_mouse_up(ev_up)  # type: ignore[arg-type]
            self.assertFalse(app._dragging)

            # Checking Fix (Default) snaps axes back to default (0.12, 0.86)
            app.var_axes_fixed.set(True)
            app._on_axes_fix_toggled()
            self.assertEqual(app._axes_pos, (0.12, 0.86))
            self.assertEqual(app._vp_scene.axes_pos, (0.12, 0.86))
        finally:
            app.destroy()

    def test_dark_widgets_type_map_visibility_and_nonperiodic_replicate(self) -> None:
        import time
        import numpy as np
        from atomic_blender.gui import AtomicBlenderApp, DarkCheckbutton, DarkSpinbox

        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "mol.xyz"
            xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
            lmp = Path(tmp) / "atoms.data"
            lmp.write_text("", encoding="utf-8")

            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.var_blender.set("")

                # Type Map is hidden for .xyz and automatically shown for LAMMPS .data
                app.var_input.set(str(xyz))
                self.assertEqual(app.lbl_type_map.winfo_manager(), "")
                app.var_input.set(str(lmp))
                self.assertEqual(app.lbl_type_map.winfo_manager(), "grid")

                # Verify View / Tool tabs, unified Display Types, and step camera
                self.assertEqual(app._active_tab, "view")
                self.assertEqual(app.f_rep.master, app.tab_view)
                self.assertEqual(app.f_geom.master, app.tab_tool)
                self.assertTrue(bool(app.f_rep_body.winfo_manager()))
                app._toggle_section(app.f_rep_body)
                self.assertFalse(bool(app.f_rep_body.winfo_manager()))
                app._toggle_section(app.f_rep_body)
                self.assertTrue(bool(app.f_rep_body.winfo_manager()))

                app._switch_tab("tool")
                self.assertEqual(app._active_tab, "tool")
                self.assertEqual(app.tab_tool.winfo_manager(), "pack")
                self.assertEqual(app.tab_view.winfo_manager(), "")
                app._switch_tab("view")
                self.assertEqual(app._active_tab, "view")
                self.assertEqual(app.tab_view.winfo_manager(), "pack")

                # Verify step rotation (default 45 deg + custom 30 deg) and step move (1.0 A + 2.5 A)
                app._structure_cache = {
                    "symbols": ["H", "H"],
                    "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "pbc": [False, False, False],
                    "has_cell": False,
                    "origin": [0.0, 0.0, 0.0],
                }
                app._rebuild_viewport(reset_camera=True, trigger_preview=False)
                assert app._vp_scene is not None
                self.assertAlmostEqual(app.var_rot_step.get(), 45.0)
                self.assertAlmostEqual(app.var_pan_step.get(), 1.0)
                q_before = app._vp_scene.quat
                t_before = app._vp_scene.target.copy()
                app._on_step_rotate(1.0, 0.0, 0.0)
                self.assertNotEqual(app._vp_scene.quat, q_before)
                q_after_yaw = app._vp_scene.quat
                app.var_rot_step.set(30.0)
                app._on_step_rotate(0.0, 0.0, 1.0)
                self.assertNotEqual(app._vp_scene.quat, q_after_yaw)
                app.var_pan_step.set(2.5)
                app._on_step_move(1.0, 0.0)
                self.assertGreater(float(np.linalg.norm(app._vp_scene.target - t_before)), 2.4)
                app._structure_cache = None

                # Non-periodic .xyz loads cleanly even when Replicate is set > 1
                app.var_input.set(str(xyz))
                app.var_rep_x.set(1)
                app.var_rep_y.set(2)
                app.var_rep_z.set(2)
                app._run_inspect_async()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and app._structure_cache is None:
                    app._poll_queue()
                    time.sleep(0.02)
                self.assertIsNotNone(app._structure_cache)
                self.assertNotIn("Inspect error", app.var_inspector.get())
                cfg = app._collect_config()
                self.assertEqual(cfg.replicate, (1, 1, 1))

                # DarkSpinbox and DarkCheckbutton interactions
                sb = DarkSpinbox(app, variable=app.var_samples, from_=1, to=2048, increment=8)
                sb._step(8)
                self.assertEqual(app.var_samples.get(), 72)
                sb._step(-8)
                self.assertEqual(app.var_samples.get(), 64)

                chk = DarkCheckbutton(app, text="Test", variable=app.var_axes_labels)
                prev = app.var_axes_labels.get()
                chk._on_click(None)  # type: ignore[arg-type]
                self.assertEqual(app.var_axes_labels.get(), not prev)
                chk.configure(state="disabled")
                chk._on_click(None)  # type: ignore[arg-type]
                self.assertEqual(app.var_axes_labels.get(), not prev)
            finally:
                app.destroy()

    def test_selection_tab_double_click_opacity_and_dark_slider(self) -> None:
        import numpy as np
        from atomic_blender.gui import AtomicBlenderApp, DarkSlider, compute_atom_alphas

        self.assertEqual(compute_atom_alphas(3, set(), 0.3, False), [1.0, 1.0, 1.0])
        self.assertEqual(compute_atom_alphas(3, {0}, 0.3, False), [0.3, 1.0, 1.0])
        self.assertEqual(compute_atom_alphas(3, {0}, 0.3, True), [1.0, 0.3, 0.3])

        structure = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-1.8, 0.0, 0.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [False, False, False],
            "has_cell": False,
            "origin": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "h2o.xyz"
            xyz.write_text("3\n\nO 0 0 0\nH 1.8 0 0\nH -1.8 0 0\n", encoding="utf-8")

            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.var_blender.set("")
                app.var_input.set(str(xyz))
                app.canvas.configure(width=600, height=600)
                app.update_idletasks()
                app._structure_cache = dict(structure)
                app._structure_version = 1
                app._rebuild_viewport(reset_camera=True, trigger_preview=False)
                assert app._vp_scene is not None

                # Atom Labels moved out of View tab's Display Types into Selection tab
                self.assertNotIn("atom_labels", app._display_checks)
                app._switch_tab("selection")
                self.assertEqual(app._active_tab, "selection")
                self.assertEqual(app.tab_selection.winfo_manager(), "pack")

                # Atom List matches 1-based labels (O1, H2, H3)
                items = [app.atom_listbox.get(i) for i in range(app.atom_listbox.size())]
                self.assertEqual(len(items), 3)
                self.assertIn("O1", items[0])
                self.assertIn("H2", items[1])
                self.assertIn("H3", items[2])

                # Toggle Atom Labels in Selection tab
                app.var_show_labels.set(True)
                app._on_labels_toggled()
                self.assertTrue(app._vp_scene.show_labels)

                # Double-click at canvas center hits O1 (index 0) and updates alphas + Selection list
                app._vp_scene.quat = (1.0, 0.0, 0.0, 0.0)
                app._vp_scene.target = np.zeros(3)
                cx_px = app.canvas.winfo_width() // 2
                cy_px = app.canvas.winfo_height() // 2
                ev_dbl = type("E", (), {"x": cx_px, "y": cy_px, "state": 0})()
                app._dragging = True
                app._on_canvas_double_click(ev_dbl)  # type: ignore[arg-type]
                self.assertFalse(app._dragging)
                self.assertEqual(app._selected_atoms, {0})
                self.assertEqual(app._structure_cache["alphas"], [0.3, 1.0, 1.0])

                # Fade Unselected flips opacity to unselected atoms [1.0, 0.3, 0.3]
                app.var_alpha_invert.set(True)
                app._on_opacity_released()
                self.assertEqual(app._structure_cache["alphas"], [1.0, 0.3, 0.3])

                # Effective CLI config writes temp structure JSON carrying per-atom alphas
                eff = app._effective_cfg_for_cli(app._collect_config())
                self.assertTrue(eff.input_path.endswith(".json"))
                payload = json.loads(Path(eff.input_path).read_text(encoding="utf-8"))
                self.assertEqual(payload["alphas"], [1.0, 0.3, 0.3])

                # Same Element with H2 selected selects H2 and H3
                app._selected_atoms = {1}
                app._select_same_element()
                self.assertEqual(app._selected_atoms, {1, 2})

                # DarkSlider programmatically updates variable
                sl = DarkSlider(app, from_=0.05, to=1.0, variable=app.var_sel_opacity, command=app._on_opacity_changed)
                sl.set_value(0.50)
                self.assertAlmostEqual(app.var_sel_opacity.get(), 0.50, places=2)
                self.assertEqual(app.var_sel_opacity_label.get(), "50%")
            finally:
                app.destroy()

    def test_qol_measurements_chips_z_range_recent_and_shortcuts(self) -> None:
        import math
        from atomic_blender.gui import AtomicBlenderApp, compute_selection_measurement

        # 1. Pure measurement calculations (1-atom, 2-atom, 3-atom, 4-atom dihedral)
        syms4 = ["C", "O", "H", "H"]
        pos4 = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
        self.assertIn("C1:", compute_selection_measurement(syms4, pos4, [0]))
        m2 = compute_selection_measurement(syms4, pos4, [0, 1])
        self.assertIn("Dist C1–O2: 1.500 Å", m2)
        m3 = compute_selection_measurement(syms4, pos4, [0, 1, 2])
        self.assertIn("Angle C1–O2–H3: 90.00°", m3)

        # Dihedral test with known 90 deg torsion along Y axis
        dih_syms = ["A", "B", "C", "D"]
        dih_pos = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
        m4 = compute_selection_measurement(dih_syms, dih_pos, [0, 1, 2, 3])
        self.assertIn("Dihedral A1–B2–C3–D4:", m4)
        self.assertIn("90.00°", m4)
        self.assertTrue(math.isfinite(float(m4.split(":")[-1].strip().rstrip("°"))))

        structure = {
            "symbols": ["C", "H", "H", "O"],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 3.5]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [False, False, False],
            "has_cell": False,
            "origin": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "mol.xyz"
            xyz.write_text("4\n\nC 0 0 0\nH 1 0 0\nH -1 0 0\nO 0 0 3.5\n", encoding="utf-8")
            prefs_file = Path(tmp) / "prefs.json"

            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app._recent_prefs_path = lambda: prefs_file  # type: ignore[method-assign]
                app.var_blender.set("")
                app.var_input.set(str(xyz))
                app.canvas.configure(width=600, height=600)
                app.update_idletasks()
                app._structure_cache = dict(structure)
                app._structure_version = 1
                app._rebuild_viewport(reset_camera=True, trigger_preview=False)

                # Element chips built dynamically for C, H, O
                self.assertEqual(set(app._elem_chip_buttons.keys()), {"C", "H", "O"})
                app._toggle_element_selection("H")
                self.assertEqual(app._selected_atoms, {1, 2})
                self.assertIn("Dist H2–H3: 2.000 Å", app.var_measurement.get())
                app._toggle_element_selection("H")
                self.assertEqual(app._selected_atoms, set())

                # Z-range selection filters to top O atom only
                app.var_z_min.set("2.0")
                app.var_z_max.set("4.0")
                app._select_by_z_range()
                self.assertEqual(app._selected_atoms, {3})
                self.assertIn("O4:", app.var_measurement.get())

                # Escape clears selection via keyboard shortcut dispatcher
                app._on_key_shortcut("Escape")
                self.assertEqual(app._selected_atoms, set())

                # L and A toggle labels and axes
                prev_lbl = app.var_show_labels.get()
                app._on_key_shortcut("l")
                self.assertEqual(app.var_show_labels.get(), not prev_lbl)
                prev_ax = app.var_show_axes.get()
                app._on_key_shortcut("a")
                self.assertEqual(app.var_show_axes.get(), not prev_ax)

                # Current viewport produces a valid PIL image for clipboard copy
                img = app._current_canvas_pil_image()
                self.assertIsNotNone(img)
                self.assertGreaterEqual(img.size[0], 600)

                # Recent files persist cleanly into prefs_file
                app._recent_files = []
                app._add_recent_file(xyz)
                saved = json.loads(prefs_file.read_text(encoding="utf-8"))
                self.assertEqual(len(saved["recent_files"]), 1)
                self.assertEqual(Path(saved["recent_files"][0]).name, "mol.xyz")
            finally:
                app.destroy()

    def test_stage2_colormaps_hbonds_material_presets_and_ghost_overlay(self) -> None:
        from PIL import Image
        from atomic_blender.gui import AtomicBlenderApp, MATERIAL_FINISH_PRESETS
        from atomic_blender.viewport3d import (
            ViewportScene,
            _find_bonds,
            compute_z_colormap_colors,
            find_dashed_contacts,
            merge_ghost_structure,
            render_colorbar_overlay,
            sample_colormap_hex,
        )

        # 1. Colormaps + Colorbar overlay
        self.assertEqual(sample_colormap_hex("viridis", 0.0), "#440154")
        self.assertEqual(sample_colormap_hex("viridis", 1.0), "#FDE725")
        pos_z = [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 10.0]]
        cols, zlo, zhi = compute_z_colormap_colors(pos_z, "coolwarm")
        self.assertEqual(len(cols), 3)
        self.assertEqual((zlo, zhi), (0.0, 10.0))
        img = Image.new("RGBA", (600, 600), (21, 23, 28, 255))
        render_colorbar_overlay(img, "coolwarm", zlo, zhi, dark_backdrop=True)

        # 2. H-bonds & interface contacts detection (excluding covalent bonds)
        hb_syms = ["O", "H", "O", "H"]
        hb_pos = [
            [0.0, 0.0, 0.0],   # O1
            [0.98, 0.0, 0.0],  # H2 bonded to O1
            [2.80, 0.0, 0.0],  # O3 (H-bond acceptor from H2: dist=1.82 A)
            [3.78, 0.0, 0.0],  # H4 bonded to O3
        ]
        import numpy as np

        bi, bj = _find_bonds(hb_syms, np.asarray(hb_pos), {"mode": "auto", "enabled": True})
        hb_pairs = find_dashed_contacts(hb_syms, hb_pos, bond_i=bi, bond_j=bj, mode="hbond", d_min=1.55, d_max=2.55)
        self.assertEqual(hb_pairs, [(1, 2)])
        iface_pairs = find_dashed_contacts(
            hb_syms, hb_pos, bond_i=bi, bond_j=bj, mode="interface", d_min=1.5, d_max=3.0, focal_indices={2, 3}
        )
        self.assertEqual(iface_pairs, [(0, 2), (1, 2), (1, 3)])

        # 3. Ghost reference overlay (no cross-bonds, preserves displacement vectors)
        main_struct = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.96, 0.0, 1.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [False, False, False],
            "has_cell": False,
            "origin": [0.0, 0.0, 0.0],
        }
        ghost_struct = {
            "symbols": ["O", "H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.96, 0.0, 0.0]],
        }
        merged = merge_ghost_structure(main_struct, ghost_struct, disp_threshold_a=0.2)
        self.assertEqual(len(merged["symbols"]), 6)
        self.assertEqual(merged["ghost_start_index"], 3)
        self.assertEqual(merged["dashed_bonds"], [[2, 5]])
        vp_g = ViewportScene.from_neutral_dict(merged, style={"bonds": {"mode": "auto", "enabled": True}})
        for i_b, j_b in zip(vp_g.bond_i, vp_g.bond_j):
            self.assertEqual(int(i_b) < 3, int(j_b) < 3)

        # 4. GUI integration: Z colormap, H-bonds, Finish preset, Ghost overlay, CLI payload
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "hb.xyz"
            xyz.write_text("4\n\nO 0 0 0\nH 0.98 0 0\nO 2.8 0 0\nH 3.78 0 0\n", encoding="utf-8")
            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.var_blender.set("")
                app.var_input.set(str(xyz))
                app.canvas.configure(width=600, height=600)
                app.update_idletasks()
                hb_struct = {
                    "symbols": hb_syms,
                    "positions": hb_pos,
                    "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "pbc": [False, False, False],
                    "has_cell": False,
                    "origin": [0.0, 0.0, 0.0],
                }
                app._base_structure_cache = dict(hb_struct)
                app._structure_cache = dict(hb_struct)
                app._structure_version = 1
                app.var_bonds.set("auto")

                app.var_color_map.set("viridis")
                app._on_colormap_changed()
                self.assertEqual(len(app._structure_cache["colors"]), 4)
                self.assertIsNotNone(app._vp_scene.colormap_info)

                app._on_display_type_clicked("hbonds")
                self.assertTrue(app.var_show_hbonds.get())
                self.assertEqual(app._structure_cache["dashed_bonds"], [[1, 2]])

                app._set_material_finish("chalk")
                eff = app._effective_cfg_for_cli(app._collect_config())
                style_out = json.loads(Path(eff.style).read_text(encoding="utf-8"))
                self.assertAlmostEqual(
                    style_out["material"]["roughness"], MATERIAL_FINISH_PRESETS["chalk"]["roughness"]
                )
                struct_out = json.loads(Path(eff.input_path).read_text(encoding="utf-8"))
                self.assertEqual(struct_out["dashed_bonds"], [[1, 2]])
                self.assertEqual(len(struct_out["colors"]), 4)

                # Top Settings drawer toggle & Reset Focus recipe
                self.assertFalse(bool(app.f_settings.winfo_manager()))
                app._toggle_settings_panel()
                self.assertTrue(bool(app.f_settings.winfo_manager()))
                app._toggle_settings_panel()
                self.assertFalse(bool(app.f_settings.winfo_manager()))

                app._reset_focus_recipes()
                self.assertEqual(app.var_color_map.get(), "none")
                self.assertEqual(app.var_unsel_scale_mode.get(), "same")
                self.assertEqual(app._selected_atoms, set())
            finally:
                app.destroy()

    def test_turnkey_recipes_overlays_hybrid_scales_and_dual_panel(self) -> None:
        from PIL import Image
        from atomic_blender.gui import (
            AtomicBlenderApp,
            auto_detect_adsorbate_indices,
            compute_adsorbate_spotlight_alphas,
            compute_atom_radius_scales,
            compute_depth_fade_alphas,
            find_neighbor_atoms,
        )
        from atomic_blender.viewport3d import (
            ViewportScene,
            choose_scale_bar_angstrom,
            compose_dual_view_panel,
            render_measurement_callout_overlay,
            render_scale_bar_overlay,
        )

        syms = ["Cu", "Cu", "Cu", "C", "O"]
        pos = [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 2.0],
            [0.0, 2.5, 4.0],
            [0.0, 2.5, 5.2],
        ]
        focal_auto = auto_detect_adsorbate_indices(syms, pos)
        self.assertEqual(focal_auto, {3, 4})

        neigh = find_neighbor_atoms(pos, {3}, cutoff_a=2.1)
        self.assertEqual(neigh, {2, 3, 4})

        focal, spot_alphas = compute_adsorbate_spotlight_alphas(syms, pos, shell_cutoff_a=2.2)
        self.assertEqual(focal, {3, 4})
        self.assertEqual(spot_alphas[3], 1.0)
        self.assertEqual(spot_alphas[4], 1.0)
        self.assertEqual(spot_alphas[2], 0.85)
        self.assertLess(spot_alphas[0], 0.5)

        fade_alphas = compute_depth_fade_alphas(pos, min_alpha=0.2, max_alpha=1.0)
        self.assertAlmostEqual(fade_alphas[0], 0.2, places=2)
        self.assertAlmostEqual(fade_alphas[-1], 1.0, places=2)

        r_scales = compute_atom_radius_scales(5, {3, 4}, sel_scale=1.12, unsel_scale=0.55)
        self.assertEqual(r_scales, [0.55, 0.55, 0.55, 1.12, 1.12])

        # Scale bar & measurement callouts & dual-view panel
        self.assertEqual(choose_scale_bar_angstrom(25.0), 5.0)
        img = Image.new("RGBA", (600, 600), (21, 23, 28, 255))
        bar_a = render_scale_bar_overlay(img, 25.0, 600, 600, dark_backdrop=True)
        self.assertEqual(bar_a, 5.0)

        struct = {
            "symbols": syms,
            "positions": pos,
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [False, False, False],
            "has_cell": False,
            "origin": [0.0, 0.0, 0.0],
            "radius_scales": r_scales,
        }
        vp = ViewportScene.from_neutral_dict(
            struct,
            style={},
            show_scale_bar=True,
            show_measure_overlay=True,
            selection_order=[2, 3, 4],
        )
        self.assertAlmostEqual(vp.radii[3] / vp.radii[2], (0.76 * 1.12) / (1.32 * 0.55), places=3)
        render_measurement_callout_overlay(img, vp, [3, 4], 600, 600)
        render_measurement_callout_overlay(img, vp, [2, 3, 4], 600, 600)
        render_measurement_callout_overlay(img, vp, [0, 2, 3, 4], 600, 600)
        panel = compose_dual_view_panel(img, img, dark_backdrop=True)
        self.assertGreater(panel.width, 1200)
        self.assertGreater(panel.height, 600)

        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "ads.xyz"
            xyz.write_text(
                "5\n\nCu 0 0 0\nCu 2.5 0 0\nCu 0 2.5 2\nC 0 2.5 4\nO 0 2.5 5.2\n",
                encoding="utf-8",
            )
            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.var_blender.set("")
                app.var_input.set(str(xyz))
                app.canvas.configure(width=600, height=600)
                app.update_idletasks()
                app._structure_cache = dict(struct)
                app._structure_version = 1
                app._rebuild_viewport(reset_camera=True, trigger_preview=False)

                # Scale bar display toggle
                self.assertIn("scale_bar", app._display_checks)
                app._on_display_type_clicked("scale_bar")
                self.assertTrue(app.var_show_scale_bar.get())
                self.assertTrue(app._vp_scene.show_scale_bar)

                # Adsorbate spotlight + hybrid scaling + custom element color export
                app._apply_adsorbate_spotlight_recipe()
                self.assertEqual(app._selected_atoms, {3, 4})
                self.assertEqual(app.var_unsel_scale_mode.get(), "slim")
                self.assertEqual(app._structure_cache["radius_scales"], [0.55, 0.55, 0.55, 1.12, 1.12])

                app.var_neighbor_cutoff.set(2.1)
                app._selected_atoms = {3}
                app._expand_selection_neighbors()
                self.assertEqual(app._selected_atoms, {2, 3, 4})

                app._custom_element_colors["Cu"] = "#FF9900"
                eff = app._effective_cfg_for_cli(app._collect_config())
                self.assertTrue(eff.input_path.endswith(".json"))
                self.assertTrue(eff.style.endswith(".json"))
                style_payload = json.loads(Path(eff.style).read_text(encoding="utf-8"))
                self.assertEqual(style_payload["elements"]["colors"]["Cu"], "#FF9900")
            finally:
                app.destroy()

    def test_stage3_and_tilted_cells(self) -> None:
        import math
        import numpy as np
        from PIL import Image
        from atomic_blender.convert import read_structure_frames
        from atomic_blender.elements import VIEW_PRESETS, classify_dropped_file
        from atomic_blender.gui import AtomicBlenderApp
        from atomic_blender.viewport3d import (
            ViewportScene,
            compose_multi_panel_strip,
            compute_cell_parameters,
            compute_slice_mask,
            find_coordination_polyhedra,
            is_tilted_cell,
            preset_to_quat,
            quat_to_axes,
            render_reference_axes_overlay,
        )

        # 1. Coordination polyhedra (TiO6 octahedron -> 8 triangular faces, 12 edges)
        oct_syms = ["Ti", "O", "O", "O", "O", "O", "O"]
        oct_pos = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ]
        polys = find_coordination_polyhedra(oct_syms, oct_pos, cutoff_a=2.3, alpha=0.40)
        self.assertEqual(len(polys), 1)
        self.assertEqual(len(polys[0]["faces"]), 8)
        self.assertEqual(len(polys[0]["edges"]), 12)

        # 2. Tilted hexagonal cell parameters, camera presets ('c' & 'c*'), and axes gizmo
        a_hex = 3.2
        hex_cell = np.array([
            [a_hex, 0.0, 0.0],
            [-0.5 * a_hex, 0.5 * math.sqrt(3.0) * a_hex, 0.0],
            [0.0, 0.0, 10.0],
        ])
        la, lb, lc, alpha, beta, gamma = compute_cell_parameters(hex_cell)
        self.assertAlmostEqual(la, a_hex, places=4)
        self.assertAlmostEqual(lb, a_hex, places=4)
        self.assertAlmostEqual(gamma, 120.0, places=3)
        self.assertTrue(is_tilted_cell(hex_cell))
        self.assertFalse(is_tilted_cell(np.diag([10.0, 10.0, 10.0])))
        self.assertIn("c*", VIEW_PRESETS)

        q_c = preset_to_quat("c", hex_cell)
        right_c, up_c, back_c = quat_to_axes(q_c)
        self.assertAlmostEqual(float(np.dot(right_c, hex_cell[0] / a_hex)), 1.0, places=4)

        mono_cell = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.5, 0.0, 8.0],
        ])
        q_cstar = preset_to_quat("c*", mono_cell)
        _, _, back_cs = quat_to_axes(q_cstar)
        self.assertAlmostEqual(float(np.dot(back_cs, np.array([0.0, 0.0, 1.0]))), 1.0, places=4)

        # Fractional cross-section slice along 'a' on a slanted monoclinic cell
        skew_pts = [[0.4, 0.0, 0.0], [1.6, 0.0, 7.8], [3.5, 0.0, 0.0]]
        mask_a = compute_slice_mask(skew_pts, mode="a", depth_fraction=0.5, cell_vecs=mono_cell)
        # Cartesian x of point 1 is 1.6 (<2), yet fractional u = (1.6 - 1.5*7.8/8)/4 = 0.034 -> kept
        self.assertEqual(mask_a, [True, True, False])
        mask_cam = compute_slice_mask(oct_pos, mode="camera", depth_fraction=0.5, quat=(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(len(mask_cam), 7)
        self.assertTrue(any(not m for m in mask_cam))

        img_cart = Image.new("RGBA", (300, 300), (24, 26, 32, 255))
        img_hex = Image.new("RGBA", (300, 300), (24, 26, 32, 255))
        render_reference_axes_overlay(img_cart, (1.0, 0.0, 0.0, 0.0))
        render_reference_axes_overlay(img_hex, (1.0, 0.0, 0.0, 0.0), cell_vecs=hex_cell)
        self.assertNotEqual(img_cart.tobytes(), img_hex.tobytes())

        # 3. Multi-panel path strip stitching
        strip = compose_multi_panel_strip(
            [img_cart, img_hex, img_cart],
            titles=["Frame 1", "Frame 2", "Frame 3"],
            dark_backdrop=True,
        )
        self.assertGreater(strip.width, 900)
        self.assertGreater(strip.height, 300)

        # 4. Trajectory reading + GUI Stage 3 integration (polyhedra, vectors, slice, frames, recipe)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            traj_xyz = tmp_path / "neb_3frames.xyz"
            lines = []
            for k in range(3):
                lines.append("7")
                lines.append('Lattice="8.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 8.0" Properties=species:S:1:pos:R:3')
                lines.append(f"Ti {4.0 + k * 0.2:.2f} 4.0 4.0")
                for dx, dy, dz in oct_pos[1:]:
                    lines.append(f"O {4.0 + dx + k * 0.2:.2f} {4.0 + dy:.2f} {4.0 + dz:.2f}")
            traj_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")

            frames = read_structure_frames(traj_xyz)
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(frames[0]["symbols"]), 7)

            recipe_path = tmp_path / "fig.recipe.json"
            app = AtomicBlenderApp()
            app.withdraw()
            try:
                app.var_blender.set("")
                app.var_input.set(str(traj_xyz))
                app.canvas.configure(width=600, height=600)
                app.update_idletasks()
                app._queue.put(("structure_frames", ("3 frames", frames)))
                app._poll_queue()

                self.assertEqual(len(app._trajectory_frames), 3)
                self.assertEqual(app.var_frame_label.get(), "Frame 1 / 3")
                app._step_trajectory_frame(1)
                self.assertEqual(app._trajectory_index, 1)
                self.assertEqual(app.var_frame_label.get(), "Frame 2 / 3")

                # Polyhedra + Lattice Vectors (a, b, c)
                app._on_display_type_clicked("polyhedra")
                app._on_display_type_clicked("cell_vectors")
                self.assertTrue(app.var_show_polyhedra.get())
                self.assertTrue(app.var_show_cell_vectors.get())
                self.assertEqual(len(app._structure_cache["polyhedra"]), 1)
                self.assertTrue(app._structure_cache["show_cell_vectors"])
                self.assertTrue(app._vp_scene.show_cell_vectors)

                # Cross-section slice
                app.var_slice_mode.set("z")
                app.var_slice_depth.set(0.5)
                app._on_slice_changed(trigger_blender=False)
                self.assertTrue(any(a <= 0.07 for a in app._structure_cache["alphas"]))
                app._reset_slice()
                self.assertEqual(app.var_slice_mode.get(), "none")

                # Recipe export / classify / drop / apply + CLI JSON payload
                recipe = app.export_figure_recipe()
                recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
                self.assertEqual(classify_dropped_file(recipe_path), "recipe")

                app.var_show_polyhedra.set(False)
                app.var_show_cell_vectors.set(False)
                app.handle_dropped_files([str(recipe_path)])
                self.assertTrue(app.var_show_polyhedra.get())
                self.assertTrue(app.var_show_cell_vectors.get())

                eff = app._effective_cfg_for_cli(app._collect_config())
                self.assertTrue(eff.input_path.endswith(".json"))
                payload = json.loads(Path(eff.input_path).read_text(encoding="utf-8"))
                self.assertEqual(len(payload["polyhedra"]), 1)
                self.assertTrue(payload["show_cell_vectors"])
            finally:
                app.destroy()

    def test_missing_blender_detection_and_download_prompt(self) -> None:
        from unittest.mock import patch
        from atomic_blender.cli import BLENDER_DOWNLOAD_URL
        from atomic_blender.gui import AtomicBlenderApp

        with patch("atomic_blender.gui.find_blender_executable", side_effect=FileNotFoundError("not found")):
            app = AtomicBlenderApp(prompt_missing_blender=False)
            app.withdraw()
            try:
                self.assertEqual(app.var_blender.get(), "")
                with (
                    patch("atomic_blender.gui.messagebox.askyesno", return_value=False) as mock_ask,
                    patch("atomic_blender.gui.webbrowser.open") as mock_open,
                ):
                    app._check_blender_on_startup()
                    mock_ask.assert_called_once()
                    mock_open.assert_not_called()

                # Startup prompt only fires once automatically; render action forces prompt again
                with (
                    patch("atomic_blender.gui.messagebox.askyesno", return_value=True) as mock_ask,
                    patch("atomic_blender.gui.webbrowser.open") as mock_open,
                ):
                    app._on_render()
                    mock_ask.assert_called_once()
                    mock_open.assert_called_once_with(BLENDER_DOWNLOAD_URL)
            finally:
                app.destroy()

    def test_desktop_icon_shortcut_and_frozen_cli_dispatch(self) -> None:
        import os
        from atomic_blender.cli import build_parser, ensure_app_icon, install_desktop_shortcut
        from atomic_blender.gui import main as gui_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ico = ensure_app_icon(tmp_path / "studio.ico")
            self.assertTrue(ico.is_file())
            self.assertGreater(ico.stat().st_size, 500)

            self.assertEqual(gui_main(["-m", "atomic_blender.cli", "styles"]), 0)
            parsed = build_parser().parse_args(["install-desktop", "--no-start-menu"])
            self.assertTrue(parsed.no_start_menu)
            self.assertFalse(parsed.no_desktop)

            if os.name == "nt":
                fake_exe = tmp_path / "AtomicBlenderStudio.exe"
                fake_exe.write_text("stub", encoding="utf-8")
                desk_dir = tmp_path / "Desktop"
                start_dir = tmp_path / "StartMenu"
                links = install_desktop_shortcut(
                    desktop=True,
                    start_menu=True,
                    target_exe=fake_exe,
                    desktop_dir=desk_dir,
                    start_menu_dir=start_dir,
                )
                self.assertEqual(len(links), 2)
                for lnk in links:
                    self.assertTrue(lnk.is_file())
                    self.assertEqual(lnk.name, "Atomic Blender Studio.lnk")

    def test_review_fixes_pbc_polyhedra_scalebar_batch_neb_and_spatial_contacts(self) -> None:
        import time
        import numpy as np
        from PIL import Image
        from ase import Atoms
        from atomic_blender.convert import atoms_to_neutral_dict, read_structure
        from atomic_blender.gui import (
            AtomicBlenderApp,
            compute_selection_measurement,
            resolve_batch_output_paths,
            select_path_strip_indices,
        )
        from atomic_blender.viewport3d import (
            ViewportScene,
            find_coordination_polyhedra,
            find_dashed_contacts,
            render_scale_bar_overlay,
        )

        # 1. Perovskite BaTiO3 (examples/04_perovskite_BaTiO3_2x2x2.cif): all 8 TiO6 octahedra recovered via PBC
        bto_cif = Path(__file__).resolve().parent.parent / "examples" / "04_perovskite_BaTiO3_2x2x2.cif"
        if bto_cif.is_file():
            bto_dict = atoms_to_neutral_dict(read_structure(bto_cif))
            polys_pbc = find_coordination_polyhedra(
                bto_dict["symbols"],
                bto_dict["positions"],
                cell_vecs=bto_dict["cell"],
                pbc=bto_dict["pbc"],
            )
            self.assertEqual(len(polys_pbc), 8)
            self.assertTrue(all(len(p["vertices"]) == 6 for p in polys_pbc))

        # 2. Non-periodic molecule inside a 10 A box preserves bond distance & selective slab PBC
        mol = Atoms("HO", positions=[[-0.2, 0.0, 0.0], [0.8, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=[False, False, False])
        mol_d = atoms_to_neutral_dict(mol, wrap=True)
        dist_ho = float(np.linalg.norm(np.array(mol_d["positions"][1]) - np.array(mol_d["positions"][0])))
        self.assertAlmostEqual(dist_ho, 1.0, places=5)
        self.assertAlmostEqual(mol_d["positions"][0][0], -0.2, places=5)

        slab = Atoms("CuH", positions=[[10.5, 0.0, -0.5], [0.5, 0.0, 2.0]], cell=[10.0, 10.0, 10.0], pbc=[True, True, False])
        slab.info["energy"] = -12.345
        slab_d = atoms_to_neutral_dict(slab, wrap=True)
        self.assertAlmostEqual(slab_d["positions"][0][0], 0.5, places=5)
        self.assertAlmostEqual(slab_d["positions"][0][2], -0.5, places=5)
        self.assertAlmostEqual(slab_d["energy_ev"], -12.345, places=5)

        # 3. Portrait scale bar chooses 1.0 A (= 120 px) instead of clamping 2.0 A to 180 px
        portrait = Image.new("RGBA", (400, 1200), (21, 23, 28, 255))
        bar_a = render_scale_bar_overlay(portrait, ortho_scale=10.0, res_w=400, res_h=1200, dark_backdrop=True)
        self.assertEqual(bar_a, 1.0)

        # 4. Batch output paths resolve stem collisions cleanly
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            out_paths = resolve_batch_output_paths(
                [Path("sample.cif"), Path("sample.xyz"), Path("other.cif"), Path("other.cif")],
                out_dir,
            )
            names = [p.name for p in out_paths]
            self.assertEqual(len(set(names)), 4)
            self.assertEqual(names[:2], ["sample_cif.png", "sample_xyz.png"])

        # 5. Path strip always includes first (0) and final (11) frames in a 12-frame NEB path
        indices_12 = select_path_strip_indices(12, max_panels=6, pinned_indices={3})
        self.assertEqual(indices_12[0], 0)
        self.assertEqual(indices_12[-1], 11)
        self.assertIn(3, indices_12)
        self.assertEqual(len(indices_12), 6)

        # 6. Measurement displacement Δr from reference & spatial-grid dashed contacts benchmark
        m_ref = compute_selection_measurement(["C"], [[1.0, 2.0, 3.0]], [0], reference_positions=[[1.0, 2.0, 2.0]])
        self.assertIn("Δr 1.000 Å", m_ref)

        # 2000-atom grid contact search finishes quickly
        pts_big = np.column_stack([
            np.linspace(0.0, 120.0, 2000),
            np.zeros(2000),
            np.zeros(2000),
        ])
        syms_big = ["H" if i % 2 == 0 else "O" for i in range(2000)]
        t0 = time.perf_counter()
        pairs_big = find_dashed_contacts(syms_big, pts_big, mode="hbond", d_min=1.5, d_max=2.5)
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertGreater(len(pairs_big), 0)

        # 7. GUI integration: Compare reference persists across trajectory frames, thumbnails keep atoms visible,
        # Frame mask & Contact cache work cleanly
        app = AtomicBlenderApp()
        app.withdraw()
        try:
            app.var_blender.set("")
            app.canvas.configure(width=800, height=500)
            app.update_idletasks()
            f0 = {
                "symbols": ["Cu", "C", "O"],
                "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.15]],
                "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                "pbc": [True, True, False],
                "has_cell": True,
                "origin": [0.0, 0.0, 0.0],
                "energy_ev": -10.0,
            }
            f1 = {
                "symbols": ["Cu", "C", "O"],
                "positions": [[0.0, 0.0, 0.0], [0.5, 0.0, 2.0], [0.5, 0.0, 3.15]],
                "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                "pbc": [True, True, False],
                "has_cell": True,
                "origin": [0.0, 0.0, 0.0],
                "energy_ev": -9.5,
            }
            app._queue.put(("structure_frames", ("2 frames", [f0, f1])))
            app._poll_queue()

            app._pin_current_as_ghost()
            self.assertIsNotNone(app._ghost_structure_cache)
            app._step_trajectory_frame(1)
            # Pinned reference survives trajectory frame change
            self.assertIsNotNone(app._ghost_structure_cache)
            self.assertEqual(app._structure_cache["ghost_start_index"], 3)

            # Clicking active representation thumbnail never hides atoms
            app._on_display_type_clicked("ball_stick")
            app._on_display_type_clicked("ball_stick")
            self.assertTrue(app._show_atoms)
            self.assertTrue(app.var_show_atoms.get())

            # Toggle output frame mask
            app.var_show_output_mask.set(True)
            app._on_output_mask_toggled()
            self.assertTrue(app._vp_scene.show_output_mask)

            # Side-by-side compare builds synchronized ghost scene
            app.var_compare_mode.set("side_by_side")
            app._on_ghost_options_changed(trigger_blender=False)
            self.assertIsNotNone(app._ghost_vp_scene)
            app._draw_3d_viewport(interactive=False)

            # Contact cache hit when changing colormap without structure change
            app.var_show_hbonds.set(True)
            app.var_hbond_mode.set("range")
            app._rebuild_viewport(reset_camera=False, trigger_preview=False)
            key1 = app._contact_cache_key
            self.assertIsNotNone(key1)
            app.var_color_map.set("viridis")
            app._on_colormap_changed()
            self.assertEqual(app._contact_cache_key, key1)
        finally:
            app.destroy()

    def test_welcome_screen_layout_and_animation_loop(self) -> None:
        from PIL import Image
        from atomic_blender.cli import package_dir
        from atomic_blender.gui import (
            AtomicBlenderApp,
            WELCOME_FRAME_INTERVAL_MS,
            _blend_hex_color,
            _welcome_header_fade_alpha,
            _welcome_prompt_fade_alpha,
            extract_input_path,
            format_input_display,
        )

        self.assertLessEqual(WELCOME_FRAME_INTERVAL_MS, 100)
        # Header: starts dim (0.15) -> bright (1.0 at step 16) -> holds faint (0.32 for step >= 36)
        self.assertAlmostEqual(_welcome_header_fade_alpha(0), 0.15, places=3)
        self.assertAlmostEqual(_welcome_header_fade_alpha(16), 1.0, places=3)
        self.assertAlmostEqual(_welcome_header_fade_alpha(36), 0.32, places=3)
        self.assertAlmostEqual(_welcome_header_fade_alpha(80), 0.32, places=3)
        # Bottom prompt: dim -> bright -> repeating
        self.assertAlmostEqual(_welcome_prompt_fade_alpha(0), 0.22, places=2)
        self.assertAlmostEqual(_welcome_prompt_fade_alpha(32), 1.0, places=2)
        self.assertAlmostEqual(_welcome_prompt_fade_alpha(64), _welcome_prompt_fade_alpha(0), places=3)

        self.assertEqual(_blend_hex_color("#000000", "#FFFFFF", 0.0), "#000000")
        self.assertEqual(_blend_hex_color("#000000", "#FFFFFF", 1.0), "#FFFFFF")
        sample_p = r"C:\tmp\demo\sample.cif"
        self.assertEqual(format_input_display(sample_p), r"sample.cif (C:\tmp\demo\sample.cif)")
        self.assertEqual(extract_input_path(format_input_display(sample_p)), sample_p)

        gif_path = package_dir() / "assets" / "welcome_loop.gif"
        self.assertTrue(gif_path.is_file(), f"Missing pre-rendered welcome GIF: {gif_path}")
        with Image.open(gif_path) as im:
            self.assertGreaterEqual(getattr(im, "n_frames", 1), 72)
            self.assertEqual(im.size, (360, 360))

        app = AtomicBlenderApp()
        app.withdraw()
        try:
            self.assertFalse(hasattr(app, "_repr_thumb_cards"))
            self.assertIn("ball_stick", app._display_checks)
            self.assertIn("licorice", app._display_checks)
            self.assertIn("vdw", app._display_checks)
            # Settings button moved into the same preview toolbar as Log toggle
            self.assertEqual(app.btn_settings.winfo_parent(), app._btn_log_toggle.winfo_parent())

            app.canvas.configure(width=900, height=700)
            app.update_idletasks()
            app._draw_3d_viewport()

            title_ids = app.canvas.find_withtag("welcome_title")
            sub_ids = app.canvas.find_withtag("welcome_subtitle")
            atom_ids = app.canvas.find_withtag("welcome_atom")
            drag_ids = app.canvas.find_withtag("welcome_drag")
            fmt_ids = app.canvas.find_withtag("welcome_formats")
            self.assertTrue(title_ids and sub_ids and atom_ids and drag_ids and fmt_ids)
            self.assertEqual(app.canvas.find_withtag("welcome_style"), ())
            self.assertEqual(app.canvas.find_withtag("welcome_style_desc"), ())
            self.assertEqual(app.canvas.find_withtag("welcome_dots"), ())
            self.assertEqual(app.canvas.itemcget(title_ids[0], "text"), "Welcome")
            self.assertEqual(app.canvas.itemcget(sub_ids[0], "text"), "Atomic Blender Studio")
            title_font = app.canvas.itemcget(title_ids[0], "font")
            self.assertIn("64", title_font)
            self.assertIn("Light", title_font)
            self.assertIn("Drag & drop", app.canvas.itemcget(drag_ids[0], "text"))
            self.assertIn("CIF", app.canvas.itemcget(fmt_ids[0], "text"))

            # Atomic Blender Studio sits directly above Welcome with zero bbox overlap
            sub_bbox = app.canvas.bbox(sub_ids[0])
            title_bbox = app.canvas.bbox(title_ids[0])
            atom_bbox = app.canvas.bbox(atom_ids[0])
            drag_bbox = app.canvas.bbox(drag_ids[0])
            self.assertIsNotNone(sub_bbox)
            self.assertIsNotNone(title_bbox)
            self.assertIsNotNone(atom_bbox)
            self.assertIsNotNone(drag_bbox)
            self.assertLessEqual(sub_bbox[3], title_bbox[1])
            self.assertLessEqual(title_bbox[3], atom_bbox[1])
            self.assertLessEqual(atom_bbox[3], drag_bbox[1])

            # Header title & subtitle share the exact white-fade fill color
            self.assertEqual(
                app.canvas.itemcget(title_ids[0], "fill"),
                app.canvas.itemcget(sub_ids[0], "fill"),
            )

            # Hovering over the atom center pauses frame advancement; leaving resumes it
            x_atom = int(round(0.5 * (atom_bbox[0] + atom_bbox[2])))
            y_atom = int(round(0.5 * (atom_bbox[1] + atom_bbox[3])))
            app._on_canvas_hover_motion(type("E", (), {"x": x_atom, "y": y_atom})())  # type: ignore[arg-type]
            self.assertTrue(app._welcome_hover_paused)
            paused_idx = app._welcome_frame_idx
            app._tick_welcome_animation()
            self.assertEqual(app._welcome_frame_idx, paused_idx)

            app._on_canvas_hover_leave(type("E", (), {})())  # type: ignore[arg-type]
            self.assertFalse(app._welcome_hover_paused)

            n_frames = len(app._welcome_photo_refs)
            self.assertGreaterEqual(n_frames, 72)
            # Header brightens then holds steady after step >= 36 while prompt keeps repeating
            fill_0 = app.canvas.itemcget(title_ids[0], "fill")
            for _ in range(16):
                app._tick_welcome_animation()
            fill_16 = app.canvas.itemcget(title_ids[0], "fill")
            self.assertNotEqual(fill_0, fill_16)
            for _ in range(20):
                app._tick_welcome_animation()
            fill_36 = app.canvas.itemcget(title_ids[0], "fill")
            prompt_36 = app.canvas.itemcget(drag_ids[0], "fill")
            for _ in range(16):
                app._tick_welcome_animation()
            fill_52 = app.canvas.itemcget(title_ids[0], "fill")
            prompt_52 = app.canvas.itemcget(drag_ids[0], "fill")
            self.assertEqual(fill_36, fill_52)
            self.assertNotEqual(prompt_36, prompt_52)
            self.assertIsNotNone(app._welcome_after_id)
        finally:
            app.destroy()
        self.assertIsNone(app._welcome_after_id)


if __name__ == "__main__":
    unittest.main()
