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
                self.assertEqual(app.var_input.get(), str(cif))
                self.assertEqual(app.var_view_mode.get(), "file")
                self.assertEqual(app.var_view_file.get(), str(view_json))
                self.assertEqual(app.var_style.get(), str(style_json))

                # Verify queued drop dispatch inside a real mainloop tick (no WNDPROC GIL crash)
                app._queue.put(("drop", [str(struct_json)]))
                app.after(150, app.destroy)
                app.mainloop()
                self.assertEqual(app.var_input.get(), str(struct_json))
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

        self.assertEqual(compute_preview_resolution(1400, 1200, max_side=560), (560, 480))
        self.assertEqual(compute_preview_resolution(800, 1400, max_side=560), (320, 560))
        self.assertEqual(compute_preview_resolution(400, 300, max_side=560), (400, 300))

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
                self.assertEqual(req1["resolution"], [560, 480])
                self.assertEqual(req1["samples"], 10)
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


if __name__ == "__main__":
    unittest.main()
