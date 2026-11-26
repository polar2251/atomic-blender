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
from atomic_blender.elements import resolve_element_color, resolve_element_radius
from atomic_blender.gui import GuiRenderConfig, build_render_cli_args, validate_gui_config


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
        pkg_style = package_dir() / "styles" / "default.json"
        root_style = package_root() / "styles" / "default.json"
        self.assertTrue(pkg_style.is_file())
        self.assertTrue(root_style.is_file())
        self.assertEqual(pkg_style.read_text(encoding="utf-8"), root_style.read_text(encoding="utf-8"))

        pkg_addon = package_dir() / "addon" / "atomic_blender_studio" / "__init__.py"
        root_addon = package_root() / "addon" / "atomic_blender_studio" / "__init__.py"
        self.assertTrue(pkg_addon.is_file())
        self.assertTrue(root_addon.is_file())
        self.assertEqual(pkg_addon.read_text(encoding="utf-8"), root_addon.read_text(encoding="utf-8"))

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
                resolution=(1200, 900),
                samples=32,
            )
            self.assertEqual(validate_gui_config(cfg), struct_file)
            cli_args = build_render_cli_args(cfg, save_blend=Path(tmp) / "scene.blend")
            self.assertEqual(cli_args[0], "render")
            self.assertIn("--fit-view", cli_args)
            self.assertIn("--no-wrap", cli_args)
            self.assertIn("--crop-z", cli_args)
            self.assertIn("auto", cli_args)
            self.assertIn("--type-map", cli_args)
            self.assertIn("1:Al,2:C", cli_args)
            self.assertIn("--bonds", cli_args)
            self.assertIn("auto", cli_args)
            self.assertIn("--save-blend", cli_args)
            # Confirm build_parser parses generated CLI args cleanly
            parsed = parser.parse_args(cli_args)
            self.assertEqual(parsed.replicate, [2, 2, 1])
            self.assertEqual(parsed.resolution, [1200, 900])
            self.assertEqual(parsed.samples, 32)


if __name__ == "__main__":
    unittest.main()
