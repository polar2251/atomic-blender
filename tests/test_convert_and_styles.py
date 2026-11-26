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


if __name__ == "__main__":
    unittest.main()
