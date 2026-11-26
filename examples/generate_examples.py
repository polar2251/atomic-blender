"""Generate public example structures with ASE and render preview PNGs."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111, molecule
import ase.io

from atomic_blender.cli import main as cli_main


def make_water_box(seed: int = 42) -> Atoms:
    """Create a small cubic periodic box containing 12 water molecules."""
    rng = np.random.default_rng(seed)
    box_len = 9.0
    water = molecule("H2O")
    water.center(about=(0.0, 0.0, 0.0))

    centers: list[np.ndarray] = []
    combined = Atoms(cell=[box_len, box_len, box_len], pbc=True)
    attempts = 0
    while len(centers) < 12 and attempts < 2000:
        attempts += 1
        cand = rng.uniform(1.0, box_len - 1.0, size=3)
        if all(np.linalg.norm(cand - c) > 2.6 for c in centers):
            centers.append(cand)
            mol = water.copy()
            angles = rng.uniform(0.0, 360.0, size=3)
            mol.rotate(angles[0], "x")
            mol.rotate(angles[1], "y")
            mol.rotate(angles[2], "z")
            mol.translate(cand)
            combined += mol
    return combined


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    renders_dir = examples_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. FCC Cu bulk (3x3x3 supercell)
    cu_bulk = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    cu_path = examples_dir / "fcc_cu_bulk.cif"
    ase.io.write(str(cu_path), cu_bulk)

    # 2. Small periodic water box (extxyz with Lattice)
    water_box = make_water_box()
    water_path = examples_dir / "water_box.xyz"
    ase.io.write(str(water_path), water_box, format="extxyz")

    # 3. CO molecule adsorbed on a 4-layer Pt(111) slab (POSCAR/VASP)
    slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    co = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])  # C down, O up
    add_adsorbate(slab, co, height=1.85, position="ontop")
    slab_path = examples_dir / "co_on_pt111.vasp"
    ase.io.write(str(slab_path), slab, format="vasp", direct=False, sort=False)

    # Render compact preview PNGs for README
    jobs = [
        (cu_path, renders_dir / "fcc_cu_bulk.png", "iso", None, "640", "640"),
        (water_path, renders_dir / "water_box.png", "iso", None, "640", "640"),
        (slab_path, renders_dir / "co_on_pt111.png", "side-oblique", "auto", "700", "600"),
    ]
    for src, out_png, view, crop_z, w, h in jobs:
        args = [
            "render",
            str(src),
            "--style",
            "default",
            "--view",
            view,
            "--resolution",
            w,
            h,
            "--samples",
            "48",
            "--out",
            str(out_png),
        ]
        if crop_z:
            args.extend(["--crop-z", crop_z])
        rc = cli_main(args)
        if rc != 0:
            raise RuntimeError(f"Failed rendering example {src.name}")


if __name__ == "__main__":
    main()
