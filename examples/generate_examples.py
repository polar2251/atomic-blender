"""Generate public example structures with ASE and render preview PNGs."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111, molecule
from ase.spacegroup import crystal
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


def generate_feature_test_structures(examples_dir: Path) -> list[Path]:
    """Create 5 test structures covering Spotlight, H-bonds, Ghost overlay, Colormaps, and Chips."""
    created: list[Path] = []

    # 1. CH3OH + coadsorbed surface *O on 4-layer Cu(111) slab (with O-H...O* H-bond)
    cu_slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0)
    top_z = float(np.max(cu_slab.positions[:, 2]))
    cell = cu_slab.cell
    # Pick a clean interior top Cu site + nearby hollow for O*
    cu_top = [i for i, p in enumerate(cu_slab.positions) if p[2] > top_z - 0.3]
    anchor = cu_slab.positions[cu_top[4]]  # center Cu atom
    o_surf = np.array([anchor[0] + 1.85, anchor[1] + 0.90, top_z + 1.15])
    # Place hydroxyl H pointing ~1.88 A from coadsorbed surface O*
    oh_o = np.array([anchor[0] + 0.15, anchor[1] + 0.10, top_z + 2.05])
    dir_to_o = o_surf - oh_o
    dir_u = dir_to_o / max(1e-6, float(np.linalg.norm(dir_to_o)))
    oh_h = oh_o + dir_u * 0.98
    c_pos = oh_o + np.array([-0.95, -0.55, 0.85])
    # Add 3 methyl H around C
    c_to_o = oh_o - c_pos
    c_u = c_to_o / float(np.linalg.norm(c_to_o))
    perp1 = np.array([-c_u[1], c_u[0], 0.0])
    if np.linalg.norm(perp1) < 1e-4:
        perp1 = np.array([1.0, 0.0, 0.0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(c_u, perp1)
    h_methyl = [
        c_pos - 0.36 * c_u + 1.03 * (np.cos(ang) * perp1 + np.sin(ang) * perp2)
        for ang in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    ]
    ads_1 = Atoms(
        ["O", "C", "O", "H", "H", "H", "H"],
        positions=[o_surf, c_pos, oh_o, oh_h, h_methyl[0], h_methyl[1], h_methyl[2]],
        cell=cell,
        pbc=True,
    )
    sys_1 = cu_slab + ads_1
    p1 = examples_dir / "01_adsorbate_spotlight_CH3OH_Cu111.vasp"
    ase.io.write(str(p1), sys_1, format="vasp", direct=False, sort=False)
    created.append(p1)

    # 2. Cyclic (H2O)5(NH3) hydrogen-bonded 6-ring (O-H...O and O-H...N distances ~1.82-1.92 A)
    ring_r = 2.72
    ring_syms: list[str] = []
    ring_pos: list[np.ndarray] = []
    n_sites = 6
    centers = [
        np.array([ring_r * np.cos(2 * np.pi * k / n_sites), ring_r * np.sin(2 * np.pi * k / n_sites), 0.15 * ((-1) ** k)])
        for k in range(n_sites)
    ]
    for k in range(n_sites):
        p_cur = centers[k]
        p_nxt = centers[(k + 1) % n_sites]
        v_bond = p_nxt - p_cur
        u_bond = v_bond / float(np.linalg.norm(v_bond))
        v_out = p_cur / max(1e-6, float(np.linalg.norm(p_cur[:2])))
        v_out[2] = 0.35 * ((-1) ** (k + 1))
        v_out /= float(np.linalg.norm(v_out))
        if k == 0:
            # NH3 donor/acceptor node
            ring_syms.extend(["N", "H", "H", "H"])
            ring_pos.append(p_cur)
            ring_pos.append(p_cur + u_bond * 1.01)
            ring_pos.append(p_cur + (0.75 * v_out + np.array([0.0, 0.0, 0.65])) / np.linalg.norm(0.75 * v_out + np.array([0.0, 0.0, 0.65])) * 1.01)
            ring_pos.append(p_cur + (0.75 * v_out + np.array([0.0, 0.0, -0.65])) / np.linalg.norm(0.75 * v_out + np.array([0.0, 0.0, -0.65])) * 1.01)
        else:
            ring_syms.extend(["O", "H", "H"])
            ring_pos.append(p_cur)
            ring_pos.append(p_cur + u_bond * 0.97)
            ring_pos.append(p_cur + v_out * 0.96)
    mol_ring = Atoms(ring_syms, positions=ring_pos)
    mol_ring.center(vacuum=4.0)
    p2 = examples_dir / "02_hbond_ring_water_ammonia.xyz"
    ase.io.write(str(p2), mol_ring, format="xyz")
    created.append(p2)

    # 3. Matched initial/final relaxation pair on Pt(111) for Ghost Overlay & Batch Render
    pt_init = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    co_i = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    add_adsorbate(pt_init, co_i, height=1.90, position="ontop")
    p3_i = examples_dir / "03_relax_initial_CO_Pt111.vasp"
    ase.io.write(str(p3_i), pt_init, format="vasp", direct=False, sort=False)
    created.append(p3_i)

    pt_final = pt_init.copy()
    # Buckle top Pt layer slightly (~0.28 A) and diffuse CO toward bridge/hollow (~1.60 A)
    top_pt_z = float(np.max(pt_init.positions[:-2, 2]))
    for idx in range(len(pt_final) - 2):
        if pt_final.positions[idx, 2] > top_pt_z - 0.3:
            pt_final.positions[idx, 2] += 0.28 if (idx % 2 == 0) else -0.10
    pt_final.positions[-2] += np.array([1.45, 0.78, -0.25])  # C moves
    pt_final.positions[-1] += np.array([1.68, 0.92, -0.12])  # O tilts
    p3_f = examples_dir / "03_relax_final_CO_Pt111.vasp"
    ase.io.write(str(p3_f), pt_final, format="vasp", direct=False, sort=False)
    created.append(p3_f)

    # 4. Tetragonal BaTiO3 2x2x2 perovskite supercell for Element Chips / Expand Shell / Z Range / Polyhedra
    bto = crystal(
        ["Ba", "Ti", "O", "O"],
        basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.514), (0.5, 0.5, 0.025), (0.5, 0.0, 0.486)],
        spacegroup=99,
        cellpar=[3.992, 3.992, 4.036, 90, 90, 90],
    ).repeat((2, 2, 2))
    p4 = examples_dir / "04_perovskite_BaTiO3_2x2x2.cif"
    ase.io.write(str(p4), bto)
    created.append(p4)

    # 5. 5-frame NEB O-adatom surface diffusion on a tilted hexagonal Cu(111) slab (gamma=120 deg)
    neb_base = fcc111("Cu", size=(3, 3, 3), vacuum=6.0)
    top_z = float(np.max(neb_base.positions[:, 2]))
    a1 = neb_base.cell[0]
    a2 = neb_base.cell[1]
    p_start = 0.28 * a1 + 0.28 * a2 + np.array([0.0, 0.0, top_z + 1.45])
    p_end = 0.62 * a1 + 0.28 * a2 + np.array([0.0, 0.0, top_z + 1.45])
    neb_frames: list[Atoms] = []
    for k in range(5):
        t = k / 4.0
        # Slight transition-state barrier arc (+0.55 A at midpoint t=0.5)
        pos_o = (1.0 - t) * p_start + t * p_end + np.array([0.0, 0.0, 0.55 * math.sin(math.pi * t)])
        frame = neb_base.copy()
        frame += Atoms("O", positions=[pos_o])
        neb_frames.append(frame)
    p5 = examples_dir / "05_neb_diffusion_5frames.xyz"
    ase.io.write(str(p5), neb_frames, format="extxyz")
    created.append(p5)

    # 6. Tilted hexagonal 2H-MoS2 3x3 monolayer (gamma=120 deg) for Tilted Cell + Trigonal Prism Polyhedra
    a_mos2 = 3.16
    mos2_cell = [
        [a_mos2, 0.0, 0.0],
        [-0.5 * a_mos2, 0.5 * math.sqrt(3.0) * a_mos2, 0.0],
        [0.0, 0.0, 14.0],
    ]
    mos2 = Atoms(
        "MoS2",
        scaled_positions=[(1.0 / 3.0, 2.0 / 3.0, 0.50), (2.0 / 3.0, 1.0 / 3.0, 0.39), (2.0 / 3.0, 1.0 / 3.0, 0.61)],
        cell=mos2_cell,
        pbc=[True, True, True],
    ).repeat((3, 3, 1))
    p6 = examples_dir / "06_tilted_hexagonal_MoS2_3x3.cif"
    ase.io.write(str(p6), mos2)
    created.append(p6)
    return created


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    generate_feature_test_structures(examples_dir)
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
