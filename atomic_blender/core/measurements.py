"""UI-agnostic selection math, focus/alpha computations, and geometry measurements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from atomic_blender.elements import METAL_ELEMENTS
from atomic_blender.viewport3d import atom_label


def compute_atom_alphas(
    n_atoms: int,
    selected: set[int],
    opacity: float,
    invert: bool,
) -> list[float]:
    """Compute per-atom alpha values from selection set, opacity slider, and invert flag."""
    if n_atoms <= 0:
        return []
    if not selected:
        return [1.0] * n_atoms
    op = max(0.05, min(1.0, float(opacity)))
    if invert:
        return [1.0 if i in selected else op for i in range(n_atoms)]
    return [op if i in selected else 1.0 for i in range(n_atoms)]


def compute_atom_radius_scales(
    n_atoms: int,
    selected: set[int],
    sel_scale: float = 1.0,
    unsel_scale: float = 1.0,
) -> list[float]:
    """Compute per-atom radius multipliers distinguishing focal and background atoms."""
    if n_atoms <= 0:
        return []
    if not selected:
        return [1.0] * n_atoms
    return [float(sel_scale) if i in selected else float(unsel_scale) for i in range(n_atoms)]


def find_neighbor_atoms(
    positions: Any,
    seed_indices: set[int],
    cutoff_a: float = 3.0,
) -> set[int]:
    """Return all atom indices within ``cutoff_a`` Angstroms of any seed atom (inclusive)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    valid_seeds = sorted(i for i in seed_indices if 0 <= i < n)
    if not valid_seeds or n == 0:
        return set(valid_seeds)
    cutoff = max(0.1, float(cutoff_a))
    seed_pts = pos[valid_seeds]
    diff = pos[:, None, :] - seed_pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    within = np.any(dists <= cutoff + 1e-6, axis=1)
    return set(np.flatnonzero(within).tolist()) | set(valid_seeds)


def auto_detect_adsorbate_indices(symbols: list[str], positions: Any) -> set[int]:
    """Heuristically identify surface/adsorbate atoms (non-metals above metal slab, or top 22% Z)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(symbols), len(pos))
    if n == 0:
        return set()
    metal_idx = [i for i in range(n) if symbols[i] in METAL_ELEMENTS]
    nonmetal_idx = [i for i in range(n) if symbols[i] not in METAL_ELEMENTS]
    if metal_idx and nonmetal_idx:
        z_med_metal = float(np.median(pos[metal_idx, 2]))
        above = {i for i in nonmetal_idx if float(pos[i, 2]) >= z_med_metal - 0.5}
        return above if above else set(nonmetal_idx)
    zs = pos[:n, 2]
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    if z_max - z_min < 1e-3:
        return {0}
    thresh = z_max - 0.22 * (z_max - z_min)
    top = {i for i in range(n) if float(zs[i]) >= thresh - 1e-6}
    return top if top else {int(np.argmax(zs))}


def compute_adsorbate_spotlight_alphas(
    symbols: list[str],
    positions: Any,
    focal_indices: set[int] | None = None,
    shell_cutoff_a: float = 3.2,
) -> tuple[set[int], list[float]]:
    """Compute focal spotlight + first coordination shell + Z-faded substrate alphas."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(symbols), len(pos))
    if n == 0:
        return set(), []
    focal = (
        {i for i in focal_indices if 0 <= i < n}
        if focal_indices
        else auto_detect_adsorbate_indices(symbols, pos)
    )
    if not focal:
        return set(), [1.0] * n
    shell = find_neighbor_atoms(pos[:n], focal, shell_cutoff_a) - focal
    zs = pos[:n, 2]
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    z_span = max(1e-3, z_max - z_min)

    alphas: list[float] = []
    for i in range(n):
        if i in focal:
            alphas.append(1.0)
        elif i in shell:
            alphas.append(0.85)
        else:
            z_norm = max(0.0, min(1.0, (float(zs[i]) - z_min) / z_span))
            alphas.append(round(0.15 + 0.28 * z_norm, 3))
    return focal, alphas


def compute_depth_fade_alphas(
    positions: Any,
    min_alpha: float = 0.18,
    max_alpha: float = 1.0,
) -> list[float]:
    """Compute smooth bottom-to-top Z-height opacity gradient across all atoms."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    if n == 0:
        return []
    zs = pos[:, 2]
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    span = z_max - z_min
    if span < 1e-6:
        return [float(max_alpha)] * n
    return [
        round(min_alpha + (max_alpha - min_alpha) * max(0.0, min(1.0, (float(z) - z_min) / span)), 3)
        for z in zs
    ]


def compute_selection_measurement(
    symbols: list[str],
    positions: Any,
    selected_order: list[int],
    reference_positions: Any = None,
) -> str:
    """Compute live coordinates, bond distance, angle, dihedral, or Δr from reference."""
    n = min(len(symbols), len(positions))
    order: list[int] = []
    seen: set[int] = set()
    for idx in selected_order:
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)

    count = len(order)
    if count == 0:
        return "Double-click 1–4 atoms for coords / distance / angle / dihedral"

    pos = np.asarray(positions, dtype=np.float64)
    labels = [atom_label(i, symbols[i]) for i in order]

    if count == 1:
        x, y, z = pos[order[0]]
        txt = f"{labels[0]}: ({x:.3f}, {y:.3f}, {z:.3f}) Å"
        if reference_positions is not None and order[0] < len(reference_positions):
            ref_p = np.asarray(reference_positions[order[0]], dtype=np.float64)
            delta = pos[order[0]] - ref_p
            dr = float(np.linalg.norm(delta))
            txt += f"  |  Δr {dr:.3f} Å (Δz {delta[2]:+.3f})"
        return txt

    if count == 2:
        delta = pos[order[1]] - pos[order[0]]
        dist = float(np.linalg.norm(delta))
        dz = abs(float(delta[2]))
        return f"Dist {labels[0]}–{labels[1]}: {dist:.3f} Å  (Δz {dz:.3f} Å)"

    if count == 3:
        v1 = pos[order[0]] - pos[order[1]]
        v2 = pos[order[2]] - pos[order[1]]
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 < 1e-9 or d2 < 1e-9:
            return f"Angle {labels[0]}–{labels[1]}–{labels[2]}: undefined (coincident atoms)"
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2) / (d1 * d2))))
        ang = math.degrees(math.acos(cos_a))
        return f"Angle {labels[0]}–{labels[1]}–{labels[2]}: {ang:.2f}°  ({d1:.3f}, {d2:.3f} Å)"

    if count == 4:
        p0, p1, p2, p3 = pos[order[0]], pos[order[1]], pos[order[2]], pos[order[3]]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = float(np.linalg.norm(b1))
        if b1_norm < 1e-9:
            return f"Dihedral {labels[0]}–{labels[1]}–{labels[2]}–{labels[3]}: undefined"
        b1_u = b1 / b1_norm
        v = b0 - np.dot(b0, b1_u) * b1_u
        w = b2 - np.dot(b2, b1_u) * b1_u
        if float(np.linalg.norm(v)) < 1e-9 or float(np.linalg.norm(w)) < 1e-9:
            return f"Dihedral {labels[0]}–{labels[1]}–{labels[2]}–{labels[3]}: collinear"
        x_val = float(np.dot(v, w))
        y_val = float(np.dot(np.cross(b1_u, v), w))
        dih = math.degrees(math.atan2(y_val, x_val))
        return f"Dihedral {labels[0]}–{labels[1]}–{labels[2]}–{labels[3]}: {dih:.2f}°"

    return f"{count} atoms selected (select 1–4 atoms to measure geometry)"
