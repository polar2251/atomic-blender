"""Lightweight NumPy 3D orthographic camera and scene model for the desktop GUI viewport.

Matches Blender's camera quaternion and orthographic framing math so views
rotated interactively in the desktop GUI reproduce 1:1 in Blender Cycles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from atomic_blender.elements import (
    COVALENT_RADII,
    resolve_element_color,
    resolve_element_radius,
)

Quat = tuple[float, float, float, float]


def quat_normalize(q: Quat) -> Quat:
    arr = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    arr /= norm
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def quat_mul(q1: Quat, q2: Quat) -> Quat:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return quat_normalize((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> Quat:
    ax = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    ax = ax / n
    half = 0.5 * angle_rad
    s = math.sin(half)
    return (math.cos(half), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s))


def quat_to_axes(q: Quat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera world-space ``(right, up, back)`` unit vectors from ``(w, x, y, z)``."""
    w, x, y, z = quat_normalize(q)
    right = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dtype=np.float64)
    up = np.array([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)], dtype=np.float64)
    back = np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dtype=np.float64)
    return right, up, back


def axes_to_quat(right: np.ndarray, up: np.ndarray, back: np.ndarray) -> Quat:
    """Convert orthonormal camera basis ``[right | up | back]`` into ``(w, x, y, z)``."""
    m = np.column_stack((right, up, back))
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize((w, x, y, z))


def direction_to_quat(view_dir: np.ndarray, up_hint: np.ndarray | None = None) -> Quat:
    """Compute camera quaternion looking from ``+view_dir`` toward the origin."""
    vd = np.asarray(view_dir, dtype=np.float64)
    vd /= max(1e-12, float(np.linalg.norm(vd)))
    forward = -vd
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_hint is None else np.asarray(up_hint, dtype=np.float64)
    up = up / max(1e-12, float(np.linalg.norm(up)))
    if abs(float(np.dot(forward, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0] if abs(forward[2]) > 0.9 else [0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= max(1e-12, float(np.linalg.norm(right)))
    cam_up = np.cross(right, forward)
    cam_up /= max(1e-12, float(np.linalg.norm(cam_up)))
    return axes_to_quat(right, cam_up, -forward)


def preset_to_quat(preset: str, cell_vecs: np.ndarray | None = None) -> Quat:
    """Match ``scene_builder.resolve_preset_orientation`` for ``iso``, ``side-oblique``, ``a``, ``b``, ``c``."""
    name = preset.lower().strip()
    if name == "iso":
        az, el = math.radians(35.0), math.radians(20.0)
        v = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float64)
        return direction_to_quat(v)
    if name == "side-oblique":
        az, el = math.radians(25.0), math.radians(10.0)
        v = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float64)
        return direction_to_quat(v)

    def _axis(idx: int, fallback: tuple[float, float, float]) -> np.ndarray:
        if cell_vecs is not None and len(cell_vecs) == 3:
            vec = np.asarray(cell_vecs[idx], dtype=np.float64)
            if float(np.linalg.norm(vec)) > 1e-6:
                return vec
        return np.array(fallback, dtype=np.float64)

    a = _axis(0, (1.0, 0.0, 0.0))
    b = _axis(1, (0.0, 1.0, 0.0))
    c = _axis(2, (0.0, 0.0, 1.0))
    if name == "a":
        return direction_to_quat(a, c)
    if name == "b":
        return direction_to_quat(b, c)
    if name == "c":
        return direction_to_quat(c, b)
    raise ValueError(f"Unknown preset: {preset!r}")


def orbit_quat(q: Quat, d_yaw_rad: float, d_pitch_rad: float) -> Quat:
    """Orbit camera quaternion smoothly around its local up and right axes."""
    right, up, _ = quat_to_axes(q)
    q_yaw = quat_from_axis_angle(up, -d_yaw_rad)
    q_next = quat_mul(q_yaw, q)
    right2, _, _ = quat_to_axes(q_next)
    q_pitch = quat_from_axis_angle(right2, -d_pitch_rad)
    return quat_mul(q_pitch, q_next)


def _shade_hex(hex_color: str, factor: float, mix_white: float = 0.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if mix_white > 0.0:
        r = int(round(r + (255 - r) * mix_white))
        g = int(round(g + (255 - g) * mix_white))
        b = int(round(b + (255 - b) * mix_white))
    r = max(0, min(255, int(round(r * factor))))
    g = max(0, min(255, int(round(g * factor))))
    b = max(0, min(255, int(round(b * factor))))
    return f"#{r:02X}{g:02X}{b:02X}"


def _find_bonds(
    symbols: list[str],
    positions: np.ndarray,
    bond_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial-hash bond detection matching ``scene_builder._find_bonds_spatial``."""
    mode = str(bond_cfg.get("mode", "table"))
    if mode == "none" or not bond_cfg.get("enabled", True) or len(symbols) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    table: dict[tuple[str, str], float] = {}
    for key, val in bond_cfg.get("cutoffs", {}).items():
        parts = [p.strip() for p in str(key).replace("_", "-").split("-") if p.strip()]
        if len(parts) == 2:
            c = float(val)
            table[(parts[0], parts[1])] = c
            table[(parts[1], parts[0])] = c

    tol = float(bond_cfg.get("covalent_tolerance", 1.2))
    pair_cutoffs: dict[tuple[str, str], float] = {}
    max_cut = 0.0
    unique_elems = sorted(set(symbols))
    for ei in unique_elems:
        for ej in unique_elems:
            if mode == "table":
                c = table.get((ei, ej), 0.0)
            elif mode == "auto":
                ri = COVALENT_RADII.get(ei, 0.75)
                rj = COVALENT_RADII.get(ej, 0.75)
                c = table.get((ei, ej), (ri + rj) * tol)
            else:
                c = table.get((ei, ej), 0.0)
            if c > 0.0:
                pair_cutoffs[(ei, ej)] = c
                max_cut = max(max_cut, c)

    if max_cut <= 0.0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    cell_size = max(max_cut, 0.5)
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, pos in enumerate(positions):
        key = (
            math.floor(float(pos[0]) / cell_size),
            math.floor(float(pos[1]) / cell_size),
            math.floor(float(pos[2]) / cell_size),
        )
        grid[key].append(idx)

    bi: list[int] = []
    bj: list[int] = []
    for (gx, gy, gz), bucket in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nbr = grid.get((gx + dx, gy + dy, gz + dz))
                    if not nbr:
                        continue
                    for i in bucket:
                        ei = symbols[i]
                        pi = positions[i]
                        for j in nbr:
                            if j <= i:
                                continue
                            cut = pair_cutoffs.get((ei, symbols[j]))
                            if cut is None:
                                continue
                            if float(np.linalg.norm(positions[j] - pi)) <= cut:
                                bi.append(i)
                                bj.append(j)
    return np.asarray(bi, dtype=np.int32), np.asarray(bj, dtype=np.int32)


@dataclass
class ViewportScene:
    """In-memory 3D representation and camera state shared with Blender Cycles."""

    symbols: list[str]
    positions: np.ndarray  # (N, 3)
    radii: np.ndarray  # (N,)
    colors: list[str]
    outline_colors: list[str]
    highlight_colors: list[str]
    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_color_mode: str
    single_bond_color: str
    bond_radius: float
    cell_edges: np.ndarray  # (E, 2, 3)
    cell_color: str
    has_cell: bool
    origin: np.ndarray  # (3,)
    cell_vecs: np.ndarray  # (3, 3)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    target: np.ndarray = None  # type: ignore[assignment]
    ortho_scale: float = 20.0
    margin: float = 1.08

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_neutral_dict(
        cls,
        structure: dict[str, Any],
        style: dict[str, Any],
        radius_scale: float = 1.0,
        bonds_mode: str | None = None,
        bond_color: str | None = None,
        show_cell: bool = True,
    ) -> ViewportScene:
        symbols = list(structure["symbols"])
        positions = np.asarray(structure["positions"], dtype=np.float64).reshape(-1, 3)
        has_cell = bool(structure.get("has_cell", True))
        origin = np.asarray(structure.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        cell_vecs = np.asarray(structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=np.float64)

        eff_scale = float(style.get("elements", {}).get("radius_scale", 1.0)) * float(radius_scale)
        elem_r: dict[str, float] = {}
        elem_c: dict[str, str] = {}
        elem_out: dict[str, str] = {}
        elem_hi: dict[str, str] = {}
        for sym in set(symbols):
            c = resolve_element_color(sym, style)
            elem_c[sym] = c
            elem_out[sym] = _shade_hex(c, 0.62)
            elem_hi[sym] = _shade_hex(c, 1.0, mix_white=0.42)
            elem_r[sym] = resolve_element_radius(sym, style) * eff_scale

        radii = np.array([elem_r[s] for s in symbols], dtype=np.float64)
        colors = [elem_c[s] for s in symbols]
        outline_colors = [elem_out[s] for s in symbols]
        highlight_colors = [elem_hi[s] for s in symbols]

        bond_cfg = dict(style.get("bonds", {}))
        if bonds_mode and bonds_mode != "style":
            bond_cfg["mode"] = bonds_mode
            bond_cfg["enabled"] = bonds_mode != "none"
        bond_i, bond_j = _find_bonds(symbols, positions, bond_cfg)

        b_color_mode = (
            bond_color if bond_color and bond_color != "style" else str(bond_cfg.get("color_mode", "single"))
        )
        single_bond_color = str(bond_cfg.get("color", "#9CA3AF"))
        bond_radius = float(bond_cfg.get("radius", 0.12))

        cell_cfg = style.get("cell", {})
        cell_enabled = show_cell and bool(cell_cfg.get("enabled", True))
        if has_cell and cell_enabled:
            a, b, c = cell_vecs[0], cell_vecs[1], cell_vecs[2]
            corners = np.array([
                origin,
                origin + a,
                origin + a + b,
                origin + b,
                origin + c,
                origin + a + c,
                origin + a + b + c,
                origin + b + c,
            ], dtype=np.float64)
            edge_idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            cell_edges = np.array([[corners[i], corners[j]] for i, j in edge_idx], dtype=np.float64)
        else:
            cell_edges = np.empty((0, 2, 3), dtype=np.float64)

        cam_cfg = style.get("camera", {})
        return cls(
            symbols=symbols,
            positions=positions,
            radii=radii,
            colors=colors,
            outline_colors=outline_colors,
            highlight_colors=highlight_colors,
            bond_i=bond_i,
            bond_j=bond_j,
            bond_color_mode=b_color_mode,
            single_bond_color=single_bond_color,
            bond_radius=bond_radius,
            cell_edges=cell_edges,
            cell_color=str(cell_cfg.get("color", "#111111")),
            has_cell=has_cell,
            origin=origin,
            cell_vecs=cell_vecs,
            margin=float(cam_cfg.get("margin", 1.08)),
        )

    def _framing_points(self, crop_z: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
        if self.has_cell:
            a, b, c = self.cell_vecs[0], self.cell_vecs[1], self.cell_vecs[2]
            if crop_z is not None and len(self.positions) > 0:
                pad_bot, pad_top = crop_z
                zs = self.positions[:, 2]
                zmin = float(np.min(zs) - pad_bot)
                zmax = float(np.max(zs) + pad_top)
                base = [self.origin, self.origin + a, self.origin + a + b, self.origin + b]
                pts = np.array(
                    [[p[0], p[1], zmin] for p in base] + [[p[0], p[1], zmax] for p in base],
                    dtype=np.float64,
                )
                cx = sum(float(p[0]) for p in base) * 0.25
                cy = sum(float(p[1]) for p in base) * 0.25
                return np.array([cx, cy, 0.5 * (zmin + zmax)], dtype=np.float64), pts

            corners = np.array([
                self.origin,
                self.origin + a,
                self.origin + a + b,
                self.origin + b,
                self.origin + c,
                self.origin + a + c,
                self.origin + a + b + c,
                self.origin + b + c,
            ], dtype=np.float64)
            return self.origin + 0.5 * (a + b + c), corners

        if len(self.positions) == 0:
            return np.zeros(3, dtype=np.float64), np.zeros((1, 3), dtype=np.float64)

        mins = np.min(self.positions - self.radii[:, None], axis=0)
        maxs = np.max(self.positions + self.radii[:, None], axis=0)
        if crop_z is not None:
            mins[2] = float(np.min(self.positions[:, 2]) - crop_z[0])
            maxs[2] = float(np.max(self.positions[:, 2]) + crop_z[1])
        target = 0.5 * (mins + maxs)
        pts = np.array(
            [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
            dtype=np.float64,
        )
        return target, pts

    def fit_camera(
        self,
        quat: Quat | None = None,
        aspect: float = 1.0,
        crop_z: tuple[float, float] | None = None,
    ) -> None:
        """Compute exact target and ``ortho_scale`` matching Blender's framing."""
        if quat is not None:
            self.quat = quat_normalize(quat)
        target, pts = self._framing_points(crop_z)
        self.target = target
        right, up, _ = quat_to_axes(self.quat)
        rel = pts - target
        xs_c = np.max(np.abs(rel @ right)) if len(pts) else 1.0
        ys_c = np.max(np.abs(rel @ up)) if len(pts) else 1.0
        half_w = max(float(xs_c), 0.5) * self.margin
        half_h = max(float(ys_c), 0.5) * self.margin
        self.ortho_scale = max(2.0 * half_w, 2.0 * half_h * max(0.05, aspect))

    def orbit(self, d_yaw_rad: float, d_pitch_rad: float) -> None:
        self.quat = orbit_quat(self.quat, d_yaw_rad, d_pitch_rad)

    def pan(self, dx_angstrom: float, dy_angstrom: float) -> None:
        right, up, _ = quat_to_axes(self.quat)
        self.target = self.target - right * dx_angstrom + up * dy_angstrom

    def zoom(self, factor: float) -> None:
        self.ortho_scale = max(0.5, min(2000.0, self.ortho_scale * factor))

    def to_view_dict(
        self,
        res_x: int,
        res_y: int,
        base_distance: float = 200.0,
        base_clip_end: float = 1000.0,
    ) -> dict[str, Any]:
        """Export current viewport camera in ``view.json`` format for Cycles."""
        _, _, back = quat_to_axes(self.quat)
        if len(self.positions) > 0:
            max_ext = float(np.max(np.linalg.norm(self.positions - self.target, axis=1)))
        else:
            max_ext = 10.0
        max_ext = max(max_ext, self.ortho_scale * 0.6)
        dist = max(base_distance, max_ext * 4.0)
        loc = self.target + back * dist
        return {
            "schema_version": 1,
            "camera_type": "ORTHO",
            "location": [float(loc[0]), float(loc[1]), float(loc[2])],
            "rotation_quaternion": [float(v) for v in self.quat],
            "ortho_scale": float(self.ortho_scale),
            "lens": 50.0,
            "clip_start": 1.0,
            "clip_end": float(max(base_clip_end, dist + max_ext * 4.0)),
            "resolution": [int(res_x), int(res_y)],
            "fit_to_scene": False,
        }
