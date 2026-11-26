"""Lightweight NumPy/Pillow 3D orthographic camera and shaded viewport renderer.

Matches Blender's camera quaternion and orthographic framing math so views
rotated interactively in the desktop GUI reproduce 1:1 in Blender Cycles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from atomic_blender.elements import (
    COVALENT_RADII,
    METAL_ELEMENTS,
    resolve_element_color,
    resolve_element_radius,
)

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _shade_hex(hex_color: str, factor: float, mix_white: float = 0.0) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    if mix_white > 0.0:
        r = int(round(r + (255 - r) * mix_white))
        g = int(round(g + (255 - g) * mix_white))
        b = int(round(b + (255 - b) * mix_white))
    r = max(0, min(255, int(round(r * factor))))
    g = max(0, min(255, int(round(g * factor))))
    b = max(0, min(255, int(round(b * factor))))
    return f"#{r:02X}{g:02X}{b:02X}"


def _shade_rgba_tuple(hex_color: str, factor: float, mix_white: float = 0.0) -> tuple[int, int, int, int]:
    r, g, b = _hex_to_rgb(_shade_hex(hex_color, factor, mix_white))
    return (r, g, b, 255)


def build_sphere_sprite(
    hex_color: str,
    roughness: float = 0.35,
    specular: float = 0.4,
    metallic: float = 0.0,
    subsurface_weight: float = 0.0,
    coat_weight: float = 0.0,
    ao_factor: float = 0.0,
    has_rim: bool = False,
    depth_shade: float = 1.0,
    size: int = 88,
) -> Any:
    """Render an anti-aliased 3D PBR-shaded sphere sprite as a ``PIL.Image`` (RGBA)."""
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    u, v = np.meshgrid(lin, lin)
    r2 = u * u + v * v
    dist = np.sqrt(r2)
    nz = np.sqrt(np.maximum(0.0, 1.0 - r2))
    nx, ny = u, -v

    # Anti-aliased sub-pixel alpha at the silhouette edge
    alpha = np.clip((1.0 - dist) * (size * 0.5) + 0.5, 0.0, 1.0)

    r_s, g_s, b_s = _hex_to_rgb(hex_color)
    base_rgb = np.array([r_s / 255.0, g_s / 255.0, b_s / 255.0], dtype=np.float32) ** 2.2

    # Key Sun light (top-left front), Fill light (lower-right), View (0, 0, 1)
    l1 = np.array([-0.42, 0.54, 0.73], dtype=np.float32)
    l1 /= float(np.linalg.norm(l1))
    l2 = np.array([0.62, -0.35, 0.70], dtype=np.float32)
    l2 /= float(np.linalg.norm(l2))

    ndotl1_raw = nx * l1[0] + ny * l1[1] + nz * l1[2]
    ndotl1 = np.maximum(0.0, ndotl1_raw)
    ndotl2 = np.maximum(0.0, nx * l2[0] + ny * l2[1] + nz * l2[2])

    # Wrap lighting simulates subtle subsurface scattering on non-metals
    sss = float(np.clip(subsurface_weight * 1.4, 0.0, 0.45))
    diff1 = np.maximum(0.0, (ndotl1_raw + sss) / (1.0 + sss))

    # Grazing edge darkening (AO / spherical curvature) + Fresnel rim light
    ao_rim = 1.0 - float(np.clip(ao_factor, 0.0, 0.8)) * 0.45 * ((1.0 - nz) ** 1.4)
    rim_boost = (0.28 if has_rim else 0.10) * ((1.0 - nz) ** 2.8)

    ambient = 0.30 * (1.0 - 0.30 * metallic)
    diffuse_weight = 1.0 - 0.45 * metallic
    lit_factor = (ambient + diffuse_weight * (0.72 * diff1 + 0.24 * ndotl2)) * ao_rim * depth_shade

    # Blinn-Phong specular highlight
    h_vec = l1 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    h_vec /= float(np.linalg.norm(h_vec))
    ndoth = np.maximum(0.0, nx * h_vec[0] + ny * h_vec[1] + nz * h_vec[2])
    shininess = max(8.0, 220.0 * ((1.0 - float(np.clip(roughness, 0.05, 0.95))) ** 2))
    spec = (ndoth ** shininess) * float(specular) * (0.70 + 0.60 * metallic)

    if coat_weight > 0.0:
        spec = spec + (ndoth ** 95.0) * float(coat_weight) * 0.38

    spec_tint = (1.0 - metallic) * np.array([1.0, 1.0, 1.0], dtype=np.float32) + metallic * np.clip(
        base_rgb * 1.35, 0.0, 1.0
    )

    rgb_lin = (
        base_rgb[None, None, :] * lit_factor[:, :, None]
        + spec_tint[None, None, :] * spec[:, :, None]
        + (0.7 * base_rgb + 0.3)[None, None, :] * rim_boost[:, :, None] * depth_shade
    )
    rgb_lin = rgb_lin / (1.0 + 0.12 * rgb_lin)
    rgb_srgb = np.clip(np.power(np.clip(rgb_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
    alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack((rgb_srgb, alpha_u8))
    return Image.fromarray(rgba, mode="RGBA")


def _needs_light_outline(symbol: str, hex_color: str) -> bool:
    r, g, b = _hex_to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return symbol == "H" or luminance > 215.0


def _needs_dark_rim_outline(hex_color: str) -> bool:
    r, g, b = _hex_to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 75.0


def _load_overlay_font(px_size: int) -> Any:
    """Load a crisp sans-serif bold font for axes and atom labels."""
    if not _HAS_PIL:
        return None
    for candidate in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, max(8, int(px_size)))
        except Exception:
            continue
    return ImageFont.load_default()


def render_reference_axes_overlay(
    target_img: Any,
    quat: Quat,
    axes_pos: tuple[float, float] = (0.12, 0.86),
    show_labels: bool = True,
    dark_backdrop: bool = True,
    arm_px: float | None = None,
) -> tuple[float, float, float]:
    """Draw a 3D depth-sorted Reference Axes (x, y, z) gizmo onto ``target_img``."""
    tw, th = target_img.size
    u = max(0.05, min(0.95, float(axes_pos[0])))
    v = max(0.05, min(0.95, float(axes_pos[1])))
    cx = u * tw
    cy = v * th

    arm_len = float(arm_px) if arm_px is not None else max(24.0, min(58.0, min(tw, th) * 0.068))
    if not _HAS_PIL:
        return cx, cy, max(34.0, arm_len * 1.15)

    ss = 1
    box_half = int(math.ceil(arm_len * 1.55))
    size_ss = box_half * 2 * ss
    badge = Image.new("RGBA", (size_ss, size_ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    oc = box_half * ss

    disc_r = arm_len * 1.30 * ss
    if dark_backdrop:
        disc_fill = (18, 22, 28, 155)
        disc_edge = (94, 108, 128, 165)
        hub_fill = (226, 232, 240, 245)
        stroke_col = (15, 18, 24, 235)
    else:
        disc_fill = (244, 246, 250, 175)
        disc_edge = (148, 163, 184, 185)
        hub_fill = (71, 85, 105, 245)
        stroke_col = (15, 23, 42, 220)

    draw.ellipse((oc - disc_r, oc - disc_r, oc + disc_r, oc + disc_r), fill=disc_fill, outline=disc_edge, width=1)

    right, up, back = quat_to_axes(quat)
    basis_defs = (
        ("x", np.array([1.0, 0.0, 0.0]), (239, 68, 68), (252, 165, 165)),
        ("y", np.array([0.0, 1.0, 0.0]), (34, 197, 94), (134, 239, 172)),
        ("z", np.array([0.0, 0.0, 1.0]), (59, 130, 246), (147, 197, 253)),
    )
    projected = []
    for lbl, vec, rgb, text_rgb in basis_defs:
        dx = float(np.dot(vec, right))
        dy = -float(np.dot(vec, up))
        dz = float(np.dot(vec, back))
        projected.append((dz, lbl, dx, dy, rgb, text_rgb))
    projected.sort(key=lambda item: item[0])

    len_ss = arm_len * ss
    shaft_w = max(2, int(round(arm_len * 0.125 * ss)))
    head_half_w = max(3.5, arm_len * 0.145 * ss)
    label_font = _load_overlay_font(max(11, int(round(arm_len * 0.36 * ss)))) if show_labels else None

    for dz, lbl, dx, dy, rgb, text_rgb in projected:
        depth_bright = 0.72 + 0.28 * ((dz + 1.0) * 0.5)
        col = tuple(max(0, min(255, int(round(c * depth_bright)))) for c in rgb) + (245,)
        plen = math.hypot(dx, dy)
        tip_x = oc + dx * len_ss
        tip_y = oc + dy * len_ss

        if plen > 0.08:
            ux, uy = dx / plen, dy / plen
            px, py = -uy, ux
            shaft_frac = 0.74
            sx = oc + dx * len_ss * shaft_frac
            sy = oc + dy * len_ss * shaft_frac
            draw.line((oc, oc, sx, sy), fill=col, width=shaft_w)

            base_x = oc + dx * len_ss * 0.70
            base_y = oc + dy * len_ss * 0.70
            draw.polygon(
                [
                    (tip_x, tip_y),
                    (base_x + px * head_half_w, base_y + py * head_half_w),
                    (base_x - px * head_half_w, base_y - py * head_half_w),
                ],
                fill=col,
            )
            lbl_x = oc + dx * (len_ss + arm_len * 0.24 * ss)
            lbl_y = oc + dy * (len_ss + arm_len * 0.24 * ss)
        else:
            r_end = shaft_w * 0.85
            draw.ellipse((tip_x - r_end, tip_y - r_end, tip_x + r_end, tip_y + r_end), fill=col)
            lbl_x = oc + (arm_len * 0.24 * ss)
            lbl_y = oc - (arm_len * 0.24 * ss)

        if show_labels and label_font is not None:
            bbox = draw.textbbox((0, 0), lbl, font=label_font)
            tw_txt = bbox[2] - bbox[0]
            th_txt = bbox[3] - bbox[1]
            tx = lbl_x - tw_txt * 0.5 - bbox[0]
            ty = lbl_y - th_txt * 0.5 - bbox[1]
            draw.text(
                (tx, ty),
                lbl,
                font=label_font,
                fill=text_rgb + (255,),
                stroke_width=1,
                stroke_fill=stroke_col,
            )

    hub_r = max(2.5, arm_len * 0.11 * ss)
    draw.ellipse((oc - hub_r, oc - hub_r, oc + hub_r, oc + hub_r), fill=hub_fill, outline=stroke_col, width=1)

    _blit_rgba(target_img, badge, int(round(cx - box_half)), int(round(cy - box_half)))
    return cx, cy, max(36.0, arm_len * 1.25)


def _apply_thin_outline(
    sprite: Any,
    outline_rgb: tuple[int, int, int] = (115, 120, 130),
    strength: float = 0.65,
) -> Any:
    """Apply a very thin subtle outer outline so white/H spheres stay crisp on light backgrounds."""
    w, h = sprite.size
    if w < 6 or h < 6:
        return sprite
    arr = np.array(sprite, dtype=np.float32)
    coords = np.arange(w, dtype=np.float32) - (w - 1) * 0.5
    u, v = np.meshgrid(coords, coords)
    dist_px = np.sqrt(u * u + v * v)
    r_px = w * 0.5
    ring = np.clip(1.0 - np.abs(dist_px - (r_px - 0.9)) / 0.9, 0.0, 1.0)
    visible = arr[:, :, 3] > 20.0
    blend = (ring * strength * visible)[:, :, None]
    if not np.any(blend):
        return sprite
    target = np.array(outline_rgb, dtype=np.float32).reshape(1, 1, 3)
    arr[:, :, :3] = arr[:, :, :3] * (1.0 - blend) + target * blend
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGBA")


def _blit_rgba(canvas: Any, sprite: Any, x0: int, y0: int) -> None:
    """Composite ``sprite`` onto ``canvas`` at ``(x0, y0)`` clipping to canvas bounds."""
    cw, ch = canvas.size
    sw, sh = sprite.size
    if x0 >= cw or y0 >= ch or x0 + sw <= 0 or y0 + sh <= 0:
        return
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    sx1 = min(sw, cw - x0)
    sy1 = min(sh, ch - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    if sx0 == 0 and sy0 == 0 and sx1 == sw and sy1 == sh:
        canvas.alpha_composite(sprite, dest=(dx0, dy0))
    else:
        canvas.alpha_composite(sprite.crop((sx0, sy0, sx1, sy1)), dest=(dx0, dy0))


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
    _base_sprites: dict[tuple[str, int], Any] = field(default_factory=dict)
    _resized_sprites: dict[tuple[str, int, int, bool], Any] = field(default_factory=dict)
    _outline_symbols: set[str] = field(default_factory=set)
    _dark_outline_symbols: set[str] = field(default_factory=set)
    show_atoms: bool = True
    show_labels: bool = False
    show_axes: bool = True
    axes_labels: bool = True
    axes_pos: tuple[float, float] = (0.12, 0.86)
    dark_backdrop: bool = True

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
        show_atoms: bool = True,
        show_labels: bool = False,
        show_axes: bool = True,
        axes_labels: bool = True,
        axes_pos: tuple[float, float] = (0.12, 0.86),
        dark_backdrop: bool = True,
    ) -> ViewportScene:
        symbols = list(structure["symbols"])
        positions = np.asarray(structure["positions"], dtype=np.float64).reshape(-1, 3)
        has_cell = bool(structure.get("has_cell", True))
        origin = np.asarray(structure.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        cell_vecs = np.asarray(structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=np.float64)

        eff_scale = float(style.get("elements", {}).get("radius_scale", 1.0)) * float(radius_scale)
        mat_cfg = style.get("material", {})
        has_rim = isinstance(style.get("lighting", {}).get("rim"), dict)

        elem_r: dict[str, float] = {}
        elem_c: dict[str, str] = {}
        elem_out: dict[str, str] = {}
        elem_hi: dict[str, str] = {}
        base_sprites: dict[tuple[str, int], Any] = {}
        outline_syms: set[str] = set()
        dark_outline_syms: set[str] = set()

        for sym in set(symbols):
            c = resolve_element_color(sym, style)
            elem_c[sym] = c
            elem_out[sym] = _shade_hex(c, 0.62)
            elem_hi[sym] = _shade_hex(c, 1.0, mix_white=0.42)
            elem_r[sym] = resolve_element_radius(sym, style) * eff_scale
            if _needs_light_outline(sym, c):
                outline_syms.add(sym)
            elif _needs_dark_rim_outline(c):
                dark_outline_syms.add(sym)

            if _HAS_PIL:
                profile = mat_cfg.get("metal" if sym in METAL_ELEMENTS else "nonmetal", {})
                if not isinstance(profile, dict):
                    profile = {}
                roughness = float(profile.get("roughness", mat_cfg.get("roughness", 0.35)))
                specular = float(profile.get("specular", mat_cfg.get("specular", 0.4)))
                metallic = float(profile.get("metallic", mat_cfg.get("metallic", 0.0)))
                sss = float(profile.get("subsurface_weight", mat_cfg.get("subsurface_weight", 0.0)))
                coat = float(profile.get("coat_weight", mat_cfg.get("coat_weight", 0.0)))
                ao_factor = float(mat_cfg.get("ao_factor", 0.0))
                base_sprites[(sym, 0)] = build_sphere_sprite(
                    hex_color=c,
                    roughness=roughness,
                    specular=specular,
                    metallic=metallic,
                    subsurface_weight=sss,
                    coat_weight=coat,
                    ao_factor=ao_factor,
                    has_rim=has_rim,
                    depth_shade=1.0,
                    size=88,
                )

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
            _base_sprites=base_sprites,
            _outline_symbols=outline_syms,
            _dark_outline_symbols=dark_outline_syms,
            show_atoms=show_atoms,
            show_labels=show_labels,
            show_axes=show_axes,
            axes_labels=axes_labels,
            axes_pos=axes_pos,
            dark_backdrop=dark_backdrop,
        )

    def _get_sprite(
        self,
        symbol: str,
        diameter_px: int,
        bucket: int = 0,
        interactive: bool = False,
    ) -> Any:
        key = (symbol, diameter_px, int(interactive), self.dark_backdrop)
        cached = self._resized_sprites.get(key)
        if cached is not None:
            return cached
        base = self._base_sprites.get((symbol, 0))
        if base is None:
            base = self._base_sprites.get((symbol, bucket))
        if base is None:
            return None
        if len(self._resized_sprites) > 200:
            self._resized_sprites.clear()
        resample = Image.Resampling.BILINEAR if interactive else Image.Resampling.BICUBIC
        resized = base.resize((diameter_px, diameter_px), resample)
        if symbol in self._outline_symbols:
            resized = _apply_thin_outline(resized)
        elif self.dark_backdrop and symbol in self._dark_outline_symbols:
            resized = _apply_thin_outline(resized, outline_rgb=(156, 163, 175), strength=0.45)
        self._resized_sprites[key] = resized
        return resized

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

    def roll(self, d_roll_rad: float) -> None:
        _, _, back = quat_to_axes(self.quat)
        q_roll = quat_from_axis_angle(back, -d_roll_rad)
        self.quat = quat_mul(q_roll, self.quat)

    def pan(self, dx_angstrom: float, dy_angstrom: float) -> None:
        right, up, _ = quat_to_axes(self.quat)
        self.target = self.target - right * dx_angstrom + up * dy_angstrom

    def zoom(self, factor: float) -> None:
        self.ortho_scale = max(0.5, min(2000.0, self.ortho_scale * factor))

    def render_image(
        self,
        canvas_w: int,
        canvas_h: int,
        res_w: int,
        res_h: int,
        interactive: bool = False,
    ) -> Any:
        """Render an anti-aliased shaded 3D view into an RGBA ``PIL.Image``."""
        if not _HAS_PIL:
            raise RuntimeError("Pillow is required for shaded image rendering.")

        cw = max(16, int(canvas_w))
        ch = max(16, int(canvas_h))
        ss = 1.0
        rw = cw
        rh = ch

        # Background canvas + clipped camera output frame
        outer_bg = (21, 23, 28, 255) if self.dark_backdrop else (232, 234, 237, 255)
        frame_bg = (36, 40, 48, 255) if self.dark_backdrop else (250, 251, 252, 255)
        border_col = (71, 85, 105, 255) if self.dark_backdrop else (148, 163, 184, 255)

        img = Image.new("RGBA", (rw, rh), outer_bg)
        pad = min(24.0 * ss, min(rw, rh) * 0.2)
        scale = max(1e-4, min((rw - pad) / max(1, res_w), (rh - pad) / max(1, res_h)))
        fw = res_w * scale
        fh = res_h * scale
        fw_i = max(4, int(round(fw)))
        fh_i = max(4, int(round(fh)))
        frame = Image.new("RGBA", (fw_i, fh_i), frame_bg)
        draw = ImageDraw.Draw(frame, "RGBA")
        cx0 = fw_i * 0.5
        cy0 = fh_i * 0.5

        px_per_a = (fw / max(1e-3, self.ortho_scale)) if res_w >= res_h else (fh / max(1e-3, self.ortho_scale))
        right, up, back = quat_to_axes(self.quat)

        draw_items: list[tuple[float, int, Any]] = []

        # 1. Cell wireframe edges
        if len(self.cell_edges) > 0:
            ce_rel = self.cell_edges - self.target
            ce_x = cx0 + (ce_rel @ right) * px_per_a
            ce_y = cy0 - (ce_rel @ up) * px_per_a
            ce_z = ce_rel @ back
            cell_w = max(1, min(4, int(round(0.11 * px_per_a))))
            cr, cg, cb = _hex_to_rgb(self.cell_color)
            cell_lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
            eff_cell_hex = "#CBD5E1" if (self.dark_backdrop and cell_lum < 60.0) else self.cell_color
            c_rgba = _shade_rgba_tuple(eff_cell_hex, 1.0)
            for idx in range(len(self.cell_edges)):
                z_mid = float(0.5 * (ce_z[idx, 0] + ce_z[idx, 1]))
                draw_items.append((
                    z_mid,
                    0,
                    ("cell", float(ce_x[idx, 0]), float(ce_y[idx, 0]), float(ce_x[idx, 1]), float(ce_y[idx, 1]), cell_w, c_rgba),
                ))

        # 2. Atoms & surface-trimmed, depth-subdivided shaded bonds
        if self.show_atoms and len(self.positions) > 0:
            rel = self.positions - self.target
            ax = cx0 + (rel @ right) * px_per_a
            ay = cy0 - (rel @ up) * px_per_a
            az = rel @ back
            z_min, z_max = float(np.min(az)), float(np.max(az))
            z_span = max(1e-3, z_max - z_min)

            if len(self.bond_i) > 0:
                bond_w = max(3, min(26, int(round(2.7 * self.bond_radius * px_per_a))))
                bi, bj = self.bond_i, self.bond_j
                for k in range(len(bi)):
                    i_idx, j_idx = int(bi[k]), int(bj[k])
                    pi3 = self.positions[i_idx]
                    pj3 = self.positions[j_idx]
                    dv = pj3 - pi3
                    dist3 = float(np.linalg.norm(dv))
                    if dist3 < 1e-6:
                        continue
                    d_hat = dv / dist3
                    # Trim bond endpoints inside sphere surfaces for clean 3D junction occlusion
                    trim_i = min(0.55 * float(self.radii[i_idx]), dist3 * 0.45)
                    trim_j = min(0.55 * float(self.radii[j_idx]), dist3 * 0.45)
                    if dist3 - trim_i - trim_j <= 0.02:
                        continue
                    p_start = pi3 + d_hat * trim_i
                    p_end = pj3 - d_hat * trim_j
                    p_mid = 0.5 * (pi3 + pj3)

                    if self.bond_color_mode == "bicolor":
                        pts3 = [
                            (p_start, p_mid, self.colors[i_idx]),
                            (p_mid, p_end, self.colors[j_idx]),
                        ]
                    else:
                        pts3 = [(p_start, p_end, self.single_bond_color)]
                    for s0, s1, b_hex in pts3:
                        r0 = s0 - self.target
                        r1 = s1 - self.target
                        sx0 = float(cx0 + np.dot(r0, right) * px_per_a)
                        sy0 = float(cy0 - np.dot(r0, up) * px_per_a)
                        sz0 = float(np.dot(r0, back))
                        sx1 = float(cx0 + np.dot(r1, right) * px_per_a)
                        sy1 = float(cy0 - np.dot(r1, up) * px_per_a)
                        sz1 = float(np.dot(r1, back))
                        draw_items.append((
                            0.5 * (sz0 + sz1),
                            1,
                            ("bond", sx0, sy0, sx1, sy1, bond_w, b_hex),
                        ))

            for idx in range(len(self.positions)):
                diam_px = max(2, int(round(2.0 * float(self.radii[idx]) * px_per_a)))
                z_norm = (float(az[idx]) - z_min) / z_span
                bucket = min(2, max(0, int(z_norm * 2.999)))
                draw_items.append((float(az[idx]), 2, ("atom", float(ax[idx]), float(ay[idx]), diam_px, idx, bucket)))

        draw_items.sort(key=lambda item: (item[0], item[1]))

        # Two-phase render: draw vector segments onto `draw`, flushing before compositing sprites
        for _z, _ord, payload in draw_items:
            tag = payload[0]
            if tag == "cell":
                _, px0, py0, px1, py1, w, c_rgba = payload
                draw.line((px0, py0, px1, py1), fill=c_rgba, width=w)
            elif tag == "bond":
                _, px0, py0, px1, py1, bw, b_hex = payload
                body_col = _shade_rgba_tuple(b_hex, 0.90)
                hi_col = _shade_rgba_tuple(b_hex, 1.0, mix_white=0.45)
                draw.line((px0, py0, px1, py1), fill=body_col, width=bw)
                r_cap = bw * 0.45
                if r_cap >= 1.2:
                    draw.ellipse((px0 - r_cap, py0 - r_cap, px0 + r_cap, py0 + r_cap), fill=body_col)
                    draw.ellipse((px1 - r_cap, py1 - r_cap, px1 + r_cap, py1 + r_cap), fill=body_col)
                if bw >= 3:
                    off = max(0.8, bw * 0.16)
                    draw.line(
                        (px0 - off, py0 - off, px1 - off, py1 - off),
                        fill=hi_col,
                        width=max(1, int(round(bw * 0.28))),
                    )
            else:
                _, sx, sy, diam_px, idx, bucket = payload
                sprite = self._get_sprite(self.symbols[idx], diam_px, bucket, interactive=interactive)
                if sprite is not None:
                    _blit_rgba(frame, sprite, int(round(sx - diam_px * 0.5)), int(round(sy - diam_px * 0.5)))
                    if self.show_labels and diam_px >= 12:
                        lbl = self.symbols[idx]
                        lfont = _load_overlay_font(max(9, min(22, int(round(diam_px * 0.42)))))
                        if lfont is not None:
                            bbox = draw.textbbox((0, 0), lbl, font=lfont)
                            tw_l = bbox[2] - bbox[0]
                            th_l = bbox[3] - bbox[1]
                            ar, ag, ab = _hex_to_rgb(self.colors[idx])
                            alum = 0.299 * ar + 0.587 * ag + 0.114 * ab
                            tfill = (15, 23, 42, 245) if alum > 155.0 else (248, 250, 252, 245)
                            tstroke = (248, 250, 252, 200) if alum > 155.0 else (15, 23, 42, 220)
                            draw.text(
                                (sx - tw_l * 0.5 - bbox[0], sy - th_l * 0.5 - bbox[1]),
                                lbl,
                                font=lfont,
                                fill=tfill,
                                stroke_width=1,
                                stroke_fill=tstroke,
                            )

        if self.show_axes:
            render_reference_axes_overlay(
                frame,
                self.quat,
                axes_pos=self.axes_pos,
                show_labels=self.axes_labels,
                dark_backdrop=self.dark_backdrop,
            )

        # Composite clipped camera frame onto canvas and draw crisp border
        fx0 = int(round((rw - fw_i) * 0.5))
        fy0 = int(round((rh - fh_i) * 0.5))
        img.paste(frame, (fx0, fy0))
        outer_draw = ImageDraw.Draw(img, "RGBA")
        outer_draw.rectangle(
            (fx0, fy0, fx0 + fw_i - 1, fy0 + fh_i - 1),
            outline=border_col,
            width=1,
        )
        return img

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
