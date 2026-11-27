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


def compute_cell_parameters(cell_vecs: Any) -> tuple[float, float, float, float, float, float]:
    """Return ``(a, b, c, alpha, beta, gamma)`` lengths (Å) and angles (deg) for a 3x3 cell."""
    cv = np.asarray(cell_vecs, dtype=np.float64).reshape(3, 3)
    a, b, c = cv[0], cv[1], cv[2]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    lc = float(np.linalg.norm(c))

    def _ang(u: np.ndarray, lu: float, v: np.ndarray, lv: float) -> float:
        if lu < 1e-8 or lv < 1e-8:
            return 90.0
        cos_v = max(-1.0, min(1.0, float(np.dot(u, v) / (lu * lv))))
        return math.degrees(math.acos(cos_v))

    return la, lb, lc, _ang(b, lb, c, lc), _ang(a, la, c, lc), _ang(a, la, b, lb)


def is_tilted_cell(cell_vecs: Any, tol_deg: float = 0.5) -> bool:
    """Return True when a 3D periodic cell is non-orthogonal (any angle differs from 90°)."""
    if cell_vecs is None:
        return False
    la, lb, lc, alpha, beta, gamma = compute_cell_parameters(cell_vecs)
    if la <= 1e-4 or lb <= 1e-4 or lc <= 1e-4:
        return False
    return (
        abs(alpha - 90.0) > tol_deg
        or abs(beta - 90.0) > tol_deg
        or abs(gamma - 90.0) > tol_deg
    )


def preset_to_quat(preset: str, cell_vecs: np.ndarray | None = None) -> Quat:
    """Match ``scene_builder.resolve_preset_orientation`` for ``iso``, ``side-oblique``, ``a/b/c``, and ``a*/b*/c*``."""
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
        up_c = np.cross(c, a)
        return direction_to_quat(c, up_c if float(np.linalg.norm(up_c)) > 1e-6 else b)
    if name == "a*":
        na = np.cross(b, c)
        return direction_to_quat(na if float(np.linalg.norm(na)) > 1e-6 else a, c)
    if name == "b*":
        nb = np.cross(c, a)
        return direction_to_quat(nb if float(np.linalg.norm(nb)) > 1e-6 else b, c)
    if name == "c*":
        nc = np.cross(a, b)
        nc_v = nc if float(np.linalg.norm(nc)) > 1e-6 else c
        up_nc = np.cross(nc_v, a)
        return direction_to_quat(nc_v, up_nc if float(np.linalg.norm(up_nc)) > 1e-6 else b)
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
    cell_vecs: Any = None,
) -> tuple[float, float, float]:
    """Draw a 3D depth-sorted Reference Axes (x,y,z or tilted a,b,c) gizmo onto ``target_img``."""
    tw, th = target_img.size
    u = max(0.05, min(0.95, float(axes_pos[0])))
    v = max(0.05, min(0.95, float(axes_pos[1])))
    cx = u * tw
    cy = v * th

    arm_len = float(arm_px) if arm_px is not None else max(22.0, min(46.0, min(tw, th) * 0.060))
    if not _HAS_PIL:
        return cx, cy, max(34.0, arm_len * 1.15)

    ss = 1
    box_half = int(math.ceil(arm_len * 1.48))
    size_ss = box_half * 2 * ss
    badge = Image.new("RGBA", (size_ss, size_ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    oc = box_half * ss

    disc_r = arm_len * 1.22 * ss
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
    if cell_vecs is not None and is_tilted_cell(cell_vecs):
        cv = np.asarray(cell_vecs, dtype=np.float64).reshape(3, 3)
        basis_defs = (
            ("a", cv[0] / max(1e-8, float(np.linalg.norm(cv[0]))), (239, 68, 68), (252, 165, 165)),
            ("b", cv[1] / max(1e-8, float(np.linalg.norm(cv[1]))), (34, 197, 94), (134, 239, 172)),
            ("c", cv[2] / max(1e-8, float(np.linalg.norm(cv[2]))), (59, 130, 246), (147, 197, 253)),
        )
    else:
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


def atom_label(idx: int, symbol: str) -> str:
    """Return 1-based indexed atom label shared between viewport and atom list."""
    return f"{symbol}{idx + 1}"


def _blit_rgba(canvas: Any, sprite: Any, x0: int, y0: int, alpha_scale: float = 1.0) -> None:
    """Composite ``sprite`` onto ``canvas`` at ``(x0, y0)`` clipping to canvas bounds."""
    cw, ch = canvas.size
    sw, sh = sprite.size
    if x0 >= cw or y0 >= ch or x0 + sw <= 0 or y0 + sh <= 0 or alpha_scale <= 0.01:
        return
    if alpha_scale < 0.995:
        s_copy = sprite.copy()
        a_band = s_copy.getchannel("A").point(lambda v, f=float(alpha_scale): int(round(v * f)))
        s_copy.putalpha(a_band)
        sprite = s_copy
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
    ghost_start_index: int | None = None,
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
                            if (
                                ghost_start_index is not None
                                and 0 < ghost_start_index < len(symbols)
                                and (i < ghost_start_index) != (j < ghost_start_index)
                            ):
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
    atom_alphas: np.ndarray = None  # type: ignore[assignment]
    selected_atoms: set[int] = field(default_factory=set)
    show_selection_highlight: bool = True
    show_scale_bar: bool = False
    show_measure_overlay: bool = True
    selection_order: list[int] = field(default_factory=list)
    dashed_pairs: list[tuple[int, int]] = field(default_factory=list)
    dashed_color: str = "#38BDF8"
    colormap_info: tuple[str, float, float] | None = None
    polyhedra: list[dict[str, Any]] = field(default_factory=list)
    show_cell_vectors: bool = False
    show_output_mask: bool = False
    _sprite_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = np.zeros(3, dtype=np.float64)
        if self.atom_alphas is None or len(self.atom_alphas) != len(self.positions):
            self.atom_alphas = np.ones(len(self.positions), dtype=np.float64)

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
        atom_alphas: Any = None,
        selected_atoms: set[int] | None = None,
        show_selection_highlight: bool = True,
        show_scale_bar: bool = False,
        show_measure_overlay: bool = True,
        selection_order: list[int] | None = None,
        colormap_info: tuple[str, float, float] | None = None,
        show_cell_vectors: bool = False,
        show_output_mask: bool = False,
    ) -> ViewportScene:
        symbols = list(structure["symbols"])
        positions = np.asarray(structure["positions"], dtype=np.float64).reshape(-1, 3)
        raw_alphas = atom_alphas if atom_alphas is not None else structure.get("alphas")
        if raw_alphas is not None and len(raw_alphas) == len(symbols):
            alphas_arr = np.clip(np.asarray(raw_alphas, dtype=np.float64), 0.02, 1.0)
        else:
            alphas_arr = np.ones(len(symbols), dtype=np.float64)
        has_cell = bool(structure.get("has_cell", True))
        origin = np.asarray(structure.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        cell_vecs = np.asarray(structure.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]), dtype=np.float64)

        eff_scale = float(style.get("elements", {}).get("radius_scale", 1.0)) * float(radius_scale)
        mat_cfg = style.get("material", {})
        has_rim = isinstance(style.get("lighting", {}).get("rim"), dict)

        raw_colors = structure.get("colors")
        use_custom_cols = isinstance(raw_colors, list) and len(raw_colors) == len(symbols)
        colors = (
            [str(c) for c in raw_colors]
            if use_custom_cols
            else [resolve_element_color(s, style) for s in symbols]
        )

        elem_r: dict[str, float] = {}
        for sym in set(symbols):
            elem_r[sym] = resolve_element_radius(sym, style) * eff_scale

        base_sprites: dict[tuple[str, int], Any] = {}
        outline_syms: set[str] = set()
        dark_outline_syms: set[str] = set()
        sprite_keys: list[str] = []
        built_keys: set[str] = set()

        for sym, c in zip(symbols, colors):
            s_key = f"{sym}:{c}" if use_custom_cols else sym
            sprite_keys.append(s_key)
            if s_key in built_keys:
                continue
            built_keys.add(s_key)
            if _needs_light_outline(sym, c):
                outline_syms.add(s_key)
            elif _needs_dark_rim_outline(c):
                dark_outline_syms.add(s_key)

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
                base_sprites[(s_key, 0)] = build_sphere_sprite(
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
        raw_r_scales = structure.get("radius_scales")
        if isinstance(raw_r_scales, list) and len(raw_r_scales) == len(symbols):
            r_mult = np.clip(np.asarray(raw_r_scales, dtype=np.float64), 0.15, 3.0)
            radii = radii * r_mult
        outline_colors = [_shade_hex(c, 0.62) for c in colors]
        highlight_colors = [_shade_hex(c, 1.0, mix_white=0.42) for c in colors]

        bond_cfg = dict(style.get("bonds", {}))
        if bonds_mode and bonds_mode != "style":
            bond_cfg["mode"] = bonds_mode
            bond_cfg["enabled"] = bonds_mode != "none"
        ghost_start = structure.get("ghost_start_index")
        bond_i, bond_j = _find_bonds(
            symbols,
            positions,
            bond_cfg,
            ghost_start_index=int(ghost_start) if ghost_start is not None else None,
        )
        raw_dashed = structure.get("dashed_bonds", [])
        dashed_pairs = [
            (int(pair[0]), int(pair[1]))
            for pair in raw_dashed
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and 0 <= int(pair[0]) < len(symbols)
            and 0 <= int(pair[1]) < len(symbols)
        ]

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
            atom_alphas=alphas_arr,
            selected_atoms=set(selected_atoms or ()),
            show_selection_highlight=show_selection_highlight,
            show_scale_bar=show_scale_bar,
            show_measure_overlay=show_measure_overlay,
            selection_order=list(selection_order or ()),
            dashed_pairs=dashed_pairs,
            dashed_color=str(structure.get("dashed_color", "#38BDF8")),
            colormap_info=colormap_info,
            polyhedra=list(structure.get("polyhedra", [])),
            show_cell_vectors=show_cell_vectors,
            show_output_mask=show_output_mask,
            _sprite_keys=sprite_keys,
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
                if abs(float(c[2])) > 1e-6:
                    t_min = (zmin - float(self.origin[2])) / float(c[2])
                    t_max = (zmax - float(self.origin[2])) / float(c[2])
                    pts = np.array([p + c * t_min for p in base] + [p + c * t_max for p in base], dtype=np.float64)
                else:
                    pts = np.array(
                        [[p[0], p[1], zmin] for p in base] + [[p[0], p[1], zmax] for p in base],
                        dtype=np.float64,
                    )
                return np.mean(pts, axis=0), pts

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

    def pick_atom_at(
        self,
        u_norm: float,
        v_norm: float,
        res_w: int,
        res_h: int,
        frame_w_px: float = 600.0,
        frame_h_px: float = 600.0,
    ) -> int | None:
        """Return index of the front-most atom hit at normalized frame coordinates ``(u_norm, v_norm)``."""
        if not self.show_atoms or len(self.positions) == 0:
            return None
        fw = max(4.0, float(frame_w_px))
        fh = max(4.0, float(frame_h_px))
        cx0 = fw * 0.5
        cy0 = fh * 0.5
        px_per_a = (fw / max(1e-3, self.ortho_scale)) if res_w >= res_h else (fh / max(1e-3, self.ortho_scale))
        right, up, back = quat_to_axes(self.quat)
        rel = self.positions - self.target
        ax = cx0 + (rel @ right) * px_per_a
        ay = cy0 - (rel @ up) * px_per_a
        az = rel @ back

        px = float(u_norm) * fw
        py = float(v_norm) * fh
        best_idx: int | None = None
        best_z = -1e18
        for idx in range(len(self.positions)):
            r_px = max(7.0, float(self.radii[idx]) * px_per_a)
            dx = px - float(ax[idx])
            dy = py - float(ay[idx])
            if dx * dx + dy * dy <= r_px * r_px:
                z_val = float(az[idx])
                if best_idx is None or z_val > best_z:
                    best_z = z_val
                    best_idx = idx
        return best_idx

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

        img = Image.new("RGBA", (rw, rh), frame_bg if self.show_output_mask else outer_bg)
        pad = min(14.0 * ss, min(rw, rh) * 0.12)
        scale = max(1e-4, min((rw - pad) / max(1, res_w), (rh - pad) / max(1, res_h)))
        fw = res_w * scale
        fh = res_h * scale
        fw_i = max(4, int(round(fw)))
        fh_i = max(4, int(round(fh)))
        frame = img if self.show_output_mask else Image.new("RGBA", (fw_i, fh_i), frame_bg)
        draw = ImageDraw.Draw(frame, "RGBA")
        cx0 = frame.width * 0.5
        cy0 = frame.height * 0.5

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

        # 1b. Unit-cell origin a, b, c lattice vector arrows
        if self.has_cell and self.show_cell_vectors and len(self.cell_edges) > 0:
            vec_defs = (
                ("a", self.cell_vecs[0], (239, 68, 68, 245)),
                ("b", self.cell_vecs[1], (34, 197, 94, 245)),
                ("c", self.cell_vecs[2], (59, 130, 246, 245)),
            )
            o_rel = self.origin - self.target
            ox = float(cx0 + np.dot(o_rel, right) * px_per_a)
            oy = float(cy0 - np.dot(o_rel, up) * px_per_a)
            oz = float(np.dot(o_rel, back))
            arr_w = max(2, min(6, int(round(0.16 * px_per_a))))
            for v_lbl, vec, v_col in vec_defs:
                v_len = float(np.linalg.norm(vec))
                if v_len < 1e-6:
                    continue
                tip3 = self.origin + vec * min(0.45, 2.8 / max(1e-3, v_len))
                t_rel = tip3 - self.target
                tx = float(cx0 + np.dot(t_rel, right) * px_per_a)
                ty = float(cy0 - np.dot(t_rel, up) * px_per_a)
                tz = float(np.dot(t_rel, back))
                draw_items.append((0.5 * (oz + tz), 3, ("cell_vec", ox, oy, tx, ty, arr_w, v_col, v_lbl)))

        # 1c. Translucent coordination polyhedra faces & edges
        if self.polyhedra:
            for poly in self.polyhedra:
                verts = np.asarray(poly.get("vertices", []), dtype=np.float64)
                faces = poly.get("faces", [])
                edges = poly.get("edges", [])
                if len(verts) < 3 or not faces:
                    continue
                v_rel = verts - self.target
                vx = cx0 + (v_rel @ right) * px_per_a
                vy = cy0 - (v_rel @ up) * px_per_a
                vz = v_rel @ back
                p_hex = str(poly.get("color", "#38BDF8"))
                p_alpha = float(poly.get("alpha", 0.36))
                pr, pg, pb = _hex_to_rgb(p_hex)
                for face in faces:
                    if len(face) != 3:
                        continue
                    i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
                    p0, p1, p2 = verts[i0], verts[i1], verts[i2]
                    fn = np.cross(p1 - p0, p2 - p0)
                    fn_len = float(np.linalg.norm(fn))
                    if fn_len < 1e-6:
                        continue
                    fu = fn / fn_len
                    # Lambertian illumination factor from camera direction (+back) & key offset
                    lambert = 0.48 + 0.52 * abs(float(np.dot(fu, back)))
                    fr = min(255, int(round(pr * lambert)))
                    fg = min(255, int(round(pg * lambert)))
                    fb = min(255, int(round(pb * lambert)))
                    fa = max(20, min(220, int(round(255.0 * p_alpha))))
                    zf = float((vz[i0] + vz[i1] + vz[i2]) / 3.0)
                    poly2d = [
                        (float(vx[i0]), float(vy[i0])),
                        (float(vx[i1]), float(vy[i1])),
                        (float(vx[i2]), float(vy[i2])),
                    ]
                    draw_items.append((zf, 1, ("poly_face", poly2d, (fr, fg, fb, fa))))
                edge_col = (min(255, pr + 45), min(255, pg + 45), min(255, pb + 45), 185)
                for edge in edges:
                    if len(edge) != 2:
                        continue
                    ea, eb = int(edge[0]), int(edge[1])
                    ze = 0.5 * float(vz[ea] + vz[eb])
                    draw_items.append((ze, 1, ("poly_edge", float(vx[ea]), float(vy[ea]), float(vx[eb]), float(vy[eb]), edge_col)))

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
                        a_i = float(self.atom_alphas[i_idx]) if i_idx < len(self.atom_alphas) else 1.0
                        a_j = float(self.atom_alphas[j_idx]) if j_idx < len(self.atom_alphas) else 1.0
                        pts3 = [
                            (p_start, p_mid, self.colors[i_idx], a_i),
                            (p_mid, p_end, self.colors[j_idx], a_j),
                        ]
                    else:
                        a_ij = min(
                            float(self.atom_alphas[i_idx]) if i_idx < len(self.atom_alphas) else 1.0,
                            float(self.atom_alphas[j_idx]) if j_idx < len(self.atom_alphas) else 1.0,
                        )
                        pts3 = [(p_start, p_end, self.single_bond_color, a_ij)]
                    for s0, s1, b_hex, b_alpha in pts3:
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
                            ("bond", sx0, sy0, sx1, sy1, bond_w, b_hex, b_alpha),
                        ))

            if len(self.dashed_pairs) > 0:
                db_w = max(2, min(10, int(round(1.5 * self.bond_radius * px_per_a))))
                dr, dg, db, _ = _shade_rgba_tuple(self.dashed_color, 1.0)
                for i_idx, j_idx in self.dashed_pairs:
                    if not (0 <= i_idx < len(self.positions) and 0 <= j_idx < len(self.positions)):
                        continue
                    pi3 = self.positions[i_idx]
                    pj3 = self.positions[j_idx]
                    dv = pj3 - pi3
                    dist3 = float(np.linalg.norm(dv))
                    if dist3 < 1e-6:
                        continue
                    d_hat = dv / dist3
                    trim_i = min(0.55 * float(self.radii[i_idx]), dist3 * 0.42)
                    trim_j = min(0.55 * float(self.radii[j_idx]), dist3 * 0.42)
                    if dist3 - trim_i - trim_j <= 0.02:
                        continue
                    r0 = (pi3 + d_hat * trim_i) - self.target
                    r1 = (pj3 - d_hat * trim_j) - self.target
                    sx0 = float(cx0 + np.dot(r0, right) * px_per_a)
                    sy0 = float(cy0 - np.dot(r0, up) * px_per_a)
                    sz0 = float(np.dot(r0, back))
                    sx1 = float(cx0 + np.dot(r1, right) * px_per_a)
                    sy1 = float(cy0 - np.dot(r1, up) * px_per_a)
                    sz1 = float(np.dot(r1, back))
                    a_db = min(
                        float(self.atom_alphas[i_idx]) if i_idx < len(self.atom_alphas) else 1.0,
                        float(self.atom_alphas[j_idx]) if j_idx < len(self.atom_alphas) else 1.0,
                    )
                    draw_items.append((
                        0.5 * (sz0 + sz1),
                        1,
                        ("dashed_bond", sx0, sy0, sx1, sy1, db_w, (dr, dg, db, max(75, int(round(230 * a_db))))),
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
                _, px0, py0, px1, py1, bw, b_hex, b_alpha = payload
                a_u8 = max(8, min(255, int(round(255.0 * b_alpha))))
                br, bg_c, bb, _ = _shade_rgba_tuple(b_hex, 0.90)
                hr, hg, hb, _ = _shade_rgba_tuple(b_hex, 1.0, mix_white=0.45)
                body_col = (br, bg_c, bb, a_u8)
                hi_col = (hr, hg, hb, a_u8)
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
            elif tag == "dashed_bond":
                _, px0, py0, px1, py1, bw, d_rgba = payload
                _draw_dashed_segment(draw, (px0, py0), (px1, py1), fill=d_rgba, width=bw, dash_px=7.0, gap_px=5.0)
            elif tag == "poly_face":
                _, poly2d, f_rgba = payload
                draw.polygon(poly2d, fill=f_rgba)
            elif tag == "poly_edge":
                _, px0, py0, px1, py1, e_rgba = payload
                draw.line((px0, py0, px1, py1), fill=e_rgba, width=1)
            elif tag == "cell_vec":
                _, px0, py0, px1, py1, aw, v_rgba, v_lbl = payload
                draw.line((px0, py0, px1, py1), fill=v_rgba, width=aw)
                dv2 = math.hypot(px1 - px0, py1 - py0)
                if dv2 >= 8.0:
                    ux, uy = (px1 - px0) / dv2, (py1 - py0) / dv2
                    perp_x, perp_y = -uy, ux
                    hl = max(6.0, aw * 2.6)
                    hw = max(3.5, aw * 1.4)
                    draw.polygon(
                        [
                            (px1 + ux * 2.0, py1 + uy * 2.0),
                            (px1 - ux * hl + perp_x * hw, py1 - uy * hl + perp_y * hw),
                            (px1 - ux * hl - perp_x * hw, py1 - uy * hl - perp_y * hw),
                        ],
                        fill=v_rgba,
                    )
                vfont = _load_overlay_font(max(11, aw * 4))
                if vfont is not None:
                    draw.text((px1 + 5.0, py1 - 8.0), v_lbl, font=vfont, fill=v_rgba)
            else:
                _, sx, sy, diam_px, idx, bucket = payload
                s_key = self._sprite_keys[idx] if idx < len(self._sprite_keys) else self.symbols[idx]
                sprite = self._get_sprite(s_key, diam_px, bucket, interactive=interactive)
                if sprite is not None:
                    a_scale = float(self.atom_alphas[idx]) if idx < len(self.atom_alphas) else 1.0
                    _blit_rgba(
                        frame,
                        sprite,
                        int(round(sx - diam_px * 0.5)),
                        int(round(sy - diam_px * 0.5)),
                        alpha_scale=a_scale,
                    )
                    if self.show_selection_highlight and idx in self.selected_atoms and diam_px >= 6:
                        r_ring = diam_px * 0.5 + 1.5
                        draw.ellipse(
                            (sx - r_ring, sy - r_ring, sx + r_ring, sy + r_ring),
                            outline=(45, 212, 191, 240),
                            width=max(2, min(4, int(round(diam_px * 0.08)))),
                        )
                    if self.show_labels and diam_px >= 10:
                        lbl = atom_label(idx, self.symbols[idx])
                        px_font = max(9, min(15, int(round(diam_px * (0.31 if len(lbl) >= 3 else 0.36)))))
                        lfont = _load_overlay_font(px_font)
                        if lfont is not None:
                            bbox = draw.textbbox((0, 0), lbl, font=lfont)
                            tw_l = bbox[2] - bbox[0]
                            th_l = bbox[3] - bbox[1]
                            ar, ag, ab = _hex_to_rgb(self.colors[idx])
                            alum = 0.299 * ar + 0.587 * ag + 0.114 * ab
                            tfill = (15, 23, 42, 245) if alum > 155.0 else (248, 250, 252, 245)
                            tstroke = (248, 250, 252, 210) if alum > 155.0 else (15, 23, 42, 225)
                            draw.text(
                                (sx - tw_l * 0.5 - bbox[0], sy - th_l * 0.5 - bbox[1]),
                                lbl,
                                font=lfont,
                                fill=tfill,
                                stroke_width=1,
                                stroke_fill=tstroke,
                            )

        # Composite / align camera frame onto canvas and draw crisp border (+ dark margin mask if show_output_mask)
        fx0 = int(round((rw - fw_i) * 0.5))
        fy0 = int(round((rh - fh_i) * 0.5))
        if self.show_output_mask:
            sub_frame = img.crop((fx0, fy0, fx0 + fw_i, fy0 + fh_i))
        else:
            sub_frame = frame

        if self.show_measure_overlay and len(self.selection_order) in (2, 3, 4):
            render_measurement_callout_overlay(
                sub_frame,
                self,
                self.selection_order,
                res_w,
                res_h,
                dark_backdrop=self.dark_backdrop,
            )

        if self.show_scale_bar:
            render_scale_bar_overlay(
                sub_frame,
                self.ortho_scale,
                res_w,
                res_h,
                dark_backdrop=self.dark_backdrop,
            )

        if self.colormap_info is not None:
            cmap_name, z_lo, z_hi = self.colormap_info
            render_colorbar_overlay(
                sub_frame,
                cmap_name,
                z_lo,
                z_hi,
                dark_backdrop=self.dark_backdrop,
            )

        if self.show_axes:
            render_reference_axes_overlay(
                sub_frame,
                self.quat,
                axes_pos=self.axes_pos,
                show_labels=self.axes_labels,
                dark_backdrop=self.dark_backdrop,
                cell_vecs=self.cell_vecs if self.has_cell else None,
            )

        img.paste(sub_frame, (fx0, fy0))
        if self.show_output_mask:
            mask_ov = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
            m_draw = ImageDraw.Draw(mask_ov, "RGBA")
            m_col = (10, 12, 16, 155)
            fx1, fy1 = fx0 + fw_i, fy0 + fh_i
            if fy0 > 0:
                m_draw.rectangle((0, 0, rw, fy0), fill=m_col)
            if fy1 < rh:
                m_draw.rectangle((0, fy1, rw, rh), fill=m_col)
            if fx0 > 0:
                m_draw.rectangle((0, fy0, fx0, fy1), fill=m_col)
            if fx1 < rw:
                m_draw.rectangle((fx1, fy0, rw, fy1), fill=m_col)
            img.alpha_composite(mask_ov)
            outer_draw = ImageDraw.Draw(img, "RGBA")
            outer_draw.rectangle(
                (fx0, fy0, fx0 + fw_i - 1, fy0 + fh_i - 1),
                outline=(20, 184, 166, 235),
                width=2,
            )
        else:
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


def render_atom_labels_and_selection_overlay(
    target_img: Any,
    vp: ViewportScene,
    res_w: int,
    res_h: int,
    draw_labels: bool = True,
    draw_selection: bool = True,
) -> None:
    """Composite indexed atom labels and selection rings onto a rendered preview image."""
    if not _HAS_PIL or len(vp.positions) == 0 or (not draw_labels and not draw_selection):
        return
    fw, fh = target_img.size
    cx0 = fw * 0.5
    cy0 = fh * 0.5
    px_per_a = (fw / max(1e-3, vp.ortho_scale)) if res_w >= res_h else (fh / max(1e-3, vp.ortho_scale))
    right, up, back = quat_to_axes(vp.quat)
    rel = vp.positions - vp.target
    ax = cx0 + (rel @ right) * px_per_a
    ay = cy0 - (rel @ up) * px_per_a
    az = rel @ back

    draw = ImageDraw.Draw(target_img, "RGBA")
    for idx in sorted(range(len(vp.positions)), key=lambda i: float(az[i])):
        sx = float(ax[idx])
        sy = float(ay[idx])
        diam_px = max(2, int(round(2.0 * float(vp.radii[idx]) * px_per_a)))
        if draw_selection and idx in vp.selected_atoms and diam_px >= 6:
            r_ring = diam_px * 0.5 + 1.5
            draw.ellipse(
                (sx - r_ring, sy - r_ring, sx + r_ring, sy + r_ring),
                outline=(45, 212, 191, 240),
                width=max(2, min(4, int(round(diam_px * 0.08)))),
            )
        if draw_labels and diam_px >= 10:
            lbl = atom_label(idx, vp.symbols[idx])
            px_font = max(9, min(15, int(round(diam_px * (0.31 if len(lbl) >= 3 else 0.36)))))
            lfont = _load_overlay_font(px_font)
            if lfont is not None:
                bbox = draw.textbbox((0, 0), lbl, font=lfont)
                tw_l = bbox[2] - bbox[0]
                th_l = bbox[3] - bbox[1]
                ar, ag, ab = _hex_to_rgb(vp.colors[idx])
                alum = 0.299 * ar + 0.587 * ag + 0.114 * ab
                tfill = (15, 23, 42, 245) if alum > 155.0 else (248, 250, 252, 245)
                tstroke = (248, 250, 252, 210) if alum > 155.0 else (15, 23, 42, 225)
                draw.text(
                    (sx - tw_l * 0.5 - bbox[0], sy - th_l * 0.5 - bbox[1]),
                    lbl,
                    font=lfont,
                    fill=tfill,
                    stroke_width=1,
                    stroke_fill=tstroke,
                )


def choose_scale_bar_angstrom(
    ortho_scale: float,
    target_fraction: float = 0.20,
    max_angstrom: float | None = None,
) -> float:
    """Choose a clean physical length in Angstroms spanning roughly ``target_fraction`` of frame."""
    candidates = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    if max_angstrom is not None:
        fitting = [c for c in candidates if c <= float(max_angstrom) + 1e-9]
        candidates = tuple(fitting) if fitting else (candidates[0],)
    desired = max(0.5, float(ortho_scale) * float(target_fraction))
    if max_angstrom is not None:
        desired = min(desired, float(max_angstrom))
    return min(candidates, key=lambda c: abs(c - desired))


def render_scale_bar_overlay(
    target_img: Any,
    ortho_scale: float,
    res_w: int,
    res_h: int,
    dark_backdrop: bool = True,
) -> float:
    """Composite an orthographic physical scale bar onto ``target_img`` in-place and return bar length (Å)."""
    if not _HAS_PIL or target_img is None:
        return choose_scale_bar_angstrom(ortho_scale)
    fw, fh = target_img.size
    px_per_a = (fw / max(1e-3, ortho_scale)) if res_w >= res_h else (fh / max(1e-3, ortho_scale))
    max_bar_a = (fw * 0.45) / max(1e-6, px_per_a)
    bar_a = choose_scale_bar_angstrom(ortho_scale, max_angstrom=max_bar_a)
    bar_px = max(8.0, bar_a * px_per_a)

    draw = ImageDraw.Draw(target_img, "RGBA")
    pad_x = max(14, int(round(min(fw, fh) * 0.03)))
    pad_y = max(14, int(round(min(fw, fh) * 0.03)))
    font_px = max(11, min(18, int(round(min(fw, fh) * 0.026))))
    font = _load_overlay_font(font_px)
    label = f"{bar_a:g} Å"
    tw, th = (len(label) * font_px * 0.6, font_px)
    if font is not None:
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]

    box_w = max(bar_px + 20.0, tw + 22.0)
    box_h = th + 18.0
    bx1 = fw - pad_x
    by1 = fh - pad_y
    bx0 = bx1 - box_w
    by0 = by1 - box_h

    bg_col = (15, 23, 42, 190) if dark_backdrop else (248, 250, 252, 210)
    fg_col = (248, 250, 252, 245) if dark_backdrop else (15, 23, 42, 245)
    border_col = (51, 65, 85, 210) if dark_backdrop else (148, 163, 184, 210)
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=6, fill=bg_col, outline=border_col, width=1)

    lx0 = 0.5 * (bx0 + bx1) - bar_px * 0.5
    lx1 = 0.5 * (bx0 + bx1) + bar_px * 0.5
    ly = by1 - 7.0
    lw = max(2, min(4, int(round(min(fw, fh) * 0.004))))
    draw.line((lx0, ly, lx1, ly), fill=fg_col, width=lw)
    draw.line((lx0, ly - 4.0, lx0, ly + 2.5), fill=fg_col, width=lw)
    draw.line((lx1, ly - 4.0, lx1, ly + 2.5), fill=fg_col, width=lw)

    if font is not None:
        tb = draw.textbbox((0, 0), label, font=font)
        tx = 0.5 * (bx0 + bx1) - (tb[2] - tb[0]) * 0.5 - tb[0]
        ty = by0 + 4.0 - tb[1]
        draw.text((tx, ty), label, font=font, fill=fg_col)
    return bar_a


def _draw_dashed_segment(
    draw: Any,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int = 2,
    dash_px: float = 7.0,
    gap_px: float = 5.0,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 2.0:
        return
    ux, uy = dx / dist, dy / dist
    cur = 0.0
    step = dash_px + gap_px
    while cur < dist:
        end = min(dist, cur + dash_px)
        draw.line(
            (x0 + ux * cur, y0 + uy * cur, x0 + ux * end, y0 + uy * end),
            fill=fill,
            width=width,
        )
        cur += step


def render_measurement_callout_overlay(
    target_img: Any,
    vp: ViewportScene,
    selected_order: list[int],
    res_w: int,
    res_h: int,
    dark_backdrop: bool = True,
) -> None:
    """Draw 2D dashed callout lines and a compact measurement badge for 2-4 ordered atoms."""
    if not _HAS_PIL or target_img is None or len(vp.positions) == 0:
        return
    n = len(vp.positions)
    order: list[int] = []
    seen: set[int] = set()
    for idx in selected_order:
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    if len(order) not in (2, 3, 4):
        return

    fw, fh = target_img.size
    cx0 = fw * 0.5
    cy0 = fh * 0.5
    px_per_a = (fw / max(1e-3, vp.ortho_scale)) if res_w >= res_h else (fh / max(1e-3, vp.ortho_scale))
    right, up, _ = quat_to_axes(vp.quat)

    pts2d: list[tuple[float, float]] = []
    for idx in order:
        rel = vp.positions[idx] - vp.target
        pts2d.append((float(cx0 + np.dot(rel, right) * px_per_a), float(cy0 - np.dot(rel, up) * px_per_a)))

    pos = vp.positions
    if len(order) == 2:
        dist = float(np.linalg.norm(pos[order[1]] - pos[order[0]]))
        badge = f"{dist:.2f} Å"
        anchor = ((pts2d[0][0] + pts2d[1][0]) * 0.5, (pts2d[0][1] + pts2d[1][1]) * 0.5)
    elif len(order) == 3:
        v1 = pos[order[0]] - pos[order[1]]
        v2 = pos[order[2]] - pos[order[1]]
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if d1 < 1e-9 or d2 < 1e-9:
            return
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2) / (d1 * d2))))
        badge = f"{math.degrees(math.acos(cos_a)):.1f}°"
        mid_arms = ((pts2d[0][0] + pts2d[2][0]) * 0.5, (pts2d[0][1] + pts2d[2][1]) * 0.5)
        anchor = (pts2d[1][0] + (mid_arms[0] - pts2d[1][0]) * 0.35, pts2d[1][1] + (mid_arms[1] - pts2d[1][1]) * 0.35)
    else:
        p0, p1, p2, p3 = pos[order[0]], pos[order[1]], pos[order[2]], pos[order[3]]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_n = float(np.linalg.norm(b1))
        if b1_n < 1e-9:
            return
        b1_u = b1 / b1_n
        v = b0 - np.dot(b0, b1_u) * b1_u
        w = b2 - np.dot(b2, b1_u) * b1_u
        if float(np.linalg.norm(v)) < 1e-9 or float(np.linalg.norm(w)) < 1e-9:
            return
        dih = math.degrees(math.atan2(float(np.dot(np.cross(b1_u, v), w)), float(np.dot(v, w))))
        badge = f"φ {dih:.1f}°"
        anchor = ((pts2d[1][0] + pts2d[2][0]) * 0.5, (pts2d[1][1] + pts2d[2][1]) * 0.5)

    draw = ImageDraw.Draw(target_img, "RGBA")
    line_col = (45, 212, 191, 240) if dark_backdrop else (13, 148, 136, 245)
    for k in range(len(pts2d) - 1):
        _draw_dashed_segment(draw, pts2d[k], pts2d[k + 1], fill=line_col, width=2)

    font_px = max(11, min(16, int(round(min(fw, fh) * 0.024))))
    font = _load_overlay_font(font_px)
    if font is None:
        return
    tb = draw.textbbox((0, 0), badge, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    pad_x, pad_y = 7.0, 4.0
    bx0 = max(6.0, min(fw - tw - 2 * pad_x - 6.0, anchor[0] - (tw * 0.5 + pad_x)))
    by0 = max(6.0, min(fh - th - 2 * pad_y - 6.0, anchor[1] - (th * 0.5 + pad_y)))
    draw.rounded_rectangle(
        (bx0, by0, bx0 + tw + 2 * pad_x, by0 + th + 2 * pad_y),
        radius=6,
        fill=(15, 23, 42, 225),
        outline=(45, 212, 191, 245),
        width=1,
    )
    draw.text((bx0 + pad_x - tb[0], by0 + pad_y - tb[1]), badge, font=font, fill=(248, 250, 252, 255))


def compose_multi_panel_strip(
    images: list[Any],
    titles: list[str] | None = None,
    dark_backdrop: bool = True,
) -> Any:
    """Stitch 1 or more rendered images horizontally into one labeled figure strip."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to compose multi-panel figures.")
    if not images:
        raise ValueError("At least one image is required.")
    imgs = [im.convert("RGBA") for im in images]
    h = max(im.height for im in imgs)
    normed = []
    for im in imgs:
        if im.height != h:
            im = im.resize((max(1, int(round(im.width * (h / im.height)))), h), Image.Resampling.LANCZOS)
        normed.append(im)

    gap = max(12, int(round(h * 0.015)))
    header_h = max(36, int(round(h * 0.06)))
    margin = max(12, int(round(h * 0.015)))
    total_w = margin * 2 + sum(im.width for im in normed) + gap * max(0, len(normed) - 1)
    total_h = margin * 2 + header_h + h

    bg = (21, 23, 28, 255) if dark_backdrop else (248, 250, 252, 255)
    fg = (248, 250, 252, 255) if dark_backdrop else (15, 23, 42, 255)
    div_col = (51, 65, 85, 255) if dark_backdrop else (203, 213, 225, 255)

    canvas = Image.new("RGBA", (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font_px = max(14, min(32, int(round(header_h * 0.56))))
    font = _load_overlay_font(font_px)

    x_cur = margin
    y_img = margin + header_h
    for idx, im in enumerate(normed):
        canvas.paste(im, (x_cur, y_img), im)
        if idx > 0:
            div_x = x_cur - gap * 0.5
            draw.line((div_x, margin, div_x, total_h - margin), fill=div_col, width=max(1, gap // 8))
        title = titles[idx] if (titles and idx < len(titles)) else f"({chr(97 + idx)})"
        if font is not None:
            tb = draw.textbbox((0, 0), title, font=font)
            tx = x_cur + im.width * 0.5 - (tb[2] - tb[0]) * 0.5 - tb[0]
            ty = margin + (header_h - (tb[3] - tb[1])) * 0.5 - tb[1]
            draw.text((tx, ty), title, font=font, fill=fg)
        x_cur += im.width + gap
    return canvas


def compose_dual_view_panel(
    top_img: Any,
    side_img: Any,
    dark_backdrop: bool = True,
    top_title: str = "(a) Top View",
    side_title: str = "(b) Side View",
) -> Any:
    """Stitch Top and Side rendered images horizontally into one labeled publication panel."""
    return compose_multi_panel_strip([top_img, side_img], [top_title, side_title], dark_backdrop=dark_backdrop)


_COLORMAP_STOPS: dict[str, list[tuple[int, int, int]]] = {
    "viridis": [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    "coolwarm": [
        (59, 76, 192),
        (124, 159, 249),
        (221, 221, 221),
        (244, 154, 123),
        (180, 4, 38),
    ],
    "plasma": [
        (13, 8, 135),
        (126, 3, 168),
        (204, 71, 120),
        (248, 149, 64),
        (240, 249, 33),
    ],
}


def sample_colormap_hex(cmap_name: str, t: float) -> str:
    """Sample a 5-stop scientific colormap at normalized ``t`` in ``[0, 1]`` as ``#RRGGBB``."""
    stops = _COLORMAP_STOPS.get(str(cmap_name).lower(), _COLORMAP_STOPS["viridis"])
    tt = max(0.0, min(1.0, float(t)))
    scaled = tt * (len(stops) - 1)
    idx = min(len(stops) - 2, int(math.floor(scaled)))
    frac = scaled - idx
    c0, c1 = stops[idx], stops[idx + 1]
    r = int(round(c0[0] + (c1[0] - c0[0]) * frac))
    g = int(round(c0[1] + (c1[1] - c0[1]) * frac))
    b = int(round(c0[2] + (c1[2] - c0[2]) * frac))
    return f"#{r:02X}{g:02X}{b:02X}"


def compute_z_colormap_colors(
    positions: Any,
    cmap_name: str,
    n_bins: int = 24,
) -> tuple[list[str], float, float]:
    """Map Z coordinates into quantized colormap hex strings and return ``(colors, z_min, z_max)``."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(pos) == 0:
        return [], 0.0, 0.0
    zs = pos[:, 2]
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    span = z_max - z_min
    bins = max(2, int(n_bins))
    out: list[str] = []
    for z in zs:
        t = 0.5 if span < 1e-6 else max(0.0, min(1.0, (float(z) - z_min) / span))
        qt = round(t * (bins - 1)) / (bins - 1)
        out.append(sample_colormap_hex(cmap_name, qt))
    return out, z_min, z_max


def render_colorbar_overlay(
    target_img: Any,
    cmap_name: str,
    z_min: float,
    z_max: float,
    title: str = "Z (Å)",
    dark_backdrop: bool = True,
) -> None:
    """Draw a compact vertical publication colorbar legend onto ``target_img``."""
    if not _HAS_PIL or target_img is None:
        return
    fw, fh = target_img.size
    draw = ImageDraw.Draw(target_img, "RGBA")
    bar_h = max(90, min(240, int(round(fh * 0.34))))
    bar_w = max(12, min(20, int(round(fw * 0.024))))
    font_px = max(10, min(15, int(round(min(fw, fh) * 0.022))))
    font = _load_overlay_font(font_px)

    ticks = [f"{z_max:.1f}", f"{0.5 * (z_min + z_max):.1f}", f"{z_min:.1f}"]
    max_tw = font_px * 3.0
    title_h = font_px + 6
    if font is not None:
        max_tw = max(draw.textbbox((0, 0), t, font=font)[2] for t in ticks + [title])
        tb_title = draw.textbbox((0, 0), title, font=font)
        title_h = (tb_title[3] - tb_title[1]) + 8

    pad = 8
    box_w = bar_w + max_tw + pad * 3
    box_h = title_h + bar_h + pad * 2
    bx1 = fw - max(12, int(round(fw * 0.025)))
    bx0 = bx1 - box_w
    by0 = max(12, int(round((fh - box_h) * 0.45)))
    by1 = by0 + box_h

    bg_col = (15, 23, 42, 195) if dark_backdrop else (248, 250, 252, 215)
    fg_col = (248, 250, 252, 245) if dark_backdrop else (15, 23, 42, 245)
    border_col = (51, 65, 85, 210) if dark_backdrop else (148, 163, 184, 210)
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=6, fill=bg_col, outline=border_col, width=1)

    gx0 = bx0 + pad
    gx1 = gx0 + bar_w
    gy0 = by0 + title_h + pad - 2
    n_steps = 32
    for s in range(n_steps):
        frac_top = 1.0 - (s + 0.5) / n_steps
        hx = sample_colormap_hex(cmap_name, frac_top)
        r, g, b = _hex_to_rgb(hx)
        sy0 = gy0 + (bar_h * s) / n_steps
        sy1 = gy0 + (bar_h * (s + 1)) / n_steps
        draw.rectangle((gx0, sy0, gx1, sy1 + 0.5), fill=(r, g, b, 245))
    draw.rectangle((gx0, gy0, gx1, gy0 + bar_h), outline=border_col, width=1)

    if font is not None:
        tb_t = draw.textbbox((0, 0), title, font=font)
        draw.text((bx0 + pad - tb_t[0], by0 + 5 - tb_t[1]), title, font=font, fill=fg_col)
        tx = gx1 + pad
        for label, frac_y in ((ticks[0], 0.0), (ticks[1], 0.5), (ticks[2], 1.0)):
            tb = draw.textbbox((0, 0), label, font=font)
            th = tb[3] - tb[1]
            ty = gy0 + frac_y * bar_h - th * 0.5 - tb[1]
            draw.text((tx - tb[0], ty), label, font=font, fill=fg_col)


def find_dashed_contacts(
    symbols: list[str],
    positions: Any,
    bond_i: Any = None,
    bond_j: Any = None,
    mode: str = "hbond",
    d_min: float = 1.55,
    d_max: float = 2.55,
    focal_indices: set[int] | None = None,
    ghost_start_index: int | None = None,
) -> list[tuple[int, int]]:
    """Detect H-bonds, interface contacts, or distance-ranged pairs using an O(N) spatial hash grid."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(symbols), len(pos))
    if n < 2:
        return []
    lo = max(0.2, min(float(d_min), float(d_max)))
    hi = max(lo, max(float(d_min), float(d_max)))
    cov_set: set[tuple[int, int]] = set()
    if bond_i is not None and bond_j is not None:
        for a, b in zip(bond_i, bond_j):
            ia, ib = int(a), int(b)
            cov_set.add((min(ia, ib), max(ia, ib)))

    acceptors = {"O", "N", "F", "Cl", "S"}
    focal = set(focal_indices or ())
    if mode == "interface" and not focal:
        return []

    cell_size = max(hi, 0.5)
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in range(n):
        sym = symbols[idx]
        if mode == "hbond" and sym != "H" and sym not in acceptors:
            continue
        px, py, pz = float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2])
        key = (
            math.floor(px / cell_size),
            math.floor(py / cell_size),
            math.floor(pz / cell_size),
        )
        grid[key].append(idx)

    pairs: list[tuple[int, int]] = []
    for (gx, gy, gz), bucket in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nbr = grid.get((gx + dx, gy + dy, gz + dz))
                    if not nbr:
                        continue
                    for i in bucket:
                        pi = pos[i]
                        si = symbols[i]
                        for j in nbr:
                            if j <= i:
                                continue
                            if (i, j) in cov_set:
                                continue
                            if ghost_start_index is not None and 0 < ghost_start_index < n:
                                if (i < ghost_start_index) != (j < ghost_start_index):
                                    continue
                            if mode == "hbond":
                                sj = symbols[j]
                                if not ((si == "H" and sj in acceptors) or (sj == "H" and si in acceptors)):
                                    continue
                            elif mode == "interface":
                                if (i in focal) == (j in focal):
                                    continue
                            dist = float(np.linalg.norm(pos[j] - pi))
                            if lo - 1e-6 <= dist <= hi + 1e-6:
                                pairs.append((i, j))
    pairs.sort()
    return pairs


def merge_ghost_structure(
    main_struct: dict[str, Any],
    ghost_struct: dict[str, Any],
    ghost_alpha: float = 0.24,
    ghost_scale: float = 0.72,
    ghost_tint: str = "#38BDF8",
    disp_threshold_a: float = 0.20,
) -> dict[str, Any]:
    """Overlay a ghost reference structure onto ``main_struct`` with displacement vectors."""
    main_syms = list(main_struct["symbols"])
    ghost_syms = list(ghost_struct["symbols"])
    main_pos = [list(map(float, p)) for p in main_struct["positions"]]
    ghost_pos = [list(map(float, p)) for p in ghost_struct["positions"]]
    n_main = len(main_syms)
    n_ghost = len(ghost_syms)

    raw_a = main_struct.get("alphas")
    main_alphas = [float(a) for a in raw_a] if isinstance(raw_a, list) and len(raw_a) == n_main else [1.0] * n_main
    raw_rs = main_struct.get("radius_scales")
    main_rs = [float(s) for s in raw_rs] if isinstance(raw_rs, list) and len(raw_rs) == n_main else [1.0] * n_main

    merged = dict(main_struct)
    merged["symbols"] = main_syms + ghost_syms
    merged["positions"] = main_pos + ghost_pos
    merged["alphas"] = main_alphas + [float(ghost_alpha)] * n_ghost
    merged["radius_scales"] = main_rs + [float(ghost_scale)] * n_ghost
    merged["ghost_start_index"] = n_main
    merged["ghost_tint"] = ghost_tint

    raw_cols = main_struct.get("colors")
    if isinstance(raw_cols, list) and len(raw_cols) == n_main:
        merged["colors"] = list(raw_cols) + [ghost_tint] * n_ghost

    disp_pairs: list[list[int]] = []
    if n_main == n_ghost and n_main > 0:
        mp = np.asarray(main_pos, dtype=np.float64)
        gp = np.asarray(ghost_pos, dtype=np.float64)
        for i in range(n_main):
            if float(np.linalg.norm(mp[i] - gp[i])) >= float(disp_threshold_a):
                disp_pairs.append([i, n_main + i])
    merged["dashed_bonds"] = disp_pairs
    return merged


def _convex_hull_3d_triangles(pts: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    """Compute outward-oriented triangles and perimeter edges of a 3D convex ligand shell."""
    k = len(pts)
    if k < 4:
        return [], []
    center = np.mean(pts, axis=0)
    coplanar_groups: dict[frozenset[int], tuple[np.ndarray, list[int]]] = {}

    for a in range(k):
        for b in range(a + 1, k):
            for c in range(b + 1, k):
                n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
                nn = float(np.linalg.norm(n))
                if nn < 1e-6:
                    continue
                u = n / nn
                if float(np.dot(u, pts[a] - center)) < 0.0:
                    u = -u
                dots = (pts - pts[a]) @ u
                if not bool(np.all(dots <= 1e-3)):
                    continue
                on_plane = sorted(i for i in range(k) if abs(float(dots[i])) <= 2e-3)
                if len(on_plane) >= 3:
                    coplanar_groups.setdefault(frozenset(on_plane), (u, on_plane))

    faces: list[list[int]] = []
    edges_set: set[tuple[int, int]] = set()
    for u, verts_idx in coplanar_groups.values():
        p_c = np.mean(pts[verts_idx], axis=0)
        ref = pts[verts_idx[0]] - p_c
        ref_n = float(np.linalg.norm(ref))
        if ref_n < 1e-6:
            continue
        e1 = ref / ref_n
        e2 = np.cross(u, e1)
        ordered = sorted(
            verts_idx,
            key=lambda vi: math.atan2(float(np.dot(pts[vi] - p_c, e2)), float(np.dot(pts[vi] - p_c, e1))),
        )
        for m in range(1, len(ordered) - 1):
            faces.append([int(ordered[0]), int(ordered[m]), int(ordered[m + 1])])
        for m in range(len(ordered)):
            u_e, v_e = int(ordered[m]), int(ordered[(m + 1) % len(ordered)])
            edges_set.add((min(u_e, v_e), max(u_e, v_e)))
    return faces, [[a, b] for a, b in sorted(edges_set)]


def find_coordination_polyhedra(
    symbols: list[str],
    positions: Any,
    style: dict[str, Any] | None = None,
    cutoff_a: float = 2.55,
    alpha: float = 0.36,
    cell_vecs: Any = None,
    pbc: Any = None,
) -> list[dict[str, Any]]:
    """Build translucent coordination polyhedra for cation centers (with periodic-image ligands when cell is periodic)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(symbols), len(pos))
    if n < 5:
        return []
    poly_centers = METAL_ELEMENTS | {"Si", "B", "P", "Ge", "As", "Sb"}
    anions = {"O", "F", "N", "Cl", "S", "Br", "I"}
    lig_idx = [i for i in range(n) if symbols[i] in anions]
    if len(lig_idx) < 4:
        return []

    shifts = [np.zeros(3, dtype=np.float64)]
    if cell_vecs is not None:
        cv = np.asarray(cell_vecs, dtype=np.float64).reshape(3, 3)
        if abs(float(np.linalg.det(cv))) > 1e-6:
            pbc_flags = [True, True, True] if pbc is None else [bool(p) for p in pbc]
            ranges = [(-1, 0, 1) if flag else (0,) for flag in pbc_flags[:3]]
            shifts = [
                da * cv[0] + db * cv[1] + dc * cv[2]
                for da in ranges[0]
                for db in ranges[1]
                for dc in ranges[2]
            ]

    lig_pos = pos[lig_idx]
    all_lig_pts = np.vstack([lig_pos + s for s in shifts])
    style_dict = style or {}
    out: list[dict[str, Any]] = []
    for i in range(n):
        sym = symbols[i]
        if sym not in poly_centers:
            continue
        dists = np.linalg.norm(all_lig_pts - pos[i], axis=1)
        cand_mask = (dists >= 0.4) & (dists <= float(cutoff_a))
        cand_pts = all_lig_pts[cand_mask]
        if len(cand_pts) < 4:
            continue
        uniq_pts: list[np.ndarray] = []
        seen_keys: set[tuple[int, int, int]] = set()
        for pt in cand_pts:
            k = (int(round(float(pt[0]) * 1000)), int(round(float(pt[1]) * 1000)), int(round(float(pt[2]) * 1000)))
            if k not in seen_keys:
                seen_keys.add(k)
                uniq_pts.append(pt)
        if not (4 <= len(uniq_pts) <= 12):
            continue
        v_pts = np.asarray(uniq_pts, dtype=np.float64)
        faces, edges = _convex_hull_3d_triangles(v_pts)
        if not faces:
            continue
        out.append({
            "center": i,
            "symbol": sym,
            "color": resolve_element_color(sym, style_dict),
            "alpha": float(alpha),
            "vertices": [[float(x), float(y), float(z)] for x, y, z in v_pts],
            "faces": faces,
            "edges": edges,
        })
    return out


def compute_slice_mask(
    positions: Any,
    mode: str = "none",
    depth_fraction: float = 0.5,
    quat: Quat = (1.0, 0.0, 0.0, 0.0),
    cell_vecs: Any = None,
    origin: Any = None,
) -> list[bool]:
    """Compute per-atom visibility mask for cross-section / cutaway planes (fractional on tilted cells)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    if n == 0 or mode == "none":
        return [True] * n
    frac = max(0.05, min(1.0, float(depth_fraction)))
    if frac >= 0.999:
        return [True] * n
    axis_map = {"x": 0, "a": 0, "y": 1, "b": 1, "z": 2, "c": 2}
    if mode in axis_map:
        axis_idx = axis_map[mode]
        used_frac = False
        if cell_vecs is not None:
            cv = np.asarray(cell_vecs, dtype=np.float64).reshape(3, 3)
            if abs(float(np.linalg.det(cv))) > 1e-6:
                orig = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
                vals = ((pos - orig) @ np.linalg.inv(cv))[:, axis_idx]
                used_frac = True
        if not used_frac:
            vals = pos[:, axis_idx]
    elif mode == "camera":
        _, _, back = quat_to_axes(quat)
        vals = (pos - np.mean(pos, axis=0)) @ back
    else:
        return [True] * n

    s_min, s_max = float(np.min(vals)), float(np.max(vals))
    if s_max - s_min < 1e-6:
        return [True] * n
    thr = s_min + frac * (s_max - s_min)
    return [bool(v <= thr + 1e-6) for v in vals]
