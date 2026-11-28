"""Hardware-accelerated OpenGL 3.3 instanced 3D viewport with Qt HUD & Blender preview overlay."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from atomic_blender.core import compute_selection_measurement
from atomic_blender.viewport3d import (
    ViewportScene,
    atom_label,
    choose_scale_bar_angstrom,
    is_tilted_cell,
    quat_to_axes,
)

try:
    from OpenGL import GL as gl

    _HAS_PYOPENGL = True
except Exception:
    gl = None  # type: ignore[assignment]
    _HAS_PYOPENGL = False


def _hex_to_rgb01(hex_col: str) -> tuple[float, float, float]:
    s = str(hex_col).strip().lstrip("#")
    if len(s) != 6:
        return (0.75, 0.78, 0.82)
    try:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except Exception:
        return (0.75, 0.78, 0.82)


def pil_to_qpixmap(pil_img: Any) -> QPixmap:
    """Convert an RGBA/RGB ``PIL.Image`` into a ``QPixmap`` without copying temp files."""
    if pil_img is None:
        return QPixmap()
    rgba = pil_img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


_SPHERE_VERT = """
#version 330 core
layout(location = 0) in vec3 iCenter;
layout(location = 1) in float iRadius;
layout(location = 2) in vec4 iColorAlpha;
layout(location = 3) in vec4 iMatParams; // x=metallic, y=roughness, z=selected, w=outlineHint

uniform vec3 uTarget;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamBack;
uniform float uPxPerA;
uniform vec2 uHalfCanvasPx;
uniform float uZHalfRange;

out vec2 vQuad;
out vec4 vColorAlpha;
out vec4 vMatParams;
out float vNdcZCenter;
out float vNdcZRadius;
out float vDiamPx;

void main() {
    vec2 corners[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    vec2 corner = corners[gl_VertexID] * 1.10;
    vec3 rel = iCenter - uTarget;
    float xc = dot(rel, uCamRight);
    float yc = dot(rel, uCamUp);
    float zc = dot(rel, uCamBack);

    float sx = xc * uPxPerA;
    float sy = yc * uPxPerA;
    float rPx = max(1.5, iRadius * uPxPerA);

    vec2 ndcCenter = vec2(sx / uHalfCanvasPx.x, sy / uHalfCanvasPx.y);
    vec2 ndcRad = vec2(rPx / uHalfCanvasPx.x, rPx / uHalfCanvasPx.y);
    float ndcZ = clamp(-zc / max(1.0, uZHalfRange), -0.92, 0.92);
    float ndcZRad = clamp(iRadius / max(1.0, uZHalfRange), 0.0002, 0.25);

    gl_Position = vec4(ndcCenter + corner * ndcRad, ndcZ, 1.0);
    vQuad = corner;
    vColorAlpha = iColorAlpha;
    vMatParams = iMatParams;
    vNdcZCenter = ndcZ;
    vNdcZRadius = ndcZRad;
    vDiamPx = 2.0 * rPx;
}
"""

_SPHERE_FRAG = """
#version 330 core
in vec2 vQuad;
in vec4 vColorAlpha;
in vec4 vMatParams;
in float vNdcZCenter;
in float vNdcZRadius;
in float vDiamPx;

uniform float uDarkBackdrop;

out vec4 FragColor;

void main() {
    float r2 = dot(vQuad, vQuad);
    float r = sqrt(max(r2, 0.0));
    float selected = vMatParams.z;
    float pxStep = 2.0 / max(4.0, vDiamPx);

    if (selected > 0.5 && r > 0.96 && r <= 1.09) {
        float rimAlpha = smoothstep(1.09, 1.04, r) * smoothstep(0.95, 0.99, r);
        FragColor = vec4(0.176, 0.831, 0.749, 0.95 * rimAlpha);
        float zNdc = clamp(vNdcZCenter - 0.15 * vNdcZRadius, -0.999, 0.999);
        gl_FragDepth = 0.5 * zNdc + 0.5;
        return;
    }

    if (r > 1.0) {
        discard;
    }

    float nz = sqrt(max(0.0, 1.0 - r2));
    vec3 N = normalize(vec3(vQuad.x, vQuad.y, nz));
    vec3 V = vec3(0.0, 0.0, 1.0);

    vec3 L_key = normalize(vec3(-0.46, 0.64, 0.61));
    vec3 L_fill = normalize(vec3(0.68, -0.24, 0.69));
    vec3 baseCol = vColorAlpha.rgb;
    float metallic = clamp(vMatParams.x, 0.0, 1.0);
    float rough = clamp(vMatParams.y, 0.06, 1.0);

    float diffKey = max(dot(N, L_key), 0.0);
    float diffFill = max(dot(N, L_fill), 0.0);
    float specPow = mix(110.0, 8.0, rough * rough);

    vec3 H_key = normalize(L_key + V);
    float specKey = pow(max(dot(N, H_key), 0.0), specPow) * mix(0.55, 1.0, 1.0 - rough);
    vec3 H_fill = normalize(L_fill + V);
    float specFill = pow(max(dot(N, H_fill), 0.0), specPow * 0.5) * 0.20;

    float fresnel = pow(1.0 - clamp(N.z, 0.0, 1.0), 3.0);
    float ao = mix(0.72, 1.0, smoothstep(-0.3, 0.95, nz));

    vec3 diffWeight = baseCol * (0.28 + 0.64 * diffKey + 0.22 * diffFill) * ao;
    vec3 specTint = mix(vec3(1.0), baseCol, metallic * 0.75);
    vec3 lit = diffWeight + specTint * (specKey + specFill);
    lit += mix(vec3(0.16, 0.20, 0.26), baseCol * 0.45, uDarkBackdrop) * fresnel * 0.35;

    if (vMatParams.w > 0.5 && r > 0.91) {
        float ringMix = smoothstep(0.91, 0.99, r) * 0.45;
        vec3 outlineCol = (uDarkBackdrop > 0.5) ? vec3(0.62, 0.66, 0.72) : vec3(0.32, 0.36, 0.42);
        lit = mix(lit, outlineCol, ringMix);
    }

    float edgeAA = 1.0 - smoothstep(1.0 - pxStep * 1.4, 1.0, r);
    float alpha = vColorAlpha.a * edgeAA;
    if (alpha < 0.01) {
        discard;
    }

    float zNdc = clamp(vNdcZCenter - nz * vNdcZRadius, -0.999, 0.999);
    gl_FragDepth = 0.5 * zNdc + 0.5;
    FragColor = vec4(clamp(lit, 0.0, 1.0), alpha);
}
"""

_BOND_VERT = """
#version 330 core
layout(location = 0) in vec3 iStart;
layout(location = 1) in vec3 iEnd;
layout(location = 2) in float iRadius;
layout(location = 3) in vec4 iColorAlpha;

uniform vec3 uTarget;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamBack;
uniform float uPxPerA;
uniform vec2 uHalfCanvasPx;
uniform float uZHalfRange;

out vec2 vAlongAcross;
out vec4 vColorAlpha;
out float vZStart;
out float vZEnd;
out float vRadiusNdcZ;

void main() {
    vec2 quad[4] = vec2[4](
        vec2(0.0, -1.0),
        vec2(0.0,  1.0),
        vec2(1.0, -1.0),
        vec2(1.0,  1.0)
    );
    vec2 q = quad[gl_VertexID];

    vec3 r0 = iStart - uTarget;
    vec3 r1 = iEnd - uTarget;
    vec2 p0 = vec2(dot(r0, uCamRight), dot(r0, uCamUp)) * uPxPerA;
    vec2 p1 = vec2(dot(r1, uCamRight), dot(r1, uCamUp)) * uPxPerA;
    float z0 = dot(r0, uCamBack);
    float z1 = dot(r1, uCamBack);

    vec2 dir = p1 - p0;
    float lenPx = length(dir);
    vec2 axis = (lenPx > 0.001) ? (dir / lenPx) : vec2(1.0, 0.0);
    vec2 perp = vec2(-axis.y, axis.x);
    float radPx = max(1.5, iRadius * uPxPerA);

    vec2 pxPos = mix(p0, p1, q.x) + perp * (q.y * radPx);
    float zA = mix(z0, z1, q.x);
    float ndcZ = clamp(-zA / max(1.0, uZHalfRange), -0.92, 0.92);

    gl_Position = vec4(pxPos.x / uHalfCanvasPx.x, pxPos.y / uHalfCanvasPx.y, ndcZ, 1.0);
    vAlongAcross = q;
    vColorAlpha = iColorAlpha;
    vZStart = z0;
    vZEnd = z1;
    vRadiusNdcZ = clamp(iRadius / max(1.0, uZHalfRange), 0.0001, 0.2);
}
"""

_BOND_FRAG = """
#version 330 core
in vec2 vAlongAcross;
in vec4 vColorAlpha;
in float vZStart;
in float vZEnd;
in float vRadiusNdcZ;

uniform float uZHalfRange;

out vec4 FragColor;

void main() {
    float across = clamp(vAlongAcross.y, -1.0, 1.0);
    float nz = sqrt(max(0.0, 1.0 - across * across));
    float shade = 0.45 + 0.55 * max(0.0, 0.55 * nz - 0.35 * across);
    float spec = pow(max(0.0, 0.72 * nz - 0.38 * across), 24.0) * 0.40;
    float aa = 1.0 - smoothstep(0.84, 1.0, abs(across));
    vec3 col = clamp(vColorAlpha.rgb * shade + vec3(spec), 0.0, 1.0);

    float zA = mix(vZStart, vZEnd, vAlongAcross.x);
    float zNdc = clamp(-zA / max(1.0, uZHalfRange) - nz * vRadiusNdcZ, -0.999, 0.999);
    gl_FragDepth = 0.5 * zNdc + 0.5;
    FragColor = vec4(col, vColorAlpha.a * aa);
}
"""

_FLAT_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;

uniform vec3 uTarget;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamBack;
uniform float uPxPerA;
uniform vec2 uHalfCanvasPx;
uniform float uZHalfRange;

out vec4 vColor;

void main() {
    vec3 rel = aPos - uTarget;
    float sx = dot(rel, uCamRight) * uPxPerA;
    float sy = dot(rel, uCamUp) * uPxPerA;
    float zc = dot(rel, uCamBack);
    float ndcZ = clamp(-zc / max(1.0, uZHalfRange), -0.95, 0.95);
    gl_Position = vec4(sx / uHalfCanvasPx.x, sy / uHalfCanvasPx.y, ndcZ, 1.0);
    vColor = aColor;
}
"""

_FLAT_FRAG = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    if (vColor.a < 0.01) {
        discard;
    }
    FragColor = vColor;
}
"""


def _compile_program(vert_src: str, frag_src: str) -> int:
    vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vs, vert_src)
    gl.glCompileShader(vs)
    if gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        err = gl.glGetShaderInfoLog(vs).decode("utf-8", errors="replace")
        gl.glDeleteShader(vs)
        raise RuntimeError(f"Vertex shader compile failed: {err}")

    fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fs, frag_src)
    gl.glCompileShader(fs)
    if gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        err = gl.glGetShaderInfoLog(fs).decode("utf-8", errors="replace")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        raise RuntimeError(f"Fragment shader compile failed: {err}")

    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if gl.glGetProgramiv(prog, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        err = gl.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        gl.glDeleteProgram(prog)
        raise RuntimeError(f"Shader program link failed: {err}")
    return int(prog)


class _ViewportOverlay(QWidget):
    """Transparent HiDPI overlay drawing Blender preview, HUD pill, axes gizmo, scale bar & labels."""

    def __init__(self, viewport: AtomicGLViewport) -> None:
        super().__init__(viewport)
        self._vp = viewport
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, _event: Any) -> None:
        vp = self._vp
        scene = vp.scene
        cw = max(1, self.width())
        ch = max(1, self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        rect = vp.frame_rect()
        dark_bg = scene.dark_backdrop if scene is not None else vp.dark_backdrop

        # 1. Fallback image or settled Blender Cycles Auto-Preview inside camera frame
        active_pm = vp.active_overlay_pixmap()
        if active_pm is not None and not active_pm.isNull() and rect is not None:
            fx0, fy0, fw, fh = rect
            painter.drawPixmap(QRectF(fx0, fy0, fw, fh), active_pm, QRectF(0, 0, active_pm.width(), active_pm.height()))

        # 2. Frame border + subtle output mask
        if rect is not None:
            fx0, fy0, fw, fh = rect
            if scene is not None and scene.show_output_mask:
                mask_col = QColor(10, 12, 16, 155)
                painter.fillRect(QRectF(0, 0, cw, fy0), mask_col)
                painter.fillRect(QRectF(0, fy0 + fh, cw, ch - (fy0 + fh)), mask_col)
                painter.fillRect(QRectF(0, fy0, fx0, fh), mask_col)
                painter.fillRect(QRectF(fx0 + fw, fy0, cw - (fx0 + fw), fh), mask_col)
                painter.setPen(QPen(QColor(20, 184, 166, 235), 2))
            else:
                border_col = QColor(71, 85, 105, 220) if dark_bg else QColor(148, 163, 184, 220)
                painter.setPen(QPen(border_col, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(fx0, fy0, fw - 1.0, fh - 1.0))

        if scene is None:
            return

        # 3. Atom labels + selection rings + measurement callout when showing live GL scene (or labels)
        if rect is not None:
            fx0, fy0, fw, fh = rect
            cx0 = fx0 + fw * 0.5
            cy0 = fy0 + fh * 0.5
            px_per_a = vp._px_per_angstrom_for_frame(fw, fh)
            right, up, back = quat_to_axes(scene.quat)
            show_sel_ring = bool(scene.show_selection_highlight and scene.selected_atoms)
            if scene.show_atoms and len(scene.positions) > 0 and (scene.show_labels or (vp.has_blender_preview() and show_sel_ring)):
                rel = scene.positions - scene.target
                ax = cx0 + (rel @ right) * px_per_a
                ay = cy0 - (rel @ up) * px_per_a
                az = rel @ back
                for idx in sorted(range(len(scene.positions)), key=lambda i: float(az[i])):
                    diam_px = max(2.0, 2.0 * float(scene.radii[idx]) * px_per_a)
                    sx, sy = float(ax[idx]), float(ay[idx])
                    if vp.has_blender_preview() and scene.show_selection_highlight and idx in scene.selected_atoms and diam_px >= 6.0:
                        r_ring = diam_px * 0.5 + 1.5
                        painter.setPen(QPen(QColor(45, 212, 191, 240), max(2.0, min(4.0, diam_px * 0.08))))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawEllipse(QPointF(sx, sy), r_ring, r_ring)
                    if scene.show_labels and diam_px >= 10.0 and not vp.is_interacting:
                        lbl = atom_label(idx, scene.symbols[idx])
                        pt_sz = max(7.0, min(11.0, diam_px * 0.24))
                        font = QFont("Segoe UI", int(round(pt_sz)))
                        font.setBold(True)
                        painter.setFont(font)
                        r_c, g_c, b_c = _hex_to_rgb01(scene.colors[idx])
                        alum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                        fg_col = QColor(15, 23, 42, 245) if alum > 0.61 else QColor(248, 250, 252, 245)
                        painter.setPen(fg_col)
                        painter.drawText(QRectF(sx - 28, sy - 10, 56, 20), Qt.AlignmentFlag.AlignCenter, lbl)

            if scene.show_measure_overlay and len(scene.selection_order) in (2, 3, 4):
                self._paint_measurement_callout(painter, scene, cx0, cy0, px_per_a, right, up, fx0, fy0, fw, fh, dark_bg)

            if scene.show_axes:
                self._paint_axes_gizmo(painter, scene, fx0, fy0, fw, fh, dark_bg)

            if scene.show_scale_bar:
                self._paint_scale_bar(painter, scene, fx0, fy0, fw, fh, px_per_a, dark_bg)

        # 4. Top-left Structure HUD pill
        self._paint_hud_pill(painter, dark_bg)

    def _paint_hud_pill(self, painter: QPainter, dark_bg: bool) -> None:
        vp = self._vp
        summary = vp.structure_summary.strip()
        mode_tag = "Blender Preview" if vp.has_blender_preview() else ("GL 3D" if vp.gl_active else "CPU 3D")
        text = f"{mode_tag}  ·  {summary}" if summary else mode_tag

        font = QFont("Segoe UI", 8)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        x0, y0 = 12.0, 10.0
        rect = QRectF(x0, y0, min(max(80.0, tw + 18.0), max(80.0, self.width() - 24.0)), th + 10.0)
        bg = QColor(17, 20, 26, 205) if dark_bg else QColor(248, 250, 252, 225)
        edge = QColor(45, 55, 72, 220) if dark_bg else QColor(203, 213, 225, 220)
        fg = QColor(226, 232, 240, 245) if dark_bg else QColor(30, 41, 59, 245)
        painter.setPen(QPen(edge, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(fg)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(rect.width() - 14))
        painter.drawText(rect.adjusted(9, 0, -9, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)

    def _paint_axes_gizmo(
        self,
        painter: QPainter,
        scene: ViewportScene,
        fx0: float,
        fy0: float,
        fw: float,
        fh: float,
        dark_bg: bool,
    ) -> None:
        arm_len = max(24.0, min(38.0, min(fw, fh) * 0.075))
        margin_r = arm_len * 1.45
        cx = max(fx0 + margin_r, min(fx0 + fw - margin_r, fx0 + scene.axes_pos[0] * fw))
        cy = max(fy0 + margin_r, min(fy0 + fh - margin_r, fy0 + scene.axes_pos[1] * fh))
        disc_r = arm_len * 1.22

        bg = QColor(18, 22, 28, 165) if dark_bg else QColor(244, 246, 250, 185)
        edge = QColor(94, 108, 128, 170) if dark_bg else QColor(148, 163, 184, 190)
        painter.setPen(QPen(edge, 1))
        painter.setBrush(bg)
        painter.drawEllipse(QPointF(cx, cy), disc_r, disc_r)

        right, up, back = quat_to_axes(scene.quat)
        if scene.has_cell and is_tilted_cell(scene.cell_vecs):
            cv = np.asarray(scene.cell_vecs, dtype=np.float64).reshape(3, 3)
            basis = (
                ("a", cv[0] / max(1e-8, float(np.linalg.norm(cv[0]))), QColor(239, 68, 68), QColor(252, 165, 165)),
                ("b", cv[1] / max(1e-8, float(np.linalg.norm(cv[1]))), QColor(34, 197, 94), QColor(134, 239, 172)),
                ("c", cv[2] / max(1e-8, float(np.linalg.norm(cv[2]))), QColor(59, 130, 246), QColor(147, 197, 253)),
            )
        else:
            basis = (
                ("x", np.array([1.0, 0.0, 0.0]), QColor(239, 68, 68), QColor(252, 165, 165)),
                ("y", np.array([0.0, 1.0, 0.0]), QColor(34, 197, 94), QColor(134, 239, 172)),
                ("z", np.array([0.0, 0.0, 1.0]), QColor(59, 130, 246), QColor(147, 197, 253)),
            )
        projected = []
        for lbl, vec, col, tcol in basis:
            dx = float(np.dot(vec, right))
            dy = -float(np.dot(vec, up))
            dz = float(np.dot(vec, back))
            projected.append((dz, lbl, dx, dy, col, tcol))
        projected.sort(key=lambda item: item[0])

        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        shaft_w = max(2.0, arm_len * 0.11)
        head_w = max(3.5, arm_len * 0.14)
        for _dz, lbl, dx, dy, col, tcol in projected:
            plen = math.hypot(dx, dy)
            tx = cx + dx * arm_len
            ty = cy + dy * arm_len
            painter.setPen(QPen(col, shaft_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            if plen > 0.08:
                ux, uy = dx / plen, dy / plen
                px_u, py_u = -uy, ux
                painter.drawLine(QPointF(cx, cy), QPointF(cx + dx * arm_len * 0.76, cy + dy * arm_len * 0.76))
                bx, by = cx + dx * arm_len * 0.70, cy + dy * arm_len * 0.70
                poly = QPolygonF([
                    QPointF(tx, ty),
                    QPointF(bx + px_u * head_w, by + py_u * head_w),
                    QPointF(bx - px_u * head_w, by - py_u * head_w),
                ])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(col)
                painter.drawPolygon(poly)
                lx = cx + dx * (arm_len + arm_len * 0.25)
                ly = cy + dy * (arm_len + arm_len * 0.25)
            else:
                painter.setBrush(col)
                painter.drawEllipse(QPointF(tx, ty), shaft_w, shaft_w)
                lx = cx + arm_len * 0.25
                ly = cy - arm_len * 0.25
            if scene.axes_labels:
                painter.setPen(tcol if dark_bg else col)
                painter.drawText(QRectF(lx - 10, ly - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, lbl)

        painter.setPen(QPen(QColor(15, 23, 42, 220), 1))
        painter.setBrush(QColor(226, 232, 240, 245))
        painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)

    def _paint_scale_bar(
        self,
        painter: QPainter,
        scene: ViewportScene,
        fx0: float,
        fy0: float,
        fw: float,
        fh: float,
        px_per_a: float,
        dark_bg: bool,
    ) -> None:
        max_bar_a = (fw * 0.45) / max(1e-6, px_per_a)
        bar_a = choose_scale_bar_angstrom(scene.ortho_scale, max_angstrom=max_bar_a)
        bar_px = max(12.0, bar_a * px_per_a)
        label = f"{bar_a:g} Å"
        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label)
        box_w = max(bar_px + 20.0, tw + 22.0)
        box_h = fm.height() + 18.0
        pad = 12.0
        bx0 = fx0 + fw - pad - box_w
        by0 = fy0 + fh - pad - box_h

        bg = QColor(15, 23, 42, 195) if dark_bg else QColor(248, 250, 252, 215)
        edge = QColor(51, 65, 85, 210) if dark_bg else QColor(148, 163, 184, 210)
        fg = QColor(248, 250, 252, 245) if dark_bg else QColor(15, 23, 42, 245)
        painter.setPen(QPen(edge, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(QRectF(bx0, by0, box_w, box_h), 6, 6)

        lx0 = bx0 + 0.5 * (box_w - bar_px)
        lx1 = lx0 + bar_px
        ly = by0 + box_h - 7.0
        painter.setPen(QPen(fg, 2))
        painter.drawLine(QPointF(lx0, ly), QPointF(lx1, ly))
        painter.drawLine(QPointF(lx0, ly - 3.5), QPointF(lx0, ly + 2.5))
        painter.drawLine(QPointF(lx1, ly - 3.5), QPointF(lx1, ly + 2.5))
        painter.drawText(QRectF(bx0, by0 + 2, box_w, fm.height() + 2), Qt.AlignmentFlag.AlignCenter, label)

    def _paint_measurement_callout(
        self,
        painter: QPainter,
        scene: ViewportScene,
        cx0: float,
        cy0: float,
        px_per_a: float,
        right: np.ndarray,
        up: np.ndarray,
        fx0: float,
        fy0: float,
        fw: float,
        fh: float,
        dark_bg: bool,
    ) -> None:
        order = [i for i in scene.selection_order if 0 <= i < len(scene.positions)]
        if len(order) not in (2, 3, 4):
            return
        pts = []
        for idx in order:
            rel = scene.positions[idx] - scene.target
            pts.append(QPointF(cx0 + float(np.dot(rel, right)) * px_per_a, cy0 - float(np.dot(rel, up)) * px_per_a))

        pen = QPen(QColor(45, 212, 191, 240) if dark_bg else QColor(13, 148, 136, 245), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for k in range(len(pts) - 1):
            painter.drawLine(pts[k], pts[k + 1])

        summary = compute_selection_measurement(scene.symbols, scene.positions, order)
        badge = summary.split(": ", 1)[-1].split("  (", 1)[0]
        anchor = QPointF((pts[0].x() + pts[-1].x()) * 0.5, (pts[0].y() + pts[-1].y()) * 0.5)
        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(badge) + 14.0
        bh = fm.height() + 8.0
        bx = max(fx0 + 6.0, min(fx0 + fw - bw - 6.0, anchor.x() - bw * 0.5))
        by = max(fy0 + 6.0, min(fy0 + fh - bh - 6.0, anchor.y() - bh * 0.5))
        rect = QRectF(bx, by, bw, bh)
        painter.setPen(QPen(QColor(45, 212, 191, 245), 1))
        painter.setBrush(QColor(15, 23, 42, 225))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QColor(248, 250, 252, 255))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge)


class AtomicGLViewport(QOpenGLWidget):
    """60fps GPU-instanced OpenGL 3.3 viewport synchronized 1:1 with ``ViewportScene`` and Blender Cycles."""

    camera_changed = Signal()
    camera_settled = Signal()
    atom_clicked = Signal(int, bool)
    interaction_started = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene: ViewportScene | None = None
        self.res_w: int = 1400
        self.res_h: int = 1400
        self.dark_backdrop: bool = True
        self.structure_summary: str = ""
        self.material_finish: str = "studio"

        self._blender_pixmap: QPixmap | None = None
        self._blender_pil: Any = None
        self._fallback_pixmap: QPixmap | None = None

        self._gl_ready: bool = False
        self._gl_failed: bool = not _HAS_PYOPENGL
        self._geom_dirty: bool = True

        self._prog_sphere: int = 0
        self._prog_bond: int = 0
        self._prog_flat: int = 0
        self._vao_sphere: int = 0
        self._vbo_sphere: int = 0
        self._vao_bond: int = 0
        self._vbo_bond: int = 0
        self._vao_lines: int = 0
        self._vbo_lines: int = 0
        self._vao_poly: int = 0
        self._vbo_poly: int = 0

        self._n_atoms: int = 0
        self._n_bonds: int = 0
        self._n_line_verts: int = 0
        self._n_poly_verts: int = 0
        self._z_half_range: float = 50.0

        self._dragging: bool = False
        self._drag_mode: str | None = None
        self._press_pos = QPointF()
        self._last_pos = QPointF()
        self._total_drag_px: float = 0.0
        self._interacting: bool = False

        self._wheel_settle_timer = QTimer(self)
        self._wheel_settle_timer.setSingleShot(True)
        self._wheel_settle_timer.setInterval(250)
        self._wheel_settle_timer.timeout.connect(self._on_wheel_settled)

        self._overlay = _ViewportOverlay(self)
        self._overlay.setGeometry(self.rect())
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def gl_active(self) -> bool:
        return bool(self._gl_ready and not self._gl_failed)

    @property
    def is_interacting(self) -> bool:
        return bool(self._interacting)

    def has_blender_preview(self) -> bool:
        return self._blender_pixmap is not None and not self._blender_pixmap.isNull()

    def active_overlay_pixmap(self) -> QPixmap | None:
        if self._blender_pixmap is not None and not self._blender_pixmap.isNull():
            return self._blender_pixmap
        if self._gl_failed and self._fallback_pixmap is not None and not self._fallback_pixmap.isNull():
            return self._fallback_pixmap
        return None

    def set_scene(
        self,
        scene: ViewportScene | None,
        res_w: int = 1400,
        res_h: int = 1400,
        dark_backdrop: bool = True,
        structure_summary: str = "",
        material_finish: str = "studio",
    ) -> None:
        self.scene = scene
        self.res_w = max(16, int(res_w))
        self.res_h = max(16, int(res_h))
        self.dark_backdrop = bool(dark_backdrop)
        self.structure_summary = structure_summary
        self.material_finish = material_finish
        self.mark_geometry_dirty()
        self.update()
        self._overlay.update()

    def mark_geometry_dirty(self) -> None:
        self._geom_dirty = True
        if self._gl_failed:
            self._refresh_fallback_pixmap()

    def set_blender_preview(self, pil_img_or_pixmap: Any) -> None:
        if pil_img_or_pixmap is None:
            self.clear_blender_preview()
            return
        if isinstance(pil_img_or_pixmap, QPixmap):
            self._blender_pixmap = pil_img_or_pixmap
            self._blender_pil = None
        else:
            self._blender_pil = pil_img_or_pixmap
            self._blender_pixmap = pil_to_qpixmap(pil_img_or_pixmap)
        self._overlay.update()

    def clear_blender_preview(self) -> None:
        if self._blender_pixmap is not None or self._blender_pil is not None:
            self._blender_pixmap = None
            self._blender_pil = None
            self._overlay.update()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
        if self._gl_failed:
            self._refresh_fallback_pixmap()

    def frame_rect(self) -> tuple[float, float, float, float] | None:
        cw = max(1.0, float(self.width()))
        ch = max(1.0, float(self.height()))
        if self._blender_pixmap is not None and not self._blender_pixmap.isNull():
            iw = max(1.0, float(self._blender_pixmap.width()))
            ih = max(1.0, float(self._blender_pixmap.height()))
            pad = min(14.0, min(cw, ch) * 0.12)
            scale = max(1e-4, min((cw - pad) / iw, (ch - pad) / ih))
            fw, fh = iw * scale, ih * scale
            return ((cw - fw) * 0.5, (ch - fh) * 0.5, fw, fh)
        if self.scene is None:
            return None
        pad = min(14.0, min(cw, ch) * 0.12)
        scale = max(1e-4, min((cw - pad) / self.res_w, (ch - pad) / self.res_h))
        fw = self.res_w * scale
        fh = self.res_h * scale
        return ((cw - fw) * 0.5, (ch - fh) * 0.5, fw, fh)

    def _px_per_angstrom_for_frame(self, fw: float, fh: float) -> float:
        ortho = self.scene.ortho_scale if self.scene is not None else 20.0
        return (fw / max(1e-3, ortho)) if self.res_w >= self.res_h else (fh / max(1e-3, ortho))

    def _px_per_angstrom(self) -> float:
        rect = self.frame_rect()
        if rect is None:
            return 20.0
        return self._px_per_angstrom_for_frame(rect[2], rect[3])

    def pick_atom(self, x: float, y: float) -> int | None:
        """Pick the front-most atom at widget coordinates ``(x, y)``."""
        if self.scene is None:
            return None
        rect = self.frame_rect()
        if rect is None:
            return None
        fx0, fy0, fw, fh = rect
        u = (float(x) - fx0) / max(1.0, fw)
        v = (float(y) - fy0) / max(1.0, fh)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return self.scene.pick_atom_at(u, v, self.res_w, self.res_h, frame_w_px=fw, frame_h_px=fh)

    def _axes_center(self) -> tuple[float, float, float] | None:
        if self.scene is None or not self.scene.show_axes:
            return None
        rect = self.frame_rect()
        if rect is None:
            return None
        fx0, fy0, fw, fh = rect
        arm_len = max(24.0, min(38.0, min(fw, fh) * 0.075))
        margin_r = arm_len * 1.45
        cx = max(fx0 + margin_r, min(fx0 + fw - margin_r, fx0 + self.scene.axes_pos[0] * fw))
        cy = max(fy0 + margin_r, min(fy0 + fh - margin_r, fy0 + self.scene.axes_pos[1] * fh))
        return (cx, cy, max(36.0, arm_len * 1.25))

    def _refresh_fallback_pixmap(self) -> None:
        if self.scene is None:
            self._fallback_pixmap = None
            return
        try:
            img = self.scene.render_image(
                max(64, self.width()),
                max(64, self.height()),
                self.res_w,
                self.res_h,
                interactive=self._interacting,
            )
            self._fallback_pixmap = pil_to_qpixmap(img)
        except Exception:
            self._fallback_pixmap = None

    def initializeGL(self) -> None:
        if _HAS_PYOPENGL is False:
            self._gl_failed = True
            return
        try:
            self._prog_sphere = _compile_program(_SPHERE_VERT, _SPHERE_FRAG)
            self._prog_bond = _compile_program(_BOND_VERT, _BOND_FRAG)
            self._prog_flat = _compile_program(_FLAT_VERT, _FLAT_FRAG)

            self._vao_sphere = int(gl.glGenVertexArrays(1))
            self._vbo_sphere = int(gl.glGenBuffers(1))
            gl.glBindVertexArray(self._vao_sphere)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_sphere)
            stride_s = 12 * 4
            for loc, cnt, off in ((0, 3, 0), (1, 1, 3 * 4), (2, 4, 4 * 4), (3, 4, 8 * 4)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, cnt, gl.GL_FLOAT, gl.GL_FALSE, stride_s, ctypes.c_void_p(off))
                gl.glVertexAttribDivisor(loc, 1)

            self._vao_bond = int(gl.glGenVertexArrays(1))
            self._vbo_bond = int(gl.glGenBuffers(1))
            gl.glBindVertexArray(self._vao_bond)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_bond)
            stride_b = 11 * 4
            for loc, cnt, off in ((0, 3, 0), (1, 3, 3 * 4), (2, 1, 6 * 4), (3, 4, 7 * 4)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, cnt, gl.GL_FLOAT, gl.GL_FALSE, stride_b, ctypes.c_void_p(off))
                gl.glVertexAttribDivisor(loc, 1)

            self._vao_lines = int(gl.glGenVertexArrays(1))
            self._vbo_lines = int(gl.glGenBuffers(1))
            gl.glBindVertexArray(self._vao_lines)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_lines)
            stride_f = 7 * 4
            for loc, cnt, off in ((0, 3, 0), (1, 4, 3 * 4)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, cnt, gl.GL_FLOAT, gl.GL_FALSE, stride_f, ctypes.c_void_p(off))

            self._vao_poly = int(gl.glGenVertexArrays(1))
            self._vbo_poly = int(gl.glGenBuffers(1))
            gl.glBindVertexArray(self._vao_poly)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_poly)
            for loc, cnt, off in ((0, 3, 0), (1, 4, 3 * 4)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, cnt, gl.GL_FLOAT, gl.GL_FALSE, stride_f, ctypes.c_void_p(off))

            gl.glBindVertexArray(0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            self._gl_ready = True
            self._geom_dirty = True
        except Exception:
            self._gl_failed = True
            self._gl_ready = False
            self._refresh_fallback_pixmap()

    def _upload_scene_buffers(self) -> None:
        self._geom_dirty = False
        if not self._gl_ready or self.scene is None:
            self._n_atoms = 0
            self._n_bonds = 0
            self._n_line_verts = 0
            self._n_poly_verts = 0
            return

        scene = self.scene
        if len(scene.positions) > 0:
            span = float(np.max(np.linalg.norm(scene.positions - scene.target, axis=1)))
            self._z_half_range = max(25.0, span * 2.5, scene.ortho_scale * 1.5)
        else:
            self._z_half_range = 50.0

        # 1. Atom instances
        if scene.show_atoms and len(scene.positions) > 0:
            n = len(scene.positions)
            rough_map = {"studio": 0.32, "chalk": 0.88, "flat": 1.0, "metallic": 0.18}
            metal_map = {"studio": 0.12, "chalk": 0.0, "flat": 0.0, "metallic": 0.82}
            r_def = rough_map.get(self.material_finish, 0.32)
            m_def = metal_map.get(self.material_finish, 0.12)

            arr = np.zeros((n, 12), dtype=np.float32)
            arr[:, 0:3] = scene.positions.astype(np.float32)
            arr[:, 3] = scene.radii.astype(np.float32)
            for idx in range(n):
                r_c, g_c, b_c = _hex_to_rgb01(scene.colors[idx])
                arr[idx, 4] = r_c
                arr[idx, 5] = g_c
                arr[idx, 6] = b_c
                arr[idx, 7] = float(scene.atom_alphas[idx]) if idx < len(scene.atom_alphas) else 1.0
                arr[idx, 8] = m_def
                arr[idx, 9] = r_def
                arr[idx, 10] = 1.0 if (scene.show_selection_highlight and idx in scene.selected_atoms) else 0.0
                s_key = scene._sprite_keys[idx] if idx < len(scene._sprite_keys) else scene.symbols[idx]
                arr[idx, 11] = 1.0 if (s_key in scene._outline_symbols or (scene.dark_backdrop and s_key in scene._dark_outline_symbols)) else 0.0

            arr_c = np.ascontiguousarray(arr, dtype=np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_sphere)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, arr_c.nbytes, arr_c, gl.GL_STATIC_DRAW)
            self._n_atoms = n
        else:
            self._n_atoms = 0

        # 2. Bond instances
        if scene.show_atoms and len(scene.bond_i) > 0:
            seg_p0, seg_p1, seg_a, seg_b, seg_colors = scene._get_trimmed_bond_segments()
            nb = len(seg_p0)
            if nb > 0:
                barr = np.zeros((nb, 11), dtype=np.float32)
                barr[:, 0:3] = seg_p0.astype(np.float32)
                barr[:, 3:6] = seg_p1.astype(np.float32)
                barr[:, 6] = float(scene.bond_radius)
                for k in range(nb):
                    r_c, g_c, b_c = _hex_to_rgb01(seg_colors[k])
                    a_k = min(
                        float(scene.atom_alphas[seg_a[k]]) if len(scene.atom_alphas) == len(scene.positions) else 1.0,
                        float(scene.atom_alphas[seg_b[k]]) if len(scene.atom_alphas) == len(scene.positions) else 1.0,
                    )
                    barr[k, 7:11] = (r_c, g_c, b_c, a_k)
                barr_c = np.ascontiguousarray(barr, dtype=np.float32)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_bond)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, barr_c.nbytes, barr_c, gl.GL_STATIC_DRAW)
                self._n_bonds = nb
            else:
                self._n_bonds = 0
        else:
            self._n_bonds = 0

        # 3. Line segments (cell edges, cell vectors, dashed H-bonds/contacts, polyhedra edges)
        line_verts: list[list[float]] = []
        if len(scene.cell_edges) > 0:
            cr, cg, cb = _hex_to_rgb01(scene.cell_color)
            lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
            if scene.dark_backdrop and lum < 0.25:
                cr, cg, cb = (0.79, 0.84, 0.88)
            for edge in scene.cell_edges:
                line_verts.append([float(edge[0, 0]), float(edge[0, 1]), float(edge[0, 2]), cr, cg, cb, 0.90])
                line_verts.append([float(edge[1, 0]), float(edge[1, 1]), float(edge[1, 2]), cr, cg, cb, 0.90])

        if scene.has_cell and scene.show_cell_vectors and len(scene.cell_edges) > 0:
            vec_defs = (
                (scene.cell_vecs[0], (0.94, 0.27, 0.27)),
                (scene.cell_vecs[1], (0.13, 0.77, 0.37)),
                (scene.cell_vecs[2], (0.23, 0.51, 0.96)),
            )
            for vec, (vr, vg, vb) in vec_defs:
                vlen = float(np.linalg.norm(vec))
                if vlen < 1e-6:
                    continue
                tip = scene.origin + vec * min(0.45, 2.8 / max(1e-3, vlen))
                line_verts.append([float(scene.origin[0]), float(scene.origin[1]), float(scene.origin[2]), vr, vg, vb, 0.95])
                line_verts.append([float(tip[0]), float(tip[1]), float(tip[2]), vr, vg, vb, 0.95])

        if scene.show_atoms and scene.dashed_pairs and len(scene.positions) > 0:
            dr, dg, db = _hex_to_rgb01(scene.dashed_color)
            for ia, ib in scene.dashed_pairs:
                if 0 <= ia < len(scene.positions) and 0 <= ib < len(scene.positions):
                    p0, p1 = scene.positions[ia], scene.positions[ib]
                    n_dash = 7
                    for d in range(0, n_dash, 2):
                        t0 = d / n_dash
                        t1 = min(1.0, (d + 0.65) / n_dash)
                        q0 = p0 * (1.0 - t0) + p1 * t0
                        q1 = p0 * (1.0 - t1) + p1 * t1
                        line_verts.append([float(q0[0]), float(q0[1]), float(q0[2]), dr, dg, db, 0.90])
                        line_verts.append([float(q1[0]), float(q1[1]), float(q1[2]), dr, dg, db, 0.90])

        poly_verts: list[list[float]] = []
        if scene.polyhedra:
            for poly in scene.polyhedra:
                verts = poly.get("vertices", [])
                faces = poly.get("faces", [])
                edges = poly.get("edges", [])
                pr, pg, pb = _hex_to_rgb01(str(poly.get("color", "#38BDF8")))
                pa = float(poly.get("alpha", 0.36))
                for f in faces:
                    if len(f) == 3:
                        for vi in f:
                            pt = verts[int(vi)]
                            poly_verts.append([float(pt[0]), float(pt[1]), float(pt[2]), pr, pg, pb, pa])
                er, eg, eb = min(1.0, pr + 0.18), min(1.0, pg + 0.18), min(1.0, pb + 0.18)
                for e in edges:
                    if len(e) == 2:
                        pa_pt, pb_pt = verts[int(e[0])], verts[int(e[1])]
                        line_verts.append([float(pa_pt[0]), float(pa_pt[1]), float(pa_pt[2]), er, eg, eb, 0.72])
                        line_verts.append([float(pb_pt[0]), float(pb_pt[1]), float(pb_pt[2]), er, eg, eb, 0.72])

        if line_verts:
            larr = np.ascontiguousarray(np.asarray(line_verts, dtype=np.float32))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_lines)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, larr.nbytes, larr, gl.GL_STATIC_DRAW)
            self._n_line_verts = len(line_verts)
        else:
            self._n_line_verts = 0

        if poly_verts:
            parr = np.ascontiguousarray(np.asarray(poly_verts, dtype=np.float32))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_poly)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, parr.nbytes, parr, gl.GL_STATIC_DRAW)
            self._n_poly_verts = len(poly_verts)
        else:
            self._n_poly_verts = 0

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _set_common_uniforms(self, prog: int, px_per_a: float) -> None:
        assert self.scene is not None
        right, up, back = quat_to_axes(self.scene.quat)
        tgt = self.scene.target
        cw = max(1.0, float(self.width()))
        ch = max(1.0, float(self.height()))
        gl.glUniform3f(gl.glGetUniformLocation(prog, "uTarget"), float(tgt[0]), float(tgt[1]), float(tgt[2]))
        gl.glUniform3f(gl.glGetUniformLocation(prog, "uCamRight"), float(right[0]), float(right[1]), float(right[2]))
        gl.glUniform3f(gl.glGetUniformLocation(prog, "uCamUp"), float(up[0]), float(up[1]), float(up[2]))
        gl.glUniform3f(gl.glGetUniformLocation(prog, "uCamBack"), float(back[0]), float(back[1]), float(back[2]))
        gl.glUniform1f(gl.glGetUniformLocation(prog, "uPxPerA"), float(px_per_a))
        gl.glUniform2f(gl.glGetUniformLocation(prog, "uHalfCanvasPx"), float(cw * 0.5), float(ch * 0.5))
        gl.glUniform1f(gl.glGetUniformLocation(prog, "uZHalfRange"), float(self._z_half_range))

    def paintGL(self) -> None:
        if not self._gl_ready or self._gl_failed:
            return
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        dark_bg = self.scene.dark_backdrop if self.scene is not None else self.dark_backdrop
        if dark_bg:
            gl.glClearColor(0.082, 0.090, 0.110, 1.0)
        else:
            gl.glClearColor(0.910, 0.918, 0.929, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self.scene is None:
            return
        if self._geom_dirty:
            self._upload_scene_buffers()

        px_per_a = self._px_per_angstrom()

        # 1. Cell / lines
        if self._n_line_verts > 0:
            gl.glUseProgram(self._prog_flat)
            self._set_common_uniforms(self._prog_flat, px_per_a)
            gl.glBindVertexArray(self._vao_lines)
            gl.glDrawArrays(gl.GL_LINES, 0, self._n_line_verts)

        # 2. Bonds
        if self._n_bonds > 0:
            gl.glUseProgram(self._prog_bond)
            self._set_common_uniforms(self._prog_bond, px_per_a)
            gl.glBindVertexArray(self._vao_bond)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, self._n_bonds)

        # 3. Atoms (instanced ray-traced spheres)
        if self._n_atoms > 0:
            gl.glUseProgram(self._prog_sphere)
            self._set_common_uniforms(self._prog_sphere, px_per_a)
            gl.glUniform1f(gl.glGetUniformLocation(self._prog_sphere, "uDarkBackdrop"), 1.0 if dark_bg else 0.0)
            gl.glBindVertexArray(self._vao_sphere)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, self._n_atoms)

        # 4. Translucent coordination polyhedra faces (depth write disabled so spheres inside stay crisp)
        if self._n_poly_verts > 0:
            gl.glDepthMask(gl.GL_FALSE)
            gl.glUseProgram(self._prog_flat)
            self._set_common_uniforms(self._prog_flat, px_per_a)
            gl.glBindVertexArray(self._vao_poly)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._n_poly_verts)
            gl.glDepthMask(gl.GL_TRUE)

        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def render_snapshot_pil(self) -> Any:
        """Return a PIL RGBA image of the current Blender preview or live OpenGL / fallback viewport."""
        from PIL import Image

        if self._blender_pil is not None:
            return self._blender_pil.copy()
        if self.gl_active:
            qimg = self.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
            bits = bytes(qimg.constBits())
            return Image.frombytes("RGBA", (qimg.width(), qimg.height()), bits, "raw", "RGBA")
        if self.scene is not None:
            return self.scene.render_image(
                max(128, self.width()),
                max(128, self.height()),
                self.res_w,
                self.res_h,
                interactive=False,
            )
        return None

    def mousePressEvent(self, event: Any) -> None:
        if self.scene is None:
            return
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        btn = event.button()

        # Check Reference Axes gizmo hit on left press
        if btn == Qt.MouseButton.LeftButton:
            ax_hit = self._axes_center()
            if ax_hit is not None:
                ax_cx, ax_cy, hit_r = ax_hit
                if (pos.x() - ax_cx) ** 2 + (pos.y() - ax_cy) ** 2 <= hit_r ** 2:
                    self._dragging = True
                    self._drag_mode = "axes"
                    self._press_pos = pos
                    self._last_pos = pos
                    self._total_drag_px = 0.0
                    return

        self.clear_blender_preview()
        self._dragging = True
        self._interacting = True
        self._press_pos = pos
        self._last_pos = pos
        self._total_drag_px = 0.0

        is_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or is_shift:
            self._drag_mode = "pan"
        else:
            self._drag_mode = "orbit"
        self.interaction_started.emit()
        self.update()
        self._overlay.update()

    def mouseMoveEvent(self, event: Any) -> None:
        if not self._dragging or self.scene is None:
            return
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        self._total_drag_px += math.hypot(dx, dy)
        if dx == 0.0 and dy == 0.0:
            return

        if self._drag_mode == "axes":
            rect = self.frame_rect()
            if rect is not None:
                fx0, fy0, fw, fh = rect
                u = max(0.06, min(0.94, (pos.x() - fx0) / max(1.0, fw)))
                v = max(0.06, min(0.94, (pos.y() - fy0) / max(1.0, fh)))
                self.scene.axes_pos = (u, v)
                self._overlay.update()
            return

        if self._drag_mode == "orbit":
            self.scene.orbit(dx * 0.0085, dy * 0.0085)
        elif self._drag_mode == "pan":
            px_per_a = self._px_per_angstrom()
            self.scene.pan(dx / max(1e-3, px_per_a), dy / max(1e-3, px_per_a))

        if self._gl_failed:
            self._refresh_fallback_pixmap()
        self.update()
        self._overlay.update()
        self.camera_changed.emit()

    def mouseReleaseEvent(self, event: Any) -> None:
        if not self._dragging:
            return
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        was_axes = self._drag_mode == "axes"
        was_click = self._total_drag_px < 4.0 and event.button() == Qt.MouseButton.LeftButton and not was_axes

        self._dragging = False
        self._drag_mode = None
        self._interacting = False

        if was_axes:
            self._overlay.update()
            self.camera_settled.emit()
            return

        if was_click:
            idx = self.pick_atom(pos.x(), pos.y())
            if idx is not None:
                additive = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
                self.atom_clicked.emit(idx, additive)
                return

        if self._gl_failed:
            self._refresh_fallback_pixmap()
        self.update()
        self._overlay.update()
        self.camera_settled.emit()

    def wheelEvent(self, event: Any) -> None:
        if self.scene is None:
            return
        delta = event.angleDelta().y() if hasattr(event, "angleDelta") else 0
        if delta == 0:
            return
        self.clear_blender_preview()
        self._interacting = True
        factor = 0.9 if delta > 0 else 1.111
        self.scene.zoom(factor)
        if self._gl_failed:
            self._refresh_fallback_pixmap()
        self.update()
        self._overlay.update()
        self.camera_changed.emit()
        self._wheel_settle_timer.start()

    def _on_wheel_settled(self) -> None:
        self._interacting = False
        if self._gl_failed:
            self._refresh_fallback_pixmap()
        self.update()
        self._overlay.update()
        self.camera_settled.emit()

    def closeEvent(self, event: Any) -> None:
        if self._gl_ready and not self._gl_failed:
            try:
                self.makeCurrent()
                vaos = [v for v in (self._vao_sphere, self._vao_bond, self._vao_lines, self._vao_poly) if v]
                vbos = [v for v in (self._vbo_sphere, self._vbo_bond, self._vbo_lines, self._vbo_poly) if v]
                if vaos:
                    gl.glDeleteVertexArrays(len(vaos), vaos)
                if vbos:
                    gl.glDeleteBuffers(len(vbos), vbos)
                for prog in (self._prog_sphere, self._prog_bond, self._prog_flat):
                    if prog:
                        gl.glDeleteProgram(prog)
                self.doneCurrent()
            except Exception:
                pass
        super().closeEvent(event)
