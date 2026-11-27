"""Standalone desktop GUI client for atomic-blender built on Python tkinter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, scrolledtext, ttk
from typing import Any
import webbrowser

import numpy as np

from atomic_blender.cli import (
    BLENDER_DOWNLOAD_URL,
    _collect_style_files,
    ensure_app_icon,
    find_blender_executable,
    install_desktop_shortcut,
    package_dir,
    resolve_style_path,
)
from atomic_blender.elements import METAL_ELEMENTS, VIEW_PRESETS, classify_dropped_file
from atomic_blender.viewport3d import (
    ViewportScene,
    atom_label,
    choose_scale_bar_angstrom,
    compose_dual_view_panel,
    compose_multi_panel_strip,
    compute_cell_parameters,
    compute_slice_mask,
    compute_z_colormap_colors,
    find_coordination_polyhedra,
    find_dashed_contacts,
    is_tilted_cell,
    merge_ghost_structure,
    preset_to_quat,
    quat_to_axes,
    render_atom_labels_and_selection_overlay,
    render_colorbar_overlay,
    render_measurement_callout_overlay,
    render_reference_axes_overlay,
    render_scale_bar_overlay,
    sample_colormap_hex,
)

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk  # type: ignore

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

WELCOME_FRAME_INTERVAL_MS = 95

WELCOME_STYLE_PHASES: tuple[tuple[str, str, str], ...] = (
    ("Studio  ·  Ball & Stick", "Balanced specular lighting with dual-color covalent bonds", "#2DD4BF"),
    ("Matte Chalk  ·  Low Glare", "Reflection-free diffuse shading for high-contrast figures", "#38BDF8"),
    ("Licorice  ·  Bond Network", "Uniform cylinder framework highlighting coordination topology", "#A78BFA"),
    ("Van der Waals  ·  Space-Filling", "Volumetric atomic radii for packing and surface inspection", "#F59E0B"),
)


def extract_input_path(raw_val: str) -> str:
    """Extract the filesystem path from either a raw path or ``filename (full_path)`` display string."""
    s = str(raw_val or "").strip()
    if s.endswith(")") and " (" in s:
        idx = s.rfind(" (")
        label = s[:idx].strip()
        candidate = s[idx + 2 : -1].strip()
        if candidate and Path(candidate).name == label:
            return candidate
    return s


def format_input_display(path_or_str: str | Path) -> str:
    """Format a file path for the Input box as ``filename (full_path)`` when a parent path is present."""
    raw = extract_input_path(str(path_or_str))
    if not raw:
        return ""
    name = Path(raw).name
    if name and name != raw:
        return f"{name} ({raw})"
    return raw


def _blend_hex_color(bg_hex: str, fg_hex: str, alpha: float) -> str:
    """Blend two ``#RRGGBB`` hex colors by ``alpha`` in ``[0, 1]`` for canvas text fades."""
    a = max(0.0, min(1.0, float(alpha)))
    b = bg_hex.lstrip("#")
    f = fg_hex.lstrip("#")
    if len(b) != 6 or len(f) != 6:
        return fg_hex
    r = int(round((1.0 - a) * int(b[0:2], 16) + a * int(f[0:2], 16)))
    g = int(round((1.0 - a) * int(b[2:4], 16) + a * int(f[2:4], 16)))
    bl = int(round((1.0 - a) * int(b[4:6], 16) + a * int(f[4:6], 16)))
    return f"#{r:02X}{g:02X}{bl:02X}"


def _welcome_header_fade_alpha(step: int) -> float:
    """One-shot header opacity: starts dim (0.08), brightens to 0.76, fades to 0.16 and holds."""
    s = max(0, int(step))
    if s <= 0:
        return 0.08
    if s <= 16:
        return 0.08 + 0.68 * (0.5 - 0.5 * math.cos(math.pi * s / 16.0))
    if s < 36:
        return 0.76 - 0.60 * (0.5 - 0.5 * math.cos(math.pi * (s - 16) / 20.0))
    return 0.16


_welcome_title_fade_alpha = _welcome_header_fade_alpha


def _welcome_prompt_fade_alpha(step: int, period: int = 64) -> float:
    """Slow repeating breathing opacity in ``[0.22, 1.0]`` for bottom drag & drop prompts."""
    p = max(2, int(period))
    s = max(0, int(step)) % p
    return 0.61 - 0.39 * math.cos(2.0 * math.pi * s / float(p))


@dataclass
class GuiRenderConfig:
    """Typed render configuration edited by the desktop GUI."""

    input_path: str
    output_path: str = ""
    style: str = "default"
    view_mode: str = "preset"  # "preset" or "file"
    view_preset: str = "iso"
    view_file: str = ""
    fit_view: bool = False
    type_map: str = ""
    replicate: tuple[int, int, int] = (1, 1, 1)
    wrap: bool = True
    crop_z_mode: str = "none"  # "none", "auto", "custom"
    crop_z_custom: str = "2.0,4.0"
    bonds: str = "style"  # "style", "table", "auto", "none"
    bond_color: str = "style"  # "style", "bicolor", "single"
    radius_scale: float = 1.0
    show_cell: bool = True
    resolution: tuple[int, int] = (1400, 1400)
    samples: int = 64
    blender_exe: str = ""
    save_blend: str = ""


def validate_gui_config(cfg: GuiRenderConfig) -> Path:
    """Validate GUI fields before invoking the CLI renderer and return resolved input path."""
    raw_in = extract_input_path(cfg.input_path)
    if not raw_in:
        raise ValueError("Please select an input structure file.")
    in_path = Path(raw_in).expanduser()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input structure file not found: {in_path}")

    if cfg.view_mode == "file":
        if not cfg.view_file.strip():
            raise ValueError("Please select a view.json file or switch to Preset mode.")
        vf = Path(cfg.view_file).expanduser()
        if not vf.is_file():
            raise FileNotFoundError(f"View JSON file not found: {vf}")

    if cfg.resolution[0] < 16 or cfg.resolution[1] < 16:
        raise ValueError("Resolution width and height must be at least 16 px.")
    if cfg.samples < 1:
        raise ValueError("Cycles samples must be >= 1.")
    if any(n < 1 for n in cfg.replicate):
        raise ValueError("Replication counts must be >= 1.")
    if cfg.radius_scale <= 0.0:
        raise ValueError("Atom scale must be > 0.")
    return in_path


def build_render_cli_args(
    cfg: GuiRenderConfig,
    out_png: Path | None = None,
    save_blend: Path | None = None,
) -> list[str]:
    """Translate ``GuiRenderConfig`` into arguments for ``python -m atomic_blender.cli``."""
    view_arg = cfg.view_file.strip() if cfg.view_mode == "file" else cfg.view_preset
    args: list[str] = [
        "render",
        cfg.input_path.strip(),
        "--style",
        cfg.style.strip() or "default",
        "--view",
        view_arg,
        "--replicate",
        str(cfg.replicate[0]),
        str(cfg.replicate[1]),
        str(cfg.replicate[2]),
        "--resolution",
        str(cfg.resolution[0]),
        str(cfg.resolution[1]),
        "--samples",
        str(cfg.samples),
    ]

    if not cfg.wrap:
        args.append("--no-wrap")
    if cfg.view_mode == "file" and cfg.fit_view:
        args.append("--fit-view")
    if cfg.type_map.strip():
        args.extend(["--type-map", cfg.type_map.strip()])
    if cfg.crop_z_mode == "auto":
        args.extend(["--crop-z", "auto"])
    elif cfg.crop_z_mode == "custom" and cfg.crop_z_custom.strip():
        args.extend(["--crop-z", cfg.crop_z_custom.strip()])
    if cfg.bonds in ("table", "auto", "none"):
        args.extend(["--bonds", cfg.bonds])
    if cfg.bond_color in ("single", "bicolor"):
        args.extend(["--bond-color", cfg.bond_color])
    if abs(cfg.radius_scale - 1.0) > 1e-6:
        args.extend(["--radius-scale", f"{cfg.radius_scale:.3f}"])
    if not cfg.show_cell:
        args.append("--no-cell")
    if cfg.blender_exe.strip():
        args.extend(["--blender", cfg.blender_exe.strip()])

    png_target = out_png if out_png is not None else (Path(cfg.output_path) if cfg.output_path.strip() else None)
    if png_target is not None:
        args.extend(["--out", str(png_target)])

    blend_target = save_blend if save_blend is not None else (Path(cfg.save_blend) if cfg.save_blend.strip() else None)
    if blend_target is not None:
        args.extend(["--save-blend", str(blend_target)])

    return args


def compute_preview_resolution(res_w: int, res_h: int, max_side: int = 400) -> tuple[int, int]:
    """Scale target output resolution down for fast interactive Blender previews."""
    rw = max(16, int(res_w))
    rh = max(16, int(res_h))
    longest = max(rw, rh)
    if longest <= max_side:
        return rw, rh
    scale = max_side / float(longest)
    return max(16, int(round(rw * scale))), max(16, int(round(rh * scale)))


class DarkCheckbutton(tk.Frame):
    """Crisp pixel-aligned dark-theme checkbox backed by a BooleanVar."""

    def __init__(
        self,
        parent: Any,
        text: str,
        variable: tk.BooleanVar,
        command: Any = None,
        bg: str = "#171A21",
        fg: str = "#E2E8F0",
        font: tuple[str, int] = ("Segoe UI", 9),
    ) -> None:
        super().__init__(parent, bg=bg)
        self._var = variable
        self._cmd = command
        self._bg = bg
        self._fg = fg
        self._disabled = False

        self._box = tk.Canvas(self, width=18, height=18, bg=bg, highlightthickness=0, cursor="hand2")
        self._box.pack(side=tk.LEFT, padx=(0, 5))
        self._lbl = tk.Label(self, text=text, bg=bg, fg=fg, font=font, cursor="hand2")
        self._lbl.pack(side=tk.LEFT)

        self._box.bind("<Button-1>", self._on_click)
        self._lbl.bind("<Button-1>", self._on_click)
        self._trace_id = self._var.trace_add("write", lambda *_: self._safe_redraw())
        self.bind("<Destroy>", self._on_destroy)
        self._redraw()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            try:
                self._var.trace_remove("write", self._trace_id)
            except Exception:
                pass

    def _on_click(self, _event: tk.Event) -> None:
        if self._disabled:
            return
        self._var.set(not bool(self._var.get()))
        self._redraw()
        if self._cmd is not None:
            self._cmd()

    def _safe_redraw(self) -> None:
        if self.winfo_exists():
            self._redraw()

    def _redraw(self) -> None:
        self._box.delete("all")
        checked = bool(self._var.get())
        if self._disabled:
            fill = "#181B21"
            outline = "#3F4654"
            check_col = "#64748B"
            self._lbl.configure(fg="#64748B")
        else:
            fill = "#0D9488" if checked else "#111318"
            outline = "#14B8A6" if checked else "#475569"
            check_col = "#FFFFFF"
            self._lbl.configure(fg=self._fg)
        self._box.create_rectangle(2, 2, 15, 15, fill=fill, outline=outline, width=1.4)
        if checked:
            self._box.create_line(
                5, 9, 7.5, 11.5, 12.5, 6, fill=check_col, width=1.8, capstyle=tk.ROUND, joinstyle=tk.ROUND
            )

    def configure(self, cnf: Any = None, **kw: Any) -> Any:  # type: ignore[override]
        if "state" in kw:
            state = kw.pop("state")
            self._disabled = str(state) == "disabled"
            cursor = "arrow" if self._disabled else "hand2"
            self._box.configure(cursor=cursor)
            self._lbl.configure(cursor=cursor)
            self._redraw()
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None


class DarkRadiobutton(tk.Frame):
    """Crisp dark-theme radio button backed by a StringVar."""

    def __init__(
        self,
        parent: Any,
        text: str,
        variable: tk.StringVar,
        value: str,
        command: Any = None,
        bg: str = "#171A21",
        fg: str = "#E2E8F0",
        font: tuple[str, int] = ("Segoe UI", 9),
    ) -> None:
        super().__init__(parent, bg=bg)
        self._var = variable
        self._value = value
        self._cmd = command
        self._bg = bg
        self._fg = fg

        self._dot = tk.Canvas(self, width=18, height=18, bg=bg, highlightthickness=0, cursor="hand2")
        self._dot.pack(side=tk.LEFT, padx=(0, 5))
        self._lbl = tk.Label(self, text=text, bg=bg, fg=fg, font=font, cursor="hand2")
        self._lbl.pack(side=tk.LEFT)

        self._dot.bind("<Button-1>", self._on_click)
        self._lbl.bind("<Button-1>", self._on_click)
        self._trace_id = self._var.trace_add("write", lambda *_: self._safe_redraw())
        self.bind("<Destroy>", self._on_destroy)
        self._redraw()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            try:
                self._var.trace_remove("write", self._trace_id)
            except Exception:
                pass

    def _on_click(self, _event: tk.Event) -> None:
        self._var.set(self._value)
        self._redraw()
        if self._cmd is not None:
            self._cmd()

    def _safe_redraw(self) -> None:
        if self.winfo_exists():
            self._redraw()

    def _redraw(self) -> None:
        self._dot.delete("all")
        selected = self._var.get() == self._value
        outline = "#14B8A6" if selected else "#475569"
        self._dot.create_oval(2, 2, 15, 15, fill="#111318", outline=outline, width=1.5)
        if selected:
            self._dot.create_oval(5, 5, 12, 12, fill="#14B8A6", outline="")


class DarkSpinbox(tk.Frame):
    """Compact dark stepper widget ([-][Entry][+]) replacing ttk.Spinbox."""

    def __init__(
        self,
        parent: Any,
        variable: tk.IntVar,
        from_: int = 1,
        to: int = 100,
        increment: int = 1,
        width: int = 4,
        command: Any = None,
    ) -> None:
        super().__init__(parent, bg="#111318", highlightbackground="#262B36", highlightthickness=1)
        self._var = variable
        self._from = int(from_)
        self._to = int(to)
        self._inc = int(increment)
        self._cmd = command

        btn_cfg = {
            "bg": "#222631",
            "fg": "#CBD5E1",
            "activebackground": "#2D3342",
            "activeforeground": "#FFFFFF",
            "bd": 0,
            "relief": tk.FLAT,
            "width": 2,
            "cursor": "hand2",
            "font": ("Segoe UI", 9, "bold"),
        }
        self._btn_dec = tk.Button(self, text="-", command=lambda: self._step(-self._inc), **btn_cfg)
        self._btn_dec.pack(side=tk.LEFT, fill=tk.Y)

        self._entry = tk.Entry(
            self,
            textvariable=self._var,
            width=width,
            justify=tk.CENTER,
            bg="#111318",
            fg="#E2E8F0",
            insertbackground="#E2E8F0",
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 9),
        )
        self._entry.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self._entry.bind("<Return>", lambda _e: self._commit_entry())
        self._entry.bind("<FocusOut>", lambda _e: self._commit_entry())
        self._entry.bind(
            "<MouseWheel>",
            lambda e: self._step(self._inc if e.delta > 0 else -self._inc),
        )

        self._btn_inc = tk.Button(self, text="+", command=lambda: self._step(self._inc), **btn_cfg)
        self._btn_inc.pack(side=tk.LEFT, fill=tk.Y)

    def _clamp(self, val: int) -> int:
        return max(self._from, min(self._to, val))

    def _step(self, delta: int) -> None:
        try:
            cur = int(self._var.get())
        except Exception:
            cur = self._from
        nxt = self._clamp(cur + delta)
        self._var.set(nxt)
        if self._cmd is not None:
            self._cmd()

    def _commit_entry(self) -> None:
        try:
            cur = int(self._var.get())
        except Exception:
            cur = self._from
        clamped = self._clamp(cur)
        if cur != clamped:
            self._var.set(clamped)
        if self._cmd is not None:
            self._cmd()


class DarkSlider(tk.Canvas):
    """Sleek modern horizontal slider with prominent circular thumb for dark themes."""

    def __init__(
        self,
        parent: Any,
        from_: float,
        to: float,
        variable: tk.DoubleVar,
        command: Any = None,
        release_command: Any = None,
        bg: str = "#171A21",
        height: int = 24,
    ) -> None:
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, cursor="hand2")
        self._from = float(from_)
        self._to = float(to)
        self._var = variable
        self._cmd = command
        self._release_cmd = release_command
        self._bg = bg
        self._hover = False
        self._dragging = False
        self._pad_x = 11

        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._trace_id = self._var.trace_add("write", lambda *_: self._safe_redraw())
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            try:
                self._var.trace_remove("write", self._trace_id)
            except Exception:
                pass

    def _set_hover(self, state: bool) -> None:
        self._hover = state
        self._redraw()

    def _value_from_x(self, x: float) -> float:
        w = max(40, self.winfo_width())
        span = max(1.0, float(w - 2 * self._pad_x))
        frac = max(0.0, min(1.0, (x - self._pad_x) / span))
        return self._from + frac * (self._to - self._from)

    def _on_press(self, event: tk.Event) -> None:
        self._dragging = True
        self.set_value(self._value_from_x(event.x))

    def _on_motion(self, event: tk.Event) -> None:
        if self._dragging:
            self.set_value(self._value_from_x(event.x))

    def _on_release(self, _event: tk.Event) -> None:
        if self._dragging:
            self._dragging = False
            self._redraw()
            if self._release_cmd is not None:
                self._release_cmd()

    def set_value(self, val: float) -> None:
        clamped = max(self._from, min(self._to, float(val)))
        self._var.set(clamped)
        self._redraw()
        if self._cmd is not None:
            self._cmd(f"{clamped:.4f}")

    def _safe_redraw(self) -> None:
        if self.winfo_exists():
            self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(40, self.winfo_width())
        h = max(20, self.winfo_height())
        cy = h * 0.5
        x0 = self._pad_x
        x1 = w - self._pad_x
        if x1 <= x0:
            return
        try:
            cur = float(self._var.get())
        except Exception:
            cur = self._from
        frac = max(0.0, min(1.0, (cur - self._from) / max(1e-9, self._to - self._from)))
        tx = x0 + frac * (x1 - x0)

        # Minimal recessed track + teal active segment
        self.create_line(x0, cy, x1, cy, fill="#222631", width=4, capstyle=tk.ROUND)
        if tx > x0 + 1:
            self.create_line(x0, cy, tx, cy, fill="#0D9488", width=4, capstyle=tk.ROUND)

        r = 7 if (self._hover or self._dragging) else 6
        thumb_fill = "#2DD4BF" if (self._hover or self._dragging) else "#14B8A6"
        self.create_oval(tx - r, cy - r, tx + r, cy + r, fill=thumb_fill, outline="#0F1115", width=1.4)
        self.create_oval(tx - 2.5, cy - 2.5, tx + 2.5, cy + 2.5, fill="#0F172A", outline="")


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


MATERIAL_FINISH_PRESETS: dict[str, dict[str, float]] = {
    "studio": {},
    "chalk": {
        "roughness": 0.88,
        "specular": 0.12,
        "metallic": 0.0,
        "coat_weight": 0.0,
        "ao_factor": 0.35,
    },
    "flat": {
        "roughness": 1.0,
        "specular": 0.0,
        "metallic": 0.0,
        "coat_weight": 0.0,
        "ao_factor": 0.0,
    },
    "metallic": {
        "roughness": 0.18,
        "specular": 0.85,
        "metallic": 0.80,
        "coat_weight": 0.25,
        "ao_factor": 0.20,
    },
}


def resolve_batch_output_paths(input_paths: list[Path], out_dir: Path) -> list[Path]:
    """Generate unique output PNG paths in ``out_dir`` even when inputs share identical stems."""
    paths = [Path(p) for p in input_paths]
    stem_counts: dict[str, int] = {}
    for p in paths:
        k = p.stem.lower()
        stem_counts[k] = stem_counts.get(k, 0) + 1
    used_lower: set[str] = set()
    resolved: list[Path] = []
    for p in paths:
        ext_tag = p.suffix.lstrip(".").lower()
        base_stem = f"{p.stem}_{ext_tag}" if stem_counts[p.stem.lower()] > 1 and ext_tag else p.stem
        cand = out_dir / f"{base_stem}.png"
        idx = 2
        while cand.name.lower() in used_lower:
            cand = out_dir / f"{base_stem}_{idx}.png"
            idx += 1
        used_lower.add(cand.name.lower())
        resolved.append(cand)
    return resolved


def select_path_strip_indices(
    total_frames: int,
    max_panels: int = 6,
    pinned_indices: set[int] | None = None,
) -> list[int]:
    """Select frame indices for a NEB/trajectory strip, always preserving first and final frames."""
    n = int(total_frames)
    limit = max(2, int(max_panels))
    if n <= 0:
        return []
    if n <= limit:
        return list(range(n))
    chosen: set[int] = {0, n - 1}
    if pinned_indices:
        for idx in sorted(pinned_indices):
            if 0 < idx < n - 1 and len(chosen) < limit:
                chosen.add(idx)
    if len(chosen) < limit:
        for cand in [int(round(x)) for x in np.linspace(0, n - 1, limit)]:
            if len(chosen) >= limit:
                break
            chosen.add(cand)
    if len(chosen) < limit:
        for cand in range(n):
            if len(chosen) >= limit:
                break
            chosen.add(cand)
    return sorted(chosen)[:limit]


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


def copy_pil_image_to_clipboard(img: Any) -> bool:
    """Copy a PIL Image to the Windows clipboard as CF_DIB. Returns True on success."""
    if not _HAS_PIL or img is None:
        return False
    if not (sys.platform.startswith("win") or os.name == "nt"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        buf = io.BytesIO()
        img.convert("RGB").save(buf, "BMP")
        dib = buf.getvalue()[14:]  # Strip 14-byte BITMAPFILEHEADER for CF_DIB
        if not dib:
            return False

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.restype = wintypes.HGLOBAL

        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL

        gmem_moveable = 0x0002
        cf_dib = 8
        h_mem = kernel32.GlobalAlloc(gmem_moveable, len(dib))
        if not h_mem:
            return False
        ptr = kernel32.GlobalLock(h_mem)
        if not ptr:
            kernel32.GlobalFree(h_mem)
            return False
        ctypes.memmove(ptr, dib, len(dib))
        kernel32.GlobalUnlock(h_mem)

        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(h_mem)
            return False
        try:
            user32.EmptyClipboard()
            if not user32.SetClipboardData(cf_dib, h_mem):
                kernel32.GlobalFree(h_mem)
                return False
            # Clipboard owns h_mem after SetClipboardData succeeds
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


class BlenderPreviewWorker:
    """Coalescing persistent Blender subprocess for fast low-res camera previews."""

    def __init__(self, on_done: Any, on_error: Any, on_status: Any = None) -> None:
        self._on_done = on_done
        self._on_error = on_error
        self._on_status = on_status
        self._cond = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._stopped = False
        self._proc: subprocess.Popen[str] | None = None
        self._active_blender: str | None = None
        self._thread: threading.Thread | None = None

    def request(self, payload: dict[str, Any]) -> None:
        """Queue the latest preview request, superseding any unstarted prior request."""
        with self._cond:
            if self._stopped:
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            self._pending = dict(payload)
            self._cond.notify_all()

    @staticmethod
    def _cleanup_popen(proc: subprocess.Popen[str]) -> None:
        """Terminate ``proc`` if still alive, reap it, and close all stdio pipes cleanly."""
        if getattr(proc, "stdin", None) is not None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1.5)
        except Exception:
            pass
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    def _close_proc(self) -> None:
        proc = self._proc
        self._proc = None
        self._active_blender = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
                proc.wait(timeout=1.5)
        except Exception:
            pass
        finally:
            self._cleanup_popen(proc)

    def _ensure_proc(self, blender_exe: str) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None and self._active_blender == blender_exe:
            return self._proc
        self._close_proc()
        pkg_dir = package_dir()
        scene_script = pkg_dir / "scene_builder.py"
        env = os.environ.copy()
        pkg_parent = str(pkg_dir.parent)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_parent}{os.pathsep}{existing_pp}" if existing_pp else pkg_parent
        kwargs: dict[str, Any] = {}
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
        proc = subprocess.Popen(
            [
                blender_exe,
                "--background",
                "--factory-startup",
                "--python",
                str(scene_script),
                "--",
                "--preview-server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            **kwargs,
        )
        assert proc.stdout is not None
        ready = False
        for line in proc.stdout:
            if line.strip() == "PREVIEW_READY":
                ready = True
                break
        if not ready:
            self._cleanup_popen(proc)
            raise RuntimeError("Blender preview server exited before becoming ready.")
        self._proc = proc
        self._active_blender = blender_exe
        return proc

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and self._pending is None:
                    self._cond.wait()
                if self._stopped:
                    break
                cmd = self._pending
                self._pending = None
            assert cmd is not None
            seq = int(cmd["seq"])
            try:
                if self._on_status is not None:
                    self._on_status(seq, "Rendering Blender preview...")
                proc = self._ensure_proc(str(cmd["blender_exe"]))
                assert proc.stdin is not None and proc.stdout is not None
                wire_cmd = {k: v for k, v in cmd.items() if k != "blender_exe"}
                proc.stdin.write(json.dumps(wire_cmd) + "\n")
                proc.stdin.flush()

                finished = False
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if line.startswith("PREVIEW_DONE "):
                        parts = line.split(" ", 2)
                        done_seq = int(parts[1])
                        out_path = Path(parts[2])
                        self._on_done(done_seq, out_path)
                        finished = True
                        break
                    if line.startswith("PREVIEW_ERR "):
                        parts = line.split(" ", 2)
                        err_seq = int(parts[1])
                        msg = parts[2] if len(parts) > 2 else "Unknown error"
                        self._on_error(err_seq, msg)
                        finished = True
                        break
                if not finished:
                    self._close_proc()
                    self._on_error(seq, "Blender preview server terminated unexpectedly.")
            except Exception as exc:
                self._close_proc()
                self._on_error(seq, str(exc))
        self._close_proc()

    def stop(self) -> None:
        """Terminate the background Blender preview subprocess cleanly."""
        with self._cond:
            self._stopped = True
            self._pending = None
            self._cond.notify_all()
        self._close_proc()


def inspect_structure_summary(
    path: str,
    type_map: str = "",
    replicate: tuple[int, int, int] = (1, 1, 1),
) -> str:
    """Return a compact one-line structure summary via ASE for the GUI inspector banner."""
    p = Path(path).expanduser()
    if not p.is_file():
        return "No structure loaded"
    if p.suffix.lower() == ".json":
        import json

        loaded = json.loads(p.read_text(encoding="utf-8"))
        data = loaded[0] if isinstance(loaded, list) and loaded else (loaded if isinstance(loaded, dict) else {})
        syms = data.get("symbols", [])
        elems = ", ".join(sorted(set(syms)))
        eff_rep = replicate if data.get("has_cell", False) else (1, 1, 1)
        n_atoms = len(syms) * max(1, eff_rep[0]) * max(1, eff_rep[1]) * max(1, eff_rep[2])
        if data.get("has_cell", False):
            vecs = data.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            la, lb, lc, alpha, beta, gamma = compute_cell_parameters(vecs)
            return (
                f"{p.name} | {n_atoms} atoms ({elems}) | "
                f"cell {la*eff_rep[0]:.2f}×{lb*eff_rep[1]:.2f}×{lc*eff_rep[2]:.2f} Å "
                f"({alpha:.1f}°, {beta:.1f}°, {gamma:.1f}°)"
            )
        return f"{p.name} | {n_atoms} atoms ({elems}) | no periodic cell"

    from atomic_blender.convert import read_structure

    atoms = read_structure(p, type_map=type_map.strip() or None)
    if replicate != (1, 1, 1) and atoms.cell.rank == 3:
        atoms = atoms.repeat(replicate)
    elems = ", ".join(sorted(set(atoms.get_chemical_symbols())))
    formula = atoms.get_chemical_formula()
    if atoms.cell.rank == 3:
        la, lb, lc, alpha, beta, gamma = atoms.cell.cellpar()
        pbc_str = "".join("T" if f else "F" for f in atoms.pbc)
        return (
            f"{formula} | {len(atoms)} atoms ({elems}) | "
            f"{la:.2f}×{lb:.2f}×{lc:.2f} Å ({alpha:.1f}°, {beta:.1f}°, {gamma:.1f}°) | PBC={pbc_str}"
        )
    return f"{formula} | {len(atoms)} atoms ({elems}) | no periodic cell"


def _open_folder(folder: Path) -> None:
    """Open ``folder`` in the OS file explorer."""
    if sys.platform.startswith("win") or os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


def _enable_hidpi() -> None:
    """Enable native DPI awareness and AppUserModelID on Windows so fonts, canvas, and taskbar icon are crisp."""
    if sys.platform.startswith("win") or os.name == "nt":
        try:
            import ctypes

            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AtomicBlender.Studio.Desktop.1")
            except Exception:
                pass
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class AtomicBlenderApp(tk.Tk):
    """Desktop GUI client for atomic-blender."""

    def __init__(self, initial_input: str | None = None, prompt_missing_blender: bool = False) -> None:
        _enable_hidpi()
        super().__init__()
        self.title("Atomic Blender Studio — Desktop Client")
        sw = max(800, int(self.winfo_screenwidth()))
        sh = max(600, int(self.winfo_screenheight()))
        min_w = min(1360, max(1160, sw - 80))
        min_h = min(900, max(780, sh - 90))
        init_w = min(1520, max(min_w, int(sw * 0.84)))
        init_h = min(980, max(min_h, int(sh * 0.88)))
        pos_x = max(0, (sw - init_w) // 2)
        pos_y = max(8, (sh - init_h) // 2 - 12)
        self.geometry(f"{init_w}x{init_h}+{pos_x}+{pos_y}")
        self.minsize(min_w, min_h)
        try:
            self.iconbitmap(default=str(ensure_app_icon()))
        except Exception:
            pass
        try:
            logo_png = package_dir() / "assets" / "logo.png"
            if _HAS_PIL and logo_png.is_file():
                self._app_icon_photo = ImageTk.PhotoImage(Image.open(logo_png).convert("RGBA"))
                self.iconphoto(True, self._app_icon_photo)
        except Exception:
            pass

        self._configure_theme()

        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self._task_start = 0.0
        self._inspect_after_id: str | None = None
        self._inspect_seq: int = 0
        self._hq_after_id: str | None = None
        self._preview_path: Path | None = None
        self._pil_image: Any = None
        self._photo_ref: Any = None
        self._vp_photo_ref: Any = None
        self._structure_cache: dict[str, Any] | None = None
        self._base_structure_cache: dict[str, Any] | None = None
        self._ghost_structure_cache: dict[str, Any] | None = None
        self._ghost_label_name: str = ""
        self._ghost_vp_scene: ViewportScene | None = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._trajectory_index: int = 0
        self._strip_pinned_frames: set[int] = set()
        self._contact_cache_key: Any = None
        self._contact_cache_pairs: list[tuple[int, int]] = []
        self._structure_version = 0
        self._written_struct_version = -1
        self._vp_scene: ViewportScene | None = None
        self._use_interactive_cam = False
        self._dragging = False
        self._drag_mode: str | None = None
        self._drag_last: tuple[int, int] = (0, 0)
        self._axes_pos: tuple[float, float] = (0.12, 0.86)
        self._show_atoms = True
        self._selected_atoms: set[int] = set()
        self._selection_order: list[int] = []
        self._custom_element_colors: dict[str, str] = {}
        self._recent_files: list[str] = self._load_recent_files()
        self._display_checks: dict[str, tuple[tk.Canvas, tk.BooleanVar]] = {}
        self._temp_view_files: list[Path] = []
        self._preview_tmp_dir = tempfile.TemporaryDirectory(prefix="atomic_blender_prev_")
        self._preview_seq = 0
        self._destroyed = False
        self._poll_after_id: str | None = None
        self._welcome_pil_frames: list[Any] = []
        self._welcome_photo_refs: list[Any] = []
        self._welcome_frame_idx: int = 0
        self._welcome_anim_step: int = 0
        self._welcome_hover_paused: bool = False
        self._welcome_after_id: str | None = None
        self._load_welcome_animation()
        self._preview_worker = BlenderPreviewWorker(
            on_done=lambda seq, p: self._queue.put(("preview_done", (seq, p))),
            on_error=lambda seq, msg: self._queue.put(("preview_err", (seq, msg))),
        )

        self._init_variables(initial_input)
        self._prompt_missing_blender = bool(prompt_missing_blender)
        self._blender_prompted = False
        self._build_layout()
        self._wndproc_ref: Any = None
        self._orig_wndproc: Any = None
        self.update_idletasks()
        self._enable_dark_titlebar()
        self._enable_file_drop()
        self._bind_keyboard_shortcuts()
        self._poll_after_id = self.after(100, self._poll_queue)
        if initial_input:
            self._schedule_inspect()
        else:
            self.after(30, self._redraw_canvas)
        if self._prompt_missing_blender and not self.var_blender.get().strip():
            self.after(250, self._check_blender_on_startup)

    def _has_valid_blender(self) -> bool:
        """Return True if a configured or auto-detected Blender executable exists on disk."""
        raw = self.var_blender.get().strip()
        if raw:
            try:
                resolved = find_blender_executable(raw)
                self.var_blender.set(str(resolved))
                return True
            except FileNotFoundError:
                return False
        try:
            detected = find_blender_executable()
            self.var_blender.set(str(detected))
            return True
        except FileNotFoundError:
            return False

    def prompt_blender_install(self, force: bool = False) -> bool:
        """Ask whether to open the official Blender download page when Blender is missing."""
        if not force and self._blender_prompted:
            return False
        self._blender_prompted = True
        open_page = messagebox.askyesno(
            "Blender Not Found",
            "Blender (4.x / 5.x) was not detected on this system.\n\n"
            "You can still use the interactive 3D viewport, measurements, slicing, and trajectory preview, "
            "but high-resolution Cycles rendering and .blend export require Blender.\n\n"
            f"Open the official Blender download page now?\n({BLENDER_DOWNLOAD_URL})",
            parent=self,
        )
        if open_page:
            webbrowser.open(BLENDER_DOWNLOAD_URL)
            self.var_status.set(f"Opened Blender download page: {BLENDER_DOWNLOAD_URL}")
            return True
        self.var_status.set("Blender not detected (3D viewport active; set path in Settings -> Blender)")
        return False

    def _check_blender_on_startup(self) -> None:
        if getattr(self, "_destroyed", False):
            return
        if not self._has_valid_blender():
            self.prompt_blender_install(force=False)

    def _ensure_blender_available(self) -> bool:
        if self._has_valid_blender():
            return True
        self.prompt_blender_install(force=True)
        return False

    def _enable_dark_titlebar(self) -> None:
        """Enable dark window chrome on Windows 10/11."""
        if sys.platform.startswith("win") or os.name == "nt":
            try:
                import ctypes

                hwnd = ctypes.windll.user32.GetParent(self.winfo_id()) or self.winfo_id()
                value = ctypes.c_int(1)
                for attr in (20, 19):
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
                    )
            except Exception:
                pass

    def _configure_theme(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        else:
            for theme in ("alt", "default"):
                if theme in style.theme_names():
                    style.theme_use(theme)
                    break

        bg = "#0F1115"
        panel = "#171A21"
        recessed = "#111318"
        ctrl = "#222631"
        ctrl_hover = "#2D3342"
        border = "#262B36"
        fg = "#E2E8F0"
        muted = "#8B95A7"
        accent = "#0D9488"
        accent_hover = "#14B8A6"

        self.configure(bg=bg)
        self.option_add("*TCombobox*Listbox.background", panel)
        self.option_add("*TCombobox*Listbox.foreground", fg)
        self.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

        style.configure(".", background=panel, foreground=fg, bordercolor=border, troughcolor=recessed)
        style.configure("Outer.TFrame", background=bg)
        style.configure("TFrame", background=panel)
        style.configure("TPanedwindow", background=bg)
        style.configure("TLabelframe", background=panel, bordercolor=panel, relief="flat", borderwidth=0)
        style.configure("TLabelframe.Label", background=panel, foreground=muted, font=("Segoe UI", 8, "bold"))
        style.configure("TLabel", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.configure("Banner.TLabel", background=bg, foreground=fg, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=panel, foreground=muted, font=("Segoe UI", 8))

        style.configure(
            "TButton",
            background=ctrl,
            foreground=fg,
            bordercolor=ctrl,
            lightcolor=ctrl,
            darkcolor=ctrl,
            padding=(8, 4),
            font=("Segoe UI", 9),
        )
        style.map(
            "TButton",
            background=[("active", ctrl_hover), ("pressed", "#1B1F29"), ("disabled", "#151820")],
            foreground=[("disabled", "#475569")],
        )

        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#FFFFFF",
            bordercolor=accent,
            lightcolor=accent,
            darkcolor=accent,
            padding=(10, 7),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_hover), ("pressed", "#0F766E"), ("disabled", "#1F2937")],
            foreground=[("disabled", "#64748B")],
        )

        style.configure(
            "Dots.TButton",
            background=panel,
            foreground=muted,
            bordercolor=panel,
            padding=(4, 1),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Dots.TButton", background=[("active", ctrl)], foreground=[("active", fg)])

        style.configure(
            "TEntry",
            fieldbackground=recessed,
            foreground=fg,
            insertcolor=fg,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            padding=(4, 3),
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", "#181B21")],
            foreground=[("disabled", "#64748B")],
        )

        style.configure(
            "TCombobox",
            fieldbackground=recessed,
            background=ctrl,
            foreground=fg,
            arrowcolor=fg,
            bordercolor=border,
            padding=(4, 3),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", recessed), ("disabled", "#181B21")],
            foreground=[("readonly", fg), ("disabled", "#64748B")],
            background=[("active", ctrl_hover)],
        )

        style.configure(
            "TSpinbox",
            fieldbackground=recessed,
            background=ctrl,
            foreground=fg,
            arrowcolor=fg,
            bordercolor=border,
            padding=(4, 2),
        )

        style.configure("TCheckbutton", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", panel)], indicatorcolor=[("selected", accent)])
        style.configure("TRadiobutton", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", panel)], indicatorcolor=[("selected", accent)])

        style.configure("Horizontal.TScale", background=panel, troughcolor=recessed, bordercolor=border)
        style.configure("Vertical.TScrollbar", background=ctrl, troughcolor=recessed, arrowcolor=muted, bordercolor=border)

    def _init_variables(self, initial_input: str | None) -> None:
        detected_blender = ""
        try:
            detected_blender = str(find_blender_executable())
        except FileNotFoundError:
            pass

        styles = sorted(_collect_style_files().keys()) or ["default"]
        self._available_styles = styles

        self.var_inspector = tk.StringVar(value="")
        self.var_input = tk.StringVar(value=format_input_display(initial_input) if initial_input else "")
        self.var_output = tk.StringVar(value="")
        if initial_input:
            self.var_output.set(str(Path(extract_input_path(initial_input)).with_suffix(".png")))

        self.var_type_map = tk.StringVar(value="")
        self.var_style = tk.StringVar(value="studio" if "studio" in styles else styles[0])
        self.var_view_mode = tk.StringVar(value="preset")
        self.var_view_preset = tk.StringVar(value="iso")
        self.var_view_file = tk.StringVar(value="")
        self.var_fit_view = tk.BooleanVar(value=False)

        self.var_rep_x = tk.IntVar(value=1)
        self.var_rep_y = tk.IntVar(value=1)
        self.var_rep_z = tk.IntVar(value=1)
        self.var_wrap = tk.BooleanVar(value=True)
        self.var_crop_z_mode = tk.StringVar(value="none")
        self.var_crop_z_custom = tk.StringVar(value="2.0,4.0")
        self.var_bonds = tk.StringVar(value="style")
        self.var_bond_color = tk.StringVar(value="style")
        self.var_radius_scale = tk.DoubleVar(value=1.0)
        self.var_radius_label = tk.StringVar(value="1.00×")
        self.var_show_cell = tk.BooleanVar(value=True)
        self.var_show_atoms = tk.BooleanVar(value=True)
        self.var_show_output_mask = tk.BooleanVar(value=False)
        self.var_disp_ball_stick = tk.BooleanVar(value=True)
        self.var_disp_licorice = tk.BooleanVar(value=False)
        self.var_disp_vdw = tk.BooleanVar(value=False)
        self.var_ghost_alpha = tk.DoubleVar(value=0.24)
        self.var_compare_mode = tk.StringVar(value="overlay")
        self.var_frame_pinned = tk.BooleanVar(value=False)
        self.var_show_labels = tk.BooleanVar(value=False)
        self.var_show_measure_overlay = tk.BooleanVar(value=True)
        self.var_show_axes = tk.BooleanVar(value=True)
        self.var_axes_labels = tk.BooleanVar(value=True)
        self.var_axes_fixed = tk.BooleanVar(value=True)
        self.var_show_scale_bar = tk.BooleanVar(value=False)
        self.var_show_hbonds = tk.BooleanVar(value=False)
        self.var_show_polyhedra = tk.BooleanVar(value=False)
        self.var_show_cell_vectors = tk.BooleanVar(value=False)
        self.var_slice_mode = tk.StringVar(value="none")
        self.var_slice_depth = tk.DoubleVar(value=0.60)
        self.var_slice_ghost = tk.BooleanVar(value=True)
        self.var_frame_label = tk.StringVar(value="Frame 1 / 1")
        self.var_frame_pos = tk.DoubleVar(value=1.0)
        self.var_hbond_mode = tk.StringVar(value="hbond")
        self.var_hbond_min = tk.DoubleVar(value=1.55)
        self.var_hbond_max = tk.DoubleVar(value=2.55)
        self.var_material_finish = tk.StringVar(value="studio")
        self.var_color_map = tk.StringVar(value="none")
        self.var_dark_backdrop = tk.BooleanVar(value=True)

        self.var_res_w = tk.IntVar(value=1400)
        self.var_res_h = tk.IntVar(value=1400)
        self.var_samples = tk.IntVar(value=64)
        self.var_blender = tk.StringVar(value=detected_blender)
        self.var_canvas_mode = tk.StringVar(value="3d")
        self.var_rot_step = tk.DoubleVar(value=45.0)
        self.var_pan_step = tk.DoubleVar(value=1.0)
        self.var_sel_opacity = tk.DoubleVar(value=0.30)
        self.var_sel_opacity_label = tk.StringVar(value="30%")
        self.var_alpha_invert = tk.BooleanVar(value=False)
        self.var_neighbor_cutoff = tk.DoubleVar(value=3.0)
        self.var_unsel_scale_mode = tk.StringVar(value="same")
        self.var_measurement = tk.StringVar(value="Double-click 1–4 atoms for coords / distance / angle / dihedral")
        self.var_z_min = tk.StringVar(value="")
        self.var_z_max = tk.StringVar(value="")
        self._active_tab = "view"
        self.var_status = tk.StringVar(value="Ready")
        self._formatting_input = False
        self.var_input.trace_add("write", lambda *_: self._on_input_var_changed())

    def _build_layout(self) -> None:
        self.f_settings = ttk.Frame(self, padding=(10, 4, 10, 6), style="Outer.TFrame")
        self._settings_visible = False
        self._build_settings_drawer(self.f_settings)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 4))
        self._paned = paned

        left_host = ttk.Frame(paned, padding=0, style="Outer.TFrame")
        left_body = ttk.Frame(left_host, padding=0, style="Outer.TFrame")
        left_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_canvas = tk.Canvas(left_body, bg="#0F1115", highlightthickness=0, width=436)
        left_scroll = ttk.Scrollbar(left_body, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas, padding=6, style="Outer.TFrame")
        left_win = left_canvas.create_window((0, 0), window=left, anchor=tk.NW)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _update_left_scroll(_e: Any = None) -> None:
            bbox = left_canvas.bbox("all")
            left_canvas.configure(scrollregion=bbox)
            need_scroll = bool(bbox and (bbox[3] - bbox[1]) > max(1, left_canvas.winfo_height()))
            if need_scroll and not left_scroll.winfo_manager():
                left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            elif not need_scroll and left_scroll.winfo_manager():
                left_scroll.pack_forget()

        left.bind("<Configure>", _update_left_scroll)
        left_canvas.bind(
            "<Configure>",
            lambda e: (left_canvas.itemconfigure(left_win, width=e.width), _update_left_scroll()),
        )
        left_canvas.bind(
            "<MouseWheel>",
            lambda e: left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            if left_scroll.winfo_manager() and (e.widget is left_canvas or str(e.widget).startswith(str(left)))
            else None,
        )

        right = ttk.Frame(paned, padding=6, style="Outer.TFrame")
        paned.add(left_host, weight=0)
        paned.add(right, weight=1)

        self._build_controls(left, action_parent=left_host)
        self._build_preview_and_log(right)

        status_bar = ttk.Frame(self, padding=(10, 2, 10, 6), style="Outer.TFrame")
        status_bar.pack(fill=tk.X)
        ttk.Label(status_bar, textvariable=self.var_status, style="Banner.TLabel", foreground="#94A3B8").pack(
            side=tk.LEFT
        )

    def _toggle_settings_panel(self) -> None:
        if not hasattr(self, "f_settings"):
            return
        if self._settings_visible:
            self.f_settings.pack_forget()
            self._settings_visible = False
        else:
            self.f_settings.pack(fill=tk.X, before=self._paned)
            self._settings_visible = True

    def _build_settings_drawer(self, parent: ttk.Frame) -> None:
        """Build collapsible top drawer for rarely changed output, Blender, size, and style/camera files."""
        card = ttk.LabelFrame(parent, text="SETTINGS", padding=(10, 8))
        card.pack(fill=tk.X)
        card.columnconfigure(0, weight=1, uniform="set_col")
        card.columnconfigure(1, weight=1, uniform="set_col")

        left_col = ttk.Frame(card)
        left_col.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        left_col.columnconfigure(0, minsize=62)
        left_col.columnconfigure(1, weight=1)

        ttk.Label(left_col, text="Output").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_col, textvariable=self.var_output).grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(left_col, text="Save", width=6, command=self._browse_output).grid(row=0, column=2, pady=2)

        ttk.Label(left_col, text="Blender").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(left_col, textvariable=self.var_blender).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(left_col, text="Find", width=6, command=self._browse_blender).grid(row=1, column=2, pady=2)

        ttk.Label(left_col, text="Style").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.combo_style = ttk.Combobox(left_col, textvariable=self.var_style, values=self._available_styles)
        self.combo_style.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        self.combo_style.bind("<<ComboboxSelected>>", lambda _e: self._on_repr_changed())
        ttk.Button(left_col, text="JSON", width=6, command=self._browse_style).grid(row=2, column=2, pady=2)

        right_col = ttk.Frame(card)
        right_col.grid(row=0, column=1, sticky=tk.NSEW, padx=(10, 0))
        right_col.columnconfigure(0, minsize=62)
        right_col.columnconfigure(1, weight=1)

        ttk.Label(right_col, text="Render").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_sz = ttk.Frame(right_col)
        row_sz.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        DarkSpinbox(row_sz, from_=64, to=8192, increment=100, variable=self.var_res_w, width=5).pack(side=tk.LEFT)
        ttk.Label(row_sz, text="×").pack(side=tk.LEFT, padx=4)
        DarkSpinbox(row_sz, from_=64, to=8192, increment=100, variable=self.var_res_h, width=5).pack(side=tk.LEFT)
        ttk.Label(row_sz, text="Samples").pack(side=tk.LEFT, padx=(10, 4))
        DarkSpinbox(row_sz, from_=1, to=2048, increment=8, variable=self.var_samples, width=4).pack(side=tk.LEFT)
        for lbl_r, rw, rh in (("700²", 700, 700), ("1400²", 1400, 1400), ("4:3", 1400, 1200)):
            ttk.Button(
                row_sz, text=lbl_r, width=5, command=lambda ww=rw, hh=rh: self._set_resolution(ww, hh)
            ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(right_col, text="ViewSrc").grid(row=1, column=0, sticky=tk.W, pady=2)
        row_vm = ttk.Frame(right_col)
        row_vm.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        DarkRadiobutton(
            row_vm, text="Preset", variable=self.var_view_mode, value="preset", command=self._sync_view_mode
        ).pack(side=tk.LEFT)
        self.combo_preset = ttk.Combobox(
            row_vm, textvariable=self.var_view_preset, values=list(VIEW_PRESETS), state="readonly", width=11
        )
        self.combo_preset.pack(side=tk.LEFT, padx=(6, 10))
        self.combo_preset.bind("<<ComboboxSelected>>", lambda _e: self._pick_view_preset(self.var_view_preset.get()))
        DarkRadiobutton(
            row_vm, text="view.json", variable=self.var_view_mode, value="file", command=self._sync_view_mode
        ).pack(side=tk.LEFT)
        self.chk_fit_view = DarkCheckbutton(row_vm, text="--fit-view", variable=self.var_fit_view)
        self.chk_fit_view.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(right_col, text="view.json").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.entry_view_file = ttk.Entry(right_col, textvariable=self.var_view_file)
        self.entry_view_file.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        self.btn_view_file = ttk.Button(right_col, text="Open", width=6, command=self._browse_view_file)
        self.btn_view_file.grid(row=2, column=2, pady=2)

        row_rec = ttk.Frame(card)
        row_rec.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        ttk.Label(row_rec, text="Figure Recipe:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row_rec, text="Save Recipe...", command=self._on_save_recipe).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row_rec, text="Load Recipe...", command=self._on_load_recipe).pack(side=tk.LEFT)
        ttk.Button(row_rec, text="Desktop Shortcut", command=self._on_create_desktop_shortcut).pack(side=tk.RIGHT)
        self._sync_view_mode()

    def _build_controls(self, parent: ttk.Frame, action_parent: ttk.Frame | None = None) -> None:
        # 1. Slim 1-line Input & auto-hidden Types (pinned above tabs)
        f_io = ttk.LabelFrame(parent, text="FILE", padding=(10, 8))
        f_io.pack(fill=tk.X, pady=(0, 8))
        f_io.columnconfigure(0, minsize=48)
        f_io.columnconfigure(1, weight=1)

        ttk.Label(f_io, text="Input").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.entry_in = ttk.Entry(f_io, textvariable=self.var_input, width=30)
        self.entry_in.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        self.entry_in.bind("<FocusOut>", lambda _e: (self._format_input_entry_display(), self._schedule_inspect()))
        row_in_btns = ttk.Frame(f_io)
        row_in_btns.grid(row=0, column=2, sticky=tk.EW, pady=2)
        ttk.Button(row_in_btns, text="Open", width=6, command=self._browse_input).pack(side=tk.LEFT)
        self.btn_recent = tk.Menubutton(
            row_in_btns,
            text="▾",
            bg="#222631",
            fg="#E2E8F0",
            activebackground="#2D3342",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=6,
            pady=3,
            cursor="hand2",
            font=("Segoe UI", 9),
        )
        self.recent_menu = tk.Menu(
            self.btn_recent,
            tearoff=0,
            bg="#171A21",
            fg="#E2E8F0",
            activebackground="#0D9488",
            activeforeground="#FFFFFF",
            disabledforeground="#475569",
        )
        self.btn_recent.configure(menu=self.recent_menu)
        self.btn_recent.pack(side=tk.LEFT, padx=(3, 0))
        self._rebuild_recent_menu()

        self.lbl_type_map = ttk.Label(f_io, text="Types")
        self.entry_type_map = ttk.Entry(f_io, textvariable=self.var_type_map)
        self.entry_type_map.bind("<FocusOut>", lambda _e: self._schedule_inspect())
        self.btn_inspect_tm = ttk.Button(f_io, text="Apply", width=6, command=self._schedule_inspect)
        self.lbl_type_map.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.entry_type_map.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        self.btn_inspect_tm.grid(row=1, column=2, pady=2)
        self._sync_type_map_visibility()

        # 2. Modern Segmented Pill Tab Switcher (View / Select / Scene)
        tab_bar = tk.Frame(parent, bg="#111318", padx=3, pady=3)
        tab_bar.pack(fill=tk.X, pady=(0, 8))
        tab_bar.columnconfigure(0, weight=1, uniform="main_tabs")
        tab_bar.columnconfigure(1, weight=1, uniform="main_tabs")
        tab_bar.columnconfigure(2, weight=1, uniform="main_tabs")

        self._tab_buttons: dict[str, tuple[tk.Button, tk.Frame]] = {}
        for col_idx, (tab_key, tab_label) in enumerate(
            (("view", "View"), ("selection", "Select"), ("tool", "Scene"))
        ):
            holder = tk.Frame(tab_bar, bg="#111318")
            holder.grid(row=0, column=col_idx, sticky=tk.EW, padx=(0, 3) if col_idx < 2 else (0, 0))
            btn = tk.Button(
                holder,
                text=tab_label,
                command=lambda k=tab_key: self._switch_tab(k),
                bd=0,
                relief=tk.FLAT,
                pady=5,
                cursor="hand2",
                font=("Segoe UI", 9, "bold"),
            )
            btn.pack(fill=tk.X)
            bar = tk.Frame(holder, height=2)
            bar.pack(fill=tk.X)
            self._tab_buttons[tab_key] = (btn, bar)

        self.tab_container = ttk.Frame(parent, style="Outer.TFrame")
        self.tab_container.pack(fill=tk.BOTH, expand=True)
        self.tab_view = ttk.Frame(self.tab_container, style="Outer.TFrame")
        self.tab_tool = ttk.Frame(self.tab_container, style="Outer.TFrame")
        self.tab_selection = ttk.Frame(self.tab_container, style="Outer.TFrame")

        # --- View Tab: Representation & Unified Camera ---
        self._build_display_types_panel(self.tab_view)

        f_cam = ttk.LabelFrame(self.tab_view, text="CAMERA", padding=(10, 8))
        f_cam.pack(fill=tk.X, pady=(0, 8))
        f_cam.columnconfigure(0, minsize=68)
        f_cam.columnconfigure(2, weight=1)

        ttk.Label(f_cam, text="Orient").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_ori = ttk.Frame(f_cam)
        row_ori.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=2)
        for c_i, (o_lbl, o_key) in enumerate(
            (("Iso", "iso"), ("Slab", "side-oblique"), ("a", "a"), ("b", "b"), ("c", "c"), ("c*", "c*"))
        ):
            row_ori.columnconfigure(c_i, weight=1, uniform="ori_btn")
            ttk.Button(row_ori, text=o_lbl, command=lambda p=o_key: self._pick_view_preset(p)).grid(
                row=0, column=c_i, sticky=tk.EW, padx=(0, 3 if c_i < 5 else 0)
            )

        ttk.Label(f_cam, text="Rotate (°)").grid(row=1, column=0, sticky=tk.W, pady=2)
        ent_rot = ttk.Entry(f_cam, textvariable=self.var_rot_step, width=5, justify=tk.CENTER)
        ent_rot.grid(row=1, column=1, sticky=tk.W, padx=(4, 6), pady=2)
        row_rot = ttk.Frame(f_cam)
        row_rot.grid(row=1, column=2, sticky=tk.EW, pady=2)
        for c_idx in range(6):
            row_rot.columnconfigure(c_idx, weight=1, uniform="cam_btn")
        for c_idx, (lbl, y_s, p_s, r_s) in enumerate((
            ("◀", -1.0, 0.0, 0.0),
            ("▶", 1.0, 0.0, 0.0),
            ("▲", 0.0, -1.0, 0.0),
            ("▼", 0.0, 1.0, 0.0),
            ("↺", 0.0, 0.0, -1.0),
            ("↻", 0.0, 0.0, 1.0),
        )):
            ttk.Button(
                row_rot,
                text=lbl,
                width=3,
                command=lambda ys=y_s, ps=p_s, rs=r_s: self._on_step_rotate(ys, ps, rs),
            ).grid(row=0, column=c_idx, sticky=tk.EW, padx=(0, 2 if c_idx < 5 else 0))

        ttk.Label(f_cam, text="Move (Å)").grid(row=2, column=0, sticky=tk.W, pady=2)
        ent_pan = ttk.Entry(f_cam, textvariable=self.var_pan_step, width=5, justify=tk.CENTER)
        ent_pan.grid(row=2, column=1, sticky=tk.W, padx=(4, 6), pady=2)
        row_pan = ttk.Frame(f_cam)
        row_pan.grid(row=2, column=2, sticky=tk.EW, pady=2)
        for c_idx in range(6):
            row_pan.columnconfigure(c_idx, weight=1, uniform="cam_btn")
        for c_idx, (lbl, dx_s, dy_s) in enumerate((
            ("◀", -1.0, 0.0),
            ("▶", 1.0, 0.0),
            ("▲", 0.0, 1.0),
            ("▼", 0.0, -1.0),
        )):
            ttk.Button(
                row_pan,
                text=lbl,
                width=3,
                command=lambda xs=dx_s, ys=dy_s: self._on_step_move(xs, ys),
            ).grid(row=0, column=c_idx, sticky=tk.EW, padx=(0, 2))
        ttk.Button(row_pan, text="Reset", command=self._on_reset_view).grid(
            row=0, column=4, columnspan=2, sticky=tk.EW
        )

        # --- Scene Tab (tab_tool): Supercell, Framing & Ghost Reference Compare ---
        self.f_geom = ttk.LabelFrame(self.tab_tool, text="SUPERCELL & FRAMING", padding=(10, 8))
        self.f_geom.pack(fill=tk.X, pady=(0, 8))
        self.f_geom_body = ttk.Frame(self.f_geom)
        self.f_geom_body.pack(fill=tk.X)
        f_geom = self.f_geom_body
        f_geom.columnconfigure(0, minsize=68)
        f_geom.columnconfigure(1, weight=1)

        ttk.Label(f_geom, text="Replicate").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_rep = ttk.Frame(f_geom)
        row_rep.grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        for var in (self.var_rep_x, self.var_rep_y, self.var_rep_z):
            sb = DarkSpinbox(row_rep, from_=1, to=20, variable=var, width=3, command=self._schedule_inspect)
            sb.pack(side=tk.LEFT, padx=(0, 4))
        DarkCheckbutton(row_rep, text="Wrap", variable=self.var_wrap).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(f_geom, text="Crop Z").grid(row=1, column=0, sticky=tk.W, pady=2)
        row_cz = ttk.Frame(f_geom)
        row_cz.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        combo_cz = ttk.Combobox(
            row_cz, textvariable=self.var_crop_z_mode, values=["none", "auto", "custom"], state="readonly", width=8
        )
        combo_cz.pack(side=tk.LEFT)
        combo_cz.bind("<<ComboboxSelected>>", lambda _e: self._sync_crop_z())
        self.entry_crop_z = ttk.Entry(row_cz, textvariable=self.var_crop_z_custom, width=10)
        self.entry_crop_z.pack(side=tk.LEFT, padx=(6, 0))
        self._sync_crop_z()

        row_smart = ttk.Frame(f_geom)
        row_smart.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        row_smart.columnconfigure(0, weight=1, uniform="preset_btn")
        row_smart.columnconfigure(1, weight=1, uniform="preset_btn")
        ttk.Button(row_smart, text="Slab Framing", command=self._apply_slab_preset).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 4)
        )
        ttk.Button(row_smart, text="Bulk Framing", command=self._apply_bulk_preset).grid(
            row=0, column=1, sticky=tk.EW
        )

        f_ghost = ttk.LabelFrame(self.tab_tool, text="COMPARE (GHOST OVERLAY)", padding=(10, 8))
        f_ghost.pack(fill=tk.X, pady=(0, 8))
        f_ghost.columnconfigure(0, weight=1)
        row_g0 = ttk.Frame(f_ghost)
        row_g0.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(row_g0, text="Load Ref...", command=self._on_load_ghost_structure).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(row_g0, text="Pin Current", command=self._pin_current_as_ghost).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row_g0, text="Clear", width=6, command=self._clear_ghost_structure).pack(side=tk.LEFT)
        row_g1 = ttk.Frame(f_ghost)
        row_g1.pack(fill=tk.X)
        DarkRadiobutton(
            row_g1,
            text="Overlay",
            variable=self.var_compare_mode,
            value="overlay",
            command=lambda: self._on_ghost_options_changed(trigger_blender=True),
        ).pack(side=tk.LEFT, padx=(0, 6))
        DarkRadiobutton(
            row_g1,
            text="Side-by-Side",
            variable=self.var_compare_mode,
            value="side_by_side",
            command=lambda: self._on_ghost_options_changed(trigger_blender=True),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.slider_ghost_alpha = DarkSlider(
            row_g1,
            from_=0.08,
            to=0.85,
            variable=self.var_ghost_alpha,
            command=lambda _v: self._on_ghost_options_changed(trigger_blender=False),
            release_command=lambda: self._on_ghost_options_changed(trigger_blender=True),
        )
        self.slider_ghost_alpha.pack(side=tk.LEFT, fill=tk.X, expand=True)

        f_slice = ttk.LabelFrame(self.tab_tool, text="CROSS-SECTION SLICE", padding=(10, 8))
        f_slice.pack(fill=tk.X, pady=(0, 8))
        f_slice.columnconfigure(1, weight=1)
        ttk.Label(f_slice, text="Plane").grid(row=0, column=0, sticky=tk.W, pady=2)
        cb_sl = ttk.Combobox(
            f_slice,
            textvariable=self.var_slice_mode,
            values=["none", "camera", "x", "y", "z", "a", "b", "c"],
            state="readonly",
            width=9,
        )
        cb_sl.grid(row=0, column=1, sticky=tk.W, padx=(6, 6), pady=2)
        cb_sl.bind("<<ComboboxSelected>>", lambda _e: self._on_slice_changed(trigger_blender=True))
        DarkCheckbutton(
            f_slice,
            text="Ghost Cut",
            variable=self.var_slice_ghost,
            command=lambda: self._on_slice_changed(trigger_blender=True),
        ).grid(row=0, column=2, sticky=tk.E)

        ttk.Label(f_slice, text="Depth").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.slider_slice = DarkSlider(
            f_slice,
            from_=0.10,
            to=1.00,
            variable=self.var_slice_depth,
            command=lambda _v: self._on_slice_changed(trigger_blender=False),
            release_command=lambda: self._on_slice_changed(trigger_blender=True),
        )
        self.slider_slice.grid(row=1, column=1, sticky=tk.EW, padx=(6, 6), pady=2)
        ttk.Button(f_slice, text="Reset", width=6, command=self._reset_slice).grid(row=1, column=2, pady=2)

        f_traj = ttk.LabelFrame(self.tab_tool, text="TRAJECTORY / NEB FRAMES", padding=(10, 8))
        f_traj.pack(fill=tk.X, pady=(0, 8))
        row_tf = ttk.Frame(f_traj)
        row_tf.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(row_tf, text="◀", width=3, command=lambda: self._step_trajectory_frame(-1)).pack(side=tk.LEFT)
        ttk.Label(row_tf, textvariable=self.var_frame_label, width=12, anchor=tk.CENTER).pack(
            side=tk.LEFT, padx=(4, 4)
        )
        ttk.Button(row_tf, text="▶", width=3, command=lambda: self._step_trajectory_frame(1)).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        DarkCheckbutton(
            row_tf, text="Pin Strip", variable=self.var_frame_pinned, command=self._on_toggle_pin_frame
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row_tf, text="Render Strip", command=self._on_render_path_strip).pack(
            side=tk.RIGHT, fill=tk.X, expand=True
        )
        self.slider_frame = DarkSlider(
            f_traj,
            from_=1.0,
            to=2.0,
            variable=self.var_frame_pos,
            command=self._on_frame_slider_changed,
            release_command=self._on_frame_slider_released,
        )
        self.slider_frame.pack(fill=tk.X)
        self.energy_canvas = tk.Canvas(
            f_traj, height=56, bg="#111318", highlightthickness=1, highlightbackground="#262B36", cursor="hand2"
        )
        self.energy_canvas.pack(fill=tk.X, pady=(5, 0))
        self.energy_canvas.bind("<Button-1>", self._on_energy_canvas_click)
        self.energy_canvas.bind("<Configure>", lambda _e: self._draw_energy_profile())

        # --- Select Tab (tab_selection) ---
        self._build_selection_tab(self.tab_selection)

        self._switch_tab("view")

        # 3. Pinned Unified Output & Action Bar
        act_host = action_parent if action_parent is not None else parent
        f_act = ttk.Frame(act_host, style="Outer.TFrame")
        if action_parent is not None:
            f_act.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(4, 6))
        else:
            f_act.pack(fill=tk.X, pady=(4, 4))
        self.btn_render = ttk.Button(
            f_act, text="Render High-Res PNG", style="Accent.TButton", command=self._on_render
        )
        self.btn_render.pack(fill=tk.X, pady=(0, 4))

        row_sub = ttk.Frame(f_act, style="Outer.TFrame")
        row_sub.pack(fill=tk.X)
        for c_i in range(4):
            row_sub.columnconfigure(c_i, weight=1, uniform="sub_act")
        ttk.Button(row_sub, text="Top+Side", command=self._on_render_top_side_panel).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 3)
        )
        ttk.Button(row_sub, text="Batch...", command=self._on_batch_render_files).grid(
            row=0, column=1, sticky=tk.EW, padx=(0, 3)
        )
        self.btn_blender = ttk.Button(row_sub, text="Blender", command=self._on_open_in_blender)
        self.btn_blender.grid(row=0, column=2, sticky=tk.EW, padx=(0, 3))
        self.btn_folder = ttk.Button(row_sub, text="Folder", command=self._on_open_folder)
        self.btn_folder.grid(row=0, column=3, sticky=tk.EW)

    def _switch_tab(self, tab_name: str) -> None:
        alias = {"select": "selection", "scene": "tool"}.get(tab_name, tab_name)
        self._active_tab = alias if alias in ("view", "tool", "selection") else "view"
        tabs = {"view": self.tab_view, "tool": self.tab_tool, "selection": self.tab_selection}
        for key, frm in tabs.items():
            if key == self._active_tab:
                frm.pack(fill=tk.X)
            else:
                frm.pack_forget()
        for key, (btn, bar) in self._tab_buttons.items():
            active = key == self._active_tab
            btn.configure(
                bg="#1C2029" if active else "#111318",
                fg="#F8FAFC" if active else "#8B95A7",
                activebackground="#222631",
                activeforeground="#FFFFFF",
            )
            bar.configure(bg="#14B8A6" if active else "#111318")

    def _build_selection_tab(self, parent: ttk.Frame) -> None:
        """Build the Select tab with Focus Recipes, Filter & Pick, Appearance, and Atom List."""
        f_foc = ttk.LabelFrame(parent, text="FOCUS RECIPES", padding=(10, 8))
        f_foc.pack(fill=tk.X, pady=(0, 8))
        for c_i in range(3):
            f_foc.columnconfigure(c_i, weight=1, uniform="foc_btn")
        ttk.Button(f_foc, text="Adsorbate Spotlight", command=self._apply_adsorbate_spotlight_recipe).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 4)
        )
        ttk.Button(f_foc, text="Z-Depth Fade", command=self._apply_depth_fade_recipe).grid(
            row=0, column=1, sticky=tk.EW, padx=(0, 4)
        )
        ttk.Button(f_foc, text="Reset Focus", command=self._reset_focus_recipes).grid(row=0, column=2, sticky=tk.EW)

        f_lbl = ttk.LabelFrame(parent, text="SELECTION", padding=(10, 8))
        f_lbl.pack(fill=tk.X, pady=(0, 8))

        row_top = ttk.Frame(f_lbl)
        row_top.pack(fill=tk.X, pady=(0, 6))
        DarkCheckbutton(
            row_top,
            text="Labels",
            variable=self.var_show_labels,
            command=self._on_labels_toggled,
        ).pack(side=tk.LEFT)
        DarkCheckbutton(
            row_top,
            text="2D Callout",
            variable=self.var_show_measure_overlay,
            command=self._on_labels_toggled,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(row_top, text="Reset Colors", command=self._reset_custom_element_colors).pack(side=tk.RIGHT)

        row_btns = ttk.Frame(f_lbl)
        row_btns.pack(fill=tk.X)
        for c_idx, (text, cb) in enumerate((
            ("All", self._select_all_atoms),
            ("Clear", self._clear_selection),
            ("Invert", self._invert_selection),
            ("Elem", self._select_same_element),
        )):
            row_btns.columnconfigure(c_idx, weight=1, uniform="sel_btn")
            ttk.Button(row_btns, text=text, command=cb).grid(
                row=0, column=c_idx, sticky=tk.EW, padx=(0, 3 if c_idx < 3 else 0)
            )

        self.f_elem_chips = ttk.Frame(f_lbl)
        self.f_elem_chips.pack(fill=tk.X, pady=(6, 0))
        self._elem_chip_buttons: dict[str, tk.Button] = {}

        row_filt = ttk.Frame(f_lbl)
        row_filt.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row_filt, text="+Å").pack(side=tk.LEFT)
        ttk.Entry(row_filt, textvariable=self.var_neighbor_cutoff, width=4, justify=tk.CENTER).pack(
            side=tk.LEFT, padx=(3, 4)
        )
        ttk.Button(row_filt, text="Shell", width=6, command=self._expand_selection_neighbors).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(row_filt, text="Z").pack(side=tk.LEFT)
        ttk.Entry(row_filt, textvariable=self.var_z_min, width=6, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(row_filt, text="~").pack(side=tk.LEFT)
        ttk.Entry(row_filt, textvariable=self.var_z_max, width=6, justify=tk.CENTER).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Button(row_filt, text="Select Z", command=self._select_by_z_range).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(
            f_lbl,
            textvariable=self.var_measurement,
            foreground="#2DD4BF",
            font=("Segoe UI", 9, "bold"),
            wraplength=396,
        ).pack(fill=tk.X, pady=(6, 0))

        f_op = ttk.LabelFrame(parent, text="APPEARANCE", padding=(10, 8))
        f_op.pack(fill=tk.X, pady=(0, 8))
        f_op.columnconfigure(0, minsize=68)
        f_op.columnconfigure(1, weight=1)

        ttk.Label(f_op, text="Opacity").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.slider_opacity = DarkSlider(
            f_op,
            from_=0.05,
            to=1.0,
            variable=self.var_sel_opacity,
            command=self._on_opacity_changed,
            release_command=self._on_opacity_released,
        )
        self.slider_opacity.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=2)
        ttk.Label(f_op, textvariable=self.var_sel_opacity_label, width=5).grid(row=0, column=2, pady=2)
        ttk.Button(f_op, text="100%", width=5, command=self._reset_opacity).grid(row=0, column=3, padx=(4, 0), pady=2)

        DarkCheckbutton(
            f_op,
            text="Fade Unselected",
            variable=self.var_alpha_invert,
            command=self._on_opacity_released,
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 2))

        row_bg_sz = ttk.Frame(f_op)
        row_bg_sz.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(2, 0))
        ttk.Label(row_bg_sz, text="Substrate").pack(side=tk.LEFT, padx=(0, 8))
        for txt, val in (("Same", "same"), ("Slim 0.55×", "slim"), ("VdW 1.35×", "vdw")):
            DarkRadiobutton(
                row_bg_sz,
                text=txt,
                variable=self.var_unsel_scale_mode,
                value=val,
                command=lambda: self._sync_selection_and_alphas(trigger_blender=True),
            ).pack(side=tk.LEFT, padx=(0, 8))

        f_list = ttk.LabelFrame(parent, text="ATOMS", padding=(8, 6))
        f_list.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        list_wrap = ttk.Frame(f_list)
        list_wrap.pack(fill=tk.BOTH, expand=True)
        self.atom_listbox = tk.Listbox(
            list_wrap,
            height=14,
            bg="#111318",
            fg="#E2E8F0",
            selectbackground="#1E293B",
            selectforeground="#2DD4BF",
            activestyle="none",
            highlightbackground="#262B36",
            highlightthickness=1,
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        atom_scroll = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self.atom_listbox.yview)
        self.atom_listbox.configure(yscrollcommand=atom_scroll.set)
        self.atom_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        atom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.atom_listbox.bind("<ButtonRelease-1>", self._on_atom_list_click)

    def _build_display_types_panel(self, parent: ttk.Frame) -> None:
        """Build minimal flat Display Types card with style thumbnails, dedicated Atoms toggle, and uniform rows."""
        f_disp = ttk.LabelFrame(parent, text="DISPLAY", padding=(8, 6))
        f_disp.pack(fill=tk.X, pady=(0, 8))
        self.f_rep = f_disp

        row_rep_hdr = ttk.Frame(f_disp)
        row_rep_hdr.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row_rep_hdr, text="Style", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Button(
            row_rep_hdr, text="•••", style="Dots.TButton", width=3, command=lambda: self._toggle_section(self.f_rep_body)
        ).pack(side=tk.RIGHT)
        DarkCheckbutton(
            row_rep_hdr,
            text="◎ Show Atoms",
            variable=self.var_show_atoms,
            command=self._on_atoms_visibility_toggled,
        ).pack(side=tk.RIGHT, padx=(0, 6))

        rows = (
            ("ball_stick", "Ball and Stick", self.var_disp_ball_stick, None),
            ("licorice", "Licorice", self.var_disp_licorice, None),
            ("vdw", "Van der Waals", self.var_disp_vdw, None),
            ("crystal_lattice", "Crystal Lattice", self.var_show_cell, lambda: self._switch_tab("tool")),
            ("cell_vectors", "Lattice Vectors (a, b, c)", self.var_show_cell_vectors, None),
            ("polyhedra", "Coordination Polyhedra", self.var_show_polyhedra, None),
            ("reference_axes", "Reference Axes", self.var_show_axes, self._toggle_axes_options),
            ("scale_bar", "Scale Bar (Å)", self.var_show_scale_bar, None),
            ("hbonds", "H-Bonds / Contacts", self.var_show_hbonds, self._toggle_hbond_options),
        )
        base_row_bg = "#171A21"
        hover_row_bg = "#1E222B"
        for key, label_txt, var, dots_cb in rows:
            row = tk.Frame(f_disp, bg=base_row_bg, padx=4, pady=2)
            row.pack(fill=tk.X)

            chk_canvas = tk.Canvas(row, width=22, height=22, bg=base_row_bg, highlightthickness=0, cursor="hand2")
            chk_canvas.pack(side=tk.LEFT, padx=(0, 6))
            self._display_checks[key] = (chk_canvas, var)

            lbl = tk.Label(
                row,
                text=label_txt,
                bg=base_row_bg,
                fg="#E2E8F0",
                font=("Segoe UI", 9),
                anchor=tk.W,
                cursor="hand2",
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def _set_row_bg(c: str, r: tk.Frame = row, ch: tk.Canvas = chk_canvas, lb: tk.Label = lbl) -> None:
                r.configure(bg=c)
                ch.configure(bg=c)
                lb.configure(bg=c)

            for w_item in (row, chk_canvas, lbl):
                w_item.bind("<Enter>", lambda _e, fn=_set_row_bg: fn(hover_row_bg))
                w_item.bind("<Leave>", lambda _e, fn=_set_row_bg: fn(base_row_bg))

            chk_canvas.bind("<Button-1>", lambda _e, k=key: self._on_display_type_clicked(k))
            lbl.bind("<Button-1>", lambda _e, k=key: self._on_display_type_clicked(k))

            if dots_cb is not None:
                btn = ttk.Button(row, text="•••", style="Dots.TButton", width=3, command=dots_cb)
                btn.pack(side=tk.RIGHT)

        self._refresh_display_checks()

        # Minimal style & scale options (toggled via •••)
        self.f_rep_body = ttk.Frame(f_disp, padding=(2, 6, 2, 2))
        self.f_rep_body.pack(fill=tk.X)
        self.f_rep_body.columnconfigure(0, minsize=56)
        self.f_rep_body.columnconfigure(1, weight=1)

        ttk.Label(self.f_rep_body, text="Finish").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_fin = ttk.Frame(self.f_rep_body)
        row_fin.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        for c_i, (fin_lbl, fin_key) in enumerate((
            ("Studio", "studio"),
            ("Chalk", "chalk"),
            ("Flat", "flat"),
            ("Metal", "metallic"),
        )):
            row_fin.columnconfigure(c_i, weight=1, uniform="fin_btn")
            ttk.Button(
                row_fin, text=fin_lbl, command=lambda fk=fin_key: self._set_material_finish(fk)
            ).grid(row=0, column=c_i, sticky=tk.EW, padx=(0, 2 if c_i < 3 else 0))

        ttk.Label(self.f_rep_body, text="Scale").grid(row=1, column=0, sticky=tk.W, pady=2)
        row_sc = ttk.Frame(self.f_rep_body)
        row_sc.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        self.slider_scale = DarkSlider(
            row_sc,
            from_=0.4,
            to=1.6,
            variable=self.var_radius_scale,
            command=self._on_scale_changed,
            release_command=self._on_repr_changed,
        )
        self.slider_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row_sc, textvariable=self.var_radius_label, width=6).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(self.f_rep_body, text="Bond").grid(row=2, column=0, sticky=tk.W, pady=2)
        row_bc = ttk.Frame(self.f_rep_body)
        row_bc.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        cb_bcol = ttk.Combobox(
            row_bc,
            textvariable=self.var_bond_color,
            values=["style", "bicolor", "single"],
            state="readonly",
            width=10,
        )
        cb_bcol.pack(side=tk.LEFT)
        cb_bcol.bind("<<ComboboxSelected>>", lambda _e: self._on_repr_changed())
        ttk.Label(row_bc, text="Color").pack(side=tk.LEFT, padx=(10, 6))
        cb_cmap = ttk.Combobox(
            row_bc,
            textvariable=self.var_color_map,
            values=["none", "viridis", "coolwarm", "plasma"],
            state="readonly",
            width=11,
        )
        cb_cmap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cb_cmap.bind("<<ComboboxSelected>>", lambda _e: self._on_colormap_changed())

        self.f_axes_opts = tk.Frame(f_disp, bg="#111318", padx=8, pady=5)
        tk.Label(self.f_axes_opts, text="Axes:", bg="#111318", fg="#8B95A7", font=("Segoe UI", 8, "bold")).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        DarkCheckbutton(
            self.f_axes_opts,
            text="x, y, z Labels",
            variable=self.var_axes_labels,
            command=self._on_axes_option_changed,
            bg="#111318",
        ).pack(side=tk.LEFT, padx=(0, 8))
        DarkCheckbutton(
            self.f_axes_opts,
            text="Fix",
            variable=self.var_axes_fixed,
            command=self._on_axes_fix_toggled,
            bg="#111318",
        ).pack(side=tk.LEFT)
        self._axes_opts_visible = False

        self.f_hbond_opts = tk.Frame(f_disp, bg="#111318", padx=8, pady=5)
        tk.Label(self.f_hbond_opts, text="Mode:", bg="#111318", fg="#8B95A7", font=("Segoe UI", 8, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        cb_hb = ttk.Combobox(
            self.f_hbond_opts,
            textvariable=self.var_hbond_mode,
            values=["hbond", "interface", "range"],
            state="readonly",
            width=9,
        )
        cb_hb.pack(side=tk.LEFT, padx=(0, 4))
        cb_hb.bind("<<ComboboxSelected>>", lambda _e: self._on_hbonds_changed())
        ttk.Entry(self.f_hbond_opts, textvariable=self.var_hbond_min, width=5, justify=tk.CENTER).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        tk.Label(self.f_hbond_opts, text="~", bg="#181A20", fg="#94A3B8").pack(side=tk.LEFT)
        ttk.Entry(self.f_hbond_opts, textvariable=self.var_hbond_max, width=5, justify=tk.CENTER).pack(
            side=tk.LEFT, padx=(2, 4)
        )
        ttk.Button(self.f_hbond_opts, text="Apply", width=5, command=self._on_hbonds_changed).pack(side=tk.LEFT)
        self._hbond_opts_visible = False

    def rotate_camera_deg(
        self,
        d_yaw_deg: float = 0.0,
        d_pitch_deg: float = 0.0,
        d_roll_deg: float = 0.0,
    ) -> None:
        """Rotate the interactive camera by explicit degree increments and refresh Blender preview."""
        if self._vp_scene is None:
            return
        if d_yaw_deg != 0.0 or d_pitch_deg != 0.0:
            self._vp_scene.orbit(math.radians(d_yaw_deg), math.radians(d_pitch_deg))
        if d_roll_deg != 0.0:
            self._vp_scene.roll(math.radians(d_roll_deg))
        self._use_interactive_cam = True
        self.var_canvas_mode.set("3d")
        self._redraw_canvas()
        self._request_blender_preview()

    def move_camera_angstrom(self, dx_a: float = 0.0, dy_a: float = 0.0) -> None:
        """Translate the interactive camera by explicit Angstrom increments and refresh Blender preview."""
        if self._vp_scene is None:
            return
        self._vp_scene.pan(dx_a, dy_a)
        self._use_interactive_cam = True
        self.var_canvas_mode.set("3d")
        self._redraw_canvas()
        self._request_blender_preview()

    def _on_step_rotate(self, yaw_sign: float, pitch_sign: float, roll_sign: float) -> None:
        try:
            step = float(self.var_rot_step.get())
        except Exception:
            step = 45.0
            self.var_rot_step.set(step)
        self.rotate_camera_deg(yaw_sign * step, pitch_sign * step, roll_sign * step)

    def _on_step_move(self, dx_sign: float, dy_sign: float) -> None:
        try:
            step = float(self.var_pan_step.get())
        except Exception:
            step = 1.0
            self.var_pan_step.set(step)
        self.move_camera_angstrom(dx_sign * step, dy_sign * step)

    def _format_input_entry_display(self) -> None:
        if getattr(self, "_formatting_input", False):
            return
        cur = self.var_input.get()
        formatted = format_input_display(cur)
        if formatted and formatted != cur:
            self._formatting_input = True
            try:
                self.var_input.set(formatted)
            finally:
                self._formatting_input = False

    def _on_input_var_changed(self) -> None:
        if not getattr(self, "_formatting_input", False):
            entry = getattr(self, "entry_in", None)
            has_focus = False
            if entry is not None:
                try:
                    has_focus = self.focus_get() is entry
                except Exception:
                    has_focus = False
            if not has_focus:
                self._format_input_entry_display()
        self._sync_type_map_visibility()

    def _sync_type_map_visibility(self) -> None:
        if not hasattr(self, "lbl_type_map"):
            return
        in_path = extract_input_path(self.var_input.get()).lower()
        show_tm = in_path.endswith((".data", ".lmp")) or bool(self.var_type_map.get().strip())
        if show_tm:
            self.lbl_type_map.grid(row=1, column=0, sticky=tk.W, pady=2)
            self.entry_type_map.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
            self.btn_inspect_tm.grid(row=1, column=2, pady=2)
        else:
            self.lbl_type_map.grid_remove()
            self.entry_type_map.grid_remove()
            self.btn_inspect_tm.grid_remove()

    def _toggle_section(self, frame: ttk.Frame) -> None:
        if frame.winfo_manager():
            frame.pack_forget()
        elif (
            frame is getattr(self, "f_rep_body", None)
            and hasattr(self, "f_axes_opts")
            and self.f_axes_opts.winfo_manager()
        ):
            frame.pack(fill=tk.X, before=self.f_axes_opts)
        else:
            frame.pack(fill=tk.X)

    def _toggle_axes_options(self) -> None:
        if self._axes_opts_visible:
            self.f_axes_opts.pack_forget()
            self._axes_opts_visible = False
        else:
            self.f_axes_opts.pack(fill=tk.X, pady=(4, 0))
            self._axes_opts_visible = True

    def _toggle_hbond_options(self) -> None:
        if getattr(self, "_hbond_opts_visible", False):
            self.f_hbond_opts.pack_forget()
            self._hbond_opts_visible = False
        else:
            self.f_hbond_opts.pack(fill=tk.X, pady=(4, 0))
            self._hbond_opts_visible = True

    def _draw_rounded_checkbox(self, canvas: tk.Canvas, checked: bool) -> None:
        canvas.delete("all")
        fill = "#0D9488" if checked else "#111318"
        outline = "#14B8A6" if checked else "#475569"
        canvas.create_rectangle(3, 3, 18, 18, fill=fill, outline=outline, width=1.4)
        if checked:
            canvas.create_line(
                6, 11, 9.5, 14.5, 15, 7, fill="#FFFFFF", width=2.0, capstyle=tk.ROUND, joinstyle=tk.ROUND
            )

    def _refresh_display_checks(self) -> None:
        for _key, (canvas, var) in self._display_checks.items():
            self._draw_rounded_checkbox(canvas, bool(var.get()))

    def _on_atoms_visibility_toggled(self) -> None:
        show = bool(self.var_show_atoms.get())
        self._show_atoms = show
        if self._vp_scene is not None:
            self._vp_scene.show_atoms = show
            self.var_canvas_mode.set("3d")
        if self._ghost_vp_scene is not None:
            self._ghost_vp_scene.show_atoms = show
        self._redraw_canvas()

    def _on_output_mask_toggled(self) -> None:
        if self._vp_scene is not None:
            self._vp_scene.show_output_mask = bool(self.var_show_output_mask.get())
            self.var_canvas_mode.set("3d")
        if self._ghost_vp_scene is not None:
            self._ghost_vp_scene.show_output_mask = bool(self.var_show_output_mask.get())
        self._redraw_canvas()

    def _on_display_type_clicked(self, key: str) -> None:
        if key in ("ball_stick", "licorice", "vdw"):
            var_map = {
                "ball_stick": self.var_disp_ball_stick,
                "licorice": self.var_disp_licorice,
                "vdw": self.var_disp_vdw,
            }
            for k_item, v_item in var_map.items():
                v_item.set(k_item == key)
            self._show_atoms = True
            if hasattr(self, "var_show_atoms"):
                self.var_show_atoms.set(True)
            if key == "ball_stick":
                self._set_representation(0.78, "auto", "bicolor")
            elif key == "licorice":
                self._set_representation(0.42, "auto", "bicolor")
            else:
                self._set_representation(1.35, "none", "style")
            return

        if key == "crystal_lattice":
            self.var_show_cell.set(not bool(self.var_show_cell.get()))
            self._refresh_display_checks()
            self._on_repr_changed()
            return

        if key == "cell_vectors":
            self.var_show_cell_vectors.set(not bool(self.var_show_cell_vectors.get()))
            self._refresh_display_checks()
            self._on_repr_changed()
            return

        if key == "polyhedra":
            self.var_show_polyhedra.set(not bool(self.var_show_polyhedra.get()))
            self._refresh_display_checks()
            self._on_repr_changed()
            return

        if key == "atom_labels":
            self.var_show_labels.set(not bool(self.var_show_labels.get()))
            self._refresh_display_checks()
            if self._vp_scene is not None:
                self._vp_scene.show_labels = bool(self.var_show_labels.get())
                self.var_canvas_mode.set("3d")
                self._redraw_canvas()
            return

        if key == "reference_axes":
            self.var_show_axes.set(not bool(self.var_show_axes.get()))
            self._refresh_display_checks()
            if self._vp_scene is not None:
                self._vp_scene.show_axes = bool(self.var_show_axes.get())
            self._redraw_canvas()
            return

        if key == "scale_bar":
            self.var_show_scale_bar.set(not bool(self.var_show_scale_bar.get()))
            self._refresh_display_checks()
            if self._vp_scene is not None:
                self._vp_scene.show_scale_bar = bool(self.var_show_scale_bar.get())
            self._redraw_canvas()
            return

        if key == "hbonds":
            self.var_show_hbonds.set(not bool(self.var_show_hbonds.get()))
            self._refresh_display_checks()
            self._on_hbonds_changed()

    def _on_axes_option_changed(self) -> None:
        if self._vp_scene is not None:
            self._vp_scene.axes_labels = bool(self.var_axes_labels.get())
        self._redraw_canvas()

    def _on_axes_fix_toggled(self) -> None:
        if self.var_axes_fixed.get():
            self._axes_pos = (0.12, 0.86)
            if self._vp_scene is not None:
                self._vp_scene.axes_pos = self._axes_pos
            self._redraw_canvas()

    def _on_backdrop_toggled(self) -> None:
        if self._vp_scene is not None:
            self._vp_scene.dark_backdrop = bool(self.var_dark_backdrop.get())
        self._redraw_canvas()

    def _recent_prefs_path(self) -> Path:
        return Path.home() / ".atomic_blender_gui.json"

    def _load_recent_files(self) -> list[str]:
        try:
            p = self._recent_prefs_path()
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                items = data.get("recent_files", [])
                if isinstance(items, list):
                    return [str(x) for x in items if isinstance(x, str) and x.strip()][:10]
        except Exception:
            pass
        return []

    def _save_recent_files(self) -> None:
        try:
            p = self._recent_prefs_path()
            payload: dict[str, Any] = {}
            if p.is_file():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = {}
            payload["recent_files"] = self._recent_files[:10]
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _add_recent_file(self, raw_path: str | Path) -> None:
        raw = str(raw_path).strip()
        if not raw:
            return
        try:
            p = Path(raw).expanduser().resolve()
            if not p.is_file():
                return
            norm = os.path.normcase(str(p))
            self._recent_files = [str(p)] + [
                x for x in self._recent_files if os.path.normcase(x) != norm
            ]
            self._recent_files = self._recent_files[:10]
            self._save_recent_files()
            self._rebuild_recent_menu()
        except Exception:
            pass

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.delete(0, tk.END)
        existing = [x for x in self._recent_files if Path(x).is_file()]
        if not existing:
            self.recent_menu.add_command(label="(No recent structures)", state=tk.DISABLED)
            return
        for path_str in existing:
            p = Path(path_str)
            self.recent_menu.add_command(
                label=f"{p.name}  ({p.parent.name})",
                command=lambda s=path_str: self._open_recent_file(s),
            )
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Clear Recent List", command=self._clear_recent_files)

    def _clear_recent_files(self) -> None:
        self._recent_files = []
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _open_recent_file(self, path_str: str) -> None:
        p = Path(path_str).expanduser()
        if not p.is_file():
            self.var_status.set(f"Recent file missing: {p.name}")
            self._rebuild_recent_menu()
            return
        self.var_input.set(str(p))
        self.var_output.set(str(p.with_suffix(".png")))
        self._schedule_inspect()
        self.var_status.set(f"Loading recent structure: {p.name}")

    def _current_canvas_pil_image(self) -> Any:
        if not _HAS_PIL:
            return None
        if self.var_canvas_mode.get() == "png" and isinstance(self._pil_image, Image.Image):
            out = self._pil_image.convert("RGBA").copy()
            res_w = max(16, int(self.var_res_w.get()))
            res_h = max(16, int(self.var_res_h.get()))
            if self._vp_scene is not None and (self.var_show_labels.get() or self._selected_atoms):
                render_atom_labels_and_selection_overlay(
                    out,
                    self._vp_scene,
                    res_w,
                    res_h,
                    draw_labels=bool(self.var_show_labels.get()),
                    draw_selection=bool(self._selected_atoms),
                )
            if (
                self._vp_scene is not None
                and self.var_show_measure_overlay.get()
                and len(self._selection_order) in (2, 3, 4)
            ):
                render_measurement_callout_overlay(
                    out,
                    self._vp_scene,
                    self._selection_order,
                    res_w,
                    res_h,
                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                )
            from_tmp_preview = (
                self._preview_path is not None
                and hasattr(self, "_preview_tmp_dir")
                and str(self._preview_path).startswith(str(self._preview_tmp_dir.name))
            )
            if self.var_show_scale_bar.get() and self._vp_scene is not None and from_tmp_preview:
                render_scale_bar_overlay(
                    out,
                    self._vp_scene.ortho_scale,
                    res_w,
                    res_h,
                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                )
            cmap_info = getattr(self, "_active_colormap_info", None)
            if cmap_info is not None and from_tmp_preview:
                render_colorbar_overlay(
                    out,
                    *cmap_info,
                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                )
            if self.var_show_axes.get() and from_tmp_preview:
                quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                render_reference_axes_overlay(
                    out,
                    quat,
                    axes_pos=self._axes_pos,
                    show_labels=bool(self.var_axes_labels.get()),
                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                )
            return out
        if self._vp_scene is not None:
            cw = max(600, int(self.canvas.winfo_width()))
            ch = max(600, int(self.canvas.winfo_height()))
            res_w = max(16, int(self.var_res_w.get()))
            res_h = max(16, int(self.var_res_h.get()))
            return self._vp_scene.render_image(cw, ch, res_w, res_h, interactive=False)
        return None

    def _on_copy_image(self) -> None:
        img = self._current_canvas_pil_image()
        if img is None:
            self.var_status.set("Nothing in viewport to copy")
            return
        if copy_pil_image_to_clipboard(img):
            w, h = img.size
            self.var_status.set(f"Copied {w}×{h} image to clipboard")
        else:
            self.var_status.set("Clipboard image copy is unavailable on this platform")

    def _is_typing_focus(self) -> bool:
        try:
            w = self.focus_get()
        except Exception:
            return False
        if w is None:
            return False
        return w.winfo_class() in {"Entry", "TEntry", "Spinbox", "TSpinbox", "Text", "TCombobox"}

    def _bind_keyboard_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self._on_key_shortcut("o", ctrl=True))
        self.bind("<Control-O>", lambda _e: self._on_key_shortcut("o", ctrl=True))
        self.bind("<Control-r>", lambda _e: self._on_key_shortcut("r", ctrl=True))
        self.bind("<Control-R>", lambda _e: self._on_key_shortcut("r", ctrl=True))
        self.bind("<Control-s>", lambda _e: self._on_key_shortcut("s", ctrl=True))
        self.bind("<Control-S>", lambda _e: self._on_key_shortcut("s", ctrl=True))
        self.bind("<Control-c>", lambda _e: self._on_key_shortcut("c", ctrl=True))
        self.bind("<Control-C>", lambda _e: self._on_key_shortcut("c", ctrl=True))
        for keysym, norm_key in (
            ("<KeyPress-r>", "r"),
            ("<KeyPress-R>", "r"),
            ("<Home>", "Home"),
            ("<KeyPress-l>", "l"),
            ("<KeyPress-L>", "l"),
            ("<KeyPress-a>", "a"),
            ("<KeyPress-A>", "a"),
            ("<Escape>", "Escape"),
            ("<Left>", "Left"),
            ("<Right>", "Right"),
            ("<Up>", "Up"),
            ("<Down>", "Down"),
            ("<Shift-Left>", "Left"),
            ("<Shift-Right>", "Right"),
            ("<Shift-Up>", "Up"),
            ("<Shift-Down>", "Down"),
            ("<KeyPress-plus>", "+"),
            ("<KeyPress-equal>", "+"),
            ("<KP_Add>", "+"),
            ("<KeyPress-minus>", "-"),
            ("<KeyPress-underscore>", "-"),
            ("<KP_Subtract>", "-"),
        ):
            is_shift = keysym.startswith("<Shift-")
            self.bind(keysym, lambda _e, k=norm_key, sh=is_shift: self._on_key_shortcut(k, shift=sh))

    def _on_key_shortcut(self, key: str, ctrl: bool = False, shift: bool = False) -> str | None:
        if ctrl:
            if key == "o":
                self._browse_input()
                return "break"
            if key in ("r", "s"):
                self._on_render()
                return "break"
            if key == "c":
                if self._is_typing_focus():
                    return None
                self._on_copy_image()
                return "break"
            return None

        if self._is_typing_focus():
            return None

        if key in ("r", "Home"):
            self._on_reset_view()
            return "break"
        if key == "l":
            self.var_show_labels.set(not bool(self.var_show_labels.get()))
            self._on_labels_toggled()
            return "break"
        if key == "a":
            self._on_display_type_clicked("reference_axes")
            return "break"
        if key == "Escape":
            self._clear_selection()
            return "break"
        if key in ("Left", "Right", "Up", "Down"):
            dx = -1.0 if key == "Left" else (1.0 if key == "Right" else 0.0)
            dy = 1.0 if key == "Up" else (-1.0 if key == "Down" else 0.0)
            if shift:
                self._on_step_move(dx, dy)
            else:
                self._on_step_rotate(dx, dy, 0.0)
            return "break"
        if key == "+":
            self._zoom_3d(0.85)
            return "break"
        if key == "-":
            self._zoom_3d(1.176)
            return "break"
        return None

    def _build_preview_and_log(self, parent: ttk.Frame) -> None:
        v_pane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        v_pane.pack(fill=tk.BOTH, expand=True)
        self._v_pane = v_pane

        f_prev = ttk.Frame(v_pane, padding=(2, 2), style="Outer.TFrame")
        tb = ttk.Frame(f_prev, padding=(4, 3))
        tb.pack(fill=tk.X, pady=(0, 4))
        DarkRadiobutton(
            tb, text="3D", variable=self.var_canvas_mode, value="3d", command=self._redraw_canvas
        ).pack(side=tk.LEFT)
        DarkRadiobutton(
            tb, text="Render", variable=self.var_canvas_mode, value="png", command=self._redraw_canvas
        ).pack(side=tk.LEFT, padx=(4, 6))
        DarkCheckbutton(
            tb, text="Frame", variable=self.var_show_output_mask, command=self._on_output_mask_toggled
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tb, text="Fit", width=4, command=self._on_reset_view).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(tb, text="Save View", command=self._on_save_view_json).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(tb, text="Copy", width=5, command=self._on_copy_image).pack(side=tk.LEFT, padx=(0, 6))
        DarkCheckbutton(
            tb, text="Dark", variable=self.var_dark_backdrop, command=self._on_backdrop_toggled
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._btn_log_toggle = ttk.Button(tb, text="Log", width=5, command=self._toggle_log_panel)
        self._btn_log_toggle.pack(side=tk.RIGHT)
        self.btn_settings = ttk.Button(tb, text="⚙ Settings", command=self._toggle_settings_panel)
        self.btn_settings.pack(side=tk.RIGHT, padx=(0, 4))

        self.canvas = tk.Canvas(f_prev, bg="#0F1115", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_lbutton_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_mouse_up)
        self.canvas.bind("<ButtonPress-2>", lambda e: self._on_canvas_pan_down(e))
        self.canvas.bind("<B2-Motion>", self._on_canvas_motion)
        self.canvas.bind("<ButtonRelease-2>", self._on_canvas_mouse_up)
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_canvas_pan_down(e))
        self.canvas.bind("<B3-Motion>", self._on_canvas_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.canvas.bind("<Button-4>", lambda _e: self._zoom_3d(0.9))
        self.canvas.bind("<Button-5>", lambda _e: self._zoom_3d(1.111))
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Motion>", self._on_canvas_hover_motion)
        self.canvas.bind("<Leave>", self._on_canvas_hover_leave)

        self._f_log = ttk.LabelFrame(v_pane, text="LOG", padding=(6, 4))
        self.log_text = scrolledtext.ScrolledText(
            self._f_log,
            height=5,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg="#111318",
            fg="#CBD5E1",
            insertbackground="#E2E8F0",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        v_pane.add(f_prev, weight=6)
        self._log_visible = False

    def _toggle_log_panel(self) -> None:
        if not hasattr(self, "_v_pane") or not hasattr(self, "_f_log"):
            return
        if self._log_visible:
            self._v_pane.forget(self._f_log)
            self._log_visible = False
        else:
            self._v_pane.add(self._f_log, weight=1)
            self._log_visible = True

    def _on_scale_changed(self, _val: str) -> None:
        self.var_radius_label.set(f"{self.var_radius_scale.get():.2f}×")

    def _on_labels_toggled(self) -> None:
        if self._vp_scene is not None:
            self._vp_scene.show_labels = bool(self.var_show_labels.get())
            self._vp_scene.show_measure_overlay = bool(self.var_show_measure_overlay.get())
        self._redraw_canvas()

    def _on_opacity_changed(self, _val: str) -> None:
        pct = int(round(float(self.var_sel_opacity.get()) * 100.0))
        self.var_sel_opacity_label.set(f"{pct}%")
        self._sync_selection_and_alphas(trigger_blender=False)

    def _on_opacity_released(self) -> None:
        pct = int(round(float(self.var_sel_opacity.get()) * 100.0))
        self.var_sel_opacity_label.set(f"{pct}%")
        self._sync_selection_and_alphas(trigger_blender=True)

    def _reset_opacity(self) -> None:
        self.var_sel_opacity.set(1.0)
        self.var_sel_opacity_label.set("100%")
        self._sync_selection_and_alphas(trigger_blender=True)

    def _update_measurement_banner(self) -> None:
        if not hasattr(self, "var_measurement"):
            return
        if self._structure_cache is None:
            self.var_measurement.set("Double-click 1–4 atoms for coords / distance / angle / dihedral")
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        n = min(len(syms), len(pos))
        self._selection_order = [i for i in self._selection_order if 0 <= i < n and i in self._selected_atoms]
        for idx in sorted(self._selected_atoms):
            if 0 <= idx < n and idx not in self._selection_order:
                self._selection_order.append(idx)
        if self._vp_scene is not None:
            self._vp_scene.selection_order = list(self._selection_order)
        ref_pos = self._ghost_structure_cache.get("positions") if self._ghost_structure_cache is not None else None
        self.var_measurement.set(compute_selection_measurement(syms, pos, self._selection_order, reference_positions=ref_pos))

    def _toggle_atom_selection(self, idx: int) -> None:
        if idx in self._selected_atoms:
            self._selected_atoms.remove(idx)
            self._selection_order = [i for i in self._selection_order if i != idx]
        else:
            self._selected_atoms.add(idx)
            if idx not in self._selection_order:
                self._selection_order.append(idx)

    def _select_all_atoms(self) -> None:
        if self._structure_cache is None:
            return
        n = len(self._structure_cache.get("symbols", []))
        self._selected_atoms = set(range(n))
        self._selection_order = list(range(n))
        self._sync_selection_and_alphas(trigger_blender=True)

    def _clear_selection(self) -> None:
        self._selected_atoms.clear()
        self._selection_order.clear()
        self._sync_selection_and_alphas(trigger_blender=True)

    def _invert_selection(self) -> None:
        if self._structure_cache is None:
            return
        n = len(self._structure_cache.get("symbols", []))
        self._selected_atoms = set(range(n)) - self._selected_atoms
        self._selection_order = sorted(self._selected_atoms)
        self._sync_selection_and_alphas(trigger_blender=True)

    def _select_same_element(self) -> None:
        if self._structure_cache is None or not self._selected_atoms:
            return
        syms = self._structure_cache.get("symbols", [])
        target_syms = {syms[i] for i in self._selected_atoms if i < len(syms)}
        new_sel = {i for i, s in enumerate(syms) if s in target_syms}
        self._selected_atoms = new_sel
        self._selection_order = [i for i in self._selection_order if i in new_sel]
        for i in sorted(new_sel):
            if i not in self._selection_order:
                self._selection_order.append(i)
        self._sync_selection_and_alphas(trigger_blender=True)

    def _rebuild_element_chips(self) -> None:
        if not hasattr(self, "f_elem_chips"):
            return
        for child in self.f_elem_chips.winfo_children():
            child.destroy()
        self._elem_chip_buttons = {}
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        counts: dict[str, int] = {}
        for s in syms:
            counts[s] = counts.get(s, 0) + 1
        row_frame: ttk.Frame | None = None
        for idx, (sym, cnt) in enumerate(counts.items()):
            if idx % 4 == 0:
                row_frame = ttk.Frame(self.f_elem_chips)
                row_frame.pack(fill=tk.X, pady=1)
            assert row_frame is not None
            btn = tk.Button(
                row_frame,
                text=f"{sym} ({cnt})",
                command=lambda s=sym: self._toggle_element_selection(s),
                relief=tk.FLAT,
                bd=1,
                padx=6,
                pady=2,
                cursor="hand2",
                font=("Segoe UI", 8, "bold"),
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            btn.bind("<Button-3>", lambda _e, s=sym: self._on_element_chip_right_click(s))
            self._elem_chip_buttons[sym] = btn
        self._refresh_element_chip_styles()

    def _on_element_chip_right_click(self, symbol: str) -> None:
        style = self._resolve_style_dict()
        cur_hex = style.get("elements", {}).get("colors", {}).get(symbol, "#CCCCCC")
        chosen = colorchooser.askcolor(color=cur_hex, title=f"Choose Color for {symbol}")
        if not chosen or not chosen[1]:
            return
        self._custom_element_colors[symbol] = str(chosen[1]).upper()
        self.var_status.set(f"{symbol} color -> {self._custom_element_colors[symbol]}")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _reset_custom_element_colors(self) -> None:
        if not self._custom_element_colors:
            self.var_status.set("Element colors already match base style")
            return
        self._custom_element_colors.clear()
        self.var_status.set("Restored default element colors")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _reset_focus_recipes(self) -> None:
        self._selected_atoms.clear()
        self._selection_order.clear()
        self.var_sel_opacity.set(0.30)
        self.var_sel_opacity_label.set("30%")
        self.var_alpha_invert.set(False)
        self.var_unsel_scale_mode.set("same")
        self.var_color_map.set("none")
        self.var_status.set("Focus & opacity reset")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _expand_selection_neighbors(self) -> None:
        if self._structure_cache is None:
            return
        pos = self._structure_cache.get("positions", [])
        if not pos:
            return
        try:
            cutoff = max(0.5, float(self.var_neighbor_cutoff.get()))
        except Exception:
            cutoff = 3.0
            self.var_neighbor_cutoff.set(cutoff)
        seeds = set(self._selected_atoms)
        if not seeds:
            seeds = auto_detect_adsorbate_indices(self._structure_cache.get("symbols", []), pos)
        expanded = find_neighbor_atoms(pos, seeds, cutoff_a=cutoff)
        self._selected_atoms = expanded
        for idx in sorted(expanded):
            if idx not in self._selection_order:
                self._selection_order.append(idx)
        self.var_status.set(f"Expanded shell (+{cutoff:.1f} Å): {len(expanded)} atoms selected")
        self._sync_selection_and_alphas(trigger_blender=True)

    def _apply_adsorbate_spotlight_recipe(self) -> None:
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        if not syms or not pos:
            return
        try:
            cutoff = max(1.0, float(self.var_neighbor_cutoff.get()))
        except Exception:
            cutoff = 3.2
        focal, alphas = compute_adsorbate_spotlight_alphas(
            syms, pos, focal_indices=self._selected_atoms or None, shell_cutoff_a=cutoff
        )
        self._selected_atoms = set(focal)
        self._selection_order = sorted(focal)
        if self.var_unsel_scale_mode.get() == "same":
            self.var_unsel_scale_mode.set("slim")
        self.var_status.set(f"Adsorbate Spotlight active ({len(focal)} focal atoms)")
        self._sync_selection_and_alphas(trigger_blender=True, alphas_override=alphas)

    def _apply_depth_fade_recipe(self) -> None:
        if self._structure_cache is None:
            return
        pos = self._structure_cache.get("positions", [])
        if not pos:
            return
        alphas = compute_depth_fade_alphas(pos)
        self.var_status.set("Applied Z-Depth Fade opacity gradient")
        self._sync_selection_and_alphas(trigger_blender=True, alphas_override=alphas)

    def _apply_smart_slab_spotlight(self) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set("side-oblique")
        self.var_crop_z_mode.set("auto")
        self.var_show_scale_bar.set(True)
        self._set_resolution(1400, 1200, trigger_preview=False)
        self._use_interactive_cam = False
        self._rebuild_viewport(reset_camera=True, trigger_preview=False)
        self._apply_adsorbate_spotlight_recipe()

    def _on_render_top_side_panel(self) -> None:
        if self._busy:
            return
        if not self._ensure_blender_available():
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        base_out = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        panel_png = base_out.with_name(f"{base_out.stem}_panel.png")
        self._task_start = time.monotonic()
        self._set_busy(True, "Rendering Top + Side panel...")
        dark_bg = bool(self.var_dark_backdrop.get())

        def worker() -> None:
            with tempfile.TemporaryDirectory(prefix="atomic_panel_") as tmp_dir:
                tmp_p = Path(tmp_dir)
                top_png = tmp_p / "top.png"
                side_png = tmp_p / "side.png"
                cmds = []
                for preset, out_p in (("c", top_png), ("side-oblique", side_png)):
                    c = GuiRenderConfig(
                        input_path=cli_cfg.input_path,
                        output_path=str(out_p),
                        style=cli_cfg.style,
                        view_mode="preset",
                        view_preset=preset,
                        view_file="",
                        fit_view=False,
                        type_map=cli_cfg.type_map,
                        replicate=cli_cfg.replicate,
                        wrap=cli_cfg.wrap,
                        crop_z_mode=cli_cfg.crop_z_mode,
                        crop_z_custom=cli_cfg.crop_z_custom,
                        bonds=cli_cfg.bonds,
                        bond_color=cli_cfg.bond_color,
                        radius_scale=cli_cfg.radius_scale,
                        show_cell=cli_cfg.show_cell,
                        resolution=cli_cfg.resolution,
                        samples=cli_cfg.samples,
                        blender_exe=cli_cfg.blender_exe,
                    )
                    args = build_render_cli_args(c, out_png=out_p)
                    self._queue.put(("log", f"$ atomic-blender {' '.join(args)}"))
                    proc = self._popen_cli(args)
                    assert proc.stdout is not None
                    for raw_line in proc.stdout:
                        self._queue.put(("log", raw_line))
                    rc = proc.wait()
                    if rc != 0:
                        self._queue.put(("done_panel", (rc, panel_png, f"Panel render exited with status {rc}")))
                        return
                    cmds.append(out_p)
                try:
                    t_img = Image.open(top_png).convert("RGBA")
                    s_img = Image.open(side_png).convert("RGBA")
                    panel = compose_dual_view_panel(t_img, s_img, dark_backdrop=dark_bg)
                    panel.save(panel_png)
                    self._queue.put(("done_panel", (0, panel_png, "")))
                except Exception as exc:
                    self._queue.put(("done_panel", (1, panel_png, str(exc))))

        threading.Thread(target=worker, daemon=True).start()

    def _set_material_finish(self, finish_key: str) -> None:
        if finish_key not in MATERIAL_FINISH_PRESETS:
            finish_key = "studio"
        self.var_material_finish.set(finish_key)
        self.var_status.set(f"Material finish: {finish_key}")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_colormap_changed(self) -> None:
        cmap = self.var_color_map.get()
        self.var_status.set("Element CPK colors" if cmap == "none" else f"Z-Height colormap: {cmap}")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_hbonds_changed(self) -> None:
        if self._structure_cache is not None:
            self._rebuild_viewport(reset_camera=False, trigger_preview=True)
        else:
            self._refresh_display_checks()

    def _rebuild_compare_state(self, trigger_preview: bool = True) -> None:
        if self._base_structure_cache is None:
            return
        if self._ghost_structure_cache is not None and self.var_compare_mode.get() == "overlay":
            merged = merge_ghost_structure(
                self._base_structure_cache,
                self._ghost_structure_cache,
                ghost_alpha=float(self.var_ghost_alpha.get()),
            )
            self._structure_cache = merged
        else:
            self._structure_cache = dict(self._base_structure_cache)
        self._structure_version += 1
        self._written_struct_version = -1
        self._rebuild_viewport(reset_camera=False, trigger_preview=trigger_preview)

    def _on_ghost_options_changed(self, trigger_blender: bool = True) -> None:
        if self._ghost_structure_cache is None:
            return
        self._rebuild_compare_state(trigger_preview=trigger_blender)

    def _apply_ghost_structure_dict(self, ghost_neutral: dict[str, Any], label_name: str = "ghost") -> None:
        if self._structure_cache is None:
            return
        if self._base_structure_cache is None:
            self._base_structure_cache = dict(self._structure_cache)
        self._ghost_structure_cache = dict(ghost_neutral)
        self._ghost_label_name = label_name
        self.var_status.set(f"Compare active ({self.var_compare_mode.get()}): {label_name}")
        self._rebuild_compare_state(trigger_preview=True)

    def _pin_current_as_ghost(self) -> None:
        if self._structure_cache is None:
            messagebox.showinfo("Compare Reference", "Load a structure first.")
            return
        ref_src = dict(self._base_structure_cache if self._base_structure_cache is not None else self._structure_cache)
        lbl = f"Frame {self._trajectory_index + 1}" if len(self._trajectory_frames) > 1 else "Pinned Current"
        self._apply_ghost_structure_dict(ref_src, label_name=lbl)

    def _on_load_ghost_structure(self) -> None:
        if self._structure_cache is None:
            messagebox.showinfo("Ghost Overlay", "Load a main structure first.")
            return
        path = filedialog.askopenfilename(
            title="Select Reference / Initial Structure for Ghost Overlay",
            filetypes=[
                ("Structure files", "*.cif *.vasp *.xyz *.extxyz *.data *.lmp *.json POSCAR CONTCAR"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            p = Path(path).expanduser()
            type_map = self.var_type_map.get().strip() or None
            if p.suffix.lower() == ".json":
                ghost_dict = json.loads(p.read_text(encoding="utf-8"))
            else:
                from atomic_blender.convert import atoms_to_neutral_dict, read_structure

                rep = (int(self.var_rep_x.get()), int(self.var_rep_y.get()), int(self.var_rep_z.get()))
                atoms = read_structure(p, type_map=type_map)
                eff_rep = rep if atoms.cell.rank == 3 else (1, 1, 1)
                ghost_dict = atoms_to_neutral_dict(
                    atoms, source=str(p), replicate=eff_rep, wrap=bool(self.var_wrap.get())
                )
            self._apply_ghost_structure_dict(ghost_dict, label_name=p.name)
        except Exception as exc:
            messagebox.showerror("Ghost Overlay Error", str(exc))

    def _clear_ghost_structure(self) -> None:
        if self._ghost_structure_cache is None or self._base_structure_cache is None:
            self.var_status.set("No ghost overlay active")
            return
        self._ghost_structure_cache = None
        self._ghost_label_name = ""
        self._ghost_vp_scene = None
        self._structure_cache = dict(self._base_structure_cache)
        self._structure_cache.pop("dashed_bonds", None)
        self._structure_cache.pop("ghost_start_index", None)
        self._structure_version += 1
        self._written_struct_version = -1
        self.var_status.set("Cleared ghost overlay")
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_batch_render_files(self) -> None:
        if self._busy:
            return
        if not self._ensure_blender_available():
            return
        paths = filedialog.askopenfilenames(
            title="Select Structures for Batch Render",
            filetypes=[
                ("Structure files", "*.cif *.vasp *.xyz *.extxyz *.data *.lmp *.json POSCAR CONTCAR"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        out_dir_str = filedialog.askdirectory(title="Select Output Folder for Batch PNGs")
        if not out_dir_str:
            return
        try:
            cfg = self._collect_config()
            eff_style = self._effective_style_path_for_cli(cfg.style)
        except Exception as exc:
            messagebox.showerror("Batch Render Error", str(exc))
            return

        out_dir = Path(out_dir_str)
        file_list = [Path(p) for p in paths]
        out_paths = resolve_batch_output_paths(file_list, out_dir)
        self._task_start = time.monotonic()
        self._set_busy(True, f"Batch rendering {len(file_list)} files...")

        view_file_arg = ""
        if self._vp_scene is not None and self._use_interactive_cam:
            view_dict = self._vp_scene.to_view_dict(cfg.resolution[0], cfg.resolution[1])
            tmp_fd, tmp_name = tempfile.mkstemp(prefix="atomic_batch_vp_", suffix=".json")
            os.close(tmp_fd)
            Path(tmp_name).write_text(json.dumps(view_dict, indent=2), encoding="utf-8")
            self._temp_view_files.append(Path(tmp_name))
            view_file_arg = tmp_name

        def worker() -> None:
            last_png: Path | None = None
            for idx, (src, out_p) in enumerate(zip(file_list, out_paths), start=1):
                c = GuiRenderConfig(
                    input_path=str(src),
                    output_path=str(out_p),
                    style=eff_style,
                    view_mode="file" if view_file_arg else cfg.view_mode,
                    view_preset=cfg.view_preset,
                    view_file=view_file_arg or cfg.view_file,
                    fit_view=False if view_file_arg else cfg.fit_view,
                    type_map=cfg.type_map,
                    replicate=cfg.replicate,
                    wrap=cfg.wrap,
                    crop_z_mode=cfg.crop_z_mode,
                    crop_z_custom=cfg.crop_z_custom,
                    bonds=cfg.bonds,
                    bond_color=cfg.bond_color,
                    radius_scale=cfg.radius_scale,
                    show_cell=cfg.show_cell,
                    resolution=cfg.resolution,
                    samples=cfg.samples,
                    blender_exe=cfg.blender_exe,
                )
                args = build_render_cli_args(c, out_png=out_p)
                self._queue.put(("log", f"[{idx}/{len(file_list)}] $ atomic-blender {' '.join(args)}"))
                proc = self._popen_cli(args)
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    self._queue.put(("log", raw_line))
                rc = proc.wait()
                if rc != 0:
                    self._queue.put(("done_batch", (rc, idx - 1, len(file_list), last_png)))
                    return
                last_png = out_p
            self._queue.put(("done_batch", (0, len(file_list), len(file_list), last_png)))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_slice_mask_to_alphas(self, alphas: list[float]) -> list[float]:
        """Attenuate cut-away atoms according to Cross-Section Slice plane, depth, and cell vectors."""
        if self._structure_cache is None or not hasattr(self, "var_slice_mode"):
            return alphas
        mode = self.var_slice_mode.get()
        if mode == "none":
            return alphas
        pos = self._structure_cache.get("positions", [])
        if len(pos) != len(alphas) or len(pos) == 0:
            return alphas
        quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
        has_cell = bool(self._structure_cache.get("has_cell", False))
        cell_vecs = self._structure_cache.get("cell") if has_cell else None
        origin = self._structure_cache.get("origin") if has_cell else None
        mask = compute_slice_mask(
            pos,
            mode=mode,
            depth_fraction=float(self.var_slice_depth.get()),
            quat=quat,
            cell_vecs=cell_vecs,
            origin=origin,
        )
        cut_alpha = 0.06 if bool(self.var_slice_ghost.get()) else 0.02
        return [a if keep else min(a, cut_alpha) for a, keep in zip(alphas, mask)]

    def _on_slice_changed(self, trigger_blender: bool = False) -> None:
        if self._structure_cache is None:
            return
        self._sync_selection_and_alphas(trigger_blender=trigger_blender)
        mode = self.var_slice_mode.get()
        if mode == "none":
            self.var_status.set("Cross-section slice disabled")
        else:
            pct = int(round(float(self.var_slice_depth.get()) * 100.0))
            self.var_status.set(f"Cross-section slice: {mode} <= {pct}%")

    def _reset_slice(self) -> None:
        self.var_slice_mode.set("none")
        self.var_slice_depth.set(0.60)
        self.var_slice_ghost.set(True)
        self._on_slice_changed(trigger_blender=True)

    def _update_frame_controls(self) -> None:
        n_frames = max(1, len(self._trajectory_frames))
        cur_1 = min(n_frames, max(1, self._trajectory_index + 1))
        if hasattr(self, "var_frame_label"):
            self.var_frame_label.set(f"Frame {cur_1} / {n_frames}")
        if hasattr(self, "slider_frame"):
            self.slider_frame._from = 1.0
            self.slider_frame._to = float(max(2, n_frames))
        if hasattr(self, "var_frame_pos"):
            self.var_frame_pos.set(float(cur_1))
        if hasattr(self, "var_frame_pinned"):
            is_end = self._trajectory_index in (0, n_frames - 1)
            self.var_frame_pinned.set(is_end or (self._trajectory_index in self._strip_pinned_frames))
        self._draw_energy_profile()

    def _on_toggle_pin_frame(self) -> None:
        if not self._trajectory_frames:
            return
        idx = self._trajectory_index
        if bool(self.var_frame_pinned.get()):
            self._strip_pinned_frames.add(idx)
        elif idx not in (0, len(self._trajectory_frames) - 1):
            self._strip_pinned_frames.discard(idx)
        else:
            self.var_frame_pinned.set(True)
        self._draw_energy_profile()

    def _draw_energy_profile(self) -> None:
        if not hasattr(self, "energy_canvas") or not self.energy_canvas.winfo_exists():
            return
        cv = self.energy_canvas
        cv.delete("all")
        w = max(180, cv.winfo_width())
        h = max(54, cv.winfo_height())
        frames = self._trajectory_frames
        n = len(frames)
        if n < 2:
            cv.create_text(w * 0.5, h * 0.5, text="Load multi-frame trajectory (.xyz/.extxyz) for NEB curve", fill="#64748B", font=("Segoe UI", 8))
            return

        energies = [f.get("energy_ev") for f in frames]
        has_e = all(isinstance(e, (int, float)) for e in energies)
        if has_e:
            e0 = float(energies[0])  # type: ignore[arg-type]
            vals = [float(e) - e0 for e in energies]  # type: ignore[arg-type]
            tag_lbl = f"ΔE (peak {max(vals):.3f} eV)"
        else:
            p0 = np.asarray(frames[0].get("positions", []), dtype=np.float64)
            vals = []
            for f in frames:
                pk = np.asarray(f.get("positions", []), dtype=np.float64)
                vals.append(float(np.sqrt(np.mean(np.sum((pk - p0) ** 2, axis=1)))) if pk.shape == p0.shape and len(pk) else 0.0)
            tag_lbl = "Path RMSD (Å)"

        vmin, vmax = min(vals), max(vals)
        span = max(1e-6, vmax - vmin)
        pad_x, pad_top, pad_bot = 14.0, 14.0, 12.0
        pts: list[tuple[float, float]] = []
        for idx, val in enumerate(vals):
            x = pad_x + (idx / max(1, n - 1)) * (w - 2.0 * pad_x)
            y = (h - pad_bot) - ((val - vmin) / span) * (h - pad_top - pad_bot)
            pts.append((x, y))
        for k in range(len(pts) - 1):
            cv.create_line(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], fill="#0D9488", width=2)
        strip_set = set(select_path_strip_indices(n, max_panels=6, pinned_indices=self._strip_pinned_frames))
        for idx, (x, y) in enumerate(pts):
            is_cur = idx == self._trajectory_index
            is_pin = idx in strip_set
            r = 4.5 if is_cur else (3.3 if is_pin else 2.5)
            fill = "#F97316" if is_cur else ("#2DD4BF" if is_pin else "#64748B")
            outline = "#FFFFFF" if is_cur else ""
            cv.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=1.2)
        cv.create_text(6, 7, text=tag_lbl, anchor=tk.W, fill="#94A3B8", font=("Segoe UI", 7))

    def _on_energy_canvas_click(self, event: tk.Event) -> None:
        n = len(self._trajectory_frames)
        if n < 2:
            return
        w = max(180, self.energy_canvas.winfo_width())
        pad_x = 14.0
        frac = max(0.0, min(1.0, (float(event.x) - pad_x) / max(1.0, w - 2.0 * pad_x)))
        idx = int(round(frac * (n - 1)))
        self._select_trajectory_frame(idx, trigger_preview=True)

    def _select_trajectory_frame(self, index: int, trigger_preview: bool = True) -> None:
        if not self._trajectory_frames:
            return
        idx = max(0, min(len(self._trajectory_frames) - 1, int(index)))
        self._trajectory_index = idx
        frame = dict(self._trajectory_frames[idx])
        self._base_structure_cache = dict(frame)
        if self._ghost_structure_cache is not None and self.var_compare_mode.get() == "overlay":
            self._structure_cache = merge_ghost_structure(
                self._base_structure_cache,
                self._ghost_structure_cache,
                ghost_alpha=float(self.var_ghost_alpha.get()),
            )
        else:
            self._structure_cache = frame
        self._structure_version += 1
        self._written_struct_version = -1
        self._update_frame_controls()
        self._rebuild_viewport(reset_camera=False, trigger_preview=trigger_preview)
        self.var_status.set(f"Trajectory frame {idx + 1} / {len(self._trajectory_frames)}")

    def _step_trajectory_frame(self, delta: int) -> None:
        if not self._trajectory_frames:
            return
        self._select_trajectory_frame(self._trajectory_index + int(delta), trigger_preview=True)

    def _on_frame_slider_changed(self, _val: str) -> None:
        if not self._trajectory_frames:
            return
        idx = int(round(float(self.var_frame_pos.get()))) - 1
        idx = max(0, min(len(self._trajectory_frames) - 1, idx))
        if idx != self._trajectory_index:
            self._select_trajectory_frame(idx, trigger_preview=False)

    def _on_frame_slider_released(self) -> None:
        if self._trajectory_frames:
            self._request_blender_preview()

    def _on_render_path_strip(self) -> None:
        if self._busy:
            return
        if not self._ensure_blender_available():
            return
        if len(self._trajectory_frames) < 2:
            messagebox.showinfo("Path Strip", "Load a multi-frame trajectory (.xyz / .extxyz / .traj) first.")
            return
        if not _HAS_PIL:
            messagebox.showerror("Pillow Required", "Pillow is required to compose a multi-frame strip.")
            return
        try:
            cfg = self._collect_config()
            in_path = Path(cfg.input_path).expanduser()
            eff_style = self._effective_style_path_for_cli(cfg.style)
            validate_gui_config(cfg)
        except Exception as exc:
            messagebox.showerror("Path Strip Failed", str(exc))
            return

        total = len(self._trajectory_frames)
        indices = select_path_strip_indices(
            total,
            max_panels=6,
            pinned_indices=self._strip_pinned_frames,
        )

        out_base = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        strip_png = out_base.with_name(f"{out_base.stem}_strip.png")
        dark_bg = bool(self.var_dark_backdrop.get())
        show_cvec = bool(self.var_show_cell_vectors.get())
        show_poly = bool(self.var_show_polyhedra.get())
        style_dict = self._resolve_style_dict()

        if self._vp_scene is not None:
            view_dict = self._vp_scene.to_view_dict(cfg.resolution[0], cfg.resolution[1])
            tmp_vfd, tmp_vname = tempfile.mkstemp(prefix="atomic_strip_vp_", suffix=".json")
            os.close(tmp_vfd)
            Path(tmp_vname).write_text(json.dumps(view_dict, indent=2), encoding="utf-8")
            self._temp_view_files.append(Path(tmp_vname))
            view_mode_eff, view_file_eff, fit_eff = "file", tmp_vname, False
        else:
            view_mode_eff, view_file_eff, fit_eff = cfg.view_mode, cfg.view_file, cfg.fit_view

        frame_jobs: list[tuple[int, str, Path]] = []
        for fi in indices:
            fdict = dict(self._trajectory_frames[fi])
            if show_cvec:
                fdict["show_cell_vectors"] = True
            if show_poly and fdict.get("symbols") and fdict.get("positions"):
                fdict["polyhedra"] = find_coordination_polyhedra(
                    fdict["symbols"],
                    fdict["positions"],
                    style=style_dict,
                    cell_vecs=fdict.get("cell") if fdict.get("has_cell") else None,
                    pbc=fdict.get("pbc"),
                )
            tmp_sfd, tmp_sname = tempfile.mkstemp(prefix=f"atomic_strip_f{fi}_", suffix=".json")
            os.close(tmp_sfd)
            Path(tmp_sname).write_text(json.dumps(fdict), encoding="utf-8")
            self._temp_view_files.append(Path(tmp_sname))
            f_png = strip_png.parent / f"{out_base.stem}_frame_{fi + 1:02d}.png"
            frame_jobs.append((fi, tmp_sname, f_png))

        self._task_start = time.monotonic()
        self._set_busy(True, f"Rendering {len(frame_jobs)}-frame path strip...")

        def worker() -> None:
            rendered_imgs: list[Any] = []
            titles: list[str] = []
            for fi, struct_json, out_p in frame_jobs:
                c = GuiRenderConfig(
                    input_path=struct_json,
                    output_path=str(out_p),
                    style=eff_style,
                    view_mode=view_mode_eff,
                    view_preset=cfg.view_preset,
                    view_file=view_file_eff,
                    fit_view=fit_eff,
                    replicate=(1, 1, 1),
                    wrap=cfg.wrap,
                    crop_z_mode=cfg.crop_z_mode,
                    crop_z_custom=cfg.crop_z_custom,
                    bonds=cfg.bonds,
                    bond_color=cfg.bond_color,
                    radius_scale=cfg.radius_scale,
                    show_cell=cfg.show_cell,
                    resolution=cfg.resolution,
                    samples=cfg.samples,
                    blender_exe=cfg.blender_exe,
                )
                args = build_render_cli_args(c, out_png=out_p)
                self._queue.put(("log", f"$ atomic-blender {' '.join(args)}"))
                proc = self._popen_cli(args)
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    self._queue.put(("log", raw_line))
                rc = proc.wait()
                if rc != 0:
                    self._queue.put(("done_strip", (rc, strip_png, f"Frame {fi + 1} exited with status {rc}")))
                    return
                rendered_imgs.append(Image.open(out_p).convert("RGBA"))
                titles.append(f"Frame {fi + 1}")
            try:
                strip_img = compose_multi_panel_strip(rendered_imgs, titles=titles, dark_backdrop=dark_bg)
                strip_img.save(strip_png)
                self._queue.put(("done_strip", (0, strip_png, "")))
            except Exception as exc:
                self._queue.put(("done_strip", (1, strip_png, str(exc))))

        threading.Thread(target=worker, daemon=True).start()

    def _on_create_desktop_shortcut(self) -> None:
        try:
            created = install_desktop_shortcut(desktop=True, start_menu=True)
            names = "\n".join(str(p) for p in created)
            self.var_status.set(f"Created {len(created)} shortcut(s) for Atomic Blender Studio")
            messagebox.showinfo("Desktop Shortcut Created", f"Created shortcut(s):\n\n{names}", parent=self)
        except Exception as exc:
            messagebox.showerror("Desktop Shortcut Failed", str(exc), parent=self)

    def export_figure_recipe(self) -> dict[str, Any]:
        """Export current visual styling, representation, slicing, overlays, and camera as a recipe dict."""
        recipe: dict[str, Any] = {
            "recipe_version": 1,
            "style": self.var_style.get(),
            "material_finish": self.var_material_finish.get(),
            "color_map": self.var_color_map.get(),
            "bonds": self.var_bonds.get(),
            "bond_color": self.var_bond_color.get(),
            "radius_scale": float(self.var_radius_scale.get()),
            "show_cell": bool(self.var_show_cell.get()),
            "show_cell_vectors": bool(self.var_show_cell_vectors.get()),
            "show_polyhedra": bool(self.var_show_polyhedra.get()),
            "show_labels": bool(self.var_show_labels.get()),
            "show_measure_overlay": bool(self.var_show_measure_overlay.get()),
            "show_axes": bool(self.var_show_axes.get()),
            "axes_labels": bool(self.var_axes_labels.get()),
            "show_scale_bar": bool(self.var_show_scale_bar.get()),
            "show_hbonds": bool(self.var_show_hbonds.get()),
            "hbond_mode": self.var_hbond_mode.get(),
            "hbond_min": float(self.var_hbond_min.get()),
            "hbond_max": float(self.var_hbond_max.get()),
            "slice_mode": self.var_slice_mode.get(),
            "slice_depth": float(self.var_slice_depth.get()),
            "slice_ghost": bool(self.var_slice_ghost.get()),
            "dark_backdrop": bool(self.var_dark_backdrop.get()),
            "custom_element_colors": dict(self._custom_element_colors),
            "resolution": [int(self.var_res_w.get()), int(self.var_res_h.get())],
            "samples": int(self.var_samples.get()),
        }
        if self._vp_scene is not None:
            recipe["camera"] = {
                "quat": [float(v) for v in self._vp_scene.quat],
                "ortho_scale": float(self._vp_scene.ortho_scale),
                "target": [float(v) for v in self._vp_scene.target],
            }
        return recipe

    def apply_figure_recipe(self, recipe: dict[str, Any], trigger_preview: bool = True) -> None:
        """Apply a saved figure recipe dict to all matching GUI visualization controls."""
        str_map = (
            ("style", self.var_style),
            ("material_finish", self.var_material_finish),
            ("color_map", self.var_color_map),
            ("bonds", self.var_bonds),
            ("bond_color", self.var_bond_color),
            ("hbond_mode", self.var_hbond_mode),
            ("slice_mode", self.var_slice_mode),
        )
        for key, var in str_map:
            if key in recipe:
                var.set(str(recipe[key]))
        bool_map = (
            ("show_cell", self.var_show_cell),
            ("show_cell_vectors", self.var_show_cell_vectors),
            ("show_polyhedra", self.var_show_polyhedra),
            ("show_labels", self.var_show_labels),
            ("show_measure_overlay", self.var_show_measure_overlay),
            ("show_axes", self.var_show_axes),
            ("axes_labels", self.var_axes_labels),
            ("show_scale_bar", self.var_show_scale_bar),
            ("show_hbonds", self.var_show_hbonds),
            ("slice_ghost", self.var_slice_ghost),
            ("dark_backdrop", self.var_dark_backdrop),
        )
        for key, var in bool_map:
            if key in recipe:
                var.set(bool(recipe[key]))
        if "radius_scale" in recipe:
            rs = float(recipe["radius_scale"])
            self.var_radius_scale.set(rs)
            self.var_radius_label.set(f"{rs:.2f}×")
        if "hbond_min" in recipe:
            self.var_hbond_min.set(float(recipe["hbond_min"]))
        if "hbond_max" in recipe:
            self.var_hbond_max.set(float(recipe["hbond_max"]))
        if "slice_depth" in recipe:
            self.var_slice_depth.set(float(recipe["slice_depth"]))
        if isinstance(recipe.get("custom_element_colors"), dict):
            self._custom_element_colors = {str(k): str(v) for k, v in recipe["custom_element_colors"].items()}
        if isinstance(recipe.get("resolution"), list) and len(recipe["resolution"]) == 2:
            self.var_res_w.set(int(recipe["resolution"][0]))
            self.var_res_h.set(int(recipe["resolution"][1]))
        if "samples" in recipe:
            self.var_samples.set(int(recipe["samples"]))
        cam = recipe.get("camera")
        if isinstance(cam, dict) and self._vp_scene is not None:
            if len(cam.get("quat", [])) == 4:
                self._vp_scene.quat = tuple(float(v) for v in cam["quat"])  # type: ignore[assignment]
            if "ortho_scale" in cam:
                self._vp_scene.ortho_scale = float(cam["ortho_scale"])
            if len(cam.get("target", [])) == 3:
                self._vp_scene.target = np.asarray(cam["target"], dtype=np.float64)
            self._use_interactive_cam = True
        self._refresh_display_checks()
        self.var_status.set("Applied figure recipe")
        if self._structure_cache is not None:
            self._rebuild_viewport(reset_camera=False, trigger_preview=trigger_preview)

    def _on_save_recipe(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Figure Recipe JSON",
            defaultextension=".json",
            initialfile="figure_recipe.json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.export_figure_recipe(), indent=2), encoding="utf-8")
        self.var_status.set(f"Saved recipe -> {Path(path).name}")

    def _on_load_recipe(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Figure Recipe JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        self.apply_figure_recipe(data, trigger_preview=True)

    def _refresh_turnkey_structure_annotations(self, style: dict[str, Any]) -> tuple[str, float, float] | None:
        """Compute active Z-height colormap and 3D dashed contact pairs on ``_structure_cache``."""
        if self._structure_cache is None:
            return None
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        n = len(syms)
        ghost_start = self._structure_cache.get("ghost_start_index")
        gs_int = int(ghost_start) if isinstance(ghost_start, int) and 0 < ghost_start < n else None
        ghost_tint = str(self._structure_cache.get("ghost_tint", "#38BDF8"))

        cmap = self.var_color_map.get() if hasattr(self, "var_color_map") else "none"
        cmap_info: tuple[str, float, float] | None = None
        prev_colors = self._structure_cache.get("colors")
        if cmap in ("viridis", "coolwarm", "plasma") and n > 0 and len(pos) == n:
            main_count = gs_int if gs_int is not None else n
            cols_main, z_lo, z_hi = compute_z_colormap_colors(pos[:main_count], cmap)
            new_cols = cols_main + ([ghost_tint] * (n - main_count) if gs_int is not None else [])
            if prev_colors != new_cols:
                self._structure_cache["colors"] = new_cols
                self._written_struct_version = -1
            cmap_info = (cmap, z_lo, z_hi)
        elif gs_int is not None:
            from atomic_blender.elements import resolve_element_color

            new_cols = [resolve_element_color(syms[i], style) for i in range(gs_int)] + [ghost_tint] * (n - gs_int)
            if prev_colors != new_cols:
                self._structure_cache["colors"] = new_cols
                self._written_struct_version = -1
        elif "colors" in self._structure_cache:
            self._structure_cache.pop("colors", None)
            self._written_struct_version = -1

        dashed: list[list[int]] = []
        if gs_int is not None and 2 * gs_int == n and len(pos) == n:
            mp = np.asarray(pos[:gs_int], dtype=np.float64)
            gp = np.asarray(pos[gs_int:], dtype=np.float64)
            for i in range(gs_int):
                if float(np.linalg.norm(mp[i] - gp[i])) >= 0.20:
                    dashed.append([i, gs_int + i])

        if hasattr(self, "var_show_hbonds") and self.var_show_hbonds.get() and n > 1 and len(pos) == n:
            try:
                d_lo = float(self.var_hbond_min.get())
                d_hi = float(self.var_hbond_max.get())
            except Exception:
                d_lo, d_hi = 1.55, 2.55
            hb_mode = self.var_hbond_mode.get()
            c_key = (
                self._structure_version,
                self.var_bonds.get(),
                hb_mode,
                round(d_lo, 4),
                round(d_hi, 4),
                frozenset(self._selected_atoms) if hb_mode == "interface" else None,
                gs_int,
            )
            if c_key == self._contact_cache_key:
                contacts = self._contact_cache_pairs
            else:
                from atomic_blender.viewport3d import _find_bonds

                bond_cfg = dict(style.get("bonds", {}))
                if self.var_bonds.get() != "style":
                    bond_cfg["mode"] = self.var_bonds.get()
                    bond_cfg["enabled"] = self.var_bonds.get() != "none"
                pos_arr = np.asarray(pos, dtype=np.float64)
                bi, bj = _find_bonds(syms, pos_arr, bond_cfg, ghost_start_index=gs_int)
                contacts = find_dashed_contacts(
                    syms,
                    pos_arr,
                    bond_i=bi,
                    bond_j=bj,
                    mode=hb_mode,
                    d_min=d_lo,
                    d_max=d_hi,
                    focal_indices=self._selected_atoms or None,
                    ghost_start_index=gs_int,
                )
                self._contact_cache_key = c_key
                self._contact_cache_pairs = list(contacts)
            seen_d = {(min(p[0], p[1]), max(p[0], p[1])) for p in dashed}
            for ia, ib in contacts:
                key = (min(ia, ib), max(ia, ib))
                if key not in seen_d:
                    seen_d.add(key)
                    dashed.append([ia, ib])

        prev_dashed = self._structure_cache.get("dashed_bonds", [])
        if prev_dashed != dashed:
            self._structure_cache["dashed_bonds"] = dashed
            self._written_struct_version = -1

        main_n = gs_int if gs_int is not None else n
        if (
            hasattr(self, "var_show_polyhedra")
            and self.var_show_polyhedra.get()
            and main_n >= 5
            and len(pos) >= main_n
        ):
            polys = find_coordination_polyhedra(
                syms[:main_n],
                pos[:main_n],
                style=style,
                cell_vecs=self._structure_cache.get("cell") if self._structure_cache.get("has_cell") else None,
                pbc=self._structure_cache.get("pbc"),
            )
            if self._structure_cache.get("polyhedra") != polys:
                self._structure_cache["polyhedra"] = polys
                self._written_struct_version = -1
        elif "polyhedra" in self._structure_cache:
            self._structure_cache.pop("polyhedra", None)
            self._written_struct_version = -1

        show_cvec = bool(getattr(self, "var_show_cell_vectors", None) and self.var_show_cell_vectors.get())
        if bool(self._structure_cache.get("show_cell_vectors", False)) != show_cvec:
            if show_cvec:
                self._structure_cache["show_cell_vectors"] = True
            else:
                self._structure_cache.pop("show_cell_vectors", None)
            self._written_struct_version = -1
        return cmap_info

    def _refresh_element_chip_styles(self) -> None:
        if not getattr(self, "_elem_chip_buttons", None) or self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        for sym, btn in self._elem_chip_buttons.items():
            indices = [i for i, s in enumerate(syms) if s == sym]
            all_sel = bool(indices) and all(i in self._selected_atoms for i in indices)
            any_sel = any(i in self._selected_atoms for i in indices)
            if all_sel:
                bg, fg, active = "#0D9488", "#FFFFFF", "#14B8A6"
            elif any_sel:
                bg, fg, active = "#1B3B3A", "#5EEAD4", "#244D4B"
            else:
                bg, fg, active = "#222631", "#CBD5E1", "#2D3342"
            btn.configure(bg=bg, fg=fg, activebackground=active, activeforeground="#FFFFFF")

    def _toggle_element_selection(self, symbol: str) -> None:
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        indices = [i for i, s in enumerate(syms) if s == symbol]
        if not indices:
            return
        if all(i in self._selected_atoms for i in indices):
            for i in indices:
                self._selected_atoms.discard(i)
            idx_set = set(indices)
            self._selection_order = [i for i in self._selection_order if i not in idx_set]
        else:
            for i in indices:
                self._selected_atoms.add(i)
                if i not in self._selection_order:
                    self._selection_order.append(i)
        self._sync_selection_and_alphas(trigger_blender=True)

    def _select_by_z_range(self) -> None:
        if self._structure_cache is None:
            return
        pos = self._structure_cache.get("positions", [])
        if not pos:
            return
        zs = [float(p[2]) for p in pos]
        raw_min = self.var_z_min.get().strip()
        raw_max = self.var_z_max.get().strip()
        try:
            z_lo = float(raw_min) if raw_min else min(zs)
            z_hi = float(raw_max) if raw_max else max(zs)
        except ValueError:
            self.var_status.set("Invalid Z range number")
            return
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo
        picked = [i for i, p in enumerate(pos) if z_lo - 1e-6 <= float(p[2]) <= z_hi + 1e-6]
        self._selected_atoms = set(picked)
        self._selection_order = list(picked)
        self.var_status.set(f"Selected {len(picked)} atoms in Z=[{z_lo:.2f}, {z_hi:.2f}] Å")
        self._sync_selection_and_alphas(trigger_blender=True)

    def _on_atom_list_click(self, event: tk.Event) -> None:
        if self._structure_cache is None or not hasattr(self, "atom_listbox"):
            return
        idx = int(self.atom_listbox.nearest(event.y))
        n = len(self._structure_cache.get("symbols", []))
        if idx < 0 or idx >= n:
            return
        self._toggle_atom_selection(idx)
        self._sync_selection_and_alphas(trigger_blender=True)

    def _refresh_atom_list(self) -> None:
        if not hasattr(self, "atom_listbox"):
            return
        top_frac = 0.0
        try:
            top_frac = float(self.atom_listbox.yview()[0])
        except Exception:
            pass
        self.atom_listbox.delete(0, tk.END)
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        alphas = self._structure_cache.get("alphas", [1.0] * len(syms))
        for idx, sym in enumerate(syms):
            mark = "●" if idx in self._selected_atoms else "○"
            lbl = atom_label(idx, sym)
            x, y, z = pos[idx] if idx < len(pos) else (0.0, 0.0, 0.0)
            a_pct = int(round((alphas[idx] if idx < len(alphas) else 1.0) * 100.0))
            line = f"{mark} {lbl:<5} ({x:6.2f},{y:6.2f},{z:6.2f})  {a_pct:3d}%"
            self.atom_listbox.insert(tk.END, line)
            if idx in self._selected_atoms:
                self.atom_listbox.itemconfig(idx, fg="#2DD4BF", bg="#162A2C")
            else:
                self.atom_listbox.itemconfig(idx, fg="#E2E8F0", bg="#111318")
        self.atom_listbox.yview_moveto(top_frac)

    def _sync_selection_and_alphas(
        self,
        trigger_blender: bool = False,
        alphas_override: list[float] | None = None,
    ) -> None:
        if self._structure_cache is None:
            self._refresh_atom_list()
            self._refresh_element_chip_styles()
            self._update_measurement_banner()
            return
        syms = self._structure_cache.get("symbols", [])
        n = len(syms)
        prev_alphas = self._structure_cache.get("alphas")
        prev_r_scales = self._structure_cache.get("radius_scales")
        if alphas_override is not None and len(alphas_override) == n:
            alphas = [max(0.02, min(1.0, float(a))) for a in alphas_override]
        else:
            alphas = compute_atom_alphas(
                n_atoms=n,
                selected=self._selected_atoms,
                opacity=float(self.var_sel_opacity.get()),
                invert=bool(self.var_alpha_invert.get()),
            )
        alphas = self._apply_slice_mask_to_alphas(alphas)
        scale_mode = self.var_unsel_scale_mode.get() if hasattr(self, "var_unsel_scale_mode") else "same"
        if scale_mode == "slim":
            sel_s, unsel_s = 1.12, 0.55
        elif scale_mode == "vdw":
            sel_s, unsel_s = 0.88, 1.35
        else:
            sel_s, unsel_s = 1.0, 1.0
        r_scales = compute_atom_radius_scales(n, self._selected_atoms, sel_scale=sel_s, unsel_scale=unsel_s)
        alphas_changed = prev_alphas != alphas
        scales_changed = prev_r_scales != r_scales
        self._structure_cache["alphas"] = alphas
        self._structure_cache["radius_scales"] = r_scales
        if alphas_changed or scales_changed:
            self._written_struct_version = -1

        if scales_changed:
            self._rebuild_viewport(reset_camera=False, trigger_preview=False)
        elif self._vp_scene is not None:
            self._vp_scene.atom_alphas = np.asarray(alphas, dtype=np.float64)
            self._vp_scene.selected_atoms = set(self._selected_atoms)
            self._vp_scene.show_measure_overlay = bool(self.var_show_measure_overlay.get())
            self._vp_scene.show_scale_bar = bool(self.var_show_scale_bar.get())
        self._refresh_atom_list()
        self._refresh_element_chip_styles()
        self._update_measurement_banner()
        self._redraw_canvas()
        if trigger_blender and (
            alphas_changed
            or scales_changed
            or any(a < 0.995 for a in alphas)
            or any(abs(s - 1.0) > 1e-3 for s in r_scales)
        ):
            self._request_blender_preview()

    def _on_canvas_double_click(self, event: tk.Event) -> None:
        self._dragging = False
        self._drag_mode = None
        if self._vp_scene is None:
            return
        rect = self._frame_rect()
        if rect is None:
            return
        x0, y0, fw, fh = rect
        u = (event.x - x0) / max(1.0, fw)
        v = (event.y - y0) / max(1.0, fh)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        picked = self._vp_scene.pick_atom_at(u, v, res_w, res_h, fw, fh)
        if picked is None:
            return
        self._toggle_atom_selection(picked)
        syms = self._structure_cache.get("symbols", []) if self._structure_cache else []
        lbl = atom_label(picked, syms[picked]) if picked < len(syms) else str(picked + 1)
        self._sync_selection_and_alphas(trigger_blender=True)
        meas = self.var_measurement.get() if len(self._selected_atoms) in (1, 2, 3, 4) else f"{len(self._selected_atoms)} selected"
        self.var_status.set(f"{lbl} toggled | {meas}")

    def _on_repr_changed(self) -> None:
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _set_representation(self, radius_scale: float, bonds: str, bond_color: str) -> None:
        self.var_radius_scale.set(radius_scale)
        self.var_radius_label.set(f"{radius_scale:.2f}×")
        self.var_bonds.set(bonds)
        self.var_bond_color.set(bond_color)
        self._show_atoms = True
        self.var_disp_ball_stick.set(bonds != "none" and radius_scale >= 0.6)
        self.var_disp_licorice.set(bonds != "none" and radius_scale < 0.6)
        self.var_disp_vdw.set(bonds == "none" and radius_scale > 1.1)
        self._refresh_display_checks()
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _apply_slab_preset(self) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set("side-oblique")
        self.var_crop_z_mode.set("auto")
        self._sync_crop_z(trigger_preview=False)
        self._set_resolution(1400, 1200, trigger_preview=False)
        self._use_interactive_cam = False
        self._rebuild_viewport(reset_camera=True, trigger_preview=True)

    def _apply_bulk_preset(self) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set("iso")
        self.var_crop_z_mode.set("none")
        self._sync_crop_z(trigger_preview=False)
        self._set_resolution(1400, 1400, trigger_preview=False)
        self._use_interactive_cam = False
        self._rebuild_viewport(reset_camera=True, trigger_preview=True)

    def _pick_view_preset(self, preset: str) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set(preset)
        self._use_interactive_cam = False
        if self._vp_scene is not None:
            self._apply_view_to_viewport(self._vp_scene)
            self.var_canvas_mode.set("3d")
            self._redraw_canvas()
            self._request_blender_preview()

    def _sync_view_mode(self) -> None:
        is_file = self.var_view_mode.get() == "file"
        self.combo_preset.configure(state="disabled" if is_file else "readonly")
        state_file = "normal" if is_file else "disabled"
        self.entry_view_file.configure(state=state_file)
        self.btn_view_file.configure(state=state_file)
        self.chk_fit_view.configure(state=state_file)

    def _sync_crop_z(self, trigger_preview: bool = True) -> None:
        is_custom = self.var_crop_z_mode.get() == "custom"
        self.entry_crop_z.configure(state="normal" if is_custom else "disabled")
        self._rebuild_viewport(reset_camera=not self._use_interactive_cam, trigger_preview=trigger_preview)

    def _set_resolution(self, w: int, h: int, trigger_preview: bool = True) -> None:
        self.var_res_w.set(w)
        self.var_res_h.set(h)
        if self._vp_scene is not None and not self._use_interactive_cam:
            self._apply_view_to_viewport(self._vp_scene)
        self._redraw_canvas()
        if trigger_preview:
            self._request_blender_preview()

    def _schedule_inspect(self) -> None:
        self._inspect_seq += 1
        if self._inspect_after_id is not None:
            self.after_cancel(self._inspect_after_id)
        self._inspect_after_id = self.after(150, self._run_inspect_async)

    def _run_inspect_async(self) -> None:
        self._inspect_after_id = None
        self._inspect_seq += 1
        req_seq = self._inspect_seq
        path = extract_input_path(self.var_input.get())
        if not path:
            self.var_inspector.set("")
            return
        type_map = self.var_type_map.get()
        rep = (int(self.var_rep_x.get()), int(self.var_rep_y.get()), int(self.var_rep_z.get()))
        wrap = bool(self.var_wrap.get())

        def worker() -> None:
            try:
                p = Path(path).expanduser()
                if not p.is_file():
                    self._queue.put(("inspect", (req_seq, path, f"File not found: {p.name}")))
                    return
                from atomic_blender.convert import read_structure_frames

                frames = read_structure_frames(
                    p,
                    type_map=type_map.strip() or None,
                    replicate=rep,
                    wrap=wrap,
                )
                eff_rep = rep if (frames and frames[0].get("has_cell", False)) else (1, 1, 1)
                summary = inspect_structure_summary(path, type_map=type_map, replicate=eff_rep)
                if len(frames) > 1:
                    summary = f"{summary} | {len(frames)} frames"
                self._queue.put(("structure_frames", (req_seq, path, summary, frames)))
            except Exception as exc:
                self._queue.put(("inspect", (req_seq, path, f"Inspect error: {exc}")))

        threading.Thread(target=worker, daemon=True).start()

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Atomic Structure",
            filetypes=[
                ("Structure files", "*.cif *.vasp *.xyz *.extxyz *.data *.lmp *.json POSCAR CONTCAR"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.var_input.set(path)
        if not self.var_output.get().strip():
            self.var_output.set(str(Path(path).with_suffix(".png")))
        self._schedule_inspect()

    def _browse_output(self) -> None:
        initial = self.var_output.get().strip() or "render.png"
        path = filedialog.asksaveasfilename(
            title="Save Output PNG",
            defaultextension=".png",
            initialfile=Path(initial).name,
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            self.var_output.set(path)

    def _browse_style(self) -> None:
        path = filedialog.askopenfilename(title="Select Style JSON", filetypes=[("JSON files", "*.json")])
        if path:
            self.var_style.set(path)
            self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _browse_view_file(self) -> None:
        path = filedialog.askopenfilename(title="Select view.json", filetypes=[("JSON files", "*.json")])
        if path:
            self.var_view_file.set(path)
            self._use_interactive_cam = False
            if self._vp_scene is not None:
                self._apply_view_to_viewport(self._vp_scene)
                self.var_canvas_mode.set("3d")
                self._redraw_canvas()
                self._request_blender_preview()

    def _browse_blender(self) -> None:
        path = filedialog.askopenfilename(title="Select Blender Executable")
        if path:
            self.var_blender.set(path)

    def _collect_config(self) -> GuiRenderConfig:
        rep = (int(self.var_rep_x.get()), int(self.var_rep_y.get()), int(self.var_rep_z.get()))
        if self._structure_cache is not None and not self._structure_cache.get("has_cell", False):
            rep = (1, 1, 1)
        return GuiRenderConfig(
            input_path=extract_input_path(self.var_input.get()),
            output_path=self.var_output.get(),
            style=self.var_style.get(),
            view_mode=self.var_view_mode.get(),
            view_preset=self.var_view_preset.get(),
            view_file=self.var_view_file.get(),
            fit_view=bool(self.var_fit_view.get()),
            type_map=self.var_type_map.get(),
            replicate=rep,
            wrap=bool(self.var_wrap.get()),
            crop_z_mode=self.var_crop_z_mode.get(),
            crop_z_custom=self.var_crop_z_custom.get(),
            bonds=self.var_bonds.get(),
            bond_color=self.var_bond_color.get(),
            radius_scale=float(self.var_radius_scale.get()),
            show_cell=bool(self.var_show_cell.get()),
            resolution=(int(self.var_res_w.get()), int(self.var_res_h.get())),
            samples=int(self.var_samples.get()),
            blender_exe=self.var_blender.get(),
        )

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status_text: str) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.btn_render.configure(state=state)
        self.btn_blender.configure(state=state)
        self.var_status.set(status_text)

    def _popen_cli(self, cli_args: list[str]) -> subprocess.Popen[str]:
        """Launch ``atomic_blender.cli`` cleanly in Python or frozen .exe mode without console flicker."""
        cmd = [sys.executable, "-m", "atomic_blender.cli", *cli_args]
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            kwargs["creationflags"] = flags
        return subprocess.Popen(cmd, **kwargs)

    def _run_cli_async(self, cli_args: list[str], mode: str, target_path: Path) -> None:
        self._task_start = time.monotonic()
        self._set_busy(True, f"Running {mode}...")
        self._append_log(f"$ atomic-blender {' '.join(cli_args)}")

        def worker() -> None:
            proc = self._popen_cli(cli_args)
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                self._queue.put(("log", raw_line))
            rc = proc.wait()
            self._queue.put(("done", (mode, rc, target_path)))

        threading.Thread(target=worker, daemon=True).start()

    def _effective_style_path_for_cli(self, style_arg: str) -> str:
        """Return ``style_arg`` or materialize a temporary style JSON with custom element colors/finish."""
        has_finish = hasattr(self, "var_material_finish") and self.var_material_finish.get() != "studio"
        if not self._custom_element_colors and not has_finish:
            return style_arg
        merged = self._resolve_style_dict()
        tmp_fd, tmp_name = tempfile.mkstemp(prefix="atomic_style_", suffix=".json")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        self._temp_view_files.append(tmp_path)
        return str(tmp_path)

    def _effective_cfg_for_cli(self, cfg: GuiRenderConfig) -> GuiRenderConfig:
        """Materialize interactive camera view.json, per-atom transparency/scales/colors, and style."""
        eff_input = cfg.input_path
        eff_rep = cfg.replicate
        eff_style = self._effective_style_path_for_cli(cfg.style)
        if self._structure_cache is not None:
            alphas = self._structure_cache.get("alphas")
            r_scales = self._structure_cache.get("radius_scales")
            has_alpha = isinstance(alphas, list) and any(a < 0.995 for a in alphas)
            has_scale = isinstance(r_scales, list) and any(abs(s - 1.0) > 1e-3 for s in r_scales)
            has_cols = bool(self._structure_cache.get("colors"))
            has_dashed = bool(self._structure_cache.get("dashed_bonds"))
            has_ghost = self._structure_cache.get("ghost_start_index") is not None
            has_poly = bool(self._structure_cache.get("polyhedra"))
            has_cvec = bool(self._structure_cache.get("show_cell_vectors"))
            if has_alpha or has_scale or has_cols or has_dashed or has_ghost or has_poly or has_cvec:
                tmp_sfd, tmp_sname = tempfile.mkstemp(prefix="atomic_sel_", suffix=".json")
                os.close(tmp_sfd)
                tmp_spath = Path(tmp_sname)
                tmp_spath.write_text(json.dumps(self._structure_cache), encoding="utf-8")
                self._temp_view_files.append(tmp_spath)
                eff_input = str(tmp_spath)
                eff_rep = (1, 1, 1)

        if self._vp_scene is None or not self._use_interactive_cam:
            if eff_input == cfg.input_path and eff_style == cfg.style:
                return cfg
            return GuiRenderConfig(
                input_path=eff_input,
                output_path=cfg.output_path,
                style=eff_style,
                view_mode=cfg.view_mode,
                view_preset=cfg.view_preset,
                view_file=cfg.view_file,
                fit_view=cfg.fit_view,
                type_map=cfg.type_map,
                replicate=eff_rep,
                wrap=cfg.wrap,
                crop_z_mode=cfg.crop_z_mode,
                crop_z_custom=cfg.crop_z_custom,
                bonds=cfg.bonds,
                bond_color=cfg.bond_color,
                radius_scale=cfg.radius_scale,
                show_cell=cfg.show_cell,
                resolution=cfg.resolution,
                samples=cfg.samples,
                blender_exe=cfg.blender_exe,
                save_blend=cfg.save_blend,
            )
        view_dict = self._vp_scene.to_view_dict(cfg.resolution[0], cfg.resolution[1])
        tmp_fd, tmp_name = tempfile.mkstemp(prefix="atomic_vp_", suffix=".json")
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_text(json.dumps(view_dict, indent=2), encoding="utf-8")
        self._temp_view_files.append(tmp_path)
        return GuiRenderConfig(
            input_path=eff_input,
            output_path=cfg.output_path,
            style=eff_style,
            view_mode="file",
            view_preset=cfg.view_preset,
            view_file=str(tmp_path),
            fit_view=False,
            type_map=cfg.type_map,
            replicate=eff_rep,
            wrap=cfg.wrap,
            crop_z_mode=cfg.crop_z_mode,
            crop_z_custom=cfg.crop_z_custom,
            bonds=cfg.bonds,
            bond_color=cfg.bond_color,
            radius_scale=cfg.radius_scale,
            show_cell=cfg.show_cell,
            resolution=cfg.resolution,
            samples=cfg.samples,
            blender_exe=cfg.blender_exe,
            save_blend=cfg.save_blend,
        )

    def _on_render(self) -> None:
        if self._busy:
            return
        if not self._ensure_blender_available():
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        out_png = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        self.var_output.set(str(out_png))
        cli_args = build_render_cli_args(cli_cfg, out_png=out_png)
        self._run_cli_async(cli_args, mode="render", target_path=out_png)

    def _on_open_in_blender(self) -> None:
        if self._busy:
            return
        if not self._ensure_blender_available():
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            blender_exe = find_blender_executable(cfg.blender_exe.strip() or None)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            messagebox.showerror("Cannot Open in Blender", str(exc))
            return

        base = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path
        blend_path = base.with_suffix(".blend")
        cli_args = build_render_cli_args(cli_cfg, out_png=None, save_blend=blend_path)
        self._pending_blender_exe = str(blender_exe)
        self._run_cli_async(cli_args, mode="open_blender", target_path=blend_path)

    def _on_open_folder(self) -> None:
        target = self.var_output.get().strip() or extract_input_path(self.var_input.get())
        if not target:
            messagebox.showinfo("Open Folder", "No input or output path selected yet.")
            return
        folder = Path(target).expanduser().resolve().parent
        if not folder.is_dir():
            messagebox.showwarning("Open Folder", f"Directory does not exist yet: {folder}")
            return
        _open_folder(folder)

    def _load_preview(self, png_path: Path) -> None:
        if not png_path.is_file():
            return
        self._preview_path = png_path
        if _HAS_PIL:
            self._pil_image = Image.open(png_path).convert("RGBA")
        else:
            self._pil_image = tk.PhotoImage(file=str(png_path))
        self.var_canvas_mode.set("png")
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        if self.var_canvas_mode.get() == "3d":
            self._draw_3d_viewport()
        else:
            self._redraw_preview()

    def _parse_crop_z_tuple(self) -> tuple[float, float] | None:
        mode = self.var_crop_z_mode.get()
        if mode == "none":
            return None
        if mode == "auto":
            return (2.0, 4.0)
        parts = [float(x.strip()) for x in self.var_crop_z_custom.get().split(",") if x.strip()]
        if len(parts) == 1:
            return (parts[0], parts[0])
        if len(parts) == 2:
            return (parts[0], parts[1])
        return None

    def _resolve_style_dict(self) -> dict[str, Any]:
        try:
            sp = resolve_style_path(self.var_style.get().strip() or "default")
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if self._custom_element_colors:
            elems = data.setdefault("elements", {})
            cols = elems.setdefault("colors", {})
            cols.update(self._custom_element_colors)
        fin = self.var_material_finish.get() if hasattr(self, "var_material_finish") else "studio"
        fin_overrides = MATERIAL_FINISH_PRESETS.get(fin)
        if fin_overrides:
            mat = data.setdefault("material", {})
            mat.update(fin_overrides)
            for sub_key in ("metal", "nonmetal"):
                if isinstance(mat.get(sub_key), dict):
                    for k, v in fin_overrides.items():
                        if k != "ao_factor":
                            mat[sub_key][k] = v
        return data

    def _apply_view_to_viewport(self, vp: ViewportScene) -> None:
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        aspect = res_w / res_h
        crop_z = self._parse_crop_z_tuple()

        if self.var_view_mode.get() == "file" and self.var_view_file.get().strip():
            vf = Path(self.var_view_file.get().strip()).expanduser()
            if vf.is_file():
                try:
                    vd = json.loads(vf.read_text(encoding="utf-8"))
                    q = tuple(float(x) for x in vd["rotation_quaternion"])
                    if bool(self.var_fit_view.get()) or bool(vd.get("fit_to_scene", False)):
                        vp.fit_camera(quat=q, aspect=aspect, crop_z=crop_z)
                    else:
                        vp.quat = q  # type: ignore[assignment]
                        vp.ortho_scale = float(vd.get("ortho_scale", vp.ortho_scale))
                        loc = np.asarray(vd["location"], dtype=np.float64)
                        _, _, back = quat_to_axes(vp.quat)
                        # Project structure center onto camera ray for consistent Z/pan origin
                        center = np.mean(vp.positions, axis=0) if len(vp.positions) else np.zeros(3)
                        dist = max(1.0, float(np.dot(loc - center, back)))
                        vp.target = loc - back * dist
                    return
                except Exception:
                    pass

        cell = vp.cell_vecs if vp.has_cell else None
        q = preset_to_quat(self.var_view_preset.get(), cell)
        vp.fit_camera(quat=q, aspect=aspect, crop_z=crop_z)

    def _rebuild_viewport(self, reset_camera: bool, trigger_preview: bool = False) -> None:
        if self._structure_cache is None:
            return
        prev_cam = None
        if not reset_camera and self._vp_scene is not None and self._use_interactive_cam:
            prev_cam = (self._vp_scene.quat, self._vp_scene.target.copy(), self._vp_scene.ortho_scale)

        style = self._resolve_style_dict()
        cmap_info = self._refresh_turnkey_structure_annotations(style)
        self._active_colormap_info = cmap_info
        if hasattr(self, "var_slice_mode") and self.var_slice_mode.get() != "none":
            n_cur = len(self._structure_cache.get("symbols", []))
            raw_a = self._structure_cache.get("alphas")
            base_a = (
                [float(x) for x in raw_a]
                if isinstance(raw_a, list) and len(raw_a) == n_cur
                else [1.0] * n_cur
            )
            sliced_a = self._apply_slice_mask_to_alphas(base_a)
            if sliced_a != raw_a:
                self._structure_cache["alphas"] = sliced_a
                self._written_struct_version = -1
        vp = ViewportScene.from_neutral_dict(
            self._structure_cache,
            style=style,
            radius_scale=float(self.var_radius_scale.get()),
            bonds_mode=self.var_bonds.get(),
            bond_color=self.var_bond_color.get(),
            show_cell=bool(self.var_show_cell.get()),
            show_atoms=bool(self._show_atoms),
            show_labels=bool(self.var_show_labels.get()),
            show_axes=bool(self.var_show_axes.get()),
            axes_labels=bool(self.var_axes_labels.get()),
            axes_pos=self._axes_pos,
            dark_backdrop=bool(self.var_dark_backdrop.get()),
            selected_atoms=self._selected_atoms,
            show_scale_bar=bool(self.var_show_scale_bar.get()),
            show_measure_overlay=bool(self.var_show_measure_overlay.get()),
            selection_order=list(self._selection_order),
            colormap_info=cmap_info,
            show_cell_vectors=bool(getattr(self, "var_show_cell_vectors", None) and self.var_show_cell_vectors.get()),
            show_output_mask=bool(getattr(self, "var_show_output_mask", None) and self.var_show_output_mask.get()),
        )
        self._refresh_display_checks()
        self._rebuild_element_chips()
        self._refresh_atom_list()
        self._update_measurement_banner()
        if reset_camera and self._structure_cache is not None:
            pos = self._structure_cache.get("positions", [])
            if pos:
                zs = [float(p[2]) for p in pos]
                self.var_z_min.set(f"{min(zs):.2f}")
                self.var_z_max.set(f"{max(zs):.2f}")
        if prev_cam is not None:
            vp.quat, vp.target, vp.ortho_scale = prev_cam
        else:
            self._apply_view_to_viewport(vp)
        self._vp_scene = vp

        if self._ghost_structure_cache is not None and self.var_compare_mode.get() == "side_by_side":
            gvp = ViewportScene.from_neutral_dict(
                self._ghost_structure_cache,
                style=style,
                radius_scale=float(self.var_radius_scale.get()),
                bonds_mode=self.var_bonds.get(),
                bond_color=self.var_bond_color.get(),
                show_cell=bool(self.var_show_cell.get()),
                show_atoms=bool(self._show_atoms),
                show_labels=bool(self.var_show_labels.get()),
                show_axes=bool(self.var_show_axes.get()),
                axes_labels=bool(self.var_axes_labels.get()),
                axes_pos=self._axes_pos,
                dark_backdrop=bool(self.var_dark_backdrop.get()),
                show_scale_bar=bool(self.var_show_scale_bar.get()),
                show_cell_vectors=bool(getattr(self, "var_show_cell_vectors", None) and self.var_show_cell_vectors.get()),
                show_output_mask=bool(getattr(self, "var_show_output_mask", None) and self.var_show_output_mask.get()),
            )
            gvp.quat = vp.quat
            gvp.target = vp.target.copy()
            gvp.ortho_scale = vp.ortho_scale
            self._ghost_vp_scene = gvp
        else:
            self._ghost_vp_scene = None
        if reset_camera:
            self.var_canvas_mode.set("3d")
        self._redraw_canvas()
        if trigger_preview:
            self._request_blender_preview()

    def _request_blender_preview(self) -> None:
        """Request a fast low-res Blender Cycles preview of the current viewport state."""
        if self._structure_cache is None or self._vp_scene is None:
            return
        blender_raw = self.var_blender.get().strip()
        if not blender_raw:
            try:
                blender_raw = str(find_blender_executable())
                self.var_blender.set(blender_raw)
            except FileNotFoundError:
                return
        blender_path = Path(blender_raw).expanduser()
        if not blender_path.is_file():
            return

        struct_json = Path(self._preview_tmp_dir.name) / "preview_structure.json"
        if self._written_struct_version != self._structure_version or not struct_json.is_file():
            struct_json.write_text(json.dumps(self._structure_cache), encoding="utf-8")
            self._written_struct_version = self._structure_version

        style_path = resolve_style_path(self.var_style.get().strip() or "default")
        style_mtime = style_path.stat().st_mtime if style_path.is_file() else 0.0
        eff_style_path = style_path
        fin_key = self.var_material_finish.get() if hasattr(self, "var_material_finish") else "studio"
        if self._custom_element_colors or fin_key != "studio":
            merged_style = self._resolve_style_dict()
            eff_style_path = Path(self._preview_tmp_dir.name) / "preview_style.json"
            eff_style_path.write_text(json.dumps(merged_style, indent=2), encoding="utf-8")
        prev_w, prev_h = compute_preview_resolution(int(self.var_res_w.get()), int(self.var_res_h.get()), max_side=640)
        prev_samples = min(8, max(4, int(self.var_samples.get())))

        crop_mode = self.var_crop_z_mode.get()
        if crop_mode == "auto":
            crop_z: str | None = "auto"
        elif crop_mode == "custom":
            crop_z = self.var_crop_z_custom.get().strip() or None
        else:
            crop_z = None

        if self._use_interactive_cam:
            view_arg: str | dict[str, Any] = self._vp_scene.to_view_dict(prev_w, prev_h)
        elif self.var_view_mode.get() == "file" and self.var_view_file.get().strip():
            view_arg = str(Path(self.var_view_file.get().strip()).expanduser())
        else:
            view_arg = self.var_view_preset.get().strip() or "iso"

        scene_key = json.dumps(
            [
                self._structure_version,
                str(style_path),
                style_mtime,
                sorted(self._custom_element_colors.items()),
                fin_key,
                self.var_bonds.get(),
                self.var_bond_color.get(),
                round(float(self.var_radius_scale.get()), 4),
                bool(self.var_show_cell.get()),
                crop_z,
                bool(self.var_fit_view.get()),
                [round(float(a), 3) for a in self._structure_cache.get("alphas", [])],
                [round(float(s), 3) for s in self._structure_cache.get("radius_scales", [])],
                list(self._structure_cache.get("colors", [])),
                list(self._structure_cache.get("dashed_bonds", [])),
                self._structure_cache.get("ghost_start_index"),
                len(self._structure_cache.get("polyhedra", [])),
                bool(self._structure_cache.get("show_cell_vectors", False)),
            ]
        )

        self._preview_seq += 1
        seq = self._preview_seq
        out_png = Path(self._preview_tmp_dir.name) / f"preview_{seq % 2}.png"
        if not self._busy:
            self.var_status.set("Rendering fast Blender preview...")
        self._preview_worker.request(
            {
                "action": "render",
                "seq": seq,
                "blender_exe": str(blender_path),
                "scene_key": scene_key,
                "structure": str(struct_json),
                "style": str(eff_style_path),
                "crop_z": crop_z,
                "resolution": [prev_w, prev_h],
                "samples": prev_samples,
                "bonds": None if self.var_bonds.get() == "style" else self.var_bonds.get(),
                "bond_color": None if self.var_bond_color.get() == "style" else self.var_bond_color.get(),
                "radius_scale": float(self.var_radius_scale.get()),
                "no_cell": not bool(self.var_show_cell.get()),
                "fit_view": bool(self.var_fit_view.get()) if self.var_view_mode.get() == "file" else False,
                "view": view_arg,
                "out": str(out_png),
            }
        )

    def _on_reset_view(self) -> None:
        self._use_interactive_cam = False
        if self._vp_scene is not None:
            self._apply_view_to_viewport(self._vp_scene)
            self.var_canvas_mode.set("3d")
            self._redraw_canvas()
            self._request_blender_preview()

    def _on_save_view_json(self) -> None:
        if self._vp_scene is None:
            messagebox.showinfo("Save view.json", "Load or drop a structure first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save view.json",
            defaultextension=".json",
            initialfile="view.json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        payload = self._vp_scene.to_view_dict(res_w, res_h)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.var_view_mode.set("file")
        self.var_view_file.set(path)
        self._sync_view_mode()
        self.var_status.set(f"Saved view -> {Path(path).name}")

    def _frame_rect(self) -> tuple[float, float, float, float] | None:
        """Return ``(x0, y0, width, height)`` of the active camera frame or PNG inside ``self.canvas``."""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if self.var_canvas_mode.get() == "png" and self._pil_image is not None and _HAS_PIL:
            iw, ih = self._pil_image.size
            scale = min((cw - 14) / max(1, iw), (ch - 14) / max(1, ih))
            fw = max(1.0, iw * scale)
            fh = max(1.0, ih * scale)
            return ((cw - fw) * 0.5, (ch - fh) * 0.5, fw, fh)
        if self._vp_scene is None:
            return None
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        pad = min(14.0, min(cw, ch) * 0.12)
        scale = max(1e-4, min((cw - pad) / res_w, (ch - pad) / res_h))
        fw = res_w * scale
        fh = res_h * scale
        return ((cw - fw) * 0.5, (ch - fh) * 0.5, fw, fh)

    def _axes_canvas_center(self) -> tuple[float, float, float] | None:
        """Return ``(cx, cy, hit_radius_px)`` of the Reference Axes gizmo in canvas coords."""
        if not self.var_show_axes.get():
            return None
        rect = self._frame_rect()
        if rect is None:
            return None
        x0, y0, fw, fh = rect
        cx = x0 + self._axes_pos[0] * fw
        cy = y0 + self._axes_pos[1] * fh
        hit_r = max(36.0, min(46.0, min(fw, fh) * 0.09))
        return (cx, cy, hit_r)

    def _on_canvas_lbutton_down(self, event: tk.Event) -> None:
        hit = self._axes_canvas_center()
        if hit is not None:
            ax_cx, ax_cy, hit_r = hit
            if (event.x - ax_cx) ** 2 + (event.y - ax_cy) ** 2 <= hit_r ** 2:
                self._dragging = True
                self._drag_mode = "axes"
                self._drag_last = (event.x, event.y)
                self.var_axes_fixed.set(False)
                return
        if self._vp_scene is None:
            return
        self._dragging = True
        self._drag_mode = "pan" if (event.state & 0x0001) else "orbit"
        self._drag_last = (event.x, event.y)

    def _on_canvas_pan_down(self, event: tk.Event) -> None:
        if self._vp_scene is None:
            return
        self._dragging = True
        self._drag_mode = "pan"
        self._drag_last = (event.x, event.y)

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if not self._dragging:
            return
        if self._drag_mode == "axes":
            rect = self._frame_rect()
            if rect is not None:
                x0, y0, fw, fh = rect
                u = max(0.06, min(0.94, (event.x - x0) / max(1.0, fw)))
                v = max(0.06, min(0.94, (event.y - y0) / max(1.0, fh)))
                self._axes_pos = (u, v)
                self.var_axes_fixed.set(False)
                if self._vp_scene is not None:
                    self._vp_scene.axes_pos = self._axes_pos
                self._redraw_canvas()
            return
        if self._vp_scene is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        if dx == 0 and dy == 0:
            return

        if self._drag_mode == "orbit":
            self._vp_scene.orbit(dx * 0.0085, dy * 0.0085)
        elif self._drag_mode == "pan":
            px_per_a = self._viewport_px_per_angstrom()
            self._vp_scene.pan(dx / max(1e-3, px_per_a), dy / max(1e-3, px_per_a))

        self.var_canvas_mode.set("3d")
        self._use_interactive_cam = True
        self._preview_seq += 1
        self._draw_3d_viewport()

    def _on_canvas_mouse_up(self, _event: tk.Event) -> None:
        if self._dragging:
            was_axes = self._drag_mode == "axes"
            self._dragging = False
            self._drag_mode = None
            if was_axes:
                self._redraw_canvas()
                return
            if self.var_canvas_mode.get() == "3d":
                self._draw_3d_viewport(interactive=False)
                self._request_blender_preview()

    def _on_canvas_wheel(self, event: tk.Event) -> None:
        if self._vp_scene is None:
            return
        factor = 0.9 if event.delta > 0 else 1.111
        self._zoom_3d(factor)

    def _schedule_hq_redraw(self) -> None:
        if self._hq_after_id is not None:
            self.after_cancel(self._hq_after_id)
        self._hq_after_id = self.after(300, self._run_hq_redraw)

    def _run_hq_redraw(self) -> None:
        self._hq_after_id = None
        if not self._dragging and self._vp_scene is not None:
            if self.var_canvas_mode.get() == "3d":
                self._draw_3d_viewport(interactive=False)
            self._request_blender_preview()

    def _zoom_3d(self, factor: float) -> None:
        if self._vp_scene is None:
            return
        self.var_canvas_mode.set("3d")
        self._vp_scene.zoom(factor)
        self._use_interactive_cam = True
        self._preview_seq += 1
        self._draw_3d_viewport(interactive=True)
        self._schedule_hq_redraw()

    def _viewport_px_per_angstrom(self) -> float:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        pad = min(14.0, min(cw, ch) * 0.12)
        scale = min((cw - pad) / res_w, (ch - pad) / res_h)
        fw = res_w * scale
        fh = res_h * scale
        ortho = self._vp_scene.ortho_scale if self._vp_scene is not None else 20.0
        return (fw / max(1e-3, ortho)) if res_w >= res_h else (fh / max(1e-3, ortho))

    def _load_welcome_animation(self) -> None:
        """Load pre-rendered Blender welcome frames (or synthesize lightweight fallback frames)."""
        if not _HAS_PIL:
            return
        frames: list[Any] = []
        gif_path = package_dir() / "assets" / "welcome_loop.gif"
        if gif_path.is_file():
            try:
                with Image.open(gif_path) as im:
                    n_frames = max(1, int(getattr(im, "n_frames", 1)))
                    for i in range(n_frames):
                        im.seek(i)
                        frames.append(im.convert("RGBA").copy())
            except Exception:
                frames = []
        if not frames:
            n_fb = 16
            for i in range(n_fb):
                ang = 2.0 * math.pi * (i / n_fb)
                fr = Image.new("RGBA", (220, 220), (15, 17, 21, 255))
                dr = ImageDraw.Draw(fr, "RGBA")
                cx0, cy0 = 110.0, 110.0
                for k, (col, r_orb, rad) in enumerate((
                    ("#FB923C", 54.0, 16.0),
                    ("#38BDF8", 54.0, 16.0),
                    ("#F1F5F9", 54.0, 16.0),
                )):
                    a = ang + k * (2.0 * math.pi / 3.0)
                    px, py = cx0 + r_orb * math.cos(a), cy0 + r_orb * math.sin(a) * 0.65
                    dr.line((cx0, cy0, px, py), fill=(148, 163, 184, 220), width=6)
                    dr.ellipse((px - rad, py - rad, px + rad, py + rad), fill=col)
                dr.ellipse((cx0 - 20, cy0 - 20, cx0 + 20, cy0 + 20), fill="#14B8A6")
                frames.append(fr)
        self._welcome_pil_frames = frames
        self._welcome_photo_refs = [ImageTk.PhotoImage(f) for f in frames]
        self._welcome_frame_idx = 0
        self._welcome_anim_step = 0

    def _welcome_style_phase_for_frame(self, idx: int) -> tuple[int, str, str, str]:
        total = max(1, len(self._welcome_photo_refs) or len(self._welcome_pil_frames) or 72)
        seg = max(1, total // len(WELCOME_STYLE_PHASES))
        phase_idx = min(len(WELCOME_STYLE_PHASES) - 1, (idx % total) // seg)
        name, desc, color = WELCOME_STYLE_PHASES[phase_idx]
        return phase_idx, name, desc, color

    def _is_point_over_welcome_atom(self, x: int, y: int) -> bool:
        if self._vp_scene is not None or self._pil_image is not None:
            return False
        bbox = self.canvas.bbox("welcome_atom")
        if bbox is None:
            return False
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        r = max(110.0, min(168.0, 0.44 * max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
        return ((x - cx) ** 2 + (y - cy) ** 2) <= (r ** 2)

    def _refresh_welcome_hover_badge(self) -> None:
        return

    def _on_canvas_hover_motion(self, event: tk.Event) -> None:
        paused = self._is_point_over_welcome_atom(int(event.x), int(event.y))
        if paused != self._welcome_hover_paused:
            self._welcome_hover_paused = paused
            self._refresh_welcome_hover_badge()

    def _on_canvas_hover_leave(self, _event: tk.Event) -> None:
        if self._welcome_hover_paused:
            self._welcome_hover_paused = False
            self._refresh_welcome_hover_badge()

    def _stop_welcome_animation(self) -> None:
        self._welcome_hover_paused = False
        if self._welcome_after_id is not None:
            try:
                self.after_cancel(self._welcome_after_id)
            except Exception:
                pass
            self._welcome_after_id = None

    def _draw_welcome_screen(self, cw: int, ch: int) -> None:
        """Render the unified Welcome screen with 1.5x molecule, white header, and repeating prompt."""
        self.canvas.delete("welcome")
        dark_bg = bool(self.var_dark_backdrop.get())
        cx = max(cw, 200) // 2
        eff_ch = ch if ch >= 400 else max(ch, int(self.canvas.cget("height") or 680))

        bg_hex = "#0F1115" if dark_bg else "#ECECEC"
        title_fg = "#F8FAFC" if dark_bg else "#0F172A"
        drag_fg = "#CBD5E1" if dark_bg else "#334155"
        formats_fg = "#64748B" if dark_bg else "#64748B"

        hdr_alpha = _welcome_header_fade_alpha(self._welcome_anim_step)
        hdr_fill = _blend_hex_color(bg_hex, title_fg, hdr_alpha)
        prompt_alpha = _welcome_prompt_fade_alpha(self._welcome_anim_step)
        drag_fill = _blend_hex_color(bg_hex, drag_fg, prompt_alpha)
        formats_fill = _blend_hex_color(bg_hex, formats_fg, prompt_alpha)

        title_pt = 64 if eff_ch >= 600 else 48
        atom_h = (
            self._welcome_pil_frames[0].height
            if self._welcome_pil_frames
            else 360
        )

        # Measure header heights first so the whole vertical stack is centered and never overlaps at any DPI
        tmp_sub = self.canvas.create_text(
            cx, 0, text="Atomic Blender Studio", anchor=tk.N, font=("Segoe UI Light", 13), fill=hdr_fill
        )
        tmp_title = self.canvas.create_text(
            cx, 0, text="Welcome", anchor=tk.N, font=("Segoe UI Light", title_pt), fill=hdr_fill
        )
        sb = self.canvas.bbox(tmp_sub)
        tb = self.canvas.bbox(tmp_title)
        self.canvas.delete(tmp_sub)
        self.canvas.delete(tmp_title)
        sub_h = (sb[3] - sb[1]) if sb else 22
        title_h = (tb[3] - tb[1]) if tb else 88
        leading_trim = int(round(title_h * 0.20))

        total_h = sub_h + max(48, title_h - leading_trim) + atom_h + 28
        y_sub_top = max(14, (eff_ch - total_h) // 2)

        self.canvas.create_text(
            cx,
            y_sub_top,
            text="Atomic Blender Studio",
            anchor=tk.N,
            font=("Segoe UI Light", 13),
            fill=hdr_fill,
            tags=("welcome", "welcome_subtitle"),
        )
        sub_bbox = self.canvas.bbox("welcome_subtitle")
        y_title_top = ((sub_bbox[3] - leading_trim) if sub_bbox else (y_sub_top + sub_h - leading_trim))
        self.canvas.create_text(
            cx,
            y_title_top,
            text="Welcome",
            anchor=tk.N,
            font=("Segoe UI Light", title_pt),
            fill=hdr_fill,
            tags=("welcome", "welcome_title"),
        )
        title_bbox = self.canvas.bbox("welcome_title")
        y_atom_top = ((title_bbox[3] - 12) if title_bbox else (y_title_top + title_h - 12))

        if self._welcome_photo_refs:
            idx = self._welcome_frame_idx % len(self._welcome_photo_refs)
            self.canvas.create_image(
                cx,
                y_atom_top,
                image=self._welcome_photo_refs[idx],
                anchor=tk.N,
                tags=("welcome", "welcome_atom"),
            )
        else:
            r = 84
            yc = y_atom_top + atom_h // 2
            self.canvas.create_oval(
                cx - r,
                yc - r,
                cx + r,
                yc + r,
                fill="#14B8A6",
                outline="#2DD4BF",
                width=2,
                tags=("welcome", "welcome_atom"),
            )

        atom_bbox = self.canvas.bbox("welcome_atom")
        y_drag = (atom_bbox[3] + 10) if atom_bbox else (y_atom_top + atom_h + 10)
        self.canvas.create_text(
            cx,
            y_drag,
            text="Drag & drop a structure file anywhere  ·  or click Open",
            anchor=tk.N,
            justify=tk.CENTER,
            font=("Segoe UI", 10),
            fill=drag_fill,
            tags=("welcome", "welcome_drag"),
        )
        drag_bbox = self.canvas.bbox("welcome_drag")
        y_formats = (drag_bbox[3] + 6) if drag_bbox else (y_drag + 22)
        self.canvas.create_text(
            cx,
            y_formats,
            text="CIF  ·  POSCAR  ·  XYZ  ·  LAMMPS  ·  XSF  ·  view / style / recipe JSON",
            anchor=tk.N,
            justify=tk.CENTER,
            font=("Segoe UI", 9),
            fill=formats_fill,
            tags=("welcome", "welcome_formats"),
        )

        if len(self._welcome_photo_refs) > 1 and self._welcome_after_id is None and not getattr(self, "_destroyed", False):
            self._welcome_after_id = self.after(WELCOME_FRAME_INTERVAL_MS, self._tick_welcome_animation)

    def _tick_welcome_animation(self) -> None:
        self._welcome_after_id = None
        if getattr(self, "_destroyed", False):
            return
        in_empty_3d = self.var_canvas_mode.get() == "3d" and self._vp_scene is None
        in_empty_png = self.var_canvas_mode.get() != "3d" and self._pil_image is None
        if not (in_empty_3d or in_empty_png):
            return
        if not self._welcome_photo_refs:
            return
        if self._welcome_hover_paused:
            if len(self._welcome_photo_refs) > 1 and not getattr(self, "_destroyed", False):
                self._welcome_after_id = self.after(WELCOME_FRAME_INTERVAL_MS, self._tick_welcome_animation)
            return
        total_frames = len(self._welcome_photo_refs)
        self._welcome_frame_idx = (self._welcome_frame_idx + 1) % total_frames
        self._welcome_anim_step += 1

        dark_bg = bool(self.var_dark_backdrop.get())
        bg_hex = "#0F1115" if dark_bg else "#ECECEC"
        title_fg = "#F8FAFC" if dark_bg else "#0F172A"
        drag_fg = "#CBD5E1" if dark_bg else "#334155"
        formats_fg = "#64748B" if dark_bg else "#64748B"

        hdr_fill = _blend_hex_color(bg_hex, title_fg, _welcome_header_fade_alpha(self._welcome_anim_step))
        prompt_alpha = _welcome_prompt_fade_alpha(self._welcome_anim_step)
        drag_fill = _blend_hex_color(bg_hex, drag_fg, prompt_alpha)
        formats_fill = _blend_hex_color(bg_hex, formats_fg, prompt_alpha)
        try:
            atom_items = self.canvas.find_withtag("welcome_atom")
            if atom_items and self.canvas.type(atom_items[0]) == "image":
                self.canvas.itemconfigure(atom_items[0], image=self._welcome_photo_refs[self._welcome_frame_idx])
                for tag_name in ("welcome_title", "welcome_subtitle"):
                    items = self.canvas.find_withtag(tag_name)
                    if items:
                        self.canvas.itemconfigure(items[0], fill=hdr_fill)
                drag_items = self.canvas.find_withtag("welcome_drag")
                if drag_items:
                    self.canvas.itemconfigure(drag_items[0], fill=drag_fill)
                fmt_items = self.canvas.find_withtag("welcome_formats")
                if fmt_items:
                    self.canvas.itemconfigure(fmt_items[0], fill=formats_fill)
            else:
                self._draw_welcome_screen(max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        except Exception:
            return
        if len(self._welcome_photo_refs) > 1 and not getattr(self, "_destroyed", False):
            self._welcome_after_id = self.after(WELCOME_FRAME_INTERVAL_MS, self._tick_welcome_animation)

    def _draw_3d_viewport(self, interactive: bool | None = None) -> None:
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if self._vp_scene is None:
            self._draw_welcome_screen(cw, ch)
            return
        self._stop_welcome_animation()

        vp = self._vp_scene
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        is_interactive = self._dragging if interactive is None else interactive

        if _HAS_PIL:
            if (
                self._ghost_vp_scene is not None
                and self.var_compare_mode.get() == "side_by_side"
                and cw >= 64
            ):
                half_w = cw // 2
                self._ghost_vp_scene.quat = vp.quat
                self._ghost_vp_scene.target = vp.target.copy()
                self._ghost_vp_scene.ortho_scale = vp.ortho_scale
                img_l = self._ghost_vp_scene.render_image(half_w, ch, res_w, res_h, interactive=is_interactive)
                img_r = vp.render_image(cw - half_w, ch, res_w, res_h, interactive=is_interactive)
                combined = Image.new("RGBA", (cw, ch), (15, 17, 21, 255))
                combined.paste(img_l, (0, 0))
                combined.paste(img_r, (half_w, 0))
                self._vp_photo_ref = ImageTk.PhotoImage(combined)
                self.canvas.create_image(0, 0, image=self._vp_photo_ref, anchor=tk.NW)
                self.canvas.create_line(half_w, 0, half_w, ch, fill="#14B8A6", width=2)
                self.canvas.create_text(
                    12, 12, text=f"Ref: {self._ghost_label_name or 'Reference'}", anchor=tk.NW,
                    fill="#38BDF8", font=("Segoe UI", 9, "bold"),
                )
                self.canvas.create_text(
                    half_w + 12, 12, text="Current", anchor=tk.NW,
                    fill="#2DD4BF", font=("Segoe UI", 9, "bold"),
                )
                return
            img = vp.render_image(cw, ch, res_w, res_h, interactive=is_interactive)
            self._vp_photo_ref = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=self._vp_photo_ref, anchor=tk.NW)
            return
        pad = min(14.0, min(cw, ch) * 0.12)
        scale = min((cw - pad) / res_w, (ch - pad) / res_h)
        fw = res_w * scale
        fh = res_h * scale
        cx0 = cw * 0.5
        cy0 = ch * 0.5
        x0, y0, x1, y1 = cx0 - fw * 0.5, cy0 - fh * 0.5, cx0 + fw * 0.5, cy0 + fh * 0.5

        # Camera output frame boundary
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#F9FAFB", outline="#CBD5E1", width=1)

        px_per_a = (fw / max(1e-3, vp.ortho_scale)) if res_w >= res_h else (fh / max(1e-3, vp.ortho_scale))
        right, up, back = quat_to_axes(vp.quat)

        draw_items: list[tuple[float, int, Any]] = []

        # 1. Cell wireframe edges
        if len(vp.cell_edges) > 0:
            ce_rel = vp.cell_edges - vp.target
            ce_x = cx0 + (ce_rel @ right) * px_per_a
            ce_y = cy0 - (ce_rel @ up) * px_per_a
            ce_z = ce_rel @ back
            cell_w = max(1, min(3, int(round(0.10 * px_per_a))))
            for idx in range(len(vp.cell_edges)):
                z_mid = float(0.5 * (ce_z[idx, 0] + ce_z[idx, 1]))
                draw_items.append((
                    z_mid,
                    0,
                    ("cell", float(ce_x[idx, 0]), float(ce_y[idx, 0]), float(ce_x[idx, 1]), float(ce_y[idx, 1]), cell_w),
                ))

        # 2. Atoms projection
        if len(vp.positions) > 0:
            rel = vp.positions - vp.target
            ax = cx0 + (rel @ right) * px_per_a
            ay = cy0 - (rel @ up) * px_per_a
            az = rel @ back

            # 3. Bonds
            if len(vp.bond_i) > 0:
                bond_w = max(1, min(14, int(round(2.0 * vp.bond_radius * px_per_a))))
                bi, bj = vp.bond_i, vp.bond_j
                if vp.bond_color_mode == "bicolor":
                    mx = 0.5 * (ax[bi] + ax[bj])
                    my = 0.5 * (ay[bi] + ay[bj])
                    mz = 0.5 * (az[bi] + az[bj])
                    for k in range(len(bi)):
                        i_idx, j_idx = int(bi[k]), int(bj[k])
                        draw_items.append((
                            float(0.5 * (az[i_idx] + mz[k])),
                            1,
                            ("bond", float(ax[i_idx]), float(ay[i_idx]), float(mx[k]), float(my[k]), bond_w, vp.colors[i_idx]),
                        ))
                        draw_items.append((
                            float(0.5 * (mz[k] + az[j_idx])),
                            1,
                            ("bond", float(mx[k]), float(my[k]), float(ax[j_idx]), float(ay[j_idx]), bond_w, vp.colors[j_idx]),
                        ))
                else:
                    for k in range(len(bi)):
                        i_idx, j_idx = int(bi[k]), int(bj[k])
                        draw_items.append((
                            float(0.5 * (az[i_idx] + az[j_idx])),
                            1,
                            ("bond", float(ax[i_idx]), float(ay[i_idx]), float(ax[j_idx]), float(ay[j_idx]), bond_w, vp.single_bond_color),
                        ))

            for idx in range(len(vp.positions)):
                r_px = float(vp.radii[idx] * px_per_a)
                draw_items.append((float(az[idx]), 2, ("atom", float(ax[idx]), float(ay[idx]), r_px, idx)))

        # Painter's sort far -> near
        draw_items.sort(key=lambda item: (item[0], item[1]))
        fast_drag = self._dragging and len(vp.positions) > 1500
        c_create_line = self.canvas.create_line
        c_create_oval = self.canvas.create_oval
        c_col = vp.cell_color

        for _z, _kind_ord, payload in draw_items:
            tag = payload[0]
            if tag == "cell":
                _, px0, py0, px1, py1, w = payload
                c_create_line(px0, py0, px1, py1, fill=c_col, width=w)
            elif tag == "bond":
                _, px0, py0, px1, py1, w, col = payload
                c_create_line(px0, py0, px1, py1, fill=col, width=w, capstyle=tk.ROUND)
            else:
                _, sx, sy, r_px, idx = payload
                if r_px < 0.8:
                    continue
                c_create_oval(
                    sx - r_px,
                    sy - r_px,
                    sx + r_px,
                    sy + r_px,
                    fill=vp.colors[idx],
                    outline=vp.outline_colors[idx],
                    width=1,
                )
                if not fast_drag and r_px >= 6.0:
                    hr = r_px * 0.34
                    hx = sx - r_px * 0.32
                    hy = sy - r_px * 0.32
                    c_create_oval(hx - hr, hy - hr, hx + hr, hy + hr, fill=vp.highlight_colors[idx], outline="")

        # Re-stroke frame border cleanly on top
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#94A3B8", width=1)

    def _redraw_preview(self) -> None:
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        dark_bg = bool(self.var_dark_backdrop.get())
        tile = 20
        c_even = "#242830" if dark_bg else "#F3F3F3"
        c_odd = "#1C1F26" if dark_bg else "#E2E2E2"
        for y in range(0, ch, tile):
            for x in range(0, cw, tile):
                color = c_even if ((x // tile + y // tile) % 2 == 0) else c_odd
                self.canvas.create_rectangle(x, y, x + tile, y + tile, fill=color, outline="")

        if self._pil_image is None:
            self._draw_welcome_screen(cw, ch)
            return
        self._stop_welcome_animation()

        if _HAS_PIL:
            iw, ih = self._pil_image.size
            scale = min((cw - 14) / max(1, iw), (ch - 14) / max(1, ih))
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            resized = self._pil_image.resize((nw, nh), Image.Resampling.LANCZOS).copy()
            resized = resized.filter(ImageFilter.UnsharpMask(radius=0.9, percent=45, threshold=2))
            res_w = max(16, int(self.var_res_w.get()))
            res_h = max(16, int(self.var_res_h.get()))
            if self._vp_scene is not None and (self.var_show_labels.get() or self._selected_atoms):
                render_atom_labels_and_selection_overlay(
                    resized,
                    self._vp_scene,
                    res_w,
                    res_h,
                    draw_labels=bool(self.var_show_labels.get()),
                    draw_selection=bool(self._selected_atoms),
                )
            if (
                self._vp_scene is not None
                and self.var_show_measure_overlay.get()
                and len(self._selection_order) in (2, 3, 4)
            ):
                render_measurement_callout_overlay(
                    resized,
                    self._vp_scene,
                    self._selection_order,
                    res_w,
                    res_h,
                    dark_backdrop=dark_bg,
                )
            from_tmp_preview = (
                self._preview_path is not None
                and hasattr(self, "_preview_tmp_dir")
                and str(self._preview_path).startswith(str(self._preview_tmp_dir.name))
            )
            if self.var_show_scale_bar.get() and self._vp_scene is not None and from_tmp_preview:
                render_scale_bar_overlay(
                    resized,
                    self._vp_scene.ortho_scale,
                    res_w,
                    res_h,
                    dark_backdrop=dark_bg,
                )
            cmap_info = getattr(self, "_active_colormap_info", None)
            if cmap_info is not None and from_tmp_preview:
                render_colorbar_overlay(
                    resized,
                    *cmap_info,
                    dark_backdrop=dark_bg,
                )
            if self.var_show_axes.get() and from_tmp_preview:
                quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                cvecs = self._vp_scene.cell_vecs if (self._vp_scene is not None and self._vp_scene.has_cell) else None
                render_reference_axes_overlay(
                    resized,
                    quat,
                    axes_pos=self._axes_pos,
                    show_labels=bool(self.var_axes_labels.get()),
                    dark_backdrop=dark_bg,
                    cell_vecs=cvecs,
                )
            self._photo_ref = ImageTk.PhotoImage(resized)
        else:
            iw = self._pil_image.width()
            ih = self._pil_image.height()
            factor = max(1, int(max(iw / max(1, cw - 16), ih / max(1, ch - 16)) + 0.999))
            self._photo_ref = self._pil_image.subsample(factor, factor)

        self.canvas.create_image(cw // 2, ch // 2, image=self._photo_ref, anchor=tk.CENTER)

    def destroy(self) -> None:
        self._destroyed = True
        for aid_attr in ("_poll_after_id", "_inspect_after_id", "_hq_after_id", "_welcome_after_id"):
            aid = getattr(self, aid_attr, None)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, aid_attr, None)
        if hasattr(self, "_preview_worker"):
            self._preview_worker.stop()
        if hasattr(self, "_preview_tmp_dir"):
            try:
                self._preview_tmp_dir.cleanup()
            except Exception:
                pass
        for tmp_p in getattr(self, "_temp_view_files", []):
            try:
                tmp_p.unlink(missing_ok=True)
            except Exception:
                pass
        super().destroy()

    def _poll_queue(self) -> None:
        self._poll_after_id = None
        if getattr(self, "_destroyed", False):
            return
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "inspect":
                if isinstance(payload, tuple) and len(payload) == 3:
                    req_seq, req_path, msg = payload
                    cur_path = extract_input_path(self.var_input.get())
                    if req_seq != self._inspect_seq or (cur_path and cur_path != req_path):
                        continue
                    self.var_inspector.set(str(msg))
                else:
                    self.var_inspector.set(str(payload))
            elif kind == "structure":
                if len(payload) == 4:
                    req_seq, req_path, summary, neutral = payload
                    cur_path = extract_input_path(self.var_input.get())
                    if req_seq != self._inspect_seq or (cur_path and cur_path != req_path):
                        continue
                else:
                    summary, neutral = payload
                self.var_inspector.set(str(summary))
                self._selected_atoms.clear()
                self._selection_order.clear()
                self._strip_pinned_frames.clear()
                self._contact_cache_key = None
                self._trajectory_frames = [dict(neutral)]
                self._trajectory_index = 0
                self._update_frame_controls()
                self._base_structure_cache = dict(neutral)
                self._ghost_structure_cache = None
                self._ghost_label_name = ""
                self._ghost_vp_scene = None
                self._structure_cache = neutral
                self._structure_version += 1
                self._use_interactive_cam = False
                in_raw = extract_input_path(self.var_input.get())
                if in_raw:
                    self._add_recent_file(in_raw)
                self._rebuild_viewport(reset_camera=True, trigger_preview=True)
            elif kind == "structure_frames":
                if len(payload) == 4:
                    req_seq, req_path, summary, frames = payload
                    cur_path = extract_input_path(self.var_input.get())
                    if req_seq != self._inspect_seq or (cur_path and cur_path != req_path):
                        continue
                else:
                    summary, frames = payload
                self.var_inspector.set(str(summary))
                self._selected_atoms.clear()
                self._selection_order.clear()
                self._strip_pinned_frames.clear()
                self._contact_cache_key = None
                self._trajectory_frames = [dict(f) for f in frames] if frames else []
                self._trajectory_index = 0
                self._update_frame_controls()
                first = dict(self._trajectory_frames[0]) if self._trajectory_frames else {}
                self._base_structure_cache = dict(first) if first else None
                self._ghost_structure_cache = None
                self._ghost_label_name = ""
                self._ghost_vp_scene = None
                self._structure_cache = first or None
                self._structure_version += 1
                self._use_interactive_cam = False
                in_raw = extract_input_path(self.var_input.get())
                if in_raw:
                    self._add_recent_file(in_raw)
                if self._structure_cache is not None:
                    self._rebuild_viewport(reset_camera=True, trigger_preview=True)
            elif kind == "preview_done":
                seq, png_path = payload
                if seq == self._preview_seq and not self._dragging:
                    self._load_preview(png_path)
                    if not self._busy:
                        self.var_status.set("Blender preview ready (drag to rotate / scroll to zoom)")
            elif kind == "preview_err":
                seq, msg = payload
                if seq == self._preview_seq:
                    self._append_log(f"Preview error: {msg}")
                    if not self._busy:
                        self.var_status.set("3D preview active (Blender preview failed)")
            elif kind == "drop":
                self.handle_dropped_files(list(payload))
            elif kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                mode, rc, target_path = payload
                elapsed = time.monotonic() - self._task_start
                if rc != 0:
                    self._set_busy(False, f"Failed (exit {rc}) after {elapsed:.1f}s")
                    messagebox.showerror("Render Failed", f"Command exited with status {rc}. Check Log for details.")
                elif mode == "render":
                    self._preview_seq += 1
                    self._set_busy(False, f"Rendered {target_path.name} in {elapsed:.1f}s")
                    cmap_info = getattr(self, "_active_colormap_info", None)
                    if (
                        (self.var_show_axes.get() or self.var_show_scale_bar.get() or cmap_info is not None)
                        and _HAS_PIL
                        and target_path.is_file()
                    ):
                        try:
                            out_img = Image.open(target_path).convert("RGBA")
                            if self.var_show_scale_bar.get() and self._vp_scene is not None:
                                render_scale_bar_overlay(
                                    out_img,
                                    self._vp_scene.ortho_scale,
                                    max(16, int(self.var_res_w.get())),
                                    max(16, int(self.var_res_h.get())),
                                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                                )
                            if cmap_info is not None:
                                render_colorbar_overlay(
                                    out_img,
                                    *cmap_info,
                                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                                )
                            if self.var_show_axes.get():
                                quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                                cvecs = (
                                    self._vp_scene.cell_vecs
                                    if (self._vp_scene is not None and self._vp_scene.has_cell)
                                    else None
                                )
                                render_reference_axes_overlay(
                                    out_img,
                                    quat,
                                    axes_pos=self._axes_pos,
                                    show_labels=bool(self.var_axes_labels.get()),
                                    dark_backdrop=bool(self.var_dark_backdrop.get()),
                                    cell_vecs=cvecs,
                                )
                            out_img.save(target_path)
                        except Exception as exc:
                            self._append_log(f"Overlay warning: {exc}")
                    self._load_preview(target_path)
                elif mode == "open_blender":
                    self._set_busy(False, f"Opening {target_path.name} in Blender...")
                    subprocess.Popen([self._pending_blender_exe, str(target_path)])
            elif kind == "done_panel":
                rc, panel_png, err_msg = payload
                elapsed = time.monotonic() - self._task_start
                if rc != 0:
                    self._set_busy(False, f"Panel failed after {elapsed:.1f}s")
                    messagebox.showerror("Panel Render Failed", err_msg or f"Command exited with status {rc}")
                else:
                    self._preview_seq += 1
                    self._set_busy(False, f"Saved panel {panel_png.name} in {elapsed:.1f}s")
                    self._load_preview(panel_png)
            elif kind == "done_strip":
                rc, strip_png, err_msg = payload
                elapsed = time.monotonic() - self._task_start
                if rc != 0:
                    self._set_busy(False, f"Path strip failed after {elapsed:.1f}s")
                    messagebox.showerror("Path Strip Failed", err_msg or f"Strip exited with status {rc}")
                else:
                    self._preview_seq += 1
                    self._set_busy(False, f"Saved path strip {strip_png.name} in {elapsed:.1f}s")
                    self._load_preview(strip_png)
            elif kind == "done_batch":
                rc, done_cnt, total_cnt, last_png = payload
                elapsed = time.monotonic() - self._task_start
                if rc != 0:
                    self._set_busy(False, f"Batch failed at {done_cnt + 1}/{total_cnt} after {elapsed:.1f}s")
                    messagebox.showerror("Batch Render Failed", f"Stopped after {done_cnt}/{total_cnt} files.")
                else:
                    self._set_busy(False, f"Batch rendered {done_cnt}/{total_cnt} PNGs in {elapsed:.1f}s")
                    if last_png is not None and Path(last_png).is_file():
                        self._load_preview(Path(last_png))
        if not getattr(self, "_destroyed", False):
            self._poll_after_id = self.after(100, self._poll_queue)

    def handle_dropped_files(self, paths: list[str]) -> None:
        """Apply dropped structure, view.json, style.json, or recipe JSON files to the GUI state."""
        for raw_path in paths:
            p = Path(raw_path).expanduser()
            if not p.is_file():
                continue
            kind = classify_dropped_file(p)
            if kind == "recipe":
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self.apply_figure_recipe(data, trigger_preview=True)
                    self.var_status.set(f"Loaded figure recipe via drop: {p.name}")
                    self._append_log(f"Dropped recipe file: {p}")
                except Exception as exc:
                    messagebox.showerror("Invalid Recipe JSON", str(exc))
            elif kind == "view":
                self.var_view_mode.set("file")
                self.var_view_file.set(str(p))
                self._sync_view_mode()
                self._use_interactive_cam = False
                if self._vp_scene is not None:
                    self._apply_view_to_viewport(self._vp_scene)
                    self.var_canvas_mode.set("3d")
                    self._redraw_canvas()
                    self._request_blender_preview()
                self.var_status.set(f"Loaded view.json via drop: {p.name}")
                self._append_log(f"Dropped view file: {p}")
            elif kind == "style":
                self.var_style.set(str(p))
                self.var_status.set(f"Loaded style JSON via drop: {p.name}")
                self._append_log(f"Dropped style file: {p}")
                self._rebuild_viewport(reset_camera=False, trigger_preview=True)
            else:
                self.var_input.set(str(p))
                self.var_output.set(str(p.with_suffix(".png")))
                self._schedule_inspect()
                self.var_status.set(f"Loaded structure via drop: {p.name}")
                self._append_log(f"Dropped structure file: {p}")

    def _enable_file_drop(self) -> None:
        """Enable native OS file drag-and-drop onto this Tkinter window."""
        if sys.platform.startswith("win") or os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32
                shell32 = ctypes.windll.shell32

                long_ptr = ctypes.c_ssize_t
                wndproc_type = ctypes.WINFUNCTYPE(long_ptr, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

                set_window_long = (
                    user32.SetWindowLongPtrW if hasattr(user32, "SetWindowLongPtrW") else user32.SetWindowLongW
                )
                set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, long_ptr]
                set_window_long.restype = long_ptr

                user32.CallWindowProcW.argtypes = [
                    long_ptr,
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                ]
                user32.CallWindowProcW.restype = long_ptr

                user32.GetParent.argtypes = [wintypes.HWND]
                user32.GetParent.restype = wintypes.HWND

                shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
                shell32.DragAcceptFiles.restype = None

                shell32.DragQueryFileW.argtypes = [
                    wintypes.HANDLE,
                    wintypes.UINT,
                    wintypes.LPWSTR,
                    wintypes.UINT,
                ]
                shell32.DragQueryFileW.restype = wintypes.UINT

                shell32.DragFinish.argtypes = [wintypes.HANDLE]
                shell32.DragFinish.restype = None

                hwnd_val = user32.GetParent(wintypes.HWND(self.winfo_id())) or self.winfo_id()
                hwnd = wintypes.HWND(hwnd_val)
                shell32.DragAcceptFiles(hwnd, True)

                if hasattr(user32, "ChangeWindowMessageFilterEx"):
                    user32.ChangeWindowMessageFilterEx.argtypes = [
                        wintypes.HWND,
                        wintypes.UINT,
                        wintypes.DWORD,
                        ctypes.c_void_p,
                    ]
                    user32.ChangeWindowMessageFilterEx.restype = wintypes.BOOL
                    for msg_id in (0x0233, 0x0049):  # WM_DROPFILES, WM_COPYGLOBALDATA
                        user32.ChangeWindowMessageFilterEx(hwnd, msg_id, 1, None)

                wm_dropfiles = 0x0233
                wm_ncdestroy = 0x0082
                gwl_wndproc = -4

                def _wndproc(h_wnd, msg, wparam, lparam):
                    if msg == wm_dropfiles:
                        hdrop = wintypes.HANDLE(wparam)
                        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
                        dropped: list[str] = []
                        for idx in range(count):
                            length = shell32.DragQueryFileW(hdrop, idx, None, 0)
                            buf = ctypes.create_unicode_buffer(length + 1)
                            shell32.DragQueryFileW(hdrop, idx, buf, length + 1)
                            if buf.value:
                                dropped.append(buf.value)
                        shell32.DragFinish(hdrop)
                        if dropped:
                            # Never call Tkinter/Tcl from a Win32 WNDPROC while mainloop holds ENTER_TCL
                            self._queue.put(("drop", dropped))
                        return 0
                    orig = self._orig_wndproc
                    if msg == wm_ncdestroy and orig:
                        set_window_long(h_wnd, gwl_wndproc, orig)
                    return user32.CallWindowProcW(orig, h_wnd, msg, wparam, lparam)

                cb = wndproc_type(_wndproc)
                self._wndproc_ref = cb
                cb_ptr = ctypes.cast(cb, ctypes.c_void_p).value
                self._orig_wndproc = set_window_long(hwnd, gwl_wndproc, cb_ptr)
            except Exception as exc:
                self._append_log(f"Drag-and-drop hook unavailable: {exc}")
            return

        # Optional tkinterdnd2 support on macOS / Linux if installed
        try:
            self.tk.eval("package require tkdnd")
            self.tk.eval(f"tkdnd::drop_target register {self._w} DND_Files")
        except Exception:
            pass


def launch_gui(initial_input: str | None = None) -> int:
    """Launch the desktop GUI event loop."""
    app = AtomicBlenderApp(initial_input=initial_input, prompt_missing_blender=True)
    app.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if len(raw_argv) >= 2 and raw_argv[0] == "-m" and raw_argv[1] == "atomic_blender.cli":
        from atomic_blender.cli import main as cli_main

        return cli_main(raw_argv[2:])
    parser = argparse.ArgumentParser(prog="atomic-blender-gui", description="Launch Atomic Blender desktop GUI")
    parser.add_argument("input", nargs="?", default=None, help="Optional structure file to prefill")
    args = parser.parse_args(raw_argv)
    return launch_gui(initial_input=args.input)


if __name__ == "__main__":
    raise SystemExit(main())
