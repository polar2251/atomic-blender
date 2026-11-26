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
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import numpy as np

from atomic_blender.cli import _collect_style_files, find_blender_executable, resolve_style_path
from atomic_blender.convert import atoms_to_neutral_dict, read_structure
from atomic_blender.elements import VIEW_PRESETS, classify_dropped_file
from atomic_blender.viewport3d import (
    ViewportScene,
    atom_label,
    preset_to_quat,
    quat_to_axes,
    render_atom_labels_and_selection_overlay,
    render_reference_axes_overlay,
)

try:
    from PIL import Image, ImageTk  # type: ignore

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


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
    if not cfg.input_path.strip():
        raise ValueError("Please select an input structure file.")
    in_path = Path(cfg.input_path).expanduser()
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
        bg: str = "#1E2128",
        fg: str = "#E5E7EB",
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
            fill = "#0D9488" if checked else "#14161A"
            outline = "#14B8A6" if checked else "#64748B"
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
        bg: str = "#1E2128",
        fg: str = "#E5E7EB",
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
        outline = "#14B8A6" if selected else "#64748B"
        self._dot.create_oval(2, 2, 15, 15, fill="#14161A", outline=outline, width=1.5)
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
        super().__init__(parent, bg="#14161A", highlightbackground="#323844", highlightthickness=1)
        self._var = variable
        self._from = int(from_)
        self._to = int(to)
        self._inc = int(increment)
        self._cmd = command

        btn_cfg = {
            "bg": "#2B303B",
            "fg": "#E5E7EB",
            "activebackground": "#363C49",
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
            bg="#14161A",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
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
        bg: str = "#1E2128",
        height: int = 26,
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

        # Recessed track + glowing teal active segment
        self.create_line(x0, cy, x1, cy, fill="#2B303B", width=6, capstyle=tk.ROUND)
        if tx > x0 + 1:
            self.create_line(x0, cy, tx, cy, fill="#0D9488", width=6, capstyle=tk.ROUND)

        r = 8 if (self._hover or self._dragging) else 7
        thumb_fill = "#2DD4BF" if (self._hover or self._dragging) else "#14B8A6"
        self.create_oval(tx - r, cy - r, tx + r, cy + r, fill=thumb_fill, outline="#F8FAFC", width=1.6)
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


def compute_selection_measurement(
    symbols: list[str],
    positions: Any,
    selected_order: list[int],
) -> str:
    """Compute live coordinates, bond distance, angle, or dihedral from 1-4 ordered atoms."""
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
        return f"{labels[0]}: ({x:.3f}, {y:.3f}, {z:.3f}) Å"

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
                proc.wait(timeout=1.5)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def _ensure_proc(self, blender_exe: str) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None and self._active_blender == blender_exe:
            return self._proc
        self._close_proc()
        scene_script = Path(__file__).resolve().parent / "scene_builder.py"
        env = os.environ.copy()
        pkg_parent = str(Path(__file__).resolve().parent.parent)
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
            try:
                proc.kill()
            except Exception:
                pass
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

        data = json.loads(p.read_text(encoding="utf-8"))
        syms = data.get("symbols", [])
        elems = ", ".join(sorted(set(syms)))
        eff_rep = replicate if data.get("has_cell", False) else (1, 1, 1)
        n_atoms = len(syms) * max(1, eff_rep[0]) * max(1, eff_rep[1]) * max(1, eff_rep[2])
        if data.get("has_cell", False):
            import math

            vecs = data.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            lens = [math.sqrt(sum(c * c for c in v)) for v in vecs]
            return (
                f"{p.name} | {n_atoms} atoms ({elems}) | "
                f"cell {lens[0]*eff_rep[0]:.2f}×{lens[1]*eff_rep[1]:.2f}×{lens[2]*eff_rep[2]:.2f} Å"
            )
        return f"{p.name} | {n_atoms} atoms ({elems}) | no periodic cell"

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
    """Enable native DPI awareness on Windows so fonts and the 3D canvas are crisp."""
    if sys.platform.startswith("win") or os.name == "nt":
        try:
            import ctypes

            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class AtomicBlenderApp(tk.Tk):
    """Desktop GUI client for atomic-blender."""

    def __init__(self, initial_input: str | None = None) -> None:
        _enable_hidpi()
        super().__init__()
        self.title("Atomic Blender Studio — Desktop Client")
        self.geometry("1280x840")
        self.minsize(1040, 680)

        self._configure_theme()

        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self._task_start = 0.0
        self._inspect_after_id: str | None = None
        self._hq_after_id: str | None = None
        self._preview_path: Path | None = None
        self._pil_image: Any = None
        self._photo_ref: Any = None
        self._vp_photo_ref: Any = None
        self._structure_cache: dict[str, Any] | None = None
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
        self._recent_files: list[str] = self._load_recent_files()
        self._display_checks: dict[str, tuple[tk.Canvas, tk.BooleanVar]] = {}
        self._temp_view_files: list[Path] = []
        self._preview_tmp_dir = tempfile.TemporaryDirectory(prefix="atomic_blender_prev_")
        self._preview_seq = 0
        self._destroyed = False
        self._poll_after_id: str | None = None
        self._preview_worker = BlenderPreviewWorker(
            on_done=lambda seq, p: self._queue.put(("preview_done", (seq, p))),
            on_error=lambda seq, msg: self._queue.put(("preview_err", (seq, msg))),
        )

        self._init_variables(initial_input)
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

        bg = "#14161B"
        panel = "#1E2128"
        recessed = "#121418"
        ctrl = "#2B303B"
        ctrl_hover = "#363C49"
        border = "#323844"
        fg = "#E5E7EB"
        muted = "#94A3B8"
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
        style.configure("TLabelframe", background=panel, bordercolor=border, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=panel, foreground="#CBD5E1", font=("Segoe UI", 9, "bold"))
        style.configure("TLabel", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.configure("Banner.TLabel", background=bg, foreground=fg, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=panel, foreground=muted, font=("Segoe UI", 8))

        style.configure(
            "TButton",
            background=ctrl,
            foreground=fg,
            bordercolor=border,
            lightcolor=ctrl,
            darkcolor=ctrl,
            padding=(8, 4),
            font=("Segoe UI", 9),
        )
        style.map(
            "TButton",
            background=[("active", ctrl_hover), ("pressed", "#242832"), ("disabled", "#1A1D24")],
            foreground=[("disabled", "#64748B")],
        )

        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#FFFFFF",
            bordercolor="#0F766E",
            lightcolor=accent,
            darkcolor=accent,
            padding=(10, 6),
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

        self.var_inspector = tk.StringVar(value="Select an atomic structure file (.cif, POSCAR, .xyz, .data, .json)")
        self.var_input = tk.StringVar(value=initial_input or "")
        self.var_output = tk.StringVar(value="")
        if initial_input:
            self.var_output.set(str(Path(initial_input).with_suffix(".png")))

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
        self.var_disp_ball_stick = tk.BooleanVar(value=True)
        self.var_disp_licorice = tk.BooleanVar(value=False)
        self.var_disp_vdw = tk.BooleanVar(value=False)
        self.var_show_labels = tk.BooleanVar(value=False)
        self.var_show_axes = tk.BooleanVar(value=True)
        self.var_axes_labels = tk.BooleanVar(value=True)
        self.var_axes_fixed = tk.BooleanVar(value=True)
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
        self.var_measurement = tk.StringVar(value="Double-click 1–4 atoms for coords / distance / angle / dihedral")
        self.var_z_min = tk.StringVar(value="")
        self.var_z_max = tk.StringVar(value="")
        self._active_tab = "view"
        self.var_status = tk.StringVar(value="Ready")
        self.var_input.trace_add("write", lambda *_: self._sync_type_map_visibility())

    def _build_layout(self) -> None:
        banner = ttk.Frame(self, padding=(10, 6, 10, 4), style="Outer.TFrame")
        banner.pack(fill=tk.X)
        ttk.Label(banner, text="Structure:", font=("Segoe UI", 9, "bold"), style="Banner.TLabel").pack(side=tk.LEFT)
        ttk.Label(banner, textvariable=self.var_inspector, foreground="#2DD4BF", style="Banner.TLabel").pack(
            side=tk.LEFT, padx=(6, 0)
        )

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

        left_host = ttk.Frame(paned, padding=0, style="Outer.TFrame")
        left_body = ttk.Frame(left_host, padding=0, style="Outer.TFrame")
        left_body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_canvas = tk.Canvas(left_body, bg="#14161B", highlightthickness=0, width=388)
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

    def _build_controls(self, parent: ttk.Frame, action_parent: ttk.Frame | None = None) -> None:
        # 1. Input & Output (pinned above tabs)
        f_io = ttk.LabelFrame(parent, text="Structure & Output", padding=8)
        f_io.pack(fill=tk.X, pady=(0, 6))
        f_io.columnconfigure(0, minsize=72)
        f_io.columnconfigure(1, weight=1)

        ttk.Label(f_io, text="Input").grid(row=0, column=0, sticky=tk.W, pady=2)
        entry_in = ttk.Entry(f_io, textvariable=self.var_input, width=34)
        entry_in.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        entry_in.bind("<FocusOut>", lambda _e: self._schedule_inspect())
        row_in_btns = ttk.Frame(f_io)
        row_in_btns.grid(row=0, column=2, sticky=tk.EW, pady=2)
        ttk.Button(row_in_btns, text="Browse...", command=self._browse_input).pack(side=tk.LEFT)
        self.btn_recent = tk.Menubutton(
            row_in_btns,
            text="▾",
            bg="#2B303B",
            fg="#E5E7EB",
            activebackground="#363C49",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            borderwidth=1,
            highlightbackground="#323844",
            highlightthickness=1,
            padx=5,
            pady=2,
            cursor="hand2",
            font=("Segoe UI", 9),
        )
        self.recent_menu = tk.Menu(
            self.btn_recent,
            tearoff=0,
            bg="#1E2128",
            fg="#E5E7EB",
            activebackground="#0D9488",
            activeforeground="#FFFFFF",
            disabledforeground="#64748B",
        )
        self.btn_recent.configure(menu=self.recent_menu)
        self.btn_recent.pack(side=tk.LEFT, padx=(3, 0))
        self._rebuild_recent_menu()

        ttk.Label(f_io, text="Output PNG").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(f_io, textvariable=self.var_output, width=34).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(f_io, text="Save As...", command=self._browse_output).grid(row=1, column=2, pady=2)

        self.lbl_type_map = ttk.Label(f_io, text="Type Map")
        self.entry_type_map = ttk.Entry(f_io, textvariable=self.var_type_map)
        self.entry_type_map.bind("<FocusOut>", lambda _e: self._schedule_inspect())
        self.btn_inspect_tm = ttk.Button(f_io, text="Inspect", command=self._schedule_inspect)
        self.lbl_type_map.grid(row=2, column=0, sticky=tk.W, pady=2)
        self.entry_type_map.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        self.btn_inspect_tm.grid(row=2, column=2, pady=2)
        self._sync_type_map_visibility()

        # 2. View / Tool / Selection Tab Switcher
        tab_bar = tk.Frame(parent, bg="#14161B")
        tab_bar.pack(fill=tk.X, pady=(0, 6))
        tab_bar.columnconfigure(0, weight=1, uniform="main_tabs")
        tab_bar.columnconfigure(1, weight=1, uniform="main_tabs")
        tab_bar.columnconfigure(2, weight=1, uniform="main_tabs")

        self._tab_buttons: dict[str, tuple[tk.Button, tk.Frame]] = {}
        for col_idx, (tab_key, tab_label) in enumerate(
            (("view", "View"), ("tool", "Tool"), ("selection", "Selection"))
        ):
            holder = tk.Frame(tab_bar, bg="#14161B")
            holder.grid(row=0, column=col_idx, sticky=tk.EW, padx=(0, 4) if col_idx < 2 else (0, 0))
            btn = tk.Button(
                holder,
                text=tab_label,
                command=lambda k=tab_key: self._switch_tab(k),
                bd=0,
                relief=tk.FLAT,
                pady=6,
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

        # --- View Tab: Unified Display Types, Step Camera View, Cycles Render ---
        self._build_display_types_panel(self.tab_view)

        f_cam = ttk.LabelFrame(self.tab_view, text="Camera View", padding=8)
        f_cam.pack(fill=tk.X, pady=(0, 6))
        f_cam.columnconfigure(0, minsize=72)
        f_cam.columnconfigure(2, weight=1)

        ttk.Label(f_cam, text="Rotate (°)").grid(row=0, column=0, sticky=tk.W, pady=2)
        ent_rot = ttk.Entry(f_cam, textvariable=self.var_rot_step, width=6, justify=tk.CENTER)
        ent_rot.grid(row=0, column=1, sticky=tk.W, padx=(4, 6), pady=2)
        row_rot = ttk.Frame(f_cam)
        row_rot.grid(row=0, column=2, sticky=tk.EW, pady=2)
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

        ttk.Label(f_cam, text="Move (Å)").grid(row=1, column=0, sticky=tk.W, pady=2)
        ent_pan = ttk.Entry(f_cam, textvariable=self.var_pan_step, width=6, justify=tk.CENTER)
        ent_pan.grid(row=1, column=1, sticky=tk.W, padx=(4, 6), pady=2)
        row_pan = ttk.Frame(f_cam)
        row_pan.grid(row=1, column=2, sticky=tk.EW, pady=2)
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

        f_ren = ttk.LabelFrame(self.tab_view, text="Cycles Render", padding=8)
        f_ren.pack(fill=tk.X, pady=(0, 6))
        f_ren.columnconfigure(0, minsize=72)
        f_ren.columnconfigure(1, weight=1)

        ttk.Label(f_ren, text="Resolution").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_res = ttk.Frame(f_ren)
        row_res.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        DarkSpinbox(row_res, from_=64, to=8192, increment=100, variable=self.var_res_w, width=5).pack(side=tk.LEFT)
        ttk.Label(row_res, text="×").pack(side=tk.LEFT, padx=5)
        DarkSpinbox(row_res, from_=64, to=8192, increment=100, variable=self.var_res_h, width=5).pack(side=tk.LEFT)

        row_res_presets = ttk.Frame(f_ren)
        row_res_presets.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=(0, 3))
        for c_idx, (label, w, h) in enumerate((("700 × 700", 700, 700), ("1400 × 1400", 1400, 1400), ("Slab 4:3", 1400, 1200))):
            row_res_presets.columnconfigure(c_idx, weight=1, uniform="res_btn")
            ttk.Button(
                row_res_presets, text=label, command=lambda ww=w, hh=h: self._set_resolution(ww, hh)
            ).grid(row=0, column=c_idx, sticky=tk.EW, padx=(0, 3 if c_idx < 2 else 0))

        ttk.Label(f_ren, text="Samples").grid(row=2, column=0, sticky=tk.W, pady=2)
        DarkSpinbox(f_ren, from_=1, to=2048, increment=8, variable=self.var_samples, width=6).grid(
            row=2, column=1, sticky=tk.W, padx=4, pady=2
        )

        ttk.Label(f_ren, text="Blender").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(f_ren, textvariable=self.var_blender).grid(row=3, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(f_ren, text="Browse...", command=self._browse_blender).grid(row=3, column=2, pady=2)

        # --- Tool Tab: Supercell & Framing + View File ---
        self.f_geom = ttk.LabelFrame(self.tab_tool, text="Supercell & Framing", padding=8)
        self.f_geom.pack(fill=tk.X, pady=(0, 6))
        self.f_geom_body = ttk.Frame(self.f_geom)
        self.f_geom_body.pack(fill=tk.X)
        f_geom = self.f_geom_body
        f_geom.columnconfigure(0, minsize=72)
        f_geom.columnconfigure(1, weight=1)

        ttk.Label(f_geom, text="Replicate").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_rep = ttk.Frame(f_geom)
        row_rep.grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        for var in (self.var_rep_x, self.var_rep_y, self.var_rep_z):
            sb = DarkSpinbox(row_rep, from_=1, to=20, variable=var, width=3, command=self._schedule_inspect)
            sb.pack(side=tk.LEFT, padx=(0, 4))
        DarkCheckbutton(row_rep, text="Wrap into cell", variable=self.var_wrap).pack(side=tk.LEFT, padx=(6, 0))

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
        ttk.Button(row_smart, text="Smart Slab Preset", command=self._apply_slab_preset).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 4)
        )
        ttk.Button(row_smart, text="Cube / Bulk Preset", command=self._apply_bulk_preset).grid(
            row=0, column=1, sticky=tk.EW
        )

        f_sv = ttk.LabelFrame(self.tab_tool, text="View File (view.json)", padding=8)
        f_sv.pack(fill=tk.X, pady=(0, 6))
        f_sv.columnconfigure(0, minsize=72)
        f_sv.columnconfigure(1, weight=1)

        row_vm = ttk.Frame(f_sv)
        row_vm.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        DarkRadiobutton(
            row_vm, text="Preset", variable=self.var_view_mode, value="preset", command=self._sync_view_mode
        ).pack(side=tk.LEFT)
        DarkRadiobutton(
            row_vm, text="view.json", variable=self.var_view_mode, value="file", command=self._sync_view_mode
        ).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(f_sv, text="Preset").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.combo_preset = ttk.Combobox(
            f_sv, textvariable=self.var_view_preset, values=list(VIEW_PRESETS), state="readonly"
        )
        self.combo_preset.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=(4, 0), pady=2)
        self.combo_preset.bind("<<ComboboxSelected>>", lambda _e: self._pick_view_preset(self.var_view_preset.get()))

        ttk.Label(f_sv, text="View JSON").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.entry_view_file = ttk.Entry(f_sv, textvariable=self.var_view_file)
        self.entry_view_file.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        self.btn_view_file = ttk.Button(f_sv, text="Browse...", command=self._browse_view_file)
        self.btn_view_file.grid(row=2, column=2, pady=2)

        self.chk_fit_view = DarkCheckbutton(
            f_sv, text="Re-fit view.json angle to structure (--fit-view)", variable=self.var_fit_view
        )
        self.chk_fit_view.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        self._sync_view_mode()

        # --- Selection Tab: Atom Labels, Opacity / Invert, Atom List ---
        self._build_selection_tab(self.tab_selection)

        self._switch_tab("view")

        # 3. Pinned Actions (always anchored at bottom of sidebar)
        act_host = action_parent if action_parent is not None else parent
        f_act = ttk.Frame(act_host, style="Outer.TFrame")
        if action_parent is not None:
            f_act.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(4, 6))
        else:
            f_act.pack(fill=tk.X, pady=(4, 4))
        self.btn_render = ttk.Button(
            f_act, text="Render High-Res Output", style="Accent.TButton", command=self._on_render
        )
        self.btn_render.pack(fill=tk.X, pady=(0, 4))

        row_sub = ttk.Frame(f_act, style="Outer.TFrame")
        row_sub.pack(fill=tk.X)
        row_sub.columnconfigure(0, weight=1, uniform="sub_act")
        row_sub.columnconfigure(1, weight=1, uniform="sub_act")
        self.btn_blender = ttk.Button(row_sub, text="Open in Blender", command=self._on_open_in_blender)
        self.btn_blender.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4))
        self.btn_folder = ttk.Button(row_sub, text="Open Output Folder", command=self._on_open_folder)
        self.btn_folder.grid(row=0, column=1, sticky=tk.EW)

    def _switch_tab(self, tab_name: str) -> None:
        self._active_tab = tab_name if tab_name in ("view", "tool", "selection") else "view"
        tabs = {"view": self.tab_view, "tool": self.tab_tool, "selection": self.tab_selection}
        for key, frm in tabs.items():
            if key == self._active_tab:
                frm.pack(fill=tk.X)
            else:
                frm.pack_forget()
        for key, (btn, bar) in self._tab_buttons.items():
            active = key == self._active_tab
            btn.configure(
                bg="#1E2128" if active else "#14161B",
                fg="#F8FAFC" if active else "#94A3B8",
                activebackground="#262A34",
                activeforeground="#FFFFFF",
            )
            bar.configure(bg="#14B8A6" if active else "#2E3440")

    def _build_selection_tab(self, parent: ttk.Frame) -> None:
        """Build the Selection tab with Atom Labels, Atom List, double-click hint, and opacity controls."""
        f_lbl = ttk.LabelFrame(parent, text="Labels & Selection", padding=8)
        f_lbl.pack(fill=tk.X, pady=(0, 6))

        row_top = ttk.Frame(f_lbl)
        row_top.pack(fill=tk.X, pady=(0, 6))
        DarkCheckbutton(
            row_top,
            text="Atom Labels (C1, H2, ...)",
            variable=self.var_show_labels,
            command=self._on_labels_toggled,
        ).pack(side=tk.LEFT)
        ttk.Label(row_top, text="Double-click atom in viewport", style="Muted.TLabel").pack(side=tk.RIGHT)

        row_btns = ttk.Frame(f_lbl)
        row_btns.pack(fill=tk.X)
        for c_idx, (text, cb) in enumerate((
            ("Select All", self._select_all_atoms),
            ("Clear", self._clear_selection),
            ("Invert", self._invert_selection),
            ("Same Elem", self._select_same_element),
        )):
            row_btns.columnconfigure(c_idx, weight=1, uniform="sel_btn")
            ttk.Button(row_btns, text=text, command=cb).grid(
                row=0, column=c_idx, sticky=tk.EW, padx=(0, 3 if c_idx < 3 else 0)
            )

        self.f_elem_chips = ttk.Frame(f_lbl)
        self.f_elem_chips.pack(fill=tk.X, pady=(6, 0))
        self._elem_chip_buttons: dict[str, tk.Button] = {}

        row_z = ttk.Frame(f_lbl)
        row_z.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row_z, text="Z (Å)").pack(side=tk.LEFT)
        ttk.Entry(row_z, textvariable=self.var_z_min, width=7).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(row_z, text="~").pack(side=tk.LEFT)
        ttk.Entry(row_z, textvariable=self.var_z_max, width=7).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(row_z, text="Select Z", command=self._select_by_z_range).pack(side=tk.LEFT)

        ttk.Label(
            f_lbl,
            textvariable=self.var_measurement,
            foreground="#2DD4BF",
            font=("Segoe UI", 9, "bold"),
            wraplength=330,
        ).pack(fill=tk.X, pady=(6, 0))

        f_op = ttk.LabelFrame(parent, text="Selection Transparency", padding=8)
        f_op.pack(fill=tk.X, pady=(0, 6))
        f_op.columnconfigure(0, minsize=72)
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
            text="Fade Unselected (선택 반대 투명화)",
            variable=self.var_alpha_invert,
            command=self._on_opacity_released,
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        f_list = ttk.LabelFrame(parent, text="Atom List", padding=6)
        f_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        list_wrap = ttk.Frame(f_list)
        list_wrap.pack(fill=tk.BOTH, expand=True)
        self.atom_listbox = tk.Listbox(
            list_wrap,
            height=14,
            bg="#121418",
            fg="#E5E7EB",
            selectbackground="#1E293B",
            selectforeground="#2DD4BF",
            activestyle="none",
            highlightbackground="#323844",
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
        """Build unified Avogadro-inspired Display Types panel with inline appearance options."""
        f_disp = ttk.LabelFrame(parent, text="Display Types", padding=(6, 4))
        f_disp.pack(fill=tk.X, pady=(0, 6))
        self.f_rep = f_disp

        rows = (
            ("ball_stick", "Ball and Stick", self.var_disp_ball_stick, lambda: self._toggle_section(self.f_rep_body)),
            ("licorice", "Licorice", self.var_disp_licorice, lambda: self._toggle_section(self.f_rep_body)),
            ("vdw", "Van der Waals", self.var_disp_vdw, lambda: self._toggle_section(self.f_rep_body)),
            ("crystal_lattice", "Crystal Lattice", self.var_show_cell, lambda: self._switch_tab("tool")),
            ("reference_axes", "Reference Axes", self.var_show_axes, self._toggle_axes_options),
        )
        for idx, (key, label_txt, var, dots_cb) in enumerate(rows):
            row_bg = "#1E2128" if idx % 2 == 0 else "#232730"
            row = tk.Frame(f_disp, bg=row_bg, padx=4, pady=2)
            row.pack(fill=tk.X)

            chk_canvas = tk.Canvas(row, width=22, height=22, bg=row_bg, highlightthickness=0, cursor="hand2")
            chk_canvas.pack(side=tk.LEFT, padx=(0, 6))
            self._display_checks[key] = (chk_canvas, var)

            lbl = tk.Label(
                row,
                text=label_txt,
                bg=row_bg,
                fg="#E5E7EB",
                font=("Segoe UI", 9),
                anchor=tk.W,
                cursor="hand2",
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            chk_canvas.bind("<Button-1>", lambda _e, k=key: self._on_display_type_clicked(k))
            lbl.bind("<Button-1>", lambda _e, k=key: self._on_display_type_clicked(k))

            if dots_cb is not None:
                btn = ttk.Button(row, text="•••", style="Dots.TButton", width=3, command=dots_cb)
                btn.pack(side=tk.RIGHT)

        self._refresh_display_checks()

        # Unified Style & Scale options (inline inside Display Types, toggled via •••)
        self.f_rep_body = ttk.Frame(f_disp, padding=(2, 6, 2, 2))
        self.f_rep_body.pack(fill=tk.X)
        self.f_rep_body.columnconfigure(0, minsize=72)
        self.f_rep_body.columnconfigure(1, weight=1)

        ttk.Label(self.f_rep_body, text="Material").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.combo_style = ttk.Combobox(
            self.f_rep_body, textvariable=self.var_style, values=self._available_styles, width=20
        )
        self.combo_style.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        self.combo_style.bind("<<ComboboxSelected>>", lambda _e: self._on_repr_changed())
        ttk.Button(self.f_rep_body, text="JSON...", command=self._browse_style).grid(row=0, column=2, pady=2)

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

        ttk.Label(self.f_rep_body, text="Bond Color").grid(row=2, column=0, sticky=tk.W, pady=2)
        cb_bcol = ttk.Combobox(
            self.f_rep_body,
            textvariable=self.var_bond_color,
            values=["style", "bicolor", "single"],
            state="readonly",
            width=12,
        )
        cb_bcol.grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)
        cb_bcol.bind("<<ComboboxSelected>>", lambda _e: self._on_repr_changed())

        self.f_axes_opts = tk.Frame(f_disp, bg="#181A20", padx=6, pady=4, highlightbackground="#2E3440", highlightthickness=1)
        self.f_axes_opts.pack(fill=tk.X, pady=(4, 0))
        tk.Label(self.f_axes_opts, text="Axes:", bg="#181A20", fg="#94A3B8", font=("Segoe UI", 8, "bold")).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        DarkCheckbutton(
            self.f_axes_opts,
            text="x, y, z Labels",
            variable=self.var_axes_labels,
            command=self._on_axes_option_changed,
            bg="#181A20",
        ).pack(side=tk.LEFT, padx=(0, 8))
        DarkCheckbutton(
            self.f_axes_opts,
            text="Fix (Default)",
            variable=self.var_axes_fixed,
            command=self._on_axes_fix_toggled,
            bg="#181A20",
        ).pack(side=tk.LEFT)
        self._axes_opts_visible = True

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

    def _sync_type_map_visibility(self) -> None:
        if not hasattr(self, "lbl_type_map"):
            return
        in_path = self.var_input.get().strip().lower()
        show_tm = in_path.endswith((".data", ".lmp")) or bool(self.var_type_map.get().strip())
        if show_tm:
            self.lbl_type_map.grid(row=2, column=0, sticky=tk.W, pady=2)
            self.entry_type_map.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
            self.btn_inspect_tm.grid(row=2, column=2, pady=2)
        else:
            self.lbl_type_map.grid_remove()
            self.entry_type_map.grid_remove()
            self.btn_inspect_tm.grid_remove()

    def _toggle_section(self, frame: ttk.Frame) -> None:
        if frame.winfo_manager():
            frame.pack_forget()
        elif frame is getattr(self, "f_rep_body", None) and hasattr(self, "f_axes_opts"):
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

    def _draw_rounded_checkbox(self, canvas: tk.Canvas, checked: bool) -> None:
        canvas.delete("all")
        fill = "#0D9488" if checked else "#14161A"
        outline = "#14B8A6" if checked else "#64748B"
        canvas.create_rectangle(3, 3, 18, 18, fill=fill, outline=outline, width=1.4)
        if checked:
            canvas.create_line(
                6, 11, 9.5, 14.5, 15, 7, fill="#FFFFFF", width=2.0, capstyle=tk.ROUND, joinstyle=tk.ROUND
            )

    def _refresh_display_checks(self) -> None:
        for canvas, var in self._display_checks.values():
            self._draw_rounded_checkbox(canvas, bool(var.get()))

    def _on_display_type_clicked(self, key: str) -> None:
        if key in ("ball_stick", "licorice", "vdw"):
            var_map = {
                "ball_stick": self.var_disp_ball_stick,
                "licorice": self.var_disp_licorice,
                "vdw": self.var_disp_vdw,
            }
            new_state = not bool(var_map[key].get())
            if new_state:
                for k_item, v_item in var_map.items():
                    v_item.set(k_item == key)
                self._show_atoms = True
                if key == "ball_stick":
                    self._set_representation(0.78, "auto", "bicolor")
                elif key == "licorice":
                    self._set_representation(0.42, "auto", "bicolor")
                else:
                    self._set_representation(1.35, "none", "style")
            else:
                var_map[key].set(False)
                self._show_atoms = False
                self._refresh_display_checks()
                if self._vp_scene is not None:
                    self._vp_scene.show_atoms = False
                    self.var_canvas_mode.set("3d")
                    self._redraw_canvas()
            return

        if key == "crystal_lattice":
            self.var_show_cell.set(not bool(self.var_show_cell.get()))
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
            from_tmp_preview = (
                self._preview_path is not None
                and hasattr(self, "_preview_tmp_dir")
                and str(self._preview_path).startswith(str(self._preview_tmp_dir.name))
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

        f_prev = ttk.LabelFrame(v_pane, text="Viewport & Render Preview", padding=4)
        tb = ttk.Frame(f_prev)
        tb.pack(fill=tk.X, pady=(0, 4))
        DarkRadiobutton(
            tb, text="3D Interactive", variable=self.var_canvas_mode, value="3d", command=self._redraw_canvas
        ).pack(side=tk.LEFT)
        DarkRadiobutton(
            tb, text="Rendered PNG", variable=self.var_canvas_mode, value="png", command=self._redraw_canvas
        ).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Button(tb, text="Reset / Fit View", command=self._on_reset_view).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tb, text="Save view.json...", command=self._on_save_view_json).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tb, text="Copy Image", command=self._on_copy_image).pack(side=tk.LEFT, padx=(0, 8))
        DarkCheckbutton(
            tb, text="Dark Backdrop", variable=self.var_dark_backdrop, command=self._on_backdrop_toggled
        ).pack(side=tk.LEFT)
        ttk.Label(
            tb,
            text="Drag: Rotate/Axes | R: Reset | L: Labels | Ctrl+C: Copy",
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(f_prev, bg="#15171C", highlightthickness=0)
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

        f_log = ttk.LabelFrame(v_pane, text="Console Log", padding=4)
        self.log_text = scrolledtext.ScrolledText(
            f_log,
            height=5,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg="#121418",
            fg="#CBD5E1",
            insertbackground="#E5E7EB",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        v_pane.add(f_prev, weight=6)
        v_pane.add(f_log, weight=1)

    def _on_scale_changed(self, _val: str) -> None:
        self.var_radius_label.set(f"{self.var_radius_scale.get():.2f}×")

    def _on_labels_toggled(self) -> None:
        if self._vp_scene is not None:
            self._vp_scene.show_labels = bool(self.var_show_labels.get())
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
        self.var_measurement.set(compute_selection_measurement(syms, pos, self._selection_order))

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
            self._elem_chip_buttons[sym] = btn
        self._refresh_element_chip_styles()

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
                bg, fg, active = "#2B303B", "#CBD5E1", "#363C49"
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
                self.atom_listbox.itemconfig(idx, fg="#2DD4BF", bg="#1B2832")
            else:
                self.atom_listbox.itemconfig(idx, fg="#E5E7EB", bg="#121418")
        self.atom_listbox.yview_moveto(top_frac)

    def _sync_selection_and_alphas(self, trigger_blender: bool = False) -> None:
        if self._structure_cache is None:
            self._refresh_atom_list()
            self._refresh_element_chip_styles()
            self._update_measurement_banner()
            return
        syms = self._structure_cache.get("symbols", [])
        n = len(syms)
        prev_alphas = self._structure_cache.get("alphas")
        alphas = compute_atom_alphas(
            n_atoms=n,
            selected=self._selected_atoms,
            opacity=float(self.var_sel_opacity.get()),
            invert=bool(self.var_alpha_invert.get()),
        )
        alphas_changed = prev_alphas != alphas
        self._structure_cache["alphas"] = alphas
        if alphas_changed:
            self._written_struct_version = -1

        if self._vp_scene is not None:
            self._vp_scene.atom_alphas = np.asarray(alphas, dtype=np.float64)
            self._vp_scene.selected_atoms = set(self._selected_atoms)
        self._refresh_atom_list()
        self._refresh_element_chip_styles()
        self._update_measurement_banner()
        self._redraw_canvas()
        if trigger_blender and (alphas_changed or any(a < 0.995 for a in alphas)):
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
        if self._inspect_after_id is not None:
            self.after_cancel(self._inspect_after_id)
        self._inspect_after_id = self.after(150, self._run_inspect_async)

    def _run_inspect_async(self) -> None:
        self._inspect_after_id = None
        path = self.var_input.get().strip()
        if not path:
            self.var_inspector.set("Select an atomic structure file (.cif, POSCAR, .xyz, .data, .json)")
            return
        type_map = self.var_type_map.get()
        rep = (int(self.var_rep_x.get()), int(self.var_rep_y.get()), int(self.var_rep_z.get()))
        wrap = bool(self.var_wrap.get())

        def worker() -> None:
            try:
                p = Path(path).expanduser()
                if not p.is_file():
                    self._queue.put(("inspect", f"File not found: {p.name}"))
                    return
                if p.suffix.lower() == ".json":
                    neutral = json.loads(p.read_text(encoding="utf-8"))
                    eff_rep = rep if neutral.get("has_cell", False) else (1, 1, 1)
                    if eff_rep != (1, 1, 1):
                        atoms = read_structure(p, type_map=type_map.strip() or None)
                        neutral = atoms_to_neutral_dict(atoms, source=str(p), replicate=eff_rep, wrap=wrap)
                else:
                    atoms = read_structure(p, type_map=type_map.strip() or None)
                    eff_rep = rep if atoms.cell.rank == 3 else (1, 1, 1)
                    neutral = atoms_to_neutral_dict(atoms, source=str(p), replicate=eff_rep, wrap=wrap)
                summary = inspect_structure_summary(path, type_map=type_map, replicate=eff_rep)
                self._queue.put(("structure", (summary, neutral)))
            except Exception as exc:
                self._queue.put(("inspect", f"Inspect error: {exc}"))

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
            input_path=self.var_input.get(),
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

    def _run_cli_async(self, cli_args: list[str], mode: str, target_path: Path) -> None:
        self._task_start = time.monotonic()
        self._set_busy(True, f"Running {mode}...")
        self._append_log(f"$ atomic-blender {' '.join(cli_args)}")

        def worker() -> None:
            cmd = [sys.executable, "-m", "atomic_blender.cli", *cli_args]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                self._queue.put(("log", raw_line))
            rc = proc.wait()
            self._queue.put(("done", (mode, rc, target_path)))

        threading.Thread(target=worker, daemon=True).start()

    def _effective_cfg_for_cli(self, cfg: GuiRenderConfig) -> GuiRenderConfig:
        """Materialize interactive camera view.json and per-atom transparency structure JSON if active."""
        eff_input = cfg.input_path
        eff_rep = cfg.replicate
        if self._structure_cache is not None:
            alphas = self._structure_cache.get("alphas")
            if isinstance(alphas, list) and any(a < 0.995 for a in alphas):
                tmp_sfd, tmp_sname = tempfile.mkstemp(prefix="atomic_sel_", suffix=".json")
                os.close(tmp_sfd)
                tmp_spath = Path(tmp_sname)
                tmp_spath.write_text(json.dumps(self._structure_cache), encoding="utf-8")
                self._temp_view_files.append(tmp_spath)
                eff_input = str(tmp_spath)
                eff_rep = (1, 1, 1)

        if self._vp_scene is None or not self._use_interactive_cam:
            if eff_input == cfg.input_path:
                return cfg
            return GuiRenderConfig(
                input_path=eff_input,
                output_path=cfg.output_path,
                style=cfg.style,
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
            style=cfg.style,
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
        target = self.var_output.get().strip() or self.var_input.get().strip()
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
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return {}

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
        prev_w, prev_h = compute_preview_resolution(int(self.var_res_w.get()), int(self.var_res_h.get()), max_side=400)
        prev_samples = min(4, max(2, int(self.var_samples.get())))

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
                self.var_bonds.get(),
                self.var_bond_color.get(),
                round(float(self.var_radius_scale.get()), 4),
                bool(self.var_show_cell.get()),
                crop_z,
                bool(self.var_fit_view.get()),
                [round(float(a), 3) for a in self._structure_cache.get("alphas", [])],
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
                "style": str(style_path),
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

    def _draw_3d_viewport(self, interactive: bool | None = None) -> None:
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if self._vp_scene is None:
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="Drag & drop a structure file (or view.json / style.json) anywhere on this window\nor click 'Browse...' to preview in interactive 3D",
                justify=tk.CENTER,
                fill="#94A3B8" if self.var_dark_backdrop.get() else "#555555",
            )
            return

        vp = self._vp_scene
        res_w = max(16, int(self.var_res_w.get()))
        res_h = max(16, int(self.var_res_h.get()))
        is_interactive = self._dragging if interactive is None else interactive

        if _HAS_PIL:
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
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="Drag & drop a structure file (or view.json / style.json) anywhere on this window\nor click 'Browse...' and 'Render High-Res Output'",
                justify=tk.CENTER,
                fill="#94A3B8" if dark_bg else "#555555",
            )
            return

        if _HAS_PIL:
            iw, ih = self._pil_image.size
            scale = min((cw - 14) / max(1, iw), (ch - 14) / max(1, ih))
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            resized = self._pil_image.resize((nw, nh), Image.Resampling.LANCZOS).copy()
            if self._vp_scene is not None and (self.var_show_labels.get() or self._selected_atoms):
                render_atom_labels_and_selection_overlay(
                    resized,
                    self._vp_scene,
                    max(16, int(self.var_res_w.get())),
                    max(16, int(self.var_res_h.get())),
                    draw_labels=bool(self.var_show_labels.get()),
                    draw_selection=bool(self._selected_atoms),
                )
            if self.var_show_axes.get():
                quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                render_reference_axes_overlay(
                    resized,
                    quat,
                    axes_pos=self._axes_pos,
                    show_labels=bool(self.var_axes_labels.get()),
                    dark_backdrop=dark_bg,
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
        for aid_attr in ("_poll_after_id", "_inspect_after_id", "_hq_after_id"):
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
                self.var_inspector.set(str(payload))
            elif kind == "structure":
                summary, neutral = payload
                self.var_inspector.set(str(summary))
                self._selected_atoms.clear()
                self._selection_order.clear()
                self._structure_cache = neutral
                self._structure_version += 1
                self._use_interactive_cam = False
                in_raw = self.var_input.get().strip()
                if in_raw:
                    self._add_recent_file(in_raw)
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
                    self._load_preview(target_path)
                    if self.var_show_axes.get() and _HAS_PIL and target_path.is_file():
                        try:
                            out_img = Image.open(target_path).convert("RGBA")
                            quat = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                            render_reference_axes_overlay(
                                out_img,
                                quat,
                                axes_pos=self._axes_pos,
                                show_labels=bool(self.var_axes_labels.get()),
                                dark_backdrop=bool(self.var_dark_backdrop.get()),
                            )
                            out_img.save(target_path)
                        except Exception as exc:
                            self._append_log(f"Axes overlay warning: {exc}")
                elif mode == "open_blender":
                    self._set_busy(False, f"Opening {target_path.name} in Blender...")
                    subprocess.Popen([self._pending_blender_exe, str(target_path)])
        if not getattr(self, "_destroyed", False):
            self._poll_after_id = self.after(100, self._poll_queue)

    def handle_dropped_files(self, paths: list[str]) -> None:
        """Apply dropped structure, view.json, or style.json files to the GUI state."""
        for raw_path in paths:
            p = Path(raw_path).expanduser()
            if not p.is_file():
                continue
            kind = classify_dropped_file(p)
            if kind == "view":
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
    app = AtomicBlenderApp(initial_input=initial_input)
    app.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atomic-blender-gui", description="Launch Atomic Blender desktop GUI")
    parser.add_argument("input", nargs="?", default=None, help="Optional structure file to prefill")
    args = parser.parse_args(argv)
    return launch_gui(initial_input=args.input)


if __name__ == "__main__":
    raise SystemExit(main())
