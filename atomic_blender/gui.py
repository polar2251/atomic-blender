"""Standalone desktop GUI client for atomic-blender built on Python tkinter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from atomic_blender.cli import _collect_style_files, find_blender_executable
from atomic_blender.convert import read_structure
from atomic_blender.elements import VIEW_PRESETS

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
        n_atoms = len(syms) * max(1, replicate[0]) * max(1, replicate[1]) * max(1, replicate[2])
        if data.get("has_cell", False):
            import math

            vecs = data.get("cell", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            lens = [math.sqrt(sum(c * c for c in v)) for v in vecs]
            return (
                f"{p.name} | {n_atoms} atoms ({elems}) | "
                f"cell {lens[0]*replicate[0]:.2f}×{lens[1]*replicate[1]:.2f}×{lens[2]*replicate[2]:.2f} Å"
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


class AtomicBlenderApp(tk.Tk):
    """Desktop GUI client for atomic-blender."""

    def __init__(self, initial_input: str | None = None) -> None:
        super().__init__()
        self.title("Atomic Blender Studio — Desktop Client")
        self.geometry("1240x820")
        self.minsize(1000, 680)

        self._configure_theme()

        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy = False
        self._task_start = 0.0
        self._inspect_after_id: str | None = None
        self._preview_path: Path | None = None
        self._pil_image: Any = None
        self._photo_ref: Any = None

        self._init_variables(initial_input)
        self._build_layout()
        self.after(100, self._poll_queue)
        if initial_input:
            self._schedule_inspect()

    def _configure_theme(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

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

        self.var_res_w = tk.IntVar(value=1400)
        self.var_res_h = tk.IntVar(value=1400)
        self.var_samples = tk.IntVar(value=64)
        self.var_blender = tk.StringVar(value=detected_blender)
        self.var_status = tk.StringVar(value="Ready")

    def _build_layout(self) -> None:
        banner = ttk.Frame(self, padding=(10, 6, 10, 2))
        banner.pack(fill=tk.X)
        ttk.Label(banner, text="Structure:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(banner, textvariable=self.var_inspector, foreground="#1E40AF").pack(side=tk.LEFT, padx=(6, 0))

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=6)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_controls(left)
        self._build_preview_and_log(right)

        status_bar = ttk.Frame(self, padding=(8, 2, 8, 6))
        status_bar.pack(fill=tk.X)
        ttk.Label(status_bar, textvariable=self.var_status).pack(side=tk.LEFT)

    def _build_controls(self, parent: ttk.Frame) -> None:
        # 1. Input & Output
        f_io = ttk.LabelFrame(parent, text="Structure & Output", padding=8)
        f_io.pack(fill=tk.X, pady=(0, 6))
        f_io.columnconfigure(1, weight=1)

        ttk.Label(f_io, text="Input").grid(row=0, column=0, sticky=tk.W, pady=2)
        entry_in = ttk.Entry(f_io, textvariable=self.var_input, width=34)
        entry_in.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        entry_in.bind("<FocusOut>", lambda _e: self._schedule_inspect())
        ttk.Button(f_io, text="Browse...", command=self._browse_input).grid(row=0, column=2, pady=2)

        ttk.Label(f_io, text="Output PNG").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(f_io, textvariable=self.var_output, width=34).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(f_io, text="Save As...", command=self._browse_output).grid(row=1, column=2, pady=2)

        ttk.Label(f_io, text="Type Map").grid(row=2, column=0, sticky=tk.W, pady=2)
        entry_tm = ttk.Entry(f_io, textvariable=self.var_type_map)
        entry_tm.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        entry_tm.bind("<FocusOut>", lambda _e: self._schedule_inspect())
        ttk.Button(f_io, text="Inspect", command=self._schedule_inspect).grid(row=2, column=2, pady=2)

        row_smart = ttk.Frame(f_io)
        row_smart.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(4, 0))
        ttk.Button(row_smart, text="Smart Slab Preset", command=self._apply_slab_preset).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(row_smart, text="Cube / Bulk Preset", command=self._apply_bulk_preset).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # 2. Style & Representation
        f_rep = ttk.LabelFrame(parent, text="Style & Representation", padding=8)
        f_rep.pack(fill=tk.X, pady=(0, 6))
        f_rep.columnconfigure(1, weight=1)

        ttk.Label(f_rep, text="Style").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.combo_style = ttk.Combobox(f_rep, textvariable=self.var_style, values=self._available_styles, width=24)
        self.combo_style.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(f_rep, text="JSON...", command=self._browse_style).grid(row=0, column=2, pady=2)

        ttk.Label(f_rep, text="Atom Scale").grid(row=1, column=0, sticky=tk.W, pady=2)
        row_sc = ttk.Frame(f_rep)
        row_sc.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        ttk.Scale(
            row_sc, from_=0.4, to=1.6, variable=self.var_radius_scale, command=self._on_scale_changed
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row_sc, textvariable=self.var_radius_label, width=6).pack(side=tk.LEFT, padx=(6, 0))

        row_repr_btns = ttk.Frame(f_rep)
        row_repr_btns.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=(0, 4))
        for lbl, rscale, bmode, bcol in (
            ("Ball & Stick", 0.78, "auto", "bicolor"),
            ("Default", 1.00, "style", "style"),
            ("Spacefill", 1.35, "none", "style"),
        ):
            ttk.Button(
                row_repr_btns,
                text=lbl,
                command=lambda rs=rscale, bm=bmode, bc=bcol: self._set_representation(rs, bm, bc),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        ttk.Label(f_rep, text="Bonds").grid(row=3, column=0, sticky=tk.W, pady=2)
        row_b = ttk.Frame(f_rep)
        row_b.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        ttk.Combobox(
            row_b, textvariable=self.var_bonds, values=["style", "table", "auto", "none"], state="readonly", width=8
        ).pack(side=tk.LEFT)
        ttk.Label(row_b, text="Color").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Combobox(
            row_b, textvariable=self.var_bond_color, values=["style", "bicolor", "single"], state="readonly", width=9
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(row_b, text="Show Cell", variable=self.var_show_cell).pack(side=tk.LEFT, padx=(8, 0))

        # 3. Camera View
        f_sv = ttk.LabelFrame(parent, text="Camera View", padding=8)
        f_sv.pack(fill=tk.X, pady=(0, 6))
        f_sv.columnconfigure(1, weight=1)

        row_vm = ttk.Frame(f_sv)
        row_vm.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Radiobutton(
            row_vm, text="Preset", variable=self.var_view_mode, value="preset", command=self._sync_view_mode
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            row_vm, text="view.json", variable=self.var_view_mode, value="file", command=self._sync_view_mode
        ).pack(side=tk.LEFT, padx=(10, 0))

        row_vp = ttk.Frame(f_sv)
        row_vp.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=2)
        for lbl, pname in (("Iso", "iso"), ("Side", "side-oblique"), ("a", "a"), ("b", "b"), ("c", "c")):
            ttk.Button(
                row_vp, text=lbl, width=5, command=lambda p=pname: self._pick_view_preset(p)
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        ttk.Label(f_sv, text="Preset").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.combo_preset = ttk.Combobox(
            f_sv, textvariable=self.var_view_preset, values=list(VIEW_PRESETS), state="readonly"
        )
        self.combo_preset.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=(4, 0), pady=2)

        ttk.Label(f_sv, text="View JSON").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.entry_view_file = ttk.Entry(f_sv, textvariable=self.var_view_file)
        self.entry_view_file.grid(row=3, column=1, sticky=tk.EW, padx=4, pady=2)
        self.btn_view_file = ttk.Button(f_sv, text="Browse...", command=self._browse_view_file)
        self.btn_view_file.grid(row=3, column=2, pady=2)

        self.chk_fit_view = ttk.Checkbutton(
            f_sv, text="Re-fit view.json angle to structure (--fit-view)", variable=self.var_fit_view
        )
        self.chk_fit_view.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        self._sync_view_mode()

        # 4. Supercell & Crop Z
        f_geom = ttk.LabelFrame(parent, text="Supercell & Framing", padding=8)
        f_geom.pack(fill=tk.X, pady=(0, 6))
        f_geom.columnconfigure(1, weight=1)

        ttk.Label(f_geom, text="Replicate").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_rep = ttk.Frame(f_geom)
        row_rep.grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        for var in (self.var_rep_x, self.var_rep_y, self.var_rep_z):
            sb = ttk.Spinbox(row_rep, from_=1, to=20, textvariable=var, width=4, command=self._schedule_inspect)
            sb.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(row_rep, text="Wrap into cell", variable=self.var_wrap).pack(side=tk.LEFT, padx=(6, 0))

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

        # 5. Render Settings
        f_ren = ttk.LabelFrame(parent, text="Cycles Render", padding=8)
        f_ren.pack(fill=tk.X, pady=(0, 8))
        f_ren.columnconfigure(1, weight=1)

        ttk.Label(f_ren, text="Resolution").grid(row=0, column=0, sticky=tk.W, pady=2)
        row_res = ttk.Frame(f_ren)
        row_res.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
        ttk.Spinbox(row_res, from_=64, to=8192, increment=100, textvariable=self.var_res_w, width=6).pack(side=tk.LEFT)
        ttk.Label(row_res, text="×").pack(side=tk.LEFT, padx=4)
        ttk.Spinbox(row_res, from_=64, to=8192, increment=100, textvariable=self.var_res_h, width=6).pack(side=tk.LEFT)
        for label, w, h in (("700²", 700, 700), ("1400²", 1400, 1400), ("Slab", 1400, 1200)):
            ttk.Button(
                row_res, text=label, width=5, command=lambda ww=w, hh=h: self._set_resolution(ww, hh)
            ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(f_ren, text="Samples").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(f_ren, from_=1, to=2048, textvariable=self.var_samples, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=4, pady=2
        )

        ttk.Label(f_ren, text="Blender").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(f_ren, textvariable=self.var_blender).grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Button(f_ren, text="Browse...", command=self._browse_blender).grid(row=2, column=2, pady=2)

        # 6. Actions
        f_act = ttk.Frame(parent)
        f_act.pack(fill=tk.X)
        self.btn_render = ttk.Button(f_act, text="Render & Preview", command=self._on_render)
        self.btn_render.pack(fill=tk.X, pady=(0, 4))

        row_sub = ttk.Frame(f_act)
        row_sub.pack(fill=tk.X)
        self.btn_blender = ttk.Button(row_sub, text="Open in Blender", command=self._on_open_in_blender)
        self.btn_blender.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_folder = ttk.Button(row_sub, text="Open Output Folder", command=self._on_open_folder)
        self.btn_folder.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_preview_and_log(self, parent: ttk.Frame) -> None:
        v_pane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        v_pane.pack(fill=tk.BOTH, expand=True)

        f_prev = ttk.LabelFrame(v_pane, text="Render Preview", padding=4)
        self.canvas = tk.Canvas(f_prev, bg="#E8E8E8", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_preview())

        f_log = ttk.LabelFrame(v_pane, text="Log", padding=4)
        self.log_text = scrolledtext.ScrolledText(f_log, height=9, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        v_pane.add(f_prev, weight=4)
        v_pane.add(f_log, weight=1)

    def _on_scale_changed(self, _val: str) -> None:
        self.var_radius_label.set(f"{self.var_radius_scale.get():.2f}×")

    def _set_representation(self, radius_scale: float, bonds: str, bond_color: str) -> None:
        self.var_radius_scale.set(radius_scale)
        self.var_radius_label.set(f"{radius_scale:.2f}×")
        self.var_bonds.set(bonds)
        self.var_bond_color.set(bond_color)

    def _apply_slab_preset(self) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set("side-oblique")
        self.var_crop_z_mode.set("auto")
        self._sync_crop_z()
        self._set_resolution(1400, 1200)

    def _apply_bulk_preset(self) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set("iso")
        self.var_crop_z_mode.set("none")
        self._sync_crop_z()
        self._set_resolution(1400, 1400)

    def _pick_view_preset(self, preset: str) -> None:
        self.var_view_mode.set("preset")
        self._sync_view_mode()
        self.var_view_preset.set(preset)

    def _sync_view_mode(self) -> None:
        is_file = self.var_view_mode.get() == "file"
        self.combo_preset.configure(state="disabled" if is_file else "readonly")
        state_file = "normal" if is_file else "disabled"
        self.entry_view_file.configure(state=state_file)
        self.btn_view_file.configure(state=state_file)
        self.chk_fit_view.configure(state=state_file)

    def _sync_crop_z(self) -> None:
        is_custom = self.var_crop_z_mode.get() == "custom"
        self.entry_crop_z.configure(state="normal" if is_custom else "disabled")

    def _set_resolution(self, w: int, h: int) -> None:
        self.var_res_w.set(w)
        self.var_res_h.set(h)

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

        def worker() -> None:
            try:
                summary = inspect_structure_summary(path, type_map=type_map, replicate=rep)
            except Exception as exc:
                summary = f"Inspect error: {exc}"
            self._queue.put(("inspect", summary))

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

    def _browse_view_file(self) -> None:
        path = filedialog.askopenfilename(title="Select view.json", filetypes=[("JSON files", "*.json")])
        if path:
            self.var_view_file.set(path)

    def _browse_blender(self) -> None:
        path = filedialog.askopenfilename(title="Select Blender Executable")
        if path:
            self.var_blender.set(path)

    def _collect_config(self) -> GuiRenderConfig:
        return GuiRenderConfig(
            input_path=self.var_input.get(),
            output_path=self.var_output.get(),
            style=self.var_style.get(),
            view_mode=self.var_view_mode.get(),
            view_preset=self.var_view_preset.get(),
            view_file=self.var_view_file.get(),
            fit_view=bool(self.var_fit_view.get()),
            type_map=self.var_type_map.get(),
            replicate=(int(self.var_rep_x.get()), int(self.var_rep_y.get()), int(self.var_rep_z.get())),
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

    def _on_render(self) -> None:
        if self._busy:
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        out_png = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        self.var_output.set(str(out_png))
        cli_args = build_render_cli_args(cfg, out_png=out_png)
        self._run_cli_async(cli_args, mode="render", target_path=out_png)

    def _on_open_in_blender(self) -> None:
        if self._busy:
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            blender_exe = find_blender_executable(cfg.blender_exe.strip() or None)
        except Exception as exc:
            messagebox.showerror("Cannot Open in Blender", str(exc))
            return

        base = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path
        blend_path = base.with_suffix(".blend")
        cli_args = build_render_cli_args(cfg, out_png=None, save_blend=blend_path)
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
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        tile = 20
        for y in range(0, ch, tile):
            for x in range(0, cw, tile):
                color = "#F3F3F3" if ((x // tile + y // tile) % 2 == 0) else "#E2E2E2"
                self.canvas.create_rectangle(x, y, x + tile, y + tile, fill=color, outline="")

        if self._pil_image is None:
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="Select a structure and click 'Render & Preview'",
                fill="#555555",
            )
            return

        if _HAS_PIL:
            iw, ih = self._pil_image.size
            scale = min((cw - 16) / max(1, iw), (ch - 16) / max(1, ih))
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            resized = self._pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(resized)
        else:
            iw = self._pil_image.width()
            ih = self._pil_image.height()
            factor = max(1, int(max(iw / max(1, cw - 16), ih / max(1, ch - 16)) + 0.999))
            self._photo_ref = self._pil_image.subsample(factor, factor)

        self.canvas.create_image(cw // 2, ch // 2, image=self._photo_ref, anchor=tk.CENTER)

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "inspect":
                self.var_inspector.set(str(payload))
            elif kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                mode, rc, target_path = payload
                elapsed = time.monotonic() - self._task_start
                if rc != 0:
                    self._set_busy(False, f"Failed (exit {rc}) after {elapsed:.1f}s")
                    messagebox.showerror("Render Failed", f"Command exited with status {rc}. Check Log for details.")
                elif mode == "render":
                    self._set_busy(False, f"Rendered {target_path.name} in {elapsed:.1f}s")
                    self._load_preview(target_path)
                elif mode == "open_blender":
                    self._set_busy(False, f"Opening {target_path.name} in Blender...")
                    subprocess.Popen([self._pending_blender_exe, str(target_path)])
        self.after(100, self._poll_queue)


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
