"""UI-agnostic configuration, presets, CLI argument building, and structure inspection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from atomic_blender.viewport3d import compute_cell_parameters

JOURNAL_PRESETS: dict[str, tuple[int, int, int]] = {
    "1400×1400": (1400, 1400, 300),
    "700×700": (700, 700, 300),
    "1400×1200": (1400, 1200, 300),
    "1920×1080": (1920, 1080, 300),
    "Nature 1-Col (300 DPI)": (1051, 1051, 300),
    "Nature 2-Col (300 DPI)": (2161, 1440, 300),
    "ACS 1-Col (600 DPI)": (2008, 2008, 600),
    "ACS 2-Col (600 DPI)": (4205, 2400, 600),
}


def embed_png_dpi(png_path: Path, dpi: int = 300) -> bool:
    """Embed physical ``dpi`` metadata into an exported PNG image when Pillow is available."""
    p = Path(png_path)
    if not p.is_file():
        return False
    target_dpi = max(72, int(dpi))
    try:
        from PIL import Image

        with Image.open(p) as im:
            im.load()
            im.save(p, format="PNG", dpi=(target_dpi, target_dpi))
        return True
    except Exception:
        return False


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

    # Defer importing convert (and ASE) until reading non-JSON structure files.
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
