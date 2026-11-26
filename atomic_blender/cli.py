"""Headless CLI for atomic-blender."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from atomic_blender.convert import convert_file

DEFAULT_BLENDER_PATHS = [
    os.environ.get("BLENDER_EXE", ""),
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
]


def package_root() -> Path:
    """Return repository/package root directory containing ``styles/`` and ``addon/``."""
    return Path(__file__).resolve().parent.parent


def resolve_style_path(style_arg: str) -> Path:
    """Resolve a style name (e.g. ``default``) or explicit ``.json`` path."""
    candidate = Path(style_arg)
    if candidate.is_file():
        return candidate.resolve()
    root = package_root()
    for path in (
        root / "styles" / f"{style_arg}.json",
        root / "styles" / style_arg,
    ):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Style not found: {style_arg!r}")


def find_blender_executable(explicit: str | None = None) -> Path:
    """Locate the Blender executable from CLI flag, env var, PATH, or default install paths."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        found = shutil.which(explicit)
        if found:
            return Path(found)
        raise FileNotFoundError(f"Blender executable not found: {explicit}")

    for item in DEFAULT_BLENDER_PATHS:
        if item and Path(item).is_file():
            return Path(item)

    found = shutil.which("blender")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "Could not locate Blender executable. Pass --blender <path> or set BLENDER_EXE."
    )


def cmd_render(args: argparse.Namespace) -> int:
    style_path = resolve_style_path(args.style)
    blender_exe = find_blender_executable(args.blender)
    scene_script = Path(__file__).resolve().parent / "scene_builder.py"

    with tempfile.TemporaryDirectory(prefix="atomic_blender_") as tmpdir:
        tmp_path = Path(tmpdir)
        if args.input.suffix.lower() == ".json" and args.replicate == [1, 1, 1] and not args.format:
            neutral_json = args.input.resolve()
        else:
            neutral_json = tmp_path / "structure.json"
            convert_file(
                input_path=args.input,
                output_json=neutral_json,
                fmt=args.format,
                type_map=args.type_map,
                lammps_style=args.lammps_style,
                replicate=tuple(args.replicate),
                wrap=not args.no_wrap,
                index=int(args.index) if args.index.lstrip("-").isdigit() else args.index,
            )

        blender_args = [
            str(blender_exe),
            "--background",
            "--factory-startup",
            "--python",
            str(scene_script),
            "--",
            "--structure",
            str(neutral_json),
            "--style",
            str(style_path),
            "--view",
            args.view,
        ]
        if args.out is not None:
            blender_args.extend(["--out", str(args.out.resolve())])
        if args.save_blend is not None:
            blender_args.extend(["--save-blend", str(args.save_blend.resolve())])
        if args.crop_z is not None:
            blender_args.extend(["--crop-z", args.crop_z])
        if args.resolution is not None:
            blender_args.extend(["--resolution", str(args.resolution[0]), str(args.resolution[1])])
        if args.samples is not None:
            blender_args.extend(["--samples", str(args.samples)])
        if args.bonds is not None:
            blender_args.extend(["--bonds", args.bonds])
        if args.fit_view:
            blender_args.append("--fit-view")

        # Ensure Blender Python can import atomic_blender.*
        env = os.environ.copy()
        root_str = str(package_root())
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pp}" if existing_pp else root_str

        proc = subprocess.run(blender_args, env=env)
        return proc.returncode


def cmd_convert(args: argparse.Namespace) -> int:
    convert_file(
        input_path=args.input,
        output_json=args.out,
        fmt=args.format,
        type_map=args.type_map,
        lammps_style=args.lammps_style,
        replicate=tuple(args.replicate),
        wrap=not args.no_wrap,
        index=int(args.index) if args.index.lstrip("-").isdigit() else args.index,
    )
    print(f"Wrote neutral JSON -> {args.out}")
    return 0


def cmd_styles(_args: argparse.Namespace) -> int:
    styles_dir = package_root() / "styles"
    if not styles_dir.is_dir():
        print("No styles directory found.")
        return 1
    for p in sorted(styles_dir.glob("*.json")):
        print(f"{p.stem}: {p}")
    return 0


def cmd_install_addon(args: argparse.Namespace) -> int:
    """Install self-contained ``atomic_blender_studio`` addon into Blender's user addons dir."""
    blender_exe = find_blender_executable(args.blender)
    if args.addons_dir is not None:
        addons_dir = args.addons_dir
    else:
        probe = subprocess.run(
            [
                str(blender_exe),
                "--background",
                "--factory-startup",
                "--python-expr",
                "import bpy; print('ADDONS_DIR=' + bpy.utils.user_resource('SCRIPTS', path='addons'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        addons_dir = None
        for line in probe.stdout.splitlines():
            if line.startswith("ADDONS_DIR="):
                addons_dir = Path(line.split("=", 1)[1].strip())
                break
        if addons_dir is None:
            raise RuntimeError("Could not detect Blender user addons directory.")

    target_dir = addons_dir / "atomic_blender_studio"
    target_dir.mkdir(parents=True, exist_ok=True)

    root = package_root()
    shutil.copy2(root / "addon" / "atomic_blender_studio" / "__init__.py", target_dir / "__init__.py")
    shutil.copy2(root / "atomic_blender" / "elements.py", target_dir / "elements.py")
    shutil.copy2(root / "atomic_blender" / "scene_builder.py", target_dir / "scene_builder.py")
    styles_target = target_dir / "styles"
    styles_target.mkdir(exist_ok=True)
    for sp in (root / "styles").glob("*.json"):
        shutil.copy2(sp, styles_target / sp.name)

    # Record current system Python & repo path so GUI addon can convert non-JSON files seamlessly
    config_file = target_dir / "config.json"
    import json

    config_file.write_text(
        json.dumps(
            {
                "python_exe": sys.executable,
                "repo_root": str(root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_enable:
        subprocess.run(
            [
                str(blender_exe),
                "--background",
                "--python-expr",
                "import bpy; bpy.ops.preferences.addon_enable(module='atomic_blender_studio'); bpy.ops.wm.save_userpref()",
            ],
            check=True,
        )
    print(f"Installed addon 'atomic_blender_studio' -> {target_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomic-blender", description="Publication atomic renderer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ren = sub.add_parser("render", help="Convert structure and render headlessly in Blender")
    p_ren.add_argument("input", type=Path, help="Input structure file or neutral JSON")
    p_ren.add_argument("--style", default="default", help="Style name (in styles/) or path to style JSON")
    p_ren.add_argument("--view", default="iso", help="View preset (iso, side-oblique, a, b, c) or view.json")
    p_ren.add_argument("--out", "-o", type=Path, default=None, help="Output PNG path")
    p_ren.add_argument("--save-blend", type=Path, default=None, help="Save Blender .blend scene")
    p_ren.add_argument("--replicate", nargs=3, type=int, default=[1, 1, 1], metavar=("NX", "NY", "NZ"))
    p_ren.add_argument("--crop-z", default=None, help="auto or bot,top margin in Angstrom")
    p_ren.add_argument("--resolution", nargs=2, type=int, default=None, metavar=("W", "H"))
    p_ren.add_argument("--samples", type=int, default=None, help="Override Cycles samples")
    p_ren.add_argument("--bonds", choices=["table", "auto", "none"], default=None)
    p_ren.add_argument("--fit-view", action="store_true", help="Re-fit view.json camera orientation to structure")
    p_ren.add_argument("--format", default=None, help="Explicit ASE format override")
    p_ren.add_argument("--type-map", default=None, help="LAMMPS type map, e.g. '1:Al,2:C,3:H'")
    p_ren.add_argument("--lammps-style", default=None, help="LAMMPS atom_style override")
    p_ren.add_argument("--no-wrap", action="store_true", help="Disable periodic wrapping")
    p_ren.add_argument("--index", default="-1", help="ASE frame index (default: -1)")
    p_ren.add_argument("--blender", default=None, help="Path to blender executable")
    p_ren.set_defaults(func=cmd_render)

    p_conv = sub.add_parser("convert", help="Convert structure file to neutral JSON via ASE")
    p_conv.add_argument("input", type=Path)
    p_conv.add_argument("-o", "--out", type=Path, required=True)
    p_conv.add_argument("--format", default=None)
    p_conv.add_argument("--type-map", default=None)
    p_conv.add_argument("--lammps-style", default=None)
    p_conv.add_argument("--replicate", nargs=3, type=int, default=[1, 1, 1], metavar=("NX", "NY", "NZ"))
    p_conv.add_argument("--no-wrap", action="store_true")
    p_conv.add_argument("--index", default="-1")
    p_conv.set_defaults(func=cmd_convert)

    p_sty = sub.add_parser("styles", help="List available built-in styles")
    p_sty.set_defaults(func=cmd_styles)

    p_inst = sub.add_parser("install-addon", help="Install the GUI N-panel addon into Blender")
    p_inst.add_argument("--blender", default=None, help="Path to blender executable")
    p_inst.add_argument("--addons-dir", type=Path, default=None, help="Override target addons directory")
    p_inst.add_argument("--no-enable", action="store_true", help="Copy files without enabling in Blender preferences")
    p_inst.set_defaults(func=cmd_install_addon)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "render" and args.out is None and args.save_blend is None:
        parser.error("render requires at least --out or --save-blend")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
