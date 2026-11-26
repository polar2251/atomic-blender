"""Headless CLI for atomic-blender."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from atomic_blender.convert import convert_file


def package_dir() -> Path:
    """Return the installed ``atomic_blender`` package directory."""
    return Path(__file__).resolve().parent


def package_root() -> Path:
    """Return parent directory of the ``atomic_blender`` package (repo root or site-packages)."""
    return package_dir().parent


def _style_search_dirs() -> list[Path]:
    """Return candidate directories containing style JSON files (packaged first, then repo)."""
    candidates = [
        package_dir() / "styles",
        package_root() / "styles",
    ]
    seen: set[Path] = set()
    dirs: list[Path] = []
    for d in candidates:
        resolved = d.resolve()
        if resolved not in seen and d.is_dir():
            seen.add(resolved)
            dirs.append(d)
    return dirs


def _collect_style_files() -> dict[str, Path]:
    """Map style stem name -> path across packaged and repo style directories."""
    styles: dict[str, Path] = {}
    for d in _style_search_dirs():
        for p in sorted(d.glob("*.json")):
            styles.setdefault(p.stem, p.resolve())
    return styles


def resolve_style_path(style_arg: str) -> Path:
    """Resolve a style name (e.g. ``default``) or explicit ``.json`` path."""
    candidate = Path(style_arg)
    if candidate.is_file():
        return candidate.resolve()
    for d in _style_search_dirs():
        for path in (d / f"{style_arg}.json", d / style_arg):
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"Style not found: {style_arg!r}")


def _candidate_blender_paths() -> list[Path]:
    """Return OS-specific standard Blender installation paths (newest version first)."""
    candidates: list[Path] = []
    if sys.platform.startswith("win") or os.name == "nt":
        patterns = [
            r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        ]
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            patterns.append(os.path.join(local_app, "Programs", "Blender Foundation", "Blender*", "blender.exe"))
        for pat in patterns:
            for match in sorted(glob.glob(pat), reverse=True):
                candidates.append(Path(match))
    elif sys.platform == "darwin":
        candidates.extend([
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        ])
    else:
        candidates.extend([
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/snap/bin/blender"),
        ])
        for match in sorted(glob.glob("/opt/blender*/blender"), reverse=True):
            candidates.append(Path(match))
    return candidates


def find_blender_executable(explicit: str | None = None) -> Path:
    """Locate the Blender executable from CLI flag, env var, PATH, or OS install paths."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        found = shutil.which(explicit)
        if found:
            return Path(found)
        raise FileNotFoundError(f"Blender executable not found: {explicit}")

    env_exe = os.environ.get("BLENDER_EXE", "").strip()
    if env_exe:
        p = Path(env_exe)
        if p.is_file():
            return p
        found = shutil.which(env_exe)
        if found:
            return Path(found)

    found = shutil.which("blender")
    if found:
        return Path(found)

    for candidate in _candidate_blender_paths():
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate Blender executable. Pass --blender <path>, add blender to PATH, or set BLENDER_EXE."
    )


def _find_addon_init() -> Path:
    """Locate the ``atomic_blender_studio/__init__.py`` template from package or repo."""
    candidates = [
        package_dir() / "addon" / "atomic_blender_studio" / "__init__.py",
        package_root() / "addon" / "atomic_blender_studio" / "__init__.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not locate bundled atomic_blender_studio/__init__.py.")


def cmd_render(args: argparse.Namespace) -> int:
    style_path = resolve_style_path(args.style)
    blender_exe = find_blender_executable(args.blender)
    scene_script = package_dir() / "scene_builder.py"

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
    styles = _collect_style_files()
    if not styles:
        print("No style JSON files found.")
        return 1
    for stem, path in sorted(styles.items()):
        print(f"{stem}: {path}")
    return 0


def _stage_addon_bundle(target_dir: Path) -> None:
    """Populate ``target_dir`` with a self-contained ``atomic_blender_studio`` addon."""
    pkg = package_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_find_addon_init(), target_dir / "__init__.py")
    shutil.copy2(pkg / "elements.py", target_dir / "elements.py")
    shutil.copy2(pkg / "scene_builder.py", target_dir / "scene_builder.py")

    styles_target = target_dir / "styles"
    styles_target.mkdir(exist_ok=True)
    for stem, sp in _collect_style_files().items():
        shutil.copy2(sp, styles_target / f"{stem}.json")

    config_payload = {
        "python_exe": sys.executable,
        "repo_root": str(package_root()),
    }
    (target_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")


def cmd_install_addon(args: argparse.Namespace) -> int:
    """Install self-contained ``atomic_blender_studio`` addon into Blender or export a zip."""
    if args.zip is not None:
        zip_path = args.zip.resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="atomic_addon_zip_") as tmpdir:
            bundle_dir = Path(tmpdir) / "atomic_blender_studio"
            _stage_addon_bundle(bundle_dir)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(bundle_dir.rglob("*")):
                    if file_path.is_file():
                        arcname = Path("atomic_blender_studio") / file_path.relative_to(bundle_dir)
                        zf.write(file_path, arcname.as_posix())
        print(f"Created addon zip -> {zip_path}")
        return 0

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
    _stage_addon_bundle(target_dir)

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


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the standalone desktop GUI client."""
    from atomic_blender.gui import launch_gui

    return launch_gui(initial_input=str(args.input) if args.input else None)


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
    p_inst.add_argument("--zip", type=Path, default=None, help="Export a standalone addon .zip instead of installing")
    p_inst.add_argument("--no-enable", action="store_true", help="Copy files without enabling in Blender preferences")
    p_inst.set_defaults(func=cmd_install_addon)

    p_gui = sub.add_parser("gui", help="Launch the standalone desktop GUI client")
    p_gui.add_argument("input", nargs="?", type=Path, default=None, help="Optional structure file to prefill")
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "render" and args.out is None and args.save_blend is None:
        parser.error("render requires at least --out or --save-blend")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
