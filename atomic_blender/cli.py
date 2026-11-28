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

BLENDER_DOWNLOAD_URL = "https://www.blender.org/download/"


def package_dir() -> Path:
    """Return the installed or PyInstaller-bundled ``atomic_blender`` package directory."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(getattr(sys, "_MEIPASS")) / "atomic_blender"
        if bundled.is_dir():
            return bundled.resolve()
    return Path(__file__).resolve().parent


def package_root() -> Path:
    """Return parent directory of the ``atomic_blender`` package (repo root or site-packages)."""
    return package_dir().parent


def _subprocess_no_window_kwargs() -> dict[str, int]:
    """Hide console windows on Windows when invoked from a GUI process."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}


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
        "Could not locate Blender executable. "
        f"Install Blender from {BLENDER_DOWNLOAD_URL}, pass --blender <path>, add blender to PATH, or set BLENDER_EXE."
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
        if args.input.suffix.lower() == ".json" and tuple(args.replicate) == (1, 1, 1) and not args.no_wrap and not args.format:
            neutral_json = args.input.resolve()
        else:
            from atomic_blender.convert import convert_file

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
        if args.bond_color is not None:
            blender_args.extend(["--bond-color", args.bond_color])
        if args.radius_scale != 1.0:
            blender_args.extend(["--radius-scale", str(args.radius_scale)])
        if args.no_cell:
            blender_args.append("--no-cell")
        if args.fit_view:
            blender_args.append("--fit-view")

        # Ensure Blender Python can import atomic_blender.*
        env = os.environ.copy()
        root_str = str(package_root())
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pp}" if existing_pp else root_str

        proc = subprocess.run(blender_args, env=env, **_subprocess_no_window_kwargs())
        return proc.returncode


def cmd_convert(args: argparse.Namespace) -> int:
    from atomic_blender.convert import convert_file

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
            **_subprocess_no_window_kwargs(),
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
            **_subprocess_no_window_kwargs(),
        )
    print(f"Installed addon 'atomic_blender_studio' -> {target_dir}")
    return 0


def render_app_logo_with_blender(
    png_out: Path | None = None,
    ico_out: Path | None = None,
    blender_exe: Path | None = None,
) -> tuple[Path, Path]:
    """Render a modern low-glare 3D molecule app icon (dark-glass squircle) in Blender Cycles."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    png_target = (png_out or (assets_dir / "logo.png")).resolve()
    ico_target = (ico_out or (assets_dir / "logo.ico")).resolve()
    blender = find_blender_executable(str(blender_exe) if blender_exe else None)
    scene_script = Path(__file__).resolve().parent / "scene_builder.py"

    # Place atoms in the iso camera frame (az=35 deg, el=20 deg) so all 3 bonds & 4 spheres read cleanly at 16-256 px
    logo_structure = {
        "schema_version": 1,
        "source": "logo",
        "symbols": ["Si", "O", "C", "N"],
        "positions": [
            [0.0, 0.0, 0.0],
            [-0.59, -0.42, 2.02],
            [1.35, -1.35, -0.92],
            [-1.36, 1.75, -1.16],
        ],
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [False, False, False],
        "has_cell": False,
        "origin": [0.0, 0.0, 0.0],
    }
    logo_style = {
        "name": "logo_modern_studio",
        "elements": {
            "colors": {
                "Si": "#14B8A6",
                "O": "#FB923C",
                "C": "#38BDF8",
                "N": "#F8FAFC",
            },
            "radii": {
                "Si": 0.88,
                "O": 0.72,
                "C": 0.72,
                "N": 0.74,
            },
            "radius_scale": 1.0,
        },
        "material": {
            "roughness": 0.48,
            "specular": 0.35,
            "metallic": 0.04,
            "coat_weight": 0.06,
            "subsurface_weight": 0.05,
            "ao_factor": 0.16,
            "metal": {"roughness": 0.44, "specular": 0.40, "metallic": 0.08, "coat_weight": 0.06},
            "nonmetal": {"roughness": 0.50, "specular": 0.32, "metallic": 0.02, "coat_weight": 0.05},
        },
        "atom_mesh": {"segments": 48, "ring_count": 24, "smooth": True},
        "bonds": {
            "enabled": True,
            "mode": "table",
            "color_mode": "bicolor",
            "radius": 0.22,
            "segments": 28,
            "smooth": True,
            "cutoffs": {"Si-O": 2.7, "Si-C": 2.7, "Si-N": 2.7},
        },
        "cell": {"enabled": False},
        "lighting": {
            "sun": {"energy": 1.75, "rotation_deg": [28.0, -18.0, 24.0]},
            "fill": {"energy": 210.0, "size": 22.0, "distance": 11.0, "elevation_deg": 24.0, "azimuth_deg": -48.0},
            "rim": {"energy": 95.0, "size": 14.0, "distance": 12.0, "elevation_deg": -15.0, "azimuth_deg": 195.0},
            "background_strength": 0.70,
            "background_color": [0.92, 0.94, 0.97],
        },
        "camera": {"margin": 1.05},
        "render": {"resolution": [640, 640], "samples": 48, "transparent": True},
    }

    with tempfile.TemporaryDirectory(prefix="atomic_logo_") as tmpdir:
        tmp_path = Path(tmpdir)
        s_json = tmp_path / "logo_struct.json"
        st_json = tmp_path / "logo_style.json"
        raw_png = tmp_path / "raw_mol.png"
        s_json.write_text(json.dumps(logo_structure, indent=2), encoding="utf-8")
        st_json.write_text(json.dumps(logo_style, indent=2), encoding="utf-8")

        env = os.environ.copy()
        root_str = str(Path(__file__).resolve().parent.parent)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pp}" if existing_pp else root_str

        subprocess.run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--python",
                str(scene_script),
                "--",
                "--structure",
                str(s_json),
                "--style",
                str(st_json),
                "--view",
                "iso",
                "--out",
                str(raw_png),
            ],
            check=True,
            env=env,
            **_subprocess_no_window_kwargs(),
        )

        fg = Image.open(raw_png).convert("RGBA")
        ss_size = 1024
        size = 512

        # Sleek dark-slate vertical gradient squircle with soft teal core glow and glass rim
        t = np.linspace(0.0, 1.0, ss_size, dtype=np.float32)[:, None]
        grad_rgb = np.empty((ss_size, ss_size, 4), dtype=np.uint8)
        grad_rgb[:, :, 0] = np.round((1.0 - t) * 24 + t * 10).astype(np.uint8)
        grad_rgb[:, :, 1] = np.round((1.0 - t) * 32 + t * 14).astype(np.uint8)
        grad_rgb[:, :, 2] = np.round((1.0 - t) * 48 + t * 22).astype(np.uint8)
        grad_rgb[:, :, 3] = 255
        grad_img = Image.fromarray(grad_rgb, mode="RGBA")

        glow = Image.new("RGBA", (ss_size, ss_size), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        glow_draw.ellipse((212, 200, 812, 800), fill=(20, 184, 166, 56))
        glow_draw.ellipse((320, 140, 704, 460), fill=(56, 189, 248, 28))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=120))
        grad_img.alpha_composite(glow)

        mask = Image.new("L", (ss_size, ss_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle((24, 24, ss_size - 24, ss_size - 24), radius=232, fill=255)

        badge_ss = Image.new("RGBA", (ss_size, ss_size), (0, 0, 0, 0))
        badge_ss.paste(grad_img, (0, 0), mask)

        rim_draw = ImageDraw.Draw(badge_ss, "RGBA")
        rim_draw.rounded_rectangle(
            (28, 28, ss_size - 28, ss_size - 28),
            radius=228,
            fill=None,
            outline=(255, 255, 255, 34),
            width=4,
        )
        rim_draw.rounded_rectangle(
            (24, 24, ss_size - 24, ss_size - 24),
            radius=232,
            fill=None,
            outline=(45, 212, 191, 140),
            width=6,
        )

        bbox = fg.getbbox()
        if bbox is not None:
            fg = fg.crop(bbox)
        target_box = int(ss_size * 0.74)
        scale = target_box / max(1, max(fg.width, fg.height))
        nw = max(1, int(round(fg.width * scale)))
        nh = max(1, int(round(fg.height * scale)))
        fg_resized = fg.resize((nw, nh), Image.Resampling.LANCZOS)
        dx = (ss_size - nw) // 2
        dy = (ss_size - nh) // 2

        # Soft drop shadow under the 3D molecule inside the squircle
        alpha_ch = fg_resized.getchannel("A")
        shadow_layer = Image.new("RGBA", (ss_size, ss_size), (0, 0, 0, 0))
        shadow_tint = Image.new("RGBA", (nw, nh), (0, 0, 0, 115))
        shadow_layer.paste(shadow_tint, (dx, dy + 18), alpha_ch)
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=24))
        badge_ss.alpha_composite(shadow_layer)
        badge_ss.alpha_composite(fg_resized, dest=(dx, dy))

        badge = badge_ss.resize((size, size), Image.Resampling.LANCZOS)

        png_target.parent.mkdir(parents=True, exist_ok=True)
        badge.save(png_target, format="PNG")
        ico_target.parent.mkdir(parents=True, exist_ok=True)
        badge.save(
            ico_target,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
    return png_target, ico_target


def render_welcome_loop_with_blender(
    gif_out: Path | None = None,
    blender_exe: Path | None = None,
    n_frames: int = 72,
) -> Path:
    """Pre-render a seamlessly looping 360° style-switching 3D welcome animation via Blender Cycles."""
    import copy
    from PIL import Image

    if n_frames < 8 or n_frames % 4 != 0:
        raise ValueError("n_frames must be a positive multiple of 4 (at least 8).")

    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    gif_target = (gif_out or (assets_dir / "welcome_loop.gif")).resolve()
    blender = find_blender_executable(str(blender_exe) if blender_exe else None)
    scene_script = Path(__file__).resolve().parent / "scene_builder.py"

    logo_structure = {
        "schema_version": 1,
        "source": "welcome_loop",
        "symbols": ["Si", "O", "C", "N"],
        "positions": [
            [0.0, 0.0, 0.0],
            [-0.59, -0.42, 2.02],
            [1.35, -1.35, -0.92],
            [-1.36, 1.75, -1.16],
        ],
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [False, False, False],
        "has_cell": False,
        "origin": [0.0, 0.0, 0.0],
    }

    base_style: dict[str, Any] = {
        "elements": {
            "colors": {"Si": "#14B8A6", "O": "#FB923C", "C": "#38BDF8", "N": "#F1F5F9"},
            "radii": {"Si": 0.86, "O": 0.70, "C": 0.70, "N": 0.74},
            "radius_scale": 1.0,
        },
        "material": {
            "roughness": 0.28,
            "specular": 0.55,
            "metallic": 0.05,
            "coat_weight": 0.05,
            "subsurface_weight": 0.02,
            "ao_factor": 0.14,
            "metal": {"roughness": 0.26, "specular": 0.65, "metallic": 0.15, "coat_weight": 0.05},
            "nonmetal": {"roughness": 0.30, "specular": 0.50, "metallic": 0.0, "coat_weight": 0.05},
        },
        "atom_mesh": {"segments": 32, "ring_count": 16, "smooth": True},
        "bonds": {
            "enabled": True,
            "mode": "table",
            "color_mode": "bicolor",
            "radius": 0.19,
            "segments": 20,
            "smooth": True,
            "cutoffs": {"Si-O": 2.7, "Si-C": 2.7, "Si-N": 2.7},
        },
        "cell": {"enabled": False},
        "lighting": {
            "sun": {"energy": 2.4, "rotation_deg": [38.0, -22.0, 32.0]},
            "fill": {"energy": 160.0, "size": 16.0, "distance": 11.0, "elevation_deg": 24.0, "azimuth_deg": -52.0},
            "rim": {"energy": 85.0, "size": 12.0, "distance": 12.0, "elevation_deg": -12.0, "azimuth_deg": 195.0},
            "background_strength": 0.55,
            "background_color": [0.9, 0.92, 0.95],
        },
        "camera": {"margin": 1.06},
        "render": {"resolution": [360, 360], "samples": 16, "transparent": True},
    }

    style_studio = copy.deepcopy(base_style)
    style_studio["name"] = "welcome_studio"

    style_matte = copy.deepcopy(base_style)
    style_matte["name"] = "welcome_matte"
    style_matte["material"] = {
        "roughness": 0.92,
        "specular": 0.05,
        "metallic": 0.0,
        "coat_weight": 0.0,
        "subsurface_weight": 0.04,
        "ao_factor": 0.16,
        "metal": {"roughness": 0.92, "specular": 0.05, "metallic": 0.0, "coat_weight": 0.0},
        "nonmetal": {"roughness": 0.92, "specular": 0.05, "metallic": 0.0, "coat_weight": 0.0},
    }

    style_licorice = copy.deepcopy(base_style)
    style_licorice["name"] = "welcome_licorice"
    style_licorice["elements"]["radius_scale"] = 0.44
    style_licorice["bonds"]["radius"] = 0.27
    style_licorice["material"]["roughness"] = 0.34

    style_vdw = copy.deepcopy(base_style)
    style_vdw["name"] = "welcome_vdw"
    style_vdw["elements"]["radius_scale"] = 1.78
    style_vdw["bonds"]["enabled"] = False
    style_vdw["bonds"]["mode"] = "none"
    style_vdw["material"] = {
        "roughness": 0.24,
        "specular": 0.72,
        "metallic": 0.28,
        "coat_weight": 0.12,
        "subsurface_weight": 0.0,
        "ao_factor": 0.16,
        "metal": {"roughness": 0.22, "specular": 0.75, "metallic": 0.35, "coat_weight": 0.15},
        "nonmetal": {"roughness": 0.28, "specular": 0.65, "metallic": 0.20, "coat_weight": 0.10},
    }

    styles_list = [style_studio, style_matte, style_licorice, style_vdw]

    driver_code = """
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

script_dir = str(Path(sys.argv[sys.argv.index('--') + 1]))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import scene_builder

cfg = json.loads(Path(sys.argv[sys.argv.index('--') + 2]).read_text(encoding='utf-8'))
structure = cfg['structure']
styles = cfg['styles']
out_dir = Path(cfg['out_dir'])
n_frames = int(cfg['n_frames'])
seg_len = n_frames // 4
target = Vector((-0.15, -0.005, -0.015))
el = math.radians(20.0)

def render_angle(scene, cam_obj, frame_idx, out_path):
    az = math.radians(35.0 + frame_idx * (360.0 / n_frames))
    view_dir = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    rot_quat = (-view_dir).to_track_quat('-Z', 'Y')
    cam_obj.location = target + view_dir * 200.0
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = rot_quat
    cam_obj.data.ortho_scale = 7.8
    scene.render.filepath = str(out_path.resolve())
    bpy.ops.render.render(write_still=True)

for s_idx, style in enumerate(styles):
    scene_builder.build_scene(
        structure=structure,
        style=style,
        view='iso',
        resolution=(360, 360),
        reset_factory=True,
    )
    scene = bpy.context.scene
    scene_builder._apply_fast_preview_cycles_settings(scene)
    cam_obj = scene.camera
    start_k = s_idx * seg_len
    for k in range(start_k, start_k + seg_len):
        render_angle(scene, cam_obj, k, out_dir / f'main_{k:02d}.png')
    for offset, tag in ((3, 'blend3'), (2, 'blend2'), (1, 'blend1')):
        pk = (start_k - offset) % n_frames
        render_angle(scene, cam_obj, pk, out_dir / f'{tag}_{pk:02d}.png')
"""

    with tempfile.TemporaryDirectory(prefix="atomic_welcome_") as tmpdir:
        tmp_path = Path(tmpdir)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        cfg_json = tmp_path / "welcome_cfg.json"
        driver_py = tmp_path / "render_welcome_driver.py"
        cfg_json.write_text(
            json.dumps(
                {
                    "structure": logo_structure,
                    "styles": styles_list,
                    "out_dir": str(frames_dir),
                    "n_frames": n_frames,
                }
            ),
            encoding="utf-8",
        )
        driver_py.write_text(driver_code, encoding="utf-8")

        env = os.environ.copy()
        root_str = str(Path(__file__).resolve().parent.parent)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pp}" if existing_pp else root_str

        subprocess.run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--python",
                str(driver_py),
                "--",
                str(scene_script.parent),
                str(cfg_json),
            ],
            check=True,
            env=env,
            **_subprocess_no_window_kwargs(),
        )

        # Solid backdrop matching the GUI canvas (#0F1115) prevents 8-bit GIF glow banding
        backdrop = Image.new("RGBA", (360, 360), (15, 17, 21, 255))

        composited: list[Image.Image] = []
        for k in range(n_frames):
            fg = Image.open(frames_dir / f"main_{k:02d}.png").convert("RGBA")
            for tag, weight in (("blend3", 0.25), ("blend2", 0.50), ("blend1", 0.75)):
                bp = frames_dir / f"{tag}_{k:02d}.png"
                if bp.is_file():
                    fg = Image.blend(fg, Image.open(bp).convert("RGBA"), weight)
                    break
            frame = backdrop.copy()
            frame.alpha_composite(fg)
            composited.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))

        gif_target.parent.mkdir(parents=True, exist_ok=True)
        composited[0].save(
            gif_target,
            save_all=True,
            append_images=composited[1:],
            duration=95,
            loop=0,
            optimize=True,
        )
    return gif_target


def ensure_app_icon(target_path: Path | None = None) -> Path:
    """Return the multi-size Windows ``.ico`` studio app icon (using bundled Blender-rendered asset)."""
    bundled_ico = package_dir() / "assets" / "logo.ico"
    ico_path = (target_path or (Path.home() / ".atomic_blender" / "atomic_blender_studio.ico")).expanduser()
    ico_path.parent.mkdir(parents=True, exist_ok=True)

    if bundled_ico.is_file() and bundled_ico.stat().st_size > 0:
        if target_path is None:
            if not ico_path.is_file() or ico_path.read_bytes() != bundled_ico.read_bytes():
                shutil.copy2(bundled_ico, ico_path)
            return ico_path
        if not ico_path.is_file() or ico_path.stat().st_size == 0:
            shutil.copy2(bundled_ico, ico_path)
        return ico_path

    if ico_path.is_file() and ico_path.stat().st_size > 0:
        return ico_path

    from PIL import Image, ImageDraw
    from atomic_blender.viewport3d import build_sphere_sprite

    size = 256
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((10, 10, 246, 246), radius=54, fill=(15, 19, 28, 255), outline=(20, 184, 166, 220), width=5)

    draw.ellipse((46, 46, 210, 210), outline=(45, 212, 191, 55), width=3)

    centers = {"top": (128.0, 74.0), "left": (78.0, 166.0), "right": (178.0, 166.0)}
    for a, b in (("top", "left"), ("top", "right"), ("left", "right")):
        p0, p1 = centers[a], centers[b]
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(148, 163, 184, 240), width=16)

    specs = (
        ("top", "#FB923C", 84, 0.90, 0.0),
        ("left", "#38BDF8", 78, 0.90, 0.0),
        ("right", "#14B8A6", 86, 0.90, 0.0),
    )
    for key, col, diam, rough, metal in specs:
        sp = build_sphere_sprite(hex_color=col, roughness=rough, specular=0.06, metallic=metal, coat_weight=0.0, size=diam)
        cx, cy = centers[key]
        canvas.alpha_composite(sp, dest=(int(round(cx - diam * 0.5)), int(round(cy - diam * 0.5))))

    canvas.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    return ico_path


def _resolve_gui_exe(target_exe: Path | None = None) -> Path:
    """Resolve the best GUI executable path (standalone dist .exe, frozen exe, or Scripts GUI launcher)."""
    if target_exe is not None:
        p = Path(target_exe).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Target executable not found: {p}")
        return p
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()

    candidates = [
        Path.cwd() / "dist" / "AtomicBlenderStudio" / "AtomicBlenderStudio.exe",
        package_root() / "dist" / "AtomicBlenderStudio" / "AtomicBlenderStudio.exe",
        Path.home() / ".atomic_blender" / "dist" / "AtomicBlenderStudio" / "AtomicBlenderStudio.exe",
        Path(sys.executable).parent / "Scripts" / "AtomicBlenderStudio.exe",
        Path(sys.executable).parent / "Scripts" / "atomic-blender-gui.exe",
    ]
    for name in ("AtomicBlenderStudio", "atomic-blender-gui"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate AtomicBlenderStudio.exe or atomic-blender-gui.exe. "
        "Run 'atomic-blender install-desktop --build-exe' or 'pip install .'."
    )


def build_standalone_exe(output_dir: Path | None = None, install_shortcuts: bool = True) -> Path:
    """Build a standalone windowless Windows ``AtomicBlenderStudio.exe`` bundle via PyInstaller."""
    if output_dir is not None:
        dist_root = Path(output_dir).expanduser().resolve()
    elif (Path.cwd() / "pyproject.toml").is_file():
        dist_root = (Path.cwd() / "dist").resolve()
    else:
        dist_root = (Path.home() / ".atomic_blender" / "dist").resolve()
    dist_root.mkdir(parents=True, exist_ok=True)

    ico_path = ensure_app_icon()
    pkg = package_dir()
    sep = os.pathsep
    add_data_entries = [
        f"{pkg / 'styles'}{sep}atomic_blender/styles",
        f"{pkg / 'addon' / 'atomic_blender_studio' / '__init__.py'}{sep}atomic_blender/addon/atomic_blender_studio",
        f"{pkg / 'scene_builder.py'}{sep}atomic_blender",
        f"{pkg / 'elements.py'}{sep}atomic_blender",
        f"{pkg / '__init__.py'}{sep}atomic_blender",
    ]
    if (pkg / "assets").is_dir():
        add_data_entries.append(f"{pkg / 'assets'}{sep}atomic_blender/assets")

    with tempfile.TemporaryDirectory(prefix="atomic_pyinstaller_") as tmpdir:
        tmp_path = Path(tmpdir)
        entry_py = tmp_path / "launch_studio.py"
        entry_py.write_text(
            "from atomic_blender.gui import main\n"
            "raise SystemExit(main())\n",
            encoding="utf-8",
        )
        runner_py = tmp_path / "run_pyinstaller.py"
        runner_py.write_text(
            "import ast\n"
            "import sys\n"
            "from pathlib import Path\n"
            "from PyInstaller import __main__ as pyi_main\n"
            "from PyInstaller.lib.modulegraph import modulegraph as mg\n\n"
            "_orig_scan_bytecode = mg.ModuleGraph._scan_bytecode\n"
            "def _safe_scan_bytecode(self, module, co, is_scanning_imports=True, *args, **kwargs):\n"
            "    try:\n"
            "        return _orig_scan_bytecode(self, module, co, is_scanning_imports, *args, **kwargs)\n"
            "    except IndexError:\n"
            "        if is_scanning_imports:\n"
            "            fn = getattr(co, 'co_filename', None) or getattr(module, 'filename', None)\n"
            "            if fn and Path(fn).is_file():\n"
            "                self._scan_ast(module, ast.parse(Path(fn).read_text(encoding='utf-8'), filename=fn))\n"
            "        return None\n"
            "mg.ModuleGraph._scan_bytecode = _safe_scan_bytecode\n"
            "pyi_main.run(sys.argv[1:])\n",
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(runner_py),
            "--noconfirm",
            "--clean",
            "--onedir",
            "--noconsole",
            "--name",
            "AtomicBlenderStudio",
            "--icon",
            str(ico_path),
            "--distpath",
            str(dist_root),
            "--workpath",
            str(tmp_path / "build"),
            "--specpath",
            str(tmp_path),
            "--hidden-import",
            "atomic_blender.cli",
            "--hidden-import",
            "atomic_blender.convert",
            "--hidden-import",
            "atomic_blender.elements",
            "--hidden-import",
            "atomic_blender.viewport3d",
            "--hidden-import",
            "atomic_blender.qt_gui",
            "--hidden-import",
            "atomic_blender.qt_gui.window",
            "--hidden-import",
            "atomic_blender.qt_gui.gl_viewport",
            "--hidden-import",
            "atomic_blender.qt_gui.theme",
            "--hidden-import",
            "OpenGL.platform.win32",
            "--hidden-import",
            "OpenGL.arrays.numpymodule",
            "--hidden-import",
            "OpenGL.arrays.vbo",
            "--collect-submodules",
            "ase.io",
            "--collect-data",
            "ase",
            "--exclude-module",
            "PySide6.QtWebEngineCore",
            "--exclude-module",
            "PySide6.QtWebEngineWidgets",
            "--exclude-module",
            "PySide6.QtQuick",
            "--exclude-module",
            "PySide6.QtQml",
            "--exclude-module",
            "PySide6.QtMultimedia",
            "--exclude-module",
            "PySide6.QtCharts",
            "--exclude-module",
            "PySide6.Qt3DCore",
            "--exclude-module",
            "PySide6.QtDesigner",
            "--exclude-module",
            "PySide6.QtPdf",
            "--exclude-module",
            "PySide6.QtSql",
            "--exclude-module",
            "PySide6.QtNetwork",
            "--exclude-module",
            "matplotlib",
            "--exclude-module",
            "torch",
            "--exclude-module",
            "pandas",
            "--exclude-module",
            "sympy",
            "--exclude-module",
            "IPython",
            "--exclude-module",
            "pytest",
            "--exclude-module",
            "notebook",
        ]
        for entry in add_data_entries:
            cmd.extend(["--add-data", entry])
        cmd.append(str(entry_py))

        subprocess.run(cmd, check=True)

    exe_path = dist_root / "AtomicBlenderStudio" / "AtomicBlenderStudio.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"PyInstaller build did not produce {exe_path}")
    if install_shortcuts:
        install_desktop_shortcut(target_exe=exe_path)
    return exe_path


def install_desktop_shortcut(
    desktop: bool = True,
    start_menu: bool = True,
    target_exe: Path | None = None,
    desktop_dir: Path | None = None,
    start_menu_dir: Path | None = None,
) -> list[Path]:
    """Create Desktop and Start Menu shortcuts (``.lnk`` on Windows) for Atomic Blender Studio."""
    if not (sys.platform.startswith("win") or os.name == "nt"):
        raise RuntimeError("Desktop .exe shortcut creation is currently supported on Windows.")

    exe_path = _resolve_gui_exe(target_exe)
    ico_path = ensure_app_icon()

    ps_script = """
$ErrorActionPreference = 'Stop'
$WshShell = New-Object -ComObject WScript.Shell
$created = @()
if ($env:AB_DO_DESKTOP -eq '1') {
    $dir = if ($env:AB_DESKTOP_DIR) { $env:AB_DESKTOP_DIR } else { [Environment]::GetFolderPath('Desktop') }
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $lnkPath = Join-Path $dir 'Atomic Blender Studio.lnk'
    $s = $WshShell.CreateShortcut($lnkPath)
    $s.TargetPath = $env:AB_TARGET_EXE
    $s.WorkingDirectory = $env:AB_WORK_DIR
    $s.IconLocation = "$($env:AB_ICON_PATH),0"
    $s.Description = 'Atomic Blender Studio — Publication Atomic Structure Renderer'
    $s.Save()
    $created += $lnkPath
}
if ($env:AB_DO_START -eq '1') {
    $dir = if ($env:AB_START_DIR) { $env:AB_START_DIR } else { [Environment]::GetFolderPath('Programs') }
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $lnkPath = Join-Path $dir 'Atomic Blender Studio.lnk'
    $s = $WshShell.CreateShortcut($lnkPath)
    $s.TargetPath = $env:AB_TARGET_EXE
    $s.WorkingDirectory = $env:AB_WORK_DIR
    $s.IconLocation = "$($env:AB_ICON_PATH),0"
    $s.Description = 'Atomic Blender Studio — Publication Atomic Structure Renderer'
    $s.Save()
    $created += $lnkPath
}
$created | ForEach-Object { Write-Output $_ }
"""
    env = os.environ.copy()
    env["AB_DO_DESKTOP"] = "1" if desktop else "0"
    env["AB_DO_START"] = "1" if start_menu else "0"
    env["AB_DESKTOP_DIR"] = str(desktop_dir) if desktop_dir is not None else ""
    env["AB_START_DIR"] = str(start_menu_dir) if start_menu_dir is not None else ""
    env["AB_TARGET_EXE"] = str(exe_path)
    env["AB_WORK_DIR"] = str(exe_path.parent)
    env["AB_ICON_PATH"] = str(ico_path)

    res = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        **_subprocess_no_window_kwargs(),
    )
    return [Path(line.strip()) for line in res.stdout.splitlines() if line.strip()]


def cmd_install_desktop(args: argparse.Namespace) -> int:
    """Build standalone .exe (optional) and create Desktop / Start Menu shortcuts."""
    target = args.target_exe
    if getattr(args, "build_exe", False):
        target = build_standalone_exe(install_shortcuts=False)
        print(f"Built standalone app -> {target}")
    created = install_desktop_shortcut(
        desktop=not args.no_desktop,
        start_menu=not args.no_start_menu,
        target_exe=target,
    )
    for p in created:
        print(f"Created shortcut -> {p}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the standalone desktop GUI client."""
    backend = getattr(args, "backend", "auto")
    initial = str(args.input) if args.input else None
    if backend in ("auto", "qt"):
        try:
            from atomic_blender.qt_gui import HAS_PYSIDE6, launch_qt_gui

            if HAS_PYSIDE6:
                return launch_qt_gui(initial_input=initial)
            if backend == "qt":
                raise RuntimeError("PySide6 backend requested but PySide6 is not installed.")
        except Exception:
            if backend == "qt":
                raise

    from atomic_blender.gui import launch_gui

    return launch_gui(initial_input=initial)


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
    p_ren.add_argument("--bond-color", choices=["style", "single", "bicolor"], default=None)
    p_ren.add_argument("--radius-scale", type=float, default=1.0, help="Scale factor for atom sphere radii")
    p_ren.add_argument("--no-cell", action="store_true", help="Hide periodic cell wireframe")
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
    p_gui.add_argument(
        "--backend",
        choices=["auto", "qt", "tk"],
        default="auto",
        help="Desktop GUI toolkit backend (default: auto -> PySide6/OpenGL with Tk fallback)",
    )
    p_gui.set_defaults(func=cmd_gui)

    p_desk = sub.add_parser("install-desktop", help="Install Desktop and Start Menu shortcuts (and optional standalone .exe)")
    p_desk.add_argument("--build-exe", action="store_true", help="Build standalone AtomicBlenderStudio.exe bundle via PyInstaller")
    p_desk.add_argument("--target-exe", type=Path, default=None, help="Explicit executable path for shortcut target")
    p_desk.add_argument("--no-desktop", action="store_true", help="Skip creating the Desktop shortcut")
    p_desk.add_argument("--no-start-menu", action="store_true", help="Skip creating the Start Menu shortcut")
    p_desk.set_defaults(func=cmd_install_desktop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "render" and args.out is None and args.save_blend is None:
        parser.error("render requires at least --out or --save-blend")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
