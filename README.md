# atomic-blender

Reusable publication-quality atomic structure renderer combining **ASE** (universal structure parsing, supercell replication, periodic wrapping) with **Blender Cycles** (consistent materials, orthographic framing, GPU/OptiX acceleration, and an interactive Blender N-panel addon).

<p align="center">
  <img src="examples/renders/fcc_cu_bulk.png" width="30%" alt="FCC Cu bulk">
  <img src="examples/renders/water_box.png" width="30%" alt="Periodic water box">
  <img src="examples/renders/co_on_pt111.png" width="30%" alt="CO on Pt(111) slab">
</p>

## Features

- **Style-driven rendering (`styles/default.json`)**: Separates element colors (with Jmol/CPK fallback for all elements), radii, Principled BSDF parameters (`Roughness = 0.35`, `Specular = 0.4`), bond cutoffs, 12-edge triclinic cell wireframe, Sun + Area fill lighting, and Cycles GPU/CPU settings into JSON.
- **Universal input via ASE -> neutral JSON**: Supports `POSCAR`/`VASP`, `.cif`, `.xyz`/`.extxyz`, LAMMPS `.data` (with `--type-map`), and any ASE-readable format without requiring ASE inside Blender's embedded Python.
- **Supercells & triclinic cells**: Supports `--replicate NX NY NZ`, periodic wrapping, general 3D triclinic cell edges, and `--crop-z auto` for slabs with large vacuum layers.
- **Headless CLI & interactive GUI workflow**: Frame structures interactively in Blender's 3D Viewport, snap the camera with **Camera to Current View**, export `view.json` with **Save View**, and reproduce the exact framing headlessly via `--view view.json`.

---

## Requirements

- **Python** `>= 3.10` with [ASE](https://wiki.fysik.dtu.dk/ase/) (`ase >= 3.22`) and `numpy`
- **Blender** `4.x` or `5.x` (tested on Blender 5.2.3 LTS with Cycles OptiX)

## Installation

Install directly from GitHub (for private access, ensure your Git credentials or SSH key are configured):

```bash
pip install git+https://github.com/polar2251/atomic-blender.git
```

Or clone and install locally (standard or editable mode):

```bash
git clone https://github.com/polar2251/atomic-blender.git
cd atomic-blender
pip install .          # standard self-contained install
# or: pip install -e . # editable development install
```

Blender is auto-detected across **Windows** (`Program Files\Blender Foundation\Blender*` and `%LOCALAPPDATA%`), **macOS** (`/Applications/Blender.app`), **Linux** (`/usr/bin/blender`, `/usr/local/bin/blender`, `/snap/bin/blender`, `/opt/blender*`), and `PATH`. If Blender is installed in a custom location, pass `--blender <path-to-blender>` or set `BLENDER_EXE`:

```powershell
$env:BLENDER_EXE = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
```

### Installing the Blender GUI Addon

Install and enable the N-panel addon directly into Blender's user addons directory:

```bash
atomic-blender install-addon
```

To export a standalone `.zip` archive for manual installation via Blender's **Edit > Preferences > Add-ons > Install from Disk**:

```bash
atomic-blender install-addon --zip atomic_blender_studio.zip
```

> **Note on Blender built-in addon naming**: Blender ships with a built-in `"Atomic Blender (PDB/XYZ)"` addon (`io_mesh_atomic`). To avoid any module, UI, or operator collision, this project registers under the distinct module name **`atomic_blender_studio`**, display name **`Atomic Blender Studio (Publication Renderer)`**, N-panel tab **`Atomic Studio`**, and operator namespace **`atomic_studio.*`**. Both addons can coexist safely.

---

## Headless CLI Usage

### Render a structure

```bash
# Default isometric view (azimuth 35°, elevation 20°)
atomic-blender render examples/fcc_cu_bulk.cif --style default --view iso --out cu.png

# Slab with large vacuum: side-oblique preset + automatic Z-range cropping
atomic-blender render examples/co_on_pt111.vasp --style default --view side-oblique --crop-z auto --resolution 1400 1200 --out slab.png

# Supercell replication + saving an editable .blend file
atomic-blender render examples/water_box.xyz --replicate 2 2 1 --view iso --out water_221.png --save-blend water_221.blend

# LAMMPS data file with explicit type mapping
atomic-blender render structure.data --type-map "1:Al,2:C,3:H" --view side-oblique --crop-z auto --out slab.png
```

### View presets

| Preset | Description |
|---|---|
| `iso` | Azimuth `35°`, elevation `20°` |
| `side-oblique` | Azimuth `25°`, elevation `10°` |
| `a` | View along lattice vector `a` |
| `b` | View along lattice vector `b` |
| `c` | View along lattice vector `c` |
| `<path/to/view.json>` | Exact camera transform, `ortho_scale`, and resolution saved from GUI |

### Additional CLI options

- `--bonds table|auto|none`: Override bond drawing mode (`table` uses style cutoffs; `auto` uses covalent radii; `none` disables bonds).
- `--fit-view`: Keep the camera orientation from `view.json` while re-fitting `target` and `ortho_scale` to a new or replicated structure.
- `--samples N`: Override Cycles sample count.
- `--no-wrap`: Keep coordinates unwrapped.
- `atomic-blender convert <input> -o structure.json`: Export neutral JSON only.
- `atomic-blender styles`: List available style JSON files.

---

## Interactive View Framing (GUI Addon Workflow)

The **Atomic Studio** panel (`View3D > Sidebar (N) > Atomic Studio`) lets you frame structures visually and export exact camera views for reproducible headless rendering:

1. **Load or open a scene**:
   - Either open a `.blend` generated by `--save-blend`, **or** in the **Atomic Studio** panel pick any `.json`, `.cif`, `POSCAR`, `.xyz`, or `.data` file under **Structure & Style** and click **Load Structure**.
   - Non-JSON files are converted transparently via the configured system Python (`Python (ASE)` field, auto-configured by `install-addon`).
2. **Adjust the viewpoint interactively**:
   - Use Blender's viewport navigation (middle-mouse orbit, `Shift` + middle-mouse pan, scroll zoom) or switch presets via **Preset -> Apply**.
3. **Snap camera & export `view.json`**:
   - Click **Camera to Current View** to align `OrthoCam` (location, quaternion orientation, and `ortho_scale`) to your current 3D Viewport.
   - Click **Save View** to write `view.json` (camera type, location, `rotation_quaternion`, `ortho_scale`, clip planes, and resolution).
   - Optionally click **Render** for an immediate preview inside Blender.
4. **Reuse `view.json` in batch/headless CLI**:
   ```bash
   atomic-blender render examples/co_on_pt111.vasp --style default --view view.json --out final.png
   ```

To apply the same saved viewing angle to a different or replicated structure, pass `--fit-view`:

```bash
atomic-blender render other_slab.vasp --style default --view view.json --fit-view --crop-z auto --out other_final.png
```

---

## Customizing & Adding Styles

Styles are stored in `styles/*.json` and passed via `--style <name>` or `--style path/to/custom.json`.

Key sections in [`styles/default.json`](styles/default.json):

- `elements.colors`: Hex sRGB overrides per element. Any element not listed automatically falls back to the Jmol/CPK palette.
- `elements.radii`: Sphere radius overrides (Å) and `radius_fallback` (`transition_metal`, `covalent_scale`, `default`).
- `material`: Principled BSDF `roughness` (`0.35`) and `specular` (`0.4`).
- `bonds`: `mode` (`table`, `auto`, `none`), `color`, `radius`, `cutoffs` (`"C-H"`, `"O-H"`, `"C-C"`, etc.), and `skip_pbc`.
- `cell`: Toggle and styling (`color`, `radius`) for the 12-edge periodic cell wireframe.
- `camera`, `lighting`, `world`, `render`: Orthographic framing margin (`1.08`), Sun + Area fill lights, neutral ambient world, Cycles device priority (`OPTIX`, `CUDA`, `CPU`), sample count (`64`), denoising, and RGBA transparency.

To create a new style, copy `styles/default.json` to `styles/my_style.json`, adjust any fields, and pass `--style my_style`.

---

## Examples

Public example inputs and regeneration script are in [`examples/`](examples/):

- [`examples/fcc_cu_bulk.cif`](examples/fcc_cu_bulk.cif) — 3×3×3 FCC Cu supercell
- [`examples/water_box.xyz`](examples/water_box.xyz) — Periodic box of 12 water molecules (`extxyz`)
- [`examples/co_on_pt111.vasp`](examples/co_on_pt111.vasp) — CO adsorbate on a 4-layer Pt(111) slab (`VASP`/`POSCAR`)

Regenerate all example inputs and preview PNGs with:

```bash
python examples/generate_examples.py
```
