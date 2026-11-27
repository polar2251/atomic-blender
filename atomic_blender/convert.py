"""Convert ASE-supported atomic structure files into neutral JSON for Blender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ase.io
from ase.data import atomic_numbers
import numpy as np


def parse_type_map(spec: str | list[str] | None) -> dict[int, int] | None:
    """Parse LAMMPS type-to-element mappings into ``{type_id: atomic_number}``.

    Accepts forms like ``"1:Al,2:C,3:H"``, ``"Al,C,H"``, or ``["1:Al", "2:C"]``.
    """
    if not spec:
        return None
    if isinstance(spec, str):
        tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
    else:
        tokens = []
        for item in spec:
            tokens.extend(t.strip() for t in item.replace(";", ",").split(",") if t.strip())
    if not tokens:
        return None

    z_map: dict[int, int] = {}
    for idx, tok in enumerate(tokens, start=1):
        if ":" in tok or "=" in tok:
            sep = ":" if ":" in tok else "="
            left, right = tok.split(sep, 1)
            type_id = int(left.strip())
            sym = right.strip()
        else:
            type_id = idx
            sym = tok
        if sym.isdigit():
            z_map[type_id] = int(sym)
        else:
            if sym not in atomic_numbers:
                raise ValueError(f"Unknown chemical symbol in type map: {sym!r}")
            z_map[type_id] = int(atomic_numbers[sym])
    return z_map


def read_structure(
    input_path: Path,
    fmt: str | None = None,
    type_map: str | list[str] | None = None,
    lammps_style: str | None = None,
    index: int | str = -1,
):
    """Read an atomic structure file via ASE with optional LAMMPS type mapping."""
    read_kwargs: dict[str, Any] = {"index": index}
    inferred_fmt = fmt
    if inferred_fmt is None and input_path.suffix.lower() in {".data", ".lmp"}:
        inferred_fmt = "lammps-data"
    if inferred_fmt is not None:
        read_kwargs["format"] = inferred_fmt

    z_map = parse_type_map(type_map)
    if inferred_fmt == "lammps-data" or (fmt is None and input_path.name.lower().startswith("data.")):
        if z_map:
            read_kwargs["Z_of_type"] = z_map
        if lammps_style:
            read_kwargs["atom_style"] = lammps_style

    atoms = ase.io.read(str(input_path), **read_kwargs)
    if z_map and inferred_fmt != "lammps-data":
        # Allow remapping symbols even on generic formats if explicitly requested
        nums = atoms.get_atomic_numbers()
        atoms.set_atomic_numbers([z_map.get(int(z), int(z)) for z in nums])
    return atoms


def atoms_to_neutral_dict(
    atoms,
    source: str = "",
    replicate: tuple[int, int, int] = (1, 1, 1),
    wrap: bool = True,
) -> dict[str, Any]:
    """Convert an ``ase.Atoms`` instance to a neutral JSON-serializable dictionary."""
    work = atoms.copy()
    rx, ry, rz = (int(replicate[0]), int(replicate[1]), int(replicate[2]))
    if rx < 1 or ry < 1 or rz < 1:
        raise ValueError(f"Replication counts must be >= 1, got {replicate}")

    has_cell = bool(work.cell.rank == 3)
    pbc_flags = [bool(p) for p in work.pbc]
    if wrap and has_cell and any(pbc_flags):
        work.wrap(pbc=pbc_flags)

    if (rx, ry, rz) != (1, 1, 1):
        if not has_cell:
            raise ValueError("Cannot replicate a non-periodic structure without a 3D cell.")
        work = work.repeat((rx, ry, rz))
        if wrap and any(pbc_flags):
            work.wrap(pbc=pbc_flags)

    energy_ev: float | None = None
    for e_key in ("energy", "free_energy", "E", "dft_energy"):
        if e_key in atoms.info:
            try:
                energy_ev = float(atoms.info[e_key])
                break
            except Exception:
                pass
    if energy_ev is None and getattr(atoms, "_calc", None) is not None:
        try:
            res = getattr(atoms._calc, "results", {})
            if isinstance(res, dict) and "energy" in res:
                energy_ev = float(res["energy"])
        except Exception:
            pass

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": source,
        "symbols": work.get_chemical_symbols(),
        "positions": [[float(x), float(y), float(z)] for x, y, z in work.get_positions()],
        "cell": [[float(x), float(y), float(z)] for x, y, z in work.cell.array],
        "pbc": pbc_flags,
        "has_cell": has_cell,
        "origin": [0.0, 0.0, 0.0],
    }
    if energy_ev is not None:
        payload["energy_ev"] = energy_ev
    return payload


def transform_neutral_dict(
    payload: dict[str, Any],
    replicate: tuple[int, int, int] = (1, 1, 1),
    wrap: bool = True,
    strict_nonperiodic_replicate: bool = True,
) -> dict[str, Any]:
    """Apply axis-wise periodic wrapping and supercell replication directly to a neutral dict."""
    rx, ry, rz = (int(replicate[0]), int(replicate[1]), int(replicate[2]))
    if rx < 1 or ry < 1 or rz < 1:
        raise ValueError(f"Replication counts must be >= 1, got {replicate}")

    out = dict(payload)
    symbols = list(out.get("symbols", []))
    pos_arr = np.asarray(out.get("positions", []), dtype=float).reshape((-1, 3))
    cell_mat = np.asarray(out.get("cell", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), dtype=float).reshape((3, 3))
    origin_arr = np.asarray(out.get("origin", [0.0, 0.0, 0.0]), dtype=float).reshape((3,))
    has_cell = bool(out.get("has_cell", False))
    raw_pbc = out.get("pbc", [has_cell, has_cell, has_cell])
    pbc_flags = [bool(raw_pbc[k]) if k < len(raw_pbc) else has_cell for k in range(3)]

    det = float(np.linalg.det(cell_mat)) if has_cell else 0.0
    if wrap and has_cell and any(pbc_flags) and abs(det) > 1e-12 and len(pos_arr) > 0:
        rel = pos_arr - origin_arr
        frac = np.linalg.solve(cell_mat.T, rel.T).T
        for axis in range(3):
            if pbc_flags[axis]:
                frac[:, axis] = np.mod(frac[:, axis], 1.0)
        pos_arr = origin_arr + frac @ cell_mat

    if (rx, ry, rz) != (1, 1, 1):
        if not has_cell or abs(det) <= 1e-12:
            if strict_nonperiodic_replicate:
                raise ValueError("Cannot replicate a non-periodic structure without a 3D cell.")
        else:
            n_orig = len(symbols)
            shifts = [
                ix * cell_mat[0] + iy * cell_mat[1] + iz * cell_mat[2]
                for ix in range(rx)
                for iy in range(ry)
                for iz in range(rz)
            ]
            rep_symbols: list[str] = []
            rep_pos_blocks: list[np.ndarray] = []
            for sh in shifts:
                rep_symbols.extend(symbols)
                if len(pos_arr) > 0:
                    rep_pos_blocks.append(pos_arr + sh)
            symbols = rep_symbols
            pos_arr = np.vstack(rep_pos_blocks) if rep_pos_blocks else pos_arr
            cell_mat = np.array([cell_mat[0] * rx, cell_mat[1] * ry, cell_mat[2] * rz], dtype=float)
            for list_key in ("alphas", "radius_scales", "colors"):
                raw_list = out.get(list_key)
                if isinstance(raw_list, list) and len(raw_list) == n_orig:
                    out[list_key] = list(raw_list) * (rx * ry * rz)

    out["symbols"] = symbols
    out["positions"] = [[float(x), float(y), float(z)] for x, y, z in pos_arr]
    out["cell"] = [[float(x), float(y), float(z)] for x, y, z in cell_mat]
    out["pbc"] = pbc_flags
    out["has_cell"] = has_cell
    out["origin"] = [float(origin_arr[0]), float(origin_arr[1]), float(origin_arr[2])]
    return out


def read_structure_frames(
    input_path: Path,
    fmt: str | None = None,
    type_map: str | list[str] | None = None,
    lammps_style: str | None = None,
    replicate: tuple[int, int, int] = (1, 1, 1),
    wrap: bool = True,
) -> list[dict[str, Any]]:
    """Read single or multi-frame trajectory files into a list of neutral structure dicts."""
    if input_path.suffix.lower() == ".json" and fmt is None:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        return [
            transform_neutral_dict(
                dict(item),
                replicate=replicate,
                wrap=wrap,
                strict_nonperiodic_replicate=False,
            )
            for item in items
        ]

    frames = read_structure(
        input_path=input_path,
        fmt=fmt,
        type_map=type_map,
        lammps_style=lammps_style,
        index=":",
    )
    atoms_list = frames if isinstance(frames, list) else [frames]
    out: list[dict[str, Any]] = []
    for atoms in atoms_list:
        eff_rep = replicate if atoms.cell.rank == 3 else (1, 1, 1)
        out.append(atoms_to_neutral_dict(atoms, source=str(input_path), replicate=eff_rep, wrap=wrap))
    return out


def convert_file(
    input_path: Path,
    output_json: Path,
    fmt: str | None = None,
    type_map: str | list[str] | None = None,
    lammps_style: str | None = None,
    replicate: tuple[int, int, int] = (1, 1, 1),
    wrap: bool = True,
    index: int | str = -1,
) -> dict[str, Any]:
    """Read structure file via ASE (or neutral JSON) and write neutral JSON to ``output_json``."""
    if input_path.suffix.lower() == ".json" and fmt is None:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Empty structure list in JSON file: {input_path}")
            idx_i = int(index) if isinstance(index, int) or str(index).lstrip("-").isdigit() else -1
            raw_frame = dict(data[idx_i])
        else:
            raw_frame = dict(data)
        payload = transform_neutral_dict(
            raw_frame,
            replicate=replicate,
            wrap=wrap,
            strict_nonperiodic_replicate=True,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    atoms = read_structure(
        input_path=input_path,
        fmt=fmt,
        type_map=type_map,
        lammps_style=lammps_style,
        index=index,
    )
    payload = atoms_to_neutral_dict(
        atoms,
        source=str(input_path),
        replicate=replicate,
        wrap=wrap,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert atomic structure files to neutral JSON.")
    parser.add_argument("input", type=Path, help="Input structure file (POSCAR, CIF, XYZ, LAMMPS data, ...)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output neutral JSON path")
    parser.add_argument("--format", dest="fmt", default=None, help="Explicit ASE format override")
    parser.add_argument("--type-map", default=None, help="LAMMPS type map, e.g. '1:Al,2:C,3:H' or 'Al,C,H'")
    parser.add_argument("--lammps-style", default=None, help="LAMMPS atom_style override (atomic, molecular, full)")
    parser.add_argument("--replicate", nargs=3, type=int, default=[1, 1, 1], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--no-wrap", action="store_true", help="Disable wrapping atoms into the unit cell")
    parser.add_argument("--index", default="-1", help="ASE frame index (default: -1)")
    args = parser.parse_args(argv)

    idx: int | str = int(args.index) if args.index.lstrip("-").isdigit() else args.index
    convert_file(
        input_path=args.input,
        output_json=args.out,
        fmt=args.fmt,
        type_map=args.type_map,
        lammps_style=args.lammps_style,
        replicate=tuple(args.replicate),
        wrap=not args.no_wrap,
        index=idx,
    )


if __name__ == "__main__":
    main()
