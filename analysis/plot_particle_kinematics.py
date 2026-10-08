#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.root_trees import resolve  # noqa: E402
from particle_kinematics_diagnostics import (  # noqa: E402
    BANK_ORDER,
    banks_with_content,
    discover_particle_prefixes,
    make_particle,
    render_report,
    required_branches,
    summary_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a generic multipage PDF of particle kinematics and selected detector-bank "
            "coordinates from a ROOT tree. Particle roles and unavailable banks are discovered "
            "from scalar branch names."
        )
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--tree", default="sEvents", help="ROOT tree (default: sEvents)")
    parser.add_argument(
        "--particle",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Particle branch prefix, such as electron, proton, gamma1, or row for an "
            "unprefixed sParticles tree. May be repeated; default discovers every particle."
        ),
    )
    parser.add_argument(
        "--bank",
        action="append",
        choices=("all", "auto", *BANK_ORDER),
        default=[],
        help=(
            "Page group to include; may be repeated. Default auto includes event/DIS/kinematics "
            "and each detector bank with available coordinates."
        ),
    )
    parser.add_argument("--where", help="Optional ROOT RDataFrame filter expression")
    parser.add_argument("--selection-mask", type=Path, help="Boolean NPY mask aligned to unfiltered input rows")
    parser.add_argument("--max-rows", type=int, help="Read at most this many rows after --where")
    parser.add_argument("--beam-energy", type=float, help="Beam energy in GeV, used to derive y when absent")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="JSON sidecar; defaults beside the PDF")
    parser.add_argument("--hash-input", action="store_true", help="SHA256 the ROOT input")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.selection_mask is not None and args.where:
        raise ValueError("--selection-mask cannot be combined with --where because filtered rows no longer align")

    tree_name, input_rows, available = inspect_tree(args.input_root, args.tree, args.dictionary)
    prefixes = selected_prefixes(args.particle, available)
    particles = [make_particle(prefix, available) for prefix in prefixes]
    banks = selected_banks(args.bank, particles, available)
    branches = required_branches(particles, banks, available)
    if not branches:
        raise RuntimeError("No scalar branches are available for the requested particles and banks")
    arrays, rows_after_filter = read_arrays(
        args.input_root,
        tree_name,
        branches,
        args.where,
        args.max_rows,
    )
    selection = load_selection(args.selection_mask, input_rows, rows_after_filter, args.max_rows)
    arrays = {name: values[selection] for name, values in arrays.items()}

    provenance = [
        f"Input ROOT: {args.input_root.resolve()}",
        f"Tree: {tree_name}; input rows: {input_rows:,}; plotted rows: {np.count_nonzero(selection):,}",
        f"Filter: {args.where or 'none'}",
        f"Selection mask: {args.selection_mask.resolve() if args.selection_mask else 'none'}",
        f"Maximum rows: {args.max_rows if args.max_rows is not None else 'none'}",
    ]
    pages, page_records = render_report(
        args.output,
        arrays,
        particles,
        banks,
        label=args.label,
        provenance=provenance,
        beam_energy=args.beam_energy,
    )
    summary = summary_for(arrays, particles, banks, page_records)
    summary.update(
        {
            "schema_version": 1,
            "label": args.label,
            "input_root": file_record(args.input_root, args.hash_input),
            "tree": tree_name,
            "input_rows": input_rows,
            "rows_after_filter_and_limit": rows_after_filter,
            "selection_mask": file_record(args.selection_mask, True) if args.selection_mask else None,
            "where": args.where,
            "max_rows": args.max_rows,
            "beam_energy_GeV": args.beam_energy,
            "available_scalar_branches": available,
            "requested_particle_prefixes": prefixes,
            "requested_branches": branches,
            "angle_convention": "particle and CVT theta/phi branches are radians and are plotted in degrees",
            "output_pdf": str(args.output.resolve()),
        }
    )
    summary_path = args.summary or args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)

    print(f"Tree: {tree_name}")
    print(f"Input rows: {input_rows}")
    print(f"Plotted rows: {summary['rows']}")
    print("Particles: " + ", ".join(particle.name for particle in particles))
    for particle in particles:
        print(f"  {particle.name} banks: {', '.join(banks_with_content(particle))}")
    print("Requested groups: " + ", ".join(banks))
    print(f"PDF pages: {pages}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary_path.resolve()}")
    return 0


def selected_prefixes(requested: list[str], available: list[str]) -> list[str]:
    discovered = discover_particle_prefixes(available)
    if not requested:
        if not discovered:
            raise RuntimeError("No particle branch triplets (P/Theta/Phi or p/theta/phi) were found")
        return discovered
    prefixes = ["" if item in {"row", "unprefixed"} else item for item in requested]
    return list(dict.fromkeys(prefixes))


def selected_banks(requested: list[str], particles, available: list[str]) -> list[str]:
    if "all" in requested:
        return list(BANK_ORDER)
    if requested and "auto" not in requested:
        return list(dict.fromkeys(requested))
    names = set(available)
    banks = [name for name in ("event", "dis") if any(branch in names for branch in group_branches(name))]
    banks.append("kinematics")
    for bank in BANK_ORDER:
        if bank in {"event", "dis", "kinematics"}:
            continue
        if any(bank in banks_with_content(particle) for particle in particles):
            banks.append(bank)
    return banks


def group_branches(group: str) -> tuple[str, ...]:
    if group == "event":
        return "runNum", "helicity", "charge"
    if group == "dis":
        return "Q2", "xB", "nu", "W", "y", "t"
    return ()


def inspect_tree(path: Path, requested_tree: str, dictionary: Path | None) -> tuple[str, int, list[str]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if dictionary and ROOT.gSystem.Load(str(dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")
    root_file = ROOT.TFile.Open(str(path.resolve()), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path.resolve()}")
    try:
        tree_name = resolve(root_file, requested_tree)
        tree = root_file.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Could not find tree {requested_tree} in {path.resolve()}")
        scalar = []
        for branch in tree.GetListOfBranches():
            class_name = str(branch.GetClassName())
            if class_name:
                continue
            leaf = branch.GetLeaf(branch.GetName())
            if leaf and int(leaf.GetLenStatic()) == 1:
                scalar.append(str(branch.GetName()))
        return tree_name, int(tree.GetEntries()), sorted(scalar)
    finally:
        root_file.Close()


def read_arrays(
    path: Path,
    tree: str,
    branches: list[str],
    where: str | None,
    max_rows: int | None,
) -> tuple[dict[str, np.ndarray], int]:
    import ROOT  # type: ignore

    frame = ROOT.RDataFrame(tree, str(path.resolve()))
    if where:
        frame = frame.Filter(where)
    if max_rows is not None:
        frame = frame.Range(max_rows)
    result = {name: np.asarray(values) for name, values in frame.AsNumpy(branches).items()}
    rows = int(next(iter(result.values())).size) if result else 0
    return result, rows


def load_selection(
    path: Path | None,
    input_rows: int,
    rows_after_filter: int,
    max_rows: int | None,
) -> np.ndarray:
    if path is None:
        return np.ones(rows_after_filter, dtype=bool)
    selection = np.asarray(np.load(path, allow_pickle=False), dtype=bool)
    if selection.ndim != 1 or selection.size != input_rows:
        raise ValueError(f"selection mask has shape {selection.shape}; expected ({input_rows},)")
    if max_rows is not None:
        selection = selection[:max_rows]
    if selection.size != rows_after_filter:
        raise ValueError("selection mask does not align with rows read from the ROOT tree")
    return selection


def file_record(path: Path, include_hash: bool) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, object] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }
    if include_hash:
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    return record


if __name__ == "__main__":
    raise SystemExit(main())
