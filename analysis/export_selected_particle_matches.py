#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.particle_matches import (
    join_selected_particle_matches,
    particle_match_summary,
)
from eppi0.root_trees import R_PARTICLES, S_PARTICLES, resolve


SELECTED_ROOT_COLUMNS = {
    "sourceFileId": "source_file_id",
    "sourceEventIndex": "source_event_index",
    "runNum": "run",
    "eventNum": "event",
    "role": "role",
    "occurrence": "occurrence",
    "particleIdx": "particle_index",
    "pid": "pid",
    "det": "detector",
    "sector": "sector",
    "p": "rec_p",
    "theta": "rec_theta",
    "phi": "rec_phi",
}

RECONSTRUCTED_ROOT_COLUMNS = {
    "event.sourceFileId": "source_file_id",
    "event.sourceEventIndex": "source_event_index",
    "rec.particleIdx": "particle_index",
    "rec.pid": "pid",
    "rec.det": "detector",
    "rec.sector": "sector",
    "rec.p": "rec_p",
    "rec.theta": "rec_theta",
    "rec.phi": "rec_phi",
    "rec.matchedGenIdx": "matched_gen_index",
    "rec.matchAngleDeg": "match_angle_deg",
    "gen.pid": "gen_pid",
    "gen.p": "gen_p",
    "gen.theta": "gen_theta",
    "gen.phi": "gen_phi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join selected sParticles roles to their matched converter "
            "rParticles REC/GEN rows."
        )
    )
    parser.add_argument("converter_root", type=Path)
    parser.add_argument("selected_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--selected-tree", default=S_PARTICLES)
    parser.add_argument("--reconstructed-tree", default=R_PARTICLES)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=10,
        help="Print converter progress every N chunks; zero disables it",
    )
    parser.add_argument(
        "--warning-angle-deg",
        type=float,
        default=2.5,
        help="Threshold summarized as near the configured 3-degree match limit",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="JSON summary path; defaults beside the output NPZ",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.progress_chunks < 0:
        raise ValueError("--progress-chunks must be non-negative")

    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary:
        status = ROOT.gSystem.Load(str(args.dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    selected_tree, selected_entries = resolve_tree(
        ROOT,
        args.selected_root,
        args.selected_tree,
        tuple(SELECTED_ROOT_COLUMNS),
    )
    reconstructed_tree, reconstructed_entries = resolve_tree(
        ROOT,
        args.converter_root,
        args.reconstructed_tree,
        tuple(RECONSTRUCTED_ROOT_COLUMNS),
    )
    selected_raw = ROOT.RDataFrame(
        selected_tree, str(args.selected_root.resolve())
    ).AsNumpy(list(SELECTED_ROOT_COLUMNS))
    selected = rename_columns(selected_raw, SELECTED_ROOT_COLUMNS)
    selected["role"] = np.asarray([str(value) for value in selected["role"]])
    if selected_entries != len(selected["role"]):
        raise RuntimeError(
            f"selected ROOT read returned {len(selected['role'])} rows; "
            f"tree reports {selected_entries}"
        )

    chunks = reconstructed_particle_chunks(
        ROOT,
        args.converter_root,
        reconstructed_tree,
        reconstructed_entries,
        args.chunk_size,
        args.progress_chunks,
    )
    matches, stats = join_selected_particle_matches(selected, chunks)
    summary = particle_match_summary(
        matches,
        stats,
        warning_angle_deg=args.warning_angle_deg,
    )
    summary.update(
        schema_version=1,
        converter_root=str(args.converter_root.resolve()),
        selected_root=str(args.selected_root.resolve()),
        reconstructed_tree=reconstructed_tree,
        selected_tree=selected_tree,
        chunk_size=args.chunk_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **matches,
        metadata_json=json.dumps(summary, sort_keys=True),
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return 0


def resolve_tree(
    ROOT,
    path: Path,
    requested: str,
    columns: tuple[str, ...],
) -> tuple[str, int]:
    resolved_path = str(path.resolve())
    root_file = ROOT.TFile.Open(resolved_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {resolved_path}")
    try:
        tree_name = resolve(root_file, requested)
        tree = root_file.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Could not find tree {requested} in {resolved_path}")
        missing = [
            name
            for name in columns
            if not tree.GetBranch(name) and not tree.GetLeaf(name)
        ]
        if missing:
            raise RuntimeError(
                f"Tree {tree_name} in {resolved_path} is missing branches: {missing}"
            )
        entries = int(tree.GetEntries())
    finally:
        root_file.Close()
    if tree_name != requested:
        print(f"Warning: using compatible tree {tree_name}", file=sys.stderr)
    return tree_name, entries


def reconstructed_particle_chunks(
    ROOT,
    path: Path,
    tree_name: str,
    entries: int,
    chunk_size: int,
    progress_chunks: int,
) -> Iterable[dict[str, np.ndarray]]:
    resolved_path = str(path.resolve())
    chunks = (entries + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, entries, chunk_size), start=1):
        stop = min(start + chunk_size, entries)
        raw = (
            ROOT.RDataFrame(tree_name, resolved_path)
            .Range(start, stop)
            .AsNumpy(list(RECONSTRUCTED_ROOT_COLUMNS))
        )
        yield rename_columns(raw, RECONSTRUCTED_ROOT_COLUMNS)
        if progress_chunks and (
            chunk_index % progress_chunks == 0 or chunk_index == chunks
        ):
            print(
                f"rParticles chunks: {chunk_index}/{chunks} "
                f"({stop}/{entries} rows)"
            )


def rename_columns(
    raw: dict[str, np.ndarray], mapping: dict[str, str]
) -> dict[str, np.ndarray]:
    return {output: np.asarray(raw[source]) for source, output in mapping.items()}


def print_summary(summary: dict[str, object]) -> None:
    selected = int(summary["selected_rows"])
    matched = int(summary["generated_matches"])
    fraction = matched / selected if selected else float("nan")
    print(f"Selected particle rows: {selected}")
    print(f"Converter rows scanned: {summary['converter_rows_scanned']}")
    print(f"Converter rows found: {summary['converter_rows_found']}")
    print(f"Generated matches: {matched}/{selected} ({fraction:.6f})")
    for group in summary["groups"]:
        group_selected = int(group["selected_rows"])
        group_matched = int(group["matched_rows"])
        group_fraction = group_matched / group_selected if group_selected else float("nan")
        print(
            f"  role={group['role']}, detector={group['detector']}: "
            f"matched={group_matched}/{group_selected} "
            f"({group_fraction:.6f}), median angle="
            f"{_format_optional(group['match_angle_median_deg'])} deg, p95="
            f"{_format_optional(group['match_angle_p95_deg'])} deg"
        )


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
