#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.event_kinematics_diagnostics import (  # noqa: E402
    VARIABLES,
    detector_map_branches,
    reconstructed_topology,
    render_report,
    report_summary,
)
from eppi0.root_trees import resolve  # noqa: E402


TOPOLOGY_BRANCHES = ("pDet", "g1Det", "g2Det")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a canonical multipage PDF of final selected EPPI0 event "
            "kinematics, both topology integrated and split by reconstructed topology."
        )
    )
    parser.add_argument("selected_root", type=Path, help="Post-processed selected ROOT file")
    parser.add_argument("--selection-mask", type=Path, help="Final exclusivity mask aligned to sEvents")
    parser.add_argument("--config", type=Path, required=True, help="Campaign analysis configuration")
    parser.add_argument("--output", type=Path, required=True, help="Output multipage PDF")
    parser.add_argument("--summary", type=Path, help="JSON audit path; defaults beside the PDF")
    parser.add_argument("--label", required=True, help="Human-readable campaign and polarity label")
    parser.add_argument("--tree", default="sEvents", help="Selected event tree (default: sEvents)")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="SHA256 the ROOT input in addition to the mask and config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    arrays, tree_name, input_rows = read_selected_root(
        args.selected_root, args.tree, args.dictionary
    )
    selection = load_selection_mask(args.selection_mask, input_rows)
    topology = reconstructed_topology(arrays)

    summary = report_summary(arrays, selection, topology)
    summary.update(
        {
            "schema_version": 1,
            "label": args.label,
            "selected_root": file_record(args.selected_root, args.hash_inputs),
            "selection_mask": (
                file_record(args.selection_mask, True) if args.selection_mask else None
            ),
            "analysis_config": file_record(args.config, True),
            "tree": tree_name,
            "beam_energy_GeV": float(config["beam_energy"]),
            "topology_definition": "group_id = 4 * proton_detector + FT_photon_count",
            "angle_conventions": {
                "trentoPhi": "stored in radians, wrapped and plotted in degrees",
                "particle_theta_phi": "stored in radians and plotted in degrees",
            },
            "display_range_policy": (
                "fixed physical ranges where specified; otherwise the finite integrated "
                "0.25%-99.75% interval with 5% padding"
            ),
        }
    )
    provenance = [
        f"Selected ROOT: {args.selected_root.resolve()}",
        f"Tree: {tree_name}; input rows: {input_rows:,}",
        (
            f"Selection mask: {args.selection_mask.resolve()}"
            if args.selection_mask
            else "Selection mask: none (all selected ROOT candidates shown)"
        ),
        f"Analysis config: {args.config.resolve()}",
        f"Beam energy: {float(config['beam_energy']):g} GeV",
    ]
    pages = render_report(
        args.output,
        arrays,
        selection,
        topology,
        label=args.label,
        provenance_lines=provenance,
    )
    summary["pdf_pages"] = pages
    summary["output_pdf"] = str(args.output.resolve())
    summary_path = args.summary or args.output.with_name(
        f"{args.output.stem}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_summary.replace(summary_path)
    print(f"Input rows: {input_rows}")
    print(f"Final selected rows: {summary['selected_rows']}")
    print(f"Topology counts: {summary['topology_counts']}")
    print(f"PDF pages: {pages}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary_path.resolve()}")
    return 0


def read_selected_root(
    path: Path, requested_tree: str, dictionary: Path | None
) -> tuple[dict[str, np.ndarray], str, int]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if dictionary:
        status = ROOT.gSystem.Load(str(dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")
    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {root_path}")
    tree_name = resolve(root_file, requested_tree)
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {requested_tree} in {root_path}")
    input_rows = int(tree.GetEntries())
    available = {branch.GetName() for branch in tree.GetListOfBranches()}
    root_file.Close()

    required = set(TOPOLOGY_BRANCHES) | {"Q2", "xB", "t", "trentoPhi"}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Tree {tree_name} is missing required branches: {missing}")
    requested = list(TOPOLOGY_BRANCHES)
    requested.extend(
        variable.branch
        for variable in VARIABLES
        if variable.branch in available and variable.branch not in requested
    )
    requested.extend(
        branch
        for branch in detector_map_branches()
        if branch in available and branch not in requested
    )
    arrays = ROOT.RDataFrame(tree_name, root_path).AsNumpy(requested)
    return {name: np.asarray(values) for name, values in arrays.items()}, tree_name, input_rows


def load_selection_mask(path: Path | None, input_rows: int) -> np.ndarray:
    if path is None:
        return np.ones(input_rows, dtype=bool)
    mask = np.asarray(np.load(path, allow_pickle=False), dtype=bool)
    if mask.ndim != 1 or mask.size != input_rows:
        raise ValueError(
            f"selection mask has shape {mask.shape}; expected ({input_rows},)"
        )
    return mask


def file_record(path: Path, include_hash: bool) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, object] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = sha256(resolved)
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
