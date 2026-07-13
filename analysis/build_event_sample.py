#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.event_sample import (
    build_generated_sample,
    generated_particle_columns,
    generated_sample_from_tree,
    join_reconstructed,
)


GEN_COLUMNS = [
    "event.runNum",
    "event.eventNum",
    "gen.pid",
    "gen.p",
    "gen.theta",
    "gen.phi",
]

GEN_SOURCE_COLUMNS = ["event.sourceFileId", "event.sourceEventIndex"]

GENERATED_SOURCE_COLUMNS = ["sourceFileId", "sourceEventIndex"]

GENERATED_EVENT_COLUMNS = [
    "runNum",
    "eventNum",
    "topologyValid",
    "Q2",
    "xB",
    "minusT",
    "trentoPhi",
    "radiative",
    "weight",
]

GENERATED_EVENT_PARTICLE_COLUMNS = [
    "electronP",
    "electronTheta",
    "electronPhi",
    "protonP",
    "protonTheta",
    "protonPhi",
    "gamma1P",
    "gamma1Theta",
    "gamma1Phi",
    "gamma2P",
    "gamma2Theta",
    "gamma2Phi",
    "pi0P",
    "pi0Theta",
    "pi0Phi",
]

REC_SOURCE_COLUMNS = ["sourceFileId", "sourceEventIndex"]

REC_KEY_COLUMNS = {"runNum", "eventNum", *REC_SOURCE_COLUMNS}

REC_COLUMN_ALIASES = {
    "Q2": "rec_Q2",
    "xB": "rec_xB",
    "t": "rec_minus_t",
    "t_pi0": "rec_minus_t_pi0",
    "trentoPhi": "rec_trento_phi",
    "pDet": "rec_proton_detector",
    "m_gg": "rec_m_gg",
    "m2_miss": "rec_m2_miss",
    "m2_epX": "rec_m2_epX",
    "m_eggX": "rec_m_eggX",
    "E_miss": "rec_E_miss",
    "pT_miss": "rec_pT_miss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join all generated EPPI0 events to selected reconstructed candidates."
    )
    parser.add_argument("matched_root", type=Path, help="Converter ROOT file")
    parser.add_argument("selected_root", type=Path, help="Output from post_process")
    parser.add_argument("output", type=Path, help="Compact event-level .npz output")
    parser.add_argument("--beam-energy", type=float,
                        help="Required only for legacy particle-level GEN input")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--generated-tree", default="GeneratedEvents")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary:
        status = ROOT.gSystem.Load(str(args.dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    matched_path = str(args.matched_root.resolve())
    input_file = ROOT.TFile.Open(matched_path, "READ")
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open converter ROOT file: {matched_path}")
    generated_tree = input_file.Get(args.generated_tree)
    has_generated_tree = bool(generated_tree)
    has_generated_source_key = has_generated_tree and all(
        generated_tree.GetBranch(name) for name in GENERATED_SOURCE_COLUMNS
    )
    generated_event_particle_columns = [
        name for name in GENERATED_EVENT_PARTICLE_COLUMNS
        if has_generated_tree and generated_tree.GetBranch(name)
    ]
    input_file.Close()

    if has_generated_tree:
        requested_gen_columns = (
            GENERATED_EVENT_COLUMNS
            + (GENERATED_SOURCE_COLUMNS if has_generated_source_key else [])
            + generated_event_particle_columns
        )
        gen = ROOT.RDataFrame(args.generated_tree, matched_path).AsNumpy(requested_gen_columns)
        generated_rows = np.asarray(gen["runNum"]).size
        generated_mask = generated_event_mask(gen)
        generated = generated_sample_from_tree(
            gen["sourceFileId"] if has_generated_source_key else np.full(
                generated_rows, np.iinfo(np.uint64).max, dtype=np.uint64
            ),
            gen["sourceEventIndex"] if has_generated_source_key else np.full(
                generated_rows, np.iinfo(np.uint64).max, dtype=np.uint64
            ),
            gen["runNum"],
            gen["eventNum"],
            gen["topologyValid"],
            gen["Q2"],
            gen["xB"],
            gen["minusT"],
            gen["trentoPhi"],
            gen["radiative"],
            gen["weight"],
        )
        generated_source = args.generated_tree
        generated_values = {
            f"gen_{name}": np.asarray(gen[name], dtype=float)[generated_mask]
            for name in generated_event_particle_columns
        }
    else:
        if args.beam_energy is None:
            raise ValueError("--beam-energy is required when GeneratedEvents is absent")
        gen = ROOT.RDataFrame(args.tree, matched_path).AsNumpy(GEN_COLUMNS)
        generated = build_generated_sample(
            gen["event.runNum"],
            gen["event.eventNum"],
            gen["gen.pid"],
            gen["gen.p"],
            gen["gen.theta"],
            gen["gen.phi"],
            args.beam_energy,
        )
        generated_source = f"{args.tree}.gen (legacy)"
        generated_values = {}

    if not generated_values:
        generated_values = read_generated_particle_columns(
            ROOT,
            matched_path,
            args.tree,
            generated,
            prefer_source_keys=has_generated_source_key,
        )

    selected_path = str(args.selected_root.resolve())
    selected_file = ROOT.TFile.Open(selected_path, "READ")
    if not selected_file or selected_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {selected_path}")
    selected_tree = selected_file.Get(args.tree)
    if not selected_tree:
        raise RuntimeError(f"Could not find tree {args.tree} in {selected_path}")
    has_source_key = all(selected_tree.GetBranch(name) for name in REC_SOURCE_COLUMNS)
    requested_rec_columns = scalar_branch_names(selected_tree)
    selected_file.Close()
    missing_keys = [name for name in ("runNum", "eventNum") if name not in requested_rec_columns]
    if missing_keys:
        raise RuntimeError(f"Selected tree is missing required branches: {missing_keys}")
    if has_source_key:
        for name in REC_SOURCE_COLUMNS:
            if name not in requested_rec_columns:
                requested_rec_columns.append(name)
    rec = ROOT.RDataFrame(args.tree, selected_path).AsNumpy(requested_rec_columns)
    rec_values = reconstructed_columns(rec, requested_rec_columns)
    sample = join_reconstructed(
        generated,
        rec["runNum"],
        rec["eventNum"],
        rec_values,
        rec_source_file_id=rec["sourceFileId"] if has_source_key else None,
        rec_source_event_index=rec["sourceEventIndex"] if has_source_key else None,
    )
    sample.update(generated_values)
    metadata = {
        "beam_energy": args.beam_energy,
        "generated_source": generated_source,
        "generated_particle_columns": sorted(generated_values),
        "matched_root": str(args.matched_root.resolve()),
        "selected_root": str(args.selected_root.resolve()),
        "generated_events": int(generated.run.size),
        "selected_reconstructed_events": int(sample["rec_selected"].sum()),
        "reconstructed_columns": sorted(rec_values),
        "schema_version": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sample, metadata_json=json.dumps(metadata, sort_keys=True))
    print(f"Generated events: {generated.run.size}")
    print(f"Selected REC matches: {sample['rec_selected'].sum()}")
    print(f"GEN particle variables carried: {len(generated_values)}")
    print(f"REC variables carried: {len(rec_values)}")
    print(f"Wrote {args.output}")
    return 0


def generated_event_mask(gen: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(gen["topologyValid"], dtype=bool)
    for name in ("Q2", "xB", "minusT", "trentoPhi"):
        valid &= np.isfinite(np.asarray(gen[name], dtype=float))
    return valid


def read_generated_particle_columns(ROOT, root_path: str, tree_name: str, generated, prefer_source_keys: bool) -> dict[str, np.ndarray]:
    column_sets = []
    if prefer_source_keys:
        column_sets.append(GEN_SOURCE_COLUMNS + GEN_COLUMNS)
    column_sets.append(GEN_COLUMNS)
    if not prefer_source_keys:
        column_sets.append(GEN_SOURCE_COLUMNS + GEN_COLUMNS)

    last_error: Exception | None = None
    for columns in column_sets:
        try:
            raw = ROOT.RDataFrame(tree_name, root_path).AsNumpy(columns)
        except Exception as exc:  # ROOT raises RuntimeError for missing dictionaries/branches.
            last_error = exc
            continue
        has_source = all(name in raw for name in GEN_SOURCE_COLUMNS)
        return generated_particle_columns(
            raw["event.runNum"],
            raw["event.eventNum"],
            raw["gen.pid"],
            raw["gen.p"],
            raw["gen.theta"],
            raw["gen.phi"],
            generated.run,
            generated.event,
            source_file_id=raw["event.sourceFileId"] if has_source else None,
            source_event_index=raw["event.sourceEventIndex"] if has_source else None,
            target_source_file_id=generated.source_file_id if has_source else None,
            target_source_event_index=generated.source_event_index if has_source else None,
        )
    message = f"Warning: could not read per-particle GEN kinematics from {tree_name}.gen"
    if last_error is not None:
        message += f": {last_error}"
    print(message, file=sys.stderr)
    return {}


def scalar_branch_names(tree) -> list[str]:
    names: list[str] = []
    for branch in tree.GetListOfBranches():
        name = branch.GetName()
        class_name = str(branch.GetClassName() or "")
        if class_name:
            continue
        leaves = branch.GetListOfLeaves()
        if not leaves or leaves.GetEntries() != 1:
            continue
        names.append(name)
    return names


def reconstructed_columns(rec: dict[str, np.ndarray], columns: list[str]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in columns:
        if name in REC_KEY_COLUMNS:
            continue
        if name not in rec:
            continue
        output_name = REC_COLUMN_ALIASES.get(name, f"rec_{name}")
        output.setdefault(output_name, rec[name])
    return output


if __name__ == "__main__":
    raise SystemExit(main())
