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

REC_COLUMNS = [
    "runNum",
    "eventNum",
    "Q2",
    "xB",
    "t",
    "trentoPhi",
    "pDet",
    "m_gg",
    "m2_miss",
    "m2_epX",
    "m_eggX",
    "E_miss",
    "pT_miss",
]

REC_SOURCE_COLUMNS = ["sourceFileId", "sourceEventIndex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join all generated EPPI0 events to selected reconstructed candidates."
    )
    parser.add_argument("matched_root", type=Path, help="Converter ROOT file")
    parser.add_argument("selected_root", type=Path, help="Output from apply_cuts")
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
    input_file.Close()

    if has_generated_tree:
        requested_gen_columns = GENERATED_EVENT_COLUMNS + (
            GENERATED_SOURCE_COLUMNS if has_generated_source_key else []
        )
        gen = ROOT.RDataFrame(args.generated_tree, matched_path).AsNumpy(requested_gen_columns)
        generated_rows = np.asarray(gen["runNum"]).size
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

    selected_path = str(args.selected_root.resolve())
    selected_file = ROOT.TFile.Open(selected_path, "READ")
    if not selected_file or selected_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {selected_path}")
    selected_tree = selected_file.Get(args.tree)
    if not selected_tree:
        raise RuntimeError(f"Could not find tree {args.tree} in {selected_path}")
    has_source_key = all(selected_tree.GetBranch(name) for name in REC_SOURCE_COLUMNS)
    selected_file.Close()
    requested_rec_columns = REC_COLUMNS + (REC_SOURCE_COLUMNS if has_source_key else [])
    rec = ROOT.RDataFrame(args.tree, selected_path).AsNumpy(requested_rec_columns)
    rec_values = {
        "rec_Q2": rec["Q2"],
        "rec_xB": rec["xB"],
        "rec_minus_t": rec["t"],
        "rec_trento_phi": rec["trentoPhi"],
        "rec_proton_detector": rec["pDet"],
        "rec_m_gg": rec["m_gg"],
        "rec_m2_miss": rec["m2_miss"],
        "rec_m2_epX": rec["m2_epX"],
        "rec_m_eggX": rec["m_eggX"],
        "rec_E_miss": rec["E_miss"],
        "rec_pT_miss": rec["pT_miss"],
    }
    sample = join_reconstructed(
        generated,
        rec["runNum"],
        rec["eventNum"],
        rec_values,
        rec_source_file_id=rec["sourceFileId"] if has_source_key else None,
        rec_source_event_index=rec["sourceEventIndex"] if has_source_key else None,
    )
    metadata = {
        "beam_energy": args.beam_energy,
        "generated_source": generated_source,
        "matched_root": str(args.matched_root.resolve()),
        "selected_root": str(args.selected_root.resolve()),
        "generated_events": int(generated.run.size),
        "selected_reconstructed_events": int(sample["rec_selected"].sum()),
        "schema_version": 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sample, metadata_json=json.dumps(metadata, sort_keys=True))
    print(f"Generated events: {generated.run.size}")
    print(f"Selected REC matches: {sample['rec_selected'].sum()}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
