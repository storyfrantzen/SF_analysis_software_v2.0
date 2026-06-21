#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.event_sample import build_generated_sample, join_reconstructed


GEN_COLUMNS = [
    "event.runNum",
    "event.eventNum",
    "gen.pid",
    "gen.p",
    "gen.theta",
    "gen.phi",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join all generated EPPI0 events to selected reconstructed candidates."
    )
    parser.add_argument("matched_root", type=Path, help="Unskimmed matched converter ROOT file")
    parser.add_argument("selected_root", type=Path, help="Output from apply_cuts")
    parser.add_argument("output", type=Path, help="Compact event-level .npz output")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument("--tree", default="Events")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary:
        status = ROOT.gSystem.Load(str(args.dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    gen = ROOT.RDataFrame(args.tree, str(args.matched_root.resolve())).AsNumpy(GEN_COLUMNS)
    generated = build_generated_sample(
        gen["event.runNum"],
        gen["event.eventNum"],
        gen["gen.pid"],
        gen["gen.p"],
        gen["gen.theta"],
        gen["gen.phi"],
        args.beam_energy,
    )

    rec = ROOT.RDataFrame(args.tree, str(args.selected_root.resolve())).AsNumpy(REC_COLUMNS)
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
        generated, rec["runNum"], rec["eventNum"], rec_values
    )
    metadata = {
        "beam_energy": args.beam_energy,
        "matched_root": str(args.matched_root.resolve()),
        "selected_root": str(args.selected_root.resolve()),
        "generated_events": int(generated.run.size),
        "selected_reconstructed_events": int(sample["rec_selected"].sum()),
        "schema_version": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sample, metadata_json=json.dumps(metadata, sort_keys=True))
    print(f"Generated events: {generated.run.size}")
    print(f"Selected REC matches: {sample['rec_selected'].sum()}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
