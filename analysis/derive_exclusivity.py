#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import from_config
from eppi0.exclusivity import DEFAULT_VARIABLES, apply_cuts, derive_cuts, load_cuts, save_cuts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive one exclusivity definition and apply it to an event sample."
    )
    parser.add_argument("sample", type=Path, help="Reference sample used to derive cuts")
    parser.add_argument("--apply-to", type=Path, help="Optional second sample; defaults to reference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--global-cuts", action="store_true")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--minimum-events", type=int, default=50)
    parser.add_argument("--reuse-cuts", action="store_true")
    return parser.parse_args()


def arrays(sample) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    values = {name: sample[name] for name in DEFAULT_VARIABLES}
    return values, (
        sample["rec_proton_detector"],
        sample["rec_Q2"],
        sample["rec_xB"],
        sample["rec_minus_t"],
    )


def main() -> int:
    args = parse_args()
    binning = from_config(args.config)
    reference = np.load(args.sample, allow_pickle=False)
    reference_values, (detector, q2, xb, minus_t) = arrays(reference)
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    if args.reuse_cuts:
        cuts = load_cuts(str(args.cuts))
    else:
        cuts = derive_cuts(
            reference_values,
            detector,
            iq2,
            ixb,
            it,
            n_sigma=args.n_sigma,
            minimum_events=args.minimum_events,
            global_mode=args.global_cuts,
        )
        args.cuts.parent.mkdir(parents=True, exist_ok=True)
        save_cuts(str(args.cuts), cuts)

    target = np.load(args.apply_to or args.sample, allow_pickle=False)
    target_values, (detector, q2, xb, minus_t) = arrays(target)
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    mask = apply_cuts(cuts, target_values, detector, iq2, ixb, it)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.mask, mask)
    print(f"Cut groups: {cuts.group_ids.size}")
    print(f"Passing events: {mask.sum()}/{mask.size}")
    print(f"Wrote {args.cuts}")
    print(f"Wrote {args.mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
