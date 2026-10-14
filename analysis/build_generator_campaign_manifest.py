#!/usr/bin/env python3
"""Discover an EXCLUGEN production and write its strict campaign manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eppi0.generators import (
    discover_exclugen_campaign,
    merge_campaign_metadata,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every EXCLUGEN LUND/config pair and write an "
            "sf-generator-campaign-v1 manifest"
        )
    )
    parser.add_argument("production_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generator", choices=("exclugen",), default="exclugen")
    parser.add_argument("--channel", default="pi0p")
    parser.add_argument("--model", default="pi0.amp2614")
    parser.add_argument("--revision")
    parser.add_argument(
        "--execution-path",
        required=True,
        choices=("farm", "local", "local-parallel"),
        help="How exclurad_py was executed; this cannot be inferred safely",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=200,
        help="Required runs per mode; use 0 to accept any positive count",
    )
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="Allow Born and radiative samples from different generator revisions",
    )
    parser.add_argument(
        "--allow-mixed-dirty-state",
        action="store_true",
        help=(
            "Permit clean and dirty stamps at one base revision; intended only "
            "for audited historical retry campaigns"
        ),
    )
    parser.add_argument("--analysis-config")
    parser.add_argument("--v-max", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.expected_runs < 0:
        raise ValueError("--expected-runs must be non-negative")
    if not math.isfinite(args.v_max) or args.v_max < 0.0:
        raise ValueError("--v-max must be finite and non-negative")
    campaign = discover_exclugen_campaign(
        args.production_root,
        execution_path=args.execution_path,
        expected_runs=args.expected_runs or None,
        expected_channel=args.channel,
        expected_model=args.model,
        expected_revision=args.revision,
        allow_revision_mismatch=args.allow_revision_mismatch,
        allow_mixed_dirty_state=args.allow_mixed_dirty_state,
        progress=print,
    )
    campaign = merge_campaign_metadata(
        campaign,
        analysis_config=args.analysis_config,
        radiative_correction={
            "v_definition": "MX2(e'p)-m_pi0^2",
            "v_max_GeV2": args.v_max,
        },
    )
    campaign.write_manifest(args.output)
    summary = {
        "manifest": str(args.output),
        "provider": campaign.provider,
        "channel": campaign.channel,
        "model": campaign.model,
        "born_runs": len(campaign.runs_for_mode("born")),
        "radiative_runs": len(campaign.runs_for_mode("rad")),
        "revisions": sorted({run.generator_revision for run in campaign.runs}),
        "dirty_state_counts": {
            state: sum(run.dirty_state == state for run in campaign.runs)
            for state in sorted({run.dirty_state for run in campaign.runs})
        },
        "execution_paths": sorted({run.execution_path for run in campaign.runs}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
