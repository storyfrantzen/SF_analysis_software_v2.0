#!/usr/bin/env python3
"""Map final HIPO simulation files to validated generator run provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eppi0.generators import build_generator_source_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an sf-generator-source-manifest-v1 mapping final HIPO files "
            "to one mode of a validated generator campaign"
        )
    )
    parser.add_argument("campaign_manifest", type=Path)
    parser.add_argument("--mode", choices=("born", "rad"), required=True)
    parser.add_argument("--hipo-directory", type=Path, required=True)
    parser.add_argument("--hipo-glob", default="*.hipo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow campaign run indices without a final HIPO mapping",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.hipo_directory.is_dir():
        raise ValueError(f"HIPO directory does not exist: {args.hipo_directory}")
    hipo_paths = sorted(args.hipo_directory.glob(args.hipo_glob))
    manifest = build_generator_source_manifest(
        args.campaign_manifest,
        hipo_paths,
        mode=args.mode,
        require_all_runs=not args.allow_partial,
    )
    manifest.write_manifest(args.output)
    summary = {
        "manifest": str(args.output),
        "schema": "sf-generator-source-manifest-v1",
        "mode": args.mode,
        "sources": len(manifest.sources),
        "providers": sorted({source.run.provider for source in manifest.sources}),
        "models": sorted({source.run.model for source in manifest.sources}),
        "revisions": sorted(
            {source.run.generator_revision for source in manifest.sources}
        ),
        "response_event_weight": 1.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
