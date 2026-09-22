#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_response_closure import render_saved_diagnostics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Rebuild a response-closure diagnostics PDF from closure_results.npz "
            "and closure_metrics.csv without rescanning ROOT inputs."
        )
    )
    result.add_argument("output_dir", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--label")
    return result


def main() -> int:
    args = parser().parse_args()
    destination = render_saved_diagnostics(
        args.output_dir,
        output=args.output,
        label=args.label,
    )
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
