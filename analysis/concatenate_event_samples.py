#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate compact MC event-sample NPZ chunks."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-duplicate-events",
        action="store_true",
        help="Permit repeated (run,event) keys; rejected by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = [np.load(path, allow_pickle=False) for path in args.inputs]
    if not chunks:
        raise ValueError("at least one input sample is required")

    fields = tuple(name for name in chunks[0].files if name != "metadata_json")
    for path, chunk in zip(args.inputs, chunks, strict=True):
        chunk_fields = tuple(name for name in chunk.files if name != "metadata_json")
        if set(chunk_fields) != set(fields):
            raise ValueError(f"sample fields do not match in {path}")
        lengths = {np.asarray(chunk[name]).shape[0] for name in fields}
        if len(lengths) != 1:
            raise ValueError(f"event arrays have inconsistent lengths in {path}")

    combined = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in fields}
    if not args.allow_duplicate_events:
        keys = np.empty(
            combined["run"].size, dtype=[("run", "<i8"), ("event", "<i8")]
        )
        keys["run"] = combined["run"]
        keys["event"] = combined["event"]
        if np.unique(keys).size != keys.size:
            raise ValueError("duplicate (run,event) keys found across event-sample chunks")

    metadata = {
        "schema_version": 1,
        "input_samples": [str(path.resolve()) for path in args.inputs],
        "chunks": len(chunks),
        "events": int(combined["run"].size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **combined,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Combined {len(chunks)} chunks and {combined['run'].size} events")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
