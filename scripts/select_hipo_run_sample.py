#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


RUN_PATTERN = re.compile(r"rec_clas_(\d+)")


def run_number(path: Path) -> int | None:
    match = RUN_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def select_sample(
    input_directory: Path,
    *,
    files_per_run: int,
    run_min: int | None = None,
    run_max: int | None = None,
    max_runs: int | None = None,
    allowed_runs: set[int] | None = None,
) -> dict[int, list[Path]]:
    if files_per_run < 1:
        raise ValueError("files per run must be positive")
    by_run: dict[int, list[Path]] = defaultdict(list)
    for path in input_directory.rglob("*.hipo"):
        run = run_number(path)
        if run is None:
            continue
        if run_min is not None and run < run_min:
            continue
        if run_max is not None and run > run_max:
            continue
        if allowed_runs is not None and run not in allowed_runs:
            continue
        by_run[run].append(path.resolve())
    runs = sorted(by_run)
    if max_runs is not None:
        if max_runs < 2:
            raise ValueError("maximum runs must be at least two")
        if max_runs < len(runs):
            indices = [
                round(index * (len(runs) - 1) / (max_runs - 1))
                for index in range(max_runs)
            ]
            runs = [runs[index] for index in indices]
    return {
        run: evenly_spaced(sorted(by_run[run]), files_per_run) for run in runs
    }


def catalog_runs(path: Path, include_classes: set[str] | None) -> set[int]:
    payload = json.loads(path.read_text())
    entries = payload.get("runs")
    if not isinstance(entries, dict):
        raise ValueError(f"run catalog has no object-valued 'runs' field: {path}")
    selected: set[int] = set()
    for raw_run, metadata in entries.items():
        run = int(raw_run)
        if include_classes is not None:
            run_class = metadata.get("run_class") if isinstance(metadata, dict) else None
            if run_class not in include_classes:
                continue
        selected.add(run)
    if not selected:
        qualifier = (
            f" for classes {sorted(include_classes)}" if include_classes else ""
        )
        raise ValueError(f"run catalog selected no runs{qualifier}: {path}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic, evenly spaced HIPO files within each run and "
            "write a manifest accepted by hipo2root as @manifest."
        )
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--files-per-run", type=int, default=1)
    parser.add_argument("--run-min", type=int)
    parser.add_argument("--run-max", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--run-catalog",
        type=Path,
        help="JSON file with a top-level runs object; only listed runs are eligible",
    )
    parser.add_argument(
        "--include-run-classes",
        nargs="+",
        help="with --run-catalog, retain only runs whose run_class is listed",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_directory.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_directory}")
    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"output manifest already exists: {args.output_manifest}; use --overwrite"
        )
    if args.include_run_classes and args.run_catalog is None:
        raise ValueError("--include-run-classes requires --run-catalog")
    allowed_runs = (
        catalog_runs(
            args.run_catalog,
            set(args.include_run_classes) if args.include_run_classes else None,
        )
        if args.run_catalog is not None
        else None
    )
    selected = select_sample(
        args.input_directory,
        files_per_run=args.files_per_run,
        run_min=args.run_min,
        run_max=args.run_max,
        max_runs=args.max_runs,
        allowed_runs=allowed_runs,
    )
    if not selected:
        raise ValueError("no HIPO files with rec_clas_RUN filenames were found")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w") as output:
        output.write("# One absolute HIPO path per line; generated by select_hipo_run_sample.py\n")
        for run in sorted(selected):
            for path in selected[run]:
                output.write(f"{path}\n")
    file_count = sum(len(paths) for paths in selected.values())
    print(
        f"Wrote {file_count} files spanning {len(selected)} runs "
        f"({min(selected)}-{max(selected)}) to {args.output_manifest}"
    )
    if allowed_runs is not None:
        missing_runs = sorted(allowed_runs.difference(selected))
        print(f"Catalog-eligible runs: {len(allowed_runs)}")
        if missing_runs:
            print(
                "Warning: no matching HIPO files were found for catalog runs: "
                + ", ".join(str(run) for run in missing_runs)
            )


if __name__ == "__main__":
    main()
