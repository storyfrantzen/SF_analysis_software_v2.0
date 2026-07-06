from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .proton_energy_loss import (
    DEFAULT_CONFIGS,
    RESIDUAL_COLUMNS,
    evaluate_correction,
    filtered_arrays,
)
from .plot_utils import save_plot


def load_params(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def residual_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"entries": 0, "mean": np.nan, "std": np.nan, "rms": np.nan}
    return {
        "entries": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["detector", "residual", "implementation", "entries", "mean", "std", "rms"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    print("detector residual implementation entries mean std rms")
    for row in rows:
        print(
            f"{row['detector']} {row['residual']} {row['implementation']} "
            f"{row['entries']} {row['mean']:.6g} {row['std']:.6g} {row['rms']:.6g}"
        )


def plot_comparison(plot_dir: Path,
                    detector_name: str,
                    residual_name: str,
                    before: np.ndarray,
                    baseline_after: np.ndarray,
                    updated_after: np.ndarray,
                    dataset_tag: str,
                    beam_energy: float | None) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    finite = np.concatenate([
        before[np.isfinite(before)],
        baseline_after[np.isfinite(baseline_after)],
        updated_after[np.isfinite(updated_after)],
    ])
    if finite.size == 0:
        return
    lo, hi = np.quantile(finite, [0.01, 0.99])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        lo -= 0.5
        hi += 0.5
    bins = np.linspace(lo, hi, 81)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(before, bins=bins, histtype="step", label="uncorrected", color="0.4")
    ax.hist(baseline_after, bins=bins, histtype="step", label="baseline", color="tab:blue")
    ax.hist(updated_after, bins=bins, histtype="step", label="updated", color="tab:orange")
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(residual_name)
    ax.set_ylabel("entries")
    ax.set_title(f"{detector_name} {residual_name} correction comparison")
    ax.legend()
    save_plot(
        fig,
        plot_dir / f"{detector_name}_{residual_name}_comparison.png",
        f"{detector_name} {residual_name} correction comparison",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two proton energy-loss correction files on one matched REC/GEN sample."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("baseline_params", type=Path)
    parser.add_argument("updated_params", type=Path)
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--detector", choices=["FD", "CD", "both"], default="both")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--updated-label", default="updated")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--beam-energy", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_params(args.baseline_params)
    updated = load_params(args.updated_params)
    detectors = ["FD", "CD"] if args.detector == "both" else [args.detector]

    rows: list[dict[str, object]] = []
    for detector_name in detectors:
        cfg = DEFAULT_CONFIGS[detector_name]
        arrays = filtered_arrays(args.input_file, args.tree, cfg, args.max_rows)
        p = arrays["rec.p"]
        theta = arrays["theta_deg"]

        for residual_name, column in RESIDUAL_COLUMNS.items():
            key = f"p_{residual_name}_{detector_name}"
            if key not in baseline or key not in updated:
                print(f"[WARN] Skipping missing correction term: {key}")
                continue

            before = arrays[column]
            baseline_after = before - evaluate_correction(baseline[key], p, theta)
            updated_after = before - evaluate_correction(updated[key], p, theta)
            series = [
                ("uncorrected", before),
                (args.baseline_label, baseline_after),
                (args.updated_label, updated_after),
            ]
            for label, values in series:
                stats = residual_stats(values)
                rows.append({
                    "detector": detector_name,
                    "residual": residual_name,
                    "implementation": label,
                    **stats,
                })

            if args.plot_dir:
                plot_comparison(
                    args.plot_dir,
                    detector_name,
                    residual_name,
                    before,
                    baseline_after,
                    updated_after,
                    args.dataset_tag,
                    args.beam_energy,
                )

    print_rows(rows)
    if args.output:
        write_csv(args.output, rows)
        print(f"Wrote comparison summary to {args.output}")


if __name__ == "__main__":
    main()
