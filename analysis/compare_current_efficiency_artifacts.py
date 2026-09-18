#!/usr/bin/env python3

"""Compare current-efficiency correction provenance and fitted inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current-efficiency correction artifacts. The first artifact "
            "is the reference for pairwise run, current, class, and yield audits."
        )
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="current_efficiency_correction.json files to compare",
    )
    parser.add_argument(
        "--run-list-limit",
        type=int,
        default=30,
        help="maximum run numbers printed per pairwise difference (default: 30)",
    )
    return parser.parse_args()


def load_artifact(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    for key in ("data_model", "gemc_model"):
        if key not in payload:
            raise ValueError(f"{path} has no {key}")
    return payload


def fractional_slope(model: Mapping) -> tuple[float, float]:
    intercept = float(model["intercept"])
    slope = float(model["slope_per_nA"])
    covariance = np.asarray(model["covariance"], dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("efficiency-model covariance must be 2x2")
    gradient = np.asarray([-slope / intercept**2, 1.0 / intercept])
    variance = float(gradient @ covariance @ gradient)
    return slope / intercept, math.sqrt(max(variance, 0.0))


def fit_included_runs(payload: Mapping) -> set[int]:
    explicit = payload.get("fit_included_runs")
    if explicit is not None:
        return {int(run) for run in explicit}
    return {
        int(run)
        for run, values in payload.get("runs", {}).items()
        if values.get("fit_included", False)
    }


def compact_runs(runs: set[int], limit: int) -> str:
    ordered = sorted(runs)
    shown = " ".join(str(run) for run in ordered[:limit])
    return shown + (" ..." if len(ordered) > limit else "")


def print_summary(path: Path, payload: Mapping) -> None:
    sources = payload.get("sources", {})
    data_fit = sources.get("data_fit", {})
    runs = payload.get("runs", {})
    included = fit_included_runs(payload)
    data_beta, data_sigma = fractional_slope(payload["data_model"])
    gemc_beta, gemc_sigma = fractional_slope(payload["gemc_model"])

    classes: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    currents: list[float] = []
    included_charge_c = 0.0
    for run in included:
        values = runs.get(str(run), {})
        classes[str(values.get("run_class"))] += 1
        qualities[str(values.get("current_quality"))] += 1
        if values.get("current_nA") is not None:
            currents.append(float(values["current_nA"]))
        included_charge_c += float(values.get("charge_c") or 0.0)

    print(f"\n{path}")
    print("  data source =", sources.get("data_sample"))
    print("  manifest =", sources.get("current_manifest"))
    print("  selection mask =", sources.get("selection_mask"))
    print("  background cuts =", sources.get("background_cuts"))
    print("  background SHA256 =", sources.get("background_cuts_sha256"))
    print("  fit model =", data_fit.get("fit_model"))
    print("  fit level =", data_fit.get("fit_level"))
    print("  period classes =", data_fit.get("period_classes"))
    print("  period intercepts =", data_fit.get("period_intercepts_events_per_nC"))
    print("  chi2 / ndf =", data_fit.get("chi2"), "/", data_fit.get("ndf"))
    print("  data fractional slope =", data_beta, "+/-", data_sigma, "nA^-1")
    print("  GEMC fractional slope =", gemc_beta, "+/-", gemc_sigma, "nA^-1")
    print("  included runs =", len(included))
    print("  included charge C =", included_charge_c)
    print("  included classes =", dict(sorted(classes.items())))
    print("  current qualities =", dict(sorted(qualities.items())))
    if currents:
        print(
            "  current min/median/max =",
            min(currents),
            float(np.median(currents)),
            max(currents),
        )
    print(
        "  downstream exclusions =",
        payload.get("analysis_selection", {}).get("excluded_runs"),
    )


def normalized_yield(values: Mapping) -> float | None:
    signal = values.get("signal_events")
    charge = float(values.get("charge_c") or 0.0)
    if signal is None or charge <= 0.0:
        return None
    return float(signal) / charge


def print_pairwise(
    reference_path: Path,
    reference: Mapping,
    path: Path,
    payload: Mapping,
    *,
    run_list_limit: int,
) -> None:
    reference_runs = fit_included_runs(reference)
    compared_runs = fit_included_runs(payload)
    common = reference_runs & compared_runs
    only_reference = reference_runs - compared_runs
    only_compared = compared_runs - reference_runs

    print("\nPAIRWISE")
    print("  A =", reference_path)
    print("  B =", path)
    print("  common fit runs =", len(common))
    print(
        "  only A =",
        len(only_reference),
        compact_runs(only_reference, run_list_limit),
    )
    print(
        "  only B =",
        len(only_compared),
        compact_runs(only_compared, run_list_limit),
    )

    for model_name in ("data_model", "gemc_model"):
        a_beta, a_sigma = fractional_slope(reference[model_name])
        b_beta, b_sigma = fractional_slope(payload[model_name])
        combined_sigma = math.hypot(a_sigma, b_sigma)
        print(f"  {model_name} fractional-slope A-B =", a_beta - b_beta)
        print(
            f"  {model_name} nominal difference significance =",
            (a_beta - b_beta) / combined_sigma if combined_sigma > 0.0 else None,
        )

    reference_records = reference.get("runs", {})
    compared_records = payload.get("runs", {})
    current_differences: list[float] = []
    class_changes: list[tuple[int, object, object]] = []
    yield_ratios: list[float] = []

    for run in common:
        a = reference_records.get(str(run), {})
        b = compared_records.get(str(run), {})
        if a.get("current_nA") is not None and b.get("current_nA") is not None:
            current_differences.append(
                float(a["current_nA"]) - float(b["current_nA"])
            )
        if a.get("run_class") != b.get("run_class"):
            class_changes.append((run, a.get("run_class"), b.get("run_class")))
        a_yield = normalized_yield(a)
        b_yield = normalized_yield(b)
        if a_yield is not None and b_yield not in (None, 0.0):
            yield_ratios.append(a_yield / b_yield)

    if current_differences:
        delta = np.asarray(current_differences)
        print(
            "  A-B current delta min/median/max =",
            float(delta.min()),
            float(np.median(delta)),
            float(delta.max()),
        )
    print("  changed run classes =", len(class_changes))
    for change in class_changes[:run_list_limit]:
        print("   ", change)
    if len(class_changes) > run_list_limit:
        print("    ...")
    if yield_ratios:
        ratios = np.asarray(yield_ratios)
        print(
            "  per-run normalized-yield A/B q05/median/q95 =",
            np.quantile(ratios, [0.05, 0.5, 0.95]),
        )


def main() -> int:
    args = parse_args()
    if args.run_list_limit < 0:
        raise ValueError("--run-list-limit must be nonnegative")
    artifacts = [(path, load_artifact(path)) for path in args.artifacts]
    for path, payload in artifacts:
        print_summary(path, payload)
    reference_path, reference = artifacts[0]
    for path, payload in artifacts[1:]:
        print_pairwise(
            reference_path,
            reference,
            path,
            payload,
            run_list_limit=args.run_list_limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
