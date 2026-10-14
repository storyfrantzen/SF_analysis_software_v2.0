#!/usr/bin/env python3
"""Estimate the effect of the published Klimenko photon-efficiency map.

This is deliberately a diagnostic, not the production DATA correction.  The
map values below are the bin annotations in Figure 1 of
``Klimenko_analysis_methods_update.pdf``.  They are applied as constants in
their displayed bins.  The production analysis instead uses the delivered
map, validity metadata, data-weighted centers, and trilinear interpolation.

The map is an FD/PCAL calibration.  FT photons receive a neutral factor.  An
event enters the sensitivity sample only when every FD photon is inside the
displayed nominal calibration domain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


P_EDGES = np.asarray([0.5, 1.1, 1.7, 2.3, 2.9, 3.9], dtype=float)
THETA_EDGES = np.asarray([6.0, 20.0, 27.0, 36.0], dtype=float)
PHI_EDGES = np.asarray([-30.0, 30.0, 90.0, 150.0, 210.0, 270.0, 330.0], dtype=float)

# Axis order is momentum, theta, wrapped phi.  Values are epsilon_DATA / epsilon_MC.
CORRECTION = np.asarray(
    [
        [
            [0.71, 0.73, 0.84, 0.88, 0.69, 0.79],
            [0.83, 0.80, 0.89, 0.90, 0.77, 0.82],
            [0.72, 0.77, 0.87, 0.74, 0.73, 0.82],
        ],
        [
            [1.02, 0.92, 0.99, 1.04, 0.86, 0.99],
            [1.04, 1.00, 1.03, 1.07, 0.94, 1.04],
            [0.80, 0.91, 1.02, 1.01, 0.85, 1.06],
        ],
        [
            [1.03, 1.05, 1.03, 1.15, 0.92, 1.06],
            [1.12, 1.08, 1.10, 1.24, 1.06, 1.09],
            [1.11, 1.10, 1.00, 1.00, 1.00, 1.20],
        ],
        [
            [1.00, 1.04, 1.06, 1.09, 0.96, 1.06],
            [1.09, 1.08, 1.07, 1.15, 0.99, 1.00],
            [1.01, 1.12, 1.04, 1.02, 0.99, 1.11],
        ],
        [
            [0.84, 0.85, 0.89, 0.88, 0.75, 0.86],
            [0.98, 0.90, 0.96, 0.96, 0.85, 0.94],
            [1.10, 1.00, 1.00, 1.00, 1.00, 1.00],
        ],
    ],
    dtype=float,
)

FT_DETECTOR = 0
FD_DETECTOR = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Event-sample NPZ")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="AAO radiative candidate sample")
    return parser.parse_args()


def wrapped_phi_degrees(phi_radians: np.ndarray) -> np.ndarray:
    degrees = np.rad2deg(np.asarray(phi_radians, dtype=float))
    return np.mod(degrees + 30.0, 360.0) - 30.0


def map_values(momentum: np.ndarray, theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    momentum = np.asarray(momentum, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)
    if momentum.shape != theta_deg.shape or momentum.shape != phi_deg.shape:
        raise ValueError("photon coordinate arrays must have equal shapes")

    wrapped_phi = np.mod(phi_deg + 30.0, 360.0) - 30.0
    ip = np.searchsorted(P_EDGES, momentum, side="right") - 1
    it = np.searchsorted(THETA_EDGES, theta_deg, side="right") - 1
    iph = np.searchsorted(PHI_EDGES, wrapped_phi, side="right") - 1
    valid = np.isfinite(momentum) & np.isfinite(theta_deg) & np.isfinite(wrapped_phi)
    valid &= (ip >= 0) & (ip < CORRECTION.shape[0])
    valid &= (it >= 0) & (it < CORRECTION.shape[1])
    valid &= (iph >= 0) & (iph < CORRECTION.shape[2])
    result = np.full(momentum.shape, np.nan, dtype=float)
    result[valid] = CORRECTION[ip[valid], it[valid], iph[valid]]
    return result


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "q10": float("nan"), "q90": float("nan")}
    q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(median),
        "q10": float(q10),
        "q90": float(q90),
    }


def topology_name(detector1: np.ndarray, detector2: np.ndarray) -> np.ndarray:
    result = np.full(detector1.shape, "other", dtype="U5")
    result[(detector1 == FD_DETECTOR) & (detector2 == FD_DETECTOR)] = "FD/FD"
    result[((detector1 == FD_DETECTOR) & (detector2 == FT_DETECTOR)) |
           ((detector1 == FT_DETECTOR) & (detector2 == FD_DETECTOR))] = "FD/FT"
    result[(detector1 == FT_DETECTOR) & (detector2 == FT_DETECTOR)] = "FT/FT"
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.sample, allow_pickle=False) as source:
        required = (
            "rec_selected", "rec_gamma1P", "rec_gamma2P",
            "rec_gamma1Theta", "rec_gamma2Theta",
            "rec_gamma1Phi", "rec_gamma2Phi",
            "rec_gamma1_detector", "rec_gamma2_detector",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"event sample is missing: {missing}")
        selected = np.asarray(source["rec_selected"], dtype=bool)
        p1 = np.asarray(source["rec_gamma1P"], dtype=float)
        p2 = np.asarray(source["rec_gamma2P"], dtype=float)
        theta1 = np.rad2deg(np.asarray(source["rec_gamma1Theta"], dtype=float))
        theta2 = np.rad2deg(np.asarray(source["rec_gamma2Theta"], dtype=float))
        phi1 = wrapped_phi_degrees(source["rec_gamma1Phi"])
        phi2 = wrapped_phi_degrees(source["rec_gamma2Phi"])
        detector1 = np.asarray(source["rec_gamma1_detector"], dtype=int)
        detector2 = np.asarray(source["rec_gamma2_detector"], dtype=int)
        generated_electron_theta = (
            np.rad2deg(np.asarray(source["gen_electronTheta"], dtype=float))
            if "gen_electronTheta" in source else None
        )

    correction1 = map_values(p1, theta1, phi1)
    correction2 = map_values(p2, theta2, phi2)
    covered1 = (detector1 == FT_DETECTOR) | (
        (detector1 == FD_DETECTOR) & np.isfinite(correction1)
    )
    covered2 = (detector2 == FT_DETECTOR) | (
        (detector2 == FD_DETECTOR) & np.isfinite(correction2)
    )
    covered = selected & covered1 & covered2

    factor1 = np.ones(selected.shape, dtype=float)
    factor2 = np.ones(selected.shape, dtype=float)
    fd1 = covered & (detector1 == FD_DETECTOR)
    fd2 = covered & (detector2 == FD_DETECTOR)
    factor1[fd1] = 1.0 / correction1[fd1]
    factor2[fd2] = 1.0 / correction2[fd2]
    event_weight = factor1 * factor2
    weights = event_weight[covered]
    topology = topology_name(detector1, detector2)

    selected_count = int(np.count_nonzero(selected))
    covered_count = int(np.count_nonzero(covered))
    summary: dict[str, object] = {
        "label": args.label,
        "input": str(args.sample.resolve()),
        "method": "figure-digitized bin-constant sensitivity; not production interpolation",
        "selected_events": selected_count,
        "covered_events": covered_count,
        "covered_fraction": covered_count / selected_count if selected_count else 0.0,
        "event_weight": finite_stats(weights),
        "weighted_to_unweighted_yield": float(np.sum(weights) / covered_count)
        if covered_count else float("nan"),
        "fraction_weight_above_1p10": float(np.mean(weights > 1.10)) if covered_count else float("nan"),
        "fraction_weight_below_0p90": float(np.mean(weights < 0.90)) if covered_count else float("nan"),
        "topologies": {},
    }
    theta_rows: list[dict[str, float | int]] = []
    if generated_electron_theta is not None:
        for threshold in (5.0, 5.5, 6.0, 6.5, 7.0):
            retained = covered & (generated_electron_theta >= threshold)
            count = int(np.count_nonzero(retained))
            theta_rows.append(
                {
                    "theta_min_deg": threshold,
                    "events": count,
                    "retained_fraction": count / covered_count if covered_count else 0.0,
                }
            )
        summary["generated_electron_theta_retention_map_covered"] = theta_rows
    topology_rows: list[dict[str, object]] = []
    for name in ("FD/FD", "FD/FT", "FT/FT", "other"):
        mask = covered & (topology == name)
        row = {
            "topology": name,
            "events": int(np.count_nonzero(mask)),
            "selected_fraction": float(np.count_nonzero(mask) / selected_count)
            if selected_count else 0.0,
            "weight": finite_stats(event_weight[mask]),
        }
        topology_rows.append(row)
        summary["topologies"][name] = row

    summary_path = args.output_dir / "klimenko_map_sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "klimenko_map_sensitivity_topologies.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("topology", "events", "selected_fraction", "weight_mean", "weight_median",
                         "weight_q10", "weight_q90"))
        for row in topology_rows:
            stats = row["weight"]
            writer.writerow((row["topology"], row["events"], row["selected_fraction"],
                             stats["mean"], stats["median"], stats["q10"], stats["q90"]))
    theta_path = args.output_dir / "klimenko_map_covered_electron_theta_retention.csv"
    if theta_rows:
        with theta_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("theta_min_deg", "events", "retained_fraction")
            )
            writer.writeheader()
            writer.writerows(theta_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, momentum, title in (
        (axes[0, 0], p1, r"$p_{\gamma_1}$ [GeV]"),
        (axes[0, 1], p2, r"$p_{\gamma_2}$ [GeV]"),
    ):
        edges = np.linspace(0.5, 3.9, 70)
        axis.hist(momentum[covered], bins=edges, histtype="step", linewidth=1.4,
                  label="unweighted")
        axis.hist(momentum[covered], bins=edges, weights=weights, histtype="step",
                  linewidth=1.4, label="map sensitivity weighted")
        axis.set_xlabel(title)
        axis.set_ylabel("events / bin")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[1, 0].hist(weights, bins=70, histtype="stepfilled", alpha=0.35)
    axes[1, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel(r"event sensitivity weight $1/(C_1C_2)$")
    axes[1, 0].set_ylabel("events")
    names = [row["topology"] for row in topology_rows if row["events"]]
    means = [row["weight"]["mean"] for row in topology_rows if row["events"]]
    axes[1, 1].bar(names, means)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("mean event sensitivity weight")
    axes[1, 1].tick_params(axis="x", rotation=25)
    fig.suptitle(
        f"{args.label}\nKlimenko Figure-1 binwise sensitivity; "
        f"map-covered {covered_count:,}/{selected_count:,} events"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = args.output_dir / "klimenko_map_sensitivity.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    if theta_rows:
        print(f"wrote {theta_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
