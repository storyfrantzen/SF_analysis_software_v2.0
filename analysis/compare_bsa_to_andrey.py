#!/usr/bin/env python3

"""Make a provisional nearest-bin comparison with Andrey's RGA BSA table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "andrey_eppi0_table_5_1_bsa.csv"
)
REQUIRED_FIELDS = {
    "q2_edges",
    "xb_edges",
    "t_edges",
    "sin_phi_amplitude",
    "sin_phi_amplitude_uncertainty",
    "sin_phi_amplitude_polarization_uncertainty",
    "sin_phi_fit_quality",
}
REGION_LABELS = {
    0: r"$\langle Q^2\rangle\approx2.58$, $\langle x_B\rangle\approx0.29$",
    1: r"$\langle Q^2\rangle\approx3.02$, $\langle x_B\rangle\approx0.39$",
    2: r"$\langle Q^2\rangle\approx4.65$, $\langle x_B\rangle\approx0.37$",
    3: r"$\langle Q^2\rangle\approx4.14$, $\langle x_B\rangle\approx0.51$",
    4: r"$\langle Q^2\rangle\approx6.64$, $\langle x_B\rangle\approx0.56$",
}
POLARITY_STYLE = {
    "out": {"color": "#1967a3", "marker": "o", "label": "torus+1 / out"},
    "in": {"color": "#c94b40", "marker": "s", "label": "torus-1 / in"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torus-plus-1", type=Path, required=True)
    parser.add_argument("--torus-minus-1", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plus-label", default="RGA Fall 2018 torus+1")
    parser.add_argument("--minus-label", default="RGA Fall 2018 torus-1")
    parser.add_argument("--q2-scale", type=float, default=0.5)
    parser.add_argument("--xb-scale", type=float, default=0.05)
    parser.add_argument("--t-scale", type=float, default=0.25)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(REQUIRED_FIELDS.difference(source.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        artifact = {name: np.asarray(source[name]) for name in REQUIRED_FIELDS}
    amplitude = np.asarray(artifact["sin_phi_amplitude"], dtype=float)
    expected = tuple(len(artifact[name]) - 1 for name in ("q2_edges", "xb_edges", "t_edges"))
    if amplitude.shape != expected:
        raise ValueError(f"{path} has amplitude shape {amplitude.shape}, expected {expected}")
    for name in (
        "sin_phi_amplitude_uncertainty",
        "sin_phi_amplitude_polarization_uncertainty",
        "sin_phi_fit_quality",
    ):
        if artifact[name].shape != amplitude.shape:
            raise ValueError(f"{path} has incompatible {name} shape")
    return artifact


def load_reference(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected = {
            "region", "polarity", "Q2_GeV2", "xB", "minus_t_GeV2",
            "A_sin_phi", "A_sin_phi_stat",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected reference columns in {path}")
        for raw in reader:
            polarity = raw["polarity"]
            if polarity not in POLARITY_STYLE:
                raise ValueError(f"unknown reference polarity {polarity!r}")
            rows.append(
                {
                    "region": int(raw["region"]),
                    "polarity": polarity,
                    "Q2": float(raw["Q2_GeV2"]),
                    "xB": float(raw["xB"]),
                    "minus_t": float(raw["minus_t_GeV2"]),
                    "reference_amplitude": float(raw["A_sin_phi"]),
                    "reference_statistical_uncertainty": float(raw["A_sin_phi_stat"]),
                }
            )
    if not rows:
        raise ValueError(f"reference table is empty: {path}")
    return rows


def finite_quantiles(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [float("nan")] * 3
    return np.quantile(finite, [0.1, 0.5, 0.9]).tolist()


def match_reference_rows(
    reference: list[dict[str, float | int | str]],
    artifacts: dict[str, dict[str, np.ndarray]],
    *,
    q2_scale: float,
    xb_scale: float,
    t_scale: float,
) -> list[dict[str, float | int | str]]:
    if min(q2_scale, xb_scale, t_scale) <= 0.0:
        raise ValueError("matching scales must be positive")
    matched: list[dict[str, float | int | str]] = []
    for row_index, row in enumerate(reference):
        polarity = str(row["polarity"])
        artifact = artifacts[polarity]
        q2 = 0.5 * np.add(artifact["q2_edges"][:-1], artifact["q2_edges"][1:])
        xb = 0.5 * np.add(artifact["xb_edges"][:-1], artifact["xb_edges"][1:])
        minus_t = 0.5 * np.add(artifact["t_edges"][:-1], artifact["t_edges"][1:])
        quality = np.asarray(artifact["sin_phi_fit_quality"], dtype=bool)
        amplitude = np.asarray(artifact["sin_phi_amplitude"], dtype=float)
        statistical = np.asarray(
            artifact["sin_phi_amplitude_uncertainty"], dtype=float
        )
        polarization = np.asarray(
            artifact["sin_phi_amplitude_polarization_uncertainty"], dtype=float
        )
        eligible = quality & np.isfinite(amplitude)
        eligible &= np.isfinite(statistical) & (statistical > 0.0)
        eligible &= np.isfinite(polarization) & (polarization >= 0.0)
        if not np.any(eligible):
            raise ValueError(f"no production-quality {polarity} BSA bins are available")
        coordinates = np.stack(np.meshgrid(q2, xb, minus_t, indexing="ij"), axis=-1)
        target = np.array([row["Q2"], row["xB"], row["minus_t"]], dtype=float)
        scales = np.array([q2_scale, xb_scale, t_scale], dtype=float)
        distance_squared = np.sum(((coordinates - target) / scales) ** 2, axis=-1)
        distance_squared[~eligible] = np.inf
        index = tuple(int(x) for x in np.unravel_index(np.argmin(distance_squared), quality.shape))
        campaign_stat = float(statistical[index])
        campaign_pol = float(polarization[index])
        difference = float(amplitude[index] - row["reference_amplitude"])
        statistical_sigma = float(
            np.hypot(campaign_stat, row["reference_statistical_uncertainty"])
        )
        available_sigma = float(np.hypot(statistical_sigma, campaign_pol))
        matched.append(
            {
                **row,
                "reference_row": row_index + 1,
                "iq2": index[0],
                "ixb": index[1],
                "it": index[2],
                "campaign_Q2_center": float(q2[index[0]]),
                "campaign_xB_center": float(xb[index[1]]),
                "campaign_minus_t_center": float(minus_t[index[2]]),
                "campaign_amplitude": float(amplitude[index]),
                "campaign_statistical_uncertainty": campaign_stat,
                "campaign_polarization_uncertainty": campaign_pol,
                "difference": difference,
                "statistical_pull": difference / statistical_sigma,
                "available_uncertainty_pull": difference / available_sigma,
                "normalized_center_distance": float(np.sqrt(distance_squared[index])),
                "delta_Q2": float(q2[index[0]] - row["Q2"]),
                "delta_xB": float(xb[index[1]] - row["xB"]),
                "delta_minus_t": float(minus_t[index[2]] - row["minus_t"]),
            }
        )
    return matched


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_pdf(
    path: Path,
    rows: list[dict[str, float | int | str]],
    labels: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), constrained_layout=True)
        for region, axis in zip(sorted(REGION_LABELS), axes.flat[:5]):
            selected = [row for row in rows if int(row["region"]) == region]
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            for row in selected:
                style = POLARITY_STYLE[str(row["polarity"])]
                ref_t = float(row["minus_t"])
                campaign_t = float(row["campaign_minus_t_center"])
                axis.plot(
                    [ref_t, campaign_t],
                    [row["reference_amplitude"], row["campaign_amplitude"]],
                    color=style["color"], alpha=0.25, linewidth=0.8,
                )
                axis.errorbar(
                    ref_t, row["reference_amplitude"],
                    yerr=row["reference_statistical_uncertainty"],
                    fmt=style["marker"], color="0.15", markerfacecolor="white",
                    markersize=5.2, capsize=2, linewidth=0.9,
                )
                total = float(
                    np.hypot(
                        row["campaign_statistical_uncertainty"],
                        row["campaign_polarization_uncertainty"],
                    )
                )
                axis.errorbar(
                    campaign_t, row["campaign_amplitude"], yerr=total,
                    fmt=style["marker"], color=style["color"], alpha=0.35,
                    markersize=5.5, capsize=2, linewidth=2.2,
                )
                axis.errorbar(
                    campaign_t, row["campaign_amplitude"],
                    yerr=row["campaign_statistical_uncertainty"],
                    fmt=style["marker"], color=style["color"],
                    markeredgecolor="white", markeredgewidth=0.45,
                    markersize=5.5, capsize=2, linewidth=1.0,
                )
            axis.set_title(f"Region {region}: {REGION_LABELS[region]}", fontsize=10)
            axis.set_xlabel(r"$-t$ (GeV$^2$)")
            axis.set_ylabel(r"$A_{LU}^{\sin\phi}$")
            axis.grid(alpha=0.2)
            axis.set_ylim(-0.08, 0.18)

        legend_axis = axes.flat[5]
        legend_axis.axis("off")
        legend_handles = [
            Line2D([0], [0], marker="o", color="0.15", markerfacecolor="white",
                   linestyle="none", label="Andrey Table 5.1 (stat.)"),
            Line2D([0], [0], marker="o", color=POLARITY_STYLE["out"]["color"],
                   linestyle="none", label=labels["out"]),
            Line2D([0], [0], marker="s", color=POLARITY_STYLE["in"]["color"],
                   linestyle="none", label=labels["in"]),
        ]
        legend_axis.legend(handles=legend_handles, loc="upper left", frameon=False)
        legend_axis.text(
            0.0, 0.58,
            "Campaign points use the nearest production-quality\n"
            "analysis-bin center in (Q2, xB, -t). Lines connect\n"
            "each reference point to its matched campaign point.\n\n"
            "Thin campaign bars: statistical uncertainty\n"
            "Faint outer bars: stat. + polarization uncertainty\n\n"
            "Provisional: detector, selection, sideband-transfer,\n"
            "and remaining systematic covariance are not included.",
            va="top", fontsize=9, linespacing=1.35,
        )
        figure.suptitle(
            "RGA Fall 2018 beam-spin asymmetry vs Andrey - provisional nearest-bin comparison",
            fontsize=14,
        )
        pdf.savefig(figure)
        plt.close(figure)


def write_diagnostics_pdf(
    path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pull = np.asarray([row["statistical_pull"] for row in rows], dtype=float)
    available_pull = np.asarray(
        [row["available_uncertainty_pull"] for row in rows], dtype=float
    )
    distance = np.asarray(
        [row["normalized_center_distance"] for row in rows], dtype=float
    )
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
        for polarity in POLARITY_STYLE:
            selected = np.asarray([row["polarity"] == polarity for row in rows])
            style = POLARITY_STYLE[polarity]
            axes[0, 0].scatter(
                np.asarray([row["minus_t"] for row in rows])[selected],
                pull[selected], label=style["label"], color=style["color"],
                marker=style["marker"], s=30,
            )
        axes[0, 0].axhspan(-2.0, 2.0, color="0.9", zorder=-2)
        axes[0, 0].axhline(0.0, color="0.4", linewidth=0.8)
        axes[0, 0].set_xlabel(r"Andrey $-t$ (GeV$^2$)")
        axes[0, 0].set_ylabel("statistical pull")
        axes[0, 0].legend(fontsize=8)

        bins = np.linspace(-6.0, 6.0, 25)
        axes[0, 1].hist(pull, bins=bins, histtype="step", linewidth=1.5,
                        label="statistical")
        axes[0, 1].hist(available_pull, bins=bins, histtype="step", linewidth=1.5,
                        label="stat. + campaign polarization")
        axes[0, 1].axvline(0.0, color="0.4", linewidth=0.8)
        axes[0, 1].set_xlabel("comparison pull")
        axes[0, 1].set_ylabel("reference rows")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].hist(distance, bins=np.linspace(0.0, max(2.0, distance.max() * 1.05), 18),
                        color="#6b7280", alpha=0.75)
        axes[1, 0].set_xlabel("normalized nearest-center distance")
        axes[1, 0].set_ylabel("reference rows")

        offsets = (
            ("delta_Q2", r"$\Delta Q^2$ (GeV$^2$)"),
            ("delta_xB", r"$\Delta x_B$"),
            ("delta_minus_t", r"$\Delta(-t)$ (GeV$^2$)"),
        )
        for name, label in offsets:
            axes[1, 1].hist(
                [row[name] for row in rows], bins=14, histtype="step",
                linewidth=1.4, label=label,
            )
        axes[1, 1].axvline(0.0, color="0.4", linewidth=0.8)
        axes[1, 1].set_xlabel("campaign center minus reference coordinate")
        axes[1, 1].set_ylabel("reference rows")
        axes[1, 1].legend(fontsize=8)
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        figure.suptitle(
            "Andrey BSA comparison diagnostics - statistical pulls and bin matching",
            fontsize=14,
        )
        pdf.savefig(figure)
        plt.close(figure)


def main() -> int:
    args = parse_args()
    plus_path = args.torus_plus_1.resolve()
    minus_path = args.torus_minus_1.resolve()
    reference_path = args.reference.resolve()
    artifacts = {
        "out": load_artifact(plus_path),
        "in": load_artifact(minus_path),
    }
    reference = load_reference(reference_path)
    rows = match_reference_rows(
        reference, artifacts,
        q2_scale=args.q2_scale, xb_scale=args.xb_scale, t_scale=args.t_scale,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "nearest_bin_comparison.csv"
    comparison_pdf = args.output_dir / "rga_bsa_vs_andrey_reference.pdf"
    diagnostics_pdf = args.output_dir / "rga_bsa_vs_andrey_reference_diagnostics.pdf"
    summary_path = args.output_dir / "comparison_summary.json"
    write_csv(comparison_csv, rows)
    write_comparison_pdf(
        comparison_pdf, rows, {"out": args.plus_label, "in": args.minus_label}
    )
    write_diagnostics_pdf(diagnostics_pdf, rows)

    pull = np.asarray([row["statistical_pull"] for row in rows], dtype=float)
    available_pull = np.asarray(
        [row["available_uncertainty_pull"] for row in rows], dtype=float
    )
    unique_matches = {
        (str(row["polarity"]), int(row["iq2"]), int(row["ixb"]), int(row["it"]))
        for row in rows
    }
    summary = {
        "schema_version": 1,
        "comparison": "RGA Fall 2018 BSA vs Andrey Table 5.1",
        "status": "provisional nearest-analysis-bin diagnostic",
        "inputs": {
            "torus_plus_1": {
                "label": args.plus_label,
                "artifact": str(plus_path),
                "sha256": sha256(plus_path),
                "reference_polarity": "out",
            },
            "torus_minus_1": {
                "label": args.minus_label,
                "artifact": str(minus_path),
                "sha256": sha256(minus_path),
                "reference_polarity": "in",
            },
            "reference": {
                "artifact": str(reference_path),
                "sha256": sha256(reference_path),
                "source": "Andrey EPPI0 analysis note, Table 5.1",
                "uncertainty": "statistical only",
            },
        },
        "matching": {
            "definition": (
                "nearest production-quality campaign bin center in Q2, xB, and -t; "
                "no event-level rebinning or bin-center translation"
            ),
            "normalized_distance": (
                "sqrt((delta Q2/q2_scale)^2 + (delta xB/xb_scale)^2 + "
                "(delta(-t)/t_scale)^2)"
            ),
            "scales": {
                "Q2_GeV2": args.q2_scale,
                "xB": args.xb_scale,
                "minus_t_GeV2": args.t_scale,
            },
            "reference_rows": len(rows),
            "distinct_campaign_bins": len(unique_matches),
            "normalized_distance_q10_median_q90": finite_quantiles(
                [row["normalized_center_distance"] for row in rows]
            ),
            "coordinate_offsets_q10_median_q90": {
                "Q2_GeV2": finite_quantiles([row["delta_Q2"] for row in rows]),
                "xB": finite_quantiles([row["delta_xB"] for row in rows]),
                "minus_t_GeV2": finite_quantiles(
                    [row["delta_minus_t"] for row in rows]
                ),
            },
        },
        "statistical_comparison": {
            "difference_definition": "campaign minus Andrey",
            "pull_definition": (
                "difference / sqrt(campaign statistical variance + Andrey "
                "statistical variance)"
            ),
            "pull_q10_median_q90": finite_quantiles(pull),
            "absolute_pull_gt_2": int(np.count_nonzero(np.abs(pull) > 2.0)),
            "absolute_pull_gt_3": int(np.count_nonzero(np.abs(pull) > 3.0)),
            "chi2": float(np.sum(pull**2)),
            "nominal_ndof": len(rows),
            "chi2_per_nominal_ndof": float(np.mean(pull**2)),
        },
        "available_uncertainty_comparison": {
            "definition": (
                "statistical denominator with campaign beam-polarization variance added; "
                "other systematics remain absent"
            ),
            "pull_q10_median_q90": finite_quantiles(available_pull),
            "absolute_pull_gt_2": int(
                np.count_nonzero(np.abs(available_pull) > 2.0)
            ),
        },
        "IBU_relationship": (
            "none: these beam-spin artifacts are helicity-yield asymmetries and do not "
            "consume unfolded cross sections or their covariance"
        ),
        "limitations": [
            "nearest-center matching is not an exact rebinning",
            "campaign detector, selection, sideband-transfer, and other systematic covariance is pending",
            "Andrey uncertainties in the supplied table are statistical only",
            "reused campaign bins make the nominal chi2 descriptive rather than a formal global goodness-of-fit test",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Reference rows: {len(rows)}")
    print(f"Distinct matched campaign bins: {len(unique_matches)}")
    print(f"Statistical pull q10/median/q90: {finite_quantiles(pull)}")
    print(f"|statistical pull| > 2: {int(np.count_nonzero(np.abs(pull) > 2.0))}")
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {comparison_pdf}")
    print(f"Wrote {diagnostics_pdf}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
