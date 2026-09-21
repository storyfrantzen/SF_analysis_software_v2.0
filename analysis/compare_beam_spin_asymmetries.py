#!/usr/bin/env python3

"""Compare independent beam-spin extractions and evaluate null harmonics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.beam_spin import sine_bin_averages


EDGE_NAMES = ("q2_edges", "xb_edges", "t_edges", "phi_edges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-null-fit-points", type=int, default=8)
    parser.add_argument(
        "--shared-events",
        action="store_true",
        help=(
            "inputs reuse the same events; label the quadrature-normalized "
            "difference as descriptive because its covariance is unavailable"
        ),
    )
    return parser.parse_args()


def finite_quantiles(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan")] * 3
    return np.quantile(values, [0.1, 0.5, 0.9]).tolist()


def fit_extended_harmonics(
    asymmetry: np.ndarray,
    uncertainty: np.ndarray,
    valid: np.ndarray,
    phi_edges_deg: np.ndarray,
    *,
    minimum_points: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit c0 + s1*sin(phi) + c1*cos(phi) + s2*sin(2phi) in each 3D bin."""

    shape = asymmetry.shape[:-1]
    coefficients = np.full(shape + (4,), np.nan)
    errors = np.full(shape + (4,), np.nan)
    chi2 = np.full(shape, np.nan)
    ndof = np.zeros(shape, dtype=np.int16)
    radians = np.deg2rad(np.asarray(phi_edges_deg, dtype=float))
    if radians.shape != (asymmetry.shape[-1] + 1,):
        raise ValueError("phi edges do not match asymmetry phi dimension")
    widths = np.diff(radians)
    design = np.column_stack(
        [
            np.ones(widths.size),
            sine_bin_averages(phi_edges_deg),
            (np.sin(radians[1:]) - np.sin(radians[:-1])) / widths,
            (np.cos(2.0 * radians[:-1]) - np.cos(2.0 * radians[1:]))
            / (2.0 * widths),
        ]
    )
    for index in np.ndindex(shape):
        keep = np.asarray(valid[index], dtype=bool)
        keep &= np.isfinite(asymmetry[index])
        keep &= np.isfinite(uncertainty[index]) & (uncertainty[index] > 0.0)
        if np.count_nonzero(keep) < minimum_points:
            continue
        x = design[keep]
        y = np.asarray(asymmetry[index][keep], dtype=float)
        weight = 1.0 / np.asarray(uncertainty[index][keep], dtype=float) ** 2
        normal = x.T @ (weight[:, None] * x)
        if not np.all(np.isfinite(normal)) or np.linalg.cond(normal) > 1.0e12:
            continue
        covariance = np.linalg.inv(normal)
        beta = covariance @ (x.T @ (weight * y))
        residual = y - x @ beta
        coefficients[index] = beta
        errors[index] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        chi2[index] = float(np.sum(weight * residual**2))
        ndof[index] = int(y.size - x.shape[1])
    return coefficients, errors, chi2, ndof


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    required = {
        *EDGE_NAMES,
        "beam_spin_asymmetry",
        "beam_spin_statistical_uncertainty",
        "beam_spin_valid",
        "sin_phi_amplitude",
        "sin_phi_amplitude_uncertainty",
        "sin_phi_amplitude_polarization_uncertainty",
        "sin_phi_fit_quality",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        return {name: np.asarray(source[name]) for name in source.files}


def null_summary(
    coefficients: np.ndarray, errors: np.ndarray, ndof: np.ndarray
) -> dict[str, object]:
    output: dict[str, object] = {
        "numerically_successful_bins": int(np.count_nonzero(ndof > 0))
    }
    for name, component in (("constant", 0), ("cos_phi", 2), ("sin_2phi", 3)):
        pull = np.divide(
            coefficients[..., component],
            errors[..., component],
            out=np.full(ndof.shape, np.nan),
            where=errors[..., component] > 0.0,
        )
        finite = np.isfinite(pull) & (ndof > 0)
        output[name] = {
            "bins": int(np.count_nonzero(finite)),
            "pull_q10_median_q90": finite_quantiles(pull[finite]),
            "absolute_pull_gt_2": int(np.count_nonzero(np.abs(pull[finite]) > 2.0)),
            "absolute_pull_gt_3": int(np.count_nonzero(np.abs(pull[finite]) > 3.0)),
        }
    return output


def compare_phi_points(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, object]:
    left_bsa = np.asarray(left["beam_spin_asymmetry"], dtype=float)
    right_bsa = np.asarray(right["beam_spin_asymmetry"], dtype=float)
    if not np.array_equal(left["phi_edges"], right["phi_edges"]):
        return {
            "available": False,
            "reason": "phi-bin edges differ; fitted 3D amplitudes remain comparable",
            "left_phi_bins": int(left_bsa.shape[-1]),
            "right_phi_bins": int(right_bsa.shape[-1]),
        }
    left_error = np.asarray(left["beam_spin_statistical_uncertainty"], dtype=float)
    right_error = np.asarray(right["beam_spin_statistical_uncertainty"], dtype=float)
    common = np.asarray(left["beam_spin_valid"], dtype=bool)
    common &= np.asarray(right["beam_spin_valid"], dtype=bool)
    sigma = np.sqrt(left_error**2 + right_error**2)
    pull = np.divide(
        right_bsa - left_bsa,
        sigma,
        out=np.full(left_bsa.shape, np.nan),
        where=common & (sigma > 0.0),
    )
    return {
        "available": True,
        "common_points": int(np.count_nonzero(common & np.isfinite(pull))),
        "statistical_pull_q10_median_q90": finite_quantiles(pull[common]),
        "statistical_absolute_pull_gt_2": int(
            np.count_nonzero(np.abs(pull[common]) > 2.0)
        ),
        "statistical_absolute_pull_gt_3": int(
            np.count_nonzero(np.abs(pull[common]) > 3.0)
        ),
    }


def main() -> int:
    args = parse_args()
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    left = load_artifact(left_path)
    right = load_artifact(right_path)
    for name in EDGE_NAMES[:3]:
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"beam-spin artifacts have incompatible {name}")

    left_amplitude = np.asarray(left["sin_phi_amplitude"], dtype=float)
    right_amplitude = np.asarray(right["sin_phi_amplitude"], dtype=float)
    if left_amplitude.shape != right_amplitude.shape:
        raise ValueError("beam-spin amplitude arrays have incompatible shapes")
    left_quality = np.asarray(left["sin_phi_fit_quality"], dtype=bool)
    right_quality = np.asarray(right["sin_phi_fit_quality"], dtype=bool)
    left_error = np.asarray(left["sin_phi_amplitude_uncertainty"], dtype=float)
    right_error = np.asarray(right["sin_phi_amplitude_uncertainty"], dtype=float)
    left_pol = np.asarray(
        left["sin_phi_amplitude_polarization_uncertainty"], dtype=float
    )
    right_pol = np.asarray(
        right["sin_phi_amplitude_polarization_uncertainty"], dtype=float
    )
    common = left_quality & right_quality
    common &= np.isfinite(left_amplitude) & np.isfinite(right_amplitude)
    statistical_sigma = np.sqrt(left_error**2 + right_error**2)
    conservative_sigma = np.sqrt(
        left_error**2 + right_error**2 + left_pol**2 + right_pol**2
    )
    difference = right_amplitude - left_amplitude
    fractional_difference = np.divide(
        difference,
        left_amplitude,
        out=np.full(difference.shape, np.nan),
        where=common & (left_amplitude != 0.0),
    )
    statistical_pull = np.divide(
        difference,
        statistical_sigma,
        out=np.full(difference.shape, np.nan),
        where=common & (statistical_sigma > 0.0),
    )
    conservative_pull = np.divide(
        difference,
        conservative_sigma,
        out=np.full(difference.shape, np.nan),
        where=common & (conservative_sigma > 0.0),
    )

    left_bsa = np.asarray(left["beam_spin_asymmetry"], dtype=float)
    right_bsa = np.asarray(right["beam_spin_asymmetry"], dtype=float)
    left_bsa_error = np.asarray(left["beam_spin_statistical_uncertainty"], dtype=float)
    right_bsa_error = np.asarray(right["beam_spin_statistical_uncertainty"], dtype=float)
    phi_point_summary = compare_phi_points(left, right)

    left_null = fit_extended_harmonics(
        left_bsa, left_bsa_error, left["beam_spin_valid"], left["phi_edges"],
        minimum_points=args.minimum_null_fit_points,
    )
    right_null = fit_extended_harmonics(
        right_bsa, right_bsa_error, right["beam_spin_valid"], right["phi_edges"],
        minimum_points=args.minimum_null_fit_points,
    )

    finite_common = common & np.isfinite(statistical_pull)
    n_common = int(np.count_nonzero(finite_common))
    correlation = float("nan")
    if n_common > 1:
        correlation = float(
            np.corrcoef(left_amplitude[finite_common], right_amplitude[finite_common])[0, 1]
        )
    summary = {
        "schema_version": 1,
        "comparison": "beam-spin extraction consistency",
        "event_relationship": "shared" if args.shared_events else "independent",
        "left": {"label": args.left_label, "artifact": str(left_path)},
        "right": {"label": args.right_label, "artifact": str(right_path)},
        "amplitude_comparison": {
            "left_production_bins": int(np.count_nonzero(left_quality)),
            "right_production_bins": int(np.count_nonzero(right_quality)),
            "common_production_bins": n_common,
            "right_minus_left_q10_median_q90": finite_quantiles(difference[finite_common]),
            "fractional_right_minus_left_q10_median_q90": finite_quantiles(
                fractional_difference[finite_common]
            ),
            "statistical_pull_q10_median_q90": finite_quantiles(
                statistical_pull[finite_common]
            ),
            "statistical_absolute_pull_gt_2": int(
                np.count_nonzero(np.abs(statistical_pull[finite_common]) > 2.0)
            ),
            "statistical_absolute_pull_gt_3": int(
                np.count_nonzero(np.abs(statistical_pull[finite_common]) > 3.0)
            ),
            "conservative_pull_q10_median_q90": finite_quantiles(
                conservative_pull[finite_common]
            ),
            "pearson_correlation": correlation,
            "statistical_chi2": float(np.nansum(statistical_pull[finite_common] ** 2)),
            "statistical_ndof": n_common,
        },
        "phi_point_comparison": phi_point_summary,
        "null_harmonics": {
            args.left_label: null_summary(left_null[0], left_null[1], left_null[3]),
            args.right_label: null_summary(right_null[0], right_null[1], right_null[3]),
        },
        "interpretation_limits": (
            (
                "the two artifacts reuse events, so their covariance is unavailable and "
                "difference-over-quadrature-error values are descriptive rather than pulls"
                if args.shared_events
                else "pulls use statistical errors; the conservative amplitude pull also "
                "treats the two polarization errors as independent"
            )
            + "; detector, selection, sideband-transfer, and other systematic covariance "
            "is pending"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "beam_spin_polarity_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "beam_spin_polarity_comparison.csv"
    q2_edges, xb_edges, t_edges = (left[name] for name in EDGE_NAMES[:3])
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(
            ["iq2", "ixb", "it", "Q2_center", "xB_center", "minus_t_center",
             "left_amplitude", "left_statistical_uncertainty",
             "right_amplitude", "right_statistical_uncertainty", "difference",
             "statistical_pull", "conservative_pull"]
        )
        for iq2, ixb, it in np.argwhere(finite_common):
            writer.writerow(
                [iq2, ixb, it, 0.5 * (q2_edges[iq2] + q2_edges[iq2 + 1]),
                 0.5 * (xb_edges[ixb] + xb_edges[ixb + 1]),
                 0.5 * (t_edges[it] + t_edges[it + 1]),
                 left_amplitude[iq2, ixb, it], left_error[iq2, ixb, it],
                 right_amplitude[iq2, ixb, it], right_error[iq2, ixb, it],
                 difference[iq2, ixb, it], statistical_pull[iq2, ixb, it],
                 conservative_pull[iq2, ixb, it]]
            )

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    diagnostic_path = args.output_dir / "beam_spin_polarity_diagnostics.pdf"
    with PdfPages(diagnostic_path) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
        if n_common:
            lo = float(np.nanmin([left_amplitude[finite_common], right_amplitude[finite_common]]))
            hi = float(np.nanmax([left_amplitude[finite_common], right_amplitude[finite_common]]))
            padding = 0.08 * max(hi - lo, 0.1)
            axes[0].errorbar(
                left_amplitude[finite_common], right_amplitude[finite_common],
                xerr=left_error[finite_common], yerr=right_error[finite_common],
                fmt="o", markersize=3, alpha=0.65, capsize=1,
            )
            axes[0].plot([lo - padding, hi + padding], [lo - padding, hi + padding], "k--")
            axes[0].set_xlim(lo - padding, hi + padding)
            axes[0].set_ylim(lo - padding, hi + padding)
        axes[0].set_xlabel(args.left_label + r" $A_{LU}^{\sin\phi}$")
        axes[0].set_ylabel(args.right_label + r" $A_{LU}^{\sin\phi}$")
        axes[0].grid(alpha=0.2)
        pulls = statistical_pull[finite_common]
        axes[1].hist(pulls, bins=np.linspace(-6.0, 6.0, 31), histtype="step", linewidth=1.5)
        axes[1].axvline(0.0, color="0.4", linewidth=0.8)
        axes[1].set_xlabel(
            "difference / quadrature statistical uncertainty"
            if args.shared_events
            else "extraction-difference pull (statistical)"
        )
        axes[1].set_ylabel("bins")
        axes[1].grid(alpha=0.2)
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
        for axis, (name, component) in zip(
            axes, (("constant", 0), (r"$\cos\phi$", 2), (r"$\sin2\phi$", 3))
        ):
            for values, errors, ndof, label in (
                (*left_null[:2], left_null[3], args.left_label),
                (*right_null[:2], right_null[3], args.right_label),
            ):
                pull = np.divide(
                    values[..., component], errors[..., component],
                    out=np.full(ndof.shape, np.nan), where=errors[..., component] > 0.0,
                )
                keep = np.isfinite(pull) & (ndof > 0)
                axis.hist(
                    pull[keep], bins=np.linspace(-6.0, 6.0, 31), histtype="step",
                    linewidth=1.4, label=label,
                )
            axis.axvline(0.0, color="0.4", linewidth=0.8)
            axis.set_title(name + " null pull")
            axis.set_xlabel("coefficient / statistical uncertainty")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("bins")
        axes[-1].legend(fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)

    overlay_path = args.output_dir / "sin_phi_amplitude_vs_t_polarity_comparison.pdf"
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    pages = 0
    with PdfPages(overlay_path) as pdf:
        for iq2, ixb in np.ndindex(left_amplitude.shape[:2]):
            left_keep = left_quality[iq2, ixb]
            right_keep = right_quality[iq2, ixb]
            if not np.any(left_keep | right_keep):
                continue
            figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.errorbar(
                t_centers[left_keep] - 0.006,
                left_amplitude[iq2, ixb, left_keep],
                yerr=left_error[iq2, ixb, left_keep], fmt="o", capsize=2,
                label=args.left_label,
            )
            axis.errorbar(
                t_centers[right_keep] + 0.006,
                right_amplitude[iq2, ixb, right_keep],
                yerr=right_error[iq2, ixb, right_keep], fmt="s", capsize=2,
                label=args.right_label,
            )
            axis.set_xlabel(r"$-t$ (GeV$^2$)")
            axis.set_ylabel(r"$A_{LU}^{\sin\phi}$")
            axis.set_title(
                rf"$Q^2\in[{q2_edges[iq2]:g},{q2_edges[iq2+1]:g}]$, "
                rf"$x_B\in[{xb_edges[ixb]:g},{xb_edges[ixb+1]:g}]$"
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1

    print(f"Left production bins: {int(np.count_nonzero(left_quality))}")
    print(f"Right production bins: {int(np.count_nonzero(right_quality))}")
    print(f"Common production bins: {n_common}")
    difference_label = (
        "Difference/quadrature-error" if args.shared_events else "Statistical pull"
    )
    print(
        f"{difference_label} q10/median/q90: "
        f"{finite_quantiles(statistical_pull[finite_common])}"
    )
    print(
        f"|{difference_label}| > 2: "
        f"{int(np.count_nonzero(np.abs(statistical_pull[finite_common]) > 2.0))}"
    )
    print(f"Phi comparison points: {phi_point_summary.get('common_points', 0)}")
    print(f"Amplitude overlay pages: {pages}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {diagnostic_path}")
    print(f"Wrote {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
