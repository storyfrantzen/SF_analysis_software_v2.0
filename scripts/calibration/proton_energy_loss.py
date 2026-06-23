from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .fit_utils import (
    adaptive_edges,
    binned_profile,
    fit_linear_form,
    fixed_edges,
    json_ready,
    MomentumForm,
    weighted_polyfit_ascending,
)
from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, define_common_proton_residuals, load_dataframe


@dataclass(frozen=True)
class DetectorFitConfig:
    detector: int
    theta_caps: tuple[float, float]
    theta_range: tuple[float, float]
    momentum_range: tuple[float, float]
    theta_bins: int
    momentum_bins: int
    residual_forms: dict[str, MomentumForm]
    theta_poly_orders: dict[str, int]
    residual_ranges: dict[str, tuple[float, float]]


DEFAULT_CONFIGS = {
    "FD": DetectorFitConfig(
        detector=1,
        theta_caps=(15.0, 40.0),
        theta_range=(15.0, 40.0),
        momentum_range=(0.55, 5.0),
        theta_bins=24,
        momentum_bins=28,
        residual_forms={
            "delta_p": MomentumForm.INV_P2,
            "delta_theta": MomentumForm.INV_P,
            "delta_phi": MomentumForm.INV_P2,
        },
        theta_poly_orders={"delta_p": 1, "delta_theta": 3, "delta_phi": 4},
        residual_ranges={
            "delta_p": (-0.06, 0.06),
            "delta_theta": (-1.0, 1.0),
            "delta_phi": (-1.0, 1.0),
        },
    ),
    "CD": DetectorFitConfig(
        detector=2,
        theta_caps=(40.0, 125.0),
        theta_range=(40.0, 58.0),
        momentum_range=(0.3, 2.4),
        theta_bins=10,
        momentum_bins=22,
        residual_forms={
            "delta_p": MomentumForm.POLY_P2,
            "delta_theta": MomentumForm.INV_P,
            "delta_phi": MomentumForm.INV_P2,
        },
        theta_poly_orders={"delta_p": 2, "delta_theta": 2, "delta_phi": 2},
        residual_ranges={
            "delta_p": (-0.2, 0.2),
            "delta_theta": (-0.2, 0.2),
            "delta_phi": (-0.2, 0.2),
        },
    ),
}


RESIDUAL_COLUMNS = {
    "delta_p": "delta_p_fit",
    "delta_theta": "delta_theta_fit",
    "delta_phi": "delta_phi_fit",
}


ResidualRangeMode = str
ThetaRangeMode = str


def evaluate_correction(term: dict[str, object],
                        p: np.ndarray,
                        theta: np.ndarray) -> np.ndarray:
    """Evaluate one exported correction term using the C++ correction convention."""
    if "momentumPowers" in term:
        momentum_powers = tuple(int(power) for power in term["momentumPowers"])
    else:
        # Accept correction files generated before the numeric-basis schema.
        momentum_powers = MomentumForm(str(term["form"])).momentum_powers
    coeffs = term["coeffs"]
    theta_parameters = np.column_stack([
        np.polynomial.polynomial.polyval(theta, coeffs[str(i)])
        for i in range(len(momentum_powers))
    ])
    momentum_terms = np.column_stack([np.power(p, power) for power in momentum_powers])
    return np.sum(momentum_terms * theta_parameters, axis=1)


def filtered_arrays(input_file: Path,
                    tree: str,
                    cfg: DetectorFitConfig,
                    max_rows: int | None) -> dict[str, np.ndarray]:
    """Load the common matched-proton sample without residual-specific cuts."""
    df = define_common_proton_residuals(load_dataframe(input_file, tree))
    df = df.Filter(f"rec.det == {cfg.detector}")
    df = df.Filter(f"theta_deg >= {cfg.theta_caps[0]} && theta_deg <= {cfg.theta_caps[1]}")
    df = df.Filter(f"rec.p >= {cfg.momentum_range[0]} && rec.p <= {cfg.momentum_range[1]}")

    return arrays_from_dataframe(
        df,
        ["rec.p", "theta_deg", "delta_p_fit", "delta_theta_fit", "delta_phi_fit"],
        max_rows=max_rows,
    )


def with_theta_caps(cfg: DetectorFitConfig,
                    theta_caps: tuple[float, float] | None) -> DetectorFitConfig:
    if theta_caps is None:
        return cfg
    lo, hi = theta_caps
    if hi <= lo:
        raise ValueError(f"Invalid theta caps: {theta_caps}")
    return replace(cfg, theta_caps=(float(lo), float(hi)))


def fixed_theta_range(cfg: DetectorFitConfig) -> tuple[float, float]:
    return cfg.theta_range


def quantile_theta_range(arrays: dict[str, np.ndarray],
                         trim_quantile: float) -> tuple[float, float]:
    if not (0.0 <= trim_quantile < 0.5):
        raise ValueError("trim_quantile must be in the interval [0, 0.5)")

    theta = np.asarray(arrays["theta_deg"], dtype=float)
    finite = theta[np.isfinite(theta)]
    if finite.size == 0:
        raise ValueError("No finite theta values available")

    lo, hi = np.quantile(finite, [trim_quantile, 1.0 - trim_quantile])
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        raise ValueError("Cannot derive a non-empty theta range from the sample")
    return float(lo), float(hi)


def resolve_theta_range(arrays: dict[str, np.ndarray],
                        cfg: DetectorFitConfig,
                        range_mode: ThetaRangeMode,
                        trim_quantile: float) -> tuple[float, float]:
    if range_mode == "fixed":
        return fixed_theta_range(cfg)
    if range_mode == "quantile":
        return quantile_theta_range(arrays, trim_quantile)
    raise ValueError(f"Unsupported theta range mode: {range_mode}")


def fixed_residual_range(residual_name: str,
                         cfg: DetectorFitConfig) -> tuple[float, float]:
    return cfg.residual_ranges[residual_name]


def quantile_residual_range(arrays: dict[str, np.ndarray],
                            residual_name: str,
                            trim_quantile: float) -> tuple[float, float]:
    if not (0.0 <= trim_quantile < 0.5):
        raise ValueError("trim_quantile must be in the interval [0, 0.5)")

    residual = np.asarray(arrays[RESIDUAL_COLUMNS[residual_name]], dtype=float)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for {residual_name}")

    lo, hi = np.quantile(finite, [trim_quantile, 1.0 - trim_quantile])
    return float(lo), float(hi)


def resolve_residual_range(arrays: dict[str, np.ndarray],
                           residual_name: str,
                           cfg: DetectorFitConfig,
                           range_mode: ResidualRangeMode,
                           trim_quantile: float) -> tuple[float, float]:
    if range_mode == "fixed":
        return fixed_residual_range(residual_name, cfg)
    if range_mode == "quantile":
        return quantile_residual_range(arrays, residual_name, trim_quantile)
    raise ValueError(f"Unsupported residual range mode: {range_mode}")


def residual_range_mask(arrays: dict[str, np.ndarray],
                        residual_name: str,
                        residual_range: tuple[float, float]) -> np.ndarray:
    """Select outliers only for the residual currently being fitted."""
    residual = np.asarray(arrays[RESIDUAL_COLUMNS[residual_name]], dtype=float)
    lo, hi = residual_range
    if hi < lo:
        raise ValueError(f"Invalid residual range for {residual_name}: {residual_range}")
    if hi == lo:
        return np.isfinite(residual) & (residual == lo)
    return np.isfinite(residual) & (residual >= lo) & (residual <= hi)


def fit_detector(arrays: dict[str, np.ndarray],
                 detector_name: str,
                 cfg: DetectorFitConfig,
                 adaptive_theta: bool,
                 min_entries: int,
                 residual_range_mode: ResidualRangeMode,
                 residual_trim_quantile: float,
                 theta_range_mode: ThetaRangeMode,
                 theta_trim_quantile: float) -> dict[str, dict[str, object]]:
    p = arrays["rec.p"]
    theta = arrays["theta_deg"]
    theta_range = resolve_theta_range(arrays, cfg, theta_range_mode, theta_trim_quantile)
    theta_mask = (
        np.isfinite(theta) & (theta >= theta_range[0]) & (theta <= theta_range[1])
    )
    print(
        f"{detector_name} theta fit range: {theta_range_mode} range "
        f"[{theta_range[0]:.6g}, {theta_range[1]:.6g}] "
        f"within caps [{cfg.theta_caps[0]:.6g}, {cfg.theta_caps[1]:.6g}], "
        f"retained {theta_mask.sum()}/{theta_mask.size} rows"
    )
    theta_edges = (
        adaptive_edges(theta, cfg.theta_bins, theta_range)
        if adaptive_theta
        else fixed_edges(cfg.theta_bins, theta_range)
    )
    p_edges = fixed_edges(cfg.momentum_bins, cfg.momentum_range)

    output: dict[str, dict[str, object]] = {}
    for residual_name, residual_column in RESIDUAL_COLUMNS.items():
        residual_range = resolve_residual_range(
            arrays, residual_name, cfg, residual_range_mode, residual_trim_quantile
        )
        range_mask = residual_range_mask(arrays, residual_name, residual_range)
        fit_p = p[range_mask]
        fit_theta = theta[range_mask]
        residual = arrays[residual_column][range_mask]
        print(
            f"{detector_name} {residual_name}: {residual_range_mode} range "
            f"[{residual_range[0]:.6g}, {residual_range[1]:.6g}], retained "
            f"{range_mask.sum()}/{range_mask.size} rows in its residual range"
        )
        form = cfg.residual_forms[residual_name]
        n_params = form.n_parameters
        param_values = [[] for _ in range(n_params)]
        param_errors = [[] for _ in range(n_params)]
        theta_centers = []

        for theta_lo, theta_hi in zip(theta_edges[:-1], theta_edges[1:]):
            theta_mask = (fit_theta >= theta_lo) & (fit_theta < theta_hi)
            profile = binned_profile(
                fit_p[theta_mask], residual[theta_mask], p_edges, min_entries
            )
            beta, beta_err = fit_linear_form(profile.centers, profile.means, form, profile.errors)
            if not np.all(np.isfinite(beta)):
                continue

            theta_centers.append(0.5 * (theta_lo + theta_hi))
            for i, value in enumerate(beta):
                param_values[i].append(value)
                param_errors[i].append(beta_err[i] if i < len(beta_err) else np.nan)

        theta_array = np.asarray(theta_centers, dtype=float)
        coeffs = {}
        order = cfg.theta_poly_orders[residual_name]
        for i, values in enumerate(param_values):
            y = np.asarray(values, dtype=float)
            yerr = np.asarray(param_errors[i], dtype=float)
            coeffs[str(i)] = json_ready(weighted_polyfit_ascending(theta_array, y, order, yerr))

        key = f"p_{residual_name}_{detector_name}"
        output[key] = {"momentumPowers": list(form.momentum_powers), "coeffs": coeffs}

    return output


def maybe_plot(arrays: dict[str, np.ndarray],
               detector_name: str,
               cfg: DetectorFitConfig,
               corrections: dict[str, dict[str, object]],
               adaptive_theta: bool,
               output_dir: Path,
               dataset_tag: str = "",
               beam_energy: float | None = None,
               residual_range_mode: ResidualRangeMode = "quantile",
               residual_trim_quantile: float = 0.01,
               theta_range_mode: ThetaRangeMode = "quantile",
               theta_trim_quantile: float = 0.001) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    p = arrays["rec.p"]
    theta = arrays["theta_deg"]
    theta_range = resolve_theta_range(arrays, cfg, theta_range_mode, theta_trim_quantile)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, col) in zip(axes, RESIDUAL_COLUMNS.items()):
        hist = ax.hist2d(p, arrays[col], bins=80, cmap="magma", cmin=1)
        fig.colorbar(hist[3], ax=ax)
        ax.set_xlabel("p_rec [GeV]")
        ax.set_ylabel(name)
        ax.set_title(f"{detector_name} {name}")
    save_plot(
        fig,
        output_dir / f"{detector_name}_proton_residuals_vs_p.png",
        f"{detector_name} proton residuals vs reconstructed momentum",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)

    theta_edges = (
        adaptive_edges(theta, cfg.theta_bins, theta_range)
        if adaptive_theta
        else fixed_edges(cfg.theta_bins, theta_range)
    )
    for residual_name, residual_column in RESIDUAL_COLUMNS.items():
        term = corrections[f"p_{residual_name}_{detector_name}"]
        applied_correction = evaluate_correction(term, p, theta)
        before = arrays[residual_column]
        after = before - applied_correction

        n_bins = len(theta_edges) - 1
        n_cols = min(5, n_bins)
        n_rows = int(np.ceil(n_bins / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols, 2.8 * n_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        residual_lo, residual_hi = resolve_residual_range(
            arrays, residual_name, cfg, residual_range_mode, residual_trim_quantile
        )
        if residual_hi == residual_lo:
            residual_lo -= 0.5
            residual_hi += 0.5
        histogram_edges = np.linspace(residual_lo, residual_hi, 61)
        for i, (theta_lo, theta_hi) in enumerate(zip(theta_edges[:-1], theta_edges[1:])):
            ax = axes.flat[i]
            upper = theta <= theta_hi if i == n_bins - 1 else theta < theta_hi
            mask = (theta >= theta_lo) & upper
            ax.hist(before[mask], bins=histogram_edges, histtype="step", label="before", color="0.35")
            ax.hist(after[mask], bins=histogram_edges, histtype="step", label="after", color="tab:blue")
            ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.6)
            ax.set_title(f"{theta_lo:.1f}–{theta_hi:.1f}°", fontsize=9)
            ax.tick_params(labelsize=8)
        for ax in axes.flat[n_bins:]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(residual_name)
        for ax in axes[:, 0]:
            ax.set_ylabel("entries")
        axes.flat[0].legend(fontsize=8)
        save_plot(
            fig,
            output_dir / f"{detector_name}_{residual_name}_before_after_by_theta.png",
            f"{detector_name} {residual_name}: before/after correction by theta bin",
            dataset_tag,
            beam_energy,
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, col) in zip(axes, RESIDUAL_COLUMNS.items()):
        hist = ax.hist2d(theta, arrays[col], bins=80, cmap="viridis", cmin=1)
        fig.colorbar(hist[3], ax=ax)
        ax.set_xlabel("theta_rec [deg]")
        ax.set_ylabel(name)
        ax.set_title(f"{detector_name} {name}")
    save_plot(
        fig,
        output_dir / f"{detector_name}_proton_residuals_vs_theta.png",
        f"{detector_name} proton residuals vs reconstructed theta",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive proton energy-loss correction coefficients from matched REC/GEMC ROOT rows."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--detector", choices=["FD", "CD", "both"], default="both")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parameters/proton_energy_loss/protonEnergyLoss_params.json"),
    )
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--fixed-theta-bins", action="store_true")
    parser.add_argument("--min-bin-entries", type=int, default=20)
    parser.add_argument(
        "--theta-range-mode",
        choices=["quantile", "fixed"],
        default="quantile",
        help=(
            "How to choose the detector theta fit domain. The default derives "
            "a central quantile range from the sample after detector caps and "
            "momentum cuts; fixed uses the historical detector theta_range."
        ),
    )
    parser.add_argument(
        "--theta-trim-quantile",
        type=float,
        default=0.001,
        help="For quantile theta mode, trim this fraction from each theta tail.",
    )
    parser.add_argument(
        "--fd-theta-caps",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Physical FD theta caps applied before deriving the fit range.",
    )
    parser.add_argument(
        "--cd-theta-caps",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Physical CD theta caps applied before deriving the fit range.",
    )
    parser.add_argument(
        "--residual-range-mode",
        choices=["quantile", "fixed"],
        default="quantile",
        help=(
            "How to choose residual inlier ranges for each fit. The default "
            "keeps the central quantile interval from the sample; fixed uses "
            "the historical hard-coded ranges."
        ),
    )
    parser.add_argument(
        "--residual-trim-quantile",
        type=float,
        default=0.01,
        help="For quantile mode, trim this fraction from each residual tail.",
    )
    parser.add_argument("--dataset-tag", default="", help="Short label shown on generated plots.")
    parser.add_argument("--beam-energy", type=float, help="Beam energy in GeV shown on generated plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detectors = ["FD", "CD"] if args.detector == "both" else [args.detector]

    combined: dict[str, dict[str, object]] = {}
    for detector_name in detectors:
        cfg = DEFAULT_CONFIGS[detector_name]
        theta_caps = args.fd_theta_caps if detector_name == "FD" else args.cd_theta_caps
        cfg = with_theta_caps(cfg, tuple(theta_caps) if theta_caps is not None else None)
        arrays = filtered_arrays(args.input_file, args.tree, cfg, args.max_rows)
        detector_corrections = fit_detector(
            arrays,
            detector_name,
            cfg,
            adaptive_theta=not args.fixed_theta_bins,
            min_entries=args.min_bin_entries,
            residual_range_mode=args.residual_range_mode,
            residual_trim_quantile=args.residual_trim_quantile,
            theta_range_mode=args.theta_range_mode,
            theta_trim_quantile=args.theta_trim_quantile,
        )
        combined.update(detector_corrections)
        if args.plot_dir:
            maybe_plot(
                arrays,
                detector_name,
                cfg,
                detector_corrections,
                not args.fixed_theta_bins,
                args.plot_dir,
                args.dataset_tag,
                args.beam_energy,
                args.residual_range_mode,
                args.residual_trim_quantile,
                args.theta_range_mode,
                args.theta_trim_quantile,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(combined, f, indent=2)
        f.write("\n")
    print(f"Wrote proton energy-loss corrections to {args.output}")


if __name__ == "__main__":
    main()
