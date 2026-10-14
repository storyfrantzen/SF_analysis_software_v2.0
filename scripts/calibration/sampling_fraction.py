from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .fit_utils import fixed_edges, json_ready, weighted_polyfit_descending
from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, define_common_electron_sf, load_dataframe


@dataclass(frozen=True)
class SamplingFractionConfig:
    momentum_range: tuple[float, float] = (1.0, 10.0)
    momentum_bins: int = 22
    poly_degree: int = 2
    min_entries: int = 30
    min_pcal_energy: float = 0.07
    diagonal_y_scale: float = 1.0
    diagonal_x_scale: float = 1.0
    diagonal_threshold: float = 0.2
    diagonal_momentum_threshold: float = 4.5
    core_clip_sigma: float = 2.0
    core_max_iterations: int = 12
    core_convergence: float = 1.0e-4
    core_histogram_bins: int = 100
    min_core_fraction: float = 0.5


@dataclass(frozen=True)
class GaussianCoreBinFit:
    momentum_mean: float
    momentum_error: float
    mu: float
    mu_error: float
    sigma: float
    sigma_error: float
    entries: int
    core_entries: int
    core_fraction: float
    reduced_chi2: float
    iterations: int
    status: str

    @property
    def valid(self) -> bool:
        return (
            self.status in {"ok", "max_iterations"}
            and np.isfinite(self.momentum_mean)
            and np.isfinite(self.mu)
            and np.isfinite(self.mu_error)
            and self.mu_error > 0
            and np.isfinite(self.sigma)
            and self.sigma > 0
            and np.isfinite(self.sigma_error)
            and self.sigma_error > 0
        )


@dataclass(frozen=True)
class SamplingFractionProfile:
    momentum: np.ndarray
    momentum_errors: np.ndarray
    mu_values: np.ndarray
    mu_errors: np.ndarray
    sigma_values: np.ndarray
    sigma_errors: np.ndarray
    counts: np.ndarray
    core_counts: np.ndarray
    core_fractions: np.ndarray
    reduced_chi2: np.ndarray
    iterations: np.ndarray
    statuses: tuple[str, ...]

    @property
    def valid(self) -> np.ndarray:
        return (
            np.isfinite(self.momentum)
            & np.isfinite(self.mu_values)
            & np.isfinite(self.mu_errors)
            & (self.mu_errors > 0)
            & np.isfinite(self.sigma_values)
            & (self.sigma_values > 0)
            & np.isfinite(self.sigma_errors)
            & (self.sigma_errors > 0)
        )


@dataclass(frozen=True)
class SamplingFractionCalibration:
    coefficients: dict[str, dict[str, list[float]]]
    parameter_profiles: dict[str, SamplingFractionProfile]
    sector_profiles: dict[str, SamplingFractionProfile]


def electron_arrays(input_file: Path,
                    tree: str,
                    max_rows: int | None) -> dict[str, np.ndarray]:
    df = define_common_electron_sf(load_dataframe(input_file, tree))
    return arrays_from_dataframe(
        df,
        ["sf_p", "sf_sector", "sf_epcal", "sf_ecin", "sampling_fraction"],
        max_rows=max_rows,
    )


def sf_diagonal_mask(arrays: dict[str, np.ndarray],
                     cfg: SamplingFractionConfig) -> np.ndarray:
    p = arrays["sf_p"]
    e_pcal = arrays["sf_epcal"]
    e_ecin = arrays["sf_ecin"]
    finite = np.isfinite(p) & np.isfinite(e_pcal) & np.isfinite(e_ecin) & (p > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        diagonal_value = (
            cfg.diagonal_y_scale * e_pcal / p
            + cfg.diagonal_x_scale * e_ecin / p
        )
    return finite & (
        (p < cfg.diagonal_momentum_threshold)
        | (diagonal_value > cfg.diagonal_threshold)
    )


def sf_preselection_mask(arrays: dict[str, np.ndarray],
                         cfg: SamplingFractionConfig) -> np.ndarray:
    e_pcal = arrays["sf_epcal"]
    return sf_diagonal_mask(arrays, cfg) & (e_pcal > cfg.min_pcal_energy)


def select_arrays(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[mask] for name, values in arrays.items()}


def effective_momentum_range(p: np.ndarray,
                             requested_range: tuple[float, float]) -> tuple[float, float]:
    requested_min, requested_max = requested_range
    if requested_min >= requested_max:
        raise ValueError("p-min must be less than p-max")

    finite_p = p[np.isfinite(p)]
    if finite_p.size == 0:
        raise ValueError("Cannot determine momentum range from an empty sample")

    sample_min = float(np.min(finite_p))
    sample_max = float(np.max(finite_p))
    effective_min = max(requested_min, sample_min)
    effective_max = min(requested_max, sample_max)
    if effective_min >= effective_max:
        raise ValueError(
            f"Requested momentum range [{requested_min}, {requested_max}] does not overlap "
            f"sample range [{sample_min}, {sample_max}]"
        )
    return effective_min, effective_max


def _invalid_core_fit(entries: int, status: str) -> GaussianCoreBinFit:
    return GaussianCoreBinFit(
        momentum_mean=np.nan,
        momentum_error=np.nan,
        mu=np.nan,
        mu_error=np.nan,
        sigma=np.nan,
        sigma_error=np.nan,
        entries=entries,
        core_entries=0,
        core_fraction=0.0,
        reduced_chi2=np.nan,
        iterations=0,
        status=status,
    )


def _truncated_normal_sigma_correction(clip_sigma: float) -> float:
    normalization = math.erf(clip_sigma / math.sqrt(2.0))
    phi = math.exp(-0.5 * clip_sigma * clip_sigma) / math.sqrt(2.0 * math.pi)
    variance = 1.0 - 2.0 * clip_sigma * phi / normalization
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("core clip sigma gives an invalid truncated-normal variance")
    return 1.0 / math.sqrt(variance)


def _mode_seed(values: np.ndarray, histogram_bins: int) -> tuple[float, float]:
    q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    quantile_sigma = max(0.5 * (q84 - q16), np.finfo(float).eps)
    lo = median - 5.0 * quantile_sigma
    hi = median + 5.0 * quantile_sigma
    if hi <= lo:
        return float(median), float(quantile_sigma)

    effective_bins = min(histogram_bins, max(10, 2 * int(np.sqrt(values.size))))
    counts, edges = np.histogram(values, bins=effective_bins, range=(lo, hi))
    peak = int(np.argmax(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centers[peak])

    half_maximum = 0.5 * counts[peak]
    left = peak
    while left > 0 and counts[left] >= half_maximum:
        left -= 1
    right = peak
    while right < len(counts) - 1 and counts[right] >= half_maximum:
        right += 1
    fwhm_sigma = (centers[right] - centers[left]) / 2.355
    if not np.isfinite(fwhm_sigma) or fwhm_sigma <= 0:
        fwhm_sigma = quantile_sigma

    sigma = max(min(quantile_sigma, float(fwhm_sigma)), np.finfo(float).eps)
    near_mode = values[np.abs(values - mode) <= 1.5 * sigma]
    if near_mode.size:
        mode = float(np.median(near_mode))
    return mode, sigma


def _gaussian_reduced_chi2(values: np.ndarray, mu: float, sigma: float,
                           clip_sigma: float) -> float:
    selected = values[np.abs(values - mu) <= clip_sigma * sigma]
    if selected.size < 20:
        return np.nan
    bins = min(50, max(15, int(np.sqrt(selected.size))))
    counts, edges = np.histogram(
        selected,
        bins=bins,
        range=(mu - clip_sigma * sigma, mu + clip_sigma * sigma),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_shape = np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
    if not np.any(expected_shape > 0):
        return np.nan
    expected = selected.size * expected_shape / np.sum(expected_shape)
    usable = expected >= 5.0
    ndof = int(np.count_nonzero(usable)) - 3
    if ndof <= 0:
        return np.nan
    chi2 = np.sum((counts[usable] - expected[usable]) ** 2 / expected[usable])
    return float(chi2 / ndof)


def fit_gaussian_core(momentum: np.ndarray,
                      sampling_fraction: np.ndarray,
                      cfg: SamplingFractionConfig) -> GaussianCoreBinFit:
    finite = np.isfinite(momentum) & np.isfinite(sampling_fraction)
    p = np.asarray(momentum[finite], dtype=float)
    sf = np.asarray(sampling_fraction[finite], dtype=float)
    entries = int(sf.size)
    if entries < cfg.min_entries:
        return _invalid_core_fit(entries, "insufficient_entries")

    mu, sigma = _mode_seed(sf, cfg.core_histogram_bins)
    sigma_correction = _truncated_normal_sigma_correction(cfg.core_clip_sigma)
    converged = False
    iterations = 0
    core = np.ones(entries, dtype=bool)
    for iterations in range(1, cfg.core_max_iterations + 1):
        core = np.abs(sf - mu) <= cfg.core_clip_sigma * sigma
        core_entries = int(np.count_nonzero(core))
        if core_entries < cfg.min_entries:
            return _invalid_core_fit(entries, "insufficient_core_entries")

        next_mu = float(np.mean(sf[core]))
        clipped_sigma = float(np.std(sf[core], ddof=1))
        next_sigma = clipped_sigma * sigma_correction
        if not np.isfinite(next_sigma) or next_sigma <= 0:
            return _invalid_core_fit(entries, "invalid_core_width")

        relative_mu = abs(next_mu - mu) / max(next_sigma, np.finfo(float).eps)
        relative_sigma = abs(next_sigma - sigma) / max(next_sigma, np.finfo(float).eps)
        mu, sigma = next_mu, next_sigma
        if max(relative_mu, relative_sigma) <= cfg.core_convergence:
            converged = True
            break

    core = np.abs(sf - mu) <= cfg.core_clip_sigma * sigma
    core_entries = int(np.count_nonzero(core))
    core_fraction = core_entries / entries
    if core_entries < cfg.min_entries:
        return _invalid_core_fit(entries, "insufficient_core_entries")
    if core_fraction < cfg.min_core_fraction:
        return _invalid_core_fit(entries, "low_core_fraction")

    p_core = p[core]
    momentum_mean = float(np.mean(p_core))
    momentum_error = (
        float(np.std(p_core, ddof=1) / np.sqrt(core_entries))
        if core_entries > 1 else np.nan
    )
    mu_error = sigma / np.sqrt(core_entries)
    sigma_error = sigma / np.sqrt(2.0 * max(1, core_entries - 1))
    return GaussianCoreBinFit(
        momentum_mean=momentum_mean,
        momentum_error=momentum_error,
        mu=mu,
        mu_error=float(mu_error),
        sigma=sigma,
        sigma_error=float(sigma_error),
        entries=entries,
        core_entries=core_entries,
        core_fraction=float(core_fraction),
        reduced_chi2=_gaussian_reduced_chi2(sf, mu, sigma, cfg.core_clip_sigma),
        iterations=iterations,
        status="ok" if converged else "max_iterations",
    )


def sampling_fraction_profile(p: np.ndarray,
                              sf: np.ndarray,
                              p_edges: np.ndarray,
                              cfg: SamplingFractionConfig) -> SamplingFractionProfile:
    fits: list[GaussianCoreBinFit] = []
    for i, (lo, hi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
        upper = p <= hi if i == len(p_edges) - 2 else p < hi
        mask = (p >= lo) & upper
        fits.append(fit_gaussian_core(p[mask], sf[mask], cfg))

    return SamplingFractionProfile(
        momentum=np.asarray([fit.momentum_mean for fit in fits]),
        momentum_errors=np.asarray([fit.momentum_error for fit in fits]),
        mu_values=np.asarray([fit.mu for fit in fits]),
        mu_errors=np.asarray([fit.mu_error for fit in fits]),
        sigma_values=np.asarray([fit.sigma for fit in fits]),
        sigma_errors=np.asarray([fit.sigma_error for fit in fits]),
        counts=np.asarray([fit.entries for fit in fits], dtype=int),
        core_counts=np.asarray([fit.core_entries for fit in fits], dtype=int),
        core_fractions=np.asarray([fit.core_fraction for fit in fits]),
        reduced_chi2=np.asarray([fit.reduced_chi2 for fit in fits]),
        iterations=np.asarray([fit.iterations for fit in fits], dtype=int),
        statuses=tuple(fit.status for fit in fits),
    )


def _profile_coefficients(profile: SamplingFractionProfile,
                          cfg: SamplingFractionConfig,
                          label: str) -> dict[str, list[float]]:
    valid = profile.valid
    if np.count_nonzero(valid) < cfg.poly_degree + 1:
        raise ValueError(
            f"{label} has only {np.count_nonzero(valid)} valid SF core bins; "
            f"need at least {cfg.poly_degree + 1}"
        )
    mu_coeffs = weighted_polyfit_descending(
        profile.momentum,
        profile.mu_values,
        cfg.poly_degree,
        profile.mu_errors,
    )
    sigma_coeffs = weighted_polyfit_descending(
        profile.momentum,
        profile.sigma_values,
        cfg.poly_degree,
        profile.sigma_errors,
    )
    sigma_poly = np.poly1d(sigma_coeffs)
    grid = np.linspace(cfg.momentum_range[0], cfg.momentum_range[1], 1000)
    minimum_sigma = float(np.min(sigma_poly(grid)))
    if not np.isfinite(minimum_sigma) or minimum_sigma <= 0:
        raise ValueError(
            f"{label} sigma polynomial is nonpositive inside the calibrated momentum range "
            f"(minimum {minimum_sigma})"
        )
    return {
        "mu_coeffs": json_ready(mu_coeffs),
        "sigma_coeffs": json_ready(sigma_coeffs),
    }


def derive_sf_calibration(arrays: dict[str, np.ndarray],
                          cfg: SamplingFractionConfig,
                          sector_independent: bool) -> SamplingFractionCalibration:
    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    p_edges = fixed_edges(cfg.momentum_bins, cfg.momentum_range)

    sector_profiles = {
        f"sector_{sec}": sampling_fraction_profile(
            p[sector == sec], sf[sector == sec], p_edges, cfg
        )
        for sec in range(1, 7)
    }
    if sector_independent:
        global_profile = sampling_fraction_profile(p, sf, p_edges, cfg)
        global_coefficients = _profile_coefficients(global_profile, cfg, "global GEMC")
        coefficients = {
            f"sector_{sec}": {
                "mu_coeffs": list(global_coefficients["mu_coeffs"]),
                "sigma_coeffs": list(global_coefficients["sigma_coeffs"]),
            }
            for sec in range(1, 7)
        }
        parameter_profiles = {"global": global_profile}
    else:
        coefficients = {
            key: _profile_coefficients(profile, cfg, key)
            for key, profile in sector_profiles.items()
        }
        parameter_profiles = dict(sector_profiles)

    return SamplingFractionCalibration(
        coefficients=coefficients,
        parameter_profiles=parameter_profiles,
        sector_profiles=sector_profiles,
    )


def derive_sf_coefficients(arrays: dict[str, np.ndarray],
                           cfg: SamplingFractionConfig,
                           sector_independent: bool) -> dict[str, dict[str, list[float]]]:
    return derive_sf_calibration(arrays, cfg, sector_independent).coefficients


def _json_floats(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _profile_metadata(profile: SamplingFractionProfile,
                      coefficients: dict[str, list[float]] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "momentumMean": _json_floats(profile.momentum),
        "momentumError": _json_floats(profile.momentum_errors),
        "mu": _json_floats(profile.mu_values),
        "muError": _json_floats(profile.mu_errors),
        "sigma": _json_floats(profile.sigma_values),
        "sigmaError": _json_floats(profile.sigma_errors),
        "entries": [int(value) for value in profile.counts],
        "coreEntries": [int(value) for value in profile.core_counts],
        "coreFraction": _json_floats(profile.core_fractions),
        "gaussianReducedChi2": _json_floats(profile.reduced_chi2),
        "iterations": [int(value) for value in profile.iterations],
        "status": list(profile.statuses),
    }
    if coefficients is not None:
        valid = profile.valid
        p = profile.momentum[valid]
        mu_fit = np.polyval(coefficients["mu_coeffs"], p)
        sigma_fit = np.polyval(coefficients["sigma_coeffs"], p)
        mu_pull = (profile.mu_values[valid] - mu_fit) / profile.mu_errors[valid]
        sigma_pull = (profile.sigma_values[valid] - sigma_fit) / profile.sigma_errors[valid]
        result["polynomialFit"] = {
            "validBins": int(np.count_nonzero(valid)),
            "muChi2Ndf": float(
                np.sum(mu_pull * mu_pull) / max(1, len(mu_pull) - len(coefficients["mu_coeffs"]))
            ),
            "sigmaChi2Ndf": float(
                np.sum(sigma_pull * sigma_pull)
                / max(1, len(sigma_pull) - len(coefficients["sigma_coeffs"]))
            ),
        }
    return result


def pull_diagnostics(arrays: dict[str, np.ndarray],
                     coefficients: dict[str, dict[str, list[float]]]) -> dict[str, object]:
    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    output: dict[str, object] = {}
    for sec in range(1, 7):
        mask = (sector == sec) & np.isfinite(p) & np.isfinite(sf)
        mu = np.polyval(coefficients[f"sector_{sec}"]["mu_coeffs"], p[mask])
        sigma = np.polyval(coefficients[f"sector_{sec}"]["sigma_coeffs"], p[mask])
        valid = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
        pulls = (sf[mask][valid] - mu[valid]) / sigma[valid]
        core = pulls[np.abs(pulls) <= 2.5]
        output[f"sector_{sec}"] = {
            "entries": int(pulls.size),
            "coreMean": float(np.mean(core)) if core.size else None,
            "coreWidth": float(np.std(core, ddof=1)) if core.size > 1 else None,
            "within3SigmaFraction": float(np.mean(np.abs(pulls) < 3.0)) if pulls.size else None,
            "within3p5SigmaFraction": float(np.mean(np.abs(pulls) < 3.5)) if pulls.size else None,
        }
    return output


def build_metadata(args: argparse.Namespace,
                   input_arrays: dict[str, np.ndarray],
                   arrays: dict[str, np.ndarray],
                   cfg: SamplingFractionConfig,
                   requested_momentum_range: tuple[float, float],
                   calibration: SamplingFractionCalibration,
                   sector_independent: bool) -> dict[str, object]:
    sector = arrays["sf_sector"].astype(int)
    sector_counts = {
        f"sector_{sec}": int(np.count_nonzero(sector == sec))
        for sec in range(1, 7)
    }
    parameter_diagnostics: dict[str, object] = {}
    for key, profile in calibration.parameter_profiles.items():
        coefficient_key = "sector_1" if key == "global" else key
        parameter_diagnostics[key] = _profile_metadata(
            profile, calibration.coefficients[coefficient_key]
        )

    metadata: dict[str, object] = {
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "datasetTag": args.dataset_tag,
        "beamEnergy": args.beam_energy,
        "inputFile": str(args.input_file),
        "inputTree": args.tree,
        "inputElectronCount": int(len(input_arrays["sf_p"])),
        "selectedElectronCount": int(len(arrays["sf_p"])),
        "sectorCounts": sector_counts,
        "sectorIndependentFit": bool(sector_independent),
        "fitConfig": {
            "momentumRange": list(cfg.momentum_range),
            "requestedMomentumRange": list(requested_momentum_range),
            "momentumBins": cfg.momentum_bins,
            "polyDegree": cfg.poly_degree,
            "minBinEntries": cfg.min_entries,
            "maxRows": args.max_rows,
            "estimator": {
                "name": "iterative_truncated_gaussian_core",
                "version": 1,
                "modeSeeded": True,
                "clipSigma": cfg.core_clip_sigma,
                "maxIterations": cfg.core_max_iterations,
                "convergence": cfg.core_convergence,
                "histogramBins": cfg.core_histogram_bins,
                "minCoreFraction": cfg.min_core_fraction,
                "truncationWidthCorrected": True,
            },
            "preselection": {
                "minPcalEnergy": cfg.min_pcal_energy,
                "diagonal": {
                    "yScale": cfg.diagonal_y_scale,
                    "xScale": cfg.diagonal_x_scale,
                    "threshold": cfg.diagonal_threshold,
                    "momentumThreshold": cfg.diagonal_momentum_threshold,
                },
            },
        },
        "parameterFitDiagnostics": parameter_diagnostics,
        "pullDiagnostics": pull_diagnostics(arrays, calibration.coefficients),
        "notes": args.note,
    }
    if sector_independent:
        metadata["sectorValidationDiagnostics"] = {
            key: _profile_metadata(profile)
            for key, profile in calibration.sector_profiles.items()
        }
    if args.run_group:
        metadata["runGroup"] = args.run_group
    if args.skim:
        metadata["skim"] = args.skim
    if args.torus is not None:
        metadata["torus"] = args.torus
    return metadata


def plot_sector_fits(arrays: dict[str, np.ndarray],
                     calibration: SamplingFractionCalibration,
                     cfg: SamplingFractionConfig,
                     sector_independent: bool,
                     output_dir: Path,
                     dataset_tag: str = "",
                     beam_energy: float | None = None) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    x = np.linspace(cfg.momentum_range[0], cfg.momentum_range[1], 300)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for idx, ax in enumerate(axes.flat, start=1):
        key = f"sector_{idx}"
        mask = sector == idx
        ax.hist2d(p[mask], sf[mask], bins=[80, 80], cmap="cividis", cmin=1)
        profile = calibration.sector_profiles[key]
        valid = profile.valid
        coefficients = calibration.coefficients[key]
        mu_poly = np.poly1d(coefficients["mu_coeffs"])
        sigma_poly = np.poly1d(coefficients["sigma_coeffs"])
        ax.errorbar(
            profile.momentum[valid],
            profile.mu_values[valid],
            xerr=profile.momentum_errors[valid],
            yerr=profile.mu_errors[valid],
            fmt="o",
            ms=3.5,
            color="white",
            ecolor="white",
            elinewidth=0.8,
            capsize=2,
            label="sector Gaussian-core mean",
            zorder=3,
        )
        ax.scatter(
            profile.momentum[valid],
            profile.mu_values[valid] + profile.sigma_values[valid],
            marker="_",
            s=55,
            color="orange",
            label="sector core mean +/- sigma",
            zorder=3,
        )
        ax.scatter(
            profile.momentum[valid],
            profile.mu_values[valid] - profile.sigma_values[valid],
            marker="_",
            s=55,
            color="orange",
            zorder=3,
        )
        fit_label = "global GEMC mu fit" if sector_independent else "mu fit"
        width_label = "global GEMC mu +/- 3 sigma" if sector_independent else "mu +/- 3 sigma fit"
        ax.plot(x, mu_poly(x), color="red", lw=2, label=fit_label)
        ax.plot(
            x,
            mu_poly(x) + 3.0 * sigma_poly(x),
            color="cyan",
            lw=1.5,
            ls="--",
            label=width_label,
        )
        ax.plot(x, mu_poly(x) - 3.0 * sigma_poly(x), color="cyan", lw=1.5, ls="--")
        suffix = " (global fit)" if sector_independent else ""
        ax.set_title(f"Sector {idx}{suffix}")
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("Sampling fraction")
        ax.grid(True, alpha=0.25)
        if idx == 1:
            ax.legend(fontsize=8, loc="best")
    save_plot(
        fig,
        output_dir / "sampling_fraction_sector_fits.png",
        "Sampling-fraction Gaussian-core sector fits",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def plot_profile_residuals(calibration: SamplingFractionCalibration,
                           cfg: SamplingFractionConfig,
                           sector_independent: bool,
                           output_dir: Path,
                           dataset_tag: str = "",
                           beam_energy: float | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for sec, ax in enumerate(axes.flat, start=1):
        key = f"sector_{sec}"
        profile = calibration.sector_profiles[key]
        valid = profile.valid
        p = profile.momentum[valid]
        coefficients = calibration.coefficients[key]
        mu_fit = np.polyval(coefficients["mu_coeffs"], p)
        sigma_fit = np.polyval(coefficients["sigma_coeffs"], p)
        mu_residual = 100.0 * (profile.mu_values[valid] - mu_fit) / mu_fit
        sigma_residual = 100.0 * (profile.sigma_values[valid] - sigma_fit) / sigma_fit
        ax.axhline(0.0, color="black", lw=0.8)
        ax.plot(p, mu_residual, "o-", ms=3, lw=1, label="mu residual")
        ax.plot(p, sigma_residual, "s-", ms=3, lw=1, label="sigma residual")
        suffix = " vs global fit" if sector_independent else ""
        ax.set_title(f"Sector {sec}{suffix}")
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("profile - fit [%]")
        ax.grid(True, alpha=0.25)
        if sec == 1:
            ax.legend(fontsize=8)
    save_plot(
        fig,
        output_dir / "sampling_fraction_profile_residuals.png",
        "Sampling-fraction core-profile fit residuals",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def plot_core_fit_quality(calibration: SamplingFractionCalibration,
                          output_dir: Path,
                          dataset_tag: str = "",
                          beam_energy: float | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for sec, ax in enumerate(axes.flat, start=1):
        profile = calibration.sector_profiles[f"sector_{sec}"]
        valid = profile.valid
        p = profile.momentum[valid]
        chi2 = profile.reduced_chi2[valid]
        retention = 100.0 * profile.core_fractions[valid]
        finite_chi2 = np.isfinite(chi2) & (chi2 > 0)
        ax.plot(p[finite_chi2], chi2[finite_chi2], "o-", ms=3, color="tab:blue", label="Gaussian chi2/ndf")
        ax.axhline(1.0, color="tab:blue", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylabel("Gaussian chi2/ndf")
        other = ax.twinx()
        other.plot(p, retention, "s-", ms=3, color="tab:orange", label="core retained")
        other.set_ylabel("core retained [%]")
        other.set_ylim(0.0, 105.0)
        ax.set_title(f"Sector {sec}")
        ax.set_xlabel("p [GeV]")
        ax.grid(True, alpha=0.2)
        if sec == 1:
            lines = ax.get_lines()[:1] + other.get_lines()
            ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    save_plot(
        fig,
        output_dir / "sampling_fraction_core_fit_quality.png",
        "Sampling-fraction per-bin Gaussian-core fit quality",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def plot_pulls(arrays: dict[str, np.ndarray],
               coefficients: dict[str, dict[str, list[float]]],
               output_dir: Path,
               dataset_tag: str = "",
               beam_energy: float | None = None) -> None:
    import matplotlib.pyplot as plt

    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    x = np.linspace(-6.0, 6.0, 500)
    normal = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for sec, ax in enumerate(axes.flat, start=1):
        mask = (sector == sec) & np.isfinite(p) & np.isfinite(sf)
        mu = np.polyval(coefficients[f"sector_{sec}"]["mu_coeffs"], p[mask])
        sigma = np.polyval(coefficients[f"sector_{sec}"]["sigma_coeffs"], p[mask])
        valid = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
        pulls = (sf[mask][valid] - mu[valid]) / sigma[valid]
        visible = pulls[np.abs(pulls) <= 6.0]
        core = pulls[np.abs(pulls) <= 2.5]
        ax.hist(visible, bins=100, range=(-6.0, 6.0), density=True, histtype="step", color="tab:blue")
        ax.plot(x, normal, color="black", ls="--", lw=1, label="standard normal")
        core_mean = float(np.mean(core)) if core.size else np.nan
        core_width = float(np.std(core, ddof=1)) if core.size > 1 else np.nan
        retained = 100.0 * np.mean(np.abs(pulls) < 3.5) if pulls.size else np.nan
        ax.set_title(
            f"Sector {sec}: core mean={core_mean:.2f}, width={core_width:.2f}\n"
            f"|z|<3.5: {retained:.2f}%"
        )
        ax.set_xlabel("(SF - mu(p)) / sigma(p)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
        if sec == 1:
            ax.legend(fontsize=8)
    save_plot(
        fig,
        output_dir / "sampling_fraction_pulls.png",
        "Sampling-fraction standardized residuals",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def plot_diagonal_cut(arrays: dict[str, np.ndarray],
                      cfg: SamplingFractionConfig,
                      output_dir: Path,
                      dataset_tag: str = "",
                      beam_energy: float | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output_dir.mkdir(parents=True, exist_ok=True)
    p = arrays["sf_p"]
    sector = arrays["sf_sector"].astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = arrays["sf_ecin"] / p
        y = arrays["sf_epcal"] / p
    diagonal_pass = sf_diagonal_mask(arrays, cfg)
    active = p >= cfg.diagonal_momentum_threshold

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    x_line = np.linspace(0.0, 0.35, 300)
    y_line = (
        cfg.diagonal_threshold - cfg.diagonal_x_scale * x_line
    ) / cfg.diagonal_y_scale
    for sec, ax in enumerate(axes.flat, start=1):
        mask = (sector == sec) & active & np.isfinite(x) & np.isfinite(y)
        ax.hist2d(
            x[mask],
            y[mask],
            bins=90,
            range=[[0.0, 0.35], [0.0, 0.35]],
            cmap="cividis",
            norm=LogNorm(),
            cmin=1,
        )
        visible_line = (y_line >= 0.0) & (y_line <= 0.35)
        ax.plot(x_line[visible_line], y_line[visible_line], color="red", lw=1.8)
        total = int(np.count_nonzero(mask))
        passed = int(np.count_nonzero(mask & diagonal_pass))
        retained = 100.0 * passed / total if total else 0.0
        ax.set_title(f"Sector {sec}: {total} active, {retained:.1f}% retained")
        ax.set_xlabel(r"$E_{ECIN}/p$")
        ax.set_ylabel(r"$E_{PCAL}/p$")
        ax.grid(True, alpha=0.2)

    save_plot(
        fig,
        output_dir / "sampling_fraction_diagonal_cut.png",
        f"Diagonal SF cut for p >= {cfg.diagonal_momentum_threshold:g} GeV "
        f"(accepted above red line)",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def print_profile_summary(label: str, profile: SamplingFractionProfile) -> None:
    valid = profile.valid
    valid_count = int(np.count_nonzero(valid))
    median_retention = (
        100.0 * float(np.nanmedian(profile.core_fractions[valid])) if valid_count else np.nan
    )
    median_chi2 = (
        float(np.nanmedian(profile.reduced_chi2[valid])) if valid_count else np.nan
    )
    statuses: dict[str, int] = {}
    for status in profile.statuses:
        statuses[status] = statuses.get(status, 0) + 1
    print(
        f"SF Gaussian-core {label}: {valid_count}/{len(profile.statuses)} valid bins, "
        f"median core retention {median_retention:.2f}%, "
        f"median Gaussian chi2/ndf {median_chi2:.3g}, statuses={statuses}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive electron sampling-fraction mu/sigma parameters.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="rParticles")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parameters/sampling_fraction/SF_sigma_cut_params.json"),
    )
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--gemc", action="store_true", help="Use one sector-independent fit copied to all sectors.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--p-min", type=float, default=1.0)
    parser.add_argument("--p-max", type=float, default=10.0)
    parser.add_argument("--p-bins", type=int, default=22)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--min-bin-entries", type=int, default=30)
    parser.add_argument("--core-clip-sigma", type=float, default=2.0)
    parser.add_argument("--core-max-iterations", type=int, default=12)
    parser.add_argument("--core-convergence", type=float, default=1.0e-4)
    parser.add_argument("--core-histogram-bins", type=int, default=100)
    parser.add_argument("--min-core-fraction", type=float, default=0.5)
    parser.add_argument("--min-pcal-energy", type=float, default=0.07)
    parser.add_argument("--diagonal-y-scale", type=float, default=1.0)
    parser.add_argument("--diagonal-x-scale", type=float, default=1.0)
    parser.add_argument("--diagonal-threshold", type=float, default=0.2)
    parser.add_argument("--diagonal-momentum-threshold", type=float, default=4.5)
    parser.add_argument("--dataset-tag", default="", help="Short label for this parameter set, e.g. 6.535RGKSKIM1.")
    parser.add_argument("--beam-energy", type=float, help="Beam energy associated with this parameter set.")
    parser.add_argument("--run-group", default="", help="Run group or campaign label, e.g. RGK.")
    parser.add_argument("--skim", default="", help="Skim or production label.")
    parser.add_argument("--torus", type=int, help="Torus polarity used for this parameter set.")
    parser.add_argument("--note", default="", help="Free-form note stored in the output metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagonal_y_scale == 0:
        raise ValueError("diagonal-y-scale must be nonzero")
    if args.core_clip_sigma <= 1.0:
        raise ValueError("core-clip-sigma must be greater than 1")
    if args.core_max_iterations < 1:
        raise ValueError("core-max-iterations must be positive")
    if args.core_convergence <= 0:
        raise ValueError("core-convergence must be positive")
    if args.core_histogram_bins < 10:
        raise ValueError("core-histogram-bins must be at least 10")
    if not 0 < args.min_core_fraction <= 1:
        raise ValueError("min-core-fraction must lie in (0, 1]")

    requested_momentum_range = (args.p_min, args.p_max)
    input_arrays = electron_arrays(args.input_file, args.tree, args.max_rows)
    preselection_cfg = SamplingFractionConfig(
        min_pcal_energy=args.min_pcal_energy,
        diagonal_y_scale=args.diagonal_y_scale,
        diagonal_x_scale=args.diagonal_x_scale,
        diagonal_threshold=args.diagonal_threshold,
        diagonal_momentum_threshold=args.diagonal_momentum_threshold,
    )
    preselection = sf_preselection_mask(input_arrays, preselection_cfg)
    arrays = select_arrays(input_arrays, preselection)
    print(
        f"SF preselection retained {len(arrays['sf_p'])}/{len(input_arrays['sf_p'])} "
        f"electrons ({100.0 * len(arrays['sf_p']) / max(1, len(input_arrays['sf_p'])):.2f}%)"
    )
    momentum_range = effective_momentum_range(arrays["sf_p"], requested_momentum_range)
    cfg = SamplingFractionConfig(
        momentum_range=momentum_range,
        momentum_bins=args.p_bins,
        poly_degree=args.poly_degree,
        min_entries=args.min_bin_entries,
        min_pcal_energy=args.min_pcal_energy,
        diagonal_y_scale=args.diagonal_y_scale,
        diagonal_x_scale=args.diagonal_x_scale,
        diagonal_threshold=args.diagonal_threshold,
        diagonal_momentum_threshold=args.diagonal_momentum_threshold,
        core_clip_sigma=args.core_clip_sigma,
        core_max_iterations=args.core_max_iterations,
        core_convergence=args.core_convergence,
        core_histogram_bins=args.core_histogram_bins,
        min_core_fraction=args.min_core_fraction,
    )
    if momentum_range != requested_momentum_range:
        print(
            f"Clamped requested momentum range {requested_momentum_range} "
            f"to sample range {momentum_range}"
        )

    calibration = derive_sf_calibration(arrays, cfg, sector_independent=args.gemc)
    for label, profile in calibration.parameter_profiles.items():
        print_profile_summary(label, profile)
    if args.gemc:
        for label, profile in calibration.sector_profiles.items():
            print_profile_summary(f"validation {label}", profile)

    output = {
        "_metadata": build_metadata(
            args,
            input_arrays,
            arrays,
            cfg,
            requested_momentum_range,
            calibration,
            sector_independent=args.gemc,
        ),
        **calibration.coefficients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote sampling-fraction coefficients to {args.output}")

    if args.plot_dir:
        plot_diagonal_cut(
            input_arrays,
            cfg,
            args.plot_dir,
            args.dataset_tag,
            args.beam_energy,
        )
        plot_sector_fits(
            arrays,
            calibration,
            cfg,
            args.gemc,
            args.plot_dir,
            args.dataset_tag,
            args.beam_energy,
        )
        plot_profile_residuals(
            calibration,
            cfg,
            args.gemc,
            args.plot_dir,
            args.dataset_tag,
            args.beam_energy,
        )
        plot_core_fit_quality(
            calibration,
            args.plot_dir,
            args.dataset_tag,
            args.beam_energy,
        )
        plot_pulls(
            arrays,
            calibration.coefficients,
            args.plot_dir,
            args.dataset_tag,
            args.beam_energy,
        )


if __name__ == "__main__":
    main()
