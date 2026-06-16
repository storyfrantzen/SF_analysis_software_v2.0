from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fit_utils import BinnedProfile, binned_profile, fixed_edges, json_ready, weighted_polyfit_descending
from .root_arrays import arrays_from_dataframe, define_common_electron_sf, load_dataframe


@dataclass(frozen=True)
class SamplingFractionConfig:
    momentum_range: tuple[float, float] = (1.0, 10.0)
    momentum_bins: int = 22
    poly_degree: int = 2
    min_entries: int = 30


@dataclass(frozen=True)
class SamplingFractionProfile:
    profile: BinnedProfile
    sigma_values: np.ndarray


def electron_arrays(input_file: Path,
                    tree: str,
                    max_rows: int | None) -> dict[str, np.ndarray]:
    df = define_common_electron_sf(load_dataframe(input_file, tree))
    return arrays_from_dataframe(df, ["sf_p", "sf_sector", "sampling_fraction"], max_rows=max_rows)


def sampling_fraction_profile(p: np.ndarray,
                              sf: np.ndarray,
                              p_edges: np.ndarray,
                              min_entries: int) -> SamplingFractionProfile:
    profile = binned_profile(p, sf, p_edges, min_entries)
    sigma_values = np.full_like(profile.means, np.nan, dtype=float)
    for i, (lo, hi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
        upper_mask = p <= hi if i == len(p_edges) - 2 else p < hi
        bin_mask = (p >= lo) & upper_mask & np.isfinite(sf)
        vals = sf[bin_mask]
        if vals.size >= min_entries:
            sigma_values[i] = float(np.std(vals, ddof=1))
    return SamplingFractionProfile(profile=profile, sigma_values=sigma_values)


def derive_sf_coefficients(arrays: dict[str, np.ndarray],
                           cfg: SamplingFractionConfig,
                           sector_independent: bool) -> dict[str, dict[str, list[float]]]:
    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    p_edges = fixed_edges(cfg.momentum_bins, cfg.momentum_range)
    centers = 0.5 * (p_edges[:-1] + p_edges[1:])

    output: dict[str, dict[str, list[float]]] = {}
    for sec in range(1, 7):
        mask = np.ones_like(p, dtype=bool) if sector_independent else sector == sec
        sf_profile = sampling_fraction_profile(p[mask], sf[mask], p_edges, cfg.min_entries)

        mu_coeffs = weighted_polyfit_descending(
            centers,
            sf_profile.profile.means,
            cfg.poly_degree,
            sf_profile.profile.errors,
        )
        sigma_coeffs = weighted_polyfit_descending(centers, sf_profile.sigma_values, cfg.poly_degree)
        output[f"sector_{sec}"] = {
            "mu_coeffs": json_ready(mu_coeffs),
            "sigma_coeffs": json_ready(sigma_coeffs),
        }

    return output


def maybe_plot(arrays: dict[str, np.ndarray],
               coeffs: dict[str, dict[str, list[float]]],
               cfg: SamplingFractionConfig,
               sector_independent: bool,
               output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    p = arrays["sf_p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["sf_sector"].astype(int)
    p_edges = fixed_edges(cfg.momentum_bins, cfg.momentum_range)
    x = np.linspace(cfg.momentum_range[0], cfg.momentum_range[1], 300)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for idx, ax in enumerate(axes.flat, start=1):
        mask = np.ones_like(p, dtype=bool) if sector_independent else sector == idx
        ax.hist2d(p[mask], sf[mask], bins=[80, 80], cmap="cividis", cmin=1)
        sf_profile = sampling_fraction_profile(p[mask], sf[mask], p_edges, cfg.min_entries)
        centers = sf_profile.profile.centers
        means = sf_profile.profile.means
        mean_errors = sf_profile.profile.errors
        sigmas = sf_profile.sigma_values
        sector_coeffs = coeffs[f"sector_{idx}"]
        mu = sector_coeffs["mu_coeffs"]
        sigma = sector_coeffs["sigma_coeffs"]
        mu_poly = np.poly1d(mu)
        sigma_poly = np.poly1d(sigma)
        valid_profile = np.isfinite(means)
        valid_sigma = valid_profile & np.isfinite(sigmas)
        ax.errorbar(
            centers[valid_profile],
            means[valid_profile],
            yerr=mean_errors[valid_profile],
            fmt="o",
            ms=3.5,
            color="white",
            ecolor="white",
            elinewidth=0.8,
            capsize=2,
            label="profile mean",
            zorder=3,
        )
        ax.scatter(
            centers[valid_sigma],
            means[valid_sigma] + sigmas[valid_sigma],
            marker="_",
            s=55,
            color="orange",
            label="profile mean +/- sigma",
            zorder=3,
        )
        ax.scatter(
            centers[valid_sigma],
            means[valid_sigma] - sigmas[valid_sigma],
            marker="_",
            s=55,
            color="orange",
            zorder=3,
        )
        ax.plot(x, mu_poly(x), color="red", lw=2, label="mu fit")
        ax.plot(x, mu_poly(x) + 3.0 * sigma_poly(x), color="cyan", lw=1.5, ls="--", label="mu +/- 3 sigma fit")
        ax.plot(x, mu_poly(x) - 3.0 * sigma_poly(x), color="cyan", lw=1.5, ls="--")
        ax.set_title(f"Sector {idx}")
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("Sampling fraction")
        ax.grid(True, alpha=0.25)
        if idx == 1:
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_fraction_sector_fits.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive electron sampling-fraction mu/sigma parameters.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--output", type=Path, default=Path("SF_sigma_cut_params.json"))
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--gemc", action="store_true", help="Use one sector-independent fit copied to all sectors.")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--p-min", type=float, default=1.0)
    parser.add_argument("--p-max", type=float, default=10.0)
    parser.add_argument("--p-bins", type=int, default=22)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--min-bin-entries", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SamplingFractionConfig(
        momentum_range=(args.p_min, args.p_max),
        momentum_bins=args.p_bins,
        poly_degree=args.poly_degree,
        min_entries=args.min_bin_entries,
    )
    arrays = electron_arrays(args.input_file, args.tree, args.max_rows)
    coeffs = derive_sf_coefficients(arrays, cfg, sector_independent=args.gemc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(coeffs, f, indent=2)
        f.write("\n")
    print(f"Wrote sampling-fraction coefficients to {args.output}")

    if args.plot_dir:
        maybe_plot(arrays, coeffs, cfg, args.gemc, args.plot_dir)


if __name__ == "__main__":
    main()
