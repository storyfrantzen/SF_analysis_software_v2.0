from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


def build_metadata(args: argparse.Namespace,
                   arrays: dict[str, np.ndarray],
                   cfg: SamplingFractionConfig,
                   requested_momentum_range: tuple[float, float],
                   sector_independent: bool) -> dict[str, object]:
    sector = arrays["sf_sector"].astype(int)
    sector_counts = {
        f"sector_{sec}": int(np.count_nonzero(sector == sec))
        for sec in range(1, 7)
    }
    metadata: dict[str, object] = {
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "datasetTag": args.dataset_tag,
        "beamEnergy": args.beam_energy,
        "inputFile": str(args.input_file),
        "inputTree": args.tree,
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
        },
        "notes": args.note,
    }
    if args.run_group:
        metadata["runGroup"] = args.run_group
    if args.skim:
        metadata["skim"] = args.skim
    if args.torus is not None:
        metadata["torus"] = args.torus
    return metadata


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
    parser.add_argument("--dataset-tag", default="", help="Short label for this parameter set, e.g. 6.535RGKSKIM1.")
    parser.add_argument("--beam-energy", type=float, help="Beam energy associated with this parameter set.")
    parser.add_argument("--run-group", default="", help="Run group or campaign label, e.g. RGK.")
    parser.add_argument("--skim", default="", help="Skim or production label.")
    parser.add_argument("--torus", type=int, help="Torus polarity used for this parameter set.")
    parser.add_argument("--note", default="", help="Free-form note stored in the output metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_momentum_range = (args.p_min, args.p_max)
    arrays = electron_arrays(args.input_file, args.tree, args.max_rows)
    momentum_range = effective_momentum_range(arrays["sf_p"], requested_momentum_range)
    cfg = SamplingFractionConfig(
        momentum_range=momentum_range,
        momentum_bins=args.p_bins,
        poly_degree=args.poly_degree,
        min_entries=args.min_bin_entries,
    )
    if momentum_range != requested_momentum_range:
        print(
            f"Clamped requested momentum range {requested_momentum_range} "
            f"to sample range {momentum_range}"
        )
    coeffs = derive_sf_coefficients(arrays, cfg, sector_independent=args.gemc)
    output = {
        "_metadata": build_metadata(
            args,
            arrays,
            cfg,
            requested_momentum_range,
            sector_independent=args.gemc,
        ),
        **coeffs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Wrote sampling-fraction coefficients to {args.output}")

    if args.plot_dir:
        maybe_plot(arrays, coeffs, cfg, args.gemc, args.plot_dir)


if __name__ == "__main__":
    main()
