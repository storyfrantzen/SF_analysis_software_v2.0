from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fit_utils import binned_profile, fixed_edges, json_ready, weighted_polyfit_descending
from .root_arrays import arrays_from_dataframe, define_common_electron_sf, load_dataframe


@dataclass(frozen=True)
class SamplingFractionConfig:
    momentum_range: tuple[float, float] = (1.0, 10.0)
    momentum_bins: int = 22
    poly_degree: int = 2
    min_entries: int = 30


def electron_arrays(input_file: Path,
                    tree: str,
                    max_rows: int | None) -> dict[str, np.ndarray]:
    df = define_common_electron_sf(load_dataframe(input_file, tree))
    return arrays_from_dataframe(df, ["rec.p", "rec.sector", "sampling_fraction"], max_rows=max_rows)


def derive_sf_coefficients(arrays: dict[str, np.ndarray],
                           cfg: SamplingFractionConfig,
                           sector_independent: bool) -> dict[str, dict[str, list[float]]]:
    p = arrays["rec.p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["rec.sector"].astype(int)
    p_edges = fixed_edges(cfg.momentum_bins, cfg.momentum_range)
    centers = 0.5 * (p_edges[:-1] + p_edges[1:])

    output: dict[str, dict[str, list[float]]] = {}
    for sec in range(1, 7):
        mask = np.ones_like(p, dtype=bool) if sector_independent else sector == sec
        profile = binned_profile(p[mask], sf[mask], p_edges, cfg.min_entries)

        sigma_values = np.full_like(profile.means, np.nan, dtype=float)
        for i, (lo, hi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
            bin_mask = mask & (p >= lo) & (p < hi) & np.isfinite(sf)
            vals = sf[bin_mask]
            if vals.size >= cfg.min_entries:
                sigma_values[i] = float(np.std(vals, ddof=1))

        mu_coeffs = weighted_polyfit_descending(centers, profile.means, cfg.poly_degree, profile.errors)
        sigma_coeffs = weighted_polyfit_descending(centers, sigma_values, cfg.poly_degree)
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
    p = arrays["rec.p"]
    sf = arrays["sampling_fraction"]
    sector = arrays["rec.sector"].astype(int)
    x = np.linspace(cfg.momentum_range[0], cfg.momentum_range[1], 300)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for idx, ax in enumerate(axes.flat, start=1):
        mask = np.ones_like(p, dtype=bool) if sector_independent else sector == idx
        ax.hist2d(p[mask], sf[mask], bins=[80, 80], cmap="cividis", cmin=1)
        sector_coeffs = coeffs[f"sector_{idx}"]
        mu = sector_coeffs["mu_coeffs"]
        sigma = sector_coeffs["sigma_coeffs"]
        mu_poly = np.poly1d(mu)
        sigma_poly = np.poly1d(sigma)
        ax.plot(x, mu_poly(x), color="red", lw=2)
        ax.plot(x, mu_poly(x) + 3.0 * sigma_poly(x), color="cyan", lw=1.5, ls="--")
        ax.plot(x, mu_poly(x) - 3.0 * sigma_poly(x), color="cyan", lw=1.5, ls="--")
        ax.set_title(f"Sector {idx}")
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("Sampling fraction")
        ax.grid(True, alpha=0.25)
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
