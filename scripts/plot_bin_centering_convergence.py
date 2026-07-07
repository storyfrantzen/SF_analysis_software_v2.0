#!/usr/bin/env python3
"""Plot and summarize bin-centering convergence across N values."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairStats:
    label: str
    low: int
    high: int
    bins: int
    median: float
    p68: float
    p90: float
    p95: float
    p99: float
    max_value: float


def _load_artifact(outdir: Path, n: int) -> dict[str, np.ndarray]:
    path = outdir / f"C_BC_N{n}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing merged bin-centering artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _finite_reliable_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    c_bc = np.asarray(data["C_BC"], dtype=float)
    return (
        np.asarray(data["reliable"], dtype=bool)
        & np.asarray(data["computed"], dtype=bool)
        & np.isfinite(c_bc)
    )


def _relative_difference(
    low_data: dict[str, np.ndarray],
    high_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    c_low = np.asarray(low_data["C_BC"], dtype=float)
    c_high = np.asarray(high_data["C_BC"], dtype=float)
    mask = (
        _finite_reliable_mask(low_data)
        & _finite_reliable_mask(high_data)
        & np.isfinite(c_low)
        & np.isfinite(c_high)
        & (c_high != 0.0)
    )
    rel = np.full(c_low.shape, np.nan, dtype=float)
    rel[mask] = np.abs(c_low[mask] / c_high[mask] - 1.0)
    return rel, mask


def _pair_stats(label: str, low: int, high: int, rel: np.ndarray, mask: np.ndarray) -> PairStats:
    values = rel[mask]
    if values.size == 0:
        return PairStats(label, low, high, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    return PairStats(
        label=label,
        low=low,
        high=high,
        bins=int(values.size),
        median=float(np.median(values)),
        p68=float(np.percentile(values, 68)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        max_value=float(np.max(values)),
    )


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.8g}"


def _write_stats_csv(path: Path, stats: list[PairStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "low_N", "high_N", "bins", "median", "p68", "p90", "p95", "p99", "max"])
        for stat in stats:
            writer.writerow(
                [
                    stat.label,
                    stat.low,
                    stat.high,
                    stat.bins,
                    _format_float(stat.median),
                    _format_float(stat.p68),
                    _format_float(stat.p90),
                    _format_float(stat.p95),
                    _format_float(stat.p99),
                    _format_float(stat.max_value),
                ]
            )


def _write_summary(
    path: Path,
    n_values: list[int],
    artifacts: dict[int, dict[str, np.ndarray]],
    adjacent_stats: list[PairStats],
    reference_stats: list[PairStats],
    reference_n: int,
) -> None:
    lines = ["# Bin-centering convergence summary", ""]
    lines.append("## Reliable bins")
    for n in n_values:
        data = artifacts[n]
        mask = _finite_reliable_mask(data)
        values = np.asarray(data["C_BC"], dtype=float)
        if mask.any():
            lines.append(
                f"- N={n}: reliable {int(mask.sum())}/{mask.size}, "
                f"mean C_BC {_format_float(float(np.nanmean(values[mask])))}"
            )
        else:
            lines.append(f"- N={n}: reliable 0/{mask.size}")

    lines.extend(["", "## Adjacent-N differences on common reliable bins"])
    for stat in adjacent_stats:
        if stat.bins == 0:
            lines.append(f"- N={stat.low} -> N={stat.high}: no common reliable bins")
        else:
            lines.append(
                f"- N={stat.low} -> N={stat.high}: bins={stat.bins}, "
                f"median={_format_float(stat.median)}, p95={_format_float(stat.p95)}, "
                f"max={_format_float(stat.max_value)}"
            )

    lines.extend(["", f"## Differences relative to N={reference_n}"])
    for stat in reference_stats:
        if stat.bins == 0:
            lines.append(f"- N={stat.low} -> N={reference_n}: no common reliable bins")
        else:
            lines.append(
                f"- N={stat.low} -> N={reference_n}: bins={stat.bins}, "
                f"median={_format_float(stat.median)}, p95={_format_float(stat.p95)}, "
                f"p99={_format_float(stat.p99)}, max={_format_float(stat.max_value)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _center_array(data: dict[str, np.ndarray], key: str, fallback_axis: int) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=float)
    shape = np.asarray(data["C_BC"]).shape
    axis = np.arange(shape[fallback_axis], dtype=float)
    expand_shape = [1] * len(shape)
    expand_shape[fallback_axis] = shape[fallback_axis]
    return np.broadcast_to(axis.reshape(expand_shape), shape)


def _write_worst_bins_csv(
    path: Path,
    low_n: int,
    reference_n: int,
    low_data: dict[str, np.ndarray],
    reference_data: dict[str, np.ndarray],
    rel: np.ndarray,
    mask: np.ndarray,
    limit: int,
) -> None:
    values = rel[mask]
    if values.size == 0:
        path.write_text("", encoding="utf-8")
        return

    coords = np.argwhere(mask)
    order = np.argsort(values)[::-1][:limit]
    x_b = _center_array(reference_data, "xB_center", 1)
    q2 = _center_array(reference_data, "q2_center", 0)
    minus_t = _center_array(reference_data, "minus_t_center", 2)
    phi = _center_array(reference_data, "phi_center", 3)
    c_low = np.asarray(low_data["C_BC"], dtype=float)
    c_ref = np.asarray(reference_data["C_BC"], dtype=float)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "low_N",
                "reference_N",
                "relative_difference",
                "C_BC_low",
                "C_BC_reference",
                "q2_bin",
                "xB_bin",
                "minus_t_bin",
                "phi_bin",
                "q2_center",
                "xB_center",
                "minus_t_center",
                "phi_center",
            ]
        )
        for rank, value_index in enumerate(order, start=1):
            index = tuple(int(i) for i in coords[value_index])
            writer.writerow(
                [
                    rank,
                    low_n,
                    reference_n,
                    _format_float(float(rel[index])),
                    _format_float(float(c_low[index])),
                    _format_float(float(c_ref[index])),
                    index[0],
                    index[1],
                    index[2],
                    index[3],
                    _format_float(float(q2[index])),
                    _format_float(float(x_b[index])),
                    _format_float(float(minus_t[index])),
                    _format_float(float(phi[index])),
                ]
            )


def _plot_convergence(
    path: Path,
    n_values: list[int],
    artifacts: dict[int, dict[str, np.ndarray]],
    reference_n: int,
    reference_rel: dict[int, tuple[np.ndarray, np.ndarray]],
    adjacent_rel: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes.ravel()

    reliable_counts = [int(_finite_reliable_mask(artifacts[n]).sum()) for n in n_values]
    ax[0].bar([str(n) for n in n_values], reliable_counts, color="#5276a7")
    ax[0].set_title("Reliable bins")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("count")

    for n in n_values:
        if n == reference_n:
            continue
        rel, mask = reference_rel[n]
        values = rel[mask]
        if values.size == 0:
            continue
        ax[1].hist(
            values,
            bins=np.logspace(-5, 0, 90),
            histtype="step",
            linewidth=1.6,
            label=f"N={n} vs {reference_n}",
        )
        xs = np.sort(values)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax[2].plot(xs, ys, linewidth=1.6, label=f"N={n} vs {reference_n}")

    ax[1].set_xscale("log")
    ax[1].set_title(f"Relative difference vs N={reference_n}")
    ax[1].set_xlabel(r"$|C_{BC}(N)/C_{BC}(N_{ref}) - 1|$")
    ax[1].set_ylabel("bins")
    ax[1].legend(fontsize="small")

    ax[2].set_xscale("log")
    ax[2].set_title("CDF vs reference")
    ax[2].set_xlabel(r"$|C_{BC}(N)/C_{BC}(N_{ref}) - 1|$")
    ax[2].set_ylabel("fraction below")
    ax[2].grid(True, alpha=0.25)
    ax[2].legend(fontsize="small")

    adjacent_labels = []
    adjacent_p95 = []
    adjacent_p99 = []
    for low, high in zip(n_values, n_values[1:]):
        rel, mask = adjacent_rel[(low, high)]
        values = rel[mask]
        if values.size == 0:
            continue
        adjacent_labels.append(f"{low}->{high}")
        adjacent_p95.append(float(np.percentile(values, 95)))
        adjacent_p99.append(float(np.percentile(values, 99)))
    x = np.arange(len(adjacent_labels))
    ax[3].plot(x, adjacent_p95, marker="o", label="p95")
    ax[3].plot(x, adjacent_p99, marker="s", label="p99")
    ax[3].set_xticks(x, adjacent_labels)
    ax[3].set_title("Adjacent-N convergence")
    ax[3].set_xlabel("comparison")
    ax[3].set_ylabel("relative difference")
    ax[3].grid(True, alpha=0.25)
    ax[3].legend()

    previous_n = n_values[-2] if reference_n == n_values[-1] and len(n_values) > 1 else n_values[-1]
    if previous_n == reference_n:
        previous_n = n_values[0]
    rel, mask = reference_rel[previous_n]
    phi_profile = np.nanpercentile(np.where(mask, rel, np.nan), 95, axis=(0, 1, 2))
    ax[4].plot(phi_profile, marker="o", color="#a95c68")
    ax[4].set_title(f"N={previous_n} vs {reference_n}: p95 by phi bin")
    ax[4].set_xlabel("phi bin index")
    ax[4].set_ylabel("p95 relative difference")
    ax[4].grid(True, alpha=0.25)

    collapsed = np.nanpercentile(np.where(mask, rel, np.nan), 95, axis=(2, 3))
    image = ax[5].imshow(collapsed, origin="lower", aspect="auto", cmap="viridis")
    ax[5].set_title(f"N={previous_n} vs {reference_n}: p95 by Q2/xB bin")
    ax[5].set_xlabel("xB bin index")
    ax[5].set_ylabel("Q2 bin index")
    fig.colorbar(image, ax=ax[5], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot bin-centering convergence across merged C_BC_N*.npz artifacts."
    )
    parser.add_argument("outdir", type=Path, help="Directory containing merged C_BC_N*.npz files")
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="N values to compare, in increasing order",
    )
    parser.add_argument("--reference-N", type=int, default=None, help="Reference N; defaults to the largest N")
    parser.add_argument(
        "--output-prefix",
        default="bin_centering_convergence",
        help="Prefix for output PNG/CSV/Markdown files",
    )
    parser.add_argument("--top-bins", type=int, default=50, help="Rows to write in the worst-bin CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = args.outdir
    n_values = sorted(dict.fromkeys(args.n_values))
    if len(n_values) < 2:
        raise ValueError("provide at least two N values")
    reference_n = args.reference_N if args.reference_N is not None else n_values[-1]
    if reference_n not in n_values:
        raise ValueError(f"--reference-N {reference_n} is not present in --n-values")

    artifacts = {n: _load_artifact(outdir, n) for n in n_values}
    adjacent_rel: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    adjacent_stats: list[PairStats] = []
    for low, high in zip(n_values, n_values[1:]):
        rel, mask = _relative_difference(artifacts[low], artifacts[high])
        adjacent_rel[(low, high)] = (rel, mask)
        adjacent_stats.append(_pair_stats("adjacent", low, high, rel, mask))

    reference_rel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reference_stats: list[PairStats] = []
    for n in n_values:
        if n == reference_n:
            continue
        rel, mask = _relative_difference(artifacts[n], artifacts[reference_n])
        reference_rel[n] = (rel, mask)
        reference_stats.append(_pair_stats("reference", n, reference_n, rel, mask))

    prefix = outdir / args.output_prefix
    summary_path = prefix.with_suffix(".md")
    stats_path = prefix.with_name(prefix.name + "_stats.csv")
    plot_path = prefix.with_suffix(".png")
    worst_n = n_values[-2] if reference_n == n_values[-1] else n_values[-1]
    worst_path = prefix.with_name(prefix.name + f"_worst_N{worst_n}_vs_N{reference_n}.csv")

    _write_summary(summary_path, n_values, artifacts, adjacent_stats, reference_stats, reference_n)
    _write_stats_csv(stats_path, adjacent_stats + reference_stats)
    rel, mask = reference_rel[worst_n]
    _write_worst_bins_csv(
        worst_path,
        worst_n,
        reference_n,
        artifacts[worst_n],
        artifacts[reference_n],
        rel,
        mask,
        args.top_bins,
    )
    _plot_convergence(plot_path, n_values, artifacts, reference_n, reference_rel, adjacent_rel)

    print(f"Wrote {summary_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {worst_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
