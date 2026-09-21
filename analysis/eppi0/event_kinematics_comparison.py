from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np

from .event_kinematics_diagnostics import (
    CORRELATIONS,
    TOPOLOGY_LABELS,
    TOPOLOGY_TITLES,
    PlotVariable,
    available_variables,
    finite_values,
    observed_topologies,
    robust_range,
)


Array = np.ndarray


METRIC_FIELDS = (
    "scope",
    "dimension",
    "variable",
    "data_entries",
    "gemc_entries",
    "jensen_shannon_divergence_bits",
    "total_variation_distance",
)


def common_variables(
    data: Mapping[str, Array], gemc: Mapping[str, Array]
) -> tuple[PlotVariable, ...]:
    data_names = {variable.branch for variable in available_variables(data)}
    return tuple(
        variable
        for variable in available_variables(gemc)
        if variable.branch in data_names
    )


def shape_metrics(data_counts: Array, gemc_counts: Array) -> tuple[float, float]:
    data = np.asarray(data_counts, dtype=float).ravel()
    gemc = np.asarray(gemc_counts, dtype=float).ravel()
    if data.shape != gemc.shape:
        raise ValueError("histograms must have matching shapes")
    if np.any(data < 0.0) or np.any(gemc < 0.0):
        raise ValueError("histograms must be nonnegative")
    data_sum = float(np.sum(data))
    gemc_sum = float(np.sum(gemc))
    if data_sum <= 0.0 or gemc_sum <= 0.0:
        return float("nan"), float("nan")
    p = data / data_sum
    q = gemc / gemc_sum
    mixture = 0.5 * (p + q)
    data_support = p > 0.0
    gemc_support = q > 0.0
    js = 0.5 * np.sum(
        p[data_support] * np.log2(p[data_support] / mixture[data_support])
    )
    js += 0.5 * np.sum(
        q[gemc_support] * np.log2(q[gemc_support] / mixture[gemc_support])
    )
    total_variation = 0.5 * np.sum(np.abs(p - q))
    return float(js), float(total_variation)


def comparison_range(
    variable: PlotVariable,
    data: Mapping[str, Array],
    gemc: Mapping[str, Array],
    data_mask: Array,
    gemc_mask: Array,
) -> tuple[float, float]:
    if variable.fixed_range is not None:
        return float(variable.fixed_range[0]), float(variable.fixed_range[1])
    data_range = robust_range(finite_values(variable, data, data_mask))
    gemc_range = robust_range(finite_values(variable, gemc, gemc_mask))
    return min(data_range[0], gemc_range[0]), max(data_range[1], gemc_range[1])


def render_comparison_report(
    output: Path,
    data: Mapping[str, Array],
    gemc: Mapping[str, Array],
    data_mask: Array,
    gemc_mask: Array,
    data_topology: Array,
    gemc_topology: Array,
    *,
    label: str,
    data_label: str,
    gemc_label: str,
    provenance_lines: Sequence[str],
    minimum_ratio_count: int = 5,
) -> tuple[int, list[dict[str, object]]]:
    data_mask = np.asarray(data_mask, dtype=bool)
    gemc_mask = np.asarray(gemc_mask, dtype=bool)
    variables = common_variables(data, gemc)
    variable_by_name = {variable.branch: variable for variable in variables}
    ranges = {
        variable.branch: comparison_range(
            variable, data, gemc, data_mask, gemc_mask
        )
        for variable in variables
    }
    topologies = sorted(
        set(observed_topologies(data_topology, data_mask))
        | set(observed_topologies(gemc_topology, gemc_mask))
    )
    scopes: list[tuple[str, Array, Array]] = [
        ("Topology integrated", data_mask, gemc_mask)
    ]
    scopes.extend(
        (
            f"{TOPOLOGY_LABELS[group]}: {TOPOLOGY_TITLES[group]}",
            data_mask & (data_topology == group),
            gemc_mask & (gemc_topology == group),
        )
        for group in topologies
    )

    metrics: list[dict[str, object]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    pages = 0
    try:
        with PdfPages(temporary_output) as pdf:
            _title_page(
                pdf,
                label,
                data_label,
                gemc_label,
                data_mask,
                gemc_mask,
                data_topology,
                gemc_topology,
                topologies,
                provenance_lines,
            )
            pages += 1
            for scope_name, scope_data, scope_gemc in scopes:
                pages_added, rows = _one_dimensional_pages(
                    pdf,
                    label,
                    scope_name,
                    data_label,
                    gemc_label,
                    data,
                    gemc,
                    scope_data,
                    scope_gemc,
                    variables,
                    ranges,
                    minimum_ratio_count,
                )
                pages += pages_added
                metrics.extend(rows)
                pages_added, rows = _correlation_pages(
                    pdf,
                    label,
                    scope_name,
                    data_label,
                    gemc_label,
                    data,
                    gemc,
                    scope_data,
                    scope_gemc,
                    variable_by_name,
                    ranges,
                    minimum_ratio_count,
                )
                pages += pages_added
                metrics.extend(rows)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return pages, metrics


def _title_page(
    pdf: PdfPages,
    label: str,
    data_label: str,
    gemc_label: str,
    data_mask: Array,
    gemc_mask: Array,
    data_topology: Array,
    gemc_topology: Array,
    topologies: Sequence[int],
    provenance_lines: Sequence[str],
) -> None:
    figure = plt.figure(figsize=(11.0, 8.5))
    figure.patch.set_facecolor("white")
    figure.text(0.07, 0.91, "Data/GEMC event-kinematics comparison", fontsize=23, weight="bold")
    figure.text(0.07, 0.855, label, fontsize=15, color="#374151")
    figure.text(
        0.07,
        0.795,
        f"{data_label}: {np.count_nonzero(data_mask):,} final candidates",
        fontsize=13,
        weight="bold",
    )
    figure.text(
        0.07,
        0.755,
        f"{gemc_label}: {np.count_nonzero(gemc_mask):,} final candidates",
        fontsize=13,
        weight="bold",
        color="#2563eb",
    )
    y = 0.695
    figure.text(0.09, y, "topology       data count (fraction)       GEMC count (fraction)", fontsize=10, family="monospace")
    y -= 0.036
    data_total = max(int(np.count_nonzero(data_mask)), 1)
    gemc_total = max(int(np.count_nonzero(gemc_mask)), 1)
    for group in topologies:
        data_count = int(np.count_nonzero(data_mask & (data_topology == group)))
        gemc_count = int(np.count_nonzero(gemc_mask & (gemc_topology == group)))
        figure.text(
            0.09,
            y,
            f"{TOPOLOGY_LABELS[group]:<10} {data_count:>10,} ({100.0 * data_count / data_total:5.2f}%)"
            f"       {gemc_count:>10,} ({100.0 * gemc_count / gemc_total:5.2f}%)",
            fontsize=10,
            family="monospace",
        )
        y -= 0.033
    figure.text(0.07, y - 0.005, "Interpretation", fontsize=13, weight="bold")
    y -= 0.052
    interpretation = (
        "Histograms compare unit-normalized reconstructed-candidate shapes after each sample's "
        "own final exclusivity mask. GEMC is not reweighted to the data distribution. Differences "
        "in Q2, xB, -t, or phi can therefore reflect the event generator as well as detector and "
        "reconstruction modeling. Ratios are displayed only where both histograms contain the "
        "configured minimum number of candidates."
    )
    for wrapped in _wrap(interpretation, 112):
        figure.text(0.09, y, wrapped, fontsize=9.2, color="#4b5563")
        y -= 0.027
    figure.text(0.07, y - 0.005, "Provenance", fontsize=13, weight="bold")
    y -= 0.052
    for line in provenance_lines:
        for wrapped in _wrap(line, 112):
            figure.text(0.09, y, wrapped, fontsize=8.1, family="monospace")
            y -= 0.025
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _one_dimensional_pages(
    pdf: PdfPages,
    label: str,
    scope_name: str,
    data_label: str,
    gemc_label: str,
    data: Mapping[str, Array],
    gemc: Mapping[str, Array],
    data_mask: Array,
    gemc_mask: Array,
    variables: Sequence[PlotVariable],
    ranges: Mapping[str, tuple[float, float]],
    minimum_ratio_count: int,
) -> tuple[int, list[dict[str, object]]]:
    pages = 0
    rows: list[dict[str, object]] = []
    for page_index, batch in enumerate(_batches(variables, 6), start=1):
        figure = plt.figure(figsize=(11.0, 8.5), constrained_layout=True)
        outer = figure.add_gridspec(3, 2)
        for panel, variable in enumerate(batch):
            subgrid = outer[panel // 2, panel % 2].subgridspec(
                2, 1, height_ratios=(3.2, 1.0), hspace=0.05
            )
            main_axis = figure.add_subplot(subgrid[0])
            ratio_axis = figure.add_subplot(subgrid[1], sharex=main_axis)
            lo, hi = ranges[variable.branch]
            edges = np.linspace(lo, hi, variable.bins + 1)
            data_values = finite_values(variable, data, data_mask)
            gemc_values = finite_values(variable, gemc, gemc_mask)
            data_counts, _ = np.histogram(data_values, bins=edges)
            gemc_counts, _ = np.histogram(gemc_values, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            data_density = _density(data_counts, widths)
            gemc_density = _density(gemc_counts, widths)
            data_error = _density_error(data_counts, widths)
            main_axis.stairs(
                gemc_density,
                edges,
                color="#2563eb",
                linewidth=1.4,
                fill=True,
                alpha=0.22,
                label=gemc_label,
            )
            main_axis.errorbar(
                centers,
                data_density,
                yerr=data_error,
                fmt="o",
                markersize=2.2,
                linewidth=0.7,
                color="#111827",
                label=data_label,
            )
            js, total_variation = shape_metrics(data_counts, gemc_counts)
            main_axis.text(
                0.98,
                0.95,
                f"JS={js:.4f} bits\nTV={total_variation:.4f}",
                transform=main_axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            main_axis.set_title(variable.title, fontsize=9.5)
            main_axis.set_ylabel("Unit-normalized density")
            main_axis.grid(alpha=0.18)
            main_axis.legend(fontsize=6.8, frameon=False)
            main_axis.tick_params(labelbottom=False)

            ratio_valid = (
                (data_counts >= minimum_ratio_count)
                & (gemc_counts >= minimum_ratio_count)
                & (gemc_density > 0.0)
            )
            ratio = np.divide(
                data_density,
                gemc_density,
                out=np.full(data_density.shape, np.nan),
                where=ratio_valid,
            )
            ratio_error = np.full(ratio.shape, np.nan)
            ratio_error[ratio_valid] = ratio[ratio_valid] * np.sqrt(
                1.0 / data_counts[ratio_valid] + 1.0 / gemc_counts[ratio_valid]
            )
            ratio_axis.axhline(1.0, color="#6b7280", linewidth=0.8)
            ratio_axis.errorbar(
                centers[ratio_valid],
                ratio[ratio_valid],
                yerr=ratio_error[ratio_valid],
                fmt="o",
                markersize=2.0,
                linewidth=0.6,
                color="#111827",
            )
            ratio_axis.set_ylim(0.0, 2.5)
            ratio_axis.set_ylabel("D/MC", fontsize=7.5)
            ratio_axis.set_xlabel(variable.x_label, fontsize=8.5)
            ratio_axis.grid(alpha=0.18)
            ratio_axis.tick_params(labelsize=7.5)
            rows.append(
                _metric_row(
                    scope_name,
                    "1D",
                    variable.branch,
                    int(np.sum(data_counts)),
                    int(np.sum(gemc_counts)),
                    js,
                    total_variation,
                )
            )
        for panel in range(len(batch), 6):
            axis = figure.add_subplot(outer[panel // 2, panel % 2])
            axis.set_visible(False)
        figure.suptitle(
            f"{label}\n{scope_name} - data/GEMC shapes ({page_index})\n"
            f"data N={np.count_nonzero(data_mask):,}; GEMC N={np.count_nonzero(gemc_mask):,}",
            fontsize=12.5,
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages, rows


def _correlation_pages(
    pdf: PdfPages,
    label: str,
    scope_name: str,
    data_label: str,
    gemc_label: str,
    data: Mapping[str, Array],
    gemc: Mapping[str, Array],
    data_mask: Array,
    gemc_mask: Array,
    variable_by_name: Mapping[str, PlotVariable],
    ranges: Mapping[str, tuple[float, float]],
    minimum_ratio_count: int,
) -> tuple[int, list[dict[str, object]]]:
    pages = 0
    rows: list[dict[str, object]] = []
    for correlation in CORRELATIONS:
        if correlation.x not in variable_by_name or correlation.y not in variable_by_name:
            continue
        x_variable = variable_by_name[correlation.x]
        y_variable = variable_by_name[correlation.y]
        data_x, data_y = _paired_values(data, data_mask, x_variable, y_variable)
        gemc_x, gemc_y = _paired_values(gemc, gemc_mask, x_variable, y_variable)
        histogram_range = (ranges[correlation.x], ranges[correlation.y])
        data_counts, x_edges, y_edges = np.histogram2d(
            data_x, data_y, bins=48, range=histogram_range
        )
        gemc_counts, _, _ = np.histogram2d(
            gemc_x, gemc_y, bins=48, range=histogram_range
        )
        data_density = data_counts / max(float(np.sum(data_counts)), 1.0)
        gemc_density = gemc_counts / max(float(np.sum(gemc_counts)), 1.0)
        positive = np.concatenate(
            [data_density[data_density > 0.0], gemc_density[gemc_density > 0.0]]
        )
        norm = None
        if positive.size:
            vmin = max(float(np.quantile(positive, 0.02)), 1.0e-12)
            vmax = max(float(np.max(positive)), 1.01 * vmin)
            norm = LogNorm(
                vmin=vmin,
                vmax=vmax,
            )
        ratio_valid = (
            (data_counts >= minimum_ratio_count)
            & (gemc_counts >= minimum_ratio_count)
            & (gemc_density > 0.0)
        )
        log_ratio = np.full(data_density.shape, np.nan)
        log_ratio[ratio_valid] = np.log10(
            data_density[ratio_valid] / gemc_density[ratio_valid]
        )
        js, total_variation = shape_metrics(data_counts, gemc_counts)

        figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), constrained_layout=True)
        data_image = axes[0].pcolormesh(
            x_edges, y_edges, data_density.T, shading="auto", cmap="viridis", norm=norm
        )
        axes[0].set_title(f"{data_label}\nunit-normalized density", fontsize=9)
        gemc_image = axes[1].pcolormesh(
            x_edges, y_edges, gemc_density.T, shading="auto", cmap="viridis", norm=norm
        )
        axes[1].set_title(f"{gemc_label}\nunit-normalized density", fontsize=9)
        ratio_image = axes[2].pcolormesh(
            x_edges,
            y_edges,
            log_ratio.T,
            shading="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        )
        axes[2].set_title(r"$\log_{10}$(data/GEMC density)", fontsize=9)
        for axis in axes:
            axis.set_xlabel(x_variable.x_label)
            axis.set_ylabel(y_variable.x_label)
        figure.colorbar(data_image, ax=axes[:2], pad=0.015, label="Probability per 2D bin")
        figure.colorbar(ratio_image, ax=axes[2], pad=0.015, label=r"$\log_{10}$(D/MC)")
        figure.suptitle(
            f"{label}\n{scope_name} - {correlation.title}\n"
            f"JS={js:.4f} bits; TV={total_variation:.4f}",
            fontsize=12,
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
        rows.append(
            _metric_row(
                scope_name,
                "2D",
                f"{correlation.y}_vs_{correlation.x}",
                int(np.sum(data_counts)),
                int(np.sum(gemc_counts)),
                js,
                total_variation,
            )
        )
    return pages, rows


def _paired_values(
    arrays: Mapping[str, Array],
    mask: Array,
    x_variable: PlotVariable,
    y_variable: PlotVariable,
) -> tuple[Array, Array]:
    x = x_variable.transform(np.asarray(arrays[x_variable.branch])[mask])
    y = y_variable.transform(np.asarray(arrays[y_variable.branch])[mask])
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _density(counts: Array, widths: Array) -> Array:
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(np.asarray(counts).shape, dtype=float)
    return np.asarray(counts, dtype=float) / total / widths


def _density_error(counts: Array, widths: Array) -> Array:
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(np.asarray(counts).shape, dtype=float)
    return np.sqrt(np.asarray(counts, dtype=float)) / total / widths


def _metric_row(
    scope: str,
    dimension: str,
    variable: str,
    data_entries: int,
    gemc_entries: int,
    js: float,
    total_variation: float,
) -> dict[str, object]:
    return {
        "scope": scope,
        "dimension": dimension,
        "variable": variable,
        "data_entries": int(data_entries),
        "gemc_entries": int(gemc_entries),
        "jensen_shannon_divergence_bits": js,
        "total_variation_distance": total_variation,
    }


def _batches(values: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]
