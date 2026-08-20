from __future__ import annotations

from pathlib import Path

import numpy as np

from .exclusivity import ExclusivityCuts, topology_ids_from_groups


Array = np.ndarray

_PDF_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.1,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "text.color": "black",
    "axes.edgecolor": "#202020",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.35,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_OBSERVED_COLOR = "#202020"
_TOTAL_COLOR = "#000000"
_CUT_COMPONENT_COLOR = "#0072B2"
_NUISANCE_COLOR = "#D55E00"
_BACKGROUND_COLOR = "#CC3311"
_BOUNDARY_COLOR = "#009E73"
_AUDIT_BOUNDARY_COLOR = "#CC79A7"


def diagnostic_group_ids(
    cuts: ExclusivityCuts,
    requested: Array | None = None,
    maximum_groups: int = 24,
) -> Array:
    """Select explicit groups or the groups with the poorest reduced chi-square."""
    if maximum_groups <= 0:
        raise ValueError("maximum_groups must be positive")
    if requested is not None:
        requested_ids = np.asarray(requested, dtype=np.int64)
        missing = requested_ids[~np.isin(requested_ids, cuts.group_ids)]
        if missing.size:
            raise ValueError(f"unknown retained exclusivity group IDs: {missing.tolist()}")
        return requested_ids
    if cuts.group_ids.size <= maximum_groups:
        return cuts.group_ids.copy()
    reduced = cuts.pearson_chi2 / np.maximum(cuts.fit_ndof, 1)
    score = np.nanmax(reduced, axis=1)
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    return cuts.group_ids[order[:maximum_groups]]


def render_diagnostics(
    cuts: ExclusivityCuts,
    output: str | Path,
    *,
    group_ids: Array | None = None,
    maximum_groups: int = 24,
) -> tuple[int, ...]:
    """Render a compact summary and one six-panel fit page per selected group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    selected_ids = diagnostic_group_ids(cuts, group_ids, maximum_groups)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.searchsorted(cuts.group_ids, selected_ids)

    with matplotlib.rc_context(rc=_PDF_STYLE):
        with PdfPages(
            output_path,
            metadata={
                "Title": "Exclusivity fit and cut-flow diagnostics",
                "Subject": cuts.estimator,
            },
        ) as pdf:
            _summary_page(cuts, selected_ids, pdf, plt)
            for group_id, position in zip(selected_ids, positions, strict=True):
                _group_page(cuts, int(group_id), int(position), pdf, plt)
    return tuple(int(item) for item in selected_ids)


def render_comparison_diagnostics(
    data_cuts: ExclusivityCuts,
    gemc_cuts: ExclusivityCuts,
    output: str | Path,
    *,
    data_label: str = "data",
    gemc_label: str = "GEMC",
) -> tuple[str, ...]:
    """Render one variable page with topology rows and data/GEMC columns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    variables, topologies = _comparison_layout(data_cuts, gemc_cuts)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(rc=_PDF_STYLE):
        with PdfPages(
            output_path,
            metadata={
                "Title": "Data and GEMC exclusivity fit comparison",
                "Subject": f"{data_cuts.estimator}; {gemc_cuts.estimator}",
            },
        ) as pdf:
            for variable_index, variable in enumerate(variables):
                _comparison_variable_page(
                    data_cuts,
                    gemc_cuts,
                    variable_index,
                    variable,
                    topologies,
                    data_label,
                    gemc_label,
                    pdf,
                    plt,
                )
    return variables


def _comparison_layout(
    data_cuts: ExclusivityCuts,
    gemc_cuts: ExclusivityCuts,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not data_cuts.global_mode or not gemc_cuts.global_mode:
        raise ValueError(
            "paired topology diagnostics require global-by-topology cut tables"
        )
    data_variables = tuple(str(item) for item in data_cuts.variables)
    gemc_variables = tuple(str(item) for item in gemc_cuts.variables)
    if data_variables != gemc_variables:
        raise ValueError(
            "data and GEMC cut tables must contain the same variables in the same order"
        )
    populated = []
    for cuts in (data_cuts, gemc_cuts):
        populated.extend(
            int(item)
            for item in topology_ids_from_groups(
                cuts.populated_group_ids, cuts.global_mode
            )
        )
    topologies = tuple(sorted(set(populated)))
    if not topologies:
        raise ValueError("data and GEMC cut tables contain no populated topologies")
    return data_variables, topologies


def _comparison_variable_page(
    data_cuts,
    gemc_cuts,
    variable_index,
    variable,
    topologies,
    data_label,
    gemc_label,
    pdf,
    plt,
) -> None:
    number_of_rows = len(topologies)
    figure_height = max(5.4, 2.7 * number_of_rows + 2.0)
    figure = plt.figure(figsize=(16, figure_height), facecolor="white")
    figure.patch.set_alpha(1.0)
    title_y = 1.0 - 0.18 / figure_height
    subtitle_y = 1.0 - 0.48 / figure_height
    column_header_y = 1.0 - 0.80 / figure_height
    legend_y = 1.0 - 1.05 / figure_height
    outer = figure.add_gridspec(
        number_of_rows,
        2,
        left=0.035,
        right=0.985,
        bottom=0.55 / figure_height,
        top=1.0 - 1.55 / figure_height,
        hspace=0.46,
        wspace=0.105,
    )
    figure.suptitle(
        f"{variable}: data and GEMC exclusivity fit comparison",
        y=title_y,
        color="black",
        fontsize=14,
    )
    figure.text(
        0.5,
        subtitle_y,
        "Common x range across all topologies and both samples; y scales are independent",
        ha="center",
        va="center",
        color="#404040",
        fontsize=8,
    )
    figure.text(
        0.255, column_header_y, data_label, ha="center", va="center", fontsize=12
    )
    figure.text(
        0.745, column_header_y, gemc_label, ha="center", va="center", fontsize=12
    )

    samples = ((data_cuts, data_label), (gemc_cuts, gemc_label))
    positions = tuple(_retained_topology_positions(cuts) for cuts, _ in samples)
    page_domain = _shared_variable_domain(
        data_cuts,
        positions[0],
        gemc_cuts,
        positions[1],
        topologies,
        variable_index,
    )
    legend_entries = {}
    shared_x_axis = None
    for row, topology in enumerate(topologies):
        for column, ((cuts, sample_label), position_map) in enumerate(
            zip(samples, positions)
        ):
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=(3.75, 1.55), wspace=0.045
            )
            plot_axis = figure.add_subplot(inner[0, 0], sharex=shared_x_axis)
            if shared_x_axis is None:
                shared_x_axis = plot_axis
            information_grid = inner[0, 1].subgridspec(
                3, 1, height_ratios=(1.05, 1.55, 0.9), hspace=0.11
            )
            text_axes = tuple(
                figure.add_subplot(information_grid[index, 0])
                for index in range(3)
            )
            position = position_map.get(topology)
            topology_label = _topology_label(topology)
            if position is None:
                _draw_missing_comparison_panel(
                    plot_axis,
                    text_axes,
                    cuts,
                    topology,
                    variable,
                    sample_label,
                    page_domain,
                )
                continue
            handles = _draw_comparison_fit_panel(
                plot_axis,
                cuts,
                position,
                variable_index,
                variable,
                topology_label,
                page_domain,
            )
            _draw_comparison_metadata(
                text_axes, cuts, position, variable_index, variable
            )
            for handle, label in handles:
                legend_entries.setdefault(label, handle)

    if legend_entries:
        figure.legend(
            legend_entries.values(),
            legend_entries.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=min(len(legend_entries), 7),
            frameon=True,
            facecolor="white",
            edgecolor="#606060",
        )
    figure.text(
        0.5,
        0.16 / figure_height,
        f"Shared {variable} range: [{page_domain[0]:.6g}, {page_domain[1]:.6g}]",
        ha="center",
        va="bottom",
        color="#404040",
        fontsize=8,
    )
    pdf.savefig(figure, facecolor="white", edgecolor="white", transparent=False)
    plt.close(figure)


def _retained_topology_positions(cuts: ExclusivityCuts) -> dict[int, int]:
    topologies = topology_ids_from_groups(cuts.group_ids, cuts.global_mode)
    if np.unique(topologies).size != topologies.size:
        raise ValueError(
            "paired topology diagnostics require at most one retained group per topology"
        )
    return {int(topology): index for index, topology in enumerate(topologies)}


def _shared_variable_domain(
    data_cuts,
    data_positions,
    gemc_cuts,
    gemc_positions,
    topologies,
    variable_index,
) -> tuple[float, float]:
    bounds = []
    for cuts, positions in (
        (data_cuts, data_positions),
        (gemc_cuts, gemc_positions),
    ):
        for topology in topologies:
            position = positions.get(topology)
            if position is None:
                continue
            bounds.append(
                (
                    float(cuts.fit_lower[position, variable_index]),
                    float(cuts.fit_upper[position, variable_index]),
                )
            )
    if not bounds:
        raise ValueError("no retained fits are available for the shared x range")
    return min(item[0] for item in bounds), max(item[1] for item in bounds)


def _topology_label(topology: int) -> str:
    proton_detector = topology // 4
    ft_photons = topology % 4
    proton_name = {1: "FD", 2: "CD"}.get(proton_detector, "other")
    photon_name = {0: "FD/FD", 1: "FD/FT", 2: "FT/FT"}.get(
        ft_photons, f"FT count {ft_photons}"
    )
    return (
        f"proton {proton_name} (pDet={proton_detector}); "
        f"photons {photon_name}"
    )


def _draw_comparison_fit_panel(
    axis,
    cuts,
    position,
    variable_index,
    variable,
    topology_label,
    domain,
):
    count = int(cuts.histogram_bin_count[position, variable_index])
    edges = cuts.histogram_edges[position, variable_index, : count + 1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed = cuts.observed_counts[position, variable_index, :count]
    expected = cuts.expected_counts[position, variable_index, :count]
    cut_signal = cuts.cut_signal_counts[position, variable_index, :count]
    noncut = cuts.noncut_component_counts[position, variable_index, :count]
    background = cuts.background_counts[position, variable_index, :count]
    axis.errorbar(
        centers,
        observed,
        yerr=np.sqrt(np.maximum(observed, 1.0)),
        fmt=".",
        color=_OBSERVED_COLOR,
        ecolor="#606060",
        markersize=2.8,
        linewidth=0.65,
        alpha=0.9,
        label="observed",
    )
    axis.plot(centers, expected, color=_TOTAL_COLOR, linewidth=1.7, label="total fit")
    axis.plot(
        centers,
        cut_signal,
        color=_CUT_COMPONENT_COLOR,
        linewidth=1.5,
        label="cut component",
    )
    if np.any(noncut > 0.0):
        axis.plot(
            centers,
            noncut,
            color=_NUISANCE_COLOR,
            linewidth=1.4,
            label="fitted nuisance/tail",
        )
    axis.plot(
        centers,
        background,
        color=_BACKGROUND_COLOR,
        linewidth=1.3,
        label="background",
    )
    axis.axvline(
        cuts.lower[position, variable_index],
        color=_BOUNDARY_COLOR,
        linewidth=1.4,
        linestyle="--",
        label="nominal cut",
    )
    axis.axvline(
        cuts.upper[position, variable_index],
        color=_BOUNDARY_COLOR,
        linewidth=1.4,
        linestyle="--",
    )
    if cuts.nminus1_audit_success[position, variable_index]:
        axis.axvline(
            cuts.nminus1_audit_lower[position, variable_index],
            color=_AUDIT_BOUNDARY_COLOR,
            linewidth=1.3,
            linestyle=":",
            label="fixed N-1 audit proposal",
        )
        axis.axvline(
            cuts.nminus1_audit_upper[position, variable_index],
            color=_AUDIT_BOUNDARY_COLOR,
            linewidth=1.3,
            linestyle=":",
        )
    if domain is not None:
        axis.set_xlim(*domain)
    axis.set_ylim(bottom=0.0)
    axis.set_title(topology_label, fontsize=9, color="black")
    axis.set_xlabel(variable, color="black")
    axis.set_ylabel("entries / bin", color="black")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    _style_axis(axis)
    return tuple(zip(*axis.get_legend_handles_labels()))


def _draw_comparison_metadata(axes, cuts, position, variable_index, variable) -> None:
    import textwrap

    metadata_axis, quality_axis, cut_axis = axes
    _style_information_axis(metadata_axis, "#F2F2F2", "#707070")
    _style_information_axis(quality_axis, "#EDF4FA", "#557A95")
    _style_information_axis(cut_axis, "#E9F6F1", _BOUNDARY_COLOR)
    reduced = cuts.pearson_chi2[position, variable_index] / max(
        cuts.fit_ndof[position, variable_index], 1
    )
    nminus1_efficiency = (
        cuts.nminus1_passing[position, variable_index]
        / cuts.nminus1_entries[position, variable_index]
        if cuts.nminus1_entries[position, variable_index]
        else float("nan")
    )
    model = textwrap.fill(
        str(cuts.fit_model[position, variable_index]),
        width=25,
        subsequent_indent="  ",
    )
    metadata_axis.text(
        0.04,
        0.9,
        "FIT METADATA",
        transform=metadata_axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        fontweight="bold",
        color="#303030",
    )
    metadata_axis.text(
        0.04,
        0.67,
        f"source: {cuts.window_source[position, variable_index]}\n"
        f"model: {model}\n"
        f"domain [{cuts.fit_lower[position, variable_index]:.5g}, "
        f"{cuts.fit_upper[position, variable_index]:.5g}]",
        transform=metadata_axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.1,
        linespacing=1.08,
        color="black",
    )
    audit_text = _audit_summary(cuts, position, variable_index)
    quality_axis.text(
        0.04,
        0.92,
        "FIT QUALITY",
        transform=quality_axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        fontweight="bold",
        color="#294C65",
    )
    quality_axis.text(
        0.04,
        0.76,
        f"chi2/ndof: {reduced:.4g}\n"
        f"deviance: {cuts.deviance[position, variable_index]:.4g}\n"
        f"delta BIC: {cuts.delta_bic[position, variable_index]:.4g}\n"
        "f(cut,nuis,bkg): "
        f"{cuts.cut_component_fractions[position, variable_index]:.3f}, "
        f"{cuts.nuisance_fractions[position, variable_index]:.3f}, "
        f"{cuts.background_fractions[position, variable_index]:.3f}\n"
        f"N-1 eff: {nminus1_efficiency:.4f}\n"
        f"{audit_text}",
        transform=quality_axis.transAxes,
        va="top",
        ha="left",
        fontsize=5.9,
        linespacing=1.05,
        color="black",
    )
    cut_axis.text(
        0.04,
        0.88,
        "NOMINAL CUT",
        transform=cut_axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        fontweight="bold",
        color="#006B4F",
    )
    cut_axis.text(
        0.04,
        0.59,
        f"[{cuts.lower[position, variable_index]:.6g}, "
        f"{cuts.upper[position, variable_index]:.6g}]",
        transform=cut_axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.3,
        linespacing=1.08,
        color="#006B4F",
    )


def _style_information_axis(axis, facecolor: str, edgecolor: str) -> None:
    axis.set_facecolor(facecolor)
    axis.patch.set_alpha(1.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(edgecolor)
        spine.set_linewidth(0.65)


def _audit_summary(cuts, position, variable_index) -> str:
    if not cuts.nminus1_audit_success[position, variable_index]:
        reason = str(cuts.nminus1_audit_reasons[position, variable_index])
        return f"audit failed: {reason[:42]}"
    return (
        "audit "
        f"[{cuts.nminus1_audit_lower[position, variable_index]:.4g}, "
        f"{cuts.nminus1_audit_upper[position, variable_index]:.4g}]"
    )


def _draw_missing_comparison_panel(
    plot_axis,
    text_axes,
    cuts,
    topology,
    variable,
    sample_label,
    domain,
) -> None:
    if domain is not None:
        plot_axis.set_xlim(*domain)
    plot_axis.set_ylim(0.0, 1.0)
    plot_axis.set_title(_topology_label(topology), fontsize=9, color="black")
    plot_axis.set_xlabel(variable, color="black")
    plot_axis.set_ylabel("entries / bin", color="black")
    plot_axis.text(
        0.5,
        0.5,
        f"No retained {sample_label} fit",
        transform=plot_axis.transAxes,
        ha="center",
        va="center",
        color="#606060",
        fontsize=9,
    )
    _style_axis(plot_axis)
    _style_information_axis(text_axes[0], "#FBEDED", "#B35C5C")
    for text_axis in text_axes[1:]:
        text_axis.set_visible(False)
    positions = [axis.get_position() for axis in text_axes]
    left = min(position.x0 for position in positions)
    bottom = min(position.y0 for position in positions)
    right = max(position.x1 for position in positions)
    top = max(position.y1 for position in positions)
    text_axes[0].set_position((left, bottom, right - left, top - bottom))
    reason = _dropped_topology_reason(cuts, topology)
    text_axes[0].text(
        0.04,
        0.9,
        "TOPOLOGY NOT RETAINED\n" + reason,
        transform=text_axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="#8B1A1A",
    )


def _dropped_topology_reason(cuts, topology: int) -> str:
    import textwrap

    dropped = topology_ids_from_groups(cuts.dropped_group_ids, cuts.global_mode)
    matches = np.flatnonzero(dropped == topology)
    if not matches.size:
        return "not populated in this sample"
    index = int(matches[0])
    return textwrap.fill(
        f"failed at {cuts.dropped_variables[index]}: "
        f"{str(cuts.dropped_reasons[index])[:120]}",
        width=34,
    )


def _summary_page(cuts, selected_ids, pdf, plt) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        constrained_layout=True,
        facecolor="white",
    )
    figure.patch.set_alpha(1.0)
    figure.suptitle(
        "Exclusivity fit audit\n"
        f"{cuts.estimator}; groups={cuts.group_ids.size}; plotted={selected_ids.size}; "
        f"fixed N-1 audit complete={cuts.nminus1_audit_complete}; "
        f"within {cuts.nminus1_audit_boundary_tolerance:.3g} tolerance="
        f"{cuts.nminus1_audit_within_tolerance}; maximum boundary change="
        f"{cuts.nminus1_audit_maximum_boundary_change:.3g}",
        color="black",
        fontsize=12,
    )
    for index, (axis, name) in enumerate(zip(axes.flat, cuts.variables, strict=True)):
        reduced = cuts.pearson_chi2[:, index] / np.maximum(cuts.fit_ndof[:, index], 1)
        efficiency = np.divide(
            cuts.nminus1_passing[:, index],
            cuts.nminus1_entries[:, index],
            out=np.full(cuts.group_ids.size, np.nan),
            where=cuts.nminus1_entries[:, index] > 0,
        )
        axis.scatter(
            reduced,
            efficiency,
            s=28,
            color=_CUT_COMPONENT_COLOR,
            edgecolor=_OBSERVED_COLOR,
            linewidth=0.4,
            alpha=1.0,
            zorder=3,
        )
        for group_id, x_value, y_value in zip(
            cuts.group_ids, reduced, efficiency, strict=True
        ):
            if group_id in selected_ids and np.isfinite(x_value + y_value):
                axis.annotate(
                    str(int(group_id)),
                    (x_value, y_value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="black",
                    fontsize=7,
                )
        axis.set_title(name, color="black")
        axis.set_xlabel("Pearson chi-square / ndof", color="black")
        axis.set_ylabel("N-1 cut efficiency", color="black")
        axis.set_ylim(-0.03, 1.03)
        _style_axis(axis)
    pdf.savefig(
        figure,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(figure)


def _group_page(cuts, group_id, position, pdf, plt) -> None:
    topology = int(
        topology_ids_from_groups(np.asarray([group_id]), cuts.global_mode)[0]
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        constrained_layout=False,
        facecolor="white",
    )
    figure.patch.set_alpha(1.0)
    figure.suptitle(
        f"group {group_id}: proton detector={topology // 4}, "
        f"FT photons={topology % 4}",
        color="black",
        fontsize=12,
    )
    legend_entries = {}
    for variable_index, (axis, name) in enumerate(
        zip(axes.flat, cuts.variables, strict=True)
    ):
        count = int(cuts.histogram_bin_count[position, variable_index])
        edges = cuts.histogram_edges[position, variable_index, : count + 1]
        centers = 0.5 * (edges[:-1] + edges[1:])
        observed = cuts.observed_counts[position, variable_index, :count]
        expected = cuts.expected_counts[position, variable_index, :count]
        cut_signal = cuts.cut_signal_counts[position, variable_index, :count]
        noncut = cuts.noncut_component_counts[position, variable_index, :count]
        background = cuts.background_counts[position, variable_index, :count]

        axis.errorbar(
            centers,
            observed,
            yerr=np.sqrt(np.maximum(observed, 1.0)),
            fmt=".",
            color=_OBSERVED_COLOR,
            ecolor="#606060",
            markersize=3.0,
            linewidth=0.7,
            alpha=0.9,
            label="observed",
        )
        axis.plot(
            centers,
            expected,
            color=_TOTAL_COLOR,
            linewidth=1.7,
            label="total fit",
        )
        axis.plot(
            centers,
            cut_signal,
            color=_CUT_COMPONENT_COLOR,
            linewidth=1.5,
            label="cut component",
        )
        if np.any(noncut > 0.0):
            axis.plot(
                centers,
                noncut,
                color=_NUISANCE_COLOR,
                linewidth=1.5,
                label="fitted nuisance/tail",
            )
        axis.plot(
            centers,
            background,
            color=_BACKGROUND_COLOR,
            linewidth=1.3,
            label="background",
        )
        axis.axvline(
            cuts.lower[position, variable_index],
            color=_BOUNDARY_COLOR,
            linewidth=1.4,
            linestyle="--",
            label="cut boundary",
        )
        axis.axvline(
            cuts.upper[position, variable_index],
            color=_BOUNDARY_COLOR,
            linewidth=1.4,
            linestyle="--",
        )
        if cuts.nminus1_audit_success[position, variable_index]:
            axis.axvline(
                cuts.nminus1_audit_lower[position, variable_index],
                color=_AUDIT_BOUNDARY_COLOR,
                linewidth=1.4,
                linestyle=":",
                label="fixed N-1 audit proposal",
            )
            axis.axvline(
                cuts.nminus1_audit_upper[position, variable_index],
                color=_AUDIT_BOUNDARY_COLOR,
                linewidth=1.4,
                linestyle=":",
            )
        reduced = cuts.pearson_chi2[position, variable_index] / max(
            cuts.fit_ndof[position, variable_index], 1
        )
        nminus1_efficiency = (
            cuts.nminus1_passing[position, variable_index]
            / cuts.nminus1_entries[position, variable_index]
            if cuts.nminus1_entries[position, variable_index]
            else float("nan")
        )
        if cuts.nminus1_audit_success[position, variable_index]:
            nominal_width = (
                cuts.upper[position, variable_index]
                - cuts.lower[position, variable_index]
            )
            audit_width = (
                cuts.nminus1_audit_upper[position, variable_index]
                - cuts.nminus1_audit_lower[position, variable_index]
            )
            denominator = max(
                abs(nominal_width), abs(audit_width), np.finfo(float).eps
            )
            audit_change = max(
                abs(
                    cuts.nminus1_audit_lower[position, variable_index]
                    - cuts.lower[position, variable_index]
                )
                / denominator,
                abs(
                    cuts.nminus1_audit_upper[position, variable_index]
                    - cuts.upper[position, variable_index]
                )
                / denominator,
            )
            audit_text = (
                f"N-1 audit change: {audit_change:.3g}; "
                f"{cuts.nminus1_audit_source[position, variable_index]}"
            )
        else:
            audit_reason = str(
                cuts.nminus1_audit_reasons[position, variable_index]
            )
            audit_text = (
                "N-1 audit failed: "
                f"{audit_reason[:72]}"
            )
        axis.set_title(
            f"{name}\n{cuts.fit_model[position, variable_index]}",
            color="black",
        )
        axis.set_xlabel(
            f"cut={cuts.cut_components[variable_index]}, "
            f"containment={cuts.cut_containments[variable_index]:.5f}",
            color="black",
        )
        axis.set_ylabel("entries / bin", color="black")
        axis.text(
            0.02,
            0.97,
            f"source: {cuts.window_source[position, variable_index]}\n"
            f"fit domain: [{cuts.fit_lower[position, variable_index]:.4g}, "
            f"{cuts.fit_upper[position, variable_index]:.4g}]\n"
            f"chi2/ndof: {reduced:.3g}; deviance: "
            f"{cuts.deviance[position, variable_index]:.3g}\n"
            f"delta BIC: {cuts.delta_bic[position, variable_index]:.3g}\n"
            f"fractions cut/nuis./bkg: "
            f"{cuts.cut_component_fractions[position, variable_index]:.3f}/"
            f"{cuts.nuisance_fractions[position, variable_index]:.3f}/"
            f"{cuts.background_fractions[position, variable_index]:.3f}\n"
            f"N-1 efficiency: {nminus1_efficiency:.4f}\n"
            f"{audit_text}",
            transform=axis.transAxes,
            va="top",
            color="black",
            fontsize=7,
            zorder=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.72,
                "edgecolor": "#808080",
                "linewidth": 0.4,
                "boxstyle": "square,pad=0.25",
            },
        )
        axis.text(
            0.98,
            0.02,
            f"nominal cut: [{cuts.lower[position, variable_index]:.6g}, "
            f"{cuts.upper[position, variable_index]:.6g}]",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#006B4F",
            fontsize=7,
            zorder=11,
            bbox={
                "facecolor": "white",
                "alpha": 0.72,
                "edgecolor": _BOUNDARY_COLOR,
                "linewidth": 0.5,
                "boxstyle": "round,pad=0.22",
            },
        )
        axis.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
        _style_axis(axis)
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            legend_entries.setdefault(label, handle)
    figure.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="center left",
        bbox_to_anchor=(0.885, 0.5),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#606060",
        labelcolor="black",
    )
    figure.tight_layout(rect=(0.0, 0.025, 0.875, 0.94), h_pad=1.5, w_pad=1.2)
    pdf.savefig(
        figure,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(figure)


def _style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.patch.set_alpha(1.0)
    axis.tick_params(axis="both", colors="black", labelcolor="black")
    axis.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color("#202020")
        spine.set_linewidth(0.8)
