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
                "alpha": 0.92,
                "edgecolor": "#808080",
                "linewidth": 0.4,
                "boxstyle": "square,pad=0.25",
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
