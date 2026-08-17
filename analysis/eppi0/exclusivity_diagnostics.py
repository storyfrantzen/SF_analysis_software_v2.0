from __future__ import annotations

from pathlib import Path

import numpy as np

from .exclusivity import ExclusivityCuts, topology_ids_from_groups


Array = np.ndarray


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

    with PdfPages(output_path) as pdf:
        _summary_page(cuts, selected_ids, pdf, plt)
        for group_id, position in zip(selected_ids, positions, strict=True):
            _group_page(cuts, int(group_id), int(position), pdf, plt)
    return tuple(int(item) for item in selected_ids)


def _summary_page(cuts, selected_ids, pdf, plt) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    figure.suptitle(
        "Exclusivity fit audit\n"
        f"{cuts.estimator}; groups={cuts.group_ids.size}; plotted={selected_ids.size}; "
        f"N-1 iterations={cuts.refinement_iterations}; "
        f"converged={cuts.refinement_converged}; "
        f"max boundary change={cuts.maximum_boundary_change:.3g}\n"
        "boundary-change history="
        + ", ".join(f"{value:.3g}" for value in cuts.boundary_change_history)
    )
    for index, (axis, name) in enumerate(zip(axes.flat, cuts.variables, strict=True)):
        reduced = cuts.pearson_chi2[:, index] / np.maximum(cuts.fit_ndof[:, index], 1)
        efficiency = np.divide(
            cuts.nminus1_passing[:, index],
            cuts.nminus1_entries[:, index],
            out=np.full(cuts.group_ids.size, np.nan),
            where=cuts.nminus1_entries[:, index] > 0,
        )
        axis.scatter(reduced, efficiency, s=18, alpha=0.75)
        for group_id, x_value, y_value in zip(
            cuts.group_ids, reduced, efficiency, strict=True
        ):
            if group_id in selected_ids and np.isfinite(x_value + y_value):
                axis.annotate(str(int(group_id)), (x_value, y_value), fontsize=6)
        axis.set_title(name)
        axis.set_xlabel("Pearson chi-square / ndof")
        axis.set_ylabel("N-1 cut efficiency")
        axis.grid(alpha=0.25)
    pdf.savefig(figure)
    plt.close(figure)


def _group_page(cuts, group_id, position, pdf, plt) -> None:
    topology = int(
        topology_ids_from_groups(np.asarray([group_id]), cuts.global_mode)[0]
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    figure.suptitle(
        f"group {group_id}: proton detector={topology // 4}, "
        f"FT photons={topology % 4}"
    )
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
            color="black",
            markersize=2.5,
            linewidth=0.6,
            label="observed",
        )
        axis.plot(centers, expected, color="black", linewidth=1.2, label="total fit")
        axis.plot(centers, cut_signal, color="tab:blue", label="cut component")
        if np.any(noncut > 0.0):
            axis.plot(centers, noncut, color="tab:orange", label="fitted nuisance/tail")
        axis.plot(centers, background, color="tab:red", label="background")
        axis.axvline(cuts.lower[position, variable_index], color="tab:green", linestyle="--")
        axis.axvline(cuts.upper[position, variable_index], color="tab:green", linestyle="--")
        reduced = cuts.pearson_chi2[position, variable_index] / max(
            cuts.fit_ndof[position, variable_index], 1
        )
        nminus1_efficiency = (
            cuts.nminus1_passing[position, variable_index]
            / cuts.nminus1_entries[position, variable_index]
            if cuts.nminus1_entries[position, variable_index]
            else float("nan")
        )
        axis.set_title(name)
        axis.set_xlabel(
            f"{cuts.fit_model[position, variable_index]}\n"
            f"cut={cuts.cut_components[variable_index]}, "
            f"containment={cuts.cut_containments[variable_index]:.5f}"
        )
        axis.set_ylabel("entries / bin")
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
            f"N-1 efficiency: {nminus1_efficiency:.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
    )
    pdf.savefig(figure)
    plt.close(figure)
