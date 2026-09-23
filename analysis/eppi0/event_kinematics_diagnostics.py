from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np

from .topology import detector_topology_id, ft_photon_count


Array = np.ndarray

TOPOLOGY_LABELS = {
    4: "pFD_fdfd",
    5: "pFD_fdft",
    6: "pFD_ftft",
    8: "pCD_fdfd",
    9: "pCD_fdft",
    10: "pCD_ftft",
}

TOPOLOGY_TITLES = {
    4: "FD proton, FD/FD photons",
    5: "FD proton, FD/FT photons",
    6: "FD proton, FT/FT photons",
    8: "CD proton, FD/FD photons",
    9: "CD proton, FD/FT photons",
    10: "CD proton, FT/FT photons",
}


def identity(values: Array) -> Array:
    return np.asarray(values, dtype=float)


def degrees(values: Array) -> Array:
    return np.rad2deg(np.asarray(values, dtype=float))


def trento_degrees(values: Array) -> Array:
    return np.mod(degrees(values), 360.0)


@dataclass(frozen=True)
class PlotVariable:
    branch: str
    title: str
    x_label: str
    section: str
    transform: Callable[[Array], Array] = identity
    fixed_range: tuple[float, float] | None = None
    bins: int = 64


VARIABLES: tuple[PlotVariable, ...] = (
    PlotVariable("Q2", r"$Q^2$", r"$Q^2$ [GeV$^2$]", "DIS"),
    PlotVariable("xB", r"$x_B$", r"$x_B$", "DIS", fixed_range=(0.0, 0.75)),
    PlotVariable("W", r"$W$", r"$W$ [GeV]", "DIS"),
    PlotVariable("y", r"$y$", r"$y$", "DIS", fixed_range=(0.0, 1.0)),
    PlotVariable("nu", r"Energy transfer $\nu$", r"$\nu$ [GeV]", "DIS"),
    PlotVariable("t", r"Momentum transfer $-t$", r"$-t$ [GeV$^2$]", "DIS"),
    PlotVariable("trentoPhi", r"Trento $\phi$", r"$\phi$ [deg]", "particle angles", transform=trento_degrees, fixed_range=(0.0, 360.0), bins=72),
    PlotVariable("electronP", "Electron momentum", r"$p_e$ [GeV]", "particle momenta"),
    PlotVariable("electronTheta", "Electron polar angle", r"$\theta_e$ [deg]", "particle angles", transform=degrees),
    PlotVariable("electronPhi", "Electron azimuth", r"$\phi_e$ [deg]", "particle angles", transform=degrees, fixed_range=(-180.0, 180.0), bins=72),
    PlotVariable("protonP", "Proton momentum", r"$p_p$ [GeV]", "particle momenta"),
    PlotVariable("protonTheta", "Proton polar angle", r"$\theta_p$ [deg]", "particle angles", transform=degrees),
    PlotVariable("protonPhi", "Proton azimuth", r"$\phi_p$ [deg]", "particle angles", transform=degrees, fixed_range=(-180.0, 180.0), bins=72),
    PlotVariable("gamma1P", "Photon 1 momentum", r"$p_{\gamma_1}$ [GeV]", "particle momenta"),
    PlotVariable("gamma1Theta", "Photon 1 polar angle", r"$\theta_{\gamma_1}$ [deg]", "particle angles", transform=degrees),
    PlotVariable("gamma1Phi", "Photon 1 azimuth", r"$\phi_{\gamma_1}$ [deg]", "particle angles", transform=degrees, fixed_range=(-180.0, 180.0), bins=72),
    PlotVariable("gamma2P", "Photon 2 momentum", r"$p_{\gamma_2}$ [GeV]", "particle momenta"),
    PlotVariable("gamma2Theta", "Photon 2 polar angle", r"$\theta_{\gamma_2}$ [deg]", "particle angles", transform=degrees),
    PlotVariable("gamma2Phi", "Photon 2 azimuth", r"$\phi_{\gamma_2}$ [deg]", "particle angles", transform=degrees, fixed_range=(-180.0, 180.0), bins=72),
    PlotVariable("pi0_p", r"$\pi^0$ momentum", r"$p_{\pi^0}$ [GeV]", "particle momenta"),
    PlotVariable("pi0_theta", r"$\pi^0$ polar angle", r"$\theta_{\pi^0}$ [deg]", "particle angles", transform=degrees),
    PlotVariable("pi0_phi", r"$\pi^0$ azimuth", r"$\phi_{\pi^0}$ [deg]", "particle angles", transform=degrees, fixed_range=(-180.0, 180.0), bins=72),
    PlotVariable("t_pi0", r"$\pi^0$-side momentum transfer", r"$-t_{\pi^0}$ [GeV$^2$]", "exclusivity"),
    PlotVariable("m_gg", r"Diphoton mass", r"$m_{\gamma\gamma}$ [GeV]", "exclusivity"),
    PlotVariable("m2_miss", r"Exclusive missing mass squared", r"$M_X^2(ep\pi^0X)$ [GeV$^2$]", "exclusivity"),
    PlotVariable("m2_epX", r"Missing $epX$ mass squared", r"$M_X^2(epX)$ [GeV$^2$]", "exclusivity"),
    PlotVariable("m2_epi0X", r"Missing $e\pi^0X$ mass squared", r"$M_X^2(e\pi^0X)$ [GeV$^2$]", "exclusivity"),
    PlotVariable("m_eggX", r"Missing $e\gamma\gamma X$ mass", r"$M_X(e\gamma\gamma X)$ [GeV]", "exclusivity"),
    PlotVariable("E_miss", "Missing energy", r"$E_{miss}$ [GeV]", "exclusivity"),
    PlotVariable("pT_miss", "Missing transverse momentum", r"$p_{T,miss}$ [GeV]", "exclusivity"),
    PlotVariable("theta_e_g1", r"Electron-photon 1 opening angle", r"$\theta_{e\gamma_1}$ [deg]", "exclusivity", transform=degrees),
    PlotVariable("theta_e_g2", r"Electron-photon 2 opening angle", r"$\theta_{e\gamma_2}$ [deg]", "exclusivity", transform=degrees),
    PlotVariable("theta_g1_g2", r"Diphoton opening angle", r"$\theta_{\gamma_1\gamma_2}$ [deg]", "exclusivity", transform=degrees),
    PlotVariable("pi0_deltaPhi", r"$\pi^0$ missing-system $\Delta\phi$", r"$\Delta\phi$ [deg]", "exclusivity", transform=degrees),
    PlotVariable("pi0_thetaX", r"$\pi^0$ missing-system opening angle", r"$\theta_{\pi^0X}$ [deg]", "exclusivity", transform=degrees),
)


SECTIONS = ("DIS", "particle momenta", "particle angles", "exclusivity")

SECTION_COLORS = {
    "DIS": "#0369a1",
    "particle momenta": "#047857",
    "particle angles": "#7c3aed",
    "exclusivity": "#b45309",
}


@dataclass(frozen=True)
class Correlation:
    x: str
    y: str
    title: str


CORRELATIONS: tuple[Correlation, ...] = (
    Correlation("xB", "Q2", r"DIS coverage: $Q^2$ vs $x_B$"),
    Correlation("xB", "t", r"Production coverage: $-t$ vs $x_B$"),
    Correlation("Q2", "W", r"DIS correlation: $W$ vs $Q^2$"),
    Correlation("electronP", "electronTheta", "Electron angle vs momentum"),
    Correlation("protonP", "protonTheta", "Proton angle vs momentum"),
    Correlation("pT_miss", "m_gg", r"$m_{\gamma\gamma}$ vs missing $p_T$"),
)


@dataclass(frozen=True)
class DetectorMap:
    key: str
    title: str
    x_branches: tuple[str, ...]
    y_branches: tuple[str, ...]
    x_label: str
    y_label: str
    x_transform: Callable[[Array], Array] = identity
    y_transform: Callable[[Array], Array] = identity
    fixed_x_range: tuple[float, float] | None = None
    fixed_y_range: tuple[float, float] | None = None
    common_coordinate_range: bool = False
    equal_aspect: bool = False
    bins: int = 80

    def __post_init__(self) -> None:
        if len(self.x_branches) != len(self.y_branches):
            raise ValueError(f"detector map {self.key} has unpaired coordinates")


DETECTOR_MAPS: tuple[DetectorMap, ...] = (
    DetectorMap(
        "electron_pcal_xy", "Electron PCAL global occupancy",
        ("electronXPCAL",), ("electronYPCAL",), "PCAL x [cm]", "PCAL y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "photon_pcal_xy", "FD photon PCAL global occupancy",
        ("gamma1XPCAL", "gamma2XPCAL"),
        ("gamma1YPCAL", "gamma2YPCAL"),
        "PCAL x [cm]", "PCAL y [cm]", common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_pcal_uv", "Electron PCAL local occupancy",
        ("electronUPCAL",), ("electronVPCAL",), "PCAL u [cm]", "PCAL v [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "photon_pcal_uv", "FD photon PCAL local occupancy",
        ("gamma1UPCAL", "gamma2UPCAL"),
        ("gamma1VPCAL", "gamma2VPCAL"),
        "PCAL u [cm]", "PCAL v [cm]", common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_ecin_uv", "Electron ECIN local occupancy",
        ("electronUECIN",), ("electronVECIN",), "ECIN u [cm]", "ECIN v [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_ecout_uv", "Electron ECOUT local occupancy",
        ("electronUECOUT",), ("electronVECOUT",), "ECOUT u [cm]", "ECOUT v [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "photon_ecin_uv", "FD photon ECIN local occupancy",
        ("gamma1UECIN", "gamma2UECIN"),
        ("gamma1VECIN", "gamma2VECIN"),
        "ECIN u [cm]", "ECIN v [cm]", common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "photon_ecout_uv", "FD photon ECOUT local occupancy",
        ("gamma1UECOUT", "gamma2UECOUT"),
        ("gamma1VECOUT", "gamma2VECOUT"),
        "ECOUT u [cm]", "ECOUT v [cm]", common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_dc_r1", "Electron DC region 1 occupancy",
        ("electronXDC1",), ("electronYDC1",), "DC R1 x [cm]", "DC R1 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_dc_r2", "Electron DC region 2 occupancy",
        ("electronXDC2",), ("electronYDC2",), "DC R2 x [cm]", "DC R2 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "electron_dc_r3", "Electron DC region 3 occupancy",
        ("electronXDC3",), ("electronYDC3",), "DC R3 x [cm]", "DC R3 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "photon_ftcal_xy", "FT photon FTCAL occupancy",
        ("gamma1XFT", "gamma2XFT"), ("gamma1YFT", "gamma2YFT"),
        "FTCAL x [cm]", "FTCAL y [cm]", common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "proton_dc_r1", "FD proton DC region 1 occupancy",
        ("protonXDC1",), ("protonYDC1",), "DC R1 x [cm]", "DC R1 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "proton_dc_r2", "FD proton DC region 2 occupancy",
        ("protonXDC2",), ("protonYDC2",), "DC R2 x [cm]", "DC R2 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "proton_dc_r3", "FD proton DC region 3 occupancy",
        ("protonXDC3",), ("protonYDC3",), "DC R3 x [cm]", "DC R3 y [cm]",
        common_coordinate_range=True, equal_aspect=True,
    ),
    DetectorMap(
        "proton_cvt_angles", "CD proton CVT layer-1 direction occupancy",
        ("protonPhiCVT",), ("protonThetaCVT",),
        r"CVT $\phi$ [deg]", r"CVT $\theta$ [deg]",
        x_transform=degrees, y_transform=degrees,
        fixed_x_range=(-180.0, 180.0), fixed_y_range=(0.0, 180.0), bins=90,
    ),
)


def detector_map_branches() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            branch
            for detector_map in DETECTOR_MAPS
            for branch in detector_map.x_branches + detector_map.y_branches
        )
    )


def available_detector_maps(arrays: Mapping[str, Array]) -> tuple[DetectorMap, ...]:
    return tuple(
        detector_map
        for detector_map in DETECTOR_MAPS
        if all(
            branch in arrays
            for branch in detector_map.x_branches + detector_map.y_branches
        )
    )


def detector_map_values(
    detector_map: DetectorMap, arrays: Mapping[str, Array], mask: Array
) -> tuple[Array, Array]:
    x_parts: list[Array] = []
    y_parts: list[Array] = []
    selected = np.asarray(mask, dtype=bool)
    for x_branch, y_branch in zip(
        detector_map.x_branches, detector_map.y_branches, strict=True
    ):
        x = detector_map.x_transform(np.asarray(arrays[x_branch])[selected])
        y = detector_map.y_transform(np.asarray(arrays[y_branch])[selected])
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            x_parts.append(np.asarray(x[finite], dtype=float))
            y_parts.append(np.asarray(y[finite], dtype=float))
    if not x_parts:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.concatenate(x_parts), np.concatenate(y_parts)


def detector_map_ranges(
    detector_map: DetectorMap, x: Array, y: Array
) -> tuple[tuple[float, float], tuple[float, float]]:
    if detector_map.common_coordinate_range:
        common = robust_range(
            np.concatenate((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        )
        return common, common
    return (
        robust_range(x, detector_map.fixed_x_range),
        robust_range(y, detector_map.fixed_y_range),
    )


def reconstructed_topology(arrays: Mapping[str, Array]) -> Array:
    return detector_topology_id(
        arrays["pDet"], ft_photon_count(arrays["g1Det"], arrays["g2Det"])
    )


def finite_values(variable: PlotVariable, arrays: Mapping[str, Array], mask: Array) -> Array:
    values = variable.transform(np.asarray(arrays[variable.branch])[mask])
    return values[np.isfinite(values)]


def robust_range(values: Array, fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    if fixed is not None:
        return float(fixed[0]), float(fixed[1])
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0, 1.0
    if clean.size == 1:
        width = max(abs(float(clean[0])) * 0.1, 0.5)
        return float(clean[0] - width), float(clean[0] + width)
    low, high = np.quantile(clean, [0.0025, 0.9975])
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        width = max(abs(float(low)) * 0.1, 0.5)
        return float(low - width), float(high + width)
    padding = 0.05 * (high - low)
    return float(low - padding), float(high + padding)


def available_variables(arrays: Mapping[str, Array]) -> tuple[PlotVariable, ...]:
    return tuple(variable for variable in VARIABLES if variable.branch in arrays)


def observed_topologies(topology: Array, mask: Array) -> list[int]:
    return [
        int(value)
        for value in sorted(np.unique(np.asarray(topology)[np.asarray(mask, dtype=bool)]))
        if int(value) in TOPOLOGY_LABELS
    ]


def report_summary(
    arrays: Mapping[str, Array], selection_mask: Array, topology: Array
) -> dict[str, object]:
    selected = np.asarray(selection_mask, dtype=bool)
    ids, counts = np.unique(np.asarray(topology)[selected], return_counts=True)
    topology_counts = {str(int(key)): int(value) for key, value in zip(ids, counts, strict=True)}
    supported = np.isin(topology, tuple(TOPOLOGY_LABELS))
    detector_maps = available_detector_maps(arrays)
    return {
        "input_rows": int(selected.size),
        "selected_rows": int(np.count_nonzero(selected)),
        "supported_topology_rows": int(np.count_nonzero(selected & supported)),
        "unsupported_topology_rows": int(np.count_nonzero(selected & ~supported)),
        "topology_counts": topology_counts,
        "available_branches": sorted(arrays),
        "plotted_variables": [item.branch for item in available_variables(arrays)],
        "detector_occupancy_maps": [item.key for item in detector_maps],
    }


def render_report(
    output: Path,
    arrays: Mapping[str, Array],
    selection_mask: Array,
    topology: Array,
    *,
    label: str,
    provenance_lines: Sequence[str],
) -> int:
    selected = np.asarray(selection_mask, dtype=bool)
    variables = available_variables(arrays)
    ranges = {
        variable.branch: robust_range(
            finite_values(variable, arrays, selected), variable.fixed_range
        )
        for variable in variables
    }
    variable_by_name = {variable.branch: variable for variable in variables}
    topologies = observed_topologies(topology, selected)
    detector_maps = available_detector_maps(arrays)
    detector_ranges = {
        detector_map.key: detector_map_ranges(
            detector_map,
            *detector_map_values(detector_map, arrays, selected),
        )
        for detector_map in detector_maps
    }
    scopes: list[tuple[str, Array]] = [("Topology integrated", selected)]
    scopes.extend(
        (
            f"{TOPOLOGY_LABELS[group]}: {TOPOLOGY_TITLES[group]}",
            selected & (topology == group),
        )
        for group in topologies
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    pages = 0
    try:
        with PdfPages(temporary_output) as pdf:
            _title_page(pdf, label, selected, topology, topologies, provenance_lines)
            pages += 1
            if topologies:
                pages += _topology_overlays(
                    pdf, label, arrays, selected, topology, topologies, variables, ranges
                )
            for scope_name, scope_mask in scopes:
                pages += _scope_pages(
                    pdf,
                    label,
                    scope_name,
                    arrays,
                    scope_mask,
                    variables,
                    variable_by_name,
                    ranges,
                )
                pages += _detector_map_pages(
                    pdf,
                    label,
                    scope_name,
                    arrays,
                    scope_mask,
                    detector_maps,
                    detector_ranges,
                )
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return pages


def _title_page(
    pdf: PdfPages,
    label: str,
    selected: Array,
    topology: Array,
    topologies: Sequence[int],
    provenance_lines: Sequence[str],
) -> None:
    figure = plt.figure(figsize=(11.0, 8.5))
    figure.patch.set_facecolor("white")
    figure.text(0.07, 0.91, "Exclusive ep -> e'p'pi0 event kinematics", fontsize=23, weight="bold")
    figure.text(0.07, 0.855, label, fontsize=15, color="#374151")
    figure.text(
        0.07,
        0.79,
        f"Final selected candidates: {np.count_nonzero(selected):,}",
        fontsize=14,
        weight="bold",
    )
    y = 0.735
    for group in topologies:
        count = int(np.count_nonzero(selected & (topology == group)))
        fraction = count / max(int(np.count_nonzero(selected)), 1)
        figure.text(
            0.09,
            y,
            f"{TOPOLOGY_LABELS[group]:<10}  {count:>10,}  ({100.0 * fraction:5.2f}%)   {TOPOLOGY_TITLES[group]}",
            fontsize=11,
            family="monospace",
        )
        y -= 0.035
    supported = np.isin(topology, tuple(TOPOLOGY_LABELS))
    unsupported = int(np.count_nonzero(selected & ~supported))
    if unsupported:
        figure.text(
            0.09,
            y,
            f"unsupported   {unsupported:>10,}  (included only in integrated pages)",
            fontsize=11,
            family="monospace",
            color="#b91c1c",
        )
        y -= 0.035
    figure.text(0.07, y - 0.015, "Provenance", fontsize=13, weight="bold")
    y -= 0.06
    for line in provenance_lines:
        for wrapped in _wrap(line, 112):
            figure.text(0.09, y, wrapped, fontsize=8.5, family="monospace")
            y -= 0.027
    figure.text(
        0.07,
        0.06,
        "Each section uses the same axis range for the integrated and topology-specific pages. "
        "Automatic ranges exclude only the outer 0.25% at each end for display; event counts and the JSON audit use every finite selected value.",
        fontsize=9,
        color="#4b5563",
        wrap=True,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _topology_overlays(
    pdf: PdfPages,
    label: str,
    arrays: Mapping[str, Array],
    selected: Array,
    topology: Array,
    topologies: Sequence[int],
    variables: Sequence[PlotVariable],
    ranges: Mapping[str, tuple[float, float]],
) -> int:
    core_names = {
        "Q2", "xB", "W", "t", "trentoPhi", "electronP", "electronTheta",
        "protonP", "protonTheta", "gamma1P", "gamma2P", "pi0_p", "m_gg",
        "E_miss", "pT_miss",
    }
    core = [variable for variable in variables if variable.branch in core_names]
    pages = 0
    colors = plt.get_cmap("tab10")
    batches = _balanced_batches(core, 6)
    for page_index, batch in enumerate(batches, start=1):
        figure, axes = _panel_figure(len(batch), maximum_panels=6)
        for axis, variable in zip(axes.flat, batch, strict=False):
            lo, hi = ranges[variable.branch]
            edges = np.linspace(lo, hi, variable.bins + 1)
            for color_index, group in enumerate(topologies):
                values = finite_values(variable, arrays, selected & (topology == group))
                in_range = values[(values >= lo) & (values <= hi)]
                if in_range.size == 0:
                    continue
                axis.hist(
                    in_range,
                    bins=edges,
                    histtype="step",
                    density=True,
                    linewidth=1.35,
                    color=colors(color_index),
                    label=TOPOLOGY_LABELS[group],
                )
            _set_variable_title(axis, variable)
            axis.set_xlabel(variable.x_label)
            axis.set_ylabel("Unit-normalized density")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, frameon=False)
        _hide_unused(axes.flat, len(batch))
        figure.suptitle(
            f"{label}\nTopology shape comparison - page {page_index} of {len(batches)}",
            fontsize=14,
            weight="semibold",
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages


def _scope_pages(
    pdf: PdfPages,
    label: str,
    scope_name: str,
    arrays: Mapping[str, Array],
    mask: Array,
    variables: Sequence[PlotVariable],
    variable_by_name: Mapping[str, PlotVariable],
    ranges: Mapping[str, tuple[float, float]],
) -> int:
    pages = 0
    count = int(np.count_nonzero(mask))
    ordered_variables = [
        variable
        for section in SECTIONS
        for variable in variables
        if variable.section == section
    ]
    batches = _balanced_batches(ordered_variables, 6)
    for page_index, batch in enumerate(batches, start=1):
        figure, axes = _panel_figure(len(batch), maximum_panels=6)
        for axis, variable in zip(axes.flat, batch, strict=False):
            values = finite_values(variable, arrays, mask)
            lo, hi = ranges[variable.branch]
            in_range = values[(values >= lo) & (values <= hi)]
            axis.hist(
                in_range,
                bins=variable.bins,
                range=(lo, hi),
                color="#2563eb",
                alpha=0.82,
                linewidth=0.35,
                edgecolor="white",
            )
            _set_variable_title(axis, variable)
            axis.set_xlabel(variable.x_label)
            axis.set_ylabel("Candidates")
            axis.grid(alpha=0.2)
            if values.size:
                under = int(np.count_nonzero(values < lo))
                over = int(np.count_nonzero(values > hi))
                axis.text(
                    0.98,
                    0.96,
                    f"N={values.size:,}\nmedian={np.median(values):.4g}\noutside view={under + over:,}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
        _hide_unused(axes.flat, len(batch))
        section_names = ", ".join(dict.fromkeys(item.section for item in batch))
        figure.suptitle(
            f"{label}\n{scope_name} - kinematic distributions - "
            f"page {page_index} of {len(batches)}\n"
            f"{section_names}; N={count:,}",
            fontsize=13,
            weight="semibold",
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1

    correlations = [
        correlation
        for correlation in CORRELATIONS
        if correlation.x in variable_by_name and correlation.y in variable_by_name
    ]
    correlation_batches = _balanced_batches(correlations, 6)
    for page_index, batch in enumerate(correlation_batches, start=1):
        figure, axes = _panel_figure(len(batch), maximum_panels=6)
        for axis, correlation in zip(axes.flat, batch, strict=False):
            x_variable = variable_by_name[correlation.x]
            y_variable = variable_by_name[correlation.y]
            x = x_variable.transform(np.asarray(arrays[correlation.x])[mask])
            y = y_variable.transform(np.asarray(arrays[correlation.y])[mask])
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            x_range = ranges[correlation.x]
            y_range = ranges[correlation.y]
            view = (
                (x >= x_range[0]) & (x <= x_range[1])
                & (y >= y_range[0]) & (y <= y_range[1])
            )
            x, y = x[view], y[view]
            if x.size:
                counts, _, _ = np.histogram2d(
                    x, y, bins=55, range=(x_range, y_range)
                )
                positive = counts[counts > 0]
                norm = (
                    LogNorm(vmin=1.0, vmax=max(float(positive.max()), 1.01))
                    if positive.size
                    else None
                )
                image = axis.hist2d(
                    x,
                    y,
                    bins=55,
                    range=(x_range, y_range),
                    cmap="viridis",
                    norm=norm,
                )[3]
                figure.colorbar(image, ax=axis, pad=0.01, label="Candidates")
            axis.set_title(correlation.title, fontsize=10)
            axis.set_xlabel(x_variable.x_label)
            axis.set_ylabel(y_variable.x_label)
            axis.grid(alpha=0.12)
        _hide_unused(axes.flat, len(batch))
        figure.suptitle(
            f"{label}\n{scope_name} - correlations - "
            f"page {page_index} of {len(correlation_batches)}; N={count:,}",
            fontsize=14,
            weight="semibold",
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages


def _detector_map_pages(
    pdf: PdfPages,
    label: str,
    scope_name: str,
    arrays: Mapping[str, Array],
    mask: Array,
    detector_maps: Sequence[DetectorMap],
    ranges: Mapping[str, tuple[tuple[float, float], tuple[float, float]]],
) -> int:
    populated: list[tuple[DetectorMap, Array, Array]] = []
    for detector_map in detector_maps:
        x, y = detector_map_values(detector_map, arrays, mask)
        if x.size:
            populated.append((detector_map, x, y))

    pages = 0
    batches = _balanced_batches(populated, 4)
    for page_index, batch in enumerate(batches, start=1):
        figure, axes = _panel_figure(len(batch), maximum_panels=4)
        for axis, (detector_map, x, y) in zip(axes.flat, batch, strict=False):
            x_range, y_range = ranges[detector_map.key]
            view = (
                (x >= x_range[0]) & (x <= x_range[1])
                & (y >= y_range[0]) & (y <= y_range[1])
            )
            x_view = x[view]
            y_view = y[view]
            counts, _, _ = np.histogram2d(
                x_view,
                y_view,
                bins=detector_map.bins,
                range=(x_range, y_range),
            )
            positive = counts[counts > 0]
            norm = (
                LogNorm(vmin=1.0, vmax=max(float(positive.max()), 1.01))
                if positive.size
                else None
            )
            image = axis.hist2d(
                x_view,
                y_view,
                bins=detector_map.bins,
                range=(x_range, y_range),
                cmap="magma",
                norm=norm,
            )[3]
            figure.colorbar(image, ax=axis, pad=0.01, label="Candidate hits")
            axis.set_title(detector_map.title, fontsize=10)
            axis.set_xlabel(detector_map.x_label)
            axis.set_ylabel(detector_map.y_label)
            axis.grid(alpha=0.12)
            if detector_map.equal_aspect:
                axis.set_aspect("equal", adjustable="box")
            axis.text(
                0.98,
                0.96,
                f"N={x.size:,}\noutside view={x.size - x_view.size:,}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
            )
        _hide_unused(axes.flat, len(batch))
        figure.suptitle(
            f"{label}\n{scope_name} - detector occupancy - "
            f"page {page_index} of {len(batches)}",
            fontsize=14,
            weight="semibold",
        )
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages


def _balanced_batches(values: Sequence, maximum_size: int) -> list[Sequence]:
    """Split values into pages whose panel counts differ by at most one."""
    if maximum_size <= 0:
        raise ValueError("maximum_size must be positive")
    if not values:
        return []
    page_count = (len(values) + maximum_size - 1) // maximum_size
    base_size, extra = divmod(len(values), page_count)
    batches: list[Sequence] = []
    start = 0
    for page_index in range(page_count):
        size = base_size + (1 if page_index < extra else 0)
        batches.append(values[start : start + size])
        start += size
    return batches


def _panel_figure(
    panel_count: int, *, maximum_panels: int
) -> tuple[plt.Figure, np.ndarray]:
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    if panel_count > maximum_panels:
        raise ValueError("panel_count exceeds maximum_panels")
    columns = 1 if panel_count == 1 else 2
    rows = (panel_count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 8.5),
        constrained_layout=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    return figure, axes


def _set_variable_title(axis: plt.Axes, variable: PlotVariable) -> None:
    axis.set_title(variable.title, fontsize=10, loc="left", weight="semibold", pad=9)
    axis.set_title(
        variable.section.upper(),
        fontsize=7,
        loc="right",
        color=SECTION_COLORS.get(variable.section, "#4b5563"),
        pad=11,
    )


def _hide_unused(axes: Iterable[plt.Axes], used: int) -> None:
    for index, axis in enumerate(axes):
        if index >= used:
            axis.set_visible(False)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]
