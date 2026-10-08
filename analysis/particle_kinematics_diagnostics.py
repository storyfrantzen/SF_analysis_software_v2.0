from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np


Array = np.ndarray
PROTON_MASS_GEV = 0.9382720813

BANK_ORDER = ("event", "dis", "kinematics", "pid", "pcal", "ecin", "ecout", "dc", "ft", "cvt")
DETECTOR_BANKS = ("pcal", "ecin", "ecout", "dc", "ft", "cvt")


@dataclass(frozen=True)
class Variable:
    key: str
    title: str
    label: str
    transform: Callable[[Array], Array] | None = None
    fixed_range: tuple[float, float] | None = None
    bins: int = 64


@dataclass(frozen=True)
class Particle:
    name: str
    prefix: str
    branches: Mapping[str, str]


EVENT_VARIABLES = (
    Variable("runNum", "Run number", "run"),
    Variable("helicity", "Raw helicity", "helicity", fixed_range=(-2.5, 2.5), bins=5),
    Variable("charge", "Event charge", "charge"),
)

DIS_VARIABLES = (
    Variable("Q2", r"Four-momentum transfer $Q^2$", r"$Q^2$ [GeV$^2$]"),
    Variable("xB", r"Bjorken $x_B$", r"$x_B$", fixed_range=(0.0, 1.0)),
    Variable("nu", r"Energy transfer $\nu$", r"$\nu$ [GeV]"),
    Variable("W", r"Invariant mass $W$", r"$W$ [GeV]"),
    Variable("y", r"Inelasticity $y$", r"$y$", fixed_range=(0.0, 1.0)),
    Variable("t", r"Momentum transfer $-t$", r"$-t$ [GeV$^2$]"),
)

PARTICLE_KEYS = {
    "p": ("P", "p"),
    "theta": ("Theta", "theta"),
    "phi": ("Phi", "phi"),
    "pid": ("Pid", "pid"),
    "det": ("Det", "det"),
    "sector": ("Sector", "sector"),
    "track_chi2n": ("TrackChi2N", "trackChi2N"),
    "pcal_x": ("XPCAL", "xPCAL"),
    "pcal_y": ("YPCAL", "yPCAL"),
    "pcal_u": ("UPCAL", "uPCAL"),
    "pcal_v": ("VPCAL", "vPCAL"),
    "pcal_w": ("WPCAL", "wPCAL"),
    "pcal_e": ("EPCAL", "E_PCAL"),
    "ecin_u": ("UECIN", "uECIN"),
    "ecin_v": ("VECIN", "vECIN"),
    "ecin_w": ("WECIN", "wECIN"),
    "ecin_e": ("EECIN", "E_ECIN"),
    "ecout_u": ("UECOUT", "uECOUT"),
    "ecout_v": ("VECOUT", "vECOUT"),
    "ecout_w": ("WECOUT", "wECOUT"),
    "ecout_e": ("EECOUT", "E_ECOUT"),
    "dc1_x": ("XDC1", "xDC1"),
    "dc1_y": ("YDC1", "yDC1"),
    "dc2_x": ("XDC2", "xDC2"),
    "dc2_y": ("YDC2", "yDC2"),
    "dc3_x": ("XDC3", "xDC3"),
    "dc3_y": ("YDC3", "yDC3"),
    "dc1_edge": ("EdgeDC1", "edgeDC1"),
    "dc2_edge": ("EdgeDC2", "edgeDC2"),
    "dc3_edge": ("EdgeDC3", "edgeDC3"),
    "ft_x": ("XFT", "xFT"),
    "ft_y": ("YFT", "yFT"),
    "ft_e": ("EFTCAL", "E_FTCAL"),
    "cvt_theta": ("ThetaCVT", "theta_cvt"),
    "cvt_phi": ("PhiCVT", "phi_cvt"),
    "cvt_edge1": ("EdgeCVT1", "edge_cvt1"),
    "cvt_edge3": ("EdgeCVT3", "edge_cvt3"),
    "cvt_edge5": ("EdgeCVT5", "edge_cvt5"),
    "cvt_edge7": ("EdgeCVT7", "edge_cvt7"),
    "cvt_edge12": ("EdgeCVT12", "edge_cvt12"),
}

BANK_KEYS = {
    "kinematics": ("p", "theta", "phi", "sector", "det"),
    "pid": ("pid", "det", "sector", "track_chi2n", "pcal_e", "ecin_e", "ecout_e"),
    "pcal": ("pcal_x", "pcal_y", "pcal_u", "pcal_v", "pcal_w", "pcal_e"),
    "ecin": ("ecin_u", "ecin_v", "ecin_w", "ecin_e"),
    "ecout": ("ecout_u", "ecout_v", "ecout_w", "ecout_e"),
    "dc": (
        "dc1_x", "dc1_y", "dc2_x", "dc2_y", "dc3_x", "dc3_y",
        "dc1_edge", "dc2_edge", "dc3_edge",
    ),
    "ft": ("ft_x", "ft_y", "ft_e"),
    "cvt": (
        "cvt_theta", "cvt_phi", "cvt_edge1", "cvt_edge3", "cvt_edge5",
        "cvt_edge7", "cvt_edge12",
    ),
}


def degrees(values: Array) -> Array:
    return np.rad2deg(np.asarray(values, dtype=float))


def finite(values: Array) -> Array:
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def robust_range(values: Array, fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    if fixed is not None:
        return float(fixed[0]), float(fixed[1])
    clean = finite(values)
    if clean.size == 0:
        return 0.0, 1.0
    if clean.size == 1:
        width = max(0.1 * abs(float(clean[0])), 0.5)
        return float(clean[0] - width), float(clean[0] + width)
    low, high = np.quantile(clean, (0.0025, 0.9975))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        width = max(0.1 * abs(float(low)), 0.5)
        return float(low - width), float(high + width)
    padding = 0.05 * (high - low)
    return float(low - padding), float(high + padding)


def branch_for(prefix: str, key: str, available: set[str]) -> str | None:
    suffix, unprefixed = PARTICLE_KEYS[key]
    candidates = (prefix + suffix,) if prefix else (unprefixed, suffix)
    return next((name for name in candidates if name in available), None)


def discover_particle_prefixes(available: Sequence[str]) -> list[str]:
    names = set(available)
    prefixes: list[str] = []
    if {"p", "theta", "phi"}.issubset(names):
        prefixes.append("")
    for branch in sorted(names):
        if not branch.endswith("P") or branch in {"selectedP"}:
            continue
        prefix = branch[:-1]
        if prefix and prefix + "Theta" in names and prefix + "Phi" in names:
            prefixes.append(prefix)
    return list(dict.fromkeys(prefixes))


def make_particle(prefix: str, available: Sequence[str], name: str | None = None) -> Particle:
    names = set(available)
    branches = {
        key: branch
        for key in PARTICLE_KEYS
        if (branch := branch_for(prefix, key, names)) is not None
    }
    if not {"p", "theta", "phi"}.issubset(branches):
        shown = prefix or "<unprefixed>"
        raise ValueError(f"Particle prefix {shown} does not provide p, theta, and phi")
    return Particle(name=name or prefix or "particle", prefix=prefix, branches=branches)


def banks_with_content(particle: Particle) -> list[str]:
    return [
        bank
        for bank in BANK_KEYS
        if any(key in particle.branches for key in BANK_KEYS[bank])
    ]


def derive_event_arrays(
    arrays: Mapping[str, Array], beam_energy: float | None = None
) -> dict[str, Array]:
    result = {name: np.asarray(values) for name, values in arrays.items()}
    if "W" not in result and "Q2" in result and "nu" in result:
        w2 = PROTON_MASS_GEV**2 + 2.0 * PROTON_MASS_GEV * result["nu"] - result["Q2"]
        result["W"] = np.sqrt(np.where(w2 >= 0.0, w2, np.nan))
    if "y" not in result and beam_energy is not None and "nu" in result:
        result["y"] = np.asarray(result["nu"], dtype=float) / float(beam_energy)
    return result


def required_branches(
    particles: Sequence[Particle], banks: Sequence[str], available: Sequence[str]
) -> list[str]:
    names = set(available)
    requested: list[str] = []
    if "event" in banks:
        requested.extend(item.key for item in EVENT_VARIABLES if item.key in names)
    if "dis" in banks:
        requested.extend(item.key for item in DIS_VARIABLES if item.key in names)
        if "W" not in names:
            requested.extend(name for name in ("Q2", "nu") if name in names)
    for particle in particles:
        for bank in banks:
            requested.extend(
                particle.branches[key]
                for key in BANK_KEYS.get(bank, ())
                if key in particle.branches
            )
    return list(dict.fromkeys(requested))


def render_report(
    output: Path,
    arrays: Mapping[str, Array],
    particles: Sequence[Particle],
    banks: Sequence[str],
    *,
    label: str,
    provenance: Sequence[str],
    beam_energy: float | None = None,
) -> tuple[int, list[dict[str, object]]]:
    values = derive_event_arrays(arrays, beam_energy)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    page_records: list[dict[str, object]] = []
    with PdfPages(temporary) as pdf:
        _title_page(pdf, values, particles, banks, label, provenance)
        page_records.append({"section": "title", "particle": None})
        if "event" in banks:
            page_records.extend(_variable_pages(pdf, values, EVENT_VARIABLES, label, "Event metadata"))
        if "dis" in banks:
            page_records.extend(_dis_pages(pdf, values, label))
        for particle in particles:
            if "kinematics" in banks:
                page_records.extend(_kinematics_pages(pdf, values, particle, label))
            if "pid" in banks:
                page_records.extend(_pid_pages(pdf, values, particle, label))
            for bank in DETECTOR_BANKS:
                if bank in banks:
                    page_records.extend(_detector_pages(pdf, values, particle, bank, label))
    temporary.replace(output)
    return len(page_records), page_records


def render_angular_coverage_report(
    output: Path,
    arrays: Mapping[str, Array],
    particles: Sequence[Particle],
    *,
    label: str,
    provenance: Sequence[str],
) -> tuple[int, list[dict[str, object]]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    records: list[dict[str, object]] = []
    with PdfPages(temporary) as pdf:
        for particle in particles:
            theta = degrees(_particle_values(arrays, particle, "theta"))
            phi = degrees(_particle_values(arrays, particle, "phi"))
            figure, axis = plt.subplots(figsize=(11.0, 8.5))
            figure.patch.set_facecolor("white")
            figure.subplots_adjust(left=0.105, right=0.89, bottom=0.12, top=0.84)
            _hist2d(
                figure,
                axis,
                theta,
                phi,
                f"{particle.name}: angular coverage",
                r"$\theta$ [deg]",
                r"$\phi$ [deg]",
                y_range=(-180.0, 180.0),
                bins=96,
            )
            figure.suptitle(label, fontsize=17, weight="semibold", x=0.105, ha="left", y=0.95)
            axis.set_title(
                rf"{particle.name}: reconstructed $\theta$ versus $\phi$",
                fontsize=14,
                loc="left",
                pad=12,
            )
            axis.tick_params(labelsize=11)
            axis.xaxis.label.set_size(13)
            axis.yaxis.label.set_size(13)
            footer = " | ".join(provenance)
            figure.text(0.105, 0.045, textwrap.shorten(footer, width=155, placeholder=" ..."), fontsize=7.5)
            pdf.savefig(figure)
            plt.close(figure)
            records.append({"section": "angular_coverage_large", "particle": particle.name})
    temporary.replace(output)
    return len(records), records


def _new_page(title: str, subtitle: str = "") -> tuple[plt.Figure, Array]:
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 8.5))
    figure.patch.set_facecolor("white")
    title_lines = textwrap.wrap(title, width=88) or [title]
    heading = "\n".join(title_lines + ([subtitle] if subtitle else []))
    figure.suptitle(
        heading,
        fontsize=13.5,
        weight="semibold",
        x=0.075,
        ha="left",
        y=0.965,
    )
    top = 0.845 if len(title_lines) == 1 else 0.815
    figure.subplots_adjust(
        left=0.075,
        right=0.94,
        bottom=0.085,
        top=top,
        wspace=0.50,
        hspace=0.44,
    )
    return figure, axes


def _title_page(
    pdf: PdfPages,
    arrays: Mapping[str, Array],
    particles: Sequence[Particle],
    banks: Sequence[str],
    label: str,
    provenance: Sequence[str],
) -> None:
    rows = int(next(iter(arrays.values())).size) if arrays else 0
    figure = plt.figure(figsize=(11.0, 8.5))
    figure.patch.set_facecolor("white")
    figure.text(0.07, 0.90, "Generic particle and detector diagnostics", fontsize=23, weight="bold")
    figure.text(0.07, 0.845, label, fontsize=15, color="#374151")
    figure.text(0.07, 0.775, f"Rows plotted: {rows:,}", fontsize=14, weight="bold")
    figure.text(0.07, 0.72, "Particles", fontsize=12.5, weight="bold")
    y = 0.68
    for particle in particles:
        available = ", ".join(banks_with_content(particle))
        prefix = particle.prefix or "unprefixed row"
        figure.text(0.09, y, f"{particle.name}: {prefix}  [{available}]", fontsize=10, family="monospace")
        y -= 0.035
    figure.text(0.07, y - 0.01, "Requested groups", fontsize=12.5, weight="bold")
    figure.text(0.09, y - 0.055, ", ".join(banks), fontsize=10, family="monospace")
    y -= 0.12
    figure.text(0.07, y, "Provenance", fontsize=12.5, weight="bold")
    y -= 0.045
    for line in provenance:
        for start in range(0, len(line), 112):
            figure.text(0.09, y, line[start : start + 112], fontsize=8.3, family="monospace")
            y -= 0.026
    figure.text(
        0.07,
        0.055,
        "Display ranges use the central 99.5% of finite values unless a physical range is fixed. "
        "The JSON sidecar records every requested, rendered, and unavailable group.",
        fontsize=9,
        color="#4b5563",
        wrap=True,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _variable_pages(
    pdf: PdfPages,
    arrays: Mapping[str, Array],
    variables: Sequence[Variable],
    label: str,
    section: str,
) -> list[dict[str, object]]:
    present = [item for item in variables if item.key in arrays]
    records: list[dict[str, object]] = []
    for start in range(0, len(present), 6):
        batch = present[start : start + 6]
        figure, axes = _new_page(label, section)
        for axis, variable in zip(axes.flat, batch, strict=False):
            _hist(axis, arrays[variable.key], variable.title, variable.label, variable.fixed_range, variable.bins)
        _hide_unused(axes, len(batch))
        pdf.savefig(figure)
        plt.close(figure)
        records.append({"section": section.lower().replace(" ", "_"), "particle": None})
    return records


def _dis_pages(pdf: PdfPages, arrays: Mapping[str, Array], label: str) -> list[dict[str, object]]:
    records = _variable_pages(pdf, arrays, DIS_VARIABLES, label, "Event kinematics")
    pairs = (
        ("xB", "Q2", r"DIS coverage", r"$x_B$", r"$Q^2$ [GeV$^2$]"),
        ("Q2", "W", r"$W$ versus $Q^2$", r"$Q^2$ [GeV$^2$]", r"$W$ [GeV]"),
        ("xB", "W", r"$W$ versus $x_B$", r"$x_B$", r"$W$ [GeV]"),
        ("nu", "Q2", r"$Q^2$ versus $\nu$", r"$\nu$ [GeV]", r"$Q^2$ [GeV$^2$]"),
    )
    available = [pair for pair in pairs if pair[0] in arrays and pair[1] in arrays]
    if available:
        figure, axes = _new_page(label, "Event-kinematic coverage")
        for axis, (x, y, title, xlabel, ylabel) in zip(axes.flat, available, strict=False):
            _hist2d(figure, axis, arrays[x], arrays[y], title, xlabel, ylabel)
        _hide_unused(axes, len(available))
        pdf.savefig(figure)
        plt.close(figure)
        records.append({"section": "dis_coverage", "particle": None})
    return records


def _particle_values(arrays: Mapping[str, Array], particle: Particle, key: str) -> Array:
    return np.asarray(arrays[particle.branches[key]])


def _kinematics_pages(
    pdf: PdfPages, arrays: Mapping[str, Array], particle: Particle, label: str
) -> list[dict[str, object]]:
    if not all(key in particle.branches for key in ("p", "theta", "phi")):
        return []
    p = _particle_values(arrays, particle, "p")
    theta = degrees(_particle_values(arrays, particle, "theta"))
    phi = degrees(_particle_values(arrays, particle, "phi"))
    figure, axes = _new_page(label, f"{particle.name}: reconstructed kinematics")
    _hist(axes.flat[0], p, "Momentum", r"$p$ [GeV]")
    _hist(axes.flat[1], theta, "Polar angle", r"$\theta$ [deg]")
    _hist(axes.flat[2], phi, "Azimuth", r"$\phi$ [deg]", (-180.0, 180.0), 72)
    _hist2d(figure, axes.flat[3], theta, phi, "Angular coverage", r"$\theta$ [deg]", r"$\phi$ [deg]", y_range=(-180.0, 180.0))
    _hist2d(figure, axes.flat[4], theta, p, "Momentum-angle coverage", r"$\theta$ [deg]", r"$p$ [GeV]")
    if "sector" in particle.branches:
        _categorical(axes.flat[5], _particle_values(arrays, particle, "sector"), "Sector occupancy", "sector")
    else:
        axes.flat[5].set_visible(False)
    pdf.savefig(figure)
    plt.close(figure)
    records = [{"section": "kinematics", "particle": particle.name}]

    if "sector" in particle.branches:
        sector = np.asarray(_particle_values(arrays, particle, "sector"), dtype=int)
        sectors = [int(value) for value in sorted(np.unique(sector)) if 1 <= int(value) <= 6]
        if sectors:
            figure, axes = _new_page(label, f"{particle.name}: sector-resolved shapes")
            for axis, values, title, xlabel, fixed in (
                (axes.flat[0], p, "Momentum by sector", r"$p$ [GeV]", None),
                (axes.flat[1], theta, "Polar angle by sector", r"$\theta$ [deg]", None),
                (axes.flat[2], phi, "Azimuth by sector", r"$\phi$ [deg]", (-180.0, 180.0)),
            ):
                _sector_overlay(axis, values, sector, sectors, title, xlabel, fixed)
            if "det" in particle.branches:
                _categorical(axes.flat[3], _particle_values(arrays, particle, "det"), "Detector-region occupancy", "detector code")
            else:
                axes.flat[3].set_visible(False)
            axes.flat[4].set_visible(False)
            axes.flat[5].set_visible(False)
            pdf.savefig(figure)
            plt.close(figure)
            records.append({"section": "sector_shapes", "particle": particle.name})
    return records


def _pid_pages(
    pdf: PdfPages, arrays: Mapping[str, Array], particle: Particle, label: str
) -> list[dict[str, object]]:
    panels: list[tuple[str, str, str]] = []
    for key, title, xlabel in (
        ("pid", "Particle identifier", "PID"),
        ("det", "Detector region", "detector code (0=FT, 1=FD, 2=CD)"),
        ("sector", "Sector", "sector"),
        ("track_chi2n", "Track fit quality", r"$\chi^2/NDF$"),
        ("pcal_e", "PCAL energy", "energy [GeV]"),
        ("ecin_e", "ECIN energy", "energy [GeV]"),
        ("ecout_e", "ECOUT energy", "energy [GeV]"),
    ):
        if key in particle.branches:
            panels.append((key, title, xlabel))
    if all(key in particle.branches for key in ("p", "pcal_e", "ecin_e", "ecout_e")):
        panels.append(("sampling_fraction", "Calorimeter sampling fraction", r"$(E_{PCAL}+E_{ECIN}+E_{ECOUT})/p$"))
    records: list[dict[str, object]] = []
    for start in range(0, len(panels), 6):
        batch = panels[start : start + 6]
        figure, axes = _new_page(label, f"{particle.name}: identity and reconstruction")
        for axis, (key, title, xlabel) in zip(axes.flat, batch, strict=False):
            if key == "sampling_fraction":
                p = np.asarray(_particle_values(arrays, particle, "p"), dtype=float)
                numerator = sum(
                    np.asarray(_particle_values(arrays, particle, energy), dtype=float)
                    for energy in ("pcal_e", "ecin_e", "ecout_e")
                )
                plotted = np.divide(numerator, p, out=np.full_like(p, np.nan), where=p > 0.0)
            else:
                plotted = _particle_values(arrays, particle, key)
            if key in {"pid", "det", "sector"}:
                _categorical(axis, plotted, title, xlabel)
            else:
                _hist(axis, plotted, title, xlabel)
        _hide_unused(axes, len(batch))
        pdf.savefig(figure)
        plt.close(figure)
        records.append({"section": "pid", "particle": particle.name})
    return records


def _detector_pages(
    pdf: PdfPages,
    arrays: Mapping[str, Array],
    particle: Particle,
    bank: str,
    label: str,
) -> list[dict[str, object]]:
    keys = set(particle.branches)
    panels: list[tuple[str, tuple[str, ...], str, str, str]] = []
    if bank == "pcal":
        panels = _calorimeter_panels("pcal", "PCAL")
    elif bank == "ecin":
        panels = _calorimeter_panels("ecin", "ECIN", include_xy=False)
    elif bank == "ecout":
        panels = _calorimeter_panels("ecout", "ECOUT", include_xy=False)
    elif bank == "dc":
        panels = [
            ("xy", (f"dc{i}_x", f"dc{i}_y"), f"DC region {i} occupancy", "x [cm]", "y [cm]")
            for i in (1, 2, 3)
        ] + [
            ("hist", (f"dc{i}_edge",), f"DC region {i} edge distance", "edge [cm]", "")
            for i in (1, 2, 3)
        ]
    elif bank == "ft":
        panels = [
            ("xy", ("ft_x", "ft_y"), "FTCAL occupancy", "x [cm]", "y [cm]"),
            ("hist", ("ft_e",), "FTCAL energy", "energy [GeV]", ""),
        ]
    elif bank == "cvt":
        panels = [
            ("angles", ("cvt_phi", "cvt_theta"), "CVT layer-1 angular occupancy", r"$\phi$ [deg]", r"$\theta$ [deg]"),
            ("hist_deg", ("cvt_theta",), "CVT layer-1 polar angle", r"$\theta$ [deg]", ""),
            ("hist_deg", ("cvt_phi",), "CVT layer-1 azimuth", r"$\phi$ [deg]", ""),
        ] + [
            ("hist", (f"cvt_edge{i}",), f"CVT layer {i} edge distance", "edge [cm]", "")
            for i in (1, 3, 5, 7, 12)
        ]
    available = [panel for panel in panels if all(key in keys for key in panel[1])]
    records: list[dict[str, object]] = []
    for start in range(0, len(available), 6):
        batch = available[start : start + 6]
        figure, axes = _new_page(label, f"{particle.name}: {bank.upper()} diagnostics")
        for axis, (kind, panel_keys, title, xlabel, ylabel) in zip(axes.flat, batch, strict=False):
            if kind in {"xy", "uv", "angles"}:
                x = _particle_values(arrays, particle, panel_keys[0])
                y = _particle_values(arrays, particle, panel_keys[1])
                if kind == "angles":
                    x, y = degrees(x), degrees(y)
                _hist2d(figure, axis, x, y, title, xlabel, ylabel, equal_aspect=kind == "xy")
            else:
                plotted = _particle_values(arrays, particle, panel_keys[0])
                if kind == "hist_deg":
                    plotted = degrees(plotted)
                _hist(axis, plotted, title, xlabel)
        _hide_unused(axes, len(batch))
        pdf.savefig(figure)
        plt.close(figure)
        records.append({"section": bank, "particle": particle.name})
    return records


def _calorimeter_panels(
    prefix: str, title: str, *, include_xy: bool = True
) -> list[tuple[str, tuple[str, ...], str, str, str]]:
    panels: list[tuple[str, tuple[str, ...], str, str, str]] = []
    if include_xy:
        panels.append(("xy", (f"{prefix}_x", f"{prefix}_y"), f"{title} global occupancy", "x [cm]", "y [cm]"))
    panels.extend(
        [
            ("uv", (f"{prefix}_u", f"{prefix}_v"), f"{title} U-V occupancy", "U [cm]", "V [cm]"),
            ("uv", (f"{prefix}_u", f"{prefix}_w"), f"{title} U-W occupancy", "U [cm]", "W [cm]"),
            ("uv", (f"{prefix}_v", f"{prefix}_w"), f"{title} V-W occupancy", "V [cm]", "W [cm]"),
            ("hist", (f"{prefix}_e",), f"{title} energy", "energy [GeV]", ""),
        ]
    )
    return panels


def _hist(
    axis: plt.Axes,
    values: Array,
    title: str,
    xlabel: str,
    fixed_range: tuple[float, float] | None = None,
    bins: int = 64,
) -> None:
    clean = finite(values)
    display = robust_range(clean, fixed_range)
    in_range = clean[(clean >= display[0]) & (clean <= display[1])]
    axis.hist(in_range, bins=bins, range=display, color="#2563eb", alpha=0.84, edgecolor="white", linewidth=0.25)
    axis.set_title(title, fontsize=9.5, loc="left", weight="semibold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Rows")
    axis.grid(alpha=0.18)
    if clean.size:
        axis.text(
            0.97,
            0.94,
            f"N={clean.size:,}\nmedian={np.median(clean):.4g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    _compact(axis)


def _categorical(axis: plt.Axes, values: Array, title: str, xlabel: str) -> None:
    clean = finite(values).astype(int)
    unique, counts = np.unique(clean, return_counts=True)
    axis.bar(unique.astype(str), counts, color="#2563eb", alpha=0.84)
    axis.set_title(title, fontsize=9.5, loc="left", weight="semibold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Rows")
    axis.grid(axis="y", alpha=0.18)
    _compact(axis)


def _hist2d(
    figure: plt.Figure,
    axis: plt.Axes,
    x: Array,
    y: Array,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    equal_aspect: bool = False,
    bins: int = 64,
) -> None:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[valid], yy[valid]
    if equal_aspect and x_range is None and y_range is None:
        common = robust_range(np.concatenate((xx, yy)))
        xr, yr = common, common
    else:
        xr = robust_range(xx, x_range)
        yr = robust_range(yy, y_range)
    view = (xx >= xr[0]) & (xx <= xr[1]) & (yy >= yr[0]) & (yy <= yr[1])
    xx, yy = xx[view], yy[view]
    if xx.size:
        counts, _, _ = np.histogram2d(xx, yy, bins=bins, range=(xr, yr))
        maximum = max(float(counts.max()), 1.01)
        image = axis.hist2d(xx, yy, bins=bins, range=(xr, yr), cmap="viridis", norm=LogNorm(vmin=1.0, vmax=maximum))[3]
        colorbar = figure.colorbar(image, ax=axis, pad=0.018, fraction=0.045)
        colorbar.ax.tick_params(labelsize=6)
    axis.set_title(title, fontsize=9.5, loc="left", weight="semibold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if equal_aspect:
        axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.1)
    _compact(axis)


def _sector_overlay(
    axis: plt.Axes,
    values: Array,
    sectors: Array,
    observed: Sequence[int],
    title: str,
    xlabel: str,
    fixed_range: tuple[float, float] | None,
) -> None:
    clean = finite(values)
    display = robust_range(clean, fixed_range)
    edges = np.linspace(display[0], display[1], 65)
    for sector in observed:
        chosen = np.asarray(values, dtype=float)[np.asarray(sectors) == sector]
        chosen = chosen[np.isfinite(chosen)]
        chosen = chosen[(chosen >= display[0]) & (chosen <= display[1])]
        if chosen.size:
            axis.hist(chosen, bins=edges, density=True, histtype="step", linewidth=1.1, label=f"S{sector}")
    axis.set_title(title, fontsize=9.5, loc="left", weight="semibold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Unit-normalized density")
    axis.legend(fontsize=6, frameon=False, ncol=2)
    axis.grid(alpha=0.18)
    _compact(axis)


def _hide_unused(axes: Array, used: int) -> None:
    for axis in axes.flat[used:]:
        axis.set_visible(False)


def _compact(axis: plt.Axes) -> None:
    axis.tick_params(labelsize=7)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)


def summary_for(
    arrays: Mapping[str, Array],
    particles: Sequence[Particle],
    requested_banks: Sequence[str],
    page_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = int(next(iter(arrays.values())).size) if arrays else 0
    particle_records: dict[str, object] = {}
    for particle in particles:
        available_banks = banks_with_content(particle)
        rendered = sorted(
            {
                str(record["section"])
                for record in page_records
                if record.get("particle") == particle.name
            }
        )
        rendered_banks = set(rendered)
        if rendered_banks & {"sector_shapes", "angular_coverage_large"}:
            rendered_banks.add("kinematics")
        missing_banks = (set(requested_banks) & set(BANK_KEYS)) - rendered_banks
        particle_records[particle.name] = {
            "branch_prefix": particle.prefix,
            "branches": dict(sorted(particle.branches.items())),
            "available_banks": available_banks,
            "rendered_sections": rendered,
            "requested_banks_without_rendered_content": sorted(missing_banks),
        }
    return {
        "rows": rows,
        "available_arrays": sorted(arrays),
        "requested_banks": list(requested_banks),
        "particles": particle_records,
        "pages": len(page_records),
        "page_records": [dict(record) for record in page_records],
    }
