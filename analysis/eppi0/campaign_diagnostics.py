from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from .binning import AnalysisBinning


REPORT_SCHEMA_VERSION = "eppi0-campaign-diagnostic-v1"


def _scalar(data, name: str, default=None):
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _format_number(value, *, digits: int = 6) -> str:
    if not _finite(value):
        return "not available"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e5 or abs(number) < 1.0e-3:
        return f"{number:.{digits}g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _format_percent(numerator: float, denominator: float) -> str:
    if not _finite(numerator) or not _finite(denominator) or float(denominator) == 0.0:
        return "not available"
    percentage = 100.0 * float(numerator) / float(denominator)
    if abs(percentage) >= 1.0e5:
        return f"{percentage:.6g}%"
    return f"{percentage:.3f}%"


def _format_count(count: int, total: int) -> str:
    return f"{count:,} / {total:,} ({_format_percent(count, total)})"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.full(5, np.nan)
    return np.quantile(finite, [0.0, 0.1, 0.5, 0.9, 1.0])


def _quantile_row(label: str, values: np.ndarray, unit: str = "") -> str:
    quantiles = _quantiles(values)
    suffix = f" {unit}" if unit else ""
    formatted = [f"{_format_number(value)}{suffix}" for value in quantiles]
    return f"| {label} | " + " | ".join(formatted) + " |"


def _as_4d(data, name: str, binning: AnalysisBinning, dtype=None) -> np.ndarray:
    values = np.asarray(data[name], dtype=dtype)
    if values.shape == binning.shape:
        return values
    if values.shape == (binning.size,):
        return binning.unflatten(values)
    raise ValueError(
        f"{name} has shape {values.shape}; expected {binning.shape} or {(binning.size,)}"
    )


def _optional_4d(data, name: str, binning: AnalysisBinning, dtype=None):
    return _as_4d(data, name, binning, dtype=dtype) if name in data.files else None


def _binning_from_cross_section(cross_section) -> AnalysisBinning:
    required = ("q2_edges", "xb_edges", "t_edges", "phi_edges")
    missing = [name for name in required if name not in cross_section.files]
    if missing:
        raise ValueError(
            "cross-section artifact is missing bin edges: " + ", ".join(missing)
        )
    return AnalysisBinning(
        cross_section["q2_edges"],
        cross_section["xb_edges"],
        cross_section["t_edges"],
        cross_section["phi_edges"],
    )


def _mask_bundle(unfolding, cross_section, binning: AnalysisBinning):
    minimum_acceptance = float(_scalar(cross_section, "minimum_acceptance", 0.005))
    efficiency = _as_4d(unfolding, "efficiency", binning, dtype=float)

    sources: dict[str, str] = {}

    def stored_or_derived(name: str, derived: np.ndarray, description: str) -> np.ndarray:
        stored = _optional_4d(cross_section, name, binning, dtype=bool)
        if stored is not None:
            sources[name] = "stored in cross-section artifact"
            return stored
        sources[name] = f"legacy fallback: {description}"
        return np.asarray(derived, dtype=bool)

    acceptance = stored_or_derived(
        "acceptance_validity_mask",
        np.isfinite(efficiency) & (efficiency > minimum_acceptance),
        f"finite efficiency > {minimum_acceptance:g}",
    )
    if "radiative_reliable" in unfolding.files:
        radiative_derived = _as_4d(
            unfolding, "radiative_reliable", binning, dtype=bool
        )
    else:
        radiative_derived = np.ones(binning.shape, dtype=bool)
    radiative = stored_or_derived(
        "radiative_validity_mask", radiative_derived, "unfolding radiative_reliable"
    )

    if "bin_centering_reliable" in cross_section.files:
        reliable = _as_4d(
            cross_section, "bin_centering_reliable", binning, dtype=bool
        )
        cbc = _as_4d(cross_section, "bin_centering_C_BC", binning, dtype=float)
        bc_derived = reliable & np.isfinite(cbc) & (cbc > 0.0)
    else:
        bc_derived = np.ones(binning.shape, dtype=bool)
    bin_centering = stored_or_derived(
        "bin_centering_validity_mask",
        bc_derived,
        "finite positive C_BC in a reliable bin, or all bins when C_BC was not applied",
    )

    yield_name = "corrected_yield" if "corrected_yield" in unfolding.files else "unfolded"
    corrected_yield = _as_4d(unfolding, yield_name, binning, dtype=float)
    yield_valid = stored_or_derived(
        "yield_validity_mask",
        np.isfinite(corrected_yield) & (corrected_yield >= 0.0),
        f"finite nonnegative {yield_name}",
    )
    uncertainty_name = (
        "corrected_uncertainty"
        if "corrected_uncertainty" in unfolding.files
        else "sigma_total"
    )
    corrected_uncertainty = _as_4d(
        unfolding, uncertainty_name, binning, dtype=float
    )
    uncertainty_valid = stored_or_derived(
        "uncertainty_validity_mask",
        np.isfinite(corrected_uncertainty) & (corrected_uncertainty > 0.0),
        f"finite positive {uncertainty_name}",
    )

    if "bin_volume" in cross_section.files:
        volume = _as_4d(cross_section, "bin_volume", binning, dtype=float)
        normalization_derived = np.isfinite(volume) & (volume > 0.0)
    else:
        normalization_derived = np.ones(binning.shape, dtype=bool)
    normalization = stored_or_derived(
        "normalization_validity_mask",
        normalization_derived,
        "finite positive physical bin volume (flux was unavailable to the fallback)",
    )

    conjunction = (
        acceptance
        & radiative
        & bin_centering
        & yield_valid
        & uncertainty_valid
        & normalization
    )
    final_stored = _optional_4d(
        cross_section, "final_validity_mask", binning, dtype=bool
    )
    if final_stored is None:
        final = conjunction
        sources["final_validity_mask"] = "legacy fallback: conjunction of component masks"
    else:
        final = final_stored
        sources["final_validity_mask"] = "stored in cross-section artifact"
        if not np.array_equal(final, conjunction):
            raise ValueError(
                "stored final_validity_mask does not equal the conjunction of its "
                "component masks"
            )

    return {
        "acceptance": acceptance,
        "radiative": radiative,
        "bin_centering": bin_centering,
        "yield": yield_valid,
        "uncertainty": uncertainty_valid,
        "normalization": normalization,
        "final": final,
    }, sources, efficiency, corrected_yield, corrected_uncertainty


def _markdown_artifact_table(paths: Iterable[tuple[str, Path]]) -> list[str]:
    lines = [
        "| Role | Artifact | Size | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for role, path in paths:
        resolved = path.resolve()
        size_mib = resolved.stat().st_size / (1024.0 * 1024.0)
        lines.append(
            f"| {role} | `{resolved}` | {size_mib:.3f} MiB | `{_sha256(resolved)}` |"
        )
    return lines


def _mask_table(masks: dict[str, np.ndarray], total: int) -> list[str]:
    labels = (
        ("acceptance", "Acceptance-valid"),
        ("radiative", "Radiative-correction-valid"),
        ("bin_centering", "Bin-centering-valid"),
        ("yield", "Finite nonnegative corrected yield"),
        ("uncertainty", "Finite positive propagated uncertainty"),
        ("normalization", "Physical normalization (positive volume and flux)"),
        ("final", "Final valid bins (all requirements)"),
    )
    lines = ["| Independent mask | Passing 4D bins | Fraction of all bins |", "|---|---:|---:|"]
    for key, label in labels:
        count = int(np.count_nonzero(masks[key]))
        lines.append(f"| {label} | {count:,} / {total:,} | {_format_percent(count, total)} |")
    return lines


def _sequential_attrition_table(masks: dict[str, np.ndarray]) -> list[str]:
    total = masks["final"].size
    current = np.ones(masks["final"].shape, dtype=bool)
    order = (
        ("normalization", "Physical normalization space"),
        ("acceptance", "+ acceptance"),
        ("radiative", "+ radiative reliability"),
        ("bin_centering", "+ bin-centering reliability"),
        ("yield", "+ corrected-yield validity"),
        ("uncertainty", "+ propagated-uncertainty validity"),
    )
    lines = [
        "| Sequential requirement | Remaining 4D bins | Removed at this step |",
        "|---|---:|---:|",
    ]
    previous = total
    for key, label in order:
        current &= masks[key]
        count = int(np.count_nonzero(current))
        lines.append(f"| {label} | {count:,} | {previous - count:,} |")
        previous = count
    return lines


def _coordinate(value_edges: np.ndarray, index: int) -> str:
    return f"[{value_edges[index]:g}, {value_edges[index + 1]:g})"


def _largest_relative_uncertainties(
    values: np.ndarray,
    uncertainties: np.ndarray,
    valid: np.ndarray,
    binning: AnalysisBinning,
    maximum_rows: int = 8,
) -> list[str]:
    relative = np.divide(
        uncertainties,
        np.abs(values),
        out=np.full(values.shape, np.inf),
        where=np.abs(values) > 0.0,
    )
    candidates = np.flatnonzero(valid.ravel())
    if not candidates.size:
        return ["No final-valid bins were available."]
    ranked = candidates[np.argsort(relative.ravel()[candidates])[::-1]][:maximum_rows]
    lines = [
        "| Q2 bin | xB bin | -t bin | phi bin [deg] | Value | Uncertainty | Relative uncertainty |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for flat in ranked:
        iq2, ixb, it, iphi = np.unravel_index(flat, values.shape)
        lines.append(
            "| "
            + " | ".join(
                (
                    _coordinate(binning.q2_edges, iq2),
                    _coordinate(binning.xb_edges, ixb),
                    _coordinate(binning.t_edges, it),
                    _coordinate(binning.phi_edges, iphi),
                    _format_number(values[iq2, ixb, it, iphi]),
                    _format_number(uncertainties[iq2, ixb, it, iphi]),
                    _format_percent(
                        uncertainties[iq2, ixb, it, iphi],
                        abs(values[iq2, ixb, it, iphi]),
                    ),
                )
            )
            + " |"
        )
    return lines


def _response_section(
    response_meta,
    masks: dict[str, np.ndarray],
    binning: AnalysisBinning,
) -> tuple[list[str], dict[str, int | float]]:
    required = ("truth_total", "reconstructed_total", "efficiency")
    missing = [name for name in required if name not in response_meta.files]
    if missing:
        raise ValueError("response metadata is missing: " + ", ".join(missing))
    for name, expected in (
        ("q2_edges", binning.q2_edges),
        ("xb_edges", binning.xb_edges),
        ("t_edges", binning.t_edges),
        ("phi_edges", binning.phi_edges),
    ):
        if name in response_meta.files:
            actual = np.asarray(response_meta[name], dtype=float)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"response metadata {name} does not match the cross section"
                )
    truth = _as_4d(response_meta, "truth_total", binning, dtype=float)
    reconstructed = _as_4d(
        response_meta, "reconstructed_total", binning, dtype=float
    )
    efficiency = _as_4d(response_meta, "efficiency", binning, dtype=float)
    physical = masks["normalization"]
    failed_acceptance = physical & ~masks["acceptance"]
    failed_truth = truth[failed_acceptance]
    populated = truth > 0.0
    relative_response_uncertainty = np.full(efficiency.shape, np.nan)
    if "response_variance_sum" in response_meta.files:
        variance = _as_4d(
            response_meta, "response_variance_sum", binning, dtype=float
        )
        relative_response_uncertainty = np.divide(
            np.sqrt(np.maximum(variance, 0.0)),
            efficiency,
            out=np.full(efficiency.shape, np.nan),
            where=np.isfinite(efficiency) & (efficiency > 0.0),
        )

    lines = [
        "## Response-MC support",
        "",
        "The response metadata stores generated-weight sums. For unweighted MC, one unit "
        "equals one generated event; for weighted MC, the support thresholds below are only "
        "heuristic unless supplemented by an effective-sample-size calculation. The relative "
        "response uncertainty is a multinomial-probability diagnostic, not the complete "
        "uncertainty on the unfolded result.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Truth-weight sum in the 4D analysis range | {_format_number(np.sum(truth))} |",
        f"| Selected reconstructed-weight sum in the 4D analysis range | {_format_number(np.sum(reconstructed))} |",
        f"| Global selected/truth-weight ratio | {_format_number(np.sum(reconstructed) / np.sum(truth) if np.sum(truth) > 0 else np.nan)} |",
        f"| Truth-populated 4D bins | {_format_count(int(np.count_nonzero(populated)), binning.size)} |",
        f"| Physical bins failing the acceptance requirement | {int(np.count_nonzero(failed_acceptance)):,} |",
        f"| Those failed bins with zero truth support | {int(np.count_nonzero(failed_truth <= 0.0)):,} |",
        f"| Those failed bins with 1-9 truth-weight units | {int(np.count_nonzero((failed_truth >= 1.0) & (failed_truth < 10.0))):,} |",
        f"| Those failed bins with 10-99 truth-weight units | {int(np.count_nonzero((failed_truth >= 10.0) & (failed_truth < 100.0))):,} |",
        f"| Those failed bins with at least 100 truth-weight units | {int(np.count_nonzero(failed_truth >= 100.0)):,} |",
        "",
        "| Distribution | Minimum | 10th percentile | Median | 90th percentile | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
        _quantile_row("Truth-weight sum in truth-populated bins", truth[populated]),
        _quantile_row("Efficiency in truth-populated bins", efficiency[populated]),
        _quantile_row(
            "Relative response-probability uncertainty",
            relative_response_uncertainty[populated],
        ),
        "",
    ]
    metrics = {
        "acceptance_failed_physical": int(np.count_nonzero(failed_acceptance)),
        "acceptance_failed_truth_lt100": int(np.count_nonzero(failed_truth < 100.0)),
        "acceptance_failed_truth_ge100": int(np.count_nonzero(failed_truth >= 100.0)),
    }
    return lines, metrics


def render_campaign_diagnostics(
    unfolding_path: str | Path,
    cross_section_path: str | Path,
    harmonics_path: str | Path,
    *,
    response_meta_path: str | Path | None = None,
    title: str = "EPPI0 campaign diagnostic",
    software_revision: str = "not recorded",
    invocation: str = "not recorded",
) -> str:
    """Render a reusable numerical end-of-campaign audit as Markdown."""
    unfolding_path = Path(unfolding_path)
    cross_section_path = Path(cross_section_path)
    harmonics_path = Path(harmonics_path)
    response_path = Path(response_meta_path) if response_meta_path is not None else None
    paths = [
        ("Unfolding", unfolding_path),
        ("Cross section", cross_section_path),
        ("Harmonic fits", harmonics_path),
    ]
    if response_path is not None:
        paths.append(("Response metadata", response_path))
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    unfolding = np.load(unfolding_path, allow_pickle=False)
    cross_section = np.load(cross_section_path, allow_pickle=False)
    harmonics = np.load(harmonics_path, allow_pickle=False)
    response_meta = (
        np.load(response_path, allow_pickle=False) if response_path is not None else None
    )
    try:
        binning = _binning_from_cross_section(cross_section)
        masks, mask_sources, efficiency, corrected_yield, corrected_uncertainty = (
            _mask_bundle(unfolding, cross_section, binning)
        )
        values = _as_4d(
            cross_section, "reduced_cross_section", binning, dtype=float
        )
        errors = _as_4d(cross_section, "uncertainty", binning, dtype=float)
        final = masks["final"]
        total = binning.size
        physical = masks["normalization"]
        physical_cells = np.any(physical, axis=-1)
        final_phi_counts = np.count_nonzero(final, axis=-1)

        harmonic_shape = binning.shape[:-1]
        fit_success = np.asarray(harmonics["fit_success"], dtype=bool)
        quality_mask = np.asarray(
            harmonics["quality_mask"] if "quality_mask" in harmonics.files else fit_success,
            dtype=bool,
        )
        if fit_success.shape != harmonic_shape or quality_mask.shape != harmonic_shape:
            raise ValueError(
                "harmonic fit arrays do not match the first three cross-section dimensions"
            )
        points = np.asarray(harmonics["points"], dtype=int)
        chi2_ndf = np.asarray(harmonics["chi2_ndf"], dtype=float)
        parameters = np.asarray(harmonics["parameters"], dtype=float)
        parameter_uncertainties = (
            np.asarray(harmonics["parameter_uncertainties"], dtype=float)
            if "parameter_uncertainties" in harmonics.files
            else np.sqrt(np.maximum(np.diagonal(harmonics["covariance"], axis1=-2, axis2=-1), 0.0))
        )
        relative_a = np.divide(
            parameter_uncertainties[..., 0],
            np.abs(parameters[..., 0]),
            out=np.full(harmonic_shape, np.nan),
            where=np.abs(parameters[..., 0]) > 0.0,
        )

        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"Report schema: `{REPORT_SCHEMA_VERSION}`",
            f"Software revision: `{software_revision}`",
            "",
            "This is a numerical audit, not an automatic physics sign-off. Counts for the "
            "individual validity masks are independent; the sequential table shows their "
            "actual combined attrition in a declared order.",
            "",
            "## Reproduction and provenance",
            "",
            f"Invocation: `{invocation}`",
            "",
            *_markdown_artifact_table(paths),
            "",
            "## Campaign normalization and applied corrections",
            "",
        ]

        original_charge = _scalar(
            unfolding,
            "current_efficiency_original_beam_charge_c",
            _scalar(unfolding, "beam_charge_original_c", np.nan),
        )
        analysis_charge = _scalar(
            unfolding,
            "current_efficiency_analysis_beam_charge_c",
            _scalar(unfolding, "beam_charge_c", np.nan),
        )
        removed_charge = (
            float(original_charge) - float(analysis_charge)
            if _finite(original_charge) and _finite(analysis_charge)
            else np.nan
        )
        signal_region = float(np.sum(unfolding["measured_signal_region"])) if "measured_signal_region" in unfolding.files else np.nan
        background = float(np.sum(unfolding["estimated_background"])) if "estimated_background" in unfolding.files else np.nan
        clipped = float(np.sum(unfolding["background_clipped_deficit"])) if "background_clipped_deficit" in unfolding.files else np.nan
        measured_total = float(np.sum(unfolding["measured"])) if "measured" in unfolding.files else np.nan
        unfolded_total = float(np.sum(unfolding["unfolded"])) if "unfolded" in unfolding.files else np.nan
        corrected_total = float(np.sum(corrected_yield))
        excluded_runs = np.asarray(
            unfolding["current_efficiency_excluded_runs"]
            if "current_efficiency_excluded_runs" in unfolding.files
            else np.empty(0),
            dtype=np.int64,
        )
        lines.extend(
            [
                "| Quantity | Value |",
                "|---|---:|",
                f"| Original integrated beam charge | {_format_number(original_charge)} C |",
                f"| Analysis integrated beam charge | {_format_number(analysis_charge)} C |",
                f"| Charge removed by run exclusions | {_format_number(removed_charge)} C ({_format_percent(removed_charge, original_charge)}) |",
                f"| Explicitly excluded runs | {excluded_runs.size:,}" + (f" (`{', '.join(map(str, excluded_runs.tolist()))}`) |" if excluded_runs.size else " |"),
                f"| Excluded selected events | {int(_scalar(unfolding, 'current_efficiency_excluded_event_count', 0)):,} |",
                f"| Integrated luminosity used downstream | {_format_number(_scalar(cross_section, 'luminosity_fb', np.nan))} fb^-1 |",
                f"| Current-efficiency correction applied | {bool(_scalar(unfolding, 'current_efficiency_applied', False))} |",
                f"| Current-efficiency reference current | {_format_number(_scalar(unfolding, 'current_efficiency_reference_current_nA', np.nan))} nA |",
                f"| D at the reference current | {_format_number(_scalar(unfolding, 'current_efficiency_D_reference', np.nan))} |",
                f"| Event-weight range | [{_format_number(_scalar(unfolding, 'current_efficiency_weight_min', np.nan))}, {_format_number(_scalar(unfolding, 'current_efficiency_weight_max', np.nan))}] |",
                f"| Background subtraction applied | {bool(_scalar(unfolding, 'background_subtraction_applied', False))} |",
                f"| Signal-region weighted count | {_format_number(signal_region)} |",
                f"| Estimated background under signal | {_format_number(background)} ({_format_percent(background, signal_region)}) |",
                f"| Clipped negative-bin deficit | {_format_number(clipped)} ({_format_percent(clipped, measured_total)}) |",
                f"| Background-negative policy | `{_scalar(unfolding, 'background_negative_policy', '')}` |",
                f"| Negative / response-supported negative REC bins | {int(_scalar(unfolding, 'background_negative_bins', 0)):,} / {int(_scalar(unfolding, 'background_negative_supported_bins', 0)):,} |",
                f"| Measured total entering unfolding | {_format_number(measured_total)} |",
                f"| Unfolded total | {_format_number(unfolded_total)} |",
                f"| Radiatively corrected total | {_format_number(corrected_total)} |",
                f"| Corrected / unfolded aggregate ratio | {_format_number(corrected_total / unfolded_total if unfolded_total else np.nan)} |",
                "",
            ]
        )

        if "bin_centering_C_BC" in cross_section.files:
            cbc = _as_4d(cross_section, "bin_centering_C_BC", binning, dtype=float)
            lines.extend(
                [
                    "Bin-centering factor distribution in bin-centering-valid bins:",
                    "",
                    "| Distribution | Minimum | 10th percentile | Median | 90th percentile | Maximum |",
                    "|---|---:|---:|---:|---:|---:|",
                    _quantile_row("C_BC", cbc[masks["bin_centering"]]),
                    "",
                ]
            )

        lines.extend(
            [
                "## Four-dimensional validity and attrition",
                "",
                f"Configured grid: `{binning.shape[0]} x {binning.shape[1]} x {binning.shape[2]} x {binning.shape[3]}` = {total:,} bins.",
                "",
                *_mask_table(masks, total),
                "",
                "Because these masks overlap, their individual failure counts must not be added. "
                "For an additive accounting, use the following sequential cut flow:",
                "",
                *_sequential_attrition_table(masks),
                "",
                "Mask provenance:",
                "",
            ]
        )
        for name, source in mask_sources.items():
            lines.append(f"- `{name}`: {source}.")
        lines.append("")

        lines.extend(
            [
                "## Three-dimensional phi coverage and harmonic yield",
                "",
                f"There are {int(np.prod(harmonic_shape)):,} configured `(Q2, xB, -t)` cells, "
                f"of which {int(np.count_nonzero(physical_cells)):,} contain at least one "
                "physically normalizable phi bin.",
                "",
                "| Final-valid phi bins in a cell | Physical 3D cells meeting threshold |",
                "|---|---:|",
            ]
        )
        thresholds = sorted({1, 4, 8, 12, 16, binning.shape[-1]})
        for threshold in thresholds:
            if threshold <= binning.shape[-1]:
                count = int(np.count_nonzero(physical_cells & (final_phi_counts >= threshold)))
                lines.append(f"| At least {threshold} | {count:,} / {int(np.count_nonzero(physical_cells)):,} |")
        raw_count = int(np.count_nonzero(fit_success))
        quality_count = int(np.count_nonzero(quality_mask))
        lines.extend(
            [
                "",
                f"- Numerically successful raw harmonic fits: **{raw_count:,}**.",
                f"- Production-quality harmonic fits: **{quality_count:,}** "
                f"({_format_percent(quality_count, raw_count)} of raw fits; "
                f"{_format_percent(quality_count, int(np.count_nonzero(physical_cells)))} of physical 3D cells).",
                f"- Physical 3D cells without a raw fit: **{int(np.count_nonzero(physical_cells & ~fit_success)):,}**.",
                "",
                "## Cross-section numerical distributions",
                "",
                "All entries in this section are restricted to `final_validity_mask`.",
                "",
                "| Distribution | Minimum | 10th percentile | Median | 90th percentile | Maximum |",
                "|---|---:|---:|---:|---:|---:|",
                _quantile_row("Reduced cross section", values[final], "nb/(GeV^2 rad)"),
                _quantile_row("Propagated uncertainty", errors[final], "nb/(GeV^2 rad)"),
            ]
        )
        relative_uncertainty = np.divide(
            errors,
            np.abs(values),
            out=np.full(values.shape, np.nan),
            where=final & (np.abs(values) > 0.0),
        )
        lines.extend(
            [
                _quantile_row("Relative uncertainty", relative_uncertainty[final]),
                "",
                "| Relative-uncertainty threshold | Final-valid bins above threshold |",
                "|---|---:|",
            ]
        )
        for threshold in (0.25, 0.5, 1.0):
            count = int(np.count_nonzero(final & (relative_uncertainty > threshold)))
            lines.append(f"| > {threshold:.0%} | {count:,} / {int(np.count_nonzero(final)):,} ({_format_percent(count, int(np.count_nonzero(final)))}) |")
        lines.extend(
            [
                "",
                "Largest relative uncertainties among final-valid bins:",
                "",
                *_largest_relative_uncertainties(values, errors, final, binning),
                "",
                "## Harmonic-fit quality",
                "",
                "Quality flags can overlap, so the rejection counts below are not additive. "
                "Except for `fit_failed`, counts are evaluated among numerically successful raw fits.",
                "",
                "| Requirement | Configured value |",
                "|---|---:|",
                f"| Minimum phi points | {int(_scalar(harmonics, 'quality_minimum_points', 4))} |",
                f"| Maximum chi2/ndf | {_format_number(_scalar(harmonics, 'quality_maximum_chi2_ndf', np.nan))} |",
                f"| Maximum covariance condition number | {_format_number(_scalar(harmonics, 'quality_maximum_covariance_condition', np.nan))} |",
                f"| Maximum sigma_A / abs(A) | {_format_number(_scalar(harmonics, 'quality_maximum_relative_A_uncertainty', np.nan))} |",
                f"| Nonnegative fitted cross section required | {bool(_scalar(harmonics, 'quality_requires_nonnegative', False))} |",
                "",
                "| Quality reason | Flagged cells | Reference population | Fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        status = np.asarray(
            harmonics["quality_status"]
            if "quality_status" in harmonics.files
            else np.zeros(harmonic_shape, dtype=np.uint16),
            dtype=np.uint16,
        )
        reason_names = (
            [str(item) for item in harmonics["quality_reason_names"]]
            if "quality_reason_names" in harmonics.files
            else []
        )
        reason_bits = (
            np.asarray(harmonics["quality_reason_bits"], dtype=np.uint16)
            if "quality_reason_bits" in harmonics.files
            else np.empty(0, dtype=np.uint16)
        )
        reason_counts: dict[str, int] = {}
        for name, bit in zip(reason_names, reason_bits):
            flagged = (status & bit) != 0
            count = int(np.count_nonzero(flagged if name == "fit_failed" else (flagged & fit_success)))
            reason_counts[name] = count
            denominator = fit_success.size if name == "fit_failed" else raw_count
            reference = "configured 3D cells" if name == "fit_failed" else "raw fits"
            lines.append(
                f"| `{name}` | {count:,} | {reference} ({denominator:,}) | "
                f"{_format_percent(count, denominator)} |"
            )
        lines.extend(
            [
                "",
                "| Distribution among raw fits | Minimum | 10th percentile | Median | 90th percentile | Maximum |",
                "|---|---:|---:|---:|---:|---:|",
                _quantile_row("Phi points", points[fit_success]),
                _quantile_row("chi2/ndf", chi2_ndf[fit_success]),
                _quantile_row("sigma_A / abs(A)", relative_a[fit_success]),
                "",
            ]
        )

        response_metrics: dict[str, int | float] = {}
        if response_meta is not None:
            response_lines, response_metrics = _response_section(
                response_meta, masks, binning
            )
            lines.extend(response_lines)

        physical_count = int(np.count_nonzero(physical))
        physical_acceptance = int(np.count_nonzero(physical & masks["acceptance"]))
        correction_common = physical & masks["acceptance"] & masks["radiative"] & masks["bin_centering"]
        correction_common_count = int(np.count_nonzero(correction_common))
        pre_uncertainty = correction_common & masks["yield"]
        uncertainty_loss = int(np.count_nonzero(pre_uncertainty & ~masks["uncertainty"]))
        lines.extend(
            [
                "## Automated triage",
                "",
                "These statements identify where attrition occurs; they do not decide whether "
                "a cut, correction, or binning change is physically justified.",
                "",
                f"- **Structural phase space:** {total - physical_count:,} / {total:,} 4D bins are outside positive physical normalization space. They should not be interpreted as lost data.",
                f"- **Acceptance:** {physical_count - physical_acceptance:,} / {physical_count:,} physical bins fail the acceptance requirement.",
                f"- **Correction overlap:** after physical normalization and acceptance, {physical_acceptance - correction_common_count:,} additional bins fail radiative or bin-centering reliability.",
                f"- **Propagated statistical support:** {uncertainty_loss:,} otherwise eligible bins fail the finite-positive uncertainty requirement.",
                f"- **Harmonic robustness:** {raw_count - quality_count:,} / {raw_count:,} raw fits fail at least one production-quality requirement.",
            ]
        )
        if response_metrics:
            lines.append(
                f"- **Response-MC support heuristic:** among physical acceptance-failed bins, "
                f"{response_metrics['acceptance_failed_truth_lt100']:,} have fewer than 100 truth-weight "
                f"units and {response_metrics['acceptance_failed_truth_ge100']:,} have at least "
                "100. The former are plausible MC-statistics candidates; the latter merit "
                "acceptance, reconstruction, migration, and selection inspection."
            )
        if reason_counts:
            nonfailure = {
                name: count for name, count in reason_counts.items()
                if name != "fit_failed" and count > 0
            }
            if nonfailure:
                leading = sorted(nonfailure.items(), key=lambda item: item[1], reverse=True)[:3]
                lines.append(
                    "- **Leading harmonic flags:** "
                    + ", ".join(f"`{name}` ({count:,})" for name, count in leading)
                    + ". Because flags overlap, this is a prioritization aid rather than an additive decomposition."
                )
        if response_meta is None:
            lines.append(
                "- **Response-MC attribution unavailable:** rerun this report with "
                "`--response-meta` to separate low generated support from well-populated, "
                "low-efficiency bins."
            )

        lines.extend(
            [
                "",
                "## Analyst sign-off checklist",
                "",
                "- [ ] Verify that the unfolding, cross-section, harmonic, response, and correction artifacts belong to the same campaign configuration and software revision.",
                "- [ ] Inspect the response truth, reconstructed, efficiency, migration, and relative-MC-uncertainty maps in the bins highlighted above.",
                "- [ ] Inspect the largest relative-uncertainty cross-section bins and decide whether they should remain individually resolved.",
                "- [ ] Inspect raw harmonic fits rejected by the dominant quality flags; record whether the limitation is phi sparsity, uncertainty, covariance, shape, or model inadequacy.",
                "- [ ] Repeat the full extraction for declared systematic variations and compare only bins/cells valid in the relevant common mask.",
                "- [ ] Record the final binning, thresholds, exclusions, exceptions, and reviewer sign-off below.",
                "",
                "### Analyst notes",
                "",
                "- Campaign decision: _pending_",
                "- Binning decision: _pending_",
                "- Accepted exceptions: _none recorded_",
                "- Reviewer/date: _pending_",
                "",
            ]
        )
        return "\n".join(lines)
    finally:
        unfolding.close()
        cross_section.close()
        harmonics.close()
        if response_meta is not None:
            response_meta.close()
