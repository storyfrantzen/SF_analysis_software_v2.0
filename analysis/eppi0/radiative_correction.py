from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .binning import AnalysisBinning
from .bin_centering import PhysicalBinReference, compute_physical_bin_reference
from .cross_section import physical_bin_volumes, virtual_photon_flux
from .event_sample import (
    ELECTRON_MASS,
    PI0_MASS,
    PROTON_MASS,
    _dis,
    _minus_t,
    _trento_phi,
)
from .exclusive_kinematics import physical_mask
from .generators import GeneratorCampaign, GeneratorRunRecord
from .phase_space import AnalysisPhaseSpace


Array = np.ndarray
LUND_TEXT_SUFFIXES = {".lund", ".txt"}
CLUSTER_UNCERTAINTY_MODEL = "consecutive-exact-lund-clusters-v1"
EXCLUGEN_UNCERTAINTY_MODEL = "independent-run-standard-error-v1"
EXCLUGEN_NORMALIZATION_METHOD = (
    "per-run sigma_nb * selected_bin_events / generated_events, averaged over runs"
)
EXCLUGEN_V_DEFINITION = "MX2(e'p) - m_pi0^2"


@dataclass(frozen=True)
class LundHistogramResult:
    counts: Array
    cluster_weight_square_sum: Array
    effective_counts: Array
    q2_min: Array
    q2_max: Array
    eprime_min: Array
    eprime_max: Array
    files: int
    events_seen: int
    topology_events: int
    topology_clusters: int
    in_range: int
    in_range_clusters: int
    maximum_cluster_multiplicity: int
    generated_q2_min: float
    generated_q2_max: float
    generated_eprime_min: float
    generated_eprime_max: float


@dataclass(frozen=True)
class RadiativeCorrectionResult:
    c_rad: Array
    delta_c: Array
    c_rad_raw: Array
    delta_c_raw: Array
    reliable: Array
    support_overlap: Array
    support_status: Array
    born: LundHistogramResult
    radiative: LundHistogramResult
    normalization_ratio: float
    born_integrated_cross_section: float | None = None
    radiative_integrated_cross_section: float | None = None


@dataclass(frozen=True)
class ExclugenModeEstimate:
    """Cross-section estimate and support accumulated from independent runs."""

    cross_section: Array
    standard_error: Array
    counts: Array
    effective_counts: Array
    run_estimates: Array
    records: tuple[GeneratorRunRecord, ...]
    generated_events: int
    selected_events: int
    in_range_events: int


@dataclass(frozen=True)
class ExclugenRadiativeCorrectionResult:
    """Analysis-binned EXCLUGEN radiative correction with full provenance."""

    c_rad: Array
    delta_c: Array
    c_rad_raw: Array
    delta_c_raw: Array
    reliable: Array
    support_overlap: Array
    support_status: Array
    born: ExclugenModeEstimate
    radiative: ExclugenModeEstimate
    campaign: GeneratorCampaign
    normalization_ratio: float
    born_integrated_cross_section: float
    radiative_integrated_cross_section: float
    v_max_GeV2: float


@dataclass(frozen=True)
class ExclugenCombinedCorrectionResult:
    """Generated ``R1`` mapped directly to Born model reference points.

    ``C_rad`` remains the downstream-compatible field name, but the stored
    quantity includes both the radiative and bin-centering transforms:

    ``C_RBC = R1_bin / (V_physical * Gamma(reference) * B0_reduced(reference))``.
    """

    correction: Array
    uncertainty: Array
    correction_raw: Array
    uncertainty_raw: Array
    reliable: Array
    support_status: Array
    radiative: ExclugenModeEstimate
    campaign: GeneratorCampaign
    reference: PhysicalBinReference
    physical_volume: Array
    born_center_reduced: Array
    born_center_electron: Array
    radiative_bin_average: Array
    v_max_GeV2: float


@dataclass(frozen=True)
class CombinedCorrectionClosure:
    """Algebraic decomposition of the combined correction through ``<B0>``."""

    direct: Array
    radiative_over_born_average: Array
    born_average_over_center: Array
    product: Array
    residual: Array
    reliable: Array


@dataclass(frozen=True)
class ExclugenCorrectionRatio:
    """One cross-section ratio in the four-sample closure matrix."""

    correction: Array
    uncertainty: Array
    correction_raw: Array
    uncertainty_raw: Array
    reliable: Array
    support_overlap: Array
    support_status: Array
    numerator: str
    denominator: str


@dataclass(frozen=True)
class ExclugenRadiativeClosureResult:
    """Four-sample EXCLUGEN correction closure for ``itarg=0`` and ``itarg=1``."""

    estimates: Mapping[str, ExclugenModeEstimate]
    ratios: Mapping[str, ExclugenCorrectionRatio]
    campaign_itarg0: GeneratorCampaign
    campaign_itarg1: GeneratorCampaign
    v_max_GeV2: float


EXCLUGEN_CLOSURE_RATIO_DEFINITIONS = {
    "total": ("rad_itarg1", "born_itarg0"),
    "internal_external_folded": ("rad_itarg1", "born_itarg1"),
    "external_born": ("born_itarg1", "born_itarg0"),
    "internal_no_external": ("rad_itarg0", "born_itarg0"),
    "external_radiative": ("rad_itarg1", "rad_itarg0"),
}


@dataclass(frozen=True)
class DirectInternalRadiativeResult:
    """Direct bin integrals of the ``itarg=0`` Born and observed models."""

    correction: Array
    correction_raw: Array
    reliable: Array
    computed: Array
    born_integral_nb: Array
    observed_integral_nb: Array
    n_physical: Array
    n_valid: Array
    n_failed: Array
    failure_fraction: Array
    samples_per_dimension: int
    bin_start: int
    bin_stop: int


InternalRadiativeEvaluator = Callable[[Array], tuple[Array, Array]]


def compute_direct_internal_radiative_correction(
    binning: AnalysisBinning,
    beam_energy: float,
    evaluator: InternalRadiativeEvaluator,
    *,
    samples_per_dimension: int = 1,
    max_failure_fraction: float = 0.0,
    bin_start: int = 0,
    bin_stop: int | None = None,
    progress_bins: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
) -> DirectInternalRadiativeResult:
    """Integrate ``R0/B0`` directly over finite analysis bins.

    The evaluator supplies cross sections differential in ``Q2, W2, t, phi``.
    The integral is performed in the analysis coordinates ``Q2, xB, -t,
    phi`` and therefore includes ``abs(dW2/dxB) = Q2/xB**2``.  This function
    deliberately contains no pre-vertex target-radiation convolution.
    """
    if samples_per_dimension <= 0:
        raise ValueError("samples_per_dimension must be positive")
    if not 0.0 <= max_failure_fraction <= 1.0:
        raise ValueError("max_failure_fraction must be between 0 and 1")
    if progress_bins < 0:
        raise ValueError("progress_bins must be non-negative")
    total_bins = binning.size
    if bin_stop is None:
        bin_stop = total_bins
    if bin_start < 0 or bin_stop < bin_start or bin_stop > total_bins:
        raise ValueError(
            f"invalid 4D bin range [{bin_start}, {bin_stop}) for {total_bins} bins"
        )

    shape = binning.shape
    correction_raw = np.full(shape, np.nan, dtype=float)
    correction = np.ones(shape, dtype=float)
    reliable = np.zeros(shape, dtype=bool)
    computed = np.zeros(shape, dtype=bool)
    born_integral = np.full(shape, np.nan, dtype=float)
    observed_integral = np.full(shape, np.nan, dtype=float)
    n_physical = np.zeros(shape, dtype=np.int64)
    n_valid = np.zeros(shape, dtype=np.int64)
    n_failed = np.zeros(shape, dtype=np.int64)
    failure_fraction = np.ones(shape, dtype=float)

    edge_sets = (
        binning.q2_edges,
        binning.xb_edges,
        binning.t_edges,
        binning.phi_edges,
    )
    processed = 0
    assigned = bin_stop - bin_start
    for index in np.ndindex(shape):
        flat_index = int(np.ravel_multi_index(index, shape))
        if flat_index < bin_start or flat_index >= bin_stop:
            continue
        computed[index] = True
        processed += 1
        points_1d = [
            _midpoint_grid(edges[position], edges[position + 1], samples_per_dimension)
            for edges, position in zip(edge_sets, index)
        ]
        q2, xb, minus_t, phi_deg = np.meshgrid(*points_1d, indexing="ij")
        signed_t = -minus_t
        physical = physical_mask(xb, q2, signed_t, beam_energy)
        if phase_space is not None and phase_space.enabled:
            physical &= phase_space.mask(q2, xb, beam_energy)
        n_physical[index] = int(np.count_nonzero(physical))
        if n_physical[index] > 0:
            evaluation_points = np.column_stack(
                (xb[physical], q2[physical], signed_t[physical], phi_deg[physical])
            )
            born_values, observed_values = evaluator(evaluation_points)
            born_values = np.asarray(born_values, dtype=float)
            observed_values = np.asarray(observed_values, dtype=float)
            expected_shape = (evaluation_points.shape[0],)
            if born_values.shape != expected_shape or observed_values.shape != expected_shape:
                raise ValueError(
                    "direct radiative evaluator returned incompatible shapes: "
                    f"born={born_values.shape}, observed={observed_values.shape}, "
                    f"expected={expected_shape}"
                )
            valid = (
                np.isfinite(born_values)
                & np.isfinite(observed_values)
                & (born_values > 0.0)
                & (observed_values > 0.0)
            )
            n_valid[index] = int(np.count_nonzero(valid))
            n_failed[index] = n_physical[index] - n_valid[index]
            failure_fraction[index] = n_failed[index] / n_physical[index]
            if np.any(valid):
                jacobian = (
                    evaluation_points[valid, 1]
                    / np.square(evaluation_points[valid, 0])
                )
                widths = [
                    float(edges[position + 1] - edges[position])
                    for edges, position in zip(edge_sets, index)
                ]
                cell_volume = (
                    widths[0]
                    * widths[1]
                    * widths[2]
                    * np.deg2rad(widths[3])
                    / samples_per_dimension**4
                )
                born_integral[index] = float(
                    cell_volume * np.sum(born_values[valid] * jacobian)
                )
                observed_integral[index] = float(
                    cell_volume * np.sum(observed_values[valid] * jacobian)
                )
                if born_integral[index] > 0.0:
                    correction_raw[index] = (
                        observed_integral[index] / born_integral[index]
                    )
                    reliable[index] = (
                        np.isfinite(correction_raw[index])
                        and correction_raw[index] > 0.0
                        and failure_fraction[index] <= max_failure_fraction
                    )
                    if reliable[index]:
                        correction[index] = correction_raw[index]
        if progress_bins > 0 and (
            processed % progress_bins == 0 or processed == assigned
        ):
            print(
                f"[PROGRESS] direct internal-RC bins {processed}/{assigned} "
                f"({100.0 * processed / max(assigned, 1):.1f}%), "
                f"reliable={int(np.count_nonzero(reliable & computed))}"
            )

    return DirectInternalRadiativeResult(
        correction=correction,
        correction_raw=correction_raw,
        reliable=reliable,
        computed=computed,
        born_integral_nb=born_integral,
        observed_integral_nb=observed_integral,
        n_physical=n_physical,
        n_valid=n_valid,
        n_failed=n_failed,
        failure_fraction=failure_fraction,
        samples_per_dimension=samples_per_dimension,
        bin_start=bin_start,
        bin_stop=bin_stop,
    )


def _midpoint_grid(low: float, high: float, points: int) -> Array:
    edges = np.linspace(float(low), float(high), int(points) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def compute_exclugen_radiative_closure(
    campaign_itarg0: GeneratorCampaign,
    campaign_itarg1: GeneratorCampaign,
    binning: AnalysisBinning,
    *,
    v_max: float = 0.2,
    min_counts: int = 5,
    max_files: int | None = None,
    progress_runs: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
) -> ExclugenRadiativeClosureResult:
    """Evaluate the matched ``B0``, ``R0``, ``B1``, and ``R1`` ratio matrix.

    ``B`` and ``R`` denote Born and full-RC samples; the suffix is the EXCLUGEN
    ``itarg`` setting.  The leading total-correction candidate is ``R1/B0``.
    All mode estimates retain their independent per-run cross-section
    normalization.  The radiative-only offline ``v`` cut is applied to both
    ``R0`` and ``R1`` and never renormalized away.
    """
    if not np.isfinite(v_max):
        raise ValueError("v_max must be finite")
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when provided")
    if progress_runs < 0:
        raise ValueError("progress_runs must be non-negative")
    _validate_closure_campaigns(campaign_itarg0, campaign_itarg1)

    estimates = {
        "born_itarg0": estimate_exclugen_mode(
            campaign_itarg0,
            "born",
            binning,
            v_max=v_max,
            max_files=max_files,
            progress_runs=progress_runs,
            phase_space=phase_space,
        ),
        "rad_itarg0": estimate_exclugen_mode(
            campaign_itarg0,
            "rad",
            binning,
            v_max=v_max,
            max_files=max_files,
            progress_runs=progress_runs,
            phase_space=phase_space,
        ),
        "born_itarg1": estimate_exclugen_mode(
            campaign_itarg1,
            "born",
            binning,
            v_max=v_max,
            max_files=max_files,
            progress_runs=progress_runs,
            phase_space=phase_space,
        ),
        "rad_itarg1": estimate_exclugen_mode(
            campaign_itarg1,
            "rad",
            binning,
            v_max=v_max,
            max_files=max_files,
            progress_runs=progress_runs,
            phase_space=phase_space,
        ),
    }
    ratios = {
        name: _exclugen_estimate_ratio(
            estimates[numerator],
            estimates[denominator],
            binning,
            min_counts=min_counts,
            numerator=numerator,
            denominator=denominator,
        )
        for name, (numerator, denominator) in EXCLUGEN_CLOSURE_RATIO_DEFINITIONS.items()
    }
    return ExclugenRadiativeClosureResult(
        estimates=estimates,
        ratios=ratios,
        campaign_itarg0=campaign_itarg0,
        campaign_itarg1=campaign_itarg1,
        v_max_GeV2=float(v_max),
    )


def _exclugen_estimate_ratio(
    numerator_estimate: ExclugenModeEstimate,
    denominator_estimate: ExclugenModeEstimate,
    binning: AnalysisBinning,
    *,
    min_counts: int,
    numerator: str,
    denominator: str,
) -> ExclugenCorrectionRatio:
    ratio_flat, uncertainty_flat = _ratio_and_error(
        numerator_estimate.cross_section,
        numerator_estimate.standard_error,
        denominator_estimate.cross_section,
        denominator_estimate.standard_error,
    )
    support_overlap = (
        (numerator_estimate.counts > 0.0)
        & (denominator_estimate.counts > 0.0)
    )
    reliable_flat = (
        support_overlap
        & (numerator_estimate.effective_counts >= min_counts)
        & (denominator_estimate.effective_counts >= min_counts)
        & np.isfinite(ratio_flat)
        & np.isfinite(uncertainty_flat)
    )
    support_status = support_status_codes(
        denominator_estimate.counts,
        numerator_estimate.counts,
        min_counts,
        born_effective_counts=denominator_estimate.effective_counts,
        radiative_effective_counts=numerator_estimate.effective_counts,
    )
    return ExclugenCorrectionRatio(
        correction=binning.unflatten(np.where(reliable_flat, ratio_flat, 1.0)),
        uncertainty=binning.unflatten(
            np.where(reliable_flat, uncertainty_flat, 1.0)
        ),
        correction_raw=binning.unflatten(ratio_flat),
        uncertainty_raw=binning.unflatten(uncertainty_flat),
        reliable=binning.unflatten(reliable_flat),
        support_overlap=binning.unflatten(support_overlap),
        support_status=binning.unflatten(support_status),
        numerator=numerator,
        denominator=denominator,
    )


def _validate_closure_campaigns(
    campaign_itarg0: GeneratorCampaign,
    campaign_itarg1: GeneratorCampaign,
) -> None:
    for label, campaign, expected_itarg in (
        ("itarg=0", campaign_itarg0, 0),
        ("itarg=1", campaign_itarg1, 1),
    ):
        if campaign.provider != "exclugen":
            raise ValueError(f"{label} closure campaign is not EXCLUGEN")
        values = {_record_itarg(record) for record in campaign.runs}
        if values != {expected_itarg}:
            raise ValueError(
                f"{label} closure campaign has physics-card itarg values "
                f"{sorted(values)}"
            )
        for mode in ("born", "rad"):
            if not campaign.runs_for_mode(mode):
                raise ValueError(f"{label} closure campaign has no {mode} runs")

    for field_name in ("provider", "channel", "model"):
        if getattr(campaign_itarg0, field_name) != getattr(campaign_itarg1, field_name):
            raise ValueError(
                f"closure campaigns have different {field_name}: "
                f"{getattr(campaign_itarg0, field_name)!r} versus "
                f"{getattr(campaign_itarg1, field_name)!r}"
            )
    for field_name in (
        "generator_revision",
        "dirty_state",
        "execution_path",
        "beam_energy_GeV",
        "generation_vcut_GeV2",
        "t_nucl_GeV2",
    ):
        values0 = {getattr(record, field_name) for record in campaign_itarg0.runs}
        values1 = {getattr(record, field_name) for record in campaign_itarg1.runs}
        if values0 != values1:
            raise ValueError(
                f"closure campaigns have different {field_name}: "
                f"{sorted(values0)} versus {sorted(values1)}"
            )

    card0 = _closure_card(campaign_itarg0.runs[0].physics_card)
    card1 = _closure_card(campaign_itarg1.runs[0].physics_card)
    if card0 != card1:
        differing = sorted(
            key for key in set(card0) | set(card1) if card0.get(key) != card1.get(key)
        )
        raise ValueError(
            "closure campaign physics cards differ beyond itarg: "
            + ", ".join(differing)
        )


def _record_itarg(record: GeneratorRunRecord) -> int:
    value = record.physics_card.get("itarg")
    if isinstance(value, bool):
        raise ValueError(f"{record.config_path}: physics-card itarg is not 0 or 1")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{record.config_path}: physics-card itarg is not 0 or 1"
        ) from error
    if result not in (0, 1) or float(value) != float(result):
        raise ValueError(f"{record.config_path}: physics-card itarg is not 0 or 1")
    return result


def _closure_card(card: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in card.items() if key != "itarg"}


def compute_exclugen_radiative_correction(
    campaign: GeneratorCampaign,
    binning: AnalysisBinning,
    *,
    v_max: float = 0.2,
    min_counts: int = 5,
    max_files: int | None = None,
    progress_runs: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
) -> ExclugenRadiativeCorrectionResult:
    """Compute ``C_rad = sigma_rad(v < v_max) / sigma_Born``.

    Each LUND run is an independent cross-section estimate.  For run ``i`` and
    bin ``b`` the estimate is ``sigma_i * n_i,b / N_i``; the central value is
    the mean across runs and its uncertainty is the standard error across those
    independent estimates.  The offline inelasticity cut is applied only to the
    radiative sample, and surviving events are deliberately not renormalized.
    """
    if campaign.provider != "exclugen":
        raise ValueError(
            "compute_exclugen_radiative_correction requires an EXCLUGEN campaign"
        )
    if not np.isfinite(v_max):
        raise ValueError("v_max must be finite")
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when provided")
    if progress_runs < 0:
        raise ValueError("progress_runs must be non-negative")

    born = estimate_exclugen_mode(
        campaign,
        "born",
        binning,
        v_max=v_max,
        max_files=max_files,
        progress_runs=progress_runs,
        phase_space=phase_space,
    )
    radiative = estimate_exclugen_mode(
        campaign,
        "rad",
        binning,
        v_max=v_max,
        max_files=max_files,
        progress_runs=progress_runs,
        phase_space=phase_space,
    )

    c_rad_flat, delta_flat = _ratio_and_error(
        radiative.cross_section,
        radiative.standard_error,
        born.cross_section,
        born.standard_error,
    )
    support_overlap = (born.counts > 0.0) & (radiative.counts > 0.0)
    reliable_flat = (
        support_overlap
        & (born.effective_counts >= min_counts)
        & (radiative.effective_counts >= min_counts)
        & np.isfinite(c_rad_flat)
        & np.isfinite(delta_flat)
    )
    c_rad_safe = np.where(reliable_flat, c_rad_flat, 1.0)
    delta_safe = np.where(reliable_flat, delta_flat, 1.0)
    support_status = support_status_codes(
        born.counts,
        radiative.counts,
        min_counts,
        born_effective_counts=born.effective_counts,
        radiative_effective_counts=radiative.effective_counts,
    )
    born_sigma = float(np.mean([record.sigma_value for record in born.records]))
    radiative_sigma = float(
        np.mean([record.sigma_value for record in radiative.records])
    )
    return ExclugenRadiativeCorrectionResult(
        c_rad=binning.unflatten(c_rad_safe),
        delta_c=binning.unflatten(delta_safe),
        c_rad_raw=binning.unflatten(c_rad_flat),
        delta_c_raw=binning.unflatten(delta_flat),
        reliable=binning.unflatten(reliable_flat),
        support_overlap=binning.unflatten(support_overlap),
        support_status=binning.unflatten(support_status),
        born=born,
        radiative=radiative,
        campaign=campaign,
        normalization_ratio=radiative_sigma / born_sigma,
        born_integrated_cross_section=born_sigma,
        radiative_integrated_cross_section=radiative_sigma,
        v_max_GeV2=float(v_max),
    )


def compute_exclugen_combined_correction(
    campaign: GeneratorCampaign,
    binning: AnalysisBinning,
    beam_energy: float,
    born_evaluator: Callable[[Array], Array],
    *,
    v_max: float = 0.2,
    min_counts: int = 5,
    max_files: int | None = None,
    progress_runs: int = 0,
    reference_samples_per_dimension: int = 20,
    phase_space: AnalysisPhaseSpace | None = None,
) -> ExclugenCombinedCorrectionResult:
    """Map generated ``R1`` bin averages to Born values at physical centroids.

    Only radiative ``itarg=1`` records are consumed.  The model is evaluated
    once per physical four-dimensional bin, at a model-independent physical-
    volume centroid.  No generated or quadrature-integrated Born denominator
    enters the nominal correction.
    """
    if campaign.provider != "exclugen":
        raise ValueError("combined correction requires an EXCLUGEN campaign")
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    if reference_samples_per_dimension <= 0:
        raise ValueError("reference_samples_per_dimension must be positive")
    radiative_records = campaign.runs_for_mode("rad")
    if not radiative_records:
        raise ValueError("EXCLUGEN campaign has no radiative runs")
    itarg_values = {_record_itarg(record) for record in radiative_records}
    if itarg_values != {1}:
        raise ValueError(
            "combined correction requires radiative itarg=1 records; "
            f"found {sorted(itarg_values)}"
        )

    phase_space = phase_space or AnalysisPhaseSpace()
    radiative = estimate_exclugen_mode(
        campaign,
        "rad",
        binning,
        v_max=v_max,
        max_files=max_files,
        progress_runs=progress_runs,
        phase_space=phase_space,
    )
    reference = compute_physical_bin_reference(
        binning,
        beam_energy,
        samples_per_dimension=reference_samples_per_dimension,
        phase_space=phase_space,
    )
    physical_volume = physical_bin_volumes(
        binning,
        beam_energy,
        q2_minimum=1.0 if phase_space.q2_min is None else phase_space.q2_min,
        w_minimum=2.0 if phase_space.w_min is None else phase_space.w_min,
        electron_p_minimum=(
            0.0
            if phase_space.electron_p_min is None
            else phase_space.electron_p_min
        ),
    )

    valid_reference = (
        (reference.n_physical > 0)
        & np.isfinite(reference.xB_center)
        & np.isfinite(reference.q2_center)
        & np.isfinite(reference.minus_t_center)
        & np.isfinite(reference.phi_center)
        & np.isfinite(physical_volume)
        & (physical_volume > 0.0)
    )
    center_reduced = np.full(binning.shape, np.nan, dtype=float)
    if np.any(valid_reference):
        points = np.column_stack(
            (
                reference.xB_center[valid_reference],
                reference.q2_center[valid_reference],
                -reference.minus_t_center[valid_reference],
                reference.phi_center[valid_reference],
            )
        )
        values = np.asarray(born_evaluator(points), dtype=float)
        expected_shape = (points.shape[0],)
        if values.shape != expected_shape:
            raise ValueError(
                f"Born evaluator returned shape {values.shape}; expected {expected_shape}"
            )
        center_reduced[valid_reference] = values

    flux = virtual_photon_flux(
        reference.q2_center, reference.xB_center, beam_energy
    )
    center_electron = flux * center_reduced
    denominator = physical_volume * center_electron
    radiative_cross_section = binning.unflatten(radiative.cross_section)
    radiative_sem = binning.unflatten(radiative.standard_error)
    radiative_counts = binning.unflatten(radiative.counts)
    radiative_effective = binning.unflatten(radiative.effective_counts)
    correction_raw = np.divide(
        radiative_cross_section,
        denominator,
        out=np.full(binning.shape, np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )
    uncertainty_raw = np.divide(
        radiative_sem,
        denominator,
        out=np.full(binning.shape, np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )
    radiative_bin_average = np.divide(
        radiative_cross_section,
        physical_volume,
        out=np.full(binning.shape, np.nan, dtype=float),
        where=np.isfinite(physical_volume) & (physical_volume > 0.0),
    )
    reliable = (
        valid_reference
        & (radiative_counts > 0.0)
        & (radiative_effective >= min_counts)
        & np.isfinite(center_reduced)
        & (center_reduced > 0.0)
        & np.isfinite(correction_raw)
        & (correction_raw > 0.0)
        & np.isfinite(uncertainty_raw)
        & (uncertainty_raw >= 0.0)
    )
    support_status = np.zeros(binning.shape, dtype=np.uint8)
    support_status[radiative_counts <= 0.0] = 1
    support_status[
        (radiative_counts > 0.0) & (radiative_effective < min_counts)
    ] = 2
    support_status[(radiative_counts > 0.0) & ~valid_reference] = 3
    support_status[
        (radiative_counts > 0.0)
        & valid_reference
        & (~np.isfinite(center_reduced) | (center_reduced <= 0.0))
    ] = 4

    return ExclugenCombinedCorrectionResult(
        correction=np.where(reliable, correction_raw, 1.0),
        uncertainty=np.where(reliable, uncertainty_raw, 1.0),
        correction_raw=correction_raw,
        uncertainty_raw=uncertainty_raw,
        reliable=reliable,
        support_status=support_status,
        radiative=radiative,
        campaign=campaign,
        reference=reference,
        physical_volume=physical_volume,
        born_center_reduced=center_reduced,
        born_center_electron=center_electron,
        radiative_bin_average=radiative_bin_average,
        v_max_GeV2=float(v_max),
    )


def combined_correction_closure(
    radiative_bin_cross_section: Array,
    born_bin_cross_section: Array,
    physical_volume: Array,
    born_center_electron: Array,
) -> CombinedCorrectionClosure:
    """Check ``<R1>/B0c = (<R1>/<B0>)*(<B0>/B0c)`` bin by bin."""
    radiative_bin_cross_section, born_bin_cross_section, physical_volume, born_center_electron = (
        np.broadcast_arrays(
            np.asarray(radiative_bin_cross_section, dtype=float),
            np.asarray(born_bin_cross_section, dtype=float),
            np.asarray(physical_volume, dtype=float),
            np.asarray(born_center_electron, dtype=float),
        )
    )
    center_integral = physical_volume * born_center_electron
    result_shape = radiative_bin_cross_section.shape
    direct = np.divide(
        radiative_bin_cross_section,
        center_integral,
        out=np.full(result_shape, np.nan),
        where=np.isfinite(center_integral) & (center_integral > 0.0),
    )
    radiative_over_born = np.divide(
        radiative_bin_cross_section,
        born_bin_cross_section,
        out=np.full(result_shape, np.nan),
        where=np.isfinite(born_bin_cross_section) & (born_bin_cross_section > 0.0),
    )
    born_average_over_center = np.divide(
        born_bin_cross_section,
        center_integral,
        out=np.full(result_shape, np.nan),
        where=np.isfinite(center_integral) & (center_integral > 0.0),
    )
    product = radiative_over_born * born_average_over_center
    residual = direct - product
    reliable = (
        np.isfinite(direct)
        & np.isfinite(radiative_over_born)
        & np.isfinite(born_average_over_center)
        & np.isfinite(product)
    )
    return CombinedCorrectionClosure(
        direct=direct,
        radiative_over_born_average=radiative_over_born,
        born_average_over_center=born_average_over_center,
        product=product,
        residual=residual,
        reliable=reliable,
    )


def estimate_exclugen_mode(
    campaign: GeneratorCampaign,
    mode: str,
    binning: AnalysisBinning,
    *,
    v_max: float,
    max_files: int | None = None,
    progress_runs: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
) -> ExclugenModeEstimate:
    """Estimate one EXCLUGEN mode without post-selection renormalization."""
    records = campaign.runs_for_mode(mode)
    if max_files is not None:
        records = records[:max_files]
    if not records:
        raise ValueError(f"EXCLUGEN campaign has no {mode!r} runs")

    estimates: list[Array] = []
    counts_total = np.zeros(binning.size, dtype=float)
    sum_weights = np.zeros(binning.size, dtype=float)
    sum_weight_squares = np.zeros(binning.size, dtype=float)
    generated_total = 0
    selected_total = 0
    in_range_total = 0

    for position, record in enumerate(records, start=1):
        electron, proton, observed = read_exclugen_lund(record.lund_path)
        if observed != record.observed_events or observed != record.requested_events:
            raise ValueError(
                f"{record.lund_path}: observed {observed} events, provenance records "
                f"observed={record.observed_events} and requested={record.requested_events}"
            )
        q2, xb = _dis(electron, record.beam_energy_GeV)
        minus_t = _minus_t(proton)
        phi = _trento_phi(electron, proton, record.beam_energy_GeV)
        flat = binning.coordinates_to_flat(q2, xb, minus_t, phi)
        selected = (
            np.isfinite(q2)
            & np.isfinite(xb)
            & np.isfinite(minus_t)
            & np.isfinite(phi)
        )
        if mode == "rad":
            inelasticity = exclugen_inelasticity_v(
                electron, proton, record.beam_energy_GeV
            )
            selected &= np.isfinite(inelasticity) & (inelasticity < v_max)
        selected_total += int(np.count_nonzero(selected))
        if phase_space is not None and phase_space.enabled:
            selected &= phase_space.mask(q2, xb, record.beam_energy_GeV)
        inside = selected & (flat >= 0) & (flat < binning.size)
        counts = np.bincount(flat[inside], minlength=binning.size).astype(float)
        event_weight = record.sigma_value / record.requested_events
        estimates.append(event_weight * counts)
        counts_total += counts
        sum_weights += event_weight * counts
        sum_weight_squares += event_weight**2 * counts
        generated_total += observed
        in_range_total += int(np.count_nonzero(inside))
        if progress_runs > 0 and (
            position % progress_runs == 0 or position == len(records)
        ):
            print(
                f"[EXCLUGEN] {mode}: estimated {position}/{len(records)} runs, "
                f"generated={generated_total}, selected={selected_total}, "
                f"analysis-bin={in_range_total}"
            )

    matrix = np.asarray(estimates, dtype=float)
    return ExclugenModeEstimate(
        cross_section=np.mean(matrix, axis=0),
        standard_error=independent_run_standard_error(matrix),
        counts=counts_total,
        effective_counts=weighted_effective_count(
            sum_weights, sum_weight_squares
        ),
        run_estimates=matrix,
        records=records,
        generated_events=generated_total,
        selected_events=selected_total,
        in_range_events=in_range_total,
    )


def independent_run_standard_error(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim < 1:
        raise ValueError("independent run estimates must have a run axis")
    if values.shape[0] < 2:
        return np.full(values.shape[1:], np.nan, dtype=float)
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def weighted_effective_count(sum_weights: Array, sum_weight_squares: Array) -> Array:
    sum_weights = np.asarray(sum_weights, dtype=float)
    sum_weight_squares = np.asarray(sum_weight_squares, dtype=float)
    if sum_weights.shape != sum_weight_squares.shape:
        raise ValueError("weight-sum arrays must have matching shapes")
    return np.divide(
        np.square(sum_weights),
        sum_weight_squares,
        out=np.zeros_like(sum_weights),
        where=sum_weight_squares > 0.0,
    )


def exclugen_inelasticity_v(
    electron: Array,
    proton: Array,
    beam_energy: float,
) -> Array:
    """Return ``MX2(e'p) - m_pi0**2`` in GeV2 for EXCLUGEN events."""
    beam_pz = np.sqrt(beam_energy**2 - ELECTRON_MASS**2)
    missing_energy = beam_energy + PROTON_MASS - electron[:, 3] - proton[:, 3]
    missing_px = -electron[:, 0] - proton[:, 0]
    missing_py = -electron[:, 1] - proton[:, 1]
    missing_pz = beam_pz - electron[:, 2] - proton[:, 2]
    missing_mass2 = (
        np.square(missing_energy)
        - np.square(missing_px)
        - np.square(missing_py)
        - np.square(missing_pz)
    )
    return missing_mass2 - PI0_MASS**2


def read_exclugen_lund(path: str | Path) -> tuple[Array, Array, int]:
    """Read electron and proton four-vectors from one complete LUND file."""
    source_path = Path(path)
    electrons: list[tuple[float, float, float, float]] = []
    protons: list[tuple[float, float, float, float]] = []
    events = 0
    with source_path.open("r", errors="replace") as source:
        while True:
            header = source.readline()
            if not header:
                break
            fields = header.split()
            if not fields:
                continue
            try:
                particle_count = int(fields[0])
            except ValueError:
                continue
            events += 1
            electron = None
            proton = None
            complete = True
            for _ in range(particle_count):
                row = source.readline()
                if not row:
                    complete = False
                    break
                parts = row.split()
                if len(parts) < 10:
                    complete = False
                    continue
                try:
                    pid = int(parts[3])
                    momentum = tuple(float(parts[index]) for index in (6, 7, 8, 9))
                except ValueError:
                    complete = False
                    continue
                if pid == 11 and electron is None:
                    electron = momentum
                elif pid == 2212 and proton is None:
                    proton = momentum
            if not complete or electron is None or proton is None:
                raise ValueError(
                    f"incomplete/non-e'p event {events} in {source_path}"
                )
            electrons.append(electron)
            protons.append(proton)
    return (
        np.asarray(electrons, dtype=float).reshape((-1, 4)),
        np.asarray(protons, dtype=float).reshape((-1, 4)),
        events,
    )


def _ratio_and_error(
    numerator: Array,
    numerator_error: Array,
    denominator: Array,
    denominator_error: Array,
) -> tuple[Array, Array]:
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0.0,
    )
    relative_numerator = np.divide(
        numerator_error,
        numerator,
        out=np.full_like(numerator_error, np.nan, dtype=float),
        where=numerator != 0.0,
    )
    relative_denominator = np.divide(
        denominator_error,
        denominator,
        out=np.full_like(denominator_error, np.nan, dtype=float),
        where=denominator != 0.0,
    )
    uncertainty = ratio * np.sqrt(
        np.square(relative_numerator) + np.square(relative_denominator)
    )
    return ratio, uncertainty


def compute_radiative_correction(
    born: str | Path,
    radiative: str | Path,
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
    max_files: int | None = None,
    min_counts: int = 5,
    normalization_ratio: float | None = None,
    born_integrated_cross_section: float | None = None,
    radiative_integrated_cross_section: float | None = None,
    progress_chunks: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
) -> RadiativeCorrectionResult:
    """Compute bin-by-bin radiative corrections from Born and radiative LUND samples.

    Histograms are accumulated in the native flat bin order and unflattened only at
    output time.  The trento phi convention is inherited from the analysis core:
    electron-proton planes through ``event_sample._trento_phi``.
    The stored correction is the radiative-to-Born cross-section ratio; downstream
    unfolding divides by it.
    """
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    born_files = _limited_lund_files(born, max_files)
    radiative_files = _limited_lund_files(radiative, max_files)
    if progress_chunks > 0:
        limit_text = f", max_files={max_files}" if max_files is not None else ""
        events_text = f", max_events={max_events}" if max_events is not None else ""
        print(
            f"[PROGRESS] Radiative correction preflight: "
            f"born_files={len(born_files)}, radiative_files={len(radiative_files)}"
            f"{limit_text}{events_text}"
        )
    born_result = histogram_lund(
        born_files,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
        progress_chunks=progress_chunks,
        progress_label="Born",
        phase_space=phase_space,
    )
    radiative_result = histogram_lund(
        radiative_files,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
        progress_chunks=progress_chunks,
        progress_label="Radiative",
        phase_space=phase_space,
    )
    if born_result.topology_events == 0:
        raise ValueError(f"Born sample has no valid generated e p pi0 events: {born}")
    if radiative_result.topology_events == 0:
        raise ValueError(f"Radiative sample has no valid generated e p pi0 events: {radiative}")

    if (born_integrated_cross_section is None) != (radiative_integrated_cross_section is None):
        raise ValueError(
            "born_integrated_cross_section and radiative_integrated_cross_section "
            "must be provided together"
        )
    if normalization_ratio is not None and born_integrated_cross_section is not None:
        raise ValueError(
            "normalization_ratio cannot be combined with integrated cross sections"
        )

    if born_integrated_cross_section is not None:
        born_integrated_cross_section = _positive_finite(
            born_integrated_cross_section, "born_integrated_cross_section"
        )
        radiative_integrated_cross_section = _positive_finite(
            radiative_integrated_cross_section, "radiative_integrated_cross_section"
        )
        normalization_ratio = (
            radiative_integrated_cross_section
            / born_integrated_cross_section
            * born_result.topology_events
            / radiative_result.topology_events
        )
    elif normalization_ratio is None:
        normalization_ratio = born_result.topology_events / radiative_result.topology_events
    normalization_ratio = float(normalization_ratio)
    if not np.isfinite(normalization_ratio) or normalization_ratio <= 0.0:
        raise ValueError("normalization_ratio must be positive and finite")

    lambda_born = born_result.counts + 0.5
    lambda_rad = radiative_result.counts + 0.5
    c_rad_flat = normalization_ratio * lambda_rad / lambda_born
    # AAO mode 3 can emit one accepted proposal m times.  Those exact LUND
    # copies have unit central weight but are one stochastic proposal, not m
    # independent observations.  For a compound-Poisson histogram the count
    # variance is sum(m**2), rather than sum(m).  Adding the same Jeffreys 0.5
    # regularizer used in the central ratio preserves the historical formula
    # exactly when every event has multiplicity one.
    born_relative_variance = (
        born_result.cluster_weight_square_sum + 0.5
    ) / np.square(lambda_born)
    radiative_relative_variance = (
        radiative_result.cluster_weight_square_sum + 0.5
    ) / np.square(lambda_rad)
    delta_flat = c_rad_flat * np.sqrt(
        born_relative_variance + radiative_relative_variance
    )
    reliable_flat = (
        (born_result.effective_counts >= min_counts)
        & (radiative_result.effective_counts >= min_counts)
    )

    c_rad_safe = np.where(reliable_flat, c_rad_flat, 1.0)
    delta_safe = np.where(reliable_flat, delta_flat, 1.0)
    support_overlap = (born_result.counts > 0.0) & (radiative_result.counts > 0.0)
    support_status = support_status_codes(
        born_result.counts,
        radiative_result.counts,
        min_counts,
        born_effective_counts=born_result.effective_counts,
        radiative_effective_counts=radiative_result.effective_counts,
    )

    return RadiativeCorrectionResult(
        c_rad=binning.unflatten(c_rad_safe),
        delta_c=binning.unflatten(delta_safe),
        c_rad_raw=binning.unflatten(c_rad_flat),
        delta_c_raw=binning.unflatten(delta_flat),
        reliable=binning.unflatten(reliable_flat),
        support_overlap=binning.unflatten(support_overlap),
        support_status=binning.unflatten(support_status),
        born=born_result,
        radiative=radiative_result,
        normalization_ratio=normalization_ratio,
        born_integrated_cross_section=born_integrated_cross_section,
        radiative_integrated_cross_section=radiative_integrated_cross_section,
    )


def _positive_finite(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def histogram_lund(
    pattern_or_dir: str | Path | Sequence[Path],
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
    max_files: int | None = None,
    progress_chunks: int = 0,
    progress_label: str = "LUND",
    phase_space: AnalysisPhaseSpace | None = None,
) -> LundHistogramResult:
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    files = _limited_lund_files(pattern_or_dir, max_files)
    counts = np.zeros(binning.size, dtype=float)
    cluster_weight_square_sum = np.zeros(binning.size, dtype=float)
    q2_min = np.full(binning.size, np.inf)
    q2_max = np.full(binning.size, -np.inf)
    eprime_min = np.full(binning.size, np.inf)
    eprime_max = np.full(binning.size, -np.inf)
    stats = _LundStats(files=len(files))

    if progress_chunks > 0:
        print(
            f"[PROGRESS] {progress_label} LUND files found: {len(files)}; "
            f"chunk_size={chunk_size}, progress_chunks={progress_chunks}"
        )

    for chunk_index, (electron, proton, multiplicity) in enumerate(
        _iter_lund_chunks(files, chunk_size, max_events, stats, progress_chunks, progress_label),
        start=1,
    ):
        q2, xb = _dis(electron, beam_energy)
        eprime = electron[:, 3]
        _update_global_ranges(stats, q2, eprime)
        minus_t = _minus_t(proton)
        phi = _trento_phi(electron, proton, beam_energy)
        flat = binning.coordinates_to_flat(q2, xb, minus_t, phi)
        inside = (flat >= 0) & (flat < binning.size)
        if phase_space is not None and phase_space.enabled:
            inside &= phase_space.mask(q2, xb, beam_energy)
        inside_flat = flat[inside]
        inside_multiplicity = multiplicity[inside]
        counts += np.bincount(
            inside_flat,
            weights=inside_multiplicity,
            minlength=binning.size,
        )
        cluster_weight_square_sum += np.bincount(
            inside_flat,
            weights=np.square(inside_multiplicity),
            minlength=binning.size,
        )
        if inside_flat.size:
            np.minimum.at(q2_min, inside_flat, q2[inside])
            np.maximum.at(q2_max, inside_flat, q2[inside])
            np.minimum.at(eprime_min, inside_flat, eprime[inside])
            np.maximum.at(eprime_max, inside_flat, eprime[inside])
        stats.in_range += int(np.sum(inside_multiplicity, dtype=np.int64))
        stats.in_range_clusters += int(np.count_nonzero(inside))
        if progress_chunks > 0 and chunk_index % progress_chunks == 0:
            print(
                f"[PROGRESS] {progress_label} LUND chunks {chunk_index}: "
                f"seen={stats.events_seen}, topology={stats.topology_events}, "
                f"in-range={stats.in_range}"
            )

    q2_min = np.where(np.isfinite(q2_min), q2_min, np.nan)
    q2_max = np.where(np.isfinite(q2_max), q2_max, np.nan)
    eprime_min = np.where(np.isfinite(eprime_min), eprime_min, np.nan)
    eprime_max = np.where(np.isfinite(eprime_max), eprime_max, np.nan)
    effective_counts = np.divide(
        np.square(counts),
        cluster_weight_square_sum,
        out=np.zeros_like(counts),
        where=cluster_weight_square_sum > 0.0,
    )

    return LundHistogramResult(
        counts=counts,
        cluster_weight_square_sum=cluster_weight_square_sum,
        effective_counts=effective_counts,
        q2_min=binning.unflatten(q2_min),
        q2_max=binning.unflatten(q2_max),
        eprime_min=binning.unflatten(eprime_min),
        eprime_max=binning.unflatten(eprime_max),
        files=stats.files,
        events_seen=stats.events_seen,
        topology_events=stats.topology_events,
        topology_clusters=stats.topology_clusters,
        in_range=stats.in_range,
        in_range_clusters=stats.in_range_clusters,
        maximum_cluster_multiplicity=stats.maximum_cluster_multiplicity,
        generated_q2_min=_finite_or_nan(stats.generated_q2_min),
        generated_q2_max=_finite_or_nan(stats.generated_q2_max),
        generated_eprime_min=_finite_or_nan(stats.generated_eprime_min),
        generated_eprime_max=_finite_or_nan(stats.generated_eprime_max),
    )


def support_status_codes(
    born_counts: Array,
    radiative_counts: Array,
    min_counts: int,
    *,
    born_effective_counts: Array | None = None,
    radiative_effective_counts: Array | None = None,
) -> Array:
    """Classify per-bin correction support.

    Codes:
      0 reliable;
      1 both samples are empty;
      2 born only;
      3 radiative only;
      4 both populated but born effective support is below min_counts;
      5 both populated but radiative effective support is below min_counts;
      6 both populated but both effective supports are below min_counts.

    Effective-count arrays default to the raw counts, preserving the public
    helper's historical behavior for independent unit-weight events.
    """
    born_counts = np.asarray(born_counts, dtype=float)
    radiative_counts = np.asarray(radiative_counts, dtype=float)
    if born_counts.shape != radiative_counts.shape:
        raise ValueError("born and radiative count arrays must have matching shapes")
    if born_effective_counts is None:
        born_effective_counts = born_counts
    if radiative_effective_counts is None:
        radiative_effective_counts = radiative_counts
    born_effective_counts = np.asarray(born_effective_counts, dtype=float)
    radiative_effective_counts = np.asarray(radiative_effective_counts, dtype=float)
    if born_effective_counts.shape != born_counts.shape:
        raise ValueError("born effective-count array must match born counts")
    if radiative_effective_counts.shape != radiative_counts.shape:
        raise ValueError("radiative effective-count array must match radiative counts")
    born_nonzero = born_counts > 0.0
    rad_nonzero = radiative_counts > 0.0
    born_low = born_effective_counts < min_counts
    rad_low = radiative_effective_counts < min_counts
    status = np.zeros(born_counts.shape, dtype=np.uint8)
    status[~born_nonzero & ~rad_nonzero] = 1
    status[born_nonzero & ~rad_nonzero] = 2
    status[~born_nonzero & rad_nonzero] = 3
    both = born_nonzero & rad_nonzero
    status[both & born_low & ~rad_low] = 4
    status[both & ~born_low & rad_low] = 5
    status[both & born_low & rad_low] = 6
    return status


@dataclass
class _LundStats:
    files: int
    events_seen: int = 0
    topology_events: int = 0
    topology_clusters: int = 0
    in_range: int = 0
    in_range_clusters: int = 0
    maximum_cluster_multiplicity: int = 0
    generated_q2_min: float = np.inf
    generated_q2_max: float = -np.inf
    generated_eprime_min: float = np.inf
    generated_eprime_max: float = -np.inf


def _iter_lund_chunks(
    files: list[Path],
    chunk_size: int,
    max_events: int | None,
    stats: _LundStats,
    progress_chunks: int = 0,
    progress_label: str = "LUND",
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    electron_rows: list[tuple[float, float, float, float]] = []
    proton_rows: list[tuple[float, float, float, float]] = []
    multiplicities: list[int] = []
    buffered_events = 0
    current_signature: tuple[str, ...] | None = None
    current_electron: tuple[float, float, float, float] | None = None
    current_proton: tuple[float, float, float, float] | None = None
    current_multiplicity = 0
    progress_events = chunk_size * progress_chunks if progress_chunks > 0 else 0
    next_event_progress = progress_events

    def append_current_cluster() -> None:
        nonlocal buffered_events
        if current_signature is None:
            return
        if current_electron is None or current_proton is None or current_multiplicity <= 0:
            raise RuntimeError("invalid internal LUND cluster state")
        electron_rows.append(current_electron)
        proton_rows.append(current_proton)
        multiplicities.append(current_multiplicity)
        buffered_events += current_multiplicity
        stats.topology_clusters += 1
        stats.maximum_cluster_multiplicity = max(
            stats.maximum_cluster_multiplicity,
            current_multiplicity,
        )

    def chunk_arrays() -> tuple[Array, Array, Array]:
        nonlocal buffered_events
        result = (
            np.asarray(electron_rows, dtype=float),
            np.asarray(proton_rows, dtype=float),
            np.asarray(multiplicities, dtype=float),
        )
        electron_rows.clear()
        proton_rows.clear()
        multiplicities.clear()
        buffered_events = 0
        return result

    for file_index, filename in enumerate(files, start=1):
        if progress_chunks > 0 and (file_index == 1 or file_index % progress_chunks == 0):
            print(
                f"[PROGRESS] {progress_label} LUND file {file_index}/{len(files)}: {filename}"
            )
        with filename.open("r", errors="replace") as source:
            while True:
                header = source.readline()
                if not header:
                    break
                fields = header.split()
                if not fields:
                    continue
                try:
                    n_particles = int(fields[0])
                except ValueError:
                    continue

                stats.events_seen += 1
                if progress_events > 0 and stats.events_seen >= next_event_progress:
                    print(
                        f"[PROGRESS] {progress_label} LUND events: "
                        f"seen={stats.events_seen}, topology={stats.topology_events}, "
                        f"pending-topology={buffered_events + current_multiplicity}"
                    )
                    next_event_progress += progress_events
                electron = None
                proton = None
                photon_count = 0
                has_pi0 = False
                valid = True
                event_lines = [header]
                for _ in range(n_particles):
                    row = source.readline()
                    if not row:
                        valid = False
                        break
                    event_lines.append(row)
                    parts = row.split()
                    if len(parts) < 10:
                        valid = False
                        continue
                    try:
                        pid = int(parts[3])
                        px = float(parts[6])
                        py = float(parts[7])
                        pz = float(parts[8])
                        energy = float(parts[9])
                    except ValueError:
                        valid = False
                        continue
                    if pid == 11 and electron is None:
                        electron = (px, py, pz, energy)
                    elif pid == 2212 and proton is None:
                        proton = (px, py, pz, energy)
                    elif pid == 111:
                        has_pi0 = True
                    elif pid == 22:
                        photon_count += 1

                topology_valid = (
                    valid
                    and electron is not None
                    and proton is not None
                    and (has_pi0 or photon_count >= 2)
                )
                if not topology_valid:
                    # An invalid/non-target event is a hard boundary: two equal
                    # valid records on either side did not come from one emitted
                    # multiplicity block.
                    if current_signature is not None:
                        append_current_cluster()
                        current_signature = None
                        current_electron = None
                        current_proton = None
                        current_multiplicity = 0
                    if buffered_events >= chunk_size:
                        yield chunk_arrays()
                    continue
                stats.topology_events += 1
                signature = tuple(event_lines)
                if signature == current_signature:
                    current_multiplicity += 1
                else:
                    if current_signature is not None:
                        append_current_cluster()
                    current_signature = signature
                    current_electron = electron
                    current_proton = proton
                    current_multiplicity = 1

                if buffered_events >= chunk_size:
                    yield chunk_arrays()

                if max_events is not None and stats.topology_events >= max_events:
                    append_current_cluster()
                    current_signature = None
                    current_electron = None
                    current_proton = None
                    current_multiplicity = 0
                    if electron_rows:
                        yield chunk_arrays()
                    return

    append_current_cluster()
    if electron_rows:
        yield chunk_arrays()


def _limited_lund_files(pattern_or_dir: str | Path | Sequence[Path], max_files: int | None) -> list[Path]:
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when provided")
    files = _lund_files(pattern_or_dir, max_files=max_files)
    if not files:
        raise FileNotFoundError(
            f"No LUND-like text files matched: {pattern_or_dir}. "
            "Directories are scanned recursively for text files with LUND event headers."
        )
    return files


def _lund_files(pattern_or_dir: str | Path | Sequence[Path], max_files: int | None = None) -> list[Path]:
    # Use a type tuple rather than ``str | Path`` here: the latter is valid as
    # a postponed annotation on Python 3.9, but not as the runtime second
    # argument to isinstance on the farm's Python 3.9 interpreter.
    if not isinstance(pattern_or_dir, (str, Path)):
        return _filter_lund_files(pattern_or_dir, max_files=max_files)
    path = Path(pattern_or_dir)
    if path.is_dir():
        if max_files is not None:
            return _filter_lund_files(path.rglob("*"), max_files=max_files)
        candidates = sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in LUND_TEXT_SUFFIXES
        )
    else:
        candidates = sorted(Path(item) for item in glob.glob(str(pattern_or_dir)))
    return _filter_lund_files(candidates, max_files=max_files)


def _filter_lund_files(candidates: Iterable[Path], max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for item in candidates:
        if not item.is_file() or item.suffix.lower() not in LUND_TEXT_SUFFIXES:
            continue
        if item.stat().st_size <= 0:
            continue
        if item.suffix.lower() == ".lund":
            files.append(item)
            if max_files is not None and len(files) >= max_files:
                break
            continue
        if (
            _looks_text(item)
            and _looks_lund_header(item)
        ):
            files.append(item)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def _looks_text(path: Path, nbytes: int = 4096) -> bool:
    try:
        with path.open("rb") as source:
            chunk = source.read(nbytes)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _looks_lund_header(path: Path, max_lines: int = 50) -> bool:
    try:
        with path.open("r", errors="replace") as source:
            for _ in range(max_lines):
                line = source.readline()
                if not line:
                    return False
                fields = line.split()
                if not fields:
                    continue
                try:
                    return int(fields[0]) > 0
                except ValueError:
                    continue
    except OSError:
        return False
    return False


def _update_global_ranges(stats: _LundStats, q2: Array, eprime: Array) -> None:
    valid_q2 = np.asarray(q2)[np.isfinite(q2)]
    if valid_q2.size:
        stats.generated_q2_min = min(stats.generated_q2_min, float(np.min(valid_q2)))
        stats.generated_q2_max = max(stats.generated_q2_max, float(np.max(valid_q2)))
    valid_eprime = np.asarray(eprime)[np.isfinite(eprime)]
    if valid_eprime.size:
        stats.generated_eprime_min = min(stats.generated_eprime_min, float(np.min(valid_eprime)))
        stats.generated_eprime_max = max(stats.generated_eprime_max, float(np.max(valid_eprime)))


def _finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan
