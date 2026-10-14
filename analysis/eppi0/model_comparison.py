from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

from .binning import AnalysisBinning
from .cross_section import virtual_photon_flux
from .exclusive_kinematics import physical_mask
from .phase_space import AnalysisPhaseSpace
from .structure_functions import epsilon_from_xb_q2


Array = np.ndarray
TheoryEvaluator = Callable[[Array], Array]


@dataclass(frozen=True)
class ModelGridResult:
    reduced_cross_section: Array
    extracted_bin_average: Array
    simple_reduced_average: Array
    center_reduced_cross_section: Array
    reliable: Array
    computed: Array
    q2_reference: Array
    xb_reference: Array
    minus_t_reference: Array
    phi_reference: Array
    n_physical: Array
    n_valid: Array
    n_failed: Array
    physical_fraction: Array
    failure_fraction: Array


class _NumericInterpolator:
    def __init__(self, points: Array, values: Array, method: str) -> None:
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        if points.ndim != 2 or values.shape != (points.shape[0],):
            raise ValueError("interpolation points and values have incompatible shapes")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
            raise ValueError("model table interpolation data must be finite")
        if method not in ("linear", "nearest"):
            raise ValueError("interpolation method must be linear or nearest")
        points, values = _deduplicate_points(points, values)
        self._fixed = np.ptp(points, axis=0) == 0.0
        self._fixed_values = points[0].copy()
        self._minimum = np.min(points, axis=0)
        self._maximum = np.max(points, axis=0)
        active_points = points[:, ~self._fixed]
        self._constant = float(values[0]) if active_points.shape[1] == 0 else None
        self._interpolator = None
        if self._constant is not None:
            if not np.allclose(values, self._constant, rtol=0.0, atol=1.0e-12):
                raise ValueError("model table repeats one coordinate with different values")
            return

        axes = [np.unique(active_points[:, axis]) for axis in range(active_points.shape[1])]
        is_grid = int(np.prod([axis.size for axis in axes])) == active_points.shape[0]
        if is_grid:
            shape = tuple(axis.size for axis in axes)
            grid_values = np.full(shape, np.nan, dtype=float)
            indices = tuple(
                np.searchsorted(axis, active_points[:, index])
                for index, axis in enumerate(axes)
            )
            grid_values[indices] = values
            is_grid = bool(np.all(np.isfinite(grid_values)))
        if is_grid:
            self._interpolator = RegularGridInterpolator(
                tuple(axes),
                grid_values,
                method=method,
                bounds_error=False,
                fill_value=np.nan,
            )
        elif method == "nearest":
            self._interpolator = NearestNDInterpolator(active_points, values)
        else:
            if active_points.shape[0] <= active_points.shape[1]:
                raise ValueError(
                    "linear scattered interpolation needs more rows than dimensions; "
                    "use --interpolation nearest or provide a denser table"
                )
            try:
                self._interpolator = LinearNDInterpolator(
                    active_points, values, fill_value=np.nan
                )
            except Exception as exc:
                raise ValueError(
                    "could not construct linear interpolation from the model table; "
                    "use a Cartesian grid or --interpolation nearest"
                ) from exc

    def __call__(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self._fixed.size:
            raise ValueError("model interpolation query has incompatible shape")
        in_bounds = np.all(
            np.isfinite(points)
            & (points >= self._minimum[None, :] - 1.0e-12)
            & (points <= self._maximum[None, :] + 1.0e-12),
            axis=1,
        )
        if np.any(self._fixed):
            in_bounds &= np.all(
                np.isclose(
                    points[:, self._fixed],
                    self._fixed_values[self._fixed][None, :],
                    rtol=0.0,
                    atol=1.0e-10,
                ),
                axis=1,
            )
        output = np.full(points.shape[0], np.nan, dtype=float)
        if not np.any(in_bounds):
            return output
        if self._constant is not None:
            output[in_bounds] = self._constant
        else:
            output[in_bounds] = np.asarray(
                self._interpolator(points[in_bounds][:, ~self._fixed]), dtype=float
            ).reshape(-1)
        return output


def _deduplicate_points(points: Array, values: Array) -> tuple[Array, Array]:
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    if unique.shape[0] == points.shape[0]:
        return points, values
    sums = np.bincount(inverse, weights=values, minlength=unique.shape[0])
    counts = np.bincount(inverse, minlength=unique.shape[0])
    means = sums / counts
    spread = np.zeros(unique.shape[0], dtype=float)
    np.maximum.at(spread, inverse, np.abs(values - means[inverse]))
    scale = np.maximum(1.0, np.abs(means))
    if np.any(spread > 1.0e-10 * scale):
        raise ValueError("model table contains duplicate coordinates with different values")
    return unique, means


class TabulatedModelEvaluator:
    """Interpolate reduced cross sections or structure functions from CSV."""

    def __init__(
        self,
        path: str | Path,
        beam_energy: float,
        interpolation: str = "linear",
    ) -> None:
        self.path = Path(path)
        self.beam_energy = float(beam_energy)
        columns = _read_numeric_csv(self.path)
        xb = _column(columns, "xb", "x_b")
        q2 = _column(columns, "q2", "q^2")
        if _has_column(columns, "minus_t", "-t"):
            signed_t = -_column(columns, "minus_t", "-t")
            self.t_input_convention = "positive minus_t"
        else:
            signed_t = _column(columns, "t")
            self.t_input_convention = "signed t"

        reduced_name = _find_column(
            columns,
            "reduced_cross_section",
            "sigma_reduced",
            "dsigma_dt_dphi",
        )
        self.mode: str
        self._reduced = None
        self._structure: dict[str, _NumericInterpolator] = {}
        if reduced_name is not None:
            phi = _column(columns, "phi_deg", "phi") % 360.0
            points = np.column_stack((xb, q2, signed_t, phi))
            periodic_points = np.concatenate(
                (
                    points,
                    points + np.asarray([0.0, 0.0, 0.0, -360.0]),
                    points + np.asarray([0.0, 0.0, 0.0, 360.0]),
                ),
                axis=0,
            )
            periodic_values = np.tile(columns[reduced_name], 3)
            self._reduced = _NumericInterpolator(
                periodic_points, periodic_values, interpolation
            )
            self.mode = "reduced_cross_section"
        else:
            points = np.column_stack((xb, q2, signed_t))
            has_separated = _has_column(columns, "sigma_t") and _has_column(
                columns, "sigma_l"
            )
            has_unseparated = _has_column(columns, "sigma_u")
            if not has_separated and not has_unseparated:
                raise ValueError(
                    "model CSV must contain reduced_cross_section with phi_deg, "
                    "sigma_T and sigma_L, or sigma_U"
                )
            names = ("sigma_t", "sigma_l") if has_separated else ("sigma_u",)
            names += ("sigma_lt", "sigma_tt")
            for name in names:
                self._structure[name] = _NumericInterpolator(
                    points, _column(columns, name), interpolation
                )
            self.mode = (
                "separated_structure_functions"
                if has_separated
                else "unseparated_structure_functions"
            )

    def __call__(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("model points must have columns (xB, Q2, t, phi_deg)")
        if self._reduced is not None:
            query = points.copy()
            query[:, 3] %= 360.0
            return self._reduced(query)

        query = points[:, :3]
        epsilon = epsilon_from_xb_q2(points[:, 1], points[:, 0], self.beam_energy)
        if "sigma_u" in self._structure:
            sigma_u = self._structure["sigma_u"](query)
        else:
            sigma_u = self._structure["sigma_t"](query) + epsilon * self._structure[
                "sigma_l"
            ](query)
        sigma_lt = self._structure["sigma_lt"](query)
        sigma_tt = self._structure["sigma_tt"](query)
        phi = np.deg2rad(points[:, 3])
        return (
            sigma_u
            + np.sqrt(2.0 * epsilon * (1.0 + epsilon)) * sigma_lt * np.cos(phi)
            + epsilon * sigma_tt * np.cos(2.0 * phi)
        ) / (2.0 * np.pi)


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _read_numeric_csv(path: Path) -> dict[str, Array]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"model CSV has no header: {path}")
        normalized = {_normalize_header(name): name for name in reader.fieldnames}
        rows = list(reader)
    if not rows:
        raise ValueError(f"model CSV has no data rows: {path}")
    columns: dict[str, Array] = {}
    for normalized_name, original_name in normalized.items():
        try:
            columns[normalized_name] = np.asarray(
                [float(row[original_name]) for row in rows], dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model CSV column {original_name!r} must be numeric"
            ) from exc
    return columns


def _find_column(columns: Mapping[str, Array], *names: str) -> str | None:
    for name in names:
        normalized = _normalize_header(name)
        if normalized in columns:
            return normalized
    return None


def _has_column(columns: Mapping[str, Array], *names: str) -> bool:
    return _find_column(columns, *names) is not None


def _column(columns: Mapping[str, Array], *names: str) -> Array:
    name = _find_column(columns, *names)
    if name is None:
        raise ValueError(f"model CSV is missing required column: {' or '.join(names)}")
    return np.asarray(columns[name], dtype=float)


def midpoint_grid(low: float, high: float, points: int) -> Array:
    edges = np.linspace(float(low), float(high), int(points) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def circular_mean_deg(values: Array) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return float(
        np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
        % 360.0
    )


def average_model_over_bins(
    binning: AnalysisBinning,
    beam_energy: float,
    evaluator: TheoryEvaluator,
    *,
    samples_per_dimension: int = 4,
    max_failure_fraction: float = 0.0,
    bin_start: int = 0,
    bin_stop: int | None = None,
    progress_chunks: int = 0,
    phase_space: AnalysisPhaseSpace | None = None,
    q2_reference: Array | None = None,
    xb_reference: Array | None = None,
    transform: Array | None = None,
    requested_mask: Array | None = None,
) -> ModelGridResult:
    """Forward-average a reduced virtual-photon model over analysis bins.

    The extraction-equivalent prediction is
    ``<Gamma * sigma_reduced> / Gamma(reference)``. ``transform`` applies the
    same final multiplicative divisor used for the data, normally its stored
    AAO ``C_BC`` artifact.
    """
    if samples_per_dimension <= 0:
        raise ValueError("samples_per_dimension must be positive")
    if not 0.0 <= max_failure_fraction <= 1.0:
        raise ValueError("max_failure_fraction must be between zero and one")
    shape = binning.shape
    total_3d = int(np.prod(shape[:3]))
    if bin_stop is None:
        bin_stop = total_3d
    if bin_start < 0 or bin_stop < bin_start or bin_stop > total_3d:
        raise ValueError(f"invalid 3D bin range [{bin_start}, {bin_stop})")
    phase_space = AnalysisPhaseSpace() if phase_space is None else phase_space

    def optional_grid(values: Array | None, default: float) -> Array:
        if values is None:
            return np.full(shape, default, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != shape:
            raise ValueError(f"comparison grid has shape {values.shape}; expected {shape}")
        return values

    q2_external = None if q2_reference is None else optional_grid(q2_reference, np.nan)
    xb_external = None if xb_reference is None else optional_grid(xb_reference, np.nan)
    transform_grid = optional_grid(transform, 1.0)
    if requested_mask is None:
        requested = np.ones(shape, dtype=bool)
    else:
        requested = np.asarray(requested_mask, dtype=bool)
        if requested.shape != shape:
            raise ValueError("requested model mask does not match analysis binning")

    reduced = np.full(shape, np.nan)
    extracted = np.full(shape, np.nan)
    simple = np.full(shape, np.nan)
    center = np.full(shape, np.nan)
    reliable = np.zeros(shape, dtype=bool)
    computed = np.zeros(shape, dtype=bool)
    q2_ref_out = np.full(shape, np.nan)
    xb_ref_out = np.full(shape, np.nan)
    mt_ref_out = np.full(shape, np.nan)
    phi_ref_out = np.full(shape, np.nan)
    n_physical = np.zeros(shape, dtype=np.int64)
    n_valid = np.zeros(shape, dtype=np.int64)
    n_failed = np.zeros(shape, dtype=np.int64)
    physical_fraction = np.zeros(shape)
    failure_fraction = np.ones(shape)
    total_grid_points = samples_per_dimension**4
    processed = 0
    assigned = bin_stop - bin_start

    for iq2, (q2_low, q2_high) in enumerate(
        zip(binning.q2_edges[:-1], binning.q2_edges[1:])
    ):
        q2_points = midpoint_grid(q2_low, q2_high, samples_per_dimension)
        for ixb, (xb_low, xb_high) in enumerate(
            zip(binning.xb_edges[:-1], binning.xb_edges[1:])
        ):
            xb_points = midpoint_grid(xb_low, xb_high, samples_per_dimension)
            for it, (mt_low, mt_high) in enumerate(
                zip(binning.t_edges[:-1], binning.t_edges[1:])
            ):
                flat3 = np.ravel_multi_index((iq2, ixb, it), shape[:3])
                if flat3 < bin_start or flat3 >= bin_stop:
                    continue
                processed += 1
                # Mark the complete assigned 3D slice, including phi bins that
                # were intentionally excluded by the data-quality mask. This
                # makes partial-artifact coverage independent of that mask.
                computed[iq2, ixb, it, :] = True
                mt_points = midpoint_grid(mt_low, mt_high, samples_per_dimension)
                q2_mesh, xb_mesh, mt_mesh = np.meshgrid(
                    q2_points, xb_points, mt_points, indexing="ij"
                )
                signed_t_mesh = -mt_mesh
                physical = physical_mask(
                    xb_mesh, q2_mesh, signed_t_mesh, beam_energy
                )
                if phase_space.enabled:
                    physical &= phase_space.mask(q2_mesh, xb_mesh, beam_energy)
                if not np.any(physical):
                    continue
                q2_phys = q2_mesh[physical]
                xb_phys = xb_mesh[physical]
                signed_t_phys = signed_t_mesh[physical]
                mt_phys = mt_mesh[physical]
                default_q2 = float(np.mean(q2_phys))
                default_xb = float(np.mean(xb_phys))
                default_mt = float(np.mean(mt_phys))

                for iphi, (phi_low, phi_high) in enumerate(
                    zip(binning.phi_edges[:-1], binning.phi_edges[1:])
                ):
                    index = (iq2, ixb, it, iphi)
                    if not requested[index]:
                        continue
                    phi_points = midpoint_grid(
                        phi_low, phi_high, samples_per_dimension
                    )
                    phi_mesh = np.broadcast_to(
                        phi_points, (q2_phys.size, phi_points.size)
                    )
                    q2_eval = np.broadcast_to(
                        q2_phys[:, None], phi_mesh.shape
                    ).ravel()
                    xb_eval = np.broadcast_to(
                        xb_phys[:, None], phi_mesh.shape
                    ).ravel()
                    t_eval = np.broadcast_to(
                        signed_t_phys[:, None], phi_mesh.shape
                    ).ravel()
                    phi_eval = phi_mesh.ravel()
                    points = np.column_stack((xb_eval, q2_eval, t_eval, phi_eval))
                    values = np.asarray(evaluator(points), dtype=float)
                    if values.shape != (points.shape[0],):
                        raise ValueError(
                            f"model evaluator returned {values.shape}; expected "
                            f"{(points.shape[0],)}"
                        )
                    flux = virtual_photon_flux(q2_eval, xb_eval, beam_energy)
                    valid = (
                        np.isfinite(values)
                        & (values >= 0.0)
                        & np.isfinite(flux)
                        & (flux > 0.0)
                    )
                    n_physical[index] = points.shape[0]
                    n_valid[index] = int(np.count_nonzero(valid))
                    n_failed[index] = points.shape[0] - n_valid[index]
                    physical_fraction[index] = points.shape[0] / total_grid_points
                    failure_fraction[index] = n_failed[index] / points.shape[0]
                    q2_ref = (
                        float(q2_external[index])
                        if q2_external is not None
                        else default_q2
                    )
                    xb_ref = (
                        float(xb_external[index])
                        if xb_external is not None
                        else default_xb
                    )
                    phi_ref = circular_mean_deg(phi_points)
                    q2_ref_out[index] = q2_ref
                    xb_ref_out[index] = xb_ref
                    mt_ref_out[index] = default_mt
                    phi_ref_out[index] = phi_ref
                    if not np.any(valid):
                        continue
                    simple[index] = float(np.mean(values[valid]))
                    reference_flux = float(
                        virtual_photon_flux(q2_ref, xb_ref, beam_energy)
                    )
                    if not np.isfinite(reference_flux) or reference_flux <= 0.0:
                        continue
                    extracted[index] = float(
                        np.mean(flux[valid] * values[valid]) / reference_flux
                    )
                    divisor = transform_grid[index]
                    if not np.isfinite(divisor) or divisor <= 0.0:
                        continue
                    reduced[index] = extracted[index] / divisor
                    center_value = np.asarray(
                        evaluator(
                            np.asarray(
                                [[xb_ref, q2_ref, -default_mt, phi_ref]], dtype=float
                            )
                        ),
                        dtype=float,
                    )
                    if center_value.shape == (1,) and np.isfinite(center_value[0]):
                        center[index] = center_value[0]
                    reliable[index] = (
                        np.isfinite(reduced[index])
                        and reduced[index] >= 0.0
                        and failure_fraction[index] <= max_failure_fraction
                    )
                if progress_chunks > 0 and processed % progress_chunks == 0:
                    print(
                        f"[PROGRESS] model 3D bins {processed}/{assigned} "
                        f"({100.0 * processed / max(assigned, 1):.1f}%), "
                        f"reliable phi bins={int(np.count_nonzero(reliable))}",
                        flush=True,
                    )

    return ModelGridResult(
        reduced_cross_section=reduced,
        extracted_bin_average=extracted,
        simple_reduced_average=simple,
        center_reduced_cross_section=center,
        reliable=reliable,
        computed=computed,
        q2_reference=q2_ref_out,
        xb_reference=xb_ref_out,
        minus_t_reference=mt_ref_out,
        phi_reference=phi_ref_out,
        n_physical=n_physical,
        n_valid=n_valid,
        n_failed=n_failed,
        physical_fraction=physical_fraction,
        failure_fraction=failure_fraction,
    )
