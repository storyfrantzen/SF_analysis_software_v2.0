from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np

from .binning import AnalysisBinning
from .cross_section import virtual_photon_flux


Array = np.ndarray

PROTON_MASS_GEV = 0.9382720813
PI0_MASS_GEV = 0.1349768


@dataclass(frozen=True)
class BinCenteringResult:
    c_bc: Array
    reliable: Array
    average_d4sigma: Array
    center_d4sigma: Array
    xB_center: Array
    q2_center: Array
    minus_t_center: Array
    phi_center: Array
    n_physical: Array
    n_valid: Array
    n_failed: Array
    physical_fraction: Array
    failure_fraction: Array


TheoryEvaluator = Callable[[Array], Array]


def midpoint_grid(lo: float, hi: float, points: int) -> Array:
    edges = np.linspace(float(lo), float(hi), int(points) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def kallen(a: Array, b: float, c: float) -> Array:
    return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c


def t_limits_pi0(xb: Array, q2: Array) -> tuple[Array, Array]:
    """Return signed physical t limits for gamma* p -> p pi0."""
    xb, q2 = np.broadcast_arrays(np.asarray(xb, float), np.asarray(q2, float))
    t_low = np.full(xb.shape, np.nan, dtype=float)
    t_high = np.full(xb.shape, np.nan, dtype=float)

    valid = (xb > 0.0) & (xb < 1.0) & (q2 > 0.0)
    w2 = np.full(xb.shape, np.nan, dtype=float)
    w2[valid] = PROTON_MASS_GEV**2 + q2[valid] * (1.0 / xb[valid] - 1.0)
    valid &= w2 > (PROTON_MASS_GEV + PI0_MASS_GEV) ** 2
    if not np.any(valid):
        return t_low, t_high

    w = np.sqrt(w2[valid])
    q0_cm = (w2[valid] - PROTON_MASS_GEV**2 - q2[valid]) / (2.0 * w)
    q_cm2 = q0_cm * q0_cm + q2[valid]
    epi_cm = (w2[valid] + PI0_MASS_GEV**2 - PROTON_MASS_GEV**2) / (2.0 * w)
    ppi_cm2 = kallen(w2[valid], PI0_MASS_GEV**2, PROTON_MASS_GEV**2) / (4.0 * w2[valid])
    subvalid = (q_cm2 > 0.0) & (ppi_cm2 > 0.0)
    if not np.any(subvalid):
        return t_low, t_high

    valid_indices = np.flatnonzero(valid)
    out_indices = valid_indices[subvalid]
    q_cm = np.sqrt(q_cm2[subvalid])
    ppi_cm = np.sqrt(ppi_cm2[subvalid])
    t_forward = (
        PI0_MASS_GEV**2
        - q2.ravel()[out_indices]
        - 2.0 * q0_cm[subvalid] * epi_cm[subvalid]
        + 2.0 * q_cm * ppi_cm
    )
    t_backward = (
        PI0_MASS_GEV**2
        - q2.ravel()[out_indices]
        - 2.0 * q0_cm[subvalid] * epi_cm[subvalid]
        - 2.0 * q_cm * ppi_cm
    )
    flat_low = t_low.ravel()
    flat_high = t_high.ravel()
    flat_low[out_indices] = np.minimum(t_backward, t_forward)
    flat_high[out_indices] = np.maximum(t_backward, t_forward)
    return t_low, t_high


def physical_mask(xb: Array, q2: Array, signed_t: Array, beam_energy: float) -> Array:
    """Return the physical ep -> e p pi0 mask for signed t values."""
    xb, q2, signed_t = np.broadcast_arrays(
        np.asarray(xb, float),
        np.asarray(q2, float),
        np.asarray(signed_t, float),
    )
    valid = (xb > 0.0) & (xb < 1.0) & (q2 > 0.0)
    w2 = PROTON_MASS_GEV**2 + q2 * (1.0 / xb - 1.0)
    valid &= w2 > (PROTON_MASS_GEV + PI0_MASS_GEV) ** 2
    nu = np.divide(q2, 2.0 * PROTON_MASS_GEV * xb, out=np.full_like(q2, np.nan), where=xb > 0.0)
    eprime = float(beam_energy) - nu
    valid &= eprime > 0.0
    sin2_half = np.divide(q2, 4.0 * float(beam_energy) * eprime, out=np.full_like(q2, np.nan), where=eprime > 0.0)
    valid &= (sin2_half > 0.0) & (sin2_half < 1.0)
    t_low, t_high = t_limits_pi0(xb, q2)
    return valid & np.isfinite(t_low) & np.isfinite(t_high) & (signed_t >= t_low) & (signed_t <= t_high)


def circular_mean_deg(values: Array) -> float:
    radians = np.deg2rad(np.asarray(values, float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360.0)


def compute_bin_centering(
    binning: AnalysisBinning,
    beam_energy: float,
    evaluator: TheoryEvaluator,
    samples_per_dimension: int = 4,
    max_failure_fraction: float = 0.0,
) -> BinCenteringResult:
    """Compute C_BC = <d4sigma>_physical_bin / d4sigma(reference point)."""
    if samples_per_dimension <= 0:
        raise ValueError("samples_per_dimension must be positive")
    if max_failure_fraction < 0.0 or max_failure_fraction > 1.0:
        raise ValueError("max_failure_fraction must be between 0 and 1")

    shape = binning.shape
    c_bc = np.ones(shape, dtype=float)
    reliable = np.zeros(shape, dtype=bool)
    average = np.full(shape, np.nan, dtype=float)
    center = np.full(shape, np.nan, dtype=float)
    xb_center = np.full(shape, np.nan, dtype=float)
    q2_center = np.full(shape, np.nan, dtype=float)
    minus_t_center = np.full(shape, np.nan, dtype=float)
    phi_center = np.full(shape, np.nan, dtype=float)
    n_physical = np.zeros(shape, dtype=np.int64)
    n_valid = np.zeros(shape, dtype=np.int64)
    n_failed = np.zeros(shape, dtype=np.int64)
    physical_fraction = np.zeros(shape, dtype=float)
    failure_fraction = np.ones(shape, dtype=float)

    total_grid_points = samples_per_dimension**4
    for iq2, (q2_lo, q2_hi) in enumerate(zip(binning.q2_edges[:-1], binning.q2_edges[1:])):
        q2_points = midpoint_grid(q2_lo, q2_hi, samples_per_dimension)
        for ixb, (xb_lo, xb_hi) in enumerate(zip(binning.xb_edges[:-1], binning.xb_edges[1:])):
            xb_points = midpoint_grid(xb_lo, xb_hi, samples_per_dimension)
            for it, (mt_lo, mt_hi) in enumerate(zip(binning.t_edges[:-1], binning.t_edges[1:])):
                minus_t_points = midpoint_grid(mt_lo, mt_hi, samples_per_dimension)
                q2_mesh, xb_mesh, minus_t_mesh = np.meshgrid(
                    q2_points,
                    xb_points,
                    minus_t_points,
                    indexing="ij",
                )
                signed_t_mesh = -minus_t_mesh
                physical = physical_mask(xb_mesh, q2_mesh, signed_t_mesh, beam_energy)
                if not np.any(physical):
                    continue

                q2_phys = q2_mesh[physical]
                xb_phys = xb_mesh[physical]
                signed_t_phys = signed_t_mesh[physical]
                minus_t_phys = minus_t_mesh[physical]
                q2_ref = float(np.mean(q2_phys))
                xb_ref = float(np.mean(xb_phys))
                signed_t_ref = float(np.mean(signed_t_phys))
                minus_t_ref = float(np.mean(minus_t_phys))

                for iphi, (phi_lo, phi_hi) in enumerate(zip(binning.phi_edges[:-1], binning.phi_edges[1:])):
                    phi_points = midpoint_grid(phi_lo, phi_hi, samples_per_dimension)
                    phi_mesh = np.broadcast_to(phi_points, (q2_phys.size, phi_points.size))
                    q2_eval = np.broadcast_to(q2_phys[:, None], phi_mesh.shape).ravel()
                    xb_eval = np.broadcast_to(xb_phys[:, None], phi_mesh.shape).ravel()
                    signed_t_eval = np.broadcast_to(signed_t_phys[:, None], phi_mesh.shape).ravel()
                    phi_eval = phi_mesh.ravel()
                    points = np.column_stack((xb_eval, q2_eval, signed_t_eval, phi_eval))

                    sigu = np.asarray(evaluator(points), dtype=float)
                    if sigu.shape != (points.shape[0],):
                        raise ValueError(f"evaluator returned shape {sigu.shape}; expected {(points.shape[0],)}")
                    flux = virtual_photon_flux(q2_eval, xb_eval, beam_energy)
                    d4sigma = flux * sigu
                    valid = np.isfinite(d4sigma) & (d4sigma > 0.0)

                    index = (iq2, ixb, it, iphi)
                    n_physical[index] = points.shape[0]
                    n_valid[index] = int(np.count_nonzero(valid))
                    n_failed[index] = points.shape[0] - n_valid[index]
                    physical_fraction[index] = points.shape[0] / total_grid_points
                    failure_fraction[index] = n_failed[index] / points.shape[0]
                    q2_center[index] = q2_ref
                    xb_center[index] = xb_ref
                    minus_t_center[index] = minus_t_ref
                    phi_ref = circular_mean_deg(phi_points)
                    phi_center[index] = phi_ref

                    if not np.any(valid):
                        continue
                    average[index] = float(np.mean(d4sigma[valid]))

                    center_point = np.array([[xb_ref, q2_ref, signed_t_ref, phi_ref]], dtype=float)
                    center_sigu = np.asarray(evaluator(center_point), dtype=float)
                    if center_sigu.shape != (1,):
                        raise ValueError(f"evaluator returned shape {center_sigu.shape}; expected (1,)")
                    center[index] = float(virtual_photon_flux(q2_ref, xb_ref, beam_energy) * center_sigu[0])
                    if np.isfinite(center[index]) and center[index] > 0.0:
                        c_bc[index] = average[index] / center[index]
                        reliable[index] = np.isfinite(c_bc[index]) and c_bc[index] > 0.0 and failure_fraction[index] <= max_failure_fraction

    return BinCenteringResult(
        c_bc=c_bc,
        reliable=reliable,
        average_d4sigma=average,
        center_d4sigma=center,
        xB_center=xb_center,
        q2_center=q2_center,
        minus_t_center=minus_t_center,
        phi_center=phi_center,
        n_physical=n_physical,
        n_valid=n_valid,
        n_failed=n_failed,
        physical_fraction=physical_fraction,
        failure_fraction=failure_fraction,
    )


@dataclass
class AaoExecutableEvaluator:
    exe: Path
    beam_energy: float
    theory: int = 5
    channel: int = 1
    resonance: int = 0
    workers: int | None = None
    chunk_size: int = 64
    verbose_failures: bool = False

    _pool: multiprocessing.pool.Pool | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "AaoExecutableEvaluator":
        worker_count = self.workers or multiprocessing.cpu_count()
        if worker_count > 1:
            self._pool = multiprocessing.Pool(processes=worker_count)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __call__(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        jobs = [
            (
                str(self.exe),
                float(self.beam_energy),
                int(self.theory),
                int(self.channel),
                int(self.resonance),
                bool(self.verbose_failures),
                tuple(map(float, point)),
            )
            for point in points
        ]
        if not jobs:
            return np.empty(0, dtype=float)
        if self._pool is not None:
            values = self._pool.map(_call_aao_xsec_job, jobs, chunksize=max(1, int(self.chunk_size)))
        else:
            values = [_call_aao_xsec_job(job) for job in jobs]
        return np.asarray(values, dtype=float)


def _call_aao_xsec_job(job: tuple[str, float, int, int, int, bool, tuple[float, float, float, float]]) -> float:
    exe, beam_energy, theory, channel, resonance, verbose_failures, point = job
    xb, q2, signed_t, phi = point
    cmd = [
        exe,
        "-xB",
        str(xb),
        "-Q2",
        str(q2),
        "-t",
        str(signed_t),
        "-phi",
        str(phi),
        "-BeamEnergy",
        str(beam_energy),
        "-theory",
        str(theory),
        "-channel",
        str(channel),
        "-resonance",
        str(resonance),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        if verbose_failures:
            print(
                "WARNING: aao_xsec failed for "
                f"xB={xb:.6g} Q2={q2:.6g} t={signed_t:.6g} phi={phi:.6g}: {exc}",
                flush=True,
            )
        return float("nan")
