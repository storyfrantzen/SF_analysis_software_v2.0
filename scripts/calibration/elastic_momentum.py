from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, has_column, load_dataframe


PROTON_MASS_GEV = 0.9382720813
RAD_TO_DEG = 180.0 / np.pi


@dataclass(frozen=True)
class ElasticFitConfig:
    beam_energy: float
    torus: int = 0
    coplanarity_max_deg: float = 3.0
    theta_balance_max_deg: float = 2.0
    theta_trim_quantile: float = 0.005
    residual_trim_quantile: float = 0.01
    theta_bins: int = 7
    phi_bins: int = 7
    theta_order: int = 2
    phi_order: int = 2
    cd_fourier_harmonics: int = 3
    min_bin_entries: int = 40
    min_region_entries: int = 800


@dataclass(frozen=True)
class RegionDiagnostics:
    label: str
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    residual_before: np.ndarray
    residual_after: np.ndarray
    profile_theta_deg: np.ndarray
    profile_phi_deg: np.ndarray
    profile_residual: np.ndarray
    region: dict[str, object]


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def sector_local_phi(phi_rad: np.ndarray, sector: np.ndarray) -> np.ndarray:
    return wrap_degrees(np.asarray(phi_rad) * RAD_TO_DEG - 60.0 * (np.asarray(sector) - 1.0))


def elastic_electron_momentum(theta_rad: np.ndarray, beam_energy: float) -> np.ndarray:
    """Massless-electron elastic momentum for a stationary proton target."""
    theta = np.asarray(theta_rad, dtype=float)
    return beam_energy / (
        1.0 + beam_energy * (1.0 - np.cos(theta)) / PROTON_MASS_GEV
    )


def elastic_proton_momentum(theta_rad: np.ndarray, beam_energy: float) -> np.ndarray:
    """Elastic recoil-proton momentum using only its reconstructed polar angle."""
    theta = np.asarray(theta_rad, dtype=float)
    cosine = np.cos(theta)
    numerator = 2.0 * PROTON_MASS_GEV * beam_energy * (beam_energy + PROTON_MASS_GEV) * cosine
    denominator = (
        (beam_energy + PROTON_MASS_GEV) ** 2 - beam_energy**2 * cosine**2
    )
    return numerator / denominator


def elastic_proton_theta_from_electron(
    electron_theta_rad: np.ndarray,
    beam_energy: float,
) -> np.ndarray:
    electron_p = elastic_electron_momentum(electron_theta_rad, beam_energy)
    transverse = electron_p * np.sin(electron_theta_rad)
    longitudinal = beam_energy - electron_p * np.cos(electron_theta_rad)
    return np.arctan2(transverse, longitudinal)


def load_elastic_arrays(input_file: Path, tree: str, max_rows: int | None) -> dict[str, np.ndarray]:
    df = load_dataframe(input_file, tree)
    required = [
        "electronP", "electronTheta", "electronPhi", "electronDet", "electronSector",
        "protonP", "protonTheta", "protonPhi", "protonDet", "protonSector",
    ]
    missing = [column for column in required if not has_column(df, column)]
    if missing:
        raise RuntimeError(
            f"{tree} is missing elastic-candidate branches: {', '.join(missing)}"
        )
    optional = [name for name in ("runNum", "nPid11", "nPid2212") if has_column(df, name)]
    return arrays_from_dataframe(df, required + optional, max_rows=max_rows)


def select_elastic_events(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    electron_theta = np.asarray(arrays["electronTheta"], dtype=float)
    proton_theta = np.asarray(arrays["protonTheta"], dtype=float)
    expected_proton_theta = elastic_proton_theta_from_electron(
        electron_theta, cfg.beam_energy
    )
    coplanarity = wrap_degrees(
        (np.asarray(arrays["protonPhi"]) - np.asarray(arrays["electronPhi"])) *
        RAD_TO_DEG - 180.0
    )
    theta_balance = (proton_theta - expected_proton_theta) * RAD_TO_DEG

    finite_columns = [
        "electronP", "electronTheta", "electronPhi", "electronDet", "electronSector",
        "protonP", "protonTheta", "protonPhi", "protonDet", "protonSector",
    ]
    mask = np.ones(electron_theta.size, dtype=bool)
    for column in finite_columns:
        mask &= np.isfinite(arrays[column])
    mask &= np.asarray(arrays["electronP"]) > 0.0
    mask &= np.asarray(arrays["protonP"]) > 0.0
    mask &= np.asarray(arrays["electronDet"]) == 1
    mask &= (np.asarray(arrays["electronSector"]) >= 1) & (
        np.asarray(arrays["electronSector"]) <= 6
    )
    mask &= np.isin(np.asarray(arrays["protonDet"]), (1, 2))
    mask &= elastic_proton_momentum(proton_theta, cfg.beam_energy) > 0.0
    if "nPid11" in arrays:
        mask &= np.asarray(arrays["nPid11"]) == 1
    if "nPid2212" in arrays:
        mask &= np.asarray(arrays["nPid2212"]) == 1
    preselection = mask.copy()
    mask &= np.abs(coplanarity) <= cfg.coplanarity_max_deg
    mask &= np.abs(theta_balance) <= cfg.theta_balance_max_deg

    selected = {name: np.asarray(values)[mask] for name, values in arrays.items()}
    selected["coplanarityDeg"] = coplanarity[mask]
    selected["thetaBalanceDeg"] = theta_balance[mask]
    summary: dict[str, object] = {
        "inputCandidates": int(mask.size),
        "preselectedCandidates": int(np.count_nonzero(preselection)),
        "selectedCandidates": int(np.count_nonzero(mask)),
        "selectedFraction": float(np.mean(mask)) if mask.size else 0.0,
        "coplanarityMaxDeg": cfg.coplanarity_max_deg,
        "thetaBalanceMaxDeg": cfg.theta_balance_max_deg,
        "preselectionAngularClosure": {
            "coplanarityDeg": _region_summary(coplanarity[preselection]),
            "thetaBalanceDeg": _region_summary(theta_balance[preselection]),
        },
        "selectedAngularClosure": {
            "coplanarityDeg": _region_summary(coplanarity[mask]),
            "thetaBalanceDeg": _region_summary(theta_balance[mask]),
        },
    }
    return selected, summary


def robust_core(values: np.ndarray) -> tuple[float, float, float, int]:
    core = np.asarray(values, dtype=float)
    core = core[np.isfinite(core)]
    if core.size < 2:
        return np.nan, np.nan, np.nan, int(core.size)
    for _ in range(8):
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        retained = core[np.abs(core - median) <= 3.0 * sigma]
        if retained.size == core.size or retained.size < 2:
            break
        core = retained
    center = float(np.mean(core))
    width = float(np.std(core, ddof=1)) if core.size > 1 else np.nan
    error = width / np.sqrt(core.size) if width > 0.0 else np.nan
    return center, error, width, int(core.size)


def _profile_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    residual: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    min_entries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_points: list[float] = []
    phi_points: list[float] = []
    residual_points: list[float] = []
    errors: list[float] = []
    for theta_index, (theta_lo, theta_hi) in enumerate(zip(theta_edges[:-1], theta_edges[1:])):
        theta_mask = (theta >= theta_lo) & (
            (theta <= theta_hi) if theta_index == len(theta_edges) - 2 else (theta < theta_hi)
        )
        for phi_index, (phi_lo, phi_hi) in enumerate(zip(phi_edges[:-1], phi_edges[1:])):
            phi_mask = (phi >= phi_lo) & (
                (phi <= phi_hi) if phi_index == len(phi_edges) - 2 else (phi < phi_hi)
            )
            cell = theta_mask & phi_mask
            if np.count_nonzero(cell) < min_entries:
                continue
            center, error, _, retained = robust_core(residual[cell])
            if retained < min_entries or not np.isfinite(center) or not np.isfinite(error) or error <= 0:
                continue
            theta_points.append(float(np.mean(theta[cell])))
            phi_points.append(float(np.mean(phi[cell])))
            residual_points.append(center)
            errors.append(error)
    return tuple(np.asarray(v, dtype=float) for v in (
        theta_points, phi_points, residual_points, errors
    ))


def _polynomial_matrix(
    theta_normalized: np.ndarray,
    phi_normalized: np.ndarray,
    theta_order: int,
    phi_order: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    definitions = [
        {"thetaPower": theta_power, "phiPower": phi_power}
        for theta_power in range(theta_order + 1)
        for phi_power in range(phi_order + 1)
    ]
    matrix = np.column_stack([
        np.power(theta_normalized, term["thetaPower"]) *
        np.power(phi_normalized, term["phiPower"])
        for term in definitions
    ])
    return matrix, definitions


def _fourier_matrix(
    theta_normalized: np.ndarray,
    phi_deg: np.ndarray,
    theta_order: int,
    harmonics: int,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    columns: list[np.ndarray] = []
    definitions: list[dict[str, int | str]] = []
    phi_rad = np.deg2rad(phi_deg)
    for theta_power in range(theta_order + 1):
        theta_term = np.power(theta_normalized, theta_power)
        columns.append(theta_term)
        definitions.append({
            "thetaPower": theta_power, "component": "constant", "harmonic": 0
        })
        for harmonic in range(1, harmonics + 1):
            columns.append(theta_term * np.cos(harmonic * phi_rad))
            definitions.append({
                "thetaPower": theta_power, "component": "cos", "harmonic": harmonic
            })
            columns.append(theta_term * np.sin(harmonic * phi_rad))
            definitions.append({
                "thetaPower": theta_power, "component": "sin", "harmonic": harmonic
            })
    return np.column_stack(columns), definitions


def evaluate_region(
    region: dict[str, object],
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    theta_normalized = (
        np.asarray(theta_deg) - float(region["thetaCenterDeg"])
    ) / float(region["thetaScaleDeg"])
    terms = list(region["terms"])
    if region["basis"] == "polynomial":
        phi_normalized = (
            np.asarray(phi_deg) - float(region["phiCenterDeg"])
        ) / float(region["phiScaleDeg"])
        return np.sum(np.column_stack([
            float(term["coefficient"]) *
            np.power(theta_normalized, int(term["thetaPower"])) *
            np.power(phi_normalized, int(term["phiPower"]))
            for term in terms
        ]), axis=1)

    phi_rad = np.deg2rad(phi_deg)
    columns: list[np.ndarray] = []
    for term in terms:
        component = str(term["component"])
        harmonic = int(term["harmonic"])
        phi_term = np.ones_like(phi_rad)
        if component == "cos":
            phi_term = np.cos(harmonic * phi_rad)
        elif component == "sin":
            phi_term = np.sin(harmonic * phi_rad)
        columns.append(
            float(term["coefficient"]) *
            np.power(theta_normalized, int(term["thetaPower"])) * phi_term
        )
    return np.sum(np.column_stack(columns), axis=1)


def fit_region(
    *,
    pid: int,
    detector: int,
    sector: int,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    residual: np.ndarray,
    basis: str,
    cfg: ElasticFitConfig,
) -> tuple[dict[str, object], RegionDiagnostics]:
    if cfg.theta_bins < 1 or cfg.phi_bins < 1:
        raise ValueError("theta_bins and phi_bins must be positive")
    if cfg.theta_order < 0 or cfg.phi_order < 0 or cfg.cd_fourier_harmonics < 0:
        raise ValueError("fit orders and Fourier harmonics must be nonnegative")
    if cfg.min_bin_entries < 2 or cfg.min_region_entries < 2:
        raise ValueError("minimum entry counts must be at least two")
    finite = np.isfinite(theta_deg) & np.isfinite(phi_deg) & np.isfinite(residual)
    theta = np.asarray(theta_deg, dtype=float)[finite]
    phi = np.asarray(phi_deg, dtype=float)[finite]
    residual_values = np.asarray(residual, dtype=float)[finite]
    phi_range = (-30.0, 30.0) if basis == "polynomial" else (-180.0, 180.0)
    phi_support = (phi >= phi_range[0]) & (phi <= phi_range[1])
    theta = theta[phi_support]
    phi = phi[phi_support]
    residual_values = residual_values[phi_support]
    if theta.size < cfg.min_region_entries:
        raise ValueError(f"region has only {theta.size} entries")

    q = cfg.theta_trim_quantile
    if not 0.0 <= q < 0.5:
        raise ValueError("theta_trim_quantile must satisfy 0 <= q < 0.5")
    theta_min, theta_max = np.quantile(theta, (q, 1.0 - q))
    if not theta_max > theta_min:
        raise ValueError("region has no finite theta span")
    theta_support = (theta >= theta_min) & (theta <= theta_max)
    theta = theta[theta_support]
    phi = phi[theta_support]
    residual_values = residual_values[theta_support]

    rq = cfg.residual_trim_quantile
    if not 0.0 <= rq < 0.5:
        raise ValueError("residual_trim_quantile must satisfy 0 <= q < 0.5")
    residual_min, residual_max = np.quantile(residual_values, (rq, 1.0 - rq))
    residual_support = (residual_values >= residual_min) & (residual_values <= residual_max)
    theta = theta[residual_support]
    phi = phi[residual_support]
    residual_values = residual_values[residual_support]
    if theta.size < cfg.min_region_entries:
        raise ValueError(f"region has only {theta.size} entries after trimming")

    theta_edges = np.unique(np.quantile(theta, np.linspace(0.0, 1.0, cfg.theta_bins + 1)))
    phi_edges = np.linspace(phi_range[0], phi_range[1], cfg.phi_bins + 1)
    profile_theta, profile_phi, profile_residual, profile_error = _profile_grid(
        theta, phi, residual_values, theta_edges, phi_edges, cfg.min_bin_entries
    )

    theta_center = 0.5 * (theta_min + theta_max)
    theta_scale = 0.5 * (theta_max - theta_min)
    theta_normalized = (profile_theta - theta_center) / theta_scale
    if basis == "polynomial":
        phi_center = 0.0
        phi_scale = 30.0
        matrix, definitions = _polynomial_matrix(
            theta_normalized,
            (profile_phi - phi_center) / phi_scale,
            cfg.theta_order,
            cfg.phi_order,
        )
        phi_variable = "sectorLocal"
    elif basis == "fourier":
        phi_center = 0.0
        phi_scale = 180.0
        matrix, definitions = _fourier_matrix(
            theta_normalized,
            profile_phi,
            cfg.theta_order,
            cfg.cd_fourier_harmonics,
        )
        phi_variable = "global"
    else:
        raise ValueError(f"unsupported basis {basis}")

    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError(
            f"region has {matrix.shape[0]} usable profile cells for {matrix.shape[1]} terms"
        )
    weights = 1.0 / profile_error
    weighted_matrix = matrix * weights[:, None]
    weighted_residual = profile_residual * weights
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_matrix, weighted_residual, rcond=None
    )
    if rank != matrix.shape[1] or not np.all(np.isfinite(coefficients)):
        raise ValueError("region fit is rank deficient")

    terms = [
        {**definition, "coefficient": float(coefficient)}
        for definition, coefficient in zip(definitions, coefficients)
    ]
    predicted = matrix @ coefficients
    chi2 = float(np.sum(np.square((profile_residual - predicted) / profile_error)))
    ndof = int(matrix.shape[0] - matrix.shape[1])
    region: dict[str, object] = {
        "pid": pid,
        "detector": detector,
        "sector": sector,
        "thetaRangeDeg": [float(theta_min), float(theta_max)],
        "phiRangeDeg": [phi_range[0], phi_range[1]],
        "thetaCenterDeg": float(theta_center),
        "thetaScaleDeg": float(theta_scale),
        "phiCenterDeg": phi_center,
        "phiScaleDeg": phi_scale,
        "phiVariable": phi_variable,
        "basis": basis,
        "terms": terms,
        "fit": {
            "entries": int(theta.size),
            "profileCells": int(matrix.shape[0]),
            "parameters": int(matrix.shape[1]),
            "chi2": chi2,
            "ndof": ndof,
            "chi2PerNdf": chi2 / ndof if ndof > 0 else None,
            "residualCoreRange": [float(residual_min), float(residual_max)],
        },
    }
    fitted_correction = evaluate_region(region, theta, phi)
    label = f"pid{pid}_det{detector}" + (f"_sector{sector}" if sector else "")
    diagnostics = RegionDiagnostics(
        label=label,
        theta_deg=theta,
        phi_deg=phi,
        residual_before=residual_values,
        residual_after=(1.0 + residual_values) / (1.0 + fitted_correction) - 1.0,
        profile_theta_deg=profile_theta,
        profile_phi_deg=profile_phi,
        profile_residual=profile_residual,
        region=region,
    )
    return region, diagnostics


def _region_summary(values: np.ndarray) -> dict[str, float]:
    center, _, width, retained = robust_core(values)
    return {
        "coreMean": center,
        "coreWidth": width,
        "coreEntries": retained,
        "median": float(np.median(values)),
    }


def derive_corrections(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
    particles: Iterable[str] = ("electron", "proton"),
) -> tuple[dict[str, object], list[RegionDiagnostics]]:
    if cfg.torus not in (-1, 1):
        raise ValueError("torus must be -1 or 1 when exporting corrections")
    selected, selection_summary = select_elastic_events(arrays, cfg)
    if selection_summary["selectedCandidates"] == 0:
        raise ValueError("no candidates survive the elastic angular selection")

    regions: list[dict[str, object]] = []
    diagnostics: list[RegionDiagnostics] = []
    skipped: list[dict[str, object]] = []
    requested = set(particles)
    particle_inputs = []
    if "electron" in requested:
        particle_inputs.append((
            "electron", 11,
            np.asarray(selected["electronP"]),
            np.asarray(selected["electronTheta"]),
            np.asarray(selected["electronPhi"]),
            np.asarray(selected["electronDet"]),
            np.asarray(selected["electronSector"]),
            elastic_electron_momentum(selected["electronTheta"], cfg.beam_energy),
        ))
    if "proton" in requested:
        particle_inputs.append((
            "proton", 2212,
            np.asarray(selected["protonP"]),
            np.asarray(selected["protonTheta"]),
            np.asarray(selected["protonPhi"]),
            np.asarray(selected["protonDet"]),
            np.asarray(selected["protonSector"]),
            elastic_proton_momentum(selected["protonTheta"], cfg.beam_energy),
        ))

    for particle_name, pid, momentum, theta_rad, phi_rad, detector, sector, expected in particle_inputs:
        residual = expected / momentum - 1.0
        theta_deg = theta_rad * RAD_TO_DEG
        for detector_id in (1, 2):
            if particle_name == "electron" and detector_id != 1:
                continue
            if detector_id == 1:
                for sector_id in range(1, 7):
                    region_mask = (detector == detector_id) & (sector == sector_id)
                    label = f"{particle_name}_FD_sector{sector_id}"
                    try:
                        region, diagnostic = fit_region(
                            pid=pid,
                            detector=detector_id,
                            sector=sector_id,
                            theta_deg=theta_deg[region_mask],
                            phi_deg=sector_local_phi(phi_rad[region_mask], sector[region_mask]),
                            residual=residual[region_mask],
                            basis="polynomial",
                            cfg=cfg,
                        )
                    except ValueError as error:
                        skipped.append({"region": label, "reason": str(error)})
                        continue
                    region["particle"] = particle_name
                    region["fit"]["before"] = _region_summary(diagnostic.residual_before)
                    region["fit"]["after"] = _region_summary(diagnostic.residual_after)
                    regions.append(region)
                    diagnostics.append(diagnostic)
            else:
                region_mask = detector == detector_id
                label = f"{particle_name}_CD"
                try:
                    region, diagnostic = fit_region(
                        pid=pid,
                        detector=detector_id,
                        sector=0,
                        theta_deg=theta_deg[region_mask],
                        phi_deg=wrap_degrees(phi_rad[region_mask] * RAD_TO_DEG),
                        residual=residual[region_mask],
                        basis="fourier",
                        cfg=cfg,
                    )
                except ValueError as error:
                    skipped.append({"region": label, "reason": str(error)})
                    continue
                region["particle"] = particle_name
                region["fit"]["before"] = _region_summary(diagnostic.residual_before)
                region["fit"]["after"] = _region_summary(diagnostic.residual_after)
                regions.append(region)
                diagnostics.append(diagnostic)

    if not regions:
        reasons = "; ".join(f"{entry['region']}: {entry['reason']}" for entry in skipped)
        raise ValueError(f"no detector region could be fitted ({reasons})")
    output: dict[str, object] = {
        "schema": "elastic_momentum_correction/v1",
        "correctionType": "fractionalMomentum",
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "selection": selection_summary,
        "fitConfiguration": {
            "thetaTrimQuantile": cfg.theta_trim_quantile,
            "residualTrimQuantile": cfg.residual_trim_quantile,
            "thetaBins": cfg.theta_bins,
            "phiBins": cfg.phi_bins,
            "thetaOrder": cfg.theta_order,
            "fdPhiOrder": cfg.phi_order,
            "cdFourierHarmonics": cfg.cd_fourier_harmonics,
            "minBinEntries": cfg.min_bin_entries,
            "minRegionEntries": cfg.min_region_entries,
        },
        "regions": regions,
        "skippedRegions": skipped,
    }
    return output, diagnostics


def plot_diagnostics(
    diagnostics: list[RegionDiagnostics],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for diagnostic in diagnostics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        scatter = axes[0].scatter(
            diagnostic.profile_theta_deg,
            diagnostic.profile_phi_deg,
            c=100.0 * diagnostic.profile_residual,
            cmap="coolwarm",
            s=35,
        )
        axes[0].set_xlabel(r"$\theta$ [deg]")
        axes[0].set_ylabel(r"$\phi$ variable [deg]")
        axes[0].set_title("profile before correction")
        fig.colorbar(scatter, ax=axes[0], label=r"$100(p_{el}/p-1)$ [%]")

        common_range = np.quantile(diagnostic.residual_before, (0.005, 0.995))
        axes[1].hist(
            100.0 * diagnostic.residual_before,
            bins=100,
            range=tuple(100.0 * common_range),
            histtype="step",
            label="before",
        )
        axes[1].hist(
            100.0 * diagnostic.residual_after,
            bins=100,
            range=tuple(100.0 * common_range),
            histtype="step",
            label="after",
        )
        axes[1].set_xlabel(r"$100(p_{el}/p-1)$ [%]")
        axes[1].set_ylabel("candidates")
        axes[1].set_title("closure")
        axes[1].legend()

        axes[2].scatter(
            diagnostic.theta_deg,
            100.0 * diagnostic.residual_after,
            s=1,
            alpha=0.08,
        )
        axes[2].axhline(0.0, color="black", linewidth=1)
        axes[2].set_xlabel(r"$\theta$ [deg]")
        axes[2].set_ylabel("post-correction residual [%]")
        axes[2].set_title("residual closure vs theta")

        save_plot(
            fig,
            output_dir / f"{diagnostic.label}.png",
            f"Elastic momentum correction: {diagnostic.label}",
            dataset_tag,
            beam_energy,
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive electron and proton fractional momentum corrections from "
            "selected elastic ep candidates."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parameters/momentum/elastic_momentum_corrections.json"),
    )
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--particle", choices=("electron", "proton", "both"), default="both")
    parser.add_argument("--coplanarity-max-deg", type=float, default=3.0)
    parser.add_argument("--theta-balance-max-deg", type=float, default=2.0)
    parser.add_argument("--theta-trim-quantile", type=float, default=0.005)
    parser.add_argument("--residual-trim-quantile", type=float, default=0.01)
    parser.add_argument("--theta-bins", type=int, default=7)
    parser.add_argument("--phi-bins", type=int, default=7)
    parser.add_argument("--theta-order", type=int, default=2)
    parser.add_argument("--fd-phi-order", type=int, default=2)
    parser.add_argument("--cd-fourier-harmonics", type=int, default=3)
    parser.add_argument("--min-bin-entries", type=int, default=40)
    parser.add_argument("--min-region-entries", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_energy <= 0.0:
        raise ValueError("beam energy must be positive")
    cfg = ElasticFitConfig(
        beam_energy=args.beam_energy,
        torus=args.torus,
        coplanarity_max_deg=args.coplanarity_max_deg,
        theta_balance_max_deg=args.theta_balance_max_deg,
        theta_trim_quantile=args.theta_trim_quantile,
        residual_trim_quantile=args.residual_trim_quantile,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        theta_order=args.theta_order,
        phi_order=args.fd_phi_order,
        cd_fourier_harmonics=args.cd_fourier_harmonics,
        min_bin_entries=args.min_bin_entries,
        min_region_entries=args.min_region_entries,
    )
    particles = ("electron", "proton") if args.particle == "both" else (args.particle,)
    arrays = load_elastic_arrays(args.input_file, args.tree, args.max_rows)
    correction, diagnostics = derive_corrections(arrays, cfg, particles)
    correction["datasetTag"] = args.dataset_tag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        json.dump(correction, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        f"Wrote {len(correction['regions'])} elastic momentum regions to {args.output}"
    )
    for skipped in correction["skippedRegions"]:
        print(f"Warning: skipped {skipped['region']}: {skipped['reason']}")
    if args.plot_dir:
        plot_diagnostics(diagnostics, args.plot_dir, args.dataset_tag, args.beam_energy)


if __name__ == "__main__":
    main()
