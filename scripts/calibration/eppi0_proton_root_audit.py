from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .elastic_momentum import apply_supported_particle_correction
from .elastic_run_validation import filter_arrays_by_run_classes, load_run_catalog
from .eppi0_momentum_validation import (
    ELECTRON_MASS_GEV,
    PROTON_MASS_GEV,
    _four_vector,
    _unit_vectors,
    load_aligned_selection_mask,
    load_eppi0_arrays,
)
from .exclusive_particle_momentum import (
    ExclusiveFitConfig,
    _base_mask,
    _read_parameters,
    proton_momentum_roots_eppi0,
)
from .plot_utils import save_plot


def _summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"entries": 0}
    quantiles = np.quantile(finite, (0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99))
    return {
        "entries": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "minimum": float(np.min(finite)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q16": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q84": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(np.max(finite)),
    }


def _nearest_root_choice(
    roots: np.ndarray,
    valid: np.ndarray,
    momentum: np.ndarray,
) -> np.ndarray:
    distance = np.where(valid, np.abs(roots - momentum[:, None]), np.inf)
    choice = np.argmin(distance, axis=1)
    choice[~np.any(valid, axis=1)] = -1
    return choice


def photon_branch_metrics(
    arrays: dict[str, np.ndarray],
    electron_momentum: np.ndarray,
    roots: np.ndarray,
    valid: np.ndarray,
    beam_energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare each implied missing pi0 with the measured diphoton system."""
    electron = _four_vector(
        electron_momentum, arrays["electronTheta"], arrays["electronPhi"],
        ELECTRON_MASS_GEV,
    )
    hadronic = np.zeros_like(electron)
    hadronic[:, 0] = beam_energy + PROTON_MASS_GEV
    hadronic[:, 3] = beam_energy
    hadronic -= electron
    proton_direction = _unit_vectors(arrays["protonTheta"], arrays["protonPhi"])
    proton_energy = np.sqrt(np.square(roots) + PROTON_MASS_GEV**2)
    proton = np.zeros((roots.shape[0], roots.shape[1], 4), dtype=float)
    proton[:, :, 0] = proton_energy
    proton[:, :, 1:] = roots[:, :, None] * proton_direction[:, None, :]
    implied_pi0 = hadronic[:, None, :] - proton

    gamma1 = _four_vector(
        arrays["gamma1P"], arrays["gamma1Theta"], arrays["gamma1Phi"], 0.0
    )
    gamma2 = _four_vector(
        arrays["gamma2P"], arrays["gamma2Theta"], arrays["gamma2Phi"], 0.0
    )
    diphoton = gamma1 + gamma2
    observed_vector = diphoton[:, 1:]
    observed_norm = np.linalg.norm(observed_vector, axis=1)
    implied_vector = implied_pi0[:, :, 1:]
    implied_norm = np.linalg.norm(implied_vector, axis=2)
    denominator = implied_norm * observed_norm[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(
            implied_vector * observed_vector[:, None, :], axis=2
        ) / denominator
    angle_deg = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    momentum_residual = np.linalg.norm(
        implied_vector - observed_vector[:, None, :], axis=2
    )
    invalid = (
        ~valid | ~np.isfinite(angle_deg) | ~np.isfinite(momentum_residual) |
        (denominator <= 0.0)
    )
    angle_deg[invalid] = np.inf
    momentum_residual[invalid] = np.inf
    return angle_deg, momentum_residual


def summarize_root_region(
    roots: np.ndarray,
    valid: np.ndarray,
    measured: np.ndarray,
    rows: np.ndarray,
    *,
    perturbation_fraction: float,
    corrected_momentum: np.ndarray | None = None,
    correction_support: np.ndarray | None = None,
    correction_fraction: np.ndarray | None = None,
    photon_direction_angle_deg: np.ndarray | None = None,
    photon_momentum_residual_gev: np.ndarray | None = None,
) -> dict[str, object]:
    rows = np.asarray(rows, dtype=bool)
    count = int(np.count_nonzero(rows))
    if count == 0:
        return {"entries": 0}
    region_roots = np.asarray(roots, dtype=float)[rows]
    region_valid = np.asarray(valid, dtype=bool)[rows]
    region_measured = np.asarray(measured, dtype=float)[rows]
    valid_count = np.sum(region_valid, axis=1)
    ambiguous = valid_count == 2
    nominal_choice = _nearest_root_choice(
        region_roots, region_valid, region_measured
    )
    selected = np.full(region_measured.shape, np.nan)
    has_root = nominal_choice >= 0
    selected[has_root] = region_roots[
        np.arange(region_roots.shape[0])[has_root], nominal_choice[has_root]
    ]
    ordered = np.sort(np.where(region_valid, region_roots, np.inf), axis=1)
    lower = ordered[:, 0]
    upper = ordered[:, 1]
    separation = upper - lower
    boundary_distance = np.abs(region_measured - 0.5 * (lower + upper))
    alternate = np.where(nominal_choice == 0, region_roots[:, 1], region_roots[:, 0])
    chosen_distance = np.abs(selected - region_measured)
    alternate_distance = np.abs(alternate - region_measured)

    minus_choice = _nearest_root_choice(
        region_roots, region_valid,
        region_measured * (1.0 - perturbation_fraction),
    )
    plus_choice = _nearest_root_choice(
        region_roots, region_valid,
        region_measured * (1.0 + perturbation_fraction),
    )
    ambiguous_entries = int(np.count_nonzero(ambiguous))

    def fraction(mask: np.ndarray, denominator: int = count) -> float:
        return float(np.count_nonzero(mask) / denominator) if denominator else 0.0

    result: dict[str, object] = {
        "entries": count,
        "noPhysicalRootFraction": fraction(valid_count == 0),
        "singlePhysicalRootFraction": fraction(valid_count == 1),
        "ambiguousRootFraction": fraction(ambiguous),
        "lowerRootChosenFractionOfAmbiguous": (
            fraction(ambiguous & (nominal_choice == 0), ambiguous_entries)
        ),
        "rootSeparationGeV": _summary(separation[ambiguous]),
        "rootSeparationOverMeasured": _summary(
            separation[ambiguous] / region_measured[ambiguous]
        ),
        "distanceToBranchBoundaryGeV": _summary(boundary_distance[ambiguous]),
        "distanceToBranchBoundaryOverMeasured": _summary(
            boundary_distance[ambiguous] / region_measured[ambiguous]
        ),
        "chosenRootAbsDistanceGeV": _summary(chosen_distance[ambiguous]),
        "alternateRootAbsDistanceGeV": _summary(alternate_distance[ambiguous]),
        "chosenFractionalResidual": _summary(
            selected[ambiguous] / region_measured[ambiguous] - 1.0
        ),
        "fractionWithinBranchBoundaryGeV": {
            f"{threshold:.2f}": fraction(
                ambiguous & (boundary_distance < threshold), ambiguous_entries
            )
            for threshold in (0.01, 0.02, 0.05, 0.10)
        },
        "branchFlipFractionOfAmbiguous": {
            "minusPerturbation": fraction(
                ambiguous & (minus_choice != nominal_choice), ambiguous_entries
            ),
            "plusPerturbation": fraction(
                ambiguous & (plus_choice != nominal_choice), ambiguous_entries
            ),
            "eitherPerturbation": fraction(
                ambiguous & (
                    (minus_choice != nominal_choice) | (plus_choice != nominal_choice)
                ),
                ambiguous_entries,
            ),
        },
    }
    if photon_direction_angle_deg is not None:
        direction_angle = np.asarray(photon_direction_angle_deg, dtype=float)[rows]
        direction_choice = np.argmin(direction_angle, axis=1)
        direction_finite = np.all(np.isfinite(direction_angle), axis=1)
        selected_direction_angle = np.full(region_measured.shape, np.nan)
        alternate_direction_angle = np.full(region_measured.shape, np.nan)
        branch_rows = ambiguous & direction_finite
        indices = np.flatnonzero(branch_rows)
        selected_direction_angle[indices] = direction_angle[
            indices, nominal_choice[indices]
        ]
        alternate_direction_angle[indices] = direction_angle[
            indices, 1 - nominal_choice[indices]
        ]
        direction_gap = np.abs(direction_angle[:, 1] - direction_angle[:, 0])
        direction_entries = int(np.count_nonzero(branch_rows))
        direction_result: dict[str, object] = {
            "entries": direction_entries,
            "choiceAgreementFraction": fraction(
                branch_rows & (direction_choice == nominal_choice), direction_entries
            ),
            "nearestMomentumRootAngleDeg": _summary(
                selected_direction_angle[branch_rows]
            ),
            "alternateRootAngleDeg": _summary(
                alternate_direction_angle[branch_rows]
            ),
            "choiceAngleGapDeg": _summary(direction_gap[branch_rows]),
            "nearestChoiceAdvantageDeg": _summary(
                alternate_direction_angle[branch_rows] -
                selected_direction_angle[branch_rows]
            ),
            "indecisiveFractionByGapDeg": {},
            "decisiveChoiceAgreementByGapDeg": {},
        }
        for threshold in (0.1, 0.5, 1.0, 2.0, 5.0):
            key = f"{threshold:.1f}"
            decisive = branch_rows & (direction_gap >= threshold)
            decisive_entries = int(np.count_nonzero(decisive))
            direction_result["indecisiveFractionByGapDeg"][key] = fraction(
                branch_rows & (direction_gap < threshold), direction_entries
            )
            direction_result["decisiveChoiceAgreementByGapDeg"][key] = (
                fraction(
                    decisive & (direction_choice == nominal_choice),
                    decisive_entries,
                ) if decisive_entries else None
            )
        result["photonDirectionCrossCheck"] = direction_result
    if photon_momentum_residual_gev is not None:
        momentum_residual = np.asarray(
            photon_momentum_residual_gev, dtype=float
        )[rows]
        momentum_choice = np.argmin(momentum_residual, axis=1)
        momentum_finite = np.all(np.isfinite(momentum_residual), axis=1)
        branch_rows = ambiguous & momentum_finite
        momentum_entries = int(np.count_nonzero(branch_rows))
        result["photonMomentumCrossCheck"] = {
            "entries": momentum_entries,
            "choiceAgreementFraction": fraction(
                branch_rows & (momentum_choice == nominal_choice),
                momentum_entries,
            ),
            "choiceResidualGapGeV": _summary(
                np.abs(momentum_residual[branch_rows, 1] -
                       momentum_residual[branch_rows, 0])
            ),
        }
    if corrected_momentum is not None and correction_support is not None:
        region_corrected = np.asarray(corrected_momentum, dtype=float)[rows]
        region_support = np.asarray(correction_support, dtype=bool)[rows]
        corrected_choice = _nearest_root_choice(
            region_roots, region_valid, region_corrected
        )
        supported_ambiguous = ambiguous & region_support
        supported_entries = int(np.count_nonzero(supported_ambiguous))
        correction_result: dict[str, object] = {
            "supportFraction": fraction(region_support),
            "ambiguousSupportedEntries": supported_entries,
            "branchFlipFractionOfAmbiguousSupported": fraction(
                supported_ambiguous & (corrected_choice != nominal_choice),
                supported_entries,
            ),
        }
        if correction_fraction is not None:
            region_correction = np.asarray(correction_fraction, dtype=float)[rows]
            correction_result["fractionalCorrection"] = _summary(
                region_correction[region_support]
            )
        result["provisionalCorrection"] = correction_result
    return result


def _region_rows(arrays: dict[str, np.ndarray], base: np.ndarray) -> list[tuple[str, np.ndarray]]:
    detector = np.asarray(arrays["protonDet"], dtype=int)
    sector = np.asarray(arrays.get("protonSector", np.zeros(base.size)), dtype=int)
    regions = [("overall", base)]
    for detector_id in sorted(int(value) for value in np.unique(detector[base])):
        detector_rows = base & (detector == detector_id)
        regions.append((f"det{detector_id}", detector_rows))
        if detector_id == 1:
            for sector_id in range(1, 7):
                rows = detector_rows & (sector == sector_id)
                if np.any(rows):
                    regions.append((f"det1_sector{sector_id}", rows))
    return regions


def _make_plots(
    roots: np.ndarray,
    valid: np.ndarray,
    measured: np.ndarray,
    detector: np.ndarray,
    base: np.ndarray,
    photon_direction_angle_deg: np.ndarray,
    report: dict[str, object],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    ambiguous = base & (np.sum(valid, axis=1) == 2)
    ordered = np.sort(np.where(valid, roots, np.inf), axis=1)
    separation = ordered[:, 1] - ordered[:, 0]
    boundary = np.abs(measured - 0.5 * (ordered[:, 0] + ordered[:, 1]))
    nominal_choice = _nearest_root_choice(roots, valid, measured)
    selected = np.full(measured.shape, np.nan)
    alternate = np.full(measured.shape, np.nan)
    indices = np.flatnonzero(ambiguous)
    selected[indices] = roots[indices, nominal_choice[indices]]
    alternate[indices] = roots[indices, 1 - nominal_choice[indices]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(314159)
    for detector_id, color in ((1, "tab:blue"), (2, "tab:orange")):
        rows = np.flatnonzero(ambiguous & (detector == detector_id))
        if rows.size > 40_000:
            rows = rng.choice(rows, 40_000, replace=False)
        axes[0, 0].scatter(
            measured[rows], selected[rows], s=2, alpha=0.08,
            color=color, label=f"det{detector_id}",
        )
        axes[0, 1].scatter(
            measured[rows], alternate[rows], s=2, alpha=0.08,
            color=color, label=f"det{detector_id}",
        )
        all_rows = ambiguous & (detector == detector_id)
        axes[1, 0].hist(
            separation[all_rows], bins=100, histtype="step", density=True,
            color=color, label=f"det{detector_id}",
        )
        axes[1, 1].hist(
            boundary[all_rows], bins=100, histtype="step", density=True,
            color=color, label=f"det{detector_id}",
        )
    finite_values = np.concatenate((measured[ambiguous], selected[ambiguous]))
    diagonal_min = float(np.nanquantile(finite_values, 0.001))
    diagonal_max = float(np.nanquantile(finite_values, 0.999))
    for axis, title in (
        (axes[0, 0], "selected nearest root"),
        (axes[0, 1], "rejected alternate root"),
    ):
        axis.plot(
            [diagonal_min, diagonal_max], [diagonal_min, diagonal_max],
            color="black", linewidth=1, linestyle="--", label="root = measured",
        )
        axis.set_xlabel("measured proton momentum [GeV]")
        axis.set_ylabel("kinematic root [GeV]")
        axis.legend(markerscale=3, fontsize=8)
        axis.set_title(title)
    axes[1, 0].set_xlabel("root separation [GeV]")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend()
    axes[1, 0].set_title("root separation")
    axes[1, 1].set_xlabel("distance to branch boundary [GeV]")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend()
    axes[1, 1].set_title("branch-choice margin")
    save_plot(
        fig, output_dir / "proton_root_audit.png",
        "ep-pi0 proton kinematic-root audit", dataset_tag, beam_energy,
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for detector_id, color in ((1, "tab:blue"), (2, "tab:orange")):
        rows = ambiguous & (detector == detector_id) & np.all(
            np.isfinite(photon_direction_angle_deg), axis=1
        )
        row_indices = np.flatnonzero(rows)
        nearest_angle = photon_direction_angle_deg[
            row_indices, nominal_choice[row_indices]
        ]
        alternate_angle = photon_direction_angle_deg[
            row_indices, 1 - nominal_choice[row_indices]
        ]
        axes[0].hist(
            nearest_angle, bins=100, range=(0.0, 30.0), histtype="step",
            density=True, color=color, label=f"det{detector_id} nearest root",
        )
        axes[0].hist(
            alternate_angle, bins=100, range=(0.0, 30.0), histtype="step",
            density=True, color=color, linestyle="--",
            label=f"det{detector_id} alternate",
        )
        advantage = alternate_angle - nearest_angle
        axes[1].hist(
            advantage, bins=120, range=(-30.0, 30.0), histtype="step",
            density=True, color=color, label=f"det{detector_id}",
        )
    axes[0].set_xlabel(r"angle(implied $\pi^0$, measured $\gamma\gamma$) [deg]")
    axes[0].set_ylabel("density")
    axes[0].set_title("independent diphoton-direction check")
    axes[0].legend(fontsize=7)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("alternate angle - nearest-root angle [deg]")
    axes[1].set_ylabel("density")
    axes[1].set_title("positive favors nearest-momentum root")
    axes[1].legend(fontsize=8)

    region_names = [name for name in ("det1", "det2") if name in report["regions"]]
    x = np.arange(len(region_names))
    direction_agreement = [
        100.0 * report["regions"][name]["photonDirectionCrossCheck"][
            "choiceAgreementFraction"
        ] for name in region_names
    ]
    momentum_agreement = [
        100.0 * report["regions"][name]["photonMomentumCrossCheck"][
            "choiceAgreementFraction"
        ] for name in region_names
    ]
    width = 0.35
    axes[2].bar(
        x - width / 2, direction_agreement, width, label="diphoton direction"
    )
    axes[2].bar(
        x + width / 2, momentum_agreement, width, label="diphoton momentum"
    )
    axes[2].set_xticks(x, region_names)
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_ylabel("agreement with nearest-p root [%]")
    axes[2].set_title("independent branch agreement")
    axes[2].legend(fontsize=8)
    save_plot(
        fig, output_dir / "proton_root_photon_agreement.png",
        "ep-pi0 proton-root agreement with measured diphoton",
        dataset_tag, beam_energy,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the two proton-momentum roots in ep-pi0 kinematics."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--electron-parameters", type=Path, required=True)
    parser.add_argument("--proton-parameters", type=Path)
    parser.add_argument("--selection-mask", type=Path)
    parser.add_argument("--selection-mask-key", default="mask")
    parser.add_argument("--run-catalog", type=Path)
    parser.add_argument("--include-run-classes", nargs="+")
    parser.add_argument("--perturbation-fraction", type=float, default=0.05)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--mgg-max-abs-gev", type=float, default=0.08)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.perturbation_fraction < 1.0:
        raise ValueError("perturbation fraction must be in (0, 1)")
    if args.include_run_classes and args.run_catalog is None:
        raise ValueError("--include-run-classes requires --run-catalog")
    arrays = load_eppi0_arrays(args.input_file, args.tree, args.max_rows)
    entries = int(np.asarray(arrays["electronP"]).size)
    electron_parameters = _read_parameters(args.electron_parameters)
    electron_p, electron_support, _ = apply_supported_particle_correction(
        arrays["electronP"], arrays["electronTheta"], arrays["electronPhi"],
        arrays["electronDet"], arrays["electronSector"],
        pid=11, parameters=electron_parameters,
    )
    cfg = ExclusiveFitConfig(
        beam_energy=args.beam_energy, torus=args.torus,
        minimum_q2=args.min_q2, minimum_w=args.min_w,
        mgg_max_abs_gev=args.mgg_max_abs_gev,
    )
    base, _ = _base_mask(arrays, electron_p, cfg)
    base &= electron_support
    selection_entries = None
    if args.selection_mask is not None:
        selection_mask = load_aligned_selection_mask(
            args.selection_mask, entries, args.selection_mask_key,
            allow_prefix=args.max_rows is not None,
        )
        selection_entries = int(np.count_nonzero(selection_mask))
        base &= selection_mask
    run_selection = None
    if args.run_catalog is not None:
        if "runNum" not in arrays:
            raise ValueError("candidate tree requires runNum for run-class filtering")
        run_numbers = np.asarray(arrays["runNum"], dtype=int)
        _, selected_mapping, run_selection = filter_arrays_by_run_classes(
            {"runNum": run_numbers.copy()}, load_run_catalog(args.run_catalog),
            args.include_run_classes, catalog_path=args.run_catalog,
        )
        base &= np.isin(run_numbers, np.asarray(sorted(selected_mapping)))

    roots, valid = proton_momentum_roots_eppi0(
        electron_p, arrays["electronTheta"], arrays["electronPhi"],
        arrays["protonTheta"], arrays["protonPhi"], args.beam_energy,
    )
    photon_direction_angle_deg, photon_momentum_residual_gev = (
        photon_branch_metrics(
            arrays, electron_p, roots, valid, args.beam_energy
        )
    )
    measured = np.asarray(arrays["protonP"], dtype=float)
    detector = np.asarray(arrays["protonDet"], dtype=int)
    corrected = correction_support = correction = None
    if args.proton_parameters is not None:
        proton_parameters = _read_parameters(args.proton_parameters)
        proton_sector = np.asarray(
            arrays.get("protonSector", np.zeros(entries)), dtype=int
        )
        corrected, correction_support, correction = apply_supported_particle_correction(
            measured, arrays["protonTheta"], arrays["protonPhi"],
            detector, proton_sector, pid=2212, parameters=proton_parameters,
        )

    regions = {
        name: summarize_root_region(
            roots, valid, measured, rows,
            perturbation_fraction=args.perturbation_fraction,
            corrected_momentum=corrected,
            correction_support=correction_support,
            correction_fraction=correction,
            photon_direction_angle_deg=photon_direction_angle_deg,
            photon_momentum_residual_gev=photon_momentum_residual_gev,
        )
        for name, rows in _region_rows(arrays, base)
    }
    report: dict[str, object] = {
        "schema": "eppi0-proton-root-audit/v2",
        "datasetTag": args.dataset_tag,
        "beamEnergyGeV": args.beam_energy,
        "torus": args.torus,
        "inputFile": str(args.input_file),
        "electronParameters": str(args.electron_parameters),
        "protonParameters": (
            str(args.proton_parameters) if args.proton_parameters else None
        ),
        "selectionMask": str(args.selection_mask) if args.selection_mask else None,
        "selection": {
            "inputCandidates": entries,
            "externalSelectionEntries": selection_entries,
            "auditCandidates": int(np.count_nonzero(base)),
            "electronSupportFraction": float(np.mean(electron_support)),
            "perturbationFraction": args.perturbation_fraction,
        },
        "regions": regions,
    }
    if run_selection is not None:
        report["runSelection"] = run_selection
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "proton_root_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output_dir / "proton_root_audit.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow([
            "region", "entries", "ambiguousRootFraction",
            "medianRootSeparationGeV", "medianBoundaryDistanceGeV",
            "flipMinus", "flipPlus", "flipEither", "provisionalFlip",
            "photonDirectionAgreement", "photonMomentumAgreement",
        ])
        for name, values in regions.items():
            flips = values.get("branchFlipFractionOfAmbiguous", {})
            provisional = values.get("provisionalCorrection", {})
            direction = values.get("photonDirectionCrossCheck", {})
            photon_momentum = values.get("photonMomentumCrossCheck", {})
            writer.writerow([
                name, values.get("entries"), values.get("ambiguousRootFraction"),
                values.get("rootSeparationGeV", {}).get("median"),
                values.get("distanceToBranchBoundaryGeV", {}).get("median"),
                flips.get("minusPerturbation"), flips.get("plusPerturbation"),
                flips.get("eitherPerturbation"),
                provisional.get("branchFlipFractionOfAmbiguousSupported"),
                direction.get("choiceAgreementFraction"),
                photon_momentum.get("choiceAgreementFraction"),
            ])
    _make_plots(
        roots, valid, measured, detector, base, photon_direction_angle_deg, report,
        args.output_dir, args.dataset_tag, args.beam_energy,
    )
    print(f"Wrote proton-root audit to {args.output_dir}")
    for name in ("overall", "det1", "det2"):
        if name not in regions:
            continue
        values = regions[name]
        flips = values["branchFlipFractionOfAmbiguous"]
        direction = values["photonDirectionCrossCheck"]
        print(
            f"  {name}: entries={values['entries']}; "
            f"ambiguous={100.0 * values['ambiguousRootFraction']:.2f}%; "
            f"median separation={values['rootSeparationGeV']['median']:.4f} GeV; "
            f"flip(+/-{100.0 * args.perturbation_fraction:.1f}%)="
            f"{100.0 * flips['eitherPerturbation']:.2f}%; "
            f"photon-direction agreement="
            f"{100.0 * direction['choiceAgreementFraction']:.2f}%"
        )


if __name__ == "__main__":
    main()
