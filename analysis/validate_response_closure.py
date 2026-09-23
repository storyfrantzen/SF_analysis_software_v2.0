#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import AnalysisBinning, from_config
from eppi0.closure import (
    ClosureScanResult,
    SplitClosureInputs,
    aggregate_metrics,
    assess_iteration_coverage,
    deterministic_folds,
    recommend_iterations,
    run_closure_scan,
    stress_weights,
)
from eppi0.phase_space import AnalysisPhaseSpace
from eppi0.root_response import (
    GENERATED_COLUMNS,
    SELECTED_COLUMNS,
    SELECTED_TOPOLOGY_COLUMNS,
    _require_tree,
    _resolve_selected_tree,
    _source_keys,
    _tree_entries,
    _tree_has_columns,
    _truth_inside_mask,
)
from eppi0.topology import detector_topology_id, ft_photon_count


DEFAULT_STRESSES = ("nominal", "q2_tilt", "xb_t_tilt", "phi_harmonic", "combined")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run deterministic split-sample GEMC closure for the response, "
            "unfolding iteration scan, refolding, pulls, and harmonic recovery."
        )
    )
    result.add_argument("converter_root", type=Path, nargs="?")
    result.add_argument("selected_root", type=Path, nargs="?")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--split-input-dir",
        type=Path,
        help=(
            "Reuse split_closure_inputs.npz and fold migration counts from a prior "
            "scan instead of rereading the ROOT trees."
        ),
    )
    result.add_argument("--dictionary", type=Path)
    result.add_argument("--selection-mask", type=Path)
    result.add_argument("--tree", default="sEvents")
    result.add_argument("--generated-tree", default="gEvents")
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--iterations", type=int, nargs="+", default=(0, 1, 2, 4, 8, 12, 25))
    result.add_argument(
        "--stress",
        nargs="+",
        choices=DEFAULT_STRESSES,
        default=DEFAULT_STRESSES,
        help="Truth-shape alternatives used as held-out pseudo-data.",
    )
    result.add_argument("--stress-strength", type=float, default=0.6)
    result.add_argument("--bootstrap", type=int, default=50)
    result.add_argument(
        "--response-uncertainty",
        choices=("analytic-diagonal", "fold-jackknife"),
        default="analytic-diagonal",
        help=(
            "Estimate response-MC uncertainty with the legacy analytic diagonal "
            "approximation or by deleting each response-training fold in turn."
        ),
    )
    result.add_argument("--seed", type=int, default=731_921)
    result.add_argument("--minimum-truth", type=float, default=20.0)
    result.add_argument("--minimum-acceptance", type=float)
    result.add_argument("--minimum-harmonic-points", type=int, default=8)
    result.add_argument("--chunk-size", type=int, default=1_000_000)
    result.add_argument("--progress-chunks", type=int, default=10)
    result.add_argument(
        "--topology-group",
        type=int,
        action="append",
        default=[],
        help=(
            "Restrict the reconstructed numerator to this topology ID. Repeat for a "
            "combined subset; omit for the topology-integrated closure."
        ),
    )
    result.add_argument("--label", default="GEMC split-sample response closure")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    if args.response_uncertainty == "fold-jackknife" and args.folds < 3:
        raise ValueError("--response-uncertainty fold-jackknife requires at least three folds")
    if args.split_input_dir is None and (
        args.converter_root is None or args.selected_root is None
    ):
        raise ValueError(
            "converter_root and selected_root are required unless --split-input-dir is used"
        )
    if args.split_input_dir is not None and (
        args.converter_root is not None or args.selected_root is not None
    ):
        raise ValueError(
            "do not pass converter_root or selected_root with --split-input-dir"
        )
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    with args.config.open(encoding="utf-8") as source:
        config = json.load(source)
    binning = from_config(args.config)
    phase_space = AnalysisPhaseSpace.from_config(config)
    beam_energy = float(config["beam_energy"])
    minimum_acceptance = (
        float(config["minimum_acceptance"])
        if args.minimum_acceptance is None
        else float(args.minimum_acceptance)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_input_dir is not None:
        inputs, scan_metadata = load_split_inputs(args.split_input_dir)
        if inputs.fold_truth_total.shape[0] != args.folds:
            raise ValueError(
                "saved split-input fold count does not match --folds: "
                f"{inputs.fold_truth_total.shape[0]} != {args.folds}"
            )
        if tuple(args.stress) != inputs.stress_names:
            raise ValueError(
                "saved split-input stresses do not match --stress: "
                f"{inputs.stress_names} != {tuple(args.stress)}"
            )
        source_settings = scan_metadata.get("reused_split_settings")
        if isinstance(source_settings, dict):
            expected_settings = {
                "fold_seed": args.seed,
                "stress_strength": args.stress_strength,
                "topology_groups": sorted(set(args.topology_group)),
            }
            for name, expected in expected_settings.items():
                if name in source_settings and source_settings[name] != expected:
                    raise ValueError(
                        f"saved split-input {name} does not match this scan: "
                        f"{source_settings[name]} != {expected}"
                    )
    else:
        inputs, scan_metadata = scan_root_inputs(
            args.converter_root,
            args.selected_root,
            binning,
            phase_space=phase_space,
            beam_energy=beam_energy,
            dictionary=args.dictionary,
            selection_mask_path=args.selection_mask,
            tree=args.tree,
            generated_tree=args.generated_tree,
            number_of_folds=args.folds,
            seed=args.seed,
            stress_names=tuple(args.stress),
            stress_strength=args.stress_strength,
            topology_groups=tuple(sorted(set(args.topology_group))),
            chunk_size=args.chunk_size,
            progress_chunks=args.progress_chunks,
        )
        save_split_inputs(args.output_dir, inputs, scan_metadata)
    result = run_closure_scan(
        inputs,
        binning,
        args.iterations,
        minimum_acceptance=minimum_acceptance,
        minimum_truth=args.minimum_truth,
        bootstrap=args.bootstrap,
        seed=args.seed,
        minimum_harmonic_points=args.minimum_harmonic_points,
        response_uncertainty=args.response_uncertainty,
    )
    summary = save_results(
        args.output_dir,
        result,
        inputs,
        binning,
        args=args,
        scan_metadata=scan_metadata,
        phase_space=phase_space,
        minimum_acceptance=minimum_acceptance,
    )
    render_diagnostics(
        args.output_dir / "closure_diagnostics.pdf",
        result,
        inputs.validation_truth,
        inputs.stress_names,
        binning,
        label=args.label,
    )
    print(f"Recommended iterations by held-out normalized MSE: {result.recommended_iterations}")
    qualified = summary["recommendation"]["coverage_qualified_iterations"]
    if qualified:
        print(
            "Coverage-qualified iterations: "
            + ", ".join(str(value) for value in qualified)
        )
        print(
            "Minimum-MSE coverage-qualified iteration: "
            f"{summary['recommendation']['coverage_qualified_recommendation']}"
        )
    else:
        print("Coverage-qualified iterations: none; do not promote this scan to nominal")
    print(f"Generated rows scanned: {scan_metadata['generated_rows']}")
    print(f"Selected rows retained: {scan_metadata['selected_rows_retained']}")
    print(f"Wrote {args.output_dir / 'closure_summary.json'}")
    print(f"Wrote {args.output_dir / 'closure_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'closure_results.npz'}")
    print(f"Wrote {args.output_dir / 'closure_diagnostics.pdf'}")
    if summary["scope"]["not_tested"]:
        print("Not tested by this utility: " + "; ".join(summary["scope"]["not_tested"]))
    return 0


def scan_root_inputs(
    converter_root: Path,
    selected_root: Path,
    binning: AnalysisBinning,
    *,
    phase_space: AnalysisPhaseSpace,
    beam_energy: float,
    dictionary: Path | None,
    selection_mask_path: Path | None,
    tree: str,
    generated_tree: str,
    number_of_folds: int,
    seed: int,
    stress_names: tuple[str, ...],
    stress_strength: float,
    topology_groups: tuple[int, ...],
    chunk_size: int,
    progress_chunks: int,
) -> tuple[SplitClosureInputs, dict[str, object]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if dictionary is not None:
        status = ROOT.gSystem.Load(str(dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")
    converter_path = str(converter_root.resolve())
    selected_path = str(selected_root.resolve())
    generated_tree = _resolve_selected_tree(ROOT, converter_path, generated_tree)
    _require_tree(ROOT, converter_path, generated_tree, GENERATED_COLUMNS)
    tree = _resolve_selected_tree(ROOT, selected_path, tree)
    selected_entries = _require_tree(ROOT, selected_path, tree, SELECTED_COLUMNS)
    topology_available = _tree_has_columns(
        ROOT, selected_path, tree, SELECTED_TOPOLOGY_COLUMNS
    )
    if topology_groups and not topology_available:
        raise RuntimeError(
            "--topology-group requires pDet, g1Det, and g2Det in the selected tree"
        )
    selected_columns = SELECTED_COLUMNS + (
        SELECTED_TOPOLOGY_COLUMNS if topology_available else []
    )
    selected = ROOT.RDataFrame(tree, selected_path).AsNumpy(selected_columns)
    selected_count = np.asarray(selected["sourceFileId"]).size
    if selected_count != selected_entries:
        raise RuntimeError("selected tree read returned an unexpected number of rows")
    retained = np.ones(selected_count, dtype=bool)
    if selection_mask_path is not None:
        mask = np.asarray(np.load(selection_mask_path, allow_pickle=False), dtype=bool)
        if mask.shape != (selected_count,):
            raise ValueError(
                f"selection mask has {mask.size} rows; expected {selected_count}"
            )
        retained &= mask
    if topology_available:
        topology = detector_topology_id(
            selected["pDet"], ft_photon_count(selected["g1Det"], selected["g2Det"])
        )
        available_topologies = np.unique(topology[retained])
        if topology_groups:
            retained &= np.isin(topology, np.asarray(topology_groups, dtype=np.int64))
    else:
        available_topologies = np.empty(0, dtype=np.int64)

    selected_keys = _source_keys(
        selected["sourceFileId"][retained], selected["sourceEventIndex"][retained]
    )
    selected_rec_flat = binning.coordinates_to_flat(
        selected["Q2"][retained],
        selected["xB"][retained],
        selected["t"][retained],
        selected["trentoPhi"][retained],
    )
    if np.unique(selected_keys).size != selected_keys.size:
        raise ValueError("selected ROOT sample contains duplicate source keys")
    order = np.argsort(selected_keys, order=selected_keys.dtype.names)
    selected_keys = selected_keys[order]
    selected_rec_flat = selected_rec_flat[order]

    generated_entries = _tree_entries(ROOT, converter_path, generated_tree)
    bins = binning.size
    stresses = len(stress_names)
    fold_truth = np.zeros((number_of_folds, bins), dtype=float)
    fold_reconstructed = np.zeros_like(fold_truth)
    fold_feed = np.zeros_like(fold_truth)
    rows: list[list[np.ndarray]] = [[] for _ in range(number_of_folds)]
    cols: list[list[np.ndarray]] = [[] for _ in range(number_of_folds)]
    migration_weights: list[list[np.ndarray]] = [[] for _ in range(number_of_folds)]
    validation_truth = np.zeros((stresses, number_of_folds, bins), dtype=float)
    validation_measured = np.zeros_like(validation_truth)
    validation_variance = np.zeros_like(validation_truth)
    matched_selected_rows = 0
    q2_range = (float(binning.q2_edges[0]), float(binning.q2_edges[-1]))
    xb_range = (float(binning.xb_edges[0]), float(binning.xb_edges[-1]))
    t_range = (float(binning.t_edges[0]), float(binning.t_edges[-1]))

    for chunk_index, start in enumerate(range(0, generated_entries, chunk_size), start=1):
        stop = min(start + chunk_size, generated_entries)
        chunk = ROOT.RDataFrame(generated_tree, converter_path).Range(start, stop).AsNumpy(
            GENERATED_COLUMNS
        )
        valid_generator = np.asarray(chunk["topologyValid"], dtype=bool)
        truth_flat = binning.coordinates_to_flat(
            chunk["Q2"], chunk["xB"], chunk["minusT"], chunk["trentoPhi"]
        )
        base_weight = np.asarray(chunk["weight"], dtype=float)
        if np.any(~np.isfinite(base_weight)) or np.any(base_weight < 0.0):
            raise ValueError("generated weights must be finite and nonnegative")
        truth_inside = _truth_inside_mask(
            valid_generator,
            truth_flat,
            chunk["Q2"],
            chunk["xB"],
            bins,
            phase_space=phase_space,
            beam_energy=beam_energy,
        )
        folds = deterministic_folds(
            chunk["sourceFileId"], chunk["sourceEventIndex"], number_of_folds, seed
        )
        factors = stress_weights(
            chunk["Q2"],
            chunk["xB"],
            chunk["minusT"],
            chunk["trentoPhi"],
            stress_names,
            strength=stress_strength,
            q2_range=q2_range,
            xb_range=xb_range,
            t_range=t_range,
        )
        stressed_weight = factors * base_weight[None, :]
        for fold in range(number_of_folds):
            truth_rows = truth_inside & (folds == fold)
            if np.any(truth_rows):
                fold_truth[fold] += np.bincount(
                    truth_flat[truth_rows],
                    weights=base_weight[truth_rows],
                    minlength=bins,
                )
                for stress_index in range(stresses):
                    validation_truth[stress_index, fold] += np.bincount(
                        truth_flat[truth_rows],
                        weights=stressed_weight[stress_index, truth_rows],
                        minlength=bins,
                    )

        if selected_keys.size:
            gen_keys = _source_keys(chunk["sourceFileId"], chunk["sourceEventIndex"])
            positions = np.searchsorted(selected_keys, gen_keys)
            bounded = positions < selected_keys.size
            matched = np.zeros(gen_keys.size, dtype=bool)
            matched[bounded] = selected_keys[positions[bounded]] == gen_keys[bounded]
            if np.any(matched):
                matched_positions = positions[matched]
                rec_flat = selected_rec_flat[matched_positions]
                matched_valid = valid_generator[matched]
                rec_inside = matched_valid & (rec_flat >= 0) & (rec_flat < bins)
                matched_selected_rows += int(np.count_nonzero(rec_inside))
                matched_fold = folds[matched]
                matched_base_weight = base_weight[matched]
                matched_truth_flat = truth_flat[matched]
                matched_truth_inside = truth_inside[matched]
                matched_stressed = stressed_weight[:, matched]
                for fold in range(number_of_folds):
                    rec_rows = rec_inside & (matched_fold == fold)
                    if not np.any(rec_rows):
                        continue
                    fold_reconstructed[fold] += np.bincount(
                        rec_flat[rec_rows],
                        weights=matched_base_weight[rec_rows],
                        minlength=bins,
                    )
                    for stress_index in range(stresses):
                        event_weights = matched_stressed[stress_index, rec_rows]
                        validation_measured[stress_index, fold] += np.bincount(
                            rec_flat[rec_rows], weights=event_weights, minlength=bins
                        )
                        validation_variance[stress_index, fold] += np.bincount(
                            rec_flat[rec_rows], weights=event_weights**2, minlength=bins
                        )
                    migrated = rec_rows & matched_truth_inside
                    if np.any(migrated):
                        rows[fold].append(rec_flat[migrated])
                        cols[fold].append(matched_truth_flat[migrated])
                        migration_weights[fold].append(matched_base_weight[migrated])
                    feed = rec_rows & ~matched_truth_inside
                    if np.any(feed):
                        fold_feed[fold] += np.bincount(
                            rec_flat[feed],
                            weights=matched_base_weight[feed],
                            minlength=bins,
                        )
        if progress_chunks > 0 and chunk_index % progress_chunks == 0:
            print(
                f"[PROGRESS] generated rows {stop}/{generated_entries} "
                f"({100.0 * stop / max(generated_entries, 1):.1f}%)"
            )

    matrices = tuple(
        csr_matrix(
            (
                _concat(migration_weights[fold], dtype=float),
                (
                    _concat(rows[fold], dtype=np.int64),
                    _concat(cols[fold], dtype=np.int64),
                ),
            ),
            shape=(bins, bins),
        )
        for fold in range(number_of_folds)
    )
    inputs = SplitClosureInputs(
        fold_truth_total=fold_truth,
        fold_reconstructed_total=fold_reconstructed,
        fold_feed_counts=fold_feed,
        fold_migration_counts=matrices,
        validation_truth=validation_truth,
        validation_measured=validation_measured,
        validation_variance=validation_variance,
        stress_names=stress_names,
    )
    inputs.validate()
    metadata: dict[str, object] = {
        "generated_rows": generated_entries,
        "selected_rows_total": selected_count,
        "selected_rows_retained": int(np.count_nonzero(retained)),
        "matched_selected_rows": matched_selected_rows,
        "available_topology_groups": available_topologies.astype(int).tolist(),
        "selected_topology_groups": list(topology_groups),
        "generated_tree": generated_tree,
        "selected_tree": tree,
    }
    return inputs, metadata


def save_split_inputs(
    output_dir: Path,
    inputs: SplitClosureInputs,
    scan_metadata: dict[str, object],
) -> None:
    np.savez_compressed(
        output_dir / "split_closure_inputs.npz",
        fold_truth_total=inputs.fold_truth_total,
        fold_reconstructed_total=inputs.fold_reconstructed_total,
        fold_feed_counts=inputs.fold_feed_counts,
        validation_truth=inputs.validation_truth,
        validation_measured=inputs.validation_measured,
        validation_variance=inputs.validation_variance,
        stress_names=np.asarray(inputs.stress_names),
        scan_metadata_json=np.asarray(json.dumps(scan_metadata, sort_keys=True)),
    )
    for fold, matrix in enumerate(inputs.fold_migration_counts):
        save_npz(output_dir / f"fold_{fold:02d}_migration_counts.npz", matrix)


def load_split_inputs(
    input_dir: Path,
) -> tuple[SplitClosureInputs, dict[str, object]]:
    """Load fold-local sufficient statistics saved by an earlier ROOT scan."""
    core_path = input_dir / "split_closure_inputs.npz"
    if not core_path.is_file():
        raise FileNotFoundError(f"missing saved closure inputs: {core_path}")
    with np.load(core_path, allow_pickle=False) as saved:
        fold_truth = np.asarray(saved["fold_truth_total"], dtype=float)
        inputs_without_migrations = {
            "fold_reconstructed_total": np.asarray(
                saved["fold_reconstructed_total"], dtype=float
            ),
            "fold_feed_counts": np.asarray(saved["fold_feed_counts"], dtype=float),
            "validation_truth": np.asarray(saved["validation_truth"], dtype=float),
            "validation_measured": np.asarray(saved["validation_measured"], dtype=float),
            "validation_variance": np.asarray(saved["validation_variance"], dtype=float),
            "stress_names": tuple(str(value) for value in saved["stress_names"].tolist()),
        }
        scan_metadata = json.loads(saved["scan_metadata_json"].item())
    source_summary = input_dir / "closure_summary.json"
    if source_summary.is_file():
        with source_summary.open(encoding="utf-8") as source:
            summary = json.load(source)
        source_settings = summary.get("settings")
        if isinstance(source_settings, dict):
            scan_metadata = dict(scan_metadata)
            scan_metadata["reused_split_settings"] = source_settings
    migrations = []
    for fold in range(fold_truth.shape[0]):
        path = input_dir / f"fold_{fold:02d}_migration_counts.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing saved fold migration counts: {path}")
        migrations.append(load_npz(path).tocsr())
    inputs = SplitClosureInputs(
        fold_truth_total=fold_truth,
        fold_migration_counts=tuple(migrations),
        **inputs_without_migrations,
    )
    inputs.validate()
    return inputs, scan_metadata


def save_results(
    output_dir: Path,
    result: ClosureScanResult,
    inputs: SplitClosureInputs,
    binning: AnalysisBinning,
    *,
    args: argparse.Namespace,
    scan_metadata: dict[str, object],
    phase_space: AnalysisPhaseSpace,
    minimum_acceptance: float,
) -> dict[str, object]:
    metrics_path = output_dir / "closure_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(result.metrics[0].keys()))
        writer.writeheader()
        writer.writerows(result.metrics)
    aggregates = aggregate_metrics(result.metrics)
    iterations = sorted({int(row["iterations"]) for row in result.metrics})
    coverage_assessment = assess_iteration_coverage(result.metrics, iterations)
    coverage_qualified = [
        int(row["iterations"])
        for row in coverage_assessment
        if bool(row["certified"])
    ]
    coverage_recommendation = (
        recommend_iterations(result.metrics, coverage_qualified)
        if coverage_qualified
        else None
    )
    summary: dict[str, object] = {
        "schema_version": 3,
        "method": "source-aware deterministic K-fold held-out GEMC closure",
        "label": args.label,
        "software_revision": _git_revision(),
        "inputs": {
            "converter_root": (
                str(args.converter_root.resolve()) if args.converter_root else None
            ),
            "selected_root": (
                str(args.selected_root.resolve()) if args.selected_root else None
            ),
            "split_input_dir": (
                str(args.split_input_dir.resolve()) if args.split_input_dir else None
            ),
            "split_input_sha256": (
                _sha256(args.split_input_dir / "split_closure_inputs.npz")
                if args.split_input_dir
                else None
            ),
            "split_migration_sha256": (
                [
                    _sha256(
                        args.split_input_dir
                        / f"fold_{fold:02d}_migration_counts.npz"
                    )
                    for fold in range(inputs.fold_truth_total.shape[0])
                ]
                if args.split_input_dir
                else None
            ),
            "analysis_config": str(args.config.resolve()),
            "analysis_config_sha256": _sha256(args.config),
            "selection_mask": str(args.selection_mask.resolve()) if args.selection_mask else None,
            "selection_mask_sha256": _sha256(args.selection_mask) if args.selection_mask else None,
            "dictionary": str(args.dictionary.resolve()) if args.dictionary else None,
        },
        "settings": {
            "folds": args.folds,
            "fold_seed": args.seed,
            "iterations": iterations,
            "bootstrap_experiments": args.bootstrap,
            "bootstrap_prior": (
                "acceptance-corrected fluctuated measured spectrum recomputed "
                "for every positive-iteration replica"
            ),
            "response_uncertainty": args.response_uncertainty,
            "minimum_acceptance": minimum_acceptance,
            "minimum_validation_truth": args.minimum_truth,
            "minimum_harmonic_points": args.minimum_harmonic_points,
            "stress_names": list(inputs.stress_names),
            "stress_strength": args.stress_strength,
            "phase_space": phase_space.description(),
            "topology_groups": sorted(set(args.topology_group)),
            "harmonic_measurement_covariance": (
                "full-estimator bootstrap covariance among phi bins within each "
                "(Q2,xB,-t) cell, including recomputation of the data-derived prior; "
                + (
                    "delete-one-training-fold jackknife response covariance added "
                    "within each phi block; "
                    if args.response_uncertainty == "fold-jackknife"
                    else "analytic response-MC variance added to the diagonal; "
                )
                + "finite-bootstrap precision "
                "uses the per-fit Hartlap (N-p-2)/(N-1) correction"
            ),
        },
        "scan": scan_metadata,
        "recommendation": {
            "iterations": result.recommended_iterations,
            "criterion": "minimum median held-out normalized MSE across folds and stresses",
            "coverage_qualified_iterations": coverage_qualified,
            "coverage_qualified_recommendation": coverage_recommendation,
            "coverage_certified": result.recommended_iterations in coverage_qualified,
            "coverage_assessment": coverage_assessment,
            "warning": (
                "The minimum-MSE iteration is a production candidate only when "
                "coverage_certified is true. Otherwise repair or augment the covariance "
                "and rerun closure; do not promote the scan to nominal."
            ),
        },
        "aggregate_metrics": aggregates,
        "scope": {
            "tested": [
                "response construction from training folds",
                "feed-in subtraction",
                "bin-by-bin and iterative Bayesian unfolding",
                "held-out truth recovery under nominal and stressed truth shapes",
                "reconstructed-level refolding",
                "A + B cos(phi) + C cos(2 phi) harmonic recovery",
            ],
            "not_tested": [
                "data m_gg sideband subtraction",
                "beam-charge and current-efficiency corrections",
                "radiative and bin-centering correction accuracy",
                "absolute luminosity normalization",
                "beam-spin asymmetry helicity and polarization handling",
                "detector mismodeling between data and GEMC",
            ],
        },
    }
    serializable_summary = _finite_json(summary)
    with (output_dir / "closure_summary.json").open("w", encoding="utf-8") as destination:
        json.dump(serializable_summary, destination, indent=2, allow_nan=False)
        destination.write("\n")
    np.savez_compressed(
        output_dir / "closure_results.npz",
        unfolded=result.unfolded,
        uncertainty=result.uncertainty,
        validity=result.validity,
        refolded=result.refolded,
        training_efficiency=result.training_efficiency,
        validation_truth=inputs.validation_truth,
        validation_measured=inputs.validation_measured,
        validation_variance=inputs.validation_variance,
        stress_names=np.asarray(inputs.stress_names),
        iterations=np.asarray(iterations, dtype=np.int32),
        recommended_iterations=np.asarray(result.recommended_iterations, dtype=np.int32),
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        summary_json=np.asarray(json.dumps(serializable_summary, sort_keys=True)),
    )
    return serializable_summary


def render_diagnostics(
    path: Path,
    result: ClosureScanResult,
    validation_truth: np.ndarray,
    stress_names: tuple[str, ...],
    binning: AnalysisBinning,
    *,
    label: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    aggregates = aggregate_metrics(result.metrics)
    iteration_values = sorted({int(row["iterations"]) for row in result.metrics})
    recommended_index = iteration_values.index(result.recommended_iterations)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
        fig.subplots_adjust(
            left=0.085,
            right=0.955,
            bottom=0.085,
            top=0.88,
            wspace=0.27,
            hspace=0.30,
        )
        for stress in stress_names:
            rows = sorted(
                [row for row in aggregates if row["stress"] == stress],
                key=lambda row: int(row["iterations"]),
            )
            x = [int(row["iterations"]) for row in rows]
            axes[0, 0].plot(
                x, [row["normalized_mse"] for row in rows], marker="o", label=stress
            )
            axes[0, 1].plot(
                x,
                [row["global_relative_bias"] for row in rows],
                marker="o",
                label=stress,
            )
            axes[1, 0].plot(x, [row["pull_mean"] for row in rows], marker="o", label=stress)
            axes[1, 1].plot(x, [row["pull_std"] for row in rows], marker="o", label=stress)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylabel("normalized MSE")
        axes[0, 1].set_ylabel("global relative bias")
        axes[1, 0].set_ylabel("pull mean")
        axes[1, 1].set_ylabel("pull width")
        pull_means = np.asarray(
            [float(row["pull_mean"]) for row in aggregates], dtype=float
        )
        finite_pull_means = pull_means[np.isfinite(pull_means)]
        if finite_pull_means.size and np.max(np.abs(finite_pull_means)) > 10.0:
            axes[1, 0].set_yscale("symlog", linthresh=1.0, linscale=1.0)
            axes[1, 0].set_ylabel("pull mean (symmetric log scale)")
        pull_widths = np.asarray(
            [float(row["pull_std"]) for row in aggregates], dtype=float
        )
        finite_pull_widths = pull_widths[np.isfinite(pull_widths)]
        positive_pull_widths = finite_pull_widths[finite_pull_widths > 0.0]
        if (
            positive_pull_widths.size == finite_pull_widths.size
            and positive_pull_widths.size > 1
            and np.max(positive_pull_widths) / np.min(positive_pull_widths) > 100.0
        ):
            axes[1, 1].set_yscale("log")
            axes[1, 1].set_ylabel("pull width (log scale)")
        for axis in axes.ravel():
            axis.axvline(
                result.recommended_iterations,
                color="black",
                linestyle="--",
                alpha=0.5,
            )
            axis.set_xticks(iteration_values)
            axis.set_xlabel("iterations")
            axis.grid(alpha=0.25)
        axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[1, 1].axhline(1.0, color="black", linewidth=0.8)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"{label}\niteration scan; dashed line is minimum median held-out "
            "normalized MSE"
        )
        pdf.savefig(fig)
        plt.close(fig)

        for stress_index, stress in enumerate(stress_names):
            target = validation_truth[stress_index]
            unfolded = result.unfolded[stress_index, recommended_index]
            sigma = result.uncertainty[stress_index, recommended_index]
            valid = result.validity[stress_index, recommended_index]
            delta = unfolded - target
            fractional = np.divide(
                delta, target, out=np.full_like(delta, np.nan), where=target > 0
            )
            pulls = np.divide(
                delta, sigma, out=np.full_like(delta, np.nan), where=sigma > 0
            )
            all_fractional = fractional[valid]
            all_pulls = pulls[valid]
            target_sum = np.sum(np.where(valid, target, 0.0), axis=0)
            unfolded_sum = np.sum(np.where(valid, unfolded, 0.0), axis=0)
            target_4d = binning.unflatten(target_sum)
            unfolded_4d = binning.unflatten(unfolded_sum)
            target_qx = np.sum(target_4d, axis=(2, 3))
            unfolded_qx = np.sum(unfolded_4d, axis=(2, 3))
            bias_qx = np.divide(
                unfolded_qx - target_qx,
                target_qx,
                out=np.full_like(target_qx, np.nan),
                where=target_qx > 0,
            )
            fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
            fig.subplots_adjust(
                left=0.085,
                right=0.955,
                bottom=0.085,
                top=0.91,
                wspace=0.30,
                hspace=0.30,
            )
            axes[0, 0].hist(
                all_fractional[np.isfinite(all_fractional)],
                bins=80,
                range=(-1, 1),
                histtype="stepfilled",
                alpha=0.7,
            )
            axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
            axes[0, 0].set_xlabel("(unfolded - held-out truth) / truth")
            axes[0, 0].set_ylabel("bins across folds")
            axes[0, 1].hist(
                all_pulls[np.isfinite(all_pulls)],
                bins=80,
                range=(-5, 5),
                histtype="stepfilled",
                alpha=0.7,
            )
            axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
            axes[0, 1].set_xlabel("closure pull")
            image = axes[1, 0].imshow(
                bias_qx,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-0.25,
                vmax=0.25,
            )
            axes[1, 0].set_xlabel("xB bin")
            axes[1, 0].set_ylabel("Q2 bin")
            axes[1, 0].set_title("relative bias summed over -t and phi")
            fig.colorbar(image, ax=axes[1, 0], label="relative bias")
            text = next(
                row for row in aggregates
                if row["stress"] == stress
                and int(row["iterations"]) == result.recommended_iterations
            )
            axes[1, 1].axis("off")
            axes[1, 1].text(
                0.0,
                1.0,
                "\n".join(
                    [
                        f"stress: {stress}",
                        f"iterations: {result.recommended_iterations}",
                        f"mean valid bins/fold: {text['valid_bins']:.1f}",
                        f"global relative bias: {text['global_relative_bias']:.4g}",
                        f"normalized MSE: {text['normalized_mse']:.4g}",
                        f"pull mean / width: {text['pull_mean']:.3f} / {text['pull_std']:.3f}",
                        "1-sigma / 2-sigma coverage: "
                        f"{text['coverage_1sigma']:.3f} / "
                        f"{text['coverage_2sigma']:.3f}",
                        f"refold chi2/ndf: {text['refold_chi2_ndf']:.3f}",
                        f"harmonic cells/fold: {text['harmonic_common_cells']:.1f}",
                        "harmonic A pull mean / width: "
                        f"{text['harmonic_A_pull_mean']:.3f} / "
                        f"{text['harmonic_A_pull_std']:.3f}",
                        "harmonic B pull mean / width: "
                        f"{text['harmonic_B_pull_mean']:.3f} / "
                        f"{text['harmonic_B_pull_std']:.3f}",
                        "harmonic C pull mean / width: "
                        f"{text['harmonic_C_pull_mean']:.3f} / "
                        f"{text['harmonic_C_pull_std']:.3f}",
                    ]
                ),
                va="top",
                family="monospace",
                fontsize=8.5,
                linespacing=1.25,
            )
            fig.suptitle(f"{label}: {stress} truth-shape closure")
            pdf.savefig(fig)
            plt.close(fig)


def render_saved_diagnostics(
    output_dir: Path,
    *,
    output: Path | None = None,
    label: str | None = None,
) -> Path:
    """Rebuild closure diagnostics from saved numerical artifacts without rescanning ROOT."""
    results_path = output_dir / "closure_results.npz"
    metrics_path = output_dir / "closure_metrics.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing saved closure results: {results_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing saved closure metrics: {metrics_path}")

    with metrics_path.open(newline="", encoding="utf-8") as source:
        metrics = tuple(_parse_metric_row(row) for row in csv.DictReader(source))
    if not metrics:
        raise ValueError(f"saved closure metrics are empty: {metrics_path}")

    with np.load(results_path, allow_pickle=False) as saved:
        summary = json.loads(saved["summary_json"].item())
        result = ClosureScanResult(
            unfolded=np.asarray(saved["unfolded"]),
            uncertainty=np.asarray(saved["uncertainty"]),
            validity=np.asarray(saved["validity"], dtype=bool),
            refolded=np.asarray(saved["refolded"]),
            training_efficiency=np.asarray(saved["training_efficiency"]),
            metrics=metrics,
            recommended_iterations=int(saved["recommended_iterations"]),
        )
        validation_truth = np.asarray(saved["validation_truth"])
        stress_names = tuple(str(value) for value in saved["stress_names"].tolist())
        binning = AnalysisBinning(
            saved["q2_edges"],
            saved["xb_edges"],
            saved["t_edges"],
            saved["phi_edges"],
        )

    destination = output or output_dir / "closure_diagnostics.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_diagnostics(
        destination,
        result,
        validation_truth,
        stress_names,
        binning,
        label=label or str(summary.get("label", "GEMC split-sample response closure")),
    )
    return destination


def _parse_metric_row(row: dict[str, str]) -> dict[str, float | int | str]:
    parsed: dict[str, float | int | str] = {}
    for key, value in row.items():
        if key == "stress":
            parsed[key] = value
        elif key in {"fold", "iterations"}:
            parsed[key] = int(value)
        else:
            parsed[key] = float(value)
    return parsed


def _concat(items: list[np.ndarray], *, dtype) -> np.ndarray:
    if not items:
        return np.empty(0, dtype=dtype)
    return np.concatenate(items).astype(dtype, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
