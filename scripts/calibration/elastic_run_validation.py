from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .elastic_momentum import (
    RAD_TO_DEG,
    ElasticFitConfig,
    derive_corrections,
    elastic_electron_momentum,
    elastic_proton_momentum,
    evaluate_region,
    load_elastic_arrays,
    mode_seeded_core,
    plot_diagnostics,
    region_support_mask,
    sector_local_phi,
    select_elastic_events,
    wrap_degrees,
)
from .plot_utils import save_plot


MODEL_ORDERS = {
    "constant": (0, 0, 0),
    "theta-linear": (1, 0, 0),
    "theta-phi": (1, 1, 1),
    "theta2-phi": (2, 1, 1),
}

MODEL_COMPLEXITY = {
    "constant": 1,
    "theta-linear": 2,
    "theta-phi": 4,
    "theta2-phi": 6,
}


@dataclass(frozen=True)
class RunBlock:
    index: int
    runs: tuple[int, ...]
    selected_candidates: int
    run_class: str | None = None

    @property
    def run_min(self) -> int:
        return min(self.runs)

    @property
    def run_max(self) -> int:
        return max(self.runs)

    @property
    def run_center(self) -> float:
        return float(np.mean(self.runs))

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "index": self.index,
            "fold": "A" if self.index % 2 == 0 else "B",
            "runs": list(self.runs),
            "runMin": self.run_min,
            "runMax": self.run_max,
            "runCenter": self.run_center,
            "selectedCandidates": self.selected_candidates,
        }
        if self.run_class is not None:
            result["runClass"] = self.run_class
        return result


def concatenate_arrays(parts: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    materialized = list(parts)
    if not materialized:
        raise ValueError("at least one array collection is required")
    common = set(materialized[0])
    for arrays in materialized[1:]:
        common &= set(arrays)
    if "runNum" not in common:
        raise ValueError("every candidate input file must have a runNum branch")
    return {
        name: np.concatenate([np.asarray(arrays[name]) for arrays in materialized])
        for name in sorted(common)
    }


def subset_arrays(
    arrays: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[mask] for name, values in arrays.items()}


def load_candidate_inputs(
    input_files: Iterable[Path], tree: str,
) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for path in input_files:
        print(f"[LOAD] {path}", flush=True)
        part = load_elastic_arrays(path, tree, None)
        entries = next(iter(part.values())).size if part else 0
        print(f"[LOAD] {path}: {entries} candidate rows", flush=True)
        parts.append(part)
    arrays = concatenate_arrays(parts)
    entries = next(iter(arrays.values())).size
    print(f"[LOAD] combined candidate rows: {entries}", flush=True)
    return arrays


def load_run_catalog(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text())
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"run catalog has no top-level runs object: {path}")
    result: dict[int, str] = {}
    for run_text, entry in runs.items():
        if not isinstance(entry, dict) or not entry.get("run_class"):
            raise ValueError(f"run catalog entry {run_text} has no run_class")
        result[int(run_text)] = str(entry["run_class"])
    if not result:
        raise ValueError(f"run catalog contains no runs: {path}")
    return result


def filter_arrays_by_run_classes(
    arrays: dict[str, np.ndarray],
    run_class_by_run: dict[int, str],
    include_run_classes: Iterable[str] | None,
    *,
    catalog_path: Path,
) -> tuple[dict[str, np.ndarray], dict[int, str], dict[str, object]]:
    """Consume and filter an array collection while limiting peak memory."""
    if "runNum" not in arrays:
        raise ValueError("candidate inputs have no runNum branch")
    run_numbers = np.asarray(arrays["runNum"], dtype=int)
    input_runs = sorted(int(run) for run in np.unique(run_numbers))
    missing = [run for run in input_runs if run not in run_class_by_run]
    if missing:
        preview = ", ".join(str(run) for run in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            f"candidate runs are absent from run catalog: {preview}{suffix}"
        )
    requested_classes = tuple(dict.fromkeys(
        str(value) for value in (include_run_classes or ())
    ))
    allowed_classes = (
        set(requested_classes)
        if requested_classes else set(run_class_by_run.values())
    )
    unknown_classes = sorted(allowed_classes - set(run_class_by_run.values()))
    if unknown_classes:
        raise ValueError(
            "run catalog contains no requested classes: "
            + ", ".join(unknown_classes)
        )
    allowed_runs = np.asarray([
        run for run in input_runs if run_class_by_run[run] in allowed_classes
    ], dtype=int)
    mask = np.isin(run_numbers, allowed_runs)
    if not np.any(mask):
        raise ValueError("run-class selection retained no candidate rows")
    filtered = arrays
    for name in tuple(filtered):
        filtered[name] = np.asarray(filtered[name])[mask]
    selected_runs = sorted(int(run) for run in np.unique(filtered["runNum"]))
    selected_mapping = {run: run_class_by_run[run] for run in selected_runs}
    class_summary = []
    for run_class in sorted(allowed_classes):
        class_runs = [
            run for run in selected_runs if selected_mapping[run] == run_class
        ]
        if not class_runs:
            continue
        class_entries = int(np.count_nonzero(np.isin(run_numbers, class_runs)))
        class_summary.append({
            "runClass": run_class,
            "runs": class_runs,
            "runCount": len(class_runs),
            "candidateRows": class_entries,
        })
    metadata: dict[str, object] = {
        "catalog": str(catalog_path),
        "includedRunClasses": (
            list(requested_classes) if requested_classes
            else sorted(allowed_classes)
        ),
        "inputCandidateRows": int(run_numbers.size),
        "selectedCandidateRows": int(np.count_nonzero(mask)),
        "excludedCandidateRows": int(np.count_nonzero(~mask)),
        "inputRuns": input_runs,
        "selectedRuns": selected_runs,
        "classes": class_summary,
    }
    print(
        "[RUN FILTER] retained "
        f"{metadata['selectedCandidateRows']}/{metadata['inputCandidateRows']} rows "
        f"from {len(selected_runs)} runs in classes "
        + ", ".join(str(value) for value in metadata["includedRunClasses"]),
        flush=True,
    )
    return filtered, selected_mapping, metadata


def _group_run_segment(
    runs: np.ndarray,
    counts: np.ndarray,
    target_selected_candidates: int,
    run_class: str | None,
) -> list[tuple[tuple[int, ...], int, str | None]]:
    grouped: list[tuple[tuple[int, ...], int, str | None]] = []
    current_runs: list[int] = []
    current_count = 0
    for run, count in zip(runs, counts):
        current_runs.append(int(run))
        current_count += int(count)
        if current_count >= target_selected_candidates:
            grouped.append((tuple(current_runs), current_count, run_class))
            current_runs = []
            current_count = 0
    if current_runs:
        if grouped and current_count < 0.5 * target_selected_candidates:
            previous_runs, previous_count, previous_class = grouped[-1]
            grouped[-1] = (
                previous_runs + tuple(current_runs),
                previous_count + current_count,
                previous_class,
            )
        else:
            grouped.append((tuple(current_runs), current_count, run_class))
    return grouped


def make_run_blocks(
    run_numbers: np.ndarray,
    target_selected_candidates: int,
    run_class_by_run: dict[int, str] | None = None,
) -> list[RunBlock]:
    if target_selected_candidates < 1:
        raise ValueError("run-block target must be positive")
    runs, counts = np.unique(np.asarray(run_numbers, dtype=int), return_counts=True)
    if runs.size < 2:
        raise ValueError("multi-run validation requires at least two runs")
    grouped: list[tuple[tuple[int, ...], int, str | None]] = []
    if run_class_by_run is None:
        grouped.extend(_group_run_segment(
            runs, counts, target_selected_candidates, None
        ))
    else:
        missing = [int(run) for run in runs if int(run) not in run_class_by_run]
        if missing:
            raise ValueError(
                "selected runs have no run-class assignment: "
                + ", ".join(str(run) for run in missing[:8])
            )
        segment_start = 0
        while segment_start < runs.size:
            run_class = run_class_by_run[int(runs[segment_start])]
            segment_stop = segment_start + 1
            while (
                segment_stop < runs.size
                and run_class_by_run[int(runs[segment_stop])] == run_class
            ):
                segment_stop += 1
            grouped.extend(_group_run_segment(
                runs[segment_start:segment_stop],
                counts[segment_start:segment_stop],
                target_selected_candidates,
                run_class,
            ))
            segment_start = segment_stop
    if len(grouped) < 2:
        midpoint = max(1, runs.size // 2)
        run_class = (
            run_class_by_run[int(runs[0])] if run_class_by_run is not None else None
        )
        grouped = [
            (
                tuple(int(value) for value in runs[:midpoint]),
                int(np.sum(counts[:midpoint])),
                run_class,
            ),
            (
                tuple(int(value) for value in runs[midpoint:]),
                int(np.sum(counts[midpoint:])),
                run_class,
            ),
        ]
    return [
        RunBlock(index, block_runs, count, run_class)
        for index, (block_runs, count, run_class) in enumerate(grouped)
    ]


def make_run_class_fold_blocks(
    run_numbers: np.ndarray,
    run_class_by_run: dict[int, str],
    run_class_order: Iterable[str],
) -> list[RunBlock]:
    classes = tuple(dict.fromkeys(str(value) for value in run_class_order))
    if len(classes) != 2:
        raise ValueError("class-fold validation requires exactly two run classes")
    run_array = np.asarray(run_numbers, dtype=int)
    selected_runs = sorted(int(run) for run in np.unique(run_array))
    missing = [run for run in selected_runs if run not in run_class_by_run]
    if missing:
        raise ValueError("class-fold validation found runs without a class")
    observed_classes = {run_class_by_run[run] for run in selected_runs}
    if observed_classes != set(classes):
        raise ValueError(
            "class-fold validation requires selected runs from exactly: "
            + ", ".join(classes)
        )
    blocks: list[RunBlock] = []
    for index, run_class in enumerate(classes):
        class_runs = tuple(
            run for run in selected_runs if run_class_by_run[run] == run_class
        )
        count = int(np.count_nonzero(np.isin(run_array, class_runs)))
        blocks.append(RunBlock(index, class_runs, count, run_class))
    return blocks


def make_fixed_run_blocks(
    run_numbers: np.ndarray,
    run_groups: Iterable[Iterable[int]],
    run_class_by_run: dict[int, str] | None = None,
) -> list[RunBlock]:
    """Reuse an existing run partition while updating selected-event counts."""
    groups = [tuple(int(run) for run in group) for group in run_groups]
    if len(groups) < 2 or any(not group for group in groups):
        raise ValueError("fixed run partition requires at least two nonempty blocks")
    flattened = [run for group in groups for run in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("fixed run partition contains a run in more than one block")
    selected_runs = set(int(run) for run in np.unique(run_numbers))
    uncovered = sorted(selected_runs - set(flattened))
    if uncovered:
        preview = ", ".join(str(run) for run in uncovered[:8])
        suffix = "..." if len(uncovered) > 8 else ""
        raise ValueError(
            f"fixed run partition does not cover selected runs: {preview}{suffix}"
        )
    run_array = np.asarray(run_numbers, dtype=int)
    blocks: list[RunBlock] = []
    for index, group in enumerate(groups):
        run_class = None
        if run_class_by_run is not None:
            classes = {run_class_by_run.get(run) for run in group}
            if None in classes:
                raise ValueError("fixed run partition contains an unclassified run")
            if len(classes) != 1:
                raise ValueError("fixed run block crosses a run-class boundary")
            run_class = str(next(iter(classes)))
        blocks.append(RunBlock(
            index=index,
            runs=group,
            selected_candidates=int(np.count_nonzero(np.isin(run_array, group))),
            run_class=run_class,
        ))
    if any(block.selected_candidates == 0 for block in blocks):
        raise ValueError("fixed run partition produced an empty selected-event block")
    return blocks


def _model_config(cfg: ElasticFitConfig, model: str) -> ElasticFitConfig:
    theta_order, phi_order, harmonics = MODEL_ORDERS[model]
    return replace(
        cfg,
        theta_order=theta_order,
        phi_order=phi_order,
        cd_fourier_harmonics=harmonics,
    )


def _particle_view(
    selected: dict[str, np.ndarray], particle: str, beam_energy: float
) -> dict[str, np.ndarray]:
    if "runNum" not in selected:
        raise ValueError("candidate tree has no runNum branch")
    if particle == "electron":
        expected = elastic_electron_momentum(selected["electronTheta"], beam_energy)
        return {
            "run": np.asarray(selected["runNum"], dtype=int),
            "detector": np.asarray(selected["electronDet"], dtype=int),
            "sector": np.asarray(selected["electronSector"], dtype=int),
            "thetaDeg": np.asarray(selected["electronTheta"], dtype=float) * RAD_TO_DEG,
            "phiRad": np.asarray(selected["electronPhi"], dtype=float),
            "residual": expected / np.asarray(selected["electronP"], dtype=float) - 1.0,
        }
    expected = elastic_proton_momentum(selected["protonTheta"], beam_energy)
    return {
        "run": np.asarray(selected["runNum"], dtype=int),
        "detector": np.asarray(selected["protonDet"], dtype=int),
        "sector": np.asarray(selected["protonSector"], dtype=int),
        "thetaDeg": np.asarray(selected["protonTheta"], dtype=float) * RAD_TO_DEG,
        "phiRad": np.asarray(selected["protonPhi"], dtype=float),
        "residual": expected / np.asarray(selected["protonP"], dtype=float) - 1.0,
    }


def _region_coordinates(
    view: dict[str, np.ndarray], region: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector = int(region["detector"])
    sector = int(region["sector"])
    mask = np.asarray(view["detector"]) == detector
    if sector:
        mask &= np.asarray(view["sector"]) == sector
    if detector == 1:
        phi = sector_local_phi(
            np.asarray(view["phiRad"])[mask], np.asarray(view["sector"])[mask]
        )
    else:
        phi = wrap_degrees(np.asarray(view["phiRad"])[mask] * RAD_TO_DEG)
    return mask, np.asarray(view["thetaDeg"])[mask], phi


def _core_summary(values: np.ndarray, cfg: ElasticFitConfig) -> dict[str, object] | None:
    estimate = mode_seeded_core(
        values,
        peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
        peak_seed_half_width=cfg.peak_seed_half_width,
        sigma_clip=cfg.core_sigma_clip,
    )
    if (
        estimate.entries < cfg.min_bin_entries
        or not np.isfinite(estimate.center)
        or not np.isfinite(estimate.error)
        or not np.isfinite(estimate.width)
    ):
        return None
    return {
        "center": estimate.center,
        "centerError": estimate.error,
        "coreWidth": estimate.width,
        "coreEntries": estimate.entries,
        "rawEntries": estimate.total_entries,
        "coreFraction": estimate.retained_fraction,
    }


def _cell_mask(
    theta: np.ndarray, phi: np.ndarray, cell: dict[str, object]
) -> np.ndarray:
    theta_lo, theta_hi = (float(value) for value in cell["thetaRangeDeg"])
    phi_lo, phi_hi = (float(value) for value in cell["phiRangeDeg"])
    theta_upper = (
        theta <= theta_hi if cell.get("thetaHighInclusive", False)
        else theta < theta_hi
    )
    phi_upper = (
        phi <= phi_hi if cell.get("phiHighInclusive", False)
        else phi < phi_hi
    )
    return (
        (theta >= theta_lo) & theta_upper
        & (phi >= phi_lo) & phi_upper
    )


def validate_correction(
    correction: dict[str, object],
    selected: dict[str, np.ndarray],
    blocks: list[RunBlock],
    holdout_fold: int,
    cfg: ElasticFitConfig,
    particle: str,
) -> list[dict[str, object]]:
    view = _particle_view(selected, particle, cfg.beam_energy)
    results: list[dict[str, object]] = []
    for region in correction["regions"]:
        region_mask, theta, phi = _region_coordinates(view, region)
        region_runs = np.asarray(view["run"])[region_mask]
        region_residual = np.asarray(view["residual"])[region_mask]
        support = region_support_mask(region, theta, phi)
        fitted = np.zeros(theta.shape, dtype=float)
        fitted[support] = evaluate_region(region, theta[support], phi[support])
        corrected = (1.0 + region_residual) / (1.0 + fitted) - 1.0
        for block in blocks:
            if block.index % 2 != holdout_fold:
                continue
            in_block = np.isin(region_runs, block.runs) & support
            before = _core_summary(region_residual[in_block], cfg)
            after = _core_summary(corrected[in_block], cfg)
            cell_before: list[float] = []
            cell_after: list[float] = []
            for cell in region["fit"]["acceptedProfileCells"]:
                in_cell = in_block & _cell_mask(theta, phi, cell)
                before_cell = _core_summary(region_residual[in_cell], cfg)
                after_cell = _core_summary(corrected[in_cell], cfg)
                if before_cell is None or after_cell is None:
                    continue
                cell_before.append(float(before_cell["center"]))
                cell_after.append(float(after_cell["center"]))
            results.append({
                "block": block.index,
                "fold": "A" if block.index % 2 == 0 else "B",
                "runMin": block.run_min,
                "runMax": block.run_max,
                "runCenter": block.run_center,
                "pid": int(region["pid"]),
                "detector": int(region["detector"]),
                "sector": int(region["sector"]),
                "supportEntries": int(np.count_nonzero(in_block)),
                "before": before,
                "after": after,
                "validatedCells": len(cell_after),
                "cellCenterRmsBefore": (
                    float(np.sqrt(np.mean(np.square(cell_before))))
                    if cell_before else None
                ),
                "cellCenterRmsAfter": (
                    float(np.sqrt(np.mean(np.square(cell_after))))
                    if cell_after else None
                ),
                "maxAbsCellCenterAfter": (
                    float(np.max(np.abs(cell_after))) if cell_after else None
                ),
            })
    return results


def common_cell_stability(
    correction: dict[str, object],
    selected: dict[str, np.ndarray],
    blocks: list[RunBlock],
    cfg: ElasticFitConfig,
    particle: str,
) -> list[dict[str, object]]:
    view = _particle_view(selected, particle, cfg.beam_energy)
    results: list[dict[str, object]] = []
    for region in correction["regions"]:
        region_mask, theta, phi = _region_coordinates(view, region)
        region_runs = np.asarray(view["run"])[region_mask]
        residual = np.asarray(view["residual"])[region_mask]
        for cell_index, cell in enumerate(region["fit"]["acceptedProfileCells"]):
            pooled_center = float(cell["center"])
            geometric_cell = _cell_mask(theta, phi, cell)
            for block in blocks:
                in_cell = geometric_cell & np.isin(region_runs, block.runs)
                estimate = _core_summary(residual[in_cell], cfg)
                if estimate is None:
                    continue
                results.append({
                    "block": block.index,
                    "fold": "A" if block.index % 2 == 0 else "B",
                    "runCenter": block.run_center,
                    "pid": int(region["pid"]),
                    "detector": int(region["detector"]),
                    "sector": int(region["sector"]),
                    "cell": cell_index,
                    "thetaMeanDeg": float(cell["thetaMeanDeg"]),
                    "phiMeanDeg": float(cell["phiMeanDeg"]),
                    "pooledCenter": pooled_center,
                    "blockCenter": float(estimate["center"]),
                    "blockCenterError": float(estimate["centerError"]),
                    "deltaFromPooledCenter": float(estimate["center"]) - pooled_center,
                    "coreEntries": int(estimate["coreEntries"]),
                })
    return results


def per_run_region_stability(
    selected: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
    particle: str,
    minimum_core_entries: int,
) -> list[dict[str, object]]:
    view = _particle_view(selected, particle, cfg.beam_energy)
    results: list[dict[str, object]] = []
    detectors = (1,) if particle == "electron" else (1, 2)
    for detector in detectors:
        sectors = range(1, 7) if detector == 1 else (0,)
        for sector in sectors:
            region = np.asarray(view["detector"]) == detector
            if sector:
                region &= np.asarray(view["sector"]) == sector
            theta = np.asarray(view["thetaDeg"])
            if cfg.theta_min_deg is not None:
                region &= theta >= cfg.theta_min_deg
            if cfg.theta_max_deg is not None:
                region &= theta <= cfg.theta_max_deg
            for run in np.unique(np.asarray(view["run"])[region]):
                sample = region & (np.asarray(view["run"]) == run)
                estimate = mode_seeded_core(
                    np.asarray(view["residual"])[sample],
                    peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
                    peak_seed_half_width=cfg.peak_seed_half_width,
                    sigma_clip=cfg.core_sigma_clip,
                )
                if (
                    estimate.entries < minimum_core_entries
                    or not np.isfinite(estimate.center)
                    or not np.isfinite(estimate.error)
                    or not np.isfinite(estimate.width)
                ):
                    continue
                results.append({
                    "run": int(run),
                    "pid": 11 if particle == "electron" else 2212,
                    "detector": detector,
                    "sector": sector,
                    "center": estimate.center,
                    "centerError": estimate.error,
                    "coreWidth": estimate.width,
                    "coreEntries": estimate.entries,
                    "rawEntries": estimate.total_entries,
                })
    return results


def _aggregate_validation(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [
        row for row in rows
        if row["cellCenterRmsAfter"] is not None and row["after"] is not None
    ]
    return {
        "regionBlocks": len(rows),
        "validatedRegionBlocks": len(valid),
        "validatedCells": int(sum(int(row["validatedCells"]) for row in valid)),
        "medianCellCenterRmsBefore": (
            float(np.median([row["cellCenterRmsBefore"] for row in valid]))
            if valid else None
        ),
        "medianCellCenterRmsAfter": (
            float(np.median([row["cellCenterRmsAfter"] for row in valid]))
            if valid else None
        ),
        "medianAbsCoreCenterAfter": (
            float(np.median([abs(float(row["after"]["center"])) for row in valid]))
            if valid else None
        ),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def _plot_validation(
    rows: list[dict[str, object]],
    output_path: Path,
    model: str,
    particle: str,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    regions = sorted({
        (int(row["detector"]), int(row["sector"])) for row in rows
    })
    if not regions:
        return
    columns = 3
    rows_count = int(np.ceil(len(regions) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(5.2 * columns, 3.6 * rows_count))
    flat_axes = list(np.atleast_1d(axes).flat)
    for axis, (detector, sector) in zip(flat_axes, regions):
        region_rows = sorted(
            [row for row in rows if row["detector"] == detector and row["sector"] == sector],
            key=lambda row: row["runCenter"],
        )
        for key, color, marker in (
            ("before", "0.55", "o"), ("after", "#d95f02", "s")
        ):
            usable = [row for row in region_rows if row[key] is not None]
            axis.errorbar(
                [row["runCenter"] for row in usable],
                [100.0 * row[key]["center"] for row in usable],
                yerr=[100.0 * row[key]["centerError"] for row in usable],
                fmt=marker + "-", color=color, label=key, markersize=4,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        label = f"det {detector}" + (f", sector {sector}" if sector else "")
        axis.set_title(label)
        axis.set_xlabel("run-block center")
        axis.set_ylabel("held-out core center [%]")
        axis.legend(fontsize="small")
    for axis in flat_axes[len(regions):]:
        axis.set_visible(False)
    save_plot(
        fig, output_path,
        f"{particle} held-out closure: {model}", dataset_tag, beam_energy,
    )
    plt.close(fig)


def _plot_per_run(
    rows: list[dict[str, object]],
    output_path: Path,
    particle: str,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    regions = sorted({
        (int(row["detector"]), int(row["sector"])) for row in rows
    })
    if not regions:
        return
    columns = 3
    rows_count = int(np.ceil(len(regions) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(5.2 * columns, 3.6 * rows_count))
    flat_axes = list(np.atleast_1d(axes).flat)
    for axis, (detector, sector) in zip(flat_axes, regions):
        points = sorted(
            [row for row in rows if row["detector"] == detector and row["sector"] == sector],
            key=lambda row: row["run"],
        )
        axis.errorbar(
            [row["run"] for row in points],
            [100.0 * row["center"] for row in points],
            yerr=[100.0 * row["centerError"] for row in points],
            fmt="o", color="#21618c", markersize=3, alpha=0.7,
        )
        axis.set_title(f"det {detector}" + (f", sector {sector}" if sector else ""))
        axis.set_xlabel("run number")
        axis.set_ylabel("raw elastic core center [%]")
    for axis in flat_axes[len(regions):]:
        axis.set_visible(False)
    save_plot(
        fig, output_path, f"{particle} run-by-run raw residual centers",
        dataset_tag, beam_energy,
    )
    plt.close(fig)


def _plot_common_cell_heatmaps(
    rows: list[dict[str, object]],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    regions = sorted({
        (int(row["detector"]), int(row["sector"])) for row in rows
    })
    for detector, sector in regions:
        region_rows = [
            row for row in rows
            if row["detector"] == detector and row["sector"] == sector
        ]
        blocks = sorted({int(row["block"]) for row in region_rows})
        cells = sorted({int(row["cell"]) for row in region_rows})
        matrix = np.full((len(blocks), len(cells)), np.nan)
        block_index = {value: index for index, value in enumerate(blocks)}
        cell_index = {value: index for index, value in enumerate(cells)}
        for row in region_rows:
            matrix[block_index[int(row["block"])], cell_index[int(row["cell"])]] = (
                100.0 * float(row["deltaFromPooledCenter"])
            )
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = max(0.1, float(np.quantile(finite, 0.98))) if finite.size else 1.0
        fig, axis = plt.subplots(figsize=(max(8.0, 0.22 * len(cells)), 4.8))
        image = axis.imshow(
            matrix, aspect="auto", interpolation="none", cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
        axis.set_xlabel("common pooled cell index")
        axis.set_ylabel("contiguous run block")
        axis.set_yticks(range(len(blocks)), labels=blocks)
        fig.colorbar(image, ax=axis, label="block center - pooled center [%]")
        suffix = f"det{detector}" + (f"_sector{sector}" if sector else "")
        save_plot(
            fig, output_dir / f"common_cell_stability_{suffix}.png",
            f"Common-cell stability: {suffix}", dataset_tag, beam_energy,
        )
        plt.close(fig)


def _region_key(region: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(region["pid"]),
        int(region["detector"]),
        int(region["sector"]),
    )


def _region_label(key: tuple[int, int, int]) -> str:
    pid, detector, sector = key
    suffix = f"sector{sector}" if sector else "all_phi"
    return f"pid{pid}_det{detector}_{suffix}"


def _correction_region_map(
    correction: dict[str, object],
) -> dict[tuple[int, int, int], dict[str, object]]:
    return {
        _region_key(region): region
        for region in correction.get("regions", [])
    }


def _fold_surface_agreement(
    pooled: dict[str, object],
    fold_corrections: dict[int, dict[str, object]],
) -> dict[tuple[int, int, int], dict[str, object]]:
    if set(fold_corrections) != {0, 1}:
        return {}
    fold_a = _correction_region_map(fold_corrections[0])
    fold_b = _correction_region_map(fold_corrections[1])
    results: dict[tuple[int, int, int], dict[str, object]] = {}
    for pooled_region in pooled.get("regions", []):
        key = _region_key(pooled_region)
        if key not in fold_a or key not in fold_b:
            continue
        cells = pooled_region["fit"]["acceptedProfileCells"]
        if not cells:
            continue
        theta = np.asarray([cell["thetaMeanDeg"] for cell in cells], dtype=float)
        phi = np.asarray([cell["phiMeanDeg"] for cell in cells], dtype=float)
        common_support = (
            region_support_mask(fold_a[key], theta, phi)
            & region_support_mask(fold_b[key], theta, phi)
        )
        if not np.any(common_support):
            continue
        difference = (
            evaluate_region(fold_a[key], theta[common_support], phi[common_support])
            - evaluate_region(fold_b[key], theta[common_support], phi[common_support])
        )
        results[key] = {
            "commonPooledCellCenters": int(np.count_nonzero(common_support)),
            "rms": float(np.sqrt(np.mean(np.square(difference)))),
            "medianAbs": float(np.median(np.abs(difference))),
            "maxAbs": float(np.max(np.abs(difference))),
        }
    return results


def _region_validation_summaries(
    rows: list[dict[str, object]],
    blocks: list[RunBlock],
    surface_agreement: dict[tuple[int, int, int], dict[str, object]],
    expected_region_keys: Iterable[tuple[int, int, int]] = (),
) -> dict[tuple[int, int, int], dict[str, object]]:
    expected_blocks = {block.index for block in blocks}
    keys = sorted(
        {_region_key(row) for row in rows}
        | set(surface_agreement)
        | set(expected_region_keys)
    )
    results: dict[tuple[int, int, int], dict[str, object]] = {}
    for key in keys:
        region_rows = [row for row in rows if _region_key(row) == key]
        summary = _aggregate_validation(region_rows)
        fold_summaries = {
            fold: _aggregate_validation([
                row for row in region_rows if row["fold"] == fold
            ])
            for fold in ("A", "B")
        }
        valid_blocks = {
            int(row["block"])
            for row in region_rows
            if row["cellCenterRmsAfter"] is not None and row["after"] is not None
        }
        agreement = surface_agreement.get(key)
        after_rms = summary["medianCellCenterRmsAfter"]
        agreement_ratio = None
        if agreement is not None and after_rms is not None and float(after_rms) > 0.0:
            agreement_ratio = float(agreement["rms"]) / float(after_rms)
        pid, detector, sector = key
        summary.update({
            "pid": pid,
            "detector": detector,
            "sector": sector,
            "region": _region_label(key),
            "completeHeldOutBlockCoverage": valid_blocks == expected_blocks,
            "heldoutFolds": fold_summaries,
            "foldSurfaceAgreement": agreement,
            "foldSurfaceAgreementToHeldOutRmsRatio": agreement_ratio,
        })
        results[key] = summary
    return results


def _eligible_region_model(summary: dict[str, object]) -> bool:
    score = summary.get("medianCellCenterRmsAfter")
    agreement = summary.get("foldSurfaceAgreement")
    if (
        score is None
        or not summary.get("completeHeldOutBlockCoverage", False)
        or agreement is None
    ):
        return False
    fold_summaries = summary.get("heldoutFolds", {})
    if any(
        fold_summaries.get(fold, {}).get("medianCellCenterRmsAfter") is None
        for fold in ("A", "B")
    ):
        return False
    return float(agreement["rms"]) <= float(score)


def _choose_region_models(
    region_summaries_by_model: dict[
        str, dict[tuple[int, int, int], dict[str, object]]
    ],
    minimum_relative_improvement: float,
) -> dict[str, object]:
    keys = sorted({
        key
        for summaries in region_summaries_by_model.values()
        for key in summaries
    })
    region_recommendations: list[dict[str, object]] = []
    for key in keys:
        eligible = sorted(
            [
                model for model, summaries in region_summaries_by_model.items()
                if key in summaries and _eligible_region_model(summaries[key])
            ],
            key=lambda model: MODEL_COMPLEXITY[model],
        )
        pid, detector, sector = key
        if not eligible:
            region_recommendations.append({
                "pid": pid,
                "detector": detector,
                "sector": sector,
                "region": _region_label(key),
                "model": None,
                "eligibleModels": [],
                "status": (
                    "no model completed every held-out block with fold-to-fold "
                    "surface RMS below its held-out cell RMS"
                ),
                "selectionTrace": [],
            })
            continue

        chosen = eligible[0]
        trace: list[dict[str, object]] = []
        for candidate in eligible[1:]:
            baseline_summary = region_summaries_by_model[chosen][key]
            candidate_summary = region_summaries_by_model[candidate][key]
            baseline_score = float(
                baseline_summary["medianCellCenterRmsAfter"]
            )
            candidate_score = float(
                candidate_summary["medianCellCenterRmsAfter"]
            )
            relative_improvement = (
                (baseline_score - candidate_score) / baseline_score
                if baseline_score > 0.0 else None
            )
            fold_improvements: dict[str, float | None] = {}
            improves_every_fold = True
            for fold in ("A", "B"):
                baseline_fold = float(
                    baseline_summary["heldoutFolds"][fold][
                        "medianCellCenterRmsAfter"
                    ]
                )
                candidate_fold = float(
                    candidate_summary["heldoutFolds"][fold][
                        "medianCellCenterRmsAfter"
                    ]
                )
                improvement = (
                    (baseline_fold - candidate_fold) / baseline_fold
                    if baseline_fold > 0.0 else None
                )
                fold_improvements[fold] = improvement
                improves_every_fold &= candidate_fold < baseline_fold
            passes_margin = (
                relative_improvement is not None
                and relative_improvement >= minimum_relative_improvement
            )
            selected = improves_every_fold and passes_margin
            trace.append({
                "baseline": chosen,
                "candidate": candidate,
                "baselineMedianHeldOutCellRms": baseline_score,
                "candidateMedianHeldOutCellRms": candidate_score,
                "relativeImprovement": relative_improvement,
                "relativeImprovementByHeldOutFold": fold_improvements,
                "improvesEveryHeldOutFold": improves_every_fold,
                "passesMinimumRelativeImprovement": passes_margin,
                "selected": selected,
            })
            if selected:
                chosen = candidate

        chosen_summary = region_summaries_by_model[chosen][key]
        region_recommendations.append({
            "pid": pid,
            "detector": detector,
            "sector": sector,
            "region": _region_label(key),
            "model": chosen,
            "eligibleModels": eligible,
            "status": "diagnostic recommendation pending systematic variations",
            "medianHeldOutCellRms": chosen_summary["medianCellCenterRmsAfter"],
            "heldoutFoldCellRms": {
                fold: chosen_summary["heldoutFolds"][fold][
                    "medianCellCenterRmsAfter"
                ]
                for fold in ("A", "B")
            },
            "foldSurfaceAgreement": chosen_summary["foldSurfaceAgreement"],
            "selectionTrace": trace,
        })

    selected_models = {
        entry["model"] for entry in region_recommendations
        if entry["model"] is not None
    }
    complete = bool(region_recommendations) and all(
        entry["model"] is not None for entry in region_recommendations
    )
    if not complete:
        overall_model = None
        status = "one or more detector regions have no validated model"
    elif len(selected_models) == 1:
        overall_model = next(iter(selected_models))
        status = "one independently selected model is recommended in every region"
    else:
        overall_model = "mixed"
        status = "model complexity is selected independently in each detector region"
    return {
        "model": overall_model,
        "status": status,
        "minimumRelativeImprovementFraction": minimum_relative_improvement,
        "requiresImprovementInEveryHeldOutFold": True,
        "requiresFoldSurfaceRmsBelowHeldOutCellRms": True,
        "systematicReviewRequired": True,
        "regions": region_recommendations,
    }


def _build_mixed_parameters(
    pooled_by_model: dict[str, dict[str, object]],
    recommendation: dict[str, object],
    dataset_tag: str,
    raw_runs: np.ndarray,
) -> dict[str, object] | None:
    assignments = recommendation.get("regions", [])
    if not assignments or any(entry.get("model") is None for entry in assignments):
        return None
    first_model = str(assignments[0]["model"])
    if first_model not in pooled_by_model:
        return None
    mixed = copy.deepcopy(pooled_by_model[first_model])
    mixed["datasetTag"] = f"{dataset_tag}_recommended_mixed"
    mixed["calibrationRole"] = "pooledMixedCandidatePendingSystematicReview"
    mixed["runCoverage"] = sorted(int(value) for value in np.unique(raw_runs))
    mixed["regions"] = []
    mixed["skippedRegions"] = []
    fit_configuration = copy.deepcopy(mixed.get("fitConfiguration", {}))
    for field in ("thetaOrder", "fdPhiOrder", "cdFourierHarmonics"):
        fit_configuration.pop(field, None)
    fit_configuration["regionSpecificModelOrders"] = True
    mixed["fitConfiguration"] = fit_configuration
    mixed["modelSelection"] = {
        "strategy": (
            "sector-specific nested held-out selection with improvement required "
            "in both folds"
        ),
        "minimumRelativeImprovementFraction": recommendation[
            "minimumRelativeImprovementFraction"
        ],
        "systematicReviewRequired": True,
        "assignments": [
            {
                "pid": entry["pid"],
                "detector": entry["detector"],
                "sector": entry["sector"],
                "model": entry["model"],
            }
            for entry in assignments
        ],
    }
    region_maps = {
        model: _correction_region_map(correction)
        for model, correction in pooled_by_model.items()
    }
    for entry in assignments:
        key = (
            int(entry["pid"]),
            int(entry["detector"]),
            int(entry["sector"]),
        )
        model = str(entry["model"])
        if model not in region_maps or key not in region_maps[model]:
            return None
        region = copy.deepcopy(region_maps[model][key])
        region["validationModel"] = model
        mixed["regions"].append(region)
    mixed["regions"].sort(
        key=lambda region: (
            int(region["pid"]), int(region["detector"]), int(region["sector"])
        )
    )
    return mixed


def _plot_region_model_comparison(
    region_summaries_by_model: dict[
        str, dict[tuple[int, int, int], dict[str, object]]
    ],
    recommendation: dict[str, object],
    output_path: Path,
    particle: str,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    keys = sorted({
        key
        for summaries in region_summaries_by_model.values()
        for key in summaries
    })
    if not keys:
        return
    assignments = {
        (int(entry["pid"]), int(entry["detector"]), int(entry["sector"])):
        entry.get("model")
        for entry in recommendation.get("regions", [])
    }
    models = sorted(region_summaries_by_model, key=lambda name: MODEL_COMPLEXITY[name])
    columns = 3
    rows_count = int(np.ceil(len(keys) / columns))
    fig, axes = plt.subplots(
        rows_count, columns, figsize=(5.2 * columns, 3.8 * rows_count)
    )
    flat_axes = list(np.atleast_1d(axes).flat)
    x_values = np.arange(len(models))
    for axis, key in zip(flat_axes, keys):
        overall: list[float] = []
        fold_a: list[float] = []
        fold_b: list[float] = []
        for model in models:
            summary = region_summaries_by_model.get(model, {}).get(key, {})
            overall.append(100.0 * float(summary["medianCellCenterRmsAfter"])
                           if summary.get("medianCellCenterRmsAfter") is not None
                           else np.nan)
            heldout = summary.get("heldoutFolds", {})
            fold_a.append(
                100.0 * float(heldout["A"]["medianCellCenterRmsAfter"])
                if heldout.get("A", {}).get("medianCellCenterRmsAfter") is not None
                else np.nan
            )
            fold_b.append(
                100.0 * float(heldout["B"]["medianCellCenterRmsAfter"])
                if heldout.get("B", {}).get("medianCellCenterRmsAfter") is not None
                else np.nan
            )
        axis.plot(x_values, overall, "o-", color="#21618c", label="combined")
        axis.plot(x_values, fold_a, "^--", color="0.45", label="held-out A")
        axis.plot(x_values, fold_b, "v--", color="#d95f02", label="held-out B")
        selected = assignments.get(key)
        if selected in models:
            index = models.index(selected)
            if np.isfinite(overall[index]):
                axis.plot(
                    index, overall[index], marker="*", color="black",
                    markersize=12, linestyle="none", label="recommended",
                )
        axis.set_xticks(x_values, labels=models, rotation=20, ha="right")
        axis.set_ylabel("held-out cell-center RMS [%]")
        axis.set_title(_region_label(key))
        axis.grid(axis="y", alpha=0.25)
    for axis in flat_axes[len(keys):]:
        axis.set_visible(False)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88),
        ncol=min(4, len(labels)),
    )
    fig.subplots_adjust(hspace=0.65)
    save_plot(
        fig, output_path,
        f"{particle} held-out model comparison by detector region",
        dataset_tag, beam_energy, tight_layout=False,
    )
    plt.close(fig)


def run_validation(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
    *,
    particle: str,
    models: list[str],
    block_target: int,
    output_dir: Path,
    dataset_tag: str,
    make_plots: bool = True,
    minimum_per_run_core_entries: int = 200,
    minimum_model_improvement_fraction: float = 0.10,
    run_groups: Iterable[Iterable[int]] | None = None,
    run_class_by_run: dict[int, str] | None = None,
    run_block_mode: str = "candidate-target",
    run_class_order: Iterable[str] = (),
    run_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    if not 0.0 <= minimum_model_improvement_fraction < 1.0:
        raise ValueError("minimum model improvement fraction must be in [0, 1)")
    shared_phi_cells = 2 if any(
        MODEL_ORDERS[model][1] > 0 or MODEL_ORDERS[model][2] > 0
        for model in models
    ) else 1
    cfg = replace(
        cfg,
        min_phi_cells_per_theta=max(
            cfg.min_phi_cells_per_theta, shared_phi_cells
        ),
    )
    selected, selection = select_elastic_events(arrays, cfg)
    if "runNum" not in selected:
        raise ValueError("candidate tree must contain runNum for multi-run validation")
    if particle == "electron":
        expected_region_keys = {(11, 1, sector) for sector in range(1, 7)}
    else:
        proton_detector = np.asarray(selected["protonDet"], dtype=int)
        proton_sector = np.asarray(selected["protonSector"], dtype=int)
        expected_region_keys = {
            (2212, 1, int(sector))
            for sector in np.unique(proton_sector[proton_detector == 1])
            if 1 <= int(sector) <= 6
        }
        if np.any(proton_detector == 2):
            expected_region_keys.add((2212, 2, 0))
    if run_block_mode not in {
        "candidate-target", "class-boundary", "class-fold",
    }:
        raise ValueError(f"unknown run-block mode: {run_block_mode}")
    if run_block_mode != "candidate-target" and run_class_by_run is None:
        raise ValueError(f"{run_block_mode} requires run-class assignments")
    block_class_mapping = (
        run_class_by_run if run_block_mode != "candidate-target" else None
    )
    if run_groups is not None:
        blocks = make_fixed_run_blocks(
            selected["runNum"], run_groups, block_class_mapping
        )
    elif run_block_mode == "class-fold":
        blocks = make_run_class_fold_blocks(
            selected["runNum"], run_class_by_run or {}, run_class_order
        )
    else:
        blocks = make_run_blocks(
            selected["runNum"], block_target, block_class_mapping
        )
    if len(blocks) < 2:
        raise ValueError("multi-run validation requires at least two run blocks")
    raw_runs = np.asarray(arrays["runNum"], dtype=int)
    folds = {
        0: tuple(run for block in blocks if block.index % 2 == 0 for run in block.runs),
        1: tuple(run for block in blocks if block.index % 2 == 1 for run in block.runs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "elastic_momentum_run_validation/v2",
        "datasetTag": dataset_tag,
        "particle": particle,
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "selection": selection,
        "runBlockDefinition": (
            "fixed-from-nominal-systematic-scan"
            if run_groups is not None else
            "selected-candidate-target"
            if run_block_mode == "candidate-target" else
            "selected-candidate-target-with-class-boundaries"
            if run_block_mode == "class-boundary" else
            "two-run-class-cross-validation"
        ),
        "runBlocks": [block.to_json() for block in blocks],
        "models": {},
    }
    if run_selection is not None:
        report["runSelection"] = copy.deepcopy(run_selection)
    per_run = per_run_region_stability(
        selected, cfg, particle, minimum_per_run_core_entries
    )
    report["perRunRegionStability"] = per_run
    if make_plots:
        _plot_per_run(
            per_run, output_dir / "per_run_raw_centers.png",
            particle, dataset_tag, cfg.beam_energy,
        )
    model_summaries: dict[str, dict[str, object]] = {}
    region_summaries_by_model: dict[
        str, dict[tuple[int, int, int], dict[str, object]]
    ] = {}
    pooled_by_model: dict[str, dict[str, object]] = {}
    for model in models:
        model_cfg = _model_config(cfg, model)
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            pooled, pooled_diagnostics = derive_corrections(
                arrays, model_cfg, particles=(particle,)
            )
        except ValueError as error:
            report["models"][model] = {
                "status": "pooled fit unavailable",
                "reason": str(error),
            }
            model_summaries[model] = {
                "completeFoldRegionCoverage": False,
                "medianCellCenterRmsAfter": None,
            }
            region_summaries_by_model[model] = _region_validation_summaries(
                [], blocks, {}, expected_region_keys
            )
            continue
        pooled["datasetTag"] = dataset_tag
        pooled["calibrationRole"] = "pooledCandidatePendingHeldOutReview"
        pooled["runCoverage"] = sorted(int(value) for value in np.unique(raw_runs))
        pooled["runBlockMode"] = run_block_mode
        if run_selection is not None:
            pooled["runSelection"] = copy.deepcopy(run_selection)
        pooled_by_model[model] = pooled
        _write_json(model_dir / "pooled_parameters.json", pooled)
        if make_plots:
            plot_diagnostics(
                pooled_diagnostics, model_dir / "pooled_fit_plots",
                f"{dataset_tag}_{model}_pooled", cfg.beam_energy,
            )
        validation_rows: list[dict[str, object]] = []
        fold_details: list[dict[str, object]] = []
        fold_corrections: dict[int, dict[str, object]] = {}
        for training_fold in (0, 1):
            training_mask = np.isin(raw_runs, folds[training_fold])
            holdout_fold = 1 - training_fold
            try:
                fold_correction, _ = derive_corrections(
                    subset_arrays(arrays, training_mask),
                    model_cfg,
                    particles=(particle,),
                )
            except ValueError as error:
                fold_details.append({
                    "trainingFold": "A" if training_fold == 0 else "B",
                    "holdoutFold": "A" if holdout_fold == 0 else "B",
                    "trainingRuns": list(folds[training_fold]),
                    "holdoutRuns": list(folds[holdout_fold]),
                    "fittedRegions": 0,
                    "skippedRegions": [{"reason": str(error)}],
                    "validation": _aggregate_validation([]),
                })
                continue
            fold_correction["datasetTag"] = (
                f"{dataset_tag}_{model}_train_{'A' if training_fold == 0 else 'B'}"
            )
            fold_correction["trainingRuns"] = list(folds[training_fold])
            fold_correction["holdoutRuns"] = list(folds[holdout_fold])
            fold_corrections[training_fold] = fold_correction
            _write_json(
                model_dir / f"train_{'A' if training_fold == 0 else 'B'}_parameters.json",
                fold_correction,
            )
            fold_rows = validate_correction(
                fold_correction, selected, blocks, holdout_fold,
                model_cfg, particle,
            )
            validation_rows.extend(fold_rows)
            fold_details.append({
                "trainingFold": "A" if training_fold == 0 else "B",
                "holdoutFold": "A" if holdout_fold == 0 else "B",
                "trainingRuns": list(folds[training_fold]),
                "holdoutRuns": list(folds[holdout_fold]),
                "fittedRegions": len(fold_correction["regions"]),
                "skippedRegions": fold_correction["skippedRegions"],
                "validation": _aggregate_validation(fold_rows),
            })
        aggregate = _aggregate_validation(validation_rows)
        expected_regions = 6 if particle == "electron" else None
        if expected_regions is not None:
            complete = len(fold_details) == 2 and all(
                detail["fittedRegions"] == expected_regions for detail in fold_details
            )
        else:
            complete = len(fold_details) == 2 and all(
                not detail["skippedRegions"] for detail in fold_details
            )
        aggregate["completeFoldRegionCoverage"] = complete
        model_summaries[model] = aggregate
        surface_agreement = _fold_surface_agreement(pooled, fold_corrections)
        region_summaries = _region_validation_summaries(
            validation_rows, blocks, surface_agreement,
            expected_region_keys,
        )
        region_summaries_by_model[model] = region_summaries
        model_report = {
            "parameterFile": f"{model}/pooled_parameters.json",
            "orders": {
                "theta": model_cfg.theta_order,
                "fdPhi": model_cfg.phi_order,
                "cdFourierHarmonics": model_cfg.cd_fourier_harmonics,
            },
            "pooledRegions": len(pooled["regions"]),
            "pooledSkippedRegions": pooled["skippedRegions"],
            "folds": fold_details,
            "validationSummary": aggregate,
            "regionValidation": list(region_summaries.values()),
            "validationRows": validation_rows,
        }
        report["models"][model] = model_report
        _write_json(model_dir / "heldout_validation.json", model_report)
        if make_plots:
            _plot_validation(
                validation_rows, model_dir / "heldout_closure.png", model,
                particle, dataset_tag, cfg.beam_energy,
            )

    recommendation = _choose_region_models(
        region_summaries_by_model, minimum_model_improvement_fraction
    )
    mixed_parameters = _build_mixed_parameters(
        pooled_by_model, recommendation, dataset_tag, raw_runs
    )
    if mixed_parameters is not None:
        mixed_filename = "recommended_mixed_parameters.json"
        _write_json(output_dir / mixed_filename, mixed_parameters)
        recommendation["parameterFile"] = mixed_filename

    if make_plots:
        _plot_region_model_comparison(
            region_summaries_by_model, recommendation,
            output_dir / "sector_model_comparison.png",
            particle, dataset_tag, cfg.beam_energy,
        )

    pooled_for_stability = mixed_parameters
    pooled_stability_model = "recommended-mixed" if mixed_parameters else None
    if pooled_for_stability is None and pooled_by_model:
        pooled_stability_model = max(
            pooled_by_model, key=lambda model: MODEL_COMPLEXITY[model]
        )
        pooled_for_stability = pooled_by_model[pooled_stability_model]
    if pooled_for_stability is not None:
        stability = common_cell_stability(
            pooled_for_stability, selected, blocks, cfg, particle
        )
        report["commonCellReferenceModel"] = pooled_stability_model
        report["commonCellStability"] = stability
        if make_plots:
            _plot_common_cell_heatmaps(
                stability, output_dir / "common_cell_stability",
                dataset_tag, cfg.beam_energy,
            )
    report["modelComparison"] = model_summaries
    report["regionModelComparison"] = {
        model: list(summaries.values())
        for model, summaries in region_summaries_by_model.items()
    }
    report["recommendation"] = recommendation
    _write_json(output_dir / "run_validation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit pooled elastic momentum surfaces and validate them on alternating "
            "contiguous run blocks without refitting the held-out events."
        )
    )
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--particle", choices=("electron", "proton"), default="electron")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_ORDERS),
        default=list(MODEL_ORDERS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--run-catalog", type=Path,
        help="JSON run catalog with a top-level runs object and run_class fields",
    )
    parser.add_argument(
        "--include-run-classes", nargs="+",
        help="retain only these run classes; requires --run-catalog",
    )
    block_group = parser.add_mutually_exclusive_group()
    block_group.add_argument(
        "--block-by-run-class", action="store_true",
        help=(
            "form candidate-target blocks within, never across, contiguous "
            "run-class periods"
        ),
    )
    block_group.add_argument(
        "--fold-by-run-class", action="store_true",
        help=(
            "with exactly two included classes, train on each class and hold out "
            "the other"
        ),
    )
    parser.add_argument("--block-target-selected", type=int, default=100_000)
    parser.add_argument("--min-per-run-core-entries", type=int, default=200)
    parser.add_argument(
        "--minimum-model-improvement-fraction", type=float, default=0.10,
        help=(
            "minimum relative held-out cell-RMS improvement required before "
            "selecting a more complex model in one detector region"
        ),
    )
    parser.add_argument("--coplanarity-max-deg", type=float, default=3.0)
    parser.add_argument("--theta-balance-max-deg", type=float, default=2.0)
    parser.add_argument("--missing-energy-max-gev", type=float, default=0.75)
    parser.add_argument("--theta-min-deg", type=float)
    parser.add_argument("--theta-max-deg", type=float)
    parser.add_argument("--theta-trim-quantile", type=float, default=0.005)
    parser.add_argument("--residual-trim-quantile", type=float, default=0.01)
    parser.add_argument("--theta-bins", type=int, default=10)
    parser.add_argument("--phi-bins", type=int, default=7)
    parser.add_argument(
        "--profile-binning", choices=("fixed", "adaptive"), default="adaptive"
    )
    parser.add_argument("--max-theta-bin-width-deg", type=float, default=0.75)
    parser.add_argument("--target-cell-entries", type=int, default=2_000)
    parser.add_argument("--min-phi-cells-per-theta", type=int, default=1)
    parser.add_argument("--min-bin-entries", type=int, default=300)
    parser.add_argument("--min-region-entries", type=int, default=1_000)
    parser.add_argument("--peak-search-max-abs-residual", type=float, default=0.10)
    parser.add_argument("--peak-seed-half-width", type=float, default=0.03)
    parser.add_argument("--core-sigma-clip", type=float, default=3.0)
    parser.add_argument("--min-core-fraction", type=float, default=0.20)
    parser.add_argument("--min-peak-significance", type=float, default=3.0)
    parser.add_argument("--max-core-width", type=float, default=0.05)
    parser.add_argument("--min-profile-cells-per-parameter", type=float, default=2.0)
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument("--max-abs-surface-correction", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_energy <= 0.0:
        raise ValueError("beam energy must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("maximum rows must be positive")
    if args.min_per_run_core_entries < 2:
        raise ValueError("minimum per-run core entries must be at least two")
    if not 0.0 <= args.minimum_model_improvement_fraction < 1.0:
        raise ValueError("minimum model improvement fraction must be in [0, 1)")
    if args.include_run_classes and args.run_catalog is None:
        raise ValueError("--include-run-classes requires --run-catalog")
    if (args.block_by_run_class or args.fold_by_run_class) and args.run_catalog is None:
        raise ValueError("run-class block modes require --run-catalog")
    if args.fold_by_run_class and len(args.include_run_classes or ()) != 2:
        raise ValueError(
            "--fold-by-run-class requires exactly two --include-run-classes"
        )
    arrays = load_candidate_inputs(args.input_files, args.tree)
    if args.max_rows is not None:
        arrays = subset_arrays(
            arrays,
            np.arange(next(iter(arrays.values())).size) < args.max_rows,
        )
    run_class_by_run: dict[int, str] | None = None
    run_selection: dict[str, object] | None = None
    if args.run_catalog is not None:
        arrays, run_class_by_run, run_selection = filter_arrays_by_run_classes(
            arrays,
            load_run_catalog(args.run_catalog),
            args.include_run_classes,
            catalog_path=args.run_catalog,
        )
    run_block_mode = (
        "class-fold" if args.fold_by_run_class else
        "class-boundary" if args.block_by_run_class else
        "candidate-target"
    )
    cfg = ElasticFitConfig(
        beam_energy=args.beam_energy,
        torus=args.torus,
        coplanarity_max_deg=args.coplanarity_max_deg,
        theta_balance_max_deg=args.theta_balance_max_deg,
        missing_energy_max_gev=args.missing_energy_max_gev,
        theta_min_deg=args.theta_min_deg,
        theta_max_deg=args.theta_max_deg,
        theta_trim_quantile=args.theta_trim_quantile,
        residual_trim_quantile=args.residual_trim_quantile,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        profile_binning=args.profile_binning,
        max_theta_bin_width_deg=args.max_theta_bin_width_deg,
        target_cell_entries=args.target_cell_entries,
        min_phi_cells_per_theta=args.min_phi_cells_per_theta,
        min_bin_entries=args.min_bin_entries,
        min_region_entries=args.min_region_entries,
        peak_search_max_abs_residual=args.peak_search_max_abs_residual,
        peak_seed_half_width=args.peak_seed_half_width,
        core_sigma_clip=args.core_sigma_clip,
        min_core_fraction=args.min_core_fraction,
        min_peak_significance=args.min_peak_significance,
        max_core_width=args.max_core_width,
        min_profile_cells_per_parameter=args.min_profile_cells_per_parameter,
        max_condition_number=args.max_condition_number,
        max_abs_surface_correction=args.max_abs_surface_correction,
    )
    report = run_validation(
        arrays, cfg,
        particle=args.particle,
        models=args.models,
        block_target=args.block_target_selected,
        output_dir=args.output_dir,
        dataset_tag=args.dataset_tag,
        make_plots=not args.no_plots,
        minimum_per_run_core_entries=args.min_per_run_core_entries,
        minimum_model_improvement_fraction=(
            args.minimum_model_improvement_fraction
        ),
        run_class_by_run=run_class_by_run,
        run_block_mode=run_block_mode,
        run_class_order=args.include_run_classes or (),
        run_selection=run_selection,
    )
    print(f"Wrote multi-run validation to {args.output_dir}")
    recommendation = report["recommendation"]
    print(
        "Diagnostic model recommendation: "
        f"{recommendation['model'] or 'none'} ({recommendation['status']})"
    )
    for region in recommendation["regions"]:
        print(
            f"  {region['region']}: {region['model'] or 'none'} "
            f"({region['status']})"
        )


if __name__ == "__main__":
    main()
