#!/usr/bin/env python3

"""Combine matching torus-polarity cross sections with binwise BLUE weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.polarity_combination import combine_polarity_measurements


EDGE_NAMES = ("q2_edges", "xb_edges", "t_edges", "phi_edges")
SHARED_FIELDS = (
    "flux_q2_coordinate",
    "flux_xb_coordinate",
    "bin_volume",
    "bin_centering_C_BC",
    "bin_centering_reliable",
    "bin_centering_q2_center",
    "bin_centering_xB_center",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="torus+1")
    parser.add_argument("--right-label", default="torus-1")
    parser.add_argument(
        "--radiative-correction",
        type=Path,
        required=True,
        help=(
            "Shared C_rad artifact; delta_C/C_rad is treated as fully correlated "
            "between polarities"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(artifact, name: str) -> str | None:
    if name not in artifact.files:
        return None
    value = np.asarray(artifact[name])
    if value.size != 1:
        raise ValueError(f"{name} is not scalar")
    return str(value.reshape(()).item())


def final_mask(artifact, path: Path) -> np.ndarray:
    values = np.asarray(artifact["reduced_cross_section"], dtype=float)
    uncertainty = np.asarray(artifact["uncertainty"], dtype=float)
    if values.shape != uncertainty.shape:
        raise ValueError(f"cross section and uncertainty shapes differ: {path}")
    if "final_validity_mask" in artifact.files:
        mask = np.asarray(artifact["final_validity_mask"], dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(f"final_validity_mask has the wrong shape: {path}")
    else:
        mask = np.ones(values.shape, dtype=bool)
    return mask & np.isfinite(values) & np.isfinite(uncertainty) & (uncertainty > 0.0)


def require_matching(left, right, name: str) -> None:
    if name not in left.files or name not in right.files:
        if name in left.files or name in right.files:
            raise ValueError(f"only one polarity contains {name}")
        return
    first = np.asarray(left[name])
    second = np.asarray(right[name])
    if first.shape != second.shape:
        raise ValueError(f"polarity artifacts have incompatible {name} shapes")
    if np.issubdtype(first.dtype, np.number):
        matches = np.allclose(first, second, rtol=1.0e-12, atol=1.0e-12, equal_nan=True)
    else:
        matches = np.array_equal(first, second)
    if not matches:
        raise ValueError(f"polarity artifacts have incompatible {name}")


def quantiles(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return [float(value) for value in np.quantile(finite, [0.1, 0.5, 0.9])]


def main() -> int:
    args = parse_args()
    left_path = args.left.resolve()
    right_path = args.right.resolve()
    radiative_path = args.radiative_correction.resolve()
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    for name in (*EDGE_NAMES, *SHARED_FIELDS):
        require_matching(left, right, name)
    for name in ("reduced_cross_section_units", "phase_space_definition"):
        first = scalar_text(left, name)
        second = scalar_text(right, name)
        if first != second:
            raise ValueError(f"polarity artifacts have incompatible {name}")

    left_values = np.asarray(left["reduced_cross_section"], dtype=float)
    right_values = np.asarray(right["reduced_cross_section"], dtype=float)
    radiative = np.load(radiative_path, allow_pickle=False)
    c_rad = np.asarray(radiative["C_rad"], dtype=float)
    delta_c = np.asarray(radiative["delta_C"], dtype=float)
    if c_rad.shape != left_values.shape or delta_c.shape != left_values.shape:
        raise ValueError("radiative correction does not match cross-section binning")
    shared_relative = np.divide(
        np.abs(delta_c),
        np.abs(c_rad),
        out=np.zeros_like(c_rad),
        where=np.isfinite(c_rad) & np.isfinite(delta_c) & (c_rad != 0.0),
    )
    result = combine_polarity_measurements(
        left_values,
        left["uncertainty"],
        final_mask(left, left_path),
        right_values,
        right["uncertainty"],
        final_mask(right, right_path),
        shared_relative_uncertainty=shared_relative,
    )

    payload = {
        "reduced_cross_section": result.values,
        "uncertainty": result.uncertainties,
        "final_validity_mask": result.valid,
        "final_validity_definition": np.asarray(
            "union of valid polarity measurements; common bins use covariance-aware BLUE"
        ),
        "combination_method": np.asarray(
            "binwise BLUE with shared delta_C/C_rad fully correlated between polarities"
        ),
        "combination_contributor_count": result.contributor_count,
        "combination_left_weight": result.left_weight,
        "combination_right_weight": result.right_weight,
        "combination_overlap_pull": result.overlap_pull,
        "combination_overlap_chi2": result.overlap_chi2,
        "combination_shared_relative_uncertainty": (
            result.shared_relative_uncertainty
        ),
        "combination_left_label": np.asarray(args.left_label),
        "combination_right_label": np.asarray(args.right_label),
        "combination_left_source": np.asarray(str(left_path)),
        "combination_right_source": np.asarray(str(right_path)),
        "combination_left_sha256": np.asarray(sha256(left_path)),
        "combination_right_sha256": np.asarray(sha256(right_path)),
        "combination_radiative_source": np.asarray(str(radiative_path)),
        "combination_radiative_sha256": np.asarray(sha256(radiative_path)),
        "combination_uncertainty_definition": np.asarray(
            "input total uncertainties with shared radiative covariance retained; "
            "other pending systematic covariance is not included"
        ),
    }
    for name in (*EDGE_NAMES, *SHARED_FIELDS):
        if name in left.files:
            payload[name] = left[name]
    for name in (
        "reduced_cross_section_units",
        "phase_space_definition",
        "global_normalization",
        "bin_volume_definition",
        "bin_volume_integration_points",
        "flux_coordinate_definition",
        "bin_centering_path",
        "bin_centering_application",
    ):
        if name in left.files:
            payload[name] = left[name]
    component_luminosity = np.asarray(
        [float(left["luminosity_fb"]), float(right["luminosity_fb"])], dtype=float
    )
    payload["component_luminosity_fb"] = component_luminosity
    payload["luminosity_fb"] = float(component_luminosity.sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    both = result.contributor_count == 2
    only_left = result.contributor_count == 1
    only_left &= result.left_weight == 1.0
    only_right = result.contributor_count == 1
    only_right &= result.right_weight == 1.0
    positive_ratio = both & (left_values > 0.0) & (right_values > 0.0)
    ratio = np.divide(
        right_values,
        left_values,
        out=np.full(left_values.shape, np.nan),
        where=positive_ratio,
    )
    finite_pull = result.overlap_pull[np.isfinite(result.overlap_pull)]
    summary = {
        "method": str(np.asarray(payload["combination_method"]).item()),
        "left": {
            "label": args.left_label,
            "source": str(left_path),
            "sha256": payload["combination_left_sha256"].item(),
        },
        "right": {
            "label": args.right_label,
            "source": str(right_path),
            "sha256": payload["combination_right_sha256"].item(),
        },
        "shared_radiative_correction": {
            "source": str(radiative_path),
            "sha256": payload["combination_radiative_sha256"].item(),
        },
        "valid_bins": {
            "left": int(np.count_nonzero(final_mask(left, left_path))),
            "right": int(np.count_nonzero(final_mask(right, right_path))),
            "both": int(np.count_nonzero(both)),
            "left_only": int(np.count_nonzero(only_left)),
            "right_only": int(np.count_nonzero(only_right)),
            "combined_union": int(np.count_nonzero(result.valid)),
        },
        "overlap": {
            "right_over_left_q10_median_q90": quantiles(ratio),
            "pull_q10_median_q90": quantiles(finite_pull),
            "absolute_pull_gt_2": int(np.count_nonzero(np.abs(finite_pull) > 2.0)),
            "absolute_pull_gt_3": int(np.count_nonzero(np.abs(finite_pull) > 3.0)),
        },
        "uncertainty_limitation": (
            "shared radiative uncertainty is correlated; pending detector, selection, "
            "bin-centering, and other systematic covariance is not included"
        ),
    }
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else args.output.resolve().with_name("combination_summary.json")
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Common valid bins: {summary['valid_bins']['both']}")
    print(f"Left-only valid bins: {summary['valid_bins']['left_only']}")
    print(f"Right-only valid bins: {summary['valid_bins']['right_only']}")
    print(f"Combined valid bins: {summary['valid_bins']['combined_union']}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
