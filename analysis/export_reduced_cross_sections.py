#!/usr/bin/env python3

"""Export valid reduced cross sections and covariance in a shareable package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


EDGE_NAMES = ("q2_edges", "xb_edges", "t_edges", "phi_edges")
REQUIRED_FIELDS = {
    *EDGE_NAMES,
    "reduced_cross_section",
    "uncertainty",
    "final_validity_mask",
}
KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
CSV_FIELDS = (
    "campaign_key",
    "campaign_label",
    "iq2",
    "ixb",
    "it",
    "iphi",
    "Q2_low_GeV2",
    "Q2_high_GeV2",
    "Q2_flux_coordinate_GeV2",
    "xB_low",
    "xB_high",
    "xB_flux_coordinate",
    "minus_t_low_GeV2",
    "minus_t_high_GeV2",
    "minus_t_center_GeV2",
    "phi_low_deg",
    "phi_high_deg",
    "phi_center_deg",
    "reduced_cross_section_nb_per_GeV2_rad",
    "propagated_statistical_and_finite_MC_uncertainty_nb_per_GeV2_rad",
    "relative_uncertainty",
    "combination_contributor_count",
    "torus_plus1_BLUE_weight",
    "torus_minus1_BLUE_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        nargs=3,
        metavar=("KEY", "LABEL", "CROSS_SECTION_NPZ"),
        action="append",
        required=True,
        help="repeat for each campaign or combined result",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-note",
        default=(
            "provisional RGA Fall 2018 extraction using closure-selected iterations, "
            "joint data-and-response count-bootstrap covariance, and unit current weights"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(artifact: dict[str, np.ndarray], name: str, default: str) -> str:
    if name not in artifact:
        return default
    value = np.asarray(artifact[name])
    if value.size != 1:
        raise ValueError(f"{name} is not scalar")
    return str(value.reshape(()).item())


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(REQUIRED_FIELDS.difference(source.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        return {name: np.asarray(source[name]) for name in source.files}


def value_grid(
    artifact: dict[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in artifact:
        return np.full(shape, np.nan)
    values = np.asarray(artifact[name], dtype=float)
    if values.shape == shape:
        return values
    if values.size == int(np.prod(shape)):
        return values.reshape(shape)
    raise ValueError(f"{name} has shape {values.shape}; expected {shape}")


def optional_grid(
    artifact: dict[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in artifact:
        return np.full(shape, np.nan)
    return value_grid(artifact, name, shape)


def validate_artifact(
    artifact: dict[str, np.ndarray], path: Path
) -> tuple[tuple[int, ...], np.ndarray]:
    shape = tuple(len(np.asarray(artifact[name])) - 1 for name in EDGE_NAMES)
    values = np.asarray(artifact["reduced_cross_section"], dtype=float)
    uncertainty = np.asarray(artifact["uncertainty"], dtype=float)
    valid = np.asarray(artifact["final_validity_mask"], dtype=bool)
    if values.shape != shape or uncertainty.shape != shape or valid.shape != shape:
        raise ValueError(f"{path} arrays do not match edge-defined shape {shape}")
    valid &= np.isfinite(values)
    valid &= np.isfinite(uncertainty) & (uncertainty > 0.0)
    if "covariance_phi" in artifact:
        covariance = np.asarray(artifact["covariance_phi"], dtype=float)
        expected = shape[:-1] + (shape[-1], shape[-1])
        if covariance.shape != expected:
            raise ValueError(f"{path} covariance_phi has shape {covariance.shape}; expected {expected}")
        diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
        comparison = valid & np.isfinite(diagonal)
        if not np.allclose(
            diagonal[comparison], uncertainty[comparison] ** 2,
            rtol=5.0e-10, atol=1.0e-18,
        ):
            raise ValueError(f"{path} covariance diagonal does not equal uncertainty squared")
    return shape, valid


def rows_for_sample(
    key: str,
    label: str,
    artifact: dict[str, np.ndarray],
    path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    shape, valid = validate_artifact(artifact, path)
    q2_edges, xb_edges, t_edges, phi_edges = (
        np.asarray(artifact[name], dtype=float) for name in EDGE_NAMES
    )
    q2_coordinate = value_grid(artifact, "flux_q2_coordinate", shape)
    xb_coordinate = value_grid(artifact, "flux_xb_coordinate", shape)
    values = np.asarray(artifact["reduced_cross_section"], dtype=float)
    errors = np.asarray(artifact["uncertainty"], dtype=float)
    contributors = optional_grid(artifact, "combination_contributor_count", shape)
    left_weight = optional_grid(artifact, "combination_left_weight", shape)
    right_weight = optional_grid(artifact, "combination_right_weight", shape)
    rows: list[dict[str, object]] = []
    for iq2, ixb, it, iphi in np.argwhere(valid):
        value = float(values[iq2, ixb, it, iphi])
        error = float(errors[iq2, ixb, it, iphi])
        rows.append(
            {
                "campaign_key": key,
                "campaign_label": label,
                "iq2": int(iq2),
                "ixb": int(ixb),
                "it": int(it),
                "iphi": int(iphi),
                "Q2_low_GeV2": float(q2_edges[iq2]),
                "Q2_high_GeV2": float(q2_edges[iq2 + 1]),
                "Q2_flux_coordinate_GeV2": float(q2_coordinate[iq2, ixb, it, iphi]),
                "xB_low": float(xb_edges[ixb]),
                "xB_high": float(xb_edges[ixb + 1]),
                "xB_flux_coordinate": float(xb_coordinate[iq2, ixb, it, iphi]),
                "minus_t_low_GeV2": float(t_edges[it]),
                "minus_t_high_GeV2": float(t_edges[it + 1]),
                "minus_t_center_GeV2": float(0.5 * (t_edges[it] + t_edges[it + 1])),
                "phi_low_deg": float(phi_edges[iphi]),
                "phi_high_deg": float(phi_edges[iphi + 1]),
                "phi_center_deg": float(0.5 * (phi_edges[iphi] + phi_edges[iphi + 1])),
                "reduced_cross_section_nb_per_GeV2_rad": value,
                "propagated_statistical_and_finite_MC_uncertainty_nb_per_GeV2_rad": error,
                "relative_uncertainty": error / abs(value) if value != 0.0 else np.nan,
                "combination_contributor_count": (
                    int(contributors[iq2, ixb, it, iphi])
                    if np.isfinite(contributors[iq2, ixb, it, iphi]) else ""
                ),
                "torus_plus1_BLUE_weight": (
                    float(left_weight[iq2, ixb, it, iphi])
                    if np.isfinite(left_weight[iq2, ixb, it, iphi]) else ""
                ),
                "torus_minus1_BLUE_weight": (
                    float(right_weight[iq2, ixb, it, iphi])
                    if np.isfinite(right_weight[iq2, ixb, it, iphi]) else ""
                ),
            }
        )
    metadata = {
        "key": key,
        "label": label,
        "source": str(path),
        "sha256": sha256(path),
        "array_shape": list(shape),
        "valid_bins": len(rows),
        "units": scalar_text(
            artifact, "reduced_cross_section_units", "nb/(GeV^2 rad)"
        ),
        "phase_space_definition": scalar_text(
            artifact, "phase_space_definition", "unavailable"
        ),
        "final_validity_definition": scalar_text(
            artifact, "final_validity_definition", "stored final_validity_mask"
        ),
        "covariance_available": "covariance_phi" in artifact,
        "covariance_definition": scalar_text(
            artifact, "covariance_phi_definition", "unavailable"
        ),
        "covariance_bootstrap_experiments": (
            int(np.asarray(artifact["covariance_phi_bootstrap_experiments"]).item())
            if "covariance_phi_bootstrap_experiments" in artifact else 0
        ),
        "uncertainty_definition": scalar_text(
            artifact,
            "combination_uncertainty_definition",
            (
                "diagonal square root of covariance_phi: propagated data and finite-"
                "response count-bootstrap uncertainty plus finite radiative-correction "
                "MC uncertainty; remaining campaign systematic covariance is absent"
            ),
        ),
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for name, value in row.items():
                if isinstance(value, (float, np.floating)):
                    formatted[name] = format(float(value), ".16g") if np.isfinite(value) else ""
                else:
                    formatted[name] = value
            writer.writerow(formatted)


def write_readme(
    path: Path,
    samples: list[dict[str, object]],
    analysis_note: str,
) -> None:
    lines = [
        "# RGA Fall 2018 reduced cross sections",
        "",
        analysis_note.rstrip(".") + ".",
        "",
        "Each campaign CSV contains only bins passing that artifact's final validity mask.",
        "The reduced cross-section unit is `nb/(GeV^2 rad)`. Phi is recorded in degrees.",
        "The Q2 and xB coordinate columns are the coordinates used to evaluate the virtual-photon flux.",
        "The -t and phi coordinate columns are geometric bin centers; all four bin edges are included.",
        "",
        "The error column is the square root of the stored covariance diagonal. It includes",
        "data counting fluctuations, finite response-simulation counting fluctuations, and",
        "finite radiative-correction simulation uncertainty. It does not include detector,",
        "selection, luminosity, current-efficiency, bin-centering-model, or other pending",
        "campaign systematic covariance. The supplied production variants use unit current",
        "weights. Treat these exports as provisional until those systematics are finalized.",
        "",
        "`rga_fa18_reduced_cross_section_phi_covariance.npz` stores the full within-cell",
        "phi covariance and validity mask for each sample. Use it instead of independent",
        "diagonal errors when fitting harmonics across phi.",
        "",
        "## Samples",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"- `{sample['key']}`: {sample['label']}",
                f"  - valid bins: {sample['valid_bins']}",
                f"  - source SHA-256: `{sample['sha256']}`",
                f"  - covariance replicas: {sample['covariance_bootstrap_experiments']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    covariance_payload: dict[str, np.ndarray] = {}
    sample_metadata: list[dict[str, object]] = []
    common_edges: dict[str, np.ndarray] | None = None
    for key, label, raw_path in args.sample:
        if not KEY_PATTERN.match(key):
            raise ValueError(f"invalid sample key {key!r}; use letters, numbers, underscores")
        path = Path(raw_path).resolve()
        artifact = load_artifact(path)
        edges = {name: np.asarray(artifact[name]) for name in EDGE_NAMES}
        if common_edges is None:
            common_edges = edges
            covariance_payload.update(edges)
        elif any(not np.array_equal(common_edges[name], edges[name]) for name in EDGE_NAMES):
            raise ValueError(f"{path} does not share the package bin edges")
        rows, metadata = rows_for_sample(key, label, artifact, path)
        write_csv(output_dir / f"{key}_reduced_cross_sections.csv", rows)
        sample_metadata.append(metadata)
        covariance_payload[f"{key}_final_validity_mask"] = np.asarray(
            artifact["final_validity_mask"], dtype=bool
        )
        if "covariance_phi" in artifact:
            covariance_payload[f"{key}_covariance_phi"] = np.asarray(
                artifact["covariance_phi"], dtype=float
            )
        covariance_payload[f"{key}_source"] = np.asarray(str(path))
        covariance_payload[f"{key}_source_sha256"] = np.asarray(metadata["sha256"])
        print(f"{key}: exported {len(rows)} valid bins")

    covariance_path = output_dir / "rga_fa18_reduced_cross_section_phi_covariance.npz"
    np.savez_compressed(covariance_path, **covariance_payload)
    summary = {
        "schema_version": 1,
        "package": "RGA Fall 2018 provisional reduced cross sections",
        "analysis_note": args.analysis_note,
        "csv_definition": "one row per final-valid 4D cross-section bin",
        "uncertainty_scope": (
            "propagated data and finite-MC counting uncertainty including finite "
            "radiative-correction MC uncertainty; remaining campaign systematic "
            "covariance is not included"
        ),
        "current_efficiency_treatment": "unit current weights",
        "samples": sample_metadata,
        "covariance_artifact": str(covariance_path),
    }
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme_path = output_dir / "README.md"
    write_readme(readme_path, sample_metadata, args.analysis_note)
    print(f"Wrote {covariance_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
