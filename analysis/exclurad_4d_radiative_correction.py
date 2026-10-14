#!/usr/bin/env python3
"""Compatibility wrapper for the maintained EXCLUGEN correction command.

New campaigns should call ``analysis/run_analysis.py radiative-correction``.
This historical entry point remains so recorded command lines still work, but
it delegates discovery, normalization, uncertainty, support, and artifact
writing to that maintained implementation instead of carrying a second engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Support both ``python -m analysis...`` and direct execution from the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eppi0.event_sample import _dis, _minus_t
from analysis.eppi0.radiative_correction import weighted_effective_count
from analysis.run_analysis import command_radiative_correction


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper: build analysis-binned C_rad through the "
            "maintained EXCLUGEN provider backend"
        )
    )
    parser.add_argument("production_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--min-counts", type=int, default=5)
    parser.add_argument("--expected-runs", type=int, default=200)
    parser.add_argument("--execution-path", default="farm")
    parser.add_argument("--generator-channel", default="pi0p")
    parser.add_argument("--generator-model", default="pi0.amp2614")
    parser.add_argument("--generator-revision")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    parser.add_argument("--allow-mixed-dirty-state", action="store_true")
    parser.add_argument("--progress-runs", type=int, default=25)
    return parser.parse_args()


def dis_kinematics(
    electron: np.ndarray,
    beam_energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for the analysis-core DIS calculation."""
    return _dis(electron, beam_energy)


def minus_t(proton: np.ndarray) -> np.ndarray:
    """Compatibility alias for the analysis-core ``-t`` calculation."""
    return _minus_t(proton)


def _command_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        generator="exclugen",
        production_root=args.production_dir,
        born=None,
        radiative=None,
        config=args.config,
        output=args.output,
        execution_path=args.execution_path,
        generator_channel=args.generator_channel,
        generator_model=args.generator_model,
        generator_revision=args.generator_revision,
        expected_runs=args.expected_runs,
        allow_revision_mismatch=args.allow_revision_mismatch,
        allow_mixed_dirty_state=args.allow_mixed_dirty_state,
        v_max=args.v_max,
        progress_runs=args.progress_runs,
        min_counts=args.min_counts,
        max_files=None,
        max_events=None,
        chunk_size=200_000,
        progress_chunks=0,
        normalization_ratio=None,
        born_integrated_cross_section=None,
        radiative_integrated_cross_section=None,
        born_normalization_file=None,
        radiative_normalization_file=None,
        max_normalization_files=None,
        diagnostic_pdf=None,
        diagnostic_csv=None,
        diagnostic_quilt=False,
        diagnostic_quilt_scale_mode="panel",
    )


def _write_summary(args: argparse.Namespace) -> Path:
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    with np.load(args.output, allow_pickle=False) as data:
        reliable = np.asarray(data["reliable"], dtype=bool)
        overlap = np.asarray(data["support_overlap"], dtype=bool)
        summary = {
            "artifact": str(args.output),
            "analysis_config": str(args.config),
            "provider": str(data["generator_provider"]),
            "model": str(data["generator_model"]),
            "channel": str(data["generator_channel"]),
            "shape_Q2_xB_t_phi": list(reliable.shape),
            "total_bins": int(reliable.size),
            "reliable_bins": int(np.count_nonzero(reliable)),
            "overlap_bins": int(np.count_nonzero(overlap)),
            "min_effective_count": int(data["min_counts"]),
            "v_definition": str(data["offline_radiative_v_definition"]),
            "v_max_GeV2": float(data["offline_radiative_v_max_GeV2"]),
            "radiative_histogram_renormalized_after_v_cut": bool(
                data["radiative_histogram_renormalized_after_v_cut"]
            ),
            "normalization": str(data["normalization_method"]),
            "born_generated_events": int(data["born_events_seen"]),
            "radiative_generated_events": int(data["radiative_events_seen"]),
            "born_analysis_bin_events": int(data["born_in_range"]),
            "radiative_vcut_analysis_bin_events": int(data["radiative_in_range"]),
            "born_sigma_nb_mean": float(data["born_integrated_cross_section"]),
            "radiative_sigma_nb_mean": float(
                data["radiative_integrated_cross_section"]
            ),
            "normalization_ratio_no_offline_cut": float(
                data["normalization_ratio"]
            ),
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def main() -> None:
    args = arguments()
    command_radiative_correction(_command_namespace(args))
    summary_path = _write_summary(args)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
