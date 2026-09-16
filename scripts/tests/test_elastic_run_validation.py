from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from scripts.calibration.elastic_momentum import (
    ElasticFitConfig,
    elastic_electron_momentum,
    elastic_proton_momentum,
    elastic_proton_theta_from_electron,
)
from scripts.calibration.elastic_run_validation import (
    MODEL_ORDERS,
    _choose_region_models,
    make_fixed_run_blocks,
    make_run_blocks,
    run_validation,
)


class ElasticRunValidationTests(unittest.TestCase):
    @staticmethod
    def _region_summary(
        sector: int, score: float, fold_a: float, fold_b: float,
        surface_rms: float = 0.01,
    ) -> dict[str, object]:
        return {
            "pid": 11,
            "detector": 1,
            "sector": sector,
            "region": f"pid11_det1_sector{sector}",
            "medianCellCenterRmsAfter": score,
            "completeHeldOutBlockCoverage": True,
            "heldoutFolds": {
                "A": {"medianCellCenterRmsAfter": fold_a},
                "B": {"medianCellCenterRmsAfter": fold_b},
            },
            "foldSurfaceAgreement": {
                "commonPooledCellCenters": 20,
                "rms": surface_rms,
                "medianAbs": 0.5 * surface_rms,
                "maxAbs": 2.0 * surface_rms,
            },
        }

    def test_quadratic_theta_phi_model_matches_six_term_fd_surface(self) -> None:
        self.assertEqual(MODEL_ORDERS["theta2-phi"], (2, 1, 1))

    def test_models_are_selected_independently_by_sector(self) -> None:
        key1 = (11, 1, 1)
        key2 = (11, 1, 2)
        summaries = {
            "constant": {
                key1: self._region_summary(1, 0.120, 0.125, 0.115),
                key2: self._region_summary(2, 0.120, 0.125, 0.115),
            },
            "theta-linear": {
                key1: self._region_summary(1, 0.090, 0.092, 0.088),
                key2: self._region_summary(2, 0.112, 0.114, 0.110),
            },
            "theta-phi": {
                key1: self._region_summary(1, 0.085, 0.086, 0.084),
                key2: self._region_summary(2, 0.085, 0.087, 0.083),
            },
            "theta2-phi": {
                key1: self._region_summary(1, 0.083, 0.084, 0.082),
                key2: self._region_summary(2, 0.082, 0.084, 0.080),
            },
        }
        recommendation = _choose_region_models(summaries, 0.10)
        chosen = {
            entry["sector"]: entry["model"]
            for entry in recommendation["regions"]
        }
        self.assertEqual(recommendation["model"], "mixed")
        self.assertEqual(chosen[1], "theta-linear")
        self.assertEqual(chosen[2], "theta-phi")

    def test_complex_model_must_improve_both_held_out_folds(self) -> None:
        key = (11, 1, 1)
        summaries = {
            "theta-linear": {
                key: self._region_summary(1, 0.100, 0.090, 0.110),
            },
            "theta-phi": {
                key: self._region_summary(1, 0.080, 0.095, 0.070),
            },
        }
        recommendation = _choose_region_models(summaries, 0.10)
        self.assertEqual(recommendation["model"], "theta-linear")
        comparison = recommendation["regions"][0]["selectionTrace"][0]
        self.assertFalse(comparison["improvesEveryHeldOutFold"])
        self.assertFalse(comparison["selected"])

    def test_run_blocks_are_contiguous_and_keep_small_tail(self) -> None:
        runs = np.repeat([100, 101, 102, 103], [40, 70, 20, 10])
        blocks = make_run_blocks(runs, 60)
        self.assertEqual(blocks[0].runs, (100, 101))
        self.assertEqual(blocks[1].runs, (102, 103))
        self.assertEqual(sum(block.selected_candidates for block in blocks), runs.size)

    def test_run_blocks_can_split_even_when_only_two_runs_are_available(self) -> None:
        runs = np.repeat([5423, 5424], [150, 250])
        blocks = make_run_blocks(runs, 1_000)
        self.assertEqual([block.runs for block in blocks], [(5423,), (5424,)])

    def test_fixed_run_blocks_preserve_partition_and_update_counts(self) -> None:
        runs = np.asarray([5423, 5423, 5424, 5425, 5425, 5425])
        blocks = make_fixed_run_blocks(runs, [(5423, 5424), (5425,)])
        self.assertEqual([block.runs for block in blocks], [(5423, 5424), (5425,)])
        self.assertEqual([block.selected_candidates for block in blocks], [3, 3])

    def test_fixed_run_blocks_reject_uncovered_selected_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover selected runs"):
            make_fixed_run_blocks(np.asarray([5423, 5424]), [(5423,), (5425,)])

    def test_pooled_models_are_evaluated_on_held_out_runs(self) -> None:
        rng = np.random.default_rng(101)
        beam_energy = 10.604
        runs = np.arange(5423, 5427)
        entries_per_run_sector = 400
        run = np.repeat(runs, 6 * entries_per_run_sector)
        sector = np.tile(
            np.repeat(np.arange(1, 7), entries_per_run_sector), runs.size
        )
        entries = run.size
        electron_theta = np.deg2rad(rng.uniform(6.2, 10.5, entries))
        local_phi = rng.uniform(-27.0, 27.0, entries)
        electron_phi_deg = local_phi + 60.0 * (sector - 1)
        electron_phi = np.deg2rad(
            (electron_phi_deg + 180.0) % 360.0 - 180.0
        )
        proton_theta = elastic_proton_theta_from_electron(
            electron_theta, beam_energy
        )
        expected_electron = elastic_electron_momentum(
            electron_theta, beam_energy
        )
        sector_bias = 0.001 + 0.0005 * sector
        run_shift = 0.0001 * (run - np.mean(runs))
        residual = sector_bias + run_shift + rng.normal(0.0, 0.008, entries)
        arrays = {
            "runNum": run,
            "electronP": expected_electron / (1.0 + residual),
            "electronTheta": electron_theta,
            "electronPhi": electron_phi,
            "electronDet": np.ones(entries, dtype=int),
            "electronSector": sector,
            "protonP": elastic_proton_momentum(proton_theta, beam_energy),
            "protonTheta": proton_theta,
            "protonPhi": electron_phi + np.pi,
            "protonDet": np.ones(entries, dtype=int),
            "protonSector": np.ones(entries, dtype=int),
            "nPid11": np.ones(entries, dtype=int),
            "nPid2212": np.ones(entries, dtype=int),
        }
        cfg = ElasticFitConfig(
            beam_energy=beam_energy, torus=1, theta_min_deg=6.1,
            theta_bins=3, phi_bins=4, profile_binning="fixed",
            min_bin_entries=20, min_region_entries=100,
            max_condition_number=100.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_validation(
                arrays, cfg, particle="electron",
                models=[
                    "constant", "theta-linear", "theta-phi", "theta2-phi"
                ],
                block_target=1_000,
                output_dir=Path(temporary_directory), dataset_tag="synthetic",
                make_plots=False, minimum_per_run_core_entries=50,
            )
            self.assertEqual(len(report["runBlocks"]), 4)
            self.assertTrue(
                report["modelComparison"]["constant"]["completeFoldRegionCoverage"]
            )
            self.assertGreater(
                report["modelComparison"]["constant"]["validatedCells"], 0
            )
            self.assertEqual(
                report["models"]["theta2-phi"]["orders"]["theta"], 2
            )
            self.assertTrue(report["perRunRegionStability"])
            self.assertTrue(
                (Path(temporary_directory) / "run_validation_report.json").is_file()
            )
            self.assertEqual(len(report["recommendation"]["regions"]), 6)
            self.assertTrue(
                (Path(temporary_directory) / "recommended_mixed_parameters.json")
                .is_file()
            )

    def test_unavailable_pooled_models_are_recorded_without_crashing(self) -> None:
        runs = np.array([5423, 5423, 5424, 5424])
        theta = np.deg2rad(np.full(4, 8.0))
        proton_theta = elastic_proton_theta_from_electron(theta, 10.604)
        arrays = {
            "runNum": runs,
            "electronP": elastic_electron_momentum(theta, 10.604),
            "electronTheta": theta,
            "electronPhi": np.zeros(4),
            "electronDet": np.ones(4, dtype=int),
            "electronSector": np.ones(4, dtype=int),
            "protonP": elastic_proton_momentum(proton_theta, 10.604),
            "protonTheta": proton_theta,
            "protonPhi": np.full(4, np.pi),
            "protonDet": np.ones(4, dtype=int),
            "protonSector": np.full(4, 4, dtype=int),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "scripts.calibration.elastic_run_validation.derive_corrections",
                side_effect=ValueError("insufficient independent cells"),
            ):
                report = run_validation(
                    arrays, ElasticFitConfig(beam_energy=10.604, torus=1),
                    particle="electron", models=["theta-phi"],
                    block_target=2, output_dir=Path(temporary_directory),
                    dataset_tag="failure-test", make_plots=False,
                )
            self.assertIsNone(report["recommendation"]["model"])
            self.assertIn("insufficient", report["models"]["theta-phi"]["reason"])


if __name__ == "__main__":
    unittest.main()
