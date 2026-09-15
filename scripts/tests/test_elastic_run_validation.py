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
    make_run_blocks,
    run_validation,
)


class ElasticRunValidationTests(unittest.TestCase):
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
            theta_bins=2, phi_bins=2, profile_binning="fixed",
            min_bin_entries=20, min_region_entries=100,
            max_condition_number=100.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_validation(
                arrays, cfg, particle="electron",
                models=["constant", "theta-linear"], block_target=1_000,
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
            self.assertTrue(report["perRunRegionStability"])
            self.assertTrue(
                (Path(temporary_directory) / "run_validation_report.json").is_file()
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
