from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.calibration.exclusive_particle_momentum import ExclusiveFitConfig
from scripts.calibration.exclusive_particle_run_validation import (
    _root_quality_selection_mask,
    _run_class_selection_mask,
    run_exclusive_validation,
)


class ExclusiveParticleRunValidationTests(unittest.TestCase):
    def test_root_quality_filters_are_recorded_independently(self) -> None:
        measured = np.asarray([1.2, 1.49, 1.8, 1.2])
        arrays = {
            "electronP": np.ones(4),
            "electronTheta": np.ones(4),
            "electronPhi": np.ones(4),
            "electronDet": np.ones(4, dtype=int),
            "electronSector": np.ones(4, dtype=int),
            "protonP": measured,
            "protonTheta": np.ones(4),
            "protonPhi": np.ones(4),
            "protonDet": np.ones(4, dtype=int),
            "protonSector": np.ones(4, dtype=int),
        }
        roots = np.tile(np.asarray([[1.0, 2.0]]), (4, 1))
        valid = np.ones(roots.shape, dtype=bool)
        photon_angles = np.asarray([
            [1.0, 10.0], [1.0, 10.0], [1.0, 10.0], [1.0, 1.2]
        ])

        def correction(*args, pid: int, **kwargs):
            if pid == 11:
                return np.ones(4), np.ones(4, dtype=bool), np.zeros(4)
            corrected = measured.copy()
            corrected[1] = 1.6
            return corrected, np.ones(4, dtype=bool), corrected / measured - 1.0

        with (
            patch(
                "scripts.calibration.exclusive_particle_run_validation."
                "apply_supported_particle_correction",
                side_effect=correction,
            ),
            patch(
                "scripts.calibration.exclusive_particle_run_validation._base_mask",
                return_value=(np.ones(4, dtype=bool), {}),
            ),
            patch(
                "scripts.calibration.exclusive_particle_run_validation."
                "proton_momentum_roots_eppi0",
                return_value=(roots, valid),
            ),
            patch(
                "scripts.calibration.exclusive_particle_run_validation."
                "photon_branch_metrics",
                return_value=(photon_angles, np.ones_like(photon_angles)),
            ),
        ):
            keep, metadata = _root_quality_selection_mask(
                arrays, {}, ExclusiveFitConfig(beam_energy=6.535, torus=1),
                np.ones(4, dtype=bool), stability_parameters={},
                photon_direction_min_gap_deg=1.0,
            )
        np.testing.assert_array_equal(keep, [True, False, False, True])
        self.assertEqual(metadata["stability"]["excludedBranchFlipEntries"], 1)
        self.assertEqual(
            metadata["photonDirection"]["excludedDisagreementEntries"], 1
        )
        self.assertEqual(metadata["retainedCohortEntries"], 2)

    def test_run_class_selection_mask_preserves_array_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            catalog = Path(directory) / "runs.json"
            catalog.write_text(json.dumps({
                "runs": {
                    "100": {"run_class": "L4"},
                    "200": {"run_class": "P3"},
                    "300": {"run_class": "P4"},
                }
            }))
            mask, metadata = _run_class_selection_mask(
                np.asarray([100, 200, 200, 300]), catalog, ["P3", "P4"]
            )
        np.testing.assert_array_equal(mask, [False, True, True, True])
        self.assertEqual(metadata["selectedCandidateRows"], 3)
        self.assertEqual(metadata["includedRunClasses"], ["P3", "P4"])

    def test_heldout_validation_selects_bilinear_surface(self) -> None:
        rng = np.random.default_rng(71)
        entries_per_run = 12_000
        runs = np.repeat(np.arange(5900, 5906), entries_per_run)
        momentum = rng.uniform(0.5, 2.0, runs.size)
        theta = rng.uniform(10.0, 30.0, runs.size)
        pn = (momentum - 1.25) / 0.75
        tn = (theta - 20.0) / 10.0
        residual = (
            0.015 + 0.012 * pn - 0.010 * tn + 0.014 * pn * tn
            + rng.normal(0.0, 0.008, runs.size)
        )
        sample = {
            "momentum": momentum,
            "thetaDeg": theta,
            "phiDeg": np.zeros(runs.size),
            "detector": np.full(runs.size, 2, dtype=int),
            "sector": np.zeros(runs.size, dtype=int),
            "residual": residual,
            "runNum": runs,
        }
        cfg = ExclusiveFitConfig(
            beam_energy=6.535, torus=1,
            momentum_bins=6, theta_bins=6,
            min_bin_entries=100, min_region_entries=1_000,
        )
        report, parameters = run_exclusive_validation(
            sample, cfg, particle="proton",
            models=["constant", "theta-linear", "momentum-theta"],
            block_target=10_000,
            minimum_improvement_fraction=0.10,
            dataset_tag="synthetic",
        )
        recommendation = report["recommendation"]["regions"][0]
        self.assertEqual(recommendation["model"], "momentum-theta")
        self.assertEqual(len(parameters["regions"]), 1)
        self.assertGreaterEqual(len(report["runBlocks"]), 2)
        bilinear = report["models"]["momentum-theta"]["regions"][0]
        self.assertTrue(bilinear["completeFoldCoverage"])
        self.assertLess(
            bilinear["medianHeldOutCellRmsAfter"],
            report["models"]["constant"]["regions"][0][
                "medianHeldOutCellRmsAfter"
            ],
        )


if __name__ == "__main__":
    unittest.main()
