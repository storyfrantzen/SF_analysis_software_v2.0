from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.calibration.exclusive_particle_momentum import ExclusiveFitConfig
from scripts.calibration.exclusive_particle_run_validation import (
    _run_class_selection_mask,
    run_exclusive_validation,
)


class ExclusiveParticleRunValidationTests(unittest.TestCase):
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
