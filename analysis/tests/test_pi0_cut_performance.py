from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_pi0_cut_performance import (  # noqa: E402
    WorkingPoint,
    fit_pi0,
    metric_row,
    required_tag_mask,
    threshold_mask,
    topology_mask,
    working_point,
)


class Pi0CutPerformanceTests(unittest.TestCase):
    def test_working_point_and_detector_dependent_thresholds(self) -> None:
        point = working_point("fd06:0.6:0.4")
        arrays = {
            "g1Det": np.array([1, 0, 1, 0]),
            "g2Det": np.array([1, 1, 0, 0]),
            "gamma1P": np.array([0.7, 0.45, 0.55, 0.45]),
            "gamma2P": np.array([0.65, 0.65, 0.45, 0.45]),
        }
        np.testing.assert_array_equal(
            threshold_mask(arrays, point),
            [True, True, False, True],
        )
        np.testing.assert_array_equal(
            topology_mask(arrays, "FD/FT"),
            [False, True, True, False],
        )

    def test_required_tags_must_be_evaluated_and_not_failed(self) -> None:
        evaluated = np.array(["electron.vertex,gamma.fiducial"] * 3)
        failed = np.array(["", "gamma.fiducial", "electron.vertex"])
        np.testing.assert_array_equal(
            required_tag_mask(evaluated, failed, ["gamma.fiducial"]),
            [True, False, True],
        )

    def test_gaussian_sideband_fit_recovers_injected_peak(self) -> None:
        rng = np.random.default_rng(20260819)
        masses = np.concatenate(
            (rng.normal(0.135, 0.009, 12000), rng.uniform(0.04, 0.22, 3000))
        )
        args = SimpleNamespace(
            peak_nsigma=3.0,
            minimum_events=100,
            fit_window_sigma=5.0,
            fit_max_iterations=100,
            fit_convergence=1.0e-5,
            fit_bins=120,
            maximum_center_deviation=0.04,
            maximum_width=0.04,
            no_continuous_refinement=False,
        )
        estimate, reason = fit_pi0(masses, args)
        self.assertEqual(reason, "")
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.center, 0.135, delta=0.001)
        self.assertAlmostEqual(estimate.sigma, 0.009, delta=0.001)

        row = metric_row(
            "data",
            "FD/FD",
            WorkingPoint("test", 0.6, 0.4),
            masses.size,
            masses.size,
            estimate,
            reason,
        )
        self.assertEqual(row["fit_status"], "ok")
        self.assertTrue(0.0 < row["background_fraction_peak"] < 1.0)
        self.assertTrue(row["signal_over_sqrt_signal_plus_background"] > 0.0)
        self.assertTrue(math.isfinite(row["sideband_subtracted_signal_peak"]))


if __name__ == "__main__":
    unittest.main()
