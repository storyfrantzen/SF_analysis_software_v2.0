from __future__ import annotations

import unittest

import numpy as np

from eppi0.event_kinematics_comparison import (
    common_variables,
    comparison_range,
    shape_metrics,
)
from eppi0.event_kinematics_diagnostics import PlotVariable


class EventKinematicsComparisonTests(unittest.TestCase):
    def test_identical_shapes_have_zero_distance(self) -> None:
        js, total_variation = shape_metrics(
            np.array([1, 3, 8, 2]), np.array([2, 6, 16, 4])
        )
        self.assertAlmostEqual(js, 0.0)
        self.assertAlmostEqual(total_variation, 0.0)

    def test_disjoint_shapes_reach_maximum_distance(self) -> None:
        js, total_variation = shape_metrics(
            np.array([4, 0, 0]), np.array([0, 0, 9])
        )
        self.assertAlmostEqual(js, 1.0)
        self.assertAlmostEqual(total_variation, 1.0)

    def test_empty_histogram_reports_unavailable_distance(self) -> None:
        js, total_variation = shape_metrics(np.zeros(3), np.ones(3))
        self.assertTrue(np.isnan(js))
        self.assertTrue(np.isnan(total_variation))

    def test_only_variables_available_in_both_samples_are_compared(self) -> None:
        data = {"Q2": np.ones(2), "xB": np.ones(2)}
        gemc = {"Q2": np.ones(3), "W": np.ones(3)}
        self.assertEqual([item.branch for item in common_variables(data, gemc)], ["Q2"])

    def test_comparison_range_gives_each_sample_equal_tail_protection(self) -> None:
        variable = PlotVariable("x", "x", "x", "test")
        data_values = np.append(np.linspace(-8.0, -6.0, 1000), 1000.0)
        gemc_values = np.append(np.linspace(5.0, 8.0, 1000), 2000.0)
        data = {"x": data_values}
        gemc = {"x": gemc_values}
        low, high = comparison_range(
            variable,
            data,
            gemc,
            np.ones(data_values.size, dtype=bool),
            np.ones(gemc_values.size, dtype=bool),
        )
        self.assertLess(low, -7.0)
        self.assertGreater(high, 8.0)
        self.assertLess(high, 2000.0)


if __name__ == "__main__":
    unittest.main()
