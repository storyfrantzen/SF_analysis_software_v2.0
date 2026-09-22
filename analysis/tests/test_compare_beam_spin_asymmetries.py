from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "compare_beam_spin_asymmetries.py"
SPEC = importlib.util.spec_from_file_location("compare_beam_spin_asymmetries", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareBeamSpinTest(unittest.TestCase):
    def test_extended_fit_recovers_signal_and_null_harmonics(self) -> None:
        edges = np.arange(0.0, 360.0 + 18.0, 18.0)
        radians = np.deg2rad(edges)
        widths = np.diff(radians)
        expected = np.array([0.03, 0.24, -0.02, 0.01])
        design = np.column_stack(
            [
                np.ones(widths.size),
                (np.cos(radians[:-1]) - np.cos(radians[1:])) / widths,
                (np.sin(radians[1:]) - np.sin(radians[:-1])) / widths,
                (np.cos(2.0 * radians[:-1]) - np.cos(2.0 * radians[1:]))
                / (2.0 * widths),
            ]
        )
        values = (
            design @ expected
        )[None, None, None, :]
        errors = np.full(values.shape, 0.02)
        coefficients, uncertainties, chi2, ndof = MODULE.fit_extended_harmonics(
            values, np.ones(values.shape) * 0.02,
            np.ones(values.shape, dtype=bool), edges,
        )
        np.testing.assert_allclose(coefficients.reshape(-1, 4)[0], expected, atol=1e-12)
        self.assertTrue(np.all(uncertainties > 0.0))
        self.assertAlmostEqual(float(chi2.item()), 0.0, places=12)
        self.assertEqual(int(ndof.item()), 16)

    def test_shared_event_comparison_accepts_different_phi_grids(self) -> None:
        left = {"phi_edges": np.linspace(0.0, 360.0, 21)}
        right = {"phi_edges": np.linspace(0.0, 360.0, 13)}
        left["beam_spin_asymmetry"] = np.zeros((1, 1, 1, 20))
        right["beam_spin_asymmetry"] = np.zeros((1, 1, 1, 12))
        summary = MODULE.compare_phi_points(left, right)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["left_phi_bins"], 20)
        self.assertEqual(summary["right_phi_bins"], 12)

    def test_null_summary_can_restrict_to_production_quality(self) -> None:
        coefficients = np.zeros((3, 4))
        errors = np.ones((3, 4))
        coefficients[:, 0] = [1.0, 8.0, 3.0]
        ndof = np.array([8, 8, 0])
        quality = np.array([True, False, True])
        summary = MODULE.null_summary(
            coefficients, errors, ndof, selection=quality
        )
        self.assertEqual(summary["numerically_successful_bins"], 2)
        self.assertEqual(summary["evaluated_bins"], 1)
        self.assertEqual(summary["constant"]["absolute_pull_gt_2"], 0)


if __name__ == "__main__":
    unittest.main()
