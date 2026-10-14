from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.polarity_combination import combine_polarity_measurements
from eppi0.binning import AnalysisBinning
from combine_polarity_cross_sections import (
    combine_phi_covariance,
    coordinate_validity_mask,
)


class PolarityCombinationTests(unittest.TestCase):
    def test_phi_covariance_follows_blue_weights_and_diagonal(self) -> None:
        left_values = np.asarray([[[[10.0, 20.0]]]])
        right_values = np.asarray([[[[14.0, 18.0]]]])
        left_uncertainty = np.asarray([[[[2.0, 3.0]]]])
        right_uncertainty = np.asarray([[[[4.0, 5.0]]]])
        valid = np.ones_like(left_values, dtype=bool)
        result = combine_polarity_measurements(
            left_values,
            left_uncertainty,
            valid,
            right_values,
            right_uncertainty,
            valid,
            shared_relative_uncertainty=0.02,
        )
        left_covariance = np.asarray([[[[[4.0, 1.0], [1.0, 9.0]]]]])
        right_covariance = np.asarray([[[[[16.0, 2.0], [2.0, 25.0]]]]])

        combined = combine_phi_covariance(
            left_covariance,
            right_covariance,
            left_weight=result.left_weight,
            right_weight=result.right_weight,
            combined_uncertainty=result.uncertainties,
            combined_valid=result.valid,
        )

        expected_off_diagonal = (
            result.left_weight[0, 0, 0, 0]
            * result.left_weight[0, 0, 0, 1]
            * left_covariance[0, 0, 0, 0, 1]
            + result.right_weight[0, 0, 0, 0]
            * result.right_weight[0, 0, 0, 1]
            * right_covariance[0, 0, 0, 0, 1]
        )
        self.assertAlmostEqual(combined[0, 0, 0, 0, 1], expected_off_diagonal)
        self.assertAlmostEqual(combined[0, 0, 0, 1, 0], expected_off_diagonal)
        np.testing.assert_allclose(
            np.diagonal(combined, axis1=-2, axis2=-1),
            result.uncertainties**2,
        )

    def test_flat_coordinate_mask_uses_legacy_bin_order(self) -> None:
        binning = AnalysisBinning(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [0.0, 1.0],
            [0.0, 180.0, 360.0],
        )
        mask = np.zeros(binning.shape, dtype=bool)
        mask[0, 1, 0, 0] = True
        flattened = coordinate_validity_mask(binning, mask, (binning.size,))

        expected = np.zeros(binning.size, dtype=bool)
        expected[int(binning.flatten(0, 1, 0, 0))] = True
        np.testing.assert_array_equal(flattened, expected)

    def test_uncorrelated_blue_and_single_source_bins(self) -> None:
        result = combine_polarity_measurements(
            np.asarray([10.0, 5.0, np.nan]),
            np.asarray([2.0, 1.0, np.nan]),
            np.asarray([True, True, False]),
            np.asarray([14.0, np.nan, 8.0]),
            np.asarray([4.0, np.nan, 2.0]),
            np.asarray([True, False, True]),
        )

        self.assertAlmostEqual(result.values[0], 10.8)
        self.assertAlmostEqual(result.uncertainties[0], np.sqrt(3.2))
        self.assertAlmostEqual(result.left_weight[0], 0.8)
        self.assertAlmostEqual(result.right_weight[0], 0.2)
        self.assertEqual(result.values[1], 5.0)
        self.assertEqual(result.uncertainties[1], 1.0)
        self.assertEqual(result.values[2], 8.0)
        self.assertEqual(result.uncertainties[2], 2.0)
        np.testing.assert_array_equal(result.contributor_count, [2, 1, 1])

    def test_shared_fractional_uncertainty_is_not_averaged_away(self) -> None:
        shared_fraction = 0.1
        independent_sigma = 2.0
        total_sigma = np.hypot(independent_sigma, shared_fraction * 10.0)
        result = combine_polarity_measurements(
            np.asarray([10.0]),
            np.asarray([total_sigma]),
            np.asarray([True]),
            np.asarray([10.0]),
            np.asarray([total_sigma]),
            np.asarray([True]),
            shared_relative_uncertainty=shared_fraction,
        )

        expected = np.hypot(independent_sigma / np.sqrt(2.0), 1.0)
        self.assertAlmostEqual(result.values[0], 10.0)
        self.assertAlmostEqual(result.uncertainties[0], expected)
        self.assertGreater(result.uncertainties[0], total_sigma / np.sqrt(2.0))
        self.assertEqual(result.overlap_pull[0], 0.0)


if __name__ == "__main__":
    unittest.main()
