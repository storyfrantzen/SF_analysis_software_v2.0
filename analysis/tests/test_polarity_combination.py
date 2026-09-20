from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.polarity_combination import combine_polarity_measurements
from eppi0.binning import AnalysisBinning
from combine_polarity_cross_sections import coordinate_validity_mask


class PolarityCombinationTests(unittest.TestCase):
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
