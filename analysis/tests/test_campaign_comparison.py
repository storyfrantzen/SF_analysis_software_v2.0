from __future__ import annotations

import unittest

import numpy as np

from eppi0.campaign_comparison import common_validity_masks


class CommonValidityMaskTests(unittest.TestCase):
    def test_intersects_only_exactly_matching_q2_bins(self) -> None:
        left = np.asarray(
            [
                [[[(True), False]]],
                [[[(True), True]]],
            ],
            dtype=bool,
        )
        right = np.asarray(
            [
                [[[(True), True]]],
                [[[(False), True]]],
                [[[(True), True]]],
            ],
            dtype=bool,
        )
        result = common_validity_masks(
            left,
            right,
            left_q2_edges=[1.0, 2.0, 3.0],
            right_q2_edges=[1.0, 2.0, 3.0, 4.0],
            left_xb_edges=[0.1, 0.2],
            right_xb_edges=[0.1, 0.2],
            left_t_edges=[0.0, 0.5],
            right_t_edges=[0.0, 0.5],
            left_phi_edges=[0.0, 180.0, 360.0],
            right_phi_edges=[0.0, 180.0, 360.0],
        )
        np.testing.assert_array_equal(result.left_q2_indices, [0, 1])
        np.testing.assert_array_equal(result.right_q2_indices, [0, 1])
        np.testing.assert_array_equal(
            result.left,
            np.asarray([[[[True, False]]], [[[False, True]]]], dtype=bool),
        )
        np.testing.assert_array_equal(
            result.right,
            np.asarray(
                [[[[True, False]]], [[[False, True]]], [[[False, False]]]],
                dtype=bool,
            ),
        )

    def test_rejects_incompatible_non_q2_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible xB"):
            common_validity_masks(
                np.ones((1, 1, 1, 1), dtype=bool),
                np.ones((1, 1, 1, 1), dtype=bool),
                left_q2_edges=[1.0, 2.0],
                right_q2_edges=[1.0, 2.0],
                left_xb_edges=[0.1, 0.2],
                right_xb_edges=[0.1, 0.3],
                left_t_edges=[0.0, 0.5],
                right_t_edges=[0.0, 0.5],
                left_phi_edges=[0.0, 360.0],
                right_phi_edges=[0.0, 360.0],
            )


if __name__ == "__main__":
    unittest.main()
