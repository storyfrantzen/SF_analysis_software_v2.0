from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.proton_energy_loss import DEFAULT_CONFIGS, residual_range_mask


class ResidualSelectionTests(unittest.TestCase):
    def test_each_fit_uses_only_its_own_residual_range(self) -> None:
        arrays = {
            "delta_p_fit": np.array([0.0, 0.1, 0.3]),
            "delta_theta_fit": np.array([0.0, 0.3, 0.1]),
            "delta_phi_fit": np.array([0.0, 0.1, 0.3]),
        }
        cfg = DEFAULT_CONFIGS["CD"]

        np.testing.assert_array_equal(
            residual_range_mask(arrays, "delta_p", cfg), [True, True, False]
        )
        np.testing.assert_array_equal(
            residual_range_mask(arrays, "delta_theta", cfg), [True, False, True]
        )
        np.testing.assert_array_equal(
            residual_range_mask(arrays, "delta_phi", cfg), [True, True, False]
        )


if __name__ == "__main__":
    unittest.main()
