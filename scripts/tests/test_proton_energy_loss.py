from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.proton_energy_loss import (
    DEFAULT_CONFIGS,
    quantile_residual_range,
    quantile_theta_range,
    residual_range_mask,
    resolve_residual_range,
    resolve_theta_range,
    with_theta_caps,
)


class ResidualSelectionTests(unittest.TestCase):
    def test_fixed_mode_uses_only_its_own_residual_range(self) -> None:
        arrays = {
            "delta_p_fit": np.array([0.0, 0.1, 0.3]),
            "delta_theta_fit": np.array([0.0, 0.3, 0.1]),
            "delta_phi_fit": np.array([0.0, 0.1, 0.3]),
        }
        cfg = DEFAULT_CONFIGS["CD"]

        np.testing.assert_array_equal(
            residual_range_mask(
                arrays,
                "delta_p",
                resolve_residual_range(arrays, "delta_p", cfg, "fixed", 0.01),
            ),
            [True, True, False],
        )
        np.testing.assert_array_equal(
            residual_range_mask(
                arrays,
                "delta_theta",
                resolve_residual_range(arrays, "delta_theta", cfg, "fixed", 0.01),
            ),
            [True, False, True],
        )
        np.testing.assert_array_equal(
            residual_range_mask(
                arrays,
                "delta_phi",
                resolve_residual_range(arrays, "delta_phi", cfg, "fixed", 0.01),
            ),
            [True, True, False],
        )

    def test_quantile_mode_derives_sample_range(self) -> None:
        arrays = {
            "delta_p_fit": np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
            "delta_theta_fit": np.array([0.0]),
            "delta_phi_fit": np.array([0.0]),
        }

        lo, hi = quantile_residual_range(arrays, "delta_p", 0.25)
        self.assertAlmostEqual(lo, -1.0)
        self.assertAlmostEqual(hi, 1.0)

        np.testing.assert_array_equal(
            residual_range_mask(arrays, "delta_p", (lo, hi)),
            [False, True, True, True, False],
        )

    def test_quantile_mode_rejects_invalid_trim_fraction(self) -> None:
        arrays = {
            "delta_p_fit": np.array([0.0, 1.0]),
            "delta_theta_fit": np.array([0.0, 1.0]),
            "delta_phi_fit": np.array([0.0, 1.0]),
        }

        with self.assertRaises(ValueError):
            quantile_residual_range(arrays, "delta_p", 0.5)


class ThetaRangeSelectionTests(unittest.TestCase):
    def test_quantile_mode_derives_theta_range_from_sample(self) -> None:
        arrays = {
            "theta_deg": np.array([40.0, 45.0, 60.0, 90.0, 125.0]),
        }

        lo, hi = quantile_theta_range(arrays, 0.25)
        self.assertAlmostEqual(lo, 45.0)
        self.assertAlmostEqual(hi, 90.0)

    def test_fixed_mode_preserves_historical_cd_range(self) -> None:
        cfg = DEFAULT_CONFIGS["CD"]
        arrays = {
            "theta_deg": np.array([40.0, 60.0, 100.0, 125.0]),
        }

        self.assertEqual(resolve_theta_range(arrays, cfg, "fixed", 0.001), (40.0, 58.0))

    def test_cd_default_caps_extend_beyond_historical_upper_edge(self) -> None:
        cfg = DEFAULT_CONFIGS["CD"]

        self.assertEqual(cfg.theta_caps, (40.0, 125.0))
        self.assertEqual(cfg.theta_range, (40.0, 58.0))

    def test_theta_cap_override_rejects_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            with_theta_caps(DEFAULT_CONFIGS["CD"], (70.0, 50.0))


if __name__ == "__main__":
    unittest.main()
