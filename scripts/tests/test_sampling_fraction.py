from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.sampling_fraction import (
    SamplingFractionConfig,
    derive_sf_calibration,
    fit_gaussian_core,
)


class GaussianCoreFitTests(unittest.TestCase):
    def test_core_fit_is_stable_against_broad_low_side_tail(self) -> None:
        rng = np.random.default_rng(17)
        core_entries = 120_000
        tail_entries = 24_000
        core_mu = 0.245
        core_sigma = 0.012
        momentum = rng.uniform(1.0, 1.2, core_entries + tail_entries)
        sampling_fraction = np.concatenate(
            [
                rng.normal(core_mu, core_sigma, core_entries),
                rng.normal(0.19, 0.05, tail_entries),
            ]
        )

        fit = fit_gaussian_core(
            momentum,
            sampling_fraction,
            SamplingFractionConfig(min_entries=100),
        )

        self.assertTrue(fit.valid)
        self.assertAlmostEqual(fit.mu, core_mu, delta=7.0e-4)
        self.assertAlmostEqual(fit.sigma, core_sigma, delta=1.2e-3)
        self.assertGreater(fit.core_fraction, 0.8)
        self.assertLess(fit.sigma, np.std(sampling_fraction, ddof=1) / 2.0)

    def test_profile_uses_observed_core_momentum_mean(self) -> None:
        rng = np.random.default_rng(23)
        momentum = 1.0 + 0.2 * rng.beta(1.0, 5.0, 50_000)
        sampling_fraction = rng.normal(0.24, 0.01, momentum.size)
        fit = fit_gaussian_core(
            momentum,
            sampling_fraction,
            SamplingFractionConfig(min_entries=100),
        )

        self.assertTrue(fit.valid)
        self.assertLess(fit.momentum_mean, 1.06)
        self.assertGreater(abs(fit.momentum_mean - 1.1), 0.03)


class SamplingFractionCalibrationTests(unittest.TestCase):
    @staticmethod
    def sample() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(31)
        entries_per_sector = 30_000
        sector = np.repeat(np.arange(1, 7), entries_per_sector)
        momentum = rng.uniform(1.0, 4.3, sector.size)
        sector_shift = 5.0e-4 * (sector - 3.5)
        mu = 0.232 + 0.009 * momentum - 0.0015 * momentum * momentum + sector_shift
        sigma = 0.011 + 0.003 / momentum
        sampling_fraction = rng.normal(mu, sigma)
        return {
            "sf_p": momentum,
            "sf_sector": sector,
            "sampling_fraction": sampling_fraction,
            "sf_epcal": 0.17 * momentum,
            "sf_ecin": 0.08 * momentum,
        }

    def test_data_fit_is_sector_dependent_and_positive(self) -> None:
        cfg = SamplingFractionConfig(
            momentum_range=(1.0, 4.3),
            momentum_bins=16,
            min_entries=100,
        )
        calibration = derive_sf_calibration(self.sample(), cfg, sector_independent=False)

        self.assertEqual(set(calibration.parameter_profiles), {
            f"sector_{sec}" for sec in range(1, 7)
        })
        self.assertNotEqual(
            calibration.coefficients["sector_1"]["mu_coeffs"],
            calibration.coefficients["sector_6"]["mu_coeffs"],
        )
        grid = np.linspace(*cfg.momentum_range, 200)
        for parameters in calibration.coefficients.values():
            self.assertTrue(np.all(np.polyval(parameters["sigma_coeffs"], grid) > 0))

    def test_gemc_global_parameters_are_copied_but_sector_profiles_remain_distinct(self) -> None:
        cfg = SamplingFractionConfig(
            momentum_range=(1.0, 4.3),
            momentum_bins=16,
            min_entries=100,
        )
        calibration = derive_sf_calibration(self.sample(), cfg, sector_independent=True)

        self.assertEqual(set(calibration.parameter_profiles), {"global"})
        reference = calibration.coefficients["sector_1"]
        for sec in range(2, 7):
            self.assertEqual(calibration.coefficients[f"sector_{sec}"], reference)
        sector_1_mu = calibration.sector_profiles["sector_1"].mu_values
        sector_6_mu = calibration.sector_profiles["sector_6"].mu_values
        self.assertGreater(float(np.nanmean(sector_6_mu - sector_1_mu)), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
