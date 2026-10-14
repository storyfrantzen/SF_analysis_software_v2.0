from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.eppi0_momentum_validation import (
    ELECTRON_MASS_GEV,
    PI0_MASS_GEV,
    PROTON_MASS_GEV,
    _four_vector,
    _m2,
    _unit_vectors,
)
from scripts.calibration.exclusive_particle_momentum import (
    ExclusiveFitConfig,
    expected_photon_energies_eppi0,
    expected_proton_momentum_eppi0,
    fit_momentum_theta_region,
    proton_momentum_roots_eppi0,
)
from scripts.calibration.elastic_momentum import apply_supported_particle_correction


class ExclusiveKinematicEstimatorTests(unittest.TestCase):
    def test_proton_solution_closes_pi0_missing_mass(self) -> None:
        beam_energy = 6.535
        electron_p = np.array([4.2, 3.6, 4.8])
        electron_theta = np.deg2rad([18.0, 22.0, 14.0])
        electron_phi = np.deg2rad([10.0, -25.0, 35.0])
        proton_theta = np.deg2rad([32.0, 38.0, 27.0])
        proton_phi = electron_phi + np.deg2rad([165.0, 175.0, 170.0])
        expected, _ = expected_proton_momentum_eppi0(
            electron_p, electron_theta, electron_phi,
            proton_theta, proton_phi, beam_energy,
            measured_proton_momentum=np.ones(3),
        )
        valid = np.isfinite(expected)
        self.assertTrue(np.any(valid))
        electron = _four_vector(
            electron_p[valid], electron_theta[valid], electron_phi[valid],
            ELECTRON_MASS_GEV,
        )
        proton = _four_vector(
            expected[valid], proton_theta[valid], proton_phi[valid],
            PROTON_MASS_GEV,
        )
        missing = np.zeros_like(electron)
        missing[:, 0] = beam_energy + PROTON_MASS_GEV
        missing[:, 3] = beam_energy
        missing -= electron + proton
        np.testing.assert_allclose(
            _m2(missing), PI0_MASS_GEV**2, rtol=0.0, atol=2.0e-12
        )

    def test_proton_root_helper_matches_selected_solution(self) -> None:
        beam_energy = 6.535
        electron_p = np.array([4.2, 3.6, 4.8])
        electron_theta = np.deg2rad([18.0, 22.0, 14.0])
        electron_phi = np.deg2rad([10.0, -25.0, 35.0])
        proton_theta = np.deg2rad([32.0, 38.0, 27.0])
        proton_phi = electron_phi + np.deg2rad([165.0, 175.0, 170.0])
        measured = np.ones(3)
        roots, valid = proton_momentum_roots_eppi0(
            electron_p, electron_theta, electron_phi,
            proton_theta, proton_phi, beam_energy,
        )
        selected, ambiguous = expected_proton_momentum_eppi0(
            electron_p, electron_theta, electron_phi,
            proton_theta, proton_phi, beam_energy, measured,
        )
        choice = np.argmin(np.where(valid, np.abs(roots - measured[:, None]), np.inf), axis=1)
        expected = roots[np.arange(roots.shape[0]), choice]
        expected[~np.any(valid, axis=1)] = np.nan
        np.testing.assert_allclose(selected, expected, equal_nan=True)
        np.testing.assert_array_equal(ambiguous, np.sum(valid, axis=1) > 1)

    def test_photon_energy_solution_recovers_exact_four_vector(self) -> None:
        theta1 = np.deg2rad(np.array([12.0, 20.0]))
        phi1 = np.deg2rad(np.array([15.0, -35.0]))
        theta2 = np.deg2rad(np.array([28.0, 35.0]))
        phi2 = np.deg2rad(np.array([80.0, 40.0]))
        energy1 = np.array([1.2, 0.8])
        energy2 = np.array([0.7, 1.1])
        n1 = _unit_vectors(theta1, phi1)
        n2 = _unit_vectors(theta2, phi2)
        target = np.column_stack((
            energy1 + energy2,
            energy1[:, None] * n1 + energy2[:, None] * n2,
        ))
        fitted1, fitted2, residual, condition = expected_photon_energies_eppi0(
            target, theta1, phi1, theta2, phi2
        )
        np.testing.assert_allclose(fitted1, energy1, atol=1.0e-12)
        np.testing.assert_allclose(fitted2, energy2, atol=1.0e-12)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)
        self.assertTrue(np.all(np.isfinite(condition)))

    def test_generic_application_respects_momentum_theta_cells(self) -> None:
        parameters = {
            "schema": "particle_momentum_correction/v2",
            "correctionType": "fractionalMomentum",
            "beamEnergyGeV": 6.535,
            "torus": 1,
            "regions": [{
                "pid": 22, "detector": 0, "sector": 0,
                "momentumRangeGeV": [0.5, 2.0],
                "momentumCenterGeV": 1.25, "momentumScaleGeV": 0.75,
                "thetaRangeDeg": [2.0, 5.0],
                "thetaCenterDeg": 3.5, "thetaScaleDeg": 1.5,
                "phiRangeDeg": [-180.0, 180.0],
                "phiCenterDeg": 0.0, "phiScaleDeg": 180.0,
                "phiVariable": "global", "basis": "polynomial",
                "terms": [{
                    "momentumPower": 0, "thetaPower": 0, "phiPower": 0,
                    "coefficient": 0.05,
                }],
                "supportCells": [{
                    "momentumRangeGeV": [0.75, 1.5],
                    "thetaRangeDeg": [2.5, 4.5],
                    "phiRangeDeg": [-180.0, 180.0],
                }],
            }],
        }
        corrected, support, correction = apply_supported_particle_correction(
            np.array([1.0, 1.8]), np.deg2rad([3.0, 3.0]), np.zeros(2),
            np.zeros(2, dtype=int), np.zeros(2, dtype=int),
            pid=22, parameters=parameters,
        )
        np.testing.assert_array_equal(support, [True, False])
        np.testing.assert_allclose(correction, [0.05, 0.0])
        np.testing.assert_allclose(corrected, [1.05, 1.8])


class ExclusiveSurfaceFitTests(unittest.TestCase):
    def test_bilinear_momentum_theta_surface_is_recovered(self) -> None:
        rng = np.random.default_rng(43)
        momentum = rng.uniform(0.5, 2.0, 80_000)
        theta = rng.uniform(8.0, 28.0, momentum.size)
        pn = (momentum - 1.25) / 0.75
        tn = (theta - 18.0) / 10.0
        residual = (
            0.02 + 0.01 * pn - 0.015 * tn + 0.008 * pn * tn
            + rng.normal(0.0, 0.01, momentum.size)
        )
        region, diagnostic = fit_momentum_theta_region(
            pid=2212, particle="proton", detector=2, sector=0,
            momentum=momentum, theta_deg=theta, phi_deg=np.zeros(momentum.size),
            residual=residual,
            cfg=ExclusiveFitConfig(
                beam_energy=6.535, torus=1, momentum_bins=8, theta_bins=8,
                min_bin_entries=300, min_region_entries=2_000,
            ),
        )
        self.assertEqual(region["fit"]["model"], "momentum-theta")
        self.assertGreater(region["fit"]["profileCells"], 40)
        before = region["fit"]["beforeSupported"]["rms"]
        after = region["fit"]["afterSupported"]["rms"]
        self.assertLess(after, 0.55 * before)
        self.assertTrue(np.any(diagnostic["support"]))


if __name__ == "__main__":
    unittest.main()
