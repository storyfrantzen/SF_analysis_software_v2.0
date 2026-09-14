from __future__ import annotations

import json
import unittest

import numpy as np

from scripts.calibration.elastic_momentum import (
    ElasticFitConfig,
    PROTON_MASS_GEV,
    elastic_electron_momentum,
    elastic_proton_momentum,
    elastic_proton_theta_from_electron,
    derive_corrections,
    evaluate_region,
    fit_region,
    select_elastic_events,
)


class ElasticKinematicsTests(unittest.TestCase):
    def test_two_body_predictions_conserve_energy_and_momentum(self) -> None:
        beam_energy = 6.535
        electron_theta = np.deg2rad(np.array([8.0, 15.0, 25.0, 35.0]))
        electron_p = elastic_electron_momentum(electron_theta, beam_energy)
        proton_theta = elastic_proton_theta_from_electron(electron_theta, beam_energy)
        proton_p = elastic_proton_momentum(proton_theta, beam_energy)

        px = electron_p * np.sin(electron_theta) - proton_p * np.sin(proton_theta)
        pz = electron_p * np.cos(electron_theta) + proton_p * np.cos(proton_theta)
        final_energy = electron_p + np.sqrt(PROTON_MASS_GEV**2 + proton_p**2)

        np.testing.assert_allclose(px, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(pz, beam_energy, atol=1.0e-12)
        np.testing.assert_allclose(
            final_energy, beam_energy + PROTON_MASS_GEV, atol=1.0e-12
        )

    def test_selection_uses_angle_closure_and_multiplicity(self) -> None:
        beam_energy = 6.535
        theta_e = np.deg2rad(np.array([20.0, 20.0, 20.0]))
        theta_p = elastic_proton_theta_from_electron(theta_e, beam_energy)
        arrays = {
            "electronP": elastic_electron_momentum(theta_e, beam_energy),
            "electronTheta": theta_e,
            "electronPhi": np.zeros(3),
            "electronDet": np.ones(3, dtype=int),
            "electronSector": np.ones(3, dtype=int),
            "protonP": elastic_proton_momentum(theta_p, beam_energy),
            "protonTheta": theta_p + np.deg2rad([0.0, 4.0, 0.0]),
            "protonPhi": np.deg2rad([180.0, 180.0, 170.0]),
            "protonDet": np.ones(3, dtype=int),
            "protonSector": np.full(3, 4, dtype=int),
            "nPid11": np.array([1, 1, 2]),
            "nPid2212": np.ones(3, dtype=int),
        }
        selected, summary = select_elastic_events(
            arrays, ElasticFitConfig(beam_energy=beam_energy)
        )
        self.assertEqual(summary["selectedCandidates"], 1)
        self.assertEqual(selected["electronP"].size, 1)


class ElasticSurfaceFitTests(unittest.TestCase):
    def test_exported_fd_regions_are_finite_and_complete(self) -> None:
        rng = np.random.default_rng(61)
        entries_per_sector = 1000
        sector = np.repeat(np.arange(1, 7), entries_per_sector)
        entries = sector.size
        electron_theta = np.deg2rad(rng.uniform(10.0, 30.0, entries))
        local_phi = rng.uniform(-28.0, 28.0, entries)
        electron_phi_deg = local_phi + 60.0 * (sector - 1)
        electron_phi = np.deg2rad((electron_phi_deg + 180.0) % 360.0 - 180.0)
        proton_theta = elastic_proton_theta_from_electron(electron_theta, 6.535)
        proton_phi = electron_phi + np.pi
        expected_electron = elastic_electron_momentum(electron_theta, 6.535)
        fractional_bias = 0.004 + 0.002 * local_phi / 30.0
        arrays = {
            "electronP": expected_electron / (1.0 + fractional_bias),
            "electronTheta": electron_theta,
            "electronPhi": electron_phi,
            "electronDet": np.ones(entries, dtype=int),
            "electronSector": sector,
            "protonP": elastic_proton_momentum(proton_theta, 6.535),
            "protonTheta": proton_theta,
            "protonPhi": proton_phi,
            "protonDet": np.ones(entries, dtype=int),
            "protonSector": np.ones(entries, dtype=int),
            "nPid11": np.ones(entries, dtype=int),
            "nPid2212": np.ones(entries, dtype=int),
        }
        correction, diagnostics = derive_corrections(
            arrays,
            ElasticFitConfig(
                beam_energy=6.535,
                torus=1,
                theta_bins=3,
                phi_bins=3,
                theta_order=1,
                phi_order=1,
                min_bin_entries=10,
                min_region_entries=200,
            ),
            particles=("electron",),
        )
        self.assertEqual(len(correction["regions"]), 6)
        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(correction["torus"], 1)
        json.dumps(correction, allow_nan=False)

    def test_fd_polynomial_surface_is_recovered(self) -> None:
        rng = np.random.default_rng(71)
        entries = 160_000
        theta = rng.uniform(12.0, 36.0, entries)
        phi = rng.uniform(-29.0, 29.0, entries)
        theta_normalized = (theta - 24.0) / 12.0
        phi_normalized = phi / 30.0
        truth = (
            0.006
            + 0.004 * theta_normalized
            - 0.003 * phi_normalized
            + 0.002 * theta_normalized * phi_normalized
            + 0.001 * phi_normalized**2
        )
        residual = truth + rng.normal(0.0, 0.008, entries)
        cfg = ElasticFitConfig(
            beam_energy=6.535,
            theta_bins=7,
            phi_bins=7,
            theta_order=2,
            phi_order=2,
            min_bin_entries=100,
            min_region_entries=1000,
        )
        region, diagnostics = fit_region(
            pid=11,
            detector=1,
            sector=1,
            theta_deg=theta,
            phi_deg=phi,
            residual=residual,
            basis="polynomial",
            cfg=cfg,
        )
        predicted = evaluate_region(region, theta, phi)
        self.assertLess(float(np.sqrt(np.mean(np.square(predicted - truth)))), 5.0e-4)
        self.assertLess(
            abs(float(np.mean(diagnostics.residual_after))),
            abs(float(np.mean(diagnostics.residual_before))) / 10.0,
        )

    def test_cd_fourier_surface_is_recovered(self) -> None:
        rng = np.random.default_rng(83)
        entries = 180_000
        theta = rng.uniform(38.0, 72.0, entries)
        phi = rng.uniform(-180.0, 180.0, entries)
        theta_normalized = (theta - 55.0) / 17.0
        truth = (
            0.005
            + 0.003 * theta_normalized
            + 0.004 * np.cos(np.deg2rad(2.0 * phi))
            - 0.002 * theta_normalized * np.sin(np.deg2rad(phi))
        )
        residual = truth + rng.normal(0.0, 0.01, entries)
        cfg = ElasticFitConfig(
            beam_energy=6.535,
            theta_bins=8,
            phi_bins=12,
            theta_order=2,
            cd_fourier_harmonics=3,
            min_bin_entries=100,
            min_region_entries=1000,
        )
        region, _ = fit_region(
            pid=2212,
            detector=2,
            sector=0,
            theta_deg=theta,
            phi_deg=phi,
            residual=residual,
            basis="fourier",
            cfg=cfg,
        )
        predicted = evaluate_region(region, theta, phi)
        self.assertLess(float(np.sqrt(np.mean(np.square(predicted - truth)))), 7.0e-4)


if __name__ == "__main__":
    unittest.main()
