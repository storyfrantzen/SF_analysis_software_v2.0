from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

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
    mode_seeded_core,
    phi_slice_line_parameters,
    plot_diagnostics,
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

    def test_selection_rejects_large_positive_missing_energy(self) -> None:
        beam_energy = 6.535
        theta_e = np.deg2rad(np.array([20.0, 20.0]))
        theta_p = elastic_proton_theta_from_electron(theta_e, beam_energy)
        electron_p = elastic_electron_momentum(theta_e, beam_energy)
        arrays = {
            "electronP": np.array([electron_p[0], 0.5 * electron_p[1]]),
            "electronTheta": theta_e,
            "electronPhi": np.zeros(2),
            "electronDet": np.ones(2, dtype=int),
            "electronSector": np.ones(2, dtype=int),
            "protonP": elastic_proton_momentum(theta_p, beam_energy),
            "protonTheta": theta_p,
            "protonPhi": np.full(2, np.pi),
            "protonDet": np.ones(2, dtype=int),
            "protonSector": np.full(2, 4, dtype=int),
            "nPid11": np.ones(2, dtype=int),
            "nPid2212": np.ones(2, dtype=int),
        }
        selected, summary = select_elastic_events(
            arrays,
            ElasticFitConfig(
                beam_energy=beam_energy,
                missing_energy_max_gev=0.75,
            ),
        )
        self.assertEqual(summary["angularSelectedCandidates"], 2)
        self.assertEqual(summary["selectedCandidates"], 1)
        self.assertLessEqual(float(selected["missingEnergyGeV"][0]), 0.75)

    def test_mode_seeded_core_finds_peak_below_majority_background(self) -> None:
        rng = np.random.default_rng(17)
        signal = rng.normal(0.012, 0.006, 2000)
        background = rng.uniform(-0.10, 0.10, 4000)
        estimate = mode_seeded_core(
            np.concatenate((signal, background)),
            peak_search_max_abs_residual=0.10,
            peak_seed_half_width=0.03,
            sigma_clip=3.0,
        )
        self.assertAlmostEqual(estimate.center, 0.012, delta=7.5e-4)
        self.assertLess(estimate.width, 0.01)
        self.assertGreater(estimate.peak_significance, 20.0)
        self.assertGreater(estimate.retained_fraction, 0.40)
        self.assertLess(estimate.retained_fraction, 0.60)


class ElasticSurfaceFitTests(unittest.TestCase):
    def test_secondary_slice_line_recovers_intercept_and_slope(self) -> None:
        cells = [
            {"phiMeanDeg": phi, "center": 0.004 + 0.002 * phi / 30.0,
             "centerError": 0.001}
            for phi in (-20.0, 0.0, 20.0)
        ]
        result = phi_slice_line_parameters(cells, 30.0)
        self.assertIsNotNone(result)
        coefficients, errors = result
        np.testing.assert_allclose(coefficients, (0.004, 0.002), atol=1.0e-12)
        self.assertTrue(np.all(np.isfinite(errors) & (errors > 0.0)))

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib is required for elastic-plot diagnostics",
    )
    def test_linear_fd_diagnostics_render_all_profile_views(self) -> None:
        rng = np.random.default_rng(29)
        entries = 8000
        theta = rng.uniform(10.0, 30.0, entries)
        phi = rng.uniform(-29.0, 29.0, entries)
        residual = 0.004 + 0.002 * phi / 30.0 + rng.normal(0.0, 0.008, entries)
        region, diagnostic = fit_region(
            pid=11, detector=1, sector=1, theta_deg=theta, phi_deg=phi,
            residual=residual, basis="polynomial",
            cfg=ElasticFitConfig(
                beam_energy=6.535, torus=1, theta_bins=3, phi_bins=3,
                theta_order=1, phi_order=1, min_bin_entries=40,
                min_region_entries=200,
            ),
        )
        accepted = region["fit"]["acceptedProfileCells"]
        self.assertTrue(all(cell["afterCenter"] is not None for cell in accepted))
        self.assertTrue(all("centerMinusSurface" in cell for cell in accepted))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            plot_diagnostics([diagnostic], output_dir, "plot-test", 6.535)
            for name in (
                "pid11_det1_sector1.png",
                "pid11_det1_sector1_profile_cell_map.png",
                "pid11_det1_sector1_profile_vs_phi_by_theta.png",
                "pid11_det1_sector1_cell_fit_discrepancy.png",
                "pid11_det1_sector1_phi_coefficients_vs_theta.png",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_adaptive_profile_cells_limit_theta_width_and_follow_acceptance(self) -> None:
        rng = np.random.default_rng(37)
        dense_theta = rng.uniform(5.6, 6.6, 12_000)
        tail_theta = 6.6 + 4.0 * rng.beta(1.0, 4.0, 4_000)
        theta = np.concatenate((dense_theta, tail_theta))
        phi_half_width = np.where(theta < 6.6, 12.0, 24.0)
        phi = rng.uniform(-phi_half_width, phi_half_width)
        residual = (
            0.004 + 0.002 * (theta - 7.0) / 3.0
            + 0.001 * phi / 30.0 + rng.normal(0.0, 0.008, theta.size)
        )
        region, _ = fit_region(
            pid=11, detector=1, sector=1, theta_deg=theta, phi_deg=phi,
            residual=residual, basis="polynomial",
            cfg=ElasticFitConfig(
                beam_energy=10.604, torus=1, theta_bins=5, phi_bins=7,
                profile_binning="adaptive", max_theta_bin_width_deg=0.5,
                target_cell_entries=500, theta_order=1, phi_order=1,
                min_bin_entries=100, min_region_entries=1000,
            ),
        )
        binning = region["fit"]["profileBinning"]
        theta_edges = np.asarray(binning["thetaEdgesDeg"], dtype=float)
        self.assertEqual(binning["mode"], "adaptive")
        self.assertGreater(binning["actualThetaBins"], 5)
        self.assertLessEqual(float(np.max(np.diff(theta_edges))), 0.5 + 1.0e-12)
        planned_phi_counts = {
            item["plannedPhiCells"] for item in binning["thetaSlices"]
        }
        self.assertGreater(len(planned_phi_counts), 1)
        self.assertTrue(all(
            min(item["phiEdgesDeg"]) > -30.0
            and max(item["phiEdgesDeg"]) < 30.0
            for item in binning["thetaSlices"] if item["plannedPhiCells"]
        ))
        self.assertTrue(all(
            cell["coreEntries"] >= 100
            for cell in region["fit"]["acceptedProfileCells"]
        ))
        self.assertTrue(all(
            item["plannedPhiCells"] >= 2
            for item in binning["thetaSlices"]
            if item["rawEntries"] >= 200
        ))

    def test_explicit_theta_floor_and_surface_guard_are_enforced(self) -> None:
        rng = np.random.default_rng(43)
        theta = rng.uniform(5.5, 9.0, 20_000)
        phi = rng.uniform(-29.0, 29.0, theta.size)
        residual = 0.004 + rng.normal(0.0, 0.008, theta.size)
        region, _ = fit_region(
            pid=11, detector=1, sector=1, theta_deg=theta, phi_deg=phi,
            residual=residual, basis="polynomial",
            cfg=ElasticFitConfig(
                beam_energy=10.604, theta_min_deg=6.1,
                theta_bins=4, phi_bins=3, theta_order=1, phi_order=1,
                min_bin_entries=100, min_region_entries=1_000,
            ),
        )
        self.assertGreaterEqual(region["thetaRangeDeg"][0], 6.1)
        self.assertEqual(len(region["fit"]["weightedDesignSingularValues"]), 4)
        with self.assertRaisesRegex(ValueError, "support grid"):
            fit_region(
                pid=11, detector=1, sector=1, theta_deg=theta, phi_deg=phi,
                residual=0.08 + rng.normal(0.0, 0.005, theta.size),
                basis="polynomial",
                cfg=ElasticFitConfig(
                    beam_energy=10.604, theta_min_deg=6.1,
                    theta_bins=4, phi_bins=3, theta_order=0, phi_order=0,
                    min_bin_entries=100, min_region_entries=1_000,
                    max_abs_surface_correction=0.05,
                ),
            )

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
        self.assertTrue(all(region["supportCells"] for region in correction["regions"]))
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
