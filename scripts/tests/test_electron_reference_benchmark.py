from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.electron_reference_benchmark import (
    YIJIE_JOSH_RGK_6535_COEFFICIENTS,
    reference_sector_phi,
    run_elastic_benchmark,
    sector_continuous_reference_phi,
    signed_reference_phi,
    unfold_reference_phi,
    yijie_josh_rgk_6535_delta_p,
)
from scripts.calibration.elastic_momentum import ElasticFitConfig, elastic_electron_momentum
from scripts.calibration.elastic_run_validation import RunBlock


class ElectronReferenceBenchmarkTests(unittest.TestCase):
    def test_reference_signed_phi_keeps_sectors_five_and_six_negative(self) -> None:
        wrapped = np.deg2rad(np.asarray([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
        np.testing.assert_allclose(
            signed_reference_phi(wrapped),
            [0.0, 60.0, 120.0, -180.0, -120.0, -60.0],
            atol=1.0e-12,
        )

    def test_reference_phi_is_continuous_through_sector_four(self) -> None:
        phi = np.deg2rad(np.asarray([179.0, -179.0, -120.0, -60.0]))
        sector = np.asarray([4, 4, 5, 6])
        np.testing.assert_allclose(
            sector_continuous_reference_phi(phi, sector),
            [179.0, 181.0, -120.0, -60.0],
            atol=1.0e-12,
        )

    def test_reference_phi_is_unfolded_across_sector_one_boundary(self) -> None:
        wrapped = np.deg2rad(np.asarray([-24.0, 0.0, 24.0, -60.0, -25.1]))
        np.testing.assert_allclose(
            unfold_reference_phi(wrapped),
            [-24.0, 0.0, 24.0, 300.0, 334.9],
            atol=1.0e-12,
        )

    def test_reference_phi_is_local_after_global_unfolding(self) -> None:
        sectors = np.arange(1, 7)
        global_phi = np.asarray([0.0, 60.0, 120.0, -180.0, -120.0, -60.0])
        np.testing.assert_allclose(
            reference_sector_phi(np.deg2rad(global_phi), sectors),
            np.zeros(6),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            reference_sector_phi(np.deg2rad([-20.0, -40.0]), [1, 6]),
            [-20.0, 20.0],
            atol=1.0e-12,
        )

    def test_reference_formula_matches_table_polynomial(self) -> None:
        theta_deg = np.asarray([8.0, 10.0, 12.0])
        phi_deg = np.asarray([2.0, 61.0, -60.0])
        sectors = np.asarray([1, 2, 6])
        result = yijie_josh_rgk_6535_delta_p(
            np.deg2rad(theta_deg), np.deg2rad(phi_deg), sectors
        )
        signed_phi = np.asarray([2.0, 61.0, -60.0])
        expected = []
        for theta, phi, sector in zip(theta_deg, signed_phi, sectors):
            c00, c01, c02, c10, c11, c12 = (
                YIJIE_JOSH_RGK_6535_COEFFICIENTS[sector - 1]
            )
            expected.append(
                c00 + c01 * theta + c02 * theta**2
                + (c10 + c11 * theta + c12 * theta**2) * phi
            )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1.0e-14)

    def test_reference_correction_is_additive_not_fractional(self) -> None:
        theta = np.deg2rad(np.asarray([9.0, 9.0]))
        phi = np.deg2rad(np.asarray([0.0, 0.0]))
        sector = np.asarray([1, 1])
        delta = yijie_josh_rgk_6535_delta_p(theta, phi, sector)
        np.testing.assert_allclose(delta[0], delta[1])
        momenta = np.asarray([3.0, 6.0])
        fractional = delta / momenta
        self.assertAlmostEqual(fractional[0], 2.0 * fractional[1])

    def test_global_and_sector_local_phi_readings_are_distinct(self) -> None:
        theta = np.deg2rad(np.asarray([10.0]))
        global_phi = np.deg2rad(np.asarray([240.0]))
        sector = np.asarray([5])
        local = yijie_josh_rgk_6535_delta_p(
            theta, global_phi, sector, phi_convention="sector-local"
        )
        global_unfolded = yijie_josh_rgk_6535_delta_p(
            theta, global_phi, sector, phi_convention="global-unfolded"
        )
        signed_global = yijie_josh_rgk_6535_delta_p(
            theta, global_phi, sector, phi_convention="signed-global"
        )
        self.assertFalse(np.allclose(local, global_unfolded))
        self.assertFalse(np.allclose(signed_global, global_unfolded))
        self.assertFalse(np.allclose(signed_global, local))

    def test_reference_surface_has_expected_scale_at_sector_centers(self) -> None:
        sectors = np.arange(1, 7)
        phi = np.deg2rad(np.asarray([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
        theta = np.deg2rad(np.full(6, 9.5))
        inferred = yijie_josh_rgk_6535_delta_p(theta, phi, sectors)
        unfolded = yijie_josh_rgk_6535_delta_p(
            theta, phi, sectors, phi_convention="global-unfolded"
        )
        self.assertLess(np.max(np.abs(inferred)), 0.01)
        self.assertGreater(abs(unfolded[4]), 0.10)

    def test_coefficient_sector_mapping_is_explicit(self) -> None:
        theta = np.deg2rad(np.asarray([9.0]))
        phi = np.deg2rad(np.asarray([0.0]))
        observed = np.asarray([1])
        identity = yijie_josh_rgk_6535_delta_p(theta, phi, observed)
        shifted = yijie_josh_rgk_6535_delta_p(
            theta, phi, observed, coefficient_sector=np.asarray([2])
        )
        self.assertFalse(np.allclose(identity, shifted))

    def test_invalid_phi_convention_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "phi_convention"):
            yijie_josh_rgk_6535_delta_p(
                np.asarray([0.1]),
                np.asarray([0.0]),
                np.asarray([1]),
                phi_convention="ambiguous",
            )

    def test_invalid_sector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sectors"):
            yijie_josh_rgk_6535_delta_p(
                np.asarray([0.1]), np.asarray([0.0]), np.asarray([0])
            )

    def test_elastic_benchmark_uses_identical_cells_for_all_methods(self) -> None:
        entries_per_sector = 80
        sectors = np.repeat(np.arange(1, 7), entries_per_sector)
        runs = np.tile(np.repeat([5900, 5980], entries_per_sector // 2), 6)
        theta = np.deg2rad(np.full(sectors.size, 9.0))
        phi_deg = 60.0 * (sectors - 1)
        phi = np.deg2rad((phi_deg + 180.0) % 360.0 - 180.0)
        expected = elastic_electron_momentum(theta, 6.535)
        residual = 0.005 + np.linspace(-0.002, 0.002, sectors.size)
        arrays = {
            "runNum": runs,
            "electronP": expected / (1.0 + residual),
            "electronTheta": theta,
            "electronPhi": phi,
            "electronDet": np.ones(sectors.size, dtype=int),
            "electronSector": sectors,
            "nPid11": np.ones(sectors.size, dtype=int),
        }
        regions = []
        for sector in range(1, 7):
            cell = {
                "thetaRangeDeg": [8.0, 10.0],
                "phiRangeDeg": [-5.0, 5.0],
                "thetaHighInclusive": True,
                "phiHighInclusive": True,
            }
            regions.append({
                "pid": 11,
                "detector": 1,
                "sector": sector,
                "thetaCenterDeg": 9.0,
                "thetaScaleDeg": 1.0,
                "phiCenterDeg": 0.0,
                "phiScaleDeg": 5.0,
                "thetaRangeDeg": [8.0, 10.0],
                "phiRangeDeg": [-5.0, 5.0],
                "basis": "polynomial",
                "terms": [{
                    "thetaPower": 0,
                    "phiPower": 0,
                    "coefficient": 0.005,
                }],
                "supportCells": [cell],
                "fit": {"acceptedProfileCells": [cell]},
            })
        parameters = {"beamEnergyGeV": 6.535, "regions": regions}
        blocks = [
            RunBlock(0, (5900,), 240, "P3"),
            RunBlock(1, (5980,), 240, "P4"),
        ]
        report = run_elastic_benchmark(
            arrays,
            parameters,
            blocks,
            ElasticFitConfig(
                beam_energy=6.535,
                electron_selection="inclusive-w",
                min_bin_entries=5,
                peak_seed_half_width=0.02,
            ),
        )
        validated = {
            method: report["methods"][method]["summary"]["validatedCells"]
            for method in ("none", "ours", "yijieJosh")
        }
        self.assertEqual(set(validated.values()), {12})
        self.assertLess(
            report["methods"]["ours"]["summary"]["medianBlockCellCenterRmsAfter"],
            report["methods"]["none"]["summary"]["medianBlockCellCenterRmsAfter"],
        )


if __name__ == "__main__":
    unittest.main()
