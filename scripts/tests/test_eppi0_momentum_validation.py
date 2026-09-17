from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from analysis.eppi0.binning import legacy_binning
from analysis.eppi0.exclusivity import ESTIMATOR, GROUPING
from scripts.calibration.eppi0_momentum_validation import (
    EXCLUSIVITY_VALUE_NAMES,
    apply_supported_electron_correction,
    compute_eppi0_observables,
    run_paired_validation,
)


def _parameters() -> dict[str, object]:
    regions = []
    for sector in range(1, 7):
        regions.append({
            "pid": 11,
            "detector": 1,
            "sector": sector,
            "thetaRangeDeg": [5.0, 15.0],
            "phiRangeDeg": [-20.0, 20.0],
            "thetaCenterDeg": 10.0,
            "thetaScaleDeg": 5.0,
            "phiCenterDeg": 0.0,
            "phiScaleDeg": 20.0,
            "basis": "polynomial",
            "terms": [{
                "thetaPower": 0,
                "phiPower": 0,
                "coefficient": 0.01,
            }],
            "supportCells": [{
                "thetaRangeDeg": [5.0, 15.0],
                "phiRangeDeg": [-20.0, 20.0],
            }],
        })
    return {
        "beamEnergyGeV": 10.604,
        "datasetTag": "synthetic",
        "regions": regions,
    }


def _arrays() -> dict[str, np.ndarray]:
    sectors = np.arange(1, 7)
    local_phi_deg = np.asarray([-8.0, -4.0, 0.0, 4.0, 8.0, 0.0])
    global_phi_deg = local_phi_deg + 60.0 * (sectors - 1)
    global_phi_deg = (global_phi_deg + 180.0) % 360.0 - 180.0
    electron_theta_deg = np.asarray([7.0, 8.0, 9.0, 10.0, 11.0, 20.0])
    return {
        "electronP": np.asarray([7.5, 7.0, 6.5, 6.0, 5.5, 4.5]),
        "electronTheta": np.deg2rad(electron_theta_deg),
        "electronPhi": np.deg2rad(global_phi_deg),
        "electronDet": np.ones(6, dtype=int),
        "electronSector": sectors,
        "protonP": np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        "protonTheta": np.deg2rad([35.0, 36.0, 37.0, 38.0, 39.0, 40.0]),
        "protonPhi": np.deg2rad(global_phi_deg + 180.0),
        "protonDet": np.ones(6, dtype=int),
        "gamma1P": np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        "gamma1Theta": np.deg2rad([15.0, 16.0, 17.0, 18.0, 19.0, 20.0]),
        "gamma1Phi": np.deg2rad(global_phi_deg + 20.0),
        "gamma1Det": np.ones(6, dtype=int),
        "gamma2P": np.asarray([0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        "gamma2Theta": np.deg2rad([20.0, 21.0, 22.0, 23.0, 24.0, 25.0]),
        "gamma2Phi": np.deg2rad(global_phi_deg - 20.0),
        "gamma2Det": np.ones(6, dtype=int),
        "runNum": np.full(6, 5423, dtype=int),
        "eventNum": np.arange(6),
    }


def _broad_cuts() -> SimpleNamespace:
    variables = tuple(EXCLUSIVITY_VALUE_NAMES)
    return SimpleNamespace(
        grouping=GROUPING,
        estimator=ESTIMATOR,
        variables=variables,
        group_ids=np.asarray([4], dtype=np.int64),
        lower=np.full((1, len(variables)), -1.0e6),
        upper=np.full((1, len(variables)), 1.0e6),
        global_mode=True,
    )


class Eppi0MomentumValidationTests(unittest.TestCase):
    def test_correction_is_gated_by_exact_support(self) -> None:
        arrays = _arrays()
        corrected, support, correction, _ = apply_supported_electron_correction(
            arrays, _parameters()
        )
        np.testing.assert_array_equal(support, [True, True, True, True, True, False])
        np.testing.assert_allclose(correction[:5], 0.01)
        self.assertEqual(correction[5], 0.0)
        np.testing.assert_allclose(corrected[:5], arrays["electronP"][:5] * 1.01)
        self.assertEqual(corrected[5], arrays["electronP"][5])

    def test_paired_validation_preserves_invariants_and_unsupported_events(self) -> None:
        arrays = _arrays()
        before = compute_eppi0_observables(arrays, arrays["electronP"], 10.604)
        for branch, observable in {
            "Q2": "Q2", "W": "W", "m_gg": "mGG", "t": "t"
        }.items():
            arrays[branch] = before[observable].copy()
        report, diagnostics = run_paired_validation(
            arrays,
            _parameters(),
            _broad_cuts(),
            legacy_binning(),
            external_selection_mask=np.ones(6, dtype=bool),
            minimum_electron_p=0.0,
            minimum_q2=0.0,
            minimum_w=0.0,
        )
        self.assertTrue(report["invariants"]["passed"])
        self.assertTrue(report["referenceRecomputationAudit"]["passed"])
        self.assertEqual(report["selection"]["fixedEntries"], 6)
        self.assertEqual(report["selection"]["fixedSupportedEntries"], 5)
        self.assertGreater(
            report["cohorts"]["fixedSupported"]["quantities"]["Q2"]["delta"]["mean"],
            0.0,
        )
        for name in ("Q2", "W", "m2Miss", "trentoPhi"):
            self.assertEqual(
                diagnostics[f"before_{name}"][5], diagnostics[f"after_{name}"][5]
            )
        self.assertEqual(
            report["cohorts"]["fixedAll"]["quantities"]["mGG"]["delta"]["std"],
            0.0,
        )

    def test_individual_cut_migration_is_not_confounded_by_base_thresholds(self) -> None:
        arrays = _arrays()
        parameters = _parameters()
        corrected, _, _, _ = apply_supported_electron_correction(arrays, parameters)
        before = compute_eppi0_observables(arrays, arrays["electronP"], 10.604)
        after = compute_eppi0_observables(arrays, corrected, 10.604)
        threshold = 0.5 * (before["W"][0] + after["W"][0])
        report, _ = run_paired_validation(
            arrays,
            parameters,
            _broad_cuts(),
            legacy_binning(),
            external_selection_mask=np.ones(6, dtype=bool),
            minimum_electron_p=0.0,
            minimum_q2=0.0,
            minimum_w=float(threshold),
        )
        self.assertGreater(
            report["migration"]["individualBaseThresholds"]["W"][
                "lostAfterCorrection"
            ],
            0,
        )
        invariant_cut = report["migration"]["individualCuts"]["rec_m_gg"]
        self.assertEqual(invariant_cut["lostAfterCorrection"], 0)
        self.assertEqual(invariant_cut["gainedAfterCorrection"], 0)


if __name__ == "__main__":
    unittest.main()
