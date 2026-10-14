from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.beam_spin import (
    HelicityAudit,
    PolarizationTable,
    corrected_helicity,
    extract_beam_spin,
    fit_sine_amplitudes,
    load_polarization_manifest,
    sine_bin_averages,
)


class BeamSpinTest(unittest.TestCase):
    def audit(self) -> HelicityAudit:
        return HelicityAudit(
            runs=np.array([11]),
            plus_charge_nc=np.array([100.0]),
            minus_charge_nc=np.array([100.0]),
            interval_runs=np.array([11]),
            interval_min=np.array([1]),
            interval_max=np.array([1000]),
            interval_pass=np.array([True]),
            interval_sign=np.array([-1], dtype=np.int8),
        )

    def polarization(self) -> PolarizationTable:
        return PolarizationTable(
            runs=np.array([11]),
            values=np.array([0.8]),
            uncertainties=np.array([0.02]),
            labels=np.array(["period-a"]),
            payload={"schema_version": 1},
        )

    def test_qadb_sign_is_applied_to_raw_helicity(self) -> None:
        helicity, accepted = corrected_helicity(
            np.array([11, 11, 11]),
            np.array([10, 20, 2000]),
            np.array([1, -1, 1]),
            self.audit(),
        )
        np.testing.assert_array_equal(helicity, [-1, 1, 0])
        np.testing.assert_array_equal(accepted, [True, True, False])

    def test_charge_balanced_asymmetry_recovers_known_value(self) -> None:
        # With P=0.8, 120 positive and 80 negative events correspond to A=0.25.
        corrected = np.r_[np.ones(120, dtype=int), -np.ones(80, dtype=int)]
        raw = -corrected  # the audit applies a -1 QADB sign correction
        count = raw.size
        result = extract_beam_spin(
            flat_bins=np.zeros(count, dtype=int),
            event_runs=np.full(count, 11),
            event_numbers=np.arange(1, count + 1),
            raw_helicity=raw,
            net_event_weights=np.ones(count),
            active_events=np.ones(count, dtype=bool),
            audit=self.audit(),
            polarization=self.polarization(),
            number_of_bins=2,
        )
        self.assertAlmostEqual(result.asymmetry[0], 0.25)
        self.assertTrue(result.valid[0])
        self.assertEqual(result.used_event_count, 200)
        self.assertAlmostEqual(result.polarization_uncertainty[0], 0.00625)

    def test_sine_fit_recovers_amplitude(self) -> None:
        edges = np.arange(0.0, 360.0 + 18.0, 18.0)
        phi = 0.5 * (edges[:-1] + edges[1:])
        amplitude = 0.31
        basis = sine_bin_averages(edges)
        values = (amplitude * basis)[None, None, None, :]
        errors = np.full(values.shape, 0.02)
        polarization_shifts = (
            0.01 * basis
        )[None, None, None, None, :]
        fit = fit_sine_amplitudes(
            values,
            errors,
            np.ones(values.shape, dtype=bool),
            phi,
            phi_edges_deg=edges,
            polarization_shifts=polarization_shifts,
        )
        self.assertAlmostEqual(fit.amplitude.item(), amplitude, places=12)
        self.assertTrue(fit.valid.item())
        self.assertTrue(fit.quality.item())
        self.assertAlmostEqual(fit.chi2.item(), 0.0, places=12)
        self.assertAlmostEqual(fit.polarization_uncertainty.item(), 0.01, places=12)

    def test_sine_bin_average_differs_from_center_for_wide_bins(self) -> None:
        edges = np.linspace(0.0, 360.0, 9)
        centers = 0.5 * (edges[:-1] + edges[1:])
        expected = np.sin(np.deg2rad(centers)) * np.sinc(1.0 / 8.0)
        np.testing.assert_allclose(sine_bin_averages(edges), expected, atol=1.0e-14)

    def test_polarization_manifest_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polarization.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": "test polarization source",
                        "periods": [
                            {"label": "a", "runs": "1-2", "polarization": 0.8,
                             "uncertainty": 0.01},
                            {"label": "b", "runs": [2, 3], "polarization": 0.9,
                             "uncertainty": 0.02},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple polarization periods"):
                load_polarization_manifest(path)


if __name__ == "__main__":
    unittest.main()
