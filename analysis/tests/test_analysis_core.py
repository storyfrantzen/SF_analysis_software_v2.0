from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse import csr_matrix


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.binning import AnalysisBinning, legacy_binning
from eppi0.cross_section import integrated_luminosity_fb, virtual_photon_flux
from eppi0.event_sample import (
    build_generated_sample,
    generated_sample_from_tree,
    join_reconstructed,
)
from eppi0.exclusivity import apply_cuts, derive_cuts
from eppi0.harmonics import fit_phi
from eppi0.response import build_response
from eppi0.unfolding import bootstrap_uncertainty, iterative_bayes


class BinningTests(unittest.TestCase):
    def test_legacy_flatten_and_unflatten_are_inverse(self) -> None:
        bins = legacy_binning()
        values = np.arange(bins.size)
        restored = bins.unflatten(values)
        self.assertEqual(restored.shape, bins.shape)
        nq2, nxb, nt, nphi = bins.shape
        for iq2 in (0, nq2 - 1):
            for ixb in (0, nxb - 1):
                for it in (0, nt - 1):
                    for iphi in (0, nphi - 1):
                        flat = bins.flatten(iq2, ixb, it, iphi)
                        self.assertEqual(restored[iq2, ixb, it, iphi], flat)

    def test_phi_wrap_and_upper_edges(self) -> None:
        bins = AnalysisBinning([1, 2], [0, 1], [0, 1], [0, 180, 360])
        flat = bins.coordinates_to_flat(
            np.array([1.5, 1.5, 2.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([-np.pi / 2, 2 * np.pi, 0.0]),
        )
        np.testing.assert_array_equal(flat, [1, 0, -1])

    def test_flatten_values_round_trip(self) -> None:
        bins = legacy_binning()
        values = np.arange(bins.size)
        np.testing.assert_array_equal(bins.flatten_values(bins.unflatten(values)), values)


class ResponseTests(unittest.TestCase):
    def test_efficiency_includes_truth_events_without_reconstruction(self) -> None:
        truth = np.array([0, 0, 1, 1, -1])
        rec = np.array([0, -1, 1, 0, 1])
        selected = np.array([True, False, True, True, True])
        result = build_response(truth, rec, selected, number_of_bins=2)
        np.testing.assert_allclose(result.truth_total, [2, 2])
        np.testing.assert_allclose(result.efficiency, [0.5, 1.0])
        self.assertAlmostEqual(result.feed_in_fraction, 0.25)
        np.testing.assert_allclose(result.feed_in_shape, [0.0, 1.0])


class EventSampleTests(unittest.TestCase):
    def test_compact_generated_tree_filters_invalid_topology(self) -> None:
        sample = generated_sample_from_tree(
            np.array([1001, 1002], dtype=np.uint64),
            np.array([0, 0], dtype=np.uint64),
            np.array([11, 11]),
            np.array([10, 11]),
            np.array([True, False]),
            np.array([1.5, np.nan]),
            np.array([0.2, np.nan]),
            np.array([0.3, np.nan]),
            np.array([0.4, np.nan]),
            np.array([True, False]),
            np.array([2.0, 1.0]),
        )
        np.testing.assert_array_equal(sample.event, [10])
        np.testing.assert_allclose(sample.weight, [2.0])

    def test_source_key_disambiguates_repeated_mc_event_numbers(self) -> None:
        generated = generated_sample_from_tree(
            np.array([1001, 1002], dtype=np.uint64),
            np.array([0, 0], dtype=np.uint64),
            np.array([11, 11]),
            np.array([1, 1]),
            np.array([True, True]),
            np.array([1.5, 1.6]),
            np.array([0.2, 0.21]),
            np.array([0.3, 0.31]),
            np.array([0.4, 0.41]),
            np.array([True, True]),
            np.array([1.0, 1.0]),
        )
        joined = join_reconstructed(
            generated,
            np.array([11, 11]),
            np.array([1, 1]),
            {"rec_Q2": np.array([1.51, 1.61])},
            rec_source_file_id=np.array([1001, 1002], dtype=np.uint64),
            rec_source_event_index=np.array([0, 0], dtype=np.uint64),
        )
        np.testing.assert_allclose(joined["rec_Q2"], [1.51, 1.61])
        np.testing.assert_array_equal(joined["rec_selected"], [True, True])

    def test_radiative_and_nonradiative_events_are_both_preserved(self) -> None:
        # Event 10: e p gamma gamma. Event 11: e p pi0 gamma_rad.
        run = np.full(8, 11)
        event = np.array([10, 10, 10, 10, 11, 11, 11, 11])
        pid = np.array([11, 2212, 22, 22, 11, 2212, 111, 22])
        momentum = np.array([4.0, 1.0, 0.8, 0.7, 4.1, 1.1, 1.0, 0.05])
        theta = np.array([0.25, 0.6, 0.3, 0.4, 0.26, 0.62, 0.35, 0.1])
        phi = np.array([0.1, 2.0, 1.0, 1.2, 0.2, 2.1, 1.1, -0.5])
        sample = build_generated_sample(run, event, pid, momentum, theta, phi, 6.535)
        np.testing.assert_array_equal(sample.event, [10, 11])
        np.testing.assert_array_equal(sample.radiative, [False, True])
        self.assertTrue(np.all(np.isfinite(sample.q2)))

    def test_join_retains_generated_event_without_rec_candidate(self) -> None:
        generated = build_generated_sample(
            np.full(8, 11),
            np.array([10, 10, 10, 10, 11, 11, 11, 11]),
            np.array([11, 2212, 22, 22, 11, 2212, 111, 22]),
            np.array([4.0, 1.0, 0.8, 0.7, 4.1, 1.1, 1.0, 0.05]),
            np.array([0.25, 0.6, 0.3, 0.4, 0.26, 0.62, 0.35, 0.1]),
            np.array([0.1, 2.0, 1.0, 1.2, 0.2, 2.1, 1.1, -0.5]),
            6.535,
        )
        joined = join_reconstructed(
            generated,
            np.array([11]),
            np.array([10]),
            {"rec_Q2": np.array([1.7])},
        )
        np.testing.assert_array_equal(joined["rec_selected"], [True, False])
        self.assertTrue(np.isnan(joined["rec_Q2"][1]))
        np.testing.assert_allclose(joined["gen_weight"], [1.0, 1.0])


class ExclusivityTests(unittest.TestCase):
    def test_global_cut_table_can_be_reused(self) -> None:
        rng = np.random.default_rng(4)
        count = 200
        values = {
            name: rng.normal(0.0, 1.0, count)
            for name in (
                "rec_m_gg", "rec_pT_miss", "rec_m2_epX", "rec_m_eggX", "rec_E_miss", "rec_m2_miss"
            )
        }
        detector = np.ones(count, dtype=int)
        zeros = np.zeros(count, dtype=int)
        cuts = derive_cuts(
            values, detector, zeros, zeros, zeros,
            topologies=(1,), minimum_events=20, global_mode=True,
        )
        mask = apply_cuts(cuts, values, detector, zeros, zeros, zeros)
        self.assertEqual(cuts.group_ids.tolist(), [1])
        self.assertGreater(mask.sum(), count // 2)


class UnfoldingTests(unittest.TestCase):
    def test_identity_response_returns_measured_spectrum(self) -> None:
        response = csr_matrix(np.eye(3))
        measured = np.array([10.0, 20.0, 30.0])
        result = iterative_bayes(response, measured, np.ones(3), 2)
        np.testing.assert_allclose(result.unfolded, measured)

    def test_bootstrap_is_reproducible(self) -> None:
        response = csr_matrix(np.eye(2))
        args = (response, np.array([20.0, 30.0]), np.ones(2), 1, np.ones(2))
        first = bootstrap_uncertainty(*args, experiments=20, seed=7)
        second = bootstrap_uncertainty(*args, experiments=20, seed=7)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])


class NormalizationTests(unittest.TestCase):
    def test_luminosity_and_flux_are_positive(self) -> None:
        self.assertGreater(integrated_luminosity_fb(1.0e-3), 0.0)
        flux = virtual_photon_flux(np.array([2.0]), np.array([0.3]), 6.535)
        self.assertGreater(flux[0], 0.0)


class HarmonicTests(unittest.TestCase):
    def test_weighted_harmonic_fit_recovers_coefficients(self) -> None:
        phi = np.arange(9.0, 360.0, 18.0)
        radians = np.deg2rad(phi)
        expected = np.array([4.0, 1.2, -0.4])
        values = expected[0] + expected[1] * np.cos(radians) + expected[2] * np.cos(2 * radians)
        fit = fit_phi(phi, values, np.full(phi.size, 0.1))
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit.parameters, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
