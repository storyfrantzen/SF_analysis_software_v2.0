from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, eye, save_npz


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.binning import AnalysisBinning, from_config, legacy_binning
from eppi0.bin_centering import compute_bin_centering, physical_mask
from eppi0.cross_section import (
    integrated_luminosity_fb,
    physical_bin_volumes,
    virtual_photon_flux,
)
from eppi0.current_efficiency import (
    RelativeLinearEfficiency,
    correction_artifact,
)
from eppi0.event_sample import (
    build_generated_sample,
    generated_particle_columns,
    generated_sample_from_tree,
    join_reconstructed,
)
from eppi0.exclusivity import (
    apply_cuts,
    derive_cuts,
    estimate_window,
    load_cuts,
    save_cuts,
)
from eppi0.exclusivity_models import estimate_model
from eppi0.harmonics import fit_phi
from eppi0.response import build_response, build_response_from_counts
from eppi0.radiative_correction import (
    _lund_files,
    compute_radiative_correction,
    histogram_lund,
    support_status_codes,
)
from eppi0.root_response import _truth_inside_mask
from eppi0.phase_space import AnalysisPhaseSpace
from eppi0.topology import INVALID_TOPOLOGY, ft_photon_count
from eppi0.unfolding import bootstrap_uncertainty, iterative_bayes
from run_analysis import (
    command_response,
    command_unfold,
    command_response_plots,
    command_bin_centering_merge,
    command_cross_section,
    _load_bin_centering_plot_artifacts,
    _normalization_npz_fields,
    _read_generator_integrated_cross_section,
    _read_generator_normalization_summary,
)
from build_event_sample import (
    reconstructed_columns,
    reverse_join_selected_events,
)


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

    def test_count_response_matches_dense_response(self) -> None:
        truth = np.array([0, 0, 1, 1, -1])
        rec = np.array([0, -1, 1, 0, 1])
        selected = np.array([True, False, True, True, True])
        dense = build_response(truth, rec, selected, number_of_bins=2)
        counted = build_response_from_counts(
            truth_total=np.array([2, 2]),
            reconstructed_total=np.array([2, 2]),
            migration_rows=np.array([0, 1, 0]),
            migration_cols=np.array([0, 1, 1]),
            migration_weights=np.ones(3),
            feed_counts=np.array([0, 1]),
        )
        np.testing.assert_allclose(counted.truth_total, dense.truth_total)
        np.testing.assert_allclose(counted.reconstructed_total, dense.reconstructed_total)
        np.testing.assert_allclose(counted.core.toarray(), dense.core.toarray())
        np.testing.assert_allclose(counted.efficiency, dense.efficiency)
        self.assertAlmostEqual(counted.feed_in_fraction, dense.feed_in_fraction)
        np.testing.assert_allclose(counted.feed_in_shape, dense.feed_in_shape)

    def test_response_command_applies_analysis_phase_space_to_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            config_path = tmpdir / "analysis.json"
            sample_path = tmpdir / "sample.npz"
            output_dir = tmpdir / "response"
            config_path.write_text(
                json.dumps(
                    {
                        "beam_energy": 6.535,
                        "binning": {
                            "Q2": [1.0, 1.5],
                            "xB": [0.1, 0.3],
                            "minus_t": [0.1, 0.3],
                            "phi_deg": [0.0, 360.0],
                        },
                        "phase_space": {"electron_p_min": 5.0},
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                sample_path,
                gen_Q2=np.array([1.2]),
                gen_xB=np.array([0.2]),
                gen_minus_t=np.array([0.2]),
                gen_trento_phi=np.array([0.0]),
                rec_Q2=np.array([1.2]),
                rec_xB=np.array([0.2]),
                rec_minus_t=np.array([0.2]),
                rec_trento_phi=np.array([0.0]),
                rec_selected=np.array([True]),
            )
            args = argparse.Namespace(
                sample=sample_path,
                config=config_path,
                output_dir=output_dir,
                selection_mask=None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                command_response(args)
            metadata = np.load(output_dir / "response_meta.npz", allow_pickle=False)
        np.testing.assert_allclose(metadata["truth_total"], [0.0])
        np.testing.assert_allclose(metadata["reconstructed_total"], [1.0])
        self.assertEqual(float(metadata["feed_in_fraction"]), 1.0)

    def test_root_response_truth_mask_applies_analysis_phase_space(self) -> None:
        mask = _truth_inside_mask(
            topology_valid=np.array([True, True, True]),
            truth_flat=np.array([0, 0, -1]),
            q2=np.array([1.2, 1.2, 1.2]),
            xb=np.array([0.2, 0.8, 0.8]),
            number_of_bins=1,
            phase_space=AnalysisPhaseSpace(electron_p_min=5.0),
            beam_energy=6.535,
        )
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_response_plots_writes_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            matrix_path = tmpdir / "response_matrix.npz"
            meta_path = tmpdir / "response_meta.npz"
            output_path = tmpdir / "response_plots.pdf"
            save_npz(
                matrix_path,
                csr_matrix(
                    np.array(
                        [
                            [0.8, 0.2],
                            [0.1, 0.7],
                        ]
                    )
                ),
            )
            np.savez_compressed(
                meta_path,
                efficiency=np.array([0.9, 0.9]),
                truth_total=np.array([100.0, 100.0]),
                reconstructed_total=np.array([90.0, 90.0]),
                q2_edges=np.array([1.0, 2.0]),
                xb_edges=np.array([0.1, 0.2, 0.3]),
                t_edges=np.array([0.1, 0.2]),
                phi_edges=np.array([0.0, 360.0]),
            )
            args = argparse.Namespace(
                response_matrix=matrix_path,
                response_meta=meta_path,
                output=output_path,
                max_points=1000,
                seed=12345,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                command_response_plots(args)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


class EventSampleTests(unittest.TestCase):
    def test_reverse_join_exports_only_valid_selected_matches(self) -> None:
        rec = {
            "runNum": np.array([11, 11, 11, 11]),
            "eventNum": np.array([1, 1, 2, 3]),
            "sourceFileId": np.array([1001, 1002, 1001, 1001], dtype=np.uint64),
            "sourceEventIndex": np.array([0, 0, 1, 2], dtype=np.uint64),
        }
        rec_values = {
            "rec_Q2": np.array([1.51, 1.61, 1.71, 1.81]),
            "rec_y": np.array([0.30, 0.33, 0.34, 0.35]),
            "rec_W": np.array([2.20, 2.23, 2.24, 2.25]),
        }
        chunks = [
            {
                "sourceFileId": np.array([1001, 1001], dtype=np.uint64),
                "sourceEventIndex": np.array([1, 2], dtype=np.uint64),
                "runNum": np.array([11, 11]),
                "eventNum": np.array([2, 3]),
                "topologyValid": np.array([True, False]),
                "Q2": np.array([1.7, np.nan]),
                "xB": np.array([0.2, np.nan]),
                "y": np.array([0.31, np.nan]),
                "W": np.array([2.21, np.nan]),
                "minusT": np.array([0.3, np.nan]),
                "trentoPhi": np.array([0.4, np.nan]),
                "radiative": np.array([True, True]),
                "weight": np.array([1.0, 1.0]),
                "electronP": np.array([4.2, np.nan]),
            },
            {
                "sourceFileId": np.array([1002, 9999], dtype=np.uint64),
                "sourceEventIndex": np.array([0, 0], dtype=np.uint64),
                "runNum": np.array([11, 11]),
                "eventNum": np.array([1, 99]),
                "topologyValid": np.array([True, True]),
                "Q2": np.array([1.6, 2.0]),
                "xB": np.array([0.21, 0.3]),
                "y": np.array([0.32, 0.4]),
                "W": np.array([2.22, 2.5]),
                "minusT": np.array([0.31, 0.5]),
                "trentoPhi": np.array([0.41, 0.6]),
                "radiative": np.array([True, True]),
                "weight": np.array([2.0, 1.0]),
                "electronP": np.array([4.1, 5.0]),
            },
        ]
        sample, stats = reverse_join_selected_events(
            rec, rec_values, chunks, ["electronP"]
        )
        np.testing.assert_array_equal(sample["source_file_id"], [1002, 1001])
        np.testing.assert_array_equal(sample["source_event_index"], [0, 1])
        np.testing.assert_allclose(sample["gen_Q2"], [1.6, 1.7])
        np.testing.assert_allclose(sample["rec_Q2"], [1.61, 1.71])
        np.testing.assert_allclose(sample["gen_y"], [0.32, 0.31])
        np.testing.assert_allclose(sample["gen_W"], [2.22, 2.21])
        np.testing.assert_allclose(sample["rec_y"], [0.33, 0.34])
        np.testing.assert_allclose(sample["rec_W"], [2.23, 2.24])
        np.testing.assert_allclose(sample["gen_electronP"], [4.1, 4.2])
        np.testing.assert_array_equal(sample["rec_selected"], [True, True])
        self.assertEqual(stats["generated_events_scanned"], 4)
        self.assertEqual(stats["valid_generated_events"], 3)
        self.assertEqual(stats["selected_reconstructed_events"], 4)
        self.assertEqual(stats["matched_generated_events"], 2)
        self.assertEqual(stats["unmatched_selected_events"], 1)
        self.assertEqual(stats["invalid_generated_matches"], 1)

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

    def test_reconstructed_columns_keep_extra_selected_scalars(self) -> None:
        rec = {
            "runNum": np.array([11, 11]),
            "eventNum": np.array([10, 11]),
            "sourceFileId": np.array([1001, 1001], dtype=np.uint64),
            "sourceEventIndex": np.array([0, 1], dtype=np.uint64),
            "Q2": np.array([1.5, 1.6]),
            "t": np.array([0.3, 0.4]),
            "passSamplingFraction": np.array([1, 0]),
            "electronP": np.array([3.0, 3.2]),
            "protonP": np.array([1.0, 1.1]),
            "gamma1P": np.array([0.8, 0.9]),
            "eDet": np.array([1, 0]),
            "pDet": np.array([1, 2]),
            "g1Det": np.array([1, 0]),
            "g2Det": np.array([1, 1]),
        }
        values = reconstructed_columns(rec, list(rec))
        self.assertNotIn("rec_runNum", values)
        self.assertNotIn("rec_sourceFileId", values)
        np.testing.assert_allclose(values["rec_Q2"], [1.5, 1.6])
        np.testing.assert_allclose(values["rec_minus_t"], [0.3, 0.4])
        np.testing.assert_array_equal(values["rec_passSamplingFraction"], [1, 0])
        np.testing.assert_allclose(values["rec_electronP"], [3.0, 3.2])
        np.testing.assert_allclose(values["rec_protonP"], [1.0, 1.1])
        np.testing.assert_allclose(values["rec_gamma1P"], [0.8, 0.9])
        np.testing.assert_array_equal(values["rec_electron_detector"], [1, 0])
        np.testing.assert_array_equal(values["rec_proton_detector"], [1, 2])
        np.testing.assert_array_equal(values["rec_gamma1_detector"], [1, 0])
        np.testing.assert_array_equal(values["rec_gamma2_detector"], [1, 1])
        np.testing.assert_array_equal(values["rec_ft_photon_count"], [0, 1])

    def test_generated_particle_columns_align_to_generated_events(self) -> None:
        run = np.full(8, 11)
        event = np.array([10, 10, 10, 10, 11, 11, 11, 11])
        pid = np.array([11, 2212, 22, 22, 11, 2212, 111, 22])
        momentum = np.array([4.0, 1.0, 0.8, 0.7, 4.1, 1.1, 1.0, 0.05])
        theta = np.array([0.25, 0.6, 0.3, 0.4, 0.26, 0.62, 0.35, 0.1])
        phi = np.array([0.1, 2.0, 1.0, 1.2, 0.2, 2.1, 1.1, -0.5])
        columns = generated_particle_columns(
            run,
            event,
            pid,
            momentum,
            theta,
            phi,
            np.array([11, 11, 11]),
            np.array([10, 11, 12]),
        )
        np.testing.assert_allclose(columns["gen_electronP"], [4.0, 4.1, np.nan])
        np.testing.assert_allclose(columns["gen_protonTheta"], [0.6, 0.62, np.nan])
        np.testing.assert_allclose(columns["gen_gamma1P"], [0.8, 0.05, np.nan])
        np.testing.assert_allclose(columns["gen_gamma2P"], [0.7, np.nan, np.nan])
        np.testing.assert_allclose(columns["gen_pi0P"][1], 1.0)
        self.assertTrue(np.isfinite(columns["gen_pi0P"][0]))

    def test_generated_particle_columns_use_source_keys_when_available(self) -> None:
        run = np.array([11, 11])
        event = np.array([10, 10])
        columns = generated_particle_columns(
            run,
            event,
            np.array([11, 11]),
            np.array([4.0, 5.0]),
            np.array([0.2, 0.3]),
            np.array([1.0, 1.1]),
            np.array([11, 11]),
            np.array([10, 10]),
            source_file_id=np.array([1001, 1002], dtype=np.uint64),
            source_event_index=np.array([0, 0], dtype=np.uint64),
            target_source_file_id=np.array([1002, 1001], dtype=np.uint64),
            target_source_event_index=np.array([0, 0], dtype=np.uint64),
        )
        np.testing.assert_allclose(columns["gen_electronP"], [5.0, 4.0])


class ExclusivityTests(unittest.TestCase):
    def _fit_model(
        self,
        variable: str,
        values: np.ndarray,
        expected_center: float | None,
        physical_lower: float | None,
    ):
        estimate, reason = estimate_model(
            values,
            variable,
            3.0,
            50,
            5.0,
            100,
            1.0e-5,
            160,
            0.1,
            3.0,
            expected_center,
            physical_lower,
            1.0,
            1.0,
        )
        self.assertIsNotNone(estimate, reason)
        return estimate

    def test_ft_photon_count_is_order_independent(self) -> None:
        counts = ft_photon_count(
            np.array([1, 0, 1, 0, 2]),
            np.array([1, 1, 0, 0, 1]),
        )
        np.testing.assert_array_equal(counts, [0, 1, 1, 2, INVALID_TOPOLOGY])

    def test_global_cut_table_can_be_reused(self) -> None:
        rng = np.random.default_rng(4)
        count = 200
        values = {
            "rec_m_gg": rng.normal(0.135, 0.01, count),
            "rec_pT_miss": rng.normal(0.1, 0.02, count),
            "rec_m2_epX": rng.normal(0.018, 0.05, count),
            "rec_m_eggX": rng.normal(0.938, 0.05, count),
            "rec_E_miss": rng.normal(0.0, 0.1, count),
            "rec_m2_miss": rng.normal(0.0, 0.05, count),
        }
        detector = np.ones(count, dtype=int)
        ft_photons = np.zeros(count, dtype=int)
        zeros = np.zeros(count, dtype=int)
        cuts = derive_cuts(
            values, detector, ft_photons, zeros, zeros, zeros,
            topologies=(1,), minimum_events=20, global_mode=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cuts.npz"
            save_cuts(str(path), cuts)
            cuts = load_cuts(str(path))
        mask = apply_cuts(cuts, values, detector, ft_photons, zeros, zeros, zeros)
        self.assertEqual(cuts.group_ids.tolist(), [4])
        self.assertEqual(cuts.populated_group_ids.tolist(), [4])
        self.assertEqual(cuts.dropped_group_ids.size, 0)
        self.assertAlmostEqual(cuts.signal_containment, 0.9973002039)
        self.assertEqual(cuts.fit_parameter_names.shape, (1, 6, 8))
        self.assertTrue(np.any(cuts.fit_parameter_names != ""))
        self.assertGreater(mask.sum(), count // 2)

    def test_mass_and_transverse_momentum_use_physical_signal_models(self) -> None:
        rng = np.random.default_rng(59)
        mgg = np.concatenate(
            (rng.normal(0.135, 0.010, 8000), rng.uniform(0.07, 0.20, 2000))
        )
        mgg_fit = self._fit_model("rec_m_gg", mgg, 0.1349768, 0.0)
        self.assertEqual(mgg_fit.fit_model, "gaussian+sideband-linear")
        self.assertAlmostEqual(mgg_fit.center, 0.135, delta=0.005)

        px = rng.normal(0.025, 0.015, 9000)
        py = rng.normal(0.0, 0.015, 9000)
        pt = np.concatenate((np.hypot(px, py), rng.uniform(0.0, 0.18, 1200)))
        pt_fit = self._fit_model("rec_pT_miss", pt, None, 0.0)
        self.assertIn(pt_fit.fit_model, ("rice+linear", "rayleigh+linear"))
        self.assertEqual(pt_fit.lower, 0.0)
        self.assertGreater(pt_fit.upper, pt_fit.center)

    def test_missing_mass_uses_split_gaussian_only_when_supported(self) -> None:
        rng = np.random.default_rng(61)
        symmetric = np.concatenate(
            (rng.normal(0.938, 0.05, 9000), rng.uniform(0.5, 1.4, 1000))
        )
        symmetric_fit = self._fit_model("rec_m_eggX", symmetric, 0.9382721, 0.0)
        self.assertEqual(symmetric_fit.fit_model, "gaussian+linear")

        left, right = 0.025, 0.090
        count = 12000
        choose_left = rng.random(count) < left / (left + right)
        residual = np.where(
            choose_left,
            -np.abs(rng.normal(0.0, left, count)),
            np.abs(rng.normal(0.0, right, count)),
        )
        asymmetric = np.concatenate(
            (0.938 + residual, rng.uniform(0.45, 1.55, 1000))
        )
        asymmetric_fit = self._fit_model(
            "rec_m_eggX", asymmetric, 0.9382721, 0.0
        )
        self.assertEqual(asymmetric_fit.fit_model, "split-gaussian+linear")
        self.assertGreater(
            asymmetric_fit.upper - asymmetric_fit.center,
            asymmetric_fit.center - asymmetric_fit.lower,
        )

    def test_missing_energy_and_missing_mass_squared_use_asymmetric_models(self) -> None:
        rng = np.random.default_rng(67)
        missing_energy = np.concatenate(
            (
                rng.normal(0.0, 0.05, 6500),
                rng.normal(0.0, 0.05, 2000) + rng.exponential(0.15, 2000),
                rng.uniform(-0.5, 0.7, 1500),
            )
        )
        energy_fit = self._fit_model("rec_E_miss", missing_energy, 0.0, None)
        self.assertEqual(
            energy_fit.fit_model, "gaussian+positive-exgaussian+linear"
        )
        self.assertGreater(
            energy_fit.upper - energy_fit.center,
            energy_fit.center - energy_fit.lower,
        )

        left, right = 0.008, 0.025
        count = 12000
        choose_left = rng.random(count) < left / (left + right)
        missing_mass_squared = np.where(
            choose_left,
            -rng.exponential(left, count),
            rng.exponential(right, count),
        )
        missing_mass_squared = np.concatenate(
            (missing_mass_squared, rng.uniform(-0.15, 0.20, 1200))
        )
        mass_squared_fit = self._fit_model(
            "rec_m2_miss", missing_mass_squared, 0.0, None
        )
        self.assertEqual(
            mass_squared_fit.fit_model, "asymmetric-laplace+linear"
        )
        self.assertGreater(
            mass_squared_fit.upper - mass_squared_fit.center,
            mass_squared_fit.center - mass_squared_fit.lower,
        )

    def test_global_mode_is_default_and_records_dropped_topologies(self) -> None:
        rng = np.random.default_rng(53)
        dense = rng.normal(0.0, 0.1, 200)
        sparse = rng.normal(1.0, 0.1, 10)
        values = {"toy_peak": np.concatenate((dense, sparse))}
        detector = np.ones(210, dtype=int)
        ft_photons = np.concatenate(
            (np.zeros(200, dtype=int), np.ones(10, dtype=int))
        )
        iq2 = np.concatenate(
            (np.zeros(100, dtype=int), np.ones(100, dtype=int), np.zeros(10, dtype=int))
        )
        zeros = np.zeros(210, dtype=int)
        cuts = derive_cuts(
            values,
            detector,
            ft_photons,
            iq2,
            zeros,
            zeros,
            variables=("toy_peak",),
            topologies=(1,),
            minimum_events=50,
        )
        self.assertTrue(cuts.global_mode)
        self.assertEqual(cuts.group_ids.tolist(), [4])
        self.assertEqual(cuts.populated_group_ids.tolist(), [4, 5])
        self.assertEqual(cuts.dropped_group_ids.tolist(), [5])
        self.assertEqual(cuts.dropped_variables.tolist(), ["toy_peak"])
        self.assertIn("too few finite entries", cuts.dropped_reasons[0])

    def test_cut_windows_are_distinct_for_zero_one_and_two_ft_photons(self) -> None:
        rng = np.random.default_rng(17)
        rows_per_topology = 100
        ft_photons = np.repeat(np.arange(3), rows_per_topology)
        values = {
            "toy_peak": np.concatenate(
                [rng.normal(10.0 * topology, 0.1, rows_per_topology) for topology in range(3)]
            )
        }
        detector = np.ones(ft_photons.size, dtype=int)
        zeros = np.zeros(ft_photons.size, dtype=int)
        cuts = derive_cuts(
            values,
            detector,
            ft_photons,
            zeros,
            zeros,
            zeros,
            variables=("toy_peak",),
            topologies=(1,),
            minimum_events=20,
            global_mode=False,
        )
        self.assertEqual(cuts.group_ids.size, 3)

        target_values = {"toy_peak": np.array([0.0, 10.0, 20.0, 0.0])}
        target_ft_photons = np.array([0, 1, 2, 2])
        target_detector = np.ones(4, dtype=int)
        target_bins = np.zeros(4, dtype=int)
        mask = apply_cuts(
            cuts,
            target_values,
            target_detector,
            target_ft_photons,
            target_bins,
            target_bins,
            target_bins,
        )
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_gaussian_core_window_rejects_broad_background_tails(self) -> None:
        rng = np.random.default_rng(23)
        signal = rng.normal(0.02, 0.06, 5000)
        background = rng.uniform(-2.0, 3.0, 2000)
        window = estimate_window(
            np.concatenate((signal, background)),
            n_sigma=3.0,
            minimum_events=50,
            expected_center=0.02,
        )
        self.assertIsNotNone(window)
        lower, upper = window
        self.assertGreater(0.5 * (upper - lower), 0.12)
        self.assertLess(0.5 * (upper - lower), 0.22)
        self.assertAlmostEqual(0.5 * (lower + upper), 0.02, delta=0.03)

    def test_gaussian_core_window_rejects_sloped_background(self) -> None:
        rng = np.random.default_rng(31)
        signal = rng.normal(0.02, 0.06, 5000)
        background = rng.triangular(-2.0, 3.0, 3.0, 2500)
        window = estimate_window(
            np.concatenate((signal, background)),
            n_sigma=3.0,
            minimum_events=50,
            expected_center=0.02,
        )
        self.assertIsNotNone(window)
        lower, upper = window
        self.assertGreater(0.5 * (upper - lower), 0.12)
        self.assertLess(0.5 * (upper - lower), 0.22)
        self.assertAlmostEqual(0.5 * (lower + upper), 0.02, delta=0.03)

    def test_background_without_signal_is_rejected(self) -> None:
        rng = np.random.default_rng(37)
        background = rng.triangular(-2.0, 3.0, 3.0, 7000)
        window = estimate_window(
            background,
            n_sigma=3.0,
            minimum_events=50,
            expected_center=0.02,
        )
        self.assertIsNone(window)

    def test_core_plus_broad_tail_recovers_narrow_resolution(self) -> None:
        rng = np.random.default_rng(41)
        core = rng.normal(0.02, 0.06, 4000)
        tail = rng.normal(0.07, 0.22, 5000)
        background = rng.uniform(-2.0, 3.0, 1000)
        window = estimate_window(
            np.concatenate((core, tail, background)),
            n_sigma=3.0,
            minimum_events=50,
            expected_center=0.018,
            maximum_center_deviation=0.20,
            maximum_sigma=0.20,
        )
        self.assertIsNotNone(window)
        lower, upper = window
        self.assertGreater(0.5 * (upper - lower), 0.12)
        self.assertLess(0.5 * (upper - lower), 0.23)
        self.assertAlmostEqual(0.5 * (lower + upper), 0.02, delta=0.04)

    def test_expected_center_validation_does_not_scale_with_bad_width(self) -> None:
        rng = np.random.default_rng(43)
        wrong_peak = rng.normal(3.5, 1.4, 4000)
        window = estimate_window(
            wrong_peak,
            n_sigma=3.0,
            minimum_events=50,
            expected_center=0.018,
            maximum_center_deviation=0.20,
            maximum_sigma=0.20,
        )
        self.assertIsNone(window)

    def test_sparse_group_uses_same_topology_fallback(self) -> None:
        rng = np.random.default_rng(29)
        dense = rng.normal(0.0, 0.1, 400)
        sparse = rng.normal(0.0, 0.1, 10)
        values = {"toy_peak": np.concatenate((dense, sparse))}
        detector = np.ones(410, dtype=int)
        ft_photons = np.zeros(410, dtype=int)
        iq2 = np.concatenate((np.zeros(400, dtype=int), np.ones(10, dtype=int)))
        zeros = np.zeros(410, dtype=int)
        cuts = derive_cuts(
            values,
            detector,
            ft_photons,
            iq2,
            zeros,
            zeros,
            variables=("toy_peak",),
            topologies=(1,),
            minimum_events=50,
            global_mode=False,
        )
        self.assertEqual(cuts.group_ids.size, 2)
        self.assertEqual(cuts.window_source[0, 0], "local")
        self.assertEqual(cuts.window_source[1, 0], "topology_fallback")
        self.assertTrue(np.all(np.isfinite(cuts.lower)))
        self.assertTrue(np.all(np.isfinite(cuts.upper)))

    def test_broad_local_hump_uses_topology_consistency_fallback(self) -> None:
        rng = np.random.default_rng(47)
        narrow = rng.normal(0.02, 0.06, 1500)
        broad = rng.normal(0.10, 0.24, 300)
        values = {"toy_peak": np.concatenate((narrow, broad))}
        detector = np.ones(1800, dtype=int)
        ft_photons = np.zeros(1800, dtype=int)
        iq2 = np.concatenate((np.zeros(1500, dtype=int), np.ones(300, dtype=int)))
        zeros = np.zeros(1800, dtype=int)
        cuts = derive_cuts(
            values,
            detector,
            ft_photons,
            iq2,
            zeros,
            zeros,
            variables=("toy_peak",),
            topologies=(1,),
            minimum_events=50,
            global_mode=False,
        )
        self.assertEqual(cuts.group_ids.size, 2)
        self.assertEqual(cuts.window_source[0, 0], "local")
        self.assertEqual(
            cuts.window_source[1, 0], "topology_consistency_fallback"
        )
        self.assertLess(cuts.sigmas[1, 0], 0.12)

    def test_tail_based_cut_table_requires_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tail_based_cuts.npz"
            np.savez_compressed(
                path,
                variables=np.array(["rec_m_gg"]),
                group_ids=np.array([4]),
                lower=np.array([[0.0]]),
                upper=np.array([[1.0]]),
                global_mode=True,
                n_sigma=3.0,
                minimum_events=50,
                grouping="proton-detector+ft-photon-count-v1",
            )
            with self.assertRaisesRegex(ValueError, "predates signal-background fitting"):
                load_cuts(str(path))

    def test_previous_fitted_cut_tables_require_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for estimator in (
                "mode-seeded-iterative-gaussian-core-v1",
                "binned-gaussian-linear-background-mixture-v2",
                "binned-gaussian-core-tail-background-mixture-v3",
                "binned-gaussian-core-tail-background-mixture-v4",
            ):
                path = Path(tmp) / f"{estimator}.npz"
                np.savez_compressed(
                    path,
                    variables=np.array(["rec_m_gg"]),
                    group_ids=np.array([4]),
                    lower=np.array([[0.0]]),
                    upper=np.array([[1.0]]),
                    global_mode=True,
                    n_sigma=3.0,
                    minimum_events=50,
                    grouping="proton-detector+ft-photon-count-v1",
                    estimator=estimator,
                )
                with self.assertRaisesRegex(
                    ValueError, "unsupported exclusivity estimator"
                ):
                    load_cuts(str(path))

    def test_legacy_cut_table_requires_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_cuts.npz"
            np.savez_compressed(
                path,
                variables=np.array(["rec_m_gg"]),
                group_ids=np.array([1]),
                lower=np.array([[0.0]]),
                upper=np.array([[1.0]]),
                global_mode=True,
                n_sigma=3.0,
                minimum_events=50,
            )
            with self.assertRaisesRegex(ValueError, "predates FT-photon topology"):
                load_cuts(str(path))


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

    def test_unfold_divides_by_radiative_correction(self) -> None:
        config = Path("configs/analysis/rgk/6.535.json")
        binning = from_config(config)
        flat = int(binning.coordinates_to_flat(
            np.asarray([1.2]),
            np.asarray([0.12]),
            np.asarray([0.2]),
            np.asarray([0.1]),
        )[0])
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_path = tmpdir / "data.npz"
            matrix_path = tmpdir / "response.npz"
            meta_path = tmpdir / "response_meta.npz"
            correction_path = tmpdir / "C_rad.npz"
            output_path = tmpdir / "unfolding.npz"
            np.savez_compressed(
                data_path,
                rec_Q2=np.asarray([1.2]),
                rec_xB=np.asarray([0.12]),
                rec_minus_t=np.asarray([0.2]),
                rec_trento_phi=np.asarray([0.1]),
                rec_selected=np.asarray([True]),
            )
            save_npz(matrix_path, eye(binning.size, format="csr"))
            np.savez_compressed(
                meta_path,
                efficiency=np.ones(binning.size),
                feed_in_fraction=0.0,
                feed_in_shape=np.zeros(binning.size),
                response_variance_sum=np.zeros(binning.size),
            )
            c_rad_flat = np.ones(binning.size)
            c_rad_flat[flat] = 2.0
            np.savez_compressed(
                correction_path,
                C_rad=binning.unflatten(c_rad_flat),
                delta_C=np.zeros(binning.shape),
                reliable=np.ones(binning.shape, dtype=bool),
            )
            args = argparse.Namespace(
                data=data_path,
                response_matrix=matrix_path,
                response_meta=meta_path,
                config=config,
                output=output_path,
                selection_mask=None,
                iterations=0,
                bootstrap=0,
                seed=12345,
                radiative_correction=correction_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                command_unfold(args)
            result = np.load(output_path, allow_pickle=False)
        self.assertEqual(result["unfolded"][flat], 1.0)
        self.assertEqual(result["corrected_yield"][flat], 0.5)

    def test_unfold_applies_run_current_weights_and_weighted_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            config_path = tmpdir / "analysis.json"
            data_path = tmpdir / "data.npz"
            matrix_path = tmpdir / "response.npz"
            meta_path = tmpdir / "response_meta.npz"
            correction_path = tmpdir / "current_efficiency_correction.json"
            output_path = tmpdir / "unfolding.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "beam_energy": 6.535,
                        "minimum_acceptance": 0.005,
                        "binning": {
                            "Q2": [1.0, 1.5],
                            "xB": [0.1, 0.3],
                            "minus_t": [0.1, 0.3],
                            "phi_deg": [0.0, 360.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                data_path,
                run=np.asarray([1001, 1002]),
                rec_Q2=np.asarray([1.2, 1.4]),
                rec_xB=np.asarray([0.2, 0.2]),
                rec_minus_t=np.asarray([0.2, 0.2]),
                rec_trento_phi=np.asarray([0.1, 0.1]),
                rec_selected=np.asarray([True, True]),
                beam_charge_c=np.asarray(2.0e-9),
            )
            save_npz(matrix_path, eye(1, format="csr"))
            np.savez_compressed(
                meta_path,
                efficiency=np.ones(1),
                feed_in_fraction=0.0,
                feed_in_shape=np.zeros(1),
                response_variance_sum=np.zeros(1),
            )
            data_model = RelativeLinearEfficiency(
                100.0, -1.0, ((0.0, 0.0), (0.0, 0.0))
            )
            gemc_model = RelativeLinearEfficiency(
                0.8, -0.002, ((0.0, 0.0), (0.0, 0.0))
            )
            records = [
                SimpleNamespace(
                    run=1001,
                    current_nA=0.0,
                    charge_c=1.0e-9,
                    run_class="L5",
                    current_quality="unflagged",
                    included=True,
                    exclusion_reason="",
                ),
                SimpleNamespace(
                    run=1002,
                    current_nA=60.0,
                    charge_c=1.0e-9,
                    run_class="P3",
                    current_quality="unflagged",
                    included=True,
                    exclusion_reason="",
                ),
            ]
            payload = correction_artifact(
                data_model=data_model,
                gemc_model=gemc_model,
                reference_current_nA=60.0,
                reference_label="merged_60nA",
                reference_response_meta=meta_path,
                run_records=records,
                sources={},
            )
            correction_path.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(
                data=data_path,
                response_matrix=matrix_path,
                response_meta=meta_path,
                config=config_path,
                output=output_path,
                selection_mask=None,
                iterations=0,
                bootstrap=0,
                seed=12345,
                radiative_correction=None,
                current_efficiency_correction=correction_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                command_unfold(args)
            with np.load(output_path, allow_pickle=False) as result:
                measured = float(result["measured"][0])
                variance = float(result["measured_variance"][0])
                q2_mean = float(result["Q2_mean"][0])
                applied = bool(result["current_efficiency_applied"])

        low_current_weight = 0.85
        high_current_weight = 0.85 / 0.4
        self.assertAlmostEqual(measured, low_current_weight + high_current_weight)
        self.assertAlmostEqual(
            variance, low_current_weight**2 + high_current_weight**2
        )
        self.assertAlmostEqual(
            q2_mean,
            (1.2 * low_current_weight + 1.4 * high_current_weight)
            / (low_current_weight + high_current_weight),
        )
        self.assertTrue(applied)


class RadiativeCorrectionTests(unittest.TestCase):
    def test_lund_histogram_streams_into_configured_bins(self) -> None:
        bins = AnalysisBinning([1.0, 1.5], [0.2, 0.3], [0.2, 0.3], [0.0, 180.0, 360.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "born.txt"
            path.write_text(_lund_event(pi0=False), encoding="utf-8")
            result = histogram_lund(path, bins, beam_energy=6.535, chunk_size=1)
        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.topology_events, 1)
        self.assertEqual(result.in_range, 1)
        self.assertEqual(result.counts.sum(), 1.0)
        self.assertTrue(np.isfinite(result.generated_q2_min))
        self.assertTrue(np.isfinite(result.generated_eprime_max))

    def test_lund_histogram_applies_analysis_phase_space(self) -> None:
        bins = AnalysisBinning([1.0, 1.5], [0.2, 0.3], [0.2, 0.3], [0.0, 180.0, 360.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "born.txt"
            path.write_text(_lund_event(pi0=False), encoding="utf-8")
            result = histogram_lund(
                path,
                bins,
                beam_energy=6.535,
                chunk_size=1,
                phase_space=AnalysisPhaseSpace(electron_p_min=5.0),
            )
        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.topology_events, 1)
        self.assertEqual(result.in_range, 0)
        self.assertEqual(result.counts.sum(), 0.0)

    def test_lund_directory_accepts_non_txt_lund_files(self) -> None:
        bins = AnalysisBinning([1.0, 1.5], [0.2, 0.3], [0.2, 0.3], [0.0, 180.0, 360.0])
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "job.log").write_text("not a LUND file\n", encoding="utf-8")
            (tmpdir / "input.inp").write_text(_lund_event(pi0=False), encoding="utf-8")
            path = tmpdir / "events.lund"
            path.write_text(_lund_event(pi0=False), encoding="utf-8")
            self.assertEqual(_lund_files(tmpdir), [path])
            result = histogram_lund(tmpdir, bins, beam_energy=6.535, chunk_size=1)
        self.assertEqual(result.files, 1)
        self.assertEqual(result.topology_events, 1)

    def test_lund_discovery_trusts_nonempty_lund_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            path = tmpdir / "events.lund"
            path.write_text("metadata preamble\nnot a standard header\n", encoding="utf-8")
            self.assertEqual(_lund_files(tmpdir), [path])

    def test_radiative_correction_keeps_native_4d_shape(self) -> None:
        bins = AnalysisBinning([1.0, 1.5], [0.2, 0.3], [0.2, 0.3], [0.0, 180.0, 360.0])
        with tempfile.TemporaryDirectory() as tmp:
            born = Path(tmp) / "born.txt"
            rad = Path(tmp) / "rad.txt"
            born.write_text(_lund_event(pi0=False), encoding="utf-8")
            rad.write_text(_lund_event(pi0=True), encoding="utf-8")
            result = compute_radiative_correction(
                born,
                rad,
                bins,
                beam_energy=6.535,
                min_counts=1,
                chunk_size=1,
            )
        self.assertEqual(result.c_rad.shape, bins.shape)
        self.assertEqual(result.delta_c.shape, bins.shape)
        self.assertEqual(result.reliable.shape, bins.shape)
        self.assertEqual(result.support_overlap.shape, bins.shape)
        self.assertEqual(result.support_status.shape, bins.shape)
        self.assertEqual(np.count_nonzero(result.reliable), 1)
        self.assertEqual(np.count_nonzero(result.support_overlap), 1)
        np.testing.assert_allclose(result.c_rad[result.reliable], [1.0])
        np.testing.assert_allclose(result.c_rad[~result.reliable], 1.0)

    def test_radiative_correction_uses_integrated_cross_sections(self) -> None:
        bins = AnalysisBinning([1.0, 1.5], [0.2, 0.3], [0.2, 0.3], [0.0, 180.0, 360.0])
        with tempfile.TemporaryDirectory() as tmp:
            born = Path(tmp) / "born.txt"
            rad = Path(tmp) / "rad.txt"
            born.write_text(_lund_event(pi0=False), encoding="utf-8")
            rad.write_text(_lund_event(pi0=True), encoding="utf-8")
            result = compute_radiative_correction(
                born,
                rad,
                bins,
                beam_energy=6.535,
                min_counts=1,
                chunk_size=1,
                born_integrated_cross_section=2.0,
                radiative_integrated_cross_section=1.0,
            )
        np.testing.assert_allclose(result.c_rad[result.reliable], [0.5])
        self.assertEqual(result.normalization_ratio, 0.5)

    def test_generator_integrated_cross_section_parser_reads_norm_and_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            norm = Path(tmp) / "aao_norad.norm"
            summary = Path(tmp) / "aao_rad.sum"
            norm.write_text("sig_sum=3.25D+00\n", encoding="utf-8")
            summary.write_text(
                " Integrated cross section = 1.0 2.5 micro-barns\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_generator_integrated_cross_section(norm), 3.25)
            self.assertEqual(_read_generator_integrated_cross_section(summary), 2.5)
            nested = Path(tmp) / "norms"
            nested.mkdir()
            (nested / "job1.norm").write_text(
                "\n".join([
                    "generator=aao_norad",
                    "integrated_cross_section_units=micro-barns",
                    "sig_int=1.5",
                    "sig_sum=2.0",
                    "events=100",
                    "ntries=1000",
                    "nevent=100",
                    "mcall_max=500",
                    "sigr_max=3.0",
                    "",
                ]),
                encoding="utf-8",
            )
            (nested / "job2.norm").write_text(
                "sig_sum=4.0\nevents=300\n", encoding="utf-8"
            )
            self.assertEqual(_read_generator_integrated_cross_section(nested), 3.5)
            summary = _read_generator_normalization_summary(nested)
            self.assertEqual(summary.integrated_cross_section, 3.5)
            self.assertEqual(summary.method, "events_weighted_mean_sig_sum")
            self.assertEqual(len(summary.records), 2)
            self.assertEqual(summary.records[0].generator, "aao_norad")
            self.assertEqual(summary.records[0].units, "micro-barns")
            self.assertEqual(summary.records[0].sig_int, 1.5)
            self.assertEqual(summary.records[0].ntries, 1000)
            npz_fields = _normalization_npz_fields("born", summary)
            np.testing.assert_allclose(npz_fields["born_normalization_sig_sum"], [2.0, 4.0])
            np.testing.assert_allclose(npz_fields["born_normalization_events"], [100.0, 300.0])
            np.testing.assert_allclose(
                npz_fields["born_normalization_ntries"],
                [1000.0, np.nan],
                equal_nan=True,
            )
            self.assertEqual(npz_fields["born_normalization_generators"][0], "aao_norad")

            legacy = Path(tmp) / "legacy"
            legacy.mkdir()
            (legacy / "job1.sum").write_text("sig_sum=2.0\n", encoding="utf-8")
            (legacy / "job2.sum").write_text("sig_sum=4.0\n", encoding="utf-8")
            self.assertEqual(_read_generator_integrated_cross_section(legacy), 3.0)

            (nested / "job1.sum").write_text("sig_sum=200.0\n", encoding="utf-8")
            self.assertEqual(_read_generator_integrated_cross_section(nested), 3.5)

            mixed = Path(tmp) / "mixed"
            mixed.mkdir()
            (mixed / "job1.norm").write_text(
                "sig_sum=2.0\nevents=100\n", encoding="utf-8"
            )
            (mixed / "job2.norm").write_text("sig_sum=4.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mixed weighted and unweighted"):
                _read_generator_integrated_cross_section(mixed)

    def test_support_status_codes_describe_unsupported_bins(self) -> None:
        born = np.array([10.0, 0.0, 4.0, 4.0, 6.0, 0.0, 4.0])
        rad = np.array([10.0, 0.0, 0.0, 6.0, 4.0, 7.0, 4.0])
        np.testing.assert_array_equal(
            support_status_codes(born, rad, min_counts=5),
            [0, 1, 2, 4, 5, 3, 6],
        )


class NormalizationTests(unittest.TestCase):
    def test_luminosity_and_flux_are_positive(self) -> None:
        self.assertGreater(integrated_luminosity_fb(1.0e-3), 0.0)
        flux = virtual_photon_flux(np.array([2.0]), np.array([0.3]), 6.535)
        self.assertGreater(flux[0], 0.0)

    def test_physical_bin_volume_includes_exclusive_t_overlap(self) -> None:
        bins = AnalysisBinning(
            [1.49, 1.51],
            [0.295, 0.305],
            [0.0, 2.0],
            [0.0, 360.0],
        )
        volumes = physical_bin_volumes(
            bins,
            6.535,
            q2_minimum=0.0,
            w_minimum=1.0,
            electron_p_minimum=0.0,
            integration_points=100,
        )
        rectangular = 0.02 * 0.01 * 2.0 * 2.0 * np.pi
        self.assertGreater(volumes[0, 0, 0, 0], 0.0)
        self.assertLess(volumes[0, 0, 0, 0], rectangular)

    def test_cross_section_uses_bin_centering_reference_for_flux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            config_path = tmpdir / "config.json"
            unfolding_path = tmpdir / "unfolding.npz"
            centering_path = tmpdir / "C_BC.npz"
            output_path = tmpdir / "cross_section.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "beam_energy": 6.535,
                        "target_length_cm": 5.0,
                        "target_density_g_cm3": 0.071,
                        "target_molar_mass_g": 1.008,
                        "pi0_to_gg_branching_ratio": 0.988,
                        "minimum_acceptance": 0.005,
                        "phase_space": {
                            "Q2_min": 1.0,
                            "W_min": 2.0,
                            "electron_p_min": 1.0,
                        },
                        "binning": {
                            "Q2": [1.2, 1.4],
                            "xB": [0.2, 0.3],
                            "minus_t": [0.2, 0.3],
                            "phi_deg": [0.0, 180.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                unfolding_path,
                beam_charge_c=1.0e-3,
                efficiency=np.asarray([1.0]),
                corrected_yield=np.asarray([100.0]),
                corrected_uncertainty=np.asarray([10.0]),
                Q2_mean=np.asarray([1.35]),
                xB_mean=np.asarray([0.28]),
                current_efficiency_applied=np.asarray(True),
                current_efficiency_D_reference=np.asarray(0.98),
                current_efficiency_model_json=np.asarray('{"schema_version": 1}'),
            )
            shape = (1, 1, 1, 1)
            np.savez_compressed(
                centering_path,
                C_BC=np.full(shape, 2.0),
                reliable=np.ones(shape, dtype=bool),
                q2_center=np.full(shape, 1.25),
                xB_center=np.full(shape, 0.24),
            )
            args = argparse.Namespace(
                unfolding_result=unfolding_path,
                config=config_path,
                output=output_path,
                bin_centering=centering_path,
                global_normalization=1.0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                command_cross_section(args)
            output = np.load(output_path, allow_pickle=False)

        flux = virtual_photon_flux(np.asarray([1.25]), np.asarray([0.24]), 6.535)[0]
        expected = (
            100.0
            * 1.0e-6
            / (float(output["luminosity_fb"]) * 0.988 * output["bin_volume"].item() * flux)
            / 2.0
        )
        self.assertAlmostEqual(output["reduced_cross_section"].item(), expected)
        self.assertEqual(output["flux_q2_mean"].item(), 1.25)
        self.assertEqual(output["flux_xb_mean"].item(), 0.24)
        self.assertEqual(output["uncentered_q2_mean"].item(), 1.35)
        self.assertEqual(output["uncentered_xb_mean"].item(), 0.28)
        self.assertTrue(output["current_efficiency_applied"].item())
        self.assertEqual(output["current_efficiency_D_reference"].item(), 0.98)
        self.assertEqual(
            output["current_efficiency_model_json"].item(), '{"schema_version": 1}'
        )


class BinCenteringTests(unittest.TestCase):
    def test_physical_mask_uses_signed_t_internally(self) -> None:
        self.assertTrue(physical_mask(np.array([0.3]), np.array([1.5]), np.array([-0.2]), 6.535)[0])
        self.assertFalse(physical_mask(np.array([0.3]), np.array([1.5]), np.array([0.2]), 6.535)[0])

    def test_flat_d4sigma_gives_unit_bin_centering(self) -> None:
        bins = AnalysisBinning([1.2, 1.4], [0.25, 0.35], [0.15, 0.25], [0.0, 180.0, 360.0])

        def flat_d4sigma(points: np.ndarray) -> np.ndarray:
            flux = virtual_photon_flux(points[:, 1], points[:, 0], 6.535)
            return np.divide(1.0, flux, out=np.full(points.shape[0], np.nan), where=flux > 0.0)

        result = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2)
        self.assertEqual(result.c_bc.shape, bins.shape)
        self.assertTrue(np.all(result.reliable))
        self.assertTrue(np.all(result.computed))
        np.testing.assert_allclose(result.c_bc, 1.0, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(result.n_physical > 0))

    def test_bin_centering_phase_space_restricts_physical_cells(self) -> None:
        bins = AnalysisBinning([1.2, 1.4], [0.25, 0.35], [0.15, 0.25], [0.0, 180.0])

        def flat_d4sigma(points: np.ndarray) -> np.ndarray:
            flux = virtual_photon_flux(points[:, 1], points[:, 0], 6.535)
            return np.divide(1.0, flux, out=np.full(points.shape[0], np.nan), where=flux > 0.0)

        unrestricted = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2)
        restricted = compute_bin_centering(
            bins,
            6.535,
            flat_d4sigma,
            samples_per_dimension=2,
            phase_space=AnalysisPhaseSpace(electron_p_min=4.2),
        )
        self.assertGreater(int(unrestricted.n_physical.sum()), int(restricted.n_physical.sum()))
        self.assertGreater(int(restricted.n_physical.sum()), 0)

    def test_partial_bin_centering_merge_matches_full_result(self) -> None:
        bins = AnalysisBinning([1.2, 1.3, 1.4], [0.25, 0.35], [0.15, 0.25], [0.0, 180.0])

        def flat_d4sigma(points: np.ndarray) -> np.ndarray:
            flux = virtual_photon_flux(points[:, 1], points[:, 0], 6.535)
            return np.divide(1.0, flux, out=np.full(points.shape[0], np.nan), where=flux > 0.0)

        full = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2)
        first = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2, bin_start=0, bin_stop=1)
        second = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2, bin_start=1, bin_stop=2)
        self.assertLess(np.count_nonzero(first.computed), full.computed.size)
        self.assertLess(np.count_nonzero(second.computed), full.computed.size)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            first_path = tmpdir / "bc_0.npz"
            second_path = tmpdir / "bc_1.npz"
            merged_path = tmpdir / "bc_merged.npz"
            _write_bin_centering_test_artifact(first_path, bins, first, bin_start=0, bin_stop=1)
            _write_bin_centering_test_artifact(second_path, bins, second, bin_start=1, bin_stop=2)
            with contextlib.redirect_stdout(io.StringIO()):
                command_bin_centering_merge(argparse.Namespace(partials=[first_path, second_path], output=merged_path))
            merged = np.load(merged_path, allow_pickle=False)
            np.testing.assert_allclose(merged["C_BC"], full.c_bc)
            np.testing.assert_array_equal(merged["reliable"], full.reliable)
            np.testing.assert_array_equal(merged["computed"], full.computed)

    def test_plot_artifact_scan_deduplicates_symlinks_and_sorts_n(self) -> None:
        bins = AnalysisBinning([1.2, 1.4], [0.25, 0.35], [0.15, 0.25], [0.0, 180.0])

        def flat_d4sigma(points: np.ndarray) -> np.ndarray:
            flux = virtual_photon_flux(points[:, 1], points[:, 0], 6.535)
            return np.divide(1.0, flux, out=np.full(points.shape[0], np.nan), where=flux > 0.0)

        result = compute_bin_centering(bins, 6.535, flat_d4sigma, samples_per_dimension=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            n2 = tmpdir / "C_BC_rgk_N2.npz"
            n4 = tmpdir / "C_BC_rgk_N4.npz"
            alias = tmpdir / "C_BC_N2.npz"
            _write_bin_centering_test_artifact(n2, bins, result, bin_start=0, bin_stop=1, n_value=2)
            _write_bin_centering_test_artifact(n4, bins, result, bin_start=0, bin_stop=1, n_value=4)
            alias.symlink_to(n2.name)
            scan = _load_bin_centering_plot_artifacts(alias, overlay_directory=tmpdir)
            self.assertEqual(list(scan), [2, 4])
            np.testing.assert_array_equal(scan[2]["valid"], result.reliable & result.computed)


def _write_bin_centering_test_artifact(
    path: Path,
    bins: AnalysisBinning,
    result,
    *,
    bin_start: int,
    bin_stop: int,
    n_value: int = 2,
) -> None:
    np.savez_compressed(
        path,
        C_BC=result.c_bc,
        reliable=result.reliable,
        computed=result.computed,
        average_d4sigma=result.average_d4sigma,
        center_d4sigma=result.center_d4sigma,
        xB_center=result.xB_center,
        q2_center=result.q2_center,
        minus_t_center=result.minus_t_center,
        phi_center=result.phi_center,
        n_physical=result.n_physical,
        n_valid=result.n_valid,
        n_failed=result.n_failed,
        physical_fraction=result.physical_fraction,
        failure_fraction=result.failure_fraction,
        q2_edges=bins.q2_edges,
        xb_edges=bins.xb_edges,
        t_edges=bins.t_edges,
        phi_edges=bins.phi_edges,
        beam_energy=6.535,
        samples_per_dimension=n_value,
        max_failure_fraction=0.0,
        theory=5,
        channel=1,
        resonance=0,
        bin_start=bin_start,
        bin_stop=bin_stop,
        total_3d_bins=2,
    )


def _lund_event(*, pi0: bool) -> str:
    electron_p = 4.0
    electron_theta = 0.2
    electron = _particle_row(11, electron_p, electron_theta, 0.0, 0.00051099895)
    proton = _particle_row(2212, 0.5, 0.6, 1.0, 0.9382720813)
    if pi0:
        meson = _particle_row(111, 0.8, 0.4, 2.0, 0.1349768)
        photon = _particle_row(22, 0.1, 0.5, -1.0, 0.0)
        particles = [electron, proton, meson, photon]
    else:
        photon_one = _particle_row(22, 0.45, 0.4, 2.0, 0.0)
        photon_two = _particle_row(22, 0.35, 0.5, -1.0, 0.0)
        particles = [electron, proton, photon_one, photon_two]
    return "4 0 0 0 0 0 0 0 0 0\n" + "".join(particles)


def _particle_row(pid: int, p: float, theta: float, phi: float, mass: float) -> str:
    px = p * np.sin(theta) * np.cos(phi)
    py = p * np.sin(theta) * np.sin(phi)
    pz = p * np.cos(theta)
    energy = np.sqrt(p * p + mass * mass)
    return f"1 0 0 {pid} 0 0 {px:.12g} {py:.12g} {pz:.12g} {energy:.12g}\n"


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
