from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from scipy.sparse import diags


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.binning import AnalysisBinning
from eppi0.closure import (
    SplitClosureInputs,
    deterministic_folds,
    run_closure_scan,
    stress_weights,
    training_response,
)
from eppi0.phase_space import AnalysisPhaseSpace
from validate_response_closure import render_saved_diagnostics, save_results
from validate_response_closure import load_split_inputs, save_split_inputs


class ClosureTests(unittest.TestCase):
    def test_deterministic_folds_are_stable_and_source_aware(self) -> None:
        source = np.repeat(np.arange(4, dtype=np.uint64), 100)
        event = np.tile(np.arange(100, dtype=np.uint64), 4)
        first = deterministic_folds(source, event, 5, seed=19)
        second = deterministic_folds(source, event, 5, seed=19)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first >= 0) & (first < 5)))
        self.assertGreater(np.unique(first).size, 1)
        changed_source = deterministic_folds(source + 1, event, 5, seed=19)
        self.assertGreater(np.count_nonzero(first != changed_source), 0)

    def test_stress_weights_are_positive_and_nontrivial(self) -> None:
        values = stress_weights(
            q2=np.array([1.0, 2.0, 3.0]),
            xb=np.array([0.1, 0.3, 0.6]),
            minus_t=np.array([0.1, 0.5, 1.5]),
            phi_rad=np.array([0.0, np.pi / 2.0, np.pi]),
            names=("nominal", "q2_tilt", "xb_t_tilt", "phi_harmonic", "combined"),
            strength=0.6,
            q2_range=(1.0, 3.0),
            xb_range=(0.1, 0.6),
            t_range=(0.1, 1.5),
        )
        self.assertEqual(values.shape, (5, 3))
        self.assertTrue(np.all(values > 0.0))
        np.testing.assert_allclose(values[0], 1.0)
        self.assertFalse(np.allclose(values[1], values[0]))
        self.assertFalse(np.allclose(values[3], values[0]))

    def test_identity_response_closes_exactly(self) -> None:
        binning = AnalysisBinning(
            [1, 2], [0.1, 0.5], [0.1, 1.0], [0, 60, 120, 180, 240, 300, 360]
        )
        folds = 3
        truth = np.tile(
            np.array([100.0, 120.0, 140.0, 160.0, 140.0, 120.0]), (folds, 1)
        )
        accepted = 0.5 * truth
        migrations = tuple(diags(accepted[index], format="csr") for index in range(folds))
        inputs = SplitClosureInputs(
            fold_truth_total=truth,
            fold_reconstructed_total=accepted,
            fold_feed_counts=np.zeros_like(truth),
            fold_migration_counts=migrations,
            validation_truth=truth[None, ...],
            validation_measured=accepted[None, ...],
            validation_variance=accepted[None, ...],
            stress_names=("nominal",),
        )
        response = training_response(inputs, 1)
        np.testing.assert_allclose(response.efficiency, 0.5)
        result = run_closure_scan(
            inputs,
            binning,
            iterations=(0, 1, 4),
            minimum_acceptance=0.01,
            minimum_truth=1.0,
            bootstrap=0,
            minimum_harmonic_points=4,
        )
        expected = np.repeat(inputs.validation_truth[:, None, ...], 3, axis=1)
        np.testing.assert_allclose(result.unfolded, expected)
        self.assertEqual(result.recommended_iterations, 0)
        for row in result.metrics:
            self.assertAlmostEqual(float(row["normalized_mse"]), 0.0)
            self.assertAlmostEqual(float(row["global_relative_bias"]), 0.0)
            self.assertAlmostEqual(float(row["refold_chi2_ndf"]), 0.0)
            self.assertEqual(int(row["harmonic_common_cells"]), 1)

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config = temporary_path / "analysis.json"
            config.write_text("{}\n", encoding="utf-8")
            args = Namespace(
                label="synthetic closure",
                converter_root=temporary_path / "converter.root",
                selected_root=temporary_path / "selected.root",
                split_input_dir=None,
                config=config,
                selection_mask=None,
                dictionary=None,
                folds=folds,
                seed=731_921,
                bootstrap=0,
                response_uncertainty="analytic-diagonal",
                minimum_truth=1.0,
                minimum_harmonic_points=4,
                stress_strength=0.6,
                topology_group=[],
            )
            save_results(
                temporary_path,
                result,
                inputs,
                binning,
                args=args,
                scan_metadata={"generated_rows": 18},
                phase_space=AnalysisPhaseSpace(),
                minimum_acceptance=0.01,
            )
            summary = json.loads(
                (temporary_path / "closure_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["recommendation"]["iterations"], 0)
            self.assertTrue((temporary_path / "closure_metrics.csv").is_file())
            self.assertTrue((temporary_path / "closure_results.npz").is_file())

            diagnostics = temporary_path / "closure_diagnostics.pdf"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                render_saved_diagnostics(
                    temporary_path,
                    output=diagnostics,
                    label=(
                        "synthetic topology-integrated response closure with a "
                        "deliberately long campaign label"
                    ),
                )
            collapsed = [
                warning
                for warning in caught
                if "axes sizes collapsed to zero" in str(warning.message)
            ]
            self.assertEqual(collapsed, [])
            self.assertTrue(diagnostics.is_file())
            self.assertGreater(diagnostics.stat().st_size, 10_000)

            save_split_inputs(temporary_path, inputs, {"generated_rows": 18})
            restored, metadata = load_split_inputs(temporary_path)
            np.testing.assert_allclose(restored.fold_truth_total, inputs.fold_truth_total)
            np.testing.assert_allclose(
                restored.fold_reconstructed_total, inputs.fold_reconstructed_total
            )
            np.testing.assert_allclose(
                restored.fold_migration_counts[0].toarray(),
                inputs.fold_migration_counts[0].toarray(),
            )
            self.assertEqual(restored.stress_names, inputs.stress_names)
            self.assertEqual(metadata["generated_rows"], 18)

    def test_fold_jackknife_response_covariance_inflates_varying_response(self) -> None:
        binning = AnalysisBinning(
            [1, 2], [0.1, 0.5], [0.1, 1.0], [0, 90, 180, 270, 360]
        )
        truth = np.full((5, 4), 200.0)
        efficiencies = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
        accepted = efficiencies[:, None] * truth
        migrations = tuple(diags(row, format="csr") for row in accepted)
        measured = np.full((1, 5, 4), 100.0)
        inputs = SplitClosureInputs(
            fold_truth_total=truth,
            fold_reconstructed_total=accepted,
            fold_feed_counts=np.zeros_like(truth),
            fold_migration_counts=migrations,
            validation_truth=np.full((1, 5, 4), 200.0),
            validation_measured=measured,
            validation_variance=measured,
            stress_names=("nominal",),
        )
        analytic = run_closure_scan(
            inputs,
            binning,
            iterations=(1,),
            minimum_acceptance=0.01,
            minimum_truth=1.0,
            bootstrap=0,
            minimum_harmonic_points=4,
            response_uncertainty="analytic-diagonal",
        )
        jackknife = run_closure_scan(
            inputs,
            binning,
            iterations=(1,),
            minimum_acceptance=0.01,
            minimum_truth=1.0,
            bootstrap=0,
            minimum_harmonic_points=4,
            response_uncertainty="fold-jackknife",
        )
        self.assertGreater(
            float(np.nanmedian(jackknife.uncertainty)),
            float(np.nanmedian(analytic.uncertainty)),
        )


if __name__ == "__main__":
    unittest.main()
