from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.data_efficiency import (
    aggregate_current_groups,
    attach_relative_efficiencies,
    build_run_yields,
    fit_linear_yield,
    fit_shared_fractional_yield,
    load_selection_mask,
)
from eppi0.current_efficiency import (
    RelativeLinearEfficiency,
    correction_artifact,
    load_current_efficiency_correction,
)
from eppi0.gemc_efficiency import (
    attach_relative_gemc_efficiencies,
    fit_linear_efficiency,
    load_gemc_efficiencies,
)
from study_data_efficiency import main as study_main


class DataEfficiencyTests(unittest.TestCase):
    def write_inputs(self, directory: Path) -> tuple[Path, Path]:
        run_specs = [
            (1001, "L5", 5.0, 95),
            (1002, "L5", 5.0, 95),
            (2001, "P4", 35.0, 65),
            (2002, "P4", 35.0, 65),
            (3001, "P3", 60.0, 40),
            (3002, "P3", 60.0, 40),
        ]
        event_runs = np.concatenate(
            [np.full(events, run, dtype=np.int32) for run, _, _, events in run_specs]
        )
        charge_runs = np.asarray([spec[0] for spec in run_specs], dtype=np.int32)
        charge_c = np.full(charge_runs.size, 1.0e-9)
        sample_path = directory / "data.npz"
        np.savez_compressed(
            sample_path,
            run=event_runs,
            rec_selected=np.ones(event_runs.size, dtype=bool),
            rec_m_gg=np.full(event_runs.size, 0.135),
            rec_proton_detector=np.ones(event_runs.size, dtype=np.int32),
            rec_ft_photon_count=np.zeros(event_runs.size, dtype=np.int32),
            beam_charge_run=charge_runs,
            beam_charge_by_run_c=charge_c,
            beam_charge_c=np.asarray(charge_c.sum()),
            run_total_events=np.full(charge_runs.size, 1000),
            run_passed_qadb_events=np.full(charge_runs.size, 900),
            run_failed_qadb_events=np.full(charge_runs.size, 100),
        )
        manifest_path = directory / "currents.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset": {"run_group": "TEST"},
                    "runs": {
                        str(run): {
                            "run_class": run_class,
                            "nominal_current_nA": current,
                            "rcdb_current_nA": current,
                            "rcdb_quality": "unflagged",
                        }
                        for run, run_class, current, _ in run_specs
                    }
                }
            ),
            encoding="utf-8",
        )
        return sample_path, manifest_path

    def test_group_fit_recovers_zero_current_yield(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample, manifest = self.write_inputs(Path(tmp))
            records, validation = build_run_yields(
                sample,
                manifest,
                include_classes=("L5", "P4", "P3"),
            )
            groups = aggregate_current_groups(records)
            fit = fit_linear_yield(records, groups)
            attach_relative_efficiencies(groups, fit)

        self.assertEqual(len(groups), 3)
        self.assertEqual(validation["signal_events"], 400)
        self.assertAlmostEqual(validation["charge_difference_c"], 0.0)
        self.assertAlmostEqual(fit.intercept_events_per_nC, 100.0)
        self.assertAlmostEqual(fit.slope_events_per_nC_per_nA, -1.0)
        self.assertEqual(fit.ndf, 1)
        self.assertAlmostEqual(groups[0].relative_efficiency, 0.95)

    def test_shared_fractional_slope_has_separate_period_intercepts(self) -> None:
        beta = -0.0025
        specifications = (
            ("E5", "early", 5.0, 100.0),
            ("E30", "early", 30.0, 100.0),
            ("E55", "early", 55.0, 100.0),
            ("L5", "late", 5.0, 125.0),
            ("L35", "late", 35.0, 125.0),
            ("L60", "late", 60.0, 125.0),
        )
        groups = [
            SimpleNamespace(
                group=run_class,
                effective_current_nA=current,
                yield_events_per_nC=intercept * (1.0 + beta * current),
                statistical_uncertainty_events_per_nC=0.25,
            )
            for run_class, _, current, intercept in specifications
        ]
        fit = fit_shared_fractional_yield(
            [],
            groups,
            period_classes={
                "early": ["E5", "E30", "E55"],
                "late": ["L5", "L35", "L60"],
            },
        )

        self.assertEqual(fit.fit_model, "shared_fractional_slope_separate_intercepts")
        self.assertAlmostEqual(fit.period_intercepts_events_per_nC["early"], 100.0)
        self.assertAlmostEqual(fit.period_intercepts_events_per_nC["late"], 125.0)
        self.assertAlmostEqual(fit.fractional_slope_per_nA, beta)
        self.assertEqual(fit.ndf, 3)
        self.assertEqual(np.asarray(fit.covariance).shape, (3, 3))
        efficiency, uncertainty = fit.relative_efficiency(60.0)
        self.assertAlmostEqual(efficiency, 0.85)
        self.assertGreater(uncertainty, 0.0)
        self.assertAlmostEqual(float(fit.predict(20.0, period="late")), 118.75)

    def test_default_style_filter_excludes_low_current_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample, manifest = self.write_inputs(Path(tmp))
            records, _ = build_run_yields(sample, manifest)
            included = [record for record in records if record.included]
            excluded_l5 = [record for record in records if record.run_class == "L5"]

        self.assertEqual({record.run_class for record in included}, {"P3", "P4"})
        self.assertTrue(
            all("run_class_not_included" in row.exclusion_reason for row in excluded_l5)
        )

    def test_fixed_mask_changes_signal_not_candidate_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                size = data["run"].size
            mask = np.zeros(size, dtype=bool)
            mask[::2] = True
            mask_path = directory / "selection.npy"
            np.save(mask_path, mask)
            records, validation = build_run_yields(
                sample,
                manifest,
                selection_mask_path=mask_path,
                include_classes=("L5", "P4", "P3"),
            )

        self.assertEqual(sum(record.candidate_events for record in records), size)
        self.assertEqual(sum(record.signal_events for record in records), int(mask.sum()))
        self.assertEqual(validation["signal_events"], int(mask.sum()))

    def test_minimum_group_charge_fraction_excludes_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                arrays = {key: data[key] for key in data.files}
            charges = np.asarray(arrays["beam_charge_by_run_c"], dtype=float)
            charges[0] = 1.0e-12
            arrays["beam_charge_by_run_c"] = charges
            arrays["beam_charge_c"] = np.asarray(charges.sum())
            np.savez_compressed(sample, **arrays)

            records, validation = build_run_yields(
                sample,
                manifest,
                include_classes=("L5", "P4", "P3"),
                minimum_group_charge_fraction=0.01,
            )
            groups = aggregate_current_groups(records)

        low_charge = next(record for record in records if record.run == 1001)
        self.assertFalse(low_charge.included)
        self.assertIn("below_minimum_group_charge_fraction", low_charge.exclusion_reason)
        self.assertAlmostEqual(low_charge.group_charge_fraction, 1.0 / 1001.0)
        self.assertEqual(validation["runs_below_minimum_group_charge_fraction"], [1001])
        l5 = next(group for group in groups if group.group == "L5")
        self.assertEqual(l5.run_numbers, [1002])

    def test_default_low_yield_filter_excludes_five_sigma_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                event_runs = np.asarray(data["run"])
            mask = np.ones(event_runs.size, dtype=bool)
            low_run_rows = np.flatnonzero(event_runs == 1001)
            mask[low_run_rows[1:]] = False
            mask_path = directory / "selection.npy"
            np.save(mask_path, mask)

            records, validation = build_run_yields(
                sample,
                manifest,
                selection_mask_path=mask_path,
                include_classes=("L5", "P4", "P3"),
            )
            groups = aggregate_current_groups(records)

        low_run = next(record for record in records if record.run == 1001)
        self.assertFalse(low_run.included)
        self.assertLess(low_run.group_yield_pull, -5.0)
        self.assertIn("below_group_mean_yield_threshold", low_run.exclusion_reason)
        self.assertEqual(validation["low_yield_sigma_threshold"], 5.0)
        self.assertEqual(validation["runs_below_group_yield_threshold"], [1001])
        l5 = next(group for group in groups if group.group == "L5")
        self.assertEqual(l5.run_numbers, [1002])

    def test_zero_low_yield_threshold_disables_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                event_runs = np.asarray(data["run"])
            mask = np.ones(event_runs.size, dtype=bool)
            low_run_rows = np.flatnonzero(event_runs == 1001)
            mask[low_run_rows[1:]] = False
            mask_path = directory / "selection.npy"
            np.save(mask_path, mask)
            records, validation = build_run_yields(
                sample,
                manifest,
                selection_mask_path=mask_path,
                include_classes=("L5", "P4", "P3"),
                low_yield_sigma_threshold=0.0,
            )

        self.assertTrue(next(record for record in records if record.run == 1001).included)
        self.assertEqual(validation["runs_below_group_yield_threshold"], [])

    def test_background_subtracted_run_yields_use_signed_sideband_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                event_runs = np.asarray(data["run"])
            signal = np.zeros(event_runs.size, dtype=bool)
            sideband = np.zeros(event_runs.size, dtype=bool)
            run_rows = np.flatnonzero(event_runs == 1001)
            signal[run_rows[:10]] = True
            sideband[run_rows[10:14]] = True
            net_weights = signal.astype(float) - 0.25 * sideband.astype(float)
            background = SimpleNamespace(
                signal_region_mask=signal,
                sideband_mask=sideband,
                net_event_weights=net_weights,
                group_ids=np.asarray([4]),
                signal_lower=np.asarray([0.11]),
                signal_upper=np.asarray([0.16]),
                fit_lower=np.asarray([0.08]),
                fit_upper=np.asarray([0.20]),
                alpha=np.asarray([0.25]),
                alpha_uncertainty=np.asarray([0.02]),
                fit_model=np.asarray(["gaussian+sideband-linear"]),
                fit_entries=np.asarray([event_runs.size]),
            )
            cuts = SimpleNamespace(variables=("rec_m_gg",))
            with (
                patch("eppi0.data_efficiency.load_cuts", return_value=cuts),
                patch(
                    "eppi0.data_efficiency.estimate_mgg_background",
                    return_value=background,
                ),
            ):
                records, validation = build_run_yields(
                    sample,
                    manifest,
                    include_classes=("L5", "P4", "P3"),
                    background_cuts_path=directory / "cuts.npz",
                )
            groups = aggregate_current_groups(records)

        run = next(record for record in records if record.run == 1001)
        self.assertEqual(validation["yield_mode"], "mgg_sideband_subtracted")
        self.assertEqual(run.signal_region_events, 10)
        self.assertEqual(run.sideband_events, 4)
        self.assertAlmostEqual(run.estimated_background_events, 1.0)
        self.assertAlmostEqual(run.signal_events, 9.0)
        self.assertAlmostEqual(run.signal_statistical_variance, 10.25)
        l5 = next(group for group in groups if group.group == "L5")
        self.assertAlmostEqual(l5.signal_events, 9.0)
        self.assertAlmostEqual(l5.signal_statistical_variance, 10.25)

    def test_gemc_response_metadata_fit_and_relative_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            edges = {
                "q2_edges": np.asarray([1.0, 2.0, 3.0]),
                "xb_edges": np.asarray([0.1, 0.2]),
                "t_edges": np.asarray([0.1, 0.2]),
                "phi_edges": np.asarray([0.0, 180.0]),
            }
            truth = np.asarray([100.0, 100.0])
            no_background = directory / "response_0.npz"
            merged = directory / "response_60.npz"
            np.savez_compressed(
                no_background,
                truth_total=truth,
                efficiency=np.asarray([0.8, 0.8]),
                **edges,
            )
            np.savez_compressed(
                merged,
                truth_total=truth,
                efficiency=np.asarray([0.68, 0.68]),
                **edges,
            )
            manifest = directory / "gemc.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "label": "no_background",
                                "current_nA": 0.0,
                                "response_meta": no_background.name,
                            },
                            {
                                "label": "merged_60nA",
                                "current_nA": 60.0,
                                "response_meta": merged.name,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            points, validation = load_gemc_efficiencies(manifest)
            fit = fit_linear_efficiency(points)
            attach_relative_gemc_efficiencies(points, fit)

        self.assertTrue(validation["truth_totals_match_reference"])
        self.assertAlmostEqual(fit.intercept, 0.8)
        self.assertAlmostEqual(fit.slope_per_nA, -0.002)
        self.assertAlmostEqual(points[1].relative_efficiency, 0.85)
        self.assertEqual(fit.ndf, 0)

    def test_missing_current_run_must_be_excluded_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            response_meta = directory / "response_meta.npz"
            np.savez_compressed(response_meta, efficiency=np.ones(1))
            model = RelativeLinearEfficiency(
                1.0, -0.001, ((0.0, 0.0), (0.0, 0.0))
            )
            records = [
                SimpleNamespace(
                    run=1001,
                    current_nA=50.0,
                    charge_c=1.0e-9,
                    run_class="P3",
                    current_quality="unflagged",
                    included=True,
                    exclusion_reason="",
                ),
                SimpleNamespace(
                    run=1002,
                    current_nA=None,
                    charge_c=0.2e-9,
                    run_class="E2",
                    current_quality="missing",
                    included=False,
                    exclusion_reason="missing_current",
                ),
            ]
            with self.assertRaisesRegex(ValueError, "1002.*no usable current"):
                correction_artifact(
                    data_model=model,
                    gemc_model=model,
                    reference_current_nA=50.0,
                    reference_label="merged_50nA",
                    reference_response_meta=response_meta,
                    run_records=records,
                    sources={},
                )
            payload = correction_artifact(
                data_model=model,
                gemc_model=model,
                reference_current_nA=50.0,
                reference_label="merged_50nA",
                reference_response_meta=response_meta,
                run_records=records,
                sources={},
                analysis_excluded_classes=("E2",),
            )
            artifact = directory / "correction.json"
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_current_efficiency_correction(artifact)

        self.assertEqual(loaded.excluded_runs, (1002,))
        self.assertEqual(float(loaded.event_weights(np.asarray([1002]))[0]), 0.0)
        self.assertAlmostEqual(loaded.original_beam_charge_c, 1.2e-9)
        self.assertAlmostEqual(loaded.analysis_beam_charge_c, 1.0e-9)

    def test_selected_run_without_charge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["run"] = np.append(arrays["run"], 9999)
            broken = directory / "broken.npz"
            np.savez_compressed(broken, **arrays)
            with self.assertRaisesRegex(ValueError, "9999"):
                build_run_yields(broken, manifest)

    def test_mask_shape_and_values_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            wrong_shape = directory / "wrong.npy"
            np.save(wrong_shape, np.ones(3, dtype=bool))
            invalid_values = directory / "invalid.npy"
            np.save(invalid_values, np.array([0, 2]))
            with self.assertRaisesRegex(ValueError, "expected"):
                load_selection_mask(wrong_shape, 4)
            with self.assertRaisesRegex(ValueError, "0/1"):
                load_selection_mask(invalid_values, 2)

    def test_cli_writes_tables_summary_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sample, manifest = self.write_inputs(directory)
            with np.load(sample) as data:
                arrays = {key: data[key] for key in data.files}
            keep = np.asarray(arrays["run"]) != 1001
            first_low_run_event = np.flatnonzero(np.asarray(arrays["run"]) == 1001)[0]
            keep[first_low_run_event] = True
            event_size = np.asarray(arrays["run"]).size
            for key, values in list(arrays.items()):
                values = np.asarray(values)
                if values.ndim == 1 and values.size == event_size:
                    arrays[key] = values[keep]
            np.savez_compressed(sample, **arrays)
            output = directory / "output"
            edges = {
                "q2_edges": np.asarray([1.0, 2.0]),
                "xb_edges": np.asarray([0.1, 0.2]),
                "t_edges": np.asarray([0.1, 0.2]),
                "phi_edges": np.asarray([0.0, 360.0]),
            }
            response_0 = directory / "response_0.npz"
            response_60 = directory / "response_60.npz"
            np.savez_compressed(
                response_0,
                truth_total=np.asarray([1000.0]),
                efficiency=np.asarray([0.8]),
                **edges,
            )
            np.savez_compressed(
                response_60,
                truth_total=np.asarray([1000.0]),
                efficiency=np.asarray([0.68]),
                **edges,
            )
            gemc_manifest = directory / "gemc.json"
            gemc_manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "label": "no_background",
                                "current_nA": 0.0,
                                "response_meta": str(response_0),
                            },
                            {
                                "label": "merged_60nA",
                                "current_nA": 60.0,
                                "response_meta": str(response_60),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            arguments = [
                "study_data_efficiency.py",
                str(sample),
                "--manifest",
                str(manifest),
                "--include-classes",
                "L5",
                "P4",
                "P3",
                "--shared-slope-period",
                "early=L5",
                "--shared-slope-period",
                "production=P4,P3",
                "--exclude-run-downstream",
                "1002",
                "--minimum-group-charge-fraction",
                "0.001",
                "--gemc-manifest",
                str(gemc_manifest),
                "--output-dir",
                str(output),
            ]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                result = study_main()
            summary = json.loads((output / "fit_summary.json").read_text(encoding="utf-8"))
            correction = load_current_efficiency_correction(
                output / "current_efficiency_correction.json"
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                summary["study"],
                "TEST charge-normalized data yield versus beam current",
            )
            self.assertEqual(
                summary["fit"]["fit_model"],
                "shared_fractional_slope_separate_intercepts",
            )
            self.assertAlmostEqual(
                summary["fit"]["period_intercepts_events_per_nC"]["early"],
                100.0,
            )
            self.assertAlmostEqual(
                summary["fit"]["period_intercepts_events_per_nC"]["production"],
                100.0,
            )
            self.assertAlmostEqual(summary["fit"]["fractional_slope_per_nA"], -0.01)
            self.assertAlmostEqual(summary["gemc"]["fit"]["intercept"], 0.8)
            self.assertAlmostEqual(correction.reference_current_nA, 60.0)
            self.assertAlmostEqual(correction.d_reference, 0.4 / 0.85)
            weight, _ = correction.weights_for_currents(60.0)
            self.assertAlmostEqual(float(weight), 0.85 / 0.4)
            self.assertIn(1001, correction.run_currents_nA)
            self.assertEqual(correction.excluded_runs, (1001, 1002))
            self.assertEqual(float(correction.event_weights(np.asarray([1001]))[0]), 0.0)
            self.assertEqual(float(correction.event_weights(np.asarray([1002]))[0]), 0.0)
            self.assertAlmostEqual(correction.original_beam_charge_c, 6.0e-9)
            self.assertAlmostEqual(correction.analysis_beam_charge_c, 4.0e-9)
            self.assertEqual(
                summary["current_efficiency_correction"]["analysis_selection"][
                    "excluded_runs"
                ],
                [1001, 1002],
            )
            self.assertEqual(
                summary["filters"][
                    "automatic_low_yield_excluded_runs_downstream"
                ],
                [1001],
            )
            self.assertTrue((output / "run_yields.csv").is_file())
            self.assertTrue((output / "current_group_yields.csv").is_file())
            self.assertTrue((output / "gemc_efficiency_points.csv").is_file())
            self.assertTrue((output / "current_efficiency_correction.json").is_file())
            self.assertGreater((output / "data_efficiency_diagnostics.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
