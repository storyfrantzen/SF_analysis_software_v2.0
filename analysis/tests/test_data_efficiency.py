from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
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
    load_selection_mask,
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
            output = directory / "output"
            arguments = [
                "study_data_efficiency.py",
                str(sample),
                "--manifest",
                str(manifest),
                "--include-classes",
                "L5",
                "P4",
                "P3",
                "--output-dir",
                str(output),
            ]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                result = study_main()
            summary = json.loads((output / "fit_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(result, 0)
            self.assertAlmostEqual(summary["fit"]["intercept_events_per_nC"], 100.0)
            self.assertTrue((output / "run_yields.csv").is_file())
            self.assertTrue((output / "current_group_yields.csv").is_file())
            self.assertGreater((output / "data_efficiency_diagnostics.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
