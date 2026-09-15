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


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.run_conditions_audit import (
    RcdbRunConditions,
    RunChargeRecord,
    aggregate_run_charge_rows,
    build_outputs,
    parse_class_assignments,
    parse_qadb_misc,
    parse_qadb_misc_code,
    parse_run_specification,
    query_rcdb,
)
from audit_run_conditions import main as audit_main


class FakeProvider:
    values = {
        (1001, "beam_current"): 49.5,
        (1001, "target"): "LH2",
        (1001, "torus_scale"): 1.0,
        (1001, "run_config"): "/trigger/v1.cnf",
        (1002, "beam_current"): -0.01,
        (1002, "target"): "LH2",
        (1002, "torus_scale"): 1.0,
        (1002, "run_config"): "/trigger/v1.cnf",
    }

    def get_condition(self, run: int, name: str):
        return SimpleNamespace(value=self.values.get((run, name)))


class RunConditionsAuditTests(unittest.TestCase):
    def test_run_specification_and_class_assignments(self) -> None:
        self.assertEqual(
            parse_run_specification("1001, 1003-1005 1010:1011"),
            {1001, 1003, 1004, 1005, 1010, 1011},
        )
        self.assertEqual(
            parse_class_assignments(["P=1001-1002", "L=1003"]),
            {1001: "P", 1002: "P", 1003: "L"},
        )
        with self.assertRaisesRegex(ValueError, "assigned to both"):
            parse_class_assignments(["P=1001", "L=1001"])

    def test_rcdb_query_preserves_negative_current_for_audit(self) -> None:
        records = query_rcdb((1001, 1002), "unused", provider=FakeProvider())
        self.assertEqual(records[1001].beam_current_nA, 49.5)
        self.assertEqual(records[1002].beam_current_nA, -0.01)
        self.assertEqual(records[1001].run_config, "/trigger/v1.cnf")

    def test_qadb_code_parser_keeps_comments_separate(self) -> None:
        text = """misc_qa_runs = [
          1001,  # low luminosity scan
          1003,  # detector issue
        ]
        """
        self.assertEqual(
            parse_qadb_misc_code(text),
            {1001: "low luminosity scan", 1003: "detector issue"},
        )
        comments, covered = parse_qadb_misc(text)
        self.assertEqual(comments, {1001: "low luminosity scan", 1003: "detector issue"})
        self.assertIsNone(covered)

    def test_qadb_json_distinguishes_clean_and_untracked_runs(self) -> None:
        comments, covered = parse_qadb_misc(
            json.dumps(
                {
                    "1001": {"comments": [], "has_misc_bit": False},
                    "1002": {
                        "comments": ["low luminosity scan"],
                        "has_misc_bit": True,
                    },
                }
            )
        )
        self.assertEqual(comments, {1002: "low luminosity scan"})
        self.assertEqual(covered, {1001, 1002})

    def test_charge_rows_from_multiple_files_are_summed(self) -> None:
        records = aggregate_run_charge_rows(
            [
                RunChargeRecord(1001, 10.0, 100, 90, 10, 1),
                RunChargeRecord(1001, 20.0, 200, 180, 20, 1),
            ]
        )
        self.assertEqual(records[1001].accumulated_charge_nC, 30.0)
        self.assertEqual(records[1001].total_events, 300)
        self.assertEqual(records[1001].source_files, 2)

    def test_manifest_preserves_manual_metadata_and_marks_review_items(self) -> None:
        base = {
            "dataset": {"run_group": "OLD"},
            "run_classes": {
                "P": {"description": "production", "nominal_current_nA": 50.0}
            },
            "runs": {
                "1001": {
                    "run_class": "P",
                    "nominal_current_nA": 50.0,
                    "rcdb_quality": "suspect",
                    "notes": "manual review retained",
                }
            },
        }
        rcdb = query_rcdb((1001, 1002), "unused", provider=FakeProvider())
        charges = {
            1001: RunChargeRecord(1001, 10.0, 100, 90, 10, 1),
            1002: RunChargeRecord(1002, 0.0, 50, 0, 50, 1),
        }
        manifest, rows, summary = build_outputs(
            (1001, 1002),
            rcdb,
            charges,
            run_group="TEST",
            period="Fall 2018",
            beam_energy_gev=10.6,
            base_manifest=base,
            qadb_comments={1002: "special run"},
            qadb_covered_runs={1001, 1002},
            processing_supplied=True,
        )
        first = manifest["runs"]["1001"]
        second = manifest["runs"]["1002"]
        self.assertEqual(first["run_class"], "P")
        self.assertEqual(first["rcdb_quality"], "suspect")
        self.assertEqual(first["notes"], "manual review retained")
        self.assertEqual(second["run_class"], "unclassified")
        self.assertEqual(second["rcdb_quality"], "suspect")
        self.assertTrue(second["qadb_misc"])
        self.assertIn("negative_rcdb_current", rows[1]["audit_flags"])
        self.assertIn("qadb_misc_review", rows[1]["audit_flags"])
        self.assertEqual(summary["runs_with_qadb_misc"], 1)

    def test_cached_cli_writes_reusable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rcdb = directory / "input_rcdb.tsv"
            rcdb.write_text(
                "run\tbeam_current_nA\ttarget\ttorus_scale\trun_config\tquery_error\n"
                "1001\t49.5\tLH2\t1.0\t/trigger/v1.cnf\t\n",
                encoding="utf-8",
            )
            charge = directory / "input_charge.tsv"
            charge.write_text(
                "run\taccumulated_charge_nC\ttotal_events\tpassed_qadb_events\t"
                "failed_qadb_events\tsource_files\n"
                "1001\t10\t100\t90\t10\t1\n",
                encoding="utf-8",
            )
            qadb = directory / "input_qadb.txt"
            qadb.write_text(
                json.dumps(
                    {
                        "1001": {
                            "comments": ["luminosity scan"],
                            "has_misc_bit": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = directory / "output"
            arguments = [
                "audit_run_conditions.py",
                "--rcdb-input",
                str(rcdb),
                "--run-charge-input",
                str(charge),
                "--qadb-misc-input",
                str(qadb),
                "--assign-class",
                "P=1001",
                "--nominal-current",
                "P=50",
                "--run-group",
                "TEST",
                "--output-dir",
                str(output),
            ]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(audit_main(), 0)
            manifest = json.loads((output / "run_currents.json").read_text())
            summary = json.loads((output / "audit_summary.json").read_text())

        self.assertEqual(manifest["runs"]["1001"]["run_class"], "P")
        self.assertEqual(manifest["runs"]["1001"]["nominal_current_nA"], 50.0)
        self.assertTrue(manifest["runs"]["1001"]["qadb_misc"])
        self.assertEqual(summary["runs"], 1)


if __name__ == "__main__":
    unittest.main()
