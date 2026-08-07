from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_root_distributions import (
    SHAPE_METRIC_FIELDS,
    comparison_metrics,
    histogram_counts,
    make_sample_summary,
    plot_column,
    read_processing_metadata,
    shape_metrics,
    unique_labeled_paths,
    write_csv,
)


class ShapeMetricTests(unittest.TestCase):
    def test_identical_shapes_have_zero_distance(self) -> None:
        reference = np.array([2.0, 3.0, 5.0])
        divergence, variation = shape_metrics(reference, 4.0 * reference)
        self.assertAlmostEqual(divergence, 0.0)
        self.assertAlmostEqual(variation, 0.0)

    def test_disjoint_shapes_reach_metric_maxima(self) -> None:
        divergence, variation = shape_metrics(
            np.array([10.0, 0.0]),
            np.array([0.0, 7.0]),
        )
        self.assertAlmostEqual(divergence, math.log(2.0))
        self.assertAlmostEqual(variation, 1.0)

    def test_empty_shape_reports_unavailable_metrics(self) -> None:
        divergence, variation = shape_metrics(np.zeros(3), np.ones(3))
        self.assertIsNone(divergence)
        self.assertIsNone(variation)

    def test_metrics_use_named_reference_instead_of_first_sample(self) -> None:
        rows = comparison_metrics(
            "Q2",
            {
                "candidate": np.array([0.0, 4.0]),
                "reference": np.array([4.0, 0.0]),
                "alternate": np.array([2.0, 2.0]),
            },
            "reference",
        )
        self.assertEqual([row["comparison"] for row in rows], ["candidate", "alternate"])
        self.assertTrue(all(row["reference"] == "reference" for row in rows))
        self.assertEqual(rows[0]["reference_entries_in_range"], 4)
        self.assertAlmostEqual(rows[0]["total_variation_distance"], 1.0)


class ProcessingSummaryTests(unittest.TestCase):
    class FakeTree:
        TotalEvents = 1000
        FailedQADB = 10
        FailedFinalState = 20
        FailedSkim = 30
        WrittenEvents = 940
        OutputRows = 2500

        def GetEntries(self):
            return 1

        def GetEntry(self, _index):
            return 1

        def GetBranch(self, name):
            return object() if hasattr(self, name) else None

    class FakeCharge:
        def GetVal(self):
            return 12.5

    class FakeFile:
        def __init__(self):
            self.closed = False

        def IsZombie(self):
            return False

        def Get(self, name):
            if name == "Summary":
                return ProcessingSummaryTests.FakeTree()
            if name == "AccumulatedCharge":
                return ProcessingSummaryTests.FakeCharge()
            return None

        def Close(self):
            self.closed = True

    class FakeTFile:
        opened = None

        @classmethod
        def Open(cls, _path, _mode):
            cls.opened = ProcessingSummaryTests.FakeFile()
            return cls.opened

    class FakeROOT:
        TFile = None

    FakeROOT.TFile = FakeTFile

    def test_processing_metadata_and_selected_fraction(self) -> None:
        metadata = read_processing_metadata(self.FakeROOT, Path("processing.root"))
        row = make_sample_summary("sample", Path("selected.root"), 250, metadata)
        self.assertEqual(metadata["total_events"], 1000)
        self.assertEqual(metadata["output_rows"], 2500)
        self.assertEqual(metadata["accumulated_charge_nC"], 12.5)
        self.assertAlmostEqual(row["selected_rows_per_input_event"], 0.25)
        self.assertTrue(self.FakeTFile.opened.closed)

    def test_unpaired_summary_keeps_optional_values_empty(self) -> None:
        row = make_sample_summary("sample", Path("selected.root"), 25, None)
        self.assertEqual(row["selected_rows"], 25)
        self.assertIsNone(row["total_events"])
        self.assertIsNone(row["selected_rows_per_input_event"])


class ComparisonOutputTests(unittest.TestCase):
    def test_duplicate_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate --sample label"):
            unique_labeled_paths(
                [("same", Path("one.root")), ("same", Path("two.root"))],
                "--sample",
            )

    def test_metric_csv_has_stable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.csv"
            rows = comparison_metrics(
                "Q2",
                {"reference": np.array([2.0, 1.0]), "candidate": np.array([1.0, 2.0])},
                "reference",
            )
            write_csv(output, SHAPE_METRIC_FIELDS, rows)
            with output.open(newline="", encoding="utf-8") as source:
                written = list(csv.DictReader(source))
        self.assertEqual(list(written[0]), SHAPE_METRIC_FIELDS)
        self.assertEqual(written[0]["reference"], "reference")
        self.assertEqual(written[0]["comparison"], "candidate")

    def test_plot_and_histogram_use_arbitrary_sample_labels(self) -> None:
        data = {
            "baseline": {"custom": np.linspace(0.0, 1.0, 100)},
            "variation": {"custom": np.linspace(0.2, 1.2, 100)},
        }
        bins, histograms = histogram_counts("custom", data)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            plot_column(
                "custom",
                histograms,
                bins,
                output,
                reference="variation",
                density=True,
                ratio=True,
            )
            plot = output / "custom.png"
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
