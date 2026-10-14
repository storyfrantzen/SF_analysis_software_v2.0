from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_hipo_run_sample import catalog_runs, run_number, select_sample


class SelectHipoRunSampleTests(unittest.TestCase):
    def test_extracts_run_and_selects_middle_file_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for run in (5423, 5424, 5425):
                directory = root / f"{run:06d}"
                directory.mkdir()
                for index in range(5):
                    (directory / f"rec_clas_{run:06d}.evio.{index:05d}.hipo").touch()
            self.assertEqual(
                run_number(root / "rec_clas_005423.evio.00000.hipo"), 5423
            )
            selected = select_sample(root, files_per_run=1)
            self.assertEqual(sorted(selected), [5423, 5424, 5425])
            self.assertTrue(all("00002" in paths[0].name for paths in selected.values()))

    def test_max_runs_spans_the_available_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for run in range(100, 110):
                directory = root / str(run)
                directory.mkdir()
                (directory / f"rec_clas_{run:06d}.hipo").touch()
            selected = select_sample(root, files_per_run=1, max_runs=3)
            self.assertEqual(sorted(selected), [100, 104, 109])

    def test_catalog_classes_restrict_eligible_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            catalog = root / "runs.json"
            catalog.write_text(
                json.dumps(
                    {
                        "runs": {
                            "5875": {"run_class": "L4"},
                            "5885": {"run_class": "P3"},
                            "5991": {"run_class": "L6"},
                            "5995": {"run_class": "E2"},
                            "5996": {"run_class": "T"},
                        }
                    }
                )
            )
            for run in (5863, 5875, 5885, 5991, 5995, 5996):
                directory = root / str(run)
                directory.mkdir()
                (directory / f"rec_clas_{run:06d}.hipo").touch()
            allowed = catalog_runs(catalog, {"L4", "P3"})
            self.assertEqual(allowed, {5875, 5885})
            selected = select_sample(root, files_per_run=1, allowed_runs=allowed)
            self.assertEqual(sorted(selected), [5875, 5885])


if __name__ == "__main__":
    unittest.main()
