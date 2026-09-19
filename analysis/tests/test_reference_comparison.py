from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eppi0.reference_comparison import (
    load_reference_table,
    match_reference_bins,
)


class ReferenceComparisonTests(unittest.TestCase):
    def _table(self, directory: Path) -> Path:
        path = directory / "reference.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "Q2",
                    "xB",
                    "minus_t",
                    "sigma_U",
                    "sigma_U_stat",
                    "sigma_U_sys",
                    "sigma_LT",
                    "sigma_LT_stat",
                    "sigma_LT_sys",
                    "sigma_TT",
                    "sigma_TT_stat",
                    "sigma_TT_sys",
                ]
            )
            writer.writerow([1.2, 0.12, 0.12, 10, 1, 2, 3, 4, 5, -6, 7, 8])
            writer.writerow([2.2, 0.22, 0.22, 20, 2, 3, 4, 5, 6, -7, 8, 9])
        return path

    def test_loads_and_matches_reference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = load_reference_table(self._table(Path(temporary)))
        np.testing.assert_allclose(table.values, [[10, 3, -6], [20, 4, -7]])
        np.testing.assert_allclose(table.statistical, [[1, 4, 7], [2, 5, 8]])
        np.testing.assert_allclose(table.systematic, [[2, 5, 8], [3, 6, 9]])
        match = match_reference_bins(
            table,
            q2_edges=[1.0, 1.5, 2.0, 2.5],
            xb_edges=[0.1, 0.15, 0.2, 0.25],
            t_edges=[0.09, 0.15, 0.2, 0.3],
        )
        np.testing.assert_array_equal(match.q2_index, [0, 2])
        np.testing.assert_array_equal(match.xb_index, [0, 2])
        np.testing.assert_array_equal(match.t_index, [0, 2])

    def test_rejects_duplicate_analysis_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._table(Path(temporary))
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            rows[2][0:3] = ["1.3", "0.13", "0.13"]
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            table = load_reference_table(path)
        with self.assertRaisesRegex(ValueError, "multiple rows"):
            match_reference_bins(
                table,
                q2_edges=[1.0, 1.5],
                xb_edges=[0.1, 0.15],
                t_edges=[0.09, 0.15],
            )

    def test_published_table_has_one_to_one_rga_bin_matches(self) -> None:
        root = Path(__file__).resolve().parents[2]
        table = load_reference_table(
            root / "data/reference/clas6_bedlinskiy_2014_structure_functions.csv"
        )
        match = match_reference_bins(
            table,
            q2_edges=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 5.5, 7.0, 10.5],
            xb_edges=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.38, 0.48, 0.58, 0.7],
            t_edges=[0.09, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0],
        )
        self.assertEqual(table.q2.size, 96)
        self.assertEqual(
            len(set(zip(match.q2_index, match.xb_index, match.t_index))),
            96,
        )


if __name__ == "__main__":
    unittest.main()
