from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class VisualizerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "events.npz"
        np.savez(
            self.input_path,
            Q2=np.array([1.5, 2.0, 2.5]),
            xB=np.array([0.2, 0.3, 0.4]),
            t=np.array([0.1, 0.2, 0.3]),
            pSector=np.array([1, 2, 3]),
        )

    def run_visualizer(self, command: list[str], output_name: str) -> str:
        output_path = self.directory / output_name
        completed = subprocess.run(
            [*command, str(self.input_path), "--output", str(output_path)],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(output_path.is_file())
        html = output_path.read_text(encoding="utf-8")
        self.assertIn("<title>Interactive histograms: events.npz</title>", html)
        self.assertIn("const payload = ", html)
        self.assertIn("init();", html)
        self.assertIn("Rows embedded: 3", completed.stdout)
        return html

    def test_package_entry_point_generates_standalone_html(self) -> None:
        self.run_visualizer([sys.executable, "-m", "visualizer"], "package.html")

    def test_legacy_entry_point_remains_compatible(self) -> None:
        package_html = self.run_visualizer(
            [sys.executable, "-m", "visualizer"], "package.html"
        )
        legacy_html = self.run_visualizer(
            [sys.executable, "analysis/interactive_histograms.py"], "legacy.html"
        )
        self.assertEqual(package_html, legacy_html)


if __name__ == "__main__":
    unittest.main()
