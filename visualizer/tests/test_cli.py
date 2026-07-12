from __future__ import annotations

import json
import shutil
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

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for browser-fit tests")
    def test_poisson_weighting_changes_the_fitted_objective(self) -> None:
        html = self.run_visualizer(
            [sys.executable, "-m", "visualizer"], "weighted.html"
        )
        script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        self.assertTrue(script.rstrip().endswith("init();"))
        script = script.rsplit("init();", 1)[0]
        script += """
const testXs = [0, 1, 2, 3, 4, 5];
const testYs = [10, 12, 15, 20, 30, 60];
const ordinary = polynomialFit(testXs, testYs, 1, "unweighted");
const poisson = polynomialFit(testXs, testYs, 1, "poisson");
const signalXs = [-3, -2, -1, 0, 1, 2, 3];
const signalYs = [5, 7, 20, 45, 21, 8, 5];
const signal = signalBackgroundFit(signalXs, signalYs, "gaussian", "poly0", "poisson");
const densityPanel = {
  signalModel: "none",
  backgroundModel: "poly1",
  fitWeighting: "poisson",
  density: true,
  fitRangeMin: NaN,
  fitRangeMax: NaN
};
const densityFit = make1dFit(testYs, 0, 6, densityPanel);
console.log(JSON.stringify({
  ordinaryCoeff: ordinary.coeff,
  poissonCoeff: poisson.coeff,
  ordinaryPearson: ordinary.quality.reduced,
  poissonPearson: poisson.quality.reduced,
  poissonMode: poisson.quality.weighting,
  signalMode: signal.quality.weighting,
  signalAmplitude: signal.signalAmplitude,
  densitySummary: densityFit.summary
}));
"""
        script_path = self.directory / "weighted_fit_test.js"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [shutil.which("node"), str(script_path)],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout.strip())
        self.assertEqual(result["poissonMode"], "poisson")
        self.assertNotAlmostEqual(
            result["ordinaryCoeff"][1], result["poissonCoeff"][1], places=6
        )
        self.assertLess(result["poissonPearson"], result["ordinaryPearson"])
        self.assertEqual(result["signalMode"], "poisson")
        self.assertGreater(result["signalAmplitude"], 0)
        self.assertIn("requires count bins", result["densitySummary"])


if __name__ == "__main__":
    unittest.main()
