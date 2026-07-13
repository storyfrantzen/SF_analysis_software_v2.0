from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from visualizer.app import build_payload


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
        self.assertIn('<option value="unbinned">Unbinned likelihood</option>', html)
        self.assertIn('id="fitScanDetail" type="range" min="1" max="5"', html)
        self.assertIn("grid-template-columns: minmax(220px, 270px)", html)
        self.assertIn('<div class="dataset-heading">', html)
        self.assertIn('<div class="subtle" id="source" hidden></div>', html)
        self.assertIn('<div class="filter-row constraint-builder">', html)
        self.assertIn('id="constraintStatus" role="status"', html)
        self.assertIn("Enter a minimum, a maximum, or both.", html)
        self.assertIn('<div class="toolbar-tile action-tile" aria-label="Plot actions">', html)
        self.assertIn('<div class="toolbar-tile count-tile" aria-label="Event counts">', html)
        self.assertIn('<div class="canvas-toolbar" id="canvasToolbar" aria-label="Display options">', html)
        self.assertIn('<span id="meanX" hidden>-</span>', html)
        self.assertIn('<span id="meanY" hidden>-</span>', html)
        self.assertIn("grid-template-columns: minmax(74px, 1.3fr)", html)
        self.assertIn('<div class="control-deck analysis-tools">', html)
        self.assertIn('<div class="sidebar-derived">', html)
        self.assertIn('<div class="control-panel fit-panel">', html)
        self.assertIn('<div class="control-panel text-panel" id="textFilterPanel">', html)
        self.assertIn(".analysis-tools .fit-panel { order: 1; }", html)
        self.assertIn(".analysis-tools .text-panel { order: 2; }", html)
        self.assertNotIn('id="opPreview"', html)
        self.assertNotIn('operation-feedback-label', html)
        self.assertIn('id="toggleFitAnnotations" aria-pressed="true"', html)
        self.assertIn("showFitAnnotations: true", html)
        self.assertIn("panel?.showFitAnnotations === false", html)
        self.assertIn('<div class="fit-output">', html)
        self.assertIn('class="fit-summary" id="fitSummary" aria-live="polite"', html)
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))", html)
        self.assertIn("function renderFitSummary(panel)", html)
        self.assertIn(".fit-model-grid select { width: auto; max-width: 190px; }", html)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", html)
        self.assertNotIn('class="plot-head"', html)
        self.assertNotIn('id="plotTitleA"', html)
        self.assertNotIn('id="panelSummaryA"', html)
        self.assertIn("const showPanelTabs = enabledPanels.length > 1", html)
        self.assertLess(html.index('id="rangeFilters"'), html.index('class="sidebar-derived"'))
        self.assertLess(html.index('class="sidebar-derived"'), html.index('id="xbins"'))
        self.assertIn('id="canvasToolbarSlotA"', html)
        self.assertIn("toolbarSlot.appendChild(canvasToolbar)", html)
        self.assertLess(html.index('id="plotA"'), html.index('id="hoverInfoA"'))
        self.assertLess(html.index('id="plotB"'), html.index('id="hoverInfoB"'))
        self.assertLess(html.index('aria-label="Event counts"'), html.index('aria-label="Display options"'))
        self.assertLess(html.index('aria-label="Display options"'), html.index('aria-label="Plot actions"'))
        self.assertLess(html.index('aria-label="Display options"'), html.index('class="plot-grid"'))
        self.assertIn("Rows embedded: 3", completed.stdout)
        return html

    def test_package_entry_point_generates_standalone_html(self) -> None:
        self.run_visualizer([sys.executable, "-m", "visualizer"], "package.html")

    def test_quantity_aware_default_axis_ranges(self) -> None:
        central_missing_mass = np.linspace(-0.08, 0.08, 100)
        arrays = {
            "eDet": np.resize(np.array([1, 1, 2, 2], dtype=float), 102),
            "pSector": np.resize(np.arange(7, dtype=float), 102),
            "m2_miss": np.concatenate(([-25.0], central_missing_mass, [40.0])),
        }
        payload = build_payload(
            Path("events.npz"),
            arrays,
            metadata={},
            downsample={"sampled": False, "originalRows": 102, "embeddedRows": 102},
            title=None,
        )
        variables = {item["name"]: item for item in payload["variables"]}
        self.assertEqual(variables["eDet"]["min"], 0.5)
        self.assertEqual(variables["eDet"]["max"], 2.5)
        self.assertGreater(variables["m2_miss"]["min"], -1.0)
        self.assertLess(variables["m2_miss"]["max"], 1.0)
        self.assertIn("pSector", {item["name"] for item in payload["sectorSplits"]})

    def test_legacy_entry_point_remains_compatible(self) -> None:
        package_html = self.run_visualizer(
            [sys.executable, "-m", "visualizer"], "package.html"
        )
        legacy_html = self.run_visualizer(
            [sys.executable, "analysis/interactive_histograms.py"], "legacy.html"
        )
        self.assertEqual(package_html, legacy_html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for browser-fit tests")
    def test_browser_fit_methods_execute_numerically(self) -> None:
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
const unbinnedValues = [];
for (let i = 0; i < 180; i++) {
  unbinnedValues.push(0.135 + 0.007 * Math.sin(i * 2.399) + 0.0015 * Math.sin(i * 0.71));
}
for (let i = 0; i < 80; i++) unbinnedValues.push(0.08 + 0.12 * (i + 0.5) / 80);
const unbinnedSpec = {signal: "gaussian", background: "poly0"};
const unbinnedPanel = {
  fitMethod: "unbinned",
  fitScanDetail: 2,
  density: false,
  fitRangeMin: NaN,
  fitRangeMax: NaN
};
const unbinned40 = unbinnedLikelihoodFit(unbinnedValues, 0.08, 0.20, unbinnedSpec, unbinnedPanel, 40);
const unbinned100 = unbinnedLikelihoodFit(unbinnedValues, 0.08, 0.20, unbinnedSpec, unbinnedPanel, 100);
const unbinnedBackground = unbinnedLikelihoodFit(
  unbinnedValues,
  0.08,
  0.20,
  {signal: "none", background: "poly3"},
  unbinnedPanel,
  40
);
const coarseShapes = unbinnedSignalCandidates("gaussian", {mean: 0, sigma: 1}, 1).length;
const fineShapes = unbinnedSignalCandidates("gaussian", {mean: 0, sigma: 1}, 5).length;
const protonFacets = orderSplitFacets(
  "pSector",
  [0, 1, 2, 3, 4, 5, 6].map(value => ({value, label: String(value), shortLabel: String(value)}))
);
const protonLayout = facetLayout({width: 1200}, protonFacets.length, "pSector", protonFacets);
const protonAxisVisibility = protonFacets.map((_, index) => facetAxisVisibility(protonLayout, index, protonFacets.length));
const clampedScaleMarkerLeft = constrainedOverlayLeft(980, 120, 1000);
const emptyConstraintValueIsNaN = Number.isNaN(parseNumber(""));
const lowerOnlyRange = [valuePassesRange(4, 3, NaN), valuePassesRange(2, 3, NaN)];
const upperOnlyRange = [valuePassesRange(4, NaN, 5), valuePassesRange(6, NaN, 5)];
console.log(JSON.stringify({
  ordinaryCoeff: ordinary.coeff,
  poissonCoeff: poisson.coeff,
  ordinaryPearson: ordinary.quality.reduced,
  poissonPearson: poisson.quality.reduced,
  poissonMode: poisson.quality.weighting,
  signalMode: signal.quality.weighting,
  signalAmplitude: signal.signalAmplitude,
  densitySummary: densityFit.summary,
  unbinnedMethod: unbinned40.method,
  unbinnedFraction: unbinned40.signalFraction,
  unbinnedMean: unbinned40.mean,
  unbinnedScanDetail: unbinned40.scanDetail,
  coarseShapes,
  fineShapes,
  unbinnedNll40: unbinned40.nll,
  unbinnedNll100: unbinned100.nll,
  unbinnedBackgroundDegree: unbinnedBackground.backgroundDegree,
  protonFacetValues: protonFacets.map(facet => facet.value),
  protonFacetPositions: protonLayout.positions,
  protonAxisVisibility,
  compactTick: formatAxisTick(0.090000),
  clampedScaleMarkerLeft,
  emptyConstraintValueIsNaN,
  lowerOnlyRange,
  upperOnlyRange
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
        self.assertEqual(result["unbinnedMethod"], "unbinned")
        self.assertGreater(result["unbinnedFraction"], 0.4)
        self.assertLess(result["unbinnedFraction"], 0.95)
        self.assertAlmostEqual(result["unbinnedMean"], 0.135, delta=0.01)
        self.assertEqual(result["unbinnedScanDetail"], 2)
        self.assertGreater(result["fineShapes"], result["coarseShapes"])
        self.assertAlmostEqual(
            result["unbinnedNll40"], result["unbinnedNll100"], places=8
        )
        self.assertEqual(result["unbinnedBackgroundDegree"], 3)
        self.assertEqual(result["protonFacetValues"], [1, 2, 3, 4, 5, 6, 0])
        self.assertEqual(
            result["protonFacetPositions"],
            [
                {"row": 0, "col": 0},
                {"row": 0, "col": 1},
                {"row": 0, "col": 2},
                {"row": 1, "col": 0},
                {"row": 1, "col": 1},
                {"row": 1, "col": 2},
                {"row": 2, "col": 1},
            ],
        )
        self.assertEqual(result["compactTick"], "0.09")
        self.assertEqual(result["clampedScaleMarkerLeft"], 876)
        self.assertTrue(result["emptyConstraintValueIsNaN"])
        self.assertEqual(result["lowerOnlyRange"], [True, False])
        self.assertEqual(result["upperOnlyRange"], [True, False])
        self.assertEqual(
            result["protonAxisVisibility"],
            [
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": False, "showYLabel": True},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": False, "showYLabel": False},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": False, "showYLabel": False},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": True, "showYLabel": True},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": False, "showYLabel": False},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": True, "showYLabel": False},
                {"showXTickLabels": True, "showYTickLabels": True, "showXLabel": True, "showYLabel": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
