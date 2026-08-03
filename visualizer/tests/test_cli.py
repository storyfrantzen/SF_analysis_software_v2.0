from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from visualizer.app import (
    add_derived_quantities,
    build_payload,
    label_for,
    normalize_visual_columns,
    sample_row_indices,
)


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
        self.assertIn('<div class="dataset-kicker" id="datasetKicker">Dataset</div>', html)
        self.assertIn('<div class="subtle" id="source" hidden></div>', html)
        self.assertIn('<body aria-busy="true">', html)
        self.assertIn('id="startupLoading" role="status"', html)
        self.assertIn("Reading embedded data…", html)
        self.assertIn('id="startupProgressPercent">0%</span>', html)
        self.assertIn('id="startupProgressBar"', html)
        self.assertIn("Decoding column ${index + 1} of ${entries.length}", html)
        self.assertIn('setStartupProgress(100, "Ready")', html)
        self.assertIn('document.body.removeAttribute("aria-busy")', html)
        self.assertIn('startupLoading.classList.add("complete")', html)
        self.assertIn('<div class="filter-row constraint-builder">', html)
        self.assertIn('id="constraintStatus" role="status"', html)
        self.assertIn("Enter a minimum, a maximum, or both.", html)
        self.assertIn('<div class="toolbar-tile action-tile" aria-label="Plot actions">', html)
        self.assertIn('id="plotTools" aria-haspopup="menu" aria-expanded="false"', html)
        self.assertIn('<button type="button" id="saveWorkspace">Save workspace</button>', html)
        self.assertIn('<button type="button" id="restoreWorkspace">Restore saved</button>', html)
        self.assertIn("function workspaceSnapshot()", html)
        self.assertIn("function restoreWorkspace(showStatus = false)", html)
        self.assertIn("function scheduleUpdate()", html)
        self.assertIn('<div class="toolbar-tile count-tile" aria-label="Dataset counts">', html)
        self.assertIn('<span class="subtle">samples</span><strong id="sampleCount">1</strong>', html)
        self.assertIn('<div class="header-utility-stack">', html)
        self.assertIn('id="datasetStatus" role="status" aria-live="polite"', html)
        self.assertIn("sampled ${payload.downsample.embeddedRows.toLocaleString()} of", html)
        self.assertIn("seed ${payload.downsample.seed}", html)
        self.assertLess(html.index('id="loadFiles"'), html.index('aria-label="Dataset counts"'))
        self.assertLess(html.index('class="plot-panel-controls"'), html.index('id="loadFiles"'))
        self.assertLess(html.index('id="sampleCount"'), html.index('id="selectedCount"'))
        self.assertLess(html.index('id="selectedCount"'), html.index('id="embeddedCount"'))
        self.assertIn('<div class="canvas-toolbar" id="canvasToolbar" aria-label="Plot controls">', html)
        self.assertIn('id="toggleCanvasToolbar" aria-expanded="true" aria-controls="canvasToolbar"', html)
        self.assertIn('<div class="toolbar-tile axis-tile" aria-label="Axis labels, binning, ranges, and ticks">', html)
        self.assertIn('<div class="toolbar-tile display-tile" aria-label="Display options">', html)
        self.assertIn('id="plotHeight" type="range" min="0.25" max="1" step="0.01" value="0.5"', html)
        self.assertIn('id="plotHeightValue">50%</span>', html)
        self.assertIn('id="plotWidth" type="range" min="0.5" max="1" step="0.01" value="1"', html)
        self.assertIn('id="plotWidthValue">100%</span>', html)
        self.assertIn(".canvas-toolbar .display-tile {", html)
        self.assertIn("grid-template-columns: repeat(3, max-content)", html)
        self.assertIn(".canvas-toolbar .action-tile {", html)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", html)
        self.assertIn('<label class="chip" id="logzChip">', html)
        self.assertIn('el("logzChip").style.display = panel.mode === "2d"', html)
        self.assertIn("grid-template-columns: minmax(470px, 1.7fr) repeat(2, minmax(240px, 1fr))", html)
        self.assertIn("grid-template-columns: repeat(5, minmax(70px, 1fr))", html)
        self.assertIn("plotHeightFraction: 0.5", html)
        self.assertIn("plotWidthFraction: 1", html)
        self.assertIn("function preparePanelCanvas(panel)", html)
        self.assertIn("width * heightScale", html)
        self.assertIn("clamp(width * heightScale + reclaimedHeight, 160, 2000)", html)
        self.assertIn("function prepareFacetCanvas(canvas, panel, facetCount", html)
        self.assertIn("canonicalPlotHeight(panel.plotHeightFraction)", html)
        self.assertIn("canonicalPlotWidth(panel.plotWidthFraction)", html)
        self.assertIn("width: 100%", html)
        self.assertIn('el("yrange").classList.toggle("axis-range-hidden", panel.mode !== "2d")', html)
        self.assertIn(".axis-y-ticks { grid-column: 4; grid-row: 2; }", html)
        self.assertIn(".axis-y-label { grid-column: 5; grid-row: 2; }", html)
        self.assertIn('id="canvasContextMenu" role="menu" hidden', html)
        self.assertIn('id="makeGhost" role="menuitem">Make ghost</button>', html)
        self.assertIn('id="clearGhost" role="menuitem">Clear ghost</button>', html)
        self.assertIn('id="toggleCanvasToolbarContext" role="menuitem">Hide plot controls</button>', html)
        self.assertIn('id="toggleMeanGuides" role="menuitemcheckbox" aria-checked="false"', html)
        self.assertIn('id="profileX" role="menuitem">Profile X</button>', html)
        self.assertIn('id="profileY" role="menuitem">Profile Y</button>', html)
        self.assertIn('id="addFunctionCurve" role="menuitem">Add function curve…</button>', html)
        self.assertIn('id="referenceCurveEditor" role="dialog"', html)
        self.assertIn('<option value="y-of-x">y = f(x)</option>', html)
        self.assertIn('<option value="dash-dot">Dash-dot</option>', html)
        self.assertIn('id="referenceCurveExpression" type="text"', html)
        self.assertIn('id="referenceLineWidth" type="number" min="0.5" max="3"', html)
        self.assertIn('addEventListener("contextmenu"', html)
        self.assertIn("ghostPlot: null", html)
        self.assertIn("referenceCurves: []", html)
        self.assertIn("showMeanGuides: false", html)
        self.assertIn("xName: panel.xvar", html)
        self.assertIn("function captureGhost(key)", html)
        self.assertIn("function profileBinAtCanvasPoint(clientX, clientY, key)", html)
        self.assertIn("function launchBinProfile(axis)", html)
        self.assertIn('activePanel = "B"', html)
        self.assertIn("compareMode = true", html)
        self.assertIn("canvasToolbarCollapsed = true", html)
        self.assertIn("maxExclusive", html)
        self.assertIn("function drawGhost1d", html)
        self.assertIn("function drawGhost2d", html)
        self.assertIn("function drawMeanGuides", html)
        self.assertIn("ctx.setLineDash([6, 4])", html)
        self.assertIn('meanGuides.textContent = panel.showMeanGuides ? "Hide mean guides" : "Show mean guides"', html)
        self.assertIn("function poissonBinError", html)
        self.assertIn("function histogramPointScaleMax", html)
        self.assertIn("function draw1dPoints", html)
        self.assertIn("function draw1dOverlayLegend", html)
        self.assertIn("value + poissonBinError(value, item.total, item.density)", html)
        self.assertIn("function compileMathExpression", html)
        self.assertIn("function referenceLineDash", html)
        self.assertIn("function drawReferenceCurves", html)
        self.assertIn('<span id="meanX" hidden>-</span>', html)
        self.assertIn('<span id="meanY" hidden>-</span>', html)
        self.assertIn("grid-template-columns: minmax(74px, 1.3fr)", html)
        self.assertIn('id="sliceControls" hidden', html)
        self.assertIn('id="sliceBins" type="number" min="1" max="24"', html)
        self.assertIn('id="sliceEdges" type="text"', html)
        self.assertIn("function numericSliceFacets", html)
        self.assertIn("function valueMatchesFacet", html)
        self.assertIn('class="quick-category collapsed" id="quickCategoryBlock"', html)
        self.assertIn('id="toggleTopology" aria-expanded="false"', html)
        self.assertIn("let topologyCollapsed = true", html)
        self.assertIn("logz: true", html)
        self.assertIn("colorScale: true", html)
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
        self.assertIn('<div class="fit-panel-layout">', html)
        self.assertIn(".fit-panel-layout.engaged {", html)
        self.assertIn('<div class="fit-controls">', html)
        self.assertIn('class="fit-summary" id="fitSummary" aria-live="polite"', html)
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))", html)
        self.assertIn(".fit-summary.sector {", html)
        self.assertIn("grid-template-columns: repeat(3, minmax(0, 1fr))", html)
        self.assertIn("isProtonSectorSplit(panel?.splitVar)", html)
        self.assertIn("function renderFitSummary(panel)", html)
        self.assertIn("const engaged = panelHasFit(panel)", html)
        self.assertIn("target.hidden = !engaged", html)
        self.assertIn(".fit-model-grid select { width: auto; max-width: 190px; }", html)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", html)
        self.assertNotIn('class="plot-head"', html)
        self.assertNotIn('id="plotTitleA"', html)
        self.assertNotIn('id="panelSummaryA"', html)
        self.assertIn("const showPanelTabs = enabledPanels.length > 1", html)
        self.assertIn('id="splitView" aria-pressed="false">split view</button>', html)
        self.assertIn('id="sharedPanelFilters" class="active" aria-pressed="true"', html)
        self.assertIn("let sharedPanelFilters = true", html)
        self.assertIn("function toggleSharedPanelFilters()", html)
        self.assertIn("copyFilterState(panels[key].filterState, sharedFilterState)", html)
        self.assertIn("selectedMask(filterStateForPanel(key))", html)
        self.assertIn("filterBadgeText(filterStateForPanel(key))", html)
        self.assertLess(html.index('id="rangeFilters"'), html.index('class="sidebar-derived"'))
        self.assertLess(html.index('class="sidebar-derived"'), html.index('id="xbins"'))
        self.assertIn('id="canvasToolbarSlotA"', html)
        self.assertIn("toolbarSlot.appendChild(canvasToolbar)", html)
        self.assertIn("function syncCanvasToolbarRail(comparing, canvasToolbar)", html)
        self.assertIn("function toggleCanvasToolbar()", html)
        self.assertIn("function syncCanvasToolbarVisibility()", html)
        self.assertIn("let canvasToolbarExpandedHeight = 0", html)
        self.assertIn("canvasToolbarExpandedHeight = measuredHeight", html)
        self.assertIn("width * heightScale + reclaimedHeight", html)
        self.assertIn("30 + 12 + reclaimedHeight", html)
        self.assertIn('el("toggleCanvasToolbarContext").addEventListener("click"', html)
        self.assertIn('contextButton.textContent = actionLabel', html)
        self.assertIn('toolbar.hidden = canvasToolbarCollapsed', html)
        self.assertIn("toolbarRailHeight + tallestHeader - headerHeights[key]", html)
        self.assertIn("slot.getBoundingClientRect().top - pane.getBoundingClientRect().top", html)
        self.assertIn("lowestCanvasTop - canvasTops[key]", html)
        self.assertIn("function alignVisibleCanvasTops(visible)", html)
        self.assertIn("requestAnimationFrame(() =>", html)
        self.assertIn(".canvas-toolbar[hidden] { display: none; }", html)
        self.assertIn(".canvas-toolbar-slot.controls-collapsed { margin-bottom: 0; }", html)
        self.assertIn(".plot-grid.compare .canvas-toolbar-slot { display: block; }", html)
        self.assertIn(".plot-grid.compare .canvas-toolbar {", html)
        self.assertIn("plotGrid.dataset.activePanel = activePanel", html)
        self.assertIn('multiple ? "Workspace" : "Dataset"', html)
        self.assertIn("samples combined", html)
        self.assertIn('el("source").hidden = !multiple', html)
        self.assertLess(html.index('id="plotA"'), html.index('id="hoverInfoA"'))
        self.assertLess(html.index('id="plotB"'), html.index('id="hoverInfoB"'))
        self.assertLess(html.index('aria-label="Dataset counts"'), html.index('aria-label="Display options"'))
        self.assertLess(html.index('aria-label="Dataset counts"'), html.index('aria-label="Plot actions"'))
        self.assertLess(html.index('aria-label="Axis labels, binning, ranges, and ticks"'), html.index('aria-label="Display options"'))
        self.assertLess(html.index('aria-label="Display options"'), html.index('aria-label="Plot actions"'))
        self.assertLess(html.index('aria-label="Display options"'), html.index('class="plot-grid"'))
        self.assertLess(html.index('aria-label="Plot actions"'), html.index('class="plot-grid"'))
        self.assertIn("Rows embedded: 3", completed.stdout)
        return html

    def test_package_entry_point_generates_standalone_html(self) -> None:
        self.run_visualizer([sys.executable, "-m", "visualizer"], "package.html")

    def test_sampling_indices_are_seeded_reproducible_and_not_a_prefix(self) -> None:
        first = sample_row_indices(10_000, 250, 12345)
        repeated = sample_row_indices(10_000, 250, 12345)
        changed = sample_row_indices(10_000, 250, 54321)
        self.assertIsNotNone(first)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))
        self.assertFalse(np.array_equal(first, np.arange(250, dtype=np.uint64)))
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertIsNone(sample_row_indices(250, 250, 12345))
        self.assertIsNone(sample_row_indices(10_000, 0, 12345))

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

    def test_particle_quantity_labels_are_standardized(self) -> None:
        self.assertEqual(label_for("electronP"), "electron p")
        self.assertEqual(label_for("protonP"), "proton p")
        self.assertEqual(label_for("rec_eIdx"), "REC electron index")
        self.assertEqual(label_for("rec_electronIdx"), "REC electron index")
        self.assertEqual(label_for("rec_pDet"), "REC proton detector")
        self.assertEqual(label_for("rec_protonP"), "REC proton p")
        self.assertEqual(label_for("rec_gamma1Sector"), "REC gamma 1 sector")
        self.assertEqual(label_for("rec_gamma1P"), "REC gamma 1 p")
        self.assertEqual(label_for("gen_gamma2P"), "GEN gamma 2 p")
        self.assertEqual(label_for("electronTheta_deg"), "electron theta deg")
        self.assertEqual(label_for("protonTheta_deg"), "proton theta deg")
        self.assertEqual(label_for("pi0_theta_deg"), "pi0 theta deg")
        self.assertEqual(label_for("rec_electronEECIN"), "REC electron E ECIN")

    def test_prefixed_particle_aliases_are_deduplicated(self) -> None:
        values = np.array([1.0, 2.0])
        normalized = normalize_visual_columns({
            "rec_eIdx": values,
            "rec_electronIdx": values,
            "rec_g1Sector": values,
            "rec_gamma1Sector": values,
            "rec_proton_detector": values,
            "rec_protonDet": values,
        })
        self.assertNotIn("rec_eIdx", normalized)
        self.assertIn("rec_electronIdx", normalized)
        self.assertNotIn("rec_g1Sector", normalized)
        self.assertIn("rec_gamma1Sector", normalized)
        self.assertNotIn("rec_proton_detector", normalized)
        self.assertIn("rec_protonDet", normalized)

    def test_tagged_cut_sets_expand_to_filterable_pass_quantities(self) -> None:
        arrays = add_derived_quantities({
            "Q2": np.array([1.0, 2.0, 3.0]),
            "evaluatedCuts": np.array([
                "electron.fiducial,proton.fiducial",
                "electron.fiducial,proton.fiducial",
                "electron.fiducial",
            ]),
            "failedCuts": np.array([
                "proton.fiducial",
                "",
                "electron.fiducial",
            ]),
        })
        np.testing.assert_allclose(
            arrays["passCut_electron_fiducial"], [1.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            arrays["passCut_proton_fiducial"][:2], [0.0, 1.0]
        )
        self.assertTrue(np.isnan(arrays["passCut_proton_fiducial"][2]))

        payload = build_payload(
            Path("diagnostics.npz"),
            {name: np.asarray(value) for name, value in arrays.items()},
            metadata={},
            downsample={"sampled": False, "originalRows": 3, "embeddedRows": 3},
            title=None,
        )
        filters = {item["name"]: item for item in payload["categoricalFilters"]}
        self.assertEqual(
            filters["passCut_electron_fiducial"]["labels"], ["fail", "pass"]
        )

    def test_pyroot_character_arrays_expand_to_cut_quantities(self) -> None:
        arrays = add_derived_quantities({
            "Q2": np.array([1.0, 2.0]),
            "evaluatedCuts": np.array([
                np.array(list("electron.fiducial,proton.fiducial")),
                np.array(list("electron.fiducial")),
            ], dtype=object),
            "failedCuts": np.array([
                np.frombuffer(b"proton.fiducial", dtype=np.uint8),
                np.array([], dtype=np.uint8),
            ], dtype=object),
        })
        np.testing.assert_allclose(
            arrays["passCut_electron_fiducial"], [1.0, 1.0]
        )
        self.assertEqual(arrays["passCut_proton_fiducial"][0], 0.0)
        self.assertTrue(np.isnan(arrays["passCut_proton_fiducial"][1]))

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
const manualSliceParse = parseManualSliceEdges("0, 0.2 0.5; 1");
const invalidSliceParse = parseManualSliceEdges("0, 0.5, 0.4, 1");
const slicePanel = {sliceBins: 6, sliceEdges: "0, 0.2, 0.5, 1"};
const sliceDefinitions = numericSliceFacets(slicePanel, "xB");
const sliceMembership = sliceDefinitions.map(definition => [
  valueMatchesFacet(definition.lower, definition),
  valueMatchesFacet(definition.upper, definition)
]);
const automaticSlices = numericSliceConfiguration({sliceBins: 4, sliceEdges: ""}, "xB");
const minimumWFunction = compileMathExpression("(2^2 - 0.9382720813^2) * x / (1 - x)", "x");
const calculatorValues = [
  minimumWFunction(0.3),
  compileMathExpression("sqrt(x^2) + sin(pi / 2)", "x")(-3),
  compileMathExpression("max(y, 2) + pow(y, 2)", "y")(3),
  compileMathExpression("2**3 + log10(100)", "x")(0)
];
const referenceStyles = [
  referenceLineDash("solid"),
  referenceLineDash("dashed"),
  referenceLineDash("dotted"),
  referenceLineDash("dash-dot"),
  referenceLineWidth({lineWidth: 1.25}),
  referenceLineWidth({lineWidth: 9})
];
const poissonErrors = [
  poissonBinError(9, 9, false),
  poissonBinError(0.3, 10, true)
];
let rejectedExpression = false;
try { compileMathExpression("window.alert(1)", "x"); } catch (_) { rejectedExpression = true; }
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
  upperOnlyRange,
  manualSliceEdges: manualSliceParse.edges,
  invalidSliceError: invalidSliceParse.error,
  sliceMembership,
  automaticSliceCount: automaticSlices.edges.length - 1,
  calculatorValues,
  referenceStyles,
  poissonErrors,
  rejectedExpression
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
        self.assertEqual(result["manualSliceEdges"], [0, 0.2, 0.5, 1])
        self.assertIn("strictly increasing", result["invalidSliceError"])
        self.assertEqual(
            result["sliceMembership"],
            [[True, False], [True, False], [True, True]],
        )
        self.assertEqual(result["automaticSliceCount"], 4)
        self.assertAlmostEqual(result["calculatorValues"][0], 1.337, delta=0.002)
        self.assertAlmostEqual(result["calculatorValues"][1], 4.0, places=10)
        self.assertAlmostEqual(result["calculatorValues"][2], 12.0, places=10)
        self.assertAlmostEqual(result["calculatorValues"][3], 10.0, places=10)
        self.assertEqual(
            result["referenceStyles"],
            [[], [7, 4], [1.25, 3], [8, 3, 1.25, 3], 1.25, 3],
        )
        self.assertAlmostEqual(result["poissonErrors"][0], 3.0)
        self.assertAlmostEqual(
            result["poissonErrors"][1], np.sqrt(3.0) / 10.0
        )
        self.assertTrue(result["rejectedExpression"])
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
