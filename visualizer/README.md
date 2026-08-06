# Interactive visualizer

The visualizer is a standalone subsystem for interactive cut studies, detector
comparisons, and downstream extraction diagnostics. It converts compact NPZ
artifacts or selected ROOT trees into a self-contained HTML application. The
generated file embeds its event data and does not require a JavaScript build or
application server.

Plot and filter settings are saved automatically in the browser for each source
dataset. The Plot actions toolbar also provides explicit Save workspace and
Restore saved controls. Plot tools such as ghost overlays, mean guides, profiles,
and reference curves are available from the visible Plot tools button as well as
the canvas context menu.

## Usage

Run the package from the repository root:

```bash
python3 -m visualizer data_events.npz \
  --output results/data_histograms.html

python3 -m visualizer selected_data.root \
  --format root --output results/selected_data_histograms.html \
  --dictionary build/libROOTBranchesDict.so

python3 -m visualizer converter_rows.root \
  --format root --max-source-events 250000 --seed 12345 \
  --output results/converter_event_sample.html \
  --dictionary build/libROOTBranchesDict.so
```

Inputs larger than `--max-events` are sampled across the complete dataset for
both NPZ and ROOT files. Sampling is deterministic and defaults to `--seed
12345`, so repeated runs select the same source rows. Pass a different seed for
a different reproducible sample, or `--max-events 0` to embed every row.

Converter ROOT trees often contain several particle rows per trigger event. Use
`--max-source-events N` to sample `N` distinct source events instead of `N`
rows. Every particle row belonging to each selected event is retained, the
selection remains deterministic under `--seed`, and the ordinary `--max-events`
row cap is disabled for that run. Event identity prefers
`sourceFileId + sourceEventIndex` and falls back to `runNum + eventNum` when
needed. Use `--max-source-events 0` to embed every source event.

New converter files provide `rEvents` for one-row-per-event
diagnostics. Its scalar topology branches include `nPid2212`,
`nPid2212FD`, and `nPid2212CD`; select that tree when studying reconstructed
multiplicity without particle-row weighting.

Use `--root-filter` to apply a ROOT `RDataFrame` expression before either row
or source-event sampling. Converter object fields keep their qualified ROOT
branch names everywhere, including `event.runNum`, `event.eventNum`, `rec.pid`,
and `rec.chi2pid`. For example, this retains every particle row from run 18480
and embeds all qualifying source events:

```bash
python3 -m visualizer converter_rows.root \
  --format root --root-filter 'event.runNum == 18480' \
  --max-source-events 0 --seed 12345 \
  --output results/run_18480.html \
  --dictionary build/libROOTBranchesDict.so
```

For quick studies that do not need a persistent HTML artifact, the launcher
below generates under a temporary directory, serves the visualizer, and removes
the temporary HTML when the server stops:

```bash
scripts/visualize.sh converter_rows.root 0 'event.runNum == 18480'
scripts/visualize.sh converter_rows.root 250000
```

The second argument is a source-event limit, not a row limit; `0` keeps every
event passing the optional filter. Sampling still defaults to seed `12345`.
The standalone `python3 -m visualizer` workflow remains available when a
portable, comprehensive HTML file is desired.

Then serve a generated file or a directory containing several visualizers:

```bash
scripts/serve_visualizer.sh results/selected_data_histograms.html
scripts/serve_visualizer.sh results
```

The historical command remains supported:

```bash
python3 analysis/interactive_histograms.py INPUT --output OUTPUT.html
```

## Histogram fits

One-dimensional histograms are drawn at bin centers as points with capped
Poisson error bars. Count-mode uncertainties use `sqrt(N)` for each bin. In
density mode the same uncertainty is rescaled by the selected total, so a bin
with raw count `N` is displayed with uncertainty `sqrt(N) / N_selected`.
Overlaid and faceted one-dimensional histograms use the same convention.

The Fit panel supports ordinary least squares, Poisson-weighted least squares,
and conditional unbinned likelihood fits. `Poisson WLS (Pearson)` iteratively
solves the normal equations with per-bin variance `max(expected count, 1)` and
reports the Pearson chi-square per degree of freedom. It applies to polynomial
backgrounds and Gaussian or Crystal Ball signal-plus-background fits. Poisson
weighting is intentionally unavailable in density mode because normalized bin
fractions are not Poisson counts.

`Unbinned likelihood` fits the individual selected values and treats the event
count as fixed. It reports a signal fraction rather than absolute signal and
background yields. Gaussian and Crystal Ball signals can be combined with
polynomial backgrounds selected through the same constant/degree 1-5 control
used by binned fits. For the unbinned likelihood, the polynomial is represented
in a Bernstein basis with nonnegative mixture coefficients so it remains a
valid probability density. The PDFs are normalized over the chosen fit range,
and changing the histogram bin count affects only the drawing.

An `Unbinned scan detail` slider controls the signal-shape search from 1
(`fastest`) through 5 (`finest`). Higher settings scan more peak-position,
width, and Crystal Ball tail candidates, retain more finalists, use a larger
preview sample, and run more mixture-fit refinement iterations. The fit summary
records the selected detail and number of scanned shapes. Both Poisson WLS and
unbinned likelihood fits operate independently in split views.

`Hide canvas fit results` suppresses the fit-optimization annotation boxes on
the active panel without disabling the fit curves or the textual Fit summary.
Each panel retains its own annotation visibility setting, which is useful when
fit boxes would obscure a dense facet layout.

The Fit card places its controls and fit-range information in a left column and
the statistics grid in a distinct right column. A single fit gets one summary
cell, while proton-sector fits use exactly three result columns so `S1`-`S3`
and `S4`-`S6` form two aligned rows; a sector-0 `CD` result is centered on a
third row. The two major columns stack only when the window becomes too narrow,
and the sector results reduce to one column on phone-sized screens. When both
signal and background are set to `none`, the fit-range and result cells are
hidden and the controls reclaim the full card width.

Default axis ranges are quantity-aware. Discrete detector codes such as `eDet`
receive half-category padding so the first and last detector bins are fully
visible. Missing-mass-squared quantities use padded 1st-to-99th-percentile
limits to keep rare extreme tails from dominating the initial display; the
events remain embedded and available through manually entered axis limits and
constraints.

Particle quantities use a consistent selector vocabulary regardless of whether
the input branch uses a short alias (`eIdx`, `pDet`, `g1Sector`) or a full name
(`electronIdx`, `protonDet`, `gamma1Sector`). Display labels use `REC`/`GEN`, the
full particle name, and a normalized quantity, while the underlying branch keys
remain unchanged. Duplicate short/full aliases are shown only once.

Proton-sector split views distinguish FD sectors 1-6 from sector 0, which
denotes CD protons. The FD facets occupy the first two three-column rows in
sector order, and the CD facet is centered by itself in the third row.
Repeated facet-axis titles are suppressed on interior panels; the bottommost
panel in each column and the leftmost panel in each row retain the relevant
titles, while the centered CD panel retains both. Numeric tick values remain
visible on every facet and use compact formatting without unnecessary trailing
zeros. The sidebar uses a narrower desktop width so more of the window is
available to the plot.

`Split by` also supports continuous numeric quantities. Selecting a numeric Z
quantity slices the active Y-versus-X histogram into equal-width facets over the
embedded Z range; the `Slices` input accepts between 1 and 24 intervals. Optional
comma-, space-, or semicolon-separated manual edges override the equal-width
count. Intervals are left-inclusive and right-exclusive, except for the final
interval, which includes its upper edge. Values outside a manual edge range are
omitted from the facets. Numeric slices retain the existing sector-facet behavior
for axes, fits, ghosts, reference curves, density, and color scales.

The sidebar presents the active dataset once in a framed identity card, gives the
constraint quantity, minimum, maximum, and action a compact single-row layout,
and uses content-sized buttons instead of stretching routine actions. Minimum
and maximum are independently optional, allowing lower-only (`x >= min`),
upper-only (`x <= max`), or bounded cuts. At least one bound is required when a
constraint is added.
Sample, selected-event, and embedded-event counts form a compact three-column
toolbar tile in that order, while active-axis means are hidden from the default
interface. Custom axis labels, bin counts, display minima and maxima, and
tick-density controls form one panel-specific tile beneath the active quantity
title. They use two aligned X/Y rows, with the vertical label pair immediately
to the right of the tick-density pair. One-dimensional plots pin Y ticks and Y
label to the same columns used in two-dimensional mode even while the Y-range
controls are hidden. The display tile includes panel-specific height and width
sliders in one-percent steps. Height runs from 25% to 100%, where 100% is the
former baseline canvas height, and now defaults to 50%. Width runs from 50% to
100% of the available panel and defaults to 100%. Profiles inherit both source
panel dimensions. Log color,
density, and color scale share that tile, which sits immediately to the right of
the axis settings and is followed
by the reset/export tile. All three tiles span the full title width as one
flush, spaced group above the canvas, with responsive wrapping when a pane is
too narrow. Hover
coordinates and bin values appear beneath the canvas.
In split view, the single toolbar moves between panel-specific rails with the
active panel. `Hide plot controls` collapses that rail and leaves a persistent
`Show plot controls` action in the panel controls. The rails compensate for
different title and active-filter badge heights, keeping the 1D and 2D canvases
flush whether the toolbar is expanded or collapsed. When controls are hidden,
the canvas reclaims the toolbar's measured height, enlarging the plotting area
without altering the panel's stored plot-aspect setting.
The panel control block provides separate `split view` and `shared panel
filters` buttons. Filter sharing is enabled by default. Disabling it gives each
panel its own topology selections, numeric constraints, and text searches;
switching panels restores the corresponding sidebar state, plot mask, filter
badge, and exported filter summary. Re-enabling sharing adopts the active
panel's filters for both panels. Plot variables and axis display ranges remain
panel-specific.
New panels start with log color and the color-scale readout enabled. The compact
topology filter starts collapsed and can be expanded with its adjacent button.
Color-scale hover readouts are clamped inside the plot pane so their numeric
values remain visible at the right edge.

The `Load File(s)` action sits in the right-side header utilities immediately
before the selected/embedded count tile. On startup, a full-window loading
screen remains visible while the embedded columns are decoded and the initial
plot state is constructed, then clears once the first visualization is ready.
With one sample, the upper-left identity badge retains the generated dataset
title. After more files are loaded, it becomes a workspace summary showing the
combined sample count and contributing filenames. Log color and color-scale
controls are shown only for 2D plots; 1D plots retain density as their sole
display toggle.

Right-clicking either canvas opens a plot menu with the ghost actions and a
`Hide plot controls` / `Show plot controls` action that mirrors the persistent
toolbar toggle. `Show mean guides` adds a vertical dashed line at the selected
events' mean X and, for two-dimensional plots, a horizontal dashed line at mean
Y. The toggle is panel-specific, follows the active filters and axis ranges,
uses each facet's own means, and is included in saved PNGs. A ghost snapshots
the active panel's current histogram bins and
remains fixed while filters change. One-dimensional
ghosts use a dashed orange step outline; two-dimensional ghosts use
intensity-weighted orange cell outlines so the live heat map remains readable.
Ghost state is independent for each panel and supports both ordinary and
faceted histograms. Changing to an incompatible variable, mode, or density view
temporarily hides the ghost without deleting it.

On a two-dimensional canvas, right-click inside a bin and choose `Profile X` or
`Profile Y` to open a one-dimensional slice in Panel 2. `Profile X` plots the X
distribution selected by the clicked Y bin; `Profile Y` plots the Y distribution
selected by the clicked X bin. The tool opens split view, makes panel filters
independent, copies the source panel's active filters into Panel 2, and adds the
clicked bin as an editable numeric constraint. Plot controls collapse by default
so the one- and two-dimensional canvases remain aligned; restore them with the
`Show plot controls` button or right-click action. Profiles created inside a facet
also retain that facet's categorical or numeric slice. Bin upper edges are
exclusive except for the final bin, matching the two-dimensional histogram's
bin membership exactly.

Right-click any two-dimensional plot and choose `Add function curve…` to open a
panel-local graphing calculator. Curves may be entered as either `y = f(x)` or
`x = f(y)`, with optional independent-variable domain limits and a custom label.
Expressions support arithmetic, powers, `pi`, `e`, and common functions such as
`sqrt`, `sin`, `cos`, `abs`, `exp`, `log`, `min`, `max`, and `pow`. Multiple
curves can be drawn simultaneously on ordinary or faceted plots and managed
independently for each panel. Each curve is associated with the plotted quantity
pair on which it was created, so it hides when the axes change and reappears when
that pair is restored. Curves default to a thin 1.25-pixel stroke; solid, dashed,
dotted, and dash-dot styles and widths from 0.5 to 3 pixels are selectable when
the curve is added.

For example, on a `Q2` versus `xB` plot, a minimum-`W` boundary can be drawn as
`y = (2^2 - 0.938272^2) * x / (1 - x)` with an optional domain such as
`0 < x < 1`. More generally, the calculator operates only in plotted
coordinates and is independent of quantity names, filters, and the underlying
physics interpretation.

Derived Operations uses a compact two-row builder in the sidebar beneath
Constraints, before the sidebar's axis-range separator. Only validation and
completion messages appear beneath the builder; the redundant expression
preview is omitted. Fit and Text Filters use full-width cards beneath the plots.
The quantity banner now leads each plot directly; redundant panel-name and
selected-quantity summaries above it are omitted. The active-panel tab is also
hidden until a second panel exists.

## Source layout

- `app.py` contains the current data adapters, payload builder, standalone HTML
  template, and browser application.
- `__main__.py` provides the package command.
- `tests/` contains visualizer-specific regression tests.
- `scripts/serve_visualizer.sh` remains at repository level because it is also
  useful for browsing other generated analysis artifacts.

The initial extraction deliberately preserves the existing implementation in
one module. Future changes can separate data adapters, derived quantities,
payload schemas, styles, plotting, filtering, and fitting behind regression
tests without changing the standalone HTML contract.

## Tests

```bash
python3 -m unittest discover -s visualizer/tests -v
```
