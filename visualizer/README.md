# Interactive visualizer

The visualizer is a standalone subsystem for interactive cut studies, detector
comparisons, and downstream extraction diagnostics. It converts compact NPZ
artifacts or selected ROOT trees into a self-contained HTML application. The
generated file embeds its event data and does not require a JavaScript build or
application server.

## Usage

Run the package from the repository root:

```bash
python3 -m visualizer data_events.npz \
  --output results/data_histograms.html

python3 -m visualizer selected_data.root \
  --format root --output results/selected_data_histograms.html \
  --dictionary build/libROOTBranchesDict.so
```

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

Proton-sector split views distinguish FD sectors 1-6 from sector 0, which
denotes CD protons. The FD facets occupy the first two three-column rows in
sector order, and the CD facet is centered by itself in the third row.
Repeated facet-axis titles are suppressed on interior panels; the bottommost
panel in each column and the leftmost panel in each row retain the relevant
titles, while the centered CD panel retains both. Numeric tick values remain
visible on every facet and use compact formatting without unnecessary trailing
zeros. The sidebar uses a narrower desktop width so more of the window is
available to the plot.

The sidebar presents the active dataset once in a framed identity card, gives the
constraint quantity, minimum, maximum, and action a compact single-row layout,
and uses content-sized buttons instead of stretching routine actions. Minimum
and maximum are independently optional, allowing lower-only (`x >= min`),
upper-only (`x <= max`), or bounded cuts. At least one bound is required when a
constraint is added.
Selected and embedded counts share one compact toolbar tile, while active-axis
means are hidden from the default interface. Reset/export actions occupy a
second tile directly beneath the event counts. Log color, density, and color
scale remain panel-specific and are centered beneath the active quantity title,
immediately above its canvas. Hover coordinates and bin values appear beneath
the canvas.
Color-scale hover readouts are clamped inside the plot pane so their numeric
values remain visible at the right edge.

Right-clicking either canvas opens a plot menu with `Make ghost`, `Replace
ghost`, and `Clear ghost` behavior. A ghost snapshots the active panel's
current histogram bins and remains fixed while filters change. One-dimensional
ghosts use a dashed orange step outline; two-dimensional ghosts use
intensity-weighted orange cell outlines so the live heat map remains readable.
Ghost state is independent for each panel and supports both ordinary and
faceted histograms. Changing to an incompatible variable, mode, or density view
temporarily hides the ghost without deleting it.

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
