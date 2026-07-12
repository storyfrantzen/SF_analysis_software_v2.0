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

Default axis ranges are quantity-aware. Discrete detector codes such as `eDet`
receive half-category padding so the first and last detector bins are fully
visible. Missing-mass-squared quantities use padded 1st-to-99th-percentile
limits to keep rare extreme tails from dominating the initial display; the
events remain embedded and available through manually entered axis limits and
constraints.

Proton-sector split views distinguish FD sectors 1-6 from sector 0, which
denotes CD protons. The FD facets occupy the first two three-column rows in
sector order, and the CD facet is centered by itself in the third row.
Repeated facet-axis titles and tick labels are suppressed on interior panels;
the bottommost panel in each column and the leftmost panel in each row retain
the relevant labels, while the centered CD panel retains both axes. Numeric
ticks use compact formatting without unnecessary trailing zeros. The sidebar
uses a narrower desktop width so more of the window is available to the plot.

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
