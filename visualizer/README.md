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

The Fit panel supports ordinary least squares and Poisson-weighted least
squares for 1D count histograms. `Poisson WLS (Pearson)` iteratively solves the
normal equations with per-bin variance `max(expected count, 1)` and reports the
Pearson chi-square per degree of freedom. It applies to polynomial backgrounds
and Gaussian or Crystal Ball signal-plus-background fits, including split
views. Poisson weighting is intentionally unavailable in density mode because
normalized bin fractions are not Poisson counts.

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
