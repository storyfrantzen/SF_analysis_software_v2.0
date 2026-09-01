# EPPI0 model comparison

The model-comparison layer puts theory and the extracted RGK/RGA observables on
the same convention before comparing them. It accepts either an AAO executable
or a tabulated prediction, averages the prediction over the selected physical
analysis bins, applies the same stored bin-centering transformation as the
data, fits the same phi harmonics, and exports structure functions with their
full propagated data covariance.

The harmonic convention is

```text
d2sigma/(dt dphi) = A + B cos(phi) + C cos(2 phi)

2 pi d2sigma/(dt dphi)
  = sigma_U
  + sqrt(2 epsilon (1 + epsilon)) sigma_LT cos(phi)
  + epsilon sigma_TT cos(2 phi)
```

where `sigma_U = sigma_T + epsilon sigma_L`. Consequently,

```text
sigma_U  = 2 pi A
sigma_LT = 2 pi B / sqrt(2 epsilon (1 + epsilon))
sigma_TT = 2 pi C / epsilon
```

At one beam energy the data determine `sigma_U`, not `sigma_T` and `sigma_L`
separately. Epsilon is evaluated at the cross-section artifact's stored flux
coordinates, inverse-variance averaged over the accepted phi points in each
harmonic fit. The complete `(A,B,C)` covariance is transformed as `J C J^T`.

## Tabulated GPD-model input

CSV headers are case-insensitive. Kinematics use `xB`, `Q2` in `GeV^2`, and
either positive `minus_t` or signed negative `t`, also in `GeV^2`. Structure
functions are in `nb/GeV^2`; reduced cross sections are in
`nb/(GeV^2 rad)`. The table may use one of three schemas.

Separated structure functions are preferred because they can be evaluated at
the different RGK and RGA epsilon values:

```csv
xB,Q2,minus_t,sigma_T,sigma_L,sigma_LT,sigma_TT
0.20,2.0,0.15,12.1,1.8,2.4,-3.2
```

An energy-specific unseparated table is also accepted:

```csv
xB,Q2,minus_t,sigma_U,sigma_LT,sigma_TT
0.20,2.0,0.15,13.4,2.4,-3.2
```

Alternatively, provide the full phi-dependent reduced cross section:

```csv
xB,Q2,t,phi_deg,reduced_cross_section
0.20,2.0,-0.15,0,2.31
0.20,2.0,-0.15,12,2.28
```

`phi_deg` is periodic. Cartesian tables use regular-grid interpolation;
scattered tables use N-dimensional interpolation. The default is linear.
`--interpolation nearest` is available for sparse inputs. Neither mode
extrapolates outside the table's kinematic envelope: uncovered evaluation
points are counted as model failures and the corresponding output bin is
rejected unless its failure fraction is allowed explicitly.

## Build a model prediction

For an RGK GPD table:

```bash
python3 analysis/run_analysis.py model-prediction \
  --config configs/analysis/rgk/6.535.json \
  --cross-section results/rgk/cross_section.npz \
  --table models/gk_rgk.csv --model-name "GK" \
  --N 4 --output results/rgk/models/gk.npz
```

For RGA, use its configuration and cross-section artifact:

```bash
python3 analysis/run_analysis.py model-prediction \
  --config configs/analysis/rga/10.604.json \
  --cross-section results/rga/cross_section.npz \
  --table models/gk_rga.csv --model-name "GK" \
  --N 4 --output results/rga/models/gk.npz
```

The command samples `N^4` midpoint cells in every `(Q2,xB,-t,phi)` bin,
retains the exclusive physical region and configured phase-space cuts, and
forms the extraction-equivalent prediction

```text
<Gamma * reduced model cross section> / Gamma(reference).
```

If the data artifact contains `bin_centering_C_BC`, that prediction is divided
by the same stored `C_BC`. This is the correct quantity to compare with the
already centered data; the pre-transformation average, simple reduced average,
center value, sampling coordinates, coverage, and failure counts remain in the
model artifact for audit. By default only data-valid bins are evaluated. Use
`--all-bins` for coverage studies.

An AAO executable can be evaluated through the same layer:

```bash
python3 analysis/run_analysis.py model-prediction \
  --config configs/analysis/rgk/6.535.json \
  --cross-section results/rgk/cross_section.npz \
  --aao-exe /path/to/aao_xsec --theory 5 --channel 1 --resonance 0 \
  --model-name "AAO theory 5" --workers 8 --N 4 \
  --output results/rgk/models/aao_theory5.npz
```

The model name is only a label. Do not describe an AAO mode as a GPD model
unless that particular executable and theory selection implements one.

For production grids, split over flattened `(Q2,xB,-t)` bins:

```bash
python3 analysis/run_analysis.py model-prediction \
  --config configs/analysis/rgk/6.535.json \
  --cross-section results/rgk/cross_section.npz \
  --table models/gk_rgk.csv --model-name "GK" --N 6 \
  --bin-chunks 100 --bin-chunk-index 0 \
  --output results/rgk/models/parts/gk_part000.npz

python3 analysis/run_analysis.py model-prediction-merge \
  results/rgk/models/parts/gk_part*.npz \
  --output results/rgk/models/gk.npz
```

The merge rejects overlaps, incomplete 3D coverage, incompatible binning, and
mixed model/source/configuration metadata.

## Export data structure functions

```bash
python3 analysis/run_analysis.py structure-functions \
  results/rgk/harmonics.npz \
  --cross-section results/rgk/cross_section.npz \
  --config configs/analysis/rgk/6.535.json \
  --output results/rgk/structure_functions.npz
```

Production harmonic `quality_mask` selection is applied by default. The output
contains `sigma_U`, `sigma_LT`, `sigma_TT`, uncertainties, full `3x3`
covariance, epsilon, reference coordinates, and a validity mask. Use
`--include-quality-rejected` only for a diagnostic variation.

## Plot and quantify comparisons

Any number of model artifacts can be overlaid:

```bash
python3 analysis/run_analysis.py model-comparison-plots \
  results/rgk/cross_section.npz results/rgk/harmonics.npz \
  results/rgk/models/gk.npz results/rgk/models/jm.npz \
  --config configs/analysis/rgk/6.535.json \
  --output-dir results/rgk/model_comparison
```

The output directory contains:

- `model_comparison_vs_phi.pdf`: data points, the accepted data harmonic fit,
  forward-averaged model points, and each model's common harmonic fit;
- `model_comparison_structure_functions.pdf`: `sigma_U`, `sigma_LT`, and
  `sigma_TT` versus `-t` for each `(Q2,xB)` bin;
- `model_comparison_summary.csv`: data/model harmonics, structure functions,
  structure-function pulls, and
  `(h_data-h_model)^T C_data^-1 (h_data-h_model)` using the full data harmonic
  covariance.

The tiny uncertainties stored for model harmonic fits are numerical weights
used to run the same fitter, not theory errors. The comparison therefore uses
experimental covariance only unless theory uncertainties are supplied and
handled in a later extension.
