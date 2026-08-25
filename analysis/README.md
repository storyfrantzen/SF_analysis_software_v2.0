# EPPI0 analysis

This directory is the maintained replacement for the legacy
`SF_analysis_software/analysis_v2.0` scripts.  The numerical core is separated
from ROOT I/O so response construction, unfolding, and normalization can be
unit-tested with ordinary NumPy arrays.

## Event-sample contract

An acceptance sample must contain one row per generated event, including events
with no accepted reconstructed candidate.  Each row needs:

- generated `Q2`, `xB`, `-t`, and Trento phi;
- a source-aware event key identifying the HIPO file and ordinal event within it;
- reconstructed values for the same coordinates, or `NaN` when no candidate is
  reconstructed;
- a reconstructed-selection flag;
- the reconstructed proton detector and selected-photon FT multiplicity;
- the generated-event weight, defaulting to one.

Reconstructed DIS and final-state cuts may be applied to
`rParticles` only after the converter has filled `gEvents`.
They define the reconstructed numerator and do not alter the generated denominator.
Generated phase-space cuts are applied later to the compact truth coordinates.
For legacy files without `gEvents`, do not apply reconstructed event
filters during conversion because the unmatched particle rows are then the only
source of the generated denominator.

Use `rEvents` for reconstructed event-level bookkeeping and
topology multiplicities. `rParticles` is particle-level and
repeats a legacy event-key object only for backward compatibility. The full
schema contract is documented in
`docs/root_tree_schema.md`.

Radiative generator events are interpreted as `e p pi0 gamma`; non-radiative
events are interpreted as `e p gamma gamma`.  Extra reconstructed photons are
allowed and the reconstructed pi0 candidate is the photon pair nearest the pi0
mass.

## Core modules

- `eppi0.binning`: the legacy CLAS12 binning and exact legacy flat-index order;
- `eppi0.response`: sparse migration matrix, efficiency, and feed-in metadata;
- `eppi0.unfolding`: iterative Bayesian unfolding and reproducible bootstrap;
- `eppi0.cross_section`: luminosity, virtual-photon flux, physical bin volume,
  and reduced cross sections;
- `eppi0.bin_centering`: physical-bin AAO model averaging and bin-centering
  corrections;
- `eppi0.exclusivity`: deterministic topology-aware data/GEMC windows with a
  non-mutating N-1 stability audit;
- `eppi0.event_sample`: radiative/non-radiative GEN construction and REC joins;
- `eppi0.data_efficiency`: run-charge joins, current grouping, and zero-current fits;
- `eppi0.harmonics`: weighted `A + B cos(phi) + C cos(2 phi)` fits.

## Dependencies

- Python 3.10 or newer;
- NumPy and SciPy for the numerical pipeline;
- Matplotlib for diagnostic PDF output;
- PyROOT for `build_event_sample.py` and `export_selected_data.py`;
- the project ROOT dictionary when object branches are not already discoverable.

Run the dependency-light tests from the repository root:

```bash
python3 -m unittest discover -s analysis/tests -v
```

## Generic ROOT distribution comparisons

`compare_root_distributions.py` overlays any number of labeled ROOT samples and
uses one labeled sample as the reference for ratio panels and quantitative shape
comparisons. The labels and paths are supplied entirely on the command line:

```bash
python3 analysis/compare_root_distributions.py --density \
  --sample reference=/path/to/reference_selected.root \
  --sample candidate_a=/path/to/candidate_a_selected.root \
  --sample candidate_b=/path/to/candidate_b_selected.root \
  --reference reference \
  --output-dir results/root_comparison
```

Every compared column receives a Jensen-Shannon divergence and total-variation
distance in `shape_metrics.csv`; both compare normalized bin probabilities over
the displayed histogram range and are independent of the `--density` display
choice. The divergence uses natural logarithms and ranges from zero to
`ln(2)`; total variation ranges from zero to one. `summary.json` contains the
same metrics together with per-column entry, mean, RMS, minimum, and maximum
values.

Pair a selected sample with its converter output to include processing counters,
accumulated charge, and the ratio of selected rows to input events in
`sample_summary.csv`:

```bash
python3 analysis/compare_root_distributions.py \
  --sample reference=/path/to/reference_selected.root \
  --sample candidate=/path/to/candidate_selected.root \
  --processing-root reference=/path/to/reference_converter.root \
  --processing-root candidate=/path/to/candidate_converter.root \
  --reference reference \
  --output-dir results/root_comparison
```

The row-to-event ratio is a selection fraction only for trees with one selected
row per event. For particle-level trees it is intentionally reported as
`selected_rows_per_input_event`, not as an efficiency. Use `--where`, `--tree`,
and `--columns` to apply the same comparison machinery to other ROOT schemas;
the default tree is the canonical selected-event tree, `sEvents`.

For the RGA 10.604 GeV workflow, use
`configs/analysis/rga/10.604.json` together with the matching processing and
post-processing files under `configs/*/rga/10.604/`.

The core numerical stages are exposed through one command:

```bash
python3 analysis/run_analysis.py response mc_events.npz \
  --config configs/analysis/rgk/6.535.json --output-dir results/response

python3 analysis/run_analysis.py radiative-correction born_lund/ rad_lund/ \
  --config configs/analysis/rgk/6.535.json --output results/C_rad.npz \
  --born-normalization-file born_lund/aao_norad.norm \
  --radiative-normalization-file rad_lund/aao_rad.norm \
  --progress-chunks 1 \
  --diagnostic-pdf results/C_rad_diagnostics.pdf \
  --diagnostic-csv results/C_rad_diagnostics.csv

python3 analysis/run_analysis.py unfold data_events.npz \
  results/response/response_matrix.npz results/response/response_meta.npz \
  --config configs/analysis/rgk/6.535.json --output results/unfolding.npz \
  --background-cuts results/data_exclusivity.npz \
  --current-efficiency-correction \
    results/data_efficiency/rgk_6.535/current_efficiency_correction.json \
  --radiative-correction results/C_rad.npz

python3 analysis/run_analysis.py bin-centering \
  --config configs/analysis/rgk/6.535.json --output results/C_BC.npz \
  --exe /path/to/aao_xsec --N 4 --workers 8 --progress-chunks 10

python3 analysis/run_analysis.py cross-section results/unfolding.npz \
  --config configs/analysis/rgk/6.535.json --output results/cross_section.npz \
  --bin-centering results/C_BC.npz

python3 analysis/run_analysis.py fit-harmonics results/cross_section.npz \
  --output results/harmonics.npz
```

`unfold --background-cuts` performs the nonpeaking-background subtraction at
the reconstructed-yield stage, before feed-in subtraction and D'Agostini
unfolding.  For each retained proton/photon topology it refits
`rec_m_gg` in the final N-1 sample (all other exclusivity cuts applied), uses
the nominal `rec_m_gg` window as the signal region, and uses the remainder of
the fitted domain as sidebands.  The fitted linear-background shape determines
the topology-specific transfer factor

```text
alpha = fitted background in signal window / fitted background in sidebands,
N_background(bin, topology) = alpha(topology) * N_sideband(bin, topology).
```

Current-efficiency event weights, when requested, are applied to both regions.
The saved variance contains signal-region counting variance, transferred
sideband counting variance, and the diagonal contribution from the bootstrapped
transfer-factor uncertainty.  The unfolding NPZ keeps the raw signal-region,
sideband, estimated-background, and subtracted spectra together with every fit
window and transfer factor.  The cross-section and harmonic artifacts carry the
background-subtraction provenance fields downstream.

The sideband and cut table must come from the same compact data sample.  If
`--selection-mask` is also supplied, `unfold` verifies that it exactly matches
the signal region reconstructed from `--background-cuts`; otherwise it stops.
The first implementation deliberately accepts only the default pooled
global-by-topology cut tables; local kinematic cut tables would conflate a
changing signal definition with the background-shape estimate.
Literal bin-by-bin subtraction can fluctuate below zero in sparse bins.
The default `--background-negative-policy error` stops if this occurs in a
reconstructed bin connected by the response matrix to any truth bin above the
configured acceptance threshold. After inspecting the recorded spectra,
`--background-negative-policy clip` explicitly permits nonnegative clipping
and records the number and total deficit of clipped bins. It is never silent.
Change the transfer-factor uncertainty sampling with
`--background-alpha-bootstrap`; zero disables that uncertainty component.

For production MC response building, avoid serializing one dense row per
generated event. Build the sparse response directly from the converter and
selected ROOT files:

```bash
python3 analysis/run_analysis.py response-root \
  6.535_rgk_eppi0_GEMC.root selected_mc.root \
  --config configs/analysis/rgk/6.535.json --output-dir results/response \
  --dictionary build/libROOTBranchesDict.dylib
```

This command histograms the `gEvents` denominator in chunks, joins only
selected REC candidates by `(sourceFileId, sourceEventIndex)`, and writes the
same `response_matrix.npz` and `response_meta.npz` consumed by `unfold`.
`build_event_sample.py` remains useful for compact debug samples and for
backward compatibility with older particle-level matched files. New files join
on `(sourceFileId, sourceEventIndex)`, because GEMC files can all have run 11
and restart their event numbers.
`export_selected_data.py` creates the compact data artifact and carries the
converter's accumulated charge into the pipeline. New converter files also
provide a `RunCharge` tree; the exporter preserves its run numbers, charges,
and QADB event counters as `beam_charge_run`, `beam_charge_by_run_c`,
`run_total_events`, `run_passed_qadb_events`, and
`run_failed_qadb_events`. This permits charge-normalized run filtering without
splitting the original HIPO production by run. The artifact also records the
selected electron, proton, and two photon detector IDs and the derived
`rec_ft_photon_count`, whose values 0, 1, and 2 are independent of photon
ordering.

## Data current-efficiency study

`study_data_efficiency.py` joins the selected-event run numbers to the QADB
charge arrays and `configs/efficiency/rgk/6.535/run_currents.json`. It writes a
complete per-run audit table, charge-aggregated current-group yields, a linear
zero-current fit, and a multipage diagnostic PDF. Its conservative default fit
uses only unflagged P3 and P4 runs:

```bash
python3 analysis/study_data_efficiency.py \
  results/data/rgk_6.535_data_events.npz \
  --output-dir results/data_efficiency/rgk_6.535_preliminary
```

Without `--selection-mask`, this is explicitly labeled as a raw
selected-candidate study rather than a background-subtracted signal-efficiency
measurement. After deriving and freezing one event-level signal mask, apply the
same mask to every current:

```bash
python3 analysis/study_data_efficiency.py \
  results/data/rgk_6.535_data_events.npz \
  --selection-mask results/data_exclusivity.npy \
  --include-classes P3 P4 L5 \
  --output-dir results/data_efficiency/rgk_6.535_fixed_selection
```

As a systematic check, replace the integer selected count by the same
topology-dependent sideband subtraction used upstream of unfolding:

```bash
python3 analysis/study_data_efficiency.py \
  results/data/rgk_6.535_data_events.npz \
  --background-cuts results/cuts/data_exclusivity.npz \
  --selection-mask results/data_exclusivity.npy \
  --include-classes L4 L5 P3 P4 \
  --output-dir results/data_efficiency/rgk_6.535_sideband_systematic
```

`--background-cuts` performs one common global-by-topology N-1
`m_gg` fit, uses each fitted linear-background transfer factor `alpha_g`, and
forms the run yield

`N_r = N_signal,r - sum_g alpha_g N_sideband,r,g`.

The optional selection mask becomes a consistency assertion: the command stops
if it differs from the signal window reconstructed from the supplied cut table.
Counting uncertainties use `N_signal + sum_g alpha_g^2 N_sideband,g`. The
bootstrap uncertainty of each common `alpha_g` is recorded in
`fit_summary.json` as a correlated systematic and is deliberately not treated
as an independent statistical error for every run or run class.

Include L5 only after confirming that its physics trigger and prescale are
compatible with P3/P4. L4 trigger tests, mixed/random-trigger L6 runs, the E2
empty-target run, and half-torus T runs are excluded unless explicitly admitted
with `--include-classes` or `--include-run`. Suspect RCDB currents remain
excluded unless their label is added with `--include-qualities`.

For each run class, the command sums signal counts and charge before taking the
ratio. Its effective current is charge weighted. The default fit therefore uses

`N_k = sum_r N_r`, `Q_k = sum_r Q_r`,
`I_k = sum_r(Q_r I_r) / Q_k`, and `Y_k = N_k / Q_k`.

The output directory contains:

- `run_yields.csv`: every charge-bearing run, its counts, charge, current
  metadata, signal-region and sideband counts, estimated background, net signal,
  inclusion decision, and exclusion reason;
- `current_group_yields.csv`: charge-aggregated fit points and their relative
  efficiencies;
- `fit_summary.json`: inputs, filters, charge validation, fit covariance,
  warnings, zero-current result, and the selected GEMC reference response;
- `current_efficiency_correction.json` (when GEMC inputs are supplied): the
  fitted data and GEMC models, `D(I)`, reference-response fingerprint, all
  usable run currents, nominal per-run unfolding weights, downstream selection,
  and the resulting analysis beam charge;
- `data_efficiency_diagnostics.pdf`: current-dependence, fit pulls, and
  included-run stability plots. With GEMC inputs, the lower panel of the first
  page is `D(I) = eta_data(I) / eta_MC(I)`. The upper panel displays
  unit-normalized data and GEMC efficiencies, their slopes, and the propagated
  data-fit uncertainty; the ratio panel intentionally has no uncertainty band.
  Its automatic vertical ranges retain run-level outliers, while its compact
  panels and fitted-domain current range avoid unnecessary page space. Excluded
  runs remain visible on the current plot as crosses colored and labeled by their
  manifest run class. The run-stability page uses a compact charge panel and
  draws each charge-weighted run-group yield across the full run range as a
  matching dotted line.

Use `--fit-level runs` only as a diagnostic. The nominal group-level fit avoids
treating the many runs within one production setting as independent current
settings.

To prevent a negligible-charge run from moving the effective current of a
run-class group, set a minimum within-group QADB-charge contribution:

```bash
python3 analysis/study_data_efficiency.py data_events.npz \
  --selection-mask results/data_exclusivity.npy \
  --include-classes L4 L5 P4 P3 \
  --minimum-group-charge-fraction 0.01 \
  --output-dir results/data_efficiency/rgk_6.535_charge_filtered
```

Here `0.01` means one percent, not one percent per run. The fraction is
calculated once within each run class after current-quality and explicit-run
filters. Rejected runs remain in `run_yields.csv` with
`below_minimum_group_charge_fraction`, and their calculated fraction remains
available in `group_charge_fraction`. The procedure is intentionally not
iterative, so the denominator and rejected set do not depend on removal order.

An optional GEMC manifest adds accepted/generated efficiencies, a linear
current fit, and an overlay on the first diagnostics page. Each response must
have been built with the same analysis binning and the same fixed numerical MC
exclusivity definition:

```json
{
  "schema_version": 1,
  "samples": [
    {
      "label": "no_background",
      "current_nA": 0.0,
      "response_meta": "/path/to/no_background/response_meta.npz"
    },
    {
      "label": "merged_60nA",
      "current_nA": 60.0,
      "response_meta": "/path/to/merged_60nA/response_meta.npz"
    }
  ]
}
```

Pass it with `--gemc-manifest /path/to/gemc_efficiency_manifest.json`. If there
is exactly one positive-current GEMC point, that point is inferred to be the
reference response used downstream. Otherwise select the actual response with
`--reference-current-na`. For example:

```bash
python3 analysis/study_data_efficiency.py data_events.npz \
  --selection-mask results/data_exclusivity.npy \
  --include-classes L4 L5 P4 P3 \
  --gemc-manifest /path/to/gemc_efficiency_manifest.json \
  --reference-current-na 60 \
  --exclude-class-downstream T \
  --exclude-class-downstream E2 \
  --output-dir results/data_efficiency/rgk_6.535
```

The GEMC numerator is `sum_i(truth_total_i * efficiency_i)`, which counts selected
events generated and reconstructed inside the analysis phase space without
including feed-in. The denominator is `sum_i(truth_total_i)`. Relative data
and GEMC efficiencies are both normalized to their fitted zero-current
intercepts on the overlay. `gemc_efficiency_points.csv` and the complete GEMC
fit are also written.

The scalar correction consumed by `unfold` is

`w_r = eta_MC(I_ref) / eta_data(I_r)`

for each selected event from run `r`. Equivalently,

`w_r = [eta_data(I_ref) / eta_data(I_r)] / D(I_ref)`.

The weighted reconstructed histogram is unfolded with the response built at
`I_ref`. Ordinary study filters (`--include-classes`, `--include-run`,
`--exclude-run`, current quality, and minimum group charge) determine only
which runs constrain the current fit; they do not remove runs from the physics
dataset. To remove runs downstream, repeat `--exclude-run-downstream RUN` or
`--exclude-class-downstream CLASS`. The correction artifact assigns those runs
zero event weight, and `unfold` removes them before the signal/background
histograms are built. It also sets the analysis charge to the original
file-level charge minus the excluded per-run charges, so excluded runs
contribute neither yield nor luminosity while the no-exclusion case preserves
the converter's normalization exactly. For example,
`--exclude-class-downstream T` removes all half-torus runs. This explicit
distinction prevents a fit-only exclusion from silently changing the physics
sample.

Every charge-bearing run that remains downstream must have a usable current.
A run without current metadata must either be repaired in the manifest or
explicitly excluded downstream; otherwise artifact construction stops. The
artifact records both original and analysis charge, included/excluded run lists,
fit inclusion, analysis inclusion, exclusion reasons, and the applied weight
for every run.

`unfold` verifies the SHA-256 fingerprint of `response_meta.npz` against the
chosen reference sample before applying any weights. It stores the complete
correction JSON, the reference current, `D(I_ref)`, weight range, unweighted
counts, weighted counts, `sum(w^2)`, original/analysis beam charge, and the
downstream-excluded run list in its output. Weighted bin means and
bootstrap statistical fluctuations use the same event weights. The fit-model
uncertainties recorded per run in the JSON are correlated calibration
uncertainties; they are intentionally preserved for systematic variations and
are not folded into the statistical `corrected_uncertainty` array.

The same two points can be supplied directly without creating a JSON file:

```bash
python3 analysis/study_data_efficiency.py data_events.npz \
  --selection-mask results/data_exclusivity.npy \
  --gemc-sample no_background 0 /path/to/no_background/response_meta.npz \
  --gemc-sample merged_60nA 60 /path/to/merged_60nA/response_meta.npz \
  --output-dir results/data_efficiency/rgk_6.535_data_gemc
```

Use the same generated events for the no-background and merged samples when
possible. The tool requires identical response bin edges and warns when the
truth totals differ. With only the zero-background and one merged-current
sample, the straight line is the assumed two-point model and cannot test
curvature. For weighted response metadata, provide
`statistical_uncertainty` in the corresponding manifest entry when a validated
uncertainty is available; otherwise the reported binomial uncertainty is only
an effective-weight approximation.

For interactive cut studies and quick detector/topology comparisons, build a
standalone histogram browser from either compact NPZ samples or selected ROOT
trees:

```bash
python3 -m visualizer data_events.npz \
  --output results/data_histograms.html

python3 -m visualizer selected_data.root \
  --format root --output results/selected_data_histograms.html \
  --dictionary build/libROOTBranchesDict.so

scripts/serve_visualizer.sh results/selected_data_histograms.html
```

The visualizer now lives in the top-level `visualizer/` package. The historical
`analysis/interactive_histograms.py` command remains as a compatibility wrapper.
See `visualizer/README.md` for its source layout and visualizer-specific tests.

Run `scripts/serve_visualizer.sh` with no path to serve a click-enabled listing
of `results/`, or `scripts/serve_visualizer.sh .` to serve the working tree. To
compare multiple event samples in one browser session, first generate an HTML
visualizer for each input file, open one of them from the served directory, then
use the top `Load File(s)` button to browse the same farm-side HTTP directory
tree and add the other generated visualizer HTML files. The loaded events are
appended in the browser and tagged with a `Sample` category, which can be used
in `Filter topology`, `Split by`, or the full category list.
The browser discovers stored scalar quantities, adds common derived views such
as degree versions of angular branches and non-duplicated signed/unsigned `t`
aliases, computes `t_min` and `t'` from existing `Q2`, `xB`, and `t` branches
where available, and supports 1D histograms, 2D histograms, detector/pass-flag
toggles, text filters, numeric constraints, and optional panel tabs that can be viewed
independently or side by side under the same filters from the toolbar above the
plot. Quantity dropdowns show visible category headings for event, kinematic,
particle, selection, and detector/exclusivity groups instead of a flat
alphabetical list. Axis tick-count
sliders adjust every displayed axis, optional presentation labels can override
the displayed X/Y axis labels per panel, and the visible display can be saved as
a PNG. 2D views can optionally draw color scales beside the histogram, including
per-sector scales in split views and hover markers showing hovered-bin values
on primary and overlay scales. Category
filters also have a collapsible `Filter topology` selector near the split
control, with numeric `Constraints` placed directly underneath it for quick
filtering. The full category list is kept as the final compact reference section
below the plot, after derived operations, fit controls, reset buttons, PNG
export, and the preview table. Active filters are also flagged
above each plot with a compact badge listing the number of active filter
dimensions and a short summary. Common display/actions such as log
color, density, reset, and PNG export sit above the plot. Density mode
normalizes bin contents while auto-scaling the vertical and color ranges to the
normalized peak, so the display does not collapse when densities are below one.
Compact `+` buttons
next to the axis selectors add optional comparison X/Y quantities on the same
axes with a distinct color map for direct within-panel comparisons. The
derived-operations menu can add browser-side comparison variables such as
`rec_theta_deg - gen_theta_deg`, ratios, fractional residuals, or sums, and the
fit menu can overlay a signal model `S` plus a background model `B`, with
Gaussian or Crystal Ball signal shapes and constant or polynomial degree 1-5
backgrounds where applicable. Background-only polynomial trend fits remain
available for 2D views, and 1D split views get independent sector-by-sector
fits with compact in-panel fit labels. Fit ranges can be selected by enabling
`click endpoints` and clicking two X positions on the plot; the displayed
histogram is unchanged while only the fit calculation and curve interval are
restricted. One-dimensional count fits can use ordinary least squares or
Poisson-weighted least squares with iterative Pearson weights; Poisson weighting
is disabled for density-normalized histograms. Conditional unbinned likelihood
fits instead use the individual selected values, report a signal fraction, and
use positive Bernstein background PDFs; histogram bins affect only their
display. Their scan-detail slider trades execution speed for a denser search of
signal position, width, and Crystal Ball tail shapes while retaining the same
constant/polynomial degree 1-5 background choices as the binned fits. Text
filters appear only when the input contains string-valued columns. Selected ROOT
inputs
expose the richest set of reconstruction filters, including `pDet`, `passFiducial`,
`passSamplingFraction`, `passExclusivity`, and selected-particle kinematics
such as `protonTheta`, so its correlation with `-t` can be explored directly.
Diagnostic post-processing outputs also provide `evaluatedCuts` and
`failedCuts`. The visualizer expands them into numeric `passCut_*` quantities,
with `1` for pass, `0` for failure, and `NAN` when that cut was not evaluated
for a row.
Newly post-processed selected files also expose selected-particle indices and
sectors, including `pIdx`, `g1Idx`, `g2Idx`, `pSector`, `g1Sector`, and
`g2Sector`, so proton and photon selections can be filtered like the electron.
When split photon quantities are available, the visualizer hides the generic
first-photon `gamma` aliases in favor of explicit `gamma1` and `gamma2` groups.
The `passSamplingFraction` flag is labeled as a cut result; selected ROOT inputs
also derive total and PCAL/ECIN/ECOUT electron sampling fractions from the stored
calorimeter energies divided by `electronP`.
Joined generated/reconstructed event samples made by `build_event_sample.py`
carry every scalar branch from the selected reconstructed tree with a `rec_`
prefix without interpreting particle roles. Post-processing itself writes the
generic role selection as standardized scalar branches such as `electronP`,
`protonP`, `gamma1P`, and `gamma2P`, with matching index, PID, detector, sector,
theta, and phi fields. Thus the NPZ builder remains schema-agnostic while MC
acceptance visualizers expose the same reconstructed filters and kinematic
branches as the selected data visualizers after the ROOT, `.npz`, and HTML are
regenerated. Newly converted MC files store complete per-particle GEN/LUND
kinematics in `gEvents`; `build_event_sample.py` carries those columns,
including `gen_electronP`, `gen_protonTheta`, `gen_gamma1Phi`, `gen_gamma2P`,
and `gen_pi0P`, so generated-vs-reconstructed residuals can be built directly
in the visualizer. Older converter ROOT files fall back to the available
`rParticles.gen` rows, which may be less complete.
The dictionary is optional for ordinary `sEvents` trees; if the named
dictionary is missing, the script continues with ROOT's built-in scalar and STL
branch readers.
The selected tree's default `t` branch remains the proton-based positive `-t`,
computed from the reconstructed recoil proton. New post-processed files also
store `t_pi0`, the positive `-t` computed from the reconstructed pi0 side,
for event-by-event comparison.

`acceptance-plots` visualizes four related response diagnostics. For bin `i`,
with `N_same,i = N(rec i and gen i)`, they are:

- `A_i = N_rec,i / N_gen,i`, the simple bin-by-bin acceptance;
- `P_i = N_same,i / N_rec,i`, the bin purity;
- `E_i = N_same,i / N_gen,i`, the same-bin efficiency;
- `epsilon_i = sum_j R[j,i]`, the total truth-bin efficiency used by IBU.

The default phi overlay includes `A_i`, `E_i`, and `epsilon_i`. Add
`--include-purity` to include `P_i`, whose scale can differ substantially from
the other three diagnostics. Add `--quilt` to prepend one stitched `Q2`-by-`xB`
page per `-t` bin to the phi PDF. Quilts share a page-wide y scale by default;
use `--quilt-scale-mode panel` for independent panel scales.

The full unfolding still uses the migration matrix `R[j,i]`, not any one of
these scalar diagnostics alone.

To inspect that sparse IBU response matrix directly, use:

```bash
python3 analysis/run_analysis.py response-plots \
  results/response/response_matrix.npz results/response/response_meta.npz \
  --output results/response/response_diagnostics.pdf
```

This PDF includes a sparse global image of `R[reco, truth]`, collapsed migration
matrices for `Q2`, `xB`, `-t`, and phi, migration-probability histograms, and
projection heatmaps showing which kinematic variables drive migration in
`(xB,Q2)` and `(phi,-t)` space. Coarse heatmaps include nonzero in-cell values,
and the projection pages use adaptive color scales so small migration
probabilities remain visible.

The compact tree path does not require `--beam-energy`, because generated
kinematics were calculated by the converter. For a legacy particle-level input,
the adapter requires `--beam-energy` to reconstruct those quantities:

```bash
python3 analysis/build_event_sample.py \
  6.535_rgk_eppi0_GEMC.root selected_mc.root mc_events.npz
```

For generated-versus-reconstructed visualization of a large production, use
the reverse join instead of materializing one output row per generated event:

```bash
python3 analysis/build_event_sample.py \
  6.535_rgk_eppi0_GEMC.root \
  6.535_rgk_eppi0_GEMC_selected.root \
  samples/rgk_6.535_matched_mc_events.npz \
  --dictionary build/libROOTBranchesDict.so \
  --matched-only --chunk-size 250000 --progress-chunks 4
```

`--matched-only` uses selected reconstructed candidates as the left table and
streams `gEvents` in chunks. It joins on
`(sourceFileId, sourceEventIndex)` and exports only candidates with valid
generated topology and kinematics. Every output row therefore has
`rec_selected=true` and carries both the generated and reconstructed scalar
columns used by the visualizer. This artifact is intended for reconstruction
diagnostics, not as the generated denominator for response construction.

`derive_exclusivity.py` uses a deterministic physics-ordered bootstrap followed
by a fixed-window N-1 stability audit and preserves the global/per-bin modes.
Global-by-topology pooling is the default because this exclusive channel is
generally too sparse for stable fine-bin fits. Pass
`--per-bin-cuts` to opt into local `(Q2, xB, -t)` groups with same-topology
fallbacks; the explicit `--global-cuts` spelling remains accepted. The signal
model follows the geometry of each exclusivity quantity:

- `rec_m_gg`: Gaussian signal with the linear-background slope constrained by
  sidebands;
- `rec_pT_miss`: Rice signal, with Rayleigh retained as its nested zero-offset
  limit;
- `rec_m2_epX`: narrow Gaussian core plus optional broad Gaussian nuisance tail
  and a nonnegative linear background;
- `rec_m_eggX`: Gaussian signal, promoted to a split Gaussian only when the
  Bayesian information criterion supports the extra asymmetry parameter;
- `rec_E_miss`: Gaussian core plus an optional positive ex-Gaussian tail; the
  tail is fitted and audited but does not widen the core-resolution cut; and
- `rec_m2_miss`: Laplace cusp, promoted to an asymmetric Laplace when supported,
  with an optional broad Laplace nuisance component that does not widen the
  narrow-cusp cut.

The `--n-sigma` value supplies the fallback Gaussian-equivalent signal
probability (for
example, `3` means 99.73%). Non-Gaussian windows are obtained from their fitted
signal CDF at that containment. The analysis config can override the
containment and choose `core` or `signal` independently for each variable under
`exclusivity.variables`. The nominal RGA and RGK configs explicitly use the
core for `m_gg`, `m2_epX`, `E_miss`, and `m2_miss`, and the complete physical
signal for `pT_miss` and `m_eggX`. Broad nuisance components do
not enlarge their associated narrow-core windows. This prevents percentile
tails, background-contaminated sigma clipping, or a broad resolution tail from
defining the core resolution. Local windows are derived separately for
`(proton detector, number of FT photons, Q2 bin, xB bin, -t bin)`. Global mode
retains the first two topology components while pooling kinematic bins. Thus
FD/FD, mixed FT/FD, and FT/FT photon pairs never share resolution windows, and
the two possible orderings of a mixed pair remain one physical category.

Nominal windows come from one fixed physics-ordered bootstrap (`m_gg`,
`pT_miss`, `m2_epX`, `m_eggX`, `E_miss`, then `m2_miss`) so that early clean
peaks suppress combinatorial background before the missing-quantity fits. The
order is based on variable identity rather than the caller's tuple order. Once
derived, these six windows are immutable. A single simultaneous N-1 audit then
refits each quantity after applying the other five nominal cuts and records the
alternative boundaries, fit population, source, any failure reason, and maximum
relative boundary displacement. Audit fits never update a cut or remove a
group. This avoids the two-cycle and multi-cycle oscillations that can occur
when correlated tails repeatedly change one another's fitting populations. The
configured audit tolerance is therefore a stability flag, not a convergence or
acceptance requirement. The cut table also records cumulative cut flow and the
fixed-window N-1 numerator and denominator for every group and variable.

When a local kinematic group has too few bootstrap-selected events or an
unstable core, its nominal window falls back to the corresponding bootstrap
sample pooled over the same proton detector and FT-photon multiplicity. The
same hierarchy is recorded separately for an N-1 audit proposal. A group is
retained only if all six nominal variables have finite windows; cuts are never
silently disabled. The cut NPZ
records every initially populated group and, for any removed group, the first
failed variable and exact fit/window rejection reason. It also records fitted
centers, characteristic scales, fit and fitted-signal entry counts, signal
fractions, peak significances, selected fit models, named fit parameters,
signal containment, fit domain, Pearson chi-square, Poisson deviance, BIC and
next-model delta-BIC, binned observations and fitted components, iteration
counts, and whether each window was global, local, or topology-pooled. The
command also prints the median center, scale, bounds,
signal fraction, significance, and model counts for every variable. Absolute
expected-center and maximum-width sanity limits reject
pathological fits rather than allowing an inflated sigma to validate itself.
In addition, a local window must remain within the configured width ratio and
center shift of its same-proton-detector/same-FT topology reference. An
otherwise valid but inconsistent local hump is recorded as a
`topology_consistency_fallback`, making this hierarchical decision auditable.
The nominal procedure is to
derive separate equal-containment windows for data and GEMC. GEMC exclusivity peaks are
often narrower than data, so equal numerical boundaries would not represent
equal resolution-relative signal regions. Save both cut tables and their
retained fractions. Applying one common numerical table to both samples remains
useful as a systematic cross-check, especially for non-Gaussian tails and
correlated sequential cuts.

Derive nominal data and GEMC windows independently:

```bash
python3 analysis/derive_exclusivity.py data_events.npz \
  --config configs/analysis/rgk/6.535.json \
  --cuts results/data_exclusivity.npz --mask results/data_exclusivity.npy \
  --diagnostics results/data_exclusivity_diagnostics.pdf

python3 analysis/derive_exclusivity.py mc_events.npz \
  --config configs/analysis/rgk/6.535.json \
  --cuts results/gemc_exclusivity.npz --mask results/gemc_exclusivity.npy \
  --diagnostics results/gemc_exclusivity_diagnostics.pdf
```

The PDF can also be regenerated without the event sample because the v8 cut
table stores the fit histograms and components:

```bash
python3 analysis/plot_exclusivity_diagnostics.py \
  results/data_exclusivity.npz \
  results/data_exclusivity_diagnostics.pdf
```

Compare the stored data and GEMC fits in one PDF with one page per exclusivity
quantity, topology columns, and data/GEMC rows:

```bash
python3 analysis/plot_exclusivity_diagnostics.py \
  results/data_exclusivity.npz \
  results/data_gemc_exclusivity_comparison.pdf \
  --gemc-cuts results/gemc_exclusivity.npz
```

Each comparison panel places fit metadata and quality metrics in separate,
opaque side panels rather than over the histogram. Each topology column has its
own horizontal range, shared only by its data and GEMC rows, so a broad topology
cannot visually compress a narrower one. Data and GEMC use distinct color
palettes, and the nominal window is listed as `[lower, upper]`. Bold topology
headers and semi-transparent vertical separators organize the columns. The
compact legend shows the data/GEMC sample encoding once and the line-style
meaning of each fit component once, rather than duplicating every component for
both samples.
Quantity names use physics-formatted labels. Paired mode requires the nominal
global-by-topology cut tables because a local table can contain many kinematic
groups for one detector topology. The main paired pages contain only topologies
retained by both data and GEMC. By default, each topology that failed retention
in either sample is then rendered on dedicated per-quantity audit pages at the
end of the PDF. Add `--omit-dropped-topologies` to suppress that appendix.

For a large per-bin table, the plotter defaults to the 24 groups with the worst
reduced chi-square. Repeat `--group-id ID` to inspect chosen groups, or change
the limit with `--maximum-groups`. A high reduced chi-square is an audit flag,
not an automatic rejection: a statistically significant but imperfect model
should be inspected rather than silently discarded.

Pass the GEMC mask to `response --selection-mask`. For data, either pass the
mask to `unfold --selection-mask` without background subtraction, or pass the
corresponding data cut table to `unfold --background-cuts` so the command can
reconstruct both its signal and N-1 sideband regions. Use `--per-bin-cuts` only
when one deliberately wants
local kinematic windows with automatic same-topology fallback instead of the
default pooled topology windows. Cut tables produced
before photon-topology grouping or before the v8 bootstrap/audited fits are
incompatible and must be re-derived rather than passed with `--reuse-cuts`.

For large GEMC production, derive the GEMC exclusivity mask directly from the
selected-candidate ROOT file instead of reading the dense generated-event NPZ:

```bash
python3 analysis/derive_exclusivity.py selected_mc.root \
  --format selected-root --dictionary build/libROOTBranchesDict.so \
  --config configs/analysis/rgk/6.535.json \
  --cuts results/gemc_exclusivity.npz --mask results/gemc_selected_exclusivity.npy
```

The selected-root mask has one row per selected ROOT candidate, so use it with
`response-root --selection-mask`. The dense NPZ mask has one row per generated
event and remains the format expected by `response --selection-mask`.

## Compact `gEvents` schema

The converter writes one row for every input MC event:

- `sourceFileId`, a deterministic hash of the input HIPO source name;
- `sourceEventIndex`, the zero-based input-event ordinal within that file;
- the original `runNum`, `eventNum` for diagnostics;
- `topologyValid`, `radiative`;
- `stratumFlatIndex` (`-1` for samples without stratum provenance);
- `weight` (unit weight unless generator chunk provenance is enabled);
- `Q2`, `nu`, `xB`, `y`, `W`, `minusT`, `trentoPhi`.

`build_event_sample.py` retains only `topologyValid` rows in the physics sample,
while the converter `Summary` tree records both total generated rows and valid
generated topologies for bookkeeping. The companion `SourceFiles` tree records
the mapping from `sourceFileId` to HIPO source name, normally the basename and
the full path only when duplicate basenames must be disambiguated.

Check both identities after converting more than one input file:

```bash
python3 analysis/check_event_keys.py 6.535_rgk_eppi0_GEMC.root
```

Repeated `(runNum,eventNum)` keys are expected for the tested GEMC production;
duplicated source-event keys are an error.

### Bin-conditional OSG weights

After finalizing a bin-conditional AAO campaign, repack its LUND files without
mixing strata or generator replicas:

```bash
python3 scripts/repack_lund_for_osg.py \
  born_rgk_conditional born_rgk_osg \
  --glob '**/*.lund' \
  --campaign-weights born_rgk_conditional/campaign_weights.json \
  --prefix aao_born
```

Chunk names follow
`aao_born__sNNNNN__gNNNN__pNNNNNN.lund`. Each chunk contains events from
exactly one stratum and one generator invocation. `chunk_provenance.json` and
`.tsv` record the source LUND file, pooled stratum weight, event count, and
identifiers for every chunk.

Submit these files through the portal's type-2 LUND workflow. Its output name
may wrap the LUND filename as
`STRINGID-LUNDFILENAME-OSGID-JOBINDEX.hipo`; the canonical
`sNNNNN__gNNNN__pNNNNNN` token remains embedded in that name. Then enable the
lookup during conversion:

```json
{
  "generatedEventTree": {
    "enabled": true
  },
  "generatorWeights": {
    "enabled": true,
    "chunkProvenance": "/path/to/born_rgk_osg/chunk_provenance.json"
  },
  "fillMC": true
}
```

`hipo2root` extracts exactly one canonical chunk token from each HIPO basename,
matches it to the provenance table, and fills
`gEvents.stratumFlatIndex` and `gEvents.weight`. It fails
rather than silently assigning unit weight when the token is missing,
ambiguous, or absent from an enabled provenance table. This supports both
directly named local HIPO files and portal type-2 names without changing the
LUND event structure. Type-1 generator submissions are not yet supported. The
`SourceFiles` tree also stores the resolved stratum and weight for file-level
auditing. The existing response builders consume `weight` directly.

The harmonic stage retains the legacy weighted fit
`A + B cos(phi) + C cos(2 phi)` and stores coefficients, full covariance,
chi-square per degree of freedom, and the number of contributing phi bins.

The radiative-correction command streams Born and radiative LUND files directly
into configured analysis bins, using the same electron-proton Trento phi
convention as the rest of this package. If the analysis config contains a
`phase_space` block, each filled bin is interpreted as the rectangular 4D bin
intersected with those generated-level DIS cuts. Both current configs require
`Q2 >= 1` and `W >= 2`. RGA directly requires generated electron momentum
`p_e >= 2 GeV`, while RGK directly requires `p_e >= 1 GeV`. These match the
nominal post-selection thresholds without a redundant `y` cut. Conversion is
intentionally padded to `1.5 GeV` for RGA and `0.3 GeV` for RGK so alternate
post-selection thresholds can be studied. Its output is a native `C_rad.npz`
artifact consumed by `unfold --radiative-correction`; reliability masks and
correction uncertainties are propagated into the self-contained unfolding
result. For AAO-generated samples, pass the generator `sig_sum` integrated cross
sections with `--born-normalization-file` and `--radiative-normalization-file`
when `.norm` or `.sum` sidecars are available. Each option may point to one
sidecar or to a directory containing many job sidecars. Directory inputs prefer
`.norm` files when present, use an `events`-weighted mean `sig_sum` when every
sidecar includes event counts, and fall back to an unweighted mean for legacy
`.sum`-only directories. Use
`--born-integrated-cross-section` and `--radiative-integrated-cross-section`
when entering the values manually. The resulting global factor is
`(Sigma_rad / Sigma_born) * (N_born / N_rad)`, so the stored `C_rad` is the
radiative-to-Born cross-section ratio rather than a raw event-density ratio.
`unfold --radiative-correction` divides unfolded yields by this factor. The
artifact also stores support diagnostics: per-bin born/radiative counts, overlap
and status masks, generated `Q2`/`Eprime` ranges, the phase-space cuts used to
define the selected bins, and the integrated cross sections used for each
sample. When sidecars are supplied, the artifact also preserves the
normalization records used to get those cross sections: sidecar paths,
combination method, `sig_sum`, `sig_int`, `events`, `ntries`, `nevent`,
`mcall_max`, `sigr_max`, generator name, and units. Regenerate the diagnostic
report later without rereading LUND files:

```bash
python3 analysis/run_analysis.py radiative-correction-plots results/C_rad.npz \
  --output results/C_rad_diagnostics.pdf --csv results/C_rad_diagnostics.csv \
  --quilt
```

The PDF includes summary/support pages, a clipped `0<C_rad<2` summary
histogram, projection heatmaps of median reliable `C_rad` and reliable-bin
fraction in `(xB,Q2)` and `(phi,-t)`, and then detailed per-`(Q2,xB,-t)` phi
pages. With `--quilt`, one stitched `Q2`-by-`xB` `C_rad`-vs-phi page per `-t`
bin is inserted before the detailed pages. Use `--quilt-scale-mode global` for
a common y scale across every panel on a page.

## Bin-centering correction

`bin-centering` computes `C_BC = <d4sigma>_physical_selected_bin /
d4sigma(center)` with AAO model calls over a midpoint grid. The selected bin is
the exclusive physical region intersected with the same optional `phase_space`
cuts from the analysis config. The analysis-facing convention remains positive
`-t`; the command converts internally to signed negative `t` only when calling
`aao_xsec`. The reference center is the geometric centroid of the sampled
physical selected midpoint cells in each bin, and the artifact stores the center
coordinates, physical fractions, failed-call fractions, phase-space cuts, and
reliability mask.

Apply the artifact during normalization with `cross-section --bin-centering`.
The cross-section stage uses the exclusive physical bin volume, including the
partial `-t` overlap at the reaction boundary, and evaluates the virtual-photon
flux at the reference `Q2` and `xB` stored in the `C_BC` artifact. The reduced
cross section and uncertainty are then divided by `C_BC`; unreliable
bin-centering bins are suppressed in the output. The uncentered event-mean
coordinates are retained separately in the cross-section artifact.

For convergence studies or production-sized grids, split the work over
flattened 3D `(Q2, xB, -t)` bins and merge the partial artifacts. A local chunk
looks like:

```bash
python3 analysis/run_analysis.py bin-centering \
  --config configs/analysis/rgk/6.535.json \
  --output results/bc_parts/C_BC_N4_part000.npz \
  --exe /path/to/aao_xsec --N 4 --workers 8 \
  --progress-chunks 10 --bin-chunks 100 --bin-chunk-index 0
```

For Slurm, use a zero-based array such as `--array=0-99` and pass
`${SLURM_ARRAY_TASK_ID}` as `--bin-chunk-index`. After all chunks finish:

```bash
python3 analysis/run_analysis.py bin-centering-merge \
  results/bc_parts/C_BC_N4_part*.npz --output results/C_BC_N4.npz
```

Repeat with larger `--N` values, for example `N=2,4,6,8`, and compare the
merged `C_BC` artifacts to assess convergence. To summarize and visualize a
scan after merging all requested `C_BC_N*.npz` files:

For the JLab production scan at `N=10,20,30`, use the repository submission
driver rather than embedding a multi-statement shell program in every SWIF job:

```bash
cd /work/clas12/storyf/SF_analysis_software_v2.0
bash scripts/submit_bin_centering_swif.sh --run
```

The driver freezes the analysis code, configuration, AAO executable, MAID
table, module setup, driver, worker, job manifest, Git state, and hashes under
the campaign correction directories. It creates two phase-0 jobs (one RGK and
one RGA real output) and releases the remaining jobs only after both phase-0
jobs succeed. The RGA result is common to the torus+1 and torus-1 campaigns.
Use `--help` to override input paths, the UTC stage stamp, or the number of
chunks.

```bash
python3 scripts/plot_bin_centering_convergence.py results/bin_centering_convergence/rgk_6.535 \
  --n-values 2 4 6 8 \
  --reference-N 8
```

The script writes a Markdown summary, pairwise statistics CSV, worst-bin CSV,
and a PNG showing reliable-bin growth, relative-difference histograms/CDFs,
adjacent-`N` convergence, and where the largest tail sits in phi and kinematic
bin index.

To plot `C_BC` versus phi with one stitched `Q2`-by-`xB` quilt per `-t` bin:

```bash
python3 analysis/run_analysis.py bin-centering-plots results/C_BC_N30.npz \
  --output results/C_BC_N30_quilts.pdf --quilt
```

Optionally overlay every available merged N artifact in a convergence-scan
directory:

```bash
python3 analysis/run_analysis.py bin-centering-plots results/C_BC_N30.npz \
  --output results/C_BC_Nscan_quilts.pdf --quilt \
  --overlay-n-directory results/bin_centering_convergence/rgk_6.535 \
  --quilt-scale-mode panel --quilts-only
```

Files are discovered with `C_BC*.npz`, labeled using their stored
`samples_per_dimension`, sorted by N, and symlink aliases are deduplicated.
Only reliable, computed phi bins are drawn. The PDF also contains detailed phi
pages for the primary positional artifact unless `--quilts-only` is passed.

## Legacy behavior intentionally corrected

- reconstructed failures never remove generated events from the denominator;
- bin means are computed after the final event mask;
- beam energy is passed consistently into both `y` and virtual-photon epsilon;
- bootstrap results can be reproduced with an explicit random seed;
- output paths and binning are configuration, not hard-coded working-directory
  side effects;
- production response building avoids dense generated-event intermediates and
  writes the response artifacts directly.

## Managing MC intermediate size

The preferred converter configuration enables `gEvents`, applies REC
topology/DIS skims only to `rParticles`, and sets
`saveUnmatchedMC` to false. This retains the generated denominator in one
lightweight scalar row per event. See
`configs/processing/rgk/6.535/eppi0_GEMC.json`.

For older files without `gEvents`, an unbiased denominator still
requires an unrestricted matched conversion with unmatched particle-level GEN
rows. Treat that legacy intermediate as a temporary scratch product:

- write it under JLab `/volatile` or another scratch filesystem, not inside the
  Git working tree;
- process manageable production chunks rather than combining every HIPO file
  into one monolithic ROOT file;
- immediately run `response-root` for each chunk or preserve only compact debug
  NPZ chunks when event-level audits are needed;
- retain response artifacts, compact debug NPZ chunks, and provenance metadata;
- verify generated-event counts and checksums before removing scratch ROOT
  files;
- concatenate compact event samples only after conversion.

Compact MC chunks can be combined with:

```bash
python3 analysis/concatenate_event_samples.py results/chunks/*.npz \
  --output results/mc_events.npz
```

Duplicate `(source_file_id,source_event_index)` keys are rejected by default,
protecting against accidentally processing the same HIPO file twice. Legacy
samples without source identity fall back to `(run,event)` and therefore cannot
safely combine files whose GEMC event numbers restart. Reconvert those files
with the current converter before multi-file acceptance production.

Setting `saveUnmatchedMC` to false is safe for acceptance only when
`gEvents` is enabled. Without that tree, generated events lacking a
reconstructed match disappear from the denominator.
