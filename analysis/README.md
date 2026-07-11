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
- the reconstructed proton topology;
- the generated-event weight, defaulting to one.

Reconstructed DIS and final-state cuts may be applied to the particle-level
`Events` tree only after the converter has filled `GeneratedEvents`. They define
the reconstructed numerator and do not alter the generated denominator.
Generated phase-space cuts are applied later to the compact truth coordinates.
For legacy files without `GeneratedEvents`, do not apply reconstructed event
filters during conversion because the unmatched particle rows are then the only
source of the generated denominator.

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
- `eppi0.exclusivity`: topology-aware sequential data/GEMC windows;
- `eppi0.event_sample`: radiative/non-radiative GEN construction and REC joins;
- `eppi0.harmonics`: weighted `A + B cos(phi) + C cos(2 phi)` fits.

## Dependencies

- Python 3.10 or newer;
- NumPy and SciPy for the numerical pipeline;
- PyROOT for `build_event_sample.py` and `export_selected_data.py`;
- the project ROOT dictionary when object branches are not already discoverable.

Run the dependency-light tests from the repository root:

```bash
python3 -m unittest discover -s analysis/tests -v
```

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

For production MC response building, avoid serializing one dense row per
generated event. Build the sparse response directly from the converter and
selected ROOT files:

```bash
python3 analysis/run_analysis.py response-root \
  6.535_rgk_eppi0_mc_acceptance.root selected_mc.root \
  --config configs/analysis/rgk/6.535.json --output-dir results/response \
  --dictionary build/libROOTBranchesDict.dylib
```

This command histograms the `GeneratedEvents` denominator in chunks, joins only
selected REC candidates by `(sourceFileId, sourceEventIndex)`, and writes the
same `response_matrix.npz` and `response_meta.npz` consumed by `unfold`.
`build_event_sample.py` remains useful for compact debug samples and for
backward compatibility with older particle-level matched files. New files join
on `(sourceFileId, sourceEventIndex)`, because GEMC files can all have run 11
and restart their event numbers.
`export_selected_data.py` creates the compact data artifact and carries the
converter's accumulated charge into the pipeline.

For interactive cut studies and quick detector/topology comparisons, build a
standalone histogram browser from either compact NPZ samples or selected ROOT
trees:

```bash
python3 analysis/interactive_histograms.py data_events.npz \
  --output results/data_histograms.html

python3 analysis/interactive_histograms.py selected_data.root \
  --format root --output results/selected_data_histograms.html \
  --dictionary build/libROOTBranchesDict.so

scripts/serve_visualizer.sh results/selected_data_histograms.html
```

Run `scripts/serve_visualizer.sh` with no path to serve a click-enabled listing
of `results/`.
The browser discovers stored scalar quantities, adds common derived views such
as degree versions of angular branches and signed/unsigned `t` aliases, and
supports 1D histograms, 2D histograms, detector/pass-flag toggles, text filters,
arbitrary numeric range filters, and optional panel tabs that can be viewed
independently or side by side under the same filters from the toolbar above the
plot. Axis tick-count
sliders adjust every displayed axis, optional presentation labels can override
the displayed X/Y axis labels per panel, and the visible display can be saved as
a PNG. 2D views can optionally draw color scales beside the histogram, including
per-sector scales in split views and hover markers showing hovered-bin values
on primary and overlay scales. Category
filters also have a `Filter topology` selector near the split
control, with the full category list kept as the final compact reference section
below the plot, after range filters, derived operations, fit controls, reset
buttons, PNG export, and the preview table. Common display/actions such as log
color, reset, and PNG export sit above the plot. Compact `+` buttons
next to the axis selectors add optional comparison X/Y quantities on the same
axes with a distinct color map for direct within-panel comparisons. The
derived-operations menu can add browser-side comparison variables such as
`rec_theta_deg - gen_theta_deg`, ratios, fractional residuals, or sums, and the
fit menu can overlay Gaussian, linear, or quadratic fits where applicable,
including independent sector-by-sector fits in split views with compact
in-panel fit labels. Text
filters appear only when the input contains string-valued columns. Selected ROOT
inputs
expose the richest set of reconstruction filters, including `pDet`, `passFiducial`,
`passSamplingFraction`, `passExclusivity`, and selected-particle kinematics
such as `protonTheta`, so plots like `theta_p` vs `-t` can be explored directly.
Newly post-processed selected files also expose selected-particle indices and
sectors, including `pIdx`, `g1Idx`, `g2Idx`, `pSector`, `g1Sector`, and
`g2Sector`, so proton and photon selections can be filtered like the electron.
The `passSamplingFraction` flag is labeled as a cut result; selected ROOT inputs
also derive total and PCAL/ECIN/ECOUT electron sampling fractions from the stored
calorimeter energies divided by `electronP`.
Joined generated/reconstructed event samples made by `build_event_sample.py`
carry every scalar branch from the selected reconstructed tree with a `rec_`
prefix, so MC acceptance visualizers expose the same reconstructed filters and
kinematic branches as the selected data visualizers after the `.npz` and HTML
are regenerated. Newly converted MC files store complete per-particle GEN/LUND
kinematics in `GeneratedEvents`; `build_event_sample.py` carries those columns,
including `gen_electronP`, `gen_protonTheta`, `gen_gamma1Phi`, `gen_gamma2P`,
and `gen_pi0P`, so generated-vs-reconstructed residuals can be built directly
in the visualizer. Older converter ROOT files fall back to the available
`Events.gen` rows, which may be less complete.
The dictionary is optional for ordinary selected `Events` trees; if the named
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
the other three diagnostics.

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
  6.535_rgk_eppi0_mc_acceptance.root selected_mc.root mc_events.npz
```

`derive_exclusivity.py` preserves the legacy sequential variable order and
global/per-bin modes. The nominal legacy-faithful procedure is to derive
separate `n`-sigma windows for data and GEMC. GEMC exclusivity peaks are often
narrower than data, so equal numerical boundaries would not represent equal
resolution-relative signal regions. Save both cut tables and their retained
fractions. Applying one common numerical table to both samples remains useful
as a systematic cross-check, especially for non-Gaussian tails and correlated
sequential cuts.

Derive nominal data and GEMC windows independently:

```bash
python3 analysis/derive_exclusivity.py data_events.npz \
  --config configs/analysis/rgk/6.535.json \
  --cuts results/data_exclusivity.npz --mask results/data_exclusivity.npy

python3 analysis/derive_exclusivity.py mc_events.npz \
  --config configs/analysis/rgk/6.535.json \
  --cuts results/gemc_exclusivity.npz --mask results/gemc_exclusivity.npy
```

Pass the GEMC mask to `response --selection-mask` and the data mask to
`unfold --selection-mask`. Use `--global-cuts` for one set of windows per proton
topology when individual kinematic bins lack sufficient statistics.

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

## Compact `GeneratedEvents` schema

The converter writes one row for every input MC event:

- `sourceFileId`, a deterministic hash of the input HIPO source name;
- `sourceEventIndex`, the zero-based input-event ordinal within that file;
- the original `runNum`, `eventNum` for diagnostics;
- `topologyValid`, `radiative`;
- `weight` (currently `1.0` until generator weights are connected);
- `Q2`, `nu`, `xB`, `y`, `W`, `minusT`, `trentoPhi`.

`build_event_sample.py` retains only `topologyValid` rows in the physics sample,
while the converter `Summary` tree records both total generated rows and valid
generated topologies for bookkeeping. The companion `SourceFiles` tree records
the mapping from `sourceFileId` to HIPO source name, normally the basename and
the full path only when duplicate basenames must be disambiguated.

Check both identities after converting more than one input file:

```bash
python3 analysis/check_event_keys.py 6.535_rgk_eppi0_mc_acceptance.root
```

Repeated `(runNum,eventNum)` keys are expected for the tested GEMC production;
duplicated source-event keys are an error.

The harmonic stage retains the legacy weighted fit
`A + B cos(phi) + C cos(2 phi)` and stores coefficients, full covariance,
chi-square per degree of freedom, and the number of contributing phi bins.

The radiative-correction command streams Born and radiative LUND files directly
into configured analysis bins, using the same electron-proton Trento phi
convention as the rest of this package. Its output is a native `C_rad.npz`
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
and status masks, generated `Q2`/`Eprime` ranges, and the integrated cross
sections used for each sample. When sidecars are supplied, the artifact also
preserves the
normalization records used to get those cross sections: sidecar paths,
combination method, `sig_sum`, `sig_int`, `events`, `ntries`, `nevent`,
`mcall_max`, `sigr_max`, generator name, and units. Regenerate the diagnostic
report later without rereading LUND files:

```bash
python3 analysis/run_analysis.py radiative-correction-plots results/C_rad.npz \
  --output results/C_rad_diagnostics.pdf --csv results/C_rad_diagnostics.csv
```

The PDF includes summary/support pages, a clipped `0<C_rad<2` summary
histogram, projection heatmaps of median reliable `C_rad` and reliable-bin
fraction in `(xB,Q2)` and `(phi,-t)`, and then detailed per-`(Q2,xB,-t)` phi
pages.

## Bin-centering correction

`bin-centering` computes `C_BC = <d4sigma>_physical_bin / d4sigma(center)` with
AAO model calls over a midpoint grid. The analysis-facing convention remains
positive `-t`; the command converts internally to signed negative `t` only when
calling `aao_xsec`. The reference center is the geometric centroid of the
sampled physical midpoint cells in each bin, and the artifact stores the center
coordinates, physical fractions, failed-call fractions, and reliability mask.

Apply the artifact during normalization with `cross-section --bin-centering`.
The reduced cross section and uncertainty are divided by `C_BC`; unreliable
bin-centering bins are suppressed in the output.

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

```bash
python3 scripts/plot_bin_centering_convergence.py results/bin_centering_convergence/rgk_6.535 \
  --n-values 2 4 6 8 \
  --reference-N 8
```

The script writes a Markdown summary, pairwise statistics CSV, worst-bin CSV,
and a PNG showing reliable-bin growth, relative-difference histograms/CDFs,
adjacent-`N` convergence, and where the largest tail sits in phi and kinematic
bin index.

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

The preferred converter configuration enables `GeneratedEvents`, applies REC
topology/DIS skims only to `Events`, and sets `saveUnmatchedMC` to false. This
retains the generated denominator in one lightweight scalar row per event. See
`configs/processing/rgk/6.535/eppi0_mc_acceptance.json`.

For older files without `GeneratedEvents`, an unbiased denominator still
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
`GeneratedEvents` is enabled. Without that tree, generated events lacking a
reconstructed match disappear from the denominator.
