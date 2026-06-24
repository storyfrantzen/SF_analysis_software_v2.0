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
  and reduced cross sections.
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
`analysis/configs/rga/10.604.json` together with the matching processing and
post-processing files under `configs/*/rga/10.604/`.

The four numerical stages are exposed through one command:

```bash
python3 analysis/run_analysis.py response mc_events.npz \
  --config analysis/configs/rgk/6.535.json --output-dir results/response

python3 analysis/run_analysis.py unfold data_events.npz \
  results/response/response_matrix.npz results/response/response_meta.npz \
  --config analysis/configs/rgk/6.535.json --output results/unfolding.npz

python3 analysis/run_analysis.py cross-section results/unfolding.npz \
  --config analysis/configs/rgk/6.535.json --output results/cross_section.npz

python3 analysis/run_analysis.py fit-harmonics results/cross_section.npz \
  --output results/harmonics.npz
```

For production MC response building, avoid serializing one dense row per
generated event. Build the sparse response directly from the converter and
selected ROOT files:

```bash
python3 analysis/run_analysis.py response-root \
  6.535_rgk_eppi0_mc_acceptance.root selected_mc.root \
  --config analysis/configs/rgk/6.535.json --output-dir results/response \
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
  --config analysis/configs/rgk/6.535.json \
  --cuts results/data_exclusivity.npz --mask results/data_exclusivity.npy

python3 analysis/derive_exclusivity.py mc_events.npz \
  --config analysis/configs/rgk/6.535.json \
  --cuts results/gemc_exclusivity.npz --mask results/gemc_exclusivity.npy
```

Pass the GEMC mask to `response --selection-mask` and the data mask to
`unfold --selection-mask`. Use `--global-cuts` for one set of windows per proton
topology when individual kinematic bins lack sufficient statistics.

## Compact `GeneratedEvents` schema

The converter writes one row for every input MC event:

- `sourceFileId`, a deterministic hash of the input HIPO basename;
- `sourceEventIndex`, the zero-based input-event ordinal within that file;
- the original `runNum`, `eventNum` for diagnostics;
- `topologyValid`, `radiative`;
- `weight` (currently `1.0` until generator weights are connected);
- `Q2`, `nu`, `xB`, `y`, `W`, `minusT`, `trentoPhi`.

`build_event_sample.py` retains only `topologyValid` rows in the physics sample,
while the converter `Summary` tree records both total generated rows and valid
generated topologies for bookkeeping. The companion `SourceFiles` tree records
the mapping from `sourceFileId` to HIPO basename.

Check both identities after converting more than one input file:

```bash
python3 analysis/check_event_keys.py 6.535_rgk_eppi0_mc_acceptance.root
```

Repeated `(runNum,eventNum)` keys are expected for the tested GEMC production;
duplicated source-event keys are an error.

The harmonic stage retains the legacy weighted fit
`A + B cos(phi) + C cos(2 phi)` and stores coefficients, full covariance,
chi-square per degree of freedom, and the number of contributing phi bins.

The unfolding command accepts the legacy radiative artifact through
`--radiative-correction C_rad.npz`; reliability masks and correction
uncertainties are propagated into the self-contained unfolding result.

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
