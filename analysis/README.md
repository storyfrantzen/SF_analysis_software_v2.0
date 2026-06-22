# EPPI0 analysis

This directory is the maintained replacement for the legacy
`SF_analysis_software/analysis_v2.0` scripts.  The numerical core is separated
from ROOT I/O so response construction, unfolding, and normalization can be
unit-tested with ordinary NumPy arrays.

## Event-sample contract

An acceptance sample must contain one row per generated event, including events
with no accepted reconstructed candidate.  Each row needs:

- generated `Q2`, `xB`, `-t`, and Trento phi;
- reconstructed values for the same coordinates, or `NaN` when no candidate is
  reconstructed;
- a reconstructed-selection flag;
- the reconstructed proton topology;
- the generated-event weight, defaulting to one.

Do not apply reconstructed DIS or final-state cuts before creating this sample.
Those cuts define the reconstructed numerator.  Generated phase-space cuts
define the denominator independently.

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

Run the dependency-light tests from the repository root:

```bash
python3 -m unittest discover -s analysis/tests -v
```

The three numerical stages are exposed through one command:

```bash
python3 analysis/run_analysis.py response mc_events.npz \
  --config analysis/configs/rgk_6.535.json --output-dir results/response

python3 analysis/run_analysis.py unfold data_events.npz \
  results/response/response_matrix.npz results/response/response_meta.npz \
  --config analysis/configs/rgk_6.535.json --output results/unfolding.npz

python3 analysis/run_analysis.py cross-section results/unfolding.npz \
  --config analysis/configs/rgk_6.535.json --output results/cross_section.npz

python3 analysis/run_analysis.py fit-harmonics results/cross_section.npz \
  --output results/harmonics.npz
```

`build_event_sample.py` creates the MC event sample by left-joining the selected
REC tree onto every GEN event in an unrestricted matched tree.
`export_selected_data.py` creates the compact data artifact and carries the
converter's accumulated charge into the pipeline.

`derive_exclusivity.py` preserves the legacy sequential variable order and
global/per-bin modes. The nominal legacy-faithful procedure is to derive
separate `n`-sigma windows for data and GEMC. GEMC exclusivity peaks are often
narrower than data, so equal numerical boundaries would not represent equal
resolution-relative signal regions. Save both cut tables and their retained
fractions. Applying one common numerical table to both samples remains useful
as a systematic cross-check, especially for non-Gaussian tails and correlated
sequential cuts.

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
- expensive ROOT reads are intended to happen once when producing the event
  sample, rather than once per analysis stage.

## Managing the unrestricted MC intermediate

The current converter represents MC truth as particle-level rows. An unbiased
acceptance denominator therefore requires an unrestricted matched conversion
with unmatched GEN rows retained. That ROOT file can be much larger than the
compact event-level NPZ ultimately used by this analysis.

Treat the unrestricted matched ROOT file as a temporary scratch product:

- write it under JLab `/volatile` or another scratch filesystem, not inside the
  Git working tree;
- process manageable production chunks rather than combining every HIPO file
  into one monolithic ROOT file;
- immediately run `build_event_sample.py` for each chunk;
- retain the compact NPZ chunks and provenance metadata;
- verify generated-event counts and checksums before removing scratch ROOT
  files;
- concatenate compact event samples only after conversion.

Compact MC chunks can be combined with:

```bash
python3 analysis/concatenate_event_samples.py results/chunks/*.npz \
  --output results/mc_events.npz
```

Duplicate `(run,event)` keys are rejected by default, protecting against
accidentally processing the same HIPO chunk twice.

Setting `saveUnmatchedMC` to false is **not** an unbiased workaround with the
current schema: generated events without a reconstructed match would disappear
from the denominator.

The preferred converter enhancement is a separate compact `GeneratedEvents`
tree with one row per generated event containing only `(run, event, Q2, xB,
-t, phi, radiative, weight)`. Once that exists, unmatched GEN particle rows can
be disabled for production acceptance work. The reconstructed particle/candidate
output can then be skimmed independently without losing the generated
denominator.
