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
global/per-bin modes. Production use should derive one nominal cut table and
reuse it for both data and GEMC via `--reuse-cuts --apply-to`; deriving separate
windows remains useful only as a detector-model diagnostic.

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
