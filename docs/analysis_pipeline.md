# Analysis Pipeline Design

The converter should make a reusable analysis ntuple. Physics selections that are tuned, compared, or varied for systematics should usually run after the ROOT file is made.

## Recommended Stages

1. HIPO to ROOT conversion

   Keep this stage focused on expensive IO, stable bookkeeping, and variables that are hard or impossible to reconstruct later from the ROOT ntuple.

   Good fits here:

   - run/event/helicity/beam charge
   - reconstructed particle four-vectors and detector coordinates
   - MC truth branches
   - loose final-state requirements used only to reduce file size
   - loose DIS skims such as broad `Q2`, `W`, and `y` ranges
   - an optional loose existential diphoton-mass skim used only to reduce file size
   - optional precomputed helper variables or cut flags, as long as the raw ingredients are still saved
   - configured proton kinematic corrections, with raw values and correction deltas saved alongside corrected branches
   - optional REC/GEN matching for calibration and acceptance studies

2. ROOT post-processing

   Apply analysis-specific and tuneable selections here. This lets you change thresholds without rereading hipo files.

   Good fits here:

   - fiducial cuts
   - exclusivity cuts
   - missing-mass, missing-energy, and angle selections
   - topology-specific particle choices when multiple candidates exist
   - systematic variations of all selection boundaries
   - final histogramming and plotting

## Data Quality

For reconstructed data, `hipo2root` can reject bad QADB bins before topology and DIS
selection. Enable it in a processing config with:

```json
"qadb": {
  "enabled": true,
  "database": "latest",
  "rejectDefects": [
    "MarginalOutlier", "TerminalOutlier", "TotalOutlier",
    "SectorLoss", "LowLiveTime"
  ],
  "allowMiscRuns": []
}
```

The defect list is an analysis choice. Add `"Misc"` only after reviewing its comments
with `qadb-info misc`; individual acceptable `Misc` runs can then be listed in
`allowMiscRuns`. QADB is bypassed for simulation (`runNum == 11`). The terminal summary
reports rejected events and accumulated DAQ-gated charge. The output ROOT file stores the
processing counters in its `Summary` tree and the file-level charge separately as a
top-level `TParameter<double>` named `AccumulatedCharge`. It also writes a
`RunCharge` tree with one row per observed run and the branches `runNum`,
`accumulatedCharge_nC`, `totalEvents`, `passedQADBEvents`, and
`failedQADBEvents`. Charge is attributed when QADB first accumulates a new good
interval, so mixed-run converter files can be normalized after filtering the
selected candidates by `runNum`. The per-run charge sum reproduces the legacy
file-level total; the latter remains available for backward compatibility.
When a processing config declares `torus`, the converter also writes that
value as the top-level `TParameter<int>` named `TorusPolarity`; zero means the
polarity was not specified by an older or polarity-independent config.

Load QADB before configuring and building on JLab, for example `module load qadb/3.4.0`.
If QADB is requested by a config but was unavailable at build time, `hipo2root` exits with
an explicit error.

### Reproducible run-condition audit

Use `analysis/audit_run_conditions.py` to compile the RCDB conditions, converter
`RunCharge` counters, and optional QADB `Misc` comments needed by the data-efficiency
study. The program writes reusable `rcdb_conditions.tsv` and
`converter_run_charge.tsv` caches, a combined `run_audit.tsv`, an
`audit_summary.json`, and a `run_currents.json` accepted directly by
`study_data_efficiency.py`.

RCDB and QADB do not define analysis classes such as production, luminosity-scan,
trigger-period, or empty-target classes. Supply a previously reviewed manifest with
`--base-manifest` to preserve those judgments while refreshing the objective metadata.
Alternatively, use repeatable `--assign-class CLASS=RUNS` and
`--nominal-current CLASS=NA` options. Previously unseen runs are deliberately labeled
`unclassified` so they cannot silently enter a fit.

For example, in `tcsh` on the farm:

```tcsh
set repo = /work/clas12/storyf/SF_analysis_software_v2.0
source "$repo/docs/jlab-module-setup.csh"
rehash

if ( ! $?RCDB_CONNECTION ) then
    setenv RCDB_CONNECTION mysql://rcdb@clasdb-farm.jlab.org/rcdb
endif

python3 "$repo/analysis/audit_run_conditions.py" \
    --processing-root "$processing" \
    --base-manifest "$previous_manifest" \
    --qadb-datasets rga_fa18_outbending \
    --run-group RGA \
    --period "Fall 2018" \
    --beam-energy-gev 10.604 \
    --output-dir "$audit"
```

The script imports the RCDB Python package directly, including the
`$RCDB_HOME/python` fallback needed when the installed `rcdb` wrapper is not
executable. A later manifest revision can reuse `--rcdb-input
$audit/rcdb_conditions.tsv`, `--run-charge-input
$audit/converter_run_charge.tsv`, and `--qadb-misc-input $audit/qadb_misc.txt`
without accessing the databases or reopening ROOT.

QADB `Misc` comments are audit information, not automatic rejection decisions. They
appear as `qadb_misc_review` flags and must be evaluated explicitly before updating
the converter's `allowMiscRuns` or the downstream run selection.

The converter's optional `diphotonMassSkim` accepts an event when any pair of
reconstructed PID-22 particles falls within the inclusive `minGeV` and
`maxGeV` interval. Set it wider than every downstream pi0 mass variation so
conversion does not define the final signal window.

## Practical Rule

If a cut is part of defining a compact but broadly reusable ntuple, it can run during conversion. If a cut is part of the physics interpretation, optimization, or systematic uncertainty, run it on ROOT.

Fiducial cuts are the one case that can reasonably live in both places. The efficient pattern is:

- during conversion: save all coordinates needed for fiducial decisions, and optionally save boolean fiducial flags
- during post-processing: decide whether the event passes the nominal fiducial cut and each systematic variation

Exclusivity cuts should normally be post-processing cuts because their exact windows depend on channel, calibration state, binning, and systematic studies.

## Code Organization Direction

The converter keeps its small hipo-level preselection helpers inside `src/apps/hipo2root.cpp`, with conversion support code under `src/conversion/`. ROOT-level post-processing keeps the generic `Cuts` module separate from the executable entry point, for example:

```text
include/post/Cuts.h
src/post/Cuts.cpp
src/apps/post_process.cpp
configs/post/*.json
```

That keeps the hipo reader stage independent from the analysis selection stage while still sharing config conventions.

Configuration files are grouped by stage under `configs/`:

- `configs/processing/` for `hipo2root` conversion configs
- `configs/post/` for `post_process` post-processing configs

## Post-Processing Cut Pattern

Post-processing cuts should prefer small named decisions over monolithic pass/fail functions. The `Cuts` class exposes evaluators that return a `CutDecision` with:

- a final pass/fail bit
- the names of failed cut components

Particle-level cuts are now configured as primitive operations inside channel roles instead of hard-coded eppi0 preselection functions. A channel declares the reconstructed particle roles it needs, the PID for each role, how many particles of that role to choose, and the cuts that apply to each candidate. For example, a CVT phi gap should be expressed as a configurable primitive:

```json
{ "name": "proton.cvt_phi_25_40", "op": "removeCVTPhi", "min": 25.0, "max": 40.0 }
```

Detector acceptance is configured at the particle-role level with `detectors`, for example `"detectors": [0, 1]` for FT or FD electrons and `"detectors": [2]` for CD protons. Leaving `detectors` empty or omitted accepts any detector.

For FD particles, fiducial tags are dispatched by detector responsibility:
electrons receive DC and ECAL cuts, protons receive DC cuts, and photons receive
ECAL cuts. This allows one RGA post-processing configuration to enable
`DCEdges_RGA`, `FT_RGA`, `ECAL_RGA`, and `CVT_RGA` without requiring nonexistent
DC information from photons or calorimeter information from protons.

This keeps a cut like `removeCVTPhi(min, max)` reusable across channels and systematic variations. The current primitive vocabulary includes `minP`, `maxP`, `pRange`, `betaRange`, `vzRange`, `minCalEnergy`, `firstPidInstance`, `rejectDetector` for backward compatibility, `rejectSameSectorAsRole`, `vertexDiff`, `removeCVTPhi`, `fiducial`, `minPcalEnergy`, `samplingFractionDiagonal`, and `samplingFractionSigma`. `vzRange` is inclusive at both configured bounds and rejects non-finite vertices. `minCalEnergy` uses the FTCAL deposit for FT particles and the summed PCAL, ECIN, and ECOUT deposits for FD particles. The combined `samplingFraction` operation remains available for older configs.

The `post_process` workflow reads `channel.particles` in order and recursively builds valid candidate combinations. This makes the topology generic enough for channels beyond eppi0. Every channel gets the generic selected-particle branches such as `selectedRoles`, `selectedIdx`, `selectedPid`, and `selectedP`, plus standardized scalar branches for every configured role. A single proton role writes `protonIdx`, `protonPid`, `protonDet`, `protonSector`, `protonP`, `protonTheta`, and `protonPhi`; repeated roles receive numbered names such as `gamma1P` and `gamma2P`. Electron-derived DIS branches `Q2`, `nu`, and `xB` are added when the `electron` role is selected. Downstream converters can therefore copy scalar branches without interpreting role vectors. Use `firstPidInstance` on the electron role when it must be the trigger/scattered electron, meaning the first particle with that PID in the reconstructed bank.

EPPI0 event samples derive an order-independent `rec_ft_photon_count` from the
selected tree's `g1Det` and `g2Det` branches. Exclusivity windows use the proton
detector and this 0/1/2 FT-photon category in both global and bin-local modes;
bin-local groups additionally include the configured Q2, xB, and -t indices.

For EPPI0, all eligible proton-by-photon-pair combinations are evaluated before
one candidate is retained. RGK uses `candidateSelection.method` set to
`pi0MassThenMissingPt`: first choose the photon pair closest to the configured
pi0 mass, then choose the proton that minimizes missing transverse momentum for
that pair. Selected-particle input indices provide the final deterministic
tie-break. The trigger-electron choice remains fixed by `firstPidInstance`; it
does not participate in the combinatorial search. Loose exclusivity is tested
only after the global winner is known, preventing a failed event from selecting
a different combination solely to pass that gate.

Post configs may share repeated topology definitions with an `extends` key. The
parent path is resolved relative to the child config. Object values merge
recursively, while arrays and scalars replace the parent, so child configs can
override names, output paths, or parameter files without duplicating the full
`channel` block.

## Proton Kinematic Corrections

`hipo2root` accepts optional proton correction coefficients through `kinematicCorrections` in the conversion config. The field can be either an inline JSON object or a path to a JSON coefficient file, resolved relative to the config file when the path is relative. `ProtonEnergyLossCorrections` translates the legacy `protonEnergyLoss_params_*.json` format into typed FD/CD correction terms with keys such as `p_delta_p_FD`, `p_delta_theta_CD`, and `p_delta_phi_CD`.

When corrections are enabled, proton `p`, `px`, `py`, `pz`, `theta`, and `phi` are the corrected values used by downstream kinematics. The original measured values remain available as `p_raw`, `theta_raw`, and `phi_raw`, and the applied corrections are saved as `delta_p`, `delta_theta`, and `delta_phi`. Non-proton particles carry raw values equal to the nominal values and zero deltas.

Correction parameters can be derived from matched REC/GEMC ROOT rows with:

```bash
python3 scripts/derive_proton_energy_loss.py matched.root \
  --detector both \
  --output parameters/proton_energy_loss/protonEnergyLoss_params.json \
  --plot-dir calibration_plots/proton_eloss \
  --dataset-tag 6.535RGK_GEMC1 \
  --beam-energy 6.535
```

The script fits residual profiles in theta and momentum bins, then writes the JSON format consumed by `hipo2root`.
When supplied, the dataset tag and beam energy are printed visibly on every plot and embedded in the PNG metadata.

The FD/CD detector, momentum, theta, and REC–GEN matching requirements form one
common proton sample. By default, the theta fit domain is derived from the
sample after broad detector caps and momentum cuts. `--theta-trim-quantile
0.001` trims the lowest and highest 0.1% of reconstructed proton theta before
setting the first and last theta-bin edges, so a tiny edge population does not
define the correction range. The historical fixed ranges are still available
with `--theta-range-mode fixed`; in particular this preserves the old CD
40–58° range for reproducibility. The default CD cap is broader, 40–125°, so
the default final CD bin edge is not hard-coded to 58°.

Each residual fit also uses a sample-derived central quantile interval,
controlled by `--residual-trim-quantile`; the default value `0.01` trims the
lowest 1% and highest 1% of that residual. The range is computed independently
for `delta_p`, `delta_theta`, and `delta_phi`, so a `delta_phi` outlier does
not remove the same proton from the `delta_p` fit. Use
`--residual-range-mode fixed` only when reproducing the historical hard-coded
residual windows is important. These windows are calibration outlier ranges;
they never reject events when the resulting corrections are applied.

## Elastic Momentum-Scale Corrections

Momentum-scale corrections are a second calibration stage, separate from the
simulation-derived proton energy-loss correction above. The converter applies
proton energy loss first and an optional data-derived elastic correction second.
This ordering makes the proton elastic fit describe the remaining tracking
momentum bias rather than absorbing material energy loss again.

The calibration uses `ep -> epX` candidates with exactly one reconstructed
electron and proton, no additional charged tracks, and any number of neutral
particles. Fiducial, electron-identification, vertex, coplanarity, polar-angle
closure, and a broad maximum missing-energy requirement select the elastic
peak. Each particle's momentum residual is constructed without using the other
particle's measured momentum:

```text
p_e,elastic(theta_e) = Ebeam / [1 + Ebeam/Mp (1 - cos(theta_e))]

p_p,elastic(theta_p) =
    2 Mp Ebeam (Ebeam + Mp) cos(theta_p)
    / [(Ebeam + Mp)^2 - Ebeam^2 cos^2(theta_p)]

fractional correction = p_elastic / p_rec - 1
```

This avoids transferring a proton momentum bias into the electron correction,
or the reverse. The particle angles are assumed to be calibrated independently;
angle closure should therefore be inspected as a systematic before accepting
the momentum coefficients.

FD corrections are fitted independently in each sector as a normalized
`theta`/sector-local-`phi` polynomial. The CD proton correction uses normalized
`theta` polynomials multiplied by global-`phi` Fourier terms, preserving
periodicity at +/-180 degrees. Residual-bin centers use a mode-seeded clipped
core, so the elastic peak need not be the majority population. Cells are
rejected on minimum core population, retained fraction, peak significance,
maximum width, and maximum plausible correction. Fits also require at least two
profile cells per parameter and a bounded weighted design condition number.
The default maximum condition number is 100. A phi-dependent FD fit must have
enough theta slices containing multiple independent phi cells; a string of
single cells following the acceptance ridge is rejected even when the linear
algebra technically has full rank. The singular values of the weighted design
matrix are recorded in the JSON. The fitted correction is also sampled on a
grid throughout every accepted support cell and the region is rejected if it
exceeds `--max-abs-surface-correction` (5% by default).
Only accepted profile-cell rectangles are exported as application support;
particles in holes or outside the fitted support retain their input momentum.
The angular-surface coefficients are still obtained from one simultaneous
weighted fit of the accepted cell centers. Diagnostic plots reorganize those
same centers into residual-versus-phi profiles within theta slices, show the
surface's cell-level discrepancies, and—for FD surfaces linear in local
phi—compare independent slice intercepts/slopes with the direct surface's
theta dependence. The secondary slice lines never set the exported correction.
Post-correction core centers are re-extracted from the corrected events in
each accepted cell, so their core membership need not match the before sample.

The default `--profile-binning fixed` preserves the original grid: quantile
theta slices crossed with one uniform phi grid. For high-statistics diagnostic
studies, `--profile-binning adaptive` instead chooses occupancy-quantile phi
edges independently inside each theta slice. `--phi-bins` is then a maximum,
and sparse slices receive fewer phi cells according to
`--target-cell-entries`. `--max-theta-bin-width-deg` additionally splits any
quantile theta interval that is too wide in physical angle. This resolves
rapidly changing low-theta behavior without leaving the sparse high-theta tail
in one multi-degree cell. The JSON records all planned theta/phi edges and
populations under `fit.profileBinning`, and `*_profile_cell_map.png` displays
the actual rectangles, retained populations, and rejected cells.

Adaptive binning is a diagnostic until the structure repeats in held-out runs.
Finer cells can expose unresolved angular dependence, but should not be judged
by a lower training chi-squared alone. Require stable cell centers, adequate
population, and independent closure before using the exported surface.

To split an existing candidate ROOT file into run-specific samples, use the
argument-based helper instead of embedding a quoted C++ expression in
`root -e`:

```bash
python3 scripts/filter_root_by_run.py \
  elastic_candidates.root elastic_candidates_run5423.root 5423
```

The helper reads `sEvents` by default, refuses to replace an existing output,
and accepts `--tree NAME` or an explicit `--overwrite` when needed.

Two adjacent runs are useful as a replication check but cannot establish a
calibration period. For the time-stability study, first make a deterministic
run-spanning HIPO manifest. One file near the middle of every run is usually a
better finite sample than processing the first N files in path order. Run the
manifest command once; then submit the converter in a detached session if
needed:

```bash
python3 scripts/select_hipo_run_sample.py \
  /cache/clas12/rg-a/production/recon/fall2018/torus+1/pass2/dst/recon \
  manifests/rga_fa18_torus+1_one_file_per_run.txt \
  --files-per-run 1

./build/hipo2root \
  configs/processing/rga/10.604/calibration/elastic_data_torus+1_run_spanning.json \
  @manifests/rga_fa18_torus+1_one_file_per_run.txt 0 100000
```

`hipo2root` expands an argument of the form `@manifest`; blank lines and lines
beginning with `#` are ignored. The committed run-spanning processing config
deliberately omits `maxEvents`, so a global event cap cannot stop at the earliest
runs and undo the run-spanning selection. The manifest is the finite-sample
limit.

After post-processing, run the multi-run validator rather than fitting each run
separately:

```bash
python3 scripts/validate_elastic_momentum_runs.py \
  10.604_rga_fa18_torus+1_elastic_candidates_run_spanning.root \
  --beam-energy 10.604 --torus 1 --particle electron \
  --theta-min-deg 6.1 --missing-energy-max-gev 0.75 \
  --profile-binning adaptive \
  --theta-bins 10 --max-theta-bin-width-deg 0.75 \
  --phi-bins 7 --target-cell-entries 2000 --min-bin-entries 300 \
  --block-target-selected 100000 \
  --models constant theta-linear theta-phi theta2-phi \
  --minimum-model-improvement-fraction 0.10 \
  --max-condition-number 100 --max-abs-surface-correction 0.05 \
  --output-dir calibration_plots/momentum/rga_fa18_torus+1_multi_run \
  --dataset-tag 10.604RGA_FA18_torus+1_multi_run
```

The validator forms contiguous run blocks with roughly the requested number of
selected elastic candidates, alternates them between folds A and B, fits one
fold, and evaluates the correction on the other without refitting. It repeats
the direction so every block is held out once. The outputs include:

- `per_run_raw_centers.png`, which reveals run shifts or calibration epochs;
- `common_cell_stability_*.png`, where every run block is evaluated in the
  pooled fit's same theta/phi cells;
- one `heldout_closure.png` and JSON summary for the sector-constant,
  theta-linear, bilinear theta/local-phi, and quadratic-theta/linear-phi
  models;
- `sector_model_comparison.png`, which compares the combined and two
  directional held-out cell RMS values and marks the independently recommended
  model in every detector region;
- `recommended_mixed_parameters.json`, assembled from the pooled region of each
  independently recommended model and explicitly marked pending systematic
  review;
- `run_validation_report.json`, including all blocks, failed regions, held-out
  metrics, and a conservative diagnostic model recommendation.

If many per-run points or common-cell pixels are missing, each run/block is too
small for the chosen minimum core population. Increase `--files-per-run` in a
new manifest rather than reducing the quality thresholds until noisy cells
look stable. A copied candidate file may also be passed as multiple positional
inputs to the validator; their arrays are concatenated before forming run
blocks.

Model selection is performed independently in every detector region. A more
complex nested model is selected only when it reduces the median held-out
cell-center RMS by the configured relative margin and improves both held-out
fold directions. Its independently fitted fold surfaces must also agree at the
pooled cell centers to better than the model's remaining held-out cell RMS. The
default 10% margin is an explicit conservative policy, not a statistical
confidence level, and can be changed with
`--minimum-model-improvement-fraction`. All pooled and mixed parameter files
remain marked pending review and must not be enabled automatically. First use
the time plots to define stable run periods; then repeat the validator within
each period and vary the missing-energy, lower-theta, topology, and binning
selections.

For an FD region the `theta2-phi` candidate has six terms: constant, theta,
theta-squared, local phi, theta times local phi, and theta-squared times local
phi. It is the fractional-correction analogue of the common additive
quadratic-theta/linear-phi form; the validator does not treat elastic data at a
single beam energy as evidence that additive and fractional momentum scaling
are interchangeable.

After fixing the candidate model family, run the one-at-a-time systematic scan
instead of launching independent validators by hand:

```bash
python3 scripts/scan_elastic_momentum_systematics.py \
  10.604_rga_fa18_torus+1_elastic_candidates_run_spanning_5f.root \
  --beam-energy 10.604 --torus 1 --particle electron \
  --theta-min-deg 6.1 --missing-energy-max-gev 0.75 \
  --profile-binning adaptive \
  --theta-bins 10 --max-theta-bin-width-deg 0.75 \
  --phi-bins 7 --target-cell-entries 2000 --min-bin-entries 300 \
  --block-target-selected 100000 \
  --models constant theta-linear theta-phi \
  --minimum-model-improvement-fraction 0.10 \
  --max-condition-number 100 --max-abs-surface-correction 0.05 \
  --missing-energy-scan-gev 0.50 0.75 1.00 \
  --theta-min-scan-deg 6.10 6.50 \
  --target-cell-entries-scan 1500 2000 3000 \
  --output-dir calibration_plots/momentum/rga_fa18_torus+1_systematics \
  --dataset-tag 10.604RGA_FA18_torus+1_systematics
```

The scan loads the candidate tree once and varies one setting at a time around
the nominal configuration. It defines the run blocks from the nominal sample
and reuses the same run membership in every variation, preventing changing
fold boundaries from masquerading as a selection effect. Individual variation
plots are omitted by default; add `--variation-plots` only when they are needed.
The aggregate outputs are:

- `systematic_scan_report.json`, containing every sector's model assignment,
  held-out RMS, and fitted-surface displacement from nominal;
- `systematic_scan_summary.tsv`, a flat table suitable for quick inspection;
- `systematic_scan_summary.png`, a model/RMS/surface-shift overview;
- one subdirectory per variation containing its full validation JSON and
  candidate parameter files.

This remains a diagnostic scan. A stable model label is not sufficient by
itself: inspect the surface RMS changes and held-out behavior before promoting
the nominal mixed parameter file.

For RGK 6.535 GeV, derive a candidate sample and parameters with:

```bash
./build/hipo2root \
  configs/processing/rgk/6.535/calibration/elastic_data.json \
  /path/to/rgk/elastic/hipo

./build/post_process \
  configs/post/rgk/6.535/calibration/elastic_candidates_data.json \
  6.535_rgk_elastic_data.root

python3 scripts/derive_elastic_momentum.py \
  6.535_rgk_elastic_candidates.root \
  --beam-energy 6.535 \
  --torus 1 \
  --missing-energy-max-gev 0.75 \
  --output parameters/momentum/6.535RGK_elastic_momentum.json \
  --plot-dir calibration_plots/momentum/6.535RGK \
  --dataset-tag 6.535RGK_elastic
```

For an electron-only high-statistics adaptive diagnostic, a conservative
starting point is:

```bash
python3 scripts/derive_elastic_momentum.py \
  10.604_rga_fa18_torus+1_elastic_candidates_trial_v2_100M.root \
  --beam-energy 10.604 --torus 1 --particle electron \
  --missing-energy-max-gev 0.75 \
  --theta-min-deg 6.1 \
  --profile-binning adaptive \
  --theta-bins 10 --max-theta-bin-width-deg 0.75 \
  --phi-bins 7 --target-cell-entries 600 --min-bin-entries 300 \
  --theta-order 1 --fd-phi-order 1 \
  --max-condition-number 100 --max-abs-surface-correction 0.05 \
  --output parameters/momentum/10.604RGA_FA18_torus+1_elastic_adaptive_diag.json \
  --plot-dir calibration_plots/momentum/rga_fa18_torus+1_adaptive_diag \
  --dataset-tag 10.604RGA_FA18_torus+1_adaptive_diag
```

Keep that output disabled in production while comparing its cell maps and
profiles with the fixed-grid baseline.

RGA has matching `elastic_data_torus+1.json` / `elastic_data_torus-1.json`
processing configs and `elastic_candidates_data_torus+1.json` /
`elastic_candidates_data_torus-1.json` post-processing configs. Keep the two
polarities separate during fitting and application.

After validating the coefficients on held-out runs, enable a parameter file in
the corresponding production processing config:

```json
"elasticMomentumCorrections":
  "../../../../parameters/momentum/6.535RGK_elastic_momentum.json"
```

The production config must also declare the same `beamEnergy` and nonzero
`torus` stored in the parameter file. Conversion fails on either mismatch.

The corrected `p`, `px`, `py`, and `pz` values then flow to all downstream
kinematics. `p_raw` remains the detector-bank value, `delta_p` is the total
applied momentum change, and `delta_p_energy_loss` plus `delta_p_elastic`
record the two stages separately. Do not enable elastic corrections in the
converter used to derive their own calibration sample.

Electron sampling-fraction PID continues to use `p_raw`, because the committed
sampling-fraction bands were calibrated against detector-bank momentum.
Analysis momentum thresholds and reconstructed physics kinematics use the
corrected `p`.

## Sampling-Fraction Parameters

For a first inclusive-electron SIDIS test, run:

```bash
./build/hipo2root \
  configs/processing/rgk/6.535/calibration/sidis_electrons_data.json \
  /path/to/hipo/files
./build/post_process \
  configs/post/rgk/6.535/calibration/electron_sf_candidates.json \
  6.535_rgk_sidis_electrons.root
python3 scripts/derive_sampling_fraction.py 6.535_rgk_electron_sf_candidates.root \
  --output parameters/sampling_fraction/SF_sigma_cut_params_6.535RGKSKIM1.json \
  --plot-dir calibration_plots/sampling_fraction \
  --dataset-tag 6.535RGKSKIM1 \
  --beam-energy 6.535 \
  --run-group RGK \
  --skim SKIM1 \
  --torus 1
```

The energy-specific processing configs keep events with at least one reconstructed electron, apply QADB filtering, and use loose DIS cuts. RGA Fall 2018 uses separate `sidis_electrons_data_torus+1.json` and `sidis_electrons_data_torus-1.json` configs. The matching candidate post-processing config selects one FD electron, applies the polarity-dependent DC edge, the RGA ECAL fiducial, and the appropriate absolute electron vertex window (`[-18, 10] cm` for torus `+1`, `[-13, 12] cm` for torus `-1`). It writes explicit selected-electron branches such as `electronP`, `electronSector`, and `electronEPCAL` while deliberately applying no sampling-fraction cuts. The derivation script then applies the minimum-PCAL and diagonal preselection before fitting the sigma band.

`post_process` prints progress every 1,000,000 input rows by default. Pass a third argument to change that interval, or `0` to disable progress messages.

`hipo2root` accepts one or more input HIPO files/directories and prints progress
every 1,000,000 input events. Its optional trailing arguments are
`[max_files] [progress_events]`; use `0` for `progress_events` to disable its
progress messages. A processing config may additionally set a positive
`maxEvents` value to impose an exact global input-event limit across all files;
this is independent of both trailing arguments.

During reader construction, `hipo2root` suppresses the known CLAS12ROOT warning
caused by `RICH::Particle` lacking the generic `detector` schema item. RICH is
not used by this converter. Other HIPO and CLAS12ROOT diagnostics are preserved.

For productions where malformed or incomplete HIPO files are possible, enable
open-only input validation in the processing config:

```json
"inputValidation": {
  "enabled": true,
  "skipMalformed": true
}
```

Each input file is first opened in an isolated child process. If the HIPO reader
aborts while reading the file index, the parent process logs the file and skips
it instead of losing the whole conversion. This guard does not read all events;
it is intended to catch file-level/index corruption, not event-level data
problems.

For the 6.535 GeV RGK test sample, use:

```bash
python3 scripts/derive_sampling_fraction.py electron_sf_candidates.root \
  --output parameters/sampling_fraction/SF_sigma_cut_params_6.535RGKSKIM1.json \
  --plot-dir calibration_plots/sampling_fraction \
  --dataset-tag 6.535RGKSKIM1 \
  --beam-energy 6.535 \
  --run-group RGK \
  --skim SKIM1 \
  --torus 1
```

The defaults reproduce the established preselection: `E_PCAL > 0.07 GeV`, followed for `p >= 4.5 GeV` by `E_PCAL/p + E_ECIN/p > 0.2`. The script writes `sampling_fraction_diagonal_cut.png` with the boundary and retained fraction in each sector, then derives the sigma coefficients only from passing electrons. All diagonal parameters have corresponding CLI overrides.

Use `--gemc` when one sector-independent MC fit should be copied to all six sectors. The output JSON contains `sector_1` through `sector_6`, each with `mu_coeffs` and `sigma_coeffs`. After deriving the file, apply all three independent SF cuts with:

```bash
./build/post_process \
  configs/post/rgk/6.535/calibration/electron_sf_selected.json \
  6.535_rgk_sidis_electrons.root
```

The application config resolves `sigma.paramsFile` relative to its own directory. Its electron cut list exposes `minPcalEnergy`, `samplingFractionDiagonal`, and `samplingFractionSigma` separately, so any component can be omitted for a systematic check.

The output JSON also includes a `_metadata` block with the dataset tag, input file, selected electron count, per-sector counts, fit settings, and timestamp. `post_process` ignores `_metadata` and reads only the sector coefficient blocks, so the same file can serve as both machine-readable parameters and a provenance record.
The dataset tag and beam energy are also printed visibly on every generated plot and embedded in its PNG metadata.

## REC/GEN Matching

`hipo2root` can emit matched REC/GEN rows by setting:

```json
{
  "fillMC": true,
  "matchMC": true,
  "saveUnmatchedMC": true,
  "matchMaxAngleDeg": 3.0
}
```

With matching enabled, each reconstructed particle row is paired with the closest unused generator particle of the same PID within `matchMaxAngleDeg`. Matched rows carry both `rec` and `gen` branches, with `rec.matchedGenIdx` and `rec.matchAngleDeg` recording the match. Reconstructed rows without a match have `matchedGenIdx == -999` and a reset `gen` branch. If `saveUnmatchedMC` is true, unmatched generator particles are also written as GEN-only rows with a reset `rec` branch. This layout supports both calibration scripts, which select matched rows, and acceptance studies, which can count unmatched generator rows.

Each reconstructed row also carries `rec.trackChi2`, `rec.trackNDF`, and
`rec.trackChi2N`. Forward-detector charged particles use their linked DC track,
central-detector charged particles use their linked CVT track, and
`trackChi2N` is computed as `trackChi2 / trackNDF` only for a positive finite
NDF. FT particles, neutral particles, and rows without an applicable track keep
all three values as `NAN`. These track-fit quantities are distinct from
`rec.chi2pid`, which is the particle-identification chi-square.

### Compact generated-event acceptance tree

Particle-level unmatched GEN rows are not required for production acceptance
studies when `generatedEventTree` is enabled:

```json
{
  "generatedEventTree": {
    "enabled": true
  },
  "fillMC": true,
  "matchMC": true,
  "saveUnmatchedMC": false
}
```

`hipo2root` fills `gEvents` immediately after reading each MC event and
before QADB, reconstructed final-state, or reconstructed DIS decisions. It has
one scalar row per input event with:

- `sourceFileId` and zero-based `sourceEventIndex`;
- original `runNum`, `eventNum`, `topologyValid`, and `radiative`;
- `stratumFlatIndex` (`-1` without bin-conditional provenance);
- `weight` (one unless `generatorWeights` is enabled);
- generated `Q2`, `nu`, `xB`, `y`, `W`, `minusT`, and `trentoPhi`.

`sourceFileId` is a deterministic hash of the input HIPO basename. When a
multi-input conversion contains duplicate basenames, the converter hashes the
full path for those duplicates instead. The companion `SourceFiles` tree stores
the ID-to-name mapping. This source-aware identity is propagated to selected REC
candidates and is the acceptance join key. It is necessary because GEMC files
use run 11 and may each restart `eventNum` at one; `(runNum,eventNum)` is
therefore diagnostic metadata, not a cross-file primary key.

For stratum-preserving AAO chunks, `hipo2root` can resolve the pooled event
weight from the filename and the repacker provenance:

```json
{
  "generatorWeights": {
    "enabled": true,
    "chunkProvenance": "/path/to/chunk_provenance.json"
  }
}
```

For portal type-2 submissions, the HIPO basename may use
`STRINGID-LUNDFILENAME-OSGID-JOBINDEX.hipo`. The converter extracts the
canonical `sNNNNN__gNNNN__pNNNNNN` token embedded in `LUNDFILENAME` and
requires exactly one match in the provenance table. A missing, ambiguous, or
unknown token is a fatal error rather than a silent unit-weight fallback. This
does not alter the standard LUND event structure. Type-1 generator submissions
are not yet supported.

Radiative topology is `e p pi0 gamma`; non-radiative topology is `e p gamma
gamma`. Invalid generator topologies remain represented with
`topologyValid=false` and reset kinematics, making event accounting explicit.

After this row is saved, the normal `finalState` and DIS skim apply only to the
`rParticles` tree. This permits a compact REC numerator without
removing generated events from the denominator. Use
`configs/processing/rgk/6.535/eppi0_GEMC.json` as the reference.

Channel-specific derived quantities should be isolated behind small channel logic functions. The eppi0 derived variables and loose exclusivity checks now live behind the eppi0 logic path, which is enabled when the configured roles include `electron`, `proton`, and two `gamma` particles. Future channels should follow that pattern: keep role selection and primitive cuts generic, then add a narrow function for channel-specific kinematics and output branches.

Shared derived-kinematics formulas live in `Kinematics`. Use that module for four-vector construction, DIS variables, missing systems, angle/delta-phi helpers, and Trento phi instead of redefining those formulas inside executables or channel logic.
