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
top-level `TParameter<double>` named `AccumulatedCharge`.

Load QADB before configuring and building on JLab, for example `module load qadb/3.1`.
If QADB is requested by a config but was unavailable at build time, `hipo2root` exits with
an explicit error.

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

This keeps a cut like `removeCVTPhi(min, max)` reusable across channels and systematic variations. The current primitive vocabulary includes `minP`, `maxP`, `pRange`, `betaRange`, `minCalEnergy`, `firstPidInstance`, `rejectDetector` for backward compatibility, `rejectSameSectorAsRole`, `vertexDiff`, `removeCVTPhi`, `fiducial`, `minPcalEnergy`, `samplingFractionDiagonal`, and `samplingFractionSigma`. The combined `samplingFraction` operation remains available for older configs.

The `post_process` workflow reads `channel.particles` in order and recursively builds valid candidate combinations. This makes the topology generic enough for channels beyond eppi0. Every channel gets the generic selected-particle branches such as `selectedRoles`, `selectedIdx`, `selectedPid`, and `selectedP`, plus electron-derived DIS branches `Q2`, `nu`, and `xB` when the `electron` role is selected. Use `firstPidInstance` on that role when it must be the trigger/scattered electron, meaning the first particle with that PID in the reconstructed bank.

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

The energy-specific processing configs keep events with at least one reconstructed electron, apply QADB filtering, and use loose DIS cuts. The corresponding RGA calibration config is `configs/processing/rga/10.604/calibration/sidis_electrons_data.json`. The candidate post-processing configs select one FD electron, apply the appropriate DC-edge and run-group ECAL fiducials, and write explicit selected-electron branches such as `electronP`, `electronSector`, and `electronEPCAL`. They deliberately apply no sampling-fraction cuts. The derivation script then applies the minimum-PCAL and diagonal preselection before fitting the sigma band.

`post_process` prints progress every 1,000,000 input rows by default. Pass a third argument to change that interval, or `0` to disable progress messages.

`hipo2root` likewise prints progress every 1,000,000 input events. Its optional
arguments are `[max_files] [progress_events]`; use `0` for `progress_events` to
disable its progress messages.

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

### Compact generated-event acceptance tree

Particle-level unmatched GEN rows are not required for production acceptance
studies when `generatedEventTree` is enabled:

```json
{
  "generatedEventTree": {
    "enabled": true,
    "treeName": "GeneratedEvents"
  },
  "fillMC": true,
  "matchMC": true,
  "saveUnmatchedMC": false
}
```

`hipo2root` fills `GeneratedEvents` immediately after reading each MC event and
before QADB, reconstructed final-state, or reconstructed DIS decisions. It has
one scalar row per input event with:

- `sourceFileId` and zero-based `sourceEventIndex`;
- original `runNum`, `eventNum`, `topologyValid`, and `radiative`;
- `weight` (currently one until generator weights are connected);
- generated `Q2`, `nu`, `xB`, `y`, `W`, `minusT`, and `trentoPhi`.

`sourceFileId` is a deterministic hash of the input HIPO basename. The
companion `SourceFiles` tree stores the ID-to-basename mapping. This source-aware
identity is propagated to selected REC candidates and is the acceptance join
key. It is necessary because GEMC files use run 11 and may each restart
`eventNum` at one; `(runNum,eventNum)` is therefore diagnostic metadata, not a
cross-file primary key.

Radiative topology is `e p pi0 gamma`; non-radiative topology is `e p gamma
gamma`. Invalid generator topologies remain represented with
`topologyValid=false` and reset kinematics, making event accounting explicit.

After this row is saved, the normal `finalState` and DIS skim apply only to the
particle-level `Events` tree. This permits a compact REC numerator without
removing generated events from the denominator. Use
`configs/processing/rgk/6.535/eppi0_mc_acceptance.json` as the reference.

Channel-specific derived quantities should be isolated behind small channel logic functions. The eppi0 derived variables and loose exclusivity checks now live behind the eppi0 logic path, which is enabled when the configured roles include `electron`, `proton`, and two `gamma` particles. Future channels should follow that pattern: keep role selection and primitive cuts generic, then add a narrow function for channel-specific kinematics and output branches.

Shared derived-kinematics formulas live in `Kinematics`. Use that module for four-vector construction, DIS variables, missing systems, angle/delta-phi helpers, and Trento phi instead of redefining those formulas inside executables or channel logic.
