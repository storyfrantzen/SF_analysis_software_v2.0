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

## Practical Rule

If a cut is part of defining a compact but broadly reusable ntuple, it can run during conversion. If a cut is part of the physics interpretation, optimization, or systematic uncertainty, run it on ROOT.

Fiducial cuts are the one case that can reasonably live in both places. The efficient pattern is:

- during conversion: save all coordinates needed for fiducial decisions, and optionally save boolean fiducial flags
- during post-processing: decide whether the event passes the nominal fiducial cut and each systematic variation

Exclusivity cuts should normally be post-processing cuts because their exact windows depend on channel, calibration state, binning, and systematic studies.

## Code Organization Direction

The current converter keeps its small hipo-level preselection helpers inside `src/hipo2root.cpp`. Future post-processing cuts should use a separate ROOT-level module, reserving the generic `Cuts` name for that purpose, for example:

```text
include/Cuts.h
src/Cuts.cpp
src/apply_cuts.cpp
configs/post/*.json
```

That keeps the hipo reader stage independent from the analysis selection stage while still sharing config conventions.

Configuration files are grouped by stage under `configs/`:

- `configs/processing/` for `hipo2root` conversion configs
- `configs/post/` for `apply_cuts` post-processing configs

## Post-Processing Cut Pattern

Post-processing cuts should prefer small named decisions over monolithic pass/fail functions. The `Cuts` class exposes evaluators that return a `CutDecision` with:

- a final pass/fail bit
- the names of failed cut components

Particle-level cuts are now configured as primitive operations inside channel roles instead of hard-coded eppi0 preselection functions. A channel declares the reconstructed particle roles it needs, the PID for each role, how many particles of that role to choose, and the cuts that apply to each candidate. For example, a CVT phi gap should be expressed as a configurable primitive:

```json
{ "name": "proton.cvt_phi_25_40", "op": "removeCVTPhi", "min": 25.0, "max": 40.0 }
```

Detector acceptance is configured at the particle-role level with `detectors`, for example `"detectors": [0, 1]` for FT or FD electrons and `"detectors": [2]` for CD protons. Leaving `detectors` empty or omitted accepts any detector.

This keeps a cut like `removeCVTPhi(min, max)` reusable across channels and systematic variations. The current primitive vocabulary includes `minP`, `maxP`, `pRange`, `betaRange`, `minCalEnergy`, `firstPidInstance`, `rejectDetector` for backward compatibility, `rejectSameSectorAsRole`, `vertexDiff`, `removeCVTPhi`, `fiducial`, and `samplingFraction`.

The `apply_cuts` workflow reads `channel.particles` in order and recursively builds valid candidate combinations. This makes the topology generic enough for channels beyond eppi0. Every channel gets the generic selected-particle branches such as `selectedRoles`, `selectedIdx`, `selectedPid`, and `selectedP`, plus electron-derived DIS branches `Q2`, `nu`, and `xB` when the `electron` role is selected. Use `firstPidInstance` on that role when it must be the trigger/scattered electron, meaning the first particle with that PID in the reconstructed bank.

## Proton Kinematic Corrections

`hipo2root` accepts optional proton correction coefficients through `kinematicCorrections` in the conversion config. The field can be either an inline JSON object or a path to a JSON coefficient file, resolved relative to the config file when the path is relative. `ProtonEnergyLossCorrections` translates the legacy `protonEnergyLoss_params_*.json` format into typed FD/CD correction terms with keys such as `p_delta_p_FD`, `p_delta_theta_CD`, and `p_delta_phi_CD`.

When corrections are enabled, proton `p`, `px`, `py`, `pz`, `theta`, and `phi` are the corrected values used by downstream kinematics. The original measured values remain available as `p_raw`, `theta_raw`, and `phi_raw`, and the applied corrections are saved as `delta_p`, `delta_theta`, and `delta_phi`. Non-proton particles carry raw values equal to the nominal values and zero deltas.

Correction parameters can be derived from matched REC/GEMC ROOT rows with:

```bash
python3 scripts/derive_proton_energy_loss.py matched.root \
  --detector both \
  --output configs/processing/protonEnergyLoss_params.json \
  --plot-dir calibration_plots/proton_eloss
```

The script fits residual profiles in theta and momentum bins, then writes the JSON format consumed by `hipo2root`.

## Sampling-Fraction Parameters

For a first inclusive-electron SIDIS test, run:

```bash
./build/hipo2root configs/processing/sidis_electrons.json /path/to/hipo/files
./build/apply_cuts configs/post/electron_sf.json sidis_electrons.root
python3 scripts/derive_sampling_fraction.py electron_sf_candidates.root \
  --output configs/post/SF_sigma_cut_params_REC.json \
  --plot-dir calibration_plots/sampling_fraction
```

The processing config keeps events with at least one reconstructed electron and loose DIS cuts. The post-processing config currently targets RGK outbending data, selects one FD electron per event, applies common DC-edge and RGK ECAL fiducial cuts, writes explicit selected-electron branches such as `electronP`, `electronSector`, and `electronEPCAL`, and leaves the sampling-fraction cut disabled so the selected sample can be used to derive the cut parameters.

Sampling-fraction mu/sigma parameters can be derived with:

```bash
python3 scripts/derive_sampling_fraction.py rec.root \
  --output configs/post/SF_sigma_cut_params_REC.json \
  --plot-dir calibration_plots/sampling_fraction
```

Use `--gemc` when one sector-independent MC fit should be copied to all six sectors. The output JSON contains `sector_1` through `sector_6`, each with `mu_coeffs` and `sigma_coeffs`, matching the format read by the post-processing `samplingFraction` cut.

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

Channel-specific derived quantities should be isolated behind small channel logic functions. The eppi0 derived variables and loose exclusivity checks now live behind the eppi0 logic path, which is enabled when the configured roles include `electron`, `proton`, and two `gamma` particles. Future channels should follow that pattern: keep role selection and primitive cuts generic, then add a narrow function for channel-specific kinematics and output branches.

Shared derived-kinematics formulas live in `Kinematics`. Use that module for four-vector construction, DIS variables, missing systems, angle/delta-phi helpers, and Trento phi instead of redefining those formulas inside executables or channel logic.
