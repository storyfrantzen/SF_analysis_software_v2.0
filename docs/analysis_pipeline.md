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
config/post_cuts_*.json
```

That keeps the hipo reader stage independent from the analysis selection stage while still sharing config conventions.

## Post-Processing Cut Pattern

Post-processing cuts should prefer small named decisions over monolithic pass/fail functions. The `Cuts` class exposes evaluators that return a `CutDecision` with:

- a final pass/fail bit
- the names of failed cut components

Particle-level cuts are now configured as primitive operations inside channel roles instead of hard-coded eppi0 preselection functions. A channel declares the reconstructed particle roles it needs, the PID for each role, how many particles of that role to choose, and the cuts that apply to each candidate. For example, a CVT phi gap should be expressed as a configurable primitive:

```json
{ "name": "proton.cvt_phi_25_40", "op": "removeCVTPhi", "min": 25.0, "max": 40.0 }
```

This keeps a cut like `removeCVTPhi(min, max)` reusable across channels and systematic variations. The current primitive vocabulary includes `minP`, `maxP`, `pRange`, `betaRange`, `minCalEnergy`, `rejectDetector`, `rejectSameSectorAsRole`, `vertexDiff`, `removeCVTPhi`, `fiducial`, and `samplingFraction`.

The `apply_cuts` workflow reads `channel.particles` in order and recursively builds valid candidate combinations. This makes the topology generic enough for channels beyond eppi0. Channel-specific derived quantities can still be layered on top: the current output preserves the eppi0 pi0/exclusivity variables when the selected roles include `electron`, `proton`, and two `gamma` particles, while all channels get generic selected-particle branches such as `selectedRoles`, `selectedIdx`, `selectedPid`, and `selectedP`.
