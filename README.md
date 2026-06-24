# SF analysis software v2.0

This project converts CLAS12 HIPO files into ROOT TTrees, builds configurable
exclusive `epπ0` candidates, and provides a NumPy/SciPy response, unfolding,
cross-section, and harmonic-analysis pipeline.

## Build

The C++ executables require ROOT and CLAS12ROOT. QADB support is enabled when
`QADB.h` is available during CMake configuration.

```bash
cmake -S . -B build
cmake --build build -j
```

See `docs/jlab-module-setup.csh` for the JLab environment setup.

## Current executable

```bash
hipo2root <config.json> <hipo_file_or_directory> [max_files] [progress_events]
post_process <post_config.json> <input.root> [progress_rows]
```

Processing configs live in `configs/processing/<run-group>/<energy>/`, so an
RGA data run looks like:

```bash
./build/hipo2root configs/processing/rga/10.604/eppi0_data.json /path/to/hipo/files
```

Production configurations are organized by run group and energy. The complete
RGA 10.604 GeV EPPI0 data/nonradiative-MC set lives under
`configs/{processing,post}/rga/10.604/`, with its numerical analysis settings at
`configs/analysis/rga/10.604.json`. See `configs/README.md` for the layout.

The RGA EPPI0 configuration accepts FD or CD protons, applies the calibrated
FD/CD proton energy-loss corrections during conversion, and applies the RGA
DC, FT, ECAL, and CVT fiducial definitions plus the torus +1 electron
sampling-fraction cuts during candidate selection.

For matched REC/GEN proton rows used to derive the RGA energy-loss correction:

```bash
./build/hipo2root \
  configs/processing/rga/10.604/calibration/proton_energy_loss_mc.json \
  /path/to/hipo/files
```

The proton energy-loss derivation fits each residual after trimming outliers
with sample-derived central quantile ranges by default. Use
`--residual-trim-quantile 0.01` to keep the central 98% of each residual
distribution, or `--residual-range-mode fixed` to reproduce the historical
hard-window behavior.

The theta fit domain is also sample-derived by default. The script first
applies broad detector caps, then uses `--theta-trim-quantile 0.001` to avoid
letting tiny edge populations define the first and last theta bins. Use
`--theta-range-mode fixed` to reproduce the historical FD/CD theta ranges.

For storage-efficient acceptance production:

```bash
./build/hipo2root configs/processing/rgk/6.535/eppi0_mc_acceptance.json /path/to/hipo/files
```

This writes a lightweight `GeneratedEvents` tree before reconstructed topology
or DIS filtering. The particle-level `Events` tree may therefore be REC-skimmed
with `saveUnmatchedMC` disabled without biasing the generated denominator.

`GeneratedEvents` has one row per input MC event. It stores the source-aware
identity `(sourceFileId, sourceEventIndex)`, the original `(runNum, eventNum)`,
generator-topology validity, a radiative flag, weight, and generated `Q2`, `nu`,
`xB`, `y`, `W`, `minusT`, and `trentoPhi`. Invalid generator topologies remain
present with `topologyValid=false`, allowing exact input-event accounting. A
`SourceFiles` tree maps each deterministic `sourceFileId` to its HIPO basename.

GEMC production files commonly restart `eventNum` at one and use `runNum == 11`,
so `(runNum, eventNum)` is not a valid cross-file identity. The analysis join
uses the source-aware key instead.

The converter currently supports:

- optional QADB filtering and accumulated-charge bookkeeping for data
- final-state filtering
- loose DIS skim cuts
- reconstructed-particle branches
- optional MC truth branches
- compact generated-event acceptance trees filled before REC filtering

`hipo2root` reports progress every 1,000,000 input events by default. Pass a
fourth argument to change that interval, or `0` to disable progress output.

By default, `finalState` rejects reconstructed particles whose PIDs are not listed in the config. Set `inclusive` to `true` for inclusive final-state skims.
Use `outputPids` when the event selection should remain broad but only selected
particle rows should be written. For example, proton energy-loss calibration
configs require events with at least one reconstructed proton and set
`outputPids: [2212]` so the ROOT tree stores only proton rows.

`post_process` performs ROOT post-processing. The initial module builds one EPPI0 candidate per event and applies configurable fiducial, sampling-fraction, topology, and loose exclusivity cuts. Post-processing configs are grouped by run group and energy, for example `configs/post/rga/10.604/eppi0_data.json`.

## Acceptance and cross-section analysis

The maintained replacement for the legacy `analysis_v2.0` scripts is under
`analysis/`. Its numerical stages require Python, NumPy, and SciPy; the two ROOT
export adapters additionally require PyROOT and the project ROOT dictionary.

The compact MC flow is:

1. Convert MC with the energy-matched `eppi0_mc*.json` processing config.
2. Run `post_process` to create one selected REC candidate per accepted event.
3. Run `analysis/build_event_sample.py` to left-join selected REC candidates
   onto every valid generated event.
4. Export selected data with `analysis/export_selected_data.py`.
5. Derive separate resolution-relative exclusivity windows for data and GEMC.
6. Build the sparse response, unfold data, normalize the cross section, and fit
   the `A + B cos(phi) + C cos(2 phi)` harmonics.

See `analysis/README.md` for commands, artifact schemas, storage guidance, and
the legacy-file fallback.

## Cut strategy

Use hipo-to-ROOT conversion for stable, IO-saving preselection and branch building. Use ROOT post-processing for tuneable physics selections such as fiducial and exclusivity cuts.

See `docs/analysis_pipeline.md` for the recommended modular layout.

## Repository layout

- `src/apps/` - executable entry points
- `src/core/`, `include/core/` - shared ROOT branch schema and kinematics helpers
- `src/conversion/`, `include/conversion/` - HIPO-to-ROOT conversion support
- `src/post/`, `include/post/` - ROOT post-processing cuts and candidate selection
- `scripts/` - calibration and analysis helper scripts
- `analysis/` - event-sample adapters and the maintained EPPI0 numerical pipeline
- `vendor/` - vendored header-only dependencies
- `configs/processing/` - hipo-to-ROOT conversion configs
- `configs/post/` - ROOT post-processing configs
- `configs/analysis/` - numerical analysis configs
- `docs/` - design notes and setup references
- `data/` - local input/output data products, ignored by git
- `build/`, `work-build/`, `cmake-build-*` - local CMake build trees, ignored by git
