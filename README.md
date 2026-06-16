# SF analysis software v2.0

This project converts CLAS12 hipo files into ROOT TTrees for downstream analysis.

## Current executable

```bash
hipo2root <config.json> <hipo_directory> [max_files]
apply_cuts <post_config.json> <input.root>
```

Processing configs live in `configs/processing/`, so a typical local run looks like:

```bash
./build/hipo2root configs/processing/eppi0.json /path/to/hipo/files
```

For matched REC/GEN rows used by calibration and acceptance studies:

```bash
./build/hipo2root configs/processing/eppi0_matched.json /path/to/hipo/files
```

The converter currently supports:

- final-state filtering
- loose DIS skim cuts
- reconstructed-particle branches
- optional MC truth branches

By default, `finalState` rejects reconstructed particles whose PIDs are not listed in the config. Set `inclusive` to `true` for inclusive final-state skims.

`apply_cuts` performs ROOT post-processing. The initial module builds one EPPI0 candidate per event and applies configurable fiducial, sampling-fraction, topology, and loose exclusivity cuts. Post-processing configs live in `configs/post/`, for example `configs/post/eppi0.json`.

## Cut strategy

Use hipo-to-ROOT conversion for stable, IO-saving preselection and branch building. Use ROOT post-processing for tuneable physics selections such as fiducial and exclusivity cuts.

See `docs/analysis_pipeline.md` for the recommended modular layout.

## Repository layout

- `src/` - converter source files
- `include/` - project headers and ROOT dictionary LinkDef
- `scripts/` - calibration and analysis helper scripts
- `vendor/` - vendored header-only dependencies
- `configs/processing/` - hipo-to-ROOT conversion configs
- `configs/post/` - ROOT post-processing configs
- `docs/` - design notes and setup references
- `data/` - local input/output data products, ignored by git
- `build/`, `work-build/`, `cmake-build-*` - local CMake build trees, ignored by git
