# SF analysis software v2.0

This project converts CLAS12 hipo files into ROOT TTrees for downstream analysis.

## Current executable

```bash
hipo2root <config.json> <hipo_directory> [max_files] [progress_events]
apply_cuts <post_config.json> <input.root> [progress_rows]
```

Processing configs live in `configs/processing/`, so a typical local run looks like:

```bash
./build/hipo2root configs/processing/eppi0.json /path/to/hipo/files
```

For matched REC/GEN rows used by calibration and acceptance studies:

```bash
./build/hipo2root configs/processing/eppi0_matched.json /path/to/hipo/files
```

For the 6.535 GeV AAORAD OSG `epπ0` comparison, the simulation processing
config uses the common reconstructed DIS region (`Q2 >= 1 GeV2`, `W >= 2 GeV`,
`y <= 0.8`), stores MC truth, and performs same-PID angular REC/GEN matching.
One shared post-processing config selects the best reconstructed `epγγ`
candidate for every generator configuration. Run the five GEMC samples with:

```bash
./scripts/run_aao_osg_comparison.csh
```

To process an existing 6.535 GeV RGK data sample with QADB and overlay it using
the identical reconstructed selection, pass its HIPO directory:

```bash
./scripts/run_aao_osg_comparison.csh /path/to/rgk/data/hipos
```

The driver processes OSG IDs 11221 through 11225, writes one ROOT pair per
configuration under `data/aao_osg_comparison/`, and creates normalized overlays,
a yield/efficiency summary, and (when data are supplied) Jensen-Shannon and
total-variation shape metrics in `data/aao_osg_comparison/plots/`. Incomplete
OSG batches are processed with a warning and their true event counts are read
from the ROOT processing summaries. The post-selection omits the data-derived
sampling-fraction sigma band so a GEMC/data disagreement in that response is not
silently cut away.

The converter currently supports:

- optional QADB filtering and accumulated-charge bookkeeping for data
- final-state filtering
- loose DIS skim cuts
- reconstructed-particle branches
- optional MC truth branches

`hipo2root` reports progress every 1,000,000 input events by default. Pass a
fourth argument to change that interval, or `0` to disable progress output.

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
