# Clean RGK 6.535 GeV production run

Use `scripts/run_rgk_6535_pipeline.sh` for an isolated start-to-finish EPPI0
run. The runner never deletes, moves, or overwrites existing test artifacts. A
new `--work-dir` is required, and an existing directory is rejected unless
`--resume` is explicitly supplied.

On ifarm, first load the environment used by this repository:

```tcsh
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.4
module load qadb/3.1
```

Then run from the repository clone. Replace the example inputs with the exact
data, GEMC, LUND, normalization, and AAO executable paths selected for the
production run:

```bash
./scripts/run_rgk_6535_pipeline.sh \
  --work-dir /volatile/clas12/osg/storyf/rgk_6.535_runs/clean_20260713_v1 \
  --data /path/to/rgk/6.535/data \
  --mc /volatile/clas12/osg/storyf/11285 \
  --born-lund /path/to/aao_norad/lund \
  --radiative-lund /path/to/aao_rad/lund \
  --born-norm /path/to/aao_norad/norm \
  --radiative-norm /path/to/aao_rad/norm \
  --aao-xsec /path/to/aao_xsec \
  --workers 8 \
  --bin-centering-N 4
```

The isolated directory contains:

- `build/`: a fresh CMake build;
- `root/`: converter and selected ROOT files;
- `results/`: exclusivity, response, radiative correction, unfolding,
  bin-centering, cross-section, harmonic, and plot artifacts;
- `logs/`: one log per stage;
- `provenance/`: the commit, dirty-tree status, inputs, and analysis config;
- `.stages/`: completion markers used by `--resume`.

If a stage fails, fix the environmental or input problem and repeat the same
command with `--resume`. Completed stages are skipped. Do not use `--resume`
with a directory belonging to a different run.

For a small end-to-end infrastructure test before production, add
`--max-files 1 --max-lund-files 1 --bin-centering-N 1`. Use production settings
for the final physics result.
