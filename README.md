# SF analysis software v2.0

This project converts CLAS12 hipo files into ROOT TTrees for downstream analysis.

## Current executable

```bash
hipo2root <config.json> <hipo_directory> [max_files]
apply_cuts <post_config.json> <input.root>
```

The converter currently supports:

- final-state filtering
- loose DIS skim cuts
- reconstructed-particle branches
- optional MC truth branches

By default, `finalState` rejects reconstructed particles whose PIDs are not listed in the config. Set `inclusive` to `true` for inclusive final-state skims.

`apply_cuts` performs ROOT post-processing. The initial module builds one EPPI0 candidate per event and applies configurable fiducial, sampling-fraction, topology, and loose exclusivity cuts. See `config/post_cuts_eppi0.json`.

## Cut strategy

Use hipo-to-ROOT conversion for stable, IO-saving preselection and branch building. Use ROOT post-processing for tuneable physics selections such as fiducial and exclusivity cuts.

See `docs/analysis_pipeline.md` for the recommended modular layout.
