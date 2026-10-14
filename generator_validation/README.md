# Generator validation reference

This directory retains one small, source-controlled AAO sampling validation
record. It is historical evidence for the sampling-model change, not a working
directory for new generator output.

The retained record is
`aao_sampling_model5_mode3_rgk_6535_2026-07-22/`:

- `provenance.txt` records the parent and AAO commits, host, compiler, Python
  environment, and MAID07 table path;
- `modules.txt` records the loaded Farm modules;
- `validation.json` contains the numerical comparison and pass decision;
- `shape_comparison.pdf` is the one-page visual comparison;
- `checksums.sha256` identifies the external inputs and the two retained
  result files.

Treat this bundle as immutable. New generator studies should live in an
external campaign directory and use `scripts/build_campaign_manifest.py` to
catalog artifacts, descriptions, checksums, and the restorable shell state.

## Archived historical material

Sixteen ignored run logs formerly mixed into this directory were reviewed and
relocated, with their original relative paths preserved, to:

```text
/work/clas12/storyf/repo_archives/SF_analysis_software_v2.0_generator_validation_logs_20261014/
```

Its `01_provenance/campaign_manifest.json` contains a description, role, tags,
notes, size, timestamp, and SHA-256 checksum for every log and the Farm module
snapshot. The generated Markdown inventory and sourceable `tcsh` environment
restore script sit beside it.

The larger NPZ, CSV, PDF, normalization, guard, and helper artifacts from the
same historical studies were already preserved in the checksum-verified
remaining-artifacts archive:

```text
/work/clas12/storyf/repo_archives/SF_analysis_software_v2.0_20261013T202813Z_remaining_20261014T011318Z/
```

Consult its `source_inventory.tsv` to locate an original
`parent/generator_validation/...` path without unpacking the archive.
