# Campaign artifact and environment manifest

`scripts/build_campaign_manifest.py` maintains a durable catalog for one
campaign directory. It preserves every discovered artifact filename together
with human descriptions and enough reviewed shell state to re-enter the
campaign later.

The catalog has three generated files:

- `01_provenance/campaign_manifest.json`: authoritative, editable metadata;
- `01_provenance/campaign_manifest.md`: generated human-readable inventory;
- `01_provenance/environment/restore_campaign_environment.tcsh`: generated
  environment restoration script.

These three control files are recorded under `catalog_files` rather than
inventorying themselves. All other regular files and symlinks below the
campaign root are cataloged. Symlink targets are recorded but never traversed.

## Create or refresh the catalog

Run the utility on `ifarm`, where it can inspect campaign files in place:

```tcsh
python3 "$repo/scripts/build_campaign_manifest.py" update "$work" \
  --checksum none \
  --shell-path "repo=$repo" \
  --shell-path "work=$work" \
  --shell-variable "campaign_label=$campaign_label"
```

Pass every `tcsh` local variable needed to interpret or resume the campaign.
A child process cannot inspect unexported parent-shell variables, so local
variables must be supplied explicitly with `--shell-path NAME=PATH` or
`--shell-variable NAME=VALUE`. Typical path variables include processing,
post-processing, and analysis configs; data and GEMC manifests; selected ROOT
files; response inputs; and correction artifacts.

The initial snapshot automatically considers the common JLab analysis
environment, including `PATH`, library and Python paths, ROOT, CLAS12ROOT,
HIPO, CCDB, RCDB, QADB, JANA, Java, and module state. Add another exported
variable or a project-specific family with:

```tcsh
  --environment-variable EXTRA_NAME \
  --environment-prefix CAMPAIGN_
```

An ordinary artifact refresh preserves the saved environment. Use
`--refresh-environment` only when intentionally recording a new environment
snapshot. That change should normally accompany a new campaign variant.

Variables whose names look like passwords, tokens, secrets, credentials, or
private keys are never recorded. Connection strings containing embedded
`user:password@host` credentials are also omitted. Do not put secrets in shell
variables passed to the utility.

## Describe artifacts

New files enter `campaign_manifest.json` with an empty `description` and
`description_required: true`. Edit these curated fields in the JSON file:

- `description`: what the artifact contains and why it matters;
- `role`: a short stage or purpose, such as `response` or `cross-section`;
- `tags`: compact search labels such as `production`, `torus+1`, or `provisional`;
- `notes`: limitations, dependencies, or review conclusions.

Do not hand-edit filesystem facts, timestamps, status, or checksums. The next
`update` preserves the curated fields and regenerates the Markdown table.

Missing files remain in the catalog with `status: missing`; they are not
silently forgotten. When both the old and new files have a unique matching
SHA-256, a rename preserves the description and records the old name in
`previous_paths`.

## Checksum policy

Choose checksum work deliberately because ROOT, HIPO, and NPZ products can be
large:

- `--checksum none`: stat files and preserve hashes for unchanged entries;
- `--checksum changed`: hash new or changed files and retain unchanged hashes;
- `--checksum all`: hash every regular file for a final freeze or transfer.

The first `changed` update hashes every existing artifact because all are new
to the catalog. Use `none` while production is active and `all` at the reviewed
freeze when full content identity is required.

## Restore the campaign shell

The generated file must be sourced from `tcsh`; executing it in a child shell
cannot change the current shell:

```tcsh
source "$work/01_provenance/environment/restore_campaign_environment.tcsh"
```

It restores recorded exported variables with `setenv`, explicitly unsets
recorded variables that were absent, and restores supplied local variables
with `set`. It also defines `campaign_root` and `campaign_manifest`.

Regenerate the restoration file from the authoritative JSON if necessary:

```tcsh
python3 "$repo/scripts/build_campaign_manifest.py" render-environment \
  "$work/01_provenance/campaign_manifest.json"
```

The snapshot restores paths and scalar variables. It does not recreate shell
history, aliases, functions, running processes, Kerberos tickets, SSH agents,
or credentials. Keep the separate `module list` provenance described by the
campaign template as a readable audit of the restored module state.

## Validate before comparison or sharing

After assigning descriptions and running the final refresh:

```tcsh
python3 "$repo/scripts/build_campaign_manifest.py" check "$work"
python3 "$repo/scripts/build_campaign_manifest.py" check "$work" \
  --verify-checksums
```

The strict check fails for unrecorded or changed files, currently missing
files, returned historical files, or present artifacts without descriptions.
Checksum verification recomputes every stored SHA-256 and can be expensive;
use it for final freezes, transfers, and recovery audits.

If a reviewed deletion is intentionally retained only as history, acknowledge
that state without dropping its filename or description:

```tcsh
python3 "$repo/scripts/build_campaign_manifest.py" check "$work" \
  --allow-missing
```
