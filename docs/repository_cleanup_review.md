# Repository cleanup review

Prepared 2026-10-13 EDT from the mounted Farm checkout. This is a review
record, not an authorization to move or delete files.

## Confirmed baseline

The user reconfirmed the following state from an `ifarm` terminal before this
inventory was made:

| Item | Confirmed state |
| --- | --- |
| Parent branch | `codex/modernize-analysis-v2` |
| Parent `HEAD` | `76664ac2531f9310e19907ae617ff422fe578470` |
| Parent upstream | Same commit; `HEAD...origin/codex/modernize-analysis-v2` is `0 0` |
| Parent tracked/untracked status | Clean |
| Parent recovery branch | `codex/farm-recovery-20261013T195457Z` at `ec1814af829c59254008868926d10b5db515ac96` |
| AAO submodule | Detached `HEAD` at recorded commit `4670cb0e1074de6ce48cf323a6434d3411d6bf5b` |
| AAO remote annotation | `remotes/origin/codex/radiative-mode4-core-tail` contains the pinned commit |
| AAO local production-named branch | `codex/radiative-mode4-core-tail` is stale at `c23607465588c1a329d88261db558470a25cdec7` |
| AAO recovery branch | `codex/farm-recovery-20261013T202024Z` at `8c447a7086f674c81b394dc56f4a07ba023d4a9f` |
| Preserved archive checksums | Both reported `OK` |

The detached submodule state is normal for a Git submodule and preserves the
parent repository's recorded commit. Do not move the local AAO branch merely
to make its name agree with the detached checkout. If AAO development resumes,
first decide whether to branch from the pinned commit or reconcile it with the
stale local branch and the remote-tracking branch.

## Recovery-branch reconciliation

Read-only `ifarm` comparisons were reviewed on 2026-10-13 EDT. Neither
recovery branch should be merged into the active branches.

### Parent recovery branch

`codex/farm-recovery-20261013T195457Z` has two checkpoint commits not present
on the active branch, while the active branch has the later polarity-support,
photon-efficiency, AAO/GEMC-planning, and cache-ignore commits. Its tip differs
from the active tip in only ten paths.

The recovery-only content provides no generic improvement that is absent from
the active tree:

- its `aao_rad_genrec` job is the older outbending-only form; the active
  `scripts/aao_rad_genrec.sbatch` preserves outbending as the default while
  adding validated `AAO_TORUS` selection and `AAO_LABEL` override support;
- its changes to `analysis/aao_hard_interference_ab_test.py` only rewrite two
  equivalent string literals;
- its two proton-energy-loss configs raise `maxEvents` from the maintained and
  documented 50 million to 100 million without a reusable semantic change;
- it lacks active tests, diagnostics, configs, and the AAO LUND plotting
  utility introduced or consolidated later.

Decision: do not merge or cherry-pick the parent recovery branch.

### AAO recovery branch

`codex/farm-recovery-20261013T202024Z` adds nine manual or batch submission
wrappers and changes two submission settings plus the mutable
`aao_rad/aao_input.inp`. The additions encode particular Mode-3 RGA/RGK
production batches, seeds, resources, paths, and generator inputs rather than
generic library or physics improvements. The archived Mode-4 material and the
recovery bundle already preserve this work.

Decision: do not merge or cherry-pick the AAO recovery branch. Keep the AAO
submodule pinned at `4670cb0e1074de6ce48cf323a6434d3411d6bf5b` for artifact
reproducibility and direct future generator development toward EXCLUGEN.

Both recovery branches, bundles, and archives remain preserved. This review
does not authorize deleting them or moving the stale local AAO branch ref.

## Classification rules

- **Core**: part of the maintained extraction, campaign, calibration, or
  validation interface.
- **Reusable diagnostic**: generic enough to retain, although it is not on the
  shortest production path.
- **Campaign/reference study**: useful for a named campaign, collaborator,
  generator, or deferred validation question; it must consume frozen
  artifacts and must not become a second extraction path.
- **Superseded experiment**: retain only for a still-open reproducibility need,
  otherwise archive outside the active repository.
- **Obsolete**: redundant or misplaced with a maintained replacement already
  present.

## Maintained surface

The following entry points form the small supported campaign path. This list
adds `build_event_sample.py`, which the campaign guide and legacy review both
require between GEMC conversion and response construction.

| Purpose | Entry point |
| --- | --- |
| HIPO conversion | `build/hipo2root` |
| Candidate construction | `build/post_process` |
| GEMC generated-left event sample | `analysis/build_event_sample.py` |
| Data export | `analysis/export_selected_data.py` |
| Data/GEMC exclusivity | `analysis/derive_exclusivity.py` |
| Response through structure functions | `analysis/run_analysis.py` |
| Closure and estimator choice | `analysis/validate_response_closure.py` |
| Event/detector diagnostics | `analysis/plot_event_kinematics.py` |
| Data/GEMC comparison | `analysis/compare_event_kinematics.py` |
| Current-efficiency study | `analysis/study_data_efficiency.py` |
| Polarity combination | `analysis/combine_polarity_cross_sections.py` |
| Beam-spin asymmetry | `analysis/beam_spin_asymmetry.py` |
| Collaborator export | `analysis/export_reduced_cross_sections.py` |

## `analysis/*.py` review

All 44 files are accounted for below.

| Classification | Files | Decision |
| --- | --- | --- |
| Core | `beam_spin_asymmetry.py`, `build_event_sample.py`, `combine_polarity_cross_sections.py`, `compare_event_kinematics.py`, `derive_exclusivity.py`, `export_reduced_cross_sections.py`, `export_selected_data.py`, `plot_event_kinematics.py`, `run_analysis.py`, `study_data_efficiency.py`, `validate_response_closure.py` | Retain and keep documented as the supported surface. |
| Reusable diagnostic/support | `audit_run_conditions.py`, `check_event_keys.py`, `compare_beam_spin_asymmetries.py`, `compare_current_efficiency_artifacts.py`, `compare_gemc_response_efficiencies.py`, `compare_root_distributions.py`, `concatenate_event_samples.py`, `export_selected_particle_matches.py`, `make_common_validity_cross_sections.py`, `make_unit_current_efficiency_artifact.py`, `optimize_beam_spin_binning.py`, `particle_kinematics_diagnostics.py`, `plot_exclusivity_diagnostics.py`, `plot_gen_rec_kinematics.py`, `plot_particle_kinematics.py`, `render_response_closure_diagnostics.py`, `study_pi0_cut_performance.py` | Retain. These cover provenance, controlled variations, generic ROOT/particle diagnostics, compact-sample handling, or rendering from frozen artifacts. `particle_kinematics_diagnostics.py` is a support module rather than a CLI. |
| Campaign/reference study | `compare_bsa_to_andrey.py`, `compare_cross_sections_to_igor.py`, `compare_structure_functions_to_igor.py`, `compare_structure_functions_to_reference.py`, `klimenko_photon_map_sensitivity.py`, `project_gemc_mc_statistics.py` | Retain while the named comparisons and planning questions remain active. Keep their outputs outside the clone and their conclusions provisional where documented. |
| Generator/radiative reference study | `aao_calibration_phi_convention.py`, `aao_hard_interference_ab_test.py`, `aao_internal_final_migration_test.py`, `aao_nested_cell_radiative_correction.py`, `aao_soft_hard_radiative_correction.py`, `aao_vcut_cell_scan.py`, `exclurad_4d_radiative_correction.py`, `exclurad_phi_radiative_correction.py`, `stage_aao_normalizations.py` | Retain for current artifact reproducibility and AAO/EXCLURAD comparison, but do not treat them as a production extraction path. Reassess as one group after an EXCLUGEN replacement exists. The first six are diagnostic investigations, not permission to restore archived Mode-4 development. |
| Compatibility/deprecation candidate | `interactive_histograms.py` | Do not remove yet. It is a thin wrapper around `python3 -m visualizer`, but `analysis/README.md` explicitly promises compatibility with existing Farm commands. Deprecate in documentation before deletion. |

No active `analysis/*.py` file can yet be called a confirmed superseded Mode-4
implementation. The archived Mode-4 output remains outside the repository, as
required by the handoff.

## `scripts/*` review

All 27 direct script entry points are accounted for below. The implementation
modules in `scripts/calibration/` and their focused tests remain coupled to the
calibration entry points and should be retained with them.

| Classification | Files | Decision |
| --- | --- | --- |
| Reusable calibration interface | `audit_eppi0_proton_roots.py`, `audit_rgk_electron_reference.py`, `check_elastic_momentum_coverage.py`, `compare_elastic_electron_reference.py`, `compare_eppi0_electron_reference.py`, `compare_proton_energy_loss.py`, `derive_elastic_momentum.py`, `derive_eppi0_momentum.py`, `derive_proton_energy_loss.py`, `derive_sampling_fraction.py`, `scan_elastic_momentum_systematics.py`, `validate_elastic_momentum_runs.py`, `validate_eppi0_electron_momentum.py`, `validate_eppi0_momentum_runs.py`, `validate_eppi0_particle_momentum.py` | Retain. These thin wrappers deliberately provide stable commands over tested calibration modules. |
| Reusable support | `filter_root_by_run.py`, `merge_momentum_corrections.py`, `select_hipo_run_sample.py`, `serve_visualizer.sh`, `visualize.sh` | Retain. Each performs a distinct, argument-driven operation. |
| Maintained RGK campaign runner | `run_rgk_6535_pipeline.sh` | Retain. It is parameterized, non-overwriting, and explicitly maintained by `docs/rgk_6535_clean_run.md`; it is not merely a hard-coded alias. |
| AAO model/bin-centering production support | `plot_bin_centering_convergence.py`, `run_bin_centering_swif_job.csh`, `submit_bin_centering_swif.sh` | Retain while AAO supplies bin-centering/model artifacts. Reassess the two SWIF wrappers when EXCLUGEN becomes the supported evaluator. |
| AAO/GEMC campaign support | `aao_lund_kinematics.py`, `aao_rad_genrec.sbatch`, `repack_lund_for_osg.py` | Retain for reproducibility and current diagnostic use, but classify as campaign-specific rather than generic extraction commands. |

No direct script is an immediate deletion candidate based on duplication alone.

## Configuration review

The original 83 JSON files are covered by the following families. The obsolete
run-current duplicate was removed and one shared Spring 2024 post-processing
parent was added, so 83 remain.

| Classification | Paths | Decision |
| --- | --- | --- |
| Core production analysis | `configs/analysis/rga/10.604.json`, `configs/analysis/rgk/6.535.json` | Retain. |
| Core RGA production | Main non-calibration files under `configs/processing/rga/10.604/` and `configs/post/rga/10.604/` for `eppi0_data`, `eppi0_data_full_dst`, `eppi0_GEMC`, and the two polarity-specific base configs | Retain. Polarity distinctions are intentional. |
| Core RGK 6.535 production | Main non-calibration files under `configs/processing/rgk/6.535/` and `configs/post/rgk/6.535/`, including `aao_rad_eppi0_loose.json` | Retain while reproducing the accepted RGK campaign. |
| Reusable calibration | Every JSON below `configs/{processing,post}/{rga/10.604,rgk/6.535}/calibration/` | Retain. Trial variants have already been consolidated into named base, data/MC, and run-spanning configurations. |
| Reusable cut diagnostics | `configs/post/rga/10.604/eppi0_cut_diagnostics*.json` | Retain. These expose tagged cuts and use inheritance for the GEMC variants. |
| Canonical run-current manifest | `configs/efficiency/rgk/6.535/run_currents.json` | Retain. This is the documented location used by `study_data_efficiency.py`. |
| Provisional Spring 2024 studies | `configs/analysis/rgk/{6.395,8.477}.json`, every processing/post JSON under those two energy directories, and the shared non-executable parent at `configs/post/rgk/spring2024/eppi0_base_baseline_uncalibrated.json` | Retain, with `baseline_uncalibrated` naming and provisional status unchanged. |
| Generator/reference diagnostics | All four JSON files under `configs/diagnostics/rga/10.604/` plus `configs/processing/rga/10.604/eppi0_genrec_diagnostic_torus*.json` | Retain while the matching AAO/EXCLURAD studies remain. These are not nominal production configs. |
| Obsolete duplicate | `configs/analysis/6.535_run_currents.json` | Removed with explicit user authorization on 2026-10-13. It was semantically identical to the canonical efficiency manifest, violated the documented config layout, and had no in-repository reference. |

The duplicate manifests have different byte-level hashes only because their
indentation and trailing newline differ; parsing both JSON documents produces
equal data.

### Configuration consolidation completed

A semantic scan found no remaining exactly duplicate JSON documents. The
6.395 and 8.477 GeV Spring 2024 post-processing bases were identical except
for `beamEnergy`, so their shared selection was moved to
`configs/post/rgk/spring2024/eppi0_base_baseline_uncalibrated.json`. The two
energy-specific bases now extend that parent and provide only their energy.
Their existing data and GEMC children continue through recursive inheritance.

All 83 JSON documents parse successfully. A resolver matching the repository's
recursive object-merge and array-replacement rules confirmed the expected
energy, torus, particle roles, and output filenames for all six Spring 2024
base/data/GEMC configs, and confirmed that their resolved selections are equal
after removing `beamEnergy`. Processing and numerical-analysis configs remain
separate because their loaders do not share the post-config inheritance
contract and their energy-specific provenance should stay explicit.

## Artifacts removed from the clone

Four ignored ROOT files occupied 61,964,534,370 bytes (57.71 GiB) in the
repository root. Their names indicate calibration/run-spanning products, not
source inputs required to understand the checkout. With explicit user
authorization, they were moved intact and checksum-verified on `ifarm` on
2026-10-13 EDT.

| Exact Farm path | Size | Proposed disposition |
| --- | ---: | --- |
| `/work/clas12/storyf/SF_analysis_software_v2.0/10.604_rga_fa18_torus+1_elastic_candidates_run_spanning_25f.root` | 5.76 GiB | Moved and verified. |
| `/work/clas12/storyf/SF_analysis_software_v2.0/6.535_rgk_elastic_candidates_run_spanning.root` | 9.25 GiB | Moved and verified. |
| `/work/clas12/storyf/SF_analysis_software_v2.0/6.535_rgk_elastic_electron_candidates_inclusive_run_spanning.root` | 4.01 GiB | Moved and verified. |
| `/work/clas12/storyf/SF_analysis_software_v2.0/6.535_rgk_elastic_electrons_inclusive_data_run_spanning.root` | 38.69 GiB | Moved and verified. |

Archive destination:

```text
/work/clas12/storyf/repo_archives/SF_analysis_software_v2.0_root_artifacts_20261013/
```

The archive contains `MANIFEST.before.txt`, `MANIFEST.after.txt`, and
`SHA256SUMS`. The before/after manifests have identical names, sizes, and
timestamps, all four checksum checks returned `OK`, and no top-level ROOT file
remains in the repository. The archive index at
`/work/clas12/storyf/repo_archives/SF_analysis_software_v2.0_archive_index.txt`
records the new location.

## Completed first cleanup batch

The first batch contained only the two high-confidence cleanup actions:

1. move the four exact ROOT artifacts above to the external archive directory
   and verify checksums; completed on `ifarm`;
2. remove `configs/analysis/6.535_run_currents.json`, leaving the semantically
   identical documented manifest at
   `configs/efficiency/rgk/6.535/run_currents.json`. This action was completed
   with explicit user authorization on 2026-10-13.

No AAO diagnostic pruning, compatibility-wrapper removal, recovery-ref
change, or archive deletion was included in this batch.

## Top-level artifact inventory and disposition

A focused `ifarm` inventory after the first cleanup batch classified the
remaining top-level artifact directories as follows:

| Path | Size / files | Git state | Decision |
| --- | ---: | --- | --- |
| `build/` | 7.7 MiB / 328 files before cleanup | All ignored | The stale tree was removed with explicit user authorization; a fresh ignored tree was then created for the clean validation build. |
| `build_rga_ft_2188af0/` | 8.1 MiB / 284 files | All ignored | Removed with explicit user authorization after log-archive verification. |
| `cmake-build-qadb360/` | 13 MiB / 319 files | All ignored | Removed with explicit user authorization after log-archive verification. |
| `calibration_plots/` | 8.7 MiB / 40 files | All tracked | Retain as source-controlled calibration/reference output. |
| `generator_validation/` | 820 KiB / 22 files | 5 tracked, 17 ignored | Retain intact for now. It mixes a committed validation record with ignored run logs and therefore is not a whole-directory cleanup target. |
| `logs/` | 50 MiB reported by `du` / 97 files | All ignored | Moved to a dated external archive and verified before the build trees were removed. The file manifest records 270,252,175 logical bytes. |
| `manifests/` | Empty | No tracked, untracked, or ignored files | Removed. Its former contents remain listed in the checksum-verified repository recovery archive. |

The three build trees are disposable caches and do not require archival. The
tracked calibration plots and the mixed generator-validation tree must not be
removed as part of a broad ignored-file cleanup. The 16 visible `*.log` files
inside `generator_validation/` can be reviewed as a separate, explicitly
enumerated batch later; their small size does not justify mixing that work into
the whole-directory cleanup.

## Completed second cleanup batch

With explicit user authorization, the 97 ignored files formerly below
`logs/` were moved on `ifarm` to:

```text
/work/clas12/storyf/repo_archives/SF_analysis_software_v2.0_logs_20261014/logs/
```

The archive contains `MANIFEST.before.txt`, `MANIFEST.after.txt`, and
`SHA256SUMS`, each with 97 entries. The before/after manifests compare equal.
The cleanup command permitted deletion only after both the manifest comparison
and `sha256sum -c` succeeded; the subsequently confirmed removal of all three
build trees therefore also confirms that guarded verification passed.

After verification, the three reproducible ignored build trees and the empty
`manifests/` directory were removed. `calibration_plots/` and
`generator_validation/` were deliberately left untouched. Git status retained
only the expected configuration and cleanup-document changes.

## Validation results

The cleaned checkout was configured and built from an absent `build/`
directory on `ifarm` on 2026-10-14 EDT. The fresh build directory is an
intentional ignored cache; the obsolete `build_rga_ft_2188af0/` and
`cmake-build-qadb360/` trees remain absent.

The authoritative Farm validation results were:

- CMake configure: passed;
- C++ build: passed;
- CTest: passed, including the post-config regression checks;
- JSON syntax validation: all 83 configuration files passed;
- analysis Python tests: passed;
- calibration/script Python tests: passed;
- visualizer Python tests: 13 passed;
- `git diff --check`: passed.

`tests/test_post_cuts.cpp` now loads all six Spring 2024 base, data, and GEMC
post-processing configs through `PostCutConfig::fromFile`. It verifies their
resolved energy, torus polarity, output filename, EPPI0 channel, and candidate
selection. This exercises the real recursive inheritance implementation rather
than duplicating it in a test-only parser.

The first visualizer run exposed a stale Node test-harness assumption: it tried
to remove a terminal synchronous `init();` call even though startup now runs
through asynchronous `startVisualizer()`. The harness was updated to suppress
the current startup call before executing the numerical JavaScript functions;
the complete 13-test visualizer suite then passed on the Farm.

## Final checkpoint audit

The final read-only audit on 2026-10-14 EDT found seven tracked changes and the
two expected new files:

- `configs/post/rgk/spring2024/eppi0_base_baseline_uncalibrated.json`;
- `docs/repository_cleanup_review.md`.

`git diff --check` passed. The AAO submodule remains pinned at
`4670cb0e1074de6ce48cf323a6434d3411d6bf5b`. The repository occupies 30 MiB
excluding `.git` and the fresh build tree; the clean validation build occupies
45 MiB and is correctly reported as ignored. The active branch remains
`codex/modernize-analysis-v2`; this audit established readiness for a reviewed
checkpoint commit.
