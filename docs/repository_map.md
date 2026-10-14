# Repository map

Use this page to choose the maintained entry point and then follow the linked
detailed document. It is intentionally a map, not a second command manual.

## Return to the project in five minutes

1. Confirm the branch, commit, worktree, and pinned submodule before changing
   code.
2. Read `docs/campaign_template.md` for the production sequence and validation
   gates.
3. Inspect the campaign's `campaign_manifest.json` and source its reviewed
   `restore_campaign_environment.tcsh` when resuming an existing campaign.
4. Use the workflow table below to select one maintained command.
5. Put new ROOT, HIPO, NPZ, PDF, log, and export products in the campaign
   directory outside this clone.

Useful orientation commands from the repository root are:

```tcsh
git status --short --branch
git --no-pager log -5 --oneline
git submodule status -- external/aao_gen
python3 analysis/run_analysis.py --help
python3 scripts/build_campaign_manifest.py --help
```

## Maintained campaign path

| Stage | Use this entry point | Result or decision |
| --- | --- | --- |
| Freeze provenance | `scripts/build_campaign_manifest.py` | Artifact catalog, descriptions, checksums, and restorable shell state |
| Convert HIPO | `build/hipo2root` | ROOT event, particle, and generated-event trees |
| Build candidates | `build/post_process` | Selected event and particle trees |
| Assemble GEMC events | `analysis/build_event_sample.py` | Generated-left compact event sample with matched REC information |
| Export selected data | `analysis/export_selected_data.py` | Compact selected-data sample |
| Derive final selections | `analysis/derive_exclusivity.py` | Separate data and GEMC cut tables and masks |
| Build the response | `analysis/run_analysis.py response` | Sparse response and response metadata |
| Certify unfolding | `analysis/validate_response_closure.py` | Closure, coverage, and estimator/iteration decision |
| Unfold data | `analysis/run_analysis.py unfold` | Unfolded yields and propagated covariance |
| Apply corrections | `analysis/run_analysis.py radiative-correction` and `bin-centering` | Frozen correction artifacts |
| Normalize cross sections | `analysis/run_analysis.py cross-section` | Reduced cross sections with covariance |
| Fit harmonics | `analysis/run_analysis.py fit-harmonics` | Audited harmonic coefficients and covariance |
| Extract structure functions | `analysis/run_analysis.py structure-functions` | Structure-function artifact |
| Audit final outputs | `analysis/run_analysis.py campaign-diagnostics` | Compact numerical Markdown report |

Use `analysis/run_analysis.py COMMAND --help` for the current arguments. The
full schemas and examples live in `analysis/README.md`.

These commands extend a sound single-campaign result; they do not replace the
path above:

| Purpose | Entry point |
| --- | --- |
| Event and detector plots | `analysis/plot_event_kinematics.py` |
| Data/GEMC shape comparison | `analysis/compare_event_kinematics.py` |
| Current-efficiency study | `analysis/study_data_efficiency.py` |
| Polarity combination | `analysis/combine_polarity_cross_sections.py` |
| Beam-spin asymmetry | `analysis/beam_spin_asymmetry.py` |
| Collaborator export | `analysis/export_reduced_cross_sections.py` |

Reference comparisons, AAO/EXCLURAD studies, calibration investigations, and
special diagnostics consume frozen campaign artifacts. They are not alternate
production pipelines.

## Directory ownership

| Path | What belongs here |
| --- | --- |
| `src/apps/` | C++ executable entry points |
| `src/core/`, `include/core/` | Shared tree schema and kinematic helpers |
| `src/conversion/`, `include/conversion/` | HIPO conversion implementation |
| `src/post/`, `include/post/` | Candidate construction and post-processing cuts |
| `analysis/*.py` | Maintained analysis CLIs and reusable diagnostics |
| `analysis/eppi0/` | Tested numerical implementation modules; normally called through a CLI |
| `scripts/` | Campaign provenance, calibration wrappers, and focused support commands |
| `scripts/calibration/` | Tested calibration implementations used by the stable wrappers |
| `configs/processing/` | Conversion, QADB, correction, and ROOT-output decisions |
| `configs/post/` | Candidate cuts, topology policy, and loose exclusivity |
| `configs/analysis/` | Beam, phase space, binning, normalization, and final selection settings |
| `configs/efficiency/` | Reviewed run-current and efficiency inputs |
| `parameters/` | Committed calibration coefficients, separate from analysis settings |
| `data/reference/` | Small reviewed collaborator or published reference tables |
| `tests/`, `analysis/tests/`, `scripts/tests/`, `visualizer/tests/` | C++, numerical, utility, and browser-interface regression tests |
| `visualizer/` | Interactive ROOT-tree cut-study application |
| `external/aao_gen/` | Pinned AAO submodule retained for reproducibility, not new generator development |
| `calibration_plots/` | Source-controlled calibration/reference plots retained by the cleanup review |
| `generator_validation/` | Small committed AAO sampling-validation reference; historical logs and bulk products are archived externally, and new output does not belong here |
| `docs/` | Maintained workflow, schema, design, and audit documentation |
| `build/` | Ignored reproducible CMake build tree |

Campaign products belong outside the clone in the numbered directory layout
defined by `docs/campaign_template.md`.

## Configuration rule of thumb

- Change `configs/processing/` when event reading, detector corrections, QADB,
  stored branches, or broad skims change.
- Change `configs/post/` when candidate composition, particle cuts, topology,
  or loose exclusivity changes.
- Change `configs/analysis/` when binning, phase space, normalization, response,
  or final resolution-relative exclusivity changes.
- Put fitted detector coefficients in `parameters/`, not in analysis configs.
- Keep data and GEMC selections independent when their fitted resolutions
  differ.

Configuration inheritance and campaign-specific examples are documented in
`configs/README.md`.

## Documentation router

| Question | Read this |
| --- | --- |
| What order should a campaign run in? | `docs/campaign_template.md` |
| How do I catalog and restore a campaign? | `docs/campaign_artifact_manifest.md` |
| What arguments and NPZ schemas do analysis commands use? | `analysis/README.md` |
| Which config family should I edit? | `configs/README.md` |
| Where are corrections and selections applied? | `docs/analysis_pipeline.md` |
| What does each ROOT tree contain? | `docs/root_tree_schema.md` |
| How are model and reference comparisons defined? | `docs/model_comparison.md` |
| What legacy behavior was corrected? | `docs/legacy_analysis_v2_review.md` |
| Why was a file retained, archived, or removed? | `docs/repository_cleanup_review.md` |
| How do I reproduce the maintained RGK example? | `docs/rgk_6535_clean_run.md` |

## Validation surfaces

Use validation proportional to the change:

```tcsh
cmake -S . -B build
cmake --build build -j 8
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s analysis/tests -v
python3 -m unittest discover -s scripts/tests -v
python3 -m unittest discover -s visualizer/tests -v
git diff --check
```

Documentation-only changes need link and diff-hygiene review. Configuration
changes also require JSON parsing and the relevant C++ or Python semantic
tests. Numerical changes require their focused suite plus the downstream
campaign gate they affect.
