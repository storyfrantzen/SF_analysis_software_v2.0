# Legacy `analysis_v2.0` review

The legacy workflow contains useful physics choices, but its event model and
execution model are coupled too tightly.  The modern implementation keeps the
physics behavior explicit while making each numerical stage independently
testable.

## Physics behavior to preserve

- CLAS12 bin edges in `Q2`, `xB`, `-t`, and 20 uniform Trento-phi bins;
- legacy flat-bin order: `xB`, `Q2`, `phi`, then `t` fastest;
- separate generated and reconstructed four-dimensional coordinates;
- response columns normalized by all generated events in the truth bin;
- explicit reconstructed feed-in from outside the generated phase space;
- topology-aware exclusivity studies for FD and CD protons;
- iterative Bayesian unfolding, acceptance-corrected comparison, bootstrap
  uncertainty, physical `Q2-xB` bin volumes, virtual-photon flux, luminosity,
  and the `pi0 -> gamma gamma` branching ratio;
- harmonic fits of the reduced cross section versus Trento phi.

## Correctness hazards found

1. `processEPPI0GEMC` intends to fill every generated event, but reconstructed
   loose-exclusivity failures execute `return`.  Those events vanish from the
   generated denominator and bias acceptance upward.
2. A reconstructed final-state or DIS skim at HIPO conversion has the same
   denominator bias.  Generated phase-space and reconstructed-selection masks
   must remain independent.
3. Data bin means in `unfold.py` are calculated before applying the supplied
   exclusivity mask despite the comment saying they represent events passing
   all cuts.
4. `virtual_photon_epsilon` uses the global beam energy while its callers pass
   an energy argument elsewhere.
5. Data and GEMC exclusivity windows are derived separately.  That is useful as
   a diagnostic, but applying different numerical definitions to the data yield
   and MC efficiency changes the measured observable.  The production default
   should derive nominal windows once and apply the same windows to both.
6. The overflow/feed-in treatment uses one global fraction and shape.  It must
   be validated with closure and systematic variations; it is not equivalent
   to modeling all out-of-range truth dimensions.
7. The response uncertainty approximation omits multinomial covariances and the
   unfolding bootstrap fluctuates data only.  These assumptions need to remain
   visible in output metadata.
8. Several scripts hard-code an obsolete dictionary path and output filenames,
   making runs environment- and working-directory-dependent.

## Efficiency problems found

- the same complete ROOT trees are converted to NumPy repeatedly by separate
  scripts;
- nested loops repeatedly construct full-event boolean masks for every
  topology and kinematic bin;
- bin means loop over every flattened bin instead of using grouped reductions;
- plotting, ROOT serialization, and numerical computation are interleaved;
- the bootstrap has no deterministic seed and repeats setup work;
- constants and bin edges are duplicated across scripts.

## Modern execution model

1. Convert HIPO once without reconstructed event rejection for MC.
2. Build one compact event-level analysis sample containing GEN coordinates,
   optional REC coordinates, selection flags, topology, weight, and provenance.
3. Read that sample once per dataset and cache the NumPy representation.
4. Derive/apply exclusivity masks through grouped flat-bin keys.
5. Build the sparse response and all metadata in one stage.
6. Unfold with a deterministic bootstrap and run closure before data results.
7. Normalize cross sections with vectorized, configuration-driven functions.
8. Plot only from self-contained result artifacts.

The remaining integration task is an event-level ROOT adapter.  It must recreate
the legacy radiative/non-radiative GEN candidate logic while guaranteeing one
output event for every generated input event.
