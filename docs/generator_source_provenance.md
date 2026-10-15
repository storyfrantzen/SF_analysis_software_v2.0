# Generator source provenance

The detector-simulation handoff is recorded by
`sf-generator-source-manifest-v1`. Each record joins one final HIPO file to the
validated generator run that produced its LUND input. The record preserves the
HIPO, LUND, and generator-config paths together with provider, Born/radiative
mode, channel, model, revision, branch, dirty state, run index, seed, requested
and observed event counts, integrated cross section and units, card and config
hashes, execution path, beam energy, generation cutoff, and nucleon `t`.

Build the source manifest after GEMC reconstruction, using the previously
validated `sf-generator-campaign-v1` manifest:

```bash
python3 analysis/build_generator_source_manifest.py \
  /path/to/generator_campaign.json \
  --mode born \
  --hipo-directory /path/to/pi0p_rga/born/dst_inb \
  --hipo-glob '*.hipo' \
  --output /path/to/pi0p_rga/born/source_manifest.json
```

The default is complete coverage: every run in the selected campaign mode must
have exactly one final HIPO file, and no unknown HIPO run index is accepted.
`--allow-partial` is intended only for an explicitly partial validation job.

Enable the catalog during conversion:

```json
{
  "fillMC": true,
  "generatedEventTree": {"enabled": true},
  "generatorProvenance": {
    "sourceManifest": "/path/to/pi0p_rga/born/source_manifest.json"
  }
}
```

`hipo2root` requires every requested HIPO input to occur in the manifest. It
copies the generator record once into the `SourceFiles` tree. A `born` source
must not contain a hard-radiative event. A `rad` source is a full-RC sample and
legitimately mixes unresolved/soft events with explicit hard-photon events, so
both event states are accepted.

With the production setting `idecay=1`, EXCLUGEN writes a status-0 `pi0`
parent followed by its two decay photons. Born and full-RC soft events
therefore contain two photons. Full-RC hard events contain three: the radiated
photon is written before the two decay photons. `gEvents.radiative` records
this event-level hard-photon state; it is not a copy of the file-level `rad`
mode. The retained parent `pi0` cannot distinguish the two states. EXCLUGEN
`gEvents.weight` remains exactly one: the recorded cross section and event
count belong to correction estimators and must not be applied to
detector-response entries.

`analysis/build_event_sample.py` carries the used `SourceFiles` records into
the output `metadata_json` as `generator_sources`, with compact provider, mode,
model, revision, and schema summaries. Generator identity is therefore
preserved without repeating long paths or hashes on every event row.

The historical `aao-osg-stratum-chunks-v1` manifest remains supported through
the existing `generatorWeights` config. That adapter extracts the canonical AAO
chunk token from each HIPO filename and retains the AAO pooled event weight.
`generatorWeights` and `generatorProvenance` are mutually exclusive.
