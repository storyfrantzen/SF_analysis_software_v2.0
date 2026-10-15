"""Generator-neutral mapping from reconstructed HIPO files to generator runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .base import GeneratorCampaign, GeneratorProvenanceError, GeneratorRunRecord


SOURCE_MANIFEST_SCHEMA = "sf-generator-source-manifest-v1"
HIPO_RUN_INDEX = re.compile(r"_(\d+)\.hipo$")


def hipo_run_index(path: str | Path) -> int:
    """Extract the terminal EXCLUGEN run index from a final HIPO filename."""

    name = Path(path).name
    match = HIPO_RUN_INDEX.search(name)
    if match is None:
        raise GeneratorProvenanceError(
            f"cannot extract generator run index from HIPO filename {name!r}"
        )
    return int(match.group(1))


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@dataclass(frozen=True)
class GeneratorSourceRecord:
    """One final HIPO source and the generator run from which it descends."""

    hipo_path: Path
    run: GeneratorRunRecord

    def __post_init__(self) -> None:
        if self.hipo_path.suffix != ".hipo":
            raise GeneratorProvenanceError(
                f"generator source is not a HIPO file: {self.hipo_path}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"hipo_path": str(self.hipo_path), **self.run.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratorSourceRecord":
        try:
            hipo_path = Path(str(value["hipo_path"]))
        except (KeyError, TypeError, ValueError) as error:
            raise GeneratorProvenanceError(
                f"invalid generator source record: {error}"
            ) from error
        run_value = dict(value)
        run_value.pop("hipo_path", None)
        return cls(hipo_path=hipo_path, run=GeneratorRunRecord.from_dict(run_value))


@dataclass(frozen=True)
class GeneratorSourceManifest:
    """Strict source catalog consumed by ``hipo2root`` and analysis metadata."""

    sources: tuple[GeneratorSourceRecord, ...]
    campaign_manifest: Path | None = None

    def __post_init__(self) -> None:
        if not self.sources:
            raise GeneratorProvenanceError("generator source manifest contains no sources")
        normalized_paths = [_normalized_path(source.hipo_path) for source in self.sources]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise GeneratorProvenanceError(
                "generator source manifest contains duplicate HIPO paths"
            )

        identities: dict[tuple[str, str, int], GeneratorRunRecord] = {}
        for source in self.sources:
            for label, digest in (
                ("card_signature", source.run.card_signature),
                ("config_sha256", source.run.config_sha256),
            ):
                if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                    raise GeneratorProvenanceError(
                        f"generator source {label} must be SHA-256 hex"
                    )
            identity = (
                source.run.provider,
                source.run.mode,
                source.run.run_index,
            )
            previous = identities.get(identity)
            if previous is not None and previous != source.run:
                raise GeneratorProvenanceError(
                    "generator source manifest contains conflicting run metadata for "
                    f"{identity}"
                )
            identities[identity] = source.run

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(sorted({source.run.mode for source in self.sources}))

    def to_manifest(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "modes": list(self.modes),
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.campaign_manifest is not None:
            value["campaign_manifest"] = str(self.campaign_manifest)
        return value

    def write_manifest(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_manifest(cls, path: str | Path) -> "GeneratorSourceManifest":
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GeneratorProvenanceError(
                f"could not read generator source manifest {source}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise GeneratorProvenanceError("generator source manifest must be an object")
        if value.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise GeneratorProvenanceError(
                f"unknown generator source schema {value.get('schema')!r}"
            )
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, list):
            raise GeneratorProvenanceError(
                "generator source manifest requires a sources array"
            )
        resolved_sources = []
        for record in raw_sources:
            if not isinstance(record, dict):
                raise GeneratorProvenanceError(
                    "generator source records must be objects"
                )
            resolved = dict(record)
            for key in ("hipo_path", "lund_path", "config_path"):
                if key not in resolved:
                    continue
                record_path = Path(str(resolved[key])).expanduser()
                if not record_path.is_absolute():
                    record_path = source.parent / record_path
                resolved[key] = str(record_path.resolve())
            resolved_sources.append(GeneratorSourceRecord.from_dict(resolved))
        campaign_manifest = value.get("campaign_manifest")
        if campaign_manifest is not None:
            campaign_manifest = Path(str(campaign_manifest)).expanduser()
            if not campaign_manifest.is_absolute():
                campaign_manifest = source.parent / campaign_manifest
            campaign_manifest = campaign_manifest.resolve()
        manifest = cls(
            sources=tuple(resolved_sources),
            campaign_manifest=(
                None
                if campaign_manifest is None
                else campaign_manifest
            ),
        )
        recorded_modes = value.get("modes")
        if recorded_modes is not None and recorded_modes != list(manifest.modes):
            raise GeneratorProvenanceError(
                "generator source manifest modes conflict with source records"
            )
        return manifest

    def source_for_hipo(self, path: str | Path) -> GeneratorSourceRecord:
        requested = _normalized_path(path)
        matches = [
            source
            for source in self.sources
            if _normalized_path(source.hipo_path) == requested
        ]
        if not matches:
            raise GeneratorProvenanceError(
                f"HIPO input is absent from generator source manifest: {path}"
            )
        return matches[0]


def build_generator_source_manifest(
    campaign_manifest: str | Path,
    hipo_paths: Iterable[str | Path],
    *,
    mode: str,
    require_all_runs: bool = True,
) -> GeneratorSourceManifest:
    """Join final HIPO files to one mode of a validated campaign manifest."""

    campaign_path = Path(campaign_manifest).resolve()
    campaign = GeneratorCampaign.from_manifest(campaign_path)
    runs = {run.run_index: run for run in campaign.runs_for_mode(mode)}
    if not runs:
        raise GeneratorProvenanceError(
            f"campaign manifest contains no {mode!r} generator runs"
        )

    indexed: dict[int, Path] = {}
    mode_token = re.compile(rf"(?:^|_){re.escape(mode)}(?:_|$)")
    for raw_path in hipo_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise GeneratorProvenanceError(f"HIPO source does not exist: {path}")
        if not mode_token.search(path.stem):
            raise GeneratorProvenanceError(
                f"HIPO filename does not identify requested mode {mode!r}: {path.name}"
            )
        index = hipo_run_index(path)
        if index in indexed:
            raise GeneratorProvenanceError(
                f"duplicate HIPO run index {index}: {indexed[index]} and {path}"
            )
        indexed[index] = path

    unknown = sorted(set(indexed) - set(runs))
    missing = sorted(set(runs) - set(indexed))
    if unknown:
        raise GeneratorProvenanceError(
            f"HIPO run indices are absent from campaign metadata: {unknown}"
        )
    if require_all_runs and missing:
        raise GeneratorProvenanceError(
            f"campaign run indices have no final HIPO mapping: {missing}"
        )
    if not indexed:
        raise GeneratorProvenanceError("no final HIPO files were provided")

    return GeneratorSourceManifest(
        sources=tuple(
            GeneratorSourceRecord(hipo_path=indexed[index], run=runs[index])
            for index in sorted(indexed)
        ),
        campaign_manifest=campaign_path,
    )
