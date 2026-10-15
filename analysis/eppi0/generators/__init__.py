"""Generator-provider interfaces and implementations."""

from .base import (
    CAMPAIGN_MANIFEST_SCHEMA,
    GeneratorCampaign,
    GeneratorProvenanceError,
    GeneratorProvider,
    GeneratorRunRecord,
    merge_campaign_metadata,
)
from .exclugen import ExclugenProvider, discover_exclugen_campaign
from .source_manifest import (
    SOURCE_MANIFEST_SCHEMA,
    GeneratorSourceManifest,
    GeneratorSourceRecord,
    build_generator_source_manifest,
    hipo_run_index,
)

__all__ = [
    "CAMPAIGN_MANIFEST_SCHEMA",
    "ExclugenProvider",
    "GeneratorCampaign",
    "GeneratorProvenanceError",
    "GeneratorProvider",
    "GeneratorRunRecord",
    "GeneratorSourceManifest",
    "GeneratorSourceRecord",
    "SOURCE_MANIFEST_SCHEMA",
    "build_generator_source_manifest",
    "discover_exclugen_campaign",
    "hipo_run_index",
    "merge_campaign_metadata",
]
