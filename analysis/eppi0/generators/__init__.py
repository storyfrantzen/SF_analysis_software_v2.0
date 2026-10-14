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

__all__ = [
    "CAMPAIGN_MANIFEST_SCHEMA",
    "ExclugenProvider",
    "GeneratorCampaign",
    "GeneratorProvenanceError",
    "GeneratorProvider",
    "GeneratorRunRecord",
    "discover_exclugen_campaign",
    "merge_campaign_metadata",
]
