"""Entity anonymization for passages and tables."""

from tendril.anon.anonymize import MODEL_ID, anonymize_texts, build_detector
from tendril.anon.replace import (
    DEFAULT_ENTITY_TAGS,
    ENTITY_LABELS,
    labels_for_tags,
    parse_entity_tags,
)
from tendril.anon.types import AnonymizedText, Replacement

__all__ = [
    "DEFAULT_ENTITY_TAGS",
    "ENTITY_LABELS",
    "MODEL_ID",
    "AnonymizedText",
    "Replacement",
    "anonymize_texts",
    "build_detector",
    "labels_for_tags",
    "parse_entity_tags",
]
