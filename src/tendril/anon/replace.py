"""Replace detected entities with stable placeholders, keeping repeats consistent."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from gheim import MergedSpan, Session, Span

from tendril.anon.types import AnonymizedText, Replacement

# Short CLI tags to gheim detector labels. ``LABEL_TO_TAG`` in gheim turns
# ``private_person`` into the ``<PERSON_1>`` placeholder.
ENTITY_LABELS: dict[str, str] = {
    "person": "private_person",
    "email": "private_email",
    "phone": "private_phone",
    "url": "private_url",
    "account": "account_number",
    "secret": "secret",
    "date": "private_date",
    "address": "private_address",
}

DEFAULT_ENTITY_TAGS: tuple[str, ...] = (
    "person",
    "email",
    "phone",
    "url",
    "account",
    "secret",
)


def default_labels() -> frozenset[str]:
    """Gheim labels redacted when the caller does not pass an allow-list."""
    return frozenset(ENTITY_LABELS[tag] for tag in DEFAULT_ENTITY_TAGS)


def parse_entity_tags(value: str) -> tuple[str, ...]:
    """Parse a comma-separated tag list, preserving order and dropping duplicates."""
    tags = [part.strip() for part in value.split(",") if part.strip()]
    if not tags:
        raise ValueError("entities must list at least one tag")
    unknown = [tag for tag in tags if tag not in ENTITY_LABELS]
    if unknown:
        choices = ", ".join(ENTITY_LABELS)
        raise ValueError(f"Unknown entity tags: {', '.join(unknown)}. Choices: {choices}")
    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    return tuple(seen)


def labels_for_tags(tags: Sequence[str]) -> frozenset[str]:
    """Map CLI tags to the gheim labels ``rewrite_text`` filters on."""
    unknown = [tag for tag in tags if tag not in ENTITY_LABELS]
    if unknown:
        choices = ", ".join(ENTITY_LABELS)
        raise ValueError(f"Unknown entity tags: {', '.join(unknown)}. Choices: {choices}")
    return frozenset(ENTITY_LABELS[tag] for tag in tags)


def _overlaps(start: int, end: int, covered: Sequence[tuple[int, int]]) -> bool:
    return any(start < covered_end and end > covered_start for covered_start, covered_end in covered)


def add_case_variants(text: str, spans: Sequence[MergedSpan]) -> list[MergedSpan]:
    """Find further mentions of a known surface, including case differences."""
    covered = [(span.start, span.end) for span in spans]
    surfaces = sorted(
        {(span.label, text[span.start : span.end]) for span in spans},
        key=lambda item: len(item[1]),
        reverse=True,
    )
    extra: list[MergedSpan] = []
    for label, surface in surfaces:
        if not surface.strip():
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if _overlaps(start, end, covered):
                continue
            extra.append(MergedSpan(start=start, end=end, label=label))
            covered.append((start, end))
    if not extra:
        return list(spans)
    return sorted([*spans, *extra], key=lambda span: (span.start, span.end))


def rewrite_text(
    text: str,
    spans: Sequence[Span],
    labels: Collection[str] | None = None,
) -> AnonymizedText:
    """Replace allowed entities in ``text``. Repeated surfaces share one placeholder.

    ``labels`` is a set of gheim detector labels. ``None`` uses
    :func:`default_labels`, which leaves dates and addresses in place.
    Placeholders come from a fresh gheim session, so ``<PERSON_1>`` is local
    to this text.
    """
    allowed = set(default_labels() if labels is None else labels)
    kept = [span for span in spans if span.label in allowed]
    if not text or not kept:
        return AnonymizedText(text=text, replacements=[])
    merged = add_case_variants(
        text,
        Session().apply_spans_detailed(text, list(kept)).merged,
    )
    if not merged:
        return AnonymizedText(text=text, replacements=[])

    session = Session()
    parts: list[str] = []
    records: list[Replacement] = []
    cursor = 0
    for span in merged:
        if span.start < cursor or span.end > len(text) or span.start >= span.end:
            continue
        surface = text[span.start : span.end]
        if not surface:
            continue
        replacement = session.allocate(span.label, surface)
        parts.append(text[cursor : span.start])
        parts.append(replacement)
        records.append(
            Replacement(
                label=span.label,
                original=surface,
                replacement=replacement,
                start=span.start,
                end=span.end,
            )
        )
        cursor = span.end
    parts.append(text[cursor:])
    return AnonymizedText(text="".join(parts), replacements=records)
