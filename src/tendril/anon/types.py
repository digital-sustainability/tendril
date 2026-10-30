"""Records produced by entity anonymization."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Replacement:
    """One detected span and the stand-in written in its place.

    ``start`` and ``end`` are character offsets into the original text.
    """

    label: str
    original: str
    replacement: str
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnonymizedText:
    """A rewritten passage and every replacement applied to it."""

    text: str
    replacements: list[Replacement]
