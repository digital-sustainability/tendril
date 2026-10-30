"""Query ids and query texts for areas of action and criteria.

Area-of-action queries are replaced by their LLM summary. Criterion queries
keep only the ``Ausschreibungskriterium`` text; their ambition levels are
rendered as a separate block. Query ids are rewritten to ``Catalog::HF-NNN`` /
``Catalog::K-NNN``. The judge and the training data share these helpers so
both see the same query wording.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_WDG_PREFIX = re.compile(r"^WDG_\d+-")
_FRAGMENT = re.compile(r"(HF|K)-0*(\d+)$", re.IGNORECASE)
_SECTION = re.compile(
    r"<(Handlungsfeld|Ausschreibungskriterium|Ambitionslevel)>\s*(.*?)\s*</\1>",
    re.DOTALL,
)
_AMBITION_HEADER = re.compile(r"(?m)^[ \t]*(Basis|Gute Praxis|Vorbild):[ \t]*")
_AMBITION_ORDER = ("Basis", "Gute Praxis", "Vorbild")


def load_query_map(path: str | Path) -> dict[str, str]:
    """Load a strict ``{"id", "text"}`` JSONL area-query map."""
    resolved = Path(path)
    queries: dict[str, str] = {}
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not query_id or not text:
                raise ValueError(
                    f"Invalid query map row {line_number} in {resolved}: "
                    "both id and text are required"
                )
            previous = queries.setdefault(query_id, text)
            if previous != text:
                raise ValueError(f"Conflicting query text for {query_id} in {resolved}")
    if not queries:
        raise ValueError(f"Query map is empty: {resolved}")
    return queries


def format_ambition_levels(values: dict[str, object]) -> str:
    """Render the per-criterion Ambitionsniveaus block like the annotation tool.

    ``values`` maps the level labels (e.g. Basis/Gut/Beste) to the level texts
    of one criterion (None/NaN/"" for unavailable levels). All levels present
    and identical → a single hint line ("Gleich für alle drei Niveaus"),
    otherwise one ``Label: text`` line per available level; "" when no level
    text is available (block not shown, e.g. Handlungsfeld target or criteria
    without level data).
    """
    cleaned: list[tuple[str, str]] = []
    for label, value in values.items():
        if value is None or (isinstance(value, float) and value != value):
            continue
        text = str(value).strip()
        if text:
            cleaned.append((label, text))
    if not cleaned:
        return ""
    if len(cleaned) == len(values) and len({text for _, text in cleaned}) == 1:
        return "Ambitionsniveaus des Kriteriums:\n\nGleich für alle drei Niveaus"
    lines = "\n".join(f"{label}: {text}" for label, text in cleaned)
    return f"Ambitionsniveaus des Kriteriums:\n{lines}"


def canonical_query_id(catalog: object, query_id: object) -> str:
    """Map a catalog column and raw query id to ``Catalog::HF-NNN`` or ``Catalog::K-NNN``.

    The catalog name comes from the catalog column (``WDG_8-Construction_Buildings``
    -> ``Construction_Buildings``). The fragment kind and number come from the raw
    query id, which for criteria is an uppercase hyphenated token such as
    ``WDG-8-CONSTRUCTION-BUILDINGS-K-057``.
    """
    name = _WDG_PREFIX.sub("", str(catalog).strip())
    match = _FRAGMENT.search(str(query_id).strip())
    if not name or match is None:
        raise ValueError(
            f"Cannot form canonical query id from catalog={catalog!r}, query_id={query_id!r}"
        )
    return f"{name}::{match.group(1).upper()}-{int(match.group(2)):03d}"


def ambition_levels(body: str) -> dict[str, str]:
    """Split an Ambitionslevel body on the Basis / Gute Praxis / Vorbild headers.

    Nested lines that contain a colon (bullet labels, "Inhalt:", and similar)
    stay inside the parent level. Empty levels are omitted. The result follows
    Basis, Gute Praxis, Vorbild regardless of source order.
    """
    text = str(body or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return {}
    matches = list(_AMBITION_HEADER.finditer(text))
    found: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        level_text = text[match.end():end].strip()
        if level_text:
            found[match.group(1)] = level_text
    return {label: found[label] for label in _AMBITION_ORDER if label in found}


def _criterion_sections(raw: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip()
        for match in _SECTION.finditer(str(raw))
    }


def criterion_query_text(raw: str) -> str:
    """Criterion body only: the Ausschreibungskriterium tag, without XML or ambitions."""
    text = _criterion_sections(raw).get("Ausschreibungskriterium", "").strip()
    if not text:
        raise ValueError("Criterion query is missing Ausschreibungskriterium")
    return text


def criterion_ambition_block(raw: str) -> str:
    """Ambition block for ``{ambition_levels}``. Empty when the criterion has none.

    Missing levels are omitted. There is no Nachweise section in this data.
    """
    parsed = ambition_levels(_criterion_sections(raw).get("Ambitionslevel", ""))
    return format_ambition_levels({label: parsed.get(label) for label in _AMBITION_ORDER})


def rewrite_queries(
    frame: pd.DataFrame,
    summaries: dict[str, str],
    query_column: str,
    query_id_column: str,
) -> pd.DataFrame:
    """Replace query ids and set ``{query}`` text plus an ambition block.

    Area rows use the summary map. Criterion rows use only the criterion text.
    Ambition levels are stored separately for the prompt placeholder.
    """
    rewritten = frame.copy()
    canonical_ids: list[str] = []
    texts: list[str] = []
    ambitions: list[str] = []
    for catalog, query_id, level, raw in zip(
        rewritten["catalog"],
        rewritten[query_id_column],
        rewritten["level"],
        rewritten[query_column],
        strict=True,
    ):
        canonical = canonical_query_id(catalog, query_id)
        if level == "area_of_action":
            try:
                text = summaries[canonical]
            except KeyError as exc:
                raise ValueError(
                    f"Area query {canonical!r} is missing from the summary map"
                ) from exc
            ambition = ""
        elif level == "criterion":
            text = criterion_query_text(raw)
            ambition = criterion_ambition_block(raw)
        else:
            raise ValueError(f"Unknown query level {level!r}")
        canonical_ids.append(canonical)
        texts.append(text)
        ambitions.append(ambition)
    rewritten[query_id_column] = canonical_ids
    rewritten[query_column] = texts
    rewritten["ambition_levels"] = ambitions
    return rewritten
