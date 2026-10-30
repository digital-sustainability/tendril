"""Build project-scoped MTEB datasets from curated spans and pipeline passages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import re
import secrets
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import orjson
import zstandard
from datasets import Dataset, Value, load_dataset
from loguru import logger

from tendril.paths import REPOSITORY_PATH

CURATION_USER = "CURATION_USER"
CURATION_FINISHED = "CURATION_FINISHED"
ANNOTATION_FINISHED = "FINISHED"
NON_ANNOTATOR_USERS = frozenset({CURATION_USER, "INITIAL_CAS", "admin"})
PROJECT_JSON = "exportedproject.json"
NE_TYPE_LOCAL = "NamedEntity"
SOFA_TYPE_LOCAL = "Sofa"

PREFIX_BEFORE_DATE = re.compile(
    r"^(?P<sector>.+?)__\d{4}-\d{2}-\d{2}__(?:simap__)?"
    r"(?P<project>\d+)__(?P<file>.+)$"
)
DATE_THEN_PROJECT = re.compile(r"^\d{4}-\d{2}-\d{2}__(?:simap__)?(?P<project>\d+)__(?P<file>.+)$")
PIPELINE_FILE = re.compile(r"^simap_(?P<project>\d+)\.json\.zst$")
IDENTIFIER_FRAGMENT = re.compile(
    r"^(?P<catalog>.+)-(?P<kind>HF|K)-(?P<number>\d+)$",
    re.IGNORECASE,
)
AREA_ID = re.compile(r"^HF-0*(?P<number>\d+)$", re.IGNORECASE)
CRITERION_ID = re.compile(r"^K-0*(?P<number>\d+)$", re.IGNORECASE)
CRITERION_DATASET = "IntelliProcure/SwissSPARK_Catalogs"
KB_LABEL_LINE = re.compile(r'rdfs:label\s+"((?:[^"\\]|\\.)*)"')
KB_SUBCLASS_LINE = re.compile(r"rdfs:subClassOf\s+<([^>]+)>")
KB_SUBJECT_LINE = re.compile(r"^<([^>]+)>")
KB_LABEL_ID = re.compile(r"^([A-Za-z_]+(?:_V\d+(?:\.\d+)*)?)-(HF|K)-(\d+)-(.*)$")
KB_NO_FIELD_MARKERS = ("kein handlungsfeld erkennbar", "no field of action")
CATALOG_VERSION = re.compile(r"_V\d+(?:\.\d+)*$", re.IGNORECASE)
DEHYPHENATION = re.compile(r"-[^\S\n]*\n\s*")
LINE_BREAK_HYPHEN = re.compile(r"-[^\S\n]*\n+\s*(?P<rest>\S+)")
MULTI_WHITESPACE = re.compile(r"\s+")
COORDINATING_CONTINUATIONS = frozenset({"und", "oder", "bzw", "bzw.", "resp", "resp.", "beziehungsweise", "u", "u."})
ANCHOR_LENGTH = 64
ANCHOR_STEP = 96
SPAN_MERGE_TOLERANCE = 5
SPAN_MERGE_RATIO_WARNING = 0.15

DEFAULT_METADATA = REPOSITORY_PATH / "data" / "exports" / "inception_metadata.json"
DEFAULT_PROJECT_MAPPING = REPOSITORY_PATH / "private" / "simap_project_aliases.json"
DEFAULT_ALIAS_KEY = REPOSITORY_PATH / "private" / "test_alias_key"
ALIAS_HASH_DOMAIN = "tendril-test-project"
ALIAS_HASH_LENGTH = 6
DEFAULT_QUERY_MAP = REPOSITORY_PATH / "data" / "queries" / "llm_summary.jsonl"
DEFAULT_RESTRICTIONS = (
    REPOSITORY_PATH / "data" / "exports" / "restrictions" / "test_projects.txt"
)

CATALOG_ALIASES = {
    "building_construction": "Construction_Buildings",
    "food": "Food",
    "furniture": "Furniture",
    "generic": "Product_Independent_Criteria",
    "lighting_electrical_appliances": "Light_Elect_Appl",
    "print_services": "Print_Services",
    "transportation": "Road_Transport",
}


@dataclass(frozen=True)
class Annotation:
    begin: int
    end: int
    identifier: str


@dataclass(frozen=True)
class CuratedDocument:
    name: str
    sofa: str
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True)
class PipelineDocument:
    file_name: str
    passages: tuple[dict, ...]


@dataclass(frozen=True)
class DocumentStates:
    curation: dict[str, str]
    finished_annotators: dict[str, tuple[str, ...]]


@dataclass
class Stats:
    curated_documents: int = 0
    unfinished_documents: int = 0
    empty_documents: int = 0
    empty_documents_without_positives: int = 0
    empty_negative_passages: int = 0
    documents_with_hf_spans: int = 0
    matched_documents: int = 0
    missing_projects: int = 0
    restricted_documents: int = 0
    missing_attachments: int = 0
    mapped_annotations: int = 0
    unmapped_annotations: int = 0
    missing_query_annotations: int = 0
    positive_spans: int = 0
    negative_passages: int = 0
    overlapping_passages: int = 0
    unlocated_passages: int = 0
    spans_before_merge: int = 0
    spans_after_merge: int = 0
    merged_span_clusters: int = 0


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_xmi(source: IO[bytes], document: str) -> CuratedDocument:
    sofa = ""
    annotations: list[Annotation] = []
    try:
        for _, elem in ET.iterparse(source, events=("end",)):
            name = local_tag(elem.tag)
            if name == SOFA_TYPE_LOCAL:
                sofa = elem.attrib.get("sofaString", sofa)
            elif name == NE_TYPE_LOCAL:
                try:
                    begin = int(elem.attrib["begin"])
                    end = int(elem.attrib["end"])
                except (KeyError, ValueError):
                    elem.clear()
                    continue
                annotations.append(
                    Annotation(
                        begin=begin,
                        end=end,
                        identifier=(elem.attrib.get("identifier") or "").strip(),
                    )
                )
            elem.clear()
    except ET.ParseError as exc:
        raise ValueError(f"Unreadable XMI for {document}: {exc}") from exc
    if not sofa:
        raise ValueError(f"No sofaString found for {document}")
    return CuratedDocument(document, sofa, tuple(annotations))


def parse_cas_zip(source: Path | bytes, document: str) -> CuratedDocument:
    archive_source: Path | io.BytesIO
    archive_source = source if isinstance(source, Path) else io.BytesIO(source)
    with zipfile.ZipFile(archive_source) as archive:
        xmi_names = sorted(name for name in archive.namelist() if name.lower().endswith(".xmi"))
        if not xmi_names:
            raise ValueError(f"No XMI file in curated CAS for {document}")
        with archive.open(xmi_names[0]) as xmi:
            return parse_xmi(xmi, document)


def resolve_curation_root(export: Path) -> Path:
    if export.name == "curation" and export.is_dir():
        return export
    root = export / "curation"
    if root.is_dir():
        return root
    raise SystemExit(f"ERROR: no curation/ directory under {export}")


def iter_curation_directory(export: Path) -> Iterator[CuratedDocument]:
    root = resolve_curation_root(export)
    for document_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cas_zip = document_dir / f"{CURATION_USER}.zip"
        xmi = document_dir / CURATION_USER / f"{CURATION_USER}.xmi"
        direct_xmi = document_dir / f"{CURATION_USER}.xmi"
        try:
            if cas_zip.is_file():
                yield parse_cas_zip(cas_zip, document_dir.name)
            elif xmi.is_file():
                with xmi.open("rb") as source:
                    yield parse_xmi(source, document_dir.name)
            elif direct_xmi.is_file():
                with direct_xmi.open("rb") as source:
                    yield parse_xmi(source, document_dir.name)
            else:
                logger.warning(f"Skipping {document_dir.name}: no curated XMI CAS")
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            logger.warning(f"Skipping {document_dir.name}: {exc}")


def curation_zip_members(names: Sequence[str]) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    for name in names:
        parts = [part for part in name.split("/") if part]
        try:
            curation_index = parts.index("curation")
        except ValueError:
            continue
        if len(parts) == curation_index + 3 and parts[-1] == f"{CURATION_USER}.zip":
            members.append((parts[curation_index + 1], name))
    return sorted(members)


def iter_curation_archive(export: Path) -> Iterator[CuratedDocument]:
    with zipfile.ZipFile(export) as archive:
        members = curation_zip_members(archive.namelist())
        if not members:
            raise SystemExit(f"ERROR: no curation/*/{CURATION_USER}.zip entries in {export}")
        for document, member in members:
            try:
                yield parse_cas_zip(archive.read(member), document)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                logger.warning(f"Skipping {document}: {exc}")


def iter_curated_documents(export: Path) -> Iterator[CuratedDocument]:
    if export.is_dir():
        yield from iter_curation_directory(export)
    elif export.is_file() and export.suffix.lower() == ".zip":
        yield from iter_curation_archive(export)
    else:
        raise SystemExit(f"ERROR: --export must be an INCEpTION directory or zip, got {export}")


def load_document_states(export: Path) -> DocumentStates:
    """Read curation states and finished annotators per document from exportedproject.json."""
    if export.is_dir():
        path = export / PROJECT_JSON
        if not path.is_file():
            raise SystemExit(f"ERROR: no {PROJECT_JSON} in {export}")
        payload = orjson.loads(path.read_bytes())
    else:
        with zipfile.ZipFile(export) as archive:
            names = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == PROJECT_JSON]
            if not names:
                raise SystemExit(f"ERROR: no {PROJECT_JSON} in {export}")
            payload = orjson.loads(archive.read(min(names, key=len)))
    curation = {str(row["name"]): str(row.get("state")) for row in payload.get("source_documents", [])}
    finished: dict[str, list[str]] = defaultdict(list)
    for row in payload.get("annotation_documents", []):
        user = str(row.get("user"))
        if user not in NON_ANNOTATOR_USERS and row.get("state") == ANNOTATION_FINISHED:
            finished[str(row["name"])].append(user)
    return DocumentStates(curation, {name: tuple(sorted(users)) for name, users in finished.items()})


def read_annotator_document(export: Path, document: str, user: str) -> CuratedDocument | None:
    if export.is_dir():
        root = export / "annotation" / document
        cas_zip = root / f"{user}.zip"
        if cas_zip.is_file():
            return parse_cas_zip(cas_zip, document)
        for xmi in (root / user / f"{user}.xmi", root / f"{user}.xmi"):
            if xmi.is_file():
                with xmi.open("rb") as source:
                    return parse_xmi(source, document)
        return None
    with zipfile.ZipFile(export) as archive:
        for name in archive.namelist():
            parts = [part for part in name.split("/") if part]
            if len(parts) >= 3 and parts[-3:] == ["annotation", document, f"{user}.zip"]:
                return parse_cas_zip(archive.read(name), document)
    return None


def left_empty_by_annotators(export: Path, document: str, annotators: Sequence[str]) -> bool:
    """True if at least one annotator finished the document and none of them marked a span."""
    if not annotators:
        return False
    for user in annotators:
        try:
            cas = read_annotator_document(export, document, user)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            logger.warning(f"Unreadable annotation of {user} for {document}: {exc}")
            return False
        if cas is None or any(annotation.end > annotation.begin for annotation in cas.annotations):
            return False
    return True


def canonical_area_id(area: str) -> str | None:
    match = AREA_ID.fullmatch(area.strip())
    if not match:
        return None
    return f"HF-{int(match.group('number')):03d}"


def load_queries(query_map: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    if not query_map.is_file():
        raise SystemExit(
            f"ERROR: query map not found: {query_map}\nRun: python scripts/queries/generate_query_maps.py"
        )
    logger.info(f"Loading query texts from {query_map}")
    by_catalog_area: dict[tuple[str, str], str] = {}
    canonical_catalogs: dict[str, str] = {}
    with query_map.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            query_id = str(row.get("id") or "").strip()
            text = str(row.get("text") or "").strip()
            catalog, separator, area_raw = query_id.partition("::")
            area = canonical_area_id(area_raw) if separator else None
            if not catalog or area is None or not text:
                raise SystemExit(f"ERROR: invalid query map row {line_number} in {query_map}")
            canonical_catalogs.setdefault(catalog.casefold(), catalog)
            key = (catalog.casefold(), area)
            previous = by_catalog_area.setdefault(key, text)
            if previous != text:
                logger.warning(f"Conflicting texts for {catalog}::{area}; keeping the first")
    if not by_catalog_area:
        raise SystemExit(f"ERROR: query map is empty: {query_map}")
    return by_catalog_area, canonical_catalogs


def canonical_criterion_id(criterion: str) -> str | None:
    match = CRITERION_ID.fullmatch(criterion.strip())
    if not match:
        return None
    return f"K-{int(match.group('number')):03d}"


def catalog_name(raw_catalog: str, catalogs: dict[str, str]) -> str:
    stripped = CATALOG_VERSION.sub("", raw_catalog)
    alias = CATALOG_ALIASES.get(stripped.casefold(), stripped)
    return catalogs.get(alias.casefold(), alias)


def load_criterion_to_area_map() -> dict[tuple[str, str], str]:
    """(catalog casefold, K-NNN) -> HF-NNN from SwissSPARK_Catalogs."""
    logger.info(f"Loading criterion-to-area map from {CRITERION_DATASET}")
    rows = load_dataset(CRITERION_DATASET, split="train")
    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        catalog = str(row["catalog"]).strip()
        criterion = canonical_criterion_id(str(row["sustainable_procurement_criterion_id"]))
        area = canonical_area_id(str(row["area_of_action_id"]))
        if not catalog or criterion is None or area is None:
            continue
        key = (catalog.casefold(), criterion)
        previous = mapping.setdefault(key, area)
        if previous != area:
            logger.warning(f"Conflicting area for {catalog}::{criterion}; keeping {previous}")
    logger.info(f"Loaded {len(mapping)} criterion-to-area mappings")
    return mapping


def iter_kb_ttl(export: Path) -> Iterator[str]:
    if export.is_dir():
        kb_dir = export / "kb"
        if not kb_dir.is_dir():
            return
        for path in sorted(kb_dir.glob("*.ttl")):
            yield path.read_text(encoding="utf-8", errors="replace")
        return
    if not export.is_file() or export.suffix.lower() != ".zip":
        return
    with zipfile.ZipFile(export) as archive:
        names = sorted(name for name in archive.namelist() if _is_kb_ttl(name))
        for name in names:
            yield archive.read(name).decode("utf-8", errors="replace")


def _is_kb_ttl(name: str) -> bool:
    parts = [part for part in name.split("/") if part]
    return len(parts) >= 2 and parts[-2] == "kb" and parts[-1].lower().endswith(".ttl")


def load_kb_areas(export: Path, catalogs: dict[str, str]) -> dict[str, str]:
    """Map a knowledge-base URI or its fragment to Catalog::HF-NNN.

    Concepts added after the catalog import use opaque hash URIs. Their
    rdfs:label still has the form <catalog>-<HF|K>-<n>-<description>, and a
    criterion points at its parent area via rdfs:subClassOf.
    """
    texts = list(iter_kb_ttl(export))
    if not texts:
        logger.info(f"No knowledge base in {export}")
        return {}
    subject_label: dict[str, str] = {}
    subject_parent: dict[str, str] = {}
    for text in texts:
        for block in text.split("\n\n"):
            subject_match = KB_SUBJECT_LINE.match(block.strip())
            if not subject_match:
                continue
            subject = subject_match.group(1)
            label_match = KB_LABEL_LINE.search(block)
            if label_match:
                subject_label[subject] = label_match.group(1)
            parent_match = KB_SUBCLASS_LINE.search(block)
            if parent_match:
                subject_parent[subject] = parent_match.group(1)

    def label_id(subject: str) -> re.Match[str] | None:
        label = subject_label.get(subject)
        return KB_LABEL_ID.match(label) if label else None

    resolved: dict[str, str] = {}

    def remember(subject: str, query_id: str) -> None:
        resolved.setdefault(subject, query_id)
        resolved.setdefault(subject.rsplit("#", 1)[-1], query_id)

    for subject, label in subject_label.items():
        if any(marker in label.casefold() for marker in KB_NO_FIELD_MARKERS):
            continue
        id_match = label_id(subject)
        if not id_match:
            continue
        if id_match.group(2).upper() == "HF":
            catalog = catalog_name(id_match.group(1), catalogs)
            area = f"HF-{int(id_match.group(3)):03d}"
            remember(subject, f"{catalog}::{area}")
            continue
        parent_match = label_id(subject_parent.get(subject, ""))
        if parent_match and parent_match.group(2).upper() == "HF":
            catalog = catalog_name(parent_match.group(1), catalogs)
            area = f"HF-{int(parent_match.group(3)):03d}"
            remember(subject, f"{catalog}::{area}")
    concepts = sum(1 for key in resolved if "://" in key)
    logger.info(f"Resolved {concepts} knowledge-base concepts from {export}")
    return resolved


def _query_if_known(
    catalog: str,
    area: str | None,
    queries: dict[tuple[str, str], str],
) -> tuple[str | None, str]:
    if not area:
        return None, "unmapped"
    query_id = f"{catalog}::{area}"
    if (catalog.casefold(), area) not in queries:
        return query_id, "missing"
    return query_id, "mapped"


def resolve_identifier(
    identifier: str,
    queries: dict[tuple[str, str], str],
    catalogs: dict[str, str],
    criterion_to_area: dict[tuple[str, str], str],
    kb_areas: dict[str, str],
) -> tuple[str | None, str]:
    """Return (query id, status). Status is mapped, missing, or unmapped.

    Readable HF fragments are normalized. K fragments are mapped to their
    parent area through SwissSPARK, then the export knowledge base. Opaque
    hash URIs are resolved from that knowledge base. An area that resolves
    but is absent from the query map is missing and is not kept.
    """
    fragment = identifier.rsplit("#", 1)[-1]
    match = IDENTIFIER_FRAGMENT.fullmatch(fragment)
    if match:
        catalog = catalog_name(match.group("catalog"), catalogs)
        number = int(match.group("number"))
        if match.group("kind").upper() == "HF":
            return _query_if_known(catalog, f"HF-{number:03d}", queries)
        area = criterion_to_area.get((catalog.casefold(), f"K-{number:03d}"))
        if area is not None:
            return _query_if_known(catalog, area, queries)
    kb_query = kb_areas.get(identifier)
    if kb_query is None:
        kb_query = kb_areas.get(fragment)
    if not kb_query:
        return None, "unmapped"
    catalog, _, area = kb_query.partition("::")
    return _query_if_known(catalog, area or None, queries)


def load_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.is_file():
        return {}, {}
    payload = orjson.loads(path.read_bytes())
    projects_by_document: dict[str, str] = {}
    sectors_by_project: dict[str, str] = {}
    for document, doc_meta in payload.items():
        catalogs = doc_meta.get("catalogs") if isinstance(doc_meta, dict) else None
        for entry in catalogs or []:
            if not isinstance(entry, dict):
                continue
            project_id = str(entry.get("project_id") or "").split("|")[-1].strip()
            if project_id:
                projects_by_document[str(document)] = project_id
                catalog = re.sub(r"^WDG_\d+-", "", str(entry.get("catalog") or ""))
                if catalog and catalog != "Product_Independent_Criteria":
                    sectors_by_project[project_id] = catalog
    return projects_by_document, sectors_by_project


def document_project_source_sector(name: str) -> tuple[str | None, str, str | None]:
    for pattern in (PREFIX_BEFORE_DATE, DATE_THEN_PROJECT):
        match = pattern.match(name)
        if match:
            sector = match.groupdict().get("sector")
            return match.group("project"), match.group("file"), sector
    return None, name, None


def pipeline_file_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SystemExit(f"ERROR: --pipeline is not a directory: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json.zst")):
        match = PIPELINE_FILE.match(path.name)
        if match:
            files[match.group("project")] = path
    if not files:
        raise SystemExit(f"ERROR: no simap_<id>.json.zst files in {directory}")
    return files


def load_pipeline_project(path: Path) -> tuple[PipelineDocument, ...]:
    payload = orjson.loads(zstandard.decompress(path.read_bytes()))
    documents: list[PipelineDocument] = []
    for document in payload.get("documents", []):
        file_name = str(document.get("file_name") or "")
        passages = tuple(
            passage for passage in document.get("content", []) if str(passage.get("content_cleaned") or "").strip()
        )
        if file_name:
            documents.append(PipelineDocument(file_name, passages))
    return tuple(documents)


def filename_variants(name: str) -> set[str]:
    base = unicodedata.normalize("NFKC", Path(name).name).casefold()
    variants = {base}
    without_suffix = str(Path(base).with_suffix(""))
    variants.add(without_suffix)
    if without_suffix.endswith((".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".csv")):
        variants.add(str(Path(without_suffix).with_suffix("")))
    return variants


def attachment_text_score(document: PipelineDocument, normalized_sofa: str) -> int:
    score = 0
    for passage in document.passages[:5]:
        text = normalize_text(str(passage["content_cleaned"]))
        if any(
            text[position : position + ANCHOR_LENGTH] in normalized_sofa for position in anchor_positions(len(text))
        ):
            score += 1
    return score


def find_pipeline_document(
    documents: Sequence[PipelineDocument],
    source_name: str,
    curation_name: str,
    sofa: str,
) -> PipelineDocument | None:
    wanted = filename_variants(source_name) | filename_variants(curation_name)
    matches = [document for document in documents if filename_variants(document.file_name) & wanted]
    if len(matches) == 1:
        return matches[0]
    candidates = matches or list(documents)
    normalized_sofa = normalize_text(sofa)
    scored = [(attachment_text_score(document, normalized_sofa), document) for document in candidates]
    best_score = max((score for score, _ in scored), default=0)
    best = [document for score, document in scored if score == best_score and score > 0]
    if len(best) == 1:
        return best[0]
    if len(best) > 1 or matches:
        logger.warning(f"Ambiguous pipeline attachment for {curation_name}; skipping")
    return None


def normalized_text_with_offsets(text: str) -> tuple[str, list[int]]:
    expanded: list[str] = []
    expanded_offsets: list[int] = []
    for original_index, original_char in enumerate(text):
        normalized_chars = unicodedata.normalize("NFKC", original_char)
        expanded.extend(normalized_chars)
        expanded_offsets.extend([original_index] * len(normalized_chars))
    normalized_source = "".join(expanded)
    output: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(normalized_source):
        dehyphenated = DEHYPHENATION.match(normalized_source, index)
        if dehyphenated:
            index = dehyphenated.end()
            continue
        char = normalized_source[index]
        if char.isspace():
            if output and output[-1] != " ":
                output.append(" ")
                offsets.append(expanded_offsets[index])
            index += 1
            while index < len(normalized_source) and normalized_source[index].isspace():
                index += 1
            continue
        folded = char.casefold()
        output.extend(folded)
        offsets.extend([expanded_offsets[index]] * len(folded))
        index += 1
    if output and output[-1] == " ":
        output.pop()
        offsets.pop()
    return "".join(output), offsets


def normalize_text(text: str) -> str:
    return normalized_text_with_offsets(text)[0]


def clean_corpus_text(text: str) -> str:
    """Undo PDF line wrapping so gold spans match pipeline passage style."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    def join_hyphen_break(match: re.Match[str]) -> str:
        rest = match.group("rest")
        token = re.split(r"\W+", rest, maxsplit=1)[0].casefold()
        if token in COORDINATING_CONTINUATIONS:
            return f"- {rest}"
        first = rest[:1]
        if first.islower():
            return rest
        return f"-{rest}"

    text = LINE_BREAK_HYPHEN.sub(join_hyphen_break, text)
    return MULTI_WHITESPACE.sub(" ", text).strip()


def anchor_positions(length: int) -> list[int]:
    if length <= ANCHOR_LENGTH:
        return [0]
    positions = list(range(0, length - ANCHOR_LENGTH + 1, ANCHOR_STEP))
    last = length - ANCHOR_LENGTH
    if positions[-1] != last:
        positions.append(last)
    return positions


def locate_normalized_passage(
    sofa: str,
    passage: str,
    cursor: int,
) -> tuple[int, int] | None:
    start = sofa.find(passage, cursor)
    if start < 0:
        start = sofa.find(passage)
    if start >= 0:
        return start, start + len(passage)

    estimated_starts: list[int] = []
    search_from = max(0, cursor - ANCHOR_LENGTH)
    for passage_start in anchor_positions(len(passage)):
        anchor = passage[passage_start : passage_start + ANCHOR_LENGTH]
        sofa_start = sofa.find(anchor, search_from)
        if sofa_start >= 0:
            estimated_starts.append(sofa_start - passage_start)
    if not estimated_starts:
        return None
    estimated_start = min(
        estimated_starts,
        key=lambda candidate: (
            candidate < cursor - ANCHOR_LENGTH,
            abs(candidate - cursor),
        ),
    )
    start = max(0, estimated_start)
    return start, min(len(sofa), start + len(passage))


def locate_passages(
    sofa: str,
    passages: Sequence[dict],
) -> list[tuple[dict, tuple[int, int] | None]]:
    normalized_sofa, offsets = normalized_text_with_offsets(sofa)
    located: list[tuple[dict, tuple[int, int] | None]] = []
    cursor = 0
    for passage in passages:
        normalized_passage = normalize_text(str(passage["content_cleaned"]))
        if not normalized_passage:
            located.append((passage, None))
            continue
        normalized_interval = locate_normalized_passage(
            normalized_sofa,
            normalized_passage,
            cursor,
        )
        if normalized_interval is None:
            located.append((passage, None))
            continue
        start, normalized_end = normalized_interval
        begin = offsets[start]
        end = offsets[normalized_end - 1] + 1
        located.append((passage, (begin, end)))
        cursor = normalized_end
    return located


def overlaps(interval: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    begin, end = interval
    return any(begin < span_end and span_begin < end for span_begin, span_end in spans)


def merge_nearby_spans(
    span_queries: dict[tuple[int, int], set[str]],
    document: str,
    tolerance: int = SPAN_MERGE_TOLERANCE,
) -> tuple[dict[tuple[int, int], set[str]], int]:
    """Merge spans within the same document whose boundaries are within
    `tolerance` characters of each other, or that overlap outright (including
    full containment). Curators disagree on exact boundaries far more often
    than they disagree on *which* passage is relevant, so treat such spans as
    one annotation and keep their enclosing interval with the union of their
    query labels.
    """
    spans = sorted(span_queries)
    parent = list(range(len(spans)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i, (begin_i, end_i) in enumerate(spans):
        for j in range(i + 1, len(spans)):
            begin_j, end_j = spans[j]
            if begin_j > end_i + tolerance:
                break
            overlapping = begin_i < end_j and begin_j < end_i
            close = abs(begin_i - begin_j) <= tolerance or abs(end_i - end_j) <= tolerance
            if overlapping or close:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(spans)):
        groups[find(index)].append(index)

    merged: dict[tuple[int, int], set[str]] = {}
    merge_events = 0
    for members in groups.values():
        member_spans = [spans[index] for index in members]
        begin = min(span[0] for span in member_spans)
        end = max(span[1] for span in member_spans)
        query_ids: set[str] = set()
        for span in member_spans:
            query_ids |= span_queries[span]
        merged[(begin, end)] = query_ids
        if len(members) > 1:
            merge_events += 1
            merged_length = end - begin
            smallest_length = min(span[1] - span[0] for span in member_spans)
            if merged_length and smallest_length / merged_length < SPAN_MERGE_RATIO_WARNING:
                logger.warning(
                    f"Merged span cluster with large size disparity in {document}: "
                    f"smallest member {smallest_length} chars vs merged span "
                    f"{merged_length} chars ({begin}:{end})"
                )
    return merged, merge_events


def add_corpus_item(corpus: dict[str, dict], corpus_id: str, text: str) -> None:
    item = {"id": corpus_id, "text": text}
    previous = corpus.setdefault(corpus_id, item)
    if previous != item:
        raise ValueError(f"Corpus ID collision for {corpus_id}")


def add_negative_passage(corpus: dict[str, dict], negatives: set[str], passage: dict, file_name: str) -> bool:
    corpus_id = str(passage.get("part_id") or "").strip()
    if not corpus_id:
        logger.warning(f"Skipping passage without part_id in {file_name}")
        return False
    add_corpus_item(corpus, corpus_id, clean_corpus_text(str(passage["content_cleaned"])))
    negatives.add(corpus_id)
    return True


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def write_qrels_tsv(path: Path, qrels: Sequence[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(qrels)


def write_dataset(
    output: Path,
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels: set[tuple[str, str, int]],
    *,
    alias: str,
    hub_repo: str | None,
    hub_private: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    corpus_rows = [corpus[key] for key in sorted(corpus)]
    query_rows = [{"id": query_id, "text": queries[query_id]} for query_id in sorted(queries)]
    qrel_rows = sorted(qrels)
    write_jsonl(output / "corpus.jsonl", corpus_rows)
    write_jsonl(output / "queries.jsonl", query_rows)
    write_qrels_tsv(output / "qrels" / "test.tsv", qrel_rows)
    if not hub_repo:
        return
    hub_datasets = {
        "corpus": Dataset.from_list(corpus_rows),
        "queries": Dataset.from_list(query_rows),
        "qrels": Dataset.from_list(
            [
                {"query-id": query_id, "corpus-id": corpus_id, "score": score}
                for query_id, corpus_id, score in qrel_rows
            ]
        ).cast_column("score", Value("int32")),
    }
    for component, dataset in hub_datasets.items():
        dataset.push_to_hub(
            hub_repo,
            config_name=f"{alias}-{component}",
            split="test",
            private=hub_private,
        )


def canonical_sector(raw_sector: str | None, catalogs: dict[str, str]) -> str | None:
    if not raw_sector:
        return None
    without_version = CATALOG_VERSION.sub("", raw_sector)
    alias = CATALOG_ALIASES.get(without_version.casefold(), without_version)
    return catalogs.get(alias.casefold())


def load_restricted_projects(path: Path) -> set[str]:
    """Return bare SIMAP IDs of projects whose tender documents restrict reuse."""
    if not path.is_file():
        raise SystemExit(f"ERROR: restrictions file not found: {path}")
    return {line.strip().removeprefix("simap_") for line in path.read_text().splitlines() if line.strip()}


def load_alias_key(path: Path) -> bytes:
    """Return the secret HMAC key for project aliases, creating it once so aliases stay stable."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
        logger.info(f"Created new alias key at {path}")
    return bytes.fromhex(path.read_text().strip())


def project_alias(key: bytes, sector: str, project: str) -> str:
    """Return <sector>-<hex>, a keyed hash of the SIMAP ID that cannot be reversed without the key."""
    digest = hmac.new(key, f"{ALIAS_HASH_DOMAIN}:{project}".encode(), hashlib.sha256).hexdigest()
    return f"{sector}-{digest[:ALIAS_HASH_LENGTH]}"


def assign_project_aliases(path: Path, sectors: dict[str, str], key: bytes) -> dict[str, str]:
    """Derive the alias of every project and write the private mapping of this build."""
    assigned = {project: project_alias(key, sector, project) for project, sector in sectors.items()}
    if len(set(assigned.values())) != len(assigned):
        raise SystemExit("ERROR: project alias collision; increase ALIAS_HASH_LENGTH")
    mapping = {f"simap_{project}": alias for project, alias in assigned.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(mapping, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")
    logger.info(f"Wrote private project mapping to {path}")
    return assigned


def build_dataset(
    export: Path,
    pipeline: Path,
    output: Path,
    project_mapping: Path = DEFAULT_PROJECT_MAPPING,
    hub_repo: str | None = None,
    hub_private: bool = True,
    query_map: Path = DEFAULT_QUERY_MAP,
    restrictions: Path | None = DEFAULT_RESTRICTIONS,
    finished_only: bool = True,
    empty_negatives: bool = False,
    alias_key: Path = DEFAULT_ALIAS_KEY,
) -> Stats:
    """Build one BEIR shard per project.

    With ``finished_only``, documents whose curation is not marked finished are
    skipped entirely, so their unreviewed passages never become negatives. With
    ``empty_negatives``, finished documents without any curated span, which every
    finished annotator also left empty, contribute all passages as negatives to
    projects that have positives. It is off by default: annotators often closed
    such documents within seconds, and some of them contain missed positives.
    """
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"ERROR: --output must be empty: {output}")
    restricted = load_restricted_projects(restrictions) if restrictions is not None else set()
    restricted_seen: set[str] = set()
    query_lookup, catalog_lookup = load_queries(query_map)
    criterion_to_area = load_criterion_to_area_map()
    kb_areas = load_kb_areas(export, catalog_lookup)
    all_query_texts = {
        f"{catalog_lookup[catalog]}::{area}": description for (catalog, area), description in query_lookup.items()
    }
    metadata, metadata_sectors = load_metadata(DEFAULT_METADATA)
    states = load_document_states(export)
    pipeline_files = pipeline_file_index(pipeline)
    pipeline_cache: dict[str, tuple[PipelineDocument, ...]] = {}
    corpora: dict[str, dict[str, dict]] = defaultdict(dict)
    selected_queries: dict[str, dict[str, str]] = defaultdict(dict)
    positive_qrels: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    negative_ids: dict[str, set[str]] = defaultdict(set)
    sector_hints: dict[str, set[str]] = defaultdict(set)
    unmapped = Counter()
    missing_queries = Counter()
    empty_documents: dict[str, list[PipelineDocument]] = defaultdict(list)
    stats = Stats()

    def match_pipeline_document(curated: CuratedDocument) -> tuple[str, PipelineDocument] | None:
        parsed_project, source_name, _ = document_project_source_sector(curated.name)
        project = metadata.get(curated.name, parsed_project)
        if not project or project not in pipeline_files:
            stats.missing_projects += 1
            return None
        if project in restricted:
            stats.restricted_documents += 1
            restricted_seen.add(project)
            return None
        if project not in pipeline_cache:
            pipeline_cache[project] = load_pipeline_project(pipeline_files[project])
        pipeline_document = find_pipeline_document(
            pipeline_cache[project],
            source_name,
            curated.name,
            curated.sofa,
        )
        if pipeline_document is None:
            stats.missing_attachments += 1
            return None
        return project, pipeline_document

    for curated in iter_curated_documents(export):
        stats.curated_documents += 1
        if finished_only and states.curation.get(curated.name) != CURATION_FINISHED:
            stats.unfinished_documents += 1
            continue
        span_queries: dict[tuple[int, int], set[str]] = defaultdict(set)
        for annotation in curated.annotations:
            if annotation.end <= annotation.begin or not annotation.identifier:
                continue
            query_id, status = resolve_identifier(
                annotation.identifier,
                query_lookup,
                catalog_lookup,
                criterion_to_area,
                kb_areas,
            )
            if status == "mapped" and query_id is not None:
                span_queries[(annotation.begin, annotation.end)].add(query_id)
                stats.mapped_annotations += 1
                continue
            if status == "missing" and query_id is not None:
                stats.missing_query_annotations += 1
                missing_queries[query_id] += 1
                continue
            stats.unmapped_annotations += 1
            unmapped[annotation.identifier.rsplit("#", 1)[-1]] += 1
        if not span_queries:
            if (
                empty_negatives
                and states.curation.get(curated.name) == CURATION_FINISHED
                and not any(annotation.end > annotation.begin for annotation in curated.annotations)
                and left_empty_by_annotators(export, curated.name, states.finished_annotators.get(curated.name, ()))
            ):
                matched = match_pipeline_document(curated)
                if matched is not None:
                    project, pipeline_document = matched
                    empty_documents[project].append(pipeline_document)
            continue
        stats.spans_before_merge += len(span_queries)
        span_queries, merge_events = merge_nearby_spans(span_queries, curated.name)
        stats.spans_after_merge += len(span_queries)
        stats.merged_span_clusters += merge_events
        stats.documents_with_hf_spans += 1

        matched = match_pipeline_document(curated)
        if matched is None:
            continue
        project, pipeline_document = matched
        _, _, filename_sector = document_project_source_sector(curated.name)
        stats.matched_documents += 1
        sector = canonical_sector(filename_sector, catalog_lookup) or canonical_sector(
            metadata_sectors.get(project),
            catalog_lookup,
        )
        if sector:
            sector_hints[project].add(sector)

        corpus = corpora[project]
        valid_spans: list[tuple[int, int]] = []
        for (begin, end), query_ids in sorted(span_queries.items()):
            text = clean_corpus_text(curated.sofa[begin:end])
            if not text:
                continue
            document_hash = hashlib.sha256(curated.name.encode()).hexdigest()[:16]
            corpus_id = f"positive::{document_hash}::{begin}:{end}"
            add_corpus_item(corpus, corpus_id, text)
            valid_spans.append((begin, end))
            stats.positive_spans += 1
            for query_id in query_ids:
                selected_queries[project][query_id] = all_query_texts[query_id]
                positive_qrels[project].add((query_id, corpus_id, 1))

        if not valid_spans:
            continue
        for passage, interval in locate_passages(curated.sofa, pipeline_document.passages):
            if interval is not None and overlaps(interval, valid_spans):
                stats.overlapping_passages += 1
                continue
            if interval is None:
                stats.unlocated_passages += 1
            if add_negative_passage(corpus, negative_ids[project], passage, pipeline_document.file_name):
                stats.negative_passages += 1

    # Empty documents only add negatives to projects that have queries.
    for project, pipeline_documents in empty_documents.items():
        if project not in corpora:
            stats.empty_documents_without_positives += len(pipeline_documents)
            continue
        stats.empty_documents += len(pipeline_documents)
        for pipeline_document in pipeline_documents:
            for passage in pipeline_document.passages:
                if add_negative_passage(corpora[project], negative_ids[project], passage, pipeline_document.file_name):
                    stats.empty_negative_passages += 1

    project_sectors: dict[str, str] = {}
    for project in corpora:
        hints = sector_hints[project]
        if not hints:
            hints = {
                query_id.split("::", 1)[0]
                for query_id in selected_queries[project]
                if not query_id.startswith("Product_Independent_Criteria::")
            }
        if len(hints) != 1:
            raise SystemExit(
                f"ERROR: expected one annotation sector for private project {project}, got {sorted(hints)}"
            )
        project_sectors[project] = next(iter(hints))
    project_aliases = assign_project_aliases(project_mapping, project_sectors, load_alias_key(alias_key))

    total_queries = 0
    total_corpus = 0
    total_qrels = 0
    for project in sorted(corpora):
        queries = selected_queries[project]
        qrels = positive_qrels[project]
        qrels.update((query_id, corpus_id, 0) for query_id in queries for corpus_id in negative_ids[project])
        alias = project_aliases[project]
        shard = output / alias
        write_dataset(
            shard,
            corpora[project],
            queries,
            qrels,
            alias=alias,
            hub_repo=hub_repo,
            hub_private=hub_private,
        )
        total_queries += len(queries)
        total_corpus += len(corpora[project])
        total_qrels += len(qrels)
        logger.info(f"Shard {alias}: {len(queries)} queries, {len(corpora[project])} corpus items, {len(qrels)} qrels")
    aliases = sorted(project_aliases.values())
    (output / "mteb_dataset.json").write_bytes(
        orjson.dumps(
            {
                "format": "mteb-retrieval-v2",
                "projects": aliases,
                "hub_repo": hub_repo,
            },
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    if unmapped:
        examples = ", ".join(f"{identifier} ({count})" for identifier, count in unmapped.most_common(10))
        logger.warning(f"Unmapped identifier examples: {examples}")
    if missing_queries:
        examples = ", ".join(f"{query_id} ({count})" for query_id, count in missing_queries.most_common(10))
        logger.warning(f"Resolved areas absent from the query map: {examples}")
    logger.info(
        "Curated docs: {}; skipped unfinished: {}; with mapped HF spans: {}; matched to pipeline: {}; "
        "missing projects: {}; missing attachments: {}",
        stats.curated_documents,
        stats.unfinished_documents,
        stats.documents_with_hf_spans,
        stats.matched_documents,
        stats.missing_projects,
        stats.missing_attachments,
    )
    if restrictions is not None:
        logger.info(
            "Excluded {} restricted projects ({} documents) listed in {}",
            len(restricted_seen),
            stats.restricted_documents,
            restrictions,
        )
    logger.info(
        "Mapped annotations: {}; unmapped annotations: {}; "
        "resolved areas absent from query map: {}; positives: {}; "
        "negatives: {}; overlap-discarded passages: {}; "
        "unlocated passages kept as negatives: {}",
        stats.mapped_annotations,
        stats.unmapped_annotations,
        stats.missing_query_annotations,
        stats.positive_spans,
        stats.negative_passages,
        stats.overlapping_passages,
        stats.unlocated_passages,
    )
    if empty_negatives:
        logger.info(
            "Empty finished documents: {} added as negatives ({} passages); {} skipped in projects without positives",
            stats.empty_documents,
            stats.empty_negative_passages,
            stats.empty_documents_without_positives,
        )
    logger.info(
        "Span merge (tolerance={} chars): {} spans -> {} spans ({} merge events)",
        SPAN_MERGE_TOLERANCE,
        stats.spans_before_merge,
        stats.spans_after_merge,
        stats.merged_span_clusters,
    )
    logger.info(
        f"Wrote {len(corpora)} project shards with {total_queries} query entries, "
        f"{total_corpus} corpus items, and {total_qrels} qrels to {output}"
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build project-scoped MTEB datasets from curated INCEpTION spans")
    parser.add_argument(
        "--export",
        required=True,
        type=Path,
        help="INCEpTION project export zip or unzipped directory",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        type=Path,
        help="Directory containing simap_<id>.json.zst pipeline outputs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for one anonymized BEIR shard per project (corpus.jsonl, queries.jsonl, qrels/test.tsv)",
    )
    parser.add_argument(
        "--project-mapping",
        type=Path,
        default=DEFAULT_PROJECT_MAPPING,
        help=f"Private SIMAP ID-to-alias JSON written by the build (default: {DEFAULT_PROJECT_MAPPING})",
    )
    parser.add_argument(
        "--alias-key",
        type=Path,
        default=DEFAULT_ALIAS_KEY,
        help=f"Secret HMAC key for project aliases, created on first use; keep it (default: {DEFAULT_ALIAS_KEY})",
    )
    parser.add_argument(
        "--query-map",
        type=Path,
        default=DEFAULT_QUERY_MAP,
        help=(
            "JSONL map of query id to text "
            f"(default: {DEFAULT_QUERY_MAP}). Generate with "
            "scripts/queries/generate_query_maps.py"
        ),
    )
    parser.add_argument(
        "--hub-repo",
        help="Optional Hugging Face dataset repository to upload MTEB configs to",
    )
    parser.add_argument(
        "--hub-private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make a newly created Hugging Face repository private (default: true)",
    )
    parser.add_argument(
        "--exclude-restricted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop projects whose tender documents restrict reuse to offer preparation (default: true)",
    )
    parser.add_argument(
        "--restrictions",
        type=Path,
        default=DEFAULT_RESTRICTIONS,
        help=(
            f"List of restricted simap_<id> projects (default: {DEFAULT_RESTRICTIONS}). "
            "Generate with scripts/test_set/find_restricted_projects.py"
        ),
    )
    parser.add_argument(
        "--finished-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only use documents whose curation is marked finished (default: true)",
    )
    parser.add_argument(
        "--empty-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add all passages of finished documents without curated spans, which every finished "
            "annotator also left empty, as negatives (default: false; such documents were often not read)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_dataset(
        args.export,
        args.pipeline,
        args.output,
        args.project_mapping,
        args.hub_repo,
        args.hub_private,
        args.query_map,
        args.restrictions if args.exclude_restricted else None,
        args.finished_only,
        args.empty_negatives,
        args.alias_key,
    )


if __name__ == "__main__":
    main()
