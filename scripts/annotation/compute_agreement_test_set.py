"""
compute_agreement_swissspark.py – Document-level IAA (α, κ) and concept γ from INCEpTION

Two-stage pipeline so the (large) export is parsed only once:

  1. extract  Read an INCEpTION export .zip (or unzipped tree) document-by-document
              into a temporary directory, write annotations.csv + coverage.csv
  2. iaa      Load those CSVs, compute pairwise document-level α/κ and concept γ
  3. all      extract then iaa

Annotators vary by document, so IAA is computed per annotator pair on the documents
that pair both annotated. Headline project and document IAA are concept-unit-weighted
means of pairwise α/κ; concept γ is a sofa-length-weighted mean of pairwise γ.
All are also reported per sector.

Document IAA: unit = (document, KB identifier), label = that identifier was
annotated anywhere in the PDF. Project IAA is the same at tender grain: unit =
(tender, KB identifier), label = that identifier was annotated anywhere in the
tender. Both use Krippendorff's α (nominal) and Cohen's κ. Concept IAA (γ): one
unit per annotation, category = KB identifier, and position still matters.
Overlap identifier rate: among unique spans that overlap the other annotator by at
least one character, the fraction whose identifier sets intersect. This ignores
one-sided marks; it is not chance-corrected. Concept γ-cat is label agreement after
positional alignment and is unstable on long continua with few units.

Who counts as having annotated a document
----------------------------------------
INCEpTION writes a CAS as soon as a document is assigned or opened, so coverage.csv
contains many empty files from people who were not actually annotating that document.

``data/exports/inception_metadata.json`` records, per document, the authoritative
list of annotators (its ``annotators`` key) alongside the WDG catalog metadata
(its ``catalogs`` key). A coder is only kept for a document if they are listed
there — this is no longer inferred from span presence. For those coders:

  * a document with 0 spans is a true void (inspected, nothing to mark) and is kept
  * pairwise overlap therefore includes those empty documents

A coder with a CAS for a document but not listed as one of its annotators is
dropped before IAA. Test files (excel_to_pdf_test.pdf) are excluded. Documents
missing from the JSON keep all their coverage rows (no annotators list to
filter against).

inception_metadata.json's annotators list is itself only an *assignment*
record (from the curation spreadsheet) and can be stale: a coder can be listed
there, and have opened every document in a sector in INCEpTION (state
IN_PROGRESS), without ever placing a span or finishing anything. When the
export's own exportedproject.json is available, its per-document FINISHED
state is cross-checked on top: a listed coder is still dropped from a sector
unless they reached FINISHED on at least one document in it. This check
applies independently of the metadata (including to documents missing from
it) — see load_annotation_states / sector_finishers.

Sectors also come from ``data/exports/inception_metadata.json`` (the document's
product WDG catalog; ``WDG_1-Product_Independent_Criteria`` is never a
document-level sector). Documents missing from the JSON are mapped via the
same ``simap|<id>`` project, then by filename.

With ``--product-independent``, generic ``generic-*`` identifiers are scored
separately **inside each product sector**, using that sector's annotators only.
There is no overall PIC continuum; concept-unit-weighted means of per-sector
PIC pairwise project and document α/κ, and a length-weighted mean of PIC concept γ,
are the aggregates.

Usage:
    uv run scripts/annotation/compute_agreement_test_set.py extract \\
        --export ~/Downloads/project-....zip --output data/swissspark_iaa

    uv run scripts/annotation/compute_agreement_test_set.py iaa \\
        --input data/swissspark_iaa --output data/swissspark_iaa

    uv run scripts/annotation/compute_agreement_test_set.py all \\
        --export ~/Downloads/project-....zip --output data/swissspark_iaa
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, BinaryIO

import krippendorff
import pandas as pd
from datasets import load_dataset
from loguru import logger
from pyannote.core import Segment
from pygamma_agreement import CombinedCategoricalDissimilarity, Continuum
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score

from tendril.paths import REPOSITORY_PATH

NE_TYPE_LOCAL = "NamedEntity"
DOC_META_LOCAL = "DocumentMetaData"
SPAN_CATEGORY = "span"

ANNOTATIONS_CSV = "annotations.csv"
COVERAGE_CSV = "coverage.csv"
STATE_CSV = "annotation_state.csv"
PAIRWISE_CSV = "iaa_pairwise.csv"
SECTOR_PAIRWISE_CSV = "iaa_by_sector.csv"
SUMMARY_TXT = "iaa_summary.txt"

DEFAULT_SKIP_CODERS = ("CURATION_USER", "admin")
# INCEpTION always ships the empty initial CAS; never treat it as an annotator.
ALWAYS_SKIP_CODERS = {"INITIAL_CAS"}
PROGRESS_EVERY = 25

# Date-prefixed food tenders without a sector token in the filename.
FOOD_PROJECT_IDS = frozenset({"206294", "257020", "257417"})
TEST_SECTOR = "TEST"
GENERIC_CATALOG = "WDG_1-Product_Independent_Criteria"
PIC_LAYER = "pic"
PRODUCT_LAYER = "product"
GENERIC_IDENTIFIER = re.compile(r"#generic-", re.I)
WDG_PREFIX = re.compile(r"^WDG_\d+-")
PREFIX_BEFORE_DATE = re.compile(r"^(.+?)__\d{4}-\d{2}-\d{2}__(?:simap__)?(\d+)__")
DATE_THEN_PROJECT = re.compile(r"^\d{4}-\d{2}-\d{2}__(?:simap__)?(\d+)__")
STARTS_WITH_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}__")
DEFAULT_METADATA = (
    REPOSITORY_PATH / "data" / "exports" / "inception_metadata.json"
)

# Raw INCEpTION identifiers are URIs ending in a fragment like
# "food-HF-003" (an Area of Action) or "food-K-019" (one of its finer-grained
# criteria). Annotators are only supposed to tag AoAs, but some CASes still
# carry criterion-level (K-<n>) tags; CRITERION_DATASET_NAME lets us collapse
# those back to their parent AoA instead of scoring them as distinct labels.
CRITERION_FRAGMENT = re.compile(r"^(?P<catalog>.+)-(?P<kind>HF|K)-(?P<number>\d+)$", re.IGNORECASE)
CATALOG_VERSION_SUFFIX = re.compile(r"_V\d+(?:\.\d+)*$", re.IGNORECASE)
CRITERION_DATASET_NAME = "IntelliProcure/SwissSPARK_Catalogs"

# Older annotation-tool catalog tokens that don't already match a canonical
# SwissSPARK_Catalogs catalog name (see scripts/test_set/build_dataset.py).
IDENTIFIER_CATALOG_ALIASES = {
    "building_construction": "Construction_Buildings",
    "food": "Food",
    "furniture": "Furniture",
    "generic": "Product_Independent_Criteria",
    "lighting_electrical_appliances": "Light_Elect_Appl",
    "print_services": "Print_Services",
    "transportation": "Road_Transport",
}
CANONICAL_CATALOG_NAMES = frozenset(
    {
        "Construction_Area",
        "Construction_Buildings",
        "Construction_Civil_Engineering_Infrastructure",
        "Electricity",
        "Food",
        "Furniture",
        "Green_Spaces",
        "IT_Software",
        "Info_Com_Tech",
        "Light_Elect_Appl",
        "Paper_and_Stationery",
        "Print_Services",
        "Product_Independent_Criteria",
        "Road_Transport",
        "Textiles",
        "Wastepaper_Cleaning_Refuse",
    }
)
CANONICAL_CATALOG_CASEFOLD = {name.casefold(): name for name in CANONICAL_CATALOG_NAMES}
CANONICAL_GENERIC_CATALOG = "Product_Independent_Criteria"


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def resolve_annotation_root(export: Path) -> Path:
    if (export / "annotation").is_dir():
        return export / "annotation"
    if export.is_dir() and export.name == "annotation":
        return export
    raise SystemExit(
        f"ERROR: no annotation/ folder under {export}. "
        "Pass the INCEpTION project export .zip, the unzipped export, or its annotation/ directory."
    )


def skip_set(skip_coders: set[str]) -> set[str]:
    return skip_coders | ALWAYS_SKIP_CODERS


def annotation_dir_prefix(names: Sequence[str]) -> str:
    for name in names:
        parts = [part for part in name.split("/") if part]
        if "annotation" in parts:
            index = parts.index("annotation")
            return "/".join(parts[: index + 1]) + "/"
    raise SystemExit("ERROR: export zip has no annotation/ folder")


def members_by_document(names: Sequence[str], prefix: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        rest = name[len(prefix) :]
        document = rest.split("/", 1)[0]
        if document:
            grouped[document].append(name)
    return grouped


def _is_inside(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: zipfile.ZipFile, members: Sequence[str], dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for member in members:
        target = (dest / member).resolve()
        if not _is_inside(dest, target) and target != dest:
            raise SystemExit(f"ERROR: refused path traversal in zip member {member!r}")
    archive.extractall(dest, members=members)


def unzip_nested_cas_archives(doc_dir: Path, skip_coders: set[str]) -> None:
    skip = skip_set(skip_coders)
    for cas_zip in list(doc_dir.glob("*.zip")):
        coder = cas_zip.stem
        if coder in skip:
            cas_zip.unlink(missing_ok=True)
            continue
        dest = doc_dir / coder
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(cas_zip) as inner:
                safe_extract(inner, inner.namelist(), dest)
        except zipfile.BadZipFile as exc:
            logger.warning(f"Skipping bad nested zip {cas_zip}: {exc}")
        cas_zip.unlink(missing_ok=True)


def iter_document_cas_files(doc_dir: Path, skip_coders: set[str]) -> Iterator[tuple[str, Path]]:
    skip = skip_set(skip_coders)
    seen: set[str] = set()
    for xmi in sorted(doc_dir.glob("*/*.xmi")):
        coder = xmi.parent.name
        if coder in skip:
            continue
        seen.add(coder)
        yield coder, xmi
    for cas_zip in sorted(doc_dir.glob("*.zip")):
        coder = cas_zip.stem
        if coder in skip or coder in seen:
            continue
        yield coder, cas_zip


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_xmi_stream(source: Path | BinaryIO, label: str) -> tuple[int, list[tuple[int, int, str]]]:
    """Stream-parse an XMI file. Returns (sofa_length, [(begin, end, identifier), ...])."""
    sofa_length = 0
    units: list[tuple[int, int, str]] = []
    try:
        for _, elem in ET.iterparse(source, events=("end",)):
            name = local_tag(elem.tag)
            if name == DOC_META_LOCAL:
                try:
                    sofa_length = int(elem.attrib.get("end", 0))
                except ValueError:
                    pass
            elif name == NE_TYPE_LOCAL:
                try:
                    begin = int(elem.attrib["begin"])
                    end = int(elem.attrib["end"])
                except (KeyError, ValueError):
                    elem.clear()
                    continue
                identifier = (elem.attrib.get("identifier") or "").strip()
                units.append((begin, end, identifier))
            elem.clear()
    except ET.ParseError as exc:
        logger.warning(f"Skipping unreadable XMI {label}: {exc}")
        return 0, []
    if sofa_length <= 0:
        sofa_length = max((end for _, end, _ in units), default=0)
    return sofa_length, units


def parse_cas(path: Path) -> tuple[int, list[tuple[int, int, str]]]:
    if path.suffix.lower() != ".zip":
        return parse_xmi_stream(path, label=str(path))
    try:
        with zipfile.ZipFile(path) as zf:
            xmi_names = [name for name in zf.namelist() if name.endswith(".xmi")]
            if not xmi_names:
                logger.warning(f"No .xmi inside {path}")
                return 0, []
            with zf.open(xmi_names[0]) as handle:
                return parse_xmi_stream(handle, label=str(path))
    except zipfile.BadZipFile as exc:
        logger.warning(f"Skipping bad zip {path}: {exc}")
        return 0, []


def write_cas_rows(
    ann_w: csv.writer,
    cov_w: csv.writer,
    document: str,
    coder: str,
    sofa_length: int,
    units: Sequence[tuple[int, int, str]],
) -> int:
    cov_w.writerow([document, coder, sofa_length])
    n_kept = 0
    for begin, end, identifier in units:
        if end <= begin:
            continue
        ann_w.writerow([document, coder, begin, end, identifier])
        n_kept += 1
    return n_kept


def extract_document_dir(
    doc_dir: Path,
    document: str,
    skip_coders: set[str],
    ann_w: csv.writer,
    cov_w: csv.writer,
    *,
    unzip_nested: bool,
) -> tuple[int, int]:
    if unzip_nested:
        unzip_nested_cas_archives(doc_dir, skip_coders)
    n_cas = 0
    n_units = 0
    for coder, cas_path in iter_document_cas_files(doc_dir, skip_coders):
        sofa_length, units = parse_cas(cas_path)
        n_units += write_cas_rows(ann_w, cov_w, document, coder, sofa_length, units)
        n_cas += 1
    return n_cas, n_units


def log_progress(done: int, total: int, extra: str = "") -> None:
    if total <= 0:
        return
    if done in {1, total} or done % PROGRESS_EVERY == 0:
        message = f"Progress {done}/{total} ({100 * done / total:.0f}%)"
        if extra:
            message = f"{message}, {extra}"
        logger.info(message)


def extract_from_zip(
    zip_path: Path,
    skip_coders: set[str],
    ann_w: csv.writer,
    cov_w: csv.writer,
    ann_f,
    cov_f,
) -> tuple[int, int]:
    logger.info(f"Reading INCEpTION export zip {zip_path}")
    n_cas = 0
    n_units = 0
    with tempfile.TemporaryDirectory(prefix="swissspark_export_") as tmp:
        tmp_path = Path(tmp)
        logger.info(f"Unpacking documents into temporary directory {tmp_path}")
        with zipfile.ZipFile(zip_path) as zf:
            prefix = annotation_dir_prefix(zf.namelist())
            by_document = members_by_document(zf.namelist(), prefix)
            documents = sorted(by_document)
            if not documents:
                raise SystemExit(f"ERROR: no documents under {prefix} in {zip_path}")
            logger.info(f"Found {len(documents)} documents in {prefix}")
            for done, document in enumerate(documents, start=1):
                safe_extract(zf, by_document[document], tmp_path)
                doc_dir = tmp_path.joinpath(*prefix.rstrip("/").split("/"), document)
                try:
                    doc_cas, doc_units = extract_document_dir(
                        doc_dir,
                        document,
                        skip_coders,
                        ann_w,
                        cov_w,
                        unzip_nested=True,
                    )
                finally:
                    shutil.rmtree(doc_dir, ignore_errors=True)
                n_cas += doc_cas
                n_units += doc_units
                logger.debug(f"{document}: {doc_cas} annotators, {doc_units} spans")
                log_progress(
                    done,
                    len(documents),
                    extra=f"{n_cas:,} CAS, {n_units:,} spans",
                )
                ann_f.flush()
                cov_f.flush()
    return n_cas, n_units


def extract_from_dir(
    export: Path,
    skip_coders: set[str],
    ann_w: csv.writer,
    cov_w: csv.writer,
    ann_f,
    cov_f,
) -> tuple[int, int]:
    annotation_root = resolve_annotation_root(export)
    doc_dirs = sorted(path for path in annotation_root.iterdir() if path.is_dir())
    if not doc_dirs:
        raise SystemExit(f"ERROR: no document folders in {annotation_root}")
    logger.info(f"Reading unzipped export {annotation_root} ({len(doc_dirs)} documents)")
    n_cas = 0
    n_units = 0
    for done, doc_dir in enumerate(doc_dirs, start=1):
        doc_cas, doc_units = extract_document_dir(
            doc_dir,
            doc_dir.name,
            skip_coders,
            ann_w,
            cov_w,
            unzip_nested=False,
        )
        n_cas += doc_cas
        n_units += doc_units
        logger.debug(f"{doc_dir.name}: {doc_cas} annotators, {doc_units} spans")
        log_progress(
            done,
            len(doc_dirs),
            extra=f"{n_cas:,} CAS, {n_units:,} spans",
        )
        ann_f.flush()
        cov_f.flush()
    return n_cas, n_units


def load_annotation_states(export: Path, skip_coders: set[str]) -> dict[tuple[str, str], str]:
    """Best-effort (document, coder) -> INCEpTION annotation state ("NEW",
    "IN_PROGRESS", "FINISHED"), read from exportedproject.json's
    annotation_documents table if it's present alongside the export.

    This is a second, INCEpTION-native signal on top of
    inception_metadata.json's annotators list (see
    filter_coverage_to_sector_annotators): the metadata only reflects who was
    *assigned* to a document per the curation spreadsheet, which can be stale
    or aspirational. A coder can be listed there -- and even have opened every
    document in a sector (IN_PROGRESS) -- without ever placing a single
    annotation or finishing any of it. exportedproject.json's own FINISHED
    flag catches that case directly. Returns {} if exportedproject.json can't
    be found (older exports, or --export pointed straight at the annotation/
    folder) -- this check is additive, extraction must not fail without it.
    """
    skip = skip_set(skip_coders)
    payload: dict | None = None
    if export.is_file() and export.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(export) as zf:
                candidates = sorted(
                    (name for name in zf.namelist() if name.endswith("exportedproject.json")),
                    key=len,
                )
                if candidates:
                    with zf.open(candidates[0]) as handle:
                        payload = json.load(handle)
        except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read exportedproject.json from {export}: {exc}")
    else:
        root = export.parent if export.is_dir() and export.name == "annotation" else export
        candidate = root / "exportedproject.json"
        if candidate.is_file():
            try:
                with candidate.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError as exc:
                logger.warning(f"Could not parse {candidate}: {exc}")
    if payload is None:
        logger.info(
            "No exportedproject.json found next to the export; skipping the "
            "INCEpTION FINISHED-state annotator cross-check"
        )
        return {}
    states: dict[tuple[str, str], str] = {}
    for row in payload.get("annotation_documents", []):
        name, user, state = row.get("name"), row.get("user"), row.get("state")
        if name and user and state and user not in skip:
            states[(name, user)] = state
    logger.info(f"Loaded {len(states):,} (document, coder) annotation states from exportedproject.json")
    return states


def cmd_extract(export: Path, output: Path, skip_coders: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    annotations_path = output / ANNOTATIONS_CSV
    coverage_path = output / COVERAGE_CSV
    with annotations_path.open("w", newline="", encoding="utf-8") as ann_f, coverage_path.open(
        "w", newline="", encoding="utf-8"
    ) as cov_f:
        ann_w = csv.writer(ann_f)
        cov_w = csv.writer(cov_f)
        ann_w.writerow(["document", "coder", "begin", "end", "identifier"])
        cov_w.writerow(["document", "coder", "sofa_length"])
        if export.is_file() and export.suffix.lower() == ".zip":
            n_cas, n_units = extract_from_zip(export, skip_coders, ann_w, cov_w, ann_f, cov_f)
        elif export.is_dir():
            n_cas, n_units = extract_from_dir(export, skip_coders, ann_w, cov_w, ann_f, cov_f)
        else:
            raise SystemExit(
                f"ERROR: --export must be an INCEpTION .zip or an unzipped directory, got {export}"
            )

    if n_cas == 0:
        raise SystemExit("ERROR: no annotator CAS files found (after skipping INITIAL_CAS/admin)")
    logger.info(f"Wrote {n_units:,} spans from {n_cas:,} CAS files → {annotations_path}")
    logger.info(f"Wrote coverage for {n_cas:,} document×coder pairs → {coverage_path}")

    states = load_annotation_states(export, skip_coders)
    if states:
        state_path = output / STATE_CSV
        with state_path.open("w", newline="", encoding="utf-8") as state_f:
            state_w = csv.writer(state_f)
            state_w.writerow(["document", "coder", "state"])
            for (document, coder), state in sorted(states.items()):
                state_w.writerow([document, coder, state])
        logger.info(f"Wrote {len(states):,} annotation states → {state_path}")


# ---------------------------------------------------------------------------
# IAA
# ---------------------------------------------------------------------------


def catalog_to_sector(catalog: str) -> str:
    return WDG_PREFIX.sub("", catalog)


def project_id_from_filename(name: str) -> str | None:
    prefix_match = PREFIX_BEFORE_DATE.match(name)
    if prefix_match and not STARTS_WITH_DATE.match(name):
        return prefix_match.group(2)
    date_match = DATE_THEN_PROJECT.match(name)
    if date_match:
        return date_match.group(1)
    return None


def filename_sector(name: str) -> str:
    """Fallback when a document is not in INCEpTION metadata."""
    if name == "excel_to_pdf_test.pdf":
        return TEST_SECTOR
    if "Harmonisiert" in name:
        return "Food"
    if "FURNITURE" in name.upper():
        return "Furniture"
    prefix_match = PREFIX_BEFORE_DATE.match(name)
    if prefix_match and not STARTS_WITH_DATE.match(name):
        return prefix_match.group(1)
    date_match = DATE_THEN_PROJECT.match(name)
    if date_match:
        project_id = date_match.group(1)
        lowered = name.lower()
        if (
            project_id in FOOD_PROJECT_IDS
            or "milch" in lowered
            or "mueesli" in lowered
            or "vollkorn" in lowered
        ):
            return "Food"
        return f"project:{project_id}"
    return "UNKNOWN"


def canonical_fragment_id(kind: str, number: str) -> str:
    return f"{kind.upper()}-{int(number):03d}"


def resolve_identifier_catalog(raw_catalog: str) -> str | None:
    stripped = CATALOG_VERSION_SUFFIX.sub("", raw_catalog)
    alias = IDENTIFIER_CATALOG_ALIASES.get(stripped.casefold(), stripped)
    return CANONICAL_CATALOG_CASEFOLD.get(alias.casefold())


def parse_identifier_fragment(identifier: str) -> tuple[str, str, str] | None:
    """Parse a raw INCEpTION identifier URI into (canonical_catalog, kind, canonical_id).

    kind is "HF" (Area of Action) or "K" (criterion). Returns None if the
    identifier doesn't match the expected fragment shape or its catalog token
    can't be resolved to a known SwissSPARK_Catalogs catalog.
    """
    if not identifier:
        return None
    fragment = identifier.rsplit("#", 1)[-1]
    match = CRITERION_FRAGMENT.fullmatch(fragment)
    if not match:
        return None
    catalog = resolve_identifier_catalog(match.group("catalog"))
    if not catalog:
        return None
    kind = match.group("kind").upper()
    return catalog, kind, canonical_fragment_id(kind, match.group("number"))


def load_criterion_to_area_map() -> dict[tuple[str, str], str]:
    """(catalog, canonical criterion id "K-NNN") -> canonical Area-of-Action id "HF-NNN"."""
    logger.info(f"Loading criterion→area mapping from {CRITERION_DATASET_NAME}")
    rows = load_dataset(CRITERION_DATASET_NAME, split="train")
    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        catalog = str(row["catalog"]).strip()
        criterion_match = re.fullmatch(
            r"K-0*(\d+)", str(row["sustainable_procurement_criterion_id"]).strip(), re.IGNORECASE
        )
        area_match = re.fullmatch(
            r"HF-0*(\d+)", str(row["area_of_action_id"]).strip(), re.IGNORECASE
        )
        if not catalog or not criterion_match or not area_match:
            continue
        key = (catalog, canonical_fragment_id("K", criterion_match.group(1)))
        mapping[key] = canonical_fragment_id("HF", area_match.group(1))
    logger.info(f"Loaded {len(mapping):,} criterion→area mappings")
    return mapping


def canonicalize_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw INCEpTION identifiers to canonical "Catalog::HF-NNN" form.

    HF-<n> (Area of Action) identifiers are just normalized (fixes catalog
    version suffixes, aliasing, casing). K-<n> (criterion) identifiers are
    mapped to their parent Area of Action via the SwissSPARK_Catalogs
    criterion table, since annotators are only supposed to tag AoAs and
    scoring criterion-level tags as if they were distinct labels manufactures
    disagreement between coders who picked different criteria under the same
    AoA. Identifiers that can't be parsed or resolved are left unchanged.
    """
    parsed = annotations["identifier"].map(parse_identifier_fragment)
    n_k = int(parsed.map(lambda info: info is not None and info[1] == "K").sum())
    n_hf = int(parsed.map(lambda info: info is not None and info[1] == "HF").sum())
    criterion_to_area = load_criterion_to_area_map() if n_k else {}

    def resolve(identifier: str, info: tuple[str, str, str] | None) -> str:
        if info is None:
            return identifier
        catalog, kind, canonical_id = info
        if kind == "HF":
            return f"{catalog}::{canonical_id}"
        area_id = criterion_to_area.get((catalog, canonical_id))
        return f"{catalog}::{area_id}" if area_id else identifier

    out = annotations.copy()
    out["identifier"] = [
        resolve(identifier, info) for identifier, info in zip(annotations["identifier"], parsed)
    ]
    n_k_resolved = int(
        sum(
            1
            for identifier, info in zip(annotations["identifier"], parsed)
            if info is not None
            and info[1] == "K"
            and criterion_to_area.get((info[0], info[2])) is not None
        )
    )
    logger.info(
        f"Canonicalized identifiers: {n_hf:,} Area-of-Action identifiers normalized, "
        f"{n_k:,} criterion-level (K-<n>) identifiers found "
        f"({n_k_resolved:,} mapped to their parent Area of Action, "
        f"{n_k - n_k_resolved:,} left unresolved)"
    )
    return out


def is_generic_identifier(identifier: str) -> bool:
    if (identifier or "").casefold().startswith(f"{CANONICAL_GENERIC_CATALOG.casefold()}::"):
        return True
    return bool(GENERIC_IDENTIFIER.search(identifier or ""))


def product_catalog(entries: Sequence[dict]) -> str | None:
    for entry in entries:
        catalog = entry.get("catalog")
        if catalog and catalog != GENERIC_CATALOG:
            return catalog_to_sector(str(catalog))
    return None


def numeric_project_id(project_id: object | None) -> str | None:
    if not project_id:
        return None
    numeric = str(project_id).split("|")[-1].strip()
    return numeric or None


def product_project_id(entries: Sequence[dict]) -> str | None:
    for entry in entries:
        catalog = entry.get("catalog")
        if catalog and catalog != GENERIC_CATALOG:
            project_id = numeric_project_id(entry.get("project_id"))
            if project_id:
                return project_id
    for entry in entries:
        project_id = numeric_project_id(entry.get("project_id"))
        if project_id:
            return project_id
    return None


def _project_keys(project_id: str) -> list[str]:
    if not project_id:
        return []
    keys = [project_id]
    numeric = project_id.split("|")[-1]
    if numeric and numeric != project_id:
        keys.append(numeric)
    return keys


class SectorMap:
    """Document → sector and → annotators from inception_metadata.json, with a
    filename fallback for sector only (a missing document has no annotators list
    to fall back to)."""

    def __init__(self, metadata: dict[str, dict]) -> None:
        self.by_document: dict[str, str] = {}
        self.by_project: dict[str, str] = {}
        self.project_by_document: dict[str, str] = {}
        self.annotators_by_document: dict[str, frozenset[str]] = {}
        for name, doc_meta in metadata.items():
            entries = doc_meta.get("catalogs", [])
            annotators = doc_meta.get("annotators", [])
            self.annotators_by_document[name] = frozenset(str(a) for a in annotators)
            project_id = product_project_id(entries)
            if project_id:
                self.project_by_document[name] = project_id
            sector = product_catalog(entries)
            if not sector:
                continue
            self.by_document[name] = sector
            for entry in entries:
                for key in _project_keys(str(entry.get("project_id") or "")):
                    self.by_project[key] = sector

    def sector_for(self, name: str) -> str:
        if name in self.by_document:
            return self.by_document[name]
        project_id = project_id_from_filename(name)
        if project_id and project_id in self.by_project:
            return self.by_project[project_id]
        return filename_sector(name)

    def project_for(self, name: str) -> str | None:
        if name in self.project_by_document:
            return self.project_by_document[name]
        return project_id_from_filename(name)

    def source_for(self, name: str) -> str:
        if name in self.by_document:
            return "metadata"
        project_id = project_id_from_filename(name)
        if project_id and project_id in self.by_project:
            return "metadata-project"
        return "filename"

    def annotators_for(self, name: str) -> frozenset[str] | None:
        """Annotators listed for this document in the metadata, or None if the
        document is not in the metadata at all."""
        return self.annotators_by_document.get(name)


def load_sector_map(path: Path) -> SectorMap:
    if not path.is_file():
        raise SystemExit(f"ERROR: metadata file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise SystemExit(f"ERROR: {path} must be a JSON object mapping document names to catalogs")
    sector_map = SectorMap(metadata)
    logger.info(
        f"Loaded sector metadata from {path} "
        f"({len(sector_map.by_document)} documents with a sector, "
        f"{len(sector_map.by_project)} project keys, "
        f"{len(sector_map.annotators_by_document)} documents with annotators)"
    )
    return sector_map


def document_sector(name: str, sector_map: SectorMap | None = None) -> str:
    if sector_map is None:
        return filename_sector(name)
    return sector_map.sector_for(name)


def sector_finishers(states: pd.DataFrame, sector_map: SectorMap | None) -> dict[str, set[str]]:
    """coder -> sectors where that coder reached INCEpTION's own FINISHED state
    on at least one document (see load_annotation_states). Used as a second,
    independent check in filter_coverage_to_sector_annotators, on top of the
    (possibly stale) inception_metadata.json annotators list."""
    finished = states[states["state"] == "FINISHED"]
    out: dict[str, set[str]] = defaultdict(set)
    for row in finished.itertuples(index=False):
        out[str(row.coder)].add(document_sector(str(row.document), sector_map))
    return out


def filter_coverage_to_sector_annotators(
    coverage: pd.DataFrame,
    annotations: pd.DataFrame,
    sector_map: SectorMap | None = None,
    states: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep coverage rows only for coders listed as that document's annotators in
    inception_metadata.json (its ``annotators`` key). This is authoritative, not
    inferred from span presence.

    A coder can be listed there and still not be a real sector annotator:
    inception_metadata.json's annotators list only reflects who was *assigned*
    to a document (per the curation spreadsheet), and that assignment can be
    stale or aspirational -- a coder can be listed, and even have opened every
    document in a sector (INCEpTION state IN_PROGRESS), without ever placing a
    single span or finishing any of it (this is exactly how an admin/QA
    account with broad but shallow access showed up as a "co-annotator" with
    zero real overlap in several sectors, even after the metadata was
    complete). When ``states`` is given (see load_annotation_states), we
    additionally require at least one FINISHED document somewhere in the
    sector before trusting the metadata's assignment for that coder -- this
    check is independent of inception_metadata.json and applies even to
    documents missing from it.

    Returns (kept, dropped): dropped holds the non-test-sector coverage rows for
    coders with a CAS in INCEpTION for that document who are not listed as one of
    its annotators, or who are listed but never finished anything in that sector.
    Documents missing from the metadata keep all their coverage rows (modulo the
    FINISHED-state check above).
    """
    tagged_coverage = coverage.copy()
    tagged_coverage["sector"] = tagged_coverage["document"].map(
        lambda name: document_sector(name, sector_map)
    )
    tagged_coverage["project"] = tagged_coverage["document"].map(
        lambda name: (
            sector_map.project_for(name) if sector_map else project_id_from_filename(name)
        )
    )

    n_before = len(tagged_coverage)
    tagged_coverage = tagged_coverage[tagged_coverage["sector"] != TEST_SECTOR]

    finishers = (
        sector_finishers(states, sector_map) if states is not None and not states.empty else None
    )
    coders_with_state = set(states["coder"].astype(str)) if states is not None else set()

    def is_annotator(row: pd.Series) -> bool:
        if sector_map is not None:
            annotators = sector_map.annotators_for(str(row["document"]))
            if annotators is not None and str(row["coder"]) not in annotators:
                return False
        if finishers is not None and str(row["coder"]) in coders_with_state:
            if row["sector"] not in finishers.get(str(row["coder"]), set()):
                return False
        return True

    if tagged_coverage.empty:
        mask = pd.Series([], dtype=bool)
    else:
        mask = tagged_coverage.apply(is_annotator, axis=1)
    filtered = tagged_coverage[mask]
    dropped = tagged_coverage[~mask]

    span_counts = (
        annotations.groupby(["document", "coder"]).size().rename("n_spans").reset_index()
    )
    with_counts = filtered.merge(span_counts, on=["document", "coder"], how="left")
    n_empty_kept = int(with_counts["n_spans"].fillna(0).eq(0).sum())
    n_dropped = n_before - len(filtered)
    sources = Counter(
        sector_map.source_for(doc) if sector_map else "filename"
        for doc in filtered["document"].unique()
    )
    source_bits = ", ".join(f"{key}={sources[key]}" for key in sorted(sources))
    finished_bit = "; FINISHED-state cross-check active" if finishers is not None else ""
    logger.info(
        f"Metadata-annotator filter: kept {len(filtered):,} of {n_before:,} coverage rows "
        f"(dropped {n_dropped:,} non-annotator/test rows; "
        f"{n_empty_kept:,} empty CAS kept as true voids; {source_bits}{finished_bit})"
    )
    return filtered, dropped


def sector_non_annotator_summary(dropped: pd.DataFrame) -> pd.DataFrame:
    """Per (coder, sector) doc counts for coverage rows dropped by
    filter_coverage_to_sector_annotators: coders with an INCEpTION CAS for a
    document (e.g. a project marked finished) who are not listed as one of its
    annotators in inception_metadata.json."""
    if dropped.empty:
        return pd.DataFrame(columns=["coder", "sector", "n_documents"])
    grouped = (
        dropped.groupby(["coder", "sector"])["document"]
        .nunique()
        .rename("n_documents")
        .reset_index()
    )
    return grouped.sort_values(["coder", "sector"])


def load_extract(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    annotations_path = input_dir / ANNOTATIONS_CSV
    coverage_path = input_dir / COVERAGE_CSV
    if not annotations_path.is_file() or not coverage_path.is_file():
        raise SystemExit(
            f"ERROR: {input_dir} must contain {ANNOTATIONS_CSV} and {COVERAGE_CSV}. "
            "Run the extract command first."
        )
    annotations = pd.read_csv(annotations_path, dtype={"identifier": "string"})
    coverage = pd.read_csv(coverage_path)
    annotations["identifier"] = annotations["identifier"].fillna("").astype(str)
    state_path = input_dir / STATE_CSV
    states = pd.read_csv(state_path) if state_path.is_file() else None
    return annotations, coverage, states


def sofa_lengths_by_document(coverage: pd.DataFrame) -> dict[str, int]:
    return coverage.groupby("document")["sofa_length"].max().astype(int).to_dict()


def shared_documents(coverage: pd.DataFrame, coder_a: str, coder_b: str) -> list[str]:
    docs_a = set(coverage.loc[coverage["coder"] == coder_a, "document"])
    docs_b = set(coverage.loc[coverage["coder"] == coder_b, "document"])
    return sorted(docs_a & docs_b)


def document_offsets(documents: Sequence[str], lengths: dict[str, int]) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for document in documents:
        offsets[document] = cursor
        cursor += int(lengths.get(document, 0))
    return offsets, cursor


def pair_units(
    annotations: pd.DataFrame,
    coder: str,
    documents: Sequence[str],
    offsets: dict[str, int],
    *,
    dedupe: bool,
    require_identifier: bool,
) -> list[tuple[int, int, str]]:
    subset = annotations[
        (annotations["coder"] == coder) & (annotations["document"].isin(documents))
    ]
    if require_identifier:
        subset = subset[subset["identifier"].str.len() > 0]
    if dedupe:
        subset = subset.drop_duplicates(["document", "begin", "end"])
    units: list[tuple[int, int, str]] = []
    for row in subset.itertuples(index=False):
        start = int(row.begin) + offsets[row.document]
        end = int(row.end) + offsets[row.document]
        if end <= start:
            continue
        category = SPAN_CATEGORY if dedupe else row.identifier
        units.append((start, end, category))
    return units


def spans_with_identifiers(
    annotations: pd.DataFrame,
    coder: str,
    documents: Sequence[str],
) -> list[tuple[str, int, int, frozenset[str]]]:
    """Unique (document, begin, end) spans with the set of non-empty identifiers."""
    subset = annotations[
        (annotations["coder"] == coder) & (annotations["document"].isin(documents))
    ]
    grouped: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in subset.itertuples(index=False):
        key = (str(row.document), int(row.begin), int(row.end))
        grouped.setdefault(key, set())
        identifier = str(row.identifier)
        if identifier:
            grouped[key].add(identifier)
    return [(document, begin, end, frozenset(ids)) for (document, begin, end), ids in grouped.items()]


def overlap_identifier_counts(
    spans_a: Sequence[tuple[str, int, int, frozenset[str]]],
    spans_b: Sequence[tuple[str, int, int, frozenset[str]]],
) -> tuple[int, int]:
    """How many unique spans in A overlap B, and how many of those share an identifier."""
    by_document: dict[str, list[tuple[int, int, frozenset[str]]]] = defaultdict(list)
    for document, begin, end, ids in spans_b:
        by_document[document].append((begin, end, ids))
    n_overlap = 0
    n_share = 0
    for document, begin, end, ids in spans_a:
        matched = False
        shared = False
        for begin_b, end_b, ids_b in by_document.get(document, []):
            if max(begin, begin_b) < min(end, end_b):
                matched = True
                if ids & ids_b:
                    shared = True
                    break
        if matched:
            n_overlap += 1
            if shared:
                n_share += 1
    return n_overlap, n_share


def overlap_identifier_rate(
    annotations: pd.DataFrame,
    coder_a: str,
    coder_b: str,
    documents: Sequence[str],
) -> dict[str, float]:
    """Pooled both directions: overlapping unique spans whose identifier sets intersect."""
    spans_a = spans_with_identifiers(annotations, coder_a, documents)
    spans_b = spans_with_identifiers(annotations, coder_b, documents)
    n_overlap_a, n_share_a = overlap_identifier_counts(spans_a, spans_b)
    n_overlap_b, n_share_b = overlap_identifier_counts(spans_b, spans_a)
    n_overlap = n_overlap_a + n_overlap_b
    n_share = n_share_a + n_share_b
    rate = (n_share / n_overlap) if n_overlap else float("nan")
    return {
        "n_overlap_a": n_overlap_a,
        "n_overlap_share_id_a": n_share_a,
        "n_overlap_b": n_overlap_b,
        "n_overlap_share_id_b": n_share_b,
        "n_overlap": n_overlap,
        "n_overlap_share_id": n_share,
        "overlap_id_rate": rate,
    }


def binary_presence_iaa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    *,
    plural: str,
    key: str,
) -> dict[str, float]:
    n_marked_a = int(sum(labels_a))
    n_marked_b = int(sum(labels_b))
    n_agree = int(sum(int(left == right) for left, right in zip(labels_a, labels_b)))
    n_both_marked = int(
        sum(int(left == 1 and right == 1) for left, right in zip(labels_a, labels_b))
    )
    n_both_empty = int(
        sum(int(left == 0 and right == 0) for left, right in zip(labels_a, labels_b))
    )
    if labels_a and labels_a == labels_b:
        alpha = 1.0
        kappa = 1.0
    else:
        try:
            alpha = float(
                krippendorff.alpha(
                    reliability_data=[labels_a, labels_b],
                    level_of_measurement="nominal",
                )
            )
        except ValueError:
            alpha = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            kappa = float(cohen_kappa_score(labels_a, labels_b, labels=[0, 1]))
    return {
        f"n_{plural}_marked_a": n_marked_a,
        f"n_{plural}_marked_b": n_marked_b,
        f"n_{plural}_agree": n_agree,
        f"n_{plural}_both_marked": n_both_marked,
        f"n_{plural}_both_empty": n_both_empty,
        f"krippendorff_alpha_{key}": alpha,
        f"cohen_kappa_{key}": kappa,
    }


def empty_presence_iaa(*, plural: str, key: str) -> dict[str, float]:
    return {
        f"n_{plural}_marked_a": 0,
        f"n_{plural}_marked_b": 0,
        f"n_{plural}_agree": 0,
        f"n_{plural}_both_marked": 0,
        f"n_{plural}_both_empty": 0,
        f"krippendorff_alpha_{key}": float("nan"),
        f"cohen_kappa_{key}": float("nan"),
    }


def identifiers_by_document(
    annotations: pd.DataFrame,
    coder: str,
    documents: Sequence[str],
) -> dict[str, set[str]]:
    """Non-empty KB identifiers used by the coder on each shared document."""
    subset = annotations[
        (annotations["coder"] == coder)
        & (annotations["document"].isin(documents))
        & (annotations["identifier"].str.len() > 0)
    ]
    by_document: dict[str, set[str]] = defaultdict(set)
    for row in subset.itertuples(index=False):
        by_document[str(row.document)].add(str(row.identifier))
    return by_document


def concept_presence_iaa(
    units: Sequence[str],
    ids_a: dict[str, set[str]],
    ids_b: dict[str, set[str]],
    *,
    count_key: str,
    plural: str,
    key: str,
) -> dict[str, float]:
    labels_a: list[int] = []
    labels_b: list[int] = []
    for unit in units:
        union = ids_a.get(unit, set()) | ids_b.get(unit, set())
        for identifier in sorted(union):
            labels_a.append(1 if identifier in ids_a.get(unit, set()) else 0)
            labels_b.append(1 if identifier in ids_b.get(unit, set()) else 0)
    out: dict[str, float | int] = {count_key: len(labels_a)}
    if not labels_a:
        out.update(empty_presence_iaa(plural=plural, key=key))
        return out
    out.update(binary_presence_iaa(labels_a, labels_b, plural=plural, key=key))
    return out


# --- MASI-based alpha (multi-label, set-valued units) --------------------
#
# krippendorff_alpha_doc/project above decomposes a document's AoA *set* into
# one binary presence/absence item per identifier in the union of what either
# coder used, then runs plain nominal alpha over that -- a pragmatic but
# nonstandard shortcut (nominal alpha assumes one value per unit; ours are
# sets). The standard distance for "coder assigns a set of labels to an item"
# is MASI (Passonneau, 2006): unit = the item itself (document or tender),
# value = the *set* of AoAs assigned, with graded partial credit for set
# overlap rather than exact-match-only. We further scope each unit to a
# single catalog (unit key = (document-or-project, catalog)) so a Food AoA and
# a Product-Independent-Criteria AoA on the same document are never compared
# as if they were interchangeable values from one flat label space -- each
# unit's value domain is exactly that catalog's own AoA pool.
CANONICAL_IDENTIFIER = re.compile(r"^(?P<catalog>[^:]+)::(?P<area>HF-\d{3})$")


def split_canonical_identifier(identifier: str) -> tuple[str, str] | None:
    match = CANONICAL_IDENTIFIER.fullmatch(identifier)
    return (match.group("catalog"), match.group("area")) if match else None


def masi_sets_by_catalog_unit(ids_by_unit: dict[str, set[str]]) -> dict[tuple[str, str], frozenset[str]]:
    """(unit, catalog) -> frozenset of AoA ids, from a {unit: {identifier, ...}}
    map (as produced by identifiers_by_document/identifiers_by_project).
    Identifiers that didn't canonicalize to "Catalog::HF-NNN" are skipped --
    MASI needs a well-defined AoA-id domain per catalog."""
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for unit, identifiers in ids_by_unit.items():
        for identifier in identifiers:
            parsed = split_canonical_identifier(identifier)
            if parsed is None:
                continue
            catalog, area = parsed
            grouped[(unit, catalog)].add(area)
    return {key: frozenset(areas) for key, areas in grouped.items()}


def masi_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Passonneau (2006) MASI: Jaccard overlap scaled by a monotonicity factor
    that rewards subset/superset relations over merely-partial overlap
    (1.0 identical, 2/3 subset/superset, 1/3 partial overlap, 0 disjoint)."""
    if not a and not b:
        return 1.0
    union = a | b
    intersection = a & b
    jaccard = len(intersection) / len(union)
    if a == b:
        monotonicity = 1.0
    elif a <= b or b <= a:
        monotonicity = 2 / 3
    elif intersection:
        monotonicity = 1 / 3
    else:
        monotonicity = 0.0
    return jaccard * monotonicity


def masi_distance(a: frozenset[str], b: frozenset[str]) -> float:
    return 1.0 - masi_similarity(a, b)


def masi_alpha(sets_a: Sequence[frozenset[str]], sets_b: Sequence[frozenset[str]]) -> float:
    """Krippendorff's alpha (1 - Do/De) using MASI distance between set-valued
    units, for exactly two coders per unit (this script only ever computes
    pairwise IAA). Do is the mean MASI distance between the two coders' sets,
    unit by unit. De is the mean MASI distance over all pairs drawn from the
    pooled 2n judgments (both coders' sets concatenated) -- the standard
    permutation-based chance baseline, using the sets actually produced rather
    than a hypothetical full label space."""
    n = len(sets_a)
    if n == 0:
        return float("nan")
    d_o = sum(masi_distance(a, b) for a, b in zip(sets_a, sets_b)) / n
    pooled = list(sets_a) + list(sets_b)
    m = len(pooled)
    pair_count = m * (m - 1) // 2
    if pair_count == 0:
        return float("nan")
    d_e = (
        sum(masi_distance(pooled[i], pooled[j]) for i in range(m) for j in range(i + 1, m))
        / pair_count
    )
    if d_e == 0:
        return 1.0 if d_o == 0 else float("nan")
    return 1.0 - d_o / d_e


def masi_presence_iaa(
    ids_a: dict[str, set[str]],
    ids_b: dict[str, set[str]],
    *,
    key: str,
) -> dict[str, float]:
    """MASI-based alpha over (unit, catalog) set-valued units -- see
    masi_alpha. ids_a/ids_b are {unit: {canonical identifier, ...}}, e.g. the
    output of identifiers_by_document/identifiers_by_project."""
    sets_a_map = masi_sets_by_catalog_unit(ids_a)
    sets_b_map = masi_sets_by_catalog_unit(ids_b)
    unit_keys = sorted(set(sets_a_map) | set(sets_b_map))
    sets_a = [sets_a_map.get(unit, frozenset()) for unit in unit_keys]
    sets_b = [sets_b_map.get(unit, frozenset()) for unit in unit_keys]
    mean_similarity = (
        sum(masi_similarity(a, b) for a, b in zip(sets_a, sets_b)) / len(unit_keys)
        if unit_keys
        else float("nan")
    )
    return {
        f"n_masi_units_{key}": len(unit_keys),
        f"masi_alpha_{key}": masi_alpha(sets_a, sets_b),
        f"masi_similarity_{key}": mean_similarity,
    }


def document_presence_iaa(
    annotations: pd.DataFrame,
    coder_a: str,
    coder_b: str,
    documents: Sequence[str],
) -> dict[str, float]:
    """Document-level concept IAA: 1 if the identifier appears anywhere in the PDF."""
    ids_a = identifiers_by_document(annotations, coder_a, documents)
    ids_b = identifiers_by_document(annotations, coder_b, documents)
    out: dict[str, float | int] = {
        "n_docs_marked_a": sum(1 for document in documents if ids_a.get(document)),
        "n_docs_marked_b": sum(1 for document in documents if ids_b.get(document)),
    }
    out.update(
        concept_presence_iaa(
            documents,
            ids_a,
            ids_b,
            count_key="n_doc_concepts",
            plural="doc_concepts",
            key="doc",
        )
    )
    out.update(masi_presence_iaa(ids_a, ids_b, key="doc"))
    return out


def document_to_project(coverage: pd.DataFrame) -> dict[str, str]:
    if "project" not in coverage.columns:
        return {}
    mapping: dict[str, str] = {}
    for document, project in zip(coverage["document"], coverage["project"]):
        if pd.isna(project) or str(project) == "":
            continue
        mapping[str(document)] = str(project)
    return mapping


def identifiers_by_project(
    annotations: pd.DataFrame,
    coder: str,
    documents: Sequence[str],
    doc_to_project: dict[str, str],
) -> dict[str, set[str]]:
    """Non-empty KB identifiers used by the coder on shared documents, pooled by tender."""
    by_project: dict[str, set[str]] = defaultdict(set)
    for document, identifiers in identifiers_by_document(
        annotations, coder, documents
    ).items():
        project = doc_to_project.get(document)
        if project:
            by_project[project].update(identifiers)
    return by_project


def project_presence_iaa(
    annotations: pd.DataFrame,
    coder_a: str,
    coder_b: str,
    documents: Sequence[str],
    doc_to_project: dict[str, str],
) -> dict[str, float]:
    """Project-level concept IAA: 1 if the identifier appears anywhere in the tender."""
    projects = sorted({doc_to_project[doc] for doc in documents if doc in doc_to_project})
    ids_a = identifiers_by_project(annotations, coder_a, documents, doc_to_project)
    ids_b = identifiers_by_project(annotations, coder_b, documents, doc_to_project)
    out: dict[str, float | int] = {
        "n_shared_projects": len(projects),
        "n_projects_marked_a": sum(1 for project in projects if ids_a.get(project)),
        "n_projects_marked_b": sum(1 for project in projects if ids_b.get(project)),
    }
    out.update(
        concept_presence_iaa(
            projects,
            ids_a,
            ids_b,
            count_key="n_project_concepts",
            plural="project_concepts",
            key="project",
        )
    )
    out.update(masi_presence_iaa(ids_a, ids_b, key="project"))
    return out


def build_continuum(
    coder_a: str,
    coder_b: str,
    units_a: Iterable[tuple[int, int, str]],
    units_b: Iterable[tuple[int, int, str]],
) -> Continuum | None:
    units_a = list(units_a)
    units_b = list(units_b)
    if not units_a and not units_b:
        return None
    continuum = Continuum()
    continuum.add_annotator(coder_a)
    continuum.add_annotator(coder_b)
    for start, end, category in units_a:
        continuum.add(coder_a, Segment(start, end), annotation=category)
    for start, end, category in units_b:
        continuum.add(coder_b, Segment(start, end), annotation=category)
    if continuum.num_annotators < 2:
        return None
    return continuum


@contextmanager
def quiet_pygamma():
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.ERROR)
    try:
        yield
    finally:
        root.setLevel(previous)


def compute_gamma(
    continuum: Continuum,
    *,
    alpha: float,
    beta: float,
    n_samples: int,
    precision_level: str | float,
    fast: bool,
) -> dict[str, float]:
    dissimilarity = CombinedCategoricalDissimilarity(alpha=alpha, beta=beta)
    with quiet_pygamma():
        result = continuum.compute_gamma(
            dissimilarity,
            n_samples=n_samples,
            precision_level=precision_level,
            fast=fast,
        )
    out = {
        "gamma": float(result.gamma),
        "observed_disorder": float(result.observed_disorder),
        "expected_disorder": float(result.expected_disorder),
    }
    if beta > 0:
        try:
            out["gamma_cat"] = float(result.gamma_cat)
        except (ValueError, ZeroDivisionError, AssertionError):
            out["gamma_cat"] = float("nan")
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def pairwise_rows(
    annotations: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    n_samples: int,
    precision_level: str | float,
    fast: bool,
    extra: dict | None = None,
) -> list[dict]:
    lengths = sofa_lengths_by_document(coverage)
    projects_by_document = document_to_project(coverage)
    coders = sorted(coverage["coder"].unique())
    rows: list[dict] = []
    pairs = list(combinations(coders, 2))
    scope = extra.get("sector") if extra else None
    scope_label = f" ({scope})" if scope else ""
    logger.info(f"Computing IAA for {len(pairs)} annotator pairs{scope_label}")
    for done, (coder_a, coder_b) in enumerate(pairs, start=1):
        documents = shared_documents(coverage, coder_a, coder_b)
        if documents:
            offsets, continuum_length = document_offsets(documents, lengths)
            span_a = pair_units(
                annotations, coder_a, documents, offsets, dedupe=True, require_identifier=False
            )
            span_b = pair_units(
                annotations, coder_b, documents, offsets, dedupe=True, require_identifier=False
            )
            concept_a = pair_units(
                annotations, coder_a, documents, offsets, dedupe=False, require_identifier=True
            )
            concept_b = pair_units(
                annotations, coder_b, documents, offsets, dedupe=False, require_identifier=True
            )

            row: dict = {
                "coder_a": coder_a,
                "coder_b": coder_b,
                "n_shared_docs": len(documents),
                "n_shared_projects": 0,
                "n_project_concepts": 0,
                "continuum_length": continuum_length,
                "n_units_a_span": len(span_a),
                "n_units_b_span": len(span_b),
                "n_units_a_concept": len(concept_a),
                "n_units_b_concept": len(concept_b),
                "n_docs_marked_a": 0,
                "n_docs_marked_b": 0,
                "n_doc_concepts": 0,
                "n_doc_concepts_marked_a": 0,
                "n_doc_concepts_marked_b": 0,
                "n_doc_concepts_agree": 0,
                "n_doc_concepts_both_marked": 0,
                "n_doc_concepts_both_empty": 0,
                "krippendorff_alpha_doc": float("nan"),
                "cohen_kappa_doc": float("nan"),
                "n_masi_units_doc": 0,
                "masi_alpha_doc": float("nan"),
                "masi_similarity_doc": float("nan"),
                "n_projects_marked_a": 0,
                "n_projects_marked_b": 0,
                "n_project_concepts_marked_a": 0,
                "n_project_concepts_marked_b": 0,
                "n_project_concepts_agree": 0,
                "n_project_concepts_both_marked": 0,
                "n_project_concepts_both_empty": 0,
                "krippendorff_alpha_project": float("nan"),
                "cohen_kappa_project": float("nan"),
                "n_masi_units_project": 0,
                "masi_alpha_project": float("nan"),
                "masi_similarity_project": float("nan"),
                "gamma_concept": float("nan"),
                "gamma_cat_concept": float("nan"),
                "observed_disorder_concept": float("nan"),
                "expected_disorder_concept": float("nan"),
                "n_overlap_a": 0,
                "n_overlap_share_id_a": 0,
                "n_overlap_b": 0,
                "n_overlap_share_id_b": 0,
                "n_overlap": 0,
                "n_overlap_share_id": 0,
                "overlap_id_rate": float("nan"),
            }
            row.update(document_presence_iaa(annotations, coder_a, coder_b, documents))
            row.update(
                project_presence_iaa(
                    annotations, coder_a, coder_b, documents, projects_by_document
                )
            )
            row.update(
                overlap_identifier_rate(annotations, coder_a, coder_b, documents)
            )
            if extra:
                row.update(extra)

            concept_continuum = build_continuum(coder_a, coder_b, concept_a, concept_b)
            if concept_continuum is not None:
                try:
                    concept = compute_gamma(
                        concept_continuum,
                        alpha=alpha,
                        beta=beta,
                        n_samples=n_samples,
                        precision_level=precision_level,
                        fast=fast,
                    )
                    row["gamma_concept"] = concept["gamma"]
                    row["gamma_cat_concept"] = concept.get("gamma_cat", float("nan"))
                    row["observed_disorder_concept"] = concept["observed_disorder"]
                    row["expected_disorder_concept"] = concept["expected_disorder"]
                except Exception as exc:
                    logger.warning(f"Concept γ failed for {coder_a}–{coder_b}: {exc}")

            rows.append(row)
        pair_label = f"{coder_a}–{coder_b}"
        if scope:
            pair_label = f"{scope} {pair_label}"
        log_progress(done, len(pairs), extra=pair_label)
    return rows


def pairwise_rows_by_sector(
    annotations: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    n_samples: int,
    precision_level: str | float,
    fast: bool,
    include_product_independent: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for sector in sorted(coverage["sector"].unique()):
        cov_s = coverage[coverage["sector"] == sector]
        docs = set(cov_s["document"])
        ann_s = annotations[annotations["document"].isin(docs)]
        logger.info(
            f"Sector {sector}: {cov_s['document'].nunique()} docs, "
            f"{cov_s['project'].nunique() if 'project' in cov_s.columns else '?'} projects, "
            f"{cov_s['coder'].nunique()} annotators"
        )
        rows.extend(
            pairwise_rows(
                ann_s,
                cov_s,
                alpha=alpha,
                beta=beta,
                n_samples=n_samples,
                precision_level=precision_level,
                fast=fast,
                extra={"sector": sector, "layer": PRODUCT_LAYER},
            )
        )
        if include_product_independent:
            ann_pic = ann_s[ann_s["identifier"].map(is_generic_identifier)]
            logger.info(
                f"Sector {sector} PIC (generic-*): "
                f"{len(ann_pic)} units, {ann_pic['coder'].nunique()} coders with generic spans"
            )
            rows.extend(
                pairwise_rows(
                    ann_pic,
                    cov_s,
                    alpha=alpha,
                    beta=beta,
                    n_samples=n_samples,
                    precision_level=precision_level,
                    fast=fast,
                    extra={"sector": sector, "layer": PIC_LAYER},
                )
            )
    return rows


def append_sector_block(
    lines: list[str],
    coverage: pd.DataFrame,
    sector_pairwise: pd.DataFrame,
    *,
    title: str,
) -> None:
    lines.append("")
    lines.append(title)
    for sector, group in sector_pairwise.groupby("sector", sort=True):
        n_docs_sector = (
            coverage.loc[coverage["sector"] == sector, "document"].nunique()
            if "sector" in coverage.columns
            else int(group["n_shared_docs"].max())
        )
        n_projects_sector = (
            coverage.loc[coverage["sector"] == sector, "project"].nunique()
            if "project" in coverage.columns
            else int(group["n_shared_projects"].max())
            if "n_shared_projects" in group.columns
            else 0
        )
        annotators = sorted(set(group["coder_a"]).union(group["coder_b"]))
        lines.append(
            f"  {sector}:  "
            f"proj-α={format_gamma(weighted_mean(group['krippendorff_alpha_project'], group['n_project_concepts']))}  "
            f"proj-κ={format_gamma(weighted_mean(group['cohen_kappa_project'], group['n_project_concepts']))}  "
            f"doc-α={format_gamma(weighted_mean(group['krippendorff_alpha_doc'], group['n_doc_concepts']))}  "
            f"doc-κ={format_gamma(weighted_mean(group['cohen_kappa_doc'], group['n_doc_concepts']))}  "
            f"proj-MASI-α={format_gamma(weighted_mean(group['masi_alpha_project'], group['n_masi_units_project']))}  "
            f"doc-MASI-α={format_gamma(weighted_mean(group['masi_alpha_doc'], group['n_masi_units_doc']))}  "
            f"concept={format_gamma(weighted_mean(group['gamma_concept'], group['continuum_length']))}  "
            f"γ-cat={format_gamma(weighted_mean(group['gamma_cat_concept'], group['continuum_length']))}  "
            f"overlap-id={format_percent(weighted_mean(group['overlap_id_rate'], group['n_overlap']))}  "
            f"({n_docs_sector} docs, {n_projects_sector} projects, {', '.join(annotators)})"
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"    {row.coder_a}–{row.coder_b}:  "
                f"{format_pair_metrics(row)}  "
                f"(project concepts {row.n_project_concepts_marked_a}/{row.n_project_concepts_marked_b} "
                f"of {row.n_project_concepts}, "
                f"doc concepts {row.n_doc_concepts_marked_a}/{row.n_doc_concepts_marked_b} "
                f"of {row.n_doc_concepts}, "
                f"units span {row.n_units_a_span}/{row.n_units_b_span}, "
                f"concept {row.n_units_a_concept}/{row.n_units_b_concept})"
            )


def format_gamma(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"


def format_pair_metrics(row: Any) -> str:
    return (
        f"proj-α={format_gamma(row.krippendorff_alpha_project)}  "
        f"proj-κ={format_gamma(row.cohen_kappa_project)}  "
        f"doc-α={format_gamma(row.krippendorff_alpha_doc)}  "
        f"doc-κ={format_gamma(row.cohen_kappa_doc)}  "
        f"proj-MASI-α={format_gamma(row.masi_alpha_project)}  "
        f"doc-MASI-α={format_gamma(row.masi_alpha_doc)}  "
        f"concept={format_gamma(row.gamma_concept)}  "
        f"γ-cat={format_gamma(row.gamma_cat_concept)}  "
        f"overlap-id={format_overlap(row.n_overlap_share_id, row.n_overlap, row.overlap_id_rate)}"
    )


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{100 * value:.1f}%"


def format_overlap(n_share: float, n_overlap: float, rate: float) -> str:
    if pd.isna(n_overlap) or n_overlap == 0:
        return "nan"
    return f"{int(n_share)}/{int(n_overlap)} ({format_percent(rate)})"


def write_summary(
    pairwise: pd.DataFrame,
    coverage: pd.DataFrame,
    path: Path,
    sector_pairwise: pd.DataFrame | None = None,
    dropped_coders: pd.DataFrame | None = None,
) -> None:
    lines = ["SwissSpark – Inter-annotator agreement", "=" * 60]
    n_coders = coverage["coder"].nunique()
    lines.append(f"Documents in coverage: {coverage['document'].nunique()}")
    lines.append(f"Annotators:            {n_coders} ({', '.join(sorted(coverage['coder'].unique()))})")
    lines.append(
        "Coverage is restricted to each document's annotators as listed in "
        "inception_metadata.json; their empty documents are kept as true voids."
    )
    if "sector" in coverage.columns:
        lines.append("")
        lines.append("Sector annotators:")
        for sector, group in coverage.groupby("sector"):
            coders = ", ".join(sorted(group["coder"].unique()))
            lines.append(
                f"  {sector}: {coders} ({group['document'].nunique()} docs"
                + (
                    f", {group['project'].nunique()} projects)"
                    if "project" in group.columns
                    else ")"
                )
            )
    if dropped_coders is not None and not dropped_coders.empty:
        lines.append("")
        lines.append(
            "Non-annotators (INCEpTION shows a CAS for the document, but the coder "
            "is either not listed as one of its annotators in inception_metadata.json, "
            "or is listed but never reached FINISHED on any document in that sector "
            "per exportedproject.json — dropped before IAA):"
        )
        for sector, group in dropped_coders.groupby("sector", sort=True):
            for row in group.itertuples(index=False):
                lines.append(f"  {sector}: {row.coder:20s}  {row.n_documents:4d} docs")

    lines.append("")
    lines.append("Documents per annotator:")
    for coder, count in coverage.groupby("coder")["document"].nunique().sort_index().items():
        lines.append(f"  {coder:20s}  {count:4d}")

    if pairwise.empty:
        lines.append("")
        lines.append("No annotator pairs share a document. Cannot compute IAA.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("")
    lines.append("Pairwise overlap (shared documents / continuum characters):")
    for row in pairwise.itertuples(index=False):
        lines.append(
            f"  {row.coder_a}–{row.coder_b}:  {row.n_shared_docs} docs, "
            f"{row.n_shared_projects} projects, length={row.continuum_length}"
        )

    w_alpha_project = weighted_mean(
        pairwise["krippendorff_alpha_project"], pairwise["n_project_concepts"]
    )
    w_kappa_project = weighted_mean(pairwise["cohen_kappa_project"], pairwise["n_project_concepts"])
    w_alpha = weighted_mean(pairwise["krippendorff_alpha_doc"], pairwise["n_doc_concepts"])
    w_kappa = weighted_mean(pairwise["cohen_kappa_doc"], pairwise["n_doc_concepts"])
    w_concept = weighted_mean(pairwise["gamma_concept"], pairwise["continuum_length"])
    w_cat = weighted_mean(pairwise["gamma_cat_concept"], pairwise["continuum_length"])
    w_overlap = weighted_mean(pairwise["overlap_id_rate"], pairwise["n_overlap"])
    w_masi_project = weighted_mean(pairwise["masi_alpha_project"], pairwise["n_masi_units_project"])
    w_masi_doc = weighted_mean(pairwise["masi_alpha_doc"], pairwise["n_masi_units_doc"])
    lines.append("")
    lines.append(
        "Project-level IAA (unit = (SIMAP tender, KB identifier); label = that "
        "identifier was annotated anywhere in the tender). Krippendorff's α (nominal) "
        "and Cohen's κ; headline weight = (tender, concept) units. α/κ = 1 when both "
        "assign the same single label to every unit (chance correction undefined). "
        "NOTE: this decomposes each tender's AoA *set* into one binary "
        "presence/absence item per identifier in the union either coder used, "
        "pooled across the whole catalog -- a pragmatic but nonstandard way to "
        "score multi-label data (see MASI below for the standard treatment); "
        "this number is likely a pessimistic lower bound, not a fair read of "
        "actual agreement."
    )
    lines.append(f"  α:             {format_gamma(w_alpha_project)}")
    lines.append(f"  κ:             {format_gamma(w_kappa_project)}")
    lines.append("")
    lines.append(
        "Document-level IAA (unit = (document, KB identifier); label = that "
        "identifier was annotated anywhere in the PDF). Krippendorff's α (nominal) "
        "and Cohen's κ; headline weight = (document, concept) units. α/κ = 1 when both "
        "assign the same single label to every unit (chance correction undefined). "
        "Same caveat as project-level α above."
    )
    lines.append(f"  α:             {format_gamma(w_alpha)}")
    lines.append(f"  κ:             {format_gamma(w_kappa)}")
    lines.append("")
    lines.append(
        "MASI-based α (unit = (tender-or-document, catalog); value = the *set* of "
        "AoA ids the coder assigned there). This is the standard treatment for "
        "multi-label annotation (Passonneau, 2006): unlike the presence α/κ above, "
        "it never explodes a set into per-label binary items, so it isn't biased "
        "pessimistic by a large, similar-looking catalog. Headline weight = "
        "(unit, catalog) count. Prefer this over the presence α/κ above."
    )
    lines.append(f"  Project MASI-α: {format_gamma(w_masi_project)}")
    lines.append(f"  Document MASI-α: {format_gamma(w_masi_doc)}")
    lines.append("")
    lines.append(
        "Length-weighted mean of pairwise concept γ (weight = concatenated sofa length):"
    )
    lines.append(f"  Concept:       {format_gamma(w_concept)}")
    lines.append(f"  Concept γ-cat: {format_gamma(w_cat)}")
    lines.append("")
    lines.append(
        "Overlap identifier rate: unique (document, begin, end) spans that overlap "
        "the other annotator by ≥1 character; fraction whose identifier sets intersect "
        "(pooled both directions; ignores one-sided spans; not chance-corrected). "
        "Headline weight = overlapping-span counts."
    )
    lines.append(f"  Overlap-id:    {format_percent(w_overlap)}")
    lines.append("")
    lines.append("Pairwise IAA:")
    for row in pairwise.itertuples(index=False):
        lines.append(
            f"  {row.coder_a}–{row.coder_b}:  "
            f"{format_pair_metrics(row)}  "
            f"(project concepts {row.n_project_concepts_marked_a}/{row.n_project_concepts_marked_b} "
            f"of {row.n_project_concepts}, "
            f"doc concepts {row.n_doc_concepts_marked_a}/{row.n_doc_concepts_marked_b} "
            f"of {row.n_doc_concepts}, "
            f"units span {row.n_units_a_span}/{row.n_units_b_span}, "
            f"concept {row.n_units_a_concept}/{row.n_units_b_concept})"
        )

    if sector_pairwise is not None and not sector_pairwise.empty:
        product_rows = sector_pairwise
        pic_rows = pd.DataFrame()
        if "layer" in sector_pairwise.columns:
            product_rows = sector_pairwise[sector_pairwise["layer"] != PIC_LAYER]
            pic_rows = sector_pairwise[sector_pairwise["layer"] == PIC_LAYER]
        if not product_rows.empty:
            append_sector_block(
                lines,
                coverage,
                product_rows,
                title="Per-sector IAA (proj-α/κ weighted by (tender, concept) units; doc-α/κ by (document, concept) units; concept γ by sofa length):",
            )
        if not pic_rows.empty:
            append_sector_block(
                lines,
                coverage,
                pic_rows,
                title=(
                    "Per-sector PIC (generic-* identifiers, same sector annotators; "
                    "not a global PIC continuum):"
                ),
            )
            w_alpha_project_pic = weighted_mean(
                pic_rows["krippendorff_alpha_project"], pic_rows["n_project_concepts"]
            )
            w_kappa_project_pic = weighted_mean(
                pic_rows["cohen_kappa_project"], pic_rows["n_project_concepts"]
            )
            w_alpha_pic = weighted_mean(
                pic_rows["krippendorff_alpha_doc"], pic_rows["n_doc_concepts"]
            )
            w_kappa_pic = weighted_mean(pic_rows["cohen_kappa_doc"], pic_rows["n_doc_concepts"])
            w_concept_pic = weighted_mean(pic_rows["gamma_concept"], pic_rows["continuum_length"])
            w_cat_pic = weighted_mean(pic_rows["gamma_cat_concept"], pic_rows["continuum_length"])
            w_overlap_pic = weighted_mean(pic_rows["overlap_id_rate"], pic_rows["n_overlap"])
            lines.append("")
            lines.append(
                "Aggregate of per-sector PIC pairwise IAA "
                "(project α/κ weight = (tender, concept) units; document α/κ weight = (document, concept) units; "
                "concept γ weight = sofa length; PIC = generic-* identifier presence):"
            )
            lines.append(f"  Project α:     {format_gamma(w_alpha_project_pic)}")
            lines.append(f"  Project κ:     {format_gamma(w_kappa_project_pic)}")
            lines.append(f"  Document α:    {format_gamma(w_alpha_pic)}")
            lines.append(f"  Document κ:    {format_gamma(w_kappa_pic)}")
            lines.append(f"  Concept:       {format_gamma(w_concept_pic)}")
            lines.append(f"  Concept γ-cat: {format_gamma(w_cat_pic)}")
            lines.append(f"  Overlap-id:    {format_percent(w_overlap_pic)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_iaa(
    input_dir: Path,
    output: Path,
    *,
    alpha: float,
    beta: float,
    n_samples: int,
    precision_level: str | float,
    fast: bool,
    metadata: Path,
    include_product_independent: bool,
) -> None:
    annotations, coverage, states = load_extract(input_dir)
    logger.info(
        f"Loaded {len(annotations):,} spans and {len(coverage):,} coverage rows "
        f"({coverage['coder'].nunique()} annotators, {coverage['document'].nunique()} documents)"
    )
    annotations = canonicalize_annotations(annotations)
    sector_map = load_sector_map(metadata)
    coverage, dropped_coverage = filter_coverage_to_sector_annotators(
        coverage, annotations, sector_map, states
    )
    dropped_coders = sector_non_annotator_summary(dropped_coverage)
    iaa_kwargs = {
        "alpha": alpha,
        "beta": beta,
        "n_samples": n_samples,
        "precision_level": precision_level,
        "fast": fast,
    }
    rows = pairwise_rows(annotations, coverage, **iaa_kwargs)
    sector_rows = pairwise_rows_by_sector(
        annotations,
        coverage,
        include_product_independent=include_product_independent,
        **iaa_kwargs,
    )
    output.mkdir(parents=True, exist_ok=True)
    pairwise = pd.DataFrame(rows)
    sector_pairwise = pd.DataFrame(sector_rows)
    pairwise_path = output / PAIRWISE_CSV
    sector_path = output / SECTOR_PAIRWISE_CSV
    pairwise.to_csv(pairwise_path, index=False)
    if not sector_pairwise.empty:
        preferred = [col for col in ("sector", "layer") if col in sector_pairwise.columns]
        ordered = preferred + [col for col in sector_pairwise.columns if col not in preferred]
        sector_pairwise = sector_pairwise[ordered]
    sector_pairwise.to_csv(sector_path, index=False)
    summary_path = output / SUMMARY_TXT
    write_summary(
        pairwise,
        coverage,
        summary_path,
        sector_pairwise=sector_pairwise,
        dropped_coders=dropped_coders,
    )
    logger.info(f"Pairwise results → {pairwise_path}")
    logger.info(f"Per-sector IAA   → {sector_path}")
    logger.info(f"Summary          → {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_skip_coders(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def parse_precision(raw: str) -> str | float:
    if raw in {"high", "medium", "low"}:
        return raw
    return float(raw)


def add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        required=True,
        type=Path,
        help="INCEpTION project export .zip, unzipped export directory, or annotation/ folder",
    )
    parser.add_argument(
        "--skip-coders",
        default=",".join(DEFAULT_SKIP_CODERS),
        help="Comma-separated coder folder names to ignore "
        f"(default: {','.join(DEFAULT_SKIP_CODERS)})",
    )


def add_iaa_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Positional weight in the combined dissimilarity (default: 1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Categorical weight for concept γ (default: 1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Random continua sampled for chance correction (default: 30)",
    )
    parser.add_argument(
        "--precision",
        default="medium",
        help='γ estimation precision: high/medium/low or a float (default: "medium")',
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast-γ (default: true). --no-fast for the canonical algorithm",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="JSON mapping INCEpTION document names to WDG catalogs "
        f"(default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--product-independent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score generic-* / product-independent criteria separately inside "
        "each product sector, using that sector's annotators only. "
        "Does not create a global PIC continuum (default: off)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Document-level IAA (α, κ) and concept γ from an INCEpTION XMI export",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Parse XMI export → annotations.csv + coverage.csv")
    add_extract_args(extract)
    extract.add_argument("--output", required=True, type=Path, help="Directory for CSV output")

    iaa = sub.add_parser("iaa", help="Load extract CSVs and compute pairwise document α/κ and concept γ")
    iaa.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing annotations.csv and coverage.csv",
    )
    iaa.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for IAA results (default: same as --input)",
    )
    add_iaa_args(iaa)

    all_cmd = sub.add_parser("all", help="extract, then iaa")
    add_extract_args(all_cmd)
    all_cmd.add_argument("--output", required=True, type=Path, help="Directory for CSVs and IAA results")
    add_iaa_args(all_cmd)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        cmd_extract(args.export, args.output, parse_skip_coders(args.skip_coders))
    elif args.command == "iaa":
        output = args.output or args.input
        cmd_iaa(
            args.input,
            output,
            alpha=args.alpha,
            beta=args.beta,
            n_samples=args.n_samples,
            precision_level=parse_precision(args.precision),
            fast=args.fast,
            metadata=args.metadata,
            include_product_independent=args.product_independent,
        )
    elif args.command == "all":
        cmd_extract(args.export, args.output, parse_skip_coders(args.skip_coders))
        cmd_iaa(
            args.output,
            args.output,
            alpha=args.alpha,
            beta=args.beta,
            n_samples=args.n_samples,
            precision_level=parse_precision(args.precision),
            fast=args.fast,
            metadata=args.metadata,
            include_product_independent=args.product_independent,
        )
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
