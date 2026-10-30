"""
Outputs:
passages.parquet  one row per surviving passage, with provenance
pairs.parquet     one row per (query, passage) pair to be judged
queries.parquet   one row per distinct query, with the tier it came from

Usage:
    python scripts/train_set/build_judge_input.py
    python scripts/train_set/build_judge_input.py --units-dir data/derived/judge --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import yaml
import zstandard
from loguru import logger
from tqdm import tqdm

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

TABLE_MARKER = re.compile(r"<(?:table|tr|td|th)\b", re.IGNORECASE)
TABLE_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
TABLE_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
FILENAME_TOKEN = re.compile(r"\.(?:docx?|pdf|xlsx?|pptx?)\b", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

# Negative-scope clause that helps retrieval but would mislead a relevance judge.
NICHT_GEMEINT = re.compile(
    r"\s*Nicht gemeint:.*?(?=\n*</HandlungsfeldZusammenfassung>|\Z)", re.IGNORECASE | re.DOTALL
)


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.reasons[reason] += 1


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def passage_uid(project_id: str, part_id: str, text: str) -> str:
    digest = hashlib.blake2b(
        f"{project_id}|{part_id}|{text}".encode(), digest_size=10
    ).hexdigest()
    return digest


def digit_symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    hits = sum(1 for ch in text if ch.isdigit() or (not ch.isalnum() and not ch.isspace()))
    return hits / len(text)


def has_repeated_span(text: str, min_chars: int, fraction: float) -> bool:
    # True if a substring covering a meaningful share of the passage repeats. Scaled to passage length

    length = max(int(min_chars), int(len(text) * fraction))
    if len(text) < length * 2:
        return False
    stride = max(1, length // 8)
    for start in range(0, len(text) - length + 1, stride):
        if text.find(text[start : start + length], start + 1) != -1:
            return True
    return False


def table_is_dense(text: str, min_fill: float, min_cells: int, min_cell_chars: int) -> bool:
    rows = TABLE_ROW.findall(text)
    if not rows:
        return False
    per_row = [TABLE_CELL.findall(row) for row in rows]
    widths = [len(cells) for cells in per_row]
    max_cols = max(widths, default=0)
    if max_cols == 0:
        return False
    fill = sum(w / max_cols for w in widths) / len(widths)
    if fill < min_fill:
        return False
    distinct = {
        normalise(HTML_TAG.sub(" ", cell))
        for cells in per_row
        for cell in cells
        if len(normalise(HTML_TAG.sub(" ", cell))) > min_cell_chars
    }
    return len(distinct) >= min_cells


def passes_quality(text: str, cfg: dict[str, Any], stats: FilterStats) -> bool:
    if len(text) < int(cfg["min_chars"]):
        stats.reject("too_short")
        return False
    if len(text) > int(cfg["max_chars"]):
        stats.reject("too_long")
        return False

    if TABLE_MARKER.search(text):
        if not table_is_dense(
            text,
            float(cfg["table_min_row_fill"]),
            int(cfg["table_min_distinct_cells"]),
            int(cfg["table_min_cell_chars"]),
        ):
            stats.reject("sparse_table")
            return False
        return True

    if digit_symbol_ratio(text) > float(cfg["max_digit_symbol_ratio"]):
        stats.reject("digit_symbol_ratio")
        return False
    if len(FILENAME_TOKEN.findall(text)) >= int(cfg["max_filename_tokens"]):
        stats.reject("filename_listing")
        return False
    if has_repeated_span(
        text, int(cfg["repeated_span_min_chars"]), float(cfg["repeated_span_fraction"])
    ):
        stats.reject("repeated_span")
        return False
    return True


def ngrams(text: str, n: int) -> set[str]:
    folded = text.casefold()
    if len(folded) < n:
        return {folded}
    return {folded[i : i + n] for i in range(len(folded) - n + 1)}


def deduplicate(
    passages: list[dict], n: int, threshold: float, stats: FilterStats
) -> list[dict]:
    """Drop near-duplicates within a project using character n-gram Jaccard."""
    # Longer passages first, so the fuller version of boilerplate survives.
    ordered = sorted(passages, key=lambda p: -len(p["text"]))
    accepted: list[tuple[int, set[str], dict]] = []

    for passage in ordered:
        grams = ngrams(passage["text"], n)
        size = len(grams)
        # Jaccard >= t forces the set sizes to be within a factor of t.
        lower, upper = size * threshold, size / threshold
        duplicate = False
        for other_size, other_grams, _ in accepted:
            if not lower <= other_size <= upper:
                continue
            intersection = len(grams & other_grams)
            if intersection / (size + other_size - intersection) >= threshold:
                duplicate = True
                break
        if duplicate:
            stats.reject("near_duplicate")
        else:
            accepted.append((size, grams, passage))
    return [passage for _, _, passage in accepted]


def iter_project_passages(
    payload: dict, wanted_files: set[str]
) -> Iterable[tuple[str, dict]]:
    for document in payload.get("documents") or []:
        file_name = str(document.get("file_name") or "")
        if file_name not in wanted_files:
            continue
        for chunk in document.get("content") or []:
            text = normalise(str(chunk.get("content_cleaned") or ""))
            if text:
                yield file_name, {"chunk": chunk, "text": text}


def load_query_texts(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    #Return embedding_text keyed by criterion ID, and by area key for virtual criteria.
    payload = orjson.loads(path.read_bytes())
    by_criterion: dict[str, str] = {}
    by_area: dict[str, str] = {}
    for point in payload["points"]:
        item = point["payload"]
        text = NICHT_GEMEINT.sub("", str(item.get("embedding_text") or "")).strip()
        if item.get("is_virtual_scope_criterion"):
            by_area[f"{item['catalog']}-{item['scope_of_action_id']}"] = text
        else:
            by_criterion[str(item["id"])] = text
    return by_criterion, by_area


def build_queries(
    areas: pd.DataFrame,
    criterion_text: dict[str, str],
    area_text: dict[str, str],
) -> pd.DataFrame:
    # One row per distinct query: the area itself plus each selected criterion.
    # Area-level rows are emitted only where a virtual criterion exists
    rows: list[dict] = []
    for record in areas.to_dict("records"):
        area_id = record["area_id"]
        if area_id in area_text:
            rows.append(
                {
                    "project_id": record["project_id"],
                    "query_id": area_id,
                    "level": "area_of_action",
                    "area_id": area_id,
                    "criterion_id": None,
                    "catalog": record["catalog"],
                    "tier": record["tier"],
                    "verdict": record["verdict"],
                    "query_text": area_text[area_id],
                }
            )
        for criterion_id in record["criterion_ids"]:
            text = criterion_text.get(criterion_id)
            if not text:
                continue
            rows.append(
                {
                    "project_id": record["project_id"],
                    "query_id": criterion_id,
                    "level": "criterion",
                    "area_id": area_id,
                    "criterion_id": criterion_id,
                    "catalog": record["catalog"],
                    "tier": record["tier"],
                    "verdict": record["verdict"],
                    "query_text": text,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["project_id", "query_id"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--units-dir", type=Path, default=None, help="Where the selection lives")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only process N projects (debug)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    filters = config["passage_filters"]

    units_dir = args.units_dir or Path(config["paths"]["output_dir"])
    output_dir = args.output_dir or units_dir
    project_dir = Path(config["paths"]["enriched_projects"])

    areas = pd.read_parquet(units_dir / "selected_areas.parquet")
    documents = pd.read_parquet(units_dir / "selected_documents.parquet")
    logger.info(f"Loaded {len(areas):,} area rows and {len(documents):,} document rows")

    criterion_text, area_text = load_query_texts(Path(config["paths"]["criteria_payloads"]))
    logger.info(f"Query texts: {len(criterion_text):,} criteria, {len(area_text):,} areas")

    queries = build_queries(areas, criterion_text, area_text)
    logger.info(f"Queries: {len(queries):,} ({queries['level'].value_counts().to_dict()})")
    n_area_suppressed = len(areas) - int((queries["level"] == "area_of_action").sum())
    if n_area_suppressed:
        logger.warning(f"  {n_area_suppressed} area-level queries suppressed (no virtual criterion)")

    wanted: dict[str, set[str]] = defaultdict(set)
    for project_id, file_name in zip(documents["project_id"], documents["file_name"]):
        wanted[project_id].add(file_name)

    project_ids = sorted(wanted)
    if args.limit:
        project_ids = project_ids[: args.limit]

    dctx = zstandard.ZstdDecompressor()
    stats = FilterStats()
    passage_rows: list[dict] = []

    for project_id in tqdm(project_ids, desc="passages"):
        path = project_dir / f"{project_id}.json.zst"
        if not path.is_file():
            logger.warning(f"missing project file: {path}")
            continue
        with path.open("rb") as fh:
            payload = orjson.loads(dctx.stream_reader(fh).read())

        candidates: list[dict] = []
        for file_name, item in iter_project_passages(payload, wanted[project_id]):
            stats.seen += 1
            text = item["text"]
            if not passes_quality(text, filters, stats):
                continue
            chunk = item["chunk"]
            candidates.append(
                {
                    "passage_uid": passage_uid(project_id, str(chunk.get("part_id") or ""), text),
                    "project_id": project_id,
                    "file_name": file_name,
                    "part_id": str(chunk.get("part_id") or ""),
                    "page": chunk.get("page"),
                    "index": chunk.get("index"),
                    "text": text,
                    "n_chars": len(text),
                }
            )

        kept = deduplicate(
            candidates, int(filters["dedup_ngram"]), float(filters["dedup_jaccard"]), stats
        )
        stats.kept += len(kept)
        passage_rows.extend(kept)

    passages = pd.DataFrame(passage_rows).drop_duplicates("passage_uid")
    logger.info(f"Passages: {stats.seen:,} seen -> {len(passages):,} kept")
    for reason, count in stats.reasons.most_common():
        logger.info(f"    rejected {reason:<22} {count:>8,} ({100 * count / max(stats.seen, 1):.1f}%)")

    pairs = queries.merge(
        passages[["passage_uid", "project_id"]], on="project_id", how="inner"
    )[["project_id", "query_id", "level", "area_id", "criterion_id", "tier", "passage_uid"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    passages.to_parquet(output_dir / "passages.parquet", index=False)
    queries.to_parquet(output_dir / "queries.parquet", index=False)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)

    logger.info("--- judge input ---")
    logger.info(f"  projects  {passages['project_id'].nunique():,}")
    logger.info(f"  documents {passages.groupby(['project_id', 'file_name']).ngroups:,}")
    logger.info(f"  passages  {len(passages):,}  (median {passages['n_chars'].median():.0f} chars)")
    logger.info(f"  queries   {len(queries):,}")
    logger.info(f"  PAIRS     {len(pairs):,}")
    logger.info(f"  by level  {pairs['level'].value_counts().to_dict()}")
    logger.info(f"  by tier   {pairs['tier'].value_counts().to_dict()}")
    logger.info(f"Wrote passages/queries/pairs parquet to {output_dir}")


if __name__ == "__main__":
    main()
