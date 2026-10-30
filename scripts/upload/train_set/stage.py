"""Stage the TENDRIL training set (LLM-judged corpus) as MTEB-style corpus/queries/qrels.

Only passages that were judged are kept. The judgments already exclude projects whose tender documents restrict
reuse; this module re-checks that against the restriction list and the test projects. Project IDs and file names
are replaced by keyed hashes so passages can still be grouped by project and document without revealing either.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger

HASH_LENGTH = 16
AREA_SUFFIX = re.compile(r"(HF-\d+)$")
CONFIGS = ("corpus", "queries", "qrels")


def load_hash_key(path: Path) -> bytes:
    """Return the secret HMAC key, creating it once so hashes stay stable across uploads."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
        logger.info(f"Created new hash key at {path}")
    return bytes.fromhex(path.read_text().strip())


def keyed_hash(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:HASH_LENGTH]


def read_project_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def stage(
    source: Path,
    out: Path,
    hash_key: Path,
    restrictions: Path,
    test_pipeline: Path,
) -> list[dict]:
    """Write ``out/data/{corpus,queries,qrels}.parquet`` and return the README ``configs`` entries."""
    judgments = pq.read_table(source / "judged" / "relevance_judgments.parquet")
    passages = pq.read_table(
        source / "passages_anon.parquet",
        columns=["passage_uid", "project_id", "file_name", "page", "text"],
    )
    queries = pq.read_table(
        source / "queries_modified.parquet",
        columns=["canonical_query_id", "level", "area_id", "query_text"],
    )

    pairs = {
        (q, d)
        for q, d in zip(
            judgments["query_id"].to_pylist(), judgments["document_id"].to_pylist()
        )
    }
    if len(pairs) != judgments.num_rows:
        raise SystemExit("ERROR: duplicate (query_id, document_id) judgments")

    judged_documents = pc.unique(judgments["document_id"])
    passages = passages.filter(
        pc.is_in(passages["passage_uid"], value_set=judged_documents)
    )
    if passages.num_rows != len(judged_documents):
        raise SystemExit(
            f"ERROR: {len(judged_documents)} judged passages but {passages.num_rows} matching passage rows"
        )

    projects = set(passages["project_id"].to_pylist())
    restricted = read_project_ids(restrictions) & projects
    if restricted:
        raise SystemExit(
            f"ERROR: {len(restricted)} restricted projects in the judged corpus"
        )
    test_projects = {
        path.name.split(".")[0] for path in test_pipeline.glob("simap_*.json.zst")
    }
    if not test_projects:
        raise SystemExit(f"ERROR: no test pipeline files in {test_pipeline}")
    if test_projects & projects:
        raise SystemExit("ERROR: test projects in the training corpus")

    key = load_hash_key(hash_key)
    project_ids = passages["project_id"].to_pylist()
    corpus = pa.table(
        {
            "id": passages["passage_uid"],
            "text": passages["text"],
            "project": [
                keyed_hash(key, f"project:{project}") for project in project_ids
            ],
            "document": [
                keyed_hash(key, f"document:{project}/{name}")
                for project, name in zip(project_ids, passages["file_name"].to_pylist())
            ],
            "page": passages["page"],
        }
    )

    judged_queries = set(pc.unique(judgments["query_id"]).to_pylist())
    query_rows: dict[str, dict] = {}
    for row in queries.to_pylist():
        query_id = row["canonical_query_id"]
        if query_id not in judged_queries:
            continue
        catalog = query_id.split("::", 1)[0]
        area = AREA_SUFFIX.search(row["area_id"] or "")
        candidate = {
            "id": query_id,
            "text": row["query_text"],
            "level": row["level"],
            "catalog": catalog,
            "area": f"{catalog}::{area.group(1)}" if area else None,
        }
        previous = query_rows.setdefault(query_id, candidate)
        if previous["text"] != candidate["text"]:
            raise SystemExit(f"ERROR: query {query_id} has more than one text")
    missing = judged_queries - set(query_rows)
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} judged queries without text, e.g. {sorted(missing)[:3]}"
        )
    query_table = pa.Table.from_pylist(
        sorted(query_rows.values(), key=lambda row: row["id"]),
        schema=pa.schema(
            [
                ("id", pa.string()),
                ("text", pa.string()),
                ("level", pa.string()),
                ("catalog", pa.string()),
                ("area", pa.string()),
            ]
        ),
    )

    qrels = pa.table(
        {
            "query-id": judgments["query_id"],
            "corpus-id": judgments["document_id"],
            "score": pc.cast(judgments["score"], pa.int32()),
        }
    )

    target = out / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name, table in (("corpus", corpus), ("queries", query_table), ("qrels", qrels)):
        pq.write_table(table, target / f"{name}.parquet")

    scores = pc.value_counts(qrels["score"]).to_pylist()
    logger.info(
        f"Staged training set: {len(projects)} projects, {corpus.num_rows} passages, {query_table.num_rows} queries, "
        f"{qrels.num_rows} qrels, score counts {sorted((row['values'], row['counts']) for row in scores)}"
    )
    return [
        {
            "config_name": name,
            "data_files": [{"split": "train", "path": f"data/{name}.parquet"}],
            **({"default": True} if name == "qrels" else {}),
        }
        for name in CONFIGS
    ]
