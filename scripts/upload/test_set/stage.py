"""Stage the TENDRIL test set as per-project MTEB retrieval configs.

Every project shard ``<alias>/{corpus.jsonl,queries.jsonl,qrels/test.tsv}`` becomes three parquet files and three
Hub configs (``<alias>-corpus``, ``<alias>-queries``, ``<alias>-qrels``, split ``test``). That is the layout
``mteb``'s ``RetrievalDatasetLoader`` and ``tendril.tasks.make_hub_task`` load.
"""

from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

COMPONENTS = ("corpus", "queries", "qrels")


def read_jsonl(path: Path) -> list[dict]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def read_qrels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                "query-id": row["query-id"],
                "corpus-id": row["corpus-id"],
                "score": int(row["score"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def check_private_aliases(
    projects: list[str], aliases: Path, restrictions: Path
) -> None:
    """Cross-check the shards against the private SIMAP mapping and the restriction list."""
    if not aliases.is_file():
        logger.warning(
            f"No private alias mapping at {aliases}; skipping the SIMAP restriction cross-check"
        )
        return
    mapping = orjson.loads(aliases.read_bytes())
    restricted = {
        line.strip() for line in restrictions.read_text().splitlines() if line.strip()
    }
    if sorted(mapping.values()) != sorted(projects):
        raise SystemExit(f"ERROR: {aliases} does not describe the staged projects")
    leaked = sorted(
        alias for project, alias in mapping.items() if project in restricted
    )
    if leaked:
        raise SystemExit(
            f"ERROR: restricted projects in the test set: {', '.join(leaked)}"
        )


def stage(
    source: Path, out: Path, repo_id: str, aliases: Path, restrictions: Path
) -> list[dict]:
    """Write parquet shards to ``out/data`` and return the README ``configs`` entries."""
    manifest = orjson.loads((source / "mteb_dataset.json").read_bytes())
    projects = [str(project) for project in manifest["projects"]]
    directories = sorted(path.name for path in source.iterdir() if path.is_dir())
    if sorted(projects) != directories:
        raise SystemExit(
            f"ERROR: mteb_dataset.json and the shard directories in {source} disagree"
        )
    check_private_aliases(projects, aliases, restrictions)

    configs: list[dict] = []
    totals = dict.fromkeys(COMPONENTS, 0)
    for project in projects:
        shard = source / project
        corpus = read_jsonl(shard / "corpus.jsonl")
        queries = read_jsonl(shard / "queries.jsonl")
        qrels = read_qrels(shard / "qrels" / "test.tsv")
        corpus_ids = {row["id"] for row in corpus}
        query_ids = {row["id"] for row in queries}
        if len(corpus_ids) != len(corpus) or len(query_ids) != len(queries):
            raise SystemExit(f"ERROR: duplicate ids in {shard}")
        dangling = [
            row
            for row in qrels
            if row["query-id"] not in query_ids or row["corpus-id"] not in corpus_ids
        ]
        if dangling:
            raise SystemExit(
                f"ERROR: {len(dangling)} qrels in {shard} reference unknown ids"
            )

        tables = {
            "corpus": pa.Table.from_pylist(
                corpus, schema=pa.schema([("id", pa.string()), ("text", pa.string())])
            ),
            "queries": pa.Table.from_pylist(
                queries, schema=pa.schema([("id", pa.string()), ("text", pa.string())])
            ),
            "qrels": pa.Table.from_pylist(
                qrels,
                schema=pa.schema(
                    [
                        ("query-id", pa.string()),
                        ("corpus-id", pa.string()),
                        ("score", pa.int32()),
                    ]
                ),
            ),
        }
        target = out / "data" / project
        target.mkdir(parents=True, exist_ok=True)
        for component, table in tables.items():
            pq.write_table(table, target / f"{component}.parquet")
            totals[component] += table.num_rows
            configs.append(
                {
                    "config_name": f"{project}-{component}",
                    "data_files": [
                        {"split": "test", "path": f"data/{project}/{component}.parquet"}
                    ],
                }
            )

    (out / "mteb_dataset.json").write_bytes(
        orjson.dumps(
            {**manifest, "projects": projects, "hub_repo": repo_id},
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    logger.info(
        f"Staged {len(projects)} test projects: {totals['queries']} query entries, "
        f"{totals['corpus']} passages, {totals['qrels']} qrels"
    )
    return configs
