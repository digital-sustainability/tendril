#!/usr/bin/env python
"""Write training queries in the wording the judge labeled and the test set uses.

Area-of-action rows get their LLM summary from ``llm_summary.jsonl``. Criterion
rows get the ``Ausschreibungskriterium`` text without XML tags, followed by a
blank line and the ambition-level block when the criterion has one. This is
the same text the judge saw as its ``Bezugstext``.

``query_id`` keeps its original value so the output still joins with
``pairs.parquet`` on ``(project_id, query_id)``. The added
``canonical_query_id`` column (``Catalog::HF-NNN`` / ``Catalog::K-NNN``) matches
``relevance_judgments.parquet`` and the test-set query ids. All other columns
and the row order are unchanged.

Example::

    python scripts/queries/build_training_queries.py --root /path/to/tendril_training
    python scripts/queries/build_training_queries.py --root /path/to/tendril_training --output /tmp/queries.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from tendril.paths import REPOSITORY_PATH
from tendril.queries import load_query_map, rewrite_queries

ROOT = REPOSITORY_PATH
DEFAULT_SUMMARY = ROOT / "data" / "queries" / "llm_summary.jsonl"
OUTPUT_NAME = "queries_modified.parquet"


def build_training_queries(queries: pd.DataFrame, summaries: dict[str, str]) -> pd.DataFrame:
    """Return ``queries`` with rewritten ``query_text`` and a ``canonical_query_id`` column."""
    rewritten = rewrite_queries(queries, summaries, "query_text", "query_id")
    texts = [
        f"{text}\n\n{ambition}" if ambition else text
        for text, ambition in zip(
            rewritten["query_text"], rewritten["ambition_levels"], strict=True
        )
    ]
    result = queries.copy()
    result["query_text"] = texts
    result["canonical_query_id"] = rewritten["query_id"].to_numpy()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing queries.parquet")
    parser.add_argument("--queries", default="queries.parquet", help="queries filename inside --root")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help="area-of-action summaries ({id, text} JSONL)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"output parquet (default: <root>/{OUTPUT_NAME})")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.root / args.queries
    output = args.output or args.root / OUTPUT_NAME
    if output.resolve() == source.resolve():
        raise SystemExit(f"Refusing to overwrite the source queries: {source}")

    queries = pd.read_parquet(source)
    summaries = load_query_map(args.summary)
    result = build_training_queries(queries, summaries)

    partial = output.with_name(output.name + ".partial")
    try:
        result.to_parquet(partial, index=False)
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    criteria = result["level"] == "criterion"
    with_ambition = result.loc[criteria, "query_text"].str.contains(
        "\n\nAmbitionsniveaus des Kriteriums:", regex=False
    )
    counts = result["level"].value_counts().to_dict()
    logger.info(f"Read {len(queries)} queries from {source}")
    logger.info(f"Rows per level: {counts}")
    logger.info(f"Criteria with ambition levels: {int(with_ambition.sum())}/{int(criteria.sum())}")
    logger.info(f"Wrote {len(result)} queries to {output}")


if __name__ == "__main__":
    main()
