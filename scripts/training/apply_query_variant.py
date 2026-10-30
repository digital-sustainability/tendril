"""Replace prepared training/evaluation query texts using canonical project query IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from prepare_judgments import require_columns

KEYS = ["project_id", "query_id"]


def count_scope_markers(texts: pd.Series) -> int:
    return int(texts.str.contains("Geltungsbereich:", case=False, regex=False).sum())


def replace_query_text(
    frame: pd.DataFrame,
    query_text_column: str,
    modified_queries: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(frame, KEYS + ["level", query_text_column], name)
    mapping = modified_queries[["project_id", "canonical_query_id", "level", "query_text"]].rename(
        columns={"canonical_query_id": "query_id", "query_text": "modified_query_text", "level": "modified_level"}
    )
    if mapping[KEYS].isna().any().any() or mapping.duplicated(KEYS).any():
        raise ValueError("Modified queries must be unique on (project_id, canonical_query_id)")
    if mapping[["modified_level", "modified_query_text"]].isna().any().any():
        raise ValueError("Modified query level/text must be non-null")
    if mapping["modified_query_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Modified query text must not be empty")

    row_order = "__query_variant_row_order"
    if row_order in frame.columns:
        raise ValueError(f"Reserved internal column already exists: {row_order}")
    original = frame.reset_index(drop=True).copy()
    original[row_order] = range(len(original))
    mapped = original.merge(mapping, on=KEYS, how="left", validate="many_to_one", indicator=True, sort=False)
    missing = mapped["_merge"] != "both"
    if missing.any():
        unmapped = mapped.loc[missing, KEYS].drop_duplicates()
        raise ValueError(
            f"{name} has {len(unmapped)} unmapped project/query keys: {unmapped.head(10).to_dict('records')}"
        )
    if not mapped["level"].eq(mapped["modified_level"]).all():
        raise ValueError(f"{name} query levels disagree with modified query metadata")

    old_text = original[query_text_column].fillna("").astype(str)
    new_text = mapped["modified_query_text"].astype(str)
    changed = int(old_text.ne(new_text).sum())
    output = (
        mapped.sort_values(row_order, kind="stable")
        .drop(columns=[row_order, "modified_level", "_merge", query_text_column])
        .rename(columns={"modified_query_text": query_text_column})
    )
    output = output.reset_index(drop=True)
    if len(output) != len(frame) or output[KEYS].ne(frame[KEYS].reset_index(drop=True)).any().any():
        raise RuntimeError(f"{name} row order or IDs changed during query replacement")
    return output, {
        "rows": len(frame),
        "unique_project_query_keys": int(frame[KEYS].drop_duplicates().shape[0]),
        "changed_query_texts": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--modified-queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")

    examples = pd.read_parquet(args.examples)
    validation = pd.read_parquet(args.validation)
    modified_queries = pd.read_parquet(args.modified_queries)
    require_columns(
        modified_queries,
        ["project_id", "canonical_query_id", "level", "query_text"],
        "modified queries",
    )
    examples_variant, examples_report = replace_query_text(examples, "query", modified_queries, "examples")
    validation_variant, validation_report = replace_query_text(validation, "query_text", modified_queries, "validation")
    args.output_dir.mkdir(parents=True)
    examples_variant.to_parquet(args.output_dir / "examples.parquet", index=False)
    validation_variant.to_parquet(args.output_dir / "validation.parquet", index=False)
    report = {
        "modified_queries_source": str(args.modified_queries),
        "join_keys": KEYS,
        "query_rows": len(modified_queries),
        "query_levels": modified_queries["level"].value_counts().to_dict(),
        "examples": examples_report,
        "validation": validation_report,
        "preserved_train_columns": [column for column in examples if column != "query"],
        "query_text_contains_xml_tags": {
            "examples": int(examples_variant["query"].str.contains(r"<[^>]+>", regex=True).sum()),
            "validation": int(validation_variant["query_text"].str.contains(r"<[^>]+>", regex=True).sum()),
        },
        "query_text_contains_scope_marker": {
            "examples": count_scope_markers(examples_variant["query"]),
            "validation": count_scope_markers(validation_variant["query_text"]),
        },
    }
    (args.output_dir / "query_variant_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
