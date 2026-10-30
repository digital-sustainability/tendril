"""Join judge verdicts with queries and anonymized passages (never the original passage texts).

Accepts the released (query_id, document_id, score) schema or normalized
(project_id, query_id, passage_uid, score) verdicts. Released query IDs use
catalog::HF-nnn / catalog::K-nnn; original query IDs are kept as source_query_id.

Usage:
    python scripts/training/prepare_judgments.py --judgments verdicts.parquet \
        --output data/derived/training/labeled.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PAIR_KEYS = ["project_id", "query_id", "passage_uid"]
DEFAULT_INPUTS = Path("data/derived/judge")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def canonical_query_ids(queries: pd.DataFrame) -> pd.Series:
    catalogs = queries["catalog"].str.replace(r"^WDG[_-]\d+[_-]", "", regex=True)
    suffixes = queries["query_id"].str.extract(r"(?i)(HF-\d+|K-\d+)$", expand=False).str.upper()
    if suffixes.isna().any() or catalogs.isna().any():
        raise ValueError("Cannot map an original query ID to a canonical judge ID")
    expected = queries["level"].map({"area_of_action": "HF-", "criterion": "K-"})
    if expected.isna().any() or not all(
        suffix.startswith(prefix) for suffix, prefix in zip(suffixes, expected, strict=True)
    ):
        raise ValueError("Canonical query suffix does not match its query level")
    return catalogs + "::" + suffixes


def prepare_judgments(inputs: Path, judgments: pd.DataFrame, allow_missing: bool = False) -> pd.DataFrame:
    released_schema = "document_id" in judgments.columns
    if released_schema:
        require_columns(judgments, ["query_id", "document_id", "score"], "judgments")
        if "project_id" in judgments or "passage_uid" in judgments:
            raise ValueError("Ambiguous mixture of released and normalized verdict schemas")
    else:
        require_columns(judgments, PAIR_KEYS + ["score"], "judgments")
    passages = pd.read_parquet(inputs / "passages_anon.parquet")
    require_columns(passages, ["project_id", "passage_uid", "file_name", "text"], "anonymized passages")
    if passages["passage_uid"].isna().any() or not passages["passage_uid"].is_unique:
        raise ValueError("Anonymized passage IDs must be globally unique and non-null")
    if released_schema:
        judgments = (
            judgments[["query_id", "document_id", "score"]]
            .rename(columns={"document_id": "passage_uid"})
            .merge(passages[["passage_uid", "project_id"]], on="passage_uid", how="left", validate="many_to_one")
        )
    if judgments[PAIR_KEYS].isna().any().any() or judgments.duplicated(PAIR_KEYS).any():
        raise ValueError("judgments contain null, unknown passage, or duplicate pair IDs")
    scores = pd.to_numeric(judgments["score"], errors="coerce")
    if not scores.isin([0, 1, 2]).all():
        raise ValueError("judgments must have a score of 0, 1, or 2 for every pair")

    pairs = pd.read_parquet(inputs / "pairs.parquet")
    queries = pd.read_parquet(inputs / "queries.parquet")
    require_columns(pairs, PAIR_KEYS + ["level", "tier"], "pairs")
    require_columns(queries, ["project_id", "query_id", "level", "query_text", "catalog"], "queries")
    allowed_projects = set(judgments["project_id"])
    pairs = pairs[pairs["project_id"].isin(allowed_projects)].copy()
    queries = queries[queries["project_id"].isin(allowed_projects)].copy()
    if released_schema:
        queries["source_query_id"] = queries["query_id"]
        queries["query_id"] = canonical_query_ids(queries)
        pairs = pairs.rename(columns={"query_id": "source_query_id"}).merge(
            queries[["project_id", "source_query_id", "query_id"]],
            on=["project_id", "source_query_id"],
            how="left",
            validate="many_to_one",
        )
        if pairs["query_id"].isna().any():
            raise ValueError("Pairs reference a query without canonical metadata")
    if pairs.duplicated(PAIR_KEYS).any():
        raise ValueError("pairs contain duplicate pair IDs")

    labeled = pairs[PAIR_KEYS + ["level", "tier"]].merge(
        judgments[PAIR_KEYS].assign(score=scores.to_numpy()),
        on=PAIR_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if int((labeled["_merge"] == "both").sum()) != len(judgments):
        raise ValueError("judgments have unknown pair IDs")
    missing_pairs = labeled.loc[labeled["_merge"] != "both", PAIR_KEYS].to_dict("records")
    if missing_pairs and not allow_missing:
        raise ValueError(f"{len(missing_pairs)} pairs without judgments; audit before using --allow-missing-judgments")
    labeled = labeled[labeled["_merge"] == "both"].copy()
    labeled = labeled.drop(columns="_merge")
    labeled["score"] = labeled["score"].astype("int8")
    query_columns = ["project_id", "query_id", "query_text", "catalog"]
    if released_schema:
        query_columns.append("source_query_id")
    labeled = labeled.merge(
        queries[query_columns],
        on=["project_id", "query_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        passages[["project_id", "passage_uid", "file_name", "text"]],
        on=["project_id", "passage_uid"],
        how="left",
        validate="many_to_one",
    )
    if labeled[["query_text", "catalog", "file_name", "text"]].isna().any().any():
        raise ValueError("pairs reference missing query or passage metadata")
    labeled.attrs["missing_pairs"] = missing_pairs
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-projects", type=int, help="Require the released project count, currently 631")
    parser.add_argument(
        "--allow-missing-judgments", action="store_true", help="Explicitly exclude unjudged pairs and write an ID audit"
    )
    args = parser.parse_args()

    labeled = prepare_judgments(args.inputs_dir, pd.read_parquet(args.judgments), args.allow_missing_judgments)
    if args.expected_projects is not None and labeled["project_id"].nunique() != args.expected_projects:
        raise ValueError("Judged project count differs from the expected release")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = pd.DataFrame(labeled.attrs["missing_pairs"], columns=PAIR_KEYS)
    if not missing.empty:
        audit_path = args.output.with_suffix(".missing.parquet")
        missing.to_parquet(audit_path, index=False)
        print(f"Excluded {len(missing):,} unjudged pairs; ID audit: {audit_path}")
    labeled.to_parquet(args.output, index=False)
    print(labeled.groupby(["level", "score"]).size().to_string())
    print(f"Projects: {labeled['project_id'].nunique():,}; source: passages_anon.parquet")
    print(f"Wrote {len(labeled):,} labeled pairs to {args.output}")


if __name__ == "__main__":
    main()
