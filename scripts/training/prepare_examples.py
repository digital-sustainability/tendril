"""Reserve validation projects, then build explicit-negative examples from judged pairs.

Usage:
    python scripts/training/prepare_examples.py --reserve-from-labeled LABELED --split SPLIT --cover-catalogs
    python scripts/training/prepare_examples.py --labeled LABELED --split SPLIT --output-dir OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_judgments import PAIR_KEYS, require_columns


def select_with_catalog_coverage(
    queries: pd.DataFrame, projects: list[str], count: int, rng: np.random.Generator
) -> list[str]:
    """Pick validation projects so every catalog keeps at least one project in training and one in validation."""
    require_columns(queries, ["project_id", "catalog"], "catalog memberships")
    if queries[["project_id", "catalog"]].isna().any().any():
        raise ValueError("Catalog memberships must be non-null")
    memberships = queries[["project_id", "catalog"]].drop_duplicates()
    by_project = memberships.groupby("project_id")["catalog"].agg(set).to_dict()
    owners = memberships.groupby("catalog")["project_id"].agg(set).to_dict()
    if any(len(group) < 2 for group in owners.values()):
        raise ValueError("Each catalog needs at least two projects to cover both splits")
    priority = rng.permutation(projects).tolist()
    selected: set[str] = set()
    uncovered = set(owners)

    def can_select(project: str) -> bool:
        return all(bool(owners[catalog] - selected - {project}) for catalog in by_project[project])

    # Cover the rarest catalog first, preferring projects that also cover other still-uncovered catalogs.
    while uncovered:
        rarest = min(uncovered, key=lambda catalog: (len(owners[catalog]), catalog))
        candidates = [
            project
            for project in priority
            if project in owners[rarest] and project not in selected and can_select(project)
        ]
        if not candidates:
            raise ValueError("Cannot cover catalogs in both splits with this selection; inspect memberships")
        project = max(candidates, key=lambda candidate: len(by_project[candidate] & uncovered))
        selected.add(project)
        uncovered -= by_project[project]
        if len(selected) > count:
            raise ValueError("Validation size is too small for catalog coverage")
    for project in priority:
        if len(selected) == count:
            break
        if project not in selected and can_select(project):
            selected.add(project)
    if len(selected) != count:
        raise ValueError("Cannot fill validation size while retaining all training catalogs")
    return sorted(selected)


def select_with_sector_coverage(
    projects: list[str], sectors: pd.DataFrame, count: int, rng: np.random.Generator
) -> list[str]:
    require_columns(sectors, ["project_id", "sector"], "sectors")
    if sectors.duplicated("project_id").any():
        raise ValueError("Sector map must have one row per project")
    table = pd.DataFrame({"project_id": projects}).merge(
        sectors[["project_id", "sector"]], on="project_id", how="left", validate="one_to_one"
    )
    if table["sector"].isna().any():
        raise ValueError("Sector map does not cover every project")
    groups = [rng.permutation(group["project_id"]).tolist() for _, group in table.groupby("sector")]
    if count < len(groups):
        raise ValueError("Validation size is too small to cover every sector")
    selected = [group.pop() for group in groups]
    remaining = [project for group in groups for project in group]
    return selected + rng.permutation(remaining)[: count - len(selected)].tolist()


def reserve_projects(
    queries: pd.DataFrame,
    fraction: float,
    seed: int,
    sectors: pd.DataFrame | None = None,
    cover_catalogs: bool = False,
) -> dict[str, list[str]]:
    """Split whole projects into training and validation; no passage of a project ever crosses the split."""
    projects = sorted(queries["project_id"].unique().tolist())
    if not 0 < fraction < 1 or len(projects) < 2:
        raise ValueError("Need at least two projects and a validation fraction between 0 and 1")
    count = max(1, min(len(projects) - 1, round(len(projects) * fraction)))
    rng = np.random.default_rng(seed)
    if cover_catalogs:
        if sectors is not None:
            raise ValueError("Choose catalog coverage or sector selection, not both")
        selected = select_with_catalog_coverage(queries, projects, count, rng)
    elif sectors is None:
        selected = rng.permutation(projects)[:count].tolist()
    else:
        selected = select_with_sector_coverage(projects, sectors, count, rng)
    validation = set(selected)
    return {"train": sorted(set(projects) - validation), "validation": sorted(validation)}


def complete_validation(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only queries judged against every passage of their project; return (kept rows, excluded groups)."""
    corpus_sizes = validation.groupby("project_id")["passage_uid"].nunique()
    judged_sizes = validation.groupby(["project_id", "query_id"])["passage_uid"].transform("nunique")
    complete = judged_sizes.eq(validation["project_id"].map(corpus_sizes))
    excluded = validation.loc[~complete, ["project_id", "query_id"]].drop_duplicates()
    return validation[complete].copy(), excluded


def sample_negatives(
    positive: dict, negatives: pd.DataFrame, same_document: int, other_document: int, rng: np.random.Generator
) -> list[dict] | None:
    """Draw distinct judged negatives, same-document first; None if the query has too few."""
    same = negatives[negatives["file_name"] == positive["file_name"]]
    other = negatives[negatives["file_name"] != positive["file_name"]]
    chosen: list[dict] = []
    for pool, requested in ((same, same_document), (other, other_document)):
        if requested and len(pool):
            chosen += pool.iloc[rng.permutation(len(pool))[:requested]].to_dict("records")
    remaining = negatives[~negatives["passage_uid"].isin([row["passage_uid"] for row in chosen])]
    deficit = same_document + other_document - len(chosen)
    chosen += remaining.iloc[rng.permutation(len(remaining))[:deficit]].to_dict("records")
    return chosen if len(chosen) == same_document + other_document else None


def build_examples(
    labeled: pd.DataFrame, split: dict[str, list[str]], same_document: int, other_document: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build training rows (query, positive, negatives) and the project-scoped validation pairs."""
    require_columns(
        labeled,
        PAIR_KEYS + ["score", "level", "tier", "query_text", "text", "file_name", "catalog"],
        "labeled pairs",
    )
    if same_document < 0 or other_document < 0 or same_document + other_document == 0:
        raise ValueError("Explicit-negative loss needs at least one negative")
    if labeled.duplicated(PAIR_KEYS).any() or not labeled["score"].isin([0, 1, 2]).all():
        raise ValueError("Labeled pairs must be unique with scores 0, 1, or 2")
    if not labeled["level"].isin(["area_of_action", "criterion"]).all():
        raise ValueError("Unknown query level")
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    if train_ids & validation_ids or set(labeled["project_id"]) != train_ids | validation_ids:
        raise ValueError("Split must cover exactly the labeled projects without overlap")
    if labeled[["query_text", "text", "file_name"]].isna().any().any():
        raise ValueError("Labeled pairs have missing text or provenance")

    validation = labeled[labeled["project_id"].isin(validation_ids)].copy()
    train = labeled[labeled["project_id"].isin(train_ids)].copy()
    train = train[~train["text"].isin(set(validation["text"]))].copy()
    rows_before = len(validation)
    validation, excluded_queries = complete_validation(validation)
    validation.attrs["excluded_query_groups"] = excluded_queries.to_dict("records")
    if validation.empty:
        raise ValueError("No complete validation queries remain")

    label_counts = train.groupby(["query_id", "text"])["score"].transform("nunique")
    conflicts = int((label_counts > 1).sum())
    train = train[label_counts == 1]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    skipped = 0
    for _, group in train.groupby(["project_id", "query_id"], sort=True):
        if group["query_text"].nunique() != 1 or group["level"].nunique() != 1:
            raise ValueError("Inconsistent query metadata within project")
        positives = group[(group["score"] == 2) & (group["tier"] != "unsure")]
        negatives = group[group["score"] == 0].drop_duplicates("text")
        for positive in positives.to_dict("records"):
            chosen = sample_negatives(positive, negatives, same_document, other_document, rng)
            if chosen is None:
                skipped += 1
                continue
            row = {key: positive[key] for key in ["project_id", "query_id", "level", "catalog"]}
            row.update(query=positive["query_text"], positive=positive["text"], positive_uid=positive["passage_uid"])
            for index, negative in enumerate(chosen, start=1):
                row[f"negative_{index}"] = negative["text"]
                row[f"negative_uid_{index}"] = negative["passage_uid"]
            rows.append(row)
    if not rows:
        raise ValueError("No eligible examples with enough distinct judged negatives")
    report = {
        "conflicting_rows_removed": conflicts,
        "insufficient_negative_positives": skipped,
        "excluded_validation_query_groups": len(excluded_queries),
        "excluded_validation_rows": rows_before - len(validation),
    }
    return pd.DataFrame(rows), validation, report


def reserve_split(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.reserve_from_labeled is None:
        parser.error(
            "Reserve the approved projects with --reserve-from-labeled; original inputs include excluded projects"
        )
    if args.split.exists():
        raise ValueError("Split already exists; refusing to overwrite reserved projects")
    columns = ["project_id", "catalog"] if args.cover_catalogs else ["project_id"]
    queries = pd.read_parquet(args.reserve_from_labeled, columns=columns)
    sectors = pd.read_csv(args.sectors, dtype=str) if args.sectors else None
    split = reserve_projects(queries, args.validation_fraction, args.seed, sectors, args.cover_catalogs)
    args.split.parent.mkdir(parents=True, exist_ok=True)
    args.split.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Reserved {len(split['validation'])} validation projects; {len(split['train'])} training projects")
    if args.cover_catalogs:
        print(f"Catalog coverage: all {queries['catalog'].nunique()} catalogs in both splits; not sector-stratified")
    elif sectors is None:
        print("No sector map provided: random project split, not sector-stratified")


def create_examples(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.output_dir is None:
        parser.error("--output-dir is required with --labeled")
    if args.output_dir.exists():
        raise ValueError("Use a fresh output directory; existing training examples will not be overwritten")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    examples, validation, report = build_examples(
        pd.read_parquet(args.labeled), split, args.same_document_negatives, args.other_document_negatives, args.seed
    )
    args.output_dir.mkdir(parents=True)
    examples.to_parquet(args.output_dir / "examples.parquet", index=False)
    validation.to_parquet(args.output_dir / "validation.parquet", index=False)
    pd.DataFrame(validation.attrs["excluded_query_groups"], columns=["project_id", "query_id"]).to_parquet(
        args.output_dir / "validation_excluded_queries.parquet", index=False
    )
    report.update(training_examples=len(examples), validation_pairs=len(validation))
    (args.output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(examples.groupby("level").size().to_string())
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reserve-from-labeled", type=Path, help="Audited, approved labeled release to split")
    parser.add_argument("--sectors", type=Path, help="Optional CSV with project_id and sector")
    parser.add_argument("--cover-catalogs", action="store_true", help="Cover all catalogs in both splits (not sectors)")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled", type=Path, help="Labeled pairs; builds examples instead of reserving a split")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--same-document-negatives", type=int, default=2)
    parser.add_argument("--other-document-negatives", type=int, default=1)
    args = parser.parse_args()
    if args.labeled is None:
        reserve_split(args, parser)
    else:
        create_examples(args, parser)


if __name__ == "__main__":
    main()
