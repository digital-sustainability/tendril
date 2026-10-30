"""Select (project, document, area of action, criterion) tuples for the LLM judge.

Selection proceeds per project:

    1. catalog gate   catalog_relevance == relevant, plus the always-on catalog
    2. area quota     stratified over area_relevance_verdicts x validated_match
    3. criteria       ranked within each selected area by retrieved.max
    4. documents      ranked by their own matching_statistics for the selected areas

Then another pass tops up criteria that fall below the coverage floor

Usage:
    python scripts/train_set/select_training_units.py
    python scripts/train_set/select_training_units.py --config scripts/train_set/config.yaml
    python scripts/train_set/select_training_units.py --n-projects 500 --limit 50
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
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

# The Road Transport catalog shipped a few areas as "-H-nnn" instead of "-HF-nnn".
MALFORMED_AREA_SUFFIX = re.compile(r"-H-(\d+)$")

# Virtual (area-level) criteria are identified by an HF suffix in the criterion ID.
VIRTUAL_CRITERION_ID = re.compile(r"-HF-\d+$")

TIERS = ("relevant_validated", "relevant_unvalidated", "unsure", "irrelevant")


@dataclass(frozen=True)
class AreaCandidate:
    area_id: str
    catalog: str
    verdict: str
    tier: str
    validated_max: float | None
    validated_n: int
    retrieved_max: float | None
    retrieved_n: int

    def rank_key(self) -> tuple[float, float, int]:
        """Validated evidence first, then retrieval strength."""
        return (
            self.validated_max if self.validated_max is not None else -1.0,
            self.retrieved_max if self.retrieved_max is not None else -1.0,
            self.validated_n,
        )


@dataclass
class Stats:
    projects_seen: int = 0
    projects_eligible: int = 0
    projects_blocked: int = 0
    projects_selected: int = 0
    tier_shortfall: Counter[str] = field(default_factory=Counter)
    areas_without_criteria: int = 0
    coverage_topups: int = 0


def normalise_area_id(area_id: str) -> str:
    return MALFORMED_AREA_SUFFIX.sub(r"-HF-\1", area_id)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_project(path: Path, dctx: zstandard.ZstdDecompressor) -> dict:
    with path.open("rb") as fh:
        return orjson.loads(dctx.stream_reader(fh).read())


def load_excluded_criteria(path: Path, recommendations: Sequence[str]) -> set[str]:
    # Version drift elimination
    if not path.is_file():
        logger.warning(f"{path} not found - no version exclusions applied")
        return set()
    frame = pd.read_excel(path, sheet_name="criteria")
    drop = frame[frame["recommendation"].isin(recommendations)]
    excluded = set(drop["qdrant_criterion_id"].dropna().astype(str))
    logger.info(f"Excluding {len(excluded)} criteria flagged {list(recommendations)}")
    return excluded


def load_payload_index(path: Path) -> tuple[dict[str, list[str]], set[str]]:
    #Map area key -> criterion IDs, and collect the virtual IDs.
    payload = orjson.loads(path.read_bytes())
    by_area: dict[str, list[str]] = defaultdict(list)
    virtual: set[str] = set()
    for point in payload["points"]:
        item = point["payload"]
        area_key = normalise_area_id(f"{item['catalog']}-{item['scope_of_action_id']}")
        if item.get("is_virtual_scope_criterion"):
            virtual.add(area_key)
        else:
            by_area[area_key].append(str(item["id"]))
    logger.info(f"Payloads: {len(by_area)} areas with criteria, {len(virtual)} virtual area criteria")
    return dict(by_area), virtual


def area_candidates(
    payload: dict,
    allowed_catalogs: set[str],
    excluded_criteria: set[str],
    allowed_areas: set[str] | None = None,
) -> list[AreaCandidate]:
    stats = (payload.get("matching_statistics") or {}).get("by_area_of_action") or {}
    candidates: list[AreaCandidate] = []

    for verdict in payload.get("area_relevance_verdicts") or []:
        catalog = str(verdict.get("catalog_name") or "")
        if catalog not in allowed_catalogs:
            continue
        area_id = normalise_area_id(f"{catalog}-{verdict.get('area_of_action_id')}")
        if allowed_areas is not None and area_id not in allowed_areas:
            continue
        classification = str(verdict.get("classification") or "")

        area_stats = stats.get(area_id) or {}
        retrieved = area_stats.get("retrieved") or {}
        validated = area_stats.get("validated_match") or {}

        if classification == "relevant":
            tier = "relevant_validated" if validated else "relevant_unvalidated"
        elif classification == "unsure":
            tier = "unsure"
        elif classification == "irrelevant":
            tier = "irrelevant"
        else:
            continue

        candidates.append(
            AreaCandidate(
                area_id=area_id,
                catalog=catalog,
                verdict=classification,
                tier=tier,
                validated_max=validated.get("max"),
                validated_n=int(validated.get("n") or 0),
                retrieved_max=retrieved.get("max"),
                retrieved_n=int(retrieved.get("n") or 0),
            )
        )
    return candidates


def select_areas(
    candidates: Sequence[AreaCandidate],
    quota: dict[str, int],
    rng: random.Random,
    stats: Stats,
) -> list[AreaCandidate]:
    """Draw the stratified area quota, best-ranked first within each tier."""
    by_tier: dict[str, list[AreaCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_tier[candidate.tier].append(candidate)

    selected: list[AreaCandidate] = []
    for tier in TIERS:
        want = int(quota.get(tier, 0))
        if want <= 0:
            continue
        pool = by_tier.get(tier, [])
        if tier == "irrelevant":
            # No statistics exist for irrelevant areas, so rank is meaningless.
            rng.shuffle(pool)
            chosen = pool[:want]
        else:
            chosen = sorted(pool, key=AreaCandidate.rank_key, reverse=True)[:want]
        if len(chosen) < want:
            stats.tier_shortfall[tier] += want - len(chosen)
        selected.extend(chosen)
    return selected


def rank_criteria_by_area(
    payload: dict,
    excluded: set[str],
) -> dict[str, list[str]]:
    # All criteria of the project, grouped by area and ranked by retrieved.max.
    by_criterion = (payload.get("matching_statistics") or {}).get("by_criterion") or {}
    scored: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for criterion_id, entry in by_criterion.items():
        if criterion_id in excluded:
            continue
        if VIRTUAL_CRITERION_ID.search(criterion_id):
            continue
        area_id = normalise_area_id(str(entry.get("area_of_action_id") or ""))
        retrieved = entry.get("retrieved") or {}
        scored[area_id].append((float(retrieved.get("max") or 0.0), str(criterion_id)))
    return {
        area_id: [criterion_id for _, criterion_id in sorted(entries, reverse=True)]
        for area_id, entries in scored.items()
    }


def select_criteria(
    ranked_by_area: dict[str, list[str]],
    areas: Sequence[str],
    per_area: int,
    excluded: set[str],
    payload_by_area: dict[str, list[str]],
    stats: Stats,
) -> dict[str, list[str]]:
    if per_area <= 0:
        return {}

    result: dict[str, list[str]] = {}
    for area_id in areas:
        chosen = ranked_by_area.get(area_id, [])[:per_area]
        if not chosen:
            # Irrelevant areas carry no statistics; fall back to the catalog export.
            fallback = [c for c in payload_by_area.get(area_id, []) if c not in excluded]
            chosen = fallback[:per_area]
            if not chosen:
                stats.areas_without_criteria += 1
        result[area_id] = chosen
    return result


def select_documents(
    payload: dict,
    areas: Sequence[AreaCandidate],
    n_documents: int,
    min_passages: int,
) -> list[tuple[str, float, int]]:
    # Rank documents for selected areas
    wanted = {area.area_id for area in areas}
    ranked: list[tuple[float, int, str]] = []

    for document in payload.get("documents") or []:
        passages = [
            p for p in (document.get("content") or [])
            if str(p.get("content_cleaned") or "").strip()
        ]
        if len(passages) < min_passages:
            continue

        doc_stats = (document.get("matching_statistics") or {}).get("by_area_of_action") or {}
        best = 0.0
        for area_id, entry in doc_stats.items():
            if normalise_area_id(area_id) not in wanted:
                continue
            validated = (entry.get("validated_match") or {}).get("max")
            retrieved = (entry.get("retrieved") or {}).get("max")
            # A validated match outranks any bare retrieval score.
            score = (validated + 1.0) if validated is not None else float(retrieved or 0.0)
            best = max(best, score)
        ranked.append((best, len(passages), str(document.get("file_name") or "")))

    ranked.sort(reverse=True)
    return [(name, score, n) for score, n, name in ranked[:n_documents]]


def apply_coverage_floor(
    rows: list[dict],
    available: dict[str, list[tuple[str, str]]],
    criteria_cache: dict[str, dict[str, list[str]]],
    floor: int,
    max_per_project: int,
    per_area: int,
    rng: random.Random,
    stats: Stats,
) -> list[dict]:
    if floor <= 0:
        return rows

    seen: dict[str, set[str]] = defaultdict(set)
    per_project: Counter[str] = Counter()
    for row in rows:
        seen[row["area_id"]].add(row["project_id"])

    extra: list[dict] = []
    for area_id, entries in available.items():
        missing = floor - len(seen.get(area_id, set()))
        if missing <= 0:
            continue
        pool = [e for e in entries if e[0] not in seen.get(area_id, set())]
        rng.shuffle(pool)
        for project_id, verdict in pool:
            if missing <= 0:
                break
            if per_project[project_id] >= max_per_project:
                continue
            extra.append(
                {
                    "project_id": project_id,
                    "area_id": area_id,
                    "catalog": area_id.rsplit("-HF-", 1)[0],
                    "verdict": verdict,
                    "tier": "coverage",
                    "validated_max": None,
                    "retrieved_max": None,
                    "criterion_ids": criteria_cache.get(project_id, {}).get(area_id, [])[:per_area],
                }
            )
            seen[area_id].add(project_id)
            per_project[project_id] += 1
            stats.coverage_topups += 1
            missing -= 1
    return rows + extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-projects", type=int, default=None, help="Override selection.n_projects")
    parser.add_argument(
        "--n-documents", type=int, default=None, help="Override selection.n_documents_per_project"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only scan this many project files (debug)")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    paths = config["paths"]
    selection = config["selection"]
    eligibility = config["eligibility"]
    exclusions = config["exclusions"]

    n_projects = args.n_projects or int(selection["n_projects"])
    n_documents = args.n_documents or int(selection["n_documents_per_project"])
    quota = dict(selection["area_quota"])
    rng = random.Random(int(config.get("random_seed", 42)))

    excluded = load_excluded_criteria(
        Path(paths["catalog_comparison"]), exclusions["comparison_recommendations"]
    )
    payload_by_area, virtual_areas = load_payload_index(Path(paths["criteria_payloads"]))

    blocklist_path = Path(exclusions["test_project_ids"])
    blocklist = {
        line.strip()
        for line in blocklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    logger.info(f"Blocklist: {len(blocklist)} test projects")

    always_on = str(eligibility["always_include_catalog"])
    min_relevant = int(eligibility["min_relevant_areas"])
    min_passages = int(eligibility["min_passages"])

    allowed_areas: set[str] | None = None
    if eligibility.get("require_virtual_area_criterion"):
        # An area without a virtual criterion has no area-level query text, so it is
        # dropped outright rather than contributing criterion-level queries alone.
        allowed_areas = virtual_areas
        logger.info(
            f"Restricting to {len(allowed_areas)} areas with a virtual criterion "
            f"({len(payload_by_area) - len(allowed_areas)} dropped)"
        )

    project_dir = Path(paths["enriched_projects"])
    files = sorted(project_dir.glob("*.json.zst"))
    if args.limit:
        files = files[: args.limit]
    logger.info(f"Scanning {len(files)} projects in {project_dir}")

    dctx = zstandard.ZstdDecompressor()
    stats = Stats()
    scanned: list[dict] = []

    for path in tqdm(files, desc="scanning"):
        project_id = path.name.removesuffix(".json.zst")
        stats.projects_seen += 1
        if project_id in blocklist:
            stats.projects_blocked += 1
            continue

        payload = load_project(path, dctx)
        if int(payload.get("passage_count") or 0) < min_passages:
            continue

        allowed = {
            str(entry.get("catalog_name"))
            for entry in (payload.get("catalog_relevance") or [])
            if str(entry.get("classification")) == "relevant"
        }
        allowed.add(always_on)

        candidates = area_candidates(payload, allowed, excluded, allowed_areas)
        n_relevant = sum(1 for c in candidates if c.verdict == "relevant")
        if n_relevant < min_relevant:
            continue

        stats.projects_eligible += 1
        scanned.append(
            {
                "project_id": project_id,
                "path": path,
                "candidates": candidates,
                "n_validated": sum(1 for c in candidates if c.tier == "relevant_validated"),
            }
        )

    logger.info(f"Eligible: {stats.projects_eligible} (blocked {stats.projects_blocked})")

    # Rank by validated evidence
    scanned.sort(key=lambda item: -item["n_validated"])
    chosen_projects = scanned[:n_projects]
    stats.projects_selected = len(chosen_projects)

    area_rows: list[dict] = []
    document_rows: list[dict] = []
    available: dict[str, list[tuple[str, str]]] = defaultdict(list)
    criteria_cache: dict[str, dict[str, list[str]]] = {}

    for item in tqdm(chosen_projects, desc="selecting"):
        payload = load_project(item["path"], dctx)
        project_id = item["project_id"]

        ranked_by_area = rank_criteria_by_area(payload, excluded)
        criteria_cache[project_id] = ranked_by_area

        # Only areas the LLM did not reject are eligible for coverage top-ups.
        for candidate in item["candidates"]:
            if candidate.verdict != "irrelevant":
                available[candidate.area_id].append((project_id, candidate.verdict))

        areas = select_areas(item["candidates"], quota, rng, stats)
        criteria = select_criteria(
            ranked_by_area,
            [a.area_id for a in areas],
            int(selection["n_criteria_per_area"]),
            excluded,
            payload_by_area,
            stats,
        )
        documents = select_documents(payload, areas, n_documents, min_passages)

        for area in areas:
            area_rows.append(
                {
                    "project_id": project_id,
                    "area_id": area.area_id,
                    "catalog": area.catalog,
                    "verdict": area.verdict,
                    "tier": area.tier,
                    "validated_max": area.validated_max,
                    "retrieved_max": area.retrieved_max,
                    "criterion_ids": criteria.get(area.area_id, []),
                }
            )
        for rank, (file_name, score, n_passages) in enumerate(documents):
            document_rows.append(
                {
                    "project_id": project_id,
                    "file_name": file_name,
                    "rank": rank,
                    "score": score,
                    "n_passages": n_passages,
                }
            )

    coverage = config["coverage"]
    area_rows = apply_coverage_floor(
        area_rows,
        available,
        criteria_cache,
        int(coverage["min_projects_per_area"]),
        int(coverage.get("max_extra_areas_per_project", 2)),
        int(selection["n_criteria_per_area"]),
        rng,
        stats,
    )

    areas_df = pd.DataFrame(area_rows)
    docs_df = pd.DataFrame(document_rows)

    output_dir = args.output_dir or Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    areas_df.to_parquet(output_dir / "selected_areas.parquet", index=False)
    docs_df.to_parquet(output_dir / "selected_documents.parquet", index=False)

    n_queries = len(areas_df) + int(areas_df["criterion_ids"].map(len).sum())
    n_passages = int(docs_df["n_passages"].sum())

    logger.info("--- selection summary ---")
    logger.info(f"  projects selected     {stats.projects_selected}")
    logger.info(f"  areas selected        {len(areas_df)}")
    logger.info(f"  criteria selected     {int(areas_df['criterion_ids'].map(len).sum())}")
    logger.info(f"  documents selected    {len(docs_df)}")
    logger.info(f"  passages in documents {n_passages:,}")
    logger.info(f"  coverage top-ups      {stats.coverage_topups}")
    logger.info(f"  areas w/o criteria    {stats.areas_without_criteria}")
    if stats.tier_shortfall:
        logger.warning(f"  tier shortfalls       {dict(stats.tier_shortfall)}")
    logger.info(f"  tier distribution     {areas_df['tier'].value_counts().to_dict()}")
    logger.info(f"  ESTIMATED PAIRS       {n_passages * (n_queries / max(stats.projects_selected, 1)):,.0f}")
    logger.info(f"Wrote {output_dir}/selected_areas.parquet and selected_documents.parquet")


if __name__ == "__main__":
    main()
