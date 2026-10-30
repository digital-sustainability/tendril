"""Paired bootstrap confidence intervals for the score difference between two systems on TENDRIL.

Both systems are scored on the same resampled data in every iteration, and the
interval is taken over the differences B - A. The statistic is the one reported
for TENDRIL: the mean over projects of the per-project score, where a project's
score is the mean over its queries.

Two resampling units are supported:

- ``project`` (default): a cluster bootstrap that draws projects with replacement.
  Queries of one project share documents, annotators and difficulty, so this is
  the unit that the benchmark generalizes over. It needs only one score per
  project, so MTEB cache directories are enough.
- ``query``: draws queries with replacement within each project and keeps the set
  of projects fixed. It only reflects query sampling and gives narrower intervals.
  It needs per-query scores, i.e. predictions written by
  ``evaluate_models.py --predictions``.

A system is given as a directory that contains either MTEB cache results
(``TENDRIL-<project>.json``, e.g. ``cache/mteb/results/<model>/<revision>``) or
predictions (``TENDRIL-<project>_predictions.json``).

Example::

    python scripts/evaluation/paired_bootstrap.py \\
        cache/mteb/results/BAAI__bge-m3/<revision> \\
        cache/mteb/results/<finetuned model>/<revision> \\
        --metric ndcg_at_10 --metric recall_at_100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval
from loguru import logger

from tendril.hub import TEST_REPO, download_test_set, resolve_revision

TASK_PREFIX = "TENDRIL-"
PREDICTIONS_SUFFIX = "_predictions.json"
SPLIT = "test"
DEFAULT_METRICS = ("ndcg_at_10",)
DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 0
PERCENTILES = (2.5, 97.5)
# MTEB metric prefix -> pytrec_eval measure prefix. MRR is computed separately, as in MTEB.
TREC_MEASURES = {"ndcg": "ndcg_cut", "map": "map_cut", "recall": "recall", "precision": "P"}


@dataclass(frozen=True)
class SystemScores:
    """Scores per project: one value per query (predictions) or a single value (cache)."""

    path: Path
    per_query: bool
    scores: dict[str, dict[str, np.ndarray]]  # project -> metric -> scores


def split_metric(metric: str) -> tuple[str, int]:
    name, separator, k = metric.rpartition("_at_")
    if not separator or not k.isdigit() or (name not in TREC_MEASURES and name != "mrr"):
        raise SystemExit(f"ERROR: unsupported metric {metric}; use e.g. ndcg_at_10, recall_at_100, mrr_at_10")
    return name, int(k)


def load_qrels(repo: str, revision: str) -> dict[str, dict[str, dict[str, int]]]:
    """Return project -> query id -> passage id -> relevance, for every project of the test set."""
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for directory in download_test_set(repo, revision):
        table = pd.read_parquet(directory / "qrels.parquet")
        project: dict[str, dict[str, int]] = {}
        for query_id, corpus_id, score in table.itertuples(index=False):
            project.setdefault(str(query_id), {})[str(corpus_id)] = int(score)
        qrels[directory.name] = project
    return qrels


def reciprocal_rank(run: dict[str, float], relevant: set[str], k: int) -> float:
    # Ties are broken by passage id descending, as in pytrec_eval and MTEB.
    ranked = sorted(run.items(), key=lambda item: (item[1], item[0]), reverse=True)[:k]
    for rank, (passage_id, _) in enumerate(ranked, start=1):
        if passage_id in relevant:
            return 1.0 / rank
    return 0.0


def query_scores(
    qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]], metrics: Sequence[str]
) -> dict[str, np.ndarray]:
    """Score every query of one project; queries without results score 0, as in MTEB's averages."""
    queries = sorted(qrels)
    parsed = {metric: split_metric(metric) for metric in metrics}
    measures = {f"{TREC_MEASURES[name]}.{k}" for name, k in parsed.values() if name != "mrr"}
    trec = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run) if measures else {}
    scores: dict[str, np.ndarray] = {}
    for metric, (name, k) in parsed.items():
        values = []
        for query_id in queries:
            if name == "mrr":
                relevant = {doc for doc, score in qrels[query_id].items() if score > 0}
                values.append(reciprocal_rank(run.get(query_id, {}), relevant, k))
            else:
                values.append(trec.get(query_id, {}).get(f"{TREC_MEASURES[name]}_{k}", 0.0))
        scores[metric] = np.asarray(values, dtype=float)
    return scores


def load_system(
    path: Path, metrics: Sequence[str], qrels: dict[str, dict[str, dict[str, int]]]
) -> SystemScores:
    predictions = {
        file.name.removeprefix(TASK_PREFIX).removesuffix(PREDICTIONS_SUFFIX): file
        for file in path.glob(f"{TASK_PREFIX}*{PREDICTIONS_SUFFIX}")
    }
    cached = {
        file.stem.removeprefix(TASK_PREFIX): file
        for file in path.glob(f"{TASK_PREFIX}*.json")
        if not file.name.endswith(PREDICTIONS_SUFFIX)
    }
    files = predictions or cached
    missing = sorted(set(qrels) - set(files))
    if missing:
        raise SystemExit(f"ERROR: {path} has no results for {len(missing)} test projects: {', '.join(missing)}")
    extra = sorted(set(files) - set(qrels))
    if extra:
        logger.warning(f"Ignoring {len(extra)} tasks in {path} that are not in the test set: {', '.join(extra)}")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for project, project_qrels in qrels.items():
        payload = json.loads(files[project].read_text(encoding="utf-8"))
        if predictions:
            scores[project] = query_scores(project_qrels, payload["default"][SPLIT], metrics)
        else:
            (result,) = payload["scores"][SPLIT]
            missing_metrics = [metric for metric in metrics if metric not in result]
            if missing_metrics:
                raise SystemExit(f"ERROR: {files[project]} has no {', '.join(missing_metrics)}")
            scores[project] = {metric: np.asarray([float(result[metric])]) for metric in metrics}
    return SystemScores(path, bool(predictions), scores)


@dataclass(frozen=True)
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    lower: float
    upper: float
    p_b_not_better: float
    p_two_sided: float


def bootstrap_difference(
    a: list[np.ndarray], b: list[np.ndarray], *, unit: str, iterations: int, rng: np.random.Generator
) -> np.ndarray:
    """Return the bootstrap distribution of mean_project(B) - mean_project(A)."""
    if unit == "project":
        diff = np.asarray([scores_b.mean() - scores_a.mean() for scores_a, scores_b in zip(a, b, strict=True)])
        draws = rng.integers(0, len(diff), size=(iterations, len(diff)))
        return diff[draws].mean(axis=1)
    total = np.zeros(iterations)
    for scores_a, scores_b in zip(a, b, strict=True):
        diff = scores_b - scores_a
        draws = rng.integers(0, len(diff), size=(iterations, len(diff)))
        total += diff[draws].mean(axis=1)
    return total / len(a)


def compare(
    system_a: SystemScores,
    system_b: SystemScores,
    metrics: Sequence[str],
    *,
    unit: str,
    iterations: int,
    seed: int,
) -> list[Comparison]:
    if unit == "query" and not (system_a.per_query and system_b.per_query):
        raise SystemExit("ERROR: --unit query needs predictions for both systems (evaluate_models.py --predictions)")
    projects = sorted(system_a.scores)
    comparisons = []
    for metric in metrics:
        a = [system_a.scores[project][metric] for project in projects]
        b = [system_b.scores[project][metric] for project in projects]
        if system_a.per_query and system_b.per_query:
            for project, scores_a, scores_b in zip(projects, a, b, strict=True):
                if len(scores_a) != len(scores_b):
                    raise SystemExit(f"ERROR: different query sets for {project}")
        mean_a = float(np.mean([scores.mean() for scores in a]))
        mean_b = float(np.mean([scores.mean() for scores in b]))
        # One generator per metric, so every metric is resampled with the same seed.
        differences = bootstrap_difference(a, b, unit=unit, iterations=iterations, rng=np.random.default_rng(seed))
        lower, upper = np.percentile(differences, PERCENTILES)
        p_not_better = float((differences <= 0).mean())
        comparisons.append(
            Comparison(
                metric=metric,
                mean_a=mean_a,
                mean_b=mean_b,
                difference=mean_b - mean_a,
                lower=float(lower),
                upper=float(upper),
                p_b_not_better=p_not_better,
                p_two_sided=min(1.0, 2 * min(p_not_better, float((differences >= 0).mean()))),
            )
        )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired bootstrap CI of the TENDRIL score difference B - A")
    parser.add_argument("system_a", type=Path, help="MTEB cache or predictions directory of system A")
    parser.add_argument("system_b", type=Path, help="MTEB cache or predictions directory of system B")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=f"MTEB metric key, repeatable (default: {', '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--unit",
        choices=("project", "query"),
        default="project",
        help="Resample projects (cluster bootstrap, default) or queries within projects",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repo", default=TEST_REPO, help=f"Hub dataset repo of the test set (default: {TEST_REPO})")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision of the test set; use the one recorded in the score JSON of the evaluation (default: main)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the results")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = tuple(args.metric or DEFAULT_METRICS)
    for metric in metrics:
        split_metric(metric)
    revision = resolve_revision(args.repo, args.revision)
    qrels = load_qrels(args.repo, revision)
    system_a = load_system(args.system_a, metrics, qrels)
    system_b = load_system(args.system_b, metrics, qrels)
    comparisons = compare(
        system_a, system_b, metrics, unit=args.unit, iterations=args.iterations, seed=args.seed
    )

    print(f"A: {system_a.path}\nB: {system_b.path}")
    print(
        f"{len(qrels)} projects, {sum(len(project) for project in qrels.values())} queries, "
        f"unit={args.unit}, {args.iterations} iterations, seed {args.seed}, {args.repo}@{revision[:8]}"
    )
    def p_value(value: float, tails: int) -> str:
        # A share of zero only bounds the p-value by the resolution of the bootstrap; two tails double it.
        return f"<{tails / args.iterations:.0e}" if value == 0 else f"{value:.4f}"

    print(f"{'metric':<18}{'A':>8}{'B':>8}{'B - A':>9}   {'95% CI':<19}{'P(B<=A)':>9}{'p (2-sided)':>13}")
    for item in comparisons:
        interval = f"[{item.lower:+.4f}, {item.upper:+.4f}]"
        print(
            f"{item.metric:<18}{item.mean_a:>8.4f}{item.mean_b:>8.4f}{item.difference:>+9.4f}   {interval:<19}"
            f"{p_value(item.p_b_not_better, 1):>9}{p_value(item.p_two_sided, 2):>13}"
        )
    if args.output is not None:
        payload = {
            "system_a": str(system_a.path),
            "system_b": str(system_b.path),
            "dataset": {"repo": args.repo, "revision": revision},
            "unit": args.unit,
            "iterations": args.iterations,
            "seed": args.seed,
            "percentiles": list(PERCENTILES),
            "comparisons": [item.__dict__ for item in comparisons],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
