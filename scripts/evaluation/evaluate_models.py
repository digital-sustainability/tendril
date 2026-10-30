import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import mteb
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.cache import ResultCache
from mteb.models.model_meta import ModelMeta
from mteb.results.task_result import TaskError, TaskResult
from rich import box
from rich.console import Console
from rich.table import Table

from tendril.hub import TEST_REPO, resolve_revision
from tendril.inference.settings import repo_root
from tendril.models import (
    KNOWN_MODELS,
    MODEL_GROUPS,
    ORACLE_NAME,
    load_model,
    models_in_group,
    output_stem,
    require_env,
    resolve_model_name,
)
from tendril.tasks import load_hub_tasks

console = Console()


REPO_ROOT = repo_root()
DEFAULT_MODEL_NAME = "mteb/baseline-bm25s"
DEFAULT_CACHE_PATH = REPO_ROOT / "cache" / "mteb"
K_VALUES = (10, 25, 50, 100)
# MTEB stores Recall@1 as "accuracy" and crashes if that cutoff is absent.
_EVAL_K_VALUES = (1, *K_VALUES)
METRICS = ("NDCG", "Precision", "Recall", "F1", "MAP", "MRR")
_MTEB_METRIC_KEYS = {
    "NDCG": "ndcg",
    "Precision": "precision",
    "Recall": "recall",
    "MAP": "map",
    "MRR": "mrr",
}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (2.5, 97.5)


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def print_known_models() -> None:
    table = Table(title="Reviewed TENDRIL models", box=box.ROUNDED, header_style="bold")
    table.add_column("Group", no_wrap=True)
    table.add_column("Model")
    table.add_column("Aliases")
    table.add_column("Env")
    table.add_column("Notes")
    for model in KNOWN_MODELS:
        table.add_row(
            model.group,
            model.name,
            ", ".join(model.aliases) or "—",
            ", ".join(model.env_vars) or "—",
            model.notes,
        )
    console.print(table)


def sector_from_task_name(name: str) -> str:
    rest = name.removeprefix("TENDRIL-")
    sector, separator, suffix = rest.rpartition("-")
    if separator and suffix.isalnum():
        return sector
    return rest


def model_meta_of(model: object) -> ModelMeta:
    if isinstance(model, ModelMeta):
        return model
    meta = getattr(model, "mteb_model_meta", None)
    if isinstance(meta, ModelMeta):
        return meta
    raise TypeError(f"Cannot read MTEB model metadata from {type(model).__name__}")


def cached_task_names(
    tasks: Sequence[AbsTask],
    meta: ModelMeta,
    cache: ResultCache | None = None,
) -> list[str]:
    result_cache = cache if cache is not None else ResultCache(cache_path=DEFAULT_CACHE_PATH)
    names: list[str] = []
    for task in tasks:
        task_name = task.metadata.name
        if result_cache.get_task_result_path(task_name, meta).is_file():
            names.append(task_name)
    return names


def warn_if_cached_results(
    tasks: Sequence[AbsTask],
    meta: ModelMeta,
    cache: ResultCache,
) -> None:
    cached = cached_task_names(tasks, meta, cache)
    if not cached:
        return
    logger.warning(
        f"Cached MTEB results found for {len(cached)}/{len(tasks)} tasks "
        f"({meta.name} @ {meta.revision}). Those scores will be reused instead of a "
        "fresh evaluation. Reused scores must already include K=10, 25, 50, and 100. "
        "Pass --no-cache to deactivate the cache for this run."
    )


def metric_name(metric: str, k: int) -> str:
    return f"{metric}@{k}"


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def use_cutoffs(tasks: Sequence[AbsTask]) -> None:
    cutoff = max(_EVAL_K_VALUES)
    for task in tasks:
        if not isinstance(task, AbsTaskRetrieval):
            raise TypeError(f"{type(task).__name__} is not a retrieval task")
        task.k_values = _EVAL_K_VALUES
        task._top_k = cutoff


def metric_names() -> list[str]:
    return [metric_name(metric, k) for metric in METRICS for k in K_VALUES]


def score_from_result(result: TaskResult, key: str) -> float:
    try:
        value = result.get_score(getter=lambda scores, score_key=key: scores[score_key])
    except KeyError as error:
        cutoffs = ", ".join(str(k) for k in K_VALUES)
        raise SystemExit(
            f"ERROR: {result.task_name} has no {key}. Cached results from another "
            f"cutoff set cannot be reused. Pass --no-cache to recompute scores at K={cutoffs}."
        ) from error
    return float(value)


def task_scores(result: TaskResult) -> dict[str, float]:
    scores = {
        metric_name(metric, k): score_from_result(result, f"{mteb_key}_at_{k}")
        for metric, mteb_key in _MTEB_METRIC_KEYS.items()
        for k in K_VALUES
    }
    for k in K_VALUES:
        scores[metric_name("F1", k)] = f1_at_k(
            scores[metric_name("Precision", k)],
            scores[metric_name("Recall", k)],
        )
    return {name: scores[name] for name in metric_names()}


@dataclass(frozen=True)
class ScoreRow:
    sector: str
    n: int
    scores: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]] | None = None


def bootstrap_mean_intervals(
    values: Sequence[Mapping[str, float]],
) -> dict[str, tuple[float, float]] | None:
    n = len(values)
    if n < 2:
        return None
    keys = metric_names()
    matrix = np.asarray([[item[key] for key in keys] for item in values], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = matrix[rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))]
    means = draws.mean(axis=1)
    lo, hi = np.percentile(means, BOOTSTRAP_PERCENTILES, axis=0)
    return {
        key: (round(float(lower), 5), round(float(upper), 5)) for key, lower, upper in zip(keys, lo, hi, strict=True)
    }


def project_scores(results: Sequence[TaskResult | TaskError]) -> list[tuple[str, dict[str, float]]]:
    scores: list[tuple[str, dict[str, float]]] = []
    for result in results:
        if isinstance(result, TaskError):
            console.print(f"[yellow]Skipping failed task:[/yellow] {result}")
            continue
        scores.append((sector_from_task_name(result.task_name), task_scores(result)))
    return scores


def aggregate_rows(
    scored: Sequence[tuple[str, dict[str, float]]],
    *,
    bootstrap: bool = True,
) -> list[ScoreRow]:
    by_sector: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sector, scores in scored:
        by_sector[sector].append(scores)

    overall = [scores for _, scores in scored]
    intervals = bootstrap_mean_intervals(overall) if bootstrap else None
    rows = [mean_row(sector, values) for sector, values in sorted(by_sector.items())]
    rows.append(mean_row("Overall", overall, intervals=intervals))
    return rows


def mean_row(
    sector: str,
    values: Sequence[Mapping[str, float]],
    *,
    intervals: Mapping[str, tuple[float, float]] | None = None,
) -> ScoreRow:
    return ScoreRow(
        sector,
        len(values),
        {key: round(mean(item[key] for item in values), 5) for key in metric_names()},
        intervals,
    )


def format_score(row: ScoreRow, key: str) -> str:
    value = f"{row.scores[key]:.3f}"
    if row.intervals is None:
        return value
    lo, hi = row.intervals[key]
    return f"{value} [{lo:.3f}, {hi:.3f}]"


def print_score_table(rows: Sequence[ScoreRow], *, model: str) -> None:
    table = Table(title=model, box=box.ROUNDED, header_style="bold", title_style="bold")
    table.add_column("Sector", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("Metric")
    for k in K_VALUES:
        table.add_column(f"@{k}", justify="right")

    last_row = len(rows) - 1
    for index, row in enumerate(rows):
        style = "bold" if row.sector == "Overall" else None
        for metric_index, metric in enumerate(METRICS):
            last_metric = metric_index == len(METRICS) - 1
            table.add_row(
                row.sector if metric_index == 0 else "",
                str(row.n) if metric_index == 0 else "",
                metric,
                *(format_score(row, metric_name(metric, k)) for k in K_VALUES),
                style=style,
                end_section=last_metric and index != last_row,
            )
    console.print(table)


def score_payload(model: str, dataset: Mapping[str, str], rows: Sequence[ScoreRow]) -> dict[str, object]:
    payload_rows: list[dict[str, object]] = []
    for row in rows:
        entry: dict[str, object] = {
            "sector": row.sector,
            "n": row.n,
            "scores": {key: row.scores[key] for key in metric_names()},
        }
        intervals = row.intervals
        if intervals is not None:
            entry["intervals"] = {key: [intervals[key][0], intervals[key][1]] for key in metric_names()}
        payload_rows.append(entry)
    payload: dict[str, object] = {
        "model": model,
        "dataset": dict(dataset),
        "k_values": list(K_VALUES),
        "metrics": list(METRICS),
        "rows": payload_rows,
    }
    if any(row.intervals is not None for row in rows):
        payload["bootstrap"] = {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "percentiles": list(BOOTSTRAP_PERCENTILES),
            "unit": "project",
        }
    return payload


def write_score_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_recorded_name(model: object, name: str) -> None:
    """Record ``name`` on the model so prediction files store it instead of a local path."""
    meta = getattr(model, "mteb_model_meta", None)
    if not isinstance(meta, ModelMeta):
        return
    model.mteb_model_meta = meta.model_copy(update={"name": name})


def evaluate_named_model(
    name: str,
    tasks: Sequence[AbsTask],
    *,
    dataset: Mapping[str, str],
    cache: ResultCache | None,
    output_dir: Path | None,
    force: bool,
    batch_size: int,
    predictions_dir: Path | None = None,
    record_as: str | None = None,
) -> None:
    resolved = resolve_model_name(name, force=force)
    recorded = record_as or resolved
    model = load_model(resolved, tasks)
    if record_as is not None:
        apply_recorded_name(model, recorded)
    encode_kwargs = {"batch_size": batch_size}
    prediction_folder = predictions_dir / output_stem(recorded) if predictions_dir is not None else None
    try:
        if cache is None:
            results = mteb.evaluate(
                model, tasks=tasks, cache=None, encode_kwargs=encode_kwargs, prediction_folder=prediction_folder
            )
        else:
            warn_if_cached_results(tasks, model_meta_of(model), cache)
            if prediction_folder is not None and cached_task_names(tasks, model_meta_of(model), cache):
                logger.warning("Tasks loaded from the cache are not re-run and get no predictions; pass --no-cache")
            results = mteb.evaluate(
                model, tasks=tasks, cache=cache, encode_kwargs=encode_kwargs, prediction_folder=prediction_folder
            )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
    scored = project_scores(results.task_results)
    if not scored:
        raise SystemExit(f"ERROR: no successful task results to aggregate for {resolved}")

    rows = aggregate_rows(scored, bootstrap=resolved != ORACLE_NAME)
    print_score_table(rows, model=recorded)
    if output_dir is not None:
        output = output_dir / f"{output_stem(recorded)}.json"
        write_score_json(output, score_payload(recorded, dataset, rows))
        console.print(f"Wrote [bold]{output}[/bold]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an MTEB model on the TENDRIL test projects")
    parser.add_argument(
        "--repo",
        default=TEST_REPO,
        help=f"Hub dataset repo of the test set (default: {TEST_REPO})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch, tag, or commit of the test set (default: main)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        default=None,
        help=(
            "MTEB model name or a local SentenceTransformer directory. "
            f"Reviewed ids are listed by --list-models. Default: {DEFAULT_MODEL_NAME}"
        ),
    )
    selection.add_argument(
        "--group",
        choices=(*MODEL_GROUPS, "oracle"),
        default=None,
        help=(
            "Evaluate every reviewed model in this group "
            "(baseline, local, api, or reference). "
            "oracle is another name for reference."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print reviewed models and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the delay before evaluating an unreviewed model",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Deactivate the MTEB result cache for this run (do not load or save cached scores)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Passages encoded per forward pass. Each batch is padded to its longest "
            "passage, so long documents need a small value (default: 32)"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Name recorded in the score JSON, the prediction files, and the output paths. "
            "A slash is written as __. Use this for a local checkpoint, for example "
            "Snowflake/snowflake-arctic-embed-l-v2.0__TENDRIL."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Directory for score JSON. Each model writes {name with / as __}.json."),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Directory for per-task rankings, written to {name with / as __}/<task>_predictions.json. "
            "Needed for query-level analyses such as scripts/evaluation/paired_bootstrap.py"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_env()
    if args.list_models:
        print_known_models()
        return

    model_name = args.model
    if model_name is None and args.group is None:
        model_name = DEFAULT_MODEL_NAME

    if args.batch_size < 1:
        raise SystemExit("ERROR: --batch-size must be at least 1")
    if args.model_name is not None and not args.model_name.strip():
        raise SystemExit("ERROR: --model-name must not be empty")
    if args.group is not None and args.model_name is not None:
        raise SystemExit("ERROR: --model-name applies to one --model, not a --group")

    # Pin the commit so every project loads the same snapshot and the score JSON records it.
    dataset = {"repo": args.repo, "revision": resolve_revision(args.repo, args.revision)}
    tasks = load_hub_tasks(dataset["repo"], revision=dataset["revision"])
    use_cutoffs(tasks)
    cache = None if args.no_cache else ResultCache(cache_path=DEFAULT_CACHE_PATH)

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    if args.group is not None:
        group = "reference" if args.group == "oracle" else args.group
        group_models = models_in_group(group)
        require_env(group_models)
        for known in group_models:
            evaluate_named_model(
                known.name,
                tasks,
                dataset=dataset,
                cache=cache,
                output_dir=args.output,
                force=args.force,
                batch_size=args.batch_size,
                predictions_dir=args.predictions,
            )
        return

    if model_name is None:
        raise SystemExit("ERROR: pass --model or --group")

    evaluate_named_model(
        model_name,
        tasks,
        dataset=dataset,
        cache=cache,
        output_dir=args.output,
        force=args.force,
        batch_size=args.batch_size,
        predictions_dir=args.predictions,
        record_as=args.model_name,
    )


if __name__ == "__main__":
    main()
