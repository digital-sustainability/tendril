#!/usr/bin/env python
"""Label parquet query-passage pairs with each model listed in a judge config.

The default config is the shortest German guidelines prompt with
``z-ai/glm-5.3-flash`` at low reasoning effort. ``{query}`` is the area-of-action
summary, or the criterion text alone. Ambition levels, when the criterion has
any, go in ``{ambition_levels}``. ``{passage}`` is the passage text.

Pairs, queries, and passages are loaded from a directory, restricted projects
are dropped, and query ids are rewritten to ``Catalog::HF-NNN`` / ``Catalog::K-NNN``.

Each remaining row is one labeling task (area or criterion, not both). Scores
are appended to ``<results_dir>/results.jsonl`` with the same checkpoint and
retry behavior as ``run_labeling.py``. There is no human 0-2 score, so
``truth`` is null.

When labeling finishes, the run also writes
``<results_dir>/relevance_judgments.parquet`` with three columns: ``query_id``,
``document_id`` (the passage uid), and ``score`` (the parsed 0/1/2 judgment).
The JSONL checkpoint is left as the full audit log.

Usage (from the repository root):
    python scripts/judge/run_labeling_parquet.py --root /path/to/parquet --dry-run --max-samples 2
    python scripts/judge/run_labeling_parquet.py --root /path/to/parquet --max-samples 4
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import judge_common as jc
import pandas as pd
import run_labeling as rl

from tendril.queries import (  # noqa: F401  (re-exported for callers and tests)
    ambition_levels,
    canonical_query_id,
    criterion_ambition_block,
    criterion_query_text,
    rewrite_queries,
)

logger = logging.getLogger(__name__)

LEVEL_TARGET = {
    "area_of_action": "area",
    "criterion": "criterion",
}
LEVEL_REPRESENTATION = {
    "area_of_action": "llm_summary",
    "criterion": "formatted",
}


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-a-judge labeling from parquet pairs")
    p.add_argument("--root", required=True, help="directory containing the parquet files")
    p.add_argument("--pairs", default="pairs.parquet", help="pairs filename inside --root")
    p.add_argument("--queries", default="queries.parquet", help="queries filename inside --root")
    p.add_argument("--passages", default="passages_anon.parquet", help="passages filename inside --root")
    p.add_argument("--query-column", default="query_text")
    p.add_argument("--passage-column", default="text")
    p.add_argument("--project-id-column", default="project_id")
    p.add_argument("--query-id-column", default="query_id")
    p.add_argument(
        "--exclude",
        default="data/exports/restrictions/train_projects.txt",
        help="project ids to drop, one per line (exact match)",
    )
    p.add_argument(
        "--summary",
        default="data/queries/llm_summary.jsonl",
        help="area-of-action query summaries ({id, text} JSONL)",
    )
    p.add_argument("--config", default="configs/prompt_length_shortest.yaml",
                   help="judge config (default: shortest guidelines, GLM 5.3 Flash low)")
    p.add_argument("--models", nargs="*", help="subset of model names from the config")
    p.add_argument("--targets", nargs="*", choices=["area", "criterion"])
    p.add_argument("--max-samples", "--limit", dest="max_samples", type=int, default=None,
                   help="label only the first N rows after exclusions (overrides runtime.max_samples)")
    p.add_argument("--max-workers", type=int, default=None, help="override runtime.max_workers")
    p.add_argument("--max-concurrent-models", type=int, default=None,
                   help="override runtime.max_concurrent_models (how many models run at once)")
    p.add_argument("--results-dir", default=None,
                   help="results dir (default: <config file stem>_<UTC ISO timestamp>, "
                        "unique per run; pass this to resume/append to an earlier run)")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="console log verbosity (default: LOG_LEVEL env var or INFO)")
    p.add_argument("--dry-run", action="store_true", help="one live call per model, print, exit")
    return p.parse_args()


def _under_root(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def _unique(columns: list[str]) -> list[str]:
    seen: list[str] = []
    for column in columns:
        if column not in seen:
            seen.append(column)
    return seen


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}; has {list(frame.columns)}")


def load_exclude_ids(path: Path) -> set[str]:
    """Project ids to drop. Blank lines are ignored. Match is exact."""
    if not path.is_file():
        raise FileNotFoundError(f"Exclude list not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_judge_input(
    root: Path,
    pairs_name: str,
    queries_name: str,
    passages_name: str,
    query_column: str,
    passage_column: str,
    project_id_column: str,
    query_id_column: str,
) -> pd.DataFrame:
    """Merge pairs, queries, and passages. Shared fields come from the queries frame."""
    if query_column == passage_column:
        raise ValueError("query column and passage column must be different names")
    pairs = pd.read_parquet(_under_root(root, pairs_name))
    queries = pd.read_parquet(_under_root(root, queries_name))
    passages = pd.read_parquet(_under_root(root, passages_name))

    query_columns = _unique([
        project_id_column, query_id_column, query_column, "level", "catalog", "criterion_id",
    ])
    pair_columns = _unique([project_id_column, query_id_column, "passage_uid"])
    passage_columns = _unique([project_id_column, "passage_uid", passage_column])
    _require_columns(queries, query_columns, queries_name)
    _require_columns(pairs, pair_columns, pairs_name)
    _require_columns(passages, passage_columns, passages_name)

    merged = pairs[pair_columns].merge(
        queries[query_columns],
        on=[project_id_column, query_id_column],
        validate="many_to_one",
    )
    merged = merged.merge(
        passages[passage_columns],
        on=[project_id_column, "passage_uid"],
        validate="many_to_one",
    )
    return merged


def exclude_projects(frame: pd.DataFrame, project_id_column: str, excluded: set[str]) -> pd.DataFrame:
    if not excluded:
        return frame.reset_index(drop=True)
    keep = ~frame[project_id_column].astype(str).isin(excluded)
    return frame.loc[keep].reset_index(drop=True)


def missing_area_summaries(
    frame: pd.DataFrame,
    summaries: dict[str, str],
    query_id_column: str,
) -> list[str]:
    area = frame.loc[frame["level"] == "area_of_action", ["catalog", query_id_column]].drop_duplicates()
    missing: list[str] = []
    for catalog, query_id in zip(area["catalog"], area[query_id_column], strict=True):
        canonical = canonical_query_id(catalog, query_id)
        if canonical not in summaries:
            missing.append(canonical)
    return missing


def _null_if_missing(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def build_tasks(
    df,
    cfg,
    prompt_text,
    models,
    query_column: str,
    passage_column: str,
    project_id_column: str,
    query_id_column: str,
):
    """One task per (row, model). The row's level selects the target instruction."""
    default_tokens = cfg["hyperparams"]["max_tokens"]
    tasks = []
    selected = [mcfg for mcfg in cfg["models"] if mcfg["name"] in models]
    for record in df.to_dict(orient="records"):
        level = record["level"]
        target = LEVEL_TARGET[level]
        tcfg = cfg["targets"][target]
        ambition = _null_if_missing(record.get("ambition_levels")) or ""
        extra = {"ambition_levels": ambition}
        if tcfg.get("task_instruction"):
            extra["task_instruction"] = tcfg["task_instruction"]
        query_id = str(record[query_id_column])
        query_text = str(record[query_column])
        passage = str(record[passage_column])
        project_id = str(record[project_id_column])
        passage_uid = record["passage_uid"]
        base = {
            "sample_id": f"{project_id}|{query_id}|{passage_uid}",
            "wdg_id": record["catalog"],
            "criterion_id": _null_if_missing(record.get("criterion_id")),
            "target": target,
            "query_id": query_id,
            "query_representation": LEVEL_REPRESENTATION[level],
            "query_text": query_text,
            "passage": passage,
            "truth": None,
            "prompt": jc.build_prompt(prompt_text, query_text, passage, extra),
        }
        for mcfg in selected:
            task = {
                **base,
                "model": mcfg["name"],
                "client": mcfg["client"],
                "model_id": mcfg["model"],
                "max_tokens": mcfg.get("max_tokens", default_tokens),
            }
            reasoning = mcfg.get("reasoning")
            if reasoning is not None:
                task["reasoning"] = reasoning
            tasks.append(task)
    return tasks


def judgment_from_row(row: dict) -> tuple[str, str, int] | None:
    """Return ``(query_id, document_id, score)`` for one checkpoint row.

    ``document_id`` is the passage uid, the third ``|`` field of ``sample_id``
    (``project_id|query_id|passage_uid``). Rows that did not parse to a 0/1/2
    score return None. A parsed score outside that range is an error.
    """
    if row.get("parse_ok") is not True:
        return None
    score = row.get("o_score")
    if score is None:
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or int(score) != score:
        raise ValueError(f"Judgment score must be an integer 0, 1, or 2, got {score!r}")
    parsed = int(score)
    if parsed not in (0, 1, 2):
        raise ValueError(f"Judgment score must be 0, 1, or 2, got {parsed}")
    query_id = row.get("query_id")
    sample_id = row.get("sample_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"Checkpoint row is missing query_id: sample_id={sample_id!r}")
    if not isinstance(sample_id, str):
        raise ValueError(f"Checkpoint row is missing sample_id for query_id={query_id!r}")
    project_id, sample_query_id, document_id = _sample_id_parts(sample_id)
    if sample_query_id != query_id:
        raise ValueError(
            f"sample_id query {sample_query_id!r} does not match query_id {query_id!r} "
            f"(project_id={project_id!r})"
        )
    return query_id, document_id, parsed


def _sample_id_parts(sample_id: str) -> tuple[str, str, str]:
    parts = sample_id.split("|")
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(
            f"sample_id must be project_id|query_id|passage_uid, got {sample_id!r}"
        )
    return parts[0], parts[1], parts[2]


def write_relevance_judgments(results_path: Path) -> Path:
    """Write ``relevance_judgments.parquet`` next to a labeling checkpoint.

    The parquet has ``query_id``, ``document_id`` (passage uid), and ``score``.
    Unparsed rows are omitted. Identical repeat scores for one pair are
    collapsed. Conflicting scores for one pair raise. This only creates or
    replaces ``relevance_judgments.parquet``; the JSONL and every other file
    in the results directory stay untouched.
    """
    results_path = Path(results_path)
    if not results_path.is_file():
        raise FileNotFoundError(f"Results checkpoint not found: {results_path}")
    destination = results_path.with_name("relevance_judgments.parquet")
    if destination.resolve() == results_path.resolve():
        raise ValueError(f"Refusing to write judgments over the checkpoint: {results_path}")
    partial = results_path.with_name("relevance_judgments.parquet.partial")

    scores_by_pair: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    n_rows = 0
    n_unscored = 0
    n_duplicate = 0
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{results_path}:{line_number} is not valid JSON") from exc
            judgment = judgment_from_row(row)
            if judgment is None:
                n_unscored += 1
                continue
            query_id, document_id, score = judgment
            key = (query_id, document_id)
            previous = scores_by_pair.get(key)
            if previous is None:
                scores_by_pair[key] = score
                order.append(key)
            elif previous != score:
                raise ValueError(
                    f"Conflicting scores for query_id={query_id!r} "
                    f"document_id={document_id!r}: {previous} and {score}"
                )
            else:
                n_duplicate += 1
            if n_rows % 200_000 == 0:
                print(f"judgments: read {n_rows} checkpoint rows", flush=True)

    order.sort()
    frame = pd.DataFrame(
        {
            "query_id": [query_id for query_id, _document_id in order],
            "document_id": [document_id for _query_id, document_id in order],
            "score": pd.array([scores_by_pair[key] for key in order], dtype="int8"),
        }
    )
    try:
        frame.to_parquet(partial, index=False)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    print(
        f"judgments={len(frame)} unscored={n_unscored} "
        f"duplicate_rows={n_duplicate} -> {destination}",
        flush=True,
    )
    return destination


def _select_levels(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    unknown = sorted(set(frame["level"].dropna().unique()) - set(LEVEL_TARGET))
    if unknown:
        raise ValueError(f"Unknown query levels: {unknown}")
    selected = frame.loc[frame["level"].map(LEVEL_TARGET).isin(targets)]
    return selected.reset_index(drop=True)


def main():
    args = parse_args()
    jc.setup_logging(args.log_level)
    cfg = jc.load_config(args.config)
    rl._HYPERPARAMS = dict(cfg["hyperparams"])
    runtime = dict(cfg["runtime"])
    if args.max_workers:
        runtime["max_workers"] = args.max_workers
    if args.max_concurrent_models is not None:
        runtime["max_concurrent_models"] = args.max_concurrent_models
    models = args.models or [m["name"] for m in cfg["models"]]
    targets = args.targets or list(cfg["targets"])
    unknown_targets = sorted(set(targets) - set(cfg["targets"]))
    if unknown_targets:
        raise ValueError(f"Config {args.config} has no targets {unknown_targets}")

    root = jc.resolve_path(args.root)
    summaries = jc.load_query_map(jc.resolve_path(args.summary))
    excluded = load_exclude_ids(jc.resolve_path(args.exclude))
    frame = load_judge_input(
        root,
        args.pairs,
        args.queries,
        args.passages,
        args.query_column,
        args.passage_column,
        args.project_id_column,
        args.query_id_column,
    )
    before = len(frame)
    frame = exclude_projects(frame, args.project_id_column, excluded)
    print(
        f"rows={before} excluded={before - len(frame)} "
        f"exclude_ids={len(excluded)} kept={len(frame)}",
        flush=True,
    )
    frame = _select_levels(frame, targets)
    if "area" in targets:
        missing = missing_area_summaries(frame, summaries, args.query_id_column)
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} area queries missing from the summary map: {preview}"
            )

    max_samples = args.max_samples if args.max_samples is not None else runtime.get("max_samples")
    if max_samples:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
        print(f"TEST MODE: max_samples={max_samples} -> first {len(frame)} rows only", flush=True)
    if args.dry_run:
        frame = frame.head(2).reset_index(drop=True)
    frame = rewrite_queries(frame, summaries, args.query_column, args.query_id_column)

    results_dir = rl.resolve_results_dir(args.results_dir, args.config)
    if not args.results_dir:
        print(f"no --results-dir given -> fresh results dir: {results_dir.name} "
              f"(pass --results-dir to append to an earlier run)", flush=True)
    results_path = results_dir / "results.jsonl"
    manifest_path = results_dir / "manifest.json"
    prompt_text, prompt_sha = jc.load_prompt(cfg)
    models_cfg = [m for m in cfg["models"] if m["name"] in models]
    model_slots = rl.resolve_model_concurrency(runtime, len(models_cfg))
    print(
        f"models={models} at once={model_slots} targets={targets} "
        f"rows={len(frame)} results={results_path}",
        flush=True,
    )

    def tasks_for(rows):
        return build_tasks(
            rows,
            cfg,
            prompt_text,
            models,
            args.query_column,
            args.passage_column,
            args.project_id_column,
            args.query_id_column,
        )

    if args.dry_run:
        tasks = tasks_for(frame)
        for mcfg in models_cfg:
            task = next((t for t in tasks if t["model"] == mcfg["name"]), None)
            if task is None:
                continue
            client = jc.make_client(
                mcfg["client"],
                mcfg["model"],
                **jc.responsible_request_kwargs(
                    mcfg,
                    runtime,
                    log_sqlite=results_dir / "responsible_request.sqlite",
                    run_name=cfg.get("experiment"),
                    run_metadata={
                        "prompt_id": cfg["prompt"]["id"],
                        "model": mcfg.get("name"),
                        "model_id": mcfg.get("model"),
                        "hyperparams": dict(rl._HYPERPARAMS),
                    },
                ),
            )
            try:
                text, usage, error, attempts, latency = rl.execute_task(task, client, runtime)
                parsed = jc.parse_judgment(task["prompt"], text, usage, latency)
                print(
                    f"\n=== {mcfg['name']} | {latency:.1f}s | attempts={attempts} | "
                    f"usage={usage}\nerror={error}\nraw={text!r}\n"
                    f"parsed={parsed.scores} method={parsed.parse_method}\n"
                    f"reasoning={parsed.reasoning!r}\nthinking={parsed.thinking!r}",
                    flush=True,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    jc.setup_logging(args.log_level, log_file=results_dir / "run.log")
    logger.info("Also logging to %s", results_dir / "run.log")
    done = rl.load_completed(results_path)
    tasks = tasks_for(frame)
    pending = [
        task
        for task in tasks
        if (
            task["sample_id"],
            task["target"],
            task["model"],
            task["query_representation"],
        )
        not in done
    ]
    print(f"total tasks={len(tasks)} completed={len(tasks) - len(pending)} pending={len(pending)}", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        rl._export_xlsx(results_path)
        write_relevance_judgments(results_path)
        return

    run_id = jc.new_run_id()
    representations = sorted({task["query_representation"] for task in tasks})
    manifest = {
        "experiment": cfg["experiment"], "run_id": run_id, "started_at": jc.now_iso(),
        "prompt_id": cfg["prompt"]["id"], "prompt_sha256": prompt_sha,
        "hyperparams": cfg["hyperparams"], "models": models_cfg, "targets": targets,
        "query_representations": representations,
        "n_excerpts": len(frame), "max_samples": max_samples,
        "max_concurrent_models": model_slots,
        "n_tasks": len(tasks), "n_pending": len(pending),
        "root": str(root),
        "exclude": str(jc.resolve_path(args.exclude)),
        "summary": str(jc.resolve_path(args.summary)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))

    all_stats, t0 = {}, time.time()

    def run_one(mcfg):
        m_tasks = [t for t in pending if t["model"] == mcfg["name"]]
        if not m_tasks:
            return mcfg["name"], None
        print(f"\n== {mcfg['name']} | {len(m_tasks)} tasks ==", flush=True)
        return (
            mcfg["name"],
            rl.run_model_tasks(
                mcfg, m_tasks, results_path, run_id,
                cfg["prompt"]["id"], prompt_sha, runtime,
                experiment=cfg.get("experiment"),
            ),
        )

    with ThreadPoolExecutor(max_workers=model_slots) as pool:
        for name, stats in pool.map(run_one, models_cfg):
            if stats:
                all_stats[name] = stats

    manifest["finished_at"] = jc.now_iso()
    manifest["wall_time_s"] = round(time.time() - t0, 1)
    manifest["stats"] = all_stats
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"\nDone in {round(time.time() - t0)}s. Results: {results_path}", flush=True)
    rl._export_xlsx(results_path)
    write_relevance_judgments(results_path)
    print(pd.DataFrame(all_stats).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
