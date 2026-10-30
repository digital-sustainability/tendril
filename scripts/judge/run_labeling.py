#!/usr/bin/env python
"""Label validation excerpts with each model listed in a judge config.

For every excerpt and both targets (area of action, criterion) the configured
models are queried with that config's prompt. Extracted scores, any rationale
the model returned, token usage, latency, the provider-reported cost (only
OpenRouter returns one via usage accounting) and the raw response are appended
to <results_dir>/results.jsonl (crash-safe, resumable).

Without --results-dir, each run writes to a fresh data/judge/<config file
stem>_<UTC ISO timestamp> directory so earlier results are never appended to
or overwritten by accident; pass --results-dir to resume or append to an
existing run. Relative results paths are resolved from the repository root.

Usage (from the repository root):
    python scripts/judge/run_labeling.py --dry-run
    python scripts/judge/run_labeling.py --config scripts/judge/configs/model_reasoning_summary_aoa.yaml --max-samples 4
    python scripts/judge/run_labeling.py

`--max-samples` (alias `--limit`) overrides `runtime.max_samples` from the
config; setting `runtime.max_samples: 10` there achieves the same without a
CLI flag (null/unset = full run). The default config is the original-wording
model and reasoning-effort sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import judge_common as jc
import pandas as pd

logger = logging.getLogger(__name__)

_PRINT_LOCK = threading.Lock()
# Used when the runtime sets neither batch_chunk_size nor throttle.max_concurrency.
_DEFAULT_BATCH_CHUNK = 256
_HYPERPARAMS: dict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def _rate_limit_wait_seconds(error: str) -> float | None:
    """Parse 'Limit resets at: ... UTC' from rate-limit errors that state their reset time."""
    match = re.search(r"Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", str(error))
    if not match:
        return None
    try:
        reset = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0.0, (reset - datetime.now(timezone.utc)).total_seconds()) + 2.0
    except ValueError:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-a-judge labeling run")
    p.add_argument("--config", default="configs/model_reasoning_simple_aoa.yaml")
    p.add_argument("--models", nargs="*", help="subset of model names from the config")
    p.add_argument("--targets", nargs="*", choices=["area", "criterion"])
    p.add_argument(
        "--query-representations",
        nargs="+",
        choices=jc.QUERY_REPRESENTATIONS,
        help="area-query wording(s); overrides query_representations in config",
    )
    p.add_argument("--max-samples", "--limit", dest="max_samples", type=int, default=None,
                   help="label only the first N excerpts (overrides runtime.max_samples)")
    p.add_argument("--max-workers", type=int, default=None, help="override runtime.max_workers")
    p.add_argument("--max-concurrent-models", type=int, default=None,
                   help="override runtime.max_concurrent_models (how many models run at once)")
    p.add_argument("--results-dir", default=None,
                   help="results dir (default: <config file stem>_<UTC ISO timestamp>, "
                        "unique per run; pass this to resume/append to an earlier run)")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="console log verbosity (default: LOG_LEVEL env var or INFO; "
                        "DEBUG also logs every HTTP request made by the LLM SDKs)")
    p.add_argument("--dry-run", action="store_true", help="one live call per model, print, exit")
    return p.parse_args()


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def resolve_batch_chunk_size(runtime: dict) -> int:
    """How many requests one ``generate_batch`` call may submit.

    ``rr.run_batch`` creates one asyncio task per request before any of them
    runs, so a full labeling list (hundreds of thousands of rows) sits at 0%
    while those tasks are allocated. Each chunk is checkpointed to the JSONL
    before the next chunk starts. ``runtime.batch_chunk_size`` wins; otherwise
    the throttle concurrency is used, which fills one in-flight window.
    """
    explicit = runtime.get("batch_chunk_size")
    if explicit is not None:
        size = _positive_int(explicit)
        if size is None:
            raise ValueError(
                f"runtime.batch_chunk_size must be an integer >= 1, got {explicit!r}"
            )
        return size
    throttle = runtime.get("throttle")
    if isinstance(throttle, dict):
        concurrency = _positive_int(throttle.get("max_concurrency"))
        if concurrency is not None:
            return concurrency
    return _DEFAULT_BATCH_CHUNK


def resolve_model_concurrency(runtime: dict, n_models: int) -> int:
    """How many models to evaluate at once.

    Each selected model runs on one thread and holds its whole task list until
    that list finishes, so this cap is the memory bound. Unset means every
    selected model starts together. The pool size is at least 1.
    """
    cap = runtime.get("max_concurrent_models")
    if cap is None:
        return max(1, n_models)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(
            f"runtime.max_concurrent_models must be an integer >= 1, got {cap!r}"
        )
    if n_models < 1:
        return 1
    return min(cap, n_models)


def resolve_results_dir(results_dir_arg: str | None, config_path: str) -> Path:
    """Results directory for this run.

    ``--results-dir`` wins and is resolved from the repository root unless it
    is already absolute. Otherwise a fresh directory under ``data/judge/`` is
    used, named from the config file stem plus the exact UTC ISO timestamp
    (e.g. ``data/judge/model_reasoning_simple_aoa_2026-10-01T08:00:00+00:00``),
    so a new run can never append to or overwrite an older results directory
    by accident.
    """
    if results_dir_arg:
        return jc.resolve_path(results_dir_arg)
    return jc.REPO_ROOT / "data" / "judge" / f"{Path(config_path).stem}_{jc.now_iso()}"


def build_tasks(
    df,
    cfg,
    prompt_text,
    prompt_sha,
    models,
    targets,
    query_representations=None,
    query_map_data=None,
):
    """One task per (excerpt, target, model)."""
    id_col = cfg["dataset"].get("id_column", "sample_id")
    passage_col = cfg["dataset"].get("passage_column", "sentence_cleaned")
    default_tokens = cfg["hyperparams"]["max_tokens"]
    tasks = []
    for mcfg in cfg["models"]:
        if mcfg["name"] not in models:
            continue
        for tgt in targets:
            tcfg = cfg["targets"][tgt]
            base_extra = ({"task_instruction": tcfg["task_instruction"]}
                          if tcfg.get("task_instruction") else {})
            ambition_cols = tcfg.get("ambition_columns") or []
            ambition_labels = tcfg.get("ambition_labels") or []
            if ambition_cols and len(ambition_labels) < len(ambition_cols):
                raise ValueError(f"target {tgt!r}: ambition_labels must cover all "
                                 f"ambition_columns ({ambition_cols})")
            representations = (
                query_representations or ["original"]
                if tgt == "area"
                else ["original"]
            )
            for representation in representations:
                for row in df.itertuples(index=False):
                    record = row._asdict()
                    extra = dict(base_extra)
                    # Ambitionsniveaus belong to the criterion; the placeholder
                    # is always provided so templates do not retain it.
                    extra["ambition_levels"] = (
                        jc.format_ambition_levels(
                            dict(
                                zip(
                                    ambition_labels,
                                    (record[c] for c in ambition_cols),
                                )
                            )
                        )
                        if ambition_cols
                        else ""
                    )
                    query_id, query_text = jc.resolve_task_query(
                        record,
                        tgt,
                        tcfg,
                        representation,
                        query_map_data or {},
                    )
                    passage = str(record[passage_col])
                    task = {
                        "sample_id": str(record[id_col]),
                        "wdg_id": record["wdg_id"],
                        "criterion_id": record["criterion_id"],
                        "target": tgt,
                        "model": mcfg["name"],
                        "client": mcfg["client"],
                        "model_id": mcfg["model"],
                        "query_id": query_id,
                        "query_representation": representation,
                        "query_text": query_text,
                        "passage": passage,
                        "truth": int(record[tcfg["truth_column"]]),
                        "prompt": jc.build_prompt(
                            prompt_text, query_text, passage, extra
                        ),
                        "max_tokens": mcfg.get("max_tokens", default_tokens),
                    }
                    reasoning = mcfg.get("reasoning")
                    if reasoning is not None:
                        task["reasoning"] = reasoning
                    tasks.append(task)
    return tasks


def load_completed(path, retry_failed: bool = False) -> set:
    """Keys of successfully labeled tasks. Error/parse-failure rows are always
    re-attempted on the next run (they may simply need a retry after a rate-limit)."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_json(path, lines=True)
    df = df[df["parse_ok"].astype(bool) & df["error"].isna()]
    return {
        (
            str(r.sample_id),
            str(r.target),
            str(r.model),
            str(getattr(r, "query_representation", "original")),
        )
        for r in df.itertuples(index=False)
    }


def execute_task(task, client, runtime):
    """Call the model with retries; returns (text, usage, error, attempts, latency_s).

    Every failed attempt and every exhausted retry is logged (WARNING/ERROR)
    with model, sample, target and the provider error message including the
    failing call site (judge_common.describe_exception); terminal failures
    additionally log the full traceback. LLM call errors are thus visible on
    the console and in the run.log file instead of only being recorded in the
    results JSONL.
    """
    messages = [{"role": "user", "content": task["prompt"]}]
    params = {k: _HYPERPARAMS[k] for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty")}
    params["max_tokens"] = task["max_tokens"]
    call_kwargs = {"return_usage": True, "timeout": runtime["timeout"]}
    if "reasoning" in task and task["reasoning"] is not None:
        reasoning_val = task["reasoning"]
        effort = reasoning_val.get("effort") if isinstance(reasoning_val, dict) else reasoning_val
        if effort is not None:
            call_kwargs["reasoning_effort"] = effort
        call_kwargs["reasoning"] = reasoning_val
    start, attempts, error, text, usage = time.time(), 0, None, None, None
    task_label = f'[{task["model"]}] {task["sample_id"]}/{task["target"]}'
    while attempts <= runtime["max_retries"]:
        attempts += 1
        try:
            text, usage = client.generate(messages, params, **call_kwargs)
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            # log with origin info (failing call site, HTTP request, cause chain)
            detail = jc.describe_exception(exc)
            if attempts <= runtime["max_retries"]:
                wait = runtime["retry_base_sleep"] * 2 ** (attempts - 1)
                if "rate limit" in error.lower():
                    reset_wait = _rate_limit_wait_seconds(error)
                    if reset_wait is not None:
                        wait = reset_wait
                    logger.warning("%s attempt %d/%d rate-limited: %s (waiting %.1fs)",
                                   task_label, attempts, runtime["max_retries"] + 1, detail, wait)
                else:
                    logger.warning("%s attempt %d/%d failed: %s (retrying in %.1fs)",
                                   task_label, attempts, runtime["max_retries"] + 1, detail, wait)
                time.sleep(wait)
            else:
                # terminal failure: also attach the full traceback (console + run.log)
                logger.error("%s failed after %d attempt(s): %s",
                             task_label, attempts, detail, exc_info=exc)
    if error is None:
        logger.debug("%s ok in %.1fs (completion_tokens=%s)",
                     task_label, time.time() - start, (usage or {}).get("completion_tokens"))
    return text, usage, error, attempts, time.time() - start


def execute_rr_batch(tasks, client, runtime, *, run_id, prompt_id):
    """Execute one chunk through paced ``rr.run_batch`` calls.

    The caller splits the full task list. This function only retries failures
    inside the chunk it was given, then returns one outcome per task.
    """
    outcomes = [None] * len(tasks)
    attempts = [0] * len(tasks)
    pending = list(range(len(tasks)))
    params = {
        key: _HYPERPARAMS[key]
        for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty")
    }

    while pending:
        requests = []
        for index in pending:
            task = tasks[index]
            request_params = dict(params)
            request_params["max_tokens"] = task["max_tokens"]
            call_kwargs = {"timeout": runtime["timeout"]}
            reasoning_val = task.get("reasoning")
            if reasoning_val is not None:
                effort = (
                    reasoning_val.get("effort")
                    if isinstance(reasoning_val, dict)
                    else reasoning_val
                )
                if effort is not None:
                    call_kwargs["reasoning_effort"] = effort
            attempts[index] += 1
            requests.append(
                {
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "params": request_params,
                    "kwargs": call_kwargs,
                }
            )

        started = time.monotonic()
        responses = client.generate_batch(
            requests,
            batch_id=f"{run_id}-{prompt_id}",
            tags={"experiment": "judge", "run_id": run_id, "prompt_id": prompt_id},
        )
        elapsed = time.monotonic() - started
        retry = []
        for index, response in zip(pending, responses, strict=True):
            if isinstance(response, BaseException):
                task = tasks[index]
                detail = jc.describe_exception(response)
                if attempts[index] <= runtime["max_retries"]:
                    retry.append(index)
                    logger.warning(
                        "[%s] %s/%s attempt %d/%d failed in RR batch: %s",
                        task["model"],
                        task["sample_id"],
                        task["target"],
                        attempts[index],
                        runtime["max_retries"] + 1,
                        detail,
                    )
                else:
                    logger.error(
                        "[%s] %s/%s failed after %d RR batch attempt(s): %s",
                        task["model"],
                        task["sample_id"],
                        task["target"],
                        attempts[index],
                        detail,
                    )
                    outcomes[index] = (
                        None,
                        None,
                        f"{type(response).__name__}: {response}",
                        attempts[index],
                        elapsed,
                    )
                continue
            text, usage = response
            outcomes[index] = (text, usage, None, attempts[index], elapsed)
        pending = retry
        if pending:
            retry_number = max(attempts[index] for index in pending)
            time.sleep(runtime["retry_base_sleep"] * 2 ** (retry_number - 1))

    return outcomes


def _truncate_middle(text: str, limit: int = 6000) -> str:
    """Keep head+tail of long (reasoning) responses; the JSON sits at the end."""
    if len(text) <= limit:
        return text
    keep = limit // 2
    return f"{text[:keep]}\n...[middle truncated; full text kept by provider]...\n{text[-keep:]}"


def _export_xlsx(results_path) -> None:
    """Refresh the Excel mirror of the JSONL checkpoint (best effort)."""
    xlsx = jc.export_results_xlsx(results_path)
    if xlsx:
        print(f"Excel mirror: {xlsx}", flush=True)


def run_model_tasks(
    model_cfg,
    tasks,
    results_path,
    run_id,
    prompt_id,
    prompt_sha,
    runtime,
    experiment: str | None = None,
):
    name = model_cfg["name"]
    client = jc.make_client(
        model_cfg["client"],
        model_cfg["model"],
        **jc.responsible_request_kwargs(
            model_cfg,
            runtime,
            log_sqlite=results_path.parent / "responsible_request.sqlite",
            run_name=experiment or "judge",
            run_metadata={
                "prompt_id": prompt_id,
                "model": model_cfg.get("name"),
                "model_id": model_cfg.get("model"),
                "hyperparams": dict(_HYPERPARAMS),
            },
        ),
    )
    stats = {"requested": len(tasks), "succeeded": 0, "parse_failures": 0, "errors": 0,
             "completion_tokens": 0, "cost": 0.0, "truncated": 0, "total_latency_s": 0.0}
    progress = {"done": 0}

    def work(task, outcome=None):
        try:
            if outcome is None:
                outcome = execute_task(task, client, runtime)
            text, usage, error, attempts, latency = outcome
            parsed = jc.parse_judgment(task["prompt"], text, usage, latency)
            scores = parsed.scores
            parse_ok = parsed.parse_ok
            usage = usage or {}
            row = {
                "run_id": run_id, "created_at": jc.now_iso(),
                "sample_id": task["sample_id"], "wdg_id": task["wdg_id"],
                "criterion_id": task["criterion_id"], "target": task["target"],
                "model": task["model"], "client": task["client"], "model_id": task["model_id"],
                "prompt_id": prompt_id, "prompt_sha256": prompt_sha,
                # exact user message sent to the LLM (full, untruncated) for audit
                "prompt": task["prompt"],
                "query_id": task.get("query_id"),
                "query_representation": task.get("query_representation", "original"),
                "query_text": task["query_text"], "passage": task["passage"], "truth": task["truth"],
                "m_score": scores.get("M") if scores else None,
                "t_score": scores.get("T") if scores else None,
                "o_score": scores.get("O") if scores else None,
                # Begruendung from the guidelines with-reasoning variant (None
                # for the other prompts); saved whenever the model returned
                # one, even if the score did not parse (helps debugging).
                "reasoning": parsed.reasoning,
                "thinking": parsed.thinking,
                "parse_method": parsed.parse_method, "parse_ok": parse_ok,
                "finish_reason": usage.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                # Host-level accounting (openrouter client only, null for all
                # other clients): serving provider, native-tokenizer counts and
                # cache/reasoning-token details — extracted in
                # tendril/inference/openrouter_client.py.
                "provider": usage.get("provider"),
                "native_prompt_tokens": usage.get("native_prompt_tokens"),
                "native_completion_tokens": usage.get("native_completion_tokens"),
                "native_reasoning_tokens": usage.get("native_reasoning_tokens"),
                "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "upstream_inference_cost": usage.get("upstream_inference_cost"),
                # Monetary cost (USD) as reported by the provider in the usage
                # object; only the openrouter client returns one, all other
                # clients leave this null.
                "cost": usage.get("cost"),
                "latency_s": round(latency, 2), "attempts": attempts,
                "error": error, "raw_response": _truncate_middle(text or ""),
            }
            jc.append_jsonl(results_path, row)
            if not parse_ok:
                # Empty/None response or unparsable text: the row is written
                # with parse_ok=False, but make the cause visible immediately.
                logger.warning("[%s] %s/%s: no score parsed (error=%s, finish_reason=%s, raw=%.200r)",
                               name, task["sample_id"], task["target"], error,
                               usage.get("finish_reason"), (text or "")[:200])
            with _PRINT_LOCK:
                stats["succeeded"] += int(parse_ok)
                stats["parse_failures"] += int(text is not None and not parse_ok)
                stats["errors"] += int(error is not None)
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                stats["cost"] += float(usage.get("cost") or 0)
                stats["truncated"] += int(usage.get("finish_reason") == "length")
                stats["total_latency_s"] += latency
                progress["done"] += 1
                if progress["done"] % 50 == 0 or progress["done"] == len(tasks):
                    print(f"  [{name}] {progress['done']}/{len(tasks)} tasks done", flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] TASK FAILURE %s/%s: %s",
                             name, task["sample_id"], task["target"], exc)
            with _PRINT_LOCK:
                stats["errors"] += 1

    try:
        if hasattr(client, "generate_batch"):
            # One rr.run_batch over the whole list allocates every asyncio task
            # before the first request is sent, and the JSONL is only written
            # after that call returns. Chunks start immediately and are
            # appended before the next chunk is submitted.
            chunk_size = resolve_batch_chunk_size(runtime)
            print(f"  [{name}] batches of {chunk_size}", flush=True)
            for start in range(0, len(tasks), chunk_size):
                chunk = tasks[start:start + chunk_size]
                outcomes = execute_rr_batch(
                    chunk,
                    client,
                    runtime,
                    run_id=run_id,
                    prompt_id=prompt_id,
                )
                for task, outcome in zip(chunk, outcomes, strict=True):
                    work(task, outcome)
        else:
            workers = model_cfg.get("max_workers", runtime["max_workers"])
            print(f"  [{name}] workers={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(work, tasks))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    # round away float-accumulation noise before the manifest/summary dump
    stats["cost"] = round(stats["cost"], 10)
    return stats


def main():
    args = parse_args()
    jc.setup_logging(args.log_level)
    cfg = jc.load_config(args.config)
    global _HYPERPARAMS
    _HYPERPARAMS = dict(cfg["hyperparams"])
    runtime = dict(cfg["runtime"])
    if args.max_workers:
        runtime["max_workers"] = args.max_workers
    if args.max_concurrent_models is not None:
        runtime["max_concurrent_models"] = args.max_concurrent_models
    models = args.models or [m["name"] for m in cfg["models"]]
    targets = args.targets or list(cfg["targets"])
    query_representations = jc.configured_query_representations(
        cfg, args.query_representations
    )
    query_map_data = jc.query_maps(query_representations)
    results_dir = resolve_results_dir(args.results_dir, args.config)
    if not args.results_dir:
        print(f"no --results-dir given -> fresh results dir: {results_dir.name} "
              f"(pass --results-dir to append to an earlier run)", flush=True)
    results_path = results_dir / "results.jsonl"
    manifest_path = results_dir / "manifest.json"
    prompt_text, prompt_sha = jc.load_prompt(cfg)
    df = jc.load_dataset(cfg)
    max_samples = args.max_samples if args.max_samples is not None else runtime.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples))
        print(f"TEST MODE: max_samples={max_samples} -> first {len(df)} excerpts only", flush=True)
    models_cfg = [m for m in cfg["models"] if m["name"] in models]
    model_slots = resolve_model_concurrency(runtime, len(models_cfg))
    print(
        f"models={models} at once={model_slots} targets={targets} "
        f"excerpts={len(df)} results={results_path}",
        flush=True,
    )

    if args.dry_run:
        tasks = build_tasks(
            df.head(2),
            cfg,
            prompt_text,
            prompt_sha,
            models,
            targets,
            query_representations,
            query_map_data,
        )
        for m in models_cfg:
            task = next((t for t in tasks if t["model"] == m["name"]), None)
            if task is None:
                continue
            client = jc.make_client(
                m["client"],
                m["model"],
                **jc.responsible_request_kwargs(
                    m,
                    runtime,
                    log_sqlite=results_dir / "responsible_request.sqlite",
                    run_name=cfg.get("experiment"),
                    run_metadata={
                        "prompt_id": cfg["prompt"]["id"],
                        "model": m.get("name"),
                        "model_id": m.get("model"),
                        "hyperparams": dict(_HYPERPARAMS),
                    },
                ),
            )
            try:
                text, usage, error, attempts, latency = execute_task(
                    task, client, runtime
                )
                parsed = jc.parse_judgment(task["prompt"], text, usage, latency)
                print(
                    f"\n=== {m['name']} | {latency:.1f}s | attempts={attempts} | "
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
    done = load_completed(results_path)
    tasks = build_tasks(
        df,
        cfg,
        prompt_text,
        prompt_sha,
        models,
        targets,
        query_representations,
        query_map_data,
    )
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
        _export_xlsx(results_path)
        return

    run_id = jc.new_run_id()
    manifest = {"experiment": cfg["experiment"], "run_id": run_id, "started_at": jc.now_iso(),
                "prompt_id": cfg["prompt"]["id"], "prompt_sha256": prompt_sha,
                "hyperparams": cfg["hyperparams"], "models": models_cfg, "targets": targets,
                "query_representations": query_representations,
                "n_excerpts": len(df), "max_samples": max_samples,
                "max_concurrent_models": model_slots,
                "n_tasks": len(tasks), "n_pending": len(pending)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))

    all_stats, t0 = {}, time.time()

    def run_one(mcfg):
        m_tasks = [t for t in pending if t["model"] == mcfg["name"]]
        if not m_tasks:
            return mcfg["name"], None
        print(f"\n== {mcfg['name']} | {len(m_tasks)} tasks ==", flush=True)
        return (mcfg["name"],
                run_model_tasks(mcfg, m_tasks, results_path, run_id,
                                cfg["prompt"]["id"], prompt_sha, runtime,
                                experiment=cfg.get("experiment")))

    # One thread per model. The pool size is the cap on models in flight.
    with ThreadPoolExecutor(max_workers=model_slots) as pool:
        for name, stats in pool.map(run_one, models_cfg):
            if stats:
                all_stats[name] = stats

    manifest["finished_at"] = jc.now_iso()
    manifest["wall_time_s"] = round(time.time() - t0, 1)
    manifest["stats"] = all_stats
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(f"\nDone in {round(time.time() - t0)}s. Results: {results_path}", flush=True)
    _export_xlsx(results_path)
    print(pd.DataFrame(all_stats).T.to_string(), flush=True)


if __name__ == "__main__":
    main()