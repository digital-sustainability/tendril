"""Plan, pilot, sweep or finalize explicit-negative embedding training on one GPU.

Each training row holds one query, one positive and its own judged negatives; the loss never uses other rows of a
batch as negatives. Models are evaluated per project with graded nDCG@10 on both query levels.

Usage:
    python scripts/training/train_model.py --mode plan --data-dir DATA --output-dir OUT
    python scripts/training/train_model.py --mode sweep --data-dir DATA --output-dir OUT [--resume]
    python scripts/training/train_model.py --mode final --data-dir DATA --output-dir OUT \
        --learning-rate 5e-5 --effective-batch 32
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = ("area_of_action", "criterion")
SWEEP_BATCHES = (32, 64, 128)
RESUME_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


class ModelSpec(NamedTuple):
    learning_rates: list[float]
    query_prefix: str
    passage_prefix: str


MODELS = {
    "BAAI/bge-m3": ModelSpec([1e-5, 3e-5, 5e-5], "", ""),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ModelSpec([1e-5, 3e-5, 5e-5], "query: ", ""),
    "nvidia/Nemotron-3-Embed-1B-BF16": ModelSpec([3e-6, 6e-6, 1e-5], "query: ", "passage: "),
    "malteos/most-embed-de": ModelSpec([3e-6, 6e-6, 1e-5], "query: ", "passage: "),
}


# --- Run bookkeeping -------------------------------------------------------------------------------------------


def has_intact_weights(directory: Path) -> bool:
    weights = list(directory.rglob("*.safetensors")) + list(directory.rglob("pytorch_model*.bin"))
    return bool(weights) and all(path.stat().st_size for path in weights)


def resumable_checkpoint(directory: Path) -> Path | None:
    """Return the latest trainer checkpoint with complete weights, optimizer state and a matching step."""
    candidates = []
    for checkpoint in directory.glob("checkpoint-*"):
        step_text = checkpoint.name.removeprefix("checkpoint-")
        if not step_text.isdigit():
            continue
        if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size for name in RESUME_FILES):
            continue
        if not has_intact_weights(checkpoint):
            continue
        try:
            state = json.loads((checkpoint / "trainer_state.json").read_text())
        except (OSError, ValueError):
            continue
        if state.get("global_step") == int(step_text):
            candidates.append((int(step_text), checkpoint))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def completed_run(directory: Path, plan: dict[str, Any], model: str, seed: int) -> dict[str, Any] | None:
    """Return the result of a finished run, or None if it has not finished; reject a mismatching configuration."""
    summary = directory / "run.json"
    if not summary.exists():
        return None
    run = json.loads(summary.read_text())
    if run["plan"] != plan or run["model"] != model or run["seed"] != seed:
        raise ValueError(f"Completed run configuration differs: {directory}")
    if not (directory / "best/modules.json").is_file() or not has_intact_weights(directory / "best"):
        raise ValueError(f"Completed run has missing saved model files: {directory}")
    score = float(run["metrics"]["eval_val_balanced_ndcg_at_10"])
    if not math.isfinite(score):
        raise ValueError(f"Completed run has a non-finite score: {directory}")
    return {**plan, "score": score, "checkpoint": str(directory / "best")}


def write_results(directory: Path, results: list[dict[str, Any]], expected_runs: int) -> None:
    summary = {
        "runs": results,
        "best": max(results, key=lambda result: result["score"]),
        "complete": len(results) == expected_runs,
        "expected_runs": expected_runs,
    }
    temporary = directory / "results.tmp.json"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "results.json")


# --- Loss ------------------------------------------------------------------------------------------------------


class ExplicitNegativeLoss(nn.Module):
    """Cross-entropy over [positive, own negatives] cosine scores; other rows of the batch are ignored."""

    def __init__(self, model: nn.Module, scale: float = 20.0) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, sentence_features: list[dict[str, torch.Tensor]], labels: Any = None) -> torch.Tensor:
        if len(sentence_features) < 3:
            raise ValueError("Loss needs query, positive, and at least one explicit negative")
        embeddings = [F.normalize(self.model(features)["sentence_embedding"], dim=-1) for features in sentence_features]
        documents = torch.stack(embeddings[1:], dim=1)
        scores = torch.einsum("bd,bkd->bk", embeddings[0], documents) * self.scale
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, targets)

    def get_config_dict(self) -> dict[str, float]:
        return {"scale": self.scale}


# --- Planning --------------------------------------------------------------------------------------------------


def plan_sweep(model: str, available: int, maximum: int = 16000) -> list[dict[str, int | float]]:
    """Plan every learning rate x effective batch on one shared example budget, so trials are comparable."""
    budget = min(maximum, available // 2)
    budget -= budget % math.lcm(*SWEEP_BATCHES)
    if budget == 0:
        raise ValueError("Sweep needs at least 256 balanced examples and a budget of at least 128")
    return [
        dict(
            learning_rate=learning_rate,
            effective_batch=batch,
            examples=budget,
            max_steps=budget // batch,
            eval_steps=max(1, (budget // batch) // 5),
        )
        for learning_rate in MODELS[model].learning_rates
        for batch in SWEEP_BATCHES
    ]


def plan_pilot(available: int, learning_rate: float, effective_batch: int, steps: int) -> dict[str, int | float]:
    if learning_rate <= 0 or effective_batch <= 0 or steps <= 0:
        raise ValueError("Pilot learning rate, batch size and steps must be positive")
    budget = effective_batch * steps
    if budget > available:
        raise ValueError("Pilot budget exceeds the balanced training pool")
    return dict(
        learning_rate=learning_rate, effective_batch=effective_batch, examples=budget, max_steps=steps, eval_steps=steps
    )


def plan_final(available: int, learning_rate: float, effective_batch: int) -> dict[str, int | float]:
    updates = max(1, math.ceil(available / effective_batch))
    return dict(learning_rate=learning_rate, effective_batch=effective_batch, eval_steps=max(1, updates // 5))


def balanced_examples(examples: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the majority query level and interleave both levels."""
    if set(examples["level"]) != set(LEVELS):
        raise ValueError("Training needs eligible positives on both query levels")
    count = int(examples.groupby("level").size().min())
    groups = [
        examples[examples["level"] == level].sample(n=count, random_state=seed).reset_index(drop=True)
        for level in LEVELS
    ]
    return pd.concat(groups, keys=range(2)).sort_index(level=1, sort_remaining=True).reset_index(drop=True)


def pilot_validation(validation: pd.DataFrame, projects: int, seed: int) -> pd.DataFrame:
    relevant = validation[validation["score"] > 0].groupby(["project_id", "level"]).size().unstack(fill_value=0)
    relevant = relevant.reindex(columns=LEVELS, fill_value=0)
    eligible = sorted(relevant.index[(relevant > 0).all(axis=1)].tolist())
    if projects <= 0 or projects > len(eligible):
        raise ValueError("Pilot validation needs enough projects with relevant queries on both levels")
    selected = np.random.default_rng(seed).permutation(eligible)[:projects]
    return validation[validation["project_id"].isin(selected)].copy()


# --- Evaluation ------------------------------------------------------------------------------------------------


def graded_ndcg(scores: np.ndarray, relevance: np.ndarray, cutoff: int = 10) -> float | None:
    """Linear-gain nDCG; None when no passage is relevant (IDCG = 0)."""
    discounts = 1 / np.log2(np.arange(2, min(cutoff, len(relevance)) + 2))
    ideal = float((np.sort(relevance)[::-1][:cutoff] * discounts).sum())
    if ideal == 0:
        return None
    return float((relevance[np.argsort(-scores, kind="stable")[:cutoff]] * discounts).sum()) / ideal


def encode(model: Any, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    return model.encode(
        [prefix + text for text in texts],
        prompt="",
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def evaluate_projects(
    model: Any, validation: pd.DataFrame, query_prefix: str, passage_prefix: str, batch_size: int
) -> dict[str, float]:
    """Rank each project's judged passages per query; average per project, then over projects, per query level."""
    project_scores: dict[str, list[float]] = {level: [] for level in LEVELS}
    excluded = {level: 0 for level in LEVELS}
    for _, project in validation.groupby("project_id"):
        passages = project[["passage_uid", "text"]].drop_duplicates("passage_uid")
        passage_ids = passages["passage_uid"].tolist()
        passage_embeddings = encode(model, passages["text"].tolist(), passage_prefix, batch_size)
        for level in LEVELS:
            subset = project[project["level"] == level]
            queries = subset[["query_id", "query_text"]].drop_duplicates("query_id")
            if queries.empty:
                continue
            scores = encode(model, queries["query_text"].tolist(), query_prefix, batch_size) @ passage_embeddings.T
            values = []
            for index, query_id in enumerate(queries["query_id"]):
                judgments = subset[subset["query_id"] == query_id].set_index("passage_uid")["score"]
                if not set(passage_ids) <= set(judgments.index):
                    raise ValueError("Validation contains unjudged query-passage pairs")
                value = graded_ndcg(scores[index], judgments.reindex(passage_ids).to_numpy())
                if value is None:
                    excluded[level] += 1
                else:
                    values.append(value)
            if values:
                project_scores[level].append(float(np.mean(values)))
    if any(not project_scores[level] for level in LEVELS):
        raise ValueError("Validation needs relevant passages on both query levels")
    metrics = {f"val_{level}_ndcg_at_10": float(np.mean(project_scores[level])) for level in LEVELS}
    metrics["val_balanced_ndcg_at_10"] = float(np.mean(list(metrics.values())))
    for level in LEVELS:
        metrics[f"val_{level}_zero_relevance_queries"] = float(excluded[level])
        metrics[f"val_{level}_projects"] = float(len(project_scores[level]))
    return metrics


# --- Training --------------------------------------------------------------------------------------------------


def negative_columns(examples: pd.DataFrame) -> list[str]:
    columns = [c for c in examples if c.startswith("negative_") and not c.startswith("negative_uid_")]
    if not columns:
        raise ValueError("No explicit-negative text columns")
    return sorted(columns, key=lambda column: int(column.split("_")[-1]))


def run_training(
    args: argparse.Namespace,
    examples: pd.DataFrame,
    validation: pd.DataFrame,
    plan: dict[str, Any],
    directory: Path,
    checkpoint: Path | None = None,
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    # Heavy imports stay local so `--mode plan` and the unit tests of the helpers above start fast.
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import SentenceEvaluator
    from transformers import EarlyStoppingCallback, set_seed

    if plan["effective_batch"] % args.rows_per_gpu:
        raise ValueError("Effective batch must be divisible by physical batch size")
    set_seed(args.seed)
    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    if args.max_length is not None:
        model.max_seq_length = args.max_length
    elif args.model == "BAAI/bge-m3":
        model.max_seq_length = 8192
    model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    spec = MODELS[args.model]
    if (spec.query_prefix or spec.passage_prefix) and any(getattr(m, "include_prompt", True) is False for m in model):
        raise ValueError("Prompt-excluding pooling needs a verified training adapter")

    negatives = negative_columns(examples)
    texts = examples[["query", "positive", *negatives]].copy()
    texts["query"] = spec.query_prefix + texts["query"]
    for column in ["positive", *negatives]:
        texts[column] = spec.passage_prefix + texts[column]
    dataset = Dataset.from_pandas(texts, preserve_index=False)

    class ProjectEvaluator(SentenceEvaluator):
        def __init__(self) -> None:
            super().__init__()
            self.primary_metric = "val_balanced_ndcg_at_10"

        def __call__(self, model: Any, output_path: Any = None, epoch: int = -1, steps: int = -1) -> dict[str, float]:
            return evaluate_projects(model, validation, spec.query_prefix, spec.passage_prefix, args.eval_batch_size)

    schedule = (
        {"max_steps": plan["max_steps"]} if args.mode in ("pilot", "sweep") else {"num_train_epochs": args.epochs}
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(directory),
        learning_rate=plan["learning_rate"],
        per_device_train_batch_size=args.rows_per_gpu,
        gradient_accumulation_steps=plan["effective_batch"] // args.rows_per_gpu,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=plan["eval_steps"],
        save_strategy="steps",
        save_steps=plan["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_balanced_ndcg_at_10",
        greater_is_better=True,
        logging_steps=max(1, plan["eval_steps"] // 5),
        logging_nan_inf_filter=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_drop_last=True,
        seed=args.seed,
        report_to="none",
        **schedule,
    )
    evaluator = ProjectEvaluator()
    baseline = evaluator(model) if baseline_metrics is None else baseline_metrics
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=ExplicitNegativeLoss(model),
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if args.mode == "final" else [],
    )
    if checkpoint is not None:
        print(f"Resuming optimizer, scheduler and RNG state from {checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    metrics = trainer.evaluate()
    score = float(metrics["eval_val_balanced_ndcg_at_10"])
    if not math.isfinite(score):
        raise ValueError("Non-finite validation result; rejecting run")
    trainer.save_model(str(directory / "best"))
    run = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "max_length": model.max_seq_length,
        "rows_per_gpu": args.rows_per_gpu,
        "eval_batch_size": args.eval_batch_size,
        "negative_count": len(negatives),
        "validation_projects": sorted(validation["project_id"].unique().tolist()),
        "resumed_from": str(checkpoint) if checkpoint else None,
        "baseline_reused": baseline_metrics is not None,
        "plan": plan,
        "baseline": baseline,
        "metrics": metrics,
    }
    (directory / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**plan, "score": score, "checkpoint": str(directory / "best")}


# --- Command line ----------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["plan", "pilot", "sweep", "final"], default="plan")
    parser.add_argument("--model", choices=list(MODELS), default="BAAI/bge-m3")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory with examples.parquet and validation.parquet"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Fresh directory (existing runs are never overwritten)"
    )
    parser.add_argument("--sweep-examples-max", type=int, default=16000)
    parser.add_argument("--rows-per-gpu", type=int, default=2, help="Physical batch; the rest is gradient accumulation")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--pilot-steps", type=int, default=5)
    parser.add_argument("--pilot-validation-projects", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed sweep trials and resume partial ones")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() > 1:
        parser.error("Only one GPU is supported; select it with CUDA_VISIBLE_DEVICES")
    if args.mode != "plan" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable: use a GPU machine with a driver-compatible PyTorch build")
    if min(args.rows_per_gpu, args.eval_batch_size, args.epochs, args.effective_batch) <= 0:
        parser.error("Batch sizes and epochs must be positive")
    if args.resume and args.mode != "sweep":
        parser.error("--resume applies only to sweeps")
    if args.mode == "final" and (args.learning_rate is None or args.learning_rate <= 0):
        parser.error("Final training requires a positive, selected --learning-rate")
    return args


def build_plans(args: argparse.Namespace, available: int) -> list[dict[str, Any]]:
    if args.mode == "pilot":
        learning_rate = args.learning_rate if args.learning_rate is not None else MODELS[args.model].learning_rates[0]
        return [plan_pilot(available, learning_rate, args.effective_batch, args.pilot_steps)]
    if args.mode == "final":
        return [plan_final(available, args.learning_rate, args.effective_batch)]
    return plan_sweep(args.model, available, args.sweep_examples_max)


def prepare_output_directory(args: argparse.Namespace, plans: list[dict[str, Any]]) -> None:
    if args.output_dir.exists() and not args.resume:
        raise ValueError("Use a fresh output directory; existing runs will not be overwritten")
    plan_file = args.output_dir / "plan.json"
    if args.resume:
        if not plan_file.exists() or json.loads(plan_file.read_text()) != plans:
            raise ValueError("Resume requires an existing, identical sweep plan")
    else:
        args.output_dir.mkdir(parents=True)
        plan_file.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")


def run_plans(
    args: argparse.Namespace, examples: pd.DataFrame, validation: pd.DataFrame, plans: list[dict[str, Any]]
) -> None:
    results: list[dict[str, Any]] = []
    baseline_metrics = None
    validation_ids = sorted(validation["project_id"].unique().tolist())
    for index, plan in enumerate(plans):
        directory = args.output_dir / f"run_{index:02d}"
        if args.resume:
            previous = completed_run(directory, plan, args.model, args.seed)
            if previous is not None:
                saved = json.loads((directory / "run.json").read_text())
                if saved["validation_projects"] != validation_ids or saved["rows_per_gpu"] != args.rows_per_gpu:
                    raise ValueError("Resume validation projects or physical training batch differ")
                if args.max_length is not None and saved["max_length"] != args.max_length:
                    raise ValueError("Resume maximum sequence length differs")
                baseline_metrics = saved["baseline"] if baseline_metrics is None else baseline_metrics
                results.append(previous)
                write_results(args.output_dir, results, len(plans))
                print(f"Reusing completed {directory.name}: score={previous['score']:.6f}", flush=True)
                continue
        checkpoint = resumable_checkpoint(directory) if args.resume else None
        if args.resume and directory.exists() and checkpoint is None:
            raise ValueError(f"Partial run has no intact resume checkpoint: {directory}; inspect before restarting")
        subset = examples.iloc[: plan["examples"]] if args.mode in ("pilot", "sweep") else examples
        results.append(run_training(args, subset, validation, plan, directory, checkpoint, baseline_metrics))
        if baseline_metrics is None:
            baseline_metrics = json.loads((directory / "run.json").read_text())["baseline"]
        write_results(args.output_dir, results, len(plans))


def main() -> None:
    args = parse_args()
    original = pd.read_parquet(args.data_dir / "examples.parquet")
    examples = balanced_examples(original, args.seed)
    print(f"Balanced pool: {len(examples)} of {len(original)} examples; majority level is downsampled")
    plans = build_plans(args, len(examples))
    prepare_output_directory(args, plans)
    if args.mode == "plan":
        print(json.dumps(plans, indent=2))
        return
    validation = pd.read_parquet(args.data_dir / "validation.parquet")
    if set(examples["project_id"]) & set(validation["project_id"]):
        raise ValueError("Training and validation projects overlap")
    if args.mode == "pilot":
        validation = pilot_validation(validation, args.pilot_validation_projects, args.seed)
        print(
            f"Pilot: {plans[0]['examples']} training examples, {validation['project_id'].nunique()} validation projects"
        )
    run_plans(args, examples, validation, plans)


if __name__ == "__main__":
    main()
