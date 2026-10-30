#!/usr/bin/env python
"""Agreement analysis for one LLM-as-a-judge labeling run.

Reads ``results.jsonl`` written by ``run_labeling.py`` and writes, next to
that file:

- ``model_metrics.csv`` — per model and target: Cohen's κ on the full 0/1/2
  scale (``kappa_graded``) and on the collapsed 0 vs 1+2 scale
  (``kappa_binary``), each with a 95% bootstrap confidence interval; mean
  absolute error; false-positive rate (predicted relevant when the human
  label is 0); parse rate, score shares, tokens, latency, and cost.
- ``inter_llm.csv`` — pairwise Cohen's κ between models on the excerpts both
  scored, graded and binary, with bootstrap intervals.
- ``human_llm.csv`` — each model against each human annotator (1, 5, and 6),
  same κ pair, MAE, and false-positive rate.
- ``all_raters.csv`` — Fleiss' κ for the three human annotators and for the
  models together, graded and binary, with bootstrap intervals.
- ``human_agreement.csv`` — pairwise Cohen's κ among annotators 1, 5, and 6,
  graded and binary, with bootstrap intervals. This is the human ceiling.
- ``consistency.csv`` — how often the area score is lower than the criterion
  score, for the human labels and for each model.
- ``results_labeled.parquet`` — the checkpoint rows with truth labels
  re-derived from the dataset (best effort; the JSONL stays the source of
  truth).

It also refreshes the ``results.xlsx`` mirror of the JSONL checkpoint and
prints a model ranking: highest mean binary κ versus the consolidated human
label, then lower MAE, lower false-positive rate, then fewer completion
tokens.

Comparisons that cannot be computed (no overlapping excerpts, fewer than
three raters for Fleiss' κ, or a degenerate label distribution) are NaN rows
with an explanatory note.

Usage (from the repository root):
    python scripts/judge/analysis.py
    python scripts/judge/analysis.py --config scripts/judge/configs/model_reasoning_summary_aoa.yaml
    python scripts/judge/analysis.py --results data/judge/model_reasoning/simple_aoa/results.jsonl
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import judge_common as jc

# Number of paired bootstrap resamples per metric (stability vs. runtime).
N_BOOTSTRAP = 1000
# Fixed RNG seed so the reported bootstrap CIs are reproducible across runs.
SEED = 42
# The three human annotators compared pairwise and jointly (Fleiss' kappa).
ANNOTATORS = (1, 5, 6)


# ---------------------------------------------------------------- metrics

def _binary(values) -> np.ndarray:
    """Collapse graded relevance labels to a binary relevant / not-relevant flag.

    Values are coerced numerically; anything unparsable — including missing
    scores (NaN) or free-text labels — becomes -1 and is therefore mapped to
    0 ("not relevant"). Used to derive the binary kappa from 0/1/2 scores.

    :param values: sequence of numeric or numeric-string labels (0/1/2 scale).
    :returns: int array of 0/1 flags; 1 = "relevant" (label > 0).
    :rtype: numpy.ndarray
    """
    return (pd.to_numeric(pd.Series(values), errors="coerce").fillna(-1) > 0).astype(int).to_numpy()


def _cohen_kappa(y1, y2, labels) -> float:
    """Cohen's kappa with a pinned label set; NaN when only one class is present.

    Pinning ``labels`` keeps the confusion matrix at the intended shape when a
    class is missing from the sample. When ``y1`` and ``y2`` share fewer than
    two of those labels (typical bootstrap resamples of an all-zero slice),
    expected agreement is 1 and kappa is undefined — return NaN without
    calling sklearn, which would otherwise emit UndefinedMetricWarning.

    :param y1: first annotator's labels.
    :param y2: second annotator's labels.
    :param labels: category values to score over, e.g. [0, 1] or [0, 1, 2].
    :returns: kappa in [-1, 1], or NaN when undefined.
    :rtype: float
    """
    y1 = np.asarray(y1, int)
    y2 = np.asarray(y2, int)
    labels = np.asarray(labels, int)
    present = np.intersect1d(np.unique(np.concatenate([y1, y2])), labels)
    if present.size < 2:
        return float("nan")
    return float(cohen_kappa_score(y1, y2, labels=labels.tolist()))


def kappa_graded(y_true, y_pred) -> float:
    """Cohen's kappa between two label sequences on the full 0/1/2 scale.

    The label set is pinned to [0, 1, 2] so the confusion matrix stays 3x3
    even when one of the three classes is absent from the compared sample.

    :param y_true: reference labels (typically the human ground truth).
    :param y_pred: compared labels (typically the LLM scores).
    :returns: kappa in [-1, 1]; 1.0 = perfect agreement, 0.0 = chance level;
        NaN when both sequences use only one of {0, 1, 2}.
    :rtype: float
    """
    return _cohen_kappa(y_true, y_pred, [0, 1, 2])


def kappa_binary(y_true, y_pred) -> float:
    """Cohen's kappa after collapsing labels to relevant (> 0) vs not (0).

    The collapsed label set is pinned to [0, 1], matching :func:`kappa_graded`.

    :param y_true: reference labels on the 0/1/2 scale.
    :param y_pred: compared labels on the 0/1/2 scale.
    :returns: binary kappa in [-1, 1]; the primary selection metric because
        graded 1-vs-2 distinctions are weakly annotated in the ground truth.
        NaN when both sequences are a single class after collapsing.
    :rtype: float
    """
    return _cohen_kappa(_binary(y_true), _binary(y_pred), [0, 1])


def _total_cost(rows: pd.DataFrame) -> float:
    """Sum of provider-reported USD on ``rows``; NaN when cost is absent.

    Uses every row, including parse failures: those calls were still billed.

    :param rows: labeling checkpoint rows that may contain a ``cost`` column.
    :returns: sum of numeric ``cost`` values, or NaN when the column is
        missing, the frame is empty, or every value is null.
    :rtype: float
    """
    if rows is None or len(rows) == 0 or "cost" not in rows.columns:
        return float("nan")
    return float(pd.to_numeric(rows["cost"], errors="coerce").sum(min_count=1))


def mae(y_true, y_pred) -> float:
    """Mean absolute error between two label vectors on the 0/1/2 scale.

    Companion metric to kappa under the heavy zero-class imbalance: unlike
    kappa it stays sensitive to small systematic offsets.

    :param y_true: reference labels (0/1/2).
    :param y_pred: compared labels (0/1/2).
    :returns: mean of |y_true - y_pred| in score units; 0.0 = perfect.
    :rtype: float
    """
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def fp_rate(y_true, y_pred) -> float:
    """False-positive rate: share of human-0 passages the model scored >= 1.

    Measures over-labeling ("hallucinated relevance"), the main error mode of
    interest given that roughly 87-92% of excerpts are true zeros.

    :param y_true: human reference labels (0/1/2).
    :param y_pred: LLM labels (0/1/2).
    :returns: rate in [0, 1]; NaN when the sample contains no human-0 labels.
    :rtype: float
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    mask = y_true == 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[mask] > 0))


def bootstrap_ci(y_true, y_pred, metric, n=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap confidence interval for an agreement metric.

    Resamples (y_true, y_pred) pairs with replacement ``n`` times, recomputes
    ``metric`` on every resample and returns its 2.5th/97.5th percentiles.
    Resamples where the metric raises (e.g. a degenerate label set) contribute
    NaN and are ignored by the percentile computation.

    :param y_true: reference labels (0/1/2); resampled jointly with y_pred.
    :param y_pred: compared labels (0/1/2).
    :param metric: callable(y_true, y_pred) -> float, e.g. :func:`kappa_binary`
        or :func:`fp_rate`.
    :param n: number of bootstrap resamples (default: ``N_BOOTSTRAP`` = 1000).
    :param seed: RNG seed for reproducibility (default: ``SEED`` = 42).
    :returns: (lo, hi) percentile bounds; (NaN, NaN) if every resample failed.
    :rtype: tuple[float, float]
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            vals.append(metric(y_true[idx], y_pred[idx]))
        except Exception:  # noqa: BLE001
            vals.append(np.nan)
    lo, hi = np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])
    return float(lo), float(hi)


def _fleiss_kappa(ratings, categories) -> float:
    """Core Fleiss' kappa for a complete items x raters label matrix.

    Fleiss' kappa measures the agreement beyond chance among m >= 3 raters
    that all rated the same n items: p_i is the per-item observed agreement,
    p_bar their mean, p_j the pooled proportion of category j and
    kappa = (p_bar - pe) / (1 - pe). Returns NaN when the prerequisites fail:
    fewer than 3 raters, a missing label (incomplete matrix), labels outside
    ``categories``, or pe == 1 (all ratings fall into one category, which
    makes kappa mathematically undefined).

    :param ratings: 2-D array-like of shape (n_items, n_raters); every rater
        must have rated every item.
    :param categories: category values to tabulate, e.g. [0, 1, 2].
    :returns: Fleiss' kappa in [-1, 1], or NaN as described above.
    :rtype: float
    """
    ratings = np.asarray(ratings, float)
    if ratings.ndim != 2:
        return float("nan")
    n_items, n_raters = ratings.shape
    if (n_items < 1 or n_raters < 3 or not np.isfinite(ratings).all()
            or not np.isin(ratings, categories).all()):
        return float("nan")
    counts = np.stack([(ratings == cat).sum(axis=1) for cat in categories],
                      axis=1).astype(float)
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    pe = float(np.sum(p_j ** 2))
    if pe >= 1.0:
        return float("nan")
    return (p_bar - pe) / (1.0 - pe)


def fleiss_kappa_graded(ratings) -> float:
    """Fleiss' kappa across 3+ raters on the full 0/1/2 relevance scale.

    Multi-rater counterpart of :func:`kappa_graded`: labels stay on the
    0/1/2 scale (no collapsing) and every rater must have rated every item
    (complete items x raters matrix). NaN when the prerequisites fail (see
    :func:`_fleiss_kappa`); callers report such comparisons as NaN rows with
    a note instead of omitting them.

    :param ratings: 2-D array-like (items x raters) of 0/1/2 labels.
    :returns: Fleiss' kappa in [-1, 1], or NaN when uncomputable.
    :rtype: float
    """
    return _fleiss_kappa(ratings, [0, 1, 2])


def fleiss_kappa_binary(ratings) -> float:
    """Fleiss' kappa across 3+ raters on the collapsed 0 vs 1+2 scale.

    Multi-rater counterpart of :func:`kappa_binary`: labels are collapsed to
    relevant (> 0) vs not relevant (0) *before* aggregation — never feed raw
    0/1/2 labels here. Requires a complete items x raters matrix with at
    least 3 raters; NaN otherwise (see :func:`_fleiss_kappa`).

    :param ratings: 2-D array-like (items x raters) of 0/1/2 labels.
    :returns: binary Fleiss' kappa in [-1, 1], or NaN when uncomputable.
    :rtype: float
    """
    ratings = np.asarray(ratings, float)
    if not np.isfinite(ratings).all():
        return float("nan")
    return _fleiss_kappa((ratings > 0).astype(float), [0, 1])


def bootstrap_ci_raters(ratings, metric, n=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap confidence interval for a multi-rater metric.

    Multi-rater counterpart of :func:`bootstrap_ci`: instead of resampling
    (y_true, y_pred) pairs, whole *items* (rows of ``ratings``) are resampled
    with replacement ``n`` times, ``metric`` is recomputed on every resample
    and its 2.5th/97.5th percentiles returned. Resamples where the metric
    yields NaN (e.g. a degenerate label distribution with pe == 1) contribute
    NaN and are ignored by the percentile computation.

    :param ratings: 2-D array-like (items x raters) resampled row-wise.
    :param metric: callable(ratings_resample) -> float, e.g.
        :func:`fleiss_kappa_graded`.
    :param n: number of bootstrap resamples (default: ``N_BOOTSTRAP`` = 1000).
    :param seed: RNG seed for reproducibility (default: ``SEED`` = 42).
    :returns: (lo, hi) percentile bounds; (NaN, NaN) if every resample failed.
    :rtype: tuple[float, float]
    """
    ratings = np.asarray(ratings, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(ratings), len(ratings))
        try:
            vals.append(metric(ratings[idx]))
        except Exception:  # noqa: BLE001
            vals.append(np.nan)
    lo, hi = np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------ computation

# Agreement metrics that receive companion <name>_lo/<name>_hi bootstrap-bound
# columns in every row built by model_target_metrics().
_METRIC_KEYS = ("kappa_graded", "kappa_binary", "mae", "fp_rate")


def model_target_metrics(results: pd.DataFrame, model: str, target: str) -> dict:
    """LLM-vs-human agreement metrics for one model on one target.

    Metric columns use only rows with a parsed score (o_score) and a known
    truth label; ``parse_rate`` is computed over *all* rows of the model/target
    pair, including unparsed ones. Every metric in :data:`_METRIC_KEYS` is
    accompanied by ``<name>_lo`` / ``<name>_hi`` 95% bootstrap bounds. Both
    kappa types are always reported: ``kappa_graded`` on the full 0/1/2 scale
    and ``kappa_binary`` after collapsing to relevant (> 0) vs not (0).

    :param results: labeling checkpoint rows; needs at least the columns
        model, target, o_score, truth, parse_ok and optionally
        completion_tokens / latency_s (missing runtime columns degrade the
        corresponding averages to NaN).
    :param model: model name as stored in ``results["model"]``.
    :param target: "area" or "criterion".
    :returns: one flat dict: identity (model, target), n, parse_rate,
        predicted-label shares pct_0/pct_1/pct_2, avg_completion_tokens,
        avg_latency_s, total_cost, the four metrics with their bootstrap
        bounds, and a ``note`` explaining any NaN kappa. When the pair has
        no usable rows, a complete all-NaN key set (plus the note) is
        returned so downstream aggregation never encounters missing columns.
    :rtype: dict
    """
    all_rows = results[(results["model"] == model) & (results["target"] == target)]
    sub = all_rows[all_rows["o_score"].notna() & all_rows["truth"].notna()]
    if len(sub) == 0:
        # complete key set (all NaN) so downstream aggregation never hits
        # missing columns when a configured model has no rows in the results
        row = {"model": model, "target": target, "n": 0, "parse_rate": np.nan,
               "pct_0": np.nan, "pct_1": np.nan, "pct_2": np.nan,
               "avg_completion_tokens": np.nan, "avg_latency_s": np.nan,
               "total_cost": _total_cost(all_rows)}
        for name in _METRIC_KEYS:
            row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = np.nan
        row["note"] = "no rows with parsed score and truth label"
        return row
    y_true, y_pred = sub["truth"].to_numpy(int), sub["o_score"].to_numpy(int)
    row = {
        "model": model,
        "target": target,
        "n": len(sub),
        "parse_rate": float(all_rows["parse_ok"].mean()) if len(all_rows) else np.nan,
        "pct_0": float(np.mean(y_pred == 0)),
        "pct_1": float(np.mean(y_pred == 1)),
        "pct_2": float(np.mean(y_pred == 2)),
        "avg_completion_tokens": float(sub["completion_tokens"].mean()) if "completion_tokens" in sub else np.nan,
        "avg_latency_s": float(sub["latency_s"].mean()) if "latency_s" in sub else np.nan,
        "total_cost": _total_cost(all_rows),
    }
    for name, fn in (("kappa_graded", kappa_graded), ("kappa_binary", kappa_binary),
                     ("mae", mae), ("fp_rate", fp_rate)):
        row[name] = fn(y_true, y_pred)
        row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(y_true, y_pred, fn)
    if any(np.isnan(row[name]) for name in ("kappa_graded", "kappa_binary")):
        row["note"] = "kappa undefined on this sample (degenerate label distribution)"
    else:
        row["note"] = ""
    return row


def inter_llm_metrics(results: pd.DataFrame, models: list, target: str) -> list:
    """Inter-LLM agreement for every unordered model pair on one target.

    For each pair (a, b) both models' parsed scores are joined on sample_id
    and restricted to excerpts that both scored and that have a truth label.
    Row semantics: ``kappa_graded`` (+ CIs) is model-a vs the human gold
    standard on the full 0/1/2 scale over those common excerpts,
    ``kappa_graded_model_b`` (+ CIs) the same for model-b, and ``kappa_binary``
    (+ CIs) the binary model-a vs model-b agreement after collapsing to
    relevant (> 0) vs not relevant (0). Every configured/discovered pair is
    reported; pairs without jointly scored excerpts get n_common = 0 and NaN
    kappas with an explanatory note instead of being omitted.

    :param results: labeling checkpoint rows.
    :param models: model names to pair up (order defines the enumeration).
    :param target: "area" or "criterion".
    :returns: one dict per pair: target, model_a, model_b, n_common,
        kappa_graded (+_lo/_hi), kappa_graded_model_b (+_lo/_hi),
        kappa_binary (+_lo/_hi), note.
    :rtype: list[dict]
    """
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            sa = results[(results["model"] == a) & (results["target"] == target)
                         & results["o_score"].notna() & results["truth"].notna()]
            sb = results[(results["model"] == b) & (results["target"] == target)
                         & results["o_score"].notna() & results["truth"].notna()]
            common = sa[["sample_id", "o_score", "truth"]].merge(
                sb[["sample_id", "o_score"]], on="sample_id", suffixes=("_a", "_b"))
            row = {"target": target, "model_a": a, "model_b": b,
                   "n_common": len(common), "note": ""}
            for name in ("kappa_graded", "kappa_graded_model_b", "kappa_binary"):
                row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = np.nan
            if len(common) == 0:
                row["note"] = "no excerpts commonly rated by both models"
                rows.append(row)
                continue
            y_true = common["truth"].to_numpy(int)
            ya, yb = common["o_score_a"].to_numpy(int), common["o_score_b"].to_numpy(int)
            row["kappa_graded"] = kappa_graded(y_true, ya)
            row["kappa_graded_lo"], row["kappa_graded_hi"] = bootstrap_ci(y_true, ya, kappa_graded)
            row["kappa_graded_model_b"] = kappa_graded(y_true, yb)
            row["kappa_graded_model_b_lo"], row["kappa_graded_model_b_hi"] = bootstrap_ci(y_true, yb, kappa_graded)
            row["kappa_binary"] = kappa_binary(ya, yb)
            row["kappa_binary_lo"], row["kappa_binary_hi"] = bootstrap_ci(ya, yb, kappa_binary)
            if any(np.isnan(row[name]) for name in
                   ("kappa_graded", "kappa_graded_model_b", "kappa_binary")):
                row["note"] = "kappa undefined on this sample (degenerate label distribution)"
            rows.append(row)
    return rows


def _annotator_int(series: pd.Series) -> pd.Series:
    """Extract the numeric part of annotator labels ("2 – Klar relevant" -> 2).

    :param series: raw annotator label column (mix of graded and text labels).
    :returns: numeric series of the first integer found per label; NaN for
        non-numeric labels (e.g. "Nicht auswertbar"), which are excluded
        from the agreement computation downstream.
    :rtype: pandas.Series
    """
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def human_reference(df: pd.DataFrame, target: str) -> list:
    """Human inter-annotator agreement — the reference ceiling (steps 1.4/1.5).

    Experiment-doc steps 1.4/1.5 are fulfilled by the existing three-annotator
    labels, so no new human coding is collected; instead the pairwise agreement
    among annotators 1, 5 and 6 on this target is reported as the ceiling that
    LLM-vs-human agreement should be read against. Every pair reports both
    kappa types with 95% bootstrap CIs — ``kappa_graded`` on the full 0/1/2
    scale and ``kappa_binary`` after collapsing to relevant (> 0) vs not (0) —
    plus MAE. Pairs without jointly valid 0/1/2 labels are reported with
    n = 0 and NaN metrics instead of being omitted; the overall three-annotator
    agreement (Fleiss' kappa) is computed by :func:`all_raters_metrics`.

    :param df: curated dataset frame containing the annotator_* columns.
    :param target: "area" (columns annotator_<i>_area_relevance) or "criterion"
        (columns annotator_<i>_criterion_relevance).
    :returns: one dict per annotator pair with n, kappa_graded (+_lo/_hi),
        kappa_binary (+_lo/_hi), mae and note.
    :rtype: list[dict]
    """
    suffix = "area" if target == "area" else "criterion"
    rows = []
    for a, b in itertools.combinations(ANNOTATORS, 2):
        ya = _annotator_int(df[f"annotator_{a}_{suffix}_relevance"])
        yb = _annotator_int(df[f"annotator_{b}_{suffix}_relevance"])
        mask = ya.notna() & yb.notna() & ya.isin([0, 1, 2]) & yb.isin([0, 1, 2])
        row = {"target": target, "annotator_pair": f"{a} vs {b}", "n": int(mask.sum()), "note": ""}
        for name in ("kappa_graded", "kappa_binary"):
            row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = np.nan
        row["mae"] = np.nan
        if mask.sum() == 0:
            row["note"] = "no excerpts with jointly valid 0/1/2 labels"
            rows.append(row)
            continue
        yt, yp = ya[mask].to_numpy(int), yb[mask].to_numpy(int)
        row["kappa_graded"] = kappa_graded(yt, yp)
        row["kappa_graded_lo"], row["kappa_graded_hi"] = bootstrap_ci(yt, yp, kappa_graded)
        row["kappa_binary"] = kappa_binary(yt, yp)
        row["kappa_binary_lo"], row["kappa_binary_hi"] = bootstrap_ci(yt, yp, kappa_binary)
        row["mae"] = mae(yt, yp)
        if np.isnan(row["kappa_graded"]) or np.isnan(row["kappa_binary"]):
            row["note"] = "kappa undefined on this sample (degenerate label distribution)"
        rows.append(row)
    return rows


def annotator_llm_metrics(df: pd.DataFrame, results: pd.DataFrame, models: list, target: str) -> list:
    """LLM-vs-individual-annotator agreement for one target (step 1.6).

    Complements the LLM-vs-gold-standard metrics (consolidated labels) with
    the same comparison against each individual annotator: every analyzed
    model is compared to annotators 1, 5 and 6 on their commonly rated
    excerpts. Both kappa types are reported with 95% bootstrap CIs —
    ``kappa_graded`` on the full 0/1/2 scale and ``kappa_binary`` after
    collapsing to relevant (> 0) vs not relevant (0) — plus MAE and
    false-positive-rate point estimates. Raw annotator labels
    ("2 – Klar relevant") are reduced to their numeric part via
    :func:`_annotator_int`; excerpts without a jointly valid 0/1/2 label pair
    are excluded from both metrics.

    :param df: curated dataset frame containing the annotator_* columns.
    :param results: labeling checkpoint rows (needs model, target, o_score).
    :param models: model names to compare (one row per model and annotator,
        even when the comparison cannot be computed).
    :param target: "area" or "criterion".
    :returns: one dict per (model, annotator): target, model, annotator, n,
        kappa_graded (+_lo/_hi), kappa_binary (+_lo/_hi), mae, fp_rate, note.
        Uncomputable comparisons are reported as NaN rows with a note.
    :rtype: list[dict]
    """
    suffix = "area" if target == "area" else "criterion"
    rows = []
    for model in models:
        sub = results[(results["model"] == model) & (results["target"] == target)
                      & results["o_score"].notna()]
        scores = None
        if len(sub):
            scores = pd.Series(sub["o_score"].to_numpy(float),
                               index=sub["sample_id"].astype(str), name="llm_score")
            scores = scores[~scores.index.duplicated(keep="last")]
        for annotator in ANNOTATORS:
            row = {"target": target, "model": model, "annotator": annotator,
                   "n": 0, "mae": np.nan, "fp_rate": np.nan, "note": ""}
            for name in ("kappa_graded", "kappa_binary"):
                row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = np.nan
            col = f"annotator_{annotator}_{suffix}_relevance"
            if scores is None:
                row["note"] = "model has no parsed scores for this target"
                rows.append(row)
                continue
            if col not in df.columns:
                row["note"] = "no such annotator column in the dataset"
                rows.append(row)
                continue
            ann_df = df[["sample_id", col]].copy()
            ann_df["sample_id"] = ann_df["sample_id"].astype(str)
            ann = pd.Series(_annotator_int(ann_df[col]).to_numpy(),
                            index=ann_df["sample_id"].to_numpy(), name="annotator_score")
            ann = ann[~ann.index.duplicated(keep="last")]
            common = pd.concat([scores, ann], axis=1, join="inner")
            valid = common["annotator_score"].isin([0, 1, 2]) & common["llm_score"].isin([0, 1, 2])
            common = common[valid]
            row["n"] = int(len(common))
            if len(common) == 0:
                row["note"] = "no excerpts with jointly valid 0/1/2 labels"
                rows.append(row)
                continue
            y_true = common["annotator_score"].to_numpy(int)
            y_pred = common["llm_score"].to_numpy(int)
            row["kappa_graded"] = kappa_graded(y_true, y_pred)
            row["kappa_graded_lo"], row["kappa_graded_hi"] = bootstrap_ci(y_true, y_pred, kappa_graded)
            row["kappa_binary"] = kappa_binary(y_true, y_pred)
            row["kappa_binary_lo"], row["kappa_binary_hi"] = bootstrap_ci(y_true, y_pred, kappa_binary)
            row["mae"] = mae(y_true, y_pred)
            row["fp_rate"] = fp_rate(y_true, y_pred)
            if any(np.isnan(row[name]) for name in ("kappa_graded", "kappa_binary")):
                row["note"] = "kappa undefined on this sample (degenerate label distribution)"
            rows.append(row)
    return rows


def _fleiss_row(target: str, scope: str, ratings) -> dict:
    """Build one all_raters row for a complete items x raters matrix.

    Computes Fleiss' kappa on the graded 0/1/2 scale and on the collapsed
    binary scale, each with 95% bootstrap bounds over items, and records
    NaN + an explanatory ``note`` whenever the prerequisites fail (fewer
    than 3 raters, no commonly rated items, or a degenerate label
    distribution that makes kappa undefined).

    :param target: "area" or "criterion".
    :param scope: human-readable rater-set description ("humans (1, 5, 6)",
        "LLMs only").
    :param ratings: 2-D array-like (items x raters) of 0/1/2 labels; every
        rater must have rated every item.
    :returns: flat dict for the all_raters.csv table.
    :rtype: dict
    """
    ratings = np.asarray(ratings, float)
    n_items, n_raters = ratings.shape if ratings.ndim == 2 else (0, 0)
    row = {"target": target, "scope": scope, "n_raters": int(n_raters),
           "n_items": int(n_items)}
    for name, fn in (("fleiss_kappa_graded", fleiss_kappa_graded),
                     ("fleiss_kappa_binary", fleiss_kappa_binary)):
        row[name] = fn(ratings)
        if n_items:
            row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci_raters(ratings, fn)
        else:
            row[f"{name}_lo"] = row[f"{name}_hi"] = np.nan
    if n_raters < 3:
        row["note"] = "insufficient raters for Fleiss' kappa (needs >= 3)"
    elif n_items == 0:
        row["note"] = "no items rated by all raters"
    elif any(np.isnan(row[name]) for name in ("fleiss_kappa_graded", "fleiss_kappa_binary")):
        row["note"] = "kappa undefined on this sample (degenerate label distribution)"
    else:
        row["note"] = ""
    return row


def all_raters_metrics(df: pd.DataFrame, results: pd.DataFrame, models: list, target: str) -> list:
    """Overall multi-rater agreement (Fleiss' kappa) for one target.

    Two scopes per target, both requiring every rater to have rated the same
    items (complete items x raters matrix):

    - ``humans (1, 5, 6)``: the three human annotators on the excerpts where
      all of them gave a jointly valid 0/1/2 label;
    - ``LLMs only``: all analyzed models on the excerpts every model parsed
      with a known truth label (same restriction as the pairwise tables).

    Fleiss' kappa is reported on the graded 0/1/2 scale and on the collapsed
    binary scale, each with 95% bootstrap CIs over items. Scopes with fewer
    than 3 raters or an empty intersection are reported as NaN rows with a
    note instead of being omitted.

    :param df: curated dataset frame containing the annotator_* columns.
    :param results: labeling checkpoint rows.
    :param models: model names forming the LLM rater set.
    :param target: "area" or "criterion".
    :returns: one dict per scope (see :func:`_fleiss_row` for the keys).
    :rtype: list[dict]
    """
    suffix = "area" if target == "area" else "criterion"
    ann_cols = [f"annotator_{a}_{suffix}_relevance" for a in ANNOTATORS]
    if all(col in df.columns for col in ann_cols):
        ann = pd.concat([_annotator_int(df[col]) for col in ann_cols], axis=1)
        mask = ann.notna().all(axis=1)
        for col in ann_cols:
            mask &= ann[col].isin([0, 1, 2])
        human_ratings = ann[mask].to_numpy(float)
    else:
        human_ratings = np.empty((0, len(ANNOTATORS)))
    score_frames = []
    for model in models:
        sub = results[(results["model"] == model) & (results["target"] == target)
                      & results["o_score"].notna() & results["truth"].notna()]
        if len(sub) == 0:
            continue
        scores = pd.Series(sub["o_score"].to_numpy(float),
                           index=sub["sample_id"].astype(str), name=model)
        score_frames.append(scores[~scores.index.duplicated(keep="last")])
    llm_ratings = np.empty((0, 0))
    if score_frames:
        llm_ratings = pd.concat(score_frames, axis=1, join="inner").to_numpy(float)
    return [
        _fleiss_row(target, "humans (" + ", ".join(str(a) for a in ANNOTATORS) + ")", human_ratings),
        _fleiss_row(target, "LLMs only", llm_ratings),
    ]


def consistency_metrics(df: pd.DataFrame, results: pd.DataFrame) -> list:
    """Rate of "area >= criterion" hierarchy-rule violations.

    A violation is a passage whose Handlungsfeld (area) score is *below* its
    criterion score, contradicting the annotation rule that field-level
    relevance can never be lower than criterion-level relevance. Reported once
    for the human ground truth and once per model (join on sample_id over each
    model's parsed area and criterion scores).

    :param df: curated dataset frame with int_rel_area / int_rel_criterion.
    :param results: labeling checkpoint rows.
    :returns: list of dicts with scope ("human (ground truth)" or
        "model <name>"), n and violation_rate; scopes without a joinable
        area/criterion pair are skipped.
    :rtype: list[dict]
    """
    rows = []
    truth_area = df.set_index("sample_id")["int_rel_area"]
    truth_crit = df.set_index("sample_id")["int_rel_criterion"]
    aligned = pd.concat([truth_area, truth_crit], axis=1).dropna()
    rows.append({"scope": "human (ground truth)",
                 "n": len(aligned),
                 "violation_rate": float((aligned["int_rel_area"] < aligned["int_rel_criterion"]).mean())})
    for model in results["model"].unique():
        wa = results[(results["model"] == model) & (results["target"] == "area") & results["o_score"].notna()]
        wc = results[(results["model"] == model) & (results["target"] == "criterion") & results["o_score"].notna()]
        merged = wa[["sample_id", "o_score"]].merge(
            wc[["sample_id", "o_score"]], on="sample_id", suffixes=("_area", "_crit"))
        if len(merged) == 0:
            continue
        rows.append({"scope": f"model {model}", "n": len(merged),
                     "violation_rate": float((merged["o_score_area"] < merged["o_score_crit"]).mean())})
    return rows


def best_model_ranking(model_metrics: pd.DataFrame) -> pd.DataFrame:
    """Step 1.7 best-model selection ranking across targets.

    Models are ranked by mean LLM-vs-human ``kappa_binary`` (human agreement
    has priority), tie-broken by lower ``mean_mae``, then lower
    ``mean_fp_rate``, then fewer ``mean_completion_tokens``. ``total_cost``
    (provider-reported USD, summed across targets) is reported but not used
    as a sort key. Models without any parsed rows (n == 0, e.g. configured
    but not present in the analyzed results file) are excluded.

    :param model_metrics: per model/target rows as built by
        :func:`model_target_metrics`.
    :returns: DataFrame with columns rank, model, mean_kappa_binary,
        mean_kappa_graded, mean_mae, mean_fp_rate, mean_completion_tokens,
        total_cost and min_parse_rate; rank 1 is the recommended model.
        Returns an empty frame with the same columns when no model has
        usable rows.
    :rtype: pandas.DataFrame
    """
    ranking_cols = ["rank", "model", "mean_kappa_binary", "mean_kappa_graded",
                    "mean_mae", "mean_fp_rate", "mean_completion_tokens",
                    "total_cost", "min_parse_rate"]
    usable = model_metrics[model_metrics["n"] > 0]
    if len(usable) == 0:
        return pd.DataFrame(columns=ranking_cols)
    aggs = dict(mean_kappa_binary=("kappa_binary", "mean"),
                mean_kappa_graded=("kappa_graded", "mean"),
                mean_mae=("mae", "mean"),
                mean_fp_rate=("fp_rate", "mean"),
                mean_completion_tokens=("avg_completion_tokens", "mean"),
                min_parse_rate=("parse_rate", "min"))
    if "total_cost" in usable.columns:
        aggs["total_cost"] = ("total_cost", "sum")
    agg = (usable.groupby("model")
           .agg(**aggs)
           .reset_index()
           .sort_values(["mean_kappa_binary", "mean_mae", "mean_fp_rate", "mean_completion_tokens"],
                        ascending=[False, True, True, True])
           .reset_index(drop=True))
    if "total_cost" not in agg.columns:
        agg["total_cost"] = np.nan
    agg.insert(0, "rank", range(1, len(agg) + 1))
    return agg[ranking_cols]



def main():
    """CLI entry point: run the full analysis for one results directory.

    Flow: parse arguments -> load the experiment config -> resolve the output
    directory (``--out-dir`` wins; else the ``--results`` file's directory;
    else ``paths.results_dir`` from the config) -> load the labeling
    checkpoint (exits when missing or empty) -> re-derive truth labels from
    the curated dataset and warn about stored-label mismatches or unmatched
    rows -> union configured models/targets with those actually present in the
    file -> write the metric tables and refresh the Excel mirror -> print the
    ranking table to stdout.

    :raises SystemExit: when the results file contains no rows.
    """
    p = argparse.ArgumentParser(description="LLM-as-a-judge agreement analysis")
    p.add_argument("--config", default="configs/model_reasoning_simple_aoa.yaml")
    p.add_argument("--results", default=None,
                   help="path to results.jsonl (models and targets are discovered "
                        "from the file and unioned with the config)")
    p.add_argument("--out-dir", default=None,
                   help="override output dir (default: the results file's directory, "
                        "or paths.results_dir when --results is not given)")
    args = p.parse_args()

    cfg = jc.load_config(args.config)
    if args.out_dir:
        out_dir = jc.resolve_path(args.out_dir)
    elif args.results:
        out_dir = Path(args.results).resolve().parent
    else:
        out_dir = jc.resolve_path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results) if args.results else out_dir / "results.jsonl"
    results = jc.read_results(results_path)
    if len(results) == 0:
        raise SystemExit(f"No results found at {results_path}")

    df = jc.load_dataset(cfg)
    results["sample_id"] = results["sample_id"].astype(str)
    truth_map = df[["sample_id", "int_rel_area", "int_rel_criterion"]].copy()
    truth_map["sample_id"] = truth_map["sample_id"].astype(str)
    results = results.merge(truth_map, on="sample_id", how="left")
    fresh = np.where(results["target"] == "area", results["int_rel_area"], results["int_rel_criterion"])
    stored = pd.to_numeric(results["truth"], errors="coerce").to_numpy()
    fresh_num = pd.to_numeric(pd.Series(fresh), errors="coerce").to_numpy()
    mismatch = int((results["o_score"].notna() & (stored != fresh_num)).sum())
    if mismatch:
        print(f"WARNING: {mismatch} rows have a stored truth label differing from the dataset")
    results["truth"] = fresh_num
    unlabeled = int(results["truth"].isna().sum())
    if unlabeled:
        print(f"WARNING: {unlabeled} rows could not be matched to a dataset truth label; "
              f"they are excluded from the metrics")

    config_models = [m["name"] for m in cfg["models"]]
    extra_models = sorted(m for m in results["model"].dropna().astype(str).unique()
                          if m not in config_models)
    if extra_models:
        print(f"NOTE: results contain {len(extra_models)} model(s) not in the config; "
              f"analyzing them too: {', '.join(extra_models)}")
    models = config_models + extra_models
    targets = list(cfg["targets"]) + sorted(t for t in results["target"].dropna().astype(str).unique()
                                            if t not in cfg["targets"])
    model_metrics = pd.DataFrame([model_target_metrics(results, m, t) for m in models for t in targets])
    inter_llm = [row for t in targets for row in inter_llm_metrics(results, models, t)]
    human_rows = [row for t in targets for row in human_reference(df, t)]
    annotator_rows = [row for t in targets for row in annotator_llm_metrics(df, results, models, t)]
    all_raters = [row for t in targets for row in all_raters_metrics(df, results, models, t)]
    consistency = consistency_metrics(df, results)

    outputs = {
        "model_metrics.csv": model_metrics,
        "inter_llm.csv": pd.DataFrame(inter_llm),
        "human_llm.csv": pd.DataFrame(annotator_rows),
        "all_raters.csv": pd.DataFrame(all_raters),
        "human_agreement.csv": pd.DataFrame(human_rows),
        "consistency.csv": pd.DataFrame(consistency),
    }
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        print(f"Wrote: {path}")
    try:
        parquet_path = out_dir / "results_labeled.parquet"
        results.to_parquet(parquet_path, index=False)
        print(f"Wrote: {parquet_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not write parquet ({exc}); JSONL remains the source of truth.")

    xlsx = jc.export_results_xlsx(results_path)
    if xlsx:
        print(f"Wrote: {xlsx}")
    rank = best_model_ranking(model_metrics)
    print("\nRanking:")
    print(rank.round(3).to_string(index=False))


if __name__ == "__main__":
    main()