#!/usr/bin/env python
"""Unit tests for analysis agreement metrics (Cohen's + Fleiss' kappa).

Covers the graded (0/1/2) vs binary (0 vs 1+2) kappa helpers, the Fleiss'
kappa multi-rater helpers (hand-computed values and NaN prerequisites), the
row builders (human pairs, LLM pairs, LLM-vs-annotator, all-raters scopes)
and their bootstrap CIs. Runnable directly (no pytest required):

    .venv/bin/python scripts/judge/tests/test_agreement.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as ex  # noqa: E402


def test_kappa_graded_pins_label_set():
    # a class missing from the sample must not change the pinned [0, 1, 2] kappa
    yt, yp = [0, 0, 1, 1], [0, 1, 1, 1]
    assert ex.kappa_graded(yt, yp) == cohen_kappa_score(yt, yp, labels=[0, 1, 2])


def test_kappa_binary_collapses():
    # 0/1/2 -> 0 vs 1+2 collapsed *before* scoring, never the raw labels
    assert abs(ex.kappa_binary([0, 0, 1, 2], [0, 1, 1, 0]) - cohen_kappa_score(
        [0, 0, 1, 1], [0, 1, 1, 0], labels=[0, 1])) < 1e-12


def test_kappa_binary_pins_label_set():
    # a class missing from the collapsed sample must not change the pinned [0, 1] kappa
    yt, yp = [0, 0, 1, 1], [0, 0, 0, 1]
    assert ex.kappa_binary(yt, yp) == cohen_kappa_score(
        [0, 0, 1, 1], [0, 0, 0, 1], labels=[0, 1])


def test_kappa_undefined_single_class_is_silent_nan():
    # all-zero (or all-one) pairs are undefined; do not call sklearn (no warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(ex.kappa_binary([0, 0, 0], [0, 0, 0]))
        assert np.isnan(ex.kappa_graded([0, 0, 0], [0, 0, 0]))
        assert np.isnan(ex.kappa_binary([1, 2, 1], [2, 1, 2]))


def test_ranking_includes_total_cost():
    metrics = pd.DataFrame([
        {"model": "a", "target": "area", "n": 10, "kappa_binary": 0.5, "kappa_graded": 0.4,
         "mae": 0.1, "fp_rate": 0.05, "avg_completion_tokens": 100, "parse_rate": 1.0,
         "total_cost": 0.01},
        {"model": "a", "target": "criterion", "n": 10, "kappa_binary": 0.4, "kappa_graded": 0.3,
         "mae": 0.1, "fp_rate": 0.05, "avg_completion_tokens": 80, "parse_rate": 1.0,
         "total_cost": 0.02},
        {"model": "b", "target": "area", "n": 10, "kappa_binary": 0.2, "kappa_graded": 0.2,
         "mae": 0.2, "fp_rate": 0.1, "avg_completion_tokens": 50, "parse_rate": 1.0,
         "total_cost": 0.005},
        {"model": "b", "target": "criterion", "n": 10, "kappa_binary": 0.2, "kappa_graded": 0.2,
         "mae": 0.2, "fp_rate": 0.1, "avg_completion_tokens": 50, "parse_rate": 1.0,
         "total_cost": 0.005},
    ])
    rank = ex.best_model_ranking(metrics)
    assert list(rank.columns)[-2] == "total_cost"
    assert rank.iloc[0]["model"] == "a"
    assert abs(rank.iloc[0]["total_cost"] - 0.03) < 1e-12
    assert abs(rank.iloc[1]["total_cost"] - 0.01) < 1e-12


def test_fleiss_graded_hand_computed():
    # 4 items x 3 raters; P_bar = 2/3, pe = 0.5 -> kappa = 1/3
    ratings = [[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]]
    assert abs(ex.fleiss_kappa_graded(ratings) - 1.0 / 3.0) < 1e-12


def test_fleiss_graded_perfect_agreement():
    # mixed categories (pe < 1) -> kappa = 1
    ratings = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0], [2, 2, 2]]
    assert ex.fleiss_kappa_graded(ratings) == 1.0


def test_fleiss_binary_collapses_correctly():
    graded = [[0, 0, 2], [1, 1, 1], [0, 2, 0], [2, 2, 2]]
    manual = [[0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1]]
    assert ex.fleiss_kappa_binary(graded) == ex.fleiss_kappa_graded(manual)
    # hand-computed on the collapsed 0/1 scale: P_bar = 2/3, pe = 5/9
    assert abs(ex.fleiss_kappa_binary(graded) - 0.25) < 1e-12


def test_fleiss_requires_three_raters():
    assert np.isnan(ex.fleiss_kappa_graded([[0, 0], [1, 1], [2, 2]]))


def test_fleiss_requires_complete_matrix():
    # a missing label makes the items x raters matrix incomplete
    assert np.isnan(ex.fleiss_kappa_graded([[0, 0, np.nan], [1, 1, 2]]))


def test_fleiss_undefined_when_single_category():
    # all ratings in one category: pe == 1 -> kappa mathematically undefined
    assert np.isnan(ex.fleiss_kappa_graded([[0, 0, 0], [0, 0, 0]]))


def test_fleiss_rejects_out_of_range_labels():
    assert np.isnan(ex.fleiss_kappa_graded([[0, 1, 3], [1, 2, 2]]))


def test_bootstrap_ci_raters_brackets_point_estimate():
    rng = np.random.default_rng(0)
    ratings = np.column_stack([
        np.concatenate([np.zeros(90), np.ones(10)]),
        np.concatenate([np.zeros(95), np.ones(5)]),
        np.concatenate([np.zeros(90), np.ones(10)]),
    ])
    point = ex.fleiss_kappa_binary(ratings)
    lo, hi = ex.bootstrap_ci_raters(ratings, ex.fleiss_kappa_binary, n=200, seed=1)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert lo - 0.15 <= point <= hi + 0.15


def _results_frame():
    # two models on one target with known truth labels
    truth = {"s1": 0, "s2": 0, "s3": 1, "s4": 2}
    ma = {"s1": 0, "s2": 0, "s3": 1, "s4": 2}
    mb = {"s1": 0, "s2": 1, "s3": 2, "s4": 2}
    rows = []
    for model, scores in (("ma", ma), ("mb", mb)):
        for sid, o in scores.items():
            rows.append({"sample_id": sid, "model": model, "target": "area",
                         "o_score": o, "truth": truth[sid], "parse_ok": True})
    return pd.DataFrame(rows)


def _annotator_frame():
    # "2 – Klar relevant" mixes a graded text label into the numeric series
    return pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "annotator_1_area_relevance": [0, 0, 1, 2],
        "annotator_5_area_relevance": [0, 1, 1, 2],
        "annotator_6_area_relevance": [0, 0, 2, "2 – Klar relevant"],
        "annotator_1_criterion_relevance": [0, 0, 0, 0],
        "annotator_5_criterion_relevance": [0, 0, 0, 0],
        "annotator_6_criterion_relevance": [0, 0, 0, 0],
    })


def test_human_reference_all_pairs_with_cis():
    rows = ex.human_reference(_annotator_frame(), "area")
    assert [r["annotator_pair"] for r in rows] == ["1 vs 5", "1 vs 6", "5 vs 6"]
    for row in rows:
        assert row["n"] == 4
        for name in ("kappa_graded", "kappa_binary"):
            assert not np.isnan(row[name])
            assert not np.isnan(row[f"{name}_lo"]) and not np.isnan(row[f"{name}_hi"])
    # degenerate (all-zero) labels: NaN kappas + note, but the row is kept
    crit = ex.human_reference(_annotator_frame(), "criterion")
    assert len(crit) == 3
    for row in crit:
        assert np.isnan(row["kappa_graded"]) and np.isnan(row["kappa_binary"])
        assert "degenerate" in row["note"]


def test_inter_llm_reports_all_kappas_with_cis():
    rows = ex.inter_llm_metrics(_results_frame(), ["ma", "mb"], "area")
    assert len(rows) == 1
    row = rows[0]
    assert row["n_common"] == 4
    truth = [0, 0, 1, 2]
    ya, yb = [0, 0, 1, 2], [0, 1, 2, 2]
    assert row["kappa_graded"] == cohen_kappa_score(truth, ya, labels=[0, 1, 2])
    assert row["kappa_graded_model_b"] == cohen_kappa_score(truth, yb, labels=[0, 1, 2])
    assert row["kappa_binary"] == cohen_kappa_score([0, 0, 1, 1], [0, 1, 1, 1])
    for name in ("kappa_graded", "kappa_graded_model_b", "kappa_binary"):
        assert not np.isnan(row[f"{name}_lo"]) and not np.isnan(row[f"{name}_hi"])


def test_inter_llm_reports_nan_for_disjoint_pairs():
    results = _results_frame()
    extra = pd.DataFrame([{"sample_id": "s9", "model": "mc", "target": "area",
                           "o_score": 0, "truth": 0, "parse_ok": True}])
    rows = ex.inter_llm_metrics(pd.concat([results, extra], ignore_index=True), ["ma", "mc"], "area")
    assert len(rows) == 1
    row = rows[0]
    assert row["n_common"] == 0
    assert np.isnan(row["kappa_graded"]) and np.isnan(row["kappa_binary"])
    assert "no excerpts commonly rated" in row["note"]


def test_annotator_llm_metrics_reports_both_kappas():
    rows = ex.annotator_llm_metrics(_annotator_frame(), _results_frame(), ["ma", "mb"], "area")
    assert len(rows) == 6
    row = rows[0]
    assert row["model"] == "ma" and row["annotator"] == 1 and row["n"] == 4
    for name in ("kappa_graded", "kappa_binary"):
        assert not np.isnan(row[name])
        assert not np.isnan(row[f"{name}_lo"]) and not np.isnan(row[f"{name}_hi"])
    # model without rows in the checkpoint: NaN row with an explanatory note
    missing = ex.annotator_llm_metrics(_annotator_frame(), _results_frame(), ["mz"], "area")
    assert missing[0]["n"] == 0 and np.isnan(missing[0]["kappa_graded"])
    assert "no parsed scores" in missing[0]["note"]


def test_all_raters_fleiss_humans_and_llms():
    rows = ex.all_raters_metrics(_annotator_frame(), _results_frame(), ["ma", "mb"], "area")
    humans = next(r for r in rows if str(r["scope"]).startswith("humans"))
    assert humans["n_raters"] == 3 and humans["n_items"] == 4
    assert not np.isnan(humans["fleiss_kappa_graded"])
    assert not np.isnan(humans["fleiss_kappa_graded_lo"]) and not np.isnan(humans["fleiss_kappa_graded_hi"])
    # two models are not enough raters for Fleiss' kappa -> NaN + note
    llms = next(r for r in rows if r["scope"] == "LLMs only")
    assert llms["n_raters"] == 2
    assert np.isnan(llms["fleiss_kappa_graded"])
    assert "insufficient raters" in llms["note"]


def _run_all():
    failures = []
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)} tests, {len(failures)} failure(s)"
          if failures else f"\nAll {len(tests)} tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())