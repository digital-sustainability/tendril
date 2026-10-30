#!/usr/bin/env python
"""Unit tests for the training-query rewrite.

Runnable directly (no pytest required):

    .venv/bin/python scripts/queries/tests/test_build_training_queries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_training_queries as btq  # noqa: E402
import pandas as pd  # noqa: E402

AREA_XML = """\
<Handlungsfeld>
Sommerlicher Wärmeschutz
</Handlungsfeld>
<HandlungsfeldZusammenfassung>
Dieses Handlungsfeld adressiert die baulichen Anforderungen.
</HandlungsfeldZusammenfassung>
"""

WITH_AMBITION = """\
<Handlungsfeld>
Partizipation
</Handlungsfeld>
<Ausschreibungskriterium>
Zugang zu Information
Allgemein: Informationen zum Projekt sind zugänglich.
</Ausschreibungskriterium>
<Ambitionslevel>
Basis: Zugang nur auf Nachfrage:
1 Punkt

Vorbild: Zugang online:
2 Punkte
</Ambitionslevel>
"""

NO_AMBITION = """\
<Handlungsfeld>
Arbeitsbedingungen
</Handlungsfeld>
<Ausschreibungskriterium>
Arbeitsbedingungen - Form der Überprüfung
Das Kriterium hat zum Ziel, die Form der Überprüfung zu bewerten.
</Ausschreibungskriterium>
"""

SUMMARIES = {"Construction_Buildings::HF-015": "Zusammenfassung sommerlicher Wärmeschutz."}


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "project_id": "simap_1",
            "catalog": "WDG_8-Construction_Buildings",
            "query_id": "WDG_8-Construction_Buildings-HF-015",
            "level": "area_of_action",
            "tier": "relevant_validated",
            "query_text": AREA_XML,
        },
        {
            "project_id": "simap_1",
            "catalog": "WDG_8-Construction_Buildings",
            "query_id": "WDG-8-CONSTRUCTION-BUILDINGS-K-012",
            "level": "criterion",
            "tier": "relevant_validated",
            "query_text": WITH_AMBITION,
        },
        {
            "project_id": "simap_1",
            "catalog": "WDG_8-Construction_Buildings",
            "query_id": "WDG-8-CONSTRUCTION-BUILDINGS-K-057",
            "level": "criterion",
            "tier": "hard_negative",
            "query_text": NO_AMBITION,
        },
    ])


def test_area_becomes_summary():
    result = btq.build_training_queries(_frame(), SUMMARIES)
    assert result.loc[0, "query_text"] == "Zusammenfassung sommerlicher Wärmeschutz."


def test_criterion_with_ambition_appends_block_after_blank_line():
    result = btq.build_training_queries(_frame(), SUMMARIES)
    assert result.loc[1, "query_text"] == (
        "Zugang zu Information\n"
        "Allgemein: Informationen zum Projekt sind zugänglich.\n"
        "\n"
        "Ambitionsniveaus des Kriteriums:\n"
        "Basis: Zugang nur auf Nachfrage:\n1 Punkt\n"
        "Vorbild: Zugang online:\n2 Punkte"
    )


def test_criterion_without_ambition_is_text_only():
    result = btq.build_training_queries(_frame(), SUMMARIES)
    assert result.loc[2, "query_text"] == (
        "Arbeitsbedingungen - Form der Überprüfung\n"
        "Das Kriterium hat zum Ziel, die Form der Überprüfung zu bewerten."
    )
    assert "<" not in "".join(result["query_text"])


def test_ids_and_other_columns_are_kept():
    frame = _frame()
    result = btq.build_training_queries(frame, SUMMARIES)
    assert result["query_id"].tolist() == frame["query_id"].tolist()
    assert result["canonical_query_id"].tolist() == [
        "Construction_Buildings::HF-015",
        "Construction_Buildings::K-012",
        "Construction_Buildings::K-057",
    ]
    assert result.columns.tolist() == [*frame.columns, "canonical_query_id"]
    assert result["tier"].tolist() == frame["tier"].tolist()
    assert frame.loc[0, "query_text"] == AREA_XML  # input is not modified


def test_missing_summary_raises():
    try:
        btq.build_training_queries(_frame(), {})
    except ValueError as exc:
        assert "Construction_Buildings::HF-015" in str(exc)
    else:
        raise AssertionError("expected ValueError for a missing area summary")


def _run_all() -> int:
    failures: list[str] = []
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
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
    print(
        f"\n{len(tests)} tests, {len(failures)} failure(s)"
        if failures else f"\nAll {len(tests)} tests passed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
