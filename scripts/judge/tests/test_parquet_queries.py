#!/usr/bin/env python
"""Unit tests for parquet query-id remapping and criterion reformatting.

Runnable directly (no pytest required):

    .venv/bin/python scripts/judge/tests/test_parquet_queries.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import run_labeling_parquet as rp  # noqa: E402

PARTIZIPATION = """\
<Handlungsfeld>
Partizipation
</Handlungsfeld>
<Ausschreibungskriterium>
Zugang zu Information
Allgemein: Informationen zum Projekt ermöglichen es, dass nicht direkt involvierte Personen Entscheidungen der Bauherrschaft nachvollziehen können.
</Ausschreibungskriterium>
<Ambitionslevel>
Basis: Das Kriterium ist teilweise erfüllt, wenn der Zugang zu Informationen erschwert ist (zum Beispiel nur auf individuelle Nachfrage):
1 Punkt

Vorbild: Das Kriterium ist erfüllt, wenn der Zugang zu Informationen sehr niederschwellig und umfassend ist (zum Beispiel online verfügbar, einfache Kontaktaufnahme):
2 Punkte
</Ambitionslevel>
"""

SCHALLSCHUTZ = """\
<Handlungsfeld>
Schallschutz
</Handlungsfeld>
<Ausschreibungskriterium>
Schallschutz innerhalb der Nutzungseinheit - Wohnen
Allgemein: Empfehlungen SIA 181.
</Ausschreibungskriterium>
<Ambitionslevel>
Basis: Nach Nutzungskategorie:
1 - 2 der nachfolgenden Bullet Points werden erreicht.
- Luftschall/Trittschall, Stufe 1:
0.25 Punkte
- Geräusche haustechnischer Anlagen, Stufe 1:
0.25 Punkte

Gute Praxis: Nach Nutzungskategorie:
3 der nachfolgenden Bullet Points werden erreicht.

Vorbild: Nach Nutzungskategorie:
4 der nachfolgenden Bullet Points werden erreicht.
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

AREA_XML = """\
<Handlungsfeld>
Sommerlicher Wärmeschutz
</Handlungsfeld>
<HandlungsfeldZusammenfassung>
Dieses Handlungsfeld adressiert die baulichen Anforderungen.
</HandlungsfeldZusammenfassung>
"""


def test_canonical_area_id():
    assert rp.canonical_query_id(
        "WDG_8-Construction_Buildings",
        "WDG_8-Construction_Buildings-HF-015",
    ) == "Construction_Buildings::HF-015"


def test_canonical_criterion_id_uses_catalog_column():
    assert rp.canonical_query_id(
        "WDG_8-Construction_Buildings",
        "WDG-8-CONSTRUCTION-BUILDINGS-K-057",
    ) == "Construction_Buildings::K-057"
    assert rp.canonical_query_id(
        "WDG_14-Paper_and_Stationery",
        "WDG-14-PAPER-AND-STATIONERY-K-014",
    ) == "Paper_and_Stationery::K-014"


def test_canonical_id_zero_pads():
    assert rp.canonical_query_id(
        "WDG_2-Food",
        "WDG_2-Food-HF-5",
    ) == "Food::HF-005"


def test_criterion_query_is_criterion_text_only():
    text = rp.criterion_query_text(PARTIZIPATION)
    block = rp.criterion_ambition_block(PARTIZIPATION)
    assert text == (
        "Zugang zu Information\n"
        "Allgemein: Informationen zum Projekt ermöglichen es, dass nicht direkt "
        "involvierte Personen Entscheidungen der Bauherrschaft nachvollziehen können."
    )
    assert "Handlungsfeld" not in text
    assert "Partizipation" not in text
    assert "Gute Praxis" not in block
    assert "Nachweise" not in block
    assert block == (
        "Ambitionsniveaus des Kriteriums:\n"
        "Basis: Das Kriterium ist teilweise erfüllt, wenn der Zugang "
        "zu Informationen erschwert ist (zum Beispiel nur auf individuelle Nachfrage):\n"
        "1 Punkt\n"
        "Vorbild: Das Kriterium ist erfüllt, wenn der Zugang zu "
        "Informationen sehr niederschwellig und umfassend ist (zum Beispiel online "
        "verfügbar, einfache Kontaktaufnahme):\n"
        "2 Punkte"
    )


def test_criterion_without_ambition_block():
    text = rp.criterion_query_text(NO_AMBITION)
    assert text == (
        "Arbeitsbedingungen - Form der Überprüfung\n"
        "Das Kriterium hat zum Ziel, die Form der Überprüfung zu bewerten."
    )
    assert rp.criterion_ambition_block(NO_AMBITION) == ""


def test_nested_colons_stay_inside_ambition_level():
    text = rp.criterion_query_text(SCHALLSCHUTZ)
    block = rp.criterion_ambition_block(SCHALLSCHUTZ)
    assert text == (
        "Schallschutz innerhalb der Nutzungseinheit - Wohnen\n"
        "Allgemein: Empfehlungen SIA 181."
    )
    assert "Luftschall/Trittschall, Stufe 1:" in block.split("Gute Praxis:")[0]
    assert block.count("\nBasis:") == 1
    assert "\nGute Praxis:" in block
    assert "\nVorbild:" in block
    assert "Stufe 1:" not in text


def test_rewrite_replaces_area_with_summary_and_formats_criterion():
    frame = pd.DataFrame([
        {
            "catalog": "WDG_8-Construction_Buildings",
            "query_id": "WDG_8-Construction_Buildings-HF-015",
            "level": "area_of_action",
            "query_text": AREA_XML,
        },
        {
            "catalog": "WDG_8-Construction_Buildings",
            "query_id": "WDG-8-CONSTRUCTION-BUILDINGS-K-057",
            "level": "criterion",
            "query_text": NO_AMBITION,
        },
    ])
    summaries = {"Construction_Buildings::HF-015": "Zusammenfassung sommerlicher Wärmeschutz."}
    rewritten = rp.rewrite_queries(frame, summaries, "query_text", "query_id")
    assert rewritten.loc[0, "query_id"] == "Construction_Buildings::HF-015"
    assert rewritten.loc[0, "query_text"] == "Zusammenfassung sommerlicher Wärmeschutz."
    assert "HandlungsfeldZusammenfassung" not in rewritten.loc[0, "query_text"]
    assert rewritten.loc[0, "ambition_levels"] == ""
    assert rewritten.loc[1, "query_id"] == "Construction_Buildings::K-057"
    assert rewritten.loc[1, "query_text"] == (
        "Arbeitsbedingungen - Form der Überprüfung\n"
        "Das Kriterium hat zum Ziel, die Form der Überprüfung zu bewerten."
    )
    assert rewritten.loc[1, "ambition_levels"] == ""


def test_exclude_projects_matches_ids_exactly():
    frame = pd.DataFrame({"project_id": ["simap_1", "simap_v2_1", "simap_2"]})
    kept = rp.exclude_projects(frame, "project_id", {"simap_1", "simap_v2_1"})
    assert kept["project_id"].tolist() == ["simap_2"]


def test_build_tasks_uses_level_and_leaves_ambition_placeholder_empty():
    frame = pd.DataFrame([{
        "project_id": "simap_1",
        "catalog": "WDG_2-Food",
        "query_id": "Food::HF-001",
        "level": "area_of_action",
        "query_text": "Zusammenfassung",
        "text": "Passage",
        "passage_uid": "abc",
        "criterion_id": None,
    }])
    cfg = {
        "hyperparams": {"max_tokens": 16},
        "targets": {
            "area": {"task_instruction": "Bewerte das Handlungsfeld."},
            "criterion": {"task_instruction": "Bewerte das Kriterium."},
        },
        "models": [{"name": "m", "client": "openrouter", "model": "x"}],
    }
    tasks = rp.build_tasks(
        frame,
        cfg,
        "Q={query}\nP={passage}\nA={ambition_levels}\n{task_instruction}",
        ["m"],
        "query_text",
        "text",
        "project_id",
        "query_id",
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task["sample_id"] == "simap_1|Food::HF-001|abc"
    assert task["target"] == "area"
    assert task["query_representation"] == "llm_summary"
    assert task["truth"] is None
    assert task["wdg_id"] == "WDG_2-Food"
    assert task["criterion_id"] is None
    assert "{ambition_levels}" not in task["prompt"]
    assert "Q=Zusammenfassung\nP=Passage\nA=\nBewerte das Handlungsfeld." == task["prompt"]


def test_build_tasks_puts_ambitions_and_passage_in_their_placeholders():
    frame = pd.DataFrame([{
        "project_id": "simap_1",
        "catalog": "WDG_8-Construction_Buildings",
        "query_id": "Construction_Buildings::K-057",
        "level": "criterion",
        "query_text": "Zugang zu Information",
        "ambition_levels": "Ambitionsniveaus des Kriteriums:\nBasis: nur auf Nachfrage",
        "text": "Die Website informiert die Nachbarschaft.",
        "passage_uid": "abc",
        "criterion_id": "WDG-8-CONSTRUCTION-BUILDINGS-K-057",
    }])
    cfg = {
        "hyperparams": {"max_tokens": 16},
        "targets": {
            "area": {"task_instruction": "Bewerte das Handlungsfeld."},
            "criterion": {"task_instruction": "Bewerte das Kriterium."},
        },
        "models": [{"name": "m", "client": "openrouter", "model": "x"}],
    }
    task = rp.build_tasks(
        frame,
        cfg,
        "Q={query}\nP={passage}\nA={ambition_levels}\n{task_instruction}",
        ["m"],
        "query_text",
        "text",
        "project_id",
        "query_id",
    )[0]
    assert task["target"] == "criterion"
    assert task["query_representation"] == "formatted"
    assert task["prompt"] == (
        "Q=Zugang zu Information\n"
        "P=Die Website informiert die Nachbarschaft.\n"
        "A=Ambitionsniveaus des Kriteriums:\nBasis: nur auf Nachfrage\n"
        "Bewerte das Kriterium."
    )


def test_relevance_judgments_keep_query_document_and_score():
    rows = [
        {
            "sample_id": "simap_1|Food::HF-001|abc",
            "query_id": "Food::HF-001",
            "o_score": 2,
            "parse_ok": True,
            "prompt": "full prompt that must not be copied",
        },
        {
            "sample_id": "simap_1|Food::K-014|def",
            "query_id": "Food::K-014",
            "o_score": 0,
            "parse_ok": False,
        },
        {
            "sample_id": "simap_2|Food::HF-001|abc",
            "query_id": "Food::HF-001",
            "o_score": 2,
            "parse_ok": True,
        },
        {
            "sample_id": "simap_9|Construction_Buildings::K-057|fff",
            "query_id": "Construction_Buildings::K-057",
            "o_score": 1,
            "parse_ok": True,
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        before = path.read_bytes()
        out = rp.write_relevance_judgments(path)
        assert path.read_bytes() == before
        assert out.name == "relevance_judgments.parquet"
        judgments = pd.read_parquet(out)
        assert list(judgments.columns) == ["query_id", "document_id", "score"]
        assert judgments.to_dict(orient="records") == [
            {"query_id": "Construction_Buildings::K-057", "document_id": "fff", "score": 1},
            {"query_id": "Food::HF-001", "document_id": "abc", "score": 2},
        ]


def test_relevance_judgments_reject_conflicting_scores():
    rows = [
        {"sample_id": "p|Food::HF-001|abc", "query_id": "Food::HF-001", "o_score": 0, "parse_ok": True},
        {"sample_id": "p|Food::HF-001|abc", "query_id": "Food::HF-001", "o_score": 2, "parse_ok": True},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        try:
            rp.write_relevance_judgments(path)
        except ValueError as exc:
            assert "Conflicting scores" in str(exc)
        else:
            raise AssertionError("expected ValueError for conflicting scores")
        assert not (Path(td) / "relevance_judgments.parquet").exists()


def test_missing_area_summary_is_an_error():
    frame = pd.DataFrame([
        {
            "catalog": "WDG_2-Food",
            "query_id": "WDG_2-Food-HF-001",
            "level": "area_of_action",
        },
    ])
    missing = rp.missing_area_summaries(frame, {}, "query_id")
    assert missing == ["Food::HF-001"]
    try:
        rp.rewrite_queries(
            frame.assign(query_text=AREA_XML),
            {},
            "query_text",
            "query_id",
        )
    except ValueError as exc:
        assert "Food::HF-001" in str(exc)
    else:
        raise AssertionError("expected ValueError for a missing area summary")


def _run_all():
    failures = []
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
