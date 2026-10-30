#!/usr/bin/env python
"""Unit tests for judge_common score extraction and prompt building.

Covers the UMBRELA-DNA baseline keys (M/T/O), the German guidelines-variant
keys ({"Bewertung": x} -> O) and the Begruendung rationale of the
with-reasoning variant ({"Begruendung": ..., "Bewertung": x}). Runnable
directly (no pytest required):

    .venv/bin/python scripts/judge/tests/test_extract_scores.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_common as jc  # noqa: E402
import run_labeling as rl  # noqa: E402


def test_baseline_json_object():
    scores, method = jc.extract_scores('{"M": 1, "T": 1, "O": 2}')
    assert scores == {"M": 1, "T": 1, "O": 2}, scores
    assert method == "json_object"


def test_baseline_array_of_objects():
    scores, method = jc.extract_scores('reasoning ... [{"M": 0, "T": 1, "O": 1}]')
    assert scores == {"M": 0, "T": 1, "O": 1}, scores
    assert method == "json_object"


def test_baseline_positional_array():
    scores, method = jc.extract_scores('[1, 0, 2]')
    assert scores == {"M": 1, "T": 0, "O": 2}, scores
    assert method == "json_array"


def test_last_object_wins():
    text = '{"M": 0, "T": 0, "O": 0} more reasoning {"M": 1, "T": 1, "O": 2}'
    scores, _ = jc.extract_scores(text)
    assert scores["O"] == 2, scores


def test_quoted_o_regex():
    scores, method = jc.extract_scores('the model said "O": 2 at the end')
    assert scores == {"M": None, "T": None, "O": 2}, scores
    assert method == "regex"


def test_clamping():
    assert jc._clamp_score("5") == 2          # out of range -> clamped
    assert jc._clamp_score("-3") == 0
    assert jc._clamp_score("abc") is None
    assert jc._clamp_score("1.4") == 1


def test_german_json_object():
    scores, method = jc.extract_scores('{"Bewertung": 1}')
    assert scores == {"M": None, "T": None, "O": 1}, scores
    assert method == "json_object"


def test_german_lowercase_key():
    scores, _ = jc.extract_scores('{"bewertung": "0"}')
    assert scores["O"] == 0, scores


def test_german_short_key():
    scores, _ = jc.extract_scores('{"B": 2}')
    assert scores["O"] == 2, scores


def test_german_label_string():
    scores, _ = jc.extract_scores('{"Bewertung": "2 — Klar relevant"}')
    assert scores["O"] == 2, scores


def test_german_decimal_comma():
    scores, _ = jc.extract_scores('{"Bewertung": "1,5"}')
    assert scores["O"] == 2, scores  # rounds to nearest int, clamped


def test_german_regex_fallback():
    scores, method = jc.extract_scores('... lange Begründung ...\nBewertung: 2')
    assert scores["O"] == 2, scores
    assert method == "regex"


def test_german_quoted_key_regex_fallback():
    scores, _ = jc.extract_scores('Ergebnis: {"Bewertung": "1"}')
    assert scores["O"] == 1, scores


def test_echoed_template_is_not_parsed():
    # The model merely echoes the instruction template -> no score must parse.
    scores, method = jc.extract_scores(
        'Gib das Objekt {"Bewertung": <Ganzzahl 0, 1 oder 2>} zurück.')
    assert scores is None, scores
    assert method == "none"


def test_german_cot_last_object_wins():
    text = ('Schritt 1: lesbar. Schritt 2: gleiches Thema.\n'
            '{"Bewertung": 0}\nweitere Überlegung\n{"Bewertung": 2}')
    scores, _ = jc.extract_scores(text)
    assert scores["O"] == 2, scores


def test_explanation_key_is_ignored():
    scores, _ = jc.extract_scores('{"Bewertung": 1, "Erklärung": "passt"}')
    assert scores["O"] == 1, scores


def test_reasoning_json_object_scores():
    # guidelines with-reasoning variant: score next to the Begruendung text
    scores, method = jc.extract_scores(
        '{"Begruendung": "Die Textstelle nennt FSC-Holz direkt.", "Bewertung": 2}')
    assert scores == {"M": None, "T": None, "O": 2}, scores
    assert method == "json_object"


def test_reasoning_extracted():
    reasoning = jc.extract_reasoning(
        '{"Begruendung": "Die Textstelle nennt FSC-Holz direkt.", "Bewertung": 2}')
    assert reasoning == "Die Textstelle nennt FSC-Holz direkt.", reasoning


def test_reasoning_umlaut_and_typo_keys():
    assert jc.extract_reasoning('{"Begründung": "Passt.", "Bewertung": 1}') == "Passt."
    assert jc.extract_reasoning('{"BEGRUENDUNG": "Passt.", "Bewertung": 1}') == "Passt."
    assert jc.extract_reasoning('{"Begruending": "Passt auch so.", "Bewertung": 0}') == "Passt auch so."


def test_reasoning_last_object_wins():
    text = ('{"Begruendung": "Erster Gedanke.", "Bewertung": 0}\n'
            '{"Bewertung": 1, "Begruendung": "Finale Begruendung."}')
    assert jc.extract_reasoning(text) == "Finale Begruendung."
    scores, _ = jc.extract_scores(text)
    assert scores["O"] == 1, scores


def test_reasoning_regex_fallback_braces_in_text():
    # braces inside the rationale break the JSON-object scan; both the score
    # (Bewertung regex) and the rationale (quoted-key regex) must survive
    text = '{"Begruendung": "Erwaegung {in Klammern} und Schluss.", "Bewertung": 1}'
    scores, method = jc.extract_scores(text)
    assert scores["O"] == 1, scores
    assert method == "regex", method
    assert jc.extract_reasoning(text) == "Erwaegung {in Klammern} und Schluss."


def test_reasoning_whitespace_collapsed_and_unescaped():
    text = '{"Begruendung": "Zeile 1\\nZeile 2  mit   Luecken", "Bewertung": 0}'
    assert jc.extract_reasoning(text) == "Zeile 1 Zeile 2 mit Luecken"


def test_reasoning_absent():
    assert jc.extract_reasoning('{"Bewertung": 2}') is None
    assert jc.extract_reasoning('Bewertung: 2 ohne JSON') is None
    assert jc.extract_reasoning('{"Bewertung": 1, "Erklärung": "passt"}') is None
    assert jc.extract_reasoning(None) is None
    assert jc.extract_reasoning("   ") is None


def test_reasoning_echoed_template_not_parsed():
    # the model merely echoes the instruction template -> nothing must parse
    text = ('Gib das Objekt {"Begruendung": <Kurze Begruendung fuer die Entscheidung>,'
            '"Bewertung": <Ganzzahl 0, 1 oder 2>} zurück.')
    assert jc.extract_reasoning(text) is None
    scores, method = jc.extract_scores(text)
    assert scores is None, scores
    assert method == "none"


def test_empty_and_none():
    assert jc.extract_scores(None) == (None, "none")
    assert jc.extract_scores("   ") == (None, "none")


def test_build_prompt_placeholders():
    template = 'A {query} B {passage} C {"Bewertung": <0|1|2>} D {task_instruction}'
    out = jc.build_prompt(template, "Q", "P", {"task_instruction": "TI"})
    assert out.startswith("A Q B P C "), out
    assert out.endswith(" D TI"), out
    assert '<0|1|2>' in out  # literal JSON braces survive


def test_build_prompt_baseline_backwards_compatible():
    out = jc.build_prompt('Query: {query} Passage: {passage}', "Q", "P")
    assert out == 'Query: Q Passage: P', out


def test_format_ambition_levels_collapsed():
    block = jc.format_ambition_levels({
        "Basis": "Gleich für alle drei Niveaus",
        "Gut": "Gleich für alle drei Niveaus",
        "Beste": "Gleich für alle drei Niveaus"})
    assert block == "Ambitionsniveaus des Kriteriums:\n\nGleich für alle drei Niveaus", block


def test_format_ambition_levels_distinct():
    block = jc.format_ambition_levels({"Basis": "≥ 50%", "Gut": "≥ 50%", "Beste": "≥ 80%"})
    assert block.startswith("Ambitionsniveaus des Kriteriums:\n"), block
    assert "Basis: ≥ 50%" in block
    assert "Beste: ≥ 80%" in block
    assert "Gleich für alle drei Niveaus" not in block, block


def test_format_ambition_levels_partial_and_missing():
    partial = jc.format_ambition_levels({"Basis": None, "Gut": "60%", "Beste": "100%"})
    assert "Gut: 60%" in partial
    assert "Beste: 100%" in partial
    assert "Basis" not in partial, partial
    assert jc.format_ambition_levels({"Basis": float("nan"), "Gut": None, "Beste": ""}) == ""


def test_build_prompt_ambition_placeholder():
    out = jc.build_prompt("A {query} {ambition_levels} B", "Q", "P", {"ambition_levels": "AMB"})
    assert out == "A Q AMB B", out


def test_area_query_id_and_summary_resolution():
    record = {
        "wdg_id": "WDG_2-Food",
        "scope_of_action_id": "HF-2",
        "scope_of_action": "Kurzer Originaltext",
    }
    target_cfg = {"query_column": "scope_of_action"}
    assert jc.area_query_id(record["wdg_id"], record["scope_of_action_id"]) == \
        "Food::HF-002"
    query_id, original = jc.resolve_task_query(
        record, "area", target_cfg, "original", {}
    )
    assert query_id == "Food::HF-002"
    assert original == "Kurzer Originaltext"
    query_id, summary = jc.resolve_task_query(
        record,
        "area",
        target_cfg,
        "llm_summary",
        {"llm_summary": {"Food::HF-002": "Ausführliche Zusammenfassung"}},
    )
    assert query_id == "Food::HF-002"
    assert summary == "Ausführliche Zusammenfassung"


def test_missing_area_summary_fails():
    record = {
        "wdg_id": "WDG_2-Food",
        "scope_of_action_id": "HF-002",
        "scope_of_action": "Original",
    }
    try:
        jc.resolve_task_query(
            record,
            "area",
            {"query_column": "scope_of_action"},
            "llm_summary",
            {"llm_summary": {}},
        )
    except ValueError as exc:
        assert "Food::HF-002" in str(exc)
    else:
        raise AssertionError("missing summary did not raise")


def test_judge_one_builds_and_parses():
    class _FakeClient:
        def generate(self, messages, params, return_usage=False, **kwargs):
            assert return_usage is True
            assert messages == [{"role": "user", "content": "Q Bezug P Text"}]
            assert params["temperature"] == 0
            assert kwargs["timeout"] == 5
            return '{"Bewertung": 2}', {"total_tokens": 7}

    result = jc.judge_one(
        _FakeClient(),
        prompt_template="Q {query} P {passage}",
        query="Bezug",
        passage="Text",
        params={"temperature": 0},
        timeout=5,
    )
    assert result.parse_ok
    assert result.scores["O"] == 2
    assert result.usage["total_tokens"] == 7


def test_export_results_xlsx():
    import json
    import tempfile

    import pandas as pd

    rows = [
        {"run_id": "r1", "sample_id": "s1", "target": "criterion", "model": "m",
         "o_score": 2, "parse_ok": True, "raw_response": 'ok\x07text'},
        {"run_id": "r2", "sample_id": "s2", "target": "area", "model": "m",
         "o_score": 0, "parse_ok": True, "raw_response": "fine"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        out = jc.export_results_xlsx(path)
        assert out is not None and out.exists() and out.name == "results.xlsx", out
        df = pd.read_excel(out)
        assert len(df) == 2, len(df)
        assert int(df.iloc[0]["o_score"]) == 2
        assert str(df.iloc[1]["raw_response"]) == "fine"
        assert "\x07" not in str(df.iloc[0]["raw_response"])  # control char stripped
    # missing file -> no crash, returns None
    assert jc.export_results_xlsx(Path("/nonexistent/results.jsonl")) is None


def test_export_results_xlsx_truncates_overlong_prompt():
    import json
    import tempfile

    import pandas as pd

    long_prompt = "X" * 40000
    rows = [
        {"run_id": "r1", "sample_id": "s1", "target": "criterion", "model": "m",
         "prompt": long_prompt, "o_score": 1, "parse_ok": True},
        {"run_id": "r2", "sample_id": "s2", "target": "area", "model": "m",
         "prompt": "kurzer prompt", "o_score": 0, "parse_ok": True},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "results.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        out = jc.export_results_xlsx(path)
        assert out is not None and out.exists(), out
        df = pd.read_excel(out)
        cell = str(df.iloc[0]["prompt"])
        assert len(cell) <= 32767, len(cell)  # Excel cell limit respected
        assert "truncated for Excel" in cell
        assert "truncated for Excel" not in str(df.iloc[1]["prompt"])  # short stays verbatim
        # the JSONL itself is never modified
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert len(json.loads(first_line)["prompt"]) == 40000


def test_run_model_tasks_persists_full_prompt():
    import tempfile

    import pandas as pd

    prompt = "PROMPT-START Bezugstext … Textstelle … PROMPT-ENDE"

    class _FakeClient:
        def generate(self, messages, params, return_usage=False, timeout=None):
            assert return_usage
            assert messages == [{"role": "user", "content": prompt}]
            return '{"Bewertung": 2}', {"prompt_tokens": 10, "completion_tokens": 5,
                                        "total_tokens": 15, "finish_reason": "stop"}

    orig_client, orig_hp = jc.make_client, rl._HYPERPARAMS
    jc.make_client = lambda *a, **k: _FakeClient()
    rl._HYPERPARAMS = {"temperature": 0.0, "top_p": 1.0,
                       "frequency_penalty": 0.0, "presence_penalty": 0.0}
    try:
        task = {"sample_id": "s1", "wdg_id": "w1", "criterion_id": "c1",
                "target": "criterion", "model": "fake", "client": "fake",
                "model_id": "fake-model", "query_text": "Bezugstext",
                "passage": "Textstelle", "truth": 2, "prompt": prompt,
                "max_tokens": 100}
        runtime = {"max_retries": 0, "timeout": 5, "rate_limit_rpm": 1000,
                   "max_workers": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.jsonl"
            stats = rl.run_model_tasks({"name": "fake", "client": "fake",
                                        "model": "fake-model"}, [task], path,
                                       "r1", "guidelines_de_v1", "cafe", runtime)
            assert stats["succeeded"] == 1, stats
            df = pd.read_json(path, lines=True)
            assert str(df.iloc[0]["prompt"]) == prompt  # full, untruncated
            assert str(df.iloc[0]["prompt_sha256"]) == "cafe"
            assert int(df.iloc[0]["o_score"]) == 2
    finally:
        jc.make_client = orig_client
        rl._HYPERPARAMS = orig_hp


def test_run_model_tasks_persists_reasoning():
    """The Begruendung from the with-reasoning variant lands in the `reasoning`
    column of the JSONL checkpoint."""
    import json
    import tempfile

    import pandas as pd

    prompt = "PROMPT-START Bezugstext … Textstelle … PROMPT-ENDE"
    response = '{"Begruendung": "Textstelle nennt FSC-Holz.", "Bewertung": 2}'

    class _FakeClient:
        def generate(self, messages, params, return_usage=False, timeout=None):
            assert return_usage
            return response, {"prompt_tokens": 10, "completion_tokens": 8,
                              "total_tokens": 18, "finish_reason": "stop"}

    orig_client, orig_hp = jc.make_client, rl._HYPERPARAMS
    jc.make_client = lambda *a, **k: _FakeClient()
    rl._HYPERPARAMS = {"temperature": 0.0, "top_p": 1.0,
                       "frequency_penalty": 0.0, "presence_penalty": 0.0}
    try:
        task = {"sample_id": "s1", "wdg_id": "w1", "criterion_id": "c1",
                "target": "criterion", "model": "fake", "client": "fake",
                "model_id": "fake-model", "query_text": "Bezugstext",
                "passage": "Textstelle", "truth": 2, "prompt": prompt,
                "max_tokens": 100}
        runtime = {"max_retries": 0, "timeout": 5, "rate_limit_rpm": 1000,
                   "max_workers": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.jsonl"
            stats = rl.run_model_tasks({"name": "fake", "client": "fake",
                                        "model": "fake-model"}, [task], path,
                                       "r1", "guidelines_de_v1_with_reasoning",
                                       "cafe", runtime)
            assert stats["succeeded"] == 1, stats
            df = pd.read_json(path, lines=True)
            assert int(df.iloc[0]["o_score"]) == 2
            assert str(df.iloc[0]["reasoning"]) == "Textstelle nennt FSC-Holz.", \
                df.iloc[0]["reasoning"]
            # full reasoning text in the JSONL (no truncation)
            assert "FSC-Holz" in json.loads(path.read_text(
                encoding="utf-8").splitlines()[0])["reasoning"]
    finally:
        jc.make_client = orig_client
        rl._HYPERPARAMS = orig_hp


def test_run_model_tasks_persists_cost():
    """The provider-reported cost from the usage object (only the openrouter
    client returns one) lands in the `cost` column of the JSONL checkpoint and
    in the stats total; clients without usage accounting persist null."""
    import tempfile

    import pandas as pd

    prompt = "PROMPT-START Bezugstext … Textstelle … PROMPT-ENDE"
    cost = 0.0000123

    class _FakeClient:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, params, return_usage=False, timeout=None):
            assert return_usage
            self.calls += 1
            usage = {"prompt_tokens": 10, "completion_tokens": 5,
                     "total_tokens": 15, "finish_reason": "stop"}
            if self.calls == 1:  # like OpenRouter with usage accounting on
                usage["cost"] = cost
            return '{"Bewertung": 2}', usage

    orig_client, orig_hp = jc.make_client, rl._HYPERPARAMS
    jc.make_client = lambda *a, **k: _FakeClient()
    rl._HYPERPARAMS = {"temperature": 0.0, "top_p": 1.0,
                       "frequency_penalty": 0.0, "presence_penalty": 0.0}
    try:
        def mk_task(sample_id):
            return {"sample_id": sample_id, "wdg_id": "w1", "criterion_id": "c1",
                    "target": "criterion", "model": "fake", "client": "fake",
                    "model_id": "fake-model", "query_text": "Bezugstext",
                    "passage": "Textstelle", "truth": 2, "prompt": prompt,
                    "max_tokens": 100}

        runtime = {"max_retries": 0, "timeout": 5, "rate_limit_rpm": 1000,
                   "max_workers": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.jsonl"
            stats = rl.run_model_tasks({"name": "fake", "client": "fake",
                                        "model": "fake-model"},
                                       [mk_task("s1"), mk_task("s2")], path,
                                       "r1", "guidelines_de_v1", "cafe", runtime)
            assert stats["succeeded"] == 2, stats
            assert stats["cost"] == round(cost, 10), stats
            df = pd.read_json(path, lines=True).set_index("sample_id")
            assert float(df.loc["s1", "cost"]) == cost  # provider cost persisted
            assert pd.isna(df.loc["s2", "cost"])        # no cost -> null/NaN
    finally:
        jc.make_client = orig_client
        rl._HYPERPARAMS = orig_hp


def test_resolve_batch_chunk_size():
    assert rl.resolve_batch_chunk_size({}) == 256
    assert rl.resolve_batch_chunk_size({"throttle": {"max_concurrency": 64}}) == 64
    assert rl.resolve_batch_chunk_size(
        {"batch_chunk_size": 32, "throttle": {"max_concurrency": 256}}
    ) == 32
    try:
        rl.resolve_batch_chunk_size({"batch_chunk_size": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for batch_chunk_size=0")


def test_run_model_tasks_checkpoints_each_rr_batch():
    """Each generate_batch chunk is appended before the next chunk is submitted."""
    import tempfile

    prompt = "PROMPT-START Bezugstext … Textstelle … PROMPT-ENDE"

    class _FakeClient:
        def __init__(self, path):
            self.path = path
            self.sizes = []
            self.lines_before = []

        def generate_batch(self, requests, **kwargs):
            text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            self.lines_before.append(len([line for line in text.splitlines() if line]))
            self.sizes.append(len(requests))
            usage = {"prompt_tokens": 3, "completion_tokens": 2,
                     "total_tokens": 5, "finish_reason": "stop"}
            return [('{"Bewertung": 1}', usage)] * len(requests)

    def mk_task(sample_id):
        return {"sample_id": sample_id, "wdg_id": "w1", "criterion_id": "c1",
                "target": "criterion", "model": "fake", "client": "fake",
                "model_id": "fake-model", "query_text": "Bezugstext",
                "passage": "Textstelle", "truth": 1, "prompt": prompt,
                "max_tokens": 100}

    orig_client, orig_hp = jc.make_client, rl._HYPERPARAMS
    rl._HYPERPARAMS = {"temperature": 0.0, "top_p": 1.0,
                       "frequency_penalty": 0.0, "presence_penalty": 0.0}
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.jsonl"
            client = _FakeClient(path)
            jc.make_client = lambda *a, **k: client
            runtime = {"max_retries": 0, "timeout": 5, "retry_base_sleep": 0,
                       "max_workers": 1, "batch_chunk_size": 2}
            stats = rl.run_model_tasks(
                {"name": "fake", "client": "fake", "model": "fake-model"},
                [mk_task(f"s{i}") for i in range(5)],
                path, "r1", "guidelines_de_v1", "cafe", runtime,
            )
            assert client.sizes == [2, 2, 1], client.sizes
            assert client.lines_before == [0, 2, 4], client.lines_before
            assert stats["succeeded"] == 5, stats
            assert len(path.read_text(encoding="utf-8").splitlines()) == 5
    finally:
        jc.make_client = orig_client
        rl._HYPERPARAMS = orig_hp


def test_resolve_model_concurrency():
    assert rl.resolve_model_concurrency({}, 16) == 16
    assert rl.resolve_model_concurrency({"max_concurrent_models": 2}, 16) == 2
    assert rl.resolve_model_concurrency({"max_concurrent_models": 32}, 4) == 4
    assert rl.resolve_model_concurrency({"max_concurrent_models": 1}, 0) == 1
    try:
        rl.resolve_model_concurrency({"max_concurrent_models": 0}, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_concurrent_models=0")


def test_resolve_results_dir():
    # explicit --results-dir wins, relative to the repo root
    assert rl.resolve_results_dir("results_x", "configs/model_reasoning_simple_aoa.yaml") == jc.REPO_ROOT / "results_x"
    # default: data/judge/<config file stem>_<exact UTC ISO timestamp>
    d = rl.resolve_results_dir(None, "configs/model_reasoning_simple_aoa.yaml")
    assert d.parent == jc.REPO_ROOT / "data" / "judge", d
    stem = Path("configs/model_reasoning_simple_aoa.yaml").stem
    assert d.name.startswith(stem + "_"), d.name
    stamp = d.name[len(stem) + 1:]
    assert datetime.fromisoformat(stamp).tzinfo is not None  # parses as ISO with tz
    # derived from the config file name, not paths.results_dir
    other = rl.resolve_results_dir(None, "configs/model_reasoning_summary_aoa.yaml")
    assert other.name.startswith("model_reasoning_summary_aoa_"), other.name
    assert other != d


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