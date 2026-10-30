#!/usr/bin/env python
"""Tests for chunk borders and placeholder anonymization.

Runnable directly (no pytest required)::

    .venv/bin/python scripts/anonymize/tests/test_anonymize.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from gheim import Span

from tendril.anon import anonymize_texts, labels_for_tags, parse_entity_tags
from tendril.anon.types import AnonymizedText, Replacement
from tendril.anon.chunking import (
    MAX_CONTENT_TOKENS,
    choose_cuts,
    chunk_char_spans,
    relocate_cut,
)


class PhraseDetector:
    def __init__(self, phrases: list[tuple[str, str]]) -> None:
        self.phrases = phrases

    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        found: list[list[Span]] = []
        for text in texts:
            spans: list[Span] = []
            for label, phrase in self.phrases:
                start = 0
                while True:
                    index = text.find(phrase, start)
                    if index < 0:
                        break
                    spans.append(
                        Span(
                            label=label,
                            start=index,
                            end=index + len(phrase),
                            text=text[index : index + len(phrase)],
                        )
                    )
                    start = index + len(phrase)
            found.append(spans)
        return found


class FirstOnlyDetector:
    """Report only the first occurrence, leaving later mentions for recovery."""

    def __init__(self, label: str, phrase: str) -> None:
        self.label = label
        self.phrase = phrase

    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        found: list[list[Span]] = []
        for text in texts:
            index = text.find(self.phrase)
            if index < 0:
                found.append([])
                continue
            found.append(
                [
                    Span(
                        label=self.label,
                        start=index,
                        end=index + len(self.phrase),
                        text=text[index : index + len(self.phrase)],
                    )
                ]
            )
        return found


def _char_offsets(n_tokens: int) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(n_tokens)]


def test_clean_cut_stays_on_the_stride() -> None:
    offsets = _char_offsets(600)
    text = "." * 600
    cuts = choose_cuts(text, offsets, lambda windows: [[] for _ in windows])
    assert cuts == [0, 460, 600], cuts
    for start, end in zip(cuts, cuts[1:], strict=False):
        assert end - start <= MAX_CONTENT_TOKENS


def test_cut_moves_off_a_crossing_entity() -> None:
    offsets = _char_offsets(600)
    chars = ["."] * 600
    chars[450:470] = ["X"] * 20
    text = "".join(chars)

    def detect_batch(windows: list[str]) -> list[list[Span]]:
        found: list[list[Span]] = []
        for window in windows:
            start = window.find("X" * 20)
            if start < 0:
                found.append([])
                continue
            found.append(
                [
                    Span(
                        label="private_person",
                        start=start,
                        end=start + 20,
                        text=window[start : start + 20],
                    )
                ]
            )
        return found

    cuts = choose_cuts(text, offsets, detect_batch)
    assert cuts == [0, 450, 600], cuts
    for start, end in zip(cuts, cuts[1:], strict=False):
        assert end - start <= MAX_CONTENT_TOKENS
        assert not (start < 450 and end > 450 and end < 470)
    pieces = chunk_char_spans(text, offsets, cuts)
    assert "".join(text[start:end] for start, end in pieces) == text
    for cut in cuts[1:-1]:
        assert not 450 < cut < 470


def test_relocate_uses_the_other_side_when_the_near_side_overflows() -> None:
    offsets = _char_offsets(600)
    assert relocate_cut([0, 460, 600], 1, 20, 470, offsets) == 470


def test_relocate_leaves_a_cut_that_cannot_keep_a_huge_entity_whole() -> None:
    offsets = _char_offsets(600)
    assert relocate_cut([0, 460, 600], 1, 0, 600, offsets) == 460


def test_repeated_entities_share_a_placeholder_inside_one_text() -> None:
    text = "Joel schrieb an joel@example.ch und rief +41 44 411 46 37 an, Joel."
    detector = PhraseDetector(
        [
            ("private_person", "Joel"),
            ("private_email", "joel@example.ch"),
            ("private_phone", "+41 44 411 46 37"),
            ("private_address", "Bahnhofstrasse 1"),
            ("private_date", "01.03.2020"),
        ]
    )
    duplicated = (
        text
        + " Mail joel@example.ch, Telefon +41 44 411 46 37,"
        + " wohnt an Bahnhofstrasse 1, am 01.03.2020."
    )
    result = anonymize_texts([duplicated], detector=detector)[0]
    by_original: dict[str, set[str]] = {}
    for item in result.replacements:
        by_original.setdefault(item.original, set()).add(item.replacement)
    assert by_original["Joel"] == {"<PERSON_1>"}
    assert by_original["joel@example.ch"] == {"<EMAIL_1>"}
    assert by_original["+41 44 411 46 37"] == {"<PHONE_1>"}
    assert "Bahnhofstrasse 1" not in by_original
    assert "01.03.2020" not in by_original
    assert result.text.count("<PERSON_1>") == 2
    assert result.text.count("<EMAIL_1>") == 2
    assert result.text.count("<PHONE_1>") == 2
    assert "Joel" not in result.text
    assert "joel@example.ch" not in result.text
    assert "+41 44 411 46 37" not in result.text
    assert "Bahnhofstrasse 1" in result.text
    assert "01.03.2020" in result.text


def test_case_variant_and_missed_repeat_share_the_placeholder() -> None:
    text = "Joel sah joel."
    result = anonymize_texts(
        [text],
        detector=FirstOnlyDetector("private_person", "Joel"),
    )[0]
    assert len(result.replacements) == 2
    assert result.replacements[0].replacement == "<PERSON_1>"
    assert result.replacements[1].replacement == "<PERSON_1>"
    assert result.replacements[0].original == "Joel"
    assert result.replacements[1].original == "joel"
    assert result.text == "<PERSON_1> sah <PERSON_1>."


def test_same_name_in_another_text_restarts_numbering() -> None:
    detector = PhraseDetector([("private_person", "Joel")])
    results = anonymize_texts(["Joel kam.", "Joel ging."], detector=detector)
    assert results[0].replacements[0].replacement == "<PERSON_1>"
    assert results[1].replacements[0].replacement == "<PERSON_1>"
    assert results[0].text == "<PERSON_1> kam."
    assert results[1].text == "<PERSON_1> ging."


def test_dates_and_addresses_stay_unless_requested() -> None:
    text = "Termin 04.05.1998 an der Bahnhofstrasse 1."
    detector = PhraseDetector(
        [
            ("private_date", "04.05.1998"),
            ("private_address", "Bahnhofstrasse 1"),
        ]
    )
    kept = anonymize_texts([text], detector=detector)[0]
    assert kept.text == text
    assert kept.replacements == []
    redacted = anonymize_texts(
        [text],
        detector=detector,
        labels=labels_for_tags(("date", "address")),
    )[0]
    assert redacted.text == "Termin <DATE_1> an der <ADDRESS_1>."
    assert {item.replacement for item in redacted.replacements} == {"<DATE_1>", "<ADDRESS_1>"}


def test_repeated_date_shares_a_placeholder_when_requested() -> None:
    text = "Am 01.03.2020 und nochmals 01.03.2020, zuvor 01.01.2019."
    result = anonymize_texts(
        [text],
        detector=EmptyDetector(),
        labels=labels_for_tags(("date",)),
    )[0]
    assert result.text == "Am <DATE_1> und nochmals <DATE_1>, zuvor <DATE_2>."


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module: Any, argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = saved


def test_collection_script_writes_a_second_collection_and_keeps_the_log() -> None:
    script_path = Path(__file__).resolve().parents[1] / "anonymize_collections.py"
    module = _load_script("anonymize_collections", script_path)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "set" / "demo"
        output = root / "set_anon"
        (source / "qrels").mkdir(parents=True)
        (source / "corpus.jsonl").write_text(
            json.dumps({"id": "doc-1", "text": "Joel kam."}) + "\n",
            encoding="utf-8",
        )
        (source / "queries.jsonl").write_text(
            json.dumps({"id": "q-1", "text": "Frage"}) + "\n",
            encoding="utf-8",
        )
        (source / "qrels" / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t1\n",
            encoding="utf-8",
        )
        other = root / "set" / "other"
        (other / "qrels").mkdir(parents=True)
        (other / "corpus.jsonl").write_text(
            json.dumps({"id": "doc-2", "text": "Joel kam."}) + "\n",
            encoding="utf-8",
        )
        (other / "queries.jsonl").write_text(
            json.dumps({"id": "q-2", "text": "Frage"}) + "\n",
            encoding="utf-8",
        )
        (other / "qrels" / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq-2\tdoc-2\t1\n",
            encoding="utf-8",
        )
        (source.parent / "mteb_dataset.json").write_text(
            json.dumps(
                {
                    "format": "mteb-retrieval-v2",
                    "hub_repo": None,
                    "projects": ["demo", "other"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        replacement = Replacement(
            label="private_person",
            original="Joel",
            replacement="Anna Meier",
            start=0,
            end=4,
        )

        sentinel = object()
        build_calls = 0

        def fake_build(batch_size: int) -> object:
            nonlocal build_calls
            build_calls += 1
            assert batch_size == 4
            return sentinel

        def fake_anonymize(texts: list[str], **kwargs: object) -> list[AnonymizedText]:
            assert texts == ["Joel kam."]
            assert kwargs["detector"] is sentinel
            assert kwargs["labels"] == labels_for_tags(
                ("person", "email", "phone", "url", "account", "secret")
            )
            return [AnonymizedText(text="Anna Meier kam.", replacements=[replacement])]

        with (
            patch.object(module, "build_detector", fake_build),
            patch.object(module, "anonymize_texts", fake_anonymize),
        ):
            _run_main(
                module,
                [
                    "anonymize_collections.py",
                    str(source.parent),
                    "--output",
                    str(output),
                    "--batch-size",
                    "4",
                ],
            )
        rewritten = json.loads((output / "demo" / "corpus.jsonl").read_text(encoding="utf-8"))
        assert rewritten == {"id": "doc-1", "text": "Anna Meier kam."}
        assert (output / "demo" / "queries.jsonl").read_text(encoding="utf-8").startswith('{"id": "q-1"')
        assert (output / "demo" / "qrels" / "test.tsv").read_text(encoding="utf-8").count("doc-1") == 1
        log = json.loads((source / "anon_replacements.jsonl").read_text(encoding="utf-8"))
        assert log["id"] == "doc-1"
        assert log["replacements"][0]["replacement"] == "Anna Meier"
        run = json.loads((source.parent / "anon_run.json").read_text(encoding="utf-8"))
        assert run["model_id"] == "joelbarmettler/gheim-ch-560m"
        assert run["batch_size"] == 4
        assert run["entities"] == ["person", "email", "phone", "url", "account", "secret"]
        assert "seed" not in run
        assert build_calls == 1
        assert (source / "corpus.jsonl").read_text(encoding="utf-8").count("Joel kam.") == 1
        other_row = json.loads((output / "other" / "corpus.jsonl").read_text(encoding="utf-8"))
        assert other_row == {"id": "doc-2", "text": "Anna Meier kam."}
        manifest = json.loads((output / "mteb_dataset.json").read_text(encoding="utf-8"))
        assert manifest["format"] == "mteb-retrieval-v2"
        assert manifest["projects"] == ["demo", "other"]


def test_parquet_script_rewrites_the_column_and_writes_a_sidecar() -> None:
    script_path = Path(__file__).resolve().parents[1] / "anonymize_parquet.py"
    module = _load_script("anonymize_parquet", script_path)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "notes.parquet"
        frame = pd.DataFrame({"text": ["Joel kam.", None], "keep": [1, 2]})
        frame.to_parquet(source)
        replacement = Replacement(
            label="private_person",
            original="Joel",
            replacement="Anna Meier",
            start=0,
            end=4,
        )

        def fake_anonymize(texts: list[str], **kwargs: object) -> list[AnonymizedText]:
            assert texts == ["Joel kam."]
            assert "private_email" in kwargs["labels"]
            assert "private_date" not in kwargs["labels"]
            assert "private_address" not in kwargs["labels"]
            return [AnonymizedText(text="Anna Meier kam.", replacements=[replacement])]

        output = root / "notes_anon.parquet"
        with patch.object(module, "anonymize_texts", fake_anonymize):
            _run_main(
                module,
                [
                    "anonymize_parquet.py",
                    str(source),
                    "--column",
                    "text",
                    "--output",
                    str(output),
                ],
            )
        rewritten = pd.read_parquet(output)
        assert rewritten["text"].tolist()[0] == "Anna Meier kam."
        assert pd.isna(rewritten["text"].tolist()[1])
        assert rewritten["keep"].tolist() == [1, 2]
        sidecar = json.loads((root / "notes_anon.replacements.jsonl").read_text(encoding="utf-8"))
        assert sidecar["row"] == 0
        assert sidecar["replacements"][0]["original"] == "Joel"


def test_parquet_script_passes_a_custom_entity_list() -> None:
    script_path = Path(__file__).resolve().parents[1] / "anonymize_parquet.py"
    module = _load_script("anonymize_parquet_entities", script_path)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "notes.parquet"
        pd.DataFrame({"text": ["Termin 04.05.1998."]}).to_parquet(source)

        def fake_anonymize(texts: list[str], **kwargs: object) -> list[AnonymizedText]:
            assert texts == ["Termin 04.05.1998."]
            assert kwargs["labels"] == labels_for_tags(("person", "date"))
            return [AnonymizedText(text="Termin <DATE_1>.", replacements=[])]

        output = root / "notes_anon.parquet"
        with patch.object(module, "anonymize_texts", fake_anonymize):
            _run_main(
                module,
                [
                    "anonymize_parquet.py",
                    str(source),
                    "--column",
                    "text",
                    "--entities",
                    "person,date",
                    "--output",
                    str(output),
                ],
            )
        rewritten = pd.read_parquet(output)
        assert rewritten["text"].tolist() == ["Termin <DATE_1>."]


def test_unknown_entity_tag_is_rejected() -> None:
    try:
        parse_entity_tags("person,hometown")
    except ValueError as exc:
        assert "hometown" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert parse_entity_tags("person,email,person") == ("person", "email")


class EmptyDetector:
    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        return [[] for _ in texts]


def test_urls_become_a_numbered_placeholder() -> None:
    text = "Siehe https://www.stadt-zuerich.ch/gsz und nochmals https://www.stadt-zuerich.ch/gsz."
    result = anonymize_texts([text], detector=EmptyDetector())[0]
    assert result.replacements
    assert {item.replacement for item in result.replacements} == {"<URL_1>"}
    assert result.text.count("<URL_1>") == 2
    assert "stadt-zuerich" not in result.text


def test_regex_catalogue_finds_spans_the_model_missed() -> None:
    text = "Mail joel@example.ch oder +41 44 411 46 37."
    result = anonymize_texts([text], detector=EmptyDetector())[0]
    originals = {item.original: item.replacement for item in result.replacements}
    assert originals["joel@example.ch"] == "<EMAIL_1>"
    assert originals["+41 44 411 46 37"] == "<PHONE_1>"
    assert "joel@example.ch" not in result.text
    assert "+41 44 411 46 37" not in result.text


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
