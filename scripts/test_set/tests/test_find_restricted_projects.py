#!/usr/bin/env python
"""Tests for candidate detection and the judge loop (LLM mocked).

Runnable directly (no pytest required)::

    .venv/bin/python scripts/test_set/tests/test_find_restricted_projects.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import zstandard
from pydantic import ValidationError

SCRIPT = Path(__file__).resolve().parents[1] / "find_restricted_projects.py"
# Process-pool workers re-import the module by name, so its directory must be importable.
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("find_restricted_projects", SCRIPT)
frp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frp
spec.loader.exec_module(frp)

POSITIVES = {
    "Die Ausschreibungsunterlagen dürfen nur für die Erstellung des Angebots verwendet werden.": "clause:de_only_for_offer",
    "Eine Veröffentlichung, kommerzielle Verwertung und Weitergabe an Dritte in irgendeiner Form ist ohne Zustimmung der Vergabestelle nicht zulässig.": "clause:de_simap_clause",
    "Toute publication, exploitation commerciale ou transmission à des tiers du dossier d'appel d'offres est interdite.": "clause:fr_simap_clause",
    "Les documents de l’appel d’offres sont uniquement destinés à l’élaboration de l’offre.": "clause:fr_only_for_offer",
    "Il documento non può essere copiato né inoltrato a terzi.": "clause:it_no_third_party",
    "Dieses Dokument ist urheberrechtlich geschützt.": "keyword:de_copyright",
}
NEGATIVES = [
    "Klassifizierung: vertraulich",
    "© 2024 Stadt Bern",
    "Les informations sont confidentielles.",
    "Il documento è riservato.",
]


def test_find_candidate() -> None:
    for text, expected in POSITIVES.items():
        candidate = frp.find_candidate(text)
        assert candidate is not None and candidate.pattern == expected, (text, candidate)
    for text in NEGATIVES:
        assert frp.find_candidate(text) is None, text


def test_restriction_verdict_requires_boolean() -> None:
    verdict = frp.RestrictionVerdict.model_validate({"restricted": True, "reason": "x"})
    assert verdict.restricted is True and verdict.reason == "x"
    try:
        frp.RestrictionVerdict.model_validate({"restricted": "yes"})
    except ValidationError:
        pass
    else:
        raise AssertionError("non-boolean verdict must fail")


def write_project(directory: Path, name: str, passages: list[str]) -> None:
    project = {"documents": [{"content": [{"content_raw": text} for text in passages]}]}
    data = zstandard.ZstdCompressor().compress(json.dumps(project).encode("utf-8"))
    (directory / f"{name}.json.zst").write_bytes(data)


def test_end_to_end_with_mocked_llm() -> None:
    clause = "Die Unterlagen dürfen nur zur Erstellung des Angebots verwendet werden."
    keyword = "Eine Offerte nur für einzelne Lose ist zulässig, ausschliesslich für das Angebot Los 2."
    calls: list[str] = []

    class FakeClient:
        def generate_structured(self, messages, schema, params=None, **kwargs):
            snippet = messages[-1]["content"]
            calls.append(snippet)
            return schema(restricted="Erstellung des Angebots" in snippet, reason="test")

        def close(self) -> None:
            return None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "projects"
        data.mkdir()
        write_project(data, "simap_1", [keyword, clause])
        write_project(data, "simap_2", [keyword])
        write_project(data, "simap_3", ["Keine Treffer hier. vertraulich"])
        output, details = root / "out.txt", root / "details.jsonl"
        argv = [str(data), "--output", str(output), "--details", str(details), "--cache", str(root / "cache.json"), "--scan-workers", "1"]
        with patch.object(frp, "OpenRouterClient", return_value=FakeClient()):
            frp.main(argv)
            assert output.read_text().split() == ["simap_1"]
            verdicts = {row["project_id"]: row for row in map(json.loads, details.read_text().splitlines())}
            assert set(verdicts) == {"simap_1", "simap_2"}
            # Clause matches are judged first, so simap_1 needs a single call.
            assert verdicts["simap_1"]["judged"] == 1 and verdicts["simap_1"]["pattern"] == "clause:de_only_for_offer"
            assert len(calls) == 2
            frp.main(argv)  # second run is served from the cache
            assert len(calls) == 2


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")
