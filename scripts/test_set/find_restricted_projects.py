"""Find projects whose tender documents restrict reuse beyond submitting an offer.

Tender documents sometimes carry a clause such as "Diese Unterlagen dürfen nur zur
Erstellung des Angebots verwendet werden". Projects with such a clause must be
excluded from the published data. This script reads the parsed project files
(``*.json.zst``), flags candidate passages with regular expressions, and asks an
LLM judge whether each candidate passage really contains such a restriction.

Candidates are passages that match a clause pattern (``CLAUSE_PATTERNS``) or a
keyword pattern (``KEYWORD_PATTERNS``). Passages of a project are judged in order,
clause matches first, and judging stops at the first positive verdict.

Example::

    python scripts/test_set/find_restricted_projects.py \\
        path/to/parsed/projects.json.zst \\
        --output data/filtering/excluded_projects.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import zstandard
from loguru import logger
from openai import OpenAIError
from pydantic import BaseModel, field_validator
from responsible_request import StructuredOutputError
from tqdm import tqdm

from tendril.inference import OpenRouterClient
from tendril.paths import REPOSITORY_PATH

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
ROOT = REPOSITORY_PATH
DEFAULT_CACHE = ROOT / "tmp" / "restriction_judge_cache.json"
SNIPPET_RADIUS = 700
WHITESPACE = re.compile(r"\s+")

DOCUMENT = r"(unterlagen|dokument\w*|pläne|angebotsanfrage|documents?|dossiers?|documentazione|documenti|atti)"

# Patterns that describe the restriction itself (high precision).
CLAUSE_PATTERNS = {
    "de_only_for_offer": (
        r"(nur|ausschlie(ss|ß)lich|einzig)\s+(für|zur|zum)\s+(die\s+|eine\s+)?"
        r"(erstellung|erarbeitung|ausarbeitung|einreichung|bearbeitung|abgabe|kalkulation)\s+.{0,30}(angebot|offert)"
    ),
    "de_simap_clause": r"kommerzielle\s+verwertung\s+(und|oder)\s+weitergabe\s+an\s+dritte",
    "de_no_third_party": DOCUMENT
    + r"\b.{0,150}(nicht|weder).{0,60}an\s+(unbeteiligte\s+)?dritte\s+weiter(zu)?(ge)?geben",
    "fr_only_for_offer": (
        r"(uniquement|exclusivement|seulement)\s+.{0,40}"
        r"(élaboration|établissement|préparation|remise|soumission|dépôt|rédaction)\s+.{0,30}offre"
    ),
    "fr_simap_clause": r"exploitation\s+commerciale.{0,40}transmission\s+à\s+des\s+tiers",
    "fr_no_third_party": r"(ne\s+peu(ven)?t|ne\s+doi(ven)?t|interdite?).{0,80}(transmis|remis|communiqués?|transmission)\s+à\s+des\s+tiers",
    "it_only_for_offer": (
        r"(unicamente|esclusivamente|solo|soltanto)\s+.{0,40}"
        r"(elaborazione|preparazione|allestimento|presentazione|redazione|inoltro)\s+.{0,30}offert"
    ),
    "it_simap_clause": r"(utilizzazione|sfruttamento|utilizzo|uso)\s+commerciale.{0,40}(trasmissione|divulgazione|inoltro)\s+a\s+terzi",
    "it_no_third_party": r"(non\s+(possono|devono|può|deve)|vietat).{0,80}(trasmess|consegnat|comunicat|trasmission|inoltrat)\w*\s+a\s+terzi",
    "en_only_for_offer": r"(solely|only|exclusively|sole\s+purpose\s+of)\s+.{0,40}(prepar|submi)\w*\s+.{0,30}(tender|bid|offer|proposal)",
    "en_no_third_party": r"(not|neither).{0,60}(passed\s+on|disclosed|forwarded|made\s+available|transferred)\s+to\s+third\s+parties",
}

# Broader keyword patterns (high recall, many false positives).
KEYWORD_PATTERNS = {
    "de_only_for": r"(nur|ausschlie(ss|ß)lich)\s+(für|zum|zur)\s+.{0,60}(angebot|offert)",
    "de_third_party": r"weitergabe\s+an\s+dritte",
    "de_further_use": r"(weitere|anderweitige)\s+verwendung",
    "de_copyright": r"urheberrechtlich\s+geschützt|alle\s+rechte\s+vorbehalten",
    "fr_only_for": r"(uniquement|exclusivement)\s+.{0,60}(offre|soumission)",
    "fr_third_party": r"transmission\s+à\s+des\s+tiers|tous\s+droits\s+réservés",
    "it_only_for": r"(unicamente|esclusivamente)\s+.{0,60}offert",
    "it_third_party": r"trasmissione\s+a\s+terzi|tutti\s+i\s+diritti\s+riservati",
}

PATTERNS = {
    **{f"clause:{name}": re.compile(pattern, re.IGNORECASE) for name, pattern in CLAUSE_PATTERNS.items()},
    **{f"keyword:{name}": re.compile(pattern, re.IGNORECASE) for name, pattern in KEYWORD_PATTERNS.items()},
}

SYSTEM_PROMPT = """\
You review excerpts from Swiss public procurement documents (German, French, Italian or English).

Decide whether the excerpt restricts how the procurement documents themselves may be used \
or passed on. This is the case when the excerpt states, for example, that the documents may \
only be used to prepare or submit an offer, that any other or further use is not allowed or \
needs consent, or that the documents must not be published, exploited commercially or passed \
on to third parties.

The answer is "false" when the excerpt only:
- classifies a document as confidential or internal (e.g. a "vertraulich" header),
- contains a copyright notice or "all rights reserved" without a usage restriction,
- requires that the bidder's offer, data or answers are treated confidentially by the authority,
- describes confidentiality obligations of the contractor during contract execution,
- uses similar words in an unrelated sense (e.g. "an offer only for one lot").

Respond with JSON only: {"restricted": true|false, "reason": "<one short sentence>"}\
"""


class RestrictionVerdict(BaseModel):
    restricted: bool
    reason: str = ""

    @field_validator("restricted", mode="before")
    @classmethod
    def require_bool(cls, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError("restricted must be a boolean")
        return value


USER_PROMPT = """\
Excerpt:
'''
{snippet}
'''

Does this excerpt restrict the use or disclosure of the procurement documents?\
"""


@dataclass(frozen=True)
class Candidate:
    pattern: str
    snippet: str

    @property
    def is_clause(self) -> bool:
        return self.pattern.startswith("clause:")


@dataclass
class Verdict:
    project_id: str
    restricted: bool
    candidates: int
    judged: int
    pattern: str | None = None
    snippet: str | None = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)


def compact(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


def snippet_around(text: str, start: int, end: int, radius: int = SNIPPET_RADIUS) -> str:
    return compact(text[max(0, start - radius) : end + radius])


def find_candidate(text: str) -> Candidate | None:
    """Return the first matching pattern of a passage, preferring clause patterns."""
    for name, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match:
            return Candidate(name, snippet_around(text, match.start(), match.end()))
    return None


def project_id(path: Path) -> str:
    return path.name.removesuffix(".json.zst")


def read_project(path: Path) -> dict:
    with path.open("rb") as handle:
        return json.loads(zstandard.ZstdDecompressor().stream_reader(handle).read())


def scan_project(path: Path) -> tuple[str, list[Candidate]]:
    project = read_project(path)
    candidates: dict[str, Candidate] = {}
    for document in project.get("documents") or []:
        for passage in document.get("content") or []:
            candidate = find_candidate(passage.get("content_raw") or "")
            if candidate is not None:
                candidates.setdefault(candidate.snippet, candidate)
    ordered = sorted(candidates.values(), key=lambda candidate: not candidate.is_clause)
    return project_id(path), ordered


def scan_projects(paths: Sequence[Path], workers: int) -> dict[str, list[Candidate]]:
    results: dict[str, list[Candidate]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_project, path) for path in paths]
        for future in tqdm(as_completed(futures), total=len(paths), desc="Scanning"):
            pid, candidates = future.result()
            if candidates:
                results[pid] = candidates
    return dict(sorted(results.items()))


def cache_key(model: str, snippet: str) -> str:
    digest = hashlib.sha256(f"{SYSTEM_PROMPT}\n{snippet}".encode("utf-8")).hexdigest()
    return f"{model}|{digest}"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: cache is not a JSON object: {path}")
    return payload


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def judge_projects(
    candidates: dict[str, list[Candidate]],
    client: OpenRouterClient,
    model: str,
    cache: dict[str, dict],
    cache_path: Path,
    workers: int,
) -> list[Verdict]:
    lock = threading.Lock()
    counters = {"cached": 0, "new": 0}

    def judge(candidate: Candidate) -> RestrictionVerdict:
        key = cache_key(model, candidate.snippet)
        with lock:
            cached = cache.get(key)
        if cached is not None:
            with lock:
                counters["cached"] += 1
            return RestrictionVerdict.model_validate(cached)
        judgement = client.generate_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.replace("{snippet}", candidate.snippet)},
            ],
            RestrictionVerdict,
            {"temperature": 0, "max_tokens": 300},
            reasoning={"enabled": False},
        )
        with lock:
            cache[key] = judgement.model_dump()
            counters["new"] += 1
        return judgement

    def one(pid: str) -> Verdict:
        verdict = Verdict(pid, restricted=False, candidates=len(candidates[pid]), judged=0)
        for candidate in candidates[pid]:
            try:
                judgement = judge(candidate)
            except (OpenAIError, StructuredOutputError, RuntimeError) as exc:
                verdict.errors.append(str(exc))
                continue
            verdict.judged += 1
            if judgement.restricted:
                verdict.restricted = True
                verdict.pattern = candidate.pattern
                verdict.snippet = candidate.snippet
                verdict.reason = judgement.reason.strip()
                break
        return verdict

    verdicts: list[Verdict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, pid) for pid in candidates]
        try:
            for done, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Judging"), 1):
                verdicts.append(future.result())
                if done % 20 == 0:
                    with lock:
                        save_cache(cache_path, cache)
        finally:
            with lock:
                save_cache(cache_path, cache)
    logger.info(f"LLM judgements: {counters['cached']} cached, {counters['new']} new")
    return sorted(verdicts, key=lambda verdict: verdict.project_id)


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{pid}\n" for pid in ids), encoding="utf-8")
    logger.info(f"Wrote {len(ids)} project ids to {path}")


def write_details(path: Path, verdicts: Sequence[Verdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(verdict.__dict__, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(verdicts)} judged projects to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List projects whose documents restrict reuse beyond submitting an offer"
    )
    parser.add_argument("input_dir", type=Path, help="Directory with enriched *.json.zst project files")
    parser.add_argument("--output", type=Path, required=True, help="Text file for project ids to exclude")
    parser.add_argument(
        "--details", type=Path, help="Optional JSONL with the verdict and evidence per candidate project"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent OpenRouter requests (default: 8)")
    parser.add_argument(
        "--scan-workers", type=int, default=16, help="Processes for reading project files (default: 16)"
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE, help=f"LLM response cache JSON (default: {DEFAULT_CACHE})"
    )
    parser.add_argument("--limit", type=int, help="Judge at most this many candidate projects (for testing)")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Write all regex candidates without asking the LLM judge",
    )
    parser.add_argument(
        "--exclude-on-error",
        action="store_true",
        help="Exclude projects whose candidates could not all be judged (default: keep them and log a warning)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = sorted(args.input_dir.glob("*.json.zst"))
    if not paths:
        raise SystemExit(f"ERROR: no *.json.zst files in {args.input_dir}")
    candidates = scan_projects(paths, max(1, args.scan_workers))
    n_passages = sum(len(items) for items in candidates.values())
    logger.info(f"{len(candidates)} of {len(paths)} projects have {n_passages} candidate passages")
    if args.limit is not None:
        candidates = dict(list(candidates.items())[: args.limit])

    if args.skip_llm:
        verdicts = [
            Verdict(pid, True, len(items), 0, items[0].pattern, items[0].snippet) for pid, items in candidates.items()
        ]
    else:
        workers = max(1, args.workers)
        client = OpenRouterClient(
            args.model,
            title="TENDRIL Restriction Filter",
            timeout=180.0,
            max_retries=6,
            throttle={"max_concurrency": workers},
        )
        try:
            cache = load_cache(args.cache)
            verdicts = judge_projects(candidates, client, args.model, cache, args.cache, workers)
        finally:
            client.close()

    failed = [verdict for verdict in verdicts if verdict.errors and not verdict.restricted]
    if failed:
        action = "excluding" if args.exclude_on_error else "keeping"
        logger.warning(
            f"{len(failed)} projects had judge errors and no positive verdict; {action} them. Rerun to retry: "
            + ", ".join(verdict.project_id for verdict in failed)
        )
    failed_ids = {verdict.project_id for verdict in failed} if args.exclude_on_error else set()
    excluded = [verdict.project_id for verdict in verdicts if verdict.restricted or verdict.project_id in failed_ids]
    logger.info(f"Excluding {len(excluded)} of {len(paths)} projects")
    write_ids(args.output, excluded)
    if args.details is not None:
        write_details(args.details, verdicts)


if __name__ == "__main__":
    main()
