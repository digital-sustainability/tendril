"""Shared utilities for the LLM-as-a-judge experiments.

Provides config loading, dataset access, prompt construction, robust score
extraction from reasoning-style model outputs, client construction and
checkpoint-file helpers for the labeling runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tendril.inference import OpenRouterClient
from tendril.inference import responsible_request_kwargs  # noqa: F401  (used as jc.responsible_request_kwargs)
from tendril.paths import REPOSITORY_PATH
from tendril.queries import format_ambition_levels  # noqa: F401  (used as jc.format_ambition_levels)
from tendril.queries import load_query_map as _load_query_map

REPO_ROOT = REPOSITORY_PATH
JUDGE_DIR = Path(__file__).resolve().parent

_JSONL_WRITE_LOCK = threading.Lock()

RESULT_COLUMNS = [
    "run_id", "created_at", "sample_id", "wdg_id", "criterion_id", "target",
    "model", "client", "model_id", "prompt_id", "prompt_sha256", "prompt",
    "query_id", "query_representation", "query_text", "passage", "truth",
    "m_score", "t_score", "o_score", "reasoning", "thinking", "parse_method", "parse_ok",
    "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens",
    # Host-level accounting from the OpenRouter usage object (openrouter
    # client only, null for all other clients): the upstream provider that
    # served the request, its native-tokenizer counts (vs the
    # OpenRouter-normalized prompt/completion/total above), cached prompt
    # tokens and reasoning tokens reported inside the token-detail objects,
    # plus the upstream provider's own (pre-markup) inference cost from
    # usage.cost_details — cross-check against `cost`.
    "provider", "native_prompt_tokens", "native_completion_tokens",
    "native_reasoning_tokens", "cached_prompt_tokens", "reasoning_tokens",
    "upstream_inference_cost",
    # `cost`: provider-reported monetary cost (USD) from the usage object;
    # only the openrouter client returns one (null for all other clients).
    "cost",
    "latency_s", "attempts", "error", "raw_response",
]

_SCORE_KEY_ALIASES = {
    "m": "M", "t": "T", "o": "O",
    # German guidelines prompt (guidelines_de): the single final rating is
    # returned under a German key and mapped onto O, keeping the row schema
    # (m_score/t_score = None) and the whole analysis stack unchanged.
    "b": "O", "bewertung": "O", "score": "O",
    "rating": "O", "relevance": "O", "final_score": "O",
}

#: SDK/HTTP loggers that emit one INFO line per request — noisy on long runs,
#: only surfaced when DEBUG logging is requested (see setup_logging).
#: openai>=3 uses the httpx2/httpcore2 forks; older stacks still use httpx.
_NOISY_SDK_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "openai")

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_WDG_PREFIX = re.compile(r"^WDG_\d+-")
_AREA_ID = re.compile(r"^HF-0*(\d+)$", re.IGNORECASE)
QUERY_REPRESENTATIONS = ("original", "llm_summary")
QUERY_MAP_FILES = {
    "original": REPO_ROOT / "data" / "queries" / "simple.jsonl",
    "llm_summary": REPO_ROOT / "data" / "queries" / "llm_summary.jsonl",
}


def setup_logging(level: str | None = None, log_file: str | Path | None = None) -> None:
    """Configure logging for the judge scripts (console + optional log file).

    Call once at program start. Level precedence: the ``level`` argument beats
    the ``LOG_LEVEL`` environment variable, which beats ``INFO``. DEBUG also
    surfaces the OpenAI SDK's per-request HTTP logs. Idempotent: safe to call
    again (e.g. to attach a log file once the results directory exists) —
    existing handlers are reconfigured instead of duplicated.

        from judge_common import setup_logging
        setup_logging("INFO", log_file=results_dir / "run.log")
    """
    resolved = str(level if level is not None else os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    if resolved not in _LOG_LEVELS:
        raise ValueError(f"Invalid log level {level!r}; use one of {sorted(_LOG_LEVELS)}")

    root = logging.getLogger()
    root.setLevel(resolved)

    def _reconfigure(handler: logging.Handler, fmt: logging.Formatter) -> None:
        handler.setFormatter(fmt)
        handler.setLevel(resolved)

    # Console: short timestamps (the run.log carries full dates).
    stream = next((h for h in root.handlers
                   if isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)), None)
    if stream is None:
        stream = logging.StreamHandler()
        root.addHandler(stream)
    _reconfigure(stream, logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"))

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # FileHandler stores os.path.abspath(filename) in baseFilename — compare
        # against the same normalization so re-calls stay idempotent.
        target = os.path.abspath(str(path))
        file_handler = next((h for h in root.handlers
                             if isinstance(h, logging.FileHandler)
                             and getattr(h, "baseFilename", None) == target), None)
        if file_handler is None:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            root.addHandler(file_handler)
        _reconfigure(file_handler, logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"))

    # The SDK loggers log every HTTP request at INFO; keep them at WARNING
    # unless the user explicitly asked for DEBUG.
    sdk_level = logging.DEBUG if root.isEnabledFor(logging.DEBUG) else logging.WARNING
    for name in _NOISY_SDK_LOGGERS:
        logging.getLogger(name).setLevel(sdk_level)


def describe_exception(exc: BaseException) -> str:
    """Render an exception with precise origin info for log lines:

    ``TypeName: message | POST <url> | caused by ... | raised at <repo file>:<line> in <func>``

    The location is the innermost frame belonging to this repository (SDK-internal
    frames are skipped), so LLM call errors pinpoint the failing client call.
    HTTP error objects exposing an httpx request (openai/httpx SDKs) also show
    the exact request method + URL that failed; explicit ``raise ... from``
    chains add "caused by" entries. Use for logging — the results JSONL keeps
    the plain ``TypeName: message`` string.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    # the HTTP request that failed, when the SDK exposes it (openai/httpx)
    request = getattr(exc, "request", None) or getattr(getattr(exc, "response", None), "request", None)
    if request is not None:
        url = str(getattr(request, "url", "") or "")
        if url:
            parts.append(f"{getattr(request, 'method', 'HTTP')} {url}".strip())
    # explicit cause chain (openai wraps httpx errors, proxies wrap SDK errors)
    cause = exc.__cause__
    for _ in range(2):
        if cause is None:
            break
        parts.append(f"caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__
    # innermost frame from this repository pinpoints the failing call site
    repo_prefix = str(REPO_ROOT) + os.sep
    for frame in reversed(traceback.extract_tb(exc.__traceback__) or ()):
        path = str(Path(frame.filename).resolve())
        if path.startswith(repo_prefix):
            parts.append(f"raised at {os.path.relpath(path, REPO_ROOT)}:{frame.lineno} in {frame.name}")
            break
    return " | ".join(parts)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base or REPO_ROOT) / p


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not (REPO_ROOT / p).exists():
        # script-level configs (e.g. configs/model_reasoning_simple_aoa.yaml) live in JUDGE_DIR
        candidate = JUDGE_DIR / p
        if candidate.exists():
            p = candidate
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return cfg


def load_dataset(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["dataset"]["path"])
    df = read_table(path)
    id_col = cfg["dataset"].get("id_column", "sample_id")
    if id_col not in df.columns:
        raise ValueError(f"Dataset {path} is missing the id column {id_col!r}")
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one flat tabular input supported by the generic judge wrapper."""
    resolved = resolve_path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(resolved, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    raise ValueError(
        f"Unsupported table format {suffix!r} for {resolved}; "
        "use xlsx, csv, jsonl, or parquet"
    )


def load_query_map(path: str | Path) -> dict[str, str]:
    """Load a strict ``{"id", "text"}`` JSONL area-query map (repo-relative paths allowed)."""
    return _load_query_map(resolve_path(path))


def query_maps(
    representations: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Load only the external maps needed by selected representations."""
    maps: dict[str, dict[str, str]] = {}
    for representation in representations:
        if representation not in QUERY_REPRESENTATIONS:
            raise ValueError(
                f"Unknown query representation {representation!r}; "
                f"use one of {QUERY_REPRESENTATIONS}"
            )
        maps[representation] = load_query_map(QUERY_MAP_FILES[representation])
    return maps


def configured_query_representations(
    cfg: dict[str, Any],
    override: Sequence[str] | None = None,
) -> list[str]:
    """Return selected area-query representations, preserving config order."""
    values: Sequence[Any]
    if override:
        values = override
    else:
        values = cfg.get("query_representations") or ["original"]
    resolved = [
        str(value.get("id") if isinstance(value, Mapping) else value)
        for value in values
    ]
    unknown = sorted(set(resolved) - set(QUERY_REPRESENTATIONS))
    if unknown:
        raise ValueError(
            f"Unknown query representations: {unknown}; "
            f"use {list(QUERY_REPRESENTATIONS)}"
        )
    return list(dict.fromkeys(resolved))


def area_query_id(wdg_id: object, scope_of_action_id: object) -> str:
    """Convert validation ids such as WDG_2-Food/HF-2 to Food::HF-002."""
    catalog = _WDG_PREFIX.sub("", str(wdg_id).strip())
    match = _AREA_ID.fullmatch(str(scope_of_action_id).strip())
    if not catalog or match is None:
        raise ValueError(
            f"Cannot form area query id from wdg_id={wdg_id!r}, "
            f"scope_of_action_id={scope_of_action_id!r}"
        )
    return f"{catalog}::HF-{int(match.group(1)):03d}"


def resolve_task_query(
    record: Mapping[str, Any],
    target: str,
    target_cfg: Mapping[str, Any],
    representation: str,
    maps: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    """Resolve ``(query_id, query_text)`` for one experiment task."""
    if target != "area":
        criterion_id = str(record.get("criterion_id") or "").strip()
        return criterion_id, str(record[target_cfg["query_column"]])

    query_id = area_query_id(record.get("wdg_id"), record.get("scope_of_action_id"))
    if representation == "original":
        return query_id, str(record[target_cfg["query_column"]])
    try:
        return query_id, maps[representation][query_id]
    except KeyError as error:
        raise ValueError(
            f"Area query {query_id!r} is missing from the {representation!r} map"
        ) from error


def load_prompt(cfg: dict) -> tuple[str, str]:
    """Return (prompt template text, sha256 of the template)."""
    path = resolve_path(cfg["prompt"]["file"], base=JUDGE_DIR)
    text = path.read_text(encoding="utf-8")
    return text, sha256_text(text)


def build_prompt(template: str, query: str, passage: str, extra: dict | None = None) -> str:
    # str.replace (not str.format): the template contains literal JSON braces.
    # `extra` allows config-driven placeholders (e.g. {task_instruction} in the
    # German guidelines prompt); unknown keys in the template stay untouched.
    prompt = template.replace("{query}", query).replace("{passage}", passage)
    for key, value in (extra or {}).items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def _clamp_score(value) -> int | None:
    """Clamp to the 0-2 scale; tolerates strings like '2 — Klar relevant'
    (leading number is used) and decimal commas ('1,5')."""
    if value is None:
        return None
    match = re.match(r"-?\d+(?:\.\d+)?", str(value).strip().replace(",", "."))
    if not match:
        return None
    return max(0, min(2, int(round(float(match.group(0))))))


def extract_scores(text) -> tuple[dict | None, str]:
    """Extract (M, T, O) scores from a possibly reasoning-heavy model output.

    Strategies, in order:
      1. last complete JSON object containing M/T/O keys (handles
         ``{"M": .., "T": .., "O": ..}`` and ``[{"M": .., ..}]``; extra keys
         such as the guidelines variants' ``Bewertung`` or ``Begruendung``
         are ignored here — the rationale text is parsed by
         :func:`extract_reasoning`)
      2. positional JSON array ``[M, T, O]``
      3. per-key regex over quoted keys (``"O": <digit>``), then the German
         ``Bewertung:`` key fallback used by the German guidelines prompts
    Returns ``(scores|None, method)``; scores is None unless O was found.
    """
    if text is None or not str(text).strip():
        return None, "none"
    text = str(text)
    scores: dict[str, int | None] = {"M": None, "T": None, "O": None}
    method = "none"

    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        normalised = {
            _SCORE_KEY_ALIASES.get(str(k).strip().lower(), str(k).strip().upper()): v
            for k, v in data.items()
        }
        found = {k: _clamp_score(normalised[k]) for k in ("M", "T", "O") if k in normalised}
        if found:
            scores.update(found)  # keep the LAST parsable object
            method = "json_object"

    if scores["O"] is None:
        arrays = re.findall(r"\[\s*([0-2])\s*,\s*([0-2])\s*,\s*([0-2])\s*\]", text)
        if arrays:
            scores = dict(zip(("M", "T", "O"), (int(v) for v in arrays[-1])))
            method = "json_array"

    for key in ("M", "T", "O"):
        if scores[key] is None:
            hits = re.findall(rf'(?is)"{key}"\s*[:=]\s*["\']?([0-2])\b', text)
            if hits:
                scores[key] = int(hits[-1])
                if method == "none":
                    method = "regex"
    if scores["O"] is None:
        hits = re.findall(r"(?is)[\"']o[\"']\s*[:=]\s*([0-2])\b", text)
        if hits:
            scores["O"] = int(hits[-1])
            method = "regex"

    if scores["O"] is None:
        # German guidelines variant: 'Bewertung: 2' or '"Bewertung": 2' outside
        # of any parsable JSON object.
        hits = re.findall(r"(?is)bewertung[\"']?\s*[:=]\s*[\"']?([0-2])\b", text)
        if hits:
            scores["O"] = int(hits[-1])
            method = "regex"

    if scores["O"] is None:
        return None, "none"
    return scores, method


#: Key spellings accepted for the guidelines with-reasoning variant's
#: rationale field, compared lower-case with umlauts folded to ASCII
#: ("Begründung" -> "begruendung"); the prompt asks for "Begruendung".
_BEGruendung_KEY_ALIASES = frozenset({"begruendung", "begruending", "begruend"})

#: Regex fallback for the rationale when no JSON object parses (e.g. braces
#: inside the rationale text break the object scan above).
_BEGruendung_VALUE_RE = re.compile(
    r'(?is)"(begruendung|begruending|begründung|begruend)"\s*:\s*'
    r'"((?:[^"\\]|\\.)*)"')


def _begruendung_key(key) -> bool:
    """True for the rationale key in any accepted spelling/case."""
    # ü -> ue ("Begründung" -> "begruendung"), like the ASCII transliteration
    normalised = str(key).strip().lower().replace("ü", "ue")
    return normalised in _BEGruendung_KEY_ALIASES


def extract_reasoning(text) -> str | None:
    """Extract the short rationale ('Begruendung') from a model output.

    The guidelines prompt variant ``guidelines_de_v1_with_reasoning`` asks the
    model to end with ``{"Begruendung": <Kurze Begruendung>, "Bewertung": 0-2}``.
    Strategies, in order:
      1. last complete JSON object containing a Begruendung key (case- and
         umlaut-tolerant: 'Begruendung'/'Begründung', also 'Begruending')
      2. quoted-key regex fallback when no JSON object parses (e.g. braces in
         the rationale text break the object scan)
    Returns the whitespace-collapsed rationale text, or None when absent.
    """
    if text is None or not str(text).strip():
        return None
    text = str(text)
    found: str | None = None

    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if _begruendung_key(key) and isinstance(value, str) and value.strip():
                found = value.strip()  # keep the LAST parsable object

    if found is None:
        for match in _BEGruendung_VALUE_RE.finditer(text):
            raw = match.group(2)
            try:
                # unescape \n, \", \\ ... exactly as json would
                found = json.loads(f'"{raw}"')
            except json.JSONDecodeError:
                found = raw
            found = found.strip()

    if not found:
        return None
    return " ".join(found.split())


def extract_thinking(text: str | None, thinking_from_usage: Any = None) -> Any:
    """Extract model thinking/chain-of-thought trace.

    Returns thinking passed from the client/provider usage (e.g. reasoning_content,
    reasoning_details, thinking), or extracts content within <think>...</think> tags
    if present in the raw text output. Returns None when absent.
    """
    if thinking_from_usage is not None:
        return thinking_from_usage
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


@dataclass(frozen=True)
class JudgeResult:
    """One model judgment, independent of any dataset wrapper."""

    prompt: str
    raw_response: str | None
    usage: dict[str, Any]
    scores: dict[str, int | None] | None
    reasoning: str | None
    thinking: Any
    parse_method: str
    latency_s: float

    @property
    def parse_ok(self) -> bool:
        return self.scores is not None and self.scores.get("O") is not None


def parse_judgment(
    prompt: str,
    response: str | None,
    usage: dict[str, Any] | None,
    latency_s: float,
) -> JudgeResult:
    """Parse a response into the common judge result used by every wrapper."""
    usage_dict = usage or {}
    scores, method = extract_scores(response)
    return JudgeResult(
        prompt=prompt,
        raw_response=response,
        usage=usage_dict,
        scores=scores,
        reasoning=extract_reasoning(response),
        thinking=extract_thinking(response, usage_dict.get("thinking")),
        parse_method=method,
        latency_s=latency_s,
    )


def judge_one(
    client: Any,
    *,
    prompt_template: str,
    query: str,
    passage: str,
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
    **call_kwargs: Any,
) -> JudgeResult:
    """Build, request, and parse one judgment without assuming a dataset.

    Retries and checkpointing intentionally remain wrapper concerns. This
    function is the shared unit used by validation and generic-row workflows.
    """
    prompt = build_prompt(prompt_template, query, passage, extra)
    started = time.monotonic()
    response, usage = client.generate(
        [{"role": "user", "content": prompt}],
        params,
        return_usage=True,
        **call_kwargs,
    )
    return parse_judgment(prompt, response, usage, time.monotonic() - started)


def make_client(client_name: str, model_id: str, **kwargs: Any):
    clients = {
        "openrouter": OpenRouterClient,
    }
    if client_name not in clients:
        raise ValueError(f"Unknown client {client_name!r}; available: {sorted(clients)}")
    return clients[client_name](model_id, **kwargs)


def append_jsonl(path: Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _JSONL_WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_results(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_json(path, lines=True)


def export_results_xlsx(results_path: str | Path) -> Path | None:
    """Write an Excel mirror of the JSONL checkpoint as <name>.xlsx next to it.

    The mirror contains the checkpoint rows verbatim (same order, including
    retried/error rows); openpyxl-illegal control characters are stripped from
    string cells. One inherent xlsx quirk: empty strings (error rows) are
    stored as empty cells and read back as NaN. Exception: the `prompt` column
    is truncated (with an explicit marker) to Excel's 32,767-char cell limit —
    the JSONL checkpoint always keeps the full prompt. Returns the xlsx path,
    or None when the JSONL is missing or empty, or on write failure — never
    raises, since the JSONL remains the source of truth (resume keys, parse
    flags).
    """
    path = resolve_path(results_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    # convert_dates=False keeps started_at/finished_at as their original ISO
    # strings (tz-aware datetimes would make ExcelWriter raise).
    df = pd.read_json(path, lines=True, convert_dates=False, keep_default_dates=False)
    ordered = [c for c in RESULT_COLUMNS if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    # Excel cells cap at 32,767 chars: truncate long prompts (and rationales)
    # in the mirror only (with an explicit marker); the JSONL checkpoint keeps
    # the full text.
    if "prompt" in df.columns:
        marker = " …[truncated for Excel; full prompt in the JSONL checkpoint]"
        df["prompt"] = df["prompt"].map(
            lambda v: v[:32000] + marker
            if isinstance(v, str) and len(v) > 32767 else v)
    if "reasoning" in df.columns:
        marker = " …[truncated for Excel; full text in the JSONL checkpoint]"
        df["reasoning"] = df["reasoning"].map(
            lambda v: v[:32000] + marker
            if isinstance(v, str) and len(v) > 32767 else v)
    if "thinking" in df.columns:
        marker = " …[truncated for Excel; full text in the JSONL checkpoint]"
        df["thinking"] = df["thinking"].map(
            lambda v: v[:32000] + marker
            if isinstance(v, str) and len(v) > 32767 else v)
    try:
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(
                    lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)
        xlsx_path = path.with_suffix(".xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
        return xlsx_path
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Excel mirror failed for {path}: {type(exc).__name__}: {exc}")
        return None