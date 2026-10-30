"""Detect entities with the Swiss gheim model and rewrite them."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol

import torch
from gheim import LocalDetector, Span
from gheim.detectors.composite import _find_regex_spans, _mask_spans
from tqdm import tqdm

from tendril.anon.chunking import (
    MAX_CONTENT_TOKENS,
    ChunkPlan,
    chunk_char_spans,
    content_token_offsets,
    resolve_cuts,
    split_overflows,
)
from tendril.anon.replace import rewrite_text
from tendril.anon.types import AnonymizedText

MODEL_ID = "joelbarmettler/gheim-ch-560m"
GROUP_SIZE = 512


class BatchDetector(Protocol):
    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        """Return detected spans for each text, with offsets into that text."""


def build_detector(batch_size: int = 16) -> LocalDetector:
    """Load the Apache-2.0 Swiss model, in bf16 when a GPU is available."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    return LocalDetector(
        model_id=MODEL_ID,
        device="auto",
        dtype=dtype,
        batch_size=batch_size,
    )


def anonymize_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 16,
    detector: BatchDetector | None = None,
    labels: Collection[str] | None = None,
) -> list[AnonymizedText]:
    """Rewrite each text. Repeats inside one text share a placeholder.

    ``labels`` is a set of gheim detector labels. ``None`` redacts persons,
    emails, phones, URLs, account numbers, and secrets, and leaves dates and
    addresses in place. ``detector`` defaults to :func:`build_detector`. Tests
    pass a fake detector so the model weights stay unloaded.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    active = detector if detector is not None else build_detector(batch_size)
    if hasattr(active, "batch_size"):
        active.batch_size = batch_size
    results: list[AnonymizedText] = []
    starts = range(0, len(texts), GROUP_SIZE)
    if len(texts) > GROUP_SIZE:
        starts = tqdm(list(starts), desc="Anonymizing")
    for start in starts:
        batch = list(texts[start : start + GROUP_SIZE])
        span_lists = detect_entities(active, batch)
        for text, spans in zip(batch, span_lists, strict=True):
            results.append(rewrite_text(text, spans, labels=labels))
    return results


def detect_entities(detector: BatchDetector, texts: Sequence[str]) -> list[list[Span]]:
    """Detect entities, splitting passages that exceed the model window.

    Checksummed and otherwise rigid formats (email, phone, URL, IBAN, AHV,
    VAT, dates, tokens) are taken from gheim's regex catalogue first. Those
    characters are blanked, then the model runs on what remains, in batch.
    ``CompositeDetector.detect`` does the same two passes, but one text at a
    time, which would give up batched forwards.
    """
    structured = [_find_regex_spans(text) if text else [] for text in texts]
    masked = [_mask_spans(text, spans) for text, spans in zip(texts, structured, strict=True)]
    model_spans = _detect_model_spans(detector, masked, texts)
    return [
        _merge_structured_and_model(regex_spans, found)
        for regex_spans, found in zip(structured, model_spans, strict=True)
    ]


def _merge_structured_and_model(structured: list[Span], model_spans: list[Span]) -> list[Span]:
    if not structured:
        return model_spans
    occupied = [(span.start, span.end) for span in structured]
    merged = list(structured)
    for span in model_spans:
        if any(span.start < end and start < span.end for start, end in occupied):
            continue
        merged.append(span)
    merged.sort(key=lambda span: (span.start, -(span.end - span.start)))
    return merged


def _detect_model_spans(
    detector: BatchDetector,
    model_texts: Sequence[str],
    surface_texts: Sequence[str],
) -> list[list[Span]]:
    """Run the model on ``model_texts`` and read surfaces from ``surface_texts``.

    The two sequences are the same length, character for character. Structured
    spans are blanked in ``model_texts`` so the model does not see them.
    """
    tokenizer = _tokenizer_for(detector)
    plans: list[ChunkPlan | None] = []
    long_plans: list[ChunkPlan] = []
    for text in model_texts:
        if not text or tokenizer is None:
            plans.append(None)
            continue
        offsets = content_token_offsets(tokenizer, text)
        if len(offsets) <= MAX_CONTENT_TOKENS:
            plans.append(None)
            continue
        plan = ChunkPlan(
            text=text,
            offsets=offsets,
            cuts=split_overflows([0, len(offsets)]),
        )
        plans.append(plan)
        long_plans.append(plan)
    if long_plans:
        resolve_cuts(long_plans, lambda windows: _detect_sorted(detector, windows))

    pieces: list[tuple[int, int, str]] = []
    for index, text in enumerate(model_texts):
        if not text:
            continue
        plan = plans[index]
        if plan is None:
            pieces.append((index, 0, text))
            continue
        for char_start, char_end in chunk_char_spans(text, plan.offsets, plan.cuts):
            if char_end <= char_start:
                continue
            pieces.append((index, char_start, text[char_start:char_end]))
    detected = _detect_sorted(detector, [chunk for _index, _offset, chunk in pieces])
    return _project(surface_texts, pieces, detected)


def _tokenizer_for(detector: BatchDetector) -> object | None:
    if not isinstance(detector, LocalDetector):
        return None
    return detector._load().tokenizer


def _detect_sorted(detector: BatchDetector, texts: list[str]) -> list[list[Span]]:
    if not texts:
        return []
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    ordered_texts = [texts[index] for index in order]
    detected = detector.detect_batch(ordered_texts)
    if len(detected) != len(ordered_texts):
        raise RuntimeError(
            f"Detector returned {len(detected)} results for {len(ordered_texts)} texts."
        )
    spans: list[list[Span]] = [[] for _ in texts]
    for index, found in zip(order, detected, strict=True):
        spans[index] = list(found)
    return spans


def _project(
    texts: Sequence[str],
    pieces: Sequence[tuple[int, int, str]],
    detected: Sequence[Sequence[Span]],
) -> list[list[Span]]:
    grouped: list[list[Span]] = [[] for _ in texts]
    seen: list[set[tuple[int, int, str]]] = [set() for _ in texts]
    for (text_index, char_offset, _chunk), found in zip(pieces, detected, strict=True):
        full = texts[text_index]
        bucket = grouped[text_index]
        known = seen[text_index]
        for span in found:
            start = int(span.start) + char_offset
            end = int(span.end) + char_offset
            if start < 0 or end > len(full) or start >= end:
                continue
            label = str(span.label)
            key = (start, end, label)
            if key in known:
                continue
            known.add(key)
            bucket.append(
                Span(
                    label=label,
                    start=start,
                    end=end,
                    text=full[start:end],
                    score=float(getattr(span, "score", 1.0)),
                )
            )
    for spans in grouped:
        spans.sort(key=lambda span: (span.start, span.end))
    return grouped
