"""Split long passages so each piece fits the detector window.

joelbarmettler/gheim-ch-560m accepts 512 tokens, two of which are reserved for special
tokens, so a piece may contain at most 510 content tokens. Cuts are placed
on a stride shorter than that limit, then a small window around each cut is
scanned. If an entity crosses the cut, the border moves to just before that
entity, or just after it when the left side would overflow the window.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

MAX_CONTENT_TOKENS = 510
TARGET_CHUNK_TOKENS = 460
BOUNDARY_HALF_WINDOW = 32
MAX_BOUNDARY_SHIFTS = 6


class CharSpan(Protocol):
    start: int
    end: int


DetectBatch = Callable[[list[str]], Sequence[Sequence[CharSpan]]]


@dataclass
class ChunkPlan:
    text: str
    offsets: list[tuple[int, int]]
    cuts: list[int]


def content_token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    """Character spans of content tokens, excluding special tokens."""
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int):
        tokenizer.model_max_length = int(1e12)
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
    finally:
        if isinstance(model_max_length, int):
            tokenizer.model_max_length = model_max_length
    offsets: list[tuple[int, int]] = []
    for start, end in encoded["offset_mapping"]:
        start_i = int(start)
        end_i = int(end)
        if start_i == 0 and end_i == 0:
            continue
        offsets.append((start_i, end_i))
    return offsets


def split_overflows(cuts: list[int]) -> list[int]:
    """Insert extra borders so no piece is longer than the model window."""
    if len(cuts) < 2:
        return list(cuts)
    bounded = [cuts[0]]
    for end in cuts[1:]:
        start = bounded[-1]
        while end - start > MAX_CONTENT_TOKENS:
            start += TARGET_CHUNK_TOKENS
            if start >= end:
                break
            bounded.append(start)
        bounded.append(end)
    return bounded


def normalize_cuts(cuts: Sequence[int], n_tokens: int) -> list[int]:
    internal = [cut for cut in sorted(set(cuts)) if 0 < cut < n_tokens]
    return [0, *internal, n_tokens]


def boundary_window(cut: int, n_tokens: int) -> tuple[int, int]:
    start = max(0, cut - BOUNDARY_HALF_WINDOW)
    end = min(n_tokens, cut + BOUNDARY_HALF_WINDOW)
    if end <= start:
        end = min(n_tokens, start + 1)
    return start, end


def token_containing(offsets: Sequence[tuple[int, int]], char_pos: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start <= char_pos < end:
            return index
        if start > char_pos:
            return index
    return len(offsets)


def token_at_or_after(offsets: Sequence[tuple[int, int]], char_pos: int) -> int:
    for index, (start, _end) in enumerate(offsets):
        if start >= char_pos:
            return index
    return len(offsets)


def cut_fits(cuts: Sequence[int], index: int, new_cut: int) -> bool:
    previous = cuts[index - 1]
    following = cuts[index + 1]
    if not previous < new_cut < following:
        return False
    if new_cut - previous > MAX_CONTENT_TOKENS:
        return False
    if following - new_cut > MAX_CONTENT_TOKENS:
        return False
    return True


def relocate_cut(
    cuts: Sequence[int],
    index: int,
    span_start: int,
    span_end: int,
    offsets: Sequence[tuple[int, int]],
) -> int:
    """Move ``cuts[index]`` so it no longer falls inside ``[span_start, span_end)``."""
    before = token_containing(offsets, span_start)
    after = token_at_or_after(offsets, span_end)
    if after - before > MAX_CONTENT_TOKENS:
        logger.warning(
            "Entity covers {} tokens, past the {}-token model window, so a chunk border cuts through it.",
            after - before,
            MAX_CONTENT_TOKENS,
        )
        return cuts[index]
    if cut_fits(cuts, index, before):
        return before
    if cut_fits(cuts, index, after):
        return after
    previous = cuts[index - 1]
    following = cuts[index + 1]
    candidates = [cut for cut in (before, after) if previous < cut < following]
    if not candidates:
        logger.warning("Could not move a chunk border off an entity without leaving an empty piece.")
        return cuts[index]

    def overflow(cut: int) -> tuple[int, int]:
        extra = max(0, cut - previous - MAX_CONTENT_TOKENS)
        extra += max(0, following - cut - MAX_CONTENT_TOKENS)
        prefer_before = 0 if cut == before else 1
        return (extra, prefer_before)

    return min(candidates, key=overflow)


def chunk_char_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    cuts: Sequence[int],
) -> list[tuple[int, int]]:
    """Map token borders onto character slices that cover ``text`` exactly."""
    n_tokens = len(offsets)
    spans: list[tuple[int, int]] = []
    for start_token, end_token in zip(cuts, cuts[1:], strict=False):
        char_start = 0 if start_token <= 0 else offsets[start_token][0]
        char_end = len(text) if end_token >= n_tokens else offsets[end_token][0]
        if char_end < char_start:
            char_end = char_start
        spans.append((char_start, char_end))
    return spans


def _crossing_span(
    spans: Sequence[CharSpan],
    window_char_start: int,
    cut_char: int,
) -> tuple[int, int] | None:
    span_start: int | None = None
    span_end: int | None = None
    for span in spans:
        absolute_start = window_char_start + int(span.start)
        absolute_end = window_char_start + int(span.end)
        if absolute_start < cut_char < absolute_end:
            span_start = absolute_start if span_start is None else min(span_start, absolute_start)
            span_end = absolute_end if span_end is None else max(span_end, absolute_end)
    if span_start is None or span_end is None:
        return None
    return span_start, span_end


def resolve_cuts(plans: Sequence[ChunkPlan], detect_batch: DetectBatch) -> None:
    """Adjust ``plan.cuts`` in place so borders avoid entities when they can."""
    active = [plan for plan in plans if len(plan.offsets) > MAX_CONTENT_TOKENS]
    for plan in active:
        plan.cuts = split_overflows(normalize_cuts(plan.cuts, len(plan.offsets)))

    for _round in range(MAX_BOUNDARY_SHIFTS):
        overflowed = False
        for plan in active:
            bounded = split_overflows(plan.cuts)
            if bounded != plan.cuts:
                overflowed = True
                plan.cuts = bounded
        windows: list[str] = []
        owners: list[tuple[ChunkPlan, int, int]] = []
        for plan in active:
            n_tokens = len(plan.offsets)
            for cut_index in range(1, len(plan.cuts) - 1):
                token_start, token_end = boundary_window(plan.cuts[cut_index], n_tokens)
                if token_end <= token_start:
                    continue
                char_start = plan.offsets[token_start][0]
                char_end = plan.offsets[token_end - 1][1]
                if char_end <= char_start:
                    continue
                windows.append(plan.text[char_start:char_end])
                owners.append((plan, cut_index, char_start))
        if not windows:
            break
        detected = detect_batch(windows)
        if len(detected) != len(windows):
            raise RuntimeError(f"Detector returned {len(detected)} results for {len(windows)} boundary windows.")
        moved = False
        for (plan, cut_index, char_start), spans in zip(owners, detected, strict=True):
            if cut_index >= len(plan.cuts) - 1:
                continue
            cut_token = plan.cuts[cut_index]
            if cut_token <= 0 or cut_token >= len(plan.offsets):
                continue
            crossing = _crossing_span(spans, char_start, plan.offsets[cut_token][0])
            if crossing is None:
                continue
            new_cut = relocate_cut(plan.cuts, cut_index, crossing[0], crossing[1], plan.offsets)
            if new_cut != plan.cuts[cut_index]:
                plan.cuts[cut_index] = new_cut
                moved = True
        for plan in active:
            plan.cuts = normalize_cuts(plan.cuts, len(plan.offsets))
        if not moved and not overflowed:
            break
    else:
        for plan in active:
            plan.cuts = split_overflows(normalize_cuts(plan.cuts, len(plan.offsets)))


def choose_cuts(
    text: str,
    offsets: Sequence[tuple[int, int]],
    detect_batch: DetectBatch,
) -> list[int]:
    """Return token indexes, including ``0`` and ``len(offsets)``, that bound each piece."""
    n_tokens = len(offsets)
    if n_tokens <= MAX_CONTENT_TOKENS:
        return [0, n_tokens]
    plan = ChunkPlan(
        text=text,
        offsets=list(offsets),
        cuts=split_overflows([0, n_tokens]),
    )
    resolve_cuts([plan], detect_batch)
    return plan.cuts
