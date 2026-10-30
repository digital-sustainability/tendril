#!/usr/bin/env python
"""Rewrite one column of a parquet file and store the replacement log beside it.

Example::

    python scripts/anonymize/anonymize_parquet.py data/notes.parquet --column text
    python scripts/anonymize/anonymize_parquet.py data/notes.parquet --column text --output data/notes_anon.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from tendril.anon import (
    DEFAULT_ENTITY_TAGS,
    anonymize_texts,
    labels_for_tags,
    parse_entity_tags,
)


def default_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_anon{path.suffix}")


def replacements_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.replacements.jsonl")


def is_missing(value: object) -> bool:
    if value is None or isinstance(value, str):
        return value is None
    return bool(pd.isna(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace detected entities in one parquet column with placeholders.",
    )
    parser.add_argument("input", type=Path, help="Parquet file to read")
    parser.add_argument("--column", required=True, help="Column to rewrite")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to the input name with _anon before the suffix",
    )
    parser.add_argument(
        "--entities",
        default=",".join(DEFAULT_ENTITY_TAGS),
        help=(
            "Comma-separated entity tags to redact. "
            "Choices: person, email, phone, url, account, secret, date, address. "
            "Default omits date and address."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be at least 1")
    try:
        labels = labels_for_tags(parse_entity_tags(args.entities))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source = args.input
    output = args.output if args.output is not None else default_output(source)
    if source.resolve() == output.resolve():
        raise SystemExit("Output path must be different from the input path.")
    if not source.is_file():
        raise SystemExit(f"Parquet file does not exist: {source}")
    frame = pd.read_parquet(source)
    if args.column not in frame.columns:
        raise SystemExit(f"Column {args.column!r} is not in {source}")

    original = frame[args.column].tolist()
    texts: list[str] = []
    positions: list[int] = []
    for position, value in enumerate(original):
        if is_missing(value):
            continue
        texts.append(value if isinstance(value, str) else str(value))
        positions.append(position)
    anonymized = anonymize_texts(texts, batch_size=args.batch_size, labels=labels)
    rewritten = list(original)
    metadata: list[dict[str, object]] = []
    for position, result in zip(positions, anonymized, strict=True):
        rewritten[position] = result.text
        metadata.append(
            {
                "row": position,
                "replacements": [item.to_dict() for item in result.replacements],
            }
        )
    frame[args.column] = rewritten
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    sidecar = replacements_path(output)
    with sidecar.open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    logger.info(
        "Wrote {} rows to {} and {} replacement records to {}",
        len(frame),
        output,
        sum(len(result.replacements) for result in anonymized),
        sidecar,
    )


if __name__ == "__main__":
    main()
