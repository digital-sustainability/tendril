#!/usr/bin/env python
"""Rewrite corpus passages in a directory of retrieval collections.

Each subdirectory that contains ``corpus.jsonl``, ``queries.jsonl``, and
``qrels/test.tsv`` is one collection. Only corpus text is rewritten. Queries
and qrels are copied so the collection stays evaluable. Replacement logs are
written next to the original ``corpus.jsonl``.

Example::

    python scripts/anonymize/anonymize_collections.py data/exports/tendril_test
    python scripts/anonymize/anonymize_collections.py data/exports/tendril_test --output data/exports/tendril_test_anon
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger

from tendril.anon import (
    DEFAULT_ENTITY_TAGS,
    MODEL_ID,
    anonymize_texts,
    build_detector,
    labels_for_tags,
    parse_entity_tags,
)
from tendril.anon.anonymize import BatchDetector


def project_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_root}")
    projects = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
        and (path / "queries.jsonl").is_file()
        and (path / "corpus.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    ]
    if not projects:
        raise SystemExit(f"No project collections found in {dataset_root}")
    return projects


def default_output(dataset_root: Path) -> Path:
    return Path(f"{dataset_root}_anon")


def write_dataset_manifest(source_root: Path, output_root: Path, projects: list[Path]) -> None:
    """Copy the MTEB dataset manifest so the anonymized export stays evaluable."""
    names = [project.name for project in projects]
    source_manifest = source_root / "mteb_dataset.json"
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise SystemExit(f"{source_manifest} is not a JSON object")
        manifest["projects"] = names
    else:
        manifest = {
            "format": "mteb-retrieval-v2",
            "hub_repo": None,
            "projects": names,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mteb_dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def anonymize_project(
    source: Path,
    destination: Path,
    *,
    labels: frozenset[str],
    batch_size: int,
    detector: BatchDetector,
) -> None:
    rows = read_jsonl(source / "corpus.jsonl")
    texts = [row["text"] if isinstance(row.get("text"), str) else "" for row in rows]
    anonymized = anonymize_texts(
        texts,
        batch_size=batch_size,
        detector=detector,
        labels=labels,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "qrels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "queries.jsonl", destination / "queries.jsonl")
    shutil.copy2(source / "qrels" / "test.tsv", destination / "qrels" / "test.tsv")
    rewritten: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for row, result in zip(rows, anonymized, strict=True):
        updated = dict(row)
        updated["text"] = result.text
        rewritten.append(updated)
        metadata.append(
            {
                "id": row.get("id"),
                "replacements": [item.to_dict() for item in result.replacements],
            }
        )
    write_jsonl(destination / "corpus.jsonl", rewritten)
    write_jsonl(source / "anon_replacements.jsonl", metadata)
    logger.info(
        "Rewrote {} passages from {} ({} replacements)",
        len(rows),
        source.name,
        sum(len(result.replacements) for result in anonymized),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite corpus passages in a set of retrieval collections, replacing detected entities with placeholders.",
    )
    parser.add_argument("input", type=Path, help="Directory of retrieval collections")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input path with _anon appended",
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
        tags = parse_entity_tags(args.entities)
        labels = labels_for_tags(tags)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_root = args.input
    output_root = args.output if args.output is not None else default_output(source_root)
    if source_root.resolve() == output_root.resolve():
        raise SystemExit("Output directory must be different from the input directory.")
    projects = project_dirs(source_root)
    logger.info("Anonymizing {} collections from {} into {}", len(projects), source_root, output_root)
    detector = build_detector(args.batch_size)
    for project in projects:
        anonymize_project(
            project,
            output_root / project.name,
            labels=labels,
            batch_size=args.batch_size,
            detector=detector,
        )
    write_dataset_manifest(source_root, output_root, projects)
    payload = {
        "model_id": MODEL_ID,
        "batch_size": args.batch_size,
        "entities": list(tags),
    }
    (source_root / "anon_run.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote run metadata to {}", source_root / "anon_run.json")


if __name__ == "__main__":
    main()
