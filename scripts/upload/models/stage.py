"""Stage a fine-tuned sentence-transformers folder for upload and check that it loads."""

from __future__ import annotations

import shutil
from pathlib import Path

from integrity import disk_sha256, sync_tree
from loguru import logger

EXCLUDED_FILES = ("training_args.bin",)
REQUIRED_FILES = (
    "modules.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
)


def stage(source: Path, out: Path, verify: bool = True) -> list[dict]:
    """Copy the model without the pickled training arguments; returns no dataset configs."""
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(
            f"ERROR: {source} is not a sentence-transformers folder (missing {', '.join(missing)})"
        )
    shutil.copytree(
        source, out, ignore=shutil.ignore_patterns(*EXCLUDED_FILES), dirs_exist_ok=True
    )
    sync_tree(out)
    for path in sorted(path for path in out.rglob("*") if path.is_file()):
        relative = path.relative_to(out)
        if disk_sha256(path) != disk_sha256(source / relative):
            raise SystemExit(
                f"ERROR: staged {relative} differs from {source / relative} on disk; the copy is corrupt"
            )
    logger.info(f"Staged files match {source} on disk")
    if verify:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(out), device="cpu")
        query = model.encode_query(["Regionale und saisonale Lebensmittel"])
        document = model.encode_document(
            ["Die Lieferantin bevorzugt Produkte aus der Region."]
        )
        logger.info(
            f"Loaded {out.name}: dim {query.shape[1]}, prompts {model.prompts}, "
            f"max_seq_length {model.max_seq_length}, similarity {float(model.similarity(query, document)[0, 0]):.3f}"
        )
    return []
