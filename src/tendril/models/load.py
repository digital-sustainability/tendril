import os
import time
from collections.abc import Sequence
from pathlib import Path

import mteb
from loguru import logger
from mteb.abstasks.abstask import AbsTask
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import MTEBModels
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from tendril.models.oracle import OracleSearch
from tendril.models.registry import (
    ORACLE_NAME,
    KnownModel,
    known_models_by_name,
    reviewed_names,
)

_TEXT_ONLY_MODALITIES = ("text",)
_UNUSED_LOAD_MODALITIES = frozenset({"image", "audio", "video"})
UNREVIEWED_DELAY_S = 10
_PROBLEM_HINTS = {
    "jinaai/jina-embeddings-v4": (
        "jinaai/jina-embeddings-v4 needs transformers<5 and is not supported in "
        "this environment. Evaluate it from a second env or repo."
    ),
    "jinaai/jina-embeddings-v3": (
        "jinaai/jina-embeddings-v3 is not a reviewed TENDRIL model and may need "
        "flash-attn plus an older Transformers stack."
    ),
}


def require_env(models: Sequence[KnownModel]) -> None:
    missing: list[str] = []
    for model in models:
        for variable in model.env_vars:
            if not os.environ.get(variable, "").strip():
                missing.append(f"{model.name} needs {variable}")
    if missing:
        raise SystemExit("ERROR: " + "; ".join(missing) + " (environment or .env)")


def registry_meta(name: str) -> ModelMeta | None:
    try:
        return mteb.get_model_meta(name, fetch_from_hf=False)
    except KeyError:
        return None


def warn_unreviewed(name: str, *, force: bool) -> None:
    hint = _PROBLEM_HINTS.get(name)
    if hint:
        logger.warning(hint)

    meta = registry_meta(name)
    if meta is None:
        logger.warning(
            f"{name} is not a model benchmarked by us and is not in the MTEB registry. "
            "get_model will fall back to a generic SentenceTransformer load. "
            "Task adapters and query/document prefixes may be missing."
        )
    else:
        extras = ", ".join(meta.extra_requirements_groups or []) or "none"
        logger.warning(
            f"{name} is in the MTEB registry but was not originally benchmarked for TENDRIL. "
            f"use_instructions={meta.use_instructions}, extras={extras}. "
            "Registry metadata can be incomplete; prefixes may still be wrong."
        )
    if force:
        return
    logger.warning(f"Continuing in {UNREVIEWED_DELAY_S}s. Pass --force to skip the wait.")
    time.sleep(UNREVIEWED_DELAY_S)


def local_sentence_transformer_dir(name: str) -> Path | None:
    """Return a local SentenceTransformer directory, or None for a Hub id.

    Absolute paths and existing directories are checkpoints. A Hub id such as
    ``BAAI/bge-m3`` is left for the MTEB registry.
    """
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        return None
    if not path.is_dir():
        raise SystemExit(f"ERROR: local model path does not exist: {path}")
    if not (path / "modules.json").is_file():
        raise SystemExit(
            f"ERROR: {path} has no modules.json, so it is not a SentenceTransformer checkpoint."
        )
    return path.resolve()


def load_local_sentence_transformer(path: Path) -> MTEBModels:
    model = SentenceTransformerEncoderWrapper(str(path))
    logger.info(
        f"Loaded {path.name} with checkpoint prompts {model.model_prompts}. "
        "Those strings are the query and document prefixes used at evaluation."
    )
    return model


def resolve_model_name(name: str, *, force: bool) -> str:
    known = known_models_by_name().get(name)
    if known is None:
        local = local_sentence_transformer_dir(name)
        if local is not None:
            logger.info(f"Using local SentenceTransformer checkpoint {local}")
            return str(local)
        warn_unreviewed(name, force=force)
        return name
    require_env((known,))
    if name != known.name:
        logger.info(f"Resolved {name} to reviewed id {known.name}")
    return known.name


def load_mteb_model(name: str) -> MTEBModels:
    meta = mteb.get_model_meta(name, fetch_from_hf=False)
    advertised = tuple(meta.modalities or _TEXT_ONLY_MODALITIES)
    unused = [modality for modality in advertised if modality in _UNUSED_LOAD_MODALITIES]
    if unused:
        kept = [modality for modality in advertised if modality not in _UNUSED_LOAD_MODALITIES]
        if not kept:
            kept = list(_TEXT_ONLY_MODALITIES)
        logger.info(
            f"{name} advertises modalities {list(advertised)}; "
            f"loading as {kept} for TENDRIL text retrieval (skips {', '.join(unused)} extras)."
        )
        meta = meta.model_copy(update={"modalities": kept})
    return meta.load_model()


def load_model(name: str, tasks: Sequence[AbsTask]) -> MTEBModels:
    if name == ORACLE_NAME:
        oracle = OracleSearch()
        oracle.register_tasks(tasks)
        return oracle
    if name in reviewed_names():
        return load_mteb_model(name)
    local = local_sentence_transformer_dir(name)
    if local is not None:
        return load_local_sentence_transformer(local)
    return mteb.get_model(name)
