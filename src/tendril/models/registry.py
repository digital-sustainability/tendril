from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

ORACLE_NAME = "tendril/oracle"

ModelGroup = Literal["baseline", "local", "api", "reference"]
MODEL_GROUPS = get_args(ModelGroup)


@dataclass(frozen=True)
class KnownModel:
    name: str
    group: ModelGroup
    aliases: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    notes: str = ""
    base: str | None = None
    revision: str | None = None


KNOWN_MODELS = (
    KnownModel(
        ORACLE_NAME,
        "reference",
        aliases=("oracle",),
        notes="Perfect ranking of relevant passages. Ceiling is limited by relevant count per query.",
    ),
    KnownModel(
        "mteb/baseline-bm25s",
        "baseline",
        aliases=("bm25s",),
        notes="Lexical baseline. CPU is fine.",
    ),
    KnownModel(
        "BAAI/bge-m3",
        "local",
        notes="Open-weight dense model. Do not add a BGE instruction prefix.",
    ),
    KnownModel(
        "malteos/most-embed-de",
        "local",
        base="nvidia/Nemotron-3-Embed-1B-BF16",
        notes="German Nemotron-3 finetune. Uses query: / passage: prefixes.",
    ),
    KnownModel(
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "local",
        aliases=("nvidia/Nemotron-3-Embed-1B",),
        notes="Official id is the BF16 checkpoint. Uses query: / passage: prefixes.",
    ),
    KnownModel(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "local",
        aliases=("Snowflake/arctic-embed-l-v2.0",),
        notes="Query-only prefix (query: ). Documents are unprefixed.",
    ),
    KnownModel(
        "openai/text-embedding-3-large",
        "api",
        env_vars=("OPENAI_API_KEY",),
        notes="No retrieval prefix. Truncates at 8191 tokens.",
    ),
    KnownModel(
        "voyageai/voyage-4-large",
        "api",
        env_vars=("VOYAGE_API_KEY",),
        notes="Official Voyage wrapper sets input_type query vs document.",
    ),
    KnownModel(
        "google/gemini-embedding-2",
        "api",
        env_vars=("GEMINI_API_KEY",),
        notes="Official Gemini wrapper applies retrieval query/document prefixes. Image/audio extras are not loaded.",
    ),
)


def known_models_by_name() -> dict[str, KnownModel]:
    mapping: dict[str, KnownModel] = {}
    for model in KNOWN_MODELS:
        mapping[model.name] = model
        for alias in model.aliases:
            mapping[alias] = model
    return mapping


def models_in_group(group: ModelGroup) -> list[KnownModel]:
    return [model for model in KNOWN_MODELS if model.group == group]


def reviewed_names() -> set[str]:
    return {model.name for model in KNOWN_MODELS}


def output_stem(model_name: str) -> str:
    path = Path(model_name)
    if path.is_absolute():
        return path.name
    return model_name.replace("/", "__")
