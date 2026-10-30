"""Stage, validate, and upload the TENDRIL datasets and fine-tuned models to the Hugging Face Hub.

Each artifact has a folder next to this script with its model or dataset card (``README.md``) and license files.
Datasets share ``licenses/OpenRAIL-D.md``. Repositories are created private unless ``--public`` is passed.

Example::

    uv run python scripts/upload/upload.py all --dry-run
    uv run python scripts/upload/upload.py test train
    uv run python scripts/upload/upload.py arctic most-embed-de
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from integrity import disk_sha256
from loguru import logger
from models import stage as model_stage
from test_set import stage as test_stage
from train_set import stage as train_stage

from tendril.paths import REPOSITORY_PATH

ORGANIZATION = "IntelliProcure"
NAME = "TENDRIL"

UPLOAD_DIR = Path(__file__).resolve().parent
DATASET_LICENSE = UPLOAD_DIR / "licenses" / "OpenRAIL-D.md"
DEFAULT_STAGING = REPOSITORY_PATH / "tmp" / "upload"
RESTRICTIONS = REPOSITORY_PATH / "data" / "exports" / "restrictions"
PRIVATE = REPOSITORY_PATH / "private"

CARD_FILES_EXCLUDED = ("stage.py", "__pycache__")
FORBIDDEN_PATTERNS = (
    "*replacements*",
    "training_args.bin",
    "*.sqlite",
    "results.jsonl",
    "backup/*",
    "anon_run.json",
)
DRAFT_MARKER = "DRAFT"
TODO_MARKER = "TODO"


@dataclass(frozen=True)
class Sources:
    """Local directories that only exist where the training data and models were produced."""

    train: Path | None = None
    test_pipeline: Path | None = None
    arctic: Path | None = None
    most_embed: Path | None = None


def required(path: Path | None, option: str) -> Path:
    if path is None:
        raise SystemExit(f"ERROR: pass {option}")
    return path


@dataclass(frozen=True)
class Artifact:
    repo_id: str
    repo_type: str
    card_dir: Path
    stage: Callable[[Path], list[dict]]
    license_file: Path | None = None


def artifacts(sources: Sources = Sources()) -> dict[str, Artifact]:
    test_repo = f"{ORGANIZATION}/{NAME}"
    return {
        "test": Artifact(
            repo_id=test_repo,
            repo_type="dataset",
            card_dir=UPLOAD_DIR / "test_set",
            stage=partial(
                _stage_test,
                REPOSITORY_PATH / "data" / "exports" / "tendril_test_anon",
                test_repo,
            ),
            license_file=DATASET_LICENSE,
        ),
        "train": Artifact(
            repo_id=f"{ORGANIZATION}/{NAME}-Train",
            repo_type="dataset",
            card_dir=UPLOAD_DIR / "train_set",
            stage=partial(_stage_train, sources.train, sources.test_pipeline),
            license_file=DATASET_LICENSE,
        ),
        "arctic": Artifact(
            repo_id=f"{ORGANIZATION}/{NAME}-arctic-embed-l-v2.0",
            repo_type="model",
            card_dir=UPLOAD_DIR / "models" / "arctic-embed-l-v2.0",
            stage=partial(_stage_model, sources.arctic, "--arctic-model"),
        ),
        "most-embed-de": Artifact(
            repo_id=f"{ORGANIZATION}/{NAME}-most-embed-de",
            repo_type="model",
            card_dir=UPLOAD_DIR / "models" / "most-embed-de",
            stage=partial(_stage_model, sources.most_embed, "--most-embed-model"),
        ),
    }


def _stage_test(source: Path, repo_id: str, out: Path) -> list[dict]:
    return test_stage.stage(
        source,
        out,
        repo_id,
        PRIVATE / "simap_project_aliases.json",
        RESTRICTIONS / "test_projects.txt",
    )


def _stage_train(source: Path | None, test_pipeline: Path | None, out: Path) -> list[dict]:
    return train_stage.stage(
        required(source, "--train-source"),
        out,
        PRIVATE / "hash_key",
        RESTRICTIONS / "train_projects.txt",
        required(test_pipeline, "--test-pipeline"),
    )


def _stage_model(source: Path | None, option: str, out: Path) -> list[dict]:
    return model_stage.stage(required(source, option), out)


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise SystemExit("ERROR: card has no YAML front matter")
    _, front, body = text.split("---\n", 2)
    return yaml.safe_load(front) or {}, body


def write_card(card: Path, target: Path, configs: list[dict]) -> None:
    metadata, body = split_front_matter(card.read_text())
    if configs:
        metadata["configs"] = configs
    front = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, width=120)
    target.write_text(f"---\n{front}---\n{body}")


def stage_artifact(key: str, artifact: Artifact, staging: Path) -> Path:
    out = staging / key
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    logger.info(f"Staging {key} into {out}")
    configs = artifact.stage(out)
    for path in sorted(artifact.card_dir.iterdir()):
        if path.name in CARD_FILES_EXCLUDED:
            continue
        if path.name == "README.md":
            write_card(path, out / "README.md", configs)
        else:
            shutil.copy2(path, out / path.name)
    if artifact.license_file is not None:
        shutil.copy2(artifact.license_file, out / "LICENSE.md")
    return out


def validate_staged(out: Path, allow_todo: bool, public: bool) -> None:
    files = sorted(
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    )
    forbidden = [
        name
        for name in files
        if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_PATTERNS)
    ]
    if forbidden:
        raise SystemExit(
            f"ERROR: forbidden files staged in {out}: {', '.join(forbidden)}"
        )
    if not (out / "README.md").is_file():
        raise SystemExit(f"ERROR: no README.md card in {out}")
    todos = [
        line.strip()
        for line in (out / "README.md").read_text().splitlines()
        if TODO_MARKER in line
    ]
    if todos and not allow_todo:
        listed = "\n  ".join(todos)
        raise SystemExit(
            f"ERROR: unresolved {TODO_MARKER} in {out / 'README.md'} (pass --allow-todo):\n  {listed}"
        )
    if todos:
        logger.warning(
            f"{len(todos)} unresolved {TODO_MARKER} lines in {out / 'README.md'}"
        )
    licenses = [out / name for name in files if name.startswith(("LICENSE", "NOTICE"))]
    if not any(path.name.startswith("LICENSE") for path in licenses):
        raise SystemExit(f"ERROR: no LICENSE file staged in {out}")
    drafts = [
        path.name
        for path in licenses
        if DRAFT_MARKER in path.read_text().splitlines()[0]
    ]
    if drafts and public:
        raise SystemExit(
            f"ERROR: refusing a public upload while {', '.join(drafts)} is marked {DRAFT_MARKER}"
        )


def describe(out: Path) -> None:
    total = 0
    for path in sorted(path for path in out.rglob("*") if path.is_file()):
        size = path.stat().st_size
        total += size
        logger.info(f"  {path.relative_to(out)}  {size / 1e6:.1f} MB")
    logger.info(f"  total {total / 1e6:.1f} MB")


def upload(key: str, artifact: Artifact, out: Path, public: bool) -> None:
    api = HfApi()
    url = api.create_repo(
        artifact.repo_id,
        repo_type=artifact.repo_type,
        private=not public,
        exist_ok=True,
    )
    info = api.repo_info(artifact.repo_id, repo_type=artifact.repo_type)
    if info.private == public:
        logger.warning(
            f"{artifact.repo_id} already exists as {'private' if info.private else 'public'}; "
            "visibility is left unchanged"
        )
    commit = api.upload_folder(
        repo_id=artifact.repo_id,
        repo_type=artifact.repo_type,
        folder_path=out,
        commit_message=f"Upload {key} artifact",
        delete_patterns="*",
    )
    logger.info(f"Uploaded {key} to {url} ({commit.oid[:8]})")
    verify_remote(api, artifact, out, commit.oid)


def verify_remote(api: HfApi, artifact: Artifact, out: Path, revision: str) -> None:
    """Compare the Hub's LFS checksums with the staged files as they are on disk."""
    info = api.repo_info(
        artifact.repo_id,
        repo_type=artifact.repo_type,
        revision=revision,
        files_metadata=True,
    )
    remote = {
        sibling.rfilename: sibling.lfs.sha256
        for sibling in info.siblings
        if sibling.lfs is not None
    }
    mismatched = [
        name for name, sha in remote.items() if disk_sha256(out / name) != sha
    ]
    if mismatched:
        raise SystemExit(
            f"ERROR: {artifact.repo_id} differs from {out} on disk: {', '.join(mismatched)}"
        )
    logger.info(f"Verified {len(remote)} LFS files of {artifact.repo_id} against {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Upload {NAME} datasets and models to the Hugging Face Hub"
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        choices=[*artifacts(), "all"],
        help="Artifacts to stage and upload",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=DEFAULT_STAGING,
        help=f"Directory for staged files (default: {DEFAULT_STAGING})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Stage and validate only; do not upload"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create new repositories as public (default: private). Existing repositories keep their visibility",
    )
    parser.add_argument(
        "--allow-todo",
        action="store_true",
        help=f"Allow cards with unresolved {TODO_MARKER} markers (private drafts only)",
    )
    parser.add_argument("--train-source", type=Path, help="Directory of the TENDRIL-Train build (for 'train')")
    parser.add_argument(
        "--test-pipeline",
        type=Path,
        help="Parsed test-set projects; the training set is checked against them (for 'train')",
    )
    parser.add_argument("--arctic-model", type=Path, help="Fine-tuned Arctic Embed directory (for 'arctic')")
    parser.add_argument("--most-embed-model", type=Path, help="Fine-tuned Most Embed directory (for 'most-embed-de')")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.public and args.allow_todo:
        raise SystemExit("ERROR: --allow-todo is not allowed together with --public")
    registry = artifacts(Sources(args.train_source, args.test_pipeline, args.arctic_model, args.most_embed_model))
    keys = (
        list(registry)
        if "all" in args.artifacts
        else list(dict.fromkeys(args.artifacts))
    )
    for key in keys:
        artifact = registry[key]
        out = stage_artifact(key, artifact, args.staging)
        validate_staged(out, args.allow_todo, args.public)
        describe(out)
        if args.dry_run:
            logger.info(
                f"Dry run: would upload {out} to {artifact.repo_type} {artifact.repo_id}"
            )
            continue
        upload(key, artifact, out, args.public)


if __name__ == "__main__":
    main()
