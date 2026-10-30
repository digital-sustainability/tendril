"""Load the published TENDRIL datasets from the Hugging Face Hub.

The Hub repos are the single source of truth for every table, figure, and
statistic. Only the scripts that build the datasets read the local exports.

The test repo stores one MTEB retrieval task per project under
``data/<Sector>-<N>/{corpus,queries,qrels}.parquet`` and lists the projects in
``mteb_dataset.json``. The training repo stores ``data/{corpus,queries,qrels}.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import orjson
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

TEST_REPO = "IntelliProcure/TENDRIL"
TRAIN_REPO = "IntelliProcure/TENDRIL-Train"
COMPONENTS = ("corpus", "queries", "qrels")
MANIFEST = "mteb_dataset.json"


def resolve_revision(repo_id: str, revision: str = "main") -> str:
    """Return the commit hash that a branch, tag, or hash of a dataset repo points to."""
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    if sha is None:
        raise SystemExit(f"Cannot resolve {repo_id}@{revision} to a commit")
    return sha


def read_projects(manifest: Path) -> list[str]:
    return [str(project) for project in orjson.loads(manifest.read_bytes())["projects"]]


def test_projects(repo_id: str = TEST_REPO, revision: str = "main") -> list[str]:
    """Return the project names listed in the test repo's manifest."""
    path = hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision)
    return read_projects(Path(path))


def download_test_set(repo_id: str = TEST_REPO, revision: str = "main") -> list[Path]:
    """Download the test set and return one directory per project, in manifest order."""
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[MANIFEST, "data/*/*.parquet"],
        )
    )
    projects = [root / "data" / name for name in read_projects(root / MANIFEST)]
    missing = [
        str(path / f"{component}.parquet")
        for path in projects
        for component in COMPONENTS
        if not (path / f"{component}.parquet").is_file()
    ]
    if missing:
        raise SystemExit(f"{repo_id}@{revision} is missing {', '.join(missing)}")
    if not projects:
        raise SystemExit(f"No project tasks listed in {repo_id}@{revision}")
    return projects


def download_train_set(repo_id: str = TRAIN_REPO, revision: str = "main") -> Path:
    """Download the training set and return its ``data`` directory."""
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/*.parquet"],
        )
    )
    data = root / "data"
    missing = [
        str(data / f"{component}.parquet")
        for component in COMPONENTS
        if not (data / f"{component}.parquet").is_file()
    ]
    if missing:
        raise SystemExit(f"{repo_id}@{revision} is missing {', '.join(missing)}")
    return data
