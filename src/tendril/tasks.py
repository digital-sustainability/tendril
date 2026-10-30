"""Load the TENDRIL test projects from the Hugging Face Hub as directly evaluable MTEB tasks.

Example::

    import mteb

    from tendril.tasks import load_hub_tasks

    model = mteb.get_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    results = mteb.evaluate(model, tasks=load_hub_tasks("IntelliProcure/TENDRIL"))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader
from mteb.abstasks.task_metadata import TaskMetadata

from tendril.hub import test_projects


def task_metadata(name: str, dataset_path: str, revision: str) -> TaskMetadata:
    return TaskMetadata(
        dataset={"path": dataset_path, "revision": revision},
        name=f"TENDRIL-{name}",
        description=f"TENDRIL sustainability-area retrieval for anonymized project {name}.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        domains=["Government", "Legal"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        is_public=False,
    )


class HubProjectRetrievalTask(AbsTaskRetrieval):
    """One uploaded project config as an MTEB retrieval task."""

    repo_id: str
    project: str
    revision: str = "main"

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        del kwargs
        if self.data_loaded:
            return
        split = RetrievalDatasetLoader(
            hf_repo=self.repo_id,
            revision=self.revision,
            split="test",
            config=self.project,
        ).load(num_proc=num_proc)
        self.dataset = {"default": {"test": split}}
        self.data_loaded = True


def _task_class_name(prefix: str, project: str) -> str:
    return f"{prefix}_{project.replace('-', '_')}"


def make_hub_task(
    repo_id: str,
    project: str,
    revision: str = "main",
) -> HubProjectRetrievalTask:
    return type(
        _task_class_name("TENDRILHub", project),
        (HubProjectRetrievalTask,),
        {
            "metadata": task_metadata(project, repo_id, revision),
            "repo_id": repo_id,
            "project": project,
            "revision": revision,
        },
    )()


def select_projects(all_projects: Sequence[str], selected: Sequence[str] | None) -> list[str]:
    if selected is None:
        return list(all_projects)
    unknown = set(selected) - set(all_projects)
    if unknown:
        raise ValueError(f"Unknown projects: {', '.join(sorted(unknown))}")
    return list(selected)


def load_hub_tasks(
    repo_id: str,
    projects: Sequence[str] | None = None,
    revision: str = "main",
) -> list[HubProjectRetrievalTask]:
    """Load all or selected uploaded project configs for ``mteb.evaluate``."""
    names = select_projects(test_projects(repo_id, revision), projects)
    return [make_hub_task(repo_id, project, revision) for project in names]
