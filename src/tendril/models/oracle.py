from collections.abc import Mapping, Sequence

from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import (
    CorpusDatasetType,
    EncodeKwargs,
    QueryDatasetType,
    RelevantDocumentsType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from tendril.models.registry import ORACLE_NAME

ORACLE_META = ModelMeta.create_empty(
    {
        "name": ORACLE_NAME,
        "revision": "1",
        "languages": ["deu-Latn"],
        "open_weights": True,
        "use_instructions": False,
        "model_type": ["sparse"],
        "modalities": ["text"],
    }
)


class OracleSearch:
    """Ranks every relevant passage first. Recall@k is still capped by how many exist."""

    def __init__(self) -> None:
        self.mteb_model_meta = ORACLE_META
        self._qrels_by_task: dict[str, dict[str, dict[str, int]]] = {}
        self._active_qrels: RelevantDocumentsType = {}

    def register_tasks(self, tasks: Sequence[AbsTask]) -> None:
        for task in tasks:
            if not task.data_loaded:
                task.load_data()
            dataset = task.dataset
            if dataset is None:
                raise RuntimeError(f"{task.metadata.name} has no dataset after load_data()")
            merged: dict[str, dict[str, int]] = {}
            for splits in dataset.values():
                for split_data in splits.values():
                    relevant = split_data["relevant_docs"]
                    if not isinstance(relevant, Mapping):
                        raise TypeError(
                            f"{task.metadata.name} relevant_docs must be a mapping, "
                            f"got {type(relevant).__name__}"
                        )
                    for query_id, docs in relevant.items():
                        if not isinstance(docs, Mapping):
                            raise TypeError(
                                f"{task.metadata.name} qrels for {query_id} must be a mapping"
                            )
                        merged.setdefault(str(query_id), {}).update(
                            {str(doc_id): int(score) for doc_id, score in docs.items()}
                        )
            self._qrels_by_task[task.metadata.name] = merged

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        del corpus, hf_split, hf_subset, encode_kwargs, num_proc
        name = task_metadata.name
        try:
            self._active_qrels = self._qrels_by_task[name]
        except KeyError as error:
            raise RuntimeError(f"Oracle has no qrels registered for {name}") from error

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        del task_metadata, hf_split, hf_subset, encode_kwargs, top_ranked, num_proc
        results: RetrievalOutputType = {}
        for query_id in queries["id"]:
            judged = self._active_qrels.get(str(query_id), {})
            relevant = [
                (doc_id, score) for doc_id, score in judged.items() if score > 0
            ]
            relevant.sort(key=lambda item: (-item[1], item[0]))
            results[str(query_id)] = {
                doc_id: float(score) for doc_id, score in relevant[:top_k]
            }
        return results
