# TENDRIL: Tender Retrieval Benchmark

TENDRIL is a German retrieval benchmark for finding sustainability-relevant passages in Swiss public procurement
tenders. Each query describes an *area of action* from our sustainability catalogs, which are based on Swiss and
EU recommendations for sustainable public procurement; within one tender project, the task is to retrieve the
passages that address it. This repository contains the code to build the
datasets, fine-tune embedding models, and evaluate retrieval systems.

## Resources

| Resource | Hugging Face | Content |
|---|---|---|
| TENDRIL | [IntelliProcure/TENDRIL](https://huggingface.co/datasets/IntelliProcure/TENDRIL) | Human-annotated test collection, one MTEB retrieval task per project |
| TENDRIL-Train | [IntelliProcure/TENDRIL-Train](https://huggingface.co/datasets/IntelliProcure/TENDRIL-Train) | LLM-judged training data with graded relevance (0/1/2) |
| Arctic Embed fine-tune | [IntelliProcure/TENDRIL-arctic-embed-l-v2.0](https://huggingface.co/IntelliProcure/TENDRIL-arctic-embed-l-v2.0) | `Snowflake/snowflake-arctic-embed-l-v2.0` fine-tuned on TENDRIL-Train |
| Most Embed fine-tune | [IntelliProcure/TENDRIL-most-embed-de](https://huggingface.co/IntelliProcure/TENDRIL-most-embed-de) | `malteos/most-embed-de` fine-tuned on TENDRIL-Train |

Both datasets are gated: accept the TENDRIL Open RAIL-D terms on the Hub, then log in with `hf auth login`.

## Quickstart

Install the dependencies with [uv](https://docs.astral.sh/uv/) (Python 3.12):

```bash
uv sync
```

Evaluate any Sentence Transformers model, from the Hub or a local directory, on TENDRIL:

```bash
uv run python examples/evaluate_tendril.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The script loads every project as an MTEB task and prints nDCG@10 per project and the mean over projects, the main
score of the benchmark. In your own code:

```python
import mteb
from sentence_transformers import SentenceTransformer

from tendril.tasks import load_hub_tasks

tasks = load_hub_tasks("IntelliProcure/TENDRIL")
results = mteb.evaluate(SentenceTransformer("path/or/model-id"), tasks=tasks)
```

To reproduce the paper's results, with all metrics, confidence intervals and significance tests, use the scripts in
[`scripts/evaluation`](scripts/evaluation/README.md).

## Repository layout

| Directory | Content |
|---|---|
| [`scripts/queries`](scripts/queries) | Query texts: area-of-action names and LLM summaries (`data/queries/`), training queries |
| [`scripts/test_set`](scripts/test_set/README.md) | Building TENDRIL from the INCEpTION export, restriction filter |
| [`scripts/annotation`](scripts/annotation/README.md) | Inter-annotator agreement of the validation and test annotations |
| [`scripts/train_set`](scripts/train_set/README.md) | Selecting the query-passage pairs of TENDRIL-Train |
| [`scripts/judge`](scripts/judge/README.md) | LLM judge: calibration runs, analysis, and labeling of TENDRIL-Train |
| [`scripts/anonymize`](scripts/anonymize) | Anonymization of passages (test and training set) |
| [`scripts/training`](scripts/training/README.md) | Preparing training examples and fine-tuning embedding models |
| [`scripts/evaluation`](scripts/evaluation/README.md) | Evaluating models, paired bootstrap and Holm correction |
| [`scripts/render`](scripts/render) | Figures and LaTeX tables of the paper |
| [`scripts/upload`](scripts/upload/README.md) | Staging and uploading datasets and models to the Hub |
| `src/tendril` | Shared library code: Hub access, MTEB tasks, models, LLM client, anonymization |
| `data` | Query maps, judge calibration results, annotation guidelines |
| `results` | Score files and significance tests of the evaluated systems |
| `examples` | Minimal usage examples |

Run the tests with `uv run pytest scripts`.

## License

The datasets are released under the [TENDRIL Open RAIL-D License](scripts/upload/licenses/OpenRAIL-D.md), which adds
use-based restrictions specific to procurement, such as no re-identification of persons. The fine-tuned models keep
the licenses of their base models (Apache-2.0 for Arctic Embed; CC-BY-NC-4.0, on top of OpenMDW-1.1, for Most Embed).
