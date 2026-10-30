# Evaluation

Scripts to evaluate retrieval models on the TENDRIL test set and to test whether one model is significantly better than
another. Run all commands from the repository root. The test set is read from the Hugging Face Hub
(`IntelliProcure/TENDRIL`); API models read their keys from `.env`.

| Script | Purpose |
|---|---|
| `evaluate_models.py` | Scores one model or a group of models on every test project and aggregates per sector |
| `paired_bootstrap.py` | Paired bootstrap confidence interval for the score difference between two models |
| `holm_correction.py` | Holm correction of the p-values of several paired bootstrap comparisons |

## Evaluating models

Every test project is one MTEB retrieval task (`TENDRIL-<project>`). The script reports NDCG, Precision, Recall, F1,
MAP and MRR at K = 10, 25, 50 and 100, as means per sector and over all projects. Each project counts once, whatever its
number of queries. The overall row carries a 95% bootstrap interval over projects (10,000 iterations, seed 0).

```bash
# List the reviewed models, their groups and required API keys
uv run python scripts/evaluation/evaluate_models.py --list-models

# One model, or every model of a group (baseline, local, api, reference).
# oracle is another name for reference. BM25 is in baseline.
uv run python scripts/evaluation/evaluate_models.py --group baseline --output results/baseline
uv run python scripts/evaluation/evaluate_models.py --group local --output results/open-weight
uv run python scripts/evaluation/evaluate_models.py --group api --output results/proprietary
uv run python scripts/evaluation/evaluate_models.py --group reference --output results/oracle

# A fine-tuned SentenceTransformer directory, recorded under the open-weight id
uv run python scripts/evaluation/evaluate_models.py \
    --model path/to/fine-tuned-model \
    --model-name Snowflake/snowflake-arctic-embed-l-v2.0__TENDRIL \
    --force --output results/fine-tuned
```

| Option | Meaning |
|---|---|
| `--model-name NAME` | Name stored in the score JSON and prediction files, and used for their paths. A slash is written as `__`. For one `--model`, not a `--group` |
| `--output DIR` | Writes `{name with / as __}.json` with the scores, the intervals and the dataset revision |
| `--predictions DIR` | Writes the rankings to `DIR/{name with / as __}/TENDRIL-<project>_predictions.json` (top 100 per query) |
| `--revision REV` | Hub branch, tag or commit of the test set (default `main`). It is resolved to a commit and recorded in the output |
| `--no-cache` | Recomputes everything instead of reusing results from `cache/mteb/` |
| `--batch-size N` | Passages per forward pass (default 32). Lower it for long passages or small GPUs |
| `--force` | Skips the delay before evaluating a model that is not in the reviewed list |

**Cache.** MTEB stores one result file per project in `cache/mteb/results/<model>/<revision>/TENDRIL-<project>.json`,
and later runs reuse it. A cached result was computed on the test set as it was then. After a new release of the test
set, run with `--no-cache` or delete the model's cache directory. Tasks taken from the cache are not re-run and get no
predictions, so pass `--no-cache` together with `--predictions`.

## Comparing two models

`paired_bootstrap.py` computes a paired bootstrap confidence interval for B − A. In every iteration both models are
scored on the same resampled data, and the 2.5th and 97.5th percentiles of the differences form the 95% interval. The
statistic is the one that `evaluate_models.py` reports: the mean over projects of each project's mean over queries.

A model is given as a directory with either MTEB cache results or predictions:

- **Cache results** (`cache/mteb/results/<model>/<revision>`) contain one score per project. They support the project
  level only, but need no new run.
- **Predictions** (`--predictions` of `evaluate_models.py`) contain the rankings. The script computes per-query scores
  with `pytrec_eval`, exactly as MTEB does, so both resampling units are available.

```bash
uv run python scripts/evaluation/paired_bootstrap.py \
    cache/mteb/results/BAAI__bge-m3/<revision> \
    cache/mteb/results/<fine-tuned model>/<revision> \
    --metric ndcg_at_10 --metric recall_at_100 \
    --revision <dataset revision from the score JSON> \
    --output results/bootstrap/bge-m3.json
```

| Option | Meaning |
|---|---|
| `--metric KEY` | MTEB metric key, repeatable: `ndcg_at_k`, `recall_at_k`, `precision_at_k`, `map_at_k`, `mrr_at_k` (default `ndcg_at_10`) |
| `--unit project` | Default. Cluster bootstrap that draws whole projects with replacement |
| `--unit query` | Draws queries with replacement within each project; the set of projects stays fixed. Needs predictions for both models |
| `--iterations N`, `--seed S` | Default 10,000 iterations and seed 0 |
| `--revision REV` | Test set revision for the qrels. Use the `dataset.revision` recorded in the score JSON of the evaluation |
| `--output FILE` | Also writes the results as JSON |

The output lists the means of A and B, the difference, its 95% interval, and two p-values. `P(B<=A)` is the share of
iterations in which B is not better than A and serves as a one-sided p-value; `p (2-sided)` doubles the smaller tail.
A p-value of `<1e-04` (one-sided) or `<2e-04` (two-sided) means that no iteration crossed zero; the resolution is
1 / iterations per tail.

```
metric                   A       B    B - A   95% CI               P(B<=A)  p (2-sided)
ndcg_at_10          0.3955  0.4648  +0.0693   [+0.0175, +0.1163]    0.0058       0.0116
```

**Which unit to report.** Queries of one project share documents, annotators and difficulty, so they are not
independent. The project-level interval accounts for that and describes how a difference generalizes to new tender
projects; use it as the main result. The query-level interval only reflects query sampling within the given projects
and is narrower.

## Correcting for multiple comparisons

Each additional test raises the chance that one of them is significant by accident. `holm_correction.py` reads the JSON
files of several `paired_bootstrap.py` runs and adjusts their two-sided p-values with the Holm step-down procedure,
which keeps the probability of any false positive within the family at `--alpha`.

```bash
# Each fine-tuned model against its base model, corrected over the four models per metric
uv run python scripts/evaluation/holm_correction.py results/bootstrap/*__TENDRIL.json --output results/bootstrap/holm.json
```

| Option | Meaning |
|---|---|
| `--family metric` | Default. Each metric is its own family, e.g. the four fine-tuning comparisons for nDCG@10 |
| `--family all` | One family over all comparisons and metrics; stricter |
| `--alpha A` | Significance level for the last column (default 0.05) |
| `--output FILE` | Also writes the adjusted p-values as JSON |

Pass only the comparisons of one research question, and state the family in the paper ("Holm-corrected over the four
models per metric"). A two-sided bootstrap p-value of 0 is treated as its bound 2 / iterations, and adjusted values
derived from it are printed as upper bounds (`<0.0008` for four comparisons and 10,000 iterations).
