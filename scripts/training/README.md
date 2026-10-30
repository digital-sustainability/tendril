# Embedding fine-tuning

Scripts to prepare judged query-passage pairs and fine-tune embedding models with an explicit-negative contrastive
loss. Run all commands from the repository root. Data and runs live in `data/derived/` and `training_runs/`, which are
git-ignored; never commit them.

```bash
uv sync --group training
```

## Inputs

The judge release contains 631 approved projects (911,540 verdicts: 839,805 score 0, 53,201 score 1, 18,534 score 2).
Projects with restricted documents are excluded and must not be reintroduced. Verdicts use
`(query_id, document_id, score)` with canonical query IDs such as `Construction_Area::HF-001`; `document_id` is the
`passage_uid` of `passages_anon.parquet`. Only the anonymized passages are read. Place `pairs.parquet`,
`queries.parquet` and `passages_anon.parquet` in `data/derived/judge/`.

## Pipeline

1. **Join verdicts** (`prepare_judgments.py`). Rejects duplicate or unknown IDs and scores outside 0/1/2. Pairs without a
   verdict (52 in 38 projects) are excluded with `--allow-missing-judgments` and written to `labeled.missing.parquet`;
   they are never treated as score 0.

   ```bash
   python scripts/training/prepare_judgments.py --judgments data/derived/judge/relevance_judgments.parquet \
       --output data/derived/training/labeled.parquet --expected-projects 631 --allow-missing-judgments
   ```

2. **Split projects** (`prepare_examples.py`). Whole projects are split (568 training, 63 validation, seed 42), so no
   passage crosses the split. `--cover-catalogs` keeps every query catalog in both parts; this is not a sector
   stratification. The split file is never overwritten.

   ```bash
   python scripts/training/prepare_examples.py --reserve-from-labeled data/derived/training/labeled.parquet \
       --split data/derived/training/split.json --cover-catalogs
   ```

3. **Build examples** (`prepare_examples.py`). One row per query with a score-2 positive (not from the `unsure` tier)
   and three distinct score-0 negatives for the same query and project, two from the positive's document and one from
   another. Score 1 is not used. Rows with contradictory labels, exact text overlap with validation, or too few
   negatives are dropped. Validation queries not judged against every passage of their project are excluded, not filled
   with negatives. Result: 13,390 examples (8,890 area, 4,500 criterion) and 103,858 validation pairs.

   ```bash
   python scripts/training/prepare_examples.py --split data/derived/training/split.json \
       --labeled data/derived/training/labeled.parquet --output-dir data/derived/training/neg3
   ```

4. **Optional: revised query texts** (`apply_query_variant.py`). Replaces only the query text, joined on
   `(project_id, canonical_query_id)`, and fails on any unmapped key or level mismatch.

   ```bash
   python scripts/training/apply_query_variant.py --examples data/derived/training/neg3/examples.parquet \
       --validation data/derived/training/neg3/validation.parquet --modified-queries queries_modified.parquet \
       --output-dir data/derived/training/neg3_modified
   ```

5. **Train** (`train_model.py`, one CUDA GPU; select it with `CUDA_VISIBLE_DEVICES`). Every run needs a fresh
   `--output-dir`.

   ```bash
   python scripts/training/train_model.py --mode plan  --data-dir DATA --output-dir OUT --model BAAI/bge-m3
   python scripts/training/train_model.py --mode pilot --data-dir DATA --output-dir OUT --learning-rate 1e-5
   python scripts/training/train_model.py --mode sweep --data-dir DATA --output-dir OUT [--resume]
   python scripts/training/train_model.py --mode final --data-dir DATA --output-dir OUT \
       --learning-rate 5e-5 --effective-batch 32
   ```

## Training and evaluation

- **Loss:** cross-entropy over cosine scores (scale 20) of `[positive, own negatives]`. Other rows in a batch are never
  negatives, so repeated queries in a batch are safe and gradient accumulation is equivalent to a larger batch.
- **Models:** `BAAI/bge-m3`, `Snowflake/snowflake-arctic-embed-l-v2.0`, `nvidia/Nemotron-3-Embed-1B-BF16` (needs
  `--trust-remote-code`), `malteos/most-embed-de`. Query and passage prefixes are fixed per model and identical in
  training and evaluation.
- **Data balance:** the majority query level is downsampled so both levels contribute equally.
- **Sweep:** three model-specific learning rates times effective batches 32/64/128, all on the same example budget.
- **Final:** up to two epochs, evaluation every fifth of an epoch, early stopping with patience 3.
- **Metric:** linear-gain nDCG@10 over all judged passages of a project, averaged per project and then over projects,
  separately per query level. The balanced score is their mean and selects checkpoints. Queries without any relevant
  passage are excluded and counted in the run summary.
- **Output:** `plan.json`, `results.json`, per-run `run.json` and `run_XX/best/`.

Validation scores select hyperparameters and checkpoints; they are not test results. A test evaluation must use the
frozen checkpoints once and be reported separately.
