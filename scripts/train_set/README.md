# Training set (TENDRIL-Train): selecting what the judge labels

Scripts that select the query-passage pairs of TENDRIL-Train before they are labeled by the LLM judge. Run all
commands from the repository root. Both scripts read `config.yaml` in this directory; inputs and outputs are under
`data/derived/judge/` (git-ignored).

1. **Select training units** (`select_training_units.py`): (project, document, area of action, criterion) tuples, per
   project with a catalog gate, a stratified area quota, and criteria and documents ranked by retrieval statistics.

   ```bash
   uv run python scripts/train_set/select_training_units.py --config scripts/train_set/config.yaml
   ```

2. **Build the judge input** (`build_judge_input.py`): `passages.parquet`, `pairs.parquet` and `queries.parquet`.

   ```bash
   uv run python scripts/train_set/build_judge_input.py --units-dir data/derived/judge
   ```

3. **Anonymize the passages** with `scripts/anonymize/anonymize_parquet.py` (column `text`), which gives
   `passages_anon.parquet`.

4. **Label the pairs** with `scripts/judge/run_labeling_parquet.py --root data/derived/judge` (see
   [the judge README](../judge/README.md)); it writes `relevance_judgments.parquet`.

5. **Prepare training examples and fine-tune** with the scripts in [`scripts/training`](../training/README.md).
