# LLM-as-a-judge

Two stages share one runner and one analysis script. Names distinguish them.

1. **Model and reasoning effort.** The same German guidelines prompt, every listed model at low and medium reasoning effort, on two area-of-action wordings:
   - `configs/model_reasoning_simple_aoa.yaml` — original area wording. Results in `data/judge/model_reasoning/simple_aoa/`.
   - `configs/model_reasoning_summary_aoa.yaml` — LLM-summary area wording. Results in `data/judge/model_reasoning/summary_aoa/`.
2. **Prompt length.** `z-ai/glm-5.3-flash` at low effort on the LLM-summary area wording, with the two shorter prompts:
   - `configs/prompt_length_shorter.yaml` — `prompts/guidelines_de_shorter.txt`. Results in `data/judge/prompt_length/shorter/`.
   - `configs/prompt_length_shortest.yaml` — `prompts/guidelines_de_shortest.txt`. Results in `data/judge/prompt_length/shortest/`.

The full prompt is `prompts/guidelines_de.txt` (prompt id `guidelines_de`). It asks for `{"Bewertung": 0|1|2}`.

The client lives in `tendril.inference` (`OpenRouterClient`).

The calibration sample that the configs read (`data/validation_set_curated.xlsx`) is not part of this repository,
because its passages are not anonymized. The analysis results of all runs are in `data/judge/`.

## Run

From the repository root, with `OPENROUTER_API_KEY` set:

```bash
python scripts/judge/run_labeling.py --dry-run
python scripts/judge/run_labeling.py --max-samples 6
python scripts/judge/run_labeling.py --config scripts/judge/configs/model_reasoning_summary_aoa.yaml \
    --results-dir data/judge/model_reasoning/summary_aoa
python scripts/judge/analysis.py --config scripts/judge/configs/model_reasoning_summary_aoa.yaml
```

`--results-dir` is relative to the repository root. Without it, a run writes a fresh `data/judge/<config stem>_<UTC timestamp>/` directory. Pass the same directory to resume. `analysis.py` reads `paths.results_dir` from the config when `--results` is omitted. What it writes and which agreement metrics it computes is described in the module docstring of `analysis.py`.

Each results directory holds `results.jsonl` (the checkpoint), `results.xlsx` (an Excel mirror), `manifest.json`, the metric CSVs, `run.log`, and `responsible_request.sqlite` (request and response traces).

## Labeling TENDRIL-Train

`run_labeling_parquet.py` labels the selected training pairs (see [`scripts/train_set`](../train_set/README.md)) with
the shortest prompt and `z-ai/glm-5.3-flash` at low effort, and writes `relevance_judgments.parquet`.

## Tests

```bash
uv run pytest scripts/judge
```
