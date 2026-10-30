#!/usr/bin/env bash
# Re-run TENDRIL evaluations into the results/ directory layout.
# BM25 -> results/baseline
# open-weight models, including BGE-M3 -> results/open-weight
# fine-tuned checkpoints -> results/fine-tuned
# API models -> results/proprietary
# oracle (reference group) -> results/oracle
#
# Set the directories of the four fine-tuned models, and optionally the GPU to use:
#   ARCTIC_MODEL=... BGE_M3_MODEL=... NEMOTRON_MODEL=... MOST_EMBED_MODEL=... [GPU=0] \
#     scripts/evaluation/rerun_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${ARCTIC_MODEL:?set ARCTIC_MODEL to the fine-tuned Arctic Embed directory}"
: "${BGE_M3_MODEL:?set BGE_M3_MODEL to the fine-tuned BGE-M3 directory}"
: "${NEMOTRON_MODEL:?set NEMOTRON_MODEL to the fine-tuned Nemotron directory}"
: "${MOST_EMBED_MODEL:?set MOST_EMBED_MODEL to the fine-tuned Most Embed directory}"
mkdir -p logs results/paired-bootstrap

echo "[$(date -Is)] groups"
uv run python -c "
from tendril.models import MODEL_GROUPS, models_in_group
for group in MODEL_GROUPS:
    print(group, [model.name for model in models_in_group(group)])
"

REV=$(uv run python -c "from tendril.hub import TEST_REPO, resolve_revision; print(resolve_revision(TEST_REPO, 'main'))")
echo "[$(date -Is)] dataset revision ${REV}"

COMMON=(--no-cache --predictions results/predictions --revision "${REV}")

run_cpu() {
  echo "[$(date -Is)] reference/oracle"
  CUDA_VISIBLE_DEVICES= uv run python scripts/evaluation/evaluate_models.py \
    --group reference "${COMMON[@]}" --output results/oracle
  echo "[$(date -Is)] api/proprietary"
  CUDA_VISIBLE_DEVICES= uv run python scripts/evaluation/evaluate_models.py \
    --group api "${COMMON[@]}" --output results/proprietary
  echo "[$(date -Is)] cpu jobs done"
}

run_gpu() {
  export CUDA_VISIBLE_DEVICES="${GPU:-0}"
  echo "[$(date -Is)] baseline (BM25)"
  uv run python scripts/evaluation/evaluate_models.py \
    --group baseline "${COMMON[@]}" --output results/baseline
  echo "[$(date -Is)] local/open-weight"
  uv run python scripts/evaluation/evaluate_models.py \
    --group local "${COMMON[@]}" --batch-size 1 --output results/open-weight

  echo "[$(date -Is)] fine-tuned arctic"
  uv run python scripts/evaluation/evaluate_models.py \
    --model "${ARCTIC_MODEL}" \
    --model-name Snowflake/snowflake-arctic-embed-l-v2.0__TENDRIL \
    --force "${COMMON[@]}" --batch-size 1 --output results/fine-tuned
  echo "[$(date -Is)] fine-tuned bge-m3"
  uv run python scripts/evaluation/evaluate_models.py \
    --model "${BGE_M3_MODEL}" \
    --model-name BAAI/bge-m3__TENDRIL \
    --force "${COMMON[@]}" --batch-size 1 --output results/fine-tuned
  echo "[$(date -Is)] fine-tuned nemotron"
  uv run python scripts/evaluation/evaluate_models.py \
    --model "${NEMOTRON_MODEL}" \
    --model-name nvidia/Nemotron-3-Embed-1B-BF16__TENDRIL \
    --force "${COMMON[@]}" --batch-size 1 --output results/fine-tuned
  echo "[$(date -Is)] fine-tuned most-embed-de"
  uv run python scripts/evaluation/evaluate_models.py \
    --model "${MOST_EMBED_MODEL}" \
    --model-name malteos/most-embed-de__TENDRIL \
    --force "${COMMON[@]}" --batch-size 1 --output results/fine-tuned
  echo "[$(date -Is)] gpu jobs done"
}

run_cpu > logs/rerun-cpu.log 2>&1 &
cpu_pid=$!
run_gpu > logs/rerun-gpu.log 2>&1 &
gpu_pid=$!
echo "[$(date -Is)] started cpu pid ${cpu_pid} gpu pid ${gpu_pid}"

set +e
wait "${cpu_pid}"
cpu_status=$?
wait "${gpu_pid}"
gpu_status=$?
set -e
echo "[$(date -Is)] cpu status ${cpu_status} gpu status ${gpu_status}"
if [[ "${cpu_status}" -ne 0 || "${gpu_status}" -ne 0 ]]; then
  exit 1
fi

METRICS=(--metric ndcg_at_10 --metric precision_at_10 --metric recall_at_10 --metric map_at_10 --metric mrr_at_10)
boot() {
  local system_a=$1 system_b=$2 output=$3
  echo "[$(date -Is)] bootstrap ${system_a} -> ${system_b}"
  uv run python scripts/evaluation/paired_bootstrap.py \
    "results/predictions/${system_a}" "results/predictions/${system_b}" \
    "${METRICS[@]}" --unit project --iterations 10000 --seed 0 \
    --revision "${REV}" --output "results/paired-bootstrap/${output}"
}

boot BAAI__bge-m3 BAAI__bge-m3__TENDRIL BAAI__bge-m3__TENDRIL.json
boot nvidia__Nemotron-3-Embed-1B-BF16 nvidia__Nemotron-3-Embed-1B-BF16__TENDRIL nvidia__Nemotron-3-Embed-1B-BF16__TENDRIL.json
boot Snowflake__snowflake-arctic-embed-l-v2.0 Snowflake__snowflake-arctic-embed-l-v2.0__TENDRIL Snowflake__snowflake-arctic-embed-l-v2.0__TENDRIL.json
boot malteos__most-embed-de malteos__most-embed-de__TENDRIL malteos__most-embed-de__TENDRIL.json
boot nvidia__Nemotron-3-Embed-1B-BF16 malteos__most-embed-de nemotron_vs_most-embed-de.json

echo "[$(date -Is)] holm"
uv run python scripts/evaluation/holm_correction.py \
  results/paired-bootstrap/BAAI__bge-m3__TENDRIL.json \
  results/paired-bootstrap/malteos__most-embed-de__TENDRIL.json \
  results/paired-bootstrap/nvidia__Nemotron-3-Embed-1B-BF16__TENDRIL.json \
  results/paired-bootstrap/Snowflake__snowflake-arctic-embed-l-v2.0__TENDRIL.json \
  --output results/paired-bootstrap/holm.json

echo "[$(date -Is)] DONE"
