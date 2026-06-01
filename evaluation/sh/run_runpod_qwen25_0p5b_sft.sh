#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT=/workspace/reasoning_vectors
VENV=/workspace/venvs/trl-h100
export PATH="$VENV/bin:$PATH"
DATA_DIR="$ROOT/training_data/math_sft_100k"
LOG="$ROOT/training_logs/sft_qwen25_0p5b_instruct_math100k_2ep_runpod.log"
CONFIG="$ROOT/evaluation/training_configs/sft_qwen25_0p5b_instruct_math100k_2ep_runpod.yaml"

mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets" "$DATA_DIR" "$ROOT/training_logs" "$ROOT/training_runs"
cd "$ROOT"

{
date "+%F %T %Z"
"$VENV/bin/hf" download Qwen/Qwen2.5-0.5B-Instruct --cache-dir "$HF_HOME/hub"
if [ ! -s "$DATA_DIR/train.jsonl" ] || [ "$(wc -l < "$DATA_DIR/train.jsonl" 2>/dev/null || echo 0)" != "100000" ]; then
  "$VENV/bin/python" evaluation/data_builders/build_math_sft_100k.py --output-dir "$DATA_DIR"
fi
"$VENV/bin/python" evaluation/data_builders/validate_math_sft_dataset.py "$DATA_DIR/train.jsonl"
"$VENV/bin/trl-kit" sft-launch --config "$CONFIG"
"$VENV/bin/trl-kit" sft-launch --config "$CONFIG" --execute
} 2>&1 | tee "$LOG"
