#!/usr/bin/env bash
set -euo pipefail

PROMPT_TYPE=${1:-qwen25-math-cot}
MODEL_NAME_OR_PATH=${2:?model path or Hugging Face model id required}
OUTPUT_DIR=${3:-outputs/reasoning_vllm}
DATA_NAMES=${4:-aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500}
MAX_TOKENS_PER_CALL=${5:-3072}
TEMPERATURE=${6:-0}
N_SAMPLING=${7:-1}

cd "$(dirname "$0")/.."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_names "$DATA_NAMES" \
    --output_dir "$OUTPUT_DIR" \
    --split test \
    --prompt_type "$PROMPT_TYPE" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature "$TEMPERATURE" \
    --n_sampling "$N_SAMPLING" \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --max_tokens_per_call "$MAX_TOKENS_PER_CALL" \
    --overwrite
