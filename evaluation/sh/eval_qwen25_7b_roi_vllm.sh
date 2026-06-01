#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121

MODELS="Qwen/Qwen2.5-7B-Instruct chrisluo5311/Qwen2.5-7B-Instruct-SFT-MetaMath-Merged-ROI chrisluo5311/Qwen2.5-7B-Instruct-SFT-GRPO-Merged-ROI" \
GPUS="0,1,2" \
OUTPUT_ROOT="outputs/qwen25_7b_instruct_roi_vllm" \
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-3072}" \
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-16}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}" \
OVERWRITE="${OVERWRITE:-1}" \
bash sh/eval_models_multi_gpu_vllm.sh
