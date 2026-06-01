#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121

MODELS="allenai/OLMo-2-1124-7B" \
GPU=3 \
OUTPUT_ROOT="outputs/olmo2_math_family_vllm" \
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-3072}" \
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-16}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
OVERWRITE="${OVERWRITE:-1}" \
bash sh/eval_models_one_gpu_vllm.sh

python3 collect_benchmark_table.py \
    --output_root "outputs/olmo2_math_family_vllm" \
    --models "allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT sunblaze-ucb/OLMo-2-7B-SFT-GRPO-MATH-1EPOCH" \
    --write "outputs/olmo2_math_family_vllm/results_all.md"
