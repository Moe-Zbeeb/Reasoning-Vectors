#!/usr/bin/env bash
set -euo pipefail
cd "/home/zbibm/reasoning vectors"
mkdir -p /home/zbibm/eval_models_14b
ln -sfn "/home/zbibm/reasoning vectors/models/RabotniKuma__Fast-OpenMath-Nemotron-14B" /home/zbibm/eval_models_14b/fast_openmath_nemotron_14b
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate trl-cuda121
MODELS="/home/zbibm/eval_models_14b/fast_openmath_nemotron_14b" \
GPUS="0" \
OUTPUT_ROOT=outputs/fast_openmath_nemotron_14b_vllm \
VLLM_BATCH_SIZE=8 \
VLLM_MAX_MODEL_LEN=4096 \
MAX_TOKENS_PER_CALL=3072 \
VLLM_GPU_MEMORY_UTILIZATION=0.9 \
bash evaluation/sh/eval_models_multi_gpu_vllm.sh
