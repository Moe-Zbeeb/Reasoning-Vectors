#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121

run_family() {
    local name="$1"
    local models="$2"
    local output_root="$3"
    local batch_size="$4"
    echo "start-${name} $(date +%F_%T_%Z)"
    MODELS="$models" \
    GPUS="${GPUS:-0,1,2}" \
    OUTPUT_ROOT="$output_root" \
    MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-3072}" \
    VLLM_BATCH_SIZE="$batch_size" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
    OVERWRITE="${OVERWRITE:-1}" \
    bash sh/eval_models_multi_gpu_vllm.sh
    local status="$?"
    echo "end-${name} status=${status} $(date +%F_%T_%Z)"
    return "$status"
}

status=0

run_family \
    "qwen25-math-15b" \
    "Qwen/Qwen2.5-Math-1.5B michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO" \
    "outputs/qwen25_math_15b_family_vllm" \
    "32" || status="$?"

run_family \
    "olmo2-math" \
    "allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT sunblaze-ucb/OLMo-2-7B-SFT-GRPO-MATH-1EPOCH" \
    "outputs/olmo2_math_family_vllm" \
    "16" || {
        olmo_status="$?"
        if [ "$status" -eq 0 ]; then
            status="$olmo_status"
        fi
    }

exit "$status"
