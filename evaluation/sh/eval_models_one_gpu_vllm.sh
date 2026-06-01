#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=${MODELS:?MODELS is required}
GPU=${GPU:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PROMPT_TYPE=${PROMPT_TYPE:-"cot"}
AIME_BENCHMARKS=${AIME_BENCHMARKS:-"aime25x8,amc23x8,aime24x8"}
MATH_BENCHMARKS=${MATH_BENCHMARKS:-"minerva_math,olympiadbench,math500"}
MAX_TOKENS_PER_CALL=${MAX_TOKENS_PER_CALL:-3072}
N_SAMPLING=${N_SAMPLING:-1}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-1}
OVERWRITE=${OVERWRITE:-0}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE:-32}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}

read -r -a MODEL_LIST <<< "$MODELS"
mkdir -p "$OUTPUT_ROOT/logs"

run_group() {
    local model=$1
    local output_dir=$2
    local datasets=$3
    local temperature=$4
    local extra_args=()
    if [ "$APPLY_CHAT_TEMPLATE" = "1" ]; then
        extra_args+=(--apply_chat_template)
    fi
    if [ "$OVERWRITE" = "1" ]; then
        extra_args+=(--overwrite)
    fi
    if [ "$VLLM_MAX_MODEL_LEN" != "0" ]; then
        extra_args+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
        --model_name_or_path "$model" \
        --data_names "$datasets" \
        --output_dir "$output_dir" \
        --split test \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample "$NUM_TEST_SAMPLE" \
        --seed 0 \
        --temperature "$temperature" \
        --n_sampling "$N_SAMPLING" \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call "$MAX_TOKENS_PER_CALL" \
        --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
        --vllm_batch_size "$VLLM_BATCH_SIZE" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        "${extra_args[@]}"
}

for model in "${MODEL_LIST[@]}"; do
    model_dir=${model//\//__}
    output_dir="$OUTPUT_ROOT/$model_dir"
    log_path="$OUTPUT_ROOT/logs/$model_dir.log"
    echo "$model gpu=$GPU log=$log_path"
    run_group "$model" "$output_dir" "$AIME_BENCHMARKS" 0.6 >"$log_path" 2>&1
    run_group "$model" "$output_dir" "$MATH_BENCHMARKS" 0 >>"$log_path" 2>&1
    python3 collect_benchmark_table.py \
        --output_root "$OUTPUT_ROOT" \
        --models "$MODELS" \
        --benchmarks "$AIME_BENCHMARKS,$MATH_BENCHMARKS" \
        --write "$OUTPUT_ROOT/results.md"
done

python3 collect_benchmark_table.py \
    --output_root "$OUTPUT_ROOT" \
    --models "$MODELS" \
    --benchmarks "$AIME_BENCHMARKS,$MATH_BENCHMARKS" \
    --write "$OUTPUT_ROOT/results.md"
