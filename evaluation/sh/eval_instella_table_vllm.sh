#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=${MODELS:-"amd/Instella-3B-Math-SFT amd/Instella-3B-Math amd/Instella-3B-Instruct"}
GPUS=${GPUS:-"0,1,2"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/instella_vllm"}
PROMPT_TYPE=${PROMPT_TYPE:-"cot"}
AIME_BENCHMARKS=${AIME_BENCHMARKS:-"aime25x8,amc23x8,aime24x8"}
MATH_BENCHMARKS=${MATH_BENCHMARKS:-"minerva_math,olympiadbench,math500"}
MAX_TOKENS_PER_CALL=${MAX_TOKENS_PER_CALL:-3072}
N_SAMPLING=${N_SAMPLING:-1}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-1}
OVERWRITE=${OVERWRITE:-0}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
BACKEND=${BACKEND:-"hf"}
USE_SAFETENSORS=${USE_SAFETENSORS:-1}
PREPARE_LOCAL_MODELS=${PREPARE_LOCAL_MODELS:-1}
MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-"models/instella_local"}

read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [ "${#GPU_LIST[@]}" -lt "${#MODEL_LIST[@]}" ]; then
    echo "Need at least one GPU id per model. Models=${#MODEL_LIST[@]} GPUs=${#GPU_LIST[@]}" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"

MODEL_PATHS=()
for model in "${MODEL_LIST[@]}"; do
    if [ "$PREPARE_LOCAL_MODELS" = "1" ]; then
        MODEL_PATHS+=("$(python3 prepare_instella_model.py --model "$model" --cache_root "$MODEL_CACHE_ROOT")")
    else
        MODEL_PATHS+=("$model")
    fi
done

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
    if [ "$BACKEND" = "vllm" ]; then
        extra_args+=(--use_vllm --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE")
    fi
    if [ "$USE_SAFETENSORS" = "1" ]; then
        extra_args+=(--use_safetensors)
    fi
    TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
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
        --save_outputs \
        --max_tokens_per_call "$MAX_TOKENS_PER_CALL" \
        "${extra_args[@]}"
}

run_model() {
    local model=$1
    local model_path=$2
    local gpu=$3
    local model_dir=${model//\//__}
    local output_dir="$OUTPUT_ROOT/$model_dir"
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_group "$model_path" "$output_dir" "$AIME_BENCHMARKS" 0.6
    run_group "$model_path" "$output_dir" "$MATH_BENCHMARKS" 0
}

pids=()

for index in "${!MODEL_LIST[@]}"; do
    model=${MODEL_LIST[$index]}
    model_path=${MODEL_PATHS[$index]}
    gpu=${GPU_LIST[$index]}
    model_dir=${model//\//__}
    log_path="$OUTPUT_ROOT/logs/$model_dir.log"
    (
        run_model "$model" "$model_path" "$gpu"
    ) >"$log_path" 2>&1 &
    pids+=("$!")
    echo "$model gpu=$gpu log=$log_path"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

python3 collect_benchmark_table.py \
    --output_root "$OUTPUT_ROOT" \
    --models "$MODELS" \
    --benchmarks "$AIME_BENCHMARKS,$MATH_BENCHMARKS" \
    --write "$OUTPUT_ROOT/results.md"

exit "$status"
