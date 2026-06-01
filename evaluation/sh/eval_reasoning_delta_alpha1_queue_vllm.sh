#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121

MERGE_ROOT="${MERGE_ROOT:-$HOME/reasoning vectors/merged_checkpoints/reasoning_delta_alpha1}"
LINK_ROOT="${LINK_ROOT:-$HOME/reasoning_delta_alpha1_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reasoning_delta_alpha1_vllm}"
PROMPT_TYPE="${PROMPT_TYPE:-cot}"
AIME_BENCHMARKS="${AIME_BENCHMARKS:-aime25x8,amc23x8,aime24x8}"
MATH_BENCHMARKS="${MATH_BENCHMARKS:-minerva_math,olympiadbench,math500}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-3072}"
N_SAMPLING="${N_SAMPLING:-1}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:--1}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
OVERWRITE="${OVERWRITE:-1}"

MODEL_KEYS=(
    primeintellect_7b
    sii_enigma_7b
    olmo2_7b
    roi_qwen25_7b
    qwen_gsm8k_1p5b
    specreasoner_1p5b
)

mkdir -p "$LINK_ROOT" "$OUTPUT_ROOT/logs"

for key in "${MODEL_KEYS[@]}"; do
    ln -sfn "$MERGE_ROOT/$key" "$LINK_ROOT/$key"
done

MODEL_PATHS=()
for key in "${MODEL_KEYS[@]}"; do
    MODEL_PATHS+=("$LINK_ROOT/$key")
done

MODEL_ARG="${MODEL_PATHS[*]}"
BENCHMARK_ARG="$AIME_BENCHMARKS,$MATH_BENCHMARKS"
QUEUE_FILE="$OUTPUT_ROOT/queue.txt"
LOCK_FILE="$OUTPUT_ROOT/queue.lock"
COLLECT_LOCK="$OUTPUT_ROOT/collect.lock"
STATUS_FILE="$OUTPUT_ROOT/status.tsv"

printf "%s\n" "${MODEL_PATHS[@]}" > "$QUEUE_FILE"
printf "time\tgpu\tmodel\tstatus\n" > "$STATUS_FILE"

safe_model_name() {
    local model="$1"
    echo "${model//\//__}"
}

collect_results() {
    python3 collect_benchmark_table.py \
        --output_root "$OUTPUT_ROOT" \
        --models "$MODEL_ARG" \
        --benchmarks "$BENCHMARK_ARG" \
        --write "$OUTPUT_ROOT/results.md" >/dev/null
}

run_group() {
    local gpu="$1"
    local model="$2"
    local output_dir="$3"
    local datasets="$4"
    local temperature="$5"
    local extra_args=()
    if [ "$OVERWRITE" = "1" ]; then
        extra_args+=(--overwrite)
    fi
    if [ "$VLLM_MAX_MODEL_LEN" != "0" ]; then
        extra_args+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
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
        --pipeline_parallel_size 1 \
        --vllm_batch_size "$VLLM_BATCH_SIZE" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --apply_chat_template \
        "${extra_args[@]}"
}

run_model() {
    local gpu="$1"
    local model="$2"
    local model_dir
    model_dir="$(safe_model_name "$model")"
    local output_dir="$OUTPUT_ROOT/$model_dir"
    local log_path="$OUTPUT_ROOT/logs/$model_dir.log"
    echo "start gpu=$gpu model=$model time=$(date +%F_%T_%Z)"
    {
        echo "start gpu=$gpu model=$model time=$(date +%F_%T_%Z)"
        run_group "$gpu" "$model" "$output_dir" "$AIME_BENCHMARKS" 0.6
        run_group "$gpu" "$model" "$output_dir" "$MATH_BENCHMARKS" 0
        echo "done gpu=$gpu model=$model time=$(date +%F_%T_%Z)"
    } >"$log_path" 2>&1
}

claim_next() {
    CLAIMED_MODEL=""
    {
        flock -x 200
        if [ -s "$QUEUE_FILE" ]; then
            CLAIMED_MODEL="$(head -n 1 "$QUEUE_FILE")"
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
            mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        fi
    } 200>"$LOCK_FILE"
    [ -n "$CLAIMED_MODEL" ]
}

record_status() {
    local gpu="$1"
    local model="$2"
    local status="$3"
    {
        flock -x 201
        printf "%s\t%s\t%s\t%s\n" "$(date +%F_%T_%Z)" "$gpu" "$model" "$status" >> "$STATUS_FILE"
        collect_results
    } 201>"$COLLECT_LOCK"
}

worker() {
    local gpu="$1"
    local worker_status=0
    while claim_next; do
        local model="$CLAIMED_MODEL"
        if run_model "$gpu" "$model"; then
            record_status "$gpu" "$model" done
        else
            record_status "$gpu" "$model" failed
            worker_status=1
        fi
    done
    return "$worker_status"
}

pids=()
for gpu in 0 1 2 3; do
    worker "$gpu" > "$OUTPUT_ROOT/logs/worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

collect_results
exit "$status"
