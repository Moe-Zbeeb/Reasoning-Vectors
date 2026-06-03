#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/reasoning_vectors_ablation
TASK_FILE=$ROOT/Reasoning-Vectors/runpod_lighteval_ablation_tasks.py
BUILDER=$ROOT/Reasoning-Vectors/runpod_build_random_vector_model.py
RUN_ID=${1:-random_vector_deepseek_qwen_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$ROOT/eval_runs/$RUN_ID
CONFIG_DIR=$ROOT/configs/$RUN_ID
LOG_DIR=$ROOT/logs/$RUN_ID
DEEPSEEK_RANDOM=$ROOT/merged/random_delta_alpha1/deepseek_fast_math_r1_14b
QWEN_RANDOM=$ROOT/merged/random_delta_alpha1/qwen_gsm8k_1p5b

cd "$ROOT"
. .venv_eval/bin/activate

export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" "$CONFIG_DIR" "$LOG_DIR" "$ROOT/merged/random_delta_alpha1"

if [ "${REBUILD_RANDOM:-0}" = "1" ] || [ ! -f "$DEEPSEEK_RANDOM/config.json" ]; then
    python "$BUILDER" \
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --sft zbeeb/deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep \
        --rl RabotniKuma/Fast-Math-R1-14B \
        --out "$DEEPSEEK_RANDOM" \
        --seed 0 \
        --alpha 1.0 \
        > "$LOG_DIR/build_deepseek_random.log" 2>&1
fi

if [ "${REBUILD_RANDOM:-0}" = "1" ] || [ ! -f "$QWEN_RANDOM/config.json" ]; then
    python "$BUILDER" \
        --base Qwen/Qwen2.5-Math-1.5B \
        --sft michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT \
        --rl michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO \
        --out "$QWEN_RANDOM" \
        --seed 0 \
        --alpha 1.0 \
        > "$LOG_DIR/build_qwen_random.log" 2>&1
fi

write_config() {
    local label=$1
    local model=$2
    local max_model_length=$3
    local max_num_seqs=$4
    local max_num_batched_tokens=$5
    local max_new_tokens=$6
    local output=$CONFIG_DIR/$label.yaml
    cat > "$output" <<YAML
model_parameters:
  model_name: "$model"
  dtype: bfloat16
  tensor_parallel_size: 1
  data_parallel_size: 1
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true
  max_model_length: $max_model_length
  swap_space: 4
  seed: 0
  trust_remote_code: true
  add_special_tokens: true
  max_num_seqs: $max_num_seqs
  max_num_batched_tokens: $max_num_batched_tokens
  generation_parameters:
    max_new_tokens: $max_new_tokens
    temperature: 0
    top_p: 1.0
    seed: 0
YAML
}

run_eval() {
    local label=$1
    local gpu=$2
    local config=$3
    local tasks=$4
    CUDA_VISIBLE_DEVICES="$gpu" lighteval vllm "$config" "$tasks" \
        --custom-tasks "$TASK_FILE" \
        --output-dir "$RUN_DIR/$label" \
        > "$LOG_DIR/$label.log" 2>&1 &
    PIDS[$label]=$!
    printf '%s %s %s\n' "$label" "$gpu" "${PIDS[$label]}" >> "$LOG_DIR/pids.tsv"
}

wait_group() {
    local status=0
    local label
    for label in "$@"; do
        if ! wait "${PIDS[$label]}"; then
            status=1
        fi
    done
    return "$status"
}

run_aime_set() {
    local prefix=$1
    local config=$2
    declare -gA PIDS=()
    : > "$LOG_DIR/pids.tsv"
    run_eval "${prefix}_aime24_s0" 0 "$config" "ablation:aime24x8_s0|0"
    run_eval "${prefix}_aime24_s1" 1 "$config" "ablation:aime24x8_s1|0"
    run_eval "${prefix}_aime24_s2" 2 "$config" "ablation:aime24x8_s2|0"
    run_eval "${prefix}_aime24_s3" 3 "$config" "ablation:aime24x8_s3|0"
    run_eval "${prefix}_aime25_s0" 4 "$config" "ablation:aime25x8_s0|0"
    run_eval "${prefix}_aime25_s1" 5 "$config" "ablation:aime25x8_s1|0"
    run_eval "${prefix}_aime25_s2" 6 "$config" "ablation:aime25x8_s2|0"
    run_eval "${prefix}_aime25_s3" 7 "$config" "ablation:aime25x8_s3|0"
    wait_group "${prefix}_aime24_s0" "${prefix}_aime24_s1" "${prefix}_aime24_s2" "${prefix}_aime24_s3" "${prefix}_aime25_s0" "${prefix}_aime25_s1" "${prefix}_aime25_s2" "${prefix}_aime25_s3"
}

write_config deepseek_aime "$DEEPSEEK_RANDOM" 32768 16 32768 8000
write_config deepseek_math500 "$DEEPSEEK_RANDOM" 32768 32 32768 3000
write_config qwen_aime "$QWEN_RANDOM" 12288 64 65536 8000
write_config qwen_math500 "$QWEN_RANDOM" 4096 256 65536 3000

run_aime_set deepseek_random "$CONFIG_DIR/deepseek_aime.yaml"
run_aime_set qwen_random "$CONFIG_DIR/qwen_aime.yaml"

declare -A PIDS=()
run_eval deepseek_random_math500 0 "$CONFIG_DIR/deepseek_math500.yaml" "ablation:math500|0"
run_eval qwen_random_math500 1 "$CONFIG_DIR/qwen_math500.yaml" "ablation:math500|0"
wait_group deepseek_random_math500 qwen_random_math500

printf '%s\n' "$RUN_DIR" > "$LOG_DIR/run_dir.txt"
