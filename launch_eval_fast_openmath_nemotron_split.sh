#!/usr/bin/env bash
set -euo pipefail
cd "/home/zbibm/reasoning vectors/evaluation"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate trl-cuda121
MODEL="/home/zbibm/eval_models_14b/fast_openmath_nemotron_14b"
OUT="outputs/fast_openmath_nemotron_14b_vllm_split/__home__zbibm__eval_models_14b__fast_openmath_nemotron_14b"
mkdir -p "$OUT" outputs/fast_openmath_nemotron_14b_vllm_split/logs
run_group() {
    local gpu=$1
    local datasets=$2
    local temperature=$3
    local log=$4
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
        --model_name_or_path "$MODEL" \
        --data_names "$datasets" \
        --output_dir "$OUT" \
        --split test \
        --prompt_type cot \
        --num_test_sample -1 \
        --seed 0 \
        --temperature "$temperature" \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call 3072 \
        --pipeline_parallel_size 1 \
        --vllm_batch_size 8 \
        --vllm_gpu_memory_utilization 0.9 \
        --apply_chat_template \
        --vllm_max_model_len 4096 >"outputs/fast_openmath_nemotron_14b_vllm_split/logs/$log" 2>&1
}
run_group 0 "aime25x8,amc23x8,aime24x8" 0.6 "aime.log" &
run_group 1 "minerva_math,math500" 0 "minerva_math500.log" &
run_group 2 "olympiadbench" 0 "olympiad.log" &
wait
python3 collect_benchmark_table.py \
    --output_root outputs/fast_openmath_nemotron_14b_vllm_split \
    --models "$MODEL" \
    --benchmarks "aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500" \
    --write outputs/fast_openmath_nemotron_14b_vllm_split/results.md
