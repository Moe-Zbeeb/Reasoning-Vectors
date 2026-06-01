#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=${MODELS:-"IIGroup/X-Coder-SFT-Qwen3-8B IIGroup/X-Coder-RL-Qwen3-8B Qwen/Qwen3-8B-Base"}
GPU=${GPU:-3}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/xcoder_humaneval_gpu3"}
MAX_TOKENS=${MAX_TOKENS:-2048}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
WORKERS=${WORKERS:-16}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "$OUTPUT_ROOT/logs" code_eval/vendor

if [ ! -d code_eval/vendor/human-eval ]; then
    git clone https://github.com/openai/human-eval.git code_eval/vendor/human-eval
fi

python -m pip install -q "setuptools<81" wheel
python -m pip install -q evalplus fire

export PYTHONPATH="$PWD/code_eval/vendor/human-eval:${PYTHONPATH:-}"

for model in $MODELS; do
    model_dir="$OUTPUT_ROOT/${model//\//__}"
    samples="$model_dir/humaneval_samples.jsonl"
    results="$model_dir/results.json"
    log_path="$OUTPUT_ROOT/logs/${model//\//__}.log"
    mkdir -p "$model_dir"
    if [ "$OVERWRITE" = "1" ] || [ ! -f "$results" ]; then
        {
            CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false python -u code_eval/generate_humaneval_vllm.py \
                --model "$model" \
                --output "$samples" \
                --temperature 0 \
                --top_p 1 \
                --max_tokens "$MAX_TOKENS" \
                --batch_size "$BATCH_SIZE" \
                --max_model_len "$MAX_MODEL_LEN" \
                --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
                --prompt_mode auto
            python -u code_eval/evaluate_humaneval.py \
                --samples "$samples" \
                --output "$results" \
                --workers "$WORKERS"
            python code_eval/collect_coding_table.py \
                --output_root "$OUTPUT_ROOT" \
                --models "$MODELS" \
                --write "$OUTPUT_ROOT/results.md"
        } >"$log_path" 2>&1
    fi
    python code_eval/collect_coding_table.py \
        --output_root "$OUTPUT_ROOT" \
        --models "$MODELS" \
        --write "$OUTPUT_ROOT/results.md"
done

python code_eval/collect_coding_table.py \
    --output_root "$OUTPUT_ROOT" \
    --models "$MODELS" \
    --write "$OUTPUT_ROOT/results.md"
