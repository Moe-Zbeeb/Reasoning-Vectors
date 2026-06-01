#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121

CONFIG_ROOT="mergekit/reverse_delta_alpha1"
OUTPUT_ROOT="$HOME/reasoning vectors/merged_checkpoints/reverse_delta_alpha1"
LOG_ROOT="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_ROOT"

run_merge() {
    local name="$1"
    local config="$CONFIG_ROOT/$name.yml"
    local output="$OUTPUT_ROOT/$name"
    local log="$LOG_ROOT/$name.log"
    echo "start $name $(date +%F_%T_%Z)"
    rm -rf "$output.tmp"
    mergekit-yaml "$config" "$output.tmp" --cuda --trust-remote-code --copy-tokenizer --safe-serialization >"$log" 2>&1
    rm -rf "$output"
    mv "$output.tmp" "$output"
    echo "done $name $(date +%F_%T_%Z)"
}

run_merge qwen_gsm8k_1p5b
run_merge primeintellect_7b
run_merge sii_enigma_7b
run_merge olmo2_7b
run_merge specreasoner_1p5b
run_merge roi_qwen25_7b

find "$OUTPUT_ROOT" -maxdepth 2 -type f \( -name "config.json" -o -name "model*.safetensors" -o -name "tokenizer*.json" \) | sort > "$OUTPUT_ROOT/artifacts.txt"
