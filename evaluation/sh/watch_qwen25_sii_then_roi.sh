#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while tmux has-session -t qwen25-7b-sii-vllm 2>/dev/null; do
    sleep 60
done

if ! tmux has-session -t qwen25-7b-roi-vllm 2>/dev/null; then
    tmux new-session -d -s qwen25-7b-roi-vllm "bash sh/eval_qwen25_7b_roi_vllm.sh"
fi
