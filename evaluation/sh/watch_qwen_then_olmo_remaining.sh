#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while tmux has-session -t qwen15b-then-olmo-vllm 2>/dev/null; do
    if tmux capture-pane -pt qwen15b-then-olmo-vllm -S -200 2>/dev/null | grep -q "end-qwen25-math-15b"; then
        tmux kill-session -t qwen15b-then-olmo-vllm 2>/dev/null || true
        break
    fi
    sleep 30
done

if ! tmux has-session -t olmo-remaining-vllm 2>/dev/null; then
    tmux new-session -d -s olmo-remaining-vllm "bash sh/eval_olmo_remaining_vllm.sh"
fi
