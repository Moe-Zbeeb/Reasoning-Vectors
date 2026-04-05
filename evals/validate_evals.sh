#!/usr/bin/env bash
# Quick smoke-test: run every task with --limit 3 on the base model (GPU 0 only).
# Exit non-zero if any task fails.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$ROOT_DIR/lm-evaluation-harness"
PYTHON_BIN="/home/zbibm/miniconda3/envs/reasoning/bin/python"
MODEL_PATH="$ROOT_DIR/models/qwen2.5-3b"
OUT_DIR="/tmp/eval_validate"
LIMIT=3
GPU=0

TASKS=(
  gsm8k
  gsm8k_cot
  gsm8k_platinum_cot_zeroshot
  gsm_plus
  minerva_math500
  minerva_math_algebra
  minerva_math_intermediate_algebra
  minerva_math_num_theory
  minerva_math_geometry
  minerva_math_counting_and_prob
  agieval_math
  agieval_sat_math
)

PASS=()
FAIL=()

cd "$HARNESS_DIR"

for task in "${TASKS[@]}"; do
  out="$OUT_DIR/$task"
  mkdir -p "$out"
  echo "──────────────────────────────────────────"
  echo "Testing: $task"

  if CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false \
      "$PYTHON_BIN" -m lm_eval run \
        --model vllm \
        --model_args "pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.5,tensor_parallel_size=1,max_model_len=4096" \
        --tasks "$task" \
        --apply_chat_template \
        --batch_size 4 \
        --seed 42 \
        --limit $LIMIT \
        --output_path "$out" \
        > "$out/run.log" 2>&1; then
    PASS+=("$task")
    # Print the metric line from results
    grep -h "acc\|exact_match\|math_verify" "$out/run.log" | tail -3 || true
    echo "  PASS"
  else
    FAIL+=("$task")
    echo "  FAIL — last 10 lines:"
    tail -10 "$out/run.log" | sed 's/^/    /'
  fi
done

echo ""
echo "══════════════════════════════════════════"
echo "RESULTS: ${#PASS[@]} passed, ${#FAIL[@]} failed"
echo "══════════════════════════════════════════"

if (( ${#PASS[@]} > 0 )); then
  echo "PASSED: ${PASS[*]}"
fi
if (( ${#FAIL[@]} > 0 )); then
  echo "FAILED: ${FAIL[*]}"
  exit 1
fi
