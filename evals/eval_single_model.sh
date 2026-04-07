#!/usr/bin/env bash
# Eval a single model across all 12 tasks using all 4 GPUs in parallel.
# Usage: bash evals/eval_single_model.sh <model_path> [results_dir]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$ROOT_DIR/lm-evaluation-harness"
PYTHON_BIN="/home/zbibm/miniconda3/envs/reasoning/bin/python"
MODEL_PATH="${1:?Usage: $0 <model_path> [results_dir]}"
MODEL_NAME="$(basename "$MODEL_PATH")"
RESULTS_ROOT="${2:-$ROOT_DIR/eval_results/$MODEL_NAME}"
BATCH_SIZE=16
GPU_MEMORY_UTILIZATION=0.4
MAX_MODEL_LEN=8192
SEED=42

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

GPU_LIST=(0 1 2 3)
declare -a ACTIVE_PIDS=()
declare -A PID_TO_GPU=() PID_TO_TASK=()

mkdir -p "$RESULTS_ROOT"
cd "$HARNESS_DIR"

prune_finished_jobs() {
  local new_pids=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      wait "$pid" && status="ok" || status="failed"
      echo "Completed: task=${PID_TO_TASK[$pid]} gpu=${PID_TO_GPU[$pid]} status=$status" >&2
      unset "PID_TO_GPU[$pid]" "PID_TO_TASK[$pid]"
    fi
  done
  ACTIVE_PIDS=("${new_pids[@]}")
}

find_free_gpu() {
  local threshold_mib=1024
  while true; do
    prune_finished_jobs >&2
    for gpu in "${GPU_LIST[@]}"; do
      local in_use=0
      for pid in "${ACTIVE_PIDS[@]}"; do
        [[ "${PID_TO_GPU[$pid]:-}" == "$gpu" ]] && { in_use=1; break; }
      done
      (( in_use )) && continue
      local used_mib
      used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d '[:space:]')"
      [[ -n "$used_mib" ]] && (( used_mib <= threshold_mib )) && { echo "$gpu"; return; }
    done
    sleep 3
  done
}

for task in "${TASKS[@]}"; do
  out="$RESULTS_ROOT/$task"
  if [[ "$(cat "$out/.status" 2>/dev/null)" == "SUCCESS" ]]; then
    echo "SKIP (already SUCCESS): $task"
    continue
  fi
  mkdir -p "$out"

  while (( ${#ACTIVE_PIDS[@]} >= ${#GPU_LIST[@]} )); do prune_finished_jobs; sleep 3; done
  gpu=$(find_free_gpu)

  echo "================================================================"
  echo "Task  : $task  |  GPU: $gpu"
  echo "================================================================"

  (
    export CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false
    if "$PYTHON_BIN" -m lm_eval run \
        --model vllm \
        --model_args "pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,tensor_parallel_size=1,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN" \
        --tasks "$task" \
        --apply_chat_template \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --output_path "$out" \
        > "$out/run.log" 2>&1; then
      echo "SUCCESS" > "$out/.status"
    else
      echo "FAILED ($?)" > "$out/.status"
    fi
  ) &

  pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_TASK["$pid"]="$task"
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do prune_finished_jobs; sleep 5; done

echo ""
echo "Done. Results: $RESULTS_ROOT"
for task in "${TASKS[@]}"; do
  status=$(cat "$RESULTS_ROOT/$task/.status" 2>/dev/null || echo "MISSING")
  printf "  %-40s %s\n" "$task" "$status"
done
