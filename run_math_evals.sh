#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$ROOT_DIR/lm-evaluation-harness"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/eval_results}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ENV_NAME="${ENV_NAME:-reasoning}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
SEED="${SEED:-42}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

PYTHON_BIN="/home/zbibm/miniconda3/envs/${ENV_NAME}/bin/python"
PIP_BIN="/home/zbibm/miniconda3/envs/${ENV_NAME}/bin/pip"

BASE_MODEL="$ROOT_DIR/models/qwen1.5B"
SFT_MODEL="$ROOT_DIR/models/output/sft/qwen1.5Bmath"

TASKS=(
  gsm8k
  gsm8k_cot
  gsm_plus
  minerva_math500
  hendrycks_math500
  agieval_math
  agieval_sat_math
)

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

model_slug() {
  basename "$1"
}

split_gpus() {
  IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
}

wait_for_gpu_release() {
  local gpu="$1"
  local used_mib threshold_mib
  threshold_mib="${GPU_REUSE_MEMORY_THRESHOLD_MIB:-1024}"

  while true; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d '[:space:]')"
    if [[ -n "$used_mib" ]] && (( used_mib <= threshold_mib )); then
      break
    fi
    echo "Waiting for GPU $gpu memory to drain (used=${used_mib:-unknown} MiB, threshold=$threshold_mib MiB)"
    sleep 5
  done
}

launch_eval() {
  local gpu="$1"
  local model_path="$2"
  local task="$3"
  local model_name out_dir log_file

  model_name="$(model_slug "$model_path")"
  out_dir="$RESULTS_ROOT/$model_name/$task"
  log_file="$out_dir/run.log"
  mkdir -p "$out_dir"

  echo
  echo "================================================================"
  echo "Model : $model_name"
  echo "Task  : $task"
  echo "GPU   : $gpu"
  echo "Output: $out_dir"
  echo "Log   : $log_file"
  echo "Backend: vllm"
  echo "================================================================"

  (
    cd "$HARNESS_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TOKENIZERS_PARALLELISM=false

    cmd=(
      "$PYTHON_BIN" -m lm_eval run
      --model vllm
      --model_args "pretrained=$model_path,dtype=auto,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,tensor_parallel_size=1,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN"
      --tasks "$task"
      --apply_chat_template
      --batch_size "$BATCH_SIZE"
      --seed "$SEED"
      --output_path "$out_dir"
    )

    if [[ "$LOG_SAMPLES" == "1" ]]; then
      cmd+=(--log_samples)
    fi

    if "${cmd[@]}" > >(tee "$log_file") 2>&1; then
      echo "SUCCESS" > "$out_dir/.status"
    else
      status=$?
      echo "FAILED ($status)" > "$out_dir/.status"
      exit "$status"
    fi
  ) &

  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_MODEL["$pid"]="$model_name"
  PID_TO_TASK["$pid"]="$task"
}

prune_finished_jobs() {
  local new_pids=()
  local pid status
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      if wait "$pid"; then
        status="ok"
      else
        status="failed"
      fi
      echo "Completed: model=${PID_TO_MODEL[$pid]} task=${PID_TO_TASK[$pid]} gpu=${PID_TO_GPU[$pid]} status=${status}"
      unset 'PID_TO_GPU[$pid]' 'PID_TO_MODEL[$pid]' 'PID_TO_TASK[$pid]'
    fi
  done
  ACTIVE_PIDS=("${new_pids[@]}")
}

wait_for_slot() {
  while (( ${#ACTIVE_PIDS[@]} >= ${#GPU_LIST[@]} )); do
    prune_finished_jobs
    sleep 5
  done
}

wait_for_all() {
  while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    prune_finished_jobs
    sleep 5
  done
}

write_summary() {
  TASKS_CSV="$(IFS=,; echo "${TASKS[*]}")" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  BASE_MODEL_NAME="$(model_slug "$BASE_MODEL")" \
  SFT_MODEL_NAME="$(model_slug "$SFT_MODEL")" \
  "$PYTHON_BIN" - <<'PY'
import csv
import glob
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
base_model = os.environ["BASE_MODEL_NAME"]
sft_model = os.environ["SFT_MODEL_NAME"]
tasks = [t for t in os.environ["TASKS_CSV"].split(",") if t]
summary_path = results_root / "summary.md"
summary_csv_path = results_root / "summary.csv"


def latest_result_json(task_dir: Path):
    paths = sorted(task_dir.glob("**/results_*.json"))
    return paths[-1] if paths else None


def task_metric_priority(task: str):
    if task in {"agieval_sat_math"}:
        return ["acc_norm", "acc"]
    if task in {"agieval_math"}:
        return ["acc", "acc_norm"]
    return [
        "exact_match,flexible-extract",
        "exact_match",
        "exact_match,strict-match",
        "math_verify",
        "acc_norm",
        "acc",
    ]


def pick_metric(result_dict: dict, task: str):
    priorities = task_metric_priority(task)
    for candidate in priorities:
        for key, value in result_dict.items():
            if key == candidate or key.startswith(candidate + ","):
                return key, value
    for key, value in result_dict.items():
        if key in {"name", "alias", "sample_len"}:
            continue
        if key.endswith("_stderr") or ",stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def load_task_result(model_name: str, task: str):
    task_dir = results_root / model_name / task
    status_path = task_dir / ".status"
    status = status_path.read_text().strip() if status_path.exists() else "MISSING"
    result_path = latest_result_json(task_dir)
    if result_path is None:
        return {"status": status, "metric": None, "value": None}

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_results = data.get("results", {}).get(task)
    if task_results is None and data.get("results"):
        task_results = next(iter(data["results"].values()))
    if task_results is None:
        return {"status": status, "metric": None, "value": None}

    metric, value = pick_metric(task_results, task)
    return {"status": status, "metric": metric, "value": value}


def fmt_value(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


rows = []
for task in tasks:
    base = load_task_result(base_model, task)
    sft = load_task_result(sft_model, task)
    metric = sft["metric"] or base["metric"] or "-"
    delta = None
    if isinstance(base["value"], (int, float)) and isinstance(sft["value"], (int, float)):
        delta = sft["value"] - base["value"]
    rows.append({
        "Task": task,
        "Metric": metric,
        "Base": fmt_value(base["value"]),
        "SFT": fmt_value(sft["value"]),
        "Delta": fmt_value(delta),
        "BaseStatus": base["status"],
        "SFTStatus": sft["status"],
    })

lines = [
    "| Task | Metric | Base qwen1.5B | SFT qwen1.5bMathSft | Delta | Base status | SFT status |",
    "|---|---|---:|---:|---:|---|---|",
]
for row in rows:
    lines.append(
        f"| {row['Task']} | {row['Metric']} | {row['Base']} | {row['SFT']} | {row['Delta']} | {row['BaseStatus']} | {row['SFTStatus']} |"
    )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Task", "Metric", "Base", "SFT", "Delta", "BaseStatus", "SFTStatus"])
    writer.writeheader()
    writer.writerows(rows)

print(summary_path.read_text(encoding="utf-8"), end="")
PY
}

require_path "$HARNESS_DIR"
require_path "$BASE_MODEL"
require_path "$SFT_MODEL"
require_path "$PYTHON_BIN"
require_path "$PIP_BIN"

mkdir -p "$RESULTS_ROOT"

cd "$HARNESS_DIR"
if ! "$PYTHON_BIN" -c 'import lm_eval, vllm, ray, sympy, math_verify, antlr4' >/dev/null 2>&1; then
  "$PIP_BIN" install -e '.[hf,math,vllm]'
  "$PIP_BIN" install ray
fi

split_gpus
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "No GPUs specified in GPU_IDS" >&2
  exit 1
fi

declare -a ACTIVE_PIDS=()
declare -A PID_TO_GPU=()
declare -A PID_TO_MODEL=()
declare -A PID_TO_TASK=()

job_index=0
for model in "$BASE_MODEL" "$SFT_MODEL"; do
  for task in "${TASKS[@]}"; do
    wait_for_slot
    gpu="${GPU_LIST[$(( job_index % ${#GPU_LIST[@]} ))]}"
    wait_for_gpu_release "$gpu"
    launch_eval "$gpu" "$model" "$task"
    job_index=$((job_index + 1))
  done
done

wait_for_all

echo
printf 'Finished. Results are under: %s\n' "$RESULTS_ROOT"
echo
write_summary
