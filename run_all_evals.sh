#!/usr/bin/env bash
# Benchmarks all 4 models in parallel, one GPU each, then prints a combined table.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/eval_results}"
ENV_NAME="${ENV_NAME:-reasoning}"
PYTHON_BIN="/home/zbibm/miniconda3/envs/${ENV_NAME}/bin/python"

MODELS=(
  "$ROOT_DIR/models/Instella-3B-Instruct"
  "$ROOT_DIR/models/Instella-3B-Math-SFT"
  "$ROOT_DIR/models/Instella-3B-Math"
  "$ROOT_DIR/models/Instella-3B-Math-GRPO-Merged"
)

GPUS=(0 1 2 3)

declare -a PIDS=()

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  gpu="${GPUS[$i]}"
  log="/tmp/eval_$(basename "$model").log"
  echo "Launching: $(basename "$model") on GPU $gpu → $log"
  GPU_IDS="$gpu" RESULTS_ROOT="$RESULTS_ROOT" ENV_NAME="$ENV_NAME" \
    bash "$ROOT_DIR/run_eval.sh" "$model" > "$log" 2>&1 &
  PIDS+=($!)
done

echo
echo "All 4 models running. Waiting for completion..."
echo

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  model="${MODELS[$i]}"
  if wait "$pid"; then
    echo "DONE: $(basename "$model")"
  else
    echo "FAILED: $(basename "$model") (see /tmp/eval_$(basename "$model").log)"
  fi
done

echo
echo "================================================================"
echo "Combined results:"
echo "================================================================"
echo

RESULTS_ROOT="$RESULTS_ROOT" \
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")" \
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
models = [Path(m) for m in os.environ["MODELS_CSV"].split(",")]
model_names = [m.name for m in models]

TASKS = [
    "gsm8k",
    "gsm8k_cot",
    "gsm_plus",
    "minerva_math500",
    "hendrycks_math500",
    "agieval_math",
    "agieval_sat_math",
    "aime24",
]

def task_metric_priority(task):
    if task in {"agieval_sat_math"}:
        return ["acc_norm", "acc"]
    if task in {"agieval_math"}:
        return ["acc", "acc_norm"]
    if task in {"minerva_math500", "hendrycks_math500"}:
        return ["math_verify", "exact_match,flexible-extract", "exact_match"]
    if task in {"aime24"}:
        return ["exact_match"]
    return [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
        "math_verify",
        "acc_norm",
        "acc",
    ]

def pick_metric(result_dict, task):
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

def load_score(model_name, task):
    task_dir = results_root / model_name / task
    paths = sorted(task_dir.glob("**/results_*.json"))
    if not paths:
        return None
    with open(paths[-1]) as f:
        data = json.load(f)
    task_results = data.get("results", {}).get(task)
    if task_results is None and data.get("results"):
        task_results = next(iter(data["results"].values()))
    if task_results is None:
        return None
    _, value = pick_metric(task_results, task)
    return value

# --- build table ---
col_w = 12
name_w = 28

# short display names
display = {
    "Instella-3B-Instruct":        "Instruct (base)",
    "Instella-3B-Math-SFT":        "Math-SFT",
    "Instella-3B-Math":            "Math-GRPO",
    "Instella-3B-Math-GRPO-Merged":"GRPO-Merged",
}

header = f"{'Task':<{name_w}}" + "".join(f"{display.get(n, n):>{col_w}}" for n in model_names)
sep    = "-" * len(header)

lines = ["", sep, header, sep]

for task in TASKS:
    scores = [load_score(n, task) for n in model_names]
    # bold best (mark with *)
    numeric = [s for s in scores if s is not None]
    best = max(numeric) if numeric else None
    cells = []
    for s in scores:
        if s is None:
            cells.append(f"{'—':>{col_w}}")
        else:
            val = f"{s*100:.2f}"
            if s == best:
                val = f"{val}*"
            cells.append(f"{val:>{col_w}}")
    lines.append(f"{task:<{name_w}}" + "".join(cells))

lines.append(sep)

# avg row
avg_scores = []
for n in model_names:
    vals = [load_score(n, t) for t in TASKS]
    vals = [v for v in vals if v is not None]
    avg_scores.append(sum(vals)/len(vals) if vals else None)

best_avg = max(v for v in avg_scores if v is not None) if any(v is not None for v in avg_scores) else None
cells = []
for s in avg_scores:
    if s is None:
        cells.append(f"{'—':>{col_w}}")
    else:
        val = f"{s*100:.2f}"
        if s == best_avg:
            val = f"{val}*"
        cells.append(f"{val:>{col_w}}")
lines.append(f"{'AVG':<{name_w}}" + "".join(cells))
lines.append(sep)
lines.append("* = best in row")
lines.append("")

result = "\n".join(lines)
print(result)

out = Path(os.environ["RESULTS_ROOT"]) / "combined_summary.md"
out.write_text(result)
print(f"Saved to: {out}")
PY
