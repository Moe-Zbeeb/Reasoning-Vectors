"""
Download, filter, and save the easy-samples GRPO training dataset.

Source: open-r1/OpenR1-Math-220K (arxiv 2502.17387)
Fields: problem, answer, source, domain, llama8b_solve_rate

Tighter solve_rate filter [0.4, 0.7] vs original [0.2, 0.8]:
- Problems the 8B model gets right 40-70% of the time
- Hard enough to have learning signal, easy enough to get right during rollouts

Run once before training:
    python grpo/prepare_dataset_easy.py
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_NAME = "open-r1/OpenR1-Math-220K"
DATASET_SPLIT = "train"
NUM_SAMPLES = 10_000
SOLVE_RATE_MIN = 0.4
SOLVE_RATE_MAX = 0.7
SEED = 42

OUTPUT_PATH = Path("/home/zbibm/Reasoning-Vectors/datasets/math_grpo_easy_10k.jsonl")

# ---------------------------------------------------------------------------

def main():
    print(f"Loading '{DATASET_NAME}' ({DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"  Full dataset: {len(ds):,} examples")

    # Compute solve rate from correctness_math_verify (list of bools per problem)
    def add_solve_rate(ex):
        cv = ex.get("correctness_math_verify") or []
        if isinstance(cv, str):
            import ast
            try:
                cv = ast.literal_eval(cv)
            except Exception:
                cv = []
        ex["solve_rate"] = sum(cv) / len(cv) if cv else None
        return ex

    if "llama8b_solve_rate" in ds.column_names:
        solve_col = "llama8b_solve_rate"
    elif "correctness_math_verify" in ds.column_names:
        print("  Computing solve_rate from correctness_math_verify ...")
        ds = ds.map(add_solve_rate, num_proc=4)
        solve_col = "solve_rate"
    else:
        solve_col = None

    before = len(ds)
    if solve_col:
        ds = ds.filter(
            lambda ex: ex[solve_col] is not None
            and SOLVE_RATE_MIN <= ex[solve_col] <= SOLVE_RATE_MAX,
            num_proc=4,
        )
        print(
            f"  After difficulty filter "
            f"[{SOLVE_RATE_MIN}, {SOLVE_RATE_MAX}]: {len(ds):,} "
            f"(removed {before - len(ds):,})"
        )
    else:
        print("  No solve_rate column found — skipping difficulty filter")

    ds = ds.shuffle(seed=SEED).select(range(min(NUM_SAMPLES, len(ds))))
    print(f"  Selected: {len(ds):,} examples")

    keep = ["problem", "answer", "source", "domain", "llama8b_solve_rate"]
    keep = [c for c in keep if c in ds.column_names]
    ds = ds.select_columns(keep)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(ds):,} examples → {OUTPUT_PATH}")

    if "source" in ds.column_names:
        counts = Counter(ex["source"] for ex in ds)
        total = len(ds)
        print("\nSource breakdown:")
        print(f"  {'Source':<30} {'Count':>6}  {'%':>5}")
        print("  " + "-" * 46)
        for src, cnt in counts.most_common():
            print(f"  {src:<30} {cnt:>6}  {cnt/total*100:>4.1f}%")

    rate_col = "llama8b_solve_rate" if "llama8b_solve_rate" in ds.column_names else ("solve_rate" if "solve_rate" in ds.column_names else None)
    if rate_col:
        rates = [ex[rate_col] for ex in ds if ex[rate_col] is not None]
        avg = sum(rates) / len(rates)
        print(f"\nSolve rate stats:  min={min(rates):.2f}  avg={avg:.2f}  max={max(rates):.2f}")


if __name__ == "__main__":
    main()
