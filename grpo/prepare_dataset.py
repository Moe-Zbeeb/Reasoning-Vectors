"""
Download, filter, and save the GRPO training dataset.

Source: open-r1/OpenR1-Math-220K (arxiv 2502.17387)
Fields: problem, answer, source, domain, llama8b_solve_rate

Run once before training:
    python grpo/prepare_dataset.py
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
SOLVE_RATE_MIN = 0.2
SOLVE_RATE_MAX = 0.8
SEED = 42

OUTPUT_PATH = Path("/home/zbibm/Reasoning-Vectors/datasets/math_grpo_10k.jsonl")

# ---------------------------------------------------------------------------

def main():
    print(f"Loading '{DATASET_NAME}' ({DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"  Full dataset: {len(ds):,} examples")

    # Filter to medium difficulty
    before = len(ds)
    if "llama8b_solve_rate" in ds.column_names:
        ds = ds.filter(
            lambda ex: ex["llama8b_solve_rate"] is not None
            and SOLVE_RATE_MIN <= ex["llama8b_solve_rate"] <= SOLVE_RATE_MAX,
            num_proc=4,
        )
        print(
            f"  After difficulty filter "
            f"[{SOLVE_RATE_MIN}, {SOLVE_RATE_MAX}]: {len(ds):,} "
            f"(removed {before - len(ds):,})"
        )
    else:
        print("  llama8b_solve_rate not found — skipping difficulty filter")

    # Deterministic shuffle + truncate
    ds = ds.shuffle(seed=SEED).select(range(min(NUM_SAMPLES, len(ds))))
    print(f"  Selected: {len(ds):,} examples")

    # Keep only the columns we need
    keep = ["problem", "answer", "source", "domain", "llama8b_solve_rate"]
    keep = [c for c in keep if c in ds.column_names]
    ds = ds.select_columns(keep)

    # Save as JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(ds):,} examples → {OUTPUT_PATH}")

    # Print source breakdown for README
    if "source" in ds.column_names:
        counts = Counter(ex["source"] for ex in ds)
        total = len(ds)
        print("\nSource breakdown:")
        print(f"  {'Source':<30} {'Count':>6}  {'%':>5}")
        print("  " + "-" * 46)
        for src, cnt in counts.most_common():
            print(f"  {src:<30} {cnt:>6}  {cnt/total*100:>4.1f}%")


if __name__ == "__main__":
    main()
