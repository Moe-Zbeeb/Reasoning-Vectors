"""
Prepare SFT and GRPO datasets.

SFT — math_sft.jsonl
    30% GSM8K      (openai/gsm8k, main, train)
    50% MATH       (qwedsacf/competition_math, all levels)
    20% NuminaMath (AI-MO/NuminaMath-CoT, amc_aime + olympiads only)

    Format: {"messages": [system, user, assistant]}
    System prompt instructs the model to put final answer in \\boxed{}.
    This fixes hendrycks_math500 which expects \\boxed{} output.

GRPO — math_grpo.jsonl
    MATH L5        (qwedsacf/competition_math, level 5) — all ~3,628
    MATH L4        (qwedsacf/competition_math, level 4) — all ~2,904
    MATH L3        (qwedsacf/competition_math, level 3) — all topics
                   non-algebra: ~2,070 (primary signal)
                   algebra:       ~400 sampled (reward anchor)
    GSM8K          dropped — model scores 66-70%, near-zero reward variance
    NuminaMath     dropped — rollout check showed 1/13 (8%) solve rate, dead gradient

    Format: {"question", "answer", "source"}
    Answers kept as raw LaTeX strings — verified at reward time with math_verify.

Run:
    python datasets/prepare.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 42
rng  = random.Random(SEED)

SFT_OUTPUT  = Path("/home/zbibm/Reasoning-Vectors/datasets/math_sft.jsonl")
GRPO_OUTPUT = Path("/home/zbibm/Reasoning-Vectors/datasets/math_grpo.jsonl")

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GSM_ANSWER_RE = re.compile(r"####\s*([\-\d,]+)")


def extract_gsm_answer(text: str):
    m = GSM_ANSWER_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None


def gsm_to_solution(answer_text: str) -> str:
    """Convert GSM8K answer field to clean solution ending with \\boxed{}."""
    clean = re.sub(r"<<[^>]+>>", "", answer_text)   # strip <<calc>> annotations
    ans   = extract_gsm_answer(answer_text)
    clean = re.sub(r"\n?####.*", "", clean).strip()
    if ans:
        clean = f"{clean}\n\nThe answer is $\\boxed{{{ans}}}$."
    return clean


def extract_boxed(solution: str):
    """Extract content of last \\boxed{} handling nested braces."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth, i = 0, idx + len(r"\boxed{") - 1
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[idx + len(r"\boxed{"):i].strip()
        i += 1
    return None


def sft_row(problem: str, solution: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem.strip()},
            {"role": "assistant", "content": solution.strip()},
        ]
    }


def save_jsonl(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------

print("Loading GSM8K (main, train) ...")
gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
print(f"  {len(gsm_ds):,} rows")

print("Loading qwedsacf/competition_math (train, all levels) ...")
math_ds = load_dataset("qwedsacf/competition_math", split="train")
print(f"  {len(math_ds):,} rows  |  levels: {dict(Counter(math_ds['level']).most_common())}")

print("Loading AI-MO/NuminaMath-CoT (train, amc_aime + olympiads) ...")
numina_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
numina_ds = numina_ds.filter(
    lambda x: x["source"] in ("amc_aime", "olympiads"),
    num_proc=4,
)
print(f"  {len(numina_ds):,} rows after source filter")

# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------
# Anchor on GSM8K (all 7,473) as the 30% → total ~24,900
# MATH:       ~12,450 → use all 12,500
# NuminaMath: ~4,980  → sample from 154k

print("\n--- Building SFT dataset ---")

sft_rows = []

# GSM8K
for ex in gsm_ds:
    sol = gsm_to_solution(ex["answer"])
    sft_rows.append(sft_row(ex["question"], sol))
gsm_count = len(gsm_ds)
print(f"  GSM8K:      {gsm_count:,}")

# MATH all levels
math_added = 0
for ex in math_ds:
    if not ex["problem"] or not ex["solution"]:
        continue
    if r"\boxed" not in ex["solution"]:
        continue
    sft_rows.append(sft_row(ex["problem"], ex["solution"]))
    math_added += 1
print(f"  MATH:       {math_added:,}")

# NuminaMath — sample to ~20% of final total
numina_target = int((gsm_count + math_added) * (0.20 / 0.80))
indices = list(range(len(numina_ds)))
rng.shuffle(indices)
numina_added = 0
for idx in indices:
    if numina_added >= numina_target:
        break
    ex = numina_ds[idx]
    if not ex["problem"] or not ex["solution"]:
        continue
    if r"\boxed" not in ex["solution"]:
        continue
    sft_rows.append(sft_row(ex["problem"], ex["solution"]))
    numina_added += 1
print(f"  NuminaMath: {numina_added:,}  (target {numina_target:,})")

rng.shuffle(sft_rows)
save_jsonl(sft_rows, SFT_OUTPUT)
print(f"\nSFT total: {len(sft_rows):,} → {SFT_OUTPUT}")

# ---------------------------------------------------------------------------
# GRPO dataset
# ---------------------------------------------------------------------------

print("\n--- Building GRPO dataset ---")

grpo_rows = []

# MATH L4 + L5 — all problems
# MATH L3 — non-algebra in full, algebra sampled to ~400 as reward anchor
L3_ALGEBRA_CAP = 400
l3_algebra_rows = []

for ex in math_ds:
    level = ex.get("level", "")
    topic = ex.get("type", "")
    if level not in ("Level 3", "Level 4", "Level 5"):
        continue
    ans = extract_boxed(ex["solution"])
    if ans is None:
        continue
    row = {
        "question": ex["problem"].strip(),
        "answer":   ans,
        "source":   f"math_{ex['level'].replace(' ', '').lower()}_{topic.lower().replace(' & ', '_').replace(' ', '_')}",
    }
    if level == "Level 3" and topic == "Algebra":
        l3_algebra_rows.append(row)
    else:
        grpo_rows.append(row)

# Sample L3 algebra down to anchor size
rng.shuffle(l3_algebra_rows)
l3_algebra_sampled = l3_algebra_rows[:L3_ALGEBRA_CAP]
grpo_rows.extend(l3_algebra_sampled)

from collections import Counter as _Counter
lvl_counts = _Counter(r["source"].split("_")[1] for r in grpo_rows)
print(f"  MATH L5: {lvl_counts.get('level5', 0):,}")
print(f"  MATH L4: {lvl_counts.get('level4', 0):,}")
print(f"  MATH L3 (non-algebra full + algebra anchor {L3_ALGEBRA_CAP}): {lvl_counts.get('level3', 0):,}")

rng.shuffle(grpo_rows)
save_jsonl(grpo_rows, GRPO_OUTPUT)

total = len(grpo_rows)
src   = Counter(r["source"] for r in grpo_rows)
print(f"\nGRPO total: {total:,} → {GRPO_OUTPUT}")
print(f"\n  {'Source':<40} {'Count':>6}  {'%':>5}")
print("  " + "-" * 52)
for s, c in sorted(src.items(), key=lambda x: -x[1]):
    print(f"  {s:<40} {c:>6}  {c/total*100:>4.1f}%")

print("\nDone.")
