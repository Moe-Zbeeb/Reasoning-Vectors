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

import re as _re
from math_verify import parse as _mv_parse

def _is_mc_answer(text: str) -> bool:
    """Return True if the boxed answer is just a multiple-choice letter (A-E)."""
    t = text.strip()
    # bare letter: A, B, C, D, E
    if _re.fullmatch(r"[A-E]", t):
        return True
    # \textbf{(A)} style or \text{(A)} style
    if _re.fullmatch(r"\\text(?:bf)?\{[\(\[]?[A-E][\)\]]?\}", t):
        return True
    return False

# ── Target mix ──────────────────────────────────────────────────────────────
# ~22% MATH L5  (cap 2,200 of 3,628 available)
# ~22% MATH L4  (cap 2,200 of 2,904 available)
# ~7%  MATH L3 algebra    (all 653 available — too few to cap higher)
# ~13% MATH L3 non-algebra (cap 1,300 of 2,070 available)
# ~22% GSM-Plus           (cap 2,200 of 10,552 available)
# ~7%  AMC/AIME           (cap 700 of ~3,500 filtered available)
# Total target: ~10,000
# ---------------------------------------------------------------------------

L5_CAP            = 1500   # slightly lower to give L3 algebra more relative weight
L4_CAP            = 1800
L3_ALGEBRA_CAP    = None   # use all (only ~651 available)
L3_NONALGEBRA_CAP = 1100
GSM_PLUS_CAP      = 2000   # ~26% — GSM-style anchor
AMC_AIME_CAP      = 650

# ── MATH L3 / L4 / L5 ───────────────────────────────────────────────────────
l5_rows, l4_rows, l3_alg_rows, l3_non_rows = [], [], [], []

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
        "source":   f"math_{level.replace(' ', '').lower()}_{topic.lower().replace(' & ', '_').replace(' ', '_')}",
    }
    if level == "Level 5":
        l5_rows.append(row)
    elif level == "Level 4":
        l4_rows.append(row)
    elif topic == "Algebra":
        l3_alg_rows.append(row)
    else:
        l3_non_rows.append(row)

rng.shuffle(l5_rows);      l5_rows      = l5_rows[:L5_CAP]
rng.shuffle(l4_rows);      l4_rows      = l4_rows[:L4_CAP]
rng.shuffle(l3_non_rows);  l3_non_rows  = l3_non_rows[:L3_NONALGEBRA_CAP]
# L3 algebra: use all (only 653)
rng.shuffle(l3_alg_rows)

grpo_rows = l5_rows + l4_rows + l3_alg_rows + l3_non_rows
print(f"  MATH L5:              {len(l5_rows):,}")
print(f"  MATH L4:              {len(l4_rows):,}")
print(f"  MATH L3 algebra:      {len(l3_alg_rows):,}  (all available)")
print(f"  MATH L3 non-algebra:  {len(l3_non_rows):,}")

# ── GSM-Plus ─────────────────────────────────────────────────────────────────
from datasets import load_dataset as _load_hf
gsm_plus_ds = _load_hf("qintongli/GSM-Plus", split="test")
gsm_plus_rows = []
for ex in gsm_plus_ds:
    ans = str(ex["answer"]).strip()
    if not ans:
        continue
    gsm_plus_rows.append({
        "question": ex["question"].strip(),
        "answer":   ans,
        "source":   "gsm_plus",
    })
rng.shuffle(gsm_plus_rows)
gsm_plus_rows = gsm_plus_rows[:GSM_PLUS_CAP]
grpo_rows.extend(gsm_plus_rows)
print(f"  GSM-Plus:             {len(gsm_plus_rows):,}")

# ── NuminaMath AMC/AIME ───────────────────────────────────────────────────────
# Filter to numeric/algebraic answers only — skip bare multiple-choice letters.
numina_ds = _load_hf("AI-MO/NuminaMath-CoT", split="train")
amc_rows = []
for ex in numina_ds:
    if ex["source"] != "amc_aime":
        continue
    ans = extract_boxed(ex["solution"])
    if ans is None:
        continue
    if _is_mc_answer(ans):
        continue
    # Require math_verify can parse it (skip unparseable expressions)
    if _mv_parse(ans) is None:
        continue
    amc_rows.append({
        "question": ex["problem"].strip(),
        "answer":   ans,
        "source":   "numina_amc_aime",
    })
rng.shuffle(amc_rows)
amc_rows = amc_rows[:AMC_AIME_CAP]
grpo_rows.extend(amc_rows)
print(f"  NuminaMath AMC/AIME:  {len(amc_rows):,}  (filtered non-MC only)")

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
