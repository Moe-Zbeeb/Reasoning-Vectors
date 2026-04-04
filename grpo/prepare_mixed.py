"""
Prepare mixed GRPO training dataset:

  1. GSM8k (openai/gsm8k, socratic, train)
     - 7,473 examples
     - Answers: integers extracted from '#### N'

  2. competition_math (qwedsacf/competition_math, train, Level 1-3 only)
     - Answers extracted from \\boxed{} in solution
     - Keeps only verifiable numeric answers: integer, decimal, fraction, tuple
     - Skips symbolic expressions that can't be reliably compared

Mixed dataset saved to: datasets/math_mixed_grpo.jsonl
Fields per row: question, answer, source

Run once before training:
    python grpo/prepare_mixed.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("/home/zbibm/Reasoning-Vectors/datasets/math_mixed_grpo.jsonl")
SEED = 42

# ---------------------------------------------------------------------------
# GSM8k helpers
# ---------------------------------------------------------------------------

GSM_ANSWER_RE = re.compile(r"####\s*([\-\d,]+)")


def extract_gsm_answer(answer_text: str):
    m = GSM_ANSWER_RE.search(answer_text)
    return m.group(1).replace(",", "").strip() if m else None


# ---------------------------------------------------------------------------
# Competition math helpers
# ---------------------------------------------------------------------------

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


def normalize_latex_number(s: str):
    """
    Convert LaTeX numeric strings to a canonical Python string.
    Returns None if the answer is symbolic/non-numeric.
    """
    s = s.strip()

    # plain integer or decimal
    if re.fullmatch(r"-?\d+", s):
        return s
    if re.fullmatch(r"-?\d*\.\d+", s):
        return s

    # negative fraction: -\frac{a}{b} or \frac{-a}{b}
    frac = re.fullmatch(
        r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}|(-?)\\frac\{\\?(-?\d+)\}\{(-?\d+)\}", s
    )
    if frac:
        try:
            from fractions import Fraction
            # try both capture group patterns
            groups = [g for g in frac.groups() if g is not None]
            # groups: sign, num, den  (or num, den without explicit sign)
            if len(groups) == 3:
                sign, num, den = groups
                val = Fraction(int(num), int(den))
                if sign == "-":
                    val = -val
            else:
                num, den = groups[-2], groups[-1]
                val = Fraction(int(num), int(den))
            return str(float(val)) if val.denominator != 1 else str(int(val))
        except Exception:
            pass

    # simple fraction without sign: \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            from fractions import Fraction
            val = Fraction(int(m.group(1)), int(m.group(2)))
            return str(float(val)) if val.denominator != 1 else str(int(val))
        except Exception:
            pass

    # tuple/coordinate: (a, b) or (a,b)
    m = re.fullmatch(r"\((-?\s*[\d\.]+)\s*,\s*(-?\s*[\d\.]+)\)", s)
    if m:
        return f"({m.group(1).strip()},{m.group(2).strip()})"

    # dollar amount: \$42409 or 42,409
    m = re.fullmatch(r"\\\$(\d[\d,]*)", s)
    if m:
        return m.group(1).replace(",", "")

    # comma-formatted integer: 42,409
    m = re.fullmatch(r"-?\d{1,3}(,\d{3})+", s)
    if m:
        return s.replace(",", "")

    # negative decimal: -\frac12 style shorthand
    m = re.fullmatch(r"-?\\frac(\d)(\d)", s)
    if m:
        try:
            from fractions import Fraction
            val = Fraction(int(m.group(1)), int(m.group(2)))
            return str(float(val))
        except Exception:
            pass

    return None  # symbolic / expression — skip


def is_verifiable(answer: str) -> bool:
    return normalize_latex_number(answer) is not None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []

    # --- GSM8k ---
    print("Loading GSM8k (socratic, train) ...")
    gsm = load_dataset("openai/gsm8k", "socratic", split="train")
    gsm_skipped = 0
    for ex in gsm:
        ans = extract_gsm_answer(ex["answer"])
        if ans is None:
            gsm_skipped += 1
            continue
        rows.append({"question": ex["question"], "answer": ans, "source": "gsm8k"})
    print(f"  GSM8k: {len(gsm) - gsm_skipped:,} kept, {gsm_skipped} skipped")

    # --- Competition Math ---
    print("Loading qwedsacf/competition_math (train, Level 1-3) ...")
    comp = load_dataset("qwedsacf/competition_math", split="train")
    comp_kept = comp_skipped_level = comp_skipped_ans = 0

    for ex in comp:
        if ex["level"] not in ("Level 1", "Level 2", "Level 3"):
            comp_skipped_level += 1
            continue
        raw_ans = extract_boxed(ex["solution"])
        if raw_ans is None:
            comp_skipped_ans += 1
            continue
        norm = normalize_latex_number(raw_ans)
        if norm is None:
            comp_skipped_ans += 1
            continue
        rows.append({
            "question": ex["problem"],
            "answer": norm,
            "source": f"competition_math_L{ex['level'].split()[-1]}",
        })
        comp_kept += 1

    print(f"  Competition math: {comp_kept:,} kept")
    print(f"    Skipped (level 4-5): {comp_skipped_level:,}")
    print(f"    Skipped (non-numeric answer): {comp_skipped_ans:,}")

    # shuffle deterministically
    import random
    rng = random.Random(SEED)
    rng.shuffle(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    src_counts = Counter(r["source"] for r in rows)
    print(f"\nSaved {total:,} examples → {OUTPUT_PATH}")
    print("\nSource breakdown:")
    print(f"  {'Source':<35} {'Count':>6}  {'%':>5}")
    print("  " + "-" * 50)
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<35} {cnt:>6}  {cnt/total*100:>4.1f}%")


if __name__ == "__main__":
    main()
