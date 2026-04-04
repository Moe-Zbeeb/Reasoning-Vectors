"""
Prepare GSM8k (socratic, train split) for GRPO training.

Extracts the final integer answer from the #### line and saves as JSONL.

Run once:
    python grpo/prepare_gsm8k.py
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("/home/zbibm/Reasoning-Vectors/datasets/gsm8k_grpo.jsonl")

ANSWER_RE = re.compile(r"####\s*([\-\d,]+)")


def extract_answer(answer_text: str):
    m = ANSWER_RE.search(answer_text)
    return m.group(1).replace(",", "").strip() if m else None


def main():
    print("Loading openai/gsm8k (socratic, train) ...")
    ds = load_dataset("openai/gsm8k", "socratic", split="train")
    print(f"  Loaded {len(ds):,} examples")

    skipped = 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in ds:
            answer = extract_answer(ex["answer"])
            if answer is None:
                skipped += 1
                continue
            f.write(json.dumps({
                "question": ex["question"],
                "answer": answer,
            }, ensure_ascii=False) + "\n")

    saved = len(ds) - skipped
    print(f"  Saved {saved:,} examples → {OUTPUT_PATH}  (skipped {skipped})")


if __name__ == "__main__":
    main()
