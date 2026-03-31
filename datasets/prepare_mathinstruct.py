#!/usr/bin/env python3
"""Prepare a balanced 25k SFT subset from TIGER-Lab/MathInstruct."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DEFAULT_TARGET_SIZE = 25_000
DEFAULT_SEED = 42
DEFAULT_MIN_OUTPUT_CHARS = 24
DEFAULT_MIN_INSTRUCTION_CHARS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert a balanced MathInstruct subset into SFT-ready JSONL."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to datasets/mathinstruct_25k.jsonl.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Exact number of samples to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic balanced sampling.",
    )
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=DEFAULT_MIN_OUTPUT_CHARS,
        help="Minimum non-whitespace assistant output length.",
    )
    parser.add_argument(
        "--min-instruction-chars",
        type=int,
        default=DEFAULT_MIN_INSTRUCTION_CHARS,
        help="Minimum non-whitespace user instruction length.",
    )
    return parser.parse_args()


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "mathinstruct_25k.jsonl"


def clean(text: str | None) -> str:
    return (text or "").strip()


def is_valid(example: dict, min_instruction_chars: int, min_output_chars: int) -> bool:
    instruction = clean(example.get("instruction"))
    output = clean(example.get("output"))
    return len(instruction) >= min_instruction_chars and len(output) >= min_output_chars


def dedup_pairs(records: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for ex in records:
        key = (clean(ex.get("instruction")), clean(ex.get("output")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def balanced_take(records: list[dict], target_size: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in records:
        groups[ex.get("source") or "unknown"].append(ex)

    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    selected = []
    positions = {key: 0 for key in keys}

    while len(selected) < target_size:
        progressed = False
        for key in keys:
            pos = positions[key]
            bucket = groups[key]
            if pos >= len(bucket):
                continue
            selected.append(bucket[pos])
            positions[key] += 1
            progressed = True
            if len(selected) == target_size:
                break
        if not progressed:
            break

    if len(selected) != target_size:
        raise ValueError(f"Could not select exactly {target_size} samples; only found {len(selected)}")

    return selected


def main() -> int:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TIGER-Lab/MathInstruct [train]...")
    ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
    records = [dict(row) for row in ds]
    print(f"Loaded {len(records)} rows")

    records = [
        ex
        for ex in records
        if is_valid(
            ex,
            min_instruction_chars=args.min_instruction_chars,
            min_output_chars=args.min_output_chars,
        )
    ]
    print(f"After basic filtering: {len(records)} rows")

    records = dedup_pairs(records)
    print(f"After instruction/output dedup: {len(records)} rows")

    if len(records) < args.target_size:
        raise ValueError(
            f"Not enough filtered samples for exact target size {args.target_size}; only {len(records)} available"
        )

    selected = balanced_take(records, target_size=args.target_size, seed=args.seed)
    print(f"Selected exactly {len(selected)} rows with seed {args.seed}")
    print(f"Writing {output_path}...")

    with output_path.open("w", encoding="utf-8") as f:
        for ex in selected:
            row = {
                "messages": [
                    {"role": "user", "content": clean(ex.get("instruction"))},
                    {"role": "assistant", "content": clean(ex.get("output"))},
                ],
                "source": ex.get("source") or "unknown",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
