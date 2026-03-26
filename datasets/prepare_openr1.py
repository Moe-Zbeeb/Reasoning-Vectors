#!/usr/bin/env python3
"""Prepare a balanced 25k SFT subset from OpenR1-Math-220k."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DEFAULT_TARGET_SIZE = 25_000
DEFAULT_SEED = 42
DEFAULT_MIN_SOLUTION_CHARS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert a balanced OpenR1-Math-220k subset into SFT-ready JSONL."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to datasets/openr1_math_default_25k.jsonl.",
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
        "--min-solution-chars",
        type=int,
        default=DEFAULT_MIN_SOLUTION_CHARS,
        help="Minimum non-whitespace solution length.",
    )
    return parser.parse_args()


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "openr1_math_default_25k.jsonl"


def is_reasoning_complete(example: dict) -> bool:
    values = example.get("is_reasoning_complete")
    return bool(values) and all(values)


def has_good_solution(example: dict, min_solution_chars: int) -> bool:
    solution = (example.get("solution") or "").strip()
    return len(solution) >= min_solution_chars


def dedup_by_uuid(records: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for ex in records:
        uuid = ex["uuid"]
        if uuid in seen:
            continue
        seen.add(uuid)
        deduped.append(ex)
    return deduped


def filter_pool(records: list[dict], min_correctness_count: int, min_solution_chars: int) -> list[dict]:
    return [
        ex
        for ex in records
        if ex.get("correctness_count", 0) >= min_correctness_count
        and is_reasoning_complete(ex)
        and has_good_solution(ex, min_solution_chars)
    ]


def balanced_take(records: list[dict], target_size: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ex in records:
        key = (ex.get("source", "unknown"), ex.get("problem_type", "unknown"))
        groups[key].append(ex)

    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(groups[key])

    selected = []
    group_positions = {key: 0 for key in group_keys}

    while len(selected) < target_size:
        progressed = False
        for key in group_keys:
            pos = group_positions[key]
            bucket = groups[key]
            if pos >= len(bucket):
                continue
            selected.append(bucket[pos])
            group_positions[key] += 1
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

    print("Loading open-r1/OpenR1-Math-220k [default]...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    records = [dict(row) for row in ds]
    print(f"Loaded {len(records)} rows")

    records = dedup_by_uuid(records)
    print(f"After uuid dedup: {len(records)} rows")

    strict_pool = filter_pool(records, min_correctness_count=2, min_solution_chars=args.min_solution_chars)
    print(f"Strict pool (correctness_count >= 2): {len(strict_pool)} rows")

    if len(strict_pool) >= args.target_size:
        pool = strict_pool
        print("Using strict pool only")
    else:
        relaxed_pool = filter_pool(records, min_correctness_count=1, min_solution_chars=args.min_solution_chars)
        print(f"Relaxed pool (correctness_count >= 1): {len(relaxed_pool)} rows")
        pool = relaxed_pool
        if len(pool) < args.target_size:
            raise ValueError(
                f"Not enough filtered samples for exact target size {args.target_size}; only {len(pool)} available"
            )
        print("Strict pool too small; using relaxed pool")

    selected = balanced_take(pool, target_size=args.target_size, seed=args.seed)
    print(f"Selected exactly {len(selected)} rows with seed {args.seed}")
    print(f"Writing {output_path}...")

    with output_path.open("w", encoding="utf-8") as f:
        for ex in selected:
            row = {
                "messages": [
                    {"role": "user", "content": ex["problem"]},
                    {"role": "assistant", "content": ex["solution"]},
                ],
                "answer": ex["answer"],
                "source": ex["source"],
                "problem_type": ex["problem_type"],
                "question_type": ex["question_type"],
                "correctness_count": ex["correctness_count"],
                "uuid": ex["uuid"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
