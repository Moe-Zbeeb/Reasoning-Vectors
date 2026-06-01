from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

DEFAULT_COUNTS = {
    "openmath": 40000,
    "numina": 25000,
    "metamath": 15000,
    "math": 15000,
    "gsm8k": 5000,
}


BOX_PATTERN = re.compile(r"\\(?:boxed|fbox)\s*\{")


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def squash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def parse_braced(value: str, open_brace_index: int) -> str | None:
    if open_brace_index < 0 or open_brace_index >= len(value) or value[open_brace_index] != "{":
        return None
    depth = 0
    out = []
    escaped = False
    for index in range(open_brace_index, len(value)):
        char = value[index]
        if index == open_brace_index:
            depth = 1
            continue
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == "{":
            depth += 1
            out.append(char)
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(char)
            continue
        out.append(char)
    return None


def last_boxed(value: str) -> str | None:
    found = None
    for match in BOX_PATTERN.finditer(value):
        parsed = parse_braced(value, match.end() - 1)
        if parsed:
            found = parsed
    return found


def unwrap_math(value: str) -> str:
    out = value.strip()
    pairs = [("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")]
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if out.startswith(left) and out.endswith(right) and len(out) > len(left) + len(right):
                out = out[len(left) : -len(right)].strip()
                changed = True
    return out


def clean_answer(value: Any) -> str:
    out = text(value)
    if not out:
        return ""
    if "####" in out:
        out = out.split("####")[-1].strip()
    boxed = last_boxed(out)
    if boxed:
        out = boxed
    out = unwrap_math(out)
    out = out.strip()
    out = re.sub(r"^answer\s*[:=]\s*", "", out, flags=re.IGNORECASE).strip()
    out = squash(out)
    if out in {"None", "null", "nan"}:
        return ""
    return out


def extract_answer_from_solution(solution: str) -> str:
    boxed = last_boxed(solution)
    if boxed:
        return clean_answer(boxed)
    if "####" in solution:
        return clean_answer(solution.split("####")[-1])
    patterns = [
        r"(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)",
        r"(?:therefore|thus),?\s*(?:the answer is)?\s*([^\n]+)$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, solution, flags=re.IGNORECASE)
        if matches:
            candidate = clean_answer(matches[-1])
            if candidate:
                return candidate
    return ""


def remove_gsm_marker(solution: str) -> str:
    return re.sub(r"\n?\s*####\s*[^\n]+\s*$", "", solution.strip()).strip()


def final_box(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def format_solution(solution: str, answer: str) -> str:
    body = remove_gsm_marker(text(solution))
    box = final_box(answer)
    if not body:
        body = "The final answer is:"
    if body.rstrip().endswith(box):
        return body.rstrip()
    return f"{body.rstrip()}\n\n{box}"


def make_record(
    source: str, problem: Any, solution: Any, answer: Any, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    problem_text = text(problem)
    solution_text = text(solution)
    answer_text = clean_answer(answer)
    if not answer_text and solution_text:
        answer_text = extract_answer_from_solution(solution_text)
    if not problem_text or not solution_text or not answer_text:
        return None
    assistant = format_solution(solution_text, answer_text)
    if not assistant.rstrip().endswith(final_box(answer_text)):
        return None
    problem_hash = stable_hash(squash(problem_text).lower())
    record = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "completion": [
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "answer": answer_text,
        "problem_hash": problem_hash,
        "metadata": metadata,
    }
    return record


def take_records(
    name: str, rows: Iterable[dict[str, Any] | None], count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    seen = set()
    scanned = 0
    skipped = 0
    for row in rows:
        scanned += 1
        if row is None:
            skipped += 1
            continue
        key = row["problem_hash"]
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        row["metadata"]["source_index"] = len(records)
        records.append(row)
        if len(records) >= count:
            break
    return records, {
        "source": name,
        "target": count,
        "kept": len(records),
        "scanned": scanned,
        "skipped": skipped,
    }


def shuffled_stream(
    dataset: str, split: str, seed: int, config: str = "default", buffer_size: int = 10000
) -> Iterable[dict[str, Any]]:
    loaded = load_dataset(dataset, config, split=split, streaming=True)
    return iter(loaded.shuffle(seed=seed, buffer_size=buffer_size))


def openmath_rows(seed: int) -> Iterable[dict[str, Any] | None]:
    for row in shuffled_stream("nvidia/OpenMathReasoning", "cot", seed):
        if text(row.get("problem_type")) != "has_answer_extracted":
            continue
        yield make_record(
            "nvidia/OpenMathReasoning:cot",
            row.get("problem"),
            row.get("generated_solution"),
            row.get("expected_answer"),
            {
                "dataset": "nvidia/OpenMathReasoning",
                "split": "cot",
                "problem_source": row.get("problem_source"),
                "generation_model": row.get("generation_model"),
                "inference_mode": row.get("inference_mode"),
            },
        )


def truthy_or_empty(value: Any) -> bool:
    out = text(value).lower()
    return out in {"", "true", "yes", "valid", "1"}


def numina_rows(seed: int) -> Iterable[dict[str, Any] | None]:
    for row in shuffled_stream("AI-MO/NuminaMath-1.5", "train", seed):
        source = text(row.get("source"))
        lower_source = source.lower()
        if "aime" in lower_source or "amc" in lower_source:
            continue
        if not truthy_or_empty(row.get("problem_is_valid")) or not truthy_or_empty(
            row.get("solution_is_valid")
        ):
            continue
        yield make_record(
            "AI-MO/NuminaMath-1.5",
            row.get("problem"),
            row.get("solution"),
            row.get("answer"),
            {
                "dataset": "AI-MO/NuminaMath-1.5",
                "split": "train",
                "source": source,
                "problem_type": row.get("problem_type"),
                "question_type": row.get("question_type"),
                "synthetic": row.get("synthetic"),
            },
        )


def metamath_rows(seed: int) -> Iterable[dict[str, Any] | None]:
    for row in shuffled_stream("meta-math/MetaMathQA", "train", seed):
        response = text(row.get("response"))
        answer = extract_answer_from_solution(response)
        yield make_record(
            "meta-math/MetaMathQA",
            row.get("query"),
            response,
            answer,
            {
                "dataset": "meta-math/MetaMathQA",
                "split": "train",
                "type": row.get("type"),
                "original_question_hash": stable_hash(text(row.get("original_question")).lower()),
            },
        )


def level_number(value: Any) -> int:
    match = re.search(r"(\d+)", text(value))
    return int(match.group(1)) if match else 0


def math_records(count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    base = []
    for row in dataset:
        answer = extract_answer_from_solution(text(row.get("solution")))
        record = make_record(
            "DigitalLearningGmbH/MATH-lighteval:train",
            row.get("problem"),
            row.get("solution"),
            answer,
            {
                "dataset": "DigitalLearningGmbH/MATH-lighteval",
                "split": "train",
                "level": row.get("level"),
                "type": row.get("type"),
                "repeated_to_fill_quota": False,
            },
        )
        if record:
            base.append(record)
    rng = random.Random(seed)
    high = [row for row in base if level_number(row["metadata"].get("level")) >= 4]
    low = [row for row in base if level_number(row["metadata"].get("level")) < 4]
    rng.shuffle(high)
    rng.shuffle(low)
    selected = (high + low)[:count]
    repeat_count = 0
    if len(selected) < count:
        pool = high or base
        for original in cycle(pool):
            if len(selected) >= count:
                break
            repeated = json.loads(json.dumps(original))
            repeated["metadata"]["repeated_to_fill_quota"] = True
            repeated["metadata"]["repeat_index"] = repeat_count
            repeated["problem_hash"] = f"{original['problem_hash']}:repeat:{repeat_count}"
            selected.append(repeated)
            repeat_count += 1
    for index, row in enumerate(selected):
        row["metadata"]["source_index"] = index
    return selected, {
        "source": "math",
        "target": count,
        "kept": len(selected),
        "unique_available": len(base),
        "level_4_5_available": len(high),
        "repeated_to_fill_quota": repeat_count,
    }


def gsm8k_records(count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows = []
    skipped = 0
    for index in indices:
        row = dataset[index]
        record = make_record(
            "openai/gsm8k:main",
            row.get("question"),
            row.get("answer"),
            row.get("answer"),
            {
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": "train",
            },
        )
        if record is None:
            skipped += 1
            continue
        record["metadata"]["source_index"] = len(rows)
        rows.append(record)
        if len(rows) >= count:
            break
    return rows, {
        "source": "gsm8k",
        "target": count,
        "kept": len(rows),
        "available": len(dataset),
        "skipped": skipped,
    }


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(row["source"] for row in records)
    bad_prompt = 0
    bad_completion = 0
    bad_box_end = 0
    empty_answer = 0
    hash_counts = Counter(row["problem_hash"].split(":repeat:")[0] for row in records)
    repeated_problem_hashes = sum(1 for value in hash_counts.values() if value > 1)
    by_source_box_bad = Counter()
    level_counts = Counter()
    for row in records:
        prompt = row.get("prompt")
        completion = row.get("completion")
        answer = text(row.get("answer"))
        source = row.get("source", "")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 2
            or prompt[0].get("role") != "system"
            or prompt[1].get("role") != "user"
        ):
            bad_prompt += 1
        if (
            not isinstance(completion, list)
            or len(completion) != 1
            or completion[0].get("role") != "assistant"
        ):
            bad_completion += 1
            content = ""
        else:
            content = text(completion[0].get("content"))
        if not answer:
            empty_answer += 1
        if answer and not content.endswith(final_box(answer)):
            bad_box_end += 1
            by_source_box_bad[source] += 1
        level = row.get("metadata", {}).get("level")
        if level:
            level_counts[str(level)] += 1
    return {
        "total": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "bad_prompt": bad_prompt,
        "bad_completion": bad_completion,
        "bad_box_end": bad_box_end,
        "empty_answer": empty_answer,
        "unique_problem_hashes_ignoring_math_repeat": len(hash_counts),
        "repeated_problem_hashes_ignoring_math_repeat": repeated_problem_hashes,
        "by_source_box_bad": dict(sorted(by_source_box_bad.items())),
        "math_level_counts": dict(sorted(level_counts.items())),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sft_config(path: Path, train_file: Path) -> None:
    content = {
        "experiment": {"name": "math-sft-100k", "method": "sft", "mode": "instruction_tuning"},
        "model": {"name_or_path": "Qwen/Qwen2.5-Math-1.5B", "trust_remote_code": True},
        "dataset": {
            "source": "json",
            "train_split": "train",
            "eval_split": "validation",
            "data_files": {"train": str(train_file), "validation": str(train_file)},
            "columns": {"prompt": "prompt", "completion": "completion"},
            "max_eval_samples": 256,
        },
        "peft": {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "trainer": {
            "output_dir": str(train_file.parent / "outputs" / "sft"),
            "overwrite_output_dir": True,
            "max_length": 4096,
            "packing": False,
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "learning_rate": 0.00001,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.03,
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 500,
            "eval_strategy": "no",
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "ddp_find_unused_parameters": False,
            "report_to": ["tensorboard"],
            "run_name": "math-sft-100k",
        },
        "runtime": {"save_model": True, "push_to_hub": False, "allow_prompt_loss": False},
        "launcher": {
            "mode": "single_gpu",
            "gpu_ids": [0],
            "num_processes": 1,
            "mixed_precision": "bf16",
            "main_process_port": 29500,
            "env": {"TOKENIZERS_PARALLELISM": "false"},
            "deepspeed": {"zero_stage": 2},
        },
    }
    try:
        import yaml

        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def parse_counts(value: str | None) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    if not value:
        return counts
    if value.lstrip().startswith("{"):
        overrides = json.loads(value)
    else:
        overrides = {}
        for item in value.split(","):
            key, raw_count = item.split("=", 1)
            overrides[key.strip()] = int(raw_count)
    for key, count in overrides.items():
        if key not in counts:
            raise ValueError(f"Unknown count key: {key}")
        counts[key] = int(count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="training_data/math_sft_100k")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--counts-json")
    args = parser.parse_args()
    counts = parse_counts(args.counts_json)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_file = output_dir / "train.jsonl"
    report_file = output_dir / "validation_report.json"
    config_file = output_dir / "sft_config.yaml"
    all_records = []
    source_reports = []
    sources = [
        (
            "openmath",
            lambda: take_records("openmath", openmath_rows(args.seed + 1), counts["openmath"]),
        ),
        ("numina", lambda: take_records("numina", numina_rows(args.seed + 2), counts["numina"])),
        (
            "metamath",
            lambda: take_records("metamath", metamath_rows(args.seed + 3), counts["metamath"]),
        ),
        ("math", lambda: math_records(counts["math"], args.seed + 4)),
        ("gsm8k", lambda: gsm8k_records(counts["gsm8k"], args.seed + 5)),
    ]
    for name, builder in sources:
        records, report = builder()
        if len(records) != counts[name]:
            raise RuntimeError(
                f"{name} produced {len(records)} rows, expected {counts[name]}; report={report}"
            )
        all_records.extend(records)
        source_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    random.Random(args.seed).shuffle(all_records)
    validation = validate(all_records)
    expected_total = sum(counts.values())
    if validation["total"] != expected_total:
        raise RuntimeError(f"Built {validation['total']} rows, expected {expected_total}")
    failures = {
        key: validation[key]
        for key in ["bad_prompt", "bad_completion", "bad_box_end", "empty_answer"]
        if validation[key]
    }
    if failures:
        raise RuntimeError(f"Validation failed: {failures}")
    write_jsonl(train_file, all_records)
    write_sft_config(config_file, train_file)
    report = {
        "counts_requested": counts,
        "source_reports": source_reports,
        "validation": validation,
        "files": {
            "train": str(train_file),
            "report": str(report_file),
            "sft_config": str(config_file),
        },
    }
    report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
