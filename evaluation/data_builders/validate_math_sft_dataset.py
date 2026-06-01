from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path).expanduser().resolve()
    counts = Counter()
    bad_prompt = 0
    bad_completion = 0
    bad_box = 0
    empty_answer = 0
    numina_amc_aime = 0
    lines = 0
    samples = []
    for line in path.open("r", encoding="utf-8"):
        lines += 1
        row = json.loads(line)
        source = row.get("source")
        counts[source] += 1
        prompt = row.get("prompt")
        completion = row.get("completion")
        answer = str(row.get("answer", "")).strip()
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
            content = str(completion[0].get("content", "")).rstrip()
        if not answer:
            empty_answer += 1
        if answer and not content.endswith(f"\\boxed{{{answer}}}"):
            bad_box += 1
        metadata_source = str(row.get("metadata", {}).get("source", "")).lower()
        if source == "AI-MO/NuminaMath-1.5" and (
            "amc" in metadata_source or "aime" in metadata_source
        ):
            numina_amc_aime += 1
        if len(samples) < 3:
            samples.append(
                {
                    "source": source,
                    "prompt_roles": [message.get("role") for message in prompt],
                    "answer": answer[:100],
                    "completion_tail": content[-160:],
                }
            )
    result = {
        "lines": lines,
        "counts": dict(sorted(counts.items())),
        "bad_prompt": bad_prompt,
        "bad_completion": bad_completion,
        "bad_box": bad_box,
        "empty_answer": empty_answer,
        "numina_amc_aime_source_rows": numina_amc_aime,
        "samples": samples,
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    failures = {key: value for key, value in result.items() if key.startswith("bad_") and value}
    if failures or empty_answer or numina_amc_aime:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
