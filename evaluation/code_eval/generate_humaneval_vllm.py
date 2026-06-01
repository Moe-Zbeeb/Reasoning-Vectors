import argparse
import json
from pathlib import Path

from human_eval.data import read_problems
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--prompt_mode", choices=["raw", "chat", "auto"], default="auto")
    return parser.parse_args()


def should_use_chat(model, mode):
    if mode == "chat":
        return True
    if mode == "raw":
        return False
    lowered = model.lower()
    return "x-coder" in lowered and "base" not in lowered


def make_chat_prompts(model, prompts, mode):
    if not should_use_chat(model, mode):
        return prompts
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        return prompts
    chat_prompts = []
    for prompt in prompts:
        content = (
            "Complete the following Python function for HumanEval. "
            "Return only the missing function body. "
            "Do not include markdown, explanations, tests, or a repeated function signature.\n\n"
            f"{prompt}"
        )
        chat_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return chat_prompts


def clean_completion(text):
    text = text.strip("\n")
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lstrip().startswith("python"):
                text = text.lstrip()[6:]
    markers = [
        "\n# Test",
        "\n# Example",
        "\nassert ",
        "\nprint(",
        "\nif __name__",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[0]
    lines = text.strip("\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0].startswith("def "):
        body = []
        started = False
        for line in lines[1:]:
            if line.startswith("    ") or line.startswith("\t"):
                started = True
                body.append(line)
            elif started:
                break
        lines = body
    if lines and lines[0].strip() and not lines[0].startswith((" ", "\t")):
        lines = [("    " + line if line.strip() else line) for line in lines]
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    problems = read_problems()
    task_ids = list(problems)
    prompts = [problems[task_id]["prompt"] for task_id in task_ids]
    generation_prompts = make_chat_prompts(args.model, prompts, args.prompt_mode)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[
            "\nclass ",
            "\ndef ",
            "\nif __name__",
            "\n# Test",
            "\n# Example",
            "\nassert ",
            "\nprint(",
        ],
    )
    rows = []
    for start in range(0, len(generation_prompts), args.batch_size):
        batch_prompts = generation_prompts[start : start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        outputs = sorted(outputs, key=lambda item: int(item.request_id))
        for offset, output in enumerate(outputs):
            rows.append(
                {
                    "task_id": task_ids[start + offset],
                    "completion": clean_completion(output.outputs[0].text),
                }
            )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
