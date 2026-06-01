import argparse
import json
from pathlib import Path

from evalplus.evaluate import evaluate as evaluate_plus
from human_eval.evaluation import evaluate_functional_correctness


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evalplus_path = Path(args.samples.replace(".jsonl", "_eval_results.json"))
    if evalplus_path.exists():
        evalplus_path.unlink()
    openai_result = evaluate_functional_correctness(
        args.samples,
        k=[1],
        n_workers=args.workers,
        timeout=args.timeout,
    )
    evaluate_plus(
        dataset="humaneval",
        samples=args.samples,
        parallel=args.workers,
        i_just_wanna_run=True,
    )
    with evalplus_path.open() as f:
        evalplus_result = json.load(f)
    evalplus_rows = [row for rows in evalplus_result["eval"].values() for row in rows]
    base_pass = sum(row["base_status"] == "pass" for row in evalplus_rows) / len(evalplus_rows)
    plus_pass = sum(
        row["base_status"] == "pass" and row["plus_status"] == "pass" for row in evalplus_rows
    ) / len(evalplus_rows)
    result = {
        "human_eval_pass_at_1": openai_result.get("pass@1"),
        "human_eval_plus_base_pass_at_1": base_pass,
        "human_eval_plus_pass_at_1": plus_pass,
        "samples": args.samples,
        "openai_result_file": args.samples + "_results.jsonl",
        "evalplus_result_file": str(evalplus_path),
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
