import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--write", default=None)
    return parser.parse_args()


def percent(value):
    if value is None:
        return "-"
    return f"{100 * value:.1f}"


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    rows = [
        "| Model | HumanEval | HumanEval+ Base | HumanEval+ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in args.models.split():
        model_dir = output_root / model.replace("/", "__")
        result_path = model_dir / "results.json"
        if result_path.exists():
            with result_path.open() as f:
                result = json.load(f)
            row = [
                model,
                percent(result.get("human_eval_pass_at_1")),
                percent(result.get("human_eval_plus_base_pass_at_1")),
                percent(result.get("human_eval_plus_pass_at_1")),
            ]
        else:
            row = [model, "-", "-", "-"]
        rows.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    table = "\n".join(rows)
    print(table)
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(table + "\n")


if __name__ == "__main__":
    main()
