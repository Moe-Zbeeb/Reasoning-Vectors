import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "aime25x8": "AIME25x8",
    "amc23x8": "AMC23x8",
    "aime24x8": "AIME24x8",
    "minerva_math": "Minerva Math",
    "olympiadbench": "OlympiadBench",
    "math500": "MATH500",
}


def split_items(value):
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def safe_model_name(model):
    return model.replace("/", "__")


def load_score(output_root, model, benchmark):
    metrics_dir = Path(output_root) / safe_model_name(model) / benchmark
    files = sorted(metrics_dir.glob("*_metrics.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    with files[-1].open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("acc")


def format_score(score):
    if score is None:
        return "-"
    return f"{float(score):.1f}"


def build_table(output_root, models, benchmarks):
    headers = ["Model"] + [DISPLAY_NAMES.get(name, name) for name in benchmarks] + ["Avg"]
    rows = []
    for model in models:
        scores = [load_score(output_root, model, benchmark) for benchmark in benchmarks]
        valid_scores = [float(score) for score in scores if score is not None]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else None
        rows.append([model] + [format_score(score) for score in scores] + [format_score(avg)])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/instella_vllm")
    parser.add_argument(
        "--models",
        default="amd/Instella-3B-Math-SFT amd/Instella-3B-Math amd/Instella-3B-Instruct",
    )
    parser.add_argument(
        "--benchmarks",
        default="aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500",
    )
    parser.add_argument("--write", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    models = split_items(args.models)
    benchmarks = split_items(args.benchmarks)
    table = build_table(args.output_root, models, benchmarks)
    print(table)
    if args.write:
        output_path = Path(args.write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
