import argparse
import json
import statistics
from pathlib import Path


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def text_tail(row, limit):
    code = row.get("code") or [""]
    return code[0][-limit:].replace("\n", " ")


def pred(row):
    values = row.get("pred") or [""]
    return values[0] if values else ""


def score(row):
    values = row.get("score") or [False]
    return bool(values[0])


def output_len(row):
    code = row.get("code") or [""]
    return len(code[0])


def summarize_lengths(label, rows):
    lengths = [output_len(row) for row in rows]
    wrong_lengths = [output_len(row) for row in rows if not score(row)]
    empty = sum(1 for row in rows if not pred(row))
    p95 = sorted(lengths)[int(0.95 * (len(lengths) - 1))]
    wrong_median = int(statistics.median(wrong_lengths)) if wrong_lengths else 0
    print(
        "  lens",
        label,
        "median",
        int(statistics.median(lengths)),
        "p95",
        int(p95),
        "max",
        max(lengths),
        "empty_pred",
        empty,
        "wrong_median",
        wrong_median,
    )


def compare(root, first_dir, second_dir, first_label, second_label, bench):
    first_path = root / first_dir / bench / "test_cot_-1_seed0_t0.6_s0_e-1.jsonl"
    second_path = root / second_dir / bench / "test_cot_-1_seed0_t0.6_s0_e-1.jsonl"
    first_rows = load_rows(first_path)
    second_rows = load_rows(second_path)
    n = min(len(first_rows), len(second_rows))
    first_correct = sum(score(row) for row in first_rows)
    second_correct = sum(score(row) for row in second_rows)
    first_only = []
    second_only = []
    both_right = 0
    both_wrong = 0
    for i in range(n):
        a = score(first_rows[i])
        b = score(second_rows[i])
        if a and b:
            both_right += 1
        elif (not a) and (not b):
            both_wrong += 1
        elif a:
            first_only.append(i)
        else:
            second_only.append(i)
    print("BENCH", bench, "n", n)
    print(
        "  correct",
        first_label,
        first_correct,
        f"{100 * first_correct / n:.1f}%",
        second_label,
        second_correct,
        f"{100 * second_correct / n:.1f}%",
    )
    print(
        "  both_right",
        both_right,
        "both_wrong",
        both_wrong,
        first_label + "_only",
        len(first_only),
        second_label + "_only",
        len(second_only),
        "net_" + first_label,
        len(first_only) - len(second_only),
    )
    summarize_lengths(first_label, first_rows)
    summarize_lengths(second_label, second_rows)
    print("  " + first_label + "-only examples")
    for i in first_only[:5]:
        a = first_rows[i]
        b = second_rows[i]
        print(
            "   rec",
            i,
            "idx",
            a.get("idx"),
            "gt",
            a.get("gt"),
            first_label + "_pred",
            pred(a),
            second_label + "_pred",
            pred(b),
        )
        print("    q", a.get("question", "")[:220].replace("\n", " "))
        print("    " + second_label + "_tail", text_tail(b, 260))
    print("  " + second_label + "-only examples")
    for i in second_only[:5]:
        a = first_rows[i]
        b = second_rows[i]
        print(
            "   rec",
            i,
            "idx",
            a.get("idx"),
            "gt",
            a.get("gt"),
            first_label + "_pred",
            pred(a),
            second_label + "_pred",
            pred(b),
        )
        print("    q", a.get("question", "")[:220].replace("\n", " "))
        print("    " + first_label + "_tail", text_tail(a, 260))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--first-dir", required=True)
    parser.add_argument("--second-dir", required=True)
    parser.add_argument("--first-label", default="first")
    parser.add_argument("--second-label", default="second")
    parser.add_argument("--benches", nargs="+", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    for bench in args.benches:
        compare(root, args.first_dir, args.second_dir, args.first_label, args.second_label, bench)


if __name__ == "__main__":
    main()
