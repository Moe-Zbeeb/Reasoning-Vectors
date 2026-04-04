"""
Plot per-step reward metrics from a GRPO trainer_state.json.

Usage:
    python grpo/plot_rewards.py logs_gsm8k/trainer_state.json
    python grpo/plot_rewards.py logs_gsm8k/trainer_state.json --out plots/gsm8k_rewards.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same").tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_json", help="Path to trainer_state.json")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same dir as state_json)")
    parser.add_argument("--window", type=int, default=20, help="Smoothing window size")
    args = parser.parse_args()

    state_path = Path(args.state_json)
    with state_path.open() as f:
        state = json.load(f)

    history = [e for e in state["log_history"] if "reward" in e]
    if not history:
        print("No reward entries found in log_history.")
        return

    steps = [e["step"] for e in history]

    metrics = {
        "Total reward":          [e["reward"] for e in history],
        "Answer correctness":    [e.get("rewards/reward_correctness/mean",
                                        e.get("rewards/reward_answer_correctness/mean", 0)) for e in history],
        "Format exact":          [e.get("rewards/reward_format/mean",
                                        e.get("rewards/reward_format_exact/mean", 0)) for e in history],
        "Clipped ratio":         [e.get("completions/clipped_ratio", 0) for e in history],
        "Entropy":               [e.get("entropy", 0) for e in history],
    }

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    fig.suptitle(f"GRPO Training Rewards — {state_path.parent.name}", fontsize=14, fontweight="bold")

    for ax, (label, values), color in zip(axes, metrics.items(), colors):
        ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(steps, smooth(values, args.window), color=color, linewidth=1.8, label=f"{label} (smooth)")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()

    out_path = Path(args.out) if args.out else state_path.parent / "reward_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    main()
