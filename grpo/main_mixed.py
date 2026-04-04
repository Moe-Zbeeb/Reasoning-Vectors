# accelerate launch --num_processes 4 --mixed_precision bf16 grpo/main_mixed.py
#
# GRPO on mixed dataset: GSM8k (7,473) + competition_math Level 1-3 (4,721)
# Sympy-based reward for numeric answer verification.

import json
import re
from fractions import Fraction
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/math_mixed_grpo.jsonl"
MODEL_PATH   = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"
OUTPUT_DIR   = Path("./output/grpo/mixed")
LOGS_DIR     = Path("./logs_mixed")

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    "You are a math problem solver.\n"
    f"Think step-by-step inside {REASONING_START} ... {REASONING_END}, "
    f"then write only the final answer inside {SOLUTION_START} ... {SOLUTION_END}.\n"
    "The answer may be a number, fraction, or coordinate pair."
)

# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_grpo_dataset(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run  python grpo/prepare_mixed.py  first."
        )
    ds = load_dataset("json", data_files=path, split="train")
    print(f"Loaded {len(ds):,} examples from {path}")
    return ds


def format_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["question"]},
        ],
        "answer": example["answer"],
    }

# ---------------------------------------------------------------------------
# Answer normalization (mirrors prepare_mixed.py logic)
# ---------------------------------------------------------------------------

def _parse_numeric(s: str):
    """
    Try to parse a string as a numeric value.
    Returns a float or None.
    Handles: integers, decimals, fractions (a/b), LaTeX \\frac{a}{b},
             negative fractions, dollar amounts, comma-formatted numbers.
    """
    s = s.strip().replace(",", "").replace("$", "").replace("\\$", "")

    # plain number
    try:
        return float(s)
    except ValueError:
        pass

    # a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            pass

    # LaTeX \frac{a}{b}
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            sign = -1 if m.group(1) == "-" else 1
            return sign * float(Fraction(int(m.group(2)), int(m.group(3))))
        except Exception:
            pass

    # shorthand -\frac12
    m = re.fullmatch(r"(-?)\\frac(\d)(\d)", s)
    if m:
        try:
            sign = -1 if m.group(1) == "-" else 1
            return sign * float(Fraction(int(m.group(2)), int(m.group(3))))
        except Exception:
            pass

    return None


def _parse_tuple(s: str):
    """Parse (a,b) or (a, b) into a tuple of floats, or None."""
    m = re.fullmatch(r"\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)", s.strip())
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            pass
    return None


def answers_match(guess: str, true_ans: str, tol: float = 1e-4) -> bool:
    """
    Return True if guess matches true_ans within tolerance.
    Handles: integers, decimals, fractions, LaTeX fractions, tuples.
    """
    g = guess.strip()
    t = true_ans.strip()

    # exact string match first (fastest path)
    if g == t:
        return True

    # try tuple comparison
    gt = _parse_tuple(g)
    tt = _parse_tuple(t)
    if gt is not None and tt is not None:
        return all(abs(a - b) <= tol for a, b in zip(gt, tt))

    # numeric comparison
    gv = _parse_numeric(g)
    tv = _parse_numeric(t)
    if gv is not None and tv is not None:
        if tv == 0:
            return abs(gv) <= tol
        return abs(gv - tv) / (abs(tv) + 1e-9) <= tol

    return False

# ---------------------------------------------------------------------------
# Format patterns
# ---------------------------------------------------------------------------

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

FORMAT_TOKENS = [REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END]

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_format(completions, **kwargs):
    """2.0 for full correct format, 0 otherwise."""
    return [
        2.0 if MATCH_FORMAT.search(c[0]["content"]) is not None else 0.0
        for c in completions
    ]


def reward_format_partial(completions, **kwargs):
    """±0.25 per format token."""
    scores = []
    for c in completions:
        r = c[0]["content"]
        scores.append(sum(0.25 if r.count(t) == 1 else -0.25 for t in FORMAT_TOKENS))
    return scores


def reward_correctness(prompts, completions, answer, **kwargs):
    """
    3.0  — answer matches (numeric/tuple comparison with tolerance)
   -0.5  — wrong answer
    0.0  — format missing, can't extract answer
    """
    scores = []
    for c, true_ans in zip(completions, answer):
        r = c[0]["content"]
        m = MATCH_FORMAT.search(r)
        if m is None:
            scores.append(0.0)
            continue
        guess = m.group(1).strip()
        scores.append(3.0 if answers_match(guess, str(true_ans)) else -0.5)
    return scores

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    raw_ds  = load_grpo_dataset(DATASET_PATH)
    dataset = raw_ds.map(format_example, remove_columns=raw_ds.column_names)

    print(f"Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_completion_length=1024,
        num_train_epochs=1,
        num_generations=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=4096,
        generation_batch_size=32,
        max_grad_norm=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_dir=str(LOGS_DIR),
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    n_steps = (len(dataset) // (training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps))

    print("\n" + "=" * 60)
    print("GRPO MIXED TRAINING CONFIG")
    print("=" * 60)
    print(f"Model:              {MODEL_PATH}")
    print(f"Dataset:            {DATASET_PATH}  ({len(dataset):,} examples)")
    print(f"  GSM8k:            7,473  (61.3%)")
    print(f"  Competition L1-3: 4,721  (38.7%)")
    print(f"Rewards:            format(2.0) + partial(±0.25) + correctness(3.0/-0.5)")
    print(f"Effective batch:    {training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps}")
    print(f"Est. steps/epoch:   ~{n_steps}")
    print(f"Output dir:         {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_format,
            reward_format_partial,
            reward_correctness,
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonlLoggerCallback(LOGS_DIR / "train_metrics.jsonl"))

    print("Starting GRPO mixed training ...\n")
    train_result = trainer.train()

    save_path = str(OUTPUT_DIR)
    print("\nSaving model ...")
    trainer.save_model(save_path)
    trainer.state.save_to_json(str(LOGS_DIR / "trainer_state.json"))

    summary = {
        "model_path": MODEL_PATH,
        "dataset": DATASET_PATH,
        "num_samples": len(dataset),
        "final_loss": train_result.training_loss,
        "global_step": trainer.state.global_step,
        "model_saved_to": save_path,
    }
    (LOGS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("GRPO MIXED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss:  {train_result.training_loss:.4f}")
    print(f"Steps:       {trainer.state.global_step}")
    print(f"Model:       {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
