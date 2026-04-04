# accelerate launch --num_processes 4 --mixed_precision bf16 grpo/main_easy.py
#
# Easy-samples GRPO run: solve_rate ∈ [0.4, 0.7]
# Key change vs main.py: format rewards halved so answer correctness dominates.

import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/math_grpo_easy_10k.jsonl"

SFT_MODEL = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"
BASE_MODEL = "/home/zbibm/Reasoning-Vectors/models/qwen2.5-3b"
MODEL_PATH = SFT_MODEL if Path(SFT_MODEL).exists() else BASE_MODEL

OUTPUT_DIR = Path("./output_easy")
LOGS_DIR = Path("./logs_easy")

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {REASONING_START} and {REASONING_END}
2. Provide your final answer between {SOLUTION_START} and {SOLUTION_END}
3. Be precise and show all calculation steps clearly."""


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
            "Run  python grpo/prepare_dataset_easy.py  first."
        )
    print(f"Loading dataset from {path} ...")
    ds = load_dataset("json", data_files=path, split="train")
    print(f"  Loaded {len(ds):,} examples")
    return ds


def format_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "answer": example["answer"],
    }


# ---------------------------------------------------------------------------
# Reward functions
# Reward weights tuned for easy samples: format rewards halved (1.5 / 0.25)
# so answer correctness (3.0 / 1.5 / 0.5) is the dominant training signal.
# ---------------------------------------------------------------------------

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

MATCH_NUMBERS = re.compile(
    rf"{re.escape(SOLUTION_START)}.*?([\d,\.]+)",
    flags=re.MULTILINE | re.DOTALL,
)


def reward_format_exact(completions, **kwargs):
    """Format reward halved to 1.5 so answer correctness dominates."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(1.5 if MATCH_FORMAT.search(response) is not None else 0.0)
    return scores


def reward_format_approximate(completions, **kwargs):
    """Partial credit halved to ±0.25 per token."""
    scores = []
    for completion in completions:
        r = completion[0]["content"]
        score = 0.0
        score += 0.25 if r.count(REASONING_START) == 1 else -0.25
        score += 0.25 if r.count(REASONING_END) == 1 else -0.25
        score += 0.25 if r.count(SOLUTION_START) == 1 else -0.25
        score += 0.25 if r.count(SOLUTION_END) == 1 else -0.25
        scores.append(score)
    return scores


def reward_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy (unchanged weights):
      3.0 — exact string match
      1.5 — within 10% numerically
      0.5 — within 20% numerically
     -0.5 — wrong answer
    """
    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := MATCH_FORMAT.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_ans in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        if guess.strip() == str(true_ans).strip():
            scores.append(3.0)
            continue
        try:
            ratio = float(guess.replace(",", "")) / float(
                str(true_ans).replace(",", "")
            )
            if 0.9 <= ratio <= 1.1:
                scores.append(1.5)
            elif 0.8 <= ratio <= 1.2:
                scores.append(0.5)
            else:
                scores.append(-0.5)
        except (ValueError, ZeroDivisionError):
            scores.append(-0.5)
    return scores


def reward_number_extraction(prompts, completions, answer, **kwargs):
    """Binary reward (1.5) for correctly parsing the numerical value."""
    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := MATCH_NUMBERS.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_ans in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            guess_val = float(guess.replace(",", ""))
            true_val = float(str(true_ans).replace(",", ""))
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    raw_ds = load_grpo_dataset(DATASET_PATH)
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
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        logging_dir=str(LOGS_DIR),
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    print("\n" + "=" * 60)
    print("GRPO EASY-SAMPLES TRAINING CONFIG")
    print("=" * 60)
    print(f"Model:                {MODEL_PATH}")
    print(f"Dataset:              {DATASET_PATH}  ({len(dataset):,} examples)")
    print(f"Solve rate filter:    [0.4, 0.7]  (easy samples)")
    print(f"Format reward:        1.5  (halved vs original 3.0)")
    print(f"Ans correct reward:   3.0 / 1.5 / 0.5 / -0.5")
    print(f"Precision:            bf16 + flash_attention_2")
    print(f"Epochs:               {training_args.num_train_epochs}")
    print(f"Effective batch:      {training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps}")
    print(f"Generations (G):      {training_args.num_generations}")
    print(f"Max completion len:   {training_args.max_completion_length}")
    print(f"vLLM:                 colocate  (gpu_mem_util=0.25)")
    print(f"Learning rate:        {training_args.learning_rate}")
    print(f"Output dir:           {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    save_path = "/home/zbibm/Reasoning-Vectors/models/output/grpo/qwen2.5-3b-math-grpo-easy"

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_format_exact,
            reward_format_approximate,
            reward_answer_correctness,
            reward_number_extraction,
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonlLoggerCallback(LOGS_DIR / "train_metrics.jsonl"))

    print("Starting GRPO easy-samples training ...\n")
    train_result = trainer.train()

    print("\nSaving model ...")
    trainer.save_model(save_path)
    trainer.state.save_to_json(str(LOGS_DIR / "trainer_state.json"))

    summary = {
        "model_path": MODEL_PATH,
        "dataset": DATASET_PATH,
        "solve_rate_filter": [0.4, 0.7],
        "num_samples": len(dataset),
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "num_generations": training_args.num_generations,
        "max_completion_length": training_args.max_completion_length,
        "learning_rate": training_args.learning_rate,
        "final_loss": train_result.training_loss,
        "global_step": trainer.state.global_step,
        "model_saved_to": save_path,
    }
    (LOGS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("GRPO EASY-SAMPLES TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss:  {train_result.training_loss:.4f}")
    print(f"Steps:       {trainer.state.global_step}")
    print(f"Model:       {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
