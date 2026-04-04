# accelerate launch --num_processes 4 --mixed_precision bf16 grpo/main_gsm8k.py
#
# GRPO on GSM8k (socratic, 7.47k train examples).
# Two rewards only: format correctness + exact integer answer match.
# Answers are always integers so no fuzzy bands — clean verifiable signal.

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

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/gsm8k_grpo.jsonl"

MODEL_PATH = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"

OUTPUT_DIR   = Path("./output/grpo/gsm8k")
LOGS_DIR     = Path("./logs_gsm8k")

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    "You are a math problem solver.\n"
    f"Think step-by-step inside {REASONING_START} ... {REASONING_END}, "
    f"then write only the final numeric answer inside {SOLUTION_START} ... {SOLUTION_END}."
)

# ---------------------------------------------------------------------------
# Logging callback — saves every logged step to JSONL for plotting
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
            "Run  python grpo/prepare_gsm8k.py  first."
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
# Reward functions
# ---------------------------------------------------------------------------

MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

MATCH_APPROXIMATE = re.compile(
    rf"{re.escape(REASONING_START)}|{re.escape(REASONING_END)}"
    rf"|{re.escape(SOLUTION_START)}|{re.escape(SOLUTION_END)}"
)


def reward_format(completions, **kwargs):
    """
    2.0 — full format present
    0.0 — missing
    Smaller than answer reward so correctness is the dominant signal.
    """
    scores = []
    for completion in completions:
        r = completion[0]["content"]
        scores.append(2.0 if MATCH_FORMAT.search(r) is not None else 0.0)
    return scores


def reward_format_partial(completions, **kwargs):
    """±0.25 per format token — keeps the model anchored to the template."""
    tokens = [REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END]
    scores = []
    for completion in completions:
        r = completion[0]["content"]
        score = sum(0.25 if r.count(t) == 1 else -0.25 for t in tokens)
        scores.append(score)
    return scores


def reward_correctness(prompts, completions, answer, **kwargs):
    """
    3.0 — exact integer match (primary signal)
   -0.5 — wrong answer
    0.0 — couldn't extract an answer (format missing)
    GSM8k answers are always integers so no fuzzy bands needed.
    """
    scores = []
    for completion, true_ans in zip(completions, answer):
        r = completion[0]["content"]
        m = MATCH_FORMAT.search(r)
        if m is None:
            scores.append(0.0)
            continue
        guess = m.group(1).strip().replace(",", "")
        true_clean = str(true_ans).strip().replace(",", "")
        if guess == true_clean:
            scores.append(3.0)
        else:
            # also accept if they match as integers (e.g. "72" vs "72.0")
            try:
                scores.append(3.0 if int(float(guess)) == int(float(true_clean)) else -0.5)
            except (ValueError, OverflowError):
                scores.append(-0.5)
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
        # Learning
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        # Batching — 4 per device × 4 GPUs × 4 accum = 64 effective
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_completion_length=1024,
        num_train_epochs=1,
        # GRPO
        num_generations=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=4096,
        generation_batch_size=32,
        # Stability
        max_grad_norm=0.1,
        bf16=True,
        gradient_checkpointing=True,
        # Logging / saving
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_dir=str(LOGS_DIR),
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    n_steps = (len(dataset) // (training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs

    print("\n" + "=" * 60)
    print("GRPO GSM8k TRAINING CONFIG")
    print("=" * 60)
    print(f"Model:              {MODEL_PATH}")
    print(f"Dataset:            {DATASET_PATH}  ({len(dataset):,} examples)")
    print(f"Rewards:            format(2.0) + partial(±0.25) + correctness(3.0/-0.5)")
    print(f"Effective batch:    {training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps}")
    print(f"Est. steps:         ~{n_steps}")
    print(f"Output dir:         {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    save_path = "/home/zbibm/Reasoning-Vectors/output/grpo/gsm8k"

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

    print("Starting GRPO GSM8k training ...\n")
    train_result = trainer.train()

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
    print("GRPO GSM8k TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss:  {train_result.training_loss:.4f}")
    print(f"Steps:       {trainer.state.global_step}")
    print(f"Model:       {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
