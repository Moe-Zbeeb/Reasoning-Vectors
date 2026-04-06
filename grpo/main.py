# accelerate launch --num_processes 4 --mixed_precision bf16 grpo/main.py

import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/math_grpo.jsonl"
MODEL_PATH   = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"
SAVE_PATH    = "/home/zbibm/Reasoning-Vectors/models/output/grpo/qwen2.5-3b-math-grpo"
OUTPUT_DIR   = Path("./output_grpo")
LOGS_DIR     = Path("./logs_grpo")

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
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
            "Run  python datasets/prepare.py  first."
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

def _extract_boxed(text: str):
    """Extract content of last \\boxed{} from model output."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth, i = 0, idx + len(r"\boxed{") - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx + len(r"\boxed{"):i].strip()
        i += 1
    return None


def reward_correctness(prompts, completions, answer, **kwargs):
    """
    1.0  — math_verify confirms answer matches
    0.0  — wrong answer, no \\boxed{}, or parse error
    """
    scores = []
    for c, true_ans in zip(completions, answer):
        guess = _extract_boxed(c[0]["content"])
        if guess is None:
            scores.append(0.0)
            continue
        try:
            pg = parse(guess)
            pt = parse(str(true_ans))
            if pg is not None and pt is not None:
                scores.append(1.0 if verify(pt, pg) else 0.0)
            else:
                scores.append(1.0 if guess.strip() == str(true_ans).strip() else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


def reward_format(prompts, completions, answer, **kwargs):
    """
    0.1  — response contains a \\boxed{} expression (format preserved)
    0.0  — no \\boxed{} found

    Small bonus to prevent the model from dropping the SFT-learned format
    under GRPO pressure. Intentionally tiny so correctness dominates.
    """
    return [0.1 if _extract_boxed(c[0]["content"]) is not None else 0.0
            for c in completions]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    raw_ds = load_grpo_dataset(DATASET_PATH)

    # 90/10 train/eval split — eval runs every quarter epoch to catch early divergence
    split    = raw_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].map(format_example, remove_columns=split["train"].column_names)
    eval_ds  = split["test"].map(format_example,  remove_columns=split["test"].column_names)

    # Steps per epoch and quarter-epoch eval cadence
    EFFECTIVE_BATCH = 4 * 4 * 2          # n_gpus × per_device_batch × grad_accum
    steps_per_epoch = len(train_ds) // EFFECTIVE_BATCH
    eval_steps      = max(10, steps_per_epoch // 4)

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
        warmup_steps=50,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,       # effective = 4×4×2 = 32
        max_completion_length=1024,
        num_train_epochs=1,
        num_generations=8,                   # 8 rollouts per prompt
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=4096,
        generation_batch_size=32,            # per_device_batch(4) × num_generations(8)
        max_grad_norm=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=eval_steps,               # every ~quarter epoch
        report_to="none",
        logging_dir=str(LOGS_DIR),
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    print("\n" + "=" * 60)
    print("GRPO TRAINING CONFIG")
    print("=" * 60)
    print(f"Model:              {MODEL_PATH}")
    print(f"Dataset:            {DATASET_PATH}  ({len(train_ds):,} train / {len(eval_ds):,} eval)")
    print(f"Rewards:            correctness (1.0) + format (0.1)")
    print(f"Per-device batch:   {training_args.per_device_train_batch_size}")
    print(f"Grad accumulation:  {training_args.gradient_accumulation_steps}")
    print(f"Effective batch:    {EFFECTIVE_BATCH}")
    print(f"Rollouts/prompt:    {training_args.num_generations}")
    print(f"Steps/epoch:        ~{steps_per_epoch}")
    print(f"Eval every:         {eval_steps} steps  (~quarter epoch)")
    print(f"Save path:          {SAVE_PATH}")
    print("=" * 60 + "\n")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_correctness, reward_format],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonlLoggerCallback(LOGS_DIR / "train_metrics.jsonl"))

    print("Starting GRPO training ...\n")
    train_result = trainer.train()

    print("\nSaving model ...")
    trainer.save_model(SAVE_PATH)
    trainer.state.save_to_json(str(LOGS_DIR / "trainer_state.json"))

    summary = {
        "model_path": MODEL_PATH,
        "dataset": DATASET_PATH,
        "num_train_samples": len(train_ds),
        "num_eval_samples": len(eval_ds),
        "final_loss": train_result.training_loss,
        "global_step": trainer.state.global_step,
        "model_saved_to": SAVE_PATH,
    }
    (LOGS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("GRPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss:  {train_result.training_loss:.4f}")
    print(f"Steps:       {trainer.state.global_step}")
    print(f"Model:       {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
