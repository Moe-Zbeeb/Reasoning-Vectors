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
MODEL_PATH   = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-7b-math"
SAVE_PATH    = "/home/zbibm/Reasoning-Vectors/models/output/grpo/qwen2.5-7b-math-grpo"
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

BOXED_RE = re.compile(r"\\boxed\{", re.DOTALL)


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


def reward_format(completions, **kwargs):
    """2.0 if response contains \\boxed{}, 0 otherwise."""
    return [
        2.0 if BOXED_RE.search(c[0]["content"]) else 0.0
        for c in completions
    ]


def reward_correctness(prompts, completions, answer, **kwargs):
    """
    3.0  — math_verify confirms answer matches
   -0.5  — wrong answer
    0.0  — no \\boxed{} found / parse error
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
                scores.append(3.0 if verify(pt, pg) else -0.5)
            else:
                scores.append(3.0 if guess.strip() == str(true_ans).strip() else -0.5)
        except Exception:
            scores.append(0.0)
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
        per_device_train_batch_size=2,       # 7B is larger
        gradient_accumulation_steps=8,       # effective = 4×2×8 = 64
        max_completion_length=1024,
        num_train_epochs=1,
        num_generations=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=4096,
        generation_batch_size=16,            # per_device_batch(2) × num_generations(8)
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

    n_steps = len(dataset) // (training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps)

    print("\n" + "=" * 60)
    print("GRPO TRAINING CONFIG")
    print("=" * 60)
    print(f"Model:              {MODEL_PATH}")
    print(f"Dataset:            {DATASET_PATH}  ({len(dataset):,} examples)")
    print(f"Rewards:            format(\\boxed{{}}) + correctness(math_verify)")
    print(f"Per-device batch:   {training_args.per_device_train_batch_size}")
    print(f"Grad accumulation:  {training_args.gradient_accumulation_steps}")
    print(f"Effective batch:    {training_args.per_device_train_batch_size * 4 * training_args.gradient_accumulation_steps}")
    print(f"Est. steps/epoch:   ~{n_steps}")
    print(f"Save path:          {SAVE_PATH}")
    print("=" * 60 + "\n")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_correctness],
        args=training_args,
        train_dataset=dataset,
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
        "num_samples": len(dataset),
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
