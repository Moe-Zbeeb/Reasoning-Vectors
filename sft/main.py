#  accelerate launch --num_processes 4 --mixed_precision bf16 sft/main.py

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/math_sft.jsonl"
MODEL_PATH   = "/home/zbibm/Reasoning-Vectors/models/qwen2.5-3b"
SAVE_PATH    = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"
OUTPUT_DIR   = Path("./output_sft")
LOGS_DIR     = Path("./logs_sft")


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


print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(dataset):,} examples")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
metrics_log_path = LOGS_DIR / "train_metrics.jsonl"
summary_log_path = LOGS_DIR / "run_summary.json"

config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,       # effective batch = 4 GPUs × 4 × 8 = 128
    learning_rate=5e-6,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=2,
    save_strategy="no",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    bf16=True,
    max_grad_norm=1.0,
    max_length=2048,
    completion_only_loss=True,
    seed=42,
    remove_unused_columns=True,
    dataloader_num_workers=2,
    report_to="none",
    logging_dir=str(LOGS_DIR),
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model=MODEL_PATH,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.add_callback(JsonlLoggerCallback(metrics_log_path))

print("\n" + "=" * 60)
print("SFT TRAINING CONFIG")
print("=" * 60)
print(f"Model:              {MODEL_PATH}")
print(f"Dataset:            {DATASET_PATH}  ({len(dataset):,} examples)")
print(f"Epochs:             3")
print(f"Per-device batch:   4")
print(f"Grad accumulation:  8")
print(f"Effective batch:    128  (4 GPUs × 4 × 8)")
print(f"Learning rate:      5e-6")
print(f"Max length:         2048")
print(f"Completion-only:    True")
print(f"Save path:          {SAVE_PATH}")
print("=" * 60 + "\n")

summary = {
    "model_path": MODEL_PATH,
    "dataset_path": DATASET_PATH,
    "num_examples": len(dataset),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 128,
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "max_length": 2048,
    "completion_only_loss": True,
}
summary_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Starting training...\n")
train_result = trainer.train()

print("\nSaving model...")
trainer.save_model(SAVE_PATH)
trainer.state.save_to_json(str(LOGS_DIR / "trainer_state.json"))

summary.update({
    "final_loss": train_result.training_loss,
    "global_step": trainer.state.global_step,
    "model_saved_to": SAVE_PATH,
})
summary_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 60)
print("SFT TRAINING COMPLETE")
print("=" * 60)
print(f"Final loss:  {train_result.training_loss:.4f}")
print(f"Steps:       {trainer.state.global_step}")
print(f"Model:       {SAVE_PATH}")
print("=" * 60)
