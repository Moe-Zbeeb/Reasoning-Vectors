#  accelerate launch --num_processes 4 --mixed_precision bf16 sft/main.py                                                                                    

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/math_sft_47k.jsonl"
MODEL_PATH = "/home/zbibm/Reasoning-Vectors/models/qwen2.5-3b"
OUTPUT_DIR = Path("./output")
LOGS_DIR = Path("./logs")


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
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)
print(f"Loaded {len(dataset)} examples")

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
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
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
print(f"Dataset: {DATASET_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Examples: {len(dataset)}")
print("Epochs: 1")
print("Batch size: 4")
print("Gradient accumulation: 8")
print("Effective batch size: 128 (4 GPUs x 4 x 8)")
print("Learning rate: 5e-6")
print("Weight decay: 0.01")
print("LR scheduler: cosine")
print("Warmup steps: 20")
print("Max length: 2048")
print("Completion-only loss: True")
print(f"Metrics log: {metrics_log_path}")
print("=" * 60 + "\n")

summary = {
    "dataset_path": DATASET_PATH,
    "model_path": MODEL_PATH,
    "num_examples": len(dataset),
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 128,
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "warmup_steps": 20,
    "max_length": 2048,
    "completion_only_loss": True,
    "bf16": True,
    "metrics_log": str(metrics_log_path),
}
summary_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Starting training...\n")
train_result = trainer.train()

print("\nSaving model...")
trainer.save_model("/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math")
trainer.state.save_to_json(str(LOGS_DIR / "trainer_state.json"))

final_summary = {
    **summary,
    "final_loss": train_result.training_loss,
    "global_step": trainer.state.global_step,
}
summary_log_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Model saved to: models/output/sft/qwen2.5-3b-math")
print(f"Trainer state saved to: {LOGS_DIR / 'trainer_state.json'}")
print("=" * 60)
