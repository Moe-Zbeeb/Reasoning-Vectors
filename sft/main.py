#!/usr/bin/env python3
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_PATH = "/home/zbibm/Reasoning-Vectors/datasets/openr1_25k.jsonl"
MODEL_PATH = "/home/zbibm/Reasoning-Vectors/models/qwen1.5B"

print("Loading dataset...")
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)
print(f"Loaded {len(dataset)} examples")

print("Loading tokenizer and formatting dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = dataset.map(format_chat, num_proc=4)
print(f"Formatted {len(dataset)} examples")

config = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=5,
    logging_steps=2,
    save_steps=50,
    save_total_limit=None,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    bf16=True,
    max_grad_norm=1.0,
    seed=42,
    remove_unused_columns=True,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    logging_dir="./logs",
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model=MODEL_PATH,
    args=config,
    train_dataset=dataset,
)

print("\n" + "=" * 60)
print(f"Dataset: {DATASET_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Examples: {len(dataset)}")
print("Epochs: 1")
print("Batch size: 16")
print("Gradient accumulation: 2")
print("Effective batch size: 32")
print("Learning rate: 1e-5")
print("Warmup steps: 5")
print("Save steps: 50")
print("=" * 60 + "\n")

print("Starting training...\n")
train_result = trainer.train()

print("\nSaving model...")
trainer.save_model("./output/final-model")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Final loss: {train_result.training_loss:.4f}")
print("Model saved to: ./output/final-model")
print("=" * 60)
