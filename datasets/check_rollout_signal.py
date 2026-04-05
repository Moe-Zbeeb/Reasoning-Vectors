"""
Quick check: how many of 20 random NuminaMath olympiad problems
get at least 1 correct rollout from the SFT model?

Run: python datasets/check_rollout_signal.py
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams

MODEL_PATH   = "/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math"
N_PROBLEMS   = 20
N_ROLLOUTS   = 8
SEED         = 42

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

rng = random.Random(SEED)

def extract_boxed(text: str):
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

def is_correct(guess: str, gold: str) -> bool:
    try:
        pg, pt = parse(guess), parse(gold)
        if pg is not None and pt is not None:
            return bool(verify(pt, pg))
        return guess.strip() == gold.strip()
    except Exception:
        return False

print("Loading NuminaMath olympiad problems...")
ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
ds = ds.filter(lambda x: x["source"] == "olympiads", num_proc=4)
print(f"  {len(ds):,} olympiad problems")

# sample N_PROBLEMS
indices = rng.sample(range(len(ds)), N_PROBLEMS)
problems = [ds[i] for i in indices]

# extract answers
def extract_boxed_from_sol(sol):
    idx = sol.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth, i = 0, idx + len(r"\boxed{") - 1
    while i < len(sol):
        if sol[i] == "{": depth += 1
        elif sol[i] == "}":
            depth -= 1
            if depth == 0:
                return sol[idx + len(r"\boxed{"):i].strip()
        i += 1
    return None

problems = [p for p in problems if extract_boxed_from_sol(p["solution"])]
print(f"  {len(problems)} problems with extractable answers")

print(f"\nLoading SFT model from {MODEL_PATH} ...")
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    gpu_memory_utilization=0.7,
    max_model_len=4096,
    tensor_parallel_size=1,
)

sampling = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    n=N_ROLLOUTS,
)

# Build prompts using chat template
tokenizer = llm.get_tokenizer()
prompts = []
gold_answers = []
for p in problems:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": p["problem"].strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)
    gold_answers.append(extract_boxed_from_sol(p["solution"]))

print(f"\nRunning {N_ROLLOUTS} rollouts on {len(prompts)} problems...")
outputs = llm.generate(prompts, sampling)

print("\n" + "="*60)
print(f"ROLLOUT SIGNAL CHECK  (SFT model, {N_ROLLOUTS} rollouts each)")
print("="*60)

problems_with_signal = 0
results = []
for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
    rollout_texts = [o.text for o in out.outputs]
    guesses = [extract_boxed(t) for t in rollout_texts]
    correct = sum(1 for g in guesses if g and is_correct(g, gold))
    has_signal = correct > 0
    if has_signal:
        problems_with_signal += 1
    results.append((i, correct, has_signal))
    print(f"  Problem {i+1:2d}: {correct}/{N_ROLLOUTS} correct  {'✓' if has_signal else '✗'}")

print("\n" + "="*60)
print(f"Problems with ≥1 correct rollout: {problems_with_signal}/{len(prompts)}")
print(f"Solve rate: {problems_with_signal/len(prompts)*100:.0f}%")
print("="*60)
print()
if problems_with_signal < 4:
    print("→ < 4/20 problems have signal. Cap NuminaMath at 1,000 or drop to AMC-only.")
elif problems_with_signal >= 8:
    print("→ ≥ 8/20 problems have signal. 1,500 is safe, could keep 2,000.")
else:
    print("→ 4-7/20 problems have signal. 1,500 is a reasonable cap.")
