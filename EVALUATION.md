# Evaluation Guide

This guide covers how to evaluate models in this repo on math benchmarks using the local `lm-evaluation-harness` checkout.

## Models

| Model | Path | Description |
|-------|------|-------------|
| Base 1.5B | `models/qwen1.5B` | Qwen 1.5B base |
| Base 3B | `models/qwen2.5-3b` | Qwen 2.5 3B base |
| Base 7B | `models/qwen2.5-7b` | Qwen 2.5 7B base |
| SFT 3B | `models/output/sft/qwen2.5-3b-math` | Fine-tuned on MathInstruct CoT |

## Benchmarks

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| `gsm8k` | Generative | flexible-extract | Grade school math word problems |
| `gsm8k_cot` | Generative | flexible-extract | GSM8K with chain-of-thought |
| `gsm_plus` | Generative | flexible-extract | Harder GSM8K variants |
| `minerva_math500` | Generative | math_verify | MATH competition problems (500 subset) |
| `hendrycks_math500` | Generative | math_verify | MATH benchmark (500 subset) |
| `agieval_math` | Generative | acc | AGIEval math problems |
| `agieval_sat_math` | Multiple choice | acc_norm | SAT math problems |

## Environment

```bash
conda activate reasoning
cd ~/Reasoning-Vectors
```

## Running All Evaluations

The main eval script runs all 7 benchmarks on both base and SFT models in parallel across 4 GPUs and writes a summary table:

```bash
bash run_math_evals.sh
```

Results are saved to `eval_results/` and a comparison table is written to `eval_results/summary.md`.

### Options

```bash
# Custom GPU memory utilization (default 0.4)
GPU_MEMORY_UTILIZATION=0.6 bash run_math_evals.sh

# Log per-sample outputs
LOG_SAMPLES=1 bash run_math_evals.sh

# Specific GPUs only
GPU_IDS=0,1 bash run_math_evals.sh
```

## Running a Single Task

```bash
cd lm-evaluation-harness

python -m lm_eval run \
  --model vllm \
  --model_args "pretrained=/home/zbibm/Reasoning-Vectors/models/qwen2.5-3b,dtype=auto,gpu_memory_utilization=0.5" \
  --tasks gsm8k \
  --apply_chat_template \
  --batch_size 16 \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/quick
```

## Smoke Test (10 samples)

Quick sanity check before a full run:

```bash
cd lm-evaluation-harness

CUDA_VISIBLE_DEVICES=0 python -m lm_eval run \
  --model vllm \
  --model_args "pretrained=/home/zbibm/Reasoning-Vectors/models/output/sft/qwen2.5-3b-math,dtype=auto,gpu_memory_utilization=0.5" \
  --tasks gsm8k_cot \
  --apply_chat_template \
  --batch_size 16 \
  --limit 10 \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/smoke
```

## Reading Results

After `run_math_evals.sh` completes:

```bash
cat eval_results/summary.md
```

Per-task result JSONs are at `eval_results/<model_name>/<task>/results_*.json`.

## Notes

- All generative tasks use `--apply_chat_template` — the models have Qwen chat templates
- `minerva_math500` and `hendrycks_math500` use `math_verify` (sympy equivalence) as primary metric, not string exact match
- `agieval_sat_math` uses logprob-based multiple choice — no generation needed
- `agieval_math` has `max_gen_toks=512` (increased from default 32 to allow full reasoning chains)
- `gsm8k_cot` strict-match regex accepts `The answer is X` with optional trailing period
