# Evaluating `qwen1.5bMathSft`

This repo now contains:

- Fine-tuned model: `/home/zbibm/Reasoning-Vectors/output/qwen1.5bMathSft`
- Evaluation framework: `/home/zbibm/Reasoning-Vectors/lm-evaluation-harness`

This guide shows how to evaluate the model on GSM8K and several math benchmarks using the local `lm-evaluation-harness` checkout.

## 1. Environment

Use the `reasoning` environment:

```bash
conda activate reasoning
cd ~/Reasoning-Vectors/lm-evaluation-harness
```

Install the harness with Hugging Face support if you have not already:

```bash
pip install -e ".[hf]"
```

## 2. Model Path

All commands below use this local model path:

```bash
MODEL=/home/zbibm/Reasoning-Vectors/output/qwen1.5bMathSft
```

The model is instruction-tuned and has a chat template, so use `--apply_chat_template` for generative math tasks.

## 3. Quick Smoke Test

Before a full run, test a few examples:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k_cot \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --limit 10 \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/smoke_gsm8k_cot
```

## 4. Recommended Main Evaluations

### GSM8K

Direct GSM8K:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/gsm8k \
  --log_samples
```

Chain-of-thought GSM8K:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k_cot \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/gsm8k_cot \
  --log_samples
```

### Math Benchmarks

Minerva MATH group:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks minerva_math \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/minerva_math \
  --log_samples
```

AGIEval MATH:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks agieval_math \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/agieval_math \
  --log_samples
```

AGIEval SAT Math:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks agieval_sat_math \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/agieval_sat_math \
  --log_samples
```

MathQA:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks mathqa \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/mathqa \
  --log_samples
```

## 5. Suggested Bundle

If you want one run covering the main math set:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k gsm8k_cot minerva_math agieval_math agieval_sat_math mathqa \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/math_suite \
  --log_samples
```

## 6. Notes On These Tasks

- `gsm8k` and `gsm8k_cot` are generative math word-problem tasks.
- `minerva_math` is a grouped benchmark over several MATH categories.
- `agieval_math` and `agieval_sat_math` are generative math tasks with task-specific answer processing.
- `mathqa` is multiple-choice. It is useful, but it is not the same style as your SFT training data.

## 7. Multi-GPU Option

Your machine has 4x A100 80GB GPUs. For this 1.5B model, a single GPU is usually enough.

If you want data-parallel evaluation across multiple GPUs, use `accelerate`:

```bash
accelerate launch -m lm_eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k_cot minerva_math \
  --apply_chat_template \
  --batch_size auto \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/multi_gpu_eval \
  --log_samples
```

For a 1.5B model, start with single-process evaluation first unless you specifically want higher throughput.

## 8. Check Available Tasks

To inspect the task names in this checkout:

```bash
lm-eval ls tasks | rg 'gsm8k|minerva|mathqa|agieval'
```

## 9. Where Results Go

Each run writes results under the `--output_path` directory you choose.

Those folders will typically contain aggregate metrics and, if you use `--log_samples`, per-sample outputs for later inspection.

## 10. Recommended First Run

Start with this:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=$MODEL,dtype=bfloat16 \
  --tasks gsm8k_cot minerva_math \
  --apply_chat_template \
  --batch_size auto \
  --device cuda \
  --output_path /home/zbibm/Reasoning-Vectors/eval_results/first_math_eval \
  --log_samples
```

That gives you one standard grade-school benchmark and one stronger math benchmark group.
