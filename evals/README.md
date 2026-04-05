# Math Evaluation Suite

Evaluates models on 12 math reasoning benchmarks using [lm-evaluation-harness](../lm-evaluation-harness) with vLLM as the backend.

## Scripts

| Script | Purpose |
|---|---|
| `run_math_evals.sh` | Full eval: runs all tasks on base + SFT model across 4 GPUs in parallel |
| `validate_evals.sh` | Smoke test: 3 samples per task on base model (GPU 0 only) to catch breakage before a full run |

## Tasks

### GSM8K family — elementary word problems

| Task | Description |
|---|---|
| `gsm8k` | Standard GSM8K (8.5k grade-school math problems) |
| `gsm8k_cot` | GSM8K with few-shot chain-of-thought prompting |
| `gsm8k_platinum_cot_zeroshot` | Reannotated, cleaner GSM8K with reduced label noise |
| `gsm_plus` | Harder GSM8K variants with perturbed conditions |

### MATH / competition math — free-form with `\boxed{}`

Uses `minerva_math` variants throughout (not `hendrycks_math`) because the Minerva tasks use `math_verify` which correctly parses `\boxed{}` output. The old Hendrycks extractor expects raw `$answer$` format and always scores 0 for models trained with `\boxed{}` prompts.

| Task | Description |
|---|---|
| `minerva_math500` | 500-problem sample, all levels, all topics |
| `minerva_math_algebra` | Algebra subset |
| `minerva_math_intermediate_algebra` | Intermediate algebra subset |
| `minerva_math_num_theory` | Number theory subset |
| `minerva_math_geometry` | Geometry subset |
| `minerva_math_counting_and_prob` | Counting & probability subset |

### Multi-choice / structured

| Task | Description |
|---|---|
| `agieval_math` | Gaokao math problems (Chinese college entrance, translated) |
| `agieval_sat_math` | SAT math problems |

## Usage

```bash
# Smoke test first (fast, ~15 min)
bash evals/validate_evals.sh

# Full eval on base + SFT model
bash evals/run_math_evals.sh

# Results land in eval_results/summary.md
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GPU_IDS` | `0,1,2,3` | Comma-separated GPU indices to use |
| `BATCH_SIZE` | `16` | vLLM batch size |
| `GPU_MEMORY_UTILIZATION` | `0.4` | vLLM memory fraction per GPU |
| `RESULTS_ROOT` | `./eval_results` | Where results are written |
| `LOG_SAMPLES` | `0` | Set to `1` to save per-sample outputs |
