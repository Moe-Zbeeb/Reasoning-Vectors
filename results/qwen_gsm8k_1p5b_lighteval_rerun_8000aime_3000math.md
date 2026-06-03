# Qwen GSM8K 1.5B LightEval Rerun: 8000 AIME and 3000 MATH500

Date: 2026-06-03

## Objective

Rerun the four Qwen GSM8K 1.5B checkpoints:

`base`, `SFT`, `GRPO`, and `base + (GRPO - SFT)`.

The positive merged checkpoint was constructed locally on the pod from:

`Qwen/Qwen2.5-Math-1.5B - michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT + michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO`

## Evaluation Setup

| Benchmark | Max generation tokens | Temperature | Engine |
| --- | ---: | ---: | --- |
| AIME24x8 | 8000 | 0 | LightEval + vLLM |
| AIME25x8 | 8000 | 0 | LightEval + vLLM |
| MATH500 | 3000 | 0 | LightEval + vLLM |

Hardware: 8x NVIDIA H200.

Run ID:

`qwen_gsm8k_four_models_8000aime_3000math_20260603_1720`

Result directory:

`/workspace/reasoning_vectors_ablation/eval_runs/qwen_gsm8k_four_models_8000aime_3000math_20260603_1720/`

## Reference Numbers

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-Math-1.5B` | 3.8 | 3.8 | 15.2 | 7.6 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | 2.9 | 3.3 | 11.0 | 5.7 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` | 4.2 | 5.8 | 48.6 | 19.5 |
| `reasoning_delta_alpha1/qwen_gsm8k_1p5b` | 4.6 | 4.2 | 44.4 | 17.7 |

## LightEval Rerun Results

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-Math-1.5B` | 2.9 | 4.2 | 54.6 | 20.6 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | 6.7 | 10.0 | 49.8 | 22.2 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` | 3.3 | 10.0 | 67.2 | 26.8 |
| Local merged `base + (GRPO - SFT)` | 10.0 | 6.7 | 67.2 | 28.0 |

## Shard Scores

| Model | AIME25 shards | AIME24 shards |
| --- | --- | --- |
| `Qwen/Qwen2.5-Math-1.5B` | 0.0, 0.0, 6.7, 5.0 | 3.3, 6.7, 0.0, 6.7 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | 13.3, 0.0, 13.3, 0.0 | 13.3, 6.7, 13.3, 6.7 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` | 0.0, 6.7, 6.7, 0.0 | 13.3, 6.7, 13.3, 6.7 |
| Local merged `base + (GRPO - SFT)` | 13.3, 6.7, 13.3, 6.7 | 6.7, 6.7, 6.7, 6.7 |

## Notes

This LightEval rerun did not reproduce the reference MATH500 scale. MATH500 is much higher for all four models under the current `ablation:math500` task and prompt in `runpod_lighteval_ablation_tasks.py`.

The published `reasoning_delta_alpha1/qwen_gsm8k_1p5b` checkpoint was not accessible from the pod without authentication, so the positive merge was constructed locally using the checked-in mergekit recipe.
