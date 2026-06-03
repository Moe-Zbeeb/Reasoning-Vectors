# Reverse Delta Ablation: Qwen GSM8K 1.5B at 3000 Tokens

Date: 2026-06-03

## Construction

`theta_negative = theta_base - (theta_RL - theta_SFT)`

Equivalent merge:

`Qwen/Qwen2.5-Math-1.5B + michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT - michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO`

Merge config:

`evaluation/mergekit/reverse_delta_alpha1/qwen_gsm8k_1p5b.yml`

## Evaluation Setup

| Benchmark | Max generation tokens | Temperature | Engine |
| --- | ---: | ---: | --- |
| AIME24x8 | 3000 | 0 | LightEval + vLLM |
| AIME25x8 | 3000 | 0 | LightEval + vLLM |
| MATH500 | 3000 | 0 | LightEval + vLLM |

Hardware: 8x NVIDIA H200.

## Results

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-Math-1.5B` | 3.8 | 3.8 | 15.2 | 7.6 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | 2.9 | 3.3 | 11.0 | 5.7 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` | 4.2 | 5.8 | 48.6 | 19.5 |
| `reasoning_delta_alpha1/qwen_gsm8k_1p5b` | 4.6 | 4.2 | 44.4 | 17.7 |
| Negative Base-GRPO+SFT, 3000 tokens | 0.0 | 10.0 | 16.8 | 8.9 |

## Delta vs Base

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Negative Base-GRPO+SFT, 3000 tokens | -3.8 | +6.2 | +1.6 | +1.3 |

## Notes

At 3000 max generation tokens for both AIME and MATH500, the negative Qwen GSM8K ablation still scores above the listed base average. It collapses AIME25x8, but improves AIME24x8 and MATH500 relative to the listed base numbers.

LightEval result files were written under:

`/workspace/reasoning_vectors_ablation/eval_runs/reverse_qwen_gsm8k_1p5b_3000/`
