# Reverse Delta Ablation: Qwen GSM8K 1.5B

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
| AIME24x8 | 8000 | 0 | LightEval + vLLM |
| AIME25x8 | 8000 | 0 | LightEval + vLLM |
| MATH500 | 3000 | 0 | LightEval + vLLM |

Hardware: 8x NVIDIA H200.

## Results

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-Math-1.5B` | 3.8 | 3.8 | 15.2 | 7.6 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | 2.9 | 3.3 | 11.0 | 5.7 |
| `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` | 4.2 | 5.8 | 48.6 | 19.5 |
| `reasoning_delta_alpha1/qwen_gsm8k_1p5b` | 4.6 | 4.2 | 44.4 | 17.7 |
| Negative Base-GRPO+SFT | 0.0 | 10.0 | 18.2 | 9.4 |

## Delta vs Base

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Negative Base-GRPO+SFT | -3.8 | +6.2 | +3.0 | +1.8 |

## Notes

The negative Qwen GSM8K ablation does not uniformly roll back below base under this LightEval setup. It drops AIME25x8 to 0.0 and remains far below the GRPO and positive-delta checkpoints on MATH500, but it improves over base on AIME24x8 and MATH500.

LightEval result files were written under:

`/workspace/reasoning_vectors_ablation/eval_runs/reverse_qwen_gsm8k_1p5b/`
