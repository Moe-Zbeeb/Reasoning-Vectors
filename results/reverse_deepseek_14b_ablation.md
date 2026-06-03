# Reverse Delta Ablation: DeepSeek 14B

Date: 2026-06-03

## Construction

`theta_negative = theta_base - (theta_RL - theta_SFT)`

Equivalent merge:

`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B + zbeeb/deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep - RabotniKuma/Fast-Math-R1-14B`

Merge config:

`evaluation/mergekit/reverse_delta_alpha1/deepseek_fast_math_r1_14b.yml`

## Evaluation Setup

| Benchmark | Max generation tokens | Temperature | Engine |
| --- | ---: | ---: | --- |
| AIME24x8 | 8000 | 0.6 | LightEval + vLLM |
| AIME25x8 | 8000 | 0.6 | LightEval + vLLM |
| MATH500 | 3000 | 0 | LightEval + vLLM |

Hardware: 8x NVIDIA H200.

## Results

| Model | AIME24x8 | AIME25x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Base DeepSeek 14B | 44.6 | 32.9 | 71.8 | 49.8 |
| SFT zbeeb 10ep | 43.3 | 30.4 | 69.6 | 47.8 |
| RL Fast-Math-R1 | 54.2 | 33.8 | 80.0 | 56.0 |
| Merged Base+RL-SFT | 52.5 | 37.5 | 80.2 | 56.7 |
| Negative Base-RL+SFT | 0.0 | 6.7 | 34.6 | 13.8 |

## Delta vs Base

| Model | AIME24x8 | AIME25x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Negative Base-RL+SFT | -44.6 | -26.2 | -37.2 | -36.0 |

## Notes

The negative ablation is substantially worse than the base model across AIME24x8, AIME25x8, and MATH500. This supports the interpretation that the positive delta `theta_RL - theta_SFT` contains useful reasoning behavior, while negating the same direction suppresses it.

LightEval result files were written under:

`/workspace/reasoning_vectors_ablation/eval_runs/reverse_deepseek_14b/`
