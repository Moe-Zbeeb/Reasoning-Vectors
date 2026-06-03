# Random Vector Controls: DeepSeek 14B and Qwen GSM8K 1.5B

Date: 2026-06-03

## Construction

Control formula:

`theta_random = theta_base + alpha * v_random`

For each floating tensor, `v_random` was sampled with seed `0` and rescaled to match the L2 norm of the corresponding true delta tensor:

`theta_RL - theta_SFT`

Settings:

| Parameter | Value |
| --- | --- |
| Alpha | 1.0 |
| Random seed | 0 |
| Scaling | Per-tensor L2 norm matching |
| Dtype | bfloat16 output |

## Model Pairs

| Family | Base | SFT | RL |
| --- | --- | --- | --- |
| DeepSeek 14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `zbeeb/deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep` | `RabotniKuma/Fast-Math-R1-14B` |
| Qwen GSM8K 1.5B | `Qwen/Qwen2.5-Math-1.5B` | `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT` | `michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO` |

## Evaluation Setup

| Benchmark | Max generation tokens | Temperature | Engine |
| --- | ---: | ---: | --- |
| AIME24x8 | 8000 | 0 | LightEval + vLLM |
| AIME25x8 | 8000 | 0 | LightEval + vLLM |
| MATH500 | 3000 | 0 | LightEval + vLLM |

Hardware: 8x NVIDIA H200.

Run ID:

`random_vector_deepseek_qwen_20260603_1804`

Result directory:

`/workspace/reasoning_vectors_ablation/eval_runs/random_vector_deepseek_qwen_20260603_1804/`

## Random Vector Results

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 14B random vector | 33.8 | 51.2 | 74.4 | 53.1 |
| Qwen GSM8K 1.5B random vector | 1.7 | 6.7 | 56.2 | 21.5 |

## Shard Scores

| Model | AIME25 shards | AIME24 shards |
| --- | --- | --- |
| DeepSeek 14B random vector | 33.3, 40.0, 33.3, 28.3 | 53.3, 46.7, 53.3, 51.7 |
| Qwen GSM8K 1.5B random vector | 0.0, 0.0, 6.7, 0.0 | 6.7, 6.7, 6.7, 6.7 |

## Comparison Notes

DeepSeek reference table:

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Base DeepSeek 14B | 32.9 | 44.6 | 71.8 | 49.8 |
| SFT zbeeb 10ep | 30.4 | 43.3 | 69.6 | 47.8 |
| RL Fast-Math-R1 | 33.8 | 54.2 | 80.0 | 56.0 |
| Merged `base + (RL - SFT)` | 37.5 | 52.5 | 80.2 | 56.7 |
| Random vector | 33.8 | 51.2 | 74.4 | 53.1 |

Qwen same-run LightEval comparison:

| Model | AIME25x8 | AIME24x8 | MATH500 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Same-run base | 2.9 | 4.2 | 54.6 | 20.6 |
| Same-run SFT | 6.7 | 10.0 | 49.8 | 22.2 |
| Same-run GRPO | 3.3 | 10.0 | 67.2 | 26.8 |
| Same-run merged `base + (GRPO - SFT)` | 10.0 | 6.7 | 67.2 | 28.0 |
| Random vector | 1.7 | 6.7 | 56.2 | 21.5 |

The random vector control is not a clean no-improvement result. DeepSeek random is above the listed base average, and Qwen random is slightly above the same-run base average under the current LightEval MATH500 task.

Qwen MATH500 should be compared to the same-run LightEval table, not the older reference table, because the current `ablation:math500` prompt/task produced a different MATH500 scale in the four-model rerun.
