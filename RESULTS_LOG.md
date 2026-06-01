# Results Log

## public checkpoints

`theta_out = theta_base + (theta_GRPO - theta_SFT)`

**Qwen GSM8K 1.5B great to use**

| Model | AIME25x8 | AMC23x8 | AIME24x8 | Minerva Math | OlympiadBench | MATH500 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-Math-1.5B | 3.8 | 20.0 | 3.8 | 3.3 | 10.4 | 15.2 | 9.4 |
| michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT | 2.9 | 14.4 | 3.3 | 3.7 | 7.0 | 11.0 | 7.0 |
| michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO | 4.2 | 25.6 | 5.8 | 9.9 | 24.1 | 48.6 | **19.7** |
| reasoning_delta_alpha1/qwen_gsm8k_1p5b | 4.6 | 27.8 | 4.2 | 13.2 | 22.2 | 44.4 | 19.4 |

**SII-Enigma 7B best performance**

| Model | AIME25x8 | AMC23x8 | AIME24x8 | Minerva Math | OlympiadBench | MATH500 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-7B-Instruct | 6.7 | 48.4 | 12.9 | 28.3 | 36.0 | 69.6 | 33.6 |
| SII-Enigma/Qwen2.5-7B-Ins-SFT-32k | 11.2 | 47.5 | 9.2 | 30.1 | 35.1 | 72.4 | 34.2 |
| SII-Enigma/Qwen2.5-7B-Ins-SFT-GRPO | 10.4 | 47.2 | 13.3 | 32.0 | 40.7 | 80.4 | **37.3** |
| reasoning_delta_alpha1/sii_enigma_7b | 2.9 | 50.9 | 10.8 | 36.4 | 40.0 | 74.0 | 35.8 |

models we will surge

| Base | RL |
| --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | RabotniKuma/Fast-Math-R1-14B |
| nvidia/OpenMath-Nemotron-14B | RabotniKuma/Fast-OpenMath-Nemotron-14B |
| Qwen/Qwen3-14B | RabotniKuma/Fast-Math-Qwen3-14B |

| **Family** | **Type** | **Model** | **AIME25x8** | **AMC23x8** | **AIME24x8** | **Minerva** | **Olympiad** | **MATH500** | **Avg** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek R1 14B | Base | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 16.7 | 52.2 | 12.9 | 38.6 | 35.0 | 72.0 | 37.9 |
| DeepSeek R1 14B | RL checkpoint | RabotniKuma/Fast-Math-R1-14B | 21.7 | 65.3 | 17.9 | 38.6 | 41.6 | 81.4 | 44.4 |
| OpenMath Nemotron 14B | Base | nvidia/OpenMath-Nemotron-14B | 11.7 | 43.1 | 9.2 | 21.3 | 33.0 | 68.4 | 31.1 |
| OpenMath Nemotron 14B | RL checkpoint | RabotniKuma/Fast-OpenMath-Nemotron-14B | 19.2 | 49.7 | 14.2 | 26.8 | 38.8 | 77.4 | 37.7 |
| Qwen3 14B | Base | Qwen/Qwen3-14B | 2.9 | 32.5 | 1.7 | 28.7 | 20.0 | 58.2 | 24.0 |
| Qwen3 14B | RL checkpoint | RabotniKuma/Fast-Math-Qwen3-14B | 13.8 | 46.2 | 10.8 | 37.9 | 33.9 | 74.4 | 36.2 |

temperature: 0
top_p: 1.0
MIN_P: 0.05
n_sampling: 1
prompt_type: cot
max_tokens_per_call: 8192
vllm_max_model_len: 32768
vllm_batch_size: 2
vllm_gpu_memory_utilization: 0.9
pipeline_parallel_size: 1
apply_chat_template: true
RAW_QUESTION_AS_USER: 1
seed: 0

## GSM8K experiment

# AIME experiment 8000 token

| **Model** | **AIME24x8** | **AIME25x8** | Delta vs Base | Delta vs SFT | Delta vs RL |
| --- | --- | --- | --- | --- | --- |
| Base DeepSeek 14B | 44.6 | 32.9 | 0.0 / 0.0 | +1.3 / +2.5 | -9.6 / -0.9 |
| SFT zbeeb 10ep | 43.3 | 30.4 | -1.3 / -2.5 | 0.0 / 0.0 | -10.9 / -3.4 |
| RL Fast-Math-R1 | 54.2 | 33.8 | +9.6 / +0.9 | +10.9 / +3.4 | 0.0 / 0.0 |
| Merged Base+RL-SFT | 52.5 | 37.5 | +7.9 / +4.6 | +9.2 / +7.1 | -1.7 / +3.7 |

# Math 500 experiment 3000 token

| **Model** | **Math-500** | Delta vs Base | Delta vs SFT | Delta vs RL |
| --- | --- | --- | --- | --- |
| base DeepSeek 14B | 71.8 | 0.0 | +2.2 | -8.2 |
| RL Fast-Math-R1 | 80.0 | +8.2 | +10.4 | 0.0 |
| SFT zbeeb 10ep | 69.6 | -2.2 | 0.0 | -10.4 |
| merged base + RL - SFT | 80.2 | +8.4 | +10.6 | +0.2 |
|  |  |  |  |  |

# MATH 3000 token

| **Model** | **Overall** | **Algebra** | **Count/Prob** | **Geometry** | **Int. Algebra** | **Num Theory** | **Prealgebra** | **Precalc** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merged | **83.45%** | 95.03% | 87.97% | 74.95% | 69.10% | 89.63% | 93.69% | 73.81% |
| rl | 80.03% | **95.20%** | 84.39% | 69.52% | 64.78% | 86.11% | 91.73% | 68.50% |
| base | 73.17% | 91.32% | 75.53% | 64.30% | 55.81% | 75.74% | 87.60% | 61.90% |
| sft | 68.24% | 88.88% | 71.94% | 55.32% | 47.95% | 72.96% | 88.63% | 52.01% |

## Qwen GSM8K 1.5B MATH LightEval run

Run date: 2026-06-02

Formula: `theta_out = theta_base + (theta_GRPO - theta_SFT)`

Dataset: `DigitalLearningGmbH/MATH-lighteval`, full MATH test split, 5000 examples.

Evaluation setup:

- LightEval custom `math_cot:*` tasks, 0-shot.
- Metric: LightEval `extractive_match`.
- Generation: temperature 0, top_p 1.0, seed 0, max_new_tokens 3000.
- Runtime: vLLM, bfloat16, 1 H200 GPU per model, 4 models evaluated in parallel.
- Model max length: 4096. Long prompts were truncated to preserve the 3000-token generation budget.
- This is not the built-in LightEval MATH maj@4 task; it is a deterministic single-sample run.

| Model | Avg | Delta vs Base | Algebra | Count/Prob | Geometry | Int. Algebra | Num Theory | Prealgebra | Precalc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen2.5-Math-1.5B | 6.67 | +0.00 | 3.88 | 10.76 | 4.18 | 6.87 | 5.74 | 9.07 | 6.23 |
| michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT | 5.93 | -0.75 | 4.04 | 8.86 | 3.76 | 8.19 | 4.81 | 6.31 | 5.49 |
| michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO | 6.18 | -0.50 | 3.79 | 9.07 | 4.18 | 7.09 | 4.63 | 7.35 | 7.14 |
| base + (GRPO - SFT) | 6.53 | -0.15 | 4.13 | 9.70 | 3.13 | 7.42 | 6.30 | 8.04 | 6.96 |

Result files on RunPod:

- `/workspace/reasoning_vectors_eval/eval_runs/base/results/Qwen/Qwen2.5-Math-1.5B/results_2026-06-01T22-41-29.299865.json`
- `/workspace/reasoning_vectors_eval/eval_runs/sft/results/michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-SFT/results_2026-06-01T22-40-01.094100.json`
- `/workspace/reasoning_vectors_eval/eval_runs/grpo/results/michaelbzhu/Qwen2.5-Math-1.5B-GSM8K-GRPO/results_2026-06-01T22-39-39.316784.json`
- `/workspace/reasoning_vectors_eval/eval_runs/merged/results/workspace/reasoning_vectors_eval/merged/qwen_gsm8k_1p5b/results_2026-06-01T22-39-04.001406.json`
