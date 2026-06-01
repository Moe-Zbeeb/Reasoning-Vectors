# Reasoning Vectors

This project includes the math reasoning evaluation harness from One-Shot-RLVR under `evaluation/`.

## Node Environment

Use `trl-cuda121` for this project on the GPU node.

```bash
source /home/zbibm/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121
python -m pip install -r evaluation/requirements-node.txt
```

## vLLM Reasoning Benchmarks

The default benchmark set is:

```text
aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500
```

Run from the project root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash evaluation/sh/eval_reasoning_vllm.sh qwen25-math-cot /path/to/model outputs/model-name
```

Arguments:

```text
1: prompt type
2: model path or Hugging Face model id
3: output directory
4: comma-separated dataset names
5: max tokens per call
6: temperature
7: number of samples per problem
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash evaluation/sh/eval_reasoning_vllm.sh qwen25-math-cot /home/zbibm/models/my-model outputs/my-model "aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500" 3072 0 1
```

## Grading Existing Outputs

```bash
python evaluation/evaluate.py --data_name math500 --file_path outputs/model-name/math500/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl
```

## New Node And Fast-Math-R1 Training

See [FAST_MATH_R1_NODE_SETUP.md](FAST_MATH_R1_NODE_SETUP.md) for a fresh-node setup guide and a Fast-Math-R1-style SFT plus GRPO training workflow.

## Source

Evaluation code and benchmark files were taken from:

https://github.com/ypwang61/One-Shot-RLVR/tree/main/Qwen2.5-Eval/evaluation
