# Fast-Math-R1 Node Setup

This guide prepares a fresh GPU node for this project and for Fast-Math-R1-style training.

The target workflow is:

1. Set up the project on a new node.
2. Install the evaluation and training environments.
3. Build or download math training data.
4. Run SFT first.
5. Run GRPO second.
6. Evaluate checkpoints with this repo's math benchmark harness.

The Fast-Math-R1 reference implementation is:

https://github.com/analokmaus/kaggle-aimo2-fast-math-r1

Their recipe trains `DeepSeek-R1-Distill-Qwen-14B` with a hard-math SFT stage, then runs GRPO with format, cosine, and length rewards to reduce overthinking while preserving accuracy.

## Hardware Target

Fast-Math-R1 was written for 8 H200 GPUs. Use that setup when training 14B models full-parameter.

For a smaller or cheaper node:

- Use the existing `Qwen2.5-0.5B` or `Qwen2.5-1.5B` SFT configs first.
- Use LoRA or QLoRA when memory is tight.
- Reduce `max_seq_length`, `max_completion_length`, batch size, and generations before attempting 14B GRPO.
- Keep evaluation on vLLM separate from training when possible.

## Recommended Paths

Use a path without spaces on new machines:

```bash
export ROOT=/workspace/reasoning_vectors
export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT" "$HF_HOME/hub" "$HF_HOME/datasets" /workspace/venvs
cd "$ROOT"
```

If you intentionally mirror the old node path `/home/zbibm/reasoning vectors`, quote it in every command. New configs should prefer `/workspace/reasoning_vectors`.

## System Packages

Install these once on the node:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux htop nvtop rsync build-essential python3-dev
git lfs install
```

Confirm the GPUs are visible:

```bash
nvidia-smi
```

## Copy This Project

If this project is in git:

```bash
git clone <this-repo-url> "$ROOT"
cd "$ROOT"
```

If you are copying from the current workstation:

```bash
rsync -az --exclude models --exclude outputs --exclude training_runs --exclude merged_checkpoints "/Users/mohammadzbeeb/My_Research/reasoning vectors/" USER@NODE:/workspace/reasoning_vectors/
```

Then on the node:

```bash
cd "$ROOT"
```

## Evaluation Environment

The project evaluation environment is named `trl-cuda121` on the current node.

```bash
conda create -n trl-cuda121 python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trl-cuda121
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r evaluation/requirements-node.txt
python -m pip install hf_transfer huggingface_hub
```

Smoke check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

## Training Kit Environment

If `training-kit` is present in the project:

```bash
cd "$ROOT/training-kit"
python -m pip install -e .
python -m pip install -r requirements/trl-cuda121.txt
cd "$ROOT"
```

Check the launcher:

```bash
trl-kit --help
```

## Build Local SFT Data

The local builder creates `training_data/math_sft_100k/train.jsonl`.

```bash
cd "$ROOT"
python evaluation/data_builders/build_math_sft_100k.py --output-dir training_data/math_sft_100k
python evaluation/data_builders/validate_math_sft_dataset.py training_data/math_sft_100k/train.jsonl
```

Expected files:

```text
training_data/math_sft_100k/train.jsonl
training_data/math_sft_100k/validation_report.json
training_data/math_sft_100k/sft_config.yaml
```

## Run Existing Project SFT

For a small first training run:

```bash
cd "$ROOT"
trl-kit sft-launch --config evaluation/training_configs/sft_qwen25_0p5b_instruct_math100k_2ep_runpod.yaml
trl-kit sft-launch --config evaluation/training_configs/sft_qwen25_0p5b_instruct_math100k_2ep_runpod.yaml --execute
```

For the 1.5B config, update paths in `evaluation/training_configs/sft_qwen25_1p5b_instruct_math100k_2ep.yaml` if the project root is not `/home/zbibm/reasoning vectors`.

```bash
cd "$ROOT"
trl-kit sft-launch --config evaluation/training_configs/sft_qwen25_1p5b_instruct_math100k_2ep.yaml
trl-kit sft-launch --config evaluation/training_configs/sft_qwen25_1p5b_instruct_math100k_2ep.yaml --execute
```

## Add The Fast-Math-R1 Recipe

Keep the Fast-Math code side-by-side with this repo so it can use its own training scripts and dependency pins.

```bash
cd "$ROOT"
git clone https://github.com/analokmaus/kaggle-aimo2-fast-math-r1 fast_math_r1_recipe
cd fast_math_r1_recipe
python -m pip install poetry
poetry config virtualenvs.in-project true
poetry lock
poetry install --no-root
```

For Qwen3 training, replace the dependency file before installing:

```bash
cd "$ROOT/fast_math_r1_recipe"
cp dev/pyproject_qwen3.toml pyproject.toml
poetry lock
poetry install --no-root
```

## Cache Base Models And Datasets

For the DeepSeek-based Fast-Math-R1 path:

```bash
cd "$ROOT/fast_math_r1_recipe"
poetry run hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
poetry run python - <<'PY'
from datasets import load_dataset
load_dataset("RabotniKuma/Fast-Math-R1-SFT")
load_dataset("RabotniKuma/Fast-Math-R1-GRPO")
PY
```

For local evaluation of released checkpoints:

```bash
cd "$ROOT"
hf download RabotniKuma/Fast-Math-R1-14B --local-dir models/RabotniKuma__Fast-Math-R1-14B
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local-dir models/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B
```

## Fast-Math Stage 1: SFT

Reference settings:

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- Dataset: `RabotniKuma/Fast-Math-R1-SFT`
- Max sequence length: `24000`
- Batch per GPU: `1`
- Gradient accumulation: `8`
- Epochs: `20`
- Optimizer: `paged_adamw_8bit`
- DeepSpeed: ZeRO-3
- Expected large-node time: about 10 hours on 8 H200 GPUs

Run:

```bash
cd "$ROOT/fast_math_r1_recipe"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 experiments/train_first_stage.py
```

The reference second stage expects this checkpoint:

```text
fast_math_r1_recipe/ft_models/train_first_stage/checkpoint-350
```

If your best SFT checkpoint differs, edit `MODEL_PATH` in `experiments/train_second_stage.py`.

## Fast-Math Stage 2: GRPO

Reference settings:

- Start model: `ft_models/train_first_stage/checkpoint-350`
- Dataset: `RabotniKuma/Fast-Math-R1-GRPO`
- Rewards: `format2`, `cosine`, `length`
- Rollout engine: vLLM
- Max prompt length: `512`
- Max completion length: `16384`
- Generations per problem: `8`
- Batch per GPU: `2`
- Gradient accumulation: `8`
- Beta: `0.04`
- Learning rate: `4e-6`
- DeepSpeed: ZeRO-2
- Expected large-node time: about 10 hours on 8 H200 GPUs

Run:

```bash
cd "$ROOT/fast_math_r1_recipe"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run accelerate launch --config_file accelerate_configs/deepspeed_zero2.yaml --num_processes 8 experiments/train_second_stage.py
```

For smaller nodes, edit `experiments/train_second_stage.py` before launching:

```text
MAX_COMPLETION_LENGTH = 8192
NUM_GENERATIONS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 8
vLLM tensor_parallel_size = visible GPU count
```

## Optional Fast-Math Variants

Fast OpenMath Nemotron:

```bash
cd "$ROOT/fast_math_r1_recipe"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 experiments/train_fast_nemotron_14b.py
```

Fast Qwen3 uses the Qwen3 dependency pins from `dev/pyproject_qwen3.toml`. The reference README describes a 4-GPU training process plus a separate vLLM server on GPUs 4-7. The cloned tree currently includes `deepspeed_zero2.yaml`, `deepspeed_zero3.yaml`, and `fsdp_config.yaml`; add a CPU-offload config before using the exact Qwen3 command from the reference README, or run on a node with enough GPU memory for the shipped ZeRO-3 config.

## Evaluate Released Or Trained Models

Run the default reasoning benchmark set:

```bash
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash evaluation/sh/eval_reasoning_vllm.sh deepseek-distill models/RabotniKuma__Fast-Math-R1-14B outputs/fast_math_r1_14b "aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500" 8192 0.6 1
```

Evaluate your own GRPO checkpoint:

```bash
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash evaluation/sh/eval_reasoning_vllm.sh deepseek-distill "$ROOT/fast_math_r1_recipe/ft_models/train_second_stage/checkpoint-40" outputs/fast_math_r1_grpo "aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500" 8192 0.6 1
```

Collect a table:

```bash
cd "$ROOT/evaluation"
python collect_benchmark_table.py --output_root ../outputs --models "fast_math_r1_14b fast_math_r1_grpo" --benchmarks aime25x8,amc23x8,aime24x8,minerva_math,olympiadbench,math500 --write ../outputs/results.md
```

## Run In Tmux

Use tmux for long jobs:

```bash
tmux new -s fast-math-sft
cd "$ROOT/fast_math_r1_recipe"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 experiments/train_first_stage.py
```

Detach with `Ctrl-b d`.

Watch logs and GPUs:

```bash
tmux attach -t fast-math-sft
nvidia-smi
```

## Fresh Node Checklist

- `nvidia-smi` shows all GPUs.
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `HF_DATASETS_CACHE` point to large storage.
- This project is copied to `$ROOT`.
- `evaluation/requirements-node.txt` is installed in `trl-cuda121`.
- `training-kit` works if using the local SFT configs.
- Fast-Math-R1 is cloned into `$ROOT/fast_math_r1_recipe`.
- `poetry install --no-root` succeeds in the Fast-Math directory.
- Base models and datasets are cached before long jobs.
- SFT runs before GRPO.
- Evaluation writes metrics under `outputs/...`.
