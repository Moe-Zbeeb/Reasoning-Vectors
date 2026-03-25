#!/usr/bin/env bash
set -e

ENV_NAME="reasoning"
PYTHON_VERSION="3.11"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

CONDA_BASE=$(conda info --base)
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
PIP="$ENV_PREFIX/bin/pip"
PYTHON="$ENV_PREFIX/bin/python"

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Upgrading pip"
"$PIP" install --upgrade pip

echo "Installing PyTorch with CUDA 12.4"
"$PIP" install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "Verifying PyTorch & CUDA"
"$PYTHON" - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Installing Hugging Face ecosystem"
"$PIP" install \
  transformers \
  datasets \
  accelerate \
  evaluate \
  peft \
  sentencepiece \
  safetensors \
  huggingface_hub \
  bitsandbytes \
  tiktoken \
  einops

echo "Installing vLLM"
"$PIP" install vllm

echo "Final verification"
"$PYTHON" - <<'PY'
import torch, transformers
import vllm

print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)

model = transformers.AutoModel.from_pretrained("bert-base-uncased")
print("Loaded:", model.__class__.__name__)
PY

echo "Environment setup complete"
