#!/usr/bin/env bash
set -euo pipefail
cd "/home/zbibm/reasoning vectors"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate trl-cuda121
mkdir -p logs
echo "START $(date)" > logs/download_zbeeb_sft_14b.log
hf download zbeeb/deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep --local-dir models/zbeeb__deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep --max-workers 4 >> logs/download_zbeeb_sft_14b.log 2>&1
echo "DONE $(date)" >> logs/download_zbeeb_sft_14b.log
