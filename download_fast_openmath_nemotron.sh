#!/usr/bin/env bash
set -euo pipefail
cd "/home/zbibm/reasoning vectors"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate trl-cuda121
mkdir -p logs
echo "START $(date)" > logs/download_fast_openmath_nemotron.log
hf download RabotniKuma/Fast-OpenMath-Nemotron-14B --local-dir models/RabotniKuma__Fast-OpenMath-Nemotron-14B --max-workers 4 >> logs/download_fast_openmath_nemotron.log 2>&1
echo "DONE $(date)" >> logs/download_fast_openmath_nemotron.log
