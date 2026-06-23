#!/bin/bash
#SBATCH --job-name=merge_hf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/ain480/training/training/logs/%x_%j.out
#SBATCH --error=/home/ain480/training/training/logs/%x_%j.err
set -o pipefail
cd /home/ain480/training/training || exit 1
source ~/.bashrc
conda activate earth_eval
python -c "import torch,peft,transformers; print('torch',torch.__version__,'peft',peft.__version__,'tf',transformers.__version__)"

RUNS=/home/ain480/training/training/runs
SCRIPT=/home/ain480/training/training/merge_hf.py

echo "######## HF MERGE 1/2: per_city ckpt-2912 -> merged_ep4_hf ########"
ADAPTER="$RUNS/mm_qwen9b_r16_per_city/checkpoints/checkpoint-2912" \
OUT="$RUNS/mm_qwen9b_r16_per_city/merged_ep4_hf" \
python "$SCRIPT" || { echo "!! per_city merge FAILED"; exit 2; }

echo "######## HF MERGE 2/2: seen_unseen ckpt-2248 -> merged_ep4_hf ########"
ADAPTER="$RUNS/mm_qwen9b_r16_seen_unseen/checkpoints/checkpoint-2248" \
OUT="$RUNS/mm_qwen9b_r16_seen_unseen/merged_ep4_hf" \
python "$SCRIPT" || { echo "!! seen_unseen merge FAILED"; exit 3; }

echo "=== ALL HF MERGES DONE ==="
ls -la "$RUNS/mm_qwen9b_r16_per_city/merged_ep4_hf" "$RUNS/mm_qwen9b_r16_seen_unseen/merged_ep4_hf"
