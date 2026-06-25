#!/bin/bash
#SBATCH --job-name=ezeval_pcbase
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ain480/EzelinyumEvaluator/logs/ezeval_pcbase_%j.out
#SBATCH --error=/home/ain480/EzelinyumEvaluator/logs/ezeval_pcbase_%j.err

# Single-model job for the rediscovered PC-base LoRA
# (checkpoint-2022 inside 20260422_074420_rtx_pro_6000_96gb).
# qwen3_5_4b_base and su_base already ran in job 152; this fills in the gap.

set -o pipefail

ROOT="/home/ain480/EzelinyumEvaluator"
mkdir -p "$ROOT/logs" "$ROOT/outputs/pc_base"
cd "$ROOT"

echo "==== ezeval_pcbase ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

source ~/.bashrc
conda activate unsloth

# Avoid the HF-hub network call that crashed split5 in job 152.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u src/run_eval.py --model pc_base 2>&1 | tee "$ROOT/outputs/pc_base/run.log"
RC=${PIPESTATUS[0]}
echo ""
echo "==== DONE rc=$RC time=$(date) ===="

# Re-run the aggregate to fold pc_base in alongside qwen3_5_4b_base + su_base.
echo ""
echo "==== AGGREGATING ===="
conda deactivate
conda activate earth_eval
python -u src/aggregate.py 2>&1 | tee "$ROOT/logs/aggregate_$SLURM_JOB_ID.log"

exit $RC
