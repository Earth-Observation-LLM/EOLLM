#!/bin/bash
#SBATCH --job-name=ezeval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/ain480/EzelinyumEvaluator/logs/%x_%j.out
#SBATCH --error=/home/ain480/EzelinyumEvaluator/logs/%x_%j.err

set -euo pipefail

MODEL_KEY="${1:?usage: sbatch run_one.sbatch <model_key>}"
ROOT="/home/ain480/EzelinyumEvaluator"
mkdir -p "$ROOT/logs" "$ROOT/outputs/$MODEL_KEY"

cd "$ROOT"
source ~/.bashrc

# Pick the right conda env per backend.
case "$MODEL_KEY" in
  qwen3_5_4b_base|su_base|split*)
    conda activate unsloth
    ;;
  *)
    conda activate earth_eval
    ;;
esac

echo "[slurm] running model=$MODEL_KEY job=$SLURM_JOB_ID node=$(hostname)"
echo "[slurm] env=$(conda info --envs | grep '*' || true)"
echo "[slurm] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u src/run_eval.py --model "$MODEL_KEY"

echo "[slurm] done."
