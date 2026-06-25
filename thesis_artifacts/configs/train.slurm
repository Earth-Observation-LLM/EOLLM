#!/usr/bin/env bash
#SBATCH --job-name=eollm_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --output=training/logs/slurm-%j.out
#SBATCH --error=training/logs/slurm-%j.out
#SBATCH --export=ALL

# train.slurm — SLURM launcher for EOLLM vision SFT.
#
# Submit with:
#   cd ~/training
#   BATCH_SIZE=86 SKIP_PROBE=1 NUM_EPOCHS=10 EARLY_STOPPING=1 \
#     WARMUP_STEPS=100 LR_SCHEDULER=cosine \
#     SPLIT_STRATEGY=splits_seen_unseen \
#     EXCLUDE_TOPICS=road_surface,green_space \
#     REPORT_TO=none \
#     sbatch training/train.slurm
#
# --export=ALL forwards every env var from the submitting shell into the job,
# so the above line-noise works exactly like the train.sh launcher did.
#
# Monitor:
#   squeue -u $USER
#   tail -F training/logs/slurm-<jobid>.out
#   scancel <jobid>     # to stop

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=========================================================="
echo "Host:       $(hostname)"
echo "Time:       $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-<no-slurm>}"
echo "Node:       ${SLURM_JOB_NODELIST:-?}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-?}"
echo "Cwd:        $(pwd)"
echo "=========================================================="

module load conda/latest cuda/13.0.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unsloth

echo "Python:     $(python --version)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Env vars:"
for v in NUM_EPOCHS LEARNING_RATE SMOKE_TEST REPORT_TO WANDB_PROJECT \
         GPU_PROFILE DATASET_DIR BASE_MODEL SEED MAX_STEPS EARLY_STOPPING \
         BATCH_SAFETY_FRAC SPLIT_STRATEGY OUTPUT_DIR SKIP_PROBE BATCH_SIZE \
         WARMUP_STEPS LR_SCHEDULER OPTIM EXCLUDE_TOPICS; do
    printf "  %-20s %s\n" "${v}=" "${!v:-<unset>}"
done
echo "=========================================================="
echo

python -u training/train.py
rc=$?

echo
echo "=== train.py exited with status ${rc} at $(date) ==="
exit ${rc}
