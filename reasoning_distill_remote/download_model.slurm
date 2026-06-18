#!/bin/bash
#SBATCH --job-name=eo_dl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# One-time: pull cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8 into the lab-ws HF cache so
# the serve job can run --offline-ish. Network download from the HF hub — no API
# key required. Run this BEFORE submit_strategies.slurm.
#
#   sbatch download_model.slurm
# (or, on the login node:  bash download_model.slurm  — the SBATCH lines are
#  ignored when run directly.)
set -o pipefail
mkdir -p logs
source ~/.bashrc
conda activate "${ENV:-earth_eval}"

MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8}"
echo "=== downloading $MODEL_ID into HF cache ($(date)) ==="
# huggingface-cli ships with transformers/hf_hub; fall back to python if absent.
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID"
else
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"
fi
rc=$?
echo "=== done rc=$rc ==="
exit $rc
