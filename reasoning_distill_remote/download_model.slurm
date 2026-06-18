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
# Use the python API directly — robust across CLI churn (`huggingface-cli` is
# deprecated/no-op on this box; the `hf` CLI exists but the python call avoids
# depending on which CLI generation is installed). Retries are built in.
python - "$MODEL_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download
mid = sys.argv[1]
path = snapshot_download(mid, max_workers=8)
print("downloaded to:", path, flush=True)
PY
rc=$?
echo "=== done rc=$rc ==="
exit $rc
