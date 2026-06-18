#!/bin/bash
#SBATCH --job-name=eo_dl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# One-time: pull the four AWQ benchmark models into the lab-ws HF cache, and
# (optionally) DELETE the big unsloth checkpoints we no longer need to reclaim
# disk. Network download from the HF hub — no API key needed.
#
#   sbatch download_models.slurm            # download all four
#   PURGE_UNSLOTH=1 sbatch download_models.slurm   # also delete unsloth--* caches
#
# (Run on the login node with `bash download_models.slurm` if you prefer — the
#  SBATCH lines are ignored.)
set -o pipefail
mkdir -p logs
source ~/.bashrc
conda activate "${ENV:-earth_eval}"

MODELS=(
  "cyankiwi/Qwen3.5-4B-AWQ-BF16-INT8"
  "cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8"
  "cyankiwi/gemma-4-12B-it-AWQ-INT4"
  "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
)

echo "=== downloading ${#MODELS[@]} models into HF cache ($(date)) ==="
for MID in "${MODELS[@]}"; do
  echo "--- $MID ---"
  python - "$MID" <<'PY'
import sys
from huggingface_hub import snapshot_download
mid = sys.argv[1]
path = snapshot_download(mid, max_workers=8)
print("downloaded:", mid, "->", path, flush=True)
PY
done

if [ "${PURGE_UNSLOTH:-0}" = "1" ]; then
  echo "=== purging unsloth--* caches to reclaim disk ==="
  for HUB in ~/.cache/huggingface/hub /home/ain480/.cache/huggingface/hub; do
    for d in "$HUB"/models--unsloth--*; do
      [ -e "$d" ] || continue
      echo "rm -rf $d"
      rm -rf "$d"
    done
  done
fi

echo "=== done ($(date)) ==="
df -h ~/.cache/huggingface 2>/dev/null | tail -1
