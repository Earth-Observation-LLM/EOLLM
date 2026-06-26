#!/bin/bash
#SBATCH --job-name=eo_attn_prune2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# attention_prune Phase 2 (prune.py) on lab-ws (RTX 6000 Pro 96GB).
#
# Re-asks each Phase-1 solved_greedy question on the attended k-image SV subset in
# a SINGLE GREEDY PASS (pure benchmark). Arms: top-k / fwd / random-k / bottom-k /
# k=0. No attention forward here — Phase 2 is just generation, so the default
# (faster) attention impl is used, not eager.
#
# Requires Phase-1 output present: solved/<model.key>/solved.jsonl
#
# Usage (from attention_prune/ on lab-ws):
#   sbatch run_prune.slurm                 # k from config (k=1)
#   K="1 2 3" sbatch run_prune.slurm       # full survival curve
#   LIMIT=20 sbatch run_prune.slurm        # smoke (cap solved records)
set -o pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs solved
echo "=== working dir: $(pwd) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/.bashrc
conda activate "${ENV:-earth_eval}"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

export DATASET_DIR="${DATASET_DIR:-/home/ain480/training/dataset_content/EODATA_compressed_final}"
echo "=== DATASET_DIR=$DATASET_DIR ==="
[ -f "$DATASET_DIR/composite_utils.py" ] || { echo "!! composite_utils.py not found under DATASET_DIR"; exit 1; }

CONFIG="${CONFIG:-config.yaml}"

EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")

echo "=== running: python prune.py --config $CONFIG ${EXTRA[*]} (K=${K:-config}) ==="
python prune.py --config "$CONFIG" "${EXTRA[@]}"
echo "=== benchmark done; running analysis ==="
python prune.py --config "$CONFIG" --analyze
echo "=== done. results in solved/<model>/prune/ ==="
