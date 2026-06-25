#!/bin/bash
#SBATCH --job-name=eo_attn_prune
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# attention_prune Phase 1 collector on lab-ws (RTX 6000 Pro 96GB).
#
# Runs solve.py: raw bf16 Qwen3.6-27B in transformers+eager, solve ladder +
# per-task prompts, then one eager attention forward on each correct answer.
# NON-quantized + eager is the only way to read attention (vLLM/AWQ cannot).
#
# solve.py imports benchmark_suite/ (images, modes, prompt, parsing) and the
# dataset's composite_utils.py. Both must be present on lab-ws:
#   - benchmark_suite at $SUITE_DIR (default ../benchmark_suite next to this repo)
#   - composite_utils under $DATASET_DIR (set below, matches benchmark_suite remote)
#
# Usage (from attention_prune/ on lab-ws):
#   sbatch run_solve.slurm                       # full 8-topic run
#   LIMIT=3 sbatch run_solve.slurm               # smoke (3 questions/topic)
#   TOPICS="junction_type road_type" sbatch run_solve.slurm
set -o pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs solved
echo "=== working dir: $(pwd) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/.bashrc
conda activate "${ENV:-earth_eval}"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

# composite_utils lives in the dataset checkout (authoritative pixel ops).
export DATASET_DIR="${DATASET_DIR:-/home/ain480/training/dataset_content/EODATA_compressed_final}"
echo "=== DATASET_DIR=$DATASET_DIR ==="
[ -f "$DATASET_DIR/composite_utils.py" ] || { echo "!! composite_utils.py not found under DATASET_DIR"; exit 1; }

CONFIG="${CONFIG:-config.yaml}"

EXTRA=()
[ -n "${LIMIT:-}" ]  && EXTRA+=(--limit-per-topic "$LIMIT")
[ -n "${TOPICS:-}" ] && EXTRA+=(--topics $TOPICS)

echo "=== running: python solve.py --config $CONFIG ${EXTRA[*]} ==="
python solve.py --config "$CONFIG" "${EXTRA[@]}"
echo "=== done. results in solved/ ==="
