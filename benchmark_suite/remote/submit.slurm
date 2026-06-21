#!/bin/bash
#SBATCH --job-name=eo_bench_suite
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# benchmark_suite vLLM sweep on lab-ws (one SLURM job, one GPU).
#
# Runs run.py against config.remote.yaml — every model there is vLLM (raw/AWQ),
# thinking OFF, all applicable ablation modes. Attention models are NOT included
# here (vLLM can't expose attention; run those locally with the transformers
# backend).
#
# Usage (from benchmark_suite/ on lab-ws):
#   sbatch remote/submit.slurm                    # all vLLM models, full benchmark
#   LIMIT=5 sbatch remote/submit.slurm            # smoke: 5 recs/topic
#   MODELS="qwen3.5-9b-awq" sbatch remote/submit.slurm   # one model
#
# Resumable at the model level: results/<key>/summary.json existing => skipped.
set -o pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs results
echo "=== working dir: $(pwd) ==="

source ~/.bashrc
conda activate "${ENV:-earth_eval}"
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"

CONFIG="${CONFIG:-remote/config.remote.yaml}"
EXTRA=()
[ -n "${LIMIT:-}" ]  && EXTRA+=(--limit-per-topic "$LIMIT")
[ -n "${MODELS:-}" ] && EXTRA+=(--models $MODELS)

echo "=== running: python run.py --config $CONFIG ${EXTRA[*]} ==="
python run.py --config "$CONFIG" "${EXTRA[@]}"
echo "=== done. results in results/ ==="
