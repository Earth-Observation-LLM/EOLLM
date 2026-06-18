#!/bin/bash
#SBATCH --job-name=eo_modes
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Multi-mode benchmark eval (full/blind/sat_only/sv_only), thinking OFF, batched.
# Self-healing batch size: tries MAX_NUM_SEQS list until one succeeds.
#
# Usage:
#   MODEL_ID=cyankiwi/Qwen3.5-9B-AWQ-4bit MODEL_NAME=qwen3.5-9b \
#   ENV=earth_eval sbatch submit_modes.sh
#   # optional: SEQS="128 96 64" MAXLEN=16384 LIMIT=0 THINKING=0
set -o pipefail   # NOT -u: /etc/bashrc references unbound vars under set -u
mkdir -p logs
cd /home/ain480/evaluation/EarthMLLMEval

source ~/.bashrc
conda activate "${ENV:-earth_eval}"

MODEL_ID="${MODEL_ID:?set MODEL_ID}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
MAXLEN="${MAXLEN:-16384}"
LIMIT="${LIMIT:-0}"
THINKING="${THINKING:-0}"   # 0 = thinking OFF (default)
SEQS="${SEQS:-128 96 64 48 32}"

echo "=== node $(hostname) | env ${ENV:-earth_eval} ==="
python -c "import vllm,transformers,torch;print('vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

THINK_FLAG=""
[ "$THINKING" = "1" ] && THINK_FLAG="--enable-thinking"
LIMIT_FLAG=""
[ "$LIMIT" != "0" ] && LIMIT_FLAG="--limit-per-topic $LIMIT"

rc=1
for s in $SEQS; do
  echo "=== attempt: max-num-seqs=$s ==="
  python eval/run_modes_vllm.py \
    --model-id "$MODEL_ID" --model-name "$MODEL_NAME" \
    --max-model-len "$MAXLEN" --max-num-seqs "$s" \
    $THINK_FLAG $LIMIT_FLAG
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== SUCCESS at max-num-seqs=$s ==="
    break
  fi
  echo "=== FAILED at max-num-seqs=$s (rc=$rc); backing off ==="
  sleep 10
done
exit $rc
