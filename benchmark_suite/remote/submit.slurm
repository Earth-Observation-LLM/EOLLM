#!/bin/bash
#SBATCH --job-name=eo_bench_suite
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# benchmark_suite vLLM sweep on lab-ws (one SLURM job, one GPU).
#
# Runs run.py against config.remote.yaml — every model there is vLLM (raw/AWQ/
# merged), thinking OFF, all applicable ablation modes (blind->sv->sat->full).
# Attention models are NOT included here (vLLM can't expose attention).
#
# MAX BATCH @ ~2% HEADROOM: if PROBE_MODEL is set, we binary-search the highest
# max_num_seqs the card sustains for that model (probe_max_seqs.py), export it as
# SUITE_MAX_NUM_SEQS (the vLLM backend honors it), and run gpu_memory_utilization
# at PROBE_GPU_UTIL (default 0.98). Both our 9B-merged models are the same size,
# so one probe value applies to all. Unset PROBE_MODEL to use the config value.
#
# Usage (from benchmark_suite/ on lab-ws):
#   PROBE_MODEL=/abs/path/to/merged sbatch remote/submit.slurm   # probe + full
#   LIMIT=2 sbatch remote/submit.slurm                           # smoke, no probe
#   MODELS="qwen9b_per_city" sbatch remote/submit.slurm          # one model
#
# Resumable at the (model,dataset) level: results/<out_key>/summary.json => skip.
set -o pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs results
echo "=== working dir: $(pwd) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/.bashrc
conda activate "${ENV:-earth_eval}"
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"

CONFIG="${CONFIG:-remote/config.remote.yaml}"
PROBE_GPU_UTIL="${PROBE_GPU_UTIL:-0.90}"
REAL_MAX_LEN="${REAL_MAX_LEN:-16384}"

# images.py imports the dataset's composite_utils.py (authoritative pixel ops).
# It lives in the dataset checkout, NOT under the suite, so point DATASET_DIR at
# it. On lab-ws the dataset is under training/, not evaluation/.
export DATASET_DIR="${DATASET_DIR:-/home/ain480/training/dataset_content/EODATA_compressed_final}"
echo "=== DATASET_DIR=$DATASET_DIR ==="
[ -f "$DATASET_DIR/composite_utils.py" ] || { echo "!! composite_utils.py not found under DATASET_DIR"; exit 1; }

# --- probe the max batch size (skipped during smoke / when PROBE_MODEL unset) ---
if [ -n "${PROBE_MODEL:-}" ] && [ -z "${LIMIT:-}" ]; then
  echo "=== probing max_num_seqs for $PROBE_MODEL (gpu_util=$PROBE_GPU_UTIL, real_max_len=$REAL_MAX_LEN) ==="
  SEQS=$(python remote/probe_max_seqs.py --model-id "$PROBE_MODEL" \
           --gpu-util "$PROBE_GPU_UTIL" --real-max-len "$REAL_MAX_LEN" \
           --ceil "${PROBE_CEIL:-512}" 2>logs/probe_${SLURM_JOB_ID:-x}.err | tail -1)
  if [[ "$SEQS" =~ ^[0-9]+$ ]] && [ "$SEQS" -gt 0 ]; then
    export SUITE_MAX_NUM_SEQS="$SEQS"
    echo "=== probe chose max_num_seqs=$SEQS (trace: logs/probe_${SLURM_JOB_ID:-x}.err) ==="
  else
    echo "!! probe failed (got '$SEQS'); falling back to config max_num_seqs. See logs/probe_${SLURM_JOB_ID:-x}.err" >&2
  fi
else
  echo "=== probe skipped (PROBE_MODEL unset or LIMIT set); using config max_num_seqs ==="
fi
# 2% headroom for the real run too (config can still override per-model).
export SUITE_GPU_MEM_UTIL="$PROBE_GPU_UTIL"

EXTRA=()
[ -n "${LIMIT:-}" ]  && EXTRA+=(--limit-per-topic "$LIMIT")
[ -n "${MODELS:-}" ] && EXTRA+=(--models $MODELS)

echo "=== running: python run.py --config $CONFIG ${EXTRA[*]} ==="
python run.py --config "$CONFIG" "${EXTRA[@]}"
echo "=== done. results in results/ ==="
