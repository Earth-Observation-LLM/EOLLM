#!/bin/bash
#SBATCH --job-name=ezeval_all
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/ain480/EzelinyumEvaluator/logs/ezeval_all_%j.out
#SBATCH --error=/home/ain480/EzelinyumEvaluator/logs/ezeval_all_%j.err

# Single-job pipeline:
#  1. Smoke test on ONE model (20 records) to catch obvious bugs.
#  2. If smoke test passes, run every model on the full 5,240-record benchmark.
#  3. After all models, run aggregate.py to build the cross-model comparison.
#
# Holds the queue with one allocation; no risk of another user grabbing the
# GPU between models.
#
# Don't use 'set -u': /etc/bashrc references undefined BASHRCSOURCED, which
# would abort the job before it starts. Don't use 'set -e' either: we want
# one model failure to log and continue, not kill the chain.
set -o pipefail

ROOT="/home/ain480/EzelinyumEvaluator"
mkdir -p "$ROOT/logs" "$ROOT/outputs"
cd "$ROOT"

# ----- bookkeeping -----
echo "==== ezeval_all SLURM job ===="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo unknown)"
echo "Workdir:   $ROOT"
echo ""

# Source the user's bashrc to get conda. Must come BEFORE any 'set -u'.
source ~/.bashrc

# Helper to activate the right conda env per model key.
activate_env_for() {
  local key="$1"
  # Deactivate any active env first to make switching safe.
  conda deactivate 2>/dev/null || true
  conda deactivate 2>/dev/null || true
  case "$key" in
    qwen3_5_4b_base|su_base|split*)
      conda activate unsloth
      ;;
    *)
      conda activate earth_eval
      ;;
  esac
  echo "[env] active=$(conda info --envs | awk '/\*/{print $1}')"
}

# Run one model. Logs go into outputs/<key>/run.log alongside its report.
run_model() {
  local key="$1"
  local extra_flags="${2:-}"
  echo ""
  echo "==== START model=$key flags='$extra_flags' time=$(date) ===="
  mkdir -p "$ROOT/outputs/$key"
  activate_env_for "$key"
  python -u src/run_eval.py --model "$key" $extra_flags \
      2>&1 | tee "$ROOT/outputs/$key/run.log"
  local rc=${PIPESTATUS[0]}
  echo "==== END   model=$key rc=$rc time=$(date) ===="
  return $rc
}

# ===== 1. SMOKE TEST =====
SMOKE_MODEL="qwen3_5_4b_base"
echo ""
echo "######## SMOKE TEST ($SMOKE_MODEL, 20 records) ########"
mkdir -p "$ROOT/outputs/_smoke_$SMOKE_MODEL"
activate_env_for "$SMOKE_MODEL"

python -u src/run_eval.py \
    --model "$SMOKE_MODEL" \
    --limit 20 \
    --outdir "$ROOT/outputs/_smoke" \
    2>&1 | tee "$ROOT/logs/smoke_${SMOKE_MODEL}.log"
SMOKE_RC=${PIPESTATUS[0]}

if [[ "$SMOKE_RC" -ne 0 ]]; then
    echo ""
    echo "!!!! SMOKE TEST FAILED (rc=$SMOKE_RC) — aborting full run."
    echo "!!!! Inspect: $ROOT/logs/smoke_${SMOKE_MODEL}.log"
    exit 1
fi

# Confirm the smoke produced a predictions file.
if [[ ! -s "$ROOT/outputs/_smoke/$SMOKE_MODEL/predictions.jsonl" ]]; then
    echo "!!!! SMOKE TEST produced no predictions — aborting."
    exit 2
fi

SMOKE_N=$(wc -l < "$ROOT/outputs/_smoke/$SMOKE_MODEL/predictions.jsonl")
echo ""
echo "######## SMOKE TEST OK: $SMOKE_N predictions written ########"
echo ""

# ===== 2. FULL EVAL OVER ALL MODELS =====
# Order: cheap baselines first (early signal), then expensive baselines, then trained.
MODELS=(
  qwen3_5_4b_base
  su_base
  split5_no_camera
)

declare -a SUCCESS=()
declare -a FAILED=()

for m in "${MODELS[@]}"; do
    if run_model "$m"; then
        SUCCESS+=("$m")
    else
        FAILED+=("$m")
        echo "!! model $m FAILED — continuing with next model."
    fi
done

echo ""
echo "######## ALL MODELS DONE ########"
echo "succeeded: ${SUCCESS[*]:-none}"
echo "failed:    ${FAILED[*]:-none}"

# ===== 3. AGGREGATE =====
echo ""
echo "######## AGGREGATING ########"
# aggregate.py doesn't depend on a model — use earth_eval (has numpy/pandas).
conda deactivate 2>/dev/null || true
conda activate earth_eval
python -u src/aggregate.py 2>&1 | tee "$ROOT/logs/aggregate.log"
AGG_RC=${PIPESTATUS[0]}

echo ""
echo "==== ezeval_all DONE ===="
echo "End: $(date)"
echo "agg_rc=$AGG_RC"
exit 0
