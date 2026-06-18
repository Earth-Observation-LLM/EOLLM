#!/bin/bash
#SBATCH --job-name=eo_modes
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Multi-MODEL benchmark-modes ablation on lab-ws (one SLURM job, one GPU).
# THINKING IS OFF for every model (letter-only output).
#
# For each model in models.sh we:
#   1. binary-search the highest --max-num-seqs the GPU sustains for this model
#      at the benchmark context (probe_max_seqs.py — same in-process LLM path the
#      real eval uses, so the ceiling is exact),
#   2. run run_modes_vllm.py BATCHED at that seq ceiling (full/sat_only/sv_only/
#      blind, all topics, 5240 recs), writing results/<short_name>/,
#   3. move to the next model.
#
# Resumable: a model whose results/<short_name>/<short_name>_summary.json already
# exists is SKIPPED (re-sbatch to continue an interrupted sweep). Force a rerun by
# deleting that summary (or set FORCE=1).
#
# Usage (from benchmark_modes_remote/ on lab-ws):
#   sbatch submit_modes.slurm                       # all models, full benchmark
#   LIMIT=5 sbatch submit_modes.slurm               # smoke: 5 recs/topic, all models
#   MODELS_FILTER="qwen3.5-4b-awq" sbatch submit_modes.slurm   # one model by short name
#   MODES="full blind" sbatch submit_modes.slurm    # restrict views
#
# NOT set -u: conda's profile references unbound vars.
set -o pipefail

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs results
echo "=== working dir: $(pwd) ==="

source ~/.bashrc
# NOTE: the env is activated PER MODEL inside the loop (models.sh may name a
# different conda env per model, e.g. gemma needs vllm_gemma4). We activate a
# default here only so the version banner below works.
conda activate "${ENV:-earth_eval}"

# --- config ---------------------------------------------------------------
DATA="${DATA:-/home/ain480/evaluation/EarthMLLMEval/data/benchmark_with_answers.jsonl}"
DATA_ROOT="${DATA_ROOT:-/home/ain480/evaluation/EarthMLLMEval/data}"
LIMIT="${LIMIT:-0}"
MODES="${MODES:-full sat_only sv_only blind}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_IMG_PX="${MAX_IMG_PX:-1120}"
FORCE="${FORCE:-0}"
MODELS_FILTER="${MODELS_FILTER:-}"   # optional: only run short_names matching this (space-sep)

source "$HERE/models.sh"

echo "=== node $(hostname) | env ${ENV:-earth_eval} ==="
python -c "import vllm,transformers,torch;print('vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Blackwell / FlashInfer workarounds (carried from the strategy-ablation job).
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT_FLAG=""; [ "$LIMIT" != "0" ] && LIMIT_FLAG="--limit-per-topic $LIMIT"
MODES_FLAG=""; [ -n "$MODES" ] && MODES_FLAG="--modes $MODES"

OVERALL_RC=0
for ENTRY in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_ID SHORT CTX MODEL_ENV <<< "$ENTRY"
  # The per-model env (4th field) is authoritative — gemma needs vllm_gemma4,
  # the Qwens need earth_eval, so a single whole-job env would break one of them.
  # FORCE_ENV is an explicit escape hatch to override every model's env.
  RUN_ENV="${FORCE_ENV:-${MODEL_ENV:-earth_eval}}"

  # optional filter
  if [ -n "$MODELS_FILTER" ]; then
    case " $MODELS_FILTER " in (*" $SHORT "*) : ;; (*) echo "--- skip $SHORT (not in MODELS_FILTER) ---"; continue ;; esac
  fi

  OUTDIR="results/$SHORT"
  SUMMARY="$OUTDIR/${SHORT}_summary.json"
  if [ "$FORCE" != "1" ] && [ -f "$SUMMARY" ]; then
    echo "--- skip $SHORT (already done: $SUMMARY; FORCE=1 to rerun) ---"
    continue
  fi

  echo ""
  echo "########################################################"
  echo "# model=$SHORT  id=$MODEL_ID  ctx=$CTX  env=$RUN_ENV  modes=$MODES  limit=$LIMIT"
  echo "########################################################"

  # Activate THIS model's conda env (gemma -> vllm_gemma4, Qwens -> earth_eval).
  conda activate "$RUN_ENV" || { echo "!!! cannot activate env $RUN_ENV; skipping $SHORT"; OVERALL_RC=1; continue; }
  python -c "import vllm,transformers;print('  env=$RUN_ENV vllm',vllm.__version__,'tf',transformers.__version__)" 2>&1 | tail -1

  # 1) probe the seq ceiling for THIS model at the benchmark context.
  echo "--- probing max-num-seqs (binary search) for $SHORT ---"
  SEQS=$(python probe_max_seqs.py --model-id "$MODEL_ID" --start "${START_SEQS:-128}" \
           --ceil "${CEIL_SEQS:-512}" --floor "${FLOOR_SEQS:-8}" --tol "${TOL_SEQS:-8}" \
           --real-max-len "$CTX" --gpu-util "$GPU_UTIL" --margin "${MARGIN:-0.98}" \
           2>>"logs/probe_${SHORT}_${SLURM_JOB_ID}.err" | tail -1)
  case "$SEQS" in (''|*[!0-9]*) SEQS="${FALLBACK_SEQS:-32}";; esac
  echo "--- chosen max-num-seqs=$SEQS for $SHORT (trace: logs/probe_${SHORT}_${SLURM_JOB_ID}.err) ---"

  # 2) run the batched eval at that seq ceiling. Thinking OFF (no --enable-thinking).
  echo "--- running benchmark modes for $SHORT (max-num-seqs=$SEQS) ---"
  python run_modes_vllm.py \
    --model-id "$MODEL_ID" --model-name "$SHORT" \
    --data "$DATA" --data-root "$DATA_ROOT" \
    --outdir "$OUTDIR" \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" \
    --gpu-mem-util "$GPU_UTIL" --max-img-px "$MAX_IMG_PX" \
    $MODES_FLAG $LIMIT_FLAG
  rc=$?
  echo "--- model=$SHORT finished rc=$rc ---"
  [ $rc -ne 0 ] && OVERALL_RC=$rc
  sleep 5
done

echo ""
echo "=== ALL MODELS DONE (rc=$OVERALL_RC) ==="
echo "per-model results under: $(pwd)/results/<short_name>/"
echo "build the cross-model tables:  python make_modes_report.py"
exit $OVERALL_RC
