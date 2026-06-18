#!/bin/bash
#SBATCH --job-name=eo_strat
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Prompting-STRATEGY ablation on lab-ws (one SLURM job, one GPU).
#
# For each requested strategy we:
#   1. binary-search the highest --max-num-seqs the GPU sustains (probe_max_seqs.py),
#      scaled to that strategy's real context length,
#   2. launch `vllm serve` IN THE BACKGROUND at that seq ceiling + context,
#   3. wait for /v1/models readiness,
#   4. run run_strategy_ablation.py against it (client concurrency saturates the
#      server), then tear the server down.
#
# Per-strategy context (max-model-len):
#   current   16384   (single short turn)
#   think16k  24576   (16k <think> + images + answer headroom)
#   multistep 24576   (4-turn growing chat history + images on T1)
#
# Usage (from reasoning_distill_remote/ on lab-ws):
#   sbatch submit_strategies.slurm                          # all 3, full benchmark
#   STRATEGIES="current" LIMIT=5 sbatch submit_strategies.slurm     # smoke one
#   STRATEGIES="current think16k multistep" LIMIT=5 sbatch submit_strategies.slurm  # smoke all
#   MODES="full blind" sbatch submit_strategies.slurm       # restrict views
#
# NOT set -u: conda's profile references unbound vars.
set -o pipefail

# sbatch sets SLURM_SUBMIT_DIR to where it was invoked — that's this dir.
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }
mkdir -p logs
echo "=== working dir: $(pwd) ==="

source ~/.bashrc
conda activate "${ENV:-earth_eval}"

MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8}"
SERVED_NAME="${SERVED_NAME:-qwen9b}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b-awq}"
STRATEGIES="${STRATEGIES:-current think16k multistep}"
MODES="${MODES:-full sat_only sv_only blind}"
LIMIT="${LIMIT:-0}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
DATA="${DATA:-/home/ain480/evaluation/EarthMLLMEval/data/benchmark_with_answers.jsonl}"
DATA_ROOT="${DATA_ROOT:-/home/ain480/evaluation/EarthMLLMEval/data}"

echo "=== node $(hostname) | env ${ENV:-earth_eval} ==="
python -c "import vllm,transformers,torch;print('vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Blackwell / FlashInfer workarounds (carried from serve_teacher.sh).
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ctx_for() {  # per-strategy max-model-len
  case "$1" in
    current)   echo 16384 ;;
    think16k)  echo 24576 ;;
    # multistep turns are UNCAPPED (model writes freely on each of 4 turns);
    # give the accumulating chat history generous room so a long observe/reason
    # turn never starves the later turns of context. 96GB box has the headroom.
    multistep) echo 32768 ;;
    *)         echo 32768 ;;
  esac
}

SERVER_PID=""
kill_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "--- stopping vLLM server (pid $SERVER_PID) ---"
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap 'kill_server' EXIT INT TERM

wait_ready() {  # poll /v1/models up to ~5 min
  for _ in $(seq 1 150); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "--- server ready ---"; return 0
    fi
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!!! server process died during startup"; return 1
    fi
    sleep 2
  done
  echo "!!! server did not become ready in time"; return 1
}

OVERALL_RC=0
for STRAT in $STRATEGIES; do
  CTX=$(ctx_for "$STRAT")
  echo ""
  echo "########################################################"
  echo "# strategy=$STRAT  ctx=$CTX  modes=$MODES  limit=$LIMIT"
  echo "########################################################"

  # 1) probe the seq ceiling for this context (fast, in-process, short-ctx probe).
  echo "--- probing max-num-seqs (binary search) ---"
  SEQS=$(python probe_max_seqs.py --model-id "$MODEL_ID" --start "${START_SEQS:-128}" \
           --ceil "${CEIL_SEQS:-512}" --floor "${FLOOR_SEQS:-8}" --tol "${TOL_SEQS:-8}" \
           --real-max-len "$CTX" --gpu-util "$GPU_UTIL" 2>>"logs/probe_${STRAT}_${SLURM_JOB_ID}.err" | tail -1)
  case "$SEQS" in (''|*[!0-9]*) SEQS="${FALLBACK_SEQS:-32}";; esac
  echo "--- chosen max-num-seqs=$SEQS (probe trace: logs/probe_${STRAT}_${SLURM_JOB_ID}.err) ---"

  # client concurrency = serve seq ceiling (saturate it), capped for sanity.
  CONC="$SEQS"; [ "$CONC" -gt 256 ] && CONC=256
  # L3: serve at exactly the seq count the client will drive — no point
  # reserving KV-cache for sequences we never issue.
  SERVE_SEQS="$CONC"

  # 2) launch the server in the background at this strategy's context.
  echo "--- launching vLLM serve (ctx=$CTX, max-num-seqs=$SERVE_SEQS) ---"
  vllm serve "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$CTX" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$SERVE_SEQS" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image": 6}' \
    > "logs/serve_${STRAT}_${SLURM_JOB_ID}.log" 2>&1 &
  SERVER_PID=$!
  echo "    server pid=$SERVER_PID, log: logs/serve_${STRAT}_${SLURM_JOB_ID}.log"

  if ! wait_ready; then
    echo "!!! [$STRAT] server failed to start; see serve log. Skipping."
    tail -40 "logs/serve_${STRAT}_${SLURM_JOB_ID}.log" || true
    kill_server; OVERALL_RC=1; continue
  fi

  # 3) run the strategy.
  LIMIT_FLAG=""; [ "$LIMIT" != "0" ] && LIMIT_FLAG="--limit-per-topic $LIMIT"
  echo "--- running strategy=$STRAT (client concurrency=$CONC) ---"
  python run_strategy_ablation.py \
    --strategy "$STRAT" \
    --base-url "http://localhost:${PORT}" \
    --served-name "$SERVED_NAME" \
    --model-name "$MODEL_NAME" \
    --data "$DATA" --data-root "$DATA_ROOT" \
    --modes $MODES \
    --concurrency "$CONC" \
    $LIMIT_FLAG
  rc=$?
  echo "--- strategy=$STRAT finished rc=$rc ---"
  [ $rc -ne 0 ] && OVERALL_RC=$rc

  # 4) tear down before the next strategy (frees the GPU + the port).
  kill_server
  sleep 5
done

echo ""
echo "=== ALL STRATEGIES DONE (rc=$OVERALL_RC) ==="
echo "results under: $(pwd)/results/<strategy>/"
exit $OVERALL_RC
