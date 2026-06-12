#!/usr/bin/env bash
# Launch the teacher server and the ablation runner in detached screens so they
# can be monitored later. Idempotent-ish: refuses to start a screen that already
# exists.
#
#   ./launch.sh server                 # start vLLM in screen "teacher" (optional)
#   ./launch.sh run [MODEL_NAME]       # ablation run in screen "ablation"
#   ./launch.sh status                 # list screens + tail logs
#
#   MODEL_NAME defaults to qwen3.5-27b. Results land in reasoning_distill/
#   results/<MODEL_NAME>_<timestamp>/.  Point at a different server/model via env:
#     SERVED_NAME=teacher BASE_URL=http://localhost:8000 ./launch.sh run gemma4-31b
#
# Monitor:  screen -r teacher   |   screen -r ablation   (detach: Ctrl-a d)
# Logs also tee'd to reasoning_distill/*.log so you can `tail -f` without attaching.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
RD="reasoning_distill"
VLLM_ENV="vllm"

start_server() {
  if screen -list | grep -q "\.teacher\b"; then
    echo "screen 'teacher' already running. Attach: screen -r teacher"; exit 1
  fi
  echo "Starting vLLM (gpu-mem-util 0.94, max-len 32768) in screen 'teacher'..."
  screen -dmS teacher bash -lc \
    "source /home/ezel/miniconda3/etc/profile.d/conda.sh && conda activate $VLLM_ENV && ./$RD/serve_teacher.sh 2>&1 | tee $RD/server.log"
  echo "Started. Watch readiness: tail -f $RD/server.log  (wait for 'Application startup complete')"
}

start_run() {
  local MODEL_NAME="${1:-qwen3.5-27b}"
  local BASE_URL="${BASE_URL:-http://localhost:8000}"
  local SERVED_NAME="${SERVED_NAME:-teacher}"
  if screen -list | grep -q "\.ablation\b"; then
    echo "screen 'ablation' already running. Attach: screen -r ablation"; exit 1
  fi
  # Guard: server must be up
  if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
    echo "Server not responding at $BASE_URL. Start your vLLM server first."
    exit 1
  fi
  echo "Starting ablation run for model='$MODEL_NAME' (per-city train+val + benchmark, all modes) in screen 'ablation'..."
  screen -dmS ablation bash -lc \
    "source /home/ezel/miniconda3/etc/profile.d/conda.sh && conda activate $VLLM_ENV && python $RD/run_ablation.py --model-name '$MODEL_NAME' --base-url '$BASE_URL' --served-name '$SERVED_NAME' --concurrency ${ABLATION_CONCURRENCY:-32} 2>&1 | tee $RD/run.log"
  echo "Started. Watch progress: tail -f $RD/run.log   (results: $RD/results/${MODEL_NAME}_*/)"
}

status() {
  echo "=== screens ==="; screen -list || true
  echo; echo "=== server.log (tail) ==="; tail -n 5 $RD/server.log 2>/dev/null || echo "(none)"
  echo; echo "=== run.log (tail) ==="; tail -n 8 $RD/run.log 2>/dev/null || echo "(none)"
}

case "${1:-}" in
  server) start_server ;;
  run)    start_run "${2:-}" ;;
  status) status ;;
  *) echo "usage: $0 {server|run [MODEL_NAME]|status}"; exit 1 ;;
esac
