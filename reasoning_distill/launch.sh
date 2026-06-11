#!/usr/bin/env bash
# Launch the teacher server and the ablation runner in detached screens so they
# can be monitored later. Idempotent-ish: refuses to start a screen that already
# exists.
#
#   ./launch.sh server     # start vLLM in screen "teacher" (run first)
#   ./launch.sh run        # start the ablation run in screen "ablation"
#   ./launch.sh status     # list screens + tail logs
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
    "conda activate $VLLM_ENV && ./$RD/serve_teacher.sh 2>&1 | tee $RD/server.log"
  echo "Started. Watch readiness: tail -f $RD/server.log  (wait for 'Application startup complete')"
}

start_run() {
  if screen -list | grep -q "\.ablation\b"; then
    echo "screen 'ablation' already running. Attach: screen -r ablation"; exit 1
  fi
  # Guard: server must be up
  if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "Teacher server not responding on :8000. Start it first: ./launch.sh server"
    exit 1
  fi
  echo "Starting ablation run (per-city train+val + benchmark, both passes) in screen 'ablation'..."
  screen -dmS ablation bash -lc \
    "conda activate $VLLM_ENV && python $RD/run_ablation.py --concurrency 24 2>&1 | tee $RD/run.log"
  echo "Started. Watch progress: tail -f $RD/run.log"
}

status() {
  echo "=== screens ==="; screen -list || true
  echo; echo "=== server.log (tail) ==="; tail -n 5 $RD/server.log 2>/dev/null || echo "(none)"
  echo; echo "=== run.log (tail) ==="; tail -n 8 $RD/run.log 2>/dev/null || echo "(none)"
}

case "${1:-}" in
  server) start_server ;;
  run)    start_run ;;
  status) status ;;
  *) echo "usage: $0 {server|run|status}"; exit 1 ;;
esac
