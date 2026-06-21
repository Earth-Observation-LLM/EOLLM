#!/usr/bin/env bash
# Push the benchmark_suite CODE to lab-ws (results are produced there).
# Run locally:  benchmark_suite/remote/sync_to_labws.sh
set -euo pipefail
SUITE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_DIR="${REMOTE_DIR:-/home/ain480/evaluation/benchmark_suite}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/logs'"
rsync -avz --delete \
  --exclude 'results/' --exclude 'logs/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude 'results_smoke/' \
  "$SUITE/"  "$REMOTE_HOST:$REMOTE_DIR/"
echo "synced code -> $REMOTE_HOST:$REMOTE_DIR"
echo "next, on $REMOTE_HOST:  cd $REMOTE_DIR && LIMIT=5 sbatch remote/submit.slurm"
