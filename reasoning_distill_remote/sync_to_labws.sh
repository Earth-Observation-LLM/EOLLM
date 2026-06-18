#!/usr/bin/env bash
# Push this dir's CODE to lab-ws (not results — those are produced there).
# Run locally:  ./sync_to_labws.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_DIR="${REMOTE_DIR:-/home/ain480/evaluation/reasoning_distill_remote}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/logs'"
rsync -avz --delete \
  --exclude 'results/' --exclude 'logs/' --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$HERE/"  "$REMOTE_HOST:$REMOTE_DIR/"
echo "synced code -> $REMOTE_HOST:$REMOTE_DIR"
echo "next, on $REMOTE_HOST:  cd $REMOTE_DIR && sbatch submit_strategies.slurm"
