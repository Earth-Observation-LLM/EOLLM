#!/usr/bin/env bash
# Push attention_prune/ AND benchmark_suite/ code to lab-ws (results produced there).
# solve.py imports benchmark_suite/{images,modes,prompt,parsing}, so both dirs must
# sit side-by-side on the remote, exactly as in this repo.
#
# Run locally:  attention_prune/sync_to_labws.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUITE="$(cd "$HERE/../benchmark_suite" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_BASE="${REMOTE_BASE:-/home/ain480/evaluation}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE/attention_prune/logs' '$REMOTE_BASE/benchmark_suite'"
rsync -avz --delete \
  --exclude 'solved/' --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pyc' \
  "$HERE/"  "$REMOTE_HOST:$REMOTE_BASE/attention_prune/"
rsync -avz --delete \
  --exclude 'results/' --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pyc' \
  "$SUITE/"  "$REMOTE_HOST:$REMOTE_BASE/benchmark_suite/"
echo "synced -> $REMOTE_HOST:$REMOTE_BASE/{attention_prune,benchmark_suite}"
echo "next, on $REMOTE_HOST:  cd $REMOTE_BASE/attention_prune && LIMIT=3 sbatch run_solve.slurm"
