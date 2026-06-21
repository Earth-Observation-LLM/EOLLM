#!/usr/bin/env bash
# Pull produced results + SLURM logs back from lab-ws.
# Run locally:  benchmark_suite/remote/pull_results.sh
set -euo pipefail
SUITE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_DIR="${REMOTE_DIR:-/home/ain480/evaluation/benchmark_suite}"

mkdir -p "$SUITE/results" "$SUITE/logs"
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/results/" "$SUITE/results/"
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/logs/"    "$SUITE/logs/" || true
echo "pulled results + logs <- $REMOTE_HOST:$REMOTE_DIR"
ls -R "$SUITE/results" | head -40
