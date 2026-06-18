#!/usr/bin/env bash
# Pull the produced results + SLURM logs back from lab-ws into this dir.
# Run locally:  ./pull_results.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_DIR="${REMOTE_DIR:-/home/ain480/evaluation/benchmark_modes_remote}"

mkdir -p "$HERE/results" "$HERE/logs"
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/results/" "$HERE/results/"
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/logs/"    "$HERE/logs/"
echo "pulled results + logs <- $REMOTE_HOST:$REMOTE_DIR"
ls -R "$HERE/results" | head -40
