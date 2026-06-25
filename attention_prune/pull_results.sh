#!/usr/bin/env bash
# Pull the collector outputs (solved/) back from lab-ws for local viewing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_BASE="${REMOTE_BASE:-/home/ain480/evaluation}"
rsync -avz "$REMOTE_HOST:$REMOTE_BASE/attention_prune/solved/"  "$HERE/solved/"
echo "pulled solved/ <- $REMOTE_HOST"
echo "next:  cd $HERE/viewer && python build_viewer.py && python serve.py"
