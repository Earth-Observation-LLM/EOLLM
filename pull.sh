#!/usr/bin/env bash
# pull.sh — pull training run artifacts back from the remote.
# Complement to sync.sh. Grabs training/runs/ only (everything else is local authority).
#
# Usage:
#   ./pull.sh                              # pull all runs
#   ./pull.sh 20260418_110000_rtx_pro_6000_96gb   # pull specific run
#   REMOTE_HOST=user@host ./pull.sh        # override default host

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
REMOTE_DIR="${REMOTE_DIR:-training}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_FILTER="${1:-}"

if [[ -n "${RUN_FILTER}" ]]; then
    SRC="${REMOTE_HOST}:${REMOTE_DIR}/training/runs/${RUN_FILTER}/"
    DST="${LOCAL_DIR}/training/runs/${RUN_FILTER}/"
    echo "Pulling specific run: ${RUN_FILTER}"
else
    SRC="${REMOTE_HOST}:${REMOTE_DIR}/training/runs/"
    DST="${LOCAL_DIR}/training/runs/"
    echo "Pulling ALL runs from ${REMOTE_HOST}"
fi

mkdir -p "${DST}"

echo "Source: ${SRC}"
echo "Target: ${DST}"
echo

rsync -avP --partial \
    --exclude='checkpoints/*/optimizer.pt' \
    --exclude='checkpoints/*/scheduler.pt' \
    "${SRC}" "${DST}"

echo
echo "Pull complete. Runs live under ${DST}"
