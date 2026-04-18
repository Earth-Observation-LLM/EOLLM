#!/usr/bin/env bash
# sync.sh — one-way push of this repo to the lab workstation.
# Resumable (rsync -P --partial). Excludes secrets, caches, archives, and
# per-run artifacts. Dataset and models DO sync (they're large; use --checksum
# on re-syncs to skip unchanged files cheaply).
#
# Usage:
#   ./sync.sh                           # normal push
#   ./sync.sh --dry-run                 # preview what would change
#   ./sync.sh --delete                  # also delete remote files not in local
#   REMOTE_HOST=user@host ./sync.sh     # override default host

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lab-ws}"
# No literal tilde — rsync is inconsistent about shell-expansion across ssh
# versions. Use a relative path; ssh resolves it against $HOME.
REMOTE_DIR="${REMOTE_DIR:-training}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

echo "Source: ${SOURCE_DIR}"
echo "Target: ${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Extra rsync args: $*"
echo
echo "NOTE: first sync is ~16 GB (dataset + base model). Re-syncs are fast."
echo

# --checksum skips unchanged files by content (not mtime) — makes re-syncs
# after touching files for no reason near-free.
# --partial keeps partial transfers to resume if interrupted.
rsync -avP --partial --checksum \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='api_keys.env' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='wandb/' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='*.zip' \
    --exclude='training/runs/' \
    --exclude='audit_scratch/' \
    --exclude='unsloth_compiled_cache/' \
    --exclude='.claude/' \
    "$@" \
    "${SOURCE_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "Sync complete. Next: ssh ${REMOTE_HOST} 'bash ${REMOTE_DIR}/remote_setup.sh'"
