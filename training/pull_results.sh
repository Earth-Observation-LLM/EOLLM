#!/usr/bin/env bash
# pull_results.sh — rsync trained LoRAs + eval/benchmark reports from lab-ws to
# the local workstation. Run this LOCALLY whenever you want to collect results.
#
# Pulls (adapter-only, no giant merged models — those aren't saved):
#   - LoRA adapters         (~200 MB each)
#   - eval/summary markdown + json + token-budget + tuning-notes + plots
#   - benchmark outputs from EzelinyumEvaluator (full + ablation modes)
#
# Excludes checkpoints/ (large, transient) and any merged/ dirs.
#
# Usage (from repo root, on your workstation):
#   bash training/pull_results.sh                 # pull everything new
#   bash training/pull_results.sh --dry-run       # preview what would transfer
#
# Local destinations:
#   training/runs_lab_ws/        <- remote training/runs/mm_* run dirs
#   training/benchmark_lab_ws/   <- EzelinyumEvaluator outputs

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REMOTE="${REMOTE:-lab-ws}"
REMOTE_RUNS="${REMOTE_RUNS:-~/training/training/runs}"
REMOTE_BENCH="${REMOTE_BENCH:-~/EzelinyumEvaluator/outputs}"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

mkdir -p training/runs_lab_ws training/benchmark_lab_ws

echo "=== Pulling training run dirs (LoRA + reports, NO checkpoints/merged) ==="
# Include the sweep run dirs (mm_*), their lora/ + report files; exclude the big stuff.
rsync -av ${DRY} \
    --include='mm_*/' \
    --include='mm_*/lora/***' \
    --include='mm_*/*.md' \
    --include='mm_*/*.txt' \
    --include='mm_*/*.json' \
    --include='mm_*/*.png' \
    --exclude='*/checkpoints/***' \
    --exclude='*/merged/***' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_RUNS}/" training/runs_lab_ws/

echo
echo "=== Pulling benchmark outputs (full + sat_only + stv_only) ==="
rsync -av ${DRY} \
    --include='*/' \
    --include='*.json' \
    --include='*.jsonl' \
    --include='*.md' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_BENCH}/" training/benchmark_lab_ws/

echo
echo "Done. Local results:"
echo "  training/runs_lab_ws/        (LoRAs + per-run reports)"
echo "  training/benchmark_lab_ws/   (benchmark: <key>/, _ablation_sat_only/<key>/, _ablation_stv_only/<key>/)"
echo
echo "NOTE: the LOCALLY-trained small models (4B/E2B) already live in"
echo "training/runs/loc_* — no pull needed for those."
