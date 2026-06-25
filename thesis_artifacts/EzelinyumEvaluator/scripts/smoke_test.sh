#!/usr/bin/env bash
# Quick smoke test on a tiny subset before submitting full jobs.
set -euo pipefail
ROOT="/home/ain480/EzelinyumEvaluator"
cd "$ROOT"
source ~/.bashrc
MODEL="${1:-qwen3_5_4b_base}"
case "$MODEL" in
  qwen3_5_4b_base|su_base|split*) conda activate unsloth ;;
  *) conda activate earth_eval ;;
esac
python -u src/run_eval.py --model "$MODEL" --limit 20
