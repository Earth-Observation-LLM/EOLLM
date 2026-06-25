#!/usr/bin/env bash
# Run the image-ablation probe against the local diffusiongemma server.
# Concurrency 4: empirically the safe ceiling — 6+ concurrent 5-image prefills
# OOM the engine on the 32GB 5090 (vision towers run bf16, not FP4), 4 is stable.
set -uo pipefail
cd /home/ezel/Development/EOLLM/reasoning_distill
exec python3 run_ablation.py \
  --base-url http://localhost:8013 \
  --served-name diffusiongemma \
  --model-name diffusiongemma \
  --label diffusiongemma_20260614_224218 \
  --sources benchmark \
  --modes full sat_only sv_only \
  --concurrency 4
