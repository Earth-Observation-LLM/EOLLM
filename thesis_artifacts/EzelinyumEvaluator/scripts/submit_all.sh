#!/usr/bin/env bash
# Submit one SLURM job per evaluable model. Independent jobs — a single
# OOM/crash will not take down the rest.
set -euo pipefail

ROOT="/home/ain480/EzelinyumEvaluator"
cd "$ROOT"

# Order: cheapest baselines first so we get an early signal, then the trained.
MODELS=(
  qwen3_5_4b_base
  llava_ov_05b
  llava_ov_7b
  qwen2_5_vl_7b
  gemma_4_e2b
  llama_3_2_11b_vision
  pixtral_12b
  su_base
  split1_no_urban
  split2a_no_geo_pc
  split2b_no_geo_su
  split3_no_mismatch
  split5_no_camera
  split6_no_hard_mismatch
)

for m in "${MODELS[@]}"; do
  echo "Submitting $m"
  sbatch --job-name="ezeval_$m" scripts/run_one.sbatch "$m"
done

echo "Done. Watch with: squeue -u \$USER"
