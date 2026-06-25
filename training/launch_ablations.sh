#!/usr/bin/env bash
# launch_ablations.sh — submit the 6 task-ablation training runs to SLURM.
#
# Submits 6 jobs, chained with --dependency=afterany so SLURM runs them
# strictly sequentially on the single GPU. Each job is independent: its own
# OUTPUT_DIR, its own log file, its own EXCLUDE_TOPICS. Failures don't abort
# the chain (afterany, not afterok) — if Split 3 OOMs, Split 5 still runs.
#
# Submission machine: any (just calls sbatch).
# Execution: SLURM normal partition, 1× RTX Pro 6000 96 GB.
#
# Usage:
#     cd ~/training
#     bash training/launch_ablations.sh                # submit all 6
#     bash training/launch_ablations.sh --dry-run      # print sbatch lines, don't submit
#     bash training/launch_ablations.sh --from 3       # skip splits 1,2a,2b — start at split 3
#                                                       # useful after partial completion
#
# Monitor:
#     squeue -u $USER
#     tail -F training/logs/ablation_split2a_per_city.out
#     scancel <jobid>     # cancel one link; downstream jobs still run (afterany)
#     scancel -u $USER    # nuke everything you own
#
# Run matrix:
#   Split 1  per_city     Urbanization Removed       (geo only — 5 tasks)
#   Split 2a per_city     Geo Removed                (urban only — 7 tasks)
#   Split 2b seen_unseen  Geo Removed (cross-city)   (flagship — both axes)
#   Split 3  per_city     Mismatch Removed           (camera_dir → mismatch?)
#   Split 5  per_city     Camera Direction Removed   (mismatch → camera_dir?)
#   Split 6  per_city     Mismatch Easy Kept Only    (easy → hard transfer?)
#                          ↳ removes mismatch_*_hard, trains on easy variants
#
# All runs share:
#   NUM_EPOCHS=5  EARLY_STOPPING=1 (patience=2)  BATCH_SIZE=96
#   WARMUP_STEPS=100  LR_SCHEDULER=cosine  REPORT_TO=none
#   LoRA r=32 α=32 (from rtx_pro_6000_96gb profile, unchanged)
#   SEED=3407 (default in config.py)
#
# Probe policy:
#   All 6 splits use SKIP_PROBE=1 with BATCH_SIZE=38. This value was discovered
#   by an earlier probe run (Job 23, cancelled) at r=16/α=16: bs=64 OOMed
#   (116% of budget); the linear-fit + binary-search probe landed on bs=38
#   (89% utilization, peak 65 GB on a 73 GB budget). All splits use the same
#   per-sample memory footprint (max 5 images, ≤1971 tokens) so the value is
#   stable across the run matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- flag parsing ---

DRY_RUN=0
START_FROM=1   # 1-indexed into the run array below
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --from)    START_FROM="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -60
            exit 0 ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1 ;;
    esac
done

# --- shared training knobs ---

COMMON_ENV=(
    "NUM_EPOCHS=5"
    "EARLY_STOPPING=1"
    # bs=96, SKIP_PROBE=1. The probe's bs=64 OVER_BUDGET (116% util) reading
    # came from worst-case 5-image p99-token samples + a synthetic 5275-token
    # upper bound. Real batches average much lighter (p50=1392). Last run
    # succeeded at bs=86 with r=32; at r=16 we have ~42M fewer trainable
    # params, so 96 should fit. If a worst-case batch OOMs in epoch 1, drop.
    "BATCH_SIZE=96"
    "WARMUP_STEPS=100"
    "LR_SCHEDULER=cosine"
    "REPORT_TO=none"
)

# --- run matrix ---
# Each entry: NAME | SPLIT_STRATEGY | EXCLUDE_TOPICS | SKIP_PROBE
# Pipe-separated so EXCLUDE_TOPICS can contain commas without quoting hell.

RUNS=(
    "split1_per_city|splits_per_city|amenity_richness,building_height,junction_type,land_use,road_type,transit_density,urban_density|1"
    "split2a_per_city|splits_per_city|camera_direction,mismatch_binary_easy,mismatch_binary_hard,mismatch_mcq_easy,mismatch_mcq_hard|1"
    "split2b_seen_unseen|splits_seen_unseen|camera_direction,mismatch_binary_easy,mismatch_binary_hard,mismatch_mcq_easy,mismatch_mcq_hard|1"
    "split3_per_city|splits_per_city|mismatch_binary_easy,mismatch_binary_hard,mismatch_mcq_easy,mismatch_mcq_hard|1"
    "split5_per_city|splits_per_city|camera_direction|1"
    "split6_per_city|splits_per_city|mismatch_binary_hard,mismatch_mcq_hard|1"
)

LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

prev_jobid=""
chain_summary=""

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    if (( idx < START_FROM )); then
        continue
    fi

    IFS='|' read -r name split exclude skip_probe <<< "${RUNS[$i]}"

    output_dir="training/runs/ablation_${name}"
    log_file="${LOGS_DIR}/ablation_${name}.out"
    job_name="abl_${name}"

    # Build the env-prefix string for sbatch.
    env_pairs=("${COMMON_ENV[@]}")
    env_pairs+=("SPLIT_STRATEGY=${split}")
    env_pairs+=("EXCLUDE_TOPICS=${exclude}")
    env_pairs+=("OUTPUT_DIR=${output_dir}")
    if [[ "${skip_probe}" == "1" ]]; then
        env_pairs+=("SKIP_PROBE=1")
    fi

    # sbatch args. afterany so a failure (OOM, crash) doesn't kill downstream
    # jobs — we'd rather get 5 splits than 0 if Split 3 explodes.
    sbatch_args=(
        --job-name="${job_name}"
        --output="${log_file}"
        --error="${log_file}"
    )
    if [[ -n "${prev_jobid}" ]]; then
        sbatch_args+=(--dependency="afterany:${prev_jobid}")
    fi

    # The training script lives at training/train.slurm (relative to repo root).
    cmd=(env "${env_pairs[@]}" sbatch "${sbatch_args[@]}" training/train.slurm)

    echo "─────────────────────────────────────────────────────────────"
    echo "Submitting [${idx}/${#RUNS[@]}]: ${name}"
    echo "  split:           ${split}"
    echo "  exclude_topics:  ${exclude}"
    echo "  output_dir:      ${output_dir}"
    echo "  log:             ${log_file}"
    echo "  skip_probe:      ${skip_probe}"
    if [[ -n "${prev_jobid}" ]]; then
        echo "  depends_on:      ${prev_jobid} (afterany)"
    else
        echo "  depends_on:      <none — runs first when GPU is free>"
    fi

    if (( DRY_RUN )); then
        echo "  [dry-run] would run:"
        printf '    %q ' "${cmd[@]}"; echo
        # Fake a job id so the chain prints meaningfully in dry-run mode.
        prev_jobid="DRY${idx}"
        chain_summary+=$'\n'"  ${idx}. ${name} (DRY${idx})"
        continue
    fi

    # Actually submit. sbatch prints "Submitted batch job <id>" on stdout.
    submit_out="$("${cmd[@]}")"
    echo "  ${submit_out}"
    jobid="$(echo "${submit_out}" | awk '{print $NF}')"
    if [[ -z "${jobid}" || ! "${jobid}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: failed to parse job id from sbatch output: ${submit_out}" >&2
        exit 1
    fi
    prev_jobid="${jobid}"
    chain_summary+=$'\n'"  ${idx}. ${name} (${jobid})"
done

echo "─────────────────────────────────────────────────────────────"
echo "Chain submitted:${chain_summary}"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -F ${LOGS_DIR}/ablation_split1_per_city.out"
echo
echo "Cancel a single link (downstream still runs, afterany):"
echo "  scancel <jobid>"
echo
echo "Cancel everything you own:"
echo "  scancel -u \$USER"
