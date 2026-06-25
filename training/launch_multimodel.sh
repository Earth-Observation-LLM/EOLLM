#!/usr/bin/env bash
# launch_multimodel.sh — submit the full multi-model training sweep to SLURM.
#
# Fans out  {4 models} × {seen_unseen, per_city, text_only} = 12 jobs, chained
# with --dependency=afterany so they run STRICTLY SEQUENTIALLY on the single
# RTX Pro 6000 96 GB. Each job runs the full per-model pipeline:
#     train  ->  full-val eval (eval_base.py)  ->  benchmark (EzelinyumEvaluator)
# via training/run_one_model.slurm.
#
# afterany (not afterok): if one run OOMs or crashes, the chain continues — we'd
# rather get 11 of 12 than 0. Each job is independent (own OUTPUT_DIR, own logs).
#
# Order (see models.sh): qwen4b → gemma_e2b → qwen27b → gemma_31b, and within
# each model seen_unseen → per_city → text_only. So the VERY FIRST job is
# "qwen4b / seen_unseen" — the validation gate. Inspect it before the big models
# (which are later in the chain) consume GPU days. Use --from to resume.
#
# Shared config (set here, passed via env-prefix; run_one_model.slurm forwards):
#     GPU_PROFILE=rtx_pro_6000_96gb   IMAGE_MAX_EDGE=512   LORA_R=16 (alpha=16)
#     NUM_EPOCHS=5   EARLY_STOPPING=1   adamw_8bit   cosine   warmup=100
#     SAVE_MERGED=0 (adapter-only)   probe per model (NO SKIP_PROBE)
#     SEED=3407
#
# Usage:
#     cd ~/training                      # repo root on lab-ws
#     bash training/launch_multimodel.sh                 # submit all 12
#     bash training/launch_multimodel.sh --dry-run       # print, don't submit
#     bash training/launch_multimodel.sh --from 4        # start at job 4 (skip 1-3)
#     bash training/launch_multimodel.sh --only qwen4b   # only that model's 3 jobs
#     bash training/launch_multimodel.sh --no-benchmark  # skip the eval-harness step
#
# Monitor:
#     squeue -u $USER
#     tail -F training/logs/mm-<model>_<cond>.out
#     scancel <jobid>     # cancel one link; downstream still runs (afterany)
#     scancel -u $USER    # nuke everything

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=models.sh
source "${SCRIPT_DIR}/models.sh"

# --- flags ---
DRY_RUN=0
START_FROM=1
ONLY_MODEL=""
NO_BENCHMARK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1; shift ;;
        --from)         START_FROM="$2"; shift 2 ;;
        --only)         ONLY_MODEL="$2"; shift 2 ;;
        --no-benchmark) NO_BENCHMARK=1; shift ;;
        -h|--help)      grep '^#' "$0" | head -50; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# --- shared env across every job ---
# NUM_EPOCHS is per-model (6 small / 12 big) — set later from the registry, NOT
# here. Everything else is uniform.
COMMON_ENV=(
    "GPU_PROFILE=${GPU_PROFILE:-rtx_pro_6000_96gb}"
    "IMAGE_MAX_EDGE=512"
    "LORA_R=16"
    "LORA_ALPHA=16"
    "EARLY_STOPPING=1"
    "WARMUP_STEPS=100"
    "LR_SCHEDULER=cosine"
    "OPTIM=adamw_8bit"
    "SAVE_MERGED=0"
    "REPORT_TO=none"
    "SEED=3407"
    # --- SPEED probe policy (user: "probe up to 128, ~2GB headroom is fine") ---
    # Push batch size as high as fits. hard_cap=128, flat margin (no tiered
    # 10-15%), 2 GB absolute reserve → the probe lands on the largest bs that
    # leaves ~2 GB free at worst-case (5-image) batches. Bigger bs = fewer
    # optimizer steps = faster epochs. The probe still measures real peak VRAM
    # on worst-case samples + does a fwd/bwd/step, so it won't pick an OOMing bs.
    "PROBE_HARD_CAP=128"
    "PROBE_RESERVE_MB=2048"
    "PROBE_FLAT_MARGIN=0.0"
    "PROBE_TARGET_LOW=0.92"
    "PROBE_TARGET_HIGH=0.99"
)
if [[ "${NO_BENCHMARK}" == "1" ]]; then
    COMMON_ENV+=("SKIP_BENCHMARK=1")
fi

LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# --- build the flat job list: model × condition (model-major, condition order
#     as in CONDITIONS so seen_unseen is first within each model) ---
JOBS=()   # each: "model_key|base|family|name|lr|epochs|cond_key|split|text_only"
for m in "${MODELS[@]}"; do
    IFS='|' read -r mkey base family name lr epochs <<< "${m}"
    # --only accepts a comma-separated list of model KEYs (e.g. qwen4b,gemma_e2b)
    if [[ -n "${ONLY_MODEL}" ]]; then
        match=0
        IFS=',' read -ra _wantk <<< "${ONLY_MODEL}"
        for w in "${_wantk[@]}"; do [[ "${w}" == "${mkey}" ]] && match=1; done
        [[ ${match} -eq 0 ]] && continue
    fi
    for c in "${CONDITIONS[@]}"; do
        IFS='|' read -r ckey split textonly <<< "${c}"
        JOBS+=("${mkey}|${base}|${family}|${name}|${lr}|${epochs}|${ckey}|${split}|${textonly}")
    done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "No jobs to submit (check --only=${ONLY_MODEL})." >&2
    exit 1
fi

echo "Sweep: ${#JOBS[@]} jobs (model × condition), chained afterany on 1 GPU."
echo "Shared: edge=512, r/alpha=16, 5 epochs, early-stop, adamw_8bit, adapter-only."
[[ "${NO_BENCHMARK}" == "1" ]] && echo "Benchmark step: DISABLED (--no-benchmark)."
echo

prev_jobid=""
chain_summary=""
for i in "${!JOBS[@]}"; do
    idx=$((i + 1))
    (( idx < START_FROM )) && continue

    IFS='|' read -r mkey base family name lr epochs ckey split textonly <<< "${JOBS[$i]}"

    model_key="mm_${mkey}_${ckey}"
    output_dir="training/runs/${model_key}"
    log_file="${LOGS_DIR}/${model_key}.out"
    job_name="mm_${mkey}_${ckey}"

    env_pairs=("${COMMON_ENV[@]}")
    env_pairs+=("BASE_MODEL=${base}")
    env_pairs+=("MODEL_FAMILY=${family}")
    env_pairs+=("MODEL_NAME=${name}")
    env_pairs+=("LEARNING_RATE=${lr}")
    env_pairs+=("NUM_EPOCHS=${epochs}")
    env_pairs+=("SPLIT_STRATEGY=${split}")
    env_pairs+=("TEXT_ONLY=${textonly}")
    env_pairs+=("OUTPUT_DIR=${output_dir}")
    env_pairs+=("MODEL_KEY=${model_key}")

    sbatch_args=(
        --job-name="${job_name}"
        --output="${log_file}"
        --error="${log_file}"
    )
    if [[ -n "${prev_jobid}" ]]; then
        sbatch_args+=(--dependency="afterany:${prev_jobid}")
    fi

    cmd=(env "${env_pairs[@]}" sbatch "${sbatch_args[@]}" training/run_one_model.slurm)

    echo "─────────────────────────────────────────────────────────────"
    echo "[${idx}/${#JOBS[@]}] ${model_key}"
    echo "  model:   ${name} (family=${family}, lr=${lr}, epochs=${epochs})"
    echo "  base:    ${base}"
    echo "  split:   ${split}   text_only=${textonly}"
    echo "  output:  ${output_dir}"
    echo "  log:     ${log_file}"
    [[ -n "${prev_jobid}" ]] && echo "  after:   ${prev_jobid} (afterany)" || echo "  after:   <none — runs first>"

    if (( DRY_RUN )); then
        echo "  [dry-run] would run:"
        printf '    %q ' "${cmd[@]}"; echo
        prev_jobid="DRY${idx}"
        chain_summary+=$'\n'"  ${idx}. ${model_key} (DRY${idx})"
        continue
    fi

    submit_out="$("${cmd[@]}")"
    echo "  ${submit_out}"
    jobid="$(echo "${submit_out}" | awk '{print $NF}')"
    if [[ -z "${jobid}" || ! "${jobid}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse job id: ${submit_out}" >&2
        exit 1
    fi
    prev_jobid="${jobid}"
    chain_summary+=$'\n'"  ${idx}. ${model_key} (${jobid})"
done

echo "─────────────────────────────────────────────────────────────"
echo "Chain submitted:${chain_summary}"
echo
echo "Monitor:    squeue -u \$USER"
echo "First log:  tail -F ${LOGS_DIR}/mm_qwen4b_seen_unseen.out"
echo "Cancel one: scancel <jobid>  (downstream still runs, afterany)"
echo "Cancel all: scancel -u \$USER"
echo
echo "Pull results to local later (run on your workstation):"
echo "  rsync -av --include='*/' --include='lora/***' --include='*.md' \\"
echo "    --include='*.json' --include='*.txt' --include='*.png' --exclude='*' \\"
echo "    lab-ws:~/training/training/runs/ ./training/runs_lab_ws/"
