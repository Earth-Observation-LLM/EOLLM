#!/usr/bin/env bash
# launch_sweep.sh — config-driven training-sweep launcher.
#
# Reads a single declarative config (default: training/sweep.conf) and fans out
# the full cartesian product  MODELS × RANKS × CONDITIONS  as SLURM jobs, chained
# --dependency=afterany so they run STRICTLY SEQUENTIALLY on the one GPU.
#
# Each job runs the full per-(model,rank,condition) pipeline via
# run_one_model.slurm:    train -> full-val eval (eval_base.py) -> benchmark.
#
# afterany (not afterok): a single OOM/crash does NOT kill the chain — we'd
# rather get N-1 of N than 0. Every job is independent (own OUTPUT_DIR + logs).
#
# Run-dir / eval-key:   mm_<SWEEP_NAME>_r<RANK>_<COND_KEY>
# Job order is rank-major then condition-order: all r16 first, then all r32;
# within a rank, seen_unseen (the validation gate) first.
#
# WHY a config file: future sweeps are ONE edit (copy sweep.conf, change the
# arrays) — no script surgery. This launcher and run_one_model.slurm never need
# to change to add a model, a rank, or a condition.
#
# Usage (repo root on lab-ws):
#   bash training/launch_sweep.sh                          # use training/sweep.conf
#   bash training/launch_sweep.sh --config training/x.conf # a different sweep
#   bash training/launch_sweep.sh --dry-run                # print, submit nothing
#   bash training/launch_sweep.sh --from 4                 # start at job 4 (resume)
#   bash training/launch_sweep.sh --only-model qwen9b      # filter by model KEY
#   bash training/launch_sweep.sh --only-rank 16           # filter by rank
#   bash training/launch_sweep.sh --no-benchmark           # train+eval only
#   bash training/launch_sweep.sh --after <jobid>          # chain after an existing job
#
# Monitor:
#   squeue -u $USER
#   tail -F training/logs/mm_<sweep>_r<rank>_<cond>.out
#   scancel <jobid>   # one link; downstream still runs (afterany)
#   scancel -u $USER  # nuke everything

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# --- flags ---
CONFIG="${SCRIPT_DIR}/sweep.conf"
DRY_RUN=0
START_FROM=1
ONLY_MODEL=""
ONLY_RANK=""
NO_BENCHMARK=0
PREV_JOBID=""           # --after: chain the first job behind an existing job id
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --from)          START_FROM="$2"; shift 2 ;;
        --only-model)    ONLY_MODEL="$2"; shift 2 ;;
        --only-rank)     ONLY_RANK="$2"; shift 2 ;;
        --no-benchmark)  NO_BENCHMARK=1; shift ;;
        --after)         PREV_JOBID="$2"; shift 2 ;;
        -h|--help)       grep '^#' "$0" | head -40; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
# shellcheck source=sweep.conf
source "${CONFIG}"

# --- validate required arrays/vars from the config ---
: "${SWEEP_NAME:?SWEEP_NAME not set in config}"
[[ ${#MODELS[@]} -gt 0 ]]     || { echo "MODELS empty in config" >&2; exit 1; }
[[ ${#RANKS[@]} -gt 0 ]]      || { echo "RANKS empty in config" >&2; exit 1; }
[[ ${#CONDITIONS[@]} -gt 0 ]] || { echo "CONDITIONS empty in config" >&2; exit 1; }

# allow CLI --no-benchmark to override config BENCHMARK
[[ "${NO_BENCHMARK}" == "1" ]] && BENCHMARK=0
BENCHMARK="${BENCHMARK:-1}"

# --- shared env forwarded to every job (from config, with sane defaults) ---
COMMON_ENV=(
    "GPU_PROFILE=${GPU_PROFILE:-rtx_pro_6000_96gb}"
    "IMAGE_MAX_EDGE=${IMAGE_MAX_EDGE:-512}"
    "EARLY_STOPPING=${EARLY_STOPPING:-1}"
    "WARMUP_STEPS=${WARMUP_STEPS:-100}"
    "WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}"
    "LR_SCHEDULER=${LR_SCHEDULER:-cosine}"
    "OPTIM=${OPTIM:-adamw_8bit}"
    "SAVE_MERGED=${SAVE_MERGED:-0}"
    "REPORT_TO=${REPORT_TO:-none}"
    "SEED=${SEED:-3407}"
    "PROBE_TARGET_HIGH=${PROBE_TARGET_HIGH:-0.98}"
    "PROBE_TARGET_LOW=${PROBE_TARGET_LOW:-0.94}"
    "PROBE_RESERVE_MB=${PROBE_RESERVE_MB:-1536}"
    "PROBE_FLAT_MARGIN=${PROBE_FLAT_MARGIN:-0.0}"
    "PROBE_HARD_CAP=${PROBE_HARD_CAP:-128}"
)
[[ "${BENCHMARK}" == "1" ]] || COMMON_ENV+=("SKIP_BENCHMARK=1")
[[ -n "${BENCHMARK_MODES:-}" ]] && COMMON_ENV+=("BENCHMARK_MODES=${BENCHMARK_MODES}")

# EXTRA_ENV: arbitrary "KEY=value" pairs a config can forward to every job
# without touching this launcher. Used by ablation configs to set e.g.
# SV_ANGLES=along_fwd (urban tasks -> sat + forward SV only, see data.py) and
# EXCLUDE_TOPICS=green_space. --export=ALL in run_one_model.slurm carries them
# through to train.py. Defined in the config as a bash array:
#   EXTRA_ENV=("SV_ANGLES=along_fwd" "EXCLUDE_TOPICS=green_space")
if [[ -n "${EXTRA_ENV+x}" ]]; then
    for kv in "${EXTRA_ENV[@]}"; do
        COMMON_ENV+=("${kv}")
    done
fi

LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# --- build the flat job list: rank-major, then model, then condition ---
#     so the chain is: all r16 (m1 c1,c2,c3 / m2 ...), then all r32, ...
JOBS=()   # each: "mkey|base|family|name|R|ALPHA|LR|epochs|ckey|split|textonly"
for r in "${RANKS[@]}"; do
    IFS='|' read -r R ALPHA LR <<< "${r}"
    [[ -n "${ONLY_RANK}" && "${R}" != "${ONLY_RANK}" ]] && continue
    for m in "${MODELS[@]}"; do
        IFS='|' read -r mkey base family name <<< "${m}"
        [[ -n "${ONLY_MODEL}" && "${mkey}" != "${ONLY_MODEL}" ]] && continue
        epochs="${EPOCHS[$mkey]:-${EPOCHS_DEFAULT:-9}}"
        for c in "${CONDITIONS[@]}"; do
            IFS='|' read -r ckey split textonly <<< "${c}"
            JOBS+=("${mkey}|${base}|${family}|${name}|${R}|${ALPHA}|${LR}|${epochs}|${ckey}|${split}|${textonly}")
        done
    done
done

[[ ${#JOBS[@]} -gt 0 ]] || { echo "No jobs (check --only-model/--only-rank)." >&2; exit 1; }

echo "═════════════════════════════════════════════════════════════"
echo "Sweep '${SWEEP_NAME}'  —  ${#JOBS[@]} jobs (model × rank × condition)"
echo "Config:    ${CONFIG}"
echo "Chained:   afterany on 1 GPU (strictly sequential)"
echo "Shared:    edge=${IMAGE_MAX_EDGE:-512}  ${OPTIM:-adamw_8bit}  ${LR_SCHEDULER:-cosine}"
echo "           probe→${PROBE_TARGET_HIGH:-0.98} of card, reserve ${PROBE_RESERVE_MB:-1536}MB"
echo "Benchmark: $([[ "${BENCHMARK}" == "1" ]] && echo "ON (${BENCHMARK_MODES:-sat+stv4 sat_only stv_only})" || echo "OFF")"
[[ -n "${PREV_JOBID}" ]] && echo "First job chains after existing job: ${PREV_JOBID}"
echo "═════════════════════════════════════════════════════════════"
echo

prev_jobid="${PREV_JOBID}"
chain_summary=""
for i in "${!JOBS[@]}"; do
    idx=$((i + 1))
    (( idx < START_FROM )) && continue

    IFS='|' read -r mkey base family name R ALPHA LR epochs ckey split textonly <<< "${JOBS[$i]}"

    model_key="mm_${SWEEP_NAME}_r${R}_${ckey}"
    output_dir="training/runs/${model_key}"
    log_file="${LOGS_DIR}/${model_key}.out"

    env_pairs=("${COMMON_ENV[@]}")
    env_pairs+=("BASE_MODEL=${base}")
    env_pairs+=("MODEL_FAMILY=${family}")
    env_pairs+=("MODEL_NAME=${name} (r${R})")
    env_pairs+=("LORA_R=${R}")
    env_pairs+=("LORA_ALPHA=${ALPHA}")
    env_pairs+=("LEARNING_RATE=${LR}")
    env_pairs+=("NUM_EPOCHS=${epochs}")
    env_pairs+=("SPLIT_STRATEGY=${split}")
    env_pairs+=("TEXT_ONLY=${textonly}")
    env_pairs+=("OUTPUT_DIR=${output_dir}")
    env_pairs+=("MODEL_KEY=${model_key}")

    sbatch_args=(
        --job-name="${model_key}"
        --output="${log_file}"
        --error="${log_file}"
    )
    [[ -n "${prev_jobid}" ]] && sbatch_args+=(--dependency="afterany:${prev_jobid}")

    cmd=(env "${env_pairs[@]}" sbatch "${sbatch_args[@]}" training/run_one_model.slurm)

    echo "─────────────────────────────────────────────────────────────"
    echo "[${idx}/${#JOBS[@]}] ${model_key}"
    echo "  model:   ${name} (family=${family})  rank=${R}/α=${ALPHA}  lr=${LR}  epochs=${epochs}"
    echo "  base:    ${base}"
    echo "  split:   ${split}   text_only=${textonly}"
    echo "  output:  ${output_dir}"
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
echo "First log:  tail -F ${LOGS_DIR}/$(IFS='|'; read -r mk b f n R A L e ck s t <<< "${JOBS[$((START_FROM-1))]}"; echo "mm_${SWEEP_NAME}_r${R}_${ck}.out")"
echo "Cancel one: scancel <jobid>  (downstream still runs, afterany)"
echo "Cancel all: scancel -u \$USER"
