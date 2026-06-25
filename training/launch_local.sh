#!/usr/bin/env bash
# launch_local.sh — run the SMALL models (4B, E2B) locally on the RTX 5090 32 GB,
# in parallel with the big models running on lab-ws. No SLURM — runs train.py
# directly, one (model, condition) at a time, sequentially.
#
# Pairs with launch_multimodel.sh (which runs the BIG models on lab-ws). Split
# rationale: 27B/31B can't fit the 32 GB 5090, but 4B/E2B do — so run them here
# while the Pro 6000 grinds the big two. ~Halves total wall-clock.
#
# Each (model, condition) runs the full local pipeline: train -> full-val eval.
# NO benchmark step here (EzelinyumEvaluator lives on lab-ws; benchmark the
# local LoRAs by rsync-ing them there, or run eval_base.py full-val locally,
# which this does).
#
# Usage (from repo root):
#   bash training/launch_local.sh --dry-run        # preview the 6 jobs
#   bash training/launch_local.sh                  # run all 6 (4B+E2B × 3 conds)
#   bash training/launch_local.sh --only qwen4b    # just the 4B's 3 conditions
#   SMOKE=1 bash training/launch_local.sh --only qwen4b   # 5-step smoke first
#
# Conda env: `unsloth` must be active (or activatable). Probe picks the bs for
# the 5090 (rtx_5090_32gb profile) — smaller than lab-ws, that's expected.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# LOCAL base-model overrides — models.sh defaults point at the lab-ws cache.
# Locally: 4B is the repo-local copy, E2B is the google id already in the local
# HF cache (verified present). Set BEFORE sourcing models.sh so its
# ${QWEN4B_BASE:-...} default picks these up; E2B id is overridden post-source.
export QWEN4B_BASE="${QWEN4B_BASE:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
source "${SCRIPT_DIR}/models.sh"

# models.sh pins E2B to "unsloth/gemma-4-E2B-it"; the LOCAL cache has the
# "google/gemma-4-E2B-it" snapshot. Rewrite the E2B entry's base in place so we
# use the already-downloaded local copy instead of pulling the unsloth mirror.
LOCAL_E2B_BASE="${LOCAL_E2B_BASE:-google/gemma-4-E2B-it}"
for i in "${!MODELS[@]}"; do
    if [[ "${MODELS[$i]}" == gemma_e2b\|* ]]; then
        IFS='|' read -r mk mb mf mn ml me <<< "${MODELS[$i]}"
        MODELS[$i]="${mk}|${LOCAL_E2B_BASE}|${mf}|${mn}|${ml}|${me}"
    fi
done

# --- flags ---
DRY_RUN=0
ONLY_MODEL="${LOCAL_ONLY:-qwen4b,gemma_e2b}"   # default: both small models
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only)    ONLY_MODEL="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | head -30; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

SMOKE="${SMOKE:-0}"

# --- shared local env (5090 profile, same recipe as the sweep) ---
# GPU_PROFILE auto-detects the 5090; we pin edge=512 + r/alpha=16 to match the
# remote runs. The 5090 profile's own lr is ignored — LEARNING_RATE per model
# from models.sh wins.
#
# SPEED probe policy, local variant (user: "1 GB empty locally is OK"). Same
# flat-margin approach as the remote sweep but with a TIGHTER 1 GB reserve since
# the 32 GB card has far less to give. The probe still picks ANY integer bs (not
# just powers of 2) — geometric climb to find the ceiling, then linear-fit +
# binary search lands on the largest bs leaving ~1 GB free at worst-case batches.
export GPU_PROFILE="${GPU_PROFILE:-rtx_5090_32gb}"
export IMAGE_MAX_EDGE="${IMAGE_MAX_EDGE:-512}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export EARLY_STOPPING="${EARLY_STOPPING:-1}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export OPTIM="${OPTIM:-adamw_8bit}"
export SAVE_MERGED="${SAVE_MERGED:-0}"
# Probe: 1 GB headroom at worst case, push bs as high as fits.
export PROBE_HARD_CAP="${PROBE_HARD_CAP:-64}"
export PROBE_RESERVE_MB="${PROBE_RESERVE_MB:-1024}"
export PROBE_FLAT_MARGIN="${PROBE_FLAT_MARGIN:-0.0}"
export PROBE_TARGET_LOW="${PROBE_TARGET_LOW:-0.92}"
export PROBE_TARGET_HIGH="${PROBE_TARGET_HIGH:-0.99}"
export REPORT_TO="${REPORT_TO:-none}"
export SEED="${SEED:-3407}"

LOG_DIR="training/logs"; mkdir -p "${LOG_DIR}"

echo "Local sweep on $(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo '?')"
echo "Models: ${ONLY_MODEL}   smoke=${SMOKE}   profile=${GPU_PROFILE}"
echo

FAILS=()
for m in "${MODELS[@]}"; do
    IFS='|' read -r mkey base family name lr epochs <<< "${m}"
    # filter to requested small models
    match=0; IFS=',' read -ra _w <<< "${ONLY_MODEL}"
    for w in "${_w[@]}"; do [[ "${w}" == "${mkey}" ]] && match=1; done
    [[ ${match} -eq 0 ]] && continue

    for c in "${CONDITIONS[@]}"; do
        IFS='|' read -r ckey split textonly <<< "${c}"
        key="loc_${mkey}_${ckey}"
        out="training/runs/${key}"
        log="${LOG_DIR}/${key}.log"

        echo "─────────────────────────────────────────────────────────────"
        echo "RUN ${key}: ${name} lr=${lr} epochs=${epochs} split=${split} text_only=${textonly}"
        echo "  output: ${out}   log: ${log}"

        run_env=(
            BASE_MODEL="${base}" MODEL_FAMILY="${family}" MODEL_NAME="${name}"
            LEARNING_RATE="${lr}" NUM_EPOCHS="${epochs}"
            SPLIT_STRATEGY="${split}" TEXT_ONLY="${textonly}"
            OUTPUT_DIR="${out}"
        )
        if [[ "${SMOKE}" == "1" ]]; then
            run_env+=(SMOKE_TEST=1)
            out="${out}__smoke"; run_env+=(OUTPUT_DIR="${out}")
        fi

        if (( DRY_RUN )); then
            echo "  [dry-run] env ${run_env[*]} python training/train.py"
            continue
        fi

        env "${run_env[@]}" python -u training/train.py 2>&1 | tee "${log}"
        rc=${PIPESTATUS[0]}
        if [[ "${rc}" -ne 0 ]]; then
            echo "!!! ${key} FAILED (exit ${rc}) — see ${log}"
            FAILS+=("${key}")
            # In smoke mode, stop on first failure (the point is to catch breakage).
            [[ "${SMOKE}" == "1" ]] && break 2
        fi

        # Full-val eval of the adapter we just saved (skip in smoke).
        if [[ "${SMOKE}" != "1" && -d "${out}/lora" ]]; then
            echo "  full-val eval (eval_base.py)..."
            env BASE_MODEL="${base}" MODEL_FAMILY="${family}" \
                SPLIT_STRATEGY="${split}" TEXT_ONLY="${textonly}" \
                EVAL_ADAPTER="$(cd "${out}/lora" && pwd)" \
                python -u training/eval_base.py 2>&1 | tee "${LOG_DIR}/${key}_eval.log" || \
                echo "  (eval failed, continuing)"
        fi
    done
done

echo "─────────────────────────────────────────────────────────────"
if [[ ${#FAILS[@]} -eq 0 ]]; then
    echo "Local sweep complete ✓"
else
    echo "FAILURES: ${FAILS[*]}"
    exit 1
fi
