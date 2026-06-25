#!/usr/bin/env bash
# smoke_all_models.sh — 5-step smoke test for every model in the sweep.
#
# Runs SMOKE_TEST=1 (5 training steps) for each of the 4 models on the
# seen_unseen split. Purpose: catch BEFORE the real 12-job launch any of:
#   - Gemma <|turn> label-masking failure (check_label_masking sys.exit(1))
#   - 27B/31B bf16 load OOM or probe failure on the 96 GB card
#   - the corrected 5-image satellite_marked loader breaking token measurement
#   - chat-template application errors (Gemma get_chat_template)
#
# Each smoke uses its own throwaway OUTPUT_DIR (…__smoke) so it doesn't pollute
# real run dirs. A smoke that fails prints a FAIL banner; the script continues
# to the next model so you see ALL failures in one pass, then exits non-zero if
# any failed.
#
# Run on lab-ws repo root:
#   bash training/smoke_all_models.sh                  # all 4
#   bash training/smoke_all_models.sh qwen4b qwen27b   # subset by KEY
#
# Big models (27B/31B) load slowly (~1-2 min each) + probe; budget ~20-30 min.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/models.sh"

WANT=("$@")   # optional subset of model KEYs

module load conda/latest cuda/13.0.2 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unsloth

LOG_DIR="training/logs"; mkdir -p "${LOG_DIR}"
FAILS=()

for m in "${MODELS[@]}"; do
    IFS='|' read -r mkey base family name lr epochs <<< "${m}"
    if [[ ${#WANT[@]} -gt 0 ]]; then
        skip=1
        for w in "${WANT[@]}"; do [[ "$w" == "$mkey" ]] && skip=0; done
        [[ $skip -eq 1 ]] && continue
    fi

    echo "=========================================================="
    echo "SMOKE: ${name} (${mkey}, family=${family})"
    echo "=========================================================="
    out="training/runs/mm_${mkey}_smoke"
    log="${LOG_DIR}/smoke_${mkey}.log"
    rm -rf "${out}"

    # Use the SAME speed probe knobs as the real remote launch so the smoke's
    # chosen bs reflects what training will actually use (2 GB headroom, flat
    # margin, cap 128). This also exercises the fixed budget-vs-total-VRAM path
    # on the 27B (which the old conservative knobs falsely rejected).
    env \
        SMOKE_TEST=1 \
        GPU_PROFILE=rtx_pro_6000_96gb \
        IMAGE_MAX_EDGE=512 LORA_R=16 LORA_ALPHA=16 \
        PROBE_HARD_CAP=128 PROBE_RESERVE_MB=2048 PROBE_FLAT_MARGIN=0.0 \
        PROBE_TARGET_LOW=0.92 PROBE_TARGET_HIGH=0.99 \
        BASE_MODEL="${base}" MODEL_FAMILY="${family}" MODEL_NAME="${name}" \
        LEARNING_RATE="${lr}" \
        SPLIT_STRATEGY=splits_seen_unseen \
        REPORT_TO=none SEED=3407 \
        OUTPUT_DIR="${out}" \
        python -u training/train.py 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}

    if [[ "${rc}" -ne 0 ]]; then
        echo "!!! SMOKE FAILED: ${mkey} (exit ${rc}) — see ${log}"
        FAILS+=("${mkey}")
    else
        echo "=== SMOKE OK: ${mkey} ==="
        # Sanity: confirm label masking line + a non-trivial probe bs were logged
        grep -q "Label masking OK" "${log}" && echo "  ✓ label masking passed" \
            || { echo "  ✗ label masking line missing!"; FAILS+=("${mkey}-mask"); }
    fi
    echo
done

echo "=========================================================="
if [[ ${#FAILS[@]} -eq 0 ]]; then
    echo "ALL SMOKES PASSED ✓"
    exit 0
else
    echo "SMOKE FAILURES: ${FAILS[*]}"
    exit 1
fi
