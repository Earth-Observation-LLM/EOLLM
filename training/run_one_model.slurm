#!/usr/bin/env bash
#SBATCH --job-name=eollm_mm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=training/logs/mm-%j.out
#SBATCH --error=training/logs/mm-%j.out
#SBATCH --export=ALL
#
# run_one_model.slurm — one (model, condition) pipeline: train -> full-val eval
#                       -> benchmark register+submit. Self-healing.
#
# Submitted by launch_multimodel.sh (one sbatch per model×condition, chained
# afterany on the single GPU). NOT meant to be run by hand — the launcher sets
# every env var. But you CAN run it standalone for debugging by exporting the
# same vars below and `sbatch training/run_one_model.slurm`.
#
# Required env (set by launcher via env-prefix; --export=ALL forwards them):
#   BASE_MODEL        HF id / path of the base model
#   MODEL_FAMILY      qwen | gemma
#   MODEL_NAME        display name
#   LEARNING_RATE     per-model LR
#   SPLIT_STRATEGY    splits_per_city | splits_seen_unseen
#   TEXT_ONLY         0 | 1
#   OUTPUT_DIR        run dir (training/runs/mm_<model>_<cond>)
#   MODEL_KEY         eval-registry key (e.g. mm_qwen27b_seen_unseen)
#
# Shared knobs (defaults baked here, override via env):
#   NUM_EPOCHS=5  EARLY_STOPPING=1  GPU_PROFILE auto (rtx_pro_6000_96gb)
#   image_max_edge=512 (forced via GPU_PROFILE override below)
#   LoRA r/alpha=16  adamw_8bit  SAVE_MERGED=0  cosine  warmup=100
#   probe per model (NO SKIP_PROBE — big models need a measured bs)
#
# Optional:
#   SKIP_BENCHMARK=1   skip the EzelinyumEvaluator step
#   SKIP_EVAL=1        skip the eval_base.py full-val step
#   EVALUATOR_ROOT=... default /home/ain480/EzelinyumEvaluator

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# --- required (fail loud if launcher didn't set them) ---
: "${BASE_MODEL:?BASE_MODEL not set}"
: "${MODEL_FAMILY:?MODEL_FAMILY not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"
: "${SPLIT_STRATEGY:?SPLIT_STRATEGY not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${MODEL_KEY:?MODEL_KEY not set}"
export BASE_MODEL MODEL_FAMILY MODEL_NAME SPLIT_STRATEGY OUTPUT_DIR

# --- shared knobs with defaults ---
export TEXT_ONLY="${TEXT_ONLY:-0}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export EARLY_STOPPING="${EARLY_STOPPING:-1}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export OPTIM="${OPTIM:-adamw_8bit}"
export SAVE_MERGED="${SAVE_MERGED:-0}"
export REPORT_TO="${REPORT_TO:-none}"
export SEED="${SEED:-3407}"
# image_max_edge=512 for all runs. The probe (not SKIP_PROBE) measures bs per
# model on the real card. We force edge=512 by pinning the profile knob via env:
# config.py PROFILES set edge per-profile, but launch wants 512 universally, so
# we override through GPU_PROFILE-independent means: IMAGE_MAX_EDGE env (see note).
export IMAGE_MAX_EDGE="${IMAGE_MAX_EDGE:-512}"

SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-/home/ain480/EzelinyumEvaluator}"

LOG_DIR="training/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================================="
echo "EOLLM MULTI-MODEL RUN — ${MODEL_NAME} / ${SPLIT_STRATEGY} / text_only=${TEXT_ONLY}"
echo "=========================================================="
echo "Host:        $(hostname)    Time: $(date)    Job: ${SLURM_JOB_ID:-<local>}"
echo "Model key:   ${MODEL_KEY}"
echo "Base model:  ${BASE_MODEL}  (family=${MODEL_FAMILY})"
echo "LR:          ${LEARNING_RATE}   epochs=${NUM_EPOCHS}   edge=${IMAGE_MAX_EDGE}"
echo "Split:       ${SPLIT_STRATEGY}   text_only=${TEXT_ONLY}"
echo "Output:      ${OUTPUT_DIR}"
echo "optim:       ${OPTIM}   save_merged=${SAVE_MERGED}"
echo "=========================================================="
echo

module load conda/latest cuda/13.0.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unsloth

echo "Python: $(python --version)   torch: $(python -c 'import torch;print(torch.__version__)')"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

# ---------------------------------------------------------------------------
# STEP 1 — train
# ---------------------------------------------------------------------------
echo "=== STEP 1: TRAIN ==="
TRAIN_LOG="${LOG_DIR}/mm-${MODEL_KEY}-train.log"
python -u training/train.py 2>&1 | tee "${TRAIN_LOG}"
rc=${PIPESTATUS[0]}
if [[ "${rc}" -ne 0 ]]; then
    echo "!!! TRAIN FAILED (exit ${rc}) — skipping eval+benchmark for ${MODEL_KEY} !!!"
    exit "${rc}"
fi

LORA_DIR="${OUTPUT_DIR}/lora"
if [[ ! -d "${LORA_DIR}" ]]; then
    echo "WARNING: ${LORA_DIR} missing — training saved no LoRA. Skipping eval+benchmark."
    exit 0
fi
LORA_ABS="$(cd "${LORA_DIR}" && pwd)"
echo "=== TRAIN OK — LoRA at ${LORA_ABS} ==="
echo

# ---------------------------------------------------------------------------
# STEP 2 — full validation eval (base+adapter via eval_base.py)
# ---------------------------------------------------------------------------
if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "=== STEP 2: FULL-VAL EVAL — SKIPPED ==="
else
    echo "=== STEP 2: FULL-VAL EVAL (eval_base.py, adapter) ==="
    EVAL_LOG="${LOG_DIR}/mm-${MODEL_KEY}-eval.log"
    # eval_base.py reads BASE_MODEL/MODEL_FAMILY (already exported) and the
    # EVAL_ADAPTER env to load base+LoRA, runs the full val set, writes a
    # verdict.md + accuracy.json under training/runs/<adaptername>_eval_full/.
    env EVAL_ADAPTER="${LORA_ABS}" \
        python -u training/eval_base.py 2>&1 | tee "${EVAL_LOG}"
    erc=${PIPESTATUS[0]}
    if [[ "${erc}" -ne 0 ]]; then
        echo "WARNING: full-val eval failed (exit ${erc}). Continuing to benchmark."
    else
        echo "=== FULL-VAL EVAL OK ==="
    fi
fi
echo

# ---------------------------------------------------------------------------
# STEP 3 — register + submit EzelinyumEvaluator benchmark (downstream job)
# ---------------------------------------------------------------------------
if [[ "${SKIP_BENCHMARK}" == "1" ]]; then
    echo "=== STEP 3: BENCHMARK — SKIPPED ==="
    echo "Done."
    exit 0
fi
if [[ ! -d "${EVALUATOR_ROOT}" ]]; then
    echo "WARNING: EVALUATOR_ROOT missing (${EVALUATOR_ROOT}); skipping benchmark."
    exit 0
fi

echo "=== STEP 3: REGISTER + SUBMIT BENCHMARK (${MODEL_KEY}) ==="
REGISTER_LOG="${LOG_DIR}/mm-${MODEL_KEY}-register.log"

# Backend for the evaluator registry: unsloth for all our LoRA finetunes.
# hf_id is the base model; lora_path is our adapter.
python - "${EVALUATOR_ROOT}/src/registry.py" "${MODEL_KEY}" "${LORA_ABS}" \
         "${BASE_MODEL}" "${MODEL_NAME} (${SPLIT_STRATEGY}, text_only=${TEXT_ONLY})" \
    <<'PY' 2>&1 | tee "${REGISTER_LOG}"
import sys, re, pathlib
reg_path, key, lora_path, hf_id, display = sys.argv[1:6]
p = pathlib.Path(reg_path); src = p.read_text()
if f'"{key}": ModelSpec(' in src:
    print(f"[register] {key} present — patching lora_path.")
    pat = re.compile(r'("' + re.escape(key) + r'": ModelSpec\([^)]*?lora_path=)f?"[^"]*"', re.DOTALL)
    new_src, n = pat.subn(rf'\g<1>"{lora_path}"', src)
    if n: p.write_text(new_src); print(f"[register] patched lora_path -> {lora_path}")
    else: print("[register] WARN: could not patch lora_path; leaving untouched.")
    sys.exit(0)
entry = f'''    "{key}": ModelSpec(
        key="{key}",
        display="{display}",
        backend="unsloth",
        hf_id="{hf_id}",
        lora_path="{lora_path}",
        family="finetuned",
        notes="multi-model sweep — {display}",
    ),
'''
lines = src.splitlines(keepends=True)
inside=False; depth=0; insert_idx=None
for i,line in enumerate(lines):
    if not inside:
        if 'REGISTRY' in line and ': dict[str, ModelSpec] = {' in line:
            inside=True; depth=1
        continue
    depth += line.count('{') - line.count('}')
    if depth==0: insert_idx=i; break
if insert_idx is None:
    print("[register] ERROR: REGISTRY close brace not found."); sys.exit(2)
lines[insert_idx:insert_idx]=[entry]; p.write_text(''.join(lines))
print(f"[register] inserted {key} at line {insert_idx+1}.")
PY
reg_rc=${PIPESTATUS[0]}
if [[ "${reg_rc}" -ne 0 ]]; then
    echo "WARNING: registry edit failed (${reg_rc}). Run benchmark manually:"
    echo "  sbatch ${EVALUATOR_ROOT}/scripts/run_one.sbatch ${MODEL_KEY}"
    exit 0
fi

# Submit THREE benchmark conditions to prove multi-perspective value:
#   sat+stv4   full dual-perspective (the headline number)
#   sat_only   street-view withheld  (how much does sat alone get?)
#   stv_only   satellite withheld    (how much does street-view alone get?)
# If full > sat_only AND full > stv_only on the 9 dual-perspective topics, the
# multi-perspective design is validated. Each is an independent ezeval job;
# they queue after this training job frees the GPU. Set BENCHMARK_MODES to
# override (space-separated).
BENCHMARK_MODES="${BENCHMARK_MODES:-sat+stv4 sat_only stv_only}"
echo "Submitting EzelinyumEvaluator jobs for ${MODEL_KEY} (modes: ${BENCHMARK_MODES})..."
for mode in ${BENCHMARK_MODES}; do
    ( cd "${EVALUATOR_ROOT}" && \
      sbatch --job-name="ezeval_${MODEL_KEY}_${mode}" scripts/run_one.sbatch "${MODEL_KEY}" "${mode}" )
done
echo
echo "=== PIPELINE COMPLETE for ${MODEL_KEY} ==="
echo "Train log:  ${TRAIN_LOG}"
echo "Run dir:    ${OUTPUT_DIR}  (LoRA: ${LORA_ABS})"
exit 0
