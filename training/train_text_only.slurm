#!/usr/bin/env bash
#SBATCH --job-name=eollm_textonly
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=training/logs/textonly-%j.out
#SBATCH --error=training/logs/textonly-%j.out
#SBATCH --export=ALL

# train_text_only.slurm — non-vision baseline run, end-to-end.
#
# Pipeline (single SLURM allocation, self-healing):
#   1. Smoke test (5 steps) with TEXT_ONLY=1. If it fails, abort — no point
#      spending a 5-epoch slot on a broken pipeline.
#   2. Full training run with TEXT_ONLY=1, same split/seed/epochs/bs as the
#      latest ablations (splits_per_city, SEED=3407, NUM_EPOCHS=5, bs=96,
#      EARLY_STOPPING=1, warmup=100, cosine, no W&B). LR stays at profile
#      default. Output dir: training/runs/textonly_baseline/.
#   3. Register the trained LoRA in EzelinyumEvaluator's registry and submit
#      a benchmark eval as a downstream SLURM job (dependency=afterok on the
#      training job).
#
# All stdout/stderr lands in training/logs/textonly-<jobid>.out.
#
# Submit from the repo root on the workstation:
#   sbatch training/train_text_only.slurm
#
# Override knobs (set as env vars before sbatch):
#   NUM_EPOCHS=5              epochs (default 5, matches ablation runs)
#   BATCH_SIZE=96             effective bs per step (default 96, matches ablations)
#   SPLIT_STRATEGY=...        default splits_per_city (matches ablations)
#   SKIP_SMOKE=1              skip step 1 (only for re-runs you've already
#                              proven work — DO NOT use on a cold setup)
#   SKIP_BENCHMARK=1          don't auto-submit the EzelinyumEvaluator job
#   EVALUATOR_ROOT=...        default /home/ain480/EzelinyumEvaluator

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# --- knobs (with defaults that match the latest ablation runs) ---
export TEXT_ONLY=1
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export EARLY_STOPPING="${EARLY_STOPPING:-1}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-splits_per_city}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export SKIP_PROBE="${SKIP_PROBE:-1}"   # text-only has no image VRAM — probe is overkill
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export REPORT_TO="${REPORT_TO:-none}"
export SEED="${SEED:-3407}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-/home/ain480/EzelinyumEvaluator}"

OUTPUT_DIR_DEFAULT="training/runs/textonly_baseline"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"

LOG_DIR="training/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================================="
echo "EOLLM TEXT-ONLY BASELINE — self-healing pipeline"
echo "=========================================================="
echo "Host:        $(hostname)"
echo "Time:        $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-<no-slurm>}"
echo "Node:        ${SLURM_JOB_NODELIST:-?}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Cwd:         $(pwd)"
echo "----------------------------------------------------------"
echo "TEXT_ONLY:        ${TEXT_ONLY}"
echo "NUM_EPOCHS:       ${NUM_EPOCHS}"
echo "BATCH_SIZE:       ${BATCH_SIZE} (SKIP_PROBE=${SKIP_PROBE})"
echo "SPLIT_STRATEGY:   ${SPLIT_STRATEGY}"
echo "EARLY_STOPPING:   ${EARLY_STOPPING}"
echo "WARMUP_STEPS:     ${WARMUP_STEPS}"
echo "LR_SCHEDULER:     ${LR_SCHEDULER}"
echo "SEED:             ${SEED}"
echo "OUTPUT_DIR:       ${OUTPUT_DIR}"
echo "SKIP_SMOKE:       ${SKIP_SMOKE}"
echo "SKIP_BENCHMARK:   ${SKIP_BENCHMARK}"
echo "EVALUATOR_ROOT:   ${EVALUATOR_ROOT}"
echo "=========================================================="
echo

module load conda/latest cuda/13.0.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unsloth

echo "Python:      $(python --version)"
echo "GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "torch:       $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo

# ---------------------------------------------------------------------------
# Step 1 — smoke test (5 steps)
# ---------------------------------------------------------------------------

if [[ "${SKIP_SMOKE}" == "1" ]]; then
    echo "=== STEP 1: SMOKE TEST — SKIPPED (SKIP_SMOKE=1) ==="
else
    echo "=========================================================="
    echo "=== STEP 1: SMOKE TEST (5 steps, TEXT_ONLY=1) ==="
    echo "=========================================================="

    SMOKE_OUT="${OUTPUT_DIR}__smoke"
    SMOKE_LOG="${LOG_DIR}/textonly-${SLURM_JOB_ID:-local}-smoke.log"

    # Use a separate OUTPUT_DIR for the smoke test so the artifacts don't
    # pollute the real run dir. SMOKE_TEST=1 internally caps to 5 steps.
    env \
        SMOKE_TEST=1 \
        OUTPUT_DIR="${SMOKE_OUT}" \
        SKIP_PROBE=1 \
        BATCH_SIZE="${BATCH_SIZE}" \
        TEXT_ONLY=1 \
        SPLIT_STRATEGY="${SPLIT_STRATEGY}" \
        SEED="${SEED}" \
        REPORT_TO=none \
        python -u training/train.py 2>&1 | tee "${SMOKE_LOG}"
    rc=${PIPESTATUS[0]}

    if [[ "${rc}" -ne 0 ]]; then
        echo
        echo "=========================================================="
        echo "!!! SMOKE TEST FAILED (exit ${rc}) — aborting before full run !!!"
        echo "Log: ${SMOKE_LOG}"
        echo "Smoke run dir: ${SMOKE_OUT}"
        echo "=========================================================="
        exit "${rc}"
    fi
    echo
    echo "=== SMOKE TEST OK — proceeding to full training ==="
    echo
fi

# ---------------------------------------------------------------------------
# Step 2 — full training run
# ---------------------------------------------------------------------------

echo "=========================================================="
echo "=== STEP 2: FULL TRAINING (TEXT_ONLY=1) ==="
echo "=========================================================="
echo "Resolved env:"
for v in NUM_EPOCHS EARLY_STOPPING SPLIT_STRATEGY BATCH_SIZE SKIP_PROBE \
         WARMUP_STEPS LR_SCHEDULER REPORT_TO SEED OUTPUT_DIR TEXT_ONLY \
         GPU_PROFILE DATASET_DIR BASE_MODEL; do
    printf "  %-20s %s\n" "${v}=" "${!v:-<unset>}"
done
echo

TRAIN_LOG="${LOG_DIR}/textonly-${SLURM_JOB_ID:-local}-train.log"

python -u training/train.py 2>&1 | tee "${TRAIN_LOG}"
rc=${PIPESTATUS[0]}

if [[ "${rc}" -ne 0 ]]; then
    echo
    echo "=========================================================="
    echo "!!! TRAINING FAILED (exit ${rc}) — skipping benchmark !!!"
    echo "Log: ${TRAIN_LOG}"
    echo "Run dir: ${OUTPUT_DIR}"
    echo "=========================================================="
    exit "${rc}"
fi

echo
echo "=== TRAINING COMPLETE — run dir: ${OUTPUT_DIR} ==="
echo

# ---------------------------------------------------------------------------
# Step 3 — auto-submit benchmark eval (downstream SLURM job)
# ---------------------------------------------------------------------------

if [[ "${SKIP_BENCHMARK}" == "1" ]]; then
    echo "=== STEP 3: BENCHMARK — SKIPPED (SKIP_BENCHMARK=1) ==="
    echo "Done."
    exit 0
fi

LORA_DIR="${OUTPUT_DIR}/lora"
if [[ ! -d "${LORA_DIR}" ]]; then
    echo "WARNING: ${LORA_DIR} does not exist — training did not save a LoRA."
    echo "Skipping benchmark submission. Inspect: ${OUTPUT_DIR}"
    exit 0
fi
# Make absolute (registry needs an absolute path).
LORA_ABS="$(cd "${LORA_DIR}" && pwd)"

# Model key for the evaluator registry. Same prefix used by the other runs
# (su_base, split1_no_urban, …) so the aggregation report picks it up
# automatically.
MODEL_KEY="textonly_baseline"

REGISTER_LOG="${LOG_DIR}/textonly-${SLURM_JOB_ID:-local}-register.log"

echo "=========================================================="
echo "=== STEP 3: REGISTER + SUBMIT BENCHMARK EVAL ==="
echo "=========================================================="
echo "Evaluator root: ${EVALUATOR_ROOT}"
echo "Model key:      ${MODEL_KEY}"
echo "LoRA path:      ${LORA_ABS}"
echo "Register log:   ${REGISTER_LOG}"
echo

if [[ ! -d "${EVALUATOR_ROOT}" ]]; then
    echo "WARNING: EVALUATOR_ROOT does not exist: ${EVALUATOR_ROOT}"
    echo "Skipping registration. Run the benchmark manually after copying the LoRA."
    exit 0
fi

# Add an entry to the evaluator's registry.py if one isn't there yet.
# Idempotent: a second sbatch on the same OUTPUT_DIR is a no-op for the
# registry edit.
python - "${EVALUATOR_ROOT}/src/registry.py" "${MODEL_KEY}" "${LORA_ABS}" <<'PY' 2>&1 | tee "${REGISTER_LOG}"
import sys, re, pathlib
reg_path, key, lora_path = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(reg_path)
src = p.read_text()
if f'"{key}": ModelSpec(' in src:
    print(f"[register] {key} already in registry — patching lora_path only.")
    # Replace any existing lora_path= line within the block for this key.
    pattern = re.compile(
        r'("' + re.escape(key) + r'": ModelSpec\([^)]*?lora_path=)f?"[^"]*"',
        flags=re.DOTALL,
    )
    new_src, n = pattern.subn(rf'\g<1>"{lora_path}"', src)
    if n == 0:
        print(f"[register] WARNING: could not patch existing lora_path for {key}. Leaving file untouched.")
    else:
        p.write_text(new_src)
        print(f"[register] Updated lora_path for {key} → {lora_path}")
    sys.exit(0)

# Insert a new entry just before the closing brace of REGISTRY.
entry = f'''    "{key}": ModelSpec(
        key="{key}",
        display="Text-only baseline (no images, per_city LoRA)",
        backend="unsloth",
        hf_id=QWEN35_BASE,
        lora_path="{lora_path}",
        family="finetuned",
        notes="TEXT_ONLY=1 ablation — same config as latest ablations, no image inputs.",
    ),
'''

# Find the line that closes REGISTRY: the first '}' that isn't preceded by an
# open '{' on a deeper level. Simplest robust approach: insert before the line
# that contains only '}' (optionally with whitespace) AFTER the REGISTRY block
# opens. Find by scanning backwards from end of file for the last bare '}'.
lines = src.splitlines(keepends=True)
insert_idx = None
inside = False
depth = 0
for i, line in enumerate(lines):
    stripped = line.strip()
    if not inside:
        if 'REGISTRY' in line and ': dict[str, ModelSpec] = {' in line:
            inside = True
            depth = 1
        continue
    depth += line.count('{') - line.count('}')
    if depth == 0:
        insert_idx = i
        break

if insert_idx is None:
    print("[register] ERROR: could not locate closing brace of REGISTRY dict.")
    sys.exit(2)

lines[insert_idx:insert_idx] = [entry]
p.write_text(''.join(lines))
print(f"[register] Inserted {key} into REGISTRY at line {insert_idx+1}.")
PY

reg_rc=${PIPESTATUS[0]}
if [[ "${reg_rc}" -ne 0 ]]; then
    echo "WARNING: registry edit failed (exit ${reg_rc}). Skipping eval submission."
    echo "Add the model manually and run: sbatch ${EVALUATOR_ROOT}/scripts/run_one.sbatch ${MODEL_KEY}"
    exit 0
fi

# Submit the eval. This is a separate SLURM job — it queues independently and
# starts as soon as a GPU is free. Our current job (this script) can exit
# cleanly once the submission is logged.
echo
echo "Submitting EzelinyumEvaluator job for ${MODEL_KEY}..."
cd "${EVALUATOR_ROOT}"
sub_out="$(sbatch --job-name="ezeval_${MODEL_KEY}" scripts/run_one.sbatch "${MODEL_KEY}" 2>&1)"
echo "  ${sub_out}"
echo
echo "=========================================================="
echo "PIPELINE COMPLETE"
echo "=========================================================="
echo "Training:  ${OUTPUT_DIR}"
echo "Train log: ${TRAIN_LOG}"
echo "Eval queued: ${sub_out}"
echo "Eval logs (once started):"
echo "  tail -F ${EVALUATOR_ROOT}/logs/ezeval_${MODEL_KEY}_*.out"
echo "=========================================================="

exit 0
