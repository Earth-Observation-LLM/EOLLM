#!/usr/bin/env bash
# start.sh — interactive launcher for EOLLM training.
#
# Walks you through the decisions, then launches training in a screen session.
#
# Flags:
#   --local                 local box: skip `module load`, source conda directly,
#                           run train.py directly (bypass train.sh).
#   --resume <run_name>     resume an existing run under training/runs/<name>.
#                           Sets OUTPUT_DIR so train.py's auto-resume triggers.
#                           NUM_EPOCHS must exceed prior epochs or trainer no-ops.
#   -y | --yes              non-interactive: accept all defaults.
#
# What this asks (unless --yes):
#   1. Auto-stop (early stopping on val accuracy)?
#      - Yes → asks for MAX_EPOCHS (safety cap)
#      - No  → asks for exact NUM_EPOCHS
#   2. W&B logging on/off?
#   3. Smoke test or real run?
#
# What this does NOT ask:
#   - Batch size. train.py runs a real-data GPU probe at sizes [1, 2, 4, 8,
#     16, 24, 32], picks the biggest that fits, backs off one tier for safety.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SESSION="${SESSION:-eollm_train}"
CONDA_ENV="${CONDA_ENV:-unsloth}"
TOOLS_ENV="${TOOLS_ENV:-tools}"
BASE_MODEL_DIR="${SCRIPT_DIR}/models/Qwen3.5-4B"
DATASET_DIR="${SCRIPT_DIR}/dataset_content/EODATA_compressed_final"
RUNS_DIR="${SCRIPT_DIR}/training/runs"
LOGS_DIR="${SCRIPT_DIR}/training/logs"

# --- flag parsing ---

LOCAL_MODE=0
RESUME_NAME=""
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)          LOCAL_MODE=1; shift ;;
        --resume)         [[ $# -ge 2 ]] || { echo "ERROR: --resume needs a name" >&2; exit 2; }
                          RESUME_NAME="$2"; shift 2 ;;
        --resume=*)       RESUME_NAME="${1#--resume=}"; shift ;;
        -y|--yes)         ASSUME_YES=1; shift ;;
        -h|--help)        sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)                echo "ERROR: unknown arg '$1'. See --help." >&2; exit 2 ;;
    esac
done

# --- helpers ---

red()    { printf '\033[31m%s\033[0m' "$*"; }
green()  { printf '\033[32m%s\033[0m' "$*"; }
yellow() { printf '\033[33m%s\033[0m' "$*"; }
bold()   { printf '\033[1m%s\033[0m'  "$*"; }

die() { echo "$(red ERROR:) $*" >&2; exit 1; }

ask() {
    local prompt="$1" default="${2:-}" ans
    if (( ASSUME_YES )); then echo "${default}"; return; fi
    if [[ -n "${default}" ]]; then
        read -r -p "${prompt} [${default}]: " ans
        echo "${ans:-${default}}"
    else
        read -r -p "${prompt}: " ans
        echo "${ans}"
    fi
}

ask_yn() {
    local prompt="$1" default="${2:-n}" ans
    if (( ASSUME_YES )); then
        [[ "${default}" == "y" ]] && return 0 || return 1
    fi
    local hint="[y/N]"; [[ "${default}" == "y" ]] && hint="[Y/n]"
    while true; do
        read -r -p "${prompt} ${hint} " ans
        ans="${ans:-${default}}"
        case "${ans,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *) echo "  (type y or n)" ;;
        esac
    done
}

ask_int() {
    local prompt="$1" default="${2:-}" ans
    while true; do
        ans="$(ask "${prompt}" "${default}")"
        if [[ "${ans}" =~ ^[0-9]+$ ]] && (( ans > 0 )); then
            echo "${ans}"; return 0
        fi
        echo "  (must be a positive integer)"
    done
}

# --- banner ---

echo
bold "╔══════════════════════════════════════════════════════════╗"; echo
bold "║  EOLLM Training Launcher — interactive pre-flight        ║"; echo
bold "╚══════════════════════════════════════════════════════════╝"; echo
echo

# --- step 1: preflight ---

echo "$(bold '[1/5]') Preflight: $( (( LOCAL_MODE )) && echo "local mode (no module load)" || echo "loading modules + checking env")..."

resolve_env_prefix() {
    local name="$1"
    conda env list | awk -v n="${name}" '$1 == n { for (i=2;i<=NF;i++) if ($i ~ /^\//) { print $i; exit } }'
}

if (( LOCAL_MODE )); then
    # Local box: find conda base, source it, use system `screen`.
    CONDA_BASE=""
    for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3" "/opt/conda"; do
        [[ -f "${p}/etc/profile.d/conda.sh" ]] && { CONDA_BASE="${p}"; break; }
    done
    if [[ -z "${CONDA_BASE}" ]] && command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] \
        || die "conda not found. Set CONDA_BASE or install miniconda."

    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    conda env list | awk '{print $1}' | grep -qxF "${CONDA_ENV}" \
        || die "conda env '${CONDA_ENV}' not found."

    command -v screen >/dev/null 2>&1 \
        || die "'screen' not on PATH. apt install screen (or conda install -n ${CONDA_ENV} -c conda-forge screen)"
    SCREEN_BIN="$(command -v screen)"

    echo "  $(green ✓) conda base:  ${CONDA_BASE}"
    echo "  $(green ✓) conda env:   ${CONDA_ENV}"
else
    # Lab workstation: module load + tools env for screen.
    module load conda/latest cuda/13.0.2 2>/dev/null || die "'module load' failed. Is this the lab workstation? Use --local for your own machine."

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    CONDA_ENV_PREFIX="$(resolve_env_prefix "${CONDA_ENV}")"
    TOOLS_ENV_PREFIX="$(resolve_env_prefix "${TOOLS_ENV}")"
    [[ -n "${CONDA_ENV_PREFIX}" ]] || die "Conda env '${CONDA_ENV}' missing. Run: bash remote_setup.sh"
    [[ -n "${TOOLS_ENV_PREFIX}" ]] || die "Conda env '${TOOLS_ENV}' missing. Run: bash remote_setup.sh"

    SCREEN_BIN="${TOOLS_ENV_PREFIX}/bin/screen"
    [[ -x "${SCREEN_BIN}" ]] || die "screen not found at ${SCREEN_BIN} — run remote_setup.sh"

    echo "  $(green ✓) conda envs:  ${CONDA_ENV}, ${TOOLS_ENV}"
fi

[[ -d "${BASE_MODEL_DIR}" ]] || die "Base model missing at ${BASE_MODEL_DIR}"
[[ -f "${DATASET_DIR}/splits_per_city/train.jsonl" ]] \
    || die "Dataset missing at ${DATASET_DIR}/splits_per_city/"

echo "  $(green ✓) screen:      ${SCREEN_BIN}"
echo "  $(green ✓) base model:  ${BASE_MODEL_DIR}"
echo "  $(green ✓) dataset:     ${DATASET_DIR}"
echo

# --- resume resolution ---

RESUME_DIR=""
prior_epochs=""
if [[ -n "${RESUME_NAME}" ]]; then
    RESUME_DIR="${RUNS_DIR}/${RESUME_NAME}"
    [[ -d "${RESUME_DIR}" ]] || die "Resume target not found: ${RESUME_DIR}"
    shopt -s nullglob
    ckpt_list=( "${RESUME_DIR}/checkpoints"/checkpoint-* )
    shopt -u nullglob
    (( ${#ckpt_list[@]} > 0 )) || die "No checkpoints in ${RESUME_DIR}/checkpoints — cannot resume."
    latest_ckpt="$(ls -d "${RESUME_DIR}/checkpoints"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' | sort -n | tail -1 | awk '{print $2}')"
    latest_step="$(basename "${latest_ckpt}" | sed 's/checkpoint-//')"
    echo "$(bold '[resume]') Resuming from: ${RESUME_NAME}"
    echo "  Latest checkpoint: checkpoint-${latest_step}"
    if [[ -f "${latest_ckpt}/trainer_state.json" ]]; then
        prior_epochs="$(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"{s.get('epoch',0):.2f}\")" "${latest_ckpt}/trainer_state.json" 2>/dev/null || echo "")"
        [[ -n "${prior_epochs}" ]] && echo "  Prior epoch:       ${prior_epochs}"
    fi
    echo
fi

# --- step 2: existing session check ---

echo "$(bold '[2/5]') Checking for existing training session..."
if "${SCREEN_BIN}" -ls 2>/dev/null | grep -qE "\.${SESSION}[[:space:]]"; then
    echo "  $(yellow '⚠') screen session '${SESSION}' is ALREADY RUNNING."
    echo
    "${SCREEN_BIN}" -ls | grep -E "\.${SESSION}[[:space:]]" || true
    echo
    echo "  1) attach  — reattach (Ctrl-a d to detach)"
    echo "  2) kill    — kill it and start fresh"
    echo "  3) abort   — exit launcher, leave session alone"
    choice="$(ask 'Your choice [1/2/3]' '1')"
    case "${choice}" in
        1|attach) exec "${SCREEN_BIN}" -x "${SESSION}" ;;
        2|kill)
            "${SCREEN_BIN}" -S "${SESSION}" -X quit
            echo "  $(yellow '→') killed. Continuing to launch a new run..."
            ;;
        *) echo "  Aborted."; exit 0 ;;
    esac
else
    echo "  $(green ✓) no existing session"
fi
echo

# --- step 3: GPU state ---

echo "$(bold '[3/5]') GPU state"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv 2>&1 | sed 's/^/  /'
echo
other=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [[ -z "${other}" ]]; then
    echo "  $(green '(no other GPU processes — all yours)')"
else
    echo "  Other users on GPU right now:"
    echo "${other}" | sed 's/^/    /'
    echo "  $(yellow 'NOTE:') shared GPU. Probe will see CURRENT free VRAM."
fi
echo

# --- step 4: training prompts ---

echo "$(bold '[4/5]') Training configuration"
echo
echo "  Auto-stop = early stopping on validation accuracy."
echo "  If val accuracy drops for 2 consecutive eval steps, training stops"
echo "  and the BEST checkpoint is kept (not the latest). Patience=2."
echo

if ask_yn "Enable auto-stop (early stopping on val accuracy)?" "y"; then
    EARLY_STOPPING=1
    echo "  $(green '→') auto-stop: ON (patience=2 on eval_accuracy)"
    NUM_EPOCHS="$(ask_int 'Max epochs (safety cap — stops earlier if accuracy plateaus)' '6')"
    echo "  $(green '→') max epochs: ${NUM_EPOCHS}"
else
    EARLY_STOPPING=0
    echo "  $(yellow '→') auto-stop: OFF (will run for the full epoch count)"
    NUM_EPOCHS="$(ask_int 'How many epochs?' '2')"
    echo "  $(green '→') epochs: ${NUM_EPOCHS}"
fi

# Warn if resuming a run that already hit this epoch count
if [[ -n "${RESUME_DIR}" && -n "${prior_epochs}" ]]; then
    if awk "BEGIN{exit !(${prior_epochs} >= ${NUM_EPOCHS})}"; then
        echo "  $(red '⚠ WARNING:') resume target is at epoch ${prior_epochs}, NUM_EPOCHS=${NUM_EPOCHS}."
        echo "  Trainer will exit immediately. Bump NUM_EPOCHS higher."
        ask_yn "  Continue anyway?" "n" || { echo "Aborted."; exit 0; }
    fi
fi
echo

if ask_yn "Enable W&B (Weights & Biases) logging?" "n"; then
    REPORT_TO=wandb
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        # Check if wandb is already logged in on this machine
        if ! conda run -n "${CONDA_ENV}" python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
            echo "  $(yellow 'NOTE:') WANDB_API_KEY unset and wandb not logged in."
            echo "         Run 'wandb login' now, or training will log offline."
            if ! ask_yn "  Continue anyway?" "y"; then
                die "Aborted by user."
            fi
        fi
    fi
    WANDB_PROJECT="$(ask 'W&B project name' 'eollm-vision-sft')"
else
    REPORT_TO=none
fi
echo

if ask_yn "Run as SMOKE TEST first (5 steps, validates plumbing)?" "n"; then
    SMOKE_TEST=1
    echo "  $(yellow '→') smoke test: ON — epochs / auto-stop ignored, only 5 steps"
else
    SMOKE_TEST=0
fi
echo

# --- step 5: confirm + launch ---

echo "$(bold '[5/5]') Launch summary"
if [[ -n "${RESUME_DIR}" ]]; then
    echo "  Mode:         $(green RESUME) from ${RESUME_NAME}"
    echo "  OUTPUT_DIR:   ${RESUME_DIR}"
else
    echo "  Mode:         fresh run (timestamped dir)"
fi
(( LOCAL_MODE )) && echo "  Host:         $(green LOCAL) (no module load)" || echo "  Host:         lab workstation"
echo "  Profile:      auto-detect"
echo "  Batch size:   AUTO (probed by train.py against real data)"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "  Mode:         $(yellow 'SMOKE TEST') (5 steps)"
else
    if [[ "${EARLY_STOPPING}" == "1" ]]; then
        echo "  Stop policy:  auto-stop (patience=2) with max ${NUM_EPOCHS} epochs"
    else
        echo "  Stop policy:  ${NUM_EPOCHS} epochs, no early stop"
    fi
fi
echo "  W&B:          ${REPORT_TO}"
[[ "${REPORT_TO}" == "wandb" ]] && echo "  Project:      ${WANDB_PROJECT}"
echo "  screen:       session '${SESSION}'  (detach: Ctrl-a d)"
echo "  Log:          logs/training_<timestamp>.log (symlinked as logs/current.log)"
echo

if ! ask_yn "Launch training?" "y"; then
    echo "Aborted."
    exit 0
fi

export SMOKE_TEST REPORT_TO EARLY_STOPPING
[[ "${SMOKE_TEST}" == "0" ]] && export NUM_EPOCHS
[[ "${REPORT_TO}" == "wandb" ]] && export WANDB_PROJECT
[[ -n "${RESUME_DIR}" ]] && export OUTPUT_DIR="${RESUME_DIR}"

if (( LOCAL_MODE )); then
    # Local: launch train.py directly in a screen session. Bypasses train.sh
    # entirely since train.sh also runs `module load`.
    mkdir -p "${LOGS_DIR}"
    ts="$(date +%Y%m%d_%H%M%S)"
    log="${LOGS_DIR}/training_${ts}.log"
    runner="${LOGS_DIR}/.runner_local_${ts}.sh"

    cat > "${runner}" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
exec > >(tee -a "${log}") 2>&1

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
cd "${SCRIPT_DIR}"

echo "=========================================================="
echo "Host:       \$(hostname) (LOCAL)"
echo "Time:       \$(date)"
echo "GPU:        \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:     \$(python --version)"
echo "Log:        ${log}"
echo "Mode:       ${RESUME_NAME:+RESUME from ${RESUME_NAME}}${RESUME_NAME:-fresh run}"
echo "Env vars:   NUM_EPOCHS=\${NUM_EPOCHS:-unset} SMOKE_TEST=\${SMOKE_TEST:-0} REPORT_TO=\${REPORT_TO:-none} EARLY_STOPPING=\${EARLY_STOPPING:-0}"
echo "=========================================================="
echo

python -u training/train.py
rc=\$?

echo
echo "=== train.py exited with status \$rc at \$(date) ==="
echo "Session stays open 60s. Ctrl-a d to detach."
sleep 60
exit \$rc
RUNNER_EOF
    chmod +x "${runner}"
    ln -sfn "${log}" "${LOGS_DIR}/current.log"

    env_export=""
    for v in NUM_EPOCHS SMOKE_TEST REPORT_TO EARLY_STOPPING WANDB_PROJECT \
             WANDB_API_KEY OUTPUT_DIR GPU_PROFILE DATASET_DIR BASE_MODEL SEED \
             MAX_STEPS BATCH_SAFETY_FRAC SPLIT_STRATEGY SKIP_PROBE BATCH_SIZE; do
        [[ -n "${!v:-}" ]] && env_export+="export ${v}=$(printf '%q' "${!v}"); "
    done

    echo
    echo "$(green '→') Launching screen session '${SESSION}'..."
    "${SCREEN_BIN}" -dmS "${SESSION}" bash -lc "${env_export}exec '${runner}'"
    sleep 1
    "${SCREEN_BIN}" -ls 2>/dev/null | grep -qE "\.${SESSION}[[:space:]]" \
        || die "screen session failed to start. Runner: ${runner}"

    echo
    echo "$(green ✓) Training launched."
    echo "  Log:      ${log}"
    echo "  Symlink:  ${LOGS_DIR}/current.log"
    echo
    echo "Monitor:"
    echo "  screen -x ${SESSION}           # attach (Ctrl-a d to detach)"
    echo "  tail -F ${LOGS_DIR}/current.log"
    echo "  nvidia-smi -l 2"
    echo "  screen -S ${SESSION} -X quit  # stop"
else
    echo
    echo "$(green '→') Handing off to train.sh start..."
    echo
    exec bash "${SCRIPT_DIR}/train.sh" start
fi
