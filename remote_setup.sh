#!/usr/bin/env bash
# remote_setup.sh — set up the `unsloth` conda env on the lab workstation.
#
# Run after the first sync:
#   ssh lab-ws "bash ~/training/remote_setup.sh"
#
# What this does:
#   1. Loads conda + cuda via OpenHPC modules (no sudo needed).
#   2. Creates (or updates) the `unsloth` conda env from environment.yml.
#      Pip pulls torch+torchvision from the CUDA 12.8 index (not PyPI).
#   3. Installs screen into a tiny `tools` env (workstation has no system screen
#      and we want `train.sh` to launch in a detachable session).
#   4. Runs a CUDA sanity check so we fail loudly if torch can't see the GPU.

set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-${HOME}/training}"
ENV_NAME="${ENV_NAME:-unsloth}"
TOOLS_ENV="${TOOLS_ENV:-tools}"
ENV_YML="${REMOTE_DIR}/environment.yml"

echo "=== EOLLM remote setup ==="
echo "Remote dir:  ${REMOTE_DIR}"
echo "Conda env:   ${ENV_NAME} (training) + ${TOOLS_ENV} (screen)"
echo "Env yml:     ${ENV_YML}"
echo

cd "${REMOTE_DIR}"

# --- Load modules (OpenHPC-style) ---
echo "Loading modules..."
module load conda/latest cuda/13.0.2

command -v conda >/dev/null || { echo "ERROR: conda not in PATH after module load" >&2; exit 1; }
echo "conda: $(conda --version)"
echo "nvcc:  $(nvcc --version | tail -1)"
echo

# Accept Anaconda ToS for default channels (conda 24+ requires this on fresh
# systems before first env create). Idempotent — re-running is a no-op.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# Make `conda activate` work inside the non-interactive script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Create/update unsloth env ---
[[ -f "${ENV_YML}" ]] || { echo "ERROR: ${ENV_YML} missing" >&2; exit 1; }

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Updating existing conda env '${ENV_NAME}' (this can take a while)..."
    conda env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune
else
    echo "Creating conda env '${ENV_NAME}' from ${ENV_YML} (~10 GB download)..."
    conda env create -n "${ENV_NAME}" -f "${ENV_YML}"
fi

# --- Install screen into tools env (only if missing) ---
if ! conda env list | awk '{print $1}' | grep -qx "${TOOLS_ENV}"; then
    echo "Creating '${TOOLS_ENV}' env with screen..."
    conda create -n "${TOOLS_ENV}" -c conda-forge -y screen
else
    echo "'${TOOLS_ENV}' env exists, skipping screen install."
fi

# --- Sanity check ---
echo
echo "=== CUDA sanity check ==="
conda activate "${ENV_NAME}"
python - <<'PY'
import sys, torch
print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}  (built for CUDA {torch.version.cuda})")
assert torch.cuda.is_available(), "CUDA not available — driver/wheel mismatch?"
props = torch.cuda.get_device_properties(0)
print(f"GPU          {torch.cuda.get_device_name(0)}  "
      f"({props.total_memory / 1024**3:.1f} GB, CC {props.major}.{props.minor})")
import transformers, unsloth, trl, peft, triton
print(f"transformers {transformers.__version__}")
print(f"unsloth      {unsloth.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
print(f"triton       {triton.__version__}")
PY

echo
echo "=== Setup complete ==="
echo "Activate with: module load conda/latest cuda/13.0.2 && conda activate ${ENV_NAME}"
echo "Launch training with: bash ${REMOTE_DIR}/train.sh start"
echo
echo "Base model must exist at ${REMOTE_DIR}/models/Qwen3.5-4B (already synced)."
echo "Set WANDB_API_KEY before launch if you want W&B logging."
