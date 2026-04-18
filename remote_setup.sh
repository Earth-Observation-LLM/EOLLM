#!/usr/bin/env bash
# remote_setup.sh — reproduce the local `unsloth` conda env on a remote box.
# Run via ssh after the first sync, e.g.:
#   ssh lab-ws "bash ~/training/remote_setup.sh"
#
# Prereqs on the remote:
#   - miniconda or anaconda installed (conda available in $PATH)
#   - CUDA 12.8+ driver for Blackwell GPUs (RTX 5090 / RTX Pro 6000)
#   - NVIDIA GPU visible to torch
#
# Env reproduction uses environment.yml (exported from the local working setup).
# The local env pins torch==2.10.0+cu128, unsloth==2026.4.2, triton==3.6.0 — do
# NOT let pip resolve these freely on the remote; they must match or Unsloth's
# Triton kernels will not load.

set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-${HOME}/training}"
ENV_NAME="${ENV_NAME:-unsloth}"
ENV_YML="${REMOTE_DIR}/environment.yml"

echo "=== EOLLM remote setup ==="
echo "Remote dir:   ${REMOTE_DIR}"
echo "Conda env:    ${ENV_NAME}"
echo "Env yml:      ${ENV_YML}"
echo

cd "${REMOTE_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH. Install miniconda first." >&2
    exit 1
fi

if [[ ! -f "${ENV_YML}" ]]; then
    echo "ERROR: ${ENV_YML} missing. Re-run sync.sh from the local box." >&2
    exit 1
fi

# Create (or update) the conda env. The `-n ${ENV_NAME}` flag overrides whatever
# `name:` is in the yml, so the env name stays consistent across boxes.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Updating existing conda env '${ENV_NAME}'..."
    conda env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune
else
    echo "Creating conda env '${ENV_NAME}' from ${ENV_YML}..."
    conda env create -n "${ENV_NAME}" -f "${ENV_YML}"
fi

echo
echo "=== Sanity check ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python - <<'PY'
import sys, torch
print(f"python  {sys.version.split()[0]}")
print(f"torch   {torch.__version__}  (CUDA {torch.version.cuda})")
assert torch.cuda.is_available(), "CUDA not available on remote!"
print(f"GPU     {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
import transformers, unsloth, trl, peft, triton
print(f"transformers {transformers.__version__}")
print(f"unsloth      {unsloth.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
print(f"triton       {triton.__version__}")
PY

echo
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then: cd ${REMOTE_DIR}/training && python train.py"
echo
echo "Before training: set WANDB_API_KEY if you want W&B logging, or leave unset"
echo "and training will fall back to REPORT_TO=none."
echo
echo "Base model must exist at ${REMOTE_DIR}/models/Qwen3.5-4B OR be provided via"
echo "BASE_MODEL=<path-or-hub-id> in env. Silent HF fallback is disabled."
