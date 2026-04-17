#!/usr/bin/env bash
# install.sh — Set up the training environment for EOLLM Vision SFT
#
# Assumes: conda env "unsloth" already exists with unsloth installed.
# This script installs/upgrades remaining dependencies inside that env.
#
# Usage:
#   conda activate unsloth
#   bash training/install.sh

set -euo pipefail

echo "=== EOLLM Training — Dependency Check ==="

# Ensure we're in the conda unsloth env
if [[ "${CONDA_DEFAULT_ENV:-}" != "unsloth" ]]; then
    echo "ERROR: activate the 'unsloth' conda environment first:"
    echo "  conda activate unsloth"
    exit 1
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

# Check critical packages
python -c "
import torch
print(f'torch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import transformers
print(f'transformers: {transformers.__version__}')
major = int(transformers.__version__.split('.')[0])
assert major >= 5, f'transformers v5+ required, got {transformers.__version__}'

import unsloth
print(f'unsloth: {unsloth.__version__}')

import trl
print(f'trl: {trl.__version__}')

import peft
print(f'peft: {peft.__version__}')

from PIL import Image
print('Pillow: OK')

import triton
print(f'triton: {triton.__version__}')
"

echo ""
echo "=== All dependencies OK ==="
echo ""
echo "To run training:"
echo "  conda activate unsloth"
echo "  cd $(dirname "$0")"
echo "  python train.py"
