#!/usr/bin/env bash
# install.sh — Verify the training environment is ready.
#
# Usage:
#   conda activate unsloth
#   bash training/install.sh

set -euo pipefail

echo "=== EOLLM Training — Environment Check ==="

if [[ "${CONDA_DEFAULT_ENV:-}" != "unsloth" ]]; then
    echo "ERROR: activate 'unsloth' conda env first: conda activate unsloth"
    exit 1
fi

echo "Env: $CONDA_DEFAULT_ENV | Python: $(python --version)"

python -c "
import torch
print(f'torch {torch.__version__} | CUDA {torch.version.cuda}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)')

import transformers
assert int(transformers.__version__.split('.')[0]) >= 5, f'Need transformers >= 5, got {transformers.__version__}'
print(f'transformers {transformers.__version__}')

import unsloth; print(f'unsloth {unsloth.__version__}')
import trl; print(f'trl {trl.__version__}')
import peft; print(f'peft {peft.__version__}')
import triton; print(f'triton {triton.__version__}')
from PIL import Image; print('Pillow OK')
import matplotlib; print(f'matplotlib {matplotlib.__version__}')

try:
    import wandb
    logged_in = wandb.api.api_key is not None
    print(f'wandb {wandb.__version__} (logged in: {logged_in})')
    if not logged_in:
        print('  -> Run: wandb login')
except ImportError:
    print('wandb NOT installed -> pip install wandb')
"

echo ""
echo "=== All OK ==="
echo "  python train.py                  # full run"
echo "  SMOKE_TEST=1 python train.py     # smoke test"
echo "  REPORT_TO=wandb python train.py  # with W&B"
