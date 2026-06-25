#!/bin/bash
# Local vLLM OpenAI-compatible server for Qwen3.5-4B + PC-base + SU-base LoRAs.
# Hit it from anywhere with `model: "qwen3.5-4b-base" | "pc-base" | "su-base"`.

set -euo pipefail

# Activate vllm env
source /home/ezel/miniconda3/etc/profile.d/conda.sh
conda activate vllm

BASE="unsloth/Qwen3.5-4B"  # already cached locally
LORAS_DIR="/home/ezel/Development/EOLLM/thesis_artifacts/loras"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Blackwell (RTX 5090, sm_120) — flashinfer JIT arch check trips on SM 12.x.
# Force native (PyTorch) top-k/top-p sampling instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDA_ARCH_LIST="12.0"

exec vllm serve "$BASE" \
    --trust-remote-code \
    --served-model-name qwen3.5-4b-base \
    --enable-lora \
    --lora-modules \
        pc-base="$LORAS_DIR/pc_base" \
        su-base="$LORAS_DIR/su_base" \
    --max-lora-rank 32 \
    --max-loras 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
