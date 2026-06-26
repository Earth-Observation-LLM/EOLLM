#!/usr/bin/env bash
# Run the raw local Qwen3.5-9B on the EOLLM benchmark via vLLM on the 5090.
#
# Two desktop-stability measures (a full 4-mode run holds ~70GB of decoded image
# pixels in host RAM at once, which froze GNOME):
#   1. ONE MODE PER PROCESS  — host RAM is fully released between modes, so peak
#      stays at a single mode's worklist instead of all four combined.
#   2. CHUNKED generate()    — the vLLM backend feeds prompts in chunks of
#      SUITE_VLLM_CHUNK so it never copies the whole worklist's pixels at once.
#
# flashinfer in the local `vllm` env mis-detects Blackwell (sm120); we sidestep
# it with FlashAttention + the PyTorch-native sampler.
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export FLASHINFER_DISABLE_VERSION_CHECK=1   # jit-cache vs flashinfer version skew
export VLLM_ATTENTION_BACKEND=FLASH_ATTN    # flashinfer attn JIT fails on sm120
export VLLM_USE_FLASHINFER_SAMPLER=0        # flashinfer sampler JIT fails on sm120
export SUITE_VLLM_CHUNK="${SUITE_VLLM_CHUNK:-1024}"

# Smallest-RAM modes first so an early failure costs the least.
MODES=(blind sat_only sv_only full)
for m in "${MODES[@]}"; do
  echo "============================================================"
  echo "=== MODE: $m  (chunk=$SUITE_VLLM_CHUNK) ==="
  echo "============================================================"
  SUITE_MODES="$m" FORCE=1 python run.py --models qwen35_9b_local "$@"
done

# Per-mode processes each overwrote summary.json/all_predictions.jsonl with only
# their own mode; rebuild the combined views from the per-mode report files.
python merge_summary.py results/qwen35_9b_local
