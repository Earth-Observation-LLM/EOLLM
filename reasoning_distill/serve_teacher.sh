#!/usr/bin/env bash
# Launch the local teacher VLM (Qwen3.5-27B-AWQ) under vLLM for the reasoning
# distillation side-track: dataset discovery + image-ablation gate.
#
# This is the VISION build (no --language-model-only) because the "full pass"
# of the ablation gate feeds images. The "blind pass" just omits the image
# parts from the same request — no separate server needed.
#
# Usage:
#   ./serve_teacher.sh            # serve on :8000
#   PORT=8001 ./serve_teacher.sh  # override port
#
# Run from the `vllm` conda/venv env you already use.
set -euo pipefail

MODEL="cyankiwi/Qwen3.5-27B-AWQ-INT8-INT4"
PORT="${PORT:-8000}"

# flashinfer-jit-cache (0.6.12) and flashinfer (0.6.11) are one patch apart in
# this env; the strict version check aborts startup. They're runtime-compatible,
# so bypass the check rather than churn the env.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# FlashInfer mis-probes the RTX 5090 (Blackwell sm_120) and aborts with
# "requires sm75 or higher" instead of falling back. Two places touch it:
#  1) attention backend — force a Blackwell-friendly one.
#  2) the top-k/top-p sampler — explicitly opt OUT so it uses the PyTorch-native
#     sampler instead of running FlashInfer's broken capability probe. (We run
#     greedy/temp-0 anyway, so the native sampler is fine.)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

# --enable-prefix-caching: the system prompt + reasoning instructions are an
#   identical prefix across all samples, and the full/blind pass of one sample
#   share the same TEXT prefix (they differ only by image tokens). Prefix cache
#   turns that shared text into a one-time cost. This is the main cost lever.
# --max-model-len 32768: Qwen3.5 reasoning needs headroom. We budget 12k tokens
#   for <think> + room for images + the answer, so a trace never gets cut off.
# --reasoning-parser qwen3: vLLM splits <think>...</think> into a separate
#   `reasoning_content` field in the response, and `content` holds just the
#   answer — so logging trace vs answer is clean, no manual string-splitting.
# NOTE: no --language-model-only — we need the vision tower for the full pass.
exec vllm serve "$MODEL" \
  --served-model-name teacher \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image": 6}'
