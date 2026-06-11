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
