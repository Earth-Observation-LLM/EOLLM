#!/usr/bin/env bash
# Serve the local diffusion-gemma teacher under vLLM via the custom Docker image.
#
# The model (nvidia/diffusiongemma-26B-A4B-it-NVFP4) is a block-diffusion VLM
# whose architecture (DiffusionGemmaForBlockDiffusion) only exists in the custom
# vllm/vllm-openai:gemma image — mainline vLLM can't load it. Weights are already
# in ~/.cache/huggingface (18G, NVFP4 + FP8 KV cache); we mount that cache so the
# container reuses them instead of re-downloading.
#
# Usage:
#   ./serve_diffusiongemma.sh            # serve on :8000
#   PORT=8001 ./serve_diffusiongemma.sh  # override port
set -euo pipefail

IMAGE="vllm/vllm-openai:gemma"
MODEL="nvidia/diffusiongemma-26B-A4B-it-NVFP4"   # resolved from the mounted HF cache
PORT="${PORT:-8013}"                              # prior run used :8013
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# The image's ENTRYPOINT is `vllm serve`, so the container CMD is just the model
# + serve flags (no leading "vllm serve").
#
# --gpus all                : expose the RTX 5090 (nvidia container toolkit).
# -v HF cache               : reuse the 18G of already-downloaded weights.
# --ipc=host                : vLLM needs large shared memory for tensor IPC.
# -p PORT:8000              : container serves on 8000; map to host PORT.
# HF_HUB_OFFLINE=1           : weights are local — never hit the network.
# Settings: served-name diffusiongemma, 16k context, 6 images/prompt.
#
# MEMORY TUNING for the 32GB RTX 5090 (the 26B NVFP4 model + FP8 KV cache is
# right at the edge of capacity):
#  - gpu-memory-utilization 0.80: weights alone are 17.93 GiB. At 0.90 util vLLM
#    reserved a 9.83 GiB KV pool (357k tokens — far more than this probe needs),
#    leaving ~0 free, and the sampler warmup's 1 GiB alloc OOM'd. 0.80 caps the
#    budget at ~25 GiB so the KV pool shrinks to a few GiB and the warmup +
#    logits scratch fit. (We don't need a huge KV pool: client concurrency 8,
#    16 seqs, 16k ctx.)
#  - max-num-seqs 16 (not 32): each concurrent sequence reserves KV-cache; 16 is
#    plenty for an offline probe at client concurrency 8 and roughly halves the
#    KV reservation vs 32.
#  - enforce-eager: skips CUDA-graph capture entirely. Graphs cost extra memory
#    and were the final straw in the OOM; eager is slightly slower per step but
#    removes the capture-time spike and the graph memory pool. Drop this flag if
#    you later find there's spare VRAM and want the throughput back.
# DAEMON-DETACHED (-d), NOT under screen and NOT -it. Earlier launches ran the
# container as the foreground of `screen ... bash -c 'exec docker run ...'`; when
# any later screen activity touched that session, a SIGTERM propagated into the
# container and vLLM cleanly shut itself down — twice, the instant the probe
# started. Running detached hands lifecycle to the docker daemon, which is immune
# to shell/screen/TTY teardown. Logs: `docker logs -f dgemma_server`.
docker rm -f dgemma_server >/dev/null 2>&1 || true   # clear any stale container
docker run -d \
  --name dgemma_server \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8013" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE}" \
  "${MODEL}" \
  --served-model-name diffusiongemma \
  --port 8013 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 16 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image": 6}'

echo "diffusiongemma server starting (detached). Follow logs with:"
echo "  docker logs -f dgemma_server"
echo "Readiness:  curl -s http://localhost:${PORT}/v1/models"
