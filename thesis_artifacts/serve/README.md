# Local OpenAI-compatible server for Qwen3.5-4B + LoRAs

vLLM serves Qwen3.5-4B base + both fine-tuned LoRAs from one process on the local RTX 5090.

## Start

```bash
./start_server.sh
```

Listens on `http://0.0.0.0:8000`. Endpoint: `http://localhost:8000/v1`.

## Available models

| `model` field | What |
|---|---|
| `qwen3.5-4b-base` | Untrained Qwen3.5-4B-VL (Unsloth) |
| `pc-base` | + per_city LoRA (62.9% on held-out benchmark) |
| `su-base` | + seen_unseen LoRA (60.4% on held-out benchmark) |

## Smoke test

```bash
conda activate vllm
python test_client.py                       # text-only on all 3
python test_client.py path/to/image.jpg     # multimodal on all 3
```

## Use from anywhere (SSH tunnel)

```bash
# From your laptop abroad:
ssh -L 8000:localhost:8000 ezel@<home-machine>
# Then on the laptop:
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
```

## Stop

```bash
pkill -f "vllm serve"
```

## Notes

- LoRAs r=32, max_loras=2 (both fit in VRAM at once).
- bf16, max_model_len=8192, gpu_mem_util=0.85 (≈27.5 GB of the 32 GB).
- All run inside the `vllm` conda env (Python 3.12, vLLM 0.21.0, torch 2.11, transformers 5.8).
- Log: `/tmp/vllm_server.log`.
