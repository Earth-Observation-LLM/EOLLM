#!/usr/bin/env python3
"""merge_hf.py — merge a LoRA adapter into its base with PLAIN transformers+PEFT.

The unsloth save_pretrained_merged path mangles Qwen3.5-VL key names (triple
`language_model` nesting + leftover PEFT `base_layer` wrappers), which vLLM
cannot load. Plain transformers (AutoModelForImageTextToText) + PeftModel +
merge_and_unload() + save_pretrained() keeps the CANONICAL HF key layout
(model.language_model.layers.*, model.visual.blocks.*) that vLLM expects.

Run in the earth_eval env (same transformers as the vLLM that will serve it).

Env:
  BASE_MODEL   base id/path  (default: from adapter_config.json)
  ADAPTER      adapter dir   (required)
  OUT          output dir    (required)
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER = os.environ["ADAPTER"]
OUT = os.environ["OUT"]
cfg = json.load(open(Path(ADAPTER) / "adapter_config.json"))
BASE = os.environ.get("BASE_MODEL") or cfg["base_model_name_or_path"]

print(f"Base:    {BASE}", flush=True)
print(f"Adapter: {ADAPTER}", flush=True)
print(f"Out:     {OUT}", flush=True)

print("Loading base (bf16, plain transformers)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

print("Attaching adapter...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER)

print("merge_and_unload()...", flush=True)
model = model.merge_and_unload()

print("Saving merged bf16 (canonical HF names)...", flush=True)
model.save_pretrained(OUT, safe_serialization=True)
processor.save_pretrained(OUT)
print(f"DONE -> {OUT}", flush=True)
