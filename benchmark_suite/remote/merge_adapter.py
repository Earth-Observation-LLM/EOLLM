#!/usr/bin/env python3
"""merge_adapter.py — merge a LoRA adapter into its base and save bf16 weights.

Same code path that produced merged_ep2 (job 230): unsloth FastVisionModel loads
the base in 16-bit, PEFT attaches the adapter, save_pretrained_merged writes the
full merged bf16 model. We merge (rather than serve LoRA in vLLM) because these
adapters target the VISION tower too, which vLLM's experimental multimodal-LoRA
path does not reliably apply — merging applies every target module exactly.

Env:
  BASE_MODEL   base id/path     (default: read from adapter_config.json)
  ADAPTER      adapter dir      (required)
  OUT          output dir       (required)
"""
import json
import os
import sys
from pathlib import Path

ADAPTER = os.environ["ADAPTER"]
OUT = os.environ["OUT"]
cfg = json.load(open(Path(ADAPTER) / "adapter_config.json"))
BASE = os.environ.get("BASE_MODEL") or cfg["base_model_name_or_path"]

print(f"Base:    {BASE}", flush=True)
print(f"Adapter: {ADAPTER}", flush=True)
print(f"Out:     {OUT}", flush=True)

from unsloth import FastVisionModel  # noqa: E402

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE,
    load_in_4bit=False, load_in_16bit=True,
    full_finetuning=False,
    use_gradient_checkpointing="unsloth",
    max_seq_length=8192,
)

from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, ADAPTER)
print("Adapter loaded. Merging...", flush=True)

# save_pretrained_merged applies merge_and_unload across ALL target modules
# (vision + language) and writes a standalone bf16 model vLLM can serve.
print("Merged. Saving merged full model (bf16)...", flush=True)
model.save_pretrained_merged(OUT, tokenizer, save_method="merged_16bit")
print(f"DONE -> {OUT}", flush=True)
