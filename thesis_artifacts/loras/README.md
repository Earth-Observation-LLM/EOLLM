# EOLLM LoRA Adapters

Two LoRA adapters trained on top of **Qwen3.5-4B** for the EOLLM urban VQA task.
Both are r=32, α=32, all-linear, trained with Unsloth.

| LoRA | Split | Epochs | Held-out per_city val | Headline |
|---|---|---:|---:|---|
| `pc_base/` | per_city (same cities, different locations) | 6 | 6,984 | **78.3%** |
| `su_base/` | seen_unseen (held-out cities) | 8 (early-stopped @ ep4) | 8,831 | **70.8%** (ep4 best: 72.4%) |

On the **5,240-Q held-out benchmark** (the public benchmark, evaluated by EzelinyumEvaluator):

| Model | Overall | Seen | Unseen |
|---|---:|---:|---:|
| Qwen3.5-4B base (no LoRA) | 35.5% | 35.7% | 34.4% |
| + `su_base` LoRA | 60.4% | 60.6% | 59.5% |
| + `pc_base` LoRA | (eval in progress) | | |

## Usage (Unsloth + PEFT)

```python
from unsloth import FastVisionModel
from peft import PeftModel

# Load base
model, processor = FastVisionModel.from_pretrained(
    "Qwen/Qwen3.5-4B",        # or local path /home/ain480/training/models/Qwen3.5-4B
    load_in_4bit=False,
    dtype="bfloat16",
)

# Attach an adapter (pick one)
model = PeftModel.from_pretrained(model, "thesis_artifacts/loras/pc_base")
# or
# model = PeftModel.from_pretrained(model, "thesis_artifacts/loras/su_base")

# Switch to inference mode (disables training-time gradient hooks)
FastVisionModel.for_inference(model)

# Run
messages = [{"role": "user", "content": [
    {"type": "image", "image": pil_image},
    {"type": "text", "text": "Your prompt..."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(images=[pil_image], text=[text], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
```

## Provenance

- `pc_base`: copied from `lab-ws:training/training/runs/20260422_074420_rtx_pro_6000_96gb/checkpoints/checkpoint-2022/` (epoch 6, step 2022).
- `su_base`: copied from `lab-ws:training/training/runs/20260423_190939_rtx_pro_6000_96gb/lora/` (final adapter after early-stop).

Training-only files (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`) intentionally excluded — these LoRAs are for **inference only**. To resume training from a checkpoint, pull from the original paths on `lab-ws`.
