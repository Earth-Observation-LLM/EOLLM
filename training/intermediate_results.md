# EOLLM Vision SFT — Training Results (Run 1)

**Date:** 2026-04-17 | **Status:** Complete (400 steps, 96 min)

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-4B (unified VLM) |
| Method | bf16 LoRA (r=16, alpha=16), NOT QLoRA |
| GPU | NVIDIA RTX 5090 (32 GB) |
| Peak VRAM | ~24 GB (74% utilization) |
| GPU utilization | 92–95% |
| Dataset | EOLLM compressed (26,931 train / 6,984 val) |
| Split strategy | splits_per_city (same cities, unseen locations) |
| Batch size | 8 × 3 grad_accum = 24 effective |
| Learning rate | 2e-4 → 0 (linear decay) |
| max_seq_length | 8192 (p99 measured at 1,535 tokens) |
| Trainable params | 38.8M / 4.58B (0.85%) |

## Dataset Summary

- **40 cities** worldwide (Amsterdam → Nairobi)
- **14 question topics:** land use, building height, road type, camera direction, mismatch detection, etc.
- **7 image modes:** satellite with markers, street-view composites, mega grids
- **Images:** 512×512 satellite PNGs + 640×640 street-view JPEGs
- Composites generated on the fly from raw sources (~7 GB vs 221 GB full)

## Training Results

| Metric | Value |
|--------|-------|
| Starting loss (step 10) | 0.364 |
| **Final loss (step 400)** | **0.131** |
| Minimum loss observed | 0.088 (step 370) |
| Random baseline (4-choice MCQ) | ~1.39 |
| **Loss reduction** | **64% from start** |
| Wall clock | 96 minutes |
| Training speed | 1.5 samples/sec |

### Loss Trajectory

```
Step   Loss     LR           Phase
  10   0.364    4.5e-05      ← warmup
  40   0.135    2.0e-04      ← sharp drop (model learns task format)
 100   0.135    1.7e-04      ← stabilizing
 200   0.125    1.1e-04      ← steady improvement
 260   0.099    7.8e-05      ← breaking below 0.1
 370   0.088    2.7e-05      ← best observed
 400   0.131    1.1e-05      ← final (minor fluctuation, normal)
```

See `training_curves.png` for the full plot.

## Evaluation — Base vs Finetuned

3 held-out validation samples:

| Sample | Topic | Gold | Base Model | Finetuned | 
|--------|-------|------|------------|-----------|
| zurich_0115 | land_use | A | C | **A** |
| buenos_aires_0035 | mismatch_mcq | C | *(rambling text)* | **C** |
| mexico_city_0001 | land_use | B | C | **B** |

| Model | Accuracy |
|-------|----------|
| Base Qwen3.5-4B | **0/3 (0%)** |
| **Finetuned (400 steps)** | **3/3 (100%)** |

The base model doesn't understand the task — it generates long explanations instead of single-letter answers. After 400 steps of LoRA fine-tuning, the model correctly answers all 3 held-out samples.

## Key Design Decisions

1. **Mismatch questions need satellite context.** The questions say "the marked satellite image" but `composite_utils` only returns the street-view composite. We explicitly add the `satellite_marked` image for all mismatch modes — without this, the model would be asked about an image it never sees.

2. **camera_direction uses 5 images per sample** (1 query SV + 4 satellite-with-arrow options). This produces ~1,500 tokens/sample vs ~350 for the majority. The 8192 max_seq_length provides ample headroom.

3. **Label masking verified.** Only the assistant response (7 tokens: `<think>\n\n</think>\n\nA<|im_end|>\n`) is trained on. The question, images, and system prompt are all masked.

## Artifacts

| File | Description |
|------|-------------|
| `qwen35-4b-vision-lora/` | LoRA adapter (portable, reload on base) |
| `qwen35-4b-vision-merged/` | Merged bf16 model (for vLLM inference) |
| `training_curves.png` | Loss + LR + distribution plots |
| `training_loss.csv` | Step-by-step training loss log |
| `eval_samples.md` | Base vs finetuned comparison |
| `token_budget.md` | Token length distribution analysis |
| `tuning_notes.md` | Batch size probe VRAM measurements |

## Next Steps

- Full 2-epoch run with periodic **validation loss** tracking
- Accuracy evaluation on full validation set (6,984 samples)
- Per-topic accuracy breakdown (which of the 14 topics does the model struggle with?)
- Benchmark evaluation on the held-out benchmark split (5,240 samples)
