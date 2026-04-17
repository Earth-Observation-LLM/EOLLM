# EOLLM Training — Complete Reference

This document captures everything about the EOLLM Vision SFT training pipeline. Use it as context when opening new sessions.

---

## What This Is

Vision SFT (Supervised Fine-Tuning) of **Qwen3.5-4B** on the EOLLM urban VQA dataset. The model learns to answer multiple-choice questions about urban environments using satellite and street-level imagery across 40 cities worldwide.

## Model

- **Base:** `Qwen/Qwen3.5-4B` — unified VLM (vision + language in one checkpoint, no separate `-VL` variant)
- **Method:** bf16 LoRA (rank 16, alpha 16) — **NOT QLoRA** (Qwen3.5 degrades with 4-bit quantization)
- **Thinking:** OFF by default on Qwen3.5-4B. Non-thinking SFT is the native format
- **Chat template:** ChatML (`<|im_start|>role\n...<|im_end|>`)
- **Trainable params:** 38.8M / 4.58B (0.85%)
- **Unsloth mirror:** `unsloth/Qwen3.5-4B`

## Hardware

- **GPU:** NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120)
- **Stack:** CUDA 12.8, torch 2.10.0+cu128, transformers 5.5.0, unsloth 2026.4.2, triton 3.6.0
- **Environment:** `conda activate unsloth`
- **Peak VRAM usage:** ~17 GB (52% of budget at bs=4) — room to increase batch size

## Dataset

- **Source:** `dataset_content/EODATA_compressed_final/` (~7 GB compressed edition)
- **Full size equivalent:** ~221 GB (composites pre-rendered). Compressed edition generates composites on the fly via `composite_utils.py`
- **Split used:** `splits_per_city` (same cities, unseen locations in validation)

| Set | Records | Unique locations |
|-----|---------|-----------------|
| Train | 26,931 | 3,186 |
| Validation | 6,984 | — |
| Benchmark | 5,240 | — |

### Topics (14)

amenity_richness, building_height, camera_direction, green_space, junction_type, land_use, mismatch_binary_easy, mismatch_binary_hard, mismatch_mcq_easy, mismatch_mcq_hard, road_surface, road_type, transit_density, urban_density

Each topic has ~2,000 training records (except green_space: 1,500 and building_height: 1,431).

### Image Modes and How They Map to Training

| image_mode | % of train | Images shown to model | Actual tokens |
|---|---|---|---|
| satellite_marked | 63% | 1 (satellite with red dot) | ~351 |
| streetview_mega | 15% | 2 (sat_marked + 1024px mega grid) | ~912 |
| streetview_binary | 7.6% | 2 (sat_marked + SV composite) | ~587 |
| satellite_arrow | 7.4% | 5 (query SV + 4 sat-with-arrows) | ~1,503 |
| streetview_composite | 7.3% | 2 (sat_marked + SV composite) | ~587 |

**Critical design decision:** Mismatch questions say "the marked satellite image" but `composite_utils.get_images_for_question()` only returns the street-view composite. `convert_record()` in `train.py` explicitly adds the `satellite_marked` image for mismatch modes.

### Token Distribution (measured on 300 samples with actual processor)

| Percentile | Tokens |
|---|---|
| min | 354 |
| p50 | 380 |
| p90 | 940 |
| p95 | 1,531 |
| p99 | 1,535 |
| max | 1,537 |

`max_seq_length` = 8192 (generous headroom over p99).

## Training Configuration

### Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Batch size | 4 per device | Largest that fit in VRAM (bs=8 OOM'd) |
| Gradient accumulation | 6 | Effective batch = 24 |
| Learning rate | 2e-4 | Standard LoRA starting point |
| LR scheduler | linear (warmup → decay) | |
| Warmup steps | 40 | ~10% of total steps |
| Optimizer | adamw_8bit | Saves VRAM vs full adamw |
| Weight decay | 0.01 | |
| LoRA rank (r) | 16 | Baseline; bump to 32 only if loss plateaus |
| LoRA alpha | 16 | alpha == r is the safe default |
| LoRA targets | all-linear | Covers q/k/v/o/gate/up/down across vision+LM |
| Vision layers | finetuned | Task needs domain-specific visual grounding |
| Precision | bf16 | Native for Blackwell |

### Mandatory Vision SFT Flags

These four must be set or the collator breaks silently:

```python
remove_unused_columns = False
dataset_text_field = ""
dataset_kwargs = {"skip_prepare_dataset": True}
max_length = 8192
```

### Label Masking

`train_on_responses_only=True` with boundaries:
- instruction: `<|im_start|>user\n`
- response: `<|im_start|>assistant\n`

Verified: only 7 tokens unmasked per sample (`<think>\n\n</think>\n\nA<|im_end|>\n`). The empty `<think>` block is Qwen3.5's default non-thinking format.

### System Prompt

```
You are an urban geography expert analyzing satellite and street-level imagery.
Answer the multiple-choice question based on the provided images.
Reply with only the letter of the correct answer (A, B, C, or D).
```

## Run 1 Results (400 steps, ~35% of 1 epoch)

| Metric | Value |
|---|---|
| Training steps | 400 / 1,123 per epoch |
| Wall clock | 96 minutes |
| Starting loss | 0.364 |
| Final loss | 0.131 |
| Min loss | 0.088 (step 370) |
| Random baseline | ~1.39 (ln(4)) |
| Loss reduction | 64% |
| Peak VRAM | 16.9 GB (52% of 32 GB) |

### Eval (3 validation samples — never seen during training)

| Sample | Topic | Gold | Base | Finetuned |
|---|---|---|---|---|
| zurich_0115 | land_use | A | C | **A** |
| buenos_aires_0035 | mismatch_mcq | C | *(rambling text)* | **C** |
| mexico_city_0001 | land_use | B | C | **B** |

**Base: 0/3 (0%) → Finetuned: 3/3 (100%)**

## How to Run

```bash
conda activate unsloth
cd training

# Smoke test (5 steps)
SMOKE_TEST=1 python train.py

# Full run
python train.py

# Override anything
MAX_STEPS=400 NUM_EPOCHS=2 LEARNING_RATE=1e-4 REPORT_TO=wandb python train.py

# Different split strategy
SPLIT_STRATEGY=splits_seen_unseen python train.py

# Force a GPU profile
GPU_PROFILE=rtx_4090_24gb python train.py
```

## File Map

```
training/
├── train.py                    ← main script (auto GPU detect, measure, probe, train, eval)
├── install.sh                  ← dependency checker
├── TRAINING.md                 ← this file
├── dataset_report.md           ← dataset schema analysis
├── intermediate_results.md     ← run 1 results writeup
│
├── training_loss.csv           ← step-by-step loss log (generated at runtime)
├── eval_loss.csv               ← validation loss log (generated at runtime, next run)
├── training_curves.png         ← loss + LR + distribution plots (generated at runtime)
├── token_budget.md             ← token distribution report (generated at runtime)
├── tuning_notes.md             ← batch probe results (generated at runtime)
├── eval_samples.md             ← base vs finetuned comparison (generated at runtime)
│
├── qwen35-4b-vision-lora/      ← LoRA adapter (always saved)
├── qwen35-4b-vision-merged/    ← merged bf16 model (saved if not smoke test)
└── outputs_qwen35_4b_vision/   ← checkpoints
```

## Gotchas / Things That Bite

1. **Never `load_in_4bit=True`** on Qwen3.5 — quality degrades
2. **Transformers v5 required** — older versions silently fail on Qwen3.5
3. **Never `.map()` with PIL images** — use list comprehension
4. **Label masking wrong = loss drops but model learns nothing** — always verify with `check_label_masking()`
5. **One large image pads the whole batch** — resize everything to `image_max_edge` upfront
6. **First epoch looks slow** — Qwen3.5's Mamba/GDN Triton kernels compile on first pass
7. **Two train.py processes = both fighting for GPU** — always kill duplicates
8. **`num_workers > 0` breaks** — PIL Images can't be pickled. Use `num_workers=0`

## Next Steps

- Full 2-epoch run with periodic validation loss
- Per-topic accuracy breakdown on full validation set
- Benchmark evaluation (5,240 held-out samples)
- Experiment with higher LoRA rank (32) if loss plateaus
- Try `splits_seen_unseen` for geographic generalization
