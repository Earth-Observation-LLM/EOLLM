# EOLLM Multi-Model Vision SFT — Final Training Recipe

The validated recipe for training the 4-model × 3-condition sweep on the
corrected dual-perspective EOLLM urban-VQA dataset. Every value here was
confirmed by a 5-step smoke test on the real hardware (lab-ws RTX Pro 6000
96 GB), not assumed.

---

## 1. The run matrix

**4 models × 3 conditions = 12 runs.**

| Model | Family | LoRA lr | Epochs | Where |
|-------|--------|---------|--------|-------|
| Qwen3.5-4B | qwen | 2e-4 | 6 | local (RTX 5090 32 GB) |
| Qwen3.5-27B | qwen | 1e-4 | 12 | remote (RTX Pro 6000 96 GB) |
| gemma-4-E2B-it | gemma | 2e-4 | 6 | local (RTX 5090 32 GB) |
| gemma-4-31B-it | gemma | 1e-4 | 12 | remote (RTX Pro 6000 96 GB) |

Conditions (per model): **seen_unseen** (vision, cross-city generalization),
**per_city** (vision), **text_only** (per_city, non-vision baseline).

---

## 2. The knobs — and why

### LoRA: r = 16, α = 16, `target_modules = all-linear`, vision+language layers on
- **r/α = 16** is the ablation-validated rank from the original 4B work — it is
  the value the whole experimental series is anchored to, so all 12 runs use it
  for comparability. The unsloth notebooks default to r=32, but matching the
  prior ablations matters more than a marginal capacity bump.
- **α = r** (16/16) is the standard recommendation (α ≥ r). No rslora, no LoftQ.
- **all-linear + finetune_vision_layers=True**: the task is genuinely visual
  (street-view detail drives 9 of 14 topics), so the vision tower must adapt,
  not just the language head. `lora_dropout=0`, `bias="none"`.

### Learning rate: 2e-4 small (4B/E2B), 1e-4 big (27B/31B)
- **Small models → 2e-4**: the validated value from the 4B runs; small models
  tolerate (and need) the higher LR to converge in 6 epochs.
- **Big models → 1e-4**: halved for stability. At a fixed LoRA rank, larger
  models have larger effective update magnitudes per step; 2e-4 risks loss
  spikes on a 27B/31B. 1e-4 is the "not too conservative, not aggressive"
  middle — fast enough to move, safe enough not to diverge.
- Cosine schedule, 100 warmup steps, weight_decay 0.01, max_grad_norm 1.0.

### Epochs: 6 small / 12 big (+ early stopping, patience 2)
- Big models are slower per epoch but data-hungrier at low LR, so they get 12
  epochs to fully fit; small models converge in 6. **Early stopping
  (patience=2 on eval accuracy)** stops any run early if it plateaus, so the
  epoch budget is a ceiling, not a mandate.

### Optimizer: adamw_8bit
- Halves optimizer-state VRAM vs adamw_torch. Essential headroom for bf16 27B/31B
  on the 96 GB card, and free on the small models. This is what the unsloth
  vision notebooks use.

### Precision: bf16 LoRA (load_in_16bit), NOT 4bit
- **The smoke test PROVED bf16 fits even for 27B/31B**, so no QLoRA quantization
  is needed. Measured chosen batch sizes (edge=512, 5-image worst case):
  - 4B → **bs 68** (~95 % of card)
  - 27B → **bs 20** (~97 %, peak 92.5 GB)
  - E2B → **bs 32** (~97 %)
  - 31B → **bs 10** × grad_accum 2 = 20 effective (~92 %, peak 87.9 GB)
- bf16 keeps full base-model precision (no quantization quality loss), which
  matters for a benchmark comparison across model sizes. 4bit was on the table
  (the unsloth 31B notebook uses it) but is unnecessary here.

### Image resolution: max edge = 512
- **Measured, not guessed.** Through the real Qwen processor, 768 and 1024 px
  produce **identical** token counts (~1985) — the vision encoder caps the patch
  grid, so >512 buys no extra resolution to the model. 512 px gives ~1409 tokens
  (−29 %), fits larger batches, and street-view road-surface/signage detail is
  still legible at 512. Prior runs wastefully used 1024.

### Batch size: probed per machine, ANY integer (not 2^x)
- No fixed batch size — `train.py`'s probe measures real peak VRAM on worst-case
  5-image batches (warmup + 3 scored fwd/bwd/step iters) and lands on the largest
  integer bs that fits. **Probe knobs (speed policy):**
  - Remote: `PROBE_HARD_CAP=128`, `PROBE_RESERVE_MB=2048`, `PROBE_FLAT_MARGIN=0.0`
    → fills the card leaving ~2 GB free at worst case.
  - Local: `PROBE_RESERVE_MB=1024`, `PROBE_FLAT_MARGIN=0.0` → ~1 GB free (32 GB
    card has less to give).
- grad_accum auto-computed so effective batch ≈ 20–32; or pinned (e.g. 31B uses
  bs 10 × accum 2 = 20).

### Save: LoRA adapter only (SAVE_MERGED=0)
- Each merged 27B/31B would be 52–62 GB; 12 of them would blow the disk. The
  eval path (`eval_base.py` with `EVAL_ADAPTER`) loads base+adapter directly, so
  the merged model is never needed. Adapter is ~200 MB.

### Seed: 3407 (everywhere)
- Single source of truth, propagated to random/numpy/torch/cuda. The seed the
  prior 4B work used.

---

## 3. Model-family handling (the correctness-critical part)

Qwen and Gemma use **different chat-turn markers**, so the
`train_on_responses_only` collator masking strings must match the model or label
masking silently breaks. Set by `MODEL_FAMILY`:

| Family | instruction part | response part | chat template |
|--------|------------------|---------------|---------------|
| qwen | `<\|im_start\|>user\n` | `<\|im_start\|>assistant\n` | model's own |
| gemma | `<\|turn>user\n` | `<\|turn>model\n` | model's own |

- **Gemma uses `<\|turn>` markers** (this build, NOT the classic
  `<start_of_turn>` from Gemma 2/3). Verified against the real processor.
- We do **NOT** call unsloth `get_chat_template("gemma-4")` — its bundled template
  has a `list + str` crash on the system turn. The model's **own** shipped
  `chat_template.jinja` handles both string and list content and renders the same
  `<\|turn>` markers.
- **System message content is a plain string** (not a list-of-parts). Gemma's
  template requires this; Qwen accepts it identically. This was the fix for the
  `TypeError: can only concatenate list (not str) to list` crash.
- `check_label_masking()` aborts the run (sys.exit 1) if the markers produce
  all-masked or nothing-masked labels — the safety net that caught nothing
  because the markers are correct.

---

## 4. Dataset input — what the model actually sees

The corrected dual-perspective loader (see `IMAGE_MAPPING.md`). The 9
`satellite_marked` topics (land_use, building_height, urban_density,
junction_type, green_space, amenity_richness, road_type, road_surface,
transit_density) now feed **marked-satellite + 4 street-view angles** (5 images),
each SV labeled by **direction (Fwd/Right/Bwd/Left)** — NOT A/B/C/D, because for
these topics A/B/C/D are the answer choices. This fixed a pipeline bug that had
withheld the street-view from 62.9 % of training records.

---

## 5. Per-run pipeline (each of the 12)

1. **Train** (`train.py`) → saves LoRA adapter.
2. **Full-val eval** (`eval_base.py` with `EVAL_ADAPTER`) → accuracy.json + verdict.md.
3. **Benchmark** (EzelinyumEvaluator), THREE conditions:
   - `sat+stv4` (full, the headline number)
   - `sat_only` (street-view withheld)
   - `stv_only` (satellite withheld)

   If full > sat_only AND full > stv_only on the 9 dual-perspective topics, the
   multi-perspective design is validated.

---

## 6. How to launch

**Remote (big models, lab-ws):**
```bash
cd ~/training
bash training/launch_multimodel.sh --only qwen27b,gemma_31b
```

**Local (small models, RTX 5090):**
```bash
conda activate unsloth
SMOKE=1 bash training/launch_local.sh --only qwen4b   # optional quick check
bash training/launch_local.sh                          # 4B + E2B × 3 conditions
```

**Pull results to local (run on workstation, whenever):**
```bash
bash training/pull_results.sh
```

---

## 7. Smoke-test validation (the receipts)

All 4 models passed a 5-step smoke on the real card with this exact recipe:

| Model | chosen bs | peak VRAM | label masking |
|-------|-----------|-----------|---------------|
| Qwen3.5-4B | 68 | 90.6 GB (95 %) | OK |
| Qwen3.5-27B | 20 | 92.5 GB (97 %) | OK |
| gemma-4-E2B-it | 32 | 92.8 GB (97 %) | OK |
| gemma-4-31B-it | 10 ×2 = 20 | 87.9 GB (92 %) | OK |

Two bugs the smoke caught and we fixed before launch:
1. **Probe budget double-counted resident weights** → falsely rejected bf16 27B
   at every bs. Fixed to budget against TOTAL VRAM.
2. **Gemma chat-template `list+str` crash** → fixed via string system content +
   the model's own template.

> Note: the training summary prints "% of 80 GB" — cosmetic; the profile's
> `vram_gb` field reads 80, the card is actually 96 GB. Peak VRAM values above
> are the true MB figures.
