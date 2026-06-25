# EOLLM — Training Runs Inventory

Generated from run artifacts on `lab-ws` (RTX PRO 6000, 96 GB) and local pulls.
All data traced from each run's own `summary.txt`, `epoch_*_verdict.md`,
`checkpoints/*/adapter_config.json`, and SLURM `.out` logs — **not** from the
launcher comments (which are stale on LoRA rank; see notes).

## Conventions (read before using the numbers)

- **Base model:** Qwen3.5-4B (local copy at `models/Qwen3.5-4B`). Never pushed
  to HF Hub → **no HF model id** for any run. Config explicitly forbids hub
  fallback; `REPORT_TO=none`.
- **Thinking:** model is thinking-**native**, but every training target is an
  **empty** `<think>\n\n</think>` block followed by the answer letter →
  **thinking state = OFF**, no thinking cutoff/budget applies.
- **What the scores are:** all accuracy figures are **VALIDATION ACCURACY**
  (greedy, multiple-choice exact-letter-match) on the split's held-out
  `validation.jsonl`. **Loss = TRAINING loss.** There is no separate
  "benchmark" eval — the validation split *is* the benchmark.
- **Early stopped?** True iff `epochs stopped at < max epochs` (patience=2 on
  val accuracy). Runs that ran the full schedule = No.
- **LoRA params:** r=16 → **38,756,352 (0.85% of 4.58 B)** — logged.
  r=32 → **~77.5 M (~1.69%)** — *derived* by 2× linear scaling, not logged.
- **Seed:** 3407. **LR schedule:** cosine, warmup 100. **α = r** for all runs.

---

## Dataset axes

The full dataset is partitioned two different ways. A "sample" belongs to the
**per_city** family.

| Split family | Path | Train | Val | What the val set is | Measures |
|---|---|---|---|---|---|
| **per_city** | `EODATA_compressed_final/splits_per_city` | 26,931 | **6,984** | All **40** cities (same cities as train, new samples). city_type = 5,657 seen + 1,327 unseen, mixed. | In-distribution recall |
| **seen_unseen** | `EODATA_compressed_final/splits_seen_unseen` | 26,963 | **8,831** | Only the **8 held-out** cities (Chicago, Seoul, Moscow, Sydney, Singapore, Cape Town, Istanbul, Rio) — **none in train**. city_type = all unseen. | True cross-city generalization |

Both val sets carry the **same 14 topics** in the same relative proportions
(big topics capped at 561/topic in per_city, 739/topic in seen_unseen).

### The 14 dataset tasks (topics)

| Topic | Tests | Random | Notes |
|---|---|---|---|
| amenity_richness | POI/amenity density | 25% | partial text-leak |
| building_height | building levels | 25% | partial |
| camera_direction | streetview facing direction | 25% | **vision-only** |
| green_space | greenery amount | 25% | **leaked (~100% even text-only)** |
| junction_type | road junction class | 25% | partial |
| land_use | zoning / land-use class | 25% | partial |
| mismatch_binary_easy | do sat & street match? (easy) | 50% | **vision-only** |
| mismatch_binary_hard | do they match? (hard) | 50% | **vision-only** |
| mismatch_mcq_easy | which image matches? (easy) | 25% | **vision-only** |
| mismatch_mcq_hard | which image matches? (hard) | 25% | **vision-only** |
| road_surface | paved/unpaved | 25% | **leaked (~94% even text-only)** |
| road_type | road classification | 25% | partial |
| transit_density | public-transit density | 25% | partial |
| urban_density | built-up density | 25% | partial |

---

## Where each ablation's data comes from (provenance)

Traced from each run's `.out` log (`SPLIT_STRATEGY`, `EXCLUDE_TOPICS`,
`Dataset:`, and the `train N → M` filtering line). **Topics are removed from
TRAINING ONLY; the val set is always the FULL 14-topic set** — so excluded
topics still get a score, which is **zero-shot transfer**, not trained
performance.

| Run | Dataset (sample) | Real/flagship? | Train filter | Train records (after filter) | Tasks EXCLUDED from training | Tasks TRAINED |
|---|---|---|---|---|---|---|
| **ablation_split2b_seen_unseen** | **seen_unseen** | ⭐ **THE REAL/FLAGSHIP ONE** (only cross-city run; "both axes") | 5 topics removed | 26,963 → **16,963** | camera_direction, mismatch_×4 | 9 (the urban/geo metadata topics) |
| ablation_split1_per_city | per_city | regular ablation (in-dist) | 7 topics removed | 26,931 → **13,500** | amenity_richness, building_height, junction_type, land_use, road_type, transit_density, urban_density | 7 (camera_direction, green_space, mismatch_×4, road_surface) |
| ablation_split2a_per_city | per_city | in-dist twin of 2b | 5 topics removed | 26,931 → **16,931** | camera_direction, mismatch_×4 | 9 |
| ablation_split3_per_city | per_city | regular ablation | 4 topics removed | 26,931 → **18,931** | mismatch_×4 | 10 |
| ablation_split5_per_city | per_city | regular ablation | 1 topic removed | 26,931 → **24,931** | camera_direction | 13 |
| ablation_split6_per_city | per_city | regular ablation | 2 topics removed | 26,931 → **22,931** | mismatch_binary_hard, mismatch_mcq_hard | 12 |

**Ablation hypotheses (from launcher):**
- split1 — *Urbanization Removed*: can geo/vision tasks stand alone?
- split2a — *Geo Removed* (in-dist): urban metadata only.
- **split2b — *Geo Removed, cross-city*: the flagship — tests both the
  topic-removal axis AND the unseen-city axis simultaneously.**
- split3 — *Mismatch Removed*: does camera_direction transfer to mismatch?
- split5 — *Camera Direction Removed*: does mismatch transfer to camera_direction?
- split6 — *Mismatch Easy Kept Only*: does easy→hard mismatch transfer?

> ⚠️ **The "regular" full runs (`20260422`, `20260423`) train on ALL 14 topics
> — they are NOT ablations.** `20260422` = per_city flagship (in-dist),
> `20260423` = seen_unseen full run (cross-city, all topics). For a clean
> apples-to-apples "real model" number, `20260422` (78.3%, in-dist) and
> `20260423` (72.5%, cross-city) are the headline regular runs.

---

## Master run table

All scores = **validation accuracy** (greedy MCQ exact-match). Loss = **training loss**.

| Training ID | Name | Type | Dataset | Trained tasks | Val N | Result type | Best val acc | Final val acc | Base acc | Train loss | Max ep | Stopped ep | Early stopped? | LoRA r | LoRA α | LoRA params | Wall clock | Peak VRAM | HF id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260422_074420_rtx_pro_6000_96gb | per-city vision (full) | Regular | per_city | all 14 | 6,984 | val acc | **78.3%** (ep6) | 78.3% | 18.0% | — | 10 | 6 | ✅ Yes | 32 | 32 | ~77.5M (~1.69%)* | 1745 min | — | — |
| 20260423_190939_rtx_pro_6000_96gb | seen/unseen vision (full, long) | Regular | seen_unseen | all 14 | 8,831 | val acc | **72.5%** (ep6) | 70.8% (ep8) | 19.0% | 0.073 | 10 | 8 | ✅ Yes | 32 | 32 | ~77.5M (~1.69%)* | 2551 min | 83.7 GB (102%) | — |
| textonly_baseline | text-only baseline | Regular | per_city | all 14 (no images) | 6,984 | val acc | **49.5%** (ep1) | 49.0% (ep3) | 35.0% | 0.141 | 5 | 3 | ✅ Yes | 16 | 16 | 38,756,352 (0.85%) | 130 min | 21.3 GB | — |
| ablation_split1_per_city | Urbanization Removed | Ablation | per_city | 7 (geo/vision) | 6,984 | val acc | **66.3%** (ep5) | 66.3% | 18.0% | 0.058 | 5 | 5 | ❌ No | 16 | 16 | 38,756,352 (0.85%) | 1014 min | 92.2 GB (113%) | — |
| ablation_split2a_per_city | Geo Removed (in-dist) | Ablation | per_city | 9 (urban) | 6,984 | val acc | **59.5%** (ep1) | 50.7% (ep3) | 18.0% | 0.122 | 5 | 3 | ✅ Yes | 16 | 16 | 38,756,352 (0.85%) | 290 min | 59.3 GB (72%) | — |
| ablation_split2b_seen_unseen | Geo Removed (cross-city) ⭐ flagship | Ablation | seen_unseen | 9 (urban) | 8,831 | val acc | **61.8%** (ep3) | 60.4% (ep5) | 19.0% | 0.095 | 5 | 5 | ❌ No | 16 | 16 | 38,756,352 (0.85%) | 543 min | 59.5 GB (73%) | — |
| ablation_split3_per_city | Mismatch Removed | Ablation | per_city | 10 | 6,984 | val acc | **63.2%** (ep5) | 63.2% | 18.0% | 0.103 | 5 | 5 | ❌ No | 16 | 16 | 38,756,352 (0.85%) | 1261 min | 90.8 GB (111%) | — |
| ablation_split5_per_city | Camera Direction Removed | Ablation | per_city | 13 | 6,984 | val acc | **76.5%** (ep5) | 76.5% | 18.0% | 0.073 | 5 | 5 | ❌ No | 16 | 16 | 38,756,352 (0.85%) | 1608 min | 91.3 GB (111%) | — |
| ablation_split6_per_city | Mismatch Easy Only | Ablation | per_city | 12 | 6,984 | val acc | **76.4%** (ep5) | 76.4% | 18.0% | 0.088 | 5 | 5 | ❌ No | 16 | 16 | 38,756,352 (0.85%) | 1495 min | 91.2 GB (111%) | — |

\* r=32 LoRA param count derived by 2× scaling of logged r=16 count (38.76M); not directly logged.

---

## Per-topic validation accuracy (best epoch of each run)

**Bold** = topic was **excluded from that run's training** → the number is
**zero-shot transfer**, not trained performance. (`green_space`/`road_surface`
are metadata-leaked, near-ceiling everywhere.)

| Topic | 22 full | 23 s/u | textonly | abl1 | abl2a | abl2b⭐ | abl3 | abl5 | abl6 |
|---|---|---|---|---|---|---|---|---|---|
| amenity_richness | 62.0 | 59.4 | 34.8 | **38.0** | **36.9** | 59.7 | 62.6 | 61.0 | 61.9 |
| building_height | 69.0 | 54.4 | 40.6 | **52.1** | 43.7 | 54.0 | 66.3 | 65.1 | 68.2 |
| camera_direction | 56.5 | 55.6 | 26.4 | 46.0 | **26.2** | **26.1** | 43.5 | **29.9** | 44.2 |
| green_space | 100 | 63.8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| junction_type | 77.0 | 71.4 | 58.0 | **60.7** | 52.0 | 69.6 | 76.6 | 78.3 | 75.9 |
| land_use | 75.9 | 69.1 | 72.0 | **50.8** | 73.6 | 76.2 | 79.0 | 77.9 | 80.7 |
| mismatch_binary_easy | 96.6 | 95.0 | 48.0 | 97.1 | **54.5** | 77.4 | **75.2** | 97.3 | 97.3 |
| mismatch_binary_hard | 90.2 | 91.6 | 52.2 | 92.3 | **54.7** | 68.1 | **65.2** | 92.2 | **82.7** |
| mismatch_mcq_easy | 99.1 | 98.5 | 25.7 | 98.0 | **26.6** | 37.3 | **31.4** | 98.4 | 97.5 |
| mismatch_mcq_hard | 90.6 | 90.5 | 21.7 | 88.2 | **23.4** | 31.1 | **29.4** | 90.2 | **81.6** |
| road_surface | 94.2 | 31.5 | 94.2 | 94.0 | 94.2 | 98.4 | 93.7 | 94.5 | 94.2 |
| road_type | 71.3 | 70.9 | 66.1 | **55.3** | 67.6 | 72.1 | 72.7 | 74.0 | 73.6 |
| transit_density | 50.8 | 36.9 | 28.0 | **31.0** | 36.7 | 44.9 | 49.0 | 48.1 | 51.0 |
| urban_density | 73.8 | 70.6 | 55.1 | **41.7** | 54.2 | 72.9 | 73.3 | 76.3 | 74.3 |

Column key: `22 full`=20260422 per_city all-14 · `23 s/u`=20260423 seen_unseen
all-14 · `textonly`=text-only baseline · `abl*`=ablations.

---

## Caveats / data-integrity notes

1. **LoRA rank conflict:** `launch_ablations.sh` comment says r=32/α=32, but
   every ablation `adapter_config.json` is **r=16/α=16**. The adapter config is
   ground truth → ablations are **r=16**. The two regular full runs
   (`20260422`, `20260423`) are genuinely **r=32**.
2. **`20260422` train loss** not written (its `summary.txt` is empty); 78.3%
   taken from its authoritative `epoch_6_verdict.md`.
3. **Cross-set comparability:** per_city (6,984 val) and seen_unseen (8,831 val)
   are different eval sets → only compare absolute % *within* the same dataset.
   per_city↔per_city and seen_unseen↔seen_unseen are valid; across is not.
4. **Ablation "overall acc" is muddy** — it averages trained + zero-shot
   (excluded) topics over the full val set. Read the per-topic table for the
   real ablation signal, not the headline.
5. **No separate benchmark exists** — the validation split is the benchmark. If
   a distinct held-out benchmark file gets run against the merged models, add a
   real "benchmark" column then.
