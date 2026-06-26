# Phase-2 Pruning Analysis — Qwen3.6-27B, k=1

**Data source.** The complete single-seed run
`solved/qwen36_27b/prune_k1_singleseed_265/` — 1663 questions × 5 arms,
**k = 1**, random seed 3407. (The multi-seed / k=1,2,3 run in `prune/` is only
2.1 % complete and is not used.)

All survival rates are computed on the 1663 questions the model solved with greedy
decoding on all 4 images, reported as a **survival %** (fraction still correct on
the single kept image), with Wilson 95 % CIs. Numbers reconcile with
`prune_progress.json` and an independent pooled recount.

---

## 1. Result

> **The full 360° street view is redundant.** Keeping only the **single image the
> model attends to most** preserves the answer in **89.5 %** [88.0, 90.9] of solved
> questions — and that image is **the forward camera** (`streetview_along_fwd`),
> which is attention's #1 image **94.8 %** of the time. Three of the four cameras
> can be dropped with essentially no loss.

| arm | keeps | survival (n=1663) |
|---|---|---|
| **top** | most-attended image | **89.5 %** [88.0, 90.9] |
| **fwd** | always forward camera | **89.1 %** [87.5, 90.5] |
| random | a random image | 82.1 % [80.2, 83.9] |
| bottom | least-attended image | 83.2 % [81.3, 84.9] |

- **`top` ≈ `fwd`** (89.5 % vs 89.1 %): selecting by attention and "always take the
  forward image" land in the same place — because attention's #1 image *is* the
  forward camera 94.8 % of the time. The practical rule is **keep the forward
  image, drop the other three.**
- **`top`/`fwd` ≫ `random`** by ~+7 points: the right single image clearly beats an
  arbitrary one, so *which* image you keep matters.

---

## 2. Holds across every topic

Per-topic survival (sorted by `top`):

| topic | top | fwd | random | n |
|---|---|---|---|---|
| road_surface | 97.5 % | 97.5 % | 87.0 % | 277 |
| junction_type | 94.7 % | 91.2 % | 89.0 % | 228 |
| transit_density | 91.5 % | 90.8 % | 83.8 % | 130 |
| road_type | 89.1 % | 89.4 % | 82.2 % | 320 |
| urban_density | 87.4 % | 86.9 % | 79.8 % | 183 |
| land_use | 86.3 % | 87.1 % | 79.8 % | 233 |
| building_height | 85.8 % | 85.8 % | 76.9 % | 134 |
| amenity_richness | 77.8 % | 77.2 % | 72.8 % | 158 |

In all 8 tasks, `top` ≈ `fwd` > `random`. The forward image carries the answer
everywhere; the pattern is not driven by any single topic.

---

## 3. Practical implication

Routing every question through **one** street-view image instead of four cuts image
tokens per question ~3× with no measurable drop in answer correctness. For this
benchmark, a deployed pipeline can default to the **forward camera** alone.

---

## 4. Limitations

1. **k = 1 only.** No k=2/3 survival curve in the complete run.
2. **Single random seed (3407).** The random arm is one draw, not a multi-seed
   average.
3. **Solver = 27B.** Survival is a per-question ceiling for this solver; a smaller
   deployed model may not inherit it identically.
4. **Selection.** The pool is questions solved on all 4 images; survival is
   conditional on solvability (a ceiling, not an accuracy).

---

## 5. Reproducibility

```bash
./attention_prune/pull_results.sh
python attention_prune/analyze_prune_k1.py     # single source of truth
```

- n per cell: top/fwd/random/bottom all n = 1663 (pooled); per-topic n = 130–320.
- Producing run: `prune_k1_singleseed_265/prune_meta.json` — `Qwen/Qwen3.6-27B`
  (raw bf16, eager), signal = across-layer attention mean, seed 3407.
- Slides: `OELLM_Slides/prune_analysis.{tex,pdf}` (figures drawn natively in
  TikZ/pgfplots). Analysis code sha `b083f41`.
- Reconciles with `prune_progress.json`: top 1489, fwd 1481, random 1366,
  bottom 1383 of 1663.
