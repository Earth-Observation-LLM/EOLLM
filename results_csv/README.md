# EOLLM consolidated results — CSV export

Generated from the three result trees (`benchmark_suite/`, `benchmark_modes_remote/`,
`reasoning_distill/`) plus the attention-prune run on lab-ws (pulled locally).
Five themed CSVs, because the result *shapes* are not mergeable into one table.

## What does the model SEE in each suite? (trust audit)

| Suite | image builder | filtering | trust |
|---|---|---|---|
| `benchmark_suite` | `images.build_images` (pixel-identical to `training/data.py`, parity-tested) | **role-based** | highest |
| `benchmark_modes_remote`, `reasoning_distill` | `run_modes_vllm.py` | **path-substring** (`/sat*`, `/sv|composite/`) | clean & deterministic |

Verified in both: `blind` → 0 images; `sat_only` → satellite role(s) only; `sv_only` →
street-view role(s) only; `full` → all. Output records log `image_roles`/`n_images`,
and the prompt text names the images shown — no silent mismatch. `streetview_mega`
feeds **2** images (sat + 1 mega grid), not 16.

## Trust flags you must respect

- **`OVERALL_all` vs `OVERALL_9urban` vs `OVERALL_8urban_noleak`** — `OVERALL_all` mixes
  the 9 satellite_marked urban-attr tasks with `mismatch_*`/`camera_direction`. The
  `full` jump on mismatch is **cross-view matching, not fusion**. For the fusion question
  use `OVERALL_9urban` (the satellite_marked set) or `OVERALL_8urban_noleak` (drops the
  leaked `green_space`). **These are the honest fusion denominators.**
- **`fusion_full_minus_best_single`** is filled only on the `full` row = `full − max(sat_only, sv_only)`.
  On the urban tasks it is ~**+2 pts** → the model does NOT meaningfully fuse views.
- **`LABEL_LEAK`** (green_space): 100% even blind on the fine-tuned model → not vision.
  **`blind_heavy`** (road_surface, urban_density): strong text shortcut.
- **`mode_NA`**: sat_only/sv_only on a single-perspective topic = degenerate (ignore).
- **sat_only / sv_only `OVERALL_all` are NOT cross-comparable** between models — different
  topics qualify for each, so the denominators differ. Compare per-topic, or use the
  urban overalls (same topic set everywhere).

## Files

1. **trained_qwen9b_modes.csv** — fine-tuned Qwen3.5-9B (per_city ckpt-2912, seen_unseen
   ckpt-2248), on benchmark + each model's own validation split, 4 modes, per-topic +
   3 overalls, with fusion. **Most trustworthy.** Headline: urban full ≈71–72%, fusion +2pts.
2. **raw_vlm_modes.csv** — zero-shot VLMs (raw Qwen3.5-9B local; Block-A AWQ 4-model
   sweep: Qwen3.5-4B/9B, Gemma-4-12B, Qwen2.5-VL-7B; think-off Qwen3.5-9B & Gemma-4-12B),
   benchmark, 4 modes, per-topic + overalls.
3. **external_rsvlm_satonly.csv** — GeoChat-7B, SkySenseGPT-7B, LHRS-Bot-Nova,
   EarthDial-4B (zero-shot RS-VLMs), sat_only, 9 urban topics. LHRS/EarthDial carry an
   OSM-provenance-overlap upper-bound caveat. Best external (EarthDial 41.3) < our
   trained sat (54.9).
4. **sample4geo_retrieval.csv** — Sample4Geo cross-view retrieval (ConvNeXt Siamese) on
   the 4 mismatch topics, cosine/threshold metric. **Not a VQA model; not comparable to
   accuracy tables.** Domain-transfer baseline.
5. **attention_prune_survival.csv** — Qwen3.6-27B (raw bf16 eager) on 8 urban sv_only
   tasks. **Survival %** (of solved_greedy questions, fraction still correct when pruned
   to top-k / fwd / random×3seeds / bottom-k / k=0 images). Two runs: `k123_multiseed`
   and `k1_singleseed`. Headline = the GAP top vs bottom/random/zero, on headline-safe
   topics only (amenity_richness, transit_density); leaky topics flagged. **Not a raw
   accuracy** — questions were pre-selected for being solvable.

## Headline reads

- Fine-tuning buys ~+17 pts over raw Qwen3.5-9B on urban tasks (full), +13.6 over best external.
- Cross-view fusion on urban attributes: ~+2 pts → essentially none; model leans on the better single view.
- Prune: k=0 floor 67% (leakage), top-k1 89.5% vs bottom-k1 83.2% vs random-k1 ~85% → modest top>random>bottom gap; the right 1 image largely suffices, but the signal is weak once leakage is accounted for.
