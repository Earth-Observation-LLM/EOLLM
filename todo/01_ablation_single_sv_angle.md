# 01 — Single street-view angle vs. four-angle grid ablation

**Status:** TODO (needs lab-ws GPU run)
**Priority:** HIGH — both adversarial reviewers flagged the 4-images-vs-1 asymmetry
as the single most legitimate threat to the "single view suffices" claim.

## Why
The `sv` condition supplies 4 street-view images; `sat` supplies 1. `sv` is usually
the stronger single view, so a reviewer can argue "sv wins because it's 4 images /
4× tokens, not because the viewpoint is better" — which would weaken the headline.
We need to show a SINGLE street-view angle ≈ the 4-angle grid (or quantify the gap).

Per-image attention already supports this (sat ~24.3% vs each SV ~18.9% per image,
i.e. comparable per image), but a direct accuracy ablation settles it.

## What to run
On the trained Qwen3.5-9B (per-city and seen_unseen merged ckpts), on the 8-task
benchmark (n=2932, drop green_space), add ablation modes:
- `sv1_fwd`  — only `streetview_along_fwd`
- `sv1_right`, `sv1_bwd`, `sv1_left` — each single angle (optional, for variance)
- compare against existing `sv_only` (4-angle grid) and `full`.

Expected/decision:
- If `sv1_*` ≈ `sv_only` (within ~2 pts): image COUNT is not the driver → claim holds, strengthens paper.
- If `sv1_*` ≪ `sv_only`: must rescope to "a second OVERHEAD image adds nothing to street view" and soften title claim.

## Where
- Harness: `benchmark_suite/` (the unified eval). Add the single-angle image_mode to the loader.
- Data: `dataset_content/EODATA_compressed_final/benchmark/` (SV angles: along_fwd/bwd, cross_left/right).
- Model ckpts: `runs/mm_qwen9b_r16_per_city/merged_ep4_hf` (+ seen_unseen).
- Mirror the existing `qwen9b_*__benchmark/*_predictions.jsonl` output format so
  `analyze_synergy.py` can ingest it.

## Paper hook
Currently framed as a limitation in `\cref{sec:limitations}`. When results land,
add a row/sentence to §6 Robustness (Single image vs four-view) and remove the
limitation caveat.
