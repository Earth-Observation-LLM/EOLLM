# 03 — Main-paper figures

**Status:** IN PROGRESS
**Priority:** HIGH — both critics flagged "zero figures" as a problem.

## Rules
- Clean matplotlib/vector from VERIFIED data, OR real dataset imagery. NO AI imagery.
- Show CIs/error bars where we have them. Show n per task. Never claim synergy = 0.
- green_space excluded. Old `figures/*.pdf` are UNSAFE (wrong full-dataset counts) — regenerate.

## Figures (ranked; from the two figure-agent plans)
1. **Teaser (Fig 1)** — one real held-out location (e.g. sydney_0002): marked sat tile +
   4 real SV + one real MCQ + a 4-bar mini plot blind/sat/sv/full for that task.
   Conveys setup + finding on real data. Images from
   `dataset_content/EODATA_compressed_final/benchmark/images/{sat,sv}/`.
2. **Synergy + positive-control caterpillar (Fig 2)** — forest plot, per-task synergy
   with bootstrap 95% CIs + line at 0 + ±3 TOST band; attribute tasks cluster near 0,
   cross-view controls at +17..+70 on same axis. Kills "insensitive test".
3. **full vs per-item oracle dumbbell (Fig 3)** — per task, full (open) vs oracle (filled),
   connector = gap; pooled gap −9.3 [−10.4,−8.2]. Label oracle as upper bound.
   (Optional) vision×fusion scatter as alternative to Fig 2.

## Data sources
- Per-task synergy CIs: `scratchpad/per_task_ci.py` (agent wrote) over
  `benchmark_suite/results/qwen9b_per_city__benchmark/*_predictions.jsonl`.
- Oracle/flip: `benchmark_suite/results/analyze_synergy.py`.
- Controls: paper Table (control) / REPORT_urban_fusion.md §6.
- Script target: one `make_figures.py` (matplotlib + PIL).

## Recommendation
Main paper: Fig 1 (teaser) + Fig 2 (caterpillar). Fig 3 + scatter + strategy +
zero-shot + attention → supplementary if space (14-page limit, currently at limit).
