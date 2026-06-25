# Thesis artifacts — EOLLM task ablation study

Local mirror of every run, config, log, and slide used in the ablation study.
Pulled from `lab-ws:~/training/training/` on 2026-05-14. Excludes checkpoints,
LoRA adapter weights, and merged model weights (kept on remote).

Source companion: `ablation_report/report.tex` in this repo has the analysis.

## Layout

```
thesis_artifacts/
├── INDEX.md                       (this file)
├── runs/                          one folder per training run
│   ├── pc_base/                   PC-base, 14 topics, 6 ep, bs=86         → 78.3%
│   ├── su_base/                   SU-base, 14 topics, 8 ep, bs=80         → 70.8%
│   ├── split1_no_urban/           7 urban withheld, 5 ep, bs=96           → 66.3%
│   ├── split2a_no_geo_pc/         5 geo withheld (PC), 3 ep (collapsed)   → 50.7%
│   ├── split2b_no_geo_su/         5 geo withheld (SU), 5 ep, bs=96        → 60.4%
│   ├── split3_no_mismatch/        4 mismatch withheld, 5 ep, bs=96        → 63.2%
│   ├── split5_no_camera/          camera_direction only, 5 ep, bs=96      → 76.5%
│   └── split6_no_hard_mismatch/   2 hard mismatch withheld, 5 ep, bs=96   → 76.4%
├── configs/                       training pipeline source code
│   ├── config.py                  GPU profiles + env overrides + paths
│   ├── train.py                   training entrypoint
│   ├── data.py                    dataset loading + EXCLUDE_TOPICS hook
│   ├── callbacks.py               per-epoch eval + verdict writer
│   ├── evaluation.py              greedy-decoding evaluator
│   ├── eval_base.py               base-model baseline evaluator
│   ├── launch_ablations.sh        SLURM submission script (the 6 ablations)
│   ├── train.slurm                SLURM job template
│   ├── install.sh                 environment setup
│   ├── TRAINING.md                training pipeline README
│   └── dataset_report.md          dataset stats by topic + city
├── logs/                          stdout from each SLURM ablation job
│   └── ablation_split{1,2a,2b,3,5,6}_*.out
└── slides/                        intermediate slide deck
    ├── slides.pdf
    └── slides.tex
```

## What's in every `runs/<name>/` folder

| File | Contents |
|------|----------|
| `epoch_N_verdict.md` | Per-epoch eval: overall %, per-topic table, vs base/random/majority, full epoch history. **Primary source for every number in `report.tex`.** |
| `eval_loss.csv` | Eval cross-entropy per epoch |
| `training_loss.csv` | Train cross-entropy per step |
| `training_curves.png` | Loss curves figure |
| `tuning_notes.md` | Probe trace, chosen batch size, steps/epoch |
| `summary.txt` | One-line summary at end of run |
| `token_budget.md` | Token-length stats over the train set |
| `eval_samples.md` | Per-topic eval sample counts |

## Hyperparameters (shared across all 6 ablations)

- `NUM_EPOCHS=5`, `EARLY_STOPPING=1` (patience=2)
- `BATCH_SIZE=96`, `SKIP_PROBE=1`
- `WARMUP_STEPS=100`, cosine LR schedule, `lr=2.5e-4`
- LoRA `r=16`, `α=16`, all-linear modules, `finetune_vision_layers=True`
- `SEED=3407`
- Hardware: 1× RTX Pro 6000 96 GB
- Base: `Qwen3.5-4B-VL` (local copy under `models/`)
- Eval: greedy decoding, deterministic. PC eval = 6984 samples, SU eval = 8831.

Baselines (PC-base / SU-base) differ:
- `pc_base`: 6 ep, bs=86, LoRA r=16/α=16
- `su_base`: 8 ep early-stopped at ep4 best (72.4%), bs=80, LoRA r=16/α=16

## Headline numbers (for fast reference)

| Run             | Overall | Trained mean | Withheld mean | Notes |
|-----------------|---------|--------------|---------------|-------|
| pc_base         | 78.3    | 78.3         | —             | reference ceiling |
| su_base         | 70.8    | 70.8         | —             | cross-city honest headline |
| split1_no_urban | 66.3    | 91.4 (7 geo) | 47.1 (7 urban) | urban tasks ≈ +22 over random |
| split2a_no_geo_pc | 50.7  | 65.4         | 37.0          | training collapsed at ep1 |
| split2b_no_geo_su | 60.4  | 70.4         | 48.0          | mismatch_bin transfers, mcq doesn't |
| split3_no_mismatch | 63.2 | 76.0         | 50.4          | mcq stuck at chance for 5 ep |
| split5_no_camera | 76.5   | 80.1         | 29.9 (camera) | camera_direction has zero transfer |
| split6_no_hard_mismatch | 76.4 | 79.6   | 82.2 (hard)   | easy variant carries hard variant |

## Three transfer regimes (the headline result)

1. **Strong within-family**: easy → hard mismatch (Split-6), gap −7 to −9 pp.
2. **Partial cross-family**: urban tasks under geo supervision (Split-1), gap −16 to −25 pp.
3. **None**: camera_direction under any supervision; mismatch_mcq when family fully withheld — both ≈ random.

## Excluded from local mirror (kept on remote)

- `runs/*/checkpoints/` — Trainer checkpoints, ~10s of GB each
- `runs/*/lora/` — final LoRA adapter
- `runs/*/merged/` — merged full-precision weights
- Older / pre-ablation timestamped runs under `runs/2026042*_rtx_pro_6000_96gb/`
  (these are dev/debug runs; the 8 included runs are the study)
