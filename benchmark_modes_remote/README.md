# benchmark_modes_remote — multi-model benchmark-modes ablation (lab-ws / SLURM)

The "Block A" experiment: run the **full ~5,240-question EOLLM benchmark** through
**several models** under the four image-ablation **modes**, all **thinking OFF**,
and produce maximal per-task tables. Tests the core claim — *does the second view
help, across architectures?* — by comparing `full` vs `sat_only` / `sv_only` /
`blind` for each model.

This reuses the proven batched eval engine (`run_modes_vllm.py`, identical to
`EarthMLLMEval/eval/`) plus the binary-search seq probe from the strategy
ablation, wired into a one-job-many-models SLURM loop.

## Modes (which images the model sees)

| mode | images | applies to |
|------|--------|------------|
| `full`     | everything the topic provides | ALL topics |
| `blind`    | none (text-only control)      | ALL topics |
| `sat_only` | satellite only                | 4× mismatch_* + camera_direction |
| `sv_only`  | street-view only              | 4× mismatch_* |

(Single-perspective `satellite_marked` topics only get `full` / `blind` — there's
no clean view to drop. `camera_direction` has no `sv_only` — its option images
are the answer.)

## Models (`models.sh`)

All AWQ-quantized, fit on the 96 GB Blackwell card, smallest→largest:

| short name | repo id |
|------------|---------|
| `qwen3.5-4b-awq`   | `cyankiwi/Qwen3.5-4B-AWQ-BF16-INT8` |
| `qwen3.5-9b-awq`   | `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8` |
| `gemma-4-12b-awq`  | `cyankiwi/gemma-4-12B-it-AWQ-INT4` |
| `qwen2.5-vl-7b-awq`| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |

Add/remove a line in `models.sh` to change the roster.

## Files

- `run_modes_vllm.py` — batched in-process vLLM eval, thinking OFF, writes
  per-(model,mode) predictions + `build_full_report` JSON. (`--data-root` added.)
- `probe_max_seqs.py` — binary-searches the highest `--max-num-seqs` the GPU
  sustains per model at the benchmark context (same in-process path the eval uses).
- `submit_modes.slurm` — one SLURM job: for each model, probe → batched eval →
  next. Resumable (skips models with an existing summary).
- `make_modes_report.py` — cross-model tables → `modes_report.md/.json` +
  `by_task.csv` + `overall.csv`.
- `download_models.slurm` — one-time HF-hub pull of the four models (+ optional
  purge of the old `unsloth--*` caches).
- `utils.py`, `metrics.py` — copies of the harness helpers (self-contained).
- `sync_to_labws.sh` / `pull_results.sh` — code push / results pull.

## Run it

```bash
# local: push code
./sync_to_labws.sh

# on lab-ws (one-time): get the models (and reclaim disk from unsloth)
cd /home/ain480/evaluation/benchmark_modes_remote
PURGE_UNSLOTH=1 sbatch download_models.slurm      # wait for it

# smoke (5 recs/topic, all models + modes) — validates the whole pipeline
LIMIT=5 sbatch submit_modes.slurm

# full run (all models, 5240 recs)
sbatch submit_modes.slurm

# local: pull results, build the maximal report
./pull_results.sh
python make_modes_report.py        # -> results/modes_report.md
```

Everything is greedy (temp 0), thinking OFF, and resumable — re-`sbatch` to
continue an interrupted sweep (done models are skipped via their `_summary.json`).

## Report (`make_modes_report.py`)

`results/modes_report.md` contains:
1. Overall accuracy — model × mode (+ bootstrap 90% CI on `full`)
2. Modality ablation — per model: full/sat_only/sv_only/blind + deltas
   (`full−blind` vision gain; `full−sat_only`/`full−sv_only` view-removal cost)
3. Per-task accuracy — model × topic matrix, one table per mode
4. Per-difficulty — model × mode × {easy,medium,hard}
5. Generalization — seen vs unseen cities, model × mode

Plus `by_task.csv` / `overall.csv` (tidy long form) and `modes_report.json`.
