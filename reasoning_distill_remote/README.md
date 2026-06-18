# reasoning_distill_remote — prompting-strategy ablation (lab-ws / SLURM)

Does *how we prompt* — not the model — make Qwen3.5-9B-AWQ actually fuse the
satellite + street-view perspectives on the 9 urban-attribute tasks? Our earlier
perspective ablation showed `full ≈ sat_only ≈ sv_only` (the second view adds
~nothing). Here we pit **3 prompting strategies** against each other on the same
model, same records, same 4 view-ablations.

## Strategies (`--strategy`)

| name | thinking | turns | gist |
|------|----------|-------|------|
| `current`   | OFF | 1 | ask directly with options → strict JSON. The baseline. |
| `think16k`  | ON  | 1 | `<think>` up to 16k tokens, force-close fallback → JSON. |
| `multistep` | OFF | 4 | **chat:** observe (per-view checklist, images here) → reason w/o options → self-check → commit (options shown, scored). |

Each runs across views **full / sat_only / sv_only / blind**. Scope: the 9
attribute tasks only (`image_mode = satellite_marked`); cross-view matching tasks
are excluded (single-view ablation is ill-posed there).

## Files

- `run_strategy_ablation.py` — the async engine (server-based, multi-turn,
  thinking toggled per request, full transcript logging, resumable).
- `probe_max_seqs.py` — binary-searches the highest `--max-num-seqs` the GPU
  sustains for a given context, fast (short-context in-process probe).
- `submit_strategies.slurm` — one SLURM job: probe → `vllm serve` (per-strategy
  context) → wait ready → run → tear down, for each strategy.
- `download_model.slurm` — one-time HF-hub pull of the AWQ model into the cache.
- `make_strategy_report.py` — cross-strategy `strategy_compare.md/.json`.
- `complementarity_gate.py` — formal PASS/FAIL synergy gate per strategy.
- `sync_to_labws.sh` / `pull_results.sh` — code push / results pull.
- `summary.py` — live + offline summary builder (copy of the local one).

## Per-strategy context (`--max-model-len`, set by the SLURM script)

`current` 16384 · `think16k` 24576 · `multistep` 24576.

## Run it

```bash
# local: push code
./sync_to_labws.sh

# on lab-ws (one-time): get the model
cd /home/ain480/evaluation/reasoning_distill_remote
sbatch download_model.slurm        # wait for it to finish

# smoke (5 records/topic, all strategies + views)
STRATEGIES="current think16k multistep" LIMIT=5 sbatch submit_strategies.slurm

# full run
sbatch submit_strategies.slurm

# local: pull results, build the report
./pull_results.sh
python make_strategy_report.py
for s in current think16k multistep; do
  python complementarity_gate.py results/$s/ablation_log.jsonl --attr-only --json results/$s/gate.json
done
```

Results land under `results/<strategy>/` (`ablation_log.jsonl`, `summary.md`,
`summary.json`, `run_meta.json`). Everything is greedy (temp 0) and resumable —
re-`sbatch` to continue an interrupted run.
