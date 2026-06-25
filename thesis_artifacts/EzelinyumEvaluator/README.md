# EzelinyumEvaluator

Thesis-grade evaluator for the EOLLM urban VQA benchmark. Fixes every flaw the
adversarial audit found in `EarthMLLMEval` and adds the missing pieces:

- **LoRA / fine-tuned-model loading** (Unsloth + PEFT). Evaluates our 7 trained
  ablations end-to-end, not just baselines.
- **Per-record option space** for F1/ROC-AUC. No more global `["A","B","C","D"]`
  assumption — binary topics get binary metrics.
- **Sum-not-max** over multi-spelling token probabilities (`"A"`, `" A"`, `"▁A"`).
- **Full seeding** (`torch`, `numpy`, `random`, CUDA, `cudnn.deterministic`).
- **Per-topic chance + majority baselines** auto-computed and reported.
- **95% Wilson CIs** on every accuracy.
- **Per-sample predictions** with full provenance — enables paired McNemar /
  bootstrap retroactively.
- **Hedging detection** in the answer parser. Refusals → `None`, counted wrong.
- **Missing-image accounting** — degraded inputs flagged, not silently dropped.
- **Identical preprocessing** across all evaluated models (or documented divergence).
- **Per-model SLURM jobs** — no shared-wallclock OOM cascade.
- **Git SHA, seed, model revision** logged in every output JSON.
- **Cross-model comparison report** at the end.

## Scope

Evaluates on the 5,240-question held-out benchmark at
`/home/ain480/evaluation/EarthMLLMEval/data/benchmark_with_answers.jsonl`.

### Models

**Baselines** (HF transformers, from `~/.cache/huggingface/hub/`):
- Qwen2.5-VL-7B-Instruct
- Qwen3.5-4B (our base, no fine-tuning)
- llava-onevision-qwen2-0.5b-ov-hf
- llava-onevision-qwen2-7b-ov-hf
- meta-llama/Llama-3.2-11B-Vision-Instruct
- mistralai/Pixtral-12B-2409
- google/gemma-4-E2B-it
- Qwen3.5-9B (skipped: text-only)

**Fine-tuned** (Unsloth FastVisionModel + PEFT, from `~/training/training/runs/`):
- SU-base (full 14 topics, seen_unseen split)
- Split-1 (no urban)
- Split-2a (no geo, PC)
- Split-2b (no geo, SU)
- Split-3 (no mismatch)
- Split-5 (no camera)
- Split-6 (no hard mismatch)
- PC-base — **unavailable** (LoRA weights not preserved during archive)

## Usage

```bash
# Submit all evaluations (one SLURM job per model)
bash scripts/submit_all.sh

# Submit a single model
sbatch scripts/run_one.sbatch <model_key>

# After all complete, build the cross-model comparison
python src/aggregate.py
```

## Output layout

```
outputs/
├── <model_key>/
│   ├── predictions.jsonl   # per-sample predictions with full provenance
│   ├── report.json         # per-topic, per-city, per-difficulty metrics + CIs
│   └── meta.json           # seed, git SHA, model revision, wall time, VRAM
└── comparison/
    ├── headline.csv        # one row per model with overall + key topics
    ├── per_topic.csv       # rows=models, cols=topics, values=acc [CI]
    └── report.md           # narrative cross-model report
```
