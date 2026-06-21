# benchmark_suite

One standardized, YAML-configured pipeline for benchmarking vision-language
models on the EOLLM urban VQA benchmark — across **image-ablation modes**, with
both an accuracy view and an **interpretability view (ATTENTION WATCH)** of which
images drove each answer.

It replaces the scattered eval scripts (`reasoning_distill/run_modes_vllm.py`,
`training/eval_base.py`, `benchmark_modes_remote/`, `EzelinyumEvaluator/`) with a
single entry point that picks the right backend per model and produces uniform,
analysis-ready output.

## What it does

- **One config, many models.** Declare each model in `config.yaml` with its
  source (`hf_id`, optional `lora_path`) and whether to capture attention. The
  backend is resolved automatically.
- **Two backends, one interface.**
  - **transformers** (`backends/hf_backend.py`) — base **and LoRA** models
    (vLLM can't serve a PEFT adapter). Runs locally.
  - **vLLM** (`backends/vllm_backend.py`) — raw / AWQ models, batched, fast.
    Runs on the remote lab-ws GPU node.
- **ATTENTION WATCH** (`backends/attention_backend.py`) — per-image attention at
  the decision moment, the visual companion to the causal fusion analysis.
- **Ablation modes** `full / sat_only / sv_only / blind`, auto-pruned to each
  topic's applicable set.
- **A viewer** (`viewer/build_viewer.py`) — a self-contained HTML page showing,
  per record, the images the model saw, its choice vs. gold, and attention bars
  with a layer slider.
- **Fusion analysis** (`fusion.py`) — accuracy-side synergy / fusion-win /
  redundancy per topic.

## Backend resolution (all-rounder)

You pick the model; the suite picks the backend (override with `backend:`):

| condition | backend | why |
|---|---|---|
| `attention: true` | transformers (eager) | only path that can expose attention |
| `lora_path` set   | transformers | vLLM can't serve a LoRA adapter |
| raw weights       | vLLM if installed, else transformers | fastest available |

`backend: vllm` together with `attention: true` (or a `lora_path`) is a hard
error — vLLM physically cannot do either.

## Quick start (local — base + LoRA + attention on the RTX 5090)

```bash
conda activate unsloth

# smoke: 3 records/topic for every model in config.yaml
python run.py --limit-per-topic 3

# a single model, full benchmark
python run.py --models qwen35_4b_pc_lora

# build the interpretability viewer for that model
python viewer/build_viewer.py --results results/qwen35_4b_pc_lora --mode full
#   -> results/qwen35_4b_pc_lora/viewer_full.html   (open in any browser)

# accuracy-side fusion analysis (needs full/sat/sv/blind coverage)
python fusion.py --results results/qwen35_4b_pc_lora --by-topic
```

## Remote (vLLM / AWQ on lab-ws)

```bash
# local: push code
benchmark_suite/remote/sync_to_labws.sh

# on lab-ws
cd /home/ain480/evaluation/benchmark_suite
LIMIT=5 sbatch remote/submit.slurm     # smoke
sbatch remote/submit.slurm             # full, all vLLM models (config.remote.yaml)

# local: pull results
benchmark_suite/remote/pull_results.sh
```

Runs are resumable at the model level (a model with an existing
`results/<key>/summary.json` is skipped; `FORCE=1` to rerun).

## config.yaml

```yaml
dataset:
  path: dataset_content/.../benchmark/benchmark_with_answers.jsonl
  image_root: dataset_content/.../benchmark
  limit_per_topic: 0          # >0 = smoke cap per topic

defaults:
  modes: [full, sat_only, sv_only, blind]   # auto-pruned per topic
  enable_thinking: false
  image_max_edge: 768
  attention: {layers: all, max_pixels: 262144, store_per_layer: true}

models:
  - key: qwen35_4b_pc_lora
    hf_id: models/Qwen3.5-4B
    lora_path: training/runs/.../checkpoint-2022   # -> transformers
    attention: true                                # -> eager; can watch attention
  - key: qwen35_9b_awq
    hf_id: cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8
    backend: vllm                                  # remote only
```

## Output (per model, under `results/<key>/`)

| file | contents |
|---|---|
| `<mode>_predictions.jsonl` | one row/record: images shown, gold, prediction, `is_correct`, `prob_dict`, `raw_response`, and (if on) the `attention` block |
| `<mode>_report.json` | accuracy + F1/ROC (if scikit-learn) + by-topic/difficulty/city slices |
| `<mode>_attention.jsonl` | the attention blocks alone (per-image + per-layer shares) |
| `summary.json`, `meta.json` | per-mode overview + provenance (git SHA, backend, config) |
| `viewer_<mode>.html` | the interpretability viewer (after `build_viewer.py`) |
| `fusion.json` | synergy / fusion-win per topic (after `fusion.py`) |

### The `attention` block

```json
{
  "n_images": 5, "choice_letter": "D", "image_attention_total_avg_pct": 8.8,
  "spans_match_images": true, "row_softmax_check": 1.0,
  "images": [
    {"image": 1, "role": "satellite_marked",
     "avg_raw_pct": 1.7, "avg_norm_pct": 29.6,
     "choice_raw_pct": 1.9, "choice_norm_pct": 31.2},
    ...
  ],
  "per_layer": [ {"layer": 0, "avg_raw_pct": [...], "choice_raw_pct": [...]}, ... ]
}
```
- `raw_pct` = share of ALL attention (images vs. text); `norm_pct` = share among
  images only. `choice_*` = at the answer token; `avg_*` = over the whole answer.

## Soundness (results you can trust)

The suite re-implements image construction instead of importing the training
package, so a parity test guards against pixel drift, and the attention path
asserts alignment invariants. Run them:

```bash
python -m tests.test_image_parity          # vendored images == training/data.py (byte-identical)
python -m tests.test_core                  # modes / prompt / parsing / metrics
python -m tests.test_dispatch              # backend resolution + worklist
python -m tests.test_attention_invariants  # spans==images, row-softmax~1, norm sums to 100  (needs GPU)
```

Key guarantees:
- **Image parity** — `images.py` is byte-identical to `training/data.py:convert_record`.
- **enable_thinking=False + left-padding** everywhere (the known 0%-vs-75% trap).
- **Attention alignment** — answer query rows `prompt_len-1 .. L-2`; image spans
  detected via `<|vision_start|>/<|vision_end|>`; `#spans == #images` asserted;
  each analyzed row sums to ~1.
- Attention is loaded via **plain transformers + eager** (unsloth's compiled path
  returns empty attentions); LoRA still applies via PEFT on top.

## Files

```
config.yaml              control surface (models, modes, dataset, attention)
run.py                   entry point: resolve backend, build worklist, write outputs
images.py                canonical image builder (== training) + role classification
modes.py                 unified ablation-mode applicability + filtering
prompt.py                single-letter MCQ prompt with image labels
parsing.py               answer-letter + hedge/refuse parser
metrics.py               accuracy / F1 / ROC / CI + slices (scikit-learn optional)
fusion.py                accuracy-side synergy / fusion-win analysis
backends/
  base.py                WorkItem / ResultRow / Backend contract
  hf_backend.py          transformers (base + LoRA)
  attention_backend.py   ATTENTION WATCH (two-pass eager)
  vllm_backend.py        vLLM (raw / AWQ)
viewer/build_viewer.py   results -> self-contained HTML
remote/                  lab-ws SLURM: config.remote.yaml, sync/submit/pull
tests/                   parity + core + dispatch + attention invariants
```
