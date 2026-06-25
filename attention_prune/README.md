# attention_prune — is the full 360° street view redundant?

A separate research branch from `benchmark_suite/`. Different question, different
requirements.

## The claim

For the 8 urban-attribute tasks (green_space deferred), an earlier ablation
already shows the **satellite (cross) view is not needed** — sv_only is enough.
This branch pushes further:

> Even the **full four-angle 360° street view is redundant**. For each question,
> the **right 1–3 street-view images — chosen by where the model looks — are
> enough**. Sometimes a single image suffices.

How we show it, in two phases:

1. **Solve + attend (Phase 1, `solve.py`).** Run a strong, *non-quantized* model
   (Qwen3.6-27B bf16 on lab-ws). For each question, get the **correct** answer,
   and on that correct answer capture **per-image attention** — how much the model
   looked at each of the 4 SV angles when it committed to its letter.

2. **Prune (Phase 2, `prune.py`, written later).** Keep only the top-k attended
   images and re-ask the question in a **single greedy pass** — a pure benchmark,
   no retries. Measure what fraction of solved questions *survive* pruning to
   k = 1, 2, 3 images.

## Honest framing (important)

The headline number is a **per-question ceiling**, reported as a **survival
percentage**, NOT a raw benchmark accuracy:

> "Of the questions the model can solve, X% remain solved when we keep only the
> attended 1–3 SV images (k=1: …, k=2: …, k=3: …)."

It is *not* "the model now scores X% with fewer images." The questions were
selected for being solvable, so quoting a raw accuracy on them would be circular.
The survival curve is the legitimate, publishable result: it measures redundancy
of the 360° view, conditioned on the question being answerable.

## Two controls Phase 2 MUST include (or the result doesn't hold up)

Two independent reviews flagged the same fatal gaps. Both fixes are nearly free
(extra greedy passes only) and are non-negotiable for the headline:

1. **Negative control — top-k vs bottom-k vs random-k vs k=0.** Survival of the
   *top*-attended subset alone is unfalsifiable: a high number is equally
   consistent with "any k images work." So at each k, evaluate the
   **bottom-k least-attended** and a **random-k** (fixed seed) subset on the same
   questions. The headline is the **gap** (top-k − random/bottom-k), with CIs. If
   the gap is ~0, attention isn't selecting anything and the honest claim shrinks
   to "the 360° view is redundant" (still real, weaker). The **k=0** arm doubles
   as a leakage detector: a question that "survives" with zero images must be
   dropped from the headline.

2. **Class-prior leakage guard (`blind_baseline.py`).** Several urban topics are
   answerable from the TEXT/option prior alone — a zero-pixel constant predictor
   scores (measured on this benchmark):

   | topic | blind acc | headline-safe? |
   |---|---|---|
   | road_surface | **0.957** | ❌ |
   | junction_type | 0.614 | ❌ |
   | road_type | 0.603 | ❌ |
   | urban_density | 0.546 | ❌ |
   | land_use | 0.523 | ❌ |
   | building_height | 0.474 | ❌ |
   | amenity_richness | 0.373 | ✅ |
   | transit_density | 0.297 | ✅ |

   On a leaky topic, attention over images is meaningless and pruning "survives"
   trivially. **Only `amenity_richness` and `transit_density` carry the headline**
   (blind < 0.45, threshold tunable). The other six are reported with their blind
   baseline shown beside the survival curve, never pooled into the headline.
   Run `python blind_baseline.py` to regenerate this table.

### Headline eligibility (recorded per question by `solve.py`)

`headline_eligible = True` only when the win was **neutral-prompt + greedy** with
a usable attention map. Steered wins are excluded because a steered attention map
is partly an artifact of the instruction ("look at the junction"), not the model's
endogenous attention; sampled wins are off-distribution from greedy. Both are
recorded and shown as clearly-labeled secondary curves.

### Stated limitations

- **Solver ≠ deploy model.** Attention/pruning are measured on raw 27B; the
  conclusion is model-specific. Cheap transfer test: run Phase-2 greedy passes on
  a smaller deployed model using the 27B-chosen top-k subsets.
- **First-correct freeze** biases the solved pool toward easy/steered solutions,
  so survival is a ceiling over an easy-skewed subpopulation (the neutral-greedy
  split exposes most of this).
- Report **Wilson/bootstrap CIs** on every survival rate and on the gap; footnote
  post-exclusion solved-N per topic·k cell (`building_height` n=190 is the floor).

## Why a non-quantized model in transformers + eager

Attention weights are obtainable **only** through `transformers` with
`attn_implementation="eager"`. vLLM (how AWQ models are served) fuses the
attention softmax into a kernel and exposes nothing — a hard wall, not a quality
issue. AWQ-in-eager is technically possible but slow and pointless. So we use the
**raw bf16 27B**, which the lab-ws RTX 6000 Pro (96 GB) holds comfortably
alongside the L² attention matrix. 27B is also a much stronger *solver*, so more
questions reach a correct answer and therefore enter the pruning experiment.

## The solve→attend contract (determinism)

The retry loop solves *fast* (no attention). Only the **winning** attempt pays
for the eager attention forward. But a temp>0 win is *sampled* and not
reproducible, so the attention must be tied to an actually-reproduced correct
answer:

| Outcome           | Meaning                                                        | Pruning |
|-------------------|---------------------------------------------------------------|---------|
| `solved_greedy`   | temp=0 win; attention forward is deterministic (gold case)    | ✅ used  |
| `solved_sampled`  | temp>0 win; re-sampled with saved seed, reproduced gold ≤3×   | ✅ used  |
| `solved_unstable` | correct once, but no seed reproduced gold for attention       | ❌ excl. |
| `unsolved`        | never produced the correct letter across the ladder           | ❌ excl. |

### Solve ladder (fast generation)

`temp=0.0` (greedy, tried first) → `0.3` → `0.7` → `1.0`, with per-task steering
prompt variants interleaved (see `prompts.py`). Freeze on the first correct
answer and record the exact winning settings (temp, seed, prompt variant). Per-task
prompts steer attention to the relevant feature but never reveal the answer —
documented as a methodological deviation, not cheating (same images, same
options).

## Layout

```
attention_prune/
  README.md          this file
  solve.py           Phase 1 collector (solve ladder + eager attention capture)
  prompts.py         per-task steering prompt bank
  config.yaml        model path, topics, ladder, attention knobs
  run_solve.slurm    lab-ws SLURM job
  submit.sh          sbatch launcher
  solved/<model>/    solved.jsonl, unsolved.jsonl, meta.json  (pulled from lab-ws)
  viewer/            static attention viewer (eyeball which signal to trust)
  prune.py           Phase 2 — written after the data is inspected
```

Image construction, ablation (`sv_only`), prompt scaffolding and answer parsing
are **reused** from `benchmark_suite/` so pixels and scoring are identical to the
main eval.
