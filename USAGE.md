# Multi-Agent Analysis Brief — `attention_prune` Phase-2 Pruning Results

You are the lead of a small analysis team. Your job: take the Phase-2 "pruning"
results from a vision-language-model experiment and produce a rigorous, honest
verdict on a specific claim. This document is **fully self-contained** — everything
you need (experiment, data schemas, traps, deliverable) is below. You do not need
to read the source code, but paths are given if you want to.

## 1. The one-sentence claim under test

> For 8 urban-attribute visual multiple-choice tasks, the full 360° street view
> (4 camera angles) is **redundant**: for each question, the **1–3 street-view
> images the model itself attends to most** are enough to keep the answer correct
> — and often **a single image** suffices.

Secondary claim that emerged and must also be evaluated:

> The single image that matters is almost always the **forward-facing** one
> (`streetview_along_fwd`).

Your task is **not** to confirm these. It is to determine **whether the data
supports them, how strongly, and with what caveats** — and to actively try to break
them (§5).

## 2. How the experiment works

Dataset: urban VQA. Each question shows **4 street-view images** (angles
`streetview_along_fwd`, `streetview_along_bwd`, `streetview_cross_left`,
`streetview_cross_right` at one location) and asks a 4-option (A/B/C/D) MCQ about
the scene. Satellite view was already shown unneeded; this is street-view-only.

Model: **Qwen3.6-27B**, raw bf16, thinking OFF, greedy. Answers are a **single
token** (the bare letter).

**Phase 1 (done).** Model solved each question on all 4 images. For the ones it got
**right with greedy**, it recorded, on the correct answer, **how much attention the
answer token paid to each image** (per layer + across-layer mean). → `solved.jsonl`.

**Phase 2 (what you analyze).** For each Phase-1 `solved_greedy` question, images
were thrown away and the question re-asked **once, greedily**, to see if the answer
**survives**. Arms differ only in *which* images are kept:

| arm | keeps (at k images) | tests |
|---|---|---|
| **top** | the k **most-attended** images | the claim |
| **fwd** | always `streetview_along_fwd` (k=1 only) | "is it just always forward?" |
| **random** | k random images (seeds 3407, 101, 202) | would *any* k work? |
| **bottom** | the k **least-attended** images | does ranking matter? |
| **zero** | **no images** (k=0) | **leakage detector** |

`k ∈ {1,2,3}`. Survival = of questions solved on 4 images, fraction still correct
on the kept subset. It is a per-question **ceiling**, reported as a **survival %**,
NEVER as a raw accuracy (questions were selected for solvability → circular).

## 3. Data files & exact schemas

Under `attention_prune/solved/qwen36_27b/`. Pull with
`./attention_prune/pull_results.sh`.

### 3a. `prune/prune_results.jsonl` — **primary input** (one JSON/line)
```json
{"question_id":"amsterdam_0064_amenity_richness_0","topic":"amenity_richness",
 "city":"Amsterdam","arm":"top","k":1,
 "kept_roles":["streetview_along_fwd"],"gold":"D","prediction":"B","correct":false}
```
Traps:
- random arm appears as **`random:3407`, `random:101`, `random:202`**. Pool all
  three (sum correct / sum n) AND report per-seed spread. Treat as base arm `random`.
- For a given (topic,k), **n is equal across top/bottom/random** (same pool). `fwd`
  only at k=1. `zero` only at k=0.
- `prediction` may be `""` → wrong.

### 3b. `solved.jsonl` — Phase-1 attention (join by `question_id`)
```json
{"question_id":"...","topic":"...","gold":"D","status":"solved_greedy",
 "headline_eligible":true,"winning":{"temp":0.0,"variant":"neutral","greedy":true},
 "options":{"A":"...","B":"...","C":"...","D":"..."},
 "attention":{"images":[{"image":1,"role":"streetview_along_fwd",
    "avg_norm_pct":33.32,"avg_raw_pct":2.76,"choice_norm_pct":33.32,"choice_raw_pct":2.76}, ...],
  "per_layer":[{"layer":0,"avg_raw_pct":[2.44,0.93,0.74,0.50],"choice_raw_pct":[...]}, ...],
  "spans_match_images":true,"row_softmax_check":1.0,"n_layers_used":16}}
```
**Critical:** answers are single-token, so `choice_*` == `avg_*` exactly — NOT two
findings. Only real attention variation is **across layers**. Phase-2 ranked by the
across-layer **mean** (`avg_norm_pct`).

### 3c. `prune/prune_meta.json` — provenance (signal, ks, seeds, model, paths, git_sha).
### 3d. `prune_k1_singleseed_265/` — earlier k=1 single-seed run; cross-check only,
authoritative run is `prune/`.

## 4. The leakage problem — dominates the verdict

Several topics are answerable from **text/options alone**.

**(a) Blind class-prior baseline** (zero-pixel modal-option predictor):

| topic | blind | safe? |
|---|---|---|
| road_surface | **0.957** | ❌ |
| junction_type | 0.614 | ❌ |
| road_type | 0.603 | ❌ |
| urban_density | 0.546 | ❌ |
| land_use | 0.523 | ❌ |
| building_height | 0.474 | ❌ |
| amenity_richness | 0.373 | ✅ |
| transit_density | 0.297 | ✅ |

Threshold 0.45 → **only `amenity_richness` and `transit_density` carry the
headline**. The other six: show blind baseline beside survival, never pool into the
headline.

**(b) The `zero` (k=0) arm is the empirical leakage detector** — worse than blind
predicted. From the k=1 run, zero-image survival:
```
road_surface     96.4%      land_use        73.8%      urban_density  62.3%
transit_density  84.6%(!)   road_type       59.4%      amenity_rich.  51.3%
building_height  51.5%      junction_type   49.1%
```
Even "safe" transit_density survives **84.6%** with zero images. **A large fraction
of "survival" in every arm is NOT image use.** Any headline ignoring this is
dishonest.

Decide how to condition (evaluate and recommend among):
- Survival **only on questions that FAIL the zero arm** (genuinely need an image) —
  cleanest image-dependent subpopulation.
- Report the **gap (top − random)** — leakage inflates both ~equally so it partly
  cancels (verify this assumption).
- Both, reconciled.

## 5. Adversarial duties (try to break the claims)

1. **Leakage cancellation.** Is the (top − random) gap real signal or noise on
   shared leakage? Recompute on the zero-arm-FAIL subset. If it collapses there,
   headline is weak.
2. **top ≈ fwd collapse.** In k=1, top ≈ fwd (attention's top image is `along_fwd`
   ~94% of the time). If top ≈ fwd, honest claim is "keep the forward image," NOT
   "attention selects the right image." On questions where attention's top ≠ fwd,
   does top still beat fwd? That subset is where attention earns its keep.
3. **Position confound.** `along_fwd` may be kept partly for its fixed grid
   position, not content. Can't fully resolve from data — **state it**; check if
   `fwd` success is suspiciously uniform across topics.
4. **bottom ≥ random anomaly.** In k=1, bottom sometimes matched/beat random. Is it
   "even the worst image is usually enough" (supports redundancy) or a ranking bug?
   Join to `solved.jsonl` to confirm bottom really kept low-attention images.
5. **CI discipline.** Wilson 95% CI on every rate and gap. A gap whose CI includes
   0 is not a finding. Report n per cell (smallest pool ~120–130 q).
6. **k-monotonicity.** Survival should weakly rise with k. top-2 < top-1 for a topic
   → flag (prompt rebuild / image-ordering bug).

## 6. Deliverable — `PRUNE_ANALYSIS.md`

1. **Verdict (3–5 sentences).** "360° redundant"? "single attended image enough"?
   "specifically forward"? Each: supported / partial / not-supported, + key number
   & CI.
2. **Honest headline:** survival on safe pool (amenity + transit), leakage-
   conditioned (§4), as a curve over k=1,2,3, with top-vs-pooled-random gap + CIs +
   per-seed random spread.
3. **Per-topic table** (all 8): blind | zero-arm | top/fwd/random/bottom at each k,
   with CIs. Leaky topics tagged, excluded from headline.
4. **top vs fwd resolution** (duty #2): % where attention disagrees with "just
   forward," and whether attention wins there.
5. **Limitations:** leakage, position confound, solver=27B≠deployed model,
   first-correct-freeze selection bias, single-token attention caveat. Explicit.
6. **Reproducibility appendix:** commands, n per cell, git sha (analysis code at
   `5b88aa9`; producing-run sha in `prune_meta.json`).

Tone: a skeptical reviewer should finish unable to find an overclaim. If weak, say
so — "360° redundant" may hold even if "attention selects the right image" does not;
those are different, separately-reportable conclusions.

## 7. Suggested multi-agent decomposition

- **A — Loader/aggregator:** parse results, pool `random:*`, build (topic,arm,k)→
  (correct,n) + per-seed table. Clean JSON, no interpretation.
- **B — Leakage analyst:** §4; zero-arm per topic, zero-FAIL subpopulation,
  recompute survival/gaps on it, decide conditioning.
- **C — Attention-credit analyst:** join `solved.jsonl`; duties #2, #4.
- **D — Statistician/skeptic:** Wilson CIs, k-monotonicity, flag CI-includes-0 gaps,
  write limitations.
- **Lead — synthesis:** reconcile; on disagreement, recompute from
  `prune_results.jsonl` directly, never average opinions.

## 8. Reference helpers (run, but verify)
```bash
python attention_prune/prune.py --analyze     # pooled-random + per-seed spread + zero-arm
python attention_prune/prune_status.py --local attention_prune/solved/qwen36_27b/prune
```
Starting point, not final word. The built-in tool does **not** condition on the
zero-arm subpopulation — that's the most important thing you add. Source:
`prune.py` (producer+analyzer), `solve.py` (Phase 1), `blind_baseline.py`,
`README.md`.
