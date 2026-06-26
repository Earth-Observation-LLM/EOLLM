# Multi-Agent Analysis Brief — `attention_prune` Phase-2 Pruning Results

You are the lead of a small analysis team. Your job: take the Phase-2 "pruning"
results from a vision-language-model experiment and produce a rigorous, honest
verdict on a specific claim. This document is **fully self-contained** — everything
you need to understand the experiment, the data, the traps, and the deliverable is
below. You do not need to read the source code, but paths are given if you want to.

If you are orchestrating sub-agents, a suggested decomposition is in §7. If you are
a single analyst, just work top-to-bottom.

---

## 1. The one-sentence claim under test

> For 8 urban-attribute visual multiple-choice tasks, the full 360° street view
> (4 camera angles) is **redundant**: for each question, the **1–3 street-view
> images the model itself attends to most** are enough to keep the answer correct
> — and often **a single image** suffices.

A secondary claim that emerged and must also be evaluated:

> The single image that matters is almost always the **forward-facing** one
> (`streetview_along_fwd`).

Your task is **not** to confirm these. It is to determine **whether the data
supports them, how strongly, and with what caveats** — and to actively try to break
them (see §5).

---

## 2. How the experiment works (context you must internalize)

The dataset: urban VQA. Each question shows a model **4 street-view images** (four
camera angles at one location: `streetview_along_fwd`, `streetview_along_bwd`,
`streetview_cross_left`, `streetview_cross_right`) and asks a 4-option (A/B/C/D)
multiple-choice question about the scene (e.g. "how rich is this area in shops?").
The satellite view was already shown unneeded in an earlier ablation; this
experiment is **street-view-only**.

The model: **Qwen3.6-27B**, raw bf16, thinking OFF, greedy decoding. Answers are a
**single token** (the bare letter, e.g. `D`).

**Phase 1 (already done).** The model solved each question on all 4 images. For the
questions it got **right with greedy decoding**, it recorded, on the correct
answer, **how much attention the answer token paid to each of the 4 images** (per
layer and as an across-layer mean). This is `solved.jsonl`.

**Phase 2 (the data you analyze).** For each Phase-1 `solved_greedy` question, the
experiment **threw images away and re-asked once, greedily**, to see if the answer
**survives**. The only thing that varies between "arms" is *which* images are kept:

| arm | keeps (at k images) | what it tests |
|---|---|---|
| **top** | the k **most-attended** images | the claim: attention picks the right image(s) |
| **fwd** | always `streetview_along_fwd` (k=1 only) | "is it just always the forward image?" |
| **random** | k random images (3 seeds: 3407, 101, 202) | control: would *any* k images work? |
| **bottom** | the k **least-attended** images | control: does the ranking matter at all? |
| **zero** | **no images** (k=0) | **leakage detector**: how much is text-only solvable |

`k ∈ {1, 2, 3}`. (k=4 prunes nothing, not run.)

**Survival** = of the questions solved on 4 images, the fraction that **stay
correct** on the kept subset. It is a per-question *ceiling*, reported as a
**survival percentage**, NEVER as a raw benchmark accuracy — the questions were
*selected for being solvable*, so a raw accuracy on them would be circular.

---

## 3. The data: files & exact schemas

All under `attention_prune/solved/qwen36_27b/`. Pull with
`./attention_prune/pull_results.sh` if not already local.

### 3a. `prune/prune_results.jsonl` — **the primary input** (one JSON object per line)

```json
{"question_id": "amsterdam_0064_amenity_richness_0",
 "topic": "amenity_richness",
 "city": "Amsterdam",
 "arm": "top",            // top | fwd | random:3407 | random:101 | random:202 | bottom | zero
 "k": 1,                  // 1, 2, 3, or 0 (zero arm)
 "kept_roles": ["streetview_along_fwd"],
 "gold": "D",
 "prediction": "B",
 "correct": false}
```

Traps:
- The random arm appears as **`random:3407`, `random:101`, `random:202`**. Pool all
  three (sum correct / sum n) AND report the **per-seed spread** so a result isn't
  one lucky draw. Treat `random:*` as base arm `random`.
- For a given (topic, k), **n is equal across top/bottom/random** (same pool). `fwd`
  exists only at k=1; `zero` only at k=0.
- `prediction` may be `""` (no parseable letter) → counts wrong.

### 3b. `solved.jsonl` — Phase-1 attention (join by `question_id` for ranking detail)

```json
{"question_id":"...","topic":"...","gold":"D",
 "status":"solved_greedy",          // pool you care about; also: solved_sampled, unsolved
 "headline_eligible":true,           // neutral-prompt greedy win w/ usable attention
 "options":{"A":"...","B":"...","C":"...","D":"..."},
 "attention":{
   "images":[{"image":1,"role":"streetview_along_fwd","avg_norm_pct":33.32,"avg_raw_pct":2.76,
              "choice_norm_pct":33.32,"choice_raw_pct":2.76}, ...],
   "per_layer":[{"layer":0,"avg_raw_pct":[2.44,0.93,0.74,0.50],"choice_raw_pct":[...]}, ...],
   "spans_match_images":true,"row_softmax_check":~1.0,"n_layers_used":16}}
```

**Critical:** answers are single-token, so `choice_*` == `avg_*` (decision-moment
and answer-averaged are the same row). Do NOT report them as two findings. The only
real attention variation is **across layers**; Phase-2 ranking used the across-layer
**mean** (`avg_norm_pct`).

### 3c. `prune/prune_meta.json` — provenance (signal, ks, seeds, model, dataset paths).
### 3d. `prune_k1_singleseed_265/` — earlier k=1 single-seed run. Use only to
cross-check that k=1 reproduces; authoritative run is `prune/`.

---

## 4. The leakage problem — **read twice, it dominates the verdict**

Several topics are answerable from **text/options alone**, no image.

**(a) Blind class-prior baseline** (zero-pixel constant predictor = always the modal
correct option), measured on this benchmark:

| topic | blind acc | headline-safe? |
|---|---|---|
| road_surface | **0.957** | no |
| junction_type | 0.614 | no |
| road_type | 0.603 | no |
| urban_density | 0.546 | no |
| land_use | 0.523 | no |
| building_height | 0.474 | no |
| amenity_richness | 0.373 | yes |
| transit_density | 0.297 | yes |

Threshold 0.45. **Only `amenity_richness` and `transit_density` can carry the
headline.** The other six are shown with their blind baseline beside the survival
and **never pooled into the headline**.

**(b) The `zero` (k=0) arm is the empirical leakage detector** and revealed worse
than the blind baseline predicted. From the k=1 run, zero-image survival was:

```
road_surface     96.4%      land_use        73.8%      urban_density  62.3%
transit_density  84.6%(!)   road_type       59.4%      amenity_rich.  51.3%
building_height  51.5%      junction_type   49.1%
```

Even "safe" `transit_density` survives **84.6%** with **zero images** — the 27B
leans on text priors at answer time far more than the modal baseline suggested. **A
large fraction of "survival" in every arm is NOT image use.** Any headline ignoring
this is dishonest.

**You must decide how to condition on this.** Evaluate and recommend among:
- Survival **only on questions that FAIL the zero arm** (genuinely need an image) —
  cleanest image-dependent subpopulation.
- The **gap (top − random)** instead of top alone — leakage inflates both arms
  ~equally, so the gap partly cancels it (verify that assumption, don't assume it).
- Both, reconciled.

---

## 5. Adversarial duties (actively try to break the claims)

A finding counts only if it survives these. Assign each to a skeptic if you have
sub-agents:

1. **Leakage cancellation check.** Is the (top − random) gap real signal or noise on
   top of huge shared leakage? Recompute the gap on the zero-arm-FAIL subset. If it
   collapses there, the headline is weak.
2. **top ≈ fwd collapse.** In k=1, `top` ≈ `fwd` (attention's top image is
   `along_fwd` ~94% of the time). If top ≈ fwd, the honest claim is "keep the
   forward image," NOT "attention selects the right image." Quantify: on questions
   where attention's top ≠ fwd, does `top` still beat `fwd`? That subset is the only
   place attention earns its keep.
3. **Position confound.** `along_fwd` may be attended/kept partly for its fixed grid
   position, not its content. Not fully resolvable from the data — but **state it**
   and check whether `fwd` success is suspiciously uniform across topics regardless
   of whether forward content is plausibly informative.
4. **bottom ≥ random anomaly.** In k=1, `bottom` sometimes matched/beat `random`.
   Does this mean "even the worst image is usually enough" (supports redundancy) or
   a ranking bug? Join to `solved.jsonl` to confirm `bottom` really kept
   low-attention images.
5. **CI discipline.** Every survival rate and gap needs a **Wilson 95% CI**. A gap
   whose CI includes 0 is not a finding. Report n for every cell; smallest topic
   pool is ~120–130 questions.
6. **k-monotonicity.** Survival should rise (weakly) with k. If top-2 < top-1 for a
   topic, something is wrong (prompt rebuild, image ordering) — flag it.

---

## 6. Deliverable

Produce a single markdown report, `PRUNE_ANALYSIS.md`, with:

1. **Verdict (3–5 sentences).** Does the data support "360° view is redundant"?
   "Single attended image is enough"? "It's specifically the forward image"? State
   each as supported / partial / not-supported, with the key number + CI.
2. **The honest headline number.** Survival on the **safe pool**
   (amenity_richness + transit_density), conditioned for leakage (§4), as a curve
   over k=1,2,3, with **top vs pooled-random gap + CIs** and the **per-seed random
   spread**.
3. **Per-topic table** for all 8 topics: blind baseline | zero-arm survival |
   top/fwd/random/bottom survival at each k, with CIs. Leaky topics tagged and
   excluded from the headline.
4. **The `top` vs `fwd` resolution** (duty #2): % of questions where attention
   disagrees with "just forward," and whether attention wins there.
5. **Limitations & threats to validity** (leakage, position confound, solver = 27B ≠
   any deployed model, first-correct-freeze selection bias, single-token attention
   caveat). Explicit, not buried.
6. **Reproducibility appendix:** exact commands, n per cell, git sha from
   `prune_meta.json.git_sha` (analysis code is at sha `5b88aa9` as of writing).

**7. A FINDINGS SLIDE at the very end — REQUIRED.** One screen, no prose padding,
no bullshit. Just the data with a single one-liner explaining each table/number.
Format it as a fenced markdown block so it reads like a slide. Rules:
- Every table or stat gets **exactly one** plain-language one-liner under it (what it
  means + the takeaway), nothing more.
- No paragraphs, no hedging adjectives, no "it is worth noting." Numbers carry the
  weight.
- Must contain: (i) the headline survival curve k=1,2,3 with top/random/gap+CI on the
  safe pool; (ii) the zero-arm leakage row; (iii) top-vs-fwd one number; (iv) the
  one-line verdict per sub-claim (redundant? / one image enough? / forward image?).
- Shape it like this skeleton (fill with real numbers):

```
=========================  FINDINGS  =========================
HEADLINE (safe pool: amenity_richness + transit_density, leakage-conditioned)
 k   top%[CI]      random%[CI]    gap pp[CI]
 1   ..  ..        ..  ..         +..  ..
 2   ..             ..             +..
 3   ..             ..             +..
 -> one line: does keeping the top-k attended images beat random, and by how much.

LEAKAGE (zero-image survival)
 safe-pool zero%: ..   leaky topics: road_surface ..%, transit_density ..% ...
 -> one line: how much "survival" is text-only, not image use.

ATTENTION vs POSITION
 attention top != forward in ..% of Qs;  on those, top beats fwd by ..pp [CI]
 -> one line: does attention add anything beyond "keep the forward image."

VERDICT
 360 view redundant?      YES/PARTIAL/NO  (+key number)
 one attended image enough? YES/PARTIAL/NO
 it's the forward image?  YES/PARTIAL/NO
=============================================================
```

Tone for the whole report: a skeptical reviewer should finish unable to find an
overclaim. If the result is weak, say so plainly — "the 360° view is redundant" may
hold even if "attention selects the right image" does not; those are different,
separately-reportable conclusions.

---

## 7. Suggested multi-agent decomposition (optional)

- **Agent A — Loader/aggregator.** Parse `prune_results.jsonl`, pool `random:*`,
  build (topic, arm, k) → (correct, n) + per-seed random table. Emit clean
  intermediate JSON. No interpretation.
- **Agent B — Leakage analyst.** Owns §4: zero-arm per topic, the zero-FAIL
  subpopulation, recompute all survival/gaps on it, decide the conditioning.
- **Agent C — Attention-credit analyst.** Joins to `solved.jsonl`; owns duties #2
  and #4 (top-vs-fwd, bottom-vs-random, where attention disagrees with position).
- **Agent D — Statistician/skeptic.** Wilson CIs everywhere, k-monotonicity, flags
  any gap whose CI includes 0, writes limitations.
- **Lead — Synthesis.** Reconciles B/C/D into verdict, headline, and the findings
  slide; resolves disagreements by **re-computing from `prune_results.jsonl`
  directly**, not by averaging opinions.

Cross-checks beat consensus: if two agents disagree on a number, recompute from the
raw file.

---

## 8. Reference helpers (run, but verify — don't trust blindly)

A built-in analysis already exists and prints much of §3/§4:

```bash
python attention_prune/prune.py --analyze        # pooled-random + per-seed spread + zero-arm
python attention_prune/prune_status.py --local attention_prune/solved/qwen36_27b/prune
```

Treat its output as a **starting point to verify and extend**. Your value-add is the
leakage conditioning (§4), the adversarial checks (§5), the honest verdict, and the
findings slide (§6). The built-in tool does NOT condition on the zero-arm
subpopulation — that is the most important thing you add.

Source for reference: `attention_prune/prune.py` (producer + analyzer),
`solve.py` (Phase 1), `blind_baseline.py` (leakage table), `README.md` (methodology).
