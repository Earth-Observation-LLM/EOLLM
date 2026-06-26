# Phase-2 Pruning Analysis — Qwen3.6-27B, k=1

**Data source.** The *complete* single-seed run
`solved/qwen36_27b/prune_k1_singleseed_265/` — 1663 questions × 5 arms,
**k = 1 only**, random seed **3407**. The multi-seed / k=1,2,3 run in `prune/` is
only **2.1 % complete** (~36 q per cell) and is unusable for confidence intervals,
so it is **not** used here. Consequence: **no per-seed random spread and no
k = 2/3 survival curve** (both listed in Limitations).

All survival rates are **per-question ceilings** on the 1663 questions the model
solved with greedy decoding on all 4 images — reported as a **survival %**, never as
raw accuracy. Every rate and gap carries a **Wilson 95 % CI**; gap CIs use the
Newcombe hybrid-score method. Numbers reconcile exactly with `prune_progress.json`
survivor counts and with an independent pooled recount.

---

## 1. Verdict (the three claims)

> **Claim A — "The full 360° street view is redundant; 1 image is often enough."**
> **SUPPORTED.** The single most-attended image keeps the answer correct in
> **89.5 % [88.0, 90.9]** of solved questions (n = 1663). Even a *random* single
> image survives **82.1 % [80.2, 83.9]**, and the *least*-attended image survives
> **83.2 % [81.3, 84.9]**. One of four views generally suffices.

> **Claim B — "The 1 image the model *attends to* most is enough (attention picks
> the right one)."**
> **PARTIALLY SUPPORTED — and the weaker reading is the honest one.** The
> top-attended image beats a *random* one by **+7.4 pts [+5.0, +9.8]** overall and
> by **+14.4 pts [+9.3, +19.4]** on the image-dependent (zero-FAIL) subset — so
> *ranking by attention does help over picking blindly*. **But** the top-attended
> image is statistically indistinguishable from *just always taking the forward
> camera*: gap(top − fwd) = **+0.5 pts [−1.6, +2.6]**, CI includes 0 at every
> scope. Attention's #1 image **is** the forward camera **94.8 %** of the time. So
> the defensible claim is *"keep the forward image,"* **not** *"attention's
> ranking selects the right image."*

> **Claim C — "The single image that matters is the forward-facing one
> (`streetview_along_fwd`)."**
> **SUPPORTED.** `fwd` alone survives **89.1 % [87.5, 90.5]**, essentially tied
> with the attention-optimal `top` arm and well above random/bottom. This is the
> real, reproducible finding.

**One-line synthesis.** *The 360° view is redundant and a single street-view image
usually suffices; the image that matters is the forward camera. The model's
attention does beat random selection, but only because it almost always points at
the forward camera — there is no evidence (at this power) that attention's ranking
adds anything beyond "use the forward image."*

---

## 2. Honest headline — leakage-safe pool

Six of eight topics are answerable from text/options alone (blind ≥ 0.45) and are
**excluded** from the headline. Only **`amenity_richness`** (blind 0.373) and
**`transit_density`** (blind 0.297) are leakage-safe.

| scope | top | fwd | random | bottom | zero | gap(top−random) |
|---|---|---|---|---|---|---|
| **SAFE topics**, all q (n=288) | 84.0 % | 83.3 % | 77.8 % | 79.9 % | 66.3 % | **+6.2 [−0.2, +12.6]** |
| **SAFE topics**, zero-FAIL (n=97) | 67.0 % | 63.9 % | 54.6 % | 58.8 % | 0 % | **+12.4 [−1.3, +25.4]** |

**Honest read of the safe pool: the top−random gap is directionally positive but
NOT significant** — both CIs include 0. The strong, significant gap
(**+14.4 pts**, zero-FAIL) comes from the *full* pool, which is dominated by leaky
topics. So:

- **"360° redundant / single image enough"** — holds even on the clean safe pool
  (top survives 84 % all-q, 67 % zero-FAIL).
- **"Attention/forward beats random"** — significant on the full pool, **only
  suggestive** (not significant) on the clean 2-topic safe pool, where n is small
  (288 / 97). State this plainly; do not headline the gap as established on safe
  topics.

The honest curve over k cannot be drawn (k=1 only); this is the single biggest gap
in the present run.

---

## 3. Per-topic table (all 8 topics, k=1)

Survival % with Wilson 95 % CI. `LEAKY` = blind ≥ 0.45 (excluded from headline).

| topic | blind | zero | top | fwd | random | bottom | n |
|---|---|---|---|---|---|---|---|
| amenity_richness ✅ | 0.373 | 51.3 % | **77.8** [70.8,83.6] | 77.2 | 72.8 | 75.3 | 158 |
| transit_density ✅ | 0.297 | 84.6 % | **91.5** [85.5,95.2] | 90.8 | 83.8 | 85.4 | 130 |
| building_height | 0.474 | 51.5 % | **85.8** [78.9,90.7] | 85.8 | 76.9 | 76.1 | 134 |
| junction_type | 0.614 | 49.1 % | **94.7** [91.0,97.0] | 91.2 | 89.0 | 89.0 | 228 |
| land_use | 0.523 | 73.8 % | 86.3 | **87.1** | 79.8 | 78.5 | 233 |
| road_surface | 0.957 | 96.4 % | **97.5** [94.9,98.8] | 97.5 | 87.0 | 89.5 | 277 |
| road_type | 0.603 | 59.4 % | 89.1 | **89.4** | 82.2 | 81.9 | 320 |
| urban_density | 0.546 | 62.3 % | **87.4** [81.8,91.5] | 86.9 | 79.8 | 84.7 | 183 |

Note `road_surface`: zero-image survival **96.4 %** ≈ its blind prior **0.957** —
the topic is almost entirely text-solvable; its high `top` survival carries **no**
image signal. `transit_density` (a "safe" topic) still survives **84.6 %** with
**zero** images — leakage is everywhere, not only in flagged topics.

---

## 4. top vs fwd — does attention earn its keep? (Adversarial duty #2)

- Attention's **rank-1 image == forward camera in 1577 / 1663 = 94.8 %** of
  questions.
- It **disagrees** (picks a side/back camera) in only **86** questions. Of those,
  attention chose `cross_left` 44×, `cross_right` 24×, `along_bwd` 18×.
- **On that disagree subset**, the attention-chosen image does beat forward:
  top **82.6 %** vs fwd **73.3 %**, gap **+9.3 pts** — **but CI [−3.2, +21.4]
  includes 0** (n = 86 is too small). So attention *may* add value when it
  disagrees with forward, but **this run cannot establish it.**

**Conclusion:** the top vs fwd distinction collapses. Report the finding as *"keep
the forward image,"* and flag the 86-question disagree subset as the place a
larger run would test whether attention's ranking is more than a forward-camera
detector.

---

## 5. Other adversarial checks

**#1 Leakage cancellation.** The gap does **not** collapse on the image-dependent
subset — it *grows* (full pool +7.4 → zero-FAIL +14.4). So the top−random gap is
real image signal, not shared-leakage noise. (It *does* lose significance on the
*safe* zero-FAIL pool purely from small n, §2.)

**#3 Position confound.** `fwd` is kept partly because it sits at a fixed grid
slot; we cannot separate "forward content" from "first/fixed position" from this
data. `fwd` success is **not** uniform across topics (51 %→98 %), which is mild
evidence it tracks content, not pure position — but the confound stands. **Stated,
not resolved.**

**#4 bottom ≥ random.** Pooled bottom **83.2 %** vs random **82.1 %**, gap
**+1.0 pts [−1.6, +3.6]** — statistically a tie. Verified **not** a ranking bug:
the bottom arm kept the genuinely least-attended (rank-4) image in **1663/1663 =
100 %** of questions. So *"even the worst single view is usually enough"* — this
**supports** the redundancy claim.

**#5 CI discipline.** All rates/gaps carry Wilson/Newcombe 95 % CIs; smallest cell
n = 130 (transit_density) for per-topic, 10–20 for some zero-FAIL safe cells
(flagged). Gaps whose CI includes 0 are explicitly marked not-a-finding.

**#6 k-monotonicity.** **Not testable** — only k=1 exists in the complete run.

---

## 6. Limitations (explicit)

1. **Leakage dominates.** 6/8 topics text-solvable; zero-image survival is 67 %
   pooled. Only 2 topics carry the headline, and on those the top−random gap is not
   significant. Any "survival" number mixes image use with answer-prior guessing.
2. **k=1 only.** No k=2/3 curve; the "1–3 images" half of the claim and
   k-monotonicity are untested in the complete run.
3. **Single random seed (3407).** No per-seed spread; the random arm is one draw,
   not the brief's pooled 3-seed estimate.
4. **Position confound.** Forward-camera success cannot be separated from its fixed
   grid position.
5. **Solver = 27B ≠ deployed model.** Survival is a ceiling for *this* solver;
   a smaller deployed model may not inherit it.
6. **First-correct-freeze selection bias.** Pool = questions solved on 4 images;
   survival is conditional and not an accuracy.
7. **Single-token attention caveat.** Answers are one token, so `choice_*` ≡
   `avg_*`; attention "ranking" is one across-layer mean per image, a coarse
   signal — the only real variation is across layers, collapsed before ranking.

---

## 7. Reproducibility appendix

```bash
# pull results
./attention_prune/pull_results.sh

# authoritative analysis (single source of truth for this report + slides)
python attention_prune/analyze_prune_k1.py

# blind baseline (leakage guard)
python attention_prune/blind_baseline.py        # threshold 0.45

# independent pooled recount (sanity)
python3 - <<'PY'
import json,collections
c=collections.defaultdict(lambda:[0,0])
for l in open('attention_prune/solved/qwen36_27b/prune_k1_singleseed_265/prune_results.jsonl'):
    d=json.loads(l); a=d['arm'].split(':')[0]
    c[a][0]+=int(d['correct']); c[a][1]+=1
for a in ['top','fwd','random','bottom','zero']:
    k,n=c[a]; print(a,k,n,round(k/n,4))
PY
```

- **n per cell:** top/fwd/random/bottom/zero all n = 1663 (pooled); per-topic
  n = 130–320; zero-FAIL pooled n = 548 (safe 97).
- **Producing run:** `prune_k1_singleseed_265/prune_meta.json` — model
  `Qwen/Qwen3.6-27B` (raw bf16, eager), signal = across-layer mean, seed 3407,
  8315 passes, `git_sha: unknown`.
- **Analysis code git sha:** `b083f41` (HEAD at time of writing; brief cited
  `5b88aa9`).
- Pooled survival reconciles with `prune_progress.json`: top 1489, fwd 1481,
  random 1366, bottom 1383, zero 1115 (of 1663).
