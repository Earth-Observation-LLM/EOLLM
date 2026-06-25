# EOLLM v2 — A Verifiable Cross-View Fusion VQA Benchmark

**Status:** draft blueprint. Section numbers map to the paper outline.
**One-line thesis (revised):** *EOLLM contributes a durable, human-grounded per-question
modality-necessity protocol for two-view benchmarks — each question certified (by human
modality-ablation, not one VLM's failures) that satellite-alone and street-view-alone both
land at chance while both-views reaches ceiling — and applies it to AUDIT the existing
cross-view benchmark (UrBench), which pairs both views and shows models fail at fusion but
never certifies fusion is required. The protocol + audit is the product; the certified EOLLM
set is the demonstration.*

> **Framing note (from primary-source UrBench read + adversarial review):** UrBench's tasks
> (Object Attribute Recognition, Object Matching) already overlap ours, so novelty is NOT the
> tasks — it is the **durable necessity certification + field audit**. Do not lead with "a new
> cross-view dataset"; lead with the gate. See §5.

---

## 0. Why the current dataset does not support its thesis (measured, not asserted)

All numbers below are computed by `reasoning_distill/complementarity_gate.py` on the real
`diffusiongemma_20260614_224218` ablation log (n = 3,135 benchmark questions that have all
four conditions). Reproduce with:

```
python reasoning_distill/complementarity_gate.py \
  reasoning_distill/results/diffusiongemma_20260614_224218/ablation_log.jsonl \
  --attr-only --chance 25
```

| metric (9 attribute tasks, overall) | value | what it means |
|---|---|---|
| acc_full | 55.0% | both perspectives |
| acc_sat_only | 54.2% | satellite alone |
| acc_sv_only | 53.7% | street-view alone |
| acc_blind | 37.9% | no images |
| **synergy** = full − best single | **+0.8** | the 2nd view adds almost nothing |
| **fusion-win** = P(full right ∧ both singles wrong) | **2.9%** | fusion *uniquely* solves this rarely |
| **fusion-hurt** = P(full wrong ∧ some single right) | **13.1%** | the 2nd view *distracts* 4.5× more than it uniquely helps |
| **redundancy** = P(sat and sv predict the SAME answer) | **67.3%** | the two views encode the same signal 2/3 of the time |
| **arbitrate** = P(full right \| sat/sv disagree) | **41.3%** | when the views conflict, fusion cannot resolve it (≈ chance) |

**Root cause (from code, `dataset/src/01_sample_locations.py` + `05_generate_questions.py`):**
every attribute label is `label = bin(one OSM scalar in a radius)` — median `building:levels`,
building count, amenity count, surface mode, stop count, etc. A single scalar → a single
label. Whichever view best correlates with that scalar recovers the answer; the other view
is redundant *by construction*. The "satellite and street views" phrasing lives only in the
question string, never in the label-generating function. Two tasks (green_space, road_surface)
are additionally answerable *blind* because the option strings are world-knowledge-rankable.

**The verdict to put in the paper:** the current tasks are not multimodal-dependent. The
extra view is net-negative (fusion-hurt 13.1% > fusion-win 2.9%). This is the motivation,
not a failure to hide.

---

## 1. The design principle (the condition every new task must satisfy)

A task is two-perspective-dependent iff the answer `Y` satisfies, informally,
**I(Y; satellite) low AND I(Y; street-view) low, but I(Y; both) high** — i.e. the answer
is *synergistic* (information present only in the joint, not either marginal), in the sense
of Partial Information Decomposition (Williams & Beer 2010; Liang et al., NeurIPS 2023).

The operational rule we design toward:

> **The label must depend on a RELATION between a referent that is localizable only from
> the satellite (where / which object / plan-view extent) and an identity/property that is
> legible only from the street-view (what it is / material / function / count of storeys).
> Delete either view and the answer becomes undetermined.**

This is grounded in a measured asymmetry (see §5 references):
- **Satellite owns:** building footprint & roof, block-interior land use, road-network
  topology, parking-lot extent, tree-canopy extent — all *occluded from any ground camera*.
- **Street-view owns:** facade material, ground-floor / storefront *function*, signage/text,
  exact storey count, street-level greenery — all *occluded by the roof from above*.
- **Fusion measurably closes the gap** exactly on *function/use* tasks: Workman et al.
  (ICCV 2017) building-function 27% (ground) / 35% (overhead) → **45% fused (+10.2)**;
  Srivastava et al. (RSE 2019) land-use +11 over street-only; Suel et al. (RSE 2021)
  income +20%.

---

## 2. The acceptance gate (validate BEFORE generating at scale)

The gate has **two layers**: a behavioral ablation (cheap, model-dependent) and a
model-agnostic information-theoretic certificate (the one that survives review).

### Layer 1 — behavioral ablation (screening). ~120–150 Qs, ≥2 architecturally different VLMs:

| check | threshold | rationale |
|---|---|---|
| blind_ok | acc_blind ≤ chance + 8pts | kills text/option leakage (necessary, not sufficient) |
| synergy_ok | full − max(sat, sv) ≥ **+8 pts** | the 2nd view adds real accuracy |
| win_ok | fusion-win ≥ **15%** | fusion *uniquely* solves a substantial slice |
| net_ok | fusion-win > fusion-hurt | the 2nd view helps more than it distracts |

Calibration: the field's "the image matters" range is a **9–16 pt** same-split full−blind gap
(VQA v1 ~9.7; VQA v2 ~16; GQA ~13). Measured fusion gains are +6.7…+11 (Workman, Srivastava).
So +8 synergy is a defensible floor — not the current +0.8.

### Layer 2 — HUMAN two-sided modality-ablation (the PRIMARY, durable certificate). ★★

**Why Layer 1 alone fails review (the adversarial reviewer's #1 reject reason):** "full > best
single" measured on *one VLM* is **circular and decays**. It measures "hard for diffusiongemma,"
not "needs both views" — and a stronger single-view model (better OCR/depth) later collapses it.
MMStar and UNO-Bench (the precedents we cite) specifically AVOID model-accuracy gates: MMStar
filters on *blind LLMs* (a statement about the text, durable), UNO-Bench on *human-judged
modality removal* (a statement about information, durable). Our Layer-1 gate must be **demoted to
a descriptive secondary statistic.**

**The durable certificate — ground necessity in human-judged information content:**

> Certify each question with a **two-sided human modality-ablation** (à la UNO-Bench's
> modality-removal gate + GQA/MMMU human ceilings): humans answer at **ceiling with BOTH views**
> but at **chance from EITHER view alone.** Because information absent from a single view cannot
> be extracted by *any* present-or-future model, this grounds "both views required" in
> information content, invariant to which VLM is tested. (UNO-Bench arXiv:2510.18915 — explicit
> "remove one modality → must be unanswerable" gate; human ceilings: GQA 89.3% / blind 42.1%,
> arXiv:1902.09506; MMMU expert 82.6–88.6%, arXiv:2311.16502.)

**Layer 3 — PID synergy (model-agnostic numeric corroboration, optional):** Decompose
I(Y; sat, sv) into Redundancy + Unique_sat + Unique_sv + **Synergy**; a fusion task has
**Synergy S > 0** (info only in the joint). Estimate S on frozen CLIP/SigLIP features vs the
label — no VLM dependence (Williams & Beer 2010, arXiv:1004.2515; estimator Liang et al.
NeurIPS 2023, arXiv:2302.12247 — *verify this ID before submission*).

**Durability argument, citably:** model-specific filtering couples benchmark validity to one
model's idiosyncratic weaknesses → erodes as models improve (a Goodhart/shortcut failure mode:
Geirhos et al. "Shortcut Learning," Nature MI 2020, arXiv:2004.07780; construct-validity decay
arXiv:2511.04703). Human/information grounding is invariant.

**The one-line claim for the paper:** *"Each question is certified two-view-necessary by a human
modality-ablation — annotators reach ceiling with both views but chance from either alone — so
the requirement holds for any model, present or future; we corroborate with positive cross-view
PID synergy and report current VLMs' single-view collapse as a secondary, descriptive finding."*

Harness: `reasoning_distill/complementarity_gate.py` (Layer 1 behavioral screen, already
written; verified to correctly FAIL all 9 current tasks). Layers 2 (human protocol) & 3 (PID
estimator): to be built.

---

## 3. The new task families (ranked; build top-down)

> Backbone asset: **~7,915 locations on disk**, each with 1 satellite (red dot) + 4
> bearing-labeled street-view frames (`along_fwd/bwd`, `cross_left/right`). Road bearing is
> stored on **100% of records**. New labels are derived from `lat/lon` + bearing + OSM
> geometry (OSM must be re-fetched or restored from cache — see §6 risk).

### Family A — Cross-view referent identification  ★ build first
*Satellite localizes the referent; street-view names it.* This is the Workman building-
function result, recast as VQA, and it is the cleanest synergy signal.

- **A1 — Marked-building function.** Satellite has ONE building footprint outlined/arrowed.
  Q: "What is the highlighted building primarily used for?" → {residential, retail/shop,
  food/drink, school/civic, industrial, place of worship}. *Sat alone:* sees the footprint
  but per-building function from overhead is poor. *SV alone:* sees facades/signage but
  doesn't know WHICH building is meant. **Label:** the OSM `building=` / `amenity=` /
  `shop=` tag of the polygon the arrow points to; the arrow direction is computed from the
  building centroid bearing relative to the camera so the correct building is visible in a
  specific SV frame.
- **A2 — Which street-view frame shows the marked spot.** Satellite arrow points from the
  camera dot toward an OSM feature. Q: "Which photo (forward/back/left/right) shows what the
  arrow points at?" → {fwd, bwd, left, right}. **Label:** pure trig — feature bearing vs the
  four canonical frame bearings (fwd = road bearing, cross_left = bearing−90°, etc.).
  *Neither view alone:* sat has no frame mapping; SV has no global frame. The road bearing
  is the bridge that deletion destroys.

### Family B — Metric-grounded attribute  ★ build second
*Satellite supplies a measurement; street-view supplies the identity that makes it answerable.*

- **B1 — Through-lane count of the camera's road.** SV confirms which road / its markings;
  satellite supplies the width/lane plan-view. Q: "How many through-lanes does the road the
  camera is on carry?" → {1, 2, 3–4, 5+}. **Label:** OSM `lanes=` on the snapped way (the
  point IS the snapped node), cross-checked against way width. *SV alone:* a wide arterial
  and a narrow local street can look identical at ground level. *Sat alone:* can't tell which
  of several nearby roads the camera stands on.
- **B2 — Footprint depth of the building ahead.** SV sees a facade; satellite sees how far it
  extends into the block. Q: "Does the building directly ahead extend deep into the block, or
  is it a thin street-front?" → {thin frontage / deep block / freestanding / corner}.
  **Label:** building polygon depth (perpendicular to the street) from OSM geometry along the
  forward bearing.

### Family C — Occlusion-complementary presence  ★ build third (or as a hard subset)
*The answer is about something only ONE view can see, anchored to the OTHER view's frame so
both are required.*

- **C1 — Behind-the-facade block interior.** Q: "Behind the row of buildings on your RIGHT,
  is the block interior open (courtyard/lot/garden) or fully built?" → {open / built / mixed}.
  *SV alone:* cannot see behind the street wall. *Sat alone:* "your right" is undefined
  without the SV frame + bearing. **Label:** land use of the polygon interior on the right-
  hand side of the road axis (bearing+90°), from OSM.

**Existing 9 attribute tasks:** retire to a clearly-labeled **single-view perception baseline**
subset (do NOT present as fusion). They become a useful *contrast* — "single-view tasks
saturate at +0.8 synergy; our fusion tasks reach +X" — which strengthens the paper.

---

## 4. Paper-grade validation plan (beyond the gate)

1. **Two-model replication.** Run the 4-condition ablation on ≥2 architecturally different
   VLMs (e.g. diffusiongemma + an autoregressive Qwen-VL) so the synergy is a property of
   the *data*, not one model. (NOTE: current Qwen-27B log ran sat/sv on a disjoint question
   subset, so existing logs cannot cross-confirm — a fresh aligned run is required.)
2. **Human ceiling on a 100-question sample:** humans-with-both-views accuracy (target ≫
   model full) and a humans-blind / options-only upper bound (target ≈ chance) to bound
   leakage.
3. **Randomize satellite orientation** (don't always put north up; rotate the crop) so the
   bearing→frame mapping must be *read*, not memorized — otherwise the model learns a fixed
   offset and ignores the satellite.
4. **Balanced answer distribution** per task (keep the existing seeded-shuffle + ≤40% letter
   check) and **balanced label priors** so blind ≈ chance is structural.
5. **Report the full 4×N ablation table + the gate metrics** as a first-class results table.
6. **★ AUDIT UrBench with our gate (the strongest single move — adversarial reviewer's #1
   recommendation, and a PRECEDENTED contribution type).** MMStar did exactly this audit on
   other benchmarks and found large degenerate fractions — **ScienceQA 57.2%** and **AI2D
   46.2%** of samples "lack visual necessity," **>20% of MMMU** answerable blind (Chen et al.,
   arXiv:2403.20330). That is the template. UrBench data is public
   (github.com/opendatalab/UrBench). Run our per-question necessity gate on UrBench's own
   "cross-view" questions and report what % are actually single-view-solvable — their
   36.2/54.6/42.3 numbers are **cross-task averages** (confirmed by reading the paper), so
   single-view solvability has **never been measured per-question** on UrBench. Then re-rank
   their 21 models on the *certified-necessary* subset and show whether the leaderboard
   changes. This reframes the paper from "we built a competing dataset" (incremental vs
   UrBench, whose tasks already overlap ours) to "we provide a durable necessity-certification
   protocol AND audit the field's existing cross-view benchmark with it" — exactly the
   contribution shape MMStar used to earn NeurIPS. The EOLLM set becomes the *demonstration*;
   the gate + audit is the *product*.

---

## 5. Positioning & must-cite prior art (from verified survey)

**⚠ The headline "satellite + street-view VQA" is already taken.** A reviewer WILL find
UrBench. The blueprint below is written to survive that, not to ignore it.

### The direct competitor — read first (primary source read in full, arXiv:2408.17267 v3)
- **UrBench** — AAAI 2025. 11.6K questions (1.1K val / 10.5K test), **14 tasks in 4 dimensions**
  (Geo-Localization, Scene Reasoning, Scene Understanding, Object Understanding), 11 cities.
  In-house data: 2,604 Google Street View + 4,239 Google Earth (L19) images, **1,965 paired**
  by coordinates; satellite side carries **OpenStreetMap** ground-object annotations.
  Cross-view pairing = **Grounding DINO boxes on street-view + ray-tracing to satellite +
  IoU>0.5 match + human check**. Questions: LMM-based + rule-based + human-based, all
  human-verified. 21 LMMs evaluated. GPT-4o **61.2%** overall, human **69.9%** (gap 17.4%).
- **CRUCIAL ASYMMETRY (our entire opening), now grounded in the paper:** their headline
  cross-view result — *"models struggle the most with cross-view tasks, averaging only 36.2%
  ... while averaging 54.6% and 42.3% in street-view and satellite-view"* — is an average
  **ACROSS DIFFERENT TASK SETS** (cross-view tasks vs single-view tasks are *different
  questions*), **NOT a per-question single-view ablation.** They never take one question,
  delete a view, and measure the drop. They **assert** necessity from task design (*"LMMs must
  utilize both perspectives"*, Intro) but never **certify** it. Their cross-view tasks score
  *"only 3% higher than random"* — proof models *fail*, not proof the task *requires* fusion.
  **That per-question necessity certification is exactly the white space EOLLM occupies.**
- **⚠ DANGER — direct task overlap.** UrBench's **Object Attribute Recognition (OAR)** already
  *"prompts LMMs to predict ground object attributes such as building floors and land use,"*
  and **Object Matching (OM)** is *"a cross-view task where LMMs predict the locations of
  objects in satellite and street views based on their cross-view correspondences."* Your
  Family A (mark building → name function/floors) and Family C overlap OAR/OM heavily. You
  MUST differentiate on the **necessity-certification** axis, not the task axis — the tasks
  themselves are largely already in UrBench. The novelty is the *gate*, not the questions.

### Methodology precedents for "the image/2nd view is necessary" (this is STANDARD, not novel)
- **Making the V in VQA Matter (VQA v2)** — Goyal et al., CVPR 2017, arXiv:1612.00837.
  Balanced complementary image pairs → blind model at chance *by construction*.
- **VQA-CP** — Agrawal et al., CVPR 2018, arXiv:1712.00377. Shift answer priors train≠test.
- **MMStar** — NeurIPS 2024 [verify venue], arXiv:2403.20330. **Adopt its metrics AND its audit
  pattern.** Filters with **blind LLMs** (keeps only visually-dependent samples), reports
  **MG (multimodal gain) = S_v − S_wv** and **ML (multimodal leakage) = max(0, S_wv − S_t)**.
  Audit precedent with hard numbers: ScienceQA 57.2% / AI2D 46.2% lack visual necessity,
  >20% MMMU blind-solvable.
- **UNO-Bench** — arXiv:2510.18915. **Nearest twin to our idea:** a construction gate —
  *"if the question becomes unanswerable or ambiguous after removing any one modality,"* it
  is certified to need both, via **human** judgment (durable). But UNO uses **modality-type**
  channels (image vs text vs audio), NOT two views of one scene — our generalization to
  view-vs-view is the delta.
- **Human-ceiling precedent** (for the two-sided certificate): GQA human 89.3% / blind LSTM
  42.1% (arXiv:1902.09506); MMMU expert 82.6–88.6% (arXiv:2311.16502).
- **Shortcut-learning frame** (why single-view-solvable = a shortcut artifact, and why
  model-specific filters decay): CLEVR (arXiv:1612.06890, bias-minimized synthetic
  construction), "Women also Snowboard" (arXiv:1803.09797, occlusion/region-ablation
  diagnostic), Geirhos et al. "Shortcut Learning" (Nature MI 2020, arXiv:2004.07780),
  construct-validity decay (arXiv:2511.04703).
- PID synergy/redundancy/uniqueness (info present only in the JOINT): Williams & Beer 2010
  (arXiv:1004.2515); multimodal operationalization Liang et al., NeurIPS 2023,
  arXiv:2302.12247 [verify ID]. NOTE: MultiBench (arXiv:2107.07502) is robustness/benchmark
  framing — do NOT cite it for the synergy taxonomy; cite the 2023 PID paper for that.

### ⚠ Pre-submission verification flags (flagged by research agents, not yet re-fetched)
- Verify **arXiv:2302.12247** is the correct ID for the Liang et al. PID/interaction paper.
- Verify **MMStar's venue** (stated NeurIPS 2024 D&B — confirm).
- The IDs marked [unverified] in the prior-art survey (CityCube, SatDreamer360, etc.) — check
  before citing. **"CityVQA" does not exist — do not cite it.**

### The cross-view-localization line we are explicitly NOT (retrieval, not VQA)
- CVUSA (ICCV'15, 1510.03743), CVACT (CVPR'19, 1903.12351), **VIGOR** (CVPR'21, 2011.12172,
  ground-offset-within-aerial-patch), University-1652 (ACM MM'20, 2002.12186, closest on
  *inputs* — multi-view ground + satellite, but building retrieval). SAFA = NeurIPS'19
  **proceedings only, no arXiv ID** (do not cite 1907.05021).

### Sat+street fusion-for-understanding (motivation; classification/regression, not VQA)
- **Workman et al., ICCV 2017, arXiv:1708.03035** — the canonical fusion ablation: building
  function 27%(ground)/35%(overhead) → **45% fused (+10.2)**. Srivastava et al. RSE'19
  (1905.01752, land-use +11 over street), Suel et al. RSE'21 (income +20%). Survey: Qiao &
  Yuan IJGIS'21 (2101.04827).

### Aerial-only RS-VQA (the "we are not satellite-only" contrast)
- RSVQA (TGRS'20, 2003.07333, OSM-derived satellite-only VQA — our ancestor), EarthVQA
  (2312.12222), GeoChat (2311.15826), VRSBench (2406.12384), LHRS-Bench (2402.02544).

### ⚠ Citation hygiene (caught in survey — fix before submission)
- **"CityVQA" does not exist.** Do not cite it. Likely you meant City-3DQA (2407.17398) or
  CityEQA (2502.12532). Remove/replace.
- BEV / "cross-view transformer" (LSS 2008.05711, BEVFormer 2203.17270, CVT 2205.02833) are
  **multi-camera→top-down, NO satellite** — pre-empt this confusion in related work, don't
  cite as prior art.

### Novelty claim (hardened)
Prior cross-view localization is *retrieval*; prior sat+street fusion is *pixel
classification/regression*; prior RS-VQA is *aerial-only*; and **UrBench — the one benchmark
that pairs both views in VQA — proves models fail at fusion but never certifies that fusion is
required.** EOLLM v2's contribution is the first **VQA benchmark with a per-question
modality-necessity gate for two complementary views of one scene** (satellite-only collapses
AND street-only collapses AND joint succeeds), used to *construct and filter* the data — a
generalization of MMStar's vision-necessity filter and UNO-Bench's modality-removal gate from
image-vs-text to **view-A-vs-view-B**, on a controlled 1-satellite + 4-directional-street rig
with a stored road bearing. **Lead with the gate; cite UrBench + MMStar + UNO-Bench head-on.**

---

## 6. Risks / what could still go wrong

- **OSM cache missing in this checkout** (`dataset/data/*.json` empty). New labels need OSM
  re-fetched from stored `lat/lon` or the cache restored from the remote. Seeds (`lat/lon`,
  `road_bearing`) are 100% present, so this is recoverable.
- **Correspondence is itself hard** (VIGOR Recall@1 ≈ 41%). Make the satellite referent
  *unambiguous* (a marked footprint / arrow) so the task tests "can't see it without the
  other view," not "can't find it" — otherwise you accidentally benchmark cross-view
  retrieval instead of fusion reasoning.
- **Street-view 4×90° frames tile 360° exactly but with zero overlap;** objects on a 45°/
  135°/… seam split across two frames. For frame-selection tasks, either avoid near-seam
  referents or (if re-fetching) widen FOV to ~100–120° for overlap.
- **Don't over-trust "blind ≈ chance"** — it's necessary but not sufficient. The current 9
  mostly aren't badly text-leaky; they're *single-view-leaky*, which only the sat/sv (not
  blind) conditions catch. The gate checks single-view explicitly for this reason.
