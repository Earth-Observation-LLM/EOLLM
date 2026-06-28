# V2 Implementation Plan — paper rewrite

**Status:** plan for review. Nothing in `main.tex` is touched yet.
**Decisions locked (user, 2026-06-28):**
1. Item-level oracle/interference analysis → **keep & port to satfwd** (verified feasible & reproducible, see below).
2. Statistical apparatus (bootstrap CIs, McNemar, TOST) → **keep all of it**.
3. Delivery → **this plan first**, then LaTeX after approval.

---

## ★ FINAL CONSOLIDATED VERDICT (both sweep waves, 12 reviewer-agents total) — BUILD-READY

**Headline: this is an accept if written defensively. Zero new compute. The thesis is intact and CLEANER on satfwd than v1 claimed.** Every fix below is writing or re-analysis on data in hand; all numbers verified ≥twice.

**The 5 things that win the paper (priority order — these ARE the build):**
1. **Reframe the cross-view control as a SENSITIVITY proof, not an input-matched comparison** (urban full=2 imgs, cross-view full=5). Carry the no-fusion claim on within-model evidence (same adapter fuses on cross-view → sat channel live; RS/raw panels → not an adapter artifact). One explicit paragraph atop §4.2. [F1 — highest leverage]
2. **Lead the item-level section with the ORACLE stat** (full−oracle = −12.3, CI [−13.7,−10.9], McNemar p≈3e-66). Present interference 9:1 as the SAME finding stated two ways (identity verified), not corroboration. **Drop the false v1 McNemar-full-vs-sat claim** (p=0.117 NS on satfwd). **Flip the TOST**: synergy IS equivalent-to-zero on satfwd (pooled +1.1, CI crosses 0). [F3]
3. **Lead the whole paper with the DISSOCIATION; demote benchmark to enabling apparatus.** Contributions: (i) task-dependence/dissociation, (ii) item-level interference, (iii) OELLM benchmark, (iv) audit protocol. Paste-ready abstract/hook/bullets/pre-emption paragraphs in agent outputs. [F4 / Marchetti]
4. **Position UrBench/CityLens as the *phenomenon* to your *mechanism*** ("convergent prior observations make this audit timely; they do not make it redundant"). Name the oracle/interference result as the unscoopable delta. Label CityLens "concurrent." [Onwe]
5. **Harden §3 with the funnel + count-integrity fixes:** released corpus (34,484/12 tasks) vs eval slice (urban7 n=2,629); **4,734 = 2,629 + 2,105** (VERIFIED); provider counts to corpus-level 31,489/1,946/1,049 (drop 4775/305/160, they sum to 5240≠4734); **kill every "2,932"**; one name **OELLM** bold-first-mention. [Quill]

**Free wins (already-computed, zero effort, high value):**
- **Leakage-robustness sentence in §4.3:** oracle gap −12.3 all items vs −12.2 on blind-WRONG items + 0% refusal all modes → kills "oracle just exploits text-leakable items." [Onwe/red-team, VERIFIED]
- **Letter-skew turned to strength:** urban7 χ²=3.6/always-A 25.2% (clean) vs corpus χ²=176/29.3% and cross-view χ²=324.8/34.4% (skew lives there) + within-item-invariance shield. [Quill, VERIFIED]
- **§6 "SV=4 images" paragraph is OBSOLETE and now HELPS** — satfwd SV condition = 1 forward view, so the image-count confound largely evaporates. Reconcile = free win. [Quill]
- **Negative-synergy tasks (amenity sat 55.3>full 53.0) = interference made concrete**, not a smell. [red-team]
- **Either-view-sufficiency (Pt10 defense) is already in the data** — both single views beat blind/chance widely; sat≈sv per task. No human study needed. [Onwe/Quill]

**NO-GO (do not request compute):** re-run raw VLMs at 1SV (wrong axis, doesn't fix F1); more attention analysis (reviewer liability); human task-validity study (accuracies already make the point). [Onwe, ruthless]

**Figures (Quill):** Fig1 pipeline (have) + Fig2 = two REAL renders `ex_land_use.png`(urban) + `ex_camera_direction.png`(cross-view) + page-1 combined teaser (Q5/Q6) + the divergent fusion-gain bar (teaser-right) + fig17_generalisation_design for §3.3 splits if space. Everything else → "see technical report [cite]".

**Build-phase open items (minor):** (a) locate exact §6 cross-modal attention numbers (benchmark_suite ATTENTION WATCH) or keep attention as hedged corroboration; (b) decide EOLLM vs OELLM and grep repo+paper for the loser; (c) confirm satfwd ep4 is the final checkpoint with the team (data already reproduces it).

---

## ⚑ PIVOTAL FINDINGS from the refinement sweep (Wave 1) — must shape §4.2

These are the highest-leverage items; all VERIFIED against prediction JSONLs.

**F1 — [MAJOR, red-team] The positive control is NOT input-matched to the headline. Reframe it.** VERIFIED: urban `full` = **2 images** uniformly (sat + forward SV); cross-view `full` = **5 images** (camera_direction: 1 SV query + 4 sat-arrow crops; mismatch: sat + 4 option-SVs — though mismatch_mega = sat + 1 mega-grid = 2). The paper's syllogism is "ablation detects fusion when present (control) ∴ urban null is real, not measurement failure." A sharp reviewer severs it: "maybe your 2-image setup just can't fuse." **FIX (pure writing, highest-leverage edit in the paper):** at the top of §4.2, (a) reframe the cross-view control as a proof of *measurement sensitivity* (the instrument + model CAN register large synergy), NOT an image-count-matched comparison — say so explicitly; (b) carry the load-bearing weight on within-model image-count-controlled evidence the paper already has: the **same adapter fuses on cross-view** (so the satellite channel is live, not dead) + the **RS/raw panels show the urban regime with no sat+1SV training** (so it's not this adapter's artifact). Make that chain one explicit paragraph. Never let a sentence imply the control holds image count fixed.

**F2 — [robustness GOLD, red-team] The interference is NOT a text-leakage artifact.** VERIFIED: full−oracle gap = −12.3 over all 7-urban items AND **−12.2 over the 1,357 blind-WRONG (vision-needed) items**. The 2nd view hurts precisely where the model needs the image. ALSO: **0% refusal / 0% hedge in all four modes** (no parsing confound). → Cite both in §4.3 as robustness; they pre-empt "your oracle just exploits text-solvable items" and "maybe the model refused."

**F3 — [MAJOR, stats] THREE v1 statistical claims are FALSE when ported to satfwd. All confirmed by the stats-auditor AND independently re-verified. Zero new compute to fix.** The v1 paper's item-level stats were computed on **5STV per-city**; porting them verbatim to satfwd would put false numbers a methods reviewer recomputes in 5 minutes. Corrected (n=2629 paired 7-urban, seed 3407, all VERIFIED twice):

| v1 claim (5STV) | TRUTH on satfwd | Action |
|---|---|---|
| McNemar full-vs-single "p<1e-3, 219/133, p=5e-6" | full-vs-**sat** 174/145 **p=0.117 (NS)**; full-vs-**sv** 349/242 **p=1.3e-5**; full-vs-**oracle** 40/363 **p≈3e-66** | Drop full-vs-sat significance. Headline = **full-vs-oracle** (p≈10⁻⁶⁶). |
| pooled synergy "+2.2 [0.9,3.3]" (excludes 0) | pooled **+1.10, CI [−0.23,+2.43] → CROSSES 0** | Replace number; CI now includes 0. |
| TOST "does NOT declare negligible — real small positive effect" | **TOST DECLARES equivalence to zero** (CI inside ±3 margin) | **Flip the conclusion.** Synergy is statistically ≈0. |
| means "+1.2/+1.3" | unweighted topic mean **+0.49**, item-weighted +0.38, pooled +1.10 | Purge 5STV means. |

  - **The headline stat is full−oracle = −12.29, CI [−13.69, −10.88] (excludes 0), McNemar p≈3e-66.** Lead the item-level section with THIS, not synergy.
  - **CRITICAL de-dup (identity):** `(full−oracle) ≡ (fusion_win−interfere)/n` — VERIFIED `(40−363)/2629 = −12.29`. The "oracle beats full by 12.3" and "interfere 9:1 over fusion-win" are the SAME statistic. Present as ONE finding, two views — NEVER as two corroborating results (reviewer catches the double-count).
  - **Guard:** because synergy is now ≈0/equivalent, LEAD with the oracle harm result (significant) so no reviewer reads "n.s. ⇒ they claimed no-effect from a null." The harm claim rides on the oracle test, not the synergy n.s.
  - Per-task synergies = **descriptive**; inference rides on pooled bootstrap + oracle McNemar → state "no multiple-comparison correction needed" once. If per-task CIs are drawn on fig:synergy, label "nominal, uncorrected, for visualization."
  - Prune 89%/67% = **retention of already-correct items, NOT accuracy** — mandatory clause. top-vs-random@k1 = 90 vs 85 (modest); k0 floor 67% = residual leakage. Wording in stats-auditor output.

**F4 — [narrative, Prof. Marchetti] Lead with the DISSOCIATION, demote benchmark to enabling apparatus.** The unattackable asset is "same model+instrument: ≈0 on urban, +71 on cross-view." Reorder contributions: (i) task-dependence/dissociation result, (ii) item-level interference, (iii) OELLM benchmark, (iv) reusable audit protocol. Full abstract, hook, contribution bullets, and two pre-emption paragraphs (Pt1/7 behavioral framing; Pt10 either-view-suffices, with sat≈sv data) drafted in agent output — paste-ready. Forward-pointer at end of §4.2 MUST carry the number ("§4.3 shows the 2nd view removes 14% of correct answers while rescuing 1.5%"), not a vague promise.

**F-prov — provenance checks done.** Sample4Geo corroboration VERIFIED (binary easy 0.639/0.646, hard 0.622/0.646 > 0.50 chance → the "~0.65 binary, cross-view signal is real" claim holds). Prune-attention: forward SV is attention's #1 image **94.8%** of the time; top≈fwd (89.5 vs 89.1). **OPEN (build-phase, minor):** the §6 cross-modal attention numbers (6.3%/2.0% cross-modal at answer token; per-image 24.3% sat vs 18.9% per-SV) are NOT in attention_prune/ (that dir is about *which SV*, not sat-vs-sv cross-modal) — they're likely from the benchmark_suite "ATTENTION WATCH" run. Locate/confirm exact values before drafting §6, or keep attention strictly as hedged corroboration (cite jain2019/wiegreffe2019, as v1 does). NOT load-bearing.

**F5 — [framing, red-team] Turn the negative-synergy tasks into evidence, not smell.** amenity_richness (sat 55.3 > full 53.0; SV hurts) and road_type (−1.4): frame as the interference made concrete — forward SV under-samples an OSM-radius-defined attribute it can't see. One sentence §4.2 + one in limitations. Also demote RS-VLM comparison explicitly ("not a superiority claim; establishes the single-overhead floor + regime-split-not-our-artifact") and fence the 27B-prune / AWQ-prompting model swaps with a stated reason in each caption.

---

## 0. The one-line summary of the change

V1 paper = "the second view adds little / mildly harms" demonstrated on the **5STV** model (sat+4SV) with an item-level oracle argument, two splits, three prompting strategies.

V2 paper = same finding **reframed as an audit** ("When Does the Second View Help?") with the **satfwd** model (sat+1SV) as primary, organized around a **regime split** (urban tasks: no fusion; cross-view tasks: all fusion), with the item-level oracle/interference analysis *ported onto satfwd* as the mechanistic core, and the 5STV model demoted to a "what do 3 more views buy?" supplement.

**Crucial verified fact:** the item-level story is *stronger* on satfwd, not weaker (table below). So we lose nothing by flipping the headline model, and we gain the "minimum-sufficient input" framing.

---

## 1. Feasibility check — DONE, all green

Ran against `benchmark_suite/results/qwen9b_satfwd_seen_unseen__benchmark/` (checkpoint `merged_ep4`, the `SatFwd_vs_5STV` headline ckpt — confirmed via `meta.json`).

- Per-mode prediction JSONLs exist (`blind/sat_only/sv_only/full_predictions.jsonl`), each record carries `question_id` (joinable across modes), `is_correct`, `topic`, `n_images`, `image_roles`. ✓
- Pooled 7-urban accuracy reproduces the published anchor **exactly**: blind/sat/sv/full = 48.4/64.1/61.1/65.2 (anchor: 48.3/64.0/61.4/65.3). ✓ → confirms ep4.
- Cross-view tasks are in the **same** run/checkpoint → Table 2 (urban) and Table 4 (cross-view) come from one model, one eval. ✓
- Item-level analysis ports cleanly (n=2629 paired 7-urban items).

### Item-level: 5STV (current paper) vs satfwd (ported) — satfwd is sharper

| Quantity | Current paper, 5STV per-city | **satfwd, ported (7-urban)** |
|---|---|---|
| Fusion-win (full✓, both single✗) | 0.9% | **1.5%** |
| Interfere (full✗, a single✓) | 10.2% | **13.8%** |
| Interfere : fusion-win ratio | ~10:1 | **9.1:1** |
| full − oracle | −9.3 | **−12.3** |
| McNemar full-vs-sat discordants (b/c) | 219/133 | **174/145** |

> Note one nuance to handle in prose: on satfwd the McNemar full-vs-sat discordant split (174 vs 145) is *less* lopsided than the oracle gap suggests, because the interference is spread across both single views; the dramatic asymmetry is clearest in full-vs-**oracle** and in the fusion-win-vs-interfere counts. We state McNemar honestly (still significant by count imbalance; recompute exact p in the build script) and lead the asymmetry argument with the fusion-win/interfere ratio + oracle gap, which is where satfwd is strongest.

**TODO in build script (not yet written):** exact McNemar p-values, bootstrap 95% CIs on pooled synergy, full−oracle, and fusion-win/interfere rates (10k resamples over the 2629 items), and the TOST equivalence test. These are mechanical given the JSONLs; I will write `make_v2_numbers.py` to emit a single JSON the LaTeX reads from, so no number is hand-typed.

---

## 2. Numbers that change (v1 → v2), all verified against `EOLLM_results_wide.xlsx` + prediction JSONLs

| Where | V1 value | V2 value | Source |
|---|---|---|---|
| Primary model `full` | 5 images (sat+4SV) | **2 images (sat+1 fwd SV)** | `SatFwd_vs_5STV` Model B |
| Headline split | per-city + seen/unseen | **seen/unseen only** | satfwd trained on SU |
| Task count | 8 (incl. road_surface) | **7** (drop road_surface too) | outline; road_surface blind_heavy |
| Urban mean blind/sat/sv/full | 56.1/67.8/68.2/70.4 (pc) | **48.3/64.0/61.4/65.3** | `SatFwd_vs_5STV` B, 7-urban |
| Mean fusion synergy | +1.2/+1.3 | **+0.49** | derived, verified |
| Per-task synergy range | −1.0..+4.5 | **−2.4 (amenity) .. +2.9 (transit)** | verified |
| Cross-view synergy | +17..+70 (5STV) | **+19 (cam_dir), +44..+71 (mismatch)** | satfwd benchmark preds |
| 5STV urban full (supp) | — | **67.6** (Δ +2.3 over satfwd) | `SatFwd_vs_5STV` A |
| Prune survival fwd-1 / k0 | — | **89% / 67%** | `Prune_Survival` 27B |
| RS-VLM sat-only range | 26.2–39.9 | **26–41** (EarthDial best 41.3) | `External_RSVLM` |
| Our trained sat-only ref | 66.6 (pc, 8-task) | **64.0** (satfwd 7-urban sat) or 54.9 (9urban, per CSV) — pick one denominator, see Q | sheets differ by scope |

**Test-set size changes:** dropping road_surface: 8-task n=2932 → 7-task n=2629 (confirmed from JSONL join). Table 1 / dataset stats table updates accordingly.

---

## 2b. Dataset-report reconciliation (authoritative source: `scratch_reports/report_12tasks/latex/main.tex` + the 12 example renders + `benchmark_with_answers.jsonl`). ALL VERIFIED.

The 12-task dataset technical report is the authoritative source for §3. Key facts and **conflicts with the current paper** found and resolved:

- **Corpus size:** 34,484 question records · 4,168 unique locations · 43,435 unique qids (aggregating both split strategies) · ~7GB · 12 task families (after retiring green_space + road_surface). 40 cities / 32 countries (Turkey+Turkiye merged from 33).
- **Splits (real, match outline Table 1 exactly):** per_city 23,431 / 6,319 · seen_unseen 23,282 / 8,179 · benchmark 4,734. The benchmark `with_answers.jsonl` on disk actually has **5,240 records / 14 topics** (it still *contains* green_space[203]+road_surface[303]; they're filtered at eval). The 4,734 = the 12 released tasks; the **7-urban eval subset = n=2,629** (verified). Current paper's "2,932 / 8 tasks" is the retired 8-task count → update to 7-task / 2,629 OR report the all-12 benchmark 4,734 with the 7-urban subset called out. **Decide denominator presentation in §3.5 (sweep Q).**
- **Satellite provider counts — current paper is WRONG-scope.** Paper says ESRI 4,775 / NAIP 305 / IGN 160 (these were *benchmark-only*). Report (full corpus): **ESRI 31,489 (91%) / NAIP 1,946 / IGN 1,049**, no S2 fallback. Pick one scope and label it; don't mix.
- **LETTER-DISTRIBUTION CONFLICT — RESOLVED IN PAPER'S FAVOR (important).** Report Limitations: corpus-wide A/B each ~10,100, C/D each ~7,100, "always-A ~29%", documented RNG artifact. Current paper §6 claims near-uniform (743/766/703/720, χ²=3.08). **Verified on the real benchmark file:**
  - Full benchmark (14 topics, n=5240): A/B/C/D = 1533/1550/1061/1096, **χ²=164 (heavily skewed), always-A=29.3%**.
  - **7 urban tasks (n=2629): A/B/C/D = 662/694/628/645, χ²=3.6, always-A=25.2% (genuinely uniform).**
  - → The skew lives in the **cross-view tasks**; the 7 urban headline set is clean. §6 robustness MUST: (a) scope the uniformity claim to the 7 urban tasks, (b) acknowledge corpus-wide skew (cite report), (c) note cross-view carries it. Update χ² to the real 3.6/n=2629. This is a cross-document landmine — a reviewer reading both would catch the contradiction; scoping it removes the attack AND the honest "always-A=29% corpus-wide" actually strengthens the leakage-audit framing.
- **water_proximity:** defined in config but **no records survive** → unreleased. Don't mention as a task; if listed anywhere in current paper, cut.
- **camera_direction render confirmed:** 4 satellite crops each with a red arrow (A/B/C/D) + **one** SV query (a single panoramic, not the 2×2 grid). Confirms 1-SV-by-design. Urban render = sat+red-dot (left) + 2×2 grid of 4 SV (right) + 4 options.
- **Better figures available (not in current paper):** `fig17_generalisation_design.pdf` (seen/unseen design: 3,807 seen-city + 927 unseen-city benchmark records), `fig18_region_coverage.pdf`, `fig15_topic_by_family.pdf`, `fig14_task_taxonomy.pdf`. Consider for §3 — fig17 is a clean splits/generalisation visual; fig14 a clean taxonomy visual. All in `scratch_reports/report_12tasks/latex/figures/`.
- **Difficulty field:** template-level annotation per task, not per-question model difficulty; "hard" bucket ≈ entirely mismatch_*_hard. State honestly if difficulty is mentioned.
- **12 example renders** exist (`scratch_reports/examples_12tasks/examples/ex_*.png`) — real per-task QA panels. Source for Figure 2 (one urban + one cross-view), higher quality than re-rendering.

---

## 3. Section-by-section: keep / cut / rewrite / move

Legend: **KEEP** (verbatim or near), **REWIRE** (same prose, new numbers), **REWRITE** (new text), **MOVE**, **CUT**, **NEW**.

### Title & Abstract
- **REWRITE title** → "When Does the Second View Help? Auditing Cross-View Fusion in Urban VLMs" (outline). *Conflict with current "Single View Might Be Enough" — outline wins. Confirm you're happy losing the current title; it's catchy but the audit title is the reviewers' pick.*
- **REWRITE abstract** from the outline's one-paragraph version, but **graft in** the item-level sentence (interference 9:1, oracle beats full by 12pts) and the "fusion +0.5 / cross-view +44–70" regime split. Strip from final §1+§5+§6 per drafting order. Keep "task-dependent" framing.

### §1 Introduction — REWRITE (keep skeleton)
- Keep the gap-framing opening (current L90–94 is good).
- **Soften "selecting rather than fusing"** everywhere → lead with the *behavioral+item-level* claim: "derives little additional benefit, and on the items the second view changes it overturns ~9 correct answers for each it rescues." This satisfies reviewer Pt1/7 *with evidence* rather than by hedging.
- Preview numbers: +0.5 urban / +19–71 cross-view / 89% prune / +2.3 from 3 more views.
- Keep the 4 contributions but rewire to satfwd primary + add the pruning/economy contribution.
- **Figure 3 decision (reviewer Pt4):** I recommend the divergent-bar fusion-gain figure stays as Fig.3 in §4 AND we *reference* it from §1; putting the actual float in §1 is unusual for ECCV 2-col and crowds the intro. → flag as your call (Q5).

### §2 Related Work — REWIRE (light)
- Current §2 is strong and already cites UrBench/CityLens convergence, Suel regression counterpoint, Liang synergy formalism. **Keep nearly verbatim.**
- One sentence add: differentiate on "we audit *where in input-mode space* performance comes from" (outline closing sentence).
- No number changes here.

### §3 Benchmark — REWIRE + one NEW subsection
- §3.1 Construction: **KEEP** (satellite-source/resolution prose is excellent and unchanged).
- §3.2 Tasks: **REWRITE** to 7 urban + promote cross-view to a named family. **ADD the Point-10 task-design defense** (reviewer): state urban tasks are *potentially answerable from either modality* and back it with the data — in `SatFwd_vs_5STV`, sat vs sv are within a few pts on most urban tasks (land_use 72.5/71.7, building_height 60.0/62.6, junction 71.6/66.5, etc.). This pre-empts "your tasks are built to not need fusion." **This is one of the two substantive reviewer fixes.**
- §3.3 Splits: **REWIRE** — seen/unseen is primary; mention per_city exists, results in supp. Update counts from the real split report (Table 1).
- §3.4 Eval protocol: **KEEP** (greedy, fixed seed, word-boundary parser, floors).
- §3.5 Coverage/limits: **REWIRE** — keep Europe-heavy honesty; one sentence retiring green_space AND road_surface.
- Dataset stats table: **REWIRE** to 7 tasks, n=2629, satfwd image mode (sat+1SV note).

### §4 Experiments — the big restructure
Outline order (with reviewer Pt5 reorder applied): 4.1 setup → 4.2 headline audit → **4.3 cross-view (was after replication; moved up)** → 4.4 replication → 4.5 pruning → 4.6 supplement. *(Note: outline §ordering and reviewer Pt5 want cross-view BEFORE replication; I've applied that.)*

- **§4.1 Setup — REWRITE**: satfwd primary (LoRA r16 α16, all-linear, bf16, 1×96GB; input = marked-sat + forward SV = 2 images). State clearly "full = 2 images here." Keep the 4-mode definitions. Keep RS-VLM table (Tab. models) — **KEEP verbatim**, still single-overhead-only.
- **§4.2 Headline audit — REWRITE** (was §5.1 `sec:main`): Table 2 = satfwd 7-urban blind/sat/sv/full + synergy, seen/unseen. New numbers. Articulate the +0.5 mean, the two negative-fusion tasks, the flat ones. Lead-in to item-level.
- **§4.2(b) / merge: Item-level — REWIRE the current `sec:itemlevel` onto satfwd numbers.** Tables `tab:oracle` + Fig `fig:oracle` regenerated from satfwd preds. This is the mechanistic heart now. Keep the three subsection structure (overturns-more-than-rescues / oracle-beats-full / synergy-small-but-positive), keep McNemar+TOST+bootstrap, update every number. **Decide:** fold item-level into §4.2 as its second half (recommended — it *is* the answer to "does the second view help?") or keep as its own §4.x. I recommend folding so the headline subsection delivers the full punch.
- **§4.3 Cross-view positive control — REWIRE** (was `sec:control`): Table 4 = satfwd cross-view, verified numbers (cam_dir 42 / mismatch 87–98, synergy +19/+44–71). Keep Sample4Geo corroboration paragraph (retrieval, 0.65 binary) — **KEEP**. Resolve open-decision #2: cam_direction lives here (it's structurally cross-view) but call out its +19 is a different magnitude than mismatch.
- **§4.4 Cross-family/scale replication — REWRITE** (merges current `sec:zeroshot`): raw VLMs (Qwen 4B/9B, Qwen2.5-VL-7B, Gemma-4-12B) + 4 RS-VLMs. Table 3 condensed. **Open data issue (outline + me):** raw VLMs were run at their default SV scheme — confirm 1 vs 4 SV; recommended resolution = keep raw at original scheme, state it (they're zero-shot, view-economy isn't their question). Regime-shape-reproduces claim. *NB current `tab:zeroshot` is 8-task; rebuild at 7-task for consistency, or footnote the scope.*
- **§4.5 Inference-time pruning — NEW** (`Prune_Survival`, 27B): Table 5, fwd-1 89% / random-1 ~85% / k0 67%, forward/top/random/bottom × k=0..3. Caption notes 27B model + survival-not-accuracy + pre-selected-solved pool. Honest about the leakage-adjusted weak signal (README headline read).
- **§4.6 Supplement: 5STV vs satfwd — NEW** (`SatFwd_vs_5STV`): Table 6, 3-more-views = +2.3 urban, comparable cross-view. "minimum-sufficient input" closing.

### §4.x Prompting strategies (current `sec:strategies`) — MOVE to supplement or CUT-down
- Current paper has the 3-strategy (plain/think16k/multistep) AWQ result. The v2 outline **doesn't mention it.** It's good evidence ("more reasoning improves single-view reading, not fusion"). → **Recommend: keep as a short §4.7 or supplementary subsection**, rewired. It directly rebuts "the model *could* fuse with more thinking." Flag as Q (your outline dropped it; I'd keep it).

### §5 Discussion — REWIRE
- Keep the two-regime framing (current §7 is on-message). Update numbers. Keep "what this paper does not claim" (outline §5 echoes it). Keep the deployment/routing implication.

### §6 Conclusion — REWRITE (short, from outline).

### §7 Future Work — NEW/REWRITE
- Add the **3D-reasoning paragraph** (outline §7) AND the reviewer-Pt9 connecting sentence: "Because only cross-view tasks consistently benefit from multiple street views, future work should exploit those views for richer geometric reasoning (local 3D reconstruction)." Keep the fusion-architecture / more-cities / socio-economic bullets.

### §6 Robustness (current `sec:robustness`) — REWIRE, keep
- Excellent material (option-text/position χ², OSM provenance, satellite resolution, image-count/attention). **Keep all**, rewire the road_surface paragraph (now retired not just flagged), update the "8 tasks"→"7 tasks" and re-derive the option-balance χ² on the 7-task set in the build script.

---

## 4. Figures & tables — disposition

| Asset | V1 | V2 action |
|---|---|---|
| Fig teaser (`fig_teaser.pdf`) | Sydney, 5STV ablation | **REGEN** with satfwd 2-image ablation (or keep + recaption: the qualitative point holds; cheaper). Q. |
| Fig synergy_control | 5STV synergy + control | **REGEN** as the divergent fusion-gain bar (Fig.3, reviewer Pt4 shape): all 11 tasks, urban near/below 0, cross-view tower. satfwd numbers. |
| Fig oracle_dumbbell | 5STV per-task oracle vs full | **REGEN** on satfwd 7-task. |
| Tab dataset | 8 tasks | REWIRE → 7 tasks, n=2629. |
| Tab models (RS-VLM specs) | — | KEEP. |
| Tab fusion-main | 5STV both splits | → **Table 2** satfwd 7-urban seen/unseen. REGEN. |
| Tab control | 5STV cross-view | → **Table 4** satfwd cross-view. REGEN. |
| Tab zeroshot | 8-task RS-VLM | → **Table 3** condensed, 7-task. REGEN. |
| Tab strategies | AWQ 3-strategy | KEEP if we retain §4.7 (Q). |
| Tab oracle | 5STV item-level | REGEN on satfwd. |
| NEW Table 5 | — | prune survival. |
| NEW Table 6 | — | 5STV vs satfwd. |
| `make_figures.py` | builds v1 figs | extend / add `make_v2_numbers.py` for the stats JSON. |

---

## 5. Reviewer fixes — explicit tracking

1. ✅ Soften "selecting"→"little additional benefit" **but** back the stronger behavioral claim with the ported item-level interference (Pt1/7). — woven into §1, §4.2, §5.
2. ✅ Task-design defense in §3.2 with sat≈sv per-task data (Pt10). — the second substantive fix.
3. ✅ Reorder §4: cross-view (§4.3) before replication (§4.4) (Pt5).
4. ✅ §7 future-work connecting sentence (Pt9).
5. ✅ Checkpoint locked: ep4, reproduced exactly (Pt5/Pt-implicit). *Still: get a human "yes ep4 is final" before camera-ready.*
6. ✅ Fig.3 divergent bar (Pt4). Placement in §1 = open (Q5).

---

## 6. Build hygiene

- Write `GAIA_ECCVW_2026_Paper_Template/make_v2_numbers.py`: reads the four satfwd prediction JSONLs (+ Prune/5STV/External sheets), emits `v2_numbers.json` with every table cell, CI, p-value, χ². LaTeX tables hand-built from that JSON (or `\input` generated `.tex`), so **no number is typed by hand**. Lets us re-run if ep4 is superseded.
- Compile check after each major section (`pdflatex main` — note axessibility only under pdfTeX; current preamble handles it).
- Commit per section per the user's commit rule (plain `git commit -m`, no co-author line).
- **No API calls** (no new evals needed — all data is on disk; this is pure analysis+writing).

---

## 7. Open questions — RESOLVED by reviewer-persona panel (2026-06-28)

Seven Opus reviewer-agents, each a distinct full-professor persona, ruled per-question on a maximize-acceptance basis. Verdicts locked below; every number they cited was re-verified against the prediction JSONLs / xlsx.

- **Q1 Title → LOCKED: "When Does the Second View Help? Auditing Cross-View Fusion in Urban Vision-Language Models"** (spell out VLMs). *Prof. Hollis:* "Single View Might Be Enough" overclaims — the paper's own cross-view tasks show +71pts, a reviewer writes "title contradicted by authors' own results." Question-form is answered on both sides of the regime split → unfalsifiable framing. "Urban Vision-Language Models" > "Geosensing" for venue searchability. Mitigation: abstract must name the audit instrument in one clause so the title's promise is sized to the evidence.
- **Q2 Item-level placement → LOCKED: HYBRID (dedicated subsection §4.3 + forward-pointer from §4.2).** *Prof. Nakamura:* a finding-stating header gets the reviewer's attention; folding emulsifies the 9:1/12-pt result into the milder +0.5 result and overloads one subsection (table+oracle+McNemar+TOST+CIs). Header = **"The second view interferes more than it helps: an item-level audit."** §4.2 closes with: "an average can hide genuine fusion on a subset of items; §4.3 shows the opposite — at the item level the second view *removes* far more correct answers than it adds." (This shifts item-level to its own §4.3; renumber the downstream §4.x accordingly — cross-view becomes §4.4, replication §4.5, pruning §4.6, supplement §4.7.)
- **Q3 Prompting-strategies → LOCKED: KEEP-COMPRESSED (one paragraph + one small table, main text).** *Prof. Varga:* kills the #1 reviewer reflex ("model could fuse if it reasoned harder") — absence of fusion under a 16k-token budget + 4-turn self-check is a *finding*, not a greedy-decoding artifact. Full subsection invites "why a different model here?"; supplementary-only means no reviewer sees it. Mandatory caveat **in the table caption**: AWQ-quantized Qwen3.5-9B, internal trend check NOT numerically comparable to satfwd headline; claim rests only on the *sign* of synergy (quantization doesn't flip it). Drop-in framing sentence provided in agent output.
- **Q4 RS-VLM denominator → LOCKED: 7-task headline set, unweighted topic mean, recompute externals from per-topic cells.** *Prof. Aldous.* **VERIFIED numbers:** GeoChat **21.0**, SkySenseGPT **27.2**, LHRS-Bot-Nova **29.5**, EarthDial **37.8**, our satfwd-sat ref **64.0** (+26.2 margin). Do NOT copy the sheet's `OVERALL_9urban` (question-weighted, 9 topics — wrong denominator AND wrong averaging rule). Caption must state: unweighted mean over the 7 headline tasks, green_space+road_surface excluded exactly as Table 2. Same exclusion rule in Table 2, Table 3, and the fusion figure — stated once, back-referenced everywhere.
- **Q5 Fig.3 placement → LOCKED: redesign teaser into a two-panel full-width `figure*[t]` on page 1; the divergent fusion-gain bar becomes the RIGHT panel (NOT a separate §4 float).** *Prof. Bellweather.* Left panel (~42%): Sydney setup (marked sat + 4 SV + tiny ablation strip). Right panel (~58%): horizontal divergent bar, all 11 tasks sorted, hard rule at 0, two-regime color tint + bracket. Push teaser width 0.82→`\linewidth`. Frees a single-column §4 slot → backfill with the oracle/dumbbell (the CI-bearing quantitative companion). Page-1 caption thesis provided in agent output ("multi-view fusion is task-shaped, not pervasive").
- **Q6 Teaser → LOCKED: REGEN-BAR-ONLY (keep 4-SV photo panel; swap embedded bar to satfwd; rewrite caption).** *Prof. Greer* caught a landmine: current bar is 5STV urban_density `[54.6,61.8,62.7,65.1]` with a "sat ≈ sv" caption, but **satfwd urban_density = `[54.4, 69.6, 59.1, 70.1]` (VERIFIED), where sat=69.6 ≠ sv=59.1** — the caption's equality claim is false for the headline model. Edits: in `make_figures.py::fig_teaser` swap `vals` to `[54.4, 69.6, 59.1, 70.1]`, change subplot title `"sat ≈ sv ≈ full ≫ blind"` → `"full ≈ best single ≫ blind"`. New caption (provided) names 4 SVs as a *dataset* property and states the model sees *2* images / "full" = sat+forward SV. **Combine with Q5: the regenerated bar IS the teaser-left ablation strip; the teaser-right is the new divergent bar.**
- **Q7 raw-VLM SV count → LOCKED: STATE-AND-SCOPE, NO RERUN (outline option b).** *Prof. de Vries:* the confound can't manufacture the effect — if raw `sv`/`full` carried 4 SVs, that's *more* evidence to fuse and they still showed ~0 urban synergy → conservative reading. Headline is within-satfwd (fixed sat+1SV), untouched by raw input scheme. Two scoping sentences (§4.1 + §4.5) provided in agent output; refuse any "like-for-like view-economy vs satfwd" framing for the raw panel. Doc-hygiene note: raw CSV doesn't log n_images; confirm count from run_modes_vllm.py configs only if camera-ready scrutiny demands.

### Net structural deltas from the panel
1. §4 renumbers: 4.1 setup · 4.2 headline audit (+fwd-pointer) · **4.3 item-level (NEW dedicated, finding-header)** · 4.4 cross-view control · 4.5 replication · 4.6 pruning · 4.7 supplement(5STV) · prompting compressed into 4.2-area or 4.3-tail as a defensive para+table.
2. Page-1 teaser becomes a **combined two-panel** object (Q5+Q6 merge): left=setup+satfwd ablation strip, right=divergent fusion-gain bar. No standalone Fig.3.
3. RS-VLM table = 7-task unweighted, externals recomputed (21.0/27.2/29.5/37.8 vs 64.0).
4. Title + abstract get the audit-instrument clause.
