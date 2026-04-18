# EOLLM Dataset v2 — Regeneration Context

**Purpose.** Single handoff document for regenerating the EOLLM VQA dataset so that a model trained on it can be honestly evaluated and published at a top-tier vision conference (CVPR/NeurIPS-class). This is not a TODO list for the current remote run — the current run uses v1 as-is. This is the spec the v2 regeneration pass must satisfy.

**Sources synthesized.**
- `AUDIT_REPORT.md` — training-pipeline audit (Agents A, B, C, 2026-04-18).
- `DATASET_ISSUES.md` — dataset deep audit (Agent D, 2026-04-18).
- `CATEGORY_A_VERIFICATION.md` — model-visibility verification (Agent E — pending at time of writing; will augment once complete).
- Domain-expert review (2026-04-18): notes on option diversity, filename obfuscation, and practical regeneration scope.

**Reader.** You are the person who will run the v2 regeneration pipeline (the user or a collaborator on the dataset-generation side). Read this end-to-end before touching anything.

---

## 0. TL;DR

The v1 dataset works as *training data* for exploratory runs but **cannot ship to a conference** in its current form. Key failures:

1. **4 of 14 topics are 100%-solvable without vision** — either via a JSONL metadata field (`image_mode`), a filename-embedded city leak (`option_stv_paths`), or a closed-set answer pool (`green_space`). Some of these reach the model; some only reach a reviewer who inspects the JSONL. Both matter.
2. **Benchmark is not disjoint** from train. 4 sample pairs are byte-identical across splits (20 images × shared hashes). The "held-out" claim in the README is false.
3. **Non-vision baseline on v1 val = 66.7%**, benchmark = 67.6%. A text-only control model reaches two-thirds of the accuracy without any image understanding, so any trained model that reports 75% is effectively demonstrating ~8 pp of learned vision + 67 pp of prior memorization.
4. **Structural diversity is low**: 10 question templates per topic (0 unique to val), 8-option pool per topic with only 2 strings ever correct for some topics, 10 templates × no paraphrase-held-out split.
5. **Class imbalance** reduces whole topics to trivial (road_surface = 95% Asphalt) or near-unlearnable (building_height skyscraper = 1.6% of train).

A v2 regeneration must fix 1–4 and substantially improve 5. Below, each issue is specified: cause, blast radius, proposed fix, acceptance check.

---

## 1. Architecture decisions to make before regenerating

These are not bugs; they are design choices the v1 pipeline got wrong and v2 must get right. Decide first, build second.

### 1.1 What the model sees vs what the JSONL exposes

The v1 JSONL exposes far more than what the model reads. `image_mode`, `city`, `mismatch_is_match`, `option_stv_paths`, etc. are all present per record. `training/data.py::convert_record` only routes a handful of those into the model's prompt — but a reviewer running a text-only baseline reads the full JSONL and wins.

**Decision for v2**: two tiers of JSONL.
- `train_internal.jsonl` — full schema, used by training code.
- `train_public.jsonl` (and val, benchmark analogues) — minimal schema: only `question_id`, `question`, `options`, `answer`, `images` (flat list), `topic`, `difficulty`. Nothing else. This is what external researchers see.

Alternative considered and rejected: strip fields dynamically during training. Harder to enforce consistency with the published benchmark.

### 1.2 Filename obfuscation

Image filenames currently encode the source city (e.g. `lisbon_0036_cross_right.jpg`). A reviewer or a file-path-aware model wins trivially on `mismatch_mcq_easy` by reading city names from paths.

**Decision for v2**: hash all image filenames with a salted SHA-256 truncated to 12 hex chars. Keep the mapping in a private `filename_index.json` so the generation pipeline can reference images by semantic key (e.g. `lisbon_0036_cross_right`) while the published dataset sees only `a3f8c1e7b2d4.jpg`. The training code should never touch the original semantic names — it reads images via the published paths only.

### 1.3 Split strategy as the published headline

v1 ships two splits (`splits_per_city` and `splits_seen_unseen`) with no clear guidance on which is canonical. `splits_per_city` has 72 leaked val-train location duplicates; `splits_seen_unseen` is clean.

**Decision for v2**: publish `splits_seen_unseen` as the canonical benchmark (cleanest, hardest, most defensible). `splits_per_city` can ship as an auxiliary split with an explicit label "in-distribution reference; not the headline metric."

### 1.4 Template provenance

v1 has 10 question templates per topic, reused verbatim between train and val. A model trained on v1 never sees a novel question phrasing.

**Decision for v2**: generate ≥40 templates per topic (LLM-paraphrase an existing seed template). Assign ≥10 to a held-out val-only pool. Record template_id per question so reviewers can reproduce template-wise breakdown.

### 1.5 Option pool sizing

v1 green_space has 8 option strings in total, of which only 2 ever appear as correct. A model can learn "only these 2 strings are ever the answer; pick whichever is present" with zero vision.

**Decision for v2**: ≥8× expansion of the option-string pool per topic (see §4.3). Every option string can appear as correct with probability proportional to the underlying physical rate (green space present / absent) and never drop below ~5% of records as correct.

---

## 2. Conference-blockers — must be fixed before any external release

### 2.1 `image_mode` / routing-field leakage

**Cause.** The generation pipeline writes `image_mode ∈ {streetview_binary, streetview_composite}` into the JSONL. `streetview_binary` always means "No match"; `streetview_composite` always means "Yes match". The field is public-schema.

**Blast radius.** All 4 mismatch topics × all 3 splits = 11,442 records. A reviewer running `python -c "import json; [print(r['image_mode'], r['answer']) for r in ...]"` sees the bijection in 30 seconds.

**Whether the model actually sees it today.** Needs verification by Category A agent (pending). Based on static read of `training/data.py`, the field is used only to route code branches — not interpolated into prompt text. If Category A confirms this, then the leak is JSONL-only, not model-visible.

**Fix.**
- Collapse `image_mode` values so the match/no-match label is NOT distinguishable from the field. E.g., both rendering branches use `image_mode: "streetview_pair"`.
- Move the routing decision to a private `render_variant` field that never makes it into `*_public.jsonl`.
- Alternatively, drop `image_mode` from the public schema entirely; the image list + record metadata already suffice for downstream code.

**Acceptance check.** For every topic, the joint distribution of `image_mode` × `answer` in the public JSONL must be within 5 pp of independence. Compute: `chi2_contingency(image_mode, answer) → p-value > 0.05`.

### 2.2 Filename city leakage in `mismatch_mcq_easy`

**Cause.** `option_stv_paths` is a list of 4 filename strings; exactly one is from the same city as the record's satellite, the other 3 are cross-city. Filename format: `{city}_{sample_id}_{angle}.jpg`. Reading the prefix gives the city; the correct answer is always the same-city group.

**Blast radius.** 2,000 train + 561 val + 421 bench = 2,982 records (100% of mismatch_mcq_easy).

**Whether the model actually sees it today.** Filenames are NOT fed to the model — `data.py` passes `PIL.Image` objects, not paths. So the leak is JSONL-only. BUT: a visual city-recognition model (CLIP, any VLM with geographic knowledge) can potentially identify the city from the SV image itself, not just the filename. That's a content-level leak, not a filename leak — see §2.3.

**Fix.**
- **Filename side**: hash all filenames (§1.2). No city string in any path.
- **Content side**: replace cross-city distractors with **same-city far-distractors**. Minimum separation 1 km from the query coordinate; must be in the same city. Removes the "which option is from Lisbon" shortcut and forces the model to actually compare satellite↔SV content.
- Alternative: drop `mismatch_mcq_easy` entirely. It's the weakest topic; salvaging it with same-city distractors gives a genuinely harder test, which might be what the paper needs anyway.

**Acceptance check.** For every `mismatch_mcq_easy` record, all 4 option SV groups must come from the same city as the satellite, at distances ≥1 km. Verify: for each record, compute min distance between satellite coord and each option's SV coord; assert ≥1 km for distractors, ≤X m for correct (where X is the tolerance for "this SV is at this sat").

### 2.3 Content-level city leakage (harder sibling of 2.2)

**Cause.** Even with hashed filenames and same-city distractors, a model that recognizes cities from SV content (e.g. canal → Amsterdam, bridges → Venice) can use city recognition as a shortcut for *any* mismatch task that crosses cities.

**Blast radius.** All mismatch topics. Harder to quantify without running a vision-only city classifier.

**Fix.**
- Enforce same-city constraint on all mismatch distractors (not just `_easy`).
- Consider same-neighborhood constraint (within 1 km) for mismatch_binary_hard. The "_hard" suffix should mean something.

**Acceptance check.** For every mismatch record, all distractor SVs are within 5 km of the satellite coord. For `_hard` variants, within 1 km.

### 2.4 `green_space` closed-set collapse

**Cause.** 8 unique option strings exist in the `green_space` pool. Only 2 of them ever appear as the correct answer. A text-only model learns "answer is whichever of these 2 strings is in the options list" and wins 100%.

**Blast radius.** 1,500 train + 267 val + 203 bench = 1,970 records (100% of green_space).

**Whether the model sees it.** Options are always in the prompt text. Unambiguously model-visible.

**Fix.**
- Regenerate green_space with uniform correct-string sampling from the full 8-string pool (or 64+-string pool after §4.3 expansion).
- Alternative if 8 strings can be collapsed semantically: recognize that green_space is effectively a binary task ("is there green space nearby, yes/no") and ship it as `mismatch_binary`-style with only 2 options — but then rebalance yes/no distribution.

**Acceptance check.** For the green_space topic, the per-option correct-rate must be ≥5% of topic records for every option. Verify: `Counter(correct_string for r in green_space_records).most_common()` — no entry may be below `0.05 * n_records`.

### 2.5 Benchmark is not disjoint from train

**Cause.** The generation pipeline didn't enforce image-content disjointness between train and benchmark. 4 sample_id pairs share byte-identical satellite PNGs + all 4 SV JPGs: `lisbon_0001↔lisbon_0036`, `kayseri_0101↔kayseri_0069`, `kayseri_0023↔kayseri_0067`, `barcelona_0109↔barcelona_0082`. Plus 4 bench coords <10 m from a train coord, 20 <100 m, 251 <1 km.

**Blast radius.** 17 image groups directly leaked; a larger number of near-duplicates. README claims benchmark is drawn from disjoint cities and locations — factually false.

**Fix.**
- At benchmark-construction time, compute SHA256 of every train+val image. Exclude any benchmark record whose images collide.
- Enforce minimum coordinate separation between benchmark and train/val coords: ≥100 m minimum, ≥1 km preferred.
- Rebuild benchmark from coords that pass this filter.

**Acceptance check.** `len(set(hashes(bench)) & set(hashes(train|val))) == 0`. Plus coord histogram: no bench coord within 100 m of any train/val coord.

### 2.6 Template non-diversity

**Cause.** 10 templates per topic, 0 templates unique to val. A model trained on v1 train has seen every val question verbatim (different coords, same phrasing).

**Blast radius.** All 14 topics. Structural.

**Fix.**
- Generate ≥40 templates per topic via an LLM paraphrase step. Seed each with a manually-written canonical template, request 40 semantically-equivalent paraphrases, deduplicate.
- Split templates: ≥10 train-only, ≥10 val-only, ≥10 benchmark-only, rest shared. Record `template_id` per question.
- Paraphrase step should preserve answer semantics verbatim — verify by regenerating the "correct answer" from the paraphrased template and checking it matches.

**Acceptance check.** Per topic: `|templates_in_train ∩ templates_in_val| = 0`, `|templates_in_val| ≥ 10`.

---

## 3. Reviewer-flags — fix for a clean submission

### 3.1 Answer-letter distribution skew (29/29/21/21 overall)

**Cause.** Option shuffling uses `random.Random(f"{sample_id}_{topic}")`. Binary-mismatch topics have only 2 options (A, B) which inflates A/B over C/D in the aggregate. The per-topic distribution is close to uniform on 4-option topics but not exactly uniform.

**Blast radius.** All topics; most-biased single topic is `amenity_richness` at 31% modal B.

**Fix.** Enforce strict uniform correct-letter distribution per topic. Post-shuffle, if the modal letter's fraction > 28%, re-shuffle the worst-offending records until uniform-ish.

**Acceptance check.** Per topic, max(P(A), P(B), P(C), P(D)) ≤ 0.27 for 4-option topics; P(A) ∈ [0.48, 0.52] for binary.

### 3.2 `road_surface` trivial via "pick shortest option" = 94%

**Cause.** 95% of correct answers are "Asphalt" (the shortest string in a 4-option list whose other entries are compound descriptors). A model that learns "pick shortest" wins.

**Blast radius.** 2,000 train + 398 val + 167 bench. Topic is near-unsolvable-by-vision given the class imbalance.

**Fix.** Two options:
- **Exclude from headline.** Document that road_surface accuracy ≤ 94% is expected as the class-imbalance ceiling. Report it as a calibration check, not a headline topic.
- **Rebalance.** Sample locations with non-Asphalt surfaces (dirt, gravel, cobblestone, paver) at 15% minimum per class. Requires re-sampling from OSM, which is a dataset-generation change.

**Acceptance check.** Per option in road_surface, min fraction as correct ≥ 0.15.

### 3.3 `building_height` skyscraper near-zero class

**Cause.** 23/1431 train records (1.6%) have "Skyscraper" as correct. Most cities in the dataset don't have skyscrapers.

**Blast radius.** 1,431 train records for building_height.

**Fix.** Over-sample cities with skyscrapers (NYC, Shanghai, Dubai, Tokyo) for building_height specifically. Alternative: merge "Skyscraper" into "High-rise" so the topic is 3-class instead of 4-class, rebalance to 33/33/33.

**Acceptance check.** Per class in building_height: min fraction ≥ 0.15.

### 3.4 `difficulty` field is topic-determined

**Cause.** Every amenity_richness record has difficulty=medium, every mismatch_binary_hard has difficulty=hard, etc. The field is a redundant encoding of topic.

**Blast radius.** Cosmetic — but papers citing "evaluated across difficulties" would be technically accurate and practically misleading.

**Fix.** Either:
- Derive per-record difficulty from actual signal (e.g. mismatch_binary_hard records where the distractor's distance from query is small = harder).
- Drop the field from the public schema. Topics already carry difficulty implicitly via their hard/easy suffix.

**Acceptance check.** Within at least one topic, ≥2 difficulty values present with comparable record counts.

### 3.5 72 val-train location duplicates in `splits_per_city`

**Cause.** Same coordinate sampled twice during generation with different `sample_id`s, then split-strategy put one copy in each split. Image bytes identical across the duplicates.

**Blast radius.** 72 val records (1% of val). `splits_seen_unseen` is unaffected.

**Fix.** In the splitting code, hash every sample's images (sat + 4 SV). Refuse to place two samples with overlapping image hashes in different splits. If detected, move both to the same split (prefer train) or drop one.

**Acceptance check.** Zero image-hash collisions across train ↔ val for any split strategy.

---

## 4. Diversity and balance — what v2 must widen

These are quantitative targets, not blockers. A v2 dataset that meets the conference-blockers but not these is technically acceptable but thin.

### 4.1 City balance

v1 has 40 cities; record distribution is roughly uniform but not enforced. Some (city, topic) cells are empty or near-empty.

**Target.** ≥30 records per (city, topic) in train. Cities with fewer OSM-viable locations in a topic (e.g. no skyscrapers in Reykjavik) can be exempted on an explicit allowlist — not silently dropped.

### 4.2 Geographic diversity within a city

v1 uses random sampling within the city bbox; near-duplicate nearby sampling creates the 72 val-train location leaks.

**Target.** Minimum pairwise distance between intra-split samples: 500 m. Minimum cross-split distance: 100 m (strict) or 1 km (preferred).

### 4.3 Option-string pool

v1 has 8 option strings per topic. v2 target: ≥40.

**How to generate.** For each topic, write a canonical correct-answer string and 3–5 canonical distractor strings. LLM-paraphrase each (GPT-4o, Claude, local Qwen — the choice doesn't matter) to produce ~10 variants per canonical. Deduplicate by cosine similarity (keep similar-enough variants; reject exact duplicates or near-duplicates above 0.95 sentence embedding similarity). Result: ~40+ option strings per topic.

**Constraint.** Every generated string must preserve the topic's underlying semantics. Validate by regenerating the "correct answer for this location" using the topic's generation logic — the string from the pool must match one of the generated canonicals' paraphrase set.

### 4.4 Question template pool

Same as §2.6 acceptance check: ≥40 templates per topic, ≥10 val-only, ≥10 benchmark-only.

### 4.5 Temporal diversity

v1 streetview_date and satellite_date fields exist but have never been audited for distribution. An ideal v2 has mixed dates (2018–2025) per city, so a model doesn't learn "2024 street view → Amsterdam" style spurious correlations.

**Target.** Per city, SV dates span ≥3 years; satellite_date span ≥3 years. Cross-check that no (topic, answer) correlates with a specific date.

---

## 5. Non-changes — things v1 got right

Record these so v2 doesn't regress:

- **Topic coverage** — 14 topics is good breadth across urban-understanding tasks.
- **Image_mode semantic routing** — the rendering decisions for each mode are visually reasonable (satellite_marked + mega composite for MCQ, arrow-overlaid satellites for camera_direction, etc.). The problem isn't the rendering; it's that `image_mode` leaks into the public schema.
- **`camera_direction` construction** — the only topic that's genuinely hard and not leaked. Preserve it in v2.
- **On-the-fly composite generation** — `composite_utils.py` keeps disk footprint small. Preserve it.
- **`splits_seen_unseen`** — clean geographic separation, zero leakage. Make this the headline split.
- **`option_arrow_angles` metadata** — useful for analyzing camera_direction performance. Keep in the internal schema.
- **`mismatch_is_match` field** — the semantic label. Keep in internal, strip from public.

---

## 6. v2 build order (suggested)

1. **Freeze the v2 schema** — decide what's in `_internal.jsonl` vs `_public.jsonl`.
2. **Rewrite `dataset/src/05_generate_questions.py`**:
   - ≥40 templates per topic, separated by split role.
   - ≥40 option strings per topic, correct-string uniform sampling.
   - Strict answer-letter uniformity enforcement.
3. **Rewrite `dataset/splitting/split_dataset.py`**:
   - Hash-based image-overlap detection across splits.
   - Coordinate-distance enforcement (§4.2).
   - Same-city distractor enforcement for mismatch topics (§2.2, §2.3).
4. **Rewrite filename handling**:
   - Hash all filenames into the published `images/` directory.
   - Keep semantic name → hash map in private `filename_index.json`.
5. **Build acceptance-test script** (`dataset/splitting/test_v2_integrity.py`):
   - Run every acceptance check listed above.
   - Fail loudly on any violation.
   - Produce a `dataset_card.md` summarizing per-topic/per-split statistics.
6. **Regenerate** with the v1 seed locations as input (so the physical sampling is comparable), but applying all v2 logic.
7. **Re-run training** against v2; verify non-vision baseline drops to <35% (up to ~30% random majority-class floor; any higher indicates residual leakage).

---

## 7. What the remote run (v1) should report

Given we're training on v1, the published results from the remote run must carry appropriate caveats:

- **Report non-vision baseline alongside every accuracy number.** v1 non-vision baseline (stacked modal-string + city-match + image_mode leak + green_space closed-set) = **66.7% val, 67.6% bench**. The model's reported accuracy minus this baseline is the real learned signal.
- **Report per-topic accuracy against per-topic random AND majority baselines.** Already implemented in `callbacks.py::_write_verdict`.
- **Exclude the 4 conference-blocker topics** from headline numbers or clearly footnote them. The 4 are: `mismatch_binary_easy`, `mismatch_binary_hard`, `mismatch_mcq_easy`, `green_space`.
- **Report on both `splits_per_city` and `splits_seen_unseen`.** `splits_seen_unseen` is the honest number.
- **Report benchmark accuracy with the 4 contaminated sample_ids excluded** (`lisbon_0001`, `kayseri_0069`, `kayseri_0023`, `barcelona_0109`, and any derived question_ids).

This framing preserves the value of the v1 run as a capability demonstration while deferring the conference-grade claim to post-v2.

---

## 8. Open decisions flagged for the dataset owner

1. **GSV redistribution legality.** The public benchmark contains Google Street View imagery. Redistribution terms need explicit legal review before shipping to a public venue. (Out of scope for this document; flag to the PI.)
2. **Regenerate or prune.** For each §2 blocker, the fix can be "prune affected records" (fast, loses samples) or "regenerate from source" (slow, preserves scale). Decide per-topic.
3. **Whether v2 keeps backward compatibility with v1 sample_ids.** If yes, fixing coords requires re-emitting same IDs with new content; if no, v1 numbers become uncomparable with v2.
4. **Benchmark-only cities.** For `splits_seen_unseen` to be a meaningful test, benchmark should ideally come from cities not in either train or val. Currently benchmark draws from the same 40 cities. Consider restricting benchmark to, say, 5 dedicated cities.

---

## 9. Appendix — audit evidence pointers

- Agent C programmatic verification of v1 leakage: `audit_scratch/out_01..13_*.txt`, scripts `audit_scratch/0{1..13}_*.py`.
- Agent D conference-grade deep audit: `DATASET_ISSUES.md`, scripts `audit_scratch/D_*.py`.
- Agent E (model-visibility verification): `CATEGORY_A_VERIFICATION.md` (pending — will confirm per-field whether v1 leaks are model-visible or JSONL-only; update §2.1 verdict once complete).
- Domain-expert review: this document, §0 and §2.
