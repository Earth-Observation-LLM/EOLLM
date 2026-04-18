# EOLLM Dataset — Audit Report (Agent D, 2026-04-18)

Dataset root: `/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final/`
All numbers are reproducible from scripts at `/home/ezel/Development/EOLLM/audit_scratch/D_*.py`.

---

## 1. Executive Summary — Top 5 issues

1. **`image_mode` field is a perfect classifier for both mismatch_binary topics** (4 topics × ~2,000 records in train, 100% of val + bench). The value `streetview_binary` always labels "No, not a match"; `streetview_composite` always labels "Yes, match". No vision required, no prior required. Trivially exploits 4/14 topics across every split. `D_04_schema`, `D_05_mismatch_leaks`.
2. **`mismatch_mcq_easy` is solvable at 100% by city-matching street view filenames alone** (561/561 val, 421/421 bench). Exactly one of the four option groups has SV images from the satellite's city; the other three are distractors from other cities. Any model that can identify a city from SV (or that reads the `option_stv_paths` string) wins without comparing to the satellite. `D_05c`.
3. **`green_space` is solvable at 100% by a closed-set trick** (267/267 val, 203/203 bench): only 2 of 8 option strings ever appear as correct answers in train; the other 6 are always distractors. A text-only model with training exposure gets 100%. `D_12`. This compounds with the 78% modal-string hit Agent C reported — the real ceiling is 100%.
4. **Combined non-vision baseline reaches 66.7% on per_city val, 67.6% on benchmark, 63.9% on seen_unseen val.** A reviewer running a text-only (no-image) Qwen baseline will report these numbers and conclude the dataset is not testing vision. `D_13`.
5. **Benchmark contains 17 byte-identical image groups shared with train** (4 sample_ids across 3 cities: kayseri_0069↔kayseri_0101, kayseri_0067↔kayseri_0023, lisbon_0036↔lisbon_0001, barcelona_0082↔barcelona_0109), plus 4 benchmark coords within 10 m of a train sample and 20 within 100 m. The "held-out disjoint benchmark" claim in the README is false in practice. `D_03_image_phash`, `D_08`.

---

## 2. Severity-Ranked Findings

### 2.1 CONFERENCE-BLOCKERS

#### B1. `image_mode` leaks the answer for mismatch_binary_{easy,hard}
- **Affected:** 4 topics × splits_per_city (4000 train, 1122 val), splits_seen_unseen (4000 train, 1478 val), benchmark (842). Total: 11,442 records.
- **What's wrong:** The generator used two `image_mode` values (`streetview_composite`, `streetview_binary`) to dispatch rendering; both topics use both values. Verified distribution:
  ```
  per_city/train:  binary_easy { binary→NO:1008, composite→YES:992 }
                   binary_hard { binary→NO:1037, composite→YES:963 }
  bench:           binary_easy { binary→NO:196,  composite→YES:225 }
                   binary_hard { binary→NO:235,  composite→YES:186 }
  (Same pattern across all splits. 0 cross-overs.)
  ```
- **Why blocker:** The JSONL field is read directly by `training/data.py:62` and `composite_utils.py:266,271`. A model doesn't even need to see the image — the scaffold gives it the answer. Reviewers running an image-free control will report 100% on these 4 topics, which invalidates the mismatch task entirely.
- **Proposed fix:** Rename `image_mode` so it doesn't distinguish match/non-match (e.g. always `streetview_composite_binary`), OR shuffle which SV source is shown under a single `image_mode`, OR split generation logic so `image_mode` reflects only the visual layout, not the label. Re-emit train/val/bench.

#### B2. `mismatch_mcq_easy`: correct option is the sole same-city SV
- **Affected:** 2000 train + 561 val + 421 bench records.
- **What's wrong:** For every record, the 4 `option_stv_paths` groups are drawn as 1 matching (sat_city) + 3 cross-city distractors. Verified: `('n_same_city', 1): 2000` and `('correct_is_the_only_same_city', True): 2000` in train; 100% in bench. A trivial classifier over SV filename prefixes or any "which option looks like Rome" heuristic wins.
- **Why blocker:** The `_easy` variant claims to test sat↔SV alignment. It actually tests city ID. A CLIP-or-worse model hits 100% without any satellite input.
- **Proposed fix:** Use multi-city distractors only when they depict similar-looking city content (e.g. geographically clustered; same-climate; same-continent). Better: replace "cross-city easy" with "same-city far-distractor easy" (minimum separation 1 km).

#### B3. `green_space` closed-set collapse
- **Affected:** 1500 train + 267 val + 203 bench records. Per_city val 100% solvable.
- **What's wrong:** 8 unique option strings exist in the pool; only 2 ever appear as correct answers:
  ```
  correct: "No, the area is densely built with no parks nearby"   1007
  correct: "Yes, green space or parks are present nearby"          493
  distractors only (never correct): 6 strings
  ```
  Because exactly one of the two "correct-pool" strings appears in every record's options, a text-only model that simply memorizes the pool achieves 100%.
- **Why blocker:** Reviewer runs a text-only baseline → 100% → topic is untestable.
- **Proposed fix:** Sample the correct answer string uniformly from the full pool of 8 (or deduplicate to 2 strings and make clear this is a binary question).

#### B4. Benchmark is not held-out: 17 byte-identical image groups with train/val
- **Affected:** 4 samples × 5 images each = ~20 images; at least 8 bench records share sat+4SV bytes with a train record.
- **Evidence:**
  ```
  lisbon_0001 ↔ lisbon_0036 (full 5-image byte-identical set)
  kayseri_0101 ↔ kayseri_0069, kayseri_0023 ↔ kayseri_0067
  barcelona_0109 ↔ barcelona_0082
  ```
  Plus 4 bench coords <10 m from a train sample, 20 <100 m, 251 <1 km. Plus 1 byte-hash question crossing bench↔train already flagged by Agent C: `lisbon_0001_mismatch_binary_1` shares content hash with `lisbon_0036_mismatch_binary_1` and the question text is identical.
- **Why blocker:** README says benchmark is drawn from "a disjoint set of cities and locations not present in either split strategy's training data." That's false. A reviewer checking bench↔train image hashes will find the collisions in <1 minute.
- **Proposed fix:** Remove collision records from benchmark (or regenerate bench from exclusive coordinates), add a CI check to the pipeline.

#### B5. Combined non-vision baseline: ~60-67% across all splits
- **Script:** `D_13_summary_leaks.py` (see Appendix A.5).
- Stacking [image_mode leak] + [mcq_easy city oracle] + [green_space closed-set] + [(city, topic) majority string] + [topic modal string]:
  - per_city val: **4659/6984 = 66.71%**
  - benchmark:    **3541/5240 = 67.58%**
  - seen_unseen val: **5641/8831 = 63.88%**
- **Why blocker:** If a trained model only beats this by, say, 15 pp, the vision contribution is ≤15 pp. Reviewers will argue the benchmark is dominated by priors and structure leaks, not perception.
- **Proposed fix:** (a) Remove the four 100%-leak topics from the headline metric, report them only as sanity checks. (b) Balance topic-correct option distributions. (c) Publish the non-vision baseline in Table 1 and the paper.

#### B6. Only 10 question templates per topic; train=val templates
- Reported by Agent C; extended here. No template is unique to val (0/14 topics). The model has seen every val question string verbatim during training. Combined with (1)–(3), the dataset can't measure generalization to novel phrasing.
- **Fix:** Generate 5-10x more paraphrases; hold out a paraphrase-disjoint slice.

### 2.2 REVIEWER-FLAG

#### R1. 72 val records at 6 byte-identical train locations (splits_per_city only)
Agent C finding confirmed. Cross-referenced with `D_03_image_phash`: the same 6 location pairs each have all 5 images byte-identical (satellite + 4 SV). This arose because the same coordinate was sampled twice during dataset generation with different `sample_id`s, then split strategy put one copy in train and one in val. `splits_seen_unseen` does not have this issue (it separates by city, not location).

#### R2. `mismatch_binary_hard` same-city negative SV may not be sufficiently ambiguous
`mismatch_strategy="same_city"` means the negative SV is drawn from the same city but a different `sample_id`. However the shown SV composite has no metadata to confuse humans; if same-city means same neighborhood (verified for spot-check record: sat_city=amsterdam, shown_sv_city=amsterdam), a model can likely still do this task. But we haven't verified geographic *proximity* of negatives — if "same city" means "same Amsterdam", negatives could be at opposite ends of the city and distinguishable by gross visual features.
- **Fix:** Quantify distance between sat-location and negative-SV location for `mismatch_binary_hard`; require negatives within, say, 1 km.

#### R3. Topic-conditional position leak in `mismatch_mcq_easy`
For 727/2000 train records, the correct SAMPLE happens to land in position "Top-Right" (538 records, 26.9%). The distribution is near-uniform, so no *position* leak; but the answer-letter distribution (with position-to-letter shuffle) is near-uniform too (≈25% each). No new finding here — just confirming position not a leak.

#### R4. `difficulty` field is perfectly determined by topic
`D_09`. Every `amenity_richness` record is `medium`; every `mismatch_binary_hard` is `hard`; etc. The field carries zero per-record information. Papers citing "evaluated across difficulties" would be misleading.
- **Fix:** Either derive per-record difficulty (e.g. based on how close the distractor values are to the correct), or drop the field.

#### R5. Road surface is trivial
95.1% of train answers are "Asphalt". Per-city asphalt rate: 31/40 cities have ≥95% asphalt; 15 cities have 100% asphalt. Agent C's "shortest option" heuristic hits 94.2% val, confirmed here. The topic contributes no signal to a learned model above a single-token prior.
- **Fix:** Rebalance road_surface sampling so each non-Asphalt class has ≥15% representation in train/val.

#### R6. Class imbalance: `skyscraper` in building_height
23/1431 (1.6%) train, 6/261 (2.3%) val, 4/190 (2.1%) bench. Effectively unlearnable. Agent C finding confirmed.

#### R7. Building_height has per-city modal dominance
E.g. Chicago = 98% "Low-rise"; Paris = 61% "Mid-rise"; Vienna = 80% "Mid-rise"; Stockholm val = 92% "Mid-rise". A model that learns city → height gets ≥57% on val without looking at buildings.

#### R8. Benchmark reuses 100% of train templates
0/14 topics have novel bench templates. Reviewers may question whether bench is a benchmark or a re-sampling.

#### R9. Benchmark answer distribution non-uniform per topic
Top examples from `out_01_basic_stats.txt`:
- `green_space` in bench: A=32% D=19% (imbalanced vs 25/25/25/25).
- `camera_direction` in bench: D=29% (close enough, but with only 4 topics having 25% vs ≥28% majority, an "always-D" attack hits ~29% overall).

#### R10. Ground truth for some "heuristic" topics may be generator-default
- `green_space` train: 1007 "No" vs 493 "Yes" (67/33). Given the closed-set of just 2 correct strings, the answer may simply be a binary city-level coin-flip rather than OSM-grounded. `D_12` shows only 1 city has modal-answer 100% (so not city-level defaulted), but the 2-string collapse is itself a symptom. We could not verify OSM grounding without API calls.

### 2.3 QUALITY

- **Q1.** Non-ASCII characters: only typographic em/en dashes (U+2014, U+2013). Benign for modern tokenizers; harmless.
- **Q2.** No duplicate options within a record; no <2-option records; no invalid answer letters; no missing required fields. Schema is clean. `D_04`.
- **Q3.** Satellite resolution uniform 512×512, SV 640×640. No EXIF. File sizes reasonable (sat 317-617 KB, SV 58-120 KB).
- **Q4.** All `image_mode` values match the expected topic dispatch when ignoring the binary ambiguity (the only "mismatches" are the topic↔mode crossings flagged in B1).
- **Q5.** SV date range 2007-09 to 2026-02; satellite_date is mostly "latest" (94%). Temporal gap between sat and SV can be 18+ years but is not split-dependent. Not an integrity issue, but worth documenting.
- **Q6.** City totals per split are imbalanced: per_city val min 74 / max 218 (2.95× ratio). Not a blocker but makes per-city analysis noisy for small cities.
- **Q7.** 9 (city, topic) pairs have 0 records in per_city val, 13 in bench. Mainly small-city + rare topic combos (Bursa/green_space, Samsun/road_surface). Could destabilize per-city metrics.

### 2.4 COSMETIC

- **C1.** `satellite_source` field is `ESRI`/`NAIP`/`IGN` etc. but `satellite_date` is often just `"latest"` — information-poor. Not a bug, but worth documenting.
- **C2.** `generation_method` is always `"template"` — field adds no information, remove or populate.
- **C3.** Benchmark public vs with-answers: identical except for null-vs-letter answer. All other fields match. OK.

---

## 3. Per-Topic Risk Assessment (can this survive peer review?)

| Topic | Verdict | Rationale |
|---|---|---|
| `amenity_richness` | Yes-with-caveat | (city,topic) modal = 48% val; needs a vision-informative gap of >20 pp to look real. |
| `building_height` | **No** as-is | Skyscraper 1.6%, city-prior 57%, narrow numeric ranges; tiny n=261 val. Rebalance required. |
| `camera_direction` | **Yes** | Only topic with no notable leak (baseline 23-29%); well-balanced; query angle always = answer's arrow angle (by construction, not a leak). Strongest topic. |
| `green_space` | **No** as-is | 100% text-only baseline. Must regenerate with uniform-distractor sampling. |
| `junction_type` | Yes-with-caveat | 67% (city,topic) modal; 60% "Unsignalized" dominant. Rebalance or accept as easy. |
| `land_use` | Yes-with-caveat | 69% (city,topic) modal; 53% "Residential" dominant. Closed-set leak (2 distractor-only strings) reduces 4-way to 6-way effective. |
| `mismatch_binary_easy` | **No** as-is | 100% image_mode leak. Must fix generator. |
| `mismatch_binary_hard` | **No** as-is | 100% image_mode leak. Same fix. |
| `mismatch_mcq_easy` | **No** as-is | 100% city-match oracle. Must redesign distractor sampling. |
| `mismatch_mcq_hard` | Yes-with-caveat | 23% baseline (random = 25%). Well-designed in principle, but all-same-city distractors means SV similarity is the actual task (test what model perceives, not just memorized). |
| `road_surface` | **No** as-is | 95% "Asphalt"; 94% "shortest option" baseline. Trivial. Rebalance or remove. |
| `road_type` | Yes-with-caveat | 64% (city,topic) modal; 60% "Local residential" dominant. |
| `transit_density` | Yes | ~45% (city,topic) baseline, lowest among the numeric topics. Best text-only-resistant grounded topic. |
| `urban_density` | Yes-with-caveat | 63% (city,topic) modal; 55% "Low density" dominant. |

Survivors that need minor care: `camera_direction`, `transit_density`, `amenity_richness`.
Fatal flaws that MUST be fixed: `green_space`, both `mismatch_binary_*`, `mismatch_mcq_easy`, `road_surface`.

---

## 4. splits_per_city vs splits_seen_unseen — which is the defensible headline?

| Dimension | `splits_per_city` | `splits_seen_unseen` |
|---|---|---|
| Train→val coord leakage | 6 exact-dup locations (72 val records, 1.0%) | 0 (min distance to any train sample is ≥10 km) |
| Train→val byte-identical images | Yes (10 confirmed, 5 sample pairs) | 0 for same-city; but: same `reykjavik_0062` appears BOTH in per_city_train AND seen_unseen_train AND seen_unseen_val via the same duplicate coordinate (see `D_03` cross-split groups of 4) |
| Tests location generalization within known cities | ✓ | ✗ (cities disjoint) |
| Tests cross-city generalization | ✗ | ✓ |
| City count val | 40 | 8 (Chicago, Seoul, Moscow, Sydney, Singapore, Cape Town, Istanbul, Rio) |
| Val size | 6984 | 8831 |
| City imbalance val | min 74 / max 218 (2.95×) | min 865 / max 1368 (1.58×) |

**Recommendation:** `splits_seen_unseen` is stronger as the headline benchmark (no coord leak, explicit geographic generalization framing, more balanced). It has two caveats:
- (a) Only 8 cities, 3 of them European-adjacent (Moscow, Istanbul, Cape Town); Sydney, Seoul, Chicago are OECD. Diversity is okay but "new cities" means 2 Asian, 2 Latin-American, 1 African, 1 Oceanic, 1 European, 1 US. A hostile reviewer can still claim overrepresentation of Western/urban cities.
- (b) The 6 duplicate-location pairs that were train/val leaks in per_city are NOT leaks in seen_unseen BUT they land entirely in seen_unseen/train (both copies). A smarter model gets 2 copies of the same location during training — minor over-sampling, not a leak.

All non-vision leaks (B1, B2, B3) apply equally to both splits. Fixing them is prerequisite either way.

---

## 5. Reproducibility / Datasheet checklist

- **Generation seed:** `dataset/src/05_generate_questions.py` uses `random.seed(42)`. Not verified that all RNG paths are seeded (Overpass caching is file-based so stable; SV/sat fetching is external). Without running the pipeline (API-gated by user rule), we cannot confirm bit-exact reproducibility. **Open question.**
- **Data provenance:** `satellite_source` is recorded per image; `streetview_source="Google Street View"` for all. Terms-of-service compliance for redistributing the 7 GB compressed dataset is not asserted in the README. Google SV TOS likely prohibits public redistribution of the raw JPEG tiles. **High-risk for publication; discuss with user.**
- **PII:** Spot-check SV images not performed (no GPU / no bandwidth to look at thousands). SV imagery is blurred by Google for faces/plates, but worth manual audit before release.
- **Licensing:** Not documented in README. Missing.
- **Evaluation script:** `benchmark_public.jsonl` can't be scored locally; user would need a submission server. Not shipped. For a proper benchmark, an `eval.py` + leaderboard are expected.
- **Datasheet coverage:** README is a usage guide, not a Gebru-style datasheet. Missing motivation, composition, collection process, preprocessing, uses, distribution, maintenance sections.

---

## 6. Open Questions for the User

1. **Are you OK redistributing Google Street View JPEGs?** Section 5 flag. If not, the dataset must ship with a fetch script + location CSV only.
2. **Is `image_mode` intended to encode the ground truth label for mismatch_binary?** If this was deliberate (to simplify the dataloader), at minimum the field name and documentation must change to reveal the leak. If accidental, regenerate with a single value for both match/non-match.
3. **Do you want to rebuild benchmark from disjoint coordinates, or accept the 17 byte-identical overlaps as "we didn't notice"?** The former requires pipeline rerun (API calls). The latter means documenting the overlap in limitations.
4. **Target reviewer:** A CVPR reviewer running text-only Qwen will report 67% baseline. Do you want to prune the 4 blocker topics before submission, or ship all 14 and cite the baseline?
5. **building_height skyscraper: undersample other buckets or oversample skyscraper?** There are only 23 skyscraper records total in train; you cannot learn the class.
6. **Closed-set leak in green_space / land_use: regenerate with full-pool uniform sampling?** (Green_space mandatory, land_use optional: the "never-correct" strings in land_use are only 2/8, a lesser issue.)
7. **splits_seen_unseen city choice**: Reviewers may ask why these 8. Is there a principled motivation (continent diversity, climate, language, etc.) or was it a random draw?
8. **camera_direction `query_sv` inclusion**: training code (`data.py:72-79`) explicitly drops the query SV. That means the model NEVER sees the SV — the task reduces to "which of 4 arrow-annotated satellites has an arrow pointing in the given direction". Is that intended? If so, the task is "read the arrow and match the angle", not "correlate SV with satellite direction".

---

## 7. Appendix — Raw Outputs

All `D_*.py` scripts and `out_D_*.txt` outputs live at `/home/ezel/Development/EOLLM/audit_scratch/`. Agent C's `01-13` scripts remain untouched.

### A.1 image_mode leak verification (`D_04` + inline)
```
per_city/train:
  mismatch_binary_easy image_mode=streetview_binary    {'NO': 1008}
  mismatch_binary_easy image_mode=streetview_composite {'YES': 992}
  mismatch_binary_hard image_mode=streetview_binary    {'NO': 1037}
  mismatch_binary_hard image_mode=streetview_composite {'YES': 963}
bench:
  mismatch_binary_easy image_mode=streetview_binary    {'NO': 196}
  mismatch_binary_easy image_mode=streetview_composite {'YES': 225}
  mismatch_binary_hard image_mode=streetview_binary    {'NO': 235}
  mismatch_binary_hard image_mode=streetview_composite {'YES': 186}
```

### A.2 mismatch_mcq_easy city-match oracle (`D_05c`)
```
per_city/train n=2000  ('n_same_city',1):2000  ('correct_is_the_only_same_city',True):2000
bench         n=421    ('n_same_city',1):421   ('correct_is_the_only_same_city',True):421
mcq_easy city-match oracle on val: 561/561=1.0000
mcq_easy city-match oracle on bench: 421/421=1.0000
```

### A.3 green_space closed-set (`D_12` + inline)
```
All green_space option strings & counts (train):
   1007  The area has extensive waterfront promenades and green space
   1007  Yes, a large public park is adjacent to this area
   1007  Yes, several pocket parks and green corridors exist
   1007  No, the area is densely built with no parks nearby
    493  No, the area is fully built-up with no visible green space
    493  Yes, green space or parks are present nearby
    493  Only small private gardens exist, no public green space
    493  The area is primarily industrial with no recreational space
Correct-only:
   1007  No, the area is densely built with no parks nearby
    493  Yes, green space or parks are present nearby
```

### A.4 Benchmark leakage (`D_08` + `D_03_image_phash`)
```
bench sample_ids in per_city (train∪val): 0 / 421
bench sample_ids in seen_unseen (train∪val): 0 / 421
bench nearest-train (per_city) buckets: {'<10m': 4, '<100m': 20, '<1km': 251, '<10km': 146}
bench coords overlapping to 1e-6: 4 / 421
Byte-hash groups: bench↔per_city_train↔seen_train shares: 17
Examples:
  lisbon_0001 ↔ lisbon_0036 (sat + 4 SV)
  kayseri_0101 ↔ kayseri_0069 (sat + 4 SV)
  kayseri_0023 ↔ kayseri_0067 (sat + 4 SV)
  barcelona_0109 ↔ barcelona_0082 (sat + 4 SV)
```

### A.5 Strongest non-vision baseline (`D_13`)
```
STRONGEST on PER_CITY val: 4659/6984 = 0.6671
  amenity_richness          269/561 = 0.4795
  building_height           149/261 = 0.5709
  camera_direction          132/561 = 0.2353
  green_space               267/267 = 1.0000
  junction_type             301/448 = 0.6719
  land_use                  388/561 = 0.6916
  mismatch_binary_easy      561/561 = 1.0000
  mismatch_binary_hard      561/561 = 1.0000
  mismatch_mcq_easy         561/561 = 1.0000
  mismatch_mcq_hard         127/561 = 0.2264
  road_surface              375/398 = 0.9422
  road_type                 359/561 = 0.6399
  transit_density           254/561 = 0.4528
  urban_density             355/561 = 0.6328
STRONGEST on BENCH: 3541/5240 = 0.6758
STRONGEST on SEEN_UNSEEN val: 5641/8831 = 0.6388
```

Pure-leak sub-test (only exploit B1+B2+B3, no priors): 1950/1950 = 100.00% correct on the 27.9% of val where at least one leak fires. Zero vision, zero training data beyond the structure.

### A.6 Building-height per-city modal (`D_07`, train)
```
Chicago          n=46  modal=Low-rise             frac=0.98
Vienna           n=81  modal=Mid-rise             frac=0.80
Tallinn          n=71  modal=Low-rise             frac=0.69
Stockholm        n=67  modal=Mid-rise             frac=0.66
Helsinki         n=76  modal=Mid-rise             frac=0.63
Berlin           n=71  modal=Low-rise             frac=0.62
Paris            n=83  modal=Mid-rise             frac=0.61
Budapest         n=67  modal=Mid-rise             frac=0.60
Zurich           n=70  modal=Low-rise             frac=0.57
...
```

### A.7 Difficulty field (`D_09`)
Every (topic, difficulty) pair is 1-to-1: difficulty=topic-constant. Example train:
```
amenity_richness   {'medium': 2000}
mismatch_binary_hard {'hard': 2000}
mismatch_binary_easy {'easy': 2000}
camera_direction   {'medium': 2000}
```

---

_End of report. 14 scripts written to audit_scratch/D_*.py; 12 corresponding out_D_*.txt snapshots committed alongside. No edits made outside audit_scratch/._
