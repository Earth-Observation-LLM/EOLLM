# Category A Verification — EOLLM Text / Metadata Leakage

**Scope:** Determine, with programmatic evidence, which JSONL fields reach the
model's tokenized prompt during training/inference for Qwen3.5-4B VL via
`training/data.py::convert_record` + Qwen ChatML template.

**Method:** Static analysis of `convert_record` + `composite_utils`, empirical
substring scan across 50 stratified records (`E_01`), ChatML-template scan with
the real Qwen3VLProcessor (`E_02`), per-topic option-string / question-template
audit (`E_03`), and image_mode × answer / filename × city cross-checks (`E_04`).

---

## 1. Verdict summary

**The dataset-as-trained is NOT compromised by text-channel metadata leaks.**
The only record fields that enter `input_ids` are `record["question"]`,
`record["options"][*]` (values), and `record["answer"]` (assistant target).
Every other field — `sample_id`, `question_id`, `city`, `country`,
`latitude/longitude`, `image_mode`, `difficulty`, `topic`, `mismatch_strategy`,
`mismatch_is_match`, `split`, `city_type`, `benchmark_city_type`,
`streetview_road_bearing`, `satellite_source`, `streetview_source`,
`satellite_date`, `streetview_date`, `query_stv_angle`, `option_arrow_angles`,
and every file path in `images` / `option_stv_paths` /
`mismatch_negative_stv_paths` — is either used to route a code branch or used
to `Image.open(...)` a file whose pixels (not filename) reach the model.

**Model-exploitable text signals that ARE real (Agent D confirmed):**
- B3 — `green_space`, `mismatch_binary_*` have only 2 distinct correct-option
  strings each; `road_surface` collapses 95.0% of correct answers to
  `"Asphalt"`. These priors ARE in the model's text context and learnable.
- B6 — All 10 validation question templates appear verbatim in train for every
  topic. Templates are not a held-out test of phrasing robustness.

**Agent D B1/B2 are NOT model-exploitable text leaks**; they are either
metadata-only reviewer leaks (B1) or pixel-space leaks (B2). Details below.

---

## 2. Field-by-field table

Two channels matter: **text** = substring appears in any `type:"text"` content
part of `messages`, which becomes input_ids after ChatML render. **image-pixels**
= value is a path that gets opened and passed as PIL.Image; pixels go to the
vision encoder, filename does not. **never** = used only to pick a code branch
or never read at all.

| Field | Channel | Audit flag | My verdict | Evidence |
|---|---|---|---|---|
| `sample_id` | never | — | no leak | 0/50 hits in text (`out_E_01.txt`); 0/15 in rendered template (`out_E_02.txt`); no occurrence in training code (`grep` above). |
| `question_id` | never | — | no leak | 0/50 hits; not passed to prompt. Only used in eval result dicts for reporting. |
| `city` | never | D-B2 (pixel) | no text leak | 0/50 hits. City never stringified. (Pixel leak via SV filename style → out of Cat A scope.) |
| `country` | never | — | no leak | 0/50 hits. |
| `latitude` | never | — | no leak | 0/50 hits. |
| `longitude` | never | — | no leak | 0/50 hits. |
| `land_use` | never* | — | no leak | 2/50 substring hits are coincidental (value `"residential"` appears inside an option string like `"...primarily residential"`; see `out_E_01b.txt`). Not read by training code (grep: no `land_use` reference in `training/`). |
| `streetview_count` | never* | — | no leak | 1/50 coincidental (`4` appearing in `"Mid-rise (4-7 floors)"`). Not read by training code. |
| `image_mode` | branch-router | D-B1 | **no text leak**; pixel-level mapping IS the task | 0/50 hits in text. Used only in `get_images_for_question` / `convert_record` to pick rendering branch. See §3 B1 below. |
| `difficulty` | never | — | no leak | 0/50 hits. Only consumed by eval metrics aggregation (`evaluation.py:145`, `eval_base.py:103`). |
| `topic` | never | — | no leak | 0/50 hits. Used only for per-topic accuracy aggregation. |
| `generation_method` | never | — | no leak | 0/50 hits. Unused in `training/`. |
| `mismatch_strategy` | never | — | no leak | 0/50 hits. Unused in `training/`. |
| `mismatch_is_match` | never | — | no leak | 0/50 hits. Unused in `training/`. |
| `mismatch_negative_stv_paths` | image-pixels | D-B1 | pixel only | `data.py:163` opens each path as PIL; filename not serialised. |
| `option_stv_paths` | image-pixels | D-B2 | pixel only | `composite_utils.py:282` opens each path to build composites; filename not serialised. 62% city-match (`out_E_04.txt`) is pixel-only. |
| `option_arrow_angles` | branch-router | — | no leak | `composite_utils.py:250–255` reads keys to compute bearings, draws arrow. Value (e.g. `"along_fwd"`) not stringified into prompt. 0 hits in `out_E_01.txt`. |
| `query_stv_path` | image-pixels | — | no leak | Opened in `composite_utils.py:259`; not stringified. |
| `query_stv_angle` | never | — | no leak | 0/50 hits. Unused in `training/`. |
| `split` | never | — | no leak | 0/50 hits. |
| `city_type` | never | — | no leak | 0/50 hits. |
| `benchmark_city_type` | never | — | no leak | 0/50 hits. |
| `images.satellite` | image-pixels | — | path only (pixels) | `data.py:152,173` + `composite_utils` `Image.open`. |
| `images.satellite_source` (ESRI/NAIP/IGN/S2) | never | — | no leak | 0/50 value-in-text hits (`out_E_01.txt`). Metadata only. |
| `images.satellite_date` | never | — | no leak | 0/50 hits. |
| `images.streetview_along_*` / `cross_*` | image-pixels | — | pixels only | `data.py:158` / `composite_utils.py:268`. |
| `images.streetview_source` | never | — | no leak | 0/50 hits. |
| `images.streetview_road_bearing` | branch-router | — | no leak | `composite_utils.py:249` used to compute arrow geometry for `satellite_arrow` mode. Numeric value not stringified. 0/50 hits. |
| `images.streetview_date` | never | — | no leak | 0/50 hits. |
| `question` | **text** | B6 | reaches model | Written into user message at `data.py:121`. |
| `options[A..D]` values | **text** | B3 | reaches model (genuine prior) | Written into user message at `data.py:121`. |
| `answer` | **text** (target) | B3 (indirect) | reaches model as label | Written into assistant message at `data.py:186`. |

*Coincidental substring overlaps, not reachable via any code-path that serialises
the field. Confirmed in `out_E_01b.txt`.

### ChatML template-level check

`E_02_tokenized_content.py` applies `Qwen3VLProcessor.apply_chat_template` on 15
records (3 per image_mode). Rendered template adds only `<|im_start|>`,
`<|im_end|>`, `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`, and a
`<think>\n\n</think>` reasoning scaffold (Qwen3 native) — no record fields are
injected by the template itself. Only field that "hits" in template scan is
`land_use` (1/15), and it is the same coincidental substring collision as in
`E_01`. See `out_E_02.txt`.

---

## 3. Agent D blocker verdicts

### B1 — "image_mode leaks the answer for mismatch_binary"

**Reclassified: reviewer-only leak, not model-exploitable via text.**

`image_mode` deterministically maps to the Yes/No answer for mismatch_binary:
`streetview_composite → 100% Yes` (1955/1955), `streetview_binary → 100% No`
(2045/2045) in train (`out_E_04.txt`). A reviewer grepping the JSONL can score
100%.

**The model never sees `image_mode` as text** (0/50 hits in `out_E_01.txt`,
0/15 in `out_E_02.txt`). The code uses `image_mode` only to pick a rendering
branch. The pixel-level difference between the two modes is the actual task
(does the SV composite match the marked sat?), so the model solving it via
pixels is correct learning, not cheating.

**Residual risk (pixel side):** I reviewed the rendering pipeline — for both
`streetview_composite` and `streetview_binary`, the sat image goes through the
same `make_sat_marked(sat_path)` with identical default args, the 4 SV
thumbnails through identical `resize_image(..., max_edge)` and
`add_corner_label(..., letter)`. No per-mode style difference that would let
the model shortcut via "red-dot style → answer". The only distinguishing
feature is whether SV pixels match the sat location, which is the intended task
signal.

**Fix needed:**
- Training-code: none.
- Dataset-release: strip `image_mode`, `mismatch_strategy`, `mismatch_is_match`
  from any publicly released JSONL so reviewers cannot grep the answer.

### B2 — "mismatch_mcq_easy correct option is the sole same-city SV"

**Reclassified: pixel-only signal, not a text leak.**

`option_stv_paths` filenames do encode city (62% of correct options come from
the record's own city prefix; `out_E_04.txt`). Filenames are never
stringified into the prompt — `composite_utils.py:282` consumes the paths
directly via `Image.open`. The model sees only the composite PIL image.

Whether the model *can* solve via pixel-level "this SV composite looks like the
same city as the sat" is a Category B question (outside this verification's
scope). Flag it for the pixel/image auditor, not the text auditor.

**Fix needed:**
- Training-code: none.
- Dataset-release: rename option SV paths to anonymous IDs before publishing
  the JSONL (or strip `option_stv_paths` entirely from the public release).
- Dataset-generation v2: for mismatch_mcq_hard, ensure wrong options include
  same-city distractors at parity with correct options to remove the
  pixel-space city shortcut.

### B3 — "green_space closed-set collapse"

**Reconfirmed: real, model-exploitable text leak.** `out_E_03.txt`:

- `green_space`: 2 distinct correct strings, 67%/33% split.
- `mismatch_binary_easy`: 2 distinct strings, 50/50.
- `mismatch_binary_hard`: 2 distinct strings, 52/48.
- `road_surface`: 95% correct = `"Asphalt"`. A text-only model always
  guessing "Asphalt" would hit ~95% on this topic.
- `junction_type`: 60% `"Unsignalized intersection"` modal.
- `road_type`: 59% `"Local residential street"` modal.
- `amenity_richness`: 38% `"Minimal — industrial area..."` modal.
- `urban_density`: 55% `"Low density (suburban/rural)"` modal.

Option text is in the prompt and in the assistant target (via option→letter
mapping). Modal priors are learnable.

**Fix needed:**
- Training-code: none (training MUST see option text).
- Dataset-generation v2: balance correct-option strings per topic. For binary
  topics, either resample to 50/50 or accept the ceiling. For `road_surface`,
  sub-sample "Asphalt" correct cases or drop the topic.

### B4 — "Benchmark has byte-identical image groups with train"

**Out of Category A scope (pixel-space).** Training code never touches
`dataset_content/EODATA_compressed_final/benchmark/*` (grep confirmed: only
`training/TRAINING.md` and `slides.tex` mention "benchmark" and not as a
file-read). The upcoming remote training run on train.jsonl/validation.jsonl
is unaffected by benchmark overlap. Benchmark overlap matters only when you
use the benchmark to claim generalisation; for that, address it before
publishing benchmark numbers.

### B5 — "Combined non-vision baseline ~67%"

**Partially reconfirmed; the ceiling is for a *reviewer with JSONL access*,
not for the model.** The 67% number in `DATASET_ISSUES.md` stacks:
image_mode→binary answer (B1), option-text modal priors (B3), filename-city
(B2), and question-token heuristics. Of these, only B3 is reachable via the
model's text channel. B1 and B2 are only reachable by the model through
pixels (where they're either the intended task or a separate image leak).
The text-only ceiling for a model trained via `convert_record` is roughly the
B3 modal-prior ceiling per topic, not 67%.

**Fix needed:**
- Report separately: (a) reviewer-with-JSONL ceiling (≈67%), (b) text-only
  model ceiling (B3 modal priors), (c) pixel-only model ceiling (B1+B2).
  Do not conflate.

### B6 — "10 templates per topic, train=val templates"

**Reconfirmed: real, model-exploitable.** For every one of 14 topics, all 10
validation templates appear verbatim in train (`out_E_03.txt`). Model can
memorise topic-token → modal-option mapping without learning phrasing
robustness.

**Fix needed:**
- Training-code: none.
- Dataset-generation v2: hold out distinct paraphrases for validation, or
  generate fresh templates at eval time.

---

## 4. Action items

### Must happen before remote launch
**None.** No training-code change is required. The training pipeline does
not leak any metadata field into input_ids. Launch is safe from a text-leak
standpoint.

### Before public benchmark / paper release
1. Strip the following fields from any publicly published JSONL (they are
   reviewer-exploitable even though model-invisible):
   `image_mode`, `mismatch_strategy`, `mismatch_is_match`,
   `mismatch_negative_stv_paths`, `option_stv_paths`, `query_stv_path`,
   `option_arrow_angles`, `sample_id`, `city`, `country`, `latitude`,
   `longitude`, `benchmark_city_type`.
2. Anonymise image filenames on release (or strip paths from JSONL and ship
   a separate resolver).
3. Report text-only, pixel-only, and combined ceilings separately (avoid the
   conflated 67% number).

### Dataset v2 regeneration (not gating the remote run)
1. B3: balance correct-option strings per topic (`road_surface`,
   `green_space`, `mismatch_binary_*`, `amenity_richness`, `urban_density`,
   `road_type`, `junction_type`).
2. B6: hold out paraphrases for validation.
3. B2: for `mismatch_mcq_*`, enforce same-city distractors at parity with
   correct options.
4. B4: de-duplicate benchmark against train at image-hash level.

---

## 5. Residual risks

1. **Pixel-space leaks are not verified by this Category A audit.** B1's
   visual "does sat match SV" is the intended task; B2's "correct SV = same
   city as sat" is a genuine pixel shortcut the model could learn. Verifying
   whether the trained model actually exploits B2 requires GPU inference on a
   same-city vs different-city controlled split — outside scope here.
2. **Rendering-pipeline equivalence between modes is confirmed only by
   code reading**, not pixel-diff. `make_sat_marked` / `resize_image` /
   `add_corner_label` use identical parameters across modes; I saw no
   mode-conditional branching in style. A pixel-diff of
   `make_sat_marked(sat_path)` output between a `streetview_composite` and a
   `streetview_binary` record using the same sat_path would be a fast
   confirmation but was not executed (no image write permitted per "read-only"
   constraint).
3. **Option text prior (B3) IS exploitable.** The model will learn
   topic → modal-option priors. Before quoting final per-topic accuracies,
   compare against the majority-baseline per topic (already computed by
   `callbacks.EvalCallback._compute_topic_baselines`).
4. **Qwen3 `<think>` scaffold:** the chat template renders
   `<think>\n\n</think>\nANSWER` in the assistant turn. With
   `train_on_responses_only` + `response_part="<|im_start|>assistant\n"`, the
   scaffold tokens become part of the loss. This is benign (same tokens every
   sample → quickly learned no-op) but worth monitoring in the first eval if
   `parse_letter` fails on an unexpected prefix. This is a training-pipeline
   question, not a leakage question, so only flagged here.

---

## 6. Appendix — raw outputs

- `audit_scratch/E_01_prompt_content.py` + `out_E_01.txt` — 50-record
  substring scan over every interesting field. Result: only `land_use` (2)
  and `streetview_count` (1) hit, both coincidental in option text.
- `audit_scratch/E_01b_verify_coincidence.py` + `out_E_01b.txt` — confirms
  those hits are substring collisions with option strings, not actual
  serialisation leaks.
- `audit_scratch/E_02_tokenized_content.py` + `out_E_02.txt` — ChatML
  template render via real `Qwen3VLProcessor` over 15 records. Only one
  `land_use` coincidence; template adds no metadata.
- `audit_scratch/E_03_option_and_template.py` + `out_E_03.txt` — per-topic
  distinct correct-option counts (B3) and train↔val template overlap (B6).
  Both fully reconfirmed.
- `audit_scratch/E_04_image_mode_answer.py` + `out_E_04.txt` — image_mode ×
  answer distribution (B1 is reviewer-only, 100% deterministic) and
  option_stv_paths city prefix audit (B2 is pixel-only, 62% same-city).
