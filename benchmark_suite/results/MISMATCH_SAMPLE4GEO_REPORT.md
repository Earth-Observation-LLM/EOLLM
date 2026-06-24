# Mismatch family — Sample4Geo cross-view retrieval baseline (complete report)

Full report for the **cross-view mismatch family** evaluated with **Sample4Geo**, a
street↔satellite **retrieval/embedding** model (Deuser et al., ICCV'23). This is a
*different model class and a different metric* from the Family-1 overhead-VQA results —
Sample4Geo emits image embeddings and is scored by **cosine similarity / retrieval**,
not by answering MCQ letters with an LLM. **These numbers are not comparable to the
Family-1 VQA table and must not be merged into it.**

- **Split:** labeled (`benchmark_with_answers.jsonl`).
- **Topics scored (4):** `mismatch_binary_easy`, `mismatch_binary_hard`,
  `mismatch_mcq_easy`, `mismatch_mcq_hard` — 421 questions each, **n = 1684**.
- **Checkpoints (both reported):** VIGOR (`vigor_cross`, preferred — urban cross-area)
  and CVUSA (`cvusa`).
- **`camera_direction` — deliberately excluded** (5th cross-view topic). Reason in §6.

---

## 1. Headline — per-topic accuracy

| topic | #q | chance | **Sample4Geo VIGOR** | **Sample4Geo CVUSA** |
|---|---:|---:|---:|---:|
| mismatch_binary_easy | 421 | 0.50 | **0.646** | 0.639 |
| mismatch_binary_hard | 421 | 0.50 | **0.646** | 0.622 |
| mismatch_mcq_easy    | 421 | 0.25 | **0.266** | 0.252 |
| mismatch_mcq_hard    | 421 | 0.25 | **0.309** | 0.261 |
| **OVERALL**          | **1684** | — | **0.467** | **0.444** |

VIGOR ≥ CVUSA on every topic — expected: VIGOR is trained on real-world **urban**
panorama↔aerial pairs, the closest domain to EOLLM street↔satellite. CVUSA is
rural/suburban US.

**One-line read:** Sample4Geo *transfers* on the **binary** yes/no (≈0.65 ≫ 0.50
chance) — it really does assign higher street↔satellite similarity to genuinely
co-located pairs, even on unseen cities and a marker overlay it never trained on. It is
**near chance on the 4-way MCQ** (≈0.27–0.31 vs 0.25) — discriminating the *correct*
nearby scene from 3 distractors across the perspective→panorama and street→overhead
gaps is much harder than a single yes/no.

---

## 2. Binary topics — full diagnostics

Binary = one street-set per question vs the satellite tile; predict **match** if
`cosine > τ` (τ calibrated, §5). Because the task is yes/no, accuracy alone hides
*how* the model errs — TPR (catch true matches) vs TNR (reject mismatches):

| ckpt | topic | acc | TPR | TNR |
|---|---|---:|---:|---:|
| VIGOR | binary_easy (cross_city neg) | 0.646 | 0.582 | 0.719 |
| VIGOR | binary_hard (same_city neg)  | 0.646 | 0.575 | 0.702 |
| VIGOR | **all binary** | **0.646** | 0.579 | 0.710 |
| CVUSA | binary_easy | 0.639 | 0.693 | 0.577 |
| CVUSA | binary_hard | 0.622 | 0.672 | 0.583 |
| CVUSA | **all binary** | **0.631** | 0.684 | 0.580 |

- **Both ckpts beat 0.50 on both TPR and TNR** → the similarity signal is genuinely
  informative, not an artifact of the threshold.
- **VIGOR is TNR-leaning** (better at rejecting mismatches), **CVUSA is TPR-leaning**
  (better at catching matches but over-predicts "match"). This follows directly from
  CVUSA producing systematically higher cosines (see §3), so more pairs clear τ.

### Similarity separation (why binary works but isn't great)

| ckpt | match sim (mean ± sd) | mismatch sim (mean ± sd) | separation |
|---|---|---|---:|
| VIGOR | 0.211 ± 0.081 | 0.162 ± 0.066 | **+0.049** |
| CVUSA | 0.274 ± 0.083 | 0.223 ± 0.073 | **+0.051** |

True pairs score ~0.05 higher than mismatched pairs — a real but **small** gap relative
to the ~0.08 spread, so the two distributions overlap heavily → modest (~0.64) binary
accuracy rather than near-perfect. CVUSA's whole distribution is shifted up (~+0.06),
which is why its τ is higher (0.228 vs 0.194).

---

## 3. MCQ topics — full diagnostics

MCQ = 4 street-set options vs the satellite tile; predict **argmax** cosine.

| ckpt | topic | strategy | acc | mean (top1−top2) sim margin |
|---|---|---|---:|---:|
| VIGOR | mcq_easy | cross_city | 0.266 | 0.057 |
| VIGOR | mcq_hard | same_city  | **0.309** | 0.053 |
| CVUSA | mcq_easy | cross_city | 0.252 | 0.047 |
| CVUSA | mcq_hard | same_city  | 0.261 | 0.042 |

- **Tiny win margins (~0.04–0.06)** between the top and 2nd-best option → the model
  barely separates its pick from the runner-up, consistent with near-chance accuracy.
- **Counter-intuitive but real: `hard` (same_city) ≥ `easy` (cross_city).** With
  same-city distractors, the 3 wrong options are *real nearby* scenes that share genuine
  visual structure with the satellite tile, giving the embedding more co-located signal
  to rank on. With cross-city distractors, the wrong options are visually unrelated, so
  the model's similarity ranking is essentially noise → it drifts to chance. (Note the
  topic naming: `easy`/`hard` refer to the *human* difficulty of the distractors, which
  is the inverse of what helps a similarity-only model.)

---

## 4. easy vs hard = negative-sampling strategy (per-strategy view)

`easy` ⇔ **cross_city** negatives; `hard` ⇔ **same_city** negatives. Per-strategy
accuracy (identical to the per-topic table, restated by mechanism):

| ckpt | kind | cross_city (easy) | same_city (hard) |
|---|---|---:|---:|
| VIGOR | binary | 0.646 | 0.646 |
| VIGOR | mcq    | 0.266 | 0.309 |
| CVUSA | binary | 0.639 | 0.622 |
| CVUSA | mcq    | 0.252 | 0.261 |

For **binary**, strategy barely matters (≤0.02) — judging one pair as match/no-match
doesn't depend on where the *negative* came from. For **mcq**, strategy matters
(same_city helps, see §3).

---

## 5. Methodology — every adaptation choice (these affect the numbers)

Sample4Geo is a **weight-shared Siamese ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k_384`,
1024-d output, `num_classes=0`) trained with InfoNCE for street↔satellite matching. It
produces **image embeddings**; there are **no MCQ letters** — we match by **cosine
similarity** and pick the closest candidate. Mapping an MCQ/binary VQA benchmark onto a
retrieval model required these explicit choices:

1. **Query = satellite tile; candidates = street-view sets.** The query is
   `images.satellite` (the original overhead tile — the red dot the VQA prompt draws is
   irrelevant to an embedding model). Candidates are street-view *sets* (4 perspective
   JPGs each). This is genuinely Sample4Geo's native street↔satellite task.
2. **Street aggregation = STITCH to panorama.** Sample4Geo's ground branch expects a
   **360° panorama** (input 384×768, 1:2 aspect). Our street views are **4 separate
   ~90° perspective JPGs** — `along_fwd`(0°), `cross_right`(90°), `along_bwd`(180°),
   `cross_left`(270°) relative to road bearing. We horizontally concatenate them in
   panorama order **fwd → right → bwd → left** (a coherent 0→360° sweep), resize to
   384×768, embed **once**. *(Chosen over mean-of-4-embeddings and max-over-angles —
   it best matches the model's training input modality. This choice materially affects
   the result and is the single biggest adaptation.)*
3. **Satellite preprocessing** = repo `get_transforms_val`: resize 384×384, ImageNet
   normalize, no augmentation.
4. **One backbone, both branches.** Siamese / weight-shared → satellite tile and
   stitched street panorama go through the *same* `model.forward`; embeddings
   L2-normalized before cosine.
5. **MCQ scoring = argmax.** Each option's 4-angle set → 1 stitched panorama → 1
   embedding; `pred = argmax_option cosine(satellite, panorama)`, mapped to its letter.
6. **Binary scoring = calibrated threshold.** One street-set per question (own SV if
   `mismatch_is_match=True`, else `mismatch_negative_stv_paths`);
   `pred = "match" if cosine(satellite, panorama) > τ`. **τ is calibrated**, not picked
   by hand: a deterministic **stratified held-out slice** (~20% of binary Qs, balanced
   by ground truth: n=170, 83 pos / 87 neg) is swept for the threshold maximizing
   **balanced accuracy**; that τ then scores the remaining binary Qs. The match/no-match
   prediction is mapped to whichever A/B option text encodes it (option order is shuffled
   per question).
   - **VIGOR:** τ = **0.1938** (calibration balanced-acc 0.717)
   - **CVUSA:** τ = **0.2278** (calibration balanced-acc 0.642)

> **This is a DOMAIN-TRANSFER (lower-bound) baseline.** Sample4Geo was trained on
> CVUSA / VIGOR (US + urban panorama↔aerial), **not** on EOLLM. Every input is
> out-of-distribution: our cities, our satellite sources (ESRI/NAIP/IGN), the red-dot
> marker (ignored by the embedding but still unseen pixels), and especially the **street
> format** — 4 stitched perspective crops standing in for a true 360° pano. Read these
> as "what an off-the-shelf cross-view retrieval model achieves here," not its home-turf
> number.

---

## 6. Why `camera_direction` is excluded (not skipped silently)

`camera_direction` (421 Qs, `image_mode=satellite_arrow`) is a 5th cross-view topic, and
the natural question is whether Sample4Geo can serve as its baseline too. **It cannot**,
for three independent reasons — so reporting a number would be misleading:

1. **It's an orientation task, not a location task.** The query is one street-view
   image; the 4 candidates are the **same satellite tile** each with a **red arrow** at
   a different compass direction. The answer is the arrow matching the camera's *viewing
   direction*. Sample4Geo's embedding answers *"are these the same place?"* — but here
   **all 4 candidates are the same place** (identical tile). It has **no notion of
   compass direction**, so it cannot rank arrows.
2. **Degenerate by construction.** Embedding 4 near-identical tiles (differing by a
   few-pixel arrow) yields 4 near-identical vectors → argmax is pure noise = chance
   (0.25). Any reported accuracy would measure rounding noise, not capability.
3. **The candidate images don't exist as files.** All 421 records have
   `option_sat_paths = null`; the arrows are rendered on-the-fly by the VQA viewer from
   `option_arrow_angles`, never materialized. There is literally no candidate image for
   an embedding model to encode.

Camera-direction is solvable trivially by a **geometric oracle** (the answer is the
option whose `option_arrow_angles` equals `query_stv_angle` → 100%), and is a meaningful
task for a *VQA* model that can read the arrow + reason about viewing direction — but it
is **outside what a cross-view retrieval embedding model can express.** Excluded for that
reason, documented here for completeness.

---

## 7. Loader, environment, run stats (reproducibility)

| Item | Value |
|---|---|
| Model | Sample4Geo (ICCV'23), ConvNeXt-Base Siamese, InfoNCE; `sample4geo.model.TimmModel` |
| Repo | `github.com/Skyy93/Sample4Geo` |
| Checkpoints | `vigor_cross/…weights_e40_0.6109.pth`, `cvusa/…weights_e40_98.6830.pth` |
| Weights source | Google Drive folder in repo README (1.63 GB zip, 5 ckpts); load `strict=False` → **0 missing / 0 unexpected** keys (exact match) |
| Backend | standalone `remote/eval_sample4geo.py` (retrieval, not the VQA suite contract) |
| Env | `sample4geo` (cloned from `geochat` → Blackwell torch 2.11.0+cu128; + timm 0.9.16, albumentations 2.0.8, opencv-python-headless 4.13) |
| GPU | 1× RTX PRO 6000 Blackwell, 96 GB, sm_120 |
| Image input | sat 384×384; street 4-angle stitch → 384×768 pano; ImageNet norm |
| Decode | none — embeddings only (no LLM, no token generation) |

| Checkpoint | n | unparseable | runtime | peak VRAM |
|---|---:|---:|---:|---:|
| Sample4Geo (VIGOR) | 1684 | 0 | 1.31 min | 0.74 GiB |
| Sample4Geo (CVUSA) | 1684 | 0 | 1.39 min | 0.74 GiB |

Tiny footprint (0.74 GiB peak) and fast — a ~200 MB ConvNeXt embedding model, no LLM.

### Exact commands

```bash
DS=/home/ain480/training/dataset_content/EODATA_compressed_final
PRE=/home/ain480/models_src/Sample4Geo/pretrained/pretrained
P=/home/ain480/.conda/envs/sample4geo/bin

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P/python remote/eval_sample4geo.py \
  --ckpt $PRE/vigor_cross/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.6109.pth \
  --tag vigor_cross --dataset_dir $DS --outdir results_sample4geo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P/python remote/eval_sample4geo.py \
  --ckpt $PRE/cvusa/convnext_base.fb_in22k_ft_in1k_384/weights_e40_98.6830.pth \
  --tag cvusa --dataset_dir $DS --outdir results_sample4geo
```

Raw per-question similarities + predictions:
`results_sample4geo/sample4geo_{vigor_cross,cvusa}_raw.jsonl` (mirrored under
`benchmark_suite/results/sample4geo_mismatch/`).
