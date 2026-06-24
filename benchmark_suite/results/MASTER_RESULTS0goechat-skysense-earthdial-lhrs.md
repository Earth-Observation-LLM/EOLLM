# EOLLM — Master Results (Family-1 overhead urban-attribute benchmark)

Consolidated reference for all models evaluated on the **9 Family-1 overhead topics**
(`land_use`, `building_height`, `urban_density`, `road_type`, `road_surface`,
`junction_type`, `green_space`, `amenity_richness`, `transit_density`), 4-way MCQ,
`image_mode == satellite_marked`, **`sat_only`** (single marked satellite tile).

Two model groups, both scored on the **same shared benchmark, satellite-only**:

1. **External zero-shot RS-VLMs** — off-the-shelf remote-sensing VLMs, never trained
   on EOLLM. Labeled split, **n = 3135**, identical item set + marked-sat tiles.
2. **Our trained Qwen3.5-9B** — fine-tuned on EOLLM; its `sat` (marked-satellite-only)
   mode is the like-for-like column (single overhead view, same input as the externals).

---

## 1. External zero-shot RS-VLMs — per-topic accuracy (n=3135)

| topic | #q | GeoChat-7B | SkySenseGPT-7B | LHRS-Bot-Nova | EarthDial-4B-RGB |
|---|---:|---:|---:|---:|---:|
| land_use | 421 | 0.487 | 0.534 | 0.463 | 0.475 |
| building_height | 190 | 0.305 | 0.347 | 0.289 | 0.326 |
| urban_density | 421 | 0.135 | 0.145 | 0.178 | **0.304** |
| road_type | 421 | 0.107 | 0.181 | 0.261 | **0.461** |
| road_surface | 303 | 0.756 | **0.802** | 0.587 | 0.584 |
| junction_type | 334 | 0.084 | 0.308 | 0.353 | **0.527** |
| green_space | 203 | 0.507 | 0.537 | 0.399 | **0.626** |
| amenity_richness | 421 | 0.135 | 0.126 | 0.178 | **0.254** |
| transit_density | 421 | 0.214 | 0.261 | **0.342** | 0.297 |
| **OVERALL Family-1** | **3135** | **0.278** | **0.334** | **0.329** | **0.413** |

**Ranking (overall):** EarthDial-4B-RGB **0.413** > SkySenseGPT-7B 0.334 >
LHRS-Bot-Nova 0.329 > GeoChat-7B 0.278.

EarthDial leads on 5 of 9 topics — concentrated in **layout/structure** (road_type,
junction_type, green_space, urban_density) where its high-res InternVL tiling
(≤6×448px + thumbnail) resolves geometry the 504px single-crop LLaVA models miss.
The LLaVA models keep **road_surface** (pure texture, view-agnostic). All collapse
toward chance (0.25) on **amenity_richness** — counting services from one overhead
tile is genuinely hard.

---

## 2. Zero-shot vs. our trained `sat` — like-for-like (single satellite view)

Trained columns: Qwen3.5-9B `sat` mode on the same benchmark — `pc sat` (per-city)
and `su sat` (seen/unseen). Values ×100.

| task | GeoChat | SkySenseGPT | LHRS-Bot | EarthDial | **our pc sat** | **our su sat** |
|---|---:|---:|---:|---:|---:|---:|
| Land use | 48.7 | 53.4 | 46.3 | 47.5 | **75.5** | **72.0** |
| Building height | 30.5 | 34.7 | 28.9 | 32.6 | **67.4** | **63.7** |
| Urban density | 13.5 | 14.5 | 17.8 | 30.4 | **61.8** | **71.5** |
| Road type | 10.7 | 18.1 | 26.1 | 46.1 | **72.4** | **69.6** |
| Road surface | 75.6 | 80.2 | 58.7 | 58.4 | **95.0** | **95.0** |
| Junction type | 8.4 | 30.8 | 35.3 | 52.7 | **73.4** | **74.0** |
| Green space | 50.7 | 53.7 | 39.9 | 62.6 | **100.0** | **99.0** |
| Amenity richness | 13.5 | 12.6 | 17.8 | 25.4 | **53.4** | **53.0** |
| Transit density | 21.4 | 26.1 | 34.2 | 29.7 | **43.7** | **45.8** |
| **OVERALL (9)** | **27.8** | **33.4** | **32.9** | **41.3** | **54.9** | **54.9** |

### Gap: best zero-shot vs. our trained `sat` (per-city)

| task | best zero-shot | our trained sat | **gap** |
|---|---:|---:|---:|
| Land use | 53.4 | 75.5 | **+22.1** |
| Building height | 34.7 | 67.4 | **+32.7** |
| Urban density | 30.4 | 61.8 | **+31.4** |
| Road type | 46.1 | 72.4 | **+26.3** |
| Road surface | 80.2 | 95.0 | **+14.8** |
| Junction type | 52.7 | 73.4 | **+20.7** |
| Green space | 62.6 | 100.0 | **+37.4** |
| Amenity richness | 25.4 | 53.4 | **+28.0** |
| Transit density | 34.2 | 43.7 | **+9.5** |
| **OVERALL** | **41.3** (EarthDial) | **54.9** | **+13.6** |

**Takeaway:** even the strongest zero-shot RS-VLM (EarthDial) trails our trained
`sat` by **13.6 pts overall**, and by 20–37 pts on most individual tasks. The gap is
smallest where a generic cue carries the day (road_surface texture +14.8,
transit_density +9.5) and largest where EOLLM-specific taxonomy grounding matters
(green_space +37.4, building_height +32.7, urban_density +31.4). Fine-tuning on EOLLM
buys a large, consistent margin over off-the-shelf RS models on the same single
satellite view.

---

## ⚠️ Contamination caveat (LHRS-Bot-Nova & EarthDial)

These two zero-shot scores are an **upper bound** on visual skill, not a clean
measurement: both share OSM provenance with our labels. **LHRS-Bot-Nova** was
pretrained on ~1M OSM-aligned RS image–text pairs (thousands of cities); **EarthDial**'s
instruction QA is itself OSM-derived. Our Family-1 labels are OSM-grounded, so high
scores may reflect shared OSM priors (and possible image overlap), not reading our
tiles. GeoChat/SkySenseGPT do not share this provenance — so the fact that they trail
EarthDial does not by itself prove EarthDial is "more capable," only that it is closer
to our label provenance and uses higher-res tiling.

---

## Loaders, checkpoints, environments

All routed through the standalone `benchmark_suite` (shared MCQ prompt, shared
marked-sat tile rendering, shared letter parsing) → question text + image
byte-identical across models; only the forward pass differs. Single GPU: **RTX PRO
6000 Blackwell, 96 GB, sm_120**. Batched, `max_new_tokens=8` (one-letter answer →
throughput is batching + near-zero decode, not multi-GPU).

| Model | Checkpoint | Backend | LLM / vision | Preproc | Env (transformers) |
|---|---|---|---|---|---|
| GeoChat-7B | `MBZUAI/geochat-7B` | `llava_native` | Vicuna-7B / CLIP-L | 504px single | geochat (4.31) |
| SkySenseGPT-7B | `ll-13/SkySenseGPT-7B-clip-lora` | `llava_native` | Vicuna-7B / CLIP-L | 504px single | geochat (4.31) |
| LHRS-Bot-Nova | `LHRS/LHRS-Bot-Nova` (Stage3) | `lhrs_native` | Llama-3-8B / SigLIP-384 + MoE perceiver | 384px single | lhrs (4.42.4) |
| EarthDial-4B-RGB | `akshaydudhane/EarthDial_4B_RGB` | `earthdial_native` | Phi-3-Mini / InternViT-300M | InternVL ≤6×448px + thumb | earthdial (4.37.2) |

### Run stats

| Model | n | unparseable | runtime | throughput | peak VRAM |
|---|---:|---:|---:|---:|---:|
| GeoChat-7B | 3135 | 0 (0.00%) | 8.7 min | 6.0 Q/s | 40.2 GiB |
| SkySenseGPT-7B | 3135 | 0 (0.00%) | 8.7 min | 6.0 Q/s | 40.2 GiB |
| LHRS-Bot-Nova | 3135 | 0 (0.00%) | 2.4 min | 21.5 Q/s | 22.0 GiB |
| EarthDial-4B-RGB | 3135 | 1 (0.03%) | 1.9 min | 27.5 Q/s | 13.6 GiB |

### Reproducibility notes

- **Blackwell sm_120 / no flash-attn.** LHRS-Bot `text_modal.py` patched
  `use_flash_attention_2=True` → `attn_implementation="sdpa"` (no accuracy impact).
  EarthDial's InternViT degrades to eager natively; Phi-3 doesn't force FA2.
- **LHRS-Bot base LLM** = `NousResearch/Meta-Llama-3-8B-Instruct` (public,
  byte-identical mirror of the gated Meta repo: LlamaForCausalLM/4096/32/128256) —
  the exact base LHRS-Bot trained on, just ungated. Not a model substitution.
- **No topics skipped**; all 9 topics × 3135 items, single marked-sat tile each
  (`n_images=1` verified). Unparseable counted as wrong, logged separately.

### Exact commands

```bash
# LHRS-Bot-Nova
DATASET_DIR=.../EODATA_compressed_final PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ~/.conda/envs/lhrs/bin/python run.py --config remote/config.lhrs.yaml --outdir results

# EarthDial-4B-RGB
DATASET_DIR=.../EODATA_compressed_final PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/.conda/envs/earthdial/bin/python run.py --config remote/config.earthdial.yaml --outdir results
```

**Detail report:** `FAMILY1_SATONLY_REPORT.md` (same directory).

---

# 3. Mismatch family (cross-view retrieval) — Sample4Geo baseline

> **Different family, different model class, different metric.** Sections 1–2 cover
> the 9 overhead **VQA** topics with MCQ-answering VLMs. This section covers the **4
> mismatch topics** (cross-view street↔satellite *matching*) with **Sample4Geo**, a
> retrieval/embedding model — **not** a VQA model. These numbers are **not
> comparable** to the Family-1 table above and must never be merged into it.

**Topics (labeled split, 421 questions each, n = 1684 total):**
`mismatch_binary_easy`, `mismatch_binary_hard`, `mismatch_mcq_easy`, `mismatch_mcq_hard`.

## 3.1 Per-topic accuracy

| topic | #q | chance | Sample4Geo (VIGOR) | Sample4Geo (CVUSA) |
|---|---:|---:|---:|---:|
| mismatch_binary_easy | 421 | 0.50 | **0.646** | 0.639 |
| mismatch_binary_hard | 421 | 0.50 | **0.646** | 0.622 |
| mismatch_mcq_easy | 421 | 0.25 | **0.266** | 0.252 |
| mismatch_mcq_hard | 421 | 0.25 | **0.309** | 0.261 |
| **OVERALL** | **1684** | — | **0.467** | **0.444** |

VIGOR (cross-area) ≥ CVUSA on every topic — expected, since VIGOR is trained on
real-world *urban* panorama↔aerial pairs, the closest domain to EOLLM street↔satellite.

## 3.2 Reading the result

- **Binary (real signal).** True-match street↔sat cosines are measurably higher than
  mismatched ones (VIGOR: TPR≈0.58, TNR≈0.71; CVUSA skews toward predicting "match":
  TPR≈0.68, TNR≈0.58). Both clearly beat the 0.50 coin-flip → Sample4Geo *transfers*:
  it assigns higher similarity to genuinely co-located street/satellite pairs even on
  cities and a marker overlay it never trained on.
- **MCQ (near chance).** Picking the correct 1-of-4 street set by argmax similarity is
  only ~0.27–0.31 (chance 0.25). Discriminating the *right* nearby urban scene from 3
  plausible distractors — across a 90°-perspective→panorama domain gap and a
  street↔overhead view gap — is much harder than the binary yes/no.
- **easy vs hard.** Differ only by negative-sampling strategy (`same_city` vs
  `cross_city`); the gap is small because cross-view cosine barely depends on whether a
  distractor is from the same city, for a model that never learned these cities.

## 3.3 ⚠️ This is a DOMAIN-TRANSFER baseline (read before citing)

Sample4Geo was trained on **CVUSA / VIGOR** (US + urban panorama↔aerial), **not** on
EOLLM imagery. Treat these numbers as a **lower-bound, off-the-shelf transfer baseline**,
not the model's home-turf performance. Every input differs from its training
distribution: our cities, our satellite source (ESRI/NAIP/IGN tiles), the red-dot
marker overlay (ignored by the embedding but still pixels it never saw), and — most
importantly — the **street input format** (see 3.4).

## 3.4 Methodology — what we had to adapt (these choices affect the number)

Sample4Geo is a **weight-shared Siamese ConvNeXt-Base** (`convnext_base.fb_in22k_ft_in1k_384`,
1024-d embedding, `num_classes=0`) trained with InfoNCE for street↔satellite matching.
It emits **image embeddings**; there are **no MCQ letters** — we match by **cosine
similarity** and pick the closest candidate. Adapting an MCQ/binary VQA benchmark to a
retrieval model required several explicit choices:

1. **Query = satellite, candidates = street-view.** Per mismatch question the query is
   `images.satellite` (the original overhead tile — the red dot the VQA prompt draws is
   irrelevant to an embedding model); candidates are street-view *sets* (4 perspective
   JPGs each). This is genuinely Sample4Geo's native street↔satellite task.
2. **Street aggregation = STITCH-to-panorama.** Sample4Geo's ground branch expects a
   **360° panorama** (input 384×768, a 1:2 aspect). Our street views are **4 separate
   ~90° perspective JPGs** (`along_fwd`=0°, `cross_right`=90°, `along_bwd`=180°,
   `cross_left`=270° relative to road bearing). We horizontally concatenate them in
   panorama order **fwd → right → bwd → left** (a coherent 0→360° sweep), then resize to
   384×768 and embed **once**. *(Chosen over mean-of-4-embeddings and max-over-angles;
   it best matches the model's training input modality.)*
3. **Satellite preprocessing.** `images.satellite` → resize 384×384, ImageNet
   normalize — the repo's `get_transforms_val` (Resize + Normalize, no augmentation).
4. **One backbone for both branches.** Sample4Geo is Siamese (weight-shared), so the
   satellite tile and the stitched street panorama go through the *same* `model.forward`.
   Embeddings L2-normalized before cosine.
5. **MCQ scoring = argmax.** Each of the 4 option street-sets → 1 stitched panorama → 1
   embedding; `pred = argmax_option cosine(satellite, panorama)`, mapped to its letter.
6. **Binary scoring = calibrated threshold.** One street-set per question (the
   question's own SV if `mismatch_is_match=True`, else `mismatch_negative_stv_paths`);
   `pred = "match" if cosine(satellite, panorama) > τ`. **τ is calibrated**, not guessed:
   a deterministic **stratified held-out slice** (~20% of binary Qs, balanced by gt:
   n=170, 83 pos / 87 neg) is swept for the threshold maximizing **balanced accuracy**;
   the chosen τ then scores the remaining binary Qs. The "match"/"no-match" prediction is
   mapped to whichever A/B option text encodes it (option text is shuffled per question).
   - **VIGOR:** τ = **0.1938** (calib balanced-acc 0.717)
   - **CVUSA:** τ = **0.2278** (calib balanced-acc 0.642)

## 3.5 Loader, checkpoints, environment, run stats

| Item | Value |
|---|---|
| Model | Sample4Geo (Deuser et al., ICCV'23), ConvNeXt-Base Siamese, InfoNCE |
| Repo | `github.com/Skyy93/Sample4Geo` (`sample4geo.model.TimmModel`) |
| Checkpoints | `vigor_cross/…weights_e40_0.6109.pth` (preferred, urban cross-area) and `cvusa/…weights_e40_98.6830.pth` — both reported |
| Weights source | Google Drive folder in repo README (1.63 GB zip, 5 ckpts); VIGOR + CVUSA, load `strict=False` → **0 missing / 0 unexpected** keys |
| Backend | standalone `remote/eval_sample4geo.py` (retrieval, not the VQA suite contract) |
| Env | `sample4geo` (cloned from `geochat` → Blackwell torch 2.11.0+cu128; + timm 0.9.16, albumentations 2.0.8, opencv-headless 4.13) |
| Image input | sat 384×384; street 4-angle stitch → 384×768 pano; ImageNet norm |
| Decode | none — embeddings only (no LLM, no token generation) |

| Checkpoint | n | unparseable | runtime | peak VRAM |
|---|---:|---:|---:|---:|
| Sample4Geo (VIGOR) | 1684 | 0 | 1.31 min | 0.74 GiB |
| Sample4Geo (CVUSA) | 1684 | 0 | 1.39 min | 0.74 GiB |

Tiny VRAM (0.74 GiB) and fast — a 200 MB-class ConvNeXt embedding model, no LLM.

### Exact commands

```bash
# Sample4Geo — both checkpoints, all 4 mismatch topics, labeled split
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

Raw per-question similarities + predictions saved to
`results_sample4geo/sample4geo_{vigor_cross,cvusa}_raw.jsonl`.
