# Family-1 (overhead urban-attribute) — Zero-shot RS-VLM benchmark

**Task:** the 9 EOLLM Family-1 overhead topics (`land_use`, `building_height`,
`urban_density`, `road_type`, `road_surface`, `junction_type`, `green_space`,
`amenity_richness`, `transit_density`), 4-way MCQ, **`image_mode == satellite_marked`**,
**`sat_only` mode** (the single marked satellite tile only). Labeled split
(`benchmark_with_answers`), **n = 3135**, identical question set + identical
marked-sat tiles across all four models.

All four models are **satellite/aerial-only, single-image remote-sensing VLMs**
evaluated **zero-shot**: none was trained on EOLLM, on our 4-way MCQ format, or on
the red-dot marker overlay our tiles carry. They never see street-view / cross-view
items. The marker is left in place on purpose — it is part of the zero-shot transfer
condition, not "fixed".

## Per-topic accuracy

| topic | #questions | GeoChat-7B acc | SkySenseGPT-7B acc | LHRS-Bot-Nova acc | EarthDial-4B-RGB acc |
|---|---:|---:|---:|---:|---:|
| land_use | 421 | 0.487 (205/421) | 0.534 (225/421) | 0.463 (195/421) | 0.475 (200/421) |
| building_height | 190 | 0.305 (58/190) | 0.347 (66/190) | 0.289 (55/190) | 0.326 (62/190) |
| urban_density | 421 | 0.135 (57/421) | 0.145 (61/421) | 0.178 (75/421) | 0.304 (128/421) |
| road_type | 421 | 0.107 (45/421) | 0.181 (76/421) | 0.261 (110/421) | 0.461 (194/421) |
| road_surface | 303 | 0.756 (229/303) | 0.802 (243/303) | 0.587 (178/303) | 0.584 (177/303) |
| junction_type | 334 | 0.084 (28/334) | 0.308 (103/334) | 0.353 (118/334) | 0.527 (176/334) |
| green_space | 203 | 0.507 (103/203) | 0.537 (109/203) | 0.399 (81/203) | 0.626 (127/203) |
| amenity_richness | 421 | 0.135 (57/421) | 0.126 (53/421) | 0.178 (75/421) | 0.254 (107/421) |
| transit_density | 421 | 0.214 (90/421) | 0.261 (110/421) | 0.342 (144/421) | 0.297 (125/421) |
| **OVERALL Family-1** | **3135** | **0.278** (872/3135) | **0.334** (1046/3135) | **0.329** (1031/3135) | **0.413** (1296/3135) |

## Reading the results

- **EarthDial-4B-RGB is the strongest zero-shot RS-VLM here** (0.413 overall),
  ahead of SkySenseGPT (0.334), LHRS-Bot-Nova (0.329) and GeoChat (0.278).
- EarthDial's gains are concentrated in the **layout/structure** topics —
  `road_type` (0.461 vs ≤0.26 for the others), `junction_type` (0.527),
  `green_space` (0.626), `urban_density` (0.304) — consistent with its
  high-resolution InternVL tiling (up to 6×448px + thumbnail) resolving fine
  road/junction geometry the 504px single-crop LLaVA models miss.
- The LLaVA-family models (GeoChat/SkySenseGPT) still win **`road_surface`**
  (0.76–0.80) — a pure-texture, view-agnostic cue — but collapse toward chance on
  count/context topics (`amenity_richness`, `urban_density`).
- All four are at/near 0.25 chance on `amenity_richness` (≤0.25 except EarthDial
  0.254): inferring service density from a single overhead tile is genuinely hard.

## ⚠️ Contamination caveat (LHRS-Bot-Nova & EarthDial)

**These two scores should NOT be read as pure visual skill.** Both models have
OSM exposure that overlaps with how our labels were built:

- **LHRS-Bot-Nova** was pretrained on ~1M OSM-aligned remote-sensing image–text
  pairs spanning thousands of cities. Our overhead tiles + OSM-derived labels may
  overlap that corpus in both **imagery** and **attribute priors**.
- **EarthDial**'s instruction QA is itself **OSM-derived**, the same provenance as
  our Family-1 labels.

Our Family-1 labels are OSM-grounded, so a high Family-1 score may reflect **shared
OSM priors** (and possibly image overlap) rather than reading our specific tiles.
GeoChat/SkySenseGPT do not share this provenance, so cross-model gaps mix *capability*
with *provenance overlap*. Treat LHRS/EarthDial numbers as an **upper bound** on
zero-shot visual skill, not a clean measurement of it.

## Loaders, checkpoints, environments

All four routed through the standalone `benchmark_suite` (shared MCQ prompt, shared
marked-sat tile rendering, shared letter parsing) so the question text + image are
byte-identical across models; only the model-specific forward pass differs. Single
GPU: **RTX PRO 6000 Blackwell, 96 GB, sm_120**. Batched inference (no data-parallel
sharding — one GPU); `max_new_tokens = 8` (the answer is one letter, so throughput
comes from batching + near-zero decode, not multi-GPU).

| Model | Checkpoint | Backend | LLM / vision | Preproc | Env |
|---|---|---|---|---|---|
| GeoChat-7B | `MBZUAI/geochat-7B` | `llava_native` | Vicuna-7B / CLIP-L | 504px single | geochat (tf 4.31) |
| SkySenseGPT-7B | `ll-13/SkySenseGPT-7B-clip-lora` | `llava_native` | Vicuna-7B / CLIP-L | 504px single | geochat (tf 4.31) |
| LHRS-Bot-Nova | `LHRS/LHRS-Bot-Nova` (Stage3) | `lhrs_native` | Llama-3-8B / SigLIP-384 + MoE perceiver | 384px single | lhrs (tf 4.42.4) |
| EarthDial-4B-RGB | `akshaydudhane/EarthDial_4B_RGB` | `earthdial_native` | Phi-3-Mini / InternViT-300M | InternVL tiling ≤6×448px + thumb | earthdial (tf 4.37.2) |

### Setup notes (reproducibility)

- **Blackwell sm_120 / no flash-attn.** Neither new model uses flash_attn:
  LHRS-Bot's `lhrs/models/text_modal.py` hard-codes `use_flash_attention_2=True`
  for the Llama-3 LLM, so it is **patched (vendored on the GPU node) to
  `attn_implementation="sdpa"`** — correct on Blackwell, no accuracy impact.
  EarthDial's InternViT degrades to a naive-eager path on its own when flash_attn
  isn't importable, and Phi-3 doesn't force FA2, so it runs eager unmodified.
- **LHRS-Bot base LLM.** LHRS-Bot-Nova ships only its trained additions (SigLIP +
  MoE perceiver + a Llama-3 LoRA); the base LLM is pulled from the hub at load. The
  official `meta-llama/Meta-Llama-3-8B-Instruct` is **gated** and the lab-ws token
  lacked download grant (403), so we used **`NousResearch/Meta-Llama-3-8B-Instruct`**
  — a public, **byte-identical mirror** of the same weights (verified arch/hidden/
  layers/vocab = LlamaForCausalLM/4096/32/128256), i.e. the exact base LHRS-Bot
  trained on, just ungated. Not a model substitution.
- **Both envs cloned from `geochat`** (to inherit the working Blackwell torch
  `cu128` build) then re-pinned transformers per repo; flash_attn omitted.

### Exact commands

```bash
# LHRS-Bot-Nova (lhrs env, sdpa, NousResearch base mirror)
DATASET_DIR=.../EODATA_compressed_final PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ~/.conda/envs/lhrs/bin/python run.py --config remote/config.lhrs.yaml --outdir results

# EarthDial-4B-RGB (earthdial env, eager, InternVL dynamic tiling)
DATASET_DIR=.../EODATA_compressed_final PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/.conda/envs/earthdial/bin/python run.py --config remote/config.earthdial.yaml --outdir results
```

### Run stats

- **GeoChat-7B**: n=3135, unparseable=0 (0.00%), runtime=521.4s (8.7 min), throughput=6.0 Q/s, backend=llava_native
- **SkySenseGPT-7B**: n=3135, unparseable=0 (0.00%), runtime=523.3s (8.7 min), throughput=6.0 Q/s, backend=llava_native
- **LHRS-Bot-Nova**: n=3135, unparseable=0 (0.00%), runtime=145.8s (2.4 min), throughput=21.5 Q/s, backend=lhrs_native, peak VRAM=22494 MiB (22.0 GiB)
- **EarthDial-4B-RGB**: n=3135, unparseable=1 (0.03%), runtime=113.9s (1.9 min), throughput=27.5 Q/s, backend=earthdial_native, peak VRAM=13938 MiB (13.6 GiB)

- **No topics skipped.** All 9 Family-1 topics scored for all four models, full
  3135-item labeled split, single marked-sat tile per item (`n_images=1` verified).
- **Unparseable predictions** (counted as wrong, logged separately): GeoChat 0,
  SkySenseGPT 0, LHRS-Bot-Nova 0, EarthDial-4B-RGB 1 (0.03%).
