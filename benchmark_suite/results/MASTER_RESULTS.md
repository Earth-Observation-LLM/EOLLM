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
