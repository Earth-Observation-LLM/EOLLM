# EOLLM Task Taxonomy

A reference description of every task (topic) in the EOLLM urban-VQA dataset, written
so it can be compared against external benchmarks (e.g. other satellite/street-view VQA
or geolocalisation datasets).

- **Total released corpus:** 39,155 question records (per-city train + val + benchmark).
- **Topics:** 14, grouped into 3 families.
- **Format:** all tasks are multiple-choice. Urban-understanding, `camera_direction`, and
  `mismatch_mcq_*` are 4-way (A–D); `mismatch_binary_*` is 2-way (yes/no match).
- **Answers are deterministic** — derived from OpenStreetMap tags, not human annotation
  (`generation_method = template` for the entire release).
- **Source of truth:** counts below were taken directly from
  `dataset_content/EODATA_compressed_final/` and cross-checked against
  `EOLLM_Report_LaTeX/latex/main.tex`.

---

## Summary table

| Topic | Family | Visual (image_mode) | # options | Difficulty | Records | Ability tested |
|---|---|---|:--:|:--:|---:|---|
| `land_use` | Urban | satellite + red dot | 4 | easy | 2,982 | Dominant urban morphology from overhead |
| `building_height` | Urban | satellite + red dot | 4 | easy | 1,882 | Built-form scale from texture & shadow |
| `urban_density` | Urban | satellite + red dot | 4 | medium | 2,982 | Density estimation in a fixed radius |
| `road_type` | Urban | satellite + red dot | 4 | easy | 2,982 | Road class from carriageway width / surroundings |
| `road_surface` | Urban | satellite + red dot | 4 | easy | 2,701 | Fine-grained material recognition (cobble vs. asphalt) |
| `junction_type` | Urban | satellite + red dot | 4 | medium | 2,782 | Intersection topology (roundabout, signalised…) |
| `green_space` | Urban | satellite + red dot | 4 | easy | 1,970 | Vegetation segmentation & proximity reasoning |
| `amenity_richness` | Urban | satellite + red dot | 4 | medium | 2,982 | Service availability from urban texture |
| `transit_density` | Urban | satellite + red dot | 4 | medium | 2,982 | Counting public-transit infrastructure |
| `camera_direction` | Cross-view | satellite + arrow/option + 1 SV | 4 | medium | 2,982 | Align a ground view with an overhead arrow |
| `mismatch_binary_easy` | Mismatch | 4-angle SV grid | 2 | easy | 2,982 | Same-location verification, cross-city distractor |
| `mismatch_binary_hard` | Mismatch | 4-angle SV grid | 2 | hard | 2,982 | Same-location verification, same-city distractor |
| `mismatch_mcq_easy` | Mismatch | 4×4 mega-grid | 4 | easy | 2,982 | Pick-the-match among 4 candidates, cross-city |
| `mismatch_mcq_hard` | Mismatch | 4×4 mega-grid | 4 | hard | 2,982 | Pick-the-match, intra-city distractors (**hardest**) |

> Ten "balanced" topics sit at exactly **2,982** records. The four under-represented
> topics (`building_height`, `green_space`, `road_surface`, `junction_type`) are limited
> by sparse OSM tag coverage (e.g. `building:levels` is sparse in São Paulo and Singapore).

---

## Family 1 — Urban understanding (9 topics)

**Common setup:** a single overhead satellite tile with a **red dot** marking the sample
point (`image_mode = satellite_marked`). The model reasons about what is at that location
from the overhead view alone. Answers come from OSM tags within a fixed radius, so the
task is purely about reading the satellite imagery.

- **`land_use`** — Identify the dominant land use (residential, commercial, industrial,
  etc.) of the marked area. Tests reading large-scale urban morphology from above.
- **`building_height`** — Estimate the scale of the built form (number of storeys / height
  band). Tests reasoning from roof texture and shadow length. *Under-represented:* depends
  on `building:levels` tags.
- **`urban_density`** — Estimate how built-up the area is within a fixed radius. A counting
  / density-estimation task.
- **`road_type`** — Classify the road class at the dot (motorway, primary, residential…)
  from carriageway width and surroundings.
- **`road_surface`** — Fine-grained material recognition: asphalt, cobblestone, unpaved,
  concrete. Hard to do from satellite alone, which makes it discriminative.
- **`junction_type`** — Recognise the intersection topology (roundabout, signalised
  crossing, T-junction, etc.). Derived from way-intersection tags.
- **`green_space`** — Reason about vegetation and proximity to parks / leisure polygons /
  water bodies. A segmentation-plus-proximity task.
- **`amenity_richness`** — Infer how many amenities (shops, services) are nearby from the
  visual texture of the area.
- **`transit_density`** — Count public-transit infrastructure (stops) within ~300 m of the
  marked point.

---

## Family 2 — Cross-view orientation (1 topic)

- **`camera_direction`** — The satellite tile shows **four labelled arrows**, one per MCQ
  option (`image_mode = satellite_arrow`), plus **one** street-view image of the query
  sample. The model must decide which arrow direction the ground-level camera is facing.
  This is the only task that jointly uses an overhead view *and* a single ground view, and
  it directly probes cross-view geometric alignment.
  - Option arrows are stored as **angle names** (`along_fwd`, `cross_left`, …) in
    `option_arrow_angles`; `composite_utils.py` converts them to bearings.
  - The shown street view is in `query_stv_path` / `query_stv_angle`.

---

## Family 3 — Mismatch / geolocalisation core (4 topics)

The most discriminative subset of the dataset. All mismatch tasks are built around
matching ground-level street views to a location.

**Easy vs. hard is about the distractor source:**

- **Easy** → distractor drawn from **another city** (`mismatch_strategy = cross_city`).
  Ambient cues — vegetation, sky, signage language, vehicle types — often leak the answer.
- **Hard** → distractor drawn from the **same city** (`mismatch_strategy = same_city`).
  Those shortcuts are defeated, isolating the geometric match between the layouts.

### Binary verification (2-way)
- **`mismatch_binary_easy`** — Shown a 2×2 grid of four street-view angles
  (`image_mode = streetview_binary`); answer whether they are the same location as the
  reference, with a **cross-city** distractor. Ground truth in `mismatch_is_match`.
- **`mismatch_binary_hard`** — Same task, but the distractor is **same-city**, so ambient
  cues no longer help. Non-match records carry `mismatch_negative_stv_paths` (the 4 SV
  angles of the non-matching location).

### MCQ pick-the-match (4-way)
- **`mismatch_mcq_easy`** — A **4×4 mega-grid** (`image_mode = streetview_mega`): four
  candidate locations × four angles each. Pick which candidate matches; distractors are
  **cross-city**. Option → 4 SV paths in `option_stv_paths`.
- **`mismatch_mcq_hard`** — Same mega-grid, but all distractors are **intra-city**. This is
  the **hardest task in the corpus** and the strongest test of pure visual geolocalisation.

---

## How difficulty maps to topics

| Difficulty | Topics |
|---|---|
| `easy` | `land_use`, `building_height`, `road_type`, `road_surface`, `green_space`, `mismatch_binary_easy`, `mismatch_mcq_easy` |
| `medium` | `urban_density`, `junction_type`, `amenity_richness`, `transit_density`, `camera_direction` |
| `hard` | `mismatch_binary_hard`, `mismatch_mcq_hard` |

Difficulty is a fixed per-topic label (not per-question): every record of a topic carries
the same difficulty.

---

## image_mode → what the model actually sees

| `image_mode` | Visual rendered | Used by |
|---|---|---|
| `satellite_marked` | Satellite tile + red dot at sample centre | all 9 urban-understanding topics |
| `satellite_arrow` | Satellite tile + a labelled arrow per MCQ option (+ 1 SV) | `camera_direction` |
| `streetview_binary` | 2×2 grid of the non-matching sample's 4 SV angles | `mismatch_binary_easy`, `mismatch_binary_hard` |
| `streetview_mega` | 4×4 mega-grid: 4 candidate locations × 4 angles | `mismatch_mcq_easy`, `mismatch_mcq_hard` |
| `streetview_composite` | 2×2 grid of the query sample's 4 SV angles | (composite helper, used in match cases) |

> Note: `streetview_mega` assembles **2 images** end-to-end for the model (not 16) — the
> 16 SV tiles are composited into the grid before being passed in.

---

## Per-record JSONL fields (for cross-dataset comparison)

| Field | Type | Notes |
|---|---|---|
| `question_id` | string | Unique per (sample, topic, variant) |
| `sample_id` | string | Location ID, e.g. `amsterdam_0017` |
| `city`, `country` | string | Place metadata |
| `latitude`, `longitude` | float | WGS-84 of the sample point |
| `question` | string | MCQ question text |
| `options` | dict | `{A,B,C,D} → text` (or `{A,B}` for binary) |
| `answer` | string | Letter; `null` in `benchmark_public` |
| `topic` | string | One of the 14 topics |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `image_mode` | string | Which visual to assemble (see above) |
| `images` | dict | Paths + provenance for the raw images |
| `split` | string | `train` / `validation` / `benchmark` |
| `city_type` / `benchmark_city_type` | string | `seen` / `unseen` (split-strategy dependent) |
| `generation_method` | string | Always `template` in this release |
| **Topic-specific:** | | |
| `query_stv_path`, `query_stv_angle` | string | The single SV shown for `camera_direction` |
| `option_arrow_angles` | dict | option → arrow angle name (`camera_direction`) |
| `option_stv_paths` | dict | option → list of 4 SV paths (`mismatch_mcq`) |
| `mismatch_strategy` | string | `same_city` / `cross_city` |
| `mismatch_is_match` | bool | Binary mismatch ground truth |
| `mismatch_negative_stv_paths` | list | 4 SV paths of the non-matching location |

---

## At-a-glance comparison axes (vs. other datasets)

- **Modality coverage:** overhead-only (9 urban topics), overhead+ground (`camera_direction`),
  ground-only (4 mismatch topics).
- **Reasoning type:** classification/recognition (urban), counting/density (`urban_density`,
  `transit_density`, `amenity_richness`), cross-view geometry (`camera_direction`),
  retrieval/verification (mismatch family).
- **Answer provenance:** fully deterministic from OSM (no human labels) → reproducible,
  but bounded by OSM tag coverage and correctness.
- **Difficulty control:** explicit, via distractor source (same-city vs. cross-city) for
  the mismatch family — a built-in shortcut-vs-no-shortcut contrast.
