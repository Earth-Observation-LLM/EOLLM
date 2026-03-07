# EOLLM Urban VQA

A pipeline for generating a multi-city Visual Question Answering (VQA) dataset focused on urban environments. Each sample pairs satellite and street-level imagery with a multiple-choice question derived from OpenStreetMap metadata. The dataset is designed for training and evaluating Vision-Language Models (VLMs) on urban understanding tasks such as land use classification, building height estimation, road type recognition, urban density assessment, green space detection, and commercial amenity density.

---

## Cities Covered

| Key | City | Country | Satellite Source | OSM Data Quality |
|-----|------|---------|-----------------|-----------------|
| `nyc` | New York City | USA | NAIP (0.6 m) | Excellent — dense building, road, and amenity coverage |
| `paris` | Paris | France | Sentinel-2 (10 m) | Excellent — best `building:levels` coverage in the dataset |
| `london` | London | UK | Sentinel-2 (10 m) | Good — note Canary Wharf has no outdoor Street View |
| `singapore` | Singapore | Singapore | Sentinel-2 (10 m) | Good — weaker `building:levels` tagging |
| `sao_paulo` | São Paulo | Brazil | Sentinel-2 (10 m) | Fair — Overpass bbox sometimes too large; see Known Limitations |
| `amsterdam` | Amsterdam | Netherlands | Sentinel-2 (10 m) | Excellent — strong `building:levels` and landuse tagging |

Each city has 10 seed locations covering a range of urban character types (commercial, residential, industrial, mixed use).

---

## Dataset Structure

The output dataset is stored in `output/dataset.jsonl`. Each line is a self-contained JSON record with the following top-level fields:

```
sample_id         — unique identifier, e.g. "nyc_0001"
location          — lat/lon, city, country, neighborhood, road name, suburb
images            — paths to satellite and street view images + metadata
question          — the MCQ question string
options           — dict {"A": ..., "B": ..., "C": ..., "D": ...}
answer            — correct option key, e.g. "B"
difficulty        — "easy" or "medium"
topic             — question type (see below)
generation_method — always "template" for this pipeline
metadata          — OSM-derived context (buildings, amenities, land use, roads)
validation        — per-sample validity flags and issue list
```

### Images per Sample

| Field | Description |
|-------|-------------|
| `satellite` | `images/sat/{sample_id}.png` — 512x512 px, 250 m buffer |
| `streetview_along_fwd` | `images/sv/{sample_id}_along_fwd.jpg` — 640x640 px, road-bearing heading |
| `streetview_along_bwd` | `images/sv/{sample_id}_along_bwd.jpg` — 640x640 px, opposite heading |
| `streetview_cross_left` | `images/sv/{sample_id}_cross_left.jpg` — 640x640 px, 90 deg left of road |
| `streetview_cross_right` | `images/sv/{sample_id}_cross_right.jpg` — 640x640 px, 90 deg right of road |

Street View images are captured with `source=outdoor`, FOV=90, pitch=-5 deg.

---

## Question / Task Types

Six task types are supported. One question is selected per sample using a scoring heuristic that maximises topic diversity across the dataset.

| Topic | Question | Answer Space |
|-------|----------|-------------|
| `land_use` | What is the primary land use character of this area? | Residential / Commercial & Office / Retail & Shopping / Industrial & Manufacturing / Mixed Use / Public Facilities & Institutions / Open Space & Recreation / Transportation & Infrastructure |
| `building_height` | What building height category best describes the dominant structures? | Low-rise (1-3 floors) / Mid-rise (4-7 floors) / High-rise (8-20 floors) / Skyscraper (20+ floors) |
| `urban_density` | What is the approximate urban density level based on building concentration? | Low density / Moderate density / High density / Very high density |
| `road_type` | What type of road infrastructure is dominant at this location? | Motorway / Trunk / Primary arterial / Secondary / Tertiary / Local residential street |
| `green_space` | Is there publicly accessible green space or parkland near this location? | Yes / No (with descriptive distractors) |
| `amenity_richness` | What is the level of commercial amenity presence in this area? | Minimal / Low / Moderate / High |

All answers are derived deterministically from OSM data. The pipeline uses `random.seed(42)` for option shuffling, so output is reproducible.

---

## Data Sources

| Source | What it provides | Notes |
|--------|-----------------|-------|
| OpenStreetMap (Overpass API) | Road network, buildings, landuse polygons, amenity nodes, parks | Downloaded once per city as two bulk queries; cached locally in `data/{city}_roads_osm.json` and `data/{city}_context_osm.json`. No per-sample API calls. |
| Google Earth Engine | Satellite tiles (NAIP or Sentinel-2) | Requires GEE project authentication. NAIP: US only, 0.6 m, 2019-2025. Sentinel-2: global, 10 m, 2023-2025, max 20% cloud cover. |
| Google Street View Static API | 4 street-level images per sample | Requires API key. Billed per image. |
| Nominatim | Reverse geocoding (suburb, postcode, admin area) | Free, no key required. Rate-limited to 1 req/s. |
| US Census ACS5 (2022) | Population, median household income, housing units | NYC samples only. Requires Census API key. |

---

## Pipeline Steps

```
run_pipeline.py
├── Step 1  01_sample_locations.py   — snap seeds to roads, extract OSM context
├── Step 2  02_fetch_satellite.py    — download satellite tiles via GEE
├── Step 3  03_fetch_streetview.py   — download 4 Street View images per sample
├── Step 4  04_enrich_metadata.py    — Nominatim reverse geocoding + Census (NYC)
├── Step 5  05_generate_questions.py — generate and select MCQ per sample
└── Step 6  06_validate.py          — per-sample and dataset-level validation
```

### Step 1 — Sample Locations

For each city, the pipeline downloads the full road network and OSM context (buildings, amenities, landuse, parks) as two single Overpass queries. The results are cached in `data/`. Each seed point is snapped to the nearest road centerline node. The road bearing at that node is stored and used in all subsequent steps to align Street View headings.

Local OSM context is extracted within a ~200 m radius (0.002 deg bounding box) of each snapped point. This yields building count, median floor count, amenity count and types, dominant landuse tag, and a park presence flag. Land use category is inferred from these signals using a rule-based function.

Output: `output/sample_locations.csv`

### Step 2 — Satellite Imagery

Tiles are fetched via the GEE Python API. NYC uses NAIP (0.6 m true-colour), all other cities use Sentinel-2 SR Harmonized (10 m, RGB bands B4/B3/B2). The pipeline mosaics images within a 250 m buffer around each sample point and exports a 512x512 PNG. Files smaller than 5 KB are flagged as warnings (likely cloud-covered or no-data tiles).

Output: `output/images/sat/{sample_id}.png`

### Step 3 — Street View

Four images are fetched per sample using the road bearing computed in Step 1: forward along road, backward along road, 90 deg left (cross street), and 90 deg right (cross street). A metadata check is performed first; samples with no outdoor coverage are skipped. Images smaller than 5 KB are treated as failed fetches.

Output: `output/images/sv/{sample_id}_{label}.jpg`

### Step 4 — Metadata Enrichment

Nominatim reverse geocoding adds suburb, city district, postcode, and administrative area to each sample. For NYC samples, the US Census FCC block API is used to look up the census tract, which is then queried against the ACS5 dataset for population, median household income, and housing unit count.

Output: `output/metadata_raw.csv`

### Step 5 — Question Generation

All candidate MCQs are generated for a sample based on available OSM data. Each question type is only generated if its underlying data field is present and plausible (e.g., `building_height` requires a valid `osm_median_levels` value). One question is selected per sample using a scoring heuristic that prefers topics underrepresented so far in the dataset and visually distinctive answers (e.g., skyscrapers or industrial areas over generic mid-rise residential). Option order is randomised with `random.seed(42)`.

### Step 6 — Validation

Per-sample checks include: question and answer presence, answer key in options, no duplicate options, satellite image present and non-trivial size, all four Street View images present and non-trivial size, plausible building level values.

Dataset-level checks include: answer distribution balance (no letter exceeds 40% of samples), topic diversity (at least 3 topics), city coverage (at least 2 cities), land use diversity (at least 3 types).

---

## Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `earthengine-api`, `requests`, `pandas`, `numpy`.

### API Keys

Create `api_keys.env` in the project root:

```
GOOGLE_STREETVIEW_KEY=AIza...
CENSUS_API_KEY=...
```

The Census key is optional; omitting it skips Census enrichment for NYC samples. The Street View key is required.

Google Earth Engine requires project authentication. Set the project ID in `src/config.py` (`GEE_PROJECT`) and authenticate via:

```bash
earthengine authenticate
```

### Running the Pipeline

```bash
# All 6 cities, 10 samples each (60 total)
python src/run_pipeline.py

# Specific cities only
python src/run_pipeline.py --cities nyc paris

# Override sample count
python src/run_pipeline.py --cities nyc paris --samples 5

# Run individual steps (standalone)
python src/01_sample_locations.py
python src/02_fetch_satellite.py
# ... etc.
```

Available city keys: `nyc`, `paris`, `london`, `singapore`, `sao_paulo`, `amsterdam`.

Each step can be run in isolation; it reads from the CSV written by the previous step when invoked directly.

---

## Output Format

The final dataset is written to `output/dataset.jsonl`. Each line is a JSON object. Example (condensed):

```json
{
  "sample_id": "nyc_0001",
  "location": {
    "latitude": 40.7580355,
    "longitude": -73.9854818,
    "city": "New York City",
    "country": "USA",
    "neighborhood": "Times Square",
    "road_name": "7th Avenue",
    "geo_suburb": "Manhattan"
  },
  "images": {
    "satellite": "images/sat/nyc_0001.png",
    "satellite_source": "NAIP",
    "satellite_date": "2023-08",
    "streetview_along_fwd": "images/sv/nyc_0001_along_fwd.jpg",
    "streetview_along_bwd": "images/sv/nyc_0001_along_bwd.jpg",
    "streetview_cross_left": "images/sv/nyc_0001_cross_left.jpg",
    "streetview_cross_right": "images/sv/nyc_0001_cross_right.jpg",
    "streetview_source": "Google Street View",
    "streetview_road_bearing": 208.5,
    "streetview_date": "2024-08"
  },
  "question": "What is the primary land use character of this area?",
  "options": {
    "A": "Public Facilities & Institutions",
    "B": "Mixed Use",
    "C": "Industrial & Manufacturing",
    "D": "Commercial & Office"
  },
  "answer": "B",
  "difficulty": "easy",
  "topic": "land_use",
  "generation_method": "template",
  "metadata": {
    "land_use_category": "mixed",
    "land_use_label": "Mixed Use",
    "osm_building_count_200m": 102,
    "osm_median_building_levels": 6,
    "osm_dominant_landuse_raw": "construction",
    "osm_amenity_count_200m": 21,
    "osm_amenity_types": ["parking", "restaurant", "theatre", "pub"],
    "osm_has_park_200m": false,
    "osm_building_types": {"yes": 72, "theatre": 11, "hotel": 8},
    "road_name": "7th Avenue",
    "road_type": "secondary",
    "census_tract": "36061011900",
    "census_population": 1040,
    "census_median_income": 15971
  },
  "validation": {
    "all_images_present": true,
    "satellite_present": true,
    "streetview_count": 4,
    "issues": []
  }
}
```

Image paths in the JSONL are relative to `output/`.

---

## Output Directory Layout

```
output/
├── dataset.jsonl              — final dataset (one record per line)
├── sample_locations.csv       — step 1 output
├── metadata_raw.csv           — step 4 output
└── images/
    ├── sat/
    │   └── {sample_id}.png    — satellite tiles
    └── sv/
        └── {sample_id}_{label}.jpg  — street view images
data/
├── {city}_roads_osm.json      — cached road network per city
└── {city}_context_osm.json    — cached buildings/amenities/landuse per city
```

---

## Known Limitations

- **São Paulo OSM context**: The Overpass bounding box for São Paulo is large enough to occasionally trigger timeout or size limit errors on the public Overpass API. The pipeline retries up to 3 times with backoff. If it still fails, the context file will be missing and affected samples will have no OSM metadata. Planned fix: switch to a Geofabrik `.pbf` extract (see Future Improvements).

- **Sentinel-2 cloud cover**: Some tiles pass the 20% cloud threshold at the mosaic level but still have localised cloud cover over the sample point. These appear as small files or blank-looking images. The pipeline flags files under 5 KB as warnings in the validation output.

- **London Canary Wharf**: This area is a private development and Google has no outdoor Street View coverage. Samples seeded here will have `sv_status != OK` and will be missing all four Street View images.

- **OSM `building:levels` coverage**: Coverage varies significantly by city. Paris and Amsterdam have thorough tagging; São Paulo and Singapore have weaker coverage. Samples without `building:levels` data cannot generate a `building_height` question.

- **`landuse=grass` ambiguity**: In OSM, `grass` is used for traffic medians, road verges, and roundabout centres, not only parks. The `infer_land_use` function guards against misclassification by requiring a building count below 15 before classifying an area as `open_space` when the dominant landuse tag is `grass`, `forest`, or `meadow`.

---

## Future Improvements

- **Replace Overpass with Geofabrik `.pbf` extracts**: Using `pyrosm` or `osmium` to parse pre-downloaded city extracts from [download.geofabrik.de](https://download.geofabrik.de) would eliminate Overpass rate limiting and timeouts entirely, enable fully offline processing, and make the pipeline robust enough to scale to many more cities. This is the highest-priority improvement, particularly for São Paulo.

- **Add more cities**: Any city can be added by appending an entry to the `CITIES` dict in `src/config.py` with a bounding box, satellite source, and a list of seed locations. No other code changes are required.

- **LLM-generated questions**: The pipeline is structured to accept questions from any source. A future step could use a VLM to generate free-form or harder questions that require cross-modal reasoning beyond what OSM tags can express.
