# EOLLM Urban VQA

A pipeline for generating a multi-city Visual Question Answering (VQA) dataset focused on urban environments. Each sample pairs satellite and street-level imagery with a multiple-choice question derived from OpenStreetMap metadata and cross-view spatial reasoning. Designed for training and evaluating Vision-Language Models (VLMs) on urban understanding tasks.

For detailed setup instructions and usage guide, see [USAGE.md](USAGE.md).

---

## Cities Covered

| Key | City | Country | Satellite Source | OSM Data Quality |
|-----|------|---------|-----------------|-----------------|
| `nyc` | New York City | USA | NAIP (0.6 m) | Excellent -- dense building, road, and amenity coverage |
| `paris` | Paris | France | Sentinel-2 (10 m) | Excellent -- best `building:levels` coverage in the dataset |
| `london` | London | UK | Sentinel-2 (10 m) | Good -- note Canary Wharf has no outdoor Street View |
| `singapore` | Singapore | Singapore | Sentinel-2 (10 m) | Good -- weaker `building:levels` tagging |
| `sao_paulo` | Sao Paulo | Brazil | Sentinel-2 (10 m) | Fair -- Overpass bbox sometimes too large; see Known Limitations |
| `amsterdam` | Amsterdam | Netherlands | Sentinel-2 (10 m) | Excellent -- strong `building:levels` and landuse tagging |

Each city has 10 seed locations covering a range of urban character types (commercial, residential, industrial, mixed use). New cities can be added by editing `src/config.py` -- see [USAGE.md](USAGE.md#adding-a-new-city).

---

## Question / Task Types

13 question types across three categories. One question is selected per sample using a scoring heuristic that maximises topic diversity across the dataset.

### OSM Metadata Questions (10 types)

| Topic | Question | Answer Space | Difficulty |
|-------|----------|-------------|------------|
| `land_use` | What is the primary land use character of this area? | Residential / Commercial & Office / Retail & Shopping / Industrial & Manufacturing / Mixed Use / Institutional / Open Space / Transport | easy |
| `building_height` | What building height category best describes the dominant structures? | Low-rise (1-3) / Mid-rise (4-7) / High-rise (8-20) / Skyscraper (20+) | easy |
| `urban_density` | What is the approximate urban density level? | Low / Moderate / High / Very high | medium |
| `road_type` | What type of road infrastructure is dominant? | Motorway / Trunk / Primary / Secondary / Tertiary / Residential | easy |
| `green_space` | Is there publicly accessible green space nearby? | Yes / No (with descriptive distractors) | easy |
| `amenity_richness` | What is the level of commercial amenity presence? | Minimal / Low / Moderate / High | medium |
| `road_surface` | What is the surface type of the main road? | Asphalt / Cobblestone/Sett / Unpaved/Gravel / Concrete | easy |
| `junction_type` | What type of road junction is shown? | Roundabout / Signalized / Unsignalized / Grade-separated | medium |
| `water_proximity` | Is there a water body near this location? | Adjacent (<50m) / Nearby (50-200m) / None (>200m) | medium |
| `transit_density` | What level of transit stop density exists? | None / Low (1-2) / Moderate (3-5) / High (6+) | medium |

### Cross-View Alignment Tasks (3 types)

These tasks test spatial reasoning between satellite and street-level views. They do not rely on OSM tags -- they use coordinate geometry, road bearing, and image composition.

| Topic | Question | Answer Space | Difficulty |
|-------|----------|-------------|------------|
| `camera_direction` | Which direction is the street view camera pointing relative to the satellite image? | Top-Left / Top-Right / Bottom-Left / Bottom-Right | medium |
| `mismatch_binary` | Does this street view correspond to this satellite image? | Yes / No (with distractors) | medium |
| `mismatch_mcq` | Which of four satellite tiles matches the street view? | Top-Left / Top-Right / Bottom-Left / Bottom-Right | hard |

All answers are derived deterministically from OSM data or coordinate geometry. Option shuffling uses a per-sample RNG seed for reproducibility.

---

## Data Sources

| Source | What It Provides | Notes |
|--------|-----------------|-------|
| OpenStreetMap (Overpass API) | Roads, buildings, landuse, amenities, parks, traffic signals, water features, transit stops | Downloaded once per city; cached in `data/`. No per-sample API calls. |
| Google Earth Engine | Satellite tiles (NAIP or Sentinel-2) | Requires GEE project. NAIP: US only, 0.6 m. Sentinel-2: global, 10 m. |
| Google Street View Static API | 4 street-level images per sample | Requires API key. Billed per image. |
| Nominatim | Reverse geocoding (suburb, postcode, admin area) | Free, no key required. Rate-limited to 1 req/s. |
| US Census ACS5 (2022) | Population, median income, housing units | NYC samples only. Free API key. |

---

## Pipeline Overview

```
run_pipeline.py
 |-- Step 1  01_sample_locations.py    -- snap seeds to roads, extract OSM context
 |-- Step 2  02_fetch_satellite.py     -- download satellite tiles via GEE
 |-- Step 3  03_fetch_streetview.py    -- download 4 Street View images per sample
 |-- Step 4  04_enrich_metadata.py     -- Nominatim geocoding + Census (NYC)
 |-- Step C  07_generate_composites.py -- create 2x2 satellite composites for mismatch tasks
 |-- Step 5  05_generate_questions.py  -- generate and select MCQ per sample
 '-- Step 6  06_validate.py           -- per-sample and dataset-level validation
```

Each step's `run()` function takes a list of sample dicts and returns an enriched list. The orchestrator threads them together and writes `output/dataset.jsonl` at the end.

---

## Output

The final dataset is written to `output/dataset.jsonl`. Each line is a self-contained JSON record:

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
    "streetview_date": "2024-08",
    "composite_4sat": "images/composite/nyc_0001_4sat.png"
  },
  "question": "What is the primary land use character of this area?",
  "options": {"A": "Institutional", "B": "Mixed Use", "C": "Industrial", "D": "Commercial & Office"},
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
    "osm_road_surface": "asphalt",
    "osm_junction_type": "signalized",
    "osm_water_distance_m": 1250.5,
    "osm_transit_stop_count": 8,
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

### Output Directory Layout

```
output/
 |-- dataset.jsonl                   -- final dataset (one record per line)
 |-- sample_locations.csv            -- step 1 output
 |-- metadata_raw.csv                -- step 4 output
 '-- images/
     |-- sat/
     |   '-- {sample_id}.png         -- 512x512 satellite tiles
     |-- sv/
     |   '-- {sample_id}_{label}.jpg -- 640x640 street view images (4 per sample)
     '-- composite/
         '-- {sample_id}_4sat.png    -- 1024x1024 composite tiles (for mismatch MCQ)
data/
 |-- {city}_roads_osm.json           -- cached OSM road network per city
 '-- {city}_context_osm.json         -- cached OSM context per city
```

---

## Known Limitations

- **Sao Paulo OSM context**: The Overpass bounding box is large and may trigger timeouts. The pipeline retries 3 times with backoff. If it fails, context data will be missing for affected samples.
- **Sentinel-2 cloud cover**: Some tiles pass the 20% cloud threshold but have localised cloud cover. Files under 5 KB are flagged as warnings.
- **London Canary Wharf**: No outdoor Street View coverage. All 4 SV images will be missing.
- **OSM `building:levels`**: Coverage varies by city. Paris and Amsterdam have strong tagging; Sao Paulo and Singapore are sparse. No `building_height` question is generated when this data is missing.
- **OSM `surface` tag**: Road surface tagging is inconsistent globally. The `road_surface` question is only generated when the tag explicitly exists.
- **`landuse=grass` ambiguity**: The `infer_land_use` function guards against misclassification by requiring a building count below 15 before classifying an area as `open_space`.

---

## Quick Start

```bash
pip install -r requirements.txt
earthengine authenticate
cp api_keys.env.example api_keys.env   # fill in your keys
bash setup_data.sh                      # pre-download OSM data
python src/run_pipeline.py --cities amsterdam --samples 3
```

For the full setup guide, see [USAGE.md](USAGE.md).
