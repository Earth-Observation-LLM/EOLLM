# EOLLM Usage Guide

Everything you need to set up, run, configure, and extend the EOLLM Urban VQA pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Keys](#api-keys)
3. [Installation](#installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Adding a New City](#adding-a-new-city)
6. [Configuring Satellite Images](#configuring-satellite-images)
7. [Enabling / Disabling Question Types](#enabling--disabling-question-types)
8. [Understanding the Output](#understanding-the-output)
9. [Using the Viewers](#using-the-viewers)
10. [Question Types Reference](#question-types-reference)
11. [Cost Estimate](#cost-estimate)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.8+** (tested with 3.10, 3.11, 3.13)
- **pip** (Python package manager)
- **bash** and **curl** (for the OSM data download script)
- **Internet connection** during pipeline execution
- **OS**: Linux or macOS recommended. Windows works with WSL or Git Bash.

---

## API Keys

### 1. Google Earth Engine -- REQUIRED (for US cities or Sentinel-2 fallback)

Downloads satellite imagery via NAIP (US) or Sentinel-2 (global fallback).

**Cost**: Free for research and non-commercial use.

**Setup**:
1. Sign up at [earthengine.google.com](https://earthengine.google.com/)
2. Create or select a Google Cloud project at [console.cloud.google.com](https://console.cloud.google.com/)
3. Enable the Earth Engine API for your project
4. Note your project ID (e.g., `my-project-123456`)
5. Set it in `dataset/src/config.py`:
   ```python
   GEE_PROJECT = "my-project-123456"
   ```
6. Authenticate (one-time):
   ```bash
   earthengine authenticate
   ```

> **Note**: Non-US, non-France cities use ESRI World Imagery (free, no GEE needed) as the primary source. GEE is only used for NAIP (US) and Sentinel-2 (global fallback).

### 2. Google Street View Static API -- REQUIRED

Downloads 4 street-level images per sample.

**Cost**: Paid. ~$7 per 1,000 requests. See [Google Maps pricing](https://developers.google.com/maps/documentation/streetview/usage-and-billing).

**Setup**:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Street View Static API** under APIs & Services > Library
3. Create an API key under APIs & Services > Credentials
4. (Recommended) Restrict the key to Street View Static API only

### 3. US Census API Key -- OPTIONAL

Adds population, median income, and housing units to NYC samples only. Everything else works without it.

**Cost**: Free.

**Setup**: Sign up at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html). Key arrives by email in minutes.

### Where to Put Your Keys

```bash
cp api_keys.env.example api_keys.env
```

Edit `api_keys.env`:
```
GOOGLE_STREETVIEW_KEY=AIzaSyD...your-key-here...
CENSUS_API_KEY=abc123...your-key-here...
```

This file is gitignored -- your keys will never be committed.

### Services That Need No Key

| Service | Purpose | Rate Limit |
|---------|---------|------------|
| OpenStreetMap Overpass API | Road networks, buildings, amenities, land use | ~2 req/min (pipeline handles retries) |
| Nominatim | Reverse geocoding | 1 req/s (pipeline enforces this) |
| IGN Geoplateforme | Satellite tiles for France | No limit documented |
| ESRI World Imagery | Satellite tiles (non-US, non-France) | No limit documented |

---

## Installation

```bash
# 1. Clone
git clone <repository-url>
cd EOLLM

# 2. Install Python dependencies
pip install -r dataset/requirements.txt

# 3. Authenticate Google Earth Engine (one-time)
earthengine authenticate

# 4. Set your GEE project ID
#    Edit dataset/src/config.py: GEE_PROJECT = "your-project-id"

# 5. Create API keys file
cp api_keys.env.example api_keys.env
#    Edit api_keys.env with your keys

# 6. Pre-download OSM data (recommended)
bash dataset/setup_data.sh
```

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `earthengine-api` | Google Earth Engine client (NAIP + Sentinel-2 satellite tiles) |
| `requests` | HTTP requests (Street View, Overpass, Nominatim, Census, ESRI) |
| `pandas` | CSV and dataframe operations |
| `numpy` | Numerical operations |
| `Pillow` | Image processing (composites, arrow overlays, marked tiles) |

---

## Running the Pipeline

### Full Run (All 6 Cities)

```bash
python dataset/src/run_pipeline.py
```

Default: 5 samples per city = 30 total. Each city has up to 10 seed locations.

### Select Specific Cities

```bash
python dataset/src/run_pipeline.py --cities nyc paris amsterdam
```

City keys: `nyc`, `paris`, `london`, `singapore`, `sao_paulo`, `amsterdam`

### Custom Sample Count

```bash
python dataset/src/run_pipeline.py --cities nyc --samples 10
```

Maximum is 10 (number of seed locations per city).

### Run Individual Steps

Each step can run standalone for debugging or re-running after a fix:

```bash
python dataset/src/01_sample_locations.py    # Step 1: snap seeds, extract OSM context
python dataset/src/02_fetch_satellite.py     # Step 2: download satellite tiles
python dataset/src/03_fetch_streetview.py    # Step 3: download street view images
python dataset/src/04_enrich_metadata.py     # Step 4: geocoding + census
python dataset/src/05_generate_questions.py  # Step 6: generate MCQs
python dataset/src/06_validate.py            # Step 7: validate dataset
```

### What Happens During a Run

```
Step 1  Sample Locations
        For each city: download OSM data (or use cache), snap seeds to nearest road,
        extract context (buildings, amenities, land use, water, transit) within buffer.
        --> dataset/output/sample_locations.csv

Step 2  Satellite Fetch
        Per sample: auto-detect source (NAIP/IGN/ESRI), download 512x512 tile,
        validate >5 KB, fall back to next source if needed.
        --> dataset/output/images/sat/{sample_id}.png

Step 3  Street View
        Per sample: check coverage, verify snap <80m, download 4 road-aligned images,
        reject tunnel images via color analysis.
        --> dataset/output/images/sv/{sample_id}_{direction}.jpg

Step 4  Metadata Enrichment
        Per sample: reverse geocode via Nominatim. For NYC: query US Census ACS5.
        --> dataset/output/metadata_raw.csv

Step 5  Composites
        Per sample: create marked satellite (red dot), arrow-annotated variants,
        2x2 street view grids, mega-composites for mismatch MCQ.
        --> dataset/output/images/sat_marked/, sat_arrow/, composite/

Step 6  Question Generation
        Per sample: generate all feasible MCQs (up to 13 types), select best one
        via diversity-aware scoring.

Step 7  Validation
        Per sample: check images, answers, options. Dataset-level: answer distribution,
        topic diversity, city coverage. CRITICAL issues remove samples.
        --> dataset/output/dataset.jsonl
```

### Estimated Time

| Step | Time | Notes |
|------|------|-------|
| Step 1 (locations) | ~1 min | Instant if OSM data is cached |
| Step 2 (satellite) | 2-5 min | Depends on source; ESRI/IGN are faster than GEE |
| Step 3 (street view) | 3-8 min | 4 images/sample + metadata check |
| Step 4 (metadata) | 1-2 min | Nominatim rate limit: 1 req/s |
| Step 5 (composites) | <1 min | Local image processing |
| Step 6 (questions) | <1 min | Local computation |
| Step 7 (validation) | <1 min | Local file checks |
| **Total** | **~10-15 min** | For 60 samples (6 cities x 10) |

---

## Adding a New City

Only one file to edit: `dataset/src/config.py`. No other code changes needed.

### 1. Find the Bounding Box

Go to [bboxfinder.com](http://bboxfinder.com/) or [boundingbox.klokantech.com](https://boundingbox.klokantech.com/). Draw a rectangle around your city. Format: `(south, west, north, east)` in decimal degrees.

**Tip**: Keep the box focused on the urban core. Large bboxes can timeout on Overpass.

### 2. Choose Seed Locations

Pick 5-10 diverse neighborhoods. Each seed is a tuple:

```python
("Neighborhood Name", latitude, longitude, "urban_character")
```

- `urban_character`: `"residential"`, `"commercial"`, `"industrial"`, or `"mixed"` (informational only)
- Get coordinates by right-clicking in Google Maps
- Mix commercial centers, residential areas, industrial zones, waterfronts

### 3. Add the Entry

```python
# In dataset/src/config.py, add to the CITIES dict:
"tokyo": {
    "name": "Tokyo",
    "country": "Japan",
    "bbox": (35.550, 139.550, 35.820, 139.930),
    "seeds": [
        ("Shibuya",    35.6580, 139.7016, "commercial"),
        ("Shinjuku",   35.6938, 139.7034, "commercial"),
        ("Asakusa",    35.7148, 139.7967, "mixed"),
        ("Akihabara",  35.7023, 139.7745, "commercial"),
        ("Setagaya",   35.6461, 139.6532, "residential"),
        # ... up to 10 seeds
    ],
},
```

### 4. Run

```bash
bash dataset/setup_data.sh                      # downloads OSM for new city
python dataset/src/run_pipeline.py --cities tokyo --samples 5
```

**Satellite source is auto-detected:**
- US coordinates -> NAIP (0.6 m/px)
- France coordinates -> IGN (0.2 m/px)
- Everything else -> ESRI (0.3-0.5 m/px)
- All fall back to Sentinel-2 (10 m/px) if primary fails

---

## Configuring Satellite Images

In `dataset/src/config.py`:

```python
# Buffer radius in meters -- controls how much area each tile covers
SAT_BUFFER_M = {
    "NAIP": 100,   # ~200m tile at 0.6 m/px -- building-level detail
    "IGN":  100,   # ~200m tile at 0.2 m/px -- extremely detailed
    "ESRI": 100,   # ~200m tile at 0.3-0.5 m/px
    "S2":   1280,  # ~2560m tile at 10 m/px -- can't zoom further
}

# Output image dimensions
SAT_IMAGE_PX = 512
```

### Buffer Guide

| Buffer | Tile Coverage | Best For |
|--------|--------------|----------|
| 50 m | ~100 x 100 m | Individual buildings, street corners |
| 100 m | ~200 x 200 m | Block-level detail (current default) |
| 250 m | ~500 x 500 m | Neighborhood context |
| 500 m | ~1 x 1 km | District overview |

**Sentinel-2 note**: 10 m native resolution means a 100 m buffer gives only ~20 real pixels across. The 1280 m default is the practical minimum for useful imagery.

---

## Enabling / Disabling Question Types

In `dataset/src/config.py`:

```python
ENABLED_QUESTION_TYPES = {
    "land_use",
    "building_height",
    "urban_density",
    "road_type",
    "road_surface",
    "junction_type",
    "water_proximity",
    "green_space",
    "amenity_richness",
    "transit_density",
    "camera_direction",
    "mismatch_binary",
    "mismatch_mcq",
}
```

Comment out or remove any type to skip it. The code for all 13 types is always present -- this set just controls which ones are generated.

### Geolocation Difficulty Strategies

```python
# "same_city": distractors from same city (harder)
# "cross_city": distractors from different cities (easier)
# "both": generates one variant of each
MISMATCH_MCQ_STRATEGY = "both"
MISMATCH_BINARY_STRATEGY = "both"
```

---

## Understanding the Output

### Directory Layout

```
dataset/output/
  dataset.jsonl                             Final dataset (1 JSON per line)
  sample_locations.csv                      Location + OSM context for all samples
  metadata_raw.csv                          Enriched with geocoding + census
  images/
    sat/           {id}.png                 512x512 satellite tiles
    sat_marked/    {id}.png                 Satellite with red center dot
    sat_arrow/     {id}_arrow_{dir}.png     Satellite with bearing-aligned arrow
    sv/            {id}_{label}.jpg         640x640 street view (4 per sample)
    composite/     {id}_4angles.png         2x2 grid of 4 SV angles
                   {id}_4angles_labeled.png Same with direction labels
                   {id}_4stv_*.png          Mega-composite for mismatch MCQ
```

### JSONL Record Structure

Each line in `dataset.jsonl`:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique ID (`nyc_0001`, `paris_0003`, etc.) |
| `location` | object | lat, lon, city, country, neighborhood, road_name, geo_suburb, geo_display_name |
| `images` | object | Relative paths to all images + source/date metadata |
| `question` | string | The selected MCQ question text |
| `options` | object | `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `answer` | string | Correct option key (`A`, `B`, `C`, or `D`) |
| `topic` | string | Question type (e.g., `land_use`, `camera_direction`) |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `question_count` | int | Total number of feasible questions for this sample |
| `questions` | array | All generated questions (not just the selected one) |
| `metadata` | object | Full OSM context, road info, census data, composite paths |
| `validation` | object | Image presence flags, issue list |

Image paths are relative to `dataset/output/`.

### Loading the Dataset in Python

```python
import json

samples = []
with open("dataset/output/dataset.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))

# Access fields
for s in samples:
    print(f"{s['sample_id']}: {s['topic']} -- {s['question']}")
    print(f"  Answer: {s['answer']} = {s['options'][s['answer']]}")
    print(f"  All questions: {s['question_count']}")
```

### Loading for ML Training

```python
import json
from PIL import Image

with open("dataset/output/dataset.jsonl") as f:
    samples = [json.loads(line) for line in f]

for s in samples:
    sat = Image.open(f"dataset/output/{s['images']['satellite']}")
    sv_fwd = Image.open(f"dataset/output/{s['images']['streetview_along_fwd']}")

    # Use all questions, not just the selected one
    for q in s["questions"]:
        question_text = q["question"]
        options = q["options"]
        answer = q["answer"]
        topic = q["topic"]
        # ... feed to your model
```

---

## Using the Viewers

### Main Dataset Viewer

```bash
bash dataset/viewer/serve.sh
```

Opens `http://localhost:8765/viewer/` in your browser.

**Features:**
- Load `dataset.jsonl` via drag-and-drop or file picker
- Filter by city, topic, difficulty, validation status
- Per-sample cards with satellite + street view thumbnails
- Click to expand: full images, question, options, metadata inspector
- Keyboard navigation (arrow keys, Escape to close)

### Geolocation Viewer

```bash
bash dataset/viewer/serve_geo.sh          # default port 8080
bash dataset/viewer/serve_geo.sh 9090     # custom port
```

Opens `http://localhost:8080/geo.html`.

**Features:**
- Dedicated view for camera_direction, mismatch_binary, mismatch_mcq
- Arrow-annotated satellite images side-by-side
- Street view 2x2 composites for option comparison
- Full-screen image inspection

### Manual Server Start

If the scripts don't work, start manually from the dataset directory:

```bash
cd dataset
python3 -m http.server 8765
# Open http://localhost:8765/viewer/ in your browser
```

---

## Question Types Reference

### How Questions Are Generated

1. **Data check**: Each question type requires specific OSM tags or metadata to exist
2. **Answer derivation**: The correct answer is computed from the data (e.g., building count -> density bin)
3. **Distractor generation**: 3 wrong options are selected from the remaining valid categories
4. **Template selection**: A random paraphrase is chosen from 10 options per topic (in `dataset/src/question_templates.py`)
5. **Scoring**: All feasible questions are scored; the best one is selected for topic diversity

### OSM Metadata Questions

| # | Topic | Source Data | When Generated | Answer Categories |
|---|-------|-----------|----------------|-------------------|
| 1 | `land_use` | OSM landuse/building tags | Valid category inferred | 8 land use types |
| 2 | `building_height` | `building:levels` tag | Median levels > 0 | 4 height ranges |
| 3 | `urban_density` | Building count in buffer | Count is a number | 4 density bins |
| 4 | `road_type` | `highway` tag on snapped road | Valid highway type | 6 road types |
| 5 | `green_space` | `leisure=park` in buffer | Park presence known | Yes / No |
| 6 | `amenity_richness` | Amenity count in buffer | Count is a number | 4 richness levels |
| 7 | `road_surface` | `surface` tag on nearby roads | Tag exists | 4 surface types |
| 8 | `junction_type` | Road intersections + signals | 2+ roads meeting | 4 junction types |
| 9 | `water_proximity` | Water features in buffer | Distance computed | 3 proximity bins |
| 10 | `transit_density` | Bus/train/tram stops in 300 m | Count available | 4 density bins |

### Cross-View Alignment Questions

| # | Topic | Source Data | When Generated | Answer Categories |
|---|-------|-----------|----------------|-------------------|
| 11 | `camera_direction` | Road bearing + arrow images | Valid bearing exists | 4 quadrant options |
| 12 | `mismatch_binary` | SV composite + marked satellite | Distractor sample available | Yes / No |
| 13 | `mismatch_mcq` | 4 SV composites + marked satellite | 3 distractors available | A / B / C / D |

### Selection Scoring

When multiple questions are feasible, the pipeline picks the best one:
- **Topic rarity bonus**: Topics used less in the dataset score higher
- **Visual distinctiveness bonus**: Extreme values score higher (skyscrapers > mid-rise, roundabouts > regular intersections)
- **One "best" question selected per sample**, but **all questions** are stored in the `questions` array

---

## Cost Estimate

### Per-Sample API Costs

| API | Calls per Sample | Billable? |
|-----|-----------------|-----------|
| Street View metadata | 1 | Yes (~$0.007) |
| Street View images | 4 | Yes (~$0.007 each) |
| Earth Engine (NAIP/S2) | 0-2 | No (free with project) |
| Nominatim geocode | 1 | No |
| Overpass (OSM) | 0 (cached) | No |
| Census (NYC only) | 0-2 | No |
| IGN / ESRI tiles | 0-1 | No |

**Total per sample: ~$0.035** (Street View only)

### Per-Run Costs

| Run Configuration | Samples | SV Calls | Est. Cost |
|-------------------|---------|----------|-----------|
| 1 city, 3 samples | 3 | 15 | ~$0.11 |
| 1 city, 10 samples | 10 | 50 | ~$0.35 |
| 6 cities, 5 samples | 30 | 150 | ~$1.05 |
| 6 cities, 10 samples | 60 | 300 | ~$2.10 |

Costs depend on your Google Cloud pricing tier. The $7/1,000 rate is the standard pay-as-you-go price.

---

## Troubleshooting

### GEE Authentication Errors

```
Error: Please authorize access to your Earth Engine account
```

Run `earthengine authenticate` and follow the browser prompts. Verify `GEE_PROJECT` in `dataset/src/config.py` matches your Cloud project ID.

### Overpass API Timeout (Sao Paulo)

```
ERROR: failed to download sao_paulo context after 3 attempts
```

The Sao Paulo bbox is large. Wait a few minutes and re-run `bash dataset/setup_data.sh` (it skips already-downloaded files). Try during off-peak hours (European night -- Overpass servers are in Germany).

### Street View Images Are Black / Tunnel-Like

The pipeline has built-in tunnel detection (color-channel spread analysis) and rejects snaps >80 m away. If dark images persist for a specific seed, move the seed coordinates slightly to a surface road.

### Satellite Tiles Are Tiny (< 5 KB)

Auto-rejected. The fallback chain tries the next source (e.g., ESRI -> S2). If all sources fail, the sample is flagged with a validation warning.

### "No road found within snap radius"

The seed coordinates are too far from any OSM-mapped road. Check the location on [openstreetmap.org](https://www.openstreetmap.org) and move the seed closer to a road.

### Missing Census Data for NYC

Census enrichment is optional. Get a free key from [census.gov](https://api.census.gov/data/key_signup.html) and add it to `api_keys.env`. Without it, NYC samples simply lack demographic fields.

### Rate Limiting (HTTP 429)

```
Rate limited (attempt 2), waiting 60s...
```

Handled automatically. The pipeline retries with backoff. Pre-downloading OSM with `dataset/setup_data.sh` avoids this during pipeline runs.

### Pipeline Crashes Mid-Run

Steps are idempotent. Re-run the pipeline or the specific step -- cached data (OSM files, already-downloaded images) will be reused.

### Viewer Shows No Data

Make sure `dataset/output/dataset.jsonl` exists (run the pipeline first). The viewer loads data via file upload -- drag the JSONL file onto the viewer page or use the file picker.
