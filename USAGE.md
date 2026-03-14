# EOLLM Usage Guide

Complete guide to setting up and using the EOLLM Urban VQA dataset pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Keys -- What You Need and Where to Get Them](#api-keys)
3. [Installation](#installation)
4. [Pre-downloading OSM Data](#pre-downloading-osm-data)
5. [Running the Pipeline](#running-the-pipeline)
6. [Adding a New City](#adding-a-new-city)
7. [Configuring Satellite Images](#configuring-satellite-images)
8. [Understanding the Output](#understanding-the-output)
9. [Using the Dataset Viewer](#using-the-dataset-viewer)
10. [Question Types Reference](#question-types-reference)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.8+** (tested with 3.10, 3.11, 3.13)
- **pip** (Python package manager)
- **curl** and **bash** (for the OSM data download script)
- **Internet connection** (required for API calls during pipeline execution)
- **Operating system**: Linux or macOS recommended. Windows works but `setup_data.sh` requires WSL or Git Bash.

---

## API Keys

The pipeline uses several external services. Here is exactly what you need, whether it costs money, and where to get each key.

### 1. Google Earth Engine (GEE) -- REQUIRED

**What it does**: Downloads satellite imagery (NAIP for US cities, Sentinel-2 for everywhere else).

**Cost**: Free for research and non-commercial use.

**How to set up**:

1. Go to [earthengine.google.com](https://earthengine.google.com/) and sign up with your Google account
2. Create a Google Cloud project (or use an existing one) at [console.cloud.google.com](https://console.cloud.google.com/)
3. Enable the Earth Engine API for your project
4. Note your **project ID** (e.g., `my-project-123456`)
5. Open `src/config.py` and set your project ID:
   ```python
   GEE_PROJECT = "my-project-123456"
   ```
6. Authenticate on your machine (one-time):
   ```bash
   earthengine authenticate
   ```
   This opens a browser window. Follow the prompts to authorize access.

### 2. Google Street View Static API -- REQUIRED

**What it does**: Downloads 4 street-level images per sample (forward, backward, left, right along the road).

**Cost**: Paid. Google charges per image request. Check [Google Maps Platform pricing](https://developers.google.com/maps/documentation/streetview/usage-and-billing) for current rates. At default settings (60 samples x 4 images = 240 requests), a single full pipeline run costs a few dollars.

**How to get the key**:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use the same one as GEE)
3. Go to **APIs & Services > Library**
4. Search for and enable **Street View Static API**
5. Go to **APIs & Services > Credentials**
6. Click **Create Credentials > API Key**
7. Copy the key (starts with `AIza...`)
8. (Recommended) Restrict the key to only the Street View Static API under **API restrictions**

### 3. US Census API Key -- OPTIONAL

**What it does**: Adds population, median household income, and housing unit count to NYC samples only. If omitted, NYC samples simply won't have Census demographics -- everything else works fine.

**Cost**: Free.

**How to get the key**:

1. Go to [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Fill in your name and email
3. You'll receive a key by email within minutes

### Where to Put Your Keys

Create a file called `api_keys.env` in the project root directory. You can copy the example file as a starting point:

```bash
cp api_keys.env.example api_keys.env
```

Then edit `api_keys.env` with your actual keys:

```
GOOGLE_STREETVIEW_KEY=AIzaSyD...your-key-here...
CENSUS_API_KEY=abc123...your-key-here...
```

**Important**: `api_keys.env` is gitignored -- your keys will never be committed.

### Services That Do NOT Need a Key

| Service | Purpose | Rate Limit |
|---------|---------|------------|
| OpenStreetMap Overpass API | Downloads road networks, buildings, amenities | Public, ~2 req/min. The pipeline handles retries. |
| Nominatim | Reverse geocoding (suburb, postcode) | 1 request/second. The pipeline enforces this. |

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd EOLLM
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `earthengine-api` | Google Earth Engine Python client for satellite imagery |
| `requests` | HTTP requests (Street View, Census, Nominatim, Overpass) |
| `pandas` | Data manipulation and CSV I/O |
| `numpy` | Numerical operations |
| `Pillow` | Image processing (composite tile generation) |

### Step 3: Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Follow the browser prompts. This stores credentials locally and only needs to be done once per machine.

### Step 4: Set Your GEE Project ID

Edit `src/config.py` and update:

```python
GEE_PROJECT = "your-project-id-here"
```

### Step 5: Create API Keys File

```bash
cp api_keys.env.example api_keys.env
```

Edit `api_keys.env` and fill in your keys (see [API Keys](#api-keys) above).

### Step 6: Pre-download OSM Data (Recommended)

```bash
bash setup_data.sh
```

This downloads all OSM road networks and context data for all 6 cities. It takes a few minutes due to Overpass API rate limits. Running this before the pipeline ensures Step 1 doesn't need to wait for downloads.

If you skip this step, the pipeline will download the data automatically on first run -- but it will be slower.

---

## Pre-downloading OSM Data

The `setup_data.sh` script downloads 12 files from the Overpass API:

| File | Contents | Typical Size |
|------|----------|-------------|
| `data/{city}_roads_osm.json` | Road network with geometry (for snapping seed points) | 5-50 MB |
| `data/{city}_context_osm.json` | Buildings, amenities, landuse, parks, traffic signals, water, transit stops | 10-100 MB |

**Features**:
- **Idempotent**: Skips files that already exist. Safe to re-run.
- **Retries**: Handles Overpass rate limiting (HTTP 429) with 60s backoff, up to 3 attempts.
- **Delays**: Waits between cities to respect Overpass fair use policy.

**When to re-run**: If you update the OSM query template in `src/config.py` (e.g., after adding new Overpass clauses), delete the cached context files and re-download:

```bash
rm data/*_context_osm.json
bash setup_data.sh
```

---

## Running the Pipeline

### Full Pipeline (All Cities)

```bash
python src/run_pipeline.py
```

This processes all 6 cities with 10 samples each (60 total). Output goes to `output/`.

### Specific Cities

```bash
python src/run_pipeline.py --cities nyc paris amsterdam
```

Available city keys: `nyc`, `paris`, `london`, `singapore`, `sao_paulo`, `amsterdam`

### Custom Sample Count

```bash
python src/run_pipeline.py --cities nyc --samples 5
```

Each city has up to 10 seed locations. Setting `--samples` higher than 10 has no effect.

### Running Individual Steps

Each step can run standalone. It reads from the CSV output of the previous step:

```bash
python src/01_sample_locations.py    # generates output/sample_locations.csv
python src/02_fetch_satellite.py     # reads sample_locations.csv, downloads sat images
python src/03_fetch_streetview.py    # reads sample_locations.csv, downloads SV images
python src/04_enrich_metadata.py     # reads sample_locations.csv, writes metadata_raw.csv
python src/05_generate_questions.py  # reads metadata_raw.csv, adds questions
python src/06_validate.py            # reads metadata_raw.csv, validates
```

This is useful for debugging or re-running a single step after fixing an issue.

### What Happens During a Pipeline Run

```
Step 1: For each city, snap seed coordinates to nearest road.
        Extract OSM context (buildings, amenities, landuse, etc.) within 200m.
        --> output/sample_locations.csv

Step 2: For each sample, download a 512x512 satellite tile from GEE.
        NYC uses NAIP (0.6m), all others use Sentinel-2 (10m).
        --> output/images/sat/{sample_id}.png

Step 3: For each sample, download 4 street-level images from Google.
        Aligned to road bearing: forward, backward, left, right.
        --> output/images/sv/{sample_id}_{direction}.jpg

Step 4: Reverse geocode each sample (suburb, postcode, admin area).
        For NYC: also query US Census for demographics.
        --> output/metadata_raw.csv

Step C: Create 2x2 satellite composites for mismatch MCQ questions.
        --> output/images/composite/{sample_id}_4sat.png

Step 5: Generate all possible MCQs per sample (up to 13 types).
        Select the best question using diversity-aware scoring.

Step 6: Validate each sample. Remove those with CRITICAL issues.
        Write final dataset.
        --> output/dataset.jsonl
```

### Estimated Time

| Step | Time | Notes |
|------|------|-------|
| Step 1 (locations) | ~1 min | Instant if OSM data is pre-cached |
| Step 2 (satellite) | 2-5 min | GEE API, ~0.5s per sample |
| Step 3 (street view) | 3-8 min | Google API, 4 images per sample with rate limiting |
| Step 4 (metadata) | 1-2 min | Nominatim, 1 req/s |
| Step C (composites) | <1 min | Local image processing |
| Step 5 (questions) | <1 min | Local computation |
| Step 6 (validation) | <1 min | Local file checks |
| **Total** | **~10-15 min** | For 60 samples (6 cities x 10) |

---

## Adding a New City

You only need to edit one file: `src/config.py`. No other code changes required.

### Step 1: Find Your City's Bounding Box

The bounding box defines the geographic area for OSM data download. Format: `(south, west, north, east)` in decimal degrees.

**How to find it**:
1. Go to [bboxfinder.com](http://bboxfinder.com/) or [boundingbox.klokantech.com](https://boundingbox.klokantech.com/)
2. Draw a rectangle around your city
3. Copy the coordinates in the format `south, west, north, east`

**Tip**: Don't make the box too large -- Overpass API has size limits. For megacities, focus on the urban core rather than the entire metropolitan area.

### Step 2: Choose Seed Locations

Seeds are specific neighborhoods/locations where samples will be generated. Each seed is a tuple:

```python
("Neighborhood Name", latitude, longitude, "urban_character")
```

- `urban_character` is one of: `"residential"`, `"commercial"`, `"industrial"`, `"mixed"`
- This field is informational only -- it doesn't affect the pipeline logic
- Use Google Maps to find latitude/longitude for each location
- Aim for diversity: mix of commercial centers, residential areas, industrial zones, parks

**Finding coordinates**: Right-click any location in Google Maps and the lat/lon appears.

### Step 3: Add the Config Entry

```python
# In src/config.py, add to the CITIES dict:

"tokyo": {
    "name": "Tokyo",
    "country": "Japan",
    "bbox": (35.550, 139.550, 35.820, 139.930),  # S, W, N, E
    "satellite_source": "S2",  # Use "NAIP" for US cities, "S2" for everything else
    "seeds": [
        ("Shibuya",      35.6580, 139.7016, "commercial"),
        ("Shinjuku",     35.6938, 139.7034, "commercial"),
        ("Asakusa",      35.7148, 139.7967, "mixed"),
        ("Akihabara",    35.7023, 139.7745, "commercial"),
        ("Roppongi",     35.6627, 139.7307, "mixed"),
        ("Ueno",         35.7141, 139.7774, "mixed"),
        ("Odaiba",       35.6267, 139.7750, "commercial"),
        ("Ikebukuro",    35.7295, 139.7109, "commercial"),
        ("Setagaya",     35.6461, 139.6532, "residential"),
        ("Koto",         35.6727, 139.8171, "industrial"),
    ],
},
```

### Step 4: Run

```bash
# Download OSM data for the new city first
bash setup_data.sh

# Run pipeline for the new city
python src/run_pipeline.py --cities tokyo --samples 10
```

The pipeline will automatically download OSM data if not cached, fetch satellite and street view imagery, and generate questions.

---

## Configuring Satellite Images

Satellite image parameters are configurable in `src/config.py`:

```python
# Buffer radius in meters (how much area the satellite tile covers)
SAT_BUFFER_M = {
    "NAIP": 250,   # 250m radius -> ~500m x 500m area at 0.6m/px
    "S2": 500,     # 500m radius -> ~1km x 1km area at 10m/px
}

# Output image size in pixels
SAT_IMAGE_PX = 512
```

### What Buffer Means

The buffer defines how much area around the sample point is captured in the satellite tile:

| Buffer | Area Covered | Best For |
|--------|-------------|----------|
| 100m | ~200m x 200m | Close-up, individual buildings |
| 250m | ~500m x 500m | Block-level urban context (default NAIP) |
| 500m | ~1km x 1km | Neighborhood-level overview (default S2) |
| 1000m | ~2km x 2km | Wide area, district-level |

**Sentinel-2 note**: S2 has 10m native resolution. A 500m buffer gives ~100 native pixels across, which GEE upsamples to 512px. Decreasing the buffer below ~250m won't add real detail -- you'll just get interpolated pixels.

**NAIP note**: NAIP has 0.6m resolution. A 250m buffer gives ~833 native pixels across, downsampled to 512px. You could increase the buffer to 500m and still retain useful detail.

---

## Understanding the Output

### Directory Layout

After a successful pipeline run:

```
output/
 |-- dataset.jsonl                        # Final dataset
 |-- sample_locations.csv                 # Step 1: seed locations + OSM context
 |-- metadata_raw.csv                     # Step 4: enriched with geocoding
 '-- images/
     |-- sat/
     |   |-- nyc_0001.png                 # 512x512 satellite tile
     |   |-- paris_0001.png
     |   '-- ...
     |-- sv/
     |   |-- nyc_0001_along_fwd.jpg       # 640x640 street view (forward)
     |   |-- nyc_0001_along_bwd.jpg       # 640x640 street view (backward)
     |   |-- nyc_0001_cross_left.jpg      # 640x640 street view (left)
     |   |-- nyc_0001_cross_right.jpg     # 640x640 street view (right)
     |   '-- ...
     '-- composite/
         |-- nyc_0001_4sat.png            # 1024x1024 mismatch MCQ tile
         '-- ...
```

### The dataset.jsonl File

Each line is a complete JSON record. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique ID like `nyc_0001` |
| `location` | object | lat, lon, city, country, neighborhood, road name, suburb |
| `images` | object | Relative paths to all images + source metadata |
| `question` | string | The MCQ question text |
| `options` | object | `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `answer` | string | Correct option key (`"A"`, `"B"`, `"C"`, or `"D"`) |
| `topic` | string | Question type (e.g., `land_use`, `camera_direction`) |
| `difficulty` | string | `"easy"`, `"medium"`, or `"hard"` |
| `metadata` | object | All OSM-derived data (buildings, amenities, land use, etc.) |
| `validation` | object | Image presence flags and issue list |

### Image Descriptions

**Satellite images** (`images/sat/`):
- 512x512 PNG, centered on the sample point
- NAIP: 0.6m resolution true-colour RGB (US only)
- Sentinel-2: 10m resolution RGB (B4/B3/B2 bands), contrast-adjusted

**Street view images** (`images/sv/`):
- 640x640 JPEG, FOV=90 degrees, pitch=-5 degrees
- 4 orientations aligned to road bearing:
  - `along_fwd`: Looking along the road (bearing direction)
  - `along_bwd`: Looking along the road (opposite direction)
  - `cross_left`: Looking 90 degrees left of the road
  - `cross_right`: Looking 90 degrees right of the road

**Composite images** (`images/composite/`):
- 1024x1024 PNG, 2x2 grid of satellite tiles
- 1 correct satellite + 3 distractors (from other locations)
- Used by the `mismatch_mcq` question type
- Position labels: A=Top-Left, B=Top-Right, C=Bottom-Left, D=Bottom-Right

---

## Using the Dataset Viewer

The repository includes a web-based viewer for exploring the generated dataset.

### Starting the Viewer

```bash
cd viewer
bash serve.sh
```

This starts a local HTTP server and opens the viewer in your browser at `http://localhost:8765/viewer/`.

**Alternatively**, you can start the server manually:

```bash
python3 -m http.server 8765
# Then open http://localhost:8765/viewer/ in your browser
```

### Viewer Features

- **Filter** by city or question topic using the dropdowns
- **Search** by neighborhood, road name, or question text
- **Sample cards** show satellite thumbnail, city/topic badges, question text, and correct answer
- **Click a card** to open the detail modal with:
  - Full satellite image
  - All 4 street view images
  - Question with highlighted correct answer
  - Full metadata table
  - Google Maps link to the location
- **Keyboard navigation**: arrow keys to browse, Escape to close modal

**Requirement**: The viewer reads from `output/dataset.jsonl`. Run the pipeline first to generate this file.

---

## Question Types Reference

Detailed reference for all 13 question types, including what data triggers each one.

### OSM Metadata Questions

| # | Topic | Data Field | Condition to Generate | Answer Categories |
|---|-------|-----------|----------------------|-------------------|
| 1 | `land_use` | `land_use_category` | Must be a valid category key | 8 land use types |
| 2 | `building_height` | `osm_median_levels` | Must be a positive integer | 4 height ranges (1-3, 4-7, 8-20, 20+) |
| 3 | `urban_density` | `osm_building_count` | Must be a number | 4 density bins (0-15, 16-50, 51-150, 151+) |
| 4 | `road_type` | `highway_type` | Must be in ROAD_LABELS dict | 6 road classifications |
| 5 | `green_space` | `osm_has_park` | Must be True or False (with enough buildings) | Yes/No with distractors |
| 6 | `amenity_richness` | `osm_amenity_count` | Must be a number | 4 levels (0, 1-4, 5-19, 20+) |
| 7 | `road_surface` | `osm_road_surface` | Must be a known surface tag | 4 surface types |
| 8 | `junction_type` | `osm_junction_type` | Must be set (requires 2+ roads meeting) | 4 junction types |
| 9 | `water_proximity` | `osm_water_distance_m` | Must not be None (water exists in city) | 3 distance bins |
| 10 | `transit_density` | `osm_transit_stop_count` | Must not be None | 4 count bins |

### Cross-View Alignment Questions

| # | Topic | Data Field | Condition to Generate | Answer Categories |
|---|-------|-----------|----------------------|-------------------|
| 11 | `camera_direction` | `road_bearing` | Must be a valid float | 4 quadrants |
| 12 | `mismatch_binary` | `mismatch_negative_sid` | Needs a distractor sample | Yes/No match |
| 13 | `mismatch_mcq` | `composite_4sat_path` | Composite image must exist | 4 tile positions |

### How Question Selection Works

When a sample has metadata for multiple question types, the pipeline generates all possible questions and then picks the best one using a scoring system:

1. **Topic diversity**: Topics used less often in the dataset get a bonus
2. **Visual distinctiveness**: Visually striking answers score higher (e.g., skyscrapers > generic mid-rise, roundabouts > regular intersections, water proximity > no water)
3. **One question per sample**: The highest-scoring question is selected

Each question has 10 template paraphrases (in `src/question_templates.py`). A random paraphrase is chosen per sample for linguistic diversity.

---

## Troubleshooting

### "earthengine.Authenticate" or GEE initialization errors

```
Error: Please authorize access to your Earth Engine account
```

**Fix**: Run `earthengine authenticate` and follow the browser prompts. Make sure `GEE_PROJECT` in `src/config.py` matches your Google Cloud project ID.

### Overpass API timeout (Sao Paulo)

```
ERROR: failed to download sao_paulo context after 3 attempts
```

**Cause**: Sao Paulo's bounding box is large and Overpass sometimes times out.

**Fix**: Wait a few minutes and re-run `bash setup_data.sh`. The script skips already-downloaded files. If it keeps failing, try during off-peak hours (European night time -- Overpass servers are in Germany).

### Street View images are all black / tunnel-like

The pipeline has built-in tunnel detection. If images are still dark:

**Cause**: Google may snap to a nearby highway tunnel instead of the surface road.

**Fix**: The pipeline already filters these (max snap distance = 80m, tunnel heuristic). If you see this consistently for a seed, move the seed coordinates slightly on the map.

### Satellite images are very small (< 5 KB)

**Cause**: Cloud-covered tiles or no-data areas in the imagery collection.

**Fix**: The pipeline flags these as warnings. For Sentinel-2, you can adjust `S2_CLOUD_THRESHOLD` in `src/config.py` (default: 20% max cloud cover). Lowering it to 10 is more restrictive but may result in fewer available images.

### "No road found within snap radius"

```
SKIPPED: no road found within snap radius
```

**Cause**: The seed coordinates are too far from any mapped road in OSM.

**Fix**: Check the seed coordinates on [openstreetmap.org](https://www.openstreetmap.org). Move the seed to a location directly on or near a mapped road.

### Missing Census data for NYC

**Cause**: Either `CENSUS_API_KEY` is not set in `api_keys.env`, or the Census API is temporarily down.

**Fix**: Census enrichment is optional. If you need it, get a free key from [census.gov](https://api.census.gov/data/key_signup.html) and add it to `api_keys.env`.

### Rate limiting (HTTP 429)

```
Rate limited (attempt 2), waiting 60s...
```

**Cause**: Too many requests to the Overpass API in a short period.

**Fix**: This is handled automatically -- the pipeline waits and retries. If you're running multiple pipeline instances, stagger them. Pre-downloading with `setup_data.sh` avoids this entirely during pipeline runs.

### Pipeline crashes mid-run

**Fix**: Individual steps can be re-run. The pipeline reads from CSV files written by previous steps. Simply re-run the failed step or the full pipeline -- steps that already have cached data will skip re-downloading.

---

## Cost Estimate

For a full run (6 cities, 10 samples each = 60 samples):

| Service | Requests | Estimated Cost |
|---------|----------|---------------|
| Google Earth Engine | 60 satellite tiles | Free |
| Google Street View API | ~240 image requests | ~$1.50-3.00 USD |
| US Census API | ~10 requests (NYC only) | Free |
| OpenStreetMap Overpass | 12 bulk queries (cached) | Free |
| Nominatim | 60 geocoding requests | Free |

**Total**: Approximately $1.50-3.00 USD per full pipeline run, depending on Street View API pricing tier.
