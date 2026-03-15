# EOLLM -- Urban Visual Question Answering Dataset Generator

A fully automated pipeline that builds a multi-city **Visual Question Answering (VQA)** dataset by pairing high-resolution satellite imagery with Google Street View photos and generating multiple-choice questions from OpenStreetMap metadata. Built for training and evaluating **Vision-Language Models** on real-world urban understanding tasks.

```
Satellite tile  +  4 Street View angles  +  OSM metadata  -->  Multiple-choice questions
```

> **See [USAGE.md](USAGE.md) for the full setup and usage guide.**

---

## What It Does

Given a list of cities and seed coordinates, the pipeline:

1. Snaps each seed to the nearest mapped road using OpenStreetMap
2. Downloads a high-resolution satellite tile from the best available source
3. Fetches 4 road-aligned Street View images (forward, backward, left, right)
4. Extracts rich urban context from OSM (buildings, land use, amenities, transit, water, junctions)
5. Enriches samples with reverse geocoding and (optionally) US Census demographics
6. Generates annotated composites (marked satellites, arrow overlays, 2x2 grids)
7. Produces up to 13 question types per sample with deterministic answer keys
8. Validates everything and writes the final dataset

**No manual annotation required.** Every answer is derived programmatically from OSM tags and coordinate geometry.

---

## Cities

| City | Country | Satellite Source | Resolution | Seed Locations |
|------|---------|-----------------|------------|----------------|
| New York City | USA | NAIP via Google Earth Engine | 0.6 m/px | Times Square, Wall Street, Williamsburg, DUMBO, LIC, UES, Harlem, Flushing, South Bronx, Staten Island |
| Paris | France | IGN Geoplateforme (WMS) | 0.2 m/px | Champs-Elysees, Le Marais, La Defense, Belleville, Saint-Germain, Montmartre, Bercy, Batignolles, Republique, Bastille |
| London | UK | ESRI World Imagery | 0.3-0.5 m/px | City of London, Canary Wharf, Camden, Shoreditch, Brixton, Kensington, Stratford, Croydon, Greenwich, Westminster |
| Singapore | Singapore | ESRI World Imagery | 0.3-0.5 m/px | Marina Bay, Chinatown, Toa Payoh, Orchard Road, Jurong East, Bukit Timah, Geylang, Punggol, Tampines, Sentosa |
| Sao Paulo | Brazil | ESRI World Imagery | 0.3-0.5 m/px | Avenida Paulista, Faria Lima, Liberdade, Vila Madalena, Mooca, Pinheiros, Centro Historico, Jardins, Bela Vista, Vila Mariana |
| Amsterdam | Netherlands | ESRI World Imagery | 0.3-0.5 m/px | Dam Square, Jordaan, De Pijp, Zuidas, Noord, Oost, Westpoort, Amstelveen, Bijlmer, Watergraafsmeer |

Satellite source is **auto-detected** from coordinates (NAIP for US, IGN for France, ESRI elsewhere). All sources fall back to Sentinel-2 (10 m/px) via GEE if the primary fails.

Adding a new city requires editing only `src/config.py` -- see [USAGE.md](USAGE.md#adding-a-new-city).

---

## Question Types

13 question types across two categories. Each sample generates **all feasible questions** (typically 5-8), and a scoring heuristic selects the best one for topic diversity.

### OSM Metadata Questions (10 types)

| Topic | Example Question | Answer Space | Difficulty |
|-------|-----------------|--------------|------------|
| `land_use` | What is the primary land use character of this area? | Residential, Commercial & Office, Retail & Shopping, Industrial, Mixed Use, Institutional, Open Space, Transport | Easy |
| `building_height` | What building height category best describes the dominant structures? | Low-rise (1-3), Mid-rise (4-7), High-rise (8-20), Skyscraper (20+) | Easy |
| `urban_density` | What is the approximate urban density level? | Low, Moderate, High, Very High | Medium |
| `road_type` | What type of road infrastructure is dominant? | Motorway, Trunk, Primary, Secondary, Tertiary, Residential | Easy |
| `green_space` | Is there publicly accessible green space nearby? | Yes / No (with descriptive distractors) | Easy |
| `amenity_richness` | What is the level of commercial amenity presence? | Minimal, Low, Moderate, High | Medium |
| `road_surface` | What is the surface type of the main road? | Asphalt, Cobblestone/Sett, Unpaved/Gravel, Concrete | Easy |
| `junction_type` | What type of road junction is shown? | Roundabout, Signalized, Unsignalized, Grade-separated | Medium |
| `water_proximity` | Is there a water body near this location? | Adjacent (<50 m), Nearby (50-150 m), None | Medium |
| `transit_density` | What level of transit stop density exists? | None, Low (1-2), Moderate (3-5), High (6+) | Medium |

### Cross-View Alignment Tasks (3 types)

These test spatial reasoning between satellite and street-level views. They **do not rely on OSM tags** -- they use coordinate geometry, road bearings, and composite images.

| Topic | Task | Answer Space | Difficulty |
|-------|------|--------------|------------|
| `camera_direction` | Match which arrow-annotated satellite image corresponds to the street view camera bearing | 4 satellite options (TL/TR/BL/BR) | Medium |
| `mismatch_binary` | Does this street view composite correspond to this marked satellite image? | Yes / No | Medium |
| `mismatch_mcq` | Which of four street view composites matches the marked satellite? | A / B / C / D | Hard |

All answers are deterministic. Option shuffling uses per-sample RNG seeds for reproducibility.

---

## Pipeline Architecture

```
src/run_pipeline.py (orchestrator)
 |
 |-- Step 1  01_sample_locations.py   Snap seeds to roads, extract OSM context
 |-- Step 2  02_fetch_satellite.py    Download satellite tiles (NAIP/IGN/ESRI/S2)
 |-- Step 3  03_fetch_streetview.py   Download 4 road-aligned Street View images
 |-- Step 4  04_enrich_metadata.py    Reverse geocoding + US Census enrichment
 |-- Step 5  07_generate_composites.py  Arrow overlays, 2x2 composites, distractor selection
 |-- Step 6  05_generate_questions.py   Generate all feasible MCQs per sample
 '-- Step 7  06_validate.py            Per-sample and dataset-level validation
                                        --> output/dataset.jsonl
```

Each step's `run()` function takes a list of sample dicts and returns an enriched list. The orchestrator threads them together and writes the final JSONL.

---

## Data Sources

| Source | What It Provides | Cost |
|--------|-----------------|------|
| **OpenStreetMap** (Overpass API) | Roads, buildings, land use, amenities, parks, water, transit stops | Free (cached per city) |
| **Google Earth Engine** | NAIP (US, 0.6 m) and Sentinel-2 (global, 10 m) satellite tiles | Free with GEE project |
| **IGN Geoplateforme** | Satellite tiles for France (0.2 m) | Free, no key needed |
| **ESRI World Imagery** | Satellite tiles for rest of world (0.3-0.5 m) | Free, no key needed |
| **Google Street View** Static API | 4 street-level images per sample (640x640) | **Paid** -- ~$7 per 1,000 requests |
| **Nominatim** | Reverse geocoding (suburb, postcode, admin area) | Free, 1 req/s |
| **US Census ACS5** | Population, income, housing (NYC only) | Free with API key |

### API Cost Estimate (default run: 6 cities, 10 samples = 60 total)

| API | Calls | Cost |
|-----|-------|------|
| Street View metadata | 60 | ~$0.42 |
| Street View images | 240 | ~$1.68 |
| **Total** | **300** | **~$2.10** |

Everything else is free. See [USAGE.md](USAGE.md#cost-estimate) for details.

---

## Output

### Final Dataset

`output/dataset.jsonl` -- one JSON record per line:

```json
{
  "sample_id": "nyc_0001",
  "location": {
    "latitude": 40.758,
    "longitude": -73.985,
    "city": "New York City",
    "country": "USA",
    "neighborhood": "Times Square",
    "road_name": "7th Avenue"
  },
  "images": {
    "satellite": "images/sat/nyc_0001.png",
    "satellite_source": "NAIP",
    "streetview_along_fwd": "images/sv/nyc_0001_along_fwd.jpg",
    "streetview_along_bwd": "images/sv/nyc_0001_along_bwd.jpg",
    "streetview_cross_left": "images/sv/nyc_0001_cross_left.jpg",
    "streetview_cross_right": "images/sv/nyc_0001_cross_right.jpg"
  },
  "question": "What is the primary land use character of this area?",
  "options": {"A": "Institutional", "B": "Mixed Use", "C": "Industrial", "D": "Commercial & Office"},
  "answer": "D",
  "difficulty": "easy",
  "topic": "land_use",
  "question_count": 7,
  "questions": [ ... ],
  "metadata": { ... },
  "validation": { ... }
}
```

### Directory Layout

```
output/
  dataset.jsonl                          Final dataset
  sample_locations.csv                   Step 1 output
  metadata_raw.csv                       Step 4 output
  images/
    sat/          {id}.png               512x512 satellite tiles
    sat_marked/   {id}.png               Satellite with red center dot
    sat_arrow/    {id}_arrow_{dir}.png   Satellite with bearing arrow
    sv/           {id}_{label}.jpg       640x640 street view (4 per sample)
    composite/    {id}_*.png             2x2 grids and mega-composites

data/
  {city}_roads_osm.json                  Cached OSM road networks
  {city}_context_osm.json                Cached OSM context data
```

---

## Interactive Viewers

Two browser-based viewers are included for exploring the dataset:

### Main Viewer (`viewer/index.html`)

Browse all samples and question types. Filter by city, topic, difficulty. Inspect images, metadata, and answers.

```bash
bash viewer/serve.sh
# Opens http://localhost:8765/viewer/
```

### Geolocation Viewer (`viewer/geo.html`)

Dedicated viewer for cross-view alignment tasks (camera_direction, mismatch_binary, mismatch_mcq). Shows arrow-annotated satellites and street view composites side-by-side.

```bash
bash viewer/serve_geo.sh
# Opens http://localhost:8080/viewer/geo.html
```

Both viewers load `output/dataset.jsonl` via file upload (drag-and-drop supported).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine (one-time)
earthengine authenticate

# Set up API keys
cp api_keys.env.example api_keys.env
# Edit api_keys.env with your Google Street View key

# Set your GEE project ID in src/config.py

# Pre-download OSM data (recommended, avoids rate limiting)
bash setup_data.sh

# Run for a single city to test
python src/run_pipeline.py --cities amsterdam --samples 3

# Full run (all 6 cities, 10 samples each)
python src/run_pipeline.py
```

For the complete setup guide, troubleshooting, and configuration reference, see **[USAGE.md](USAGE.md)**.

---

## Key Design Decisions

- **Multi-source satellite fallback**: NAIP -> IGN -> ESRI -> Sentinel-2, auto-selected by coordinates
- **Road-aligned Street View**: Camera headings derived from OSM road bearing, not arbitrary angles
- **Tunnel detection**: Color-channel analysis rejects underground/tunnel Street View images
- **Per-sample RNG seeding**: Deterministic but varied distractor selection via `Random(sample_id + topic)`
- **All questions stored**: Each sample keeps its full `questions` array; the "best" one is selected for backward compatibility
- **Same-city and cross-city distractors**: Geolocation tasks generate both difficulty variants
- **Validation gates**: CRITICAL issues remove samples; WARNINGs flag them for review

---

## Known Limitations

- **Sao Paulo**: Large Overpass bbox can timeout. Pipeline retries 3x with backoff.
- **London Canary Wharf**: No outdoor Street View coverage. All 4 SV images will be missing.
- **Satellite quality**: Tiles under 5 KB are auto-rejected and the next source in the fallback chain is tried.
- **`building:levels`**: Sparse in Sao Paulo and Singapore -- no `building_height` questions for those samples.
- **Road surface tags**: Inconsistent globally. `road_surface` questions only generated when the OSM `surface` tag exists.
- **Sentinel-2 fallback**: 10 m/px native resolution is much coarser than primary sources. Used only as last resort.

---

## Repository Structure

```
EOLLM/
  src/
    run_pipeline.py             Orchestrator -- runs all steps, writes JSONL
    config.py                   Cities, seeds, constants, satellite config
    01_sample_locations.py      OSM road snapping + context extraction
    02_fetch_satellite.py       Multi-source satellite fetching
    03_fetch_streetview.py      Google Street View download + tunnel detection
    04_enrich_metadata.py       Nominatim geocoding + Census enrichment
    05_generate_questions.py    MCQ generation for 13 question types
    06_validate.py              Per-sample and dataset-level validation
    07_generate_composites.py   Image composites, arrows, distractor grids
    question_templates.py       10 paraphrases per question type (130+ templates)
    utils.py                    Haversine, bbox, bearing utilities
  viewer/
    index.html                  Full dataset browser
    geo.html                    Geolocation question viewer
    serve.sh                    Start local server for main viewer
    serve_geo.sh                Start local server for geo viewer
  setup_data.sh                 Pre-download all OSM data
  api_keys.env.example          API key template
  requirements.txt              Python dependencies
  USAGE.md                      Detailed setup and usage guide
```
