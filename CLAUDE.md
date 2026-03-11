# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

EOLLM is a pipeline for generating a multi-city Visual Question Answering (VQA) dataset pairing satellite and street-level imagery with multiple-choice questions derived from OpenStreetMap metadata. Target use case: training/evaluating Vision-Language Models on urban understanding tasks.

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (one-time)
earthengine authenticate

# Run full pipeline (6 cities, 10 samples each = 60 total)
python src/run_pipeline.py

# Specific cities / sample count
python src/run_pipeline.py --cities nyc paris --samples 5

# Run individual steps in isolation
python src/01_sample_locations.py
python src/02_fetch_satellite.py
python src/03_fetch_streetview.py
python src/04_enrich_metadata.py
python src/05_generate_questions.py
python src/06_validate.py
```

## API Keys

Create `api_keys.env` in the project root:
```
GOOGLE_STREETVIEW_KEY=AIza...
CENSUS_API_KEY=...          # optional; only used for NYC samples
```

Set `GEE_PROJECT` in `src/config.py` to your Google Earth Engine project ID.

## Architecture

The pipeline is a linear 6-step chain. Each step's `run()` function takes a list of sample dicts and returns an enriched list. `run_pipeline.py` imports steps via `importlib` (modules are named with numeric prefixes) and threads them together, writing `output/dataset.jsonl` at the end.

| Step | File | Input | Output |
|------|------|-------|--------|
| 1 | `01_sample_locations.py` | City config seeds | `output/sample_locations.csv`, OSM data cached in `data/` |
| 2 | `02_fetch_satellite.py` | Sample list | `output/images/sat/{id}.png` |
| 3 | `03_fetch_streetview.py` | Sample list | `output/images/sv/{id}_{label}.jpg` (4 per sample) |
| 4 | `04_enrich_metadata.py` | Sample list | `output/metadata_raw.csv` |
| 5 | `05_generate_questions.py` | Sample list | MCQ fields added to each sample |
| 6 | `06_validate.py` | Sample list | Validation flags added; final `output/dataset.jsonl` written by orchestrator |

## Key Configuration (`src/config.py`)

- **Adding a city**: append an entry to `CITIES` dict with `name`, `country`, `bbox` (S,W,N,E), `satellite_source` (`"NAIP"` for US, `"S2"` elsewhere), and a `seeds` list of `(label, lat, lon, urban_character)` tuples. No other code changes needed.
- **OSM data**: downloaded once per city as two Overpass queries (roads + context) and cached in `data/{city}_roads_osm.json` and `data/{city}_context_osm.json`.
- **Question types**: `land_use`, `building_height`, `urban_density`, `road_type`, `green_space`, `amenity_richness`. Selection per sample uses a scoring heuristic for topic diversity.
- **Reproducibility**: `random.seed(42)` is used for option shuffling.

## Output

Final dataset: `output/dataset.jsonl` — one JSON record per line with fields: `sample_id`, `location`, `images`, `question`, `options`, `answer`, `difficulty`, `topic`, `generation_method`, `metadata`, `validation`. Image paths in the JSONL are relative to `output/`.

## Known Issues

- **São Paulo**: Overpass API may timeout due to large bbox; pipeline retries 3x with backoff. Missing context file = no OSM metadata for affected samples.
- **London Canary Wharf**: No outdoor Street View coverage; all 4 SV images will be missing.
- **Sentinel-2 cloud cover**: Files under 5 KB are flagged as warnings (blank/cloud-covered tiles).
- **`building:levels` coverage**: Sparse in São Paulo and Singapore; samples without it cannot generate `building_height` questions.

# USER RULES
- after every significant change, bugfix, feature addition commit, but do not add yourself as a contributer, commit normally as git commig -m '...' and so on
- NEVER MAKE CALLS TO THE APIs WITHOUT ASKING THE USER