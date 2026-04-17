# Dataset Report — EOLLM Compressed (splits_per_city)

Generated: 2026-04-17

## Split sizes

| Split | Records | Unique sample_ids |
|-------|---------|-------------------|
| train | 26,931 | 3,186 |
| validation | 6,984 | — |

One sample can produce multiple question records (different topics).

## Topic distribution (train)

| Topic | Count |
|-------|-------|
| amenity_richness | 2,000 |
| camera_direction | 2,000 |
| land_use | 2,000 |
| mismatch_binary_hard | 2,000 |
| mismatch_binary_easy | 2,000 |
| mismatch_mcq_hard | 2,000 |
| road_surface | 2,000 |
| transit_density | 2,000 |
| urban_density | 2,000 |
| junction_type | 2,000 |
| mismatch_mcq_easy | 2,000 |
| road_type | 2,000 |
| green_space | 1,500 |
| building_height | 1,431 |

## image_mode distribution (train)

| image_mode | Count | Images per sample | Notes |
|---|---|---|---|
| satellite_marked | 16,931 | 1 | 512×512 sat + red dot |
| streetview_mega | 4,000 | 1 (but opens 16 JPEGs) | 1024×1024 mega composite |
| streetview_binary | 2,045 | 1 | 512×512 from neg paths |
| satellite_arrow | 2,000 | 1 primary + 1 query_sv | Multi-image: sat + SV |
| streetview_composite | 1,955 | 1 | 512×512 from 4 angles |

## Image dimensions (raw sources)

- Satellite: **512×512** PNG (100% uniform across sampled records)
- Street view: **640×640** JPEG (100% uniform across sampled records)

## Text statistics

| Field | Min | Max | Avg |
|-------|-----|-----|-----|
| Question text (chars) | 34 | 111 | 82 |
| Option text (chars) | 7 | 60 | 23 |
| Full prompt (Q + options, chars) | 117 | 323 | 183 |
| Answer | 1 | 1 | Always single letter (A/B/C/D) |

## Answer balance

| Answer | Count | % |
|--------|-------|---|
| A | 7,717 | 28.6% |
| B | 7,642 | 28.4% |
| D | 5,846 | 21.7% |
| C | 5,726 | 21.3% |

Slight imbalance toward A/B. Not severe enough to warrant resampling.

## Difficulty distribution

| Difficulty | Count |
|------------|-------|
| easy | 12,931 |
| medium | 10,000 |
| hard | 4,000 |

## Cities

40 cities total. Range: Paris (874) to Nairobi (271).

## Field mapping for training

| Role | Source field | Notes |
|------|-------------|-------|
| Instruction text | `question` + `options` | Formatted as "Q\nA. ...\nB. ...\nC. ...\nD. ..." |
| Answer | `answer` | Single letter: A, B, C, or D |
| Primary image | `get_images_for_question(record)["primary"]` | PIL Image, varies by image_mode |
| Secondary image | `get_images_for_question(record)["query_sv"]` | Only for camera_direction (satellite_arrow mode) |

## Multi-image handling

Only `camera_direction` (2,000 records, 7.4% of train) produces two images:
1. Satellite with directional arrows (primary)
2. The query street-view photo (query_sv)

Both go into the user content list. All other modes produce a single image.

## Anomalies / notes

- `mismatch_binary_hard` with `mismatch_is_match=True` uses `streetview_composite` mode (shows query location's own SV), while `mismatch_is_match=False` uses `streetview_binary` mode (shows the negative location's SV). Both present a satellite image separately in the question text context — the composite_utils handles this correctly.
- No missing image paths detected in sampled records.
- No train/validation sample_id overlap expected (per_city split strategy shuffles locations within each city).
