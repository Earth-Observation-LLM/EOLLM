# Benchmark-modes sweep — cross-model report

Models: gemma-4-12b-awq, qwen2.5-vl-7b-awq, qwen3.5-4b-awq, qwen3.5-9b-awq
Modes: full, sat_only, sv_only, blind
Thinking: **OFF** (letter-only output) for every model.

## 1. Overall accuracy (model × mode)

| model | full | sat_only | sv_only | blind | full 90% CI | n(full) |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-4-12b-awq | 50.3 | 33.6 | 37.7 | 23.5 | 49.2–51.4 | 5240 |
| qwen2.5-vl-7b-awq | 47.2 | 35.6 | 37.3 | 38.3 | 46.1–48.2 | 5240 |
| qwen3.5-4b-awq | 49.0 | 33.8 | 37.0 | 38.9 | 47.9–50.2 | 5240 |
| qwen3.5-9b-awq | 51.1 | 34.6 | 36.0 | 38.1 | 50.0–52.3 | 5240 |

## 2. Modality ablation (per model)

`full−blind` = total vision gain. `full−sat_only` / `full−sv_only` = how much accuracy is lost when the *other* view is removed (≈0 means that view was redundant). Percentage-point differences.

| model | full | sat_only | sv_only | blind | full−blind | full−sat_only | full−sv_only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-4-12b-awq | 50.3 | 33.6 | 37.7 | 23.5 | +26.8 | +16.7 | +12.6 |
| qwen2.5-vl-7b-awq | 47.2 | 35.6 | 37.3 | 38.3 | +8.8 | +11.6 | +9.9 |
| qwen3.5-4b-awq | 49.0 | 33.8 | 37.0 | 38.9 | +10.1 | +15.2 | +12.0 |
| qwen3.5-9b-awq | 51.1 | 34.6 | 36.0 | 38.1 | +13.0 | +16.5 | +15.1 |

## 3. Per-task accuracy (model × topic), one table per mode

### mode = `full`

| topic | gemma-4-12b-awq | qwen2.5-vl-7b-awq | qwen3.5-4b-awq | qwen3.5-9b-awq |
| --- | --- | --- | --- | --- |
| amenity_richness | 35.6 | 31.8 | 35.4 | 36.3 |
| building_height | 61.6 | 57.9 | 61.6 | 56.8 |
| camera_direction | 25.2 | 24.7 | 21.4 | 23.5 |
| green_space | 58.6 | 81.8 | 41.4 | 36.9 |
| junction_type | 66.5 | 57.5 | 50.0 | 55.7 |
| land_use | 47.3 | 55.6 | 55.6 | 53.2 |
| mismatch_binary_easy | 79.8 | 59.1 | 74.3 | 87.2 |
| mismatch_binary_hard | 64.6 | 45.6 | 58.0 | 72.9 |
| mismatch_mcq_easy | 43.2 | 39.7 | 45.8 | 51.3 |
| mismatch_mcq_hard | 34.7 | 33.7 | 38.5 | 39.4 |
| road_surface | 78.2 | 90.8 | 91.4 | 85.5 |
| road_type | 65.3 | 50.6 | 57.0 | 53.0 |
| transit_density | 28.0 | 28.0 | 34.2 | 31.4 |
| urban_density | 37.3 | 41.8 | 36.3 | 39.0 |

### mode = `sat_only`

| topic | gemma-4-12b-awq | qwen2.5-vl-7b-awq | qwen3.5-4b-awq | qwen3.5-9b-awq |
| --- | --- | --- | --- | --- |
| camera_direction | 22.3 | 29.2 | 20.7 | 25.7 |
| mismatch_binary_easy | 43.9 | 53.2 | 51.1 | 47.5 |
| mismatch_binary_hard | 54.6 | 44.7 | 47.0 | 55.1 |
| mismatch_mcq_easy | 21.9 | 24.9 | 25.9 | 23.5 |
| mismatch_mcq_hard | 25.2 | 25.9 | 24.2 | 21.1 |

### mode = `sv_only`

| topic | gemma-4-12b-awq | qwen2.5-vl-7b-awq | qwen3.5-4b-awq | qwen3.5-9b-awq |
| --- | --- | --- | --- | --- |
| mismatch_binary_easy | 52.0 | 53.7 | 51.1 | 54.2 |
| mismatch_binary_hard | 47.5 | 44.2 | 46.3 | 44.9 |
| mismatch_mcq_easy | 27.3 | 27.3 | 25.2 | 23.3 |
| mismatch_mcq_hard | 24.0 | 24.0 | 25.4 | 21.6 |

### mode = `blind`

| topic | gemma-4-12b-awq | qwen2.5-vl-7b-awq | qwen3.5-4b-awq | qwen3.5-9b-awq |
| --- | --- | --- | --- | --- |
| amenity_richness | 24.2 | 14.3 | 19.0 | 21.1 |
| building_height | 32.6 | 44.7 | 46.3 | 45.8 |
| camera_direction | 5.9 | 25.4 | 25.2 | 26.1 |
| green_space | 76.4 | 78.8 | 53.7 | 61.6 |
| junction_type | 7.5 | 47.0 | 47.0 | 41.6 |
| land_use | 17.8 | 54.4 | 38.2 | 38.7 |
| mismatch_binary_easy | 46.6 | 47.5 | 50.6 | 45.1 |
| mismatch_binary_hard | 55.8 | 52.3 | 46.6 | 54.4 |
| mismatch_mcq_easy | 0.7 | 24.9 | 26.1 | 27.1 |
| mismatch_mcq_hard | 0.7 | 26.4 | 24.9 | 24.2 |
| road_surface | 26.4 | 82.8 | 87.5 | 52.8 |
| road_type | 15.0 | 25.4 | 31.1 | 39.9 |
| transit_density | 26.8 | 26.6 | 27.1 | 26.4 |
| urban_density | 22.6 | 24.9 | 47.7 | 49.9 |

## 4. Per-difficulty accuracy (model × mode)

| model | mode | easy | medium | hard |
| --- | --- | --- | --- | --- |
| gemma-4-12b-awq | full | 61.6 | 37.3 | 49.6 |
| gemma-4-12b-awq | sat_only | 32.9 | 22.3 | 39.9 |
| gemma-4-12b-awq | sv_only | 39.7 | — | 35.7 |
| gemma-4-12b-awq | blind | 26.6 | 17.8 | 28.3 |
| qwen2.5-vl-7b-awq | full | 59.4 | 35.9 | 39.7 |
| qwen2.5-vl-7b-awq | sat_only | 39.1 | 29.2 | 35.3 |
| qwen2.5-vl-7b-awq | sv_only | 40.5 | — | 34.1 |
| qwen2.5-vl-7b-awq | blind | 47.8 | 26.8 | 39.3 |
| qwen3.5-4b-awq | full | 61.3 | 34.8 | 48.2 |
| qwen3.5-4b-awq | sat_only | 38.5 | 20.7 | 35.6 |
| qwen3.5-4b-awq | sv_only | 38.1 | — | 35.9 |
| qwen3.5-4b-awq | blind | 45.3 | 32.6 | 35.7 |
| qwen3.5-9b-awq | full | 61.8 | 36.4 | 56.2 |
| qwen3.5-9b-awq | sat_only | 35.5 | 25.7 | 38.1 |
| qwen3.5-9b-awq | sv_only | 38.7 | — | 33.3 |
| qwen3.5-9b-awq | blind | 42.3 | 32.7 | 39.3 |

## 5. Generalization: seen vs unseen cities (model × mode)

`benchmark_city_type` split. Gap = seen − unseen (pp).

| model | mode | seen | unseen | seen−unseen |
| --- | --- | --- | --- | --- |
| gemma-4-12b-awq | full | 50.6 | 49.2 | +1.4 |
| gemma-4-12b-awq | sat_only | 33.0 | 35.9 | -2.9 |
| gemma-4-12b-awq | sv_only | 37.0 | 40.7 | -3.7 |
| gemma-4-12b-awq | blind | 22.9 | 25.9 | -2.9 |
| qwen2.5-vl-7b-awq | full | 47.8 | 44.4 | +3.5 |
| qwen2.5-vl-7b-awq | sat_only | 35.9 | 34.2 | +1.7 |
| qwen2.5-vl-7b-awq | sv_only | 38.1 | 34.0 | +4.1 |
| qwen2.5-vl-7b-awq | blind | 38.9 | 35.8 | +3.1 |
| qwen3.5-4b-awq | full | 49.8 | 45.6 | +4.2 |
| qwen3.5-4b-awq | sat_only | 34.3 | 31.8 | +2.5 |
| qwen3.5-4b-awq | sv_only | 37.4 | 35.5 | +1.8 |
| qwen3.5-4b-awq | blind | 38.6 | 39.8 | -1.1 |
| qwen3.5-9b-awq | full | 51.3 | 50.4 | +0.9 |
| qwen3.5-9b-awq | sat_only | 34.2 | 36.1 | -1.9 |
| qwen3.5-9b-awq | sv_only | 36.1 | 35.5 | +0.6 |
| qwen3.5-9b-awq | blind | 36.9 | 43.0 | -6.1 |
