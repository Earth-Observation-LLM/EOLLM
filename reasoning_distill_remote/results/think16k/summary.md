# Ablation summary — think16k
_started 2026-06-18 12:33:33  •  updated 2026-06-18 12:46:38_

**Progress:** 180 done / 180  •  errors 0  •  ctx-overflow 0  •  parse-retried 10  •  unparsed 1
**Overall acc:** 47.22%  •  median latency 32.3s  •  median completion 1039 tok
**Throughput:** 13.7 gen/min  •  ETA 0.0 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 60.0% | 60.0% | 40.0% | 0.0% | 5 |
| building_height | 60.0% | 60.0% | 60.0% | 40.0% | 5 |
| green_space | 0.0% | 0.0% | 20.0% | 20.0% | 5 |
| junction_type | 60.0% | 80.0% | 100.0% | 60.0% | 5 |
| land_use | 60.0% | 60.0% | 60.0% | 60.0% | 5 |
| road_surface | 80.0% | 80.0% | 100.0% | 80.0% | 5 |
| road_type | 40.0% | 40.0% | 60.0% | 0.0% | 5 |
| transit_density | 20.0% | 40.0% | 20.0% | 40.0% | 5 |
| urban_density | 40.0% | 40.0% | 20.0% | 40.0% | 5 |

## Thinking / truncation

| mode | n | median tok | p90 tok | max tok | truncated |
|---|---|---|---|---|---|
| full | 45 | 1458 | 5244 | 14151 | 1 (2.2%) |
| sat_only | 45 | 1337 | 3369 | 6599 | 0 (0.0%) |
| sv_only | 45 | 1142 | 3384 | 6689 | 0 (0.0%) |
| blind | 45 | 855 | 1454 | 2176 | 0 (0.0%) |

## Visual-dependency buckets (full vs blind; sat vs sv)

| topic | visual | leaked | hard | regress | sat-solves | sv-solves | both | neither |
|---|---|---|---|---|---|---|---|---|
| amenity_richness | 3 | 0 | 2 | 0 | 1 | – | 2 | 2 |
| building_height | 2 | 1 | 1 | 1 | – | – | 3 | 2 |
| green_space | 0 | 0 | 4 | 1 | – | 1 | – | 4 |
| junction_type | 1 | 2 | 1 | 1 | – | 1 | 4 | – |
| land_use | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 |
| road_surface | 1 | 3 | 0 | 1 | – | 1 | 4 | – |
| road_type | 2 | 0 | 3 | 0 | – | 1 | 2 | 2 |
| transit_density | 1 | 0 | 2 | 2 | 1 | – | 1 | 3 |
| urban_density | 1 | 1 | 2 | 1 | 1 | – | 1 | 3 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_