# Ablation summary — current
_started 2026-06-18 12:30:43  •  updated 2026-06-18 12:30:49_

**Progress:** 180 done / 180  •  errors 0  •  ctx-overflow 0  •  parse-retried 0  •  unparsed 0
**Overall acc:** 45.56%  •  median latency 5.6s  •  median completion 7 tok
**Throughput:** 1693.4 gen/min  •  ETA 0.0 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 40.0% | 40.0% | 40.0% | 0.0% | 5 |
| building_height | 40.0% | 40.0% | 60.0% | 40.0% | 5 |
| green_space | 0.0% | 0.0% | 0.0% | 60.0% | 5 |
| junction_type | 80.0% | 60.0% | 100.0% | 80.0% | 5 |
| land_use | 60.0% | 40.0% | 60.0% | 40.0% | 5 |
| road_surface | 100.0% | 80.0% | 100.0% | 80.0% | 5 |
| road_type | 40.0% | 40.0% | 20.0% | 20.0% | 5 |
| transit_density | 40.0% | 60.0% | 40.0% | 20.0% | 5 |
| urban_density | 20.0% | 40.0% | 20.0% | 40.0% | 5 |

## Thinking / truncation

| mode | n | median tok | p90 tok | max tok | truncated |
|---|---|---|---|---|---|
| full | 45 | 0 | 0 | 0 | 0 (0.0%) |
| sat_only | 45 | 0 | 0 | 0 | 0 (0.0%) |
| sv_only | 45 | 0 | 0 | 0 | 0 (0.0%) |
| blind | 45 | 0 | 0 | 0 | 0 (0.0%) |

## Visual-dependency buckets (full vs blind; sat vs sv)

| topic | visual | leaked | hard | regress | sat-solves | sv-solves | both | neither |
|---|---|---|---|---|---|---|---|---|
| amenity_richness | 2 | 0 | 3 | 0 | 1 | 1 | 1 | 2 |
| building_height | 0 | 2 | 3 | 0 | – | 1 | 2 | 2 |
| green_space | 0 | 0 | 2 | 3 | – | – | – | 5 |
| junction_type | 1 | 3 | 0 | 1 | – | 2 | 3 | – |
| land_use | 1 | 2 | 2 | 0 | – | 1 | 2 | 2 |
| road_surface | 1 | 4 | 0 | 0 | – | 1 | 4 | – |
| road_type | 2 | 0 | 2 | 1 | 1 | – | 1 | 3 |
| transit_density | 1 | 1 | 3 | 0 | 1 | – | 2 | 2 |
| urban_density | 0 | 1 | 3 | 1 | 1 | – | 1 | 3 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_