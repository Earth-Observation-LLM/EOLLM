# Ablation summary — multistep
_started 2026-06-18 12:49:20  •  updated 2026-06-18 12:50:15_

**Progress:** 180 done / 180  •  errors 0  •  ctx-overflow 0  •  parse-retried 0  •  unparsed 0
**Overall acc:** 53.33%  •  median latency 42.5s  •  median completion 616 tok
**Throughput:** 194.9 gen/min  •  ETA 0.0 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 60.0% | 60.0% | 60.0% | 40.0% | 5 |
| building_height | 60.0% | 40.0% | 60.0% | 40.0% | 5 |
| green_space | 20.0% | 40.0% | 60.0% | 100.0% | 5 |
| junction_type | 60.0% | 80.0% | 100.0% | 40.0% | 5 |
| land_use | 60.0% | 80.0% | 60.0% | 40.0% | 5 |
| road_surface | 80.0% | 60.0% | 100.0% | 80.0% | 5 |
| road_type | 40.0% | 40.0% | 80.0% | 0.0% | 5 |
| transit_density | 40.0% | 60.0% | 20.0% | 20.0% | 5 |
| urban_density | 40.0% | 40.0% | 20.0% | 40.0% | 5 |

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
| amenity_richness | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| building_height | 1 | 2 | 2 | 0 | – | 1 | 2 | 2 |
| green_space | 0 | 1 | 0 | 4 | 1 | 2 | 1 | 1 |
| junction_type | 2 | 1 | 1 | 1 | – | 1 | 4 | – |
| land_use | 2 | 1 | 1 | 1 | 2 | 1 | 2 | – |
| road_surface | 0 | 4 | 1 | 0 | – | 2 | 3 | – |
| road_type | 2 | 0 | 3 | 0 | – | 2 | 2 | 1 |
| transit_density | 1 | 1 | 3 | 0 | 2 | – | 1 | 2 |
| urban_density | 0 | 2 | 3 | 0 | 1 | – | 1 | 3 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_