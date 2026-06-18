# Ablation summary — multistep
_started 2026-06-18 18:31:33  •  updated 2026-06-18 19:52:46_

**Progress:** 12540 done / 12540  •  errors 0  •  ctx-overflow 0  •  parse-retried 0  •  unparsed 0
**Overall acc:** 49.57%  •  median latency 96.8s  •  median completion 591 tok
**Throughput:** 154.4 gen/min  •  ETA 0.0 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 37.1% | 38.7% | 36.1% | 25.9% | 421 |
| building_height | 61.1% | 54.7% | 58.4% | 43.2% | 190 |
| green_space | 64.5% | 62.1% | 65.0% | 68.5% | 203 |
| junction_type | 63.5% | 59.3% | 62.9% | 46.4% | 334 |
| land_use | 49.4% | 50.1% | 53.0% | 50.6% | 421 |
| road_surface | 89.8% | 80.9% | 89.4% | 83.8% | 303 |
| road_type | 53.4% | 48.5% | 63.2% | 33.3% | 421 |
| transit_density | 29.7% | 26.8% | 31.4% | 27.8% | 421 |
| urban_density | 41.6% | 43.7% | 45.4% | 35.9% | 421 |

## Thinking / truncation

| mode | n | median tok | p90 tok | max tok | truncated |
|---|---|---|---|---|---|
| full | 3135 | 0 | 0 | 0 | 0 (0.0%) |
| sat_only | 3135 | 0 | 0 | 0 | 0 (0.0%) |
| sv_only | 3135 | 0 | 0 | 0 | 0 (0.0%) |
| blind | 3135 | 0 | 0 | 0 | 0 (0.0%) |

## Visual-dependency buckets (full vs blind; sat vs sv)

| topic | visual | leaked | hard | regress | sat-solves | sv-solves | both | neither |
|---|---|---|---|---|---|---|---|---|
| amenity_richness | 111 | 45 | 201 | 64 | 64 | 53 | 99 | 205 |
| building_height | 51 | 65 | 57 | 17 | 28 | 35 | 76 | 51 |
| green_space | 43 | 88 | 21 | 51 | 34 | 40 | 92 | 37 |
| junction_type | 106 | 106 | 73 | 49 | 33 | 45 | 165 | 91 |
| land_use | 93 | 115 | 115 | 98 | 44 | 56 | 167 | 154 |
| road_surface | 40 | 232 | 9 | 22 | 22 | 48 | 223 | 10 |
| road_type | 136 | 89 | 145 | 51 | 40 | 102 | 164 | 115 |
| transit_density | 35 | 90 | 269 | 27 | 28 | 47 | 85 | 261 |
| urban_density | 94 | 81 | 176 | 70 | 46 | 53 | 138 | 184 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_