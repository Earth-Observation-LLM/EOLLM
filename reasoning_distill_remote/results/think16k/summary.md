# Ablation summary — think16k
_started 2026-06-18 13:06:05  •  updated 2026-06-18 18:28:47_

**Progress:** 12540 done / 12540  •  errors 0  •  ctx-overflow 0  •  parse-retried 1110  •  unparsed 143
**Overall acc:** 47.18%  •  median latency 94.8s  •  median completion 1160 tok
**Throughput:** 38.9 gen/min  •  ETA 0.0 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 36.3% | 37.3% | 34.9% | 20.9% | 421 |
| building_height | 62.6% | 51.6% | 61.6% | 38.4% | 190 |
| green_space | 41.4% | 38.4% | 49.8% | 59.1% | 203 |
| junction_type | 57.5% | 54.8% | 59.0% | 32.3% | 334 |
| land_use | 51.8% | 45.1% | 51.5% | 51.5% | 421 |
| road_surface | 83.2% | 75.2% | 84.8% | 94.4% | 303 |
| road_type | 58.4% | 50.4% | 64.4% | 34.4% | 421 |
| transit_density | 31.6% | 33.3% | 31.6% | 26.6% | 421 |
| urban_density | 38.7% | 38.0% | 43.9% | 32.3% | 421 |

## Thinking / truncation

| mode | n | median tok | p90 tok | max tok | truncated |
|---|---|---|---|---|---|
| full | 3135 | 1380 | 4205 | 15299 | 33 (1.1%) |
| sat_only | 3135 | 1331 | 3548 | 17849 | 23 (0.7%) |
| sv_only | 3135 | 1221 | 3563 | 16101 | 53 (1.7%) |
| blind | 3135 | 1000 | 4205 | 16098 | 34 (1.1%) |

## Visual-dependency buckets (full vs blind; sat vs sv)

| topic | visual | leaked | hard | regress | sat-solves | sv-solves | both | neither |
|---|---|---|---|---|---|---|---|---|
| amenity_richness | 131 | 22 | 202 | 66 | 55 | 45 | 102 | 219 |
| building_height | 68 | 51 | 49 | 22 | 28 | 47 | 70 | 45 |
| green_space | 6 | 78 | 77 | 42 | 4 | 27 | 74 | 98 |
| junction_type | 144 | 48 | 82 | 60 | 44 | 58 | 139 | 93 |
| land_use | 90 | 128 | 114 | 89 | 42 | 69 | 148 | 162 |
| road_surface | 10 | 242 | 7 | 44 | 19 | 48 | 209 | 27 |
| road_type | 138 | 108 | 138 | 37 | 36 | 95 | 176 | 114 |
| transit_density | 93 | 40 | 216 | 72 | 46 | 39 | 94 | 242 |
| urban_density | 105 | 58 | 180 | 78 | 34 | 59 | 126 | 202 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_