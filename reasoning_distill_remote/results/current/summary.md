# Ablation summary — current
_started 2026-06-18 12:58:22  •  updated 2026-06-18 13:03:23_

**Progress:** 12540 done / 12540  •  errors 0  •  ctx-overflow 0  •  parse-retried 0  •  unparsed 0
**Overall acc:** 46.19%  •  median latency 6.2s  •  median completion 7 tok
**Throughput:** 2429.0 gen/min  •  ETA 0.1 min

## Accuracy — topic × mode

| topic | full | sat_only | sv_only | blind | n |
|---|---|---|---|---|---|
| amenity_richness | 36.8% | 36.1% | 33.5% | 19.5% | 421 |
| building_height | 61.1% | 47.9% | 64.7% | 45.8% | 190 |
| green_space | 36.5% | 35.5% | 37.9% | 64.5% | 203 |
| junction_type | 62.6% | 48.8% | 62.6% | 48.5% | 334 |
| land_use | 53.0% | 50.1% | 52.7% | 43.5% | 421 |
| road_surface | 82.2% | 78.9% | 87.1% | 67.7% | 303 |
| road_type | 47.5% | 40.1% | 57.5% | 34.9% | 421 |
| transit_density | 32.1% | 30.9% | 30.4% | 27.3% | 421 |
| urban_density | 38.5% | 39.2% | 44.9% | 40.4% | 421 |

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
| amenity_richness | 126 | 29 | 213 | 53 | 55 | 44 | 97 | 225 |
| building_height | 47 | 69 | 56 | 18 | 19 | 51 | 72 | 48 |
| green_space | 5 | 69 | 67 | 62 | 3 | 8 | 69 | 123 |
| junction_type | 80 | 129 | 92 | 33 | 39 | 85 | 124 | 86 |
| land_use | 110 | 113 | 128 | 70 | 44 | 55 | 167 | 155 |
| road_surface | 68 | 181 | 30 | 24 | 19 | 44 | 220 | 20 |
| road_type | 90 | 110 | 184 | 37 | 22 | 95 | 147 | 157 |
| transit_density | 32 | 103 | 274 | 12 | 30 | 28 | 100 | 263 |
| urban_density | 92 | 70 | 159 | 100 | 36 | 60 | 129 | 196 |

_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_