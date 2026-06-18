# Strategy comparison — Qwen3.5-9B-AWQ on the 9 attribute tasks

_full = marked-sat + 4 SV angles · sat_only = sat · sv_only = SV grid · blind = no images · greedy (temp 0)_

## Overall accuracy — strategy × view

| strategy | full | sat_only | sv_only | blind | synergy (full−best1) | vision (full−blind) | n |
|---|---|---|---|---|---|---|---|
| current | 48.6 | 44.4 | 50.9 | 40.9 | -2.3 | +7.7 | 3135 |
| think16k | 49.8 | 46.1 | 51.8 | 41.0 | -2.1 | +8.8 | 3135 |
| multistep | 51.7 | 49.4 | 53.8 | 43.4 | -2.2 | +8.3 | 3135 |

_synergy > 0 ⇒ the strategy makes the SECOND view add accuracy a single view can't._

## `full` accuracy by topic — across strategies

| topic | current | think16k | multistep |
|---|---|---|---|
| land_use | 53.0 | 51.8 | 49.4 |
| building_height | 61.1 | 62.6 | 61.1 |
| urban_density | 38.5 | 38.7 | 41.6 |
| junction_type | 62.6 | 57.5 | 63.5 |
| green_space | 36.5 | 41.4 | 64.5 |
| amenity_richness | 36.8 | 36.3 | 37.1 |
| road_type | 47.5 | 58.4 | 53.4 |
| road_surface | 82.2 | 83.2 | 89.8 |
| transit_density | 32.1 | 31.6 | 29.7 |

## current — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 53.0 | 50.1 | 52.7 | 43.5 | +0.2 |
| building_height | 61.1 | 47.9 | 64.7 | 45.8 | -3.7 |
| urban_density | 38.5 | 39.2 | 44.9 | 40.4 | -6.4 |
| junction_type | 62.6 | 48.8 | 62.6 | 48.5 | +0.0 |
| green_space | 36.5 | 35.5 | 37.9 | 64.5 | -1.5 |
| amenity_richness | 36.8 | 36.1 | 33.5 | 19.5 | +0.7 |
| road_type | 47.5 | 40.1 | 57.5 | 34.9 | -10.0 |
| road_surface | 82.2 | 78.9 | 87.1 | 67.7 | -5.0 |
| transit_density | 32.1 | 30.9 | 30.4 | 27.3 | +1.2 |

## think16k — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 51.8 | 45.1 | 51.5 | 51.5 | +0.2 |
| building_height | 62.6 | 51.6 | 61.6 | 38.4 | +1.1 |
| urban_density | 38.7 | 38.0 | 43.9 | 32.3 | -5.2 |
| junction_type | 57.5 | 54.8 | 59.0 | 32.3 | -1.5 |
| green_space | 41.4 | 38.4 | 49.8 | 59.1 | -8.4 |
| amenity_richness | 36.3 | 37.3 | 34.9 | 20.9 | -1.0 |
| road_type | 58.4 | 50.4 | 64.4 | 34.4 | -5.9 |
| road_surface | 83.2 | 75.2 | 84.8 | 94.4 | -1.7 |
| transit_density | 31.6 | 33.3 | 31.6 | 26.6 | -1.7 |

## multistep — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 49.4 | 50.1 | 53.0 | 50.6 | -3.6 |
| building_height | 61.1 | 54.7 | 58.4 | 43.2 | +2.6 |
| urban_density | 41.6 | 43.7 | 45.4 | 35.9 | -3.8 |
| junction_type | 63.5 | 59.3 | 62.9 | 46.4 | +0.6 |
| green_space | 64.5 | 62.1 | 65.0 | 68.5 | -0.5 |
| amenity_richness | 37.1 | 38.7 | 36.1 | 25.9 | -1.7 |
| road_type | 53.4 | 48.5 | 63.2 | 33.3 | -9.7 |
| road_surface | 89.8 | 80.9 | 89.4 | 83.8 | +0.3 |
| transit_density | 29.7 | 26.8 | 31.4 | 27.8 | -1.7 |

## Verdict

- **current**: full 48.6%  ·  synergy -2.3 (second view ~useless)
- **think16k**: full 49.8%  ·  synergy -2.1 (second view ~useless)  ·  full vs current baseline: +1.2
- **multistep**: full 51.7%  ·  synergy -2.2 (second view ~useless)  ·  full vs current baseline: +3.1

_Run `complementarity_gate.py results/<strategy>/ablation_log.jsonl --attr-only` for the formal PASS/FAIL synergy gate per strategy._