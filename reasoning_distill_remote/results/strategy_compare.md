# Strategy comparison — Qwen3.5-9B-AWQ on the 9 attribute tasks

_full = marked-sat + 4 SV angles · sat_only = sat · sv_only = SV grid · blind = no images · greedy (temp 0)_

## Overall accuracy — strategy × view

| strategy | full | sat_only | sv_only | blind | synergy (full−best1) | vision (full−blind) | n |
|---|---|---|---|---|---|---|---|
| current | 46.7 | 44.4 | 48.9 | 42.2 | -2.2 | +4.4 | 45 |
| think16k | 46.7 | 51.1 | 53.3 | 37.8 | -6.7 | +8.9 | 45 |
| multistep | 51.1 | 55.6 | 62.2 | 44.4 | -11.1 | +6.7 | 45 |

_synergy > 0 ⇒ the strategy makes the SECOND view add accuracy a single view can't._

## `full` accuracy by topic — across strategies

| topic | current | think16k | multistep |
|---|---|---|---|
| land_use | 60.0 | 60.0 | 60.0 |
| building_height | 40.0 | 60.0 | 60.0 |
| urban_density | 20.0 | 40.0 | 40.0 |
| junction_type | 80.0 | 60.0 | 60.0 |
| green_space | 0.0 | 0.0 | 20.0 |
| amenity_richness | 40.0 | 60.0 | 60.0 |
| road_type | 40.0 | 40.0 | 40.0 |
| road_surface | 100.0 | 80.0 | 80.0 |
| transit_density | 40.0 | 20.0 | 40.0 |

## current — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 60.0 | 40.0 | 60.0 | 40.0 | +0.0 |
| building_height | 40.0 | 40.0 | 60.0 | 40.0 | -20.0 |
| urban_density | 20.0 | 40.0 | 20.0 | 40.0 | -20.0 |
| junction_type | 80.0 | 60.0 | 100.0 | 80.0 | -20.0 |
| green_space | 0.0 | 0.0 | 0.0 | 60.0 | +0.0 |
| amenity_richness | 40.0 | 40.0 | 40.0 | 0.0 | +0.0 |
| road_type | 40.0 | 40.0 | 20.0 | 20.0 | +0.0 |
| road_surface | 100.0 | 80.0 | 100.0 | 80.0 | +0.0 |
| transit_density | 40.0 | 60.0 | 40.0 | 20.0 | -20.0 |

## think16k — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 60.0 | 60.0 | 60.0 | 60.0 | +0.0 |
| building_height | 60.0 | 60.0 | 60.0 | 40.0 | +0.0 |
| urban_density | 40.0 | 40.0 | 20.0 | 40.0 | +0.0 |
| junction_type | 60.0 | 80.0 | 100.0 | 60.0 | -40.0 |
| green_space | 0.0 | 0.0 | 20.0 | 20.0 | -20.0 |
| amenity_richness | 60.0 | 60.0 | 40.0 | 0.0 | +0.0 |
| road_type | 40.0 | 40.0 | 60.0 | 0.0 | -20.0 |
| road_surface | 80.0 | 80.0 | 100.0 | 80.0 | -20.0 |
| transit_density | 20.0 | 40.0 | 20.0 | 40.0 | -20.0 |

## multistep — topic × view (synergy detail)

| topic | full | sat_only | sv_only | blind | full−best1 |
|---|---|---|---|---|---|
| land_use | 60.0 | 80.0 | 60.0 | 40.0 | -20.0 |
| building_height | 60.0 | 40.0 | 60.0 | 40.0 | +0.0 |
| urban_density | 40.0 | 40.0 | 20.0 | 40.0 | +0.0 |
| junction_type | 60.0 | 80.0 | 100.0 | 40.0 | -40.0 |
| green_space | 20.0 | 40.0 | 60.0 | 100.0 | -40.0 |
| amenity_richness | 60.0 | 60.0 | 60.0 | 40.0 | +0.0 |
| road_type | 40.0 | 40.0 | 80.0 | 0.0 | -40.0 |
| road_surface | 80.0 | 60.0 | 100.0 | 80.0 | -20.0 |
| transit_density | 40.0 | 60.0 | 20.0 | 20.0 | -20.0 |

## Verdict

- **current**: full 46.7%  ·  synergy -2.2 (second view ~useless)
- **think16k**: full 46.7%  ·  synergy -6.7 (second view ~useless)  ·  full vs current baseline: +0.0
- **multistep**: full 51.1%  ·  synergy -11.1 (second view ~useless)  ·  full vs current baseline: +4.4

_Run `complementarity_gate.py results/<strategy>/ablation_log.jsonl --attr-only` for the formal PASS/FAIL synergy gate per strategy._