# EzelinyumEvaluator — Cross-model Comparison

Models evaluated: **4**. Benchmark: 5,240 held-out questions across 40 cities, 14 topics.

## Headline

| Rank | Model | Family | Acc | 95% CI | Hedged | Refused |
|---|---|---|---:|---|---:|---:|
| 1 | **PC-base — per_city LoRA (full 14 topics, 6 epochs)** | finetuned | 0.629 | [0.615, 0.642] | 0 | 0 |
| 2 | **SU-base — seen_unseen LoRA** | finetuned | 0.604 | [0.590, 0.617] | 0 | 0 |
| 3 | **Gemma-4 E2B-it (instruction-tuned, multimodal)** | baseline | 0.422 | [0.409, 0.436] | 0 | 0 |
| 4 | **Qwen3.5-4B base (untrained, thinking OFF)** | baseline | 0.355 | [0.342, 0.368] | 0 | 0 |

## Per-topic accuracy

| Topic | Rand | Maj | pc_base | su_base | gemma_4_e2b | qwen3_5_4b_base |
|---|---|---|---|---|---|---|
| amenity_richness | 0.25 | 0.29 | 0.53 | 0.57 | 0.33 | 0.25 |
| building_height | 0.25 | 0.28 | 0.66 | 0.61 | 0.48 | 0.25 |
| camera_direction | 0.25 | 0.29 | 0.24 | 0.26 | 0.24 | 0.08 |
| green_space | 0.25 | 0.32 | 1.00 | 0.63 | 0.54 | 0.41 |
| junction_type | 0.25 | 0.29 | 0.76 | 0.70 | 0.62 | 0.40 |
| land_use | 0.25 | 0.29 | 0.74 | 0.75 | 0.46 | 0.29 |
| mismatch_binary_easy | 0.50 | 0.51 | 0.53 | 0.56 | 0.52 | 0.52 |
| mismatch_binary_hard | 0.50 | 0.53 | 0.44 | 0.50 | 0.43 | 0.45 |
| mismatch_mcq_easy | 0.25 | 0.26 | 0.76 | 0.73 | 0.27 | 0.32 |
| mismatch_mcq_hard | 0.25 | 0.27 | 0.60 | 0.60 | 0.25 | 0.31 |
| road_surface | 0.25 | 0.27 | 0.96 | 0.75 | 0.93 | 0.72 |
| road_type | 0.25 | 0.27 | 0.71 | 0.74 | 0.40 | 0.46 |
| transit_density | 0.25 | 0.29 | 0.48 | 0.45 | 0.33 | 0.24 |
| urban_density | 0.25 | 0.28 | 0.71 | 0.70 | 0.38 | 0.37 |

## Seen vs Unseen cities

| Model | Seen acc | Unseen acc | Gap |
|---|---:|---:|---:|
| PC-base — per_city LoRA (full 14 topics, 6 epochs) | 0.638 | 0.589 | +0.050 |
| SU-base — seen_unseen LoRA | 0.606 | 0.595 | +0.010 |
| Gemma-4 E2B-it (instruction-tuned, multimodal) | 0.424 | 0.417 | +0.006 |
| Qwen3.5-4B base (untrained, thinking OFF) | 0.357 | 0.344 | +0.013 |

## Per-difficulty

| Model | easy | medium | hard |
|---|---:|---:|---:|
| PC-base — per_city LoRA (full 14 topics, 6 epochs) | 0.745 | 0.536 | 0.523 |
| SU-base — seen_unseen LoRA | 0.688 | 0.528 | 0.546 |
| Gemma-4 E2B-it (instruction-tuned, multimodal) | 0.494 | 0.371 | 0.343 |
| Qwen3.5-4B base (untrained, thinking OFF) | 0.426 | 0.261 | 0.380 |