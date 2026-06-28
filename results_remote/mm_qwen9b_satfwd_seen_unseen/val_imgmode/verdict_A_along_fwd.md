# lora — Full Val Eval

**Model:** Qwen/Qwen3.5-9B + adapter /home/ain480/training/training/runs/mm_qwen9b_satfwd_r16_seen_unseen/lora

**Samples:** 8831 (full validation set)

**Config:** bs=32, workers=8, max_new_tokens=32, greedy, lenient-parse

**Eval time:** 1216s (20.3 min)


## Overall: 76.2% (6727/8831)


## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total |
|-------|----------|---------------|
| amenity_richness | 63.5% | 469/739 |
| building_height | 57.7% | 157/272 |
| camera_direction | 43.3% | 320/739 |
| green_space | 72.8% | 163/224 |
| junction_type | 74.7% | 386/517 |
| land_use | 75.4% | 557/739 |
| mismatch_binary_easy | 95.7% | 707/739 |
| mismatch_binary_hard | 92.0% | 680/739 |
| mismatch_mcq_easy | 98.1% | 725/739 |
| mismatch_mcq_hard | 85.9% | 635/739 |
| road_surface | 98.6% | 422/428 |
| road_type | 77.8% | 575/739 |
| transit_density | 52.4% | 387/739 |
| urban_density | 73.6% | 544/739 |

## Per-Difficulty Accuracy

| Difficulty | Accuracy | Correct/Total |
|------------|----------|---------------|
| easy | 85.2% | 3306/3880 |
| hard | 89.0% | 1315/1478 |
| medium | 60.6% | 2106/3473 |