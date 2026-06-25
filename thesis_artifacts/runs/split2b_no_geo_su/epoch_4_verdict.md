# Epoch 4 Verdict

**Step:** 708 | **Eval time:** 2186s | **Samples:** 8831

## Overall: 60.9% (5374/8831)

**vs Base model:** 19.0% → 60.9% (**+41.9%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 60.4% | 446/739 | +20.4% | +35.4% (rand=25%) | +34.0% (maj=26%) |
| building_height | 53.7% | 146/272 | +53.7% | +28.7% (rand=25%) | +25.0% (maj=29%) |
| camera_direction | 26.8% | 198/739 | +26.8% | +1.8% (rand=25%) | +0.1% (maj=27%) |
| green_space | 100.0% | 224/224 | +100.0% | +75.0% (rand=25%) | +74.1% (maj=26%) |
| junction_type | 69.4% | 359/517 | +52.8% | +44.4% (rand=25%) | +41.6% (maj=28%) |
| land_use | 75.6% | 559/739 | +75.6% | +50.6% (rand=25%) | +48.3% (maj=27%) |
| mismatch_binary_easy | 80.0% | 591/739 | +57.8% | +30.0% (rand=50%) | +29.5% (maj=50%) |
| mismatch_binary_hard | 69.7% | 515/739 | +44.7% | +19.7% (rand=50%) | +18.5% (maj=51%) |
| mismatch_mcq_easy | 35.0% | 259/739 | +10.0% | +10.0% (rand=25%) | +8.5% (maj=27%) |
| mismatch_mcq_hard | 30.7% | 227/739 | +30.7% | +5.7% (rand=25%) | +4.3% (maj=26%) |
| road_surface | 98.1% | 420/428 | +20.4% | +73.1% (rand=25%) | +70.1% (maj=28%) |
| road_type | 71.3% | 527/739 | +71.3% | +46.3% (rand=25%) | +44.4% (maj=27%) |
| transit_density | 48.8% | 361/739 | +21.6% | +23.8% (rand=25%) | +23.3% (maj=26%) |
| urban_density | 73.3% | 542/739 | +73.3% | +48.3% (rand=25%) | +47.1% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 58.9% | +39.9% |
| 2 | 59.0% | +0.1% |
| 3 | 61.8% | +2.8% |
| 4 | 60.9% | -0.9% |

**Plateauing.** (61.8% → 60.9%)
