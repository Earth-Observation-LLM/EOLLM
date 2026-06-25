# Epoch 3 Verdict

**Step:** 531 | **Eval time:** 2184s | **Samples:** 8831

## Overall: 61.8% (5456/8831)

**vs Base model:** 19.0% → 61.8% (**+42.8%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 62.9% | 465/739 | +22.9% | +37.9% (rand=25%) | +36.5% (maj=26%) |
| building_height | 51.1% | 139/272 | +51.1% | +26.1% (rand=25%) | +22.4% (maj=29%) |
| camera_direction | 27.1% | 200/739 | +27.1% | +2.1% (rand=25%) | +0.4% (maj=27%) |
| green_space | 100.0% | 224/224 | +100.0% | +75.0% (rand=25%) | +74.1% (maj=26%) |
| junction_type | 72.9% | 377/517 | +56.3% | +47.9% (rand=25%) | +45.1% (maj=28%) |
| land_use | 74.0% | 547/739 | +74.0% | +49.0% (rand=25%) | +46.7% (maj=27%) |
| mismatch_binary_easy | 82.5% | 610/739 | +60.3% | +32.5% (rand=50%) | +32.1% (maj=50%) |
| mismatch_binary_hard | 71.7% | 530/739 | +46.7% | +21.7% (rand=50%) | +20.6% (maj=51%) |
| mismatch_mcq_easy | 33.8% | 250/739 | +8.8% | +8.8% (rand=25%) | +7.3% (maj=27%) |
| mismatch_mcq_hard | 31.4% | 232/739 | +31.4% | +6.4% (rand=25%) | +5.0% (maj=26%) |
| road_surface | 98.6% | 422/428 | +20.8% | +73.6% (rand=25%) | +70.6% (maj=28%) |
| road_type | 72.7% | 537/739 | +72.7% | +47.7% (rand=25%) | +45.7% (maj=27%) |
| transit_density | 49.1% | 363/739 | +21.8% | +24.1% (rand=25%) | +23.5% (maj=26%) |
| urban_density | 75.8% | 560/739 | +75.8% | +50.8% (rand=25%) | +49.5% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 58.9% | +39.9% |
| 2 | 59.0% | +0.1% |
| 3 | 61.8% | +2.8% |

**Model still improving.** (+2.8% from previous epoch)
