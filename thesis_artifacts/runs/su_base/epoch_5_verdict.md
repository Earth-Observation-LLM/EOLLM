# Epoch 5 Verdict

**Step:** 1365 | **Eval time:** 2149s | **Samples:** 8831

## Overall: 71.9% (6348/8831)

**vs Base model:** 19.0% → 71.9% (**+52.9%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 54.5% | 403/739 | +14.5% | +29.5% (rand=25%) | +28.1% (maj=26%) |
| building_height | 54.0% | 147/272 | +54.0% | +29.0% (rand=25%) | +25.4% (maj=29%) |
| camera_direction | 55.5% | 410/739 | +55.5% | +30.5% (rand=25%) | +28.8% (maj=27%) |
| green_space | 71.0% | 159/224 | +71.0% | +46.0% (rand=25%) | +45.1% (maj=26%) |
| junction_type | 75.6% | 391/517 | +59.0% | +50.6% (rand=25%) | +47.8% (maj=28%) |
| land_use | 74.7% | 552/739 | +74.7% | +49.7% (rand=25%) | +47.4% (maj=27%) |
| mismatch_binary_easy | 96.2% | 711/739 | +74.0% | +46.2% (rand=50%) | +45.7% (maj=50%) |
| mismatch_binary_hard | 89.0% | 658/739 | +64.0% | +39.0% (rand=50%) | +37.9% (maj=51%) |
| mismatch_mcq_easy | 98.1% | 725/739 | +73.1% | +73.1% (rand=25%) | +71.6% (maj=27%) |
| mismatch_mcq_hard | 87.7% | 648/739 | +87.7% | +62.7% (rand=25%) | +61.3% (maj=26%) |
| road_surface | 30.6% | 131/428 | -47.2% | +5.6% (rand=25%) | +2.6% (maj=28%) |
| road_type | 73.1% | 540/739 | +73.1% | +48.1% (rand=25%) | +46.1% (maj=27%) |
| transit_density | 47.2% | 349/739 | +20.0% | +22.2% (rand=25%) | +21.7% (maj=26%) |
| urban_density | 70.9% | 524/739 | +70.9% | +45.9% (rand=25%) | +44.7% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |
| 3 | 70.7% | +4.7% |
| 4 | 72.4% | +1.7% |
| 5 | 71.9% | -0.6% |

**Plateauing.** (72.4% → 71.9%)
