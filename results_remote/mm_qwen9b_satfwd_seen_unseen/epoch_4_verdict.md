# Epoch 4 Verdict

**Step:** 1816 | **Eval time:** 2220s | **Samples:** 8831

## Overall: 75.2% (6640/8831)

**vs Base model:** 42.0% → 75.2% (**+33.2%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 62.5% | 462/739 | +42.5% | +37.5% (rand=25%) | +36.1% (maj=26%) |
| building_height | 58.8% | 160/272 | +8.8% | +33.8% (rand=25%) | +30.1% (maj=29%) |
| camera_direction | 45.1% | 333/739 | -4.9% | +20.1% (rand=25%) | +18.4% (maj=27%) |
| green_space | 72.3% | 162/224 | +22.3% | +47.3% (rand=25%) | +46.4% (maj=26%) |
| junction_type | 73.3% | 379/517 | +23.3% | +48.3% (rand=25%) | +45.5% (maj=28%) |
| land_use | 73.3% | 542/739 | +37.0% | +48.3% (rand=25%) | +46.0% (maj=27%) |
| mismatch_binary_easy | 94.5% | 698/739 | +38.9% | +44.5% (rand=50%) | +44.0% (maj=50%) |
| mismatch_binary_hard | 91.2% | 674/739 | +28.7% | +41.2% (rand=50%) | +40.1% (maj=51%) |
| mismatch_mcq_easy | 97.8% | 723/739 | +97.8% | +72.8% (rand=25%) | +71.3% (maj=27%) |
| mismatch_mcq_hard | 84.7% | 626/739 | +59.7% | +59.7% (rand=25%) | +58.3% (maj=26%) |
| road_surface | 98.6% | 422/428 | -1.4% | +73.6% (rand=25%) | +70.6% (maj=28%) |
| road_type | 75.9% | 561/739 | +25.9% | +50.9% (rand=25%) | +49.0% (maj=27%) |
| transit_density | 48.3% | 357/739 | +11.9% | +23.3% (rand=25%) | +22.7% (maj=26%) |
| urban_density | 73.2% | 541/739 | +62.1% | +48.2% (rand=25%) | +47.0% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 69.2% | +27.2% |
| 2 | 73.3% | +4.1% |
| 3 | 76.0% | +2.7% |
| 4 | 75.2% | -0.8% |

**Plateauing.** (76.0% → 75.2%)
