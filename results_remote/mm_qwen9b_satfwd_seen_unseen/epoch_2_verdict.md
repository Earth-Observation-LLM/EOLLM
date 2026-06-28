# Epoch 2 Verdict

**Step:** 908 | **Eval time:** 2249s | **Samples:** 8831

## Overall: 73.3% (6475/8831)

**vs Base model:** 42.0% → 73.3% (**+31.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 59.5% | 440/739 | +39.5% | +34.5% (rand=25%) | +33.2% (maj=26%) |
| building_height | 54.0% | 147/272 | +4.0% | +29.0% (rand=25%) | +25.4% (maj=29%) |
| camera_direction | 39.8% | 294/739 | -10.2% | +14.8% (rand=25%) | +13.1% (maj=27%) |
| green_space | 64.7% | 145/224 | +14.7% | +39.7% (rand=25%) | +38.8% (maj=26%) |
| junction_type | 71.6% | 370/517 | +21.6% | +46.6% (rand=25%) | +43.7% (maj=28%) |
| land_use | 70.8% | 523/739 | +34.4% | +45.8% (rand=25%) | +43.4% (maj=27%) |
| mismatch_binary_easy | 92.8% | 686/739 | +37.3% | +42.8% (rand=50%) | +42.4% (maj=50%) |
| mismatch_binary_hard | 89.4% | 661/739 | +26.9% | +39.4% (rand=50%) | +38.3% (maj=51%) |
| mismatch_mcq_easy | 94.5% | 698/739 | +94.5% | +69.5% (rand=25%) | +67.9% (maj=27%) |
| mismatch_mcq_hard | 82.0% | 606/739 | +57.0% | +57.0% (rand=25%) | +55.6% (maj=26%) |
| road_surface | 98.6% | 422/428 | -1.4% | +73.6% (rand=25%) | +70.6% (maj=28%) |
| road_type | 77.0% | 569/739 | +27.0% | +52.0% (rand=25%) | +50.1% (maj=27%) |
| transit_density | 51.4% | 380/739 | +15.1% | +26.4% (rand=25%) | +25.8% (maj=26%) |
| urban_density | 72.3% | 534/739 | +61.1% | +47.3% (rand=25%) | +46.0% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 69.2% | +27.2% |
| 2 | 73.3% | +4.1% |

**Model still improving.** (+4.1% from previous epoch)
