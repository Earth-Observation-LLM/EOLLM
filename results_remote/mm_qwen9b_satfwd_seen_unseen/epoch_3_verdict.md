# Epoch 3 Verdict

**Step:** 1362 | **Eval time:** 2210s | **Samples:** 8831

## Overall: 76.0% (6712/8831)

**vs Base model:** 42.0% → 76.0% (**+34.0%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 63.2% | 467/739 | +43.2% | +38.2% (rand=25%) | +36.8% (maj=26%) |
| building_height | 57.0% | 155/272 | +7.0% | +32.0% (rand=25%) | +28.3% (maj=29%) |
| camera_direction | 43.0% | 318/739 | -7.0% | +18.0% (rand=25%) | +16.4% (maj=27%) |
| green_space | 72.8% | 163/224 | +22.8% | +47.8% (rand=25%) | +46.9% (maj=26%) |
| junction_type | 74.5% | 385/517 | +24.5% | +49.5% (rand=25%) | +46.6% (maj=28%) |
| land_use | 75.0% | 554/739 | +38.6% | +50.0% (rand=25%) | +47.6% (maj=27%) |
| mismatch_binary_easy | 95.5% | 706/739 | +40.0% | +45.5% (rand=50%) | +45.1% (maj=50%) |
| mismatch_binary_hard | 92.0% | 680/739 | +29.5% | +42.0% (rand=50%) | +40.9% (maj=51%) |
| mismatch_mcq_easy | 98.1% | 725/739 | +98.1% | +73.1% (rand=25%) | +71.6% (maj=27%) |
| mismatch_mcq_hard | 86.1% | 636/739 | +61.1% | +61.1% (rand=25%) | +59.7% (maj=26%) |
| road_surface | 98.6% | 422/428 | -1.4% | +73.6% (rand=25%) | +70.6% (maj=28%) |
| road_type | 78.2% | 578/739 | +28.2% | +53.2% (rand=25%) | +51.3% (maj=27%) |
| transit_density | 51.3% | 379/739 | +14.9% | +26.3% (rand=25%) | +25.7% (maj=26%) |
| urban_density | 73.6% | 544/739 | +62.5% | +48.6% (rand=25%) | +47.4% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 69.2% | +27.2% |
| 2 | 73.3% | +4.1% |
| 3 | 76.0% | +2.7% |

**Model still improving.** (+2.7% from previous epoch)
