# Epoch 3 Verdict

**Step:** 819 | **Eval time:** 2191s | **Samples:** 8831

## Overall: 70.7% (6243/8831)

**vs Base model:** 19.0% → 70.7% (**+51.7%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.6% | 455/739 | +21.6% | +36.6% (rand=25%) | +35.2% (maj=26%) |
| building_height | 51.8% | 141/272 | +51.8% | +26.8% (rand=25%) | +23.2% (maj=29%) |
| camera_direction | 42.5% | 314/739 | +42.5% | +17.5% (rand=25%) | +15.8% (maj=27%) |
| green_space | 71.9% | 161/224 | +71.9% | +46.9% (rand=25%) | +46.0% (maj=26%) |
| junction_type | 67.5% | 349/517 | +50.8% | +42.5% (rand=25%) | +39.7% (maj=28%) |
| land_use | 69.1% | 511/739 | +69.1% | +44.1% (rand=25%) | +41.8% (maj=27%) |
| mismatch_binary_easy | 94.0% | 695/739 | +71.8% | +44.0% (rand=50%) | +43.6% (maj=50%) |
| mismatch_binary_hard | 86.3% | 638/739 | +61.3% | +36.3% (rand=50%) | +35.2% (maj=51%) |
| mismatch_mcq_easy | 95.3% | 704/739 | +70.3% | +70.3% (rand=25%) | +68.7% (maj=27%) |
| mismatch_mcq_hard | 81.9% | 605/739 | +81.9% | +56.9% (rand=25%) | +55.5% (maj=26%) |
| road_surface | 51.9% | 222/428 | -25.9% | +26.9% (rand=25%) | +23.8% (maj=28%) |
| road_type | 70.0% | 517/739 | +70.0% | +45.0% (rand=25%) | +43.0% (maj=27%) |
| transit_density | 49.5% | 366/739 | +22.3% | +24.5% (rand=25%) | +24.0% (maj=26%) |
| urban_density | 76.5% | 565/739 | +76.5% | +51.5% (rand=25%) | +50.2% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |
| 3 | 70.7% | +4.7% |

**Model still improving.** (+4.7% from previous epoch)
