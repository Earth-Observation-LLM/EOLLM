# Epoch 2 Verdict

**Step:** 546 | **Eval time:** 2179s | **Samples:** 8831

## Overall: 66.0% (5826/8831)

**vs Base model:** 19.0% → 66.0% (**+47.0%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 35.7% | 264/739 | -4.3% | +10.7% (rand=25%) | +9.3% (maj=26%) |
| building_height | 43.8% | 119/272 | +43.8% | +18.8% (rand=25%) | +15.1% (maj=29%) |
| camera_direction | 28.7% | 212/739 | +28.7% | +3.7% (rand=25%) | +2.0% (maj=27%) |
| green_space | 42.4% | 95/224 | +42.4% | +17.4% (rand=25%) | +16.5% (maj=26%) |
| junction_type | 68.3% | 353/517 | +51.6% | +43.3% (rand=25%) | +40.4% (maj=28%) |
| land_use | 72.9% | 539/739 | +72.9% | +47.9% (rand=25%) | +45.6% (maj=27%) |
| mismatch_binary_easy | 93.5% | 691/739 | +71.3% | +43.5% (rand=50%) | +43.0% (maj=50%) |
| mismatch_binary_hard | 81.2% | 600/739 | +56.2% | +31.2% (rand=50%) | +30.0% (maj=51%) |
| mismatch_mcq_easy | 95.8% | 708/739 | +70.8% | +70.8% (rand=25%) | +69.3% (maj=27%) |
| mismatch_mcq_hard | 82.9% | 613/739 | +82.9% | +57.9% (rand=25%) | +56.6% (maj=26%) |
| road_surface | 68.7% | 294/428 | -9.1% | +43.7% (rand=25%) | +40.7% (maj=28%) |
| road_type | 72.3% | 534/739 | +72.3% | +47.3% (rand=25%) | +45.3% (maj=27%) |
| transit_density | 38.7% | 286/739 | +11.4% | +13.7% (rand=25%) | +13.1% (maj=26%) |
| urban_density | 70.1% | 518/739 | +70.1% | +45.1% (rand=25%) | +43.8% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |

**Plateauing.** (67.2% → 66.0%)
