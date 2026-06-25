# Epoch 5 Verdict

**Step:** 1195 | **Eval time:** 1681s | **Samples:** 6984

## Overall: 76.4% (5339/6984)

**vs Base model:** 18.0% → 76.4% (**+58.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.9% | 347/561 | +1.9% | +36.9% (rand=25%) | +30.5% (maj=31%) |
| building_height | 68.2% | 178/261 | +68.2% | +43.2% (rand=25%) | +40.2% (maj=28%) |
| camera_direction | 44.2% | 248/561 | +44.2% | +19.2% (rand=25%) | +17.8% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 75.9% | 340/448 | +50.9% | +50.9% (rand=25%) | +48.0% (maj=28%) |
| land_use | 80.7% | 453/561 | +60.7% | +55.7% (rand=25%) | +52.6% (maj=28%) |
| mismatch_binary_easy | 97.3% | 546/561 | +87.3% | +47.3% (rand=50%) | +45.3% (maj=52%) |
| mismatch_binary_hard | 82.7% | 464/561 | +82.7% | +32.7% (rand=50%) | +30.5% (maj=52%) |
| mismatch_mcq_easy | 97.5% | 547/561 | +97.5% | +72.5% (rand=25%) | +70.8% (maj=27%) |
| mismatch_mcq_hard | 81.6% | 458/561 | +69.1% | +56.6% (rand=25%) | +55.8% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 73.6% | 413/561 | +59.3% | +48.6% (rand=25%) | +47.8% (maj=26%) |
| transit_density | 51.0% | 286/561 | +51.0% | +26.0% (rand=25%) | +22.6% (maj=28%) |
| urban_density | 74.3% | 417/561 | +64.3% | +49.3% (rand=25%) | +48.3% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 68.5% | +50.5% |
| 2 | 73.3% | +4.9% |
| 3 | 75.1% | +1.7% |
| 4 | 76.3% | +1.2% |
| 5 | 76.4% | +0.2% |

**Model still improving.** (+0.2% from previous epoch)
