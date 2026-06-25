# Epoch 3 Verdict

**Step:** 531 | **Eval time:** 1723s | **Samples:** 6984

## Overall: 50.7% (3538/6984)

**vs Base model:** 18.0% → 50.7% (**+32.7%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 36.9% | 207/561 | -23.1% | +11.9% (rand=25%) | +5.5% (maj=31%) |
| building_height | 43.7% | 114/261 | +43.7% | +18.7% (rand=25%) | +15.7% (maj=28%) |
| camera_direction | 26.2% | 147/561 | +26.2% | +1.2% (rand=25%) | -0.2% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 52.0% | 233/448 | +27.0% | +27.0% (rand=25%) | +24.1% (maj=28%) |
| land_use | 73.6% | 413/561 | +53.6% | +48.6% (rand=25%) | +45.5% (maj=28%) |
| mismatch_binary_easy | 54.5% | 306/561 | +44.5% | +4.5% (rand=50%) | +2.5% (maj=52%) |
| mismatch_binary_hard | 54.7% | 307/561 | +54.7% | +4.7% (rand=50%) | +2.5% (maj=52%) |
| mismatch_mcq_easy | 26.6% | 149/561 | +26.6% | +1.6% (rand=25%) | -0.2% (maj=27%) |
| mismatch_mcq_hard | 23.4% | 131/561 | +10.9% | -1.6% (rand=25%) | -2.5% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 67.6% | 379/561 | +53.3% | +42.6% (rand=25%) | +41.7% (maj=26%) |
| transit_density | 36.7% | 206/561 | +36.7% | +11.7% (rand=25%) | +8.4% (maj=28%) |
| urban_density | 54.2% | 304/561 | +44.2% | +29.2% (rand=25%) | +28.2% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.5% | +41.5% |
| 2 | 50.4% | -9.1% |
| 3 | 50.7% | +0.3% |

**Model still improving.** (+0.3% from previous epoch)
