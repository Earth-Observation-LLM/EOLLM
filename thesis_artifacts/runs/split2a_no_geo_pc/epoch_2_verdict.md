# Epoch 2 Verdict

**Step:** 354 | **Eval time:** 1724s | **Samples:** 6984

## Overall: 50.4% (3517/6984)

**vs Base model:** 18.0% → 50.4% (**+32.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 36.2% | 203/561 | -23.8% | +11.2% (rand=25%) | +4.8% (maj=31%) |
| building_height | 44.4% | 116/261 | +44.4% | +19.4% (rand=25%) | +16.5% (maj=28%) |
| camera_direction | 26.4% | 148/561 | +26.4% | +1.4% (rand=25%) | +0.0% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 58.3% | 261/448 | +33.3% | +33.3% (rand=25%) | +30.4% (maj=28%) |
| land_use | 72.9% | 409/561 | +52.9% | +47.9% (rand=25%) | +44.7% (maj=28%) |
| mismatch_binary_easy | 51.7% | 290/561 | +41.7% | +1.7% (rand=50%) | -0.4% (maj=52%) |
| mismatch_binary_hard | 49.6% | 278/561 | +49.6% | -0.4% (rand=50%) | -2.7% (maj=52%) |
| mismatch_mcq_easy | 26.6% | 149/561 | +26.6% | +1.6% (rand=25%) | -0.2% (maj=27%) |
| mismatch_mcq_hard | 25.1% | 141/561 | +12.6% | +0.1% (rand=25%) | -0.7% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 66.8% | 375/561 | +52.6% | +41.8% (rand=25%) | +41.0% (maj=26%) |
| transit_density | 34.9% | 196/561 | +34.9% | +9.9% (rand=25%) | +6.6% (maj=28%) |
| urban_density | 55.1% | 309/561 | +45.1% | +30.1% (rand=25%) | +29.1% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.5% | +41.5% |
| 2 | 50.4% | -9.1% |

**WARNING: Accuracy dropped 59.5% → 50.4%. Possible overfitting.**
