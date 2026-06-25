# Epoch 4 Verdict

**Step:** 792 | **Eval time:** 1713s | **Samples:** 6984

## Overall: 62.6% (4371/6984)

**vs Base model:** 18.0% → 62.6% (**+44.6%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 62.2% | 349/561 | +2.2% | +37.2% (rand=25%) | +30.8% (maj=31%) |
| building_height | 66.3% | 173/261 | +66.3% | +41.3% (rand=25%) | +38.3% (maj=28%) |
| camera_direction | 39.9% | 224/561 | +39.9% | +14.9% (rand=25%) | +13.5% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 75.4% | 338/448 | +50.4% | +50.4% (rand=25%) | +47.5% (maj=28%) |
| land_use | 80.7% | 453/561 | +60.7% | +55.7% (rand=25%) | +52.6% (maj=28%) |
| mismatch_binary_easy | 73.1% | 410/561 | +63.1% | +23.1% (rand=50%) | +21.0% (maj=52%) |
| mismatch_binary_hard | 66.0% | 370/561 | +66.0% | +16.0% (rand=50%) | +13.7% (maj=52%) |
| mismatch_mcq_easy | 30.8% | 173/561 | +30.8% | +5.8% (rand=25%) | +4.1% (maj=27%) |
| mismatch_mcq_hard | 29.2% | 164/561 | +16.7% | +4.2% (rand=25%) | +3.4% (maj=26%) |
| road_surface | 94.0% | 374/398 | +19.0% | +69.0% (rand=25%) | +64.8% (maj=29%) |
| road_type | 72.7% | 408/561 | +58.4% | +47.7% (rand=25%) | +46.9% (maj=26%) |
| transit_density | 46.2% | 259/561 | +46.2% | +21.2% (rand=25%) | +17.8% (maj=28%) |
| urban_density | 72.9% | 409/561 | +62.9% | +47.9% (rand=25%) | +46.9% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 57.8% | +39.8% |
| 2 | 59.3% | +1.4% |
| 3 | 62.7% | +3.5% |
| 4 | 62.6% | -0.2% |

**Plateauing.** (62.7% → 62.6%)
