# Epoch 5 Verdict

**Step:** 990 | **Eval time:** 1687s | **Samples:** 6984

## Overall: 63.2% (4417/6984)

**vs Base model:** 18.0% → 63.2% (**+45.2%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 62.6% | 351/561 | +2.6% | +37.6% (rand=25%) | +31.2% (maj=31%) |
| building_height | 66.3% | 173/261 | +66.3% | +41.3% (rand=25%) | +38.3% (maj=28%) |
| camera_direction | 43.5% | 244/561 | +43.5% | +18.5% (rand=25%) | +17.1% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 76.6% | 343/448 | +51.6% | +51.6% (rand=25%) | +48.7% (maj=28%) |
| land_use | 79.0% | 443/561 | +59.0% | +54.0% (rand=25%) | +50.8% (maj=28%) |
| mismatch_binary_easy | 75.2% | 422/561 | +65.2% | +25.2% (rand=50%) | +23.2% (maj=52%) |
| mismatch_binary_hard | 65.2% | 366/561 | +65.2% | +15.2% (rand=50%) | +13.0% (maj=52%) |
| mismatch_mcq_easy | 31.4% | 176/561 | +31.4% | +6.4% (rand=25%) | +4.6% (maj=27%) |
| mismatch_mcq_hard | 29.4% | 165/561 | +16.9% | +4.4% (rand=25%) | +3.6% (maj=26%) |
| road_surface | 93.7% | 373/398 | +18.7% | +68.7% (rand=25%) | +64.6% (maj=29%) |
| road_type | 72.7% | 408/561 | +58.4% | +47.7% (rand=25%) | +46.9% (maj=26%) |
| transit_density | 49.0% | 275/561 | +49.0% | +24.0% (rand=25%) | +20.7% (maj=28%) |
| urban_density | 73.3% | 411/561 | +63.3% | +48.3% (rand=25%) | +47.2% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 57.8% | +39.8% |
| 2 | 59.3% | +1.4% |
| 3 | 62.7% | +3.5% |
| 4 | 62.6% | -0.2% |
| 5 | 63.2% | +0.7% |

**Model still improving.** (+0.7% from previous epoch)
