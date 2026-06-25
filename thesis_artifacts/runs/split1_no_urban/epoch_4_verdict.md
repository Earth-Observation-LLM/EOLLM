# Epoch 4 Verdict

**Step:** 564 | **Eval time:** 1719s | **Samples:** 6984

## Overall: 66.1% (4615/6984)

**vs Base model:** 18.0% → 66.1% (**+48.1%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 38.5% | 216/561 | -21.5% | +13.5% (rand=25%) | +7.1% (maj=31%) |
| building_height | 50.2% | 131/261 | +50.2% | +25.2% (rand=25%) | +22.2% (maj=28%) |
| camera_direction | 41.9% | 235/561 | +41.9% | +16.9% (rand=25%) | +15.5% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 61.8% | 277/448 | +36.8% | +36.8% (rand=25%) | +33.9% (maj=28%) |
| land_use | 50.3% | 282/561 | +30.3% | +25.3% (rand=25%) | +22.1% (maj=28%) |
| mismatch_binary_easy | 98.0% | 550/561 | +88.0% | +48.0% (rand=50%) | +46.0% (maj=52%) |
| mismatch_binary_hard | 92.7% | 520/561 | +92.7% | +42.7% (rand=50%) | +40.5% (maj=52%) |
| mismatch_mcq_easy | 97.9% | 549/561 | +97.9% | +72.9% (rand=25%) | +71.1% (maj=27%) |
| mismatch_mcq_hard | 88.4% | 496/561 | +75.9% | +63.4% (rand=25%) | +62.6% (maj=26%) |
| road_surface | 93.7% | 373/398 | +18.7% | +68.7% (rand=25%) | +64.6% (maj=29%) |
| road_type | 55.1% | 309/561 | +40.8% | +30.1% (rand=25%) | +29.2% (maj=26%) |
| transit_density | 31.4% | 176/561 | +31.4% | +6.4% (rand=25%) | +3.0% (maj=28%) |
| urban_density | 41.7% | 234/561 | +31.7% | +16.7% (rand=25%) | +15.7% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.6% | +41.6% |
| 2 | 63.4% | +3.8% |
| 3 | 64.8% | +1.4% |
| 4 | 66.1% | +1.2% |

**Model still improving.** (+1.2% from previous epoch)
