# Epoch 5 Verdict

**Step:** 705 | **Eval time:** 1720s | **Samples:** 6984

## Overall: 66.3% (4631/6984)

**vs Base model:** 18.0% → 66.3% (**+48.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 38.0% | 213/561 | -22.0% | +13.0% (rand=25%) | +6.6% (maj=31%) |
| building_height | 52.1% | 136/261 | +52.1% | +27.1% (rand=25%) | +24.1% (maj=28%) |
| camera_direction | 46.0% | 258/561 | +46.0% | +21.0% (rand=25%) | +19.6% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 60.7% | 272/448 | +35.7% | +35.7% (rand=25%) | +32.8% (maj=28%) |
| land_use | 50.8% | 285/561 | +30.8% | +25.8% (rand=25%) | +22.6% (maj=28%) |
| mismatch_binary_easy | 97.1% | 545/561 | +87.1% | +47.1% (rand=50%) | +45.1% (maj=52%) |
| mismatch_binary_hard | 92.3% | 518/561 | +92.3% | +42.3% (rand=50%) | +40.1% (maj=52%) |
| mismatch_mcq_easy | 98.0% | 550/561 | +98.0% | +73.0% (rand=25%) | +71.3% (maj=27%) |
| mismatch_mcq_hard | 88.2% | 495/561 | +75.7% | +63.2% (rand=25%) | +62.4% (maj=26%) |
| road_surface | 94.0% | 374/398 | +19.0% | +69.0% (rand=25%) | +64.8% (maj=29%) |
| road_type | 55.3% | 310/561 | +41.0% | +30.3% (rand=25%) | +29.4% (maj=26%) |
| transit_density | 31.0% | 174/561 | +31.0% | +6.0% (rand=25%) | +2.7% (maj=28%) |
| urban_density | 41.7% | 234/561 | +31.7% | +16.7% (rand=25%) | +15.7% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.6% | +41.6% |
| 2 | 63.4% | +3.8% |
| 3 | 64.8% | +1.4% |
| 4 | 66.1% | +1.2% |
| 5 | 66.3% | +0.2% |

**Model still improving.** (+0.2% from previous epoch)
