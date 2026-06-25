# Epoch 6 Verdict

**Step:** 2022 | **Eval time:** 1745s | **Samples:** 6984

## Overall: 78.3% (5469/6984)

**vs Base model:** 18.0% → 78.3% (**+60.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 62.0% | 348/561 | +2.0% | +37.0% (rand=25%) | +30.7% (maj=31%) |
| building_height | 69.0% | 180/261 | +69.0% | +44.0% (rand=25%) | +41.0% (maj=28%) |
| camera_direction | 56.5% | 317/561 | +56.5% | +31.5% (rand=25%) | +30.1% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 77.0% | 345/448 | +52.0% | +52.0% (rand=25%) | +49.1% (maj=28%) |
| land_use | 75.9% | 426/561 | +55.9% | +50.9% (rand=25%) | +47.8% (maj=28%) |
| mismatch_binary_easy | 96.6% | 542/561 | +86.6% | +46.6% (rand=50%) | +44.6% (maj=52%) |
| mismatch_binary_hard | 90.2% | 506/561 | +90.2% | +40.2% (rand=50%) | +38.0% (maj=52%) |
| mismatch_mcq_easy | 99.1% | 556/561 | +99.1% | +74.1% (rand=25%) | +72.4% (maj=27%) |
| mismatch_mcq_hard | 90.6% | 508/561 | +78.1% | +65.6% (rand=25%) | +64.7% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 71.3% | 400/561 | +57.0% | +46.3% (rand=25%) | +45.5% (maj=26%) |
| transit_density | 50.8% | 285/561 | +50.8% | +25.8% (rand=25%) | +22.5% (maj=28%) |
| urban_density | 73.8% | 414/561 | +63.8% | +48.8% (rand=25%) | +47.8% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 71.5% | +53.5% |
| 2 | 72.3% | +0.8% |
| 3 | 75.0% | +2.7% |
| 4 | 76.4% | +1.4% |
| 5 | 76.9% | +0.5% |
| 6 | 78.3% | +1.4% |

**Model still improving.** (+1.4% from previous epoch)
