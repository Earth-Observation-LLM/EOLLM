# Epoch 3 Verdict

**Step:** 717 | **Eval time:** 1716s | **Samples:** 6984

## Overall: 75.1% (5242/6984)

**vs Base model:** 18.0% → 75.1% (**+57.1%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 58.3% | 327/561 | -1.7% | +33.3% (rand=25%) | +26.9% (maj=31%) |
| building_height | 62.5% | 163/261 | +62.5% | +37.5% (rand=25%) | +34.5% (maj=28%) |
| camera_direction | 39.9% | 224/561 | +39.9% | +14.9% (rand=25%) | +13.5% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 74.6% | 334/448 | +49.6% | +49.6% (rand=25%) | +46.7% (maj=28%) |
| land_use | 79.9% | 448/561 | +59.9% | +54.9% (rand=25%) | +51.7% (maj=28%) |
| mismatch_binary_easy | 96.8% | 543/561 | +86.8% | +46.8% (rand=50%) | +44.7% (maj=52%) |
| mismatch_binary_hard | 80.6% | 452/561 | +80.6% | +30.6% (rand=50%) | +28.3% (maj=52%) |
| mismatch_mcq_easy | 97.0% | 544/561 | +97.0% | +72.0% (rand=25%) | +70.2% (maj=27%) |
| mismatch_mcq_hard | 83.1% | 466/561 | +70.6% | +58.1% (rand=25%) | +57.2% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 73.3% | 411/561 | +59.0% | +48.3% (rand=25%) | +47.4% (maj=26%) |
| transit_density | 50.4% | 283/561 | +50.4% | +25.4% (rand=25%) | +22.1% (maj=28%) |
| urban_density | 72.2% | 405/561 | +62.2% | +47.2% (rand=25%) | +46.2% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 68.5% | +50.5% |
| 2 | 73.3% | +4.9% |
| 3 | 75.1% | +1.7% |

**Model still improving.** (+1.7% from previous epoch)
