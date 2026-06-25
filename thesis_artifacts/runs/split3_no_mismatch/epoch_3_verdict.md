# Epoch 3 Verdict

**Step:** 594 | **Eval time:** 1706s | **Samples:** 6984

## Overall: 62.7% (4382/6984)

**vs Base model:** 18.0% → 62.7% (**+44.7%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.3% | 344/561 | +1.3% | +36.3% (rand=25%) | +29.9% (maj=31%) |
| building_height | 64.8% | 169/261 | +64.8% | +39.8% (rand=25%) | +36.8% (maj=28%) |
| camera_direction | 42.8% | 240/561 | +42.8% | +17.8% (rand=25%) | +16.4% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 74.6% | 334/448 | +49.6% | +49.6% (rand=25%) | +46.7% (maj=28%) |
| land_use | 77.9% | 437/561 | +57.9% | +52.9% (rand=25%) | +49.7% (maj=28%) |
| mismatch_binary_easy | 77.0% | 432/561 | +67.0% | +27.0% (rand=50%) | +25.0% (maj=52%) |
| mismatch_binary_hard | 69.0% | 387/561 | +69.0% | +19.0% (rand=50%) | +16.8% (maj=52%) |
| mismatch_mcq_easy | 29.8% | 167/561 | +29.8% | +4.8% (rand=25%) | +3.0% (maj=27%) |
| mismatch_mcq_hard | 26.6% | 149/561 | +14.1% | +1.6% (rand=25%) | +0.7% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 72.7% | 408/561 | +58.4% | +47.7% (rand=25%) | +46.9% (maj=26%) |
| transit_density | 47.4% | 266/561 | +47.4% | +22.4% (rand=25%) | +19.1% (maj=28%) |
| urban_density | 72.5% | 407/561 | +62.5% | +47.5% (rand=25%) | +46.5% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 57.8% | +39.8% |
| 2 | 59.3% | +1.4% |
| 3 | 62.7% | +3.5% |

**Model still improving.** (+3.5% from previous epoch)
