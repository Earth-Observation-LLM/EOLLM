# Epoch 2 Verdict

**Step:** 478 | **Eval time:** 1716s | **Samples:** 6984

## Overall: 73.3% (5122/6984)

**vs Base model:** 18.0% → 73.3% (**+55.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 58.6% | 329/561 | -1.4% | +33.6% (rand=25%) | +27.3% (maj=31%) |
| building_height | 57.9% | 151/261 | +57.9% | +32.9% (rand=25%) | +29.9% (maj=28%) |
| camera_direction | 41.7% | 234/561 | +41.7% | +16.7% (rand=25%) | +15.3% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 72.1% | 323/448 | +47.1% | +47.1% (rand=25%) | +44.2% (maj=28%) |
| land_use | 76.3% | 428/561 | +56.3% | +51.3% (rand=25%) | +48.1% (maj=28%) |
| mismatch_binary_easy | 95.7% | 537/561 | +85.7% | +45.7% (rand=50%) | +43.7% (maj=52%) |
| mismatch_binary_hard | 80.0% | 449/561 | +80.0% | +30.0% (rand=50%) | +27.8% (maj=52%) |
| mismatch_mcq_easy | 95.5% | 536/561 | +95.5% | +70.5% (rand=25%) | +68.8% (maj=27%) |
| mismatch_mcq_hard | 76.8% | 431/561 | +64.3% | +51.8% (rand=25%) | +51.0% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 70.1% | 393/561 | +55.8% | +45.1% (rand=25%) | +44.2% (maj=26%) |
| transit_density | 47.6% | 267/561 | +47.6% | +22.6% (rand=25%) | +19.3% (maj=28%) |
| urban_density | 71.7% | 402/561 | +61.7% | +46.7% (rand=25%) | +45.6% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 68.5% | +50.5% |
| 2 | 73.3% | +4.9% |

**Model still improving.** (+4.9% from previous epoch)
