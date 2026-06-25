# Epoch 1 Verdict

**Step:** 141 | **Eval time:** 1719s | **Samples:** 6984

## Overall: 59.6% (4165/6984)

**vs Base model:** 18.0% → 59.6% (**+41.6%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 38.3% | 215/561 | -21.7% | +13.3% (rand=25%) | +7.0% (maj=31%) |
| building_height | 42.1% | 110/261 | +42.1% | +17.1% (rand=25%) | +14.2% (maj=28%) |
| camera_direction | 26.0% | 146/561 | +26.0% | +1.0% (rand=25%) | -0.4% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 54.5% | 244/448 | +29.5% | +29.5% (rand=25%) | +26.6% (maj=28%) |
| land_use | 44.4% | 249/561 | +24.4% | +19.4% (rand=25%) | +16.2% (maj=28%) |
| mismatch_binary_easy | 94.5% | 530/561 | +84.5% | +44.5% (rand=50%) | +42.4% (maj=52%) |
| mismatch_binary_hard | 84.0% | 471/561 | +84.0% | +34.0% (rand=50%) | +31.7% (maj=52%) |
| mismatch_mcq_easy | 93.6% | 525/561 | +93.6% | +68.6% (rand=25%) | +66.8% (maj=27%) |
| mismatch_mcq_hard | 79.3% | 445/561 | +66.8% | +54.3% (rand=25%) | +53.5% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 43.7% | 245/561 | +29.4% | +18.7% (rand=25%) | +17.8% (maj=26%) |
| transit_density | 30.3% | 170/561 | +30.3% | +5.3% (rand=25%) | +2.0% (maj=28%) |
| urban_density | 30.8% | 173/561 | +20.8% | +5.8% (rand=25%) | +4.8% (maj=26%) |