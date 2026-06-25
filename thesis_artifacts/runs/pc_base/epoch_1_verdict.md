# Epoch 1 Verdict

**Step:** 337 | **Eval time:** 1767s | **Samples:** 6984

## Overall: 71.5% (4993/6984)

**vs Base model:** 18.0% → 71.5% (**+53.5%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 49.2% | 276/561 | -10.8% | +24.2% (rand=25%) | +17.8% (maj=31%) |
| building_height | 62.5% | 163/261 | +62.5% | +37.5% (rand=25%) | +34.5% (maj=28%) |
| camera_direction | 25.3% | 142/561 | +25.3% | +0.3% (rand=25%) | -1.1% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 69.9% | 313/448 | +44.9% | +44.9% (rand=25%) | +42.0% (maj=28%) |
| land_use | 76.8% | 431/561 | +56.8% | +51.8% (rand=25%) | +48.7% (maj=28%) |
| mismatch_binary_easy | 91.6% | 514/561 | +81.6% | +41.6% (rand=50%) | +39.6% (maj=52%) |
| mismatch_binary_hard | 87.2% | 489/561 | +87.2% | +37.2% (rand=50%) | +34.9% (maj=52%) |
| mismatch_mcq_easy | 95.0% | 533/561 | +95.0% | +70.0% (rand=25%) | +68.3% (maj=27%) |
| mismatch_mcq_hard | 82.7% | 464/561 | +70.2% | +57.7% (rand=25%) | +56.9% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 73.3% | 411/561 | +59.0% | +48.3% (rand=25%) | +47.4% (maj=26%) |
| transit_density | 44.6% | 250/561 | +44.6% | +19.6% (rand=25%) | +16.2% (maj=28%) |
| urban_density | 65.1% | 365/561 | +55.1% | +40.1% (rand=25%) | +39.0% (maj=26%) |