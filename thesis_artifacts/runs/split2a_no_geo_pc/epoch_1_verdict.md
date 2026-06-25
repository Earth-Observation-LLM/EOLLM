# Epoch 1 Verdict

**Step:** 177 | **Eval time:** 1718s | **Samples:** 6984

## Overall: 59.5% (4153/6984)

**vs Base model:** 18.0% → 59.5% (**+41.5%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 50.4% | 283/561 | -9.6% | +25.4% (rand=25%) | +19.1% (maj=31%) |
| building_height | 60.5% | 158/261 | +60.5% | +35.5% (rand=25%) | +32.6% (maj=28%) |
| camera_direction | 24.1% | 135/561 | +24.1% | -0.9% (rand=25%) | -2.3% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 67.9% | 304/448 | +42.9% | +42.9% (rand=25%) | +40.0% (maj=28%) |
| land_use | 76.3% | 428/561 | +56.3% | +51.3% (rand=25%) | +48.1% (maj=28%) |
| mismatch_binary_easy | 80.9% | 454/561 | +70.9% | +30.9% (rand=50%) | +28.9% (maj=52%) |
| mismatch_binary_hard | 72.5% | 407/561 | +72.5% | +22.5% (rand=50%) | +20.3% (maj=52%) |
| mismatch_mcq_easy | 34.6% | 194/561 | +34.6% | +9.6% (rand=25%) | +7.8% (maj=27%) |
| mismatch_mcq_hard | 30.5% | 171/561 | +18.0% | +5.5% (rand=25%) | +4.6% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 72.0% | 404/561 | +57.7% | +47.0% (rand=25%) | +46.2% (maj=26%) |
| transit_density | 39.9% | 224/561 | +39.9% | +14.9% (rand=25%) | +11.6% (maj=28%) |
| urban_density | 62.2% | 349/561 | +52.2% | +37.2% (rand=25%) | +36.2% (maj=26%) |