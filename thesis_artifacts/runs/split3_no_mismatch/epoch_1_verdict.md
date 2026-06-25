# Epoch 1 Verdict

**Step:** 198 | **Eval time:** 1703s | **Samples:** 6984

## Overall: 57.8% (4040/6984)

**vs Base model:** 18.0% → 57.8% (**+39.8%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 52.6% | 295/561 | -7.4% | +27.6% (rand=25%) | +21.2% (maj=31%) |
| building_height | 49.8% | 130/261 | +49.8% | +24.8% (rand=25%) | +21.8% (maj=28%) |
| camera_direction | 23.0% | 129/561 | +23.0% | -2.0% (rand=25%) | -3.4% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 66.1% | 296/448 | +41.1% | +41.1% (rand=25%) | +38.2% (maj=28%) |
| land_use | 76.1% | 427/561 | +56.1% | +51.1% (rand=25%) | +48.0% (maj=28%) |
| mismatch_binary_easy | 75.0% | 421/561 | +65.0% | +25.0% (rand=50%) | +23.0% (maj=52%) |
| mismatch_binary_hard | 67.0% | 376/561 | +67.0% | +17.0% (rand=50%) | +14.8% (maj=52%) |
| mismatch_mcq_easy | 28.7% | 161/561 | +28.7% | +3.7% (rand=25%) | +2.0% (maj=27%) |
| mismatch_mcq_hard | 27.6% | 155/561 | +15.1% | +2.6% (rand=25%) | +1.8% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 70.9% | 398/561 | +56.7% | +45.9% (rand=25%) | +45.1% (maj=26%) |
| transit_density | 45.6% | 256/561 | +45.6% | +20.6% (rand=25%) | +17.3% (maj=28%) |
| urban_density | 63.1% | 354/561 | +53.1% | +38.1% (rand=25%) | +37.1% (maj=26%) |