# Epoch 1 Verdict

**Step:** 239 | **Eval time:** 1711s | **Samples:** 6984

## Overall: 68.5% (4781/6984)

**vs Base model:** 18.0% → 68.5% (**+50.5%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 47.8% | 268/561 | -12.2% | +22.8% (rand=25%) | +16.4% (maj=31%) |
| building_height | 61.3% | 160/261 | +61.3% | +36.3% (rand=25%) | +33.3% (maj=28%) |
| camera_direction | 25.8% | 145/561 | +25.8% | +0.8% (rand=25%) | -0.5% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 65.8% | 295/448 | +40.8% | +40.8% (rand=25%) | +37.9% (maj=28%) |
| land_use | 77.4% | 434/561 | +57.4% | +52.4% (rand=25%) | +49.2% (maj=28%) |
| mismatch_binary_easy | 93.2% | 523/561 | +83.2% | +43.2% (rand=50%) | +41.2% (maj=52%) |
| mismatch_binary_hard | 75.8% | 425/561 | +75.8% | +25.8% (rand=50%) | +23.5% (maj=52%) |
| mismatch_mcq_easy | 93.0% | 522/561 | +93.0% | +68.0% (rand=25%) | +66.3% (maj=27%) |
| mismatch_mcq_hard | 76.3% | 428/561 | +63.8% | +51.3% (rand=25%) | +50.4% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 66.7% | 374/561 | +52.4% | +41.7% (rand=25%) | +40.8% (maj=26%) |
| transit_density | 39.8% | 223/561 | +39.8% | +14.8% (rand=25%) | +11.4% (maj=28%) |
| urban_density | 61.0% | 342/561 | +51.0% | +36.0% (rand=25%) | +34.9% (maj=26%) |