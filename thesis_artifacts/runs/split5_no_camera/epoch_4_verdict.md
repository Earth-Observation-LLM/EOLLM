# Epoch 4 Verdict

**Step:** 1040 | **Eval time:** 1689s | **Samples:** 6984

## Overall: 75.9% (5304/6984)

**vs Base model:** 18.0% → 75.9% (**+57.9%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 59.0% | 331/561 | -1.0% | +34.0% (rand=25%) | +27.6% (maj=31%) |
| building_height | 65.5% | 171/261 | +65.5% | +40.5% (rand=25%) | +37.5% (maj=28%) |
| camera_direction | 29.8% | 167/561 | +29.8% | +4.8% (rand=25%) | +3.4% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 77.0% | 345/448 | +52.0% | +52.0% (rand=25%) | +49.1% (maj=28%) |
| land_use | 77.5% | 435/561 | +57.5% | +52.5% (rand=25%) | +49.4% (maj=28%) |
| mismatch_binary_easy | 97.1% | 545/561 | +87.1% | +47.1% (rand=50%) | +45.1% (maj=52%) |
| mismatch_binary_hard | 91.4% | 513/561 | +91.4% | +41.4% (rand=50%) | +39.2% (maj=52%) |
| mismatch_mcq_easy | 98.2% | 551/561 | +98.2% | +73.2% (rand=25%) | +71.5% (maj=27%) |
| mismatch_mcq_hard | 90.6% | 508/561 | +78.1% | +65.6% (rand=25%) | +64.7% (maj=26%) |
| road_surface | 94.5% | 376/398 | +19.5% | +69.5% (rand=25%) | +65.3% (maj=29%) |
| road_type | 73.4% | 412/561 | +59.2% | +48.4% (rand=25%) | +47.6% (maj=26%) |
| transit_density | 48.0% | 269/561 | +48.0% | +23.0% (rand=25%) | +19.6% (maj=28%) |
| urban_density | 73.8% | 414/561 | +63.8% | +48.8% (rand=25%) | +47.8% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 70.7% | +52.7% |
| 2 | 71.8% | +1.0% |
| 3 | 74.9% | +3.1% |
| 4 | 75.9% | +1.1% |

**Model still improving.** (+1.1% from previous epoch)
