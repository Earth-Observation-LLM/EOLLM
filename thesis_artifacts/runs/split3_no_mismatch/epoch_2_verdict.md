# Epoch 2 Verdict

**Step:** 396 | **Eval time:** 1702s | **Samples:** 6984

## Overall: 59.3% (4139/6984)

**vs Base model:** 18.0% → 59.3% (**+41.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 53.7% | 301/561 | -6.3% | +28.7% (rand=25%) | +22.3% (maj=31%) |
| building_height | 62.8% | 164/261 | +62.8% | +37.8% (rand=25%) | +34.9% (maj=28%) |
| camera_direction | 33.0% | 185/561 | +33.0% | +8.0% (rand=25%) | +6.6% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 71.4% | 320/448 | +46.4% | +46.4% (rand=25%) | +43.5% (maj=28%) |
| land_use | 76.5% | 429/561 | +56.5% | +51.5% (rand=25%) | +48.3% (maj=28%) |
| mismatch_binary_easy | 67.0% | 376/561 | +57.0% | +17.0% (rand=50%) | +15.0% (maj=52%) |
| mismatch_binary_hard | 59.5% | 334/561 | +59.5% | +9.5% (rand=50%) | +7.3% (maj=52%) |
| mismatch_mcq_easy | 31.4% | 176/561 | +31.4% | +6.4% (rand=25%) | +4.6% (maj=27%) |
| mismatch_mcq_hard | 28.5% | 160/561 | +16.0% | +3.5% (rand=25%) | +2.7% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 73.6% | 413/561 | +59.3% | +48.6% (rand=25%) | +47.8% (maj=26%) |
| transit_density | 44.6% | 250/561 | +44.6% | +19.6% (rand=25%) | +16.2% (maj=28%) |
| urban_density | 69.3% | 389/561 | +59.3% | +44.3% (rand=25%) | +43.3% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 57.8% | +39.8% |
| 2 | 59.3% | +1.4% |

**Model still improving.** (+1.4% from previous epoch)
