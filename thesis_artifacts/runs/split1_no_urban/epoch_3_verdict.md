# Epoch 3 Verdict

**Step:** 423 | **Eval time:** 1725s | **Samples:** 6984

## Overall: 64.8% (4529/6984)

**vs Base model:** 18.0% → 64.8% (**+46.8%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 39.0% | 219/561 | -21.0% | +14.0% (rand=25%) | +7.7% (maj=31%) |
| building_height | 49.4% | 129/261 | +49.4% | +24.4% (rand=25%) | +21.5% (maj=28%) |
| camera_direction | 40.5% | 227/561 | +40.5% | +15.5% (rand=25%) | +14.1% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 59.2% | 265/448 | +34.2% | +34.2% (rand=25%) | +31.2% (maj=28%) |
| land_use | 52.9% | 297/561 | +32.9% | +27.9% (rand=25%) | +24.8% (maj=28%) |
| mismatch_binary_easy | 92.7% | 520/561 | +82.7% | +42.7% (rand=50%) | +40.6% (maj=52%) |
| mismatch_binary_hard | 87.5% | 491/561 | +87.5% | +37.5% (rand=50%) | +35.3% (maj=52%) |
| mismatch_mcq_easy | 97.1% | 545/561 | +97.1% | +72.1% (rand=25%) | +70.4% (maj=27%) |
| mismatch_mcq_hard | 86.1% | 483/561 | +73.6% | +61.1% (rand=25%) | +60.2% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 56.3% | 316/561 | +42.0% | +31.3% (rand=25%) | +30.5% (maj=26%) |
| transit_density | 31.6% | 177/561 | +31.6% | +6.6% (rand=25%) | +3.2% (maj=28%) |
| urban_density | 38.9% | 218/561 | +28.9% | +13.9% (rand=25%) | +12.8% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.6% | +41.6% |
| 2 | 63.4% | +3.8% |
| 3 | 64.8% | +1.4% |

**Model still improving.** (+1.4% from previous epoch)
