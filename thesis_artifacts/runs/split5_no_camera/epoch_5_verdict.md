# Epoch 5 Verdict

**Step:** 1300 | **Eval time:** 1687s | **Samples:** 6984

## Overall: 76.5% (5345/6984)

**vs Base model:** 18.0% → 76.5% (**+58.5%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.0% | 342/561 | +1.0% | +36.0% (rand=25%) | +29.6% (maj=31%) |
| building_height | 65.1% | 170/261 | +65.1% | +40.1% (rand=25%) | +37.2% (maj=28%) |
| camera_direction | 29.9% | 168/561 | +29.9% | +4.9% (rand=25%) | +3.6% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 78.3% | 351/448 | +53.3% | +53.3% (rand=25%) | +50.4% (maj=28%) |
| land_use | 77.9% | 437/561 | +57.9% | +52.9% (rand=25%) | +49.7% (maj=28%) |
| mismatch_binary_easy | 97.3% | 546/561 | +87.3% | +47.3% (rand=50%) | +45.3% (maj=52%) |
| mismatch_binary_hard | 92.2% | 517/561 | +92.2% | +42.2% (rand=50%) | +39.9% (maj=52%) |
| mismatch_mcq_easy | 98.4% | 552/561 | +98.4% | +73.4% (rand=25%) | +71.7% (maj=27%) |
| mismatch_mcq_hard | 90.2% | 506/561 | +77.7% | +65.2% (rand=25%) | +64.3% (maj=26%) |
| road_surface | 94.5% | 376/398 | +19.5% | +69.5% (rand=25%) | +65.3% (maj=29%) |
| road_type | 74.0% | 415/561 | +59.7% | +49.0% (rand=25%) | +48.1% (maj=26%) |
| transit_density | 48.1% | 270/561 | +48.1% | +23.1% (rand=25%) | +19.8% (maj=28%) |
| urban_density | 76.3% | 428/561 | +66.3% | +51.3% (rand=25%) | +50.3% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 70.7% | +52.7% |
| 2 | 71.8% | +1.0% |
| 3 | 74.9% | +3.1% |
| 4 | 75.9% | +1.1% |
| 5 | 76.5% | +0.6% |

**Model still improving.** (+0.6% from previous epoch)
