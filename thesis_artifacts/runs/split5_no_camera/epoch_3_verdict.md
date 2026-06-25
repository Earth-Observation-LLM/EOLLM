# Epoch 3 Verdict

**Step:** 780 | **Eval time:** 1717s | **Samples:** 6984

## Overall: 74.9% (5229/6984)

**vs Base model:** 18.0% → 74.9% (**+56.9%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 55.6% | 312/561 | -4.4% | +30.6% (rand=25%) | +24.2% (maj=31%) |
| building_height | 61.7% | 161/261 | +61.7% | +36.7% (rand=25%) | +33.7% (maj=28%) |
| camera_direction | 26.2% | 147/561 | +26.2% | +1.2% (rand=25%) | -0.2% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 75.9% | 340/448 | +50.9% | +50.9% (rand=25%) | +48.0% (maj=28%) |
| land_use | 78.6% | 441/561 | +58.6% | +53.6% (rand=25%) | +50.4% (maj=28%) |
| mismatch_binary_easy | 96.1% | 539/561 | +86.1% | +46.1% (rand=50%) | +44.0% (maj=52%) |
| mismatch_binary_hard | 90.0% | 505/561 | +90.0% | +40.0% (rand=50%) | +37.8% (maj=52%) |
| mismatch_mcq_easy | 98.4% | 552/561 | +98.4% | +73.4% (rand=25%) | +71.7% (maj=27%) |
| mismatch_mcq_hard | 88.4% | 496/561 | +75.9% | +63.4% (rand=25%) | +62.6% (maj=26%) |
| road_surface | 94.5% | 376/398 | +19.5% | +69.5% (rand=25%) | +65.3% (maj=29%) |
| road_type | 73.6% | 413/561 | +59.3% | +48.6% (rand=25%) | +47.8% (maj=26%) |
| transit_density | 48.3% | 271/561 | +48.3% | +23.3% (rand=25%) | +20.0% (maj=28%) |
| urban_density | 72.9% | 409/561 | +62.9% | +47.9% (rand=25%) | +46.9% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 70.7% | +52.7% |
| 2 | 71.8% | +1.0% |
| 3 | 74.9% | +3.1% |

**Model still improving.** (+3.1% from previous epoch)
