# Epoch 4 Verdict

**Step:** 1348 | **Eval time:** 1744s | **Samples:** 6984

## Overall: 76.4% (5335/6984)

**vs Base model:** 18.0% → 76.4% (**+58.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 57.2% | 321/561 | -2.8% | +32.2% (rand=25%) | +25.8% (maj=31%) |
| building_height | 65.1% | 170/261 | +65.1% | +40.1% (rand=25%) | +37.2% (maj=28%) |
| camera_direction | 51.0% | 286/561 | +51.0% | +26.0% (rand=25%) | +24.6% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 76.6% | 343/448 | +51.6% | +51.6% (rand=25%) | +48.7% (maj=28%) |
| land_use | 77.0% | 432/561 | +57.0% | +52.0% (rand=25%) | +48.8% (maj=28%) |
| mismatch_binary_easy | 94.5% | 530/561 | +84.5% | +44.5% (rand=50%) | +42.4% (maj=52%) |
| mismatch_binary_hard | 88.8% | 498/561 | +88.8% | +38.8% (rand=50%) | +36.5% (maj=52%) |
| mismatch_mcq_easy | 98.2% | 551/561 | +98.2% | +73.2% (rand=25%) | +71.5% (maj=27%) |
| mismatch_mcq_hard | 88.2% | 495/561 | +75.7% | +63.2% (rand=25%) | +62.4% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 74.7% | 419/561 | +60.4% | +49.7% (rand=25%) | +48.8% (maj=26%) |
| transit_density | 47.2% | 265/561 | +47.2% | +22.2% (rand=25%) | +18.9% (maj=28%) |
| urban_density | 68.3% | 383/561 | +58.3% | +43.3% (rand=25%) | +42.2% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 71.5% | +53.5% |
| 2 | 72.3% | +0.8% |
| 3 | 75.0% | +2.7% |
| 4 | 76.4% | +1.4% |

**Model still improving.** (+1.4% from previous epoch)
