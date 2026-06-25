# Epoch 5 Verdict

**Step:** 1685 | **Eval time:** 1747s | **Samples:** 6984

## Overall: 76.9% (5372/6984)

**vs Base model:** 18.0% → 76.9% (**+58.9%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 53.5% | 300/561 | -6.5% | +28.5% (rand=25%) | +22.1% (maj=31%) |
| building_height | 64.4% | 168/261 | +64.4% | +39.4% (rand=25%) | +36.4% (maj=28%) |
| camera_direction | 53.1% | 298/561 | +53.1% | +28.1% (rand=25%) | +26.7% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 70.1% | 314/448 | +45.1% | +45.1% (rand=25%) | +42.2% (maj=28%) |
| land_use | 77.5% | 435/561 | +57.5% | +52.5% (rand=25%) | +49.4% (maj=28%) |
| mismatch_binary_easy | 96.8% | 543/561 | +86.8% | +46.8% (rand=50%) | +44.7% (maj=52%) |
| mismatch_binary_hard | 92.0% | 516/561 | +92.0% | +42.0% (rand=50%) | +39.8% (maj=52%) |
| mismatch_mcq_easy | 98.0% | 550/561 | +98.0% | +73.0% (rand=25%) | +71.3% (maj=27%) |
| mismatch_mcq_hard | 90.6% | 508/561 | +78.1% | +65.6% (rand=25%) | +64.7% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 71.3% | 400/561 | +57.0% | +46.3% (rand=25%) | +45.5% (maj=26%) |
| transit_density | 48.5% | 272/561 | +48.5% | +23.5% (rand=25%) | +20.1% (maj=28%) |
| urban_density | 75.9% | 426/561 | +65.9% | +50.9% (rand=25%) | +49.9% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 71.5% | +53.5% |
| 2 | 72.3% | +0.8% |
| 3 | 75.0% | +2.7% |
| 4 | 76.4% | +1.4% |
| 5 | 76.9% | +0.5% |

**Model still improving.** (+0.5% from previous epoch)
