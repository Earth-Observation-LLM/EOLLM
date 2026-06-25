# Epoch 2 Verdict

**Step:** 674 | **Eval time:** 1789s | **Samples:** 6984

## Overall: 72.3% (5048/6984)

**vs Base model:** 18.0% → 72.3% (**+54.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 49.2% | 276/561 | -10.8% | +24.2% (rand=25%) | +17.8% (maj=31%) |
| building_height | 60.2% | 157/261 | +60.2% | +35.2% (rand=25%) | +32.2% (maj=28%) |
| camera_direction | 35.1% | 197/561 | +35.1% | +10.1% (rand=25%) | +8.7% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 71.7% | 321/448 | +46.7% | +46.7% (rand=25%) | +43.7% (maj=28%) |
| land_use | 75.9% | 426/561 | +55.9% | +50.9% (rand=25%) | +47.8% (maj=28%) |
| mismatch_binary_easy | 93.6% | 525/561 | +83.6% | +43.6% (rand=50%) | +41.5% (maj=52%) |
| mismatch_binary_hard | 85.0% | 477/561 | +85.0% | +35.0% (rand=50%) | +32.8% (maj=52%) |
| mismatch_mcq_easy | 94.3% | 529/561 | +94.3% | +69.3% (rand=25%) | +67.6% (maj=27%) |
| mismatch_mcq_hard | 81.5% | 457/561 | +69.0% | +56.5% (rand=25%) | +55.6% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 71.1% | 399/561 | +56.8% | +46.1% (rand=25%) | +45.3% (maj=26%) |
| transit_density | 44.9% | 252/561 | +44.9% | +19.9% (rand=25%) | +16.6% (maj=28%) |
| urban_density | 69.5% | 390/561 | +59.5% | +44.5% (rand=25%) | +43.5% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 71.5% | +53.5% |
| 2 | 72.3% | +0.8% |

**Model still improving.** (+0.8% from previous epoch)
