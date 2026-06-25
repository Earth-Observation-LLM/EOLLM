# Epoch 2 Verdict

**Step:** 520 | **Eval time:** 1719s | **Samples:** 6984

## Overall: 71.8% (5013/6984)

**vs Base model:** 18.0% → 71.8% (**+53.8%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 50.6% | 284/561 | -9.4% | +25.6% (rand=25%) | +19.3% (maj=31%) |
| building_height | 61.3% | 160/261 | +61.3% | +36.3% (rand=25%) | +33.3% (maj=28%) |
| camera_direction | 27.5% | 154/561 | +27.5% | +2.5% (rand=25%) | +1.1% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 72.3% | 324/448 | +47.3% | +47.3% (rand=25%) | +44.4% (maj=28%) |
| land_use | 75.6% | 424/561 | +55.6% | +50.6% (rand=25%) | +47.4% (maj=28%) |
| mismatch_binary_easy | 96.1% | 539/561 | +86.1% | +46.1% (rand=50%) | +44.0% (maj=52%) |
| mismatch_binary_hard | 83.6% | 469/561 | +83.6% | +33.6% (rand=50%) | +31.4% (maj=52%) |
| mismatch_mcq_easy | 94.3% | 529/561 | +94.3% | +69.3% (rand=25%) | +67.6% (maj=27%) |
| mismatch_mcq_hard | 82.7% | 464/561 | +70.2% | +57.7% (rand=25%) | +56.9% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 70.6% | 396/561 | +56.3% | +45.6% (rand=25%) | +44.7% (maj=26%) |
| transit_density | 45.5% | 255/561 | +45.5% | +20.5% (rand=25%) | +17.1% (maj=28%) |
| urban_density | 66.5% | 373/561 | +56.5% | +41.5% (rand=25%) | +40.5% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 70.7% | +52.7% |
| 2 | 71.8% | +1.0% |

**Model still improving.** (+1.0% from previous epoch)
