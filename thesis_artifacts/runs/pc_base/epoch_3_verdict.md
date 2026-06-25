# Epoch 3 Verdict

**Step:** 1011 | **Eval time:** 1769s | **Samples:** 6984

## Overall: 75.0% (5237/6984)

**vs Base model:** 18.0% → 75.0% (**+57.0%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 51.9% | 291/561 | -8.1% | +26.9% (rand=25%) | +20.5% (maj=31%) |
| building_height | 60.2% | 157/261 | +60.2% | +35.2% (rand=25%) | +32.2% (maj=28%) |
| camera_direction | 46.0% | 258/561 | +46.0% | +21.0% (rand=25%) | +19.6% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 72.8% | 326/448 | +47.8% | +47.8% (rand=25%) | +44.9% (maj=28%) |
| land_use | 76.5% | 429/561 | +56.5% | +51.5% (rand=25%) | +48.3% (maj=28%) |
| mismatch_binary_easy | 96.3% | 540/561 | +86.3% | +46.3% (rand=50%) | +44.2% (maj=52%) |
| mismatch_binary_hard | 88.6% | 497/561 | +88.6% | +38.6% (rand=50%) | +36.4% (maj=52%) |
| mismatch_mcq_easy | 96.8% | 543/561 | +96.8% | +71.8% (rand=25%) | +70.1% (maj=27%) |
| mismatch_mcq_hard | 86.8% | 487/561 | +74.3% | +61.8% (rand=25%) | +61.0% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 73.1% | 410/561 | +58.8% | +48.1% (rand=25%) | +47.2% (maj=26%) |
| transit_density | 44.6% | 250/561 | +44.6% | +19.6% (rand=25%) | +16.2% (maj=28%) |
| urban_density | 72.5% | 407/561 | +62.5% | +47.5% (rand=25%) | +46.5% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 71.5% | +53.5% |
| 2 | 72.3% | +0.8% |
| 3 | 75.0% | +2.7% |

**Model still improving.** (+2.7% from previous epoch)
