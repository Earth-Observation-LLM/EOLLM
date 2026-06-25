# Epoch 6 Verdict

**Step:** 1638 | **Eval time:** 2154s | **Samples:** 8831

## Overall: 72.5% (6401/8831)

**vs Base model:** 19.0% → 72.5% (**+53.5%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 63.6% | 470/739 | +23.6% | +38.6% (rand=25%) | +37.2% (maj=26%) |
| building_height | 54.8% | 149/272 | +54.8% | +29.8% (rand=25%) | +26.1% (maj=29%) |
| camera_direction | 57.0% | 421/739 | +57.0% | +32.0% (rand=25%) | +30.3% (maj=27%) |
| green_space | 71.4% | 160/224 | +71.4% | +46.4% (rand=25%) | +45.5% (maj=26%) |
| junction_type | 69.4% | 359/517 | +52.8% | +44.4% (rand=25%) | +41.6% (maj=28%) |
| land_use | 71.4% | 528/739 | +71.4% | +46.4% (rand=25%) | +44.1% (maj=27%) |
| mismatch_binary_easy | 95.9% | 709/739 | +73.7% | +45.9% (rand=50%) | +45.5% (maj=50%) |
| mismatch_binary_hard | 91.1% | 673/739 | +66.1% | +41.1% (rand=50%) | +39.9% (maj=51%) |
| mismatch_mcq_easy | 98.0% | 724/739 | +73.0% | +73.0% (rand=25%) | +71.4% (maj=27%) |
| mismatch_mcq_hard | 88.9% | 657/739 | +88.9% | +63.9% (rand=25%) | +62.5% (maj=26%) |
| road_surface | 35.5% | 152/428 | -42.3% | +10.5% (rand=25%) | +7.5% (maj=28%) |
| road_type | 70.4% | 520/739 | +70.4% | +45.4% (rand=25%) | +43.4% (maj=27%) |
| transit_density | 46.5% | 344/739 | +19.3% | +21.5% (rand=25%) | +21.0% (maj=26%) |
| urban_density | 72.4% | 535/739 | +72.4% | +47.4% (rand=25%) | +46.1% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |
| 3 | 70.7% | +4.7% |
| 4 | 72.4% | +1.7% |
| 5 | 71.9% | -0.6% |
| 6 | 72.5% | +0.6% |

**Model still improving.** (+0.6% from previous epoch)
