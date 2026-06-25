# Epoch 4 Verdict

**Step:** 1092 | **Eval time:** 2148s | **Samples:** 8831

## Overall: 72.4% (6397/8831)

**vs Base model:** 19.0% → 72.4% (**+53.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 65.4% | 483/739 | +25.4% | +40.4% (rand=25%) | +39.0% (maj=26%) |
| building_height | 55.9% | 152/272 | +55.9% | +30.9% (rand=25%) | +27.2% (maj=29%) |
| camera_direction | 48.0% | 355/739 | +48.0% | +23.0% (rand=25%) | +21.4% (maj=27%) |
| green_space | 72.3% | 162/224 | +72.3% | +47.3% (rand=25%) | +46.4% (maj=26%) |
| junction_type | 68.1% | 352/517 | +51.4% | +43.1% (rand=25%) | +40.2% (maj=28%) |
| land_use | 74.2% | 548/739 | +74.2% | +49.2% (rand=25%) | +46.8% (maj=27%) |
| mismatch_binary_easy | 93.8% | 693/739 | +71.6% | +43.8% (rand=50%) | +43.3% (maj=50%) |
| mismatch_binary_hard | 89.7% | 663/739 | +64.7% | +39.7% (rand=50%) | +38.6% (maj=51%) |
| mismatch_mcq_easy | 97.6% | 721/739 | +72.6% | +72.6% (rand=25%) | +71.0% (maj=27%) |
| mismatch_mcq_hard | 87.4% | 646/739 | +87.4% | +62.4% (rand=25%) | +61.0% (maj=26%) |
| road_surface | 38.8% | 166/428 | -39.0% | +13.8% (rand=25%) | +10.7% (maj=28%) |
| road_type | 70.8% | 523/739 | +70.8% | +45.8% (rand=25%) | +43.8% (maj=27%) |
| transit_density | 50.6% | 374/739 | +23.3% | +25.6% (rand=25%) | +25.0% (maj=26%) |
| urban_density | 75.6% | 559/739 | +75.6% | +50.6% (rand=25%) | +49.4% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |
| 3 | 70.7% | +4.7% |
| 4 | 72.4% | +1.7% |

**Model still improving.** (+1.7% from previous epoch)
