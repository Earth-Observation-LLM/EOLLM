# Epoch 7 Verdict

**Step:** 1911 | **Eval time:** 2151s | **Samples:** 8831

## Overall: 72.2% (6372/8831)

**vs Base model:** 19.0% → 72.2% (**+53.2%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.6% | 455/739 | +21.6% | +36.6% (rand=25%) | +35.2% (maj=26%) |
| building_height | 52.6% | 143/272 | +52.6% | +27.6% (rand=25%) | +23.9% (maj=29%) |
| camera_direction | 56.7% | 419/739 | +56.7% | +31.7% (rand=25%) | +30.0% (maj=27%) |
| green_space | 69.2% | 155/224 | +69.2% | +44.2% (rand=25%) | +43.3% (maj=26%) |
| junction_type | 69.2% | 358/517 | +52.6% | +44.2% (rand=25%) | +41.4% (maj=28%) |
| land_use | 71.0% | 525/739 | +71.0% | +46.0% (rand=25%) | +43.7% (maj=27%) |
| mismatch_binary_easy | 97.3% | 719/739 | +75.1% | +47.3% (rand=50%) | +46.8% (maj=50%) |
| mismatch_binary_hard | 92.6% | 684/739 | +67.6% | +42.6% (rand=50%) | +41.4% (maj=51%) |
| mismatch_mcq_easy | 98.1% | 725/739 | +73.1% | +73.1% (rand=25%) | +71.6% (maj=27%) |
| mismatch_mcq_hard | 91.1% | 673/739 | +91.1% | +66.1% (rand=25%) | +64.7% (maj=26%) |
| road_surface | 44.6% | 191/428 | -33.2% | +19.6% (rand=25%) | +16.6% (maj=28%) |
| road_type | 67.7% | 500/739 | +67.7% | +42.7% (rand=25%) | +40.7% (maj=27%) |
| transit_density | 41.3% | 305/739 | +14.0% | +16.3% (rand=25%) | +15.7% (maj=26%) |
| urban_density | 70.4% | 520/739 | +70.4% | +45.4% (rand=25%) | +44.1% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 67.2% | +48.2% |
| 2 | 66.0% | -1.3% |
| 3 | 70.7% | +4.7% |
| 4 | 72.4% | +1.7% |
| 5 | 71.9% | -0.6% |
| 6 | 72.5% | +0.6% |
| 7 | 72.2% | -0.3% |

**Plateauing.** (72.5% → 72.2%)
