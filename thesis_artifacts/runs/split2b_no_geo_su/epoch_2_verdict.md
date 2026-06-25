# Epoch 2 Verdict

**Step:** 354 | **Eval time:** 2185s | **Samples:** 8831

## Overall: 59.0% (5207/8831)

**vs Base model:** 19.0% → 59.0% (**+40.0%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 61.7% | 456/739 | +21.7% | +36.7% (rand=25%) | +35.3% (maj=26%) |
| building_height | 50.4% | 137/272 | +50.4% | +25.4% (rand=25%) | +21.7% (maj=29%) |
| camera_direction | 27.5% | 203/739 | +27.5% | +2.5% (rand=25%) | +0.8% (maj=27%) |
| green_space | 100.0% | 224/224 | +100.0% | +75.0% (rand=25%) | +74.1% (maj=26%) |
| junction_type | 74.7% | 386/517 | +58.0% | +49.7% (rand=25%) | +46.8% (maj=28%) |
| land_use | 74.6% | 551/739 | +74.6% | +49.6% (rand=25%) | +47.2% (maj=27%) |
| mismatch_binary_easy | 69.1% | 511/739 | +46.9% | +19.1% (rand=50%) | +18.7% (maj=50%) |
| mismatch_binary_hard | 65.4% | 483/739 | +40.4% | +15.4% (rand=50%) | +14.2% (maj=51%) |
| mismatch_mcq_easy | 31.9% | 236/739 | +6.9% | +6.9% (rand=25%) | +5.4% (maj=27%) |
| mismatch_mcq_hard | 25.0% | 185/739 | +25.0% | +0.0% (rand=25%) | -1.4% (maj=26%) |
| road_surface | 97.9% | 419/428 | +20.1% | +72.9% (rand=25%) | +69.9% (maj=28%) |
| road_type | 71.7% | 530/739 | +71.7% | +46.7% (rand=25%) | +44.8% (maj=27%) |
| transit_density | 44.9% | 332/739 | +17.7% | +19.9% (rand=25%) | +19.4% (maj=26%) |
| urban_density | 75.0% | 554/739 | +75.0% | +50.0% (rand=25%) | +48.7% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 58.9% | +39.9% |
| 2 | 59.0% | +0.1% |

**Model still improving.** (+0.1% from previous epoch)
