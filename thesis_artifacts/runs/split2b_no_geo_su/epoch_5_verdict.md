# Epoch 5 Verdict

**Step:** 885 | **Eval time:** 2190s | **Samples:** 8831

## Overall: 60.4% (5334/8831)

**vs Base model:** 19.0% → 60.4% (**+41.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 59.7% | 441/739 | +19.7% | +34.7% (rand=25%) | +33.3% (maj=26%) |
| building_height | 54.0% | 147/272 | +54.0% | +29.0% (rand=25%) | +25.4% (maj=29%) |
| camera_direction | 26.1% | 193/739 | +26.1% | +1.1% (rand=25%) | -0.5% (maj=27%) |
| green_space | 100.0% | 224/224 | +100.0% | +75.0% (rand=25%) | +74.1% (maj=26%) |
| junction_type | 69.6% | 360/517 | +53.0% | +44.6% (rand=25%) | +41.8% (maj=28%) |
| land_use | 76.2% | 563/739 | +76.2% | +51.2% (rand=25%) | +48.8% (maj=27%) |
| mismatch_binary_easy | 77.4% | 572/739 | +55.2% | +27.4% (rand=50%) | +26.9% (maj=50%) |
| mismatch_binary_hard | 68.1% | 503/739 | +43.1% | +18.1% (rand=50%) | +16.9% (maj=51%) |
| mismatch_mcq_easy | 37.3% | 276/739 | +12.3% | +12.3% (rand=25%) | +10.8% (maj=27%) |
| mismatch_mcq_hard | 31.1% | 230/739 | +31.1% | +6.1% (rand=25%) | +4.7% (maj=26%) |
| road_surface | 98.4% | 421/428 | +20.6% | +73.4% (rand=25%) | +70.3% (maj=28%) |
| road_type | 72.1% | 533/739 | +72.1% | +47.1% (rand=25%) | +45.2% (maj=27%) |
| transit_density | 44.9% | 332/739 | +17.7% | +19.9% (rand=25%) | +19.4% (maj=26%) |
| urban_density | 72.9% | 539/739 | +72.9% | +47.9% (rand=25%) | +46.7% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 58.9% | +39.9% |
| 2 | 59.0% | +0.1% |
| 3 | 61.8% | +2.8% |
| 4 | 60.9% | -0.9% |
| 5 | 60.4% | -0.5% |

**Plateauing.** (60.9% → 60.4%)
