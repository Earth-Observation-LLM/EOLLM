# Epoch 1 Verdict

**Step:** 273 | **Eval time:** 2182s | **Samples:** 8831

## Overall: 67.2% (5937/8831)

**vs Base model:** 19.0% → 67.2% (**+48.2%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 57.8% | 427/739 | +17.8% | +32.8% (rand=25%) | +31.4% (maj=26%) |
| building_height | 44.1% | 120/272 | +44.1% | +19.1% (rand=25%) | +15.4% (maj=29%) |
| camera_direction | 26.5% | 196/739 | +26.5% | +1.5% (rand=25%) | -0.1% (maj=27%) |
| green_space | 72.8% | 163/224 | +72.8% | +47.8% (rand=25%) | +46.9% (maj=26%) |
| junction_type | 72.7% | 376/517 | +56.1% | +47.7% (rand=25%) | +44.9% (maj=28%) |
| land_use | 70.8% | 523/739 | +70.8% | +45.8% (rand=25%) | +43.4% (maj=27%) |
| mismatch_binary_easy | 93.5% | 691/739 | +71.3% | +43.5% (rand=50%) | +43.0% (maj=50%) |
| mismatch_binary_hard | 81.5% | 602/739 | +56.5% | +31.5% (rand=50%) | +30.3% (maj=51%) |
| mismatch_mcq_easy | 89.6% | 662/739 | +64.6% | +64.6% (rand=25%) | +63.1% (maj=27%) |
| mismatch_mcq_hard | 78.9% | 583/739 | +78.9% | +53.9% (rand=25%) | +52.5% (maj=26%) |
| road_surface | 60.7% | 260/428 | -17.0% | +35.7% (rand=25%) | +32.7% (maj=28%) |
| road_type | 71.4% | 528/739 | +71.4% | +46.4% (rand=25%) | +44.5% (maj=27%) |
| transit_density | 46.5% | 344/739 | +19.3% | +21.5% (rand=25%) | +21.0% (maj=26%) |
| urban_density | 62.5% | 462/739 | +62.5% | +37.5% (rand=25%) | +36.3% (maj=26%) |