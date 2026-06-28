# Epoch 1 Verdict

**Step:** 454 | **Eval time:** 2247s | **Samples:** 8831

## Overall: 69.2% (6109/8831)

**vs Base model:** 42.0% → 69.2% (**+27.2%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 48.0% | 355/739 | +28.0% | +23.0% (rand=25%) | +21.7% (maj=26%) |
| building_height | 51.5% | 140/272 | +1.5% | +26.5% (rand=25%) | +22.8% (maj=29%) |
| camera_direction | 29.0% | 214/739 | -21.0% | +4.0% (rand=25%) | +2.3% (maj=27%) |
| green_space | 52.2% | 117/224 | +2.2% | +27.2% (rand=25%) | +26.3% (maj=26%) |
| junction_type | 68.7% | 355/517 | +18.7% | +43.7% (rand=25%) | +40.8% (maj=28%) |
| land_use | 72.5% | 536/739 | +36.2% | +47.5% (rand=25%) | +45.2% (maj=27%) |
| mismatch_binary_easy | 96.1% | 710/739 | +40.5% | +46.1% (rand=50%) | +45.6% (maj=50%) |
| mismatch_binary_hard | 87.7% | 648/739 | +25.2% | +37.7% (rand=50%) | +36.5% (maj=51%) |
| mismatch_mcq_easy | 94.5% | 698/739 | +94.5% | +69.5% (rand=25%) | +67.9% (maj=27%) |
| mismatch_mcq_hard | 79.2% | 585/739 | +54.2% | +54.2% (rand=25%) | +52.8% (maj=26%) |
| road_surface | 98.6% | 422/428 | -1.4% | +73.6% (rand=25%) | +70.6% (maj=28%) |
| road_type | 69.0% | 510/739 | +19.0% | +44.0% (rand=25%) | +42.1% (maj=27%) |
| transit_density | 40.6% | 300/739 | +4.2% | +15.6% (rand=25%) | +15.0% (maj=26%) |
| urban_density | 70.2% | 519/739 | +59.1% | +45.2% (rand=25%) | +44.0% (maj=26%) |