# Epoch 4 Verdict

**Step:** 956 | **Eval time:** 1681s | **Samples:** 6984

## Overall: 76.3% (5327/6984)

**vs Base model:** 18.0% → 76.3% (**+58.3%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 59.9% | 336/561 | -0.1% | +34.9% (rand=25%) | +28.5% (maj=31%) |
| building_height | 65.9% | 172/261 | +65.9% | +40.9% (rand=25%) | +37.9% (maj=28%) |
| camera_direction | 45.1% | 253/561 | +45.1% | +20.1% (rand=25%) | +18.7% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 76.3% | 342/448 | +51.3% | +51.3% (rand=25%) | +48.4% (maj=28%) |
| land_use | 79.0% | 443/561 | +59.0% | +54.0% (rand=25%) | +50.8% (maj=28%) |
| mismatch_binary_easy | 97.3% | 546/561 | +87.3% | +47.3% (rand=50%) | +45.3% (maj=52%) |
| mismatch_binary_hard | 83.8% | 470/561 | +83.8% | +33.8% (rand=50%) | +31.6% (maj=52%) |
| mismatch_mcq_easy | 98.0% | 550/561 | +98.0% | +73.0% (rand=25%) | +71.3% (maj=27%) |
| mismatch_mcq_hard | 81.3% | 456/561 | +68.8% | +56.3% (rand=25%) | +55.4% (maj=26%) |
| road_surface | 94.0% | 374/398 | +19.0% | +69.0% (rand=25%) | +64.8% (maj=29%) |
| road_type | 73.6% | 413/561 | +59.3% | +48.6% (rand=25%) | +47.8% (maj=26%) |
| transit_density | 51.7% | 290/561 | +51.7% | +26.7% (rand=25%) | +23.4% (maj=28%) |
| urban_density | 74.0% | 415/561 | +64.0% | +49.0% (rand=25%) | +48.0% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 68.5% | +50.5% |
| 2 | 73.3% | +4.9% |
| 3 | 75.1% | +1.7% |
| 4 | 76.3% | +1.2% |

**Model still improving.** (+1.2% from previous epoch)
