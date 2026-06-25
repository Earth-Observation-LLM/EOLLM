# Epoch 2 Verdict

**Step:** 282 | **Eval time:** 1721s | **Samples:** 6984

## Overall: 63.4% (4428/6984)

**vs Base model:** 18.0% → 63.4% (**+45.4%**)

## Per-Topic Accuracy

| Topic | Accuracy | Correct/Total | vs Base | vs Random | vs Majority |
|-------|----------|---------------|---------|-----------|-------------|
| amenity_richness | 37.3% | 209/561 | -22.7% | +12.3% (rand=25%) | +5.9% (maj=31%) |
| building_height | 49.0% | 128/261 | +49.0% | +24.0% (rand=25%) | +21.1% (maj=28%) |
| camera_direction | 31.4% | 176/561 | +31.4% | +6.4% (rand=25%) | +5.0% (maj=26%) |
| green_space | 100.0% | 267/267 | +66.7% | +75.0% (rand=25%) | +71.9% (maj=28%) |
| junction_type | 62.1% | 278/448 | +37.1% | +37.1% (rand=25%) | +34.2% (maj=28%) |
| land_use | 54.9% | 308/561 | +34.9% | +29.9% (rand=25%) | +26.7% (maj=28%) |
| mismatch_binary_easy | 96.4% | 541/561 | +86.4% | +46.4% (rand=50%) | +44.4% (maj=52%) |
| mismatch_binary_hard | 89.1% | 500/561 | +89.1% | +39.1% (rand=50%) | +36.9% (maj=52%) |
| mismatch_mcq_easy | 95.2% | 534/561 | +95.2% | +70.2% (rand=25%) | +68.4% (maj=27%) |
| mismatch_mcq_hard | 83.1% | 466/561 | +70.6% | +58.1% (rand=25%) | +57.2% (maj=26%) |
| road_surface | 94.2% | 375/398 | +19.2% | +69.2% (rand=25%) | +65.1% (maj=29%) |
| road_type | 53.7% | 301/561 | +39.4% | +28.7% (rand=25%) | +27.8% (maj=26%) |
| transit_density | 30.7% | 172/561 | +30.7% | +5.7% (rand=25%) | +2.3% (maj=28%) |
| urban_density | 30.8% | 173/561 | +20.8% | +5.8% (rand=25%) | +4.8% (maj=26%) |

## Full Eval History

| Epoch | Accuracy | Delta |
|-------|----------|-------|
| 1 | 59.6% | +41.6% |
| 2 | 63.4% | +3.8% |

**Model still improving.** (+3.8% from previous epoch)
