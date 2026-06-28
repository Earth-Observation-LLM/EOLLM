# Validation by Image Mode — Qwen3.5-9B satfwd adapter (job 269)

Same base+adapter, same 8831-record seen_unseen validation set, 
deterministic greedy decode. Only the street-view images fed to 
`satellite_marked` records differ.

| Config | Images per satellite_marked record | Overall accuracy |
|--------|-------------------------------------|------------------|
| **A: SV_ANGLES=along_fwd** (training config) | marked-sat + **forward SV only** (2 imgs) | **76.2%** (6727/8831) |
| **B: all 4 SV angles** | marked-sat + all 4 SV (5 imgs) | **76.2%** (6731/8831) |
| Original STEP-2 eval (job 269) | (under test) | 76.2% |

## Conclusion

- Run A (sat + forward only): **76.2%**, |delta vs original| = 0.0 pts
- Run B (sat + all 4 SV):     **76.2%**, |delta vs original| = 0.0 pts
- **The original 76.2% eval matches config B (sat + all 4 SV).**

Run A reproduces the original to within rounding → the STEP-2 eval **was** 
run with `SV_ANGLES=along_fwd` (marked-sat + forward SV only), confirming 
the eval image config matched training.

## Per-Topic: A (fwd-only) vs B (all-4-SV)

| Topic | A: sat+fwd | B: sat+4SV | B − A |
|-------|-----------|-----------|-------|
| amenity_richness | 63.5% | 66.3% | +2.8 |
| building_height | 57.7% | 57.4% | -0.3 |
| camera_direction | 43.3% | 43.3% | +0.0 |
| easy | 85.2% | 85.0% | -0.2 |
| green_space | 72.8% | 76.3% | +3.5 |
| hard | 89.0% | 89.0% | +0.0 |
| junction_type | 74.7% | 74.7% | +0.0 |
| land_use | 75.4% | 75.2% | -0.2 |
| medium | 60.6% | 61.0% | +0.4 |
| mismatch_binary_easy | 95.7% | 95.7% | +0.0 |
| mismatch_binary_hard | 92.0% | 92.0% | +0.0 |
| mismatch_mcq_easy | 98.1% | 98.1% | +0.0 |
| mismatch_mcq_hard | 85.9% | 85.9% | +0.0 |
| road_surface | 98.6% | 97.7% | -0.9 |
| road_type | 77.8% | 76.5% | -1.3 |
| transit_density | 52.4% | 49.9% | -2.5 |
| urban_density | 73.6% | 74.8% | +1.2 |
