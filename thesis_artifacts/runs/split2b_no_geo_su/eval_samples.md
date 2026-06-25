# Final Eval Samples (finetuned)

For a fair base-vs-finetuned comparison, run `python eval_base.py` and
then `EVAL_ADAPTER=<run_dir>/lora python eval_base.py` — identical code path.

## istanbul_0101_mismatch_binary_0 (mismatch_binary_hard)

**Question:** Does the ground-level imagery correspond to the point marked on the satellite view?

**Gold:** A

**Predicted:** 'B' → parsed `B` ✗

## seoul_0085_road_type_0 (road_type)

**Question:** What best describes the road that runs through this location?

**Gold:** C

**Predicted:** 'B' → parsed `B` ✗

## singapore_0100_mismatch_mcq_1 (mismatch_mcq_easy)

**Question:** Looking at the composite street view grid, which group corresponds to the red dot location?

**Gold:** A

**Predicted:** 'D' → parsed `D` ✗

