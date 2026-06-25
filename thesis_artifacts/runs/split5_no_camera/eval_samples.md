# Final Eval Samples (finetuned)

For a fair base-vs-finetuned comparison, run `python eval_base.py` and
then `EVAL_ADAPTER=<run_dir>/lora python eval_base.py` — identical code path.

## zurich_0115_land_use_0 (land_use)

**Question:** Based on the imagery, what best describes how this area is primarily used?

**Gold:** A

**Predicted:** 'A' → parsed `A` ✓

## buenos_aires_0035_mismatch_mcq_1 (mismatch_mcq_easy)

**Question:** Based on the marked satellite image, which of the four street view sets is the correct match?

**Gold:** C

**Predicted:** 'C' → parsed `C` ✓

## mexico_city_0001_land_use_0 (land_use)

**Question:** Considering both the aerial view and the street-level perspective, how is this land primarily utilized?

**Gold:** B

**Predicted:** 'B' → parsed `B` ✓

