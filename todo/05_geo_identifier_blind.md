# 05 — Geo-identifier-stripped blind re-run

**Status:** TODO (lower priority; logic defense already in paper)
**Priority:** MEDIUM

## Why
Prompts may include city/country/coords. A reviewer could argue blind/vision
accuracy partly reflects the model recalling OSM/geo priors from the place name,
not the image. Already defended logically (synergy is a within-model difference, so
a uniform geo prior cancels), but a direct check is stronger.

## What to run
- Inspect the MCQ prompt template: does `city`/`country`/`latitude`/`longitude`
  leak into the text shown to the model? (Check the benchmark_suite prompt builder.)
- If yes: re-run `blind` mode with geo identifiers stripped. Show accuracy drops
  toward chance-ish without geo hints (or doesn't, if they weren't used).
- Confirm the synergy/interference results are unchanged (they should be — vision
  modes don't depend on the place name).

## Paper hook
One sentence in §6 Robustness (OSM provenance). Currently the paper makes the
within-model-difference argument; add the empirical blind delta if run.
