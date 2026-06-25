# 12 — Two straight-view images ablation (panorama not always needed)

**Status:** RESEARCH IDEA (user's plan) — extension
**Depends on / pairs with:** task 10 (attention selection), task 01 (single-angle).

## Idea
Show that you don't need the full panorama / 4-angle street-view set — TWO
straight-view images (or the 2 the model attends to most) are enough for the
attribute tasks. This generalizes "single view might be enough" to "even within
street view, fewer images suffice."

## Method
- Pick the 2 images per item by: (a) attention-when-correct (task 10), or (b) a
  fixed pair (e.g. fwd + the cross angle), or (c) the router (task 11).
- Re-evaluate on the 8-task benchmark; compare {2 selected SV} vs {4 SV grid} vs full.
- Expected: {2 SV} ≈ {4 SV} ≈ full for most attribute tasks.

## Payoff
- Reinforces the redundancy story at a second granularity (within-modality).
- Cheaper serving (2 images not 5).
- Directly answers the "sv wins only because 4 images" confound from the other side.

## Caveats
- If using attention-selected pairs, it's an oracle selector (peeks at correctness) —
  state it, or validate the selection rule generalizes on held-out items.
