# 02 — Unmarked-satellite ablation

**Status:** TODO (needs lab-ws GPU run)
**Priority:** HIGH — reviewers' "the model just reads the red dot" objection.

## Why
The satellite tile carries a red location marker (`image_mode == satellite_marked`).
A reviewer can argue the model attends to the marker overlay artifact rather than
imagery, or that the marker leaks the answer region — corrupting `sat` and `full`.

## What to run
Re-run `sat_only` and `full` on a subset (or all) of the 8-task benchmark with an
UNMARKED satellite tile (the raw `images.satellite` with no red dot), holding
everything else fixed. Compare synergy and accuracy marked vs unmarked.

Decision:
- If synergy + per-task accuracy ≈ unchanged → marker exonerated, the redundancy is not a marker artifact.
- If they shift materially → the marker matters; report and rescope.

Also: the attention check already shows mass is not concentrated on a tiny dot
region (spatial attention within the tile) — include if available.

## Where
- The raw unmarked tile is `images.satellite`; the marker is drawn by the viewer/
  composite step (`composite_utils.py` / `satellite_marked` render). Feed the raw tile.
- Same harness + output format as task 01.

## Paper hook
Framed as future check in §6 Robustness (satellite marker). When done, replace with
the measured marked-vs-unmarked delta.
