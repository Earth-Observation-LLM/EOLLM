# Tuning Notes

max_seq_length: 8192

## Probe Sample Selection

- records scanned: 13500
- stage-A top-N: 200
- real probe samples (K): 8
- real sample token lengths: [1971, 1971, 1971, 1971, 1971, 1971, 1971, 1971]
- max real token length: 1971
- synthetic sample included: True
- synthetic token length: 5275

## Probe Trace

| BS | Status | Peak Reserved (MB) | Peak Allocated (MB) | Budget (MB) | Util |
|----|--------|--------------------|---------------------|-------------|------|
| 96 | SKIPPED (forced) | — | — | — | — |

## Summary

- chosen bs: **96**
- peak at chosen bs: None MB reserved
- budget at chosen bs: 85814 MB
- free VRAM at probe: 87566 MB / 97241 MB total
- absolute reserve: 0 MB
- margin applied at chosen bs: 2% (tiered)
- grad_accum: 1, effective batch: 96
- steps/epoch: 141
