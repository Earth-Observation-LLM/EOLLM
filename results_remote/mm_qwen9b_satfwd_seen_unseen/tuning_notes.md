# Tuning Notes

max_seq_length: 8192

## Probe Sample Selection

- records scanned: 25420
- stage-A top-N: 200
- real probe samples (K): 8
- real sample token lengths: [1395, 1395, 1395, 1395, 1395, 1395, 1395, 1395]
- max real token length: 1395
- synthetic sample included: True
- synthetic token length: 1435

## Probe Trace

| BS | Status | Peak Reserved (MB) | Peak Allocated (MB) | Budget (MB) | Util |
|----|--------|--------------------|---------------------|-------------|------|
| 56 | SKIPPED (forced) | — | — | — | — |

## Summary

- chosen bs: **56**
- peak at chosen bs: None MB reserved
- budget at chosen bs: 76635 MB
- free VRAM at probe: 78199 MB / 97241 MB total
- absolute reserve: 0 MB
- margin applied at chosen bs: 2% (tiered)
- grad_accum: 1, effective batch: 56
- steps/epoch: 454
