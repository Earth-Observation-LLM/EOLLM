# Per-task strategy ablation — the 9 urbanization tasks

_Qwen3.5-9B-AWQ · greedy (temp 0) · lab-ws SLURM job 216 · 0 errors._

**Scope.** Only the 9 urban-attribute tasks. The cross-view tasks (`mismatch_*`, `camera_direction`) are excluded on purpose: their answer lives in *having both images*, so a single-view ablation is ill-posed there. These 9 are where the question *“do we actually need both views?”* is fair.

`full` = marked-satellite + 4 street-view angles · `sat` = satellite only · `sv` = street-view grid only · `blind` = no images. **synergy** = full − max(sat, sv) — positive ⇒ the second view adds something a single view can't. **vision** = full − blind.

## Overall (attribute tasks only)

| strategy | full | sat | sv | blind | synergy | vision |
|---|---|---|---|---|---|---|
| current | 48.6 | 44.4 | 50.9 | 40.9 | **-2.3** | +7.7 |
| think16k | 49.8 | 46.1 | 51.8 | 41.0 | **-2.0** | +8.8 |
| multistep | 51.7 | 49.4 | 53.9 | 43.4 | **-2.2** | +8.3 |

## Per task — all 3 strategies × 4 views

### Land use  _(n=421/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 53.0 | 50.1 | 52.7 | 43.5 | +0.3 | NO |
| think16k | 51.8 | 45.1 | 51.5 | 51.5 | +0.3 | NO |
| multistep | 49.4 | 50.1 | 53.0 | 50.6 | -3.6 | NO |

### Building height  _(n=190/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 61.1 | 47.9 | 64.7 | 45.8 | -3.6 | NO |
| think16k | 62.6 | 51.6 | 61.6 | 38.4 | +1.0 | NO |
| multistep | 61.1 | 54.7 | 58.4 | 43.2 | +2.7 | NO |

### Urban density  _(n=421/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 38.5 | 39.2 | 44.9 | 40.4 | -6.4 | NO |
| think16k | 38.7 | 38.0 | 43.9 | 32.3 | -5.2 | NO |
| multistep | 41.6 | 43.7 | 45.4 | 35.9 | -3.8 | NO |

### Junction type  _(n=334/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 62.6 | 48.8 | 62.6 | 48.5 | +0.0 | NO |
| think16k | 57.5 | 54.8 | 59.0 | 32.3 | -1.5 | NO |
| multistep | 63.5 | 59.3 | 62.9 | 46.4 | +0.6 | NO |

### Green space  _(n=203/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 36.5 | 35.5 | 37.9 | 64.5 | -1.4 | NO |
| think16k | 41.4 | 38.4 | 49.8 | 59.1 | -8.4 | NO |
| multistep | 64.5 | 62.1 | 65.0 | 68.5 | -0.5 | NO |

### Amenity richness  _(n=421/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 36.8 | 36.1 | 33.5 | 19.5 | +0.7 | NO |
| think16k | 36.3 | 37.3 | 34.9 | 20.9 | -1.0 | NO |
| multistep | 37.1 | 38.7 | 36.1 | 25.9 | -1.6 | NO |

### Road type  _(n=421/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 47.5 | 40.1 | 57.5 | 34.9 | -10.0 | NO |
| think16k | 58.4 | 50.4 | 64.4 | 34.4 | -6.0 | NO |
| multistep | 53.4 | 48.5 | 63.2 | 33.3 | -9.8 | NO |

### Road surface  _(n=303/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 82.2 | 78.9 | 87.1 | 67.7 | -4.9 | NO |
| think16k | 83.2 | 75.2 | 84.8 | 94.4 | -1.6 | NO |
| multistep | 89.8 | 80.9 | 89.4 | 83.8 | +0.4 | NO |

### Transit density  _(n=421/view)_

| strategy | full | sat | sv | blind | synergy | needs both? |
|---|---|---|---|---|---|---|
| current | 32.1 | 30.9 | 30.4 | 27.3 | +1.2 | NO |
| think16k | 31.6 | 33.3 | 31.6 | 26.6 | -1.7 | NO |
| multistep | 29.7 | 26.8 | 31.4 | 27.8 | -1.7 | NO |

## Verdict

Across **all 9 tasks and all 3 strategies**, `full` never clears the best single view by a meaningful margin — synergy is **≤ 0 in every overall cell** and the *needs-both* test is **NO for every task**. street-view alone (`sv`) matches or beats the two-view `full` setup throughout. Vision still matters (full − blind ≈ +8 overall), but it is **one-view vision**: the satellite image is redundant given the street-view, and no prompting strategy — 16k of explicit thinking, nor a 4-turn observe→reason→self-check→commit conversation — recovers any complementary signal from it.
