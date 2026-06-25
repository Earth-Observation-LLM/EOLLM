# 10 — Attention-based image selection (no router)

**Status:** RESEARCH IDEA (user's plan) — next paper / extension
**Origin:** From the attention analysis (`benchmark_suite/results/qwen35_4b_awq_attn100/`),
the model concentrates most of its image attention on ~1–2 of the 5 images when it
answers correctly.

## Hypothesis
When the model answers CORRECTLY, the image(s) it attends to most are the
informative one(s) for that item/task. So we can *select* the right images by
reading attention on correct answers — no trained router needed.

## Method sketch
1. On the attention-instrumented runs (full_attention.jsonl has per-image
   `image_attention` + per-layer, choice-token attention), filter to `is_correct == True`.
2. For each task, rank the 5 images (sat-marked + 4 SV) by attention mass; identify
   the top-1 / top-2 most-attended images.
3. Aggregate per task: which view/image does "correct-answer attention" prefer?
   (Likely 1 SV angle for most attribute tasks, sat for layout tasks.)
4. Use that as an oracle/heuristic image selector → feed only the selected image(s)
   → re-evaluate accuracy. Compare to `full` and to a single fixed view.

## Payoff
If feeding only the attention-selected image(s) ≈ full, that's direct evidence the
extra images are unnecessary, and gives a cheap test-time policy. Pairs with the
2-image straight-view ablation (task 12).

## Caveats
- Attention ≠ importance (cite Jain&Wallace, Wiegreffe&Pinter). Frame selection as a
  heuristic validated by the downstream accuracy, not as proof of importance.
- Selecting using `is_correct` peeks at the label → it's an oracle/upper-bound
  selector, like the per-item oracle in the current paper. State that.
- Use the 9B (or a stronger model than the 4B-AWQ attention run) for the final claim.
