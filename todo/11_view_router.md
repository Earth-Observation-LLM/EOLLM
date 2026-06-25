# 11 — Train a view/image router

**Status:** RESEARCH IDEA (user's plan) — next paper / extension

## Idea
Train a lightweight router that, per question (or per task), selects which view(s)
or image(s) to feed the VLM — instead of always feeding all 5 images. Motivated by
the current paper's finding that `full` underperforms a per-item single-view oracle
by 8–9 pts: a real router could recover part of that oracle gap and beat `full`
while using fewer images.

## Design options
- **Task-level router:** map topic → best single view (cheap; the paper already
  shows which view wins per task). Almost free baseline.
- **Item-level router:** small classifier on the question text (+ maybe thumbnail)
  → predict {sat, sv, both}. Train on which view was correct per item (the oracle
  signal). Risk: needs to generalize, not just memorize.
- **Attention-guided (see task 10):** use the model's own attention to pick images
  at test time (no separate training).

## Eval
- Compare router-selected-view accuracy vs `full`, vs best fixed single view, vs the
  per-item oracle (upper bound). Report cost (images/tokens served) alongside accuracy.
- Show the router closes part of the full→oracle gap — turns the negative result into
  an actionable method (addresses "no method proposed" critique of the current paper).

## Note
This is the "solution" the current paper deliberately does NOT claim (we couldn't
find one in time). A working router would be the natural follow-up contribution.
