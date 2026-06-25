# EOLLM — 3 Fusion Tasks (plain English)

**The idea in one line:** the answer comes from crossing **what the place FEELS like on the ground**
(only the 4 street photos can tell you) with **what the place actually IS in the hidden layout**
(only the satellite can tell you). The two don't line up — so you need both views to answer.

**Why this works when the old tasks didn't:** if a single street photo or a zoomed-in satellite
could answer it, one view is enough and there's no fusion. These three avoid that because each one
mixes a *feeling* (which the satellite can't see) with a *hidden fact* (which the street can't see).

---

## Task 1 — Edge or heart of the city? (the cleanest one)

**Question:** "Looking at the 4 street views together with the satellite, this spot is:"
- Enclosed, and deep inside the built-up area
- Enclosed, but actually right at the city's edge (open land is just behind the wall you face)
- Open, and genuinely at the edge
- Open, but still deep inside the built-up area

**Why both views are needed:** the street tells you whether it *feels* open or closed-in. The
satellite tells you whether you're at the *rim* of the city or buried deep inside. Those two facts
are independent — a street that feels totally enclosed can be one block from open countryside, and
a wide open-feeling plaza can be in the dead center of the city. Neither view alone can tell these
apart.

**Useful for:** how central or edge-like a place really is (vs. how it feels) — city legibility,
edge effects, where the urban-to-rural transition is.

---

## Task 2 — Quiet street, or secret through-route?

**Question:** "This street and its place in the road network are best described as:"
- Quiet, and genuinely off the main network
- A busy main road (and it looks like one)
- Looks quiet, but is actually a through-route people cut through
- Looks like a main road, but actually goes nowhere (a stub / dead link)

**Why both views are needed:** the street tells you whether it *feels* calm or busy. The satellite
tells you whether the road actually *connects through* the wider network or dead-ends out of sight.
A calm-looking residential street can secretly be a rat-run; a wide road can lead nowhere. You can't
see "does it connect 200m ahead" from the ground, and you can't feel "is it calm" from above.

**Useful for:** finding cut-through / rat-run streets, livability, traffic-calming — the most common
distinction in street planning. (Fires on ~75% of points.)

---

## Task 3 — Is the boundary real, or just on the map?

**Question:** "The satellite shows a possible boundary near this point. Reading the 4 street views,
that boundary is:"
- Real, and you're standing on it (the two sides genuinely differ on the ground)
- Real, but just beyond you (you're still inside one area)
- Not a real boundary (both sides look the same on the ground — it's only a line on the map)
- A meeting of several different areas at once

**Why both views are needed:** the satellite *always* shows a candidate boundary line — so on its
own it can't tell a real lived edge from an administrative line that means nothing on the street.
Only the 4 street views can confirm whether the two sides are actually different. This task turns
the satellite's weakness (it sees lines that may not be real) into the whole point of the question.

**Useful for:** finding where neighborhoods *actually* change vs. where a map says they do — real
problem in planning and neighborhood analysis.

---

## What they all share
- **Input:** always 4 street-view photos + 1 satellite image.
- **Answer:** 4 plain text options (no "point at a direction").
- **The trick:** every task crosses a *feeling* (street-only) with a *hidden layout fact*
  (satellite-only) that don't correlate — so one view is never enough.
- **Common, not rare:** they fire on ordinary residential/mixed streets, not on rare features.

## One thing to check before building
Validate each task on a small pilot: a satellite-only model AND a street-only model should both do
clearly worse than the both-views model. If one view alone already scores high, the task isn't real
fusion and we drop or fix it.
