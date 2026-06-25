# Design Brief: Satellite + Street-View FUSION VQA Tasks

**You are a creative benchmark designer. I need new VQA task designs for an urban-understanding
dataset. Read the full context, then propose tasks that satisfy ALL FOUR hard constraints below.
Be scientifically creative, but reject ruthlessly anything that violates a constraint.**

---

## 1. What the dataset is

**EOLLM** — an urban-understanding VQA benchmark. Each sample is ONE geographic location that
comes with **two complementary image perspectives**:

- **Satellite**: ONE top-down aerial image. A **red dot** marks the exact location point.
- **Street-view**: FOUR ground-level photos taken at that point, oriented to the road:
  `along_fwd` (looking forward along the road), `along_bwd` (backward), `cross_left` (90° left),
  `cross_right` (90° right). The **road bearing** (compass heading of the road) is stored and
  available on 100% of records. The 4 frames are ~90° each and tile the full 360°.

Each location also has: `lat`, `lon`, and recoverable **OpenStreetMap (OSM)** geometry.
Questions are **4-option multiple choice**. There are **~7,900 locations** across many world
cities. The thesis of the dataset: urban understanding needs BOTH a bird's-eye view AND a
street-level view, and a model must **FUSE** them.

## 2. The problem we are fixing (why we need your help)

We ran an image-ablation on a strong VLM. For each question we measured accuracy under four
conditions: **full** (both views), **sat_only**, **sv_only**, **blind** (no images). On our
current 9 attribute tasks (land_use, building_height, urban_density, road_type, road_surface,
green_space, amenity_richness, junction_type, transit_density):

| condition | accuracy |
|---|---|
| full (both) | 55.0% |
| sat_only | 54.2% |
| sv_only | 53.7% |
| blind | 37.9% |

**Giving the model BOTH views beats the best SINGLE view by only +0.8 points.** Deeper analysis:
the two views predict the SAME answer 67% of the time (redundant); fusion *uniquely* solves only
2.9% of questions while it *hurts* on 13.1%. The second perspective is dead weight.

**Root cause:** every current label is `bin(one OSM scalar in a radius)` — e.g. building_height
= bin(median `building:levels`), urban_density = bin(building count). A single number → a single
label. Whichever view best correlates with that scalar recovers the answer; the other view is
redundant **by construction**. The task never forces the model to take information that ONLY
exists in view A and combine it with information that ONLY exists in view B.

## 3. The design law you must satisfy

A task genuinely requires fusion iff the answer is a RELATION:

> **Y = g(A, B)**, where the answer depends on (i) a referent or measurement that exists ONLY in
> the satellite/plan-view AND (ii) a property that exists ONLY in the street-view/eye-level —
> such that deleting EITHER view leaves the answer undetermined.

- **Satellite/plan-view owns** (measured, citable asymmetry): road network & topology,
  connectivity, building footprints & counts, block structure, spatial extent, relative position
  / which-of-several, what's behind a facade, plan-view geometry.
- **Street-view/eye-level owns**: what the street physically affords — function, access, barrier,
  surface, activity, signage, what is actually there at ground level — i.e. things the roof
  occludes from above.

**THE ACID TEST (apply to every task you propose):** *If I handed the model the correct referent
for FREE (perfect localization), would it STILL need both views to answer?*
- If **NO** → the satellite merely POINTS and the street-view merely NAMES → this is
  "localize-then-name", two sequential single-view steps, NOT fusion → **REJECT**.
- If **YES** → the answer is a genuine joint function → keep.

**THE CORRELATION TRAP (a deeper failure than the acid test — read carefully).** A task can pass
the acid test and STILL be single-view-solvable, because in the real built environment the two
views' signals are *physically correlated*. Examples that killed earlier candidates:
- "Are buildings uniform or varying in height across the block?" — footprint-area variance (fully
  in satellite) correlates with height variance (r≈0.5–0.7), so satellite-alone predicts the
  answer above chance WITHOUT seeing storeys.
- "Which landmark is closer?" — apparent size in the street-view frame encodes distance
  (monocular depth), so street-view-alone answers it WITHOUT the map.
- "Is this street retail or residential?" — both views independently nail land use; redundant.
The test that matters: **is the answer-class statistically independent of each single view's
content?** If a strong satellite-only OR street-only model could guess above chance via a
real-world correlation, the task leaks. The strongest tasks are ones where the map's claim and
the street's reality can genuinely *diverge* (a topological road that's actually stairs; a
"connected" junction that's grade-separated) — divergence breaks the correlation.

**THE CITY-PRIOR LEAK (affects ALL tasks — a construction requirement).** The satellite tile
reveals the *city* (grid vs canal vs superblock vs informal morphology), and city is a massive
prior on every label (Manhattan = tall, suburban US = setback, European core = built-to-line).
A model can shortcut via "which city → modal answer" WITHOUT fusing. Any task must be labelable
such that the answer can be **balanced ~25/25/25/25 WITHIN each city** — not just globally. If a
task's answer is near-deterministic given the city, it leaks regardless of design.

## 4. The FOUR hard constraints (a task violating ANY one is rejected)

**CONSTRAINT 1 — GENUINE FUSION.** Must pass the acid test above. Y = g(A,B) is a relation, not
a single-view attribute with the other view as decoration.

**CONSTRAINT 2 — RELIABLE OSM LABELS ONLY.** The ground-truth label must be computable from
**dense, well-populated** OSM layers:
- ✅ ALLOWED (dense/reliable): road ways + `highway` class + connectivity/topology, intersection
  geometry, building footprint polygons & counts, `landuse` polygons, `leisure=park`/green
  polygons, water polygons, transit nodes, geometric distance/bearing/extent.
- ❌ FORBIDDEN (sparse/stale/uneven by city): `building:levels` (storey counts),
  `building:material`, `shop=*` / POI completeness, `building:use`, anything requiring a count of
  amenities. "How many commercial floors" is exactly the kind of UNRELIABLE label to avoid.
- If a task's label needs storey counts, material, or POI completeness → redesign or drop it.

**CONSTRAINT 3 — ABOUT URBANIZATION, GENUINELY USEFUL.** The question must be one a city planner,
resident, or navigation/accessibility system would actually care about: walkability,
connectivity, accessibility, severance/barriers, land-use structure, public-space access, street
function, urban form. NOT a geometry puzzle ("which frame shows the arrow points at"), NOT
bookkeeping, NOT a contrived spatial riddle. It must test real urban understanding.

**CONSTRAINT 4 — NOT A CONSISTENCY / MISMATCH CHECK.** The dataset ALREADY has a cross-view
"mismatch / matching" family (questions of the form "do these two views show the same place" /
"does the street confirm what the map implies" / "do the two views agree"). Do NOT propose
agreement/consistency/contradiction tasks — they are redundant. The fusion must produce a NEW
answer (a category, a direction, a count, an ordering), not a yes/no agreement between views.

## 5. Dead-ends we already explored (do NOT re-propose these)

- **Storey/height aggregation** ("are the buildings uniform or varying in height across the
  block") — violates Constraint 2 (needs `building:levels`).
- **Retail / amenity fraction** ("are most ground-floor units retail") — violates Constraint 2
  (POI completeness is the worst OSM layer).
- **Setback contradiction / "does the street confirm the map's setback"** — violates Constraint 4
  (consistency check).
- **Rooftop reveal / block-interior content** ("what's on the roof / inside the courtyard") —
  fails the acid test: that info is in the SATELLITE alone; street-view only localizes →
  localize-then-name.
- **Facade material × footprint shape** — violates Constraint 2 (material is sparse) and is a
  conjunction not a true relation.

## 6. Promising directions (seeds — push further, don't just copy)

Think hard about **road connectivity × what the street physically affords**, since the road
network is the MOST reliable OSM layer AND the most urbanization-relevant:

- **Connection-type at a map junction.** The map shows a route leaving the dot in some direction.
  What does the street physically afford that way? {drivable through-road · pedestrian-only
  stairs/steps · ramp / non-vehicular · blocked/dead-end (gate, bollards, wall)}. The MAP says a
  route is claimed; only the STREET says what kind of route it physically is; the BEARING binds
  the map direction to the right frame. (The classic "the routing engine thinks this is a street
  but it's a staircase" problem.)
- **Public-space access direction.** The map shows a park / plaza / water adjacent to the dot. In
  WHICH street-view direction is its actual public, usable entrance/edge (vs. a fenced/blank
  back)? Plan-view says where the green is; eye-level says where you can actually get in.
- **Urban-form boundary direction.** The dot sits on a boundary the map encodes (e.g. built block
  ↔ open space, dense ↔ sparse, one land-use ↔ another). In which street-view direction does the
  built character actually change on the ground?
- **Severance / crossability.** The map shows a road/rail/water the dot is beside; can a
  pedestrian actually cross it here, per the street (crossing, bridge, underpass vs. barrier)?

For each, the satellite/map supplies the **claimed structure / which-direction / which-of-several**
and the street-view supplies the **physical affordance / access / barrier** — and the answer is a
direction or a category that lives in neither view alone.

## 7. What to produce

Propose **8–12 candidate tasks**. For EACH:
1. **Task name** + the exact 4-option MCQ **question template** (as shown to the model).
2. The **4 answer options**.
3. **Why satellite-alone fails** and **why street-view-alone fails** — be specific and physical.
4. **ACID TEST verdict**: given the referent for free, is it STILL two-view? One sentence on why
   it's genuine synergy and not localize-then-name.
5. **Label derivation** from RELIABLE OSM + geometry + bearing ONLY (name the exact dense OSM
   layer; no forbidden tags). If any step needs a vision model or human pass, flag it explicitly.
6. **Urbanization value**: the real urban question it answers.
7. **Confirm** it is NOT a consistency/mismatch check (Constraint 4) and does NOT use forbidden
   tags (Constraint 2).

Then **RANK** the candidates by (fusion-cleanliness × OSM-label-reliability × urbanization-value),
and name the **single strongest task** with a one-line argument for why it provably requires
fusion. Reject ruthlessly anything that leans on storey counts, POI completeness, is a
consistency check, or fails the acid test.

**Bonus if you can:** for your top task, give the one-line *proof sketch* that no single-view
oracle (however perfect at its own view) can exceed chance on the discriminating answer classes —
i.e. argue the answer information is genuinely absent from each marginal view.
