# EOLLM Fusion Task Set — Final Shortlist Spec

**Purpose:** the candidate fusion tasks that survive ALL constraints, with exact label
derivation, for team review. Produced by: 1 generator + 1 constraint-clean regenerator +
2 adversarial red-teams (shortcut-leak + buildability), all Opus.

## The bar every task here passes
1. **Genuine fusion** — Y = g(A,B); passes the acid test (given the referent free, still two-view).
2. **No correlation leak** — map-claim and street-reality can genuinely DIVERGE, so a single-view
   model can't exploit a real-world correlation (this is what killed height/retail/depth tasks).
3. **Reliable OSM labels** — road network + `highway` class + topology + `bridge`/`tunnel`/`layer`
   + (for some) `landuse`/`leisure` polygons. NO storey counts, NO `shop=*`/POI, NO building
   polygons unless flagged.
4. **About urbanization** — connectivity, accessibility, severance, street function, urban form.
5. **Not a consistency/mismatch check** — output is a new category/direction, not "do views agree."
6. **City-prior balanced** — answer balanceable ~25/25/25/25 WITHIN each city (construction rule).

## Buildability ground-truth (from reading the actual pipeline)
- ✅ **Roads have real geometry** (`out body geom`) and **road bearing is on 100% of records**.
  Frame orientation is deterministic: fwd=bearing, right=bearing−90, bwd=bearing+180,
  left=bearing+90.
- ❌ **Buildings are stored as CENTROIDS only** (`out tags center`) — no footprint polygons.
- ❌ **`shop=*` is never queried**; POI = `amenity` nodes only (undercounts retail badly).
- ❌ Stored OSM is **aggregate-only** (counts + 200 m median levels) — no per-element values.
- ⇒ **Tier-1 tasks below need ONLY road data you already fetch** (+ a topology graph build, no
  new API). **Tier-2 needs a fresh Overpass re-query** (`landuse`/`leisure` polygons or
  `highway=steps` if not retained) — flagged per task.

---

# TIER 1 — buildable from road graph + bearing, no new API, cleanest labels

## T1. Route severance — "the map says a road continues; can you actually drive through?"  ★ FLAGSHIP
**Q:** "The map shows a road leaving the marked point toward the {forward/left} direction. Looking
that way in the street view, what does the street physically allow?"
**Options:** A) Drivable through-road to the next street; B) Pedestrian-only (steps/bollards/path);
C) Dead-end / physical barrier (gate, wall); D) Blocked — under construction / temporarily closed.

- **Sat fails:** OSM renders the line as continuing; top-down can't see bollards, steps, a locked
  gate, or that the "road" is really a footpath.
- **SV fails:** sees the bollards/steps/gate but can't tell a cul-de-sac from a through-road — they
  look identical from the curb; needs the map to know a route is *claimed* onward.
- **Acid test PASS:** map asserts connectivity, street adjudicates whether it's real for a car —
  neither view contains "claimed-route × physical-permission" alone.
- **No correlation leak:** the value of this task IS the divergence (a topological road that's
  actually stairs). The satellite *cannot* correlate its way out — OSM/imagery render it normal.
- **Label (reliable, road-only):** build the road graph from the `highway` ways already fetched.
  Take the way leaving the dot along the chosen bearing. (B) if that way is
  `highway=pedestrian/footway/steps/path`; (C) if it ends at a degree-1 node (dead-end) or a
  `barrier`; (A) if `highway=service/residential/...` continues to a degree≥3 node. (D) is
  imagery-only (construction) — treat as a rare class, label via a vision/human pass or omit.
  *Caveat:* the (A)/(B)/(C) split is pure graph+class (clean); (D) needs vision — ship A/B/C as
  the reliable 3-way+chance, or keep D small and human-verified.
- **Urban value:** the #1 routing/last-mile failure — "the map says turn here, you can't."
- **City balance:** balanceable; steps/dead-ends/through-roads exist in every city.

## T2. Grade separation — "do these two routes actually connect here, or stack?"
**Q:** "The map shows two routes meeting at this point. On the ground, do they meet at the same
level (you can move between them), or is one grade-separated so there's no connection here?"
**Options:** A) At-grade — you can cross/turn here; B) Road bridges OVER (no connection here);
C) Road dips UNDER (no connection here); D) Rail/path crosses, vehicles cannot.

- **Sat fails:** in 2-D top-down an overpass and an at-grade junction draw the SAME crossing lines
  — the third dimension is invisible.
- **SV fails:** sees the bridge soffit / road climbing away / level crossing, but can't tell the
  two map lines are meant to be one junction.
- **Acid test PASS:** map asserts "two routes coincide here," street reveals interconnect vs stack.
  Pure 2-D-vs-3-D synergy; no single-view correlate (that's the whole point).
- **Label (reliable):** crossing ways + shared-node test + `bridge=yes`/`tunnel=yes`/`layer`.
  Share a node → A. Cross in geometry, no shared node, one carries `bridge`/`layer>0` → B;
  `tunnel`/`layer<0` → C; crossing way is `railway`/`highway=footway` → D. `bridge`/`tunnel`/
  `layer` are topology-critical and densely tagged on grade separations.
- **Urban value:** grade separation is a primary cause of urban severance; routers + pedestrians
  both depend on it.
- **City balance:** balanceable by sampling crossings; needs locations AT crossings (yield filter).

## T3. Street function — "network role vs eye-level regime"
**Q:** "Given how this street sits in the surrounding network (map) and how it looks on the ground,
what role does it actually play in the {forward} direction?"
**Options:** A) Through-traffic artery (multi-lane, moving traffic); B) Local access street (calm,
residential/parking); C) Shared / pedestrian-priority space; D) Service / back lane.

- **Sat fails:** reveals network position (high- vs low-betweenness connector) but a wide quiet
  street vs a narrow busy one is unresolved from above.
- **SV fails:** reveals the eye-level regime (lanes, traffic, calming, shared surface) but not the
  street's role in the wider network.
- **Acid test PASS:** position (map) × regime (street) are independent and can diverge — a
  topologically major street that's been pedestrianized; a minor street acting as a rat-run.
- **Label (reliable):** `highway` class + simple betweenness/degree over the road graph.
  primary/secondary/trunk + high through-connectivity → A; residential/living_street → B;
  pedestrian/living_street → C; service (esp. `service=alley`) → D.
- **⚠ Leak watch (red-team):** `highway` class correlates with street appearance, so a single view
  may guess above chance. MITIGATE by oversampling the DIVERGENT cells (pedestrianized arterial;
  busy minor street) so class⊥appearance in the labelled distribution. Verify with the gate
  (require sat_only AND sv_only ≤ chance+8 on a matched held-out set) before scaling.
- **Urban value:** street-function classification — core to traffic calming, safety, street design.

---

# TIER 2 — buildable but needs a fresh Overpass re-query (polygons / extra layers)

## T4. Land-use transition direction — "which way does the character change?"
**Q:** "This point sits on a land-use boundary. In which street-view direction does the built
character actually change?"  **Options:** A) along_fwd; B) along_bwd; C) cross_left; D) cross_right.

- **Sat fails:** gives the `landuse` polygons and the boundary AXIS, but residential vs
  light-industrial rooftops can look alike from above.
- **SV fails:** reads "houses here, warehouses there" at eye level but doesn't know a boundary
  exists or which compass direction is which.
- **Acid test PASS:** map encodes the boundary axis, street confirms which side flips and how —
  map alone can't pick the frame; SV alone can't know there's a boundary.
- **Label:** edge between two `landuse` polygons through/near the dot; outward normal → nearest of
  4 frame bearings = answer. **Needs `landuse` polygon re-query** (moderately dense layer).
- **Urban value:** urban-form / zoning transition legibility.
- **⚠ Leak watch:** if the two land uses look distinct from above, sat-alone may pick the axis.
  Direction (4-way among frames) is harder to leak than the binary; keep it as direction-MCQ.

## T5. Public-space access direction — "which way is the park's real entrance?"
**Q:** "The map shows a park/green space touching this point. Which street-view direction faces its
actual public access (open edge/entrance), not a fence or back boundary?"
**Options:** A) along_fwd; B) along_bwd; C) cross_left; D) cross_right.

- **Sat fails:** shows the `leisure=park` polygon and which side the green is on, but can't tell an
  open lawn edge from a railing/hedge — both read as "green meets street."
- **SV fails:** sees railing vs open path but doesn't know a park exists that way or its extent.
- **Acid test PASS:** map locates the adjacency, street tests its accessibility.
- **Label:** `leisure=park`/`landuse=recreation_ground` polygons + bearing → frame facing the park.
  STRONG form needs a `highway=footway`/`barrier`/`entrance` crossing the boundary in that frame's
  cone. **Needs polygon + path/barrier re-query.** Without access nodes, label = "park-facing
  direction" (still useful, slightly weaker).
- **Urban value:** public-space accessibility / walkability / equity.

---

# REJECTED (recorded so the team sees the reasoning)
- **Height/storey aggregation, facade-deep×tall** — needs per-building `building:levels` (sparse)
  AND footprint polygons (not stored); footprint-area↔height correlation leaks to sat-only.
- **Retail fraction** — `shop=*` not queried; POI completeness worst OSM layer; retail legible
  from BOTH views (redundant).
- **Relative ordering (which landmark closer)** — apparent size in the frame encodes depth →
  sv-only solves it; identity↔OSM matching needs human curation; yield ~3–6%.
- **Setback contradiction, corner-entrance** — consistency-check flavored / need polygons +
  entrance tags; discriminating axis is vision-only.
- **Which-road-am-I-on** — fails acid test (given the road free, it's localize-then-name).

# CROSS-CUTTING CONSTRUCTION RULES (apply to whatever the team picks)
1. **Balance the answer ~25/25/25/25 WITHIN each city** (not globally) — kills the
   satellite-reveals-city → modal-answer shortcut (the most dangerous leak found).
2. **Oversample DIVERGENT cells** (map-says-X-but-street-shows-Y) so map⊥appearance in the labels.
3. **Randomize satellite orientation** (don't always put north up) so bearing→frame is read,
   not memorized.
4. **Validate with the gate BEFORE scaling:** require `acc_blind`, `acc_sat_only`, `acc_sv_only`
   all ≤ chance+8 on a matched ~150-question pilot; only then generate the full set.
   (`reasoning_distill/complementarity_gate.py`.)
5. **Yield is the real risk** for T2/T4/T5 (need locations AT crossings/boundaries/parks) — expect
   a few hundred to low-thousands of questions per task; report what's dropped.
