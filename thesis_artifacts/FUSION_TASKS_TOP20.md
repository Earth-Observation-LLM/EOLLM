# EOLLM Fusion Tasks — Top 20 (Final Shortlist for Team Review)

**Synthesis of 5 independent model proposals (Claude, Grok, GPT, Gemini, Qwen, Kimi) +
4 internal adversarial agents.** Ranked by your 3 criteria — **broad necessity**, **real
usability**, **OSM-grounding** — plus the fusion bar.

## FIXED SPEC (applies to every task)
- **Input:** always **4 street-view frames + 1 satellite image** (red dot marks the point).
- **Answer:** **4 text labels** (multiple choice). NO "pick a frame" answers. Where a task is
  directional, the labels name **what is there** ("toward the main road", "toward the dead-end",
  "toward the water"), NEVER the frame name ("cross_left"). This forces the model to *identify* the
  referent, not emit a frame index.
- **Fusion bar (acceptance):** a task qualifies iff **full-input accuracy > EACH single-view
  accuracy by our gate margin** — single view may be decent, it just must be clearly beaten:
  `synergy = full − max(sat, sv) ≥ +8 pts` AND `fusion-win = P(full right ∧ both singles wrong)
  ≥ 15%` AND `full > both singles`. Single-view being *possible but insufficient* is exactly the
  target — graceful degradation, not chance-level. The gate decides per task before scaling.
- **City-balance (mandatory):** balance answer labels ~25/25/25/25 *within each city* to kill the
  satellite-reveals-city → modal-answer prior leak.

## How to read this table
- **Conv.** = how many of the 5 external models independently proposed this family (consensus signal).
- **Label** = ground-truth source: 🟢 **Auto** (dense OSM geometry/tags + bearing, no vision) ·
  🟡 **Auto-filter + vision-answer** (OSM selects candidates; VLM/human assigns the eye-level label).

---

## TIER A — Build first. Highest necessity × usability × OSM-grounding, cleanest fusion.

### 1. Forward link form (route severance) ★ FLAGSHIP — consensus #1 (all 6)
**Q:** "Standing at the marked point, the map shows a road continuing in the direction you are facing.
Physically, what is that onward route?"
**Labels:** Drivable through-road · Pedestrian-only steps/stairs · Path or ramp (non-vehicular) ·
Dead-end / private (no public through-route).
**Single-view but bad:** sat-only guesses "road" from the line (often wrong when it's stairs/gated);
sv-only sees the surface but mis-reads cul-de-sac vs through. **Full needs both:** map's claimed
continuation × street's physical form.
**Label:** 🟢 Auto — `highway` class of the bearing-matched forward way + node-degree topology
(`steps`→stairs; `footway/path/cycleway`→path; degree-1/`noexit`→dead-end; road class→drivable).
**Necessity:** THE routing/last-mile failure ("map says go, you can't"). Universal.

### 2. Forward crossing grade relation ★ — consensus #1 (all 6)
**Q:** "Following the road forward, it meets another road, railway, or waterway. How do they relate
physically?"
**Labels:** At-grade (same level) · Road passes OVER (bridge/flyover) · Road passes UNDER
(tunnel/underpass) · No major crossing ahead.
**Single-view but bad:** sat-only has NO Z-axis (over/under/at-grade look identical from above) →
near-random among the three; sv-only sees a bridge/tunnel but can't bind it to the mapped crossing.
**Full needs both:** map says "crosses X here" × street says "who's above whom."
**Label:** 🟢 Auto — forward way ∩ (`waterway`/`railway`/major `highway`); `bridge=*`→over,
`tunnel=*`/`layer<0`→under, shared node→at-grade. Densely tagged (topology-critical).
**Necessity:** Grade separation = primary cause of urban severance. Strongest "single view physically
cannot" claim in the set.

### 3. Barrier crossing reachability (severance) — consensus 5/5  [directional → text labels]
**Q:** "A river, railway, or major road runs near the marked point. Where is the nearest crossing a
pedestrian or cyclist can actually use?"
**Labels:** Toward the forward road · Toward the road behind you · Toward the left cross-street ·
Toward the right cross-street. *(named by the network feature, gate-checked that a model can't read
the label off frame geometry alone)*
**Single-view but bad:** sat-only can't confirm a crossing permits pedestrians / sees no underpass;
sv-only sees only foreground, misses a crossing around the corner. **Full needs both.**
**Label:** 🟢 Auto — nearest `waterway`/`railway`/`motorway/trunk/primary` barrier; crossings = ways
with `bridge`/`tunnel`/`ferry` or `footway/path/cycleway` spanning it (foot-access filter); bearing→label.
**Necessity:** Severance + transport equity ("wrong side of the tracks"). Core walkability metric.

### 4. Filtered permeability / modal connectivity — consensus 4/5
**Q:** "The map shows the forward road continuing. What is its real connectivity for different travel
modes?"
**Labels:** Through-route for all modes · Vehicular dead-end but pedestrian/cycle through-route
(filtered) · Absolute dead-end for all modes · Pedestrian/cycle-only by design.
**Single-view but bad:** sat-only sees a continuous pavement ribbon (bollards sub-pixel) → over-predicts
"through for all"; sv-only sees bollards but can't tell if it connects through beyond them.
**Label:** 🟡 Auto-filter (`highway` topology + connectivity to a through-street) + vision for the
bollard/gate state. **Necessity:** 15-minute-city / Low-Traffic-Neighbourhood core metric.

### 5. Onward route mode-and-continuity status — consensus 4/5 (GPT/Qwen "continuation status")
**Q:** "For the road that continues forward from the marked point, what is its combined status?"
**Labels:** Vehicular through-connection · Pedestrian/cycle-only through-connection · Vehicular local
access or dead-end · Pedestrian/cycle-only local access or dead-end.
**Single-view but bad:** sat-only gets through-vs-dead-end (topology) but not mode (bollards invisible);
sv-only gets mode (motorable vs stairs) but not through-vs-dead-end. **Each owns ONE bit → full needs
both.** This 2×2 is the cleanest "each view supplies one independent bit" structure.
**Label:** 🟡 Auto-filter (graph reachability + `highway` class) + vision for mode where untagged.
**Necessity:** Navigation, emergency access, walkability continuity.

---

## TIER B — Strong, build second. Slightly lower yield or needs a polygon re-query.

### 6. Misclassified-as-stairs detector (routing-bug) — consensus 4/5  [directional → text]
**Q:** "Among the routes the map shows leaving this junction (none of which the map labels as steps),
which one is physically a staircase on the ground?"
**Labels:** The forward route · The route behind · The left route · The right route. *(text-named; the
point is divergence: map says non-stairs, street shows stairs)*
**Single-view but bad:** sat-only can't see steps (under-tagged); sv-only "where are stairs" is partly
solvable but can't tell which the map mis-labels. **Full needs both.**
**Label:** 🟡 Auto-filter (candidate ways NOT tagged `steps`) + vision. ⚠ low yield → mine hilly cities.
**Necessity:** Famous "routed a wheelchair onto stairs" bug. High accessibility value.

### 7. Adjacent open-space access type — consensus 5/5  [recast directional → type labels]
**Q:** "The map shows a park / green / water space touching this point. What kind of access does its
edge present here on the ground?"
**Labels:** Open, walk-in edge or gateway · Gated/controlled entrance · Fenced / walled / sealed edge ·
No usable edge (private buildings between road and space).
**Single-view but bad:** sat-only sees the polygon but not whether the edge is open vs walled; sv-only
sees a fence/gate but doesn't know it's the mapped park or its extent. **Full needs both.**
**Label:** 🟡 polygon (`leisure=park`/`recreation_ground`/`natural=water`) + "edge touched by a public
`highway`" proxy + vision. **Needs polygon re-query.** **Necessity:** Green/blue-space access equity.

### 8. Street function (network role × eye-level regime) — consensus 4/5
**Q:** "Given how this street sits in the surrounding network and how it looks on the ground, what role
does it actually play?"
**Labels:** Through-traffic artery · Local access street · Shared / pedestrian-priority space ·
Service / back lane.
**Single-view but bad:** sat-only knows network position, not regime; sv-only knows regime, not role.
**They diverge** (pedestrianized arterial; minor rat-run) → full needs both.
**Label:** 🟡 `highway` class + betweenness/degree + vision for calming/regime. ⚠ class↔appearance
correlates → oversample divergent cells, gate-verify. **Necessity:** Street-function / "stroad" / traffic
calming policy.

### 9. Transverse barrier identity (rail vs water) — consensus 2/5
**Q:** "Two linear barriers — a railway AND a canal/river — run near this point. Which one does the
visible crossing structure actually cross?"
**Labels:** The railway · The waterway · Both have a crossing here · Neither (no usable crossing here).
**Single-view but bad:** sat-only can't confirm a physical crossing at this point; sv-only — a narrow
canal and single-track rail look near-identical under a bridge. **Full needs both.**
**Label:** 🟡 `railway=*` + `waterway=*` co-located + vision. Restrict to rail+canal legacy cities.
**Necessity:** Which infrastructure is the active severance at a neighborhood edge.

### 10. Land-use transition character — consensus 3/5  [recast directional → type labels]
**Q:** "This point sits on a mapped land-use boundary. What is the built-character change you actually
see across it?"
**Labels:** Residential → commercial/retail · Residential → industrial · Built-up → open/green ·
No visible change (boundary is administrative only).
**Single-view but bad:** sat-only gives the boundary but residential vs light-industrial roofs look alike;
sv-only reads character but doesn't know a boundary exists. **Full needs both.**
**Label:** 🟡 edge between two `landuse` polygons + vision. **Needs polygon re-query.** ⚠ correlation
risk → sample administrative-boundary cases. **Necessity:** Zoning vs perceived boundary; district legibility.

---

## TIER C — Useful, narrower or more clustered. Build if capacity allows.

### 11. Waterfront access type — conv 4/5  [recast → type labels]
**Q:** "A river/canal/coast is mapped near this point. What does its edge offer on the ground here?"
**Labels:** Public promenade/path along the water · Road/bridge access to the water · Steps down to the
water · No public access (walled/private/no route). **Label:** 🟡 `waterway`/`coastline` + buffered
`highway=footway/path` + vision. **Necessity:** Blue-space access, flood/heat-edge connectivity.

### 12. Nearest pedestrian crossing reachability — conv 2/5  [directional → text]
**Q:** "Crossing the road you're on, where is the nearest marked or signalized pedestrian crossing?"
**Labels:** Just ahead along the road · Behind you along the road · Across to the left side · Across to
the right side. **Label:** 🟢 mostly Auto — `highway=crossing` nodes on the road + bearing→label.
**Necessity:** Pedestrian safety / Vision Zero / accessibility for elderly, children, wheelchair users.

### 13. Transit access reachability — conv 3/5  [directional → text]
**Q:** "A transit stop/platform is mapped nearby. Where does a pedestrian go to reach it?"
**Labels:** Ahead along this road · Back along this road · Onto the left cross-street · Onto the right
cross-street. **Label:** 🟡 `bus_stop`/`platform` projected to sidewalk + vision. **Necessity:**
First/last-mile multimodal accessibility.

### 14. Cycleway continuity status — conv 1/5
**Q:** "The map tags this road as a cycleway. How does the physical cycle provision continue?"
**Labels:** Continuous in both directions · Continues ahead only · Continues behind only · Discontinuous
/ signed-only (no physical lane). **Label:** 🟡 `highway=cycleway`/`cycleway=*` + vision per frame.
**Necessity:** Cycle-network continuity = top predictor of bike mode-share.

### 15. Arterial edge severance type — conv 2/5
**Q:** "This is a major arterial. How does its edge relate to the adjacent neighborhood?"
**Labels:** Direct frontage/driveways onto the carriageway · Buffered by a parallel service road · Walled
/ sound-barrier severed · Open natural buffer. **Label:** 🟡 `highway=primary/secondary` + parallel
`service` / adjacency + vision. **Necessity:** Community severance, noise mitigation.

### 16. Mapped stairway approach — conv 3/5  [directional → text]
**Q:** "A mapped stairway is nearby. Where do you approach it from your road?"
**Labels:** Straight ahead · Behind you · To the left · To the right. **Label:** 🟢 Auto — `highway=steps`
nearest + bearing (existence OSM-given; street confirms). **Necessity:** Accessibility (ramps vs stairs),
emergency egress. Clustered to hilly cities.

### 17. Junction open-exit fraction — conv 2/5
**Q:** "The map shows several roads meeting here. How many are actually open to vehicles on the ground?"
**Labels:** All of them · Most (at least half) · Few (less than half) · None / all blocked. **Label:** 🟡
incident `highway` count + bearing-match + vision per exit. **Necessity:** "Paper street" / real vs
mapped connectivity audit.

### 18. Pedestrian-crossing protection at a junction — conv 1/5
**Q:** "How does this junction's size relate to the pedestrian protection actually provided?"
**Labels:** Compact and well-protected · Broad but with median refuges · Broad with no protection (stroad)
· Physically severed (fenced, no at-grade crossing). **Label:** 🟡 lane-count/area + vision. ⚠ control
footprint across classes. **Necessity:** Vision-Zero pedestrian safety.

### 19. Block cut-through presence — conv 3/5  [recast → type labels]
**Q:** "The map shows a dense building block at this point. What kind of pedestrian route through/around
it does the street reveal?"
**Labels:** An open passage cutting through the block · A gated/private passage (no public through) ·
Continuous building wall (no cut-through) · Only the perimeter sidewalk (no interior route). **Label:** 🟡
footpath attaching two roads, building-flanked, + vision. **Needs polygon re-query.** **Necessity:** Urban
grain / permeability in dense cores.

### 20. Service/delivery access presence — conv 1/5  [recast → type labels]
**Q:** "What kind of vehicle access does the building/lot at this point have from the street?"
**Labels:** Direct driveway/service entrance on this street · Access via a rear service lane · Parking-lot
entrance · No vehicle access (pedestrian frontage only). **Label:** 🟡 `highway=service` into building/
`landuse=parking` polygon + vision. **Needs polygon re-query.** **Necessity:** Freight, curb, parking policy.

---

## Recommended build order (for the meeting)
**#1 (Forward link form) and #2 (Grade separation) first** — top every model's ranking, most
OSM-automatic (🟢, least vision), cleanest "single view is insufficient" argument, universal
routing/severance need, and **no polygon re-query** (road graph over data you already fetch). Then
**#3, #4, #5** for severance/connectivity breadth (also road-graph-only). Tier B/C add breadth but many
need a one-time Overpass re-query (`landuse`/`leisure`/building polygons) + a vision-labelling pass.

## Two rules for ALL 20 (non-negotiable)
1. **Balance answer labels ~25/25/25/25 WITHIN each city** — kills the satellite→city→modal-answer leak.
2. **Gate before scaling:** run `reasoning_distill/complementarity_gate.py` on a ~150-q pilot; require
   `full > each single by ≥8 pts` and `fusion-win ≥ 15%`. Single-view *may* be decent — it just must
   be clearly beaten. A task that fails the gate isn't fusion, however good the design argument sounds.

## Honesty note for the team (the 🟡 tasks)
🟡 tasks use OSM to *select & filter* candidates but a VLM/human to assign the *eye-level answer label*
(is it really stairs / a gate / an open edge?). Standard practice (UrBench, MMStar do this), but it
means **labeling cost is real** and the benchmark is **curated, not fully auto-generated**. The most
automatic (least vision) tasks — **#1, #2, #3, #12, #16** — are the safest "we can label at scale" bets.
