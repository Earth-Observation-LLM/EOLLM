# EOLLM — 5 Fusion Tasks (common, useful, fusion-forced)

**The design rule (learned from rejecting 20):**
> **Satellite = the WHERE / HOW-BIG** (context, scale, extent, layout — coarse, ambiguous).
> **Street-view = the WHAT-IT'S-LIKE** (use, condition, character at eye level — fine).
> The answer is **one fused TEXT category**, asked at **almost every urban point**, where the
> satellite gives an *ambiguous* reading and the street *flips/resolves* it — and the street
> alone can't supply the scale/extent. **No rare geometry. No "find the nearest X" (satellite
> localizes that = single-view). No directional answers.**

**Fixed spec:** input = **4 street-view + 1 satellite** (red dot on the point). Answer = **4 text
labels**. Referent is **the point under the dot itself**, never "the nearest something."

**How the benchmark stays fusion AND common:** the *question type* applies to ~all points, but we
**mine the instances where the two views disagree** — generate sat-only and sv-only predictions
during construction and keep the cases where one view alone is wrong (MMStar-style visual-necessity
filter). Validate every task on a ~150-q pilot with `complementarity_gate.py`: require
`full > each single by ≥8 pts` before scaling.

---

## TASK 1 — Open-space character ★ BEST
**Q:** "The marked point is on or beside an open (unbuilt) space. What kind of open space is it?"
**Labels:** Maintained green park / recreation · Hard paved plaza or public square · Parking lot ·
Vacant / scrub / derelict ground.

- **Satellite insufficient:** top-down, an unbuilt patch is just "not-a-building." A green tinge =
  manicured park OR overgrown lot OR wooded embankment OR sports pitch. A grey patch = busy plaza OR
  dead parking lot OR builder's yard. Satellite reliably gives *open space + how big*, not *what it is*.
- **Street insufficient:** you see grass or asphalt in front of you but can't tell a 2 m verge from
  the edge of a 5-hectare park — **extent is invisible at eye level**.
- **Full needs both:** satellite supplies *extent* (excludes tiny yards from "park"); street supplies
  *character* (park vs scrub vs parking vs plaza).
- **Common:** >40–50% of points fall on/beside some open space (verges, lots, plazas, parks).
- **Label:** 🟡 OSM `landuse`/`leisure` (park/grass/recreation/parking) + footprint-absence gives the
  coarse seed; **vision pass** separates maintained-park vs scrub-vs-derelict and busy-vs-dead.
- **Use:** open-space inventory, green-access equity, spotting vacant/underused land for redevelopment.

## TASK 2 — Built intensity & enclosure
**Q:** "Standing at the marked point, how built-up and enclosed is the setting?"
**Labels:** Open / low-rise (sky-dominated) · Mid-rise continuous street wall · Dense high-rise canyon ·
Sparse / gap-toothed (vacant or broken frontage).

- **Satellite insufficient:** footprints give plan *coverage* but — with `building:levels` barred —
  **top-down can't see height**. The same dense footprint grid is 2-storey terraces OR 20-storey
  slabs: identical plan, opposite enclosure.
- **Street insufficient:** the frames show how tall/continuous the walls are *here*, but not whether
  it's a one-block anomaly, a CBD core, or a lone tower in a low field — needs surrounding massing.
- **Full needs both:** satellite = horizontal density + extent; street = the vertical wall it can't see.
- **Common:** ~100% — every street point has an enclosure character.
- **Label:** 🟡 OSM building-footprint *coverage ratio* in a radius (horizontal density seed) +
  **vision pass** for the height/enclosure class.
- **Use:** urban form, walkability, canyon heat/wind microclimate, density policy, character areas.

## TASK 3 — Functional character (land use, done right)
**Q:** "Taking the satellite context with the street frames, this location is primarily:"
**Labels:** Residential neighbourhood · Commercial / mixed high-street · Industrial / logistics /
utilitarian · Civic / institutional / campus.

- **Satellite insufficient:** massing hints (big-box → industry, fine grain → housing) but a large
  flat-roof block is equally a warehouse, a supermarket, a school, or a sports hall. Massing collapses
  these.
- **Street insufficient:** frontage reveals shopfronts / loading docks / front doors / institutional
  signage — but not whether it's one shop on a residential street or a 1 km retail corridor, nor the
  campus scale. **Function needs extent.**
- **Full needs both:** satellite = grain/layout/extent; street = the frontage activity that
  disambiguates same-massing cases.
- **Common:** ~100%. This is your old `land_use` task fixed — keep only same-massing-different-function
  cases where one view is wrong.
- **Label:** 🟢/🟡 best OSM ground truth — dense `landuse` polygons (residential/commercial/industrial/
  retail) + footprint grain; **vision check** only for the ambiguous warehouse-vs-supermarket-vs-school.
- **Use:** land-use classification, zoning verification, mixed-use & employment-land tracking. Bedrock.

## TASK 4 — Street-life activity (amenity richness, de-leaked)
**Q:** "How active is street life at the marked point?"
**Labels:** Commercially busy (shops/cafés, people about) · Quietly residential · Car-dominated through-
route (few pedestrians) · Empty / vacant / inactive.

- **Satellite insufficient:** road width + building density suggest it *should* be busy, but top-down
  can't see whether the ground floor is open shops or shutters, or whether a plaza is a market or a
  dead void.
- **Street insufficient:** you see activity in front of you but can't tell a one-off busy corner from a
  sustained district — **context/extent is the satellite's job**.
- **Full needs both:** satellite = the corridor's scale/position; street = the actual eye-level activity.
- **Common:** ~100%. This is your `amenity_richness` task fixed — fused, no shop-POI dependence.
- **Label:** 🟡 OSM road class + density for the coarse seed; **vision pass** for the activity class.
  (Do NOT use `shop=*` POI counts — sparse/leaky.)
- **Use:** high-street vitality, retail-health monitoring, 15-minute-city activity mapping.

## TASK 5 — Road space allocation (who is the street for)
**Q:** "At the marked point, the road space is mostly given to:"
**Labels:** Motor traffic (wide carriageway, few people) · Mixed traffic with real pedestrian provision ·
Pedestrian-dominant / traffic-calmed · Pedestrianised / car-free.

- **Satellite insufficient:** it measures *width* + road class precisely, but width ≠ allocation. A wide
  strip is 4 lanes of cars OR a tree-lined boulevard with broad footways OR a same-coloured car-free
  promenade. Top-down can't reliably split footway vs carriageway or see bollards/calming.
- **Street insufficient:** ground frames show cars/footways/people but not whether this is a short
  calmed segment on an otherwise arterial road, nor the road's network role.
- **Full needs both:** satellite = total width + class (the envelope); street = the actual allocation
  of that envelope to cars vs people.
- **Common:** ~100% — every point is on/beside a road. ⚠ Thinner single-view margin (satellite gets
  partial credit from width+class) — rely on disagreement-mining + the gate here especially.
- **Label:** 🟡 OSM `highway` class + road geometry width (the dense road network; `pedestrian`/
  `living_street` are road-network tags, allowed) + **vision** to confirm real provision.
- **Use:** complete-streets / modal-share assessment, road-diet candidate identification, accessibility.

---

## Ranking
1. **Task 1 — Open-space character.** Purest: satellite genuinely can't resolve park/plaza/parking/
   scrub; street resolves it; street alone can't supply extent. Common, useful, cleanly orthogonal.
2. **Task 2 — Built intensity & enclosure.** Height-vs-plan-density gap is real now that levels are
   barred. Universal urban-form metric.
3. **Task 3 — Functional character.** Best OSM ground truth (dense landuse); same-massing cases give
   the street a real job. Your land_use task, fixed.
4. **Task 4 — Street-life activity.** Universal, useful; your amenity task, fused and de-leaked.
5. **Task 5 — Road space allocation.** Strong concept but thinnest single-view margin — lean hardest
   on disagreement-mining + the gate.

## The two rules that make all 5 work (non-negotiable)
1. **Mine the disagreement cases.** The question is common, but keep instances where sat-only and
   sv-only disagree (or one is confidently wrong). That's what converts a common question into a
   *fusion* benchmark — and it's exactly what `complementarity_gate.py` measures.
2. **Balance labels ~25/25/25/25 within each city**, and gate-validate a ~150-q pilot
   (`full > each single by ≥8 pts`) before scaling. Single-view may be decent — it must be clearly beaten.

## Why these beat the old 9
Same broad, useful questions (place type, density, land use, activity, street allocation) — but framed
as **coarse-satellite × fine-street**, and built by **keeping the cases where one view is wrong**. The
old tasks failed because the label was `bin(one OSM scalar)` that one view fully recovered; these put
half the answer in each view by construction.
