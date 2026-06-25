# Fusion-Forcing Task Candidates (generated + acid-tested)

Design law: **Y = g(A,B)** must be a RELATION between a plan-view-only referent/measurement
(satellite) and an eye-level-only property (street-view). Acid test: *given the correct
referent for free, is it STILL two-view?* If no → localize-then-name → REJECT.

## Survivors (ranked by acid-test cleanliness × build/label risk)

### ★ FLAGSHIP FAMILY — Cross-view aggregation ("one view enumerates the set, the other measures each element, answer = a statistic over the pair")

The answer is a **statistic** (count-weighted aggregate / ratio / variance / trend). Satellite
owns the **index set** (how many buildings/units front the segment, in what plan order —
invisible at eye level from occlusion/foreshortening). Street-view owns the **per-element
value** (storeys / retail-vs-blank — invisible from above). A statistic over (index set,
values) is **not expressible in either view alone** → structurally fusion, not by distractor luck.

- **#3 Height aggregation (BEST BUILDABLE).** "The satellite shows N buildings fronting this
  segment. Using the ground photos, are they uniform in height or does height vary >2 storeys?"
  Options: A) uniform low ≤3; B) uniform mid/high ≥4; C) varies, stepping up away from dot;
  D) varies, irregular. Label: count OSM polygons in a road-buffer (zero noise), heights from
  `building:levels` prior verified vs along-frames, compute variance/trend along bearing.
  Risk: needs multiple mapped+visible buildings → filter to dense segments.
- **#7 Retail-fraction (same structure, generalizes the family).** "OSM shows K ground-floor
  units on this segment; are MOST active retail at street level, or mostly residential/blank?"
  Ratio = numerator-type (retail, eye-level) / denominator (unit count, plan-only). Risk:
  OSM POI completeness varies by city → noisier; lead with #3, offer #7 to show generality.

### #12 Relative ordering ("which is nearer the dot")
"Two distinctive features appear ahead — {glass tower} and {church spire}. Per the map, which
is physically closer to the red dot?" A) X; B) Y; C) equidistant; D) one is on a different
street. Identity (which-is-which) is eye-level; metric ordering/true distance is plan-view —
binding identity to geometry. Cleanest "fusion = bind SV identity to sat coordinate." Risk:
needs 2 identifiable mapped landmarks in frame → subset; identity↔OSM matching is the noisy part.

### #1 Facade-tall × footprint-deep (2×2)
"Does the building extend deeper into the block than a 1-bay shopfront, AND is the facade >3
storeys?" A) deep+tall; B) deep+short; C) shallow+tall; D) shallow+short. Depth from polygon
(road-normal extent), storeys from along-frames. Orthogonal 2-bit conjunction; given the
building free you still need both bits. Risk: storey-count noise; sparse `building:levels`.

### #6 Setback contradiction
"Overhead geometry implies a set-back open front (driveway/garden). Does SV confirm an open
setback, or built-to-line wall/fence?" A) setback+open; B) setback+walled; C) no setback+
recessed; D) build-to-line both. Setback distance = plan-only; frontage permeability =
eye-level-only. Mine for the rare contradiction classes B/C. Risk: most are A/D (imbalance).

### #10 Corner-store orientation ("which street does the entrance face?")
Coordinate-relabeling: entrance pixel is eye-level; "this frontage = cross vs along street" is
plan+bearing. Answer is the relation (which-street), not the thing (door). Risk: entrance tags
sparse, chamfered-corner ambiguity, some pose risk.

## Rejected (boundary cases — useful to show the law's edge)
- **#9 rooftop reveal, #13 block-interior** — info lives in SATELLITE alone; SV only localizes →
  localize-then-name. REJECT.
- **#11 implied-vs-actual use** — collapses to "always trust SV." Keep only as curated mismatch
  subset, if at all.
- **#5 facade-material × footprint-shape** — borderline conjunction not relation; keep only with
  bit-sharing distractors (2 options share each bit) so both views are forced.
- **#14 shadow-height × storey-count consistency** — scientifically elegant (two independent
  height estimators compared) but needs sun angle/timestamp + shadow segmentation → too noisy.
- **#2 walkability, #4 which-corner-tallest** — great concepts but HIGH correspondence/pose risk
  (map a sat region to the right SV pixel via approximate Street View heading).

## Why the flagship is the most defensible "requires fusion" claim
You can write a **one-line proof**: the answer is a statistic whose numerator/denominator (or
index-set/values) live in *different* views, so no single-view oracle — however perfect at its
own view — can even *express* the statistic, let alone exceed chance on the contrastive classes
("uniform" vs "varies"; "mostly retail" vs "mostly blank"). This defeats the standard skeptic
rebuttal ("the model just learned which view to trust") — there is no view that contains the ratio.
