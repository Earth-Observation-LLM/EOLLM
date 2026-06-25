GROK: **Task 1: Direction to next through-road**  
**Template:** "From the marked point, the map shows four possible routes leaving the intersection. Which direction leads to a true through-road that continues at least one block beyond this junction?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Why satellite-alone fails:** The map only shows the claimed topology; it does not reveal whether any claimed link is actually a stub, ramp, or path that terminates immediately.  
**Why street-view-alone fails:** The four frames show local surfaces but provide no information about network continuation beyond the visible segment.  

**ACID TEST:** Even given the four candidate directions, the model still needs the satellite topology to know which route the map claims continues and the street view to verify physical continuity.  

**Label derivation:** OSM `highway=*` ways + intersection node degrees (≥3) + geometry of the outgoing edges. Compute degree after removing the central node; retain pairs where one outgoing edge has degree ≥2 beyond the node. No forbidden tags.  

**Urbanization value:** Identifies real routing choices vs map artifacts for navigation and emergency access.  

**Constraint check:** Passes 1–4.

**Task 2: Pedestrian-cut vs mapped cul-de-sac**  
**Template:** "Which direction shows a mapped dead-end that actually continues as a narrow pedestrian path or stairs?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Why satellite-alone fails:** OSM marks the link as a cul-de-sac; satellite cannot distinguish between a true barrier and a legal footpath.  
**Why street-view-alone fails:** Without the map degree, the walker cannot know whether the visible path contradicts the network topology.  

**ACID TEST:** The claimed topology (satellite) and the physical path quality (street) must be compared for the answer.  

**Label derivation:** OSM `highway` ways filtered to degree-1 terminal nodes + geometric length threshold.  

**Urbanization value:** Locates unmapped pedestrian shortcuts and severance errors.

**Task 3: Motor-vehicle barrier direction**  
**Template:** "At this mapped four-way junction, in which direction is vehicle passage physically prevented by bollards, a gate, or a continuous curb?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM intersection nodes (degree ≥4) + outgoing edges.  

**Urbanization value:** Detects traffic-calming and filtered permeability.

**Task 4: Park-edge permeability**  
**Template:** "The satellite shows a park polygon adjacent to the point. Which street-view direction contains the only mapped and physically open public entrance within 30 m?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM `leisure=park` polygons + nearest `highway=footway|path|pedestrian` edges touching the polygon boundary.  

**Urbanization value:** Actual versus nominal park access.

**Task 5: Grade-separated crossing**  
**Template:** "The map shows a rail or major road directly beside the point. Which direction allows a pedestrian to cross to the far side without using an overpass or underpass?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM `railway=*` or `highway=primary|motorway` lines + nearest crossing nodes.  

**Urbanization value:** Pedestrian severance measurement.

**Task 6: Mapped vs actual road surface continuity**  
**Template:** "Which outgoing mapped road segment is actually a set of stairs rather than a drivable carriageway?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM `highway` classification vs geometry length.

**Urbanization value:** Routing engine error correction for emergency and delivery vehicles.

**Task 7: Block-internal vs perimeter access**  
**Template:** "The satellite shows a dense block perimeter. Which direction is the only pedestrian route that leads inside the block (alley or passage) rather than along the perimeter?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM closed way building footprints + internal footway segments.  

**Urbanization value:** Micro-accessibility in dense historic cores.

**Task 8: Water-edge public access**  
**Template:** "The satellite shows a river or canal touching the point. Which direction is the only publicly accessible bank (no private fence or wall)?"  
**Options:** along_fwd / along_bwd / cross_left / cross_right  

**Label derivation:** OSM `waterway=river|canal` + adjacent `highway=footway` or `leisure` polygons.  

**Urbanization value:** Blue-space accessibility planning.

**Strongest task:** Task 1 (Direction to next through-road).  

**One-line proof sketch:** Satellite supplies only the claimed node degree after central-node removal; street-view supplies only the local surface geometry and sight-line; neither view alone encodes whether a *claimed* continuing edge remains physically continuous beyond the visible frame, so the four-class answer distribution is independent of either marginal view.

Gemini: Here are 8 new VQA tasks designed under strict adherence to your four constraints. They abandon the "localize-then-name" trap completely, ensuring that $Y = g(Sat, SV)$ demands the cross-product of **topological/geometric extent** (exclusive to the satellite) and **physical/micro-affordance** (exclusive to the street view).

---

### 1. Modal Connectivity & Filtered Permeability  *(Top-Ranked Task)*
**Task Name:** Filtered Connectivity State
**Question Template:** Examine the road segment continuing behind the camera (`along_bwd`). Based on its mapped topology and physical street-level design, what is its actual connectivity profile to the wider network?
**4 Answer Options:**
1. Vehicular & pedestrian through-route (continuous flow for all modes).
2. Filtered permeability: Vehicular dead-end, but acts as a pedestrian/bicycle through-route.
3. Absolute dead-end: Cul-de-sac or gated private enclosure for all modes.
4. Dedicated active transit: Mapped and built exclusively as a pedestrian/cyclist path.
**Why Sat-alone fails:** Satellite (RGB) shows a continuous ribbon of pavement spanning from one block to the next. Sub-pixel physical filters like iron bollards, cycle-gates, or chain-link fences across the road are invisible from space.
**Why SV-alone fails:** The street-view sees bollards or a gate. But because SV lacks spatial depth beyond the occlusion, it cannot tell if the road past the gate genuinely connects to the next neighborhood (a through-route) or if it terminates 30 meters later in a closed garage courtyard (a true dead-end).
**ACID TEST Verdict:** YES. Given the exact segment, Satellite provides the *destination/topology* (connects vs terminates), and SV provides the *barrier state* (bollards vs none)—both are mandatory to classify the modal profile.
**Label Derivation:** Fully automated. OSM `highway=*` tags, topological connectivity of the way to another through-street, and the presence of intersecting `barrier=bollard/cycle_barrier` or `motor_vehicle=no`/`foot=yes`.
**Urbanization Value:** Directly measures the "15-minute city" concept, active mobility networks, and the prevalence of car-centric cul-de-sacs vs pedestrian porosity.
**Constraint Check:** No storeys/POI counts. Requires fusion to formulate a new category. Not a mismatch check.

### 2. Transverse Severance Affordance
**Task Name:** Crossing Typology
**Question Template:** At this exact location, the primary route crosses a major transverse urban feature (such as a river, railway, or major arterial). How is this crossing vertically separated, and what is the nature of the feature beneath/above?
**4 Answer Options:**
1. Overpass / Bridge crossing over a natural water body.
2. Overpass / Bridge crossing over constructed infrastructure (railway or road).
3. Underpass / Tunnel passing beneath infrastructure.
4. At-grade / Level intersection with the infrastructure (no vertical separation).
**Why Sat-alone fails:** Dense tree canopy and orthorectification make an at-grade rail crossing, a short underpass, and a pedestrian overpass structurally identical from above. It severely lacks Z-axis distinction.
**Why SV-alone fails:** When the camera is driving *on* a bridge with concrete guardrails, it clearly detects the Z-axis affordance (we are on an overpass, ruling out 3 & 4). However, the guardrails and perspective occlude the transverse feature underneath. It cannot tell if it is crossing a river or a railway yard.
**ACID TEST Verdict:** YES. Even if perfectly localized to the crossing, SV provides the vertical `layer`, and Sat provides the `transverse landuse/geometry`. 
**Label Derivation:** Automated. Intersecting OSM ways where node contains `layer!=0`, `bridge=yes`, or `tunnel=yes`, combined with the tags of the transverse way (`waterway=*`, `railway=*`, `highway=*`).
**Urbanization Value:** Identifying severe urban barriers, safety of active transit corridors, and the cost/infrastructure density of the built environment.
**Constraint Check:** No POI data. Uses highly reliable road/rail/water/bridge tags.

### 3. Superblock Micro-Permeability
**Task Name:** Block Interior Access
**Question Template:** The satellite footprint reveals a large, continuous block structure. Does the street-view reveal any functional micro-permeability (such as alleys or paths) that mitigates this massing, and where does it lead?
**4 Answer Options:**
1. Yes, a dedicated pedestrian cut-through breaches the block fully to the next street.
2. Yes, a vehicular access lane penetrates the interior but operates as a private courtyard/garage (dead-end).
3. No, the street-facade presents an unbroken physical wall/building line with no public gaps.
4. No, gaps exist between buildings, but they are fully blocked by dense vegetation or security gates at the street edge.
**Why Sat-alone fails:** From above, gaps between building polygons are obvious. However, satellite cannot see if that spatial gap is closed at the street boundary by a 2-meter iron gate or a solid high wall.
**Why SV-alone fails:** SV sees an open dark alleyway. But because it has no aerial map, it cannot fathom the depth of the block. It doesn't know whether this alley miraculously connects to the parallel street 150m away, or if it stops short.
**ACID TEST Verdict:** YES. SV identifies the *street-edge affordance* (open gap vs physical gate); Sat maps the *interior spatial continuity* (through-line vs courtyard).
**Label Derivation:** Geometry intersection of OSM `building` polygons, `highway=footway/service/residential`, and bounding-box topological depth.
**Urbanization Value:** Quantifying urban grain, walkability, block scale, and hidden pedestrian infrastructure. 
**Constraint Check:** Avoids building use/storeys. Uses highly reliable `building` and `highway` polygons.

### 4. Topological Grade Reality 
**Task Name:** Elevation Accommodation
**Question Template:** The map designates a continuous, connected route extending `along_fwd`. How does the physical street-level geometry handle the spatial elevation changes along this route?
**4 Answer Options:**
1. Vehicular Grade: At-grade paved slope seamlessly accommodating vehicles.
2. Pedestrian Steps: The plotted route is constructed as physical urban staircases.
3. Dedicated Ramp/Switchback: Constructed purely as a non-stepped incline/ramp for active transit.
4. Sunken Trench / Retained Wall: The route relies on a deeply excavated trench or sheer retaining wall infrastructure.
**Why Sat-alone fails:** Top-down orthophotos cannot distinguish a 30% grade set of concrete stairs from a steep pedestrian concrete path (`highway=path`), due to shadowing and lack of DEM relief.
**Why SV-alone fails:** SV clearly sees a staircase going up between houses. But it cannot know if these stairs are simply the private front access to a high block of flats, or if they are the actual *mapped public right-of-way* traversing the neighborhood topology.
**ACID TEST Verdict:** YES. SV categorizes the *surface geometry* (Steps vs Flat); Sat provides the *topological classification* (Public network node).
**Label Derivation:** Automated. `highway=steps` vs `highway=pedestrian/path` vs `highway=residential`. Includes evaluation of `cutting=yes` or `barrier=retaining_wall` adjacencies.
**Urbanization Value:** ADA / wheelchair accessibility routing, micro-mobility limitations (the universal "car routed down public stairs" problem).
**Constraint Check:** Highly reliable tagging (Stairs are heavily mapped due to routing importance).

### 5. Arterial Edge Severance
**Task Name:** Major Road Permeability
**Question Template:** This road functions as a high-capacity arterial or primary route. How does its physical edge design dictate local access and integration with the adjacent built environment?
**4 Answer Options:**
1. Fully Integrated: Direct residential or commercial driveways connect straight to the main carriageway.
2. Service Road Buffered: A physical tree line or median geometrically separates the high-speed main lanes from a local parallel access street.
3. Medically/Acoustically Walled: Continuous physical walls or sound barriers sever the arterial visually and physically from adjacent structures.
4. Open Natural Buffer: A wide, unbuilt natural setback exists, but no walls or parallel service roads are present.
**Why Sat-alone fails:** Satellite cannot reliably resolve thin vertical sound walls vs dense linear tree canopies natively. It often confuses a wide parallel multi-use path with a service road.
**Why SV-alone fails:** The SV camera sees a concrete wall to the right. Without the aerial map, it doesn't know it's traversing a neighborhood-bisecting primary highway—it just sees a road with a wall.
**ACID TEST Verdict:** YES. Map class ensures context (Arterial); SV provides the exact ground barrier typology (Noise wall vs open driveway). 
**Label Derivation:** OSM `highway=primary/secondary`. Integrated = building polygons immediately adjacent over `access` points. Service road = parallel `highway=service`. Walled = `barrier=noise_wall/wall/fence` hugging route.
**Urbanization Value:** Assessing community severance, environmental noise mitigation, and highway-vs-neighborhood dynamics.
**Constraint Check:** Combines pure edge distances and `barrier` tags with `highway` classification.

### 6. Public vs. Proximate Amenity Edge
**Task Name:** Natural/Public Space Interface
**Question Template:** The satellite reveals a large-scale natural or recreational area (e.g., park, forest, water body) adjoining this block. What does the street-view reveal about the exact border separating the road from this space?
**4 Answer Options:**
1. Hard Barrier Severance: High fence, solid wall, or severe grade drop physically blocks access.
2. Soft Filtered Space: Easily permeable boundaries (bollards, low curbs, wide footpaths) grant immediate entry.
3. Obscured / Private Frontage: Private buildings or continuous private driveways sit between the road and the natural feature.
4. No Major Amenity Mapped: (Distractor / Negation option for dataset balancing).
**Why Sat-alone fails:** Can easily measure spatial adjacency (water polygon is 5 meters from road centerline). Cannot resolve if those 5 meters contain a sheer 10ft retaining wall or a gently sloping grassy bank. 
**Why SV-alone fails:** SV sees a brick wall, but doesn't know what's behind it. Or SV sees private houses, totally unaware that an enormous urban park sits immediately 20m behind their backyards.
**ACID TEST Verdict:** YES. Sat confirms the *identity and location* of the hidden amenity. SV evaluates the *physical boundary texture*. 
**Label Derivation:** Distance threshold from road to `natural=wood/water` or `leisure=park`. Interface type backed by `barrier=fence/wall` and `building` polygon intersections. (Requires careful geometric setup to ensure clean labeling).
**Urbanization Value:** Measuring "true" access to green/blue spaces vs spatial proximity (a massive difference in urban equity).
**Constraint Check:** Completely relies on standard dense polygons (`natural` / `leisure`). 

### 7. Pedestrian Intersection Geometry 
**Task Name:** Crossing Vulnerability Index
**Question Template:** At this junction, how does the mapped spatial footprint of the intersection interact with the physical protections afforded to pedestrians?
**4 Answer Options:**
1. Compact footprint with strong physical protection (e.g., raised tables, tightened curb radii).
2. Broad, multi-lane footprint with physical median refuges or islands to break up the crossing.
3. Broad, multi-lane footprint with zero physical medians or protections (wide asphalt expanse).
4. Physically severed: Fencing or guardrails completely block pedestrians from crossing the wide intersection at grade.
**Why Sat-alone fails:** From space, it calculates exactly how wide the asphalt is and how many lanes cross. But it cannot discern physical traffic calming (raised tables, micro-islands, pedestrian guardrails that blend into shadows).
**Why SV-alone fails:** Perspective warping! From ground level, a 3-lane wide intersection can look enormous, and judging the true intersection grid footprint (arterial-arterial vs arterial-connector) is highly flawed.
**ACID TEST Verdict:** YES. Sat provides the literal geometric area and lane counts. SV verifies the vertical/physical pedestrian accommodations.
**Label Derivation:** Automated. `lanes>2` or junction bounding area. Intersected with `crossing:island=yes`, `traffic_calming=table`, or `barrier=pedestrian_fence` around junction nodes.
**Urbanization Value:** Vital for pedestrian safety, Vision Zero tracking, and identifying "stroads" vs safe streets.
**Constraint Check:** Uses topology and highway sub-tags, avoiding sparse shop or height data. 

### 8. Transport Shelter Adjacency
**Task Name:** Transit Platform Severance
**Question Template:** A mapped public transit node (bus/tram) exists in the immediate vicinity on one side of this dual-direction route. How must a pedestrian entering from the *opposite* physical side of the corridor access it? 
**4 Answer Options:**
1. Direct at-grade crosswalk across the carriageway.
2. Physical median/island crossing across wide lanes.
3. Complete severing requiring grade-separated bridge/underpass to reach the platform.
4. Unprotected / Informal crossing (no marked geometry across the road).
**Why Sat-alone fails:** Identifies the transit node, but overhead foliage and platform roofs completely obscure if a marked crosswalk, a footbridge, or an underpass links the two sides of the carriage. 
**Why SV-alone fails:** Looking at crossing infrastructure, SV cannot know if it exists simply to cross the road, or specifically to service a major busway/tram platform hidden behind a physical divider or tunnel entrance on the opposite side.
**ACID TEST Verdict:** YES. Sat gives the exact functional target (the transit node). SV dictates the physical navigation method across the barrier.
**Label Derivation:** OSM `highway=bus_stop` or `public_transport=platform` node. Fused with `highway=crossing` (and `crossing=*` sub-tags like `island` or `uncontrolled`) vs `layer` changes.
**Urbanization Value:** Vital for public transit accessibility maps and identifying hazardous first-mile/last-mile transit layouts.
**Constraint Check:** No storeys/roofs. Highly mapped global transport nodes.

---

### RANKING & WINNING TASK PROOF

**Ranking Criteria:** Rank = (Fusion Cleanliness) × (OSM Reliability) × (Urban Value).
1. **Modal Connectivity & Filtered Permeability** (Winner) - *Immaculate fusion, 100% reliable street grid tags, massive planning value.*
2. Transverse Severance Affordance - *Deep topological Z-axis evaluation.*
3. Superblock Micro-Permeability - *Strong test of urban grain/walkability.*
4. Topological Grade Reality - *Extremely fun edge-cases; solves map routing errors.*
5. Pedestrian Intersection Geometry 
6. Arterial Edge Severance 
7. Public vs. Proximate Amenity Edge
8. Transport Shelter Adjacency

#### 🏆 Strongest Task: Modal Connectivity & Filtered Permeability

**One-line argument for fusion:** The Satellite maps a mathematically connected route, but the Street-View reveals physical bollards that subvert the map's default assumption, forcing a joint conclusion that neither view contains individually.

**Mathematical Proof Sketch against Marginal Oracles:**
Let random variable $T$ = Topologcial status (Connected vs Dead-End) — visible ONLY to a perfect Satellite Oracle ($P_{sat}$). Let random variable $B$ = Barrier status (Open vs Bollards) — visible ONLY to a perfect Street-View Oracle ($P_{sv}$). 
The answer class $Y$ maps uniquely to tuples $(T, B)$. E.g., Filtered Permeability = $(Connected, Bollards)$.
1. If $P_{sv}$ sees Bollards, $B$ is known. However, bollards are installed on true dead-ends AND through-routes. Because $P_{sv}$ cannot see past the visual horizon to determine if the road eventually connects to the global grid, $T$ remains conditionally independent of $B$ in $P_{sv}$'s frame. Thus, $P(Y | SV) \le 50\%$.
2. If $P_{sat}$ sees connection, $T$ is known. However, bollards ($B$) are structurally sub-pixel geometry in top-down aerials. Cities implement bollards on seemingly random residential grids. Without $B$, $P(Y | Sat) \le 50\%$.
3. Therefore, both marginal models are mathematically bounded well below ground-truth probability because $Y$ is an exclusive $(AND)$ gate over two physically disjoint spatial dimensions (Macro-scale and Micro-Z-axis).

GPT: ## Verdict

The strongest designs are **not** “find the branch in satellite, name it in street-view.” That fails your acid test. The best tasks make the answer a **routing/access relation**:

> **map/satellite gives the strategic urban structure** — barrier, continuation, block permeability, open-space adjacency, graph destination
> **street-view gives the physically usable affordance** — motorable, pedestrian-only, steps, visible connector direction
> **answer is the joint urban action/category**, not either piece alone.

I would enforce this global construction rule for every task:

**Per-city balancing is mandatory.** For each task, only keep cities where all 4 answer classes can be sampled at roughly 25/25/25/25. If a city cannot support balanced classes, exclude that city for that task. Otherwise the city-prior leak comes back immediately.

Also: no `building:levels`, no POI/shop counts, no materials, no `barrier=*`, no fence/gate labels unless derived through reliable mapped highway connectivity.

---

# Ranked candidate tasks

## 1. **Severance Detour Direction** — strongest

**Question template**

> A major mapped corridor or water edge near the red dot creates a desired pedestrian crossing/continuation direction in the satellite view. Using the street-view frames, what is the first usable movement direction toward that mapped crossing/connection?

**Options**

A. Continue roughly straight toward the mapped crossing/connection
B. Turn left first to reach the usable connector
C. Turn right first to reach the usable connector
D. Turn back/opposite first; the usable connector is behind the point

**Why satellite-alone fails**

Satellite/plan-view can identify the barrier, the opposite side, and the desired crossing direction. But it cannot reliably tell which ground-level movement is actually usable: crosswalk, footpath, steps, underpass entrance, blocked median, kerb, etc. The usable first move is a street-level affordance.

**Why street-view-alone fails**

Street-view can show a pedestrian connector, stairs, path, or crossing, but it does not know the mapped target across the barrier or which crossing/continuation is strategically relevant in the road graph. It sees affordances, not the plan-level destination structure.

**ACID TEST verdict**

Passes if the referent is defined as the **mapped crossing target**, not the whole answer. Given the target for free, the model still needs street-view to know the usable first move; given the visible connector for free, it still needs satellite/OSM geometry to know whether that connector is aligned with the mapped crossing target.

**Label derivation**

Use only:

* OSM `highway` ways and `highway` class
* road graph connectivity/topology
* water polygons
* geometric distance/bearing
* stored road bearing

Derive:

1. Detect eligible barriers/corridors near the red dot: high-class roads such as `primary`, `trunk`, `secondary`, or water polygons.
2. Compute the desired map direction from the red dot to the nearest graph-relevant opposite-side continuation or crossing target.
3. Find the first accessible mapped connector from the red-dot road using `highway=footway`, `pedestrian`, `steps`, `cycleway`, or low-speed road classes.
4. Bucket the angular offset between desired map direction and first usable connector direction into straight / left / right / back.

No vision model or human label is needed.

**Urbanization value**

This directly measures **pedestrian severance and detour cost**: can a person cross or continue where the map suggests, or must they detour left/right/back?

**Constraint confirmation**

Not a mismatch check. It asks for a **movement instruction**. Uses no forbidden tags.

---

## 2. **Continuation Routing Status**

**Question template**

> For the route that is the mapped geometric continuation of the road through the red dot, what is its combined routing status?

**Options**

A. Vehicular through-connection
B. Pedestrian/cycle-only through-connection
C. Vehicular local access or dead-end
D. Pedestrian/cycle-only local access or dead-end

**Why satellite-alone fails**

Satellite can often see whether the road/path continues through the block or terminates, but it cannot reliably distinguish motorable access from bollard-protected pedestrian space, stairs, cycleway, or narrow service access.

**Why street-view-alone fails**

Street-view can see whether the continuation is physically motorable or pedestrian-only, but it cannot reliably know whether it connects through to another mapped street, terminates inside the block, or loops behind buildings.

**ACID TEST verdict**

Passes. Even if the continuation branch is given for free, the answer still requires both: **street-view for physical modality** and **satellite/OSM topology for through-vs-dead-end status**.

**Label derivation**

Use:

* OSM `highway` class
* graph connectivity
* road bearing
* local way geometry

Derive:

1. Snap red dot to nearest OSM road way.
2. Use stored road bearing to identify the forward continuation leg.
3. Classify modality from `highway` class:

   * motorable: `residential`, `living_street`, `service`, `unclassified`, `tertiary+`
   * active/non-motor: `footway`, `pedestrian`, `cycleway`, `steps`, selected `path`
4. Classify topology by graph reachability within a threshold, e.g. connects to another public road within 80–120 m versus terminal/local spur.

No human pass.

**Urbanization value**

Useful for navigation, emergency access, walkability, routing, and detecting where apparent street continuity becomes pedestrian-only or dead-ended.

**Constraint confirmation**

Not consistency/mismatch. No storeys, POIs, material, or amenity counts.

---

## 3. **Lowest-Motor Exit Network Role**

**Question template**

> At this multi-exit location, one visible exit has the lowest motor-vehicle accessibility. What urban network role does that exit play?

**Options**

A. Through-block shortcut to another mapped street
B. Access into a mapped park/open space
C. Connection toward a mapped water edge or major corridor edge
D. Local dead-end/service spur

**Why satellite-alone fails**

Satellite can identify possible graph roles of exits, but it cannot know which exit is physically lowest-motor-access: stairs, pedestrianized passage, bollards, cycle path, narrow footway, etc.

**Why street-view-alone fails**

Street-view can identify the least motorable exit, but not whether that route connects through the block, enters a park, reaches a waterfront edge, or dies inside the block.

**ACID TEST verdict**

Passes if the free referent is only the spatial exit, not its class. The model still needs street-view to determine low-motor affordance and satellite/OSM geometry to determine network role.

**Label derivation**

Use:

* OSM `highway` ways/classes
* graph connectivity
* `leisure=park` / green polygons
* water polygons
* geometric endpoint analysis

Derive:

1. At the red-dot node, enumerate outgoing mapped legs.
2. Rank legs by motor accessibility from `highway` class.
3. Pick the lowest-motor leg.
4. Follow its graph endpoint/destination:

   * reaches another road: through-block shortcut
   * enters green polygon: park/open-space access
   * approaches/crosses water/major corridor edge
   * terminates locally: dead-end/service spur

No forbidden tags.

**Urbanization value**

Measures pedestrian permeability, informal shortcuts, park access, and non-car network structure.

**Constraint confirmation**

Not an agreement check. Uses only road graph, highway classes, green/water polygons.

---

## 4. **Open-Space Access Offset**

**Question template**

> The satellite view shows mapped green/open space adjacent to the red-dot street. Where is the usable mapped access route to that space relative to the nearest open-space edge?

**Options**

A. Directly at the nearest edge
B. Offset to the left along the street
C. Offset to the right along the street
D. Requires backtracking/opposite movement; no nearby direct edge access

**Why satellite-alone fails**

Satellite can show the green polygon and nearest edge, but cannot reliably show whether the edge is publicly usable at ground level or where the actual entry route is.

**Why street-view-alone fails**

Street-view can show a gate/path/opening, but without plan-view geometry it cannot know whether this is the nearest mapped open-space edge or an offset access point.

**ACID TEST verdict**

Passes. Given the green-space polygon for free, street-view is still needed to locate usable access. Given a visible path for free, satellite/OSM is still needed to know its relation to the mapped open-space edge.

**Label derivation**

Use:

* `leisure=park`, green/open polygons
* OSM `highway=footway/pedestrian/path/steps/service/residential`
* road graph connectivity
* bearing from red dot to polygon boundary and to access connector

Derive:

1. Select red-dot locations within threshold of a mapped green/open polygon.
2. Compute nearest boundary point direction.
3. Find nearest mapped route crossing into or running into the polygon.
4. Bucket route direction relative to nearest-boundary direction.

No POI or amenity counts.

**Urbanization value**

Measures whether green space is actually accessible from the street or merely adjacent on the map.

**Constraint confirmation**

Not a consistency task. No forbidden tags. Avoid using `gate`, `fence`, or `barrier`.

---

## 5. **Waterfront Access Outcome**

**Question template**

> The satellite view shows a mapped water edge near the red dot. What kind of route connects the street point toward or across that water edge?

**Options**

A. Vehicular road access or bridge-like road connection
B. Pedestrian/cycle promenade or footpath access
C. Steps/stair access toward the water edge
D. No mapped public route connection within the local threshold

**Why satellite-alone fails**

Satellite can identify water proximity and route geometry, but not reliably whether the route is motorable, pedestrian-only, or stairs.

**Why street-view-alone fails**

Street-view can show a promenade, stairs, road, or path, but it cannot know whether that route actually reaches/crosses the mapped water edge beyond the visible scene.

**ACID TEST verdict**

Passes. The answer combines **water-edge graph relation** with **ground-level route affordance**.

**Label derivation**

Use:

* water polygons
* OSM `highway` ways/classes
* graph distance to water boundary
* geometry/bearing

Derive:

1. Select points near mapped water polygons.
2. Find mapped route from red-dot road toward/crossing water boundary.
3. Classify first route segment by `highway` class:

   * motorable road
   * footway/pedestrian/cycleway/path
   * steps
   * no graph connection within threshold

No visual or human labeling.

**Urbanization value**

Useful for waterfront accessibility, promenade networks, flood-edge access, and pedestrian routing.

**Constraint confirmation**

Not mismatch/agreement. No forbidden tags.

---

## 6. **Block-Permeability Passage Function**

**Question template**

> A physically usable passage or narrow connector is present near the red dot. What map-level permeability function does that connector provide?

**Options**

A. Through-block connection to another mapped street
B. Access into an internal courtyard/building-cluster space
C. Access into mapped green/open space
D. Local terminal/service access only

**Why satellite-alone fails**

Satellite can see block structure and possible passages, but cannot reliably know whether the connector is physically usable at street level or what type of access it affords.

**Why street-view-alone fails**

Street-view can show an alley, service passage, pedestrian way, or narrow entrance, but cannot see its full block-scale graph outcome behind the facade.

**ACID TEST verdict**

Passes. Given the passage geometry, the answer still requires street-view for physical usability/class and satellite/OSM for block-scale destination.

**Label derivation**

Use:

* OSM `highway` ways/classes
* building footprint polygons
* road graph connectivity
* green polygons

Derive:

1. Detect small connector ways near the red dot.
2. Use `highway` class to classify physical route type.
3. Follow the graph:

   * reaches another street: through-block
   * terminates inside building-footprint enclosure: courtyard/building-cluster access
   * enters green polygon: open-space access
   * dead-end/service: terminal access

No building use, no POIs, no material.

**Urbanization value**

Captures fine-grained pedestrian permeability and block porosity.

**Constraint confirmation**

Not a consistency check. Uses only allowed OSM layers.

---

## 7. **Road-Hierarchy Leg With Lowest Motor Access**

**Question template**

> At this junction, which mapped road-network role is occupied by the leg with the lowest ground-level motor-vehicle access?

**Options**

A. The highest-hierarchy road leg
B. The same-road continuation leg
C. A lower-hierarchy side-street leg
D. A local spur/dead-end leg

**Why satellite-alone fails**

Satellite can infer road hierarchy and junction geometry, but not reliably whether a leg is restricted by pedestrianization, steps, bollards, or non-motor design.

**Why street-view-alone fails**

Street-view can show the lowest-motor leg, but cannot reliably know whether it is the main hierarchy leg, a side street, or a graph dead-end.

**ACID TEST verdict**

Passes. The answer is not “which direction is blocked?”; it is the **network role** of the physically lowest-motor leg.

**Label derivation**

Use:

* OSM `highway` class
* graph degree
* road bearing
* continuation angle
* dead-end detection

Derive:

1. Enumerate junction legs.
2. Classify each leg’s mapped hierarchy from `highway`.
3. Classify motor accessibility from `highway` class.
4. Pick the lowest-motor leg and assign its network role.

No sparse tags.

**Urbanization value**

Good for detecting pedestrianized cores, service-only streets, and hierarchy/access conflicts.

**Constraint confirmation**

Not mismatch. No forbidden tags.

---

## 8. **Transit Approach Connector Type**

**Question template**

> The map contains a transit stop or station access near the red dot. What type of street-level route connects this point toward that transit access?

**Options**

A. Direct road/sidewalk-edge approach
B. Pedestrian-only connector
C. Steps/stair connector
D. Indirect/backtracking approach through another leg

**Why satellite-alone fails**

Satellite/OSM can locate the transit node and shortest graph path, but cannot reliably identify the ground-level affordance of the first approach segment.

**Why street-view-alone fails**

Street-view may show a sidewalk, stairs, or path, but without the mapped transit node it does not know which visible route is the transit approach.

**ACID TEST verdict**

Passes. The transit target is map-defined; the usable approach affordance is street-defined.

**Label derivation**

Use:

* transit nodes
* OSM road graph
* `highway` class of shortest path first segment
* bearing/geometry

Derive:

1. Select locations within graph-distance threshold of transit nodes.
2. Compute shortest mapped walking/road path from red-dot road to transit node.
3. Classify first segment as road-edge, pedestrian-only, steps, or indirect/backtracking based on bearing and `highway` class.

No amenity counts. Transit nodes are allowed by your spec.

**Urbanization value**

Measures first/last-mile accessibility.

**Constraint confirmation**

Not consistency. No POI completeness or amenity counting.

---

## 9. **Land-Use Boundary Crossing Mechanism**

**Question template**

> The red dot lies near a mapped boundary between two urban land-use areas. What physical route mechanism crosses or mediates that boundary?

**Options**

A. Vehicular street crossing the boundary
B. Pedestrian/cycle path crossing the boundary
C. Steps/stair route crossing the boundary
D. No mapped public route crossing nearby

**Why satellite-alone fails**

Satellite/OSM can identify the boundary and crossing geometry, but cannot reliably tell whether the crossing route is motorable, pedestrian-only, or steps.

**Why street-view-alone fails**

Street-view can show a road/path/stairs, but cannot know that this route crosses a mapped land-use boundary or what urban fabric lies beyond.

**ACID TEST verdict**

Passes. The answer combines land-use boundary geometry with ground-level route affordance.

**Label derivation**

Use:

* OSM `landuse` polygons
* OSM `highway` ways/classes
* geometry intersection between ways and land-use boundaries
* road bearing

Derive:

1. Select red dots close to a land-use polygon boundary.
2. Identify mapped way crossing the boundary near the dot.
3. Classify crossing way by `highway` class.

No POIs, no building use.

**Urbanization value**

Captures permeability between urban fabrics: residential-to-commercial, built-up-to-open, industrial-to-street network, etc.

**Constraint confirmation**

Not mismatch. Uses allowed land-use and highway layers only.

---

## 10. **Active-Mobility Route Downstream Outcome**

**Question template**

> A visible active-mobility route leaves the red-dot street. What does the map say happens to that route after its next mapped connection?

**Options**

A. It continues as a through active-mobility route
B. It merges into a motorable street
C. It enters mapped green/open space
D. It terminates or reaches a local dead-end

**Why satellite-alone fails**

Satellite can follow route topology, but cannot reliably know which ground-level route is the active-mobility one when multiple paths/driveways/alleys exist.

**Why street-view-alone fails**

Street-view can identify the active-mobility route, but not what happens after the next mapped connection beyond the visible scene.

**ACID TEST verdict**

Passes. Even with the route geometry, the answer needs ground-level active-mobility affordance and map-level downstream topology.

**Label derivation**

Use:

* OSM `highway=footway/cycleway/pedestrian/steps/path`
* road graph connectivity
* green polygons
* highway class of downstream segment

Derive:

1. Select an active-mobility way near the red dot.
2. Follow it to the next graph node or connection.
3. Classify downstream outcome: active continuation, motor-road merge, green-space entry, terminal/dead-end.

No human pass.

**Urbanization value**

Useful for walkability/cyclability continuity and micromobility routing.

**Constraint confirmation**

Not consistency. No forbidden tags.

---

# Final ranking

| Rank | Task                                        | Fusion cleanliness | OSM reliability | Urban value | Verdict                                          |
| ---: | ------------------------------------------- | ------------------ | --------------- | ----------- | ------------------------------------------------ |
|    1 | Severance Detour Direction                  | Very high          | High            | Very high   | Best overall                                     |
|    2 | Continuation Routing Status                 | High               | Very high       | Very high   | Safest OSM task                                  |
|    3 | Lowest-Motor Exit Network Role              | High               | High            | High        | Strong fusion if multi-exit filtered             |
|    4 | Block-Permeability Passage Function         | High               | High            | High        | Very useful, needs clean graph filters           |
|    5 | Road-Hierarchy Leg With Lowest Motor Access | High               | Very high       | Medium-high | Good, but hierarchy can leak via satellite width |
|    6 | Active-Mobility Route Downstream Outcome    | Medium-high        | High            | High        | Good for walkability                             |
|    7 | Open-Space Access Offset                    | Medium-high        | Medium-high     | High        | Good, but park-path OSM completeness varies      |
|    8 | Land-Use Boundary Crossing Mechanism        | Medium             | Medium-high     | Medium-high | Useful but land-use polygons uneven              |
|    9 | Waterfront Access Outcome                   | Medium-high        | Medium          | High        | Strong where water exists, city-prior risk       |
|   10 | Transit Approach Connector Type             | Medium             | Medium          | High        | Useful, but transit OSM completeness varies      |

---

# Single strongest task

## **Severance Detour Direction**

One-line argument:

> The satellite/OSM view supplies the **desired crossing target across a barrier**, the street-view supplies the **first physically usable movement**, and the answer is their **relative detour direction**; neither view contains that relation alone.

## Bonus proof sketch

Let:

* `D` = map-derived desired direction toward the crossing/continuation target
* `C` = street-derived first usable connector direction
* `Y = bucket(C − D)` into `{straight, left, right, back}`

With per-city balancing such that:

> `P(Y = straight | city) = P(Y = left | city) = P(Y = right | city) = P(Y = back | city) = 0.25`

and additionally:

> `Y` is balanced conditional on `D` and conditional on `C`

then:

* a satellite-only oracle may know `D`, but not `C`, so `Y` remains uniformly distributed;
* a street-view-only oracle may know `C`, but not `D`, so `Y` remains uniformly distributed;
* only the joint observation of `D` and `C` determines `Y`.

That is the cleanest information-theoretic fusion structure among the candidates.

qwen: Below are 11 candidate fusion tasks, ranked from strongest to weakest. Each is written with the exact 4-option MCQ framing, a blunt acid-test verdict, and a defensible OSM derivation path. After the list, I give the ranked summary and the **single strongest pick** with the proof-sketch you asked for.

---

## TASK 1 — `larger_green_entrance_direction` ⭐ TOP

**Question template:** *"The plan-view map shows two green polygons near the red dot — a smaller one (~0–0.5 ha, typically a pocket garden) and a larger one (>1 ha, municipal park or square). In which street-view direction is the PHYSICALLY USABLE ENTRANCE to the LARGER polygon (archway, gate opening, ramp, or path-through-boundary) — as opposed to a fenced or back-face edge?"*

**Options:** `{along_fwd · along_bwd · cross_left · cross_right}`

**Why satellite-alone fails:** The map encodes polygon areas precisely and thus which of the two greens is "the larger one"; it may also hint at footpaths approaching them. But `leisure=park` polygons almost never carry an `entrance=*` node that is correct and dense enough to recover *which side* is the public-facing edge, especially for older parks whose footpath network inside the boundary doesn't align with the public-access perimeter. Without the street image, the satellite can tell you "large green is to the NW" but cannot tell you whether its NW side is the gate, the back of a pavilion, a fence, or a river wall.

**Why street-view-alone fails:** The four frames typically show only the **edge you are standing at** of ONE green polygon — a fence, a gate, a hedge, or an arch. The model has no way to know whether that green polygon is the big municipal park or the little triangle-garden across the road. Two adjacent greens of different OSM areas can look identical at their street-facing edge (both might be walled, both might show a gate). The street-view cannot recover `area > 1 ha` without a plan-view measurement.

**ACID TEST verdict:** Even if I gifted the model "the large green is polygon P at bearing 310°," it *still* cannot answer which of the four street-view edges that polygon presents at the dot's position — because a wall in `cross_left` might belong to the large green or the small adjacent one, and only the map's polygon extent disambiguates *which frame-edge belongs to which polygon*. Genuine synergy, not localize-then-name.

**Label derivation (no forbidden tags):**
- `leisure IN (park, garden, playground)` and `landuse IN (recreation_ground)` polygons from OSM.
- Compute polygon area in m² via shapely on OSM geometry → label "small/large" by threshold.
- Keep only locations where the dot falls within ≈50 m of **exactly two** such polygons with clearly different areas (ratio ≥ 3:1) — this gives you ~2–5 % of your 7,900 locs per city. Filter to ensure balance: within each city, the "larger green's entrance direction" should land roughly equally on each of the four frames. If a city shows strong directional bias (e.g., European river parks always approached from one side), stratify on dot placement.
- Ground-truth label for the 4-option answer: the **direction whose frame shows a physical entrance onto the polygon selected as "larger"**. This step does require a *visual* labelling pass (or a strong VLM judge) — flag it explicitly as "label = VLM-verified entrance on the *larger* polygon's edge; area-filtered from OSM." No forbidden tags are used for filtering; vision is used only for the entrance/no-entrance binary on a pre-filtered candidate.

**Urbanization value:** This is the *public-realm access* question every walkability audit asks: "where can a resident actually enter the usable open space their city claims to provide?" Mismatches between mapped green-area statistics and real physical access are a persistent equity problem — deprived neighbourhoods have "green" on the map but no entrance from the street.

**Constraint 4 check:** Not a consistency / mismatch task. Answer is a direction.
**Constraint 2 check:** Uses only polygon identity + area (dense) — no `building:levels`, no POIs, no `entrance=*` node reliance.

---

## TASK 2 — `two_barriers_which_one_is_crossed` ⭐

**Question template:** *"The map marks TWO distinct linear barriers crossing this area — a `railway=*` line AND a `waterway=*` (canal / river / stream). In the street-view, which of these two physical features does the visible crossing structure actually CROSS (allowing pedestrian or vehicular passage over it)?"*

**Options:** `{railway · waterway · both_have_a_crossing · neither_has_a_crossing}`

**Why satellite-alone fails:** OSM tags crossings (`railway=crossing`, `bridge=yes`) only sparsely and inconsistently. Even dense-OSM cities have thousands of undocumented grade crossings. The map cannot, from tags alone, reliably say whether a *physical* pedestrian crossing exists at *this specific dot's position* on the railway, versus 200 m away.

**Why street-view-alone fails:** The four frames can show a bridge, a footbridge, or a level crossing in one or more directions. However, a narrow canal and a single-track railway look **visually similar** under a bridge (parallel, linear, ~3–6 m wide, similar abutments). The street-view cannot tell whether the linear feature under the bridge you see is the mapped railway or the mapped canal without the map's semantic identity.

**ACID TEST verdict:** With perfect localisation ("a crossing is in frame `cross_left`"), the street-view alone still cannot disambiguate railway vs. waterway because the two barriers run parallel in many legacy industrial corridors and look near-identical from a pedestrian's eye level. Requires satellite identity to answer.

**Label derivation:**
- OSM: `railway IN (rail, light_rail, tram)` + `waterway IN (canal, river, stream)` ways.
- Keep locs whose dot is within a configurable radius of BOTH a railway way and a waterway way, ideally where the two cross or run parallel within 150 m.
- Answer derived by VLM judge looking at visible crossing structure and matching it against the map-identity of which linear feature is being crossed. OSM does the **filtering**; vision does the **labelling**.
- Balance within city: pick cities with both rail and canal legacy (Amsterdam, Manchester, Venice, Hamburg, Boston, Osaka).

**Urbanization value:** Railway severance is a canonical urban-equity issue ("wrong side of the tracks"). Whether a canal vs. a railway is the *actual* crossingable barrier at a neighbourhood's edge tells planners which infrastructure is acting as a severance and which has a gateway.

**Constraint 4:** Produces a feature-identity label (railway / waterway / both / neither), not an agree/disagree.
**Constraint 2:** Uses only the dense `railway=` and `waterway=` way layers.

---

## TASK 3 — `junction_open_exit_count`

**Question template:** *"The map's road topology shows N exit-ways meeting this junction at the red dot. Looking at all four street-view frames, what fraction of those map-claimed exits are PHYSICALLY open to vehicular traffic here (no fixed bollards, no gate, no staircase substitution, no dead-end wall)?"*

**Options:** `{all · most (≥ half but not all) · few (< half) · none / all blocked}`

**Why satellite-alone fails:** Topology tells you *how many roads are claimed*. OSM sporadically tags `barrier=bollard` / `gate` / `access=private`, but the tag density is highly uneven; the *actual state* of a bollard (permanent vs. removable, raised vs. lowered) requires a visual look.

**Why street-view-alone fails:** The model sees the visible scene in each frame, but it doesn't know the *intended* count the planner mapped. A visible path might not be a "map-claimed road" at all (could be a driveway, informal footpath, alley), and a genuine road might be barely visible because of occlusion or is accessed via a narrow gap the model might ignore. The answer is *count / N* — both numerator and denominator require cross-view reasoning.

**ACID TEST verdict:** With a free gift of "there are N exits", the street-view must assess each one's physical state *and bind it to the specific map-exit*. A visible path at 45° from road-bearing forward might correspond to a tagged service road or to no way at all; only the map tells it. Yes, genuine two-view.

**Label derivation:**
- OSM junction extraction: count distinct `highway=*` ways incident to the node at the dot (or within a 10 m buffer if the dot is on-segment).
- For each exit way, compute bearing from dot → match to one of four frames by angular proximity to (fwd, bwd, left, right).
- For each matched frame, VLM labels `{open vehicular · blocked / stairs / gate / wall · not visually identifiable}`.
- Final option-bucketing from fraction open.

**Urbanization value:** Classic "paper street" and "filtered-permeability" audit. The answer tells a walking-network planner how much of the declared junction network is *actually usable*, which is the core metric of real (versus theoretical) connectivity.

**Constraint 4:** Output is a *count class*, not an agree/disagree.
**Constraint 2:** Only dense `highway=*` topology.

---

## TASK 4 — `route_class_calming_type`

**Question template:** *"The plan-view OSM tags the road along the current road-bearing direction as `highway=[primary / secondary / tertiary]`. What physical TRAFFIC-CALMING or access-control feature is visible in the street view along that classified route?"*

**Options:** `{no_calming / free-flow · traffic_calming_only (humps, tables, chicanes, chokers) · hard_access_control (bollards, planters, gates, ANPR cameras) · full_pedestrianisation / shared_surface (no road markings at all)}`

**Why satellite-alone fails:** The satellite image is top-down — bollards, humps, chicanes and shared-surface markings are all sub-resolution or invisible from above. OSM `traffic_calming=*` is one of the most patchy tags known. The map has the class but not the calming.

**Why street-view-alone fails:** The street-view clearly shows humps, bollards or a shared surface — but the *same visual scene* corresponds to many possible map-classes (a bollarded residential road and a bollarded "primary" low-traffic-neighbourhood boundary look almost identical visually). Without the satellite telling you the *declared class*, the urban-meaning of the calming is undefined, and the MCQ answer bucket (`hard_access_control` vs `full_pedestrianisation`) is anchored to the *classified-as* route.

**ACID TEST verdict:** If I gift the model "this is the highway=primary along bearing X," the street-view is **still** required to classify calming, and the *answer bucket itself* is keyed to the satellite class (you cannot assign "primary-but-hard-blocked" without both). Yes, two-view: the label is the *interaction* of declared role and realised physical state — the central audit item in LTN, School Street, and Superblock evaluations.

**Label derivation:**
- OSM: `highway=primary | secondary | tertiary` way at the dot. Road bearing gives `along_fwd`.
- VLM labels the calming in that frame → categorical bin.
- Within each city, balance across the four bins (most cities will skew toward "no calming" — restrict to dense urban cores, and optionally to locations inside a known LTN/congestion-zone envelope to lift the interesting bins).

**Urbanization value:** This is exactly the audit question for Low-Traffic Neighbourhoods, Barcelona Superblocks, Paris "écoles-rues," and London School Streets: *"This is a nominally primary route on the map — has the street actually been physically de-prioritised, and how?"* Planners live and die by this metric.

**Constraint 2:** Only `highway=*` classification.
**Constraint 4:** Not a mismatch; answer is the *physical state category*.

---

## TASK 5 — `parallel_service_lane_provision`

**Question template:** *"The plan-view map shows this location sits between a MAJOR carriageway and a PARALLEL `highway=` side-road (service/frontage road) — the service way is on one lateral side of the dot. Looking at the cross-view frame that faces that service-way side, what is its ACTUAL physical provision?"*

**Options:** `{dedicated_cycleway (painted or protected) · bus-only_lane · vehicular_service_road (cars allowed) · pedestrian-only_footway}`

**Why satellite-alone fails:** Top-down cannot distinguish cycleway paint vs. bus-lane paint vs. car-permitted service road vs. pedestrianised frontage — lane widths and surface markings are visually near-indistinguishable from above except in the newest imagery.

**Why street-view-alone fails:** The `cross_left` or `cross_right` frame clearly shows cycle markings, bus lane or cars — but it cannot tell the model *which side is the parallel service-way the map identified* and which side is just a random slip-road or driveway. Without satellite to tell the model "look at cross_right specifically," the model has no way to pick the correct frame.

**ACID TEST verdict:** The satellite identifies **which of the two lateral directions** is the mapped service-way (based on the highway way that runs parallel to the main carriageway within, say, 30 m). Only after that binding does the correct cross-frame become the target. Street-view then classifies. Neither view produces both halves. Genuine fusion.

**Label derivation:**
- OSM: find two `highway=*` ways whose bearing vectors are nearly parallel (Δθ < 15°) and separated laterally by < 50 m.
- Tag one `main` (by higher highway class / more lanes) and one `service` (the other).
- Determine service-way side relative to the dot using the signed perpendicular offset → `cross_left` or `cross_right`.
- VLM labels the chosen cross-frame.

**Urbanization value:** "Parallel-provision" audits (segregated cycleway vs. bus lane vs. car access on a frontage road) are core input for corridor re-design studies.

**Constraint 2:** Only dense `highway=*` geometry + classification.
**Constraint 4:** Output is provision-type, not a match/mismatch.

---

## TASK 6 — `misclassified_route_is_staircase_direction`

**Question template:** *"The map classifies three or more exit-ways from this junction as `highway=[residential / service / path]`. One of them is physically a STAIRCASE or a stepped pedestrian path in the street view. In which direction is this route that the map treats as non-staircase but that the street shows as steps?"*

**Options:** `{along_fwd · along_bwd · cross_left · cross_right}`

**Why satellite-alone fails:** OSM does contain `highway=steps`, but it is *massively under-tagged* — many urban stairs are mislabelled as `footway`, `path`, or even `residential` at the street-segment scale. From the satellite image, a covered or narrow stairwell between buildings is indistinguishable from a service laneway.

**Why street-view-alone fails:** The model sees stairs in one frame — but to answer this specific question, it must know whether the *map* classifies that feature as stairs. If the answer is just "where are stairs," it's street-view-solvable. The question is **"where is it that the street shows stairs AND the map does NOT call them stairs, among the N candidate routes?"** That's a relation between map-claimed-class and street-observed-reality, filtered on divergence.

**ACID TEST verdict:** Yes: a staircase in `cross_left` is not the right answer if the map already calls it `highway=steps` — we're asking for the *misclassified* one among the candidates. Satellite tells us which routes are map-claimed-as-non-stairs; street-view tells us which ones are physically stairs; the answer is the direction where both conditions hold. Two-view by construction.

**Label derivation:**
- OSM: ways meeting at the node at the dot, with `highway IN (residential, service, footway, path, unclassified, living_street)` — *deliberately excluding* `steps` from the candidate pool.
- Match each incident way's bearing to one of the four frames.
- VLM label: "is this frame's egress a staircase / stepped path?"
- Answer = the frame direction for which VLM="staircase."
- **Balance caveat:** Misclassifications are rare per city. Collect only from cities with hilly terrain (Lisbon, Genova, SF, Chongqing, Valparaíso, Porto) + a VLM-pre-label pass. You may end up with ~300–600 qualified locs globally — flag this if you need a fixed sample size.

**Urbanization value:** *The* benchmark question for routing-engine audits — the Google-Maps-routes-me-onto-a-staircase bug is famous and disproportionately impacts wheelchair users, pram-pushers, and delivery workers. A model that can detect this divergence is directly usable for accessibility-layer correction.

**Constraint 2:** Only dense `highway=*` class and topology — no storey counts or amenity tags.
**Constraint 4:** Output is a direction (of divergence), not "do they match?".

---

## TASK 7 — `cycleway_continuity_direction`

**Question template:** *"The plan-view OSM tags the road along the current `road_bearing` as `highway=cycleway`. In which of the four street-view directions does the PHYSICAL cycleway PROVISION (dedicated lane, painted bike lane, or segregated track) visibly CONTINUE, vs terminate, narrow away or revert to mixed traffic?"*

**Options:** `{continues in both along_fwd and along_bwd · continues only in along_fwd · continues only in along_bwd · terminates in both directions (discontinuous / signed-only)}`

**Why satellite-alone fails:** From top-down, painted bike-lane markings are usually invisible, and segregated tracks at this scale blur with the road shoulder. OSM `cycleway=*` is a known under-tagged attribute.

**Why street-view-alone fails:** The visible markings in each direction say what's there, but without the map telling you *which way is the tagged cycleway direction* (along_fwd per the stored road_bearing), you cannot know which frame is the "continuation" direction and which is the cross-street. A painted bike-lane in `cross_left` is not a continuation.

**ACID TEST verdict:** Yes. Satellite provides *canonical cycleway direction*; street-view provides *physical presence per frame*. The binding ("continues in fwd, terminates in bwd") is genuinely joint.

**Label derivation:**
- OSM: `highway=cycleway` or ways with `cycleway=lane/track/opposite` along the dot segment.
- `along_fwd` is given from stored `road_bearing`; `along_bwd` is the opposite.
- VLM labels each of the four frames: `{visible_cycle_provision · none}`.
- Answer = joint state across `along_fwd` + `along_bwd`.

**Urbanization value:** Cycle-network *continuity* is the single biggest predictor of bike mode-share. The question "is this cycleway a real continuous corridor or just mapped as one" is exactly what every bike-ped master-plan measures.

**Constraints 2 & 4 clean.**

---

## TASK 8 — `bus_stop_pair_shelter_direction`

**Question template:** *"The OSM map marks TWO `highway=bus_stop` nodes adjacent along this road — one in the `along_fwd` direction of the road-bearing and one in the `along_bwd` direction. Looking at those two frames, which direction shows a PHYSICAL SHELTER (roofed structure), as opposed to a simple flag / pole only?"*

**Options:** `{along_fwd_sheltered · along_bwd_sheltered · both_sheltered · neither_sheltered (flag_only or absent)}`

**Why satellite-alone fails:** OSM has a `shelter=yes` tag on `highway=bus_stop`, but its completeness is *notoriously* patchy city-to-city (often < 40 % even in well-mapped areas), and satellite imagery cannot confirm presence of a shelter structure (small roof, ~2 × 3 m, low contrast).

**Why street-view-alone fails:** The model can see shelters in `along_fwd` and/or `along_bwd`. But if only *one* stop is mapped, the other pole might be an ad-hoc flag, informal stop, or decommissioned stop — the model doesn't know. Also: if only one stop is mapped, the `both`/`neither` options become meaningless, which is why you *need* the satellite to guarantee the two-way pair structure to make the 4-way answer space coherent.

**ACID TEST verdict:** Yes. Satellite guarantees the two-node structure; street-view observes shelter-status per node; the 4-option answer space requires both halves. Without satellite the question collapses to "what do I see forward?", which is just street-view.

**Label derivation:**
- OSM: find pairs of `highway=bus_stop` nodes within ≈ 60 m of each other, both lying approximately on the dot's tagged way.
- Determine which is approximately `along_fwd` and which `along_bwd` using geometry.
- VLM label per frame: `{shelter · flag_only · not_installed / invisible}`.
- Answer = joint 4-way state.

**Urbanization value:** Transit-equity micro-audit: *"Do both directions of this bus stop have weather protection, or only the peak-direction one?"* This is a standard KPI for transit agencies and is almost never auditable remotely without ground photography.

**Constraints 2 & 4 clean.** (Using `highway=bus_stop` is allowed; deliberately NOT requiring `shelter=*` OSM tag — that's the point: OSM shelter is sparse, so we go to vision for the answer and use OSM only for the candidate selection.)

---

## TASK 9 — `waterway_crossing_kind`

**Question template:** *"The map shows the route at the red dot crosses a `waterway=` feature (canal, stream, small river). From the street view, what TYPE of crossing is physically present?"*

**Options:** `{vehicular_bridge · pedestrian_footbridge · ford_or_shallow_drive-through · no_crossing (culvert or invisible at surface)}`

**Why satellite-alone fails:** The OSM way crosses the waterway polygon → the bridge exists. But OSM tags `bridge=yes` only sporadically, and cannot distinguish vehicular vs footbridge width from the tag. Top-down image often can't distinguish footbridge width from a narrow vehicle bridge either.

**Why street-view-alone fails:** A bridge is visible in the frame, but without the satellite the model has no certainty this is a *waterway* crossing — urban cuttings, sunken roads, and railway underbridges look visually similar at eye level. The map confirms the presence of water underneath.

**ACID TEST verdict:** Passes, though less strongly than tasks above. Giving the referent for free ("this is the water-crossing") lets street-view mostly answer; the satellite's contribution is mostly to confirm the water underpinning and anchor the answer in the right frame. Weakest of the "strong" tasks.

**Label derivation:**
- OSM: intersection of a `highway=*` way with a `waterway=*` way or a water-polygon boundary.
- VLM labels the visible crossing type in the matched frame.

**Urbanization value:** Flood-resilience and mobility audits care whether a waterway is crossed at-grade (ford — flood-vulnerable) or by a bridge (resilient). The distinction matters most in the Global South, where fords remain common.

**Constraints 2 & 4 clean.**

---

## TASK 10 — `landuse_boundary_direction_of_change`

**Question template:** *"According to OSM `landuse=` polygons, the red dot sits at the BOUNDARY between a `residential`-zoned polygon and a `commercial/retail/industrial` one. Along the road (along_fwd / along_bwd), in which direction does the visible built character actually TRANSITION (storefronts appear, warehouse frontage, or the residential row ends)?"*

**Options:** `{along_fwd · along_bwd · both (symmetric) · neither (transition is lateral, off-road, or invisible)}`

**Why satellite-alone fails:** The plan-view shows the coloured polygons touching, but cannot tell you which way along the road the *perceptual boundary* is experienced. A linear commercial strip can sit entirely within what OSM calls "residential" if the boundary was drawn along a rear lot line.

**Why street-view-alone fails:** The frames show built character per direction but do not know where the *map's* boundary lies (no painted line exists in reality). Whether a transition is happening relative to the OSM-declared boundary needs the satellite's declaration.

**ACID TEST verdict:** Weak-to-moderate. Street-view can often identify "along_fwd looks commercial, along_bwd looks residential" and that strongly correlates with OSM's landuse — real correlation trap here (city prior + OSM-draws-boundary-where-character-changes). Still, OSM landuse boundaries are sometimes drawn on political rather than morphological lines, which breaks correlation in ~10–20 % of cases.

**Label derivation:**
- OSM: dot is within ≈20 m of a boundary between `landuse=residential` and `landuse IN (retail, commercial, industrial)`.
- VLM: classify `along_fwd` and `along_bwd` frames into `{commercial_frontage · residential · other}`.
- Boundary direction = direction in which character visibly transitions.

**Urbanization value:** "Perceptual boundary vs. zoned boundary" is the core audit for walkable-district branding and Business Improvement Districts.

**Constraints 2 & 4 clean.**

---

## TASK 11 — `deadend_map_vs_street_barrier`

**Question template:** *"The map shows a `highway=[service | residential]` segment terminating at the red dot (a tagged cul-de-sac with a turning-head polygon, or no continuation). Physically, in the terminus-facing frame, what is the ACTUAL end-point?"*

**Options:** `{truly_open_terminus_with_turning_circle · gated_or_locked · bollarded (filterable_perm) · stairway_or_ramp_continuation_not_tagged}`

**Why satellite-alone fails:** `barrier=gate` / `barrier=bollard` tags are sparse; a cul-de-sac in OSM can be gated, open, bollarded, or secretly continue as steps, and OSM alone rarely tells which.

**Why street-view-alone fails:** The frame shows a barrier, but without the satellite the model doesn't know this is a *claimed* dead-end (maybe it's just a driveway, maybe an alley through to the next street). The 4-option answer space is meaningful only when anchored to the OSM dead-end declaration.

**ACID TEST verdict:** Moderate. If I give you for free "this is the dead-end of the named cul-de-sac in direction D," street-view can answer. So this is closer to localize-then-name than tasks 1–2. Include with caution; likely the weakest of the set.

**Label derivation:**
- OSM: a way ending at a node at the dot (or with a `turning_circle`), with no continuation beyond — a topological dead-end. `barrier=*` may or may not be tagged.
- VLM labels the visible end-state.

**Urbanization value:** Security gating / filtered-permeability auditing — gated cul-de-sacs are a major access/equity topic.

**Constraints 2 & 4 clean.**

---

# RANKING (fusion-cleanliness × OSM-reliability × urbanization value)

| Rank | Task | Fusion | OSM density | Urban value | Notes |
|---|---|---|---|---|---|
| **1** | **1. `larger_green_entrance_direction`** | ★★★★★ | park polygons very dense | walkability equity | cleanest synergy; answer is a direction that only emerges from cross-view binding |
| **2** | **2. `two_barriers_which_one_is_crossed`** | ★★★★★ | railway + waterway dense in legacy cities | severance equity | disambiguates visually-similar linear features via map identity |
| **3** | 3. `junction_open_exit_count` | ★★★★☆ | topology excellent | connectivity audit | count-class answer forces both halves |
| **4** | 6. `misclassified_route_is_staircase_direction` | ★★★★★ | highway class excellent | accessibility / routing bug | strongest *divergence* task; sample-size may be small |
| **5** | 4. `route_class_calming_type` | ★★★★☆ | highway class excellent | LTN / Superblock audits | interaction-keyed answer bucket |
| **6** | 5. `parallel_service_lane_provision` | ★★★★☆ | highway geometry excellent | corridor-design input | lateral binding makes it fusion |
| **7** | 7. `cycleway_continuity_direction` | ★★★☆☆ | cycleway tags good | bike-network continuity | direction anchor is bearing + OSM class |
| **8** | 8. `bus_stop_pair_shelter_direction` | ★★★☆☆ | bus_stop nodes good | transit-equity | needs guaranteed two-way pair |
| **9** | 9. `waterway_crossing_kind` | ★★★☆☆ | waterway ways good | flood resilience | near localize-then-name but saved by water-confirmation |
| **10** | 10. `landuse_boundary_direction` | ★★☆☆☆ | landuse polygons good | BID / zoning audit | correlation trap (real-boundary ≈ perceived boundary) |
| **11** | 11. `deadend_terminus_barrier` | ★★☆☆☆ | topology good + sparse barrier | gating audit | close to localize-then-name |

---

## 🏆 SINGLE STRONGEST TASK — **Task 1: `larger_green_entrance_direction`**

**One-line argument for why it provably requires fusion:**

The answer (direction) is the intersection of two properties that are **each unobservable from one view alone** — *which polygon is the larger of the two* is a plan-view area comparison the street-view cannot perform (area is not a street-level observable, and the two adjacent polygons typically present near-identical street-face edges), and *which polygon's boundary-edge at the dot has a walkable entrance* is a pedestrian-scale affordance invisible from top-down. Binding the satellite's **area-identity** ("the larger polygon is the one to the NW") to the street-view's **edge-type** ("at my cross_left, this particular wall / gate belongs to which polygon") yields the direction answer; deleting either view collapses the question to an unanswerable guess between two visually indistinguishable candidates.

**Proof sketch of no single-view oracle exceeding chance:**

A perfect street-view oracle can see all entrances near the dot and assign each to a direction, but it cannot resolve the *membership* ambiguity: when a wall in `cross_left` and a gate in `along_fwd` each border a green polygon, and the two polygons share a corner within 30 m of the dot (a very common configuration for park-pocket-garden pairs), no purely visual cue (wall height, gate size, hedge type) reliably identifies which polygon is which, because both polygons can have identically-walled and identically-gated edges. A perfect satellite-view oracle knows polygon area and thus which is "the larger," and knows each polygon's extent, but cannot resolve which edge-segment of the larger polygon is the **entrance edge** versus a service edge, locked edge, or river-wall edge — because OSM `leisure=park` polygons almost never carry dense and reliable `entrance=*` nodes, and plan-view imagery cannot distinguish an unlocked public gate from an identical locked service gate at the pixel resolutions we work with. The answer therefore lives in `Entrance(larger_polygon) ∩ VisibleFrame(dot)`, which is a set-intersection neither marginal oracle can compute alone. With four answer options and no strong city-prior (a city's parks do not systematically put the entrance of the *larger* park on a specific frame relative to the dot), a single-view oracle's expected accuracy upper-bounds at chance + small correlation (bounded by the empirical correlation between polygon area ranking and entrance orientation, empirically ≤ 5 pp).

---

**What I rejected and why (for your audit):**
- **`bus_stop_shelter_direction` (single stop):** localize-then-name; direction answers street-view alone.
- **"What type of crossing at a railway?":** street-view alone answers once bearing points the right way.
- **Storey/material/shop POI tasks:** ruled out by Constraint 2.
- **Agreement/consistency/mismatch framing:** ruled out by Constraint 4 — every fusion task above produces a *category, count or direction*, never a yes/no-agree.
- **Geometry-puzzle "which-frame-does-the-arrow-point-at":** excluded per Constraint 3.

kimi k2.7:
 Below are **10 task candidates** designed to break the “scalar-bin fusion curse.” Each answer is a **relation** between a satellite-owned, plan-view **structure/topology/direction** and a street-owned **physical affordance / access**. They use only dense OSM layers and address real urban-planning questions.

---

### 1. Forward Link Form at the Dot
**Question (MCQ):**  
*“You are at the red dot, standing on a road. Looking forward, in the `along_fwd` view, the map shows a route continuing in the road’s forward direction. What kind of route is it physically?”*

**Options:**  
A. A regular drivable road (open to cars)  
B. Pedestrian-only stairs or steps  
C. A non-vehicular path / ramp / track  
D. A dead-end or private driveway / no continuation for public through traffic

**Satellite-alone fails:**  
The plan-view can only see a thin line / trace or a narrow opening between roofs. A 2 m-wide cut could be a footway, stairs, a ramp, or a service driveway; satellite cannot see vertical steps, surface texture, or “no entry” bollards at ground level.

**Street-view-alone fails:**  
The forward frame shows what is directly in front, but it does not know which of the several visible road/path fragments is the **OSM-mapped forward way** that leaves the junction; a curb-cut into a courtyard and a public footway can look nearly identical.

**Acid-test verdict:**  
Yes. Even given the exact outgoing OSM segment, its physical form (drivable vs. stairs vs. path vs. cul-de-sac) is readable only at eye level. Even given a forward street frame, the topological identity of the forward OSM way must come from the map.

**Label derivation:**  
Use OSM `highway=*` ways. Find the road node nearest the dot, select the outgoing `highway` way whose first segment bearing is within ±45° of `road_bearing`. Then use `highway=*`:  
- drivable road → A (`motorway/trunk/primary/secondary/tertiary/unclassified/residential/living_street/service`)  
- `highway=steps` → B  
- `highway=footway/path/cycleway/bridleway/track` → C  
- dead-end (`noexit=yes`, node degree 1, or `access=private` driveway) → D

**Urbanization value:**  
Directly tests whether the street network is walkable/rollable or is broken by stairs, dead-ends, or private drive-throughs — critical for routing, logistics, and accessibility.

**Constraint check:**  
Not a yes/no mismatch task. Uses only `highway=*` tags, which are globally dense. No `building:levels`, materials, shops, or POIs.

---

### 2. Vertical Relation at a Forward Crossing
**Question:**  
*“Following the road forward from the red dot, the map shows that your road will cross another road, railway, or waterway. How will your road physically relate to that crossing?”*

**Options:**  
A. At-grade crossing (level with the other feature)  
B. Road goes over the other feature (bridge / flyover)  
C. Road goes under the other feature (tunnel / underpass)  
D. No such major crossing is mapped within the next segment

**Satellite-alone fails:**  
A top-down image can **never** observe vertical elevation or tunnel buriedness; two crossed lines look identical whether they are an ordinary intersection, a flyover, or a hidden tunnel under buildings/trees.

**Street-view-alone fails:**  
The `along_fwd` frame may show a bridge deck or a dark tunnel mouth, but it cannot tell that **it is the upcoming crossing of a specific railway/waterway** and not just a local grade change or covered arcade.

**Acid-test verdict:**  
Yes. The map provides the topological statement “the forward way crosses feature X within Δ meters”; the street view resolves the vertical Z relation. Either one alone is indeterminate.

**Label derivation:**  
Find the forward `highway` way as in Task 1. Walk its first edges until it intersects an OSM `waterway=*`, `railway=*`, or major road (`highway=primary/trunk/motorway`). If the crossing segment has any `bridge=*` → B; any `tunnel=*` → C; otherwise an actual intersection → A; if no feature is intersected within a fixed length (e.g., 75 m) → D.

**Urbanization value:**  
Grade severance dramatically changes walkability and emergency-route feasibility; this quantifies where the map topology is vertically “hiding” accessibility issues.

**Constraint check:**  
Pure topology + `bridge`/`tunnel` tags — both dense and globally reliable. Not an agreement check.

---

### 3. Direction to the Nearest Usable Crossing of a Major Barrier
**Question:**  
*“A major river, railway, or fast main road runs near the red dot. In which direction should you walk along the road network to reach the nearest usable bridge, underpass, or ferry crossing that lets pedestrians/cyclists cross that barrier?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
From above an underpass is invisible and a footbridge may look identical to a road bridge. Even when a crossing is visible, its **permitted mode** (cars only vs. pedestrians allowed) cannot be read from roofs.

**Street-view-alone fails:**  
The four frames show only the immediate foreground; the actual nearest crossing may lie behind, around the corner, or be hidden by buildings. Without the map, the frames are unrelated to the nearest barrier.

**Acid-test verdict:**  
Yes. The map gives the nearest linear barrier and the location of every crossing structure; the street frames verify that the crossing is physically open/useable; neither marginal view alone knows both.

**Label derivation:**  
1. Identify the nearest linear barrier within radius R: OSM `waterway=*`, `railway=*`, or `highway=motorway/trunk/primary`.  
2. Find all OSM ways that intersect it and carry a `bridge=*`, `tunnel=*`, or `ferry=*` tag, or are themselves `highway=footway/path/cycleway` that connect across the barrier.  
3. Keep only those with explicit or implicit foot/cycle access (avoid pure motorway bridges unless `foot=yes`).  
4. Compute bearing from dot to the nearest such crossing; assign to the nearest sector among the four.

**Urbanization value:**  
Measures **severance** — whether a city’s large infrastructure slices neighborhoods into disconnected pieces, a core walkability/transport equity indicator.

**Constraint check:**  
Uses only network geometry + dense barrier/crossing tags. Answer is a direction, not a “do views agree” check. No forbidden tags.

---

### 4. Nearest Accessible Edge of Public Open Space
**Question:**  
*“The map shows a park, square, or water area near the red dot. Which street-view direction faces the nearest publicly reachable edge of that open space (where the ground is actually open, not guarded by a back wall or fence)?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
Satellite sees the green polygon, but it cannot tell which side has a gate/path and which side is a fenced backyard, a loading bay, or a seawall.

**Street-view-alone fails:**  
A frame may show a gate to a park, but without the map it cannot tell whether that gate belongs to the **nearest mapped public open space** or to a private compound; nor can it know whether a different, closer usable edge lies in another direction.

**Acid-test verdict:**  
Yes. The map pinpoints the open-space polygon; the frames distinguish public frontage vs. barriers. The needed answer (direction) is the combination of both.

**Label derivation:**  
1. Take the nearest large OSM polygon of type `leisure=park/garden/recreation_ground`, `landuse=grass/forest`, `natural=water`, `waterway=riverbank`, or `landuse=reservoir/basin` within radius R.  
2. Define an “accessible edge point” as a point on the polygon boundary that is touched by or within δ of an OSM `highway=*` (road/footway/path).  
3. Choose the nearest such point; compute bearing from dot; assign to the closest 90° sector.

*Note:* We **do not** rely on sparse `barrier=*` tags; the proxy “touches a public highway” is enough and derives from dense layers.

**Urbanization value:**  
Captures public-space access and frontage quality — **walkable green/water contact** vs. fenced-off green space.

**Constraint check:**  
No forbidden tags; not a consistency task.

---

### 5. Direction to Nearest Transit Boarding Access
**Question:**  
*“A public transport stop or platform is mapped near the red dot. Which direction should a pedestrian follow from the road to reach it?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
A small shelter or bus pole is invisible at satellite resolution; even if visible, the **safe/legal boarding side** (which side of the dual carriageway) is unknown from above.

**Street-view-alone fails:**  
The frames may or may not contain the shelter; without knowing where the nearest mapped transit node is relative to the dot, there is no way to decide which sector is correct.

**Acid-test verdict:**  
Yes. The map gives the transit node location/topology; the street view gives the identifiable stop/shelter/pole; the answer is their angular alignment.

**Label derivation:**  
1. Find the nearest OSM transit feature within radius R: `public_transport=platform`, `public_transport=stop_position`, or `highway=bus_stop`.  
2. Project it onto the nearest OSM road/sidewalk way.  
3. Compute bearing from dot to the projected access point.  
4. Assign to the closest of the four sectors.

**Urbanization value:**  
Measures last-mile public-transport accessibility and the legibility of transit access — directly relevant to multimodal trip planning.

**Constraint check:**  
Uses only transit nodes and the road network, both dense in urban areas. Not a POI-count; not a mismatch check.

---

### 6. Direction of a Pedestrian Cut-Through Between Buildings
**Question:**  
*“A narrow public alley, arcade, or footpath cuts through the building block near the red dot. In which direction do you enter that cut-through from the road you are on?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
Roof projections hide narrow passages; a 2 m-wide alley between buildings looks like a continued solid block from above, especially under tree cover.

**Street-view-alone fails:**  
The entrance to a cut-through looks like any gateway, garage opening, or building arcade; without the map one cannot know it connects through to another public street.

**Acid-test verdict:**  
Yes. The satellite/plan-view gives the block topology and the proof that the footway *penetrates* the building cluster; the street view validates the actual ground-level opening.

**Label derivation:**  
1. Consider OSM `highway=footway/path/cycleway/steps` ways within radius R.  
2. Keep those whose endpoints attach to **two different** OSM road ways, whose length is short (<80–100 m), and that have OSM building polygons within a small buffer on both sides for most of their length.  
3. Compute the bearing from the dot to the nearest such cut-through entrance; assign sector.

**Urbanization value:**  
Tests permeability of the urban fabric — a key determinant of walkability, shade, and safety for pedestrians.

**Constraint check:**  
Uses building footprints (dense) + `highway=*` paths. No forbidden tags.

---

### 7. Nearest Service / Delivery / Parking Access Direction
**Question:**  
*“A service road, driveway, or parking-aisle connects a nearby building or parking lot to the public road at the red dot. In which direction is the nearest such access point?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
Service lanes and parking aisles are often partially covered by canopy/buildings or too thin to resolve; a satellite image cannot reliably distinguish a public through-street from a private ramp.

**Street-view-alone fails:**  
A driveway is visible, but the question asks for the **nearest mapped access** to a specific building or parking lot; without the map one cannot know which driveway belongs to the target parcel or whether a nearer access is hidden around a corner.

**Acid-test verdict:**  
Yes. The map ties the access way to the building/parking polygon; the street view reads whether the access actually opens to the street.

**Label derivation:**  
1. Find OSM `highway=service` ways that end on, or are contained within, an OSM building footprint polygon or a `landuse=parking` polygon.  
2. Compute distance and bearing from dot to the nearest such service-way entry point; assign to a sector.

**Urbanization value:**  
Directly relevant to freight, delivery, curb management, and parking policy.

**Constraint check:**  
Uses dense `highway=service`, building footprints, and parking polygons. No POI counting.

---

### 8. Nearest Waterfront / Promenade Path Direction
**Question:**  
*“A river, canal, lake, or coastline is mapped near the red dot. Which direction faces the nearest public path or promenade running alongside that water edge?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
Satellite can see the water edge but cannot tell which stretch is walkable promenade, which is a private marina, and which is a walled embankment.

**Street-view-alone fails:**  
A frame might show a railing and water, but without the map it cannot determine that this is the nearest accessible **mapped water-edge path** relative to the dot, or distinguish a brief river glimpse from a continuous promenade.

**Acid-test verdict:**  
Yes. Map gives the waterway and the continuous path strip; street view verifies whether the path is physically present and open.

**Label derivation:**  
1. Find the nearest OSM `waterway=*` or `natural=coastline` line within R.  
2. Find OSM `highway=footway/path/cycleway` ways that reside within a narrow buffer of the waterway and have length ≥30 m.  
3. Compute bearing to the closest such path; assign sector.

**Urbanization value:**  
Blue/green space access and water-front connectivity — quality-of-life and flood/heat mitigation questions.

**Constraint check:**  
No forbidden tags; relies on dense waterway and highway layers.

---

### 9. Nearest Marked Pedestrian Crossing on the Road
**Question:**  
*“You are crossing the road that runs through the red dot. In which cross-street direction should you look to find the nearest marked pedestrian crossing (zebra, signalized crossing, or footbridge) of that road?”*

**Options:**  
A. `cross_left`  
B. `cross_right`  
C. `along_fwd`  
D. `along_bwd`

**Satellite-alone fails:**  
A painted crossing is far below satellite resolution; a footbridge is sometimes visible, but it cannot be distinguished from a sign gantry or car bridge.

**Street-view-alone fails:**  
The `cross_left`/`cross_right` views show the near sides of the street, but the nearest marked crossing may be tens of meters away, around a bend, or on the far side of an obstruction. The map is needed to choose the actual nearest.

**Acid-test verdict:**  
Yes. Map identifies the crossing node along the road to which the dot belongs; street view verifies the crossing markings/structure; the answer is the local side direction.

**Label derivation:**  
1. Find OSM nodes with `highway=crossing` that lie on the road way nearest the dot, or within a small buffer (e.g., 5 m).  
2. Optionally refine by `crossing=*` values: keep only `zebra`, `traffic_signals`, `uncontrolled`, etc.  
3. For each candidate, determine whether it lies to the left or right of the travel direction defined by `road_bearing`, or forward/backward along the road; assign sector.

**Urbanization value:**  
Directly measures pedestrian-level road safety and crossing accessibility for older people, children, and wheelchair users.

**Constraint check:**  
`highway=crossing` is dense at pedestrian crossings. Not a mismatch check. No forbidden tags.

---

### 10. Nearest Mapped Stairway / Step Route Direction
**Question:**  
*“A mapped pedestrian stairway or set of steps is located near the red dot, connecting different ground levels. In which direction do you approach that stairway from the road you are on?”*

**Options:**  
A. `along_fwd`  
B. `along_bwd`  
C. `cross_left`  
D. `cross_right`

**Satellite-alone fails:**  
Stairs are narrow, often shaded or covered, and have the same footprint as a small footway from above; only the eye-level view reveals the vertical steps or handrails.

**Street-view-alone fails:**  
A street frame may show steps nearby, but the model needs the map to know which visible stairway is the **nearest mapped** one and whether it is reachable without trespassing.

**Acid-test verdict:**  
Yes. Map gives the staircase location and its relation to the road network; street view gives the physical stair/ramp distinction; the answer is their relative direction.

**Label derivation:**  
1. Find OSM ways/nodes with `highway=steps` within radius R.  
2. Compute bearing from dot to the nearest stairway; assign sector.  
3. Restrict to stairways whose closest OSM road is within walking distance and connected.

**Urbanization value:**  
Key for accessibility: ramps vs. stairs determine whether the neighborhood is passable for wheelchairs, strollers, or bicycles; also relevant for emergency egress.

**Constraint check:**  
Uses only `highway=steps`, a dense and clearly defined OSM layer. No forbidden tags.

---

## Ranking

Ranked by **(fusion cleanliness × OSM reliability × urban value)**:

| Rank | Task | Why it ranks here |
|------|------|-------------------|
| **1** | **2. Vertical Relation at a Forward Crossing** | Cleanest cut: satellite has the topology, street has the Z-axis, and **no real-world correlation** can leak because vertical grade is invisible from above. OSM `bridge`/`tunnel` are globally dense. |
| 2 | 3. Nearest Crossing of a Major Barrier | High urban relevance (severance); uses dense network + bridge/tunnel; direction answer forces neither view to be enough. |
| 3 | 6. Pedestrian Cut-Through Direction | Strong fusion: map supplies block topology, street supplies the tiny opening; very useful walkability indicator; relies only on dense building footprint/path layers. |
| 4 | 1. Forward Link Form at Junction | Very reliable (`highway=*` only); physical form of the forward exit is street-level information, but a bit more solvable by width cues from very good satellite models. |
| 5 | 4. Nearest Accessible Open-Space Edge | Great planning value, but the OSM proxy (highway touches polygon) is noisier than bridge/tunnel tags. |
| 6 | 5. Nearest Transit Boarding Direction | Useful for multimodal accessibility, but OSM transit nodes are not as uniformly dense as roads. |
| 7 | 8. Waterfront Promenade Direction | Useful where waterways exist; less universally applicable. |
| 8 | 9. Nearest Marked Pedestrian Crossing | Urban value is high, but `crossing=*` sub-tags are uneven across countries; OSM crossing nodes themselves are dense. |
| 9 | 7. Service/Delivery Access Direction | Routing value, but service ways and parking polygons can be ambiguous or missing in areas with informal parking. |
| 10 | 10. Nearest Stairway Direction | Reliable and accessibility-relevant, but more geographically clustered (hilly cities). |

---

## Single Strongest Task: **Vertical Relation at a Forward Crossing**

**One-line argument:**  
The map tells the model **which route is about to cross what**, and the street view tells the model **who is above whom**; neither view alone determines the relative elevation of two intersecting lines, so the answer is intrinsically a relation, not a scalar.

**Proof sketch against single-view oracles:**

- **Satellite-only:** From above, two linear features crossing produce the same 2-D projection for an at-grade intersection, a flyover, or an underpass. A tunnel is often hidden by buildings or tree canopies, and a bridge deck is locally continuous with the road surface, so a perfect satellite oracle cannot score above chance on bridge/tunnel/at-grade among crossing cases.
- **Street-view-only:** A single forward photo shows bridge railings, a tunnel mouth, or plain road — but the model does not know whether the structure crosses the nearest railway/waterway or is just a local dip/cover, nor does it know the exact spatial relation to the road network; therefore it is also deprived of the joint information that defines the label.

Hence any oracle constrained to only one view lacks a necessary component of the relation `Y = g(map_topology, ground_vertical_affordance)`, making this a provably fusion-dependent task.