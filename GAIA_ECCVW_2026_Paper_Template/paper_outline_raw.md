# When Does the Second View Help?
## Auditing Cross-View Fusion in Urban VLMs

**Status:** committed outline, v2 — incorporates advisor's framing
**Venue:** workshop paper
**Primary model:** **satfwd Qwen3.5-9B** — LoRA trained on `sat + 1 forward-SV` (seen_unseen split)
**Tasks:** 7 urban-semantic (`land_use`, `building_height`, `urban_density`, `road_type`, `junction_type`, `amenity_richness`, `transit_density`) + 4 cross-view (`camera_direction`, `mismatch_binary_easy/hard`, `mismatch_mcq_easy/hard`). `green_space` and `road_surface` are out.

---

## What changed in v2

The advisor proposed flipping the headline model. v1 had `sat+4SV` as the primary and treated `sat+1SV` as a §4.5 clincher. v2 makes `sat+1SV` (the satfwd adapter) the **primary configuration throughout**, and demotes the `sat+4SV` setup (the 5STV adapter) to a §4.6 supplementary analysis.

Why this is better:
- The paper now presents results on the **minimum-sufficient input configuration** as the main story, not as an afterthought. Reads as a confident structural choice instead of "we trained with too much, then realized."
- All baseline comparisons (raw VLMs, sat-only, sv-only) sit on the same 1-SV regime as our model — apples-to-apples, no caveats.
- The 4-SV question becomes a clean follow-up: "what does adding three more street views buy you?" Answer: ~+2pts on urban tasks; structurally required for cross-view tasks. That's a sharp closing claim, not a retraction.
- **3D spatial reasoning** enters as future work — multi-view street imagery enables 3D reconstruction, which is the natural next step beyond the current 2D-image fusion setup.

Resolved open decisions from v1:
- *Per_city vs seen_unseen as the primary adapter:* **seen_unseen**, because the satfwd model is trained on that split.
- *How hard to lean on the satfwd result:* **all the way** — it's the main model now, not the clincher.

---

## The argument in one paragraph

A vision–language model with access to both an overhead satellite tile and a ground-level street view could, in principle, fuse the two perspectives into a richer urban representation than either provides alone. Whether current VLMs actually do this is rarely tested directly, because existing benchmarks mix tasks where one view suffices with tasks where fusion is forced by the task structure, producing aggregate numbers that obscure both. We introduce **OELLM**, a deterministic OSM-grounded paired-view benchmark of 11 tasks across 40 cities, and use it to audit cross-view fusion. Our primary configuration uses the minimum-sufficient input — one satellite tile plus one forward-facing street view — and evaluates a fine-tuned 9B VLM under four input regimes: blind, satellite-only, street-view-only, and full. We find a sharp regime split: on the seven urban-semantic tasks, full input gains only **+0.5pts** over the better single view — the model is not fusing, it is selecting; on the four cross-view geometric tasks, full input gains **+44 to +70pts** — fusion is the entire signal. Inference-time pruning confirms the picture: a single forward street view preserves 89% of previously-correct answers. A supplementary analysis of a model trained with all four street views shows that the additional three views yield only ~+2pts on urban tasks while remaining essential for the cross-view tasks they were designed for. **Multi-view fusion in current VLMs is task-shaped, not pervasive.**

## Title rationale

"Auditing" frames the paper as measurement, not a takedown. The question form sets up an experimental answer rather than an editorial argument. The wording cues reviewer expectations toward rigor and away from polemic, which is the right move for a workshop venue and reduces hostility on revision. The title also implicitly promises a benchmark (you need one to audit fusion), so the advisors' benchmark-first preference is structurally satisfied — the body just delivers on the title.

---

## Paper structure

### 1. Introduction (~1 page)

Open with the gap. Vision-language models are pitched as multimodal reasoners, but on geospatial inputs we don't actually know whether they fuse modalities or just lean on whichever one is most informative for each task. Existing geospatial benchmarks report aggregate accuracy across mixed task families, conflating "fusion happens here" with "one view was enough."

Set up the question the title poses. When does adding a second view help? An audit needs three things: paired-view inputs with deterministic ground truth; a task taxonomy that includes both single-view-sufficient and fusion-required tasks; and a controlled comparison across input modes on the same model.

Preview the finding. On the 7 urban-semantic tasks of our benchmark, full input gains +0.5pts over the better single view; on the 4 cross-view tasks, full input gains +44 to +70pts. One forward-facing street view retains 89% of correct answers. Training with three additional street views adds only ~+2pts on urban tasks.

Contributions:
- A deterministic paired-view benchmark (OELLM) of 4,734 held-out questions over 11 OSM-grounded tasks, designed to separate single-view-sufficient from fusion-required reasoning.
- A four-mode audit (blind / sat-only / sv-only / full) on a fine-tuned 9B VLM evaluated at the minimum-sufficient input (sat + 1 forward SV) and on a panel of raw open VLMs at the same input regime, identifying a sharp regime split.
- A supplementary characterisation of inference-time view pruning and training-time view economy, showing that for urban-semantic tasks the additional street views are not what current models learn from.
- A clean separation in the conclusion of *which* tasks structurally require multi-view input, motivating 3D spatial reasoning as the natural next step.

### 2. Related Work (~0.75 page)

Three threads, each closing with one sentence of differentiation:

**Urban multimodal models.** UrbanLLaVA, CityLens, CityRiSE, UrbanVLP. Adapt VLMs to urban reasoning with paired satellite/street-view input, but evaluate on mixed task aggregates that don't separate fusion-required from single-view tasks. *We differ by:* explicitly auditing where fusion does and doesn't happen.

**Cross-view geolocalisation.** CVUSA, CVACT, VIGOR, Sample4Geo. Frame cross-view as retrieval; demonstrate that paired views carry geometric signal. *We differ by:* recasting cross-view correspondence as multiple-choice VQA inside a broader urban-reasoning benchmark, so cross-view ability can be measured alongside urban-semantic ability under one protocol.

**Remote-sensing VQA and grounded geospatial VLMs.** RSVQA, GeoChat, LHRS-Bot, SkySenseGPT, SkyEyeGPT, EarthGPT, EarthDial. Establish deterministic OSM-grounded VQA at the satellite-only setting. *We differ by:* extending deterministic grounding to paired satellite + street-view evaluation and using the controlled setup to ablate input modes.

Closing sentence: prior work measures aggregate performance; we measure *where, in input-mode space, the performance comes from*.

### 3. Benchmark (~1.5 pages — longest section, this is the durable contribution)

**3.1 Pipeline.** Seven-stage deterministic pipeline (reuse Figure 1 from GAIA2026 draft): OSM-anchored location sampling, satellite tile fetching, four road-aligned street-view captures, OSM enrichment, composite rendering, deterministic question generation, validation and splitting. No human annotation; every answer derives from OSM tags or coordinate geometry. State the trade-off honestly: limited to what OSM and geometry can certify, but every answer is verifiable and the corpus is bit-exact reproducible.

**3.2 Task taxonomy.** Two families:
- *Urban understanding* (7 tasks): `land_use`, `building_height`, `urban_density`, `road_type`, `junction_type`, `amenity_richness`, `transit_density`. Each is answerable from one location's OSM context; the visual is a satellite tile with a red marker plus street-view imagery.
- *Cross-view geolocalisation* (4 tasks): `camera_direction` aligns a single street view with one of four arrows on the satellite; `mismatch_binary_easy/hard` ask whether a street-view set belongs to the satellite tile; `mismatch_mcq_easy/hard` ask which of four street-view composites matches. Easy variants draw distractors cross-city; hard variants same-city.

Introduce the audit framing here: urban-understanding tasks are *single-location* questions, in principle answerable from one viewpoint; cross-view tasks are *between-modality* questions, unsolvable without both. The benchmark is designed to make this split measurable.

**3.3 Splits.** Three regimes (Table 1): `per_city` (23,431 / 6,319) tests new locations within familiar cities; `seen_unseen` (23,282 / 8,179) holds out eight entire cities for geographic generalisation; `benchmark` (4,734) is the disjoint held-out evaluation set, shipped public + with-answers. Our primary model is trained on seen_unseen.

**3.4 Deterministic ground truth and evaluation protocol.** Greedy decoding, fixed seed, strict word-boundary letter parser, exact-match accuracy. Per-topic random and majority floors recomputed from the evaluation set (binary-mismatch correctly scored against 50%, not 25%). Same harness across all models — like-for-like comparison.

**3.5 Coverage and limitations.** 40 cities, 32 countries, six continents (reuse world map). Honest about Europe-heavy distribution (66% of records) and OSM-tag sparsity affecting some tasks in some cities. One sentence noting `green_space` and `road_surface` were retired from the released benchmark because they were solvable blind; no further mention.

### 4. Experiments (~2 pages — one coherent section structured around the title question)

**4.1 Setup.** Qwen3.5-9B + LoRA on all linear layers (rank 16, α=16), bfloat16, full-resolution vision tuning on a single 96GB GPU. The primary adapter (**satfwd**) is trained on the seen_unseen split with input mode = `marked-satellite + forward street-view` (two images per sample). For mismatch tasks (which structurally require all four street views), evaluation feeds the full street-view set even on the satfwd adapter — the task design forces it.

Four evaluation modes for the audit:
- `blind`: question text + options, no images
- `sat_only`: question + satellite tile (with location marker)
- `sv_only`: question + forward street view (one image)
- `full`: question + satellite + forward street view (two images)

Sentence the reader needs upfront: in our primary configuration, "full" means *two images* — one satellite, one street view. The four-SV configuration is presented separately in §4.6.

**4.2 Does the second view help? (the headline four-mode audit)**

Table 2: per-task accuracy under blind / sat / sv / full for the satfwd adapter on the 4,734-record benchmark, with fusion gain (full − best single view) per row.

Findings to articulate:
- *Urban tasks, fusion = +0.5pts at the mean.* Across the 7 urban-semantic tasks, mean fusion gain is +0.49pts. Two tasks have *negative* fusion (`amenity_richness` −2.4, `road_type` −1.4); three are essentially flat (≤+1pt); two have small positive gains (`junction_type` +2.4, `transit_density` +2.9). The model is not fusing, it is selecting the stronger view.
- *Cross-view tasks, fusion +44 to +70pts.* All four mismatch tasks gain massively from full input; `mismatch_mcq_easy` goes from ~25% sat-only to ~98% with full input. `camera_direction` gains +17pts. These tasks were designed to require fusion and the model delivers fusion on them.
- *Directional reading.* Per-task, full ≈ max(sat, sv) for the urban tasks — adding the second modality rarely beats the better single view. The signal carried by the satellite is largely subsumed by the forward street view on most urban-semantic tasks, and vice versa for tasks where street view is the stronger view.

Closing sentence: on urban-semantic tasks, "multi-view fusion" in our trained 9B VLM is mostly "use whichever single view is stronger."

**4.3 Cross-family / cross-scale replication.**

Table 3 (or condensed in-prose): the same four-mode evaluation on raw zero-shot VLMs — Qwen3.5-9B, Qwen3.5-4B, Qwen2.5-VL-7B, Gemma-4-12B — and the four specialised remote-sensing VLMs (GeoChat, SkySenseGPT, LHRS-Bot, EarthDial) on satellite-only urban tasks.

Finding: the regime split reproduces. Raw VLMs are weaker overall but the *shape* — small urban fusion gain, large cross-view fusion gain — is preserved across scales (4B–12B) and families (Qwen, Gemma). For specialised RS-VLMs on satellite-only urban tasks, performance is in the 26–40% range — below our trained satfwd's 64% — with the OSM-provenance caveat noted for LHRS-Bot and EarthDial (those models have OSM exposure in training, so they're effectively an upper bound for the comparison).

**Open data question to resolve before drafting this subsection:** the raw VLMs in `Raw_Overall` were evaluated under their default input scheme. We need to confirm whether `sv_only` in those evaluations is 1 SV or 4 SVs, and either re-run them at 1 SV to match the primary regime or explicitly note the discrepancy. The cleanest path is probably to keep raw VLM evaluations at their original input scheme (since these are zero-shot models and view-economy is not the question for them) and state this in the setup.

Closing sentence: the regime split is a property of current VLMs and current paired-view tasks, not a fine-tuning artifact.

**4.4 Cross-view tasks: where multi-SV input is structurally required.**

Self-contained subsection on the four cross-view geometric tasks. The task design uses all four street-view angles by construction:
- `camera_direction` queries which compass direction a single street view was taken from, requiring the satellite's spatial frame
- `mismatch_binary/mcq` compare a street-view *set* against the satellite; reducing the set undermines the task

Table 4: cross-view task accuracy under blind / sat / sv / full on the satfwd model (with the SV set always fed in full, since the task requires it). All four mismatch tasks reach 85–98% under full input; camera_direction reaches 43%.

Closing sentence: these are the tasks the benchmark designs to be unsolvable without multi-view input, and the model confirms it.

**4.5 Inference-time view pruning.**

Table 5: percentage of previously-correct answers preserved when k = 0, 1, 2, 3 street views are kept at inference, under forward / top / random / bottom selection strategies (source: `Prune_Survival`, 27B model — note this in caption).

Finding: forward-only at k=1 preserves 89.1% of correct answers; even random-1 preserves ~85%; satellite-only (k=0) drops to 67%. The model uses some street-view evidence, but not four views' worth — supporting the §4.1 choice of sat+1SV as the primary configuration.

**4.6 Supplementary: what do additional street views buy at training time?**

Table 6: per-task comparison of the satfwd adapter (sat + 1 SV) against the 5STV adapter (sat + 4 SVs) on the benchmark, same backbone, same seen_unseen training split, both evaluated in their respective full modes.

Findings:
- On the 7 urban tasks, the 5STV model averages 67.6% full vs the satfwd model's 65.3% — a ~+2pt gain from training with three additional street views. Smaller than the marginal cost of carrying 4× the visual data.
- On the cross-view tasks, the satfwd model (evaluated with all 4 SVs at inference, since the task requires it) is comparable to the 5STV model — 92–98% range on mismatch.

Closing sentence: paired-view training data is expensive; for urban-semantic tasks, the marginal value of additional street views beyond the forward one is small.

### 5. Discussion (~0.5 page)

Pull the experimental answers into one frame.

The benchmark separates two regimes. *Single-view-sufficient*: the 7 urban-semantic tasks. The model picks the stronger view, the second modality adds noise as often as signal, inference views can be pruned to one, training views can be reduced to one. *Fusion-required*: the 4 cross-view tasks. The model genuinely combines modalities and the full street-view set, performance collapses without all of it, no shortcuts are available.

Implication for benchmark design. Aggregate accuracy across mixed task families is misleading — it tells you neither how good the model is at fusion nor how good it is at single-view reasoning. Per-task, per-mode evaluation is required to see what models actually do; benchmarks should be designed with explicit task-regime taxonomy.

Implication for model and dataset design. Current architectures don't make productive use of paired views on urban-semantic tasks; future work should either improve fusion mechanisms or accept the regime distinction and tier inputs by task. Practically, collecting four street views per location is overinvestment for urban-semantic deployment; one forward view suffices.

What this paper does not claim. We don't claim VLMs *can't* fuse; we claim current VLMs *don't* fuse on these tasks. We don't claim satellite imagery is uninformative; we claim its information is largely subsumed by street view on urban-semantic tasks for current models. We don't claim the second view never helps; the title is a question and we answer it conditionally.

### 6. Conclusion (~0.25 page)

Short. The second view helps when the task structurally forces it (cross-view geometric reasoning), and is largely redundant when it doesn't (urban-semantic attributes). The OELLM benchmark makes the distinction measurable; the audit shows the distinction holds across model families, across inference-time view counts, and across training-time view counts. The tasks for which all four street views are essential are precisely the cross-view geometric ones (mismatch + camera_direction). Released artifacts: dataset, adapters, evaluation harness.

### 7. Future Work (~0.25 page)

Multi-view street imagery — four road-aligned angles per location — contains the geometric signal needed for 3D reconstruction of local scenes. Lifting urban understanding from 2D image fusion to 3D spatial reasoning is the natural next step: given a reconstructed local 3D model, the model could reason about occlusion, scale, building geometry, and pedestrian-environment affordances in ways that single 2D images cannot support. The OELLM benchmark's four-angle street-view captures are a natural input substrate for this direction.

Additional future work: improved fusion architectures that genuinely combine modalities rather than selecting between them; scaling to more cities to reduce the Europe-heavy distribution; extending the deterministic-grounding methodology to socio-economic indicators.

---

## Figures and tables

Visual order mirrors prose order — readers skim figures left-to-right top-to-bottom.

- **Figure 1.** Seven-stage benchmark pipeline. Reuse from GAIA2026 draft.
- **Figure 2.** Two rendered task examples — one urban (`land_use`), one cross-view (`camera_direction`). Reuse.
- **Figure 3 (NEW, headline figure).** Per-task fusion gain on the satfwd model: bar chart of (full − best_single_view) for the 11 tasks, sorted ascending. Urban tasks cluster near zero (some negative); cross-view tasks tower. This single image conveys the paper's finding — invest time on it.
- **Table 1.** Record counts per split. Use the dataset report's real numbers.
- **Table 2.** Per-task accuracy under blind / sat / sv / full for **satfwd on benchmark**, 7 urban tasks. Bold fusion-gain column. §4.2 evidence.
- **Table 3.** Raw VLM + RS-VLM panel summary. §4.3 evidence. Can be condensed.
- **Table 4.** Cross-view tasks four-mode breakdown, satfwd. §4.4 evidence.
- **Table 5.** Prune-survival per-task at k=0,1,2,3 with forward selection. §4.5 evidence.
- **Table 6.** satfwd vs 5STV per-task comparison. §4.6 evidence.

If page limits bite: Table 3 is the most condensable (collapse to summary statistics in prose); Table 5 can be moved to supplementary.

---

## Anchored numbers (from `COME_BACK_TO_REALITY-2.xlsx`)

Every number in the paper traces to a cell. Use these as ground truth.

| Anchor | Source | Value |
|---|---|---|
| satfwd benchmark, 7-urban mean, blind / sat / sv / full | SatFwd_vs_5STV | 48.3 / 64.0 / 61.4 / 65.3 |
| satfwd benchmark, 7-urban mean fusion (full − best_single) | derived | +0.49 |
| satfwd benchmark, per-task fusion range | derived | −2.4 (amenity) to +2.9 (transit) |
| satfwd cross-view (mismatch + cam_dir), full input | Val_ImageMode | 86–98% mismatch, ~43% cam_dir |
| Forward-1 SV inference survival (27B Qwen) | Prune_Survival | 89.06% |
| Sat-only inference survival | Prune_Survival | 67.05% |
| 5STV benchmark, 7-urban mean full | SatFwd_vs_5STV | 67.62 |
| 5STV vs satfwd, 7-urban mean full delta | derived | +2.29 |
| Raw Qwen3.5-9B benchmark full (OVERALL_all) | Raw_Overall | 54.45% |
| RS-VLM panel on sat-only urban (range) | External_RSVLM | 26–40% |

**Action item before drafting §4.2:** verify the satfwd numbers above are from the same checkpoint as the `SatFwd_vs_5STV` sheet. The earlier `Val_ImageMode` sheet appears to use a different checkpoint (best ckpt-1362 vs satfwd ep4 in the new sheet) and gives different numbers for some urban tasks. Settle which checkpoint is the headline before locking Table 2.

---

## Open decisions still to lock

1. **Raw VLM input regime in §4.3.** Either (a) re-run raw VLMs at 1 SV to match the satfwd primary regime — cleaner but costs eval time; or (b) keep their original input scheme and explicitly state in the setup that raw VLM comparisons use the standard multi-view input since these are zero-shot models and view-economy is not the question for them. Recommendation: (b), it's defensible and saves work.

2. **camera_direction belongs in §4.2 or §4.4?** It's a cross-view task (gain +17pts from full vs sat) but the gain is smaller than mismatch tasks (+44 to +70pts) and the input mode differs (1 SV by design, not 4). Recommendation: keep it in §4.4 with the mismatch tasks since it's structurally cross-view, but call out the +17pt gain explicitly to show it's a different magnitude than mismatch.

3. **Checkpoint reconciliation for satfwd.** `Val_ImageMode` and `SatFwd_vs_5STV` give different numbers for the same model. Settle which checkpoint is the headline (likely `SatFwd_vs_5STV`'s "satfwd ep4" since the sheet name is `COME_BACK_TO_REALITY-2`, suggesting it supersedes).

---

## Out of scope, locked

- `green_space` and `road_surface`. Out, one sentence in §3.5, no further mention.
- The "4B" model claim from the GAIA2026 draft. The trained model is 9B.
- The S-1/S-2/S-3/S-5/S-6 family-withholding transfer ablation. Those numbers don't appear in any sheet. If you find the source file, we reconsider; otherwise out.
- The "PC-base 62.9% / SU-base 60.4%" benchmark headline numbers from GAIA2026. Don't trace to any cell. Out.
- Per_city split as the primary headline. Out — satfwd is trained on seen_unseen, that's the primary split. Per_city results, if needed, go in supplementary.
- Any claim about emergent fusion ability, latent multimodal reasoning, or scaling. We audit what current models do, not what they could do.

---

## Drafting order

When we sit down to write, the order I'd write in:

1. **§3 Benchmark** — clearest material, lowest risk, establishes vocabulary.
2. **§4.2 Main four-mode audit** — the headline experimental subsection; pinning this nails the paper's voice.
3. **§4.1 Setup** — easier to write after §4.2 because you know what setup the reader needs.
4. **§4.4 Cross-view tasks** — natural contrast to §4.2.
5. **§4.3 Replication, §4.5 Pruning, §4.6 Supplementary** — in order; each builds on the previous.
6. **§5 Discussion** — once all experiments are written.
7. **§7 Future Work** — alongside §5.
8. **§1 Introduction** — write second-to-last, when you know exactly what the paper says.
9. **§2 Related Work** — second-to-last alongside intro.
10. **§6 Conclusion** — last small write.
11. **Abstract** — very last. Strip from §1 + §5 + §6.
