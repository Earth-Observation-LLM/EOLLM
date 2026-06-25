# EOLLM paper — TODO / roadmap

Tracking tasks for the GAIA@ECCV 2026 paper
("Single View Might Be Enough"). Each task is a separate Markdown.

Paper lives in: `GAIA_ECCVW_2026_Paper_Template/` (main.tex = blind submission,
main_named.tex = named proof). Verified-numbers note:
`scratchpad/grounding_notes.md` (session) and `benchmark_suite/results/analyze_synergy.py`.

## Submission-blocking (do before submitting)
- [ ] [01_ablation_single_sv_angle.md](01_ablation_single_sv_angle.md) — defuse "4 images vs 1" confound
- [ ] [02_ablation_unmarked_satellite.md](02_ablation_unmarked_satellite.md) — defuse "reads the red dot" confound
- [ ] [03_figures.md](03_figures.md) — main-paper figures (in progress)
- [ ] [04_deai_prose_pass.md](04_deai_prose_pass.md) — remove LLM-cadence prose (prior-rejection wound)
- [ ] [05_geo_identifier_blind.md](05_geo_identifier_blind.md) — strip city/coords from prompt, re-run blind

## Research follow-ups (next paper / extensions)
- [ ] [10_attention_image_selection.md](10_attention_image_selection.md) — use attention-when-correct to pick informative images
- [ ] [11_view_router.md](11_view_router.md) — train a router that selects which view/image(s) to feed
- [ ] [12_two_straightview_ablation.md](12_two_straightview_ablation.md) — show 2 straight-view images ≈ full (panorama not always needed)

## Notes
- All agents/figures must use VERIFIED numbers only (the old `EOLLM_Report_LaTeX/latex/`
  report is discredited; old `figures/*.pdf` carry wrong full-dataset counts — regenerate).
- No AI-generated imagery anywhere (prior paper rejected partly over a Gemini watermark).
- green_space task is EXCLUDED everywhere (100% blind text leak).
