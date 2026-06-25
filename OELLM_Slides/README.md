# OELLM — Defense Slides (~8 min, 9 slides)

Beamer deck for the graduation-project defense. Palette and figures match the OELLM poster.

## Files
- `slides.tex` — the deck source (9 frames).
- `slides.pdf` — compiled output (already built).
- `figures/` — all images the deck references (self-contained; no external paths).

## Compile

The deck was built with **tectonic** (self-contained, auto-downloads packages):

```bash
tectonic slides.tex
```

Or with a standard TeX Live install:

```bash
latexmk -pdf slides.tex      # or: pdflatex slides.tex  (run twice)
```

The deck uses the **metropolis** beamer theme. If it is not installed:
- TeX Live:  `tlmgr install beamertheme-metropolis`
- Or replace `\usetheme{metropolis}` near the top with `\usetheme{default}`
  (the OELLM colour overrides still apply; the closing slide is theme-independent).

## Slide arc
1. Title
2. The problem — urban understanding + cross-view; unimodal shortcut learning
3. The dataset — OSM-grounded pipeline, 40 cities, 39,155 records
4. Method — Qwen3.5-4B-VL + LoRA, three splits, greedy eval
5. Headline result — 4B model beats six open VLMs (62.9 / 60.4)
6. The finding — small PC↔SU gap = geographic generalisation
7. Ablations — three transfer regimes (within-family / cross-family / zero)
8. Impact & future
9. Conclusion (standout)

## Notes
- Two harmless "Overfull \vbox" warnings (problem slide image column, paired-story
  slide) — no visible clipping in the rendered PDF.
- Frame numbering shows "/8" because the closing slide is a plain frame excluded
  from the count; cosmetic only.
- To swap a figure, drop a replacement into `figures/` with the same filename.
