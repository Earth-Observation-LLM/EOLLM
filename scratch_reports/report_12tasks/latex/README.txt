EOLLM Dataset Technical Report — LaTeX source
==============================================

Layout
------
  main.tex              Single-file source (no \input chain).
  figures/              Vector PDFs for all 18 statistical figures.
  examples/             PNG photos for the 12 example QA panels.
  README.txt            This file.

Compile
-------
Requires a working TeX distribution (TeX Live, MacTeX, or MiKTeX) with
the standard packages used in main.tex (graphicx, tabularx, hyperref,
booktabs, tcolorbox, microtype, titlesec, subcaption, listings).

  pdflatex main.tex
  pdflatex main.tex      # second pass for ToC

Output: main.pdf

Notes for printing
------------------
* All statistical figures (figures/fig*.pdf) are vector PDFs and can
  be enlarged to any poster size without loss.
* Example QA panels (examples/*.png) embed real satellite + Street
  View photographs at ~512–1024 px resolution. Pleasant up to A3.
  For larger prints, re-render them at higher tile size with the
  build_report_figures.py script (set tile_size=512 in
  composite_utils calls).
* If you need a single bundle, this folder is self-contained — zip
  the entire latex/ directory.

Regenerating the figures from scratch
-------------------------------------
The script that built every figure lives at:
  ../scripts/build_report_figures.py
Run with any env that has matplotlib, numpy, Pillow. It regenerates the
released splits on the fly from dataset_merged.jsonl using the project
splitter (splitting/split_dataset.py, seed=42); point it at a different
merged file with the EOLLM_MERGED environment variable.

Note: green_space and road_surface were retired from the released
benchmark (solved trivially by all evaluated models), so the corpus now
spans 12 task types / 34,484 question records.
