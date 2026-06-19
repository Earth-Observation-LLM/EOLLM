#!/usr/bin/env python3
"""
Beamer deck — urbanization tasks only, two sections, minimal text (numbers carry it):

  Section 1  4-model benchmark sweep: per urbanization task, full vs blind across
             gemma-4-12b / qwen2.5-vl-7b / qwen3.5-4b / qwen3.5-9b.
             (This sweep only asked the 9 attribute tasks in full + blind.)

  Section 2  Strategy run (Qwen3.5-9B): per urbanization task, full / sat / sv / blind
             — the run that DID ask attribute tasks in all four views, so it directly
             shows full ~ sv ~ sat (the second view is redundant).

Reads:
  benchmark_modes_remote/results/attr_fullblind_4model.json
  reasoning_distill_remote/results/attr_by_task.json   (strategy run, 'current' = thinking-off baseline)

Writes:
  benchmark_modes_remote/results/slides_urban/urban_views.tex (+ .pdf via tectonic)
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
STRAT = HERE.parent / "reasoning_distill_remote" / "results" / "attr_by_task.json"

NICE = {
    "land_use": "Land use", "building_height": "Building height",
    "urban_density": "Urban density", "junction_type": "Junction type",
    "green_space": "Green space", "amenity_richness": "Amenity richness",
    "road_type": "Road type", "road_surface": "Road surface",
    "transit_density": "Transit density",
}
MNICE = {
    "gemma-4-12b-awq": "Gemma-4-12B", "qwen2.5-vl-7b-awq": "Qwen2.5-VL-7B",
    "qwen3.5-4b-awq": "Qwen3.5-4B", "qwen3.5-9b-awq": "Qwen3.5-9B",
}
TASKS = ["land_use", "building_height", "urban_density", "junction_type", "green_space",
         "amenity_richness", "road_type", "road_surface", "transit_density"]


def cgain(x):
    if x >= 5.0:
        return rf"\textcolor{{good}}{{{x:+.1f}}}"
    if x <= 0.0:
        return rf"\textcolor{{bad}}{{{x:+.1f}}}"
    return f"{x:+.1f}"


def cdelta(x):  # full - best single view (section 2 synergy)
    if x >= 3.0:
        return rf"\textcolor{{good}}{{{x:+.1f}}}"
    if x <= -1.0:
        return rf"\textcolor{{bad}}{{{x:+.1f}}}"
    return f"{x:+.1f}"


def main():
    d4 = json.load(open(RES / "attr_fullblind_4model.json"))
    ds = json.load(open(STRAT))
    M = d4["models"]
    accs = ds["acc"]["current"]  # thinking-off baseline = the comparable mode

    o = []
    P = o.append
    P(r"\documentclass[aspectratio=169]{beamer}")
    P(r"\usetheme{metropolis}")
    P(r"\usepackage{booktabs}\usepackage{xcolor}\usepackage{array}\usepackage{multirow}")
    P(r"\definecolor{good}{HTML}{2E7D32}\definecolor{bad}{HTML}{C62828}\definecolor{accent}{HTML}{1565C0}")
    P(r"\setbeamercolor{frametitle}{bg=accent}")
    P(r"\setbeamertemplate{section page}{\centering\usebeamerfont{section title}\insertsectionhead\par}")
    P(r"\title{Do urbanization tasks need both views?}")
    P(r"\date{}\author{}")
    P(r"\begin{document}")
    P(r"\maketitle")

    # ---------------- Section 1
    P(r"\section{4-model sweep \textemdash{} full vs blind}")
    P(r"\begin{frame}{4-model sweep \textbf{$\cdot$} full vs blind \textbf{$\cdot$} 9 urbanization tasks}")
    P(r"\centering\scriptsize")
    P(r"\begin{tabular}{l" + "rr" * len(M) + "}")
    P(r"\toprule")
    P(r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{MNICE[m]}}}" for m in M) + r" \\")
    P("".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(M))))
    P(r"task & " + " & ".join(r"full & $\Delta$bl" for _ in M) + r" \\")
    P(r"\midrule")
    for t in TASKS:
        cells = []
        for m in M:
            full = d4["full"][m][t][0]
            blind = d4["blind"][m][t][0]
            cells.append(f"{full:.1f}")
            cells.append(cgain(round(full - blind, 1)))
        P(rf"{NICE[t]} & " + " & ".join(cells) + r" \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vspace{2pt}\par\tiny full $=$ both views \textbf{$\cdot$} $\Delta$bl $=$ full $-$ blind (vision gain) \textbf{$\cdot$} thinking off \textbf{$\cdot$} n$=$190--421/task")
    P(r"\end{frame}")

    # ---------------- Section 2
    P(r"\section{Single-model \textemdash{} all four views}")
    P(r"\begin{frame}{Qwen3.5-9B \textbf{$\cdot$} full / sat / sv / blind \textbf{$\cdot$} 9 urbanization tasks}")
    P(r"\centering\scriptsize")
    P(r"\begin{tabular}{lccccc}")
    P(r"\toprule")
    P(r"task & full & sat & sv & blind & full$-$best1 \\")
    P(r"\midrule")
    for t in TASKS:
        a = accs[t]
        syn = round(a["full"] - max(a["sat_only"], a["sv_only"]), 1)
        P(rf"{NICE[t]} & {a['full']:.1f} & {a['sat_only']:.1f} & {a['sv_only']:.1f} & "
          rf"{a['blind']:.1f} & {cdelta(syn)} \\")
    P(r"\midrule")
    # overall row (n-weighted)
    import math
    tot = {k: 0.0 for k in ["full", "sat_only", "sv_only", "blind"]}
    N = 0
    for t in TASKS:
        n = accs[t]["n"]
        N += n
        for k in tot:
            tot[k] += accs[t][k] / 100 * n
    ov = {k: round(100 * tot[k] / N, 1) for k in tot}
    syn = round(ov["full"] - max(ov["sat_only"], ov["sv_only"]), 1)
    P(rf"\textbf{{overall}} & \textbf{{{ov['full']:.1f}}} & {ov['sat_only']:.1f} & "
      rf"\textbf{{{ov['sv_only']:.1f}}} & {ov['blind']:.1f} & {cdelta(syn)} \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vspace{2pt}\par\tiny sat $=$ satellite only \textbf{$\cdot$} sv $=$ street-view only \textbf{$\cdot$} full$-$best1 $=$ full $-\max(\text{sat},\text{sv})$ \textbf{$\cdot$} thinking off")
    P(r"\end{frame}")

    P(r"\end{document}")

    outdir = RES / "slides_urban"
    outdir.mkdir(exist_ok=True)
    (outdir / "urban_views.tex").write_text("\n".join(o))
    print("wrote", outdir / "urban_views.tex")


if __name__ == "__main__":
    main()
