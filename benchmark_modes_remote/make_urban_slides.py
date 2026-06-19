#!/usr/bin/env python3
"""
Beamer deck — which tasks need both views, which don't. Minimal text (numbers carry it).

  Section 1  URBANIZATION tasks (4-model sweep): per task, full vs blind across
             gemma-4-12b / qwen2.5-vl-7b / qwen3.5-4b / qwen3.5-9b.
             (This sweep only asked the 9 attribute tasks in full + blind.)

  Section 2  URBANIZATION tasks (Qwen3.5-9B strategy run): per task, full/sat/sv/blind.
             The run that DID ask attribute tasks in all four views, so it directly
             shows full ~ sv ~ sat — the second view is redundant.

  Section 3  CROSS-VIEW tasks (4-model sweep, the ONLY run with them): per task,
             full/sat/sv/blind + full-best1, across all 4 models. Here full crushes any
             single view (mismatch) — these tasks genuinely need both. camera_direction
             sits near chance (kept in-table, unframed).

  Section 4  Contrast: full-best1 by task family — positive (need both) vs <=0 (redundant).

Reads:
  benchmark_modes_remote/results/attr_fullblind_4model.json
  benchmark_modes_remote/results/crossview_4model.json
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

CV = ["mismatch_binary_easy", "mismatch_binary_hard", "mismatch_mcq_easy",
      "mismatch_mcq_hard", "camera_direction"]
CVNICE = {
    "mismatch_binary_easy": "Mismatch binary (easy)",
    "mismatch_binary_hard": "Mismatch binary (hard)",
    "mismatch_mcq_easy": "Mismatch MCQ (easy)",
    "mismatch_mcq_hard": "Mismatch MCQ (hard)",
    "camera_direction": "Camera direction",
}


def fmt(x):
    return "--" if x is None else f"{x:.1f}"


def best1(a):  # full - max(sat, sv), skipping missing views
    singles = [v for v in (a["sat_only"], a["sv_only"]) if v is not None]
    if not singles or a["full"] is None:
        return None
    return round(a["full"] - max(singles), 1)


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
    cv = json.load(open(RES / "crossview_4model.json"))
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

    # ---------------- Section 3: cross-view tasks (one frame per task, 4 models x 4 views)
    P(r"\section{Cross-view tasks \textemdash{} all four views}")
    cv_chunks = [CV[i:i + 3] for i in range(0, len(CV), 3)]
    for ci, ch in enumerate(cv_chunks, 1):
        P(rf"\begin{{frame}}{{Cross-view tasks \textbf{{$\cdot$}} full / sat / sv / blind ({ci}/{len(cv_chunks)})}}")
        P(r"\centering\scriptsize")
        P(r"\begin{tabular}{llcccc>{\bfseries}c}")
        P(r"\toprule")
        P(r"task & model & full & sat & sv & blind & full$-$best1 \\")
        P(r"\midrule")
        for t in ch:
            for j, m in enumerate(M):
                a = cv["acc"][m][t]
                tcell = rf"\multirow{{4}}{{*}}{{{CVNICE[t]}}}" if j == 0 else ""
                b1 = best1(a)
                bcell = cdelta(b1) if b1 is not None else "--"
                P(rf"{tcell} & {MNICE[m]} & {fmt(a['full'])} & {fmt(a['sat_only'])} & "
                  rf"{fmt(a['sv_only'])} & {fmt(a['blind'])} & {bcell} \\")
            P(r"\midrule")
        o[-1] = r"\bottomrule"
        P(r"\end{tabular}")
        P(r"\vspace{2pt}\par\tiny full$-$best1 $=$ full $-\max(\text{sat},\text{sv})$ \textbf{$\cdot$} camera\_direction has no sv\_only (sv $=$ --)")
        P(r"\end{frame}")

    # ---------------- Section 4: the contrast — full-best1 by family (Qwen3.5-9B)
    P(r"\section{Which tasks need both views?}")
    P(r"\begin{frame}{full $-$ best single view \textbf{$\cdot$} Qwen3.5-9B \textbf{$\cdot$} by task family}")
    P(r"\centering\scriptsize")
    P(r"\begin{tabular}{ll>{\bfseries}cl}")
    P(r"\toprule")
    P(r"family & task & full$-$best1 & reading \\")
    P(r"\midrule")
    # cross-view (need both) — from 4-model sweep, qwen3.5-9b
    qm = "qwen3.5-9b-awq"
    for t in CV:
        b1 = best1(cv["acc"][qm][t])
        rd = r"\textcolor{good}{needs both}" if (b1 is not None and b1 >= 3.0) else (
            r"\textcolor{bad}{near chance}" if t == "camera_direction" else "weak")
        P(rf"cross-view & {CVNICE[t]} & {cdelta(b1) if b1 is not None else '--'} & {rd} \\")
    P(r"\midrule")
    # urbanization (redundant) — from strategy run, current
    for t in TASKS:
        a = accs[t]
        b1 = round(a["full"] - max(a["sat_only"], a["sv_only"]), 1)
        rd = r"\textcolor{bad}{2nd view redundant}" if b1 <= 0 else "marginal"
        P(rf"urbanization & {NICE[t]} & {cdelta(b1)} & {rd} \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vspace{2pt}\par\tiny cross-view from 4-model sweep (Qwen3.5-9B) \textbf{$\cdot$} urbanization from strategy run (Qwen3.5-9B, thinking off)")
    P(r"\end{frame}")

    P(r"\end{document}")

    outdir = RES / "slides_urban"
    outdir.mkdir(exist_ok=True)
    (outdir / "urban_views.tex").write_text("\n".join(o))
    print("wrote", outdir / "urban_views.tex")


if __name__ == "__main__":
    main()
