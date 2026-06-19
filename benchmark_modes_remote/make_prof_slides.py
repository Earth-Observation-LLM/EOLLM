#!/usr/bin/env python3
"""
Professor-facing deck. Two stories, each with an OVERALL anchor + a per-task breakdown
of the 9 urbanization tasks. Numbers carry it.

  Story 1  Qwen3.5-9B: does adding reasoning help fuse the two views?
           current / think16k / multistep  x  full / sat / sv / blind.
           Overall first, then per-task. Point: sv >= full everywhere -> nothing works.

  Story 2  The full models on the benchmark (full mode): vision helps, but it is
           one-view vision. Overall full vs blind per model, then per-task full accuracy.

Reads:
  reasoning_distill_remote/results/attr_by_task.json   (strategy run; current/think16k/multistep)
  benchmark_modes_remote/results/<model>/<model>_{full,blind}_report.json

Writes:
  benchmark_modes_remote/results/slides_prof/prof_views.tex (+ .pdf via tectonic)
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
STRAT = HERE.parent / "reasoning_distill_remote" / "results" / "attr_by_task.json"

TASKS = ["land_use", "building_height", "urban_density", "junction_type", "green_space",
         "amenity_richness", "road_type", "road_surface", "transit_density"]
NICE = {
    "land_use": "Land use", "building_height": "Building height",
    "urban_density": "Urban density", "junction_type": "Junction type",
    "green_space": "Green space", "amenity_richness": "Amenity richness",
    "road_type": "Road type", "road_surface": "Road surface",
    "transit_density": "Transit density",
}
MODELS = ["gemma-4-12b-awq", "qwen2.5-vl-7b-awq", "qwen3.5-4b-awq", "qwen3.5-9b-awq"]
MNICE = {"gemma-4-12b-awq": "Gemma-4-12B", "qwen2.5-vl-7b-awq": "Qwen2.5-VL-7B",
         "qwen3.5-4b-awq": "Qwen3.5-4B", "qwen3.5-9b-awq": "Qwen3.5-9B"}
STRATS = ["current", "think16k", "multistep"]
SNICE = {"current": "current (direct)", "think16k": "think16k", "multistep": "multistep (4-turn)"}


def overall_strat(acc, s):
    tot = {k: 0.0 for k in ["full", "sat_only", "sv_only", "blind"]}
    N = 0
    for t in TASKS:
        n = acc[s][t]["n"]
        N += n
        for k in tot:
            tot[k] += acc[s][t][k] / 100 * n
    return {k: round(100 * tot[k] / N, 1) for k in tot}


def cdelta(x):
    if x >= 3.0:
        return rf"\textcolor{{good}}{{{x:+.1f}}}"
    if x <= -1.0:
        return rf"\textcolor{{bad}}{{{x:+.1f}}}"
    return f"{x:+.1f}"


def main():
    ds = json.load(open(STRAT))["acc"]
    full = {m: json.load(open(RES / m / f"{m}_full_report.json"))["by_topic"] for m in MODELS}
    blind = {m: json.load(open(RES / m / f"{m}_blind_report.json"))["by_topic"] for m in MODELS}
    full_ov = {m: round(100 * json.load(open(RES / m / f"{m}_full_report.json"))["overall"]["accuracy"], 1) for m in MODELS}
    blind_ov = {m: round(100 * json.load(open(RES / m / f"{m}_blind_report.json"))["overall"]["accuracy"], 1) for m in MODELS}

    o = []
    P = o.append
    P(r"\documentclass[aspectratio=169]{beamer}")
    P(r"\usetheme{metropolis}")
    P(r"\usepackage{booktabs}\usepackage{xcolor}\usepackage{array}")
    P(r"\definecolor{good}{HTML}{2E7D32}\definecolor{bad}{HTML}{C62828}\definecolor{accent}{HTML}{1565C0}")
    P(r"\setbeamercolor{frametitle}{bg=accent}")
    P(r"\setbeamertemplate{section page}{\centering\usebeamerfont{section title}\insertsectionhead\par}")
    P(r"\title{Do these tasks need both views?}")
    P(r"\subtitle{Single view is enough \textbf{$\cdot$} extra reasoning does not change it}")
    P(r"\date{}\author{}")
    P(r"\begin{document}")
    P(r"\maketitle")

    # ===================== STORY 1
    P(r"\section{Qwen3.5-9B \textemdash{} does reasoning help fuse the two views?}")

    # 1a overall
    P(r"\begin{frame}{Overall \textbf{$\cdot$} 9 urbanization tasks \textbf{$\cdot$} Qwen3.5-9B}")
    P(r"\centering")
    P(r"\begin{tabular}{lcccc>{\bfseries}cc}")
    P(r"\toprule")
    P(r"strategy & full & sat & sv & blind & full$-$best1 & full$-$blind \\")
    P(r"\midrule")
    for s in STRATS:
        ov = overall_strat(ds, s)
        syn = round(ov["full"] - max(ov["sat_only"], ov["sv_only"]), 1)
        vis = round(ov["full"] - ov["blind"], 1)
        P(rf"{SNICE[s]} & {ov['full']:.1f} & {ov['sat_only']:.1f} & "
          rf"\textbf{{{ov['sv_only']:.1f}}} & {ov['blind']:.1f} & {cdelta(syn)} & {vis:+.1f} \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vfill\footnotesize")
    P(r"\textbf{full$-$best1} $=$ full $-\max(\text{sat},\text{sv})$. Negative $\Rightarrow$ the second view does \emph{not} help.\\")
    P(r"Street-view alone (sv) $\geq$ both views (full) for \emph{every} strategy --- thinking and multi-step")
    P(r"raise the ceiling but lift the single view in lockstep, so fusion never starts working.")
    P(r"\end{frame}")

    # 1b per-task, one frame per strategy
    for s in STRATS:
        P(rf"\begin{{frame}}{{Per task \textbf{{$\cdot$}} {SNICE[s]} \textbf{{$\cdot$}} Qwen3.5-9B}}")
        P(r"\centering\scriptsize")
        P(r"\begin{tabular}{lcccc>{\bfseries}c}")
        P(r"\toprule")
        P(r"task & full & sat & sv & blind & full$-$best1 \\")
        P(r"\midrule")
        for t in TASKS:
            a = ds[s][t]
            syn = round(a["full"] - max(a["sat_only"], a["sv_only"]), 1)
            P(rf"{NICE[t]} & {a['full']:.1f} & {a['sat_only']:.1f} & {a['sv_only']:.1f} & "
              rf"{a['blind']:.1f} & {cdelta(syn)} \\")
        P(r"\midrule")
        ov = overall_strat(ds, s)
        syn = round(ov["full"] - max(ov["sat_only"], ov["sv_only"]), 1)
        P(r"\textbf{OVERALL} & " + rf"\textbf{{{ov['full']:.1f}}} & \textbf{{{ov['sat_only']:.1f}}} & "
          rf"\textbf{{{ov['sv_only']:.1f}}} & \textbf{{{ov['blind']:.1f}}} & \textbf{{{cdelta(syn)}}} \\")
        P(r"\bottomrule")
        P(r"\end{tabular}")
        P(r"\end{frame}")

    # ===================== STORY 2
    P(r"\section{The full models on the benchmark}")

    # 2a overall
    P(r"\begin{frame}{Overall \textbf{$\cdot$} full mode \textbf{$\cdot$} 4 models}")
    P(r"\centering")
    P(r"\begin{tabular}{lcc>{\bfseries}c}")
    P(r"\toprule")
    P(r"model & full & blind & vision gain \\")
    P(r"\midrule")
    for m in MODELS:
        P(rf"{MNICE[m]} & {full_ov[m]:.1f} & {blind_ov[m]:.1f} & {cdelta(round(full_ov[m]-blind_ov[m],1))} \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vfill\footnotesize vision gain $=$ full $-$ blind. Vision genuinely helps --- but (Story 1) it is \emph{one-view} vision.")
    P(r"\end{frame}")

    # 2b per-task full accuracy across models
    P(r"\begin{frame}{Per task \textbf{$\cdot$} full-mode accuracy \textbf{$\cdot$} 4 models}")
    P(r"\centering\scriptsize")
    P(r"\begin{tabular}{l" + "c" * len(MODELS) + "}")
    P(r"\toprule")
    P(r"task & " + " & ".join(MNICE[m] for m in MODELS) + r" \\")
    P(r"\midrule")
    for t in TASKS:
        P(rf"{NICE[t]} & " + " & ".join(f"{round(100*full[m][t]['accuracy'],1):.1f}" for m in MODELS) + r" \\")
    P(r"\midrule")
    P(r"\textbf{OVERALL} & " + " & ".join(rf"\textbf{{{full_ov[m]:.1f}}}" for m in MODELS) + r" \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vfill\footnotesize OVERALL is over the whole benchmark (all topics); per-task rows are the 9 urbanization tasks.")
    P(r"\end{frame}")

    # takeaway
    P(r"\begin{frame}{Takeaway}")
    P(r"\begin{itemize}")
    P(r"  \item For the 9 urbanization tasks, a \textbf{single view is enough}: street-view alone matches or")
    P(r"        beats the two-view setup (full$-$best1 $\leq 0$) for every strategy and every task.")
    P(r"  \item \textbf{Adding reasoning does not fix it}: 16k-token thinking and a 4-turn observe$\to$reason$\to$")
    P(r"        self-check$\to$commit conversation raise accuracy, but the single view rises with it.")
    P(r"  \item Across all 4 models vision \emph{does} help (full $>$ blind) --- but it is one-view vision.")
    P(r"\end{itemize}")
    P(r"\end{frame}")

    P(r"\end{document}")

    txt = "\n".join(o)
    outdir = RES / "slides_prof"
    outdir.mkdir(exist_ok=True)
    (outdir / "prof_views.tex").write_text(txt)
    print("wrote", outdir / "prof_views.tex")


if __name__ == "__main__":
    main()
