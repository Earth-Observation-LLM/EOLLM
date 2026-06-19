#!/usr/bin/env python3
"""
Per-TASK strategy ablation report, scoped to the 9 urban-attribute (urbanization) tasks.

Why per-task and attr-only: the overall accuracy blends two task families. The cross-view
tasks (mismatch_*, camera_direction) put the answer INSIDE the pair of images, so a
single-view ablation there is ill-posed — withholding one view makes the question
unanswerable by construction. The 9 attribute tasks are the ones where "do you actually
need both views?" is a fair, well-posed question. This report answers exactly that, task
by task.

Reads results/attr_by_task.json + results/attr_overall.json (produced inline by the
monitoring run) and emits:
  - results/pertask_report.md          human-readable per-task tables + verdicts
  - results/slides/strategy_ablation.tex   Beamer deck (compile with tectonic/pdflatex)
"""
import json, os, pathlib

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"

NICE = {
    "land_use": "Land use",
    "building_height": "Building height",
    "urban_density": "Urban density",
    "junction_type": "Junction type",
    "green_space": "Green space",
    "amenity_richness": "Amenity richness",
    "road_type": "Road type",
    "road_surface": "Road surface",
    "transit_density": "Transit density",
}
SNICE = {"current": "current", "think16k": "think16k", "multistep": "multistep"}


def load():
    d = json.load(open(RES / "attr_by_task.json"))
    ov = json.load(open(RES / "attr_overall.json"))
    return d, ov


def synergy(a):  # full - best single view
    return round(a["full"] - max(a["sat_only"], a["sv_only"]), 1)


def vision(a):  # full - blind
    return round(a["full"] - a["blind"], 1)


def needs_both(a):
    """Well-posed 'needs both views' test: full must clearly beat the best single view
    AND blind must be well below full (vision matters at all). We use +3.0 abs pts on
    synergy as the bar — generous; nothing in this run clears even 0."""
    return "YES" if (synergy(a) >= 3.0 and vision(a) >= 5.0) else "NO"


# ---------------------------------------------------------------- markdown
def md(d, ov):
    S, V, T = d["strategies"], d["views"], d["tasks"]
    acc = d["acc"]
    L = []
    L.append("# Per-task strategy ablation — the 9 urbanization tasks\n")
    L.append("_Qwen3.5-9B-AWQ · greedy (temp 0) · lab-ws SLURM job 216 · 0 errors._\n")
    L.append(
        "**Scope.** Only the 9 urban-attribute tasks. The cross-view tasks "
        "(`mismatch_*`, `camera_direction`) are excluded on purpose: their answer lives "
        "in *having both images*, so a single-view ablation is ill-posed there. These 9 "
        "are where the question *“do we actually need both views?”* is fair.\n"
    )
    L.append(
        "`full` = marked-satellite + 4 street-view angles · `sat` = satellite only · "
        "`sv` = street-view grid only · `blind` = no images. "
        "**synergy** = full − max(sat, sv) — positive ⇒ the second view adds something a "
        "single view can't. **vision** = full − blind.\n"
    )

    # overall
    L.append("## Overall (attribute tasks only)\n")
    L.append("| strategy | full | sat | sv | blind | synergy | vision |")
    L.append("|---|---|---|---|---|---|---|")
    for s in S:
        o = ov[s]
        L.append(
            f"| {SNICE[s]} | {o['full']:.1f} | {o['sat_only']:.1f} | {o['sv_only']:.1f} | "
            f"{o['blind']:.1f} | **{synergy(o):+.1f}** | {vision(o):+.1f} |"
        )
    L.append("")

    # per task — all 3 strategies x 4 views
    L.append("## Per task — all 3 strategies × 4 views\n")
    for t in T:
        n = acc[S[0]][t]["n"]
        L.append(f"### {NICE[t]}  _(n={n}/view)_\n")
        L.append("| strategy | full | sat | sv | blind | synergy | needs both? |")
        L.append("|---|---|---|---|---|---|---|")
        for s in S:
            a = acc[s][t]
            L.append(
                f"| {SNICE[s]} | {a['full']:.1f} | {a['sat_only']:.1f} | {a['sv_only']:.1f} | "
                f"{a['blind']:.1f} | {synergy(a):+.1f} | {needs_both(a)} |"
            )
        L.append("")

    # verdict
    L.append("## Verdict\n")
    L.append(
        "Across **all 9 tasks and all 3 strategies**, `full` never clears the best single "
        "view by a meaningful margin — synergy is **≤ 0 in every overall cell** and the "
        "*needs-both* test is **NO for every task**. street-view alone (`sv`) matches or "
        "beats the two-view `full` setup throughout. Vision still matters (full − blind "
        "≈ +8 overall), but it is **one-view vision**: the satellite image is redundant "
        "given the street-view, and no prompting strategy — 16k of explicit thinking, nor "
        "a 4-turn observe→reason→self-check→commit conversation — recovers any "
        "complementary signal from it.\n"
    )
    (RES / "pertask_report.md").write_text("\n".join(L))
    print("wrote", RES / "pertask_report.md")


# ---------------------------------------------------------------- beamer
def tex_esc(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def beamer(d, ov):
    S, V, T = d["strategies"], d["views"], d["tasks"]
    acc = d["acc"]

    def color_syn(x):
        # red for <=0, green for clearly positive; this run is all red.
        if x >= 3.0:
            return rf"\textcolor{{good}}{{{x:+.1f}}}"
        if x <= -1.0:
            return rf"\textcolor{{bad}}{{{x:+.1f}}}"
        return rf"{x:+.1f}"

    out = []
    P = out.append
    P(r"\documentclass[aspectratio=169]{beamer}")
    P(r"\usetheme{metropolis}")
    P(r"\usepackage{booktabs}")
    P(r"\usepackage{xcolor}")
    P(r"\usepackage{array}")
    P(r"\usepackage{multirow}")
    P(r"\definecolor{good}{HTML}{2E7D32}")
    P(r"\definecolor{bad}{HTML}{C62828}")
    P(r"\definecolor{accent}{HTML}{1565C0}")
    P(r"\setbeamercolor{frametitle}{bg=accent}")
    P(r"\newcommand{\hl}[1]{\textbf{\textcolor{accent}{#1}}}")
    P(r"\title{Do we need both views?}")
    P(r"\subtitle{A prompting-strategy ablation on urban-attribute VQA}")
    P(r"\author{Qwen3.5-9B-AWQ \;$\cdot$\; 9 urbanization tasks \;$\cdot$\; greedy decoding}")
    P(r"\date{}")
    P(r"\begin{document}")
    P(r"\maketitle")

    # --- the question
    P(r"\begin{frame}{The question}")
    P(r"\begin{itemize}")
    P(r"  \item Each sample pairs a \hl{marked satellite} image with a \hl{4-angle street-view} grid.")
    P(r"  \item For the \hl{9 urban-attribute tasks}, we ask: does the model actually \emph{need both views}?")
    P(r"  \item We compare four inputs --- \texttt{full} (both), \texttt{sat} only, \texttt{sv} only, \texttt{blind} (none) ---")
    P(r"        under \hl{three prompting strategies}, on the same records, greedy decoding.")
    P(r"\end{itemize}")
    P(r"\vfill")
    P(r"\footnotesize Cross-view tasks (\texttt{mismatch\_*}, \texttt{camera\_direction}) are excluded:")
    P(r"their answer lives in \emph{having both images}, so a single-view ablation is ill-posed there.")
    P(r"\end{frame}")

    # --- strategies
    P(r"\begin{frame}{Three strategies, same model}")
    P(r"\begin{description}")
    P(r"  \item[current] thinking off, one turn, ask directly. The baseline.")
    P(r"  \item[think16k] thinking \emph{on}, up to a 16k-token \texttt{<think>} budget, one turn.")
    P(r"  \item[multistep] thinking off, a 4-turn conversation: \emph{observe} each image $\to$")
    P(r"        \emph{reason} without options $\to$ \emph{self-check} $\to$ \emph{commit}. Turns uncapped.")
    P(r"\end{description}")
    P(r"\vfill")
    P(r"\centering\footnotesize \textbf{synergy} $=$ full $-\max(\text{sat},\text{sv})$ \quad")
    P(r"\textbf{vision} $=$ full $-$ blind")
    P(r"\end{frame}")

    # --- headline overall
    P(r"\begin{frame}{Headline: the second view earns nothing}")
    P(r"\centering")
    P(r"\begin{tabular}{lcccc>{\bfseries}cc}")
    P(r"\toprule")
    P(r"strategy & full & sat & sv & blind & synergy & vision \\")
    P(r"\midrule")
    for s in S:
        o = ov[s]
        P(rf"{SNICE[s]} & {o['full']:.1f} & {o['sat_only']:.1f} & {o['sv_only']:.1f} & "
          rf"{o['blind']:.1f} & {color_syn(synergy(o))} & {vision(o):+.1f} \\")
    P(r"\bottomrule")
    P(r"\end{tabular}")
    P(r"\vfill")
    P(r"\begin{itemize}")
    P(r"  \item \hl{sv-only $\geq$ full} for every strategy --- synergy is \textcolor{bad}{negative} throughout.")
    P(r"  \item Vision helps ($\approx +8$ over blind), but it is \hl{one-view vision}: street-view alone.")
    P(r"\end{itemize}")
    P(r"\end{frame}")

    # --- per-task frames (3 tasks per frame for readability)
    chunks = [T[i:i + 3] for i in range(0, len(T), 3)]
    for ci, ch in enumerate(chunks, 1):
        P(rf"\begin{{frame}}{{Per task --- all strategies $\times$ all views ({ci}/{len(chunks)})}}")
        P(r"\centering\scriptsize")
        P(r"\begin{tabular}{llcccc>{\bfseries}c}")
        P(r"\toprule")
        P(r"task & strat & full & sat & sv & blind & syn \\")
        P(r"\midrule")
        for t in ch:
            n = acc[S[0]][t]["n"]
            for j, s in enumerate(S):
                a = acc[s][t]
                # task label on the middle row of the 3-strategy block
                tcell = rf"\multirow{{3}}{{*}}{{{NICE[t]}}}" if j == 1 else ""
                P(rf"{tcell} & {SNICE[s]} & {a['full']:.1f} & {a['sat_only']:.1f} & "
                  rf"{a['sv_only']:.1f} & {a['blind']:.1f} & {color_syn(synergy(a))} \\")
            P(r"\midrule")
        # replace trailing midrule with bottomrule
        out[-1] = r"\bottomrule"
        P(r"\end{tabular}")
        P(r"\end{frame}")

    # --- takeaway
    P(r"\begin{frame}{Takeaway}")
    P(r"\begin{itemize}")
    P(r"  \item On all \hl{9 urbanization tasks}, across all \hl{3 strategies}, \texttt{full}")
    P(r"        never beats the best single view by a meaningful margin.")
    P(r"  \item The \hl{satellite view is redundant} given the street-view: \texttt{sv}-only")
    P(r"        matches or beats the two-view setup everywhere.")
    P(r"  \item Better prompting raises the ceiling (multistep $+3.1$ over baseline \texttt{full}),")
    P(r"        but it lifts \texttt{sv}-only \emph{in lockstep} --- the gap never closes.")
    P(r"  \item \hl{Conclusion:} for these tasks we do \emph{not} need both views. One well-chosen")
    P(r"        view (street-level) carries the signal; the second image does not earn its cost.")
    P(r"\end{itemize}")
    P(r"\end{frame}")

    P(r"\end{document}")

    slides = RES / "slides"
    slides.mkdir(exist_ok=True)
    (slides / "strategy_ablation.tex").write_text("\n".join(out))
    print("wrote", slides / "strategy_ablation.tex")


if __name__ == "__main__":
    d, ov = load()
    md(d, ov)
    beamer(d, ov)
