#!/usr/bin/env python3
"""
Generate the main-paper figures for "Single View Might Be Enough".

All numbers are verified from the per-item prediction logs (trained Qwen3.5-9B,
per-city split, 8 attribute tasks, n=2932). Cross-view control values are the
per-task synergies from the held-out benchmark. Teaser imagery is unmodified
dataset content (one real held-out Sydney location).

Outputs vector PDFs to GAIA_ECCVW_2026_Paper_Template/figures_new/:
  fig_synergy_control.pdf  - per-task synergy + CIs + cross-view controls (one axis)
  fig_oracle_dumbbell.pdf  - full vs per-item single-view oracle, per task
  fig_teaser.pdf           - one real location: sat + 4 SV + a real MCQ + ablation bars

Design: colorblind-safe (Okabe-Ito), thin gray axes, error bars where we have CIs,
n shown per task, no decorative imagery. Run with the repo's python (matplotlib+PIL).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_new")
os.makedirs(OUT, exist_ok=True)
DATA_IMG = "/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final/benchmark/images"

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#999999"

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,      # embed TrueType (no Type-3), reviewer-friendly
    "ps.fonttype": 42,
})

# ----------------------------------------------------------------------------
# VERIFIED DATA (trained Qwen3.5-9B, per-city, 8 attribute tasks, n shown)
# per-task: (name, n, synergy, ci_lo, ci_hi, vision_contrib, full, oracle)
ATTR = [
    ("transit_density", 421, +4.5, +0.0, +7.8, 21.9, 50.1, 63.4),
    ("urban_density",   421, +2.4, -2.4, +5.0, 10.5, 65.1, 79.8),
    ("amenity_richness",421, +2.1, -1.7, +5.5, 32.3, 58.7, 70.1),
    ("junction_type",   334, +1.8, -1.5, +5.1, 13.8, 75.1, 82.9),
    ("road_surface",    303, +0.7, -2.0, +2.6,  0.0, 95.7, 98.0),
    ("land_use",        421, -0.2, -2.9, +1.4,  6.7, 75.3, 81.9),
    ("building_height", 190, -0.5, -5.3, +2.1, 21.6, 66.8, 77.4),
    ("road_type",       421, -1.0, -3.8, +1.9,  7.8, 76.5, 82.4),
]
# pooled synergy (item-weighted) and oracle gap, per-city
POOL_SYN, POOL_SYN_LO, POOL_SYN_HI = 2.2, 0.9, 3.3

# cross-view CONTROL tasks: (name, synergy)  [point estimates, held-out benchmark]
CONTROL = [
    ("camera_direction",       17.3),
    ("mismatch_binary_hard",   44.9),
    ("mismatch_binary_easy",   44.7),
    ("mismatch_mcq_hard",      62.7),
    ("mismatch_mcq_easy",      69.8),
]


def nice_name(s):
    return s.replace("_", "\\_") if False else s.replace("_", " ")


# ============================================================================
# FIG 1 — synergy caterpillar + positive control on one axis
# ============================================================================
def fig_synergy_control():
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # layout rows: controls on top (high synergy), a gap, then attribute tasks
    attr = ATTR
    ctrl = CONTROL
    y_attr = list(range(len(attr)))
    gap = 1.0
    y_ctrl = [len(attr) + gap + i for i in range(len(ctrl))]

    # equivalence band & zero line
    ax.axvspan(-3, 3, color="#EEEEEE", zorder=0, label="_nolegend_")
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)

    # attribute tasks with CIs
    for i, (name, n, syn, lo, hi, *_rest) in enumerate(attr):
        ax.plot([lo, hi], [i, i], color=GREY, lw=1.1, zorder=2)
        ax.plot([syn], [i], "o", color=BLUE, ms=4.5, zorder=3)
    # control tasks (point estimates, square markers)
    for j, (name, syn) in enumerate(ctrl):
        ax.plot([syn], [y_ctrl[j]], "s", color=VERM, ms=4.5, zorder=3)

    # y labels with n for attributes
    yticks = y_attr + y_ctrl
    ylabels = [f"{nice_name(a[0])}  (n={a[1]})" for a in attr] + \
              [nice_name(c[0]) for c in ctrl]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    # separating annotation
    ax.text(72, (len(attr) - 0.5 + y_ctrl[0]) / 2, "", va="center")
    ax.annotate("attribute tasks", xy=(71, (len(attr) - 1) / 2.0),
                fontsize=7.5, color=BLUE, rotation=90, va="center", ha="left")
    ax.annotate("cross-view\ncontrols", xy=(71, sum(y_ctrl) / len(y_ctrl)),
                fontsize=7.5, color=VERM, rotation=90, va="center", ha="left")

    ax.set_xlim(-8, 78)
    ax.set_xlabel(r"Fusion synergy:  full $-$ max(sat, sv)   (accuracy points)")
    ax.set_ylim(-0.7, y_ctrl[-1] + 0.7)

    leg = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markersize=6, label="attribute task (95% CI)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=VERM,
                   markersize=6, label="cross-view control (point est.)"),
        Patch(facecolor="#EEEEEE", edgecolor="none", label=r"$\pm$3 pt equiv. band"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=6.6, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_synergy_control.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG 2 — full vs per-item single-view oracle (dumbbell)
# ============================================================================
def fig_oracle_dumbbell():
    # sort by gap (oracle - full) descending so biggest waste on top
    rows = sorted(ATTR, key=lambda r: (r[7] - r[6]), reverse=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ys = list(range(len(rows)))
    for i, r in enumerate(rows):
        full, orac = r[6], r[7]
        ax.plot([full, orac], [i, i], color=GREY, lw=1.6, zorder=1)
        ax.annotate(f"$-${orac-full:.1f}", xy=((full+orac)/2, i), fontsize=6,
                    color="#555555", ha="center", va="bottom", xytext=(0, 2.5),
                    textcoords="offset points")
    ax.scatter([r[6] for r in rows], ys, facecolors="white", edgecolors=BLUE,
               linewidths=1.3, s=42, zorder=2, label="full (both views)")
    ax.scatter([r[7] for r in rows], ys, color=VERM, s=42, zorder=2,
               label="per-item single-view oracle")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{nice_name(r[0])}  (n={r[1]})" for r in rows])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(45, 108)
    ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False,
              bbox_to_anchor=(1.0, 0.52))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # pooled annotation (above the plot to avoid the x-axis)
    ax.set_title("pooled: full 69.5% vs. oracle 78.8% "
                 "(gap $-$9.3, 95% CI [$-$10.4, $-$8.2])",
                 fontsize=6.6, color="#333333", pad=4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_oracle_dumbbell.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG 3 — teaser: one real Sydney location + a real MCQ + ablation bars
# ============================================================================
def fig_teaser():
    sid = "sydney_0002"
    sat = Image.open(f"{DATA_IMG}/sat/{sid}.png").convert("RGB")
    sv_files = ["along_fwd", "cross_right", "along_bwd", "cross_left"]
    sv_labels = ["forward", "right", "backward", "left"]
    svs = [Image.open(f"{DATA_IMG}/sv/{sid}_{a}.jpg").convert("RGB") for a in sv_files]

    fig = plt.figure(figsize=(6.9, 3.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.35, 1.0],
                          width_ratios=[1, 1, 1, 1, 1],
                          hspace=0.30, wspace=0.10)

    # top row: satellite (col 0) + 4 street views (cols 1-4)
    ax_sat = fig.add_subplot(gs[0, 0])
    ax_sat.imshow(sat)
    ax_sat.set_title("Satellite (ESRI $\\sim$0.4 m/px)", fontsize=7)
    ax_sat.set_xticks([]); ax_sat.set_yticks([])
    for s in ax_sat.spines.values():
        s.set_edgecolor(BLUE); s.set_linewidth(1.2)

    for k, (img, lab) in enumerate(zip(svs, sv_labels)):
        axk = fig.add_subplot(gs[0, k + 1])
        axk.imshow(img)
        axk.set_title(f"street: {lab}", fontsize=7)
        axk.set_xticks([]); axk.set_yticks([])
        for s in axk.spines.values():
            s.set_edgecolor(GREEN); s.set_linewidth(1.0)

    # bottom-left (cols 0-2): the real question
    ax_q = fig.add_subplot(gs[1, 0:3])
    ax_q.axis("off")
    qtext = ("Q  (urban density):  What level of building concentration\n"
             "is visible in this area?\n\n"
             "   A.  Low density (suburban / rural)        [gold]\n"
             "   B.  High density        C.  Moderate density\n"
             "   D.  Very high density")
    ax_q.text(0.0, 0.92, qtext, fontsize=7.0, va="top", family="monospace")

    # bottom-right (cols 3-4): the ablation bars
    ax_b = fig.add_subplot(gs[1, 3:5])
    conds = ["blind", "sat", "sv", "full"]
    vals = [54.6, 61.8, 62.7, 65.1]   # urban_density, per-city, real
    colors = [GREY, BLUE, GREEN, VERM]
    bars = ax_b.bar(conds, vals, color=colors, width=0.68)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("accuracy (%)", fontsize=7)
    ax_b.tick_params(labelsize=7)
    for b, v in zip(bars, vals):
        ax_b.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}",
                  ha="center", fontsize=6.6)
    ax_b.set_title("sat $\\approx$ sv $\\approx$ full $\\gg$ blind", fontsize=7)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    fig.suptitle("One held-out location (Sydney): each single view already "
                 "answers the question; the second view adds little.",
                 fontsize=8.0, y=0.99)
    p = os.path.join(OUT, "fig_teaser.pdf")
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_synergy_control()
    fig_oracle_dumbbell()
    fig_teaser()
    print("done ->", OUT)
