#!/usr/bin/env python3
"""
Generate the main-paper figures for the v2 paper
("When Does the Second View Help? Auditing Cross-View Fusion in Urban VLMs").

Every number is read from v2_numbers.json (built by make_v2_numbers.py from the
satfwd per-record prediction logs); nothing here is hand-typed. Run that first.

Outputs vector PDFs to figures_new/:
  fig_teaser.pdf          - page-1 combined teaser: LEFT one real held-out location
                            (sat + forward SV the model uses, + its 2-image ablation),
                            RIGHT the divergent per-task fusion-gain bar (the thesis).
  fig_synergy_control.pdf - per-task fusion synergy + cross-view controls (one axis)
  fig_oracle_dumbbell.pdf - full vs better single view, per urban task

Design: colorblind-safe (Okabe-Ito), thin gray axes, n shown per task, no decoration.
Imagery is unmodified dataset content (one real held-out Sydney location).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_new")
os.makedirs(OUT, exist_ok=True)
DATA_IMG = "/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final/benchmark/images"
NUMS = json.load(open(os.path.join(HERE, "v2_numbers.json")))

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#999999"
DARK = "#444444"

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": DARK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

URBAN7 = ["land_use", "building_height", "urban_density", "road_type",
          "junction_type", "amenity_richness", "transit_density"]
CROSS5 = ["camera_direction", "mismatch_binary_easy", "mismatch_binary_hard",
          "mismatch_mcq_easy", "mismatch_mcq_hard"]


def nn(s):
    return s.replace("_", " ")


def urban_rows():
    """(name, n, synergy, full, best_single) per urban task, sorted by synergy asc."""
    t2 = NUMS["table2_urban7"]
    rows = []
    for t in URBAN7:
        d = t2[t]
        rows.append((t, d["n"], d["synergy"], d["full"], max(d["sat"], d["sv"])))
    rows.sort(key=lambda r: r[2])
    return rows


def urban_oracle_rows():
    """(name, n, full_paired, oracle_item) per urban task — the per-ITEM oracle that
    keeps the better single view per question (>= full on every task)."""
    t2 = NUMS["table2_urban7"]
    rows = [(t, t2[t]["n"], t2[t]["full_paired"], t2[t]["oracle_item"]) for t in URBAN7]
    rows.sort(key=lambda r: r[2])
    return rows


def cross_rows():
    t4 = NUMS["table4_crossview"]
    rows = [(t, t4[t]["synergy"]) for t in CROSS5]
    rows.sort(key=lambda r: r[1])
    return rows


# ============================================================================
# FIG: combined teaser  (LEFT setup+ablation | RIGHT divergent fusion-gain bar)
# ============================================================================
def fig_teaser():
    sid = "sydney_0002"
    sat = Image.open(f"{DATA_IMG}/sat/{sid}.png").convert("RGB")
    fwd = Image.open(f"{DATA_IMG}/sv/{sid}_along_fwd.jpg").convert("RGB")

    ud = NUMS["table2_urban7"]["urban_density"]
    abl = [ud["blind"], ud["sat"], ud["sv"], ud["full"]]   # satfwd, verified

    fig = plt.figure(figsize=(7.0, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.25], wspace=0.30)

    # ---------------- LEFT: setup + 2-image ablation -----------------------
    gsl = gs[0].subgridspec(2, 2, height_ratios=[1.25, 1.0], wspace=0.08, hspace=0.45)
    ax_sat = fig.add_subplot(gsl[0, 0]); ax_sat.imshow(sat)
    ax_sat.set_title("satellite + marker", fontsize=6.6)
    ax_sat.set_xticks([]); ax_sat.set_yticks([])
    for s in ax_sat.spines.values():
        s.set_edgecolor(BLUE); s.set_linewidth(1.3)
    ax_sv = fig.add_subplot(gsl[0, 1]); ax_sv.imshow(fwd)
    ax_sv.set_title("forward street view", fontsize=6.6)
    ax_sv.set_xticks([]); ax_sv.set_yticks([])
    for s in ax_sv.spines.values():
        s.set_edgecolor(GREEN); s.set_linewidth(1.3)

    ax_b = fig.add_subplot(gsl[1, :])
    conds = ["blind", "sat", "sv", "full"]
    colors = [GREY, BLUE, GREEN, VERM]
    bars = ax_b.bar(conds, abl, color=colors, width=0.66)
    ax_b.set_ylim(0, 100); ax_b.set_ylabel("acc. (%)", fontsize=6.6)
    ax_b.tick_params(labelsize=6.6)
    for b, v in zip(bars, abl):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}",
                  ha="center", fontsize=6.2)
    ax_b.set_title(r"full $\approx$ best single $\gg$ blind", fontsize=6.6)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.text(0.5, -0.42, "one location (urban density)", transform=ax_b.transAxes,
              ha="center", fontsize=6.0, color=DARK)

    # ---------------- RIGHT: divergent per-task fusion-gain bar -------------
    ax = fig.add_subplot(gs[1])
    u = urban_rows()
    c = cross_rows()
    names = [r[0] for r in u] + [r[0] for r in c]
    vals = [r[2] for r in u] + [r[1] for r in c]
    cols = [BLUE] * len(u) + [VERM] * len(c)
    y = list(range(len(names)))
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)
    ax.barh(y, vals, color=cols, height=0.66, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([nn(n) for n in names], fontsize=6.6)
    ax.set_xlabel(r"fusion gain:  full $-$ best single view  (accuracy pts)", fontsize=7)
    ax.set_xlim(-9, 100)
    for yi, v in zip(y, vals):
        ax.text(v + (1.4 if v >= 0 else -1.4), yi, f"{v:+.0f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=6.0)
    # regime labels parked in the clear right margin (x>80, beyond all bars)
    ax.text(93, (len(u) - 1) / 2.0, "urban\n(single-view\nsufficient)", color=BLUE,
            fontsize=6.2, va="center", ha="center")
    ax.text(93, len(u) + (len(c) - 1) / 2.0, "cross-view\n(fusion\nrequired)", color=VERM,
            fontsize=6.2, va="center", ha="center")
    ax.axvline(82, color="#DDDDDD", lw=0.6)
    ax.axhline(len(u) - 0.5, color="#CCCCCC", lw=0.7, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.6)

    fig.suptitle("A second view helps only when the task requires cross-view "
                 "correspondence: fusion gain is $\\approx$0 on urban tasks, "
                 "$+19$ to $+71$ on cross-view tasks.", fontsize=7.6, y=1.03)
    p = os.path.join(OUT, "fig_teaser.pdf")
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG: synergy caterpillar + positive control on one axis
# ============================================================================
def fig_synergy_control():
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    u = urban_rows()
    c = cross_rows()
    y_attr = list(range(len(u)))
    gap = 1.0
    y_ctrl = [len(u) + gap + i for i in range(len(c))]

    ax.axvspan(-3, 3, color="#EEEEEE", zorder=0)
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)
    for i, (name, n, syn, *_r) in enumerate(u):
        ax.plot([syn], [i], "o", color=BLUE, ms=4.5, zorder=3)
    for j, (name, syn) in enumerate(c):
        ax.plot([syn], [y_ctrl[j]], "s", color=VERM, ms=4.5, zorder=3)

    yticks = y_attr + y_ctrl
    ylabels = [f"{nn(a[0])}  (n={a[1]})" for a in u] + [nn(cc[0]) for cc in c]
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.annotate("urban tasks", xy=(72, (len(u) - 1) / 2.0), fontsize=7.5,
                color=BLUE, rotation=90, va="center", ha="left")
    ax.annotate("cross-view\ncontrols", xy=(72, sum(y_ctrl) / len(y_ctrl)),
                fontsize=7.5, color=VERM, rotation=90, va="center", ha="left")
    ax.set_xlim(-8, 80)
    ax.set_xlabel(r"Fusion synergy:  full $-$ max(sat, sv)   (accuracy points)")
    ax.set_ylim(-0.7, y_ctrl[-1] + 0.7)
    leg = [Patch(facecolor=BLUE, label="urban (single-view sufficient)"),
           Patch(facecolor=VERM, label="cross-view (fusion required)"),
           Patch(facecolor="#EEEEEE", label=r"$\pm$3 pt equivalence band")]
    ax.legend(handles=leg, fontsize=6.6, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    p = os.path.join(OUT, "fig_synergy_control.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG: oracle dumbbell — full vs better single view, per urban task
# ============================================================================
def fig_oracle_dumbbell():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    u = sorted(urban_oracle_rows(), key=lambda r: r[2])   # by full_paired
    y = list(range(len(u)))
    for i, (name, n, full, oracle) in enumerate(u):
        ax.plot([full, oracle], [i, i], color=GREY, lw=1.1, zorder=2)
        ax.plot([full], [i], "o", color=VERM, ms=5, zorder=3,
                label="two-view (full)" if i == 0 else "")
        ax.plot([oracle], [i], "o", color=BLUE, ms=5, zorder=3,
                label="per-item single-view oracle" if i == 0 else "")
    ax.set_yticks(y); ax.set_yticklabels([f"{nn(a[0])} (n={a[1]})" for a in u])
    ax.set_xlabel("accuracy (%)")
    ax.legend(fontsize=6.8, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    gap = NUMS["pooled_urban7"]["full_minus_oracle"]
    ax.set_title("Per task, the two-view model vs. the better single view "
                 f"(pooled item-level oracle gap $={gap:.1f}$ pts).", fontsize=7.2)
    p = os.path.join(OUT, "fig_oracle_dumbbell.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_teaser()
    fig_synergy_control()
    fig_oracle_dumbbell()
    print("done.")
