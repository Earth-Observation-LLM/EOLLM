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
    abl = [ud["sat"], ud["sv"], ud["full"]]   # satfwd, verified; v4: no blind

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
    conds = ["sat", "sv", "full"]
    colors = [BLUE, GREEN, VERM]
    bars = ax_b.bar(conds, abl, color=colors, width=0.60)
    ax_b.set_ylim(0, 100); ax_b.set_ylabel("acc. (%)", fontsize=6.6)
    ax_b.tick_params(labelsize=6.6)
    for b, v in zip(bars, abl):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}",
                  ha="center", fontsize=6.2)
    ax_b.set_title(r"full $\approx$ best single view", fontsize=6.6)
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
    ax.text(93, (len(u) - 1) / 2.0, "urban\nattribute", color=BLUE,
            fontsize=6.2, va="center", ha="center")
    ax.text(93, len(u) + (len(c) - 1) / 2.0, "cross-view\n(by design)", color=VERM,
            fontsize=6.2, va="center", ha="center")
    ax.axvline(82, color="#DDDDDD", lw=0.6)
    ax.axhline(len(u) - 0.5, color="#CCCCCC", lw=0.7, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.6)

    fig.suptitle("On urban-attribute tasks the second view adds $\\approx$0 over the "
                 "better single view; the large cross-view gains are from tasks built "
                 "to require both views.", fontsize=7.6, y=1.03)
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
# FIG: item-level audit — diverging bar (rescued vs overturned), 9:1 imbalance
# ============================================================================
def fig_item_level():
    il = NUMS["item_level_urban7"]
    n = il["n"]
    resc = il["fusion_win"]; resc_pct = il["fusion_win_pct"]
    over = il["interfere"]; over_pct = il["interfere_pct"]
    ratio = il["ratio_interfere_to_win"]
    unchanged = n - resc - over

    fig, ax = plt.subplots(figsize=(5.2, 1.7))
    # two horizontal bars on a shared zero baseline (✓/✗ as Unicode; no LaTeX escapes)
    ax.barh([1], [resc], color=GREEN, height=0.60, zorder=3)
    ax.barh([0], [-over], color=VERM, height=0.60, zorder=3)
    ax.axvline(0, color="#555555", lw=0.9, zorder=4)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["rescued\n(full ✓, both single ✗)",
                        "overturned\n(full ✗, a single ✓)"], fontsize=6.6)
    # value labels just OUTSIDE the bar tips, pointing toward the axis edges (clear of y-labels)
    ax.text(resc + 14, 1, f"{resc}  ({resc_pct:.1f}%)", va="center", ha="left", fontsize=7.2, color=GREEN)
    ax.text(-over + 14, 0, f"{over}  ({over_pct:.1f}%)", va="center", ha="left", fontsize=7.2,
            color="white", fontweight="bold", zorder=5)
    ax.text(0, 1.7, f"{ratio:.0f} : 1 against fusion", ha="center", fontsize=7.4, color="#333333")
    ax.set_xlim(-430, 260)
    ax.set_xlabel(f"items the second view changes  (of {n:,}; {unchanged:,} unchanged)", fontsize=6.9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-400, -200, 0, 200]); ax.set_xticklabels(["400", "200", "0", "200"], fontsize=6.4)
    p = os.path.join(OUT, "fig_item_level.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG: full vs per-item oracle scatter — all urban tasks below y=x
# ============================================================================
def fig_oracle_scatter():
    u = urban_oracle_rows()   # (name, n, full_paired, oracle_item)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    lo, hi = 45, 90
    # faint shade below the diagonal
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=GREY, alpha=0.08, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=0.8, zorder=1)
    # per-task label nudges to avoid collisions in the upper-right cluster
    nudge = {"road_type": (4, 4), "junction_type": (4, -7), "land_use": (-2, -8),
             "urban_density": (4, -2), "building_height": (4, -2),
             "amenity_richness": (4, 2), "transit_density": (4, -2)}
    for name, nn_, full, oracle in u:
        ax.plot(oracle, full, "o", color=VERM, ms=6, zorder=3)
        dx, dy = nudge.get(name, (4, -3))
        ax.annotate(nn(name), (oracle, full), fontsize=5.6, color="#333333",
                    xytext=(dx, dy), textcoords="offset points", zorder=4)
    ax.text(0.04, 0.93,
            "all tasks below $y=x$:\na per-item single-view\noracle beats the two-view\nmodel on every task",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color="#555555")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("per-item single-view oracle (%)", fontsize=7.5)
    ax.set_ylabel("two-view full (%)", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    p = os.path.join(OUT, "fig_oracle_scatter.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG (optional): street-view pruning survival — ordered bars at k=1 + k=0 floor
# ============================================================================
def fig_prune():
    # v4 FIX: every bar here is the SAME quantity — survival of already-correct
    # answers (fraction of items the model got right with all street views that are
    # still right after pruning). The old version drew the k=0 value as a vertical
    # line slicing through the k=1 bars on an axis labelled "k=1", which read as if
    # the satellite-only number were an accuracy / a k=1 selection rule. Here k=0 is
    # just another survival bar, set apart and clearly labelled, so the axis is
    # purely survival and nothing is mixed.
    pr = NUMS["prune_survival_27b"]
    # k=1 selection rules (one street view kept), then the k=0 reference (none kept).
    arms = [("attention top-1", pr["top_k1"] * 100, BLUE, "$k{=}1$"),
            ("forward view",    pr["fwd_k1"] * 100, GREEN, "$k{=}1$"),
            ("random-1",        pr["random_k1_mean"] * 100, GREY, "$k{=}1$"),
            ("worst-1",         pr["bottom_k1"] * 100, GREY, "$k{=}1$"),
            ("no street view",  pr["zero_k0"] * 100, VERM, "$k{=}0$")]
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    y = list(range(len(arms)))[::-1]
    for yi, (lab, val, col, ktag) in zip(y, arms):
        ax.barh([yi], [val], color=col, height=0.62, zorder=3)
        ax.text(val - 1.2, yi, f"{val:.0f}", va="center", ha="right",
                fontsize=6.8, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a[0]}  ({a[3]})" for a in arms], fontsize=6.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("survival of already-correct answers (%)", fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    p = os.path.join(OUT, "fig_prune.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ============================================================================
# FIG: combined item-level panel (diverging bar on top, oracle scatter below) —
# one float that carries both the imbalance and the per-task oracle gap.
# ============================================================================
def fig_item_combined():
    il = NUMS["item_level_urban7"]
    n = il["n"]; resc = il["fusion_win"]; resc_pct = il["fusion_win_pct"]
    over = il["interfere"]; over_pct = il["interfere_pct"]; ratio = il["ratio_interfere_to_win"]
    unchanged = n - resc - over
    u = urban_oracle_rows()

    fig = plt.figure(figsize=(6.6, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)

    # --- left: diverging bar ---
    axb = fig.add_subplot(gs[0])
    axb.barh([1], [resc], color=GREEN, height=0.58, zorder=3)
    axb.barh([0], [-over], color=VERM, height=0.58, zorder=3)
    axb.axvline(0, color="#555555", lw=0.9, zorder=4)
    axb.set_yticks([1, 0])
    axb.set_yticklabels(["rescued\n(full ✓, both ✗)", "overturned\n(full ✗, a single ✓)"], fontsize=6.4)
    axb.text(resc + 14, 1, f"{resc} ({resc_pct:.1f}%)", va="center", ha="left", fontsize=6.8, color=GREEN)
    axb.text(-over + 14, 0, f"{over} ({over_pct:.1f}%)", va="center", ha="left", fontsize=6.8,
             color="white", fontweight="bold", zorder=5)
    axb.text(0, 1.72, f"{ratio:.0f} : 1 against fusion", ha="center", fontsize=7.0, color="#333333")
    axb.set_xlim(-430, 250)
    axb.set_xlabel(f"items changed (of {n:,}; {unchanged:,} unchanged)", fontsize=6.6)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.tick_params(axis="y", length=0)
    axb.set_xticks([-400, -200, 0, 200]); axb.set_xticklabels(["400", "200", "0", "200"], fontsize=6.0)

    # --- right: oracle scatter ---
    ax = fig.add_subplot(gs[1])
    lo, hi = 45, 90
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=GREY, alpha=0.08, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=0.8, zorder=1)
    nudge = {"road_type": (3, 3), "junction_type": (3, -6), "land_use": (-1, -7),
             "urban_density": (3, -2), "building_height": (3, -2),
             "amenity_richness": (3, 2), "transit_density": (3, -2)}
    for name, nn_, full, oracle in u:
        ax.plot(oracle, full, "o", color=VERM, ms=5, zorder=3)
        dx, dy = nudge.get(name, (3, -3))
        ax.annotate(nn(name), (oracle, full), fontsize=4.8, color="#333333",
                    xytext=(dx, dy), textcoords="offset points", zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("single-view oracle (%)", fontsize=6.8)
    ax.set_ylabel("two-view full (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.0)
    ax.text(0.04, 0.96, "all below $y=x$", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, color="#555555")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    p = os.path.join(OUT, "fig_item_combined.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_teaser()
    fig_synergy_control()
    fig_item_level()
    fig_oracle_scatter()
    fig_item_combined()
    fig_prune()
    print("done.")
