#!/usr/bin/env python3
"""Single-page PDF report of the diffusiongemma image-ablation, grounded entirely
in results/diffusiongemma_20260614_224218/ablation_log.jsonl. No hardcoded numbers
— everything below is computed from the log at run time.

SCOPE: the 9 attribute-classification tasks only. Cross-view matching tasks
(mismatch_*, camera_direction) are out of scope — the task there IS aligning two
views, so a single-perspective ablation is ill-posed — and are excluded.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

LOG = Path("/home/ezel/Development/EOLLM/reasoning_distill/results/"
           "diffusiongemma_20260614_224218/ablation_log.jsonl")
OUT = Path("/tmp/diffusiongemma_ablation_report.pdf")

# The 9 attribute-classification tasks (image_mode == satellite_marked).
ATTR = ["land_use", "building_height", "urban_density", "junction_type",
        "green_space", "amenity_richness", "road_type", "road_surface",
        "transit_density"]
EXCLUDED = {"mismatch_binary_easy", "mismatch_binary_hard",
            "mismatch_mcq_easy", "mismatch_mcq_hard", "camera_direction"}
MODES = ["full", "sat_only", "sv_only", "blind"]

# ---- aggregate from the log (attribute tasks only) ----
acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # acc[topic][mode]=[correct,n]
overall = defaultdict(lambda: [0, 0])
for line in open(LOG):
    if not line.strip():
        continue
    r = json.loads(line)
    if "error" in r:
        continue
    t, m = r.get("topic"), r.get("mode")
    if t is None or m is None or t in EXCLUDED:
        continue
    c = 1 if r.get("correct") else 0
    acc[t][m][0] += c
    acc[t][m][1] += 1
    overall[m][0] += c
    overall[m][1] += 1


def p(cm):
    c, n = cm
    return 100 * c / n if n else None


def cell(cm):
    v = p(cm)
    return f"{v:.1f}" if v is not None else "—"


# ---- figure: one page (A4 portrait) ----
fig = plt.figure(figsize=(8.27, 11.69))
gs = GridSpec(5, 1, height_ratios=[0.62, 2.45, 3.55, 1.55, 1.15],
              hspace=0.62, left=0.07, right=0.955, top=0.96, bottom=0.035)

# header
ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
n_records = overall["blind"][1]
ax0.text(0, 0.85, "diffusiongemma — perspective ablation on attribute tasks",
         fontsize=15, fontweight="bold")
ax0.text(0, 0.45, "nvidia/diffusiongemma-26B-A4B-it-NVFP4  ·  EOLLM benchmark split  ·  "
                  "9 attribute-classification tasks  ·  greedy (temp 0)",
         fontsize=8.5, color="#444")
ax0.text(0, 0.12,
         "full = satellite + street-view · sat_only = satellite · sv_only = street-view · "
         "blind = no images (text+options).",
         fontsize=7.6, color="#666", style="italic")
ax0.text(0, -0.18,
         "Cross-view matching tasks (mismatch_*, camera_direction) excluded — "
         "the task there IS aligning two views, so a single-view ablation is ill-posed.",
         fontsize=7.6, color="#666", style="italic")

# ---- grouped bar chart ----
ax1 = fig.add_subplot(gs[1])
colors = {"full": "#1f4e79", "sat_only": "#5b9bd5",
          "sv_only": "#a9d08e", "blind": "#c9c9c9"}
x = np.arange(len(ATTR))
w = 0.2
for i, m in enumerate(MODES):
    vals = [p(acc[t][m]) if p(acc[t][m]) is not None else 0 for t in ATTR]
    bars = ax1.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[m])
ax1.axhline(25, ls=":", lw=0.8, color="#999")
ax1.text(len(ATTR) - 0.45, 26.5, "25% chance (4-opt)", fontsize=6.5, color="#999", ha="right")
ax1.set_xticks(x)
ax1.set_xticklabels(ATTR, rotation=35, ha="right", fontsize=8)
ax1.set_ylabel("accuracy (%)", fontsize=9)
ax1.set_ylim(0, 100)
ax1.legend(ncol=4, fontsize=8.5, loc="upper center",
           bbox_to_anchor=(0.5, 1.13), frameon=False)
ax1.set_xlim(-0.5, len(ATTR) - 0.5)
ax1.tick_params(labelsize=8)
ax1.grid(axis="y", lw=0.3, color="#ddd", zorder=0)

# ---- per-task results table ----
ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
ax2.text(0, 1.0, "Per-task accuracy (%)   —   n = benchmark questions",
         fontsize=9.5, fontweight="bold", transform=ax2.transAxes)
rows = []
for t in ATTR:
    n = acc[t]["blind"][1] or acc[t]["full"][1]
    f, b = p(acc[t]["full"]), p(acc[t]["blind"])
    bestsingle = max(p(acc[t]["sat_only"]) or 0, p(acc[t]["sv_only"]) or 0)
    gain_blind = f"{f - b:+.1f}" if (f is not None and b is not None) else "—"
    gain_single = f"{f - bestsingle:+.1f}" if f is not None else "—"
    rows.append([t, str(n), cell(acc[t]["full"]), cell(acc[t]["sat_only"]),
                 cell(acc[t]["sv_only"]), cell(acc[t]["blind"]),
                 gain_single, gain_blind])
# overall row
of, ob = p(overall["full"]), p(overall["blind"])
best_ov = max(p(overall["sat_only"]) or 0, p(overall["sv_only"]) or 0)
rows.append(["OVERALL", "", cell(overall["full"]), cell(overall["sat_only"]),
             cell(overall["sv_only"]), cell(overall["blind"]),
             f"{of - best_ov:+.1f}" if of else "—",
             f"{of - ob:+.1f}" if of and ob else "—"])
cols = ["task", "n", "full", "sat", "sv", "blind", "full−best1", "full−blind"]
tbl = ax2.table(cellText=rows, colLabels=cols, loc="center",
                cellLoc="center", colLoc="center", bbox=[0, 0.0, 1, 0.95])
tbl.auto_set_font_size(False); tbl.set_fontsize(7.6)
for (rr, cc), co in tbl.get_celld().items():
    co.set_linewidth(0.3)
    if rr == 0:
        co.set_facecolor("#1f4e79"); co.set_text_props(color="white", fontweight="bold")
    else:
        is_ov = rows[rr - 1][0] == "OVERALL"
        if is_ov:
            co.set_facecolor("#1f4e79"); co.set_text_props(color="white", fontweight="bold")
        else:
            co.set_facecolor("#eef2f8" if rr % 2 else "#dfe7f2")
    if cc == 0:
        co._loc = "left"
        co.set_text_props(ha="left", **({"color": "white", "fontweight": "bold"}
                                        if rr > 0 and rows[rr - 1][0] == "OVERALL" else {}))

# ---- image-mode mapping table ----
ax3 = fig.add_subplot(gs[3]); ax3.axis("off")
ax3.text(0, 1.0, "Which images go to which task   (inputs to the model)",
         fontsize=9.5, fontweight="bold", transform=ax3.transAxes)
mrows = [
    ["all 9 attribute tasks\n(land_use, building_height, urban_density,\n"
     "junction_type, green_space, amenity_richness,\nroad_type, road_surface, transit_density)",
     "satellite_marked",
     "full: 1 marked satellite + 4 street-view angles (Fwd/Bwd/Left/Right)\n"
     "sat_only: the marked satellite only\n"
     "sv_only: the 4 street-view angles only\n"
     "blind: no images (question + options text)"],
]
mt = ax3.table(cellText=mrows,
               colLabels=["task(s)", "image_mode", "images per ablation mode"],
               loc="center", cellLoc="left", colLoc="left", bbox=[0, 0.0, 1, 0.80])
mt.auto_set_font_size(False); mt.set_fontsize(7.2)
for (rr, cc), co in mt.get_celld().items():
    co.set_linewidth(0.3); co._loc = "left"; co.set_text_props(ha="left", va="center")
    co.set_width({0: 0.36, 1: 0.18, 2: 0.46}[cc])
    if rr == 0:
        co.set_facecolor("#1f4e79"); co.set_text_props(color="white", fontweight="bold", ha="left")
    else:
        co.set_facecolor("#f7f7f7")
    co.set_height(0.62)

# ---- takeaway footer (grounded) ----
ax4 = fig.add_subplot(gs[4]); ax4.axis("off")
a_full = p(overall["full"]); a_sat = p(overall["sat_only"])
a_sv = p(overall["sv_only"]); a_blind = p(overall["blind"])
# tasks where blind >= full (text/option-solvable leak)
leak = [t for t in ATTR if (p(acc[t]["blind"]) or 0) >= (p(acc[t]["full"]) or 0)]
msg = (
    "FINDING — on attribute tasks the two perspectives are NOT being combined.\n"
    f"  • full {a_full:.1f}%  ≈  sat_only {a_sat:.1f}%  ≈  sv_only {a_sv:.1f}%  —  full beats\n"
    f"     the best single view by only {a_full - max(a_sat, a_sv):+.1f} pts; both views add ~nothing over one.\n"
    f"  • Vision still helps vs blind {a_blind:.1f}% ({a_full - a_blind:+.1f} pts), but a single\n"
    "     perspective already captures essentially all of that gain.\n"
    + (f"  • {', '.join(leak)} score ≥ as well BLIND as with images\n"
       "     → partly solvable from text/options alone.\n" if leak else "")
    + "  Implication: these tasks do not force cross-perspective reasoning;\n"
      "  the dual-image framing is not earning its cost."
)
ax4.text(0, 0.96, msg, fontsize=7.4, va="top", transform=ax4.transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff6e5", ec="#e0a000", lw=0.8))

with PdfPages(OUT) as pdf:
    pdf.savefig(fig)
plt.close(fig)
print("wrote", OUT, "| attribute tasks:", len(ATTR),
      "| overall full/sat/sv/blind =",
      f"{a_full:.1f}/{a_sat:.1f}/{a_sv:.1f}/{a_blind:.1f}")
