"""
Build all figures for the ablation report.
Data is hard-coded from per-epoch verdicts (lab-ws, dumped 2026-04-30).
"""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-run, per-epoch, per-topic accuracy. Numbers extracted from
# epoch_*_verdict.md files for each run on lab-ws.
# Topic order is fixed and shared across runs.
# ---------------------------------------------------------------------------

TOPICS = [
    "amenity_richness", "building_height", "camera_direction", "green_space",
    "junction_type", "land_use", "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard", "road_surface", "road_type",
    "transit_density", "urban_density",
]

# Each run: list of dicts {epoch: {topic: acc}} plus 'overall' list aligned with epochs.
RUNS = {
    # ----- baselines -----
    "PC-base": {
        "split": "per_city",
        "trained": set(TOPICS),  # all 14
        "epochs": list(range(1, 7)),
        "overall": [71.5, 72.3, 75.0, 76.4, 76.9, 78.3],
        "topic": {
            "amenity_richness":     [49.2, 49.2, 51.9, 57.2, 53.5, 62.0],
            "building_height":      [62.5, 60.2, 60.2, 65.1, 64.4, 69.0],
            "camera_direction":     [25.3, 35.1, 46.0, 51.0, 53.1, 56.5],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [69.9, 71.7, 72.8, 76.6, 70.1, 77.0],
            "land_use":             [76.8, 75.9, 76.5, 77.0, 77.5, 75.9],
            "mismatch_binary_easy": [91.6, 93.6, 96.3, 94.5, 96.8, 96.6],
            "mismatch_binary_hard": [87.2, 85.0, 88.6, 88.8, 92.0, 90.2],
            "mismatch_mcq_easy":    [95.0, 94.3, 96.8, 98.2, 98.0, 99.1],
            "mismatch_mcq_hard":    [82.7, 81.5, 86.8, 88.2, 90.6, 90.6],
            "road_surface":         [94.2, 94.2, 94.2, 94.2, 94.2, 94.2],
            "road_type":            [73.3, 71.1, 73.1, 74.7, 71.3, 71.3],
            "transit_density":      [44.6, 44.9, 44.6, 47.2, 48.5, 50.8],
            "urban_density":        [65.1, 69.5, 72.5, 68.3, 75.9, 73.8],
        },
    },
    "SU-base": {
        "split": "seen_unseen",
        "trained": set(TOPICS),
        "epochs": list(range(1, 9)),
        "overall": [67.2, 66.0, 70.7, 72.4, 71.9, 72.5, 72.2, 70.8],
        "topic": {
            "amenity_richness":     [57.8, 35.7, 61.6, 65.4, 54.5, 63.6, 61.6, 59.4],
            "building_height":      [44.1, 43.8, 51.8, 55.9, 54.0, 54.8, 52.6, 54.4],
            "camera_direction":     [26.5, 28.7, 42.5, 48.0, 55.5, 57.0, 56.7, 55.6],
            "green_space":          [72.8, 42.4, 71.9, 72.3, 71.0, 71.4, 69.2, 63.8],
            "junction_type":        [72.7, 68.3, 67.5, 68.1, 75.6, 69.4, 69.2, 71.4],
            "land_use":             [70.8, 72.9, 69.1, 74.2, 74.7, 71.4, 71.0, 69.1],
            "mismatch_binary_easy": [93.5, 93.5, 94.0, 93.8, 96.2, 95.9, 97.3, 95.0],
            "mismatch_binary_hard": [81.5, 81.2, 86.3, 89.7, 89.0, 91.1, 92.6, 91.6],
            "mismatch_mcq_easy":    [89.6, 95.8, 95.3, 97.6, 98.1, 98.0, 98.1, 98.5],
            "mismatch_mcq_hard":    [78.9, 82.9, 81.9, 87.4, 87.7, 88.9, 91.1, 90.5],
            "road_surface":         [60.7, 68.7, 51.9, 38.8, 30.6, 35.5, 44.6, 31.5],
            "road_type":            [71.4, 72.3, 70.0, 70.8, 73.1, 70.4, 67.7, 70.9],
            "transit_density":      [46.5, 38.7, 49.5, 50.6, 47.2, 46.5, 41.3, 36.9],
            "urban_density":        [62.5, 70.1, 76.5, 75.6, 70.9, 72.4, 70.4, 70.6],
        },
    },
    # ----- ablations -----
    "Split-1 (no urban)": {
        "split": "per_city",
        "trained": {"camera_direction", "green_space", "mismatch_binary_easy",
                    "mismatch_binary_hard", "mismatch_mcq_easy", "mismatch_mcq_hard",
                    "road_surface"},
        "epochs": list(range(1, 6)),
        "overall": [59.6, 63.4, 64.8, 66.1, 66.3],
        "topic": {
            "amenity_richness":     [38.3, 37.3, 39.0, 38.5, 38.0],
            "building_height":      [42.1, 49.0, 49.4, 50.2, 52.1],
            "camera_direction":     [26.0, 31.4, 40.5, 41.9, 46.0],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [54.5, 62.1, 59.2, 61.8, 60.7],
            "land_use":             [44.4, 54.9, 52.9, 50.3, 50.8],
            "mismatch_binary_easy": [94.5, 96.4, 92.7, 98.0, 97.1],
            "mismatch_binary_hard": [84.0, 89.1, 87.5, 92.7, 92.3],
            "mismatch_mcq_easy":    [93.6, 95.2, 97.1, 97.9, 98.0],
            "mismatch_mcq_hard":    [79.3, 83.1, 86.1, 88.4, 88.2],
            "road_surface":         [94.2, 94.2, 94.2, 93.7, 94.0],
            "road_type":            [43.7, 53.7, 56.3, 55.1, 55.3],
            "transit_density":      [30.3, 30.7, 31.6, 31.4, 31.0],
            "urban_density":        [30.8, 30.8, 38.9, 41.7, 41.7],
        },
    },
    "Split-2a (no geo, PC)": {
        "split": "per_city",
        "trained": {"amenity_richness", "building_height", "green_space", "junction_type",
                    "land_use", "road_surface", "road_type", "transit_density", "urban_density"},
        "epochs": list(range(1, 4)),
        "overall": [59.5, 50.4, 50.7],
        "topic": {
            "amenity_richness":     [50.4, 36.2, 36.9],
            "building_height":      [60.5, 44.4, 43.7],
            "camera_direction":     [24.1, 26.4, 26.2],
            "green_space":          [100.0, 100.0, 100.0],
            "junction_type":        [67.9, 58.3, 52.0],
            "land_use":             [76.3, 72.9, 73.6],
            "mismatch_binary_easy": [80.9, 51.7, 54.5],
            "mismatch_binary_hard": [72.5, 49.6, 54.7],
            "mismatch_mcq_easy":    [34.6, 26.6, 26.6],
            "mismatch_mcq_hard":    [30.5, 25.1, 23.4],
            "road_surface":         [94.2, 94.2, 94.2],
            "road_type":            [72.0, 66.8, 67.6],
            "transit_density":      [39.9, 34.9, 36.7],
            "urban_density":        [62.2, 55.1, 54.2],
        },
    },
    "Split-2b (no geo, SU)": {
        "split": "seen_unseen",
        "trained": {"amenity_richness", "building_height", "green_space", "junction_type",
                    "land_use", "road_surface", "road_type", "transit_density", "urban_density"},
        "epochs": list(range(1, 6)),
        "overall": [58.9, 59.0, 61.8, 60.9, 60.4],
        "topic": {
            "amenity_richness":     [58.1, 61.7, 62.9, 60.4, 59.7],
            "building_height":      [47.1, 50.4, 51.1, 53.7, 54.0],
            "camera_direction":     [27.1, 27.5, 27.1, 26.8, 26.1],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [75.2, 74.7, 72.9, 69.4, 69.6],
            "land_use":             [70.4, 74.6, 74.0, 75.6, 76.2],
            "mismatch_binary_easy": [75.0, 69.1, 82.5, 80.0, 77.4],
            "mismatch_binary_hard": [67.9, 65.4, 71.7, 69.7, 68.1],
            "mismatch_mcq_easy":    [31.1, 31.9, 33.8, 35.0, 37.3],
            "mismatch_mcq_hard":    [26.7, 25.0, 31.4, 30.7, 31.1],
            "road_surface":         [98.8, 97.9, 98.6, 98.1, 98.4],
            "road_type":            [73.1, 71.7, 72.7, 71.3, 72.1],
            "transit_density":      [45.7, 44.9, 49.1, 48.8, 44.9],
            "urban_density":        [71.3, 75.0, 75.8, 73.3, 72.9],
        },
    },
    "Split-3 (no mismatch)": {
        "split": "per_city",
        "trained": {"amenity_richness", "building_height", "camera_direction", "green_space",
                    "junction_type", "land_use", "road_surface", "road_type",
                    "transit_density", "urban_density"},
        "epochs": list(range(1, 6)),
        "overall": [57.8, 59.3, 62.7, 62.6, 63.2],
        "topic": {
            "amenity_richness":     [52.6, 53.7, 61.3, 62.2, 62.6],
            "building_height":      [49.8, 62.8, 64.8, 66.3, 66.3],
            "camera_direction":     [23.0, 33.0, 42.8, 39.9, 43.5],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [66.1, 71.4, 74.6, 75.4, 76.6],
            "land_use":             [76.1, 76.5, 77.9, 80.7, 79.0],
            "mismatch_binary_easy": [75.0, 67.0, 77.0, 73.1, 75.2],
            "mismatch_binary_hard": [67.0, 59.5, 69.0, 66.0, 65.2],
            "mismatch_mcq_easy":    [28.7, 31.4, 29.8, 30.8, 31.4],
            "mismatch_mcq_hard":    [27.6, 28.5, 26.6, 29.2, 29.4],
            "road_surface":         [94.2, 94.2, 94.2, 94.0, 93.7],
            "road_type":            [70.9, 73.6, 72.7, 72.7, 72.7],
            "transit_density":      [45.6, 44.6, 47.4, 46.2, 49.0],
            "urban_density":        [63.1, 69.3, 72.5, 72.9, 73.3],
        },
    },
    "Split-5 (no camera)": {
        "split": "per_city",
        "trained": {"amenity_richness", "building_height", "green_space", "junction_type",
                    "land_use", "mismatch_binary_easy", "mismatch_binary_hard",
                    "mismatch_mcq_easy", "mismatch_mcq_hard", "road_surface", "road_type",
                    "transit_density", "urban_density"},
        "epochs": list(range(1, 6)),
        "overall": [70.7, 71.8, 74.9, 75.9, 76.5],
        "topic": {
            "amenity_richness":     [51.7, 50.6, 55.6, 59.0, 61.0],
            "building_height":      [54.4, 61.3, 61.7, 65.5, 65.1],
            "camera_direction":     [26.7, 27.5, 26.2, 29.8, 29.9],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [69.0, 72.3, 75.9, 77.0, 78.3],
            "land_use":             [75.6, 75.6, 78.6, 77.5, 77.9],
            "mismatch_binary_easy": [92.0, 96.1, 96.1, 97.1, 97.3],
            "mismatch_binary_hard": [86.5, 83.6, 90.0, 91.4, 92.2],
            "mismatch_mcq_easy":    [95.7, 94.3, 98.4, 98.2, 98.4],
            "mismatch_mcq_hard":    [81.6, 82.7, 88.4, 90.6, 90.2],
            "road_surface":         [94.2, 94.2, 94.5, 94.5, 94.5],
            "road_type":            [70.4, 70.6, 73.6, 73.4, 74.0],
            "transit_density":      [44.6, 45.5, 48.3, 48.0, 48.1],
            "urban_density":        [61.1, 66.5, 72.9, 73.8, 76.3],
        },
    },
    "Split-6 (no hard mismatch)": {
        "split": "per_city",
        "trained": {"amenity_richness", "building_height", "camera_direction", "green_space",
                    "junction_type", "land_use", "mismatch_binary_easy", "mismatch_mcq_easy",
                    "road_surface", "road_type", "transit_density", "urban_density"},
        "epochs": list(range(1, 6)),
        "overall": [68.5, 73.3, 75.1, 76.3, 76.4],
        "topic": {
            "amenity_richness":     [47.8, 58.6, 58.3, 59.9, 61.9],
            "building_height":      [61.3, 57.9, 62.5, 65.9, 68.2],
            "camera_direction":     [25.8, 41.7, 39.9, 45.1, 44.2],
            "green_space":          [100.0, 100.0, 100.0, 100.0, 100.0],
            "junction_type":        [65.8, 72.1, 74.6, 76.3, 75.9],
            "land_use":             [77.4, 76.3, 79.9, 79.0, 80.7],
            "mismatch_binary_easy": [93.2, 95.7, 96.8, 97.3, 97.3],
            "mismatch_binary_hard": [75.8, 80.0, 80.6, 83.8, 82.7],
            "mismatch_mcq_easy":    [93.0, 95.5, 97.0, 98.0, 97.5],
            "mismatch_mcq_hard":    [76.3, 76.8, 83.1, 81.3, 81.6],
            "road_surface":         [94.2, 94.2, 94.2, 94.0, 94.2],
            "road_type":            [66.7, 70.1, 73.3, 73.6, 73.6],
            "transit_density":      [39.8, 47.6, 50.4, 51.7, 51.0],
            "urban_density":        [61.0, 71.7, 72.2, 74.0, 74.3],
        },
    },
}

RANDOM_BASELINE = {t: 25.0 for t in TOPICS}
RANDOM_BASELINE.update({"mismatch_binary_easy": 50.0, "mismatch_binary_hard": 50.0})

PC_RUNS = ["PC-base", "Split-1 (no urban)", "Split-2a (no geo, PC)",
           "Split-3 (no mismatch)", "Split-5 (no camera)", "Split-6 (no hard mismatch)"]
SU_RUNS = ["SU-base", "Split-2b (no geo, SU)"]

COLORS = {
    "PC-base":                   "#1f77b4",
    "SU-base":                   "#1f77b4",
    "Split-1 (no urban)":        "#d62728",
    "Split-2a (no geo, PC)":     "#2ca02c",
    "Split-2b (no geo, SU)":     "#2ca02c",
    "Split-3 (no mismatch)":     "#ff7f0e",
    "Split-5 (no camera)":       "#9467bd",
    "Split-6 (no hard mismatch)":"#8c564b",
}
LINESTYLE = {
    "PC-base": "-", "SU-base": "-",
    "Split-1 (no urban)": "--",
    "Split-2a (no geo, PC)": "--",
    "Split-2b (no geo, SU)": "--",
    "Split-3 (no mismatch)": "--",
    "Split-5 (no camera)": "--",
    "Split-6 (no hard mismatch)": "--",
}


# ---------------------------------------------------------------------------
# Figure 1 — Overall accuracy trajectory, all 6 runs
# ---------------------------------------------------------------------------
def fig_overall():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, runs, title in [
        (axes[0], PC_RUNS, "per_city eval (6984 samples)"),
        (axes[1], SU_RUNS, "seen_unseen eval (8831 samples)"),
    ]:
        for r in runs:
            d = RUNS[r]
            ax.plot(d["epochs"], d["overall"], marker="o", label=r,
                    color=COLORS[r], linestyle=LINESTYLE[r], linewidth=2)
            # final-value annotation
            ax.annotate(f"{d['overall'][-1]:.1f}%",
                        (d["epochs"][-1], d["overall"][-1]),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color=COLORS[r])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Overall val accuracy (%)")
    axes[0].set_ylim(45, 82)
    fig.suptitle("Overall validation accuracy by epoch", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_overall.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Withheld-task transfer for Split 1 (urban tasks, never trained)
# ---------------------------------------------------------------------------
def fig_split1_withheld():
    withheld = ["amenity_richness", "building_height", "junction_type", "land_use",
                "road_type", "transit_density", "urban_density"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    s1 = RUNS["Split-1 (no urban)"]
    pc = RUNS["PC-base"]
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(withheld):
        ax.plot(s1["epochs"], s1["topic"][t], marker="o", color=cmap(i),
                linestyle="--", linewidth=1.6, label=f"{t} (Split-1)")
        # baseline final value as horizontal mark
        ax.axhline(pc["topic"][t][-1], color=cmap(i), linestyle=":",
                   linewidth=0.8, alpha=0.5)
    ax.axhline(25, color="grey", linestyle="-.", linewidth=0.8, label="random (25%)")
    ax.set_title("Split-1: per-topic accuracy on withheld URBAN tasks vs PC-base final\n"
                 "(dashed = Split-1 trajectory; dotted horiz = PC-base final on same topic)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(s1["epochs"])
    ax.set_ylim(20, 85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_split1_withheld.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Mismatch+camera_dir transfer (withheld in 2a/2b/3)
# ---------------------------------------------------------------------------
def fig_geo_withheld():
    withheld = ["camera_direction", "mismatch_binary_easy", "mismatch_binary_hard",
                "mismatch_mcq_easy", "mismatch_mcq_hard"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.8), sharey=True)
    runs_to_plot = ["Split-2a (no geo, PC)", "Split-2b (no geo, SU)",
                    "Split-3 (no mismatch)", "Split-5 (no camera)",
                    "Split-6 (no hard mismatch)"]
    for ax, t in zip(axes, withheld):
        for r in runs_to_plot:
            d = RUNS[r]
            # Only plot a run on a topic if the topic is withheld for that run.
            if t in d["trained"]:
                continue
            ax.plot(d["epochs"], d["topic"][t], marker="o", linestyle="--",
                    color=COLORS[r], linewidth=1.6, label=r)
        # baselines
        pc_v = RUNS["PC-base"]["topic"][t][-1]
        su_v = RUNS["SU-base"]["topic"][t][-1]
        ax.axhline(pc_v, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"PC-base ({pc_v:.0f})")
        ax.axhline(su_v, color="#1f77b4", linestyle="-.", linewidth=1, alpha=0.5,
                   label=f"SU-base ({su_v:.0f})")
        ax.axhline(RANDOM_BASELINE[t], color="grey", linestyle="-", linewidth=0.6,
                   alpha=0.6, label=f"rand ({RANDOM_BASELINE[t]:.0f})")
        ax.set_title(t.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylim(15, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(fontsize=11, loc="center left", bbox_to_anchor=(1.0, 0.5),
                    frameon=True, framealpha=0.95, handlelength=2.2, borderpad=0.6,
                    labelspacing=0.5)
    fig.suptitle("Withheld-task transfer (only runs where the topic is withheld are plotted)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_geo_withheld.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Trained-task accuracy: ablation vs PC-base on the same topic.
# Does removing tasks make the kept ones better? (No.)
# ---------------------------------------------------------------------------
def fig_trained_vs_baseline():
    runs = ["Split-1 (no urban)", "Split-2a (no geo, PC)", "Split-3 (no mismatch)",
            "Split-5 (no camera)", "Split-6 (no hard mismatch)"]
    pc = RUNS["PC-base"]

    deltas = {}
    for r in runs:
        d = RUNS[r]
        out = []
        for t in TOPICS:
            if t not in d["trained"]:
                continue
            out.append((t, d["topic"][t][-1] - pc["topic"][t][-1]))
        deltas[r] = sorted(out, key=lambda x: x[1])

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=False)
    for ax, r in zip(axes, runs):
        topics_sorted = [t for t, _ in deltas[r]]
        vals = [v for _, v in deltas[r]]
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
        ax.barh(topics_sorted, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_title(f"{r}: trained-task delta vs PC-base (negative = ablation hurt the trained task)",
                     fontsize=10)
        ax.set_xlabel("Δ accuracy (pp)")
        ax.set_xlim(-25, 6)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Counter-hypothesis check: does focusing the model on fewer tasks "
                 "lift the kept tasks above the full-baseline?",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_trained_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Random/majority leakage flags (green_space, road_surface)
# ---------------------------------------------------------------------------
def fig_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, t in zip(axes, ["green_space", "road_surface"]):
        for r in PC_RUNS:
            d = RUNS[r]
            ax.plot(d["epochs"], d["topic"][t], marker="o",
                    color=COLORS[r], linestyle=LINESTYLE[r], linewidth=1.5, label=r)
        for r in SU_RUNS:
            d = RUNS[r]
            ax.plot(d["epochs"], d["topic"][t], marker="s",
                    color=COLORS[r], linestyle=":", linewidth=1.5, label=r)
        ax.axhline(25, color="grey", linestyle="-", linewidth=0.6, label="random (25%)")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.set_ylim(20, 105)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Leakage / triviality check: topics that pin to ~100% across all runs are not informative",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_leakage.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_overall()
    fig_split1_withheld()
    fig_geo_withheld()
    fig_trained_vs_baseline()
    fig_leakage()
    print("Wrote figures to", OUT)
    for f in sorted(OUT.glob("*.pdf")):
        print(" ", f.name, f.stat().st_size, "bytes")
