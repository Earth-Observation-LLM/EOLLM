"""
Dataset Statistics & Visualization Script

Generates a comprehensive set of plots and a summary stats file for the merged dataset.

Usage:
    python src/dataset_stats.py [--input PATH] [--outdir DIR]

Defaults:
    --input   /home/alperen/Documents/EODATA/dataset_merged.jsonl
    --outdir  /home/alperen/Documents/EODATA/stats/

Outputs (all saved to --outdir):
    01_question_type_freq.png          — question type frequency (count + %)
    02_questions_per_sample_dist.png   — histogram of questions-per-sample
    03_answer_dist_overall.png         — A/B/C/D distribution across all questions
    04_answer_dist_per_type.png        — A/B/C/D distribution per question type (heatmap)
    05_difficulty_dist.png             — difficulty distribution overall + per type
    06_city_sample_count.png           — samples per city
    07_city_question_count.png         — total questions per city
    08_city_sv_coverage.png            — street view completeness per city
    09_question_type_per_city.png      — question type breakdown per city (stacked bar)
    10_land_use_dist.png               — land use category distribution
    11_building_height_dist.png        — building height category distribution
    12_urban_density_dist.png          — urban density category distribution
    13_road_type_dist.png              — road type distribution
    14_amenity_count_dist.png          — amenity count distribution (histogram)
    15_building_count_dist.png         — building count distribution (histogram)
    16_transit_stop_dist.png           — transit stop count distribution
    17_metadata_coverage_heatmap.png   — metadata field availability by city (heatmap)
    18_questions_per_sample_by_city.png — box plot of question count per sample per city
    19_type_cooccurrence.png           — question type co-occurrence matrix per sample
    20_raw_landuse_dist.png            — raw OSM landuse tag distribution (top 20)
    summary_stats.txt                  — key numbers in plain text
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("tab10")
FIG_DPI = 150
LABEL_FONTSIZE = 9

TOPIC_ORDER = [
    "land_use", "building_height", "road_type", "urban_density",
    "green_space", "amenity_richness", "road_surface", "junction_type",
    "water_proximity", "transit_density", "camera_direction",
    "mismatch_binary", "mismatch_mcq",
]

ANSWER_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}
DIFFICULTY_COLORS = {"easy": "#55A868", "medium": "#4C72B0", "hard": "#C44E52"}


# ── Data loading ───────────────────────────────────────────────────────────

def load_dataset(path: str):
    """Load dataset_merged.jsonl and return list of records."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataframes(records):
    """
    Build two DataFrames:
      samples_df  — one row per sample
      questions_df — one row per question
    """
    sample_rows = []
    question_rows = []

    for rec in records:
        sid = rec["sample_id"]
        city = rec.get("location", {}).get("city", "Unknown")
        country = rec.get("location", {}).get("country", "Unknown")
        meta = rec.get("metadata", {})
        val = rec.get("validation", {})
        sv_count = val.get("streetview_count", 4)

        lu_cat = meta.get("land_use_category") or "unknown"
        bc = meta.get("osm_building_count")
        ac = meta.get("osm_amenity_count")
        levels = meta.get("osm_median_building_levels")
        water = meta.get("osm_water_distance_m")
        transit = meta.get("osm_transit_stop_count")
        road_type = meta.get("road_type") or "unknown"
        road_surface = meta.get("osm_road_surface") or "unknown"
        junction = meta.get("osm_junction_type") or "unknown"
        raw_lu = meta.get("osm_dominant_landuse_raw") or "unknown"

        questions = rec.get("questions", [])
        q_count = len(questions)
        topics_in_sample = [q.get("topic") for q in questions]

        sample_rows.append({
            "sample_id": sid,
            "city": city,
            "country": country,
            "sv_count": sv_count,
            "question_count": q_count,
            "land_use_category": lu_cat,
            "osm_building_count": bc,
            "osm_amenity_count": ac,
            "osm_median_building_levels": levels,
            "osm_water_distance_m": water,
            "osm_transit_stop_count": transit,
            "osm_has_park": meta.get("osm_has_park"),
            "road_type": road_type,
            "osm_road_surface": road_surface,
            "osm_junction_type": junction,
            "osm_dominant_landuse_raw": raw_lu,
            "topics_in_sample": topics_in_sample,
        })

        for q in questions:
            question_rows.append({
                "sample_id": sid,
                "city": city,
                "topic": q.get("topic", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
                "answer": q.get("answer", "?"),
                "n_options": len(q.get("options", {})),
            })

    samples_df = pd.DataFrame(sample_rows)
    questions_df = pd.DataFrame(question_rows)
    return samples_df, questions_df


# ── Helpers ────────────────────────────────────────────────────────────────

def savefig(fig, outdir, filename):
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def add_pct_labels(ax, total, fmt="{:.1f}%", fontsize=LABEL_FONTSIZE):
    """Add percentage labels on top of bar chart bars."""
    for bar in ax.patches:
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + total * 0.002,
            fmt.format(100 * h / total),
            ha="center", va="bottom", fontsize=fontsize,
        )


# ── Plot functions ─────────────────────────────────────────────────────────

def plot_question_type_freq(questions_df, outdir):
    """01 — question type frequency."""
    counts = questions_df["topic"].value_counts()
    # keep TOPIC_ORDER, append any extras
    order = [t for t in TOPIC_ORDER if t in counts.index] + \
            [t for t in counts.index if t not in TOPIC_ORDER]
    counts = counts.reindex(order)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("tab10", len(counts)))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.002,
                f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="bottom",
                fontsize=8, linespacing=1.4)
    ax.set_title("Question Type Frequency Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Number of Questions")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    savefig(fig, outdir, "01_question_type_freq.png")


def plot_questions_per_sample_dist(samples_df, outdir):
    """02 — histogram of questions per sample."""
    qc = samples_df["question_count"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    bins = range(int(qc.min()), int(qc.max()) + 2)
    ax.hist(qc, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.axvline(qc.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={qc.mean():.1f}")
    ax.axvline(qc.median(), color="orange", linestyle="-.", linewidth=1.5, label=f"Median={qc.median():.0f}")
    ax.set_title("Distribution: Questions per Sample", fontweight="bold")
    ax.set_xlabel("Number of Questions")
    ax.set_ylabel("Number of Samples")
    ax.legend()

    # CDF
    ax2 = axes[1]
    sorted_qc = np.sort(qc)
    cdf = np.arange(1, len(sorted_qc) + 1) / len(sorted_qc)
    ax2.plot(sorted_qc, cdf * 100, color=PALETTE[1], linewidth=2)
    ax2.set_title("CDF: Questions per Sample", fontweight="bold")
    ax2.set_xlabel("Number of Questions")
    ax2.set_ylabel("Cumulative % of Samples")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(True, alpha=0.4)

    stats_text = (f"Min: {qc.min()}  Max: {qc.max()}\n"
                  f"Mean: {qc.mean():.1f}  Median: {qc.median():.0f}\n"
                  f"Std: {qc.std():.1f}")
    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    savefig(fig, outdir, "02_questions_per_sample_dist.png")


def plot_answer_dist_overall(questions_df, outdir):
    """03 — overall A/B/C/D distribution across all questions."""
    # Separate 4-option and 2-option questions
    q4 = questions_df[questions_df["n_options"] == 4]
    q2 = questions_df[questions_df["n_options"] == 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, title, letters in [
        (axes[0], q4, f"Answer Distribution — 4-option questions\n(n={len(q4):,})", ["A","B","C","D"]),
        (axes[1], q2, f"Answer Distribution — 2-option questions\n(n={len(q2):,})", ["A","B"]),
    ]:
        counts = df["answer"].value_counts().reindex(letters, fill_value=0)
        total = counts.sum()
        bars = ax.bar(counts.index, counts.values,
                      color=[ANSWER_COLORS.get(l, "#888") for l in letters],
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
                    f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.axhline(total / len(letters), color="gray", linestyle="--",
                   linewidth=1.2, label=f"Expected ({100/len(letters):.0f}%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Correct Answer")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    savefig(fig, outdir, "03_answer_dist_overall.png")


def plot_answer_dist_per_type(questions_df, outdir):
    """04 — answer distribution per question type (heatmap of %)."""
    q4 = questions_df[questions_df["n_options"] == 4]
    pivot = q4.groupby(["topic", "answer"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["A","B","C","D"], fill_value=0)
    # Reorder rows
    row_order = [t for t in TOPIC_ORDER if t in pivot.index] + \
                [t for t in pivot.index if t not in TOPIC_ORDER]
    pivot = pivot.reindex(row_order)
    # Convert to % per row
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=20, vmax=30, ax=ax, linewidths=0.5,
                cbar_kws={"label": "% of questions with this answer"})
    ax.set_title("Correct Answer Distribution per Question Type (%)", fontweight="bold")
    ax.set_xlabel("Correct Answer Option")
    ax.set_ylabel("Question Type")
    fig.tight_layout()
    savefig(fig, outdir, "04_answer_dist_per_type.png")


def plot_difficulty_dist(questions_df, outdir):
    """05 — difficulty distribution overall and per type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    ax = axes[0]
    counts = questions_df["difficulty"].value_counts()
    order = ["easy", "medium", "hard"]
    counts = counts.reindex([d for d in order if d in counts.index])
    total = counts.sum()
    bars = ax.bar(counts.index, counts.values,
                  color=[DIFFICULTY_COLORS.get(d, "#888") for d in counts.index],
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
                f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Difficulty Distribution — All Questions", fontweight="bold")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Per type (stacked bar)
    ax2 = axes[1]
    pivot = questions_df.groupby(["topic", "difficulty"]).size().unstack(fill_value=0)
    row_order = [t for t in TOPIC_ORDER if t in pivot.index]
    pivot = pivot.reindex(row_order)
    diff_cols = [d for d in ["easy", "medium", "hard"] if d in pivot.columns]
    pivot = pivot[diff_cols]
    # Normalize to %
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind="barh", stacked=True, ax=ax2,
                   color=[DIFFICULTY_COLORS.get(d, "#888") for d in diff_cols],
                   edgecolor="white", linewidth=0.5)
    ax2.set_title("Difficulty Distribution per Question Type", fontweight="bold")
    ax2.set_xlabel("% of Questions")
    ax2.set_ylabel("Question Type")
    ax2.legend(title="Difficulty", loc="lower right")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    savefig(fig, outdir, "05_difficulty_dist.png")


def plot_city_sample_count(samples_df, outdir):
    """06 — samples per city (horizontal bar)."""
    counts = samples_df["city"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 12))
    bars = ax.barh(counts.index, counts.values,
                   color=sns.color_palette("viridis", len(counts)),
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)
    ax.set_title("Samples per City", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Samples")
    ax.axvline(counts.mean(), color="red", linestyle="--",
               label=f"Mean = {counts.mean():.0f}")
    ax.legend()
    fig.tight_layout()
    savefig(fig, outdir, "06_city_sample_count.png")


def plot_city_question_count(samples_df, questions_df, outdir):
    """07 — total questions per city."""
    counts = questions_df["city"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 12))
    bars = ax.barh(counts.index, counts.values,
                   color=sns.color_palette("plasma", len(counts)),
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_title("Total Questions per City", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Questions")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    savefig(fig, outdir, "07_city_question_count.png")


def plot_city_sv_coverage(samples_df, outdir):
    """08 — street view completeness per city."""
    sv = samples_df.groupby("city").apply(
        lambda g: pd.Series({
            "full_4": (g["sv_count"] == 4).mean() * 100,
            "3": (g["sv_count"] == 3).mean() * 100,
            "lt3": (g["sv_count"] < 3).mean() * 100,
        }), include_groups=False
    ).reset_index()

    cities_sorted = sv.sort_values("full_4")["city"]
    sv = sv.set_index("city").reindex(cities_sorted)

    fig, ax = plt.subplots(figsize=(9, 12))
    sv[["full_4", "3", "lt3"]].plot(kind="barh", stacked=True, ax=ax,
        color=["#55A868", "#DD8452", "#C44E52"],
        edgecolor="white", linewidth=0.4)
    ax.set_title("Street View Completeness per City", fontsize=14, fontweight="bold")
    ax.set_xlabel("% of Samples")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(["4 angles (full)", "3 angles", "< 3 angles"], loc="lower right")
    fig.tight_layout()
    savefig(fig, outdir, "08_city_sv_coverage.png")


def plot_question_type_per_city(questions_df, outdir):
    """09 — question type breakdown per city (stacked 100% bar)."""
    pivot = questions_df.groupby(["city", "topic"]).size().unstack(fill_value=0)
    col_order = [t for t in TOPIC_ORDER if t in pivot.columns]
    pivot = pivot[col_order]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Sort cities by total sample count
    city_totals = pivot.sum(axis=1).sort_values(ascending=True)
    pivot_pct = pivot_pct.reindex(city_totals.index)

    colors = sns.color_palette("tab10", len(col_order)) + \
             sns.color_palette("Set2", max(0, len(col_order) - 10))

    fig, ax = plt.subplots(figsize=(14, 12))
    pivot_pct.plot(kind="barh", stacked=True, ax=ax,
                   color=colors[:len(col_order)],
                   edgecolor="white", linewidth=0.3, width=0.8)
    ax.set_title("Question Type Distribution per City (% of questions)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("% of Questions")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(title="Question Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "09_question_type_per_city.png")


def plot_land_use_dist(samples_df, outdir):
    """10 — land use category distribution."""
    counts = samples_df["land_use_category"].value_counts()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Set2", len(counts)),
                  edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
                f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Land Use Category Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Land Use Category")
    ax.set_ylabel("Number of Samples")
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    fig.tight_layout()
    savefig(fig, outdir, "10_land_use_dist.png")


def plot_building_height_dist(samples_df, outdir):
    """11 — building height category distribution (from samples with data)."""
    df = samples_df[samples_df["osm_median_building_levels"].notna()].copy()
    df["levels"] = df["osm_median_building_levels"].astype(float)

    def bin_height(lvl):
        if lvl <= 3:   return "Low-rise (1-3)"
        if lvl <= 7:   return "Mid-rise (4-7)"
        if lvl <= 20:  return "High-rise (8-20)"
        return "Skyscraper (20+)"

    df["height_cat"] = df["levels"].apply(bin_height)
    cat_order = ["Low-rise (1-3)", "Mid-rise (4-7)", "High-rise (8-20)", "Skyscraper (20+)"]
    counts = df["height_cat"].value_counts().reindex(cat_order, fill_value=0)
    total_with_data = len(df)
    total_all = len(samples_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Blues_d", 4),
                  edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_with_data * 0.003,
                f"{val:,}\n({100*val/total_with_data:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Building Height Distribution\n(n={total_with_data:,} samples with data / {total_all:,} total)",
                 fontweight="bold")
    ax.set_xlabel("Height Category")
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=10, ha="right")

    # Histogram of raw levels
    ax2 = axes[1]
    ax2.hist(df["levels"], bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax2.set_title("Raw Building Levels (floors) — Histogram", fontweight="bold")
    ax2.set_xlabel("Median Building Levels")
    ax2.set_ylabel("Count")
    ax2.axvline(df["levels"].median(), color="red", linestyle="--",
                label=f"Median={df['levels'].median():.1f}")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, outdir, "11_building_height_dist.png")


def plot_urban_density_dist(samples_df, outdir):
    """12 — urban density category distribution."""
    df = samples_df[samples_df["osm_building_count"].notna()].copy()
    df["bc"] = df["osm_building_count"].astype(float)

    def bin_density(bc):
        if bc <= 15:   return "Low (0-15)"
        if bc <= 50:   return "Moderate (16-50)"
        if bc <= 150:  return "High (51-150)"
        return "Very High (151+)"

    df["density_cat"] = df["bc"].apply(bin_density)
    cat_order = ["Low (0-15)", "Moderate (16-50)", "High (51-150)", "Very High (151+)"]
    counts = df["density_cat"].value_counts().reindex(cat_order, fill_value=0)
    total = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Oranges_d", 4),
                  edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{val:,}\n({100*val/total:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Urban Density Category Distribution", fontweight="bold")
    ax.set_xlabel("Density Category")
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=10, ha="right")

    ax2 = axes[1]
    clipped = df["bc"].clip(upper=300)
    ax2.hist(clipped, bins=40, color=PALETTE[1], edgecolor="white", linewidth=0.5)
    ax2.set_title("Building Count Distribution (clipped at 300)", fontweight="bold")
    ax2.set_xlabel("Building Count (200m radius)")
    ax2.set_ylabel("Count")
    ax2.axvline(df["bc"].median(), color="red", linestyle="--",
                label=f"Median={df['bc'].median():.0f}")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, outdir, "12_urban_density_dist.png")


def plot_road_type_dist(samples_df, outdir):
    """13 — road type distribution."""
    counts = samples_df["road_type"].value_counts()
    total = counts.sum()
    order = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unknown"]
    counts = counts.reindex([r for r in order if r in counts.index] +
                            [r for r in counts.index if r not in order])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("tab10", len(counts)),
                  edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{val:,}\n({100*val/total:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Road Type Distribution (Samples)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Road Type")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    savefig(fig, outdir, "13_road_type_dist.png")


def plot_amenity_count_dist(samples_df, outdir):
    """14 — amenity count distribution."""
    df = samples_df[samples_df["osm_amenity_count"].notna()].copy()
    df["ac"] = df["osm_amenity_count"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    clipped = df["ac"].clip(upper=50)
    ax.hist(clipped, bins=51, color=PALETTE[2], edgecolor="white", linewidth=0.5)
    ax.set_title("Amenity Count Distribution (clipped at 50)", fontweight="bold")
    ax.set_xlabel("Amenity Count (radius)")
    ax.set_ylabel("Number of Samples")
    ax.axvline(df["ac"].median(), color="red", linestyle="--",
                label=f"Median={df['ac'].median():.0f}")
    ax.axvline(df["ac"].mean(), color="orange", linestyle="-.",
                label=f"Mean={df['ac'].mean():.1f}")
    ax.legend()

    # Binned categories
    ax2 = axes[1]
    def bin_amenity(ac):
        if ac == 0:   return "Minimal (0)"
        if ac <= 4:   return "Low (1-4)"
        if ac <= 19:  return "Moderate (5-19)"
        return "High (20+)"
    df["amenity_cat"] = df["ac"].apply(bin_amenity)
    cat_order = ["Minimal (0)", "Low (1-4)", "Moderate (5-19)", "High (20+)"]
    counts = df["amenity_cat"].value_counts().reindex(cat_order, fill_value=0)
    total = len(df)
    bars = ax2.bar(counts.index, counts.values,
                   color=sns.color_palette("Greens_d", 4), edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.003,
                 f"{val:,}\n({100*val/total:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_title("Amenity Richness Category Distribution", fontweight="bold")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    savefig(fig, outdir, "14_amenity_count_dist.png")


def plot_building_count_dist(samples_df, outdir):
    """15 — building count raw distribution (log scale)."""
    df = samples_df[samples_df["osm_building_count"].notna()].copy()
    df["bc"] = df["osm_building_count"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(df["bc"].clip(upper=500), bins=50, color=PALETTE[3],
            edgecolor="white", linewidth=0.5)
    ax.set_title("Building Count Distribution (clipped at 500)", fontweight="bold")
    ax.set_xlabel("Building Count (200m radius)")
    ax.set_ylabel("Number of Samples")
    ax.axvline(df["bc"].median(), color="red", linestyle="--",
                label=f"Median={df['bc'].median():.0f}")
    ax.axvline(df["bc"].mean(), color="orange", linestyle="-.",
                label=f"Mean={df['bc'].mean():.1f}")
    ax.legend()

    ax2 = axes[1]
    # City-level box plot of building count — top 15 cities by median
    city_medians = df.groupby("city")["bc"].median().sort_values(ascending=False).head(15)
    subset = df[df["city"].isin(city_medians.index)]
    city_order = city_medians.index.tolist()
    data = [subset[subset["city"] == c]["bc"].values for c in city_order]
    bp = ax2.boxplot(data, vert=False, patch_artist=True,
                     medianprops=dict(color="red", linewidth=2))
    colors = sns.color_palette("tab20", len(city_order))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_yticklabels(city_order, fontsize=8)
    ax2.set_title("Building Count by City (top 15 by median)", fontweight="bold")
    ax2.set_xlabel("Building Count")

    fig.tight_layout()
    savefig(fig, outdir, "15_building_count_dist.png")


def plot_transit_stop_dist(samples_df, outdir):
    """16 — transit stop count distribution."""
    df = samples_df[samples_df["osm_transit_stop_count"].notna()].copy()
    df["tc"] = df["osm_transit_stop_count"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(df["tc"].clip(upper=20), bins=21, color=PALETTE[4],
            edgecolor="white", linewidth=0.5)
    ax.set_title("Transit Stop Count Distribution (clipped at 20)", fontweight="bold")
    ax.set_xlabel("Transit Stop Count (300m radius)")
    ax.set_ylabel("Number of Samples")
    ax.axvline(df["tc"].median(), color="red", linestyle="--",
                label=f"Median={df['tc'].median():.0f}")
    ax.axvline(df["tc"].mean(), color="orange", linestyle="-.",
                label=f"Mean={df['tc'].mean():.1f}")
    ax.legend()

    ax2 = axes[1]
    def bin_transit(tc):
        if tc == 0:  return "None (0)"
        if tc <= 2:  return "Low (1-2)"
        if tc <= 5:  return "Moderate (3-5)"
        return "High (6+)"
    df["transit_cat"] = df["tc"].apply(bin_transit)
    cat_order = ["None (0)", "Low (1-2)", "Moderate (3-5)", "High (6+)"]
    counts = df["transit_cat"].value_counts().reindex(cat_order, fill_value=0)
    total = len(df)
    bars = ax2.bar(counts.index, counts.values,
                   color=sns.color_palette("Purples_d", 4), edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.003,
                 f"{val:,}\n({100*val/total:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_title("Transit Density Category Distribution", fontweight="bold")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    savefig(fig, outdir, "16_transit_stop_dist.png")


def plot_metadata_coverage_heatmap(samples_df, outdir):
    """17 — metadata field availability by city (heatmap)."""
    meta_fields = {
        "land_use_cat": "land_use_category",
        "building_levels": "osm_median_building_levels",
        "building_count": "osm_building_count",
        "amenity_count": "osm_amenity_count",
        "has_park": "osm_has_park",
        "road_type": "road_type",
        "road_surface": "osm_road_surface",
        "junction_type": "osm_junction_type",
        "water_distance": "osm_water_distance_m",
        "transit_stops": "osm_transit_stop_count",
        "road_bearing": None,  # always present (from road_type)
    }

    # Build availability per city
    rows = []
    for city, grp in samples_df.groupby("city"):
        row = {"city": city, "n": len(grp)}
        for label, field in meta_fields.items():
            if field is None:
                row[label] = 100.0
            else:
                available = grp[field].notna() & (grp[field] != "unknown") & (grp[field] != "")
                row[label] = 100 * available.mean()
        rows.append(row)

    cov_df = pd.DataFrame(rows).set_index("city")
    # Sort by total coverage
    cov_df["avg"] = cov_df[[c for c in cov_df.columns if c != "n"]].mean(axis=1)
    cov_df = cov_df.sort_values("avg", ascending=True)
    matrix = cov_df[[c for c in cov_df.columns if c not in ["n", "avg"]]]

    fig, ax = plt.subplots(figsize=(13, 12))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, ax=ax,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "% of samples with field populated"})
    ax.set_title("Metadata Field Coverage by City (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metadata Field")
    ax.set_ylabel("City")
    fig.tight_layout()
    savefig(fig, outdir, "17_metadata_coverage_heatmap.png")


def plot_questions_per_sample_by_city(samples_df, outdir):
    """18 — box plot of question count per sample, per city."""
    city_medians = samples_df.groupby("city")["question_count"].median().sort_values()
    city_order = city_medians.index.tolist()

    fig, ax = plt.subplots(figsize=(9, 12))
    data = [samples_df[samples_df["city"] == c]["question_count"].values for c in city_order]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    colors = sns.color_palette("tab20", len(city_order))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yticklabels(city_order, fontsize=9)
    ax.set_title("Questions per Sample by City", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Questions per Sample")
    ax.axvline(samples_df["question_count"].mean(), color="gray",
               linestyle="--", linewidth=1, label=f"Global mean={samples_df['question_count'].mean():.1f}")
    ax.legend()
    fig.tight_layout()
    savefig(fig, outdir, "18_questions_per_sample_by_city.png")


def plot_type_cooccurrence(samples_df, outdir):
    """19 — question type co-occurrence: how often do two types appear in the same sample."""
    topics_list = [t for t in TOPIC_ORDER]
    n = len(topics_list)
    co = np.zeros((n, n), dtype=int)
    topic_idx = {t: i for i, t in enumerate(topics_list)}

    for topics in samples_df["topics_in_sample"]:
        present = set(t for t in topics if t in topic_idx)
        for t1 in present:
            for t2 in present:
                co[topic_idx[t1], topic_idx[t2]] += 1

    # Normalize by total samples
    total = len(samples_df)
    co_pct = 100 * co / total

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.eye(n, dtype=bool)  # mask diagonal
    sns.heatmap(co_pct, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=topics_list, yticklabels=topics_list,
                ax=ax, linewidths=0.3, mask=mask,
                cbar_kws={"label": "% of samples where both types appear"})
    # Diagonal separately (always 100% with itself)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#d3d3d3"))
        ax.text(i + 0.5, i + 0.5, f"{co_pct[i,i]:.0f}%",
                ha="center", va="center", fontsize=7, color="gray")
    ax.set_title("Question Type Co-occurrence (% of samples containing both types)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "19_type_cooccurrence.png")


def plot_raw_landuse_dist(samples_df, outdir):
    """20 — raw OSM landuse tag distribution (top 20 values)."""
    counts = samples_df["osm_dominant_landuse_raw"].value_counts()
    top20 = counts.head(20)
    total = counts.sum()
    other_count = counts[20:].sum()

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("tab20", len(top20))
    bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, top20.values[::-1]):
        ax.text(val + total * 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({100*val/total:.1f}%)", va="center", fontsize=9)
    ax.set_title(f"Raw OSM Landuse Tag Distribution (Top 20 of {len(counts)} unique values)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Samples")
    if other_count:
        ax.text(0.98, 0.01, f"Other ({len(counts)-20} values): {other_count:,} samples",
                transform=ax.transAxes, ha="right", fontsize=9, color="gray")
    fig.tight_layout()
    savefig(fig, outdir, "20_raw_landuse_dist.png")


# ── Summary text ───────────────────────────────────────────────────────────

def write_summary(samples_df, questions_df, outdir):
    qc = samples_df["question_count"]
    lines = [
        "=" * 60,
        "EOLLM DATASET — SUMMARY STATISTICS",
        "=" * 60,
        "",
        f"Dataset path : /home/alperen/Documents/EODATA/dataset_merged.jsonl",
        "",
        "── SCALE ──────────────────────────────────────────────────",
        f"Total samples        : {len(samples_df):,}",
        f"Total questions      : {len(questions_df):,}",
        f"Number of cities     : {samples_df['city'].nunique()}",
        f"Number of countries  : {samples_df['country'].nunique()}",
        f"Question types       : {questions_df['topic'].nunique()}",
        "",
        "── QUESTIONS PER SAMPLE ───────────────────────────────────",
        f"Min    : {qc.min()}",
        f"Max    : {qc.max()}",
        f"Mean   : {qc.mean():.2f}",
        f"Median : {qc.median():.0f}",
        f"Std    : {qc.std():.2f}",
        f"Mode   : {qc.mode().iloc[0]}",
        "",
        "── STREET VIEW COVERAGE ───────────────────────────────────",
        f"Full (4 angles)  : {(samples_df['sv_count']==4).sum():,}  ({100*(samples_df['sv_count']==4).mean():.1f}%)",
        f"3 angles         : {(samples_df['sv_count']==3).sum():,}  ({100*(samples_df['sv_count']==3).mean():.1f}%)",
        f"< 3 angles       : {(samples_df['sv_count']<3).sum():,}  ({100*(samples_df['sv_count']<3).mean():.1f}%)",
        "",
        "── QUESTION TYPE DISTRIBUTION ─────────────────────────────",
    ]
    topic_counts = questions_df["topic"].value_counts()
    total_q = len(questions_df)
    for topic, cnt in topic_counts.items():
        lines.append(f"  {topic:<25} {cnt:>7,}  ({100*cnt/total_q:.1f}%)")
    lines += [
        "",
        "── DIFFICULTY DISTRIBUTION ────────────────────────────────",
    ]
    diff_counts = questions_df["difficulty"].value_counts()
    for diff, cnt in diff_counts.items():
        lines.append(f"  {diff:<10} {cnt:>7,}  ({100*cnt/total_q:.1f}%)")
    lines += [
        "",
        "── ANSWER KEY DISTRIBUTION (all 4-option questions) ───────",
    ]
    q4 = questions_df[questions_df["n_options"] == 4]
    ans_counts = q4["answer"].value_counts().reindex(["A","B","C","D"], fill_value=0)
    for ans, cnt in ans_counts.items():
        lines.append(f"  {ans}  {cnt:>7,}  ({100*cnt/len(q4):.1f}%)")
    lines += [
        "",
        "── METADATA AVAILABILITY ──────────────────────────────────",
        f"  land_use_category         : {samples_df['land_use_category'].notna().sum():,}  ({100*samples_df['land_use_category'].notna().mean():.1f}%)",
        f"  osm_building_count        : {samples_df['osm_building_count'].notna().sum():,}  ({100*samples_df['osm_building_count'].notna().mean():.1f}%)",
        f"  osm_amenity_count         : {samples_df['osm_amenity_count'].notna().sum():,}  ({100*samples_df['osm_amenity_count'].notna().mean():.1f}%)",
        f"  osm_median_building_levels: {samples_df['osm_median_building_levels'].notna().sum():,}  ({100*samples_df['osm_median_building_levels'].notna().mean():.1f}%)",
        f"  osm_water_distance_m      : {samples_df['osm_water_distance_m'].notna().sum():,}  ({100*samples_df['osm_water_distance_m'].notna().mean():.1f}%)",
        f"  osm_transit_stop_count    : {samples_df['osm_transit_stop_count'].notna().sum():,}  ({100*samples_df['osm_transit_stop_count'].notna().mean():.1f}%)",
        f"  osm_road_surface          : {(samples_df['osm_road_surface']!='unknown').sum():,}  ({100*(samples_df['osm_road_surface']!='unknown').mean():.1f}%)",
        f"  osm_junction_type         : {(samples_df['osm_junction_type']!='unknown').sum():,}  ({100*(samples_df['osm_junction_type']!='unknown').mean():.1f}%)",
        "",
        "── LAND USE DISTRIBUTION ──────────────────────────────────",
    ]
    lu_counts = samples_df["land_use_category"].value_counts()
    for lu, cnt in lu_counts.items():
        lines.append(f"  {lu:<20} {cnt:>5,}  ({100*cnt/len(samples_df):.1f}%)")
    lines += [
        "",
        "── SAMPLES PER CITY (top 10 and bottom 10) ────────────────",
        "  Top 10:",
    ]
    city_counts = samples_df["city"].value_counts()
    for city, cnt in city_counts.head(10).items():
        lines.append(f"    {city:<25} {cnt:>4}")
    lines.append("  Bottom 10:")
    for city, cnt in city_counts.tail(10).items():
        lines.append(f"    {city:<25} {cnt:>4}")
    lines += ["", "=" * 60]

    out_path = os.path.join(outdir, "summary_stats.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: summary_stats.txt")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics and plots")
    parser.add_argument(
        "--input",
        default="/home/alperen/Documents/EODATA/dataset_merged.jsonl",
        help="Path to dataset_merged.jsonl",
    )
    parser.add_argument(
        "--outdir",
        default="/home/alperen/Documents/EODATA/stats",
        help="Output directory for plots and summary",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading dataset from {args.input} ...")
    records = load_dataset(args.input)
    print(f"Loaded {len(records):,} samples. Building dataframes ...")
    samples_df, questions_df = build_dataframes(records)
    print(f"Generating {20} plots + summary ...\n")

    plot_question_type_freq(questions_df, args.outdir)
    plot_questions_per_sample_dist(samples_df, args.outdir)
    plot_answer_dist_overall(questions_df, args.outdir)
    plot_answer_dist_per_type(questions_df, args.outdir)
    plot_difficulty_dist(questions_df, args.outdir)
    plot_city_sample_count(samples_df, args.outdir)
    plot_city_question_count(samples_df, questions_df, args.outdir)
    plot_city_sv_coverage(samples_df, args.outdir)
    plot_question_type_per_city(questions_df, args.outdir)
    plot_land_use_dist(samples_df, args.outdir)
    plot_building_height_dist(samples_df, args.outdir)
    plot_urban_density_dist(samples_df, args.outdir)
    plot_road_type_dist(samples_df, args.outdir)
    plot_amenity_count_dist(samples_df, args.outdir)
    plot_building_count_dist(samples_df, args.outdir)
    plot_transit_stop_dist(samples_df, args.outdir)
    plot_metadata_coverage_heatmap(samples_df, args.outdir)
    plot_questions_per_sample_by_city(samples_df, args.outdir)
    plot_type_cooccurrence(samples_df, args.outdir)
    plot_raw_landuse_dist(samples_df, args.outdir)
    write_summary(samples_df, questions_df, args.outdir)

    print(f"\nAll outputs saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
