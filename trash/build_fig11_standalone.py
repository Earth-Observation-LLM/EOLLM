#!/usr/bin/env python3
"""Standalone, self-contained generator for fig11 (geographic scatter).

Embeds all data (4,168 sample locations + 40 city labels + a coarse world
country outline) directly in this file. The only external dependency is
matplotlib. Tweak the TUNABLE PARAMETERS block below and re-run.

    python build_fig11_standalone.py
        -> writes  fig11_geographic_scatter.pdf  (vector, scalable)
                   fig11_geographic_scatter.png  (raster, PNG_DPI)

Use a --variant flag to switch built-in presets:
    python build_fig11_standalone.py --variant original
    python build_fig11_standalone.py --variant compact   # bigger markers/labels
"""
from __future__ import annotations

import argparse
import base64
import json
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches


# =====================================================================
# TUNABLE PARAMETERS — edit these freely
# =====================================================================
# Presets: "original" matches the dense academic-report rendering;
# "compact" makes points and labels much larger so the figure stays
# readable when shrunk into a small column / slide tile.
PRESETS = {
    "original": dict(
        figsize         = (16, 8.5),
        marker_s        = 12,
        marker_alpha    = 0.75,
        marker_edge_c   = "none",
        marker_edge_w   = 0.0,
        country_face_c  = "#e9eef3",
        country_edge_c  = "#9aa7b3",
        country_edge_w  = 0.4,
        label_fontsize  = 8,
        label_offset    = (4, 4),
        label_bbox_pad  = 0.15,
        label_bbox_alpha= 0.65,
        axis_label_fs   = 11,
        tick_fs         = 11,
        title_fs        = 13,
        title_show      = True,
        legend_fs       = 9,
        seen_color      = "#1f77b4",
        unseen_color    = "#ff7f0e",
        seen_text_color = "#1f3a5f",
        unseen_text_color = "#7a3b00",
        png_dpi         = 200,
    ),
    "compact": dict(
        figsize         = (16, 8.5),
        marker_s        = 55,
        marker_alpha    = 0.9,
        marker_edge_c   = "white",
        marker_edge_w   = 0.6,
        country_face_c  = "#e9eef3",
        country_edge_c  = "#7a8794",
        country_edge_w  = 0.7,
        label_fontsize  = 13,
        label_offset    = (6, 6),
        label_bbox_pad  = 0.22,
        label_bbox_alpha= 0.78,
        axis_label_fs   = 13,
        tick_fs         = 11,
        title_fs        = 14,
        title_show      = True,
        legend_fs       = 13,
        seen_color      = "#1f77b4",
        unseen_color    = "#ff7f0e",
        seen_text_color = "#1f3a5f",
        unseen_text_color = "#7a3b00",
        png_dpi         = 300,
    ),
    # Poster preset: ONE bigger marker per city (no per-sample dots),
    # numbered so a side index can identify them without cluttering
    # the map with names. Designed for a small card on an A1 poster.
    "poster": dict(
        figsize         = (22, 4.0),
        marker_s        = 110,
        marker_alpha    = 0.95,
        marker_edge_c   = "white",
        marker_edge_w   = 1.4,
        country_face_c  = "#eef2f6",
        country_edge_c  = "#9aa7b3",
        country_edge_w  = 0.5,
        label_fontsize  = 10,          # numeric labels inside markers
        label_offset    = (0, 0),
        label_bbox_pad  = 0.0,
        label_bbox_alpha= 0.0,
        axis_label_fs   = 10,
        tick_fs         = 9,
        title_fs        = 13,
        title_show      = False,
        legend_fs       = 10,
        seen_color      = "#1f6bb5",
        unseen_color    = "#e8731b",
        seen_text_color = "white",
        unseen_text_color = "white",
        png_dpi         = 300,
    ),
}

# Per-city label overrides (e.g. shorter aliases). Empty by default;
# add entries like "New York City": "NYC" if a label is too long.
CITY_LABEL_OVERRIDES: dict[str, str] = {
    # "New York City": "NYC",
    # "Rio de Janeiro": "Rio",
}

# Manual nudges for label placement, in matplotlib "offset points".
# Useful for the dense Europe cluster. Example:
#   CITY_LABEL_NUDGE = {"Zurich": (-25, -12)}
CITY_LABEL_NUDGE: dict[str, tuple[float, float]] = {
    # "Zurich":    (-30, -12),
    # "Paris":     (-25,   6),
}

# Plot extents — set to (None, None) to autoscale.
XLIM = (-180, 180)
YLIM = (-60, 85)
# =====================================================================


mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#555555",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =====================================================================
# Embedded data — 4,168 unique sample locations across 40 cities
# Tuple format: (latitude, longitude, city_name, role)
#   role is one of {"seen", "unseen"} — 32 seen cities, 8 unseen cities
# =====================================================================
LOCATIONS: list[tuple[float, float, str, str]] = [
    (52.28788, 4.90882, 'Amsterdam', 'seen'),
    (52.29065, 4.99199, 'Amsterdam', 'seen'),
    (52.29083, 4.73262, 'Amsterdam', 'seen'),
    (52.29103, 4.90061, 'Amsterdam', 'seen'),
    (52.29118, 4.91609, 'Amsterdam', 'seen'),
    (52.29361, 4.85046, 'Amsterdam', 'seen'),
    (52.29504, 4.84368, 'Amsterdam', 'seen'),
    (52.29593, 4.87138, 'Amsterdam', 'seen'),
    (52.29895, 4.90728, 'Amsterdam', 'seen'),
    (52.30051, 4.92799, 'Amsterdam', 'seen'),
    (52.30116, 4.79682, 'Amsterdam', 'seen'),
    (52.3012, 4.98149, 'Amsterdam', 'seen'),
    (52.30258, 4.9177, 'Amsterdam', 'seen'),
    (52.30708, 4.81835, 'Amsterdam', 'seen'),
    (52.30826, 4.9435, 'Amsterdam', 'seen'),
    (52.31132, 4.96532, 'Amsterdam', 'seen'),
    (52.31171, 4.99907, 'Amsterdam', 'seen'),
    (52.3121, 4.81429, 'Amsterdam', 'seen'),
    (52.31308, 4.81311, 'Amsterdam', 'seen'),
    (52.31384, 4.95618, 'Amsterdam', 'seen'),
    (52.31396, 4.80963, 'Amsterdam', 'seen'),
    (52.31482, 4.81509, 'Amsterdam', 'seen'),
    (52.31676, 4.87523, 'Amsterdam', 'seen'),
    (52.31734, 4.93204, 'Amsterdam', 'seen'),
    (52.32117, 4.8672, 'Amsterdam', 'seen'),
    (52.32359, 4.96475, 'Amsterdam', 'seen'),
    (52.32386, 4.89148, 'Amsterdam', 'seen'),
    (52.32453, 4.89464, 'Amsterdam', 'seen'),
    (52.32522, 4.83596, 'Amsterdam', 'seen'),
    (52.32778, 4.88566, 'Amsterdam', 'seen'),
    (52.32785, 4.96267, 'Amsterdam', 'seen'),
    (52.32896, 4.95255, 'Amsterdam', 'seen'),
    (52.32921, 4.86184, 'Amsterdam', 'seen'),
    (52.33058, 4.9529, 'Amsterdam', 'seen'),
    (52.33194, 4.85391, 'Amsterdam', 'seen'),
    (52.33222, 4.93411, 'Amsterdam', 'seen'),
    (52.33266, 4.79221, 'Amsterdam', 'seen'),
    (52.33436, 4.88453, 'Amsterdam', 'seen'),
    (52.33467, 4.99817, 'Amsterdam', 'seen'),
    (52.33659, 4.97077, 'Amsterdam', 'seen'),
    (52.33696, 4.96347, 'Amsterdam', 'seen'),
    (52.33726, 4.7886, 'Amsterdam', 'seen'),
    (52.33902, 4.96815, 'Amsterdam', 'seen'),
    (52.33902, 4.98871, 'Amsterdam', 'seen'),
    (52.3395, 4.98682, 'Amsterdam', 'seen'),
    (52.34007, 4.76311, 'Amsterdam', 'seen'),
    (52.34194, 4.98399, 'Amsterdam', 'seen'),
    (52.342, 4.87676, 'Amsterdam', 'seen'),
    (52.34307, 4.76477, 'Amsterdam', 'seen'),
    (52.34334, 4.76482, 'Amsterdam', 'seen'),
    (52.34498, 4.84197, 'Amsterdam', 'seen'),
    (52.3462, 4.86034, 'Amsterdam', 'seen'),
    (52.34676, 4.76849, 'Amsterdam', 'seen'),
    (52.34744, 4.97188, 'Amsterdam', 'seen'),
    (52.34757, 4.89793, 'Amsterdam', 'seen'),
    (52.34834, 4.92137, 'Amsterdam', 'seen'),
    (52.34891, 4.96763, 'Amsterdam', 'seen'),
    (52.35195, 4.97555, 'Amsterdam', 'seen'),
    (52.35212, 4.94948, 'Amsterdam', 'seen'),
    (52.3567, 4.96406, 'Amsterdam', 'seen'),
    (52.35686, 4.88315, 'Amsterdam', 'seen'),
    (52.35849, 4.92614, 'Amsterdam', 'seen'),
    (52.3585, 4.90233, 'Amsterdam', 'seen'),
    (52.36088, 4.99073, 'Amsterdam', 'seen'),
    (52.36129, 4.98454, 'Amsterdam', 'seen'),
    (52.36407, 4.85795, 'Amsterdam', 'seen'),
    (52.36464, 4.79704, 'Amsterdam', 'seen'),
    (52.3664, 4.8692, 'Amsterdam', 'seen'),
    (52.36673, 4.80641, 'Amsterdam', 'seen'),
    (52.36693, 4.94072, 'Amsterdam', 'seen'),
    (52.36795, 4.8009, 'Amsterdam', 'seen'),
    (52.37011, 4.83419, 'Amsterdam', 'seen'),
    (52.37178, 4.97133, 'Amsterdam', 'seen'),
    (52.37244, 4.81208, 'Amsterdam', 'seen'),
    (52.37391, 4.81344, 'Amsterdam', 'seen'),
    (52.37406, 4.86093, 'Amsterdam', 'seen'),
    (52.37458, 4.81339, 'Amsterdam', 'seen'),
    (52.37621, 4.82749, 'Amsterdam', 'seen'),
    (52.37637, 4.8891, 'Amsterdam', 'seen'),
    (52.37726, 4.88332, 'Amsterdam', 'seen'),
    (52.38085, 4.81541, 'Amsterdam', 'seen'),
    (52.38204, 4.9047, 'Amsterdam', 'seen'),
    (52.38207, 4.88296, 'Amsterdam', 'seen'),
    (52.3823, 4.74342, 'Amsterdam', 'seen'),
    (52.38529, 4.86572, 'Amsterdam', 'seen'),
    (52.3854, 4.74808, 'Amsterdam', 'seen'),
    (52.38745, 4.94669, 'Amsterdam', 'seen'),
    (52.38868, 4.84596, 'Amsterdam', 'seen'),
    (52.38973, 4.85555, 'Amsterdam', 'seen'),
    (52.38974, 4.79442, 'Amsterdam', 'seen'),
    (52.39034, 4.90643, 'Amsterdam', 'seen'),
    (52.39038, 4.81861, 'Amsterdam', 'seen'),
    (52.39042, 4.92069, 'Amsterdam', 'seen'),
    (52.39198, 4.84503, 'Amsterdam', 'seen'),
    (52.392, 4.84445, 'Amsterdam', 'seen'),
    (52.39215, 4.87086, 'Amsterdam', 'seen'),
    (52.39314, 4.87244, 'Amsterdam', 'seen'),
    (52.39359, 4.94246, 'Amsterdam', 'seen'),
    (52.39607, 4.91826, 'Amsterdam', 'seen'),
    (52.39618, 4.95975, 'Amsterdam', 'seen'),
    (52.39707, 4.73043, 'Amsterdam', 'seen'),
    (52.40322, 4.94194, 'Amsterdam', 'seen'),
    (52.40803, 4.93973, 'Amsterdam', 'seen'),
    (52.40875, 4.82903, 'Amsterdam', 'seen'),
    (52.40967, 4.74727, 'Amsterdam', 'seen'),
    (52.41424, 4.79881, 'Amsterdam', 'seen'),
    (52.41516, 4.94593, 'Amsterdam', 'seen'),
    (52.41845, 4.91385, 'Amsterdam', 'seen'),
    (52.42235, 4.95678, 'Amsterdam', 'seen'),
    (52.42236, 4.88846, 'Amsterdam', 'seen'),
    (52.42249, 4.73635, 'Amsterdam', 'seen'),
    (52.4227, 4.96465, 'Amsterdam', 'seen'),
    (52.42576, 4.88212, 'Amsterdam', 'seen'),
    (52.42795, 4.99624, 'Amsterdam', 'seen'),
    (52.4283, 4.79964, 'Amsterdam', 'seen'),
    (52.42861, 4.98612, 'Amsterdam', 'seen'),
    (39.75178, 32.87371, 'Ankara', 'seen'),
    (39.75676, 32.86453, 'Ankara', 'seen'),
    (39.76548, 32.76647, 'Ankara', 'seen'),
    (39.77516, 32.82164, 'Ankara', 'seen'),
    (39.78799, 32.71904, 'Ankara', 'seen'),
    (39.78868, 32.86479, 'Ankara', 'seen'),
    (39.79115, 32.81099, 'Ankara', 'seen'),
    (39.79337, 32.8493, 'Ankara', 'seen'),
    (39.79799, 32.81012, 'Ankara', 'seen'),
    (39.80022, 32.75096, 'Ankara', 'seen'),
    (39.80166, 32.73013, 'Ankara', 'seen'),
    (39.8024, 32.80946, 'Ankara', 'seen'),
    (39.80614, 32.80752, 'Ankara', 'seen'),
    (39.8194, 32.70785, 'Ankara', 'seen'),
    (39.82091, 32.72153, 'Ankara', 'seen'),
    (39.82331, 32.70994, 'Ankara', 'seen'),
    (39.82804, 32.65378, 'Ankara', 'seen'),
    (39.82861, 32.66325, 'Ankara', 'seen'),
    (39.83918, 32.64923, 'Ankara', 'seen'),
    (39.84186, 32.94421, 'Ankara', 'seen'),
    (39.84438, 32.67406, 'Ankara', 'seen'),
    (39.85309, 32.95819, 'Ankara', 'seen'),
    (39.86397, 32.66697, 'Ankara', 'seen'),
    (39.86574, 32.65287, 'Ankara', 'seen'),
    (39.86616, 32.68939, 'Ankara', 'seen'),
    (39.86725, 32.70894, 'Ankara', 'seen'),
    (39.86809, 32.86202, 'Ankara', 'seen'),
    (39.87608, 32.95545, 'Ankara', 'seen'),
    (39.88459, 32.91903, 'Ankara', 'seen'),
    (39.88477, 32.91858, 'Ankara', 'seen'),
    (39.88485, 32.64159, 'Ankara', 'seen'),
    (39.88562, 32.84089, 'Ankara', 'seen'),
    (39.88995, 32.87404, 'Ankara', 'seen'),
    (39.89015, 32.90345, 'Ankara', 'seen'),
    (39.8969, 32.80233, 'Ankara', 'seen'),
    (39.90044, 32.96053, 'Ankara', 'seen'),
    (39.90648, 32.80753, 'Ankara', 'seen'),
    (39.90739, 32.96379, 'Ankara', 'seen'),
    (39.90911, 32.74328, 'Ankara', 'seen'),
    (39.90922, 32.72835, 'Ankara', 'seen'),
    (39.92157, 32.82635, 'Ankara', 'seen'),
    (39.93652, 32.64258, 'Ankara', 'seen'),
    (39.94289, 32.60099, 'Ankara', 'seen'),
    (39.94317, 32.65581, 'Ankara', 'seen'),
    (39.94776, 32.74201, 'Ankara', 'seen'),
    (39.95625, 32.88996, 'Ankara', 'seen'),
    (39.96303, 32.97478, 'Ankara', 'seen'),
    (39.96559, 32.61458, 'Ankara', 'seen'),
    (39.96884, 32.80831, 'Ankara', 'seen'),
    (39.96926, 32.59326, 'Ankara', 'seen'),
    (39.97124, 32.56043, 'Ankara', 'seen'),
    (39.97262, 32.70124, 'Ankara', 'seen'),
    (39.97265, 32.69449, 'Ankara', 'seen'),
    (39.97411, 32.89508, 'Ankara', 'seen'),
    (39.97563, 32.64748, 'Ankara', 'seen'),
    (39.9773, 32.93409, 'Ankara', 'seen'),
    (39.97932, 32.64892, 'Ankara', 'seen'),
    (39.97982, 32.7033, 'Ankara', 'seen'),
    (39.98387, 32.61405, 'Ankara', 'seen'),
    (39.98447, 32.90407, 'Ankara', 'seen'),
    (39.9859, 32.7543, 'Ankara', 'seen'),
    (39.9945, 32.80835, 'Ankara', 'seen'),
    (39.99923, 32.82148, 'Ankara', 'seen'),
    (40.01034, 32.86562, 'Ankara', 'seen'),
    (40.01201, 32.77545, 'Ankara', 'seen'),
    (40.0175, 32.71632, 'Ankara', 'seen'),
    (40.0205, 32.65357, 'Ankara', 'seen'),
    (40.02495, 32.74693, 'Ankara', 'seen'),
    (40.02661, 32.89604, 'Ankara', 'seen'),
    (40.02801, 32.66421, 'Ankara', 'seen'),
    (40.03572, 32.75632, 'Ankara', 'seen'),
    (40.03925, 32.79619, 'Ankara', 'seen'),
    (40.04297, 32.87703, 'Ankara', 'seen'),
    (40.04568, 32.62347, 'Ankara', 'seen'),
    (40.04798, 32.6317, 'Ankara', 'seen'),
    (40.04818, 32.76145, 'Ankara', 'seen'),
    (40.05175, 32.85621, 'Ankara', 'seen'),
    (40.05279, 32.8996, 'Ankara', 'seen'),
    (40.05309, 32.64563, 'Ankara', 'seen'),
    (40.06121, 32.99929, 'Ankara', 'seen'),
    (40.09352, 32.81034, 'Ankara', 'seen'),
    (40.09576, 32.97929, 'Ankara', 'seen'),
    (36.85769, 30.74394, 'Antalya', 'seen'),
    (36.85806, 30.63235, 'Antalya', 'seen'),
    (36.85966, 30.73886, 'Antalya', 'seen'),
    (36.86004, 30.73045, 'Antalya', 'seen'),
    (36.86143, 30.73795, 'Antalya', 'seen'),
    (36.86276, 30.63451, 'Antalya', 'seen'),
    (36.86381, 30.64157, 'Antalya', 'seen'),
    (36.86441, 30.74842, 'Antalya', 'seen'),
    (36.86498, 30.6337, 'Antalya', 'seen'),
    (36.86577, 30.73627, 'Antalya', 'seen'),
    (36.86626, 30.72514, 'Antalya', 'seen'),
    (36.86803, 30.7484, 'Antalya', 'seen'),
    (36.87019, 30.73519, 'Antalya', 'seen'),
    (36.8704, 30.74258, 'Antalya', 'seen'),
    (36.87052, 30.73693, 'Antalya', 'seen'),
    (36.8718, 30.64455, 'Antalya', 'seen'),
    (36.87278, 30.64686, 'Antalya', 'seen'),
    (36.8744, 30.6359, 'Antalya', 'seen'),
    (36.87542, 30.65904, 'Antalya', 'seen'),
    (36.8757, 30.63471, 'Antalya', 'seen'),
    (36.87688, 30.72018, 'Antalya', 'seen'),
    (36.87711, 30.74272, 'Antalya', 'seen'),
    (36.87769, 30.72524, 'Antalya', 'seen'),
    (36.87779, 30.73203, 'Antalya', 'seen'),
    (36.87881, 30.63924, 'Antalya', 'seen'),
    (36.87913, 30.6617, 'Antalya', 'seen'),
    (36.88043, 30.70651, 'Antalya', 'seen'),
    (36.88105, 30.63987, 'Antalya', 'seen'),
    (36.88109, 30.74718, 'Antalya', 'seen'),
    (36.88147, 30.72381, 'Antalya', 'seen'),
    (36.88226, 30.64188, 'Antalya', 'seen'),
    (36.88228, 30.70738, 'Antalya', 'seen'),
    (36.88358, 30.63231, 'Antalya', 'seen'),
    (36.88375, 30.63919, 'Antalya', 'seen'),
    (36.88403, 30.67666, 'Antalya', 'seen'),
    (36.88468, 30.73783, 'Antalya', 'seen'),
    (36.88482, 30.68623, 'Antalya', 'seen'),
    (36.8854, 30.64951, 'Antalya', 'seen'),
    (36.88695, 30.65468, 'Antalya', 'seen'),
    (36.88712, 30.70948, 'Antalya', 'seen'),
    (36.89017, 30.68103, 'Antalya', 'seen'),
    (36.89059, 30.66281, 'Antalya', 'seen'),
    (36.89208, 30.67561, 'Antalya', 'seen'),
    (36.89218, 30.71715, 'Antalya', 'seen'),
    (36.8925, 30.66168, 'Antalya', 'seen'),
    (36.89388, 30.68785, 'Antalya', 'seen'),
    (36.89406, 30.74584, 'Antalya', 'seen'),
    (36.89468, 30.71929, 'Antalya', 'seen'),
    (36.89482, 30.67274, 'Antalya', 'seen'),
    (36.89496, 30.66112, 'Antalya', 'seen'),
    (36.89497, 30.66176, 'Antalya', 'seen'),
    (36.895, 30.72676, 'Antalya', 'seen'),
    (36.89537, 30.73892, 'Antalya', 'seen'),
    (36.89631, 30.71133, 'Antalya', 'seen'),
    (36.89681, 30.73429, 'Antalya', 'seen'),
    (36.89689, 30.63262, 'Antalya', 'seen'),
    (36.89716, 30.63404, 'Antalya', 'seen'),
    (36.89779, 30.7348, 'Antalya', 'seen'),
    (36.89841, 30.6854, 'Antalya', 'seen'),
    (36.89853, 30.65628, 'Antalya', 'seen'),
    (36.8986, 30.68667, 'Antalya', 'seen'),
    (36.89936, 30.72517, 'Antalya', 'seen'),
    (36.90002, 30.69458, 'Antalya', 'seen'),
    (36.90013, 30.70609, 'Antalya', 'seen'),
    (36.90072, 30.72521, 'Antalya', 'seen'),
    (36.90125, 30.64606, 'Antalya', 'seen'),
    (36.90256, 30.67291, 'Antalya', 'seen'),
    (36.90296, 30.70014, 'Antalya', 'seen'),
    (36.90306, 30.67095, 'Antalya', 'seen'),
    (36.90309, 30.68519, 'Antalya', 'seen'),
    (36.90364, 30.73456, 'Antalya', 'seen'),
    (36.90377, 30.65703, 'Antalya', 'seen'),
    (36.90403, 30.6409, 'Antalya', 'seen'),
    (36.90408, 30.63418, 'Antalya', 'seen'),
    (36.90415, 30.69041, 'Antalya', 'seen'),
    (36.90502, 30.73068, 'Antalya', 'seen'),
    (36.90668, 30.69122, 'Antalya', 'seen'),
    (36.90714, 30.63971, 'Antalya', 'seen'),
    (36.90892, 30.68259, 'Antalya', 'seen'),
    (36.90984, 30.63723, 'Antalya', 'seen'),
    (36.90991, 30.73939, 'Antalya', 'seen'),
    (36.91048, 30.63907, 'Antalya', 'seen'),
    (36.91098, 30.73762, 'Antalya', 'seen'),
    (36.91101, 30.72886, 'Antalya', 'seen'),
    (36.91125, 30.74081, 'Antalya', 'seen'),
    (36.91164, 30.69626, 'Antalya', 'seen'),
    (36.91222, 30.71424, 'Antalya', 'seen'),
    (36.91276, 30.74943, 'Antalya', 'seen'),
    (36.91299, 30.63614, 'Antalya', 'seen'),
    (36.91325, 30.71894, 'Antalya', 'seen'),
    (36.91345, 30.71708, 'Antalya', 'seen'),
    (36.91529, 30.68728, 'Antalya', 'seen'),
    (36.91644, 30.6936, 'Antalya', 'seen'),
    (36.91709, 30.71138, 'Antalya', 'seen'),
    (36.91722, 30.68911, 'Antalya', 'seen'),
    (36.91889, 30.64468, 'Antalya', 'seen'),
    (36.91929, 30.68172, 'Antalya', 'seen'),
    (36.91955, 30.72878, 'Antalya', 'seen'),
    (36.91989, 30.71934, 'Antalya', 'seen'),
    (37.94075, 23.69647, 'Athens', 'seen'),
    (37.94102, 23.73678, 'Athens', 'seen'),
    (37.9412, 23.7335, 'Athens', 'seen'),
    (37.94126, 23.71727, 'Athens', 'seen'),
    (37.94155, 23.70277, 'Athens', 'seen'),
    (37.94155, 23.74172, 'Athens', 'seen'),
    (37.94231, 23.73771, 'Athens', 'seen'),
    (37.94232, 23.77769, 'Athens', 'seen'),
    (37.94272, 23.73119, 'Athens', 'seen'),
    (37.9428, 23.74797, 'Athens', 'seen'),
    (37.94282, 23.73292, 'Athens', 'seen'),
    (37.94319, 23.72823, 'Athens', 'seen'),
    (37.94327, 23.76036, 'Athens', 'seen'),
    (37.94417, 23.68373, 'Athens', 'seen'),
    (37.94456, 23.68714, 'Athens', 'seen'),
    (37.94508, 23.72333, 'Athens', 'seen'),
    (37.94628, 23.7315, 'Athens', 'seen'),
    (37.94807, 23.77423, 'Athens', 'seen'),
    (37.94818, 23.70723, 'Athens', 'seen'),
    (37.9486, 23.76346, 'Athens', 'seen'),
    (37.94911, 23.70348, 'Athens', 'seen'),
    (37.94992, 23.74316, 'Athens', 'seen'),
    (37.95071, 23.76152, 'Athens', 'seen'),
    (37.95224, 23.73513, 'Athens', 'seen'),
    (37.95292, 23.69359, 'Athens', 'seen'),
    (37.95392, 23.77197, 'Athens', 'seen'),
    (37.95449, 23.72767, 'Athens', 'seen'),
    (37.95449, 23.75622, 'Athens', 'seen'),
    (37.95534, 23.74085, 'Athens', 'seen'),
    (37.95633, 23.69732, 'Athens', 'seen'),
    (37.95746, 23.73012, 'Athens', 'seen'),
    (37.9582, 23.75876, 'Athens', 'seen'),
    (37.95978, 23.69706, 'Athens', 'seen'),
    (37.96118, 23.74659, 'Athens', 'seen'),
    (37.96157, 23.72176, 'Athens', 'seen'),
    (37.96281, 23.68341, 'Athens', 'seen'),
    (37.96294, 23.74967, 'Athens', 'seen'),
    (37.96353, 23.69804, 'Athens', 'seen'),
    (37.96385, 23.74971, 'Athens', 'seen'),
    (37.96493, 23.73464, 'Athens', 'seen'),
    (37.96507, 23.73591, 'Athens', 'seen'),
    (37.9657, 23.70713, 'Athens', 'seen'),
    (37.96655, 23.70369, 'Athens', 'seen'),
    (37.96753, 23.69186, 'Athens', 'seen'),
    (37.96787, 23.73534, 'Athens', 'seen'),
    (37.96852, 23.76094, 'Athens', 'seen'),
    (37.96943, 23.72777, 'Athens', 'seen'),
    (37.97093, 23.75877, 'Athens', 'seen'),
    (37.971, 23.71468, 'Athens', 'seen'),
    (37.97161, 23.68929, 'Athens', 'seen'),
    (37.97175, 23.73235, 'Athens', 'seen'),
    (37.97196, 23.73396, 'Athens', 'seen'),
    (37.97198, 23.71638, 'Athens', 'seen'),
    (37.9753, 23.70331, 'Athens', 'seen'),
    (37.97593, 23.71585, 'Athens', 'seen'),
    (37.97615, 23.68678, 'Athens', 'seen'),
    (37.97779, 23.71099, 'Athens', 'seen'),
    (37.97794, 23.74901, 'Athens', 'seen'),
    (37.97803, 23.71543, 'Athens', 'seen'),
    (37.97984, 23.71408, 'Athens', 'seen'),
    (37.98064, 23.77768, 'Athens', 'seen'),
    (37.98133, 23.73885, 'Athens', 'seen'),
    (37.98156, 23.68114, 'Athens', 'seen'),
    (37.98323, 23.70565, 'Athens', 'seen'),
    (37.98324, 23.7138, 'Athens', 'seen'),
    (37.98433, 23.78765, 'Athens', 'seen'),
    (37.98444, 23.75454, 'Athens', 'seen'),
    (37.98502, 23.77135, 'Athens', 'seen'),
    (37.98503, 23.77539, 'Athens', 'seen'),
    (37.98541, 23.72579, 'Athens', 'seen'),
    (37.98655, 23.7127, 'Athens', 'seen'),
    (37.98717, 23.71061, 'Athens', 'seen'),
    (37.98752, 23.6896, 'Athens', 'seen'),
    (37.98803, 23.72703, 'Athens', 'seen'),
    (37.98874, 23.7623, 'Athens', 'seen'),
    (37.989, 23.74229, 'Athens', 'seen'),
    (37.98992, 23.72231, 'Athens', 'seen'),
    (37.99086, 23.73656, 'Athens', 'seen'),
    (37.99106, 23.77118, 'Athens', 'seen'),
    (37.99135, 23.7341, 'Athens', 'seen'),
    (37.99227, 23.69029, 'Athens', 'seen'),
    (37.99392, 23.74565, 'Athens', 'seen'),
    (37.99396, 23.74505, 'Athens', 'seen'),
    (37.99442, 23.72231, 'Athens', 'seen'),
    (37.99517, 23.70576, 'Athens', 'seen'),
    (37.99542, 23.68376, 'Athens', 'seen'),
    (37.99573, 23.72004, 'Athens', 'seen'),
    (37.99575, 23.76345, 'Athens', 'seen'),
    (37.9968, 23.75461, 'Athens', 'seen'),
    (37.99692, 23.68263, 'Athens', 'seen'),
    (37.99727, 23.78219, 'Athens', 'seen'),
    (37.99767, 23.68742, 'Athens', 'seen'),
    (37.99779, 23.73744, 'Athens', 'seen'),
    (37.99792, 23.76819, 'Athens', 'seen'),
    (37.99801, 23.78741, 'Athens', 'seen'),
    (37.99812, 23.68761, 'Athens', 'seen'),
    (37.99841, 23.73939, 'Athens', 'seen'),
    (37.99888, 23.71361, 'Athens', 'seen'),
    (37.99905, 23.69855, 'Athens', 'seen'),
    (38.00022, 23.68643, 'Athens', 'seen'),
    (38.00051, 23.71915, 'Athens', 'seen'),
    (38.00074, 23.68743, 'Athens', 'seen'),
    (38.0014, 23.71965, 'Athens', 'seen'),
    (38.00262, 23.717, 'Athens', 'seen'),
    (38.00389, 23.68086, 'Athens', 'seen'),
    (38.00423, 23.76643, 'Athens', 'seen'),
    (38.00442, 23.75851, 'Athens', 'seen'),
    (38.00467, 23.70606, 'Athens', 'seen'),
    (38.00602, 23.76002, 'Athens', 'seen'),
    (38.00658, 23.72349, 'Athens', 'seen'),
    (38.00677, 23.74113, 'Athens', 'seen'),
    (38.00724, 23.71502, 'Athens', 'seen'),
    (38.00818, 23.73092, 'Athens', 'seen'),
    (38.00851, 23.7607, 'Athens', 'seen'),
    (38.00858, 23.70959, 'Athens', 'seen'),
    (38.00878, 23.7583, 'Athens', 'seen'),
    (38.0088, 23.74628, 'Athens', 'seen'),
    (38.00947, 23.75193, 'Athens', 'seen'),
    (38.00988, 23.76065, 'Athens', 'seen'),
    (41.35279, 2.13602, 'Barcelona', 'seen'),
    (41.35284, 2.10787, 'Barcelona', 'seen'),
    (41.35445, 2.14781, 'Barcelona', 'seen'),
    (41.35464, 2.1377, 'Barcelona', 'seen'),
    (41.35657, 2.14565, 'Barcelona', 'seen'),
    (41.35779, 2.11334, 'Barcelona', 'seen'),
    (41.35819, 2.12229, 'Barcelona', 'seen'),
    (41.35978, 2.1482, 'Barcelona', 'seen'),
    (41.36264, 2.12316, 'Barcelona', 'seen'),
    (41.36446, 2.08678, 'Barcelona', 'seen'),
    (41.36701, 2.11033, 'Barcelona', 'seen'),
    (41.36706, 2.17474, 'Barcelona', 'seen'),
    (41.36734, 2.12903, 'Barcelona', 'seen'),
    (41.36844, 2.18745, 'Barcelona', 'seen'),
    (41.36845, 2.15801, 'Barcelona', 'seen'),
    (41.36857, 2.11477, 'Barcelona', 'seen'),
    (41.37022, 2.11327, 'Barcelona', 'seen'),
    (41.37024, 2.13543, 'Barcelona', 'seen'),
    (41.37087, 2.08468, 'Barcelona', 'seen'),
    (41.37104, 2.13464, 'Barcelona', 'seen'),
    (41.37281, 2.13165, 'Barcelona', 'seen'),
    (41.37349, 2.14751, 'Barcelona', 'seen'),
    (41.37369, 2.12608, 'Barcelona', 'seen'),
    (41.37419, 2.1756, 'Barcelona', 'seen'),
    (41.3754, 2.1355, 'Barcelona', 'seen'),
    (41.37646, 2.16605, 'Barcelona', 'seen'),
    (41.37676, 2.15849, 'Barcelona', 'seen'),
    (41.37914, 2.12776, 'Barcelona', 'seen'),
    (41.38053, 2.12575, 'Barcelona', 'seen'),
    (41.3817, 2.13329, 'Barcelona', 'seen'),
    (41.38335, 2.10774, 'Barcelona', 'seen'),
    (41.38397, 2.11694, 'Barcelona', 'seen'),
    (41.38447, 2.09422, 'Barcelona', 'seen'),
    (41.3858, 2.15823, 'Barcelona', 'seen'),
    (41.38786, 2.11557, 'Barcelona', 'seen'),
    (41.38844, 2.10357, 'Barcelona', 'seen'),
    (41.38905, 2.0859, 'Barcelona', 'seen'),
    (41.39032, 2.13762, 'Barcelona', 'seen'),
    (41.39067, 2.13132, 'Barcelona', 'seen'),
    (41.39075, 2.12057, 'Barcelona', 'seen'),
    (41.39103, 2.19337, 'Barcelona', 'seen'),
    (41.39309, 2.20313, 'Barcelona', 'seen'),
    (41.3948, 2.11028, 'Barcelona', 'seen'),
    (41.39564, 2.14406, 'Barcelona', 'seen'),
    (41.39637, 2.1031, 'Barcelona', 'seen'),
    (41.39654, 2.14548, 'Barcelona', 'seen'),
    (41.3977, 2.12905, 'Barcelona', 'seen'),
    (41.39798, 2.17827, 'Barcelona', 'seen'),
    (41.39915, 2.13405, 'Barcelona', 'seen'),
    (41.40025, 2.21092, 'Barcelona', 'seen'),
    (41.40028, 2.17746, 'Barcelona', 'seen'),
    (41.4009, 2.13598, 'Barcelona', 'seen'),
    (41.40147, 2.16915, 'Barcelona', 'seen'),
    (41.40201, 2.19894, 'Barcelona', 'seen'),
    (41.4025, 2.12, 'Barcelona', 'seen'),
    (41.40571, 2.15743, 'Barcelona', 'seen'),
    (41.40614, 2.18901, 'Barcelona', 'seen'),
    (41.40631, 2.21751, 'Barcelona', 'seen'),
    (41.40644, 2.10617, 'Barcelona', 'seen'),
    (41.40721, 2.08291, 'Barcelona', 'seen'),
    (41.40764, 2.11338, 'Barcelona', 'seen'),
    (41.40867, 2.13511, 'Barcelona', 'seen'),
    (41.40878, 2.18208, 'Barcelona', 'seen'),
    (41.40879, 2.08626, 'Barcelona', 'seen'),
    (41.40894, 2.18187, 'Barcelona', 'seen'),
    (41.40927, 2.15558, 'Barcelona', 'seen'),
    (41.40949, 2.08199, 'Barcelona', 'seen'),
    (41.41071, 2.15282, 'Barcelona', 'seen'),
    (41.41263, 2.1809, 'Barcelona', 'seen'),
    (41.41451, 2.17197, 'Barcelona', 'seen'),
    (41.41477, 2.12976, 'Barcelona', 'seen'),
    (41.41743, 2.14113, 'Barcelona', 'seen'),
    (41.41859, 2.08536, 'Barcelona', 'seen'),
    (41.41993, 2.15502, 'Barcelona', 'seen'),
    (41.4201, 2.16613, 'Barcelona', 'seen'),
    (41.42024, 2.20716, 'Barcelona', 'seen'),
    (41.42031, 2.14839, 'Barcelona', 'seen'),
    (41.42063, 2.21911, 'Barcelona', 'seen'),
    (41.42081, 2.11822, 'Barcelona', 'seen'),
    (41.42226, 2.1613, 'Barcelona', 'seen'),
    (41.42226, 2.1613, 'Barcelona', 'seen'),
    (41.42238, 2.08865, 'Barcelona', 'seen'),
    (41.42327, 2.17683, 'Barcelona', 'seen'),
    (41.42465, 2.15858, 'Barcelona', 'seen'),
    (41.42548, 2.17712, 'Barcelona', 'seen'),
    (41.42606, 2.13944, 'Barcelona', 'seen'),
    (41.42774, 2.1739, 'Barcelona', 'seen'),
    (41.42776, 2.16015, 'Barcelona', 'seen'),
    (41.42793, 2.0964, 'Barcelona', 'seen'),
    (41.4282, 2.16661, 'Barcelona', 'seen'),
    (41.42859, 2.13354, 'Barcelona', 'seen'),
    (41.42977, 2.19845, 'Barcelona', 'seen'),
    (41.43145, 2.12646, 'Barcelona', 'seen'),
    (41.4315, 2.13425, 'Barcelona', 'seen'),
    (41.43224, 2.08383, 'Barcelona', 'seen'),
    (41.43224, 2.08383, 'Barcelona', 'seen'),
    (41.43241, 2.18641, 'Barcelona', 'seen'),
    (41.43303, 2.13365, 'Barcelona', 'seen'),
    (41.43342, 2.17869, 'Barcelona', 'seen'),
    (41.43585, 2.09006, 'Barcelona', 'seen'),
    (41.43863, 2.19944, 'Barcelona', 'seen'),
    (41.43878, 2.17168, 'Barcelona', 'seen'),
    (41.43973, 2.08158, 'Barcelona', 'seen'),
    (41.44158, 2.17664, 'Barcelona', 'seen'),
    (41.44198, 2.17963, 'Barcelona', 'seen'),
    (41.44219, 2.08335, 'Barcelona', 'seen'),
    (41.44233, 2.18321, 'Barcelona', 'seen'),
    (41.44443, 2.14813, 'Barcelona', 'seen'),
    (41.44514, 2.1995, 'Barcelona', 'seen'),
    (41.44536, 2.2091, 'Barcelona', 'seen'),
    (41.44574, 2.18605, 'Barcelona', 'seen'),
    (41.44593, 2.21041, 'Barcelona', 'seen'),
    (41.44644, 2.09577, 'Barcelona', 'seen'),
    (41.44744, 2.13823, 'Barcelona', 'seen'),
    (41.44763, 2.17106, 'Barcelona', 'seen'),
    (41.44823, 2.15474, 'Barcelona', 'seen'),
    (41.44841, 2.17149, 'Barcelona', 'seen'),
    (41.44869, 2.19067, 'Barcelona', 'seen'),
    (41.45115, 2.12066, 'Barcelona', 'seen'),
    (52.40474, 13.39175, 'Berlin', 'seen'),
    (52.40567, 13.20954, 'Berlin', 'seen'),
    (52.40846, 13.3623, 'Berlin', 'seen'),
    (52.40928, 13.27108, 'Berlin', 'seen'),
    (52.41371, 13.35328, 'Berlin', 'seen'),
    (52.41534, 13.44426, 'Berlin', 'seen'),
    (52.41551, 13.3401, 'Berlin', 'seen'),
    (52.41657, 13.5273, 'Berlin', 'seen'),
    (52.41728, 13.40773, 'Berlin', 'seen'),
    (52.42078, 13.41274, 'Berlin', 'seen'),
    (52.42456, 13.54618, 'Berlin', 'seen'),
    (52.42604, 13.40162, 'Berlin', 'seen'),
    (52.4275, 13.2701, 'Berlin', 'seen'),
    (52.42819, 13.26004, 'Berlin', 'seen'),
    (52.42885, 13.52646, 'Berlin', 'seen'),
    (52.42974, 13.54957, 'Berlin', 'seen'),
    (52.43747, 13.26955, 'Berlin', 'seen'),
    (52.43935, 13.44907, 'Berlin', 'seen'),
    (52.4395, 13.4266, 'Berlin', 'seen'),
    (52.44099, 13.35085, 'Berlin', 'seen'),
    (52.4413, 13.29853, 'Berlin', 'seen'),
    (52.44206, 13.29411, 'Berlin', 'seen'),
    (52.44286, 13.22631, 'Berlin', 'seen'),
    (52.44391, 13.25906, 'Berlin', 'seen'),
    (52.44408, 13.25943, 'Berlin', 'seen'),
    (52.44446, 13.57383, 'Berlin', 'seen'),
    (52.44596, 13.2672, 'Berlin', 'seen'),
    (52.44727, 13.40709, 'Berlin', 'seen'),
    (52.44949, 13.3422, 'Berlin', 'seen'),
    (52.45131, 13.48184, 'Berlin', 'seen'),
    (52.4515, 13.57293, 'Berlin', 'seen'),
    (52.45162, 13.58569, 'Berlin', 'seen'),
    (52.45426, 13.35611, 'Berlin', 'seen'),
    (52.45434, 13.45359, 'Berlin', 'seen'),
    (52.45434, 13.46353, 'Berlin', 'seen'),
    (52.45513, 13.41699, 'Berlin', 'seen'),
    (52.45523, 13.3505, 'Berlin', 'seen'),
    (52.45887, 13.53206, 'Berlin', 'seen'),
    (52.45892, 13.3981, 'Berlin', 'seen'),
    (52.45983, 13.58955, 'Berlin', 'seen'),
    (52.46376, 13.47179, 'Berlin', 'seen'),
    (52.46409, 13.4099, 'Berlin', 'seen'),
    (52.46443, 13.32065, 'Berlin', 'seen'),
    (52.46713, 13.44211, 'Berlin', 'seen'),
    (52.46832, 13.49402, 'Berlin', 'seen'),
    (52.46844, 13.29177, 'Berlin', 'seen'),
    (52.47236, 13.43265, 'Berlin', 'seen'),
    (52.4725, 13.45851, 'Berlin', 'seen'),
    (52.47407, 13.44389, 'Berlin', 'seen'),
    (52.47567, 13.42232, 'Berlin', 'seen'),
    (52.47923, 13.47421, 'Berlin', 'seen'),
    (52.47937, 13.59379, 'Berlin', 'seen'),
    (52.48372, 13.56842, 'Berlin', 'seen'),
    (52.4838, 13.36265, 'Berlin', 'seen'),
    (52.48662, 13.58258, 'Berlin', 'seen'),
    (52.49127, 13.491, 'Berlin', 'seen'),
    (52.49177, 13.37787, 'Berlin', 'seen'),
    (52.49207, 13.28087, 'Berlin', 'seen'),
    (52.49836, 13.50185, 'Berlin', 'seen'),
    (52.4995, 13.20691, 'Berlin', 'seen'),
    (52.50039, 13.37428, 'Berlin', 'seen'),
    (52.50265, 13.59126, 'Berlin', 'seen'),
    (52.50336, 13.44732, 'Berlin', 'seen'),
    (52.50482, 13.55928, 'Berlin', 'seen'),
    (52.50619, 13.43095, 'Berlin', 'seen'),
    (52.50649, 13.44233, 'Berlin', 'seen'),
    (52.50655, 13.28562, 'Berlin', 'seen'),
    (52.51615, 13.42916, 'Berlin', 'seen'),
    (52.51889, 13.41123, 'Berlin', 'seen'),
    (52.52146, 13.54116, 'Berlin', 'seen'),
    (52.52147, 13.24695, 'Berlin', 'seen'),
    (52.52396, 13.36721, 'Berlin', 'seen'),
    (52.52484, 13.20545, 'Berlin', 'seen'),
    (52.52835, 13.5716, 'Berlin', 'seen'),
    (52.52917, 13.30225, 'Berlin', 'seen'),
    (52.53427, 13.54396, 'Berlin', 'seen'),
    (52.53452, 13.47903, 'Berlin', 'seen'),
    (52.53455, 13.5053, 'Berlin', 'seen'),
    (52.53642, 13.39753, 'Berlin', 'seen'),
    (52.53651, 13.2938, 'Berlin', 'seen'),
    (52.53668, 13.59926, 'Berlin', 'seen'),
    (52.53683, 13.59363, 'Berlin', 'seen'),
    (52.53822, 13.56887, 'Berlin', 'seen'),
    (52.54035, 13.57432, 'Berlin', 'seen'),
    (52.54127, 13.23379, 'Berlin', 'seen'),
    (52.54247, 13.41613, 'Berlin', 'seen'),
    (52.5464, 13.37588, 'Berlin', 'seen'),
    (52.54704, 13.46952, 'Berlin', 'seen'),
    (52.55223, 13.39551, 'Berlin', 'seen'),
    (52.55266, 13.40715, 'Berlin', 'seen'),
    (52.55503, 13.34077, 'Berlin', 'seen'),
    (52.56036, 13.40575, 'Berlin', 'seen'),
    (52.56497, 13.53097, 'Berlin', 'seen'),
    (52.56498, 13.51165, 'Berlin', 'seen'),
    (52.56546, 13.58846, 'Berlin', 'seen'),
    (52.5714, 13.45101, 'Berlin', 'seen'),
    (52.57288, 13.38335, 'Berlin', 'seen'),
    (52.57591, 13.26641, 'Berlin', 'seen'),
    (52.57846, 13.31662, 'Berlin', 'seen'),
    (52.57954, 13.44954, 'Berlin', 'seen'),
    (52.58296, 13.28355, 'Berlin', 'seen'),
    (52.58372, 13.46452, 'Berlin', 'seen'),
    (52.58442, 13.2932, 'Berlin', 'seen'),
    (52.5863, 13.39146, 'Berlin', 'seen'),
    (52.58757, 13.43126, 'Berlin', 'seen'),
    (52.58822, 13.55848, 'Berlin', 'seen'),
    (52.58923, 13.23018, 'Berlin', 'seen'),
    (52.59006, 13.43128, 'Berlin', 'seen'),
    (52.59491, 13.50101, 'Berlin', 'seen'),
    (50.82005, 4.3334, 'Brussels', 'seen'),
    (50.82087, 4.35272, 'Brussels', 'seen'),
    (50.82089, 4.40432, 'Brussels', 'seen'),
    (50.82112, 4.37235, 'Brussels', 'seen'),
    (50.82152, 4.4021, 'Brussels', 'seen'),
    (50.82181, 4.39184, 'Brussels', 'seen'),
    (50.82189, 4.37565, 'Brussels', 'seen'),
    (50.82195, 4.32996, 'Brussels', 'seen'),
    (50.82229, 4.35758, 'Brussels', 'seen'),
    (50.8225, 4.36143, 'Brussels', 'seen'),
    (50.82377, 4.33904, 'Brussels', 'seen'),
    (50.82379, 4.37616, 'Brussels', 'seen'),
    (50.8241, 4.3596, 'Brussels', 'seen'),
    (50.8246, 4.34828, 'Brussels', 'seen'),
    (50.82474, 4.33659, 'Brussels', 'seen'),
    (50.8254, 4.39173, 'Brussels', 'seen'),
    (50.82579, 4.40965, 'Brussels', 'seen'),
    (50.82583, 4.35003, 'Brussels', 'seen'),
    (50.82667, 4.35312, 'Brussels', 'seen'),
    (50.82678, 4.39839, 'Brussels', 'seen'),
    (50.82769, 4.40294, 'Brussels', 'seen'),
    (50.82802, 4.35303, 'Brussels', 'seen'),
    (50.82803, 4.35958, 'Brussels', 'seen'),
    (50.82847, 4.35734, 'Brussels', 'seen'),
    (50.83018, 4.3963, 'Brussels', 'seen'),
    (50.83082, 4.36514, 'Brussels', 'seen'),
    (50.83108, 4.33897, 'Brussels', 'seen'),
    (50.83134, 4.3789, 'Brussels', 'seen'),
    (50.83151, 4.3854, 'Brussels', 'seen'),
    (50.83155, 4.32267, 'Brussels', 'seen'),
    (50.83159, 4.38961, 'Brussels', 'seen'),
    (50.8321, 4.35858, 'Brussels', 'seen'),
    (50.8332, 4.40978, 'Brussels', 'seen'),
    (50.83327, 4.3998, 'Brussels', 'seen'),
    (50.83381, 4.33156, 'Brussels', 'seen'),
    (50.83449, 4.36176, 'Brussels', 'seen'),
    (50.83471, 4.33882, 'Brussels', 'seen'),
    (50.83478, 4.39568, 'Brussels', 'seen'),
    (50.83518, 4.3333, 'Brussels', 'seen'),
    (50.83554, 4.36981, 'Brussels', 'seen'),
    (50.83581, 4.36486, 'Brussels', 'seen'),
    (50.83615, 4.32659, 'Brussels', 'seen'),
    (50.83745, 4.35239, 'Brussels', 'seen'),
    (50.83767, 4.33411, 'Brussels', 'seen'),
    (50.83802, 4.37516, 'Brussels', 'seen'),
    (50.83847, 4.40168, 'Brussels', 'seen'),
    (50.83881, 4.38515, 'Brussels', 'seen'),
    (50.83946, 4.40836, 'Brussels', 'seen'),
    (50.84031, 4.39728, 'Brussels', 'seen'),
    (50.84067, 4.33866, 'Brussels', 'seen'),
    (50.84201, 4.37095, 'Brussels', 'seen'),
    (50.84207, 4.33649, 'Brussels', 'seen'),
    (50.84272, 4.36221, 'Brussels', 'seen'),
    (50.84274, 4.346, 'Brussels', 'seen'),
    (50.84349, 4.3231, 'Brussels', 'seen'),
    (50.84357, 4.3468, 'Brussels', 'seen'),
    (50.84374, 4.32795, 'Brussels', 'seen'),
    (50.84392, 4.37462, 'Brussels', 'seen'),
    (50.84393, 4.39724, 'Brussels', 'seen'),
    (50.84437, 4.40144, 'Brussels', 'seen'),
    (50.84473, 4.33311, 'Brussels', 'seen'),
    (50.84489, 4.39719, 'Brussels', 'seen'),
    (50.84502, 4.38582, 'Brussels', 'seen'),
    (50.84538, 4.39662, 'Brussels', 'seen'),
    (50.84553, 4.39099, 'Brussels', 'seen'),
    (50.84568, 4.38959, 'Brussels', 'seen'),
    (50.84575, 4.38961, 'Brussels', 'seen'),
    (50.84579, 4.39788, 'Brussels', 'seen'),
    (50.84591, 4.37922, 'Brussels', 'seen'),
    (50.84604, 4.34963, 'Brussels', 'seen'),
    (50.84642, 4.36938, 'Brussels', 'seen'),
    (50.84645, 4.33168, 'Brussels', 'seen'),
    (50.84664, 4.33495, 'Brussels', 'seen'),
    (50.84702, 4.34433, 'Brussels', 'seen'),
    (50.84843, 4.3609, 'Brussels', 'seen'),
    (50.84925, 4.32031, 'Brussels', 'seen'),
    (50.84945, 4.35313, 'Brussels', 'seen'),
    (50.84979, 4.39207, 'Brussels', 'seen'),
    (50.85, 4.32956, 'Brussels', 'seen'),
    (50.8502, 4.32307, 'Brussels', 'seen'),
    (50.85059, 4.32987, 'Brussels', 'seen'),
    (50.85088, 4.38977, 'Brussels', 'seen'),
    (50.85131, 4.36176, 'Brussels', 'seen'),
    (50.85164, 4.32469, 'Brussels', 'seen'),
    (50.85177, 4.36503, 'Brussels', 'seen'),
    (50.85182, 4.38555, 'Brussels', 'seen'),
    (50.85272, 4.39486, 'Brussels', 'seen'),
    (50.85299, 4.38808, 'Brussels', 'seen'),
    (50.85353, 4.33554, 'Brussels', 'seen'),
    (50.85408, 4.32302, 'Brussels', 'seen'),
    (50.85435, 4.33285, 'Brussels', 'seen'),
    (50.85523, 4.35232, 'Brussels', 'seen'),
    (50.85531, 4.3786, 'Brussels', 'seen'),
    (50.85543, 4.34678, 'Brussels', 'seen'),
    (50.85543, 4.38116, 'Brussels', 'seen'),
    (50.85551, 4.39151, 'Brussels', 'seen'),
    (50.85563, 4.38744, 'Brussels', 'seen'),
    (50.85571, 4.35516, 'Brussels', 'seen'),
    (50.85663, 4.40221, 'Brussels', 'seen'),
    (50.8568, 4.33966, 'Brussels', 'seen'),
    (50.8578, 4.39044, 'Brussels', 'seen'),
    (50.85809, 4.35174, 'Brussels', 'seen'),
    (50.85912, 4.35153, 'Brussels', 'seen'),
    (50.85926, 4.32848, 'Brussels', 'seen'),
    (50.85976, 4.38639, 'Brussels', 'seen'),
    (50.86032, 4.35333, 'Brussels', 'seen'),
    (50.86059, 4.3853, 'Brussels', 'seen'),
    (50.86172, 4.38351, 'Brussels', 'seen'),
    (50.86406, 4.33493, 'Brussels', 'seen'),
    (50.86471, 4.32172, 'Brussels', 'seen'),
    (50.865, 4.36947, 'Brussels', 'seen'),
    (50.8657, 4.33524, 'Brussels', 'seen'),
    (50.86597, 4.36056, 'Brussels', 'seen'),
    (50.86786, 4.34758, 'Brussels', 'seen'),
    (50.86854, 4.32459, 'Brussels', 'seen'),
    (50.86974, 4.40207, 'Brussels', 'seen'),
    (50.87121, 4.35539, 'Brussels', 'seen'),
    (47.45996, 19.11022, 'Budapest', 'seen'),
    (47.46011, 19.11752, 'Budapest', 'seen'),
    (47.46101, 19.0833, 'Budapest', 'seen'),
    (47.46106, 19.03579, 'Budapest', 'seen'),
    (47.46195, 19.0909, 'Budapest', 'seen'),
    (47.4626, 19.03352, 'Budapest', 'seen'),
    (47.46284, 19.04222, 'Budapest', 'seen'),
    (47.46312, 19.08983, 'Budapest', 'seen'),
    (47.46365, 19.04633, 'Budapest', 'seen'),
    (47.46389, 19.07316, 'Budapest', 'seen'),
    (47.46402, 19.05296, 'Budapest', 'seen'),
    (47.46494, 19.09966, 'Budapest', 'seen'),
    (47.466, 19.05451, 'Budapest', 'seen'),
    (47.46678, 19.02536, 'Budapest', 'seen'),
    (47.46754, 19.08531, 'Budapest', 'seen'),
    (47.46779, 19.09995, 'Budapest', 'seen'),
    (47.46792, 19.11602, 'Budapest', 'seen'),
    (47.46879, 19.02753, 'Budapest', 'seen'),
    (47.46891, 19.0705, 'Budapest', 'seen'),
    (47.46925, 19.07582, 'Budapest', 'seen'),
    (47.47003, 19.10173, 'Budapest', 'seen'),
    (47.47005, 19.09174, 'Budapest', 'seen'),
    (47.47042, 19.0646, 'Budapest', 'seen'),
    (47.47068, 19.03341, 'Budapest', 'seen'),
    (47.47076, 19.02261, 'Budapest', 'seen'),
    (47.47077, 19.02728, 'Budapest', 'seen'),
    (47.47104, 19.07415, 'Budapest', 'seen'),
    (47.47117, 19.0853, 'Budapest', 'seen'),
    (47.47297, 19.02779, 'Budapest', 'seen'),
    (47.47308, 19.09547, 'Budapest', 'seen'),
    (47.4732, 19.10413, 'Budapest', 'seen'),
    (47.47324, 19.06366, 'Budapest', 'seen'),
    (47.47369, 19.11626, 'Budapest', 'seen'),
    (47.47377, 19.11119, 'Budapest', 'seen'),
    (47.47453, 19.08803, 'Budapest', 'seen'),
    (47.47478, 19.02682, 'Budapest', 'seen'),
    (47.47648, 19.0588, 'Budapest', 'seen'),
    (47.47704, 19.09104, 'Budapest', 'seen'),
    (47.47791, 19.07287, 'Budapest', 'seen'),
    (47.47798, 19.05874, 'Budapest', 'seen'),
    (47.47869, 19.08297, 'Budapest', 'seen'),
    (47.47902, 19.10142, 'Budapest', 'seen'),
    (47.47902, 19.10142, 'Budapest', 'seen'),
    (47.47952, 19.05276, 'Budapest', 'seen'),
    (47.47953, 19.08101, 'Budapest', 'seen'),
    (47.48007, 19.07143, 'Budapest', 'seen'),
    (47.48008, 19.05871, 'Budapest', 'seen'),
    (47.4805, 19.10851, 'Budapest', 'seen'),
    (47.48082, 19.11927, 'Budapest', 'seen'),
    (47.48094, 19.05773, 'Budapest', 'seen'),
    (47.48098, 19.05027, 'Budapest', 'seen'),
    (47.48174, 19.07775, 'Budapest', 'seen'),
    (47.48297, 19.10444, 'Budapest', 'seen'),
    (47.48416, 19.06305, 'Budapest', 'seen'),
    (47.4844, 19.08007, 'Budapest', 'seen'),
    (47.48514, 19.10672, 'Budapest', 'seen'),
    (47.48619, 19.04199, 'Budapest', 'seen'),
    (47.48658, 19.0654, 'Budapest', 'seen'),
    (47.48739, 19.06644, 'Budapest', 'seen'),
    (47.48748, 19.10398, 'Budapest', 'seen'),
    (47.48895, 19.09477, 'Budapest', 'seen'),
    (47.48923, 19.05978, 'Budapest', 'seen'),
    (47.4904, 19.05475, 'Budapest', 'seen'),
    (47.49156, 19.07577, 'Budapest', 'seen'),
    (47.49262, 19.02014, 'Budapest', 'seen'),
    (47.49269, 19.05808, 'Budapest', 'seen'),
    (47.49345, 19.0312, 'Budapest', 'seen'),
    (47.49392, 19.0237, 'Budapest', 'seen'),
    (47.49436, 19.03451, 'Budapest', 'seen'),
    (47.49668, 19.11119, 'Budapest', 'seen'),
    (47.49714, 19.11788, 'Budapest', 'seen'),
    (47.49788, 19.10768, 'Budapest', 'seen'),
    (47.49956, 19.09686, 'Budapest', 'seen'),
    (47.49963, 19.0875, 'Budapest', 'seen'),
    (47.49967, 19.07557, 'Budapest', 'seen'),
    (47.50084, 19.03949, 'Budapest', 'seen'),
    (47.50112, 19.07836, 'Budapest', 'seen'),
    (47.50118, 19.07, 'Budapest', 'seen'),
    (47.50183, 19.09052, 'Budapest', 'seen'),
    (47.50338, 19.106, 'Budapest', 'seen'),
    (47.50368, 19.10299, 'Budapest', 'seen'),
    (47.50398, 19.03753, 'Budapest', 'seen'),
    (47.50405, 19.06503, 'Budapest', 'seen'),
    (47.50417, 19.05964, 'Budapest', 'seen'),
    (47.5049, 19.0254, 'Budapest', 'seen'),
    (47.50493, 19.06288, 'Budapest', 'seen'),
    (47.50563, 19.1148, 'Budapest', 'seen'),
    (47.50696, 19.11068, 'Budapest', 'seen'),
    (47.50743, 19.10391, 'Budapest', 'seen'),
    (47.50799, 19.02936, 'Budapest', 'seen'),
    (47.50857, 19.06613, 'Budapest', 'seen'),
    (47.50979, 19.05374, 'Budapest', 'seen'),
    (47.51162, 19.08017, 'Budapest', 'seen'),
    (47.51165, 19.11013, 'Budapest', 'seen'),
    (47.512, 19.07132, 'Budapest', 'seen'),
    (47.51218, 19.05709, 'Budapest', 'seen'),
    (47.51231, 19.0563, 'Budapest', 'seen'),
    (47.51413, 19.02606, 'Budapest', 'seen'),
    (47.5168, 19.10175, 'Budapest', 'seen'),
    (47.5172, 19.10527, 'Budapest', 'seen'),
    (47.51756, 19.0309, 'Budapest', 'seen'),
    (47.51803, 19.03623, 'Budapest', 'seen'),
    (47.51993, 19.10003, 'Budapest', 'seen'),
    (47.52, 19.11136, 'Budapest', 'seen'),
    (47.52011, 19.11324, 'Budapest', 'seen'),
    (47.52016, 19.05892, 'Budapest', 'seen'),
    (47.52129, 19.0357, 'Budapest', 'seen'),
    (47.52145, 19.10622, 'Budapest', 'seen'),
    (47.52201, 19.11006, 'Budapest', 'seen'),
    (47.52364, 19.02992, 'Budapest', 'seen'),
    (47.5243, 19.11633, 'Budapest', 'seen'),
    (47.52488, 19.03036, 'Budapest', 'seen'),
    (47.52499, 19.11773, 'Budapest', 'seen'),
    (47.52619, 19.06973, 'Budapest', 'seen'),
    (47.52635, 19.05489, 'Budapest', 'seen'),
    (47.52843, 19.09789, 'Budapest', 'seen'),
    (47.52855, 19.07989, 'Budapest', 'seen'),
    (47.52867, 19.07092, 'Budapest', 'seen'),
    (47.5288, 19.11955, 'Budapest', 'seen'),
    (47.52937, 19.06325, 'Budapest', 'seen'),
    (-34.63941, -58.37221, 'Buenos Aires', 'seen'),
    (-34.63877, -58.42796, 'Buenos Aires', 'seen'),
    (-34.63876, -58.38126, 'Buenos Aires', 'seen'),
    (-34.63829, -58.41885, 'Buenos Aires', 'seen'),
    (-34.63826, -58.44363, 'Buenos Aires', 'seen'),
    (-34.6378, -58.43794, 'Buenos Aires', 'seen'),
    (-34.63749, -58.40758, 'Buenos Aires', 'seen'),
    (-34.63625, -58.36371, 'Buenos Aires', 'seen'),
    (-34.63612, -58.4431, 'Buenos Aires', 'seen'),
    (-34.63595, -58.41619, 'Buenos Aires', 'seen'),
    (-34.63562, -58.35709, 'Buenos Aires', 'seen'),
    (-34.63363, -58.38123, 'Buenos Aires', 'seen'),
    (-34.63295, -58.41783, 'Buenos Aires', 'seen'),
    (-34.63217, -58.43064, 'Buenos Aires', 'seen'),
    (-34.62991, -58.39164, 'Buenos Aires', 'seen'),
    (-34.62983, -58.43689, 'Buenos Aires', 'seen'),
    (-34.62972, -58.35532, 'Buenos Aires', 'seen'),
    (-34.62919, -58.39653, 'Buenos Aires', 'seen'),
    (-34.62833, -58.40848, 'Buenos Aires', 'seen'),
    (-34.62792, -58.42383, 'Buenos Aires', 'seen'),
    (-34.62781, -58.36885, 'Buenos Aires', 'seen'),
    (-34.62661, -58.36416, 'Buenos Aires', 'seen'),
    (-34.6262, -58.38806, 'Buenos Aires', 'seen'),
    (-34.6254, -58.37748, 'Buenos Aires', 'seen'),
    (-34.62506, -58.39904, 'Buenos Aires', 'seen'),
    (-34.62405, -58.34404, 'Buenos Aires', 'seen'),
    (-34.62384, -58.34721, 'Buenos Aires', 'seen'),
    (-34.6232, -58.40724, 'Buenos Aires', 'seen'),
    (-34.6229, -58.4406, 'Buenos Aires', 'seen'),
    (-34.62199, -58.36532, 'Buenos Aires', 'seen'),
    (-34.62097, -58.4496, 'Buenos Aires', 'seen'),
    (-34.62042, -58.40628, 'Buenos Aires', 'seen'),
    (-34.62021, -58.39164, 'Buenos Aires', 'seen'),
    (-34.61965, -58.37431, 'Buenos Aires', 'seen'),
    (-34.61862, -58.44647, 'Buenos Aires', 'seen'),
    (-34.61862, -58.38453, 'Buenos Aires', 'seen'),
    (-34.61842, -58.37017, 'Buenos Aires', 'seen'),
    (-34.61839, -58.39898, 'Buenos Aires', 'seen'),
    (-34.61808, -58.41269, 'Buenos Aires', 'seen'),
    (-34.61774, -58.35045, 'Buenos Aires', 'seen'),
    (-34.61747, -58.40813, 'Buenos Aires', 'seen'),
    (-34.61664, -58.41692, 'Buenos Aires', 'seen'),
    (-34.61589, -58.37057, 'Buenos Aires', 'seen'),
    (-34.61568, -58.41112, 'Buenos Aires', 'seen'),
    (-34.61537, -58.38307, 'Buenos Aires', 'seen'),
    (-34.6146, -58.42033, 'Buenos Aires', 'seen'),
    (-34.61446, -58.40184, 'Buenos Aires', 'seen'),
    (-34.61324, -58.41143, 'Buenos Aires', 'seen'),
    (-34.61244, -58.3875, 'Buenos Aires', 'seen'),
    (-34.61233, -58.36955, 'Buenos Aires', 'seen'),
    (-34.61172, -58.35853, 'Buenos Aires', 'seen'),
    (-34.6112, -58.41029, 'Buenos Aires', 'seen'),
    (-34.6105, -58.43676, 'Buenos Aires', 'seen'),
    (-34.60948, -58.36363, 'Buenos Aires', 'seen'),
    (-34.60806, -58.39367, 'Buenos Aires', 'seen'),
    (-34.60501, -58.36079, 'Buenos Aires', 'seen'),
    (-34.60368, -58.44814, 'Buenos Aires', 'seen'),
    (-34.60342, -58.40222, 'Buenos Aires', 'seen'),
    (-34.60299, -58.42284, 'Buenos Aires', 'seen'),
    (-34.60093, -58.39568, 'Buenos Aires', 'seen'),
    (-34.59957, -58.3714, 'Buenos Aires', 'seen'),
    (-34.59809, -58.36331, 'Buenos Aires', 'seen'),
    (-34.59809, -58.36331, 'Buenos Aires', 'seen'),
    (-34.59805, -58.40579, 'Buenos Aires', 'seen'),
    (-34.59773, -58.40351, 'Buenos Aires', 'seen'),
    (-34.59587, -58.37674, 'Buenos Aires', 'seen'),
    (-34.59501, -58.43811, 'Buenos Aires', 'seen'),
    (-34.59482, -58.382, 'Buenos Aires', 'seen'),
    (-34.5941, -58.41746, 'Buenos Aires', 'seen'),
    (-34.59271, -58.44505, 'Buenos Aires', 'seen'),
    (-34.59268, -58.39493, 'Buenos Aires', 'seen'),
    (-34.59217, -58.37463, 'Buenos Aires', 'seen'),
    (-34.59177, -58.40911, 'Buenos Aires', 'seen'),
    (-34.59161, -58.39216, 'Buenos Aires', 'seen'),
    (-34.59144, -58.39685, 'Buenos Aires', 'seen'),
    (-34.59001, -58.4282, 'Buenos Aires', 'seen'),
    (-34.58955, -58.3804, 'Buenos Aires', 'seen'),
    (-34.58946, -58.44752, 'Buenos Aires', 'seen'),
    (-34.58897, -58.43056, 'Buenos Aires', 'seen'),
    (-34.58889, -58.40524, 'Buenos Aires', 'seen'),
    (-34.5887, -58.43397, 'Buenos Aires', 'seen'),
    (-34.58851, -58.4009, 'Buenos Aires', 'seen'),
    (-34.58819, -58.37434, 'Buenos Aires', 'seen'),
    (-34.58616, -58.43556, 'Buenos Aires', 'seen'),
    (-34.58544, -58.39368, 'Buenos Aires', 'seen'),
    (-34.5851, -58.40006, 'Buenos Aires', 'seen'),
    (-34.58464, -58.38423, 'Buenos Aires', 'seen'),
    (-34.58443, -58.41957, 'Buenos Aires', 'seen'),
    (-34.5844, -58.3991, 'Buenos Aires', 'seen'),
    (-34.58422, -58.38213, 'Buenos Aires', 'seen'),
    (-34.58368, -58.44638, 'Buenos Aires', 'seen'),
    (-34.58355, -58.42953, 'Buenos Aires', 'seen'),
    (-34.58179, -58.44139, 'Buenos Aires', 'seen'),
    (-34.58171, -58.37862, 'Buenos Aires', 'seen'),
    (-34.58151, -58.41816, 'Buenos Aires', 'seen'),
    (-34.58106, -58.37387, 'Buenos Aires', 'seen'),
    (-34.57994, -58.3757, 'Buenos Aires', 'seen'),
    (-34.57926, -58.43653, 'Buenos Aires', 'seen'),
    (-34.57872, -58.38231, 'Buenos Aires', 'seen'),
    (-34.57788, -58.39975, 'Buenos Aires', 'seen'),
    (-34.57784, -58.40863, 'Buenos Aires', 'seen'),
    (-34.57526, -58.44675, 'Buenos Aires', 'seen'),
    (-34.57481, -58.39068, 'Buenos Aires', 'seen'),
    (-34.5742, -58.42002, 'Buenos Aires', 'seen'),
    (-34.5742, -58.38946, 'Buenos Aires', 'seen'),
    (-34.57268, -58.41101, 'Buenos Aires', 'seen'),
    (-34.57054, -58.41246, 'Buenos Aires', 'seen'),
    (-34.56932, -58.41822, 'Buenos Aires', 'seen'),
    (-34.5693, -58.4186, 'Buenos Aires', 'seen'),
    (-34.56841, -58.40887, 'Buenos Aires', 'seen'),
    (-34.56826, -58.44651, 'Buenos Aires', 'seen'),
    (-34.56735, -58.4047, 'Buenos Aires', 'seen'),
    (-34.56708, -58.40774, 'Buenos Aires', 'seen'),
    (-34.56604, -58.43246, 'Buenos Aires', 'seen'),
    (-34.5645, -58.44892, 'Buenos Aires', 'seen'),
    (-34.56241, -58.42285, 'Buenos Aires', 'seen'),
    (40.13087, 28.93579, 'Bursa', 'seen'),
    (40.1347, 29.29498, 'Bursa', 'seen'),
    (40.14111, 29.29382, 'Bursa', 'seen'),
    (40.14879, 28.95677, 'Bursa', 'seen'),
    (40.14941, 28.95832, 'Bursa', 'seen'),
    (40.16186, 28.92146, 'Bursa', 'seen'),
    (40.16679, 29.10027, 'Bursa', 'seen'),
    (40.17476, 29.2087, 'Bursa', 'seen'),
    (40.17794, 29.0905, 'Bursa', 'seen'),
    (40.17801, 29.09965, 'Bursa', 'seen'),
    (40.17886, 29.09772, 'Bursa', 'seen'),
    (40.17919, 29.08516, 'Bursa', 'seen'),
    (40.17951, 29.13309, 'Bursa', 'seen'),
    (40.18087, 29.02056, 'Bursa', 'seen'),
    (40.18346, 29.12315, 'Bursa', 'seen'),
    (40.18422, 28.95766, 'Bursa', 'seen'),
    (40.18769, 29.0402, 'Bursa', 'seen'),
    (40.19231, 29.09628, 'Bursa', 'seen'),
    (40.19349, 28.94912, 'Bursa', 'seen'),
    (40.1941, 29.08722, 'Bursa', 'seen'),
    (40.19428, 29.14134, 'Bursa', 'seen'),
    (40.1943, 29.20199, 'Bursa', 'seen'),
    (40.19495, 29.12308, 'Bursa', 'seen'),
    (40.19557, 29.18263, 'Bursa', 'seen'),
    (40.19561, 28.99843, 'Bursa', 'seen'),
    (40.19625, 29.12289, 'Bursa', 'seen'),
    (40.19665, 29.08016, 'Bursa', 'seen'),
    (40.19697, 29.17745, 'Bursa', 'seen'),
    (40.19747, 29.02635, 'Bursa', 'seen'),
    (40.19792, 28.93989, 'Bursa', 'seen'),
    (40.19859, 28.95773, 'Bursa', 'seen'),
    (40.19948, 29.0325, 'Bursa', 'seen'),
    (40.20054, 29.13767, 'Bursa', 'seen'),
    (40.20675, 29.13566, 'Bursa', 'seen'),
    (40.20782, 29.19423, 'Bursa', 'seen'),
    (40.20825, 28.99482, 'Bursa', 'seen'),
    (40.20866, 29.12951, 'Bursa', 'seen'),
    (40.21011, 29.13732, 'Bursa', 'seen'),
    (40.21187, 29.27127, 'Bursa', 'seen'),
    (40.21402, 28.94562, 'Bursa', 'seen'),
    (40.21415, 29.12467, 'Bursa', 'seen'),
    (40.21493, 29.20355, 'Bursa', 'seen'),
    (40.21577, 29.09846, 'Bursa', 'seen'),
    (40.22012, 29.19273, 'Bursa', 'seen'),
    (40.22238, 29.12212, 'Bursa', 'seen'),
    (40.22321, 28.92134, 'Bursa', 'seen'),
    (40.22325, 29.04147, 'Bursa', 'seen'),
    (40.22462, 28.97362, 'Bursa', 'seen'),
    (40.22607, 28.90391, 'Bursa', 'seen'),
    (40.2273, 28.99661, 'Bursa', 'seen'),
    (40.22887, 29.02007, 'Bursa', 'seen'),
    (40.22931, 29.25819, 'Bursa', 'seen'),
    (40.23349, 29.08717, 'Bursa', 'seen'),
    (40.23447, 29.20674, 'Bursa', 'seen'),
    (40.23493, 29.17633, 'Bursa', 'seen'),
    (40.2373, 29.04211, 'Bursa', 'seen'),
    (40.24023, 29.15205, 'Bursa', 'seen'),
    (40.24039, 28.91278, 'Bursa', 'seen'),
    (40.24086, 28.95964, 'Bursa', 'seen'),
    (40.24144, 29.02129, 'Bursa', 'seen'),
    (40.24182, 28.97829, 'Bursa', 'seen'),
    (40.24618, 29.24727, 'Bursa', 'seen'),
    (40.24661, 29.20546, 'Bursa', 'seen'),
    (40.25094, 29.29388, 'Bursa', 'seen'),
    (40.25158, 29.19169, 'Bursa', 'seen'),
    (40.25226, 29.01983, 'Bursa', 'seen'),
    (40.25496, 28.95276, 'Bursa', 'seen'),
    (40.25617, 29.20301, 'Bursa', 'seen'),
    (40.2578, 28.90011, 'Bursa', 'seen'),
    (40.25907, 29.30105, 'Bursa', 'seen'),
    (40.26778, 29.12311, 'Bursa', 'seen'),
    (40.26922, 29.04617, 'Bursa', 'seen'),
    (40.27034, 29.11072, 'Bursa', 'seen'),
    (40.27073, 29.25738, 'Bursa', 'seen'),
    (40.27454, 28.96954, 'Bursa', 'seen'),
    (40.27769, 28.90235, 'Bursa', 'seen'),
    (40.28864, 29.00785, 'Bursa', 'seen'),
    (40.28893, 29.0153, 'Bursa', 'seen'),
    (40.29025, 28.98413, 'Bursa', 'seen'),
    (40.29081, 28.98211, 'Bursa', 'seen'),
    (40.29559, 29.01075, 'Bursa', 'seen'),
    (40.29701, 28.99609, 'Bursa', 'seen'),
    (40.29735, 29.01857, 'Bursa', 'seen'),
    (40.31593, 29.15937, 'Bursa', 'seen'),
    (40.32332, 29.27157, 'Bursa', 'seen'),
    (40.33124, 29.08535, 'Bursa', 'seen'),
    (40.33125, 29.08487, 'Bursa', 'seen'),
    (40.33949, 29.05503, 'Bursa', 'seen'),
    (40.34069, 28.94212, 'Bursa', 'seen'),
    (40.34566, 29.11314, 'Bursa', 'seen'),
    (40.34644, 28.90592, 'Bursa', 'seen'),
    (40.34951, 28.96692, 'Bursa', 'seen'),
    (40.35114, 28.92612, 'Bursa', 'seen'),
    (40.36629, 29.06523, 'Bursa', 'seen'),
    (40.37622, 29.13326, 'Bursa', 'seen'),
    (40.3799, 29.10203, 'Bursa', 'seen'),
    (40.38158, 29.21979, 'Bursa', 'seen'),
    (40.40264, 29.10493, 'Bursa', 'seen'),
    (40.41846, 29.16367, 'Bursa', 'seen'),
    (40.42302, 29.17495, 'Bursa', 'seen'),
    (40.44276, 29.14709, 'Bursa', 'seen'),
    (40.44735, 29.1995, 'Bursa', 'seen'),
    (40.44835, 29.13997, 'Bursa', 'seen'),
    (-34.19464, 18.37145, 'Cape Town', 'unseen'),
    (-34.17347, 18.34129, 'Cape Town', 'unseen'),
    (-34.14987, 18.42452, 'Cape Town', 'unseen'),
    (-34.14719, 18.40969, 'Cape Town', 'unseen'),
    (-34.12943, 18.93055, 'Cape Town', 'unseen'),
    (-34.12068, 18.90034, 'Cape Town', 'unseen'),
    (-34.11012, 18.91874, 'Cape Town', 'unseen'),
    (-34.10578, 18.37472, 'Cape Town', 'unseen'),
    (-34.10019, 18.84801, 'Cape Town', 'unseen'),
    (-34.08929, 18.46512, 'Cape Town', 'unseen'),
    (-34.07925, 18.44534, 'Cape Town', 'unseen'),
    (-34.07877, 18.45243, 'Cape Town', 'unseen'),
    (-34.07571, 18.83834, 'Cape Town', 'unseen'),
    (-34.06035, 18.45957, 'Cape Town', 'unseen'),
    (-34.05429, 18.59381, 'Cape Town', 'unseen'),
    (-34.05374, 18.81004, 'Cape Town', 'unseen'),
    (-34.05228, 18.58713, 'Cape Town', 'unseen'),
    (-34.05224, 18.71047, 'Cape Town', 'unseen'),
    (-34.04946, 18.80771, 'Cape Town', 'unseen'),
    (-34.04761, 18.34729, 'Cape Town', 'unseen'),
    (-34.03948, 18.47371, 'Cape Town', 'unseen'),
    (-34.02604, 18.65928, 'Cape Town', 'unseen'),
    (-34.01895, 18.36109, 'Cape Town', 'unseen'),
    (-34.01737, 18.71723, 'Cape Town', 'unseen'),
    (-34.00981, 18.41968, 'Cape Town', 'unseen'),
    (-34.00425, 18.67794, 'Cape Town', 'unseen'),
    (-33.99881, 18.46625, 'Cape Town', 'unseen'),
    (-33.99842, 18.66831, 'Cape Town', 'unseen'),
    (-33.99718, 18.55065, 'Cape Town', 'unseen'),
    (-33.99214, 18.71618, 'Cape Town', 'unseen'),
    (-33.9911, 18.43899, 'Cape Town', 'unseen'),
    (-33.99077, 18.70764, 'Cape Town', 'unseen'),
    (-33.98978, 18.77624, 'Cape Town', 'unseen'),
    (-33.98571, 18.53374, 'Cape Town', 'unseen'),
    (-33.98528, 18.6504, 'Cape Town', 'unseen'),
    (-33.98336, 18.71335, 'Cape Town', 'unseen'),
    (-33.9801, 18.4946, 'Cape Town', 'unseen'),
    (-33.97921, 18.4638, 'Cape Town', 'unseen'),
    (-33.97771, 18.85954, 'Cape Town', 'unseen'),
    (-33.977, 18.64064, 'Cape Town', 'unseen'),
    (-33.97675, 18.67331, 'Cape Town', 'unseen'),
    (-33.96953, 18.55959, 'Cape Town', 'unseen'),
    (-33.96664, 18.48154, 'Cape Town', 'unseen'),
    (-33.96454, 18.45864, 'Cape Town', 'unseen'),
    (-33.96364, 18.45971, 'Cape Town', 'unseen'),
    (-33.96021, 18.57636, 'Cape Town', 'unseen'),
    (-33.95785, 18.65578, 'Cape Town', 'unseen'),
    (-33.95571, 18.48021, 'Cape Town', 'unseen'),
    (-33.95334, 18.6727, 'Cape Town', 'unseen'),
    (-33.95191, 18.80463, 'Cape Town', 'unseen'),
    (-33.94738, 18.6579, 'Cape Town', 'unseen'),
    (-33.94308, 18.3846, 'Cape Town', 'unseen'),
    (-33.94277, 18.77607, 'Cape Town', 'unseen'),
    (-33.94257, 18.68418, 'Cape Town', 'unseen'),
    (-33.93135, 18.71436, 'Cape Town', 'unseen'),
    (-33.93021, 18.44068, 'Cape Town', 'unseen'),
    (-33.92763, 18.85113, 'Cape Town', 'unseen'),
    (-33.91794, 18.50561, 'Cape Town', 'unseen'),
    (-33.91367, 18.6868, 'Cape Town', 'unseen'),
    (-33.90959, 18.60803, 'Cape Town', 'unseen'),
    (-33.909, 18.52171, 'Cape Town', 'unseen'),
    (-33.90831, 18.57681, 'Cape Town', 'unseen'),
    (-33.90703, 18.68504, 'Cape Town', 'unseen'),
    (-33.90676, 18.60946, 'Cape Town', 'unseen'),
    (-33.90317, 18.48302, 'Cape Town', 'unseen'),
    (-33.90056, 18.55336, 'Cape Town', 'unseen'),
    (-33.89972, 18.84426, 'Cape Town', 'unseen'),
    (-33.89406, 18.63081, 'Cape Town', 'unseen'),
    (-33.89342, 18.70154, 'Cape Town', 'unseen'),
    (-33.88455, 18.50656, 'Cape Town', 'unseen'),
    (-33.8748, 18.64977, 'Cape Town', 'unseen'),
    (-33.86886, 18.54359, 'Cape Town', 'unseen'),
    (-33.86624, 18.54091, 'Cape Town', 'unseen'),
    (-33.86595, 18.73099, 'Cape Town', 'unseen'),
    (-33.86121, 18.65697, 'Cape Town', 'unseen'),
    (-33.85667, 18.86455, 'Cape Town', 'unseen'),
    (-33.85492, 18.73883, 'Cape Town', 'unseen'),
    (-33.85302, 18.64279, 'Cape Town', 'unseen'),
    (-33.85115, 18.63078, 'Cape Town', 'unseen'),
    (-33.84979, 18.71235, 'Cape Town', 'unseen'),
    (-33.84825, 18.74726, 'Cape Town', 'unseen'),
    (-33.84572, 18.67512, 'Cape Town', 'unseen'),
    (-33.84381, 18.71487, 'Cape Town', 'unseen'),
    (-33.83712, 18.87209, 'Cape Town', 'unseen'),
    (-33.83347, 18.72424, 'Cape Town', 'unseen'),
    (-33.82922, 18.72742, 'Cape Town', 'unseen'),
    (-33.82097, 18.66469, 'Cape Town', 'unseen'),
    (-33.80803, 18.69411, 'Cape Town', 'unseen'),
    (-33.80332, 18.87446, 'Cape Town', 'unseen'),
    (-33.80187, 18.61814, 'Cape Town', 'unseen'),
    (-33.76475, 18.76291, 'Cape Town', 'unseen'),
    (-33.75404, 18.45558, 'Cape Town', 'unseen'),
    (-33.75127, 18.70434, 'Cape Town', 'unseen'),
    (41.66034, -87.54712, 'Chicago', 'unseen'),
    (41.66145, -87.58607, 'Chicago', 'unseen'),
    (41.66198, -87.77215, 'Chicago', 'unseen'),
    (41.66232, -87.70804, 'Chicago', 'unseen'),
    (41.66492, -87.65134, 'Chicago', 'unseen'),
    (41.66648, -87.78745, 'Chicago', 'unseen'),
    (41.66695, -87.74861, 'Chicago', 'unseen'),
    (41.67201, -87.76299, 'Chicago', 'unseen'),
    (41.67228, -87.64672, 'Chicago', 'unseen'),
    (41.68203, -87.83099, 'Chicago', 'unseen'),
    (41.69601, -87.63526, 'Chicago', 'unseen'),
    (41.69735, -87.8003, 'Chicago', 'unseen'),
    (41.70147, -87.64214, 'Chicago', 'unseen'),
    (41.70814, -87.68007, 'Chicago', 'unseen'),
    (41.70997, -87.73367, 'Chicago', 'unseen'),
    (41.71093, -87.75545, 'Chicago', 'unseen'),
    (41.71382, -87.55094, 'Chicago', 'unseen'),
    (41.72051, -87.81073, 'Chicago', 'unseen'),
    (41.72196, -87.83016, 'Chicago', 'unseen'),
    (41.72563, -87.8291, 'Chicago', 'unseen'),
    (41.72708, -87.75322, 'Chicago', 'unseen'),
    (41.73052, -87.70211, 'Chicago', 'unseen'),
    (41.73082, -87.74848, 'Chicago', 'unseen'),
    (41.73255, -87.64369, 'Chicago', 'unseen'),
    (41.73751, -87.53732, 'Chicago', 'unseen'),
    (41.73855, -87.82029, 'Chicago', 'unseen'),
    (41.73927, -87.70453, 'Chicago', 'unseen'),
    (41.74152, -87.90348, 'Chicago', 'unseen'),
    (41.742, -87.73848, 'Chicago', 'unseen'),
    (41.74205, -87.82743, 'Chicago', 'unseen'),
    (41.74696, -87.73065, 'Chicago', 'unseen'),
    (41.74921, -87.55272, 'Chicago', 'unseen'),
    (41.74924, -87.76446, 'Chicago', 'unseen'),
    (41.74936, -87.75837, 'Chicago', 'unseen'),
    (41.75202, -87.79487, 'Chicago', 'unseen'),
    (41.75212, -87.78009, 'Chicago', 'unseen'),
    (41.75232, -87.6546, 'Chicago', 'unseen'),
    (41.75298, -87.55048, 'Chicago', 'unseen'),
    (41.75839, -87.76103, 'Chicago', 'unseen'),
    (41.76043, -87.65399, 'Chicago', 'unseen'),
    (41.76058, -87.84179, 'Chicago', 'unseen'),
    (41.76277, -87.68992, 'Chicago', 'unseen'),
    (41.76321, -87.62811, 'Chicago', 'unseen'),
    (41.76414, -87.59883, 'Chicago', 'unseen'),
    (41.76434, -87.70657, 'Chicago', 'unseen'),
    (41.76503, -87.89679, 'Chicago', 'unseen'),
    (41.76894, -87.63863, 'Chicago', 'unseen'),
    (41.77301, -87.73238, 'Chicago', 'unseen'),
    (41.77969, -87.91321, 'Chicago', 'unseen'),
    (41.78043, -87.63015, 'Chicago', 'unseen'),
    (41.78307, -87.89834, 'Chicago', 'unseen'),
    (41.78315, -87.91679, 'Chicago', 'unseen'),
    (41.79121, -87.81072, 'Chicago', 'unseen'),
    (41.79995, -87.76373, 'Chicago', 'unseen'),
    (41.80057, -87.69276, 'Chicago', 'unseen'),
    (41.80229, -87.82524, 'Chicago', 'unseen'),
    (41.80708, -87.75048, 'Chicago', 'unseen'),
    (41.80908, -87.86804, 'Chicago', 'unseen'),
    (41.81374, -87.68948, 'Chicago', 'unseen'),
    (41.81418, -87.93132, 'Chicago', 'unseen'),
    (41.81943, -87.8539, 'Chicago', 'unseen'),
    (41.82778, -87.84309, 'Chicago', 'unseen'),
    (41.8334, -87.71816, 'Chicago', 'unseen'),
    (41.83584, -87.80113, 'Chicago', 'unseen'),
    (41.8379, -87.8226, 'Chicago', 'unseen'),
    (41.83992, -87.89907, 'Chicago', 'unseen'),
    (41.84264, -87.89007, 'Chicago', 'unseen'),
    (41.84396, -87.65203, 'Chicago', 'unseen'),
    (41.8469, -87.80626, 'Chicago', 'unseen'),
    (41.84732, -87.86403, 'Chicago', 'unseen'),
    (41.85004, -87.8223, 'Chicago', 'unseen'),
    (41.85128, -87.67568, 'Chicago', 'unseen'),
    (41.85444, -87.78183, 'Chicago', 'unseen'),
    (41.85785, -87.93015, 'Chicago', 'unseen'),
    (41.85946, -87.80889, 'Chicago', 'unseen'),
    (41.86214, -87.75064, 'Chicago', 'unseen'),
    (41.86452, -87.74227, 'Chicago', 'unseen'),
    (41.86631, -87.83715, 'Chicago', 'unseen'),
    (41.87048, -87.87209, 'Chicago', 'unseen'),
    (41.88529, -87.77959, 'Chicago', 'unseen'),
    (41.89077, -87.90813, 'Chicago', 'unseen'),
    (41.89526, -87.65572, 'Chicago', 'unseen'),
    (41.89651, -87.66853, 'Chicago', 'unseen'),
    (41.89875, -87.74229, 'Chicago', 'unseen'),
    (41.9001, -87.87521, 'Chicago', 'unseen'),
    (41.90102, -87.66927, 'Chicago', 'unseen'),
    (41.90513, -87.9118, 'Chicago', 'unseen'),
    (41.90527, -87.8003, 'Chicago', 'unseen'),
    (41.90654, -87.70139, 'Chicago', 'unseen'),
    (41.91547, -87.93724, 'Chicago', 'unseen'),
    (41.91689, -87.86471, 'Chicago', 'unseen'),
    (41.91825, -87.78283, 'Chicago', 'unseen'),
    (41.925, -87.67999, 'Chicago', 'unseen'),
    (41.92765, -87.69629, 'Chicago', 'unseen'),
    (41.93343, -87.74172, 'Chicago', 'unseen'),
    (41.93413, -87.64433, 'Chicago', 'unseen'),
    (41.94425, -87.75138, 'Chicago', 'unseen'),
    (41.94543, -87.65245, 'Chicago', 'unseen'),
    (41.9457, -87.78371, 'Chicago', 'unseen'),
    (41.95146, -87.83662, 'Chicago', 'unseen'),
    (41.95437, -87.86257, 'Chicago', 'unseen'),
    (41.95581, -87.85916, 'Chicago', 'unseen'),
    (41.95641, -87.81094, 'Chicago', 'unseen'),
    (41.96292, -87.866, 'Chicago', 'unseen'),
    (41.96913, -87.74672, 'Chicago', 'unseen'),
    (41.97225, -87.86734, 'Chicago', 'unseen'),
    (41.97349, -87.8185, 'Chicago', 'unseen'),
    (41.9736, -87.73744, 'Chicago', 'unseen'),
    (41.98031, -87.69908, 'Chicago', 'unseen'),
    (41.98339, -87.70897, 'Chicago', 'unseen'),
    (41.98396, -87.80055, 'Chicago', 'unseen'),
    (41.99101, -87.841, 'Chicago', 'unseen'),
    (41.99124, -87.79878, 'Chicago', 'unseen'),
    (41.99791, -87.94001, 'Chicago', 'unseen'),
    (42.00144, -87.72641, 'Chicago', 'unseen'),
    (42.00884, -87.84072, 'Chicago', 'unseen'),
    (42.01017, -87.84922, 'Chicago', 'unseen'),
    (42.01769, -87.83509, 'Chicago', 'unseen'),
    (60.14827, 24.88695, 'Helsinki', 'seen'),
    (60.14993, 24.8769, 'Helsinki', 'seen'),
    (60.15048, 24.8808, 'Helsinki', 'seen'),
    (60.15139, 24.9126, 'Helsinki', 'seen'),
    (60.15148, 24.88971, 'Helsinki', 'seen'),
    (60.15178, 24.91646, 'Helsinki', 'seen'),
    (60.15346, 24.88277, 'Helsinki', 'seen'),
    (60.15646, 24.95142, 'Helsinki', 'seen'),
    (60.15647, 24.8742, 'Helsinki', 'seen'),
    (60.1568, 24.93325, 'Helsinki', 'seen'),
    (60.15756, 24.9498, 'Helsinki', 'seen'),
    (60.15785, 24.91522, 'Helsinki', 'seen'),
    (60.15795, 24.87188, 'Helsinki', 'seen'),
    (60.15869, 24.89054, 'Helsinki', 'seen'),
    (60.15892, 24.91816, 'Helsinki', 'seen'),
    (60.16006, 24.8728, 'Helsinki', 'seen'),
    (60.16194, 24.95623, 'Helsinki', 'seen'),
    (60.16417, 24.88587, 'Helsinki', 'seen'),
    (60.16469, 24.96981, 'Helsinki', 'seen'),
    (60.16558, 24.92703, 'Helsinki', 'seen'),
    (60.1659, 24.91156, 'Helsinki', 'seen'),
    (60.16596, 24.97523, 'Helsinki', 'seen'),
    (60.16665, 24.92091, 'Helsinki', 'seen'),
    (60.16946, 24.94927, 'Helsinki', 'seen'),
    (60.16976, 24.95966, 'Helsinki', 'seen'),
    (60.17007, 24.9221, 'Helsinki', 'seen'),
    (60.17133, 24.96161, 'Helsinki', 'seen'),
    (60.17145, 24.91859, 'Helsinki', 'seen'),
    (60.1724, 24.9154, 'Helsinki', 'seen'),
    (60.17376, 24.9671, 'Helsinki', 'seen'),
    (60.17406, 24.95179, 'Helsinki', 'seen'),
    (60.17491, 24.92678, 'Helsinki', 'seen'),
    (60.17542, 24.95963, 'Helsinki', 'seen'),
    (60.17592, 24.9605, 'Helsinki', 'seen'),
    (60.17651, 24.95454, 'Helsinki', 'seen'),
    (60.18037, 24.97329, 'Helsinki', 'seen'),
    (60.18066, 24.92457, 'Helsinki', 'seen'),
    (60.18072, 25.0171, 'Helsinki', 'seen'),
    (60.18122, 25.01051, 'Helsinki', 'seen'),
    (60.18351, 25.01279, 'Helsinki', 'seen'),
    (60.1842, 24.92722, 'Helsinki', 'seen'),
    (60.18453, 24.94589, 'Helsinki', 'seen'),
    (60.18558, 24.9238, 'Helsinki', 'seen'),
    (60.18579, 24.9931, 'Helsinki', 'seen'),
    (60.1861, 24.90638, 'Helsinki', 'seen'),
    (60.18611, 24.98634, 'Helsinki', 'seen'),
    (60.18631, 24.99523, 'Helsinki', 'seen'),
    (60.18648, 24.92255, 'Helsinki', 'seen'),
    (60.18652, 24.89903, 'Helsinki', 'seen'),
    (60.18656, 24.99221, 'Helsinki', 'seen'),
    (60.1867, 24.91686, 'Helsinki', 'seen'),
    (60.18777, 24.98036, 'Helsinki', 'seen'),
    (60.18807, 24.87461, 'Helsinki', 'seen'),
    (60.18829, 24.87199, 'Helsinki', 'seen'),
    (60.1887, 24.89066, 'Helsinki', 'seen'),
    (60.18872, 24.96151, 'Helsinki', 'seen'),
    (60.18977, 24.9828, 'Helsinki', 'seen'),
    (60.19013, 25.01625, 'Helsinki', 'seen'),
    (60.19026, 24.90879, 'Helsinki', 'seen'),
    (60.19039, 24.9528, 'Helsinki', 'seen'),
    (60.19049, 24.91713, 'Helsinki', 'seen'),
    (60.19131, 24.87508, 'Helsinki', 'seen'),
    (60.19181, 24.94308, 'Helsinki', 'seen'),
    (60.19196, 25.01939, 'Helsinki', 'seen'),
    (60.1921, 24.93262, 'Helsinki', 'seen'),
    (60.19254, 24.98351, 'Helsinki', 'seen'),
    (60.19264, 24.87253, 'Helsinki', 'seen'),
    (60.19307, 24.98324, 'Helsinki', 'seen'),
    (60.19369, 24.97783, 'Helsinki', 'seen'),
    (60.19381, 24.97768, 'Helsinki', 'seen'),
    (60.19435, 24.95268, 'Helsinki', 'seen'),
    (60.19468, 24.89847, 'Helsinki', 'seen'),
    (60.19544, 24.9678, 'Helsinki', 'seen'),
    (60.19605, 24.92763, 'Helsinki', 'seen'),
    (60.19609, 24.88417, 'Helsinki', 'seen'),
    (60.19645, 24.92354, 'Helsinki', 'seen'),
    (60.19705, 24.96031, 'Helsinki', 'seen'),
    (60.19715, 24.87264, 'Helsinki', 'seen'),
    (60.19775, 24.96381, 'Helsinki', 'seen'),
    (60.19837, 24.96932, 'Helsinki', 'seen'),
    (60.19869, 24.97071, 'Helsinki', 'seen'),
    (60.1991, 24.89945, 'Helsinki', 'seen'),
    (60.20013, 24.92148, 'Helsinki', 'seen'),
    (60.20019, 24.96197, 'Helsinki', 'seen'),
    (60.20032, 24.90003, 'Helsinki', 'seen'),
    (60.20143, 24.92671, 'Helsinki', 'seen'),
    (60.20154, 24.93735, 'Helsinki', 'seen'),
    (60.20203, 24.92025, 'Helsinki', 'seen'),
    (60.20232, 24.92851, 'Helsinki', 'seen'),
    (60.20274, 24.92944, 'Helsinki', 'seen'),
    (60.20355, 24.96245, 'Helsinki', 'seen'),
    (60.20355, 24.96533, 'Helsinki', 'seen'),
    (60.20378, 24.87261, 'Helsinki', 'seen'),
    (60.20412, 24.96386, 'Helsinki', 'seen'),
    (60.20413, 24.87708, 'Helsinki', 'seen'),
    (60.20414, 24.90337, 'Helsinki', 'seen'),
    (60.20446, 24.95619, 'Helsinki', 'seen'),
    (60.20446, 24.95619, 'Helsinki', 'seen'),
    (60.20472, 24.88324, 'Helsinki', 'seen'),
    (60.2055, 24.89894, 'Helsinki', 'seen'),
    (60.20604, 24.97081, 'Helsinki', 'seen'),
    (60.20634, 24.94803, 'Helsinki', 'seen'),
    (60.20677, 24.95923, 'Helsinki', 'seen'),
    (60.20687, 24.94673, 'Helsinki', 'seen'),
    (60.20696, 24.95847, 'Helsinki', 'seen'),
    (60.20731, 24.92517, 'Helsinki', 'seen'),
    (60.2079, 24.92084, 'Helsinki', 'seen'),
    (60.20799, 24.95082, 'Helsinki', 'seen'),
    (60.20835, 24.97844, 'Helsinki', 'seen'),
    (60.20841, 24.90836, 'Helsinki', 'seen'),
    (60.2085, 24.94541, 'Helsinki', 'seen'),
    (60.20874, 24.9383, 'Helsinki', 'seen'),
    (60.20884, 24.92954, 'Helsinki', 'seen'),
    (60.20907, 24.89155, 'Helsinki', 'seen'),
    (60.20951, 24.97057, 'Helsinki', 'seen'),
    (40.85264, 29.1135, 'Istanbul', 'unseen'),
    (40.87106, 29.12597, 'Istanbul', 'unseen'),
    (40.87984, 29.3738, 'Istanbul', 'unseen'),
    (40.88829, 29.37753, 'Istanbul', 'unseen'),
    (40.89719, 29.34291, 'Istanbul', 'unseen'),
    (40.90102, 29.28206, 'Istanbul', 'unseen'),
    (40.90319, 29.17313, 'Istanbul', 'unseen'),
    (40.91152, 29.38441, 'Istanbul', 'unseen'),
    (40.9146, 29.1695, 'Istanbul', 'unseen'),
    (40.91581, 29.30074, 'Istanbul', 'unseen'),
    (40.93355, 29.13754, 'Istanbul', 'unseen'),
    (40.94261, 29.10946, 'Istanbul', 'unseen'),
    (40.94754, 29.11361, 'Istanbul', 'unseen'),
    (40.95757, 28.83176, 'Istanbul', 'unseen'),
    (40.96208, 29.17647, 'Istanbul', 'unseen'),
    (40.96422, 28.80281, 'Istanbul', 'unseen'),
    (40.96911, 29.10378, 'Istanbul', 'unseen'),
    (40.96932, 29.19672, 'Istanbul', 'unseen'),
    (40.97058, 29.09569, 'Istanbul', 'unseen'),
    (40.97093, 29.07962, 'Istanbul', 'unseen'),
    (40.97229, 28.59602, 'Istanbul', 'unseen'),
    (40.97348, 29.13514, 'Istanbul', 'unseen'),
    (40.98336, 29.11399, 'Istanbul', 'unseen'),
    (40.98553, 29.25791, 'Istanbul', 'unseen'),
    (40.99003, 29.25789, 'Istanbul', 'unseen'),
    (40.99116, 29.04433, 'Istanbul', 'unseen'),
    (40.99206, 29.1213, 'Istanbul', 'unseen'),
    (40.99567, 28.82322, 'Istanbul', 'unseen'),
    (41.00018, 29.39005, 'Istanbul', 'unseen'),
    (41.00027, 29.39481, 'Istanbul', 'unseen'),
    (41.00311, 29.4284, 'Istanbul', 'unseen'),
    (41.00653, 28.88462, 'Istanbul', 'unseen'),
    (41.00757, 28.78948, 'Istanbul', 'unseen'),
    (41.00877, 29.26017, 'Istanbul', 'unseen'),
    (41.00903, 29.28404, 'Istanbul', 'unseen'),
    (41.0112, 28.91237, 'Istanbul', 'unseen'),
    (41.01128, 28.59893, 'Istanbul', 'unseen'),
    (41.01169, 29.03517, 'Istanbul', 'unseen'),
    (41.01345, 29.1915, 'Istanbul', 'unseen'),
    (41.01385, 28.87177, 'Istanbul', 'unseen'),
    (41.01411, 29.21897, 'Istanbul', 'unseen'),
    (41.01686, 29.27353, 'Istanbul', 'unseen'),
    (41.02365, 29.07901, 'Istanbul', 'unseen'),
    (41.02548, 28.60234, 'Istanbul', 'unseen'),
    (41.02558, 28.82752, 'Istanbul', 'unseen'),
    (41.02765, 28.65922, 'Istanbul', 'unseen'),
    (41.03088, 28.87819, 'Istanbul', 'unseen'),
    (41.03098, 28.84966, 'Istanbul', 'unseen'),
    (41.03339, 29.17869, 'Istanbul', 'unseen'),
    (41.04252, 28.68274, 'Istanbul', 'unseen'),
    (41.04391, 28.70609, 'Istanbul', 'unseen'),
    (41.0461, 28.75454, 'Istanbul', 'unseen'),
    (41.04749, 29.2836, 'Istanbul', 'unseen'),
    (41.05155, 29.18079, 'Istanbul', 'unseen'),
    (41.05185, 29.18257, 'Istanbul', 'unseen'),
    (41.05749, 29.06029, 'Istanbul', 'unseen'),
    (41.05752, 29.0866, 'Istanbul', 'unseen'),
    (41.0582, 28.82333, 'Istanbul', 'unseen'),
    (41.06053, 28.93861, 'Istanbul', 'unseen'),
    (41.06209, 28.65442, 'Istanbul', 'unseen'),
    (41.06689, 28.92286, 'Istanbul', 'unseen'),
    (41.06841, 28.61772, 'Istanbul', 'unseen'),
    (41.07279, 28.75444, 'Istanbul', 'unseen'),
    (41.07624, 28.79127, 'Istanbul', 'unseen'),
    (41.07949, 28.99294, 'Istanbul', 'unseen'),
    (41.08433, 29.15819, 'Istanbul', 'unseen'),
    (41.09214, 29.02716, 'Istanbul', 'unseen'),
    (41.09483, 29.35993, 'Istanbul', 'unseen'),
    (41.10039, 28.86709, 'Istanbul', 'unseen'),
    (41.10775, 28.74234, 'Istanbul', 'unseen'),
    (41.10835, 29.02134, 'Istanbul', 'unseen'),
    (41.12163, 29.4149, 'Istanbul', 'unseen'),
    (41.12365, 29.24942, 'Istanbul', 'unseen'),
    (41.12419, 28.76567, 'Istanbul', 'unseen'),
    (41.13564, 29.1406, 'Istanbul', 'unseen'),
    (41.14228, 29.10319, 'Istanbul', 'unseen'),
    (41.14239, 29.14588, 'Istanbul', 'unseen'),
    (41.14554, 28.60517, 'Istanbul', 'unseen'),
    (41.14908, 29.1837, 'Istanbul', 'unseen'),
    (41.15963, 28.67345, 'Istanbul', 'unseen'),
    (41.16237, 28.58092, 'Istanbul', 'unseen'),
    (41.16363, 29.01303, 'Istanbul', 'unseen'),
    (41.16592, 28.83959, 'Istanbul', 'unseen'),
    (41.16746, 28.9995, 'Istanbul', 'unseen'),
    (41.16814, 28.83356, 'Istanbul', 'unseen'),
    (41.16968, 28.61435, 'Istanbul', 'unseen'),
    (41.16973, 29.0498, 'Istanbul', 'unseen'),
    (41.18066, 28.77801, 'Istanbul', 'unseen'),
    (41.18177, 29.01496, 'Istanbul', 'unseen'),
    (41.18316, 28.59839, 'Istanbul', 'unseen'),
    (41.18472, 28.69814, 'Istanbul', 'unseen'),
    (41.18543, 29.16395, 'Istanbul', 'unseen'),
    (41.18877, 28.63857, 'Istanbul', 'unseen'),
    (41.18885, 28.70132, 'Istanbul', 'unseen'),
    (41.19584, 29.43083, 'Istanbul', 'unseen'),
    (38.28409, 27.28013, 'Izmir', 'seen'),
    (38.28541, 27.19862, 'Izmir', 'seen'),
    (38.30071, 27.13909, 'Izmir', 'seen'),
    (38.31406, 27.19156, 'Izmir', 'seen'),
    (38.33199, 27.10935, 'Izmir', 'seen'),
    (38.3333, 27.26405, 'Izmir', 'seen'),
    (38.33412, 27.1371, 'Izmir', 'seen'),
    (38.33525, 26.90643, 'Izmir', 'seen'),
    (38.34559, 27.2162, 'Izmir', 'seen'),
    (38.34597, 27.22176, 'Izmir', 'seen'),
    (38.35342, 27.08891, 'Izmir', 'seen'),
    (38.35457, 27.27851, 'Izmir', 'seen'),
    (38.35672, 27.23176, 'Izmir', 'seen'),
    (38.36135, 27.14431, 'Izmir', 'seen'),
    (38.36182, 27.27258, 'Izmir', 'seen'),
    (38.36595, 27.12543, 'Izmir', 'seen'),
    (38.37034, 27.17767, 'Izmir', 'seen'),
    (38.37131, 27.14238, 'Izmir', 'seen'),
    (38.37295, 27.16923, 'Izmir', 'seen'),
    (38.37873, 27.05602, 'Izmir', 'seen'),
    (38.38637, 27.19256, 'Izmir', 'seen'),
    (38.38858, 27.09751, 'Izmir', 'seen'),
    (38.38964, 27.17408, 'Izmir', 'seen'),
    (38.39032, 27.13658, 'Izmir', 'seen'),
    (38.39269, 27.06995, 'Izmir', 'seen'),
    (38.39342, 27.23504, 'Izmir', 'seen'),
    (38.39456, 27.18847, 'Izmir', 'seen'),
    (38.40035, 27.09138, 'Izmir', 'seen'),
    (38.40364, 27.00304, 'Izmir', 'seen'),
    (38.4205, 27.18744, 'Izmir', 'seen'),
    (38.43412, 27.19294, 'Izmir', 'seen'),
    (38.43435, 27.16154, 'Izmir', 'seen'),
    (38.43978, 27.20585, 'Izmir', 'seen'),
    (38.44161, 27.14598, 'Izmir', 'seen'),
    (38.45242, 27.31177, 'Izmir', 'seen'),
    (38.453, 27.1717, 'Izmir', 'seen'),
    (38.46634, 27.08251, 'Izmir', 'seen'),
    (38.47434, 27.2101, 'Izmir', 'seen'),
    (38.47706, 27.09962, 'Izmir', 'seen'),
    (38.47818, 27.06566, 'Izmir', 'seen'),
    (38.47843, 27.14043, 'Izmir', 'seen'),
    (38.48552, 27.18994, 'Izmir', 'seen'),
    (38.49288, 27.17213, 'Izmir', 'seen'),
    (38.49397, 27.15321, 'Izmir', 'seen'),
    (38.49593, 26.94545, 'Izmir', 'seen'),
    (38.49741, 27.10009, 'Izmir', 'seen'),
    (38.49816, 27.04844, 'Izmir', 'seen'),
    (38.49847, 27.08846, 'Izmir', 'seen'),
    (38.4996, 27.29795, 'Izmir', 'seen'),
    (38.49999, 27.2964, 'Izmir', 'seen'),
    (38.50506, 27.13745, 'Izmir', 'seen'),
    (38.50725, 27.2693, 'Izmir', 'seen'),
    (38.50896, 27.29167, 'Izmir', 'seen'),
    (38.51566, 27.04253, 'Izmir', 'seen'),
    (38.51576, 27.22108, 'Izmir', 'seen'),
    (38.5185, 27.043, 'Izmir', 'seen'),
    (38.52565, 27.0417, 'Izmir', 'seen'),
    (38.52838, 27.05072, 'Izmir', 'seen'),
    (38.57141, 26.99337, 'Izmir', 'seen'),
    (38.59962, 27.06588, 'Izmir', 'seen'),
    (38.63591, 35.40812, 'Kayseri', 'seen'),
    (38.64276, 35.58332, 'Kayseri', 'seen'),
    (38.64622, 35.46546, 'Kayseri', 'seen'),
    (38.64623, 35.52115, 'Kayseri', 'seen'),
    (38.64741, 35.44629, 'Kayseri', 'seen'),
    (38.65092, 35.43896, 'Kayseri', 'seen'),
    (38.654, 35.63123, 'Kayseri', 'seen'),
    (38.65618, 35.42143, 'Kayseri', 'seen'),
    (38.66073, 35.62319, 'Kayseri', 'seen'),
    (38.66081, 35.46879, 'Kayseri', 'seen'),
    (38.6655, 35.4972, 'Kayseri', 'seen'),
    (38.66713, 35.52282, 'Kayseri', 'seen'),
    (38.67037, 35.63605, 'Kayseri', 'seen'),
    (38.67037, 35.63605, 'Kayseri', 'seen'),
    (38.67173, 35.52608, 'Kayseri', 'seen'),
    (38.67293, 35.55766, 'Kayseri', 'seen'),
    (38.67446, 35.47446, 'Kayseri', 'seen'),
    (38.68054, 35.50001, 'Kayseri', 'seen'),
    (38.68447, 35.39852, 'Kayseri', 'seen'),
    (38.68585, 35.56771, 'Kayseri', 'seen'),
    (38.68978, 35.39862, 'Kayseri', 'seen'),
    (38.68978, 35.39862, 'Kayseri', 'seen'),
    (38.69299, 35.44607, 'Kayseri', 'seen'),
    (38.69357, 35.54697, 'Kayseri', 'seen'),
    (38.70002, 35.52697, 'Kayseri', 'seen'),
    (38.70134, 35.54153, 'Kayseri', 'seen'),
    (38.71162, 35.57099, 'Kayseri', 'seen'),
    (38.71312, 35.42328, 'Kayseri', 'seen'),
    (38.71585, 35.43344, 'Kayseri', 'seen'),
    (38.71887, 35.50104, 'Kayseri', 'seen'),
    (38.72001, 35.54681, 'Kayseri', 'seen'),
    (38.72025, 35.4961, 'Kayseri', 'seen'),
    (38.72393, 35.40464, 'Kayseri', 'seen'),
    (38.72519, 35.58613, 'Kayseri', 'seen'),
    (38.72531, 35.50938, 'Kayseri', 'seen'),
    (38.72531, 35.50938, 'Kayseri', 'seen'),
    (38.72611, 35.59582, 'Kayseri', 'seen'),
    (38.72756, 35.52186, 'Kayseri', 'seen'),
    (38.72813, 35.59, 'Kayseri', 'seen'),
    (38.72903, 35.44442, 'Kayseri', 'seen'),
    (38.72908, 35.46649, 'Kayseri', 'seen'),
    (38.7293, 35.52517, 'Kayseri', 'seen'),
    (38.72972, 35.51094, 'Kayseri', 'seen'),
    (38.73212, 35.40373, 'Kayseri', 'seen'),
    (38.73391, 35.50804, 'Kayseri', 'seen'),
    (38.73504, 35.49632, 'Kayseri', 'seen'),
    (38.7364, 35.37635, 'Kayseri', 'seen'),
    (38.73649, 35.49868, 'Kayseri', 'seen'),
    (38.73779, 35.44717, 'Kayseri', 'seen'),
    (38.73811, 35.38962, 'Kayseri', 'seen'),
    (38.73919, 35.42604, 'Kayseri', 'seen'),
    (38.73969, 35.58058, 'Kayseri', 'seen'),
    (38.74034, 35.40203, 'Kayseri', 'seen'),
    (38.74133, 35.56289, 'Kayseri', 'seen'),
    (38.74245, 35.44, 'Kayseri', 'seen'),
    (38.74277, 35.56518, 'Kayseri', 'seen'),
    (38.74404, 35.38016, 'Kayseri', 'seen'),
    (38.74452, 35.41322, 'Kayseri', 'seen'),
    (38.74542, 35.45621, 'Kayseri', 'seen'),
    (38.74825, 35.37475, 'Kayseri', 'seen'),
    (38.75216, 35.39787, 'Kayseri', 'seen'),
    (38.75234, 35.3746, 'Kayseri', 'seen'),
    (38.75312, 35.60745, 'Kayseri', 'seen'),
    (38.75552, 35.52662, 'Kayseri', 'seen'),
    (38.76245, 35.49829, 'Kayseri', 'seen'),
    (38.76543, 35.45469, 'Kayseri', 'seen'),
    (38.77002, 35.56917, 'Kayseri', 'seen'),
    (38.7715, 35.52467, 'Kayseri', 'seen'),
    (38.77257, 35.57301, 'Kayseri', 'seen'),
    (38.78301, 35.44886, 'Kayseri', 'seen'),
    (38.78895, 35.47185, 'Kayseri', 'seen'),
    (38.79279, 35.42471, 'Kayseri', 'seen'),
    (38.79763, 35.6248, 'Kayseri', 'seen'),
    (38.80161, 35.54252, 'Kayseri', 'seen'),
    (38.80221, 35.52504, 'Kayseri', 'seen'),
    (38.80279, 35.59929, 'Kayseri', 'seen'),
    (38.8202, 35.45666, 'Kayseri', 'seen'),
    (38.82346, 35.43491, 'Kayseri', 'seen'),
    (38.70183, -9.17286, 'Lisbon', 'seen'),
    (38.70199, -9.19838, 'Lisbon', 'seen'),
    (38.70253, -9.17145, 'Lisbon', 'seen'),
    (38.70326, -9.18562, 'Lisbon', 'seen'),
    (38.7059, -9.15571, 'Lisbon', 'seen'),
    (38.70705, -9.16751, 'Lisbon', 'seen'),
    (38.70784, -9.16499, 'Lisbon', 'seen'),
    (38.70873, -9.19131, 'Lisbon', 'seen'),
    (38.70882, -9.16183, 'Lisbon', 'seen'),
    (38.71047, -9.1804, 'Lisbon', 'seen'),
    (38.71087, -9.15867, 'Lisbon', 'seen'),
    (38.71105, -9.16805, 'Lisbon', 'seen'),
    (38.71144, -9.12883, 'Lisbon', 'seen'),
    (38.71344, -9.14868, 'Lisbon', 'seen'),
    (38.71368, -9.18097, 'Lisbon', 'seen'),
    (38.71401, -9.16978, 'Lisbon', 'seen'),
    (38.71473, -9.16068, 'Lisbon', 'seen'),
    (38.7152, -9.15422, 'Lisbon', 'seen'),
    (38.71525, -9.13713, 'Lisbon', 'seen'),
    (38.71673, -9.17088, 'Lisbon', 'seen'),
    (38.71691, -9.17994, 'Lisbon', 'seen'),
    (38.71704, -9.17997, 'Lisbon', 'seen'),
    (38.71912, -9.18494, 'Lisbon', 'seen'),
    (38.7202, -9.16415, 'Lisbon', 'seen'),
    (38.72158, -9.16202, 'Lisbon', 'seen'),
    (38.72257, -9.13669, 'Lisbon', 'seen'),
    (38.72391, -9.18221, 'Lisbon', 'seen'),
    (38.72422, -9.19395, 'Lisbon', 'seen'),
    (38.7255, -9.16261, 'Lisbon', 'seen'),
    (38.72578, -9.16264, 'Lisbon', 'seen'),
    (38.72725, -9.1931, 'Lisbon', 'seen'),
    (38.7274, -9.19283, 'Lisbon', 'seen'),
    (38.7275, -9.19587, 'Lisbon', 'seen'),
    (38.72774, -9.11008, 'Lisbon', 'seen'),
    (38.72793, -9.17365, 'Lisbon', 'seen'),
    (38.72853, -9.15988, 'Lisbon', 'seen'),
    (38.7287, -9.17432, 'Lisbon', 'seen'),
    (38.72952, -9.15089, 'Lisbon', 'seen'),
    (38.72985, -9.17801, 'Lisbon', 'seen'),
    (38.72985, -9.17801, 'Lisbon', 'seen'),
    (38.72985, -9.1673, 'Lisbon', 'seen'),
    (38.73079, -9.1712, 'Lisbon', 'seen'),
    (38.73136, -9.14625, 'Lisbon', 'seen'),
    (38.733, -9.1502, 'Lisbon', 'seen'),
    (38.73333, -9.18632, 'Lisbon', 'seen'),
    (38.73385, -9.19863, 'Lisbon', 'seen'),
    (38.73405, -9.17045, 'Lisbon', 'seen'),
    (38.73462, -9.11141, 'Lisbon', 'seen'),
    (38.73476, -9.16038, 'Lisbon', 'seen'),
    (38.73495, -9.1663, 'Lisbon', 'seen'),
    (38.73572, -9.13719, 'Lisbon', 'seen'),
    (38.73666, -9.13211, 'Lisbon', 'seen'),
    (38.73671, -9.16998, 'Lisbon', 'seen'),
    (38.7379, -9.12448, 'Lisbon', 'seen'),
    (38.73802, -9.12332, 'Lisbon', 'seen'),
    (38.73803, -9.14778, 'Lisbon', 'seen'),
    (38.73845, -9.12389, 'Lisbon', 'seen'),
    (38.73856, -9.12451, 'Lisbon', 'seen'),
    (38.73984, -9.13369, 'Lisbon', 'seen'),
    (38.74002, -9.17396, 'Lisbon', 'seen'),
    (38.74218, -9.10448, 'Lisbon', 'seen'),
    (38.74303, -9.17716, 'Lisbon', 'seen'),
    (38.74403, -9.16683, 'Lisbon', 'seen'),
    (38.74419, -9.18263, 'Lisbon', 'seen'),
    (38.74438, -9.1433, 'Lisbon', 'seen'),
    (38.74442, -9.1873, 'Lisbon', 'seen'),
    (38.74448, -9.12809, 'Lisbon', 'seen'),
    (38.7446, -9.13604, 'Lisbon', 'seen'),
    (38.74477, -9.19276, 'Lisbon', 'seen'),
    (38.74487, -9.14537, 'Lisbon', 'seen'),
    (38.74503, -9.16621, 'Lisbon', 'seen'),
    (38.74507, -9.16202, 'Lisbon', 'seen'),
    (38.74661, -9.15836, 'Lisbon', 'seen'),
    (38.74674, -9.1375, 'Lisbon', 'seen'),
    (38.74715, -9.10318, 'Lisbon', 'seen'),
    (38.74719, -9.14203, 'Lisbon', 'seen'),
    (38.74868, -9.19876, 'Lisbon', 'seen'),
    (38.74958, -9.11419, 'Lisbon', 'seen'),
    (38.74975, -9.15243, 'Lisbon', 'seen'),
    (38.75042, -9.1423, 'Lisbon', 'seen'),
    (38.75117, -9.14479, 'Lisbon', 'seen'),
    (38.75174, -9.18165, 'Lisbon', 'seen'),
    (38.75219, -9.11666, 'Lisbon', 'seen'),
    (38.75276, -9.18014, 'Lisbon', 'seen'),
    (38.75337, -9.10614, 'Lisbon', 'seen'),
    (38.75345, -9.11218, 'Lisbon', 'seen'),
    (38.75353, -9.18211, 'Lisbon', 'seen'),
    (38.75363, -9.17476, 'Lisbon', 'seen'),
    (38.75423, -9.19117, 'Lisbon', 'seen'),
    (38.7546, -9.17051, 'Lisbon', 'seen'),
    (38.75483, -9.19205, 'Lisbon', 'seen'),
    (38.75486, -9.11054, 'Lisbon', 'seen'),
    (38.7555, -9.12146, 'Lisbon', 'seen'),
    (38.75613, -9.11829, 'Lisbon', 'seen'),
    (38.75651, -9.10017, 'Lisbon', 'seen'),
    (38.75664, -9.19163, 'Lisbon', 'seen'),
    (38.75674, -9.15522, 'Lisbon', 'seen'),
    (38.75695, -9.14502, 'Lisbon', 'seen'),
    (38.7576, -9.13002, 'Lisbon', 'seen'),
    (38.75772, -9.17535, 'Lisbon', 'seen'),
    (38.7583, -9.13946, 'Lisbon', 'seen'),
    (38.75918, -9.19624, 'Lisbon', 'seen'),
    (38.75918, -9.19098, 'Lisbon', 'seen'),
    (38.75926, -9.11449, 'Lisbon', 'seen'),
    (38.75946, -9.1869, 'Lisbon', 'seen'),
    (38.75946, -9.17683, 'Lisbon', 'seen'),
    (38.75947, -9.19948, 'Lisbon', 'seen'),
    (38.75966, -9.16658, 'Lisbon', 'seen'),
    (51.29297, -0.22255, 'London', 'seen'),
    (51.30092, -0.34049, 'London', 'seen'),
    (51.30104, -0.23243, 'London', 'seen'),
    (51.30656, -0.06515, 'London', 'seen'),
    (51.31271, -0.31531, 'London', 'seen'),
    (51.3151, -0.31719, 'London', 'seen'),
    (51.31696, -0.30215, 'London', 'seen'),
    (51.31774, 0.22028, 'London', 'seen'),
    (51.31799, -0.17252, 'London', 'seen'),
    (51.33312, 0.07537, 'London', 'seen'),
    (51.34152, -0.39115, 'London', 'seen'),
    (51.34243, -0.23454, 'London', 'seen'),
    (51.35032, -0.24433, 'London', 'seen'),
    (51.35185, -0.4347, 'London', 'seen'),
    (51.36581, -0.30251, 'London', 'seen'),
    (51.36753, -0.37621, 'London', 'seen'),
    (51.36883, -0.40017, 'London', 'seen'),
    (51.37319, 0.0037, 'London', 'seen'),
    (51.37534, 0.19899, 'London', 'seen'),
    (51.37661, 0.09677, 'London', 'seen'),
    (51.38093, -0.45612, 'London', 'seen'),
    (51.38143, -0.24642, 'London', 'seen'),
    (51.38212, 0.31826, 'London', 'seen'),
    (51.38236, -0.12451, 'London', 'seen'),
    (51.38663, -0.38119, 'London', 'seen'),
    (51.38872, -0.30407, 'London', 'seen'),
    (51.39589, -0.34044, 'London', 'seen'),
    (51.39657, -0.0334, 'London', 'seen'),
    (51.40355, -0.39276, 'London', 'seen'),
    (51.4049, 0.15934, 'London', 'seen'),
    (51.405, -0.02959, 'London', 'seen'),
    (51.40506, -0.38266, 'London', 'seen'),
    (51.40582, 0.00323, 'London', 'seen'),
    (51.41103, -0.20738, 'London', 'seen'),
    (51.42472, 0.05064, 'London', 'seen'),
    (51.42604, -0.23683, 'London', 'seen'),
    (51.42725, -0.12173, 'London', 'seen'),
    (51.42796, -0.37499, 'London', 'seen'),
    (51.43004, -0.18153, 'London', 'seen'),
    (51.43534, -0.43503, 'London', 'seen'),
    (51.43645, -0.46715, 'London', 'seen'),
    (51.44078, -0.1604, 'London', 'seen'),
    (51.44083, -0.19164, 'London', 'seen'),
    (51.44569, 0.30098, 'London', 'seen'),
    (51.45545, -0.33031, 'London', 'seen'),
    (51.456, -0.31173, 'London', 'seen'),
    (51.45808, 0.0728, 'London', 'seen'),
    (51.46145, -0.28186, 'London', 'seen'),
    (51.4654, -0.14129, 'London', 'seen'),
    (51.46686, -0.34127, 'London', 'seen'),
    (51.46933, -0.41942, 'London', 'seen'),
    (51.47292, 0.20034, 'London', 'seen'),
    (51.47852, 0.10485, 'London', 'seen'),
    (51.48888, 0.10623, 'London', 'seen'),
    (51.49509, -0.47551, 'London', 'seen'),
    (51.5, -0.1929, 'London', 'seen'),
    (51.50178, -0.16097, 'London', 'seen'),
    (51.50478, -0.33498, 'London', 'seen'),
    (51.50959, -0.39016, 'London', 'seen'),
    (51.50979, -0.35816, 'London', 'seen'),
    (51.51083, -0.05245, 'London', 'seen'),
    (51.51302, 0.03411, 'London', 'seen'),
    (51.51308, -0.05727, 'London', 'seen'),
    (51.51759, 0.0191, 'London', 'seen'),
    (51.51812, -0.40843, 'London', 'seen'),
    (51.52134, -0.15254, 'London', 'seen'),
    (51.52461, -0.35328, 'London', 'seen'),
    (51.52552, 0.16535, 'London', 'seen'),
    (51.52978, -0.03731, 'London', 'seen'),
    (51.53004, -0.08129, 'London', 'seen'),
    (51.53258, -0.29381, 'London', 'seen'),
    (51.53373, -0.42331, 'London', 'seen'),
    (51.53541, 0.13877, 'London', 'seen'),
    (51.5379, 0.0608, 'London', 'seen'),
    (51.55592, -0.10579, 'London', 'seen'),
    (51.55752, 0.15914, 'London', 'seen'),
    (51.55787, 0.2765, 'London', 'seen'),
    (51.55927, -0.13313, 'London', 'seen'),
    (51.5633, -0.33407, 'London', 'seen'),
    (51.56421, -0.45591, 'London', 'seen'),
    (51.5694, -0.055, 'London', 'seen'),
    (51.57287, 0.04835, 'London', 'seen'),
    (51.58142, -0.27767, 'London', 'seen'),
    (51.58608, -0.3356, 'London', 'seen'),
    (51.58983, 0.0195, 'London', 'seen'),
    (51.59091, 0.01964, 'London', 'seen'),
    (51.59152, -0.43317, 'London', 'seen'),
    (51.59308, -0.16071, 'London', 'seen'),
    (51.60402, 0.21386, 'London', 'seen'),
    (51.60513, 0.31077, 'London', 'seen'),
    (51.60517, -0.48002, 'London', 'seen'),
    (51.60668, 0.02788, 'London', 'seen'),
    (51.60898, 0.17817, 'London', 'seen'),
    (51.61551, -0.0815, 'London', 'seen'),
    (51.61875, -0.21316, 'London', 'seen'),
    (51.61991, -0.28574, 'London', 'seen'),
    (51.63739, -0.47901, 'London', 'seen'),
    (51.64059, 0.05228, 'London', 'seen'),
    (51.64198, 0.28372, 'London', 'seen'),
    (51.64418, -0.30454, 'London', 'seen'),
    (51.65626, 0.13342, 'London', 'seen'),
    (51.65921, 0.06953, 'London', 'seen'),
    (51.6632, -0.35861, 'London', 'seen'),
    (51.66662, 0.2792, 'London', 'seen'),
    (51.66704, -0.35927, 'London', 'seen'),
    (51.67147, -0.2754, 'London', 'seen'),
    (51.6717, -0.21386, 'London', 'seen'),
    (51.6787, -0.41316, 'London', 'seen'),
    (51.6795, -0.24751, 'London', 'seen'),
    (51.68235, -0.09844, 'London', 'seen'),
    (19.34177, -99.11356, 'Mexico City', 'seen'),
    (19.34266, -99.13104, 'Mexico City', 'seen'),
    (19.34481, -99.10924, 'Mexico City', 'seen'),
    (19.34493, -99.18331, 'Mexico City', 'seen'),
    (19.34557, -99.08258, 'Mexico City', 'seen'),
    (19.34798, -99.08556, 'Mexico City', 'seen'),
    (19.34802, -99.11214, 'Mexico City', 'seen'),
    (19.34845, -99.08037, 'Mexico City', 'seen'),
    (19.34909, -99.11962, 'Mexico City', 'seen'),
    (19.35069, -99.16021, 'Mexico City', 'seen'),
    (19.35401, -99.11304, 'Mexico City', 'seen'),
    (19.35635, -99.13146, 'Mexico City', 'seen'),
    (19.36, -99.12948, 'Mexico City', 'seen'),
    (19.36154, -99.23307, 'Mexico City', 'seen'),
    (19.36379, -99.08597, 'Mexico City', 'seen'),
    (19.36931, -99.17001, 'Mexico City', 'seen'),
    (19.37046, -99.14904, 'Mexico City', 'seen'),
    (19.37053, -99.13829, 'Mexico City', 'seen'),
    (19.37335, -99.16745, 'Mexico City', 'seen'),
    (19.37451, -99.23955, 'Mexico City', 'seen'),
    (19.37501, -99.14467, 'Mexico City', 'seen'),
    (19.37664, -99.17159, 'Mexico City', 'seen'),
    (19.3794, -99.10875, 'Mexico City', 'seen'),
    (19.37976, -99.23928, 'Mexico City', 'seen'),
    (19.38042, -99.23624, 'Mexico City', 'seen'),
    (19.38093, -99.09121, 'Mexico City', 'seen'),
    (19.38133, -99.22003, 'Mexico City', 'seen'),
    (19.38194, -99.18912, 'Mexico City', 'seen'),
    (19.3869, -99.20331, 'Mexico City', 'seen'),
    (19.38798, -99.15841, 'Mexico City', 'seen'),
    (19.38972, -99.18717, 'Mexico City', 'seen'),
    (19.39165, -99.19059, 'Mexico City', 'seen'),
    (19.39207, -99.17567, 'Mexico City', 'seen'),
    (19.39327, -99.115, 'Mexico City', 'seen'),
    (19.39376, -99.20915, 'Mexico City', 'seen'),
    (19.39457, -99.15834, 'Mexico City', 'seen'),
    (19.39477, -99.09664, 'Mexico City', 'seen'),
    (19.39673, -99.22842, 'Mexico City', 'seen'),
    (19.39829, -99.21628, 'Mexico City', 'seen'),
    (19.40055, -99.12104, 'Mexico City', 'seen'),
    (19.40099, -99.18602, 'Mexico City', 'seen'),
    (19.40143, -99.09556, 'Mexico City', 'seen'),
    (19.40216, -99.14852, 'Mexico City', 'seen'),
    (19.40297, -99.11468, 'Mexico City', 'seen'),
    (19.40308, -99.13721, 'Mexico City', 'seen'),
    (19.40427, -99.16069, 'Mexico City', 'seen'),
    (19.40484, -99.11349, 'Mexico City', 'seen'),
    (19.40544, -99.13135, 'Mexico City', 'seen'),
    (19.40904, -99.11938, 'Mexico City', 'seen'),
    (19.40913, -99.16559, 'Mexico City', 'seen'),
    (19.40975, -99.19425, 'Mexico City', 'seen'),
    (19.4099, -99.14758, 'Mexico City', 'seen'),
    (19.41108, -99.12625, 'Mexico City', 'seen'),
    (19.41201, -99.1591, 'Mexico City', 'seen'),
    (19.41331, -99.17226, 'Mexico City', 'seen'),
    (19.41589, -99.07001, 'Mexico City', 'seen'),
    (19.41633, -99.17247, 'Mexico City', 'seen'),
    (19.41856, -99.09863, 'Mexico City', 'seen'),
    (19.41899, -99.12429, 'Mexico City', 'seen'),
    (19.42032, -99.09103, 'Mexico City', 'seen'),
    (19.42169, -99.20541, 'Mexico City', 'seen'),
    (19.4233, -99.21057, 'Mexico City', 'seen'),
    (19.4237, -99.07314, 'Mexico City', 'seen'),
    (19.42476, -99.0951, 'Mexico City', 'seen'),
    (19.42504, -99.0888, 'Mexico City', 'seen'),
    (19.42631, -99.21918, 'Mexico City', 'seen'),
    (19.43037, -99.15975, 'Mexico City', 'seen'),
    (19.43072, -99.17045, 'Mexico City', 'seen'),
    (19.43147, -99.13733, 'Mexico City', 'seen'),
    (19.43364, -99.09794, 'Mexico City', 'seen'),
    (19.43373, -99.23484, 'Mexico City', 'seen'),
    (19.43543, -99.08358, 'Mexico City', 'seen'),
    (19.4358, -99.19393, 'Mexico City', 'seen'),
    (19.43803, -99.08167, 'Mexico City', 'seen'),
    (19.43815, -99.19376, 'Mexico City', 'seen'),
    (19.43975, -99.21452, 'Mexico City', 'seen'),
    (19.44004, -99.17535, 'Mexico City', 'seen'),
    (19.44046, -99.19902, 'Mexico City', 'seen'),
    (19.44176, -99.16052, 'Mexico City', 'seen'),
    (19.44342, -99.15896, 'Mexico City', 'seen'),
    (19.44437, -99.18845, 'Mexico City', 'seen'),
    (19.44485, -99.08093, 'Mexico City', 'seen'),
    (19.44642, -99.18979, 'Mexico City', 'seen'),
    (19.44771, -99.20149, 'Mexico City', 'seen'),
    (19.45212, -99.20688, 'Mexico City', 'seen'),
    (19.45665, -99.1399, 'Mexico City', 'seen'),
    (19.45795, -99.10837, 'Mexico City', 'seen'),
    (19.4582, -99.08294, 'Mexico City', 'seen'),
    (19.46562, -99.12135, 'Mexico City', 'seen'),
    (19.46807, -99.08485, 'Mexico City', 'seen'),
    (19.47009, -99.18133, 'Mexico City', 'seen'),
    (19.47175, -99.18727, 'Mexico City', 'seen'),
    (19.47247, -99.07529, 'Mexico City', 'seen'),
    (19.47334, -99.16792, 'Mexico City', 'seen'),
    (19.47512, -99.19179, 'Mexico City', 'seen'),
    (19.47655, -99.15944, 'Mexico City', 'seen'),
    (19.47724, -99.2166, 'Mexico City', 'seen'),
    (19.47816, -99.11368, 'Mexico City', 'seen'),
    (19.47872, -99.15185, 'Mexico City', 'seen'),
    (19.47903, -99.20596, 'Mexico City', 'seen'),
    (19.48113, -99.09666, 'Mexico City', 'seen'),
    (19.48264, -99.13003, 'Mexico City', 'seen'),
    (19.48327, -99.1863, 'Mexico City', 'seen'),
    (19.48338, -99.20206, 'Mexico City', 'seen'),
    (19.48458, -99.18158, 'Mexico City', 'seen'),
    (19.48475, -99.19651, 'Mexico City', 'seen'),
    (19.48651, -99.11371, 'Mexico City', 'seen'),
    (19.48734, -99.12444, 'Mexico City', 'seen'),
    (19.4876, -99.22384, 'Mexico City', 'seen'),
    (19.48853, -99.19667, 'Mexico City', 'seen'),
    (19.49012, -99.18623, 'Mexico City', 'seen'),
    (19.49152, -99.17967, 'Mexico City', 'seen'),
    (19.49246, -99.08798, 'Mexico City', 'seen'),
    (19.49396, -99.14916, 'Mexico City', 'seen'),
    (19.49465, -99.08822, 'Mexico City', 'seen'),
    (19.49861, -99.14879, 'Mexico City', 'seen'),
    (19.49881, -99.22974, 'Mexico City', 'seen'),
    (19.49953, -99.11549, 'Mexico City', 'seen'),
    (55.70227, 37.52896, 'Moscow', 'unseen'),
    (55.70276, 37.5013, 'Moscow', 'unseen'),
    (55.70277, 37.67648, 'Moscow', 'unseen'),
    (55.7028, 37.67555, 'Moscow', 'unseen'),
    (55.70295, 37.60255, 'Moscow', 'unseen'),
    (55.70328, 37.69583, 'Moscow', 'unseen'),
    (55.70392, 37.64159, 'Moscow', 'unseen'),
    (55.70614, 37.59106, 'Moscow', 'unseen'),
    (55.70677, 37.50473, 'Moscow', 'unseen'),
    (55.70697, 37.67353, 'Moscow', 'unseen'),
    (55.70709, 37.64851, 'Moscow', 'unseen'),
    (55.70858, 37.51012, 'Moscow', 'unseen'),
    (55.70956, 37.65764, 'Moscow', 'unseen'),
    (55.71148, 37.62565, 'Moscow', 'unseen'),
    (55.7115, 37.6274, 'Moscow', 'unseen'),
    (55.71152, 37.60882, 'Moscow', 'unseen'),
    (55.71154, 37.67204, 'Moscow', 'unseen'),
    (55.7127, 37.5227, 'Moscow', 'unseen'),
    (55.71836, 37.63423, 'Moscow', 'unseen'),
    (55.71837, 37.58206, 'Moscow', 'unseen'),
    (55.72006, 37.61534, 'Moscow', 'unseen'),
    (55.72175, 37.58455, 'Moscow', 'unseen'),
    (55.72182, 37.67791, 'Moscow', 'unseen'),
    (55.72188, 37.51239, 'Moscow', 'unseen'),
    (55.72388, 37.57955, 'Moscow', 'unseen'),
    (55.72493, 37.55178, 'Moscow', 'unseen'),
    (55.72572, 37.65584, 'Moscow', 'unseen'),
    (55.72697, 37.57682, 'Moscow', 'unseen'),
    (55.73018, 37.69094, 'Moscow', 'unseen'),
    (55.73125, 37.52314, 'Moscow', 'unseen'),
    (55.73251, 37.60336, 'Moscow', 'unseen'),
    (55.73264, 37.56084, 'Moscow', 'unseen'),
    (55.73434, 37.59734, 'Moscow', 'unseen'),
    (55.7355, 37.59447, 'Moscow', 'unseen'),
    (55.73563, 37.59656, 'Moscow', 'unseen'),
    (55.73571, 37.51495, 'Moscow', 'unseen'),
    (55.73594, 37.63982, 'Moscow', 'unseen'),
    (55.73613, 37.67107, 'Moscow', 'unseen'),
    (55.73689, 37.60213, 'Moscow', 'unseen'),
    (55.73694, 37.62011, 'Moscow', 'unseen'),
    (55.73842, 37.5075, 'Moscow', 'unseen'),
    (55.73884, 37.59583, 'Moscow', 'unseen'),
    (55.73914, 37.57632, 'Moscow', 'unseen'),
    (55.73938, 37.64701, 'Moscow', 'unseen'),
    (55.73978, 37.5674, 'Moscow', 'unseen'),
    (55.7399, 37.50327, 'Moscow', 'unseen'),
    (55.74166, 37.6831, 'Moscow', 'unseen'),
    (55.7423, 37.59531, 'Moscow', 'unseen'),
    (55.7424, 37.6378, 'Moscow', 'unseen'),
    (55.74251, 37.54888, 'Moscow', 'unseen'),
    (55.74256, 37.67638, 'Moscow', 'unseen'),
    (55.74316, 37.63621, 'Moscow', 'unseen'),
    (55.7432, 37.5506, 'Moscow', 'unseen'),
    (55.74325, 37.53649, 'Moscow', 'unseen'),
    (55.74381, 37.67739, 'Moscow', 'unseen'),
    (55.74453, 37.68262, 'Moscow', 'unseen'),
    (55.74567, 37.54969, 'Moscow', 'unseen'),
    (55.74723, 37.54254, 'Moscow', 'unseen'),
    (55.74949, 37.58169, 'Moscow', 'unseen'),
    (55.74975, 37.65509, 'Moscow', 'unseen'),
    (55.74995, 37.67976, 'Moscow', 'unseen'),
    (55.75007, 37.57717, 'Moscow', 'unseen'),
    (55.75213, 37.56948, 'Moscow', 'unseen'),
    (55.75352, 37.62252, 'Moscow', 'unseen'),
    (55.75373, 37.65306, 'Moscow', 'unseen'),
    (55.75381, 37.5706, 'Moscow', 'unseen'),
    (55.75461, 37.56533, 'Moscow', 'unseen'),
    (55.75508, 37.57628, 'Moscow', 'unseen'),
    (55.75518, 37.57738, 'Moscow', 'unseen'),
    (55.75545, 37.69548, 'Moscow', 'unseen'),
    (55.75681, 37.52908, 'Moscow', 'unseen'),
    (55.75826, 37.62541, 'Moscow', 'unseen'),
    (55.75998, 37.69133, 'Moscow', 'unseen'),
    (55.76136, 37.64536, 'Moscow', 'unseen'),
    (55.76179, 37.68602, 'Moscow', 'unseen'),
    (55.7629, 37.64158, 'Moscow', 'unseen'),
    (55.76394, 37.64117, 'Moscow', 'unseen'),
    (55.76447, 37.59275, 'Moscow', 'unseen'),
    (55.76737, 37.59268, 'Moscow', 'unseen'),
    (55.76923, 37.64713, 'Moscow', 'unseen'),
    (55.76971, 37.66026, 'Moscow', 'unseen'),
    (55.7731, 37.6058, 'Moscow', 'unseen'),
    (55.77558, 37.65592, 'Moscow', 'unseen'),
    (55.77584, 37.65704, 'Moscow', 'unseen'),
    (55.77751, 37.50862, 'Moscow', 'unseen'),
    (55.77901, 37.59792, 'Moscow', 'unseen'),
    (55.77903, 37.62409, 'Moscow', 'unseen'),
    (55.78368, 37.56876, 'Moscow', 'unseen'),
    (55.78377, 37.63595, 'Moscow', 'unseen'),
    (55.78432, 37.62774, 'Moscow', 'unseen'),
    (55.78694, 37.58124, 'Moscow', 'unseen'),
    (55.78763, 37.62232, 'Moscow', 'unseen'),
    (55.78809, 37.54303, 'Moscow', 'unseen'),
    (55.7889, 37.56752, 'Moscow', 'unseen'),
    (55.79109, 37.55006, 'Moscow', 'unseen'),
    (55.79257, 37.52885, 'Moscow', 'unseen'),
    (55.79287, 37.69826, 'Moscow', 'unseen'),
    (55.79428, 37.68644, 'Moscow', 'unseen'),
    (55.79454, 37.57067, 'Moscow', 'unseen'),
    (55.79536, 37.5426, 'Moscow', 'unseen'),
    (55.79595, 37.63509, 'Moscow', 'unseen'),
    (55.79617, 37.56208, 'Moscow', 'unseen'),
    (55.79644, 37.55366, 'Moscow', 'unseen'),
    (55.79668, 37.56002, 'Moscow', 'unseen'),
    (55.79825, 37.59673, 'Moscow', 'unseen'),
    (55.79901, 37.55146, 'Moscow', 'unseen'),
    (55.7995, 37.5193, 'Moscow', 'unseen'),
    (18.87203, 72.95726, 'Mumbai', 'seen'),
    (18.8777, 72.98183, 'Mumbai', 'seen'),
    (18.88279, 72.9678, 'Mumbai', 'seen'),
    (18.88358, 72.98378, 'Mumbai', 'seen'),
    (18.88368, 72.95768, 'Mumbai', 'seen'),
    (18.88754, 72.91134, 'Mumbai', 'seen'),
    (18.89009, 72.98776, 'Mumbai', 'seen'),
    (18.89323, 72.96516, 'Mumbai', 'seen'),
    (18.89584, 72.9831, 'Mumbai', 'seen'),
    (18.90411, 72.97365, 'Mumbai', 'seen'),
    (18.91515, 72.81951, 'Mumbai', 'seen'),
    (18.92703, 72.81972, 'Mumbai', 'seen'),
    (18.95219, 72.79368, 'Mumbai', 'seen'),
    (18.95661, 72.84282, 'Mumbai', 'seen'),
    (18.96883, 72.82043, 'Mumbai', 'seen'),
    (18.97838, 72.82445, 'Mumbai', 'seen'),
    (18.9865, 72.85727, 'Mumbai', 'seen'),
    (19.00545, 72.81376, 'Mumbai', 'seen'),
    (19.01278, 72.89142, 'Mumbai', 'seen'),
    (19.01658, 72.88824, 'Mumbai', 'seen'),
    (19.01737, 72.83036, 'Mumbai', 'seen'),
    (19.02446, 72.89002, 'Mumbai', 'seen'),
    (19.03746, 72.94693, 'Mumbai', 'seen'),
    (19.04068, 72.94052, 'Mumbai', 'seen'),
    (19.04116, 72.93874, 'Mumbai', 'seen'),
    (19.05325, 72.83707, 'Mumbai', 'seen'),
    (19.05403, 72.91107, 'Mumbai', 'seen'),
    (19.05548, 72.83691, 'Mumbai', 'seen'),
    (19.05588, 72.89196, 'Mumbai', 'seen'),
    (19.05942, 72.90498, 'Mumbai', 'seen'),
    (19.06762, 72.83515, 'Mumbai', 'seen'),
    (19.06951, 72.84442, 'Mumbai', 'seen'),
    (19.07394, 72.85131, 'Mumbai', 'seen'),
    (19.07635, 72.88368, 'Mumbai', 'seen'),
    (19.07739, 72.8673, 'Mumbai', 'seen'),
    (19.07862, 72.91011, 'Mumbai', 'seen'),
    (19.08389, 72.92112, 'Mumbai', 'seen'),
    (19.09698, 72.84347, 'Mumbai', 'seen'),
    (19.10053, 72.84627, 'Mumbai', 'seen'),
    (19.10334, 72.88612, 'Mumbai', 'seen'),
    (19.10587, 72.8361, 'Mumbai', 'seen'),
    (19.12214, 72.84786, 'Mumbai', 'seen'),
    (19.12256, 72.88574, 'Mumbai', 'seen'),
    (19.12949, 72.87604, 'Mumbai', 'seen'),
    (19.13534, 72.83204, 'Mumbai', 'seen'),
    (19.13748, 72.87006, 'Mumbai', 'seen'),
    (19.14602, 72.86973, 'Mumbai', 'seen'),
    (19.15032, 72.82512, 'Mumbai', 'seen'),
    (19.15179, 72.98615, 'Mumbai', 'seen'),
    (19.15761, 72.78658, 'Mumbai', 'seen'),
    (19.16958, 72.85351, 'Mumbai', 'seen'),
    (19.16969, 72.83519, 'Mumbai', 'seen'),
    (19.17382, 72.86348, 'Mumbai', 'seen'),
    (19.17421, 72.8478, 'Mumbai', 'seen'),
    (19.17623, 72.96074, 'Mumbai', 'seen'),
    (19.17952, 72.97505, 'Mumbai', 'seen'),
    (19.19066, 72.94438, 'Mumbai', 'seen'),
    (19.20192, 72.94515, 'Mumbai', 'seen'),
    (19.20318, 72.82931, 'Mumbai', 'seen'),
    (19.2095, 72.86794, 'Mumbai', 'seen'),
    (19.21511, 72.97571, 'Mumbai', 'seen'),
    (19.22732, 72.84966, 'Mumbai', 'seen'),
    (19.22947, 72.86492, 'Mumbai', 'seen'),
    (19.2346, 72.85851, 'Mumbai', 'seen'),
    (19.23689, 72.86961, 'Mumbai', 'seen'),
    (19.24275, 72.84503, 'Mumbai', 'seen'),
    (19.24337, 72.98282, 'Mumbai', 'seen'),
    (19.26004, 72.79339, 'Mumbai', 'seen'),
    (19.26537, 72.97166, 'Mumbai', 'seen'),
    (19.26843, 72.96616, 'Mumbai', 'seen'),
    (-1.44512, 36.711, 'Nairobi', 'seen'),
    (-1.43609, 36.99627, 'Nairobi', 'seen'),
    (-1.42139, 37.01487, 'Nairobi', 'seen'),
    (-1.4163, 36.87174, 'Nairobi', 'seen'),
    (-1.39897, 36.7991, 'Nairobi', 'seen'),
    (-1.39756, 36.94146, 'Nairobi', 'seen'),
    (-1.39516, 36.77802, 'Nairobi', 'seen'),
    (-1.38935, 36.7658, 'Nairobi', 'seen'),
    (-1.37112, 36.98505, 'Nairobi', 'seen'),
    (-1.36871, 36.72373, 'Nairobi', 'seen'),
    (-1.35794, 36.67584, 'Nairobi', 'seen'),
    (-1.35692, 36.94303, 'Nairobi', 'seen'),
    (-1.34704, 36.74532, 'Nairobi', 'seen'),
    (-1.33862, 36.77278, 'Nairobi', 'seen'),
    (-1.30758, 37.08389, 'Nairobi', 'seen'),
    (-1.30156, 36.89131, 'Nairobi', 'seen'),
    (-1.29692, 37.0115, 'Nairobi', 'seen'),
    (-1.29153, 37.06532, 'Nairobi', 'seen'),
    (-1.29089, 36.905, 'Nairobi', 'seen'),
    (-1.28904, 36.68352, 'Nairobi', 'seen'),
    (-1.27999, 36.77306, 'Nairobi', 'seen'),
    (-1.2766, 36.67809, 'Nairobi', 'seen'),
    (-1.27547, 36.71569, 'Nairobi', 'seen'),
    (-1.27239, 36.99248, 'Nairobi', 'seen'),
    (-1.26825, 37.03701, 'Nairobi', 'seen'),
    (-1.266, 36.88876, 'Nairobi', 'seen'),
    (-1.26169, 37.07815, 'Nairobi', 'seen'),
    (-1.26108, 36.68995, 'Nairobi', 'seen'),
    (-1.25887, 36.70321, 'Nairobi', 'seen'),
    (-1.25818, 36.98661, 'Nairobi', 'seen'),
    (-1.25423, 36.93463, 'Nairobi', 'seen'),
    (-1.25264, 36.98292, 'Nairobi', 'seen'),
    (-1.24125, 36.79229, 'Nairobi', 'seen'),
    (-1.22906, 36.65152, 'Nairobi', 'seen'),
    (-1.21816, 36.79765, 'Nairobi', 'seen'),
    (-1.21611, 36.84853, 'Nairobi', 'seen'),
    (-1.21533, 36.65356, 'Nairobi', 'seen'),
    (-1.2129, 36.77545, 'Nairobi', 'seen'),
    (-1.20863, 36.79828, 'Nairobi', 'seen'),
    (-1.20825, 36.94238, 'Nairobi', 'seen'),
    (-1.20132, 36.78104, 'Nairobi', 'seen'),
    (-1.19308, 36.94059, 'Nairobi', 'seen'),
    (-1.19066, 36.85136, 'Nairobi', 'seen'),
    (-1.19032, 36.85076, 'Nairobi', 'seen'),
    (-1.19013, 36.88654, 'Nairobi', 'seen'),
    (-1.18344, 36.73462, 'Nairobi', 'seen'),
    (-1.17408, 36.92609, 'Nairobi', 'seen'),
    (-1.16318, 36.75854, 'Nairobi', 'seen'),
    (-1.15869, 37.01499, 'Nairobi', 'seen'),
    (-1.13447, 37.02092, 'Nairobi', 'seen'),
    (40.50911, -74.24341, 'New York City', 'seen'),
    (40.5121, -74.20988, 'New York City', 'seen'),
    (40.51827, -74.22445, 'New York City', 'seen'),
    (40.52523, -74.22144, 'New York City', 'seen'),
    (40.55776, -74.16002, 'New York City', 'seen'),
    (40.56087, -73.90787, 'New York City', 'seen'),
    (40.57004, -74.12208, 'New York City', 'seen'),
    (40.5762, -73.84112, 'New York City', 'seen'),
    (40.58771, -74.13639, 'New York City', 'seen'),
    (40.59366, -74.25538, 'New York City', 'seen'),
    (40.59592, -74.06532, 'New York City', 'seen'),
    (40.59874, -73.97894, 'New York City', 'seen'),
    (40.59906, -74.11664, 'New York City', 'seen'),
    (40.60086, -74.10647, 'New York City', 'seen'),
    (40.60793, -73.93418, 'New York City', 'seen'),
    (40.61045, -74.00813, 'New York City', 'seen'),
    (40.61125, -74.1696, 'New York City', 'seen'),
    (40.61252, -73.98989, 'New York City', 'seen'),
    (40.61401, -74.09622, 'New York City', 'seen'),
    (40.6175, -73.74189, 'New York City', 'seen'),
    (40.61767, -74.03045, 'New York City', 'seen'),
    (40.61906, -73.95513, 'New York City', 'seen'),
    (40.63013, -73.91839, 'New York City', 'seen'),
    (40.63015, -73.93581, 'New York City', 'seen'),
    (40.64967, -73.83364, 'New York City', 'seen'),
    (40.65019, -74.21113, 'New York City', 'seen'),
    (40.65957, -74.21632, 'New York City', 'seen'),
    (40.65998, -73.70784, 'New York City', 'seen'),
    (40.66065, -74.209, 'New York City', 'seen'),
    (40.66106, -74.21516, 'New York City', 'seen'),
    (40.66451, -73.97694, 'New York City', 'seen'),
    (40.66722, -74.25146, 'New York City', 'seen'),
    (40.66788, -73.78679, 'New York City', 'seen'),
    (40.66813, -73.7101, 'New York City', 'seen'),
    (40.66872, -73.95773, 'New York City', 'seen'),
    (40.67099, -73.71329, 'New York City', 'seen'),
    (40.67515, -73.72955, 'New York City', 'seen'),
    (40.67736, -73.89622, 'New York City', 'seen'),
    (40.68004, -74.21857, 'New York City', 'seen'),
    (40.68164, -73.95131, 'New York City', 'seen'),
    (40.69201, -74.10753, 'New York City', 'seen'),
    (40.69222, -74.08009, 'New York City', 'seen'),
    (40.69246, -73.85143, 'New York City', 'seen'),
    (40.69804, -73.74064, 'New York City', 'seen'),
    (40.70071, -73.92276, 'New York City', 'seen'),
    (40.70831, -73.71322, 'New York City', 'seen'),
    (40.71176, -73.89365, 'New York City', 'seen'),
    (40.71246, -73.84683, 'New York City', 'seen'),
    (40.71251, -73.81243, 'New York City', 'seen'),
    (40.7133, -73.99106, 'New York City', 'seen'),
    (40.71341, -73.83796, 'New York City', 'seen'),
    (40.71651, -74.19839, 'New York City', 'seen'),
    (40.72065, -73.92222, 'New York City', 'seen'),
    (40.72443, -74.24048, 'New York City', 'seen'),
    (40.7267, -73.7955, 'New York City', 'seen'),
    (40.72689, -73.73705, 'New York City', 'seen'),
    (40.7292, -73.94603, 'New York City', 'seen'),
    (40.72994, -73.72494, 'New York City', 'seen'),
    (40.73052, -74.15197, 'New York City', 'seen'),
    (40.73133, -73.74944, 'New York City', 'seen'),
    (40.73778, -73.77338, 'New York City', 'seen'),
    (40.74118, -74.07539, 'New York City', 'seen'),
    (40.74218, -73.69423, 'New York City', 'seen'),
    (40.74338, -73.74484, 'New York City', 'seen'),
    (40.74809, -74.16464, 'New York City', 'seen'),
    (40.7492, -73.75625, 'New York City', 'seen'),
    (40.74983, -74.21187, 'New York City', 'seen'),
    (40.7584, -73.80048, 'New York City', 'seen'),
    (40.76017, -73.73698, 'New York City', 'seen'),
    (40.76109, -73.68246, 'New York City', 'seen'),
    (40.76118, -73.86633, 'New York City', 'seen'),
    (40.76456, -73.76956, 'New York City', 'seen'),
    (40.76842, -73.87247, 'New York City', 'seen'),
    (40.77071, -73.95681, 'New York City', 'seen'),
    (40.77361, -74.04203, 'New York City', 'seen'),
    (40.78475, -74.12561, 'New York City', 'seen'),
    (40.78493, -74.03549, 'New York City', 'seen'),
    (40.79016, -74.00536, 'New York City', 'seen'),
    (40.79061, -73.83809, 'New York City', 'seen'),
    (40.79221, -73.71139, 'New York City', 'seen'),
    (40.79346, -74.17911, 'New York City', 'seen'),
    (40.79597, -74.18246, 'New York City', 'seen'),
    (40.79684, -74.11765, 'New York City', 'seen'),
    (40.80422, -73.87543, 'New York City', 'seen'),
    (40.8063, -74.00964, 'New York City', 'seen'),
    (40.80751, -73.99649, 'New York City', 'seen'),
    (40.81221, -74.16075, 'New York City', 'seen'),
    (40.81323, -74.20389, 'New York City', 'seen'),
    (40.83177, -74.07875, 'New York City', 'seen'),
    (40.83238, -73.7476, 'New York City', 'seen'),
    (40.83371, -74.01367, 'New York City', 'seen'),
    (40.83492, -73.70856, 'New York City', 'seen'),
    (40.8367, -73.89136, 'New York City', 'seen'),
    (40.83965, -73.91589, 'New York City', 'seen'),
    (40.84269, -73.86541, 'New York City', 'seen'),
    (40.84608, -74.12501, 'New York City', 'seen'),
    (40.84791, -73.85668, 'New York City', 'seen'),
    (40.84924, -73.86921, 'New York City', 'seen'),
    (40.85338, -74.03631, 'New York City', 'seen'),
    (40.85457, -73.88536, 'New York City', 'seen'),
    (40.86101, -73.98958, 'New York City', 'seen'),
    (40.86294, -73.91528, 'New York City', 'seen'),
    (40.86327, -73.8703, 'New York City', 'seen'),
    (40.86389, -73.80183, 'New York City', 'seen'),
    (40.88114, -73.96683, 'New York City', 'seen'),
    (40.88551, -73.86824, 'New York City', 'seen'),
    (40.89167, -73.98573, 'New York City', 'seen'),
    (40.90226, -73.86833, 'New York City', 'seen'),
    (40.90241, -73.94318, 'New York City', 'seen'),
    (40.90657, -74.15649, 'New York City', 'seen'),
    (40.91051, -74.08236, 'New York City', 'seen'),
    (40.9177, -73.81895, 'New York City', 'seen'),
    (40.91909, -74.10658, 'New York City', 'seen'),
    (35.13988, 33.34254, 'Nicosia', 'seen'),
    (35.14061, 33.37749, 'Nicosia', 'seen'),
    (35.14068, 33.37809, 'Nicosia', 'seen'),
    (35.14164, 33.36028, 'Nicosia', 'seen'),
    (35.14166, 33.34874, 'Nicosia', 'seen'),
    (35.14226, 33.33761, 'Nicosia', 'seen'),
    (35.14237, 33.33407, 'Nicosia', 'seen'),
    (35.14246, 33.3581, 'Nicosia', 'seen'),
    (35.14247, 33.36818, 'Nicosia', 'seen'),
    (35.14295, 33.37633, 'Nicosia', 'seen'),
    (35.14386, 33.36272, 'Nicosia', 'seen'),
    (35.14387, 33.35354, 'Nicosia', 'seen'),
    (35.1442, 33.38191, 'Nicosia', 'seen'),
    (35.14485, 33.35435, 'Nicosia', 'seen'),
    (35.14618, 33.34368, 'Nicosia', 'seen'),
    (35.14649, 33.33897, 'Nicosia', 'seen'),
    (35.14801, 33.36, 'Nicosia', 'seen'),
    (35.14835, 33.34611, 'Nicosia', 'seen'),
    (35.14888, 33.37054, 'Nicosia', 'seen'),
    (35.14954, 33.33857, 'Nicosia', 'seen'),
    (35.14988, 33.38219, 'Nicosia', 'seen'),
    (35.14991, 33.36904, 'Nicosia', 'seen'),
    (35.15137, 33.36631, 'Nicosia', 'seen'),
    (35.15182, 33.34303, 'Nicosia', 'seen'),
    (35.15185, 33.38759, 'Nicosia', 'seen'),
    (35.15296, 33.36998, 'Nicosia', 'seen'),
    (35.15349, 33.35879, 'Nicosia', 'seen'),
    (35.15481, 33.3715, 'Nicosia', 'seen'),
    (35.1549, 33.35351, 'Nicosia', 'seen'),
    (35.15518, 33.33579, 'Nicosia', 'seen'),
    (35.15549, 33.36963, 'Nicosia', 'seen'),
    (35.15553, 33.36751, 'Nicosia', 'seen'),
    (35.15572, 33.35442, 'Nicosia', 'seen'),
    (35.1561, 33.3877, 'Nicosia', 'seen'),
    (35.1563, 33.36751, 'Nicosia', 'seen'),
    (35.15688, 33.34257, 'Nicosia', 'seen'),
    (35.1585, 33.38357, 'Nicosia', 'seen'),
    (35.15957, 33.35742, 'Nicosia', 'seen'),
    (35.16005, 33.3382, 'Nicosia', 'seen'),
    (35.16044, 33.37289, 'Nicosia', 'seen'),
    (35.16055, 33.37206, 'Nicosia', 'seen'),
    (35.16059, 33.3714, 'Nicosia', 'seen'),
    (35.16067, 33.34835, 'Nicosia', 'seen'),
    (35.16081, 33.3786, 'Nicosia', 'seen'),
    (35.16089, 33.3771, 'Nicosia', 'seen'),
    (35.16167, 33.36059, 'Nicosia', 'seen'),
    (35.16175, 33.37438, 'Nicosia', 'seen'),
    (35.1626, 33.36406, 'Nicosia', 'seen'),
    (35.163, 33.34491, 'Nicosia', 'seen'),
    (35.16346, 33.34906, 'Nicosia', 'seen'),
    (35.16365, 33.35302, 'Nicosia', 'seen'),
    (35.16382, 33.35222, 'Nicosia', 'seen'),
    (35.16462, 33.35816, 'Nicosia', 'seen'),
    (35.16469, 33.36565, 'Nicosia', 'seen'),
    (35.16573, 33.34181, 'Nicosia', 'seen'),
    (35.16577, 33.34803, 'Nicosia', 'seen'),
    (35.16609, 33.38403, 'Nicosia', 'seen'),
    (35.16688, 33.38263, 'Nicosia', 'seen'),
    (35.16824, 33.3453, 'Nicosia', 'seen'),
    (35.16837, 33.37793, 'Nicosia', 'seen'),
    (35.16926, 33.3684, 'Nicosia', 'seen'),
    (35.16953, 33.33802, 'Nicosia', 'seen'),
    (35.17019, 33.35109, 'Nicosia', 'seen'),
    (35.17088, 33.35869, 'Nicosia', 'seen'),
    (35.17111, 33.38373, 'Nicosia', 'seen'),
    (35.1713, 33.34731, 'Nicosia', 'seen'),
    (35.17217, 33.37318, 'Nicosia', 'seen'),
    (35.17248, 33.34904, 'Nicosia', 'seen'),
    (35.17291, 33.33881, 'Nicosia', 'seen'),
    (35.17298, 33.33327, 'Nicosia', 'seen'),
    (35.17351, 33.36472, 'Nicosia', 'seen'),
    (35.17355, 33.34485, 'Nicosia', 'seen'),
    (35.17368, 33.3526, 'Nicosia', 'seen'),
    (35.17465, 33.33969, 'Nicosia', 'seen'),
    (35.17514, 33.37696, 'Nicosia', 'seen'),
    (35.17541, 33.35339, 'Nicosia', 'seen'),
    (35.17583, 33.33632, 'Nicosia', 'seen'),
    (35.17638, 33.38682, 'Nicosia', 'seen'),
    (35.17651, 33.37909, 'Nicosia', 'seen'),
    (35.17705, 33.34535, 'Nicosia', 'seen'),
    (35.17747, 33.33279, 'Nicosia', 'seen'),
    (35.17754, 33.33516, 'Nicosia', 'seen'),
    (35.17763, 33.33392, 'Nicosia', 'seen'),
    (35.17871, 33.34078, 'Nicosia', 'seen'),
    (35.18002, 33.33342, 'Nicosia', 'seen'),
    (35.18019, 33.38317, 'Nicosia', 'seen'),
    (35.18029, 33.38575, 'Nicosia', 'seen'),
    (35.18053, 33.33037, 'Nicosia', 'seen'),
    (35.18218, 33.3844, 'Nicosia', 'seen'),
    (35.18356, 33.38161, 'Nicosia', 'seen'),
    (35.18548, 33.3822, 'Nicosia', 'seen'),
    (35.18561, 33.37529, 'Nicosia', 'seen'),
    (35.18612, 33.38456, 'Nicosia', 'seen'),
    (35.18667, 33.38574, 'Nicosia', 'seen'),
    (35.18903, 33.38594, 'Nicosia', 'seen'),
    (35.1894, 33.38084, 'Nicosia', 'seen'),
    (35.18947, 33.38951, 'Nicosia', 'seen'),
    (48.81519, 2.40133, 'Paris', 'seen'),
    (48.81818, 2.39479, 'Paris', 'seen'),
    (48.81888, 2.2417, 'Paris', 'seen'),
    (48.81945, 2.35753, 'Paris', 'seen'),
    (48.81956, 2.39232, 'Paris', 'seen'),
    (48.8207, 2.23672, 'Paris', 'seen'),
    (48.82088, 2.26564, 'Paris', 'seen'),
    (48.82257, 2.29288, 'Paris', 'seen'),
    (48.82365, 2.41779, 'Paris', 'seen'),
    (48.82451, 2.27433, 'Paris', 'seen'),
    (48.82472, 2.41388, 'Paris', 'seen'),
    (48.82474, 2.25732, 'Paris', 'seen'),
    (48.82499, 2.38852, 'Paris', 'seen'),
    (48.82534, 2.30419, 'Paris', 'seen'),
    (48.82568, 2.29919, 'Paris', 'seen'),
    (48.82632, 2.2786, 'Paris', 'seen'),
    (48.82638, 2.29694, 'Paris', 'seen'),
    (48.82746, 2.36978, 'Paris', 'seen'),
    (48.8286, 2.36117, 'Paris', 'seen'),
    (48.82905, 2.24758, 'Paris', 'seen'),
    (48.82943, 2.26259, 'Paris', 'seen'),
    (48.83011, 2.35093, 'Paris', 'seen'),
    (48.8311, 2.35405, 'Paris', 'seen'),
    (48.83142, 2.32977, 'Paris', 'seen'),
    (48.83218, 2.25265, 'Paris', 'seen'),
    (48.83239, 2.40402, 'Paris', 'seen'),
    (48.8326, 2.40014, 'Paris', 'seen'),
    (48.83345, 2.27679, 'Paris', 'seen'),
    (48.83501, 2.37262, 'Paris', 'seen'),
    (48.8351, 2.30904, 'Paris', 'seen'),
    (48.83549, 2.34776, 'Paris', 'seen'),
    (48.83591, 2.23464, 'Paris', 'seen'),
    (48.8362, 2.39478, 'Paris', 'seen'),
    (48.8365, 2.23796, 'Paris', 'seen'),
    (48.83679, 2.29948, 'Paris', 'seen'),
    (48.83698, 2.24313, 'Paris', 'seen'),
    (48.83728, 2.30132, 'Paris', 'seen'),
    (48.83755, 2.30199, 'Paris', 'seen'),
    (48.83819, 2.29218, 'Paris', 'seen'),
    (48.8386, 2.29491, 'Paris', 'seen'),
    (48.83922, 2.3921, 'Paris', 'seen'),
    (48.84018, 2.38687, 'Paris', 'seen'),
    (48.84037, 2.25159, 'Paris', 'seen'),
    (48.84048, 2.25935, 'Paris', 'seen'),
    (48.84062, 2.26226, 'Paris', 'seen'),
    (48.84077, 2.24518, 'Paris', 'seen'),
    (48.84092, 2.38495, 'Paris', 'seen'),
    (48.84106, 2.24051, 'Paris', 'seen'),
    (48.84109, 2.34113, 'Paris', 'seen'),
    (48.84125, 2.35046, 'Paris', 'seen'),
    (48.84142, 2.23184, 'Paris', 'seen'),
    (48.84164, 2.23586, 'Paris', 'seen'),
    (48.8423, 2.34082, 'Paris', 'seen'),
    (48.84239, 2.29107, 'Paris', 'seen'),
    (48.84452, 2.36441, 'Paris', 'seen'),
    (48.84624, 2.24625, 'Paris', 'seen'),
    (48.84659, 2.30879, 'Paris', 'seen'),
    (48.84678, 2.30589, 'Paris', 'seen'),
    (48.84683, 2.34785, 'Paris', 'seen'),
    (48.84694, 2.25443, 'Paris', 'seen'),
    (48.8475, 2.26738, 'Paris', 'seen'),
    (48.84811, 2.23912, 'Paris', 'seen'),
    (48.84951, 2.2729, 'Paris', 'seen'),
    (48.84989, 2.34001, 'Paris', 'seen'),
    (48.85133, 2.2333, 'Paris', 'seen'),
    (48.85155, 2.36382, 'Paris', 'seen'),
    (48.85332, 2.28177, 'Paris', 'seen'),
    (48.8534, 2.2608, 'Paris', 'seen'),
    (48.85423, 2.22628, 'Paris', 'seen'),
    (48.85446, 2.29805, 'Paris', 'seen'),
    (48.85603, 2.29742, 'Paris', 'seen'),
    (48.85614, 2.34668, 'Paris', 'seen'),
    (48.85694, 2.29035, 'Paris', 'seen'),
    (48.85695, 2.32854, 'Paris', 'seen'),
    (48.85709, 2.36443, 'Paris', 'seen'),
    (48.85797, 2.29757, 'Paris', 'seen'),
    (48.86077, 2.3601, 'Paris', 'seen'),
    (48.86269, 2.40666, 'Paris', 'seen'),
    (48.86292, 2.38099, 'Paris', 'seen'),
    (48.86461, 2.41855, 'Paris', 'seen'),
    (48.86546, 2.38636, 'Paris', 'seen'),
    (48.8673, 2.27975, 'Paris', 'seen'),
    (48.86792, 2.32466, 'Paris', 'seen'),
    (48.86927, 2.32004, 'Paris', 'seen'),
    (48.87073, 2.29448, 'Paris', 'seen'),
    (48.87091, 2.36919, 'Paris', 'seen'),
    (48.87152, 2.39714, 'Paris', 'seen'),
    (48.87161, 2.3757, 'Paris', 'seen'),
    (48.87224, 2.41682, 'Paris', 'seen'),
    (48.87271, 2.28019, 'Paris', 'seen'),
    (48.87331, 2.26661, 'Paris', 'seen'),
    (48.87331, 2.35031, 'Paris', 'seen'),
    (48.87337, 2.30127, 'Paris', 'seen'),
    (48.87365, 2.32707, 'Paris', 'seen'),
    (48.87366, 2.32777, 'Paris', 'seen'),
    (48.87379, 2.33634, 'Paris', 'seen'),
    (48.87443, 2.35068, 'Paris', 'seen'),
    (48.87594, 2.23417, 'Paris', 'seen'),
    (48.87656, 2.25161, 'Paris', 'seen'),
    (48.87898, 2.39111, 'Paris', 'seen'),
    (48.88076, 2.25409, 'Paris', 'seen'),
    (48.88112, 2.28051, 'Paris', 'seen'),
    (48.88241, 2.41899, 'Paris', 'seen'),
    (48.88259, 2.39043, 'Paris', 'seen'),
    (48.88584, 2.35461, 'Paris', 'seen'),
    (48.88618, 2.25633, 'Paris', 'seen'),
    (48.88702, 2.33969, 'Paris', 'seen'),
    (48.88811, 2.38636, 'Paris', 'seen'),
    (48.88916, 2.28328, 'Paris', 'seen'),
    (48.89061, 2.39535, 'Paris', 'seen'),
    (48.89095, 2.38672, 'Paris', 'seen'),
    (48.89185, 2.32185, 'Paris', 'seen'),
    (48.89208, 2.28773, 'Paris', 'seen'),
    (48.89274, 2.24086, 'Paris', 'seen'),
    (48.89836, 2.27706, 'Paris', 'seen'),
    (48.89962, 2.40572, 'Paris', 'seen'),
    (48.90037, 2.2855, 'Paris', 'seen'),
    (48.9012, 2.34075, 'Paris', 'seen'),
    (64.12026, -21.88114, 'Reykjavik', 'seen'),
    (64.12028, -21.89367, 'Reykjavik', 'seen'),
    (64.12071, -21.89, 'Reykjavik', 'seen'),
    (64.12161, -21.86808, 'Reykjavik', 'seen'),
    (64.12502, -21.85794, 'Reykjavik', 'seen'),
    (64.12505, -21.91411, 'Reykjavik', 'seen'),
    (64.12508, -21.85448, 'Reykjavik', 'seen'),
    (64.12532, -21.91257, 'Reykjavik', 'seen'),
    (64.12598, -21.92833, 'Reykjavik', 'seen'),
    (64.12688, -21.85356, 'Reykjavik', 'seen'),
    (64.12703, -21.86694, 'Reykjavik', 'seen'),
    (64.12725, -21.88734, 'Reykjavik', 'seen'),
    (64.12815, -21.9008, 'Reykjavik', 'seen'),
    (64.12831, -21.95531, 'Reykjavik', 'seen'),
    (64.12832, -21.897, 'Reykjavik', 'seen'),
    (64.12838, -21.86331, 'Reykjavik', 'seen'),
    (64.12859, -21.87037, 'Reykjavik', 'seen'),
    (64.12863, -21.94626, 'Reykjavik', 'seen'),
    (64.12874, -21.87126, 'Reykjavik', 'seen'),
    (64.12878, -21.94647, 'Reykjavik', 'seen'),
    (64.12892, -21.95698, 'Reykjavik', 'seen'),
    (64.12904, -21.95235, 'Reykjavik', 'seen'),
    (64.12915, -21.85151, 'Reykjavik', 'seen'),
    (64.12917, -21.95727, 'Reykjavik', 'seen'),
    (64.12961, -21.94974, 'Reykjavik', 'seen'),
    (64.1297, -21.95011, 'Reykjavik', 'seen'),
    (64.13008, -21.95414, 'Reykjavik', 'seen'),
    (64.1305, -21.91122, 'Reykjavik', 'seen'),
    (64.13183, -21.85442, 'Reykjavik', 'seen'),
    (64.13221, -21.8829, 'Reykjavik', 'seen'),
    (64.13289, -21.89615, 'Reykjavik', 'seen'),
    (64.133, -21.90759, 'Reykjavik', 'seen'),
    (64.13316, -21.89785, 'Reykjavik', 'seen'),
    (64.13317, -21.92765, 'Reykjavik', 'seen'),
    (64.13358, -21.92541, 'Reykjavik', 'seen'),
    (64.1336, -21.95141, 'Reykjavik', 'seen'),
    (64.1336, -21.94822, 'Reykjavik', 'seen'),
    (64.13374, -21.93313, 'Reykjavik', 'seen'),
    (64.13381, -21.93317, 'Reykjavik', 'seen'),
    (64.13426, -21.94987, 'Reykjavik', 'seen'),
    (64.13487, -21.85137, 'Reykjavik', 'seen'),
    (64.1351, -21.91127, 'Reykjavik', 'seen'),
    (64.13566, -21.90908, 'Reykjavik', 'seen'),
    (64.13582, -21.86821, 'Reykjavik', 'seen'),
    (64.13647, -21.9094, 'Reykjavik', 'seen'),
    (64.13667, -21.90924, 'Reykjavik', 'seen'),
    (64.1367, -21.85155, 'Reykjavik', 'seen'),
    (64.13711, -21.85914, 'Reykjavik', 'seen'),
    (64.13785, -21.91712, 'Reykjavik', 'seen'),
    (64.13797, -21.93799, 'Reykjavik', 'seen'),
    (64.13859, -21.91374, 'Reykjavik', 'seen'),
    (64.13864, -21.88561, 'Reykjavik', 'seen'),
    (64.13896, -21.85396, 'Reykjavik', 'seen'),
    (64.13899, -21.96331, 'Reykjavik', 'seen'),
    (64.13912, -21.85146, 'Reykjavik', 'seen'),
    (64.13949, -21.88056, 'Reykjavik', 'seen'),
    (64.13961, -21.94183, 'Reykjavik', 'seen'),
    (64.14005, -21.89941, 'Reykjavik', 'seen'),
    (64.14024, -21.87158, 'Reykjavik', 'seen'),
    (64.14031, -21.88848, 'Reykjavik', 'seen'),
    (64.14072, -21.96469, 'Reykjavik', 'seen'),
    (64.14072, -21.96469, 'Reykjavik', 'seen'),
    (64.14087, -21.86711, 'Reykjavik', 'seen'),
    (64.14102, -21.9629, 'Reykjavik', 'seen'),
    (64.14118, -21.93372, 'Reykjavik', 'seen'),
    (64.14133, -21.85734, 'Reykjavik', 'seen'),
    (64.14144, -21.86598, 'Reykjavik', 'seen'),
    (64.14197, -21.96393, 'Reykjavik', 'seen'),
    (64.14225, -21.9292, 'Reykjavik', 'seen'),
    (64.14227, -21.93462, 'Reykjavik', 'seen'),
    (64.14326, -21.85294, 'Reykjavik', 'seen'),
    (64.14344, -21.86653, 'Reykjavik', 'seen'),
    (64.14412, -21.86921, 'Reykjavik', 'seen'),
    (64.14421, -21.88914, 'Reykjavik', 'seen'),
    (64.14458, -21.92982, 'Reykjavik', 'seen'),
    (64.14518, -21.91134, 'Reykjavik', 'seen'),
    (64.14537, -21.95798, 'Reykjavik', 'seen'),
    (64.14552, -21.95337, 'Reykjavik', 'seen'),
    (64.14561, -21.90444, 'Reykjavik', 'seen'),
    (64.1461, -21.85104, 'Reykjavik', 'seen'),
    (64.14631, -21.89527, 'Reykjavik', 'seen'),
    (64.14633, -21.86525, 'Reykjavik', 'seen'),
    (64.14682, -21.90907, 'Reykjavik', 'seen'),
    (64.14682, -21.90907, 'Reykjavik', 'seen'),
    (64.14688, -21.90859, 'Reykjavik', 'seen'),
    (64.14693, -21.9048, 'Reykjavik', 'seen'),
    (64.14701, -21.90696, 'Reykjavik', 'seen'),
    (64.14711, -21.899, 'Reykjavik', 'seen'),
    (64.14714, -21.86077, 'Reykjavik', 'seen'),
    (64.14749, -21.93012, 'Reykjavik', 'seen'),
    (64.14754, -21.89441, 'Reykjavik', 'seen'),
    (64.14765, -21.95715, 'Reykjavik', 'seen'),
    (64.14771, -21.92371, 'Reykjavik', 'seen'),
    (64.14786, -21.92617, 'Reykjavik', 'seen'),
    (64.14796, -21.86275, 'Reykjavik', 'seen'),
    (64.14799, -21.94848, 'Reykjavik', 'seen'),
    (64.14808, -21.89209, 'Reykjavik', 'seen'),
    (64.14814, -21.88842, 'Reykjavik', 'seen'),
    (64.14818, -21.86344, 'Reykjavik', 'seen'),
    (64.14826, -21.95498, 'Reykjavik', 'seen'),
    (64.14882, -21.86572, 'Reykjavik', 'seen'),
    (64.14887, -21.9765, 'Reykjavik', 'seen'),
    (64.14915, -21.93313, 'Reykjavik', 'seen'),
    (64.14933, -21.87715, 'Reykjavik', 'seen'),
    (64.14961, -21.93982, 'Reykjavik', 'seen'),
    (64.14968, -21.97124, 'Reykjavik', 'seen'),
    (64.14983, -21.97436, 'Reykjavik', 'seen'),
    (64.14988, -21.95298, 'Reykjavik', 'seen'),
    (64.14994, -21.94824, 'Reykjavik', 'seen'),
    (64.15103, -21.87597, 'Reykjavik', 'seen'),
    (64.15122, -21.86865, 'Reykjavik', 'seen'),
    (64.15127, -21.87878, 'Reykjavik', 'seen'),
    (64.15132, -21.88043, 'Reykjavik', 'seen'),
    (64.15255, -21.95426, 'Reykjavik', 'seen'),
    (64.15348, -21.86887, 'Reykjavik', 'seen'),
    (64.15506, -21.87358, 'Reykjavik', 'seen'),
    (-23.04769, -43.521, 'Rio de Janeiro', 'unseen'),
    (-23.01746, -43.46426, 'Rio de Janeiro', 'unseen'),
    (-23.00964, -43.46903, 'Rio de Janeiro', 'unseen'),
    (-23.00472, -43.5603, 'Rio de Janeiro', 'unseen'),
    (-23.00358, -43.32239, 'Rio de Janeiro', 'unseen'),
    (-23.00283, -43.41956, 'Rio de Janeiro', 'unseen'),
    (-23.00007, -43.63712, 'Rio de Janeiro', 'unseen'),
    (-22.99454, -43.27399, 'Rio de Janeiro', 'unseen'),
    (-22.9889, -43.32368, 'Rio de Janeiro', 'unseen'),
    (-22.98854, -43.32428, 'Rio de Janeiro', 'unseen'),
    (-22.9795, -43.22871, 'Rio de Janeiro', 'unseen'),
    (-22.97492, -43.27513, 'Rio de Janeiro', 'unseen'),
    (-22.97338, -43.39181, 'Rio de Janeiro', 'unseen'),
    (-22.97204, -43.49178, 'Rio de Janeiro', 'unseen'),
    (-22.96791, -43.3813, 'Rio de Janeiro', 'unseen'),
    (-22.96592, -43.19836, 'Rio de Janeiro', 'unseen'),
    (-22.9622, -43.20902, 'Rio de Janeiro', 'unseen'),
    (-22.94066, -43.17042, 'Rio de Janeiro', 'unseen'),
    (-22.93239, -43.37316, 'Rio de Janeiro', 'unseen'),
    (-22.93097, -43.55684, 'Rio de Janeiro', 'unseen'),
    (-22.92901, -43.11799, 'Rio de Janeiro', 'unseen'),
    (-22.92684, -43.63031, 'Rio de Janeiro', 'unseen'),
    (-22.92465, -43.36253, 'Rio de Janeiro', 'unseen'),
    (-22.92456, -43.24855, 'Rio de Janeiro', 'unseen'),
    (-22.92126, -43.64918, 'Rio de Janeiro', 'unseen'),
    (-22.9189, -43.26175, 'Rio de Janeiro', 'unseen'),
    (-22.90789, -43.56917, 'Rio de Janeiro', 'unseen'),
    (-22.90609, -43.66538, 'Rio de Janeiro', 'unseen'),
    (-22.89153, -43.34629, 'Rio de Janeiro', 'unseen'),
    (-22.89123, -43.66819, 'Rio de Janeiro', 'unseen'),
    (-22.88774, -43.42139, 'Rio de Janeiro', 'unseen'),
    (-22.88712, -43.3124, 'Rio de Janeiro', 'unseen'),
    (-22.8853, -43.45986, 'Rio de Janeiro', 'unseen'),
    (-22.88349, -43.43195, 'Rio de Janeiro', 'unseen'),
    (-22.88186, -43.25782, 'Rio de Janeiro', 'unseen'),
    (-22.88125, -43.45449, 'Rio de Janeiro', 'unseen'),
    (-22.87702, -43.37019, 'Rio de Janeiro', 'unseen'),
    (-22.87567, -43.46998, 'Rio de Janeiro', 'unseen'),
    (-22.87277, -43.63097, 'Rio de Janeiro', 'unseen'),
    (-22.8708, -43.63209, 'Rio de Janeiro', 'unseen'),
    (-22.87078, -43.36263, 'Rio de Janeiro', 'unseen'),
    (-22.87023, -43.44646, 'Rio de Janeiro', 'unseen'),
    (-22.86759, -43.2893, 'Rio de Janeiro', 'unseen'),
    (-22.86097, -43.53226, 'Rio de Janeiro', 'unseen'),
    (-22.85997, -43.78769, 'Rio de Janeiro', 'unseen'),
    (-22.85639, -43.38627, 'Rio de Janeiro', 'unseen'),
    (-22.85305, -43.64435, 'Rio de Janeiro', 'unseen'),
    (-22.85076, -43.53386, 'Rio de Janeiro', 'unseen'),
    (-22.85063, -43.33869, 'Rio de Janeiro', 'unseen'),
    (-22.82828, -43.28755, 'Rio de Janeiro', 'unseen'),
    (-22.82471, -43.55902, 'Rio de Janeiro', 'unseen'),
    (-22.81829, -43.64769, 'Rio de Janeiro', 'unseen'),
    (-22.816, -43.61107, 'Rio de Janeiro', 'unseen'),
    (-22.81481, -43.29174, 'Rio de Janeiro', 'unseen'),
    (-22.81196, -43.38808, 'Rio de Janeiro', 'unseen'),
    (-22.80495, -43.35147, 'Rio de Janeiro', 'unseen'),
    (-22.80112, -43.65207, 'Rio de Janeiro', 'unseen'),
    (-22.80093, -43.30529, 'Rio de Janeiro', 'unseen'),
    (-22.8003, -43.36385, 'Rio de Janeiro', 'unseen'),
    (-22.79903, -43.43353, 'Rio de Janeiro', 'unseen'),
    (-22.79746, -43.63628, 'Rio de Janeiro', 'unseen'),
    (-22.79662, -43.42529, 'Rio de Janeiro', 'unseen'),
    (-22.79565, -43.32613, 'Rio de Janeiro', 'unseen'),
    (-22.7936, -43.30973, 'Rio de Janeiro', 'unseen'),
    (-22.79163, -43.30754, 'Rio de Janeiro', 'unseen'),
    (-22.79095, -43.29364, 'Rio de Janeiro', 'unseen'),
    (-22.78741, -43.1853, 'Rio de Janeiro', 'unseen'),
    (-22.78564, -43.4323, 'Rio de Janeiro', 'unseen'),
    (-22.78131, -43.38538, 'Rio de Janeiro', 'unseen'),
    (-22.77964, -43.71878, 'Rio de Janeiro', 'unseen'),
    (-22.77836, -43.38854, 'Rio de Janeiro', 'unseen'),
    (-22.7768, -43.34742, 'Rio de Janeiro', 'unseen'),
    (-22.77612, -43.31124, 'Rio de Janeiro', 'unseen'),
    (-22.77442, -43.37551, 'Rio de Janeiro', 'unseen'),
    (-22.76795, -43.28676, 'Rio de Janeiro', 'unseen'),
    (-22.76346, -43.35049, 'Rio de Janeiro', 'unseen'),
    (-22.75745, -43.35867, 'Rio de Janeiro', 'unseen'),
    (-22.7559, -43.43734, 'Rio de Janeiro', 'unseen'),
    (-22.75543, -43.50053, 'Rio de Janeiro', 'unseen'),
    (-22.75044, -43.6251, 'Rio de Janeiro', 'unseen'),
    (-22.74324, -43.61006, 'Rio de Janeiro', 'unseen'),
    (-22.743, -43.45901, 'Rio de Janeiro', 'unseen'),
    (-22.74007, -43.69136, 'Rio de Janeiro', 'unseen'),
    (-22.73688, -43.74025, 'Rio de Janeiro', 'unseen'),
    (-22.73175, -43.36271, 'Rio de Janeiro', 'unseen'),
    (-22.7304, -43.59431, 'Rio de Janeiro', 'unseen'),
    (41.79211, 12.35459, 'Rome', 'seen'),
    (41.79623, 12.40617, 'Rome', 'seen'),
    (41.79788, 12.51528, 'Rome', 'seen'),
    (41.80513, 12.60885, 'Rome', 'seen'),
    (41.80545, 12.42199, 'Rome', 'seen'),
    (41.80612, 12.42962, 'Rome', 'seen'),
    (41.80949, 12.55692, 'Rome', 'seen'),
    (41.8141, 12.60925, 'Rome', 'seen'),
    (41.81626, 12.48396, 'Rome', 'seen'),
    (41.81715, 12.38229, 'Rome', 'seen'),
    (41.82367, 12.49214, 'Rome', 'seen'),
    (41.82453, 12.60757, 'Rome', 'seen'),
    (41.82561, 12.56047, 'Rome', 'seen'),
    (41.82624, 12.41233, 'Rome', 'seen'),
    (41.83133, 12.4128, 'Rome', 'seen'),
    (41.83434, 12.46946, 'Rome', 'seen'),
    (41.83511, 12.59916, 'Rome', 'seen'),
    (41.83515, 12.53086, 'Rome', 'seen'),
    (41.83603, 12.5516, 'Rome', 'seen'),
    (41.83754, 12.59813, 'Rome', 'seen'),
    (41.83958, 12.3536, 'Rome', 'seen'),
    (41.84184, 12.35204, 'Rome', 'seen'),
    (41.84196, 12.47235, 'Rome', 'seen'),
    (41.84379, 12.51516, 'Rome', 'seen'),
    (41.84836, 12.48666, 'Rome', 'seen'),
    (41.84846, 12.36297, 'Rome', 'seen'),
    (41.85434, 12.55988, 'Rome', 'seen'),
    (41.86112, 12.47842, 'Rome', 'seen'),
    (41.86285, 12.45363, 'Rome', 'seen'),
    (41.86353, 12.40623, 'Rome', 'seen'),
    (41.86433, 12.4251, 'Rome', 'seen'),
    (41.86709, 12.54847, 'Rome', 'seen'),
    (41.86892, 12.35858, 'Rome', 'seen'),
    (41.87106, 12.53509, 'Rome', 'seen'),
    (41.87286, 12.42539, 'Rome', 'seen'),
    (41.87339, 12.57239, 'Rome', 'seen'),
    (41.87453, 12.46038, 'Rome', 'seen'),
    (41.87468, 12.59498, 'Rome', 'seen'),
    (41.87503, 12.43194, 'Rome', 'seen'),
    (41.8776, 12.54702, 'Rome', 'seen'),
    (41.87836, 12.40688, 'Rome', 'seen'),
    (41.87848, 12.55901, 'Rome', 'seen'),
    (41.87927, 12.57859, 'Rome', 'seen'),
    (41.87979, 12.59602, 'Rome', 'seen'),
    (41.88169, 12.35834, 'Rome', 'seen'),
    (41.88316, 12.51762, 'Rome', 'seen'),
    (41.8833, 12.60768, 'Rome', 'seen'),
    (41.88451, 12.55435, 'Rome', 'seen'),
    (41.88725, 12.53408, 'Rome', 'seen'),
    (41.88774, 12.49437, 'Rome', 'seen'),
    (41.88791, 12.49005, 'Rome', 'seen'),
    (41.88965, 12.46117, 'Rome', 'seen'),
    (41.89068, 12.47252, 'Rome', 'seen'),
    (41.89235, 12.51279, 'Rome', 'seen'),
    (41.89278, 12.36747, 'Rome', 'seen'),
    (41.89401, 12.5928, 'Rome', 'seen'),
    (41.89483, 12.60392, 'Rome', 'seen'),
    (41.89572, 12.40932, 'Rome', 'seen'),
    (41.89599, 12.36845, 'Rome', 'seen'),
    (41.89639, 12.44843, 'Rome', 'seen'),
    (41.89763, 12.42226, 'Rome', 'seen'),
    (41.90106, 12.56534, 'Rome', 'seen'),
    (41.90389, 12.38548, 'Rome', 'seen'),
    (41.90404, 12.39633, 'Rome', 'seen'),
    (41.90468, 12.53251, 'Rome', 'seen'),
    (41.90488, 12.476, 'Rome', 'seen'),
    (41.90804, 12.47595, 'Rome', 'seen'),
    (41.90863, 12.54465, 'Rome', 'seen'),
    (41.90892, 12.46624, 'Rome', 'seen'),
    (41.91345, 12.57186, 'Rome', 'seen'),
    (41.91374, 12.39614, 'Rome', 'seen'),
    (41.91865, 12.41265, 'Rome', 'seen'),
    (41.92036, 12.47442, 'Rome', 'seen'),
    (41.9204, 12.35585, 'Rome', 'seen'),
    (41.92271, 12.53278, 'Rome', 'seen'),
    (41.92323, 12.5529, 'Rome', 'seen'),
    (41.92425, 12.38424, 'Rome', 'seen'),
    (41.92771, 12.44646, 'Rome', 'seen'),
    (41.92801, 12.40679, 'Rome', 'seen'),
    (41.92929, 12.43808, 'Rome', 'seen'),
    (41.92964, 12.41957, 'Rome', 'seen'),
    (41.93099, 12.56412, 'Rome', 'seen'),
    (41.93696, 12.51356, 'Rome', 'seen'),
    (41.93761, 12.49646, 'Rome', 'seen'),
    (41.94055, 12.59672, 'Rome', 'seen'),
    (41.94248, 12.60347, 'Rome', 'seen'),
    (41.94265, 12.56306, 'Rome', 'seen'),
    (41.94398, 12.55893, 'Rome', 'seen'),
    (41.94455, 12.53208, 'Rome', 'seen'),
    (41.94459, 12.36396, 'Rome', 'seen'),
    (41.94509, 12.38756, 'Rome', 'seen'),
    (41.94596, 12.46496, 'Rome', 'seen'),
    (41.94764, 12.46621, 'Rome', 'seen'),
    (41.94946, 12.58717, 'Rome', 'seen'),
    (41.95094, 12.42975, 'Rome', 'seen'),
    (41.95545, 12.5193, 'Rome', 'seen'),
    (41.95837, 12.61354, 'Rome', 'seen'),
    (41.95851, 12.48532, 'Rome', 'seen'),
    (41.96029, 12.55129, 'Rome', 'seen'),
    (41.96227, 12.3638, 'Rome', 'seen'),
    (41.97066, 12.61729, 'Rome', 'seen'),
    (41.97572, 12.61828, 'Rome', 'seen'),
    (41.97736, 12.58352, 'Rome', 'seen'),
    (41.97909, 12.42538, 'Rome', 'seen'),
    (41.97912, 12.41847, 'Rome', 'seen'),
    (41.97917, 12.41205, 'Rome', 'seen'),
    (41.98304, 12.5164, 'Rome', 'seen'),
    (41.98376, 12.50724, 'Rome', 'seen'),
    (41.98456, 12.445, 'Rome', 'seen'),
    (41.98938, 12.4155, 'Rome', 'seen'),
    (41.99155, 12.48115, 'Rome', 'seen'),
    (41.99912, 12.47987, 'Rome', 'seen'),
    (41.26054, 36.32332, 'Samsun', 'seen'),
    (41.2608, 36.34258, 'Samsun', 'seen'),
    (41.26107, 36.28673, 'Samsun', 'seen'),
    (41.26182, 36.37241, 'Samsun', 'seen'),
    (41.26241, 36.33882, 'Samsun', 'seen'),
    (41.26324, 36.36425, 'Samsun', 'seen'),
    (41.26434, 36.32785, 'Samsun', 'seen'),
    (41.26436, 36.31153, 'Samsun', 'seen'),
    (41.26465, 36.37025, 'Samsun', 'seen'),
    (41.26477, 36.36177, 'Samsun', 'seen'),
    (41.26539, 36.28179, 'Samsun', 'seen'),
    (41.26549, 36.33395, 'Samsun', 'seen'),
    (41.2657, 36.32567, 'Samsun', 'seen'),
    (41.2664, 36.29007, 'Samsun', 'seen'),
    (41.26666, 36.34332, 'Samsun', 'seen'),
    (41.26706, 36.36621, 'Samsun', 'seen'),
    (41.26722, 36.34855, 'Samsun', 'seen'),
    (41.268, 36.32306, 'Samsun', 'seen'),
    (41.26881, 36.37551, 'Samsun', 'seen'),
    (41.26902, 36.31418, 'Samsun', 'seen'),
    (41.26945, 36.35392, 'Samsun', 'seen'),
    (41.2698, 36.35827, 'Samsun', 'seen'),
    (41.27016, 36.35795, 'Samsun', 'seen'),
    (41.27096, 36.35255, 'Samsun', 'seen'),
    (41.27142, 36.28607, 'Samsun', 'seen'),
    (41.2715, 36.32518, 'Samsun', 'seen'),
    (41.27362, 36.35021, 'Samsun', 'seen'),
    (41.27529, 36.32932, 'Samsun', 'seen'),
    (41.27644, 36.2898, 'Samsun', 'seen'),
    (41.27737, 36.35475, 'Samsun', 'seen'),
    (41.27827, 36.33468, 'Samsun', 'seen'),
    (41.27922, 36.32556, 'Samsun', 'seen'),
    (41.2795, 36.29434, 'Samsun', 'seen'),
    (41.28003, 36.29356, 'Samsun', 'seen'),
    (41.28008, 36.28482, 'Samsun', 'seen'),
    (41.28031, 36.34625, 'Samsun', 'seen'),
    (41.28061, 36.33491, 'Samsun', 'seen'),
    (41.2813, 36.29604, 'Samsun', 'seen'),
    (41.28179, 36.34893, 'Samsun', 'seen'),
    (41.28219, 36.29941, 'Samsun', 'seen'),
    (41.28233, 36.3035, 'Samsun', 'seen'),
    (41.28568, 36.29957, 'Samsun', 'seen'),
    (41.28569, 36.34445, 'Samsun', 'seen'),
    (41.28608, 36.33695, 'Samsun', 'seen'),
    (41.28615, 36.28268, 'Samsun', 'seen'),
    (41.28801, 36.33288, 'Samsun', 'seen'),
    (41.28811, 36.31899, 'Samsun', 'seen'),
    (41.28818, 36.3407, 'Samsun', 'seen'),
    (41.28821, 36.32842, 'Samsun', 'seen'),
    (41.28835, 36.30575, 'Samsun', 'seen'),
    (41.28892, 36.28519, 'Samsun', 'seen'),
    (41.29606, 36.33488, 'Samsun', 'seen'),
    (41.29636, 36.33478, 'Samsun', 'seen'),
    (41.29743, 36.33258, 'Samsun', 'seen'),
    (41.29801, 36.29433, 'Samsun', 'seen'),
    (41.29987, 36.31529, 'Samsun', 'seen'),
    (41.3, 36.29429, 'Samsun', 'seen'),
    (41.30075, 36.28407, 'Samsun', 'seen'),
    (41.30125, 36.2869, 'Samsun', 'seen'),
    (41.30165, 36.33151, 'Samsun', 'seen'),
    (41.30182, 36.29155, 'Samsun', 'seen'),
    (41.30329, 36.32183, 'Samsun', 'seen'),
    (41.30337, 36.28352, 'Samsun', 'seen'),
    (41.30413, 36.33042, 'Samsun', 'seen'),
    (41.30475, 36.31698, 'Samsun', 'seen'),
    (41.3055, 36.28938, 'Samsun', 'seen'),
    (41.30559, 36.28714, 'Samsun', 'seen'),
    (41.30597, 36.2918, 'Samsun', 'seen'),
    (41.30621, 36.3205, 'Samsun', 'seen'),
    (41.30624, 36.31418, 'Samsun', 'seen'),
    (41.30672, 36.3261, 'Samsun', 'seen'),
    (41.30792, 36.30141, 'Samsun', 'seen'),
    (41.30829, 36.31419, 'Samsun', 'seen'),
    (41.30857, 36.28802, 'Samsun', 'seen'),
    (41.30959, 36.31321, 'Samsun', 'seen'),
    (41.31288, 36.30302, 'Samsun', 'seen'),
    (41.31288, 36.30302, 'Samsun', 'seen'),
    (41.31452, 36.28911, 'Samsun', 'seen'),
    (41.31462, 36.28898, 'Samsun', 'seen'),
    (41.31475, 36.29039, 'Samsun', 'seen'),
    (41.31677, 36.29689, 'Samsun', 'seen'),
    (41.31762, 36.28286, 'Samsun', 'seen'),
    (41.31775, 36.30889, 'Samsun', 'seen'),
    (41.31985, 36.32747, 'Samsun', 'seen'),
    (41.31995, 36.31528, 'Samsun', 'seen'),
    (41.32062, 36.31767, 'Samsun', 'seen'),
    (41.32081, 36.32351, 'Samsun', 'seen'),
    (41.3216, 36.28759, 'Samsun', 'seen'),
    (41.32524, 36.28094, 'Samsun', 'seen'),
    (41.32974, 36.29416, 'Samsun', 'seen'),
    (37.48024, 126.97164, 'Seoul', 'unseen'),
    (37.48209, 126.96659, 'Seoul', 'unseen'),
    (37.48302, 126.92835, 'Seoul', 'unseen'),
    (37.48322, 127.06319, 'Seoul', 'unseen'),
    (37.48359, 126.9595, 'Seoul', 'unseen'),
    (37.48468, 126.95396, 'Seoul', 'unseen'),
    (37.48487, 127.0655, 'Seoul', 'unseen'),
    (37.48554, 126.97873, 'Seoul', 'unseen'),
    (37.48571, 126.94083, 'Seoul', 'unseen'),
    (37.48596, 126.96879, 'Seoul', 'unseen'),
    (37.48674, 126.95521, 'Seoul', 'unseen'),
    (37.48782, 127.0447, 'Seoul', 'unseen'),
    (37.48814, 126.94062, 'Seoul', 'unseen'),
    (37.4893, 127.07739, 'Seoul', 'unseen'),
    (37.49125, 126.90848, 'Seoul', 'unseen'),
    (37.49172, 127.06843, 'Seoul', 'unseen'),
    (37.49175, 127.05705, 'Seoul', 'unseen'),
    (37.49199, 126.94387, 'Seoul', 'unseen'),
    (37.49482, 126.96045, 'Seoul', 'unseen'),
    (37.49668, 126.99521, 'Seoul', 'unseen'),
    (37.50127, 127.00156, 'Seoul', 'unseen'),
    (37.50194, 127.0952, 'Seoul', 'unseen'),
    (37.50227, 127.04483, 'Seoul', 'unseen'),
    (37.50269, 127.04886, 'Seoul', 'unseen'),
    (37.50291, 127.08559, 'Seoul', 'unseen'),
    (37.50339, 127.08049, 'Seoul', 'unseen'),
    (37.50365, 126.89794, 'Seoul', 'unseen'),
    (37.50905, 126.96719, 'Seoul', 'unseen'),
    (37.50945, 127.02869, 'Seoul', 'unseen'),
    (37.51043, 127.05868, 'Seoul', 'unseen'),
    (37.51091, 127.02093, 'Seoul', 'unseen'),
    (37.51114, 127.05635, 'Seoul', 'unseen'),
    (37.51161, 127.0705, 'Seoul', 'unseen'),
    (37.51167, 127.09032, 'Seoul', 'unseen'),
    (37.51262, 126.95154, 'Seoul', 'unseen'),
    (37.51272, 127.10509, 'Seoul', 'unseen'),
    (37.51466, 127.04915, 'Seoul', 'unseen'),
    (37.51511, 126.99628, 'Seoul', 'unseen'),
    (37.51675, 126.97465, 'Seoul', 'unseen'),
    (37.51724, 126.93313, 'Seoul', 'unseen'),
    (37.5177, 127.02622, 'Seoul', 'unseen'),
    (37.51798, 127.0733, 'Seoul', 'unseen'),
    (37.51949, 127.06075, 'Seoul', 'unseen'),
    (37.52135, 126.88171, 'Seoul', 'unseen'),
    (37.52156, 126.99476, 'Seoul', 'unseen'),
    (37.52185, 126.92752, 'Seoul', 'unseen'),
    (37.52299, 126.89303, 'Seoul', 'unseen'),
    (37.52353, 127.02695, 'Seoul', 'unseen'),
    (37.52445, 126.91808, 'Seoul', 'unseen'),
    (37.52455, 127.04351, 'Seoul', 'unseen'),
    (37.5247, 126.88348, 'Seoul', 'unseen'),
    (37.52648, 127.08991, 'Seoul', 'unseen'),
    (37.52651, 127.02996, 'Seoul', 'unseen'),
    (37.5268, 127.05463, 'Seoul', 'unseen'),
    (37.52773, 127.01256, 'Seoul', 'unseen'),
    (37.52998, 126.927, 'Seoul', 'unseen'),
    (37.53345, 126.91131, 'Seoul', 'unseen'),
    (37.5347, 126.97906, 'Seoul', 'unseen'),
    (37.53493, 126.98467, 'Seoul', 'unseen'),
    (37.53971, 126.98654, 'Seoul', 'unseen'),
    (37.53982, 126.97087, 'Seoul', 'unseen'),
    (37.5409, 127.0131, 'Seoul', 'unseen'),
    (37.5419, 126.93171, 'Seoul', 'unseen'),
    (37.54221, 126.93304, 'Seoul', 'unseen'),
    (37.54232, 127.09243, 'Seoul', 'unseen'),
    (37.54269, 127.06443, 'Seoul', 'unseen'),
    (37.54417, 127.08415, 'Seoul', 'unseen'),
    (37.54486, 126.88278, 'Seoul', 'unseen'),
    (37.54586, 126.9267, 'Seoul', 'unseen'),
    (37.55105, 126.92424, 'Seoul', 'unseen'),
    (37.5516, 127.05808, 'Seoul', 'unseen'),
    (37.55167, 127.10014, 'Seoul', 'unseen'),
    (37.55188, 126.9125, 'Seoul', 'unseen'),
    (37.55215, 127.02852, 'Seoul', 'unseen'),
    (37.55255, 127.01323, 'Seoul', 'unseen'),
    (37.55339, 127.00894, 'Seoul', 'unseen'),
    (37.55446, 126.98292, 'Seoul', 'unseen'),
    (37.55636, 127.0756, 'Seoul', 'unseen'),
    (37.55709, 127.04295, 'Seoul', 'unseen'),
    (37.55814, 126.92564, 'Seoul', 'unseen'),
    (37.55938, 126.92149, 'Seoul', 'unseen'),
    (37.56122, 126.9805, 'Seoul', 'unseen'),
    (37.56607, 127.04377, 'Seoul', 'unseen'),
    (37.56656, 126.96441, 'Seoul', 'unseen'),
    (37.56699, 126.94723, 'Seoul', 'unseen'),
    (37.56881, 127.01783, 'Seoul', 'unseen'),
    (37.57043, 127.06625, 'Seoul', 'unseen'),
    (37.57222, 126.9381, 'Seoul', 'unseen'),
    (37.57579, 126.89783, 'Seoul', 'unseen'),
    (37.57586, 126.97921, 'Seoul', 'unseen'),
    (37.57755, 127.03888, 'Seoul', 'unseen'),
    (37.57841, 126.97958, 'Seoul', 'unseen'),
    (37.57899, 126.97955, 'Seoul', 'unseen'),
    (37.58134, 127.0971, 'Seoul', 'unseen'),
    (37.58168, 127.06787, 'Seoul', 'unseen'),
    (37.58185, 126.94276, 'Seoul', 'unseen'),
    (37.58225, 126.98685, 'Seoul', 'unseen'),
    (37.58343, 126.99626, 'Seoul', 'unseen'),
    (37.58513, 127.05425, 'Seoul', 'unseen'),
    (37.5857, 126.92755, 'Seoul', 'unseen'),
    (37.58686, 126.97301, 'Seoul', 'unseen'),
    (37.58761, 127.00607, 'Seoul', 'unseen'),
    (37.58791, 127.0813, 'Seoul', 'unseen'),
    (37.58852, 126.90807, 'Seoul', 'unseen'),
    (37.59003, 127.0217, 'Seoul', 'unseen'),
    (37.59287, 127.0053, 'Seoul', 'unseen'),
    (37.59385, 127.05421, 'Seoul', 'unseen'),
    (37.59508, 127.08897, 'Seoul', 'unseen'),
    (37.59666, 127.03257, 'Seoul', 'unseen'),
    (37.59814, 127.07701, 'Seoul', 'unseen'),
    (37.59849, 127.03718, 'Seoul', 'unseen'),
    (37.59906, 127.06236, 'Seoul', 'unseen'),
    (37.59968, 126.97564, 'Seoul', 'unseen'),
    (1.27178, 103.62091, 'Singapore', 'unseen'),
    (1.27554, 103.82947, 'Singapore', 'unseen'),
    (1.27996, 103.62167, 'Singapore', 'unseen'),
    (1.28149, 103.86876, 'Singapore', 'unseen'),
    (1.28287, 103.8324, 'Singapore', 'unseen'),
    (1.28579, 103.80261, 'Singapore', 'unseen'),
    (1.28905, 103.83685, 'Singapore', 'unseen'),
    (1.29251, 103.62197, 'Singapore', 'unseen'),
    (1.30132, 103.87559, 'Singapore', 'unseen'),
    (1.30517, 103.65581, 'Singapore', 'unseen'),
    (1.30609, 103.77528, 'Singapore', 'unseen'),
    (1.30861, 103.7576, 'Singapore', 'unseen'),
    (1.31042, 103.86083, 'Singapore', 'unseen'),
    (1.31066, 103.67086, 'Singapore', 'unseen'),
    (1.31119, 104.01394, 'Singapore', 'unseen'),
    (1.31215, 103.79188, 'Singapore', 'unseen'),
    (1.31562, 103.68007, 'Singapore', 'unseen'),
    (1.31703, 103.76449, 'Singapore', 'unseen'),
    (1.31979, 103.79695, 'Singapore', 'unseen'),
    (1.32129, 103.9455, 'Singapore', 'unseen'),
    (1.32185, 103.70323, 'Singapore', 'unseen'),
    (1.32217, 103.70436, 'Singapore', 'unseen'),
    (1.3243, 103.79338, 'Singapore', 'unseen'),
    (1.3246, 103.69381, 'Singapore', 'unseen'),
    (1.32549, 103.64584, 'Singapore', 'unseen'),
    (1.32871, 103.84296, 'Singapore', 'unseen'),
    (1.32966, 103.81405, 'Singapore', 'unseen'),
    (1.33079, 103.64198, 'Singapore', 'unseen'),
    (1.33233, 103.70371, 'Singapore', 'unseen'),
    (1.33275, 103.70848, 'Singapore', 'unseen'),
    (1.33459, 104.02251, 'Singapore', 'unseen'),
    (1.3349, 103.97113, 'Singapore', 'unseen'),
    (1.33492, 103.84008, 'Singapore', 'unseen'),
    (1.33713, 103.77366, 'Singapore', 'unseen'),
    (1.33824, 103.91099, 'Singapore', 'unseen'),
    (1.33861, 103.78341, 'Singapore', 'unseen'),
    (1.33914, 103.92795, 'Singapore', 'unseen'),
    (1.34756, 103.78734, 'Singapore', 'unseen'),
    (1.34759, 103.83621, 'Singapore', 'unseen'),
    (1.34844, 103.77057, 'Singapore', 'unseen'),
    (1.34923, 103.74472, 'Singapore', 'unseen'),
    (1.34951, 103.70865, 'Singapore', 'unseen'),
    (1.35117, 103.88312, 'Singapore', 'unseen'),
    (1.35206, 103.73734, 'Singapore', 'unseen'),
    (1.35245, 103.93415, 'Singapore', 'unseen'),
    (1.35325, 103.8511, 'Singapore', 'unseen'),
    (1.35372, 103.59492, 'Singapore', 'unseen'),
    (1.35383, 103.59488, 'Singapore', 'unseen'),
    (1.35754, 103.60346, 'Singapore', 'unseen'),
    (1.35791, 103.94626, 'Singapore', 'unseen'),
    (1.35863, 103.76332, 'Singapore', 'unseen'),
    (1.3605, 103.88321, 'Singapore', 'unseen'),
    (1.36367, 103.828, 'Singapore', 'unseen'),
    (1.36489, 103.74399, 'Singapore', 'unseen'),
    (1.36877, 103.75174, 'Singapore', 'unseen'),
    (1.37706, 103.8867, 'Singapore', 'unseen'),
    (1.37822, 103.61111, 'Singapore', 'unseen'),
    (1.37942, 103.88624, 'Singapore', 'unseen'),
    (1.38067, 103.89716, 'Singapore', 'unseen'),
    (1.38419, 103.89531, 'Singapore', 'unseen'),
    (1.38965, 103.75135, 'Singapore', 'unseen'),
    (1.39884, 103.88943, 'Singapore', 'unseen'),
    (1.40484, 103.60907, 'Singapore', 'unseen'),
    (1.40601, 103.91273, 'Singapore', 'unseen'),
    (1.41038, 103.74609, 'Singapore', 'unseen'),
    (1.41193, 103.89196, 'Singapore', 'unseen'),
    (1.41311, 103.91153, 'Singapore', 'unseen'),
    (1.41757, 103.90487, 'Singapore', 'unseen'),
    (1.41908, 103.63591, 'Singapore', 'unseen'),
    (1.42492, 103.64424, 'Singapore', 'unseen'),
    (1.4259, 103.84319, 'Singapore', 'unseen'),
    (1.43161, 103.7245, 'Singapore', 'unseen'),
    (1.43374, 103.82956, 'Singapore', 'unseen'),
    (1.43383, 103.93479, 'Singapore', 'unseen'),
    (1.4356, 103.82317, 'Singapore', 'unseen'),
    (1.43595, 103.64694, 'Singapore', 'unseen'),
    (1.43844, 103.80185, 'Singapore', 'unseen'),
    (1.43881, 103.77718, 'Singapore', 'unseen'),
    (1.44236, 103.7174, 'Singapore', 'unseen'),
    (1.44456, 103.79331, 'Singapore', 'unseen'),
    (1.4468, 103.81192, 'Singapore', 'unseen'),
    (1.44692, 103.64259, 'Singapore', 'unseen'),
    (1.45515, 103.65555, 'Singapore', 'unseen'),
    (1.45684, 103.84023, 'Singapore', 'unseen'),
    (1.45685, 103.65227, 'Singapore', 'unseen'),
    (1.45987, 103.91015, 'Singapore', 'unseen'),
    (1.46192, 103.83785, 'Singapore', 'unseen'),
    (1.46218, 103.73889, 'Singapore', 'unseen'),
    (1.46457, 103.93821, 'Singapore', 'unseen'),
    (1.46494, 103.9224, 'Singapore', 'unseen'),
    (1.46814, 103.86907, 'Singapore', 'unseen'),
    (1.46995, 103.61803, 'Singapore', 'unseen'),
    (1.47477, 103.6374, 'Singapore', 'unseen'),
    (1.47478, 103.75877, 'Singapore', 'unseen'),
    (1.4755, 103.74597, 'Singapore', 'unseen'),
    (1.47552, 103.8705, 'Singapore', 'unseen'),
    (59.87859, 30.25968, 'St. Petersburg', 'seen'),
    (59.87895, 30.39598, 'St. Petersburg', 'seen'),
    (59.87895, 30.39598, 'St. Petersburg', 'seen'),
    (59.88041, 30.31968, 'St. Petersburg', 'seen'),
    (59.8873, 30.38356, 'St. Petersburg', 'seen'),
    (59.88869, 30.31707, 'St. Petersburg', 'seen'),
    (59.88879, 30.32345, 'St. Petersburg', 'seen'),
    (59.89106, 30.3112, 'St. Petersburg', 'seen'),
    (59.89459, 30.28418, 'St. Petersburg', 'seen'),
    (59.89473, 30.33995, 'St. Petersburg', 'seen'),
    (59.89488, 30.25018, 'St. Petersburg', 'seen'),
    (59.89841, 30.34168, 'St. Petersburg', 'seen'),
    (59.89854, 30.36589, 'St. Petersburg', 'seen'),
    (59.89982, 30.24513, 'St. Petersburg', 'seen'),
    (59.89983, 30.29959, 'St. Petersburg', 'seen'),
    (59.90255, 30.32063, 'St. Petersburg', 'seen'),
    (59.90406, 30.27785, 'St. Petersburg', 'seen'),
    (59.90455, 30.34382, 'St. Petersburg', 'seen'),
    (59.90486, 30.28833, 'St. Petersburg', 'seen'),
    (59.90851, 30.30262, 'St. Petersburg', 'seen'),
    (59.90858, 30.36898, 'St. Petersburg', 'seen'),
    (59.90869, 30.35379, 'St. Petersburg', 'seen'),
    (59.90878, 30.29238, 'St. Petersburg', 'seen'),
    (59.90916, 30.3048, 'St. Petersburg', 'seen'),
    (59.91094, 30.31856, 'St. Petersburg', 'seen'),
    (59.91148, 30.30051, 'St. Petersburg', 'seen'),
    (59.91301, 30.26892, 'St. Petersburg', 'seen'),
    (59.91307, 30.27068, 'St. Petersburg', 'seen'),
    (59.91413, 30.38723, 'St. Petersburg', 'seen'),
    (59.91515, 30.30312, 'St. Petersburg', 'seen'),
    (59.91522, 30.33236, 'St. Petersburg', 'seen'),
    (59.91569, 30.38215, 'St. Petersburg', 'seen'),
    (59.91694, 30.32477, 'St. Petersburg', 'seen'),
    (59.91747, 30.3911, 'St. Petersburg', 'seen'),
    (59.91988, 30.27544, 'St. Petersburg', 'seen'),
    (59.92076, 30.34583, 'St. Petersburg', 'seen'),
    (59.92164, 30.33202, 'St. Petersburg', 'seen'),
    (59.92208, 30.36005, 'St. Petersburg', 'seen'),
    (59.92441, 30.30667, 'St. Petersburg', 'seen'),
    (59.92526, 30.29892, 'St. Petersburg', 'seen'),
    (59.92542, 30.29103, 'St. Petersburg', 'seen'),
    (59.92544, 30.30185, 'St. Petersburg', 'seen'),
    (59.92736, 30.27991, 'St. Petersburg', 'seen'),
    (59.92766, 30.28051, 'St. Petersburg', 'seen'),
    (59.9283, 30.35947, 'St. Petersburg', 'seen'),
    (59.92881, 30.35643, 'St. Petersburg', 'seen'),
    (59.92889, 30.24811, 'St. Petersburg', 'seen'),
    (59.92899, 30.30269, 'St. Petersburg', 'seen'),
    (59.93059, 30.28495, 'St. Petersburg', 'seen'),
    (59.93061, 30.24517, 'St. Petersburg', 'seen'),
    (59.93236, 30.23995, 'St. Petersburg', 'seen'),
    (59.93275, 30.36542, 'St. Petersburg', 'seen'),
    (59.93314, 30.34373, 'St. Petersburg', 'seen'),
    (59.93357, 30.37617, 'St. Petersburg', 'seen'),
    (59.93442, 30.39236, 'St. Petersburg', 'seen'),
    (59.9358, 30.24335, 'St. Petersburg', 'seen'),
    (59.93734, 30.30884, 'St. Petersburg', 'seen'),
    (59.93865, 30.36051, 'St. Petersburg', 'seen'),
    (59.93873, 30.28009, 'St. Petersburg', 'seen'),
    (59.93877, 30.34507, 'St. Petersburg', 'seen'),
    (59.93882, 30.32595, 'St. Petersburg', 'seen'),
    (59.93903, 30.3939, 'St. Petersburg', 'seen'),
    (59.9394, 30.37661, 'St. Petersburg', 'seen'),
    (59.93946, 30.29314, 'St. Petersburg', 'seen'),
    (59.94158, 30.29048, 'St. Petersburg', 'seen'),
    (59.94185, 30.2667, 'St. Petersburg', 'seen'),
    (59.94205, 30.33795, 'St. Petersburg', 'seen'),
    (59.94308, 30.37794, 'St. Petersburg', 'seen'),
    (59.94309, 30.39128, 'St. Petersburg', 'seen'),
    (59.94345, 30.30085, 'St. Petersburg', 'seen'),
    (59.94362, 30.37475, 'St. Petersburg', 'seen'),
    (59.94383, 30.23551, 'St. Petersburg', 'seen'),
    (59.94397, 30.35322, 'St. Petersburg', 'seen'),
    (59.9453, 30.3299, 'St. Petersburg', 'seen'),
    (59.94642, 30.36267, 'St. Petersburg', 'seen'),
    (59.94658, 30.2678, 'St. Petersburg', 'seen'),
    (59.9469, 30.37873, 'St. Petersburg', 'seen'),
    (59.94713, 30.2833, 'St. Petersburg', 'seen'),
    (59.94725, 30.33392, 'St. Petersburg', 'seen'),
    (59.94905, 30.24386, 'St. Petersburg', 'seen'),
    (59.95237, 30.32948, 'St. Petersburg', 'seen'),
    (59.95274, 30.36306, 'St. Petersburg', 'seen'),
    (59.95515, 30.28267, 'St. Petersburg', 'seen'),
    (59.95555, 30.34153, 'St. Petersburg', 'seen'),
    (59.9557, 30.32233, 'St. Petersburg', 'seen'),
    (59.95659, 30.378, 'St. Petersburg', 'seen'),
    (59.95702, 30.31512, 'St. Petersburg', 'seen'),
    (59.95875, 30.32446, 'St. Petersburg', 'seen'),
    (59.95957, 30.33881, 'St. Petersburg', 'seen'),
    (59.95968, 30.34108, 'St. Petersburg', 'seen'),
    (59.95979, 30.38799, 'St. Petersburg', 'seen'),
    (59.9626, 30.28192, 'St. Petersburg', 'seen'),
    (59.96471, 30.39459, 'St. Petersburg', 'seen'),
    (59.96536, 30.3176, 'St. Petersburg', 'seen'),
    (59.96578, 30.34431, 'St. Petersburg', 'seen'),
    (59.96642, 30.26441, 'St. Petersburg', 'seen'),
    (59.96803, 30.31527, 'St. Petersburg', 'seen'),
    (59.96825, 30.36592, 'St. Petersburg', 'seen'),
    (59.96882, 30.27273, 'St. Petersburg', 'seen'),
    (59.96889, 30.34012, 'St. Petersburg', 'seen'),
    (59.96961, 30.28826, 'St. Petersburg', 'seen'),
    (59.97045, 30.36345, 'St. Petersburg', 'seen'),
    (59.97219, 30.31997, 'St. Petersburg', 'seen'),
    (59.97292, 30.37984, 'St. Petersburg', 'seen'),
    (59.97368, 30.28264, 'St. Petersburg', 'seen'),
    (59.97678, 30.29388, 'St. Petersburg', 'seen'),
    (59.97695, 30.31656, 'St. Petersburg', 'seen'),
    (59.977, 30.27401, 'St. Petersburg', 'seen'),
    (59.97767, 30.36562, 'St. Petersburg', 'seen'),
    (59.2908, 18.09245, 'Stockholm', 'seen'),
    (59.29137, 17.95872, 'Stockholm', 'seen'),
    (59.29161, 18.08256, 'Stockholm', 'seen'),
    (59.29207, 18.10995, 'Stockholm', 'seen'),
    (59.29293, 18.05875, 'Stockholm', 'seen'),
    (59.29336, 17.97433, 'Stockholm', 'seen'),
    (59.29359, 18.10576, 'Stockholm', 'seen'),
    (59.29362, 18.10267, 'Stockholm', 'seen'),
    (59.29417, 18.04813, 'Stockholm', 'seen'),
    (59.29443, 18.04712, 'Stockholm', 'seen'),
    (59.29571, 18.09257, 'Stockholm', 'seen'),
    (59.29592, 18.0083, 'Stockholm', 'seen'),
    (59.29785, 18.00086, 'Stockholm', 'seen'),
    (59.29849, 18.05465, 'Stockholm', 'seen'),
    (59.30064, 17.96516, 'Stockholm', 'seen'),
    (59.30083, 18.10026, 'Stockholm', 'seen'),
    (59.30083, 18.10026, 'Stockholm', 'seen'),
    (59.30243, 17.95772, 'Stockholm', 'seen'),
    (59.3027, 18.00008, 'Stockholm', 'seen'),
    (59.30337, 17.94296, 'Stockholm', 'seen'),
    (59.30597, 18.07646, 'Stockholm', 'seen'),
    (59.30804, 18.11038, 'Stockholm', 'seen'),
    (59.30907, 17.99355, 'Stockholm', 'seen'),
    (59.30916, 17.9894, 'Stockholm', 'seen'),
    (59.30939, 18.08384, 'Stockholm', 'seen'),
    (59.31029, 18.0667, 'Stockholm', 'seen'),
    (59.31044, 18.02747, 'Stockholm', 'seen'),
    (59.31095, 18.07094, 'Stockholm', 'seen'),
    (59.31134, 18.00779, 'Stockholm', 'seen'),
    (59.31403, 18.10327, 'Stockholm', 'seen'),
    (59.31404, 18.10305, 'Stockholm', 'seen'),
    (59.31464, 18.05897, 'Stockholm', 'seen'),
    (59.31723, 18.04285, 'Stockholm', 'seen'),
    (59.31758, 17.97005, 'Stockholm', 'seen'),
    (59.31763, 18.00625, 'Stockholm', 'seen'),
    (59.31807, 17.95075, 'Stockholm', 'seen'),
    (59.31913, 18.0596, 'Stockholm', 'seen'),
    (59.31937, 18.07062, 'Stockholm', 'seen'),
    (59.32001, 17.9453, 'Stockholm', 'seen'),
    (59.32096, 18.04916, 'Stockholm', 'seen'),
    (59.321, 18.07006, 'Stockholm', 'seen'),
    (59.32256, 18.0983, 'Stockholm', 'seen'),
    (59.32279, 18.10116, 'Stockholm', 'seen'),
    (59.32372, 18.10203, 'Stockholm', 'seen'),
    (59.32391, 18.08249, 'Stockholm', 'seen'),
    (59.32482, 18.08663, 'Stockholm', 'seen'),
    (59.32646, 18.07458, 'Stockholm', 'seen'),
    (59.32649, 18.01946, 'Stockholm', 'seen'),
    (59.32724, 18.01323, 'Stockholm', 'seen'),
    (59.32905, 17.94186, 'Stockholm', 'seen'),
    (59.33162, 18.01365, 'Stockholm', 'seen'),
    (59.33191, 17.96314, 'Stockholm', 'seen'),
    (59.33195, 18.00681, 'Stockholm', 'seen'),
    (59.33224, 17.96238, 'Stockholm', 'seen'),
    (59.33268, 17.99162, 'Stockholm', 'seen'),
    (59.33277, 17.97269, 'Stockholm', 'seen'),
    (59.33335, 17.95683, 'Stockholm', 'seen'),
    (59.33608, 18.06109, 'Stockholm', 'seen'),
    (59.33624, 18.01873, 'Stockholm', 'seen'),
    (59.33659, 18.05239, 'Stockholm', 'seen'),
    (59.33682, 18.10091, 'Stockholm', 'seen'),
    (59.33724, 18.01213, 'Stockholm', 'seen'),
    (59.33766, 18.00807, 'Stockholm', 'seen'),
    (59.33863, 17.94228, 'Stockholm', 'seen'),
    (59.33872, 17.97794, 'Stockholm', 'seen'),
    (59.33879, 18.05516, 'Stockholm', 'seen'),
    (59.3388, 18.0211, 'Stockholm', 'seen'),
    (59.33949, 18.11316, 'Stockholm', 'seen'),
    (59.33974, 18.02796, 'Stockholm', 'seen'),
    (59.33989, 17.94174, 'Stockholm', 'seen'),
    (59.34058, 18.0506, 'Stockholm', 'seen'),
    (59.34063, 18.05464, 'Stockholm', 'seen'),
    (59.34206, 18.06635, 'Stockholm', 'seen'),
    (59.34354, 18.03828, 'Stockholm', 'seen'),
    (59.34363, 17.9567, 'Stockholm', 'seen'),
    (59.34409, 18.07227, 'Stockholm', 'seen'),
    (59.34436, 18.04297, 'Stockholm', 'seen'),
    (59.34453, 18.01942, 'Stockholm', 'seen'),
    (59.34465, 18.05459, 'Stockholm', 'seen'),
    (59.34482, 18.01796, 'Stockholm', 'seen'),
    (59.34513, 17.97295, 'Stockholm', 'seen'),
    (59.3473, 18.03648, 'Stockholm', 'seen'),
    (59.3486, 18.06193, 'Stockholm', 'seen'),
    (59.35187, 18.05816, 'Stockholm', 'seen'),
    (59.35309, 18.08793, 'Stockholm', 'seen'),
    (59.35445, 17.98192, 'Stockholm', 'seen'),
    (59.35451, 17.9828, 'Stockholm', 'seen'),
    (59.35662, 17.98519, 'Stockholm', 'seen'),
    (59.35672, 17.96726, 'Stockholm', 'seen'),
    (59.35724, 18.00851, 'Stockholm', 'seen'),
    (59.35826, 18.12997, 'Stockholm', 'seen'),
    (59.35849, 18.10654, 'Stockholm', 'seen'),
    (59.35862, 18.00733, 'Stockholm', 'seen'),
    (59.35868, 17.97616, 'Stockholm', 'seen'),
    (59.35877, 17.95951, 'Stockholm', 'seen'),
    (59.36028, 17.9665, 'Stockholm', 'seen'),
    (59.3603, 17.9833, 'Stockholm', 'seen'),
    (59.36062, 17.99396, 'Stockholm', 'seen'),
    (59.36146, 17.96748, 'Stockholm', 'seen'),
    (59.36207, 18.11413, 'Stockholm', 'seen'),
    (59.36525, 18.01513, 'Stockholm', 'seen'),
    (59.36665, 17.98799, 'Stockholm', 'seen'),
    (59.36878, 18.0093, 'Stockholm', 'seen'),
    (59.36912, 18.06643, 'Stockholm', 'seen'),
    (-34.13273, 150.66477, 'Sydney', 'unseen'),
    (-34.12753, 150.69169, 'Sydney', 'unseen'),
    (-34.122, 150.67048, 'Sydney', 'unseen'),
    (-34.12051, 151.0713, 'Sydney', 'unseen'),
    (-34.11748, 150.80008, 'Sydney', 'unseen'),
    (-34.09034, 150.81955, 'Sydney', 'unseen'),
    (-34.08978, 150.81248, 'Sydney', 'unseen'),
    (-34.0673, 151.12889, 'Sydney', 'unseen'),
    (-34.05513, 150.7238, 'Sydney', 'unseen'),
    (-34.04402, 150.74761, 'Sydney', 'unseen'),
    (-34.03264, 151.10943, 'Sydney', 'unseen'),
    (-34.03243, 151.16835, 'Sydney', 'unseen'),
    (-34.03139, 150.67497, 'Sydney', 'unseen'),
    (-34.00955, 151.11337, 'Sydney', 'unseen'),
    (-34.00599, 151.00819, 'Sydney', 'unseen'),
    (-34.00365, 150.76442, 'Sydney', 'unseen'),
    (-34.00159, 151.07852, 'Sydney', 'unseen'),
    (-33.9879, 150.84574, 'Sydney', 'unseen'),
    (-33.98624, 150.90482, 'Sydney', 'unseen'),
    (-33.97482, 151.21648, 'Sydney', 'unseen'),
    (-33.97461, 150.89705, 'Sydney', 'unseen'),
    (-33.97183, 151.01537, 'Sydney', 'unseen'),
    (-33.97084, 150.86695, 'Sydney', 'unseen'),
    (-33.95441, 151.10187, 'Sydney', 'unseen'),
    (-33.95152, 151.13656, 'Sydney', 'unseen'),
    (-33.95025, 150.88224, 'Sydney', 'unseen'),
    (-33.94365, 151.26172, 'Sydney', 'unseen'),
    (-33.93895, 150.71081, 'Sydney', 'unseen'),
    (-33.93553, 151.23697, 'Sydney', 'unseen'),
    (-33.92695, 151.07693, 'Sydney', 'unseen'),
    (-33.91059, 151.18805, 'Sydney', 'unseen'),
    (-33.91045, 150.86057, 'Sydney', 'unseen'),
    (-33.90794, 151.01171, 'Sydney', 'unseen'),
    (-33.89798, 150.94835, 'Sydney', 'unseen'),
    (-33.89225, 151.23029, 'Sydney', 'unseen'),
    (-33.89018, 151.1823, 'Sydney', 'unseen'),
    (-33.88362, 150.9076, 'Sydney', 'unseen'),
    (-33.87814, 150.92013, 'Sydney', 'unseen'),
    (-33.87709, 150.87531, 'Sydney', 'unseen'),
    (-33.87512, 151.04506, 'Sydney', 'unseen'),
    (-33.87142, 150.93711, 'Sydney', 'unseen'),
    (-33.87128, 151.2302, 'Sydney', 'unseen'),
    (-33.86529, 151.0744, 'Sydney', 'unseen'),
    (-33.86492, 151.05463, 'Sydney', 'unseen'),
    (-33.86034, 151.16992, 'Sydney', 'unseen'),
    (-33.85902, 151.19601, 'Sydney', 'unseen'),
    (-33.85516, 150.9189, 'Sydney', 'unseen'),
    (-33.84555, 151.11007, 'Sydney', 'unseen'),
    (-33.84406, 150.89, 'Sydney', 'unseen'),
    (-33.84081, 150.96664, 'Sydney', 'unseen'),
    (-33.83865, 151.21241, 'Sydney', 'unseen'),
    (-33.83851, 151.12355, 'Sydney', 'unseen'),
    (-33.83643, 150.9551, 'Sydney', 'unseen'),
    (-33.83593, 151.23084, 'Sydney', 'unseen'),
    (-33.83377, 151.0008, 'Sydney', 'unseen'),
    (-33.82642, 150.98738, 'Sydney', 'unseen'),
    (-33.8264, 151.14913, 'Sydney', 'unseen'),
    (-33.81576, 150.7983, 'Sydney', 'unseen'),
    (-33.8123, 151.01722, 'Sydney', 'unseen'),
    (-33.81069, 151.2571, 'Sydney', 'unseen'),
    (-33.80942, 151.13866, 'Sydney', 'unseen'),
    (-33.79877, 150.90398, 'Sydney', 'unseen'),
    (-33.78817, 151.27617, 'Sydney', 'unseen'),
    (-33.788, 150.70332, 'Sydney', 'unseen'),
    (-33.78584, 150.9545, 'Sydney', 'unseen'),
    (-33.78514, 151.04914, 'Sydney', 'unseen'),
    (-33.78409, 150.71073, 'Sydney', 'unseen'),
    (-33.77788, 150.90381, 'Sydney', 'unseen'),
    (-33.77566, 151.20585, 'Sydney', 'unseen'),
    (-33.77407, 150.65398, 'Sydney', 'unseen'),
    (-33.76797, 150.74549, 'Sydney', 'unseen'),
    (-33.7631, 150.81991, 'Sydney', 'unseen'),
    (-33.76134, 150.93018, 'Sydney', 'unseen'),
    (-33.75851, 150.71152, 'Sydney', 'unseen'),
    (-33.75849, 151.06104, 'Sydney', 'unseen'),
    (-33.75582, 151.06597, 'Sydney', 'unseen'),
    (-33.74515, 150.84263, 'Sydney', 'unseen'),
    (-33.74322, 150.78876, 'Sydney', 'unseen'),
    (-33.7423, 150.99954, 'Sydney', 'unseen'),
    (-33.73793, 151.03096, 'Sydney', 'unseen'),
    (-33.73355, 151.11961, 'Sydney', 'unseen'),
    (-33.73091, 151.09122, 'Sydney', 'unseen'),
    (-33.72892, 150.91154, 'Sydney', 'unseen'),
    (-33.72845, 151.05673, 'Sydney', 'unseen'),
    (-33.72656, 150.71666, 'Sydney', 'unseen'),
    (-33.71963, 150.93069, 'Sydney', 'unseen'),
    (-33.71576, 150.99604, 'Sydney', 'unseen'),
    (-33.71541, 150.69744, 'Sydney', 'unseen'),
    (-33.715, 150.83456, 'Sydney', 'unseen'),
    (-33.70453, 150.911, 'Sydney', 'unseen'),
    (-33.70153, 150.86915, 'Sydney', 'unseen'),
    (-33.70015, 151.09995, 'Sydney', 'unseen'),
    (-33.6996, 151.11985, 'Sydney', 'unseen'),
    (-33.69581, 151.1513, 'Sydney', 'unseen'),
    (-33.69368, 150.84031, 'Sydney', 'unseen'),
    (-33.69243, 150.901, 'Sydney', 'unseen'),
    (-33.68913, 151.11621, 'Sydney', 'unseen'),
    (-33.68707, 150.80878, 'Sydney', 'unseen'),
    (-33.6844, 150.76349, 'Sydney', 'unseen'),
    (-33.68063, 150.76655, 'Sydney', 'unseen'),
    (-33.68009, 150.70506, 'Sydney', 'unseen'),
    (-33.67381, 151.31898, 'Sydney', 'unseen'),
    (-33.67342, 150.6773, 'Sydney', 'unseen'),
    (-33.66411, 151.10115, 'Sydney', 'unseen'),
    (-33.65409, 150.81516, 'Sydney', 'unseen'),
    (-33.64677, 151.07357, 'Sydney', 'unseen'),
    (-33.6449, 151.07941, 'Sydney', 'unseen'),
    (-33.64176, 151.25871, 'Sydney', 'unseen'),
    (-33.64034, 150.87293, 'Sydney', 'unseen'),
    (-33.63527, 151.14641, 'Sydney', 'unseen'),
    (-33.63344, 150.84021, 'Sydney', 'unseen'),
    (-33.59783, 150.75714, 'Sydney', 'unseen'),
    (-33.5963, 150.77743, 'Sydney', 'unseen'),
    (-33.58148, 151.18522, 'Sydney', 'unseen'),
    (24.95942, 121.58062, 'Taipei', 'seen'),
    (24.96144, 121.52926, 'Taipei', 'seen'),
    (24.96386, 121.64905, 'Taipei', 'seen'),
    (24.96718, 121.5271, 'Taipei', 'seen'),
    (24.96733, 121.64545, 'Taipei', 'seen'),
    (24.96883, 121.51836, 'Taipei', 'seen'),
    (24.97318, 121.47531, 'Taipei', 'seen'),
    (24.97318, 121.5161, 'Taipei', 'seen'),
    (24.97453, 121.51875, 'Taipei', 'seen'),
    (24.97472, 121.65327, 'Taipei', 'seen'),
    (24.9763, 121.54376, 'Taipei', 'seen'),
    (24.98138, 121.56407, 'Taipei', 'seen'),
    (24.9849, 121.51614, 'Taipei', 'seen'),
    (24.98673, 121.45314, 'Taipei', 'seen'),
    (24.98731, 121.52456, 'Taipei', 'seen'),
    (24.99026, 121.46447, 'Taipei', 'seen'),
    (24.99155, 121.54259, 'Taipei', 'seen'),
    (24.99175, 121.4999, 'Taipei', 'seen'),
    (24.99327, 121.56008, 'Taipei', 'seen'),
    (24.99341, 121.61465, 'Taipei', 'seen'),
    (24.99375, 121.54563, 'Taipei', 'seen'),
    (24.99631, 121.5605, 'Taipei', 'seen'),
    (24.99632, 121.5404, 'Taipei', 'seen'),
    (24.99812, 121.59259, 'Taipei', 'seen'),
    (25.00558, 121.53732, 'Taipei', 'seen'),
    (25.00589, 121.5768, 'Taipei', 'seen'),
    (25.00647, 121.47986, 'Taipei', 'seen'),
    (25.00662, 121.63, 'Taipei', 'seen'),
    (25.01003, 121.48962, 'Taipei', 'seen'),
    (25.01324, 121.52983, 'Taipei', 'seen'),
    (25.01646, 121.65488, 'Taipei', 'seen'),
    (25.0167, 121.51633, 'Taipei', 'seen'),
    (25.01836, 121.56011, 'Taipei', 'seen'),
    (25.02197, 121.57504, 'Taipei', 'seen'),
    (25.02573, 121.45931, 'Taipei', 'seen'),
    (25.02631, 121.54044, 'Taipei', 'seen'),
    (25.02741, 121.52608, 'Taipei', 'seen'),
    (25.02927, 121.49834, 'Taipei', 'seen'),
    (25.0316, 121.57383, 'Taipei', 'seen'),
    (25.03356, 121.51331, 'Taipei', 'seen'),
    (25.035, 121.61873, 'Taipei', 'seen'),
    (25.03749, 121.47573, 'Taipei', 'seen'),
    (25.0385, 121.49507, 'Taipei', 'seen'),
    (25.04224, 121.57204, 'Taipei', 'seen'),
    (25.04505, 121.58749, 'Taipei', 'seen'),
    (25.05128, 121.59084, 'Taipei', 'seen'),
    (25.05271, 121.61257, 'Taipei', 'seen'),
    (25.0531, 121.55513, 'Taipei', 'seen'),
    (25.05571, 121.466, 'Taipei', 'seen'),
    (25.05813, 121.56476, 'Taipei', 'seen'),
    (25.05855, 121.49813, 'Taipei', 'seen'),
    (25.05988, 121.56249, 'Taipei', 'seen'),
    (25.06138, 121.61921, 'Taipei', 'seen'),
    (25.06181, 121.62918, 'Taipei', 'seen'),
    (25.06233, 121.60677, 'Taipei', 'seen'),
    (25.0632, 121.61599, 'Taipei', 'seen'),
    (25.06826, 121.48247, 'Taipei', 'seen'),
    (25.06884, 121.52806, 'Taipei', 'seen'),
    (25.06919, 121.53812, 'Taipei', 'seen'),
    (25.07394, 121.65094, 'Taipei', 'seen'),
    (25.0743, 121.46969, 'Taipei', 'seen'),
    (25.07557, 121.45827, 'Taipei', 'seen'),
    (25.07604, 121.59031, 'Taipei', 'seen'),
    (25.07825, 121.5213, 'Taipei', 'seen'),
    (25.07849, 121.61669, 'Taipei', 'seen'),
    (25.07905, 121.6491, 'Taipei', 'seen'),
    (25.07965, 121.59525, 'Taipei', 'seen'),
    (25.08107, 121.49439, 'Taipei', 'seen'),
    (25.08143, 121.60093, 'Taipei', 'seen'),
    (25.08154, 121.47705, 'Taipei', 'seen'),
    (25.08451, 121.63671, 'Taipei', 'seen'),
    (25.08839, 121.60244, 'Taipei', 'seen'),
    (25.08895, 121.58525, 'Taipei', 'seen'),
    (25.09183, 121.46292, 'Taipei', 'seen'),
    (25.093, 121.527, 'Taipei', 'seen'),
    (25.09564, 121.49613, 'Taipei', 'seen'),
    (25.09985, 121.46345, 'Taipei', 'seen'),
    (25.1019, 121.55269, 'Taipei', 'seen'),
    (25.10275, 121.55804, 'Taipei', 'seen'),
    (25.10445, 121.45272, 'Taipei', 'seen'),
    (25.10754, 121.64577, 'Taipei', 'seen'),
    (25.10783, 121.52395, 'Taipei', 'seen'),
    (25.11226, 121.52999, 'Taipei', 'seen'),
    (25.1163, 121.6278, 'Taipei', 'seen'),
    (25.11713, 121.46401, 'Taipei', 'seen'),
    (25.11911, 121.52326, 'Taipei', 'seen'),
    (25.12343, 121.61125, 'Taipei', 'seen'),
    (25.12697, 121.57413, 'Taipei', 'seen'),
    (25.1275, 121.55831, 'Taipei', 'seen'),
    (25.12887, 121.60615, 'Taipei', 'seen'),
    (25.13109, 121.51518, 'Taipei', 'seen'),
    (25.13721, 121.56696, 'Taipei', 'seen'),
    (25.14433, 121.56468, 'Taipei', 'seen'),
    (25.15187, 121.50334, 'Taipei', 'seen'),
    (25.15419, 121.50446, 'Taipei', 'seen'),
    (25.15657, 121.54046, 'Taipei', 'seen'),
    (25.16225, 121.4723, 'Taipei', 'seen'),
    (25.16397, 121.45087, 'Taipei', 'seen'),
    (25.17618, 121.65221, 'Taipei', 'seen'),
    (25.17712, 121.55748, 'Taipei', 'seen'),
    (25.18019, 121.48453, 'Taipei', 'seen'),
    (25.18147, 121.65337, 'Taipei', 'seen'),
    (25.18158, 121.57348, 'Taipei', 'seen'),
    (25.18734, 121.49188, 'Taipei', 'seen'),
    (25.18746, 121.48662, 'Taipei', 'seen'),
    (25.19929, 121.46373, 'Taipei', 'seen'),
    (25.20712, 121.48293, 'Taipei', 'seen'),
    (59.41044, 24.71562, 'Tallinn', 'seen'),
    (59.41069, 24.72514, 'Tallinn', 'seen'),
    (59.41121, 24.7561, 'Tallinn', 'seen'),
    (59.41147, 24.71405, 'Tallinn', 'seen'),
    (59.41161, 24.75638, 'Tallinn', 'seen'),
    (59.41186, 24.75108, 'Tallinn', 'seen'),
    (59.41209, 24.71126, 'Tallinn', 'seen'),
    (59.41215, 24.70691, 'Tallinn', 'seen'),
    (59.41223, 24.74408, 'Tallinn', 'seen'),
    (59.4137, 24.70808, 'Tallinn', 'seen'),
    (59.41431, 24.71577, 'Tallinn', 'seen'),
    (59.41518, 24.79759, 'Tallinn', 'seen'),
    (59.41591, 24.73816, 'Tallinn', 'seen'),
    (59.41596, 24.76332, 'Tallinn', 'seen'),
    (59.41598, 24.71199, 'Tallinn', 'seen'),
    (59.41602, 24.75054, 'Tallinn', 'seen'),
    (59.4164, 24.74251, 'Tallinn', 'seen'),
    (59.41671, 24.71659, 'Tallinn', 'seen'),
    (59.41707, 24.75688, 'Tallinn', 'seen'),
    (59.41756, 24.73671, 'Tallinn', 'seen'),
    (59.41803, 24.70314, 'Tallinn', 'seen'),
    (59.41832, 24.75806, 'Tallinn', 'seen'),
    (59.4185, 24.74499, 'Tallinn', 'seen'),
    (59.41908, 24.79867, 'Tallinn', 'seen'),
    (59.41987, 24.7547, 'Tallinn', 'seen'),
    (59.41996, 24.71933, 'Tallinn', 'seen'),
    (59.42039, 24.70887, 'Tallinn', 'seen'),
    (59.42075, 24.71658, 'Tallinn', 'seen'),
    (59.4209, 24.76662, 'Tallinn', 'seen'),
    (59.42114, 24.76821, 'Tallinn', 'seen'),
    (59.42179, 24.77504, 'Tallinn', 'seen'),
    (59.42217, 24.72566, 'Tallinn', 'seen'),
    (59.4236, 24.74059, 'Tallinn', 'seen'),
    (59.42362, 24.74146, 'Tallinn', 'seen'),
    (59.42379, 24.78366, 'Tallinn', 'seen'),
    (59.42393, 24.78154, 'Tallinn', 'seen'),
    (59.42398, 24.75474, 'Tallinn', 'seen'),
    (59.42463, 24.76972, 'Tallinn', 'seen'),
    (59.42476, 24.73931, 'Tallinn', 'seen'),
    (59.42501, 24.70867, 'Tallinn', 'seen'),
    (59.42509, 24.74355, 'Tallinn', 'seen'),
    (59.42553, 24.71541, 'Tallinn', 'seen'),
    (59.42594, 24.78911, 'Tallinn', 'seen'),
    (59.42658, 24.74037, 'Tallinn', 'seen'),
    (59.42682, 24.75049, 'Tallinn', 'seen'),
    (59.42765, 24.77103, 'Tallinn', 'seen'),
    (59.42882, 24.75024, 'Tallinn', 'seen'),
    (59.42908, 24.77449, 'Tallinn', 'seen'),
    (59.42928, 24.76118, 'Tallinn', 'seen'),
    (59.42929, 24.70446, 'Tallinn', 'seen'),
    (59.4293, 24.78852, 'Tallinn', 'seen'),
    (59.42955, 24.78354, 'Tallinn', 'seen'),
    (59.4296, 24.71251, 'Tallinn', 'seen'),
    (59.42962, 24.71613, 'Tallinn', 'seen'),
    (59.42993, 24.71175, 'Tallinn', 'seen'),
    (59.42996, 24.70804, 'Tallinn', 'seen'),
    (59.43002, 24.71366, 'Tallinn', 'seen'),
    (59.43036, 24.76274, 'Tallinn', 'seen'),
    (59.43103, 24.78856, 'Tallinn', 'seen'),
    (59.43273, 24.75488, 'Tallinn', 'seen'),
    (59.43277, 24.78627, 'Tallinn', 'seen'),
    (59.43277, 24.78627, 'Tallinn', 'seen'),
    (59.43316, 24.73168, 'Tallinn', 'seen'),
    (59.43322, 24.769, 'Tallinn', 'seen'),
    (59.43387, 24.78899, 'Tallinn', 'seen'),
    (59.43431, 24.75112, 'Tallinn', 'seen'),
    (59.43438, 24.76736, 'Tallinn', 'seen'),
    (59.43496, 24.78705, 'Tallinn', 'seen'),
    (59.43499, 24.70905, 'Tallinn', 'seen'),
    (59.43503, 24.75768, 'Tallinn', 'seen'),
    (59.43523, 24.77954, 'Tallinn', 'seen'),
    (59.43587, 24.78394, 'Tallinn', 'seen'),
    (59.43592, 24.78385, 'Tallinn', 'seen'),
    (59.43594, 24.70587, 'Tallinn', 'seen'),
    (59.43649, 24.79882, 'Tallinn', 'seen'),
    (59.43714, 24.7136, 'Tallinn', 'seen'),
    (59.43723, 24.79589, 'Tallinn', 'seen'),
    (59.43733, 24.76274, 'Tallinn', 'seen'),
    (59.4375, 24.79222, 'Tallinn', 'seen'),
    (59.43754, 24.71378, 'Tallinn', 'seen'),
    (59.43803, 24.77701, 'Tallinn', 'seen'),
    (59.43832, 24.78758, 'Tallinn', 'seen'),
    (59.43906, 24.722, 'Tallinn', 'seen'),
    (59.43979, 24.70773, 'Tallinn', 'seen'),
    (59.44063, 24.71544, 'Tallinn', 'seen'),
    (59.44112, 24.77972, 'Tallinn', 'seen'),
    (59.44274, 24.76659, 'Tallinn', 'seen'),
    (59.44319, 24.78688, 'Tallinn', 'seen'),
    (59.44334, 24.72418, 'Tallinn', 'seen'),
    (59.44347, 24.7116, 'Tallinn', 'seen'),
    (59.4435, 24.7997, 'Tallinn', 'seen'),
    (59.44401, 24.75604, 'Tallinn', 'seen'),
    (59.44418, 24.75723, 'Tallinn', 'seen'),
    (59.44431, 24.7968, 'Tallinn', 'seen'),
    (59.44502, 24.72055, 'Tallinn', 'seen'),
    (59.44503, 24.7243, 'Tallinn', 'seen'),
    (59.44531, 24.75774, 'Tallinn', 'seen'),
    (59.44583, 24.7184, 'Tallinn', 'seen'),
    (59.4468, 24.7573, 'Tallinn', 'seen'),
    (59.4471, 24.71488, 'Tallinn', 'seen'),
    (59.44717, 24.74352, 'Tallinn', 'seen'),
    (59.44741, 24.74298, 'Tallinn', 'seen'),
    (59.44758, 24.75674, 'Tallinn', 'seen'),
    (59.44758, 24.75674, 'Tallinn', 'seen'),
    (59.4476, 24.7569, 'Tallinn', 'seen'),
    (59.44853, 24.74335, 'Tallinn', 'seen'),
    (59.44875, 24.72161, 'Tallinn', 'seen'),
    (59.44914, 24.72427, 'Tallinn', 'seen'),
    (59.44915, 24.73266, 'Tallinn', 'seen'),
    (59.45004, 24.73729, 'Tallinn', 'seen'),
    (59.45066, 24.73941, 'Tallinn', 'seen'),
    (59.45137, 24.70779, 'Tallinn', 'seen'),
    (59.45155, 24.72932, 'Tallinn', 'seen'),
    (59.45356, 24.70316, 'Tallinn', 'seen'),
    (59.45357, 24.71444, 'Tallinn', 'seen'),
    (35.53298, 139.63108, 'Tokyo', 'seen'),
    (35.54147, 139.68713, 'Tokyo', 'seen'),
    (35.5468, 139.57422, 'Tokyo', 'seen'),
    (35.54921, 139.67108, 'Tokyo', 'seen'),
    (35.55099, 139.73159, 'Tokyo', 'seen'),
    (35.55417, 139.75327, 'Tokyo', 'seen'),
    (35.55493, 139.57387, 'Tokyo', 'seen'),
    (35.55598, 139.62674, 'Tokyo', 'seen'),
    (35.5567, 139.7538, 'Tokyo', 'seen'),
    (35.56082, 139.58637, 'Tokyo', 'seen'),
    (35.56661, 139.70742, 'Tokyo', 'seen'),
    (35.57404, 139.65622, 'Tokyo', 'seen'),
    (35.57678, 139.68824, 'Tokyo', 'seen'),
    (35.57729, 139.71562, 'Tokyo', 'seen'),
    (35.57754, 139.62672, 'Tokyo', 'seen'),
    (35.57864, 139.68448, 'Tokyo', 'seen'),
    (35.58745, 139.75109, 'Tokyo', 'seen'),
    (35.59899, 139.60942, 'Tokyo', 'seen'),
    (35.60494, 139.70759, 'Tokyo', 'seen'),
    (35.60583, 139.64489, 'Tokyo', 'seen'),
    (35.60607, 139.68156, 'Tokyo', 'seen'),
    (35.60795, 139.65142, 'Tokyo', 'seen'),
    (35.60887, 139.67198, 'Tokyo', 'seen'),
    (35.61144, 139.71311, 'Tokyo', 'seen'),
    (35.61209, 139.59982, 'Tokyo', 'seen'),
    (35.62244, 139.66993, 'Tokyo', 'seen'),
    (35.6264, 139.72538, 'Tokyo', 'seen'),
    (35.62754, 139.5843, 'Tokyo', 'seen'),
    (35.62782, 139.77865, 'Tokyo', 'seen'),
    (35.63161, 139.64178, 'Tokyo', 'seen'),
    (35.63422, 139.91137, 'Tokyo', 'seen'),
    (35.6361, 139.70373, 'Tokyo', 'seen'),
    (35.64172, 139.74609, 'Tokyo', 'seen'),
    (35.64547, 139.89903, 'Tokyo', 'seen'),
    (35.647, 139.85228, 'Tokyo', 'seen'),
    (35.65058, 139.63404, 'Tokyo', 'seen'),
    (35.65073, 139.62989, 'Tokyo', 'seen'),
    (35.65162, 139.78581, 'Tokyo', 'seen'),
    (35.65271, 139.63689, 'Tokyo', 'seen'),
    (35.65521, 139.8511, 'Tokyo', 'seen'),
    (35.65545, 139.70566, 'Tokyo', 'seen'),
    (35.65732, 139.67937, 'Tokyo', 'seen'),
    (35.65738, 139.82019, 'Tokyo', 'seen'),
    (35.66806, 139.82831, 'Tokyo', 'seen'),
    (35.66974, 139.82194, 'Tokyo', 'seen'),
    (35.66981, 139.56283, 'Tokyo', 'seen'),
    (35.67369, 139.73602, 'Tokyo', 'seen'),
    (35.67449, 139.72913, 'Tokyo', 'seen'),
    (35.67525, 139.68056, 'Tokyo', 'seen'),
    (35.67569, 139.67792, 'Tokyo', 'seen'),
    (35.67614, 139.66866, 'Tokyo', 'seen'),
    (35.67616, 139.89167, 'Tokyo', 'seen'),
    (35.67778, 139.75295, 'Tokyo', 'seen'),
    (35.67929, 139.59079, 'Tokyo', 'seen'),
    (35.67948, 139.77995, 'Tokyo', 'seen'),
    (35.68175, 139.69353, 'Tokyo', 'seen'),
    (35.68209, 139.57066, 'Tokyo', 'seen'),
    (35.69011, 139.69227, 'Tokyo', 'seen'),
    (35.69099, 139.85011, 'Tokyo', 'seen'),
    (35.69214, 139.75781, 'Tokyo', 'seen'),
    (35.69269, 139.62694, 'Tokyo', 'seen'),
    (35.69313, 139.65023, 'Tokyo', 'seen'),
    (35.69955, 139.77268, 'Tokyo', 'seen'),
    (35.70242, 139.60546, 'Tokyo', 'seen'),
    (35.70242, 139.64145, 'Tokyo', 'seen'),
    (35.70411, 139.77818, 'Tokyo', 'seen'),
    (35.70593, 139.68516, 'Tokyo', 'seen'),
    (35.70732, 139.86612, 'Tokyo', 'seen'),
    (35.70775, 139.76046, 'Tokyo', 'seen'),
    (35.70801, 139.83437, 'Tokyo', 'seen'),
    (35.70834, 139.64216, 'Tokyo', 'seen'),
    (35.7089, 139.82382, 'Tokyo', 'seen'),
    (35.7095, 139.69061, 'Tokyo', 'seen'),
    (35.71495, 139.91737, 'Tokyo', 'seen'),
    (35.71586, 139.91025, 'Tokyo', 'seen'),
    (35.7176, 139.74256, 'Tokyo', 'seen'),
    (35.72138, 139.80559, 'Tokyo', 'seen'),
    (35.72227, 139.56793, 'Tokyo', 'seen'),
    (35.72418, 139.56016, 'Tokyo', 'seen'),
    (35.72756, 139.62006, 'Tokyo', 'seen'),
    (35.72945, 139.61878, 'Tokyo', 'seen'),
    (35.73111, 139.88613, 'Tokyo', 'seen'),
    (35.73201, 139.7687, 'Tokyo', 'seen'),
    (35.73392, 139.73183, 'Tokyo', 'seen'),
    (35.735, 139.74078, 'Tokyo', 'seen'),
    (35.73751, 139.89267, 'Tokyo', 'seen'),
    (35.74099, 139.56139, 'Tokyo', 'seen'),
    (35.74277, 139.80809, 'Tokyo', 'seen'),
    (35.74586, 139.8624, 'Tokyo', 'seen'),
    (35.74926, 139.7416, 'Tokyo', 'seen'),
    (35.74962, 139.67073, 'Tokyo', 'seen'),
    (35.75336, 139.73165, 'Tokyo', 'seen'),
    (35.75857, 139.62347, 'Tokyo', 'seen'),
    (35.76017, 139.58908, 'Tokyo', 'seen'),
    (35.76225, 139.60642, 'Tokyo', 'seen'),
    (35.76229, 139.57924, 'Tokyo', 'seen'),
    (35.76282, 139.89469, 'Tokyo', 'seen'),
    (35.76318, 139.85449, 'Tokyo', 'seen'),
    (35.77671, 139.87405, 'Tokyo', 'seen'),
    (35.78292, 139.83329, 'Tokyo', 'seen'),
    (35.78421, 139.71712, 'Tokyo', 'seen'),
    (35.78672, 139.76075, 'Tokyo', 'seen'),
    (35.78707, 139.81573, 'Tokyo', 'seen'),
    (35.78761, 139.85809, 'Tokyo', 'seen'),
    (35.78931, 139.88746, 'Tokyo', 'seen'),
    (35.78988, 139.88222, 'Tokyo', 'seen'),
    (35.79084, 139.67598, 'Tokyo', 'seen'),
    (35.79265, 139.91178, 'Tokyo', 'seen'),
    (35.79393, 139.62845, 'Tokyo', 'seen'),
    (35.79626, 139.68648, 'Tokyo', 'seen'),
    (35.79944, 139.88347, 'Tokyo', 'seen'),
    (35.80437, 139.91667, 'Tokyo', 'seen'),
    (35.80438, 139.56297, 'Tokyo', 'seen'),
    (35.80912, 139.85796, 'Tokyo', 'seen'),
    (35.80921, 139.87812, 'Tokyo', 'seen'),
    (35.81191, 139.67707, 'Tokyo', 'seen'),
    (35.81525, 139.91953, 'Tokyo', 'seen'),
    (35.81975, 139.59308, 'Tokyo', 'seen'),
    (43.59707, -79.56755, 'Toronto', 'seen'),
    (43.60486, -79.63197, 'Toronto', 'seen'),
    (43.61458, -79.48726, 'Toronto', 'seen'),
    (43.61849, -79.62336, 'Toronto', 'seen'),
    (43.62327, -79.5689, 'Toronto', 'seen'),
    (43.63569, -79.5207, 'Toronto', 'seen'),
    (43.63606, -79.54488, 'Toronto', 'seen'),
    (43.63737, -79.58172, 'Toronto', 'seen'),
    (43.63786, -79.58243, 'Toronto', 'seen'),
    (43.63817, -79.5011, 'Toronto', 'seen'),
    (43.6382, -79.52074, 'Toronto', 'seen'),
    (43.64369, -79.40383, 'Toronto', 'seen'),
    (43.64454, -79.48736, 'Toronto', 'seen'),
    (43.64621, -79.51983, 'Toronto', 'seen'),
    (43.64766, -79.63663, 'Toronto', 'seen'),
    (43.66078, -79.48602, 'Toronto', 'seen'),
    (43.66529, -79.6042, 'Toronto', 'seen'),
    (43.66543, -79.60385, 'Toronto', 'seen'),
    (43.67107, -79.53356, 'Toronto', 'seen'),
    (43.67117, -79.55227, 'Toronto', 'seen'),
    (43.67318, -79.37821, 'Toronto', 'seen'),
    (43.6739, -79.39683, 'Toronto', 'seen'),
    (43.67512, -79.45618, 'Toronto', 'seen'),
    (43.67792, -79.49698, 'Toronto', 'seen'),
    (43.67821, -79.33997, 'Toronto', 'seen'),
    (43.68361, -79.56598, 'Toronto', 'seen'),
    (43.68575, -79.34008, 'Toronto', 'seen'),
    (43.68877, -79.30335, 'Toronto', 'seen'),
    (43.69202, -79.4366, 'Toronto', 'seen'),
    (43.69607, -79.48715, 'Toronto', 'seen'),
    (43.69924, -79.54177, 'Toronto', 'seen'),
    (43.70142, -79.24878, 'Toronto', 'seen'),
    (43.70392, -79.5672, 'Toronto', 'seen'),
    (43.71111, -79.33494, 'Toronto', 'seen'),
    (43.71315, -79.24458, 'Toronto', 'seen'),
    (43.71664, -79.29078, 'Toronto', 'seen'),
    (43.71678, -79.42535, 'Toronto', 'seen'),
    (43.71865, -79.35989, 'Toronto', 'seen'),
    (43.71969, -79.42086, 'Toronto', 'seen'),
    (43.72158, -79.23817, 'Toronto', 'seen'),
    (43.72273, -79.34898, 'Toronto', 'seen'),
    (43.72291, -79.40842, 'Toronto', 'seen'),
    (43.72345, -79.25708, 'Toronto', 'seen'),
    (43.72508, -79.436, 'Toronto', 'seen'),
    (43.73181, -79.31315, 'Toronto', 'seen'),
    (43.73201, -79.50104, 'Toronto', 'seen'),
    (43.73319, -79.47462, 'Toronto', 'seen'),
    (43.73352, -79.44, 'Toronto', 'seen'),
    (43.73496, -79.61768, 'Toronto', 'seen'),
    (43.73666, -79.25605, 'Toronto', 'seen'),
    (43.73684, -79.43669, 'Toronto', 'seen'),
    (43.73696, -79.57262, 'Toronto', 'seen'),
    (43.73712, -79.20546, 'Toronto', 'seen'),
    (43.7376, -79.52841, 'Toronto', 'seen'),
    (43.73792, -79.51959, 'Toronto', 'seen'),
    (43.73796, -79.46383, 'Toronto', 'seen'),
    (43.74397, -79.5931, 'Toronto', 'seen'),
    (43.74599, -79.23351, 'Toronto', 'seen'),
    (43.75046, -79.47077, 'Toronto', 'seen'),
    (43.75129, -79.56963, 'Toronto', 'seen'),
    (43.75152, -79.32717, 'Toronto', 'seen'),
    (43.75194, -79.26509, 'Toronto', 'seen'),
    (43.75253, -79.56357, 'Toronto', 'seen'),
    (43.75442, -79.25095, 'Toronto', 'seen'),
    (43.75495, -79.22152, 'Toronto', 'seen'),
    (43.75531, -79.17739, 'Toronto', 'seen'),
    (43.75788, -79.44334, 'Toronto', 'seen'),
    (43.75906, -79.31698, 'Toronto', 'seen'),
    (43.75913, -79.20586, 'Toronto', 'seen'),
    (43.76037, -79.43116, 'Toronto', 'seen'),
    (43.7606, -79.3877, 'Toronto', 'seen'),
    (43.76194, -79.3271, 'Toronto', 'seen'),
    (43.76583, -79.52593, 'Toronto', 'seen'),
    (43.76919, -79.24624, 'Toronto', 'seen'),
    (43.77071, -79.32739, 'Toronto', 'seen'),
    (43.7719, -79.20122, 'Toronto', 'seen'),
    (43.77501, -79.17738, 'Toronto', 'seen'),
    (43.77541, -79.16755, 'Toronto', 'seen'),
    (43.7772, -79.51355, 'Toronto', 'seen'),
    (43.77827, -79.14708, 'Toronto', 'seen'),
    (43.78106, -79.13005, 'Toronto', 'seen'),
    (43.78341, -79.44784, 'Toronto', 'seen'),
    (43.78391, -79.21594, 'Toronto', 'seen'),
    (43.78456, -79.5463, 'Toronto', 'seen'),
    (43.78701, -79.22125, 'Toronto', 'seen'),
    (43.78821, -79.19139, 'Toronto', 'seen'),
    (43.79075, -79.47528, 'Toronto', 'seen'),
    (43.79138, -79.31195, 'Toronto', 'seen'),
    (43.79285, -79.39164, 'Toronto', 'seen'),
    (43.79537, -79.32957, 'Toronto', 'seen'),
    (43.79958, -79.35317, 'Toronto', 'seen'),
    (43.79978, -79.37311, 'Toronto', 'seen'),
    (43.80039, -79.40434, 'Toronto', 'seen'),
    (43.80121, -79.29364, 'Toronto', 'seen'),
    (43.80315, -79.38117, 'Toronto', 'seen'),
    (43.80365, -79.39135, 'Toronto', 'seen'),
    (43.80386, -79.37953, 'Toronto', 'seen'),
    (43.80464, -79.13635, 'Toronto', 'seen'),
    (43.80494, -79.3161, 'Toronto', 'seen'),
    (43.8059, -79.41953, 'Toronto', 'seen'),
    (43.80761, -79.22572, 'Toronto', 'seen'),
    (43.80953, -79.20721, 'Toronto', 'seen'),
    (43.80981, -79.30699, 'Toronto', 'seen'),
    (43.81019, -79.6145, 'Toronto', 'seen'),
    (43.81128, -79.35361, 'Toronto', 'seen'),
    (43.81275, -79.60419, 'Toronto', 'seen'),
    (43.81388, -79.42699, 'Toronto', 'seen'),
    (43.81472, -79.3452, 'Toronto', 'seen'),
    (43.81887, -79.40001, 'Toronto', 'seen'),
    (43.8194, -79.63108, 'Toronto', 'seen'),
    (43.8263, -79.31067, 'Toronto', 'seen'),
    (43.82857, -79.1435, 'Toronto', 'seen'),
    (43.83316, -79.48089, 'Toronto', 'seen'),
    (43.8395, -79.296, 'Toronto', 'seen'),
    (43.84002, -79.28748, 'Toronto', 'seen'),
    (43.84135, -79.44993, 'Toronto', 'seen'),
    (43.84404, -79.39245, 'Toronto', 'seen'),
    (43.84473, -79.57992, 'Toronto', 'seen'),
    (48.16949, 16.34879, 'Vienna', 'seen'),
    (48.1704, 16.39749, 'Vienna', 'seen'),
    (48.17112, 16.41721, 'Vienna', 'seen'),
    (48.17154, 16.34796, 'Vienna', 'seen'),
    (48.17282, 16.39534, 'Vienna', 'seen'),
    (48.17302, 16.39199, 'Vienna', 'seen'),
    (48.17399, 16.39898, 'Vienna', 'seen'),
    (48.17432, 16.32997, 'Vienna', 'seen'),
    (48.17447, 16.36784, 'Vienna', 'seen'),
    (48.17494, 16.40208, 'Vienna', 'seen'),
    (48.17494, 16.40208, 'Vienna', 'seen'),
    (48.17508, 16.39795, 'Vienna', 'seen'),
    (48.1756, 16.33669, 'Vienna', 'seen'),
    (48.17588, 16.39106, 'Vienna', 'seen'),
    (48.17731, 16.37336, 'Vienna', 'seen'),
    (48.17763, 16.38178, 'Vienna', 'seen'),
    (48.17774, 16.36586, 'Vienna', 'seen'),
    (48.17787, 16.37928, 'Vienna', 'seen'),
    (48.17858, 16.38434, 'Vienna', 'seen'),
    (48.17933, 16.39545, 'Vienna', 'seen'),
    (48.18019, 16.34902, 'Vienna', 'seen'),
    (48.18089, 16.38596, 'Vienna', 'seen'),
    (48.18203, 16.34903, 'Vienna', 'seen'),
    (48.18359, 16.39286, 'Vienna', 'seen'),
    (48.18537, 16.38748, 'Vienna', 'seen'),
    (48.18603, 16.40298, 'Vienna', 'seen'),
    (48.18607, 16.41285, 'Vienna', 'seen'),
    (48.18619, 16.33587, 'Vienna', 'seen'),
    (48.18642, 16.33603, 'Vienna', 'seen'),
    (48.18646, 16.38622, 'Vienna', 'seen'),
    (48.1865, 16.36354, 'Vienna', 'seen'),
    (48.18658, 16.34182, 'Vienna', 'seen'),
    (48.18675, 16.34136, 'Vienna', 'seen'),
    (48.18678, 16.34748, 'Vienna', 'seen'),
    (48.18762, 16.33713, 'Vienna', 'seen'),
    (48.1897, 16.41651, 'Vienna', 'seen'),
    (48.18981, 16.40465, 'Vienna', 'seen'),
    (48.19064, 16.33882, 'Vienna', 'seen'),
    (48.19135, 16.3725, 'Vienna', 'seen'),
    (48.19223, 16.38488, 'Vienna', 'seen'),
    (48.19291, 16.3785, 'Vienna', 'seen'),
    (48.19375, 16.34303, 'Vienna', 'seen'),
    (48.19376, 16.37392, 'Vienna', 'seen'),
    (48.19389, 16.33406, 'Vienna', 'seen'),
    (48.19421, 16.40502, 'Vienna', 'seen'),
    (48.1985, 16.33897, 'Vienna', 'seen'),
    (48.2002, 16.41494, 'Vienna', 'seen'),
    (48.20089, 16.37251, 'Vienna', 'seen'),
    (48.20091, 16.33806, 'Vienna', 'seen'),
    (48.20117, 16.41165, 'Vienna', 'seen'),
    (48.20138, 16.37595, 'Vienna', 'seen'),
    (48.20234, 16.35372, 'Vienna', 'seen'),
    (48.20467, 16.39393, 'Vienna', 'seen'),
    (48.20538, 16.37391, 'Vienna', 'seen'),
    (48.20603, 16.33833, 'Vienna', 'seen'),
    (48.20625, 16.41813, 'Vienna', 'seen'),
    (48.20634, 16.3508, 'Vienna', 'seen'),
    (48.20746, 16.33946, 'Vienna', 'seen'),
    (48.20815, 16.36359, 'Vienna', 'seen'),
    (48.2099, 16.37126, 'Vienna', 'seen'),
    (48.21, 16.35717, 'Vienna', 'seen'),
    (48.21003, 16.35697, 'Vienna', 'seen'),
    (48.21003, 16.35697, 'Vienna', 'seen'),
    (48.21036, 16.37791, 'Vienna', 'seen'),
    (48.21052, 16.38399, 'Vienna', 'seen'),
    (48.21109, 16.38523, 'Vienna', 'seen'),
    (48.2114, 16.39729, 'Vienna', 'seen'),
    (48.21207, 16.36247, 'Vienna', 'seen'),
    (48.21219, 16.33878, 'Vienna', 'seen'),
    (48.21225, 16.39076, 'Vienna', 'seen'),
    (48.21234, 16.37642, 'Vienna', 'seen'),
    (48.21266, 16.38453, 'Vienna', 'seen'),
    (48.2133, 16.36928, 'Vienna', 'seen'),
    (48.21389, 16.41599, 'Vienna', 'seen'),
    (48.21471, 16.35629, 'Vienna', 'seen'),
    (48.21577, 16.38058, 'Vienna', 'seen'),
    (48.21582, 16.40449, 'Vienna', 'seen'),
    (48.21705, 16.33591, 'Vienna', 'seen'),
    (48.21756, 16.40118, 'Vienna', 'seen'),
    (48.21851, 16.33559, 'Vienna', 'seen'),
    (48.2187, 16.34092, 'Vienna', 'seen'),
    (48.21912, 16.36359, 'Vienna', 'seen'),
    (48.21953, 16.4125, 'Vienna', 'seen'),
    (48.21996, 16.37562, 'Vienna', 'seen'),
    (48.22258, 16.37591, 'Vienna', 'seen'),
    (48.22309, 16.34294, 'Vienna', 'seen'),
    (48.22367, 16.39736, 'Vienna', 'seen'),
    (48.2238, 16.40749, 'Vienna', 'seen'),
    (48.22462, 16.34878, 'Vienna', 'seen'),
    (48.22498, 16.40261, 'Vienna', 'seen'),
    (48.22506, 16.40599, 'Vienna', 'seen'),
    (48.22514, 16.37244, 'Vienna', 'seen'),
    (48.22518, 16.34889, 'Vienna', 'seen'),
    (48.22532, 16.3509, 'Vienna', 'seen'),
    (48.22826, 16.39431, 'Vienna', 'seen'),
    (48.22826, 16.39431, 'Vienna', 'seen'),
    (48.22852, 16.34162, 'Vienna', 'seen'),
    (48.22869, 16.40111, 'Vienna', 'seen'),
    (48.23034, 16.33573, 'Vienna', 'seen'),
    (48.23054, 16.41372, 'Vienna', 'seen'),
    (48.23128, 16.346, 'Vienna', 'seen'),
    (48.23129, 16.41505, 'Vienna', 'seen'),
    (48.23132, 16.38489, 'Vienna', 'seen'),
    (48.2318, 16.34142, 'Vienna', 'seen'),
    (48.23196, 16.36406, 'Vienna', 'seen'),
    (48.23232, 16.33942, 'Vienna', 'seen'),
    (48.23356, 16.41961, 'Vienna', 'seen'),
    (48.23384, 16.34433, 'Vienna', 'seen'),
    (48.23454, 16.40914, 'Vienna', 'seen'),
    (48.23535, 16.38169, 'Vienna', 'seen'),
    (48.23774, 16.38376, 'Vienna', 'seen'),
    (48.2398, 16.37131, 'Vienna', 'seen'),
    (48.23988, 16.39968, 'Vienna', 'seen'),
    (48.23991, 16.35505, 'Vienna', 'seen'),
    (47.34031, 8.53157, 'Zurich', 'seen'),
    (47.34272, 8.56734, 'Zurich', 'seen'),
    (47.34344, 8.52305, 'Zurich', 'seen'),
    (47.34396, 8.53513, 'Zurich', 'seen'),
    (47.34577, 8.57269, 'Zurich', 'seen'),
    (47.34667, 8.52029, 'Zurich', 'seen'),
    (47.34719, 8.52822, 'Zurich', 'seen'),
    (47.34763, 8.57917, 'Zurich', 'seen'),
    (47.34847, 8.5623, 'Zurich', 'seen'),
    (47.35221, 8.50637, 'Zurich', 'seen'),
    (47.35225, 8.50401, 'Zurich', 'seen'),
    (47.35392, 8.56369, 'Zurich', 'seen'),
    (47.35555, 8.55368, 'Zurich', 'seen'),
    (47.35606, 8.54982, 'Zurich', 'seen'),
    (47.35616, 8.57093, 'Zurich', 'seen'),
    (47.35622, 8.55021, 'Zurich', 'seen'),
    (47.35656, 8.53484, 'Zurich', 'seen'),
    (47.35702, 8.57845, 'Zurich', 'seen'),
    (47.35823, 8.51632, 'Zurich', 'seen'),
    (47.35826, 8.57539, 'Zurich', 'seen'),
    (47.35877, 8.49738, 'Zurich', 'seen'),
    (47.359, 8.50677, 'Zurich', 'seen'),
    (47.35935, 8.50908, 'Zurich', 'seen'),
    (47.35945, 8.57702, 'Zurich', 'seen'),
    (47.35995, 8.53249, 'Zurich', 'seen'),
    (47.36092, 8.53081, 'Zurich', 'seen'),
    (47.362, 8.57805, 'Zurich', 'seen'),
    (47.36265, 8.49698, 'Zurich', 'seen'),
    (47.36357, 8.53263, 'Zurich', 'seen'),
    (47.36466, 8.55058, 'Zurich', 'seen'),
    (47.36478, 8.53065, 'Zurich', 'seen'),
    (47.36519, 8.55309, 'Zurich', 'seen'),
    (47.36555, 8.53376, 'Zurich', 'seen'),
    (47.3664, 8.52247, 'Zurich', 'seen'),
    (47.36745, 8.56357, 'Zurich', 'seen'),
    (47.36769, 8.54124, 'Zurich', 'seen'),
    (47.36934, 8.5722, 'Zurich', 'seen'),
    (47.36934, 8.5722, 'Zurich', 'seen'),
    (47.37055, 8.51533, 'Zurich', 'seen'),
    (47.37128, 8.52733, 'Zurich', 'seen'),
    (47.37197, 8.51825, 'Zurich', 'seen'),
    (47.37235, 8.57297, 'Zurich', 'seen'),
    (47.37292, 8.56251, 'Zurich', 'seen'),
    (47.37313, 8.49543, 'Zurich', 'seen'),
    (47.37427, 8.5491, 'Zurich', 'seen'),
    (47.37449, 8.53584, 'Zurich', 'seen'),
    (47.37469, 8.54765, 'Zurich', 'seen'),
    (47.37496, 8.52354, 'Zurich', 'seen'),
    (47.37513, 8.52995, 'Zurich', 'seen'),
    (47.37578, 8.53912, 'Zurich', 'seen'),
    (47.37581, 8.5184, 'Zurich', 'seen'),
    (47.37614, 8.52436, 'Zurich', 'seen'),
    (47.3764, 8.54075, 'Zurich', 'seen'),
    (47.37658, 8.49767, 'Zurich', 'seen'),
    (47.37743, 8.57262, 'Zurich', 'seen'),
    (47.37775, 8.53767, 'Zurich', 'seen'),
    (47.3781, 8.52094, 'Zurich', 'seen'),
    (47.38105, 8.56367, 'Zurich', 'seen'),
    (47.38135, 8.53575, 'Zurich', 'seen'),
    (47.38179, 8.52914, 'Zurich', 'seen'),
    (47.38188, 8.5022, 'Zurich', 'seen'),
    (47.38231, 8.50824, 'Zurich', 'seen'),
    (47.38374, 8.50673, 'Zurich', 'seen'),
    (47.3838, 8.50851, 'Zurich', 'seen'),
    (47.38381, 8.55385, 'Zurich', 'seen'),
    (47.38412, 8.54038, 'Zurich', 'seen'),
    (47.38449, 8.5569, 'Zurich', 'seen'),
    (47.38498, 8.52803, 'Zurich', 'seen'),
    (47.38569, 8.55957, 'Zurich', 'seen'),
    (47.38571, 8.57876, 'Zurich', 'seen'),
    (47.38718, 8.5092, 'Zurich', 'seen'),
    (47.38758, 8.53573, 'Zurich', 'seen'),
    (47.3911, 8.55091, 'Zurich', 'seen'),
    (47.39163, 8.50505, 'Zurich', 'seen'),
    (47.3936, 8.53802, 'Zurich', 'seen'),
    (47.39375, 8.49549, 'Zurich', 'seen'),
    (47.39389, 8.50523, 'Zurich', 'seen'),
    (47.39396, 8.52173, 'Zurich', 'seen'),
    (47.39454, 8.53138, 'Zurich', 'seen'),
    (47.39559, 8.54278, 'Zurich', 'seen'),
    (47.39609, 8.50978, 'Zurich', 'seen'),
    (47.39645, 8.52109, 'Zurich', 'seen'),
    (47.39751, 8.54622, 'Zurich', 'seen'),
    (47.39794, 8.51501, 'Zurich', 'seen'),
    (47.39847, 8.5472, 'Zurich', 'seen'),
    (47.3988, 8.50212, 'Zurich', 'seen'),
    (47.39892, 8.55389, 'Zurich', 'seen'),
    (47.399, 8.49915, 'Zurich', 'seen'),
    (47.39924, 8.49833, 'Zurich', 'seen'),
    (47.39952, 8.49679, 'Zurich', 'seen'),
    (47.39953, 8.54317, 'Zurich', 'seen'),
    (47.3997, 8.57279, 'Zurich', 'seen'),
    (47.40069, 8.50179, 'Zurich', 'seen'),
    (47.40108, 8.53782, 'Zurich', 'seen'),
    (47.4022, 8.50265, 'Zurich', 'seen'),
    (47.40361, 8.53911, 'Zurich', 'seen'),
    (47.40377, 8.5486, 'Zurich', 'seen'),
    (47.40547, 8.56769, 'Zurich', 'seen'),
    (47.40552, 8.49907, 'Zurich', 'seen'),
    (47.40593, 8.52759, 'Zurich', 'seen'),
    (47.40599, 8.5588, 'Zurich', 'seen'),
    (47.40673, 8.57478, 'Zurich', 'seen'),
    (47.40681, 8.53934, 'Zurich', 'seen'),
    (47.40776, 8.54325, 'Zurich', 'seen'),
    (47.40802, 8.57448, 'Zurich', 'seen'),
    (47.41001, 8.52948, 'Zurich', 'seen'),
]


# =====================================================================
# Embedded world country outline (Natural Earth 1:110m, coords rounded
# to 2 decimals; zlib + base64). Decoded lazily.
# =====================================================================
_WORLD_B64 = (
    "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"
)


def _decode_world():
    raw = zlib.decompress(base64.b64decode("".join(_WORLD_B64)))
    return json.loads(raw)


def _polys(feat):
    g = feat["geometry"]
    if g["type"] == "Polygon":
        return [g["coordinates"]]
    if g["type"] == "MultiPolygon":
        return g["coordinates"]
    return []


def render_poster(out_stem: str | None = None) -> None:
    """Compact 3-panel poster map: World + Europe + Türkiye zoom.

    - One numbered marker per city (no per-sample dots).
    - Distinct colour per city to disambiguate overlapping locations.
    - Black outline = unseen city; thin white outline = seen city.
    - Right-side 2-column index maps numbers to city names.

    The Europe zoom carves out the dense Central-Europe cluster; the
    Türkiye zoom separates the 5 Turkish cities. The world view marks
    every city, with the zoom regions outlined as rectangles.
    """
    P = PRESETS["poster"]
    if out_stem is None:
        out_stem = "fig11_geographic_scatter_poster"

    # Aggregate to one centroid per city, keeping role.
    city_pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    city_role: dict[str, str] = {}
    for lat, lon, city, role in LOCATIONS:
        city_pts[city].append((lat, lon))
        city_role[city] = role

    # Stable ordering: seen cities first (alphabetical), then unseen.
    seen_cities   = sorted(c for c, r in city_role.items() if r == "seen")
    unseen_cities = sorted(c for c, r in city_role.items() if r == "unseen")
    ordered = seen_cities + unseen_cities
    idx_of = {c: i + 1 for i, c in enumerate(ordered)}

    # Two-tone scheme: seen vs unseen. Numbers carry the per-city
    # identity, so we don't need 40 different hues.
    def _to_rgb(hex_str):
        hex_str = hex_str.lstrip("#")
        return tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
    seen_rgb   = _to_rgb(P["seen_color"])
    unseen_rgb = _to_rgb(P["unseen_color"])
    city_color = {c: (unseen_rgb if city_role[c] == "unseen" else seen_rgb)
                  for c in ordered}

    world = _decode_world()

    # Zoom regions: (lon_min, lon_max, lat_min, lat_max).
    EU_BOX = (-12,  35, 35, 62)     # Western/Central Europe + Iceland
    TR_BOX = ( 25,  45, 35, 43)     # Türkiye + Cyprus

    fig = plt.figure(figsize=P["figsize"])
    # 4 columns side-by-side: World | Europe zoom | Türkiye zoom | Index.
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[2.6, 1.4, 0.75, 1.20],
        wspace=0.12,
        left=0.030, right=0.993, top=0.93, bottom=0.13,
    )
    ax_world  = fig.add_subplot(gs[0, 0])
    ax_eu     = fig.add_subplot(gs[0, 1])
    ax_tr     = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis("off")

    def _draw_countries(ax):
        for feat in world["features"]:
            for poly in _polys(feat):
                for ring in poly:
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    ax.fill(xs, ys,
                            facecolor=P["country_face_c"],
                            edgecolor=P["country_edge_c"],
                            linewidth=P["country_edge_w"], zorder=1)

    def _city_centroid(city):
        pts = city_pts[city]
        clat = sum(p[0] for p in pts) / len(pts)
        clon = sum(p[1] for p in pts) / len(pts)
        return clat, clon

    def _plot_cities(ax, marker_size, font_size,
                     bounds=None, only=None):
        for city in ordered:
            clat, clon = _city_centroid(city)
            if bounds is not None:
                lon_lo, lon_hi, lat_lo, lat_hi = bounds
                if not (lon_lo <= clon <= lon_hi and
                        lat_lo <= clat <= lat_hi):
                    continue
            if only is not None and city not in only:
                continue
            face = city_color[city]
            ax.scatter([clon], [clat],
                       s=marker_size, c=[face], alpha=0.95,
                       edgecolors="white", linewidths=0.9, zorder=3)
            ax.text(clon, clat, str(idx_of[city]),
                    ha="center", va="center",
                    fontsize=font_size, fontweight="bold",
                    color="white", zorder=4)

    # ---- World panel ----------------------------------------------------
    _draw_countries(ax_world)
    _plot_cities(ax_world, marker_size=130, font_size=7.0)
    # Outline the two zoom boxes on the world map.
    for box, label in [(EU_BOX, "EU"), (TR_BOX, "TR")]:
        lon_lo, lon_hi, lat_lo, lat_hi = box
        ax_world.add_patch(mpatches.Rectangle(
            (lon_lo, lat_lo), lon_hi - lon_lo, lat_hi - lat_lo,
            fill=False, edgecolor="#d63b3b", linewidth=1.0,
            linestyle=(0, (4, 2)), zorder=2.5))
        ax_world.text(lon_hi, lat_hi, f" {label}",
                      fontsize=8, color="#d63b3b",
                      va="bottom", ha="left", zorder=2.5)
    ax_world.set_xlim(*XLIM); ax_world.set_ylim(*YLIM)
    ax_world.set_aspect("equal", adjustable="box")
    ax_world.set_xlabel("Longitude", fontsize=P["axis_label_fs"])
    ax_world.set_ylabel("Latitude",  fontsize=P["axis_label_fs"])
    ax_world.tick_params(axis="both", labelsize=P["tick_fs"])
    ax_world.set_title("World", fontsize=P["title_fs"], pad=4)

    # ---- Europe zoom ----------------------------------------------------
    _draw_countries(ax_eu)
    _plot_cities(ax_eu, marker_size=210, font_size=9.0, bounds=EU_BOX)
    ax_eu.set_xlim(EU_BOX[0], EU_BOX[1])
    ax_eu.set_ylim(EU_BOX[2], EU_BOX[3])
    ax_eu.set_aspect("equal", adjustable="box")
    ax_eu.set_title("Europe zoom", fontsize=P["title_fs"], pad=4)
    ax_eu.tick_params(axis="both", labelsize=P["tick_fs"])
    for s in ("top", "right"): ax_eu.spines[s].set_visible(False)

    # ---- Türkiye + E. Med zoom -----------------------------------------
    _draw_countries(ax_tr)
    _plot_cities(ax_tr, marker_size=210, font_size=9.0, bounds=TR_BOX)
    ax_tr.set_xlim(TR_BOX[0], TR_BOX[1])
    ax_tr.set_ylim(TR_BOX[2], TR_BOX[3])
    ax_tr.set_aspect("equal", adjustable="box")
    ax_tr.set_title("Türkiye + E. Med.", fontsize=P["title_fs"], pad=4)
    ax_tr.tick_params(axis="both", labelsize=P["tick_fs"])
    for s in ("top", "right"): ax_tr.spines[s].set_visible(False)

    legend_items = [
        mpatches.Patch(color=P["seen_color"],
                       label=f"Seen cities ({len(seen_cities)})"),
        mpatches.Patch(color=P["unseen_color"],
                       label=f"Unseen cities ({len(unseen_cities)})"),
    ]
    ax_world.legend(handles=legend_items, loc="lower left",
                    fontsize=P["legend_fs"], framealpha=0.92,
                    borderpad=0.4, handlelength=1.4)

    # ---- Right-side index: 2 columns of "n  CityName" rows -------------
    n = len(ordered)
    n_col = 2
    rows_per_col = (n + n_col - 1) // n_col
    col_xs = [0.00, 0.50]
    y_top, y_bot = 0.995, 0.005
    row_h = (y_top - y_bot) / rows_per_col
    for i, city in enumerate(ordered):
        col = i // rows_per_col
        row = i %  rows_per_col
        y = y_top - row * row_h - row_h * 0.5
        x = col_xs[col]
        role = city_role[city]
        face = city_color[city]
        ax_legend.scatter([x + 0.050], [y],
                          s=180, c=[face], alpha=0.95,
                          edgecolors="white", linewidths=0.9,
                          transform=ax_legend.transAxes, zorder=2,
                          clip_on=False)
        ax_legend.text(x + 0.050, y, str(idx_of[city]),
                       ha="center", va="center",
                       fontsize=8.5, fontweight="bold",
                       color="white",
                       transform=ax_legend.transAxes, zorder=3)
        weight = "bold" if role == "unseen" else "normal"
        ax_legend.text(x + 0.115, y, city,
                       ha="left", va="center",
                       fontsize=10.0, color="#222222", fontweight=weight,
                       transform=ax_legend.transAxes, zorder=3)

    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / f"{out_stem}.pdf"
    png_path = out_dir / f"{out_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=P["png_dpi"])
    plt.close(fig)
    print(f"wrote  {pdf_path}")
    print(f"wrote  {png_path}  (dpi={P['png_dpi']})")


def render(preset_name: str, out_stem: str | None = None) -> None:
    if preset_name not in PRESETS:
        raise SystemExit(f"unknown preset {preset_name!r}; "
                         f"choose from {list(PRESETS)}")
    P = PRESETS[preset_name]
    if out_stem is None:
        out_stem = ("fig11_geographic_scatter"
                    if preset_name == "original"
                    else f"fig11_geographic_scatter_{preset_name}")

    lats = [r[0] for r in LOCATIONS]
    lons = [r[1] for r in LOCATIONS]
    cols = [P["seen_color"] if r[3] == "seen" else P["unseen_color"]
            for r in LOCATIONS]

    city_xy: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for lat, lon, city, role in LOCATIONS:
        city_xy[city].append((lat, lon, role))

    world = _decode_world()

    fig, ax = plt.subplots(figsize=P["figsize"])

    for feat in world["features"]:
        for poly in _polys(feat):
            for ring in poly:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.fill(xs, ys,
                        facecolor=P["country_face_c"],
                        edgecolor=P["country_edge_c"],
                        linewidth=P["country_edge_w"], zorder=1)

    ax.scatter(lons, lats,
               s=P["marker_s"], c=cols, alpha=P["marker_alpha"],
               edgecolors=P["marker_edge_c"],
               linewidths=P["marker_edge_w"], zorder=3)

    for city, pts in city_xy.items():
        clat = sum(p[0] for p in pts) / len(pts)
        clon = sum(p[1] for p in pts) / len(pts)
        role = pts[0][2]
        txt_color = (P["seen_text_color"] if role == "seen"
                     else P["unseen_text_color"])
        label = CITY_LABEL_OVERRIDES.get(city, city)
        nudge = CITY_LABEL_NUDGE.get(city, P["label_offset"])
        ax.annotate(label, (clon, clat),
                    fontsize=P["label_fontsize"], color=txt_color,
                    fontweight="bold",
                    xytext=nudge, textcoords="offset points", zorder=4,
                    bbox=dict(boxstyle=f"round,pad={P['label_bbox_pad']}",
                              facecolor="white", edgecolor="none",
                              alpha=P["label_bbox_alpha"]))

    ax.set_xlabel("Longitude", fontsize=P["axis_label_fs"])
    ax.set_ylabel("Latitude",  fontsize=P["axis_label_fs"])
    ax.tick_params(axis="both", labelsize=P["tick_fs"])
    if P["title_show"]:
        n_uniq = len(LOCATIONS)
        ax.set_title(
            f"Geographic coverage — {n_uniq:,} unique locations across "
            f"40 cities  (blue = seen, orange = unseen)",
            fontsize=P["title_fs"])
    if XLIM != (None, None): ax.set_xlim(*XLIM)
    if YLIM != (None, None): ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    legend_items = [
        mpatches.Patch(color=P["seen_color"],   label="Seen cities (32)"),
        mpatches.Patch(color=P["unseen_color"], label="Unseen cities (8)"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=P["legend_fs"], framealpha=0.92)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / f"{out_stem}.pdf"
    png_path = out_dir / f"{out_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=P["png_dpi"])
    plt.close(fig)
    print(f"wrote  {pdf_path}")
    print(f"wrote  {png_path}  (dpi={P['png_dpi']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="compact",
                    choices=list(PRESETS),
                    help="which preset to render (default: compact)")
    ap.add_argument("--out", default=None,
                    help="output filename stem (no extension); default "
                         "depends on variant")
    args = ap.parse_args()
    if args.variant == "poster":
        render_poster(args.out)
    else:
        render(args.variant, args.out)
