#!/usr/bin/env python3
"""Build the EOLLM training-runs inventory workbook.

Two sheets:
  1. Summary       — one row per run: config, dataset, LoRA, accuracy headline.
  2. Detailed      — per-topic validation accuracy for every run, with
                     zero-shot (excluded-from-training) cells flagged.

All values traced from run artifacts on lab-ws (summary.txt, epoch verdicts,
adapter_config.json, SLURM .out logs). See TRAINING_RUNS_INVENTORY.md for
provenance and caveats.
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------- styling ----------
HDR_FILL   = PatternFill("solid", fgColor="1F3864")   # dark navy
HDR_FONT   = Font(bold=True, color="FFFFFF", size=11)
SUB_FILL   = PatternFill("solid", fgColor="2E5496")   # mid blue (group hdr)
SUB_FONT   = Font(bold=True, color="FFFFFF", size=10)
REG_FILL   = PatternFill("solid", fgColor="E2EFDA")   # light green  (regular runs)
ABL_FILL   = PatternFill("solid", fgColor="FCE4D6")   # light orange (ablations)
FLAG_FILL  = PatternFill("solid", fgColor="FFF2CC")   # gold (flagship)
ZS_FILL    = PatternFill("solid", fgColor="F2F2F2")   # grey (zero-shot cell)
ZS_FONT    = Font(italic=True, color="808080")
LEAK_FILL  = PatternFill("solid", fgColor="FFE6E6")   # pink (leaked topic)
BEST_FONT  = Font(bold=True, color="C00000")
CENTER     = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT       = Alignment(horizontal="left",   vertical="center", wrap_text=True)
thin = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

def style_header(ws, row, ncols, fill=HDR_FILL, font=HDR_FONT):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = fill; cell.font = font
        cell.alignment = CENTER; cell.border = BORDER

def box(ws, r1, r2, c1, c2):
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            ws.cell(row=r, column=c).border = BORDER

wb = Workbook()

# ============================================================ SHEET 1: SUMMARY
ws = wb.active
ws.title = "Summary"

title = ("EOLLM Training Runs — Summary   |   base=Qwen3.5-4B (not HF-pushed)   |   "
         "thinking: native=Yes, state=OFF (empty <think>)   |   scores = VALIDATION accuracy (greedy MCQ exact-match)   |   loss = TRAINING loss")
ws.merge_cells("A1:T1")
ws["A1"] = title
ws["A1"].font = Font(bold=True, size=11, color="1F3864")
ws["A1"].alignment = LEFT
ws.row_dimensions[1].height = 30

cols = ["Training ID", "Name", "Type", "Dataset\n(sample)", "Real/\nFlagship",
        "Trained\ntasks", "Val\nN", "Result\ntype", "Best val\nacc %",
        "Final val\nacc %", "Base\nacc %", "Train\nloss", "Max\nep",
        "Stopped\nep", "Early\nstopped?", "LoRA\nr", "LoRA\nα",
        "LoRA params\n(% of 4.58B)", "Wall clock\n(min)", "Peak VRAM"]
hdr_row = 3
for i, c in enumerate(cols, 1):
    ws.cell(row=hdr_row, column=i, value=c)
style_header(ws, hdr_row, len(cols))

# (is_regular, flagship, row data...)
rows = [
    # regular full runs
    (True, False, ["20260422_074420_rtx_pro_6000_96gb", "per-city vision (full)", "Regular", "per_city", "in-dist full", "all 14", 6984, "val acc", 78.3, 78.3, 18.0, "n/a", 10, 6, "Yes", 32, 32, "~77.5M (~1.69%)*", 1745, "—"]),
    (True, False, ["20260423_190939_rtx_pro_6000_96gb", "seen/unseen vision (full, long)", "Regular", "seen_unseen", "cross-city full", "all 14", 8831, "val acc", 72.5, 70.8, 19.0, 0.073, 10, 8, "Yes", 32, 32, "~77.5M (~1.69%)*", 2551, "83.7 GB (102%)"]),
    (True, False, ["textonly_baseline", "text-only baseline", "Regular", "per_city", "no-image ctrl", "all 14 (text only)", 6984, "val acc", 49.5, 49.0, 35.0, 0.141, 5, 3, "Yes", 16, 16, "38,756,352 (0.85%)", 130, "21.3 GB"]),
    # ablations
    (False, False, ["ablation_split1_per_city", "Urbanization Removed", "Ablation", "per_city", "in-dist", "7 (geo/vision)", 6984, "val acc", 66.3, 66.3, 18.0, 0.058, 5, 5, "No", 16, 16, "38,756,352 (0.85%)", 1014, "92.2 GB (113%)"]),
    (False, False, ["ablation_split2a_per_city", "Geo Removed (in-dist)", "Ablation", "per_city", "2b in-dist twin", "9 (urban)", 6984, "val acc", 59.5, 50.7, 18.0, 0.122, 5, 3, "Yes", 16, 16, "38,756,352 (0.85%)", 290, "59.3 GB (72%)"]),
    (False, True,  ["ablation_split2b_seen_unseen", "Geo Removed (cross-city)", "Ablation", "seen_unseen", "FLAGSHIP", "9 (urban)", 8831, "val acc", 61.8, 60.4, 19.0, 0.095, 5, 5, "No", 16, 16, "38,756,352 (0.85%)", 543, "59.5 GB (73%)"]),
    (False, False, ["ablation_split3_per_city", "Mismatch Removed", "Ablation", "per_city", "in-dist", "10", 6984, "val acc", 63.2, 63.2, 18.0, 0.103, 5, 5, "No", 16, 16, "38,756,352 (0.85%)", 1261, "90.8 GB (111%)"]),
    (False, False, ["ablation_split5_per_city", "Camera Direction Removed", "Ablation", "per_city", "in-dist", "13", 6984, "val acc", 76.5, 76.5, 18.0, 0.073, 5, 5, "No", 16, 16, "38,756,352 (0.85%)", 1608, "91.3 GB (111%)"]),
    (False, False, ["ablation_split6_per_city", "Mismatch Easy Only", "Ablation", "per_city", "in-dist", "12", 6984, "val acc", 76.4, 76.4, 18.0, 0.088, 5, 5, "No", 16, 16, "38,756,352 (0.85%)", 1495, "91.2 GB (111%)"]),
]

r = hdr_row + 1
for is_reg, flag, data in rows:
    for i, v in enumerate(data, 1):
        cell = ws.cell(row=r, column=i, value=v)
        cell.alignment = LEFT if i in (1, 2) else CENTER
        cell.border = BORDER
        fill = FLAG_FILL if flag else (REG_FILL if is_reg else ABL_FILL)
        cell.fill = fill
    ws.cell(row=r, column=9).font = BEST_FONT  # best val acc bold red
    r += 1

# notes
note_r = r + 1
notes = [
    "* r=32 LoRA param count derived by 2x scaling of logged r=16 count (38,756,352); not directly logged.",
    "Regular runs train on ALL 14 topics (not ablations). 20260422 = in-dist headline; 20260423 = cross-city headline; textonly = no-image control.",
    "Ablations remove topics from TRAINING ONLY; val set is always the full 14 topics -> excluded-topic scores are ZERO-SHOT transfer (see Detailed sheet, grey cells).",
    "Cross-set comparability: per_city val=6,984 (40 seen cities); seen_unseen val=8,831 (8 held-out cities). Only compare absolute % WITHIN the same dataset.",
    "LoRA rank: launch_ablations.sh comment says r=32 but ablation adapter_config.json = r=16 (ground truth). Ablations=r16; the two full vision runs=r32.",
    "'Best' vs 'Final': best = peak-epoch val acc; final = last-epoch val acc. They differ when a run kept training past its peak (e.g. 20260423, textonly, 2a).",
    "No separate benchmark exists — the validation split IS the benchmark.",
]
for n in notes:
    ws.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=20)
    c = ws.cell(row=note_r, column=1, value="• " + n)
    c.font = Font(size=9, italic=True, color="595959"); c.alignment = LEFT
    note_r += 1

widths = [34, 26, 10, 12, 14, 16, 7, 9, 10, 10, 9, 8, 6, 8, 9, 6, 6, 19, 11, 16]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.row_dimensions[hdr_row].height = 42
ws.freeze_panes = "C4"

# =========================================================== SHEET 2: DETAILED
ws2 = wb.create_sheet("Detailed Results")

ws2.merge_cells("A1:K1")
ws2["A1"] = ("EOLLM — Per-Topic VALIDATION Accuracy (best epoch).   "
             "GREY+italic = topic EXCLUDED from that run's training (zero-shot transfer).   "
             "PINK topic = metadata-leaked (near-ceiling even text-only).")
ws2["A1"].font = Font(bold=True, size=11, color="1F3864")
ws2["A1"].alignment = LEFT
ws2.row_dimensions[1].height = 30

run_cols = [
    ("20260422\nper_city all-14", "per_city", set()),
    ("20260423\nseen_unseen all-14", "seen_unseen", set()),
    ("textonly\nbaseline", "per_city", set()),
    ("abl1\nUrban removed", "per_city", {"amenity_richness","building_height","junction_type","land_use","road_type","transit_density","urban_density"}),
    ("abl2a\nGeo removed", "per_city", {"camera_direction","mismatch_binary_easy","mismatch_binary_hard","mismatch_mcq_easy","mismatch_mcq_hard"}),
    ("abl2b ⭐\nGeo rm x-city", "seen_unseen", {"camera_direction","mismatch_binary_easy","mismatch_binary_hard","mismatch_mcq_easy","mismatch_mcq_hard"}),
    ("abl3\nMismatch rm", "per_city", {"mismatch_binary_easy","mismatch_binary_hard","mismatch_mcq_easy","mismatch_mcq_hard"}),
    ("abl5\nCamDir rm", "per_city", {"camera_direction"}),
    ("abl6\nMismatch-easy only", "per_city", {"mismatch_binary_hard","mismatch_mcq_hard"}),
]

LEAKED = {"green_space", "road_surface"}
VISION = {"camera_direction","mismatch_binary_easy","mismatch_binary_hard","mismatch_mcq_easy","mismatch_mcq_hard"}

# topic -> [22, 23, textonly, abl1, abl2a, abl2b, abl3, abl5, abl6]
topics = {
    "amenity_richness":      [62.0,59.4,34.8,38.0,36.9,59.7,62.6,61.0,61.9],
    "building_height":       [69.0,54.4,40.6,52.1,43.7,54.0,66.3,65.1,68.2],
    "camera_direction":      [56.5,55.6,26.4,46.0,26.2,26.1,43.5,29.9,44.2],
    "green_space":           [100,63.8,100,100,100,100,100,100,100],
    "junction_type":         [77.0,71.4,58.0,60.7,52.0,69.6,76.6,78.3,75.9],
    "land_use":              [75.9,69.1,72.0,50.8,73.6,76.2,79.0,77.9,80.7],
    "mismatch_binary_easy":  [96.6,95.0,48.0,97.1,54.5,77.4,75.2,97.3,97.3],
    "mismatch_binary_hard":  [90.2,91.6,52.2,92.3,54.7,68.1,65.2,92.2,82.7],
    "mismatch_mcq_easy":     [99.1,98.5,25.7,98.0,26.6,37.3,31.4,98.4,97.5],
    "mismatch_mcq_hard":     [90.6,90.5,21.7,88.2,23.4,31.1,29.4,90.2,81.6],
    "road_surface":          [94.2,31.5,94.2,94.0,94.2,98.4,93.7,94.5,94.2],
    "road_type":             [71.3,70.9,66.1,55.3,67.6,72.1,72.7,74.0,73.6],
    "transit_density":       [50.8,36.9,28.0,31.0,36.7,44.9,49.0,48.1,51.0],
    "urban_density":         [73.8,70.6,55.1,41.7,54.2,72.9,73.3,76.3,74.3],
}
RAND = {"mismatch_binary_easy":50,"mismatch_binary_hard":50}  # else 25

hdr2 = 3
ws2.cell(row=hdr2, column=1, value="Topic")
ws2.cell(row=hdr2, column=2, value="Random\n%")
for j, (label, _, _) in enumerate(run_cols, start=3):
    ws2.cell(row=hdr2, column=j, value=label)
style_header(ws2, hdr2, 2 + len(run_cols))

rr = hdr2 + 1
topic_order = ["amenity_richness","building_height","camera_direction","green_space",
               "junction_type","land_use","mismatch_binary_easy","mismatch_binary_hard",
               "mismatch_mcq_easy","mismatch_mcq_hard","road_surface","road_type",
               "transit_density","urban_density"]
for t in topic_order:
    tc = ws2.cell(row=rr, column=1, value=t)
    tc.alignment = LEFT; tc.border = BORDER
    tc.font = Font(bold=True)
    if t in LEAKED: tc.fill = LEAK_FILL
    rc = ws2.cell(row=rr, column=2, value=RAND.get(t, 25)); rc.alignment = CENTER; rc.border = BORDER
    for j, ((label, ds, excl), val) in enumerate(zip(run_cols, topics[t]), start=3):
        cell = ws2.cell(row=rr, column=j, value=val)
        cell.alignment = CENTER; cell.border = BORDER
        if t in excl:                       # zero-shot (excluded from training)
            cell.fill = ZS_FILL; cell.font = ZS_FONT
    rr += 1

# overall row
ws2.cell(row=rr, column=1, value="OVERALL (full val set)").font = Font(bold=True, color="1F3864")
ws2.cell(row=rr, column=1).alignment = LEFT; ws2.cell(row=rr, column=1).border = BORDER
ws2.cell(row=rr, column=2, value="").border = BORDER
overall = [78.3,72.5,49.5,66.3,59.5,61.8,63.2,76.5,76.4]
for j, v in enumerate(overall, start=3):
    c = ws2.cell(row=rr, column=j, value=v)
    c.alignment = CENTER; c.border = BORDER; c.font = BEST_FONT
    c.fill = PatternFill("solid", fgColor="DDEBF7")
rr += 2

leg = [
    "Legend:  GREY italic cell = topic excluded from that run's TRAINING -> number is ZERO-SHOT transfer, not trained performance.",
    "         PINK topic name = metadata-leaked (solvable from text alone; near-ceiling everywhere — discount it).",
    "         OVERALL row = headline val accuracy over the FULL 14-topic val set (mixes trained + zero-shot topics for ablations — read per-topic for the real signal).",
    "         Columns abl2b vs abl2a: same topics excluded, but 2b is cross-city (seen_unseen) and 2a is in-distribution (per_city) — the pair isolates the generalization gap.",
    "         Vision-only topics (camera_direction, mismatch_*) are where images matter most; compare any vision run vs textonly to see the real vision lift.",
]
for n in leg:
    ws2.merge_cells(start_row=rr, start_column=1, end_row=rr, end_column=11)
    c = ws2.cell(row=rr, column=1, value=n)
    c.font = Font(size=9, italic=True, color="595959"); c.alignment = LEFT
    rr += 1

ws2.column_dimensions["A"].width = 22
ws2.column_dimensions["B"].width = 8
for j in range(3, 3 + len(run_cols)):
    ws2.column_dimensions[get_column_letter(j)].width = 13
ws2.row_dimensions[hdr2].height = 40
ws2.freeze_panes = "C4"

out = "/home/ezel/Development/EOLLM/thesis_artifacts/EOLLM_training_runs.xlsx"
wb.save(out)
print("WROTE", out)
