#!/usr/bin/env python3
"""Analyze ablation results — single model or cross-model comparison.

Reads results/<label>/ablation_log.jsonl (the new per-model layout). Prints the
per-topic x mode accuracy table, think/truncation stats, and visual-dependency
buckets — the same structured view written live to summary.md, but on demand and
with cross-model compare.

  python analyze.py                       # latest run under results/
  python analyze.py --label qwen3.5-27b_20260612_080000
  python analyze.py --log path/to/ablation_log.jsonl
  python analyze.py --compare             # all models side by side, per topic/mode
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary as summ

RESULTS = Path(__file__).resolve().parent / "results"
MODES = ["full", "sat_only", "sv_only", "blind"]


def latest_label():
    if not RESULTS.exists():
        return None
    dirs = [d for d in RESULTS.iterdir() if (d / "ablation_log.jsonl").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime).name


def single(log_path, source=None):
    rows = summ.load_rows(log_path)
    if source:
        rows = [r for r in rows if r.get("source") == source]
    s = summ.build_summary(rows)
    print(summ.render_markdown(s))


def compare(source=None):
    """One row per topic, columns = each model's full/blind (and the gap)."""
    labels = sorted(d.name for d in RESULTS.iterdir()
                    if (d / "ablation_log.jsonl").exists())
    if not labels:
        print("no results found under", RESULTS)
        return
    per_model = {}  # label -> accuracy_by_topic_mode
    for lab in labels:
        rows = summ.load_rows(RESULTS / lab / "ablation_log.jsonl")
        if source:
            rows = [r for r in rows if r.get("source") == source]
        per_model[lab] = summ.build_summary(rows)["accuracy_by_topic_mode"]

    topics = sorted({t for m in per_model.values() for t in m})
    print(f"# Cross-model comparison  (source={source or 'all'})\n")
    # full accuracy then blind accuracy per model
    for metric in ("full", "blind"):
        print(f"## {metric} accuracy")
        hdr = "| topic | " + " | ".join(labels) + " |"
        print(hdr)
        print("|" + "---|" * (len(labels) + 1))
        for t in topics:
            cells = []
            for lab in labels:
                cell = per_model[lab].get(t, {}).get(metric)
                cells.append(f"{cell['acc']}%" if cell else "–")
            print(f"| {t} | " + " | ".join(cells) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="results/<label>/ to analyze")
    ap.add_argument("--log", default=None, help="explicit ablation_log.jsonl path")
    ap.add_argument("--source", default=None,
                    help="filter to one source (train/validation/benchmark)")
    ap.add_argument("--compare", action="store_true",
                    help="compare all models under results/ side by side")
    args = ap.parse_args()

    if args.compare:
        compare(args.source)
        return

    if args.log:
        log = args.log
    else:
        label = args.label or latest_label()
        if not label:
            print("no results found. run run_ablation.py first.")
            return
        log = str(RESULTS / label / "ablation_log.jsonl")
        print(f"# analyzing: {label}\n")
    single(log, args.source)


if __name__ == "__main__":
    main()
