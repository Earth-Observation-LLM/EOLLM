#!/usr/bin/env python3
"""blind_baseline.py — per-topic zero-pixel class-prior baseline (leakage guard).

A reviewer's first attack on this branch: if a topic is answerable from the TEXT
and OPTIONS alone (a skewed answer prior), then attention over the images is
meaningless for it and any "pruning survives" result on that topic is trivial. So
before trusting a survival curve we must know, per topic, how high a model that
sees NO images can score.

This needs no model: the strongest zero-pixel predictor for an MCQ benchmark with
a fixed option set is "always answer the modal correct option (by its TEXT, since
letter assignment is shuffled)". We compute that per topic and flag leaky ones.

Topics whose blind baseline is high cannot carry the headline claim. The Phase-2
analysis must (a) report this baseline next to every survival curve, and (b)
exclude or clearly flag topics above a threshold (default 0.45 — meaningfully
above 4-way chance 0.25 / 2-way 0.5).

Usage:
    python blind_baseline.py
    python blind_baseline.py --threshold 0.4 --data <path.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATA = REPO / "dataset_content/EODATA_compressed_final/benchmark/benchmark_with_answers.jsonl"

URBAN = ["amenity_richness", "building_height", "junction_type", "land_use",
         "road_surface", "road_type", "transit_density", "urban_density"]


def gold_text(rec):
    """The TEXT of the correct option (letter assignment is shuffled, so the prior
    lives on the option value, not the letter)."""
    ans = rec.get("answer")
    letter = None
    if isinstance(ans, str):
        a = ans.strip().upper()
        letter = a[0] if a and a[0] in "ABCD" else None
    opts = rec.get("options") or {}
    if letter and letter in opts:
        return opts[letter]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    by_topic = collections.defaultdict(list)
    for r in recs:
        if r.get("topic") in URBAN:
            by_topic[r["topic"]].append(r)

    print(f"{'topic':20s} {'n':>5s} {'#opt':>5s} {'chance':>7s} "
          f"{'blind_acc':>10s} {'modal_option'}")
    print("-" * 92)
    rows = []
    for topic in URBAN:
        rs = by_topic.get(topic, [])
        if not rs:
            continue
        n = len(rs)
        counts = collections.Counter()
        n_opts = collections.Counter()
        for r in rs:
            gt = gold_text(r)
            if gt is not None:
                counts[gt] += 1
            n_opts[len(r.get("options") or {})] += 1
        modal_opt, modal_n = counts.most_common(1)[0] if counts else ("?", 0)
        blind = modal_n / n if n else 0.0
        avg_opts = max(n_opts, key=n_opts.get) if n_opts else 4
        chance = 1.0 / avg_opts
        leaky = blind >= args.threshold
        flag = "  <== LEAKY (exclude/flag from headline)" if leaky else ""
        rows.append((topic, blind, leaky))
        print(f"{topic:20s} {n:5d} {avg_opts:5d} {chance:7.3f} "
              f"{blind:10.3f} {str(modal_opt)[:34]!s}{flag}")

    leaky = [t for t, b, lk in rows if lk]
    clean = [t for t, b, lk in rows if not lk]
    print("\nHEADLINE-SAFE topics (blind < %.2f): %s" % (args.threshold, clean or "(none!)"))
    print("LEAKY topics (report baseline beside survival, do NOT pool into headline): %s"
          % (leaky or "(none)"))


if __name__ == "__main__":
    main()
