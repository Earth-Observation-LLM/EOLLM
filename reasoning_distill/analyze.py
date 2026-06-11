#!/usr/bin/env python3
"""Per-topic image-ablation analysis over ablation_log.jsonl.

Reads the log produced by run_ablation.py and prints, per (source, topic):
  full%   blind%   gap   visual/leaked/hard/regress buckets   truncated

gap = full% - blind% = how much the images actually help = visual dependency.
High blind% (≈ full%) => topic is solvable without vision => structural leak.

This is the publishable "teacher benchmarked on our dataset" table. Safe to run
mid-run on a partial log.

  python analyze.py
  python analyze.py --source benchmark
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="reasoning_distill/ablation_log.jsonl")
    ap.add_argument("--source", default=None, help="filter to one source")
    args = ap.parse_args()

    # key -> {full: letter/correct, blind: letter/correct}
    rec = defaultdict(dict)
    errors = trunc = 0
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                errors += 1
                continue
            if args.source and r["source"] != args.source:
                continue
            rec[(r["source"], r["question_id"], r["topic"])][r["pass"]] = r["correct"]
            if r.get("think_truncated"):
                trunc += 1

    # aggregate per (source, topic) over records with BOTH passes
    agg = defaultdict(lambda: defaultdict(int))
    for (source, qid, topic), passes in rec.items():
        if "full" not in passes or "blind" not in passes:
            continue
        fc, bc = passes["full"], passes["blind"]
        a = agg[(source, topic)]
        a["n"] += 1
        a["full"] += fc
        a["blind"] += bc
        bucket = ("visual" if fc and not bc else "leaked" if fc and bc
                  else "regress" if not fc and bc else "hard")
        a[bucket] += 1

    print(f"{'source':11s} {'topic':22s} {'n':>5} {'full%':>6} {'blind%':>7} "
          f"{'gap':>5} {'vis':>5} {'leak':>5} {'hard':>5} {'regr':>5}")
    print("-" * 92)
    for (source, topic) in sorted(agg):
        a = agg[(source, topic)]
        n = a["n"] or 1
        fp, bp = 100 * a["full"] / n, 100 * a["blind"] / n
        print(f"{source:11s} {topic:22s} {a['n']:>5} {fp:>5.0f}% {bp:>6.0f}% "
              f"{fp-bp:>+4.0f}% {a['visual']:>5} {a['leaked']:>5} "
              f"{a['hard']:>5} {a['regress']:>5}")

    print(f"\ncomplete record-pairs: {sum(a['n'] for a in agg.values())} | "
          f"truncated-think passes: {trunc} | errors: {errors}")
    print("gap>0 = vision helps; gap≈0 with high blind% = structural leak.")


if __name__ == "__main__":
    main()
