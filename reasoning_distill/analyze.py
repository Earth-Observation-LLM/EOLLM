#!/usr/bin/env python3
"""Per-topic, per-mode ablation analysis over ablation_log.jsonl.

Modes: full (all imgs) / sat_only / sv_only / blind (no imgs).
Not every topic has every mode (satellite_marked topics are full/blind only;
camera_direction has no sv_only). Missing modes print as '-'.

Reads what each (source, topic, mode) scored and reports accuracy side by side so
you can see which perspective carries the signal:
  full≈blind                  -> text/structural leak (vision doesn't help)
  sat_only≈full, sv_only low  -> satellite alone solves it
  sv_only≈full, sat_only low  -> street-view alone solves it
  full > both single-view     -> genuinely needs BOTH perspectives

Safe to run mid-run on a partial log.
  python analyze.py
  python analyze.py --source benchmark
"""
import argparse
import json
from collections import defaultdict

MODES = ["full", "sat_only", "sv_only", "blind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="reasoning_distill/ablation_log.jsonl")
    ap.add_argument("--source", default=None, help="filter to one source")
    args = ap.parse_args()

    # (source, topic, mode) -> [correct, total]
    agg = defaultdict(lambda: [0, 0])
    trunc = errors = overflow = 0
    think_chars = []
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                errors += 1
                continue
            if args.source and r.get("source") != args.source:
                continue
            key = (r["source"], r["topic"], r["mode"])
            agg[key][1] += 1
            agg[key][0] += int(r["correct"])
            if r.get("think_truncated"):
                trunc += 1
            if r.get("ctx_overflow"):
                overflow += 1
            think_chars.append(len(r.get("reasoning", "") or ""))

    # collect topics per source
    rows = defaultdict(dict)  # (source, topic) -> mode -> (corr,tot)
    for (source, topic, mode), (c, t) in agg.items():
        rows[(source, topic)][mode] = (c, t)

    hdr = f"{'source':11s} {'topic':22s} {'n':>5}"
    for m in MODES:
        hdr += f" {m:>9}"
    print(hdr)
    print("-" * len(hdr))
    for (source, topic) in sorted(rows):
        modes = rows[(source, topic)]
        n = max((t for (_, t) in modes.values()), default=0)
        line = f"{source:11s} {topic:22s} {n:>5}"
        for m in MODES:
            if m in modes:
                c, t = modes[m]
                line += f" {100*c/(t or 1):>8.0f}%"
            else:
                line += f" {'-':>9}"
        print(line)

    import statistics
    tc = sorted(think_chars) or [0]
    med = statistics.median(tc)
    p95 = tc[min(len(tc) - 1, int(0.95 * len(tc)))]
    print(f"\nthink chars: median {int(med)} p95 {int(p95)} max {max(tc)}  "
          f"(~chars/4 ≈ tokens)")
    print(f"forced-close(truncated): {trunc} | ctx-overflow: {overflow} | errors: {errors}")
    print("Read: full≈blind=leak | one single-view≈full=that perspective solves it"
          " | full>both=needs both.")


if __name__ == "__main__":
    main()
