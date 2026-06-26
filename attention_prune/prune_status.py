#!/usr/bin/env python3
"""prune_status.py — glance at the Phase-2 (prune.py) benchmark's progress.

Phase 2 writes prune/prune_progress.json every 10 passes: how many (question,arm,k)
greedy passes are done, the rate/ETA, and a running tally of survivors per arm·k.
This prints that, and — because the running tally needs a denominator to mean
anything — derives a live survival RATE per arm from prune_results.jsonl when it
can fetch it, plus the headline signal: the top-vs-random GAP.

Run with no args to scp from lab-ws; --local to read an already-pulled copy.

Usage:
    python prune_status.py                 # scp from lab-ws and print
    python prune_status.py --watch         # refresh every 30s
    python prune_status.py --local solved/qwen36_27b/prune
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_HOST_DEFAULT = "lab-ws"
REMOTE_BASE_DEFAULT = "/home/ain480/evaluation/attention_prune"

SAFE_TOPICS = {"amenity_richness", "transit_density"}


def _scp(host, remote, dest: Path) -> bool:
    try:
        subprocess.run(["scp", "-q", f"{host}:{remote}", str(dest)],
                       check=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def fetch(host, base, model_key, tmpdir: Path):
    """Pull progress.json (always) and results.jsonl (best-effort) into tmpdir."""
    rdir = f"{base}/solved/{model_key}/prune"
    prog = tmpdir / "prune_progress.json"
    res = tmpdir / "prune_results.jsonl"
    ok_prog = _scp(host, f"{rdir}/prune_progress.json", prog)
    _scp(host, f"{rdir}/prune_results.jsonl", res)   # may be large; ignore failure
    return (prog if ok_prog else None), (res if res.exists() else None)


def _survival_from_results(res: Path):
    """Per-arm survival rate (correct/total) and the SAFE-pool top-vs-random gap,
    computed from the partial results file. Returns (per_arm, headline) or None."""
    if not res or not res.exists():
        return None
    tot = defaultdict(int)        # arm -> n
    ok = defaultdict(int)         # arm -> n_correct
    safe = defaultdict(lambda: [0, 0])  # arm -> [correct, n] over SAFE topics
    n = 0
    for line in open(res):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue            # tolerate a half-written trailing line
        n += 1
        arm = r.get("arm", "?")
        tot[arm] += 1
        ok[arm] += int(r.get("correct", False))
        if r.get("topic") in SAFE_TOPICS:
            c = safe[arm]
            c[1] += 1
            c[0] += int(r.get("correct", False))
    per_arm = {a: (ok[a], tot[a], 100 * ok[a] / tot[a] if tot[a] else 0.0)
               for a in sorted(tot)}
    headline = None
    st, sr = safe.get("top"), safe.get("random")
    if st and sr and st[1] and sr[1]:
        tp = 100 * st[0] / st[1]
        rp = 100 * sr[0] / sr[1]
        headline = {"top_pct": tp, "random_pct": rp, "gap_pp": tp - rp,
                    "n_top": st[1], "n_random": sr[1]}
    return {"rows": n, "per_arm": per_arm, "headline": headline}


def show(prog: Path | None, res: Path | None):
    if prog is None or not prog.exists():
        print("(no prune_progress.json yet — job not started, or check host/base)")
        return
    d = json.loads(prog.read_text())
    pct = d.get("pct", 0)
    bar_n = int(pct / 5)
    bar = "#" * bar_n + "-" * (20 - bar_n)
    print(f"\n  PRUNE  [{bar}] {pct:5.1f}%  {d.get('done',0)}/{d.get('total',0)} passes")
    print(f"  {d.get('rate_s_per_pass','?')}s/pass  | ETA {d.get('eta_min','?')} min")

    sv = _survival_from_results(res)
    if sv and sv["per_arm"]:
        print(f"\n  survival rate by arm (over {sv['rows']} passes done):")
        # order arms meaningfully
        for a in [x for x in ["top", "fwd", "random", "bottom", "zero"]
                  if x in sv["per_arm"]]:
            c, n, p = sv["per_arm"][a]
            print(f"    {a:7s} {p:5.1f}%  ({c}/{n})")
        h = sv["headline"]
        if h:
            print(f"\n  SAFE-pool (amenity+transit) headline so far:")
            print(f"    top-1 {h['top_pct']:.1f}%  | random-1 {h['random_pct']:.1f}%  "
                  f"| GAP {h['gap_pp']:+.1f}pp   (n_top={h['n_top']})")
        else:
            print("\n  (safe-topic gap not computable yet — need top & random passes "
                  "on amenity_richness/transit_density)")
    else:
        # fall back to the raw running tally if we couldn't pull results
        tally = d.get("survivors_by_arm_k")
        if tally:
            print(f"  survivors (raw count, no denominator): {tally}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=REMOTE_HOST_DEFAULT)
    ap.add_argument("--base", default=REMOTE_BASE_DEFAULT)
    ap.add_argument("--model", default="qwen36_27b")
    ap.add_argument("--local", default=None,
                    help="read a local prune/ dir instead of scp")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    tmp = HERE / ".prune_status_tmp"
    tmp.mkdir(exist_ok=True)

    def once():
        if args.local:
            d = Path(args.local)
            show(d / "prune_progress.json", d / "prune_results.jsonl")
        else:
            prog, res = fetch(args.host, args.base, args.model, tmp)
            show(prog, res)

    if args.watch:
        try:
            while True:
                once()
                time.sleep(30)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        once()


if __name__ == "__main__":
    main()
