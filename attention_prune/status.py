#!/usr/bin/env python3
"""status.py — glance at the remote collector's progress without tailing logs.

Reads the heartbeat progress.json that solve.py writes every few questions. Run
with no args to scp it from lab-ws; pass --local to read an already-pulled copy.

Usage:
    python status.py                          # scp from lab-ws and print
    python status.py --watch                  # refresh every 30s
    python status.py --local solved/qwen36_27b/progress.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_HOST_DEFAULT = "lab-ws"
REMOTE_BASE_DEFAULT = "/home/ain480/evaluation/attention_prune"


def fetch_remote(host, base, model_key, dest: Path):
    remote = f"{host}:{base}/solved/{model_key}/progress.json"
    subprocess.run(["scp", "-q", remote, str(dest)], check=True)
    return dest


def show(p: Path):
    if not p.exists():
        print(f"(no progress.json yet at {p})")
        return
    d = json.loads(p.read_text())
    bar_n = int(d.get("pct", 0) / 5)
    bar = "#" * bar_n + "-" * (20 - bar_n)
    print(f"\n  {d.get('model','?')}   [{bar}] {d.get('pct',0):5.1f}%  "
          f"{d.get('done',0)}/{d.get('total',0)}")
    print(f"  solved {d.get('solved',0)} ({d.get('solve_rate_pct',0)}%)  "
          f"| {d.get('rate_s_per_q','?')}s/q  | ETA {d.get('eta_min','?')} min")
    print(f"  statuses: {d.get('status_counts', {})}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=REMOTE_HOST_DEFAULT)
    ap.add_argument("--base", default=REMOTE_BASE_DEFAULT)
    ap.add_argument("--model", default="qwen36_27b")
    ap.add_argument("--local", default=None, help="read a local progress.json instead of scp")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    tmp = HERE / ".progress_remote.json"

    def once():
        if args.local:
            show(Path(args.local))
        else:
            try:
                fetch_remote(args.host, args.base, args.model, tmp)
                show(tmp)
            except subprocess.CalledProcessError:
                print("(could not scp progress.json — job not started yet, or "
                      f"check host/base: {args.host}:{args.base})")

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
