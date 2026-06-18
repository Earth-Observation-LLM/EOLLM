#!/usr/bin/env python3
"""Cross-strategy report — grounded entirely in results/<strategy>/ablation_log.jsonl.

Compares the prompting strategies (current / think16k / multistep) head-to-head on
the 9 attribute tasks, across the 4 views (full / sat_only / sv_only / blind).

Outputs (to --outdir, default results/):
  - strategy_compare.md   : human-readable tables + the headline verdict
  - strategy_compare.json : the same numbers, machine-readable

Headline question per strategy: does it lift `full` above max(sat_only, sv_only)
— i.e. make the SECOND view actually matter? (synergy > 0). And does any strategy
beat the `current` baseline overall / per topic?

No hardcoded numbers — everything is computed from the logs at run time.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATTR = ["land_use", "building_height", "urban_density", "junction_type",
        "green_space", "amenity_richness", "road_type", "road_surface",
        "transit_density"]
MODES = ["full", "sat_only", "sv_only", "blind"]
STRATEGIES = ["current", "think16k", "multistep"]


def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            rows.append(r)
    return rows


def acc(cm):
    c, n = cm
    return 100 * c / n if n else None


def aggregate(rows):
    """rows -> {(topic, mode): [correct, n]} and {mode: [correct, n]}."""
    tm = defaultdict(lambda: [0, 0])
    ov = defaultdict(lambda: [0, 0])
    for r in rows:
        t, m = r.get("topic"), r.get("mode")
        if t not in ATTR or m not in MODES:
            continue
        c = 1 if r.get("correct") else 0
        tm[(t, m)][0] += c
        tm[(t, m)][1] += 1
        ov[m][0] += c
        ov[m][1] += 1
    return tm, ov


def fmt(v):
    return f"{v:.1f}" if v is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"),
                    help="dir holding <strategy>/ablation_log.jsonl")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    args = ap.parse_args()

    results = Path(args.results)
    outdir = Path(args.outdir) if args.outdir else results
    outdir.mkdir(parents=True, exist_ok=True)
    strategies = args.strategies or [s for s in STRATEGIES
                                     if (results / s / "ablation_log.jsonl").exists()]
    if not strategies:
        raise SystemExit(f"no <strategy>/ablation_log.jsonl found under {results}")

    data = {}  # strategy -> (tm, ov, n_rows)
    for s in strategies:
        rows = load_log(results / s / "ablation_log.jsonl")
        tm, ov = aggregate(rows)
        data[s] = (tm, ov, len(rows))

    out = {"strategies": {}, "overall": {}}
    L = ["# Strategy comparison — Qwen3.5-9B-AWQ on the 9 attribute tasks", ""]
    L.append("_full = marked-sat + 4 SV angles · sat_only = sat · sv_only = SV grid · "
             "blind = no images · greedy (temp 0)_")
    L.append("")

    # ---- overall table: strategy x mode ----
    L.append("## Overall accuracy — strategy × view")
    L.append("")
    L.append("| strategy | full | sat_only | sv_only | blind | synergy (full−best1) | vision (full−blind) | n |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in strategies:
        tm, ov, nrows = data[s]
        a = {m: acc(ov[m]) for m in MODES}
        best1 = max([x for x in (a["sat_only"], a["sv_only"]) if x is not None], default=None)
        syn = (a["full"] - best1) if (a["full"] is not None and best1 is not None) else None
        vis = (a["full"] - a["blind"]) if (a["full"] is not None and a["blind"] is not None) else None
        ntot = max((ov[m][1] for m in MODES), default=0)
        L.append(f"| {s} | {fmt(a['full'])} | {fmt(a['sat_only'])} | {fmt(a['sv_only'])} | "
                 f"{fmt(a['blind'])} | {('%+.1f'%syn) if syn is not None else '—'} | "
                 f"{('%+.1f'%vis) if vis is not None else '—'} | {ntot} |")
        out["overall"][s] = {"acc": a, "synergy": syn, "vision_gain": vis,
                             "n_rows": nrows}
    L.append("")
    L.append("_synergy > 0 ⇒ the strategy makes the SECOND view add accuracy a single view can't._")
    L.append("")

    # ---- per-topic full-accuracy across strategies (does any strategy win?) ----
    L.append("## `full` accuracy by topic — across strategies")
    L.append("")
    L.append("| topic | " + " | ".join(strategies) + " |")
    L.append("|---|" + "---|" * len(strategies))
    for t in ATTR:
        cells = []
        for s in strategies:
            tm, _, _ = data[s]
            cells.append(fmt(acc(tm[(t, "full")])))
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- per-strategy full vs best-single per topic (synergy detail) ----
    for s in strategies:
        tm, ov, _ = data[s]
        L.append(f"## {s} — topic × view (synergy detail)")
        L.append("")
        L.append("| topic | full | sat_only | sv_only | blind | full−best1 |")
        L.append("|---|---|---|---|---|---|")
        topic_syn = {}
        for t in ATTR:
            a = {m: acc(tm[(t, m)]) for m in MODES}
            best1 = max([x for x in (a["sat_only"], a["sv_only"]) if x is not None], default=None)
            syn = (a["full"] - best1) if (a["full"] is not None and best1 is not None) else None
            topic_syn[t] = syn
            L.append(f"| {t} | {fmt(a['full'])} | {fmt(a['sat_only'])} | {fmt(a['sv_only'])} | "
                     f"{fmt(a['blind'])} | {('%+.1f'%syn) if syn is not None else '—'} |")
        out["strategies"][s] = {"topic_synergy": topic_syn}
        L.append("")

    # ---- verdict ----
    L.append("## Verdict")
    L.append("")
    base = "current" if "current" in data else strategies[0]
    base_full = acc(data[base][1]["full"])
    for s in strategies:
        a = {m: acc(data[s][1][m]) for m in MODES}
        best1 = max([x for x in (a["sat_only"], a["sv_only"]) if x is not None], default=None)
        syn = (a["full"] - best1) if (a["full"] is not None and best1 is not None) else None
        delta_base = (a["full"] - base_full) if (a["full"] is not None and base_full is not None) else None
        verdict = []
        if syn is not None:
            verdict.append(f"synergy {syn:+.1f} ({'second view HELPS' if syn > 1 else 'second view ~useless'})")
        if s != base and delta_base is not None:
            verdict.append(f"full vs {base} baseline: {delta_base:+.1f}")
        L.append(f"- **{s}**: full {fmt(a['full'])}%  ·  " + "  ·  ".join(verdict))
    L.append("")
    L.append("_Run `complementarity_gate.py results/<strategy>/ablation_log.jsonl --attr-only` "
             "for the formal PASS/FAIL synergy gate per strategy._")

    md = "\n".join(L)
    (outdir / "strategy_compare.md").write_text(md)
    (outdir / "strategy_compare.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {outdir}/strategy_compare.md and .json")
    print(f"strategies: {strategies}")
    for s in strategies:
        print(f"  {s}: full={fmt(acc(data[s][1]['full']))}  n_rows={data[s][2]}")


if __name__ == "__main__":
    main()
