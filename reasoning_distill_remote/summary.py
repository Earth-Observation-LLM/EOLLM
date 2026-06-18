#!/usr/bin/env python3
"""Structured summary builder for the ablation runs.

Single source of truth for turning a list of result rows into the rich,
analysis-ready summary dict (per-topic x mode accuracy, think/truncation stats,
throughput, visual-dependency buckets). Used both live (during a run) and
offline (analyze.py), so the live view and the final report are identical.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

MODES = ["full", "sat_only", "sv_only", "blind"]


def _pctl(xs, q):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def build_summary(rows, meta=None):
    """rows: list of logged dicts (may include {'error':...} rows).
    Returns a JSON-serializable summary dict."""
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]

    # ---- per (source, topic, mode) accuracy ----
    cell = defaultdict(lambda: {"correct": 0, "total": 0})
    # think-token usage per mode
    think_by_mode = defaultdict(list)
    trunc_by_mode = defaultdict(int)
    n_by_mode = defaultdict(int)
    overflow = 0
    retried = 0
    unparsed = 0
    latencies = []
    completion_tokens = []

    for r in ok:
        key = f"{r['source']}|{r['topic']}|{r['mode']}"
        cell[key]["total"] += 1
        cell[key]["correct"] += int(r.get("correct", False))
        m = r["mode"]
        n_by_mode[m] += 1
        # think tokens ~ completion tokens of phase 1; use usage if present else chars/4
        u = r.get("usage") or {}
        ctok = u.get("completion_tokens")
        if isinstance(ctok, (int, float)):
            completion_tokens.append(ctok)
        think_by_mode[m].append(len(r.get("reasoning", "") or "") // 4)
        if r.get("think_truncated"):
            trunc_by_mode[m] += 1
        if r.get("ctx_overflow"):
            overflow += 1
        if r.get("parse_attempts", 1) > 1:
            retried += 1
        if not r.get("letter"):
            unparsed += 1
        if isinstance(r.get("latency_s"), (int, float)):
            latencies.append(r["latency_s"])

    # ---- per-topic x mode accuracy table (aggregated over sources too) ----
    topic_mode = defaultdict(lambda: {"correct": 0, "total": 0})
    src_topic_mode = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in ok:
        topic_mode[(r["topic"], r["mode"])]["correct"] += int(r.get("correct", False))
        topic_mode[(r["topic"], r["mode"])]["total"] += 1
        src_topic_mode[(r["source"], r["topic"], r["mode"])]["correct"] += int(r.get("correct", False))
        src_topic_mode[(r["source"], r["topic"], r["mode"])]["total"] += 1

    topics = sorted({t for (t, _m) in topic_mode})
    accuracy_table = {}
    for t in topics:
        accuracy_table[t] = {}
        for m in MODES:
            c = topic_mode.get((t, m))
            if c and c["total"]:
                accuracy_table[t][m] = {
                    "acc": round(100 * c["correct"] / c["total"], 1),
                    "n": c["total"],
                }

    # ---- per-source breakdown of the same table ----
    by_source = {}
    sources = sorted({s for (s, _t, _m) in src_topic_mode})
    for s in sources:
        by_source[s] = {}
        s_topics = sorted({t for (ss, t, _m) in src_topic_mode if ss == s})
        for t in s_topics:
            by_source[s][t] = {}
            for m in MODES:
                c = src_topic_mode.get((s, t, m))
                if c and c["total"]:
                    by_source[s][t][m] = {
                        "acc": round(100 * c["correct"] / c["total"], 1),
                        "n": c["total"],
                    }

    # ---- visual-dependency buckets (per topic): pair full vs blind, sat vs sv ----
    # group rows by (source, question_id) and look across that record's modes
    rec_modes = defaultdict(dict)  # (src,qid) -> mode -> correct(bool)
    rec_topic = {}
    for r in ok:
        qid = r.get("question_id")
        if qid is None:
            continue
        rec_modes[(r.get("source"), qid)][r["mode"]] = bool(r.get("correct"))
        rec_topic[(r.get("source"), qid)] = r["topic"]
    buckets = defaultdict(lambda: defaultdict(int))  # topic -> bucket -> n
    for k, modes in rec_modes.items():
        t = rec_topic[k]
        if "full" in modes and "blind" in modes:
            f, b = modes["full"], modes["blind"]
            bk = ("visual" if f and not b else "leaked" if f and b
                  else "regress" if not f and b else "hard")
            buckets[t][f"fb_{bk}"] += 1
        if "sat_only" in modes and "sv_only" in modes:
            so, sv = modes["sat_only"], modes["sv_only"]
            if so and not sv:
                buckets[t]["sat_solves"] += 1
            elif sv and not so:
                buckets[t]["sv_solves"] += 1
            elif so and sv:
                buckets[t]["both_solve"] += 1
            else:
                buckets[t]["neither_solves"] += 1

    # ---- think / throughput stats ----
    think_stats = {}
    for m in MODES:
        tk = think_by_mode.get(m, [])
        if tk:
            think_stats[m] = {
                "n": n_by_mode[m],
                "median_think_tok": int(statistics.median(tk)),
                "p90_think_tok": int(_pctl(tk, 0.9)),
                "max_think_tok": max(tk),
                "truncated": trunc_by_mode[m],
                "truncated_pct": round(100 * trunc_by_mode[m] / n_by_mode[m], 1),
            }

    overall_correct = sum(c["correct"] for c in cell.values())
    overall_total = sum(c["total"] for c in cell.values())

    summary = {
        "meta": meta or {},
        "progress": {
            "completed": len(ok),
            "errors": len(errs),
            "ctx_overflow": overflow,
            "parse_retried": retried,
            "unparsed_after_retries": unparsed,
            "overall_acc": round(100 * overall_correct / overall_total, 2) if overall_total else None,
            "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
            "median_completion_tok": int(statistics.median(completion_tokens)) if completion_tokens else None,
        },
        "accuracy_by_topic_mode": accuracy_table,
        "accuracy_by_source": by_source,
        "think_stats": think_stats,
        "visual_dependency": {t: dict(b) for t, b in buckets.items()},
    }
    return summary


def render_markdown(summary):
    """Human-readable snapshot of the summary dict."""
    L = []
    meta = summary.get("meta", {})
    p = summary["progress"]
    L.append(f"# Ablation summary — {meta.get('model', '?')}")
    if meta.get("started"):
        L.append(f"_started {meta['started']}  •  updated {meta.get('updated','?')}_")
    L.append("")
    L.append(f"**Progress:** {p['completed']} done"
             + (f" / {meta['total']}" if meta.get("total") else "")
             + f"  •  errors {p['errors']}  •  ctx-overflow {p['ctx_overflow']}"
             + f"  •  parse-retried {p['parse_retried']}  •  unparsed {p['unparsed_after_retries']}")
    if p.get("overall_acc") is not None:
        L.append(f"**Overall acc:** {p['overall_acc']}%  •  "
                 f"median latency {p['median_latency_s']}s  •  "
                 f"median completion {p['median_completion_tok']} tok")
    if meta.get("gens_per_min"):
        L.append(f"**Throughput:** {meta['gens_per_min']} gen/min"
                 + (f"  •  ETA {meta['eta']}" if meta.get("eta") else ""))
    L.append("")

    # accuracy table
    L.append("## Accuracy — topic × mode")
    L.append("")
    L.append("| topic | full | sat_only | sv_only | blind | n |")
    L.append("|---|---|---|---|---|---|")
    for t in sorted(summary["accuracy_by_topic_mode"]):
        row = summary["accuracy_by_topic_mode"][t]
        def cell(m):
            return f"{row[m]['acc']}%" if m in row else "–"
        n = max((row[m]["n"] for m in row), default=0)
        L.append(f"| {t} | {cell('full')} | {cell('sat_only')} | "
                 f"{cell('sv_only')} | {cell('blind')} | {n} |")
    L.append("")

    # think stats
    L.append("## Thinking / truncation")
    L.append("")
    L.append("| mode | n | median tok | p90 tok | max tok | truncated |")
    L.append("|---|---|---|---|---|---|")
    for m in MODES:
        if m in summary["think_stats"]:
            s = summary["think_stats"][m]
            L.append(f"| {m} | {s['n']} | {s['median_think_tok']} | "
                     f"{s['p90_think_tok']} | {s['max_think_tok']} | "
                     f"{s['truncated']} ({s['truncated_pct']}%) |")
    L.append("")

    # visual dependency
    L.append("## Visual-dependency buckets (full vs blind; sat vs sv)")
    L.append("")
    L.append("| topic | visual | leaked | hard | regress | sat-solves | sv-solves | both | neither |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in sorted(summary["visual_dependency"]):
        b = summary["visual_dependency"][t]
        L.append(f"| {t} | {b.get('fb_visual',0)} | {b.get('fb_leaked',0)} | "
                 f"{b.get('fb_hard',0)} | {b.get('fb_regress',0)} | "
                 f"{b.get('sat_solves','–')} | {b.get('sv_solves','–')} | "
                 f"{b.get('both_solve','–')} | {b.get('neither_solves','–')} |")
    L.append("")
    L.append("_visual=full✓blind✗ (needs vision) • leaked=full✓blind✓ (text-solvable)_")
    return "\n".join(L)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows
