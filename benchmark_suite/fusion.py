#!/usr/bin/env python3
"""fusion.py — accuracy-side view fusion analysis (companion to ATTENTION WATCH).

Attention shows where the model LOOKED; this shows what each view CONTRIBUTES to
the answer — the causal side. For every question evaluated under all of
full/sat_only/sv_only/blind it computes, per topic and overall:

  acc_full / acc_sat / acc_sv / acc_blind  — raw accuracies
  vision_gain = acc_full - acc_blind        — how much vision helps at all
  synergy     = acc_full - max(sat, sv)     — benefit of having BOTH views
  fusion_win  = P(full right AND both singles wrong)   — only-fusion-solves
  fusion_hurt = P(full wrong AND some single right)    — 2nd view distracts
  redundancy  = P(sat and sv predict the SAME letter)  — signal overlap
  arbitrate   = P(full right | sat,sv disagree)        — conflict resolution

Reads a model's results dir (the per-mode *_predictions.jsonl the suite writes)
and prints a table + writes fusion.json. Adapted from
reasoning_distill/complementarity_gate.py to the suite's record format.

Usage:
    python fusion.py --results results/qwen35_4b_pc_lora
    python fusion.py --results results/qwen35_4b_pc_lora --by-topic
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODES = ["full", "sat_only", "sv_only", "blind"]


def load_by_question(results_dir: Path):
    """{question_id: {mode: (correct, prediction, topic)}} from per-mode preds."""
    byq = defaultdict(dict)
    topic_of = {}
    for mode in MODES:
        p = results_dir / f"{mode}_predictions.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["question_id"]
            byq[qid][mode] = (bool(r["is_correct"]), r.get("prediction"))
            topic_of[qid] = r.get("topic")
    return byq, topic_of


def metrics_for(qids, byq):
    acc = {m: [0, 0] for m in MODES}
    fusion_win = fusion_hurt = n = 0
    agree = agree_total = 0
    arb_right = arb_total = 0
    for q in qids:
        d = byq[q]
        if not all(m in d for m in MODES):
            continue  # only questions evaluated under ALL four modes
        n += 1
        for m in MODES:
            acc[m][1] += 1
            acc[m][0] += int(d[m][0])
        sat_ok, sat_p = d["sat_only"]
        sv_ok, sv_p = d["sv_only"]
        full_ok = d["full"][0]
        either = sat_ok or sv_ok
        if full_ok and not either:
            fusion_win += 1
        if (not full_ok) and either:
            fusion_hurt += 1
        if sat_p is not None and sv_p is not None:
            agree_total += 1
            if sat_p == sv_p:
                agree += 1
            else:
                arb_total += 1
                if full_ok:
                    arb_right += 1
    if n == 0:
        return None

    def a(m):
        return acc[m][0] / acc[m][1] * 100 if acc[m][1] else float("nan")

    full, sat, sv, blind = a("full"), a("sat_only"), a("sv_only"), a("blind")
    return {
        "n": n,
        "acc_full": round(full, 2), "acc_sat": round(sat, 2),
        "acc_sv": round(sv, 2), "acc_blind": round(blind, 2),
        "vision_gain": round(full - blind, 2),
        "synergy": round(full - max(sat, sv), 2),
        "fusion_win": round(fusion_win / n * 100, 2),
        "fusion_hurt": round(fusion_hurt / n * 100, 2),
        "redundancy": round(agree / agree_total * 100, 2) if agree_total else None,
        "arbitrate": round(arb_right / arb_total * 100, 2) if arb_total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--by-topic", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results)
    byq, topic_of = load_by_question(results_dir)
    if not byq:
        raise SystemExit(f"no per-mode predictions in {results_dir}")

    overall = metrics_for(list(byq.keys()), byq)
    out = {"overall": overall, "by_topic": {}}

    by_topic = defaultdict(list)
    for q, t in topic_of.items():
        by_topic[t].append(q)
    for t, qids in sorted(by_topic.items()):
        m = metrics_for(qids, byq)
        if m:
            out["by_topic"][t] = m

    (results_dir / "fusion.json").write_text(json.dumps(out, indent=2))

    def row(name, m):
        if not m:
            return f"{name:24s}  (no full 4-mode coverage)"
        return (f"{name:24s} n={m['n']:5d}  full={m['acc_full']:5.1f}  "
                f"sat={m['acc_sat']:5.1f}  sv={m['acc_sv']:5.1f}  blind={m['acc_blind']:5.1f}  "
                f"| vision+{m['vision_gain']:+5.1f}  synergy{m['synergy']:+5.1f}  "
                f"win={m['fusion_win']:4.1f}  hurt={m['fusion_hurt']:4.1f}")

    print(f"\n=== fusion analysis: {results_dir.name} ===")
    print(row("OVERALL", overall))
    if args.by_topic:
        print("-" * 110)
        for t, m in out["by_topic"].items():
            print(row(t, m))
    print(f"\nwrote {results_dir/'fusion.json'}")


if __name__ == "__main__":
    main()
