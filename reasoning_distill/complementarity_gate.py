#!/usr/bin/env python3
"""
complementarity_gate.py — Modality-necessity acceptance gate for two-perspective VQA.

Given an ablation log (one JSON object per line, with at least the fields
`question_id`, `mode` in {full, sat_only, sv_only, blind}, `topic`, `correct`),
compute the complementarity metrics that decide whether a task GENUINELY requires
fusing satellite + street-view, and emit a PASS/FAIL verdict per task.

This is the gate you run on ~100-150 candidate questions BEFORE committing compute
to generating the full set. A task only earns the "two-perspective" label if the
SECOND view contributes accuracy that NEITHER single view can.

Metrics (per task and overall), all on the set of questions that have all 4 modes:
  acc_full / acc_sat / acc_sv / acc_blind
  synergy        = acc_full - max(acc_sat, acc_sv)          # headline gap
  vision_gain    = acc_full - acc_blind                      # is vision used at all
  fusion_win     = P(full right AND both singles wrong)      # ONLY fusion solves it
  fusion_hurt    = P(full wrong AND some single right)       # 2nd view distracts
  redundancy     = P(sat_only and sv_only give SAME answer)  # views encode same signal
  arbitrate      = P(full right | sat/sv disagree)           # can fusion use a conflict?

Acceptance (all must hold for a task to count as two-perspective-dependent):
  blind_ok    : acc_blind <= chance + 8pts        (kills text/option leakage)
  synergy_ok  : synergy   >= +8 pts
  win_ok      : fusion_win >= 15%
  net_ok      : fusion_win >  fusion_hurt          (2nd view helps more than it hurts)

Usage:
  python complementarity_gate.py LOG.jsonl [--chance 25] [--min-n 30] \
      [--attr-only] [--json OUT.json]
"""
import argparse, collections, json, sys

MODES = ["full", "sat_only", "sv_only", "blind"]
ATTR = ["land_use", "building_height", "urban_density", "junction_type",
        "green_space", "amenity_richness", "road_type", "road_surface",
        "transit_density"]


def topic_of(rec):
    if rec.get("topic"):
        return rec["topic"]
    qid = rec.get("question_id", "")
    for t in sorted(ATTR, key=len, reverse=True):
        if t in qid:
            return t
    return "unknown"


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def index_by_question(rows):
    """question_id -> {mode: (correct_bool, predicted_letter, topic)}"""
    byq = collections.defaultdict(dict)
    for r in rows:
        m = r.get("mode")
        if m not in MODES:
            continue
        byq[r["question_id"]][m] = (
            bool(r.get("correct")), r.get("letter"), topic_of(r))
    return byq


def metrics_for(qids, byq):
    acc = {m: [0, 0] for m in MODES}
    fusion_win = fusion_hurt = n = 0
    agree = agree_total = 0
    arb_right = arb_total = 0
    for q in qids:
        d = byq[q]
        if not all(m in d for m in MODES):
            continue
        n += 1
        for m in MODES:
            acc[m][1] += 1
            acc[m][0] += int(d[m][0])
        sat_ok, sat_p, _ = d["sat_only"]
        sv_ok, sv_p, _ = d["sv_only"]
        full_ok = d["full"][0]
        either = sat_ok or sv_ok
        if full_ok and not either:
            fusion_win += 1
        if (not full_ok) and either:
            fusion_hurt += 1
        # redundancy / arbitration use predicted letters
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
        "acc_full": full, "acc_sat": sat, "acc_sv": sv, "acc_blind": blind,
        "synergy": full - max(sat, sv),
        "vision_gain": full - blind,
        "fusion_win": fusion_win / n * 100,
        "fusion_hurt": fusion_hurt / n * 100,
        "redundancy": (agree / agree_total * 100) if agree_total else float("nan"),
        "arbitrate": (arb_right / arb_total * 100) if arb_total else float("nan"),
    }


def verdict(m, chance):
    checks = {
        "blind_ok": m["acc_blind"] <= chance + 8,
        "synergy_ok": m["synergy"] >= 8.0,
        "win_ok": m["fusion_win"] >= 15.0,
        "net_ok": m["fusion_win"] > m["fusion_hurt"],
    }
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--chance", type=float, default=25.0,
                    help="random-guess accuracy (default 25 for 4-way MCQ)")
    ap.add_argument("--min-n", type=int, default=30,
                    help="skip tasks with fewer than this many 4-mode questions")
    ap.add_argument("--attr-only", action="store_true",
                    help="restrict to the 9 attribute tasks")
    ap.add_argument("--json", help="write full metrics to this JSON file")
    args = ap.parse_args()

    rows = load(args.log)
    byq = index_by_question(rows)

    # group question ids by topic
    topic_qs = collections.defaultdict(list)
    for q, d in byq.items():
        t = next((v[2] for v in d.values()), "unknown")
        if args.attr_only and t not in ATTR:
            continue
        topic_qs[t].append(q)

    out = {}
    print(f"{'task':20s} {'n':>4s} {'full':>5s} {'sat':>5s} {'sv':>5s} "
          f"{'blnd':>5s} {'syn':>6s} {'win%':>5s} {'hurt%':>6s} "
          f"{'redun%':>7s} {'arb%':>5s}  verdict")
    print("-" * 100)
    all_qs = []
    for t in sorted(topic_qs):
        qs = topic_qs[t]
        m = metrics_for(qs, byq)
        if m is None or m["n"] < args.min_n:
            continue
        all_qs += qs
        ok, checks = verdict(m, args.chance)
        out[t] = {**m, "pass": ok, "checks": checks}
        tag = "PASS ✓" if ok else "fail  " + ",".join(
            k for k, v in checks.items() if not v)
        print(f"{t:20s} {m['n']:4d} {m['acc_full']:5.1f} {m['acc_sat']:5.1f} "
              f"{m['acc_sv']:5.1f} {m['acc_blind']:5.1f} {m['synergy']:+6.1f} "
              f"{m['fusion_win']:5.1f} {m['fusion_hurt']:6.1f} "
              f"{m['redundancy']:7.1f} {m['arbitrate']:5.1f}  {tag}")

    agg = metrics_for(all_qs, byq)
    if agg:
        ok, checks = verdict(agg, args.chance)
        out["__overall__"] = {**agg, "pass": ok, "checks": checks}
        print("-" * 100)
        tag = "PASS ✓" if ok else "fail  " + ",".join(
            k for k, v in checks.items() if not v)
        print(f"{'OVERALL':20s} {agg['n']:4d} {agg['acc_full']:5.1f} "
              f"{agg['acc_sat']:5.1f} {agg['acc_sv']:5.1f} {agg['acc_blind']:5.1f} "
              f"{agg['synergy']:+6.1f} {agg['fusion_win']:5.1f} "
              f"{agg['fusion_hurt']:6.1f} {agg['redundancy']:7.1f} "
              f"{agg['arbitrate']:5.1f}  {tag}")

    print("\nGATE: a task is two-perspective-dependent iff blind≈chance, synergy≥+8,")
    print("fusion_win≥15%, and fusion_win>fusion_hurt. redundancy (lower=better) and")
    print("arbitrate (higher=better) are diagnostics, not pass/fail.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
