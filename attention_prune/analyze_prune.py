#!/usr/bin/env python3
"""analyze_prune.py — authoritative Phase-2 pruning analysis (k=1,2,3 multi-seed).

Single source of truth for the slides. Reads the COMPLETE multi-seed run in
solved/qwen36_27b/prune/ (job 266): top/bottom at k=1,2,3, fwd at k=1, three random
seeds (3407,101,202) at k=1,2,3, and the k=0 zero arm. n=1663 solved-greedy
questions per cell.

Produces:
  - per-topic survival by arm (k=1), Wilson CIs, blind baseline, LEAKY tags
  - pooled random over 3 seeds + per-seed spread (so the gap isn't one lucky draw)
  - zero-arm leakage detector + zero-FAIL conditioned survival
  - top-vs-fwd resolution (where attention's #1 disagrees with 'just forward')
  - bottom>=random sanity check
  - the k=1,2,3 survival curve for top / random / bottom

Replaces analyze_prune_k1.py (single-seed). Run: python analyze_prune.py
"""
from __future__ import annotations
import json, math, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRUNE = HERE / "solved/qwen36_27b/prune/prune_results.jsonl"
SOLVED = HERE / "solved/qwen36_27b/solved.jsonl"
BENCH = HERE.parent / "dataset_content/EODATA_compressed_final/benchmark/benchmark_with_answers.jsonl"

URBAN = ["amenity_richness", "building_height", "junction_type", "land_use",
         "road_surface", "road_type", "transit_density", "urban_density"]
SAFE = ["amenity_richness", "transit_density"]
FWD_ROLE = "streetview_along_fwd"
RANDOM_SEEDS = ["3407", "101", "202"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-half)/d, (c+half)/d)


def wilson_diff_ci(k1, n1, k2, n2, z=1.96):
    """Newcombe hybrid-score CI for difference of two proportions (p1-p2)."""
    if n1 == 0 or n2 == 0:
        return (float('nan'), float('nan'))
    l1, u1 = wilson(k1, n1, z); l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1/n1, k2/n2
    lo = (p1 - p2) - math.sqrt((p1-l1)**2 + (u2-p2)**2)
    hi = (p1 - p2) + math.sqrt((u1-p1)**2 + (p2-l2)**2)
    return (lo, hi)


def fmt_ci(k, n):
    if n == 0:
        return "  n/a  "
    lo, hi = wilson(k, n)
    return f"{k/n:5.1%} [{lo:4.1%},{hi:4.1%}] n={n}"


# ---- load prune results ----
rows = [json.loads(l) for l in open(PRUNE) if l.strip()]

# index: byq[qid][(base_arm,k)] = correct ; random pooled across seeds separately
# We store ALL arms with their k so we can build the survival curve.
# base_arm: 'random' (pooled) gets one entry per seed -> keep seed in key for spread.
qtopic = {}
# data[(arm_label, k)][qid] = correct   where arm_label in
#   top, fwd, bottom, zero, random (pooled — multiple seeds OR'd?  No: keep per-seed)
# For pooled random we aggregate counts across seeds (sum k, sum n) — a question
# appears once per seed, so n triples; that's the correct pooled denominator.
cell = collections.defaultdict(dict)   # (arm,k) -> {qid: correct}  (random keeps seed)
for r in rows:
    qtopic[r["question_id"]] = r["topic"]
    arm = r["arm"]; k = r.get("k")
    if arm.startswith("random:"):
        cell[("random", k, arm.split(":")[1])][r["question_id"]] = r["correct"]
    else:
        cell[(arm, k)][r["question_id"]] = r["correct"]

ARMS_K1 = ["top", "fwd", "random", "bottom", "zero"]


def survivals(arm, k, qids):
    """Return (k_correct, n) for an arm at subset size k over the given qids.
    For random, pools across all seeds (n triples)."""
    if arm == "random":
        kk = nn = 0
        for s in RANDOM_SEEDS:
            d = cell.get(("random", k, s), {})
            for q in qids:
                if q in d:
                    nn += 1
                    kk += bool(d[q])
        return kk, nn
    d = cell.get((arm, k), {})
    kk = sum(1 for q in qids if d.get(q))
    nn = sum(1 for q in qids if q in d)
    return kk, nn


# all qids = union over top k=1 (every question ran every arm)
all_qids = list(cell[("top", 1)].keys())
by_topic_qids = collections.defaultdict(list)
for q in all_qids:
    by_topic_qids[qtopic[q]].append(q)

# ---- blind baseline ----
def gold_text(rec):
    ans = rec.get("answer")
    letter = ans.strip().upper()[0] if isinstance(ans, str) and ans.strip() else None
    opts = rec.get("options") or {}
    return opts.get(letter) if letter in (opts or {}) else None

blind = {}
brecs = collections.defaultdict(list)
for l in open(BENCH):
    if not l.strip():
        continue
    rec = json.loads(l)
    if rec.get("topic") in URBAN:
        brecs[rec["topic"]].append(rec)
for t in URBAN:
    rs = brecs[t]; c = collections.Counter(gold_text(r) for r in rs if gold_text(r))
    blind[t] = c.most_common(1)[0][1] / len(rs) if rs else 0.0

# zero-FAIL subset: questions where zero arm (k=0) got it WRONG
zero_d = cell.get(("zero", 0), {})
zerofail = {t: [q for q in by_topic_qids[t] if (q in zero_d and not zero_d[q])] for t in URBAN}

print("="*108)
print("PHASE-2 PRUNING ANALYSIS  (k=1,2,3 multi-seed; random pooled over 3407/101/202; n=1663)")
print("="*108)

# ---- main per-topic table (k=1) ----
print("\n## PER-TOPIC SURVIVAL @ k=1 (all questions in pool)")
print(f"{'topic':18s} {'blind':>6s} | " + " | ".join(f"{a:>26s}" for a in ARMS_K1))
for t in URBAN:
    qids = by_topic_qids[t]
    cells = []
    for a in ARMS_K1:
        kc = 0 if a == "zero" else 1
        kk, nn = survivals(a, kc, qids)
        cells.append(fmt_ci(kk, nn))
    tag = " LEAKY" if blind[t] >= 0.45 else ""
    print(f"{t:18s} {blind[t]:6.3f} | " + " | ".join(f"{c:>26s}" for c in cells) + tag)

# ---- pooled k=1 row + per-seed random spread ----
print("\n## POOLED @ k=1  (all topics)")
for a in ["top", "fwd", "random", "bottom", "zero"]:
    kc = 0 if a == "zero" else 1
    kk, nn = survivals(a, kc, all_qids)
    print(f"  {a:8s} {fmt_ci(kk, nn)}")
print("  -- random per-seed spread @ k=1:")
for s in RANDOM_SEEDS:
    d = cell.get(("random", 1, s), {})
    kk = sum(1 for q in all_qids if d.get(q)); nn = sum(1 for q in all_qids if q in d)
    print(f"     seed {s:>5s}: {fmt_ci(kk, nn)}")

# ---- GAP (top - random) by scope ----
print("\n## GAP (top - random) @ k=1   [Newcombe 95% CI]")
for scope_name, getq in [("ALL topics", lambda t: by_topic_qids[t]),
                         ("SAFE topics (amenity+transit)", lambda t: by_topic_qids[t] if t in SAFE else []),
                         ("ALL zero-FAIL", lambda t: zerofail[t]),
                         ("SAFE zero-FAIL", lambda t: zerofail[t] if t in SAFE else [])]:
    qids = [q for t in URBAN for q in getq(t)]
    kt, nt = survivals("top", 1, qids); kr, nr = survivals("random", 1, qids)
    kf, nf = survivals("fwd", 1, qids); kb, nb = survivals("bottom", 1, qids)
    kz, nz = survivals("zero", 0, qids)
    if not nt:
        continue
    gap = kt/nt - kr/nr if nr else float('nan')
    lo, hi = wilson_diff_ci(kt, nt, kr, nr)
    print(f"\n  [{scope_name}]  n_top={nt} n_rand={nr}")
    print(f"    top    = {fmt_ci(kt,nt)}")
    print(f"    fwd    = {fmt_ci(kf,nf)}")
    print(f"    random = {fmt_ci(kr,nr)}")
    print(f"    bottom = {fmt_ci(kb,nb)}")
    if nz: print(f"    zero   = {fmt_ci(kz,nz)}")
    print(f"    GAP(top-random) = {gap:+.1%}  CI[{lo:+.1%},{hi:+.1%}]  "
          f"{'**includes 0 -> not a finding**' if lo<=0<=hi else 'excludes 0'}")

# ---- k=1,2,3 SURVIVAL CURVE ----
print("\n## SURVIVAL CURVE  (pooled all topics)")
print(f"{'arm':8s} | " + " | ".join(f"{'k='+str(k):>26s}" for k in [1,2,3]))
for a in ["top", "random", "bottom"]:
    cells = []
    for k in [1,2,3]:
        kk, nn = survivals(a, k, all_qids)
        cells.append(fmt_ci(kk, nn))
    print(f"{a:8s} | " + " | ".join(f"{c:>26s}" for c in cells))
# also fwd@1 and zero@0 reference lines
kf, nf = survivals("fwd", 1, all_qids); kz, nz = survivals("zero", 0, all_qids)
print(f"  (ref) fwd@k=1 = {fmt_ci(kf,nf)}   zero@k=0 = {fmt_ci(kz,nz)}")

# ---- per-topic k=1 forward|random|gap (the slide table) ----
print("\n## SLIDE TABLE: per-topic forward | random | gap @ k=1")
print(f"{'topic':18s} {'fwd':>7s} {'random':>7s} {'gap':>7s}")
for t in URBAN:
    qids = by_topic_qids[t]
    kf, nf = survivals("fwd", 1, qids); kr, nr = survivals("random", 1, qids)
    gap = kf/nf - kr/nr if nf and nr else float('nan')
    print(f"{t:18s} {kf/nf:7.1%} {kr/nr:7.1%} {gap:+7.1%}")
kf, nf = survivals("fwd", 1, all_qids); kr, nr = survivals("random", 1, all_qids)
print(f"{'POOLED':18s} {kf/nf:7.1%} {kr/nr:7.1%} {kf/nf-kr/nr:+7.1%}")

# ---- TOP vs FWD resolution ----
print("\n## TOP vs FWD — attention rank-1 role from solved.jsonl")
rank1 = {}
for l in open(SOLVED):
    d = json.loads(l)
    if d.get("status") != "solved_greedy":
        continue
    imgs = d["attention"]["images"]
    rank1[d["question_id"]] = max(imgs, key=lambda im: im["avg_norm_pct"])["role"]

agree = sum(1 for q in all_qids if rank1.get(q) == FWD_ROLE)
disagree = [q for q in all_qids if rank1.get(q) and rank1[q] != FWD_ROLE]
print(f"  attention rank-1 == along_fwd : {agree}/{len(all_qids)} = {agree/len(all_qids):.1%}")
print(f"  attention rank-1 != along_fwd : {len(disagree)} questions")
kt = sum(cell[("top",1)].get(q, False) for q in disagree)
kf = sum(cell[("fwd",1)].get(q, False) for q in disagree)
n = len(disagree)
if n:
    print(f"  On disagree subset: top = {fmt_ci(kt,n)}   fwd = {fmt_ci(kf,n)}")
    ld, hd = wilson_diff_ci(kt, n, kf, n)
    print(f"  GAP(top-fwd) on disagree = {(kt-kf)/n:+.1%} CI[{ld:+.1%},{hd:+.1%}] "
          f"{'includes 0' if ld<=0<=hd else 'EXCLUDES 0 -> attention wins'}")
    print("  rank-1 role when != fwd:", dict(collections.Counter(rank1[q] for q in disagree)))

# ---- bottom vs random sanity ----
print("\n## BOTTOM vs RANDOM @ k=1")
kb, nb = survivals("bottom", 1, all_qids); kr, nr = survivals("random", 1, all_qids)
lo, hi = wilson_diff_ci(kb, nb, kr, nr)
print(f"  bottom = {fmt_ci(kb,nb)}   random = {fmt_ci(kr,nr)}")
print(f"  GAP(bottom-random) = {(kb/nb-kr/nr):+.1%} CI[{lo:+.1%},{hi:+.1%}]")

print("\nDONE.")
