#!/usr/bin/env python3
"""analyze_prune_k1.py — authoritative Phase-2 (k=1) pruning analysis.

Single source of truth for PRUNE_ANALYSIS.md and the slides. Computes, from the
COMPLETE single-seed k=1 run (prune_k1_singleseed_265/), everything the brief asks
for: per-topic survival by arm, blind baseline, zero-arm leakage detector,
zero-FAIL conditioned survival, top-vs-fwd resolution, bottom>=random check, Wilson
CIs on every rate and gap.

NOTE on data source: the multi-seed / k=1,2,3 run in prune/ is only 2.1% complete
(~36 q/cell) -> unusable for CIs. Per user decision we analyze the complete
single-seed (random seed 3407) k=1 run. Consequences (state in limitations):
  - no per-seed random spread (single seed only)
  - no k=2 / k=3 survival curve (k=1 only)
"""
from __future__ import annotations
import json, math, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRUNE = HERE / "solved/qwen36_27b/prune_k1_singleseed_265/prune_results.jsonl"
SOLVED = HERE / "solved/qwen36_27b/solved.jsonl"
BENCH = HERE.parent / "dataset_content/EODATA_compressed_final/benchmark/benchmark_with_answers.jsonl"

URBAN = ["amenity_richness", "building_height", "junction_type", "land_use",
         "road_surface", "road_type", "transit_density", "urban_density"]
SAFE = ["amenity_richness", "transit_density"]
FWD_ROLE = "streetview_along_fwd"


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi)."""
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


# ---- load prune results, normalize random:* -> random ----
rows = [json.loads(l) for l in open(PRUNE) if l.strip()]
for r in rows:
    if r["arm"].startswith("random"):
        r["base_arm"] = "random"
    else:
        r["base_arm"] = r["arm"]

# per question_id, arm -> correct (k=1; zero is k=0)
# structure: byq[qid][arm] = bool correct ; also topic
byq = collections.defaultdict(dict)
qtopic = {}
for r in rows:
    qtopic[r["question_id"]] = r["topic"]
    byq[r["question_id"]][r["base_arm"]] = r["correct"]

ARMS = ["top", "fwd", "random", "bottom", "zero"]

# ---- blind baseline (from benchmark, modal correct-option-TEXT) ----
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


# ---- aggregate survival per (topic, arm) ----
def agg(qids, arm):
    n = sum(1 for q in qids if arm in byq[q])
    k = sum(1 for q in qids if byq[q].get(arm))
    return k, n

all_qids = list(byq.keys())
by_topic_qids = collections.defaultdict(list)
for q in all_qids:
    by_topic_qids[qtopic[q]].append(q)

# zero-FAIL subset: questions where zero arm got it WRONG (genuinely need an image)
zerofail = {t: [q for q in by_topic_qids[t] if (q in byq and 'zero' in byq[q] and not byq[q]['zero'])]
            for t in URBAN}

print("="*100)
print("PHASE-2 k=1 PRUNING ANALYSIS  (single-seed 3407, complete run)")
print("="*100)

# ---- main per-topic table ----
print("\n## PER-TOPIC SURVIVAL (all questions in pool)")
print(f"{'topic':18s} {'blind':>6s} | " + " | ".join(f"{a:>26s}" for a in ARMS))
for t in URBAN:
    qids = by_topic_qids[t]
    cells = []
    for a in ARMS:
        kk, nn = agg(qids, a)
        cells.append(fmt_ci(kk, nn))
    tag = " LEAKY" if blind[t] >= 0.45 else ""
    print(f"{t:18s} {blind[t]:6.3f} | " + " | ".join(f"{c:>26s}" for c in cells) + tag)

# ---- ZERO-FAIL conditioned survival (image-dependent subpopulation) ----
print("\n## SURVIVAL ON ZERO-FAIL SUBSET (questions the model gets WRONG with no image)")
print(f"{'topic':18s} {'n_zf':>5s} | " + " | ".join(f"{a:>26s}" for a in ['top','fwd','random','bottom']))
for t in URBAN:
    qids = zerofail[t]
    cells = []
    for a in ['top','fwd','random','bottom']:
        kk, nn = agg(qids, a)
        cells.append(fmt_ci(kk, nn))
    tag = " LEAKY" if blind[t] >= 0.45 else ""
    print(f"{t:18s} {len(qids):5d} | " + " | ".join(f"{c:>26s}" for c in cells) + tag)

# ---- top vs random GAP, all-pool and zero-fail, with diff CI ----
print("\n## GAP (top - random)  [Newcombe 95% CI]   pooled within scope")
def pooled(qids_groups, arm):
    k = sum(byq[q].get(arm, False) for grp in qids_groups for q in grp)
    n = sum(1 for grp in qids_groups for q in grp if arm in byq[q])
    return k, n

for scope_name, getq in [("ALL topics", lambda t: by_topic_qids[t]),
                         ("SAFE topics (amenity+transit)", lambda t: by_topic_qids[t] if t in SAFE else []),
                         ("ALL zero-FAIL", lambda t: zerofail[t]),
                         ("SAFE zero-FAIL", lambda t: zerofail[t] if t in SAFE else [])]:
    grps = [getq(t) for t in URBAN]
    kt, nt = pooled(grps, "top"); kr, nr = pooled(grps, "random")
    kf, nf = pooled(grps, "fwd"); kb, nb = pooled(grps, "bottom"); kz, nz = pooled(grps, "zero")
    gap = kt/nt - kr/nr if nt and nr else float('nan')
    lo, hi = wilson_diff_ci(kt, nt, kr, nr)
    print(f"\n  [{scope_name}]  n_top={nt} n_rand={nr}")
    print(f"    top    = {fmt_ci(kt,nt)}")
    print(f"    fwd    = {fmt_ci(kf,nf)}")
    print(f"    random = {fmt_ci(kr,nr)}")
    print(f"    bottom = {fmt_ci(kb,nb)}")
    if nz: print(f"    zero   = {fmt_ci(kz,nz)}")
    print(f"    GAP(top-random) = {gap:+.1%}  CI[{lo:+.1%},{hi:+.1%}]  "
          f"{'**includes 0 -> not a finding**' if lo<=0<=hi else 'excludes 0'}")
    # top-fwd gap
    gtf = kt/nt - kf/nf if nt and nf else float('nan')
    ltf, htf = wilson_diff_ci(kt, nt, kf, nf)
    print(f"    GAP(top-fwd)    = {gtf:+.1%}  CI[{ltf:+.1%},{htf:+.1%}]  "
          f"{'includes 0' if ltf<=0<=htf else 'excludes 0'}")

# ---- TOP vs FWD resolution (duty #2): need attention rank-1 role from solved.jsonl ----
print("\n## TOP vs FWD — does attention's #1 image disagree with 'just forward'?")
rank1 = {}
for l in open(SOLVED):
    d = json.loads(l)
    if d.get("status") != "solved_greedy":
        continue
    imgs = d["attention"]["images"]
    top_img = max(imgs, key=lambda im: im["avg_norm_pct"])
    rank1[d["question_id"]] = top_img["role"]

agree = sum(1 for q in all_qids if rank1.get(q) == FWD_ROLE)
disagree_qids = [q for q in all_qids if rank1.get(q) and rank1[q] != FWD_ROLE]
print(f"  attention rank-1 == along_fwd : {agree}/{len(all_qids)} = {agree/len(all_qids):.1%}")
print(f"  attention rank-1 != along_fwd : {len(disagree_qids)} questions  <- where attention EARNS its keep")
# on disagree subset: does top beat fwd?
kt = sum(byq[q].get('top', False) for q in disagree_qids)
kf = sum(byq[q].get('fwd', False) for q in disagree_qids)
n = len(disagree_qids)
print(f"  On disagree subset: top survival = {fmt_ci(kt,n)}")
print(f"                      fwd survival = {fmt_ci(kf,n)}")
ld, hd = wilson_diff_ci(kt, n, kf, n)
print(f"  GAP(top-fwd) on disagree subset = {(kt-kf)/n:+.1%} CI[{ld:+.1%},{hd:+.1%}] "
      f"{'includes 0' if ld<=0<=hd else 'EXCLUDES 0 -> attention wins'}")
# what roles does attention pick when not fwd?
rc = collections.Counter(rank1[q] for q in disagree_qids)
print("  rank-1 role distribution when != fwd:", dict(rc))

# ---- bottom >= random anomaly (duty #4) ----
print("\n## BOTTOM vs RANDOM (duty #4)")
kb, nb = pooled([by_topic_qids[t] for t in URBAN], "bottom")
kr, nr = pooled([by_topic_qids[t] for t in URBAN], "random")
print(f"  pooled bottom = {fmt_ci(kb,nb)}   random = {fmt_ci(kr,nr)}")
lo, hi = wilson_diff_ci(kb, nb, kr, nr)
print(f"  GAP(bottom-random) = {(kb/nb-kr/nr):+.1%} CI[{lo:+.1%},{hi:+.1%}]")
# verify bottom really kept low-attention images: rank-4 role
rank4 = {}
for l in open(SOLVED):
    d = json.loads(l)
    if d.get("status") != "solved_greedy": continue
    imgs = d["attention"]["images"]
    rank4[d["question_id"]] = min(imgs, key=lambda im: im["avg_norm_pct"])["role"]
# bottom arm kept_roles check
bottom_kept = {}
for r in rows:
    if r["base_arm"] == "bottom":
        bottom_kept[r["question_id"]] = r["kept_roles"][0] if r["kept_roles"] else None
match = sum(1 for q in bottom_kept if bottom_kept[q] == rank4.get(q))
print(f"  bottom arm kept the rank-4 (least-attended) image in {match}/{len(bottom_kept)} = "
      f"{match/len(bottom_kept):.1%} of questions (sanity: should be ~100%)")

print("\nDONE.")
