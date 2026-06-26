#!/usr/bin/env python3
"""Generate figures for the Phase-2 pruning slide deck. Pulls numbers straight
from analyze_prune_k1.py logic so the figures cannot drift from the report."""
import json, math, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AP = HERE.parent.parent / "attention_prune"
PRUNE = AP / "solved/qwen36_27b/prune_k1_singleseed_265/prune_results.jsonl"
SOLVED = AP / "solved/qwen36_27b/solved.jsonl"
BENCH = AP.parent / "dataset_content/EODATA_compressed_final/benchmark/benchmark_with_answers.jsonl"
OUT = HERE

URBAN = ["amenity_richness","building_height","junction_type","land_use",
         "road_surface","road_type","transit_density","urban_density"]
SAFE = {"amenity_richness","transit_density"}
ARMS = ["top","fwd","random","bottom","zero"]
COL = {"top":"#0D47A1","fwd":"#1565C0","random":"#9E9E9E","bottom":"#C62828","zero":"#37474F"}

def wilson(k,n,z=1.96):
    if n==0: return (0,0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n)
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return (p,(c-h)/d,(c+h)/d)

rows=[json.loads(l) for l in open(PRUNE) if l.strip()]
byq=collections.defaultdict(dict); qtopic={}
for r in rows:
    a=r["arm"].split(":")[0]; qtopic[r["question_id"]]=r["topic"]
    byq[r["question_id"]][a]=r["correct"]
bytopic=collections.defaultdict(list)
for q in byq: bytopic[qtopic[q]].append(q)
zerofail={t:[q for q in bytopic[t] if not byq[q].get("zero",True)] for t in URBAN}

def agg(qids,a):
    n=sum(1 for q in qids if a in byq[q]); k=sum(1 for q in qids if byq[q].get(a))
    return k,n
def pooled(scope,a):
    qs=[q for t in scope for q in bytopic[t]]; return agg(qs,a)
def pooled_zf(scope,a):
    qs=[q for t in scope for q in zerofail[t]]; return agg(qs,a)

# blind
def gold_text(rec):
    ans=rec.get("answer"); L=ans.strip().upper()[0] if isinstance(ans,str) and ans.strip() else None
    o=rec.get("options") or {}; return o.get(L)
brecs=collections.defaultdict(list)
for l in open(BENCH):
    if l.strip():
        r=json.loads(l)
        if r.get("topic") in URBAN: brecs[r["topic"]].append(r)
blind={t:(collections.Counter(gold_text(r) for r in brecs[t] if gold_text(r)).most_common(1)[0][1]/len(brecs[t])) for t in URBAN}

# ---------- FIG 1: pooled survival bars (all topics) with CI ----------
fig,ax=plt.subplots(figsize=(7.2,4.0))
vals=[]; los=[]; his=[]
for a in ARMS:
    k,n=pooled(URBAN,a); p,lo,hi=wilson(k,n); vals.append(p); los.append(p-lo); his.append(hi-p)
x=np.arange(len(ARMS))
bars=ax.bar(x,[v*100 for v in vals],yerr=[ [l*100 for l in los],[h*100 for h in his]],
            color=[COL[a] for a in ARMS],capsize=4,width=0.62)
for xi,v in zip(x,vals): ax.text(xi,v*100+1.5,f"{v*100:.1f}",ha="center",fontsize=10,fontweight="bold")
ax.axhline(25,ls=":",c="#888",lw=1); ax.text(len(ARMS)-0.5,26,"chance 25%",fontsize=8,c="#888",ha="right")
ax.set_xticks(x); ax.set_xticklabels(["top\n(attn #1)","fwd\n(forward)","random","bottom\n(attn #4)","zero\n(no img)"],fontsize=9)
ax.set_ylabel("survival %  (of 1663 solved q)"); ax.set_ylim(0,105)
ax.set_title("k=1 survival, pooled over all 8 topics",fontsize=12)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"prune_pooled.pdf"); plt.close(fig)

# ---------- FIG 2: top vs fwd vs random on SAFE pool (all-q & zero-fail) ----------
fig,ax=plt.subplots(figsize=(7.2,4.0))
groups=[("SAFE\nall-q",lambda a:pooled(SAFE,a)),("SAFE\nzero-FAIL",lambda a:pooled_zf(SAFE,a))]
arms3=["top","fwd","random","zero"]
w=0.2
for gi,(gn,fn) in enumerate(groups):
    for ai,a in enumerate(arms3):
        k,n=fn(a); p,lo,hi=wilson(k,n)
        xx=gi+ (ai-1.5)*w
        ax.bar(xx,p*100,width=w,color=COL[a],yerr=[[ (p-lo)*100],[(hi-p)*100]],capsize=3,
               label=a if gi==0 else None)
        ax.text(xx,p*100+1.5,f"{p*100:.0f}",ha="center",fontsize=8)
ax.set_xticks([0,1]); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("survival %"); ax.set_ylim(0,100)
ax.set_title("Leakage-safe pool (amenity + transit): top$\\approx$fwd $>$ random",fontsize=11)
ax.legend(ncol=4,fontsize=8,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.28))
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"prune_safe.pdf"); plt.close(fig)

# ---------- FIG 3: per-topic top survival vs blind & zero ----------
fig,ax=plt.subplots(figsize=(8.2,4.2))
order=sorted(URBAN,key=lambda t:blind[t])
y=np.arange(len(order));
top=[pooled([t],"top")[0]/pooled([t],"top")[1] for t in order]
zero=[ (lambda kn:kn[0]/kn[1])(pooled([t],"zero")) for t in order]
ax.barh(y+0.0,[v*100 for v in top],height=0.38,color="#0D47A1",label="top (attn #1 img)")
ax.barh(y-0.4,[v*100 for v in zero],height=0.38,color="#37474F",label="zero (no image)")
ax.scatter([blind[t]*100 for t in order],y-0.2,marker="D",color="#FFB300",zorder=5,label="blind prior",s=42,edgecolor="k",linewidth=0.5)
for i,t in enumerate(order):
    tag="  ✓safe" if t in SAFE else "  leaky"
    ax.text(101,i-0.2,tag,fontsize=8,va="center",color=("#2E7D32" if t in SAFE else "#C62828"))
ax.set_yticks(y); ax.set_yticklabels([t.replace("_"," ") for t in order],fontsize=9)
ax.set_xlabel("%"); ax.set_xlim(0,118); ax.axvline(25,ls=":",c="#888",lw=1)
ax.set_title("Per-topic: top survival vs zero-image survival vs blind prior",fontsize=11)
ax.legend(fontsize=8,loc="lower right",frameon=False)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"prune_pertopic.pdf"); plt.close(fig)

# ---------- FIG 4: gap(top-random) across scopes ----------
fig,ax=plt.subplots(figsize=(7.2,3.6))
def gap_ci(scope,zf=False):
    f=pooled_zf if zf else pooled
    kt,nt=f(scope,"top"); kr,nr=f(scope,"random")
    pt,_,_=wilson(kt,nt); pr,_,_=wilson(kr,nr)
    pt2,lt,ut=wilson(kt,nt); pr2,lr,ur=wilson(kr,nr)
    g=pt-pr; lo=g-math.sqrt((pt-lt)**2+(ur-pr)**2); hi=g+math.sqrt((ut-pt)**2+(pr-lr)**2)
    return g*100,lo*100,hi*100
scopes=[("All, all-q",lambda:gap_ci(URBAN)),("All, zero-FAIL",lambda:gap_ci(URBAN,True)),
        ("Safe, all-q",lambda:gap_ci(SAFE)),("Safe, zero-FAIL",lambda:gap_ci(SAFE,True))]
ys=np.arange(len(scopes))
for i,(nm,fn) in enumerate(scopes):
    g,lo,hi=fn()
    sig = lo>0
    ax.errorbar(g,i,xerr=[[g-lo],[hi-g]],fmt="o",color=("#2E7D32" if sig else "#C62828"),capsize=4,ms=7)
    ax.text(hi+0.8,i,f"{g:+.1f} [{lo:+.0f},{hi:+.0f}]"+("  ✓" if sig else "  n.s."),va="center",fontsize=9,
            color=("#2E7D32" if sig else "#C62828"))
ax.axvline(0,c="k",lw=1)
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in scopes]); ax.set_xlim(-6,30)
ax.set_xlabel("gap(top $-$ random),  %  [Newcombe 95% CI]")
ax.set_title("Does the attended image beat a random one?",fontsize=11)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"prune_gap.pdf"); plt.close(fig)

print("figures written to", OUT)
