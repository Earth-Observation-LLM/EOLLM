#!/usr/bin/env python3
"""Score the four zero-shot RS-VLMs (GeoChat, SkySenseGPT, LHRS-Bot-Nova,
EarthDial) sat_only results into ONE per-topic Family-1 markdown table.

Run on lab-ws against results/<key>/sat_only_predictions.jsonl. Per-topic +
overall Family-1 accuracy per model, unparseable rate, runtime/throughput/VRAM
from each model's meta.json (+ optional VRAM sampler log).
"""
import json, sys, os
from collections import defaultdict

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"

FAMILY1 = ["land_use", "building_height", "urban_density", "road_type",
           "road_surface", "junction_type", "green_space", "amenity_richness",
           "transit_density"]
# (results-dir key, display, vram log or None)
MODELS = [
    ("geochat_7b", "GeoChat-7B", None),
    ("skysensegpt_7b", "SkySenseGPT-7B", None),
    ("lhrs_bot_nova", "LHRS-Bot-Nova", "logs/vram_lhrs.log"),
    ("earthdial_4b_rgb", "EarthDial-4B-RGB", "logs/vram_ed.log"),
]


def load(key):
    pred_path = os.path.join(RESULTS, key, "sat_only_predictions.jsonl")
    if not os.path.exists(pred_path):
        return None
    per_topic = defaultdict(lambda: [0, 0])
    unparse = n = 0
    for line in open(pred_path):
        r = json.loads(line)
        per_topic[r["topic"]][1] += 1
        if r["prediction"] is None or r["prediction"] not in ("A", "B", "C", "D"):
            unparse += 1
        if r["is_correct"]:
            per_topic[r["topic"]][0] += 1
        n += 1
    meta = {}
    mp = os.path.join(RESULTS, key, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    return {"per_topic": per_topic, "unparse": unparse, "n": n, "meta": meta}


stats = {k: load(k) for k, _, _ in MODELS}


def acc(d, t):
    if not d:
        return "—"
    c, n = d["per_topic"].get(t, [0, 0])
    return f"{c/n:.3f} ({c}/{n})" if n else "—"


def overall(d):
    if not d:
        return "—"
    c = sum(v[0] for v in d["per_topic"].values())
    n = sum(v[1] for v in d["per_topic"].values())
    return f"**{c/n:.3f}** ({c}/{n})" if n else "—"


disp = {k: d for k, d, _ in MODELS}
hdr = "| topic | #questions | " + " | ".join(disp[k] + " acc" for k, _, _ in MODELS) + " |"
sep = "|---|---:|" + "---:|" * len(MODELS)
lines = [hdr, sep]
for t in FAMILY1:
    nq = max((stats[k]["per_topic"].get(t, [0, 0])[1] if stats[k] else 0)
             for k, _, _ in MODELS)
    row = [acc(stats[k], t) for k, _, _ in MODELS]
    lines.append(f"| {t} | {nq} | " + " | ".join(row) + " |")
tot = max((stats[k]["n"] if stats[k] else 0) for k, _, _ in MODELS)
lines.append(f"| **OVERALL Family-1** | **{tot}** | "
             + " | ".join(overall(stats[k]) for k, _, _ in MODELS) + " |")

print("\n".join(lines))
print("\n### Run stats")
for key, d, vlog in MODELS:
    s = stats.get(key)
    if not s:
        print(f"- **{d}**: NO RESULTS")
        continue
    m = s["meta"]
    el = m.get("elapsed_s")
    qps = (s["n"] / el) if el else None
    vram = None
    if vlog and os.path.exists(vlog):
        vals = [int(x) for x in open(vlog).read().split() if x.strip().isdigit()]
        vram = max(vals) if vals else None
    extra = f", peak VRAM={vram} MiB ({vram/1024:.1f} GiB)" if vram else ""
    print(f"- **{d}**: n={s['n']}, unparseable={s['unparse']} "
          f"({s['unparse']/s['n']*100:.2f}%), "
          f"runtime={el}s ({el/60:.1f} min), "
          f"throughput={qps:.1f} Q/s, backend={m.get('backend')}{extra}")
