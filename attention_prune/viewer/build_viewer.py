#!/usr/bin/env python3
"""build_viewer.py — self-contained HTML viewer for the solved.jsonl collector
output.

Goal: let you EYEBALL, per solved question, how the model's attention is spread
across the 4 street-view angles, and HOW THAT RANKING CHANGES ACROSS LAYERS, so
you can pick the layer(s) to rank Phase-2 pruning on. For each solved question:
  - the 4 street-view angle images (rebuilt byte-identical via the suite, inlined),
  - ONE attention bar per image (answer-token attention). NOTE: every answer is a
    single token (the bare letter), so the "decision-moment" (choice) and
    "answer-averaged" (avg) signals are mathematically IDENTICAL — the only real
    variation in this data is ACROSS LAYERS. There is therefore no choice/avg
    toggle; the only knob that matters is the layer scrubber.
  - a layer scrubber (per_layer shares): 'mean' shows the across-layer mean, any
    L0..L15 shows that single layer's normalized per-image shares. Bars, numbers,
    and the green top-image outline all update together.
  - a compact per-card layer×image grid that highlights the argmax image per
    layer, so you can see at a glance whether late layers sharpen onto one angle.
  - the question / options / gold / prediction, the winning ladder settings,
    and the solve status (greedy / sampled / unstable).

Reads attention_prune/solved/<model>/solved.jsonl. Images are rebuilt from each
record through benchmark_suite.images so the viewer shows exactly the pixels the
model saw, keeping solved.jsonl small.

Usage:
    python build_viewer.py --solved ../solved/qwen36_27b
    python build_viewer.py --solved ../solved/qwen36_27b --limit 400 --status solved_greedy
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRUNE = HERE.parent
REPO = PRUNE.parent

THUMB_PX = 320


def _img_b64(pil) -> str:
    im = pil.convert("RGB")
    im.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _remap(path: str, recorded_root: str | None, local_root: Path) -> str:
    """Records carry lab-ws absolute image paths; remap them onto the local tree.

    The directory layout under image_root is identical on both machines, so we
    swap the recorded image_root prefix for the local one. Falls back to matching
    by the tail after '/benchmark/' if the prefix doesn't line up."""
    if not path:
        return path
    if Path(path).exists():
        return path
    if recorded_root and path.startswith(recorded_root):
        cand = str(local_root / Path(path).relative_to(recorded_root))
        if Path(cand).exists():
            return cand
    marker = "/benchmark/"
    if marker in path:
        cand = str(local_root / path.split(marker, 1)[1])
        if Path(cand).exists():
            return cand
    return path


def build(solved_dir: Path, image_root: Path, limit: int, status_filter: str | None):
    sp = solved_dir / "solved.jsonl"
    if not sp.exists():
        raise SystemExit(f"no solved.jsonl in {solved_dir}")
    meta = {}
    mp = solved_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    recorded_root = (meta.get("dataset", {}) or {}).get("image_root")

    cards = []
    n = 0
    for line in open(sp):
        if not line.strip():
            continue
        r = json.loads(line)
        if status_filter and r.get("status") != status_filter:
            continue
        attn = r.get("attention")  # None for solved_unstable (no bars, still shown)
        # solved.jsonl stores image_paths (not the full record), so we load the SV
        # angle files the collector recorded — these are the raw streetview source
        # files; the corner DIRECTION label the model saw is cosmetic for eyeballing.
        from PIL import Image
        imgs_b64 = []
        for role, path in zip(r.get("image_roles", []), r.get("image_paths", [])):
            lp = _remap(path, recorded_root, image_root)
            if lp and Path(lp).exists():
                imgs_b64.append({"role": role, "src": _img_b64(Image.open(lp))})
            else:
                imgs_b64.append({"role": role, "src": None})
        cards.append({
            "question_id": r.get("question_id"), "topic": r.get("topic"),
            "city": r.get("city"), "question": r.get("question"),
            "options": r.get("options"), "gold": r.get("gold"),
            "prediction": r.get("prediction"), "status": r.get("status"),
            "winning": r.get("winning"), "note": r.get("note"),
            "images": imgs_b64, "attention": attn,
        })
        n += 1
        if limit and n >= limit:
            break

    html = _render(cards, meta, solved_dir.name)
    out = solved_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(cards)} cards)")
    return out


def _render(cards, meta, title) -> str:
    data = json.dumps(cards, ensure_ascii=False)
    metaj = json.dumps(meta, ensure_ascii=False)
    return _TEMPLATE.replace("__DATA__", data).replace("__META__", metaj).replace("__TITLE__", title)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>attention_prune — __TITLE__</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#0e1116;color:#e6edf3}
 header{padding:12px 18px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:5}
 header h1{font-size:16px;margin:0 0 4px}
 .note{font-size:12px;color:#8b949e;margin:0 0 8px;max-width:980px}
 .controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px}
 select,input{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:5px;padding:4px 6px}
 .wrap{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:16px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:9px;padding:12px}
 .q{font-weight:600;margin-bottom:4px}
 .meta{font-size:12px;color:#8b949e;margin-bottom:8px}
 .badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;margin-right:5px}
 .b-greedy{background:#196c2e}.b-sampled{background:#9e6a03}.b-unstable{background:#6e2630}
 .imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px}
 .imw{border:1px solid #21262d;border-radius:6px;padding:6px;background:#0d1117}
 .imw img{width:100%;border-radius:4px;display:block}
 .role{font-size:11px;color:#8b949e;margin:4px 0 2px}
 .bar{height:10px;background:#21262d;border-radius:4px;overflow:hidden;margin:2px 0}
 .bar>span{display:block;height:100%;background:#3b82f6}
 .barlab{display:flex;justify-content:space-between;font-size:10px;color:#8b949e}
 .opts{font-size:12px;margin-top:8px;color:#c9d1d9}
 .opt-gold{color:#3fb950;font-weight:600}
 .topwin{outline:2px solid #3fb950}
 .grid{margin-top:10px;font-size:10px;border-collapse:collapse;width:100%}
 .grid th,.grid td{border:1px solid #21262d;padding:1px 3px;text-align:center;color:#8b949e}
 .grid th{color:#c9d1d9;font-weight:600}
 .grid td.am{background:#1f6feb33;color:#e6edf3;font-weight:600}
 .grid td.cur{outline:1px solid #f0883e}
 .grid .rowh{text-align:left;color:#c9d1d9;white-space:nowrap;max-width:120px;overflow:hidden;text-overflow:ellipsis}
 .more{display:block;margin:18px auto;padding:8px 18px;background:#21262d;border:1px solid #30363d;
       border-radius:6px;color:#e6edf3;cursor:pointer;font-size:13px}
</style></head>
<body>
<header>
 <h1>attention_prune — __TITLE__</h1>
 <p class="note">Answers are single-token, so decision-moment == answer-averaged — the only real
 variation is <b>across layers</b>. Use the layer scrubber: <b>mean</b> = across-layer mean;
 <b>L0..Ln</b> = that single layer's normalized per-image shares. Bars, numbers, the green
 top-image outline, and the layer grid (argmax per layer in blue) all update with the slider.</p>
 <div class="controls">
  <span id="count"></span>
  <label>topic <select id="ftopic"></select></label>
  <label>status <select id="fstatus"></select></label>
  <label>layer <input type="range" id="layer" min="-1" max="-1" value="-1">
   <span id="layerlab">mean</span></label>
 </div>
</header>
<div class="wrap" id="wrap"></div>
<div id="morebox"></div>
<script>
const CARDS = __DATA__;
const META = __META__;
const PAGE = 60;           // cards rendered per chunk (perf: avoid one giant DOM)
let shownCount = 0;        // how many of the current filtered set are in the DOM

// per-image normalized shares (sum~100) for a given layer; layerIdx<0 => across-layer mean.
function sharesFor(card, layerIdx){
  const a = card.attention; if(!a) return null;
  if(layerIdx<0 || !a.per_layer){           // across-layer mean (stored, normalized)
    return a.images.map(im=>im.avg_norm_pct);
  }
  const pl = a.per_layer[layerIdx]; if(!pl) return a.images.map(im=>im.avg_norm_pct);
  const raw = pl.avg_raw_pct; const tot = raw.reduce((s,x)=>s+x,0)||1e-9;
  return raw.map(x=>100*x/tot);
}
// raw (un-normalized) per-image pct for a layer, for the "raw%" number; null at mean.
function rawFor(card, layerIdx){
  const a=card.attention; if(!a||layerIdx<0||!a.per_layer) return null;
  const pl=a.per_layer[layerIdx]; return pl? pl.avg_raw_pct : null;
}
function maxLayers(){
  let m=0; for(const c of CARDS){ if(c.attention&&c.attention.per_layer) m=Math.max(m,c.attention.per_layer.length);} return m;
}
function filtered(){
  const topic=document.getElementById('ftopic').value;
  const status=document.getElementById('fstatus').value;
  return CARDS.filter(c=>(topic==='*'||c.topic===topic)&&(status==='*'||c.status===status));
}

function layerGrid(c, curLayer){
  // image rows x layer cols, argmax-per-layer cell highlighted; current layer column outlined.
  const a=c.attention; if(!a||!a.per_layer) return '';
  const nL=a.per_layer.length, nI=a.images.length;
  // argmax image per layer
  const arg=a.per_layer.map(pl=>{const r=pl.avg_raw_pct; let bi=0; for(let i=1;i<r.length;i++) if(r[i]>r[bi]) bi=i; return bi;});
  let h='<table class="grid"><tr><th>img</th>';
  for(let l=0;l<nL;l++) h+=`<th class="${l===curLayer?'cur':''}">${l}</th>`;
  h+='</tr>';
  for(let i=0;i<nI;i++){
    const lab=(a.images[i].role||('img'+(i+1))).replace('streetview_','');
    h+=`<tr><td class="rowh">${i+1} ${lab}</td>`;
    for(let l=0;l<nL;l++){
      const raw=a.per_layer[l].avg_raw_pct; const tot=raw.reduce((s,x)=>s+x,0)||1e-9;
      const share=100*raw[i]/tot;
      const cls=(arg[l]===i?'am ':'')+(l===curLayer?'cur':'');
      h+=`<td class="${cls.trim()}">${share<10?share.toFixed(0):Math.round(share)}</td>`;
    }
    h+='</tr>';
  }
  return h+'</table>';
}

function cardHTML(c, layerIdx){
  const shares = sharesFor(c, layerIdx);
  const raws = rawFor(c, layerIdx);
  const top = shares? shares.indexOf(Math.max(...shares)) : -1;
  const badge = c.status==='solved_greedy'?'b-greedy':c.status==='solved_sampled'?'b-sampled':'b-unstable';
  let html = `<div class="q">${c.question||''}</div>`+
    `<div class="meta"><span class="badge ${badge}">${c.status}</span>`+
    `${c.topic} · ${c.city||''} · gold <b>${c.gold}</b> · pred <b>${c.prediction||'-'}</b>`+
    (c.winning?` · t=${c.winning.temp} ${c.winning.variant} (n=${c.winning.n_attempts})`:'')+`</div>`;
  html += `<div class="imgs">`;
  c.images.forEach((im,i)=>{
    const isTop=(i===top);
    const sh = shares? shares[i].toFixed(1):'-';
    const rw = raws? (' · raw '+raws[i].toFixed(2)+'%') : '';
    html += `<div class="imw ${isTop?'topwin':''}">`+
      (im.src?`<img src="${im.src}">`:`<div class="role">[missing image]</div>`)+
      `<div class="role">${im.role}</div>`+
      `<div class="barlab"><span>attention${rw}</span><span>${sh}%</span></div>`+
      `<div class="bar"><span style="width:${shares?shares[i]:0}%"></span></div>`+
      `</div>`;
  });
  html += `</div>`;
  if(c.options){
    html += `<div class="opts">`+Object.keys(c.options).sort().map(k=>
      `<div class="${k===c.gold?'opt-gold':''}">${k}. ${c.options[k]}</div>`).join('')+`</div>`;
  }
  html += layerGrid(c, layerIdx);
  if(c.note) html += `<div class="meta">${c.note}</div>`;
  return html;
}

function paint(){
  // re-skin already-rendered cards in place when only the layer changes (cheap).
  const layerIdx=parseInt(document.getElementById('layer').value);
  document.getElementById('layerlab').textContent = layerIdx<0?'mean':('L'+layerIdx);
  const wrap=document.getElementById('wrap');
  const set=filtered();
  for(let i=0;i<wrap.children.length;i++){
    wrap.children[i].innerHTML = cardHTML(set[i], layerIdx);
  }
}

function rebuild(){
  // full rebuild when the filter set changes; renders first PAGE cards + a "load more".
  shownCount=0;
  document.getElementById('wrap').innerHTML='';
  appendChunk();
}
function appendChunk(){
  const layerIdx=parseInt(document.getElementById('layer').value);
  document.getElementById('layerlab').textContent = layerIdx<0?'mean':('L'+layerIdx);
  const set=filtered();
  const wrap=document.getElementById('wrap');
  const end=Math.min(shownCount+PAGE, set.length);
  for(let i=shownCount;i<end;i++){
    const div=document.createElement('div'); div.className='card';
    div.innerHTML=cardHTML(set[i], layerIdx); wrap.appendChild(div);
  }
  shownCount=end;
  document.getElementById('count').textContent = `${set.length} cards (showing ${shownCount})`;
  const mb=document.getElementById('morebox'); mb.innerHTML='';
  if(shownCount<set.length){
    const b=document.createElement('button'); b.className='more';
    b.textContent=`load more (${set.length-shownCount} left)`;
    b.onclick=appendChunk; mb.appendChild(b);
  }
}

function init(){
  const topics=[...new Set(CARDS.map(c=>c.topic))].sort();
  const statuses=[...new Set(CARDS.map(c=>c.status))].sort();
  const ft=document.getElementById('ftopic'); ft.innerHTML='<option value="*">all</option>'+topics.map(t=>`<option>${t}</option>`).join('');
  const fs=document.getElementById('fstatus'); fs.innerHTML='<option value="*">all</option>'+statuses.map(s=>`<option>${s}</option>`).join('');
  const ml=maxLayers(); const lr=document.getElementById('layer'); lr.max=String(ml-1);
  // filter changes => rebuild the set; layer changes => repaint in place (no DOM churn).
  document.getElementById('ftopic').addEventListener('input',rebuild);
  document.getElementById('fstatus').addEventListener('input',rebuild);
  document.getElementById('layer').addEventListener('input',paint);
  rebuild();
}
init();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", required=True, help="solved/<model> dir")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--status", default=None, help="filter to one status")
    args = ap.parse_args()

    solved_dir = Path(args.solved)
    if not solved_dir.is_absolute():
        solved_dir = (HERE / args.solved).resolve()
    meta = {}
    mp = solved_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    # image_root is the LOCAL tree to read pixels from; the recorded (lab-ws)
    # root lives in meta and is used only to remap path prefixes.
    image_root = args.image_root or str(REPO / "dataset_content/EODATA_compressed_final/benchmark")
    build(solved_dir, Path(image_root), args.limit, args.status)


if __name__ == "__main__":
    main()
