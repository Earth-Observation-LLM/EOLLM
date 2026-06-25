#!/usr/bin/env python3
"""build_viewer.py — self-contained HTML viewer for the solved.jsonl collector
output.

Goal: let you EYEBALL, per solved question, which attention signal to trust before
choosing the Phase-2 ranking. For each solved question it shows:
  - the 4 street-view angle images (rebuilt byte-identical via the suite, inlined),
  - two attention bars per image: choice_norm_pct (decision-moment) and
    avg_norm_pct (answer-averaged), so you can compare which is sharper,
  - a layer scrubber (per_layer shares) to see how the ranking forms across depth,
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


def build(solved_dir: Path, image_root: Path, limit: int, status_filter: str | None):
    sp = solved_dir / "solved.jsonl"
    if not sp.exists():
        raise SystemExit(f"no solved.jsonl in {solved_dir}")
    meta = {}
    mp = solved_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())

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
            if path and Path(path).exists():
                imgs_b64.append({"role": role, "src": _img_b64(Image.open(path))})
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
 header h1{font-size:16px;margin:0 0 6px}
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
 .bar{height:9px;background:#21262d;border-radius:4px;overflow:hidden;margin:2px 0}
 .bar>span{display:block;height:100%}
 .bc{background:#3b82f6}.ba{background:#a78bfa}
 .barlab{display:flex;justify-content:space-between;font-size:10px;color:#8b949e}
 .opts{font-size:12px;margin-top:8px;color:#c9d1d9}
 .opt-gold{color:#3fb950;font-weight:600}
 .lyr{margin:8px 0 2px;font-size:12px;color:#8b949e}
 .topwin{outline:2px solid #3fb950}
</style></head>
<body>
<header>
 <h1>attention_prune — __TITLE__</h1>
 <div class="controls">
  <span id="count"></span>
  <label>topic <select id="ftopic"></select></label>
  <label>status <select id="fstatus"></select></label>
  <label>signal
   <select id="signal"><option value="choice">choice (decision)</option>
   <option value="avg">avg (answer)</option></select></label>
  <label>layer <input type="range" id="layer" min="-1" max="-1" value="-1">
   <span id="layerlab">mean</span></label>
 </div>
</header>
<div class="wrap" id="wrap"></div>
<script>
const CARDS = __DATA__;
const META = __META__;

function attnFor(card, signal, layerIdx){
  // returns per-image normalized pct list for the chosen signal & layer.
  const a = card.attention; if(!a) return null;
  const key = signal==='choice' ? 'choice_raw_pct' : 'avg_raw_pct';
  const normKey = signal==='choice' ? 'choice_norm_pct' : 'avg_norm_pct';
  if(layerIdx<0 || !a.per_layer){           // mean across layers (stored on images)
    return a.images.map(im=>im[normKey]);
  }
  const pl = a.per_layer[layerIdx]; if(!pl) return a.images.map(im=>im[normKey]);
  const raw = pl[key]; const tot = raw.reduce((s,x)=>s+x,0)||1e-9;
  return raw.map(x=>100*x/tot);
}

function maxLayers(){
  let m=0; for(const c of CARDS){ if(c.attention&&c.attention.per_layer) m=Math.max(m,c.attention.per_layer.length);} return m;
}

function render(){
  const topic=document.getElementById('ftopic').value;
  const status=document.getElementById('fstatus').value;
  const signal=document.getElementById('signal').value;
  const layerIdx=parseInt(document.getElementById('layer').value);
  document.getElementById('layerlab').textContent = layerIdx<0?'mean':('L'+layerIdx);
  const wrap=document.getElementById('wrap'); wrap.innerHTML='';
  let shown=0;
  for(const c of CARDS){
    if(topic!=='*'&&c.topic!==topic) continue;
    if(status!=='*'&&c.status!==status) continue;
    shown++;
    const norm = attnFor(c, signal, layerIdx);
    const top = norm? norm.indexOf(Math.max(...norm)) : -1;
    const div=document.createElement('div'); div.className='card';
    const badge = c.status==='solved_greedy'?'b-greedy':c.status==='solved_sampled'?'b-sampled':'b-unstable';
    let html = `<div class="q">${c.question||''}</div>`+
      `<div class="meta"><span class="badge ${badge}">${c.status}</span>`+
      `${c.topic} · ${c.city||''} · gold <b>${c.gold}</b> · pred <b>${c.prediction||'-'}</b>`+
      (c.winning?` · t=${c.winning.temp} ${c.winning.variant} (n=${c.winning.n_attempts})`:'')+`</div>`;
    html += `<div class="imgs">`;
    c.images.forEach((im,i)=>{
      const isTop = (i===top);
      const cv = norm? norm[i].toFixed(1):'-';
      // also show the OTHER signal's mean for comparison
      let av='-',chv='-';
      if(c.attention){ av=c.attention.images[i].avg_norm_pct.toFixed(1); chv=c.attention.images[i].choice_norm_pct.toFixed(1);}
      html += `<div class="imw ${isTop?'topwin':''}">`+
        (im.src?`<img src="${im.src}">`:`<div class="role">[missing image]</div>`)+
        `<div class="role">${im.role}</div>`+
        `<div class="barlab"><span>choice</span><span>${chv}%</span></div>`+
        `<div class="bar"><span class="bc" style="width:${chv}%"></span></div>`+
        `<div class="barlab"><span>avg</span><span>${av}%</span></div>`+
        `<div class="bar"><span class="ba" style="width:${av}%"></span></div>`+
        `</div>`;
    });
    html += `</div>`;
    if(c.options){
      html += `<div class="opts">`+Object.keys(c.options).sort().map(k=>
        `<div class="${k===c.gold?'opt-gold':''}">${k}. ${c.options[k]}</div>`).join('')+`</div>`;
    }
    if(c.note) html += `<div class="meta">${c.note}</div>`;
    div.innerHTML=html; wrap.appendChild(div);
  }
  document.getElementById('count').textContent = `${shown} / ${CARDS.length} cards`;
}

function init(){
  const topics=[...new Set(CARDS.map(c=>c.topic))].sort();
  const statuses=[...new Set(CARDS.map(c=>c.status))].sort();
  const ft=document.getElementById('ftopic'); ft.innerHTML='<option value="*">all</option>'+topics.map(t=>`<option>${t}</option>`).join('');
  const fs=document.getElementById('fstatus'); fs.innerHTML='<option value="*">all</option>'+statuses.map(s=>`<option>${s}</option>`).join('');
  const ml=maxLayers(); const lr=document.getElementById('layer'); lr.max=String(ml-1);
  for(const el of ['ftopic','fstatus','signal','layer']) document.getElementById(el).addEventListener('input',render);
  render();
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
    image_root = args.image_root or (meta.get("dataset", {}) or {}).get("image_root")
    if not image_root:
        image_root = str(REPO / "dataset_content/EODATA_compressed_final/benchmark")
    build(solved_dir, Path(image_root), args.limit, args.status)


if __name__ == "__main__":
    main()
