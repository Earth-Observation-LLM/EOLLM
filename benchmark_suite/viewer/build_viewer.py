#!/usr/bin/env python3
"""build_viewer.py — turn a model's results into a self-contained HTML viewer.

The point of the suite is to UNDERSTAND the model: for each benchmark record you
want to see the exact images it was shown, the option it chose vs the gold, and —
when ATTENTION WATCH ran — how much of its decision-moment attention fell on each
image, with a slider to scrub through layers.

This reads results/<key>/<mode>_predictions.jsonl (+ the attention block embedded
there) and emits results/<key>/viewer_<mode>.html with every image base64-inlined,
so the file is portable (open it anywhere, no server). Images are rebuilt from the
record via images.build_images — byte-identical to what the model saw — so the
predictions files stay small.

Usage:
    python viewer/build_viewer.py --results results/pc_lora_viewer
    python viewer/build_viewer.py --results results/pc_lora_viewer --mode full
    python viewer/build_viewer.py --results results/pc_lora_viewer --limit 200
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
REPO = SUITE.parent
sys.path.insert(0, str(SUITE))

import images as suite_images
import modes as suite_modes

THUMB_PX = 320  # embedded thumbnail longest edge (keeps the HTML reasonable)


def _img_b64(pil) -> str:
    im = pil.convert("RGB")
    im.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_meta(results_dir: Path) -> dict:
    mp = results_dir / "meta.json"
    return json.loads(mp.read_text()) if mp.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/<model_key> dir")
    ap.add_argument("--mode", default=None, help="ablation mode (default: first found)")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--data", default=None, help="benchmark jsonl (defaults from meta/config)")
    args = ap.parse_args()

    results_dir = Path(args.results)
    meta = _load_meta(results_dir)

    # locate the benchmark + image root (meta has them, else default local)
    default_bench = REPO / "dataset_content" / "EODATA_compressed_final" / "benchmark"
    data_path = Path(args.data) if args.data else (default_bench / "benchmark_with_answers.jsonl")
    image_root = default_bench
    max_edge = meta.get("image_max_edge", 512)

    # which mode
    modes_present = sorted(
        m for m in (p.name.split("_predictions.jsonl")[0]
                    for p in results_dir.glob("*_predictions.jsonl"))
        if m != "all")
    mode = args.mode or (modes_present[0] if modes_present else None)
    if not mode:
        raise SystemExit(f"no *_predictions.jsonl in {results_dir}")
    print(f"building viewer for {results_dir.name} mode={mode} (modes present: {modes_present})")

    # index the benchmark by question_id so we can rebuild the exact images
    bench = {}
    for line in open(data_path):
        if line.strip():
            rec = json.loads(line)
            bench[rec["question_id"]] = rec

    rows = [json.loads(l) for l in open(results_dir / f"{mode}_predictions.jsonl") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    cards = []
    for r in rows:
        qid = r["question_id"]
        rec = bench.get(qid)
        imgs_b64 = []
        if rec is not None:
            full = suite_images.build_images(rec, str(image_root), max_edge)
            sel = suite_modes.filter_images_for_mode(full, r["ablation_mode"])
            # align to the roles recorded at eval time (defensive)
            for s in sel:
                imgs_b64.append({"role": s["role"], "src": _img_b64(s["image"])})
        cards.append({
            "qid": qid, "topic": r["topic"], "difficulty": r.get("difficulty"),
            "city": r.get("city"), "question": r["question"], "options": r["options"],
            "gold": r["gold"], "prediction": r["prediction"], "correct": r["is_correct"],
            "raw": r.get("raw_response", ""), "prob_dict": r.get("prob_dict"),
            "images": imgs_b64,
            "attention": r.get("attention"),
        })

    html = render_html(results_dir.name, mode, meta, cards, modes_present)
    out_path = results_dir / f"viewer_{mode}.html"
    out_path.write_text(html)
    print(f"wrote {out_path}  ({len(cards)} records, {out_path.stat().st_size/1e6:.1f} MB)")


def render_html(model_key, mode, meta, cards, modes_present):
    data_json = json.dumps(cards, ensure_ascii=False)
    meta_json = json.dumps({"model": model_key, "mode": mode,
                            "display": meta.get("display"), "backend": meta.get("backend"),
                            "modes_present": modes_present}, ensure_ascii=False)
    return _TEMPLATE.replace("/*__DATA__*/", data_json).replace("/*__META__*/", meta_json)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>benchmark_suite viewer</title>
<style>
  :root{--ok:#1a7f37;--bad:#cf222e;--bg:#0d1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e;--bar:#388bfd;}
  *{box-sizing:border-box} body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
  header{padding:10px 16px;background:var(--panel);border-bottom:1px solid #30363d;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header b{font-size:15px} .mut{color:var(--mut)}
  .wrap{display:flex;height:calc(100vh - 52px)}
  .list{width:330px;overflow:auto;border-right:1px solid #30363d}
  .item{padding:8px 12px;border-bottom:1px solid #21262d;cursor:pointer}
  .item:hover{background:#1c2330} .item.sel{background:#222c3c}
  .badge{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
  .ok{background:var(--ok)} .bad{background:var(--bad)}
  .detail{flex:1;overflow:auto;padding:18px 24px}
  .imgs{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}
  .imgcard{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:8px;width:230px}
  .imgcard img{width:100%;border-radius:4px;display:block}
  .imgcard .role{font-size:12px;color:var(--mut);margin-top:6px;word-break:break-word}
  .attnrow{display:flex;align-items:center;gap:8px;margin:4px 0}
  .attnrow .lab{width:170px;font-size:12px;color:var(--mut);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .track{flex:1;background:#21262d;border-radius:4px;height:16px;position:relative;overflow:hidden}
  .fill{height:100%;background:var(--bar)}
  .pct{width:54px;font-size:12px;text-align:right}
  .opt{padding:3px 0} .opt.gold{color:var(--ok);font-weight:600} .opt.pred{text-decoration:underline}
  .opt.predwrong{color:var(--bad)}
  .filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  select,input{background:#0d1117;color:var(--fg);border:1px solid #30363d;border-radius:6px;padding:4px 6px}
  .kpi{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:6px 10px}
  h2{margin:6px 0} code{background:#21262d;padding:1px 5px;border-radius:4px}
  .slider{display:flex;align-items:center;gap:10px;margin:10px 0}
</style></head>
<body>
<header>
  <b id="title">viewer</b>
  <span class="mut" id="subtitle"></span>
  <span class="filters">
    <label>topic <select id="ftopic"><option value="">all</option></select></label>
    <label><input type="checkbox" id="fwrong"> only wrong</label>
    <span class="kpi" id="acc"></span>
  </span>
</header>
<div class="wrap">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><p class="mut">Select a record on the left.</p></div>
</div>
<script>
const DATA = /*__DATA__*/;
const META = /*__META__*/;
let curLayer = "agg";        // "agg" = mean over layers, or a layer index
let selected = 0;

document.getElementById('title').textContent = (META.display||META.model) + "  ·  mode=" + META.mode;
document.getElementById('subtitle').textContent =
  `${DATA.length} records · backend=${META.backend||'?'} · modes: ${(META.modes_present||[]).join(', ')}`;

// topic filter options
const topics = [...new Set(DATA.map(d=>d.topic))].sort();
const ft = document.getElementById('ftopic');
topics.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;ft.appendChild(o);});

function filtered(){
  const t = ft.value, wrongOnly = document.getElementById('fwrong').checked;
  return DATA.map((d,i)=>({d,i})).filter(({d})=>
    (!t || d.topic===t) && (!wrongOnly || d.correct===false));
}

function renderList(){
  const items = filtered();
  const acc = (()=>{const v=DATA.filter(d=>d.gold);const c=v.filter(d=>d.correct).length;
    return v.length? (100*c/v.length).toFixed(1)+'% ('+c+'/'+v.length+')':'—';})();
  document.getElementById('acc').textContent = "acc " + acc;
  const L = document.getElementById('list'); L.innerHTML='';
  items.forEach(({d,i})=>{
    const div=document.createElement('div');
    div.className='item'+(i===selected?' sel':'');
    div.innerHTML=`<span class="badge ${d.correct?'ok':'bad'}"></span>`+
      `<b>${d.gold||'?'}</b>→<b>${d.prediction||'?'}</b> `+
      `<span class="mut">${d.topic}</span><br><span class="mut" style="font-size:12px">${d.qid}</span>`;
    div.onclick=()=>{selected=i;renderList();renderDetail();};
    L.appendChild(div);
  });
}

function attnForLayer(att){
  // returns [{role, avg, choice}] using curLayer (agg = stored image-level, else per_layer)
  if(!att || !att.images) return null;
  if(curLayer==="agg"){
    return att.images.map(im=>({role:im.role, avg:im.avg_norm_pct, choice:im.choice_norm_pct,
                                avgraw:im.avg_raw_pct, choiceraw:im.choice_raw_pct}));
  }
  const pl = (att.per_layer||[]).find(p=>String(p.layer)===String(curLayer));
  if(!pl) return null;
  const norm = a=>{const s=a.reduce((x,y)=>x+y,0)||1e-9; return a.map(v=>100*v/s);};
  const an = norm(pl.avg_raw_pct), cn = norm(pl.choice_raw_pct);
  return att.images.map((im,k)=>({role:im.role, avg:an[k], choice:cn[k],
                                  avgraw:pl.avg_raw_pct[k], choiceraw:pl.choice_raw_pct[k]}));
}

function bar(label, pct){
  return `<div class="attnrow"><div class="lab" title="${label}">${label}</div>`+
    `<div class="track"><div class="fill" style="width:${Math.min(100,pct).toFixed(1)}%"></div></div>`+
    `<div class="pct">${pct.toFixed(1)}%</div></div>`;
}

function renderDetail(){
  const d = DATA[selected]; if(!d){return;}
  const D=document.getElementById('detail');
  let h = `<h2>${d.topic} <span class="mut">${d.qid}</span></h2>`;
  h += `<div class="mut">${d.city||''} · ${d.difficulty||''}</div>`;
  h += `<p>${d.question}</p>`;
  // options
  h += '<div>';
  for(const [k,v] of Object.entries(d.options||{})){
    let cls='opt';
    if(k===d.gold) cls+=' gold';
    if(k===d.prediction) cls += (d.correct?' pred':' predwrong pred');
    h += `<div class="${cls}">${k}. ${v}${k===d.gold?'  ✓ gold':''}${k===d.prediction?'  ← model':''}</div>`;
  }
  h += '</div>';
  if(d.prob_dict){
    h += '<p class="mut">P(letter): '+Object.entries(d.prob_dict)
      .map(([k,v])=>`${k} ${(100*v).toFixed(0)}%`).join('  ')+'</p>';
  }
  // images
  h += '<div class="imgs">';
  (d.images||[]).forEach((im,k)=>{
    h += `<div class="imgcard"><img src="${im.src}"><div class="role">Image ${k+1}: ${im.role}</div></div>`;
  });
  h += '</div>';
  // attention
  const att = d.attention;
  if(att && att.images){
    const layers = (att.per_layer||[]).map(p=>p.layer);
    h += `<h2>Attention <span class="mut">(${att.image_attention_total_avg_pct?.toFixed?.(1) ?? att.image_attention_total_avg_pct}% of attention on images; rest on text)</span></h2>`;
    h += `<div class="mut" style="font-size:12px">spans match images: ${att.spans_match_images} · row-softmax: ${att.row_softmax_check} · layers stored: ${layers.length}</div>`;
    if(layers.length){
      h += `<div class="slider"><label>layer:</label>`+
        `<select id="layersel"><option value="agg">mean (all)</option>`+
        layers.map(l=>`<option value="${l}">${l}</option>`).join('')+`</select>`+
        `<span class="mut">choosing image share per image (normalized among images)</span></div>`;
    }
    const a = attnForLayer(att);
    if(a){
      h += '<div class="mut" style="font-size:12px;margin-top:6px">decision-moment (choice token):</div>';
      a.forEach((x,k)=> h += bar(`Image ${k+1}: ${x.role}`, x.choice));
      h += '<div class="mut" style="font-size:12px;margin-top:8px">averaged over answer:</div>';
      a.forEach((x,k)=> h += bar(`Image ${k+1}: ${x.role}`, x.avg));
    }
  } else if(att && att.note){
    h += `<p class="mut">attention: ${att.note}</p>`;
  }
  h += `<p class="mut">raw model output: <code>${(d.raw||'').replace(/</g,'&lt;')}</code></p>`;
  D.innerHTML = h;
  const ls = document.getElementById('layersel');
  if(ls){ ls.value=curLayer; ls.onchange=()=>{curLayer=ls.value;renderDetail();}; }
}

ft.onchange=()=>{renderList();};
document.getElementById('fwrong').onchange=()=>{renderList();};
renderList(); renderDetail();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
