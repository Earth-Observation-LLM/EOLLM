"""app_shell.py — the served front-end (no inlined data; fetches on demand).

Kept separate from serve.py so the HTML/JS blob doesn't drown the server logic.
The page loads ALL records' metadata once (small), renders the full filterable
list, and pulls a record's images + attention only when it is clicked.
"""

APP_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>benchmark_suite · attention viewer</title>
<style>
  :root{--ok:#1a7f37;--bad:#cf222e;--bg:#0d1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e;--bar:#388bfd;--sat:#d29922;--sv:#388bfd;}
  *{box-sizing:border-box} body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
  header{padding:9px 16px;background:var(--panel);border-bottom:1px solid #30363d;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  header b{font-size:15px}.mut{color:var(--mut)}
  .wrap{display:flex;height:calc(100vh - 50px)}
  .list{width:320px;overflow:auto;border-right:1px solid #30363d}
  .item{padding:7px 12px;border-bottom:1px solid #21262d;cursor:pointer;font-size:13px}
  .item:hover{background:#1c2330}.item.sel{background:#222c3c}
  .badge{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
  .ok{background:var(--ok)}.bad{background:var(--bad)}
  .detail{flex:1;overflow:auto;padding:16px 22px}
  .stats{width:300px;overflow:auto;border-left:1px solid #30363d;padding:12px 14px;background:#0f141b}
  .imgs{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
  .imgcard{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:7px;width:210px}
  .imgcard img{width:100%;border-radius:4px;display:block;min-height:80px;background:#21262d}
  .imgcard .role{font-size:12px;color:var(--mut);margin-top:5px;word-break:break-word}
  .attnrow{display:flex;align-items:center;gap:8px;margin:3px 0}
  .attnrow .lab{width:165px;font-size:12px;color:var(--mut);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .track{flex:1;background:#21262d;border-radius:4px;height:15px;overflow:hidden}
  .fill{height:100%;background:var(--bar)}.fill.sat{background:var(--sat)}.fill.sv{background:var(--sv)}
  .pct{width:50px;font-size:12px;text-align:right}
  .opt{padding:2px 0}.opt.gold{color:var(--ok);font-weight:600}.opt.pred{text-decoration:underline}.opt.predwrong{color:var(--bad)}
  select,input{background:#0d1117;color:var(--fg);border:1px solid #30363d;border-radius:6px;padding:3px 6px}
  .kpi{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:5px 9px;font-size:13px}
  h2{margin:8px 0 4px}h3{margin:12px 0 4px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
  code{background:#21262d;padding:1px 5px;border-radius:4px}
  .split{display:flex;height:18px;border-radius:4px;overflow:hidden;margin:4px 0;border:1px solid #30363d}
  .split .s{height:100%}.split .ssat{background:var(--sat)}.split .ssv{background:var(--sv)}
  .row2{display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}
  table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:2px 4px;text-align:right}th:first-child,td:first-child{text-align:left}
  .legend{font-size:11px}.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px}
  .insight{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:7px 9px;margin:5px 0;display:flex;gap:9px;align-items:center}
  .insight .big{font-size:21px;font-weight:700;color:var(--bar);min-width:48px;text-align:right}
  .insight .ilab{font-size:11px;line-height:1.3}
  .optgrid{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
  .optbox{border:1px solid #30363d;border-radius:6px;padding:4px 8px;font-size:12px}
  .optbox.gold{border-color:var(--ok)}.optbox.pick{box-shadow:0 0 0 2px var(--bar) inset}
</style></head>
<body>
<header>
  <b id="title">attention viewer</b>
  <span class="filters">
    <label>topic <select id="ftopic"><option value="">all</option></select></label>
    <label>mode <select id="fmode"><option value="">all</option></select></label>
    <label><input type="checkbox" id="fwrong"> only wrong</label>
    <span class="kpi" id="acc"></span>
    <span class="mut" id="count"></span>
  </span>
</header>
<div class="wrap">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><p class="mut">Loading…</p></div>
  <div class="stats" id="stats"></div>
</div>
<script>
let RECORDS=[], META={}, curLayer="agg", selected=null;
const $=id=>document.getElementById(id);

function bucket(role){const r=role.toLowerCase();
  if(r.includes('satellite')||r.includes('sat_')) return 'sat';
  if(r.includes('streetview')||r.includes('stv')) return 'sv'; return 'other';}

async function boot(){
  META = await (await fetch('/api/meta')).json();
  RECORDS = await (await fetch('/api/records')).json();
  $('title').textContent = (META.display||META.model)+'  ·  mode='+META.mode+'  ·  '+META.n_records+' records';
  // filters
  const topics=[...new Set(RECORDS.map(r=>r.topic))].sort();
  topics.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;$('ftopic').appendChild(o);});
  const modes=[...new Set(RECORDS.map(r=>r.image_mode))].sort();
  modes.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;$('fmode').appendChild(o);});
  $('ftopic').onchange=$('fmode').onchange=$('fwrong').onchange=renderList;
  renderStats();
  renderList();
  if(RECORDS.length) select(filtered()[0]?.i ?? 0);
}

function filtered(){
  const t=$('ftopic').value, m=$('fmode').value, w=$('fwrong').checked;
  return RECORDS.filter(r=>(!t||r.topic===t)&&(!m||r.image_mode===m)&&(!w||r.correct===false));
}

function renderList(){
  const items=filtered();
  const c=items.filter(r=>r.gold).length, ok=items.filter(r=>r.correct).length;
  $('acc').textContent='acc '+(c?(100*ok/c).toFixed(1)+'%':'—');
  $('count').textContent=items.length+' shown';
  const L=$('list');L.innerHTML='';
  const frag=document.createDocumentFragment();
  for(const r of items){
    const d=document.createElement('div');
    d.className='item'+(r.i===selected?' sel':'');
    d.innerHTML=`<span class="badge ${r.correct?'ok':'bad'}"></span><b>${r.gold||'?'}</b>→<b>${r.pred||'?'}</b> `+
      `<span class="mut">${r.topic}</span>`;
    d.onclick=()=>select(r.i);
    frag.appendChild(d);
  }
  L.appendChild(frag);
}

async function select(i){
  selected=i; renderList();
  $('detail').innerHTML='<p class="mut">loading…</p>';
  const d=await (await fetch('/api/record/'+i)).json();
  renderDetail(d);
}

function attnForLayer(att){
  if(!att||!att.images) return null;
  if(curLayer==="agg") return att.images.map(im=>({role:im.role,avg:im.avg_norm_pct,choice:im.choice_norm_pct}));
  const pl=(att.per_layer||[]).find(p=>String(p.layer)===String(curLayer));
  if(!pl) return null;
  const norm=a=>{const s=a.reduce((x,y)=>x+y,0)||1e-9;return a.map(v=>100*v/s);};
  const an=norm(pl.avg_raw_pct), cn=norm(pl.choice_raw_pct);
  return att.images.map((im,k)=>({role:im.role,avg:an[k],choice:cn[k]}));
}
function bar(label,pct,cls){return `<div class="attnrow"><div class="lab" title="${label}">${label}</div>`+
  `<div class="track"><div class="fill ${cls||''}" style="width:${Math.min(100,pct).toFixed(1)}%"></div></div>`+
  `<div class="pct">${pct.toFixed(1)}%</div></div>`;}

function renderDetail(d){
  let h=`<h2>${d.topic} <span class="mut">${d.qid}</span></h2>`;
  h+=`<div class="mut">${d.city||''} · ${d.difficulty||''}</div><p>${d.question}</p><div>`;
  for(const [k,v] of Object.entries(d.options||{})){
    let cls='opt'; if(k===d.gold)cls+=' gold'; if(k===d.prediction)cls+=(d.correct?' pred':' predwrong pred');
    h+=`<div class="${cls}">${k}. ${v}${k===d.gold?'  ✓ gold':''}${k===d.prediction?'  ← model':''}</div>`;}
  h+='</div>';
  if(d.prob_dict) h+='<p class="mut">P(letter): '+Object.entries(d.prob_dict).map(([k,v])=>`${k} ${(100*v).toFixed(0)}%`).join('  ')+'</p>';
  // images (lazy-loaded)
  h+='<div class="imgs">';
  (d.image_roles||[]).forEach((role,k)=>{
    h+=`<div class="imgcard"><img loading="lazy" src="/api/image/${d.i}/${k}"><div class="role">Image ${k+1}: ${role}</div></div>`;});
  h+='</div>';
  const att=d.attention;
  if(att&&att.images){
    const ex=d.extras||{};
    const layers=(att.per_layer||[]).map(p=>p.layer);
    h+=`<h2>Attention <span class="mut">(${(att.image_attention_total_avg_pct??0).toFixed?.(1)??att.image_attention_total_avg_pct}% on images; rest on text)</span></h2>`;
    if(ex.kind==='perspective' && ex.sat_share_pct!==undefined){
      h+=`<div class="row2"><span><span class="sw" style="background:var(--sat)"></span>satellite ${ex.sat_share_pct}%</span>`+
        `<span>street-view ${ex.sv_share_pct}% <span class="sw" style="background:var(--sv)"></span></span></div>`+
        `<div class="split"><div class="s ssat" style="width:${ex.sat_share_pct}%"></div><div class="s ssv" style="width:${ex.sv_share_pct}%"></div></div>`;
    }
    if(ex.kind==='option' && ex.option_attn){
      // attention on each ANSWER-OPTION image, with gold/pick highlight + verdict
      h+='<div class="mut" style="font-size:12px">attention on each option image:</div><div class="optgrid">';
      for(const [L,v] of Object.entries(ex.option_attn)){
        const cls=(L===ex.gold?'gold ':'')+(L===ex.prediction?'pick':'');
        h+=`<div class="optbox ${cls}">${L}: ${v}%${L===ex.gold?' ✓':''}${L===ex.prediction?' ←pick':''}</div>`;}
      h+='</div>';
      const verdict = ex.looked_at_gold && !d.correct ?
          '⚠ looked MOST at the gold option but still chose wrong — decision failure, not perception' :
          (ex.argmax_option===d.prediction ? 'attended most to the option it picked' :
           'attended most at option '+ex.argmax_option+', picked '+d.prediction);
      h+=`<div class="mut" style="font-size:12px">${verdict}</div>`;
    }
    h+=`<div class="mut" style="font-size:12px">fixation entropy ${ex.entropy} (0=one image,1=spread) · top: Image ${ex.top_image.image} ${ex.top_image.role} (${ex.top_image.choice_norm_pct.toFixed(1)}%)</div>`;
    h+=`<div class="mut" style="font-size:12px;margin:6px 0">spans match images: ${att.spans_match_images} · row-softmax: ${att.row_softmax_check}</div>`;
    if(layers.length) h+=`<div style="margin:8px 0"><label>layer: <select id="layersel"><option value="agg">mean (all)</option>`+
      layers.map(l=>`<option value="${l}">${l}</option>`).join('')+`</select></label></div>`;
    const a=attnForLayer(att);
    if(a){
      h+='<div class="mut" style="font-size:12px">decision-moment (choice token):</div>';
      a.forEach((x,k)=>h+=bar(`Image ${k+1}: ${x.role}`,x.choice,bucket(x.role)));
      h+='<div class="mut" style="font-size:12px;margin-top:8px">averaged over answer:</div>';
      a.forEach((x,k)=>h+=bar(`Image ${k+1}: ${x.role}`,x.avg,bucket(x.role)));
    }
  } else if(att&&att.note){h+=`<p class="mut">attention: ${att.note}</p>`;}
  h+=`<p class="mut">raw output: <code>${(d.raw||'').replace(/</g,'&lt;')}</code></p>`;
  $('detail').innerHTML=h;
  const ls=$('layersel'); if(ls){ls.value=curLayer;ls.onchange=()=>{curLayer=ls.value;select(d.i);};}
}

function renderStats(){
  const s=META.stats; if(!s||!s.n){$('stats').innerHTML='<p class="mut">no attention stats</p>';return;}
  let h=`<h3>insights (n=${s.n})</h3>`;
  // headline wisdom insights
  for(const i of (s.insights||[])){
    if(i.pct!==undefined && i.pct!==null){
      h+=`<div class="insight"><div class="big">${i.pct}%</div><div class="ilab">${i.label} <span class="mut">(${i.k}/${i.n})</span></div></div>`;
    } else if(i.correct!==undefined){
      h+=`<div class="insight"><div class="ilab"><b>${i.label}</b><br>correct ${i.correct} vs wrong ${i.wrong} <span class="mut">(n ${i.n_correct}/${i.n_wrong})</span></div></div>`;
    }
  }
  h+=`<div class="mut" style="font-size:11px;margin:6px 0">descriptive only — attention is not explanation; the causal companion is fusion.py (ablation synergy). extraction clean: ${s.hygiene.clean_pct}%</div>`;
  // sat vs sv (perspective)
  if(s.sat_vs_sv){const sv=s.sat_vs_sv;
    h+=`<h3>satellite vs street-view <span class="mut">(perspective modes, n=${sv.n})</span></h3>`+
      `<div class="legend"><span class="sw" style="background:var(--sat)"></span>satellite ${sv.satellite_pct}% `+
      `<span class="sw" style="background:var(--sv)"></span>street-view ${sv.streetview_pct}%</div>`+
      `<div class="split"><div class="s ssat" style="width:${sv.satellite_pct}%"></div><div class="s ssv" style="width:${sv.streetview_pct}%"></div></div>`;}
  h+=`<h3>image vs text</h3><div class="kpi">${s.image_vs_text.mean_image_total_pct}% of attention on images</div>`;
  // per-topic (kind-aware columns)
  h+='<h3>per-topic</h3><table><tr><th>topic</th><th>acc</th><th>sat/sv · gold-hit</th></tr>';
  for(const [t,d] of Object.entries(s.by_topic)){
    let col = d.kind==='perspective' ? `${d.sat_share}/${d.sv_share}` :
              d.kind==='option' ? `gold ${d.gold_option_attn_hit_pct??'–'}%` : '–';
    h+=`<tr><td title="${t}">${t.replace(/_/g,' ').slice(0,15)}</td><td>${d.acc}</td><td>${col}</td></tr>`;}
  h+='</table>';
  // per-layer
  h+='<h3>image attn by layer</h3>';
  const mx=Math.max(...s.by_layer.map(l=>l.img_total_choice_pct),0.01);
  s.by_layer.forEach(l=>{h+=`<div class="attnrow"><div class="lab">L${l.layer}</div><div class="track"><div class="fill" style="width:${100*l.img_total_choice_pct/mx}%"></div></div><div class="pct">${l.img_total_choice_pct}</div></div>`;});
  $('stats').innerHTML=h;
}

boot();
</script>
</body></html>
"""
