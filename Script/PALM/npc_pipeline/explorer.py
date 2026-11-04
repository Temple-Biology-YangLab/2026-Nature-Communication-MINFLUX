# Copyright (C) 2026 Wenlan Yu
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-contained interactive HTML explorer of points.csv."""
import json

import numpy as np


CENTRE_METHODS = [('geometric', 'Circular fit'),
                  ('locmofit', 'LocMoFit ring')]

SOURCE_ORDER = [('smap_locs', 'SMAP localizations', 'locs'),
                ('smap_traces', 'SMAP traces', 'traces'),
                ('gdsc_locs', 'GDSC localizations', 'locs'),
                ('gdsc_traces', 'GDSC traces', 'traces'),
                ('ref_nup96', 'reference (control)', 'ref')]

CONTROL_SOURCE = 'ref_nup96'

POINT_COLUMNS = ['dx_nm', 'dy_nm', 'rdx_nm', 'rdy_nm', 'locprec_nm',
                 'n_ref_locs', 'R_nm', 'order_ratio', 'pore']


def panel_specs(molecules):
    """The ten panels, labelled with the actual molecule names."""
    out = []
    for meth, mlabel in CENTRE_METHODS:
        for src, slabel, cls in SOURCE_ORDER:
            if cls == 'ref':
                label = f'{molecules.camr} {slabel}'
            else:
                label = f'{molecules.camt} {slabel}'
            out.append(dict(method=meth, source=src, label=label,
                            mlabel=mlabel, cls=cls,
                            control=(src == CONTROL_SOURCE)))
    return out


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:18px;
      background:#0d1117;color:#c9d1d9}
 h1{font-size:17px;margin:0 0 4px} .sub{color:#8b949e;font-size:12px;margin-bottom:14px}
 .warn{background:#2d2211;border:1px solid #d29922;color:#e3b341;padding:9px 12px;
       border-radius:6px;font-size:12px;margin-bottom:14px;line-height:1.45}
 .ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:12px;
      background:#161b22;padding:11px 14px;border-radius:8px;border:1px solid #30363d}
 .ctl.filters{border-color:#2ea043}
 label{font-size:12px;color:#8b949e;display:flex;gap:6px;align-items:center}
 .tag{font-size:11px;color:#2ea043;font-weight:600;margin-right:2px}
 select{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;
        border-radius:5px;padding:4px 7px;font-size:12px}
 input[type=range]{vertical-align:middle;width:96px}
 input[type=number]{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;
        border-radius:5px;padding:3px 5px;font-size:12px;width:60px;
        -moz-appearance:textfield}
 input[type=number]:focus{border-color:#58a6ff;outline:none}
 .rowlab{font-size:12px;font-weight:600;color:#8b949e;margin:14px 0 7px;
         letter-spacing:.04em;text-transform:uppercase}
 /* Fixed five columns. Scrolls sideways rather than reflowing, so the two
    method rows stay column-aligned at any window width. */
 .gridwrap{overflow-x:auto;padding-bottom:4px}
 .grid{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:12px;
       min-width:1240px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:11px}
 .card h3{margin:0 0 7px;font-size:12px;font-weight:600;line-height:1.35}
 .n{color:#8b949e;font-weight:400}
 canvas{width:100%;height:auto;background:#000;border-radius:5px;display:block}
 .ctrl{border-color:#d29922}
 .stat{font-size:11px;color:#8b949e;margin-top:5px;font-variant-numeric:tabular-nums}
 .foot{color:#484f58;font-size:11px;margin-top:16px;line-height:1.5}
 code{color:#8b949e}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">Each panel pools every pore, centred on its fitted centre.
Filters apply live to every panel. Counts follow the extent setting, so
<code>n</code> is what is actually drawn.</div>
<div class="warn"><b>Control panel (gold border):</b> __CAMR__ rotated by its own
measured phase. That is circular &mdash; the rotation is chosen to make the
8-fold add up, so it will look 8-fold even for random points. Treat it as a
mechanical check, not as evidence. The __CAMT__ panels take their angle from
__CAMR__ and are not circular.</div>
__REGNOTE__

<div class="ctl filters">
  <span class="tag">FILTERS</span>
  <label><b style="color:#58a6ff">__CAMR__</b> prec &le;
    <input id="precR" type="range" min="4" max="60" value="60" step="1">
    <input id="precR_n" type="number" min="4" max="60" value="60" step="1"> nm</label>
  <label><b style="color:#f0883e">__CAMT__ pts</b> prec &le;
    <input id="precP" type="range" min="4" max="60" value="60" step="1">
    <input id="precP_n" type="number" min="4" max="60" value="60" step="1"> nm</label>
  <label><b style="color:#3fb950">__CAMT__ trace</b> prec &le;
    <input id="precT" type="range" min="4" max="60" value="60" step="1">
    <input id="precT_n" type="number" min="4" max="60" value="60" step="1"> nm</label>
  <label>__CAMR__ locs / pore &ge;
    <input id="nref" type="range" min="0" max="60" value="0" step="1">
    <input id="nref_n" type="number" min="0" max="500" value="0" step="1"></label>
  <label>pore radius
    <input id="rmin" type="range" min="0" max="120" value="20" step="1">
    <input id="rmin_n" type="number" min="0" max="400" value="20" step="1">&ndash;
    <input id="rmax" type="range" min="0" max="200" value="120" step="1">
    <input id="rmax_n" type="number" min="0" max="400" value="120" step="1"> nm</label>
  <label>8-fold order/null &ge;
    <input id="ordr" type="range" min="0" max="3" value="0" step="0.1">
    <input id="ordr_n" type="number" min="0" max="10" value="0" step="0.1"></label>
</div>

<div class="ctl">
  <label>view <select id="mode"><option value="kde">density</option><option value="pts">points</option></select></label>
  <label>extent
    <input id="ext" type="range" min="10" max="200" value="100" step="5">
    <input id="ext_n" type="number" min="1" max="500" value="100" step="1"> nm</label>
  <label>smoothing
    <input id="bw" type="range" min="1" max="30" value="6" step="1">
    <input id="bw_n" type="number" min="0.5" max="60" value="6" step="0.5"> nm</label>
  <label>ring guide
    <input id="ring" type="range" min="0" max="120" value="__RING__" step="1">
    <input id="ring_n" type="number" min="0" max="300" value="__RING__" step="0.1"> nm</label>
  <label><input id="rot" type="checkbox"> 8-fold rotated</label>
</div>

<div id="rows"></div>

<div class="foot">__FOOT__</div>

<script>
// each point: [dx, dy, rdx, rdy, locprec, n_ref_locs, pore_R, order_ratio, pore]
const DATA = __DATA__;
const PANELS = __PANELS__;
const ROWS = __ROWS__;
const D=0,DY=1,RX=2,RY=3,PREC=4,NREF=5,PORER=6,ORD=7,PORE=8;

function keep(p, prec, nref, rmin, rmax, ordr){
  // NaN (shipped as null) passes every filter, so a source that lacks a value
  // is not silently emptied.
  if(p[PREC]!==null && p[PREC]>prec) return false;
  if(p[NREF]!==null && p[NREF]<nref) return false;
  if(p[PORER]!==null && (p[PORER]<rmin||p[PORER]>rmax)) return false;
  if(p[ORD]!==null && p[ORD]<ordr) return false;
  return true;
}

function draw(cv, pts, ext, bw, mode, ring, fold, rot, cmap, ptcol){
  const S=340; cv.width=S; cv.height=S;
  const g=cv.getContext('2d'); g.fillStyle='#000'; g.fillRect(0,0,S,S);
  const sc=S/(2*ext), cx=S/2, cy=S/2;
  const ax=rot?RX:D, ay=rot?RY:DY;
  if(!pts.length){g.fillStyle='#484f58';g.font='12px sans-serif';
    g.fillText('no data',S/2-24,S/2);return;}
  if(mode==='pts'){
    g.fillStyle=ptcol;
    for(const p of pts){const X=cx+p[ax]*sc,Y=cy-p[ay]*sc;
      if(X>=0&&X<S&&Y>=0&&Y<S){g.fillRect(X-0.7,Y-0.7,1.4,1.4);}}
  } else {
    const N=90, grid=new Float32Array(N*N), st=2*ext/N;
    for(const p of pts){const i=Math.floor((p[ax]+ext)/st), j=Math.floor((p[ay]+ext)/st);
      if(i>=0&&i<N&&j>=0&&j<N) grid[j*N+i]++;}
    const sig=Math.max(0.5,bw/st), rad=Math.ceil(sig*2.5);
    const k=[]; let ks=0;
    for(let d=-rad;d<=rad;d++){const v=Math.exp(-d*d/(2*sig*sig));k.push(v);ks+=v;}
    const tmp=new Float32Array(N*N), out=new Float32Array(N*N);
    for(let j=0;j<N;j++)for(let i=0;i<N;i++){let s=0;
      for(let d=-rad;d<=rad;d++){const ii=i+d; if(ii>=0&&ii<N) s+=grid[j*N+ii]*k[d+rad];}
      tmp[j*N+i]=s/ks;}
    for(let j=0;j<N;j++)for(let i=0;i<N;i++){let s=0;
      for(let d=-rad;d<=rad;d++){const jj=j+d; if(jj>=0&&jj<N) s+=tmp[jj*N+i]*k[d+rad];}
      out[j*N+i]=s/ks;}
    let mx=0; for(const v of out) if(v>mx) mx=v;
    const img=g.createImageData(S,S);
    for(let Y=0;Y<S;Y++)for(let X=0;X<S;X++){
      const i=Math.floor(X/S*N), j=Math.floor((S-1-Y)/S*N);
      const t=mx>0?out[j*N+i]/mx:0;
      let r,gg,b;
      if(cmap==='ice'){            // reference: black -> blue -> cyan -> white
        b =Math.min(255,t*3*255);
        gg=Math.min(255,Math.max(0,(t-0.33)*3*255));
        r =Math.min(255,Math.max(0,(t-0.66)*3*255));
      } else {                      // target: black -> red -> orange -> white
        r =Math.min(255,t*3*255);
        gg=Math.min(255,Math.max(0,(t-0.33)*3*255));
        b =Math.min(255,Math.max(0,(t-0.66)*3*255));
      }
      const o=(Y*S+X)*4; img.data[o]=r; img.data[o+1]=gg; img.data[o+2]=b; img.data[o+3]=255;}
    g.putImageData(img,0,0);
  }
  if(ring>0){g.strokeStyle='rgba(88,166,255,0.85)';g.lineWidth=1;
    g.beginPath();g.arc(cx,cy,ring*sc,0,7);g.stroke();}
  if(rot){g.strokeStyle='rgba(210,153,34,0.30)';g.lineWidth=0.7;
    for(let a=0;a<fold;a++){const th=a*2*Math.PI/fold;g.beginPath();g.moveTo(cx,cy);
      g.lineTo(cx+Math.cos(th)*ext*sc,cy-Math.sin(th)*ext*sc);g.stroke();}}
}

const $ = id => document.getElementById(id);

/* Pair a slider with a number box. Either one drives the other; the number box
   is the authority for values outside the slider's convenient range, so the
   slider clamps but the typed value is kept. */
function link(id, onchange){
  const s=$(id), n=$(id+'_n');
  if(!s) return null;
  if(!n) { s.addEventListener('input', onchange); return ()=>+s.value; }
  let value = +s.value;
  s.addEventListener('input', ()=>{ value=+s.value; n.value=s.value; onchange(); });
  const commit = ()=>{
    const v = parseFloat(n.value);
    if(!isFinite(v)) return;
    value = v;
    s.value = Math.min(+s.max, Math.max(+s.min, v));   // clamp the slider only
    onchange();
  };
  n.addEventListener('input', commit);
  n.addEventListener('change', commit);
  return ()=>value;
}

/* Build the grid once; render() only updates pixels and captions. */
const cards = {};
function build(){
  const host=$('rows');
  host.innerHTML='';
  for(const row of ROWS){
    const lab=document.createElement('div');
    lab.className='rowlab'; lab.textContent=row.label;
    host.appendChild(lab);
    const wrap=document.createElement('div'); wrap.className='gridwrap';
    const grid=document.createElement('div'); grid.className='grid';
    for(const p of PANELS.filter(q=>q.method===row.method)){
      const d=document.createElement('div');
      d.className='card'+(p.control?' ctrl':'');
      d.innerHTML='<h3>'+p.label+' <span class="n">&mdash; '+p.mlabel+'</span></h3>';
      const cv=document.createElement('canvas'); d.appendChild(cv);
      const st=document.createElement('div'); st.className='stat';
      d.appendChild(st); grid.appendChild(d);
      cards[p.method+'|'+p.source]={canvas:cv, stat:st, spec:p};
    }
    wrap.appendChild(grid); host.appendChild(wrap);
  }
}

let get = {};
function render(){
  const ext=get.ext(), bw=get.bw(), ring=get.ring(),
        mode=$('mode').value, rot=$('rot').checked,
        nref=get.nref(), rmin=get.rmin(), rmax=get.rmax(), ordr=get.ordr();
  const PR={ref:get.precR(), locs:get.precP(), traces:get.precT()};
  const ax=rot?RX:D, ay=rot?RY:DY;

  for(const p of PANELS){
    const c=cards[p.method+'|'+p.source];
    const all=DATA[p.method+'|'+p.source]||[];
    const pts=all.filter(q=>keep(q,PR[p.cls],nref,rmin,rmax,ordr));
    // Counts describe what is DRAWN: inside the current extent, in the frame
    // currently displayed. Change the extent and these move.
    const inview=pts.filter(q=>Math.abs(q[ax])<=ext && Math.abs(q[ay])<=ext);
    const npore=new Set(inview.map(q=>q[PORE])).size;
    c.stat.textContent='n = '+inview.length.toLocaleString()+' / '
      +all.length.toLocaleString()+'   pores = '+npore.toLocaleString();
    draw(c.canvas,pts,ext,bw,mode,ring,8,rot,
         p.cls==='ref'?'ice':'fire',
         p.cls==='ref'?'rgba(88,166,255,0.5)'
           :(p.cls==='traces'?'rgba(63,185,80,0.5)':'rgba(240,136,62,0.5)'));
  }
}

build();
for(const id of ['precR','precP','precT','nref','rmin','rmax','ordr',
                 'ext','bw','ring']) get[id]=link(id, ()=>render());
$('mode').addEventListener('input', render);
$('rot').addEventListener('input', render);
render();
</script>
</body></html>
"""


def _clean(v):
    """JSON-safe number: NaN and inf become null, which passes every filter."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 2)


def build_html(df, out_path, molecules, max_pts=15000, seed=0,
               title=None, registration_note=None, extra_note=None):
    """Write the explorer."""
    rng = np.random.default_rng(seed)
    panels = panel_specs(molecules)
    data = {}
    sampled = []
    for p in panels:
        g = df[(df.centre_method == p['method']) & (df.source == p['source'])]
        n_full = len(g)
        if n_full > max_pts:
            g = g.iloc[rng.choice(n_full, max_pts, replace=False)]
            sampled.append(f"{p['method']}/{p['source']} "
                           f"({max_pts:,} of {n_full:,})")
        arr = g.reindex(columns=POINT_COLUMNS).to_numpy(float)
        data[f"{p['method']}|{p['source']}"] = [[_clean(a) for a in row]
                                                for row in arr]

    rows = [dict(method=m, label=f'{lab} — pore centres')
            for m, lab in CENTRE_METHODS]

    title = title or f'{molecules.camt} / {molecules.camr} NPC explorer'

    foot = [
        'Rows are pore-centre methods, columns are localization sources; the '
        'grid does not reflow so the two rows stay column-aligned.',
        'Counts read <b>n = points inside the current extent / points shipped '
        'for that panel</b>, and <b>pores</b> is the number of distinct pores '
        'contributing to what is in view.',
        'Every slider has a number box; typing a value outside the slider '
        'range still applies, the slider just clamps its handle.',
        'A missing value passes every filter by design, so a source with no '
        'precision column is not silently emptied.',
    ]
    if sampled:
        foot.append('Downsampled panels (seeded, reproducible): '
                    + '; '.join(sampled) + '.')
    if extra_note:
        foot.append(extra_note)

    regnote = ''
    if registration_note:
        regnote = ('<div class="warn" style="border-color:#388bfd;'
                   'background:#0d1d2d;color:#79c0ff">'
                   + registration_note + '</div>')

    html = (HTML
            .replace('__DATA__', json.dumps(data))
            .replace('__PANELS__', json.dumps(panels))
            .replace('__ROWS__', json.dumps(rows))
            .replace('__REGNOTE__', regnote)
            .replace('__FOOT__', '<br>'.join(foot))
            .replace('__RING__', f'{molecules.ref_radius_nm:g}')
            .replace('__CAMT__', molecules.camt)
            .replace('__CAMR__', molecules.camr)
            .replace('__TITLE__', title))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path
