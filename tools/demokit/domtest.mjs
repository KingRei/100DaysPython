import fs from 'fs';
// --- tiny DOM stub, enough to exercise every render path ---
class El {
  constructor(tag){ this.tag=tag; this.children=[]; this.attrs={}; this._text='';
    this._html=''; this.style={}; this.dataset={};
    this.classList={ _s:new Set(),
      toggle:(c,on)=>{ on?this.classList._s.add(c):this.classList._s.delete(c); },
      add:c=>this.classList._s.add(c), contains:c=>this.classList._s.has(c) }; }
  get firstChild(){ return this.children[0] || null; }
  appendChild(c){ this.children.push(c); return c; }
  removeChild(c){ this.children = this.children.filter(x=>x!==c); }
  setAttribute(k,v){ if(v===undefined||v===null||Number.isNaN(v)) throw new Error('bad attr '+k+'='+v); this.attrs[k]=String(v); }
  getAttribute(k){ return this.attrs[k]; }
  addEventListener(){}
  set textContent(v){ if(v===undefined) throw new Error('undefined textContent on '+this.tag); this._text=String(v); this.children=[]; }
  get textContent(){ return this._text; }
  set innerHTML(v){ if(v===undefined) throw new Error('undefined innerHTML on '+this.tag); this._html=String(v); this.children=[]; }
  get innerHTML(){ return this._html; }
  set className(v){ this._cls=v; }  get className(){ return this._cls||''; }
}
const ids = {};
const IDS = ['tabs','extra','stage','stage-title','legend','narr','panels','code','idea',
             'stepno','prev','next','play','speed','btn-zh','btn-en','state-card'];
IDS.forEach(i=>{ ids[i]=new El('div'); ids[i].value='800'; });
const i18nEls = ['title','sub','play','speed','state','code','idea'].map(k=>{
  const e=new El('span'); e.attrs['data-i18n']=k; return e; });
global.document = {
  documentElement:new El('html'),
  getElementById:id=>{ if(!ids[id]) throw new Error('missing #'+id); return ids[id]; },
  createElement:t=>new El(t), createElementNS:(ns,t)=>new El(t),
  querySelectorAll:sel=> sel==='[data-i18n]' ? i18nEls : [],
  addEventListener:()=>{}
};
global.setInterval=()=>1; global.clearInterval=()=>{};
const f = process.argv[2];
new Function(fs.readFileSync(f,'utf8') + '\n' + fs.readFileSync('drive.js','utf8'))();
