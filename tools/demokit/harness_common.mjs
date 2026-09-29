/* shared plumbing for geomtest.mjs / texttest.mjs:
   boots a bundle (out/dayNN.html.js) under a minimal DOM stub and hands back
   every recorded frame, tagged with tab / variant / frame index.            */
import fs from 'fs';

class El {
  constructor(tag){ this.tag = tag; this.children = []; this.attrs = {}; this._text = '';
    this._html = ''; this.style = {}; this.dataset = {};
    this.classList = {_s:new Set(), toggle:(c, on) => { on ? this.classList._s.add(c) : this.classList._s.delete(c); },
      add:c => this.classList._s.add(c), contains:c => this.classList._s.has(c)}; }
  get firstChild(){ return this.children[0] || null; }
  appendChild(c){ this.children.push(c); return c; }
  removeChild(c){ this.children = this.children.filter(x => x !== c); }
  setAttribute(k, v){ this.attrs[k] = String(v); }
  getAttribute(k){ return this.attrs[k]; }
  addEventListener(){}
  set textContent(v){ this._text = String(v); this.children = []; }
  get textContent(){ return this._text; }
  set innerHTML(v){ this._html = String(v); this.children = []; }
  get innerHTML(){ return this._html; }
  set className(v){ this._cls = v; }  get className(){ return this._cls || ''; }
}

export function boot(file){
  const ids = {};
  ['tabs', 'extra', 'stage', 'stage-title', 'legend', 'narr', 'panels', 'code', 'idea',
   'stepno', 'prev', 'next', 'play', 'speed', 'btn-zh', 'btn-en', 'state-card']
    .forEach(i => { ids[i] = new El('div'); ids[i].value = '800'; });
  global.document = {
    documentElement:new El('html'),
    getElementById:id => (ids[id] || (ids[id] = new El('div'))),
    createElement:t => new El(t), createElementNS:(ns, t) => new El(t),
    querySelectorAll:() => [], addEventListener:() => {}
  };
  global.setInterval = () => 1; global.clearInterval = () => {};
  const probe = '\n;module.exports = {TABS:TABS, setLang:setLang};';
  const mod = {exports:{}};
  new Function('module', fs.readFileSync(file, 'utf8') + probe)(mod);
  return mod.exports;
}

/* every frame of every tab x variant, with its effective view box */
export function allFrames(TABS){
  const out = [];
  TABS.forEach((tab, ti) => {
    const nv = Math.max(1, (tab.variants || []).length);
    for (let v = 0; v < nv; v++){
      const frames = tab.build(v) || [];
      frames.forEach((f, fi) => out.push({
        tab:tab, ti:ti, v:v, fi:fi, frame:f,
        view:f.view || tab.view || [10, 6.4]
      }));
    }
  });
  return out;
}

/* the text-width model the skill prescribes: CJK glyphs are square, Latin ~0.55 */
export const CJK = ch => {
  const c = ch.codePointAt(0);
  return (c >= 0x2e80 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff) ||
         (c >= 0xff00 && c <= 0xff60) || (c >= 0x3000 && c <= 0x303f);
};
export function textWidth(s, fs){
  let w = 0;
  for (const ch of String(s)) w += CJK(ch) ? 1.0 : 0.55;
  return w * fs;
}
/* both languages of a {zh,en} narration-style object */
export function variants(v){
  if (v == null) return [];
  if (typeof v === 'string') return [v];
  return [v.zh, v.en].filter(x => x != null).map(String);
}
export const where = r => r.tab.id + '/v' + r.v + '/f' + r.fi;
