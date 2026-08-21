// exercise every tab / variant / frame / language
let checked = 0, svgAttrs = 0;
for (const lang of ['zh','en']){
  setLang(lang);
  for (let t = 0; t < TABS.length; t++){
    tabIx = t;
    const nv = Math.max(1, (TABS[t].variants || []).length);
    for (let v = 0; v < nv; v++){
      varIx = v; load();
      if (!frames.length) throw new Error('no frames: tab '+t+' var '+v);
      for (let i = 0; i < frames.length; i++){
        cur = i; render();
        const f = frames[i];
        if (!tr(f.msg)) throw new Error('empty narration @'+t+'/'+v+'/'+i);
        if (typeof f.line !== 'number' || TABS[t].code[f.line] === undefined)
          throw new Error('bad code line '+f.line+' @'+t+'/'+v+'/'+i);
        const stage = document.getElementById('stage');
        stage.children.forEach(el => {
          Object.values(el.attrs).forEach(a => {
            svgAttrs++;
            if (/NaN|undefined/.test(a)) throw new Error('bad svg attr '+a);
          });
        });
        checked++;
      }
      console.log(lang, TABS[t].id, 'variant', v, '->', frames.length, 'frames');
    }
  }
}
console.log('OK: rendered', checked, 'frames,', svgAttrs, 'svg attributes clean');
