/* geometry harness - every shape of every frame must live inside its view box.
   Shapes key their kind on `s.t` ('r' rect, 'c' circle, 'e' edge, 't' text);
   `s.t` is ALSO the text payload for text shapes, so read s.t as the kind only
   when it is one of the four letters and the shape carries the matching fields.
   usage: node geomtest.mjs out/day38.html.js [--tol 0.02]                    */
import {boot, allFrames, textWidth, variants, where} from './harness_common.mjs';

const file = process.argv[2];
const tol = Number((process.argv.find(a => a.startsWith('--tol=')) || '--tol=0.02').slice(6));
const {TABS} = boot(file);
const rows = allFrames(TABS);

let shapes = 0, kinds = {r:0, c:0, e:0, t:0, '?':0};
const bad = [];
function chk(r, tag, x0, y0, x1, y1){
  const [W, H] = r.view;
  const over = [];
  if (x0 < -tol) over.push('left ' + x0.toFixed(2));
  if (y0 < -tol) over.push('top ' + y0.toFixed(2));
  if (x1 > W + tol) over.push('right ' + x1.toFixed(2) + ' > ' + W);
  if (y1 > H + tol) over.push('bottom ' + y1.toFixed(2) + ' > ' + H);
  if (over.length) bad.push(where(r) + '  ' + tag + '  [' + over.join(', ') + ']');
}
for (const r of rows){
  for (const s of (r.frame.shapes || [])){
    shapes++;
    const k = s.t;
    if (k === 'r' && s.w != null && s.h != null){
      kinds.r++;
      chk(r, 'rect ' + JSON.stringify(variants(s.lab)[0] || ''), s.x, s.y, s.x + s.w, s.y + s.h);
      const cx = s.x + s.w / 2;
      if (s.sub != null){
        const fs = s.subfs || .28, y = s.y + s.h / 2 + s.h + .34;
        variants(s.sub).forEach(t => chk(r, 'rect-sub ' + JSON.stringify(t),
          cx - textWidth(t, fs) / 2, y - fs, cx + textWidth(t, fs) / 2, y + fs * .25));
      }
      if (s.top != null){
        const fs = s.topfs || .28, y = s.y + s.h / 2 - s.h / 2 - .22;
        variants(s.top).forEach(t => chk(r, 'rect-top ' + JSON.stringify(t),
          cx - textWidth(t, fs) / 2, y - fs, cx + textWidth(t, fs) / 2, y + fs * .25));
      }
    } else if (k === 'c' && s.r != null){
      kinds.c++;
      chk(r, 'circle ' + JSON.stringify(variants(s.lab)[0] || ''), s.x - s.r, s.y - s.r, s.x + s.r, s.y + s.r);
      if (s.sub != null){
        const fs = s.subfs || .28, y = s.y + s.r + .34;
        variants(s.sub).forEach(t => chk(r, 'circle-sub ' + JSON.stringify(t),
          s.x - textWidth(t, fs) / 2, y - fs, s.x + textWidth(t, fs) / 2, y + fs * .25));
      }
      if (s.top != null){
        const fs = s.topfs || .28, y = s.y - s.r - .22;
        variants(s.top).forEach(t => chk(r, 'circle-top ' + JSON.stringify(t),
          s.x - textWidth(t, fs) / 2, y - fs, s.x + textWidth(t, fs) / 2, y + fs * .25));
      }
    } else if (k === 'e' && s.x1 != null){
      kinds.e++;
      chk(r, 'edge', Math.min(s.x1, s.x2), Math.min(s.y1, s.y2),
                     Math.max(s.x1, s.x2), Math.max(s.y1, s.y2));
    } else if (k === 't' || (s.x != null && s.y != null && s.s !== undefined)){
      kinds.t++;
      const fs = s.fs || .32, anchor = s.anchor || 'middle';
      variants(s.s).forEach(t => {
        const w = textWidth(t, fs);
        const x0 = anchor === 'start' ? s.x : anchor === 'end' ? s.x - w : s.x - w / 2;
        chk(r, 'text ' + JSON.stringify(t.slice(0, 28)), x0, s.y - fs, x0 + w, s.y + fs * .25);
      });
    } else kinds['?']++;
  }
}
console.log('geometry: ' + rows.length + ' frames, ' + shapes + ' shapes  ' + JSON.stringify(kinds));
if (kinds.r + kinds.c + kinds.e + kinds.t === 0){ console.log('!! harness read nothing - check s.t'); process.exit(2); }
if (kinds['?']) console.log('note: ' + kinds['?'] + ' shapes of unknown kind');
if (!bad.length){ console.log('OK: every shape inside its view (tol ' + tol + ')'); process.exit(0); }
const uniq = [...new Set(bad)];
console.log(uniq.length + ' out-of-bounds findings (' + bad.length + ' occurrences):');
uniq.slice(0, 500).forEach(b => console.log('  ' + b));
process.exit(1);
