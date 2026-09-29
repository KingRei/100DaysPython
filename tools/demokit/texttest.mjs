/* text-width harness - models glyph width as fs * (1.0 per CJK char +
   0.55 per Latin char) and checks, in BOTH languages:
     - captions / free text (s.t === 't') fit inside the frame's view, honouring anchor
     - labels inside a box (s.t === 'r' / 'c') fit inside that box
     - sub / top annotations fit inside the view
     - nothing is rendered below a readable font size
   usage: node texttest.mjs out/day38.html.js [--minfs=0.18]                  */
import {boot, allFrames, textWidth, variants, where} from './harness_common.mjs';

const file = process.argv[2];
const minfs = Number((process.argv.find(a => a.startsWith('--minfs=')) || '--minfs=0.18').slice(8));
const {TABS} = boot(file);
const rows = allFrames(TABS);

const bad = [], tiny = [];
let texts = 0, labels = 0, smallest = 9;
const add = (r, msg) => bad.push(where(r) + '  ' + msg);

function inView(r, tag, x, fs, anchor, t){
  const w = textWidth(t, fs), W = r.view[0];
  const x0 = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2;
  if (x0 < -0.02 || x0 + w > W + 0.02)
    add(r, tag + ' ' + JSON.stringify(t.slice(0, 34)) + ' width ' + w.toFixed(2) +
           ' spans ' + x0.toFixed(2) + '..' + (x0 + w).toFixed(2) + ' (view 0..' + W + ')');
}
for (const r of rows){
  for (const s of (r.frame.shapes || [])){
    const k = s.t;
    if (k === 't' && s.s !== undefined){
      const fs = s.fs || .32;
      variants(s.s).forEach(t => { if (!t) return; texts++; smallest = Math.min(smallest, fs);
        if (fs < minfs) tiny.push(where(r) + '  text ' + JSON.stringify(t.slice(0, 30)) + ' fs ' + fs.toFixed(3));
        inView(r, 'text', s.x, fs, s.anchor || 'middle', t); });
    } else if ((k === 'r' || k === 'c') && s.lab != null && s.lab !== ''){
      const fs = s.fs || .40;
      // rects keep a little inner padding; circles may use the full chord
      const box = k === 'r' ? s.w - .06 : 2 * s.r;
      variants(s.lab).forEach(t => { if (!t) return; labels++; smallest = Math.min(smallest, fs);
        const w = textWidth(t, fs);
        if (w > box)
          add(r, (k === 'r' ? 'box' : 'circle') + ' label ' + JSON.stringify(t) +
                 ' width ' + w.toFixed(2) + ' > usable ' + box.toFixed(2));
      });
    }
    if ((k === 'r' || k === 'c') && s.sub != null){
      const cx = k === 'r' ? s.x + s.w / 2 : s.x;
      variants(s.sub).forEach(t => inView(r, 'sub', cx, s.subfs || .28, 'middle', t));
    }
    if ((k === 'r' || k === 'c') && s.top != null){
      const cx = k === 'r' ? s.x + s.w / 2 : s.x;
      variants(s.top).forEach(t => inView(r, 'top', cx, s.topfs || .28, 'middle', t));
    }
    if (k === 'e' && s.lab != null){
      const cx = (s.x1 + s.x2) / 2 + (s.lx || 0);
      variants(s.lab).forEach(t => inView(r, 'edge-label', cx, s.fs || .30, 'middle', t));
    }
  }
}
console.log('text: ' + rows.length + ' frames, ' + texts + ' strings, ' + labels +
            ' in-box labels, smallest font ' + smallest.toFixed(3));
if (texts + labels === 0){ console.log('!! harness read nothing - check s.t'); process.exit(2); }
if (tiny.length){
  console.log(tiny.length + ' below fs ' + minfs + ':');
  [...new Set(tiny)].slice(0, 20).forEach(t => console.log('  ' + t));
}
if (!bad.length && !tiny.length){ console.log('OK: every string fits'); process.exit(0); }
const uniq = [...new Set(bad)];
console.log(uniq.length + ' overflow findings (' + bad.length + ' occurrences):');
uniq.slice(0, 500).forEach(b => console.log('  ' + b));
process.exit(bad.length ? 1 : 0);
