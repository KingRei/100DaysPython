// DAY: 48
// TITLE_ZH: 決策樹、隨機森林與 Gradient Boosting：表格資料上最常用的預測模型怎麼運作
// TITLE_EN: Decision trees, random forests and gradient boosting: how the most widely used model on tabular data works
// SUB_ZH: 決策樹是這個系列裡唯一一個沒有係數也沒有距離函數的模型——它只是在問問題。所以它最好讀，也最好騙：長到底的樹在自己的資料上拿 1.000、impurity 算出來的特徵重要度會把一整欄純雜訊排第一、少掉一列就換一棵樹、而它永遠預測不出沒見過的數字。森林用獨立的深樹殺 variance，boosting 用相依的淺樹殺 bias，而 boosting 的訓練損失會一路往下，同時模型越來越差。
// SUB_EN: A decision tree is the only model in this series with no coefficients and no distance function - it asks questions. That makes it the easiest to read and the easiest to fool: grown to purity it scores 1.000 on its own data, impurity-based feature importance ranks a pure-noise column first, removing one row gives you a different tree, and it can never predict a number it has not seen. A forest kills variance with independent deep trees, boosting kills bias with dependent shallow ones - and boosting's training loss falls forever while the model gets worse.
// FOLDER: day%2048%20-%20decision%20trees%20random%20forests%20and%20boosting
// MEDIUM: https://medium.com/100-days-of-python
const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const curV = () => (typeof varIx !== 'undefined' ? varIx : 0);

const isCJK = ch => { const c = ch.codePointAt(0);
  return (c >= 0x2e80 && c <= 0x9fff) || (c >= 0xff00 && c <= 0xff60) || (c >= 0x3000 && c <= 0x303f); };
function tw(str, fs){ let w = 0;
  for (const ch of String(str)) w += isCJK(ch) ? 1.0 : .55;
  return w * fs; }
function fitT(x, y, s, opt){
  opt = Object.assign({}, opt || {});
  const fs = opt.fs || .32, a = opt.anchor || 'middle', pad = .20;
  const room = a === 'start' ? VIEW[0] - pad - x
             : a === 'end'   ? x - pad
             : 2 * Math.min(x - pad, VIEW[0] - pad - x);
  const strs = (typeof s === 'string') ? [s] : [s.zh, s.en].filter(t => t != null);
  const w = Math.max.apply(null, strs.map(t => tw(t, fs)));
  if (w > room) opt.fs = fs * room / w;
  return S.t(x, y, s, opt);
}
function fitFs(lab, w, fs){
  const strs = (typeof lab === 'string') ? [lab] : [lab.zh, lab.en];
  const need = Math.max.apply(null, strs.map(t => tw(t, fs)));
  return need > w - .08 ? fs * (w - .08) / need : fs;
}
const box = (x, y, w, h, s, lab, fs) => S.r(x, y, w, h, s, lab, {fs:fitFs(lab, w, fs || h * .5)});
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
const side = (sh, x, y0, lines, dy) => (lines || []).forEach((l, k) =>
  sh.push(fitT(x, y0 + k * (dy || .5), l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
const fmtN = x => !isFinite(x) ? (Number.isNaN(x) ? 'nan' : (x > 0 ? 'inf' : '-inf'))
  : String(Math.round(x * 1e3) / 1e3);

/* ================================================================ shared geometry */
// A plot box in view coordinates.  Data y grows upward, the SVG y grows downward,
// so every mapping flips it once - exactly once.
function Plot(x0, y0, w, h, lim){
  this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; this.lim = lim;
}
Plot.prototype.px = function(p){
  const [[ax, ay], [bx, by]] = this.lim, pad = .07;
  const ux = (p[0] - ax) / (bx - ax), uy = (p[1] - ay) / (by - ay);
  return [this.x0 + this.w * (pad + ux * (1 - 2 * pad)),
          this.y0 + this.h * (1 - pad - uy * (1 - 2 * pad))];
};
Plot.prototype.frame = function(sh, st){
  sh.push(S.r(this.x0, this.y0, this.w, this.h, st || 'ghost', null));
  return sh;
};
function dots(sh, plot, pts, styles, labs, r){
  pts.forEach((p, i) => {
    const q = plot.px(p);
    sh.push(S.c(q[0], q[1], r || .12, styles[i] || 'idle', labs ? labs[i] : null,
                labs ? {fs:.20} : {}));
  });
  return sh;
}
function crosses(sh, plot, pts, styles, labs, ghostAt){
  pts.forEach((p, i) => {
    const st = styles[i] || 'hot';
    if (!isFinite(p[0]) || !isFinite(p[1])){
      // nan has no position: draw it where it used to be, as a box that says so.
      const g = plot.px((ghostAt && ghostAt[i]) || [0, 0]);
      sh.push(box(g[0] - .42, g[1] - .22, .84, .44, 'bad', 'nan', .26));
      if (labs && labs[i]) sh.push(fitT(g[0], g[1] - .46, labs[i], {c:COL.red, fs:.22}));
      return;
    }
    const q = plot.px(p), d = .17;
    sh.push(S.e(q[0] - d, q[1] - d, q[0] + d, q[1] + d, {s:st, arrow:false, w:.05}));
    sh.push(S.e(q[0] - d, q[1] + d, q[0] + d, q[1] - d, {s:st, arrow:false, w:.05}));
    if (labs && labs[i]) sh.push(fitT(q[0] + .30, q[1] - .02, labs[i],
                                      {anchor:'start', c:COL.orange, fs:.24}));
  });
  return sh;
}
function bars(sh, x0, yBase, bw, gap, vals, styles, labs, maxH, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const h = Math.max(.04, maxH * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, yBase - h, bw, h, styles[i] || 'idle', null));
    if (labs && labs[i] != null)
      sh.push(fitT(x + bw / 2, h > .55 ? yBase - h + .26 : yBase - h - .22, labs[i],
                   {c:COL.pale, fs:.24}));
  });
  return sh;
}

/* ================================================================ the data */
// 24 hand-placed points.  Class 1 occupies the lower-right block, except for one
// row at (0.70, 0.30) that is labelled 0 - the whole of the "memorise one row" story.
const P24 = [
  [0.10, 0.15, 0], [0.18, 0.45, 0], [0.12, 0.72, 0], [0.26, 0.88, 0], [0.22, 0.28, 0],
  [0.08, 0.58, 0], [0.33, 0.18, 0], [0.36, 0.66, 0], [0.30, 0.40, 0],
  [0.62, 0.12, 1], [0.74, 0.22, 1], [0.88, 0.16, 1], [0.66, 0.34, 1], [0.82, 0.40, 1],
  [0.94, 0.28, 1], [0.58, 0.44, 1],
  [0.60, 0.66, 0], [0.72, 0.78, 0], [0.86, 0.70, 0], [0.92, 0.88, 0], [0.64, 0.92, 0],
  [0.78, 0.58, 0], [0.90, 0.56, 0],
  [0.70, 0.30, 0]
];
const LIM01 = [[0, 0], [1, 1]];
// 14 points of y = sin(4.2x) + 0.25x, rounded so that python and this page agree exactly
const X1D = [0.04, 0.11, 0.18, 0.25, 0.32, 0.39, 0.46, 0.53, 0.60, 0.67, 0.74, 0.81, 0.88, 0.95];
const Y1D = [0.177, 0.473, 0.731, 0.930, 1.054, 1.095, 1.050, 0.925, 0.732, 0.489,
             0.219, -0.055, -0.306, -0.513];

/* ================================================================ tiny tree code */
function giniOf(rows){
  if (!rows.length) return 0;
  let c1 = 0;
  rows.forEach(r => { c1 += r[2]; });
  const p = c1 / rows.length;
  return 1 - p * p - (1 - p) * (1 - p);
}
function majority(rows){
  let c1 = 0;
  rows.forEach(r => { c1 += r[2]; });
  return c1 * 2 >= rows.length ? (c1 * 2 === rows.length ? 0 : 1) : 0;
}
function bestSplit2D(rows){
  const par = giniOf(rows);
  let best = null;
  for (let j = 0; j < 2; j++){
    const vals = Array.from(new Set(rows.map(r => r[j]))).sort((a, b) => a - b);
    for (let i = 0; i + 1 < vals.length; i++){
      const t = (vals[i] + vals[i + 1]) / 2;
      const L = rows.filter(r => r[j] <= t), R = rows.filter(r => r[j] > t);
      if (!L.length || !R.length) continue;
      const child = (L.length * giniOf(L) + R.length * giniOf(R)) / rows.length;
      const g = par - child;
      if (!best || g > best.gain + 1e-12) best = {gain:g, feat:j, thr:t, L:L, R:R};
    }
  }
  return best;
}
// grow depth-first, recording one entry per split in the order it is made
function growRecorded(rows, maxDepth){
  const steps = [];
  const root = {rows:rows, box:[0, 0, 1, 1], depth:0, leaf:true, value:majority(rows)};
  const leaves = [root];
  (function rec(node){
    if (node.depth >= maxDepth || giniOf(node.rows) < 1e-12) return;
    const sp = bestSplit2D(node.rows);
    if (!sp || sp.gain <= 1e-12) return;
    const [bx, by, bw, bh] = node.box;
    const l = {rows:sp.L, depth:node.depth + 1, leaf:true, value:majority(sp.L),
               box:sp.feat === 0 ? [bx, by, sp.thr - bx, bh] : [bx, by, bw, sp.thr - by]};
    const r = {rows:sp.R, depth:node.depth + 1, leaf:true, value:majority(sp.R),
               box:sp.feat === 0 ? [sp.thr, by, bx + bw - sp.thr, bh]
                                 : [bx, sp.thr, bw, by + bh - sp.thr]};
    node.leaf = false; node.feat = sp.feat; node.thr = sp.thr; node.left = l; node.right = r;
    const k = leaves.indexOf(node);
    leaves.splice(k, 1, l, r);
    steps.push({feat:sp.feat, thr:sp.thr, gain:sp.gain, depth:node.depth,
                n:node.rows.length, leaves:leaves.map(q => ({box:q.box.slice(), value:q.value,
                                                            n:q.rows.length, pure:giniOf(q.rows) < 1e-12})),
                acc:accOf(leaves)});
    rec(l); rec(r);
  })(root);
  return {steps:steps, root:root};
}
function accOf(leaves){
  let ok = 0, n = 0;
  leaves.forEach(q => {
    q.rows.forEach(r => { if (r[2] === q.value) ok++; });
    n += q.rows.length;
  });
  return ok / n;
}
// a 1-D regression tree, used by the extrapolation and boosting tabs
function regTree(xs, ys, maxDepth){
  const mean = a => a.reduce((p, q) => p + q, 0) / a.length;
  const vr = a => { const m = mean(a); return mean(a.map(v => (v - m) * (v - m))); };
  function grow(idx, d){
    const yy = idx.map(i => ys[i]);
    if (d >= maxDepth || idx.length < 2 || vr(yy) < 1e-15) return {leaf:true, v:mean(yy)};
    const vals = Array.from(new Set(idx.map(i => xs[i]))).sort((a, b) => a - b);
    let best = null;
    for (let i = 0; i + 1 < vals.length; i++){
      const t = (vals[i] + vals[i + 1]) / 2;
      const L = idx.filter(k => xs[k] <= t), R = idx.filter(k => xs[k] > t);
      if (!L.length || !R.length) continue;
      const c = (L.length * vr(L.map(k => ys[k])) + R.length * vr(R.map(k => ys[k]))) / idx.length;
      const g = vr(yy) - c;
      if (!best || g > best.g + 1e-12) best = {g:g, t:t, L:L, R:R};
    }
    if (!best) return {leaf:true, v:mean(yy)};
    return {leaf:false, thr:best.t, left:grow(best.L, d + 1), right:grow(best.R, d + 1)};
  }
  const root = grow(xs.map((_, i) => i), 0);
  return x => { let nd = root; while (!nd.leaf) nd = x <= nd.thr ? nd.left : nd.right; return nd.v; };
}
const meanOf = a => a.reduce((p, q) => p + q, 0) / a.length;
const mseOf = (a, b) => meanOf(a.map((v, i) => (v - b[i]) * (v - b[i])));

/* ================================================================ tab 1: growing */
const CODE_GROW = [
  'def grow(X, y, depth):',
  '    node = Node(majority(y), gini(y))',
  '    if gini(y) == 0 or depth == max_depth:',
  '        return node                      # a leaf holds a constant',
  '    gain, j, t = best_split(X, y)        # every (column, threshold)',
  '    node.feat, node.thr = j, t',
  '    m = X[:, j] <= t',
  '    node.left  = grow(X[m],  y[m],  depth + 1)',
  '    node.right = grow(X[~m], y[~m], depth + 1)',
  '    return node'
];
const GROWN = growRecorded(P24, 6);
function growStage(st, o){
  const sh = [], plot = new Plot(.45, .55, 4.6, 4.5, LIM01);
  plot.frame(sh);
  (st ? st.leaves : [{box:[0, 0, 1, 1], value:0, n:24, pure:false}]).forEach(q => {
    const a = plot.px([q.box[0], q.box[1] + q.box[3]]);
    const b = plot.px([q.box[0] + q.box[2], q.box[1]]);
    sh.push(S.r(a[0], a[1], b[0] - a[0], b[1] - a[1],
                q.pure ? (q.value === 1 ? 'act' : 'ok') : 'idle', null));
  });
  dots(sh, plot, P24.map(p => [p[0], p[1]]),
       P24.map(p => (p[2] === 1 ? 'act' : 'ok')), null, .11);
  if (o.mark != null){
    const q = plot.px([P24[o.mark][0], P24[o.mark][1]]);
    sh.push(S.c(q[0], q[1], .26, 'bad', null));
  }
  side(sh, 5.35, 1.1, o.side, .46);
  return caps(sh, o);
}
function growFrames(){
  const F = new Frames();
  const total = GROWN.steps.length;
  F.push({
    shapes:growStage(null, {
      cap:{zh:'24 個點，兩個類別，一個還沒問任何問題的根節點',
           en:'24 points, two classes, and a root that has asked nothing yet'},
      cap2:{zh:'gini 0.4132，把每個點都猜成多數類的話準確率 0.7083',
            en:'gini 0.4132; calling everything the majority class scores 0.7083'},
      side:[[{zh:'葉子 1，準確率 0.7083', en:'1 leaf, accuracy 0.7083'}, COL.purpleL, .30],
            [{zh:'葉子裡放的是一個常數', en:'a leaf holds one constant'}, COL.grey, .26]]}),
    panels:[{lbl:'leaves', chips:[chip('1', 'dim')]},
            {lbl:'train accuracy', chips:[chip('0.7083', 'dim')]}],
    line:1,
    msg:{zh:'一棵還沒分裂的樹就是「把每個點都猜成多數類」。接下來每一次分裂，都是在所有（欄位, 門檻）組合裡挑出讓加權 gini 掉最多的那一個。',
         en:'A tree with no splits is just "call everything the majority class". Every split from here is chosen by searching all (column, threshold) pairs for the one that drops the weighted gini most.'}
  });
  GROWN.steps.forEach((st, i) => {
    const nm = st.feat === 0 ? 'x' : 'y';
    const last = i === total - 1;
    F.push({
      shapes:growStage(st, {
        mark:last ? 23 : null,
        cap:{zh:'第 ' + (i + 1) + ' 次分裂（深度 ' + st.depth + '）：' + nm + ' <= ' + st.thr.toFixed(3) +
                '，gini 下降 ' + st.gain.toFixed(5),
             en:'split ' + (i + 1) + ' at depth ' + st.depth + ': ' + nm + ' <= ' + st.thr.toFixed(3) +
                ', gini drops by ' + st.gain.toFixed(5)},
        cap2:{zh:'這一刀切的是 ' + st.n + ' 個點所在的那一塊',
              en:'it cuts the region holding ' + st.n + ' of the points'},
        side:[[{zh:'葉子 ' + st.leaves.length + '，準確率 ' + st.acc.toFixed(4),
                en:st.leaves.length + ' leaves, accuracy ' + st.acc.toFixed(4)}, COL.purpleL, .30],
              [{zh:'每一塊都是軸對齊的長方形', en:'every region is an axis-aligned box'}, COL.grey, .26]]}),
      panels:[{lbl:'leaves', chips:[chip(String(st.leaves.length), 'hot')]},
              {lbl:'train accuracy', chips:[chip(st.acc.toFixed(4), last ? 'bad' : 'ok')]},
              {lbl:'gini gain', chips:[chip(st.gain.toFixed(5), st.gain < .05 ? 'dim' : 'ok')]}],
      line:st.depth === 0 ? 4 : (st.gain < .05 ? 5 : 4),
      msg:i === 0
        ? {zh:'第一刀把下半部切出來，準確率從 0.7083 跳到 ' + st.acc.toFixed(4) + '。注意切線永遠跟座標軸平行——樹沒有辦法畫斜線，只能用階梯去逼近。',
           en:'The first cut takes off the lower band and accuracy jumps from 0.7083 to ' + st.acc.toFixed(4) + '. Note the cut is parallel to an axis: a tree cannot draw a diagonal, only a staircase that approximates one.'}
        : (st.gain < .05
           ? {zh:'這一刀的 gini 只掉 ' + st.gain.toFixed(5) + '，準確率一點都沒動（' + st.acc.toFixed(4) + '）。它存在的唯一理由是讓下一刀有機會切出那個落單的點。',
              en:'This cut drops gini by only ' + st.gain.toFixed(5) + ' and moves accuracy not at all (' + st.acc.toFixed(4) + '). Its only purpose is to give the next cut a chance at the one odd row.'}
           : (last
              ? {zh:'準確率 ' + st.acc.toFixed(4) + '。最後兩刀合起來只為了把紅圈裡那一個點單獨關進一片小長方形——那不是學到了什麼，那是把它背起來。只要每個不純的節點還找得到增益為正的分裂，樹就一定能走到 1.0000——下一個分頁有一個連這個條件都不成立的反例。',
                 en:'Accuracy ' + st.acc.toFixed(4) + '. The last two cuts exist only to lock the circled point into a box of its own - that is not learning, it is memorising. As long as every impure node still offers a split with positive gain, a tree always reaches 1.0000 - and the next tab holds a data set where even that fails.'}
              : {zh:'再一刀。每一刀都讓訓練準確率往上或持平，永遠不會往下——所以「訓練準確率」這個數字不可能告訴你什麼時候該停。',
                 en:'One more cut. Every cut moves training accuracy up or leaves it alone, never down - which is why that number can never tell you when to stop.'}))
    });
  });
  return F.list;
}

/* ================================================================ tab 2: XOR */
const XOR_PTS = (function(){
  // a deterministic 8x8 lattice.  It is perfectly symmetric, so every candidate
  // split has a gain of exactly zero - one of the two ways a tree stops short of 1.000
  const out = [];
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 8; j++){
      const x = -0.875 + i * 0.25, y = -0.875 + j * 0.25;
      out.push([x, y, ((x > 0) !== (y > 0)) ? 1 : 0]);
    }
  return out;
})();
const CODE_XOR = [
  'gain, j, t = best_split(X, y)   # the greedy root',
  '',
  '# y = (x > 0) XOR (y > 0)',
  '#   gain of splitting x at 0 : 0.000000',
  '#   gain of splitting y at 0 : 0.000000',
  '#   gain greedy actually took: whatever noise offers',
  '',
  '# a depth-2 tree represents this exactly.',
  '# greedy top-down search never reaches it.'
];
function xorStage(o){
  const sh = [], plot = new Plot(1.95, .55, 5.9, 4.5, [[-1, -1], [1, 1]]);
  plot.frame(sh);
  dots(sh, plot, XOR_PTS.map(p => [p[0], p[1]]),
       XOR_PTS.map(p => (p[2] === 1 ? 'act' : 'ok')), null, .11);
  (o.lines || []).forEach(ln => {
    const a = plot.px(ln.feat === 0 ? [ln.thr, -1] : [-1, ln.thr]);
    const b = plot.px(ln.feat === 0 ? [ln.thr, 1] : [1, ln.thr]);
    sh.push(S.e(a[0], a[1], b[0], b[1], {s:ln.st, arrow:false, w:.07}));
  });
  side(sh, .35, 1.1, o.side, .46);
  return caps(sh, o);
}
function xorFrames(v){
  const F = new Frames();
  const par = giniOf(XOR_PTS);
  const gainAt = (feat, thr) => {
    const L = XOR_PTS.filter(r => r[feat] <= thr), R = XOR_PTS.filter(r => r[feat] > thr);
    return par - (L.length * giniOf(L) + R.length * giniOf(R)) / XOR_PTS.length;
  };
  const greedy = bestSplit2D(XOR_PTS);
  const steps = v === 0 ? [
    {lines:[], cap:{zh:'y = (x > 0) XOR (y > 0)，完全沒有標籤雜訊',
                    en:'y = (x > 0) XOR (y > 0), with no label noise at all'},
     cap2:{zh:'對角線上的兩塊是一類，另外兩塊是另一類',
           en:'two opposite quadrants are one class, the other two are the other'},
     side:[[{zh:'根節點 gini ' + par.toFixed(4), en:'root gini ' + par.toFixed(4)}, COL.purpleL, .28]],
     line:2,
     msg:{zh:'先看一眼資料。深度 2 的樹可以完美表達它：x 在 0 切一刀，兩個子節點各自在 y 的 0 再切一刀，四片純葉子。',
          en:'Look at the data first. A depth-2 tree represents it exactly: cut x at 0, then cut y at 0 inside both children, and you have four pure leaves.'}},
    {lines:[{feat:0, thr:0, st:'ok'}], cap:{zh:'在 x = 0 切一刀：gini 下降 ' + gainAt(0, 0).toFixed(6),
                                            en:'cut x at 0: gini drops by ' + gainAt(0, 0).toFixed(6)},
     cap2:{zh:'左右兩邊各有一半的點是類別 1——這一刀什麼都沒分開',
           en:'both halves are still half class 1 - this cut separates nothing'},
     side:[[{zh:'增益 ' + gainAt(0, 0).toFixed(6), en:'gain ' + gainAt(0, 0).toFixed(6)}, COL.tealL, .28],
           [{zh:'看起來完全不值得做', en:'it looks completely worthless'}, COL.grey, .26]],
     line:3,
     msg:{zh:'這就是整個問題：那一刀的即時收益是 0。貪婪演算法只看「現在這一步」，而正確的那一步在當下看起來毫無價值。',
          en:'That is the whole problem: the immediate gain of the correct cut is 0. A greedy search only looks at this step, and the right step looks worthless at this step.'}},
    {lines:[{feat:0, thr:0, st:'ok'}, {feat:1, thr:0, st:'ok'}],
     cap:{zh:'再切一刀，四片葉子全部純淨，準確率 1.0000',
          en:'one more cut and all four leaves are pure: accuracy 1.0000'},
     cap2:{zh:'兩刀，深度 2，完全正確', en:'two cuts, depth 2, exactly right'},
     side:[[{zh:'準確率 1.0000', en:'accuracy 1.0000'}, COL.tealL, .28],
           [{zh:'但沒有任何貪婪搜尋會走到這裡', en:'and no greedy search ever gets here'}, COL.red, .26]],
     line:7,
     msg:{zh:'收益全部藏在第二刀。可以被表達，但到不了——找最佳的樹是 NP-hard，所以實際在用的樹學習器幾乎都是貪婪的（精確求解器確實存在——MIP、DL8.5、MurTree、GOSDT——但那是給另一種規模的問題用的另一種工具）。',
          en:'All of the gain is in the second cut. Representable and unreachable at the same time. Finding the optimal tree is NP-hard, so essentially every tree learner in production is greedy. Exact solvers do exist - MIP formulations, DL8.5, MurTree, GOSDT - but they are a different tool for a different size of problem.'}}
  ] : [
    {lines:[{feat:greedy.feat, thr:greedy.thr, st:'bad'}],
     cap:{zh:'貪婪實際挑的那一刀：' + (greedy.feat === 0 ? 'x' : 'y') + ' <= ' + greedy.thr.toFixed(3) +
             '，增益 ' + greedy.gain.toFixed(6),
          en:'what greedy actually takes: ' + (greedy.feat === 0 ? 'x' : 'y') + ' <= ' +
             greedy.thr.toFixed(3) + ', gain ' + greedy.gain.toFixed(6)},
     cap2:{zh:'它切在邊緣，削下一小條——因為那一條剛好不太平衡',
           en:'it slices a sliver off the edge, because that sliver happens to be slightly unbalanced'},
     side:[[{zh:'增益 ' + greedy.gain.toFixed(6), en:'gain ' + greedy.gain.toFixed(6)}, COL.red, .28],
           [{zh:'而且沒有一刀比它更好', en:'and no candidate is any better'}, COL.grey, .26]],
     line:0,
     msg:{zh:'在這個格點上，連邊緣那一條的增益也是 0——所有候選都一樣差，所以挑到哪一個完全看浮點數和排序。真實資料上則是雜訊決定，而結果一樣：根節點被浪費掉。',
          en:'On this lattice even the edge sliver has gain 0 - every candidate is equally bad, so which one wins comes down to floating point and ordering. On real data it is sampling noise that decides, with the same outcome: the root is spent on nothing.'}},
    {lines:[{feat:greedy.feat, thr:greedy.thr, st:'bad'}, {feat:0, thr:0, st:'ghost'},
            {feat:1, thr:0, st:'ghost'}],
     cap:{zh:'虛線是那棵存在卻走不到的深度 2 的樹',
          en:'the dashed lines are the depth-2 tree that exists and cannot be reached'},
     cap2:{zh:'根節點一旦用掉，後面的分裂都是在修補',
           en:'once the root is spent, every later split is patching'},
     side:[[{zh:'可表達 ≠ 找得到', en:'representable is not reachable'}, COL.red, .28],
           [{zh:'這是貪婪的定義，不是 bug', en:'this is what greedy means, not a bug'}, COL.grey, .26]],
     line:8,
     msg:{zh:'實務上的解法不是把搜尋改好，而是換一塊更大的積木。注意不是「boosting 就能解」：一堆只有一刀的樹加起來是可加模型，表達不了 XOR 的任何一部分，而這個分頁的格點完美平衡，樹樁在這裡連 0.5 都超不過。在 trees.py 那份 400 點的隨機 XOR 上，深度 1 跑 200 輪只到 0.6900（高過 0.5 只因為有限樣本不平衡），深度 2 則是 1.0000。',
          en:'The practical answer is not a better search but a bigger building block - and note that boosting alone is not it: a sum of single-split trees is additive and captures no part of XOR, and the lattice on this tab is perfectly balanced, so stumps cannot even beat 0.5 on it. On the 400-point random XOR in trees.py, 200 rounds of depth 1 reach 0.6900 - above 0.5 only because a finite sample is never perfectly balanced - while depth 2 reaches 1.0000.'}}
  ];
  steps.forEach(s => F.push({
    shapes:xorStage(s),
    panels:[{lbl:'root gain', chips:[chip((v === 0 ? (s.lines.length ? gainAt(0, 0) : par) : greedy.gain).toFixed(6),
                                          v === 0 ? 'ok' : 'bad')]}],
    line:s.line, msg:s.msg}));
  return F.list;
}

/* ================================================================ tab 3: importance */
// every number here is printed by trees.py section 5
const IMP_NAMES = ['real', 'cont. noise', 'noise b1', 'noise b2', 'noise b3'];
const IMP_GINI = [0.347, 0.580, 0.022, 0.025, 0.026];
const IMP_PERM = [0.168, 0.033, 0.007, -0.019, 0.006];
const IMP_CANDS = [1, 209, 1, 1, 1];
const CODE_IMP = [
  '# what the tree reports about itself',
  'imp[node.feat] += n * gini - nL * giniL - nR * giniR',
  'imp /= imp.sum()',
  '',
  '# what actually breaks when a column is destroyed',
  'base = score(model.predict(X_holdout), y_holdout)',
  'for j in range(d):',
  '    Xp = X_holdout.copy()',
  '    Xp[:, j] = rng.permutation(Xp[:, j])',
  '    drop[j] = base - score(model.predict(Xp), y_holdout)'
];
function impFrames(){
  const F = new Frames();
  const panelBars = (sh, x0, vals, step, fmt, colOf) => {
    const BW = .58, GAP = .26, BASE = 4.10, MAXH = 2.5;
    const top = Math.max.apply(null, vals.map(Math.abs));
    vals.forEach((v, i) => {
      if (i > step) return;
      const h = Math.max(.05, MAXH * Math.abs(v) / top);
      const x = x0 + i * (BW + GAP);
      sh.push(S.r(x, v >= 0 ? BASE - h : BASE, BW, h, colOf(i), null));
      sh.push(fitT(x + BW / 2, v >= 0 ? BASE - h - .20 : BASE + h + .30, fmt(v),
                   {c:v >= 0 ? COL.pale : COL.grey, fs:.24}));
    });
    IMP_NAMES.forEach((nm, i) => {
      if (i > step) return;
      sh.push(fitT(x0 + i * (BW + GAP) + BW / 2, 4.62, nm, {c:COL.grey, fs:.21}));
    });
  };
  const steps = [
    {step:4, which:'gini'}, {step:4, which:'perm'}
  ];
  for (let k = 0; k < 2; k++){
    const gini = k === 0;
    const sh = [];
    panelBars(sh, .80, gini ? IMP_GINI : IMP_PERM, 4,
              v => (gini ? v.toFixed(3) : (v >= 0 ? '+' : '') + v.toFixed(3)),
              i => (i === 0 ? 'ok' : (i === 1 ? 'bad' : 'done')));
    sh.push(S.e(.70, 4.10, 5.10, 4.10, {s:'soft', arrow:false, w:.03}));
    side(sh, 5.55, 1.15, [
      [gini ? {zh:'樹自己報告的重要度', en:'what the tree says about itself'}
            : {zh:'打亂一欄，看分數掉多少', en:'shuffle one column, watch the score fall'},
       COL.purpleL, .30],
      [{zh:'根節點可用的門檻數：', en:'thresholds offered at the root:'}, COL.grey, .26],
      [{zh:IMP_CANDS.join('  /  '), en:IMP_CANDS.join('  /  ')}, COL.orange, .28],
      [{zh:'真訊號 1 個，連續雜訊 209 個', en:'1 for the real column, 209 for the noise'},
       COL.grey, .24]
    ], .44);
    caps(sh, {
      cap:gini ? {zh:'impurity 重要度把純雜訊那一欄排第一（0.580 對 0.347）',
                  en:'impurity importance ranks the pure-noise column first: 0.580 against 0.347'}
               : {zh:'在沒看過的資料上做 permutation：真訊號 +0.168，雜訊 +0.033',
                  en:'permutation importance on held-out rows: the real column +0.168, the noise +0.033'},
      cap2:gini ? {zh:'同一棵樹，訓練準確率 1.000、測試 0.656',
                   en:'the same tree scores 1.000 on its training rows and 0.656 on held-out ones'}
                : {zh:'問的是「毀掉這一欄會壞掉多少」，而不是「這一欄分裂過幾次」',
                   en:'it asks what breaks when the column is destroyed, not how often the column was used'}
    });
    F.push({shapes:sh,
      panels:[{lbl:gini ? 'impurity importance' : 'permutation importance',
               chips:IMP_NAMES.map((nm, i) => chip(nm + ' ' + (gini ? IMP_GINI[i].toFixed(3)
                                                                   : IMP_PERM[i].toFixed(3)),
                                                   i === 0 ? 'ok' : (i === 1 ? 'bad' : 'dim')))}],
      line:gini ? 1 : 9,
      msg:gini
        ? {zh:'impurity 重要度加總的是「這一欄的分裂一共移除了多少不純度」。問題在於它完全沒有問「移除得有沒有意義」——一個有 209 個候選門檻的連續雜訊欄，光靠運氣就能在訓練資料上切出很多看起來很漂亮的刀。',
           en:'Impurity importance adds up how much impurity each column\'s splits removed. It never asks whether removing it meant anything - and a continuous noise column with 209 candidate thresholds can, by luck alone, cut the training data in ways that look excellent.'}
        : {zh:'permutation 把一欄打亂再量分數掉多少，問的正是那個被跳過的問題。注意它一定要在沒看過的資料上做：在訓練資料上量，它會連「背起來用的那一欄」也一起獎勵。',
           en:'Permutation importance shuffles one column and measures how far the score falls - exactly the question the other table skips. It has to be measured on held-out rows: run it on the training set and it will happily reward the column the tree memorised with.'}});
  }
  return F.list;
}

/* ================================================================ tab 4: forest */
// measured by trees.py section 7 on 900 rows / 10 columns, 3 forest seeds
const ROOTS_BAG = [90, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const ROOTS_RF = [29, 18, 9, 15, 7, 3, 2, 1, 5, 1];
const FOREST_NUM = {one:0.773, bag:0.857, rf:0.873, disBag:0.244, disRf:0.304,
                    oobBag:0.826, oobRf:0.834};
const CODE_RF = [
  'for _ in range(n_trees):',
  '    idx = rng.integers(0, n, n)          # bagging: different rows',
  '    feats = rng.choice(d, max_features)  # different questions',
  '    trees.append(DecisionTree(max_features=feats).fit(X[idx], y[idx]))',
  '',
  '# out of bag is free: each bootstrap leaves out about',
  '# (1 - 1/n)^n -> 1/e = 36.8% of the rows'
];
function forestFrames(v){
  const F = new Frames();
  const roots = v === 0 ? ROOTS_BAG : ROOTS_RF;
  const nm = v === 0 ? {zh:'只做 bagging（10 欄全開）', en:'bagging only, all 10 columns'}
                     : {zh:'隨機森林（每次隨機 3 欄）', en:'random forest, 3 random columns per split'};
  const acc = v === 0 ? FOREST_NUM.bag : FOREST_NUM.rf;
  const dis = v === 0 ? FOREST_NUM.disBag : FOREST_NUM.disRf;
  const oob = v === 0 ? FOREST_NUM.oobBag : FOREST_NUM.oobRf;
  for (let step = 0; step < 3; step++){
    const sh = [];
    const BW = .54, GAP = .26, BASE = 4.20, MAXH = 2.6;
    roots.forEach((c, i) => {
      const h = Math.max(.05, MAXH * c / 90);
      const x = .80 + i * (BW + GAP);
      sh.push(S.r(x, BASE - h, BW, h, c === Math.max.apply(null, roots) ? 'hot' : 'ok', null));
      if (c) sh.push(fitT(x + BW / 2, BASE - h - .20, String(c), {c:COL.pale, fs:.24}));
      sh.push(fitT(x + BW / 2, 4.66, String(i), {c:COL.grey, fs:.22}));
    });
    sh.push(S.e(.70, 4.20, 8.95, 4.20, {s:'soft', arrow:false, w:.03}));
    sh.push(fitT(4.85, 5.02, {zh:'90 棵樹各自的根節點用了哪一欄',
                              en:'which column each of 90 trees opens with'}, {c:COL.grey, fs:.26}));
    const lines = [[nm, COL.purpleL, .30]];
    if (step >= 1) lines.push([{zh:'測試準確率 ' + acc.toFixed(3) + '（單棵樹 ' + FOREST_NUM.one.toFixed(3) + '）',
                                en:'test accuracy ' + acc.toFixed(3) + ' (one tree alone: ' + FOREST_NUM.one.toFixed(3) + ')'},
                               COL.tealL, .28]);
    if (step >= 2) lines.push([{zh:'兩棵樹意見不同的比例 ' + dis.toFixed(3),
                                en:'two trees disagree on ' + dis.toFixed(3) + ' of the rows'}, COL.orange, .28],
                              [{zh:'out-of-bag 分數 ' + oob.toFixed(3) + '（不用切驗證集）',
                                en:'out-of-bag score ' + oob.toFixed(3) + ', no validation split needed'},
                               COL.grey, .26]);
    side(sh, .80, .72, lines, .44);
    caps(sh, {
      cap:step === 0 ? nm
        : (step === 1 ? {zh:'平均 30 棵樹：' + FOREST_NUM.one.toFixed(3) + ' → ' + acc.toFixed(3),
                         en:'averaging 30 trees: ' + FOREST_NUM.one.toFixed(3) + ' -> ' + acc.toFixed(3)}
                      : {zh:'意見分歧度 ' + dis.toFixed(3) + '，oob ' + oob.toFixed(3),
                         en:'disagreement ' + dis.toFixed(3) + ', out-of-bag ' + oob.toFixed(3)}),
      cap2:v === 0 ? {zh:'900 列、10 欄，其中一欄主宰答案',
                      en:'900 rows, 10 columns, one of which dominates the answer'}
                   : {zh:'同一份資料，只是每次分裂只能看 3 欄',
                      en:'the same data, except each split may only look at 3 columns'}});
    F.push({shapes:sh,
      panels:[{lbl:'root column', chips:roots.map((c, i) => chip(i + ':' + c,
                 c === 0 ? 'dim' : (c === Math.max.apply(null, roots) ? 'hot' : 'ok')))},
              {lbl:'test / disagree', chips:[chip(acc.toFixed(3), 'ok'), chip(dis.toFixed(3), 'act')]}],
      line:step === 2 ? 5 : (v === 0 ? 1 : 2),
      msg:step === 0
        ? (v === 0
           ? {zh:'只做 bagging 的話，90 棵樹裡有 90 棵都用第 0 欄當根節點。它們看到的列不一樣，但問的第一個問題一模一樣。',
              en:'With bagging alone, 90 trees out of 90 open with column 0. They saw different rows, but they all asked the same first question.'}
           : {zh:'每次分裂只隨機看 3 欄之後，根節點散開了：29 / 18 / 9 / 15 / 7 ……。第 0 欄還是最常被挑，但它不再壟斷。',
              en:'Once each split may only see 3 random columns, the roots spread out: 29 / 18 / 9 / 15 / 7 and so on. Column 0 is still the most popular, but it no longer monopolises.'})
        : step === 1
          ? {zh:'平均之後，' + FOREST_NUM.one.toFixed(3) + ' 變成 ' + acc.toFixed(3) + '。這一段的收益幾乎全部來自 bagging——單棵樹 variance 很高（抽掉一列就換一棵樹），而平均正是用來消 variance 的。',
             en:'Averaging turns ' + FOREST_NUM.one.toFixed(3) + ' into ' + acc.toFixed(3) + '. Almost all of that comes from bagging: a single tree has enormous variance (remove one row and you get a different tree), and averaging is what variance is vulnerable to.'}
          : {zh:'分歧度 ' + dis.toFixed(3) + ' 才是重點：平均一群意見相同的預測器什麼都買不到。隨機選欄位把分歧度從 0.244 推到 0.304，準確率只多 +0.017——機制很清楚，但它的貢獻比大家講的小。oob 分數則是真的免費：每棵樹大約有 36.8% 的列沒看過。',
             en:'The disagreement number is the point: averaging predictors that agree buys nothing. Feature subsampling pushes it from 0.244 to 0.304 and accuracy by only +0.017 - the mechanism is clear, the size of it is smaller than it is usually sold as. The out-of-bag score really is free: each tree never saw about 36.8% of the rows.'}});
  }
  return F.list;
}

/* ================================================================ tab 5: extrapolation */
const CODE_EXT = [
  'def predict(x):',
  '    node = root',
  '    while not node.is_leaf:',
  '        node = node.left if x <= node.thr else node.right',
  '    return node.value        # a constant, for every x in that leaf',
  '',
  '# the last leaf has no right edge, so its constant',
  '# is the answer for every x from here to infinity'
];
function extFrames(){
  const F = new Frames();
  const P1 = X1D.map((x, i) => [x, Y1D[i]]);
  const YLO = -0.75, YHI = 1.30;
  const steps = [1, 2, 3];
  steps.forEach((depth, k) => {
    const pred = regTree(X1D, Y1D, depth);
    const sh = [];
    const plot = new Plot(.45, .55, 5.1, 4.4, [[0, YLO], [1.55, YHI]]);
    plot.frame(sh);
    // the step function, sampled densely so the steps are visible
    let prev = null;
    for (let i = 0; i <= 155; i++){
      const x = i / 100;                       // 0 .. 1.55
      const yv = pred(x);
      const q = plot.px([x, yv]);
      if (prev) sh.push(S.e(prev[0], prev[1], q[0], q[1], {s:'hot', arrow:false, w:.055}));
      prev = q;
    }
    const edge = plot.px([X1D[X1D.length - 1], YLO]);
    const edgeTop = plot.px([X1D[X1D.length - 1], YHI]);
    sh.push(S.e(edge[0], edge[1], edgeTop[0], edgeTop[1], {s:'ghost', arrow:false, w:.03}));
    dots(sh, plot, P1, P1.map(() => 'ok'), null, .10);
    side(sh, 5.75, 1.15, [
      [{zh:'深度 ' + depth, en:'depth ' + depth}, COL.purpleL, .30],
      [{zh:Math.pow(2, depth) + ' 片葉子', en:Math.pow(2, depth) + ' leaves'}, COL.grey, .26],
      [{zh:'訓練 MSE ' + mseOf(X1D.map(pred), Y1D).toFixed(5),
        en:'train MSE ' + mseOf(X1D.map(pred), Y1D).toFixed(5)}, COL.tealL, .28],
      [{zh:'f(1.5) = ' + pred(1.5).toFixed(3), en:'f(1.5) = ' + pred(1.5).toFixed(3)},
       COL.orange, .28],
      [{zh:'f(100) = ' + pred(100).toFixed(3), en:'f(100) = ' + pred(100).toFixed(3)},
       COL.red, .28]
    ], .42);
    caps(sh, {
      cap:{zh:'深度 ' + depth + ' 的迴歸樹：' + Math.pow(2, depth) + ' 個常數拼成一個階梯',
           en:'a depth-' + depth + ' regression tree: ' + Math.pow(2, depth) + ' constants glued into a staircase'},
      cap2:{zh:'虛線右邊沒有任何訓練資料，而樹在那裡完全是水平的',
            en:'there is no training data right of the dashed line, and the tree is perfectly flat there'}});
    F.push({shapes:sh,
      panels:[{lbl:'leaves', chips:[chip(String(Math.pow(2, depth)), 'hot')]},
              {lbl:'train MSE', chips:[chip(mseOf(X1D.map(pred), Y1D).toFixed(5), 'ok')]},
              {lbl:'f(100)', chips:[chip(pred(100).toFixed(3), 'bad')]}],
      line:4,
      msg:k < steps.length - 1
        ? {zh:'加深一層，階梯變細，訓練 MSE 掉到 ' + mseOf(X1D.map(pred), Y1D).toFixed(5) + '。但不管切多細，每一片葉子裡放的都還是一個常數。',
           en:'One more level, finer steps, training MSE down to ' + mseOf(X1D.map(pred), Y1D).toFixed(5) + '. However fine the steps get, every leaf still holds one constant.'}
        : {zh:'最右邊那一片葉子沒有右邊界，所以它那個常數就是從資料邊緣一路到無限遠的答案：x = 1.5 給 ' + pred(1.5).toFixed(3) + '，x = 100 也給 ' + pred(100).toFixed(3) + '。森林也救不了——把一堆水平線平均起來還是水平線。輸出是訓練目標的非負加權平均的模型，全都有這個性質；實務上的解法不是換一棵更好的樹，而是先把趨勢單獨模型化，再讓樹去配剩下的部分。',
           en:'The rightmost leaf has no right edge, so its constant is the answer from the edge of the data all the way out: ' + pred(1.5).toFixed(3) + ' at x = 1.5 and ' + pred(100).toFixed(3) + ' at x = 100. A forest does not help - averaging flat lines gives a flat line. Every model whose output is a non-negative weighted average of training targets has this property, and the fix is not a better tree but modelling the trend separately and letting the tree fit what is left.'}});
  });
  return F.list;
}

/* ================================================================ tab 6: boosting */
const CODE_BOOST = [
  'F = np.full(len(y), y.mean())        # the best constant',
  'for _ in range(n_trees):',
  '    resid = y - F                    # = -dL/dF for L = 0.5 (y - F)^2',
  '    tree = DecisionTree(max_depth=1).fit(X, resid)',
  '    F += lr * tree.predict(X)',
  '',
  '# "fit the next tree to what is left over" is not a',
  '# separate idea - it is what the gradient looks like.'
];
function boostFrames(){
  const F = new Frames();
  const LR = 0.5, ROUNDS = 6;
  const f0 = meanOf(Y1D);
  let Fv = Y1D.map(() => f0);
  const hist = [{Fv:Fv.slice(), thr:null, mse:mseOf(Fv, Y1D)}];
  for (let r = 0; r < ROUNDS; r++){
    const resid = Y1D.map((y, i) => y - Fv[i]);
    const st = regTree(X1D, resid, 1);
    Fv = Fv.map((v, i) => v + LR * st(X1D[i]));
    hist.push({Fv:Fv.slice(), mse:mseOf(Fv, Y1D), resid:resid.slice()});
  }
  const YLO = -0.80, YHI = 1.30;
  hist.forEach((h, r) => {
    const sh = [];
    const plot = new Plot(.45, .55, 4.9, 4.4, [[0, YLO], [1.0, YHI]]);
    plot.frame(sh);
    const zero = plot.px([0, 0]), zend = plot.px([1, 0]);
    sh.push(S.e(zero[0], zero[1], zend[0], zend[1], {s:'ghost', arrow:false, w:.025}));
    // residual sticks: how far each point still is from the model
    X1D.forEach((x, i) => {
      const a = plot.px([x, h.Fv[i]]), b = plot.px([x, Y1D[i]]);
      sh.push(S.e(a[0], a[1], b[0], b[1], {s:'bad', arrow:false, w:.04}));
    });
    // the model so far, as a step function
    let prev = null;
    for (let i = 0; i <= 120; i++){
      const x = i / 120;
      let yv = f0;
      for (let k = 0; k < r; k++){
        const resid = Y1D.map((y, j) => y - hist[k].Fv[j]);
        yv += LR * regTree(X1D, resid, 1)(x);
      }
      const q = plot.px([x, yv]);
      if (prev) sh.push(S.e(prev[0], prev[1], q[0], q[1], {s:'hot', arrow:false, w:.055}));
      prev = q;
    }
    dots(sh, plot, X1D.map((x, i) => [x, Y1D[i]]), X1D.map(() => 'ok'), null, .10);
    side(sh, 5.55, 1.15, [
      [r === 0 ? {zh:'F0 = 平均數 ' + f0.toFixed(4), en:'F0 = the mean, ' + f0.toFixed(4)}
               : {zh:'第 ' + r + ' 棵樹之後', en:'after tree ' + r}, COL.purpleL, .30],
      [{zh:'MSE ' + h.mse.toFixed(5), en:'MSE ' + h.mse.toFixed(5)}, COL.tealL, .28],
      [{zh:'紅線＝剩下的殘差', en:'red = the residual left'}, COL.red, .26],
      [{zh:'每棵樹一刀，lr = 0.5', en:'one split per tree, lr = 0.5'}, COL.grey, .24]
    ], .42);
    caps(sh, {
      cap:r === 0 ? {zh:'起點：把每個點都預測成平均數 ' + f0.toFixed(4) + '，MSE ' + h.mse.toFixed(5),
                     en:'the start: predict the mean ' + f0.toFixed(4) + ' for everything, MSE ' + h.mse.toFixed(5)}
                  : {zh:'第 ' + r + ' 棵樹（一刀）之後，MSE ' + h.mse.toFixed(5),
                     en:'after tree ' + r + ' - a single split - MSE is ' + h.mse.toFixed(5)},
      cap2:{zh:'下一棵樹要配的不是 y，而是紅線的長度',
            en:'the next tree is fitted to the red sticks, not to y'}});
    F.push({shapes:sh,
      panels:[{lbl:'trees', chips:[chip(String(r), r ? 'hot' : 'dim')]},
              {lbl:'MSE', chips:[chip(h.mse.toFixed(5), 'ok')]}],
      line:r === 0 ? 0 : 2,
      msg:r === 0
        ? {zh:'平方損失下最好的常數就是平均數（Day 45）。這就是 F0，也是整個模型的第一項。',
           en:'The best constant under squared loss is the mean (Day 45). That is F0, and it is the first term of the model.'}
        : (r === 1
           ? {zh:'第一棵樹只有一刀，卻把 MSE 從 ' + hist[0].mse.toFixed(5) + ' 砍到 ' + h.mse.toFixed(5) + '。它配的是殘差 y − F，而對平方損失來說殘差就是 −dL/dF：所以「配剩下的」跟「往梯度反方向走」是同一件事。',
              en:'One split takes the MSE from ' + hist[0].mse.toFixed(5) + ' to ' + h.mse.toFixed(5) + '. It was fitted to the residual y − F, and for squared loss the residual IS −dL/dF - so "fit what is left" and "step against the gradient" are the same sentence.'}
           : (r === ROUNDS
              ? {zh:'六刀之後 MSE 來到 ' + h.mse.toFixed(5) + '，而每一棵樹都只有一刀、自己什麼都做不了。這就是 boosting 殺 bias 的方式：刻意從一個很差的模型出發，小步小步走過去。換掉損失函數，下一棵樹要配的東西就跟著換——那就是「gradient」這個字的全部內容。',
                 en:'Six splits later the MSE is ' + h.mse.toFixed(5) + ', and no individual tree here can do anything at all. That is how boosting attacks bias: start deliberately underfitted and walk in small steps. Change the loss and the target of the next tree changes with it - that is the whole content of the word gradient.'}
              : {zh:'再一棵樹，再一小步。lr = 0.5 代表每棵樹只走它想走的一半——步子小、棵數多，最後停的地方比較好。',
                 en:'One more tree, one more small step. lr = 0.5 means each tree only moves half as far as it wants: smaller steps, more of them, and a better place to stop.'}))});
  });
  return F.list;
}

/* ================================================================ tab 7: the paths */
const CODE_LC = [
  'def hasPathSum(root, target):',
  '    if root is None:',
  '        return False                # NOT "target == 0"',
  '    if root.left is None and root.right is None:',
  '        return target == root.val   # a leaf has NO children',
  '    rest = target - root.val',
  '    return (hasPathSum(root.left, rest)',
  '            or hasPathSum(root.right, rest))'
];
// [1, 2] for LeetCode 112, [1, 2, 3, null, 5] for LeetCode 257
const LC_TREES = [
  {nodes:[{id:0, v:1, x:4.9, y:1.30, kids:[1]}, {id:1, v:2, x:3.6, y:2.60, kids:[]}],
   name:{zh:'[1, 2]，target 1', en:'[1, 2], target 1'}},
  {nodes:[{id:0, v:1, x:4.9, y:1.30, kids:[1, 2]},
          {id:1, v:2, x:3.4, y:2.60, kids:[3]},
          {id:2, v:3, x:6.4, y:2.60, kids:[]},
          {id:3, v:5, x:2.6, y:3.90, kids:[]}],
   name:{zh:'[1, 2, 3, null, 5]', en:'[1, 2, 3, null, 5]'}}
];
function lcStage(v, hot, ghost, o){
  const sh = [], T_ = LC_TREES[v];
  T_.nodes.forEach(nd => {
    nd.kids.forEach(k => {
      const ch = T_.nodes[T_.nodes.findIndex(q => q.id === k)];
      sh.push(S.e(nd.x, nd.y + .30, ch.x, ch.y - .30, {s:'soft', arrow:false, w:.035}));
    });
  });
  if (ghost){
    const nd = T_.nodes[0];
    sh.push(S.e(nd.x, nd.y + .30, nd.x + 1.3, nd.y + 1.0, {s:'ghost', arrow:false, w:.035}));
    sh.push(S.c(nd.x + 1.3, nd.y + 1.3, .30, 'bad', 'None', {fs:.20}));
  }
  T_.nodes.forEach(nd => {
    sh.push(S.c(nd.x, nd.y, .30, (hot || []).indexOf(nd.id) >= 0 ? 'hot' : 'ok',
                String(nd.v), {fs:.26}));
  });
  side(sh, .40, 4.60, o.side, .42);
  return caps(sh, o);
}
function lcFrames(v){
  const F = new Frames();
  const steps = v === 0 ? [
    {hot:[0], ghost:false,
     cap:{zh:'LeetCode 112：有沒有一條「根到葉」的路徑加起來等於 target？',
          en:'LeetCode 112: is there a root-to-leaf path whose values sum to target?'},
     cap2:{zh:'這棵樹只有兩個節點，節點 1 有左子樹、沒有右子樹',
           en:'two nodes: the root has a left child and no right child'},
     side:[[{zh:'target 1，根節點值 1，剩下 0', en:'target 1, root value 1, remainder 0'},
            COL.purpleL, .28]],
     line:5,
     msg:{zh:'正確答案是 False：唯一的葉子是節點 2，而 1 + 2 = 3，不是 1。',
          en:'The correct answer is False: the only leaf is the node holding 2, and 1 + 2 = 3, not 1.'}},
    {hot:[0], ghost:true,
     cap:{zh:'把「走到 None」當成「走到葉子」的寫法',
          en:'the spelling that treats "ran out of tree" as "reached a leaf"'},
     cap2:{zh:'if root is None: return target == 0',
           en:'if root is None: return target == 0'},
     side:[[{zh:'右子樹是 None，而剩下的 target 剛好是 0', en:'the right child is None, and the remainder is exactly 0'},
            COL.red, .28],
           [{zh:'於是它回報 True', en:'so it reports True'}, COL.red, .26]],
     line:2,
     msg:{zh:'它走進那個根本不存在的右子樹，發現 target 剛好用完，就回報了一條不存在的路徑。3,000 棵隨機小樹裡，兩種寫法有 121 次不同（4.0%）——夠常見到會被抓，又夠罕見到能通過幾個範例。',
          en:'It walks into the right child that does not exist, finds the target already used up, and reports a root-to-leaf path that is not there. Over 3,000 random small trees the two spellings differ 121 times, 4.0% - common enough to be caught, rare enough to pass a handful of examples.'}},
    {hot:[0, 1], ghost:false,
     cap:{zh:'葉子的定義是「沒有小孩」，不是「下一步是 None」',
          en:'a leaf is a node with NO children, not a step that landed on None'},
     cap2:{zh:'決策樹永遠不會踩到這個坑：它建的每個內部節點都剛好有兩個小孩',
           en:'a decision tree never hits this: every internal node it builds gets exactly two children'},
     side:[[{zh:'1 + 2 = 3 ≠ 1 ⇒ False', en:'1 + 2 = 3, not 1  =>  False'}, COL.tealL, .28]],
     line:4,
     msg:{zh:'LeetCode 的樹可以只有一個小孩，決策樹不行——所以這個 bug 只在前者出現。而「根到葉的一條路徑」正是一棵決策樹對一筆資料的完整回答。',
          en:'LeetCode trees may have a single child; decision trees may not, which is why the bug only lives in the former. And "one root-to-leaf path" is exactly what a decision tree hands you as its complete answer for one row.'}}
  ] : [
    {hot:[0, 1, 3], ghost:false,
     cap:{zh:'LeetCode 257：把每一條根到葉的路徑都列出來',
          en:'LeetCode 257: list every root-to-leaf path'},
     cap2:{zh:'正確答案 [\'1->2->5\', \'1->3\']', en:"the correct answer is ['1->2->5', '1->3']"},
     side:[[{zh:'第一條路徑：1 → 2 → 5', en:'the first path: 1 -> 2 -> 5'}, COL.purpleL, .28]],
     line:0,
     msg:{zh:'一棵決策樹就是它所有根到葉路徑的集合——把模型攤開成一串「如果…而且…就…」的規則，靠的就是這個走訪。',
          en:'A decision tree IS the set of its root-to-leaf paths: turning a fitted model into a list of "if ... and ... then ..." rules is exactly this walk.'}},
    {hot:[0, 2], ghost:false,
     cap:{zh:'第二條路徑：1 → 3', en:'the second path: 1 -> 3'},
     cap2:{zh:'只要每次遞迴都傳一份新的 list，就不會出事',
           en:'as long as each recursive call gets a fresh list, nothing can go wrong'},
     side:[[{zh:'acc = acc + [str(node.val)]', en:'acc = acc + [str(node.val)]'}, COL.tealL, .28],
           [{zh:'一個新 list，不是就地修改', en:'a new list, not a mutated one'}, COL.grey, .26]],
     line:0,
     msg:{zh:'走訪本身三行就寫完。會出事的從來不是走訪，是那個累積器。',
          en:'The walk itself is three lines. What goes wrong is never the walk; it is the accumulator.'}},
    {hot:[0, 1, 2, 3], ghost:false,
     cap:{zh:'共用一個 list、不複製、不 pop 的版本回傳 [\'1->2->5->3\', \'1->2->5->3\']',
          en:"one shared list with no copy and no undo returns ['1->2->5->3', '1->2->5->3']"},
     cap2:{zh:'兩條路徑都錯，而且錯成同一個值',
           en:'both paths are wrong, and both are the same wrong value'},
     side:[[{zh:'存進去的是那個 list 本身', en:'what was stored is the list itself'}, COL.red, .28],
           [{zh:'它在第一片葉子之後還在長', en:'and it kept growing after the first leaf'}, COL.red, .26]],
     line:0,
     msg:{zh:'回傳的是累積器的「最終狀態」兩份，因為在第一片葉子存進去的就是那個還會繼續長的 list。這跟 Day 39 回溯搜尋裡 res.append(path) 沒複製是同一個 bug，只是換到樹上。要嘛進入時複製，要嘛在遞迴前後 append / pop。',
          en:'What came back is the accumulator\'s final state, twice, because the list appended at the first leaf is the one that kept growing afterwards. This is Day 39\'s res.append(path) bug in a tree instead of a backtracking search. Either copy on the way in, or append and pop around the recursive calls.'}}
  ];
  steps.forEach(s => F.push({
    shapes:lcStage(v, s.hot, s.ghost, s),
    panels:[{lbl:v === 0 ? 'answer' : 'paths found',
             chips:v === 0
               ? [chip(s.ghost ? 'True (wrong)' : 'False', s.ghost ? 'bad' : 'ok')]
               : [chip("'1->2->5'", 'ok'), chip("'1->3'", 'ok')]}],
    line:s.line, msg:s.msg}));
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title:{zh:'Day 48 — 決策樹、隨機森林與 Gradient Boosting',
         en:'Day 48 - Decision trees, random forests and gradient boosting'},
  sub:{zh:'同一塊積木用三種方式：一棵樹會背答案，獨立的樹平均掉 variance，相依的樹逐步啃掉 bias。',
       en:'One building block, three ways: one tree memorises, independent trees average variance away, dependent trees eat into bias.'},
  tabs:[
    {
      id:'grow', label:{zh:'把一棵樹種出來', en:'growing one tree'},
      stage:{zh:'每一刀都讓訓練準確率往上或持平，永遠不會往下',
             en:'every cut moves training accuracy up or leaves it alone, never down'},
      view:VIEW,
      idea:{zh:'決策樹是這個系列裡唯一一個沒有係數也沒有距離函數的模型——它只是在問問題，而且問的永遠是「某一欄有沒有小於某個數」，所以每一刀都跟座標軸平行，每一片葉子都是一個長方形。24 個點長出 4 刀：第一刀（y ≤ 0.445，gini 降 0.17014）把準確率從 0.7083 推到 0.7917，第二刀（x ≤ 0.455，降 0.34028）推到 0.9583，第三刀只降 0.03125 而且準確率一點都沒動，第四刀才把那個落單的點單獨關進一片小長方形，準確率 1.0000。最後兩刀合起來做的事情不是學習，是把一列資料背起來——而且只要每個不純的節點還找得到增益為正的分裂——只有完美對稱、或「特徵相同但標籤不同的重複列」會讓它失效——不設限的樹就一定能走到 1.0000。這就是 Day 47 的 k = 1 最近鄰換了一頂帽子：長到底的樹是它自己訓練集的查表。',
            en:'A decision tree is the only model in this series with no coefficients and no distance function. It asks questions, and the question is always "is this column below this number", so every cut is parallel to an axis and every leaf is a box. Four cuts grow on 24 points: the first (y <= 0.445, gini down 0.17014) takes accuracy from 0.7083 to 0.7917, the second (x <= 0.455, down 0.34028) to 0.9583, the third drops gini by only 0.03125 and moves accuracy not at all, and the fourth finally locks the one odd row into a box of its own at 1.0000. Those last two cuts are not learning, they are memorising a single row - and as long as every impure node still offers a split with positive gain - which fails only on exact symmetry or on identical rows that disagree about the label - an unrestricted tree gets to 1.0000. This is Day 47\'s k = 1 nearest neighbour wearing a different hat: a fully grown tree is a lookup table for its own training set.'},
      legend:[['#ff9736', {zh:'這一刀', en:'the cut being made'}],
              ['#3fe0dd', {zh:'類別 0 / 純淨的葉子', en:'class 0, and leaves that are pure'}],
              ['#9d6bff', {zh:'類別 1', en:'class 1'}],
              ['#ff5c5c', {zh:'被背起來的那一列', en:'the row being memorised'}]],
      get code(){ return CODE_GROW; },
      build:growFrames
    },
    {
      id:'xor', label:{zh:'貪婪看不到下一步', en:'greedy cannot look ahead'},
      stage:{zh:'可以被表達，而且到不了', en:'representable and unreachable at the same time'},
      view:VIEW,
      variants:[{zh:'正確的兩刀', en:'the two cuts that work'},
                {zh:'貪婪實際挑的', en:'what greedy actually takes'}],
      idea:{zh:'y = (x > 0) XOR (y > 0)，一點標籤雜訊都沒有，而且深度 2 的樹可以完美表達它：x 在 0 切一刀，兩個子節點再各自在 y 的 0 切一刀，四片純葉子、準確率 1.0000。問題是貪婪搜尋永遠走不到那裡——在根節點，x 和 y 各自都不帶任何資訊，所以「在 0 切一刀」的 gini 增益是 0。整個收益藏在第二刀裡，而貪婪只看第一刀。於是根節點被花在雜訊上（在真實資料上它會切在範圍邊緣，削下一條剛好不太平衡的薄片），後面的每一刀都在補救。這不是實作沒寫好：找最佳的樹是 NP-hard，所以實際在用的樹學習器幾乎都是貪婪的。實務上的出路也不是把搜尋改好，而是換一塊更大的積木：單刀樹的和是可加模型，表達不了 XOR 的任何一部分——在 trees.py 那份 400 點的隨機 XOR 上，深度 1 跑 200 輪只有 0.6900，而且高過 0.5 純粹來自有限樣本的不平衡；深度 2 是第一個裝得下的大小，200 輪就到 1.0000。這個分頁的格點是完美平衡的，所以樹樁在這裡連 0.5 都超不過。',
            en:'y = (x > 0) XOR (y > 0), with no label noise whatsoever, and a depth-2 tree represents it exactly: cut x at 0, then cut y at 0 inside both children, four pure leaves, accuracy 1.0000. Greedy search never gets there. At the root neither column carries any information on its own, so the gini gain of cutting at 0 is 0. All of the payoff is in the second cut, and greedy only looks at the first. The root is therefore spent on noise - on real data it lands at the edge of the range, slicing off a sliver that happens to be slightly unbalanced - and every later cut is patching. This is not a sloppy implementation: finding the optimal tree is NP-hard, so essentially every tree learner in production is greedy. The practical escape is not a better search but a bigger building block: a sum of single-split trees is additive and captures no part of XOR. On the 400-point random XOR in trees.py, 200 rounds of depth 1 reach 0.6900 - above 0.5 only because a finite sample is never perfectly balanced; depth 2 is the first size that holds the interaction inside one tree. The lattice on this tab is perfectly balanced, so stumps cannot even beat 0.5 on it.'},
      legend:[['#3fe0dd', {zh:'類別 0 / 有用的一刀', en:'class 0, and the cut that works'}],
              ['#9d6bff', {zh:'類別 1', en:'class 1'}],
              ['#ff5c5c', {zh:'貪婪挑的那一刀', en:'the cut greedy takes'}],
              ['#2f5661', {zh:'走不到的那棵樹', en:'the tree that cannot be reached'}]],
      get code(){ return CODE_XOR; },
      build:xorFrames
    },
    {
      id:'importance', label:{zh:'特徵重要度在說謊', en:'the importance table lies'},
      stage:{zh:'它算的是「有幾次機會」，不是「有沒有用到」',
             en:'it counts opportunity, not use'},
      view:VIEW,
      idea:{zh:'300 列資料：第 0 欄是二元的、而且就是答案（20% 的標籤被翻掉），第 1 欄是連續的純雜訊，第 2–4 欄是二元雜訊。把樹長到底之後，impurity 重要度把那一欄純雜訊排第一——0.580 對真訊號的 0.347，1.67 倍。機制在候選門檻數：一個有 210 個相異值的連續欄位，在根節點提供 209 個候選門檻，二元欄位提供 1 個。機會多 209 倍，光靠運氣就能切出很多在訓練資料上看起來很漂亮的刀，而 impurity 重要度加總的正是「這一欄的分裂一共移除了多少不純度」，它沒有辦法問「移除得有沒有意義」。permutation 重要度問的就是那個被跳過的問題：把一欄打亂，看分數掉多少——真訊號 +0.168，雜訊 +0.033。但它一定要在模型沒看過的資料上量，在訓練資料上量的話，它會把「樹拿來背答案的那一欄」也一起獎勵。這棵樹的訓練準確率是 1.000，測試是 0.656。',
            en:'300 rows: column 0 is binary and IS the answer (20% of labels flipped), column 1 is continuous pure noise, columns 2 to 4 are binary noise. Grown to purity, impurity importance ranks the pure-noise column first - 0.580 against 0.347 for the real signal, a factor of 1.67. The mechanism is the candidate count: a continuous column with 210 distinct values offers 209 thresholds at the root, a binary column offers 1. Two hundred times the chances, and luck alone produces cuts that look excellent on the training data. Impurity importance sums how much impurity each column\'s splits removed; it has no way to ask whether removing it meant anything. Permutation importance asks exactly that - shuffle a column, measure how far the score falls - and gives the real column +0.168 against the noise column\'s +0.033. It must be measured on rows the model never saw, though: run it on the training set and it will reward the column the tree memorised with. This tree scores 1.000 on its training rows and 0.656 on held-out ones.'},
      legend:[['#3fe0dd', {zh:'真訊號那一欄', en:'the column that is the answer'}],
              ['#ff5c5c', {zh:'連續的純雜訊', en:'the continuous noise column'}],
              ['#2f5661', {zh:'二元雜訊', en:'binary noise'}],
              ['#ff9736', {zh:'根節點的候選門檻數', en:'thresholds offered at the root'}]],
      get code(){ return CODE_IMP; },
      build:impFrames
    },
    {
      id:'forest', label:{zh:'森林', en:'the forest'},
      stage:{zh:'平均一群意見相同的預測器，什麼都買不到',
             en:'averaging predictors that agree buys nothing'},
      view:VIEW,
      variants:[{zh:'只做 bagging', en:'bagging only'}, {zh:'隨機森林', en:'random forest'}],
      idea:{zh:'森林有兩個旋鈕，做的是不同的事。bagging 讓每棵樹看到不一樣的「列」，max_features 讓每棵樹問不一樣的「問題」。在一份有主宰欄位的資料上（900 列、10 欄），只做 bagging 的話 90 棵樹裡有 90 棵都用第 0 欄當根節點；每次分裂只隨機看 3 欄之後，根節點散成 29 / 18 / 9 / 15 / 7 / 3 / 2 / 1 / 5 / 1。數字上：單棵樹 0.773，bagging 0.857（+0.083，這一段才是大頭），再加上隨機選欄位 0.873（+0.017，比一般講的小）。真正明顯動起來的是意見分歧度，0.244 → 0.304——而那正是平均在吃的東西。另外 out-of-bag 分數是真的免費：每個 bootstrap 樣本大約漏掉 (1 − 1/n)^n → 1/e = 36.8% 的列，所以每一列都有大約三分之一的樹沒看過它，可以拿那些樹來評分，不用另外切驗證集。',
            en:'A forest has two knobs doing different jobs. Bagging makes each tree see different rows; max_features makes each tree ask different questions. On data with a dominant column (900 rows, 10 columns), bagging alone gives 90 trees out of 90 that open with column 0; letting each split see only 3 random columns spreads the roots to 29 / 18 / 9 / 15 / 7 / 3 / 2 / 1 / 5 / 1. The numbers: one tree 0.773, bagging 0.857 (+0.083, and that is where most of the win is), feature subsampling on top 0.873 (+0.017, smaller than it is usually sold as). What clearly moves is the disagreement between trees, 0.244 to 0.304 - and that is the quantity averaging feeds on. The out-of-bag score really is free: each bootstrap sample misses about (1 - 1/n)^n -> 1/e = 36.8% of the rows, so every row is unseen by roughly a third of the trees and can be scored with exactly those, without setting aside a validation split.'},
      legend:[['#ff9736', {zh:'被挑最多次的欄位', en:'the most popular root column'}],
              ['#3fe0dd', {zh:'其他欄位', en:'the other columns'}],
              ['#9d6bff', {zh:'意見分歧度', en:'disagreement between trees'}],
              ['#2f5661', {zh:'從沒當過根節點', en:'never chosen as a root'}]],
      get code(){ return CODE_RF; },
      build:forestFrames
    },
    {
      id:'extrap', label:{zh:'樹走不出訓練範圍', en:'past the last training point'},
      stage:{zh:'葉子裡是一個常數，所以最後一階一路平到無限遠',
             en:'a leaf holds a constant, so the last step runs flat forever'},
      view:VIEW,
      idea:{zh:'迴歸樹把一條曲線近似成一個階梯：每一片葉子放一個常數，切得越細階梯越密，訓練 MSE 從 0.08117（深度 1）掉到 0.00851（深度 3）。但不管切多細，最右邊那一片葉子沒有右邊界——它那個常數就是從資料邊緣一路到無限遠的答案。這份資料的最後一個點在 x = 0.95，而模型在 x = 1.5 和 x = 100 給的是同一個數字。森林也救不了：把一堆水平線平均起來還是水平線。而且樹連「比最大的葉子平均值更大的數字」都說不出口，因為每個輸出都是訓練目標的非負加權平均——所有長這樣的模型都有這個性質。實務上的解法不是換一棵更好的樹，而是把趨勢單獨模型化（例如先配一條線），再讓樹去配剩下的殘差。',
            en:'A regression tree approximates a curve with a staircase: every leaf holds a constant, finer splits give finer steps, and the training MSE falls from 0.08117 at depth 1 to 0.00851 at depth 3. However fine it gets, the rightmost leaf has no right edge - its constant is the answer from the edge of the data all the way out. The last training point here is at x = 0.95, and the model returns the same number at x = 1.5 and at x = 100. A forest does not help: averaging flat lines gives a flat line. A tree also cannot utter a number larger than its largest leaf mean, because every output is a non-negative weighted average of training targets - which is true of every model built that way. The fix is not a better tree; it is to model the trend separately, fit a line first, and let the tree take the residual.'},
      legend:[['#ff9736', {zh:'樹的預測', en:'what the tree predicts'}],
              ['#3fe0dd', {zh:'訓練資料', en:'the training data'}],
              ['#2f5661', {zh:'資料的右邊界', en:'the edge of the data'}],
              ['#ff5c5c', {zh:'外插出來的常數', en:'the constant it extrapolates with'}]],
      get code(){ return CODE_EXT; },
      build:extFrames
    },
    {
      id:'boost', label:{zh:'Boosting', en:'boosting'},
      stage:{zh:'梯度下降，只是被下降的那個東西是一個函數',
             en:'gradient descent, where the thing being descended is a function'},
      view:VIEW,
      idea:{zh:'Day 45 在參數空間裡走：w ← w − lr · dL/dw。這裡模型是一堆樹的和，被改進的是函數 F 本身：F0 是平均數（平方損失下最好的常數），殘差 y − F 對 L = ½(y − F)² 來說剛好就是 −dL/dF，而一步就是「配在殘差上的一棵小樹」乘上 lr。所以「配剩下的」不是另一個想法，它就是平方損失的梯度長成的樣子——換個損失函數，下一棵樹要配的東西就跟著換，那是「gradient」這個字的全部內容。這裡每棵樹只有一刀、自己什麼都做不了，六棵疊起來把 MSE 從 0.25744 帶到 0.02688。boosting 打的是 bias：故意從一個很差的模型出發，小步小步走過去。代價是它的訓練損失是單調遞減的，結構上不可能告訴你什麼時候該停——所以每一個 boosting 套件都有這個參數——但沒有一個會強迫你給，它們都會很樂意跑到最後一棵。',
            en:'Day 45 walked in parameter space: w <- w - lr * dL/dw. Here the model is a sum of trees and the thing being improved is the function F itself. F0 is the mean, the best constant under squared loss; the residual y - F is exactly -dL/dF for L = 0.5(y - F)^2; and one step is a small tree fitted to that residual, scaled by lr. So "fit what is left over" is not a separate idea, it is what the gradient of squared loss looks like - change the loss and the target of the next tree changes with it, which is the whole content of the word gradient. Every tree here is a single split and can do nothing on its own; six of them stacked take the MSE from 0.25744 to 0.02688. Boosting attacks bias: start deliberately underfitted and walk over in small steps. The price is that its training loss is monotone decreasing and is structurally incapable of telling you when to stop, which is why every boosting library has a parameter for held-out data - and why none of them require one, since they will all happily run to the end.'},
      legend:[['#ff9736', {zh:'目前的模型 F', en:'the model F so far'}],
              ['#3fe0dd', {zh:'訓練資料', en:'the training data'}],
              ['#ff5c5c', {zh:'還沒解釋掉的殘差', en:'the residual still unexplained'}],
              ['#2f5661', {zh:'y = 0', en:'y = 0'}]],
      get code(){ return CODE_BOOST; },
      build:boostFrames
    },
    {
      id:'paths', label:{zh:'LeetCode 112 與 257', en:'LeetCode 112 and 257'},
      stage:{zh:'一條路徑是一個預測，全部的路徑就是整個模型',
             en:'one path is one prediction; all the paths are the model'},
      view:VIEW,
      variants:[{zh:'112 Path Sum', en:'112 Path Sum'}, {zh:'257 Binary Tree Paths', en:'257 Binary Tree Paths'}],
      idea:{zh:'兩題都在問根到葉的路徑，而那正是一棵決策樹的預測（一條路徑）和整個模型（所有路徑）的形狀。112 的坑在葉子的定義：葉子是「沒有小孩的節點」，不是「下一步踩到 None」。把 None 當成葉子來判斷剩下的 target，會讓只有一個小孩的節點被算成葉子，於是 [1, 2] 配 target 1 會回報 True（正確答案是 False）。3,000 棵隨機小樹裡兩種寫法差 121 次，4.0%。決策樹不會踩到這個坑，因為它建的每個內部節點都剛好有兩個小孩。257 的坑在累積器：在葉子把那個還會繼續長的 list 直接存進去，回來的就是它的最終狀態——[1, 2, 3, null, 5] 會得到兩份一模一樣的 \'1->2->5->3\'。這跟 Day 39 回溯裡 res.append(path) 沒複製是同一個 bug。要嘛進入時複製一份，要嘛在遞迴前後 append / pop。',
            en:'Both problems are about root-to-leaf paths, which is the shape of a decision tree\'s prediction (one path) and of the whole model (all the paths). The trap in 112 is the definition of a leaf: a leaf is a node with no children, not a step that landed on None. Testing the remainder when the recursion runs off the tree counts a one-child node as a leaf, so [1, 2] with target 1 returns True when the answer is False. Over 3,000 random small trees the two spellings differ 121 times, 4.0%. A decision tree never hits it, because every internal node it builds gets exactly two children. The trap in 257 is the accumulator: appending the live list at a leaf stores something that keeps growing, so [1, 2, 3, null, 5] comes back as two identical copies of \'1->2->5->3\'. That is Day 39\'s res.append(path) bug without the copy, moved into a tree. Either copy on the way in, or append and pop around the recursive calls.'},
      legend:[['#ff9736', {zh:'目前走到的節點', en:'the node being visited'}],
              ['#3fe0dd', {zh:'樹的節點', en:'the nodes of the tree'}],
              ['#ff5c5c', {zh:'不存在的小孩', en:'the child that does not exist'}],
              ['#2f5661', {zh:'邊', en:'edges'}]],
      get code(){ return CODE_LC; },
      build:lcFrames
    }
  ]
};

/* =========================================================================
   100 Days of Python - shared demo engine
   palette: teal/blue-green base, purple pointers, orange "current step"
   ========================================================================= */
const T = {
  title:{zh:DAY_META.title.zh, en:DAY_META.title.en},
  sub:{zh:DAY_META.sub.zh, en:DAY_META.sub.en},
  play:{zh:'▶ 播放', en:'▶ Play'}, pause:{zh:'❚❚ 暫停', en:'❚❚ Pause'},
  speed:{zh:'速度', en:'Speed'}, state:{zh:'演算法狀態', en:'Algorithm state'},
  code:{zh:'程式碼', en:'Code'}, idea:{zh:'重點', en:'The idea'}
};
let LANG = 'zh';
const tr = o => (o == null ? '' : (typeof o === 'string' ? o : (o[LANG] != null ? o[LANG] : o.en)));
const $ = id => document.getElementById(id);

function setLang(l){
  LANG = l;
  document.documentElement.lang = l === 'zh' ? 'zh-Hant' : 'en';
  $('btn-zh').classList.toggle('on', l === 'zh');
  $('btn-en').classList.toggle('on', l === 'en');
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (T[k]) el.textContent = tr(T[k]);
  });
  buildTabs(); render();
}

/* --------------------------------------------------------------- palette */
const STY = {
  idle :{fill:'#07293a', stroke:'#0a6b74', text:'#dff2f5', w:.045, glow:0},
  soft :{fill:'#052330', stroke:'#0a6b74', text:'#a8c8d0', w:.035, glow:0},
  hot  :{fill:'#3a2109', stroke:'#ff9736', text:'#ffbe6b', w:.075, glow:.65},
  act  :{fill:'#241542', stroke:'#9d6bff', text:'#c7a6ff', w:.070, glow:.55},
  ok   :{fill:'#08414a', stroke:'#3fe0dd', text:'#d9ffff', w:.070, glow:.5},
  done :{fill:'#062430', stroke:'#2f5661', text:'#7f9aa3', w:.035, glow:0},
  bad  :{fill:'#3a0d0d', stroke:'#ff5c5c', text:'#ff9a9a', w:.070, glow:.4},
  ghost:{fill:'none',    stroke:'#2f5661', text:'#7f9aa3', w:.035, glow:0, dash:'.12 .10'}
};
const COL = {teal:'#12b3b8', tealL:'#3fe0dd', purple:'#9d6bff', purpleL:'#c7a6ff',
             orange:'#ff9736', orangeL:'#ffbe6b', red:'#ff5c5c', grey:'#8fa3ac',
             pale:'#dff2f5', white:'#ffffff'};
const LEGEND = {
  hot:[COL.orange, {zh:'目前這一步', en:'current step'}],
  act:[COL.purple, {zh:'指標 / 走訪位置', en:'pointer / cursor'}],
  ok:[COL.tealL,  {zh:'完成 / 結果', en:'done / result'}],
  bad:[COL.red,   {zh:'失敗 / 要避開的寫法', en:'failure / the wrong way'}],
  done:['#2f5661', {zh:'已處理完', en:'already done'}],
  soft:['#0a6b74', {zh:'其他元素', en:'other items'}],
  idle:['#0a6b74', {zh:'尚未處理', en:'untouched'}],
  ghost:['#123f4d', {zh:'尚未處理', en:'not yet reached'}]
};
const leg = (...keys) => keys.map(k => LEGEND[k]);

/* ---------------------------------------------------------- frame buffer */
function Frames(){
  this.list = [];
  this.push = (o) => this.list.push({
    shapes:JSON.parse(JSON.stringify(o.shapes || [])),
    panels:JSON.parse(JSON.stringify(o.panels || [])),
    view:(o.view || null), line:(o.line == null ? 0 : o.line), msg:o.msg
  });
}
/* shape helpers - engines call these, the renderer just draws */
const S = {
  r:(x, y, w, h, s, lab, o) => Object.assign({t:'r', x:x, y:y, w:w, h:h, s:s || 'idle', lab:lab}, o || {}),
  c:(x, y, r, s, lab, o) => Object.assign({t:'c', x:x, y:y, r:r, s:s || 'idle', lab:lab}, o || {}),
  e:(x1, y1, x2, y2, o) => Object.assign({t:'e', x1:x1, y1:y1, x2:x2, y2:y2}, o || {}),
  t:(x, y, s, o) => Object.assign({t:'t', x:x, y:y, s:s}, o || {})
};
/* a labelled row of array cells; returns shapes */
function cellRow(vals, x0, y, w, h, opt){
  opt = opt || {};
  const out = [], st = opt.states || {};
  vals.forEach((v, i) => {
    out.push(S.r(x0 + i * w, y, w - (opt.gap == null ? .06 : opt.gap), h, st[i] || 'idle',
                 v == null ? '' : String(v), {fs:opt.fs || h * .52}));
    if (opt.index !== false)
      out.push(S.t(x0 + i * w + (w - .06) / 2, y + h + (opt.ilift || .34),
                   opt.labels ? opt.labels[i] : String(i),
                   {c:st[i] && st[i] !== 'idle' ? COL.orangeL : COL.grey, fs:opt.ifs || .30}));
  });
  if (opt.title) out.push(S.t(x0 - .22, y + h * .62, opt.title, {c:COL.tealL, fs:.32, anchor:'end'}));
  return out;
}
/* binary-tree layout from a heap-style array (index 0 = root, 2i+1 / 2i+2) */
function heapTreeShapes(arr, x0, y0, w, rowH, states, opt){
  opt = opt || {};
  const n = arr.length, out = [], R = opt.r || .34;
  const depth = i => Math.floor(Math.log2(i + 1));
  const maxD = n ? depth(n - 1) : 0;
  const px = i => {
    const d = depth(i), first = Math.pow(2, d) - 1, k = i - first;
    const slots = Math.pow(2, d), span = w;
    return x0 + span * (k + .5) / slots;
  };
  const py = i => y0 + depth(i) * rowH;
  for (let i = 1; i < n; i++){
    if (arr[i] == null) continue;
    const p = Math.floor((i - 1) / 2);
    out.push(S.e(px(p), py(p), px(i), py(i), {pad:R + .04,
      s:(states && (states[i] === 'hot' || states[i] === 'act')) ? states[i] : 'idle'}));
  }
  for (let i = 0; i < n; i++){
    if (arr[i] == null) continue;
    out.push(S.c(px(i), py(i), R, (states && states[i]) || 'idle', String(arr[i]), {fs:R * .95}));
    if (opt.showIndex)
      out.push(S.t(px(i), py(i) + R + .34, String(i), {c:COL.grey, fs:.26}));
  }
  return out;
}

/* -------------------------------------------------------------- renderer */
const svgNS = 'http://www.w3.org/2000/svg';
const mk = (tag, a) => { const e = document.createElementNS(svgNS, tag);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const clear = svg => { while (svg.firstChild) svg.removeChild(svg.firstChild); };

function defs(svg){
  const d = mk('defs', {});
  const f = mk('filter', {id:'glow', x:'-70%', y:'-70%', width:'240%', height:'240%'});
  f.appendChild(mk('feGaussianBlur', {stdDeviation:'.055', result:'b'}));
  const m = mk('feMerge', {});
  m.appendChild(mk('feMergeNode', {in:'b'}));
  m.appendChild(mk('feMergeNode', {in:'SourceGraphic'}));
  f.appendChild(m); d.appendChild(f);
  Object.keys(STY).forEach(k => {
    const mk2 = mk('marker', {id:'ar-' + k, viewBox:'0 0 10 10', refX:'8.5', refY:'5',
      markerWidth:'5.2', markerHeight:'5.2', orient:'auto-start-reverse'});
    mk2.appendChild(mk('path', {d:'M 0 1 L 9 5 L 0 9 z', fill:STY[k].stroke}));
    d.appendChild(mk2);
  });
  svg.appendChild(d);
}

function drawShape(svg, sh){
  const st = STY[sh.s || 'idle'];
  if (sh.t === 'e'){
    let {x1, y1, x2, y2} = sh;
    if (sh.pad){
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
      x1 += dx / L * sh.pad; y1 += dy / L * sh.pad;
      x2 -= dx / L * sh.pad; y2 -= dy / L * sh.pad;
    }
    const a = {x1:x1.toFixed(3), y1:y1.toFixed(3), x2:x2.toFixed(3), y2:y2.toFixed(3),
      stroke:st.stroke, 'stroke-width':(sh.w || st.w || .05), 'stroke-linecap':'round',
      opacity:(sh.o == null ? (sh.s && sh.s !== 'idle' ? 1 : .75) : sh.o)};
    if (sh.dash || st.dash) a['stroke-dasharray'] = sh.dash || st.dash;
    if (sh.arrow !== false) a['marker-end'] = 'url(#ar-' + (sh.s || 'idle') + ')';
    svg.appendChild(mk('line', a));
    if (sh.lab != null)
      svg.appendChild(txt((x1 + x2) / 2 + (sh.lx || 0), (y1 + y2) / 2 + (sh.ly || -.16),
        tr(sh.lab), st.stroke, sh.fs || .30, 'middle'));
    return;
  }
  if (sh.t === 't'){
    svg.appendChild(txt(sh.x, sh.y, tr(sh.s), sh.c || COL.pale, sh.fs || .32,
      sh.anchor || 'middle', sh.o));
    return;
  }
  let node;
  if (sh.t === 'c'){
    node = mk('circle', {cx:sh.x.toFixed(3), cy:sh.y.toFixed(3), r:sh.r.toFixed(3),
      fill:st.fill, stroke:st.stroke, 'stroke-width':st.w});
  } else {
    node = mk('rect', {x:sh.x.toFixed(3), y:sh.y.toFixed(3), width:sh.w.toFixed(3),
      height:sh.h.toFixed(3), rx:(sh.rx == null ? .09 : sh.rx), fill:st.fill,
      stroke:st.stroke, 'stroke-width':st.w});
  }
  if (st.dash || sh.dash) node.setAttribute('stroke-dasharray', sh.dash || st.dash);
  if (st.glow) node.setAttribute('filter', 'url(#glow)');
  if (sh.o != null) node.setAttribute('opacity', sh.o);
  svg.appendChild(node);
  const cx = sh.t === 'c' ? sh.x : sh.x + sh.w / 2;
  const cy = sh.t === 'c' ? sh.y : sh.y + sh.h / 2;
  if (sh.lab != null && sh.lab !== '')
    svg.appendChild(txt(cx + (sh.dx || 0), cy + (sh.fs || .40) * .35, tr(sh.lab), st.text,
      sh.fs || .40, 'middle'));
  if (sh.sub != null)
    svg.appendChild(txt(cx, cy + (sh.t === 'c' ? sh.r : sh.h) + .34, tr(sh.sub),
      sh.subc || COL.grey, sh.subfs || .28, 'middle'));
  if (sh.top != null)
    svg.appendChild(txt(cx, cy - (sh.t === 'c' ? sh.r : sh.h / 2) - .22, tr(sh.top),
      sh.topc || COL.purpleL, sh.topfs || .28, 'middle'));
}
function txt(x, y, s, c, fs, anchor, o){
  const t = mk('text', {x:(+x).toFixed(3), y:(+y).toFixed(3), 'text-anchor':anchor || 'middle',
    fill:c, 'font-size':(+fs).toFixed(3)});
  if (o != null) t.setAttribute('opacity', o);
  t.textContent = s;
  return t;
}

function renderStage(frame){
  const svg = $('stage');
  clear(svg); defs(svg);
  const v = frame.view || curTab().view || [10, 6.4];
  svg.setAttribute('viewBox', '0 0 ' + v[0] + ' ' + v[1]);
  (frame.shapes || []).forEach(sh => drawShape(svg, sh));
}

function drawPanels(frame){
  const box = $('panels'); box.innerHTML = '';
  (frame.panels || []).forEach(p => {
    const d = document.createElement('div'); d.className = 'panel';
    const l = document.createElement('div'); l.className = 'lbl'; l.textContent = tr(p.lbl);
    const c = document.createElement('div'); c.className = 'chips';
    (p.chips.length ? p.chips : [{t:'—', cls:'empty'}]).forEach(ch => {
      const s = document.createElement('span');
      s.className = 'chip ' + (ch.cls || ''); s.textContent = ch.t; c.appendChild(s);
    });
    d.appendChild(l); d.appendChild(c); box.appendChild(d);
  });
}
function drawCode(frame){
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  $('code').innerHTML = curTab().code.map((ln, i) => {
    const h = esc(ln).replace(/(#.*)$/, '<span class="cm">$1</span>');
    return '<span class="ln' + (i === frame.line ? ' on' : '') + '">' + (h || ' ') + '</span>';
  }).join('');
}
function drawLegend(){
  const keys = curTab().legend || ['hot', 'act', 'ok', 'idle'];
  $('legend').innerHTML = keys.map(k => {
    const [c, txt] = Array.isArray(k) ? k : LEGEND[k];
    return '<span><i style="background:' + c + '"></i>' + tr(txt) + '</span>';
  }).join('');
}

let tabIx = 0, varIx = 0, frames = [], cur = 0, timer = null;
const TABS = DAY_META.tabs;
const curTab = () => TABS[tabIx];

function render(){
  if (!frames.length) return;
  cur = Math.max(0, Math.min(cur, frames.length - 1));
  const f = frames[cur];
  $('stage-title').textContent = tr(curTab().stage);
  renderStage(f); drawPanels(f); drawCode(f); drawLegend();
  $('narr').innerHTML = tr(f.msg);
  $('idea').innerHTML = tr(curTab().idea);
  $('stepno').textContent = (cur + 1) + ' / ' + frames.length;
  $('prev').disabled = cur === 0;
  $('next').disabled = cur === frames.length - 1;
}
function buildTabs(){
  $('tabs').innerHTML = '';
  TABS.forEach((t, i) => {
    const b = document.createElement('button');
    b.textContent = tr(t.label);
    if (i === tabIx) b.classList.add('on');
    b.onclick = () => { tabIx = i; varIx = 0; load(); };
    $('tabs').appendChild(b);
  });
  const ex = $('extra'); ex.innerHTML = '';
  const vs = curTab().variants;
  if (vs) vs.forEach((v, i) => {
    const b = document.createElement('button');
    b.textContent = tr(v);
    if (i === varIx) b.classList.add('on');
    b.onclick = () => { varIx = i; load(); };
    ex.appendChild(b);
  });
}
function load(){
  stop();
  frames = curTab().build(varIx) || [];
  cur = 0; buildTabs(); render();
}
function step(d){ stop(); cur += d; render(); }
function reset(){ stop(); cur = 0; render(); }
function stop(){ if (timer){ clearInterval(timer); timer = null; $('play').textContent = tr(T.play); } }
function togglePlay(){
  if (timer){ stop(); return; }
  if (cur >= frames.length - 1) cur = 0;
  $('play').textContent = tr(T.pause);
  timer = setInterval(() => {
    if (cur >= frames.length - 1){ stop(); return; }
    cur++; render();
  }, Number($('speed').value));
}
$('speed').addEventListener('input', () => { if (timer){ stop(); togglePlay(); } });
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight'){ step(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft'){ step(-1); e.preventDefault(); }
  else if (e.key === ' '){ togglePlay(); e.preventDefault(); }
});
setLang('zh');
load();
