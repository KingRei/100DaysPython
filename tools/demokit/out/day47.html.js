// DAY: 47
// TITLE_ZH: k-means 與 k-nearest neighbours：兩個只有距離函數和一條規則的演算法，以及八種不會報錯的壞法
// TITLE_EN: k-means and k-nearest neighbours: two algorithms that are a distance function plus a rule, and eight ways to break them without an error
// SUB_ZH: 規則各自只有一兩行，而且幾乎不會寫錯；會咬人的東西全部住在距離函數裡——而距離函數從來不丟例外，它只是安靜地回答另一個問題。目標函數永遠沒辦法告訴你 k 是多少、隨機起點會換掉答案、空掉的 centroid 會讓整份資料塌成一團 nan、欄位的單位決定了分群結果、update 規則配錯距離就沒有終止證明、平方距離只認得圓盤、k=1 只要點兩兩相異就一定在自己的訓練集上拿 1.000、而多加幾百個雜訊欄位會讓「最近」失去意義。
// SUB_EN: Each rule is one or two lines and almost impossible to get wrong; everything that bites lives in the distance function - which never raises, it just quietly answers a different question. The objective can never tell you k, the random start changes the answer, an emptied centroid collapses the whole data set into nan, a column's units decide the clustering, an update rule paired with the wrong distance has no termination proof, squared distance only knows discs, k=1 scores 1.000 on its own training set whenever the points are distinct, and a few hundred noise columns make "nearest" stop meaning anything.
// FOLDER: day%2047%20-%20k-means%20and%20knn
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

/* ================================================================ shared maths */
const d2 = (a, b) => (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
function assign(X, C){ return X.map(p => {
  let bi = 0, bd = Infinity;
  C.forEach((c, j) => { const d = d2(p, c); if (d < bd){ bd = d; bi = j; } });
  return bi; }); }
function inertia(X, C, lab){ return X.reduce((a, p, i) => a + d2(p, C[lab[i]]), 0); }
function means(X, lab, C){
  return C.map((c, j) => {
    const m = X.filter((_, i) => lab[i] === j);
    if (!m.length) return c.slice();
    return [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length];
  });
}
// Lloyd, recorded one half-step at a time: the point of the tab is that the
// assignment and the update are two different arguments for the same decrease.
function lloydTrace(X, C0, maxIter){
  let C = C0.map(c => c.slice());
  const steps = [];
  for (let it = 0; it < (maxIter || 12); it++){
    const lab = assign(X, C);
    steps.push({kind:'assign', C:C.map(c => c.slice()), lab:lab.slice(), J:inertia(X, C, lab), it:it});
    const nc = means(X, lab, C);
    const moved = nc.some((c, j) => Math.abs(c[0] - C[j][0]) > 1e-12 || Math.abs(c[1] - C[j][1]) > 1e-12);
    steps.push({kind:'update', C:nc.map(c => c.slice()), lab:lab.slice(),
                J:inertia(X, nc, lab), it:it, done:!moved});
    C = nc;
    if (!moved) break;
  }
  return steps;
}
function lloydBest(X, k){
  // every k-subset of the points as a starting set: deterministic, and the same
  // number the python module reports.
  let best = Infinity;
  const idx = [];
  (function pick(start, left){
    if (!left){
      let C = idx.map(i => X[i].slice());
      for (let it = 0; it < 30; it++){
        const lab = assign(X, C), nc = means(X, lab, C);
        if (nc.every((c, j) => Math.abs(c[0] - C[j][0]) < 1e-12 && Math.abs(c[1] - C[j][1]) < 1e-12)) break;
        C = nc;
      }
      best = Math.min(best, inertia(X, C, assign(X, C)));
      return;
    }
    for (let i = start; i <= X.length - left; i++){ idx.push(i); pick(i + 1, left - 1); idx.pop(); }
  })(0, k);
  return best;
}

/* ================================================================ tab 1: Lloyd */
const P12 = [[1.0,1.0],[1.6,1.2],[1.2,1.8],[1.9,1.0],[5.0,1.2],[5.6,1.0],
             [5.2,1.9],[4.8,1.6],[3.0,4.6],[3.6,4.9],[2.8,5.3],[3.4,4.2]];
const LIM12 = [[0.4, 0.4], [6.2, 5.9]];
const INIT_GOOD = [1, 4, 5], INIT_BAD = [8, 9, 11];
const CODE_LLOYD = [
  'def kmeans(X, k, C):',
  '    while True:',
  '        labels = np.argmin(sq_dists(X, C), axis=1)   # assign',
  '        new = np.array([X[labels == j].mean(axis=0)  # update',
  '                        for j in range(k)])',
  '        if np.allclose(new, C):',
  '            return C, labels',
  '        C = new'
];
const CLS = ['ok', 'act', 'done'];
// up to six clusters need six distinguishable outlines, not three repeated
const CLS6 = ['ok', 'act', 'done', 'idle', 'bad', 'ghost'];
function lloydStage(st, o){
  const sh = [], plot = new Plot(.45, .55, 5.0, 4.4, LIM12);
  plot.frame(sh);
  dots(sh, plot, P12, P12.map((_, i) => st.lab ? CLS[st.lab[i]] : 'idle'));
  crosses(sh, plot, st.C, st.C.map(() => 'hot'), st.C.map((_, j) => String(j)));
  side(sh, 5.75, 1.15, o.side, .46);
  return caps(sh, o);
}
function lloydFrames(v){
  const F = new Frames(), init = (v === 0 ? INIT_GOOD : INIT_BAD).map(i => P12[i].slice());
  const steps = lloydTrace(P12, init);
  const lastJ = steps[steps.length - 1].J;
  steps.forEach((st, i) => {
    const prev = i ? steps[i - 1].J : null;
    const drop = prev === null ? null : prev - st.J;
    const isA = st.kind === 'assign';
    F.push({
      shapes:lloydStage(st, {
        cap:{zh:(isA ? '分配：每個點加入最近的 centroid' : '更新：每個 centroid 移到成員的平均'),
             en:(isA ? 'assign: every point joins its nearest centroid'
                     : 'update: every centroid moves to the mean of its members')},
        cap2:{zh:'inertia（平方距離總和）= ' + fmtN(st.J) + (drop === null ? '' : '，降了 ' + fmtN(drop)),
              en:'inertia (sum of squared distances) = ' + fmtN(st.J) +
                 (drop === null ? '' : ', down by ' + fmtN(drop))},
        side:[
          [{zh:'第 ' + (st.it + 1) + ' 輪', en:'round ' + (st.it + 1)}, COL.purpleL, .30],
          [{zh:'每群大小 ' + [0,1,2].map(j => st.lab.filter(l => l === j).length).join(' / '),
            en:'cluster sizes ' + [0,1,2].map(j => st.lab.filter(l => l === j).length).join(' / ')},
           COL.grey, .26]
        ]
      }),
      panels:[{lbl:'inertia', chips:[chip(fmtN(st.J), st.done ? 'ok' : 'hot')]},
              {lbl:'sizes', chips:[0,1,2].map(j =>
                chip(String(st.lab.filter(l => l === j).length), CLS[j]))}],
      line:isA ? 2 : 3,
      msg:st.done
        ? (v === 0
           ? {zh:'沒有 centroid 再移動，所以分配也不會再變——演算法停了，inertia ' + fmtN(lastJ) + '。注意它停下來的理由不是「找到最好的答案」，只是「沒東西再變了」。',
              en:'No centroid moved, so no assignment can change either - the algorithm stops at inertia ' + fmtN(lastJ) + '. Note that the reason it stopped is not "this is the best answer", only "nothing changed".'}
           : {zh:'一樣停了，一樣不報錯，inertia ' + fmtN(lastJ) + ' 卻是好起點的十倍以上。兩個 centroid 掉進同一群，所以上面那群被切成兩半、下面兩群被合成一團——這就是為什麼 k-means 要重跑好幾次再挑 inertia 最小的那次。',
              en:'It stops the same way and complains about nothing, at an inertia more than ten times the good start. Two centroids landed in the same group, so that group was split in half and the other two were merged - which is why k-means is run several times and the lowest inertia kept.'})
        : (isA
           ? {zh:'分配這一步一定讓 inertia 下降或持平：每個點都自己挑了最便宜的 centroid，沒有人能挑到更差的。',
              en:'The assignment step can only lower the objective or leave it alone: every point picks its own cheapest centroid, so nobody can end up worse off.'}
           : {zh:'更新這一步也一定讓 inertia 下降或持平，理由完全不同：平均數就是「平方距離總和」的最小值點。這兩個理由合起來就是 Lloyd 的終止證明——而第六個分頁會把其中一個拆掉。',
              en:'The update step also only lowers it, for a completely different reason: the mean is the minimiser of the sum of squared distances. Those two arguments together are Lloyd\'s termination proof - and one of them gets taken away in the sixth tab.'})
    });
  });
  return F.list;
}

/* ================================================================ tab 2: choosing k */
const U12 = [[4.83,4.85],[3.09,1.71],[0.32,2.3],[2.45,0.27],[0.29,6.0],[3.91,1.41],
             [2.61,5.85],[5.39,5.07],[2.35,2.96],[4.06,0.36],[3.33,1.63],[5.28,0.39]];
const CODE_K = [
  'for k in range(1, 7):',
  '    C, labels, J = kmeans(X, k)',
  '    print(k, J)          # J falls every single time',
  '',
  '# J(k=n) = 0 exactly, so "minimise J" always answers k = n.',
  '# the elbow is a judgement call about where falling stops',
  '# being worth it - it is not a computation.'
];
function kStage(vals, step, pts, o){
  const sh = [], plot = new Plot(.45, 1.35, 3.5, 3.30, [[0.1, 0.1], [6.2, 6.2]]);
  plot.frame(sh);
  const lab = o.lab;
  dots(sh, plot, pts, pts.map((_, i) => lab ? CLS6[lab[i]] : 'soft'));
  bars(sh, 4.65, 4.65, .62, .22, vals.slice(0, step + 1),
       vals.slice(0, step + 1).map((_, i) => i === step ? 'hot' : 'done'),
       vals.slice(0, step + 1).map(v => fmtN(Math.round(v * 10) / 10)), 2.85, vals[0]);
  vals.slice(0, step + 1).forEach((_, i) =>
    sh.push(fitT(4.65 + i * .84 + .31, 4.95, 'k=' + (i + 1), {c:COL.grey, fs:.24})));
  side(sh, .45, .60, o.side, .42);
  return caps(sh, o);
}
function kFrames(v){
  const F = new Frames(), pts = v === 0 ? P12 : U12;
  const vals = [1, 2, 3, 4, 5, 6].map(k => lloydBest(pts, k));
  const drops = vals.slice(1).map((x, i) => 100 * (vals[i] - x) / vals[i]);
  vals.forEach((J, i) => {
    let C = null, lab = null;
    // rebuild one good partition for this k so the scatter matches the bar
    {
      let bestJ = Infinity;
      const pick = (start, left, acc) => {
        if (!left){
          let cc = acc.map(j => pts[j].slice());
          for (let t = 0; t < 30; t++){
            const l = assign(pts, cc), nc = means(pts, l, cc);
            if (nc.every((c, j) => Math.abs(c[0] - cc[j][0]) < 1e-12 && Math.abs(c[1] - cc[j][1]) < 1e-12)) break;
            cc = nc;
          }
          const l = assign(pts, cc), jj = inertia(pts, cc, l);
          if (jj < bestJ - 1e-12){ bestJ = jj; C = cc; lab = l; }
          return;
        }
        for (let s = start; s <= pts.length - left; s++) pick(s + 1, left - 1, acc.concat([s]));
      };
      pick(0, i + 1, []);
    }
    F.push({
      shapes:kStage(vals, i, pts, {
        lab:lab,
        cap:{zh:'k = ' + (i + 1) + '：inertia ' + fmtN(Math.round(J * 1000) / 1000),
             en:'k = ' + (i + 1) + ': inertia ' + fmtN(Math.round(J * 1000) / 1000)},
        cap2:i === 0 ? {zh:'k = 1 的 inertia 就是資料對自己平均數的離散程度，跟演算法無關',
                        en:'the k = 1 inertia is just the spread of the data around its own mean - no algorithm involved'}
                     : {zh:'比 k = ' + i + ' 少了 ' + drops[i - 1].toFixed(1) + '%',
                        en:drops[i - 1].toFixed(1) + '% lower than k = ' + i},
        side:[[v === 0 ? {zh:'三群真的存在', en:'three groups really are there'}
                       : {zh:'均勻亂數，沒有任何結構', en:'uniform random noise, no structure at all'},
               v === 0 ? COL.tealL : COL.red, .28],
              [{zh:'每個點都塗上它所屬的群', en:'each point is coloured by its cluster'}, COL.grey, .24]]
      }),
      panels:[{lbl:'inertia', chips:[chip(fmtN(Math.round(J * 10) / 10), 'hot')]},
              {lbl:'drop', chips:[chip(i === 0 ? '-' : drops[i - 1].toFixed(1) + '%',
                                       i === 0 ? 'dim' : (drops[i - 1] > 70 ? 'ok' : 'dim'))]}],
      line:i === 5 ? 4 : 1,
      msg:i === 0
        ? {zh:'k = 1 時每個點都屬於同一群，inertia 就是整份資料的離散程度。接下來每加一個 centroid，這個數字只會往下走。',
           en:'At k = 1 every point is in the same cluster and the inertia is just the spread of the data. From here, every extra centroid can only push the number down.'}
        : (v === 0 && i === 2)
          ? {zh:'k = 2 → 3 一口氣砍掉 ' + drops[1].toFixed(1) + '%，而下一步只砍掉 ' + drops[2].toFixed(1) + '%。這個落差就是「手肘」，但注意它是我們看圖看出來的，不是演算法算出來的。',
             en:'k = 2 -> 3 removes ' + drops[1].toFixed(1) + '% and the next step only ' + drops[2].toFixed(1) + '%. That gap is the elbow - but notice we found it by looking, not by computing.'}
          : (v === 1 && i === 2)
            ? {zh:'同一份演算法、同一條往下的曲線——只是這份資料裡什麼都沒有。' + drops[1].toFixed(1) + '% 之後是 ' + drops[2].toFixed(1) + '%，沒有手肘，而它照樣交給你三群長得很整齊的雜訊。',
               en:'Same algorithm, same falling curve - except there is nothing in this data. ' + drops[1].toFixed(1) + '% is followed by ' + drops[2].toFixed(1) + '%, there is no elbow, and it still hands back three tidy clusters of noise.'}
            : {zh:'再加一個 centroid，inertia 再降一點。到 k = 12（每個點自己一群）時它會剛好是 0——所以最小化這個目標函數永遠答 k = n。',
               en:'One more centroid, one more drop. At k = 12 - one cluster per point - it would be exactly 0, which is why minimising this objective always answers k = n.'}
    });
  });
  return F.list;
}

/* ================================================================ tab 3: an empty cluster */
const XE = [[0.0,0.0],[0.5,0.45],[0.9,0.05],[5.0,5.0],[5.4,4.6]];
const CE = [[0.45,0.17],[2.6,2.6],[5.2,4.8]];
const LIM_E = [[-0.5, -0.5], [6.0, 5.6]];
const CODE_EMPTY = [
  'for j in range(k):',
  '    members = X[labels == j]',
  '    C[j] = members.mean(axis=0)   # empty -> nan, and a warning',
  '',
  '# np.argmin does NOT skip nan: it returns its index.',
  '# scikit-learn moves the orphan to the point that is',
  '# furthest from any centroid.  Doing nothing is the bug.'
];
function emptyStage(o){
  const sh = [], plot = new Plot(.45, .55, 4.3, 4.4, LIM_E);
  plot.frame(sh);
  dots(sh, plot, XE, XE.map((_, i) => o.ptSt ? o.ptSt[i] : 'soft'), null, .15);
  crosses(sh, plot, o.cen || CE, (o.cen || CE).map((_, j) => o.cenSt ? o.cenSt[j] : 'hot'),
          (o.cen || CE).map((_, j) => String(j)), CE);
  side(sh, 5.1, 1.0, o.side, .46);
  return caps(sh, o);
}
function emptyFrames(){
  const F = new Frames();
  const lab = assign(XE, CE);
  const counts = [0, 1, 2].map(j => lab.filter(l => l === j).length);
  const NANC = [[0.15, 0.05], [NaN, NaN], [5.05, 5.05]];
  const steps = [
    {ptSt:XE.map(() => 'soft'), cenSt:['hot', 'hot', 'hot'],
     cap:{zh:'5 個點分成兩堆，3 個 centroid', en:'5 points in two groups, 3 centroids'},
     cap2:{zh:'中間那個不是任何點的最近鄰，這就是被擠出去的 centroid',
           en:'the middle one is nearest to nothing: a centroid that got squeezed out'},
     side:[[{zh:'centroid 1 在 (2.5, 2.5)', en:'centroid 1 sits at (2.5, 2.5)'}, COL.purpleL, .28]],
     line:0,
     msg:{zh:'這不是刻意安排的壞例子：只要有一輪 centroid 移動得夠多，就會有一個被擠到沒人要的位置。',
          en:'This is not a contrived setup: as soon as one round moves the centroids far enough, one of them can end up somewhere nobody wants.'}},
    {ptSt:lab.map(l => CLS[l]), cenSt:['ok', 'bad', 'ok'],
     cap:{zh:'分配結果 [' + lab.join(', ') + ']：cluster 1 拿到 ' + counts[1] + ' 個點',
          en:'assignment [' + lab.join(', ') + ']: cluster 1 got ' + counts[1] + ' points'},
     cap2:{zh:'兩個群各拿到 ' + counts[0] + ' 和 ' + counts[2] + ' 個點',
           en:'the other two took ' + counts[0] + ' and ' + counts[2] + ' points'},
     side:[[{zh:'空的群：members.mean() → nan', en:'empty cluster: members.mean() -> nan'}, COL.red, .28],
           [{zh:'numpy 只說 "Mean of empty slice."', en:'numpy only says "Mean of empty slice."'}, COL.grey, .24],
           [{zh:'而且是寫到 stderr 的 warning', en:'as a warning, on stderr'}, COL.grey, .24]],
     line:2,
     msg:{zh:'空集合的平均是 nan。numpy 發一個 RuntimeWarning，而一個 warning 本來就很容易淹沒在一整頁輸出裡。',
          en:'The mean of an empty set is nan. numpy emits a RuntimeWarning, and a single warning is easy to lose in a page of output.'}},
    {ptSt:XE.map(() => 'bad'), cen:NANC, cenSt:['done', 'bad', 'done'],
     cap:{zh:'下一輪的距離：每個點到 centroid 1 都是 nan', en:'next round: every distance to centroid 1 is nan'},
     cap2:{zh:'np.argmin([0.246, nan, 50.08]) = 1 —— 它回傳 nan 的位置，不是跳過它',
           en:'np.argmin([0.246, nan, 50.08]) = 1 - it returns the index of the nan, it does not skip it'},
     side:[[{zh:'新的分配：[1, 1, 1, 1, 1]', en:'new assignment: [1, 1, 1, 1, 1]'}, COL.red, .28],
           [{zh:'全部的點都加入了那個死掉的 centroid', en:'every point joined the dead centroid'}, COL.grey, .24]],
     line:4,
     msg:{zh:'這才是真正的災難：nan 不會被忽略，它會贏。argmin 把 nan 當成最小值回傳，於是整份資料在一輪之內塌進同一群。',
          en:'This is where it really goes wrong: the nan is not ignored, it wins. argmin returns it as the minimum, and the whole data set collapses into one cluster in a single round.'}},
    {ptSt:XE.map(() => 'bad'), cen:[[NaN, NaN], [2.36, 2.02], [NaN, NaN]], cenSt:['bad', 'done', 'bad'],
     cap:{zh:'再更新一次：模型剩下一個 centroid 加兩個 nan',
          en:'one more update: the model is one centroid plus two nans'},
     cap2:{zh:'[[nan, nan], [2.36, 2.02], [nan, nan]] —— 但 len(C) 還是 3',
           en:'[[nan, nan], [2.36, 2.02], [nan, nan]] - and len(C) is still 3'},
     side:[[{zh:'回報 3 群，實際上只有 1 群', en:'reports 3 clusters, actually has 1'}, COL.red, .28],
           [{zh:'整段崩潰只有一個 stderr warning', en:'the whole collapse is one stderr warning'}, COL.grey, .24]],
     line:6,
     msg:{zh:'下游拿到的是一個形狀完全正確的 (3, 2) 陣列，裡面有兩列 nan。改用 np.nanargmin 只是把問題藏得更深一層——真正的修法是把失去成員的 centroid 搬到離所有 centroid 最遠的那個點。',
          en:'Downstream gets a perfectly shaped (3, 2) array with two rows of nan in it. Switching to np.nanargmin only hides the problem one level deeper - the real fix is to move the orphaned centroid to the point furthest from any centroid.'}}
  ];
  steps.forEach(s => F.push({
    shapes:emptyStage(s),
    panels:[{lbl:'cluster sizes', chips:(s.ptSt[0] === 'soft'
      ? ['-', '-', '-'] : (s.cen ? ['0', '5', '0'] : counts.map(String)))
      .map((t, j) => chip(t, t === '0' ? 'bad' : 'ok'))}],
    line:s.line, msg:s.msg}));
  return F.list;
}

/* ================================================================ tab 4: units */
const REQ = [[400,0.14],[1500,0.18],[2600,0.12],[3700,0.17],
             [600,0.48],[1700,0.52],[2900,0.47],[3500,0.53],
             [500,0.86],[1600,0.83],[2700,0.88],[3800,0.84]];
const TRUTH = [0,0,0,0,1,1,1,1,2,2,2,2];
const CODE_UNITS = [
  'X = np.column_stack([prompt_tokens, hit_rate])',
  'labels = kmeans(X, 3)[1]        # cuts on tokens only',
  '',
  'Xz = (X - X.mean(0)) / X.std(0)  # "every column matters equally"',
  'Xh = X[:, 1:2]                   # "only this column matters"',
  '',
  '# scaling is not a fix, it is a choice of weights -',
  '# and no amount of it can tell you which columns matter.'
];
function bestLabels(D, k){
  let bestJ = Infinity, bl = null;
  const pick = (start, left, acc) => {
    if (!left){
      let cc = acc.map(j => D[j].slice());
      for (let t = 0; t < 40; t++){
        const l = assign(D, cc), nc = means(D, l, cc);
        if (nc.every((c, j) => Math.abs(c[0] - cc[j][0]) < 1e-12 && Math.abs(c[1] - cc[j][1]) < 1e-12)) break;
        cc = nc;
      }
      const l = assign(D, cc), jj = inertia(D, cc, l);
      if (jj < bestJ - 1e-12){ bestJ = jj; bl = l; }
      return;
    }
    for (let s = start; s <= D.length - left; s++) pick(s + 1, left - 1, acc.concat([s]));
  };
  pick(0, k, []);
  return bl;
}
function purity(lab, truth, k){
  let ok = 0;
  for (let j = 0; j < k; j++){
    const c = [0, 0, 0];
    truth.forEach((t, i) => { if (lab[i] === j) c[t]++; });
    ok += Math.max.apply(null, c);
  }
  return ok / truth.length;
}
const UNIT_VIEWS = [
  {id:0, name:{zh:'原始單位', en:'raw units'}},
  {id:1, name:{zh:'標準化之後', en:'standardised'}},
  {id:2, name:{zh:'只用命中率', en:'hit rate only'}}
];
function unitsFrames(v){
  const F = new Frames();
  const col = j => REQ.map(r => r[j]);
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const std = a => Math.sqrt(mean(a.map(x => (x - mean(a)) * (x - mean(a)))));
  const t = col(0), h = col(1);
  const D = v === 0 ? REQ
          : v === 1 ? REQ.map(r => [(r[0] - mean(t)) / std(t), (r[1] - mean(h)) / std(h)])
                    : REQ.map(r => [0, r[1]]);
  const lab = bestLabels(D, 3), pu = purity(lab, TRUTH, 3);
  const plot = new Plot(.45, .60, 5.1, 4.3, [[0, 0], [4100, 1.0]]);
  const steps = [
    {show:'truth', cap:{zh:'360 筆請求的縮影：prompt token 數 200–4000，cache 命中率 0–1',
                        en:'a miniature of 360 requests: prompt tokens 200-4000, cache hit rate 0-1'},
     cap2:{zh:'三個真實族群只差在命中率：0.15、0.50、0.85',
           en:'the three real populations differ only in hit rate: 0.15, 0.50, 0.85'}},
    {show:'result', cap:{zh:UNIT_VIEWS[v].name.zh + '：purity ' + pu.toFixed(3),
                         en:UNIT_VIEWS[v].name.en + ': purity ' + pu.toFixed(3)},
     cap2:v === 0 ? {zh:'token 軸跨 3600、命中率軸跨 0.76，平方後差了兩千萬倍',
                     en:'tokens span 3600, hit rate spans 0.76 - squared, twenty million times the weight'}
        : v === 1 ? {zh:'兩欄現在都是 sd = 1，所以那欄雜訊拿到了和答案一樣的票數',
                     en:'both columns now have sd = 1, so the noise column gets exactly as many votes as the answer'}
                  : {zh:'把無關的那欄丟掉，分群就回來了', en:'drop the irrelevant column and the grouping comes back'}}
  ];
  steps.forEach((s, i) => {
    const sh = [];
    plot.frame(sh);
    dots(sh, plot, REQ, REQ.map((_, j) => s.show === 'truth' ? 'soft' : CLS[lab[j]]), null, .14);
    sh.push(fitT(.50, .40, {zh:'↑ 縱軸：cache 命中率', en:'↑ vertical axis: cache hit rate'},
                 {anchor:'start', c:COL.grey, fs:.24}));
    sh.push(fitT(3.0, 5.20, {zh:'→ 橫軸：prompt token 數', en:'→ horizontal axis: prompt tokens'},
                 {c:COL.grey, fs:.24}));
    if (s.show === 'truth')
      [0.15, 0.50, 0.85].forEach((y, j) => {
        const a = plot.px([0, y]), b = plot.px([4100, y]);
        sh.push(S.e(a[0], a[1], b[0], b[1], {s:'ghost', arrow:false, w:.02}));
        sh.push(fitT(b[0] + .18, b[1], 'group ' + j, {anchor:'start', c:COL.grey, fs:.22}));
      });
    side(sh, 5.95, 1.2, [
      [UNIT_VIEWS[v].name, COL.purpleL, .30],
      [{zh:'purity ' + pu.toFixed(3), en:'purity ' + pu.toFixed(3)},
       pu > 0.99 ? COL.tealL : COL.orange, .28],
      [{zh:'亂猜是 0.333', en:'guessing scores 0.333'}, COL.grey, .24]
    ], .46);
    caps(sh, s);
    F.push({shapes:sh,
      panels:[{lbl:'purity', chips:[chip(pu.toFixed(3), pu > 0.99 ? 'ok' : (pu < 0.4 ? 'bad' : 'hot'))]}],
      line:v === 0 ? 1 : (v === 1 ? 3 : 4),
      msg:s.show === 'truth'
        ? {zh:'先看真相：三條水平帶。token 數在三群裡都是 200 到 4000，所以它完全不帶資訊。',
           en:'The truth first: three horizontal bands. The token count runs from 200 to 4000 in all three groups, so it carries no information at all.'}
        : v === 0
          ? {zh:'原始單位下，距離幾乎完全由 token 數決定，所以 k-means 切出三條垂直帶，purity ' + pu.toFixed(3) + '——跟亂猜一樣。它沒有壞掉，它只是回答了另一個問題，而且還附上一個看起來很正常的 inertia。',
             en:'In raw units the distance is decided almost entirely by the token count, so k-means cuts three vertical bands and scores ' + pu.toFixed(3) + ' - the same as guessing. Nothing broke; it answered a different question, and attached a perfectly normal-looking inertia to it.'}
          : v === 1
            ? {zh:'標準化常被當成修法，但它其實是另一個同樣武斷的選擇：「每一欄一樣重要」。那欄純雜訊現在拿到和答案一樣的票數，所以 purity 只到 ' + pu.toFixed(3) + ' 就停了。',
               en:'Standardising is usually called the fix. It is really a different arbitrary choice - "every column matters equally" - so the pure-noise column now gets exactly as many votes as the answer, and purity stops at ' + pu.toFixed(3) + '.'}
            : {zh:'丟掉無關的那欄就回到 ' + pu.toFixed(3) + '。沒有任何縮放法能告訴你哪些欄位相關——那是你的知識，不是資料的性質。',
               en:'Dropping the irrelevant column gets back to ' + pu.toFixed(3) + '. No scaling scheme can tell you which columns are relevant: that is your knowledge, not a property of the data.'}});
  });
  return F.list;
}

/* ================================================================ tab 5: k-nearest */
const TRN = [[1.0,1.0],[1.4,1.8],[0.8,2.4],[1.9,1.2],[1.2,3.1],[2.1,2.6],
             [4.8,2.1],[4.6,3.0],[5.4,1.3],[5.8,2.5],[5.1,3.6],[6.1,1.6],[4.9,0.9],
             [3.3,2.15]];
const TRN_Y = [0,0,0,0,0,0, 1,1,1,1,1,1,1, 1];
const QRY = [2.8, 2.2];
const LIM_K = [[0.3, 0.3], [6.6, 4.1]];
const KS = [1, 5, 13];
const CODE_KNN = [
  'd2 = ((Xtrain - q) ** 2).sum(axis=1)',
  'nn = np.argsort(d2)[:k]',
  'return np.bincount(ytrain[nn]).argmax()',
  '',
  '# there is no training step: the data is the model.',
  '# k is the smoothing knob - and on a training point',
  '# the nearest neighbour is the point itself.'
];
function knnStage(o){
  const sh = [], plot = new Plot(.45, .70, 5.2, 4.1, LIM_K);
  plot.frame(sh);
  // inside the circle a point votes and shows its class brightly; outside it keeps
  // a different outline per class so the shape of the data is still readable.
  const st = TRN.map((_, i) => o.inK && o.inK.indexOf(i) >= 0
    ? (TRN_Y[i] === 0 ? 'ok' : 'act')
    : (TRN_Y[i] === 0 ? 'idle' : 'ghost'));
  if (o.ring != null){
    const q = plot.px(QRY), far = plot.px(TRN[o.ring]);
    const r = Math.sqrt((q[0] - far[0]) * (q[0] - far[0]) + (q[1] - far[1]) * (q[1] - far[1]));
    sh.push(S.c(q[0], q[1], r, 'ghost', null));
  }
  dots(sh, plot, TRN, st, null, .14);
  const q = plot.px(QRY);
  sh.push(S.e(q[0] - .19, q[1], q[0] + .19, q[1], {s:'hot', arrow:false, w:.055}));
  sh.push(S.e(q[0], q[1] - .19, q[0], q[1] + .19, {s:'hot', arrow:false, w:.055}));
  sh.push(fitT(q[0], q[1] + .42, {zh:'查詢點', en:'query'}, {c:COL.orange, fs:.24}));
  side(sh, 6.0, 1.15, o.side, .46);
  return caps(sh, o);
}
function knnFrames(v){
  const F = new Frames(), k = KS[v];
  const order = TRN.map((p, i) => [d2(p, QRY), i]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
  const dists = TRN.map(p => d2(p, QRY)).sort((a, b) => a - b);
  const steps = [{n:0, inK:[], c0:0, c1:0, ring:null, d:null}];
  for (let n = 1; n <= k; n++){
    const inK = order.slice(0, n);
    const c0 = inK.filter(i => TRN_Y[i] === 0).length, c1 = n - c0;
    steps.push({n:n, inK:inK, c0:c0, c1:c1, ring:order[n - 1], d:dists[n - 1]});
  }
  steps.forEach((s, i) => {
    const last = i === steps.length - 1;
    const win = s.c0 > s.c1 ? 0 : 1;
    const first = s.n === 0;
    F.push({
      shapes:knnStage({
        inK:s.inK, ring:s.ring,
        cap:first
          ? {zh:'14 個訓練點，一個查詢點。沒有模型要訓練——資料本身就是模型。',
             en:'14 training points and one query. There is no model to train - the data is the model.'}
          : {zh:'第 ' + s.n + ' 近的鄰居，距離平方 ' + fmtN(Math.round(s.d * 1000) / 1000) +
                '，類別 ' + TRN_Y[s.ring],
             en:'neighbour ' + s.n + ', squared distance ' + fmtN(Math.round(s.d * 1000) / 1000) +
                ', class ' + TRN_Y[s.ring]},
        cap2:first
          ? {zh:'接下來一個一個把最近的鄰居納進來，看票數怎麼變',
             en:'the nearest neighbours are taken in one at a time, and the votes move with them'}
          : {zh:'目前票數：類別 0 有 ' + s.c0 + ' 票，類別 1 有 ' + s.c1 + ' 票',
             en:'votes so far: class 0 has ' + s.c0 + ', class 1 has ' + s.c1},
        side:[[{zh:'k = ' + k, en:'k = ' + k}, COL.purpleL, .30],
              [{zh:'目前領先：類別 ' + win, en:'leading: class ' + win},
               win === 0 ? COL.tealL : COL.purple, .28],
              [{zh:'6 個類別 0、8 個類別 1', en:'6 of class 0, 8 of class 1'}, COL.grey, .24]]
      }),
      panels:[{lbl:'votes', chips:[chip('class 0: ' + s.c0, 'ok'), chip('class 1: ' + s.c1, 'act')]},
              {lbl:'answer', chips:[chip(first ? '-' : 'class ' + win, last ? 'hot' : 'dim')]}],
      line:first ? 4 : (last ? 2 : 1),
      msg:first
        ? {zh:'KNN 的訓練階段是空的：把資料存起來就結束了。所有的成本都搬到查詢時間，每一次查詢都要重算一次全部的距離。',
           en:'The training phase of KNN is empty: store the data and you are done. All of the cost moves to query time, and every single query recomputes every distance.'}
        : !last
        ? (s.n === 1
           ? {zh:'最近的鄰居是那個落單的類別 1 的點——它就坐在類別 0 那群的旁邊。k = 1 只看它一個。',
              en:'The nearest neighbour is the lone class 1 point sitting right next to the class 0 group. At k = 1 it is the only vote.'}
           : {zh:'再納入一個鄰居。KNN 沒有訓練階段，所以這個圈圈是每次查詢都要重新畫的——成本從訓練搬到了查詢。',
              en:'One more neighbour joins. KNN has no training phase, so this circle is redrawn for every single query - the cost moved from training to query time.'})
        : (k === 1
           ? {zh:'k = 1 回答類別 1，完全由一個離群點決定。順帶一提：如果查詢點本身就是訓練點，最近的鄰居就是它自己——所以只要訓練點兩兩相異，k = 1 在自己的訓練集上就一定 100% 正確。',
              en:'k = 1 answers class 1, decided entirely by one stray point. And note: if the query is itself a training point, its nearest neighbour is itself - so as long as no two training points coincide, k = 1 is 100% correct on its own training set by construction.'}
           : k === 5
             ? {zh:'k = 5 把那個離群點稀釋掉，答案翻成類別 0（' + s.c0 + ' 比 ' + s.c1 + '）。k 就是平滑的旋鈕，而訓練集上的分數指著錯的方向——它在 k = 1 時最高。',
                en:'k = 5 dilutes the stray point and the answer flips to class 0 (' + s.c0 + ' to ' + s.c1 + '). k is the smoothing knob, and the training score points the wrong way along it: it is highest at k = 1.'}
             : {zh:'k = 13 幾乎是整個訓練集，所以每個查詢都會拿到全域多數（' + s.c1 + ' 比 ' + s.c0 + '）。模型被壓平成一個常數，而且只要資料不平衡，它看起來還會很準。',
                en:'k = 13 is nearly the whole training set, so every query gets the global majority (' + s.c1 + ' to ' + s.c0 + '). The model has flattened into a constant - and on imbalanced data it will still look accurate.'})
    });
  });
  return F.list;
}

/* ================================================================ tab 6: the curse */
// Every number here is computed by clustering_knn.py: contrast(500, d) and the
// accuracy of k=11 on two informative columns padded with noise.
const DIMS = [2, 4, 8, 16, 64, 256, 1024];
const CONTRAST = [81.292, 9.940, 3.413, 1.520, 0.544, 0.231, 0.110];
const NDIM = [2, 4, 10, 30, 100, 400];
const NACC = [0.940, 0.940, 0.920, 0.860, 0.790, 0.720];
const CODE_CURSE = [
  'X = rng.random((500, d))',
  'dd = np.sqrt(sq_dists(queries, X))',
  'contrast = (dd.max(1) - dd.min(1)) / dd.min(1)',
  '',
  '# d=2    -> 81.29      d=1024 -> 0.11',
  '# the furthest point stops being further away, because',
  '# every extra column adds one more draw to an average.'
];
function curseFrames(){
  const F = new Frames();
  DIMS.forEach((d, i) => {
    const sh = [];
    bars(sh, .75, 4.55, .72, .34, CONTRAST.slice(0, i + 1).map(x => Math.log10(1 + x)),
         CONTRAST.slice(0, i + 1).map((_, j) => j === i ? 'hot' : 'done'),
         CONTRAST.slice(0, i + 1).map(x => x.toFixed(2)), 2.70, Math.log10(1 + CONTRAST[0]));
    DIMS.slice(0, i + 1).forEach((dd, j) =>
      sh.push(fitT(.75 + j * 1.06 + .36, 4.86, 'd=' + dd, {c:COL.grey, fs:.24})));
    sh.push(fitT(.75, .70, {zh:'(最遠 − 最近) / 最近，長條高度取對數',
                            en:'(furthest - nearest) / nearest, bar height on a log scale'},
                 {anchor:'start', c:COL.pale, fs:.28}));
    side(sh, .75, 1.10, [
      [{zh:'d = 2：最遠的點比最近的點遠 ' + (1 + CONTRAST[0]).toFixed(0) + ' 倍',
        en:'d = 2: the furthest point is ' + (1 + CONTRAST[0]).toFixed(0) + 'x further than the nearest'},
       COL.tealL, .26],
      [{zh:'d = 1024：只遠 ' + (1 + CONTRAST[6]).toFixed(2) + ' 倍',
        en:'d = 1024: only ' + (1 + CONTRAST[6]).toFixed(2) + 'x further'}, COL.red, .26]
    ], .40);
    caps(sh, {cap:{zh:'d = ' + d + '：對比度 ' + CONTRAST[i].toFixed(3),
                   en:'d = ' + d + ': contrast ' + CONTRAST[i].toFixed(3)},
              cap2:{zh:'500 個均勻亂數點，50 次查詢的平均',
                    en:'500 uniform random points, averaged over 50 queries'}});
    F.push({shapes:sh,
      panels:[{lbl:'dims', chips:[chip(String(d), 'hot')]},
              {lbl:'contrast', chips:[chip(CONTRAST[i].toFixed(3), i > 3 ? 'bad' : 'ok')]}],
      line:2,
      msg:i === 0
        ? {zh:'在兩維裡「最近」是真的近：最遠的點差了 81 倍。這就是我們對距離的全部直覺的來源。',
           en:'In two dimensions "nearest" really is near: the furthest point is 81 times further away. This is where all of our intuition about distance comes from.'}
        : i === DIMS.length - 1
          ? {zh:'到了 1024 維，最遠的點只比最近的遠 11%。沒有任何東西壞掉，只是「最近」這個詞不再攜帶資訊——而 KNN 和 k-means 的全部內容就是這個詞。',
             en:'By 1024 dimensions the furthest point is only 11% further than the nearest. Nothing is broken; the word "nearest" has simply stopped carrying information - and that word is the entire content of both algorithms.'}
          : {zh:'每多一維就多一次加總。距離是在越來越多的座標上取平均，而平均會往中間集中。',
             en:'Every extra dimension adds one more term to the sum. The distance is an average over more and more coordinates, and averages concentrate.'}});
  });
  NDIM.forEach((d, i) => {
    const sh = [];
    bars(sh, .95, 4.55, .80, .42, NACC.slice(0, i + 1),
         NACC.slice(0, i + 1).map((_, j) => j === i ? 'hot' : 'done'),
         NACC.slice(0, i + 1).map(x => x.toFixed(3)), 2.70, 1.0);
    NDIM.slice(0, i + 1).forEach((dd, j) =>
      sh.push(fitT(.95 + j * 1.22 + .40, 4.86, 'd=' + dd, {c:COL.grey, fs:.24})));
    sh.push(fitT(.95, .70, {zh:'k = 11 的測試準確率：兩欄有用的資料，其餘全是雜訊',
                            en:'test accuracy at k = 11: two useful columns, the rest pure noise'},
                 {anchor:'start', c:COL.pale, fs:.28}));
    side(sh, .95, 1.10, [
      [{zh:'那兩欄從頭到尾沒有變過', en:'the two useful columns never changed'}, COL.purpleL, .26],
      [{zh:'變的只有要平均幾欄', en:'the only thing that grew is how many columns are averaged'}, COL.grey, .26]
    ], .40);
    caps(sh, {cap:{zh:'d = ' + d + '：準確率 ' + NACC[i].toFixed(3),
                   en:'d = ' + d + ': accuracy ' + NACC[i].toFixed(3)},
              cap2:{zh:'400 個點，300 訓練 / 100 測試', en:'400 points, 300 train / 100 test'}});
    F.push({shapes:sh,
      panels:[{lbl:'dims', chips:[chip(String(d), 'hot')]},
              {lbl:'accuracy', chips:[chip(NACC[i].toFixed(3), NACC[i] > 0.9 ? 'ok' : 'bad')]}],
      line:5,
      msg:i === NDIM.length - 1
        ? {zh:'從 0.940 掉到 0.720，而問題本身一個字都沒改。這是維度詛咒的實務版本：它不長得像幾何問題，它長得像「模型最近變差了」。',
           en:'From 0.940 down to 0.720, with not one thing about the problem changed. This is the practical version of the curse: it does not look like a geometry problem, it looks like "the model got worse lately".'}
        : {zh:'多加幾欄雜訊，準確率就往下掉一點。沒有例外、沒有警告，只有一個安靜停止運作的模型。',
           en:'A few more noise columns, a little less accuracy. No exception, no warning - just a model that quietly stopped working.'}});
  });
  return F.list;
}

/* ================================================================ tab 7: LeetCode 973 */
const PTS973 = [[3,3],[5,-1],[-2,4],[0,6],[1,1],[-4,-2],[6,2],[2,-3]];
const K973 = 3;
const LIM973 = [[-5, -4], [7, 7]];
const CODE_973 = [
  'heap = []',
  'for x, y in points:',
  '    heapq.heappush(heap, (-(x * x + y * y), x, y))',
  '    if len(heap) > k:',
  '        heapq.heappop(heap)      # drop the worst so far',
  'return [[x, y] for _, x, y in heap]',
  '',
  '# no sqrt: it is monotone, so it cannot change the order,',
  '# both comparisons are exact here; the sqrt is simply work'
];
function kcStage(o){
  const sh = [], plot = new Plot(.45, .55, 4.5, 4.5, LIM973);
  plot.frame(sh);
  const o0 = plot.px([0, 0]);
  sh.push(S.c(o0[0], o0[1], .09, 'hot', null));
  sh.push(fitT(o0[0] + .30, o0[1] + .28, 'origin', {anchor:'start', c:COL.orange, fs:.22}));
  if (o.radius != null){
    const far = plot.px([Math.sqrt(o.radius), 0]);
    sh.push(S.c(o0[0], o0[1], far[0] - o0[0], 'ghost', null));
  }
  dots(sh, plot, PTS973, o.st, null, .14);
  PTS973.forEach((p, i) => {
    if (o.st[i] === 'idle' || o.st[i] === 'soft') return;
    const q = plot.px(p);
    sh.push(fitT(q[0], q[1] - .34, String(p[0] * p[0] + p[1] * p[1]),
                 {c:o.st[i] === 'ok' ? COL.tealL : (o.st[i] === 'bad' ? COL.red : COL.grey),
                  fs:.22}));
  });
  side(sh, 5.25, 1.1, o.side, .46);
  return caps(sh, o);
}
function kcFrames(){
  const F = new Frames();
  const key = p => p[0] * p[0] + p[1] * p[1];
  let heap = [];   // kept sorted descending by key; the front is the one to drop
  PTS973.forEach((p, i) => {
    heap.push(p);
    heap.sort((a, b) => key(b) - key(a));
    const dropped = heap.length > K973 ? heap.shift() : null;
    const inHeap = heap.slice();
    const st = PTS973.map((q, j) => j > i ? 'idle'
      : inHeap.indexOf(q) >= 0 ? 'ok' : 'done');
    if (dropped) st[PTS973.indexOf(dropped)] = 'bad';
    st[i] = st[i] === 'ok' ? 'ok' : 'bad';
    F.push({
      shapes:kcStage({st:st, radius:inHeap.length === K973 ? key(inHeap[0]) : null,
        cap:{zh:'看第 ' + (i + 1) + ' 個點 (' + p.join(', ') + ')，x²+y² = ' + key(p),
             en:'point ' + (i + 1) + ' is (' + p.join(', ') + '), x²+y² = ' + key(p)},
        cap2:dropped
          ? {zh:'堆滿了 ' + (K973 + 1) + ' 個，丟掉最差的 (' + dropped.join(', ') + ')，x²+y² = ' + key(dropped),
             en:'the heap held ' + (K973 + 1) + ', so the worst one (' + dropped.join(', ') + ') with x²+y² = ' + key(dropped) + ' is dropped'}
          : {zh:'堆還沒滿 ' + K973 + ' 個，直接收下', en:'the heap is not yet ' + K973 + ' big, so it is simply kept'},
        side:[[{zh:'目前的 ' + K973 + ' 名：' + inHeap.map(q => '(' + q.join(',') + ')').join(' '),
                en:'current top ' + K973 + ': ' + inHeap.map(q => '(' + q.join(',') + ')').join(' ')},
               COL.tealL, .26],
              [{zh:'堆裡最差的 x²+y² = ' + key(inHeap[0]),
                en:'worst key in the heap: x²+y² = ' + key(inHeap[0])}, COL.grey, .24]]}),
      panels:[{lbl:'heap (max by d²)', chips:inHeap.slice().reverse().map(q =>
                chip('(' + q.join(',') + ') ' + key(q), 'ok'))},
              {lbl:'dropped', chips:[chip(dropped ? '(' + dropped.join(',') + ')' : '-',
                                          dropped ? 'bad' : 'dim')]}],
      line:dropped ? 4 : 2,
      msg:i === PTS973.length - 1
        ? {zh:'答案就是堆裡剩下的那 ' + K973 + ' 個，而且順序不重要——題目要的是一個集合。這正是不用排序的理由：排序回答的是比「哪 k 個最近」更強的問題，多做的工就是 log n 那一項。',
           en:'The answer is whatever is left in the heap, in any order - the problem asks for a set. That is exactly why no sort is needed: sorting answers a stronger question than "which k are closest", and the extra work is the log n factor.'}
        : dropped
          ? {zh:'堆裡永遠只留 k 個，所以每一步是 O(log k) 而不是 O(log n)。整個過程比較的都是 x²+y²，不是距離本身——sqrt 是單調的，不可能改變排序，但它會把整數變成浮點數。',
             en:'The heap never holds more than k, so each step is O(log k) rather than O(log n). Everything is compared on x²+y², never on the distance itself: sqrt is monotone so it cannot change the order, but it does turn exact integers into floats.'}
          : {zh:'前 k 個點直接收下。注意這裡用的是 max-heap：要能丟掉「目前最差的那個」，所以 key 取負號推進 Python 的 min-heap。',
             en:'The first k points are simply kept. Note this is a max-heap: the thing we need to drop is the worst one so far, so the key is negated to fit Python\'s min-heap.'}
    });
  });
  return F.list;
}

/* ================================================================ tab 8: LeetCode 658 */
const ARR658 = [1, 2, 3, 4, 5, 6, 8, 10];
const CODE_658 = [
  'lo, hi = 0, len(arr) - k',
  'while lo < hi:',
  '    mid = (lo + hi) // 2',
  '    if x - arr[mid] > arr[mid + k] - x:   # `>`, not `>=`',
  '        lo = mid + 1',
  '    else:',
  '        hi = mid',
  'return arr[lo:lo + k]'
];
const CASES658 = [
  {k:3, x:9, strict:true},
  {k:2, x:3, strict:true},
  {k:2, x:3, strict:false}
];
function win658(sh, lo, k, hi, mid, x, done, strict){
  const X0 = .75, CW = 1.02;
  ARR658.forEach((v, i) => {
    const inWin = mid != null && i >= mid && i < mid + k;
    const st = done ? (inWin ? (strict ? 'ok' : 'bad') : 'idle')
             : (mid != null && (i === mid || i === mid + k)) ? 'hot' : (inWin ? 'act' : 'idle');
    sh.push(box(X0 + i * CW, 2.2, CW - .10, .75, st, String(v), .34));
    sh.push(fitT(X0 + i * CW + (CW - .10) / 2, 3.28, String(i), {c:COL.grey, fs:.22}));
  });
  [['lo', lo, COL.tealL], ['hi', hi, COL.purpleL]].forEach(([lab, v, c], j) => {
    if (v == null || v >= ARR658.length) return;
    sh.push(fitT(X0 + v * CW + (CW - .10) / 2, 1.70 + j * .40, lab + ' = ' + v, {c:c, fs:.26}));
  });
  sh.push(fitT(X0, 1.20, {zh:'x = ' + x + '，k = ' + k, en:'x = ' + x + ', k = ' + k},
               {anchor:'start', c:COL.orange, fs:.30}));
  return sh;
}
function w658Frames(v){
  const c = CASES658[v], k = c.k, x = c.x, strict = c.strict;
  const F = new Frames();
  let lo = 0, hi = ARR658.length - k;
  const push = (mid, left, right, took, done) => {
    const sh = win658([], lo, k, hi, mid, x, done, strict);
    side(sh, .75, 4.15, [
      [done
        ? {zh:'答案 [' + ARR658.slice(lo, lo + k).join(', ') + ']',
           en:'answer [' + ARR658.slice(lo, lo + k).join(', ') + ']'}
        : {zh:'視窗 [' + mid + ', ' + (mid + k) + ')：丟掉 arr[' + mid + '] 要付 ' + left +
             '，換到 arr[' + (mid + k) + '] 省 ' + right,
           en:'window [' + mid + ', ' + (mid + k) + '): dropping arr[' + mid + '] costs ' + left +
             ', gaining arr[' + (mid + k) + '] saves ' + right},
       done ? (strict ? COL.tealL : COL.red) : COL.pale, .28],
      [{zh:strict ? '條件用 `>`：平手時留左邊（比較小的那個贏）'
                  : '條件用 `>=`：平手時往右移（比較大的那個贏）',
        en:strict ? 'the test uses `>`: on a tie keep the left window, so the smaller value wins'
                  : 'the test uses `>=`: on a tie move right, so the larger value wins'},
       strict ? COL.grey : COL.red, .24]
    ], .42);
    caps(sh, {cap:done
        ? {zh:'收斂到 lo = ' + lo + '，回傳 arr[' + lo + ':' + (lo + k) + ']',
           en:'converged to lo = ' + lo + ', returning arr[' + lo + ':' + (lo + k) + ']'}
        : {zh:'mid = ' + mid + '：' + left + (took === 'right' ? ' > ' : ' ≤ ') + right +
             ' ⇒ ' + (took === 'right' ? 'lo = mid + 1' : 'hi = mid'),
           en:'mid = ' + mid + ': ' + left + (took === 'right' ? ' > ' : ' <= ') + right +
             ' => ' + (took === 'right' ? 'lo = mid + 1' : 'hi = mid')},
      cap2:done
        ? {zh:'排序好的陣列裡，離 x 最近的 k 個值一定是連續的——所以答案是一個視窗',
           en:'in a sorted array the k values closest to x are contiguous - so the answer is a window'}
        : {zh:'搜尋的對象是視窗的左緣，不是值本身',
           en:'the search runs over the left edge of the window, not over the values'}});
    F.push({shapes:sh,
      panels:[{lbl:'lo / hi', chips:[chip('lo ' + lo, 'ok'), chip('hi ' + hi, 'act')]},
              {lbl:'window', chips:ARR658.slice(done ? lo : mid, (done ? lo : mid) + k)
                .map(t => chip(String(t), done ? (strict ? 'ok' : 'bad') : 'hot'))}],
      line:done ? 7 : 3,
      msg:done
        ? (strict
           ? {zh:'回傳 [' + ARR658.slice(lo, lo + k).join(', ') + ']。整個搜尋只看了 log(n−k) 個視窗，因為排序好的陣列讓「最近的 k 個」變成連續的一段——這就是一維版本的 k-nearest，也是唯一一個不用 O(n·d) 的情況。',
              en:'It returns [' + ARR658.slice(lo, lo + k).join(', ') + ']. The whole search looked at log(n - k) windows, because a sorted array makes "the k closest" contiguous - this is the one-dimensional case of k-nearest, and the only one that escapes O(n*d).'}
           : {zh:'同一個迴圈，`>` 換成 `>=`，答案變成 [' + ARR658.slice(lo, lo + k).join(', ') +
                 ']，而正確答案是 [2, 3]：平手時題目要比較小的那個。這一個字元在三個範例上全部通過，只在 3.5% 的隨機輸入上跟正解不同。',
              en:'Same loop, `>` swapped for `>=`, and the answer becomes [' + ARR658.slice(lo, lo + k).join(', ') +
                 '] when the correct one is [2, 3]: on a tie the problem wants the smaller value. That one character passes all three worked examples and differs from the reference on 3.5% of random inputs.'})
        : {zh:'比較的是「丟掉左端要付多少」對上「換到右端能省多少」。兩邊都是 x 到端點的距離，所以根本不需要掃過視窗裡的值。',
           en:'The comparison is "what dropping the left end costs" against "what gaining the right end saves". Both are just distances from x to an endpoint, so none of the values inside the window are ever examined.'}});
  };
  let guard = 0;
  while (lo < hi && guard++ < 20){
    const mid = (lo + hi) >> 1;
    const left = x - ARR658[mid], right = ARR658[mid + k] - x;
    const goRight = strict ? left > right : left >= right;
    push(mid, left, right, goRight ? 'right' : 'left', false);
    if (goRight) lo = mid + 1; else hi = mid;
  }
  push(lo, 0, 0, null, true);
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title:{zh:'Day 47 — k-means 與 k-nearest neighbours', en:'Day 47 - k-means and k-nearest neighbours'},
  sub:{zh:'兩個演算法各自只有一條規則，而所有會咬人的東西都住在那個不會報錯的距離函數裡。',
       en:'Two algorithms, one rule each, and everything that bites lives in the distance function - which never raises.'},
  tabs:[
    {
      id:'lloyd', label:{zh:'Lloyd 的兩個半步', en:"Lloyd's two half-steps"},
      stage:{zh:'分配與更新，各自用完全不同的理由讓同一個數字下降',
             en:'assign and update, each lowering the same number for a completely different reason'},
      view:VIEW,
      variants:[{zh:'好的起點', en:'a good start'}, {zh:'倒楣的起點', en:'an unlucky start'}],
      idea:{zh:'k-means 的迴圈只有兩行，而它們是兩個獨立的論證。分配：每個點自己挑最近的 centroid，沒有人會挑到更差的，所以 inertia 不會上升。更新：平均數就是「平方距離總和」的最小值點，所以把 centroid 移到成員平均也不會讓 inertia 上升。可能的分割方式是有限的，加上一個只會往下走的目標函數，迴圈一定會停——但「停下來」不等於「找到最好的答案」。同一份 12 個點、同一段程式，從三群各挑一個點當起點收斂到 inertia 2.805；三個起點全掉進同一群的話，收斂到 30.098：上面那群被切成兩半、下面兩群被黏成一團，而且一樣不報錯。這就是 scikit-learn 一直留著 n_init 的理由；它的 "auto" 預設在 k-means++ 下只跑一次、在隨機初始化下跑十次，而那個差別本身就是在說 k-means++ 值多少。',
            en:'The k-means loop is two lines, and they are two separate arguments. Assignment: every point picks its own nearest centroid, so nobody ends up worse off and the inertia cannot rise. Update: the mean is the minimiser of the sum of squared distances, so moving each centroid to its members\' mean cannot raise it either. A finite number of possible partitions plus an objective that only falls means the loop has to stop - but stopping is not the same as being right. On the same 12 points and the same code, one centroid per group converges to inertia 2.805; three starting points inside one group converge to 30.098, with that group split in half and the other two merged, reported with equal confidence. This is why scikit-learn still has an n_init parameter at all; its "auto" default runs one start with k-means++ and ten with random initialisation, and that gap is itself a statement about what k-means++ buys.'},
      legend:[['#ff9736', {zh:'centroid', en:'centroid'}],
              ['#3fe0dd', {zh:'第 0 群', en:'cluster 0'}],
              ['#9d6bff', {zh:'第 1 群', en:'cluster 1'}],
              ['#2f5661', {zh:'第 2 群', en:'cluster 2'}]],
      get code(){ return CODE_LLOYD; },
      build:lloydFrames
    },
    {
      id:'elbow', label:{zh:'目標函數選不了 k', en:'the objective cannot pick k'},
      stage:{zh:'inertia 每加一個 centroid 就下降，有沒有結構都一樣',
             en:'the inertia falls for every extra centroid, structure or not'},
      view:VIEW,
      variants:[{zh:'三群真的存在', en:'three real groups'}, {zh:'均勻亂數', en:'uniform noise'}],
      idea:{zh:'inertia 是 k 的遞減函數，k = n 時剛好等於 0——所以「最小化目標函數」永遠回答 k = n，它沒有能力告訴你該分幾群。手肘法看的是「下降的幅度什麼時候不再划算」，那是人看圖做的判斷，不是計算。在三群真的存在的資料上，k=2→3 砍掉 90.8%、下一步只砍 22.0%，手肘很清楚；換成 12 個均勻亂數點，42.2% 之後是 45.6%，完全沒有手肘——但 k-means 照樣交給你三群大小很整齊的雜訊，而且每一群都有一個合理的 centroid。這就是為什麼 silhouette、gap statistic 這類方法存在：gap statistic 拿這條曲線去跟隨機資料上的曲線比，silhouette 則問每個點離自己這群比離第二近的那群近多少——兩個都拒絕回答「inertia 有多小」。',
            en:'Inertia is a decreasing function of k and hits exactly 0 at k = n, so "minimise the objective" always answers k = n - it has no power to tell you how many groups there are. The elbow method asks when the drop stops being worth it, which is a judgement made by looking, not a computation. On data that really does have three groups, k=2 to 3 removes 90.8% and the next step only 22.0%, so the elbow is obvious; on 12 uniform random points, 42.2% is followed by 45.6% and there is no elbow at all - yet k-means still hands back three evenly sized clusters of noise, each with a perfectly reasonable centroid. That is why silhouette scores and the gap statistic exist: the gap statistic compares this curve against the same curve on random data, and the silhouette asks each point how much nearer it is to its own cluster than to the next one. Neither answers the question "how small is the inertia".'},
      legend:[['#ff9736', {zh:'目前這個 k', en:'the current k'}],
              ['#2f5661', {zh:'前面幾個 k', en:'earlier values of k'}],
              ['#3fe0dd', {zh:'分到的群（顏色）', en:'cluster membership (colour)'}]],
      get code(){ return CODE_K; },
      build:kFrames
    },
    {
      id:'empty', label:{zh:'空掉的 centroid', en:'an emptied centroid'},
      stage:{zh:'一個 nan，一輪之內吃掉整份資料', en:'one nan, and the whole data set is gone in a round'},
      view:VIEW,
      idea:{zh:'centroid 失去全部成員不是罕見情況：只要某一輪其他 centroid 移動得夠多，就有人會被擠到沒人要的位置。空集合取平均是 nan，numpy 發一個 RuntimeWarning 到 stderr 就結束了。真正的問題在下一行：np.argmin 不會跳過 nan，它回傳 nan 的位置。於是下一輪每個點到那個死掉的 centroid 的距離都是 nan，而 nan 贏了所有比較——整份資料在一輪之內全部加入同一群。最後交出去的是一個形狀完全正確的 (k, 2) 陣列，裡面有 k−1 列 nan，而且 len(C) 還是 k，所以任何「有幾群」的檢查都會過。改用 np.nanargmin 只是把問題往下藏一層；scikit-learn 的做法是把失去成員的 centroid 搬到離所有 centroid 最遠的那個點。',
            en:'A centroid losing every member is not an exotic case: as soon as one round moves the others far enough, somebody gets squeezed out. The mean of an empty set is nan, and numpy says so with one RuntimeWarning on stderr. The real problem is the next line: np.argmin does not skip a nan, it returns its index. So on the next round every distance to the dead centroid is nan, nan wins every comparison, and the entire data set joins one cluster in a single round. What comes out is a perfectly shaped (k, 2) array with k-1 rows of nan in it, and len(C) is still k, so any "how many clusters" check passes. Switching to np.nanargmin just hides it one level deeper; what scikit-learn does is relocate the orphan to the point furthest from any centroid.'},
      legend:['hot', 'ok', 'bad', 'done'],
      get code(){ return CODE_EMPTY; },
      build:emptyFrames
    },
    {
      id:'units', label:{zh:'單位決定分群', en:'the units decide the clustering'},
      stage:{zh:'同一份資料、三種距離函數、三個不同的答案',
             en:'one data set, three distance functions, three different answers'},
      view:VIEW,
      variants:[{zh:'原始單位', en:'raw units'}, {zh:'標準化', en:'standardised'}, {zh:'只用命中率', en:'hit rate only'}],
      idea:{zh:'12 筆請求，兩欄：prompt token 數（200–4000）和 cache 命中率（0–1）。三個真實族群只差在命中率。token 那一欄跨度大了幾千倍，平方之後在距離裡佔了千萬倍的份量，所以原始單位的 k-means 切出三條垂直的 token 帶，purity 0.333——跟亂猜一樣，而且它回報的 inertia 是一個完全正常的數字。標準化通常被叫做「修法」，但它其實是另一個同樣武斷的選擇：「每一欄一樣重要」。那欄純雜訊現在拿到和答案一樣的票數，所以 purity 只到 0.667。把無關的那欄丟掉才回到 1.000。縮放決定了距離的意思，但沒有任何縮放法能告訴你哪些欄位相關——那是你的領域知識，不是資料的性質。第六個分頁的雜訊維度是同一件事換個長相。',
            en:'Twelve requests, two columns: prompt tokens (200 to 4000) and a cache hit rate (0 to 1). The three real populations differ only in hit rate. The token column spans thousands of times more, and squared that is millions of times more weight in the distance, so raw k-means cuts three vertical token bands and scores a purity of 0.333 - the same as guessing - while reporting a perfectly normal inertia. Standardising is usually called the fix; it is really a different arbitrary choice, "every column matters equally", which lets the pure-noise column cast exactly as many votes as the answer, and purity stops at 0.667. Dropping the irrelevant column is what gets back to 1.000. Scaling decides what the distance means, but no scaling scheme can tell you which columns are relevant: that is your domain knowledge, not a property of the data. The noise-dimension tab is the same fact wearing a different face.'},
      legend:[['#3fe0dd', {zh:'第 0 群', en:'cluster 0'}],
              ['#9d6bff', {zh:'第 1 群', en:'cluster 1'}],
              ['#2f5661', {zh:'第 2 群', en:'cluster 2'}]],
      get code(){ return CODE_UNITS; },
      build:unitsFrames
    },
    {
      id:'knn', label:{zh:'k 近鄰投票', en:'the k-nearest vote'},
      stage:{zh:'沒有訓練階段，k 就是平滑的旋鈕', en:'no training step, and k is the smoothing knob'},
      view:VIEW,
      variants:[{zh:'k = 1', en:'k = 1'}, {zh:'k = 5', en:'k = 5'}, {zh:'k = 13', en:'k = 13'}],
      idea:{zh:'KNN 沒有訓練：資料本身就是模型，成本整個搬到查詢時間。14 個訓練點裡有一個類別 1 的點落在類別 0 那群旁邊；k = 1 只看它，所以查詢點被判成類別 1；k = 5 把它稀釋掉，答案翻成類別 0（4 比 1）；k = 13 幾乎是整個訓練集，於是每個查詢都拿到全域多數，模型被壓平成一個常數。k 太小過擬合、太大退化成「永遠猜多數」，而中間那段才有用。最要命的是怎麼挑 k：在訓練集上量準確率的話，k = 1 一定是 1.000，因為訓練點的最近鄰就是它自己——只要點兩兩相異這就恆成立，所以那個 1.000 什麼都沒證明。把自己排除掉之後，同一份資料在真的 moons 上掉到 0.956，而測試準確率要到 k = 11 才到頂（與 k = 25 並列）。',
            en:'KNN has no training step: the data is the model, and the whole cost moves to query time. One of the 14 training points is a class 1 point sitting beside the class 0 group. At k = 1 it is the only vote, so the query is called class 1; at k = 5 it is diluted and the answer flips to class 0, four votes to one; at k = 13, nearly the whole training set, every query gets the global majority and the model has flattened into a constant. Too small and it overfits, too large and it degenerates into always guessing the majority. The nastiest part is choosing k: measured on the training set, k = 1 always scores 1.000, because the nearest neighbour of a training point is itself. That holds for any data set with distinct points, so the 1.000 proves nothing. Excluding the point itself drops the same measurement to 0.956 on the real moons data, and the held-out accuracy only peaks at k = 11, tied with k = 25.'},
      legend:[['#ff9736', {zh:'查詢點', en:'the query'}],
              ['#3fe0dd', {zh:'投票中的類別 0', en:'class 0, voting'}],
              ['#9d6bff', {zh:'投票中的類別 1', en:'class 1, voting'}],
              ['#0a6b74', {zh:'圈外，沒有投票權', en:'outside the circle, no vote'}]],
      get code(){ return CODE_KNN; },
      build:knnFrames
    },
    {
      id:'curse', label:{zh:'維度詛咒', en:'the curse of dimensionality'},
      stage:{zh:'多加的欄位不會帶來資訊，只會帶來平均',
             en:'extra columns do not add information, they add averaging'},
      view:VIEW,
      idea:{zh:'兩個演算法的全部內容都是「最近」這個詞，而這個詞在高維度會失去意義。500 個均勻亂數點，d = 2 時最遠的點比最近的遠 82 倍，d = 1024 時只遠 1.11 倍：距離是在越來越多的座標上取平均，而平均會往中間集中。這不是理論潔癖——實務版本長得完全不像幾何問題：保持兩欄真正有用的資料不動，其餘全部補上純雜訊，k = 11 的測試準確率從 0.940 一路掉到 0.720，沒有例外、沒有警告，只有一個安靜變差的模型。兩個教訓：先降維或先選欄位，再談距離；以及精確的最近鄰索引（k-d tree 之類）大概在幾十維之後就不再贏過暴力掃描（這是經驗法則，不是這裡量到的數字），所以真實系統換成近似最近鄰，用 recall 低於 1.0 去換速度。',
            en:'The entire content of both algorithms is the word "nearest", and that word stops meaning anything in high dimensions. With 500 uniform random points, at d = 2 the furthest point is 82 times further away than the nearest; at d = 1024 it is 1.11 times. The distance is an average over more and more coordinates, and averages concentrate. This is not a theoretical nicety - the practical version does not look like geometry at all: keep two genuinely useful columns, pad with pure noise, and the held-out accuracy of k = 11 falls from 0.940 to 0.720 with no exception and no warning, just a model that quietly got worse. Two lessons follow: reduce or select the columns before you talk about distance, and remember that exact nearest-neighbour indexes such as k-d trees stop beating brute force somewhere in the low tens of dimensions - a rule of thumb rather than anything measured here - which is why real systems switch to approximate nearest neighbours and buy speed with a recall below 1.0.'},
      legend:['hot', 'done', 'ok', 'bad'],
      get code(){ return CODE_CURSE; },
      build:curseFrames
    },
    {
      id:'topk', label:{zh:'LeetCode 973', en:'LeetCode 973'},
      stage:{zh:'要的是一個集合，不是一個順序——所以不必排序',
             en:'the answer is a set, not an order - so no sort is needed'},
      view:VIEW,
      idea:{zh:'「離原點最近的 k 個點」是最小的 k-nearest：只有一個查詢，而且查詢點固定在原點。排序當然會對，但排序回答的是比題目更強的問題，多出來的就是 log n 那一項。大小為 k 的 max-heap 把每一步壓到 O(log k)：推進去，一旦變成 k+1 個就丟掉最差的那個。Day 21 的 quickselect 可以做到期望 O(n)，代價是把陣列翻過來、而且不保證順序——不過題目本來就說順序隨便。另一個細節是整個過程比較的是 x²+y² 而不是距離：sqrt 是單調的，不可能改變排序，但它會把整數換成浮點數，所以它是純成本。',
            en:'"The k points closest to the origin" is the smallest possible k-nearest problem: a single query, pinned at the origin. Sorting works, but it answers a stronger question than the one asked, and the extra work is the log n factor. A max-heap of size k makes each step O(log k): push, and drop the worst as soon as there are k + 1. Day 21\'s quickselect gets to O(n) expected, at the cost of permuting the array and guaranteeing nothing about order - which is fine, because the problem explicitly allows any order. The other detail is that everything is compared on x²+y² rather than the distance: sqrt is monotone so it cannot change the ordering, but it does turn exact integers into floats, which makes it pure cost.'},
      legend:['ok', 'bad', 'hot', 'done'],
      get code(){ return CODE_973; },
      build:kcFrames
    },
    {
      id:'window', label:{zh:'LeetCode 658', en:'LeetCode 658'},
      stage:{zh:'一維的 k 近鄰：答案是連續的一段，所以二分它的左緣',
             en:'k-nearest in one dimension: the answer is contiguous, so binary search its left edge'},
      view:VIEW,
      variants:[{zh:'x = 9', en:'x = 9'}, {zh:'平手，用 `>`', en:'a tie, with `>`'}, {zh:'平手，用 `>=`', en:'a tie, with `>=`'}],
      idea:{zh:'排序好的陣列裡，離 x 最近的 k 個值一定是連續的一段，所以答案是一個視窗，而要找的是它的左緣。比較的是「丟掉 arr[mid] 要付 x − arr[mid]」對上「換到 arr[mid+k] 能省 arr[mid+k] − x」，兩邊都是到端點的距離，所以視窗裡的值一個都不用看，整件事是 O(log(n−k))。那個 `>` 不能寫成 `>=`：平手時題目要留比較小的值，也就是留左邊的視窗。換成 `>=` 之後三個官方範例全部照樣通過，只在大約 3.5% 的隨機輸入上給出不同的答案。這一題是這一天唯一不用 O(n·d) 的情況——而它能便宜，靠的完全是「一維可以排序」，一旦變成二維就沒有這種順序了。',
            en:'In a sorted array the k values closest to x are contiguous, so the answer is a window and what you search for is its left edge. The comparison is "dropping arr[mid] costs x - arr[mid]" against "gaining arr[mid+k] saves arr[mid+k] - x" - both are just distances to an endpoint, so none of the values inside the window are ever examined and the whole thing is O(log(n - k)). The `>` must not become `>=`: on a tie the problem wants the smaller value, which means keeping the left window. With `>=` all three worked examples still pass, and the answer differs from the reference on about 3.5% of random inputs. This is the only case in the whole day that escapes O(n*d), and the reason it can is entirely that one dimension can be sorted - in two dimensions there is no such order.'},
      legend:['hot', 'act', 'ok', 'bad'],
      get code(){ return CODE_658; },
      build:w658Frames
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
