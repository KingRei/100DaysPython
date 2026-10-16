// DAY: 45
// TITLE_ZH: 梯度下降家族：SGD、momentum 與 Adam，以及學習率失效的那個精確臨界點
// TITLE_EN: The gradient descent family: SGD, momentum and Adam, and the exact point where a learning rate stops working
// SUB_ZH: 二次碗上的學習率有精確上限 2/最大曲率：0.0999 收斂、0.1001 飛到無限大，而且不會丟任何例外。momentum 把步數對條件數的依賴從 kappa 變成 sqrt(kappa)。Adam 的細節寫錯也不會報錯：少了 bias correction 步子會太「大」（第 12 步 6.6 倍）、epsilon 放進根號裡會讓小梯度座標的步長縮 99 倍、L2 放進梯度會讓梯度最大的權重被衰減得最少。
// SUB_EN: On a quadratic bowl the learning rate has an exact ceiling of 2/largest curvature: 0.0999 converges, 0.1001 runs off to infinity, and nothing raises. Momentum turns the step count's dependence on the condition number from kappa into sqrt(kappa). Adam's details fail silently too: without bias correction the steps are too LARGE (6.6x at step 12), an epsilon moved inside the square root shrinks small-gradient steps 99x, and L2 folded into the gradient decays the largest-gradient weights least.
// FOLDER: day%2045%20-%20gradient%20descent%20momentum%20and%20adam
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
const fmt = n => { const s = String(n), neg = s[0] === '-', d = neg ? s.slice(1) : s;
  return (neg ? '-' : '') + d.replace(/\B(?=(\d{3})+(?!\d))/g, ','); };
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
const side = (sh, x, y0, lines, dy) => (lines || []).forEach((l, k) =>
  sh.push(fitT(x, y0 + k * (dy || .5), l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));

/* ================================================================ the shared problem
   f(w) = 0.5 * (CURV[0] w0^2 + CURV[1] w1^2), exactly as in optimizers.py */
const CURV = [1, 20];
const LMAX = 20, LMIN = 1, KAPPA = LMAX / LMIN;
const LR_CEIL = 2 / LMAX;
const LR_BEST = 2 / (LMIN + LMAX);
const RATE_GD = (KAPPA - 1) / (KAPPA + 1);
const RATE_MOM = (Math.sqrt(KAPPA) - 1) / (Math.sqrt(KAPPA) + 1);
const BETA_BEST = RATE_MOM * RATE_MOM;
const LR_MOM = Math.pow(2 / (Math.sqrt(LMAX) + Math.sqrt(LMIN)), 2);
const grad = w => [CURV[0] * w[0], CURV[1] * w[1]];
const norm = w => Math.hypot(w[0], w[1]);
const lossOf = w => .5 * (CURV[0] * w[0] * w[0] + CURV[1] * w[1] * w[1]);
function sgdRun(lr, beta, w0, steps){
  let w = w0.slice(), v = [0, 0];
  const out = [w.slice()];
  for (let k = 0; k < steps; k++){
    const g = grad(w);
    v = [beta * v[0] + g[0], beta * v[1] + g[1]];
    const s = beta ? v : g;
    w = [w[0] - lr * s[0], w[1] - lr * s[1]];
    out.push(w.slice());
  }
  return out;
}
function stepsTo(lr, beta, w0, tol){
  let w = w0.slice(), v = [0, 0];
  const start = norm(w);
  for (let k = 1; k <= 20000; k++){
    const g = grad(w);
    v = [beta * v[0] + g[0], beta * v[1] + g[1]];
    const s = beta ? v : g;
    w = [w[0] - lr * s[0], w[1] - lr * s[1]];
    if (!isFinite(w[0]) || !isFinite(w[1])) return -1;
    if (norm(w) <= (tol || 1e-6) * start) return k;
  }
  return -1;
}
const fmtN = x => Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)
  ? x.toExponential(1) : (Math.round(x * 1e4) / 1e4).toString();

/* the bowl drawn as contours, with a path on top */
const BX = 5.15, BY = 2.5, BS = .68;
const pxw = a => BX + a * BS, pyw = b => BY - b * BS;
function bowl(sh, levels){
  (levels || [18, 9, 4, 1.4, .35]).forEach(f => {
    const a0 = Math.sqrt(2 * f / CURV[0]), a1 = Math.sqrt(2 * f / CURV[1]);
    let prev = null;
    for (let t = 0; t <= 64; t++){
      const th = t / 64 * 2 * Math.PI;
      const p = [pxw(a0 * Math.cos(th)), pyw(a1 * Math.sin(th))];
      if (prev) sh.push(S.e(prev[0], prev[1], p[0], p[1], {s:'idle', w:.02, arrow:false, o:.5}));
      prev = p;
    }
  });
  sh.push(S.c(pxw(0), pyw(0), .07, 'ok', ''));
}
function pathOn(sh, traj, upto, col){
  for (let k = 1; k <= upto && k < traj.length; k++){
    const a = traj[k - 1], b = traj[k];
    if (!isFinite(b[0]) || !isFinite(b[1]) || Math.abs(b[1]) > 2.6 || Math.abs(b[0]) > 7) continue;
    sh.push(S.e(pxw(a[0]), pyw(a[1]), pxw(b[0]), pyw(b[1]),
                {s:k === upto ? 'hot' : col, w:k === upto ? .055 : .035, arrow:false}));
  }
  if (upto > 0 && upto < traj.length){
    const p = traj[upto];
    if (isFinite(p[0]) && Math.abs(p[1]) < 2.6 && Math.abs(p[0]) < 7)
      sh.push(S.c(pxw(p[0]), pyw(p[1]), .1, 'hot', ''));
  }
}

/* ================================================================ tab 1: the ceiling */
const CODE_CEIL = [
  '# plain gradient descent on 0.5 * sum(curv * w**2)',
  'w = w - lr * (curv * w)',
  '',
  '# so every coordinate is just multiplied, every step, by',
  'multiplier = 1 - lr * curv',
  '',
  '# which converges exactly while |multiplier| < 1, i.e.',
  'lr < 2 / curv.max()'
];
const CEIL_LRS = [0.05, 0.09, 0.0999, 0.1001, 0.11, 0.2];
function ceilFrames(v){
  const F = new Frames(), lr = CEIL_LRS[v], w0 = [4, 1];
  const mult = Math.max(Math.abs(1 - lr * CURV[0]), Math.abs(1 - lr * CURV[1]));
  const bad = mult > 1;
  const traj = sgdRun(lr, 0, w0, 60);
  const shoot = k => {
    const sh = [];
    bowl(sh);
    pathOn(sh, traj, k, 'act');
    side(sh, .35, 3.95, [
      ['lr = ' + lr, COL.orangeL, .34],
      [{zh:'w0 方向（曲率 1）的乘數 ' + fmtN(1 - lr * CURV[0]), en:'multiplier along w0 (curvature 1): ' + fmtN(1 - lr * CURV[0])}, COL.grey, .26],
      [{zh:'w1 方向（曲率 20）的乘數 ' + fmtN(1 - lr * CURV[1]), en:'multiplier along w1 (curvature 20): ' + fmtN(1 - lr * CURV[1])}, bad ? COL.red : COL.tealL, .26],
      [{zh:'第 ' + k + ' 步：w = [' + fmtN(traj[k][0]) + ', ' + fmtN(traj[k][1]) + ']', en:'step ' + k + ': w = [' + fmtN(traj[k][0]) + ', ' + fmtN(traj[k][1]) + ']'}, COL.pale, .26]], .42);
    return sh;
  };
  F.push({shapes:caps(shoot(0), {cap:{zh:'lr = ' + lr + '：每步把每個座標乘上 1 − lr × 曲率', en:'lr = ' + lr + ': each step multiplies each coordinate by 1 - lr x curvature'},
      cap2:{zh:'上限是 2 / 20 = 0.1，' + (bad ? '這個 lr 在上限之上' : '這個 lr 在上限之下'), en:'the ceiling is 2 / 20 = 0.1, and this lr is ' + (bad ? 'above it' : 'below it')}}),
    panels:[{lbl:'lr', chips:[chip(String(lr), bad ? 'bad' : 'act')]}, {lbl:'multiplier', chips:[chip(fmtN(mult), bad ? 'bad' : 'ok')]}], line:4,
    msg:{zh:'純梯度下降在二次碗上做的事只有一件：每一步把每個座標乘上 (1 − lr × 曲率)。所以收斂條件是白送的——所有乘數的絕對值都要小於 1，也就是 lr < 2 / 最大曲率 = ' + LR_CEIL + '。',
         en:'Plain gradient descent on a quadratic does exactly one thing: every step it multiplies each coordinate by (1 - lr x curvature). So the convergence condition comes for free - every multiplier must be below 1 in absolute value, i.e. lr < 2 / largest curvature = ' + LR_CEIL + '.'}});
  [1, 2, 3, 6, 12, 25, 45, 60].forEach(k => {
    const w = traj[k], big = Math.abs(w[1]) > Math.abs(traj[0][1]);
    F.push({shapes:caps(shoot(k), {cap:{zh:'第 ' + k + ' 步：|w| = ' + fmtN(norm(w)), en:'step ' + k + ': |w| = ' + fmtN(norm(w))},
        cap2:bad ? {zh:'w1 每步放大 ' + fmtN(Math.abs(1 - lr * CURV[1])) + ' 倍', en:'w1 grows by ' + fmtN(Math.abs(1 - lr * CURV[1])) + 'x every step'}
                 : {zh:'兩個座標都在縮小', en:'both coordinates are shrinking'}}),
      panels:[{lbl:'step', chips:[chip(String(k), 'hot')]}, {lbl:'w1', chips:[chip(fmtN(w[1]), big ? 'bad' : 'ok')]}], line:bad ? 7 : 1,
      msg:k <= 2 ? (bad ? {zh:'w1 的乘數絕對值是 ' + fmtN(Math.abs(1 - lr * CURV[1])) + '，大於 1：它每一步都衝過頭而且越衝越遠。注意 w0 那個方向完全正常——發散的訓練常常還有一半參數看起來很合理。',
                           en:'The multiplier along w1 has absolute value ' + fmtN(Math.abs(1 - lr * CURV[1])) + ', above 1: it overshoots further every step. Note that w0 is behaving perfectly - a diverged run often still has half its parameters looking reasonable.'}
                        : {zh:'w1 的乘數是 ' + fmtN(1 - lr * CURV[1]) + '：負號代表它每步都衝過最低點再翻回來，這就是鋸齒。w0 的乘數 ' + fmtN(1 - lr * CURV[0]) + ' 很接近 1，所以那個方向是用爬的。',
                           en:'The multiplier along w1 is ' + fmtN(1 - lr * CURV[1]) + ': the minus sign means it overshoots the bottom and flips back every step - that is the zigzag. Along w0 the multiplier is ' + fmtN(1 - lr * CURV[0] ) + ', so close to 1 that it crawls.'})
        : k >= 45 ? (bad ? {zh:'到第 ' + k + ' 步 w1 已經是 ' + fmtN(w[1]) + '。沒有任何例外被丟出來，loss 只是一路長到 inf 再變 nan。',
                            en:'By step ' + k + ', w1 is ' + fmtN(w[1]) + '. No exception is ever raised; the loss just grows to inf and then nan.'}
                         : {zh:'穩定收斂。要把誤差縮小一百萬倍需要 ' + stepsTo(lr, 0, w0) + ' 步——限制步長的是最陡的方向，決定進度的卻是最平的方向。',
                            en:'Converging steadily. Shrinking the error a millionfold takes ' + stepsTo(lr, 0, w0) + ' steps - the steepest direction caps the step size while the flattest one sets the progress.'})
        : {zh:bad ? 'w1 繼續放大，w0 繼續正常縮小。' : '鋸齒的振幅在縮小，但橫向的進展很慢。',
           en:bad ? 'w1 keeps growing while w0 keeps shrinking normally.' : 'The zigzag is shrinking, but sideways progress is slow.'}});
  });
  return F.list;
}

/* ================================================================ tab 2: momentum */
const CODE_MOM = [
  'class SGD:',
  '    def step(self, w, g):',
  '        if not self.momentum:',
  '            return w - self.lr * g',
  '        self.v = self.momentum * self.v + g   # a running sum of gradients',
  '        return w - self.lr * self.v',
  '',
  '# rate per step:  plain  (k - 1)/(k + 1)        = 0.9048',
  '#                 momentum (sqrt(k) - 1)/(sqrt(k) + 1) = 0.6345'
];
function momFrames(v){
  const F = new Frames(), w0 = [1, 1], useM = v === 1;
  const lr = useM ? LR_MOM : LR_BEST, beta = useM ? BETA_BEST : 0;
  const rate = useM ? RATE_MOM : RATE_GD;
  const N = 150;
  const mine = sgdRun(lr, beta, w0, N), other = sgdRun(useM ? LR_BEST : LR_MOM, useM ? 0 : BETA_BEST, w0, N);
  const total = stepsTo(lr, beta, w0), otherTotal = stepsTo(useM ? LR_BEST : LR_MOM, useM ? 0 : BETA_BEST, w0);
  const EX0 = 1.5, EX1 = 9.3, EY0 = 4.6, EY1 = .95;
  const ex = k => EX0 + (EX1 - EX0) * Math.min(k, N) / N;
  const ey = e => EY0 - (EY0 - EY1) * (Math.log10(Math.max(e, 1e-7)) + 7) / 7;
  const err = traj => k => norm(traj[k]) / norm(w0);
  const curve = (sh, traj, upto, st, w) => {
    let prev = null;
    for (let k = 0; k <= upto; k++){
      const e = err(traj)(k);
      if (!isFinite(e) || e < 1e-7) break;
      const pt = [ex(k), ey(e)];
      if (prev) sh.push(S.e(prev[0], prev[1], pt[0], pt[1], {s:st, w:w, arrow:false}));
      prev = pt;
    }
    return prev;
  };
  const draw = k => {
    const sh = [];
    sh.push(S.e(EX0, EY0, EX1, EY0, {s:'idle', w:.025, arrow:false}));
    sh.push(S.e(EX0, EY0, EX0, EY1, {s:'idle', w:.025, arrow:false}));
    [0, 50, 100, 150].forEach(t => sh.push(S.t(ex(t), EY0 + .3, String(t), {c:COL.grey, fs:.2})));
    [1e-6, 1e-4, 1e-2, 1].forEach(e => sh.push(S.t(EX0 - .12, ey(e) + .07, String(e), {anchor:'end', c:COL.grey, fs:.2})));
    sh.push(S.t((EX0 + EX1) / 2, EY0 + .62, {zh:'步數', en:'steps'}, {c:COL.grey, fs:.22}));
    sh.push(S.t(EX0 - .1, EY1 - .25, {zh:'誤差（佔起點的比例，對數）', en:'error as a fraction of the start (log)'}, {anchor:'start', c:COL.grey, fs:.22}));
    /* the other method, faded, for comparison */
    curve(sh, other, N, 'done', .03);
    sh.push(fitT(Math.min(ex(otherTotal) + .1, 7.6), ey(1e-6) - .3, (useM ? {zh:'純 GD：' + otherTotal + ' 步', en:'plain GD: ' + otherTotal + ' steps'} : {zh:'momentum：' + otherTotal + ' 步', en:'momentum: ' + otherTotal + ' steps'}), {anchor:'start', c:COL.grey, fs:.22}));
    /* theory */
    let tprev = null;
    for (let t = 0; t <= N; t += 3){
      const e = Math.pow(rate, t);
      if (e < 1e-7) break;
      const pt = [ex(t), ey(e)];
      if (tprev) sh.push(S.e(tprev[0], tprev[1], pt[0], pt[1], {s:'act', w:.02, arrow:false, dash:'.08 .07'}));
      tprev = pt;
    }
    const tip = curve(sh, mine, k, useM ? 'ok' : 'hot', .05);
    if (tip) sh.push(S.c(tip[0], tip[1], .09, 'hot', ''));
    sh.push(S.e(EX0, ey(1e-6), EX1, ey(1e-6), {s:'idle', w:.02, arrow:false, dash:'.1 .08'}));
    side(sh, 5.3, 1.4, [
      [useM ? {zh:'momentum，beta ' + fmtN(BETA_BEST), en:'momentum, beta ' + fmtN(BETA_BEST)} : {zh:'純梯度下降', en:'plain gradient descent'}, useM ? COL.tealL : COL.orangeL, .3],
      ['lr = ' + fmtN(lr), COL.pale, .24],
      [{zh:'每步乘數 ' + fmtN(rate), en:'multiplier ' + fmtN(rate) + ' per step'}, COL.purpleL, .24],
      [{zh:'紫虛線＝理論', en:'purple dashed = theory'}, COL.purpleL, .22],
      [{zh:'縮小一百萬倍：' + total + ' 步', en:'a millionfold: ' + total + ' steps'}, useM ? COL.tealL : COL.orangeL, .26]], .38);
    return sh;
  };
  F.push({shapes:caps(draw(0), {cap:useM ? {zh:'momentum：沿著「過去梯度的累積和」走', en:'momentum: step along the running sum of past gradients'} : {zh:'純梯度下降：每步只看當下的梯度', en:'plain gradient descent: each step sees only the current gradient'},
      cap2:{zh:'同一個條件數 20 的山谷，灰線是另一種方法供對照', en:'the same ravine with condition number 20; the grey curve is the other method'}}),
    panels:[{lbl:'rate', chips:[chip(fmtN(rate), useM ? 'ok' : 'hot')]}, {lbl:'steps', chips:[chip(String(total), 'act')]}], line:useM ? 4 : 3,
    msg:useM ? {zh:'momentum 累積過去的梯度。在鋸齒方向，正負交替的梯度在總和裡互相抵消；在谷底方向，它們方向一致因而累加。理論說每步的誤差乘數會從 ' + fmtN(RATE_GD) + ' 改善成 ' + fmtN(RATE_MOM) + '，也就是 kappa 變成 sqrt(kappa)。',
                en:'Momentum accumulates past gradients. Along the zigzag the alternating gradients cancel inside the sum; along the valley floor they agree and add up. Theory says the per-step error multiplier improves from ' + fmtN(RATE_GD) + ' to ' + fmtN(RATE_MOM) + ' - kappa becoming sqrt(kappa).'}
             : {zh:'用這個碗最好的學習率 2/(1+20) = ' + fmtN(LR_BEST) + '。每步誤差乘數是 (kappa−1)/(kappa+1) = ' + fmtN(RATE_GD) + '，紫色虛線就是這個預測。',
                en:'Using this bowl\'s best learning rate, 2/(1+20) = ' + fmtN(LR_BEST) + '. The per-step error multiplier is (kappa-1)/(kappa+1) = ' + fmtN(RATE_GD) + ', and the purple dashed line is that prediction.'}});
  [2, 5, 12, 25, 45, 80, 150].forEach(k => {
    const e = err(mine)(k);
    F.push({shapes:caps(draw(k), {cap:{zh:'第 ' + k + ' 步：誤差剩下起點的 ' + fmtN(e), en:'step ' + k + ': the error is ' + fmtN(e) + ' of the start'},
        cap2:e < 1e-7 ? {zh:'已經低於圖表的 1e-7 底線；虛線預測 ' + fmtN(Math.pow(rate, k)), en:'already below the chart floor of 1e-7; the dashed line predicts ' + fmtN(Math.pow(rate, k))}
                      : {zh:'虛線（每步 ×' + fmtN(rate) + '）預測 ' + fmtN(Math.pow(rate, k)), en:'the dashed line (x' + fmtN(rate) + ' per step) predicts ' + fmtN(Math.pow(rate, k))}}),
      panels:[{lbl:'step', chips:[chip(String(k), 'hot')]}, {lbl:'error', chips:[chip(fmtN(e), 'act')]}], line:useM ? 5 : 3,
      msg:k <= 5 ? (useM ? {zh:'前幾步 momentum 的累積和裡還沒東西，所以跟普通 GD 幾乎一樣——它的好處要幾步之後才開始顯現，這也是實測比漸近理論慢一點的原因。',
                            en:'For the first few steps the running sum has nothing in it, so momentum tracks plain GD - its advantage only builds up after a while, which is also why the measurement lags the asymptotic theory.'}
                         : {zh:'誤差開始以固定比例縮小，實測貼著紫色虛線。', en:'The error starts shrinking by a constant factor, tracking the purple dashed line.'})
        : k >= 80 ? (useM ? {zh:'實測 ' + total + ' 步對上理論的 ' + Math.round(Math.log(1e-6) / Math.log(rate)) + ' 步；差距來自前面那段暫態。比純 GD 的 ' + otherTotal + ' 步少了 ' + (otherTotal / total).toFixed(1) + ' 倍——這就是 momentum 存在的全部理由。',
                             en:'Measured ' + total + ' steps against a predicted ' + Math.round(Math.log(1e-6) / Math.log(rate)) + '; the gap is that early transient. Against plain GD\'s ' + otherTotal + ' steps that is ' + (otherTotal / total).toFixed(1) + 'x fewer - the entire reason momentum exists.'}
                          : {zh:'實測 ' + total + ' 步，理論 ' + Math.round(Math.log(1e-6) / Math.log(rate)) + ' 步——只差一步。這代表「每步乘一個固定倍數」這個模型抓對了正在發生的事。',
                             en:'Measured ' + total + ' steps against a predicted ' + Math.round(Math.log(1e-6) / Math.log(rate)) + ' - one step apart. That is a good sign the "multiply by a constant each step" model is the right one.'})
        : {zh:'誤差每步縮小 ' + fmtN(rate) + ' 倍，是一條直線——因為縱軸是對數。', en:'The error shrinks by ' + fmtN(rate) + ' per step, which is a straight line because the vertical axis is logarithmic.'}});
  });
  const betas = [0, BETA_BEST, 0.9];
  F.push({shapes:caps(draw(N), {cap:{zh:'beta = 0.9 不是自然常數', en:'beta = 0.9 is not a constant of nature'},
      cap2:{zh:'固定 lr 0.04：' + betas.map(b => 'beta ' + fmtN(b) + ' → ' + stepsTo(0.04, b, [4, 1])).join('，') + ' 步',
            en:'at a fixed lr 0.04: ' + betas.map(b => 'beta ' + fmtN(b) + ' -> ' + stepsTo(0.04, b, [4, 1])).join(', ') + ' steps'}}),
    panels:betas.map(b => ({lbl:'beta ' + fmtN(b), chips:[chip(stepsTo(0.04, b, [4, 1]) + ' steps', b === BETA_BEST ? 'ok' : 'dim')]})), line:4,
    msg:{zh:'這個碗最好的 beta 是 ' + fmtN(BETA_BEST) + '。從別處抄來的 0.9 比調過的值更差（' + stepsTo(0.04, 0.9, [4, 1]) + ' 對 ' + stepsTo(0.04, BETA_BEST, [4, 1]) + ' 步）。beta 在做的事是替一個振盪器調阻尼，它有一個取決於問題的正確答案。',
         en:'The best beta for this bowl is ' + fmtN(BETA_BEST) + '. Copying 0.9 from elsewhere does worse than the tuned value (' + stepsTo(0.04, 0.9, [4, 1]) + ' against ' + stepsTo(0.04, BETA_BEST, [4, 1]) + ' steps). Beta is damping an oscillator, and it has a right answer that depends on the problem.'}});
  return F.list;
}

/* ================================================================ tab 3: Adam, step by step */
const CODE_ADAM = [
  'self.t += 1',
  'self.m = b1 * self.m + (1 - b1) * g          # average of the gradient',
  'self.v = b2 * self.v + (1 - b2) * g * g      # average of its square',
  'mh = self.m / (1 - b1 ** self.t)             # bias correction',
  'vh = self.v / (1 - b2 ** self.t)',
  'w = w - lr * mh / (sqrt(vh) + eps)           # eps OUTSIDE the sqrt',
  '',
  '# without the two corrections the early steps are too LARGE:',
  '#   (1 - 0.9**t) / sqrt(1 - 0.999**t)  =  3.16x at t=1, 6.57x at t=12'
];
const biasFactor = t => (1 - Math.pow(.9, t)) / Math.sqrt(1 - Math.pow(.999, t));
function adamRun(lr, corrected, steps, w0){
  let w = w0.slice(), m = [0, 0], vv = [0, 0];
  const out = [{w:w.slice(), m:m.slice(), v:vv.slice(), step:0, alt:0}];
  for (let t = 1; t <= steps; t++){
    const g = grad(w);
    m = [.9 * m[0] + .1 * g[0], .9 * m[1] + .1 * g[1]];
    vv = [.999 * vv[0] + .001 * g[0] * g[0], .999 * vv[1] + .001 * g[1] * g[1]];
    /* both formulas evaluated on the SAME m and v, so the ratio is exactly the
       theoretical bias factor rather than a comparison of two different runs */
    const sizeWith = (c1, c2) => {
      const d0 = Math.sqrt(vv[0] / c2) + 1e-8, d1 = Math.sqrt(vv[1] / c2) + 1e-8;
      return [lr * (m[0] / c1) / d0, lr * (m[1] / c1) / d1];
    };
    const corr = sizeWith(1 - Math.pow(.9, t), 1 - Math.pow(.999, t));
    const unc = sizeWith(1, 1);
    const used = corrected ? corr : unc, alt = corrected ? unc : corr;
    w = [w[0] - used[0], w[1] - used[1]];
    out.push({w:w.slice(), m:m.slice(), v:vv.slice(),
              step:Math.hypot(used[0], used[1]), alt:Math.hypot(alt[0], alt[1])});
  }
  return out;
}
function adamFrames(v){
  const F = new Frames(), corrected = v === 0, lr = .1, w0 = [1, 1];
  const hist = adamRun(lr, corrected, 40, w0);
  const draw = k => {
    const sh = [];
    bowl(sh, [10.5, 5, 2, .6, .15]);
    pathOn(sh, hist.map(h => h.w), k, corrected ? 'ok' : 'bad');
    const h = hist[k];
    const ratio = k > 0 ? (corrected ? h.alt / h.step : h.step / h.alt) : 1;
    side(sh, .35, 3.80, [
      [corrected ? {zh:'Adam（含 bias correction）', en:'Adam (with bias correction)'} : {zh:'Adam（少了 bias correction）', en:'Adam (bias correction removed)'}, corrected ? COL.tealL : COL.red, .3],
      ['t = ' + k + '   m = [' + fmtN(h.m[0]) + ', ' + fmtN(h.m[1]) + ']   v = [' + fmtN(h.v[0]) + ', ' + fmtN(h.v[1]) + ']', COL.purpleL, .22],
      [{zh:'這一步走了 ' + fmtN(h.step), en:'this step moved ' + fmtN(h.step)}, COL.orangeL, .26],
      k > 0 ? [{zh:'同一組 m、v 用' + (corrected ? '未修正' : '修正') + '公式會走 ' + fmtN(h.alt), en:'the ' + (corrected ? 'uncorrected' : 'corrected') + ' formula on the same m, v: ' + fmtN(h.alt)}, COL.grey, .24] : ['', COL.grey, .24],
      k > 0 ? [{zh:'未修正 ÷ 修正 = ' + ratio.toFixed(3) + '（理論 ' + biasFactor(k).toFixed(3) + '）', en:'uncorrected / corrected = ' + ratio.toFixed(3) + ' (theory ' + biasFactor(k).toFixed(3) + ')'}, ratio > 1.05 ? COL.red : COL.tealL, .26] : ['', COL.grey, .26]], .36);
    return sh;
  };
  F.push({shapes:caps(draw(0), {cap:{zh:'Adam 維護兩個移動平均：梯度的 m、梯度平方的 v', en:'Adam keeps two running averages: m of the gradient, v of its square'},
      cap2:{zh:'步長 lr × m / √v：每個座標大約都走 lr', en:'the step is lr x m / sqrt(v): each coordinate moves about lr'}}),
    panels:[{lbl:'t', chips:[chip('0', 'empty')]}], line:corrected ? 3 : 7,
    msg:corrected ? {zh:'m 和 v 都從零開始，所以一開始都被低估。修正的做法是把 m 除以 1 − 0.9^t、v 除以 1 − 0.999^t。每一步我都會用「同一組 m、v」把兩個公式都算一次，這樣兩者的比值就剛好是理論值，而不是在比兩條不同的軌跡。',
                     en:'Both m and v start at zero, so both are underestimates early on. The fix divides m by 1 - 0.9^t and v by 1 - 0.999^t. At every step both formulas are evaluated on the *same* m and v, so their ratio is exactly the theoretical one rather than a comparison of two different trajectories.'}
                  : {zh:'把兩行修正拿掉。坊間說法是這會讓前幾步太小——實際上太大，因為 v 在根號底下，兩邊的偏差不會互相抵消。',
                     en:'With both correction lines removed. The folklore says this makes the early steps too small; it makes them too large, because v sits under a square root and the two biases do not cancel.'}});
  [1, 2, 5, 12, 20, 40].forEach(k => {
    const h = hist[k], ratio = corrected ? h.alt / h.step : h.step / h.alt;
    F.push({shapes:caps(draw(k), {cap:{zh:'t = ' + k + '：未修正的步長是修正版的 ' + ratio.toFixed(2) + ' 倍', en:'t = ' + k + ': the uncorrected step is ' + ratio.toFixed(2) + 'x the corrected one'},
        cap2:{zh:'理論值 (1 − 0.9^t) / √(1 − 0.999^t) = ' + biasFactor(k).toFixed(3), en:'theory: (1 - 0.9^t) / sqrt(1 - 0.999^t) = ' + biasFactor(k).toFixed(3)}}),
      panels:[{lbl:'t', chips:[chip(String(k), 'hot')]}, {lbl:'step', chips:[chip(fmtN(h.step), corrected ? 'ok' : 'bad')]},
              {lbl:'unc / corr', chips:[chip(ratio.toFixed(2) + 'x', ratio > 1.05 ? 'bad' : 'ok')]}], line:corrected ? (k <= 2 ? 3 : 5) : 8,
      msg:k === 1 ? {zh:'t = 1：m 只有梯度的 0.1 倍、v 只有梯度平方的 0.001 倍。未修正時 m/√v = 0.1g / (0.0316|g|) = 3.16 倍——第一步就比你指定的學習率大了三倍以上。修正是把 m 除以 1 − 0.9 = 0.1、v 除以 1 − 0.999 = 0.001，兩者就回到梯度本身的尺度。',
                     en:'At t = 1, m is 0.1x the gradient and v is 0.001x its square. Uncorrected, m/sqrt(v) = 0.1g / (0.0316|g|) = 3.16x - the very first step is over three times the learning rate you asked for. The correction divides m by 1 - 0.9 = 0.1 and v by 1 - 0.999 = 0.001, putting both back on the gradient\'s scale.'}
        : k === 12 ? {zh:'第 12 步最糟：' + biasFactor(12).toFixed(2) + ' 倍。偏差不是單調消失的——它先變大再慢慢回到 1，到第 1000 步都還有 ' + biasFactor(1000).toFixed(2) + ' 倍。',
                      en:'Step 12 is the worst at ' + biasFactor(12).toFixed(2) + 'x. The bias does not fade monotonically: it grows first, then creeps back towards 1 - still ' + biasFactor(1000).toFixed(2) + 'x at step 1,000.'}
        : k === 40 ? (corrected ? {zh:'修正版的步長一路維持在 lr 的尺度上。Adam 的步長幾乎與梯度大小無關，所以兩個座標以差不多的速度前進——這正是它在尺度差很多的參數上好用的原因。',
                                   en:'The corrected step stays on the scale of lr throughout. Adam\'s step barely depends on the gradient\'s size, so both coordinates advance at a similar pace - which is why it works well when parameters live on very different scales.'}
                                : {zh:'沒有修正的版本在最初幾百步用的是比你指定的大好幾倍的學習率，而那剛好是模型最脆弱的時候。這也是訓練腳本開頭要做 learning rate warmup 的同一個理由。',
                                   en:'The uncorrected version spends its first few hundred steps at a learning rate several times larger than the one you asked for - exactly when a model is most fragile. It is the same early-training fragility that the warmup at the top of a training script is dealing with.'})
        : {zh:'偏差還在：這一步比另一個公式大 ' + ratio.toFixed(2) + ' 倍，跟理論的 ' + biasFactor(k).toFixed(2) + ' 倍一致。', en:'The bias is still there: this step is ' + ratio.toFixed(2) + 'x the other formula, matching the theoretical ' + biasFactor(k).toFixed(2) + 'x.'}});
  });
  return F.list;
}

/* ================================================================ tab 4: the epsilon */
const CODE_EPS = [
  'denom = sqrt(vh) + eps      # PyTorch, and what Adam\'s paper means',
  'denom = sqrt(vh + eps)      # looks identical, is not',
  '',
  '# eps = 1e-8, so sqrt(eps) = 1e-4:',
  '#   v = 1e-12 -> 1.010e-06  vs  1.000e-04    99x',
  '#   v = 1e-16 -> 2.000e-08  vs  1.000e-04  5000x'
];
const EPS_V = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-16];
function epsFrames(){
  const F = new Frames();
  const rows = EPS_V.map(v => {
    const a = Math.sqrt(v) + 1e-8, b = Math.sqrt(v + 1e-8);
    return {v:v, a:a, b:b, r:b / a};
  });
  const draw = k => {
    const sh = [];
    sh.push(S.t(1.5, .75, 'v', {c:COL.purpleL, fs:.28}));
    sh.push(S.t(3.9, .75, '√v + eps', {c:COL.tealL, fs:.28}));
    sh.push(S.t(6.3, .75, '√(v + eps)', {c:COL.orangeL, fs:.28}));
    sh.push(S.t(8.7, .75, {zh:'步長縮小', en:'step shrinks'}, {c:COL.red, fs:.26}));
    rows.forEach((r, i) => {
      if (i >= k) return;
      const y = 1.35 + i * .62, on = i === k - 1;
      sh.push(box(.7, y, 1.7, .5, on ? 'hot' : 'idle', r.v.toExponential(0), .24));
      sh.push(box(3.1, y, 1.7, .5, on ? 'ok' : 'soft', r.a.toExponential(2), .22));
      sh.push(box(5.5, y, 1.7, .5, on ? 'bad' : 'soft', r.b.toExponential(2), .22));
      sh.push(box(7.9, y, 1.7, .5, r.r > 10 ? 'bad' : 'soft', r.r.toFixed(1) + 'x', .24));
    });
    return sh;
  };
  F.push({shapes:caps(draw(0), {cap:{zh:'兩種寫法都能防止除以零', en:'both forms prevent a division by zero'},
      cap2:{zh:'但 eps = 1e-8 的平方根是 1e-4，大了四個數量級', en:'but eps = 1e-8 has sqrt(eps) = 1e-4, four orders of magnitude larger'}}),
    panels:[{lbl:'eps', chips:[chip('1e-8', 'act')]}], line:0,
    msg:{zh:'Adam 的分母寫成 √v + eps 還是 √(v + eps)，看起來只是括號位置不同。eps 的用途是防止除以零，兩種都做得到——差別在它實際有多大。',
         en:'Writing Adam\'s denominator as sqrt(v) + eps or sqrt(v + eps) looks like a question of bracket placement. Both prevent dividing by zero; the difference is how large the guard actually is.'}});
  rows.forEach((r, i) => {
    F.push({shapes:caps(draw(i + 1), {cap:{zh:'v = ' + r.v.toExponential(0) + '：兩個分母差 ' + r.r.toFixed(1) + ' 倍', en:'v = ' + r.v.toExponential(0) + ': the two denominators differ by ' + r.r.toFixed(1) + 'x'},
        cap2:{zh:'√v = ' + Math.sqrt(r.v).toExponential(1) + '，而 √eps = 1e-4', en:'sqrt(v) = ' + Math.sqrt(r.v).toExponential(1) + ', while sqrt(eps) = 1e-4'}}),
      panels:[{lbl:'v', chips:[chip(r.v.toExponential(0), 'hot')]}, {lbl:'ratio', chips:[chip(r.r.toFixed(1) + 'x', r.r > 10 ? 'bad' : 'ok')]}], line:r.r > 10 ? 1 : 0,
      msg:i === 0 ? {zh:'梯度不小的時候 √v 遠大於 eps，兩種寫法完全一樣——這就是為什麼這個 bug 在一般測試裡看不出來。', en:'When gradients are healthy, sqrt(v) dwarfs eps and the two forms agree exactly - which is why this bug is invisible in an ordinary test.'}
        : r.r > 10 ? {zh:'到這裡 √(v + eps) 已經被 eps 主導、鎖在 1e-4 附近，而 √v + eps 還跟著真正的 v 走。步長被硬生生除以 ' + r.r.toFixed(0) + ' 倍——發生在梯度很小的座標上：很少被用到的 embedding 列、幾乎死掉的神經元。',
                      en:'By here sqrt(v + eps) is dominated by eps and pinned near 1e-4, while sqrt(v) + eps still tracks the real v. The step is divided by ' + r.r.toFixed(0) + 'x - and this happens exactly on the coordinates with small gradients: a rarely-used embedding row, a nearly-dead unit.'}
        : {zh:'差距開始出現了。', en:'The gap is starting to open.'}});
  });
  F.push({shapes:caps(draw(rows.length), {cap:{zh:'PyTorch 用的是 √v + eps', en:'PyTorch uses sqrt(v) + eps'},
      cap2:{zh:'這個錯誤只在梯度很小的地方發作，而那正是你最不會去看的地方', en:'this one only fires where gradients are small - exactly where nobody looks'}}),
    panels:[{lbl:'worst', chips:[chip(rows[rows.length - 1].r.toFixed(0) + 'x', 'bad')]}], line:0,
    msg:{zh:'兩種寫法在健康的梯度上一模一樣，在小梯度上差上千倍。訓練不會報錯，只會有一部分參數學得異常地慢。', en:'The two forms are identical on healthy gradients and differ by thousands of times on small ones. Training never errors; some parameters just learn inexplicably slowly.'}});
  return F.list;
}

/* ================================================================ tab 5: Adam + L2 vs AdamW */
const CODE_WD = [
  '# Adam + L2: the penalty goes into the gradient...',
  'g = g + wd * w',
  'm = b1 * m + (1 - b1) * g',
  'v = b2 * v + (1 - b2) * g * g',
  'w = w - lr * mh / (sqrt(vh) + eps)    # ...so wd * w is divided by sqrt(v)',
  '',
  '# AdamW: the decay never touches the gradient',
  'w = w - lr * mh / (sqrt(vh) + eps)',
  'w = w - lr * wd * w'
];
/* sqrt(v) per weight, measured by optimizers.py on the least-squares problem */
const WD_SQRTV = [0.40692, 0.14583, 4.05861];
const WD_SCALE = [1, 1, 8];
function wdFrames(v){
  const F = new Frames(), decoupled = v === 1;
  const inv = WD_SQRTV.map(x => 1 / x);
  const mean = inv.reduce((a, b) => a + b, 0) / inv.length;
  const rel = decoupled ? [1, 1, 1] : inv.map(x => x / mean);
  const draw = k => {
    const sh = [];
    sh.push(S.t(4.9, .7, decoupled ? {zh:'AdamW：衰減直接作用在 w 上', en:'AdamW: the decay is applied straight to w'} : {zh:'Adam + L2：衰減被 √v 除過', en:'Adam + L2: the decay is divided by sqrt(v)'}, {c:decoupled ? COL.tealL : COL.red, fs:.3}));
    for (let j = 0; j < 3; j++){
      const y = 1.3 + j * 1.15, on = j === k - 1, shown = j < k;
      sh.push(S.t(2.2, y + .3, 'weight ' + j, {anchor:'end', c:COL.purpleL, fs:.28}));
      sh.push(fitT(2.2, y + .66, {zh:'尺度 ' + WD_SCALE[j] + '，√v ' + WD_SQRTV[j].toFixed(2), en:'scale ' + WD_SCALE[j] + ', √v ' + WD_SQRTV[j].toFixed(2)}, {anchor:'end', c:COL.grey, fs:.2}));
      if (!shown) continue;
      const w = Math.max(rel[j] * 2.4, .04);
      sh.push(S.r(2.45, y, w, .5, on ? 'hot' : decoupled ? 'ok' : 'bad', '', {rx:.04}));
      sh.push(S.t(2.45 + w + .15, y + .33, rel[j].toFixed(2) + 'x', {anchor:'start', c:on ? COL.orangeL : decoupled ? COL.tealL : COL.red, fs:.26}));
    }
    return sh;
  };
  F.push({shapes:caps(draw(0), {cap:{zh:'同一份資料，三個權重，其中一個特徵被放大 8 倍', en:'the same data, three weights, one feature scaled up 8x'},
      cap2:{zh:'長條＝衰減實際拉動每個權重的力道（平均正規化成 1）', en:'bar = how hard the decay actually pulls on each weight (average normalised to 1)'}}),
    panels:[{lbl:'wd', chips:[chip('0.1', 'act')]}], line:decoupled ? 8 : 1,
    msg:decoupled ? {zh:'AdamW 把衰減從梯度裡拿出來，直接從 w 減掉 lr × wd × w。於是每個權重受到的衰減力道完全相同，跟它的梯度多大無關。',
                     en:'AdamW takes the decay out of the gradient and subtracts lr x wd x w from w directly. Every weight then feels exactly the same pull, regardless of how large its gradients are.'}
                  : {zh:'L2 正則化通常是把 wd × w 加進梯度實作的。但在 Adam 內部，這個和接著會被除以 √v——所以每個權重實際受到的衰減被縮放了 1/√v。',
                     en:'L2 regularisation is usually implemented by adding wd * w to the gradient. Inside Adam that sum then gets divided by sqrt(v) - so the decay each weight feels is scaled by 1 / sqrt(v).'}});
  [1, 2, 3].forEach(j => {
    F.push({shapes:caps(draw(j), {cap:{zh:'weight ' + (j - 1) + '：√v = ' + WD_SQRTV[j - 1].toFixed(2) + '，衰減力道 ' + rel[j - 1].toFixed(2) + ' 倍', en:'weight ' + (j - 1) + ': sqrt(v) = ' + WD_SQRTV[j - 1].toFixed(2) + ', decay strength ' + rel[j - 1].toFixed(2) + 'x'},
        cap2:decoupled ? {zh:'AdamW 對三個權重都是 1.00 倍', en:'AdamW gives all three exactly 1.00x'} : {zh:'1/√v 正規化後的相對力道', en:'1 / sqrt(v), normalised so the average is 1'}}),
      panels:[{lbl:'weight', chips:[chip(String(j - 1), 'hot')]}, {lbl:'decay', chips:[chip(rel[j - 1].toFixed(2) + 'x', decoupled ? 'ok' : (rel[j - 1] < .5 ? 'bad' : 'act'))]}], line:decoupled ? 8 : 4,
      msg:j === 3 ? (decoupled ? {zh:'三個都是 1.00 倍。這兩行就是 torch.optim.Adam(weight_decay=…) 和 torch.optim.AdamW 的全部差別，也是 AdamW 成為 transformer 訓練預設值的原因。',
                                  en:'All three are 1.00x. Those two lines are the entire difference between torch.optim.Adam(weight_decay=...) and torch.optim.AdamW, and why AdamW is the default for transformer training.'}
                               : {zh:'weight 2 的特徵被放大 8 倍，所以它的梯度遠大於其他人，√v = ' + WD_SQRTV[2].toFixed(2) + '。結果它受到的衰減只有 weight 0 的 ' + (rel[0] / rel[2]).toFixed(0) + ' 分之一、weight 1 的 ' + (rel[1] / rel[2]).toFixed(0) + ' 分之一——**最需要被壓住的權重被壓得最少**，方向剛好相反。',
                                  en:'Weight 2\'s feature is scaled 8x, so its gradients dwarf the others and sqrt(v) = ' + WD_SQRTV[2].toFixed(2) + '. It ends up decayed ' + (rel[0] / rel[2]).toFixed(0) + 'x less than weight 0 and ' + (rel[1] / rel[2]).toFixed(0) + 'x less than weight 1 - the weights that most need shrinking shrink least.'})
        : {zh:'衰減力道正比於 1/√v：梯度越大的權重，分母越大，被拉動得越少。', en:'The pull is proportional to 1 / sqrt(v): the larger a weight\'s gradients, the larger its denominator, the less it moves.'}});
  });
  return F.list;
}

/* ================================================================ tab 6: LeetCode 462 */
const CODE_462 = [
  'class Solution:                               # LeetCode 462',
  '    def minMoves2(self, nums):',
  '        nums = sorted(nums)',
  '        mid = nums[len(nums) // 2]            # the MEDIAN, not the mean',
  '        return sum(abs(x - mid) for x in nums)',
  '',
  '# d|x - c|/dc is the SIGN of (c - x), so the total derivative is',
  '# "how many points are below me minus how many are above me",',
  '# which is zero exactly at the median.'
];
const L462 = [1, 2, 3, 100];
const movesTo = c => L462.reduce((s, x) => s + Math.abs(x - c), 0);
function lc462Frames(v){
  const useMean = v === 1;
  const F = new Frames();
  const sorted = L462.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = Math.round(L462.reduce((a, b) => a + b, 0) / L462.length);
  const target = useMean ? mean : median;
  const AX0 = .8, AX1 = 9.4, lo = 0, hi = 105;
  const nx = x => AX0 + (AX1 - AX0) * (x - lo) / (hi - lo);
  const draw = (c, upto) => {
    const sh = [];
    sh.push(S.e(AX0, 2.3, AX1, 2.3, {s:'idle', w:.025, arrow:false}));
    [0, 25, 50, 75, 100].forEach(t => sh.push(S.t(nx(t), 2.72, String(t), {c:COL.grey, fs:.2})));
    /* the target */
    sh.push(S.e(nx(c), 1.5, nx(c), 2.22, {s:useMean ? 'bad' : 'ok', w:.05}));
    sh.push(S.t(nx(c), 1.25, (useMean ? 'mean ' : 'median ') + c, {c:useMean ? COL.red : COL.tealL, fs:.3}));
    /* the four points; 1, 2 and 3 sit almost on top of each other, so stagger them */
    L462.forEach((x, i) => {
      const done = i < upto;
      if (done) sh.push(S.e(nx(x), 2.3, nx(c), 2.3, {s:'hot', w:.06, arrow:false}));
      sh.push(S.c(nx(x), 2.3, .12, done ? 'ok' : 'act', ''));
      const lift = x < 10 ? [3.0, 3.34, 3.0][i % 3] : 3.0;
      sh.push(S.e(nx(x), 2.44, nx(x), lift - .18, {s:'idle', w:.02, arrow:false, o:.6}));
      sh.push(S.t(nx(x), lift, String(x), {c:COL.purpleL, fs:.26}));
    });
    /* the arithmetic, as a list instead of labels on the line */
    let tot = 0;
    L462.forEach((x, i) => {
      if (i >= upto) return;
      tot += Math.abs(x - c);
      sh.push(S.t(.7, 3.88 + i * .40, '|' + x + ' − ' + c + '| = ' + Math.abs(x - c),
                  {anchor:'start', c:COL.orangeL, fs:.28}));
    });
    if (upto > 0) sh.push(S.t(4.3, 3.88 + (upto - 1) * .40, (upto === L462.length ? '= ' : '… ') + tot,
                              {anchor:'start', c:upto === L462.length ? (useMean ? COL.red : COL.tealL) : COL.grey, fs:.3}));
    return sh;
  };
  F.push({shapes:caps(draw(target, 0), {cap:{zh:'nums = [' + L462.join(', ') + ']：把全部移到同一個值，最少要幾步？', en:'nums = [' + L462.join(', ') + ']: fewest single-unit moves to make them all equal'},
      cap2:useMean ? {zh:'這次用平均數 ' + mean + ' 當目標', en:'this time the target is the mean, ' + mean} : {zh:'用中位數 ' + median + ' 當目標', en:'the target is the median, ' + median}}),
    panels:[{lbl:'target', chips:[chip(String(target), useMean ? 'bad' : 'ok')]}, {lbl:'moves', chips:[chip('0', 'empty')]}], line:3,
    msg:useMean ? {zh:'平均數是最自然的猜測，而且它是對的——只是對另一個問題。平均數最小化的是「平方」誤差，也就是 Day 44 那條最小平方公式建立其上的東西。這題要的是絕對差之和。',
                   en:'The mean is the natural guess, and it is the right answer - to a different question. The mean minimises *squared* error, the thing Day 44\'s least-squares formula was built on. This problem asks for the sum of absolute differences.'}
                : {zh:'成本是 sum|x − c|。|x − c| 對 c 的導數是 c − x 的正負號，所以總導數就是「比我小的有幾個，減掉比我大的有幾個」——這個值剛好在中位數處為零。',
                   en:'The cost is sum|x - c|. The derivative of |x - c| with respect to c is the sign of c - x, so the total derivative is "how many points are below me minus how many are above me" - which is zero exactly at the median.'}});
  L462.forEach((x, i) => {
    const partial = L462.slice(0, i + 1).reduce((s, y) => s + Math.abs(y - target), 0);
    F.push({shapes:caps(draw(target, i + 1), {cap:{zh:'把 ' + x + ' 移到 ' + target + '：' + Math.abs(x - target) + ' 步（累計 ' + partial + '）', en:'moving ' + x + ' to ' + target + ' costs ' + Math.abs(x - target) + ' (running total ' + partial + ')'},
        cap2:null}),
      panels:[{lbl:'target', chips:[chip(String(target), useMean ? 'bad' : 'ok')]}, {lbl:'moves', chips:[chip(String(partial), 'hot')]}], line:4,
      msg:x === 100 ? {zh:'那個離群值 100 本身就要 ' + Math.abs(100 - target) + ' 步。' + (useMean ? '它把平均數整個拉過去，於是另外三個點全部付出代價。' : '中位數完全不理會它有多遠——只在乎它在哪一邊。') ,
                       en:'The outlier at 100 alone costs ' + Math.abs(100 - target) + '. ' + (useMean ? 'It dragged the mean towards itself, and the other three points pay for that.' : 'The median does not care how far away it is, only which side it is on.')}
        : {zh:'每一點的成本就是它到目標的距離。', en:'Each point costs its distance to the target.'}});
  });
  const other = useMean ? movesTo(median) : movesTo(mean);
  F.push({shapes:caps(draw(target, L462.length), {cap:{zh:(useMean ? '平均數 ' : '中位數 ') + target + ' 的總成本 ' + movesTo(target), en:'total with the ' + (useMean ? 'mean ' : 'median ') + target + ': ' + movesTo(target)},
      cap2:{zh:'另一種選法是 ' + other + (useMean ? '（中位數，比較小）' : '（平均數，比較大）'), en:'the other choice gives ' + other + (useMean ? ' (the median - smaller)' : ' (the mean - larger)')}}),
    panels:[{lbl:'moves', chips:[chip(String(movesTo(target)), useMean ? 'bad' : 'ok')]}], line:4,
    msg:useMean ? {zh:'平均數給 ' + movesTo(mean) + '，中位數給 ' + movesTo(median) + '。這個 bug 的惡劣之處：在 [1, 2, 3] 和 [1, 10, 2, 9] 上兩者答案一模一樣，隨手測一下會過——要有離群值才會分出差別，而且錯的那個只是「比較大」，看起來完全合理。',
                   en:'The mean gives ' + movesTo(mean) + ' and the median gives ' + movesTo(median) + '. What makes this nasty: on [1, 2, 3] and [1, 10, 2, 9] the two agree exactly, so a quick test passes. It takes an outlier to separate them, and the wrong answer is merely larger - entirely plausible.'}
              : {zh:'總共 ' + movesTo(median) + ' 步。元素個數是偶數時，2 到 3 之間的任何值都一樣好——最小值點是一個區間，不是一個點。拿梯度下降跑 sum|x − c| 會停在 2.354，成本一樣是 100。',
                 en:'A total of ' + movesTo(median) + '. With an even count, every value between 2 and 3 is equally good - the minimiser is an interval, not a point. Running gradient descent on sum|x - c| lands at 2.354, which also costs exactly 100.'}});
  return F.list;
}

/* ================================================================ tab 7: LeetCode 852 */
const CODE_852 = [
  'class Solution:                               # LeetCode 852',
  '    def peakIndexInMountainArray(self, arr):',
  '        lo, hi = 0, len(arr) - 1',
  '        while lo < hi:',
  '            mid = (lo + hi) // 2',
  '            if arr[mid] < arr[mid + 1]:       # the SIGN of the slope',
  '                lo = mid + 1                  # still going up',
  '            else:',
  '                hi = mid                      # already coming down',
  '        return lo'
];
function mountainArr(n, peak){
  const up = [], down = [];
  for (let i = 0; i <= peak; i++) up.push((i + 1) * 3);
  for (let i = 1; i < n - peak; i++) down.push(up[up.length - 1] - i * 2);
  return up.concat(down);
}
function lc852Frames(v){
  const F = new Frames();
  const N = 31, PEAK = 11, arr = mountainArr(N, PEAK);
  const linear = v === 1;
  const AX0 = .6, AW = 8.9, bw = AW / N, BASE = 4.75;
  const top = Math.max.apply(null, arr);
  const draw = (lo, hi, mid, seen) => {
    const sh = [];
    for (let i = 0; i < N; i++){
      const h = .35 + 2.9 * arr[i] / top;
      const inRange = i >= lo && i <= hi;
      const st = i === mid || i === mid + 1 ? 'hot' : seen.includes(i) ? 'act' : inRange ? 'idle' : 'done';
      sh.push(S.r(AX0 + i * bw, BASE - h, bw - .06, h, st, '', {rx:.03}));
    }
    sh.push(S.e(AX0, BASE + .03, AX0 + AW, BASE + .03, {s:'idle', w:.025, arrow:false}));
    sh.push(S.t(AX0 + lo * bw + bw / 2, BASE + .4, 'lo', {c:COL.purpleL, fs:.24}));
    sh.push(S.t(AX0 + hi * bw + bw / 2, BASE + .4, 'hi', {c:COL.purpleL, fs:.24}));
    sh.push(S.t(AX0 + PEAK * bw + bw / 2, BASE - .35 - 2.9 - .2, {zh:'真正的峰頂', en:'the real peak'}, {c:COL.tealL, fs:.22}));
    return sh;
  };
  if (linear){
    let best = 0;
    const seen = [];
    F.push({shapes:caps(draw(0, N - 1, -1, []), {cap:{zh:'線性掃描：一個一個看過去', en:'a linear scan: look at every element'},
        cap2:{zh:'找得到答案，但題目要求 O(log n)', en:'it finds the answer, but the problem demands O(log n)'}}),
      panels:[{lbl:'looked at', chips:[chip('0', 'empty')]}], line:0,
      msg:{zh:'線性掃描當然找得到峰頂。問題是題目明確要求 O(log n)——在 10 萬個元素上，一個一個看是不被接受的。', en:'A scan finds the peak, of course. The catch is that the problem explicitly asks for O(log n) - on 100,000 elements, looking at each one is rejected.'}});
    [3, 7, 11, 16, 24, 30].forEach(i => {
      for (let j = seen.length; j <= i; j++){ seen.push(j); if (arr[j] > arr[best]) best = j; }
      F.push({shapes:caps(draw(0, N - 1, -1, seen.slice()), {cap:{zh:'看過 ' + (i + 1) + ' 個元素，目前最大在 index ' + best, en:'looked at ' + (i + 1) + ' elements, largest so far at index ' + best},
          cap2:{zh:'必須一路看到底才能確定', en:'it has to reach the end before it can be sure'}}),
        panels:[{lbl:'looked at', chips:[chip(String(i + 1), 'hot')]}, {lbl:'best', chips:[chip(String(best), 'act')]}], line:0,
        msg:i >= 30 ? {zh:'看完全部 ' + N + ' 個才敢回答。二分搜尋版本只要看 ' + (2 * Math.ceil(Math.log2(N))) + ' 個以內——在 4,001 個元素的山上是 24 個。', en:'It needed all ' + N + ' elements before answering. The binary search needs at most about ' + (2 * Math.ceil(Math.log2(N))) + ' - and 24 on a 4,001-element mountain.'}
          : {zh:'掃描不知道自己已經過了峰頂，因為它沒有利用「這是一座山」這個條件。', en:'The scan does not know it has passed the peak, because it never uses the fact that this is a mountain.'}});
    });
    return F.list;
  }
  let lo = 0, hi = N - 1, probes = 0;
  const seen = [];
  F.push({shapes:caps(draw(lo, hi, -1, []), {cap:{zh:'arr[mid] < arr[mid + 1] 就是斜率的正負號', en:'arr[mid] < arr[mid + 1] is the sign of the slope'},
      cap2:{zh:'在一座山上，這個比較在峰頂前全為真、峰頂後全為假——是一個排序好的布林陣列', en:'on a mountain it is true before the peak and false at or after it: a sorted boolean array'}}),
    panels:[{lbl:'lo', chips:[chip('0', 'act')]}, {lbl:'hi', chips:[chip(String(N - 1), 'act')]}, {lbl:'probes', chips:[chip('0', 'empty')]}], line:2,
    msg:{zh:'離散陣列在 mid 與 mid+1 之間的斜率，就只是一個比較。因為它在峰頂前恆真、峰頂後恆假，所以這是一個排序好的布林陣列，Day 23 的二分搜尋直接適用。',
         en:'The slope of a discrete array between mid and mid + 1 is just a comparison. Because it is true before the peak and false after it, this is a sorted boolean array and Day 23\'s binary search applies directly.'}});
  while (lo < hi){
    const mid = Math.floor((lo + hi) / 2), up = arr[mid] < arr[mid + 1];
    probes += 2; seen.push(mid, mid + 1);
    const nlo = up ? mid + 1 : lo, nhi = up ? hi : mid;
    F.push({shapes:caps(draw(lo, hi, mid, seen.slice()), {cap:{zh:'mid = ' + mid + '：arr[' + mid + '] = ' + arr[mid] + (up ? ' < ' : ' > ') + arr[mid + 1] + ' = arr[' + (mid + 1) + ']', en:'mid = ' + mid + ': arr[' + mid + '] = ' + arr[mid] + (up ? ' < ' : ' > ') + arr[mid + 1] + ' = arr[' + (mid + 1) + ']'},
        cap2:up ? {zh:'還在上坡 → 峰頂在右邊，lo = ' + nlo, en:'still going up -> the peak is to the right, lo = ' + nlo}
                : {zh:'已經在下坡 → mid 可能就是峰頂，hi = ' + nhi, en:'already coming down -> mid could be the peak, hi = ' + nhi}}),
      panels:[{lbl:'lo', chips:[chip(String(nlo), 'act')]}, {lbl:'hi', chips:[chip(String(nhi), 'act')]}, {lbl:'probes', chips:[chip(String(probes), 'hot')]}], line:up ? 6 : 8,
      msg:up ? {zh:'斜率是正的，所以峰頂一定嚴格在 mid 右邊——mid 本身可以直接丟掉，lo = mid + 1。', en:'The slope is positive, so the peak is strictly to the right of mid - mid itself can be discarded, lo = mid + 1.'}
             : {zh:'斜率是負的，所以峰頂在 mid 或它左邊。注意這裡是 hi = mid 而不是 mid − 1：mid 本身還可能是答案，丟掉它就會漏解。',
                en:'The slope is negative, so the peak is at mid or to its left. Note hi = mid, not mid - 1: mid itself may still be the answer, and discarding it loses the peak.'}});
    lo = nlo; hi = nhi;
  }
  F.push({shapes:caps(draw(lo, hi, -1, seen.slice()), {cap:{zh:'lo == hi == ' + lo + '：峰頂（值 ' + arr[lo] + '）', en:'lo == hi == ' + lo + ': the peak, value ' + arr[lo]},
      cap2:{zh:'看了 ' + probes + ' 個元素，而不是 ' + N + ' 個', en:'looked at ' + probes + ' elements instead of ' + N}}),
    panels:[{lbl:'peak', chips:[chip(String(lo), 'ok')]}, {lbl:'probes', chips:[chip(String(probes), 'ok')]}], line:9,
    msg:{zh:'說它是梯度上升，是因為它用到的唯一資訊就是斜率的正負號——只不過它不是照著斜率走一步，而是每探測一次就刪掉剩下空間的一半。連續版本是三分搜尋。那為什麼還需要梯度下降？因為維度：二分與三分搜尋統治一維單峰問題，碰到一百萬個參數都撐不住，而梯度一次就給出這一百萬個方向。',
         en:'Calling it gradient ascent is fair because the only information it uses is the sign of the slope - but instead of taking a step proportional to it, each probe deletes half the remaining space. The continuous twin is ternary search. So why does gradient descent exist at all? Dimensionality: binary and ternary search own one-dimensional unimodal problems and neither survives a million parameters, where the gradient gives a direction in all million at once.'}});
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title:{zh:'Day 45 — 梯度下降家族：SGD、momentum 與 Adam', en:'Day 45 - The gradient descent family: SGD, momentum and Adam'},
  sub:{zh:'學習率的精確上限、momentum 把 kappa 變成 sqrt(kappa)，以及 Adam 那些寫錯也不會報錯的細節。', en:'The exact learning-rate ceiling, momentum turning kappa into sqrt(kappa), and the Adam details that fail without ever raising.'},
  tabs:[
    {
      id:'ceiling', label:{zh:'學習率上限', en:'the lr ceiling'},
      stage:{zh:'f(w) = 0.5 (w0² + 20 w1²)：每步把每個座標乘上 1 − lr × 曲率', en:'f(w) = 0.5 (w0² + 20 w1²): each step multiplies each coordinate by 1 - lr x curvature'},
      view:VIEW,
      variants:CEIL_LRS.map(l => ({zh:'lr ' + l, en:'lr ' + l})),
      idea:{zh:'純梯度下降在二次碗上只做一件事：每步把每個座標乘上 (1 − lr × 曲率)。所以收斂條件是白送的——所有乘數絕對值都要小於 1，即 lr < 2 / 最大曲率 = 0.1。0.0999 收斂、0.1001 發散，而且不會丟任何例外：loss 只是一路長到 inf 再變 nan，用 lr = 0.11 從 (4, 1) 跑 400 步後 w = [2.3e-20, 4.7e+31]，一個座標完美、另一個飛走。上限以下則是另一半問題：最陡的方向決定你能用多大的步長，最平的方向決定你實際的進度，兩者的比值就是條件數。',
            en:'Plain gradient descent on a quadratic does exactly one thing: every step it multiplies each coordinate by (1 - lr x curvature). The convergence condition is therefore free - every multiplier must be below 1 in absolute value, i.e. lr < 2 / largest curvature = 0.1. At 0.0999 it converges, at 0.1001 it diverges, and nothing ever raises: the loss just grows to inf and then nan. After 400 steps at lr 0.11 from (4, 1), w = [2.3e-20, 4.7e+31] - one coordinate perfect, the other gone. Below the ceiling sits the other half of the problem: the steepest direction caps the step size while the flattest sets the progress, and their ratio is the condition number.'},
      legend:[['#ff9736', {zh:'目前這一步', en:'the current step'}], ['#9d6bff', {zh:'走過的路徑', en:'the path so far'}], ['#3fe0dd', {zh:'最低點與等高線', en:'the minimum and the contours'}], ['#ff5c5c', {zh:'乘數大於 1：發散', en:'multiplier above 1: diverging'}]],
      get code(){ return CODE_CEIL; },
      build:ceilFrames
    },
    {
      id:'momentum', label:{zh:'momentum', en:'momentum'},
      stage:{zh:'同一個山谷：誤差每步縮小的比例', en:'the same ravine: how much the error shrinks per step'},
      view:VIEW,
      variants:[{zh:'純梯度下降', en:'plain GD'}, {zh:'加 momentum', en:'with momentum'}],
      idea:{zh:'momentum 累積過去梯度的和再沿著它走。鋸齒方向的正負梯度在總和裡互相抵消，谷底方向的則一致累加。理論：每步誤差乘數從 (kappa−1)/(kappa+1) = 0.9048 改善成 (√kappa−1)/(√kappa+1) = 0.6345，也就是步數對條件數的依賴從 kappa 變成 √kappa。實測對理論：純 GD 139 步對 138、momentum 39 步對 30（差距來自漸近速率忽略的暫態），仍少 3.6 倍。注意 beta = 0.9 不是自然常數：這個碗最好的 beta 是 0.4026，固定 lr = 0.04 時 beta 0 要 338 步、0.4026 要 190 步、0.9 要 245 步——抄來的 0.9 比調過的差。',
            en:'Momentum accumulates a running sum of past gradients and steps along it. Along the zigzag the alternating gradients cancel inside the sum; along the valley floor they agree and add. Theory: the per-step error multiplier improves from (kappa-1)/(kappa+1) = 0.9048 to (sqrt(kappa)-1)/(sqrt(kappa)+1) = 0.6345, so the step count depends on sqrt(kappa) rather than kappa. Measured against theory: plain GD 139 steps against 138, momentum 39 against 30 (the gap is the transient the asymptotic rate ignores) - still 3.6x fewer. And beta = 0.9 is not a constant of nature: the best beta for this bowl is 0.4026, and at a fixed lr 0.04, beta 0 takes 338 steps, 0.4026 takes 190 and 0.9 takes 245.'},
      legend:[['#ff9736', {zh:'目前這一步', en:'the current step'}], ['#3fe0dd', {zh:'momentum 的路徑與誤差曲線', en:'the momentum path and its error curve'}], ['#9d6bff', {zh:'理論預測（虛線）', en:'the theoretical prediction (dashed)'}], ['#8fa3ac', {zh:'等高線', en:'contours'}]],
      get code(){ return CODE_MOM; },
      build:momFrames
    },
    {
      id:'adam', label:{zh:'Adam 的 bias correction', en:'Adam bias correction'},
      stage:{zh:'m 與 v 都從零開始，而它們的偏差不會互相抵消', en:'m and v both start at zero, and their biases do not cancel'},
      view:VIEW,
      variants:[{zh:'有修正', en:'corrected'}, {zh:'少了修正', en:'correction removed'}],
      idea:{zh:'Adam 維護梯度的移動平均 m 與梯度平方的移動平均 v，步長是 lr × m / √v，所以每個座標大約都走 lr、與梯度尺度無關。兩個平均都從零開始因而早期被低估，標準修正是除以 1 − 0.9^t 與 1 − 0.999^t。坊間說少了修正會讓前幾步「太小」——實際上太大，因為 v 在根號底下：未修正 ÷ 已修正 = (1 − 0.9^t)/√(1 − 0.999^t)，t=1 是 3.16 倍、t=12 最糟 6.57 倍、t=1000 還有 1.26 倍。在真問題上用 lr 0.05 量，前三步是 0.087/0.086/0.086（有修正）對 0.274/0.363/0.404（沒有）。',
            en:'Adam keeps a running average m of the gradient and v of its square; the step is lr x m / sqrt(v), so each coordinate moves about lr regardless of gradient scale. Both averages start at zero and are underestimates early, so the standard fix divides by 1 - 0.9^t and 1 - 0.999^t. Folklore says dropping it makes early steps too small; it makes them too large, because v is under a square root: uncorrected / corrected = (1 - 0.9^t) / sqrt(1 - 0.999^t), which is 3.16x at t=1, a worst 6.57x at t=12, and still 1.26x at t=1000. Measured on the real problem at lr 0.05, the first three steps are 0.087/0.086/0.086 corrected against 0.274/0.363/0.404 without.'},
      legend:[['#ff9736', {zh:'目前這一步', en:'the current step'}], ['#3fe0dd', {zh:'修正後的路徑', en:'the corrected path'}], ['#ff5c5c', {zh:'少了修正的路徑', en:'the uncorrected path'}], ['#9d6bff', {zh:'m 與 v 的目前值', en:'the current m and v'}]],
      get code(){ return CODE_ADAM; },
      build:adamFrames
    },
    {
      id:'eps', label:{zh:'epsilon 放哪裡', en:'where the epsilon goes'},
      stage:{zh:'√v + eps 與 √(v + eps)：看起來只是括號位置', en:'sqrt(v) + eps against sqrt(v + eps): it looks like bracket placement'},
      view:VIEW,
      idea:{zh:'兩種寫法都能防止除以零，但 eps = 1e-8 的平方根是 1e-4，大了四個數量級。梯度健康時 √v 遠大於 eps，兩者完全一樣——所以一般測試看不出來。但在 v = 1e-12 的座標上兩個分母差 99 倍、v = 1e-16 時差 5000 倍，第二種寫法把步長硬生生除掉。這只在梯度很小的地方發作：很少被用到的 embedding 列、幾乎死掉的神經元——也就是你最不會去看的地方。PyTorch 用的是 √v + eps。',
            en:'Both forms prevent a division by zero, but eps = 1e-8 has sqrt(eps) = 1e-4, four orders of magnitude larger. When gradients are healthy sqrt(v) dwarfs eps and the two agree exactly, which is why an ordinary test never catches it. On a coordinate with v = 1e-12 the denominators differ by 99x, and at v = 1e-16 by 5000x, with the second form dividing the step away. It only fires where gradients are small - a rarely-used embedding row, a nearly-dead unit - exactly where nobody looks. PyTorch uses sqrt(v) + eps.'},
      legend:[['#ff9736', {zh:'目前這一列', en:'the current row'}], ['#3fe0dd', {zh:'√v + eps（PyTorch）', en:'sqrt(v) + eps (PyTorch)'}], ['#ff5c5c', {zh:'√(v + eps)，以及它造成的倍數', en:'sqrt(v + eps), and the factor it costs'}], ['#9d6bff', {zh:'v 的值', en:'the value of v'}]],
      get code(){ return CODE_EPS; },
      build:epsFrames
    },
    {
      id:'adamw', label:{zh:'Adam + L2 vs AdamW', en:'Adam + L2 vs AdamW'},
      stage:{zh:'weight decay 實際拉動每個權重的力道', en:'how hard weight decay actually pulls on each weight'},
      view:VIEW,
      variants:[{zh:'Adam + L2', en:'Adam + L2'}, {zh:'AdamW', en:'AdamW'}],
      idea:{zh:'L2 正則化通常是把 wd × w 加進梯度。但在 Adam 內部這個和接著會被除以 √v，所以每個權重受到的衰減被縮放了 1/√v——梯度最大的權重被衰減得最少，方向剛好相反。在最小平方問題上跑 400 步後三個權重的 √v 是 [0.41, 0.15, 4.06]，把平均衰減強度正規化成 1，Adam + L2 施加的是 [0.77, 2.15, 0.08]、AdamW 是 [1, 1, 1]：8 倍尺度特徵上的那個權重被衰減得比 weight 0 少 10 倍、比 weight 1 少 28 倍。AdamW 的修法是不讓衰減經過梯度，直接從 w 減掉 lr × wd × w——這兩行就是 torch.optim.Adam(weight_decay=…) 和 torch.optim.AdamW 的全部差別。',
            en:'L2 regularisation is usually implemented by adding wd * w to the gradient. Inside Adam that sum is then divided by sqrt(v), so the decay each weight feels is scaled by 1 / sqrt(v) - the weights with the largest gradients are decayed least, which is backwards. After 400 steps on the least-squares problem the three weights have sqrt(v) = [0.41, 0.15, 4.06]; normalising the average decay strength to 1, Adam + L2 applies [0.77, 2.15, 0.08] while AdamW applies [1, 1, 1]. The weight on the 8x-scaled feature is decayed 10x less than weight 0 and 28x less than weight 1. AdamW keeps the decay out of the gradient and subtracts lr * wd * w from w directly - those two lines are the entire difference between torch.optim.Adam(weight_decay=...) and torch.optim.AdamW.'},
      legend:[['#ff9736', {zh:'目前這個權重', en:'the current weight'}], ['#3fe0dd', {zh:'AdamW：三個都一樣', en:'AdamW: identical for all three'}], ['#ff5c5c', {zh:'Adam + L2：被 1/√v 扭曲', en:'Adam + L2: distorted by 1 / sqrt(v)'}], ['#9d6bff', {zh:'權重與它的 √v', en:'the weight and its sqrt(v)'}]],
      get code(){ return CODE_WD; },
      build:wdFrames
    },
    {
      id:'lc462', label:{zh:'中位數（LC 462）', en:'the median (LC 462)'},
      stage:{zh:'把 [1, 2, 3, 100] 全部移到同一個值，最少幾步？', en:'fewest single-unit moves to make [1, 2, 3, 100] all equal'},
      view:VIEW,
      variants:[{zh:'中位數（正確）', en:'median (right)'}, {zh:'平均數（錯誤）', en:'mean (wrong)'}],
      idea:{zh:'成本是 sum|x − c|。|x − c| 對 c 的導數是正負號，所以總導數是「比我小的個數減掉比我大的個數」，剛好在中位數處為零。陷阱是伸手去拿平均數——平均數最小化的是平方誤差，也就是 Day 44 最小平方建立其上的東西。惡劣之處在於 [1, 2, 3] 和 [1, 10, 2, 9] 上兩者答案一模一樣，隨手測會過；要有離群值才分得出來，而且錯的答案只是「比較大」（146 對 100），看起來完全合理。拿今天的梯度下降跑 sum|x − c| 會停在 2.354，成本一樣是 100：元素個數為偶數時最小值點是一個區間。注意那裡的 1/√k 排程是在做實事——正負號梯度在最佳解附近不會變小，固定步長會永遠震盪，跟 mini-batch 的雜訊地板是同一個機制。',
            en:'The cost is sum|x - c|. The derivative of |x - c| is a sign, so the total derivative is "how many points are below me minus how many are above", which is zero exactly at the median. The trap is reaching for the mean, which minimises squared error - the thing Day 44\'s least squares was built on. What makes it nasty is that on [1, 2, 3] and [1, 10, 2, 9] the two agree exactly, so a quick test passes; it takes an outlier to separate them, and the wrong answer is merely larger (146 against 100) and entirely plausible. Running today\'s gradient descent on sum|x - c| lands at 2.354 and also costs 100, because with an even count the minimiser is an interval. The 1/sqrt(k) schedule there is doing real work: a sign gradient never shrinks near the optimum, so a fixed step size would oscillate forever - the same mechanism as the mini-batch noise floor.'},
      legend:[['#ff9736', {zh:'這一點要移動的距離', en:'the distance this point must move'}], ['#3fe0dd', {zh:'中位數，以及已經算完的點', en:'the median, and points already counted'}], ['#ff5c5c', {zh:'平均數（錯的目標）', en:'the mean: the wrong target'}], ['#9d6bff', {zh:'資料點', en:'the data points'}]],
      get code(){ return CODE_462; },
      build:lc462Frames
    },
    {
      id:'lc852', label:{zh:'山頂（LC 852）', en:'the peak (LC 852)'},
      stage:{zh:'嚴格遞增再嚴格遞減：找出峰頂的索引', en:'strictly up then strictly down: find the index of the peak'},
      view:VIEW,
      variants:[{zh:'二分搜尋 O(log n)', en:'binary search, O(log n)'}, {zh:'線性掃描 O(n)', en:'linear scan, O(n)'}],
      idea:{zh:'離散陣列在 mid 與 mid+1 之間的斜率就只是 arr[mid] < arr[mid+1] 這個比較。在一座山上，它在峰頂前恆為真、峰頂及之後恆為假——是一個排序好的布林陣列，Day 23 的二分搜尋直接適用。關鍵細節是下坡時寫 hi = mid 而不是 mid − 1：mid 本身還可能是峰頂。題目明確要求 O(log n)，所以線性掃描雖然答案正確仍不被接受；31 個元素的山只要看 10 個，4,001 個元素的山只要 24 個。說它是梯度上升很公平，因為它用到的唯一資訊就是斜率正負號——只是它不照斜率走一步，而是每次探測刪掉一半空間。那為什麼還需要梯度下降？維度：二分與三分搜尋統治一維單峰問題，碰到一百萬個參數都撐不住。',
            en:'The slope of a discrete array between mid and mid + 1 is just the comparison arr[mid] < arr[mid + 1]. On a mountain it is true before the peak and false at or after it - a sorted boolean array, so Day 23\'s binary search applies directly. The detail that matters is writing hi = mid rather than mid - 1 on the way down: mid itself may still be the peak. The problem demands O(log n), so a linear scan is rejected despite being correct; a 31-element mountain needs 10 probes and a 4,001-element one needs 24. Calling it gradient ascent is fair, since the only information used is the sign of the slope - but each probe deletes half the space instead of taking a step. Why have gradient descent at all? Dimensionality: binary and ternary search own one-dimensional unimodal problems and neither survives a million parameters.'},
      legend:[['#ff9736', {zh:'正在比較的 mid 與 mid+1', en:'the mid and mid+1 being compared'}], ['#9d6bff', {zh:'已經看過的元素', en:'elements already looked at'}], ['#0a6b74', {zh:'還在搜尋範圍內', en:'still in the search range'}], ['#2f5661', {zh:'已經被刪掉', en:'already discarded'}]],
      get code(){ return CODE_852; },
      build:lc852Frames
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
