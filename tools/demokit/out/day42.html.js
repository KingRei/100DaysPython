// DAY: 42
// TITLE_ZH: 隨機化演算法與蒙地卡羅：用亂數做估計，以及怎麼測試用到亂數的程式
// TITLE_EN: Randomized algorithms and Monte Carlo: estimating with random numbers, and testing code that uses them
// SUB_ZH: 往正方形丟飛鏢，落在四分之一圓裡的比例乘 4 就是 π：丟 1,000 支誤差約 0.05，丟 100,000 支約 0.005——多丟 100 倍，誤差只少 10 倍。亂數程式最難的是測試：洗牌、蓄水池抽樣、拒絕取樣、加權抽樣，各有一個只差 1 的寫法讓結果偏掉，丟一萬次看不出來；把每一種亂數選擇都走一遍、用分數精確加總，偏差就無處可躲。
// SUB_EN: Throw darts at a square; four times the share that lands in the quarter circle is π. 1,000 darts give an error of about 0.05, 100,000 about 0.005 - 100 times the darts for an error only 10 times smaller. The hard part of random code is testing it: shuffling, reservoir sampling, rejection sampling and weighted picks each have an off-by-one that skews the result where ten thousand trials cannot see it. Replay every possible sequence of random choices and add them up as exact fractions, and the bias has nowhere to hide.
// FOLDER: day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo
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
const box = (x, y, w, h, s, lab, fs) => S.r(x, y, w, h, s, lab, {fs:fitFs(String(lab), w, fs || h * .5)});
const pyList = a => '[' + a.join(', ') + ']';
const fmt = n => { const s = String(n), neg = s[0] === '-', d = neg ? s.slice(1) : s;
  return (neg ? '-' : '') + d.replace(/\B(?=(\d{3})+(?!\d))/g, ','); };
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
/* a dot: a zero-length round-capped line (no glow filter, cheap for thousands) */
const dot = (x, y, s, w) => S.e(x, y, x + .001, y, {s:s, w:w || .075, arrow:false, o:1});
const gcdN = (a, b) => { while (b) [a, b] = [b, a % b]; return a; };
const frac = (k, n) => { if (!k) return '0'; const g = gcdN(k, n); return (n / g === 1) ? String(k / g) : (k / g) + '/' + (n / g); };

/* ---------------------------------------------- seeded PRNG (mulberry32) */
function rngOf(seed){
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rrOf = rng => n => Math.floor(rng() * n);

/* exact_distribution, as in randomized.py: replay every sequence of randrange
   choices once (an odometer), weight each run by the product of 1/bound */
function everyRun(run){
  const out = []; let ch = [];
  for (;;){
    let pos = 0; const bounds = [];
    const rr = n => { if (pos >= ch.length) ch.push(0); bounds.push(n); return ch[pos++]; };
    const res = run(rr);
    ch = ch.slice(0, pos);
    out.push({res:res, choices:ch.slice(), bounds:bounds});
    let k = pos - 1;
    while (k >= 0 && ch[k] + 1 >= bounds[k]) k--;
    if (k < 0) break;
    ch[k]++; ch = ch.slice(0, k + 1);
  }
  return out;
}

/* ================================================================ tab 1: Monte Carlo pi */
const CODE_PI = [
  'def estimate_pi(n, rng):',
  '    inside = 0',
  '    for _ in range(n):',
  '        x, y = rng.random(), rng.random()',
  '        if x * x + y * y < 1.0:',
  '            inside += 1',
  '    return 4 * inside / n',
  '',
  '# one dart: 4 with chance pi/4, else 0',
  'SIGMA_PI = 4 * sqrt(pi/4 * (1 - pi/4))   # 1.642',
  'def pi_std(n): return SIGMA_PI / sqrt(n)'
];
const SIGMA_PI = 4 * Math.sqrt(Math.PI / 4 * (1 - Math.PI / 4));
const PX0 = .5, PY0 = .35, PSZ = 4.6;
function piStage(darts, n, o){
  const sh = [];
  sh.push(S.r(PX0, PY0, PSZ, PSZ, 'soft', ''));
  const cx = PX0, cy = PY0 + PSZ, K = 40;
  for (let k = 0; k < K; k++){
    const a0 = Math.PI / 2 * k / K, a1 = Math.PI / 2 * (k + 1) / K;
    sh.push(S.e(cx + PSZ * Math.cos(a0), cy - PSZ * Math.sin(a0), cx + PSZ * Math.cos(a1), cy - PSZ * Math.sin(a1),
                {s:'act', w:.05, arrow:false}));
  }
  for (let i = 0; i < n; i++){
    const d = darts[i], last = o.hiLast && i === n - 1;
    sh.push(dot(PX0 + d[0] * PSZ, PY0 + (1 - d[1]) * PSZ, d[2] ? 'ok' : 'hot', last ? .2 : (n > 600 ? .06 : .09)));
  }
  sh.push(S.t(PX0, PY0 + PSZ + .32, '0', {c:COL.grey, fs:.26}));
  sh.push(S.t(PX0 + PSZ, PY0 + PSZ + .32, '1', {c:COL.grey, fs:.26}));
  sh.push(S.t(PX0 - .2, PY0 + .1, '1', {c:COL.grey, fs:.26}));
  (o.side || []).forEach((l, k) => sh.push(fitT(5.55, .75 + k * .62, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .32})));
  return caps(sh, o);
}
function piFrames(v){ return v === 1 ? lawFrames() : dartFrames(); }
function dartFrames(){
  const F = new Frames(), rng = rngOf(42), darts = [];
  for (let i = 0; i < 2000; i++){ const x = rng(), y = rng(); darts.push([x, y, x * x + y * y < 1 ? 1 : 0]); }
  const insideUpTo = n => darts.slice(0, n).reduce((s, d) => s + d[2], 0);
  F.push({shapes:piStage(darts, 0, {side:[[{zh:'正方形面積 1', en:'square: area 1'}, COL.grey],
                                          [{zh:'四分之一圓面積 π/4', en:'quarter circle: area π/4'}, COL.purpleL]],
                                   cap:{zh:'隨機丟一支飛鏢，落在圓裡的機率就是兩塊面積的比 π/4', en:'a random dart lands inside with probability = ratio of the areas = π/4'},
                                   cap2:{zh:'所以 4 × (落在圓裡的比例) 會趨近 π', en:'so 4 x (share that lands inside) tends to π'}}),
    panels:[{lbl:'n', chips:[chip('0', 'empty')]}], line:0,
    msg:{zh:'Monte Carlo 的想法：一個很難直接算的數，如果能寫成「某件隨機事件發生的機率」，就可以用丟骰子的次數去估。正方形 [0,1] × [0,1] 的面積是 1，裡面以原點為圓心、半徑 1 的四分之一圓面積是 π/4。均勻丟一支飛鏢，落在圓裡的機率就是 π/4。',
         en:'The Monte Carlo idea: if a number that is hard to compute can be written as "the probability that some random event happens", you can estimate it by counting how often the event happens. The unit square has area 1; the quarter circle of radius 1 around the origin has area π/4. A uniformly random dart lands inside with probability exactly π/4.'}});
  [1, 10, 50, 100, 300, 1000, 2000].forEach(n => {
    const ins = insideUpTo(n), est = 4 * ins / n, err = Math.abs(est - Math.PI), se = SIGMA_PI / Math.sqrt(n), d = darts[n - 1];
    F.push({shapes:piStage(darts, n, {hiLast:n <= 10,
        side:[['n = ' + fmt(n), COL.purpleL, .36],
              [{zh:'落在圓裡 ' + fmt(ins), en:'inside ' + fmt(ins)}, COL.tealL],
              ['π ≈ 4 × ' + fmt(ins) + ' / ' + fmt(n) + ' = ' + est.toFixed(4), COL.orangeL],
              [{zh:'實際誤差 ' + err.toFixed(4), en:'actual error ' + err.toFixed(4)}, COL.pale],
              [{zh:'一個標準誤差 1.642 / √n = ' + se.toFixed(4), en:'one standard error 1.642 / √n = ' + se.toFixed(4)}, COL.grey, .28]],
        cap:n === 1 ? {zh:'第一支：(' + d[0].toFixed(3) + ', ' + d[1].toFixed(3) + ')，x² + y² = ' + (d[0] * d[0] + d[1] * d[1]).toFixed(3) + (d[2] ? ' < 1，在圓裡' : ' ≥ 1，在圓外'),
                                en:'dart 1: (' + d[0].toFixed(3) + ', ' + d[1].toFixed(3) + '), x² + y² = ' + (d[0] * d[0] + d[1] * d[1]).toFixed(3) + (d[2] ? ' < 1, inside' : ' >= 1, outside')}
                    : {zh:n + ' 支飛鏢：估計值 ' + est.toFixed(4) + '，誤差 ' + err.toFixed(4), en:n + ' darts: estimate ' + est.toFixed(4) + ', error ' + err.toFixed(4)},
        cap2:{zh:'青色：圓裡（算 4 分）；橘色：圓外（算 0 分）', en:'teal: inside (scores 4); orange: outside (scores 0)'}}),
      panels:[{lbl:'n', chips:[chip(fmt(n), 'act')]}, {lbl:'inside', chips:[chip(fmt(ins), 'ok')]}, {lbl:'π ≈', chips:[chip(est.toFixed(4), 'hot')]}],
      line:n === 1 ? 4 : 6,
      msg:n === 1 ? {zh:'每支飛鏢就是兩個 rng.random()。判斷 x² + y² < 1 不需要開根號——距離小於 1 和距離平方小於 1 是同一件事。這支飛鏢單獨看只能給出 4 或 0，離 π 很遠；估計值是很多支平均出來的。',
                      en:'Each dart is two calls to rng.random(). The test x² + y² < 1 needs no square root - distance below 1 and squared distance below 1 are the same thing. On its own a dart only says 4 or 0, nowhere near π; the estimate is the average over many darts.'}
        : n === 100 ? {zh:'100 支：估計值 ' + est.toFixed(4) + '。一支飛鏢的得分是「4 或 0」，它的標準差是 4 × √(π/4 × (1 − π/4)) = 1.642；平均 n 支以後，標準差除以 √n，也就是 0.164——這就是「一個標準誤差」。實際誤差大概會落在這個數字附近。',
                       en:'100 darts: estimate ' + est.toFixed(4) + '. One dart scores 4 or 0, and that score has standard deviation 4 x √(π/4 x (1 - π/4)) = 1.642; averaging n darts divides it by √n, giving 0.164 - one standard error. The actual error usually lands around that size.'}
        : n === 2000 ? {zh:'2,000 支：誤差 ' + err.toFixed(4) + '，標準誤差 ' + se.toFixed(4) + '。從 100 支到 2,000 支多丟 20 倍，標準誤差只少 √20 ≈ 4.5 倍。這是 Monte Carlo 的代價：要多一位正確小數，飛鏢要多 100 倍。下一個選項把這條規律畫出來。',
                        en:'2,000 darts: error ' + err.toFixed(4) + ', standard error ' + se.toFixed(4) + '. Going from 100 to 2,000 darts is 20 times the work but only √20 ≈ 4.5 times less error. That is the price of Monte Carlo: one more correct decimal costs 100 times the darts. The next option draws the law.'}
        : {zh:n + ' 支：落在圓裡 ' + ins + ' 支，π ≈ ' + est.toFixed(4) + '。估計值會上下晃，因為每支飛鏢都是隨機的；同一個 seed 會重現同一批飛鏢，這是能測試 Monte Carlo 程式的前提。',
           en:n + ' darts: ' + ins + ' inside, π ≈ ' + est.toFixed(4) + '. The estimate wobbles because every dart is random; the same seed reproduces the same darts, which is what makes Monte Carlo code testable at all.'}});
  });
  return F.list;
}
/* rms error over many runs vs the 1.642 / sqrt(N) law */
const LAW = [[100, 400], [1000, 200], [10000, 100], [100000, 20]];
function rmsRuns(){
  const rng = rngOf(7);
  return LAW.map(([n, runs]) => {
    let s2 = 0;
    for (let r = 0; r < runs; r++){
      let ins = 0;
      for (let i = 0; i < n; i++){ const x = rng(), y = rng(); if (x * x + y * y < 1) ins++; }
      const e = 4 * ins / n - Math.PI; s2 += e * e;
    }
    return {n:n, runs:runs, rms:Math.sqrt(s2 / runs), law:SIGMA_PI / Math.sqrt(n)};
  });
}
const LX = n => 1.2 + (Math.log10(n) - 2) / 3 * 4.2;
const LY = e => .5 + (-.5 - Math.log10(e)) / 3 * 4.1;
function lawStage(pts, upto, o){
  const sh = [];
  sh.push(S.e(1.2, 4.6, 5.6, 4.6, {s:'idle', w:.03, arrow:false}));
  sh.push(S.e(1.2, 4.6, 1.2, .35, {s:'idle', w:.03, arrow:false}));
  [100, 1000, 10000, 100000].forEach(n => sh.push(S.t(LX(n), 4.95, fmt(n), {c:COL.grey, fs:.24})));
  [.1, .01, .001].forEach(e => sh.push(S.t(1.08, LY(e) + .08, String(e), {c:COL.grey, fs:.24, anchor:'end'})));
  sh.push(S.t(3.4, 5.25, {zh:'飛鏢數 n（對數刻度）', en:'darts n (log scale)'}, {c:COL.grey, fs:.24}));
  sh.push(S.e(LX(100), LY(SIGMA_PI / 10), LX(100000), LY(SIGMA_PI / Math.sqrt(100000)), {s:'act', w:.04, arrow:false, dash:'.14 .10'}));
  for (let k = 0; k < upto; k++){
    const p = pts[k];
    sh.push(dot(LX(p.n), LY(p.rms), k === upto - 1 ? 'hot' : 'ok', .2));
  }
  (o.side || []).forEach((l, k) => sh.push(fitT(5.95, .75 + k * .6, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
  return caps(sh, o);
}
function lawFrames(){
  const F = new Frames(), pts = rmsRuns();
  F.push({shapes:lawStage(pts, 0, {side:[[{zh:'紫色虛線：1.642 / √n', en:'dashed purple: 1.642 / √n'}, COL.purpleL]],
      cap:{zh:'對數座標上，誤差 ∝ 1/√n 是一條斜率 −1/2 的直線', en:'on log-log axes, error ∝ 1/√n is a straight line of slope -1/2'},
      cap2:{zh:'點：同一個 n 重跑很多次，誤差平方取平均再開根號（root-mean-square）', en:'dots: rerun many times at the same n, then the root-mean-square error'}}),
    panels:[], line:9,
    msg:{zh:'一次估計的誤差是運氣，看不出規律；要看規律就得在同一個 n 重跑很多次，量平均起來有多大。理論值是 1.642 / √n：n 變 100 倍，√n 變 10 倍，誤差少 10 倍。在兩軸都取對數的圖上，這是一條每往右一格（n × 10）就往下半格（誤差 ÷ √10）的直線。',
         en:'The error of one estimate is luck; to see the law you rerun many times at the same n and measure how big the error is on average. Theory says 1.642 / √n: 100 times the darts is 10 times √n, so 10 times less error. With both axes logarithmic that is a straight line that drops half a step (error ÷ √10) for every step right (n x 10).'}});
  pts.forEach((p, k) => {
    F.push({shapes:lawStage(pts, k + 1, {side:[['n = ' + fmt(p.n), COL.purpleL, .34],
          [{zh:'重跑 ' + p.runs + ' 次', en:p.runs + ' runs'}, COL.grey],
          ['rms ' + p.rms.toFixed(5), COL.orangeL],
          [{zh:'理論 ' + p.law.toFixed(5), en:'law ' + p.law.toFixed(5)}, COL.purpleL],
          [{zh:'比值 ' + (p.rms / p.law).toFixed(2), en:'ratio ' + (p.rms / p.law).toFixed(2)}, COL.pale]],
        cap:{zh:'n = ' + fmt(p.n) + '：實測 ' + p.rms.toFixed(5) + '，理論 ' + p.law.toFixed(5), en:'n = ' + fmt(p.n) + ': measured ' + p.rms.toFixed(5) + ', law ' + p.law.toFixed(5)},
        cap2:k ? {zh:'比上一個點多 10 倍飛鏢，誤差變成 ' + (p.rms / pts[k - 1].rms).toFixed(2) + ' 倍（理論 0.32 倍）', en:'10x the darts of the previous dot; error x ' + (p.rms / pts[k - 1].rms).toFixed(2) + ' (law: x 0.32)'}
               : {zh:'第一個點：100 支飛鏢，誤差大約 0.16', en:'first dot: 100 darts, error around 0.16'}}),
      panels:[{lbl:'n', chips:[chip(fmt(p.n), 'act')]}, {lbl:'rms', chips:[chip(p.rms.toFixed(5), 'hot')]}, {lbl:{zh:'理論', en:'law'}, chips:[chip(p.law.toFixed(5), 'ok')]}],
      line:k === pts.length - 1 ? 10 : 6,
      msg:k === pts.length - 1
        ? {zh:'100,000 支飛鏢：誤差約 ' + p.rms.toFixed(4) + '，四個點都貼著虛線。照這條線推算，要 6 位正確小數（誤差 5e-7）得丟約 10.8 兆支飛鏢。重跑次數少的點（這裡只跑 ' + p.runs + ' 次）本身也是估計，會離虛線遠一點；要量「誤差有多大」也受同一條 1/√n 規則管。',
           en:'100,000 darts: error about ' + p.rms.toFixed(4) + ', and all four dots sit on the dashed line. Following the line, six correct decimals (error 5e-7) would need about 10.8 trillion darts. A dot with few reruns (only ' + p.runs + ' here) is itself an estimate and can stray from the line - measuring the error obeys the same 1/√n rule.'}
        : {zh:'n = ' + fmt(p.n) + ' 重跑 ' + p.runs + ' 次，誤差的 root-mean-square 是 ' + p.rms.toFixed(5) + '，理論 ' + p.law.toFixed(5) + '。誤差和維度、形狀都無關，只看 √n——這正是 Monte Carlo 在高維積分（還有強化學習估計期望報酬）裡好用的原因：格點法的格子數會隨維度爆炸，丟飛鏢不會。',
           en:'n = ' + fmt(p.n) + ' over ' + p.runs + ' runs: root-mean-square error ' + p.rms.toFixed(5) + ', law ' + p.law.toFixed(5) + '. The error depends on √n only, not on the dimension or the shape - which is why Monte Carlo wins at high-dimensional integrals (and at estimating expected returns in reinforcement learning): a grid explodes with the dimension, darts do not.'}});
  });
  return F.list;
}

/* ================================================================ tab 2: Fermat vs Miller-Rabin on 561 */
const CODE_PRIME = [
  'def fermat_passes(n, a):',
  '    return pow(a, n - 1, n) == 1',
  '',
  'def miller_rabin_passes(n, a):',
  '    d, s = n - 1, 0',
  '    while d % 2 == 0:',
  '        d //= 2; s += 1',
  '    x = pow(a, d, n)',
  '    if x == 1 or x == n - 1:',
  '        return True',
  '    for _ in range(s - 1):',
  '        x = x * x % n',
  '        if x == n - 1:',
  '            return True',
  '    return False   # a square root of 1 other than +-1'
];
const powMod = (a, e, m) => { let r = 1; a %= m; while (e){ if (e & 1) r = r * a % m; a = a * a % m; e = Math.floor(e / 2); } return r; };
const fermatPass = (n, a) => powMod(a, n - 1, n) === 1;
function mrTrace(n, a){
  let d = n - 1, s = 0; while (d % 2 === 0){ d /= 2; s++; }
  const xs = [powMod(a, d, n)];
  for (let k = 1; k < s; k++) xs.push(xs[k - 1] * xs[k - 1] % n);
  let pass = xs[0] === 1 || xs[0] === n - 1;
  for (let k = 1; k < s && !pass; k++) if (xs[k] === n - 1) pass = true;
  return {d:d, s:s, xs:xs, last:xs[s - 1] * xs[s - 1] % n, pass:pass};
}
const PN = 561, PCOLS = 31;
function primeStage(state, o){
  const sh = [], w = .27, h = .2, x0 = .52, y0 = .45;
  for (let a = 2; a <= PN - 2; a++){
    const k = a - 2, x = x0 + (k % PCOLS) * w, y = y0 + Math.floor(k / PCOLS) * (h + .02);
    sh.push(S.r(x, y, w - .04, h, state(a), '', {rx:.03}));
  }
  (o.lines || []).forEach((l, k) => sh.push(fitT(4.9, 4.72 + k * .42, l[0], {c:l[1] || COL.pale, fs:l[2] || .30})));
  return caps(sh, o);
}
function primeFrames(v){
  const F = new Frames(), mr = v === 1, test = a => mr ? mrTrace(PN, a).pass : fermatPass(PN, a);
  const liars = []; for (let a = 2; a <= PN - 2; a++) if (test(a)) liars.push(a);
  const N = PN - 3, name = mr ? 'Miller-Rabin' : 'Fermat';
  const T2 = mrTrace(PN, 2);
  F.push({shapes:primeStage(() => 'ghost', {lines:[[{zh:'561 = 3 × 11 × 17，是合數', en:'561 = 3 x 11 x 17 - composite'}, COL.orangeL, .34]],
      cap:{zh:'每一格是一個底數 a = 2 … 559（共 ' + N + ' 個）', en:'each cell is one base a = 2 … 559 (' + N + ' in all)'},
      cap2:{zh:'紅色：這個底數讓 ' + name + ' 誤判成質數', en:'red: this base fools ' + name + ' into saying "prime"'}}),
    panels:[{lbl:'n', chips:[chip('561', 'act')]}], line:mr ? 3 : 0,
    msg:mr ? {zh:'Miller-Rabin 也是拿一個底數 a 去算 a 的某個次方，但它不只看最後的結果是不是 1，而是看「怎麼走到 1 的」。質數的模數下，平方等於 1 的數只有 1 和 −1；如果途中出現別的平方根，就證明 n 是合數。',
              en:'Miller-Rabin also raises a base a to a power, but it does not only check that the end result is 1 - it checks how the sequence got there. Modulo a prime, the only numbers whose square is 1 are 1 and -1; any other square root of 1 along the way proves n composite.'}
           : {zh:'Fermat 小定理：如果 n 是質數，任何 a 都滿足 a^(n−1) ≡ 1 (mod n)。反過來檢查——算出來不是 1 就一定是合數。但算出來是 1 不代表是質數。561 是最小的 Carmichael 數：和它互質的底數全部會說「質數」。',
              en:'Fermat\'s little theorem: if n is prime then a^(n-1) ≡ 1 (mod n) for every a. Run it backwards - a result other than 1 proves n composite. But a result of 1 does not prove n prime. 561 is the smallest Carmichael number: every base coprime to it answers "prime".'}});
  /* the base-2 walkthrough */
  F.push({shapes:primeStage(a => a === 2 ? (test(2) ? 'bad' : 'ok') : 'ghost', {lines:mr
        ? [['560 = 2^' + T2.s + ' × ' + T2.d + '   x = 2^' + T2.d + ' mod 561 = ' + T2.xs[0], COL.purpleL],
           [{zh:'平方：' + T2.xs.join(' → ') + '，所以 2^560 = 1', en:'squaring: ' + T2.xs.join(' → ') + ', so 2^560 = 1'}, COL.orangeL]]
        : [['2^560 mod 561 = ' + powMod(2, 560, 561), COL.orangeL, .34]],
      cap:mr ? {zh:'67² ≡ 1，但 67 不是 ±1：找到一個不該存在的平方根', en:'67² ≡ 1 but 67 is not ±1: a square root that should not exist'}
             : {zh:'結果是 1，Fermat 回答「質數」——錯', en:'the result is 1, so Fermat says "prime" - wrong'},
      cap2:mr ? {zh:'質數的模數下 x² ≡ 1 只有 x ≡ ±1，所以 561 一定是合數', en:'modulo a prime x² ≡ 1 only for x ≡ ±1, so 561 must be composite'}
              : {zh:'Fermat 只看終點，看不到中間經過什麼', en:'Fermat only sees the end point, not the way there'}}),
    panels:[{lbl:'a', chips:[chip('2', 'act')]}, {lbl:'x', chips:(mr ? T2.xs : [1]).map(x => chip(String(x), x === 1 ? 'ok' : 'hot'))}],
    line:mr ? 11 : 1,
    msg:mr ? {zh:'560 = 2⁴ × 35。先算 x = 2³⁵ mod 561 = ' + T2.xs[0] + '，再平方 ' + (T2.s - 1) + ' 次：' + T2.xs.join(' → ') + '，最後這個 1 是 2²⁸⁰；再平方一次就是 Fermat 看到的 2⁵⁶⁰ = 1。Miller-Rabin 看見 67 → 1：67 × 67 = 4489 = 8 × 561 + 1，67 是 1 的平方根卻不是 ±1。質數的模數不可能有這種數，所以 561 被證明是合數。',
              en:'560 = 2^4 x 35. First x = 2^35 mod 561 = ' + T2.xs[0] + ', then square ' + (T2.s - 1) + ' times: ' + T2.xs.join(' → ') + ', where the final 1 is 2^280; one more squaring gives the 2^560 = 1 that Fermat sees. Miller-Rabin sees 67 → 1: 67 x 67 = 4489 = 8 x 561 + 1, so 67 is a square root of 1 that is not ±1. No prime modulus has one, so 561 is proven composite.'}
           : {zh:'2⁵⁶⁰ mod 561 = 1，跟質數的反應一模一樣。原因：561 = 3 × 11 × 17，而 2、10、16 都整除 560，所以 a⁵⁶⁰ 對 3、11、17 各自都 ≡ 1，合起來就對 561 ≡ 1。只要 a 不是 3、11、17 的倍數都會這樣。',
              en:'2^560 mod 561 = 1, exactly what a prime would give. The reason: 561 = 3 x 11 x 17, and 2, 10 and 16 all divide 560, so a^560 ≡ 1 modulo 3, 11 and 17 separately, hence modulo 561. Every a that is not a multiple of 3, 11 or 17 does the same.'}});
  const rows = Math.ceil(N / PCOLS), chunks = [6, 12, rows];
  chunks.forEach((r, ci) => {
    const lim = 2 + r * PCOLS;
    const seen = liars.filter(a => a < lim).length, tot = Math.min(lim, PN - 1) - 2;
    const last = ci === chunks.length - 1;
    F.push({shapes:primeStage(a => a >= lim ? 'ghost' : test(a) ? 'bad' : 'done', {lines:[[name + ': ' + seen + ' / ' + tot + (last ? ' (' + (100 * seen / tot).toFixed(1) + '%)' : ''), last ? COL.red : COL.orangeL, .34]],
        cap:{zh:'已檢查 ' + tot + ' 個底數，其中 ' + seen + ' 個回答「質數」', en:tot + ' bases checked, ' + seen + ' answer "prime"'},
        cap2:last ? (mr ? {zh:'說謊的底數：' + liars.join('、'), en:'lying bases: ' + liars.join(', ')} : {zh:'不說謊的只剩 3、11、17 的倍數', en:'only multiples of 3, 11 and 17 tell the truth'})
                  : {zh:'灰色：這個底數抓到 561 是合數', en:'grey: this base catches 561 as composite'}}),
      panels:[{lbl:{zh:'說謊', en:'liars'}, chips:[chip(String(seen), 'hot')]}, {lbl:{zh:'檢查', en:'checked'}, chips:[chip(String(tot), 'dim')]}],
      line:mr ? 14 : 1,
      msg:!last ? {zh:'一列一列試底數。' + name + ' 每次說「合數」都一定對，只有說「質數」可能錯——這是 Monte Carlo 演算法：執行時間固定，答案有一定機率錯。',
                   en:'Trying the bases row by row. Whenever ' + name + ' says "composite" it is certainly right; only "prime" can be wrong - that is a Monte Carlo algorithm: fixed running time, an answer that may be wrong with some probability.'}
        : mr ? {zh:'Miller-Rabin 在 ' + N + ' 個底數裡只有 ' + seen + ' 個說謊（' + (100 * seen / N).toFixed(1) + '%）。定理保證任何奇合數的說謊底數不超過 1/4，所以隨機挑 k 個底數，全部說謊的機率 ≤ 4^−k。反過來，如果固定用 2 到 37 這 12 個質數當底數，3.18 × 10²³ 以下（所有 64 位元整數）都已經被驗證過不會錯，2⁶¹ − 1 這種 19 位數也一樣，完全不用因數分解。',
                en:'Miller-Rabin is fooled by only ' + seen + ' of ' + N + ' bases (' + (100 * seen / N).toFixed(1) + '%). A theorem guarantees at most 1/4 of the bases lie for any odd composite, so k random bases all lie with probability <= 4^-k. And with the 12 fixed prime bases 2 to 37, every n below 3.18 x 10^23 (every 64-bit integer) has been checked to come out right - a 19-digit 2^61 - 1 included, with no factoring at all.'}
             : {zh:'Fermat 在 ' + N + ' 個底數裡有 ' + seen + ' 個說謊（' + (100 * seen / N).toFixed(0) + '%）——所有和 561 互質的底數。對 Carmichael 數，多試幾個隨機底數也救不回來，因為說謊的比例沒有上限。下一個選項換 Miller-Rabin。',
                en:'Fermat is fooled by ' + seen + ' of ' + N + ' bases (' + (100 * seen / N).toFixed(0) + '%) - every base coprime to 561. For a Carmichael number more random bases do not help, because nothing bounds the share of liars. The next option switches to Miller-Rabin.'}});
  });
  return F.list;
}

/* ================================================================ tab 3: shuffling */
const CODE_SHUF = [
  'def shuffle_naive(a, randrange):',
  '    for i in range(n):',
  '        j = randrange(n)              # any slot',
  '        a[i], a[j] = a[j], a[i]',
  '',
  'def shuffle_off_by_one(a, randrange):',
  '    for i in range(len(a) - 1, 0, -1):',
  '        j = randrange(i)              # 0..i-1: i never stays',
  '        a[i], a[j] = a[j], a[i]',
  '',
  'def fisher_yates(a, randrange):',
  '    for i in range(len(a) - 1, 0, -1):',
  '        j = randrange(i + 1)          # 0..i inclusive',
  '        a[i], a[j] = a[j], a[i]'
];
const SH_LINES = [[1, 2, 3], [6, 7, 8], [11, 12, 13]];
const ORDERS = ['abc', 'acb', 'bac', 'bca', 'cab', 'cba'];
/* each shuffle records its swaps when given a log array */
function shuf(v, rr, log){
  const a = ['a', 'b', 'c'], n = 3;
  const sw = (i, j) => { const pre = a.slice(); [a[i], a[j]] = [a[j], a[i]]; if (log) log.push({i:i, j:j, pre:pre, post:a.slice()}); };
  if (v === 0) for (let i = 0; i < n; i++) sw(i, rr(n));
  else for (let i = n - 1; i > 0; i--) sw(i, rr(v === 1 ? i : i + 1));
  return a.join('');
}
const BX = k => 1.0 + k * 1.38, BBASE = 4.75, BSCALE = 4.8;
function shufStage(arr, st, counts, runs, o){
  const sh = [];
  arr.forEach((c, k) => sh.push(S.r(.8 + k * 1.0, .45, .86, .8, st[k] || 'idle', c, {fs:.42, top:k === 0 ? 'a[0]' : 'a[' + k + ']', topfs:.22})));
  (o.side || []).forEach((l, k) => sh.push(fitT(4.25, .7 + k * .48, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
  const fairY = BBASE - BSCALE / 6;
  sh.push(S.e(.75, fairY, 9.35, fairY, {s:'act', w:.03, arrow:false, dash:'.12 .10'}));
  sh.push(S.t(9.4, fairY + .08, '1/6', {c:COL.purpleL, fs:.24, anchor:'start'}));
  ORDERS.forEach((ord, k) => {
    const c = counts[ord] || 0, f = runs ? c / (o.total || runs) : 0, h = Math.max(f * BSCALE, .02);
    const s = !runs ? 'ghost' : o.hiOrd === ord ? 'hot' : c === 0 ? 'ghost' : (o.bad && Math.abs(f - 1 / 6) > 1e-9) ? 'bad' : 'ok';
    sh.push(S.r(BX(k), BBASE - h, .9, h, s, '', {rx:.03}));
    sh.push(S.t(BX(k) + .45, BBASE + .32, ord, {c:o.hiOrd === ord ? COL.orangeL : COL.pale, fs:.28}));
    if (runs) sh.push(S.t(BX(k) + .45, BBASE - h - .12, o.total ? String(c) : (c ? c + '/' + runs : '0'), {c:COL.grey, fs:.22}));
  });
  return caps(sh, o);
}
function shufFrames(v){
  const F = new Frames(), all = everyRun(rr => shuf(v, rr)), runs = all.length;
  const name = [{zh:'天真洗牌', en:'naive shuffle'}, {zh:'randrange(i) 的寫錯', en:'the randrange(i) slip'}, {zh:'Fisher-Yates', en:'Fisher-Yates'}][v];
  const counts = {}, L = SH_LINES[v];
  F.push({shapes:shufStage(['a', 'b', 'c'], {}, {}, 0, {side:[[v === 0 ? {zh:'每一步 j 有 3 種：3 × 3 × 3 = 27 種跑法', en:'3 choices of j per step: 3 x 3 x 3 = 27 runs'}
                                                                  : v === 1 ? {zh:'j 有 2 種、再 1 種：2 × 1 = 2 種跑法', en:'2 choices then 1: 2 x 1 = 2 runs'}
                                                                  : {zh:'j 有 3 種、再 2 種：3 × 2 = 6 種跑法', en:'3 choices then 2: 3 x 2 = 6 runs'}, COL.purpleL],
                                                           [{zh:'每一種跑法機率相同', en:'every run is equally likely'}, COL.grey]],
      cap:{zh:name.zh + '：把每一種亂數選擇都跑一次，數每個順序出現幾次', en:name.en + ': run every possible sequence of random choices once and count each order'},
      cap2:{zh:'公平的洗牌：6 種順序各 1/6', en:'a fair shuffle: each of the 6 orders 1/6'}}),
    panels:[{lbl:{zh:'跑法', en:'runs'}, chips:[chip(String(runs), 'act')]}], line:L[0],
    msg:v === 0 ? {zh:'丟一萬次再看長條圖是不是一樣高，偏差小的時候根本看不出來。今天的測法是精確的：把 randrange 換成一個「依序吐出每一種選擇」的假亂數，所有跑法各跑一次，每種跑法的機率都一樣，於是每個順序的機率就是「次數 / 跑法數」，用分數算，沒有誤差。',
                   en:'Shuffling ten thousand times and eyeballing the histogram cannot see a small bias. Today\'s test is exact: replace randrange with a fake that hands out every possible choice in turn, run each sequence of choices once, and since every run is equally likely, each order\'s probability is count / runs - exact fractions, no noise.'}
      : v === 1 ? {zh:'Fisher-Yates 只改一個字：randrange(i + 1) 寫成 randrange(i)。看起來像差一點點，實際上讓 j 永遠小於 i——位置 i 一定會換成別人。',
                   en:'Fisher-Yates with one character changed: randrange(i + 1) written as randrange(i). It looks like a tiny slip, but now j is always below i - slot i always receives someone else.'}
      : {zh:'Fisher-Yates 從最後一格往前填：第 i 格從還沒定下來的 0…i 這 i + 1 個裡均勻挑一個。3 個元素有 3 × 2 = 6 種跑法，剛好等於 3! = 6 種順序。',
         en:'Fisher-Yates fills from the back: slot i takes a uniform pick among the i + 1 items 0…i not yet placed. Three items give 3 x 2 = 6 runs - exactly the 3! = 6 orders.'}});
  const stepRun = (r, idx) => {
    const log = []; let k = 0; const rr = () => r.choices[k++];
    shuf(v, rr, log);
    log.forEach((s, si) => {
      const st = {}; st[s.i] = 'act'; st[s.j] = s.i === s.j ? 'act' : 'hot';
      F.push({shapes:shufStage(s.post, st, counts, idx, {total:runs, side:[[{zh:'第 ' + (idx + 1) + ' 種跑法，第 ' + (si + 1) + ' 步', en:'run ' + (idx + 1) + ', step ' + (si + 1)}, COL.pale],
              ['i = ' + s.i + ', j = ' + s.j, COL.orangeL],
              [{zh:'選擇：' + pyList(r.choices), en:'choices: ' + pyList(r.choices)}, COL.grey]],
          cap:{zh:'交換 a[' + s.i + '] 和 a[' + s.j + ']：' + s.pre.join('') + ' → ' + s.post.join(''), en:'swap a[' + s.i + '] and a[' + s.j + ']: ' + s.pre.join('') + ' → ' + s.post.join('')},
          cap2:{zh:v === 0 ? 'j 可以是任何位置 0…2' : v === 1 ? 'j 只能是 0…' + (s.i - 1) + '，i 自己被排除' : 'j 可以是 0…' + s.i + '，包括 i 自己', en:v === 0 ? 'j may be any slot 0…2' : v === 1 ? 'j can only be 0…' + (s.i - 1) + ', i itself is excluded' : 'j may be 0…' + s.i + ', i itself included'}}),
        panels:[{lbl:'i', chips:[chip(String(s.i), 'act')]}, {lbl:'j', chips:[chip(String(s.j), 'hot')]}, {lbl:'a', chips:s.post.map(c => chip(c))}],
        line:si === 0 ? L[1] : L[2],
        msg:v === 0 ? {zh:'第 ' + (si + 1) + ' 步：i = ' + s.i + '，j 從 0、1、2 三個裡挑（這一種跑法挑到 ' + s.j + '）。天真洗牌每一步都讓 j 是任何位置，所以 3 步共有 3³ = 27 種同樣可能的跑法。',
                       en:'Step ' + (si + 1) + ': i = ' + s.i + ', and j is any of 0, 1, 2 (this run picks ' + s.j + '). The naive shuffle lets j be any slot at every step, so 3 steps give 3^3 = 27 equally likely runs.'}
          : v === 1 ? {zh:'i = ' + s.i + '：randrange(' + s.i + ') 只會給 0…' + (s.i - 1) + '，所以 ' + s.pre[s.i] + ' 一定被換走。這正好是 Sattolo 演算法——它只產生「每個元素都離開原位、整體是一個大循環」的排列。',
                       en:'i = ' + s.i + ': randrange(' + s.i + ') only returns 0…' + (s.i - 1) + ', so ' + s.pre[s.i] + ' is always moved away. This is exactly Sattolo\'s algorithm - it only produces permutations where every item leaves its place and the whole thing is one big cycle.'}
          : {zh:'i = ' + s.i + '：從 0…' + s.i + ' 這 ' + (s.i + 1) + ' 個還沒定下來的裡挑一個放到第 ' + s.i + ' 格（這一種跑法挑到 ' + s.j + '）。每個元素落在第 ' + s.i + ' 格的機率都是 1/' + (s.i + 1) + '，之後這格就不再動。',
             en:'i = ' + s.i + ': pick one of the ' + (s.i + 1) + ' items in 0…' + s.i + ' still free and put it in slot ' + s.i + ' (this run picks ' + s.j + '). Each item lands in slot ' + s.i + ' with probability 1/' + (s.i + 1) + ', and the slot never changes again.'}});
    });
    counts[r.res] = (counts[r.res] || 0) + 1;
  };
  const showRun = (idx, upto) => {
    const r = all[upto - 1];
    F.push({shapes:shufStage(r.res.split(''), {0:'ok', 1:'ok', 2:'ok'}, counts, upto, {hiOrd:r.res, total:runs,
        side:[[{zh:'已跑 ' + upto + ' / ' + runs + ' 種', en:upto + ' of ' + runs + ' runs done'}, COL.pale],
              [{zh:'最後一種：' + pyList(r.choices) + ' → ' + r.res, en:'latest: ' + pyList(r.choices) + ' → ' + r.res}, COL.orangeL]],
        cap:{zh:'長條 = 目前為止的次數（滿分是 ' + runs + ' 種跑法）', en:'bars = count so far (out of ' + runs + ' runs in total)'},
        cap2:{zh:'橘色：剛剛那一種跑法得到的順序', en:'orange: the order the latest run produced'}}),
      panels:[{lbl:{zh:'已跑', en:'done'}, chips:[chip(upto + '/' + runs, 'act')]}], line:L[2],
      msg:{zh:'跑法 ' + (idx + 1) + (upto > idx + 1 ? ' 到 ' + upto : '') + '：結果 ' + r.res + '。每一種跑法都只跑一次，不靠運氣，所以這些長條高度是精確值，不是估計。',
           en:'Run ' + (idx + 1) + (upto > idx + 1 ? ' to ' + upto : '') + ': result ' + r.res + '. Every run happens exactly once, nothing is left to luck, so the bar heights are exact values, not estimates.'}});
  };
  stepRun(all[0], 0);
  if (v === 0){
    for (let k = 1; k < 9; k++) counts[all[k].res] = (counts[all[k].res] || 0) + 1;
    showRun(1, 9);
    for (let k = 9; k < 18; k++) counts[all[k].res] = (counts[all[k].res] || 0) + 1;
    showRun(9, 18);
    for (let k = 18; k < 27; k++) counts[all[k].res] = (counts[all[k].res] || 0) + 1;
    showRun(18, 27);
  } else if (v === 1){
    showRun(0, 1); stepRun(all[1], 1); showRun(1, 2);
  } else {
    showRun(0, 1);
    for (let k = 1; k < runs; k++){ counts[all[k].res] = (counts[all[k].res] || 0) + 1; showRun(k, k + 1); }
  }
  const line = ORDERS.map(o => o + ':' + (counts[o] || 0)).join('  ');
  F.push({shapes:shufStage(['a', 'b', 'c'], {}, counts, runs, {bad:v !== 2,
      side:[[line, v === 2 ? COL.tealL : COL.red, .28],
            [v === 0 ? {zh:'27 不是 6 的倍數：不可能平分', en:'27 is not a multiple of 6: no even split possible'}
             : v === 1 ? {zh:'只到得了 bca、cab 兩種', en:'reaches only bca and cab'}
             : {zh:'6 種跑法、6 種順序，一對一', en:'6 runs, 6 orders, one each'}, v === 2 ? COL.tealL : COL.orangeL]],
      cap:v === 0 ? {zh:'天真洗牌：acb、bac、bca 各 5/27，其他各 4/27', en:'naive: acb, bac, bca get 5/27 each, the rest 4/27'}
         : v === 1 ? {zh:'randrange(i)：abc、acb、bac、cba 的機率是 0', en:'randrange(i): abc, acb, bac and cba have probability 0'}
         : {zh:'Fisher-Yates：6 種順序各 1/6', en:'Fisher-Yates: each of the 6 orders exactly 1/6'},
      capc:v === 2 ? COL.tealL : COL.red,
      cap2:v === 0 ? {zh:'4 個元素：256 種跑法分給 24 種順序，最多的是最少的 1.88 倍', en:'4 items: 256 runs over 24 orders, the most common 1.88x the rarest'}
          : v === 1 ? {zh:'4 個元素：只到得了 24 種裡的 6 種，而且沒有任何元素留在原位', en:'4 items: reaches 6 of 24 orders, none with an item left in place'}
          : {zh:'LeetCode 384 的 shuffle() 就是這個迴圈', en:'LeetCode 384\'s shuffle() is exactly this loop'}}),
    panels:[{lbl:{zh:'結果', en:'result'}, chips:ORDERS.map(o => chip(o + ' ' + frac(counts[o] || 0, runs), (counts[o] || 0) * 6 === runs ? 'ok' : 'hot'))}],
    line:L[1],
    msg:v === 0 ? {zh:'27 種跑法要分給 6 種順序，27 ÷ 6 不是整數，所以不管程式怎麼寫，「每一步任選一格」都不可能公平：有 3 種順序拿到 5 次、3 種拿到 4 次。這個 5/27 對 4/27 的差距，丟 60,000 次洗牌時大約是 11,000 對 8,900，混在雜訊裡不容易看出是錯的；精確列舉則一眼就看得到。',
                   en:'27 runs over 6 orders, and 27 / 6 is not a whole number, so "swap with any slot" cannot be fair no matter what: three orders get 5 runs and three get 4. Over 60,000 real shuffles that 5/27 vs 4/27 shows up as roughly 11,000 vs 8,900 - easy to mistake for noise; exact enumeration makes it obvious.'}
      : v === 1 ? {zh:'兩種跑法都產生「每個字母都離開原位」的順序。abc 永遠不會出現——洗完牌和原本一樣，本來就該有 1/6 的機會。用亂數抽樣測，只要不特別去檢查 abc 有沒有出現，很容易漏掉。',
                   en:'Both runs produce an order where every letter has moved. abc never appears - and an honest shuffle leaves the deck unchanged 1/6 of the time. A sampling test misses this unless it specifically checks whether abc ever shows up.'}
      : {zh:'6 種跑法剛好一對一對應 6 種順序，所以每種都是 1/6。一般地，n 個元素的跑法數是 n × (n−1) × … × 2 = n!，正好是排列數，每種跑法產生不同的排列。O(n) 時間、原地、不需要額外記憶體。',
         en:'The 6 runs map one-to-one onto the 6 orders, so each is 1/6. In general n items give n x (n-1) x … x 2 = n! runs - exactly the number of permutations, each run producing a different one. O(n) time, in place, no extra memory.'}});
  return F.list;
}

/* ================================================================ tab 4: reservoir sampling */
const CODE_RES = [
  'def reservoir_one(stream, randrange):',
  '    chosen = None',
  '    for i, x in enumerate(stream, 1):    # i = 1, 2, 3 ...',
  '        if randrange(i) == 0:            # 1 chance in i',
  '            chosen = x',
  '    return chosen',
  '',
  'def reservoir_off_by_one(stream, randrange):',
  '    for i, x in enumerate(stream):       # i = 0, 1, 2 ...',
  '        if i == 0:',
  '            chosen = x; continue',
  '        if randrange(i) == 0:            # should be i + 1',
  '            chosen = x',
  '    return chosen'
];
const STREAM = ['a', 'b', 'c', 'd', 'e'];
function resRun(v, items, rr){
  let chosen = null;
  if (v === 0){ items.forEach((x, k) => { if (rr(k + 1) === 0) chosen = x; }); }
  else items.forEach((x, k) => { if (k === 0){ chosen = x; return; } if (rr(k) === 0) chosen = x; });
  return chosen;
}
/* exact chance that each item is held after the first t items */
function holdDist(v, t){
  const all = everyRun(rr => resRun(v, STREAM.slice(0, t), rr)), c = {};
  all.forEach(r => c[r.res] = (c[r.res] || 0) + 1);
  return {c:c, runs:all.length};
}
function resStage(t, st, hold, o){
  const sh = [];
  STREAM.forEach((x, k) => sh.push(S.r(.8 + k * 1.0, .45, .86, .8, st[k] || (k < t ? 'done' : 'ghost'), x, {fs:.42, top:'i = ' + (o.base0 ? k : k + 1), topfs:.22})));
  sh.push(S.t(6.1, .7, {zh:'手上', en:'held'}, {c:COL.grey, fs:.26}));
  sh.push(S.r(5.75, .85, .7, .7, o.held ? 'ok' : 'ghost', o.held || '', {fs:.4}));
  (o.side || []).forEach((l, k) => sh.push(k === 0 ? fitT(6.75, 1.3, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})
                                                   : fitT(.8, 1.25 + k * .4, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .28})));
  if (hold){
    STREAM.forEach((x, k) => {
      const c = hold.c[x] || 0, f = c / hold.runs, h = Math.max(f * 2.2, .02);
      sh.push(S.r(BX(k), BBASE - h, .9, h, k >= t ? 'ghost' : c === 0 ? 'bad' : o.hiK === k ? 'hot' : 'ok', '', {rx:.03}));
      sh.push(S.t(BX(k) + .45, BBASE + .32, x, {c:COL.pale, fs:.28}));
      if (k < t) sh.push(S.t(BX(k) + .45, BBASE - h - .12, frac(c, hold.runs), {c:COL.grey, fs:.24}));
    });
  }
  return caps(sh, o);
}
function resFrames(v){
  const F = new Frames(), rng = rngOf(v ? 11 : 5), rr = rrOf(rng);
  F.push({shapes:resStage(0, {}, holdDist(v, 0), {base0:v === 1,
      side:[['n = ?', COL.purpleL], [{zh:'資料一個一個流過，不知道總共有幾個', en:'items stream past one at a time; the total is unknown'}, COL.pale], [{zh:'手上只能記一個', en:'there is room to keep just one'}, COL.grey]],
      cap:{zh:'要從長度未知的資料流裡，公平地留下一個', en:'keep one item, fairly, from a stream of unknown length'},
      cap2:{zh:'LeetCode 382 的 linked list、398 的重複數字都是這個問題', en:'LeetCode 382\'s linked list and 398\'s repeated numbers are both this problem'}}),
    panels:[{lbl:{zh:'手上', en:'held'}, chips:[chip('—', 'empty')]}], line:v ? 8 : 2,
    msg:v ? {zh:'同一個迴圈，但 i 從 0 開始數。第 0 個直接拿著（不然 randrange(0) 會出錯），之後用 randrange(i) 決定要不要換——看起來和正確版一樣是「1/i」，但 i 少了 1。',
             en:'The same loop with i counted from 0. Item 0 is simply taken (randrange(0) would raise), and from then on randrange(i) decides whether to switch - it still reads as "1/i", but i is one short.'}
          : {zh:'如果知道長度 n，抽一個 randrange(n) 就好；但資料流（log、linked list、網路封包）往往不知道有多長，也存不下。蓄水池抽樣只走一遍、只記一個：第 i 個來的時候，以 1/i 的機率把手上的換成它。',
             en:'With the length n known you would just draw randrange(n); but a stream (a log, a linked list, network packets) often has an unknown length and does not fit in memory. Reservoir sampling makes one pass and keeps one item: when item i arrives, swap it in with probability 1/i.'}});
  let held = null;
  STREAM.forEach((x, k) => {
    const i = v ? k : k + 1;
    let roll = null, take;
    if (v && k === 0) take = true;
    else { roll = rr(i); take = roll === 0; }
    if (take) held = x;
    const hold = holdDist(v, k + 1), st = {}; st[k] = take ? 'hot' : 'act';
    const chance = v && k === 0 ? {zh:'直接拿', en:'taken outright'} : frac(1, i);
    F.push({shapes:resStage(k + 1, st, hold, {base0:v === 1, held:held, hiK:k,
        side:[['i = ' + i + ': ' + x, COL.purpleL, .32],
              [{zh:'換的機率 ' + (typeof chance === 'string' ? chance : chance.zh), en:'switch chance ' + (typeof chance === 'string' ? chance : chance.en)}, COL.orangeL],
              [roll == null ? {zh:'（沒有擲）', en:'(no roll)'} : {zh:'這次 randrange(' + i + ') = ' + roll + (take ? '：換' : '：不換'), en:'this time randrange(' + i + ') = ' + roll + (take ? ': switch' : ': keep')}, take ? COL.tealL : COL.grey]],
        cap:v === 1 && k === 1 ? {zh:'randrange(1) 永遠是 0：b 一定把 a 換掉', en:'randrange(1) is always 0: b always replaces a'}
           : {zh:'看過 ' + (k + 1) + ' 個以後，每個被留下的機率：' + STREAM.slice(0, k + 1).map(y => y + ' ' + frac(hold.c[y] || 0, hold.runs)).join('、'),
              en:'after ' + (k + 1) + ' items, the chance each is held: ' + STREAM.slice(0, k + 1).map(y => y + ' ' + frac(hold.c[y] || 0, hold.runs)).join(', ')},
        capc:v === 1 && k >= 1 ? COL.red : COL.pale,
        cap2:{zh:'長條是把所有亂數選擇都跑一遍的精確值，和這一次擲出什麼無關', en:'bars are exact, from every possible sequence of rolls - not from this one run'}}),
      panels:[{lbl:'i', chips:[chip(String(i), 'act')]}, {lbl:{zh:'手上', en:'held'}, chips:[chip(held, 'ok')]}],
      line:v ? (k === 0 ? 10 : take ? 12 : 11) : (take ? 4 : 3),
      msg:v === 0
        ? (k === 0 ? {zh:'第 1 個：randrange(1) 只會是 0，所以一定拿著 a。目前 a 被留下的機率 1。',
                      en:'Item 1: randrange(1) can only be 0, so a is always taken. a is held with probability 1.'}
           : {zh:'第 ' + i + ' 個 ' + x + '：以 1/' + i + ' 換進來。為什麼前面每個也都剛好是 1/' + i + '？前 ' + (i - 1) + ' 個裡每個原本是 1/' + (i - 1) + '，這一步沒被換掉的機率是 ' + (i - 1) + '/' + i + '，相乘 1/' + (i - 1) + ' × ' + (i - 1) + '/' + i + ' = 1/' + i + '。整個流結束時，a 的機率是 1 × 1/2 × 2/3 × 3/4 × 4/5 = 1/5，中間全部約掉。',
              en:'Item ' + i + ', ' + x + ': switched in with chance 1/' + i + '. Why does every earlier item also sit at exactly 1/' + i + '? Each of the first ' + (i - 1) + ' had 1/' + (i - 1) + ', and survives this step with ' + (i - 1) + '/' + i + ': 1/' + (i - 1) + ' x ' + (i - 1) + '/' + i + ' = 1/' + i + '. At the end a holds with 1 x 1/2 x 2/3 x 3/4 x 4/5 = 1/5 - everything in the middle cancels.'})
        : (k === 0 ? {zh:'i = 0 的 a 直接拿著。', en:'a at i = 0 is simply taken.'}
           : k === 1 ? {zh:'i = 1 的 b：randrange(1) 只有 0 這個值，換的機率是 1——a 一定被丟掉。從這裡開始 a 的機率是 0，後面不管怎麼抽都回不來。LeetCode 382 跑 30,000 次，值 1 一次都沒出現。',
                        en:'b at i = 1: randrange(1) has only the value 0, so the switch chance is 1 - a is always dropped. From here on a\'s chance is 0 and nothing later can bring it back. Running LeetCode 382 30,000 times, the value 1 never appears.'}
           : {zh:'i = ' + i + ' 的 ' + x + '：換的機率 1/' + i + '，但它其實是第 ' + (i + 1) + ' 個，應該是 1/' + (i + 1) + '。結果 b 之後每個都是 1/' + i + '：整體變成 a 0、其餘各 1/4。',
              en:x + ' at i = ' + i + ': switched in with 1/' + i + ', but it is really item ' + (i + 1) + ' and should get 1/' + (i + 1) + '. So from b on everyone gets 1/' + i + ': overall a 0, the others 1/4 each.'})});
  });
  const fin = holdDist(v, 5);
  F.push({shapes:resStage(5, {}, fin, {base0:v === 1, held:held,
      side:[['n = 5', COL.purpleL], [STREAM.map(y => y + ':' + frac(fin.c[y] || 0, fin.runs)).join('  '), v ? COL.red : COL.tealL, .28],
            [v ? {zh:'a 永遠不會被抽到', en:'a is never returned'} : {zh:'5 個各 1/5', en:'1/5 each of 5'}, v ? COL.red : COL.tealL]],
      cap:v ? {zh:'寫錯版：a 0、b c d e 各 1/4', en:'the slip: a 0, b c d e 1/4 each'} : {zh:'正確版：每個都是 1/5，而且從頭到尾不需要知道 n = 5', en:'correct: each exactly 1/5, without ever knowing n = 5'},
      capc:v ? COL.red : COL.tealL,
      cap2:v ? {zh:'修法：enumerate(stream, 1)，或 randrange(i + 1)', en:'fix: enumerate(stream, 1), or randrange(i + 1)'}
             : {zh:'留 k 個（Algorithm R）：第 i 個以 k/i 的機率換掉隨機一個；k = 2 從 5 個裡抽，10 種組合各 1/10', en:'keep k (Algorithm R): item i enters with chance k/i; k = 2 from 5 gives 10 pairs at 1/10 each'}}),
    panels:[{lbl:{zh:'結果', en:'result'}, chips:STREAM.map(y => chip(y + ' ' + frac(fin.c[y] || 0, fin.runs), (fin.c[y] || 0) * 5 === fin.runs ? 'ok' : 'hot'))}],
    line:v ? 11 : 3,
    msg:v ? {zh:'用抽樣測（丟 30,000 次看直方圖），b c d e 平均得很漂亮，很可能就放過了；只有剛好去看 a 有沒有出現才抓得到。精確列舉直接寫出 a: 0，這種錯就藏不住。',
             en:'A sampling test (30,000 draws, look at the histogram) shows b, c, d, e beautifully even and may well pass; only a check that a ever appears would catch it. Exact enumeration prints a: 0 and the bug has nowhere to hide.'}
          : {zh:'不論流多長，每個元素最後被留下的機率都是 1/n，O(1) 記憶體、只走一遍。LeetCode 398 一樣：只對等於 target 的位置做蓄水池，第 seen 個符合的以 1/seen 換進來。',
             en:'However long the stream, each item ends up held with chance 1/n, in O(1) memory and one pass. LeetCode 398 is the same: run the reservoir over only the positions equal to target, switching in the seen-th match with chance 1/seen.'}});
  return F.list;
}

/* ================================================================ tab 5: rejection sampling, LC 470 */
const CODE_470 = [
  'class Solution:',
  '    def rand10(self):',
  '        while True:',
  '            idx = (rand7() - 1) * 7 + rand7()   # 1..49',
  '            if idx <= 40:                       # keep 40 cells',
  '                return (idx - 1) % 10 + 1',
  '            # 41..49: throw away, draw again',
  '',
  'def rand10_mod_bias(rand7):     # no rejection',
  '    idx = (rand7() - 1) * 7 + rand7()',
  '    return idx % 10 + 1          # 49 cells, 10 digits'
];
const GX0 = 1.0, GY0 = .75, GW = .58, GH = .54;
function gridStage(lab, st, o){
  const sh = [];
  for (let r = 1; r <= 7; r++){
    sh.push(S.t(GX0 - .2, GY0 + (r - 1) * GH + .33, String(r), {c:COL.purpleL, fs:.24, anchor:'end'}));
    sh.push(S.t(GX0 + (r - 1) * GW + (GW - .06) / 2, GY0 - .12, String(r), {c:COL.purpleL, fs:.24}));
  }
  sh.push(S.t(GX0 - .55, GY0 - .12, '1st\\2nd', {c:COL.grey, fs:.2}));
  for (let idx = 1; idx <= 49; idx++){
    const r = Math.floor((idx - 1) / 7), c = (idx - 1) % 7;
    sh.push(S.r(GX0 + c * GW, GY0 + r * GH, GW - .06, GH - .06, st(idx), lab(idx), {fs:.24}));
  }
  (o.side || []).forEach((l, k) => sh.push(fitT(5.45, .9 + k * .5, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
  return caps(sh, o);
}
const keepLab = idx => idx <= 40 ? String((idx - 1) % 10 + 1) : '×';
const modLab = idx => String(idx % 10 + 1);
function r10Frames(v){ return v === 1 ? modFrames() : rejFrames(); }
function rejFrames(){
  const F = new Frames(), rng = rngOf(470), rand7 = () => Math.floor(rng() * 7) + 1;
  const baseSt = idx => idx <= 40 ? 'idle' : 'ghost';
  F.push({shapes:gridStage(keepLab, baseSt, {side:[[{zh:'兩次 rand7 → 49 格，機率都是 1/49', en:'two rand7 calls → 49 cells, each 1/49'}, COL.purpleL],
                                                   [{zh:'前 40 格：1 到 10 各 4 格', en:'first 40 cells: 1 to 10, 4 cells each'}, COL.tealL],
                                                   [{zh:'最後 9 格（×）：丟掉重抽', en:'last 9 cells (x): throw away, redraw'}, COL.red]],
      cap:{zh:'LeetCode 470：只有 rand7()，要做出均勻的 rand10()', en:'LeetCode 470: build a uniform rand10() from nothing but rand7()'},
      cap2:{zh:'列 = 第一次 rand7，欄 = 第二次 rand7', en:'row = first rand7, column = second rand7'}}),
    panels:[], line:3,
    msg:{zh:'一次 rand7 只有 7 種結果，7 不是 10 的倍數，怎麼對應都不公平。兩次 rand7 當成二位數的 7 進位：(第一次 − 1) × 7 + 第二次，得到 1 到 49 每個機率都是 1/49。49 還是不能被 10 整除，所以只留前 40 格（1 到 10 各 4 格），落在後 9 格就整組丟掉重來——這叫拒絕取樣（rejection sampling）。',
         en:'One rand7 has 7 outcomes, and 7 is not a multiple of 10, so no mapping can be fair. Two rand7 calls form a two-digit base-7 number: (first - 1) x 7 + second gives 1 to 49, each with chance 1/49. 49 is still not divisible by 10, so keep only the first 40 cells (four of each digit) and throw away anything in the last 9 and draw again - rejection sampling.'}});
  let got = [], calls = 0, draws = 0;
  while (got.length < 5 || draws < 7){
    const a = rand7(), b = rand7(), idx = (a - 1) * 7 + b; calls += 2; draws++;
    const ok = idx <= 40, d = ok ? (idx - 1) % 10 + 1 : null;
    if (ok) got.push(d);
    F.push({shapes:gridStage(keepLab, i => i === idx ? (ok ? 'hot' : 'bad') : baseSt(i), {
        side:[['rand7() = ' + a + ', rand7() = ' + b, COL.purpleL],
              ['idx = (' + a + ' - 1) × 7 + ' + b + ' = ' + idx, COL.orangeL],
              [ok ? {zh:'≤ 40：回傳 (' + idx + ' - 1) % 10 + 1 = ' + d, en:'<= 40: return (' + idx + ' - 1) % 10 + 1 = ' + d} : {zh:'> 40：丟掉，重抽', en:'> 40: throw away, draw again'}, ok ? COL.tealL : COL.red],
              [{zh:'到目前為止：' + (got.length ? got.join(' ') : '—'), en:'so far: ' + (got.length ? got.join(' ') : '—')}, COL.pale],
              [{zh:'rand7 呼叫 ' + calls + ' 次', en:calls + ' rand7 calls'}, COL.grey]],
        cap:ok ? {zh:'第 ' + draws + ' 組落在第 ' + idx + ' 格：得到 ' + d, en:'draw ' + draws + ' lands on cell ' + idx + ': gives ' + d}
               : {zh:'第 ' + draws + ' 組落在第 ' + idx + ' 格：拒絕', en:'draw ' + draws + ' lands on cell ' + idx + ': rejected'},
        capc:ok ? COL.pale : COL.red,
        cap2:{zh:'被接受的那一組，在 40 格裡仍然是均勻的', en:'an accepted draw is still uniform over the 40 kept cells'}}),
      panels:[{lbl:'idx', chips:[chip(String(idx), ok ? 'hot' : 'bad')]}, {lbl:'rand10', chips:got.length ? got.map(x => chip(String(x), 'ok')) : [chip('—', 'empty')]}],
      line:ok ? 5 : 6,
      msg:ok ? {zh:'idx = ' + idx + ' ≤ 40，接受，回傳 ' + d + '。為什麼丟掉一部分還是公平？因為「接受」這件事只看 idx 是不是 ≤ 40；在被接受的條件下，40 格每格機率一樣，而 1 到 10 各佔 4 格。',
                en:'idx = ' + idx + ' <= 40: accepted, return ' + d + '. Why is throwing some draws away still fair? Acceptance only depends on whether idx <= 40; given acceptance the 40 cells are equally likely, and each digit owns 4 of them.'}
             : {zh:'idx = ' + idx + ' > 40，落在被丟掉的 9 格。不能把它硬對到某個數字（那就變成不公平），只能整組重來。被拒絕的機率是 9/49，所以平均要 49/40 組、也就是 2 × 49/40 = 2.45 次 rand7 才拿到一個 rand10。',
                en:'idx = ' + idx + ' > 40, one of the 9 discarded cells. Forcing it onto some digit would break fairness, so the whole draw is redone. Rejection happens 9/49 of the time, so on average it takes 49/40 draws - 2 x 49/40 = 2.45 rand7 calls - per rand10.'}});
  }
  /* measure, do not hard-code */
  const M = 100000, cnt = new Array(11).fill(0); let tot = 0;
  for (let k = 0; k < M; k++){
    for (;;){ const idx = (rand7() - 1) * 7 + rand7(); tot += 2; if (idx <= 40){ cnt[(idx - 1) % 10 + 1]++; break; } }
  }
  const cs = cnt.slice(1);
  F.push({shapes:gridStage(keepLab, i => i <= 40 ? 'ok' : 'bad', {
      side:[[{zh:'接受率 40/49', en:'acceptance 40/49'}, COL.tealL],
            [{zh:'期望 rand7 次數 2 × 49/40 = 2.45', en:'expected rand7 calls 2 x 49/40 = 2.45'}, COL.purpleL],
            [{zh:'這裡實測 ' + fmt(M) + ' 次：' + (tot / M).toFixed(3), en:'measured here over ' + fmt(M) + ': ' + (tot / M).toFixed(3)}, COL.orangeL],
            [{zh:'每個數字 ' + fmt(Math.min.apply(null, cs)) + ' 到 ' + fmt(Math.max.apply(null, cs)) + ' 次', en:'each digit ' + fmt(Math.min.apply(null, cs)) + ' to ' + fmt(Math.max.apply(null, cs)) + ' times'}, COL.pale],
            [{zh:'（期望各 10,000）', en:'(expect 10,000 each)'}, COL.grey]],
      cap:{zh:'40 格各 1/40，每個數字 4 格：剛好 1/10', en:'40 cells at 1/40, 4 per digit: exactly 1/10'}, capc:COL.tealL,
      cap2:{zh:'把被拒絕的「剩下的亂數」接著用，平均可降到約 2.19 次', en:'reusing the leftover randomness of rejected draws brings it down to about 2.19 calls'}}),
    panels:[{lbl:{zh:'平均呼叫', en:'mean calls'}, chips:[chip((tot / M).toFixed(3), 'hot')]}, {lbl:{zh:'次數', en:'counts'}, chips:cs.map((c, k) => chip((k + 1) + ':' + fmt(c), 'ok'))}],
    line:4,
    msg:{zh:'拒絕取樣是「從一個簡單的分布做出另一個分布」的通用招：想要的範圍裝不滿，就抽一個比較大、但每格機率相同的範圍，把多出來的丟掉。代價是期望次數 = 1 / 接受率。用 rand7 擲兩次，接受率 40/49，所以平均 2.45 次 rand7 換一個 rand10。',
         en:'Rejection sampling is the general way to turn one distribution into another: when the range you want does not fill the one you have, draw from a larger range whose cells are equally likely and throw away the excess. The cost is expected attempts = 1 / acceptance rate. With two rand7 calls the acceptance is 40/49, so on average 2.45 rand7 calls buy one rand10.'}});
  return F.list;
}
function modFrames(){
  const F = new Frames(), cnt = {};
  for (let idx = 1; idx <= 49; idx++){ const d = idx % 10 + 1; cnt[d] = (cnt[d] || 0) + 1; }
  F.push({shapes:gridStage(modLab, () => 'idle', {side:[[{zh:'不拒絕：49 格全用 % 10', en:'no rejection: all 49 cells % 10'}, COL.orangeL],
                                                    [{zh:'49 不是 10 的倍數', en:'49 is not a multiple of 10'}, COL.red]],
      cap:{zh:'少寫拒絕的那一步，看起來省事', en:'skipping the rejection step looks like a shortcut'},
      cap2:{zh:'每格還是 1/49，但每個數字分到的格數不一樣', en:'each cell is still 1/49, but the digits do not get the same number of cells'}}),
    panels:[], line:10,
    msg:{zh:'常見的偷懶寫法：兩次 rand7 得到 1 到 49，直接 % 10。每一格的機率確實相同，但 49 格分給 10 個數字分不平——有一個數字只拿到 4 格，其他拿 5 格。',
         en:'The common shortcut: two rand7 calls give 1 to 49, then % 10. Every cell is still equally likely, but 49 cells cannot be split evenly over 10 digits - one digit gets only 4 cells and the others 5.'}});
  [1, 2].forEach(dg => {
    F.push({shapes:gridStage(modLab, i => i % 10 + 1 === dg ? (dg === 1 ? 'bad' : 'hot') : 'done', {
        side:[[{zh:'數字 ' + dg + '：' + cnt[dg] + ' 格', en:'digit ' + dg + ': ' + cnt[dg] + ' cells'}, dg === 1 ? COL.red : COL.orangeL, .34],
              [{zh:'機率 ' + cnt[dg] + '/49', en:'probability ' + cnt[dg] + '/49'}, COL.pale]],
        cap:{zh:'idx % 10 + 1 = ' + dg + ' 的格子：' + Array.from({length:49}, (_, k) => k + 1).filter(i => i % 10 + 1 === dg).join('、'),
             en:'cells with idx % 10 + 1 = ' + dg + ': ' + Array.from({length:49}, (_, k) => k + 1).filter(i => i % 10 + 1 === dg).join(', ')},
        capc:dg === 1 ? COL.red : COL.pale,
        cap2:{zh:'公平應該是 4.9 格——不可能', en:'fair would be 4.9 cells - impossible'}}),
      panels:[{lbl:{zh:'數字', en:'digit'}, chips:[chip(String(dg), dg === 1 ? 'bad' : 'hot')]}, {lbl:{zh:'格數', en:'cells'}, chips:[chip(String(cnt[dg]), 'act')]}],
      line:10,
      msg:dg === 1 ? {zh:'數字 1 只來自 idx = 10、20、30、40 這 4 格，機率 4/49 ≈ 8.2%，比應有的 10% 少了快兩成。',
                      en:'Digit 1 only comes from idx = 10, 20, 30, 40 - 4 cells, probability 4/49 ≈ 8.2%, almost a fifth short of the 10% it should have.'}
                   : {zh:'數字 2 來自 1、11、21、31、41 這 5 格，機率 5/49 ≈ 10.2%。其餘 3 到 10 也一樣是 5 格。差距不大，丟幾千次可能看不出來；但數格子立刻看出不公平。',
                      en:'Digit 2 comes from 1, 11, 21, 31, 41 - 5 cells, 5/49 ≈ 10.2%. Digits 3 to 10 also get 5 each. The gap is small enough to hide in a few thousand trials; counting cells shows it at once.'}});
  });
  F.push({shapes:gridStage(modLab, i => i % 10 + 1 === 1 ? 'bad' : 'idle', {
      side:[[Object.keys(cnt).map(d => d + ':' + cnt[d]).join(' '), COL.orangeL, .28],
            [{zh:'總共 49 = 4 + 9 × 5', en:'total 49 = 4 + 9 x 5'}, COL.pale]],
      cap:{zh:'修法只有一個：裝不滿的部分要丟掉重抽', en:'the only fix: throw away the cells that do not fill evenly and redraw'}, capc:COL.tealL,
      cap2:{zh:'同樣的道理：randrange(n) 內部也是用拒絕取樣避免 % 偏差', en:'same reason randrange(n) uses rejection internally rather than a biased %'}}),
    panels:[{lbl:{zh:'格數', en:'cells'}, chips:Object.keys(cnt).map(d => chip(d + ':' + cnt[d] + '/49', cnt[d] === 4 ? 'bad' : 'hot'))}],
    line:10,
    msg:{zh:'任何時候「一個大小 m 的均勻亂數取 % n」而 m 不是 n 的倍數，就有這個偏差。Python 的 random.randrange 不會這樣，因為它先抽足夠的 bit，超出範圍就重抽——正是左邊那個拒絕取樣。',
         en:'Whenever a uniform number with m values is taken % n and m is not a multiple of n, this bias appears. Python\'s random.randrange avoids it by drawing enough bits and redrawing when the value is out of range - exactly the rejection sampler of the first option.'}});
  return F.list;
}

/* ================================================================ tab 6: weighted pick, LC 528 */
const CODE_528 = [
  'class Solution:',
  '    def __init__(self, w):',
  '        self.prefix = list(accumulate(w))      # [1, 3] -> [1, 4]',
  '',
  '    def pickIndex(self):',
  '        target = randrange(self.prefix[-1])    # 0..total-1',
  '        return bisect_right(self.prefix, target)',
  '',
  'def pick_index_off_by_one(w):',
  '    prefix = list(accumulate(w))',
  '    target = randint(0, prefix[-1])            # 0..total: one too many',
  '    return bisect_left(prefix, target)'
];
const CODE_SOFT = [
  'def sample_token(logits, temperature, rng):',
  '    m = max(logits)',
  '    e = [exp((z - m) / temperature) for z in logits]',
  '    p = [x / sum(e) for x in e]                # softmax',
  '    prefix = list(accumulate(p))',
  '    target = rng.random() * prefix[-1]',
  '    return bisect_right(prefix, target)        # same as LC 528'
];
const bisectRight = (a, x) => { let lo = 0, hi = a.length; while (lo < hi){ const m = (lo + hi) >> 1; if (x < a[m]) hi = m; else lo = m + 1; } return lo; };
const bisectLeft = (a, x) => { let lo = 0, hi = a.length; while (lo < hi){ const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; };
const NL0 = .9, NLW = 8.0;
function wStage(w, nT, owner, hiT, counts, o){
  const sh = [], prefix = []; let s = 0; w.forEach(x => prefix.push(s += x));
  const total = prefix[prefix.length - 1], cw = NLW / nT;
  /* the slices each index owns */
  let x0 = NL0;
  w.forEach((wi, k) => {
    const n = owner.filter(i => i === k).length;
    const ww = n * cw;
    sh.push(S.r(x0, .55, ww - .06, .62, k === o.hiIx ? 'hot' : 'act', {zh:'index ' + k + '（' + n + ' 格）', en:'index ' + k + ' (' + n + ' cell' + (n === 1 ? '' : 's') + ')'}, {fs:fitFs({zh:'index ' + k + '（' + n + ' 格）', en:'index ' + k + ' (' + n + ' cells)'}, ww - .06, .28)}));
    x0 += ww;
  });
  for (let t = 0; t < nT; t++)
    sh.push(S.r(NL0 + t * cw, 1.45, cw - .06, .6, t === hiT ? 'hot' : owner[t] === 0 ? 'soft' : 'idle', String(t), {fs:.3}));
  sh.push(S.t(NL0 - .06, 1.82, {zh:'target', en:'target'}, {c:COL.grey, fs:.2, anchor:'end'}));
  (o.lines || []).forEach((l, k) => sh.push(fitT(NL0, 2.95 + k * .46, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
  if (counts){
    w.forEach((_, k) => {
      const c = counts[k] || 0, f = counts.n ? c / counts.n : 0, y = 4.25 + k * .5;
      sh.push(S.t(NL0 + 1.2, y + .3, 'index ' + k, {c:COL.pale, fs:.26, anchor:'end'}));
      sh.push(S.r(NL0 + 1.35, y, Math.max(f * 6.2, .02), .38, o.want && Math.abs(f - o.want[k]) > 1e-9 ? 'bad' : 'ok', '', {rx:.03}));
      sh.push(S.t(NL0 + 1.45 + f * 6.2, y + .28, counts.n ? frac(c, counts.n) : '', {c:COL.grey, fs:.24, anchor:'start'}));
    });
  }
  return caps(sh, o);
}
function wFrames(v){
  if (v === 2) return softFrames();
  const w = [1, 3], prefix = [1, 4], total = 4, bug = v === 1, nT = bug ? total + 1 : total;
  const pick = t => bug ? bisectLeft(prefix, t) : bisectRight(prefix, t);
  const owner = Array.from({length:nT}, (_, t) => pick(t));
  const want = [1 / 4, 3 / 4];
  const F = new Frames(), counts = {n:0};
  F.push({shapes:wStage(w, nT, owner, -1, null, {lines:[['w = [1, 3]   prefix = [1, 4]', COL.purpleL],
        [bug ? {zh:'target = randint(0, 4)：0…4 共 5 個值，用 bisect_left', en:'target = randint(0, 4): 5 values 0…4, with bisect_left'}
             : {zh:'target = randrange(4)：0…3 共 4 個值，用 bisect_right', en:'target = randrange(4): 4 values 0…3, with bisect_right'}, COL.orangeL]],
      cap:{zh:'LeetCode 528：index i 被抽到的機率要等於 w[i] / sum(w)', en:'LeetCode 528: index i must come out with probability w[i] / sum(w)'},
      cap2:{zh:'上排：每個 index 擁有幾個 target；下排：所有可能的 target', en:'top: how many targets each index owns; bottom: every possible target'}}),
    panels:[{lbl:'prefix', chips:prefix.map(p => chip(String(p), 'act'))}], line:bug ? 10 : 2,
    msg:bug ? {zh:'常見的寫法：random.randint(0, total) 再 bisect_left。randint 是兩端都包含的，所以它有 total + 1 = 5 個值——比總重量多一個。多出來的那一個會落到誰身上？',
               en:'A common version: random.randint(0, total) and then bisect_left. randint includes both ends, so it has total + 1 = 5 values - one more than the total weight. Whose slice does the extra one land in?'}
            : {zh:'把權重排成一條長度 sum(w) = 4 的線，index 0 佔 1 格、index 1 佔 3 格。prefix 是每一段的右端：[1, 4]。均勻抽一個 0…3 的 target，看它落在哪一段——落在哪一段的機率就正比於那一段的長度。找段落就是在 prefix 上二分搜尋，O(log n)。',
               en:'Lay the weights out on a line of length sum(w) = 4: index 0 owns 1 cell, index 1 owns 3. prefix holds the right end of each piece: [1, 4]. Draw a uniform target in 0…3 and see which piece it lands in - the chance of each piece is proportional to its length. Finding the piece is a binary search on prefix, O(log n).'}});
  for (let t = 0; t < nT; t++){
    const ix = pick(t); counts[ix] = (counts[ix] || 0) + 1; counts.n++;
    F.push({shapes:wStage(w, nT, owner, t, counts, {hiIx:ix, want:bug ? want : null, lines:[
          [(bug ? 'bisect_left' : 'bisect_right') + '([1, 4], ' + t + ') = ' + ix, COL.orangeL],
          [bug ? (t === 0 ? {zh:'target 0 是多出來的那個值，也落在 index 0', en:'target 0 is the extra value, and it also lands on index 0'}
                          : t === 1 ? {zh:'bisect_left 遇到相等（1 == prefix[0]）停在左邊', en:'bisect_left stops left on a tie (1 == prefix[0])'}
                          : {zh:'落在 index 1', en:'lands on index 1'})
               : (t === 0 ? {zh:'0 < prefix[0] = 1：index 0', en:'0 < prefix[0] = 1: index 0'}
                          : {zh:'第一個 > ' + t + ' 的 prefix 是 4：index 1', en:'the first prefix > ' + t + ' is 4: index 1'}), COL.pale]],
        cap:{zh:'已列舉 ' + counts.n + ' / ' + nT + ' 個 target', en:counts.n + ' of ' + nT + ' targets enumerated'},
        cap2:{zh:'每個 target 機率都是 1/' + nT + '，所以 index 的機率 = 它擁有的 target 數 / ' + nT, en:'each target has chance 1/' + nT + ', so an index\'s chance = targets it owns / ' + nT}}),
      panels:[{lbl:'target', chips:[chip(String(t), 'hot')]}, {lbl:'index', chips:[chip(String(ix), 'ok')]}],
      line:bug ? 11 : 6,
      msg:bug ? (t === 1 ? {zh:'target = 1 等於 prefix[0]。bisect_left 回傳「第一個 ≥ 1 的位置」= 0，所以 index 0 也拿到 1。加上多出來的 0，index 0 一共拿到 0 和 1 兩個 target，在 5 個裡面——2/5，而它應該只有 1/4。',
                           en:'target = 1 equals prefix[0]. bisect_left returns the first position >= 1, which is 0, so index 0 gets 1 too. Together with the extra 0, index 0 owns targets 0 and 1 out of 5 - 2/5, when it should have 1/4.'}
                         : {zh:'target = ' + t + ' → index ' + ix + '。', en:'target = ' + t + ' → index ' + ix + '.'})
              : (t === 0 ? {zh:'target = 0，bisect_right 找第一個嚴格大於 0 的 prefix：prefix[0] = 1，所以是 index 0。index 0 只擁有 target 0 這一格。',
                            en:'target = 0: bisect_right finds the first prefix strictly greater than 0, prefix[0] = 1, so index 0. Index 0 owns only target 0.'}
                         : {zh:'target = ' + t + '：第一個嚴格大於 ' + t + ' 的 prefix 是 prefix[1] = 4，所以是 index 1。注意 target = 1 剛好等於 prefix[0]：用 bisect_right 才會跳過它，歸給 index 1。',
                            en:'target = ' + t + ': the first prefix strictly greater than ' + t + ' is prefix[1] = 4, so index 1. Note target = 1 equals prefix[0]: bisect_right is what skips past it and gives it to index 1.'})});
  }
  F.push({shapes:wStage(w, nT, owner, -1, counts, {want:bug ? want : null, lines:[
        [bug ? {zh:'index 0：2/5 = 40%（應為 25%）', en:'index 0: 2/5 = 40% (should be 25%)'} : {zh:'index 0：1/4，index 1：3/4', en:'index 0: 1/4, index 1: 3/4'}, bug ? COL.red : COL.tealL],
        [bug ? {zh:'w = [1, 99] 時：2/101 ≈ 2%，應為 1%', en:'with w = [1, 99]: 2/101 ≈ 2%, should be 1%'} : {zh:'建 prefix O(n)，每次抽 O(log n)', en:'O(n) to build prefix, O(log n) per pick'}, COL.pale]],
      cap:bug ? {zh:'多一個值 + 錯的 bisect：第一個 index 多拿一格', en:'one extra value + the wrong bisect: the first index gets a bonus cell'}
              : {zh:'正確：target 在 0…total − 1，配 bisect_right', en:'correct: target in 0…total - 1, paired with bisect_right'},
      capc:bug ? COL.red : COL.tealL,
      cap2:bug ? {zh:'修法：randrange(total) + bisect_right，或 randint(1, total) + bisect_left', en:'fix: randrange(total) + bisect_right, or randint(1, total) + bisect_left'}
               : {zh:'兩種正確配對：0…total−1 用 bisect_right；1…total 用 bisect_left', en:'two correct pairings: 0…total-1 with bisect_right; 1…total with bisect_left'}}),
    panels:[{lbl:{zh:'結果', en:'result'}, chips:[0, 1].map(k => chip('index ' + k + ' ' + frac(counts[k] || 0, counts.n), bug ? 'hot' : 'ok'))}],
    line:bug ? 10 : 6,
    msg:bug ? {zh:'w = [1, 3] 時 index 0 從 25% 變 40%，丟幾千次就看得出來；但權重懸殊時偏差變小：w = [1, 99] 只是 1% 變 2%，抽樣測試很容易放過，而精確列舉照樣算出 2/101。兩個錯其實是同一件事：target 的範圍和 bisect 的方向要成對。',
               en:'With w = [1, 3], index 0 goes from 25% to 40%, which a few thousand picks would show; with lopsided weights the bias shrinks: w = [1, 99] is just 1% becoming 2%, easy for a sampling test to wave through, while exact enumeration still prints 2/101. The two mistakes are one: the target\'s range and the bisect\'s direction must be a matched pair.'}
            : {zh:'index 0 擁有 1 個 target、index 1 擁有 3 個，所以機率就是 1/4、3/4，等於 w / sum(w)。每次抽 target 都是均勻的，所有的「加權」都放在 prefix 的長度裡。下一個選項：大型語言模型挑下一個 token，用的就是這一招。',
               en:'Index 0 owns 1 target and index 1 owns 3, so the chances are 1/4 and 3/4 - exactly w / sum(w). The target is always uniform; all of the weighting lives in the lengths in prefix. Next option: a large language model picks its next token with exactly this step.'}});
  return F.list;
}
const TOKENS = ['the', 'a', 'cat'], LOGITS = [2.0, 1.0, .1];
function softmax(z, T){ const m = Math.max.apply(null, z), e = z.map(x => Math.exp((x - m) / T)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); }
function softStage(p, o){
  const sh = []; let x = NL0;
  p.forEach((pi, k) => {
    const ww = pi * NLW, lab = TOKENS[k];
    sh.push(S.r(x, .9, Math.max(ww - .05, .02), .75, ['ok', 'act', 'hot'][k], ww > .5 ? lab : '', {fs:fitFs(lab, Math.max(ww - .05, .1), .32)}));
    sh.push(S.t(x + ww / 2, .7, pi.toFixed(3), {c:COL.grey, fs:.22}));
    x += ww;
  });
  sh.push(S.t(NL0, 2.0, '0', {c:COL.grey, fs:.24}));
  sh.push(S.t(NL0 + NLW, 2.0, '1', {c:COL.grey, fs:.24}));
  (o.lines || []).forEach((l, k) => sh.push(fitT(NL0, 2.75 + k * .48, l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
  return caps(sh, o);
}
function softFrames(){
  const F = new Frames();
  F.push({shapes:softStage([1 / 3, 1 / 3, 1 / 3], {lines:[[{zh:'模型對下一個字的分數（logits）：the 2.0、a 1.0、cat 0.1', en:'the model\'s scores (logits) for the next word: the 2.0, a 1.0, cat 0.1'}, COL.purpleL],
        [{zh:'softmax(logits / T) 變成機率，再做 LC 528 的加權抽樣', en:'softmax(logits / T) makes probabilities, then an LC 528 pick'}, COL.pale]],
      cap:{zh:'機率就是 [0, 1) 上的長度；抽一個均勻的 target，看它落在哪一段', en:'probabilities are lengths on [0, 1); draw a uniform target and see which piece it hits'},
      cap2:{zh:'這一格只是三等分的示意，下一步才是真的 softmax', en:'this frame is just an even split; the real softmax comes next'}}),
    panels:[], line:0,
    msg:{zh:'大型語言模型每產生一個 token，最後一步就是這一頁的加權抽樣：模型給每個候選字一個分數（logit），softmax 把分數變成機率，再在 prefix sum 上二分搜尋一個均勻的 target。temperature 在 softmax 之前把分數除掉，決定分布有多尖。',
         en:'Every token a large language model produces ends with this tab\'s weighted pick: the model scores each candidate (a logit), softmax turns the scores into probabilities, and a binary search over the prefix sums places a uniform target. Temperature divides the scores before softmax and decides how peaked the distribution is.'}});
  [.5, 1.0, 2.0].forEach(T => {
    const p = softmax(LOGITS, T), pre = []; let s = 0; p.forEach(x => pre.push(s += x));
    const tgt = .7, ix = bisectRight(pre, tgt);
    F.push({shapes:softStage(p, {lines:[['temperature ' + T.toFixed(1) + ':  ' + TOKENS.map((t, k) => t + ' ' + p[k].toFixed(3)).join('   '), COL.orangeL],
          ['prefix = ' + pyList(pre.map(x => x.toFixed(3))), COL.purpleL],
          [{zh:'例：target = 0.7 → bisect_right = ' + ix + '（' + TOKENS[ix] + '）', en:'e.g. target = 0.7 → bisect_right = ' + ix + ' (' + TOKENS[ix] + ')'}, COL.pale]],
        cap:T < 1 ? {zh:'溫度低：分數差距被放大，幾乎總是挑最高分的 the', en:'low temperature: gaps are magnified, almost always the top word "the"'}
           : T > 1 ? {zh:'溫度高：分數差距被壓小，分布變平，冷門的字也常被抽到', en:'high temperature: gaps shrink, the distribution flattens, rare words come up often'}
                   : {zh:'溫度 1：直接用模型的分數', en:'temperature 1: the model\'s scores as they are'},
        cap2:{zh:'temperature → 0 等於永遠挑最大（greedy decoding）', en:'temperature → 0 means always the top one (greedy decoding)'}}),
      panels:[{lbl:'T', chips:[chip(T.toFixed(1), 'act')]}, {lbl:'p', chips:TOKENS.map((t, k) => chip(t + ' ' + p[k].toFixed(3), k === 0 ? 'ok' : 'dim'))}],
      line:T === 1 ? 6 : 2,
      msg:{zh:'temperature ' + T.toFixed(1) + '：the ' + p[0].toFixed(3) + '、a ' + p[1].toFixed(3) + '、cat ' + p[2].toFixed(3) + '。' + (T < 1 ? '除以 0.5 等於把分數差距加倍，指數函數再把差距放大，第一名拿走大部分的長度。' : T > 1 ? '除以 2 把差距減半，cat 的長度變成溫度 0.5 時的十倍左右。寫作、聊天想要多樣性會調高溫度。' : 'softmax 先減掉最大值再取 exp，是為了避免溢位，不改變結果。'),
           en:'temperature ' + T.toFixed(1) + ': the ' + p[0].toFixed(3) + ', a ' + p[1].toFixed(3) + ', cat ' + p[2].toFixed(3) + '. ' + (T < 1 ? 'Dividing by 0.5 doubles the gaps, the exponential magnifies them, and the top word takes most of the length.' : T > 1 ? 'Dividing by 2 halves the gaps; cat\'s share is about ten times what it was at 0.5. Higher temperature is how chat and writing get variety.' : 'Softmax subtracts the max before exp only to avoid overflow; the result is unchanged.')}});
  });
  return F.list;
}

const DAY_META = {
  title:{zh:'Day 42 — 隨機化演算法與蒙地卡羅', en:'Day 42 - Randomized algorithms and Monte Carlo'},
  sub:{zh:'丟飛鏢估 π、用隨機底數判斷質數，以及怎麼精確地測試一段用到亂數的程式。', en:'Estimating π with darts, testing primes with random bases, and testing random code exactly.'},
  tabs:[
    {
      id:'pi', label:{zh:'丟飛鏢估 π', en:'darts for π'},
      stage:{zh:'在正方形裡均勻丟飛鏢：落在四分之一圓裡的比例 × 4 ≈ π', en:'throw darts uniformly at the square: 4 x the share inside the quarter circle ≈ π'},
      view:VIEW,
      variants:[{zh:'一支一支丟', en:'one dart at a time'},
                {zh:'誤差 ∝ 1/√n', en:'error ∝ 1/√n'}],
      idea:{zh:'四分之一圓佔正方形 π/4 的面積，所以均勻亂丟的飛鏢有 π/4 的機率落在圓裡，4 × 比例就是 π 的估計。一支飛鏢只會得 4 或 0 分，標準差 1.642；平均 n 支以後誤差約 1.642 / √n：1,000 支約 0.05、100,000 支約 0.005——多丟 100 倍，誤差只少 10 倍，要 6 位正確小數得丟約 10.8 兆支。這條 1/√n 和維度無關，所以高維積分、強化學習估期望報酬都靠它。這一頁的飛鏢用固定 seed 的亂數產生，同一個 seed 會重現同一批飛鏢。',
            en:'The quarter circle covers π/4 of the square, so a uniformly random dart lands inside with probability π/4, and 4 x the share inside estimates π. A single dart scores 4 or 0 with standard deviation 1.642; averaging n darts gives an error of about 1.642 / √n: about 0.05 at 1,000 darts and 0.005 at 100,000 - 100 times the darts for 10 times less error, and six correct decimals would need about 10.8 trillion darts. The 1/√n does not depend on the dimension, which is why high-dimensional integrals and expected returns in reinforcement learning rely on it. The darts on this page come from a seeded generator, so the same seed replays the same darts.'},
      legend:[['#3fe0dd', {zh:'圓裡的飛鏢 / 實測誤差', en:'dart inside / measured error'}],
              ['#ff9736', {zh:'圓外的飛鏢 / 最新一點', en:'dart outside / newest dot'}],
              ['#9d6bff', {zh:'四分之一圓 / 1.642 / √n', en:'quarter circle / 1.642 / √n'}]],
      get code(){ return CODE_PI; },
      build:piFrames
    },
    {
      id:'prime', label:{zh:'Monte Carlo 判質數', en:'Monte Carlo primality'},
      stage:{zh:'561 = 3 × 11 × 17：558 個底數裡，有幾個會說它是質數？', en:'561 = 3 x 11 x 17: how many of the 558 bases call it prime?'},
      view:VIEW,
      variants:[{zh:'Fermat', en:'Fermat'},
                {zh:'Miller-Rabin', en:'Miller-Rabin'}],
      idea:{zh:'隨機化演算法分兩種：Las Vegas 答案永遠對、只有時間是隨機的（隨機 pivot 的 quicksort：已排序的 2,000 個數，固定拿第一個當 pivot 要 1,999,000 次比較，隨機 pivot 平均約 24,730 次）；Monte Carlo 時間固定、答案可能錯。質數測試是 Monte Carlo：說「合數」一定對，說「質數」可能錯。Fermat 對 Carmichael 數 561 有 318/558（57%）的底數說謊，多試幾個也沒用；Miller-Rabin 多檢查「1 的平方根只能是 ±1」，只有 8/558（1.4%）說謊，而且保證任何奇合數都不超過 1/4。固定用 2 到 37 這 12 個質數當底數，所有 64 位元整數都不會錯。',
            en:'Randomized algorithms come in two kinds: Las Vegas is always right and only its running time is random (quicksort with a random pivot: on 2,000 already sorted numbers, first-element pivots cost 1,999,000 comparisons, random pivots about 24,730 on average); Monte Carlo has a fixed running time and may be wrong. Primality testing is Monte Carlo: "composite" is always right, "prime" may be wrong. Fermat is fooled by 318 of 558 bases (57%) on the Carmichael number 561, and more tries do not help; Miller-Rabin also checks that 1 has no square roots but ±1, is fooled by only 8 of 558 (1.4%), and is guaranteed at most 1/4 for any odd composite. With the 12 fixed prime bases 2 to 37 it is never wrong on a 64-bit integer.'},
      legend:[['#ff5c5c', {zh:'說謊的底數（說是質數）', en:'lying base (says prime)'}],
              ['#2f5661', {zh:'抓到是合數', en:'catches the composite'}],
              ['#ff9736', {zh:'平方的過程', en:'the squaring chain'}]],
      get code(){ return CODE_PRIME; },
      build:primeFrames
    },
    {
      id:'shuffle', label:{zh:'洗牌（LC 384）', en:'shuffling (LC 384)'},
      stage:{zh:'把每一種亂數選擇都跑一次：每個順序各出現幾次？', en:'run every possible sequence of random choices once: how often does each order come out?'},
      view:VIEW,
      variants:[{zh:'天真：j 任選', en:'naive: any j'},
                {zh:'randrange(i)', en:'randrange(i)'},
                {zh:'Fisher-Yates', en:'Fisher-Yates'}],
      idea:{zh:'測試亂數程式最可靠的方法不是丟一萬次看直方圖，而是把 randrange 換成「依序吐出每一種選擇」的假亂數，所有跑法各跑一次、用分數加總。3 張牌：每一步任選一格有 27 種跑法，27 不能被 6 整除，所以 acb、bac、bca 各 5/27，其他 4/27；把 randrange(i + 1) 寫成 randrange(i) 就變成 Sattolo 演算法，只到得了 bca、cab，abc 機率是 0；Fisher-Yates 6 種跑法一對一對應 6 種順序，各 1/6。LeetCode 384 的 shuffle() 就是 Fisher-Yates。',
            en:'The reliable way to test random code is not ten thousand trials and a histogram; it is to replace randrange with a fake that hands out every possible choice in turn, run each sequence once and add up exact fractions. Three cards: "swap with any slot" has 27 runs, 27 is not divisible by 6, so acb, bac, bca get 5/27 and the rest 4/27; writing randrange(i) instead of randrange(i + 1) turns it into Sattolo\'s algorithm, which reaches only bca and cab, with abc at probability 0; Fisher-Yates has 6 runs mapping one-to-one onto the 6 orders, 1/6 each. LeetCode 384\'s shuffle() is Fisher-Yates.'},
      legend:[['#9d6bff', {zh:'位置 i / 公平線 1/6', en:'slot i / the fair 1/6 line'}],
              ['#ff9736', {zh:'挑到的 j / 剛得到的順序', en:'chosen j / the order just produced'}],
              ['#3fe0dd', {zh:'出現次數', en:'how often it comes out'}],
              ['#ff5c5c', {zh:'不公平', en:'unfair'}]],
      get code(){ return CODE_SHUF; },
      build:shufFrames
    },
    {
      id:'reservoir', label:{zh:'蓄水池抽樣（LC 382）', en:'reservoir (LC 382)'},
      stage:{zh:'不知道有幾個、只能記一個：第 i 個以 1/i 的機率換進來', en:'unknown length, room for one: item i swaps in with chance 1/i'},
      view:VIEW,
      variants:[{zh:'正確：1/i', en:'correct: 1/i'},
                {zh:'i 從 0 數', en:'i counted from 0'}],
      idea:{zh:'第 i 個以 1/i 換進來，前面每個沒被換掉的機率是 (i−1)/i，乘起來一路約掉：a 最後被留下的機率 1 × 1/2 × 2/3 × 3/4 × 4/5 = 1/5，每個元素都是 1/n，而且從頭到尾不需要知道 n。把 i 從 0 開始數，randrange(1) 永遠是 0，b 一定把 a 換掉：結果 a 0、其他各 1/4。LeetCode 382 跑 30,000 次，值 1 一次都沒出現；但 b c d e 平均得很漂亮，抽樣測試很容易放過。留 k 個的版本是 Algorithm R：k = 2 從 5 個裡抽，10 種組合各 1/10。',
            en:'Item i swaps in with chance 1/i and every earlier item survives with (i-1)/i, so the products cancel all the way: a ends up held with 1 x 1/2 x 2/3 x 3/4 x 4/5 = 1/5, every item gets 1/n, and n is never needed. Count i from 0 and randrange(1) is always 0, so b always replaces a: a 0, the others 1/4 each. Running LeetCode 382 30,000 times, the value 1 never appears - yet b, c, d, e look beautifully even, and a sampling test easily lets it through. Keeping k items is Algorithm R: k = 2 from 5 gives 10 pairs at 1/10 each.'},
      legend:[['#ff9736', {zh:'這一個換進來了', en:'this one swapped in'}],
              ['#9d6bff', {zh:'這一個沒換', en:'this one did not'}],
              ['#3fe0dd', {zh:'被留下的機率', en:'chance of being held'}],
              ['#ff5c5c', {zh:'機率 0', en:'probability 0'}]],
      get code(){ return CODE_RES; },
      build:resFrames
    },
    {
      id:'rand10', label:{zh:'拒絕取樣（LC 470）', en:'rejection (LC 470)'},
      stage:{zh:'兩次 rand7 = 49 格；留 40 格、丟 9 格', en:'two rand7 calls = 49 cells; keep 40, throw away 9'},
      view:VIEW,
      variants:[{zh:'拒絕取樣', en:'rejection sampling'},
                {zh:'直接 % 10', en:'just % 10'}],
      idea:{zh:'(rand7() − 1) × 7 + rand7() 得到 1 到 49 均勻；49 不是 10 的倍數，所以只留前 40 格（1 到 10 各 4 格），其餘丟掉重抽。接受率 40/49，期望 2 × 49/40 = 2.45 次 rand7；把被拒絕那一組剩下的亂數接著用，可以降到約 2.19 次。偷懶不拒絕、直接 idx % 10 + 1：數字 1 只拿到 4/49，其他 5/49。同樣的道理，Python 的 randrange 內部也是用拒絕取樣避免 % 的偏差。',
            en:'(rand7() - 1) x 7 + rand7() is uniform over 1 to 49; 49 is not a multiple of 10, so keep the first 40 cells (four of each digit) and redraw on the rest. Acceptance is 40/49, an expected 2 x 49/40 = 2.45 rand7 calls; reusing the leftover randomness of a rejected draw brings it down to about 2.19. Skip the rejection and use idx % 10 + 1, and digit 1 gets 4/49 while the others get 5/49. Python\'s randrange uses rejection internally for the same reason.'},
      legend:[['#ff9736', {zh:'這一組落在的格子', en:'the cell this draw lands on'}],
              ['#ff5c5c', {zh:'被丟掉的格子 / 少一格的數字', en:'a discarded cell / the short digit'}],
              ['#3fe0dd', {zh:'接受', en:'accepted'}],
              ['#9d6bff', {zh:'rand7 的值', en:'rand7 values'}]],
      get code(){ return CODE_470; },
      build:r10Frames
    },
    {
      id:'weight', label:{zh:'加權抽樣（LC 528）', en:'weighted pick (LC 528)'},
      stage:{zh:'prefix sum + 二分搜尋：每個 index 擁有幾個 target', en:'prefix sums + binary search: how many targets each index owns'},
      view:VIEW,
      variants:[{zh:'randrange + bisect_right', en:'randrange + bisect_right'},
                {zh:'randint + bisect_left', en:'randint + bisect_left'},
                {zh:'softmax 與溫度', en:'softmax and temperature'}],
      idea:{zh:'把權重接成一條線，prefix sum 是每一段的右端；均勻抽一個 0…total − 1 的 target，bisect_right 找出它落在哪一段，O(log n)。w = [1, 3]：index 0 擁有 1 個 target、index 1 擁有 3 個，機率 1/4、3/4。改成 randint(0, total) 加 bisect_left，target 多一個值、又在相等時往左停，index 0 變成 2/5；w = [1, 99] 時是 2/101 對 1/100，抽樣很難看出來。大型語言模型挑下一個 token 就是這一步：softmax(logits / temperature) 再做加權抽樣，溫度 0.5 時 the 0.864，溫度 2.0 時 0.502。',
            en:'Lay the weights end to end; prefix sums mark where each piece ends. Draw a uniform target in 0…total - 1 and bisect_right finds its piece in O(log n). With w = [1, 3], index 0 owns 1 target and index 1 owns 3: 1/4 and 3/4. Switch to randint(0, total) with bisect_left and the target has one value too many and stops left on ties, so index 0 gets 2/5; with w = [1, 99] it is 2/101 against 1/100, hard to see by sampling. A large language model picks its next token with this exact step: softmax(logits / temperature), then a weighted pick - "the" gets 0.864 at temperature 0.5 and 0.502 at 2.0.'},
      legend:[['#9d6bff', {zh:'每個 index 擁有的範圍', en:'the range each index owns'}],
              ['#ff9736', {zh:'目前的 target / 它選中的 index', en:'the current target / the index it picks'}],
              ['#3fe0dd', {zh:'機率正確', en:'correct probability'}],
              ['#ff5c5c', {zh:'機率偏掉', en:'skewed probability'}]],
      get code(){ return curV() === 2 ? CODE_SOFT : CODE_528; },
      build:wFrames
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
