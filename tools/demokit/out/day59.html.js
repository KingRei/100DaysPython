// DAY: 59
// TITLE_ZH: 取樣策略：temperature、top-k、top-p 怎麼決定下一個 token
// TITLE_EN: Temperature, top-k and top-p: how the next token gets chosen
// SUB_ZH: 模型吐出來的是一整排 logits，把它變成一個 token 的那一步就是取樣。永遠取最大值，模型只會講同一句話；照原始機率抽，尾巴那幾萬個不可能的 token 遲早會被抽中。temperature 改形狀，top-k 和 top-p 把尾巴剪掉，而兩者的先後順序會改變答案。
// SUB_EN: A model emits a whole row of logits; turning that row into one token is the sampling step. Always take the maximum and the model repeats itself; draw from the raw distribution and the tail - tens of thousands of individually hopeless tokens - eventually gets drawn. Temperature reshapes, top-k and top-p truncate, and the order they are applied in changes the answer.
// FOLDER: day%2059%20-%20sampling%20strategies
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
const fmtN = x => x == null ? 'nan' : String(Math.round(x * 1e4) / 1e4);
const PURPLE = '#b45cff';

function barRow(sh, x0, base, bw, gap, vals, sts, labs, maxh, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const hgt = Math.max(.05, maxh * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, base - hgt, bw, hgt, sts[i] || 'ok', null));
    if (labs && labs[i] != null) sh.push(fitT(x + bw / 2, base - hgt - .20, labs[i], {c:COL.pale, fs:.24}));
  });
  return sh;
}
// a 0..1 x 0..1 plot box; data y grows upward, SVG y grows downward
function Panel(x0, y0, w, h){ this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; }
Panel.prototype.px = function(p){
  return [this.x0 + this.w * p[0], this.y0 + this.h * (1 - p[1])];
};
Panel.prototype.frame = function(sh){
  sh.push(S.r(this.x0, this.y0, this.w, this.h, 'ghost', null));
  return sh;
};
function polyline(sh, panel, pts, st, w){
  for (let i = 1; i < pts.length; i++){
    const a = panel.px(pts[i - 1]), b = panel.px(pts[i]);
    sh.push(S.e(a[0], a[1], b[0], b[1], {s:st, arrow:false, w:w || .055}));
  }
  return sh;
}
const FRAG = {"reqs": 240, "rejC": 12, "rejFrag": 12, "rejP": 5, "utilC": 65.7, "utilP": 68.5, "capacity": 16384};
const STEP = {"ctx": "the small cat ", "labs": ["s", "r", "w", "NL", "b", "_"], "probs": [0.6099, 0.2232, 0.1637, 0.0001, 0.0001, 0.0001], "tail": 0.003123, "rest": 21, "risk30": 0.0896, "H": 1.385, "nuc": 3, "V": 24};
const GREEDY = {"text": "the red cat sang in the garden.", "total": 1250};
const TEMP = [{"t": 0.0, "H": 0.0, "g": 1.0, "d": 1}, {"t": 0.3, "H": 0.2, "g": 1.0, "d": 79}, {"t": 0.7, "H": 0.269, "g": 1.0, "d": 98}, {"t": 1.0, "H": 0.336, "g": 0.9, "d": 88}, {"t": 1.3, "H": 0.557, "g": 0.55, "d": 53}, {"t": 1.7, "H": 1.182, "g": 0.05, "d": 5}, {"t": 2.0, "H": 1.787, "g": 0.01, "d": 1}];
const RESCUE = [{"name": "no truncation", "g": 0.0, "d": 0}, {"name": "top-k 1 (greedy)", "g": 1.0, "d": 2}, {"name": "top-k 3", "g": 0.0, "d": 0}, {"name": "top-p 0.9", "g": 0.02, "d": 2}, {"name": "top-p 0.7", "g": 0.89, "d": 85}, {"name": "top-p 0.5", "g": 1.0, "d": 80}, {"name": "min-p 0.1", "g": 1.0, "d": 96}, {"name": "typical 0.9", "g": 0.02, "d": 2}, {"name": "typical 0.5", "g": 0.94, "d": 84}];
const SEQ = {"best": "a small dog sat in the garden.", "bestLp": -6.244, "greedy": "the red cat sang in the garden.", "greedyLp": -6.717, "rank": 44, "n": 1250, "ratio": 1.61, "legalMass": 0.9088, "expDistinct": 87.3, "draws": 100};
const TYPC = {"steps": 1875, "same": 1875, "drops": 0, "H": 2.28, "mass": 0.25, "topPn": 1, "typN": 9, "rows": [{"name": "dominant, p=0.60", "s": 0.51, "d": 1.77}, {"name": "middle, p=0.03", "s": 3.51, "d": 1.23}, {"name": "tail, p=0.0001", "s": 9.21, "d": 6.93}]};
const SHAPE = {"peaked": {"ctx": "the small cat sa", "labs": ["t", "n", ".", "NL", "b", "c", "d"], "probs": [0.5551, 0.4362, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004], "nuc": 2, "k3": 0.9917, "mp": 2}, "flat": {"ctx": "the", "labs": ["r", "q", "o", "s", "c", "b", "d"], "probs": [0.2742, 0.1977, 0.1951, 0.1734, 0.0714, 0.0446, 0.0434], "nuc": 5, "k3": 0.6669, "mp": 7}, "nPeaked": 249, "nFlat": 61, "nucPeaked": 1.0, "nucFlat": 3.46, "k1Peaked": 0.0045, "k1Flat": 0.5413};
const ORDER = {"ctx": "the small cat s", "t": 2.5, "p": 0.9, "n_a": 18, "n_b": 2, "max_a": 0.3943, "max_b": 0.5445, "tv": 0.2759};
const PIVOT = {"steps": [{"mid": 0.13708, "mass": 0.8404, "n": 4}, {"mid": 0.06854, "mass": 0.9118, "n": 5}, {"mid": 0.10281, "mass": 0.8404, "n": 4}, {"mid": 0.08568, "mass": 0.8404, "n": 4}, {"mid": 0.07711, "mass": 0.8404, "n": 4}, {"mid": 0.07282, "mass": 0.8404, "n": 4}, {"mid": 0.07068, "mass": 0.9118, "n": 5}, {"mid": 0.07175, "mass": 0.8404, "n": 4}, {"mid": 0.07122, "mass": 0.9118, "n": 5}, {"mid": 0.07149, "mass": 0.8404, "n": 4}], "n": 5, "mass": 0.9118, "sorted": 5, "probs": [0.2742, 0.1977, 0.1951, 0.1734, 0.0714, 0.0446, 0.0434], "labs": ["r", "q", "o", "s", "c", "b", "d"]};
const COST = {"V": 151936, "k": 50, "full": 2.6, "partial": 0.15, "ratio": 17};
const BUG = {"kept": 0.9969, "over": 0.0027};

/* shared: a little probability bar chart with optional cut line */
function probBars(sh, x0, base, probs, labs, keep, maxh, bw, gap){
  bw = bw || .62; gap = gap || .22; maxh = maxh || 2.4;
  const hi = Math.max.apply(null, probs);
  probs.forEach((v, i) => {
    const h = Math.max(.06, maxh * v / hi), x = x0 + i * (bw + gap);
    const on = !keep || keep[i];
    sh.push(S.r(x, base - h, bw, h, on ? 'hot' : 'ghost', null));
    sh.push(fitT(x + bw / 2, base + .30, labs[i], {c:on ? COL.pale : COL.grey, fs:.26}));
    sh.push(fitT(x + bw / 2, base - h - .16, v < .001 ? '~0' : v.toFixed(2),
                 {c:on ? COL.tealL : COL.grey, fs:.22}));
  });
  return sh;
}

/* ================================================================ tab 1: the tail */
const CODE_TAIL = [
  'logits = model(tokens)[-1]            # one row, one entry per token',
  'p = softmax(logits)',
  '',
  'tail = p.sum() - np.sort(p)[-3:].sum()   # everything below the top 3',
  'risk = 1 - (1 - tail) ** 30              # over a 30-step sentence',
  '',
  'tok = argmax(p)                       # greedy: always the same sentence'
];
function tailStage(o){
  const sh = [];
  sh.push(fitT(4.9, 1.05, {zh:'context：' + JSON.stringify(STEP.ctx),
                           en:'context ' + JSON.stringify(STEP.ctx)},
               {c:COL.grey, fs:.28}));
  probBars(sh, 1.35, 4.35, STEP.probs, STEP.labs,
           o.upto >= 2 ? STEP.probs.map(v => v > .01) : null, 2.5, .66, .36);
  if (o.upto >= 1){
    sh.push(S.r(7.75, 4.23, 1.30, .12, 'bad', null));
    sh.push(fitT(8.40, 4.75, {zh:'其他 ' + STEP.rest + ' 個字元', en:STEP.rest + ' more characters'},
                 {c:COL.red, fs:.24}));
    sh.push(fitT(8.40, 5.05, (STEP.tail * 100).toFixed(2) + '%', {c:COL.red, fs:.26}));
  }
  if (o.upto >= 3){
    sh.push(fitT(4.9, 1.60, {zh:'greedy：' + JSON.stringify(GREEDY.text),
                             en:'greedy: ' + JSON.stringify(GREEDY.text)},
                 {c:COL.tealL, fs:.26}));
  }
  return caps(sh, o);
}
function tailFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:tailStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'一個 decode step：一排 logits 進來，一個 token 出去',
         en:'one decode step: a row of logits in, one token out'},
    panels:[{lbl:'vocabulary', chips:[chip(String(STEP.V) + ' chars', 'dim')]},
            {lbl:'entropy', chips:[chip(STEP.H.toFixed(2) + ' bits', 'dim')]}],
    msg:{zh:'模型每一步給的是一整排分數，softmax 之後變成機率。這個 context 下前三名就吃掉 ' + ((STEP.probs[0] + STEP.probs[1] + STEP.probs[2]) * 100).toFixed(1) + '%——看起來很乾淨。',
         en:'Every step the model emits a whole row of scores, and softmax turns it into a distribution. Here the top three characters already hold ' + ((STEP.probs[0] + STEP.probs[1] + STEP.probs[2]) * 100).toFixed(1) + '% of it, which looks clean enough.'}});
  p({upto:1, line:3,
    cap:{zh:'但尾巴不是零', en:'but the tail is not zero'},
    panels:[{lbl:'tail mass', chips:[chip((STEP.tail * 100).toFixed(2) + '%', 'bad')]},
            {lbl:'tail tokens', chips:[chip(String(STEP.rest), 'dim')]}],
    msg:{zh:'剩下 ' + STEP.rest + ' 個字元加起來只有 ' + (STEP.tail * 100).toFixed(2) + '%。真實模型的尾巴是十幾萬個 token——每一個都不可能，加起來卻是一個會發生的事件。',
         en:'The other ' + STEP.rest + ' characters are worth ' + (STEP.tail * 100).toFixed(2) + '% in total. On a real model that tail is a hundred thousand tokens: each one impossible, all of them together an event that happens.'}});
  p({upto:1, line:4,
    cap:{zh:'一句話 30 步，踩到尾巴的機率是 ' + (STEP.risk30 * 100).toFixed(1) + '%',
         en:'over a 30-step sentence the chance of stepping into it is ' + (STEP.risk30 * 100).toFixed(1) + '%'},
    panels:[{lbl:'per step', chips:[chip((STEP.tail * 100).toFixed(2) + '%', 'dim')]},
            {lbl:'per sentence', chips:[chip((STEP.risk30 * 100).toFixed(1) + '%', 'bad')]}],
    msg:{zh:(STEP.tail * 100).toFixed(2) + '% 看起來可以忽略，但要連續 30 步都不踩到才行：1 − ' + (1 - STEP.tail).toFixed(4) + '³⁰ ＝ ' + (STEP.risk30 * 100).toFixed(1) + '%。而且只要踩到一次就毀了整句——接下來模型得從一個它沒看過的前綴繼續寫。',
         en:'A per-step ' + (STEP.tail * 100).toFixed(2) + '% looks ignorable until you need thirty steps in a row to avoid it: 1 - ' + (1 - STEP.tail).toFixed(4) + '^30 = ' + (STEP.risk30 * 100).toFixed(1) + '%. One hit ruins the sentence, because the model then has to continue from a prefix it has never seen.'}});
  p({upto:3, line:6,
    cap:{zh:'那永遠取最大值呢？', en:'so why not always take the maximum?'},
    panels:[{lbl:'greedy grammatical', chips:[chip('100%', 'ok')]},
            {lbl:'distinct sentences', chips:[chip('1 / ' + GREEDY.total, 'bad')]}],
    msg:{zh:'greedy 永遠正確，而且永遠講同一句話：文法 100%，但 ' + GREEDY.total + ' 句合法句子裡它只會說 1 句。這不是「比較差的取樣」，是完全沒有取樣。接下來三個分頁就是在這兩端之間找位置。',
         en:'Greedy is always correct and always says the same thing: 100% grammatical, 1 distinct sentence out of ' + GREEDY.total + ' legal ones. That is not worse sampling, it is no sampling. The next three tabs are about finding a place between these two ends.'}});
  return F.list;
}

/* ================================================================ tab 2: the sequence level */
const CODE_SEQ = [
  '# score every sentence the grammar admits, exactly',
  'def sentence_logprob(model, s, prefix="\\n"):',
  '    ctx, tot = prefix, 0.0',
  '    for ch in s + "\\n":',
  '        tot += log(softmax(model.logits(ctx))[model.stoi[ch]])',
  '        ctx += ch',
  '    return tot',
  '',
  'scored = sorted((sentence_logprob(m, s), s) for s in all_sentences())',
  '# greedy maximises each STEP, which is not the same as maximising this'
];
function seqStage(o){
  const sh = [];
  const rows = [[{zh:'整句機率最高的', en:'the most likely sentence'}, SEQ.best, SEQ.bestLp, 'ok'],
                [{zh:'greedy 產生的', en:'what greedy produces'}, SEQ.greedy, SEQ.greedyLp, 'hot']];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.30 - i * 1.30;
    sh.push(fitT(.70, y + .62, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
    sh.push(box(.70, y, 7.10, .58, r[3], r[1], .26));
    sh.push(fitT(9.20, y + .29, 'log p ' + r[2].toFixed(2), {anchor:'end', c:COL.pale, fs:.26}));
  });
  if (o.upto >= 2){
    const w = 8.50, x0 = .70, frac = SEQ.rank / SEQ.n;
    sh.push(S.r(x0, 1.30, w, .34, 'ghost', null));
    sh.push(S.r(x0, 1.30, Math.max(.06, w * frac), .34, 'bad', null));
    sh.push(fitT(x0, 1.00, {zh:'greedy 的句子在 ' + SEQ.n.toLocaleString() + ' 句裡排第 ' + SEQ.rank,
                            en:'the greedy sentence ranks ' + SEQ.rank + ' of ' + SEQ.n.toLocaleString()},
                 {anchor:'start', c:COL.red, fs:.28}));
  }
  return caps(sh, o);
}
function seqFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:seqStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:8,
    cap:{zh:'把 ' + SEQ.n.toLocaleString() + ' 句合法句子全部算一遍',
         en:'score all ' + SEQ.n.toLocaleString() + ' legal sentences exactly'},
    panels:[{lbl:'sentences', chips:[chip(SEQ.n.toLocaleString(), 'dim')]},
            {lbl:'best log p', chips:[chip(SEQ.bestLp.toFixed(2), 'ok')]}],
    msg:{zh:'文法小到可以把 1,250 句全部列出來，每一句的機率就是逐步 log 機率相加——所以「整句最可能的是哪一句」這個問題有精確答案，不用搜尋也不用估計。',
         en:'The grammar is small enough to enumerate all 1,250 sentences, and each one\'s probability is just the sum of its per-step log probabilities. So "which sentence is the most likely" has an exact answer here - no search, no estimate.'}});
  p({upto:2, line:9,
    cap:{zh:'而 greedy 給的不是那一句', en:'and greedy does not produce it'},
    panels:[{lbl:'greedy rank', chips:[chip(SEQ.rank + ' / ' + SEQ.n.toLocaleString(), 'bad')]},
            {lbl:'less likely by', chips:[chip(SEQ.ratio + 'x', 'bad')]}],
    msg:{zh:'greedy 每一步都取最大值，結果整句排第 ' + SEQ.rank + '，比最好的那句低 ' + SEQ.ratio + ' 倍。原因是它會走進一個「之後沒有好結尾」的前綴，而且回不去——**逐步 argmax ≠ 序列 argmax**。這就是為什麼「在序列上搜尋」是另一種 decoding 策略，而不是同一件事調參數。',
         en:'Greedy takes the maximum at every step and lands on rank ' + SEQ.rank + ', ' + SEQ.ratio + 'x less likely than the best sentence. It walks into a prefix from which no very good ending exists and cannot go back: **argmax per step is not argmax over sequences**. That is why searching over sequences is a different decoding strategy rather than a parameter change.'}});
  p({upto:2, line:9,
    cap:{zh:'順便驗證整套量測是不是可信的', en:'and two checks that the measuring rig is sound'},
    panels:[{lbl:'legal sentence mass', chips:[chip((SEQ.legalMass * 100).toFixed(1) + '%', 'ok')]},
            {lbl:'measured at t=1.0', chips:[chip('90%', 'ok')]},
            {lbl:'expected distinct', chips:[chip(SEQ.expDistinct.toFixed(1), 'ok'),
                                             chip('measured 88', 'ok')]}],
    msg:{zh:'這 1,250 句佔掉模型「句子層級」機率的 ' + (SEQ.legalMass * 100).toFixed(1) + '%，而 t=1.0 實際抽樣量到的合文法率是 90%；理論上抽 ' + SEQ.draws + ' 句應該會有 ' + SEQ.expDistinct.toFixed(1) + ' 句不重複，實測是 88。兩邊都對得上，所以後面那些比率可以信。',
         en:'These 1,250 sentences hold ' + (SEQ.legalMass * 100).toFixed(1) + '% of the model\'s sentence-level probability, and sampling at t=1.0 measures a 90% grammatical rate. Drawing ' + SEQ.draws + ' sentences should give ' + SEQ.expDistinct.toFixed(1) + ' distinct ones; the measurement says 88. Theory and sampling agree, so the rates in the other tabs can be trusted.'}});
  return F.list;
}

/* ================================================================ tab 3: temperature */
const CODE_TEMP = [
  'def temperature(logits, t):',
  '    if t <= 0:                      # t = 0 means argmax, not a division',
  '        return one_hot(argmax(logits))',
  '    return logits / t               # then softmax',
  '',
  '# t < 1 sharpens: the gap between logits is magnified',
  '# t > 1 flattens: every token moves toward 1/V'
];
function tempStage(o){
  const sh = [], X0 = .75, GW = 1.26, BW = .45, H = 2.55, Y = 1.55;
  TEMP.forEach((r, i) => {
    if (i > o.upto) return;
    const x = X0 + i * GW;
    sh.push(S.r(x, Y, BW, Math.max(.05, H * r.g), i === o.upto ? 'hot' : 'ok', null));
    sh.push(S.r(x + BW + .10, Y, BW, Math.max(.05, H * r.d / 100), 'act', null));
    sh.push(fitT(x + BW, Y - .26, 't=' + r.t, {c:COL.grey, fs:.24}));
    sh.push(fitT(x + BW / 2, Y + H * r.g + .22, Math.round(r.g * 100) + '%',
                 {c:COL.pale, fs:.22}));
    sh.push(fitT(x + 1.5 * BW + .10, Y + H * r.d / 100 + .22, String(r.d),
                 {c:'#c7a6ff', fs:.22}));
  });
  sh.push(fitT(.75, 4.80, {zh:'藍：文法正確的比例（100 個樣本）　紫：不重複的句子數',
                           en:'teal: grammatical, % of 100 samples    purple: distinct sentences'},
               {anchor:'start', c:COL.grey, fs:.25}));
  return caps(sh, o);
}
function tempFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:tempStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'t = 0：正確，但只有一句', en:'t = 0: correct, and only one sentence'},
    panels:[{lbl:'t = 0', chips:[chip('100% ok', 'ok'), chip('1 distinct', 'bad')]}],
    msg:{zh:'temperature 0 在程式裡不是「除以 0」，是直接走 argmax 那條路。所有伺服器都這樣特判，不然會變成 NaN。',
         en:'Temperature 0 is not a division by zero in any real implementation: it is a separate argmax path. Every server special-cases it, otherwise the softmax returns NaN.'}});
  p({upto:2, line:4,
    cap:{zh:'稍微加熱：文法沒掉，句子一下子多了起來',
         en:'a little heat: grammar holds, variety appears'},
    panels:[{lbl:'t = 0.3', chips:[chip('100%', 'ok'), chip('79', 'act')]},
            {lbl:'t = 0.7', chips:[chip('100%', 'ok'), chip('98', 'act')]}],
    msg:{zh:'t = 0.7 時 100 個樣本給出 98 個不同的句子，而且每一句都合文法。這就是實務上 0.6～0.9 最常見的原因——在這個區間，多出來的隨機性幾乎都落在「模型本來就覺得差不多」的分岔點上。',
         en:'At t = 0.7, 100 samples give 98 different sentences and every one of them parses. That is why real defaults sit around 0.6 to 0.9: in that range the extra randomness lands almost entirely on branch points the model was genuinely undecided about.'}});
  p({upto:TEMP.length - 1, line:6,
    cap:{zh:'再熱就崩了，而且崩得很快', en:'past that it collapses, and it collapses fast'},
    panels:[{lbl:'t = 1.3', chips:[chip('55%', 'bad')]},
            {lbl:'t = 1.7', chips:[chip('5%', 'bad')]},
            {lbl:'t = 2.0', chips:[chip('1%', 'bad')]}],
    msg:{zh:'從 t=1.0 到 1.7，entropy 只從 ' + TEMP[3].H.toFixed(2) + ' 漲到 ' + TEMP[5].H.toFixed(2) + ' bits，文法正確率卻從 90% 掉到 5%。注意紫色也跟著掉——亂到一個程度之後連「不重複的合法句子」都生不出來，所以這不是在用正確性換多樣性，是兩個一起輸。',
         en:'From t=1.0 to t=1.7 the entropy only moves from ' + TEMP[3].H.toFixed(2) + ' to ' + TEMP[5].H.toFixed(2) + ' bits while the grammatical rate falls from 90% to 5%. Notice the purple bars fall too: past a point you stop buying variety at all, because a broken sentence is not a different sentence. Both columns lose together.'}});
  return F.list;
}

/* ================================================================ tab 3: truncation */
const CODE_TRUNC = [
  'def top_k(p, k):                      # keep a fixed count',
  '    cut = np.partition(p, -k)[-k]',
  '    q = np.where(p >= cut, p, 0.0)',
  '    return q / q.sum()                # renormalise - never skip this',
  '',
  'def top_p(p, thr):                    # keep a fixed mass (the nucleus)',
  '    order = np.argsort(-p)',
  '    n = np.searchsorted(np.cumsum(p[order]), thr) + 1',
  '    q = zeros_like(p); q[order[:n]] = p[order[:n]]',
  '    return q / q.sum()',
  '',
  'def min_p(p, frac):                   # keep what is worth frac of the best',
  '    q = np.where(p >= frac * p.max(), p, 0.0)',
  '    return q / q.sum()'
];
function truncStage(o){
  const sh = [], Y = 4.62;
  RESCUE.forEach((r, i) => {
    if (i > o.upto) return;
    const y = Y - i * .50;
    sh.push(box(.65, y - .24, 2.55, .48, i === o.upto ? 'hot' : 'idle', r.name, .25));
    sh.push(S.r(3.45, y - .20, Math.max(.05, 3.0 * r.g), .40,
                r.g > .8 ? 'ok' : 'bad', null));
    sh.push(fitT(6.62, y, Math.round(r.g * 100) + '%',
                 {c:r.g > .8 ? COL.tealL : COL.red, fs:.26}));
    sh.push(S.r(7.15, y - .20, Math.max(.05, 1.8 * r.d / 100), .40, 'act', null));
    sh.push(fitT(9.15, y, String(r.d), {c:'#c7a6ff', fs:.26, anchor:'end'}));
  });
  sh.push(fitT(.65, 5.05, {zh:'全部都是 t = 2.0，只是多套了一條截斷規則',
                           en:'every row is temperature 2.0 plus one truncation rule'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function truncFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:truncStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'起點：t = 2.0，0% 合文法', en:'the starting point: t = 2.0, 0% grammatical'},
    panels:[{lbl:'no truncation', chips:[chip('0%', 'bad'), chip('0 distinct', 'bad')]}],
    msg:{zh:'上一個分頁最糟的那一列。接下來溫度完全不動，只改「允許被抽到的 token 有哪些」。',
         en:'The worst row of the previous tab. From here the temperature never changes; the only thing that changes is which tokens are allowed to be drawn at all.'}});
  p({upto:2, line:1,
    cap:{zh:'top-k：固定數量，而且很難挑', en:'top-k: a fixed count, and hard to pick'},
    panels:[{lbl:'top-k 1', chips:[chip('100%', 'ok'), chip('2 distinct', 'bad')]},
            {lbl:'top-k 3', chips:[chip('0%', 'bad')]}],
    msg:{zh:'top-k 1 是 greedy 換個名字，100% 但只有 2 句——分數沒有意義。往上調到 3 就整個崩掉：t=2.0 之後第 2、3 名常常已經是垃圾字元。固定數量的問題就在這裡，它不看分佈長什麼樣子。',
         en:'top-k 1 is greedy wearing a different name: a perfect score worth nothing, 2 distinct sentences. Move to 3 and it collapses, because at t=2.0 the second and third places are often junk already. That is the flaw in a fixed count: it does not look at the shape of the step.'}});
  p({upto:5, line:6,
    cap:{zh:'top-p：固定質量，鬆緊自己調', en:'top-p: a fixed mass, tightening itself'},
    panels:[{lbl:'top-p 0.9', chips:[chip('2%', 'bad')]},
            {lbl:'top-p 0.7', chips:[chip('89%', 'ok'), chip('85', 'act')]},
            {lbl:'top-p 0.5', chips:[chip('100%', 'ok'), chip('80', 'act')]}],
    msg:{zh:'注意 0.9 救不回來：temperature 先把分佈壓平了，0.9 的 nucleus 就跟著變寬（下一個分頁會看到這個順序問題）。0.5 完全救回來，而且還留著 80 句不同的話。這裡的門檻值不能照搬到真模型——我們的詞彙表只有 24 個字元，真模型是十幾萬個 token。',
         en:'Watch 0.9 fail: temperature flattened the distribution first, so a 0.9 nucleus widened with it (the order problem, next tab). At 0.5 the grammar comes back completely and 80 distinct sentences survive. Do not copy these thresholds onto a real model - this vocabulary is 24 characters, not 150,000 tokens.'}});
  p({upto:6, line:12,
    cap:{zh:'min-p：拿「最好的那個」當尺', en:'min-p: measure against the best token'},
    panels:[{lbl:'min-p 0.1', chips:[chip('100%', 'ok'), chip('96 distinct', 'ok')]}],
    msg:{zh:'min-p 問的是「這個 token 至少值最佳 token 的十分之一嗎」。模型有把握的時候自動收緊、沒把握的時候自動放寬，而且實作上只是一次比較——不用排序、不用前綴和。在這張表上它同時拿到 100% 與最多的句子數。',
         en:'min-p asks whether a token is worth at least a tenth of the best one. It tightens by itself when the model is sure and loosens when it is not, and it costs one comparison per token - no sort, no cumulative sum. On this table it takes both columns: 100% grammatical and the widest variety on the page.'}});
  p({upto:RESCUE.length - 1, line:6,
    cap:{zh:'第四條規則：typical sampling', en:'a fourth rule: typical sampling'},
    panels:[{lbl:'typical 0.9', chips:[chip(Math.round(RESCUE[7].g * 100) + '%', 'bad')]},
            {lbl:'typical 0.5', chips:[chip(Math.round(RESCUE[8].g * 100) + '%', 'ok'),
                                       chip(String(RESCUE[8].d), 'act')]}],
    msg:{zh:'typical sampling 問的是第三種問題（下下個分頁會拆開講）：留下「意外程度最接近這一步熵值」的 token。在這張表上它跟同樣 mass 的 top-p 幾乎一模一樣——0.9 同樣救不回來、0.5 同樣回到 9 成以上。這個「看不出差別」本身就是結果：24 個字元的詞彙表不夠大到讓兩條規則分開。',
         en:'Typical sampling asks a third kind of question (unpacked two tabs on): keep the tokens whose surprise is closest to this step\'s entropy. On this table it tracks top-p at the same mass almost exactly - 0.9 fails the same way, 0.5 recovers the same way. That lack of a difference is itself the result: a 24-character vocabulary is not big enough for the two rules to separate.'}});
  return F.list;
}

/* ================================================================ tab 4: shape */
const CODE_SHAPE = [
  '# the same two rules on two kinds of step',
  'certain = softmax(model("the small cat sa"))   # one token at 0.56',
  'unsure  = softmax(model("<start> the "))       # four tokens near 0.2',
  '',
  'nucleus_size(certain, 0.9)   # 2',
  'nucleus_size(unsure,  0.9)   # 5',
  '',
  '# top-k 3 on the unsure step keeps only 0.667 of the mass'
];
function shapeStage(o){
  const sh = [];
  if (o.upto >= 3){
    sh.push(fitT(4.9, 1.30, {zh:'一個刻意造出來的分佈：熵 = ' + TYPC.H + ' nats',
                             en:'a distribution built to separate them: entropy ' + TYPC.H + ' nats'},
                 {c:COL.tealL, fs:.30}));
    TYPC.rows.forEach((r, i) => {
      const y = 4.10 - i * .95;
      sh.push(box(.70, y, 3.30, .58, i === 1 ? 'hot' : 'idle', r.name, .25));
      sh.push(fitT(4.40, y + .29, {zh:'意外程度 ' + r.s.toFixed(2),
                                   en:'surprise ' + r.s.toFixed(2)},
                   {anchor:'start', c:COL.pale, fs:.27}));
      sh.push(fitT(9.20, y + .29, {zh:'離熵 ' + r.d.toFixed(2), en:'distance ' + r.d.toFixed(2)},
                   {anchor:'end', c:i === 1 ? '#ffbe6b' : COL.grey, fs:.27}));
    });
    return caps(sh, o);
  }
  const cols = [[SHAPE.peaked, {zh:'模型很有把握', en:'the model is certain'}, .75],
                [SHAPE.flat, {zh:'模型沒把握', en:'the model is unsure'}, 5.35]];
  cols.forEach((c, ci) => {
    const d = c[0], x0 = c[2];
    sh.push(fitT(x0 + 1.85, 1.10, c[1], {c:COL.tealL, fs:.28}));
    const keep = d.probs.map((_, i) => o.upto === 0 ? i < 3
                                     : o.upto === 1 ? i < d.nuc
                                     : i < d.mp);
    probBars(sh, x0, 4.15, d.probs, d.labs, keep, 2.25, .40, .14);
    const note = o.upto === 0 ? {zh:'top-k 3 留 3 個，質量 ' + d.k3.toFixed(3),
                                 en:'top-k 3 keeps 3, mass ' + d.k3.toFixed(3)}
               : o.upto === 1 ? {zh:'top-p 0.9 留 ' + d.nuc + ' 個',
                                 en:'top-p 0.9 keeps ' + d.nuc}
               : {zh:'min-p 0.1 留 ' + d.mp + ' 個', en:'min-p 0.1 keeps ' + d.mp};
    sh.push(fitT(x0 + 1.85, 4.85, note, {c:COL.pale, fs:.27}));
  });
  return caps(sh, o);
}
function shapeFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:shapeStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'同一個 k，兩種分佈', en:'one k, two distributions'},
    panels:[{lbl:'certain, k=3 mass', chips:[chip(SHAPE.peaked.k3.toFixed(3), 'ok')]},
            {lbl:'unsure, k=3 mass', chips:[chip(SHAPE.flat.k3.toFixed(3), 'bad')]}],
    msg:{zh:'左邊那一步只有兩個合理的字元，k=3 等於多放了一個垃圾進來；右邊那一步有四個差不多的字元，k=3 卻把第四名砍掉、只留下 ' + SHAPE.flat.k3.toFixed(3) + ' 的質量。同一個 k，兩邊都錯。',
         en:'On the left there are only two sensible characters, so k=3 admits one piece of junk. On the right four characters are nearly tied, and k=3 cuts the fourth, keeping only ' + SHAPE.flat.k3.toFixed(3) + ' of the mass. One k, wrong on both sides.'}});
  p({upto:0, line:7,
    cap:{zh:'這不是特例：整份語料都在這兩種之間跳',
         en:'and this is not a special case: the whole corpus alternates'},
    panels:[{lbl:'peaked steps', chips:[chip(String(SHAPE.nPeaked), 'dim')]},
            {lbl:'flat steps', chips:[chip(String(SHAPE.nFlat), 'dim')]},
            {lbl:'k=1 throws away', chips:[chip((SHAPE.k1Peaked * 100).toFixed(2) + '%', 'ok'),
                                           chip((SHAPE.k1Flat * 100).toFixed(0) + '%', 'bad')]}],
    msg:{zh:'掃過整份語料：' + SHAPE.nPeaked + ' 步是「有把握」（最高機率 > 90%）、' + SHAPE.nFlat + ' 步是「沒把握」。k=1 在前者丟掉 ' + (SHAPE.k1Peaked * 100).toFixed(2) + '%（純粹是平滑的雜訊），在後者丟掉 ' + (SHAPE.k1Flat * 100).toFixed(0) + '%。一個句子裡這兩種狀態會來回切換很多次。',
         en:'Sweeping the corpus: ' + SHAPE.nPeaked + ' steps are peaked (top probability above 90%) and ' + SHAPE.nFlat + ' are flat. k=1 throws away ' + (SHAPE.k1Peaked * 100).toFixed(2) + '% on the first kind - pure smoothing noise - and ' + (SHAPE.k1Flat * 100).toFixed(0) + '% on the second. A single sentence switches between the two many times.'}});
  p({upto:1, line:5,
    cap:{zh:'top-p 要的是質量，所以數量自己跟著變',
         en:'top-p asks for a mass, so the count follows the step'},
    panels:[{lbl:'nucleus, certain', chips:[chip(String(SHAPE.peaked.nuc), 'ok')]},
            {lbl:'nucleus, unsure', chips:[chip(String(SHAPE.flat.nuc), 'ok')]}],
    msg:{zh:'同一個 p=0.9：有把握的那一步只留 ' + SHAPE.peaked.nuc + ' 個，沒把握的那一步留 ' + SHAPE.flat.nuc + ' 個。平均起來是 ' + SHAPE.nucPeaked.toFixed(2) + ' 對 ' + SHAPE.nucFlat.toFixed(2) + '——這就是 nucleus sampling 這個名字的意思：核心有多大由分佈自己決定。',
         en:'One setting, p=0.9: the certain step keeps ' + SHAPE.peaked.nuc + ' candidates and the unsure step keeps ' + SHAPE.flat.nuc + '. Averaged over the corpus that is ' + SHAPE.nucPeaked.toFixed(2) + ' against ' + SHAPE.nucFlat.toFixed(2) + '. That is what nucleus sampling means: the size of the nucleus is decided by the distribution, not by you.'}});
  p({upto:2, line:7,
    cap:{zh:'min-p：只要跟第一名比', en:'min-p: compare against the leader only'},
    panels:[{lbl:'min-p, certain', chips:[chip(String(SHAPE.peaked.mp), 'ok')]},
            {lbl:'min-p, unsure', chips:[chip(String(SHAPE.flat.mp), 'ok')]}],
    msg:{zh:'min-p 連累積和都不用算：每個 token 跟 0.1×最大值比一次就好。它在有把握的那一步收到 ' + SHAPE.peaked.mp + ' 個、沒把握的那一步放到 ' + SHAPE.flat.mp + ' 個，形狀上跟 top-p 同向，但代價是一次比較。',
         en:'min-p does not even need a cumulative sum: compare each token once against 0.1 times the maximum. It tightens to ' + SHAPE.peaked.mp + ' on the certain step and opens to ' + SHAPE.flat.mp + ' on the unsure one - the same direction as top-p, for the price of one comparison.'}});
  p({upto:3, line:7,
    cap:{zh:'而 typical sampling 問的是第三種問題', en:'and typical sampling asks a third kind of question'},
    panels:[{lbl:'same as top-p on', chips:[chip(TYPC.same + ' / ' + TYPC.steps + ' steps', 'dim')]},
            {lbl:'here, top-p keeps', chips:[chip(String(TYPC.topPn), 'act')]},
            {lbl:'typical keeps', chips:[chip(String(TYPC.typN), 'hot')]}],
    msg:{zh:'typical sampling 留下「意外程度 −log p 最接近這一步熵值」的 token——背後的主張是自然語言每個詞攜帶的資訊量大致固定，所以「比平均還要理所當然」的 token 跟「比平均還要意外」的一樣不典型。在我們的語料上它跟 top-p ' + TYPC.steps.toLocaleString() + ' 步完全沒分開過（詞彙表只有 24 個字元），所以這裡刻意造一個會分開的分佈：要 ' + TYPC.mass + ' 的質量時，top-p 留 ' + TYPC.topPn + ' 個（那個 0.60 的），typical 留 ' + TYPC.typN + ' 個而且**把最可能的那個丟掉**——因為中間那一組才是資訊量典型的那一組。',
         en:'Typical sampling keeps the tokens whose surprise, -log p, is closest to this step\'s entropy. The claim behind it is that natural language carries a roughly constant amount of information per word, so a token far more obvious than average is as atypical as one far less. On our corpus it never separated from top-p in ' + TYPC.steps.toLocaleString() + ' steps - 24 characters is too small - so this distribution is built to separate them: asked for ' + TYPC.mass + ' of the mass, top-p keeps ' + TYPC.topPn + ' token (the 0.60 one) and typical keeps ' + TYPC.typN + ', dropping the most likely token entirely, because the middle group is the one carrying a typical amount of information.'}});
  return F.list;
}

/* ================================================================ tab 5: order + the GPU */
const CODE_ORDER = [
  '# the order every server uses:',
  'logits = penalties(logits, seen_tokens)',
  'logits = logits / temperature',
  'p = softmax(logits)',
  'p = top_k(p, k)',
  'p = top_p(p, thr)              # the nucleus is cut on the HOT shape',
  '',
  '# sort-free top-p, which is what the GPU kernel does:',
  'lo, hi = 0.0, p.max()',
  'for _ in range(24):            # 24 reductions, no ordering',
  '    mid = (lo + hi) / 2',
  '    mass = p[p >= mid].sum()',
  '    lo, hi = (mid, hi) if mass >= thr else (lo, mid)'
];
function orderStage(o){
  const sh = [];
  if (o.upto <= 1){
    const rows = [[{zh:'先 temperature，再 top-p（實際採用）',
                    en:'temperature first, then top-p (what servers do)'}, ORDER.n_a, ORDER.max_a],
                  [{zh:'先 top-p，再 temperature', en:'top-p first, then temperature'},
                   ORDER.n_b, ORDER.max_b]];
    rows.forEach((r, i) => {
      if (i > o.upto) return;
      const y = 4.10 - i * 1.45;
      sh.push(fitT(.70, y + .62, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
      for (let j = 0; j < Math.min(r[1], 20); j++)
        sh.push(S.r(.70 + j * .42, y, .34, .46, j === 0 ? 'hot' : 'act', null));
      sh.push(fitT(9.10, y + .23, r[1] + (r[1] > 1 ? ' tokens' : ' token'),
                   {anchor:'end', c:COL.pale, fs:.28}));
    });
    sh.push(fitT(4.9, 1.05, {zh:'同一個 context、同一組參數（t=' + ORDER.t + '、p=' + ORDER.p + '）',
                             en:'same context, same parameters (t=' + ORDER.t + ', p=' + ORDER.p + ')'},
                 {c:COL.grey, fs:.26}));
  } else {
    const st = PIVOT.steps[Math.min(o.upto - 2, PIVOT.steps.length - 1)];
    probBars(sh, 1.55, 4.20, PIVOT.probs, PIVOT.labs,
             PIVOT.probs.map(v => v >= st.mid), 2.45, .66, .32);
    const hi = Math.max.apply(null, PIVOT.probs), yline = 4.20 - 2.45 * st.mid / hi;
    sh.push(S.e(1.25, yline, 8.55, yline, {s:'bad', arrow:false, w:.05}));
    sh.push(fitT(8.55, yline - .20, 'pivot ' + st.mid.toFixed(3),
                 {anchor:'end', c:COL.red, fs:.24}));
    sh.push(fitT(4.9, 1.05, {zh:'門檻之上的質量 ' + st.mass.toFixed(4) + '，共 ' + st.n + ' 個 token',
                             en:'mass above the threshold ' + st.mass.toFixed(4) + ', ' + st.n + ' tokens'},
                 {c:COL.pale, fs:.28}));
  }
  return caps(sh, o);
}
function orderFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:orderStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:3,
    cap:{zh:'先加熱，再切 nucleus', en:'heat first, then cut the nucleus'},
    panels:[{lbl:'candidates', chips:[chip(String(ORDER.n_a), 'hot')]},
            {lbl:'top probability', chips:[chip(ORDER.max_a.toFixed(3), 'dim')]}],
    msg:{zh:'temperature 先把分佈壓平，top-p 才去數質量——壓平之後要湊到 0.9 就得收更多 token，所以 nucleus 變成 ' + ORDER.n_a + ' 個。這是 vLLM 和 SGLang 實際採用的順序。',
         en:'Temperature flattens first and only then does top-p count mass. On a flattened distribution it takes more tokens to reach 0.9, so the nucleus widens to ' + ORDER.n_a + '. This is the order vLLM and SGLang actually use.'}});
  p({upto:1, line:6,
    cap:{zh:'反過來，差很多', en:'the other way round is a different distribution'},
    panels:[{lbl:'candidates', chips:[chip(String(ORDER.n_b), 'ok')]},
            {lbl:'total variation', chips:[chip(ORDER.tv.toFixed(3), 'bad')]}],
    msg:{zh:'先切再加熱的話，候選集合是用「冷的」形狀決定的，只有 ' + ORDER.n_b + ' 個，加熱只是在這兩個之間重新分配。兩者的 total variation distance 是 ' + ORDER.tv.toFixed(3) + '——不是小數點後的差別，是不同的分佈。所以 API 文件沒寫順序時，同樣的參數在兩個服務上可以給出不同的行為。',
         en:'Cut first and the candidate set is decided on the cold shape - only ' + ORDER.n_b + ' tokens - and the heat merely redistributes between them. The two results sit ' + ORDER.tv.toFixed(3) + ' apart in total variation: not a rounding difference, a different distribution. When an API does not document the order, the same parameters can behave differently on two services.'}});
  for (let i = 0; i < 4; i++){
    const st = PIVOT.steps[i];
    p({upto:2 + i, line:11,
      cap:{zh:'而在 GPU 上，top-p 根本不排序', en:'and on a GPU, top-p does not sort at all'},
      panels:[{lbl:'iteration', chips:[chip(String(i + 1) + ' / 24', 'dim')]},
              {lbl:'mass kept', chips:[chip(st.mass.toFixed(4), st.mass >= .9 ? 'ok' : 'bad')]},
              {lbl:'tokens', chips:[chip(String(st.n), 'act')]}],
      msg:{zh:'把「留哪些 token」換個問法：找一個機率門檻，讓門檻以上的質量剛好到 0.9。這是一維的單調問題，可以二分搜尋——每一輪只是「比較 ＋ 加總」，' + COST.V.toLocaleString() + ' 個 token 的 reduction，完全不用排序。這一輪門檻 ' + st.mid.toFixed(3) + '、質量 ' + st.mass.toFixed(4) + (st.mass >= .9 ? '（夠，門檻往上）' : '（不夠，門檻往下）'),
           en:'Ask the question differently: find a probability threshold such that the mass above it just reaches 0.9. That is one-dimensional and monotone, so it can be bisected - each round is a compare-and-sum reduction over ' + COST.V.toLocaleString() + ' tokens, with no ordering anywhere. This round: threshold ' + st.mid.toFixed(3) + ', mass ' + st.mass.toFixed(4) + (st.mass >= .9 ? ' - enough, raise the bar.' : ' - not enough, lower it.')}});
  }
  p({upto:2 + PIVOT.steps.length - 1, line:12,
    cap:{zh:'24 輪之後，跟排序版答案一樣', en:'after 24 rounds, the same answer as the sorted version'},
    panels:[{lbl:'sort-free', chips:[chip(PIVOT.n + ' tokens', 'ok')]},
            {lbl:'sorted', chips:[chip(PIVOT.sorted + ' tokens', 'ok')]},
            {lbl:'full sort costs', chips:[chip(COST.ratio + 'x more', 'bad')]}],
    msg:{zh:'兩邊都是 ' + PIVOT.n + ' 個 token、質量 ' + PIVOT.mass.toFixed(4) + '。為什麼要這麼麻煩：真實詞彙表 ' + COST.V.toLocaleString() + ' 個 token，而取樣是「每個序列每一步」都要做一次。完整排序約 ' + COST.full + 'M 次比較，部分選擇約 ' + COST.partial + 'M，差 ' + COST.ratio + ' 倍——這就是 SGLang 的 top_p_renorm 和 vLLM 的 FlashInfer sampling kernel 在做的事。',
         en:'Both give ' + PIVOT.n + ' tokens holding ' + PIVOT.mass.toFixed(4) + '. Why bother: a real vocabulary is ' + COST.V.toLocaleString() + ' tokens and sampling runs once per sequence per decode step. A full sort is about ' + COST.full + 'M comparisons against ' + COST.partial + 'M for a partial selection, ' + COST.ratio + 'x apart. This is what SGLang\'s top_p_renorm and vLLM\'s FlashInfer sampling kernels do.'}});
  return F.list;
}

/* ================================================================ tabs 6-7: the servers */
function mapStage(o){
  const sh = [], rows = o.rows;
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.30 - i * .80;
    sh.push(box(.70, y, 3.30, .58, 'idle', r[0], .26));
    sh.push(S.e(4.15, y + .29, 4.75, y + .29, {s:'ghost', arrow:true, w:.05}));
    sh.push(box(4.90, y, 4.30, .58, i === o.upto ? 'hot' : 'ok', r[1], .26));
  });
  sh.push(fitT(2.35, 5.08, {zh:'這個檔案裡的東西', en:'what this file calls it'},
               {c:COL.grey, fs:.26}));
  sh.push(fitT(7.05, 5.08, o.title, {c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
const CODE_VLLM = [
  'from vllm import LLM, SamplingParams',
  '',
  'out = LLM("Qwen/Qwen3-8B").generate(prompts, SamplingParams(',
  '    temperature=0.8,        # tab 2',
  '    top_k=50, top_p=0.95,   # tabs 3 and 4, applied in that order',
  '    min_p=0.0,',
  '    repetition_penalty=1.1, # applied BEFORE temperature',
  '    seed=1234,              # per-request generator: a retry reproduces',
  '    n=4, logprobs=5, max_tokens=256))'
];
const VLLM_ROWS = [
  [{zh:'temperature(logits, t)', en:'temperature(logits, t)'},
   {zh:'temperature（0 ＝ 走 argmax）', en:'temperature (0 takes the argmax path)'}],
  [{zh:'top_k(p, k) / top_p(p, thr)', en:'top_k(p, k) / top_p(p, thr)'},
   {zh:'top_k, top_p, min_p', en:'top_k, top_p, min_p'}],
  [{zh:'第 7 節的先後順序', en:'the order in section 7'},
   {zh:'penalties → temperature → top-k → top-p', en:'penalties -> temperature -> top-k -> top-p'}],
  [{zh:'rng.choice(...)', en:'rng.choice(...)'}, {zh:'seed＝每個請求自己的亂數產生器',
   en:'seed: a per-request generator'}],
  [{zh:'第 2 節印出來的那幾列', en:'the rows printed in section 2'},
   {zh:'logprobs=5', en:'logprobs=5'}]
];
function vllmFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:VLLM_ROWS, title:'vLLM'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:4,
    cap:{zh:'vLLM：整組旋鈕就是一個物件', en:'vLLM: the whole set of knobs is one object'},
    panels:[{lbl:'temperature', chips:[chip('0.8', 'hot')]},
            {lbl:'top_k / top_p', chips:[chip('50', 'act'), chip('0.95', 'act')]}],
    msg:{zh:'`SamplingParams` 是 per-request 的，但工作是 per-step 的：排程器把當下所有在跑的序列塞進同一個 sampling kernel，各自帶著自己的 temperature 和 top-p。所以每個請求可以設不同的參數，而成本還是一次 kernel。',
         en:'`SamplingParams` is per request, but the work is per step: the scheduler packs every running sequence into one sampling kernel, each carrying its own temperature and top-p. Different parameters per request, still one kernel.'}});
  p({upto:3, line:7,
    cap:{zh:'順序是寫死的，而且就是第 7 節那個順序',
         en:'the order is fixed, and it is the one from section 7'},
    panels:[{lbl:'order', chips:[chip('penalties', 'dim'), chip('temperature', 'hot'),
                                 chip('top-k', 'act'), chip('top-p', 'act')]}],
    msg:{zh:'`repetition_penalty` 在 temperature 之前動 logits（對已經出現過的 token 除以 1.1），然後才是加熱與截斷。`seed` 給的是每個請求自己的產生器——同樣的 seed 重送一次會得到同一串輸出，這在重現 bug 的時候是唯一能抓的東西。',
         en:'`repetition_penalty` edits the logits before temperature (dividing the ones already seen by 1.1), and only then comes the heat and the truncation. `seed` gives the request its own generator, so resending with the same seed reproduces the output exactly - usually the only handle you have when reproducing a bug.'}});
  p({upto:4, line:8,
    cap:{zh:'n=4 和 logprobs 都是前面幾天的東西',
         en:'n=4 and logprobs are both earlier days showing up'},
    panels:[{lbl:'n=4', chips:[chip('shares prompt blocks', 'ok')]},
            {lbl:'logprobs=5', chips:[chip('the top rows of tab 1', 'dim')]}],
    msg:{zh:'`n=4` 是同一個 prompt 抽四次——KV 的部分在前一天已經解決了（refcount ＋ copy-on-write），這一天只是各自丟四顆不同的骰子。`logprobs=5` 回傳的就是第一個分頁那張圖的前五列。',
         en:'`n=4` draws four samples from one prompt - the KV side was solved the previous day with reference counts and copy-on-write, and today only adds four different dice. `logprobs=5` returns the top five rows of the chart in tab one.'}});
  return F.list;
}
const CODE_SGLANG = [
  'curl localhost:30000/generate -d \'{',
  '  "text": "...",',
  '  "sampling_params": {"temperature": 0.8, "top_k": 50, "top_p": 0.95,',
  '                      "min_p": 0.0, "repetition_penalty": 1.1,',
  '                      "frequency_penalty": 0.2, "n": 4}}\'',
  '',
  '# server side:',
  'python -m sglang.launch_server --model ... \\',
  '    --sampling-backend flashinfer      # fused top-k / top-p kernels',
  '    --sampling-backend pytorch         # the sort-based reference path'
];
const SGLANG_ROWS = [
  [{zh:'top_p(p, thr)', en:'top_p(p, thr)'}, {zh:'top_p_renorm（無排序）', en:'top_p_renorm (sort-free)'}],
  [{zh:'pivot_search()', en:'pivot_search()'}, {zh:'flashinfer 的門檻搜尋 kernel', en:'the FlashInfer threshold kernel'}],
  [{zh:'把不要的 token 設成 0', en:'zeroing the tokens you do not want'},
   {zh:'custom logit processor', en:'a custom logit processor'}],
  [{zh:'文法檢查（本日的評分方式）', en:'the grammar check used to score today'},
   {zh:'grammar-constrained decoding', en:'grammar-constrained decoding'}]
];
function sglangFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:SGLANG_ROWS, title:'SGLang'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:8,
    cap:{zh:'SGLang：同樣的旋鈕，重點在 kernel', en:'SGLang: the same knobs, the interest is in the kernel'},
    panels:[{lbl:'--sampling-backend', chips:[chip('flashinfer', 'hot'), chip('pytorch', 'dim')]}],
    msg:{zh:'兩個 backend 算的是同一個分佈，差在怎麼算：pytorch 那條路會真的排序，flashinfer 那條走的是上一個分頁的門檻搜尋。在高併發 decode 下，取樣是每一步都要做的事，這個差別會直接進到 ITL。',
         en:'The two backends compute the same distribution and differ in how: the pytorch path really sorts, the flashinfer path runs the threshold search from the previous tab. Under high-concurrency decode, sampling happens every step, so the difference lands directly in inter-token latency.'}});
  p({upto:3, line:3,
    cap:{zh:'而「把機率設成 0」是一個公開的掛鉤',
         en:'and "set this probability to zero" is a public hook'},
    panels:[{lbl:'logit processor', chips:[chip('mask before temperature', 'act')]},
            {lbl:'uses', chips:[chip('JSON schema', 'ok'), chip('banned phrases', 'ok')]}],
    msg:{zh:'今天所有的截斷規則本質上都是「在 softmax 之前把某些 logit 設成 −inf」。把條件從「機率太小」換成「這個 token 會讓 JSON 不合法」，就是 structured output；換成「這個詞被禁用」，就是內容過濾。同一個掛鉤，差別只在那個判斷式。',
         en:'Every truncation rule today is the same operation: set some logits to minus infinity before the softmax. Change the condition from "too unlikely" to "this token would make the JSON invalid" and you have structured output; change it to "this phrase is banned" and you have content filtering. One hook, different predicates.'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'取樣策略：temperature、top-k、top-p', en:'Temperature, top-k and top-p'},
  sub: {zh:'一排 logits 要變成一個 token：形狀怎麼改、尾巴怎麼剪、順序為什麼會改變答案',
        en:'turning a row of logits into one token: reshaping, truncating, and why the order changes the answer'},
  tabs: [
    {id:'tail', label:{zh:'尾巴的問題', en:'the problem is the tail'},
     stage:{zh:'一步看起來很乾淨，一句話就不是了', en:'one step looks clean, a sentence does not'},
     view:VIEW, code:CODE_TAIL, build:tailFrames,
     idea:{zh:'這個 context 下前三名吃掉 99.7%，其他 ' + STEP.rest + ' 個字元只有 ' + (STEP.tail * 100).toFixed(2) + '%。但一句話要走 30 步，至少踩到一次尾巴的機率是 ' + (STEP.risk30 * 100).toFixed(1) + '%——而反過來永遠取最大值的 greedy，' + GREEDY.total + ' 句合法句子裡只會說 1 句。',
           en:'The top three characters hold 99.7% here and the other ' + STEP.rest + ' hold ' + (STEP.tail * 100).toFixed(2) + '%. But a sentence is 30 steps, so the chance of stepping into that tail at least once is ' + (STEP.risk30 * 100).toFixed(1) + '%. The opposite extreme, greedy, says exactly 1 of the ' + GREEDY.total + ' legal sentences.'},
     legend:[['#ff9736', {zh:'被保留的 token', en:'tokens kept'}],
             ['#ff5c5c', {zh:'尾巴', en:'the tail'}],
             ['#8fa3ac', {zh:'被切掉的', en:'cut'}]]},
    {id:'seq', label:{zh:'greedy 不是最可能的句子', en:'greedy is not the likeliest sentence'},
     stage:{zh:'逐步最大 ≠ 整句最大', en:'best at each step is not best overall'},
     view:VIEW, code:CODE_SEQ, build:seqFrames,
     idea:{zh:'文法只有 ' + SEQ.n.toLocaleString() + ' 句，全部精確算完之後可以直接檢查：greedy 產生的句子排第 ' + SEQ.rank + '，比最可能的那句低 ' + SEQ.ratio + ' 倍。逐步 argmax 會走進「之後沒有好結尾」的前綴而且回不去，這是序列搜尋（beam search 那一類）存在的理由。順帶兩個驗證：合法句子佔 ' + (SEQ.legalMass * 100).toFixed(1) + '% 的句子層級機率（實測合文法率 90%），抽 100 句的不重複期望 ' + SEQ.expDistinct.toFixed(1) + '（實測 88）。',
           en:'With only ' + SEQ.n.toLocaleString() + ' legal sentences every one can be scored exactly, and then checked: the greedy sentence ranks ' + SEQ.rank + ', ' + SEQ.ratio + 'x less likely than the best. Step-wise argmax walks into a prefix with no good ending and cannot back out, which is the reason sequence search exists. Two side checks: the legal sentences hold ' + (SEQ.legalMass * 100).toFixed(1) + '% of the sentence-level probability (measured grammatical rate 90%) and 100 draws should give ' + SEQ.expDistinct.toFixed(1) + ' distinct sentences (measured 88).'},
     legend:[['#12b3b8', {zh:'整句機率最高的', en:'the most likely sentence'}],
             ['#ff9736', {zh:'greedy 產生的', en:'what greedy produces'}],
             ['#ff5c5c', {zh:'它的排名', en:'where it ranks'}]]},
    {id:'temp', label:{zh:'temperature', en:'temperature'},
     stage:{zh:'多樣性買得到，正確性要付錢', en:'variety bought, correctness paid'},
     view:VIEW, code:CODE_TEMP, build:tempFrames,
     idea:{zh:'logits 除以 t 再 softmax。t=0.7 時 100 個樣本有 98 句不同而且全部合文法；t=1.7 只剩 5%。entropy 只翻一倍，正確率卻從 90% 掉到 5%——而且多樣性也跟著掉，因為壞掉的句子不算另一種說法。',
           en:'Divide the logits by t, then softmax. At t=0.7, 100 samples give 98 distinct sentences and all of them parse; at t=1.7 only 5% do. The entropy merely doubles while correctness falls from 90% to 5% - and variety falls with it, because a broken sentence is not a different sentence.'},
     legend:[['#12b3b8', {zh:'文法正確的比例', en:'grammatical rate'}],
             ['#9d6bff', {zh:'不重複的句子數', en:'distinct sentences'}]]},
    {id:'trunc', label:{zh:'截斷：top-k / top-p / min-p', en:'truncation: top-k, top-p, min-p'},
     stage:{zh:'溫度不動，只改「誰可以被抽到」', en:'same heat, different candidate set'},
     view:VIEW, code:CODE_TRUNC, build:truncFrames,
     idea:{zh:'全部固定在 t=2.0（0% 合文法），只加一條截斷規則：top-k 3 仍是 0%、top-p 0.9 只有 2%、top-p 0.5 回到 100%、min-p 0.1 回到 100% 而且有 96 句不同。隨機性沒被拿掉，被拿掉的是本來就不可能對的那一部分。',
           en:'Everything is pinned at t=2.0 (0% grammatical) and one truncation rule is added: top-k 3 stays at 0%, top-p 0.9 reaches 2%, top-p 0.5 returns to 100%, and min-p 0.1 returns to 100% with 96 distinct sentences. The randomness was not removed - the part of it that could never have been right was.'},
     legend:[['#12b3b8', {zh:'救回來了', en:'recovered'}],
             ['#ff5c5c', {zh:'沒救回來', en:'still broken'}],
             ['#9d6bff', {zh:'不重複的句子數', en:'distinct sentences'}]]},
    {id:'shape', label:{zh:'為什麼不是固定數量', en:'why not a fixed count'},
     stage:{zh:'有把握的一步，和沒把握的一步', en:'a certain step and an unsure one'},
     view:VIEW, code:CODE_SHAPE, build:shapeFrames,
     idea:{zh:'整份語料裡 ' + SHAPE.nPeaked + ' 步是有把握的、' + SHAPE.nFlat + ' 步不是。k=1 在前者只丟掉 ' + (SHAPE.k1Peaked * 100).toFixed(2) + '%（平滑的雜訊），在後者丟掉 ' + (SHAPE.k1Flat * 100).toFixed(0) + '%。top-p 要的是質量不是數量，所以 nucleus 在兩種步上自己變成 ' + SHAPE.peaked.nuc + ' 和 ' + SHAPE.flat.nuc + ' 個。',
           en:'Across the corpus ' + SHAPE.nPeaked + ' steps are peaked and ' + SHAPE.nFlat + ' are not. k=1 throws away ' + (SHAPE.k1Peaked * 100).toFixed(2) + '% on the first kind (smoothing noise) and ' + (SHAPE.k1Flat * 100).toFixed(0) + '% on the second. top-p asks for mass rather than count, so the nucleus becomes ' + SHAPE.peaked.nuc + ' and ' + SHAPE.flat.nuc + ' by itself.'},
     legend:[['#ff9736', {zh:'留下來的', en:'kept'}],
             ['#8fa3ac', {zh:'切掉的', en:'cut'}]]},
    {id:'order', label:{zh:'順序，以及 GPU 怎麼做', en:'order, and how a GPU does it'},
     stage:{zh:'先加熱還是先截斷，不是同一件事', en:'heat then cut is not cut then heat'},
     view:VIEW, code:CODE_ORDER, build:orderFrames,
     idea:{zh:'同一個 context、同一組 t=' + ORDER.t + '／p=' + ORDER.p + '：先加熱再切得到 ' + ORDER.n_a + ' 個候選，先切再加熱只有 ' + ORDER.n_b + ' 個，total variation 差 ' + ORDER.tv.toFixed(3) + '。伺服器一律用前者。另一半：top-p 在 GPU 上是二分搜尋一個機率門檻（24 輪 reduction），不排序就得到跟排序版一樣的 ' + PIVOT.n + ' 個 token。',
           en:'Same context, same t=' + ORDER.t + ' and p=' + ORDER.p + ': heating first leaves ' + ORDER.n_a + ' candidates, truncating first leaves ' + ORDER.n_b + ', and the two distributions are ' + ORDER.tv.toFixed(3) + ' apart in total variation. Servers use the former. The other half of the tab: on a GPU top-p is a bisection on a probability threshold - 24 reductions, no ordering, the same ' + PIVOT.n + ' tokens the sorted version returns.'},
     legend:[['#9d6bff', {zh:'候選 token', en:'candidate tokens'}],
             ['#ff5c5c', {zh:'目前的門檻', en:'the current threshold'}],
             ['#ff9736', {zh:'門檻之上', en:'above the threshold'}]]},
    {id:'vllm', label:{zh:'在 vLLM 裡', en:'in vLLM'},
     stage:{zh:'SamplingParams 的每一個欄位', en:'every field of SamplingParams'},
     view:VIEW, code:CODE_VLLM, build:vllmFrames,
     idea:{zh:'`SamplingParams` 是 per-request 的，但 sampling 是 per-step 的一次 kernel，所有在跑的序列各自帶參數一起算。`repetition_penalty` 在 temperature 之前改 logits，`seed` 讓同一個請求可以重現，`n=4` 的 KV 共用是前一天的 refcount。',
           en:'`SamplingParams` is per request while sampling is one kernel per step, with every running sequence carrying its own parameters. `repetition_penalty` edits the logits before temperature, `seed` makes a request reproducible, and the KV sharing behind `n=4` is the previous day\'s reference count.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'vLLM 裡的名字', en:'what vLLM calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'sglang', label:{zh:'在 SGLang 裡', en:'in SGLang'},
     stage:{zh:'同樣的分佈，不同的算法', en:'the same distribution, computed differently'},
     view:VIEW, code:CODE_SGLANG, build:sglangFrames,
     idea:{zh:'`--sampling-backend flashinfer` 走的是無排序的門檻搜尋，`pytorch` 走真的排序——同一個分佈、不同的成本，而且因為 decode 每一步都要取樣，這個差別會進到 ITL。另外，「把某些 logit 設成 −inf」是公開掛鉤：換個判斷式就是 JSON schema 或禁用詞。',
           en:'`--sampling-backend flashinfer` runs the sort-free threshold search and `pytorch` really sorts - the same distribution at different cost, and since decode samples on every step the difference reaches inter-token latency. Separately, "set these logits to minus infinity" is a public hook: change the predicate and you get JSON schema enforcement or banned phrases.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'SGLang 裡的名字', en:'what SGLang calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]}
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
