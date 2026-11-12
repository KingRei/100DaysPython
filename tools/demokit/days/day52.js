// DAY: 52
// TITLE_ZH: Attention 從零實作：一個加權平均，以及所有人都在改的那四個地方
// TITLE_EN: Attention from scratch: one weighted average, and the four things everyone changes about it
// SUB_ZH: Attention 就是一個加權平均，權重來自對相似度分數做 softmax。難的全部不在那條式子，而在「怎麼算出這個平均，又不要真的把它平均的那個東西生出來」——溢位、為什麼要除 sqrt(d)、mask 要放在 softmax 之前、n² 的記憶體帳單、FlashAttention、paged KV、MQA/GQA/MLA，以及 delta rule 這種固定大小狀態的線性注意力。
// SUB_EN: Attention is a weighted average whose weights come from a softmax over similarity scores. Nothing hard lives in that formula; it lives in computing the average without building the thing it averages over - overflow, the sqrt(d), masking before the softmax rather than after, the quadratic memory bill, FlashAttention, paged KV, MQA/GQA/MLA, and the fixed-size state of delta-rule linear attention.
// FOLDER: day%2052%20-%20attention%20from%20scratch
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
const W = {"free": [[0.0504, 0.0099, 0.0127, 0.2369, 0.4417, 0.2484], [0.2483, 0.2565, 0.1878, 0.0799, 0.0467, 0.1808], [0.0493, 0.0053, 0.0072, 0.0193, 0.0352, 0.8837], [0.4893, 0.2591, 0.0435, 0.1356, 0.0429, 0.0297], [0.0574, 0.0441, 0.0256, 0.1508, 0.5481, 0.174], [0.1304, 0.1326, 0.0332, 0.1081, 0.4016, 0.1941]], "ok": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4919, 0.5081, 0.0, 0.0, 0.0, 0.0], [0.798, 0.0857, 0.1162, 0.0, 0.0, 0.0], [0.5276, 0.2793, 0.0469, 0.1462, 0.0, 0.0], [0.0695, 0.0534, 0.031, 0.1825, 0.6635, 0.0], [0.1304, 0.1326, 0.0332, 0.1081, 0.4016, 0.1941]], "bad": [[0.0504, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2483, 0.2565, 0.0, 0.0, 0.0, 0.0], [0.0493, 0.0053, 0.0072, 0.0, 0.0, 0.0], [0.4893, 0.2591, 0.0435, 0.1356, 0.0, 0.0], [0.0574, 0.0441, 0.0256, 0.1508, 0.5481, 0.0], [0.1304, 0.1326, 0.0332, 0.1081, 0.4016, 0.1941]], "mask": [[true, false, false, false, false, false], [true, true, false, false, false, false], [true, true, true, false, false, false], [true, true, true, true, false, false], [true, true, true, true, true, false], [true, true, true, true, true, true]], "okSum": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "badSum": [0.0504, 0.5048, 0.0618, 0.9274, 0.826, 1.0]};
const OVF = {"k32": 89, "k16": 12, "max16": 65504.0, "naive": ["nan", "nan", "nan"], "safe": [0.2447, 0.6652, 0.09]};
const SCALE = [{"d": 8, "std": 2.846, "sqrt": 2.83, "raw": 0.3221, "scaled": 0.0437}, {"d": 64, "std": 7.911, "sqrt": 8.0, "raw": 0.7067, "scaled": 0.0404}, {"d": 512, "std": 22.563, "sqrt": 22.63, "raw": 0.9048, "scaled": 0.0423}, {"d": 4096, "std": 64.109, "sqrt": 64.0, "raw": 0.9688, "scaled": 0.0428}];
const FLASH = [{"blk": 1, "err": 6.661e-16}, {"blk": 8, "err": 3.886e-16}, {"blk": 32, "err": 3.608e-16}, {"blk": 128, "err": 3.886e-16}, {"blk": 256, "err": 4.996e-16}];
const MEM = [{"ctx": 1024, "gb": 0.06}, {"ctx": 4096, "gb": 1.0}, {"ctx": 16384, "gb": 16.0}, {"ctx": 131072, "gb": 1024.0}];
const KV = {"rows": [{"name": "MHA", "bytes": 524288, "kb": 512, "gb": 64.0, "vs": 1.0}, {"name": "GQA", "bytes": 131072, "kb": 128, "gb": 16.0, "vs": 4.0}, {"name": "MLA", "bytes": 36864, "kb": 36, "gb": 4.5, "vs": 14.2}, {"name": "MQA", "bytes": 16384, "kb": 16, "gb": 2.0, "vs": 32.0}], "stateMB": 32};
const PAGE = {"lengths": [12, 300, 48, 900, 7, 2048, 64, 130], "cap": 4096, "slots": 32768, "used": 3509, "waste": 89.3, "paged": [{"page": 16, "pages": 222, "slots": 3552, "waste": 1.2}, {"page": 64, "pages": 59, "slots": 3776, "waste": 7.1}, {"page": 256, "pages": 19, "slots": 4864, "waste": 27.9}]};
const DELTA = {"rows": [{"n": 8, "err": 1e-12, "lost": 0}, {"n": 16, "err": 1e-12, "lost": 0}, {"n": 17, "err": 1.33, "lost": 1}, {"n": 24, "err": 1.69, "lost": 8}, {"n": 32, "err": 1.87, "lost": 16}], "smear": 1.66};
const LC239 = {"nums": [1, 3, -1, -3, 5, 3, 6, 7], "k": 3, "steps": [{"i": 0, "v": 1, "popped": [], "dq": [1], "out": [], "win": [0, 0]}, {"i": 1, "v": 3, "popped": [1], "dq": [3], "out": [], "win": [0, 1]}, {"i": 2, "v": -1, "popped": [], "dq": [3, -1], "out": [3], "win": [0, 2]}, {"i": 3, "v": -3, "popped": [], "dq": [3, -1, -3], "out": [3, 3], "win": [1, 3]}, {"i": 4, "v": 5, "popped": [3, -1, -3], "dq": [5], "out": [3, 3, 5], "win": [2, 4]}, {"i": 5, "v": 3, "popped": [], "dq": [5, 3], "out": [3, 3, 5, 5], "win": [3, 5]}, {"i": 6, "v": 6, "popped": [5, 3], "dq": [6], "out": [3, 3, 5, 5, 6], "win": [4, 6]}, {"i": 7, "v": 7, "popped": [6], "dq": [7], "out": [3, 3, 5, 5, 6, 7], "win": [5, 7]}], "ans": [3, 3, 5, 5, 6, 7]};
const LC703 = {"k": 3, "init": [4, 5, 8, 2], "steps": [{"v": 3, "act": "drop", "heap": [4, 5, 8], "root": 4}, {"v": 5, "act": "replace", "heap": [5, 5, 8], "root": 5}, {"v": 10, "act": "replace", "heap": [5, 8, 10], "root": 5}, {"v": 9, "act": "replace", "heap": [8, 9, 10], "root": 8}, {"v": 4, "act": "drop", "heap": [8, 9, 10], "root": 8}]};

function grid(sh, x0, y0, M, cw, mask, hotAt){
  M.forEach((row, i) => row.forEach((v, j) => {
    const allowed = !mask || mask[i][j];
    const st = !allowed ? 'done' : (hotAt && v > hotAt ? 'hot' : 'ok');
    sh.push(box(x0 + j * cw, y0 + (M.length - 1 - i) * cw, cw * .92, cw * .92, st,
                allowed ? v.toFixed(2) : '', .24));
  }));
  return sh;
}

/* ================================================================ tab 1: the weights */
const CODE_W = [
  'scores = Q @ K.T / sqrt(d)      # one number per pair',
  'scores = where(mask, scores, -inf)',
  'w      = softmax(scores)        # each row sums to 1',
  'out    = w @ V                  # a weighted average',
  '',
  '# every output row is a convex combination of the value',
  '# rows, so attention can never invent a value - only',
  '# decide how much of each one to use.'
];
function wStage(o){
  const sh = [], cw = .62, x0 = 2.95, y0 = 1.35;
  grid(sh, x0, y0, o.masked ? W.ok : W.free, cw, o.masked ? W.mask : null, .30);
  sh.push(fitT(x0 + 3 * cw, y0 - .34, {zh:'keys（可以看的位置）', en:'keys (what it may look at)'},
               {c:COL.grey, fs:.24}));
  sh.push(fitT(x0 - .20, y0 + 3 * cw, {zh:'queries', en:'queries'},
               {anchor:'end', c:COL.grey, fs:.24}));
  if (o.sums){
    const src = o.bad ? W.badSum : W.okSum;
    src.forEach((v, i) => sh.push(fitT(x0 + 6 * cw + .40, y0 + (5 - i) * cw + .26,
                                       v.toFixed(2), {anchor:'start', c:o.bad && v < .999 ? COL.red : COL.tealL, fs:.26})));
    sh.push(fitT(x0 + 6 * cw + .72, y0 + 6 * cw + .18, {zh:'每列加總', en:'row sum'},
                 {c:o.bad ? COL.red : COL.tealL, fs:.24}));
  }
  return caps(sh, o);
}
function wFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:wStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({line:2, cap:{zh:'6 個 query、6 個 key，權重矩陣是 6 x 6',
                  en:'6 queries, 6 keys, so the weight matrix is 6 by 6'},
    panels:[{lbl:'row 0', chips:W.free[0].map(v => chip(v.toFixed(2), v > .3 ? 'hot' : 'dim'))}],
    msg:{zh:'每一格是一個 query 和一個 key 的相似度，過完 softmax 之後。每一列加起來是 1，所以輸出是那 6 個 value 的凸組合——attention 生不出新的值，它只能決定每個值用多少。',
         en:'Each cell is the similarity of one query to one key, after the softmax. Every row sums to 1, so the output is a convex combination of the six value rows - attention cannot invent a value, it can only decide how much of each one to use.'}});
  p({sums:true, line:2, cap:{zh:'每一列加起來都是 1', en:'every row sums to 1'},
    panels:[{lbl:'row sums', chips:W.okSum.map(() => chip('1.00', 'ok'))}],
    msg:{zh:'這就是「softmax 把分數變成分布」的意思：非負、加總為一。接下來每一個分頁都是在動這張表的某一個地方。',
         en:'That is what the softmax buys: non-negative weights that sum to one. Every tab after this one changes something about how this table gets built.'}});
  p({masked:true, sums:true, line:1,
    cap:{zh:'因果 mask：位置 i 不准看 j > i', en:'causal mask: position i may not look at j > i'},
    panels:[{lbl:'row 0', chips:[chip('1.00', 'hot')]},
            {lbl:'row sums', chips:W.okSum.map(v => chip(v.toFixed(2), 'ok'))}],
    msg:{zh:'灰掉的格子是未來。做法是在 softmax 之前把那些分數設成 -inf，exp(-inf) 就是 0，所以它們拿到的權重剛好是 0，而每一列還是加總為 1。第 0 列只看得到自己，所以它的權重是 1.00。',
         en:'The greyed cells are the future. The way to remove them is to set those scores to -inf BEFORE the softmax: exp(-inf) is 0, so they get exactly zero weight and the row still sums to 1. Row 0 can only see itself, so its weight is 1.00.'}});
  p({masked:true, sums:true, bad:true, line:1,
    cap:{zh:'同一件事做在 softmax 之後', en:'the same masking, done after the softmax'},
    panels:[{lbl:'row sums now', chips:W.badSum.map(v => chip(v.toFixed(4), v < .999 ? 'bad' : 'ok'))}],
    msg:{zh:'如果先 softmax 再把未來的格子歸零，每一列就不再加總為 1——第 0 列只剩 ' + W.badSum[0].toFixed(4) + '。等於每個早期位置都被「它不准看的那些未來」按比例縮小了一次，越早的位置縮得越兇。不會噴錯、不會有 nan、loss 照樣會降。',
         en:'Softmax first and zero the future afterwards, and the rows stop summing to 1 - row 0 carries ' + W.badSum[0].toFixed(4) + '. Every early position is scaled down by however much future it was not allowed to see, and the earlier the position the worse it is. Nothing raises, nothing is nan, and the loss still goes down.'}});
  return F.list;
}

/* ================================================================ tab 2: overflow and sqrt(d) */
const CODE_SM = [
  '# softmax(x) == softmax(x - c) for ANY constant c,',
  '# because c cancels between numerator and denominator.',
  'm = x.max(axis=-1, keepdims=True)',
  'e = exp(x - m)            # largest exponent is exp(0) = 1',
  'return e / e.sum(axis=-1, keepdims=True)',
  '',
  '# and divide the scores by sqrt(d) first, or the logits',
  '# grow like sqrt(d) and the softmax becomes an argmax.'
];
function smStage(o){
  const sh = [];
  if (o.mode === 'ovf'){
    sh.push(fitT(4.9, 1.15, 'softmax([1000, 1001, 999])', {c:COL.tealL, fs:.34}));
    sh.push(fitT(3.40, 2.10, {zh:'照公式寫', en:'as written'}, {anchor:'end', c:COL.grey, fs:.28}));
    OVF.naive.forEach((v, i) => sh.push(box(3.60 + i * 1.45, 1.82, 1.30, .56, 'bad', String(v), .28)));
    sh.push(fitT(3.40, 3.00, {zh:'先減最大值', en:'max subtracted'}, {anchor:'end', c:COL.grey, fs:.28}));
    OVF.safe.forEach((v, i) => sh.push(box(3.60 + i * 1.45, 2.72, 1.30, .56, 'ok', v.toFixed(4), .28)));
    side(sh, 1.35, 4.00, [
      [{zh:'float32 到 exp(' + OVF.k32 + ') 就是 inf', en:'float32: exp(' + OVF.k32 + ') is already inf'}, COL.grey, .28],
      [{zh:'float16 到 exp(' + OVF.k16 + ') 就是 inf（上限 ' + OVF.max16 + '）',
        en:'float16: exp(' + OVF.k16 + ') is already inf (it holds up to ' + OVF.max16 + ')'}, COL.red, .28]], .46);
  } else {
    sh.push(fitT(1.45, 1.25, 'd', {anchor:'start', c:COL.grey, fs:.26}));
    sh.push(fitT(3.30, 1.25, {zh:'分數的標準差', en:'std of the raw scores'}, {anchor:'end', c:COL.grey, fs:.26}));
    sh.push(fitT(4.55, 1.25, 'sqrt(d)', {anchor:'end', c:COL.grey, fs:.26}));
    sh.push(fitT(6.70, 1.25, {zh:'最大權重（沒除）', en:'max weight, unscaled'}, {anchor:'end', c:COL.red, fs:.26}));
    sh.push(fitT(8.50, 1.25, {zh:'有除', en:'scaled'}, {anchor:'end', c:COL.tealL, fs:.26}));
    SCALE.forEach((r, i) => {
      if (i > o.upto) return;
      const y = 1.85 + i * .62;
      sh.push(fitT(1.45, y, String(r.d), {anchor:'start', c:COL.pale, fs:.28}));
      sh.push(fitT(3.30, y, r.std.toFixed(3), {anchor:'end', c:COL.pale, fs:.28}));
      sh.push(fitT(4.55, y, r.sqrt.toFixed(2), {anchor:'end', c:COL.grey, fs:.28}));
      sh.push(fitT(6.70, y, r.raw.toFixed(4), {anchor:'end', c:COL.red, fs:.28}));
      sh.push(fitT(8.50, y, r.scaled.toFixed(4), {anchor:'end', c:COL.tealL, fs:.28}));
    });
  }
  return caps(sh, o);
}
function smFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:smStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({mode:'ovf', line:0, cap:{zh:'exp() 很容易就 inf', en:'exp() reaches inf easily'},
    panels:[{lbl:'naive', chips:OVF.naive.map(v => chip(String(v), 'bad'))},
            {lbl:'max subtracted', chips:OVF.safe.map(v => chip(v.toFixed(4), 'ok'))}],
    msg:{zh:'exp(1000) 是 inf，而 inf / inf 是 nan。修法只有一行而且是精確的，不是近似：softmax(x) 等於 softmax(x − c)，因為 c 在分子分母互相消掉。取 c = max(x)，最大的指數就剛好是 exp(0) = 1，什麼都溢不出去。float16 只要 logit 到 ' + OVF.k16 + ' 就會發生。',
         en:'exp(1000) is inf and inf / inf is nan. The fix is one line and it is exact, not an approximation: softmax(x) equals softmax(x - c) because c cancels between numerator and denominator. Take c = max(x) and the largest exponent is exactly exp(0) = 1. In float16 a logit of ' + OVF.k16 + ' is already enough to hit this.'}});
  SCALE.forEach((r, i) => {
    p({mode:'scale', upto:i, line:6,
      cap:{zh:'一個分數是 d 項的內積，標準差會跟著 sqrt(d) 長',
           en:'a score is a dot product of d terms, so its spread grows like sqrt(d)'},
      panels:[{lbl:'d', chips:[chip(String(r.d), 'hot')]},
              {lbl:'std / sqrt(d)', chips:[chip(r.std.toFixed(3), 'act'), chip(r.sqrt.toFixed(2), 'dim')]},
              {lbl:'max weight', chips:[chip(r.raw.toFixed(4), 'bad'), chip(r.scaled.toFixed(4), 'ok')]}],
      msg:{zh:'d = ' + r.d + '：原始分數的標準差 ' + r.std.toFixed(3) + '，而 sqrt(d) 是 ' + r.sqrt.toFixed(2) + '——兩個數字一路跟著走。不除 sqrt(d) 的話，平均每一列的最大權重是 ' + r.raw.toFixed(4) + '；除了之後是 ' + r.scaled.toFixed(4) + '。' + (r.d >= 512 ? '最大權重接近 1 就等於 argmax，其他 key 幾乎拿不到梯度。' : ''),
           en:'d = ' + r.d + ': the raw scores have a standard deviation of ' + r.std.toFixed(3) + ' against sqrt(d) = ' + r.sqrt.toFixed(2) + ' - the two track each other all the way. Without the division the average row has a maximum weight of ' + r.raw.toFixed(4) + '; with it, ' + r.scaled.toFixed(4) + '.' + (r.d >= 512 ? ' A maximum weight near 1 is an argmax, and almost no gradient reaches the other keys.' : '')}});
  });
  return F.list;
}

/* ================================================================ tab 3: flash */
const CODE_FLASH = [
  'for block of keys:',
  '    s      = Q @ K_blk.T / sqrt(d)',
  '    new_m  = maximum(run_max, s.max(-1))',
  '    alpha  = exp(run_max - new_m)      # rescale what we have',
  '    p      = exp(s - new_m)',
  '    run_sum = run_sum * alpha + p.sum(-1)',
  '    out     = out * alpha + p @ V_blk',
  '    run_max = new_m',
  'return out / run_sum'
];
function flashStage(o){
  const sh = [];
  const cols = 8, cw = .80, x0 = .95, y0 = 1.45;
  for (let j = 0; j < cols; j++){
    const done = j < o.upto, cur = j === o.upto;
    sh.push(box(x0 + j * (cw + .10), y0, cw, .62, cur ? 'hot' : (done ? 'done' : 'idle'),
                'K' + j, .28));
    sh.push(box(x0 + j * (cw + .10), y0 + .80, cw, .62, cur ? 'hot' : (done ? 'done' : 'idle'),
                'V' + j, .28));
  }
  sh.push(fitT(4.9, y0 - .32, {zh:'一次只看一塊 key / value', en:'one block of keys and values at a time'},
               {c:COL.grey, fs:.26}));
  side(sh, 1.00, 3.35, [
    [{zh:'帶著兩個數字：目前最大值、exp 的累計和', en:'carry two numbers: the running max and the running sum'}, COL.pale, .28],
    [{zh:'最大值一變，就把已經累積的乘上 exp(舊 − 新)', en:'when the max moves, rescale what you have by exp(old - new)'}, COL.orange, .28]], .50);
  if (o.table){
    sh.push(fitT(2.25, 4.62, {zh:'每塊大小', en:'block size'}, {anchor:'end', c:COL.grey, fs:.26}));
    sh.push(fitT(2.25, 5.06, {zh:'與完整矩陣的差', en:'vs the full matrix'}, {anchor:'end', c:COL.grey, fs:.26}));
    FLASH.forEach((r, i) => {
      const x = 2.85 + i * 1.42;
      sh.push(fitT(x, 4.62, String(r.blk), {c:COL.pale, fs:.26}));
      sh.push(fitT(x, 5.06, r.err.toExponential(0), {c:COL.tealL, fs:.24}));
    });
  }

  return caps(sh, o);
}
function flashFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:flashStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const msgs = [
    {zh:'第一塊：沒有歷史可以縮放，直接算出這一塊的最大值和 exp 累計和。',
     en:'The first block: there is nothing accumulated yet, so just take this block maximum and its sum of exponentials.'},
    {zh:'第二塊：如果這塊的最大值比之前大，就把已經累積的輸出和累計和都乘上 exp(舊最大 − 新最大)，再把這塊加進去。',
     en:'Second block: if this block has a bigger maximum, multiply the accumulated output and sum by exp(old max - new max), then add this block in.'},
    {zh:'第三塊：一樣。注意記憶體從頭到尾只有「一塊分數」加上每一列兩個數字——n x n 的分數矩陣一次都沒有被生出來。',
     en:'Third block, same move. Note the memory: one block of scores plus two numbers per query row. The n by n score matrix is never built.'},
    {zh:'繼續走完所有的 key。每一塊都只被讀一次。',
     en:'Keep going through the keys. Each block is read exactly once.'}];
  for (let i = 0; i < 4; i++) p({upto:i, line:i === 0 ? 1 : 3, cap:{zh:'FlashAttention 的全部內容就是那個 rescale',
                                                                    en:'the rescaling is the whole of FlashAttention'},
    panels:[{lbl:'blocks done', chips:[chip(String(i + 1) + ' / 8', 'act')]},
            {lbl:'state per row', chips:[chip('running max', 'hot'), chip('running sum', 'hot')]}],
    msg:msgs[i]});
  p({upto:8, table:true, line:8,
    cap:{zh:'同一個總和的精確改寫，差只在浮點最後一位', en:'an exact rewrite of the same sum, down to the last bit of a float'},
    panels:[{lbl:'block size', chips:FLASH.map(r => chip(String(r.blk), 'dim'))},
            {lbl:'max difference', chips:FLASH.map(r => chip(r.err.toExponential(0), 'ok'))}],
    msg:{zh:'不管每塊切多大，結果跟直接做完整矩陣的差距都在 1e-15 等級，也就是浮點數的最後一位。這是同一個總和的精確改寫，不是近似——所以 FlashAttention 改的是記憶體和速度，不是模型。它買到的是下一個分頁。',
         en:'At every block size the answer differs from the full-matrix version by about 1e-15, which is the last bit of a float. This is an exact rewrite of the same sum, not an approximation of it - which is why FlashAttention changes the memory and the speed and not the model. What it buys is the next tab.'}});
  return F.list;
}

/* ================================================================ tab 4: the memory bill */
const CODE_MEM = [
  'score matrix = n * n * heads * 2 bytes   # per layer',
  'kv cache     = layers * 2 * kv_heads * head_dim * 2 bytes',
  '               # ... per TOKEN',
  '',
  '# MHA: kv_heads == heads        MQA: kv_heads == 1',
  '# GQA: in between               MLA: cache one low-rank',
  '#                                    vector instead of K and V'
];
function memStage(o){
  const sh = [];
  if (o.mode === 'score'){
    const base = 4.30, maxh = 2.40, bw = .95;
    MEM.forEach((r, i) => {
      if (i > o.upto) return;
      const x = 1.60 + i * 1.75;
      const h = Math.max(.08, maxh * Math.log10(r.gb + 1) / Math.log10(MEM[MEM.length - 1].gb + 1));
      sh.push(S.r(x, base - h, bw, h, i === o.upto ? 'bad' : 'soft', null));
      sh.push(fitT(x + bw / 2, base - h - .24, r.gb >= 1 ? r.gb.toFixed(0) + ' GB' : (r.gb * 1024).toFixed(0) + ' MB',
                   {c:COL.red, fs:.28}));
      sh.push(fitT(x + bw / 2, base + .34, r.ctx.toLocaleString(), {c:COL.grey, fs:.26}));
    });
    sh.push(fitT(4.9, 5.05, {zh:'一層、32 個 head、fp16 的分數矩陣（長條取對數）',
                             en:'the score matrix for one layer, 32 heads, fp16 (bars are log-scaled)'},
                 {c:COL.tealL, fs:.28}));
  } else {
    const base = 4.30, maxh = 2.30, bw = 1.15;
    KV.rows.forEach((r, i) => {
      if (i > o.upto) return;
      const x = 1.45 + i * 1.95;
      const h = Math.max(.06, maxh * r.bytes / KV.rows[0].bytes);
      sh.push(S.r(x, base - h, bw, h, i === o.upto ? 'hot' : 'ok', null));
      sh.push(fitT(x + bw / 2, base - h - .24, r.gb.toFixed(1) + ' GB', {c:COL.tealL, fs:.28}));
      sh.push(fitT(x + bw / 2, base + .34, r.name, {c:COL.pale, fs:.30}));
      sh.push(fitT(x + bw / 2, base + .70, r.kb + ' KB/token', {c:COL.grey, fs:.24}));
    });
    sh.push(fitT(4.9, 5.05, {zh:'128k context 的 KV cache：32 層、32 head、head dim 128、fp16',
                             en:'KV cache at a 128k context: 32 layers, 32 heads, head dim 128, fp16'},
                 {c:COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function memFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:memStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  MEM.forEach((r, i) => {
    p({mode:'score', upto:i, line:0,
      cap:{zh:'分數矩陣每一對位置佔一格', en:'one entry per pair of positions'},
      panels:[{lbl:'context', chips:[chip(r.ctx.toLocaleString(), 'hot')]},
              {lbl:'per layer', chips:[chip(r.gb >= 1 ? r.gb + ' GB' : (r.gb * 1024).toFixed(0) + ' MB', 'bad')]}],
      msg:{zh:'context ' + r.ctx.toLocaleString() + ' 時，光是一層的分數矩陣就是 ' + (r.gb >= 1 ? r.gb + ' GB' : (r.gb * 1024).toFixed(0) + ' MB') + '。' + (i === MEM.length - 1 ? '這不是你配得出來的東西——而 FlashAttention 從來不配它，一次只拿一塊，所以記憶體變成線性，算術還是平方。平方的「工作量」不會消失，消失的只是平方的「儲存」。' : ''),
           en:'At a context of ' + r.ctx.toLocaleString() + ' the score matrix for a single layer is ' + (r.gb >= 1 ? r.gb + ' GB' : (r.gb * 1024).toFixed(0) + ' MB') + '.' + (i === MEM.length - 1 ? ' That is not a thing you can allocate - and FlashAttention never does. It holds one block at a time, so the memory becomes linear while the arithmetic stays quadratic. The quadratic work does not go away, only the quadratic storage does.' : '')}});
  });
  const names = {MHA:'每個 head 都有自己的 K 和 V', GQA:'每 4 個 head 共用一組 K 和 V',
                 MLA:'每個 token 只存一個低秩向量，讀的時候再還原出每個 head 的 K 和 V',
                 MQA:'全部 32 個 head 共用同一組 K 和 V'};
  const namesEn = {MHA:'every head has its own K and V', GQA:'groups of four heads share one K and V',
                   MLA:'cache one low-rank vector per token and rebuild per-head K and V at read time',
                   MQA:'all 32 heads share a single K and V'};
  KV.rows.forEach((r, i) => {
    p({mode:'kv', upto:i, line:1,
      cap:{zh:'KV cache 是每個 token 都要付的', en:'the KV cache is paid per token'},
      panels:[{lbl:r.name, chips:[chip(r.kb + ' KB/token', 'hot'), chip(r.gb.toFixed(1) + ' GB', 'act')]},
              {lbl:'vs MHA', chips:[chip(r.vs.toFixed(1) + 'x', 'ok')]}],
      msg:{zh:r.name + '：' + names[r.name] + '。每個 token ' + r.kb + ' KB，128k context 就是 ' + r.gb.toFixed(1) + ' GB' + (r.vs === 1 ? '，這是基準線。' : '，是 MHA 的 1/' + r.vs.toFixed(1) + '。') + (r.name === 'MQA' ? ' 省最多，代價是 32 個 head 從此看同一份 K 和 V，head 之間不能再各看各的。' : '') + (r.name === 'MLA' ? ' head 仍然各自獨立，代價是讀的時候要多算，而且帶位置資訊的那一小塊要走另一條路。' : ''),
           en:r.name + ': ' + namesEn[r.name] + '. That is ' + r.kb + ' KB per token, or ' + r.gb.toFixed(1) + ' GB at a 128k context' + (r.vs === 1 ? ', which is the baseline.' : ', ' + r.vs.toFixed(1) + ' times smaller than MHA.') + (r.name === 'MQA' ? ' It saves the most and gives up the most: all 32 heads now read the same K and V, so they can no longer attend to different things.' : '') + (r.name === 'MLA' ? ' The heads stay distinct; the cost is extra arithmetic at read time and a separate small path for the part that carries position information.' : '')}});
  });
  p({mode:'kv', upto:KV.rows.length - 1, line:2,
    cap:{zh:'四個都還是隨 context 線性成長', en:'all four still grow linearly with the context'},
    panels:[{lbl:'delta-rule state', chips:[chip(KV.stateMB + ' MB', 'ok'), chip('constant', 'ok')]}],
    msg:{zh:'MQA、GQA、MLA 動的都是「每個 token 要存多少」，沒有一個改掉「每個 token 都要存」。delta rule 那一類改掉的正是這件事：狀態是 ' + KV.stateMB + ' MB，context 是一千還是一百萬都一樣。代價在最後一個分頁。',
         en:'MQA, GQA and MLA all change how much each token costs; none of them changes that every token costs something. The delta-rule family changes exactly that: the state is ' + KV.stateMB + ' MB whether the context is a thousand tokens or a million. What it costs is the last tab.'}});
  return F.list;
}

/* ================================================================ tab 5: paged */
const CODE_PAGE = [
  '# contiguous: reserve for the longest it might get',
  'reserved = n_sequences * max_length',
  '',
  '# paged: hand out fixed-size pages on demand',
  'pages = sum(ceil(length / page_size) for each sequence)',
  '',
  '# the only waste left is the tail of the last page,',
  '# at most page_size - 1 slots per sequence.'
];
function pageStage(o){
  const sh = [], base = 4.30, maxh = 2.30, bw = 1.10;
  const vals = [{lab:{zh:'一段連續空間', en:'one block each'}, w:PAGE.waste, st:'bad'}]
    .concat(PAGE.paged.map(p => ({lab:{zh:'每頁 ' + p.page, en:'pages of ' + p.page}, w:p.waste,
                                  st:p.page === 16 ? 'ok' : 'soft'})));
  vals.forEach((v, i) => {
    if (i > o.upto) return;
    const x = 1.55 + i * 1.95, h = Math.max(.05, maxh * v.w / 100);
    sh.push(S.r(x, base - h, bw, h, v.st, null));
    sh.push(fitT(x + bw / 2, base - h - .24, v.w.toFixed(1) + '%',
                 {c:v.st === 'bad' ? COL.red : COL.tealL, fs:.30}));
    sh.push(fitT(x + bw / 2, base + .34, v.lab, {c:COL.grey, fs:.26}));
  });
  sh.push(fitT(4.9, 5.05, {zh:'8 條序列，每條都可能長到 ' + PAGE.cap.toLocaleString() + ' 個 token，浪費掉的格子',
                           en:'wasted KV slots: 8 sequences that could each reach ' + PAGE.cap.toLocaleString() + ' tokens'},
               {c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
function pageFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:pageStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'實際長度 ' + PAGE.lengths.join(', '), en:'the actual lengths are ' + PAGE.lengths.join(', ')},
    panels:[{lbl:'reserved / used', chips:[chip(PAGE.slots.toLocaleString(), 'bad'), chip(PAGE.used.toLocaleString(), 'ok')]},
            {lbl:'wasted', chips:[chip(PAGE.waste + '%', 'bad')]}],
    msg:{zh:'連續配置的 cache 一旦給出去就不能搬，所以每條序列都必須按「它可能長到多長」來保留——' + PAGE.slots.toLocaleString() + ' 格保留、只用了 ' + PAGE.used.toLocaleString() + ' 格，' + PAGE.waste + '% 是空的。',
         en:'A contiguous cache cannot move once it is handed out, so every sequence has to be reserved for the longest it might get: ' + PAGE.slots.toLocaleString() + ' slots reserved, ' + PAGE.used.toLocaleString() + ' used, ' + PAGE.waste + '% of it empty.'}});
  PAGE.paged.forEach((pg, i) => {
    p({upto:i + 1, line:4,
      cap:{zh:'改成邊長邊發固定大小的頁', en:'hand out fixed-size pages as the sequence grows'},
      panels:[{lbl:'page size', chips:[chip(String(pg.page), 'hot')]},
              {lbl:'pages / wasted', chips:[chip(String(pg.pages), 'act'), chip(pg.waste + '%', pg.waste < 10 ? 'ok' : 'dim')]}],
      msg:{zh:'每頁 ' + pg.page + ' 格：' + pg.pages + ' 頁、' + pg.slots.toLocaleString() + ' 格，浪費 ' + pg.waste + '%。剩下的浪費只有「每條序列最後一頁的尾巴」，不管序列長到多長都最多 ' + (pg.page - 1) + ' 格。' + (pg.page === 256 ? '頁太大就又開始浪費了——頁大小是在「尾巴的浪費」和「頁表的大小」之間選。' : ''),
           en:'Pages of ' + pg.page + ': ' + pg.pages + ' pages, ' + pg.slots.toLocaleString() + ' slots, ' + pg.waste + '% wasted. The only waste left is the tail of each sequence last page - at most ' + (pg.page - 1) + ' slots, however long the sequence gets.' + (pg.page === 256 ? ' Make the pages too big and the waste comes back: the page size is a trade between the tail and the size of the page table.' : '')}});
  });
  p({upto:3, line:7,
    cap:{zh:'attention 於是改成透過頁表讀', en:'attention then reads through a page table'},
    panels:[{lbl:'contiguous', chips:[chip(PAGE.waste + '%', 'bad')]},
            {lbl:'pages of 16', chips:[chip(PAGE.paged[0].waste + '%', 'ok')]}],
    msg:{zh:'代價是 attention 不能再把 K 和 V 當成一塊平坦的陣列讀，要先查頁表。這就是作業系統對記憶體做的同一件事、同一個理由——而且順便讓兩條序列可以共用同一頁前綴。',
         en:'The cost is that attention can no longer read K and V as one flat array; it goes through a page table first. That is the same indirection an operating system uses for exactly the same reason - and it is what lets two sequences share a prefix page.'}});
  return F.list;
}

/* ================================================================ tab 6: delta rule */
const CODE_DELTA = [
  'S <- S (I - beta k k^T) + beta v k^T',
  'read: S @ q   (a real layer has its own query projection)',
  '',
  '# the first term ERASES whatever S currently associates',
  '# with k; the second writes v there instead.',
  '#',
  '# Gated DeltaNet adds one scalar decay per head;',
  '# Kimi KDA makes that decay per channel.'
];
function deltaStage(o){
  const sh = [], X0 = 1.55, Y0 = 1.70, WD = 6.4, HT = 2.50;
  sh.push(S.e(X0, Y0 + HT, X0 + WD, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  const maxN = DELTA.rows[DELTA.rows.length - 1].n;
  DELTA.rows.forEach((r, i) => {
    if (i > o.upto) return;
    const x = X0 + WD * (r.n - DELTA.rows[0].n) / (maxN - DELTA.rows[0].n);
    const y = Y0 + HT - HT * r.lost / 17;
    sh.push(S.c(x, y, .13, r.lost > 0 ? 'bad' : 'ok', null));
    sh.push(fitT(x, Y0 + HT + .32, String(r.n), {c:COL.grey, fs:.24}));
    if (r.lost > 0) sh.push(fitT(x, y - .30, String(r.lost), {c:COL.red, fs:.26}));
  });
  sh.push(fitT(X0 + WD / 2, Y0 + HT + .76, {zh:'寫進一個 16 x 16 狀態的 key-value 對數',
                                            en:'key-value pairs written into one 16 x 16 state'},
               {c:COL.grey, fs:.26}));
  sh.push(fitT(X0 + .20, Y0 - .06, {zh:'被蓋掉的關聯數', en:'associations lost'},
               {anchor:'start', c:COL.grey, fs:.24}));
  return caps(sh, o);
}
function deltaFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:deltaStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const msgs = {
    8:  {zh:'8 對寫進 16 x 16 的狀態：全部都讀得回來，那個 1e-12 是程式正規化 key 時加的 epsilon，不是互相干擾。',
         en:'Eight pairs into a 16 by 16 state: all of them read back, and the 1e-12 is the epsilon the code adds when it normalises the keys, not interference.'},
    16: {zh:'16 對：剛好填滿。d 維空間裡最多就只有 d 個互相正交的 key，所以 d 個關聯剛好塞得下而且互不干擾。',
         en:'Sixteen pairs: exactly full. A d-dimensional space has at most d mutually orthogonal keys, so d associations fit without disturbing each other.'},
    17: {zh:'第 17 對：剛好蓋掉一個。最新寫進去的那一個永遠是完美的——被犧牲的是舊的。',
         en:'The seventeenth pair overwrites exactly one. The newest write is always perfect; it is the old ones that go.'},
    24: {zh:'24 對：蓋掉 8 個。', en:'Twenty-four pairs: eight associations gone.'},
    32: {zh:'32 對：蓋掉 16 個，也就是狀態能記的全部。而且這還是「key 互相正交」的最好情況——換成一般的隨機 key，8 對就已經有 ' + DELTA.smear.toFixed(2) + ' 的誤差，因為每一次寫入都會把已經在裡面的抹糊一點。',
         en:'Thirty-two pairs: sixteen gone, which is everything the state could hold. And this is the best case, with orthogonal keys - with ordinary random keys, eight pairs already give a worst error of ' + DELTA.smear.toFixed(2) + ', because every write smears the ones already there.'}};
  DELTA.rows.forEach((r, i) => {
    p({upto:i, line:0,
      cap:{zh:'固定大小的記憶體，有上限的回想', en:'constant memory, bounded recall'},
      panels:[{lbl:'pairs written', chips:[chip(String(r.n), 'hot')]},
              {lbl:'worst error / lost', chips:[chip(r.err.toExponential(0), r.lost ? 'bad' : 'ok'),
                                                chip(String(r.lost), r.lost ? 'bad' : 'ok')]}],
      msg:msgs[r.n]});
  });
  p({upto:DELTA.rows.length - 1, line:6,
    cap:{zh:'所以這不是「線性注意力比較差」', en:'so it is not that linear attention is worse'},
    panels:[{lbl:'softmax attention', chips:[chip('exact recall', 'ok'), chip('cache grows', 'bad')]},
            {lbl:'delta rule', chips:[chip('bounded recall', 'bad'), chip(KV.stateMB + ' MB fixed', 'ok')]}],
    msg:{zh:'softmax attention 把每個 key 和 value 都留著、每一步重讀一次，所以它精確、而 cache 一直長。delta rule 把它們寫進一個固定大小的矩陣，所以記憶體是常數、而回想有上限。gated 的那一支再加上遺忘係數，讓舊的關聯是慢慢衰減而不是等著被蓋掉：Gated DeltaNet 是每個 head 一個純量，Kimi 的 KDA 細到每個通道一個——狀態仍然是一個固定大小的矩陣。實務上這些層會跟少數幾層完整 attention 交錯使用：完整的那幾層負責精確的長距離回想，線性的那些層用常數成本扛其他全部。',
         en:'Softmax attention keeps every key and value and re-reads them all at every step, so it is exact and its cache keeps growing. The delta rule writes them into one fixed-size matrix, so the memory is constant and the recall has a ceiling. Gated variants add a forget factor so old associations decay instead of waiting to be overwritten - Gated DeltaNet uses one scalar per head, Kimi KDA one per channel; the state is still a fixed-size matrix. In practice these layers get interleaved with a few full-attention ones: the full layers do the exact long-range recall, the linear layers carry everything else at constant cost per token.'}});
  return F.list;
}

/* ================================================================ tab 7: LeetCode */
const CODE_LC = [
  'dq = deque()                       # indices, values decreasing',
  'for i, v in enumerate(nums):',
  '    while dq and nums[dq[-1]] <= v:',
  '        dq.pop()                   # it can never be a maximum again',
  '    dq.append(i)',
  '    if dq[0] <= i - k: dq.popleft()',
  '    if i >= k - 1: out.append(nums[dq[0]])'
];
const CODE_LC2 = [
  'self.heap = list(nums)             # keep only the k largest',
  'heapq.heapify(self.heap)',
  'while len(self.heap) > k: heapq.heappop(self.heap)',
  '',
  'def add(self, val):',
  '    if len(self.heap) < self.k:  heapq.heappush(self.heap, val)',
  '    elif val > self.heap[0]:     heapq.heapreplace(self.heap, val)',
  '    return self.heap[0]            # root = kth largest'
];
function lcStage(o){
  const sh = [], cw = 1.02, X0 = .75, Y0 = 1.25;
  LC239.nums.forEach((v, j) => {
    const inWin = o.win && j >= o.win[0] && j <= o.win[1];
    const held = o.dq && o.dq.indexOf(v) >= 0 && inWin;
    sh.push(box(X0 + j * cw, Y0, cw * .90, .72, held ? 'act' : (inWin ? 'ok' : 'idle'),
                String(v), .30));
    sh.push(fitT(X0 + j * cw + cw * .45, Y0 - .18, String(j), {c:COL.grey, fs:.22}));
  });
  if (o.win){
    const x = X0 + o.win[0] * cw - .08, w = (o.win[1] - o.win[0] + 1) * cw - cw * .10 + .16;
    sh.push(S.r(x, Y0 - .10, w, .92, 'ghost', null));
  }
  side(sh, .75, 2.70, [
    [{zh:'deque（由大到小）', en:'deque, values decreasing'}, COL.grey, .26],
    [(o.dq || []).join('  '), COL.purpL, .34],
    [o.popped && o.popped.length
      ? {zh:'因為比 ' + o.v + ' 小又比它早，永遠贏不了：丟掉 ' + o.popped.join(', '),
         en:'smaller than ' + o.v + ' and older, so they can never win again: dropped ' + o.popped.join(', ')}
      : '', COL.red, .26],
    [{zh:'已輸出的每窗最大值', en:'window maxima emitted so far'}, COL.grey, .26],
    [(o.out || []).join('  '), COL.tealL, .34]
  ], .52);
  return caps(sh, o);
}
function heapStage(o){
  const sh = [], X0 = 2.30, cw = 1.30;
  (o.heap || []).forEach((v, j) => {
    sh.push(box(X0 + j * cw, 2.40, cw * .86, .86, j === 0 ? 'act' : 'ok', String(v), .34));
  });
  sh.push(fitT(X0 + cw * .43, 3.62, {zh:'堆頂 = 第 k 大', en:'root = kth largest'},
               {c:COL.purpL, fs:.26}));
  if (o.v != null){
    sh.push(fitT(X0 + 3.6 * cw, 1.70, {zh:'送進來 ' + o.v, en:'add(' + o.v + ')'},
                 {c:COL.pale, fs:.32}));
    sh.push(fitT(X0 + 3.6 * cw, 2.14,
      o.act === 'drop' ? {zh:'比堆頂小，立刻丟掉', en:'smaller than the root, dropped at once'}
                       : {zh:'擠掉原本的堆頂', en:'pushes the old root out'},
      {c:o.act === 'drop' ? COL.red : COL.tealL, fs:.26}));
  }
  return caps(sh, o);
}
function lcFrames(v){
  const F = new Frames();
  if (v === 1){
    const p = (o) => F.push({shapes:heapStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
    p({heap:[4, 5, 8], line:2,
      cap:{zh:'LeetCode 703：資料流裡的第 k 大', en:'LeetCode 703: kth largest in a stream'},
      panels:[{lbl:'k', chips:[chip(String(LC703.k), 'hot')]},
              {lbl:'heap', chips:[4, 5, 8].map(x => chip(String(x), 'ok'))}],
      msg:{zh:'起手先把 ' + LC703.init.join(', ') + ' 丟進一個 min-heap，再把多的彈掉，只留最大的 ' + LC703.k + ' 個。留 min-heap 而不是 max-heap，是因為我們要隨時能丟掉「這 ' + LC703.k + ' 個裡最小的那個」——那正好是堆頂。',
           en:'Start by heapifying ' + LC703.init.join(', ') + ' and popping until only the ' + LC703.k + ' largest remain. It is a min-heap rather than a max-heap because what we need to discard at any moment is the smallest of the ' + LC703.k + ' we are keeping - and that is exactly the root.'}});
    LC703.steps.forEach(s => {
      p({heap:s.heap, v:s.v, act:s.act, line:6,
        cap:{zh:'每次只做一次 push 再 pop', en:'one push and one pop per call'},
        panels:[{lbl:'add', chips:[chip(String(s.v), 'hot')]},
                {lbl:'kth largest', chips:[chip(String(s.root), 'ok')]}],
        msg:s.act === 'drop'
          ? {zh:'送進來的 ' + s.v + ' 比堆頂 ' + s.root + ' 小，表示它連前 ' + LC703.k + ' 名都排不進去，直接丟。第 ' + LC703.k + ' 大仍然是 ' + s.root + '。',
             en:'The new value ' + s.v + ' is below the root ' + s.root + ', so it cannot be in the top ' + LC703.k + ' and is thrown away. The kth largest is still ' + s.root + '.'}
          : {zh:s.v + ' 擠掉原本的堆頂，堆變成 ' + s.heap.join(', ')+ '，第 ' + LC703.k + ' 大是 ' + s.root + '。堆永遠只有 ' + LC703.k + ' 個元素，所以每次 add 是 O(log k)、記憶體是 O(k)，跟資料流多長沒有關係。',
             en:s.v + ' pushes the old root out; the heap becomes ' + s.heap.join(', ') + ' and the kth largest is ' + s.root + '. The heap never holds more than ' + LC703.k + ' elements, so each add costs O(log k) and the memory is O(k) no matter how long the stream runs.'}});
    });
    return F.list;
  }
  const p = (o) => F.push({shapes:lcStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  LC239.steps.forEach(s => {
    p({win:s.win, dq:s.dq, out:s.out, popped:s.popped, v:s.v,
      line:s.popped.length ? 3 : (s.out.length ? 6 : 4),
      cap:{zh:'LeetCode 239：每個長度 ' + LC239.k + ' 的窗要一個最大值',
           en:'LeetCode 239: one maximum per window of ' + LC239.k},
      panels:[{lbl:'i / value', chips:[chip(String(s.i), 'dim'), chip(String(s.v), 'hot')]},
              {lbl:'deque', chips:s.dq.map(x => chip(String(x), 'act'))},
              {lbl:'output', chips:s.out.length ? s.out.map(x => chip(String(x), 'ok')) : [chip('-', 'empty')]}],
      msg:s.popped.length
        ? {zh:'新來的是 ' + s.v + '。' + s.popped.join(', ') + ' 比它小、又比它早離開窗口，所以在任何還包含它們的窗裡 ' + s.v + ' 也都在：它們永遠不可能再當最大值，直接從尾巴彈掉。整個 deque 每個元素只進一次出一次，所以 n 個元素總共 O(n)。',
           en:'The new value is ' + s.v + '. Each of ' + s.popped.join(', ') + ' is both smaller and leaves the window earlier, so every window that still contains one of them also contains ' + s.v + ': they can never be a maximum again and get popped off the back. Every element enters and leaves the deque once, which is what makes the whole scan O(n).'}
        : {zh:'新來的 ' + s.v + ' 沒有蓋掉任何人，直接接在尾巴。deque 從頭到尾由大到小，所以' + (s.out.length ? '頭就是這個窗的最大值 ' + s.out[s.out.length - 1] + '。' : '頭就是目前的最大值。'),
           en:'The new value ' + s.v + ' does not dominate anything, so it just joins the back. The deque stays in decreasing order, so the front is ' + (s.out.length ? 'the maximum of this window, ' + s.out[s.out.length - 1] + '.' : 'the current maximum.')}});
  });
  p({win:[5, 7], dq:LC239.steps[7].dq, out:LC239.ans, popped:[], v:7, line:6,
    cap:{zh:'跟 attention 的關係', en:'what this has to do with attention'},
    panels:[{lbl:'answer', chips:LC239.ans.map(x => chip(String(x), 'ok'))},
            {lbl:'cost', chips:[chip('O(n)', 'ok'), chip('O(k)', 'ok')]}],
    msg:{zh:'這兩題做的是同一件事，也正是 online softmax 做的事：掃過一串資料，只保留「之後還可能用得到」的那一小部分，其餘的看過就丟。239 保留的是還可能當最大值的那幾個，703 保留的是最大的 k 個，FlashAttention 保留的是一個 running max 跟一個 running sum——三個都是用固定大小的狀態換掉「把整串留著」。',
         en:'Both problems do the same thing, and it is the thing online softmax does: sweep a stream, keep only the small part that could still matter, and throw the rest away as you pass it. 239 keeps the values that could still become a maximum, 703 keeps the k largest, FlashAttention keeps a running maximum and a running sum - three versions of trading the whole stream for a fixed-size state.'}});
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'Attention 從零實作', en:'Attention from scratch'},
  sub: {zh:'一個加權平均，以及所有人都在改的那四個地方',
        en:'one weighted average, and the four things everyone changes about it'},
  tabs: [
    {id:'w', label:{zh:'權重與 mask', en:'weights and masking'},
     stage:{zh:'權重矩陣與 causal mask', en:'the weight matrix and the causal mask'},
     view:VIEW, code:CODE_W, build:wFrames,
     idea:{zh:'attention 的輸出是 V 的加權平均，權重是對 Q·K/sqrt(d) 做 softmax。causal mask 必須在 softmax 之前把分數設成 -inf；放在之後每一列就不再加總為 1。',
           en:'The output is a weighted average of V whose weights are a softmax over Q·K/sqrt(d). A causal mask has to set the scores to -inf before the softmax; applied afterwards, the rows no longer sum to 1.'},
     legend:[['#12b3b8', {zh:'允許看的位置', en:'allowed position'}],
             ['#ff9736', {zh:'權重偏高', en:'high weight'}],
             ['#8fa3ac', {zh:'被 mask 擋掉', en:'masked out'}]]},
    {id:'sm', label:{zh:'softmax 與 sqrt(d)', en:'softmax and the sqrt(d)'},
     stage:{zh:'溢位與尺度', en:'overflow and scale'},
     view:VIEW, code:CODE_SM, build:smFrames,
     idea:{zh:'exp 在 fp32 到 89、fp16 到 12 就爆掉，所以 softmax 一定先減去每列最大值。除以 sqrt(d) 則是因為 d 維內積的標準差本來就是 sqrt(d)，不除的話維度越大 softmax 越尖。',
           en:'exp overflows at 89 in fp32 and at 12 in fp16, so softmax always subtracts the row maximum first. The sqrt(d) is there because a d-dimensional dot product has standard deviation sqrt(d); without it the softmax gets sharper as the model gets wider.'},
     legend:[['#ff5c5c', {zh:'溢位 / 未縮放', en:'overflow or unscaled'}],
             ['#12b3b8', {zh:'減最大值 / 除 sqrt(d)', en:'max subtracted, scaled'}]]},
    {id:'flash', label:{zh:'FlashAttention', en:'FlashAttention'},
     stage:{zh:'一塊一塊地算 softmax', en:'softmax one block at a time'},
     view:VIEW, code:CODE_FLASH, build:flashFrames,
     idea:{zh:'一塊一塊地走 K 和 V，一路維護 running max 與 running sum，把舊的部分和乘上 exp(舊max - 新max) 補正。跟一次算完整個矩陣的差只在浮點數最後一兩位，但沒有任何時刻需要把 n x n 的分數留著。',
           en:'Walk K and V one block at a time, carrying a running maximum and a running sum, and rescale the partial result by exp(old max - new max). The answer matches the full matrix to the last few bits, but the n by n scores never exist all at once.'},
     legend:[['#ff9736', {zh:'正在處理的塊', en:'block being processed'}],
             ['#12b3b8', {zh:'已累積的部分和', en:'accumulated so far'}]]},
    {id:'mem', label:{zh:'記憶體帳單', en:'the memory bill'},
     stage:{zh:'分數矩陣與 KV cache', en:'the score matrix and the KV cache'},
     view:VIEW, code:CODE_MEM, build:memFrames,
     idea:{zh:'分數矩陣是 n 的平方，128k context 一層就 1024 GB；KV cache 則是每個 token 一筆固定開銷，MHA / GQA / MLA / MQA 差到 32 倍。前者靠 FlashAttention 消掉，後者靠共用 K/V 或壓成 latent。',
           en:'The score matrix grows with n squared - 1024 GB for one layer at a 128k context - while the KV cache is a fixed cost per token that differs by a factor of 32 between MHA, GQA, MLA and MQA. FlashAttention removes the first; sharing or compressing K and V shrinks the second.'},
     legend:[['#ff5c5c', {zh:'放不下', en:'does not fit'}],
             ['#12b3b8', {zh:'每 token 的 KV 大小', en:'KV bytes per token'}]]},
    {id:'page', label:{zh:'PagedAttention', en:'PagedAttention'},
     stage:{zh:'固定大小的頁', en:'fixed-size pages'},
     view:VIEW, code:CODE_PAGE, build:pageFrames,
     idea:{zh:'連續配置的 KV cache 必須按最長可能長度保留，實測 89.3% 是空的。改成固定大小的頁隨長隨發，浪費掉到 1.2%，代價是 attention 要透過頁表讀 K 和 V。',
           en:'A contiguous KV cache has to be reserved for the longest length a sequence might reach, and 89.3% of it sits empty. Handing out fixed-size pages as the sequence grows cuts the waste to 1.2%, at the cost of reading K and V through a page table.'},
     legend:[['#ff5c5c', {zh:'保留但沒用到', en:'reserved and unused'}],
             ['#12b3b8', {zh:'分頁後的浪費', en:'waste after paging'}]]},
    {id:'delta', label:{zh:'delta rule 與 KDA', en:'the delta rule and KDA'},
     stage:{zh:'固定大小的狀態', en:'a fixed-size state'},
     view:VIEW, code:CODE_DELTA, build:deltaFrames,
     idea:{zh:'把 key-value 對寫進一個固定大小的矩陣而不是一直長的 cache：記憶體變成常數，但一個 d x d 的狀態最多只裝得下 d 個互不干擾的關聯，第 d+1 個開始就會蓋掉舊的。',
           en:'Write key-value pairs into one fixed-size matrix instead of an ever-growing cache: the memory becomes constant, but a d by d state holds at most d associations without interference, and the next one starts overwriting the old.'},
     legend:[['#12b3b8', {zh:'全部讀得回來', en:'everything reads back'}],
             ['#ff5c5c', {zh:'已被蓋掉', en:'overwritten'}]]},
    {id:'lc', label:{zh:'LeetCode 239 / 703', en:'LeetCode 239 / 703'},
     stage:{zh:'只留之後還用得到的', en:'keep only what can still matter'},
     view:VIEW, build:lcFrames,
     variants:[{zh:'239 滑動窗最大值', en:'239 sliding window maximum'},
               {zh:'703 資料流第 k 大', en:'703 kth largest in a stream'}],
     get code(){ return curV() === 0 ? CODE_LC : CODE_LC2; },
     idea:{zh:'兩題都是「掃過一串資料，只留之後還用得到的那一小部分」：239 用單調 deque 留還可能當最大值的，703 用大小 k 的 min-heap 留最大的 k 個。online softmax 留的是 running max 和 running sum，是同一種做法。',
           en:'Both sweep a stream and keep only the part that could still matter: 239 keeps, in a monotonic deque, the values that could still become a maximum; 703 keeps the k largest in a size-k min-heap. Online softmax keeps a running maximum and a running sum - the same move.'},
     legend:[['#ff9736', {zh:'deque / 堆裡的值', en:'held in the deque or heap'}],
             ['#12b3b8', {zh:'已輸出的答案', en:'emitted answer'}],
             ['#8fa3ac', {zh:'已離開窗口', en:'outside the window'}]]}
  ]
};
