// DAY: 50
// TITLE_ZH: 評估指標與交叉驗證：決定要看 accuracy、F1 還是 AUC
// TITLE_EN: Evaluation metrics and cross-validation: deciding whether to look at accuracy, F1 or AUC
// SUB_ZH: 分類器交給你的是一個分數，不是一個決定。你之後報出來的每一個數字——accuracy、precision、recall、F1、AUC——都是同樣那四個計數的不同比值，而選一個指標就是在選你願意犯哪一種錯。一個明明學到東西的模型可以輸給一個常數；兩個模型可以被兩條大家都在引用的曲線排出完全相反的順序；一折資料可以一個正樣本都沒有；而一條看起來很小心的流程可以把一個實際值 0.4750 的模型報成 0.9900。
// SUB_EN: A classifier hands you a score, not a decision. Every number you then report - accuracy, precision, recall, F1, AUC - is a different ratio of the same four counts, and choosing one is choosing which mistakes you are willing to make. A model that clearly learned something can lose to a constant; two models can be ranked in opposite orders by the two curves everyone quotes; a fold can end up with no positives in it; and a careful-looking pipeline can report 0.9900 for a model worth 0.4750.
// FOLDER: day%2050%20-%20evaluation%20metrics%20and%20cross%20validation
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
const CONF = {"const":{"tn":1980, "fp":0, "fn":20, "tp":0, "acc":0.99, "prec":null, "rec":0.0, "f1":null, "mcc":null, "bal":0.5}, "model":{"tn":1979, "fp":1, "fn":20, "tp":0, "acc":0.9895, "prec":0.0, "rec":0.0, "f1":0.0, "mcc":-0.0022, "bal":0.4997}, "maj":0, "base":0.99, "n":2000, "pos":20, "topScore":0.6481, "auc":0.9027, "ap":0.2033};
const SWEEP = [{"t":0.5, "flag":1, "acc":0.9895, "prec":0.0, "rec":0.0, "f1":0.0, "tp":0}, {"t":0.3, "flag":1, "acc":0.9895, "prec":0.0, "rec":0.0, "f1":0.0, "tp":0}, {"t":0.2, "flag":3, "acc":0.9895, "prec":0.3333, "rec":0.05, "f1":0.087, "tp":1}, {"t":0.15, "flag":4, "acc":0.989, "prec":0.25, "rec":0.05, "f1":0.0833, "tp":1}, {"t":0.1, "flag":20, "acc":0.987, "prec":0.35, "rec":0.35, "f1":0.35, "tp":7}, {"t":0.05, "flag":88, "acc":0.956, "prec":0.1136, "rec":0.5, "f1":0.1852, "tp":10}, {"t":0.03, "flag":220, "acc":0.894, "prec":0.0636, "rec":0.7, "f1":0.1167, "tp":14}, {"t":0.02, "flag":413, "acc":0.7995, "prec":0.0387, "rec":0.8, "f1":0.0739, "tp":16}, {"t":0.01, "flag":845, "acc":0.5865, "prec":0.0225, "rec":0.95, "f1":0.0439, "tp":19}];
const CURVE = {"broad":{"roc":[[0.0, 0.0], [0.0086, 0.15], [0.0157, 0.5], [0.0237, 0.75], [0.0333, 0.85], [0.0429, 0.95], [0.0535, 0.95], [0.0641, 0.95], [0.0742, 1.0], [0.0848, 1.0], [0.0955, 1.0], [0.1061, 1.0], [0.1167, 1.0], [0.1273, 1.0], [0.1379, 1.0], [0.1485, 1.0], [0.1591, 1.0], [0.1697, 1.0], [0.1803, 1.0], [0.1909, 1.0], [0.2015, 1.0], [0.2121, 1.0], [0.2227, 1.0], [0.2333, 1.0], [0.2439, 1.0], [0.2545, 1.0], [0.2652, 1.0], [0.2758, 1.0], [0.2864, 1.0], [0.297, 1.0], [0.3071, 1.0], [0.3177, 1.0], [0.3283, 1.0], [0.3389, 1.0], [0.3495, 1.0], [0.3601, 1.0], [0.3707, 1.0], [0.3813, 1.0], [0.3919, 1.0], [0.4025, 1.0], [0.4131, 1.0], [0.4237, 1.0], [0.4343, 1.0], [0.4449, 1.0], [0.4556, 1.0], [0.4662, 1.0], [0.4768, 1.0], [0.4874, 1.0], [0.498, 1.0], [0.5086, 1.0], [0.5192, 1.0], [0.5298, 1.0], [0.5404, 1.0], [0.551, 1.0], [0.5616, 1.0], [0.5722, 1.0], [0.5828, 1.0], [0.5934, 1.0], [0.604, 1.0], [0.6146, 1.0], [0.6247, 1.0], [0.6354, 1.0], [0.646, 1.0], [0.6566, 1.0], [0.6672, 1.0], [0.6778, 1.0], [0.6884, 1.0], [0.699, 1.0], [0.7096, 1.0], [0.7202, 1.0], [0.7308, 1.0], [0.7414, 1.0], [0.752, 1.0], [0.7626, 1.0], [0.7732, 1.0], [0.7838, 1.0], [0.7944, 1.0], [0.8051, 1.0], [0.8157, 1.0], [0.8263, 1.0], [0.8369, 1.0], [0.8475, 1.0], [0.8581, 1.0], [0.8687, 1.0], [0.8793, 1.0], [0.8899, 1.0], [0.9005, 1.0], [0.9111, 1.0], [0.9217, 1.0], [1.0, 1.0]], "pr":[[0.0, 0.0], [0.2, 0.1905], [0.55, 0.2619], [0.75, 0.2381], [0.85, 0.2024], [0.95, 0.181], [0.95, 0.1508], [0.95, 0.1293], [1.0, 0.119], [1.0, 0.1058], [1.0, 0.0952], [1.0, 0.0866], [1.0, 0.0794], [1.0, 0.0733], [1.0, 0.068], [1.0, 0.0635], [1.0, 0.0595], [1.0, 0.056], [1.0, 0.0529], [1.0, 0.0501], [1.0, 0.0476], [1.0, 0.0454], [1.0, 0.0433], [1.0, 0.0415], [1.0, 0.0398], [1.0, 0.0382], [1.0, 0.0367], [1.0, 0.0353], [1.0, 0.0341], [1.0, 0.0329], [1.0, 0.0318], [1.0, 0.0308], [1.0, 0.0298], [1.0, 0.0289], [1.0, 0.0281], [1.0, 0.0272], [1.0, 0.0265], [1.0, 0.0258], [1.0, 0.0251], [1.0, 0.0244], [1.0, 0.0238], [1.0, 0.0233], [1.0, 0.0227], [1.0, 0.0222], [1.0, 0.0217], [1.0, 0.0212], [1.0, 0.0207], [1.0, 0.0203], [1.0, 0.0199], [1.0, 0.0195], [1.0, 0.0191], [1.0, 0.0187], [1.0, 0.0183], [1.0, 0.018], [1.0, 0.0177], [1.0, 0.0173], [1.0, 0.017], [1.0, 0.0167], [1.0, 0.0164], [1.0, 0.0162], [1.0, 0.0159], [1.0, 0.0156], [1.0, 0.0154], [1.0, 0.0151], [1.0, 0.0149], [1.0, 0.0147], [1.0, 0.0145], [1.0, 0.0142], [1.0, 0.014], [1.0, 0.0138], [1.0, 0.0136], [1.0, 0.0134], [1.0, 0.0133], [1.0, 0.0131], [1.0, 0.0129], [1.0, 0.0127], [1.0, 0.0126], [1.0, 0.0124], [1.0, 0.0122], [1.0, 0.0121], [1.0, 0.0119], [1.0, 0.0118], [1.0, 0.0116], [1.0, 0.0115], [1.0, 0.0114], [1.0, 0.0112], [1.0, 0.0111], [1.0, 0.011], [1.0, 0.0108], [1.0, 0.01]], "auc":0.98, "ap":0.2124, "topk":{"10":0, "20":3, "50":12, "100":19}}, "sharp":{"roc":[[0.0, 0.0], [0.0061, 0.5], [0.0172, 0.5], [0.0288, 0.5], [0.0399, 0.5], [0.0515, 0.5], [0.0626, 0.5], [0.0742, 0.5], [0.0854, 0.5], [0.097, 0.5], [0.1081, 0.5], [0.1197, 0.5], [0.1308, 0.5], [0.1424, 0.5], [0.1535, 0.5], [0.1652, 0.5], [0.1763, 0.5], [0.1879, 0.5], [0.199, 0.5], [0.2101, 0.5], [0.2217, 0.5], [0.2328, 0.5], [0.2444, 0.5], [0.2556, 0.5], [0.2672, 0.5], [0.2783, 0.5], [0.2899, 0.5], [0.301, 0.5], [0.3126, 0.5], [0.3237, 0.5], [0.3354, 0.5], [0.3465, 0.5], [0.3581, 0.5], [0.3692, 0.5], [0.3808, 0.5], [0.3919, 0.5], [0.403, 0.5], [0.4146, 0.5], [0.4258, 0.5], [0.4374, 0.5], [0.4485, 0.5], [0.4601, 0.5], [0.4712, 0.5], [0.4828, 0.5], [0.4939, 0.5], [0.5056, 0.5], [0.5167, 0.5], [0.5283, 0.5], [0.5394, 0.5], [0.551, 0.5], [0.5621, 0.5], [0.5737, 0.5], [0.5848, 0.5], [0.5965, 0.5], [0.6076, 0.5], [0.6187, 0.5], [0.6303, 0.5], [0.6414, 0.5], [0.653, 0.5], [0.6641, 0.5], [0.6758, 0.5], [0.6869, 0.5], [0.6985, 0.5], [0.7096, 0.5], [0.7212, 0.5], [0.7323, 0.5], [0.7434, 0.55], [0.754, 0.6], [0.7652, 0.65], [0.7758, 0.7], [0.7859, 0.85], [0.797, 0.85], [0.8081, 0.85], [0.8197, 0.85], [0.8298, 0.95], [0.8409, 1.0], [0.852, 1.0], [0.8636, 1.0], [0.8747, 1.0], [0.8864, 1.0], [0.8975, 1.0], [0.9091, 1.0], [0.9202, 1.0], [0.9318, 1.0], [0.9429, 1.0], [0.9545, 1.0], [0.9657, 1.0], [0.9773, 1.0], [0.9884, 1.0], [1.0, 1.0]], "pr":[[0.05, 1.0], [0.5, 0.4348], [0.5, 0.2222], [0.5, 0.1471], [0.5, 0.1111], [0.5, 0.0885], [0.5, 0.0741], [0.5, 0.0633], [0.5, 0.0556], [0.5, 0.0493], [0.5, 0.0444], [0.5, 0.0403], [0.5, 0.037], [0.5, 0.0342], [0.5, 0.0317], [0.5, 0.0297], [0.5, 0.0278], [0.5, 0.0262], [0.5, 0.0247], [0.5, 0.0234], [0.5, 0.0222], [0.5, 0.0212], [0.5, 0.0202], [0.5, 0.0193], [0.5, 0.0185], [0.5, 0.0178], [0.5, 0.0171], [0.5, 0.0165], [0.5, 0.0159], [0.5, 0.0153], [0.5, 0.0148], [0.5, 0.0143], [0.5, 0.0139], [0.5, 0.0135], [0.5, 0.0131], [0.5, 0.0127], [0.5, 0.0124], [0.5, 0.012], [0.5, 0.0117], [0.5, 0.0114], [0.5, 0.0111], [0.5, 0.0109], [0.5, 0.0106], [0.5, 0.0104], [0.5, 0.0101], [0.5, 0.0099], [0.5, 0.0097], [0.5, 0.0095], [0.5, 0.0093], [0.5, 0.0091], [0.5, 0.0089], [0.5, 0.0087], [0.5, 0.0086], [0.5, 0.0084], [0.5, 0.0082], [0.5, 0.0081], [0.5, 0.0079], [0.5, 0.0078], [0.5, 0.0077], [0.5, 0.0075], [0.5, 0.0074], [0.5, 0.0073], [0.5, 0.0072], [0.5, 0.0071], [0.5, 0.007], [0.5, 0.0068], [0.55, 0.0074], [0.6, 0.008], [0.65, 0.0085], [0.7, 0.009], [0.85, 0.0108], [0.85, 0.0107], [0.85, 0.0105], [0.85, 0.0104], [0.95, 0.0114], [1.0, 0.0119], [1.0, 0.0117], [1.0, 0.0116], [1.0, 0.0114], [1.0, 0.0113], [1.0, 0.0111], [1.0, 0.011], [1.0, 0.0109], [1.0, 0.0107], [1.0, 0.0106], [1.0, 0.0105], [1.0, 0.0104], [1.0, 0.0102], [1.0, 0.0101], [1.0, 0.01]], "auc":0.6082, "ap":0.5049, "topk":{"10":10, "20":10, "50":10, "100":10}}, "pos":20, "neg":1980};
const FOLDS = {"plain":[0, 6, 2, 1, 0, 3, 1, 4, 1, 2], "strat":[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], "runs":166, "trials":200};
const LEAK = [{"id":"group", "bad":0.99, "good":0.475}, {"id":"select", "bad":0.7837, "good":0.5169}, {"id":"scale", "bad":0.7167, "good":0.7167, "gap":0.00015}];
const WINNER = [{"m":1, "pick":0.5061, "fresh":0.494}, {"m":10, "pick":0.5563, "fresh":0.5044}, {"m":50, "pick":0.58, "fresh":0.5064}, {"m":200, "pick":0.5981, "fresh":0.4952}];
const LC493 = {"nums":[7, 2, 9, 1, 0, 4], "left":[2, 7, 9], "right":[0, 1, 4], "steps":[{"v":2, "j0":0, "j":1, "add":1, "cnt":1}, {"v":7, "j0":1, "j":2, "add":2, "cnt":3}, {"v":9, "j0":2, "j":3, "add":3, "cnt":6}], "cross":6, "total":8, "brute":8};
const LC169 = {"nums":[2, 2, 1, 3, 2, 1, 2], "steps":[{"v":2, "cand":2, "c":1}, {"v":2, "cand":2, "c":2}, {"v":1, "cand":2, "c":1}, {"v":3, "cand":2, "c":0}, {"v":2, "cand":2, "c":1}, {"v":1, "cand":2, "c":0}, {"v":2, "cand":2, "c":1}], "ans":2, "count":4, "noMaj":[1, 2, 3], "noMajSays":3, "noMajChecked":null};

const FRUIT = {"n":100, "apples":10, "tp":6, "fp":2, "fn":4, "tn":88, "acc":0.94, "prec":0.75, "rec":0.6, "fpr":0.0222, "f1":0.6667, "lazyAcc":0.9, "lazyRec":0.0};
const BAL_SPREAD = {auc:0.00146, ap:0.72529, ratio:496, above:135, flagged:155};
const BALANCE = [{"keep":1980, "rate":0.01, "auc":0.98, "ap":0.2124}, {"keep":495, "rate":0.0388, "auc":0.9807, "ap":0.5331}, {"keep":124, "rate":0.1389, "auc":0.9797, "ap":0.8142}, {"keep":40, "rate":0.3333, "auc":0.9811, "ap":0.9377}];

/* ================================================================ tab 1: the four counts */
const CODE_COUNTS = [
  'tn, fp, fn, tp = confusion(y_true, y_pred)',
  '',
  'accuracy  = (tp + tn) / (tp + tn + fp + fn)',
  'precision = tp / (tp + fp)      # of what we flagged',
  'recall    = tp / (tp + fn)      # of what was there',
  'F1        = 2 * p * r / (p + r)',
  '',
  '# four counts, and every metric is one ratio of them.'
];
function matrix(sh, x0, y0, title, m, hot){
  const cw = 1.30, ch = .78;
  sh.push(fitT(x0 + cw, y0 - .34, title, {c:COL.tealL, fs:.28}));
  sh.push(fitT(x0 + cw * .5, y0 + .04, {zh:'猜 0', en:'guessed 0'}, {c:COL.grey, fs:.22}));
  sh.push(fitT(x0 + cw * 1.5, y0 + .04, {zh:'猜 1', en:'guessed 1'}, {c:COL.grey, fs:.22}));
  sh.push(fitT(x0 - .14, y0 + ch * .70, {zh:'真的 0', en:'really 0'}, {anchor:'end', c:COL.grey, fs:.22}));
  sh.push(fitT(x0 - .14, y0 + ch * 1.70, {zh:'真的 1', en:'really 1'}, {anchor:'end', c:COL.grey, fs:.22}));
  const grid = [[m.tn, m.fp], [m.fn, m.tp]];
  for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++){
    const isTP = (r === 1 && c === 1);
    sh.push(box(x0 + c * cw, y0 + .22 + r * ch, cw, ch, (hot && isTP) ? 'hot' : 'idle',
                String(grid[r][c]), .34));
  }
  return sh;
}
function countsStage(o){
  const sh = [];
  matrix(sh, 1.55, 1.15, {zh:'永遠回答 ' + CONF.maj, en:'always answer ' + CONF.maj}, CONF.const, o.hot);
  matrix(sh, 5.75, 1.15, {zh:'模型，切在 0.5', en:'the model, cut at 0.5'}, CONF.model, o.hot);
  const rows = [['accuracy', CONF.const.acc, CONF.model.acc],
                ['recall', CONF.const.rec, CONF.model.rec],
                ['precision', CONF.const.prec, CONF.model.prec],
                ['F1', CONF.const.f1, CONF.model.f1],
                ['MCC', CONF.const.mcc, CONF.model.mcc]];
  const show = Math.min(o.rows || 0, rows.length);
  for (let i = 0; i < show; i++){
    const y = 3.35 + i * .46, r = rows[i];
    const dead = (r[1] == null || r[2] == null);
    sh.push(fitT(3.00, y, r[0], {anchor:'end', c:dead ? COL.red : COL.pale, fs:.28}));
    sh.push(fitT(4.15, y, fmtN(r[1]), {anchor:'end', c:dead ? COL.red : COL.tealL, fs:.28}));
    sh.push(fitT(8.35, y, fmtN(r[2]), {anchor:'end', c:dead ? COL.red : COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function countsFrames(){
  const F = new Frames();
  const push = (o) => F.push({shapes:countsStage(o), panels:o.panels, line:o.line,
                              msg:o.msg, view:VIEW});
  push({rows:0, line:0,
    cap:{zh:CONF.n + ' 列，其中 ' + CONF.pos + ' 列是正樣本',
         en:CONF.n + ' rows, ' + CONF.pos + ' of them positive'},
    panels:[{lbl:'counts', chips:[chip('tn ' + CONF.model.tn, 'dim'), chip('fp ' + CONF.model.fp, 'dim'),
                                  chip('fn ' + CONF.model.fn, 'dim'), chip('tp ' + CONF.model.tp, 'bad')]}],
    msg:{zh:'左邊是「什麼都不做、一律回答 ' + CONF.maj + '」，右邊是訓練過的 logistic regression 切在 0.5。兩張表幾乎一模一樣：右邊只多抓了一列，而且抓錯。',
         en:'On the left, do nothing and always answer ' + CONF.maj + '. On the right, a trained logistic regression cut at 0.5. The two tables are nearly identical: the right one flagged one extra row, and got it wrong.'}});
  push({rows:0, hot:true, line:0,
    cap:{zh:'橘色那一格是這份工作真正要的東西', en:'the amber cell is the one the job is about'},
    panels:[{lbl:'true positives', chips:[chip(String(CONF.const.tp), 'bad'), chip(String(CONF.model.tp), 'bad')]}],
    msg:{zh:'tp 是「真的是正樣本、而且被抓出來」的列數。兩邊都是 0。沒有任何一個欺詐案件、沒有任何一個故障被找到。',
         en:'tp is the number of rows that really are positive and were flagged. Both sides are 0. Not one fraud, not one failure, found.'}});
  for (let k = 1; k <= 5; k++){
    const names = ['accuracy', 'recall', 'precision', 'F1', 'MCC'];
    const msgs = [
      {zh:'accuracy 把 tp 和 tn 加在一起。這裡 tn 佔了 99%，所以 accuracy 基本上是在報告 tn——兩邊都是 0.99 左右，而且常數那邊還比較高。',
       en:'accuracy adds tp and tn together. tn is 99% of the table here, so accuracy is mostly a report about tn - both sides sit near 0.99, and the constant wins.'},
      {zh:'recall 的分母是「真的有幾個正樣本」，分子是 tp。兩邊都是 0.0000，這是第一個說實話的數字。',
       en:'recall divides tp by how many positives really exist. Both sides are 0.0000 - the first number that tells the truth.'},
      {zh:'precision 的分母是「你標了幾列」。常數一列都沒標，所以是 0/0＝nan。它拒絕回答，而那個拒絕本身就是資訊——它們注意到這裡沒有東西可以評。',
       en:'precision divides by how many rows you flagged. The constant flagged none, so it is 0/0 = nan. It refuses to answer, and that refusal is itself the information: it noticed there is nothing to score.'},
      {zh:'F1 是 precision 和 recall 的調和平均，所以只要有一個是 0 或 nan，它就跟著垮掉。',
       en:'F1 is the harmonic mean of precision and recall, so it collapses the moment either one is 0 or nan.'},
      {zh:'MCC 同時用到四個計數，所以只要有一整行或一整列是 0，它的分母就是 0——它也拒絕回答。而 accuracy 照樣若無其事地給了你一個好看的數字。',
       en:'MCC uses all four counts, so a whole row or column of zeros makes its denominator 0 - it refuses too. Accuracy, meanwhile, cheerfully hands you a good-looking number.'}];
    push({rows:k, hot:true, line:k <= 1 ? 2 : (k === 2 ? 4 : (k === 3 ? 3 : 5)),
      cap:{zh:'左欄＝常數，右欄＝模型', en:'left column = the constant, right column = the model'},
      panels:[{lbl:names[k - 1], chips:[chip(fmtN([CONF.const.acc, CONF.const.rec, CONF.const.prec, CONF.const.f1, CONF.const.mcc][k - 1]), 'dim'),
                                        chip(fmtN([CONF.model.acc, CONF.model.rec, CONF.model.prec, CONF.model.f1, CONF.model.mcc][k - 1]), 'act')]}],
      msg:msgs[k - 1]});
  }
  return F.list;
}

/* ================================================================ tab 2: four denominators */
const CODE_FRUIT = [
  'accuracy  = (tp + tn) / everything',
  'precision = tp / (tp + fp)   # of the SIDE you picked',
  'recall    = tp / (tp + fn)   # of ALL the apples',
  'FPR       = fp / (fp + tn)   # of ALL the oranges',
  '',
  '# precision divides by what the MODEL said.',
  '# recall divides by what the TRUTH said.',
  '# accuracy divides by everything - which is why the',
  '# bigger class decides most of it.'
];
function fruitStage(o){
  const sh = [], cw = 2.45, ch = 1.00, x0 = 2.05, y0 = 1.55;
  sh.push(fitT(x0 + cw * .5, y0 - .26, {zh:'說是蘋果', en:'called APPLE'}, {c:COL.tealL, fs:.28}));
  sh.push(fitT(x0 + cw * 1.5, y0 - .26, {zh:'說是橘子', en:'called ORANGE'}, {c:COL.tealL, fs:.28}));
  sh.push(fitT(x0 - .16, y0 + ch * .55, {zh:'真的是蘋果', en:'really apple'}, {anchor:'end', c:COL.tealL, fs:.24}));
  sh.push(fitT(x0 - .16, y0 + ch * 1.55, {zh:'真的是橘子', en:'really orange'}, {anchor:'end', c:COL.tealL, fs:.24}));
  const grid = [[[FRUIT.tp, 'hot'], [FRUIT.fn, 'bad']], [[FRUIT.fp, 'bad'], [FRUIT.tn, 'ok']]];
  for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++){
    let st = grid[r][c][1];
    if (o.lit === 'prec') st = (c === 0) ? grid[r][c][1] : 'done';
    if (o.lit === 'rec') st = (r === 0) ? grid[r][c][1] : 'done';
    if (o.lit === 'fpr') st = (r === 1) ? grid[r][c][1] : 'done';
    sh.push(box(x0 + c * cw, y0 + r * ch, cw, ch, st, String(grid[r][c][0]), .40));
  }
  const rows = [
    ['acc', {zh:'accuracy  = (tp + tn) / 100', en:'accuracy  = (tp + tn) / 100'}, FRUIT.acc, COL.grey],
    ['prec', {zh:'precision = tp / (tp + fp)', en:'precision = tp / (tp + fp)'}, FRUIT.prec, COL.orange],
    ['rec', {zh:'recall    = tp / (tp + fn)', en:'recall    = tp / (tp + fn)'}, FRUIT.rec, COL.tealL],
    ['fpr', {zh:'FPR       = fp / (fp + tn)', en:'FPR       = fp / (fp + tn)'}, FRUIT.fpr, COL.red]];
  rows.forEach((r, i) => {
    const y = 4.15 + i * .46, on = (o.lit === r[0] || o.lit == null);
    sh.push(fitT(2.05, y, r[1], {anchor:'start', c:on ? r[3] : COL.grey, fs:.28}));
    sh.push(fitT(7.70, y, fmtN(r[2]), {anchor:'end', c:on ? r[3] : COL.grey, fs:.28}));
  });
  return caps(sh, o);
}
function fruitFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:fruitStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({line:0,
    cap:{zh:FRUIT.n + ' 顆水果，' + FRUIT.apples + ' 顆是蘋果，一台蘋果偵測器',
         en:FRUIT.n + ' pieces of fruit, ' + FRUIT.apples + ' of them apples, one apple detector'},
    panels:[{lbl:'counts', chips:[chip('tp ' + FRUIT.tp, 'hot'), chip('fp ' + FRUIT.fp, 'bad'),
                                  chip('fn ' + FRUIT.fn, 'bad'), chip('tn ' + FRUIT.tn, 'ok')]}],
    msg:{zh:'偵測器把 ' + (FRUIT.tp + FRUIT.fp) + ' 顆放到蘋果那一邊：' + FRUIT.tp + ' 顆真的是蘋果、' + FRUIT.fp + ' 顆是橘子。剩下的 ' + (FRUIT.tn + FRUIT.fn) + ' 顆放著不管，裡面有它漏掉的 ' + FRUIT.fn + ' 顆蘋果。四個指標就是從這四格拿不同的分母。',
         en:'The detector puts ' + (FRUIT.tp + FRUIT.fp) + ' pieces on the apple side: ' + FRUIT.tp + ' real apples and ' + FRUIT.fp + ' oranges. It leaves ' + (FRUIT.tn + FRUIT.fn) + ' alone, including the ' + FRUIT.fn + ' apples it missed. The four metrics are four different denominators taken from these four cells.'}});
  p({lit:'prec', line:1,
    cap:{zh:'precision：你挑出來的那一「側」裡，對了多少', en:'precision: of the SIDE you picked, how much is right'},
    panels:[{lbl:'precision', chips:[chip(FRUIT.tp + ' / ' + (FRUIT.tp + FRUIT.fp), 'hot'), chip(fmtN(FRUIT.prec), 'act')]}],
    msg:{zh:'precision 的分母是整個左欄，也就是「模型說是蘋果」的那 ' + (FRUIT.tp + FRUIT.fp) + ' 顆。拼字就是記憶法：preci-SI-on 問的是你挑出來的那一側（side）。分母是**模型說了什麼**。',
         en:'The denominator of precision is the whole left column - the ' + (FRUIT.tp + FRUIT.fp) + ' pieces the model called apple. The spelling is the mnemonic: preci-SI-on asks about the SIDE you picked. Its denominator is what the MODEL said.'}});
  p({lit:'rec', line:2,
    cap:{zh:'recall：「所有」蘋果裡，你拿回來幾顆', en:'recall: of ALL the apples, how many came back'},
    panels:[{lbl:'recall', chips:[chip(FRUIT.tp + ' / ' + (FRUIT.tp + FRUIT.fn), 'hot'), chip(fmtN(FRUIT.rec), 'ok')]}],
    msg:{zh:'recall 的分母是整個上列，也就是真的是蘋果的那 ' + (FRUIT.tp + FRUIT.fn) + ' 顆。rec-ALL 問的是所有（all）的蘋果。分母是**真實標籤說了什麼**。它的另一個名字是真陽率 TPR，等一下畫 ROC 時縱軸就是它。',
         en:'The denominator of recall is the whole top row - the ' + (FRUIT.tp + FRUIT.fn) + ' pieces that really are apples. rec-ALL asks about ALL the apples. Its denominator is what the TRUTH said. Its other name is the true positive rate, and it is the y-axis of the ROC curve.'}});
  p({lit:'fpr', line:3,
    cap:{zh:'FPR：「所有」橘子裡，你抓錯幾顆', en:'FPR: of ALL the oranges, how many did you grab'},
    panels:[{lbl:'FPR', chips:[chip(FRUIT.fp + ' / ' + (FRUIT.fp + FRUIT.tn), 'bad'), chip(fmtN(FRUIT.fpr), 'bad')]}],
    msg:{zh:'FPR 是另一個類別上的鏡像：分母是整個下列，真的是橘子的那 ' + (FRUIT.fp + FRUIT.tn) + ' 顆。它和 recall 一樣只看單一類別，所以兩個都不知道兩類各有幾顆——這正是 ROC 兩條軸挑它們的原因。',
         en:'FPR is the mirror on the other class: its denominator is the whole bottom row, the ' + (FRUIT.fp + FRUIT.tn) + ' pieces that really are oranges. Like recall it looks inside a single class, so neither knows how many of each class exist - which is exactly why the ROC curve uses these two.'}});
  p({lit:'acc', line:0,
    cap:{zh:'accuracy：四格全部加起來', en:'accuracy: all four cells at once'},
    panels:[{lbl:'accuracy', chips:[chip(fmtN(FRUIT.acc), 'dim')]},
            {lbl:'say orange every time', chips:[chip(fmtN(FRUIT.lazyAcc), 'bad'), chip('recall ' + fmtN(FRUIT.lazyRec), 'bad')]}],
    msg:{zh:'recall 只數蘋果、FPR 只數橘子、precision 數的是「你」放到蘋果那邊的那幾顆（大小由模型決定），而 accuracy 數的是整籃——那是資料集決定的、模型改不了。100 顆裡 90 顆是橘子，所以 accuracy 大部分的值在開機前就定了：全部說成橘子可以拿 ' + fmtN(FRUIT.lazyAcc) + '，對照真偵測器的 ' + fmtN(FRUIT.acc) + '——而它一顆蘋果都沒找到（recall ' + fmtN(FRUIT.lazyRec) + '）。這就是為什麼有了 accuracy 還需要另外兩個。',
         en:'recall counts apples only, FPR counts oranges only, precision counts the pieces YOU put on the apple side so the model sets its size - and accuracy counts the whole basket, which the dataset fixes and the model cannot change. With 90 oranges in 100, most of accuracy is settled before the detector is switched on: calling everything an orange scores ' + fmtN(FRUIT.lazyAcc) + ' against the real one at ' + fmtN(FRUIT.acc) + ' - while finding no apples at all (recall ' + fmtN(FRUIT.lazyRec) + '). That is why precision and recall exist alongside it.'}});
  return F.list;
}

/* ================================================================ tab: why those axes */
const CODE_AXES = [
  'recall = tp / (tp + fn)    # only rows that really are POSITIVE',
  'FPR    = fp / (fp + tn)    # only rows that really are NEGATIVE',
  '',
  '# neither denominator mixes the classes, so neither axis',
  '# knows how many of each class there are.  Up is benefit,',
  '# right is cost, and the diagonal is guessing at random.'
];
function axesStage(o){
  const sh = [], base = 4.55, maxh = 2.45, bw = .56, gap = .22;
  BALANCE.forEach((b, i) => {
    if (i > o.upto) return;
    const x = 1.45 + i * 2.05;
    barRow(sh, x, base, bw, gap, [b.auc, b.ap], ['ok', 'hot'],
           [fmtN(b.auc), fmtN(b.ap)], maxh, 1.0);
    sh.push(fitT(x + bw + gap / 2, base + .34, {zh:'留 ' + b.keep + ' 個負樣本', en:b.keep + ' negatives'},
                 {c:COL.grey, fs:.24}));
    sh.push(fitT(x + bw + gap / 2, base + .70, {zh:'正樣本 ' + Math.round(100 * b.rate) + '%', en:Math.round(100 * b.rate) + '% positive'},
                 {c:COL.grey, fs:.24}));
  });
  sh.push(fitT(2.40, 1.25, {zh:'ROC-AUC', en:'ROC-AUC'}, {c:COL.tealL, fs:.30}));
  sh.push(fitT(6.40, 1.25, {zh:'average precision', en:'average precision'}, {c:COL.orange, fs:.30}));
  return caps(sh, o);
}
function axesFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:axesStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const first = BALANCE[0], last = BALANCE[BALANCE.length - 1];
  p({upto:0, line:1,
    cap:{zh:'同一個模型、同一組排名，只有負樣本被隨機丟掉', en:'one model, one ranking, only negatives thrown away at random'},
    panels:[{lbl:'all negatives', chips:[chip('ROC-AUC ' + fmtN(first.auc), 'ok'), chip('AP ' + fmtN(first.ap), 'bad')]}],
    msg:{zh:'起點：1,980 個負樣本、正樣本只佔 1%。ROC-AUC ' + fmtN(first.auc) + '、average precision ' + fmtN(first.ap) + '。接下來每一步都只是把負樣本隨機丟掉一些，模型完全沒動。每根長條是 50 次抽樣的平均。',
         en:'Starting point: 1,980 negatives and positives at 1% of the rows. ROC-AUC ' + fmtN(first.auc) + ', average precision ' + fmtN(first.ap) + '. Every step from here only throws negatives away at random; the model never changes. Each bar is the mean over 50 draws.'}});
  for (let i = 1; i < BALANCE.length; i++){
    const b = BALANCE[i];
    p({upto:i, line:0,
      cap:{zh:'正樣本比例變成 ' + Math.round(100 * b.rate) + '%', en:'positives are now ' + Math.round(100 * b.rate) + '% of the rows'},
      panels:[{lbl:'ROC-AUC', chips:[chip(fmtN(b.auc), 'ok')]},
              {lbl:'average precision', chips:[chip(fmtN(b.ap), 'hot')]}],
      msg:{zh:'留 ' + b.keep + ' 個負樣本：ROC-AUC ' + fmtN(b.auc) + '（幾乎沒動），average precision ' + fmtN(b.ap) + '。recall 和 FPR 的分母各自只數一個類別，所以兩條軸都不知道類別比例；precision 的分母裡同時有 fp 和 tp，所以它完全感覺得到。',
           en:'Keeping ' + b.keep + ' negatives: ROC-AUC ' + fmtN(b.auc) + ' (barely moved), average precision ' + fmtN(b.ap) + '. The denominators of recall and FPR each count one class only, so neither ROC axis knows the balance; precision has fp and tp in the same denominator, so it feels all of it.'}});
  }
  p({upto:BALANCE.length - 1, line:4,
    cap:{zh:'這個「幾乎不受影響」同時是該用它和不該用它的理由', en:'that near-invariance is both the reason to use it and the reason not to'},
    panels:[{lbl:'ROC-AUC moved by', chips:[chip(String(BAL_SPREAD.auc), 'ok')]},
            {lbl:'avg precision moved by', chips:[chip(String(BAL_SPREAD.ap), 'hot')]}],
    msg:{zh:'正樣本從 1% 變成 ' + Math.round(100 * last.rate) + '%，平均 ROC-AUC 只移動了 ' + BAL_SPREAD.auc + '，平均 average precision 移動了 ' + BAL_SPREAD.ap + '——遠了 ' + BAL_SPREAD.ratio + ' 倍。當測試集的類別比例跟線上不一樣時，你要的是會搬家的那個數字；當「正樣本很稀少」本身就是問題時，你要的剛好不是它。',
         en:'The positives go from 1% to ' + Math.round(100 * last.rate) + '% while the mean ROC-AUC moves by ' + BAL_SPREAD.auc + ' and the mean average precision moves by ' + BAL_SPREAD.ap + ', ' + BAL_SPREAD.ratio + ' times further. When your test set was sampled with a different balance from production you want the number that transfers; when the rarity of the positives IS the problem, that is exactly the number you do not want.'}});
  return F.list;
}

/* ================================================================ tab 2: the threshold */
const CODE_CUT = [
  'scores = model.predict_proba(X)      # one number per row',
  'pred   = (scores >= threshold)       # the decision',
  '',
  '# nothing is retrained between the rows below.  The model is',
  '# fixed; only the threshold moves.  0.5 is a choice, and on',
  '# this data the highest score any row got was ' + CONF.topScore + '.'
];
function cutStage(o){
  const sh = [], P = new Panel(.95, .95, 5.35, 3.55);
  P.frame(sh);
  const xs = SWEEP.map((r, i) => i / (SWEEP.length - 1));
  const series = [['recall', SWEEP.map(r => r.rec), 'ok'],
                  ['precision', SWEEP.map(r => r.prec == null ? 0 : r.prec), 'hot'],
                  ['F1', SWEEP.map(r => r.f1), 'act'],
                  ['accuracy', SWEEP.map(r => r.acc), 'done']];
  series.forEach(s => polyline(sh, P, xs.map((x, i) => [x, s[1][i]]), s[2], .05));
  SWEEP.forEach((r, i) => {
    if (i % 2 === 0) sh.push(fitT(P.px([xs[i], 0])[0], P.y0 + P.h + .34, String(r.t), {c:COL.grey, fs:.22}));
  });
  const cur = SWEEP[o.i];
  const cx = P.px([xs[o.i], 0])[0];
  sh.push(S.e(cx, P.y0, cx, P.y0 + P.h, {s:'hot', arrow:false, w:.04}));
  sh.push(fitT(cx, P.y0 - .18, {zh:'切在 ' + cur.t, en:'cut at ' + cur.t}, {c:COL.orange, fs:.26}));
  sh.push(fitT(P.x0 + P.w / 2, P.y0 + P.h + .78, {zh:'同一組分數，只有切點在動',
    en:'the same scores, only the cut moves'}, {c:COL.grey, fs:.24}));
  side(sh, 6.70, 1.25, [
    [{zh:'標出來的列數 ' + cur.flag, en:'rows flagged ' + cur.flag}, COL.pale, .28],
    [{zh:'其中真的是正樣本 ' + cur.tp, en:'of them really positive ' + cur.tp}, COL.orange, .28],
    [{zh:'recall ' + fmtN(cur.rec), en:'recall ' + fmtN(cur.rec)}, COL.tealL, .28],
    [{zh:'precision ' + fmtN(cur.prec), en:'precision ' + fmtN(cur.prec)}, COL.orange, .28],
    [{zh:'F1 ' + fmtN(cur.f1), en:'F1 ' + fmtN(cur.f1)}, PURPLE, .28],
    [{zh:'accuracy ' + fmtN(cur.acc), en:'accuracy ' + fmtN(cur.acc)}, COL.grey, .28]], .46);
  return caps(sh, o);
}
function cutFrames(){
  const F = new Frames();
  const bestF1 = SWEEP.reduce((a, b) => b.f1 > a.f1 ? b : a);
  const msgs = [
    {zh:'0.5 是大多數程式庫的預設值，不是任何人為這份資料選的。模型在 2,000 列裡只標了 ' + SWEEP[0].flag + ' 列，recall 0.0000——看起來完全沒用。',
     en:'0.5 is a library default, not a decision anyone made about this data. The model flags ' + SWEEP[0].flag + ' row out of 2,000 and recall is 0.0000 - it looks useless.'},
    {zh:'切到 0.3 還是一樣。最高的那個分數只有 ' + CONF.topScore + '，所以 0.5 和 0.3 之間根本沒有東西。',
     en:'At 0.3 nothing changes. The highest score any row received is ' + CONF.topScore + ', so there is nothing between 0.5 and 0.3 to find.'},
    {zh:'切到 0.2，第一個真的正樣本出現了：precision 0.3333，recall 0.0500。模型一直都知道這列，只是沒有人問到那麼低。',
     en:'At 0.2 the first real positive appears: precision 0.3333, recall 0.0500. The model knew about this row all along; nobody asked that low.'},
    {zh:'0.15 多標了一列而且標錯，所以 precision 掉下來。這就是這條曲線的形狀：每往下切一格，你用 precision 換 recall。',
     en:'0.15 flags one more row and gets it wrong, so precision drops. That is the shape of the whole curve: every step down trades precision for recall.'},
    {zh:'0.10 是 F1 最高的地方：' + fmtN(bestF1.f1) + '，標 ' + bestF1.flag + ' 列抓到 ' + bestF1.tp + ' 個。注意 accuracy 是 ' + fmtN(bestF1.acc) + '，比什麼都不抓的時候還低。',
     en:'0.10 is where F1 peaks: ' + fmtN(bestF1.f1) + ', flagging ' + bestF1.flag + ' rows and catching ' + bestF1.tp + '. Note accuracy is ' + fmtN(bestF1.acc) + ', lower than when it caught nothing.'},
    {zh:'0.05 抓到一半的正樣本，但要看 88 列。你的審查量能決定了這裡該停在哪，不是 F1。',
     en:'0.05 catches half the positives but asks you to review 88 rows. Your review capacity decides where to stop here, not F1.'},
    {zh:'0.03：recall 0.7000，precision 0.0636。220 列裡有 14 個真的。',
     en:'0.03: recall 0.7000, precision 0.0636. Of 220 flagged rows, 14 are real.'},
    {zh:'0.02：recall 0.8000，而 accuracy 已經掉到 0.7995——低於「全部回答 0」的 0.9900 一大截。',
     en:'0.02: recall 0.8000, and accuracy has fallen to 0.7995 - far below the 0.9900 of answering 0 everywhere.'},
    {zh:'0.01 抓到 20 個正樣本中的 19 個（recall 0.9500），代價是標了 845 列、accuracy 0.5865。accuracy 隨著模型變有用而一路下降，因為有用就是拿 true negative 去換 true positive。',
     en:'0.01 catches 19 of the 20 positives (recall 0.9500), at the cost of flagging 845 rows and an accuracy of 0.5865. Accuracy falls as the model becomes useful, because being useful means spending true negatives to buy true positives.'}];
  SWEEP.forEach((r, i) => {
    F.push({shapes:cutStage({i:i, cap:{zh:'模型沒有重訓，動的只有門檻',
                                       en:'the model is not retrained; only the threshold moves'}}),
      panels:[{lbl:'threshold', chips:[chip(String(r.t), 'hot')]},
              {lbl:'flagged / caught', chips:[chip(String(r.flag), 'act'), chip(String(r.tp), 'ok')]},
              {lbl:'precision / recall', chips:[chip(fmtN(r.prec), 'dim'), chip(fmtN(r.rec), 'dim')]}],
      line:1, msg:msgs[i], view:VIEW});
  });
  return F.list;
}

/* ================================================================ tab 3: the two curves */
const CODE_CURVE = [
  'roc:  x = fp / (fp + tn)     # divided by ALL the negatives',
  '      y = tp / (tp + fn)',
  '',
  'pr:   x = tp / (tp + fn)',
  '      y = tp / (tp + fp)     # divided by what you FLAGGED',
  '',
  '# same scores, same labels, two different denominators.'
];
function curveStage(o){
  const sh = [];
  const panels = [['roc', .70, {zh:'ROC 曲線', en:'the ROC curve'}, {zh:'偽陽性率', en:'false positive rate'}, {zh:'recall', en:'recall'}],
                  ['pr', 5.35, {zh:'precision-recall 曲線', en:'the precision-recall curve'}, {zh:'recall', en:'recall'}, {zh:'precision', en:'precision'}]];
  panels.forEach(([key, x0, title, xlab, ylab], pi) => {
    if (o.only != null && o.only !== pi) return;
    const P = new Panel(x0, 1.05, 3.55, 3.30);
    P.frame(sh);
    sh.push(fitT(P.x0 + P.w / 2, P.y0 - .30, title, {c:COL.tealL, fs:.28}));
    sh.push(fitT(P.x0, P.y0 - .02, ylab, {anchor:'start', c:COL.grey, fs:.22}));
    sh.push(fitT(P.x0 + P.w / 2, P.y0 + P.h + .36, xlab, {c:COL.grey, fs:.22}));
    if (key === 'roc') polyline(sh, P, [[0, 0], [1, 1]], 'ghost', .03);
    (o.show || []).forEach(nm => {
      polyline(sh, P, CURVE[nm][key], nm === 'broad' ? 'ok' : 'hot', .055);
    });
    (o.show || []).forEach((nm, k) => {
      const v = key === 'roc' ? CURVE[nm].auc : CURVE[nm].ap;
      const yy = key === 'roc' ? (nm === 'broad' ? .22 : .10) : (nm === 'broad' ? .80 : .68);
      sh.push(fitT(P.x0 + P.w - .12, P.px([0, yy])[1], nm + ' ' + fmtN(v),
                   {anchor:'end', c:nm === 'broad' ? COL.tealL : COL.orange, fs:.26}));
    });
  });
  if (o.topk){
    const ks = [10, 20, 50, 100];
    ks.forEach((k, i) => {
      const x = 1.30 + i * 2.05;
      barRow(sh, x, 4.30, .55, .22, [CURVE.broad.topk[String(k)], CURVE.sharp.topk[String(k)]],
             ['ok', 'hot'], [String(CURVE.broad.topk[String(k)]), String(CURVE.sharp.topk[String(k)])],
             2.55, 20);
      sh.push(fitT(x + .66, 4.62, {zh:'前 ' + k + ' 名', en:'top ' + k}, {c:COL.grey, fs:.24}));
    });
    sh.push(fitT(4.9, 1.00, {zh:'20 個正樣本裡，有幾個排在分數最高的 k 列內',
      en:'how many of the 20 positives are in the k highest scores'}, {c:COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function curveFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:curveStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({show:['broad'], only:0, line:0,
    cap:{zh:CURVE.pos + ' 個正樣本、' + CURVE.neg + ' 個負樣本', en:CURVE.pos + ' positives, ' + CURVE.neg + ' negatives'},
    panels:[{lbl:'broad ROC-AUC', chips:[chip(fmtN(CURVE.broad.auc), 'ok')]}],
    msg:{zh:'broad 這個模型把整體分得很開，ROC-AUC ' + fmtN(CURVE.broad.auc) + '。照這個數字，它接近完美。',
         en:'The model called broad separates the population well: ROC-AUC ' + fmtN(CURVE.broad.auc) + '. By that number it is close to perfect.'}});
  p({show:['broad', 'sharp'], only:0, line:0,
    cap:{zh:'同一批標籤，另一個模型', en:'the same labels, a second model'},
    panels:[{lbl:'ROC-AUC', chips:[chip('broad ' + fmtN(CURVE.broad.auc), 'ok'), chip('sharp ' + fmtN(CURVE.sharp.auc), 'bad')]}],
    msg:{zh:'sharp 只把一半的正樣本排在所有負樣本前面，另一半埋在很低的地方（那十個只贏過 17%–26% 的負樣本、平均 22%，所以 ROC-AUC 是 (10 x 1.0000 + 10 x 0.2163) / 20），所以 ROC-AUC 只有 ' + fmtN(CURVE.sharp.auc) + '。到這裡為止，broad 大勝。',
         en:'sharp puts half the positives above every negative and buries the other half low - those ten outrank 17% to 26% of the negatives, 22% on average, so its ROC-AUC works out to (10 x 1.0000 + 10 x 0.2163) / 20 = ' + fmtN(CURVE.sharp.auc) + '. So far broad wins easily.'}});
  p({show:['broad', 'sharp'], line:4,
    cap:{zh:'換一條曲線，順序整個反過來', en:'change the curve and the order flips'},
    panels:[{lbl:'ROC-AUC', chips:[chip(fmtN(CURVE.broad.auc), 'ok'), chip(fmtN(CURVE.sharp.auc), 'bad')]},
            {lbl:'avg precision', chips:[chip(fmtN(CURVE.broad.ap), 'bad'), chip(fmtN(CURVE.sharp.ap), 'ok')]}],
    msg:{zh:'average precision 給 sharp ' + fmtN(CURVE.sharp.ap) + '，給 broad 只有 ' + fmtN(CURVE.broad.ap) + '。兩個數字都沒算錯，差別在分母：ROC 的 x 軸除以 ' + CURVE.neg + ' 個負樣本，而 broad 要抓滿 20 個得降到 0.5897，有 ' + BAL_SPREAD.above + ' 個負樣本在那條線之上——除以 ' + CURVE.neg + ' 只動 0.0682；同樣這 ' + BAL_SPREAD.above + ' 個，佔你為此標出來的 ' + BAL_SPREAD.flagged + ' 列的 87%。',
         en:'Average precision gives sharp ' + fmtN(CURVE.sharp.ap) + ' and broad only ' + fmtN(CURVE.broad.ap) + '. Neither is miscomputed; the difference is one denominator. To catch all 20 positives broad has to come down to a score of 0.5897, and ' + BAL_SPREAD.above + ' negatives sit above that line. The ROC x-axis divides those ' + BAL_SPREAD.above + ' by all ' + CURVE.neg + ' negatives, so they move it by 0.0682. Precision divides them by the ' + BAL_SPREAD.flagged + ' rows you flagged to get there: 87% of your review queue is wrong, and only precision says so.'}});
  p({topk:true, line:4,
    cap:{zh:'把曲線換成實際要做的決定', en:'replace the curves with the decision you actually make'},
    panels:[{lbl:'top 10', chips:[chip('broad ' + CURVE.broad.topk['10'], 'bad'), chip('sharp ' + CURVE.sharp.topk['10'], 'ok')]},
            {lbl:'top 100', chips:[chip('broad ' + CURVE.broad.topk['100'], 'ok'), chip('sharp ' + CURVE.sharp.topk['100'], 'bad')]}],
    msg:{zh:'如果分析師一天只能看 10 列，sharp 給他 ' + CURVE.sharp.topk['10'] + ' 個真案件，broad 給他 ' + CURVE.broad.topk['10'] + ' 個。如果可以看 100 列，broad 給 ' + CURVE.broad.topk['100'] + '、sharp 還是 ' + CURVE.sharp.topk['100'] + '。先決定一天看幾列，再決定看哪個指標——順序反過來就會選錯模型。',
         en:'If an analyst can review 10 rows a day, sharp hands them ' + CURVE.sharp.topk['10'] + ' real cases and broad hands them ' + CURVE.broad.topk['10'] + '. At 100 rows a day broad gives ' + CURVE.broad.topk['100'] + ' and sharp still gives ' + CURVE.sharp.topk['100'] + '. Decide how many rows a day you can review, then decide which metric to read - the other order picks the wrong model.'}});
  return F.list;
}

/* ================================================================ tab 4: folds */
const CODE_FOLDS = [
  'idx = np.arange(n)',
  'rng.shuffle(idx)                     # labels never looked at',
  'folds = np.array_split(idx, k)',
  '',
  '# stratified: do the same thing once per class, so every fold',
  '# keeps the overall positive rate to within one row.'
];
function foldsStage(o){
  const sh = [], cw = .72, ch = .70;
  const rows = [[{zh:'先洗牌再切', en:'shuffle, then cut'}, FOLDS.plain, o.markPlain],
                [{zh:'每個類別各自切', en:'cut each class'}, FOLDS.strat, false]];
  rows.forEach(([nm, counts, mark], r) => {
    if (r === 1 && !o.showStrat) return;
    const y0 = 1.25 + r * 1.30;
    sh.push(fitT(1.32, y0 - .20, nm, {anchor:'start', c:COL.tealL, fs:.26}));
    counts.forEach((c, i) => {
      const empty = (c === 0);
      sh.push(box(1.32 + i * (cw + .08), y0, cw, ch,
                  (mark && empty) ? 'bad' : (empty ? 'soft' : 'ok'), String(c), .34));
    });
  });
  if (o.hist){
    sh.push(fitT(4.9, 4.30, {zh:'200 次不同的洗牌裡，有 ' + FOLDS.runs + ' 次（' + Math.round(100 * FOLDS.runs / FOLDS.trials) + '%）至少有一折是空的',
      en:FOLDS.runs + ' of ' + FOLDS.trials + ' shuffles (' + Math.round(100 * FOLDS.runs / FOLDS.trials) + '%) produce at least one fold with no positives'},
      {c:COL.red, fs:.28}));
  }
  return caps(sh, o);
}
function foldsFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:foldsStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({line:1,
    cap:{zh:'2,000 列、20 個正樣本，切成 10 折', en:'2,000 rows, 20 positives, cut into 10 folds'},
    panels:[{lbl:'positives per fold', chips:FOLDS.plain.map(c => chip(String(c), c === 0 ? 'bad' : 'dim'))}],
    msg:{zh:'把列號洗牌再切成 10 份，是最常見的寫法。它從頭到尾沒有看過標籤。',
         en:'Shuffle the row numbers and cut them into ten blocks - the most common way to do it. It never looks at the labels.'}});
  p({markPlain:true, line:1,
    cap:{zh:'兩折一個正樣本都沒有', en:'two folds have no positives at all'},
    panels:[{lbl:'recall on an empty fold', chips:[chip('0 / 0', 'bad'), chip('nan', 'bad')]}],
    msg:{zh:'紅色那兩折裡沒有任何正樣本，所以 recall 是 0/0。np.mean 會讓整個交叉驗證分數變成 nan；np.nanmean 會讓那一折安靜地從平均裡消失——兩種都不會報錯。',
         en:'Those two folds contain no positives, so recall is 0/0. np.mean turns the whole cross-validation score into nan; np.nanmean quietly drops the fold from the average. Neither raises.'}});
  p({markPlain:true, hist:true, line:1,
    cap:{zh:'這不是運氣不好', en:'this is not bad luck'},
    panels:[{lbl:'shuffles with an empty fold', chips:[chip(FOLDS.runs + ' / ' + FOLDS.trials, 'bad')]}],
    msg:{zh:'換隨機種子重跑 200 次，' + FOLDS.runs + ' 次至少有一折是空的。20 個正樣本分到 10 折，平均每折 2 個，而 2 很容易是 0。',
         en:'Re-run with 200 different seeds and ' + FOLDS.runs + ' of them have at least one empty fold. Twenty positives over ten folds averages two per fold, and two is easily zero.'}});
  p({showStrat:true, line:5,
    cap:{zh:'分層：每個類別各自切一次', en:'stratified: cut each class separately'},
    panels:[{lbl:'stratified', chips:FOLDS.strat.map(c => chip(String(c), 'ok'))}],
    msg:{zh:'對 0 和 1 各做一次同樣的切法，每折就都拿到 2 個正樣本。這多寫一行，而且它移除的不是「誤差」而是「整類失敗」——不是讓分數更準，是讓分數存在。',
         en:'Do the same cut once for the zeros and once for the ones, and every fold gets two positives. It is one extra line, and what it removes is not error but a whole failure mode - it does not make the score more accurate, it makes the score exist.'}});
  return F.list;
}

/* ================================================================ tab 5: leakage */
const CODE_LEAK = [
  'for f in folds:',
  '    train = rows not in f',
  '    # everything the model touches must be fitted HERE:',
  '    cols       = select_columns(X[train], y[train])',
  '    mean, std  = X[train].mean(0), X[train].std(0)',
  '    model      = fit(X[train][:, cols], y[train])',
  '',
  '# anything fitted before this loop has seen f.'
];
const LEAK_META = [
  {id:'group', zh:'同一個客戶同時出現在兩邊', en:'the same customer on both sides'},
  {id:'select', zh:'切之前就先挑好欄位', en:'columns picked before the split'},
  {id:'scale', zh:'用全部的列算平均與標準差', en:'scaled using every row'}
];
function leakStage(o){
  const sh = [], base = 4.35, maxh = 2.45, bw = .62, gap = .24;
  LEAK_META.forEach((m, i) => {
    if (i > o.upto) return;
    const d = LEAK[i], x = 1.30 + i * 2.70;
    barRow(sh, x, base, bw, gap, [d.bad, d.good], ['bad', 'ok'],
           [fmtN(d.bad), fmtN(d.good)], maxh, 1.0);
    sh.push(fitT(x + bw + gap / 2, base + .36, {zh:m.zh, en:m.en}, {c:COL.grey, fs:.24}));
  });
  const y50 = base - maxh * .5;
  sh.push(S.e(1.15, y50, 8.65, y50, {s:'ghost', arrow:false, w:.03}));
  sh.push(fitT(8.80, y50, '0.5', {anchor:'start', c:COL.grey, fs:.24}));
  sh.push(fitT(2.60, 1.30, {zh:'流程報出來的', en:'what the pipeline reports'}, {c:COL.red, fs:.26}));
  sh.push(fitT(6.20, 1.30, {zh:'同一個模型真正值多少', en:'what the model is really worth'}, {c:COL.tealL, fs:.26}));
  return caps(sh, o);
}
function leakFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:leakStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'150 個客戶，每人 4 列，標籤屬於客戶', en:'150 customers, 4 rows each, the label belongs to the customer'},
    panels:[{lbl:'random folds', chips:[chip(fmtN(LEAK[0].bad), 'bad')]},
            {lbl:'group folds', chips:[chip(fmtN(LEAK[0].good), 'ok')]}],
    msg:{zh:'分類器是「抄最近那一列的標籤」，它沒有任何參數可以學。隨機切折時它拿到 ' + fmtN(LEAK[0].bad) + '，因為每一列測試資料的另外三列兄弟就躺在訓練集裡。照客戶切，同一個分類器是 ' + fmtN(LEAK[0].good) + '——低於擲銅板。',
         en:'The classifier copies the label of the nearest row; it has no parameters to learn. With random folds it scores ' + fmtN(LEAK[0].bad) + ', because the three siblings of every test row are sitting in the training set. Split by customer and the same classifier scores ' + fmtN(LEAK[0].good) + ' - below a coin flip.'}});
  p({upto:1, line:3,
    cap:{zh:'60 列、400 欄純雜訊、擲銅板的標籤', en:'60 rows, 400 columns of noise, a coin-flip label'},
    panels:[{lbl:'picked on all rows', chips:[chip(fmtN(LEAK[1].bad), 'bad')]},
            {lbl:'picked inside each fold', chips:[chip(fmtN(LEAK[1].good), 'ok')]}],
    msg:{zh:'這張表裡沒有任何可學的東西。先用全部 60 列挑出「最相關的 5 欄」再做交叉驗證，平均拿到 ' + fmtN(LEAK[1].bad) + '；把挑欄位搬進每一折、只用該折的訓練列，拿到 ' + fmtN(LEAK[1].good) + '（兩個都是 20 張獨立抽出來的雜訊表的平均）。挑選看過每一個標籤，而在它之後才開始的交叉驗證救不回來。',
         en:'There is nothing in this table to find. Pick the five most correlated columns using all 60 rows and then cross-validate: ' + fmtN(LEAK[1].bad) + '. Move the selection inside each fold, using only that fold training rows: ' + fmtN(LEAK[1].good) + '. Both are means over 20 independently drawn noise tables. The selection saw every label, and a cross-validation that starts after it cannot undo it.'}});
  p({upto:2, line:4,
    cap:{zh:'三種洩漏不是一樣嚴重的', en:'the three are not equally bad'},
    panels:[{lbl:'scaler, leaking vs honest', chips:[chip(fmtN(LEAK[2].bad), 'dim'), chip(fmtN(LEAK[2].good), 'dim')]},
            {lbl:'largest score gap', chips:[chip(String(LEAK[2].gap), 'dim')]}],
    msg:{zh:'第三種是「用全部的列算平均與標準差」。注意這裡換資料了：不是 (b) 那張純雜訊表，而是另外 60 列、8 個欄位而且真的有訊號，模型也換成 logistic regression；兩個數字之間唯一的差別是平均與標準差在哪裡被 fit 出來。它幾乎沒有影響：兩邊 accuracy 一樣，分數最大差距只有 ' + LEAK[2].gap + '。平均和標準差是整欄的兩個摘要值，五折一次留出來的那 11 到 13 列幾乎推不動它們。但這只在「每一列可以互換」時成立——當測試列是另一個時間段或另一群人，那個全域平均就是關於未來的資訊。前兩種才是把雜訊變成結果的那兩種。',
         en:'The third is taking the column means and standard deviations from every row. Note the data changed: this is not the pure-noise table from (b) but a different 60 rows - 8 columns with real signal - scored by logistic regression, and the only thing differing between the two numbers is where the mean and standard deviation were fitted. It does almost nothing: the two accuracies are identical and the largest gap between any two scores is ' + LEAK[2].gap + '. A mean and a standard deviation are two summaries of a whole column, so the 11 to 13 rows a fifth holds out barely move them - but that only holds while the rows are interchangeable. When the test rows are a later period or a different population, the global mean is a fact about the future. The first two are the ones that turn noise into a result.'}});
  return F.list;
}

/* ================================================================ tab 6: the two problems */
const CODE_493 = [
  'def sort_count(a):',
  '    left, cl = sort_count(a[:mid])',
  '    right, cr = sort_count(a[mid:])',
  '    j = 0',
  '    for v in left:                  # both halves are sorted',
  '        while j < len(right) and v > 2 * right[j]:',
  '            j += 1                  # j never goes back',
  '        count += j                  # a whole block at once',
  '    merge(left, right)              # counting is a separate pass'
];
const CODE_169 = [
  'count, candidate = 0, None',
  'for v in nums:',
  '    if count == 0:',
  '        candidate = v',
  '    count += 1 if v == candidate else -1',
  'return candidate',
  '',
  '# a value holding more than half the array cannot be',
  '# cancelled away, however the others are arranged.'
];
function lc493Stage(o){
  const sh = [];
  sh.push(fitT(4.9, .95, {zh:'nums = [' + LC493.nums.join(', ') + ']，數出 i < j 且 nums[i] > 2 * nums[j] 的對數',
    en:'nums = [' + LC493.nums.join(', ') + '], count the pairs i < j with nums[i] > 2 * nums[j]'},
    {c:COL.tealL, fs:.28}));
  const cw = .80, ch = .72;
  sh.push(fitT(1.70, 1.42, {zh:'左半（已排序）', en:'left half, sorted'}, {anchor:'start', c:COL.grey, fs:.24}));
  LC493.left.forEach((v, i) => {
    const st = (o.step != null && i === o.step) ? 'hot' : (o.step != null && i < o.step ? 'done' : 'idle');
    sh.push(box(1.70 + i * (cw + .12), 1.60, cw, ch, st, String(v), .34));
  });
  sh.push(fitT(1.70, 2.52, {zh:'右半（已排序）', en:'right half, sorted'}, {anchor:'start', c:COL.grey, fs:.24}));
  const st = o.step != null ? LC493.steps[o.step] : null;
  LC493.right.forEach((v, i) => {
    let cls = 'idle';
    if (st && i < st.j) cls = 'act';
    sh.push(box(1.70 + i * (cw + .12), 2.70, cw, ch, cls, String(v), .34));
  });
  if (st){
    const jx = 1.70 + st.j * (cw + .12);
    sh.push(S.e(jx, 3.70, jx, 3.48, {s:'hot', w:.05}));
    sh.push(fitT(jx, 4.00, 'j = ' + st.j, {c:COL.orange, fs:.26}));
    side(sh, 5.95, 1.75, [
      [{zh:'v = ' + st.v, en:'v = ' + st.v}, COL.orange, .30],
      [{zh:'j 從 ' + st.j0 + ' 走到 ' + st.j, en:'j moves from ' + st.j0 + ' to ' + st.j}, COL.pale, .28],
      [{zh:'這一輪加 ' + st.add + ' 對', en:'this row adds ' + st.add + ' pairs'}, COL.tealL, .28],
      [{zh:'跨兩半累計 ' + st.cnt, en:'running cross count ' + st.cnt}, PURPLE, .28]], .48);
  }
  return caps(sh, o);
}
function lc169Stage(o){
  const sh = [], cw = .80, ch = .72;
  sh.push(fitT(4.9, 1.05, {zh:'nums = [' + LC169.nums.join(', ') + ']，找出出現超過一半的那個值',
    en:'nums = [' + LC169.nums.join(', ') + '], find the value appearing more than half the time'},
    {c:COL.tealL, fs:.28}));
  LC169.nums.forEach((v, i) => {
    const cls = i === o.step ? 'hot' : (i < o.step ? 'done' : 'idle');
    sh.push(box(1.45 + i * (cw + .14), 1.75, cw, ch, cls, String(v), .34));
  });
  const s = LC169.steps[o.step];
  side(sh, 1.45, 3.15, [
    [{zh:'候選 candidate = ' + s.cand, en:'candidate = ' + s.cand}, COL.orange, .30],
    [{zh:'計數 count = ' + s.c, en:'count = ' + s.c}, PURPLE, .30],
    [{zh:s.c === 0 ? '歸零了，下一個值會接手當候選' : (s.v === s.cand ? '跟候選一樣，計數加一' : '跟候選不同，計數減一'),
      en:s.c === 0 ? 'back to zero: the next value takes over as candidate'
                   : (s.v === s.cand ? 'same as the candidate, so the count goes up'
                                     : 'different from the candidate, so the count goes down')}, COL.pale, .28]], .50);
  if (o.done){
    sh.push(fitT(4.9, 4.70, {zh:'答案 ' + LC169.ans + '，它在 ' + LC169.nums.length + ' 列裡出現 ' + LC169.count + ' 次',
      en:'answer ' + LC169.ans + ', which appears ' + LC169.count + ' times in ' + LC169.nums.length + ' rows'},
      {c:COL.tealL, fs:.30}));
    sh.push(fitT(4.9, 5.12, {zh:'沒有「保證存在多數」時：[' + LC169.noMaj.join(', ') + '] 它仍然回答 ' + LC169.noMajSays,
      en:'without the promise that a majority exists: [' + LC169.noMaj.join(', ') + '] still returns ' + LC169.noMajSays},
      {c:COL.red, fs:.28}));
  }
  return caps(sh, o);
}
function lcFrames(){
  const F = new Frames();
  if (curV() === 0){
    F.push({shapes:lc493Stage({cap:{zh:'先把兩半各自排好，再數跨兩半的對數',
      en:'sort each half first, then count the pairs that cross between them'}}),
      panels:[{lbl:'answer', chips:[chip(String(LC493.total), 'dim')]}], line:0,
      msg:{zh:'暴力法要看 n(n-1)/2 對。merge sort 的做法是：兩半各自排好之後，左邊的 v 只要大於 2 * right[j]，right 裡 j 之前的每一個都算一對——一次比較結算一整塊。',
           en:'Brute force looks at n(n-1)/2 pairs. The merge-sort move is: once both halves are sorted, if v is greater than 2 * right[j] then every element before j on the right also qualifies - one comparison settles a whole block.'}, view:VIEW});
    LC493.steps.forEach((s, i) => {
      F.push({shapes:lc493Stage({step:i, cap:{zh:'j 只會往前走，整趟加起來最多走過右半一次',
        en:'j only moves forward; across the whole pass it crosses the right half once'}}),
        panels:[{lbl:'v', chips:[chip(String(s.v), 'hot')]},
                {lbl:'j', chips:[chip(String(s.j0) + ' -> ' + String(s.j), 'act')]},
                {lbl:'cross count', chips:[chip(String(s.cnt), 'ok')]}],
        line:5 + Math.min(i, 2),
        msg:{zh:'v = ' + s.v + '：j 從 ' + s.j0 + ' 推到 ' + s.j + '，一次加 ' + s.add + ' 對，累計 ' + s.cnt + '。注意 j 沒有退回去過——這就是為什麼整趟是 O(n) 而不是 O(n²)。',
             en:'v = ' + s.v + ': j advances from ' + s.j0 + ' to ' + s.j + ', adding ' + s.add + ' pairs at once, running total ' + s.cnt + '. Note that j never went back - that is why the whole pass is O(n) and not O(n squared).'}, view:VIEW});
    });
    F.push({shapes:lc493Stage({step:LC493.steps.length - 1, cap:{zh:'跨兩半 ' + LC493.cross + ' 對，加上兩半內部的，總共 ' + LC493.total,
      en:LC493.cross + ' pairs cross the halves; with the ones inside each half the answer is ' + LC493.total}}),
      panels:[{lbl:'cross / total', chips:[chip(String(LC493.cross), 'act'), chip(String(LC493.total), 'ok')]},
              {lbl:'brute force', chips:[chip(String(LC493.brute), 'dim')]}],
      line:8,
      msg:{zh:'計數一定要自己走一趟。merge 的前進條件是 left[i] <= right[j]，而我們要數的是 left[i] > 2 * right[j]——兩個不同的順序。把計數塞進 merge 裡會少算，而且它照樣回傳一個數字。這天的 ROC-AUC 是同一件事：它就是「有幾對正樣本排在負樣本前面」。',
           en:'The counting has to be its own pass. The merge advances on left[i] <= right[j] while the pairs we want need left[i] > 2 * right[j] - two different orders. Folding the count into the merge undercounts, and still returns a number. ROC-AUC on this page is the same shape: it is exactly how many positive-negative pairs come out in the right order.'}, view:VIEW});
  } else {
    LC169.steps.forEach((s, i) => {
      F.push({shapes:lc169Stage({step:i, done:(i === LC169.steps.length - 1),
        cap:{zh:'一個候選、一個計數，掃過去一次', en:'one candidate, one count, a single pass'}}),
        panels:[{lbl:'candidate', chips:[chip(String(s.cand), 'hot')]},
                {lbl:'count', chips:[chip(String(s.c), s.c === 0 ? 'bad' : 'act')]}],
        line:s.c === 0 ? 3 : 4,
        msg:{zh:'讀到 ' + s.v + '：' + (s.v === s.cand ? '跟候選一樣，count 變成 ' + s.c
                : '跟候選不同，count 變成 ' + s.c) + (s.c === 0 ? '。歸零之後，下一個值直接當新候選。' : '。'),
             en:'Read ' + s.v + ': ' + (s.v === s.cand ? 'it matches the candidate, so count becomes ' + s.c
                : 'it differs from the candidate, so count becomes ' + s.c) + (s.c === 0 ? '. At zero the next value takes over as the candidate.' : '.')}, view:VIEW});
    });
    F.push({shapes:lc169Stage({step:LC169.steps.length - 1, done:true,
      cap:{zh:'這就是不平衡資料上 accuracy 真正在量的東西', en:'this is what accuracy on imbalanced data is really measuring'}}),
      panels:[{lbl:'answer', chips:[chip(String(LC169.ans), 'ok')]},
              {lbl:'no majority', chips:[chip('says ' + LC169.noMajSays, 'bad'), chip('checked: none', 'ok')]}],
      line:5,
      msg:{zh:'LeetCode 保證多數一定存在，所以不用驗。拿掉那個保證，[' + LC169.noMaj.join(', ') + '] 它還是回答 ' + LC169.noMajSays + '，而 ' + LC169.noMajSays + ' 只出現一次——補一趟計數才知道。它永遠回傳一個數字、永遠不丟例外，跟這一整天的失敗長得一模一樣：答案格式正確，意義是零。',
           en:'LeetCode promises a majority exists, so no check is needed. Remove that promise and [' + LC169.noMaj.join(', ') + '] still returns ' + LC169.noMajSays + ', which appears once in three rows - only a second counting pass tells you. It returns a number every time and never raises, which is the shape of every failure on this page: a well-formed answer that means nothing.'}, view:VIEW});
  }
  return F.list;
}

/* ================================================================ metadata */
const DAY_META = {
  title:{zh:'Day 50 — 評估指標與交叉驗證', en:'Day 50 - Evaluation metrics and cross-validation'},
  sub:{zh:'分類器給的是分數不是決定；每個指標都是同樣四個計數的不同比值，選指標就是選你願意犯哪種錯。',
       en:'A classifier gives you a score, not a decision. Every metric is a different ratio of the same four counts, and choosing one is choosing which mistakes you accept.'},
  tabs:[
    {
      id:'counts', label:{zh:'四個計數', en:'the four counts'},
      stage:{zh:'一個學到東西的模型，輸給一個常數', en:'a model that learned something, losing to a constant'},
      view:VIEW,
      idea:{zh:'2,000 列裡有 20 個正樣本。左邊是「一律回答 0」，右邊是訓練過的 logistic regression 切在 0.5。兩張混淆矩陣幾乎一樣，而常數的 accuracy 0.9900 還高過模型的 0.9895。accuracy 把 tp 和 tn 加在一起，而 tn 佔了這張表的 99%，所以它報告的其實是 tn。recall 兩邊都是 0.0000；precision 和 MCC 在常數那邊直接是 nan，因為分母是 0——它們拒絕回答，而那個拒絕是這一頁上唯一一個承認「這裡沒有東西可以評」的訊號。同一組分數的 ROC-AUC 是 ' + CONF.auc + '，所以模型確實學到了東西，只是 0.5 這個切點沒有人為這份資料選過。',
            en:'2,000 rows with 20 positives. On the left, always answer 0; on the right, a trained logistic regression cut at 0.5. The two confusion matrices are nearly identical, and the constant scores 0.9900 against the model 0.9895. Accuracy adds tp and tn together, and tn is 99% of this table, so what it reports is tn. Recall is 0.0000 on both sides; precision and MCC come back nan for the constant because their denominators are 0 - they refuse to answer, and that refusal is the one signal here that admits there is nothing to score. The same scores give a ROC-AUC of ' + CONF.auc + ', so the model did learn something; nobody chose 0.5 for this data.'},
      legend:[['#12b3b8', {zh:'計數與比值', en:'counts and ratios'}],
              ['#ff9736', {zh:'true positive', en:'true positives'}],
              ['#ff5c5c', {zh:'拒絕回答的指標', en:'a metric that refuses to answer'}]],
      get code(){ return CODE_COUNTS; },
      build:countsFrames
    },
    {
      id:'fruit', label:{zh:'四個分母', en:'four denominators'},
      stage:{zh:'一籃水果，四個不同的分母', en:'one basket of fruit, four different denominators'},
      view:VIEW,
      idea:{zh:'100 顆水果、10 顆蘋果，一台蘋果偵測器把 ' + (FRUIT.tp + FRUIT.fp) + ' 顆放到蘋果那一邊（' + FRUIT.tp + ' 顆真蘋果、' + FRUIT.fp + ' 顆橘子），漏掉 ' + FRUIT.fn + ' 顆。四個指標就是從同樣四格拿四個不同的分母：accuracy 除以全部（' + fmtN(FRUIT.acc) + '）；precision 除以你挑出來的那一欄（' + FRUIT.tp + '/' + (FRUIT.tp + FRUIT.fp) + ' = ' + fmtN(FRUIT.prec) + '）——拼字就是記憶法，preci-SI-on 問的是你挑的那一「側」；recall 除以真的是蘋果的那一列（' + FRUIT.tp + '/' + (FRUIT.tp + FRUIT.fn) + ' = ' + fmtN(FRUIT.rec) + '）——rec-ALL 問的是「所有」蘋果；FPR 除以真的是橘子的那一列（' + FRUIT.fp + '/' + (FRUIT.fp + FRUIT.tn) + ' = ' + fmtN(FRUIT.fpr) + '）。為什麼有了 accuracy 還要另外兩個？recall 和 FPR 各自只在一個類別裡算，precision 的分母是「模型挑了幾顆」所以大小由模型決定，而 accuracy 的分母是整籃、由資料集決定且模型改不了：100 顆裡 90 顆橘子，所以把每顆水果都說成橘子可以拿 ' + fmtN(FRUIT.lazyAcc) + '，而它一顆蘋果都沒找到。',
            en:'100 pieces of fruit, 10 of them apples, and a detector that puts ' + (FRUIT.tp + FRUIT.fp) + ' on the apple side (' + FRUIT.tp + ' real apples, ' + FRUIT.fp + ' oranges) and misses ' + FRUIT.fn + '. The four metrics are four different denominators taken from the same four cells. Accuracy divides by everything (' + fmtN(FRUIT.acc) + '). Precision divides by the column you picked (' + FRUIT.tp + '/' + (FRUIT.tp + FRUIT.fp) + ' = ' + fmtN(FRUIT.prec) + ') - the spelling is the mnemonic, preci-SI-on asks about the SIDE you picked. Recall divides by the row that really is apple (' + FRUIT.tp + '/' + (FRUIT.tp + FRUIT.fn) + ' = ' + fmtN(FRUIT.rec) + ') - rec-ALL asks about ALL the apples. FPR divides by the row that really is orange (' + FRUIT.fp + '/' + (FRUIT.fp + FRUIT.tn) + ' = ' + fmtN(FRUIT.fpr) + '). Why keep precision and recall when accuracy exists? recall and FPR each stay inside one class, precision has a denominator the model itself sets by choosing what to flag, and accuracy has a denominator the dataset fixes and the model cannot change: with 90 oranges in 100, calling every piece of fruit an orange scores ' + fmtN(FRUIT.lazyAcc) + ' while finding no apples at all.'},
      legend:[['#ff9736', {zh:'抓對的蘋果 tp', en:'apples caught, tp'}],
              ['#ff5c5c', {zh:'兩種錯誤 fp / fn', en:'the two mistakes, fp and fn'}],
              ['#12b3b8', {zh:'放對的橘子 tn', en:'oranges left alone, tn'}]],
      get code(){ return CODE_FRUIT; },
      build:fruitFrames
    },
    {
      id:'cut', label:{zh:'門檻', en:'the threshold'},
      stage:{zh:'同一組分數，九個決定', en:'one set of scores, nine decisions'},
      view:VIEW,
      idea:{zh:'模型輸出的是分數，變成決定需要一個門檻，而 0.5 是程式庫的預設值不是你的選擇。這一頁從頭到尾沒有重訓任何東西，只有切點在動：0.5 標出 1 列、recall 0.0000；0.10 是 F1 最高的地方（' + SWEEP[4].f1 + '），標 ' + SWEEP[4].flag + ' 列抓到 ' + SWEEP[4].tp + ' 個；0.01 的 recall 到 0.9500，但要看 845 列。最值得記住的是 accuracy 那條線：它在模型什麼都不抓的時候最高（0.9895），在 F1 最好的時候是 0.9870，在 recall 0.9500 的時候只剩 0.5865。accuracy 隨著模型變有用而下降，因為有用就是拿 true negative 去換 true positive。真正該決定的是你一天能審幾列，而不是哪個指標好看。',
            en:'A model outputs a score; turning it into a decision needs a threshold, and 0.5 is a library default rather than your choice. Nothing is retrained across this tab - only the cut moves. At 0.5 it flags 1 row and recall is 0.0000. At 0.10 F1 peaks at ' + SWEEP[4].f1 + ', flagging ' + SWEEP[4].flag + ' rows and catching ' + SWEEP[4].tp + '. At 0.01 recall reaches 0.9500 but you are reviewing 845 rows. The line worth remembering is accuracy: highest (0.9895) when the model catches nothing, 0.9870 where F1 is best, and 0.5865 where recall is 0.9500. Accuracy falls as the model becomes useful, because being useful means spending true negatives to buy true positives. The decision to make is how many rows a day you can review, not which metric looks best.'},
      legend:[['#12b3b8', {zh:'recall', en:'recall'}],
              ['#ff9736', {zh:'precision', en:'precision'}],
              ['#b45cff', {zh:'F1', en:'F1'}],
              ['#8fa3ac', {zh:'accuracy', en:'accuracy'}]],
      get code(){ return CODE_CUT; },
      build:cutFrames
    },
    {
      id:'curves', label:{zh:'兩條曲線', en:'the two curves'},
      stage:{zh:'兩個模型，兩個相反的排名', en:'two models, two opposite rankings'},
      view:VIEW,
      idea:{zh:'同一批標籤、兩個模型。broad 把整體分得很開，ROC-AUC ' + CURVE.broad.auc + '；sharp 只把一半的正樣本排在所有負樣本之上、另一半埋在只贏過 22% 負樣本的位置，ROC-AUC 只有 ' + CURVE.sharp.auc + '。換成 average precision，順序整個反過來：sharp ' + CURVE.sharp.ap + '、broad ' + CURVE.broad.ap + '。兩邊都沒算錯，差別是一個分母——ROC 的 x 軸除以全部 ' + CURVE.neg + ' 個負樣本，而 broad 要抓滿 20 個得降到 0.5897，有 ' + BAL_SPREAD.above + ' 個負樣本在那條線之上——除以 ' + CURVE.neg + ' 只動 0.0682；同樣這 ' + BAL_SPREAD.above + ' 個，佔你為此標出來的 ' + BAL_SPREAD.flagged + ' 列的 87%，那才是審查佇列的樣子。要決定該信哪一個，看最後那張圖：分數最高的 10 列裡，sharp 有 ' + CURVE.sharp.topk['10'] + ' 個真案件，broad 有 ' + CURVE.broad.topk['10'] + ' 個；到前 50 列就反過來了（broad ' + CURVE.broad.topk['50'] + '、sharp ' + CURVE.sharp.topk['50'] + '），前 100 列是 broad ' + CURVE.broad.topk['100'] + '、sharp ' + CURVE.sharp.topk['100'] + '。先決定一天能審幾列，再決定看哪個指標。',
            en:'The same labels, two models. broad separates the population well with a ROC-AUC of ' + CURVE.broad.auc + '; sharp only ranks half the positives above every negative and buries the rest where they outrank 22% of them, scoring ' + CURVE.sharp.auc + '. Switch to average precision and the order flips: sharp ' + CURVE.sharp.ap + ' against broad ' + CURVE.broad.ap + '. Neither is miscomputed; the difference is one denominator. The ROC x-axis divides by all ' + CURVE.neg + ' negatives, and broad has to come down to 0.5897 to catch all 20, at which point ' + BAL_SPREAD.above + ' negatives sit above the line - ' + BAL_SPREAD.above + '/' + CURVE.neg + ' = 0.0682 on the ROC axis, but ' + BAL_SPREAD.above + ' of the ' + BAL_SPREAD.flagged + ' rows you flagged, 87% of the review queue. The last frame decides it: in the ten highest-scoring rows sharp holds ' + CURVE.sharp.topk['10'] + ' real cases and broad holds ' + CURVE.broad.topk['10'] + '; by the top 50 it has already reversed (broad ' + CURVE.broad.topk['50'] + ', sharp ' + CURVE.sharp.topk['50'] + '), and at the top 100 it is broad ' + CURVE.broad.topk['100'] + ' against sharp ' + CURVE.sharp.topk['100'] + '. Decide the review budget first, then the metric.'},
      legend:[['#12b3b8', {zh:'broad', en:'broad'}],
              ['#ff9736', {zh:'sharp', en:'sharp'}],
              ['#8fa3ac', {zh:'隨機猜的對角線', en:'the chance diagonal'}]],
      get code(){ return CODE_CURVE; },
      build:curveFrames
    },
    {
      id:'axes', label:{zh:'ROC 的兩條軸', en:'the ROC axes'},
      stage:{zh:'為什麼橫軸縱軸是那兩個', en:'why those two numbers are the axes'},
      view:VIEW,
      idea:{zh:'ROC 橫軸放 FPR、縱軸放 recall，不是為了對稱。recall = tp/(tp+fn) 只數真的是正樣本的列，FPR = fp/(fp+tn) 只數真的是負樣本的列——兩個分母都沒混到另一個類別，所以兩條軸都不知道每類各有幾個。往上是好處（抓到更多真正的正樣本），往右是代價（誤標更多真正的負樣本），每個門檻一個點，對角線就是亂猜。這件事可以量：把 broad 的 20 個正樣本全留著、隨機丟掉負樣本，正樣本從 1% 變成 ' + Math.round(100 * BALANCE[BALANCE.length - 1].rate) + '%，平均 ROC-AUC 只移動 ' + BAL_SPREAD.auc + '，平均 average precision 移動 ' + BAL_SPREAD.ap + '——遠了 ' + BAL_SPREAD.ratio + ' 倍。這個「幾乎不受影響」同時是該用 ROC-AUC 和不該用它的理由（只留 40 個負樣本時，50 次單獨抽樣其實從 0.9400 到 1.0000 都有，不漂移的是平均值）。',
            en:'The ROC curve puts FPR across and recall up, and not for symmetry. recall = tp/(tp+fn) counts only rows that really are positive; FPR = fp/(fp+tn) counts only rows that really are negative - neither denominator mixes the classes, so neither axis knows how many of each there are. Up is the benefit (more real positives caught), right is the cost (more real negatives falsely flagged), each threshold is one point, and the diagonal is guessing at random. It is measurable: keep all 20 positives of broad and throw negatives away, and as the positives go from 1% to ' + Math.round(100 * BALANCE[BALANCE.length - 1].rate) + '% of the rows the mean ROC-AUC moves by ' + BAL_SPREAD.auc + ' and the mean average precision moves by ' + BAL_SPREAD.ap + ', ' + BAL_SPREAD.ratio + ' times further. That near-invariance is both the reason to use ROC-AUC and the reason not to - the 50 individual draws at 40 negatives actually run from 0.9400 to 1.0000, and what does not drift is their average.'},
      legend:[['#12b3b8', {zh:'ROC-AUC', en:'ROC-AUC'}],
              ['#ff9736', {zh:'average precision', en:'average precision'}]],
      get code(){ return CODE_AXES; },
      build:axesFrames
    },
    {
      id:'folds', label:{zh:'切折', en:'splitting into folds'},
      stage:{zh:'一折裡一個正樣本都沒有', en:'a fold with no positives in it'},
      view:VIEW,
      idea:{zh:'把列號洗牌再切成 10 份，是最常見的交叉驗證寫法，而它從頭到尾沒有看過標籤。20 個正樣本分到 10 折，平均每折 2 個，而 2 很容易變成 0：這一次有兩折是空的，換 200 個隨機種子重跑，' + FOLDS.runs + ' 次至少有一折是空的。空折上的 recall 是 0/0 ＝ nan，用 np.mean 整個分數變 nan，用 np.nanmean 那一折安靜地從平均裡消失——兩種都不會報錯。分層切折就是「對 0 和 1 各做一次同樣的切法」，每折都拿到 2 個正樣本。它多一行，移除的不是誤差而是一整類失敗。',
            en:'Shuffling row numbers and cutting them into ten blocks is the most common way to cross-validate, and it never looks at the labels. Twenty positives over ten folds averages two per fold, and two is easily zero: this shuffle leaves two folds empty, and over 200 seeds ' + FOLDS.runs + ' of them produce at least one empty fold. Recall on an empty fold is 0/0 = nan; np.mean turns the whole score into nan and np.nanmean quietly drops the fold from the average, and neither raises. Stratifying means doing the same cut once for the zeros and once for the ones, so every fold gets two positives. One extra line, and what it removes is not error but a whole failure mode.'},
      legend:[['#12b3b8', {zh:'有正樣本的折', en:'a fold with positives in it'}],
              ['#ff5c5c', {zh:'0 個正樣本', en:'no positives at all'}]],
      get code(){ return CODE_FOLDS; },
      build:foldsFrames
    },
    {
      id:'leak', label:{zh:'三種洩漏', en:'three kinds of leakage'},
      stage:{zh:'同樣的量測，兩個差很多的答案', en:'the same measurement, two very different answers'},
      view:VIEW,
      idea:{zh:'三種「測試列的資訊跑進模型」的方式，同一套量測跑兩次。(a) 同一個客戶的 4 列被隨機切散：分類器只是「抄最近那一列」，隨機折 ' + LEAK[0].bad + '、照客戶切 ' + LEAK[0].good + '，它什麼都沒學到，只是認出了躺在訓練集裡的三個兄弟。(b) 先用全部 60 列挑出「最相關的 5 欄」再交叉驗證：在一張 400 欄純雜訊、標籤是擲銅板的表上拿到 ' + LEAK[1].bad + '，把挑欄位搬進每一折則是 ' + LEAK[1].good + '——真值是 0.5。(c) 用全部的列算平均與標準差（這裡換成另外 60 列、8 欄有訊號、logistic regression）：兩邊 accuracy 一樣，最大分數差 ' + LEAK[2].gap + '。三種不是一樣嚴重的，而這點本身就是重點：(c) 在列可以互換時幾乎無害（但測試列若是另一個時間段，全域平均就是未來的資訊），(a) 和 (b) 才是把雜訊變成結果的那兩種。',
            en:'Three ways information about the test rows reaches the model, each measured twice with the same procedure. (a) The four rows of one customer split at random: the classifier only copies the nearest row, and it scores ' + LEAK[0].bad + ' with random folds against ' + LEAK[0].good + ' with group folds - it learned nothing and recognised the three siblings sitting in the training set. (b) Picking the five most correlated columns on all 60 rows before cross-validating: ' + LEAK[1].bad + ' on a table of 400 pure-noise columns with coin-flip labels, against ' + LEAK[1].good + ' when the selection moves inside each fold - the truth is 0.5. (c) Taking the column means and standard deviations from every row - on a different 60 rows, 8 columns with real signal, scored by logistic regression: identical accuracy, largest score gap ' + LEAK[2].gap + '. They are not equally bad, and that is the point: (c) is near-harmless while rows are interchangeable (but if the test rows are a later period, the global mean is a fact about the future), while (a) and (b) are the two that turn noise into a result.'},
      legend:[['#ff5c5c', {zh:'洩漏的流程報出來的', en:'what the leaking pipeline reports'}],
              ['#12b3b8', {zh:'同一個模型真正值多少', en:'what the same model is really worth'}]],
      get code(){ return CODE_LEAK; },
      build:leakFrames
    },
    {
      id:'lc', label:{zh:'LeetCode 493 / 169', en:'LeetCode 493 / 169'},
      stage:{zh:'數對數，與多數元素', en:'counting pairs, and the majority element'},
      view:VIEW,
      variants:[{zh:'493 數對數', en:'493 counting pairs'}, {zh:'169 多數元素', en:'169 majority element'}],
      idea:{zh:'兩題都直接對應今天的內容。493 要數 i < j 且 nums[i] > 2 * nums[j] 的對數：答案是一個「對」的計數，而唯一不能做的就是把那些對生出來。兩半各自排好之後，左邊的 v 只要大於 2 * right[j]，right 裡 j 之前的每一個都算一對，一次比較結算一整塊，而且 j 整趟只往前走。坑在於計數要自己走一趟：merge 的前進條件是 left[i] <= right[j]，我們要的是 left[i] > 2 * right[j]，兩個不同的順序，塞在一起會少算而且照樣回傳數字。今天的 ROC-AUC 是同一個形狀——它就是「有幾對正樣本排在負樣本前面」，' + CURVE.pos + ' x ' + CURVE.neg + ' 對用排名一次算完。169 是 Boyer-Moore：一個候選、一個計數，相同加一不同減一，超過一半的值消不掉。它就是這一天 accuracy 在不平衡資料上真正量到的那個基準線（' + CONF.base + '）。LeetCode 保證多數存在，拿掉保證它仍然回傳一個數字——跟整天的失敗同一個形狀。',
            en:'Both problems are today restated. 493 counts the pairs i < j with nums[i] > 2 * nums[j]: the answer is a count over pairs, and the one thing you must not do is build them. Once both halves are sorted, if v exceeds 2 * right[j] then everything before j on the right qualifies too, so one comparison settles a block and j only ever moves forward. The trap is that the counting needs its own pass: the merge advances on left[i] <= right[j] while the pairs need left[i] > 2 * right[j], two different orders, and folding them together undercounts while still returning a number. ROC-AUC is the same shape - it is exactly how many positive-negative pairs come out in the right order, ' + CURVE.pos + ' x ' + CURVE.neg + ' of them settled once by sorting. 169 is Boyer-Moore: one candidate, one count, up on a match and down otherwise, and a value holding more than half cannot be cancelled away. It is the baseline accuracy was really measuring on imbalanced data (' + CONF.base + '). LeetCode promises a majority exists; without that promise it still returns a number - the same shape as every failure today.'},
      legend:[['#ff9736', {zh:'目前在看的元素', en:'the element being read'}],
              ['#9d6bff', {zh:'累計的計數', en:'the running count'}],
              ['#12b3b8', {zh:'已經結算過的', en:'already settled'}]],
      get code(){ return curV() === 0 ? CODE_493 : CODE_169; },
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
