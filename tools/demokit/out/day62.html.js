// DAY: 62
// TITLE_ZH: Mixture of Experts：一個 FFN 變成很多個，而每個 token 只跑其中兩個
// TITLE_EN: Mixture of Experts: one FFN becomes many, and each token runs only two of them
// SUB_ZH: Day 53 量過：transformer 三分之二的參數在 FFN，而每個 token 都要付全部。把那一個 FFN 換成 E 份、每個 token 只送去最好的 k 份，參數乘上 E 而每個 token 的算術只乘 k。代價是：router 要自己學會把 token 分散開，而當 experts 住在不同 GPU 上時，每個 token 都得被送過去再送回來。
// SUB_EN: Day 53 measured it: two thirds of a transformer's parameters sit in the feed-forward block and every token pays for all of them. Replace that one FFN with E copies and send each token to the best k, and the parameters multiply by E while the arithmetic per token multiplies by k. The price: the router has to learn to spread the tokens, and when the experts live on different GPUs every token has to be shipped there and back.
// FOLDER: day%2062%20-%20moe%20and%20routing
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
const ACC = {"d_model": 4096, "d_ff": 14336, "experts": 64, "k": 2, "shared": 1, "dense_params": 176160768, "moe_params": 11450449920, "active_params": 528482304, "param_ratio": 65.0, "flop_ratio": 3.0, "capacity_per_flop": 21.667, "model_total": 698.477, "model_active": 32.237, "dense_m": 176.2, "moe_b": 11.45, "active_m": 528.5, "total_b": 698.0, "act_b": 32.0, "moe_dff": 917504, "act_dff": 28672};
const TRAIN = {"dense": 0.0363, "tokens": 2048, "clusters": 8, "experts": 16, "gain": 0.368};
const RUNS = [{"name": "no balancing", "loss": 0.023, "counts": [579, 554, 550, 475, 461, 405, 352, 323, 293, 84, 14, 3, 1, 1, 1, 0], "mm": 2.26, "dead": 1, "entropy": 0.812, "purity": 0.641}, {"name": "auxiliary loss", "loss": 0.0241, "counts": [508, 471, 430, 390, 353, 340, 331, 306, 272, 201, 155, 143, 87, 67, 24, 18], "mm": 1.98, "dead": 0, "entropy": 0.922, "purity": 0.645}, {"name": "bias, no loss", "loss": 0.0257, "counts": [270, 260, 260, 260, 260, 259, 259, 259, 258, 258, 256, 254, 253, 252, 249, 229], "mm": 1.05, "dead": 0, "entropy": 1.0, "purity": 0.568}];
const CAP = {"plain": [{"f": 1.0, "factor": 1.0, "capacity": 256, "dropped": 1688, "drop_rate": 0.4121, "padding": 0.0}, {"f": 1.25, "factor": 1.25, "capacity": 320, "dropped": 1139, "drop_rate": 0.2781, "padding": 0.25}, {"f": 1.5, "factor": 1.5, "capacity": 384, "dropped": 720, "drop_rate": 0.1758, "padding": 0.5}, {"f": 2.0, "factor": 2.0, "capacity": 512, "dropped": 147, "drop_rate": 0.0359, "padding": 1.0}], "free": [{"f": 1.0, "factor": 1.0, "capacity": 256, "dropped": 43, "drop_rate": 0.0105, "padding": 0.0}, {"f": 1.25, "factor": 1.25, "capacity": 320, "dropped": 0, "drop_rate": 0.0, "padding": 0.25}, {"f": 1.5, "factor": 1.5, "capacity": 384, "dropped": 0, "drop_rate": 0.0, "padding": 0.5}, {"f": 2.0, "factor": 2.0, "capacity": 512, "dropped": 0, "drop_rate": 0.0, "padding": 1.0}]};
const EP = {"gpus": 4, "per": 4, "tokens": [1582, 798, 1361, 355], "mm": 1.54, "idle": 0.353, "tokensB": [1022, 1001, 1032, 1041], "mmB": 1.02, "mb": 67.1, "fp8mb": 50.3};
const TP = {"tp": 58.7, "ep": 117.4, "ratio": 2.0, "k": 2};
const DP = {"tokens": 4096, "k": 8, "moved": 32768, "mb_bf16": 469.762, "mb_fp8": 234.881, "cross_share": 0.875, "ms_bf16": 8.588, "ms_fp8": 4.294, "speedup": 2.0};

const PURPLE2 = '#c7a6ff';

/* ================================================================ tab 1: the accounting */
const CODE_ACC = [
  '# one SwiGLU FFN: gate, up, down',
  'dense = 3 * d_model * d_ff',
  '',
  '# E experts of the same shape, k of them per token (+1 always-on shared)',
  'moe_params   = (E + shared) * dense',
  'active_param = (k + shared) * dense',
  '',
  '# parameters scale with E, arithmetic per token scales with k'
];
function accStage(o){
  const sh = [];
  const rows = [[{zh:'一個 dense FFN', en:'one dense FFN'}, ACC.dense_m, ACC.dense_m, 'ok'],
                [{zh:ACC.experts + ' 個 expert ＋ ' + ACC.shared + ' 個 shared',
                  en:ACC.experts + ' experts plus ' + ACC.shared + ' shared'},
                 ACC.moe_b * 1000, ACC.active_m, 'hot']];
  const top = ACC.moe_b * 1000;
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.10 - i * 1.40;
    sh.push(fitT(.70, y + .92, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
    sh.push(S.r(.70, y + .46, Math.max(.06, 8.4 * r[1] / top), .40, 'ghost', null));
    sh.push(S.r(.70, y, Math.max(.06, 8.4 * r[2] / top), .40, r[3], null));
    sh.push(fitT(9.30, y + .66, {zh:'總參數 ' + (r[1] / 1000 >= 1 ? (r[1] / 1000).toFixed(1) + ' B' : r[1].toFixed(0) + ' M'),
                                 en:'total ' + (r[1] / 1000 >= 1 ? (r[1] / 1000).toFixed(1) + ' B' : r[1].toFixed(0) + ' M')},
                 {anchor:'end', c:COL.grey, fs:.25}));
    sh.push(fitT(9.30, y + .14, {zh:'每個 token 跑 ' + r[2].toFixed(0) + ' M',
                                 en:'active ' + r[2].toFixed(0) + ' M per token'},
                 {anchor:'end', c:COL.pale, fs:.25}));
  });
  if (o.upto >= 1)
    sh.push(fitT(4.9, 1.00, {zh:'參數 ×' + ACC.param_ratio.toFixed(0) + '，每個 token 的算術只 ×' + ACC.flop_ratio.toFixed(0),
                             en:ACC.param_ratio.toFixed(0) + 'x the parameters for ' + ACC.flop_ratio.toFixed(0) + 'x the arithmetic per token'},
                 {c:COL.tealL, fs:.32}));
  return caps(sh, o);
}
function accFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:accStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'起點：FFN 佔了模型三分之二的參數', en:'the starting point: the FFN is two thirds of the model'},
    panels:[{lbl:'d_model', chips:[chip(String(ACC.d_model), 'dim')]},
            {lbl:'d_ff', chips:[chip(String(ACC.d_ff), 'dim')]},
            {lbl:'params', chips:[chip(ACC.dense_m.toFixed(0) + ' M', 'ok')]}],
    msg:{zh:'一個 SwiGLU 的 FFN 有 gate／up／down 三個矩陣，' + ACC.dense_m.toFixed(0) + ' M 參數，而**每個 token 都要把它們全部讀過一遍**。Day 53 已經量過：transformer 的參數大部分在這裡，不在 attention。',
         en:'A SwiGLU feed-forward block is three matrices - gate, up, down - ' + ACC.dense_m.toFixed(0) + ' M parameters, and **every token reads all of them**. Day 53 measured where a transformer\'s parameters live: mostly here, not in attention.'}});
  p({upto:1, line:7,
    cap:{zh:'把它複製 ' + ACC.experts + ' 份，每個 token 只用 ' + ACC.k + ' 份',
         en:'copy it ' + ACC.experts + ' times and use ' + ACC.k + ' per token'},
    panels:[{lbl:'total', chips:[chip(ACC.moe_b.toFixed(1) + ' B', 'act')]},
            {lbl:'active', chips:[chip(ACC.active_m.toFixed(0) + ' M', 'ok')]},
            {lbl:'capacity / cost', chips:[chip(ACC.capacity_per_flop.toFixed(0) + 'x', 'hot')]}],
    msg:{zh:'參數變成 ' + ACC.param_ratio.toFixed(0) + ' 倍，每個 token 的算術只變成 ' + ACC.flop_ratio.toFixed(0) + ' 倍（top-' + ACC.k + ' 再加一個總是會跑的 shared expert）。換個說法：這一層等於一個寬度 ' + ACC.moe_dff.toLocaleString() + ' 的 FFN，但每個 token 只會啟動其中 ' + ACC.act_dff.toLocaleString() + '。整個 61 層的模型就是 ' + ACC.total_b.toFixed(0) + ' B 參數、每個 token 只跑 ' + ACC.act_b.toFixed(0) + ' B——這兩年所有大型開源 MoE 的形狀都是這樣。',
         en:'The parameters go up ' + ACC.param_ratio.toFixed(0) + 'x and the arithmetic per token only ' + ACC.flop_ratio.toFixed(0) + 'x (top-' + ACC.k + ' plus one always-on shared expert). Put differently: the layer is a feed-forward block of width ' + ACC.moe_dff.toLocaleString() + ' that only ever activates ' + ACC.act_dff.toLocaleString() + ' of it per token. Across 61 layers that is ' + ACC.total_b.toFixed(0) + ' B parameters of which ' + ACC.act_b.toFixed(0) + ' B run for any given token - the shape of every large open MoE of the last two years.'}});
  return F.list;
}

/* ================================================================ tab 2: the router */
const CODE_ROUTER = [
  'logits = x @ Wg                 # (tokens, experts)',
  'probs  = softmax(logits)',
  'idx    = argsort(-probs)[:, :k] # the discrete, non-differentiable part',
  'w      = probs[idx]             # the weights that DO get a gradient',
  'y      = sum_j w[:, j] * expert[idx[:, j]](x)',
  '',
  '# a bias may be added for SELECTION only - never to w'
];
function routerStage(o){
  const sh = [];
  const probs = [.34, .27, .15, .11, .08, .05];
  const keep = [true, true, false, false, false, false];
  probs.forEach((v, i) => {
    const x = .95 + i * 1.45, h = 2.2 * v / probs[0];
    sh.push(S.r(x, 3.90 - h, 1.05, h, o.upto >= 1 ? (keep[i] ? 'hot' : 'ghost') : 'ok', null));
    sh.push(fitT(x + .52, 4.20, 'E' + (i + 1), {c:keep[i] || o.upto < 1 ? COL.pale : COL.grey, fs:.26}));
    sh.push(fitT(x + .52, 3.60, v.toFixed(2), {c:COL.grey, fs:.24}));
  });
  sh.push(fitT(.70, 4.80, {zh:'softmax 之後每個 expert 的機率', en:'the gate probability of each expert'},
               {anchor:'start', c:COL.grey, fs:.26}));
  if (o.upto >= 2){
    sh.push(fitT(4.9, 1.70, {zh:'y = 0.34·E1(x) ＋ 0.27·E2(x)', en:'y = 0.34 * E1(x) + 0.27 * E2(x)'},
                 {c:COL.tealL, fs:.30}));
    sh.push(fitT(4.9, 1.15, {zh:'被選中的那兩個才有梯度；沒被選中的 expert 不會變好，也就永遠選不上',
                             en:'only the chosen two get a gradient; an expert that is never chosen never improves'},
                 {c:COL.red, fs:.26}));
  }
  return caps(sh, o);
}
function routerFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:routerStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'router 就是一個 linear ＋ softmax', en:'the router is one linear layer and a softmax'},
    panels:[{lbl:'Wg', chips:[chip('d_model x E', 'dim')]}],
    msg:{zh:'對每個 token 算出每個 expert 的分數，softmax 成機率。這是整個 MoE 唯一「學哪個 token 該去哪裡」的地方，而且它只有一層 linear。',
         en:'Score every expert for every token and softmax the scores. This is the only place in an MoE that learns which token goes where, and it is a single linear layer.'}});
  p({upto:1, line:3,
    cap:{zh:'取 top-k，其餘丟掉', en:'keep the top k, drop the rest'},
    panels:[{lbl:'k', chips:[chip('2', 'hot')]},
            {lbl:'dropped', chips:[chip('4 experts', 'dim')]}],
    msg:{zh:'選 top-k 這一步是**離散的**，沒有梯度穿得過去。所以 router 學到東西的唯一管道，是「它已經選上的那幾個 expert 的權重 w」——這正是 MoE 難訓練的根源。',
         en:'Taking the top k is **discrete**: no gradient flows through the selection. The router\'s only channel for learning is the weights w of the experts it already picked - which is the root of every training difficulty in MoE.'}});
  p({upto:2, line:5,
    cap:{zh:'輸出是被選中 expert 的加權和', en:'the output is the weighted sum of the chosen experts'},
    panels:[{lbl:'weights', chips:[chip('0.34', 'ok'), chip('0.27', 'ok')]}],
    msg:{zh:'所以會發生這件事：一開始運氣好被選上的 expert 拿到梯度、變得更好、於是更常被選上；沒被選上的永遠不會變好。**這是一個會自我放大的迴圈**，下一個分頁就是它的實測結果。',
         en:'Which sets up the feedback loop: an expert that gets picked early receives gradient, improves, and gets picked more; one that is not picked never improves. **It is self-reinforcing**, and the next tab is what that looks like when measured.'}});
  return F.list;
}

/* ================================================================ tab 3: collapse and balancing */
const CODE_BAL = [
  '# (a) the auxiliary loss, Switch / GShard',
  'f = tokens_routed_to / (T * k)        # what happened: no gradient',
  'P = probs.mean(axis=0)                # differentiable',
  'aux = E * (f * P).sum()               # flat when both are flat',
  '',
  '# (b) loss-free balancing, DeepSeek-V3',
  'idx = argsort(-(probs + bias))[:, :k] # bias affects SELECTION only',
  'w   = probs[idx]                      # ...never the weight',
  'bias -= rate * sign(counts - counts.mean())'
];
function balStage(o){
  const sh = [];
  const runs = RUNS.slice(0, o.upto + 1);
  const top = RUNS[0].counts[0];
  runs.forEach((r, i) => {
    const x0 = .70 + i * 3.10;
    const mean = r.counts.reduce(function(a, b){ return a + b; }, 0) / r.counts.length;
    r.counts.forEach(function(c, j){
      const h = Math.max(.02, 2.3 * c / top);
      sh.push(S.r(x0 + j * .168, 4.30 - h, .140, h,
                  i === 0 ? 'bad' : (i === 1 ? 'ok' : 'act'), null));
    });
    sh.push(S.e(x0 - .05, 4.30 - 2.3 * mean / top, x0 + 16 * .168,
                4.30 - 2.3 * mean / top, {s:'ghost', arrow:false, w:.03}));
    sh.push(fitT(x0 + 1.30, 4.85, r.name, {c:COL.tealL, fs:.27}));
    sh.push(fitT(x0 + 1.30, 1.35, {zh:'最忙/平均 ' + r.mm.toFixed(2) + '　死掉 ' + r.dead,
                                   en:'busiest/avg ' + r.mm.toFixed(2) + ',  ' + r.dead + ' dead'},
                 {c:COL.pale, fs:.25}));
    sh.push(fitT(x0 + 1.30, .95, {zh:'loss ' + r.loss.toFixed(4) + '　純度 ' + r.purity.toFixed(2),
                                  en:'loss ' + r.loss.toFixed(4) + ',  purity ' + r.purity.toFixed(2)},
                 {c:COL.grey, fs:.25}));
  });
  return caps(sh, o);
}
function balFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:balStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'不加任何東西，router 會塌掉', en:'left alone, the router collapses'},
    panels:[{lbl:'dense baseline', chips:[chip(TRAIN.dense.toFixed(4), 'dim')]},
            {lbl:'MoE loss', chips:[chip(RUNS[0].loss.toFixed(4), 'ok'),
                                    chip((TRAIN.gain * 100).toFixed(0) + '% better', 'ok')]},
            {lbl:'busiest/avg', chips:[chip(RUNS[0].mm.toFixed(2), 'bad')]}],
    msg:{zh:'真的訓練一個：' + TRAIN.tokens + ' 個 token、' + TRAIN.clusters + ' 種 regime、' + TRAIN.experts + ' 個 expert、top-2。MoE 確實贏過 dense（' + RUNS[0].loss.toFixed(4) + ' vs ' + TRAIN.dense.toFixed(4) + '），但負載是 ' + RUNS[0].counts.slice(0, 3).join('／') + ' … ' + RUNS[0].counts.slice(-3).join('／') + '：最忙的是平均的 ' + RUNS[0].mm.toFixed(2) + ' 倍，而且有 ' + RUNS[0].dead + ' 個 expert 一個 token 都沒拿到。那些參數照樣佔記憶體、照樣要參加 all-to-all。',
         en:'Train one for real: ' + TRAIN.tokens + ' tokens, ' + TRAIN.clusters + ' regimes, ' + TRAIN.experts + ' experts, top-2. The MoE does beat the dense layer (' + RUNS[0].loss.toFixed(4) + ' against ' + TRAIN.dense.toFixed(4) + '), but the load is ' + RUNS[0].counts.slice(0, 3).join(', ') + ' ... ' + RUNS[0].counts.slice(-3).join(', ') + ': the busiest expert takes ' + RUNS[0].mm.toFixed(2) + 'x the average and ' + RUNS[0].dead + ' expert never sees a token at all. Those parameters still occupy memory and still take part in the all-to-all.'}});
  p({upto:1, line:4,
    cap:{zh:'(a) 加一個平衡用的輔助 loss', en:'(a) add an auxiliary balancing loss'},
    panels:[{lbl:'busiest/avg', chips:[chip(RUNS[0].mm.toFixed(2), 'bad'),
                                       chip(RUNS[1].mm.toFixed(2), 'ok')]},
            {lbl:'dead', chips:[chip(String(RUNS[1].dead), 'ok')]}],
    msg:{zh:'Switch／GShard 的做法：加上 E × Σ(實際路由比例 × 平均機率)。兩個因子缺一不可——實際比例沒有梯度，平均機率才有，相乘之後「已經很忙的 expert」會被推開機率。loss 只退了 ' + ((RUNS[1].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '%，死掉的 expert 歸零。',
         en:'Switch and GShard add E * sum(fraction routed * mean probability). Both factors are needed: the fraction is what actually happened and carries no gradient, the probability does, and multiplying them pushes probability away from experts that are already busy. The loss gives up ' + ((RUNS[1].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '% and the dead expert comes back.'}});
  p({upto:2, line:7,
    cap:{zh:'(b) 不加 loss，只加一個選擇用的 bias', en:'(b) no loss at all: a bias used only for selection'},
    panels:[{lbl:'busiest/avg', chips:[chip(RUNS[2].mm.toFixed(2), 'ok')]},
            {lbl:'entropy', chips:[chip(RUNS[2].entropy.toFixed(3), 'ok')]},
            {lbl:'purity', chips:[chip(RUNS[2].purity.toFixed(2), 'bad')]}],
    msg:{zh:'DeepSeek-V3 的做法：每個 expert 一個 bias，只加在**挑選**用的分數上，乘在 expert 輸出上的權重仍然用原本的機率。所以它搬動 token，卻不會產生跟任務 loss 打架的梯度。結果是最平的負載（' + RUNS[2].mm.toFixed(2) + '、entropy ' + RUNS[2].entropy.toFixed(3) + '），代價是 loss 退 ' + ((RUNS[2].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '%、純度從 ' + RUNS[0].purity.toFixed(2) + ' 掉到 ' + RUNS[2].purity.toFixed(2) + '。兩種方法贏的欄位不一樣：輔助 loss 保住專一性，bias 把負載壓得更平。',
         en:'DeepSeek-V3 gives each expert a bias added to the **selection** score only; the weight multiplying the expert output stays the unbiased probability. So it moves tokens without creating a gradient that fights the task loss. It produces the flattest load on the page (' + RUNS[2].mm.toFixed(2) + ', entropy ' + RUNS[2].entropy.toFixed(3) + ') and costs ' + ((RUNS[2].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '% of loss and purity falling from ' + RUNS[0].purity.toFixed(2) + ' to ' + RUNS[2].purity.toFixed(2) + '. The two methods win different columns: the auxiliary loss keeps the specialisation, the bias flattens the load harder.'}});
  return F.list;
}

/* ================================================================ tab 4: capacity */
const CODE_CAP = [
  'capacity = ceil(factor * k * tokens / experts)',
  '',
  'for each expert:',
  '    keep the first `capacity` tokens routed to it',
  '    the rest skip the layer through the residual',
  '',
  '# factor 1.0 = buffers hold exactly the average load'
];
function capStage(o){
  const sh = [];
  CAP.plain.forEach((r, i) => {
    if (i > o.upto) return;
    const x = .95 + i * 2.30;
    sh.push(fitT(x + .80, 4.70, {zh:'factor ' + r.f, en:'factor ' + r.f}, {c:COL.tealL, fs:.27}));
    sh.push(S.r(x, 4.10 - 2.2 * r.drop_rate, .72, Math.max(.04, 2.2 * r.drop_rate), 'bad', null));
    sh.push(S.r(x + .85, 4.10 - 2.2 * CAP.free[i].drop_rate, .72,
                Math.max(.04, 2.2 * CAP.free[i].drop_rate), 'act', null));
    sh.push(fitT(x + .36, 4.35 - 2.2 * r.drop_rate, (r.drop_rate * 100).toFixed(0) + '%',
                 {c:COL.red, fs:.24}));
    sh.push(fitT(x + 1.21, 4.35 - 2.2 * CAP.free[i].drop_rate,
                 (CAP.free[i].drop_rate * 100).toFixed(1) + '%', {c:'#ffbe6b', fs:.24}));
    sh.push(fitT(x + .80, 1.55, 'buf ' + r.capacity, {c:COL.grey, fs:.24}));
    sh.push(fitT(x + .80, 1.20, '+' + (r.padding * 100).toFixed(0) + '% pad',
                 {c:COL.grey, fs:.24}));
  });
  sh.push(fitT(.95, .80, {zh:'紅：不平衡的 router　橘：平衡過的 router（掉的 token 比例）',
                           en:'red: unbalanced router.  amber: balanced router.  bars are dropped tokens'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function capFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:capStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'kernel 要固定形狀，所以每個 expert 有一個固定大小的 buffer',
         en:'kernels want fixed shapes, so each expert gets a fixed buffer'},
    panels:[{lbl:'factor 1.0', chips:[chip(String(CAP.plain[0].capacity), 'dim')]},
            {lbl:'dropped', chips:[chip((CAP.plain[0].drop_rate * 100).toFixed(0) + '%', 'bad')]}],
    msg:{zh:'capacity = factor × k × tokens / experts。factor 1.0 時 buffer 剛好等於平均負載，所以**超過平均的 token 全部被丟掉**（它們直接沿 residual 跳過這一層）：不平衡的 router 掉了 ' + (CAP.plain[0].drop_rate * 100).toFixed(0) + '%，平衡過的只掉 ' + (CAP.free[0].drop_rate * 100).toFixed(1) + '%。',
         en:'capacity = factor * k * tokens / experts. At factor 1.0 the buffer is exactly the average load, so **every token above average is dropped** - it skips the layer through the residual. The unbalanced router loses ' + (CAP.plain[0].drop_rate * 100).toFixed(0) + '% of its tokens; the balanced one loses ' + (CAP.free[0].drop_rate * 100).toFixed(1) + '%.'}});
  p({upto:CAP.plain.length - 1, line:6,
    cap:{zh:'調大 factor 可以不掉 token，但每個 expert 都要陪著 padding',
         en:'a bigger factor drops nothing, and pads every expert to fit the busiest'},
    panels:[{lbl:'factor 2.0', chips:[chip((CAP.plain[3].drop_rate * 100).toFixed(1) + '%', 'act')]},
            {lbl:'padding', chips:[chip('+' + (CAP.plain[3].padding * 100).toFixed(0) + '%', 'bad')]}],
    msg:{zh:'把 factor 調到 2.0，不平衡的 router 仍掉 ' + (CAP.plain[3].drop_rate * 100).toFixed(1) + '%，而且**每一個** expert 的 buffer 都變成兩倍——padding 是整排都要付的算術。平衡過的 router 在 factor ' + CAP.free[1].f + ' 就已經 0%。所以 load balance 不是精度上的潔癖，它直接決定你能用多小的 capacity factor。',
         en:'At factor 2.0 the unbalanced router still drops ' + (CAP.plain[3].drop_rate * 100).toFixed(1) + '% and **every** expert\'s buffer has doubled - the padding is arithmetic paid across the whole layer. The balanced router is already at zero by factor ' + CAP.free[1].f + '. Load balance is not an accuracy nicety: it decides how small a capacity factor you can afford.'}});
  return F.list;
}

/* ================================================================ tab 5: expert parallelism */
const CODE_EP = [
  '# E experts over N GPUs: each GPU owns E/N of them',
  'dispatch = all_to_all(tokens -> the GPU that owns their expert)',
  'y_local  = expert_forward(received_tokens)',
  'combine  = all_to_all(results -> back to the token owner)',
  '',
  '# two collectives per MoE layer, and everyone waits for the slowest rank',
  'idle = 1 - mean(tokens_per_gpu) / max(tokens_per_gpu)'
];
function epStage(o){
  const sh = [];
  const tok = o.upto >= 2 ? EP.tokensB : EP.tokens;
  const top = Math.max.apply(null, EP.tokens);
  for (let g = 0; g < EP.gpus; g++){
    const x = .85 + g * 2.25;
    sh.push(box(x, 4.00, 1.85, .62, 'idle', 'GPU ' + g + ' : ' + EP.per + ' experts', .24));
    if (o.upto >= 1){
      const h = Math.max(.06, 2.1 * tok[g] / top);
      sh.push(S.r(x, 3.40 - h, 1.85, h, tok[g] === Math.max.apply(null, tok) ? 'bad' : 'ok', null));
      sh.push(fitT(x + .92, 3.65, String(tok[g]), {c:COL.pale, fs:.25}));
      sh.push(S.e(x + .92, 3.95, x + .92, 3.48, {s:'ghost', arrow:true, w:.05}));
    }
  }
  sh.push(fitT(.85, 4.90, {zh:'dispatch：每個 token 送去擁有它 expert 的那張卡',
                           en:'dispatch: every token travels to the GPU that owns its expert'},
               {anchor:'start', c:COL.grey, fs:.26}));
  if (o.upto >= 1)
    sh.push(fitT(4.9, .95, {zh:(o.upto >= 2 ? '平衡後' : '不平衡時') + '最忙/平均 ' + (o.upto >= 2 ? EP.mmB : EP.mm).toFixed(2) +
                                (o.upto >= 2 ? '' : '，其他卡閒置 ' + (EP.idle * 100).toFixed(0) + '%'),
                            en:(o.upto >= 2 ? 'balanced: ' : 'unbalanced: ') + 'busiest/average ' + (o.upto >= 2 ? EP.mmB : EP.mm).toFixed(2) +
                               (o.upto >= 2 ? '' : ', the others idle ' + (EP.idle * 100).toFixed(0) + '% of the expert compute')},
                 {c:o.upto >= 2 ? COL.tealL : COL.red, fs:.29}));
  return caps(sh, o);
}
function epFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:epStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'experts 放不進一張卡，所以切開放', en:'the experts do not fit on one GPU, so they are split'},
    panels:[{lbl:'experts / GPU', chips:[chip(String(EP.per), 'dim')]},
            {lbl:'collectives', chips:[chip('2 per layer', 'act')]}],
    msg:{zh:'expert parallelism：E 個 expert 分給 N 張卡，每張卡擁有 E/N 個。於是一個 token 如果它的 expert 在別張卡上，就得被送過去（dispatch）、算完再送回來（combine）——**每一層兩次 all-to-all，每一步、每一層都要**。',
         en:'Expert parallelism splits E experts over N GPUs, E/N each. A token whose expert lives elsewhere has to be sent there (dispatch) and its result sent back (combine) - **two all-to-all collectives per MoE layer, every layer, every step**.'}});
  p({upto:1, line:6,
    cap:{zh:'而 router 的不平衡，直接變成 GPU 的閒置', en:'and the router\'s imbalance becomes idle GPUs'},
    panels:[{lbl:'tokens per GPU', chips:[chip(EP.tokens.join(' / '), 'bad')]},
            {lbl:'idle', chips:[chip((EP.idle * 100).toFixed(0) + '%', 'bad')]}],
    msg:{zh:'collective 是同步的：所有卡都要等最慢的那張。不平衡的 router 讓四張卡拿到 ' + EP.tokens.join('／') + ' 個 token，最忙的是平均的 ' + EP.mm.toFixed(2) + ' 倍，所以其他卡在 expert 計算期間閒置 ' + (EP.idle * 100).toFixed(0) + '%。**這才是 load balance 真正的價錢**，不是 loss 上的那幾個百分點。',
         en:'A collective is synchronous: everyone waits for the slowest rank. The unbalanced router gives the four GPUs ' + EP.tokens.join(', ') + ' tokens, the busiest taking ' + EP.mm.toFixed(2) + 'x the average, so the others idle for ' + (EP.idle * 100).toFixed(0) + '% of the expert computation. **That is the real price of imbalance**, not the few percent of loss.'}});
  p({upto:2, line:6,
    cap:{zh:'平衡過的 router，同樣四張卡', en:'the same four GPUs with a balanced router'},
    panels:[{lbl:'tokens per GPU', chips:[chip(EP.tokensB.join(' / '), 'ok')]},
            {lbl:'busiest/avg', chips:[chip(EP.mmB.toFixed(2), 'ok')]},
            {lbl:'EP vs TP bytes', chips:[chip(TP.ep.toFixed(0) + ' MB', 'act'),
                                          chip(TP.tp.toFixed(0) + ' MB', 'dim')]}],
    msg:{zh:'為什麼是 expert parallel 而不是把每個 expert 切開給所有卡（tensor parallel）？兩者每張卡放的權重一樣多，差在**每個 token 要搬多少**：TP 每層要對整個活化做一次 all-reduce（' + TP.tp.toFixed(0) + ' MB），EP 是對被路由的 token 做兩次 all-to-all（' + TP.ep.toFixed(0) + ' MB，k=' + TP.k + ' 所以是 ' + TP.ratio.toFixed(0) + ' 倍）。EP 搬得比較多，但它是稀疏、不規則、而且**可以跟計算重疊**的流量；層末的 all-reduce 不行。實務上兩個一起用：attention 用 TP，experts 用 EP。',
         en:'Why expert parallelism rather than splitting each expert across all GPUs (tensor parallelism)? Both hold the same weights per GPU; the difference is what moves per token. TP all-reduces the whole activation once per layer (' + TP.tp.toFixed(0) + ' MB); EP all-to-alls only the routed tokens, twice (' + TP.ep.toFixed(0) + ' MB, which is ' + TP.ratio.toFixed(0) + 'x at k=' + TP.k + '). EP moves more bytes, but the traffic is sparse, uneven and **overlappable with compute**, which an all-reduce at the end of a layer is not. Real systems use both: TP inside attention, EP across the experts.'}});
  return F.list;
}

/* ================================================================ tab 6: DeepEP */
const CODE_DEEPEP = [
  '# high-throughput kernels (prefill, training):',
  '#   NVLink inside the node, then ONE RDMA send per destination node',
  '# low-latency kernels (decode):',
  '#   pure RDMA, no forwarding hop, with a hook so the receive is not a barrier',
  '',
  'dispatch in fp8   -> half the bytes',
  'combine  in bf16  -> summing k expert outputs needs the precision'
];
function deepStage(o){
  const sh = [];
  const rows = [[{zh:'bf16 dispatch', en:'bf16 dispatch'}, DP.mb_bf16, DP.ms_bf16, 'bad'],
                [{zh:'fp8 dispatch', en:'fp8 dispatch'}, DP.mb_fp8, DP.ms_fp8, 'ok']];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 3.90 - i * 1.25;
    sh.push(fitT(.70, y + .62, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
    sh.push(S.r(.70, y, Math.max(.06, 7.2 * r[1] / DP.mb_bf16), .44, r[3], null));
    sh.push(fitT(9.30, y + .22, r[1].toFixed(0) + ' MB　' + r[2].toFixed(2) + ' ms',
                 {anchor:'end', c:COL.grey, fs:.26}));
  });
  sh.push(fitT(.70, 4.85, {zh:DP.tokens + ' 個 token、top-' + DP.k + '、d_model 7168 的一次 dispatch',
                           en:'one dispatch of ' + DP.tokens + ' tokens, top-' + DP.k + ', d_model 7168'},
               {anchor:'start', c:COL.grey, fs:.26}));
  if (o.upto >= 2){
    sh.push(box(.70, 1.10, 4.0, .62, 'act', {zh:'高吞吐：NVLink 轉送 ＋ 每節點一次 RDMA',
                                             en:'high throughput: NVLink forward, one RDMA per node'}, .23));
    sh.push(box(5.20, 1.10, 4.0, .62, 'hot', {zh:'低延遲：純 RDMA，無轉送，可重疊',
                                              en:'low latency: pure RDMA, no hop, overlappable'}, .23));
  }
  return caps(sh, o);
}
function deepFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:deepStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'這個 all-to-all 到底有多大', en:'how big this all-to-all actually is'},
    panels:[{lbl:'tokens moved', chips:[chip(DP.moved.toLocaleString(), 'dim')]},
            {lbl:'bytes', chips:[chip(DP.mb_bf16.toFixed(0) + ' MB', 'bad')]}],
    msg:{zh:DP.tokens + ' 個 token、top-' + DP.k + '，等於要搬 ' + DP.moved.toLocaleString() + ' 份 d_model 長度的向量：一次 dispatch ' + DP.mb_bf16.toFixed(0) + ' MB，而且 combine 還要再搬一次。節點內是 NVLink，跨節點是 RDMA，速度差好幾倍——' + (DP.cross_share * 100).toFixed(0) + '% 的目的地在別的節點。',
         en:'With ' + DP.tokens + ' tokens and top-' + DP.k + ' routing, ' + DP.moved.toLocaleString() + ' vectors of length d_model have to move: ' + DP.mb_bf16.toFixed(0) + ' MB for one dispatch, and the combine moves them back. Inside a node that is NVLink; across nodes it is RDMA at a fraction of the speed, and ' + (DP.cross_share * 100).toFixed(0) + '% of the destinations are on another node.'}});
  p({upto:1, line:5,
    cap:{zh:'FP8 dispatch：位元組減半', en:'FP8 dispatch: half the bytes'},
    panels:[{lbl:'fp8', chips:[chip(DP.mb_fp8.toFixed(0) + ' MB', 'ok'),
                               chip(DP.ms_fp8.toFixed(2) + ' ms', 'ok')]},
            {lbl:'speedup', chips:[chip(DP.speedup.toFixed(1) + 'x', 'hot')]}],
    msg:{zh:'前一天那套東西直接用在網路線上：把 token 用 FP8 送出去，' + DP.mb_bf16.toFixed(0) + ' MB 變 ' + DP.mb_fp8.toFixed(0) + ' MB、' + DP.ms_bf16.toFixed(2) + ' ms 變 ' + DP.ms_fp8.toFixed(2) + ' ms。但 **combine 要留 bf16**——把 k 個 expert 的輸出相加，用 FP8 會掉太多精度。dispatch 和 combine 用不同精度，這個不對稱是刻意的。',
         en:'The previous day applied to a network link: send the tokens in FP8 and ' + DP.mb_bf16.toFixed(0) + ' MB becomes ' + DP.mb_fp8.toFixed(0) + ' MB, ' + DP.ms_bf16.toFixed(2) + ' ms becomes ' + DP.ms_fp8.toFixed(2) + ' ms. But **the combine stays in bf16**: summing k expert outputs in FP8 loses too much. The asymmetry between dispatch and combine precision is deliberate.'}});
  p({upto:2, line:3,
    cap:{zh:'兩種 kernel，對應兩種階段', en:'two sets of kernels, for the two phases'},
    panels:[{lbl:'prefill / training', chips:[chip('high throughput', 'act')]},
            {lbl:'decode', chips:[chip('low latency', 'hot')]}],
    msg:{zh:'DeepEP 就是為這個 collective 寫的函式庫。高吞吐 kernel 會先用 NVLink 把 token 轉送到目的節點上的某張卡，再**每個目的節點只走一次 RDMA**——頻寬最省，適合 prefill 與訓練。低延遲 kernel 相反：純 RDMA、不轉送，因為 decode 等不起那一跳；而且它提供 hook 讓接收不是一道屏障，**attention 可以在 MoE 的 token 還在路上時就先算**。沒有這個重疊，每層兩次 all-to-all 是撐不住的。',
         en:'DeepEP is the library written for this collective. Its high-throughput kernels forward a token over NVLink to the destination node and then send it **once per destination node** over RDMA, which minimises bytes and suits prefill and training. The low-latency kernels do the opposite - pure RDMA, no forwarding hop, because decode cannot afford it - and expose a hook so the receive is not a barrier: **attention can run while the MoE tokens are still in flight**. Without that overlap, two all-to-alls per layer would not be survivable.'}});
  return F.list;
}

/* ================================================================ tabs 7-8: the servers */
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
const CODE_SGL = [
  'python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 \\',
  '    --tp 8 --ep-size 8 \\             # experts split across the 8 GPUs',
  '    --enable-deepep-moe \\            # the DeepEP kernels',
  '    --deepep-mode auto \\             # normal for prefill, low_latency for decode',
  '    --enable-eplb \\                  # rebalance experts across ranks',
  '    --ep-num-redundant-experts 32    # replicate the hot ones'
];
const SGL_ROWS = [
  [{zh:'experts 分到幾張卡', en:'experts split across GPUs'}, {zh:'--ep-size', en:'--ep-size'}],
  [{zh:'兩次 all-to-all', en:'the two all-to-alls'}, {zh:'--enable-deepep-moe', en:'--enable-deepep-moe'}],
  [{zh:'高吞吐 vs 低延遲 kernel', en:'throughput vs latency kernels'}, {zh:'--deepep-mode normal / low_latency / auto', en:'--deepep-mode normal / low_latency / auto'}],
  [{zh:'負載不平衡', en:'the load imbalance'}, {zh:'--enable-eplb（搬動 expert）', en:'--enable-eplb: move experts between ranks'}],
  [{zh:'最忙的那幾個 expert', en:'the hottest experts'}, {zh:'--ep-num-redundant-experts（複製）', en:'--ep-num-redundant-experts: replicate them'}]
];
function sglFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:SGL_ROWS, title:'SGLang'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:2, line:3,
    cap:{zh:'SGLang：EP 是伺服器旗標，跟 TP 併用', en:'SGLang: expert parallelism is a server flag, alongside TP'},
    panels:[{lbl:'--ep-size', chips:[chip('8', 'hot')]},
            {lbl:'--deepep-mode', chips:[chip('auto', 'act')]}],
    msg:{zh:'`--ep-size` 決定 experts 切給幾張卡，`--enable-deepep-moe` 把那兩次 all-to-all 換成 DeepEP 的 kernel，而 `--deepep-mode` 就是上一個分頁的兩種 kernel：prefill 用 normal（高吞吐）、decode 用 low_latency，`auto` 則依 batch 自動切。',
         en:'`--ep-size` decides how many GPUs the experts are split across, `--enable-deepep-moe` swaps the two all-to-alls for DeepEP\'s kernels, and `--deepep-mode` is the previous tab: `normal` (throughput) for prefill, `low_latency` for decode, `auto` to switch per batch.'}});
  p({upto:4, line:5,
    cap:{zh:'而平衡在上線之後還要再做一次', en:'and the balancing has to happen again at serving time'},
    panels:[{lbl:'--enable-eplb', chips:[chip('move experts', 'ok')]},
            {lbl:'redundant experts', chips:[chip('32', 'act')]}],
    msg:{zh:'訓練時的平衡 loss 壓平的是**平均**，但真實流量不是平均：某一分鐘的請求可能全都愛同一批 expert。所以 SGLang 會量每個 expert 的實際負載、週期性地**把 expert 在 rank 之間搬動**（EPLB），並且把最熱的那幾個**複製**到多張卡，讓它們的 token 有地方可去。這是第 3、5 個分頁的問題在線上重新出現一次。',
         en:'A balancing loss flattens the **average**, and real traffic is not the average: one minute of requests can all want the same experts. So SGLang measures the per-expert load and periodically **moves experts between ranks** (EPLB), and replicates the hottest ones so their tokens have more than one place to go. It is the problem from tabs three and five reappearing in production.'}});
  return F.list;
}
const CODE_VLLM = [
  'vllm serve deepseek-ai/DeepSeek-V3 \\',
  '    --tensor-parallel-size 8 \\',
  '    --enable-expert-parallel \\        # EP for the experts instead of TP',
  '    --data-parallel-size 2            # DP attention, EP experts',
  '',
  '# DeepEP or pplx-kernels underneath for the all-to-all when available'
];
const VLLM_ROWS = [
  [{zh:'experts 用 EP 切', en:'shard the experts with EP'}, {zh:'--enable-expert-parallel', en:'--enable-expert-parallel'}],
  [{zh:'attention 複製', en:'replicate the attention'}, {zh:'--data-parallel-size（DP attention）', en:'--data-parallel-size: DP attention'}],
  [{zh:'all-to-all kernel', en:'the all-to-all kernels'}, {zh:'DeepEP／pplx-kernels', en:'DeepEP or pplx-kernels'}]
];
function vllmFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:VLLM_ROWS, title:'vLLM'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:3,
    cap:{zh:'vLLM：同一個地方，不同的旗標', en:'vLLM: the same place, different flags'},
    panels:[{lbl:'--enable-expert-parallel', chips:[chip('EP', 'hot')]},
            {lbl:'--data-parallel-size', chips:[chip('2', 'act')]}],
    msg:{zh:'最後一行是值得記住的組合：**attention 用 data parallel 複製、experts 用 expert parallel 切開**。理由很直接——attention 沒有 expert 可以分，而 MoE 的權重不切根本放不下。兩個伺服器最後都走到這個配置。',
         en:'The last line is the combination worth knowing: **attention replicated with data parallelism, experts sharded with expert parallelism**. The reason is blunt - attention has no experts to spread, and the MoE weights do not fit otherwise. Both servers end up here.'}});
  p({upto:2, line:5,
    cap:{zh:'要盯的數字是平衡，不是平均', en:'what to watch is the balance, not the average'},
    panels:[{lbl:'watch', chips:[chip('per-rank load', 'bad')]}],
    msg:{zh:'在哪個伺服器上都一樣：要看的是**每個 rank 的負載分佈**，不是平均吞吐。如果有一張卡長期拿到最熱的 experts，其他卡每層都要等它兩次，而這個洞是任何 kernel 調校都補不回來的。',
         en:'On either server the thing to watch is the **per-rank load distribution**, not the average throughput. If one rank consistently holds the hottest experts, every other rank waits for it twice per layer, and no amount of kernel tuning recovers that.'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'Mixture of Experts 與路由', en:'Mixture of Experts and routing'},
  sub: {zh:'一個 FFN 變成很多個、每個 token 只跑兩個，以及這件事在 GPU 之間造成的全部後果',
        en:'one FFN becomes many, each token runs two of them, and everything that follows once they live on different GPUs'},
  tabs: [
    {id:'acc', label:{zh:'從 FFN 到 MoE', en:'from FFN to MoE'},
     stage:{zh:'參數乘以 E，算術只乘以 k', en:'parameters times E, arithmetic times k'},
     view:VIEW, code:CODE_ACC, build:accFrames,
     idea:{zh:'一個 SwiGLU FFN 是 ' + ACC.dense_m.toFixed(0) + ' M 參數、每個 token 全跑。換成 ' + ACC.experts + ' 個 expert ＋ ' + ACC.shared + ' 個 shared 之後是 ' + ACC.moe_b.toFixed(1) + ' B，但每個 token 只動 ' + ACC.active_m.toFixed(0) + ' M：參數 ×' + ACC.param_ratio.toFixed(0) + '、算術 ×' + ACC.flop_ratio.toFixed(0) + '。整個模型 ' + ACC.total_b.toFixed(0) + ' B 參數、每 token 跑 ' + ACC.act_b.toFixed(0) + ' B。',
           en:'One SwiGLU FFN is ' + ACC.dense_m.toFixed(0) + ' M parameters and every token pays for all of them. With ' + ACC.experts + ' experts plus ' + ACC.shared + ' shared it becomes ' + ACC.moe_b.toFixed(1) + ' B while a token only touches ' + ACC.active_m.toFixed(0) + ' M: ' + ACC.param_ratio.toFixed(0) + 'x the parameters for ' + ACC.flop_ratio.toFixed(0) + 'x the arithmetic. Across the model that is ' + ACC.total_b.toFixed(0) + ' B parameters of which ' + ACC.act_b.toFixed(0) + ' B run per token.'},
     legend:[['#8fa3ac', {zh:'總參數', en:'total parameters'}],
             ['#ff9736', {zh:'每個 token 實際跑的', en:'active per token'}]]},
    {id:'router', label:{zh:'router', en:'the router'},
     stage:{zh:'linear ＋ softmax ＋ top-k', en:'a linear layer, a softmax, and a top-k'},
     view:VIEW, code:CODE_ROUTER, build:routerFrames,
     idea:{zh:'router 只是一層 linear：算分數、softmax、取 top-k、用那幾個機率當權重加權相加。難點是 **top-k 是離散的沒有梯度**，router 只能透過「已經選上的 expert 的權重」學習——於是早期被選上的 expert 會愈來愈強，這是一個會自我放大的迴圈。',
           en:'The router is one linear layer: score, softmax, take the top k, and weight the chosen experts by those probabilities. The difficulty is that **the top-k is discrete and carries no gradient**, so the router can only learn through the weights of experts it already picked - which makes early winners stronger, a self-reinforcing loop.'},
     legend:[['#ff9736', {zh:'被選上的', en:'chosen'}],
             ['#8fa3ac', {zh:'沒被選上的', en:'not chosen'}]]},
    {id:'balance', label:{zh:'塌掉與平衡', en:'collapse and balancing'},
     stage:{zh:'三種 router，同一個模型', en:'three routers, one model'},
     view:VIEW, code:CODE_BAL, build:balFrames,
     idea:{zh:'實際訓練：MoE 比 dense 好 ' + (TRAIN.gain * 100).toFixed(0) + '%，但沒有平衡時最忙/平均 ' + RUNS[0].mm.toFixed(2) + '、' + RUNS[0].dead + ' 個 expert 完全沒被用到。輔助 loss（Switch/GShard）把它壓到 ' + RUNS[1].mm.toFixed(2) + '，只退 loss ' + ((RUNS[1].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '%；只加選擇用 bias 的 loss-free 方案（DeepSeek-V3）壓到 ' + RUNS[2].mm.toFixed(2) + '，退 ' + ((RUNS[2].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '%。',
           en:'Trained for real, the MoE beats the dense layer by ' + (TRAIN.gain * 100).toFixed(0) + '%, but with no balancing the busiest expert takes ' + RUNS[0].mm.toFixed(2) + 'x the average and ' + RUNS[0].dead + ' expert is never used. The Switch/GShard auxiliary loss brings that to ' + RUNS[1].mm.toFixed(2) + ' for ' + ((RUNS[1].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '% of loss; the loss-free selection bias (DeepSeek-V3) reaches ' + RUNS[2].mm.toFixed(2) + ' for ' + ((RUNS[2].loss / RUNS[0].loss - 1) * 100).toFixed(0) + '%.'},
     legend:[['#ff5c5c', {zh:'沒有平衡', en:'no balancing'}],
             ['#12b3b8', {zh:'輔助 loss', en:'auxiliary loss'}],
             ['#9d6bff', {zh:'選擇 bias', en:'selection bias'}]]},
    {id:'cap', label:{zh:'capacity factor', en:'the capacity factor'},
     stage:{zh:'丟掉 token，還是陪著 padding', en:'drop tokens, or pad every expert'},
     view:VIEW, code:CODE_CAP, build:capFrames,
     idea:{zh:'capacity = factor × k × tokens / experts。factor 1.0 時不平衡的 router 掉 ' + (CAP.plain[0].drop_rate * 100).toFixed(0) + '% 的 token，平衡過的只掉 ' + (CAP.free[0].drop_rate * 100).toFixed(1) + '%；要讓不平衡的 router 不掉 token 就得把 factor 拉到 2.0，而那代表**每一個** expert 的 buffer 都變兩倍。',
           en:'capacity = factor * k * tokens / experts. At factor 1.0 the unbalanced router drops ' + (CAP.plain[0].drop_rate * 100).toFixed(0) + '% of its tokens and the balanced one ' + (CAP.free[0].drop_rate * 100).toFixed(1) + '%. Stopping the drops without balancing means a factor of 2.0 - which doubles the buffer of **every** expert.'},
     legend:[['#ff5c5c', {zh:'不平衡時掉的', en:'dropped, unbalanced'}],
             ['#9d6bff', {zh:'平衡後掉的', en:'dropped, balanced'}]]},
    {id:'ep', label:{zh:'expert parallelism', en:'expert parallelism'},
     stage:{zh:'每層兩次 all-to-all', en:'two all-to-alls per layer'},
     view:VIEW, code:CODE_EP, build:epFrames,
     idea:{zh:'experts 切給 ' + EP.gpus + ' 張卡、每張 ' + EP.per + ' 個，於是 token 要被送去、再送回來。不平衡的 router 讓四張卡拿到 ' + EP.tokens.join('／') + '（最忙/平均 ' + EP.mm.toFixed(2) + '），其他卡閒置 ' + (EP.idle * 100).toFixed(0) + '%；平衡後是 ' + EP.mmB.toFixed(2) + '。EP 比 TP 多搬 ' + TP.ratio.toFixed(0) + ' 倍的位元組，但那是稀疏、可重疊的流量。',
           en:'With the experts split over ' + EP.gpus + ' GPUs, ' + EP.per + ' each, tokens must be shipped there and back. The unbalanced router gives the four GPUs ' + EP.tokens.join(', ') + ' (busiest/average ' + EP.mm.toFixed(2) + ') and leaves the others idle ' + (EP.idle * 100).toFixed(0) + '% of the expert compute; balanced it is ' + EP.mmB.toFixed(2) + '. EP moves ' + TP.ratio.toFixed(0) + 'x the bytes of a TP all-reduce, but the traffic is sparse and overlappable.'},
     legend:[['#ff5c5c', {zh:'最忙的那張卡', en:'the slowest GPU'}],
             ['#12b3b8', {zh:'其他卡', en:'the others'}]]},
    {id:'deepep', label:{zh:'DeepEP', en:'DeepEP'},
     stage:{zh:'讓那兩次 all-to-all 付得起', en:'making the two all-to-alls affordable'},
     view:VIEW, code:CODE_DEEPEP, build:deepFrames,
     idea:{zh:DP.tokens + ' token × top-' + DP.k + ' 的一次 dispatch 是 ' + DP.mb_bf16.toFixed(0) + ' MB；FP8 dispatch 減半成 ' + DP.mb_fp8.toFixed(0) + ' MB（combine 仍用 bf16，因為要加總 k 份輸出）。高吞吐 kernel 走 NVLink 轉送＋每節點一次 RDMA（prefill／訓練），低延遲 kernel 純 RDMA 不轉送並提供 hook 讓接收不是屏障（decode），**attention 可以在 token 還在路上時就先算**。',
           en:'One dispatch of ' + DP.tokens + ' tokens at top-' + DP.k + ' is ' + DP.mb_bf16.toFixed(0) + ' MB; an FP8 dispatch halves it to ' + DP.mb_fp8.toFixed(0) + ' MB while the combine stays bf16, because it sums k expert outputs. The high-throughput kernels forward over NVLink and send once per destination node (prefill and training); the low-latency kernels use pure RDMA with no hop and expose a hook so the receive is not a barrier, letting **attention run while the tokens are still in flight**.'},
     legend:[['#ff5c5c', {zh:'bf16 dispatch', en:'bf16 dispatch'}],
             ['#12b3b8', {zh:'fp8 dispatch', en:'fp8 dispatch'}]]},
    {id:'sglang', label:{zh:'在 SGLang 裡', en:'in SGLang'},
     stage:{zh:'EP、DeepEP 模式、EPLB', en:'EP, the DeepEP modes, and EPLB'},
     view:VIEW, code:CODE_SGL, build:sglFrames,
     idea:{zh:'`--ep-size` 切 experts、`--enable-deepep-moe` 換 kernel、`--deepep-mode` 選高吞吐或低延遲。而 `--enable-eplb` 與 `--ep-num-redundant-experts` 是在處理訓練時的平衡 loss 處理不了的事：真實流量某一分鐘就是會偏心，所以要能**把 expert 在 rank 之間搬動、並複製最熱的那幾個**。',
           en:'`--ep-size` splits the experts, `--enable-deepep-moe` swaps in the kernels, `--deepep-mode` picks throughput or latency. `--enable-eplb` and `--ep-num-redundant-experts` handle what a training-time balancing loss cannot: real traffic is skewed minute to minute, so experts have to be **moved between ranks and the hottest ones replicated**.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'SGLang 裡的名字', en:'what SGLang calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'vllm', label:{zh:'在 vLLM 裡', en:'in vLLM'},
     stage:{zh:'DP attention ＋ EP experts', en:'DP attention with EP experts'},
     view:VIEW, code:CODE_VLLM, build:vllmFrames,
     idea:{zh:'`--enable-expert-parallel` 把 experts 改成 EP，底下有 DeepEP／pplx-kernels。值得記的組合是 **attention 用 data parallel 複製、experts 用 expert parallel 切**：attention 沒有 expert 可分，而 MoE 權重不切放不下。要盯的數字是每個 rank 的負載分佈，不是平均吞吐。',
           en:'`--enable-expert-parallel` switches the experts to EP, with DeepEP or pplx-kernels underneath. The combination to remember is **attention replicated with data parallelism while the experts are sharded with expert parallelism**: attention has no experts to spread and the MoE weights do not fit otherwise. The number to watch is the per-rank load, not average throughput.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'vLLM 裡的名字', en:'what vLLM calls it'}],
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
