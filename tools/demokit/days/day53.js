// DAY: 53
// TITLE_ZH: Multi-Head Attention 與 Transformer Block
// TITLE_EN: Multi-head attention and the transformer block
// SUB_ZH: 一個 block 只有兩個混合器：attention 跨位置混、MLP 跨通道混，外面包殘差與 normalisation。h 個 head 不多花任何參數，只是把同樣四個矩陣切成 h 份；而幾乎每一種寫錯的方式，都會交出一個形狀完全正確的陣列。
// SUB_EN: A block is two mixers - attention across positions, an MLP across features - wrapped in residual connections and a normalisation. h heads cost no extra parameters; they cut the same four matrices h ways. And almost every way of getting it wrong returns an array of exactly the right shape.
// FOLDER: day%2053%20-%20multi-head%20attention%20and%20the%20transformer%20block
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
const SPLIT = {"X": [[2.04, -2.56, 0.42, -0.57, -0.45, -0.22, -2.02, -0.23], [-0.87, 3.32, 0.23, -0.35, -0.28, -0.67, -1.06, -0.39], [0.48, -0.24, 0.96, -0.2, 0.02, 1.55, 0.55, -0.51], [-0.18, 0.54, 1.94, -0.27, -0.24, 1.0, -0.89, -0.29]], "good": [[[2.04, -2.56, 0.42, -0.57], [-0.87, 3.32, 0.23, -0.35], [0.48, -0.24, 0.96, -0.2], [-0.18, 0.54, 1.94, -0.27]], [[-0.45, -0.22, -2.02, -0.23], [-0.28, -0.67, -1.06, -0.39], [0.02, 1.55, 0.55, -0.51], [-0.24, 1.0, -0.89, -0.29]]], "bad": [[[2.04, -2.56, 0.42, -0.57], [-0.45, -0.22, -2.02, -0.23], [-0.87, 3.32, 0.23, -0.35], [-0.28, -0.67, -1.06, -0.39]], [[0.48, -0.24, 0.96, -0.2], [0.02, 1.55, 0.55, -0.51], [-0.18, 0.54, 1.94, -0.27], [-0.24, 1.0, -0.89, -0.29]]], "max": 1.911, "rel": 153.0, "sameShape": true};
const HEADS = [{"h": 1, "dh": 64, "params": 16384}, {"h": 2, "dh": 32, "params": 16384}, {"h": 4, "dh": 16, "params": 16384}, {"h": 8, "dh": 8, "params": 16384}, {"h": 16, "dh": 4, "params": 16384}, {"h": 64, "dh": 1, "params": 16384}];
const PATT = {"prev": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], "first": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "best": [[0.205, 0.081, 0.157, 0.114, 0.11, 0.084, 0.107, 0.142], [0.209, 0.085, 0.162, 0.099, 0.091, 0.09, 0.115, 0.148], [0.163, 0.207, 0.15, 0.106, 0.139, 0.075, 0.036, 0.125], [0.141, 0.083, 0.239, 0.084, 0.107, 0.146, 0.058, 0.143], [0.142, 0.105, 0.095, 0.221, 0.127, 0.045, 0.108, 0.157], [0.117, 0.068, 0.191, 0.113, 0.087, 0.148, 0.113, 0.162], [0.047, 0.065, 0.152, 0.065, 0.05, 0.334, 0.18, 0.107], [0.163, 0.062, 0.074, 0.091, 0.083, 0.049, 0.283, 0.196]], "two": 0.0, "single": 0.1446, "zero": 0.8746, "rigged": 5.7e-30, "inputs": 12};
const CANC = [{"off": 0, "two": 1.019437, "one": 1.019437, "rel": 0.0, "distinct": 512}, {"off": 100, "two": 1.019437, "one": 1.018555, "rel": 0.000865, "distinct": 512}, {"off": 1000, "two": 1.019434, "one": 1.0625, "rel": 0.0422, "distinct": 511}, {"off": 10000, "two": 1.019408, "one": -8.0, "rel": 8.85, "distinct": 475}, {"off": 100000, "two": 1.019417, "one": 1024.0, "rel": 1000.0, "distinct": 304}, {"off": 1000000, "two": 1.020317, "one": 0.0, "rel": 1.0, "distinct": 77}];
const NORM = {"mean": 6.925, "std": 1.943, "lnMean": 2.6e-16, "lnStd": 1.0, "rmsMean": 0.9628, "rmsStd": 0.2701, "lnShift": 6.1e-15, "rmsShift": 0.8014};
const DEPTH = {"pre": [37.9, 46.3, 52.9, 58.8, 63.7, 71.6, 76.9, 81.9, 87.4, 91.3, 98.0, 97.7, 104.4, 111.6, 122.8, 127.1, 140.2, 147.6, 151.8, 153.7, 164.1, 169.5, 173.3, 174.1, 180.2, 184.5, 188.8, 193.0, 195.3, 196.2, 199.2, 197.0], "post": [32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0], "sigPre": [1.355, 1.682, 1.952, 2.243, 2.471, 2.854, 3.183, 3.58, 3.991, 4.387, 4.636, 4.604, 4.684, 4.925, 5.368, 5.599, 5.764, 5.988, 6.256, 6.52, 6.837, 6.872, 6.896, 7.029, 7.123, 7.155, 6.941, 6.934, 6.983, 7.304, 7.464, 7.505], "sigPost": [1.134, 1.162, 1.178, 1.237, 1.162, 1.243, 1.349, 1.441, 1.174, 1.161, 1.266, 1.45, 1.681, 1.614, 1.649, 1.541, 1.281, 1.32, 1.413, 1.314, 1.294, 0.943, 0.958, 0.867, 0.839, 0.694, 0.739, 0.931, 0.835, 0.901, 0.877, 1.104], "relPre": 0.0381, "relPost": 0.0345};
const GROW = {"start": 32.0, "plain": [39.8, 45.9, 52.0, 57.9, 71.5, 79.1, 85.0, 85.9, 93.7, 92.8, 100.0, 100.9, 115.2, 120.5, 122.6, 120.6, 123.9, 130.0, 135.3, 134.0, 136.1, 140.7, 142.7, 149.1, 150.7, 154.7, 160.9, 161.1, 169.6, 171.2, 176.2, 181.8], "scaled": [32.2, 32.2, 32.2, 32.3, 32.4, 32.5, 32.7, 32.8, 33.0, 33.1, 33.3, 33.4, 33.6, 33.8, 33.9, 33.9, 34.1, 34.2, 34.3, 34.4, 34.6, 34.7, 34.6, 34.8, 34.8, 34.9, 35.1, 35.3, 35.5, 35.7, 35.8, 35.8], "ratio": 5.7, "ratioScaled": 1.12, "sqrtL": 5.66};
const BUDGET = [{"d": 768, "h": 12, "attn": 2359296, "mlp": 4718592, "share": 67.0, "cross": 4608}, {"d": 2048, "h": 16, "attn": 16777216, "mlp": 33554432, "share": 67.0, "cross": 12288}, {"d": 4096, "h": 32, "attn": 67108864, "mlp": 134217728, "share": 67.0, "cross": 24576}, {"d": 8192, "h": 64, "attn": 268435456, "mlp": 536870912, "share": 67.0, "cross": 49152}];
const FLOPS = [{"n": 512, "lin": 103079215104, "quad": 2147483648, "share": 2.0}, {"n": 2048, "lin": 412316860416, "quad": 34359738368, "share": 7.7}, {"n": 8192, "lin": 1649267441664, "quad": 549755813888, "share": 25.0}, {"n": 32768, "lin": 6597069766656, "quad": 8796093022208, "share": 57.1}];
const LC189 = {"nums": [1, 2, 3, 4, 5, 6, 7], "k": 3, "steps": [{"a": [1, 2, 3, 4, 5, 6, 7], "lo": 0, "hi": 6, "what": "all"}, {"a": [7, 6, 5, 4, 3, 2, 1], "lo": 0, "hi": 2, "what": "first"}, {"a": [5, 6, 7, 4, 3, 2, 1], "lo": 3, "hi": 6, "what": "second"}, {"a": [5, 6, 7, 1, 2, 3, 4], "lo": -1, "hi": -1, "what": "done"}], "cycleBad": [5, 2, 1, 4, 3, 6], "cycleGood": [5, 6, 1, 2, 3, 4], "oneOk": [6, 1, 2, 3, 4, 5]};
const LC303 = {"nums": [-2, 0, 3, -5, 2, -1], "pre": [0, -2, -2, 1, -4, -2, -3], "queries": [{"l": 0, "r": 2, "v": 1}, {"l": 2, "r": 5, "v": -1}, {"l": 0, "r": 5, "v": -3}], "varTrue": 1.080859, "varPre": 28.860555};


/* ---------------------------------------------------------------- helpers */
function cellGrid(sh, x0, y0, M, cw, ch, colAt, fs){
  M.forEach((row, i) => row.forEach((v, j) => {
    sh.push(box(x0 + j * cw, y0 + (M.length - 1 - i) * ch, cw * .92, ch * .88,
                colAt(i, j, v), v === null ? '' : String(v), fs || .22));
  }));
  return sh;
}
function lineChart(sh, x0, y0, w, h, series, top){
  sh.push(S.e(x0, y0, x0 + w, y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(x0, y0, x0, y0 + h, {s:'ghost', arrow:false, w:.04}));
  series.forEach(ser => {
    const vals = ser.vals;
    for (let i = 1; i < vals.length; i++){
      if (ser.upto != null && i > ser.upto) break;
      sh.push(S.e(x0 + w * (i - 1) / (vals.length - 1), y0 + h * vals[i - 1] / top,
                  x0 + w * i / (vals.length - 1), y0 + h * vals[i] / top,
                  {s:ser.st, arrow:false, w:.06}));
    }
  });
  return sh;
}

/* ================================================================ tab 1: the block */
const CODE_BLOCK = [
  '# pre-LN: normalise what each branch reads',
  'x = x + attention(layer_norm(x))',
  'x = x + mlp(layer_norm(x))',
  '',
  '# post-LN: normalise after the add',
  'x = layer_norm(x + attention(x))',
  'x = layer_norm(x + mlp(x))'
];
function blockStage(o){
  const sh = [], W2 = 3.0;
  const pre = [{t:{zh:'LayerNorm', en:'LayerNorm'}, s:'hot'},
               {t:{zh:'multi-head attention', en:'multi-head attention'}, s:'ok'},
               {t:{zh:'+  殘差', en:'+  residual'}, s:'act'},
               {t:{zh:'LayerNorm', en:'LayerNorm'}, s:'hot'},
               {t:{zh:'MLP（4 倍寬）', en:'MLP  (4x wide)'}, s:'ok'},
               {t:{zh:'+  殘差', en:'+  residual'}, s:'act'}];
  const post = [{t:{zh:'multi-head attention', en:'multi-head attention'}, s:'ok'},
                {t:{zh:'+  殘差', en:'+  residual'}, s:'act'},
                {t:{zh:'LayerNorm', en:'LayerNorm'}, s:'hot'},
                {t:{zh:'MLP（4 倍寬）', en:'MLP  (4x wide)'}, s:'ok'},
                {t:{zh:'+  殘差', en:'+  residual'}, s:'act'},
                {t:{zh:'LayerNorm', en:'LayerNorm'}, s:'hot'}];
  const rows = o.post ? post : pre;
  rows.forEach((r, i) => {
    const y = .95 + i * .70;
    const on = o.upto == null || i <= o.upto;
    sh.push(box(1.20, y, W2, .56, on ? r.s : 'idle', r.t, .28));
  });
  sh.push(fitT(2.70, .62, 'x', {c:COL.tealL, fs:.30}));
  sh.push(S.e(.75, .95, .75, .95 + rows.length * .70 - .14,
              {s:o.post ? 'bad' : 'ok', arrow:true, w:.06}));
  sh.push(fitT(.62, .72, o.post ? {zh:'這條路每一層都被正規化一次', en:'a norm on this path, every block'}
                                : {zh:'這條路一次都沒有被正規化', en:'no norm on this path'},
               {anchor:'start', c:o.post ? COL.red : COL.tealL, fs:.26}));
  side(sh, 5.10, 1.25, o.notes || [], .52);
  return caps(sh, o);
}
function blockFrames(v){
  const F = new Frames(), post = v === 1;
  const p = (o) => F.push({shapes:blockStage(Object.assign({post:post}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:-1, line:post ? 5 : 1,
    cap:post ? {zh:'post-LN：先加再正規化', en:'post-LN: add first, normalise after'}
             : {zh:'pre-LN：先正規化再進分支', en:'pre-LN: normalise what the branch reads'},
    notes:[[{zh:'一個 block 做兩件事：', en:'a block does two things:'}, COL.grey, .28],
           [{zh:'attention 跨「位置」混合', en:'attention mixes across positions'}, COL.tealL, .28],
           [{zh:'MLP 跨「通道」混合', en:'the MLP mixes across features'}, COL.tealL, .28],
           [{zh:'兩個分支都寫回同一條殘差流', en:'both branches write into one residual stream'}, COL.grey, .26]],
    panels:[{lbl:'shape', chips:[chip('(16, 64)', 'ok'), chip('-> (16, 64)', 'ok')]}],
    msg:{zh:'一個 block 從頭到尾不改變殘差流的形狀——輸入 (16, 64)、輸出 (16, 64)。這就是為什麼可以疊 32 層：每一層都是「在同一條流上再加一點東西」。attention 負責跨位置混合（每個通道用同一組權重），MLP 負責跨通道混合（每個位置用同一組權重）。',
         en:'A block never changes the shape of the residual stream: (16, 64) in, (16, 64) out. That is what lets you stack 32 of them, because every block is just "add something else into the same stream". Attention mixes across positions, using the same weights for every channel; the MLP mixes across channels, using the same weights at every position.'}});
  for (let i = 0; i < 6; i++){
    const label = (post ? ['attention', '+ residual', 'LayerNorm', 'MLP', '+ residual', 'LayerNorm']
                        : ['LayerNorm', 'attention', '+ residual', 'LayerNorm', 'MLP', '+ residual'])[i];
    p({upto:i, line:post ? (i < 3 ? 5 : 6) : (i < 3 ? 1 : 2),
      cap:{zh:'第 ' + (i + 1) + ' 步：' + label, en:'step ' + (i + 1) + ': ' + label},
      panels:[{lbl:'step', chips:[chip(label, 'hot')]}],
      msg:msgFor(post, i)});
  }
  p({upto:5, line:post ? 6 : 2,
    cap:{zh:'差別只有一行的位置', en:'the difference is where one line goes'},
    panels:[{lbl:'pre-LN ||x||', chips:[chip(String(DEPTH.pre[31]), 'act')]},
            {lbl:'post-LN ||x||', chips:[chip(String(DEPTH.post[31]), 'ok')]}],
    msg:{zh:'兩種寫法用的是同樣的零件，差別只有 LayerNorm 放在加法的哪一邊。post-LN 把正規化放在殘差相加之後，所以從輸入到輸出的那條路每一層都會被正規化一次，疊 32 層之後 ||x|| 永遠是 ' + DEPTH.post[31] + '；pre-LN 只正規化「分支讀到的東西」，殘差路徑本身沒有被動過，32 層之後長到 ' + DEPTH.pre[31] + '。也因為沒人管它，pre-LN 的堆疊最後一定要補一個 LayerNorm 再接輸出層。',
         en:'Both use the same parts; the only difference is which side of the addition the LayerNorm sits on. Post-LN normalises after the residual add, so the path from input to output passes a norm in every block and ||x|| is pinned at ' + DEPTH.post[31] + ' after 32 of them. Pre-LN normalises only what each branch reads, leaving the residual path untouched, so it grows to ' + DEPTH.pre[31] + '. Because nothing bounds it, a pre-LN stack has to end with one final LayerNorm before the output head.'}});
  return F.list;
}
function msgFor(post, i){
  const preM = [
    {zh:'先正規化。注意被正規化的是「attention 要讀的那份複本」，不是殘差流本身。',
     en:'Normalise first - and what gets normalised is the copy attention reads, not the residual stream itself.'},
    {zh:'attention 跨位置混合：每個位置變成其他位置的加權平均，而每個通道都用同一組權重。',
     en:'Attention mixes across positions: every position becomes a weighted average of the others, with the same weights for every channel.'},
    {zh:'加回去。殘差流是一條從輸入直通輸出的路，每個分支只是往上面「再寫一點」。',
     en:'Add it back. The residual stream is a path straight from the input to the output, and each branch only writes a little more onto it.'},
    {zh:'第二個正規化，同樣只作用在 MLP 要讀的那份複本上。',
     en:'The second normalisation, again only on the copy the MLP reads.'},
    {zh:'MLP 跨通道混合：先放大到 4 倍寬、過一個非線性、再投影回來；每個位置各做各的，位置之間完全不互通。',
     en:'The MLP mixes across features: expand to 4x width, apply a non-linearity, project back. It runs on each position independently and never moves information between positions.'},
    {zh:'再加回去，block 結束。輸出的形狀跟輸入一模一樣。',
     en:'Add it back and the block is done, with the same shape it started with.'}];
  const postM = [
    {zh:'post-LN 直接拿殘差流去做 attention，沒有先正規化。',
     en:'Post-LN feeds the residual stream straight into attention, with no normalisation first.'},
    {zh:'加回去——但這還不是 block 的輸出。',
     en:'Add it back - but this is not the output of the block yet.'},
    {zh:'正規化放在這裡：它作用在「殘差相加之後」，所以主幹本身被改寫了。',
     en:'The normalisation goes here, after the residual add, so the trunk itself is rewritten.'},
    {zh:'MLP 一樣直接吃正規化過的主幹。',
     en:'The MLP reads the normalised trunk directly.'},
    {zh:'第二次相加。', en:'The second add.'},
    {zh:'再正規化一次。所以 post-LN 的 block 輸出必定是「剛被正規化過」的，||x|| 永遠是 ' + DEPTH.post[0] + '。',
     en:'And normalise again. A post-LN block therefore always outputs something that was just normalised, which is why ||x|| is always ' + DEPTH.post[0] + '.'}];
  return (post ? postM : preM)[i];
}

/* ================================================================ tab 2: splitting heads */
const CODE_SPLIT = [
  '# correct: group the FEATURE axis, then move it to the front',
  'X.reshape(n, h, d // h).transpose(1, 0, 2)',
  '',
  '# wrong: the same shape, from the wrong axis',
  'X.reshape(h, n, d // h)',
  '',
  '# both are (h, n, d_head).  nothing raises.'
];
function splitStage(o){
  const sh = [], cw = .55, ch = .52, n = SPLIT.X.length, d = SPLIT.X[0].length;
  const colAt = (i, j) => (o.mode === 'bad'
      ? (i < n / 2 ? 'hot' : 'ok')
      : (j < d / 2 ? 'hot' : 'ok'));
  cellGrid(sh, 1.05, 2.30, SPLIT.X, cw, ch, (i, j, v) => colAt(i, j), .22);
  sh.push(fitT(1.05 + d * cw / 2, 2.30 + n * ch + .30,
               o.mode === 'bad' ? {zh:'X.reshape(2, 4, 4)', en:'X.reshape(2, 4, 4)'}
                                : {zh:'split_heads(X, 2)', en:'split_heads(X, 2)'},
               {c:COL.pale, fs:.30}));
  sh.push(fitT(1.05 + d * cw / 2, 1.90,
               o.mode === 'bad' ? {zh:'head 0 拿到：前兩個 token 的全部通道', en:'head 0 gets: the first two tokens, all their features'}
                                : {zh:'head 0 拿到：每個 token 的前四個通道', en:'head 0 gets: the first four features of every token'},
               {c:o.mode === 'bad' ? COL.red : COL.hot, fs:.28}));
  side(sh, 6.10, 2.60, o.notes || [], .50);
  return caps(sh, o);
}
function splitFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:splitStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({mode:'good', line:1,
    cap:{zh:'4 個 token、8 個通道、2 個 head', en:'4 tokens, 8 features, 2 heads'},
    panels:[{lbl:'head 0', chips:[chip('features 0-3', 'hot')]},
            {lbl:'head 1', chips:[chip('features 4-7', 'ok')]}],
    msg:{zh:'多頭要切的是「通道」這個軸：每個 head 拿到每一個 token 的一塊通道，所以每個 head 都看得到整條序列，只是看的是不同的特徵子空間。reshape 成 (n, h, d_head) 之後再 transpose 把 head 軸搬到最前面。',
         en:'Splitting heads means cutting the FEATURE axis: each head gets a slice of the features of every token, so every head still sees the whole sequence, just through a different subspace. Reshape to (n, h, d_head) and then move the head axis to the front.'}});
  p({mode:'bad', line:4,
    cap:{zh:'少了那個 transpose 會怎樣', en:'what happens without the transpose'},
    cap2:{zh:'形狀一樣、不會丟例外、梯度照流', en:'same shape, no exception, gradients still flow'},
    panels:[{lbl:'head 0', chips:[chip('tokens 0-1', 'bad')]},
            {lbl:'head 1', chips:[chip('tokens 2-3', 'bad')]}],
    msg:{zh:'直接 reshape 成 (h, n, d_head) 也會給你一個形狀正確的陣列，但它切的是「位置」：head 0 拿到前兩個 token 的全部通道、head 1 拿到後兩個。而且它切出來的那些「位置」其實是半個 token（token 0 的前四個通道、token 0 的後四個通道、token 1 的前四個⋯），merge 回去之後連輸出的每一列都不再對應一個 token。',
         en:'Reshaping straight to (h, n, d_head) also gives an array of the right shape, but it cuts positions: head 0 gets the first two tokens with all their features, head 1 gets the last two. Worse, the positions it attends over are half-tokens - token 0 features 0-3, token 0 features 4-7, token 1 features 0-3 - so after merging, the output rows no longer correspond to tokens at all.'}});
  p({mode:'bad', line:6,
    cap:{zh:'這就是「形狀對、答案錯」', en:'the right shape and the wrong answer'},
    panels:[{lbl:'same shape', chips:[chip(SPLIT.sameShape ? 'True' : 'False', 'ok')]},
            {lbl:'output gap', chips:[chip(SPLIT.max.toFixed(3), 'bad'),
                                      chip(SPLIT.rel + '%', 'bad')]}],
    msg:{zh:'在 16 個 token、64 維、8 個 head 的設定下量：兩種切法的層輸出最大差 ' + SPLIT.max.toFixed(3) + '，整體差 ' + SPLIT.rel + '%（以輸出自己的範數為分母）。沒有例外、沒有形狀錯誤，模型照樣能訓練，只是比較差而且你找不到原因——這類 bug 唯一抓得到的方法是寫一個「head j 應該等於 X[:, j*d_head:(j+1)*d_head]」的 assert。',
         en:'Measured on 16 tokens, 64 features and 8 heads: the two splittings put the layer output a maximum of ' + SPLIT.max.toFixed(3) + ' apart, which is ' + SPLIT.rel + '% of the output norm. No exception, no shape error, and a model built on it still trains - just worse, with nothing to point at. The only thing that catches this is an assert that head j equals X[:, j*d_head:(j+1)*d_head].'}});
  return F.list;
}

/* ================================================================ tab 3: what heads buy */
const CODE_HEADS = [
  '# h never changes the parameter count',
  'Wq = Wk = Wv = Wo = (d_model, d_model)',
  '',
  '# it changes how many n x n weight matrices',
  '# the layer gets to produce: one per head.',
  '#',
  '# each head applies ITS matrix to every channel it carries.'
];
function headsStage(o){
  const sh = [], cw = .30, n = PATT.prev.length;
  const mats = [{M:PATT.prev, t:{zh:'head 1：前一個 token', en:'head 1: the previous token'}, s:'ok'},
                {M:PATT.first, t:{zh:'head 2：第一個 token', en:'head 2: the first token'}, s:'hot'},
                {M:PATT.best, t:{zh:'單一 head 能做到的最好', en:'the best a single head can do'}, s:'bad'}];
  mats.forEach((m, k) => {
    if (k > o.upto) return;
    const x0 = .95 + k * 3.00;
    m.M.forEach((row, i) => row.forEach((v, j) => {
      sh.push(S.r(x0 + j * cw, 2.00 + (n - 1 - i) * cw, cw * .88, cw * .88,
                  v > 0.02 ? m.s : 'idle', null));
    }));
    sh.push(fitT(x0 + n * cw / 2, 1.70, m.t, {c:COL.pale, fs:.26}));
  });
  if (o.table){
    sh.push(fitT(4.9, 5.00, {zh:'同樣四個矩陣，h 只決定切成幾份', en:'the same four matrices, cut h ways'},
                 {c:COL.grey, fs:.26}));
  }
  return caps(sh, o);
}
function headsFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:headsStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:-1, line:1, table:true,
    cap:{zh:'h 個 head 不多花任何參數', en:'h heads cost no extra parameters'},
    panels:[{lbl:'h = 1', chips:[chip(HEADS[0].params.toLocaleString(), 'ok'), chip('d_head 64', 'dim')]},
            {lbl:'h = 8', chips:[chip(HEADS[3].params.toLocaleString(), 'ok'), chip('d_head 8', 'dim')]}],
    msg:{zh:'Wq/Wk/Wv/Wo 永遠是四個 d_model x d_model 的矩陣，' + HEADS[0].params.toLocaleString() + ' 個參數，跟 h 無關。h 只決定把它們切成幾個子空間：h=1 時 d_head 64、h=8 時 d_head 8、h=64 時每個 head 只剩一維。換到的是「每層產生 h 個 n x n 權重矩陣」而不是一個。',
         en:'Wq, Wk, Wv and Wo are always four d_model by d_model matrices - ' + HEADS[0].params.toLocaleString() + ' parameters here, whatever h is. All h does is cut them into subspaces: d_head 64 at h=1, 8 at h=8, and a single dimension each at h=64. What you buy is h weight matrices of size n by n per layer instead of one.'}});
  p({upto:0, line:4,
    cap:{zh:'一個需要「同時做兩件事」的目標', en:'a target that needs two mixings at once'},
    panels:[{lbl:'channels 0-3', chips:[chip('previous token', 'ok')]}],
    msg:{zh:'設一個目標：一半的通道要「前一個 token 的值」，另一半要「第一個 token 的值」。兩個都是 attention 做得到的事——第一個 head 的權重矩陣就長這樣，對角線往下移一格（位置 0 沒有前一個，所以看自己）。',
         en:'Here is a target: half the channels want the value at the previous token, the other half want the value at the first token. Both are things attention can do - this is the first head, the diagonal shifted down by one, with position 0 attending to itself because it has no previous token.'}});
  p({upto:1, line:4,
    cap:{zh:'第二個 head 做另外一件', en:'the second head does the other one'},
    panels:[{lbl:'channels 4-7', chips:[chip('first token', 'hot')]},
            {lbl:'two heads', chips:[chip('MSE ' + PATT.two.toFixed(1), 'ok')]}],
    msg:{zh:'第二個 head 的權重全部壓在第 0 行。兩個 head 各做一件、各自只負責自己那一半通道，在 ' + PATT.inputs + ' 組不同輸入上都把目標做到 MSE ' + PATT.two.toFixed(1) + '——完全重現。',
         en:'The second head puts all of its weight on column 0. Two heads, one pattern each, each carrying its own half of the channels, reproduce the target at MSE ' + PATT.two.toFixed(1) + ' on all ' + PATT.inputs + ' inputs - exactly.'}});
  p({upto:2, line:6,
    cap:{zh:'一個 head 做得到嗎', en:'can one head do it'},
    panels:[{lbl:'best single head', chips:[chip('MSE ' + PATT.single.toFixed(4), 'bad')]},
            {lbl:'predict zero', chips:[chip('MSE ' + PATT.zero.toFixed(4), 'dim')]}],
    msg:{zh:'不行，量出來是 MSE ' + PATT.single.toFixed(4) + '（全部猜 0 是 ' + PATT.zero.toFixed(4) + '）。這一輪給單一 head 的東西比真的 head 還多：它有自己的 d x d 通道矩陣（那就是 Wv 和 Wo 合起來的效果），而且每組輸入都可以自由挑一個注意力矩陣——真的 head 挑不了，它的權重得從 Q、K 算出來。**不給它那個通道矩陣的比較是作弊的**：n = d = 8 時，單一組輸入只要解 (A V) M = target 就能把隨機的 A 配到 MSE ' + PATT.rigged.toExponential(0) + '，那是在講這個玩具不是在講 head。它真正做不到的是「對不同通道用不同的注意力矩陣」，而那正是這個目標要的。',
         en:'No, and the number is ' + PATT.single.toFixed(4) + ' against ' + PATT.zero.toFixed(4) + ' for predicting zero. This run gives the single head more than a real one gets: its own d by d channel matrix - which is what Wv and Wo amount to - and a free choice of attention matrix per input, which a real head cannot have because its weights come out of Q and K. Without that channel matrix the comparison would be rigged: at n = d = 8 a single input can be fitted to MSE ' + PATT.rigged.toExponential(0) + ' by solving (A V) M = target for M alone with a random A, which says nothing about heads. What one head still cannot do is use a DIFFERENT attention matrix for different channels, and that is exactly what the target asks for.'}});
  return F.list;
}

/* ================================================================ tab 4: normalisation */
const CODE_NORM = [
  '# two passes: subtract the mean, then measure the spread',
  'mu  = x.mean(-1)',
  'var = ((x - mu) ** 2).mean(-1)',
  '',
  '# one pass, same algebra, different arithmetic',
  'var = (x ** 2).mean(-1) - mu ** 2',
  '#     two large numbers differenced to get a small one'
];
function normStage(o){
  const sh = [], X0 = 1.40, Y0 = 1.55, WD = 6.9, HT = 2.55;
  sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  const ly = (v) => Y0 + HT * (Math.max(-4, Math.log10(Math.max(v, 1e-4))) + 4) / 8;
  CANC.forEach((r, i) => {
    if (i > o.upto) return;
    const x = X0 + WD * i / (CANC.length - 1);
    if (i > 0){
      const q = CANC[i - 1];
      sh.push(S.e(X0 + WD * (i - 1) / (CANC.length - 1), ly(q.rel), x, ly(r.rel),
                  {s:'bad', arrow:false, w:.06}));
    }
    sh.push(S.c(x, ly(r.rel), .12, r.one <= 0 ? 'bad' : 'hot', null));
    sh.push(fitT(x, Y0 - .28, r.off.toLocaleString(), {c:COL.grey, fs:.24}));
  });
  sh.push(S.e(X0, ly(1), X0 + WD, ly(1), {s:'ghost', arrow:false, w:.03}));
  sh.push(fitT(X0 + .15, ly(1) + .24, {zh:'錯得跟答案本身一樣大', en:'as wrong as the answer itself'},
               {anchor:'start', c:COL.grey, fs:.24}));
  sh.push(fitT(X0 + .15, Y0 + HT + .30,
               {zh:'一趟版本的相對誤差（兩趟版本整排都在 ' + CANC[0].two.toFixed(4) + ' 附近）',
                en:'relative error of the one-pass variance (two passes stay at about ' + CANC[0].two.toFixed(4) + ')'},
               {anchor:'start', c:COL.grey, fs:.25}));
  sh.push(fitT(X0 + WD / 2, Y0 - .72, {zh:'加到每個值上的常數（float32）', en:'constant added to every value (float32)'},
               {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function normFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:normStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'同樣 512 個數字，只是整排加上一個常數', en:'the same 512 numbers, shifted by a constant'},
    panels:[{lbl:'offset 0', chips:[chip(CANC[0].two.toFixed(6), 'ok'), chip(CANC[0].one.toFixed(6), 'ok')]}],
    msg:{zh:'沒有位移時兩種算法逐位元相同（' + CANC[0].two.toFixed(6) + '）。一趟版本少掃一次資料，所以它一直有人寫。',
         en:'With no offset the two agree to the last bit (' + CANC[0].two.toFixed(6) + '). The one-pass version saves a sweep over the data, which is why it keeps getting written.'}});
  CANC.slice(1).forEach((r, k) => {
    const i = k + 1;
    p({upto:i, line:5,
      cap:{zh:'殘差流本來就會累積一個平均值', en:'a residual stream accumulates a mean offset'},
      panels:[{lbl:'offset', chips:[chip(r.off.toLocaleString(), 'hot')]},
              {lbl:'two-pass / one-pass', chips:[chip(r.two.toFixed(6), 'ok'),
                                                 chip(r.one.toFixed(6), r.one <= 0 ? 'bad' : 'dim')]},
              {lbl:'distinct float32', chips:[chip(String(r.distinct), 'dim')]}],
      msg:r.one <= 0
        ? {zh:'位移 ' + r.off.toLocaleString() + '：一趟版本給 ' + r.one.toFixed(1) + '——變異數是負的，開根號就是 nan。同一份資料兩趟版本仍然回報 ' + r.two.toFixed(6) + '，而且這個陣列裡還有 ' + r.distinct + ' 個相異的 float32 值。壞掉的不是資料，是那條恆等式。',
           en:'At an offset of ' + r.off.toLocaleString() + ' the one-pass version returns ' + r.one.toFixed(1) + ' - a negative variance, whose square root is nan. The two-pass version still reports ' + r.two.toFixed(6) + ' from the same array, which still holds ' + r.distinct + ' distinct float32 values. The data is fine; the identity is not.'}
        : {zh:'位移 ' + r.off.toLocaleString() + '：一趟版本已經偏到 ' + r.one.toFixed(6) + '（相對誤差 ' + (r.rel * 100).toPrecision(2) + '%），而且不會報錯——根號裡的 eps 讓它保持有限，LayerNorm 只是用了錯的尺度。',
           en:'At an offset of ' + r.off.toLocaleString() + ' the one-pass version is already at ' + r.one.toFixed(6) + ' (' + (r.rel * 100).toPrecision(2) + '% off), and nothing raises: the eps inside the square root keeps it finite, and the layer simply normalises by the wrong number.'}});
  });
  p({upto:CANC.length - 1, line:2,
    cap:{zh:'LayerNorm 與 RMSNorm 放棄的是不同的東西', en:'LayerNorm and RMSNorm give up different things'},
    panels:[{lbl:'shift by +100', chips:[chip('LN ' + NORM.lnShift.toExponential(0), 'ok'),
                                         chip('RMS ' + NORM.rmsShift.toFixed(4), 'bad')]},
            {lbl:'row mean / std', chips:[chip(String(NORM.mean), 'dim'), chip(String(NORM.std), 'dim')]}],
    msg:{zh:'LayerNorm 減平均再除標準差，所以把整列加 100 之後輸出只變 ' + NORM.lnShift.toExponential(0) + '；RMSNorm 不減平均，同樣的位移讓它變 ' + NORM.rmsShift.toFixed(4) + '。RMSNorm 換到的是少一次 reduction、少一組 beta 參數，放棄的是平移不變性。兩者都對「整列乘上一個常數」免疫，而那正是殘差流真的會做的事。',
         en:'LayerNorm subtracts the mean and divides by the standard deviation, so shifting a row by +100 moves its output by ' + NORM.lnShift.toExponential(0) + '. RMSNorm never subtracts the mean, and the same shift moves it by ' + NORM.rmsShift.toFixed(4) + '. What RMSNorm buys is one fewer reduction and one fewer parameter vector; what it gives up is shift invariance. Both are invariant to scaling a row, which is the thing the residual stream actually does.'}});
  return F.list;
}

/* ================================================================ tab 5: depth */
const CODE_DEPTH = [
  'x0 = input                      # 32 blocks stacked',
  'x1 = x0 + eps * direction       # a tiny nudge',
  'for block in blocks:',
  '    x0, x1 = block(x0), block(x1)',
  '',
  '# influence = ||x1 - x0|| / eps   (forward-mode mirror',
  '#                                  of the gradient at the bottom)'
];
function depthStage(o){
  const sh = [], X0 = 1.40, Y0 = 1.55, WD = 6.9, HT = 2.55, TOP = 210;
  const sers = [];
  if (o.show.indexOf('pre') >= 0) sers.push({vals:DEPTH.pre, st:'hot', upto:o.upto});
  if (o.show.indexOf('post') >= 0) sers.push({vals:DEPTH.post, st:'ok', upto:o.upto});
  if (o.show.indexOf('grow') >= 0) sers.push({vals:GROW.plain, st:'act', upto:o.upto});
  if (o.show.indexOf('scaled') >= 0) sers.push({vals:GROW.scaled, st:'done', upto:o.upto});
  lineChart(sh, X0, Y0, WD, HT, sers.map(s => ({vals:s.vals, st:s.st, upto:s.upto})), TOP);
  [1, 8, 16, 32].forEach(b => sh.push(fitT(X0 + WD * (b - 1) / 31, Y0 - .28, String(b),
                                           {c:COL.grey, fs:.24})));
  [0, 100, 200].forEach(v => sh.push(fitT(X0 - .18, Y0 + HT * v / TOP, String(v),
                                          {anchor:'end', c:COL.grey, fs:.24})));
  sh.push(fitT(X0 + WD / 2, Y0 - .72, {zh:'疊了幾個 block', en:'blocks stacked'}, {c:COL.grey, fs:.26}));
  sh.push(fitT(X0 + .15, Y0 + HT + .30, {zh:'殘差流的範數', en:'norm of the residual stream'},
               {anchor:'start', c:COL.grey, fs:.25}));
  side(sh, 8.55, 2.30, o.notes || [], .46);
  return caps(sh, o);
}
function depthFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:depthStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({show:['post'], upto:31, line:3,
    cap:{zh:'post-LN：每個 block 之後都剛被正規化過', en:'post-LN: just normalised, after every block'},
    panels:[{lbl:'||x|| after 1 / 32', chips:[chip(String(DEPTH.post[0]), 'ok'),
                                              chip(String(DEPTH.post[31]), 'ok')]}],
    msg:{zh:'post-LN 的 block 最後一個動作就是 LayerNorm，所以不管疊幾層，輸出的範數都不再跟深度有關——這裡 gamma = 1、beta = 0，所以它剛好是 ' + DEPTH.post[31] + '（＝sqrt(16 x 64)）。這條線是平的，而且它是被壓平的。',
         en:'The last thing a post-LN block does is normalise, so however many you stack, the norm coming out stops depending on the depth - with gamma = 1 and beta = 0 as initialised here it is exactly ' + DEPTH.post[31] + ', which is sqrt(16 x 64). The line is flat, and it is flat because something flattens it.'}});
  p({show:['post', 'pre'], upto:31, line:3,
    cap:{zh:'pre-LN：殘差路徑沒有人管', en:'pre-LN: nothing touches the residual path'},
    panels:[{lbl:'pre-LN ||x||', chips:[chip(String(DEPTH.pre[0]), 'hot'), chip(String(DEPTH.pre[31]), 'hot')]},
            {lbl:'post-LN ||x||', chips:[chip(String(DEPTH.post[31]), 'ok')]}],
    msg:{zh:'pre-LN 只正規化「分支讀到的複本」，主幹從輸入一路直通輸出，所以範數從 ' + DEPTH.pre[0] + ' 長到 ' + DEPTH.pre[31] + '。這就是為什麼 pre-LN 的堆疊一定要在最後補一個 LayerNorm 才接輸出層——不然送進去的東西大小隨深度跑。',
         en:'Pre-LN normalises only the copy each branch reads, leaving the trunk running straight from input to output, so the norm grows from ' + DEPTH.pre[0] + ' to ' + DEPTH.pre[31] + '. That is why a pre-LN stack needs one final LayerNorm before the output head: otherwise what arrives there scales with depth.'}});
  p({show:['post', 'pre'], upto:31, line:5,
    cap:{zh:'那「訊號」呢', en:'and what about the signal'},
    panels:[{lbl:'influence pre / post', chips:[chip(DEPTH.sigPre[31].toFixed(2), 'hot'),
                                                chip(DEPTH.sigPost[31].toFixed(2), 'ok')]},
            {lbl:'as a share of ||x||', chips:[chip(DEPTH.relPre.toFixed(4), 'dim'),
                                               chip(DEPTH.relPost.toFixed(4), 'dim')]}],
    msg:{zh:'在輸入上推一個很小的擾動，看它到頂端還剩多少：pre-LN 放大成 ' + DEPTH.sigPre[31].toFixed(2) + ' 倍、post-LN 是 ' + DEPTH.sigPost[31].toFixed(2) + ' 倍。但 pre-LN 的流本身也變大了，所以換算成「佔整條流的比例」兩邊幾乎一樣（' + DEPTH.relPre.toFixed(4) + ' 對 ' + DEPTH.relPost.toFixed(4) + '）——**這個量測並沒有證明 pre-LN 帶比較多訊號**，它證明的是結構差異：pre-LN 留了一條完全沒被正規化的路，post-LN 每層都在那條路上放一個正規化。',
         en:'Push a tiny perturbation into the input and see how much is left at the top: pre-LN amplifies it to ' + DEPTH.sigPre[31].toFixed(2) + 'x, post-LN to ' + DEPTH.sigPost[31].toFixed(2) + 'x. But the pre-LN stream itself grew, so as a share of the stream the two are nearly identical (' + DEPTH.relPre.toFixed(4) + ' against ' + DEPTH.relPost.toFixed(4) + '). This measurement does NOT show pre-LN carrying more signal. What it shows is the structural difference: pre-LN leaves one path that is never normalised, post-LN puts a normalisation on that path in every block.'}});
  p({show:['grow'], upto:31, line:3,
    cap:{zh:'而那條沒人管的流，是一個連加', en:'and that unbounded stream is a running sum'},
    panels:[{lbl:'||x|| at 0 / 32', chips:[chip(String(GROW.start), 'act'),
                                           chip(String(GROW.plain[31]), 'act')]},
            {lbl:'growth', chips:[chip(GROW.ratio + 'x', 'bad'), chip('sqrt(32) = ' + GROW.sqrtL, 'dim')]}],
    msg:{zh:'32 個 block、每個 block 兩個分支都往同一條流上加東西，範數從 ' + GROW.start + ' 長到 ' + GROW.plain[31] + '＝' + GROW.ratio + ' 倍——正是「把 32 份互相獨立的東西加進同一個連加」預期的 sqrt(32) = ' + GROW.sqrtL + '。',
         en:'With 32 blocks, each writing two branches into the same stream, the norm goes from ' + GROW.start + ' to ' + GROW.plain[31] + ', a factor of ' + GROW.ratio + ' - which is the sqrt(32) = ' + GROW.sqrtL + ' expected of a running sum of 32 independent pieces.'}});
  p({show:['grow', 'scaled'], upto:31, line:3,
    cap:{zh:'所以 init 要先把它付掉', en:'so the initialisation pays for it up front'},
    panels:[{lbl:'unscaled', chips:[chip(GROW.ratio + 'x', 'bad')]},
            {lbl:'1/sqrt(2L) per branch', chips:[chip(GROW.ratioScaled + 'x', 'ok')]}],
    msg:{zh:'把每個分支寫進去的東西先乘上 1/sqrt(2L)（L 是層數、每層兩個分支），32 層之後範數只變 ' + GROW.ratioScaled + ' 倍。「殘差分支要縮放初始化」講的就是這件事：流是一個連加，而連加會長大。',
         en:'Scale what each branch writes by 1/sqrt(2L) - L layers, two branches each - and after 32 blocks the norm has moved by only ' + GROW.ratioScaled + 'x. That is the whole content of "scale the residual branches at initialisation": the stream is a running sum, and running sums grow.'}});
  return F.list;
}

/* ================================================================ tab 6: the budget */
const CODE_BUDGET = [
  'attention params = 4 * d_model * d_model',
  'mlp params       = 2 * d_model * d_ff      # d_ff = 4 * d_model',
  '',
  'linear MACs = n * (4*d^2 + 2*d*d_ff)',
  'quad   MACs = 2 * n * n * d',
  '',
  '# equal when n = (4*d^2 + 2*d*d_ff) / (2*d) = 6 * d_model'
];
function budgetStage(o){
  const sh = [];
  if (o.mode === 'params'){
    const r = BUDGET[2], mx = r.mlp, base = 4.10, maxh = 2.20;
    [{v:r.attn, s:'ok', t:{zh:'attention', en:'attention'}},
     {v:r.mlp, s:'hot', t:{zh:'MLP', en:'MLP'}}].forEach((b, i) => {
      const hgt = maxh * b.v / mx, x = 2.10 + i * 2.10;
      sh.push(S.r(x, base - hgt, 1.20, hgt, b.s, null));
      sh.push(fitT(x + .60, base - hgt - .26, (b.v / 1e6).toFixed(0) + 'M',
                   {c:b.s === 'hot' ? COL.hot : COL.tealL, fs:.30}));
      sh.push(fitT(x + .60, base + .32, b.t, {c:COL.grey, fs:.28}));
    });
    sh.push(fitT(6.90, 2.60, {zh:'d_model 4096 的一個 block：\n' + r.share + '% 是 MLP',
                              en:'one block at d_model 4096:\n' + r.share + '% of it is the MLP'},
                 {c:COL.pale, fs:.28}));
  } else {
    const X0 = 1.40, Y0 = 1.55, WD = 6.9, HT = 2.55;
    sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'ghost', arrow:false, w:.04}));
    sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
    FLOPS.forEach((r, i) => {
      if (i > o.upto) return;
      const x = X0 + WD * i / (FLOPS.length - 1), hgt = Math.max(.06, HT * r.share / 100);
      sh.push(S.r(x - .45, Y0, .90, hgt, r.share > 50 ? 'bad' : 'ok', null));
      sh.push(fitT(x, Y0 + hgt + .26, r.share.toFixed(1) + '%', {c:COL.tealL, fs:.28}));
      sh.push(fitT(x, Y0 - .28, r.n.toLocaleString(), {c:COL.grey, fs:.24}));
    });
    sh.push(fitT(X0 + WD / 2, Y0 - .72, {zh:'context 長度', en:'context length'}, {c:COL.grey, fs:.26}));
    sh.push(fitT(X0 + .15, Y0 + HT + .30,
                 {zh:'block 裡「二次方」那部分的佔比（d_model 4096）',
                  en:'share of the block that is quadratic (d_model 4096)'},
                 {anchor:'start', c:COL.grey, fs:.25}));
  }
  return caps(sh, o);
}
function budgetFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:budgetStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({mode:'params', line:1,
    cap:{zh:'一個 block 的參數在哪裡', en:'where a block keeps its parameters'},
    panels:[{lbl:'attention', chips:[chip(BUDGET[2].attn.toLocaleString(), 'ok')]},
            {lbl:'MLP', chips:[chip(BUDGET[2].mlp.toLocaleString(), 'hot'),
                               chip(BUDGET[2].share + '%', 'hot')]}],
    msg:{zh:'attention 是四個 d_model x d_model，MLP 是兩個 d_model x 4·d_model，所以 MLP 永遠是 attention 的兩倍：每個 block 有 ' + BUDGET[2].share + '% 的參數在那個「大家最少討論」的前饋層裡。這個比例跟 d_model 無關，四種大小都是 ' + BUDGET[0].share + '%。',
         en:'Attention is four d_model by d_model matrices; the MLP is two of d_model by 4 d_model, so the MLP is always twice attention: ' + BUDGET[2].share + '% of every block sits in the feed-forward part people discuss least. The ratio does not depend on d_model - all four sizes come out at ' + BUDGET[0].share + '%.'}});
  FLOPS.forEach((r, i) => {
    p({mode:'flops', upto:i, line:4,
      cap:{zh:'而計算量的形狀跟 context 有關', en:'the compute, though, depends on the context'},
      panels:[{lbl:'context', chips:[chip(r.n.toLocaleString(), 'hot')]},
              {lbl:'quadratic share', chips:[chip(r.share.toFixed(1) + '%', r.share > 50 ? 'bad' : 'ok')]}],
      msg:{zh:'context ' + r.n.toLocaleString() + '：投影與 MLP 這些跟 n 成正比的部分 ' + (r.lin / 1e9).toFixed(0) + ' G 次乘加，attention 的兩個矩陣乘法（算分數、加權求和）' + (r.quad / 1e9).toFixed(0) + ' G 次，二次方的佔 ' + r.share.toFixed(1) + '%。' + (i === FLOPS.length - 1 ? '注意 head 數在這兩欄裡完全不出現——h 切的是同樣的矩陣、餵的是同樣的兩個矩陣乘法。' : ''),
           en:'At a context of ' + r.n.toLocaleString() + ': the parts that scale with n - the projections and the MLP - are ' + (r.lin / 1e9).toFixed(0) + ' G multiply-accumulates, attention\'s two matmuls are ' + (r.quad / 1e9).toFixed(0) + ' G, and the quadratic part is ' + r.share.toFixed(1) + '% of the block.' + (i === FLOPS.length - 1 ? ' The head count appears in neither column: h cuts the same matrices and feeds the same two matmuls.' : '')}});
  });
  p({mode:'flops', upto:FLOPS.length - 1, line:6,
    cap:{zh:'交叉點永遠是 6 倍的 d_model', en:'the crossover is always 6 d_model'},
    panels:BUDGET.map(b => ({lbl:'d_model ' + b.d, chips:[chip(b.cross.toLocaleString(), 'act')]})),
    msg:{zh:'把兩邊設成相等就解得出來：n = (4d² + 2d·d_ff) / (2d)，在 d_ff = 4·d_model 時就是 6·d_model。d_model 768 是 ' + BUDGET[0].cross.toLocaleString() + '、4096 是 ' + BUDGET[2].cross.toLocaleString() + '。短於這個長度，block 的成本由線性的那半決定；長過去才輪到二次方那半。',
         en:'Set the two halves equal and solve: n = (4d^2 + 2 d d_ff) / (2d), which with d_ff = 4 d_model is exactly 6 d_model - ' + BUDGET[0].cross.toLocaleString() + ' at d_model 768 and ' + BUDGET[2].cross.toLocaleString() + ' at 4096. Below that context the block is dominated by the linear half; above it, by the quadratic one.'}});
  return F.list;
}

/* ================================================================ tab 7: LeetCode */
const CODE_LC189 = [
  'k %= n                      # k can exceed n',
  'reverse(nums, 0, n - 1)     # whole array',
  'reverse(nums, 0, k - 1)     # first k',
  'reverse(nums, k, n - 1)     # the rest',
  '',
  '# three reversals, no second array, O(n) time O(1) space'
];
const CODE_LC303 = [
  'self.pre = [0]',
  'for v in nums:',
  '    self.pre.append(self.pre[-1] + v)',
  '',
  'def sumRange(self, left, right):',
  '    return self.pre[right + 1] - self.pre[left]'
];
function lcStage(o){
  const sh = [], cw = 1.05, X0 = 1.10, Y0 = 2.60;
  (o.arr || []).forEach((v, j) => {
    const inPart = o.lo >= 0 && j >= o.lo && j <= o.hi;
    sh.push(box(X0 + j * cw, Y0, cw * .88, .74, inPart ? 'act' : 'ok', String(v), .32));
    sh.push(fitT(X0 + j * cw + cw * .44, Y0 - .22, String(j), {c:COL.grey, fs:.22}));
  });
  if (o.lo >= 0){
    sh.push(fitT(X0 + ((o.lo + o.hi) / 2 + .44) * cw, Y0 + 1.00,
                 {zh:'接下來反轉這一段', en:'reverse this stretch next'}, {c:COL.hot, fs:.28}));
  }
  side(sh, 1.10, 1.55, o.notes || [], .46);
  return caps(sh, o);
}
function preStage(o){
  const sh = [], cw = 1.12, X0 = 1.72;
  LC303.nums.forEach((v, j) => {
    const inQ = o.q && j >= o.q.l && j <= o.q.r;
    sh.push(box(X0 + j * cw, 3.30, cw * .88, .70, inQ ? 'act' : 'ok', String(v), .30));
  });
  LC303.pre.forEach((v, j) => {
    const edge = o.q && (j === o.q.l || j === o.q.r + 1);
    sh.push(box(X0 - cw * .20 + j * cw, 2.20, cw * .88, .70, edge ? 'hot' : 'idle',
                String(v), .30));
  });
  sh.push(fitT(X0 - .42, 3.65, {zh:'nums', en:'nums'}, {anchor:'end', c:COL.grey, fs:.26}));
  sh.push(fitT(X0 - .42, 2.55, {zh:'prefix', en:'prefix'}, {anchor:'end', c:COL.grey, fs:.26}));
  if (o.q){
    sh.push(fitT(4.9, 1.65, 'sumRange(' + o.q.l + ', ' + o.q.r + ') = pre[' + (o.q.r + 1) +
                 '] - pre[' + o.q.l + '] = ' + LC303.pre[o.q.r + 1] + ' - ' +
                 LC303.pre[o.q.l] + ' = ' + o.q.v, {c:COL.tealL, fs:.30}));
  }
  return caps(sh, o);
}
function lcFrames(v){
  const F = new Frames();
  if (v === 1){
    const p = (o) => F.push({shapes:preStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
    p({line:2,
      cap:{zh:'LeetCode 303：一趟前綴和，之後每次查詢一次減法', en:'LeetCode 303: one prefix pass, then one subtraction per query'},
      panels:[{lbl:'nums', chips:LC303.nums.map(x => chip(String(x), 'ok'))}],
      msg:{zh:'prefix[i] 是前 i 個元素的和，所以 prefix 比 nums 多一格、第 0 格是 0。那一格不是裝飾——有了它，left = 0 的查詢才不用寫特例。',
           en:'prefix[i] is the sum of the first i elements, so the prefix array is one longer than nums and starts with a 0. That extra slot is not decoration: it is what lets a query with left = 0 avoid being a special case.'}});
    LC303.queries.forEach(q => {
      p({q:q, line:5,
        cap:{zh:'每次查詢都是 O(1)', en:'every query is O(1)'},
        panels:[{lbl:'query', chips:[chip('[' + q.l + ', ' + q.r + ']', 'hot')]},
                {lbl:'answer', chips:[chip(String(q.v), 'ok')]}],
        msg:{zh:'sumRange(' + q.l + ', ' + q.r + ') = pre[' + (q.r + 1) + '] − pre[' + q.l + '] = ' + LC303.pre[q.r + 1] + ' − ' + LC303.pre[q.l] + ' = ' + q.v + '。建表 O(n) 一次，之後不管問幾次、問多長的區間都是一次減法。',
             en:'sumRange(' + q.l + ', ' + q.r + ') = pre[' + (q.r + 1) + '] - pre[' + q.l + '] = ' + LC303.pre[q.r + 1] + ' - ' + LC303.pre[q.l] + ' = ' + q.v + '. The table costs O(n) once, and after that every query is one subtraction, however many you ask and however long the range.'}});
    });
    p({line:5,
      cap:{zh:'同一招拿去算變異數就會踩到今天的地雷', en:'extend the same trick to variance and it meets today\'s trap'},
      panels:[{lbl:'window variance', chips:[chip(LC303.varTrue.toFixed(4), 'ok')]},
              {lbl:'from float32 prefix sums', chips:[chip(LC303.varPre.toFixed(4), 'bad')]}],
      msg:{zh:'把 x 和 x² 各做一條前綴和，任何區間的變異數就是兩次減法——正是第 4 個分頁那條恆等式。在一段平均值 10,000、真變異數 ' + LC303.varTrue.toFixed(4) + ' 的 float32 資料上，用前綴和算出來是 ' + LC303.varPre.toFixed(4) + '。前綴和本身沒問題（區間和是對的），出事的是「平方和很大、答案很小」這個組合。',
           en:'Keep prefix sums of x and of x squared and the variance of any window is two subtractions - exactly the identity from the normalisation tab. On float32 data with a mean of 10,000 and a true window variance of ' + LC303.varTrue.toFixed(4) + ', the prefix-sum route returns ' + LC303.varPre.toFixed(4) + '. Prefix sums are not the problem and the range sums are right; the problem is a large sum of squares carrying a small answer.'}});
    return F.list;
  }
  const p = (o) => F.push({shapes:lcStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const msgs = [
    {zh:'要把陣列往右轉 ' + LC189.k + ' 格，等於「最後 ' + LC189.k + ' 個搬到最前面，兩段內部順序都不變」。先把整個陣列反轉。',
     en:'Rotating right by ' + LC189.k + ' means the last ' + LC189.k + ' elements move to the front, with both stretches keeping their internal order. Start by reversing the whole array.'},
    {zh:'整個反轉之後，該去前面的那 ' + LC189.k + ' 個已經在前面了，只是順序反了。把前 ' + LC189.k + ' 個再反轉回來。',
     en:'After the full reversal the ' + LC189.k + ' elements that belong at the front are at the front, just backwards. Reverse that first stretch back.'},
    {zh:'剩下那一段同樣反轉回來。',
     en:'Do the same to the rest.'},
    {zh:'完成：' + LC189.steps[3].a.join(', ') + '。三次反轉、不配置第二個陣列、O(n) 時間 O(1) 空間；而且 k 可能大於 n，所以第一行的 k %= n 不能省。',
     en:'Done: ' + LC189.steps[3].a.join(', ') + '. Three reversals, no second array, O(n) time and O(1) space - and k can exceed n, which is why the k %= n on the first line is not optional.'}];
  LC189.steps.forEach((s, i) => {
    p({arr:s.a, lo:s.lo, hi:s.hi, line:i === 3 ? 5 : i + 1,
      cap:{zh:'LeetCode 189：把陣列右轉 ' + LC189.k + ' 格', en:'LeetCode 189: rotate the array right by ' + LC189.k},
      panels:[{lbl:'array', chips:s.a.map(x => chip(String(x), 'ok'))},
              {lbl:'step', chips:[chip(s.what, 'hot')]}],
      msg:msgs[i]});
  });
  p({arr:LC189.cycleBad, lo:-1, hi:-1, line:5,
    cap:{zh:'另一個 O(1) 空間解法，以及它的陷阱', en:'the other O(1)-space method, and its trap'},
    panels:[{lbl:'all gcd(n, k) starts', chips:LC189.cycleGood.map(x => chip(String(x), 'ok'))},
            {lbl:'only index 0', chips:LC189.cycleBad.map(x => chip(String(x), 'bad'))}],
    msg:{zh:'另一種寫法是沿著「每個元素往後跳 k 格」的循環走。它是對的，但必須從 gcd(n, k) 個不同起點各走一次：在 [1..6]、k = 2 時只從 index 0 出發會得到 ' + LC189.cycleBad.join(', ') + '——只搬動了一半，而回傳的仍然是一個長度正確、型別正確的陣列。k = 1 時 gcd(6, 1) = 1，同一段程式碼就完全正確（' + LC189.oneOk.join(', ') + '），所以小測資根本看不出來。跟今天的 head 切錯是同一種形狀的 bug。',
         en:'The other method walks the permutation cycles, each element jumping k places. It is correct, but it has to start from gcd(n, k) different places: on [1..6] with k = 2, starting only at index 0 gives ' + LC189.cycleBad.join(', ') + ' - half the array moved, and what comes back is still an array of the right length and the right type. With k = 1, gcd(6, 1) = 1 and the same code is completely right (' + LC189.oneOk.join(', ') + '), so a small test never shows it. That is the same shape of bug as splitting the heads along the wrong axis.'}});
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'Multi-Head Attention 與 Transformer Block',
          en:'Multi-head attention and the transformer block'},
  sub: {zh:'兩個混合器、一條殘差流，以及幾個形狀永遠正確的 bug',
        en:'two mixers, one residual stream, and a few bugs that always return the right shape'},
  tabs: [
    {id:'block', label:{zh:'一個 block', en:'the block'},
     stage:{zh:'attention 跨位置、MLP 跨通道', en:'attention across positions, the MLP across features'},
     view:VIEW, code:CODE_BLOCK, build:blockFrames,
     variants:[{zh:'pre-LN', en:'pre-LN'}, {zh:'post-LN', en:'post-LN'}],
     idea:{zh:'一個 block 只有兩個混合器：attention 讓每個位置變成其他位置的加權平均，MLP 讓每個位置自己的通道互相混合。兩者都寫回同一條殘差流，形狀從頭到尾不變，所以才能疊。LayerNorm 放在相加之前還是之後，決定「從輸入直通輸出的那條路」有沒有被正規化。',
           en:'A block has two mixers: attention makes every position a weighted average of the others, and the MLP mixes each position\'s own features. Both write into the same residual stream, whose shape never changes, which is what lets blocks stack. Whether the LayerNorm sits before or after the addition decides whether the path straight from input to output passes through a normalisation.'},
     legend:[['#ffbe6b', {zh:'LayerNorm', en:'LayerNorm'}],
             ['#12b3b8', {zh:'分支（attention / MLP）', en:'a branch (attention or MLP)'}],
             ['#9d6bff', {zh:'寫回殘差流', en:'written into the residual stream'}]]},
    {id:'split', label:{zh:'切 head', en:'splitting heads'},
     stage:{zh:'切通道，不是切位置', en:'cut the features, not the positions'},
     view:VIEW, code:CODE_SPLIT, build:splitFrames,
     idea:{zh:'多頭切的是「通道」這個軸：每個 head 拿到每個 token 的一塊特徵。少掉那個 transpose、直接 reshape 成 (h, n, d_head)，會變成切「位置」——形狀一模一樣、不丟例外，但每個 head 只看得到半條序列，層輸出差 ' + SPLIT.rel + '%。',
           en:'Splitting heads cuts the feature axis: every head gets a slice of the features of every token. Drop the transpose and reshape straight to (h, n, d_head) and it cuts positions instead - identical shape, no exception, but each head sees only half the sequence and the layer output moves by ' + SPLIT.rel + '%.'},
     legend:[['#ff9736', {zh:'head 0 拿到的', en:'what head 0 gets'}],
             ['#12b3b8', {zh:'head 1 拿到的', en:'what head 1 gets'}]]},
    {id:'heads', label:{zh:'head 買到什麼', en:'what heads buy'},
     stage:{zh:'同樣的參數，更多種混合', en:'the same parameters, more mixings'},
     view:VIEW, code:CODE_HEADS, build:headsFrames,
     idea:{zh:'h 不改變參數量，它只決定同樣四個矩陣被切成幾個子空間，以及這一層能產生幾個 n x n 的權重矩陣。那很重要，因為一個 head 只有一個注意力矩陣、而它要套用在自己帶的每一個通道上：一半通道要「前一個 token」、另一半要「第一個 token」時，兩個 head 在 ' + PATT.inputs + ' 組輸入上都精確，而單一 head（連通道矩陣都給它）停在 MSE ' + PATT.single.toFixed(4) + '。',
           en:'h does not change the parameter count; it decides how many subspaces the same four matrices are cut into, and how many n by n weight matrices the layer gets to produce. That matters because a head has one attention matrix and has to apply it to every channel it carries: when half the channels want the previous token and half want the first, two heads are exact on all ' + PATT.inputs + ' inputs while a single head - given a channel matrix too - stops at MSE ' + PATT.single.toFixed(4) + '.'},
     legend:[['#12b3b8', {zh:'head 1 的注意力', en:'head 1\'s attention'}],
             ['#ff9736', {zh:'head 2 的注意力', en:'head 2\'s attention'}],
             ['#ff5c5c', {zh:'單一 head 的最佳折衷', en:'the single head\'s compromise'}]]},
    {id:'norm', label:{zh:'正規化的算術', en:'the arithmetic of normalising'},
     stage:{zh:'一趟的變異數會變成負的', en:'the one-pass variance goes negative'},
     view:VIEW, code:CODE_NORM, build:normFrames,
     idea:{zh:'var = E[x²] − E[x]² 少掃一次資料，代價是把兩個大數相減去得到一個小數。殘差流本來就會累積一個平均值：float32 下位移到 10,000 時一趟版本回報 −8（負的變異數，開根號是 nan），而同一份資料兩趟版本仍然是 ' + CANC[3].two.toFixed(6) + '。',
           en:'var = E[x^2] - E[x]^2 saves a pass over the data and pays for it by differencing two large numbers to get a small one. A residual stream accumulates a mean offset, and in float32 an offset of 10,000 makes the one-pass version report -8 - a negative variance, whose square root is nan - while two passes still return ' + CANC[3].two.toFixed(6) + ' from the same array.'},
     legend:[['#ff9736', {zh:'一趟版本的誤差', en:'error of the one-pass version'}],
             ['#ff5c5c', {zh:'負的或零的變異數', en:'a negative or zero variance'}]]},
    {id:'depth', label:{zh:'疊 32 層', en:'stacking 32 blocks'},
     stage:{zh:'殘差流是一個連加', en:'the residual stream is a running sum'},
     view:VIEW, code:CODE_DEPTH, build:depthFrames,
     idea:{zh:'post-LN 的 block 最後一步就是正規化，所以輸出的範數不再隨深度跑（這個初始化下剛好是 ' + DEPTH.post[31] + '）；pre-LN 的主幹沒人管，32 層後長到 ' + DEPTH.pre[31] + '，所以最後一定要補一個 LayerNorm。把每個分支寫進去的量乘上 1/sqrt(2L)，32 層的成長從 ' + GROW.ratio + ' 倍壓到 ' + GROW.ratioScaled + ' 倍。',
           en:'A post-LN block ends with a normalisation, so the norm coming out stops depending on depth - exactly ' + DEPTH.post[31] + ' at this initialisation, however many you stack. Pre-LN leaves the trunk alone and it reaches ' + DEPTH.pre[31] + ' after 32 blocks, which is why the stack needs one final LayerNorm. Scaling what each branch writes by 1/sqrt(2L) takes the growth over 32 blocks from ' + GROW.ratio + 'x down to ' + GROW.ratioScaled + 'x.'},
     legend:[['#ff9736', {zh:'pre-LN 的殘差流', en:'the pre-LN stream'}],
             ['#12b3b8', {zh:'post-LN 的殘差流', en:'the post-LN stream'}],
             ['#9d6bff', {zh:'分支乘過 1/sqrt(2L)', en:'branches scaled by 1/sqrt(2L)'}]]},
    {id:'budget', label:{zh:'參數與計算量', en:'parameters and FLOPs'},
     stage:{zh:'三分之二是 MLP', en:'two thirds of it is the MLP'},
     view:VIEW, code:CODE_BUDGET, build:budgetFrames,
     idea:{zh:'attention 是四個 d_model x d_model、MLP 是兩個 d_model x 4·d_model，所以每個 block 的 ' + BUDGET[2].share + '% 參數在 MLP。計算量則分成跟 n 成正比與跟 n² 成正比兩半，交叉點正好是 6·d_model（d_model 4096 時是 ' + BUDGET[2].cross.toLocaleString() + '）。head 數在這兩件事裡都不出現。',
           en:'Attention is four d_model by d_model matrices and the MLP is two of d_model by 4 d_model, so ' + BUDGET[2].share + '% of every block is the MLP. The compute splits into a part linear in n and a part quadratic in n, and they cross at exactly 6 d_model - ' + BUDGET[2].cross.toLocaleString() + ' at d_model 4096. The head count appears in neither.'},
     legend:[['#12b3b8', {zh:'attention', en:'attention'}],
             ['#ff9736', {zh:'MLP', en:'the MLP'}],
             ['#ff5c5c', {zh:'二次方那半過半了', en:'the quadratic half has taken over'}]]},
    {id:'lc', label:{zh:'LeetCode 189 / 303', en:'LeetCode 189 / 303'},
     stage:{zh:'就地重排，與一次算好的前綴', en:'rearranging in place, and precomputing once'},
     view:VIEW, build:lcFrames,
     variants:[{zh:'189 旋轉陣列', en:'189 rotate array'}, {zh:'303 區間和', en:'303 range sum'}],
     get code(){ return curV() === 0 ? CODE_LC189 : CODE_LC303; },
     idea:{zh:'189 用三次反轉把陣列就地右轉，不配置第二個陣列——而它的另一個解法（沿循環搬）必須從 gcd(n, k) 個起點各走一次，少了這件事會「搬一半、回傳一個長度正確的陣列」，跟今天切錯 head 的 bug 是同一種形狀。303 用一趟前綴和把每次區間查詢變成一次減法；把同一招拿去算變異數，就直接撞上這天的那條恆等式。',
           en:'189 rotates an array in place with three reversals and no second array - while its other solution, walking the permutation cycles, needs gcd(n, k) starting points, and without them it moves half the array and returns something of exactly the right length: the same shape of bug as splitting the heads wrong. 303 turns every range query into one subtraction with a single prefix pass; extend that trick to variance and it runs straight into today\'s identity.'},
     legend:[['#12b3b8', {zh:'陣列 / 區間', en:'the array or the range'}],
             ['#ff9736', {zh:'正在處理的那一段', en:'the stretch being worked on'}],
             ['#9d6bff', {zh:'用到的 prefix 端點', en:'the prefix endpoints used'}]]}
  ]
};
