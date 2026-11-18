// DAY: 61
// TITLE_ZH: 量化：用更少的位元存每個權重，以及那個四捨五入的代價
// TITLE_EN: Quantization: fewer bits per weight, and what the rounding costs
// SUB_ZH: bf16 的 70B 模型是 140 GB，而每個 decode step 都要把它全部讀一遍——所以「每個權重幾個 bit」同時決定了卡裝不裝得下與每個 token 多久。問題從來不是「int8 準不準」，而是：哪些數字共用一個 scale、離群值放哪裡、以及四捨五入是要讓權重誤差最小還是讓這一層的輸出誤差最小。
// SUB_EN: A 70B model in bf16 is 140 GB and every decode step reads all of it, so bits per weight is both whether it fits and how long a token takes. The question is never "is int8 accurate": it is which numbers share a scale, where the outliers go, and whether the rounding minimises the error in the weights or in the layer's output.
// FOLDER: day%2061%20-%20quantization
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
const MEM = [{"name": "bf16", "bits": 16, "gb": 140.0, "ms": 28.0}, {"name": "fp8", "bits": 8, "gb": 70.0, "ms": 14.0}, {"name": "int8", "bits": 8, "gb": 70.0, "ms": 14.0}, {"name": "int4 + group 128", "bits": 4.12, "gb": 36.1, "ms": 7.2}, {"name": "mxfp4", "bits": 4.25, "gb": 37.2, "ms": 7.4}];
const GRID = {"scale": 0.0009291338582677165, "rows": [{"x": 0.031, "q": 33, "deq": 0.030661, "err": 0.000339}, {"x": -0.0072, "q": -8, "deq": -0.007433, "err": 0.000233}, {"x": 0.0009, "q": 1, "deq": 0.000929, "err": 2.9e-05}, {"x": 0.118, "q": 127, "deq": 0.118, "err": 0.0}]};
const FMT = [{"name": "int8, per tensor", "bits": 8.0, "rel": 0.03103, "worst": 0.001158}, {"name": "int8, per row", "bits": 8.0, "rel": 0.01444, "worst": 0.001152}, {"name": "fp8 e4m3", "bits": 8.0, "rel": 0.02658, "worst": 0.010417}, {"name": "fp8 e5m2", "bits": 8.0, "rel": 0.05281, "worst": 0.020768}, {"name": "int4, per row", "bits": 4.0, "rel": 0.26197, "worst": 0.020971}, {"name": "int4, group 128", "bits": 4.12, "rel": 0.17584, "worst": 0.020679}, {"name": "mxfp4, block 32", "bits": 4.25, "rel": 0.12301, "worst": 0.060913}];
const DR = {"int8_steps": 254, "int8_ratio": 127.0, "e4m3_ratio": 229376.0, "e5m2_ratio": 3758096384.0, "fp4_levels": 15};
const ERR = [{"name": "weights only, per row", "rel": 0.01462}, {"name": "activations only, per tensor", "rel": 0.06897}, {"name": "activations only, per token", "rel": 0.03394}, {"name": "both", "rel": 0.03703}];
const MP = {"kept": 6, "share": 0.01172, "rel": 0.00686, "all_int8": 0.03703};
const SQ = [{"alpha": 0.0, "range": 43.9, "err": 0.05468, "plain": 0.03703}, {"alpha": 0.25, "range": 17.1, "err": 0.02243, "plain": 0.03703}, {"alpha": 0.5, "range": 6.6, "err": 0.01348, "plain": 0.03703}, {"alpha": 0.75, "range": 2.6, "err": 0.01925, "plain": 0.03703}, {"alpha": 1.0, "range": 1.0, "err": 0.04043, "plain": 0.03703}];
const RT = {"rtn_weight_err": 0.17584, "gptq_weight_err": 0.31186, "rtn_out": 0.17612, "gptq_out": 0.13051, "awq_out": 0.10059, "awq_alpha": 0.36842, "bits": 4, "group": 128};
const QAT = {"bits": 3, "ptq": 2.18, "qat": 0.837, "last": 1.226, "osc": 1.17, "tail": 4.01, "ratio": 2.6, "oPtq": 1.632, "oQat": 1.474, "oRatio": 1.11, "hist": [1.209, 1.818, 2.406, 1.485, 3.021, 4.222, 3.543, 3.365, 4.187, 3.23, 6.342, 2.239, 2.484, 5.061, 3.244]};
const OUT = {"nOut": 6, "d": 512, "ratio": 102};
const SPEC = [{"name": "fp32", "bits": 32, "e": 8, "m": 23, "bias": 127, "max": 3.4028234663852886e+38, "minNormal": 1.1754943508222875e-38, "eps": 1.1920928955078125e-07, "digits": 6.9, "range": 2.4283359705420498e+83, "use": "the default everywhere", "kind": "float"}, {"name": "tf32", "bits": 19, "e": 8, "m": 10, "bias": 127, "max": 3.4011621342146535e+38, "minNormal": 1.1754943508222875e-38, "eps": 0.0009765625, "digits": 3.0, "range": 2.962830083359828e+79, "use": "NVIDIA tensor cores: fp32 range, fp16 precision", "kind": "float"}, {"name": "bf16", "bits": 16, "e": 8, "m": 7, "bias": 127, "max": 3.3895313892515355e+38, "minNormal": 1.1754943508222875e-38, "eps": 0.0078125, "digits": 2.1, "range": 3.6908728444394537e+78, "use": "training: fp32 range, half the bytes", "kind": "float"}, {"name": "fp16", "bits": 16, "e": 5, "m": 10, "bias": 15, "max": 65504.0, "minNormal": 6.103515625e-05, "eps": 0.0009765625, "digits": 3.0, "range": 1098974756864.0, "use": "older training, needs loss scaling", "kind": "float"}, {"name": "fp8 e4m3fn", "bits": 8, "e": 4, "m": 3, "bias": 7, "max": 448.0, "minNormal": 0.015625, "eps": 0.125, "digits": 0.9, "range": 229376.0, "use": "FP8 forward pass: weights and activations", "kind": "float"}, {"name": "fp8 e5m2", "bits": 8, "e": 5, "m": 2, "bias": 15, "max": 57344.0, "minNormal": 6.103515625e-05, "eps": 0.25, "digits": 0.6, "range": 3758096384.0, "use": "FP8 gradients: more range, less precision", "kind": "float"}, {"name": "fp6 e3m2", "bits": 6, "e": 3, "m": 2, "bias": 3, "max": 28.0, "minNormal": 0.25, "eps": 0.25, "digits": 0.6, "range": 448.0, "use": "MX block element, more range", "kind": "float"}, {"name": "fp6 e2m3", "bits": 6, "e": 2, "m": 3, "bias": 1, "max": 7.5, "minNormal": 1.0, "eps": 0.125, "digits": 0.9, "range": 60.0, "use": "MX block element, more precision", "kind": "float"}, {"name": "fp4 e2m1", "bits": 4, "e": 2, "m": 1, "bias": 1, "max": 6.0, "minNormal": 1.0, "eps": 0.5, "digits": 0.3, "range": 12.0, "use": "MXFP4 / NVFP4 element", "kind": "float"}, {"name": "int8", "bits": 8, "e": 0, "m": 7, "bias": null, "max": 127.0, "minNormal": 1.0, "eps": 0.007874015748031496, "digits": 2.1, "range": 127.0, "use": "a uniform grid: the scale carries the magnitude", "kind": "int"}, {"name": "int4", "bits": 4, "e": 0, "m": 3, "bias": null, "max": 7.0, "minNormal": 1.0, "eps": 0.14285714285714285, "digits": 0.8, "range": 7.0, "use": "a uniform grid: the scale carries the magnitude", "kind": "int"}];
const BITS = {"x": 0.15625, "sign": "0", "exp": "01111100", "man": "01000000000000000000000", "stored": 124, "bias": 127, "unbiased": -3, "mantissa": 1.25};
const MX = [{"name": "MXFP8 e4m3", "el": 8, "block": 32, "bpv": 8.25, "maxEl": 448.0}, {"name": "MXFP6 e3m2", "el": 6, "block": 32, "bpv": 6.25, "maxEl": 28.0}, {"name": "MXFP4 e2m1", "el": 4, "block": 32, "bpv": 4.25, "maxEl": 6.0}];
const METH = [{"name": "round to nearest (RTN)", "what": "W", "bits": "8 / 4", "cost": "free", "idea": "absmax scale per tensor, row or group; the baseline everything is compared to"}, {"name": "MSE / percentile clipping", "what": "W, A", "bits": "8 / 4", "cost": "calibration", "idea": "search the clipping threshold that minimises squared error instead of using max"}, {"name": "LLM.int8()", "what": "W + A", "bits": "8 (+16)", "cost": "free", "idea": "split the matmul: outlier columns stay in 16-bit, the rest go to int8"}, {"name": "ZeroQuant / per-token dynamic", "what": "W + A", "bits": "8", "cost": "free", "idea": "per-token activation scales computed at run time, per-column weight scales"}, {"name": "SmoothQuant", "what": "W + A", "bits": "8", "cost": "calibration", "idea": "migrate range from activations into weights: X W = (X/s)(W s)"}, {"name": "AWQ", "what": "W", "bits": "4", "cost": "calibration", "idea": "scale up the channels the activations actually use before rounding"}, {"name": "GPTQ (from OBQ / OBS)", "what": "W", "bits": "4 / 3", "cost": "calibration", "idea": "round column by column and push the error into the rest using H = X^T X"}, {"name": "SqueezeLLM", "what": "W", "bits": "4 / 3", "cost": "calibration", "idea": "sensitivity-weighted k-means codebook plus a sparse outlier set"}, {"name": "SpQR", "what": "W", "bits": "~4", "cost": "calibration", "idea": "keep the few most sensitive weights in high precision, quantize the rest hard"}, {"name": "HQQ", "what": "W", "bits": "4 / 3 / 2", "cost": "free", "idea": "optimise the scale and zero-point with a robust loss; no calibration data"}, {"name": "QuIP#", "what": "W", "bits": "2", "cost": "calibration", "idea": "rotate the weights to make them incoherent, then lattice-quantize"}, {"name": "AQLM", "what": "W", "bits": "2", "cost": "training", "idea": "additive codebooks over groups, fine-tuned end to end"}, {"name": "NF4 / QLoRA", "what": "W", "bits": "4", "cost": "free", "idea": "levels at the quantiles of a normal distribution, plus a second quantization of the scales"}, {"name": "GGUF k-quants", "what": "W", "bits": "2-8", "cost": "free", "idea": "blocks of 16-32 with a super-block scale; the llama.cpp family (Q4_K_M, ...)"}, {"name": "FP8 (e4m3 / e5m2)", "what": "W + A + grad", "bits": "8", "cost": "free", "idea": "a float grid plus a per-tensor or per-row scale; the hardware path"}, {"name": "MX / NVFP4", "what": "W + A", "bits": "4-8", "cost": "free", "idea": "block floating point: one shared exponent per 16-32 values"}, {"name": "KV cache quantization", "what": "KV", "bits": "8 / 4", "cost": "free", "idea": "the cache is the memory, so quantize K and V per head or per token"}, {"name": "QAT (STE, LSQ, PACT)", "what": "W + A", "bits": "4 / 3 / 2", "cost": "training", "idea": "train through the rounding; LSQ learns the step size, PACT the clipping"}];
const NOTE = {"name": "W4A16", "bpw": 4.125, "group": 128};

function barRows(sh, rows, y0, dy, x0, wmax, val, lab, right, sty){
  rows.forEach((r, i) => {
    const y = y0 - i * dy;
    sh.push(fitT(x0 - .12, y + .20, lab(r), {anchor:'end', c:COL.pale, fs:.26}));
    sh.push(S.r(x0, y, Math.max(.06, wmax * val(r)), .40, sty ? sty(r) : 'ok', null));
    sh.push(fitT(9.30, y + .20, right(r), {anchor:'end', c:COL.grey, fs:.25}));
  });
  return sh;
}

/* ================================================================ tab 1: why */
const CODE_WHY = [
  'bytes = params * bits / 8',
  'read_time = bytes / bandwidth      # every decode step reads ALL the weights',
  '',
  '70e9 * 16 / 8 = 140 GB   @ 5 TB/s -> 28 ms per step',
  '70e9 *  4 / 8 =  35 GB   @ 5 TB/s ->  7 ms per step'
];
function whyStage(o){
  const sh = [];
  const top = MEM[0].gb;
  MEM.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.20 - i * .80;
    sh.push(fitT(2.30, y + .20, r.name, {anchor:'end', c:COL.pale, fs:.27}));
    sh.push(S.r(2.50, y, Math.max(.06, 5.2 * r.gb / top), .44,
                i === o.upto ? 'hot' : 'ok', null));
    sh.push(fitT(9.30, y + .20, r.gb.toFixed(1) + ' GB　' + r.ms.toFixed(1) + ' ms',
                 {anchor:'end', c:COL.grey, fs:.25}));
  });
  sh.push(fitT(.70, 4.95, {zh:'70B 模型的權重，以及在 5 TB/s 下讀一遍要多久',
                           en:'the weights of a 70B model, and one read at 5 TB/s'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function whyFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:whyStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:3,
    cap:{zh:'bf16：140 GB，每個 token 都要讀一遍', en:'bf16: 140 GB, read once per token'},
    panels:[{lbl:'bf16', chips:[chip('140 GB', 'dim'), chip('28 ms', 'bad')]}],
    msg:{zh:'前面幾天已經量過：decode 慢是慢在「每一步把整個模型的權重讀一遍」。所以每個權重幾個 bit，同時決定了卡裝不裝得下、以及每個 token 要多久。',
         en:'The earlier days measured it: decode is slow because every step reads the whole model from memory. So bits per weight decides both whether the model fits on the card and how long a token takes.'}});
  p({upto:MEM.length - 1, line:4,
    cap:{zh:'4 bit：兩個數字一起砍四倍', en:'4 bits: both numbers cut by four'},
    panels:[{lbl:'int4 + group', chips:[chip(MEM[3].gb.toFixed(1) + ' GB', 'ok'),
                                        chip(MEM[3].ms.toFixed(1) + ' ms', 'ok')]},
            {lbl:'mxfp4', chips:[chip(MEM[4].bits.toFixed(2) + ' bits', 'ok')]}],
    msg:{zh:'4 bit 的格式實際上不是 4.00 bit：每個 group 還要存一個 scale，group 128 的 int4 是 ' + MEM[3].bits.toFixed(2) + ' bit，MXFP4（每 32 個共用一個 8-bit 指數）是 ' + MEM[4].bits.toFixed(2) + ' bit。這一系列裡沒有別的東西能同時把容量和延遲砍成四分之一——剩下的就是問：四捨五入要付多少。',
         en:'A 4-bit format is never exactly 4.00 bits: every group stores a scale too, so int4 with a group of 128 is ' + MEM[3].bits.toFixed(2) + ' bits and MXFP4 (one 8-bit exponent per 32 values) is ' + MEM[4].bits.toFixed(2) + '. Nothing else in this series cuts capacity and latency by four at once. The rest of the day is what the rounding costs.'}});
  return F.list;
}

/* ================================================================ tab 2: the grid */
const CODE_GRID = [
  'scale = abs(x).max() / qmax        # qmax = 127 for int8',
  'q     = round(x / scale)           # what gets stored',
  'x_hat = q * scale                  # what the matmul sees',
  '',
  '# every error is at most half a step,',
  '# and the step is set by the LARGEST value in the group'
];
function gridStage(o){
  const sh = [];
  const hi = Math.max.apply(null, GRID.rows.map(r => Math.abs(r.x)));
  GRID.rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.20 - i * .85;
    sh.push(fitT(.70, y + .20, r.x.toFixed(4), {anchor:'start', c:COL.pale, fs:.28}));
    sh.push(S.e(2.10, y + .20, 2.70, y + .20, {s:'ghost', arrow:true, w:.05}));
    sh.push(box(2.90, y, 1.40, .44, i === o.upto ? 'hot' : 'ok', 'q = ' + r.q, .26));
    sh.push(S.e(4.45, y + .20, 5.05, y + .20, {s:'ghost', arrow:true, w:.05}));
    sh.push(fitT(5.25, y + .20, r.deq.toFixed(6), {anchor:'start', c:COL.pale, fs:.27}));
    sh.push(fitT(9.30, y + .20, {zh:'誤差 ' + r.err.toFixed(6), en:'error ' + r.err.toFixed(6)},
                 {anchor:'end', c:r.err > GRID.scale / 2 ? COL.red : COL.grey, fs:.25}));
  });
  sh.push(fitT(.70, 4.95, 'scale = max|x| / 127 = ' + GRID.scale.toFixed(6),
               {anchor:'start', c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
function gridFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:gridStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:2,
    cap:{zh:'一個量化過的數字就是「整數 × scale」', en:'a quantized number is an integer times a scale'},
    panels:[{lbl:'scale', chips:[chip(GRID.scale.toFixed(6), 'dim')]}],
    msg:{zh:'挑一個 scale、除下去四捨五入成整數、用的時候再乘回來。存進記憶體的只有那個整數（int8 就是 1 個 byte），scale 每個 group 存一個。',
         en:'Pick a scale, divide and round to an integer, multiply back when you use it. Only the integer is stored - one byte for int8 - and the scale is stored once per group.'}});
  p({upto:GRID.rows.length - 1, line:5,
    cap:{zh:'而步長是被「group 裡最大的那個」決定的', en:'and the step is set by the largest value in the group'},
    panels:[{lbl:'largest', chips:[chip(GRID.rows[3].x.toFixed(4), 'hot')]},
            {lbl:'smallest', chips:[chip(GRID.rows[2].x.toFixed(4), 'dim')]}],
    msg:{zh:'看最後兩列：0.118 是最大值，它剛好落在格點上、誤差 0；0.0009 只有一格，誤差相對它自己來說很大。這就是整個主題的全部——**哪些數字被迫共用一個 scale**。一個離群值會把 scale 撐大，然後其他所有數字都只剩幾個格點可用。',
         en:'Look at the last two rows: 0.118 is the largest value, lands exactly on a grid point, error zero; 0.0009 gets a single level and its relative error is enormous. That is the whole subject - **which numbers are forced to share a scale**. One outlier stretches the scale and everything else is left with a handful of levels.'}});
  return F.list;
}

/* ================================================================ tab 3: who shares a scale */
const CODE_SHARE = [
  'quant_int(W, 8)              # one scale for the whole tensor',
  'quant_int(W, 8, axis=-1)     # one scale per row  (per output channel)',
  'quant_int(W, 4, group=128)   # one scale per 128 weights in a row',
  'mxfp4(W, block=32)           # one shared EXPONENT per 32 weights',
  '',
  '# cost of a scale: 16 bits per group.  group=128 -> +0.125 bits/weight'
];
function shareStage(o){
  const sh = [];
  const top = Math.max.apply(null, FMT.map(r => r.rel));
  FMT.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.40 - i * .62;
    sh.push(fitT(2.90, y + .18, r.name, {anchor:'end', c:COL.pale, fs:.25}));
    sh.push(S.r(3.10, y, Math.max(.05, 4.4 * r.rel / top), .36,
                r.bits <= 4.3 ? 'act' : (i === o.upto ? 'hot' : 'ok'), null));
    sh.push(fitT(9.30, y + .18, r.bits.toFixed(2) + ' bits　' + r.rel.toFixed(5),
                 {anchor:'end', c:COL.grey, fs:.24}));
  });
  sh.push(fitT(.70, 5.05, {zh:'同一組權重，不同的 scale 分組（越短越好）',
                           en:'the same weights, different groupings - shorter is better'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function shareFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:shareStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:2,
    cap:{zh:'一個 scale 換成一列一個', en:'one scale for everything, or one per row'},
    panels:[{lbl:'per tensor', chips:[chip(FMT[0].rel.toFixed(5), 'bad')]},
            {lbl:'per row', chips:[chip(FMT[1].rel.toFixed(5), 'ok')]}],
    msg:{zh:'一列一個 scale，成本是每 512 個權重多存一個 float（＋0.03 bit），誤差就少了一半以上。原因就是上一個分頁：強迫整個矩陣共用最大值，等於讓最大的那一列決定所有人的步長。',
         en:'A scale per row costs one extra float per 512 weights - 0.03 bits - and cuts the error by more than half. The reason is the previous tab: one scale for the whole matrix lets the largest row decide everyone else\'s step size.'}});
  p({upto:FMT.length - 1, line:5,
    cap:{zh:'位元換 scale：4 bit 但 group 小，贏過 8 bit 但一個 scale',
         en:'bits against scales: int4 with small groups beats int8 with one scale'},
    panels:[{lbl:'int4 per row', chips:[chip(FMT[4].rel.toFixed(5), 'bad')]},
            {lbl:'int4 group 128', chips:[chip(FMT[5].rel.toFixed(5), 'act')]},
            {lbl:'mxfp4 block 32', chips:[chip(FMT[6].rel.toFixed(5), 'ok')]}],
    msg:{zh:'4 bit 一列一個 scale 的誤差是 ' + FMT[4].rel.toFixed(3) + '，改成每 128 個一個 scale 就降到 ' + FMT[5].rel.toFixed(3) + '，而 MXFP4（每 32 個共用一個 2 的冪次指數）是 ' + FMT[6].rel.toFixed(3) + '——**比 int8 per tensor 的 ' + FMT[0].rel.toFixed(3) + ' 還好，而且只用一半的位元**。這就是實務上的取捨：每個值用更少的位元，但 scale 給得更細。',
         en:'At 4 bits a per-row scale gives ' + FMT[4].rel.toFixed(3) + '; one scale per 128 weights gives ' + FMT[5].rel.toFixed(3) + '; and MXFP4, which shares a power-of-two exponent across 32 values, gives ' + FMT[6].rel.toFixed(3) + ' - better than int8 with one scale for the whole tensor (' + FMT[0].rel.toFixed(3) + ') at half the bits. That is the trade everyone actually makes: fewer bits per value, more scales.'}});
  return F.list;
}

/* ================================================================ tab 4: integers against floats */
const CODE_FLOAT = [
  '# int8: 254 evenly spaced levels',
  'q = round(x / scale)',
  '',
  '# fp8 e4m3: 3 mantissa bits, exponent from -6 to 8, max 448',
  'e = clip(floor(log2(|x|)), emin, emax)',
  'q = round(x / 2**(e - m_bits)) * 2**(e - m_bits)',
  '',
  '# fp4 e2m1 has 8 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6'
];
function floatStage(o){
  const sh = [];
  const lines = [[{zh:'int8：254 個等距格點', en:'int8: 254 evenly spaced levels'},
                  DR.int8_ratio, 'ok'],
                 [{zh:'fp8 e4m3：3 bit 尾數', en:'fp8 e4m3: 3 mantissa bits'},
                  DR.e4m3_ratio, 'act'],
                 [{zh:'fp8 e5m2：2 bit 尾數、更大指數', en:'fp8 e5m2: 2 mantissa bits, wider exponent'},
                  DR.e5m2_ratio, 'hot']];
  lines.forEach((l, i) => {
    if (i > o.upto) return;
    const y = 4.30 - i * 1.05;
    sh.push(fitT(.70, y + .62, l[0], {anchor:'start', c:COL.tealL, fs:.27}));
    const w = 1.2 + 7.2 * Math.log10(l[1]) / Math.log10(DR.e5m2_ratio);
    sh.push(S.r(.70, y, w, .40, l[2], null));
    sh.push(fitT(9.30, y + .20, {zh:'動態範圍 ' + l[1].toExponential(1),
                                 en:'dynamic range ' + l[1].toExponential(1)},
                 {anchor:'end', c:COL.grey, fs:.25}));
  });
  if (o.upto >= 2){
    sh.push(fitT(4.9, .95, {zh:'同樣 8 bit：整數換到精度、浮點換到範圍',
                            en:'the same 8 bits: integers buy precision, floats buy range'},
                 {c:COL.pale, fs:.28}));
  }
  return caps(sh, o);
}
function floatFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:floatStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'整數：格點等距', en:'integers: the levels are evenly spaced'},
    panels:[{lbl:'levels', chips:[chip(String(DR.int8_steps), 'dim')]},
            {lbl:'range', chips:[chip(DR.int8_ratio.toFixed(0) + ':1', 'bad')]}],
    msg:{zh:'int8 的格點是等距的，所以**絕對**誤差到處一樣大。對接近最大值的數很好，對小數字很糟——一個只有最大值千分之一的數，根本分不到幾個格點。',
         en:'int8 spaces its levels evenly, so the *absolute* error is the same everywhere. That is fine for values near the maximum and bad for small ones: a number a thousandth of the maximum gets almost no levels at all.'}});
  p({upto:2, line:5,
    cap:{zh:'浮點：格點隨大小變密變疏', en:'floats: the spacing grows with the magnitude'},
    panels:[{lbl:'e4m3', chips:[chip('forward', 'ok')]},
            {lbl:'e5m2', chips:[chip('gradients', 'act')]}],
    msg:{zh:'浮點把位元花在指數上，格點變成幾何排列，所以**相對**誤差到處差不多。e4m3（3 bit 尾數）是前向傳遞用的格式；e5m2（2 bit 尾數、多一位指數）是給梯度用的——梯度的數量級跨度大得多，但對精度的要求低。FP8 訓練同時用兩種，就是這個原因。',
         en:'A float spends bits on an exponent, so the levels are geometrically spaced and the *relative* error is roughly constant. e4m3 (3 mantissa bits) is the forward-pass format; e5m2 (2 mantissa bits, one more exponent bit) exists for gradients, which span a far wider range and care less about precision. That is why FP8 training uses both.'}});
  return F.list;
}

/* ================================================================ tab 5: outliers */
const CODE_OUT = [
  '# quantize one side at a time and see which one hurts',
  'Y  = X @ W.T',
  'Yw = X @ quant_int(W, 8, axis=-1).T          # weights only',
  'Ya = quant_int(X, 8) @ W.T                   # activations only',
  '',
  '# SmoothQuant: X @ W.T == (X / s) @ (W * s).T for any per-channel s',
  's = max|X_j|**alpha / max|W_j|**(1 - alpha)'
];
function outStage(o){
  const sh = [];
  if (o.upto === 0){
    const top = Math.max.apply(null, ERR.map(r => r.rel));
    ERR.forEach((r, i) => {
      const y = 4.20 - i * .85;
      sh.push(fitT(4.00, y + .20, r.name, {anchor:'end', c:COL.pale, fs:.25}));
      sh.push(S.r(4.20, y, Math.max(.06, 3.9 * r.rel / top), .40,
                  r.rel > 2 * ERR[0].rel ? 'bad' : 'ok', null));
      sh.push(fitT(9.30, y + .20, r.rel.toFixed(5), {anchor:'end', c:COL.grey, fs:.25}));
    });
    sh.push(fitT(.70, 4.95, {zh:OUT.nOut + ' / ' + OUT.d + ' 個輸入 channel 的值是中位數的 ' + OUT.ratio + ' 倍',
                             en:OUT.nOut + ' of ' + OUT.d + ' input channels carry ' + OUT.ratio + 'x the median value'},
                 {anchor:'start', c:COL.grey, fs:.26}));
  } else {
    const top = Math.max.apply(null, SQ.map(r => r.err));
    SQ.forEach((r, i) => {
      const y = 4.20 - i * .72;
      sh.push(fitT(2.60, y + .18, 'alpha ' + r.alpha.toFixed(2), {anchor:'end', c:COL.pale, fs:.26}));
      sh.push(S.r(2.80, y, Math.max(.06, 4.2 * r.err / top), .38,
                  r.err === Math.min.apply(null, SQ.map(x => x.err)) ? 'ok' : 'act', null));
      sh.push(fitT(9.30, y + .18, {zh:'活化動態範圍 ' + r.range.toFixed(1) + '　誤差 ' + r.err.toFixed(5),
                                   en:'act range ' + r.range.toFixed(1) + ',  error ' + r.err.toFixed(5)},
                   {anchor:'end', c:COL.grey, fs:.24}));
    });
    sh.push(fitT(.70, 4.95, {zh:'SmoothQuant：把動態範圍從活化搬到權重',
                             en:'SmoothQuant: move the range out of the activations and into the weights'},
                 {anchor:'start', c:COL.grey, fs:.26}));
    if (o.upto >= 2)
      sh.push(fitT(4.9, .85, {zh:'留 ' + MP.kept + ' 個 column 在 16-bit（佔 ' + (MP.share * 100).toFixed(1) + '%）：' + MP.all_int8.toFixed(5) + ' → ' + MP.rel.toFixed(5),
                              en:'keeping ' + MP.kept + ' columns in 16-bit (' + (MP.share * 100).toFixed(1) + '% of them): ' + MP.all_int8.toFixed(5) + ' -> ' + MP.rel.toFixed(5)},
                   {c:COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function outFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:outStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:4,
    cap:{zh:'難的是活化，不是權重', en:'the hard part is the activations, not the weights'},
    panels:[{lbl:'weights only', chips:[chip(ERR[0].rel.toFixed(5), 'ok')]},
            {lbl:'activations only', chips:[chip(ERR[1].rel.toFixed(5), 'bad')]}],
    msg:{zh:'權重接近高斯分佈，一列一個 scale 就夠。活化不是：' + OUT.nOut + ' 個 channel 的值是中位數的 ' + OUT.ratio + ' 倍，而它們會把整個 tensor 的 scale 撐大，其他數字就只剩幾個格點。**這就是為什麼 weight-only 量化是預設、而量化活化需要額外手段**。',
         en:'The weights are nearly Gaussian and a per-row scale fits them. The activations are not: ' + OUT.nOut + ' channels carry ' + OUT.ratio + 'x the median value, and they stretch the scale for the whole tensor so everything else is left with a few levels. That is why weight-only quantization is the default and quantizing activations needs a trick.'}});
  p({upto:1, line:6,
    cap:{zh:'搬走它：SmoothQuant', en:'move it: SmoothQuant'},
    panels:[{lbl:'alpha 0', chips:[chip(SQ[0].err.toFixed(5), 'bad')]},
            {lbl:'alpha 0.5', chips:[chip(SQ[2].err.toFixed(5), 'ok')]},
            {lbl:'alpha 1', chips:[chip(SQ[4].err.toFixed(5), 'bad')]}],
    msg:{zh:'X W^T ＝ (X/s)(W s)^T 對任何 per-channel 的 s 都成立，所以可以把某個 channel 的動態範圍從活化那邊搬一部分到權重那邊。alpha=0 不搬、alpha=1 全搬（換成權重壞掉），中間才有用——這裡 alpha 0.5 把輸出誤差從 ' + SQ[0].plain.toFixed(5) + ' 降到 ' + SQ[2].err.toFixed(5) + '。而且 s 可以折進前一層的 layer norm，所以執行時完全免費。',
         en:'X W^T = (X/s)(W s)^T for any per-channel s, so a channel\'s dynamic range can be moved from the activations into the weights. alpha=0 moves nothing and alpha=1 moves everything (breaking the weights instead); the useful values are in between - here alpha 0.5 takes the output error from ' + SQ[0].plain.toFixed(5) + ' to ' + SQ[2].err.toFixed(5) + '. And s folds into the previous layer norm, so it costs nothing at run time.'}});
  p({upto:2, line:4,
    cap:{zh:'或者乾脆留著它：LLM.int8()', en:'or keep them: LLM.int8()'},
    panels:[{lbl:'16-bit columns', chips:[chip(String(MP.kept), 'dim')]},
            {lbl:'error', chips:[chip(MP.all_int8.toFixed(5), 'bad'), chip(MP.rel.toFixed(5), 'ok')]}],
    msg:{zh:'第三條路是混合精度：把 ' + (MP.share * 100).toFixed(1) + '% 的 column 留在 16-bit、其餘走 int8，誤差從 ' + MP.all_int8.toFixed(5) + ' 降到 ' + MP.rel.toFixed(5) + '。代價是要拆成兩個矩陣乘法、而且形狀不規則，所以實際速度沒有帳面上好看。第四條路其實前面講過了：**給它們自己的 scale**——per-token、group-wise、MXFP4，都是同一個念頭。',
         en:'The third option is mixed precision: leave ' + (MP.share * 100).toFixed(1) + '% of the columns in 16-bit and send the rest to int8, which takes the error from ' + MP.all_int8.toFixed(5) + ' to ' + MP.rel.toFixed(5) + '. The cost is two matmuls with an irregular shape, so it is slower than it looks. The fourth option already appeared two tabs ago: give them their own scale - per-token, group-wise, MXFP4 are all the same idea.'}});
  return F.list;
}

/* ================================================================ tab 6: better rounding */
const CODE_ROUND = [
  '# GPTQ: round one column, push the error into the columns not yet done',
  'H = X.T @ X / n            # which directions the real inputs excite',
  'Hinv = inv(H + damp)',
  'for j in range(d):',
  '    q = round(W[:, j] / scale) * scale',
  '    err = (W[:, j] - q) / Hinv[j, j]',
  '    W[:, j+1:] -= outer(err, Hinv[j, j+1:])      # compensate',
  '',
  '# AWQ: do not change the rounding, rescale the columns that matter',
  's = mean|X_j| ** alpha ; quant(W * s) / s'
];
function roundStage(o){
  const sh = [];
  const rows = [[{zh:'round to nearest', en:'round to nearest'}, RT.rtn_out, RT.rtn_weight_err, 'bad'],
                [{zh:'GPTQ', en:'GPTQ'}, RT.gptq_out, RT.gptq_weight_err, 'ok'],
                [{zh:'AWQ (alpha ' + RT.awq_alpha.toFixed(2) + ')', en:'AWQ (alpha ' + RT.awq_alpha.toFixed(2) + ')'}, RT.awq_out, null, 'act']];
  const top = rows[0][1];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.00 - i * 1.15;
    sh.push(fitT(.70, y + .62, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
    sh.push(S.r(.70, y, Math.max(.06, 5.0 * r[1] / top), .42, r[3], null));
    sh.push(fitT(9.30, y + .21, {zh:'輸出誤差 ' + r[1].toFixed(5) + (r[2] ? '　權重誤差 ' + r[2].toFixed(5) : ''),
                                 en:'output ' + r[1].toFixed(5) + (r[2] ? ',  weights ' + r[2].toFixed(5) : '')},
                 {anchor:'end', c:COL.grey, fs:.25}));
  });
  sh.push(fitT(.70, 4.95, {zh:'4 bit、每 128 個權重一個 scale', en:'4-bit weights, one scale per 128'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function roundFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:roundStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:5,
    cap:{zh:'最直覺的做法：每個權重各自取最近的格點', en:'the obvious rule: round every weight to its nearest level'},
    panels:[{lbl:'RTN output', chips:[chip(RT.rtn_out.toFixed(5), 'bad')]}],
    msg:{zh:'round-to-nearest 把每個權重當成獨立的數字處理。問題是沒有人在乎權重本身——在乎的是這一層的輸出。',
         en:'Round-to-nearest treats every weight as an independent number. The trouble is that nobody cares about the weights; what matters is the output of the layer.'}});
  p({upto:1, line:6,
    cap:{zh:'GPTQ：權重更差，輸出更好', en:'GPTQ: worse weights, better output'},
    panels:[{lbl:'weight error', chips:[chip(RT.rtn_weight_err.toFixed(3), 'ok'),
                                        chip(RT.gptq_weight_err.toFixed(3), 'bad')]},
            {lbl:'output error', chips:[chip(RT.rtn_out.toFixed(3), 'bad'),
                                        chip(RT.gptq_out.toFixed(3), 'ok')]}],
    msg:{zh:'看兩欄的方向相反：GPTQ 的**權重誤差比 RTN 還大**（' + RT.gptq_weight_err.toFixed(3) + ' vs ' + RT.rtn_weight_err.toFixed(3) + '），輸出誤差卻更小（' + RT.gptq_out.toFixed(3) + ' vs ' + RT.rtn_out.toFixed(3) + '）。做法是一次處理一個 column，然後把剛才犯的誤差**推到還沒處理的 column 上**，推的方向由 H = XᵀX 決定——也就是真實輸入實際會激發哪些方向的二階資訊。',
         en:'The two columns move in opposite directions: GPTQ ends up with *worse* weights than round-to-nearest (' + RT.gptq_weight_err.toFixed(3) + ' against ' + RT.rtn_weight_err.toFixed(3) + ') and a better layer output (' + RT.gptq_out.toFixed(3) + ' against ' + RT.rtn_out.toFixed(3) + '). It rounds one column at a time and pushes the error it just made into the columns it has not done yet, in the direction H = XᵀX says the real inputs actually excite.'}});
  p({upto:2, line:9,
    cap:{zh:'AWQ：不改四捨五入，改縮放', en:'AWQ: do not change the rounding, change the scale'},
    panels:[{lbl:'AWQ', chips:[chip(RT.awq_out.toFixed(5), 'ok')]},
            {lbl:'alpha', chips:[chip(RT.awq_alpha.toFixed(2), 'dim')]}],
    msg:{zh:'AWQ 從另一邊解同一個問題：四捨五入的誤差正比於那一組的 scale，所以把「活化真的會用到」的 column 先乘大再量化（活化那邊除回來），同樣的四捨五入對它們的傷害就小。指數 alpha 是搜出來的，不是推導出來的——這裡搜到 ' + RT.awq_alpha.toFixed(2) + '。兩者都需要校準資料：幾條真實序列，離線跑一次。',
         en:'AWQ attacks the same problem from the other side: rounding error is proportional to the scale of a group, so the columns the activations actually use are multiplied up before rounding (and divided back on the activation side), which makes the same rounding hurt them less. The exponent is searched rather than derived - here it lands at ' + RT.awq_alpha.toFixed(2) + '. Both methods need calibration data: a handful of real sequences, run once, offline.'}});
  return F.list;
}

/* ================================================================ tab 7: QAT */
const CODE_QAT = [
  '# forward: rounded.  backward: pretend round() was the identity.',
  'wq = fake_quant(w, bits)',
  'g  = X.T @ (X @ wq - y) * 2 / n       # straight-through estimator',
  'w -= lr * g                            # the float weights keep moving',
  '',
  '# PyTorch: a custom autograd Function, or torchao.quantization.qat',
  '# JAX:     x + stop_gradient(round(x) - x)'
];
function qatStage(o){
  const sh = [];
  const pairs = [[{zh:'相關的輸入（像真的一層）', en:'correlated inputs, like a real layer'}, QAT.ptq, QAT.qat, QAT.ratio],
                 [{zh:'幾乎正交的輸入', en:'nearly orthogonal inputs'}, QAT.oPtq, QAT.oQat, QAT.oRatio]];
  pairs.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.05 - i * 1.70, top = Math.max(r[1], r[2]);
    sh.push(fitT(.70, y + .95, r[0], {anchor:'start', c:COL.tealL, fs:.28}));
    sh.push(S.r(.70, y + .46, Math.max(.06, 5.6 * r[1] / top), .40, 'bad', null));
    sh.push(S.r(.70, y - .06, Math.max(.06, 5.6 * r[2] / top), .40, 'ok', null));
    sh.push(fitT(9.30, y + .66, {zh:'先訓練再四捨五入 ' + r[1].toFixed(3), en:'train then round ' + r[1].toFixed(3)},
                 {anchor:'end', c:COL.red, fs:.25}));
    sh.push(fitT(9.30, y + .14, {zh:'訓練時就帶著四捨五入 ' + r[2].toFixed(3), en:'train through rounding ' + r[2].toFixed(3)},
                 {anchor:'end', c:COL.tealL, fs:.25}));
    sh.push(fitT(9.30, y - .36, r[3].toFixed(2) + 'x', {anchor:'end', c:COL.pale, fs:.26}));
  });
  return caps(sh, o);
}
function qatFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:qatStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:3,
    cap:{zh:'訓練時就讓前向看到四捨五入', en:'let the forward pass see the rounding during training'},
    panels:[{lbl:'PTQ', chips:[chip(QAT.ptq.toFixed(3), 'bad')]},
            {lbl:'QAT', chips:[chip(QAT.qat.toFixed(3), 'ok'), chip(QAT.ratio.toFixed(2) + 'x', 'ok')]}],
    msg:{zh:'round() 幾乎處處梯度為零，所以反向傳遞直接假裝它是恆等函數——straight-through estimator。權重仍然以浮點存在、前向看到的是四捨五入後的值，於是最佳化器會找到「**四捨五入之後**才好」的權重，而那跟「最好的浮點權重再四捨五入」不是同一件事。' + QAT.bits + ' bit 下差了 ' + QAT.ratio.toFixed(2) + ' 倍。',
         en:'round() has zero gradient almost everywhere, so the backward pass simply pretends it is the identity - the straight-through estimator. The weights stay in float while the forward pass sees rounded values, so the optimiser lands on weights that are good *after* rounding, which is not the same thing as the best float weights rounded. At ' + QAT.bits + ' bits that is a ' + QAT.ratio.toFixed(2) + 'x difference.'}});
  p({upto:1, line:4,
    cap:{zh:'而這個好處是從哪裡來的', en:'and where that gain actually comes from'},
    panels:[{lbl:'correlated', chips:[chip(QAT.ratio.toFixed(2) + 'x', 'ok')]},
            {lbl:'orthogonal', chips:[chip(QAT.oRatio.toFixed(2) + 'x', 'bad')]}],
    msg:{zh:'把輸入換成幾乎正交的，好處幾乎消失（' + QAT.oRatio.toFixed(2) + 'x）。因為這時每個權重各自獨立，「最好的量化權重」就真的等於「最好的權重再量化」，沒有東西可以學。**好處完全來自「一個權重可以吸收另一個權重的四捨五入誤差」**——也就是 GPTQ 用 H = XᵀX 閉式算出來的同一個性質。訓練是把它學出來，GPTQ 是把它算出來；輸入不重疊時兩者都沒有用。',
         en:'Swap the inputs for nearly orthogonal ones and the gain almost disappears (' + QAT.oRatio.toFixed(2) + 'x). With orthogonal inputs each weight stands alone, so the best quantized weights really are the quantized best weights and there is nothing to learn. The gain comes entirely from one weight being able to absorb another one\'s rounding error - the same property GPTQ computes in closed form through H = XᵀX. Training discovers it; GPTQ derives it; neither works on inputs that do not overlap.'}});
  p({upto:1, line:2,
    cap:{zh:'還有一個論文放在註腳的細節', en:'and one detail the papers put in a footnote'},
    panels:[{lbl:'best', chips:[chip(QAT.qat.toFixed(3), 'ok')]},
            {lbl:'last step', chips:[chip(QAT.last.toFixed(3), 'bad')]},
            {lbl:'swing', chips:[chip('±' + QAT.osc.toFixed(2), 'dim')]}],
    msg:{zh:'QAT 的 loss 不是單調下降的：最後 300 步的標準差是 ' + QAT.osc.toFixed(2) + '，最後一步是 ' + QAT.last.toFixed(3) + '、最好的是 ' + QAT.qat.toFixed(3) + '。因為前向是一個階梯函數，權重一跨過格點邊界輸出就跳一下。所以 QAT 都是用驗證集挑 checkpoint，而不是直接拿最後一步——面對不平滑的目標函數，那是唯一合理的做法。',
         en:'The QAT loss does not fall monotonically: over the last 300 steps it swings with a standard deviation of ' + QAT.osc.toFixed(2) + ', the final iterate scores ' + QAT.last.toFixed(3) + ' and the best one ' + QAT.qat.toFixed(3) + '. The forward pass is a step function, so a weight crossing a rounding boundary moves the output discontinuously. QAT runs therefore pick a checkpoint by validation loss rather than taking the last step - the only sane thing to do with a non-smooth objective.'}});
  return F.list;
}

/* ================================================================ tabs 8-9: the frameworks */
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
const CODE_TORCH = [
  'import torch',
  'a8 = a.to(torch.float8_e4m3fn)          # a real dtype',
  'torch._scaled_mm(a8, b8, scale_a, scale_b, out_dtype=torch.bfloat16)',
  '',
  'from torchao.quantization import quantize_, Int4WeightOnlyConfig',
  'quantize_(model, Int4WeightOnlyConfig(group_size=128))   # tab 3',
  '',
  'class RoundSTE(torch.autograd.Function):                 # tab 7',
  '    @staticmethod',
  '    def forward(ctx, x): return torch.round(x)',
  '    @staticmethod',
  '    def backward(ctx, g): return g'
];
const TORCH_ROWS = [
  [{zh:'quant_int(W, 8, axis=-1)', en:'quant_int(W, 8, axis=-1)'},
   {zh:'torchao Int8WeightOnlyConfig', en:'torchao Int8WeightOnlyConfig'}],
  [{zh:'group=128', en:'group=128'}, {zh:'Int4WeightOnlyConfig(group_size=128)', en:'Int4WeightOnlyConfig(group_size=128)'}],
  [{zh:'quant_fp8(x, "e4m3")', en:'quant_fp8(x, "e4m3")'}, {zh:'torch.float8_e4m3fn ＋ _scaled_mm', en:'torch.float8_e4m3fn and _scaled_mm'}],
  [{zh:'gptq() / awq_scale()', en:'gptq() / awq_scale()'}, {zh:'torchao GPTQ／AWQ（要校準資料）', en:'torchao GPTQ / AWQ, with calibration data'}],
  [{zh:'fake_quant ＋ 直通梯度', en:'fake_quant and the straight-through gradient'},
   {zh:'autograd.Function 或 torchao.quantization.qat', en:'an autograd.Function, or torchao.quantization.qat'}]
];
function torchFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:TORCH_ROWS, title:'PyTorch'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:5,
    cap:{zh:'PyTorch：量化的是「模組」', en:'PyTorch quantizes modules'},
    panels:[{lbl:'torchao', chips:[chip('quantize_(model, cfg)', 'hot')]}],
    msg:{zh:'`quantize_(model, Int4WeightOnlyConfig(group_size=128))` 就是第三個分頁那一列：4 bit、每 128 個權重一個 scale。它會把 Linear 的權重換成一個量化過的 tensor 子類別，forward 時再反量化——所以模型結構不變，變的是裡面的 tensor。',
         en:'`quantize_(model, Int4WeightOnlyConfig(group_size=128))` is literally the row from tab three: 4 bits with a scale per 128 weights. It swaps the Linear layer\'s weight for a quantized tensor subclass that dequantizes in the forward pass, so the model structure is untouched and only the tensors change.'}});
  p({upto:2, line:3,
    cap:{zh:'FP8 是真的 dtype，但 scale 是你的事', en:'FP8 is a real dtype, and the scale is yours'},
    panels:[{lbl:'dtype', chips:[chip('float8_e4m3fn', 'ok')]},
            {lbl:'matmul', chips:[chip('torch._scaled_mm', 'act')]}],
    msg:{zh:'`torch.float8_e4m3fn` 是真的 tensor dtype，`torch._scaled_mm(a, b, scale_a, scale_b)` 吃 FP8 輸入與 scale、回傳 bf16。注意 **scale 要自己算**——那就是第三、四個分頁在講的事。torchao.float8 則把整套包好：前向與反向都用 FP8 矩陣乘法，per-tensor 或 per-row scale，這就是 FP8 預訓練的做法。',
         en:'`torch.float8_e4m3fn` is a real tensor dtype and `torch._scaled_mm(a, b, scale_a, scale_b)` takes FP8 inputs plus scales and returns bf16. Note that **the scale is yours to compute** - which is exactly tabs three and four. torchao.float8 wraps the whole thing: FP8 matmuls in both the forward and backward passes with per-tensor or per-row scales, which is how FP8 pretraining runs are done.'}});
  p({upto:4, line:7,
    cap:{zh:'而 QAT 就是一個 autograd.Function', en:'and QAT is one autograd.Function'},
    panels:[{lbl:'forward', chips:[chip('round(x)', 'dim')]},
            {lbl:'backward', chips:[chip('g', 'ok')]}],
    msg:{zh:'直通估計在 PyTorch 裡是一個 `autograd.Function`：forward 回傳 round(x)，backward 原封不動把梯度傳回去。實務上用 `torchao.quantization.qat`（QATConfig／FakeQuantizeConfig）包好，但裡面就是這六行。',
         en:'The straight-through estimator is one `autograd.Function`: the forward returns round(x) and the backward hands the gradient straight back. In practice `torchao.quantization.qat` (QATConfig / FakeQuantizeConfig) wraps it, but this is the whole mechanism.'}});
  return F.list;
}
const CODE_JAX = [
  'import jax, jax.numpy as jnp',
  'x8 = x.astype(jnp.float8_e4m3fn)',
  'jax.lax.dot_general(a8, b8, dims,',
  '                    preferred_element_type=jnp.bfloat16)   # accumulate wide',
  '',
  '# the straight-through estimator, with no autograd class:',
  'def ste_round(x):',
  '    return x + jax.lax.stop_gradient(jnp.round(x) - x)',
  '',
  '# AQT / Qwix wrap a dot_general so the same function trains and serves'
];
const JAX_ROWS = [
  [{zh:'quant_fp8(x, "e4m3")', en:'quant_fp8(x, "e4m3")'}, {zh:'jnp.float8_e4m3fn', en:'jnp.float8_e4m3fn'}],
  [{zh:'累加用更寬的精度', en:'accumulate in a wider type'}, {zh:'preferred_element_type', en:'preferred_element_type'}],
  [{zh:'fake_quant ＋ 直通梯度', en:'fake_quant and the straight-through gradient'},
   {zh:'x + stop_gradient(round(x) - x)', en:'x + stop_gradient(round(x) - x)'}],
  [{zh:'整套量化流程', en:'the whole quantization flow'}, {zh:'AQT／Qwix（包住 dot_general）', en:'AQT / Qwix, wrapping dot_general'}]
];
function jaxFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:JAX_ROWS, title:'JAX'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:4,
    cap:{zh:'JAX：量化的是「運算」', en:'JAX quantizes operations'},
    panels:[{lbl:'dtype', chips:[chip('jnp.float8_e4m3fn', 'ok')]},
            {lbl:'accumulate', chips:[chip('bfloat16', 'dim')]}],
    msg:{zh:'JAX 沒有「模組」可以替換，所以量化發生在函數裡：`dot_general(..., preferred_element_type=jnp.bfloat16)` 的意思就是「用 8 bit 乘、用 16 bit 累加」。這也是兩邊最大的心智差異——PyTorch 換掉 module 裡的 tensor，JAX 換掉那個 op。',
         en:'JAX has no module to swap, so quantization happens inside the function: `dot_general(..., preferred_element_type=jnp.bfloat16)` says "multiply in 8 bits, accumulate in 16". That is the real mental difference between the two: PyTorch replaces the tensor inside a module, JAX replaces the operation.'}});
  p({upto:2, line:7,
    cap:{zh:'直通估計在 JAX 裡是一行恆等式', en:'the straight-through estimator is one identity in JAX'},
    panels:[{lbl:'value', chips:[chip('round(x)', 'dim')]},
            {lbl:'gradient', chips:[chip('as if x', 'ok')]}],
    msg:{zh:'`x + stop_gradient(round(x) - x)`：前向算出來就是 round(x)，而梯度因為 stop_gradient 的關係只流過前面那個 x——等於把 round 的導數當成 1。沒有自訂 autograd 類別，就是一行代數。看到這個寫法就知道那裡在做 QAT。',
         en:'`x + stop_gradient(round(x) - x)`: the forward value is exactly round(x), and the gradient only flows through the leading x, which is the same as pretending round has derivative 1. No custom autograd class, just algebra - and recognising this line is how you spot QAT in a JAX codebase.'}});
  p({upto:3, line:9,
    cap:{zh:'而函式庫把 dot_general 包起來', en:'and the libraries wrap dot_general'},
    panels:[{lbl:'AQT', chips:[chip('quantized training', 'ok')]},
            {lbl:'Qwix', chips:[chip('PTQ / QAT / LoRA', 'act')]}],
    msg:{zh:'AQT（google/aqt）是量化訓練的函式庫：你把一個 dot_general 包起來，它處理 scale、校準與反向傳遞，於是同一個函式可以用 int8 訓練、也可以用 int8 服務。Qwix 是比較新的那個，PTQ／QAT／LoRA 都蓋到，而且不用改寫模型。對照 PyTorch：那邊是 torchao 換 module 裡的 tensor，這邊是包住運算本身——**scale、分組、直通梯度這些算術完全一樣**。',
         en:'AQT (google/aqt) is the quantized-training library: wrap a dot_general and it handles the scales, the calibration and the backward pass, so the same function trains in int8 and serves in int8. Qwix is the newer one, covering PTQ, QAT and LoRA flows without rewriting the model. Against PyTorch: torchao swaps the tensor inside a module, JAX wraps the operation - and the arithmetic underneath (scales, grouping, straight-through gradients) is identical.'}});
  return F.list;
}

/* ================================================================ tab: the format table */
const CODE_SPEC = [
  '# a float is three fields packed into one word:',
  '#   sign | exponent | mantissa',
  '# value = (-1)**sign * 2**(exponent - bias) * 1.mantissa',
  '',
  'bias = 2**(e_bits - 1) - 1',
  'emax = 2**e_bits - 1 - bias - (1 if has_infinities else 0)',
  'largest  = (2 - 2**-m_bits) * 2**emax',
  'rel_step = 2**-m_bits            # how much precision you get',
  '',
  'np.float32(0.15625).view(np.uint32)   # see the bits for yourself'
];
function specStage(o){
  const sh = [];
  if (o.upto === 0){
    const fields = [[{zh:'sign 1 bit', en:'sign, 1 bit'}, BITS.sign, 'hot', 1.0],
                    [{zh:'exponent 8 bits', en:'exponent, 8 bits'}, BITS.exp, 'ok', 2.6],
                    [{zh:'mantissa 23 bits', en:'mantissa, 23 bits'}, BITS.man.slice(0, 12) + '...', 'act', 4.2]];
    fields.forEach((f, i) => {
      const y = 4.10 - i * 1.05;
      sh.push(fitT(.70, y + .62, f[0], {anchor:'start', c:COL.tealL, fs:.27}));
      sh.push(box(.70, y, f[3], .52, f[2], f[1], .26));
    });
    sh.push(fitT(5.20, 4.36, {zh:'＋1', en:'+1'}, {anchor:'start', c:COL.pale, fs:.28}));
    sh.push(fitT(5.20, 3.31, {zh:'存 ' + BITS.stored + ' − bias ' + BITS.bias + ' = ' + BITS.unbiased,
                              en:'stored ' + BITS.stored + ' - bias ' + BITS.bias + ' = ' + BITS.unbiased},
                 {anchor:'start', c:COL.pale, fs:.27}));
    sh.push(fitT(5.20, 2.26, {zh:'1.01… = ' + BITS.mantissa, en:'1.01... = ' + BITS.mantissa},
                 {anchor:'start', c:COL.pale, fs:.27}));
    sh.push(fitT(4.9, 1.10, BITS.mantissa + ' x 2^' + BITS.unbiased + ' = ' + BITS.x,
                 {c:COL.tealL, fs:.32}));
  } else {
    const rows = SPEC.filter((r, i) => o.upto >= 2 || r.bits >= 8);
    const unit = 0.175;
    rows.forEach((r, i) => {
      const y = 4.78 - i * .42;
      sh.push(fitT(2.15, y + .14, r.name, {anchor:'end', c:COL.tealL, fs:.24}));
      let x = 2.30;
      sh.push(S.r(x, y, unit, .32, 'hot', null)); x += unit;
      if (r.e){ sh.push(box(x, y, unit * r.e, .32, 'ok', unit * r.e > .42 ? 'E' + r.e : '', .20)); x += unit * r.e; }
      sh.push(box(x, y, unit * r.m, .32, 'act', unit * r.m > .42 ? (r.e ? 'M' + r.m : r.m + 'b') : '', .20));
      sh.push(fitT(7.70, y + .14, 'max ' + r.max.toPrecision(3),
                   {anchor:'end', c:COL.grey, fs:.23}));
      sh.push(fitT(9.30, y + .14, {zh:'步長 ' + r.eps.toPrecision(2), en:'step ' + r.eps.toPrecision(2)},
                   {anchor:'end', c:COL.grey, fs:.23}));
    });
  }
  return caps(sh, o);
}
function specFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:specStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'先看 fp32 在記憶體裡到底長什麼樣', en:'first: what an fp32 actually looks like in memory'},
    panels:[{lbl:'value', chips:[chip(String(BITS.x), 'dim')]},
            {lbl:'exponent', chips:[chip(BITS.stored + ' - ' + BITS.bias, 'ok')]}],
    msg:{zh:'一個浮點數是三個欄位塞在同一個字裡：符號、指數、尾數，讀法是 (−1)^sign × 2^(指數−bias) × 1.尾數。指數存的是**加了 bias 的無號數**（這樣兩個浮點數當成整數比大小，順序還是對的）；而小數點前面那個 1 **根本不存**，是隱含的，等於白賺一位精度。',
         en:'A float is three fields packed into one word - sign, exponent, mantissa - and read as (-1)^sign x 2^(exponent - bias) x 1.mantissa. The exponent is stored **biased, as an unsigned number**, so that comparing two floats as integers still gives the right order; and the leading 1 before the point is **not stored at all**, which is one free bit of precision.'}});
  p({upto:1, line:7,
    cap:{zh:'於是每個格式就是「指數幾位、尾數幾位」', en:'so a format is just: how many exponent bits, how many mantissa bits'},
    panels:[{lbl:'exponent bits', chips:[chip('range', 'ok')]},
            {lbl:'mantissa bits', chips:[chip('precision', 'act')]}],
    msg:{zh:'指數位數買的是**範圍**，尾數位數買的是**精度**。bf16 跟 fp32 的指數同樣是 8 位，所以 fp32 的任何數值放進 bf16 都不會溢位；fp16 在同樣 16 位下換成「精度多、範圍少」，這正是 fp16 訓練需要 loss scaling 而 bf16 不用的原因。',
         en:'Exponent bits buy **range** and mantissa bits buy **precision**. bf16 has the same 8 exponent bits as fp32, so no fp32 value overflows when cast to bf16; fp16 makes the opposite trade at the same 16 bits - more precision, far less range - which is exactly why fp16 training needs loss scaling and bf16 does not.'}});
  p({upto:2, line:8,
    cap:{zh:'同樣的取捨在 8、6、4 位上重演一次', en:'and the same trade repeats at 8, 6 and 4 bits'},
    panels:[{lbl:'e4m3fn', chips:[chip('448', 'ok')]},
            {lbl:'e5m2', chips:[chip('57344', 'act')]},
            {lbl:'fp4 e2m1', chips:[chip('max 6', 'dim')]}],
    msg:{zh:'名字就是欄位寬度：e5m2 ＝ 5 位指數、2 位尾數。`float8_e4m3fn` 的 **fn ＝ finite**，它放棄無限大，換到多一個指數級距（最大 448 而不是 240）。MX 格式再往下：' + MX.map(r => r.name + ' ' + r.bpv.toFixed(2) + ' bit').join('、') + '——每 ' + MX[0].block + ' 個元素共用一個 E8M0（純 2 的冪次）指數，所以套用 scale 是加指數而不是乘法。',
         en:'The name is the field widths: e5m2 is 5 exponent bits and 2 mantissa bits. The **fn in float8_e4m3fn means finite** - it gives up infinities to gain one more binade of range (448 instead of 240). The MX formats go further: ' + MX.map(r => r.name + ' at ' + r.bpv.toFixed(2) + ' bits').join(', ') + ' - every ' + MX[0].block + ' elements share one E8M0 exponent, which is a pure power of two, so applying the scale is an exponent add rather than a multiply.'}});
  return F.list;
}

/* ================================================================ tab: the methods */
const CODE_METH = [
  '# three questions separate every method:',
  '#   1. what gets quantized      - weights, activations, gradients, the KV cache',
  '#   2. how the scale is set     - absmax, MSE search, migrated, learned',
  '#   3. is the rounding improved - round-to-nearest, error feedback, trained',
  '',
  '# and what it costs to apply:',
  '#   free        : run it on the checkpoint',
  '#   calibration : a few hundred real sequences, offline',
  '#   training    : a training run'
];
function methStage(o){
  const sh = [];
  const groups = ['free', 'calibration', 'training'];
  const rows = METH.filter(function(r){ return groups.indexOf(r.cost) <= o.upto; });
  const show = rows.slice(0, 11);
  show.forEach(function(r, i){
    const y = 4.80 - i * .42;
    const st = r.cost === 'free' ? 'ok' : (r.cost === 'calibration' ? 'act' : 'hot');
    sh.push(box(.70, y, 3.30, .34, st, r.name, .22));
    sh.push(fitT(4.20, y + .17, r.what, {anchor:'start', c:COL.grey, fs:.22}));
    sh.push(fitT(6.00, y + .17, r.bits, {anchor:'start', c:COL.pale, fs:.22}));
    sh.push(fitT(9.30, y + .17, r.cost,
                 {anchor:'end', c:st === 'ok' ? COL.tealL : (st === 'act' ? '#c7a6ff' : COL.red), fs:.22}));
  });
  if (rows.length > show.length)
    sh.push(fitT(4.9, .62, {zh:'…還有 ' + (rows.length - show.length) + ' 個，完整表在 README 與文章裡',
                            en:'... and ' + (rows.length - show.length) + ' more; the full table is in the README'},
                 {c:COL.grey, fs:.24}));
  return caps(sh, o);
}
function methFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:methStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:6,
    cap:{zh:'不用任何資料就能套的', en:'the ones that need no data at all'},
    panels:[{lbl:'cost', chips:[chip('free', 'ok')]}],
    msg:{zh:'第一層階梯：這些方法對著 checkpoint 直接跑就好。RTN、LLM.int8()、HQQ、NF4／QLoRA、GGUF 的 k-quants、FP8、MX／NVFP4、KV cache 量化都在這一層——它們之間的差別是「scale 怎麼挑」和「格點長什麼樣」，不是需不需要資料。',
         en:'The first rung: these run on the checkpoint and nothing else. RTN, LLM.int8(), HQQ, NF4/QLoRA, the GGUF k-quants, FP8, MX/NVFP4 and KV-cache quantization all live here. What separates them is how the scale is chosen and what the grid looks like, not whether they need data.'}});
  p({upto:1, line:7,
    cap:{zh:'需要幾個校準 batch 的', en:'the ones that want a few calibration batches'},
    panels:[{lbl:'cost', chips:[chip('calibration', 'act')]}],
    msg:{zh:'第二層：SmoothQuant、AWQ、GPTQ、SqueezeLLM、SpQR、QuIP# 都需要看過「這個模型實際會吃到什麼」。幾百條真實序列、離線跑一次，很便宜——但從此結果取決於你餵了什麼進去，這是實務上最大的差別。',
         en:'The second rung: SmoothQuant, AWQ, GPTQ, SqueezeLLM, SpQR and QuIP# all need to see what the model is actually used on. A few hundred real sequences, run once, offline - cheap, but from here the result depends on what you fed it, which is the real practical difference.'}});
  p({upto:2, line:8,
    cap:{zh:'需要一次訓練的', en:'the ones that want a training run'},
    panels:[{lbl:'cost', chips:[chip('training', 'bad')]},
            {lbl:'W4', chips:[chip(NOTE.bpw.toFixed(2) + ' bits', 'dim')]}],
    msg:{zh:'第三層只有 QAT 與 AQLM 這類：它們能到最低的位元數，但需要一次訓練，所以在「不是自己訓練這個模型」的團隊裡很少見。順帶記號：**' + NOTE.name + ' ＝ 4-bit 權重、16-bit 活化**（weight-only，部署預設），而每 ' + NOTE.group + ' 個權重一個 scale 時 W4 實際是 ' + NOTE.bpw.toFixed(2) + ' bit。W8A8／FP8 連活化一起量化，換到的是 weight-only 拿不到的 tensor core 吞吐量，在 prefill 才有意義。',
         en:'The third rung is QAT and codebook methods like AQLM: they reach the lowest bit widths and they need a training run, which is why they stay rare outside the labs that trained the model. The notation while we are here: **' + NOTE.name + ' is 4-bit weights with 16-bit activations** - weight-only, the serving default - and with a scale per ' + NOTE.group + ' weights W4 really costs ' + NOTE.bpw.toFixed(2) + ' bits. W8A8 and FP8 quantize activations too, buying tensor-core throughput that weight-only cannot, which matters in compute-bound prefill.'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'量化：更少的位元，和四捨五入的代價', en:'Quantization: fewer bits, and what the rounding costs'},
  sub: {zh:'哪些數字共用一個 scale、離群值放哪裡、以及要讓權重誤差最小還是輸出誤差最小',
        en:'which numbers share a scale, where the outliers go, and whether to minimise the error in the weights or in the output'},
  tabs: [
    {id:'why', label:{zh:'為什麼要做', en:'why bother'},
     stage:{zh:'每個 token 都要把權重讀一遍', en:'every token reads all the weights'},
     view:VIEW, code:CODE_WHY, build:whyFrames,
     idea:{zh:'70B 模型：bf16 是 ' + MEM[0].gb.toFixed(0) + ' GB、一次讀 ' + MEM[0].ms.toFixed(0) + ' ms；int4（group 128，實際 ' + MEM[3].bits.toFixed(2) + ' bit）是 ' + MEM[3].gb.toFixed(1) + ' GB、' + MEM[3].ms.toFixed(1) + ' ms。容量與延遲一起砍四倍，代價就是四捨五入。',
           en:'A 70B model in bf16 is ' + MEM[0].gb.toFixed(0) + ' GB and a ' + MEM[0].ms.toFixed(0) + ' ms read; int4 with a group of 128 (really ' + MEM[3].bits.toFixed(2) + ' bits) is ' + MEM[3].gb.toFixed(1) + ' GB and ' + MEM[3].ms.toFixed(1) + ' ms. Capacity and latency both cut by four, and the price is the rounding.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'已經看過的', en:'rows already shown'}]]},
    {id:'grid', label:{zh:'一個量化的數字', en:'what a quantized number is'},
     stage:{zh:'整數 × scale', en:'an integer times a scale'},
     view:VIEW, code:CODE_GRID, build:gridFrames,
     idea:{zh:'scale = max|x| / 127，四捨五入到整數格點，用的時候乘回來。每個誤差最多半格，而**半格有多大是 group 裡最大的那個數決定的**——所以整個主題只剩一個問題：誰跟誰共用 scale。',
           en:'scale = max|x| / 127, round to an integer, multiply back. Every error is at most half a step, and **the size of that step is set by the largest value in the group** - which reduces the whole subject to one question: who shares a scale with whom.'},
     legend:[['#ff9736', {zh:'目前這個數字', en:'the value being shown'}],
             ['#12b3b8', {zh:'已量化的', en:'already quantized'}]]},
    {id:'formats', label:{zh:'所有格式一覽', en:'every format on one page'},
     stage:{zh:'sign / exponent / mantissa', en:'sign, exponent, mantissa'},
     view:VIEW, code:CODE_SPEC, build:specFrames,
     idea:{zh:'浮點＝(−1)^sign × 2^(指數−bias) × 1.尾數，指數位數買範圍、尾數位數買精度。fp32 S1E8M23、bf16 S1E8M7（跟 fp32 同指數故不溢位）、fp16 S1E5M10（精度多但範圍少，所以要 loss scaling）、fp8 e4m3fn 最大 448（fn＝沒有無限大）、e5m2 最大 57344、fp4 e2m1 最大 6；MX 系列每 32 個共用一個 E8M0 指數 ⇒ ' + MX.map(function(r){return r.name + ' ' + r.bpv.toFixed(2) + ' bit';}).join('、') + '。',
           en:'A float is (-1)^sign x 2^(exponent - bias) x 1.mantissa: exponent bits buy range, mantissa bits buy precision. fp32 is S1E8M23, bf16 S1E8M7 (same exponent as fp32, so casting never overflows), fp16 S1E5M10 (more precision, far less range - hence loss scaling), fp8 e4m3fn tops out at 448 (fn means no infinities), e5m2 at 57344, fp4 e2m1 at 6. The MX formats share one E8M0 exponent per 32 values: ' + MX.map(function(r){return r.name + ' at ' + r.bpv.toFixed(2) + ' bits';}).join(', ') + '.'},
     legend:[['#ff9736', {zh:'sign', en:'sign'}],
             ['#12b3b8', {zh:'exponent：範圍', en:'exponent: range'}],
             ['#9d6bff', {zh:'mantissa：精度', en:'mantissa: precision'}]]},
    {id:'share', label:{zh:'誰共用一個 scale', en:'who shares a scale'},
     stage:{zh:'位元換 scale', en:'bits against scales'},
     view:VIEW, code:CODE_SHARE, build:shareFrames,
     idea:{zh:'同一組權重：int8 per tensor ' + FMT[0].rel.toFixed(4) + '、per row ' + FMT[1].rel.toFixed(4) + '、int4 per row ' + FMT[4].rel.toFixed(4) + '、int4 group 128 ' + FMT[5].rel.toFixed(4) + '、MXFP4 block 32 ' + FMT[6].rel.toFixed(4) + '。**4 bit 但 scale 細，贏過 8 bit 但一個 scale**。',
           en:'The same weights: int8 per tensor ' + FMT[0].rel.toFixed(4) + ', per row ' + FMT[1].rel.toFixed(4) + ', int4 per row ' + FMT[4].rel.toFixed(4) + ', int4 with a group of 128 ' + FMT[5].rel.toFixed(4) + ', MXFP4 with a block of 32 ' + FMT[6].rel.toFixed(4) + '. **Four bits with fine-grained scales beats eight bits with one scale.**'},
     legend:[['#12b3b8', {zh:'8 bit 的方案', en:'8-bit schemes'}],
             ['#9d6bff', {zh:'4 bit 的方案', en:'4-bit schemes'}]]},
    {id:'floats', label:{zh:'整數還是浮點', en:'integers or floats'},
     stage:{zh:'同樣 8 bit，兩種格點', en:'the same 8 bits, two grids'},
     view:VIEW, code:CODE_FLOAT, build:floatFrames,
     idea:{zh:'int8 是 ' + DR.int8_steps + ' 個等距格點、動態範圍 ' + DR.int8_ratio.toFixed(0) + ':1，絕對誤差固定；FP8 把位元花在指數上，e4m3 的範圍是 ' + DR.e4m3_ratio.toExponential(1) + '、相對誤差固定。e4m3（3 bit 尾數）給前向，e5m2（多一位指數）給梯度——FP8 訓練兩種都用。',
           en:'int8 gives ' + DR.int8_steps + ' evenly spaced levels and a ' + DR.int8_ratio.toFixed(0) + ':1 range, so the absolute error is constant. FP8 spends bits on an exponent: e4m3 covers ' + DR.e4m3_ratio.toExponential(1) + ' with a roughly constant relative error. e4m3 (3 mantissa bits) is for the forward pass, e5m2 (one more exponent bit) for gradients - FP8 training uses both.'},
     legend:[['#12b3b8', {zh:'int8', en:'int8'}],
             ['#9d6bff', {zh:'fp8 e4m3', en:'fp8 e4m3'}],
             ['#ff9736', {zh:'fp8 e5m2', en:'fp8 e5m2'}]]},
    {id:'methods', label:{zh:'所有方法一覽', en:'every method on one page'},
     stage:{zh:'量化什麼、scale 怎麼挑、四捨五入有沒有改良', en:'what is quantized, how the scale is set, whether the rounding is improved'},
     view:VIEW, code:CODE_METH, build:methFrames,
     idea:{zh:'三個問題分開所有方法：量化什麼（權重／活化／梯度／KV）、scale 怎麼挑（absmax／MSE 搜尋／搬移／學出來）、四捨五入有沒有被改良（RTN／誤差回饋／訓練）。實務上最有用的分類是代價：不用資料（RTN、LLM.int8()、HQQ、NF4、GGUF k-quants、FP8、MX、KV 量化）、需要校準（SmoothQuant、AWQ、GPTQ、SqueezeLLM、SpQR、QuIP#）、需要訓練（QAT、AQLM）。',
           en:'Three questions separate every method: what is quantized (weights, activations, gradients, KV cache), how the scale is chosen (absmax, MSE search, migrated, learned) and whether the rounding itself is improved (round-to-nearest, error feedback, trained). The most useful axis in practice is what it costs: nothing (RTN, LLM.int8(), HQQ, NF4, GGUF k-quants, FP8, MX, KV quantization), calibration (SmoothQuant, AWQ, GPTQ, SqueezeLLM, SpQR, QuIP#), or a training run (QAT, AQLM).'},
     legend:[['#12b3b8', {zh:'不用資料', en:'no data needed'}],
             ['#9d6bff', {zh:'需要校準', en:'needs calibration'}],
             ['#ff9736', {zh:'需要訓練', en:'needs training'}]]},
    {id:'outliers', label:{zh:'離群值', en:'the outliers'},
     stage:{zh:'難的是活化不是權重', en:'the activations, not the weights'},
     view:VIEW, code:CODE_OUT, build:outFrames,
     idea:{zh:OUT.nOut + ' / ' + OUT.d + ' 個 channel 的值是中位數的 ' + OUT.ratio + ' 倍：只量化權重誤差 ' + ERR[0].rel.toFixed(4) + '，只量化活化（per tensor）就是 ' + ERR[1].rel.toFixed(4) + '。三種解法：留著（LLM.int8()，' + MP.all_int8.toFixed(4) + '→' + MP.rel.toFixed(4) + '）、搬走（SmoothQuant alpha 0.5，' + SQ[2].err.toFixed(4) + '）、或給它自己的 scale（per-token／group／MXFP4）。',
           en:OUT.nOut + ' of ' + OUT.d + ' channels carry ' + OUT.ratio + 'x the median value: quantizing only the weights costs ' + ERR[0].rel.toFixed(4) + ', quantizing only the activations per tensor costs ' + ERR[1].rel.toFixed(4) + '. Three fixes: keep them (LLM.int8(), ' + MP.all_int8.toFixed(4) + ' to ' + MP.rel.toFixed(4) + '), move them (SmoothQuant at alpha 0.5, ' + SQ[2].err.toFixed(4) + '), or give them their own scale (per-token, group-wise, MXFP4).'},
     legend:[['#ff5c5c', {zh:'被離群值毀掉的', en:'ruined by the outliers'}],
             ['#12b3b8', {zh:'還好的', en:'still fine'}],
             ['#9d6bff', {zh:'SmoothQuant 的 alpha', en:'SmoothQuant alphas'}]]},
    {id:'round', label:{zh:'更好的四捨五入', en:'rounding better'},
     stage:{zh:'GPTQ 與 AWQ', en:'GPTQ and AWQ'},
     view:VIEW, code:CODE_ROUND, build:roundFrames,
     idea:{zh:'4 bit／group 128：RTN 輸出誤差 ' + RT.rtn_out.toFixed(4) + '、GPTQ ' + RT.gptq_out.toFixed(4) + '、AWQ ' + RT.awq_out.toFixed(4) + '。**GPTQ 的權重誤差反而更大**（' + RT.gptq_weight_err.toFixed(3) + ' vs ' + RT.rtn_weight_err.toFixed(3) + '）——因為它要的不是好的權重，是好的輸出：逐 column 四捨五入，再用 H = XᵀX 把誤差推到還沒處理的 column。',
           en:'At 4 bits with a group of 128: round-to-nearest gives an output error of ' + RT.rtn_out.toFixed(4) + ', GPTQ ' + RT.gptq_out.toFixed(4) + ', AWQ ' + RT.awq_out.toFixed(4) + '. **GPTQ\'s weights are worse** (' + RT.gptq_weight_err.toFixed(3) + ' against ' + RT.rtn_weight_err.toFixed(3) + ') because it is not after good weights but a good output: round column by column and push the error into the remaining columns through H = XᵀX.'},
     legend:[['#ff5c5c', {zh:'round to nearest', en:'round to nearest'}],
             ['#12b3b8', {zh:'GPTQ', en:'GPTQ'}],
             ['#9d6bff', {zh:'AWQ', en:'AWQ'}]]},
    {id:'qat', label:{zh:'訓練時就量化', en:'training through the rounding'},
     stage:{zh:'直通估計，以及它的好處從哪來', en:'the straight-through estimator, and where its gain comes from'},
     view:VIEW, code:CODE_QAT, build:qatFrames,
     idea:{zh:QAT.bits + ' bit：先訓練再量化 ' + QAT.ptq.toFixed(3) + '，訓練時帶著量化 ' + QAT.qat.toFixed(3) + '（' + QAT.ratio.toFixed(2) + 'x）。但把輸入換成幾乎正交的，好處只剩 ' + QAT.oRatio.toFixed(2) + 'x——**好處完全來自「一個權重能吸收另一個的四捨五入誤差」**，跟 GPTQ 用 H 算出來的是同一個性質。另外 loss 會震盪（±' + QAT.osc.toFixed(2) + '），所以要用驗證集挑 checkpoint。',
           en:'At ' + QAT.bits + ' bits: train then round gives ' + QAT.ptq.toFixed(3) + ', training through the rounding gives ' + QAT.qat.toFixed(3) + ' (' + QAT.ratio.toFixed(2) + 'x). Swap the inputs for nearly orthogonal ones and the gain collapses to ' + QAT.oRatio.toFixed(2) + 'x: **it comes entirely from one weight absorbing another one\'s rounding error**, the same property GPTQ computes through H. The loss also oscillates (±' + QAT.osc.toFixed(2) + '), so QAT picks a checkpoint by validation loss.'},
     legend:[['#ff5c5c', {zh:'先訓練再量化', en:'train then round'}],
             ['#12b3b8', {zh:'訓練時就量化', en:'train through the rounding'}]]},
    {id:'torch', label:{zh:'在 PyTorch 裡', en:'in PyTorch'},
     stage:{zh:'量化的是模組裡的 tensor', en:'quantizing the tensor inside a module'},
     view:VIEW, code:CODE_TORCH, build:torchFrames,
     idea:{zh:'`quantize_(model, Int4WeightOnlyConfig(group_size=128))` 就是第三個分頁那一列；`torch.float8_e4m3fn` 是真 dtype，`torch._scaled_mm` 吃 FP8 ＋ scale（**scale 要自己算**）；QAT 則是一個 `autograd.Function`：forward 回 round(x)、backward 原樣傳回。',
           en:'`quantize_(model, Int4WeightOnlyConfig(group_size=128))` is the row from tab three; `torch.float8_e4m3fn` is a real dtype and `torch._scaled_mm` takes FP8 plus scales (**the scale is yours to compute**); and QAT is one `autograd.Function` whose forward returns round(x) and whose backward hands the gradient back untouched.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'PyTorch 裡的名字', en:'what PyTorch calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'jax', label:{zh:'在 JAX 裡', en:'in JAX'},
     stage:{zh:'量化的是那個運算', en:'quantizing the operation itself'},
     view:VIEW, code:CODE_JAX, build:jaxFrames,
     idea:{zh:'JAX 沒有 module 可以換，量化寫在函數裡：`dot_general(..., preferred_element_type=bfloat16)` ＝ 8 bit 乘、16 bit 累加；直通估計是一行 `x + stop_gradient(round(x) - x)`；AQT／Qwix 把 dot_general 包起來，同一個函式可以訓練也可以服務。',
           en:'JAX has no module to swap, so the numerics live in the function: `dot_general(..., preferred_element_type=bfloat16)` means multiply in 8 bits and accumulate in 16; the straight-through estimator is the single line `x + stop_gradient(round(x) - x)`; and AQT / Qwix wrap a dot_general so one function both trains and serves.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'JAX 裡的名字', en:'what JAX calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]}
  ]
};
