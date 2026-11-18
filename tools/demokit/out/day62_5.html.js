// DAY: 62.5
// TITLE_ZH: GPU 記憶體階層與 roofline：為什麼有些 kernel 再怎麼優化都不會變快
// TITLE_EN: The memory hierarchy and the roofline: why some kernels never get faster
// SUB_ZH: 這個系列裡每一個效能數字最後都是記憶體數字——decode 慢是因為每個 token 都要讀一遍權重、KV cache 是容量上限、MoE 是通訊問題、量化有用是因為位元變少。這一天把底下那個模型寫出來：階層長什麼樣、roofline 怎麼決定一個 kernel 是被算力還是被記憶體卡住，以及 tiling（包含 FlashAttention）為什麼是唯一能提高運算密度的辦法。
// SUB_EN: Every performance number in this series turned out to be a memory number: decode is slow because it reads the weights once per token, the KV cache is the capacity limit, MoE is a communication problem, quantization helps because fewer bits means fewer bytes. This day writes down the model underneath: the hierarchy, the roofline that decides whether a kernel is limited by arithmetic or by memory, and why tiling - FlashAttention included - is the only way to raise arithmetic intensity.
// FOLDER: day%2062.5%20-%20memory%20hierarchy%20and%20roofline
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
const HIER = {"MI300X": [{"name": "registers (VGPR per CU)", "gbs": 100000.0, "ns": 1, "cap": 524288, "src": "estimate"}, {"name": "LDS / shared memory per CU", "gbs": 20000.0, "ns": 20, "cap": 65536, "src": "estimate"}, {"name": "L2 per XCD (8 of them)", "gbs": 10000.0, "ns": 200, "cap": 4194304, "src": "estimate"}, {"name": "Infinity Cache (MALL)", "gbs": 17000.0, "ns": 300, "cap": 268435456, "src": "spec"}, {"name": "HBM3", "gbs": 5300.0, "ns": 600, "cap": 206158430208, "src": "spec"}, {"name": "host memory over PCIe 5 x16", "gbs": 64.0, "ns": 2000, "cap": 0, "src": "spec"}, {"name": "another node over RDMA", "gbs": 50.0, "ns": 10000, "cap": 0, "src": "spec"}], "H200 SXM": [{"name": "registers (per SM)", "gbs": 100000.0, "ns": 1, "cap": 262144, "src": "estimate"}, {"name": "shared memory per SM", "gbs": 20000.0, "ns": 20, "cap": 233472, "src": "estimate"}, {"name": "L2 cache", "gbs": 7000.0, "ns": 200, "cap": 52428800, "src": "estimate"}, {"name": "HBM3e", "gbs": 4800.0, "ns": 600, "cap": 151397597184, "src": "spec"}, {"name": "host memory over PCIe 5 x16", "gbs": 64.0, "ns": 2000, "cap": 0, "src": "spec"}, {"name": "another node over RDMA", "gbs": 50.0, "ns": 10000, "cap": 0, "src": "spec"}], "H100 SXM": [{"name": "registers (per SM)", "gbs": 100000.0, "ns": 1, "cap": 262144, "src": "estimate"}, {"name": "shared memory per SM", "gbs": 20000.0, "ns": 20, "cap": 233472, "src": "estimate"}, {"name": "L2 cache", "gbs": 7000.0, "ns": 200, "cap": 52428800, "src": "estimate"}, {"name": "HBM3", "gbs": 3350.0, "ns": 600, "cap": 85899345920, "src": "spec"}, {"name": "host memory over PCIe 5 x16", "gbs": 64.0, "ns": 2000, "cap": 0, "src": "spec"}, {"name": "another node over RDMA", "gbs": 50.0, "ns": 10000, "cap": 0, "src": "spec"}]};
const SPECS = {"MI300X": {"vendor": "AMD, CDNA 3", "tflops": 1307.0, "balance": 246.6, "hbm": 5300.0}, "H200 SXM": {"vendor": "NVIDIA, Hopper", "tflops": 990.0, "balance": 206.2, "hbm": 4800.0}, "H100 SXM": {"vendor": "NVIDIA, Hopper", "tflops": 990.0, "balance": 295.5, "hbm": 3350.0}};
const SEQ = [{"kb": 64, "gbs": 17.7}, {"kb": 256, "gbs": 23.8}, {"kb": 1024, "gbs": 23.5}, {"kb": 4096, "gbs": 25.4}, {"kb": 16384, "gbs": 12.1}, {"kb": 65536, "gbs": 19.3}, {"kb": 262144, "gbs": 15.1}];
const RND = [{"kb": 16, "ns": 1.4}, {"kb": 64, "ns": 1.05}, {"kb": 256, "ns": 1.12}, {"kb": 1024, "ns": 2.29}, {"kb": 4096, "ns": 7.04}, {"kb": 16384, "ns": 11.29}, {"kb": 65536, "ns": 11.12}];
const RF = [{"name": "decode: one linear (GEMV)", "ai": 1.0, "bound": "memory", "ms": 0.022}, {"name": "prefill: one linear (GEMM)", "ai": 3185.78, "bound": "compute", "ms": 0.368}, {"name": "decode: attention over KV", "ai": 1.0, "bound": "memory", "ms": 0.013}, {"name": "prefill: attention (no tiling)", "ai": 128.0, "bound": "memory", "ms": 0.405}, {"name": "RMSNorm", "ai": 1.0, "bound": "memory", "ms": 0.013}, {"name": "residual add", "ai": 0.17, "bound": "memory", "ms": 0.019}, {"name": "softmax over the vocabulary", "ai": 1.25, "bound": "memory", "ms": 0.0}, {"name": "decode: the whole model", "ai": 1.0, "bound": "memory", "ms": 26.415}];
const MACH = {"name": "MI300X", "tflops": 1307.0, "gbs": 5300.0, "balance": 246.6};
const BATCH = [{"b": 1, "ai": 1.0, "bound": "memory", "ms": 0.022, "us": 22.16}, {"b": 4, "ai": 4.0, "bound": "memory", "ms": 0.022, "us": 5.54}, {"b": 16, "ai": 15.96, "bound": "memory", "ms": 0.022, "us": 1.39}, {"b": 64, "ai": 63.43, "bound": "memory", "ms": 0.022, "us": 0.35}, {"b": 256, "ai": 247.17, "bound": "compute", "ms": 0.023, "us": 0.09}, {"b": 1024, "ai": 896.0, "bound": "compute", "ms": 0.092, "us": 0.09}];
const TILE = {"n": 4096, "tile": 104, "naiveGB": 549.8, "tiledGB": 5.35, "ratio": 102.7, "naiveAI": 0.25, "tiledAI": 25.67, "sweep": [{"t": 8, "fits": true, "gb": 68.79, "ai": 2.0}, {"t": 16, "fits": true, "gb": 34.43, "ai": 3.99}, {"t": 32, "fits": true, "gb": 17.25, "ai": 7.97}, {"t": 64, "fits": true, "gb": 8.66, "ai": 15.88}, {"t": 128, "fits": false, "gb": 4.36, "ai": 31.51}, {"t": 256, "fits": false, "gb": 2.21, "ai": 62.06}, {"t": 512, "fits": false, "gb": 1.14, "ai": 120.47}]};
const FL = {"seq": 4096, "block": 64, "passes": 64, "standard_gb": 4.429, "tiled_gb": 4.362, "tiled_hbm_gb": 0.134, "ratio": 1.015, "ratio_hbm": 33.0, "standard_ai": 62.061, "tiled_ai": 2048.0, "score_matrix_gb": 1.074, "kv_per_head_mb": 2.097, "kv_fits_cache": true, "cache_mb": 268.435, "sram_kb": 64.0, "memory_ratio": 16384.0, "standard_passes": 4, "scaling": [{"seq": 512, "std": 0.08, "hbm": 0.02, "ratio": 5.0, "mat": 0.02, "fits": true}, {"seq": 2048, "std": 1.14, "hbm": 0.07, "ratio": 17.0, "mat": 0.27, "fits": true}, {"seq": 8192, "std": 17.45, "hbm": 0.27, "ratio": 65.0, "mat": 4.29, "fits": true}, {"seq": 32768, "std": 275.95, "hbm": 1.07, "ratio": 257.0, "mat": 68.72, "fits": true}]};
const LIT = {"bandwidth_gbs": 5300.0, "latency_ns": 600.0, "bytes_in_flight": 3180000.0, "requests_in_flight": 24843.8, "bytes_per_request": 128};
const ST = [{"s": 1, "useful": 16.78, "fetched": 16.78, "eff": 1.0, "ms": 0.86}, {"s": 2, "useful": 8.39, "fetched": 16.78, "eff": 0.5, "ms": 0.71}, {"s": 4, "useful": 4.19, "fetched": 16.78, "eff": 0.25, "ms": 0.49}, {"s": 8, "useful": 2.1, "fetched": 16.78, "eff": 0.125, "ms": 0.42}, {"s": 16, "useful": 1.05, "fetched": 16.78, "eff": 0.062, "ms": 0.59}, {"s": 32, "useful": 0.52, "fetched": 8.39, "eff": 0.062, "ms": 0.35}];
const LOG = (v, lo, hi) => (Math.log(Math.max(v, lo)) / Math.LN10 - Math.log(lo) / Math.LN10)
                           / (Math.log(hi) / Math.LN10 - Math.log(lo) / Math.LN10);

/* ================================================================ tab 1: the hierarchy */
const CODE_HIER = [
  '# a level of the hierarchy is three numbers:',
  '#   capacity   - how much fits',
  '#   bandwidth  - how fast it streams',
  '#   latency    - how long one access takes',
  '',
  '# and the ratio that matters is between neighbours:',
  'shared_memory_bandwidth / hbm_bandwidth   # ~4x',
  'hbm_bandwidth / pcie_bandwidth            # ~80x'
];
function hierStage(o){
  const sh = [];
  const name = o.machine || 'MI300X';
  const rows = HIER[name];
  const w = [0.8, 1.3, 1.9, 2.6, 3.4, 4.2, 4.9];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.55 - i * .62;
    sh.push(fitT(3.30, y + .27, r.name, {anchor:'end', c:COL.tealL, fs:.24}));
    sh.push(S.r(3.45, y, w[i], .50, i === o.upto ? 'hot' : 'ok', null));
    sh.push(fitT(8.95, y + .27, r.gbs.toLocaleString() + ' GB/s (' +
                 (r.src === 'spec' ? 'spec' : 'est.') + ')',
                 {anchor:'end', c:COL.pale, fs:.23}));
    sh.push(fitT(9.60, y + .27, r.ns.toLocaleString() + ' ns', {anchor:'end', c:COL.grey, fs:.22}));
  });
  sh.push(fitT(.70, 5.05, {zh:name + '（' + SPECS[name].vendor + '，' + SPECS[name].tflops.toFixed(0) + ' TFLOP/s bf16）：越寬＝容量越大',
                           en:name + ' (' + SPECS[name].vendor + ', ' + SPECS[name].tflops.toFixed(0) + ' TFLOP/s bf16) - wider is bigger'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function hierFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:hierStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:3, machine:'MI300X', line:1,
    cap:{zh:'晶片裡面：暫存器、LDS、L2、Infinity Cache',
         en:'on the chip: registers, LDS, L2 and the Infinity Cache'},
    panels:[{lbl:'registers', chips:[chip('100,000 GB/s', 'ok')]},
            {lbl:'shared', chips:[chip('20,000 GB/s', 'act')]},
            {lbl:'L2', chips:[chip('7,000 GB/s', 'dim')]}],
    msg:{zh:'先說來源：標 (spec) 的是廠商公布的（產品頁、data sheet、CDNA 3／Hopper 白皮書、PCIe 與乙太標準），標 (est.) 的是我自己估的數量級——**晶片內部的頻寬與延遲幾乎不會寫在規格書上**，所以那些全是估計值。每一層大約比上一層慢一個數量級、大兩個數量級；晶片內的幾層加起來只有幾十 MB，所有「把資料留在這裡」的優化爭的都是這幾十 MB。',
         en:'Provenance first: rows marked (spec) are published by the vendor - product pages, data sheets, the CDNA 3 and Hopper whitepapers, the PCIe and Ethernet standards - and rows marked (est.) are my own order-of-magnitude figures, because **on-chip bandwidths and latencies are almost never in a datasheet**. Each level is roughly an order of magnitude slower and two orders larger than the one above, and the on-chip levels add up to a few tens of megabytes: every optimisation that keeps data close is competing for exactly those.'}});
  p({upto:HIER['MI300X'].length - 1, machine:'MI300X', line:7,
    cap:{zh:'晶片外面：HBM、主機記憶體、網路', en:'off the chip: HBM, host memory, the network'},
    panels:[{lbl:'HBM', chips:[chip('5,300 GB/s', 'dim')]},
            {lbl:'PCIe', chips:[chip('64 GB/s', 'bad')]},
            {lbl:'RDMA', chips:[chip('50 GB/s', 'bad')]}],
    msg:{zh:'shared memory 到 HBM 是每個 kernel 都在對抗的那一步。而最後兩列就是前面幾天：**主機記憶體那一列是 HiCache 分層存在的理由**，**RDMA 那一列是 DeepEP 要有兩套 kernel 的理由**。同一張表，三個不同的日子。',
         en:'The step from shared memory to HBM is the one every kernel in this series has been fighting. And the last two rows are earlier days: **the host-memory row is why HiCache has tiers**, and **the RDMA row is why DeepEP ships two different sets of kernels**. One table, three different days.'}});
  p({upto:HIER['H200 SXM'].length - 1, machine:'H200 SXM', line:7,
    cap:{zh:'換一台：同樣的形狀，不同的數字', en:'a different part: the same shape, different numbers'},
    panels:[{lbl:'MI300X', chips:[chip(SPECS['MI300X'].hbm.toLocaleString() + ' GB/s', 'ok'),
                                  chip(SPECS['MI300X'].balance + ' F/B', 'dim')]},
            {lbl:'H200 SXM', chips:[chip(SPECS['H200 SXM'].hbm.toLocaleString() + ' GB/s', 'act'),
                                    chip(SPECS['H200 SXM'].balance + ' F/B', 'dim')]}],
    msg:{zh:'換成 NVIDIA 的卡，階層的形狀一模一樣，只是數字不同：**L2 大很多（50 MB）、每個 SM 的 shared memory 也多（228 KB vs MI300X 的 64 KB）**——而 shared memory 的大小正是後面 tiling 那幾頁 block 大小的決定因素；代價是中間少了一層 Infinity Cache。HBM 這一列 ' + SPECS['H200 SXM'].hbm.toLocaleString() + ' GB/s，machine balance ' + SPECS['H200 SXM'].balance + ' FLOPs/byte。',
         en:'Move to an NVIDIA part and the shape of the hierarchy is identical, only the numbers move: **a far larger L2 (50 MB) and more shared memory per SM (228 KB against the MI300X\'s 64 KB)** - and shared memory is exactly what sets the block size in the tiling tabs below. The price is that there is no Infinity Cache in between. HBM runs at ' + SPECS['H200 SXM'].hbm.toLocaleString() + ' GB/s, for a machine balance of ' + SPECS['H200 SXM'].balance + ' FLOPs per byte.'}});
  p({upto:HIER['H100 SXM'].length - 1, machine:'H100 SXM', line:7,
    cap:{zh:'三張卡的 machine balance', en:'the machine balance of all three'},
    panels:[{lbl:'MI300X', chips:[chip(SPECS['MI300X'].hbm.toLocaleString() + ' GB/s', 'ok'),
                                  chip(SPECS['MI300X'].balance + ' F/B', 'dim')]},
            {lbl:'H200 SXM', chips:[chip(SPECS['H200 SXM'].hbm.toLocaleString() + ' GB/s', 'ok'),
                                    chip(SPECS['H200 SXM'].balance + ' F/B', 'act')]},
            {lbl:'H100 SXM', chips:[chip(SPECS['H100 SXM'].hbm.toLocaleString() + ' GB/s', 'dim'),
                                    chip(SPECS['H100 SXM'].balance + ' F/B', 'dim')]}],
    msg:{zh:'**H200 跟 H100 是同一顆晶片**，算力一樣是 ' + SPECS['H100 SXM'].tflops.toFixed(0) + ' TFLOP/s，差別只在記憶體：80 GB HBM3 變成 141 GB HBM3e、3,350 GB/s 變成 4,800 GB/s。所以 machine balance 從 ' + SPECS['H100 SXM'].balance + ' 掉到 ' + SPECS['H200 SXM'].balance + '——**被記憶體卡住的工作，這個數字愈小愈好**，三張卡的順序是 H200、MI300X（' + SPECS['MI300X'].balance + '）、H100。形狀一樣，所以後面的結論三張卡都成立；數字則一律用 MI300X 的。',
         en:'**H200 is the same die as H100**, same ' + SPECS['H100 SXM'].tflops.toFixed(0) + ' TFLOP/s; only the memory changed - 80 GB of HBM3 became 141 GB of HBM3e, and 3,350 GB/s became 4,800 GB/s. That drops the machine balance from ' + SPECS['H100 SXM'].balance + ' to ' + SPECS['H200 SXM'].balance + ', and **for anything memory-bound lower is better**, so the order is H200, then MI300X (' + SPECS['MI300X'].balance + '), then H100. The shape is the same on all three, so nothing below changes qualitatively; the numbers used below are the MI300X ones.'}});
  return F.list;
}

/* ================================================================ tab 2: measured */
const CODE_MEAS = [
  '# (a) in order: the prefetcher sees it coming',
  'a = np.ones(n, dtype=np.float32)',
  'a.sum()                       # 4n bytes read, n adds',
  '',
  '# (b) at random: it cannot',
  'idx = rng.permutation(n)',
  'a[idx].sum()                  # same bytes, unpredictable order'
];
function measStage(o){
  const sh = [];
  if (o.upto === 0){
    const top = Math.max.apply(null, SEQ.map(r => r.gbs));
    SEQ.forEach((r, i) => {
      const y = 4.40 - i * .52;
      sh.push(fitT(2.20, y + .16, r.kb < 1024 ? r.kb + ' KB' : (r.kb / 1024) + ' MB',
                   {anchor:'end', c:COL.grey, fs:.25}));
      sh.push(S.r(2.40, y, Math.max(.06, 5.4 * r.gbs / top), .34, 'ok', null));
      sh.push(fitT(9.30, y + .16, r.gbs.toFixed(1) + ' GB/s', {anchor:'end', c:COL.pale, fs:.24}));
    });
    sh.push(fitT(.70, 5.00, {zh:'(a) 依序讀：幾乎看不到階層', en:'(a) reading in order: the hierarchy barely shows'},
                 {anchor:'start', c:COL.tealL, fs:.27}));
  } else {
    const top = Math.max.apply(null, RND.map(r => r.ns));
    RND.forEach((r, i) => {
      const y = 4.40 - i * .52;
      sh.push(fitT(2.20, y + .16, r.kb < 1024 ? r.kb + ' KB' : (r.kb / 1024) + ' MB',
                   {anchor:'end', c:COL.grey, fs:.25}));
      sh.push(S.r(2.40, y, Math.max(.06, 5.4 * r.ns / top), .34, 'bad', null));
      sh.push(fitT(9.30, y + .16, r.ns.toFixed(2) + ' ns', {anchor:'end', c:COL.pale, fs:.24}));
    });
    sh.push(fitT(.70, 5.00, {zh:'(b) 隨機讀：整個階層跑出來了',
                             en:'(b) reading at random: the whole hierarchy appears'},
                 {anchor:'start', c:COL.tealL, fs:.27}));
  }
  return caps(sh, o);
}
function measFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:measStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const sMax = Math.max.apply(null, SEQ.map(r => r.gbs)), sMin = Math.min.apply(null, SEQ.map(r => r.gbs));
  const rMax = Math.max.apply(null, RND.map(r => r.ns)), rMin = Math.min.apply(null, RND.map(r => r.ns));
  p({upto:0, line:3,
    cap:{zh:'這台機器上實測（沒有 GPU，但階層是一樣的）',
         en:'measured on this machine - no GPU here, but the same shape'},
    panels:[{lbl:'best', chips:[chip(sMax.toFixed(1) + ' GB/s', 'ok')]},
            {lbl:'worst', chips:[chip(sMin.toFixed(1) + ' GB/s', 'dim')]},
            {lbl:'spread', chips:[chip((sMax / sMin).toFixed(1) + 'x', 'dim')]}],
    msg:{zh:'把陣列加總是純頻寬的工作，照理說一離開快取就該掉下去。實際上只差 ' + (sMax / sMin).toFixed(1) + ' 倍——因為**硬體 prefetcher 看得到循序讀，提早把資料抓進來**，大部分的延遲根本沒有發生。',
         en:'Summing an array is pure bandwidth, so it ought to fall off a cliff once the array leaves cache. It moves by only ' + (sMax / sMin).toFixed(1) + 'x, because **the prefetcher sees a sequential read coming and starts it early** - most of the latency never happens.'}});
  p({upto:1, line:6,
    cap:{zh:'同樣的位元組，換成猜不到的順序', en:'the same bytes, in an order nothing can predict'},
    panels:[{lbl:'in cache', chips:[chip(rMin.toFixed(2) + ' ns', 'ok')]},
            {lbl:'out of cache', chips:[chip(rMax.toFixed(2) + ' ns', 'bad')]},
            {lbl:'spread', chips:[chip((rMax / rMin).toFixed(0) + 'x', 'bad')]}],
    msg:{zh:'一換成隨機順序，階層立刻現形：在快取裡一次存取 ' + rMin.toFixed(2) + ' ns，掉出快取 ' + rMax.toFixed(2) + ' ns，差 ' + (rMax / rMin).toFixed(0) + ' 倍。**規格表賣你的是頻寬，決定你拿不拿得到的是局部性。** 這個系列裡所有看起來像指標追逐的結構——radix tree、block table、expert 權重的 gather——付的都是這一欄的價錢。',
         en:'Switch to a random order and the hierarchy appears immediately: ' + rMin.toFixed(2) + ' ns per access while it fits in cache, ' + rMax.toFixed(2) + ' ns when it does not, a factor of ' + (rMax / rMin).toFixed(0) + '. **Bandwidth is what the specification sells; locality decides whether you get it.** Every structure in this series that looks like a pointer chase - a radix tree, a block table, a gather of expert weights - pays the second column.'}});
  return F.list;
}

/* ================================================================ tab 3: the roofline */
const CODE_RF = [
  'intensity = flops / bytes_moved       # a property of the kernel',
  'balance   = peak_flops / bandwidth    # a property of the machine',
  '',
  'time  = max(flops / peak_flops, bytes / bandwidth)',
  'bound = "compute" if intensity > balance else "memory"'
];
function rfStage(o){
  const sh = [];
  const X0 = 1.40, Y0 = 4.55, W = 7.4, H = 3.1;
  const lo = 0.1, hi = 10000;
  const px = (ai) => X0 + W * LOG(ai, lo, hi);
  const py = (tf) => Y0 - H * LOG(tf, 1, 2000);
  sh.push(S.e(X0, Y0, X0 + W, Y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 - H, {s:'ghost', arrow:false, w:.04}));
  const ridge = MACH.balance, peak = MACH.tflops;
  let prev = null;
  [0.1, 1, 10, 100, ridge, 1000, 10000].forEach(function(ai){
    const tf = Math.min(peak, ai * MACH.gbs / 1000);
    const pt = [px(ai), py(tf)];
    if (prev) sh.push(S.e(prev[0], prev[1], pt[0], pt[1], {s:'bad', arrow:false, w:.055}));
    prev = pt;
  });
  sh.push(S.e(px(ridge), Y0, px(ridge), py(peak), {s:'ghost', arrow:false, w:.03}));
  sh.push(fitT(px(ridge), py(peak) - .28, {zh:'ridge ' + ridge.toFixed(0) + ' FLOPs/byte',
                                           en:'ridge ' + ridge.toFixed(0) + ' FLOPs/byte'},
               {c:'#ffbe6b', fs:.25}));
  [[0.1, '0.1'], [1, '1'], [100, '100'], [10000, '10k']].forEach(function(t){
    sh.push(fitT(px(t[0]), Y0 + .30, t[1], {c:COL.grey, fs:.23}));
  });
  sh.push(fitT(X0 + W / 2, Y0 + .68, {zh:'運算密度（FLOPs / byte）', en:'arithmetic intensity, FLOPs per byte'},
               {c:COL.grey, fs:.26}));
  const picks = ['residual add', 'decode: the whole model', 'prefill: attention (no tiling)',
                 'prefill: one linear (GEMM)'];
  picks.forEach(function(name, i){
    if (i > o.upto) return;
    const r = RF.filter(function(x){ return x.name === name; })[0];
    const tf = Math.min(peak, r.ai * MACH.gbs / 1000);
    sh.push(S.c(px(r.ai), py(tf), .10, 'hot', null));
    sh.push(fitT(px(r.ai), py(tf) + (i % 2 ? .34 : -.30),
                 name.replace('prefill: ', '').replace('decode: ', ''),
                 {c:COL.tealL, fs:.23}));
  });
  return caps(sh, o);
}
function rfFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:rfStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  const by = {}; RF.forEach(function(r){ by[r.name] = r; });
  p({upto:1, line:4,
    cap:{zh:MACH.name + ' 只有兩個數字', en:'the ' + MACH.name + ' is two numbers'},
    panels:[{lbl:'peak', chips:[chip(MACH.tflops + ' TFLOP/s', 'dim')]},
            {lbl:'bandwidth', chips:[chip(MACH.gbs + ' GB/s', 'dim')]},
            {lbl:'balance', chips:[chip(MACH.balance + ' FLOPs/byte', 'hot')]}],
    msg:{zh:'兩個數字的比值就是 machine balance：' + MACH.balance + ' FLOPs/byte。**一個 kernel 如果每讀一個 byte 做的運算比這個少，那它無論怎麼寫都不可能被算力卡住**——時間就是位元組除以頻寬。整個模型跑一次 decode 的運算密度是 ' + by['decode: the whole model'].ai + '，離 ridge 差兩個數量級。',
         en:'The ratio of the two is the machine balance: ' + MACH.balance + ' FLOPs per byte. **A kernel that does less arithmetic per byte than that cannot be compute-bound however it is written** - its time is bytes divided by bandwidth. A whole decode step sits at ' + by['decode: the whole model'].ai + ' FLOPs per byte, two orders of magnitude below the ridge.'}});
  p({upto:3, line:4,
    cap:{zh:'同一層 linear，兩邊都站得到', en:'the same linear layer sits on both sides'},
    panels:[{lbl:'decode GEMV', chips:[chip(by['decode: one linear (GEMV)'].ai.toFixed(2), 'bad')]},
            {lbl:'prefill GEMM', chips:[chip(by['prefill: one linear (GEMM)'].ai.toFixed(0), 'ok')]},
            {lbl:'residual add', chips:[chip(by['residual add'].ai.toFixed(2), 'bad')]}],
    msg:{zh:'同一組權重、同一個 kernel：decode 時是矩陣×向量，運算密度 ' + by['decode: one linear (GEMV)'].ai.toFixed(2) + '，被記憶體卡死；prefill 時是矩陣×矩陣，密度 ' + by['prefill: one linear (GEMM)'].ai.toFixed(0) + '，被算力卡住。**變的不是 kernel，是列數。** 而 RMSNorm 和 residual add 這種逐元素運算密度不到 1，時間完全是搬運——所以把它們融合進鄰居（省掉一讀一寫，不是省掉乘法）才會那麼值錢。',
         en:'Same weights, same kernel: in decode it is a matrix-vector product at ' + by['decode: one linear (GEMV)'].ai.toFixed(2) + ' FLOPs per byte and memory-bound; in prefill it is a matrix-matrix product at ' + by['prefill: one linear (GEMM)'].ai.toFixed(0) + ' and compute-bound. **The kernel did not change; the number of rows did.** And the pointwise rows - RMSNorm, a residual add - are below 1 FLOP per byte, so their time is entirely traffic, which is why fusing them into their neighbours (removing a read and a write, not a multiply) is worth so much.'}});
  return F.list;
}

/* ================================================================ tab 4: batching */
const CODE_BATCH = [
  '# the weights are read once whatever the batch size',
  'flops = 2 * batch * d_model * d_ff',
  'bytes = d_model * d_ff * 2     +  2 * batch * d_model * 2',
  '#       ^ the weights, once       ^ the activations, per row',
  '',
  'intensity = flops / bytes      # grows linearly with the batch'
];
function batchStage(o){
  const sh = [];
  const top = Math.max.apply(null, BATCH.map(r => r.us));
  BATCH.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.40 - i * .70;
    sh.push(fitT(1.60, y + .20, 'batch ' + r.b, {anchor:'end', c:COL.pale, fs:.26}));
    sh.push(S.r(1.80, y, Math.max(.06, 4.6 * r.us / top), .40,
                r.bound === 'memory' ? 'bad' : 'ok', null));
    sh.push(fitT(9.30, y + .20, {zh:r.ai.toFixed(1) + ' FLOPs/byte　每 token ' + r.us.toFixed(1) + ' µs',
                                 en:r.ai.toFixed(1) + ' FLOPs/byte,  ' + r.us.toFixed(1) + ' us per token'},
                 {anchor:'end', c:COL.grey, fs:.24}));
  });
  sh.push(fitT(.70, 4.95, {zh:'一層 linear 的每 token 成本（越短越好）',
                           en:'cost per token of one linear layer - shorter is better'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function batchFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:batchStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:2, line:3,
    cap:{zh:'小 batch：權重讀一遍，只服務幾列', en:'small batch: read all the weights, serve a few rows'},
    panels:[{lbl:'batch 1', chips:[chip(BATCH[0].us.toFixed(1) + ' us/token', 'bad')]},
            {lbl:'intensity', chips:[chip(BATCH[0].ai.toFixed(2), 'bad')]}],
    msg:{zh:'batch 1 的時候，整層權重被讀一遍只為了算一列——運算密度 ' + BATCH[0].ai.toFixed(2) + '，機器的算力完全在空轉。',
         en:'At batch 1 the whole weight matrix is read to compute a single row: ' + BATCH[0].ai.toFixed(2) + ' FLOPs per byte, and the arithmetic units idle.'}});
  p({upto:BATCH.length - 1, line:5,
    cap:{zh:'加大 batch 就是沿著 roofline 往右走', en:'batching is how you move along the roofline'},
    panels:[{lbl:'batch 256', chips:[chip(BATCH[4].bound, 'ok')]},
            {lbl:'per token', chips:[chip(BATCH[0].us.toFixed(1) + ' us', 'bad'),
                                     chip(BATCH[BATCH.length - 1].us.toFixed(2) + ' us', 'ok')]}],
    msg:{zh:'權重不管 batch 多大都只讀一遍，所以運算密度隨 batch 線性成長，直到越過 ridge。每個 token 的成本從 ' + BATCH[0].us.toFixed(1) + ' µs 掉到 ' + BATCH[BATCH.length - 1].us.toFixed(2) + ' µs——**兩個數量級**，而這就是 serving 的全部經濟學。前面講 speculative decoding 在大 batch 停止划算，量的是同一條曲線的另一端。',
         en:'The weights are read once whatever the batch, so intensity grows linearly until it crosses the ridge. The cost per token falls from ' + BATCH[0].us.toFixed(1) + ' us to ' + BATCH[BATCH.length - 1].us.toFixed(2) + ' us - **two orders of magnitude**, and that is the entire economics of serving. The day on decoding measured the same curve from the other end, when speculation stopped paying at large batch.'}});
  return F.list;
}

/* ================================================================ tab 5: tiling */
const CODE_TILE = [
  '# without tiling, B is re-read for every row of A',
  'bytes = 2 * n**3 * 4',
  '',
  '# with square tiles of size T that stay in cache:',
  'bytes = 2 * n**3 / T * 4          # T times less traffic',
  '',
  '# and T is not free: three tiles must fit at once',
  'T = int(sqrt(cache_floats / 3))'
];
function tileStage(o){
  const sh = [];
  const top = TILE.sweep[0].gb;
  TILE.sweep.forEach((r, i) => {
    if (i > o.upto + 2) return;
    const y = 4.50 - i * .55;
    sh.push(fitT(1.70, y + .17, 'T = ' + r.t, {anchor:'end', c:COL.pale, fs:.25}));
    sh.push(S.r(1.90, y, Math.max(.06, 4.6 * r.gb / top), .36, r.fits ? 'ok' : 'ghost', null));
    sh.push(fitT(9.30, y + .17, {zh:r.gb.toFixed(1) + ' GB　' + (r.fits ? '裝得下' : '裝不下'),
                                 en:r.gb.toFixed(1) + ' GB   ' + (r.fits ? 'fits in cache' : 'does not fit')},
                 {anchor:'end', c:r.fits ? COL.grey : COL.red, fs:.24}));
  });
  sh.push(fitT(.70, 5.05, {zh:TILE.n + 'x' + TILE.n + ' 的 matmul：搬多少位元組（不做 tiling 是 ' + TILE.naiveGB.toFixed(0) + ' GB）',
                           en:'a ' + TILE.n + 'x' + TILE.n + ' matmul: bytes moved (untiled: ' + TILE.naiveGB.toFixed(0) + ' GB)'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function tileFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:tileStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:5,
    cap:{zh:'kernel 不能選 FLOPs，只能選重讀幾次', en:'a kernel cannot choose its FLOPs, only how often it re-reads'},
    panels:[{lbl:'untiled', chips:[chip(TILE.naiveGB.toFixed(0) + ' GB', 'bad'),
                                   chip(TILE.naiveAI.toFixed(2) + ' F/B', 'bad')]},
            {lbl:'tiles of ' + TILE.tile, chips:[chip(TILE.tiledGB.toFixed(1) + ' GB', 'ok'),
                                                 chip(TILE.tiledAI.toFixed(1) + ' F/B', 'ok')]}],
    msg:{zh:'同樣的 matmul、同樣的 FLOPs：不做 tiling 要搬 ' + TILE.naiveGB.toFixed(0) + ' GB，切成 ' + TILE.tile + '×' + TILE.tile + ' 的 tile 只要 ' + TILE.tiledGB.toFixed(1) + ' GB，少 ' + TILE.ratio.toFixed(0) + ' 倍。**倍數剛好等於 tile 大小**，因為 B 的每一塊從「每列讀一次」變成「每個 tile-row 讀一次」。',
         en:'Same matmul, same FLOPs: untiled it moves ' + TILE.naiveGB.toFixed(0) + ' GB, in ' + TILE.tile + 'x' + TILE.tile + ' tiles it moves ' + TILE.tiledGB.toFixed(1) + ' GB, ' + TILE.ratio.toFixed(0) + 'x less. **The factor is exactly the tile size**, because each block of B goes from being read once per row to once per tile-row.'}});
  p({upto:TILE.sweep.length, line:7,
    cap:{zh:'而 tile 多大不是自由參數', en:'and the tile size is not a free parameter'},
    panels:[{lbl:'rule', chips:[chip('T = sqrt(cache / 3)', 'hot')]},
            {lbl:'here', chips:[chip(String(TILE.tile), 'ok')]}],
    msg:{zh:'三塊 tile（A、B、C 各一塊）必須同時待在快取裡，所以 T = √(cache/3)，這裡是 ' + TILE.tile + '。表裡「裝不下」的那幾列是模型在騙你：tile 一旦留不住就會被逐出再抓回來，預測的流量根本拿不到。**每一個函式庫裡的 blocked kernel，核心就是這一個算式加上細節。**',
         en:'Three tiles - one of A, one of B, one of C - must sit in cache at once, so T = sqrt(cache / 3), which is ' + TILE.tile + ' here. The rows marked "does not fit" are the model lying: a tile that cannot stay resident is evicted and re-fetched, so the predicted traffic never materialises. **Every blocked kernel in every library is this one calculation plus the details.**'}});
  return F.list;
}

/* ================================================================ tab 6: flash */
const CODE_FLASH = [
  '# standard: the seq x seq matrix lives in HBM and is walked several times',
  'S = Q @ K.T                 # written',
  'P = softmax(S)              # read, written',
  'O = P @ V                   # read',
  '',
  '# tiled: blocks of Q, K, V, O in SRAM; the matrix is never allocated',
  'for q_block in Q:           # Br rows at a time',
  '    for kv_block in zip(K, V):',
  '        online_softmax_update(acc, q_block, kv_block)'
];
function flashStage(o){
  const sh = [];
  const bars = [[{zh:'score 矩陣（必須常駐）', en:'the score matrix, resident'}, FL.score_matrix_gb, 'bad'],
                [{zh:'standard 搬運量', en:'standard, moved'}, FL.standard_gb, 'act'],
                [{zh:'tiled：算進每一層', en:'tiled, counting every level'}, FL.tiled_gb, 'idle'],
                [{zh:'tiled：真的到 HBM 的', en:'tiled, what reaches HBM'}, FL.tiled_hbm_gb, 'ok']];
  const top = Math.max.apply(null, bars.map(b => b[1]));
  bars.forEach((b, i) => {
    if (i > o.upto) return;
    const y = 4.40 - i * .85;
    sh.push(fitT(.70, y + .62, b[0], {anchor:'start', c:COL.tealL, fs:.27}));
    sh.push(S.r(.70, y, Math.max(.06, 6.6 * b[1] / top), .44, b[2], null));
    sh.push(fitT(9.30, y + .22, b[1].toFixed(2) + ' GB', {anchor:'end', c:COL.pale, fs:.26}));
  });
  if (o.upto >= 3)
    sh.push(fitT(4.9, .80, {zh:'算進每一層只少 ' + FL.ratio.toFixed(1) + ' 倍；真的到 HBM 的少 ' + FL.ratio_hbm.toFixed(1) + ' 倍；而那 ' + FL.score_matrix_gb.toFixed(2) + ' GB 從未被配置',
                            en:'counting every level: ' + FL.ratio.toFixed(1) + 'x. Reaching HBM: ' + FL.ratio_hbm.toFixed(1) + 'x. And the ' + FL.score_matrix_gb.toFixed(2) + ' GB is never allocated'},
                 {c:COL.pale, fs:.26}));
  return caps(sh, o);
}
function flashFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:flashStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:1,
    cap:{zh:'FlashAttention 就是上一個分頁的論證，用在 attention 上',
         en:'FlashAttention is the previous tab\'s argument, applied to attention'},
    panels:[{lbl:'score matrix', chips:[chip(FL.score_matrix_gb.toFixed(2) + ' GB', 'bad')]},
            {lbl:'standard', chips:[chip(FL.standard_gb.toFixed(2) + ' GB', 'act')]}],
    msg:{zh:'Day 52 已經證過 online softmax 的恆等式：一趟掃過 K、答案一位不差。它沒講的是「為什麼值得」，而答案就是上一個分頁那張表。標準做法把 ' + FL.seq + '×' + FL.seq + ' 的分數矩陣放進 HBM，再走過它好幾趟。',
         en:'Day 52 proved the online-softmax identity: one pass over the keys, the same answer to the last bit. What it did not say is why that is worth doing, and the answer is the previous tab. The standard implementation puts the ' + FL.seq + 'x' + FL.seq + ' score matrix in HBM and then walks over it several times.'}});
  p({upto:3, line:6,
    cap:{zh:'三個誠實的觀察', en:'three honest observations'},
    panels:[{lbl:'traffic', chips:[chip(FL.ratio.toFixed(1) + 'x', 'act')]},
            {lbl:'block', chips:[chip(FL.block + ' rows', 'dim')]},
            {lbl:'vs SRAM', chips:[chip(Math.round(FL.memory_ratio).toLocaleString() + 'x', 'bad')]}],
    msg:{zh:'第一：**算進每一層的話流量幾乎沒少（' + FL.ratio.toFixed(1) + ' 倍）**——' + FL.sram_kb.toFixed(0) + ' KB 的 LDS、d_head 128 時 block 只有 ' + FL.block + ' 列，K/V 要重讀 ' + FL.passes + ' 次。第二：**那些重讀大多沒有到 HBM**：一個 head 的 K+V 只有 ' + FL.kv_per_head_mb.toFixed(1) + ' MB，而 Infinity Cache 有 ' + FL.cache_mb.toFixed(0) + ' MB，所以真的跨到 HBM 的只有 ' + FL.tiled_hbm_gb.toFixed(2) + ' GB（少 ' + FL.ratio_hbm.toFixed(1) + ' 倍）——**只算一層的模型會算錯**。第三：**流量從來不是主要的獎品**，真正買到的是那 ' + FL.score_matrix_gb.toFixed(2) + ' GB 根本不存在（LDS 的 ' + Math.round(FL.memory_ratio).toLocaleString() + ' 倍），這是容量的論證。',
         en:'First: **counted naively the traffic barely moves (' + FL.ratio.toFixed(1) + 'x)** - with ' + FL.sram_kb.toFixed(0) + ' KB of LDS and d_head 128 the block is ' + FL.block + ' rows and K and V are re-read ' + FL.passes + ' times. Second: **those re-reads mostly never reach HBM**: K and V for one head are ' + FL.kv_per_head_mb.toFixed(1) + ' MB against a ' + FL.cache_mb.toFixed(0) + ' MB Infinity Cache, so what actually crosses to HBM is ' + FL.tiled_hbm_gb.toFixed(2) + ' GB, ' + FL.ratio_hbm.toFixed(1) + 'x less - **a model that counts one level will mispredict**. Third: **the traffic was never the main prize** - the ' + FL.score_matrix_gb.toFixed(2) + ' GB is never allocated at all, ' + Math.round(FL.memory_ratio).toLocaleString() + 'x the LDS the kernel runs in, which is a capacity argument.'}});
  p({upto:3, line:8,
    cap:{zh:'backward 也是同一招：只存 logsumexp，再重算',
         en:'the backward pass is the same trick: store the logsumexp, recompute'},
    panels:[{lbl:'stored', chips:[chip('1 number per row', 'ok')]},
            {lbl:'seq 32768', chips:[chip(FL.scaling[3].mat.toFixed(0) + ' GB matrix', 'bad')]}],
    msg:{zh:'反向傳遞一樣不存分數矩陣：它只留每一列的 log-sum-exp（一個數字，不是 seq 個數字），再從 Q 和 K 把分數重算出來。**重算比來回搬便宜**，而為什麼便宜，第三個分頁的表已經回答了。seq 32768 時那個矩陣是 ' + FL.scaling[3].mat.toFixed(0) + ' GB——沒有人能存它。',
         en:'The backward pass does not store the score matrix either: it keeps the log-sum-exp per row - one number instead of seq numbers - and recomputes the scores from Q and K. **Recomputation is cheaper than the round trip**, and the roofline table is why. At seq 32768 that matrix would be ' + FL.scaling[3].mat.toFixed(0) + ' GB, which nobody is going to store.'}});
  return F.list;
}

/* ================================================================ tab 7: latency */
const CODE_LAT = [
  '# Little\'s law: to keep the pipe full you need',
  'bytes_in_flight = bandwidth * latency',
  '5300e9 * 600e-9 = 3.2 MB   # on one GPU, at once',
  '',
  '# and a cache line is fetched whole, whatever you asked for',
  'a[::16].sum()      # a sixteenth of the payload, all of the lines'
];
function latStage(o){
  const sh = [];
  if (o.upto === 0){
    sh.push(box(.70, 3.60, 3.0, .70, 'ok', LIT.bandwidth_gbs.toLocaleString() + ' GB/s', .30));
    sh.push(fitT(4.00, 3.95, 'x', {c:COL.pale, fs:.32}));
    sh.push(box(4.40, 3.60, 2.2, .70, 'act', LIT.latency_ns + ' ns', .30));
    sh.push(fitT(6.90, 3.95, '=', {c:COL.pale, fs:.32}));
    sh.push(box(7.30, 3.60, 1.9, .70, 'hot', (LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB', .30));
    sh.push(fitT(4.9, 2.70, {zh:'同時必須有 ' + Math.round(LIT.requests_in_flight).toLocaleString() + ' 個請求在路上（每個 ' + LIT.bytes_per_request + ' bytes）',
                             en:Math.round(LIT.requests_in_flight).toLocaleString() + ' requests have to be in flight at once, at ' + LIT.bytes_per_request + ' bytes each'},
                 {c:COL.tealL, fs:.30}));
  } else {
    ST.forEach((r, i) => {
      const y = 4.30 - i * .62;
      sh.push(fitT(1.70, y + .18, 'stride ' + r.s, {anchor:'end', c:COL.pale, fs:.25}));
      sh.push(S.r(1.90, y, Math.max(.06, 3.4 * r.useful / ST[0].useful), .38, 'ok', null));
      sh.push(S.r(5.50, y, Math.max(.06, 3.4 * r.fetched / ST[0].fetched), .38, 'bad', null));
      sh.push(fitT(9.30, y + .18, (r.eff * 100).toFixed(0) + '%　' + r.ms.toFixed(2) + ' ms',
                   {anchor:'end', c:COL.grey, fs:.23}));
    });
    sh.push(fitT(1.90, 4.95, {zh:'有用的位元組', en:'useful bytes'}, {anchor:'start', c:COL.tealL, fs:.25}));
    sh.push(fitT(5.50, 4.95, {zh:'實際抓進來的', en:'bytes actually fetched'}, {anchor:'start', c:COL.red, fs:.25}));
  }
  return caps(sh, o);
}
function latFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:latStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'光有頻寬不夠：還要有足夠多的請求在路上',
         en:'bandwidth alone is not enough: the requests have to be in flight'},
    panels:[{lbl:'in flight', chips:[chip((LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB', 'hot')]},
            {lbl:'requests', chips:[chip(Math.round(LIT.requests_in_flight).toLocaleString(), 'bad')]}],
    msg:{zh:'記憶體系統有第二個數字，規格表很少跟第一個放在一起：**一次存取要多久**。要把管線填滿，任何時刻都必須有「頻寬 × 延遲」這麼多位元組在路上——這裡是 ' + (LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB，也就是 ' + Math.round(LIT.requests_in_flight).toLocaleString() + ' 個同時在飛的請求。所以一個「發一次 load 然後等」的 kernel 就算完全沒有多餘工作也會慢：硬體需要的是數千個未完成請求，而那只能靠很多執行緒、或一個執行緒展開很多次載入來供應。',
         en:'A memory system has a second number that specifications rarely put beside the first: **how long one access takes**. To keep the pipe full you need bandwidth times latency bytes in flight at every moment - ' + (LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB here, or ' + Math.round(LIT.requests_in_flight).toLocaleString() + ' simultaneous requests. Which is why a kernel that issues one load and waits is slow even with no redundant work: the hardware wants thousands of outstanding requests, and only many threads - or one thread unrolled over many loads - can supply them.'}});
  p({upto:1, line:5,
    cap:{zh:'而且 cache line 是整條抓的（本機實測）',
         en:'and a cache line is fetched whole - measured here'},
    panels:[{lbl:'stride 1', chips:[chip('100%', 'ok')]},
            {lbl:'stride 16', chips:[chip((ST[4].eff * 100).toFixed(0) + '%', 'bad')]},
            {lbl:'time', chips:[chip(ST[0].ms.toFixed(2) + ' ms', 'dim'),
                                chip(ST[4].ms.toFixed(2) + ' ms', 'bad')]}],
    msg:{zh:'到 stride 16 為止，「實際抓進來」那一欄完全不動：不管你跳多遠，每條 cache line 都被碰到，所以有用的比例一路掉到 ' + (ST[4].eff * 100).toFixed(0) + '%，而時間不但沒變少、還變多了。**這就是 GPU 上「coalescing」的全部內容**，在 CPU 上則是 array-of-structs 比 struct-of-arrays 慢的原因——同一個硬體事實。',
         en:'Up to stride 16 the "fetched" column does not move at all: every cache line is touched whatever the stride, so the useful fraction collapses to ' + (ST[4].eff * 100).toFixed(0) + '% while the time goes *up*. **That is the entire content of coalescing on a GPU**, and on a CPU it is why an array of structs loses to a struct of arrays. One hardware fact, two names.'}});
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
  'with torch.profiler.profile(activities=[CUDA]) as p:',
  '    model(x)',
  'p.key_averages().table(sort_by="self_cuda_time_total")',
  '',
  'torch.cuda.max_memory_allocated()',
  'torch.cuda.memory._record_memory_history()',
  '',
  'torch.compile(model)                       # fuses the pointwise chains',
  'F.scaled_dot_product_attention(q, k, v)    # dispatches to a tiled kernel'
];
const TORCH_ROWS = [
  [{zh:'哪個 kernel 花了時間', en:'which kernel spent the time'}, {zh:'torch.profiler', en:'torch.profiler'}],
  [{zh:'峰值記憶體與去向', en:'peak memory and where it went'}, {zh:'max_memory_allocated ／ _record_memory_history', en:'max_memory_allocated / _record_memory_history'}],
  [{zh:'逐元素 kernel 的流量', en:'the traffic of pointwise kernels'}, {zh:'torch.compile 的融合', en:'fusion via torch.compile'}],
  [{zh:'tiling 過的 attention', en:'a tiled attention kernel'}, {zh:'scaled_dot_product_attention ／ sdpa_kernel', en:'scaled_dot_product_attention / sdpa_kernel'}]
];
function torchFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:TORCH_ROWS, title:'PyTorch'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:3,
    cap:{zh:'profiler 回答的是位元組，不是 FLOPs', en:'the profiler answers in bytes, not FLOPs'},
    panels:[{lbl:'profiler', chips:[chip('per-kernel time', 'ok')]},
            {lbl:'memory', chips:[chip('peak + history', 'act')]}],
    msg:{zh:'把 roofline 放在手邊再去讀 profile，是完全不同的體驗：一個只跑到峰值算力 20% 的 kernel 不一定寫得爛，它可能已經到了峰值頻寬的 95%，而解法是**給它更高的運算密度**，不是給它更多執行緒。',
         en:'Reading a profile with the roofline in hand is a different experience: a kernel at 20% of peak FLOPs is not necessarily badly written - it may be at 95% of peak bandwidth, and the fix is **more arithmetic per byte**, not more threads.'}});
  p({upto:3, line:7,
    cap:{zh:'而最高槓桿的工具是融合', en:'and the highest-leverage tool is fusion'},
    panels:[{lbl:'torch.compile', chips:[chip('fuses pointwise', 'ok')]},
            {lbl:'SDPA', chips:[chip('tiled attention', 'act')]}],
    msg:{zh:'第三個分頁說了：逐元素 kernel 的時間完全是搬運。所以 `torch.compile` 把 norm＋add＋activation 融成一個 kernel，省掉的是一讀一寫而不是乘法——這是對記憶體有界的工作最有效的一招。而 `F.scaled_dot_product_attention` 就是幫你呼叫別人 tiling 好的 attention kernel（必要時用 `sdpa_kernel` 指定後端）。',
         en:'Tab three said it: a pointwise kernel\'s time is entirely traffic. So `torch.compile` fuses norm + add + activation into one kernel and saves a read and a write rather than a multiply - the most effective single move for memory-bound work. And `F.scaled_dot_product_attention` is how you call the tiled attention kernel somebody already wrote (with `sdpa_kernel` to pin the backend).'}});
  return F.list;
}
const CODE_JAX = [
  'compiled = jax.jit(f).lower(x).compile()',
  'compiled.memory_analysis()     # bytes in, out, temporaries',
  'compiled.cost_analysis()       # the FLOP count to divide by',
  '',
  'jax.jit(f, donate_argnums=(0,))   # reuse the input buffer',
  'jax.profiler.trace("/tmp/tb")',
  '',
  '# and when the compiler\'s tiling is not enough: Pallas,',
  '# where you choose the block sizes against the same constraint'
];
const JAX_ROWS = [
  [{zh:'流量與 FLOPs', en:'traffic and FLOPs'}, {zh:'memory_analysis ／ cost_analysis', en:'memory_analysis / cost_analysis'}],
  [{zh:'逐元素 kernel 的流量', en:'the traffic of pointwise kernels'}, {zh:'XLA 預設就融合', en:'XLA fuses by default'}],
  [{zh:'少配置一塊 buffer', en:'one buffer fewer'}, {zh:'donate_argnums', en:'donate_argnums'}],
  [{zh:'自己選 tile 大小', en:'choosing the block sizes yourself'}, {zh:'Pallas（＝Triton 那個模型）', en:'Pallas, the Triton model'}]
];
function jaxFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:JAX_ROWS, title:'JAX'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:3,
    cap:{zh:'編譯器可以在你跑之前就給你 roofline',
         en:'the compiler can hand you a roofline before you run anything'},
    panels:[{lbl:'memory_analysis', chips:[chip('bytes', 'ok')]},
            {lbl:'cost_analysis', chips:[chip('FLOPs', 'act')]}],
    msg:{zh:'`compiled.memory_analysis()` 給你這個函式搬了多少位元組、`cost_analysis()` 給你 FLOPs——兩個相除就是運算密度。**這是編譯器在你還沒執行之前就算好的 roofline**，而 XLA 預設就會融合，所以大部分逐元素的流量不用你開口就消失了。',
         en:'`compiled.memory_analysis()` gives the bytes this function moves and `cost_analysis()` gives the FLOPs; divide one by the other and you have its arithmetic intensity. **That is a roofline computed by the compiler before you run anything** - and XLA fuses by default, so most of the pointwise traffic disappears without being asked.'}});
  p({upto:3, line:7,
    cap:{zh:'而 tiling 住在不同的地方', en:'and the tiling lives somewhere else'},
    panels:[{lbl:'XLA', chips:[chip('tiles the fused region', 'ok')]},
            {lbl:'Pallas', chips:[chip('you choose the blocks', 'act')]}],
    msg:{zh:'跟 PyTorch 真正的差別在於 tiling 住在哪裡：PyTorch 是你呼叫別人 tiling 好的 kernel，或自己用 Triton 寫一個；JAX 是編譯器自己把融合後的區塊 tiling 掉，而當那還不夠好的時候你降到 Pallas——那就是 Triton 的模型：**你自己選 block 大小，而你在對抗的約束就是第五個分頁那一條。**',
         en:'The real difference from PyTorch is where the tiling lives. In PyTorch you call a kernel somebody tiled for you, or you write one in Triton. In JAX the compiler tiles the fused region itself, and when that is not good enough you drop to Pallas - which is the Triton model: **you choose the block sizes, and the constraint you are choosing against is the one in tab five.**'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'記憶體階層與 roofline', en:'The memory hierarchy and the roofline'},
  sub: {zh:'為什麼有些 kernel 再怎麼優化都不會變快，以及 tiling 為什麼是唯一的出路',
        en:'why some kernels never get faster, and why tiling is the only way out'},
  tabs: [
    {id:'hier', label:{zh:'階層', en:'the hierarchy'},
     stage:{zh:'容量、頻寬、延遲', en:'capacity, bandwidth, latency'},
     view:VIEW, code:CODE_HIER, build:hierFrames,
     idea:{zh:'每一層大約比上一層慢一個數量級、大兩個數量級。shared memory → HBM 是每個 kernel 都在對抗的那一步；而**主機記憶體那一列是 HiCache 分層的理由、RDMA 那一列是 DeepEP 要兩套 kernel 的理由**——同一張表，三個不同的日子。',
           en:'Each level is about an order of magnitude slower and two orders larger than the one above. The step from shared memory to HBM is what every kernel here fights, and the last two rows are earlier days: **host memory is why HiCache has tiers, RDMA is why DeepEP ships two sets of kernels**.'},
     legend:[['#ff9736', {zh:'目前這一層', en:'the level being shown'}],
             ['#12b3b8', {zh:'已經看過的', en:'levels already shown'}]]},
    {id:'meas', label:{zh:'實測：順序決定一切', en:'measured: order is everything'},
     stage:{zh:'依序讀 vs 隨機讀', en:'reading in order against at random'},
     view:VIEW, code:CODE_MEAS, build:measFrames,
     idea:{zh:'同一台機器、同樣的位元組：依序讀只差 1.x 倍（prefetcher 把延遲藏掉了），隨機讀差 ' + (Math.max.apply(null, RND.map(r => r.ns)) / Math.min.apply(null, RND.map(r => r.ns))).toFixed(0) + ' 倍。**規格表賣你的是頻寬，決定你拿不拿得到的是局部性。**',
           en:'Same machine, same bytes: read in order the spread is only about 1.x (the prefetcher hides the latency), read at random it is ' + (Math.max.apply(null, RND.map(r => r.ns)) / Math.min.apply(null, RND.map(r => r.ns))).toFixed(0) + 'x. **Bandwidth is what the specification sells; locality decides whether you get it.**'},
     legend:[['#12b3b8', {zh:'依序讀（GB/s）', en:'in order, GB/s'}],
             ['#ff5c5c', {zh:'隨機讀（ns／次）', en:'at random, ns per access'}]]},
    {id:'roof', label:{zh:'roofline', en:'the roofline'},
     stage:{zh:'運算密度決定被誰卡住', en:'arithmetic intensity decides the limit'},
     view:VIEW, code:CODE_RF, build:rfFrames,
     idea:{zh:'機器只有兩個數字，比值是 machine balance（' + MACH.balance + ' FLOPs/byte）。低於它的 kernel 無論怎麼寫都不可能被算力卡住。**同一層 linear：decode 時密度 1.0 被記憶體卡死、prefill 時密度 3186 被算力卡住——變的不是 kernel，是列數。**',
           en:'A machine is two numbers and their ratio is the balance (' + MACH.balance + ' FLOPs per byte); anything below it cannot be compute-bound however it is written. **The same linear layer: 1.0 FLOPs per byte in decode and memory-bound, 3186 in prefill and compute-bound - the kernel did not change, the number of rows did.**'},
     legend:[['#ff5c5c', {zh:'roofline', en:'the roofline'}],
             ['#ff9736', {zh:'這個系列裡的 kernel', en:'kernels from this series'}]]},
    {id:'batch', label:{zh:'batch 怎麼移動你', en:'what batching does'},
     stage:{zh:'沿著 roofline 往右走', en:'moving along the roofline'},
     view:VIEW, code:CODE_BATCH, build:batchFrames,
     idea:{zh:'權重不管 batch 多大都只讀一遍，所以運算密度隨 batch 線性成長：batch 1 的密度 1.0、batch 256 越過 ridge，而每個 token 的成本從 ' + BATCH[0].us.toFixed(1) + ' µs 掉到 ' + BATCH[BATCH.length - 1].us.toFixed(2) + ' µs。**這就是 serving 的全部經濟學**，也是 speculative decoding 在大 batch 停止划算的同一條曲線。',
           en:'The weights are read once whatever the batch, so intensity grows linearly: 1.0 at batch 1, past the ridge by batch 256, and the cost per token falls from ' + BATCH[0].us.toFixed(1) + ' to ' + BATCH[BATCH.length - 1].us.toFixed(2) + ' us. **That is the entire economics of serving** - and the same curve that makes speculative decoding stop paying at large batch.'},
     legend:[['#ff5c5c', {zh:'被記憶體卡住', en:'memory bound'}],
             ['#12b3b8', {zh:'被算力卡住', en:'compute bound'}]]},
    {id:'tile', label:{zh:'tiling', en:'tiling'},
     stage:{zh:'唯一能提高運算密度的辦法', en:'the only way to raise arithmetic intensity'},
     view:VIEW, code:CODE_TILE, build:tileFrames,
     idea:{zh:'kernel 不能選 FLOPs，只能選「同樣的位元組重讀幾次」。' + TILE.n + 'x' + TILE.n + ' 的 matmul：不 tiling 搬 ' + TILE.naiveGB.toFixed(0) + ' GB，切成 ' + TILE.tile + ' 的 tile 只搬 ' + TILE.tiledGB.toFixed(1) + ' GB（少 ' + TILE.ratio.toFixed(0) + ' 倍，**倍數剛好等於 tile 大小**）。而 tile 大小由快取決定：T = sqrt(cache/3)。',
           en:'A kernel cannot choose its FLOPs, only how often it re-reads the same bytes. A ' + TILE.n + 'x' + TILE.n + ' matmul moves ' + TILE.naiveGB.toFixed(0) + ' GB untiled and ' + TILE.tiledGB.toFixed(1) + ' GB in tiles of ' + TILE.tile + ' - ' + TILE.ratio.toFixed(0) + 'x less, **and the factor is exactly the tile size**. The tile size itself is set by the cache: T = sqrt(cache / 3).'},
     legend:[['#12b3b8', {zh:'裝得進快取', en:'fits in cache'}],
             ['#8fa3ac', {zh:'裝不進（模型失效）', en:'does not fit: the model stops holding'}]]},
    {id:'flash', label:{zh:'FlashAttention', en:'FlashAttention'},
     stage:{zh:'同一個論證，用在 attention 上', en:'the same argument, applied to attention'},
     view:VIEW, code:CODE_FLASH, build:flashFrames,
     idea:{zh:'Day 52 證過 online softmax 的恆等式；這裡是「為什麼值得」。誠實地說：**流量只少 ' + FL.ratio.toFixed(1) + ' 倍**（d_head 128 下 block 只有 ' + FL.block + ' 列），真正的獎品是那 ' + FL.score_matrix_gb.toFixed(2) + ' GB 的分數矩陣**從來沒有被配置過**——那是容量的論證，也是長 context 能被服務的理由。backward 一樣只存 logsumexp 再重算。',
           en:'Day 52 proved the online-softmax identity; this is why it is worth doing. Honestly: **the traffic win is only ' + FL.ratio.toFixed(1) + 'x** (at d_head 128 the block is just ' + FL.block + ' rows). The real prize is that the ' + FL.score_matrix_gb.toFixed(2) + ' GB score matrix **is never allocated** - a capacity argument, and the reason long contexts are servable. The backward pass does the same thing: store the logsumexp, recompute the scores.'},
     legend:[['#ff5c5c', {zh:'必須常駐的矩陣', en:'the resident matrix'}],
             ['#9d6bff', {zh:'standard 搬運量', en:'standard traffic'}],
             ['#12b3b8', {zh:'tiled 搬運量', en:'tiled traffic'}]]},
    {id:'lat', label:{zh:'延遲與 cache line', en:'latency and cache lines'},
     stage:{zh:'光有頻寬不夠', en:'bandwidth alone is not enough'},
     view:VIEW, code:CODE_LAT, build:latFrames,
     idea:{zh:'Little 法則：要填滿管線就必須隨時有「頻寬 × 延遲」＝ ' + (LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB（約 ' + Math.round(LIT.requests_in_flight).toLocaleString() + ' 個請求）在路上，所以「發一次 load 然後等」的 kernel 再乾淨也慢。另一半是 cache line 整條抓：本機實測 stride 16 時有用比例只剩 ' + (ST[4].eff * 100).toFixed(0) + '%，而時間還變多——這就是 coalescing。',
           en:'Little\'s law: keeping the pipe full needs bandwidth times latency - ' + (LIT.bytes_in_flight / 1e6).toFixed(1) + ' MB, about ' + Math.round(LIT.requests_in_flight).toLocaleString() + ' requests - in flight at all times, so a kernel that issues one load and waits is slow however clean it is. The other half is that cache lines are fetched whole: measured here, at stride 16 only ' + (ST[4].eff * 100).toFixed(0) + '% of the fetched bytes are useful and the time goes up. That is coalescing.'},
     legend:[['#12b3b8', {zh:'有用的位元組', en:'useful bytes'}],
             ['#ff5c5c', {zh:'實際抓進來的', en:'bytes fetched'}]]},
    {id:'torch', label:{zh:'在 PyTorch 裡', en:'in PyTorch'},
     stage:{zh:'profiler 與融合', en:'the profiler and fusion'},
     view:VIEW, code:CODE_TORCH, build:torchFrames,
     idea:{zh:'`torch.profiler` 告訴你時間花在哪個 kernel、`max_memory_allocated` 與 `_record_memory_history` 告訴你記憶體去哪了。而最高槓桿的是 `torch.compile` 的融合（省掉一讀一寫）與 `scaled_dot_product_attention`（呼叫別人 tiling 好的 kernel）。**拿著 roofline 讀 profile**：只到峰值算力 20% 的 kernel 可能已經在峰值頻寬的 95%。',
           en:'`torch.profiler` says which kernel spent the time; `max_memory_allocated` and `_record_memory_history` say where the memory went. The highest-leverage tools are `torch.compile` fusion (saving a read and a write) and `scaled_dot_product_attention` (calling a kernel somebody tiled). **Read the profile with the roofline in hand**: a kernel at 20% of peak FLOPs may be at 95% of peak bandwidth.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'PyTorch 的工具', en:'the PyTorch tool'}],
             ['#8fa3ac', {zh:'這個檔案裡的東西', en:'what this file calls it'}]]},
    {id:'jax', label:{zh:'在 JAX 裡', en:'in JAX'},
     stage:{zh:'編譯器先算給你看', en:'the compiler computes it for you'},
     view:VIEW, code:CODE_JAX, build:jaxFrames,
     idea:{zh:'`memory_analysis()` 給位元組、`cost_analysis()` 給 FLOPs，相除就是運算密度——**編譯器在你執行之前就算好的 roofline**。XLA 預設融合，所以逐元素的流量自己消失；而 tiling 住在編譯器裡，不夠好時降到 Pallas 自己選 block 大小，對抗的就是 tiling 那一頁的約束。',
           en:'`memory_analysis()` gives the bytes and `cost_analysis()` the FLOPs; divide them and you have the intensity - **a roofline the compiler computed before you ran anything**. XLA fuses by default so pointwise traffic disappears on its own, and the tiling lives in the compiler: when that is not enough you drop to Pallas and choose the block sizes against the constraint from the tiling tab.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'JAX 的工具', en:'the JAX tool'}],
             ['#8fa3ac', {zh:'這個檔案裡的東西', en:'what this file calls it'}]]}
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
