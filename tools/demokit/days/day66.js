// DAY: 66
// TITLE_ZH: HNSW：不看大部分資料就找到最近的向量
// TITLE_EN: HNSW: finding the nearest vectors without looking at most of them
// SUB_ZH: 暴力搜尋要讀過每一個向量，HNSW 只讀幾百個而且通常給一樣的答案。這一天從零把索引蓋出來——分層的圖、貪婪下降、ef 候選名單、鄰居挑選的啟發式——然後量三件真正決定值不值得的事：召回、距離計算次數，以及刪掉東西之後會發生什麼。順帶一個反直覺的實測：那個「分層」其實沒有帶來速度。
// SUB_EN: A brute-force search reads every vector; HNSW reads a few hundred and usually returns the same answer. This day builds the index from scratch - the layered graph, the greedy descent, the ef candidate list, the neighbour heuristic - and measures the three things that decide whether it was worth it: recall, distance computations, and what happens after you delete things. Including one counter-intuitive measurement: the hierarchy is not where the speed comes from.
// FOLDER: day%2066%20-%20hnsw
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
const CFG = {"docs": 3000, "dims": 48, "queries": 120, "M": 12, "efc": 60, "build": 12.5};
const HEAD = {"bf_dist": 3000, "hnsw_dist": 908, "recall": 0.947, "frac": 0.302};
const CURVE = [{"ef": 10, "dist": 319, "frac": 0.106, "recall": 0.629}, {"ef": 25, "dist": 567, "frac": 0.189, "recall": 0.849}, {"ef": 50, "dist": 908, "frac": 0.302, "recall": 0.947}, {"ef": 100, "dist": 1432, "frac": 0.477, "recall": 0.996}, {"ef": 200, "dist": 2071, "frac": 0.69, "recall": 1.0}];
const LAYERS = [{"layer": 0, "nodes": 3000, "share": 1.0, "expected": 3000.0}, {"layer": 1, "nodes": 260, "share": 0.0867, "expected": 250.0}, {"layer": 2, "nodes": 21, "share": 0.007, "expected": 20.8}, {"layer": 3, "nodes": 5, "share": 0.0017, "expected": 1.7}];
const FLAT = [{"name": "with the express lanes", "dist": 908, "recall": 0.947}, {"name": "bottom layer only, same entry point", "dist": 886, "recall": 0.947}, {"name": "bottom layer only, random entry point", "dist": 886, "recall": 0.947}];
const MS = [{"M": 6, "edges": 13.1, "link": 52, "vec": 192, "over": 0.27, "build": 1.3, "recall": 0.936}, {"M": 12, "edges": 24.9, "link": 100, "vec": 192, "over": 0.52, "build": 3.5, "recall": 0.996}, {"M": 20, "edges": 40.9, "link": 164, "vec": 192, "over": 0.85, "build": 8.8, "recall": 0.999}];
const HEUR = [{"name": "plain nearest-M", "recall": 0.754, "same": 0.455, "build": 1.0}, {"name": "heuristic (diverse)", "recall": 0.799, "same": 0.389, "build": 2.2}];
const DEL = [{"deleted": 0.1, "recall": 0.985, "dist": 757, "waste": 0.102, "rebuilt": false, "secs": 0}, {"deleted": 0.3, "recall": 0.974, "dist": 757, "waste": 0.298, "rebuilt": false, "secs": 0}, {"deleted": 0.3, "recall": 0.993, "dist": 666, "waste": 0, "rebuilt": true, "secs": 5.6}];
const SCALE = [{"docs": 500, "dist": 446, "frac": 0.891, "recall": 0.998}, {"docs": 1000, "dist": 644, "frac": 0.644, "recall": 0.993}, {"docs": 2000, "dist": 824, "frac": 0.412, "recall": 0.972}, {"docs": 3000, "dist": 908, "frac": 0.302, "recall": 0.947}];
const FAIL = {"ef": 5, "k": 10, "clamped": 0.629, "unclamped": 0.403, "short": 1.0};
const CIRC = {"bf": 0.947, "self": 1.0};
const NORM = {"recall": 0.25, "share": 0.25, "hits": 1.0};
const WALK = {"points": [[0.943, 0.511], [0.976, 0.081], [0.607, 0.376], [0.802, 0.175], [0.872, 0.544], [0.902, 0.477], [0.43, 0.789], [0.984, 0.37], [0.969, 0.929], [0.178, 0.609], [0.705, 0.943], [0.666, 0.133], [0.498, 0.494], [0.5, 0.959], [0.35, 0.224], [0.522, 0.641], [0.939, 0.582], [0.268, 0.93], [0.492, 0.676], [0.476, 0.217], [0.693, 0.771], [0.191, 0.46], [0.362, 0.171], [0.221, 0.963], [0.884, 0.382], [0.739, 0.042], [0.915, 0.537], [0.82, 0.276], [0.376, 0.348], [0.972, 0.429], [0.5, 0.956], [0.903, 0.395], [0.307, 0.807], [0.107, 0.392], [0.876, 0.918], [0.233, 0.079], [0.788, 0.622], [0.238, 0.506], [0.071, 0.7], [0.527, 0.705], [0.088, 0.645], [0.37, 0.499], [0.335, 0.634], [0.568, 0.617], [0.063, 0.225], [0.072, 0.383], [0.481, 0.112], [0.668, 0.487], [0.5, 0.597], [0.935, 0.589], [0.093, 0.146], [0.694, 0.323], [0.755, 0.511], [0.38, 0.739], [0.183, 0.285], [0.292, 0.415], [0.907, 0.813], [0.822, 0.631], [0.008, 0.286], [0.502, 0.898]], "query": [0.6200000047683716, 0.550000011920929], "layers": [{"layer": 0, "nodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59], "edges": [[0, 26], [0, 5], [0, 16], [0, 49], [0, 4], [0, 29], [0, 31], [0, 24], [1, 3], [1, 25], [1, 27], [1, 7], [1, 11], [1, 24], [1, 5], [2, 51], [2, 47], [2, 12], [2, 19], [2, 28], [2, 27], [2, 11], [2, 24], [3, 27], [3, 11], [3, 25], [3, 51], [3, 24], [3, 7], [4, 26], [4, 57], [4, 52], [4, 5], [4, 16], [4, 49], [4, 36], [5, 31], [5, 26], [5, 29], [5, 24], [5, 16], [5, 49], [6, 53], [6, 39], [6, 59], [6, 17], [6, 32], [6, 18], [6, 15], [6, 30], [7, 29], [7, 27], [7, 31], [7, 24], [7, 26], [8, 34], [8, 56], [8, 10], [8, 20], [8, 49], [8, 16], [9, 40], [9, 37], [9, 42], [9, 38], [9, 21], [9, 41], [9, 33], [9, 32], [10, 20], [10, 34], [10, 13], [10, 30], [10, 59], [10, 56], [11, 25], [11, 46], [11, 51], [11, 19], [11, 27], [11, 22], [12, 48], [12, 41], [12, 43], [12, 15], [12, 47], [12, 18], [12, 28], [13, 30], [13, 59], [13, 17], [13, 32], [13, 39], [13, 20], [14, 22], [14, 28], [14, 54], [14, 19], [14, 46], [14, 35], [14, 55], [14, 50], [15, 18], [15, 48], [15, 43], [15, 39], [15, 53], [15, 42], [16, 49], [16, 26], [16, 57], [16, 29], [16, 36], [17, 23], [17, 32], [17, 30], [17, 53], [17, 38], [18, 39], [18, 53], [18, 48], [18, 43], [18, 42], [19, 46], [19, 22], [19, 28], [19, 51], [20, 36], [20, 39], [20, 56], [20, 43], [20, 59], [21, 37], [21, 33], [21, 55], [21, 45], [21, 54], [21, 41], [22, 35], [22, 46], [22, 28], [22, 54], [22, 50], [22, 44], [23, 32], [23, 30], [23, 38], [23, 40], [24, 31], [24, 27], [24, 52], [24, 29], [24, 26], [25, 27], [25, 46], [26, 49], [26, 29], [26, 57], [26, 31], [27, 51], [27, 31], [27, 29], [28, 55], [28, 41], [28, 54], [29, 31], [30, 59], [30, 32], [30, 39], [32, 53], [32, 42], [32, 59], [33, 45], [33, 54], [33, 58], [33, 37], [33, 44], [33, 55], [34, 56], [35, 50], [35, 44], [35, 46], [35, 58], [36, 57], [36, 52], [36, 49], [36, 47], [37, 55], [37, 41], [37, 42], [37, 40], [37, 45], [38, 40], [39, 43], [39, 48], [39, 53], [39, 59], [40, 45], [41, 55], [41, 42], [41, 48], [42, 53], [42, 48], [43, 47], [43, 48], [44, 58], [44, 50], [44, 54], [44, 45], [45, 58], [45, 54], [45, 50], [47, 52], [47, 51], [47, 48], [48, 53], [49, 57], [50, 58], [50, 54], [52, 57], [53, 59], [54, 55], [54, 58], [56, 57]]}, {"layer": 1, "nodes": [0, 1, 4, 7, 9, 20, 28, 31, 34, 39, 47, 51, 59], "edges": [[0, 4], [0, 31], [0, 7], [0, 20], [1, 7], [1, 4], [1, 28], [4, 47], [4, 31], [4, 7], [7, 31], [9, 28], [9, 39], [9, 59], [9, 20], [20, 39], [20, 34], [20, 59], [28, 51], [28, 47], [28, 39], [31, 51], [34, 59], [39, 59], [39, 47], [47, 51]]}, {"layer": 2, "nodes": [20, 28], "edges": [[20, 28]]}, {"layer": 3, "nodes": [20], "edges": []}, {"layer": 4, "nodes": [20], "edges": []}], "hops": [{"layer": 4, "src": 20, "dst": 20}, {"layer": 3, "src": 20, "dst": 20}, {"layer": 2, "src": 20, "dst": 20}, {"layer": 1, "src": 20, "dst": 47}], "expanded": [47, 43, 48, 15, 12, 52, 2, 18, 39, 36, 57, 20, 51], "top": [47, 43, 48, 15, 12], "entry": 20};

/* ---------------------------------------------------------------- shared */
const R3 = x => Number(x).toFixed(3);
const PCT = x => Math.round(x * 100 - 1e-9) + '%';
function rowBars(sh, rows, o){
  const x0 = o.x0 || 3.30, w = o.w || 3.6, y0 = o.y0 || 1.60, dy = o.dy || 0.95;
  const top = o.top || Math.max.apply(null, rows.map(r => r.value));
  rows.forEach((r, i) => {
    const y = y0 + i * dy;
    sh.push(fitT(x0 - .15, y + .32, r.label, {anchor:'end', c:COL.tealL, fs:o.fs || .26}));
    const bw = Math.max(.12, w * r.value / top);
    sh.push(S.r(x0, y, bw, .58, r.sty || 'ok', null));
    sh.push(fitT(x0 + bw + .14, y + .32, r.right, {anchor:'start',
            c:r.sty === 'bad' ? COL.red : COL.tealL, fs:o.fs || .26}));
    if (r.far) sh.push(fitT(9.5, y + .32, r.far, {anchor:'end', c:COL.grey, fs:.25}));
  });
  return sh;
}

/* ================================================================ tab 1: the walk */
const CODE_WALK = [
  'def search(q, k, ef):',
  '    ep = entry_point',
  '    for layer in range(top, 0, -1):      # express lanes, one hop at a time',
  '        ep = greedy_step(q, ep, layer)   # ef = 1 up here',
  '',
  '    cands = search_layer(q, ep, ef, 0)   # bottom layer: keep ef alive',
  '    return sorted(cands)[:k]'
];
function walkStage(phase){
  const sh = [];
  const P = WALK.points, q = WALK.query;
  const place = (p, ox, oy, s) => [ox + p[0] * s, oy + p[1] * s];
  const layersShown = [Math.min(2, WALK.layers.length - 1), 1, 0];
  const S0 = 2.45, origins = [[.60, 1.35], [3.65, 1.35], [6.70, 1.35]];
  layersShown.forEach((layer, panel) => {
    const [ox, oy] = origins[panel];
    const g = WALK.layers[layer];
    const active = panel <= phase;
    sh.push(fitT(ox + S0 / 2, oy - .25, 'layer ' + layer + ' - ' + g.nodes.length +
                 ' of ' + WALK.layers[0].nodes.length,
                 {c:active ? COL.tealL : COL.grey, fs:.26}));
    g.edges.forEach(e => {
      const a = place(P[e[0]], ox, oy, S0), b = place(P[e[1]], ox, oy, S0);
      sh.push(S.e(a[0], a[1], b[0], b[1], {s:active ? 'soft' : 'ghost', w:.025,
                                           arrow:false}));
    });
    g.nodes.forEach(n => {
      const p = place(P[n], ox, oy, S0);
      const hit = layer === 0 && WALK.top.indexOf(n) >= 0 && phase >= 2;
      const seen = layer === 0 && phase >= 2 && WALK.expanded.indexOf(n) >= 0;
      sh.push(S.c(p[0], p[1], hit ? .11 : .055,
                  hit ? 'ok' : (seen ? 'hot' : (active ? 'act' : 'ghost')), null));
    });
    const qp = place(q, ox, oy, S0);
    sh.push(S.c(qp[0], qp[1], .10, 'bad', null));
    if (active) WALK.hops.forEach(h => {
      if (h.layer !== layer) return;
      const a = place(P[h.src], ox, oy, S0), b = place(P[h.dst], ox, oy, S0);
      sh.push(S.e(a[0], a[1], b[0], b[1], {s:'bad', w:.06}));
    });
  });
  return sh;
}
function walkFrames(){
  const F = new Frames();
  const msgs = [
    {zh:'索引是**好幾層圖**：最底層放全部的點，每往上一層只留大約 1/M。搜尋從最上層的入口點開始。',
     en:'The index is **a stack of graphs**: the bottom layer holds every point and each layer above keeps about 1/M of the one below. The search starts at the entry point on the top layer.'},
    {zh:'上面幾層只做一件事：**每次往離查詢更近的鄰居走一步**，走不動了就掉到下一層。這一段的 ef 是 1——它只是在找一個好的起點。',
     en:'The upper layers do exactly one thing: **step to whichever neighbour is closer to the query**, and drop a level when no neighbour improves. ef is 1 up here - this part is only looking for a good place to start.'},
    {zh:'到了最底層才真正展開：同時保留 ef 個候選、把它們的鄰居都算一遍。這一次只算了 ' + WALK.expanded.length + ' 個點的距離（總共 ' + WALK.layers[0].nodes.length + ' 個）。在真正的語料上是 ' + HEAD.hnsw_dist + ' / ' + HEAD.bf_dist + '（' + PCT(HEAD.frac) + '），召回 ' + R3(HEAD.recall) + '。',
     en:'Only at the bottom does it really expand: keep ef candidates alive and compute distances to all of their neighbours. This walk touched ' + WALK.expanded.length + ' of ' + WALK.layers[0].nodes.length + ' points. On the real corpus that is ' + HEAD.hnsw_dist + ' of ' + HEAD.bf_dist + ' (' + PCT(HEAD.frac) + ') for recall ' + R3(HEAD.recall) + '.'}
  ];
  [0, 1, 2].forEach(p => {
    F.push({shapes:caps(walkStage(p), {cap:p === 2
              ? {zh:'底層：ef 個候選一起活著', en:'the bottom layer: ef candidates kept alive'}
              : {zh:'上層：一次一步的貪婪下降', en:'the upper layers: one greedy hop at a time'}}),
            line:[1, 3, 5][p], view:VIEW,
            panels:[{lbl:'distances', chips:[chip(String(HEAD.hnsw_dist), 'ok')]},
                    {lbl:'brute force', chips:[chip(String(HEAD.bf_dist), 'dim')]},
                    {lbl:'recall@10', chips:[chip(R3(HEAD.recall), 'act')]}],
            msg:msgs[p]});
  });
  return F.list;
}

/* ================================================================ tab 2: layers */
const CODE_LAYERS = [
  '# each point picks a level once, and keeps it forever',
  'level = floor(-ln(uniform()) * (1 / ln(M)))',
  '',
  '# so layer l holds about N / M**l points:',
  '#   layer 0 = everything, layer 1 = 1/M, layer 2 = 1/M**2 ...'
];
function layerFrames(){
  const F = new Frames();
  const sh = [];
  sh.push(fitT(4.9, .92, {zh:'每往上一層，只留下約 1/M 的點',
                          en:'each layer up keeps about 1/M of the points'},
               {c:COL.tealL, fs:.31}));
  const top = LAYERS[0].nodes;
  LAYERS.forEach((r, i) => {
    const y = 1.55 + i * 1.00;
    sh.push(fitT(2.15, y + .32, 'layer ' + r.layer, {anchor:'end', c:COL.tealL, fs:.28}));
    const w = Math.max(.10, 5.2 * r.nodes / top);
    sh.push(S.r(2.30, y, w, .58, i === 0 ? 'act' : 'ok', null));
    sh.push(fitT(2.42 + w, y + .32, r.nodes.toLocaleString() + ' points', {anchor:'start',
            c:COL.tealL, fs:.26}));
    sh.push(fitT(9.5, y + .32, 'expected ' + r.expected.toLocaleString(),
                 {anchor:'end', c:COL.grey, fs:.24}));
  });
  F.push({shapes:caps(sh, {cap:{zh:'N = ' + CFG.docs.toLocaleString() + '、M = ' + CFG.M,
                                en:'N = ' + CFG.docs.toLocaleString() + ', M = ' + CFG.M}}),
          line:1, view:VIEW,
          panels:[{lbl:'layers', chips:[chip(String(LAYERS.length), 'ok')]},
                  {lbl:'top layer', chips:[chip(LAYERS[LAYERS.length - 1].nodes + ' points', 'act')]}],
          msg:{zh:'每個點插入時抽一次層數，抽到的分布讓**每一層大約是下一層的 1/M**（實測 ' + LAYERS.map(r => r.nodes).join(' → ') + '，理論值 ' + LAYERS.map(r => Math.round(r.expected)).join(' → ') + '）。這就是 Day 31 skip list 的同一招：在底下那層「全部的資料」之上疊幾層稀疏的索引，只是這裡的「下一個元素」沒有定義，所以用圖代替串列。',
               en:'Every point draws a level once when it is inserted, from a distribution that makes **each layer about 1/M of the one below** (measured ' + LAYERS.map(r => r.nodes).join(' → ') + ' against an expected ' + LAYERS.map(r => Math.round(r.expected)).join(' → ') + '). That is the Day 31 skip list again: a few sparse index levels above the layer that holds everything - except here "the next element" is undefined, so a graph stands in for the linked list.'}});
  return F.list;
}

/* ================================================================ tab 3: ef */
const CODE_EF = [
  'cands = search_layer(q, entry, ef, layer=0)',
  '',
  '# ef is how many candidates stay alive during the walk:',
  '#   bigger ef -> more neighbours expanded',
  '#             -> more distance computations',
  '#             -> better chance of escaping a local pocket'
];
function efFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'ef：唯一的查詢期旋鈕', en:'ef: the only query-time knob'},
                 {c:COL.tealL, fs:.31}));
    const rows = [];
    CURVE.forEach((r, i) => {
      if (i > upto) return;
      rows.push({label:'ef ' + r.ef, value:r.dist, right:r.dist + ' distances',
                 far:'recall ' + R3(r.recall), sty:r.ef === 50 ? 'hot' : 'ok'});
    });
    rows.push({label:{zh:'暴力搜尋', en:'brute force'}, value:HEAD.bf_dist,
               right:HEAD.bf_dist + ' distances', far:'recall 1.000', sty:'bad'});
    return rowBars(sh, rows, {y0:1.45, dy:.82, w:3.4, top:HEAD.bf_dist});
  };
  [1, 3, CURVE.length - 1].forEach((u, i) => {
    F.push({shapes:caps(stage(u), {cap:{zh:'同一個索引，只改 ef', en:'the same index, only ef changes'}}),
            line:[0, 2, 5][i], view:VIEW,
            panels:[{lbl:'ef ' + CURVE[u].ef, chips:[chip(R3(CURVE[u].recall), 'ok'),
                     chip(CURVE[u].dist + ' dist', 'act')]}],
            msg:[
              {zh:'ef = ' + CURVE[0].ef + ' 時只算了 ' + CURVE[0].dist + ' 次距離（語料的 ' + PCT(CURVE[0].frac) + '），但召回只有 ' + R3(CURVE[0].recall) + '——貪婪走路很容易卡在一個小窪地裡。',
               en:'At ef = ' + CURVE[0].ef + ' the walk computes ' + CURVE[0].dist + ' distances (' + PCT(CURVE[0].frac) + ' of the corpus) and recall is only ' + R3(CURVE[0].recall) + ': a greedy walk gets stuck in a local pocket very easily.'},
              {zh:'把候選名單開大，走得出去的機會就變高：ef ' + CURVE[0].ef + ' → ' + CURVE[2].ef + ' 讓召回 ' + R3(CURVE[0].recall) + ' → ' + R3(CURVE[2].recall) + '，代價是距離計算 ' + CURVE[0].dist + ' → ' + CURVE[2].dist + '。**召回是候選名單買來的，不是運氣。**',
               en:'A longer candidate list is what lets the walk escape: ef ' + CURVE[0].ef + ' → ' + CURVE[2].ef + ' takes recall from ' + R3(CURVE[0].recall) + ' to ' + R3(CURVE[2].recall) + ' for ' + CURVE[0].dist + ' → ' + CURVE[2].dist + ' distance computations. **Recall is bought with the candidate list, not with luck.**'},
              {zh:'再往上就是遞減報酬：ef ' + CURVE[CURVE.length - 1].ef + ' 幾乎是滿分，但已經要掃 ' + PCT(CURVE[CURVE.length - 1].frac) + ' 的語料——接近暴力搜尋。**索引的價值就在這條曲線的膝點附近**，而膝點會隨語料大小移動，所以 ef 是要定期重調的。',
               en:'Past that it is diminishing returns: ef ' + CURVE[CURVE.length - 1].ef + ' is nearly perfect but already reads ' + PCT(CURVE[CURVE.length - 1].frac) + ' of the corpus, which is most of the way back to brute force. **The value of the index lives near the knee of this curve**, and the knee moves as the corpus grows - which is why ef is a setting to re-tune, not to set once.'}
            ][i]});
  });
  return F.list;
}

/* ================================================================ tab 4: the flat graph */
const CODE_FLAT = [
  '# the experiment: throw the upper layers away',
  'cands = search_layer(q, entry_point, ef, layer=0)     # same entry',
  'cands = search_layer(q, random_point, ef, layer=0)    # random entry',
  '',
  '# published result (ECIR 2026): on high-dimensional data a flat',
  '# graph matches HNSW and saves 38-39% of peak build memory at 100M'
];
function flatFrames(){
  const F = new Frames();
  const sh = [];
  sh.push(fitT(4.9, .92, {zh:'把上層拿掉，結果一模一樣',
                          en:'remove the upper layers and nothing changes'},
               {c:COL.tealL, fs:.31}));
  rowBars(sh, FLAT.map((r, i) => ({label:r.name, value:r.dist, right:r.dist + ' distances',
                                   far:'recall ' + R3(r.recall), sty:i === 0 ? 'act' : 'ok'})),
          {y0:1.75, dy:1.05, x0:4.30, w:2.8, top:Math.max.apply(null, FLAT.map(r => r.dist))});
  F.push({shapes:caps(sh, {cap:{zh:'同一個索引，只是不走上層',
                                en:'the same index, searched without the express lanes'}}),
          line:1, view:VIEW,
          panels:[{lbl:'with layers', chips:[chip(FLAT[0].dist + ' dist', 'act')]},
                  {lbl:'flat', chips:[chip(FLAT[2].dist + ' dist', 'ok')]}],
          msg:{zh:'**這是這一天最反直覺的一條**：只搜最底層、甚至從隨機一點出發，距離計算 ' + FLAT[0].dist + ' → ' + FLAT[2].dist + '、召回 ' + R3(FLAT[0].recall) + ' → ' + R3(FLAT[2].recall) + '，等於沒差。不是實作寫壞了——這重現了一篇已發表的結果（ECIR 2026）：在高維資料上階層沒有可量測的好處，而拿掉它在一億向量的規模可以省 38–39% 的建置記憶體。他們的解釋是 **hub highway**：高維空間裡少數幾個點是「大家的鄰居」，底層圖自己就已經有快速道路了。更早的研究則發現階層只在大約 32 維以下才有幫助。',
               en:'**This is the most counter-intuitive measurement of the day.** Searching the bottom layer only - even starting from a random point - moves the work from ' + FLAT[0].dist + ' to ' + FLAT[2].dist + ' distances and the recall from ' + R3(FLAT[0].recall) + ' to ' + R3(FLAT[2].recall) + '. That is not a broken implementation: it reproduces a published result (ECIR 2026) which found the hierarchy gives no measurable benefit on high-dimensional data and that dropping it saves 38-39% of peak build memory at 100M vectors. Their explanation is the **hub highway**: in high dimensions a few points are everybody\'s neighbour, so the bottom graph already contains the express lanes. Earlier work found the layers only help below about 32 dimensions.'}});
  return F.list;
}

/* ================================================================ tab 5: M */
const CODE_M = [
  '# M is how many neighbours each point keeps (2M on the bottom layer)',
  'ix = HNSW(dim, M=12, ef_construction=60)',
  '',
  '# the links are stored next to the vectors:',
  '#   bytes per point = dim * 4  +  edges_per_node * 4'
];
function mFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'M 同時是記憶體旋鈕與建置時間旋鈕',
                            en:'M is the memory knob and the build-time knob at once'},
                 {c:COL.tealL, fs:.30}));
    MS.forEach((r, i) => {
      if (i > upto) return;
      const y = 1.65 + i * 1.15;
      sh.push(fitT(1.75, y + .33, 'M = ' + r.M, {anchor:'end', c:COL.tealL, fs:.28}));
      sh.push(S.r(1.90, y, 1.7, .62, 'idle', 'vector', {fs:.24}));
      const wl = Math.max(.15, 1.7 * r.over);
      sh.push(S.r(3.60, y, wl, .62, 'hot', wl > .8 ? 'links' : null, {fs:.24}));
      sh.push(fitT(3.72 + wl, y + .33, r.vec + ' + ' + r.link + ' bytes',
                   {anchor:'start', c:COL.grey, fs:.25}));
      sh.push(fitT(8.30, y + .33, r.edges + ' edges/node', {anchor:'end', c:COL.grey, fs:.25}));
      sh.push(fitT(9.55, y + .33, R3(r.recall), {anchor:'end', c:COL.tealL, fs:.26}));
    });
    return sh;
  };
  [0, MS.length - 1].forEach((u, i) => {
    F.push({shapes:caps(stage(u), {cap:{zh:'1,000 個點，ef = 50 時的召回',
                                        en:'1,000 points, recall measured at ef = 50'}}),
            line:1, view:VIEW,
            panels:[{lbl:'M = ' + MS[u].M, chips:[chip(R3(MS[u].recall), 'ok'),
                     chip(MS[u].link + ' B links', 'hot')]}],
            msg:[
              {zh:'M 是每個點保留幾個鄰居（最底層放 2M 個）。M = ' + MS[0].M + ' 時每個點只多 ' + MS[0].link + ' 個位元組的連結，是向量本身的 ' + MS[0].over + ' 倍，召回 ' + R3(MS[0].recall) + '。',
               en:'M is how many neighbours each point keeps (2M on the bottom layer). At M = ' + MS[0].M + ' the links add ' + MS[0].link + ' bytes per point - ' + MS[0].over + 'x the vector itself - for recall ' + R3(MS[0].recall) + '.'},
              {zh:'把 M 開大，召回會上去但很快飽和（' + MS.map(r => R3(r.recall)).join(' → ') + '），而**連結的記憶體與建置時間不會飽和**：M = ' + MS[MS.length - 1].M + ' 時連結已經跟向量本身一樣貴（' + MS[MS.length - 1].over + ' 倍），建置也一路變慢（每插入一個點都要跑一次 ef_construction 規模的搜尋）。所以要問的從來不是「最好的 M 是多少」，而是「**達到我要的召回所需的最小 M**」。',
               en:'Raising M raises recall but saturates quickly (' + MS.map(r => R3(r.recall)).join(' → ') + '), while **the link memory and the build time do not**: at M = ' + MS[MS.length - 1].M + ' the links cost as much as the vectors themselves (' + MS[MS.length - 1].over + 'x) and the build - an ef_construction-sized search for every point inserted - keeps getting slower. So the question is never "what is the best M" but "**what is the smallest M that reaches the recall I need**".'}
            ][i]});
  });
  return F.list;
}

/* ================================================================ tab 6: heuristic */
const CODE_HEUR = [
  '# plain: keep the m nearest candidates',
  'return sorted(candidates)[:m]',
  '',
  '# heuristic: keep a candidate only if it is closer to the new point',
  '# than to anything already kept - that is what saves the long edges',
  'for d, c in sorted(candidates):',
  '    if all(d < dist(c, kept) for kept in chosen): chosen.append(c)'
];
function heurFrames(){
  const F = new Frames();
  const sh = [];
  sh.push(fitT(4.9, .92, {zh:'挑鄰居的方式決定圖走不走得出去',
                          en:'how neighbours are chosen decides whether the graph can be left'},
               {c:COL.tealL, fs:.30}));
  rowBars(sh, HEUR.map((r, i) => ({label:r.name, value:r.recall,
                                   right:'recall ' + R3(r.recall),
                                   far:PCT(r.same) + ' of edges stay inside the cluster',
                                   sty:i === 1 ? 'ok' : 'bad'})),
          {y0:2.00, dy:1.30, x0:3.10, w:3.0, top:1});
    F.push({shapes:caps(sh, {cap:{zh:'12 個群集、M = 6、ef = 25 的設定',
                                en:'12 clusters, M = 6, ef = 25'}}),
          line:4, view:VIEW,
          panels:[{lbl:'plain', chips:[chip(R3(HEUR[0].recall), 'bad')]},
                  {lbl:'heuristic', chips:[chip(R3(HEUR[1].recall), 'ok')]}],
          msg:{zh:'只留「最近的 m 個」會讓鄰居名單整串都是自己群集裡的點，於是**離開群集的那些邊就不見了**，貪婪走路被關在裡面。啟發式的規則是：一個候選只有在「離新點比離任何已選的鄰居都近」時才留下——這樣長程邊才活得下來。這裡值 ' + Math.round(100 * (HEUR[1].recall - HEUR[0].recall)) + ' 個百分點的召回，代價是比較慢的插入（每個候選都要再跟已經選上的那些比一次）。M 開大之後兩者會收斂——**啟發式是鄰居預算很緊時用來把召回買回來的手段**。',
               en:'Keeping simply the m nearest fills a point\'s neighbour list with its own cluster, so **the edges that leave the cluster disappear** and the greedy walk is locked inside. The heuristic keeps a candidate only if it is closer to the new point than to anything already chosen, which is what preserves the long-range links. Here that is worth ' + Math.round(100 * (HEUR[1].recall - HEUR[0].recall)) + ' points of recall at the price of a slower insert, since every candidate now has to be compared against the ones already kept. With a generous M the two converge - **the heuristic is how you buy recall back when the neighbour budget is tight**.'}});
  return F.list;
}

/* ================================================================ tab 7: deletions */
const CODE_DEL = [
  'def delete(ids):',
  '    tombstones.update(ids)        # the point stays in the graph',
  '',
  '# at query time it is filtered out of the results,',
  '# but it is still visited, and still costs a distance computation'
];
function delFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'刪除＝墓碑：點還在圖裡，只是不會被回傳',
                            en:'a deletion is a tombstone: the point stays in the graph'},
                 {c:COL.tealL, fs:.30}));
    const top = Math.max.apply(null, DEL.map(r => r.dist));
    DEL.forEach((r, i) => {
      if (i > upto) return;
      const y = 1.70 + i * 1.15;
      const name = r.rebuilt ? 'rebuilt' : PCT(r.deleted) + ' deleted';
      sh.push(fitT(2.75, y + .33, name, {anchor:'end', c:COL.tealL, fs:.27}));
      const w = 4.0 * r.dist / top;
      sh.push(S.r(2.90, y, w * (1 - r.waste), .62, 'ok', null));
      if (r.waste) sh.push(S.r(2.90 + w * (1 - r.waste), y, w * r.waste, .62, 'bad', null));
      sh.push(fitT(3.02 + w, y + .33, r.dist + ' distances', {anchor:'start', c:COL.grey, fs:.25}));
      sh.push(fitT(9.55, y + .33, 'recall ' + R3(r.recall), {anchor:'end', c:COL.tealL, fs:.26}));
    });
    return sh;
  };
  [1, DEL.length - 1].forEach((u, i) => {
    F.push({shapes:caps(stage(u), {cap:{zh:'紅色＝花在永遠回傳不了的點上的距離計算',
                                        en:'red: distance computations spent on points that cannot be returned'}}),
            line:1, view:VIEW,
            panels:[{lbl:'30% deleted', chips:[chip(PCT(DEL[1].waste) + ' wasted', 'bad')]},
                    {lbl:'rebuilt', chips:[chip(DEL[2].dist + ' dist', 'ok')]}],
            msg:[
              {zh:'圖索引刪一個點，正確做法要去修每一個指向它的鄰居名單——沒有人在線上做這件事。所以實務上是**墓碑**：點留在圖裡當踏腳石，只是從結果裡濾掉。代價是最後那欄：刪掉 ' + PCT(DEL[1].deleted) + ' 之後，每個查詢有 ' + PCT(DEL[1].waste) + ' 的距離計算是花在永遠不能回傳的點上，而每查詢的工作量一點都沒變（' + DEL[0].dist + '）。',
               en:'Deleting a point from a graph index properly means repairing every neighbour list that points at it, and nobody does that online. So in practice it is a **tombstone**: the point stays in the graph as a stepping stone and is filtered out of the results. The cost is the last column: after ' + PCT(DEL[1].deleted) + ' is deleted, ' + PCT(DEL[1].waste) + ' of every query\'s distance computations go to points that can never be returned - and the work per query has not moved (' + DEL[0].dist + ').'},
              {zh:'重建就把這些浪費掃掉：距離計算 ' + DEL[1].dist + ' → ' + DEL[2].dist + '、召回回到 ' + R3(DEL[2].recall) + '。**所以每一個提供「刪除」的向量資料庫，其實是在提供一個重整排程**——唯一的問題是誰來跑它、什麼時候跑。',
               en:'Rebuilding sweeps the waste away: distance computations ' + DEL[1].dist + ' → ' + DEL[2].dist + ' and recall back to ' + R3(DEL[2].recall) + '. **Every vector database that offers deletion is really offering a compaction schedule** - the only question is who runs it and when.'}
            ][i]});
  });
  return F.list;
}

/* ================================================================ tab 8: scaling */
const CODE_SCALE = [
  '# brute force: every vector, every query',
  'scores = q @ data.T                 # N distance computations',
  '',
  '# HNSW: the walk length grows like log N, not like N',
  'ids = index.search(q, k=10, ef=50)  # a few hundred, whatever N is'
];
function scaleFrames(){
  const F = new Frames();
  const sh = [];
  sh.push(fitT(4.9, .92, {zh:'語料變六倍，工作量只變兩倍',
                          en:'six times the data, twice the work'}, {c:COL.tealL, fs:.31}));
  const top = SCALE[SCALE.length - 1].docs;
  SCALE.forEach((r, i) => {
    const y = 1.55 + i * 1.05;
    sh.push(fitT(2.35, y + .33, r.docs.toLocaleString() + ' vectors',
                 {anchor:'end', c:COL.tealL, fs:.26}));
    const wb = 5.0 * r.docs / top;
    sh.push(S.r(2.50, y, wb, .28, 'bad', null));
    const wh = 5.0 * r.dist / top;
    sh.push(S.r(2.50, y + .32, Math.max(.08, wh), .28, 'ok', null));
    sh.push(fitT(2.62 + Math.max(wb, wh), y + .33,
                 r.docs.toLocaleString() + '  vs  ' + r.dist, {anchor:'start',
                 c:COL.grey, fs:.25}));
    sh.push(fitT(9.55, y + .33, 'recall ' + R3(r.recall), {anchor:'end', c:COL.tealL, fs:.25}));
  });
  F.push({shapes:caps(sh, {cap:{zh:'紅＝暴力搜尋（每個向量），藍＝HNSW（ef 50）',
                                en:'red = brute force (every vector), teal = HNSW at ef 50'}}),
          line:4, view:VIEW,
          panels:[{lbl:'500', chips:[chip(PCT(SCALE[0].frac), 'bad')]},
                  {lbl:CFG.docs.toLocaleString(), chips:[chip(PCT(SCALE[SCALE.length - 1].frac), 'ok')]}],
          msg:{zh:'語料從 ' + SCALE[0].docs + ' 變成 ' + SCALE[SCALE.length - 1].docs.toLocaleString() + '（六倍），暴力搜尋的距離計算當然也是六倍，而 HNSW 從 ' + SCALE[0].dist + ' 只長到 ' + SCALE[SCALE.length - 1].dist + '——**每查詢的工作量大致隨語料的對數成長**，這就是索引的全部價值。注意召回同時從 ' + R3(SCALE[0].recall) + ' 慢慢掉到 ' + R3(SCALE[SCALE.length - 1].recall) + '：**語料長大之後 ef 要重調**，不是只有資料換了才要調。',
               en:'The corpus goes from ' + SCALE[0].docs + ' to ' + SCALE[SCALE.length - 1].docs.toLocaleString() + ' - six times - so brute force does six times the work, while HNSW goes from ' + SCALE[0].dist + ' to only ' + SCALE[SCALE.length - 1].dist + ' distance computations. **The work per query grows roughly like the logarithm of the corpus**, and that is the entire value of the index. Notice the recall drifting down from ' + R3(SCALE[0].recall) + ' to ' + R3(SCALE[SCALE.length - 1].recall) + ' at the same time: **ef has to be re-tuned when a corpus grows**, not only when it changes.'}});
  return F.list;
}

/* ================================================================ tab 9: failures */
const CODE_FAIL = [
  'ix.set_ef(5); ix.knn_query(q, k=10)   # 1. ef below k',
  '',
  'ix = hnswlib.Index(space="ip", ...)   # 2. inner product,',
  'ix.add_items(unnormalised_vectors)    #    vectors not normalised',
  '',
  'gold = ix.knn_query(q, k=10)          # 3. ground truth from the index itself'
];
function failFrames(){
  const F = new Frames();
  const mk = (which) => {
    const sh = [];
    if (which === 0){
      sh.push(fitT(4.9, .92, {zh:'1. ef 比 k 小', en:'1. ef smaller than k'},
                   {c:COL.tealL, fs:.31}));
      rowBars(sh, [
        {label:{zh:'ef = ' + FAIL.ef + '（原樣）', en:'ef = ' + FAIL.ef + ', as asked'},
         value:FAIL.unclamped, right:'recall ' + R3(FAIL.unclamped), sty:'bad',
         far:PCT(FAIL.short) + ' of queries returned fewer than k'},
        {label:{zh:'ef 被夾到 k', en:'ef clamped up to k'}, value:FAIL.clamped,
         right:'recall ' + R3(FAIL.clamped), sty:'hot'},
        {label:{zh:'ef = 50', en:'ef = 50'}, value:HEAD.recall,
         right:'recall ' + R3(HEAD.recall), sty:'ok'}],
        {y0:1.90, dy:1.15, x0:3.40, w:3.2, top:1});
    } else if (which === 1){
      sh.push(fitT(4.9, .92, {zh:'2. 用內積但沒有正規化',
                              en:'2. inner product on unnormalised vectors'},
                   {c:COL.tealL, fs:.31}));
      rowBars(sh, [
        {label:{zh:'正確（cosine）', en:'cosine, as intended'}, value:HEAD.recall,
         right:'recall ' + R3(HEAD.recall), sty:'ok'},
        {label:{zh:'其中 ' + PCT(NORM.share) + ' 的向量被放大 3 倍',
                en:PCT(NORM.share) + ' of vectors scaled 3x'}, value:NORM.recall,
         right:'recall ' + R3(NORM.recall), sty:'bad',
         far:PCT(NORM.hits) + ' of the hits come from the scaled group'}],
        {y0:2.20, dy:1.30, x0:3.70, w:3.0, top:1});
    } else {
      sh.push(fitT(4.9, .92, {zh:'3. 用索引自己當答案來評分',
                              en:'3. grading the index against itself'},
                   {c:COL.tealL, fs:.31}));
      rowBars(sh, [
        {label:{zh:'對照暴力搜尋', en:'against brute force'}, value:CIRC.bf,
         right:R3(CIRC.bf), sty:'ok'},
        {label:{zh:'對照它自己', en:'against its own output'}, value:CIRC.self,
         right:R3(CIRC.self), sty:'bad'}],
        {y0:2.20, dy:1.30, x0:3.70, w:3.0, top:1});
    }
    return sh;
  };
  const msgs = [
    {zh:'ef 是走路時還活著的候選數量，**它同時是答案數量的上限**：ef = ' + FAIL.ef + '、k = ' + FAIL.k + ' 時有 ' + PCT(FAIL.short) + ' 的查詢根本回不滿 k 筆（召回 ' + R3(FAIL.unclamped) + '）。函式庫通常會偷偷把 ef 夾到 k，於是你不會看到錯誤，只會看到 ' + R3(FAIL.clamped) + ' 的召回——**ef 是答案品質的地板，不是效能設定**。',
     en:'ef is how many candidates stay alive, and **it is also a ceiling on how many answers can exist**: at ef = ' + FAIL.ef + ' with k = ' + FAIL.k + ', ' + PCT(FAIL.short) + ' of queries cannot even return k results (recall ' + R3(FAIL.unclamped) + '). Libraries quietly clamp ef up to k, so you never see an error - only a recall of ' + R3(FAIL.clamped) + '. **ef is a floor on answer quality, not a performance setting.**'},
    {zh:'把 ' + PCT(NORM.share) + ' 的向量乘以 3 之後用內積搜尋：對照真正的 cosine 答案，召回只剩 ' + R3(NORM.recall) + '，而且**前十名有 ' + PCT(NORM.hits) + ' 來自那群被放大的向量**。索引沒有壞——它在回答另一個問題（「誰的投影最大」），而且不會有任何訊息告訴你。',
     en:'Scale ' + PCT(NORM.share) + ' of the vectors by three and search with inner product: against the true cosine answer the recall is ' + R3(NORM.recall) + ', and **' + PCT(NORM.hits) + ' of the top ten come from the scaled group**. The index is not broken - it is answering a different question ("whose projection is largest"), and nothing says so.'},
    {zh:'最後一個是方法上的：如果你的 ground truth 是用同一個索引產生的，召回永遠是 ' + R3(CIRC.self) + '。真正的數字要跟**暴力搜尋**比（這裡 ' + R3(CIRC.bf) + '）。近似索引的評估，一定要有一個不近似的對照組。',
     en:'The last one is methodological: if the ground truth is produced by the same index, recall is always ' + R3(CIRC.self) + '. The real number comes from comparing against **brute force** (' + R3(CIRC.bf) + ' here). Evaluating an approximate index always needs a non-approximate reference.'}
  ];
  [0, 1, 2].forEach(i => {
    F.push({shapes:caps(mk(i), {cap:{zh:'三個都不會報錯', en:'none of the three raises an error'}}),
            line:[0, 3, 5][i], view:VIEW,
            panels:[{lbl:'what it looks like', chips:[chip('fine', 'dim')]},
                    {lbl:'what it is', chips:[chip('wrong', 'bad')]}],
            msg:msgs[i]});
  });
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'HNSW 分層近鄰圖', en:'HNSW, a layered proximity graph'},
  sub: {zh:'暴力搜尋要讀過每一個向量，這個索引只讀 ' + PCT(HEAD.frac) + '，而且答對 ' + R3(HEAD.recall),
        en:'brute force reads every vector; this index reads ' + PCT(HEAD.frac) + ' of them and gets ' + R3(HEAD.recall) + ' of the answers right'},
  tabs: [
    {id:'walk', label:{zh:'走一次', en:'the walk'},
     stage:{zh:'從上層入口點貪婪下降，底層才展開 ef 個候選',
            en:'greedy descent from one entry point, ef candidates only at the bottom'},
     view:VIEW, code:CODE_WALK, build:walkFrames,
     idea:{zh:'索引不是把資料切桶，是**把資料連成一張圖**：每次跳到離查詢更近的鄰居，走不動就下一層。' + CFG.docs.toLocaleString() + ' 個向量裡只算了 ' + HEAD.hnsw_dist + ' 個距離（' + PCT(HEAD.frac) + '），recall@10 是 ' + R3(HEAD.recall) + '。',
           en:'The index does not bucket the data, it **wires the data into a graph**: hop to whichever neighbour is closer to the query, drop a layer when stuck. Out of ' + CFG.docs.toLocaleString() + ' vectors it computes ' + HEAD.hnsw_dist + ' distances (' + PCT(HEAD.frac) + ') for recall@10 ' + R3(HEAD.recall) + '.'},
     legend:[['#ff5c5c', {zh:'查詢', en:'the query'}],
             ['#ff9736', {zh:'走到並算過距離的點', en:'points reached and measured'}],
             ['#3fe0dd', {zh:'最後回傳的答案', en:'the answers returned'}],
             ['#0a6b74', {zh:'從頭到尾沒碰到', en:'never touched'}]]},
    {id:'layers', label:{zh:'為什麼分層', en:'the layers'},
     stage:{zh:'每個點丟一次骰子決定爬多高', en:'each point draws its level once and keeps it'},
     view:VIEW, code:CODE_LAYERS, build:layerFrames,
     idea:{zh:'層級用幾何分布抽：進入上一層的機率是 1/M。於是每層大約是下一層的 1/' + CFG.M + '——實測 ' + LAYERS.map(r => r.nodes).join(' / ') + '，跟理論值 ' + LAYERS.map(r => Math.round(r.expected)).join(' / ') + ' 幾乎一樣。',
           en:'Levels are drawn from a geometric distribution with probability 1/M of climbing one more, so each layer holds about 1/' + CFG.M + ' of the one below: measured ' + LAYERS.map(r => r.nodes).join(' / ') + ' against a predicted ' + LAYERS.map(r => Math.round(r.expected)).join(' / ') + '.'},
     legend:[['#3fe0dd', {zh:'該層實際的點數', en:'points actually on the layer'}],
             ['#8fa3ac', {zh:'理論值 N / M^l', en:'the prediction, N / M^l'}]]},
    {id:'ef', label:{zh:'ef 這個旋鈕', en:'the ef knob'},
     stage:{zh:'ef 是走路時同時活著的候選數', en:'ef is how many candidates stay alive during the walk'},
     view:VIEW, code:CODE_EF, build:efFrames,
     idea:{zh:'ef 是**唯一一個上線後還能轉的旋鈕**：不用重建索引就能換召回。' + CURVE.map(r => 'ef ' + r.ef + '→' + R3(r.recall)).join('、') + '，而距離計算從 ' + CURVE[0].dist + ' 爬到 ' + CURVE[CURVE.length - 1].dist + '。',
           en:'ef is **the one knob you can still turn in production**: it trades recall for work without rebuilding anything. ' + CURVE.map(r => 'ef ' + r.ef + ' gives ' + R3(r.recall)).join(', ') + ', while the distance computations climb from ' + CURVE[0].dist + ' to ' + CURVE[CURVE.length - 1].dist + '.'},
     legend:[['#3fe0dd', {zh:'召回', en:'recall'}],
             ['#ff9736', {zh:'距離計算次數', en:'distance computations'}]]},
    {id:'flat', label:{zh:'階層有用嗎', en:'does the hierarchy help'},
     stage:{zh:'把上層全拆掉，只留最底層那張圖', en:'throw the upper layers away and keep only the bottom graph'},
     view:VIEW, code:CODE_FLAT, build:flatFrames,
     idea:{zh:'名字裡的 H 是 hierarchical，但把它拆掉之後召回一樣是 ' + R3(FLAT[1].recall) + '、距離計算還少了一點（' + FLAT[0].dist + ' → ' + FLAT[1].dist + '）。**這是可重現的已知結果**，不是這裡的巧合。',
           en:'The H in the name is for hierarchical, yet removing it leaves recall at ' + R3(FLAT[1].recall) + ' and even shaves the distance computations (' + FLAT[0].dist + ' → ' + FLAT[1].dist + '). **This is a reproducible published result**, not an accident of this corpus.'},
     legend:[['#3fe0dd', {zh:'有階層', en:'with the hierarchy'}],
             ['#ff9736', {zh:'只剩單層', en:'flat, bottom layer only'}]]},
    {id:'m', label:{zh:'M 的代價', en:'the price of M'},
     stage:{zh:'召回會飽和，記憶體與建置時間不會', en:'recall saturates, memory and build time do not'},
     view:VIEW, code:CODE_M, build:mFrames,
     idea:{zh:'M 決定每個點存幾條邊。' + MS.map(r => 'M=' + r.M + '→' + R3(r.recall)).join('、') + '：召回早就不動了，但連結的位元組從 ' + MS[0].link + ' 漲到 ' + MS[MS.length - 1].link + '（向量本身 ' + MS[0].vec + '），建置時間 ' + MS[0].build + ' s → ' + MS[MS.length - 1].build + ' s。',
           en:'M sets how many edges each point stores. ' + MS.map(r => 'M=' + r.M + ' gives ' + R3(r.recall)).join(', ') + ': recall stopped moving long ago, but the link bytes go from ' + MS[0].link + ' to ' + MS[MS.length - 1].link + ' against ' + MS[0].vec + ' for the vector itself, and the build from ' + MS[0].build + ' s to ' + MS[MS.length - 1].build + ' s.'},
     legend:[['#0a6b74', {zh:'向量本身', en:'the vector itself'}],
             ['#ff9736', {zh:'連結佔掉的位元組', en:'the bytes spent on links'}]]},
    {id:'heur', label:{zh:'挑鄰居', en:'choosing neighbours'},
     stage:{zh:'只留最近的，就會把離開群集的邊砍光', en:'keeping only the nearest deletes every edge that leaves the cluster'},
     view:VIEW, code:CODE_HEUR, build:heurFrames,
     idea:{zh:'啟發式的規則只有一句：候選要**離新點比離任何已選的鄰居都近**才留下。這樣長程邊才活得下來，召回 ' + R3(HEUR[0].recall) + ' → ' + R3(HEUR[1].recall) + '，代價是比較慢的插入。',
           en:'The heuristic is one sentence: keep a candidate only if it is **closer to the new point than to anything already kept**. That is what preserves the long-range edges, lifting recall ' + R3(HEUR[0].recall) + ' → ' + R3(HEUR[1].recall) + ' at the price of a slower insert.'},
     legend:[['#ff5c5c', {zh:'只留最近的 m 個', en:'simply the m nearest'}],
             ['#3fe0dd', {zh:'啟發式（多樣性）', en:'the diversity heuristic'}]]},
    {id:'del', label:{zh:'刪除', en:'deletions'},
     stage:{zh:'刪除是墓碑，不是刪掉', en:'a deletion is a tombstone, not a removal'},
     view:VIEW, code:CODE_DEL, build:delFrames,
     idea:{zh:'點留在圖裡當踏腳石，只是從結果濾掉：刪了 ' + PCT(DEL[1].deleted) + ' 之後每個查詢有 ' + PCT(DEL[1].waste) + ' 的距離計算換不到任何結果。重建掃掉之後 ' + DEL[1].dist + ' → ' + DEL[2].dist + '。**「支援刪除」其實是「有一個重整排程」**。',
           en:'The point stays in the graph as a stepping stone and is filtered out of the results: after deleting ' + PCT(DEL[1].deleted) + ', ' + PCT(DEL[1].waste) + ' of every query\'s distance computations can never pay off. A rebuild sweeps it away, ' + DEL[1].dist + ' → ' + DEL[2].dist + '. **"Supports deletes" really means "has a compaction schedule".**'},
     legend:[['#3fe0dd', {zh:'換得到結果的工作', en:'work that can produce a result'}],
             ['#ff5c5c', {zh:'花在墓碑上的工作', en:'work spent on tombstones'}]]},
    {id:'scale', label:{zh:'規模', en:'scaling'},
     stage:{zh:'語料六倍，工作量兩倍', en:'six times the corpus, twice the work'},
     view:VIEW, code:CODE_SCALE, build:scaleFrames,
     idea:{zh:'暴力搜尋跟語料嚴格成正比，HNSW 大致隨**對數**成長：' + SCALE[0].docs + ' → ' + SCALE[SCALE.length - 1].docs.toLocaleString() + ' 個向量，距離計算只從 ' + SCALE[0].dist + ' 變 ' + SCALE[SCALE.length - 1].dist + '。但召回也同時從 ' + R3(SCALE[0].recall) + ' 掉到 ' + R3(SCALE[SCALE.length - 1].recall) + '——**語料長大就要重調 ef**。',
           en:'Brute force is strictly proportional to the corpus while HNSW grows roughly with its **logarithm**: from ' + SCALE[0].docs + ' to ' + SCALE[SCALE.length - 1].docs.toLocaleString() + ' vectors the distance computations go only from ' + SCALE[0].dist + ' to ' + SCALE[SCALE.length - 1].dist + '. Recall drifts down from ' + R3(SCALE[0].recall) + ' to ' + R3(SCALE[SCALE.length - 1].recall) + ' at the same time - **a growing corpus needs ef re-tuned**.'},
     legend:[['#ff5c5c', {zh:'暴力搜尋', en:'brute force'}],
             ['#3fe0dd', {zh:'HNSW（ef 50）', en:'HNSW at ef 50'}]]},
    {id:'fail', label:{zh:'不會報錯的錯', en:'quiet failures'},
     stage:{zh:'三個都給你結果，三個都是錯的', en:'all three return results, all three are wrong'},
     view:VIEW, code:CODE_FAIL, build:failFrames,
     idea:{zh:'ef 比 k 小（回不滿 k 筆，' + R3(FAIL.unclamped) + '）、用內積但向量沒正規化（' + R3(NORM.recall) + '，前十名 ' + PCT(NORM.hits) + ' 來自被放大的那群）、拿索引自己當 ground truth（永遠 ' + R3(CIRC.self) + '）。**近似索引一定要有一個不近似的對照組**。',
           en:'ef below k, so fewer than k results exist (' + R3(FAIL.unclamped) + '); inner product on unnormalised vectors (' + R3(NORM.recall) + ', with ' + PCT(NORM.hits) + ' of the top ten coming from the scaled group); and grading the index against its own output (always ' + R3(CIRC.self) + '). **An approximate index always needs a non-approximate reference.**'},
     legend:[['#3fe0dd', {zh:'正確的那條', en:'the correct one'}],
             ['#ff5c5c', {zh:'壞掉但不會說', en:'broken, and silent about it'}]]}
  ]
};
