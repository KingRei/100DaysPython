// DAY: 19
// TITLE_ZH: A*：讓搜尋帶著一個猜測往前走
// TITLE_EN: A* - letting the search carry a guess
// SUB_ZH: 把 Dijkstra 的優先順序從 g 換成 f = g + h，答案不變、看的地圖少一大半；猜過頭就會安安靜靜地給你一條比較貴的路。
// SUB_EN: Swap Dijkstra's priority from g to f = g + h and the answer stays the same while most of the map goes unvisited - but overestimate, and you quietly get a costlier path.
// FOLDER: day%2019%20-%20a%20star%20and%20parallel%20search
// MEDIUM: https://medium.com/100-days-of-python

/* ------------------------------------------------------------------ the map */
const GRID = ['111111111', '111111111', '115555511', '111111111',
              '111111111', '111111111'];
const ROWS = 6, COLS = 9;
const SR = 2, SC = 0, GR = 2, GC = 8;
const CW = .82, GX0 = .48, GY0 = .62;
const GVIEW = [8.4, 6.2];

const cst = (r, c) => Number(GRID[r][c]);
const kk = (r, c) => r * COLS + c;
const NB = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const manh = (r, c) => Math.abs(r - GR) + Math.abs(c - GC);

function nbrs(r, c){
  const out = [];
  NB.forEach(([dr, dc]) => {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) out.push([nr, nc, cst(nr, nc)]);
  });
  return out;
}

/* one grid picture */
function gridShapes(o){
  o = o || {};
  const g = o.g || {}, cl = o.closed || {}, fr = o.frontier || {}, pa = o.path || {};
  const out = [];
  for (let r = 0; r < ROWS; r++){
    for (let c = 0; c < COLS; c++){
      const k = kk(r, c);
      let s = 'ghost';
      if (cl[k]) s = 'done';
      if (fr[k]) s = 'act';
      if (pa[k]) s = o.wrong ? 'bad' : 'ok';
      if (o.cur && o.cur[0] === r && o.cur[1] === c) s = 'hot';
      let lab = (g[k] == null) ? '' : String(g[k]);
      if (r === SR && c === SC) lab = 'S';
      if (r === GR && c === GC) lab = 'G';
      const opt = {fs:.30, rx:.06};
      if (cst(r, c) > 1) opt.dash = '.13 .11';
      out.push(S.r(GX0 + c * CW, GY0 + r * CW, CW - .07, CW - .07, s, lab, opt));
    }
  }
  if (o.note) out.push(S.t(GVIEW[0] / 2, GY0 + ROWS * CW + .52, o.note,
                           {c:o.noteC || COL.tealL, fs:.34}));
  out.push(S.t(GVIEW[0] / 2, .34, o.top || {zh:'虛線格子＝泥地，踏進去要付 5',
                                            en:'dashed cells are mud - 5 to step onto'},
               {c:COL.grey, fs:.28}));
  return out;
}

const fmt = x => (Math.abs(x - Math.round(x)) < 1e-9) ? String(Math.round(x)) : x.toFixed(1);
const cmp = (a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);

/* --------------------------------------------------- Dijkstra / A* / friends
   cfg = {h, weight, greedy, why(u), name}                                    */
function searchRun(cfg){
  const F = new Frames();
  const h = cfg.h || (() => 0);
  const wt = cfg.weight == null ? 1 : cfg.weight;
  const greedy = !!cfg.greedy;
  const pri = (r, c, gv) => greedy ? h(r, c) : gv + wt * h(r, c);

  const g = {}, par = {}, closed = {};
  g[kk(SR, SC)] = 0; par[kk(SR, SC)] = null;
  let pq = [[pri(SR, SC, 0), SR, SC]];
  const expanded = [];
  let mudSeen = false;

  const frontierMap = () => { const m = {}; pq.forEach(e => { if (!closed[kk(e[1], e[2])]) m[kk(e[1], e[2])] = 1; }); return m; };
  const panels = (cur) => [
    {lbl:{zh:'frontier（依 f 排序，最小的先出）', en:'frontier (ordered by f, smallest first)'},
     chips:pq.filter(e => !closed[kk(e[1], e[2])]).sort(cmp).slice(0, 6)
             .map((e, i) => ({t:'(' + e[1] + ',' + e[2] + ')  f=' + fmt(e[0]), cls:i ? 'act' : 'hot'}))},
    {lbl:{zh:'已展開（pop 過就定案）', en:'expanded (popped, and final)'},
     chips:[{t:String(expanded.length), cls:'ok'}]},
    {lbl:{zh:'目前這一格的帳', en:'the arithmetic for this cell'},
     chips:cur ? cur : [{t:'—', cls:'empty'}]}
  ];

  F.push({shapes:gridShapes({g:{[kk(SR, SC)]:0}, frontier:{[kk(SR, SC)]:1},
            note:cfg.opening.note}),
          panels:panels([{t:'h(S) = ' + h(SR, SC), cls:'act'}]),
          view:GVIEW, line:4, msg:cfg.opening.msg});

  while (pq.length){
    pq.sort(cmp);
    const [f, r, c] = pq.shift();
    const k = kk(r, c);
    if (closed[k]) continue;
    closed[k] = 1;
    expanded.push([r, c]);

    const gv = g[k], hv = h(r, c);
    const cur = [{t:'g = ' + gv, cls:'ok'}, {t:'h = ' + hv, cls:'act'},
                 {t:'f = ' + fmt(f), cls:'hot'}];
    let msg = cfg.why(r, c, gv, hv, f, expanded.length, mudSeen);
    if (cst(r, c) > 1 && !mudSeen){ mudSeen = true; }

    if (r === GR && c === GC){
      // rebuild the path and stop
      const path = {}; let cu = k;
      while (cu != null){ path[cu] = 1; cu = par[cu]; }
      F.push({shapes:gridShapes({g:g, closed:closed, path:path, cur:[r, c],
                note:cfg.done.note(gv, expanded.length), noteC:cfg.wrong ? COL.red : COL.tealL,
                wrong:cfg.wrong}),
              panels:panels(cur), view:GVIEW, line:10,
              msg:cfg.done.msg(gv, expanded.length)});
      break;
    }

    F.push({shapes:gridShapes({g:g, closed:closed, frontier:frontierMap(), cur:[r, c]}),
            panels:panels(cur), view:GVIEW, line:9, msg:msg});

    nbrs(r, c).forEach(([nr, nc, w]) => {
      const nk = kk(nr, nc), ng = gv + w;
      if (g[nk] == null || ng < g[nk]){
        g[nk] = ng; par[nk] = k;
        pq.push([pri(nr, nc, ng), nr, nc]);
      }
    });
  }
  return F.list;
}

/* ------------------------------------------------------- narration builders */
const WHY_DIJ = (r, c, gv, hv, f, n) => ({
  zh:'pop (' + r + ',' + c + ')：<b>f = g = ' + fmt(f) + '</b>，h 是 0 所以完全沒有方向感。' +
     'Dijkstra 挑的是「目前最便宜」，不管那格在地圖的哪一邊——' +
     '所以它會把整圈半徑 ' + gv + ' 的格子都翻過一遍（目前第 ' + n + ' 格）。',
  en:'pop (' + r + ',' + c + '): <b>f = g = ' + fmt(f) + '</b>. With h = 0 there is no sense of direction. ' +
     'Dijkstra always takes the cheapest cell so far, wherever it is - so it sweeps the whole ring of radius ' +
     gv + ' before moving on (cell ' + n + ' so far).'
});
const WHY_AST = (r, c, gv, hv, f, n) => ({
  zh:'pop (' + r + ',' + c + ')：<b>f = g + h = ' + gv + ' + ' + hv + ' = ' + fmt(f) + '</b>。' +
     '往回走的格子 g 會變大、h 也變大，f 立刻墊高，於是永遠排在後面；' +
     '朝著 G 的格子 h 一路遞減，才排得到前面（目前第 ' + n + ' 格）。',
  en:'pop (' + r + ',' + c + '): <b>f = g + h = ' + gv + ' + ' + hv + ' = ' + fmt(f) + '</b>. ' +
     'A cell behind us has both a larger g and a larger h, so its f is doubly penalised and it sinks in the queue; ' +
     'only cells whose h keeps dropping stay near the front (cell ' + n + ' so far).'
});
const WHY_W5 = (r, c, gv, hv, f, n) => ({
  zh:'pop (' + r + ',' + c + ')：<b>f = g + 5h = ' + gv + ' + 5x' + hv + ' = ' + fmt(f) + '</b>。' +
     '猜測被放大五倍，g 的比重被壓扁——繞過泥地要多走的兩步，在 f 裡看起來比' +
     '直接踩進泥地還貴，於是搜尋一頭鑽進去。',
  en:'pop (' + r + ',' + c + '): <b>f = g + 5h = ' + gv + ' + 5x' + hv + ' = ' + fmt(f) + '</b>. ' +
     'The guess is amplified five times, which flattens the weight of g: the two extra steps of the detour ' +
     'now look more expensive than walking into the mud, so the search dives straight in.'
});
const WHY_GR = (r, c, gv, hv, f, n) => ({
  zh:'pop (' + r + ',' + c + ')：<b>f = h = ' + hv + '</b>，g 完全被丟掉。' +
     '「離 G 比較近」變成唯一的理由，走過的成本一毛都不算——' +
     '這就是為什麼它一路直線衝進泥地還覺得自己做得很好。',
  en:'pop (' + r + ',' + c + '): <b>f = h = ' + hv + '</b> - g is thrown away entirely. ' +
     '"Closer to G" becomes the only argument and the bill so far counts for nothing, ' +
     'which is why it walks a straight line into the mud and feels good about it.'
});

/* ------------------------------------------------------- delta-stepping ---- */
const WEDGES = [
  [0, 1, 7], [0, 3, 1], [0, 5, 3], [0, 7, 7],
  [0, 13, 1], [1, 2, 1], [1, 4, 7], [1, 10, 1],
  [2, 6, 1], [2, 11, 1], [3, 2, 1], [4, 12, 1],
  [5, 8, 7], [5, 13, 1], [7, 8, 3], [8, 9, 7],
  [10, 12, 1], [11, 10, 9], [12, 7, 9], [12, 10, 2],
  [12, 11, 3], [13, 2, 2], [13, 11, 7]
];
const NV = 14;
function wadj(){
  const a = {};
  for (let i = 0; i < NV; i++) a[i] = [];
  WEDGES.forEach(([u, v, w]) => { a[u].push([v, w]); a[v].push([u, w]); });
  return a;
}

/* faithful port of delta_stepping() in astar.py - phases and relaxation count */
function deltaRun(delta){
  const adj = wadj();
  const dist = {0:0};
  let buckets = {0:{0:1}};
  const evs = [];
  let relax = 0, b = 0;
  const keys = () => Object.keys(buckets).map(Number).filter(k => Object.keys(buckets[k]).length);
  while (keys().length){
    if (!buckets[b] || !Object.keys(buckets[b]).length) b = Math.min(...keys());
    const heavy = [];
    while (buckets[b] && Object.keys(buckets[b]).length){
      const frontier = Object.keys(buckets[b]).map(Number).sort((x, y) => x - y);
      delete buckets[b];
      evs.push({b:b, verts:frontier, relax:relax, dist:Object.assign({}, dist), heavy:0});
      frontier.forEach(u => {
        adj[u].forEach(([v, w]) => {
          relax++;
          const nd = dist[u] + w;
          if (dist[v] == null || nd < dist[v]){
            if (w <= delta){
              if (dist[v] != null){
                const old = Math.floor(dist[v] / delta);
                if (buckets[old]) delete buckets[old][v];
              }
              dist[v] = nd;
              (buckets[Math.floor(nd / delta)] = buckets[Math.floor(nd / delta)] || {})[v] = 1;
            } else heavy.push([u, v, w]);
          }
        });
      });
      evs[evs.length - 1].after = relax;
    }
    heavy.forEach(([u, v, w]) => {
      relax++;
      const nd = dist[u] + w;
      if (dist[v] == null || nd < dist[v]){
        dist[v] = nd;
        (buckets[Math.floor(nd / delta)] = buckets[Math.floor(nd / delta)] || {})[v] = 1;
      }
    });
    if (evs.length) evs[evs.length - 1].heavy = heavy.length;
    const kk2 = keys();
    const nb = {}; kk2.forEach(k => nb[k] = buckets[k]); buckets = nb;
    if (kk2.length) b = Math.min(...kk2);
  }
  return {dist:dist, evs:evs, relax:relax};
}

const DV = [8.8, 6.3];
function deltaShapes(evs, upto, delta, relax){
  const out = [];
  const y0 = 1.02, rh = .50, x0 = 2.30, cwv = .50;
  out.push(S.t(DV[0] / 2, .40, {zh:'delta = ' + delta + '　（每一列＝一個可以整批平行放鬆的 batch）',
                                en:'delta = ' + delta + '  (each row is one batch that could be relaxed in parallel)'},
               {c:COL.tealL, fs:.34}));
  const seen = {};
  for (let i = 0; i <= upto && i < evs.length; i++){
    const e = evs[i];
    const cur = i === upto;
    out.push(S.t(.28, y0 + i * rh + .19,
                 {zh:'phase ' + (i + 1) + '　bucket ' + e.b, en:'phase ' + (i + 1) + '   bucket ' + e.b},
                 {c:cur ? COL.orange : COL.grey, fs:.28, anchor:'start'}));
    e.verts.forEach((v, j) => {
      const again = seen[v] ? 1 : 0;
      const st = cur ? (again ? 'bad' : 'hot') : (again ? 'bad' : 'done');
      out.push(S.r(x0 + j * cwv, y0 + i * rh, cwv - .07, rh - .10, st, String(v), {fs:.26, rx:.05}));
      seen[v] = 1;
    });
    if (cur) out.push(S.t(x0 + Math.max(e.verts.length, 1) * cwv + .30, y0 + i * rh + .19,
                          {zh:'+' + (e.after - e.relax) + ' 次邊放鬆', en:'+' + (e.after - e.relax) + ' relaxations'},
                          {c:COL.orange, fs:.26, anchor:'start'}));
  }
  out.push(S.t(DV[0] / 2, DV[1] - .22,
               {zh:'累計邊放鬆 ' + relax + ' 次　·　紅色＝這個點之前已經出現過，這次是重算',
                en:'edge relaxations so far: ' + relax + '  ·  red = this vertex appeared in an earlier phase - repeated work'},
               {c:COL.grey, fs:.27}));
  return out;
}

function deltaFrames(delta){
  const R = deltaRun(delta);
  const F = new Frames();
  const distChips = d => {
    const ks = Object.keys(d).map(Number).sort((a, b) => a - b);
    return ks.map(v => ({t:v + ':' + d[v], cls:'ok'}));
  };
  R.evs.forEach((e, i) => {
    const b4 = {};
    for (let j = 0; j < i; j++) R.evs[j].verts.forEach(v => b4[v] = 1);
    const again = e.verts.filter(v => b4[v]);
    const msg = {
      zh:'phase ' + (i + 1) + '：bucket ' + e.b + ' 一次settle ' + e.verts.length + ' 個點' +
         (e.verts.length > 1 ? '——它們彼此不互相依賴，理論上可以丟給 ' + e.verts.length + ' 個核心同時放鬆。' : '（只有一個點，這一步跟 Dijkstra 沒兩樣）。') +
         (again.length ? '<b>但 ' + again.join(', ') + ' 之前已經被 settle 過</b>，' +
          '因為 delta 太寬，當時的距離還不是最終值，現在得重來一次——這就是batch 變大要付的帳。' :
          ' 這一批花了 ' + (e.after - e.relax) + ' 次邊放鬆。') +
         (e.heavy ? '　phase 結束時還有 ' + e.heavy + ' 條 heavy edge（w > delta）要補放鬆：它們一定會把點推到後面的 bucket，早算晚算一樣，所以延後。' : ''),
      en:'phase ' + (i + 1) + ': bucket ' + e.b + ' settles ' + e.verts.length + ' vertices at once' +
         (e.verts.length > 1 ? ' - none of them depends on another, so all ' + e.verts.length + ' could be relaxed on separate cores.' : ' (a single vertex - this step is exactly Dijkstra).') +
         (again.length ? ' <b>But ' + again.join(', ') + ' was settled in an earlier phase</b>: delta is wide enough that the distance was not final then, so the work is redone now - that is the bill for bigger batches.' :
          ' This batch cost ' + (e.after - e.relax) + ' edge relaxations.') +
         (e.heavy ? ' At the end of the phase ' + e.heavy + ' heavy edges (w > delta) are relaxed: they can only push a vertex into a later bucket, so deferring them changes nothing.' : '')
    };
    F.push({shapes:deltaShapes(R.evs, i, delta, e.after), view:DV, line:delta >= 99 ? 11 : 6,
            panels:[{lbl:{zh:'這一批（可平行）', en:'this batch (parallel)'},
                     chips:e.verts.map(v => ({t:String(v), cls:b4[v] ? 'bad' : 'hot'}))},
                    {lbl:{zh:'目前的 dist（可能還會被改小）', en:'dist so far (still shrinkable)'},
                     chips:distChips(e.dist)},
                    {lbl:{zh:'累計邊放鬆', en:'edge relaxations'},
                     chips:[{t:String(e.after), cls:'act'}]}],
            msg:msg});
  });
  const last = R.evs.length;
  F.push({shapes:deltaShapes(R.evs, last - 1, delta, R.relax), view:DV, line:15,
          panels:[{lbl:{zh:'最終 dist', en:'final dist'}, chips:distChips(R.dist)},
                  {lbl:{zh:'phases（＝平行時的深度）', en:'phases (= depth if parallel)'},
                   chips:[{t:String(R.evs.length), cls:'hot'}]},
                  {lbl:{zh:'邊放鬆總數（＝總工作量）', en:'edge relaxations (= total work)'},
                   chips:[{t:String(R.relax), cls:'act'}]}],
          msg:{zh:'收工：<b>' + R.evs.length + ' 個 phase、' + R.relax + ' 次邊放鬆</b>。' +
                  'delta 小＝每批很純、幾乎不重算，但要等很多輪；delta 大＝輪數變少，' +
                  '代價是同一個點被反覆放鬆。答案永遠一樣，變的只是「深度」與「工作量」怎麼分配——' +
                  'delta 就是這條線上的旋鈕。',
               en:'Done: <b>' + R.evs.length + ' phases, ' + R.relax + ' edge relaxations</b>. ' +
                  'A small delta keeps every batch clean and wastes almost nothing, but you wait through many rounds; ' +
                  'a large delta cuts the rounds at the cost of relaxing the same vertex again and again. ' +
                  'The answer never changes - only how the cost is split between depth and work. delta is the knob on that line.'}});
  return F.list;
}

/* --------------------------------------------- level-synchronous BFS ------- */
function bfsFrames(){
  const F = new Frames();
  const seen = {}, done = {};
  let frontier = [[SR, SC]];
  seen[kk(SR, SC)] = 1;
  const levels = [];
  let lv = 0;
  while (frontier.length){
    levels.push(frontier.length);
    const fm = {}; frontier.forEach(([r, c]) => fm[kk(r, c)] = 1);
    const g = {};
    Object.keys(done).forEach(k => g[k] = null);
    F.push({shapes:gridShapes({closed:done, frontier:fm,
              note:{zh:'level ' + lv + '　·　這一層有 ' + frontier.length + ' 格，可以同時展開',
                    en:'level ' + lv + '  ·  ' + frontier.length + ' cells here, all expandable at once'},
              top:{zh:'BFS 不是「一個 queue」，而是「一層 frontier」', en:'BFS as a frontier, not a queue'}}),
            view:GVIEW, line:7,
            panels:[{lbl:{zh:'目前這一層（可平行）', en:'current level (parallel)'},
                     chips:frontier.map(([r, c]) => ({t:'(' + r + ',' + c + ')', cls:'hot'}))},
                    {lbl:{zh:'每層寬度', en:'width of each level'},
                     chips:levels.map((w, i) => ({t:'L' + i + ':' + w, cls:i === lv ? 'hot' : 'ok'}))},
                    {lbl:{zh:'深度（＝要等的輪數）', en:'depth (rounds you must wait)'},
                     chips:[{t:String(lv + 1), cls:'act'}]}],
            msg:{zh:'level ' + lv + ' 有 <b>' + frontier.length + ' 格</b>。' +
                    '同一層裡的格子彼此沒有先後關係——誰先展開結果都一樣，' +
                    '所以這個 for 迴圈才是可以平行的那一段；唯一共用的狀態是 <code>seen</code>，' +
                    '平行版把它換成 atomic 的 test-and-set 就好。總工作量還是 O(V+E)，' +
                    '真正變短的是<b>深度</b>：你要等的是層數，不是格子數。',
                 en:'Level ' + lv + ' holds <b>' + frontier.length + ' cells</b>. ' +
                    'Cells inside one level do not depend on each other - expanding them in any order gives the same result, ' +
                    'which is why this loop is the parallel part. The only shared state is <code>seen</code>, and a parallel ' +
                    'version replaces it with an atomic test-and-set. The work stays O(V+E); what shrinks is <b>depth</b> - ' +
                    'you wait for the number of levels, not the number of cells.'}});
    frontier.forEach(([r, c]) => done[kk(r, c)] = 1);
    const nxt = [];
    frontier.forEach(([r, c]) => nbrs(r, c).forEach(([nr, nc]) => {
      if (!seen[kk(nr, nc)]){ seen[kk(nr, nc)] = 1; nxt.push([nr, nc]); }
    }));
    frontier = nxt; lv++;
  }
  const widest = Math.max(...levels);
  F.push({shapes:gridShapes({closed:done,
            note:{zh:'12 層走完全圖：work O(V+E) 不變，depth = ' + levels.length,
                  en:'the whole grid in ' + levels.length + ' levels: work O(V+E), depth = ' + levels.length}}),
          view:GVIEW, line:12,
          panels:[{lbl:{zh:'每層寬度', en:'width of each level'},
                   chips:levels.map((w, i) => ({t:'L' + i + ':' + w, cls:w === widest ? 'hot' : 'ok'}))},
                  {lbl:{zh:'深度', en:'depth'}, chips:[{t:String(levels.length), cls:'act'}]},
                  {lbl:{zh:'最寬的一層（＝能用上的核心數上限）', en:'widest level (= usable parallelism)'},
                   chips:[{t:String(widest), cls:'hot'}]}],
          msg:{zh:'總共 <b>' + levels.length + ' 層</b>，最寬的一層只有 <b>' + widest + ' 格</b>。' +
                  '這就是平行 BFS 的天花板：再多核心也用不上，因為同一時間只有 ' + widest +
                  ' 件事可做。<b>work（總量）跟 depth（關鍵路徑）是兩個要分開看的成本</b>，' +
                  '這也是為什麼 Dijkstra 難平行——它的 depth 等於 V，每次只肯 settle 一個點。',
               en:'<b>' + levels.length + ' levels</b>, and the widest holds only <b>' + widest + ' cells</b>. ' +
                  'That is the ceiling for a parallel BFS: extra cores sit idle because at any moment there are only ' +
                  widest + ' independent things to do. <b>Work (total) and depth (critical path) are two separate costs</b>, ' +
                  'and that is exactly why Dijkstra resists parallelism - its depth is V, since it settles one vertex at a time.'}});
  return F.list;
}

/* ------------------------------------------------------ LeetCode 1091 ------ */
const LCB = [[0, 0, 0, 1, 0, 0],
             [1, 1, 0, 1, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 0],
             [1, 1, 1, 0, 0, 0]];
const LN = 6, LCW = .96, LX0 = .55, LY0 = .95, LVIEW = [6.9, 7.3];
const lkey = (r, c) => r * LN + c;
const cheb = (r, c) => Math.max(LN - 1 - r, LN - 1 - c);

function lcShapes(o){
  o = o || {};
  const out = [], g = o.g || {}, cl = o.closed || {}, fr = o.frontier || {}, pa = o.path || {};
  for (let r = 0; r < LN; r++){
    for (let c = 0; c < LN; c++){
      const k = lkey(r, c);
      let s, lab = '';
      if (LCB[r][c]){ s = 'idle'; lab = ''; }
      else {
        s = 'ghost';
        if (cl[k]) s = 'done';
        if (fr[k]) s = 'act';
        if (pa[k]) s = 'ok';
        if (o.cur && o.cur[0] === r && o.cur[1] === c) s = 'hot';
        if (g[k] != null) lab = String(g[k]);
      }
      out.push(S.r(LX0 + c * LCW, LY0 + r * LCW, LCW - .08, LCW - .08, s, lab,
                   {fs:.32, rx:.06, dash:LCB[r][c] ? '.10 .10' : undefined}));
    }
  }
  out.push(S.t(LVIEW[0] / 2, .42, o.top || {zh:'八方向：格子上的數字＝走到這裡經過幾格',
                                            en:'8 directions; the number is how many cells the path uses so far'},
               {c:COL.grey, fs:.28}));
  if (o.note) out.push(S.t(LVIEW[0] / 2, LY0 + LN * LCW + .48, o.note,
                           {c:o.noteC || COL.tealL, fs:.34}));
  return out;
}

function lcFrames(useH){
  const F = new Frames();
  const h = useH ? cheb : (() => 0);
  const g = {}, par = {}, closed = {};
  const sk = lkey(0, 0);
  g[sk] = 1; par[sk] = null;
  let pq = [[1 + h(0, 0), 0, 0]];
  let n = 0;
  const front = () => { const m = {}; pq.forEach(e => { if (!closed[lkey(e[1], e[2])]) m[lkey(e[1], e[2])] = 1; }); return m; };
  F.push({shapes:lcShapes({g:{[sk]:1}, frontier:{[sk]:1},
            note:useH ? {zh:'h = Chebyshev = max(|dr|, |dc|)', en:'h = Chebyshev = max(|dr|, |dc|)'}
                      : {zh:'h = 0，也就是純 BFS/Dijkstra', en:'h = 0 - plain BFS/Dijkstra'}}),
          view:LVIEW, line:useH ? 4 : 5,
          panels:[{lbl:{zh:'heuristic', en:'heuristic'},
                   chips:[{t:useH ? 'max(|dr|,|dc|) = ' + cheb(0, 0) : 'h = 0', cls:'act'}]},
                  {lbl:{zh:'已展開', en:'expanded'}, chips:[{t:'0', cls:'ok'}]},
                  {lbl:{zh:'注意', en:'careful'},
                   chips:[{t:useH ? 'Manhattan is NOT admissible here' : 'no direction at all', cls:useH ? 'bad' : 'dim'}]}],
          msg:useH ?
            {zh:'八方向的格子，一步可以斜著走，所以「一步最多讓 |dr| 和 |dc| 各減 1」。' +
                '真正的下界是 <b>Chebyshev max(|dr|,|dc|)</b>，不是 Manhattan——' +
                'Manhattan 會把一步斜走算成兩步，<b>高估</b>剩下的距離，變成不可採納（inadmissible），' +
                '答案就可能不是最短的。這一格 h = ' + cheb(0, 0) + '，剛好是「全斜線衝過去」的步數。',
             en:'Diagonal moves are allowed, so one step can cut both |dr| and |dc| by one. The real lower bound is ' +
                '<b>Chebyshev, max(|dr|,|dc|)</b>, not Manhattan - Manhattan counts a diagonal step twice and ' +
                '<b>overestimates</b> what is left, which makes it inadmissible and the answer no longer guaranteed shortest. ' +
                'Here h = ' + cheb(0, 0) + ', exactly the number of steps a clear diagonal run would take.'} :
            {zh:'先看沒有 heuristic 的版本：h = 0，f 就退回 g，這正是 BFS/Dijkstra。' +
                '它會像水波一樣往四面八方漫開，包含完全背對終點的方向。',
             en:'Start with no heuristic: h = 0 makes f collapse back to g, which is exactly BFS/Dijkstra. ' +
                'It spreads out like a ripple in every direction, including straight away from the goal.'}});

  while (pq.length){
    pq.sort(cmp);
    const [f, r, c] = pq.shift();
    const k = lkey(r, c);
    if (closed[k]) continue;
    closed[k] = 1; n++;
    if (r === LN - 1 && c === LN - 1){
      const path = {}; let cu = k;
      while (cu != null){ path[cu] = 1; cu = par[cu]; }
      F.push({shapes:lcShapes({g:g, closed:closed, path:path, cur:[r, c],
                note:{zh:'答案 ' + g[k] + ' 格　·　展開 ' + n + ' 格',
                      en:'answer ' + g[k] + ' cells  ·  ' + n + ' expanded'}}),
              view:LVIEW, line:9,
              panels:[{lbl:{zh:'答案', en:'answer'}, chips:[{t:String(g[k]), cls:'ok'}]},
                      {lbl:{zh:'展開格數', en:'cells expanded'}, chips:[{t:String(n), cls:'hot'}]},
                      {lbl:{zh:'heuristic', en:'heuristic'},
                       chips:[{t:useH ? 'Chebyshev' : 'h = 0', cls:'act'}]}],
              msg:{zh:'走到右下角，答案 <b>' + g[k] + ' 格</b>，一共展開 <b>' + n + ' 格</b>。' +
                      (useH ? '兩個版本的答案完全一樣，差的只是看了多少地圖——' +
                              '這就是 heuristic 的全部價值：<b>不改答案，只改工作量</b>。' +
                              '面試時真正的考點不是會不會寫 A*，而是知不知道這裡只能用 Chebyshev。' :
                              '記下這個展開數，等一下換成 Chebyshev 再看一次。'),
                   en:'Bottom-right reached: the answer is <b>' + g[k] + ' cells</b> after expanding <b>' + n + '</b>. ' +
                      (useH ? 'Both versions return the same answer; the only difference is how much of the board was looked at. ' +
                              'That is the whole value of a heuristic: <b>it changes the work, never the answer</b>. ' +
                              'In an interview the real test is not whether you can write A*, but whether you know Chebyshev is the only admissible choice here.' :
                              'Remember that count - now switch to Chebyshev and watch it again.')}});
      break;
    }
    const gv = g[k];
    F.push({shapes:lcShapes({g:g, closed:closed, frontier:front(), cur:[r, c]}),
            view:LVIEW, line:8,
            panels:[{lbl:{zh:'frontier（f 最小者先）', en:'frontier (smallest f first)'},
                     chips:pq.filter(e => !closed[lkey(e[1], e[2])]).sort(cmp).slice(0, 6)
                             .map((e, i) => ({t:'(' + e[1] + ',' + e[2] + ') f=' + e[0], cls:i ? 'act' : 'hot'}))},
                    {lbl:{zh:'已展開', en:'expanded'}, chips:[{t:String(n), cls:'ok'}]},
                    {lbl:{zh:'這一格', en:'this cell'},
                     chips:[{t:'g=' + gv, cls:'ok'}, {t:'h=' + h(r, c), cls:'act'}, {t:'f=' + f, cls:'hot'}]}],
            msg:{zh:'pop (' + r + ',' + c + ')，g = ' + gv + '、h = ' + h(r, c) + '、f = ' + f + '。' +
                    (useH ? '因為 h 是「一路斜衝」的步數，只要這一格離終點近，f 就壓得低，' +
                            '整個搜尋被拉成一條沿著對角線的細帶。' :
                            'h = 0，所以排序只看走了幾格：離終點越遠的格子也照樣輪得到。'),
                 en:'pop (' + r + ',' + c + ') with g = ' + gv + ', h = ' + h(r, c) + ', f = ' + f + '. ' +
                    (useH ? 'Because h counts a clear diagonal run, any cell near the goal gets a low f and the search ' +
                            'is squeezed into a narrow band along the diagonal.' :
                            'With h = 0 the ordering only knows how far we have walked, so cells facing away from the goal get their turn too.')}});
    for (let dr = -1; dr <= 1; dr++){
      for (let dc = -1; dc <= 1; dc++){
        if (!dr && !dc) continue;
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= LN || nc < 0 || nc >= LN || LCB[nr][nc]) continue;
        const nk = lkey(nr, nc), ng = gv + 1;
        if (g[nk] == null || ng < g[nk]){
          g[nk] = ng; par[nk] = k;
          pq.push([ng + h(nr, nc), nr, nc]);
        }
      }
    }
  }
  return F.list;
}

/* ------------------------------------------------------------- code panes -- */
const CODE_A = [
  'def best_first(start, goal, h=None, weight=1.0, greedy=False):',
  '    h = h or (lambda cell: 0)          # h = 0  ->  Dijkstra, exactly',
  '    key = (lambda cell, g: h(cell)) if greedy \\',
  '          else (lambda cell, g: g + weight * h(cell))',
  '    pq = [(key(start, 0), start)]      # ordered by f, not by g',
  '    dist = {start: 0}; parent = {start: None}; closed = set()',
  '    while pq:',
  '        _, u = heapq.heappop(pq)       # the smallest f wins',
  '        if u in closed: continue',
  '        closed.add(u)                  # popped == final, if h is consistent',
  '        if u == goal: break',
  '        for v, w in neighbours(u):     # w = cost of stepping ONTO v',
  '            nd = dist[u] + w',
  '            if v not in dist or nd < dist[v]:',
  '                dist[v] = nd; parent[v] = u',
  '                heapq.heappush(pq, (key(v, nd), v))'
];
const CODE_D = [
  'def delta_stepping(adj, start, delta):',
  '    dist = {start: 0}',
  '    buckets = {0: {start}}             # bucket i holds [i*d, (i+1)*d)',
  '    while buckets:',
  '        b = min(buckets); heavy = []',
  '        while buckets.get(b):          # re-scan: a light edge can re-fill b',
  '            frontier = buckets.pop(b)',
  '            for u in frontier:         # <- one parallel batch',
  '                for v, w in adj[u]:',
  '                    nd = dist[u] + w',
  '                    if nd < dist.get(v, INF):',
  '                        if w <= delta:         # light: may land back in b',
  '                            move_to_bucket(v, nd)',
  '                        else:                  # heavy: defer to phase end',
  '                            heavy.append((u, v, w))',
  '        for u, v, w in heavy:          # distances can no longer shrink here',
  '            move_to_bucket(v, dist[u] + w)'
];
const CODE_B = [
  'def parallel_bfs_levels(adj, start):',
  '    seen = {start}',
  '    frontier = [start]                 # a frontier, not a queue',
  '    levels = []',
  '    while frontier:',
  '        levels.append(frontier)',
  '        nxt = []',
  '        for u in frontier:             # <- every u here is independent',
  '            for v, _ in adj[u]:',
  '                if v not in seen:      # the only shared state',
  '                    seen.add(v)',
  '                    nxt.append(v)',
  '        frontier = nxt                 # depth = len(levels)',
  '    return levels'
];
const CODE_LC = [
  'def shortest_path_binary_matrix(grid):',
  '    n = len(grid)',
  '    if grid[0][0] or grid[n-1][n-1]: return -1',
  '    # 8 directions -> one step cuts BOTH |dr| and |dc| by one,',
  '    h = lambda r, c: max(n-1-r, n-1-c)   # so Chebyshev, not Manhattan',
  '    pq = [(h(0, 0) + 1, 1, 0, 0)]        # (f, g, r, c)',
  '    best = {(0, 0): 1}',
  '    while pq:',
  '        f, g, r, c = heapq.heappop(pq)',
  '        if (r, c) == (n-1, n-1): return g',
  '        for nr, nc in eight_neighbours(r, c):',
  '            if grid[nr][nc] == 0 and g + 1 < best.get((nr, nc), INF):',
  '                best[(nr, nc)] = g + 1',
  '                heapq.heappush(pq, (g + 1 + h(nr, nc), g + 1, nr, nc))',
  '    return -1'
];

/* --------------------------------------------------------------- the tabs -- */
const CFG_DIJ = {
  h:() => 0, why:WHY_DIJ,
  opening:{note:{zh:'h = 0：優先權就是 g，這就是 Dijkstra', en:'h = 0: priority is just g - this is Dijkstra'},
           msg:{zh:'起點 S 在左邊，終點 G 在右邊，中間那排虛線是泥地（踏進去要 5）。' +
                   '先用 <b>h = 0</b> 跑：f 退化成 g，優先權只看「已經花了多少」——' +
                   '也就是原封不動的 Dijkstra。看它會翻多少格。',
                en:'S is on the left, G on the right, and the dashed strip in the middle is mud (5 to step onto). ' +
                   'First run with <b>h = 0</b>: f collapses to g and the priority only knows what has already been spent - ' +
                   'that is Dijkstra, unchanged. Watch how much of the map it turns over.'}},
  done:{note:(g, n) => ({zh:'成本 ' + g + '　·　展開 ' + n + ' 格', en:'cost ' + g + '  ·  ' + n + ' cells expanded'}),
        msg:(g, n) => ({zh:'到 G，成本 <b>' + g + '</b>，代價是展開了 <b>' + n + ' 格</b>——' +
                           '幾乎整張地圖。Dijkstra 沒有任何理由偏心，所以它公平地把每個方向都走了一遍。' +
                           '現在把同一份程式碼的 h 換掉，答案不該變，變的是這片藍色要縮多少。',
                        en:'G reached with cost <b>' + g + '</b>, after expanding <b>' + n + ' cells</b> - almost the whole map. ' +
                           'Dijkstra has no reason to favour any direction, so it fairly walks all of them. ' +
                           'Now swap h in the very same code: the answer must not change, only how much of this blue area survives.'})}
};
const CFG_AST = {
  h:manh, why:WHY_AST,
  opening:{note:{zh:'h = Manhattan：可採納且一致', en:'h = Manhattan: admissible and consistent'},
           msg:{zh:'同一份程式碼，只是 h 換成 Manhattan 距離。' +
                   'h(S) = ' + manh(SR, SC) + '，實際還要花 10——<b>猜得比真的小</b>，這叫可採納（admissible），' +
                   '保證答案還是最佳解。它同時也滿足 h(u) ≤ w(u,v) + h(v)（一致性），' +
                   '所以「pop 出來就定案」這件事仍然安全。',
                en:'Same code, but h is now the Manhattan distance. h(S) = ' + manh(SR, SC) +
                   ' while the true remaining cost is 10 - <b>the guess is below the truth</b>, which is what admissible means, ' +
                   'and it keeps the answer optimal. It also satisfies h(u) <= w(u,v) + h(v) (consistency), ' +
                   'so "popped means final" is still safe.'}},
  done:{note:(g, n) => ({zh:'成本 ' + g + '（一樣）　·　只展開 ' + n + ' 格', en:'cost ' + g + ' (identical)  ·  only ' + n + ' expanded'}),
        msg:(g, n) => ({zh:'成本還是 <b>' + g + '</b>，跟 Dijkstra 一模一樣，但只展開 <b>' + n + ' 格</b>，' +
                           '大約是它的四分之一。這就是 A* 的全部：<b>答案不變，工作量變少</b>。' +
                           '注意它一開始被 Manhattan 拐進泥地方向，但泥地那 5 把 g 撐高之後，' +
                           '繞道的 f 反而更小，搜尋自己修正了回來——因為 g 還在，猜錯得起。',
                        en:'The cost is still <b>' + g + '</b>, identical to Dijkstra, but only <b>' + n + ' cells</b> were expanded, ' +
                           'roughly a quarter. That is all A* is: <b>the same answer for less work</b>. ' +
                           'Notice it is initially lured toward the mud by Manhattan, but once the 5s inflate g the detour has the smaller f ' +
                           'and the search corrects itself - it can afford a bad guess because g is still in the sum.'})}
};
const CFG_W5 = {
  h:manh, weight:5, wrong:true, why:WHY_W5,
  opening:{note:{zh:'f = g + 5h：猜測被放大五倍', en:'f = g + 5h: the guess, amplified five times'},
           msg:{zh:'把 h 乘上 5。h 本身沒變，仍然是 Manhattan——但 <b>5h 已經高估了</b>：' +
                   'h(S) = ' + manh(SR, SC) + '，5h = ' + (5 * manh(SR, SC)) + '，實際只要 10。' +
                   '一旦高估，可採納性就沒了，A* 的最佳性保證跟著沒了。它不會報錯，只會給你一條比較貴的路。',
                en:'Multiply h by 5. The heuristic itself is unchanged - still Manhattan - but <b>5h now overestimates</b>: ' +
                   'h(S) = ' + manh(SR, SC) + ', 5h = ' + (5 * manh(SR, SC)) + ', while the true cost is 10. ' +
                   'The moment it overestimates, admissibility is gone and so is the optimality guarantee. Nothing raises an error; ' +
                   'you simply get a more expensive path.'}},
  done:{note:(g, n) => ({zh:'成本 ' + g + '（最佳解是 10）　·　只展開 ' + n + ' 格',
                         en:'cost ' + g + ' (optimal is 10)  ·  only ' + n + ' expanded'}),
        msg:(g, n) => ({zh:'展開只剩 <b>' + n + ' 格</b>，看起來很棒——但成本是 <b>' + g + '</b>，' +
                           '最佳解是 10。它一頭鑽進泥地，還很有自信。' +
                           '這是整個 A* 最值得記住的一句：<b>失敗不會當掉，只會安靜地變貴</b>。' +
                           '好消息是有界的：weight = w 的答案最多是最佳解的 w 倍，' +
                           '所以要快就明講要多快，不要以為自己還在算最短路。',
                        en:'Only <b>' + n + ' cells</b> expanded, which looks like a win - but the cost is <b>' + g +
                           '</b> against an optimum of 10. It dives into the mud, confidently. ' +
                           'This is the line worth remembering about A*: <b>failure does not crash, it quietly costs more</b>. ' +
                           'The good news is that it is bounded: with weight w the answer is at most w times the optimum, ' +
                           'so if you want speed, say how much - do not pretend you are still computing a shortest path.'})}
};
const CFG_GR = {
  h:manh, greedy:true, wrong:true, why:WHY_GR,
  opening:{note:{zh:'f = h：只看猜測，完全不看走過的成本', en:'f = h: only the guess, the bill so far is ignored'},
           msg:{zh:'最後一種：直接把 g 丟掉，f = h，這叫 greedy best-first。' +
                   '它每一步都選「看起來離 G 最近」的格子，泥地要五塊錢這件事完全不在它的排序裡。' +
                   '速度很快，而且——沒有任何誤差上界。',
                en:'The last variant: drop g entirely, f = h, known as greedy best-first. ' +
                   'Every step it takes whichever cell looks closest to G; the fact that mud costs five never enters the ordering. ' +
                   'It is fast, and it comes with no error bound whatsoever.'}},
  done:{note:(g, n) => ({zh:'成本 ' + g + '　·　最佳解 10　·　2.8 倍',
                         en:'cost ' + g + '  ·  optimal 10  ·  2.8x'}),
        msg:(g, n) => ({zh:'只展開 <b>' + n + ' 格</b>就到終點，但成本 <b>' + g + '</b>，是最佳解的 2.8 倍。' +
                           '跟 weighted A* 不一樣的是：weighted A* 至少還有「w 倍」的保證，' +
                           'greedy 什麼保證都沒有——地圖只要設計得壞一點，它可以差到任意倍。' +
                           '<b>g 才是那個把搜尋拉回現實的項</b>，h 只是給它一個方向。',
                        en:'Only <b>' + n + ' cells</b> expanded, but the cost is <b>' + g + '</b> - 2.8 times the optimum. ' +
                           'The difference from weighted A* matters: weighted A* still guarantees a factor of w, while greedy guarantees nothing at all. ' +
                           'Make the map slightly nastier and it can be arbitrarily bad. ' +
                           '<b>g is the term that keeps the search honest</b>; h only points it in a direction.'})}
};

const DAY_META = {
  title:{zh:'Day 19 · A* 與平行圖搜尋', en:'Day 19 · A* and parallel graph search'},
  sub:{zh:'f = g + h：走過的帳 + 對剩下的猜測。猜得保守 → 答案不變、工作量砍掉四分之三；猜過頭 → 不會當掉，只會安靜地給你一條貴的路。',
       en:'f = g + h: the bill so far plus a guess about the rest. Guess low and the answer is unchanged for a quarter of the work; guess high and nothing crashes - you just quietly get a costlier path.'},
  tabs:[
    {id:'astar', label:{zh:'Dijkstra → A*', en:'Dijkstra -> A*'},
     stage:{zh:'同一份程式碼，只換 h', en:'one implementation, two heuristics'},
     view:GVIEW, variants:[{zh:'h = 0（Dijkstra）', en:'h = 0 (Dijkstra)'}, {zh:'h = Manhattan（A*）', en:'h = Manhattan (A*)'}],
     idea:{zh:'Dijkstra 依 g 排序，A* 依 f = g + h 排序。只要 h 從不高估（admissible），A* 找到的還是最佳解；差別只在它願意忽略多少張地圖。格子裡的數字是 g。',
           en:'Dijkstra orders the frontier by g, A* by f = g + h. As long as h never overestimates (admissible), A* still returns the optimal path; the only difference is how much of the map it is willing to ignore. The number in each cell is g.'},
     legend:['hot', 'act', 'done', 'ok', ['#8fa3ac', {zh:'虛線＝泥地（踏進去 5）', en:'dashed = mud (costs 5)'}]],
     code:CODE_A,
     build:(v) => searchRun(v ? CFG_AST : CFG_DIJ)},
    {id:'wrong', label:{zh:'猜過頭', en:'when the guess is too big'},
     stage:{zh:'不會當掉，只會變貴', en:'no crash - just a worse path'},
     view:GVIEW, variants:[{zh:'weighted A*（w = 5）', en:'weighted A* (w = 5)'}, {zh:'greedy（f = h）', en:'greedy (f = h)'}],
     idea:{zh:'可採納性是「答案對不對」的分界線，不是效能選項。h 一旦高估，A* 仍然會回傳一條路、仍然不會報錯，只是那條路比較貴——weighted A* 至少有 w 倍的上界，greedy 連上界都沒有。',
           en:'Admissibility is the line between right and wrong, not a performance switch. Once h overestimates, A* still returns a path and still raises nothing; the path is simply worse. Weighted A* at least bounds the damage at w times optimal - greedy has no bound at all.'},
     legend:['hot', 'act', 'done', 'bad', ['#8fa3ac', {zh:'虛線＝泥地（踏進去 5）', en:'dashed = mud (costs 5)'}]],
     code:CODE_A,
     build:(v) => searchRun(v ? CFG_GR : CFG_W5)},
    {id:'delta', label:{zh:'Δ-stepping', en:'delta-stepping'},
     stage:{zh:'在 Dijkstra 與 Bellman-Ford 之間轉旋鈕', en:'a knob between Dijkstra and Bellman-Ford'},
     view:DV, variants:[{zh:'delta = 1', en:'delta = 1'}, {zh:'delta = 4', en:'delta = 4'}, {zh:'delta = 99', en:'delta = 99'}],
     idea:{zh:'Dijkstra 一次只肯 settle 一個點，這正是它難平行的原因。Δ-stepping 把距離分進寬度 delta 的 bucket，整桶一起放鬆：delta → 0 是 Dijkstra，delta → ∞ 是 Bellman-Ford。輪數變少的代價是同一個點被重算。',
           en:'Dijkstra settles exactly one vertex at a time, which is precisely what makes it hard to parallelise. Delta-stepping drops tentative distances into buckets of width delta and relaxes a whole bucket at once: delta -> 0 is Dijkstra, delta -> infinity is Bellman-Ford. Fewer rounds are paid for with repeated work.'},
     legend:['hot', 'done', ['#ff5c5c', {zh:'之前已 settle 過，重算', en:'settled before - repeated work'}]],
     code:CODE_D,
     build:(v) => deltaFrames([1, 4, 99][v])},
    {id:'bfs', label:{zh:'平行 BFS', en:'parallel BFS'},
     stage:{zh:'work 與 depth 是兩筆帳', en:'work and depth are two different bills'},
     view:GVIEW,
     idea:{zh:'把 BFS 從「一個 queue」改寫成「一層 frontier」，同一層的點彼此獨立，可以整批展開。總工作量還是 O(V+E)，真正變短的是深度——但最寬的那一層就是你能用上的核心數上限。',
           en:'Rewrite BFS from "a queue" into "a frontier": vertices in one level are independent and can be expanded together. The work stays O(V+E); what shrinks is depth - and the widest level is the ceiling on how many cores you can actually use.'},
     legend:['hot', 'done'],
     code:CODE_B,
     build:() => bfsFrames()},
    {id:'lc', label:{zh:'LeetCode 1091', en:'LeetCode 1091'},
     stage:{zh:'八方向的可採納 h 不是 Manhattan', en:'with 8 directions the admissible h is not Manhattan'},
     view:LVIEW, variants:[{zh:'h = 0（BFS）', en:'h = 0 (BFS)'}, {zh:'h = Chebyshev（A*）', en:'h = Chebyshev (A*)'}],
     idea:{zh:'Shortest Path in Binary Matrix：八方向、每格算一步。因為一步可以同時縮短列與行的差距，唯一不會高估的猜測是 Chebyshev max(|dr|,|dc|)；用 Manhattan 會把斜走算成兩步而高估，答案就不保證最短。',
           en:'Shortest Path in Binary Matrix: 8 directions, every cell counts as one step. Because a single step can shrink both the row and the column gap, the only guess that never overestimates is Chebyshev, max(|dr|,|dc|). Manhattan counts a diagonal as two steps, overestimates, and the answer stops being guaranteed shortest.'},
     legend:['hot', 'act', 'done', 'ok', ['#0a6b74', {zh:'牆', en:'wall'}]],
     code:CODE_LC,
     build:(v) => lcFrames(!!v)}
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
