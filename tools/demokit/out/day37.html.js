// DAY: 37
// TITLE_ZH: Day 37 — 狀態壓縮 DP 與 Held-Karp：用一個整數當集合索引，精確解出 TSP
// TITLE_EN: State-compression DP and Held-Karp - indexing a table by a set to solve the TSP exactly
// SUB_ZH: Day 33 到 36 的表格索引一直在換形狀：前綴、區間、子樹。今天索引變成一個集合——把「哪些城市已經走過」壓成一個整數的二進位位元。集合當索引之後，填表順序反而是免費的，因為加一個位元一定讓整數變大；真正難的是狀態要放什麼。TSP 的無聲錯誤就在這裡：只記 dp[mask] 少了「現在站在哪」，程式照跑、答案照給，只是它算出來的是最小生成樹，永遠比最佳路線小一點。
// SUB_EN: From day 33 to day 36 the shape of the table index kept changing - a prefix, an interval, a subtree. Today it becomes a set, packed into the binary digits of a single integer. Once a set is the index the fill order comes for free, because switching a bit on can only make the integer larger; the hard part moves into deciding what the state must remember. That is exactly where the silent bug of the day lives: dropping the "where am I standing now" half of the state leaves a recurrence that still runs, still returns a number, and quietly computes a minimum spanning tree instead of a tour.
// FOLDER: day%2037%20-%20bitmask%20dp%20and%20tsp
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const fmt = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const bitsOf = (m, n) => { const o = []; for (let i = 0; i < n; i++) if (m >> i & 1) o.push(i); return o; };
const pcount = m => { let c = 0; while (m){ c += m & 1; m >>= 1; } return c; };
const maskStr = (m, n) => { let s = ''; for (let i = n - 1; i >= 0; i--) s += (m >> i & 1); return s; };
const INF = 1e9;

/* the five-stop instance used by the first two tabs - a subset of the eight
   stops in bitmask_dp.py, small enough that every state fits on one screen */
const N5 = ['TPE', 'TXG', 'SML', 'TNN', 'HUN'];
const D5 = [[0, 132, 149, 264, 116], [132, 0, 42, 137, 96], [149, 42, 0, 116, 74],
            [264, 137, 116, 0, 180], [116, 96, 74, 180, 0]];
const XY5 = [[157.6, 336.3], [68.7, 238.7], [90.9, 203.1], [21.2, 109.9], [162.6, 220.9]];

/* the full eight-stop instance, for the greedy tab */
const N8 = ['TPE', 'HSZ', 'TXG', 'SML', 'TNN', 'KHH', 'TTT', 'HUN'];
const XY8 = [[157.6, 336.3], [98.0, 312.0], [68.7, 238.7], [90.9, 203.1],
             [21.2, 109.9], [30.3, 69.9], [115.1, 84.4], [162.6, 220.9]];
const D8 = [[0,64,132,149,264,295,255,116],[64,0,79,109,216,251,228,112],
            [132,79,0,42,137,173,161,96],[149,109,42,0,116,146,121,74],
            [264,216,137,116,0,41,97,180],[295,251,173,146,41,0,86,201],
            [255,228,161,121,97,86,0,145],[116,112,96,74,180,201,145,0]];

/* map real coordinates into a rectangle - note SVG y grows downward, so north
   (a large latitude) has to land on a SMALL y */
function mapper(XY, x0, y0, w, h){
  const xs = XY.map(p => p[0]), ys = XY.map(p => p[1]);
  const xa = Math.min.apply(null, xs), xb = Math.max.apply(null, xs);
  const ya = Math.min.apply(null, ys), yb = Math.max.apply(null, ys);
  return XY.map(p => [x0 + (p[0] - xa) / (xb - xa) * w, y0 + (yb - p[1]) / (yb - ya) * h]);
}
function cityNodes(P, names, states, r){
  return P.map((p, i) => S.c(p[0], p[1], r || .26, states[i] || 'soft', names[i], {fs:.26}));
}
function pathEdges(P, order, st, opt){
  opt = opt || {};
  const out = [];
  for (let k = 0; k + 1 < order.length; k++)
    out.push(S.e(P[order[k]][0], P[order[k]][1], P[order[k + 1]][0], P[order[k + 1]][1],
                 {s:st, arrow:opt.arrow === true, pad:.30, w:opt.w || .07,
                  lab:opt.labs ? String(opt.D[order[k]][order[k + 1]]) : null, fs:.25}));
  return out;
}

/* ======================================================================== *
 * Tab 1 - the state: same set of cities, different bill
 * ======================================================================== */
const CODE_STATE = [
  'dp = [[INF] * n for _ in range(1 << n)]   # dp[mask][last]',
  'dp[1][0] = 0                              # started at city 0',
  '',
  'for mask in range(1 << n):                # already a valid fill order',
  '    for last in range(n):',
  '        if dp[mask][last] == INF: continue',
  '        for nxt in range(n):',
  '            if mask >> nxt & 1: continue',
  '            cost = dp[mask][last] + D[last][nxt]   # needs `last`',
  '            nmask = mask | 1 << nxt',
  '            if cost < dp[nmask][nxt]:',
  '                dp[nmask][nxt] = cost'
];

const SUB3 = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]];
function perms3(a){
  const out = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++)
    if (i !== j && j !== k && i !== k) out.push([a[i], a[j], a[k]]);
  return out;
}
const setStr = s => '{' + s.map(i => N5[i]).join(',') + '}';
const routeStr = p => 'TPE → ' + p.map(i => N5[i]).join(' → ');
const routeCost = p => D5[0][p[0]] + D5[p[0]][p[1]] + D5[p[1]][p[2]];

const RY0 = 1.75, RDY = 0.50;
function routeShapes(sub, perms, colour){
  const out = [S.t(0.55, 1.20, {zh:'集合 ' + setStr(sub) + '：6 種走法',
                                en:'set ' + setStr(sub) + ': 6 orders'},
                   {c:COL.purpleL, fs:.30, anchor:'start'})];
  perms.forEach((p, i) => {
    out.push(S.t(0.55, RY0 + i * RDY, routeStr(p), {c:colour(p, i), fs:.30, anchor:'start'}));
    out.push(S.t(4.40, RY0 + i * RDY, fmt(routeCost(p)) + ' km',
                 {c:colour(p, i), fs:.30, anchor:'end'}));
  });
  return out;
}
function stateBox(x, y, w, h, sty, last, cost, extra){
  const o = [S.r(x, y, w, h, sty, 'last = ' + N5[last] + '   ' + fmt(cost) + ' km', {fs:.29})];
  if (extra) o.push(S.t(x + w / 2, y + h + .32, extra, {c:COL.grey, fs:.26}));
  return o;
}

function stateFrames(v){
  const F = new Frames();
  const byMask = v === 1;
  F.push({shapes:[
      S.t(4.9, .80, {zh:'24 條前綴，壓成幾個狀態？',
                     en:'24 prefixes - how many states?'},
          {c:COL.tealL, fs:.42}),
      S.t(4.9, 1.70, {zh:'從 TPE 出發再走三個城市：4 × 3 × 2 = 24 種走法',
                      en:'From TPE, three more cities: 4 x 3 x 2 = 24 orders'},
          {c:COL.pale, fs:.31}),
      S.t(4.9, 2.40, byMask
          ? {zh:'這一版只用 mask：只問走過哪些，不問現在在哪',
             en:'Indexed by mask alone: which cities, not where I stand'}
          : {zh:'這一版用 (mask, last)：走過哪些，現在在哪',
             en:'Indexed by (mask, last): which cities, and where I stand'},
          {c:COL.pale, fs:.30}),
      S.t(4.9, 3.30, {zh:'覆蓋同一組城市的兩條走法，可以只留便宜的嗎？',
                      en:'Two orders over the same set - keep only the cheaper?'},
          {c:COL.orangeL, fs:.32}),
      S.t(4.9, 4.05, {zh:'要看接下來那一段還需不需要知道更多事',
                      en:'Depends on what pricing the next hop needs to know'},
          {c:COL.grey, fs:.29}),
      S.t(4.9, 5.20, {zh:'五個停靠點：TPE、TXG、SML、TNN、HUN',
                      en:'Five stops: TPE, TXG, SML, TNN, HUN (km, straight line)'},
          {c:COL.grey, fs:.27})],
    panels:[{lbl:{zh:'索引', en:'index'},
             chips:[chip(byMask ? 'dp[mask]' : 'dp[mask][last]', byMask ? 'bad' : 'hot')]}],
    line:0,
    msg:byMask
      ? {zh:'先看一個看起來很合理的省法：既然做 DP 就是要合併重複的子問題，那「走過同一組城市」聽起來就該是同一個子問題。這一版就照這個直覺寫，把 24 條前綴壓成 4 個 dp[mask]。'
        , en:'Start from a reasonable-sounding economy. The whole point of DP is to merge repeated subproblems, and "has visited the same set of cities" certainly sounds like the same subproblem. This run follows that instinct and squeezes the 24 prefixes into 4 dp[mask] entries.'}
      : {zh:'Day 33 到 36 的表格索引是前綴、區間、子樹；今天索引是一個集合，而集合直接用一個整數的位元表示。先把問題問清楚：24 條前綴，到底可以合併成幾個狀態？'
        , en:'Days 33 to 36 indexed the table by a prefix, an interval, a subtree. Today the index is a set, written straight into the bits of an integer. First get the question right: those 24 prefixes collapse into how many states?'}});

  SUB3.forEach((sub, si) => {
    const perms = perms3(sub);
    const lasts = sub.slice().sort((a, b) => a - b);
    const best = {};
    perms.forEach(p => { const L = p[2];
      if (best[L] == null || routeCost(p) < best[L]) best[L] = routeCost(p); });
    const cheapest = Math.min.apply(null, perms.map(routeCost));

    F.push({shapes:routeShapes(sub, perms, () => COL.pale).concat([
        S.t(7.1, 1.20, {zh:'先不合併', en:'not merged yet'}, {c:COL.grey, fs:.30})]),
      panels:[{lbl:{zh:'這一組', en:'this set'}, chips:[chip(setStr(sub), 'hot')]},
              {lbl:{zh:'走法', en:'orders'}, chips:[chip('6', 'dim')]}],
      line:3,
      msg:{zh:'覆蓋 ' + setStr(sub) + ' 這一組城市有 6 種走法，價錢從 ' + fmt(cheapest) +
              ' km 到 ' + fmt(Math.max.apply(null, perms.map(routeCost))) +
              ' km 都有。接下來要決定的是：這 6 條裡面，哪些可以互相取代？',
           en:'Six orders cover the set ' + setStr(sub) + ', priced anywhere from ' +
              fmt(cheapest) + ' km to ' + fmt(Math.max.apply(null, perms.map(routeCost))) +
              ' km. The decision to make is which of the six may stand in for one another.'}});

    if (byMask){
      const sh = routeShapes(sub, perms, p => routeCost(p) === cheapest ? COL.tealL : '#54707c');
      sh.push(S.r(5.9, 2.55, 3.4, .80, 'bad', 'dp[mask] = ' + fmt(cheapest),
                  {fs:.34, top:setStr(sub), topc:COL.grey}));
      sh.push(S.t(7.6, 3.75, {zh:'6 條 → 1 個狀態', en:'6 orders -> 1 state'}, {c:COL.red, fs:.30}));
      sh.push(S.t(7.6, 4.25, {zh:'但現在站在哪？', en:'but standing where?'}, {c:COL.red, fs:.30}));
      F.push({shapes:sh,
        panels:[{lbl:{zh:'這一組', en:'this set'}, chips:[chip(setStr(sub), 'hot')]},
                {lbl:{zh:'留下', en:'kept'}, chips:[chip('dp = ' + fmt(cheapest), 'bad')]}],
        line:0,
        msg:{zh:'只用 mask 的話，6 條全部壓成一個數字 ' + fmt(cheapest) +
                ' km，留下的是 ' + routeStr(perms.filter(p => routeCost(p) === cheapest)[0]) +
                '。省得很漂亮——直到要加下一段為止：下一段的長度是從「最後停的那個城市」算起的，而那件事剛剛被丟掉了。',
             en:'With the mask alone all six collapse into the single number ' + fmt(cheapest) +
                ' km, kept from ' + routeStr(perms.filter(p => routeCost(p) === cheapest)[0]) +
                '. A beautiful saving - right up to the next hop, whose length is measured from the city the prefix ended on, and that is precisely what was just thrown away.'}});
    } else {
      const sh = routeShapes(sub, perms, p => routeCost(p) === best[p[2]] ? COL.tealL : '#54707c');
      lasts.forEach((L, i) => {
        stateBox(5.9, 1.60 + i * 1.25, 3.4, .78, 'ok', L, best[L]).forEach(s => sh.push(s));
      });
      sh.push(S.t(7.6, 5.55, {zh:'6 條 → 3 個狀態', en:'6 orders -> 3 states'}, {c:COL.tealL, fs:.32}));
      F.push({shapes:sh,
        panels:[{lbl:{zh:'這一組', en:'this set'}, chips:[chip(setStr(sub), 'hot')]},
                {lbl:{zh:'狀態', en:'states'},
                 chips:lasts.map(L => chip(N5[L] + ' ' + fmt(best[L]), 'ok'))}],
        line:11,
        msg:{zh:'依照「最後停在哪」分成三組，每組留下便宜的那一條。' +
                '為什麼這樣合併是安全的？因為接下來要付的每一段路，只跟終點城市有關，跟前面怎麼繞完全無關。' +
                '所以同組之內貴的那條永遠贏不了，可以放心丟掉。',
             en:'Group the six by the city they end on and keep the cheaper of each pair. ' +
                'Why is that merge safe? Because every kilometre still to be paid depends only on the city the prefix ends at, never on how it got there. ' +
                'Inside a group the dearer route can never come back to win, so it can be dropped without a second thought.'}});
    }
  });

  if (byMask){
    F.push({shapes:[
        S.t(4.9, 1.10, {zh:'下一步走到 TNN，要加幾公里？',
                        en:'Next hop to TNN - how many km?'},
            {c:COL.orangeL, fs:.38}),
        S.r(1.0, 2.00, 3.0, .80, 'bad', 'dp[mask] = 290',
            {fs:.34, top:setStr(SUB3[0]), topc:COL.grey}),
        S.t(2.5, 3.35, {zh:'表裡只有這個數字', en:'this number is all the table has'},
            {c:COL.grey, fs:.27}),
        S.t(6.9, 2.50, 'D[?][TNN]', {c:COL.red, fs:.46}),
        S.t(6.9, 3.35, {zh:'? 可能是 TXG 或 SML', en:'? could be TXG or SML'},
            {c:COL.red, fs:.29}),
        S.t(4.9, 4.60, {zh:'唯一寫得出來的是「接到最近的已走訪城市」',
                        en:'All that stays writable: join to the nearest visited city'},
            {c:COL.pale, fs:.29}),
        S.t(4.9, 5.20, {zh:'那長出來的是一棵樹，不是一條環',
                        en:'What that grows is a tree, not a tour'},
            {c:COL.pale, fs:.30})],
      panels:[{lbl:{zh:'缺少的資訊', en:'the missing fact'},
               chips:[chip('last', 'bad')]}],
      line:8,
      msg:{zh:'這就是整天最重要的一格：D[last][nxt] 這一行需要 last，而 dp[mask] 裡根本沒有 last。' +
              '注意程式不會壞掉，也不會丟例外——它只會退化成另一個問題的解答，而那個答案永遠比正確答案小一點，看起來完全合理。',
           en:'This is the whole day in one frame: the line D[last][nxt] needs last, and dp[mask] has no last in it. ' +
              'Nothing crashes, nothing raises - the code simply degenerates into the answer to a different question, and that answer is always a little smaller than the truth and entirely plausible.'}});
  } else {
    F.push({shapes:[
        S.t(4.9, 1.05, {zh:'24 條前綴 → 12 個狀態', en:'24 prefixes -> 12 states'},
            {c:COL.tealL, fs:.46}),
        S.t(4.9, 2.00, {zh:'合併後要付的帳不變：剩下的路只看終點城市',
                        en:'The bill left to pay only looks at the final city'},
            {c:COL.pale, fs:.32}),
        S.t(4.9, 2.90, {zh:'n 個城市：(n-1)! 條路線 對上 2ⁿ·n 個狀態',
                        en:'n cities: (n-1)! routes against 2^n * n states'},
            {c:COL.pale, fs:.32}),
        S.r(1.3, 3.55, 3.0, .85, 'bad', '(n-1)!', {fs:.40, sub:{zh:'列舉所有路線', en:'enumerate every route'}}),
        S.r(5.5, 3.55, 3.0, .85, 'ok', '2ⁿ · n²', {fs:.40, sub:{zh:'Held-Karp 的表', en:'the Held-Karp table'}}),
        S.t(4.9, 5.55, {zh:'n = 15 時：6,227,020,800 對上 7,372,800。',
                        en:'At n = 15: 6,227,020,800 against 7,372,800.'},
            {c:COL.orangeL, fs:.33})],
      panels:[{lbl:{zh:'前綴', en:'prefixes'}, chips:[chip('24', 'dim')]},
              {lbl:{zh:'狀態', en:'states'}, chips:[chip('12', 'ok')]}],
      line:11,
      msg:{zh:'把四組加起來：24 條前綴只剩 12 個狀態，而且完全沒有近似、沒有放棄任何可能的路線。' +
              '這就是狀態壓縮 DP 的全部內容——指數還在，但底數從階乘掉到 2ⁿ，n = 15 的時候差了將近一千倍。',
           en:'Add the four groups up: the 24 prefixes are down to 12 states, with nothing approximated and no route quietly discarded. ' +
              'That is all state-compression DP is - the exponential stays, but the base drops from a factorial to 2^n, which at n = 15 is a factor of nearly a thousand.'}});
  }
  return F.list;
}

/* ======================================================================== *
 * Tab 2 - Held-Karp: one row per subset, one column per "where I am now"
 * ======================================================================== */
const CODE_HK = [
  'def held_karp(D):',
  '    n = len(D); full = (1 << n) - 1',
  '    dp  = [[INF] * n for _ in range(1 << n)]',
  '    par = [[-1] * n for _ in range(1 << n)]',
  '    dp[1][0] = 0                       # at city 0, nothing else seen',
  '    for mask in range(1 << n):         # ascending IS a fill order',
  '        for last in range(n):',
  '            if dp[mask][last] == INF: continue',
  '            for nxt in range(n):',
  '                if mask >> nxt & 1: continue',
  '                nm = mask | 1 << nxt',
  '                c  = dp[mask][last] + D[last][nxt]',
  '                if c < dp[nm][nxt]:',
  '                    dp[nm][nxt], par[nm][nxt] = c, last',
  '    return min(dp[full][j] + D[j][0] for j in range(1, n))'
];

const P5 = mapper(XY5, 0.55, 1.35, 3.30, 3.95);
const GX = 5.05, GW = 0.80, GY = 1.30, GH = 0.28;

function heldKarp(D){
  const n = D.length, full = (1 << n) - 1;
  const dp = [], par = [];
  for (let m = 0; m <= full; m++){ dp.push(new Array(n).fill(INF)); par.push(new Array(n).fill(-1)); }
  dp[1][0] = 0;
  for (let mask = 0; mask <= full; mask++)
    for (let last = 0; last < n; last++){
      if (dp[mask][last] >= INF) continue;
      for (let nxt = 0; nxt < n; nxt++){
        if (mask >> nxt & 1) continue;
        const nm = mask | 1 << nxt, c = dp[mask][last] + D[last][nxt];
        if (c < dp[nm][nxt]){ dp[nm][nxt] = c; par[nm][nxt] = last; }
      }
    }
  let best = INF, end = -1;
  for (let j = 1; j < n; j++) if (dp[full][j] + D[j][0] < best){ best = dp[full][j] + D[j][0]; end = j; }
  const tour = []; let m = full, c = end;
  while (c !== -1){ tour.push(c); const p = par[m][c]; m ^= 1 << c; c = p; }
  tour.reverse();
  return {dp:dp, par:par, best:best, tour:tour};
}
const HK5 = heldKarp(D5);

function pathTo(par, mask, last){
  const out = []; let m = mask, c = last;
  while (c !== -1){ out.push(c); const p = par[m][c]; m ^= 1 << c; c = p; }
  return out.reverse();
}
/* the 16 masks that contain city 0, cheapest-to-build first */
const MASKS5 = (function(){
  const a = [];
  for (let m = 1; m < 32; m++) if (m & 1) a.push(m);
  a.sort((x, y) => pcount(x) - pcount(y) || x - y);
  return a;
})();

function gridShapes(upto, hotMask, hotLast){
  const out = [];
  for (let j = 0; j < 5; j++)
    out.push(S.t(GX + j * GW + GW / 2 - .03, GY - .18, N5[j], {c:COL.tealL, fs:.24}));
  out.push(S.t(GX - .18, GY - .18, 'mask', {c:COL.grey, fs:.24, anchor:'end'}));
  MASKS5.forEach((m, r) => {
    const y = GY + r * GH;
    out.push(S.t(GX - .18, y + GH * .70, maskStr(m, 5),
                 {c:m === hotMask ? COL.orangeL : COL.grey, fs:.22, anchor:'end'}));
    for (let j = 0; j < 5; j++){
      const v = HK5.dp[m][j], shown = r <= upto && v < INF;
      let sty = 'ghost';
      if (shown) sty = (m === hotMask && j === hotLast) ? 'hot' : (m === hotMask ? 'act' : 'soft');
      out.push(S.r(GX + j * GW, y, GW - .06, GH - .05, sty, shown ? String(v) : '', {fs:.21, rx:.05}));
    }
  });
  return out;
}

function hkFrames(){
  const F = new Frames();
  F.push({shapes:cityNodes(P5, N5, {0:'ok'}).concat([
      S.t(2.20, .72, {zh:'五個停靠點', en:'five stops'}, {c:COL.tealL, fs:.34}),
      S.t(2.20, 1.14, {zh:'從 TPE 出發，回到 TPE', en:'start at TPE, return to TPE'},
          {c:COL.grey, fs:.26}),
      S.t(7.05, .72, 'dp[mask][last]', {c:COL.tealL, fs:.34})],
      gridShapes(-1, -1, -1)),
    panels:[{lbl:{zh:'已填狀態', en:'states filled'}, chips:[chip('0 / 33', 'dim')]},
            {lbl:{zh:'路線總數', en:'routes in total'}, chips:[chip('24', 'dim')]}],
    line:2,
    msg:{zh:'一列是一個「已走過的集合」，一欄是「現在站在哪」。只有含 TPE 的 16 個 mask 有意義，因為旅程一定從 TPE 開始。灰色的格子是不可能的狀態：last 不在 mask 裡面。',
         en:'A row is a set of cities already visited, a column is the city I am standing on. Only the 16 masks containing TPE mean anything, because the trip always starts there. The greyed cells are impossible states, where last is not a member of mask.'}});

  MASKS5.forEach((m, r) => {
    let bestLast = -1, bestV = INF;
    for (let j = 0; j < 5; j++) if (HK5.dp[m][j] < bestV){ bestV = HK5.dp[m][j]; bestLast = j; }
    const vis = bitsOf(m, 5);
    const st = {}; vis.forEach(i => st[i] = 'done'); if (bestLast >= 0) st[bestLast] = 'hot';
    const path = bestLast >= 0 ? pathTo(HK5.par, m, bestLast) : [];
    const filled = MASKS5.slice(0, r + 1)
      .reduce((a, mm) => a + HK5.dp[mm].filter(v => v < INF).length, 0);
    const row = [];
    for (let j = 0; j < 5; j++) if (HK5.dp[m][j] < INF) row.push(chip(N5[j] + ' ' + HK5.dp[m][j], j === bestLast ? 'hot' : 'dim'));
    F.push({shapes:cityNodes(P5, N5, st)
        .concat(pathEdges(P5, path, 'ok', {arrow:true, labs:true, D:D5}))
        .concat(gridShapes(r, m, bestLast))
        .concat([S.t(2.20, .72, setStr(vis), {c:COL.orangeL, fs:.28}),
                 S.t(2.20, 1.14, {zh:'最便宜的前綴停在 ' + N5[bestLast],
                                  en:'cheapest prefix ends at ' + N5[bestLast]},
                     {c:COL.grey, fs:.26}),
                 S.t(7.05, .72, 'mask = ' + maskStr(m, 5) + '  (' + m + ')',
                     {c:COL.tealL, fs:.30})]),
      panels:[{lbl:{zh:'已填狀態', en:'states filled'}, chips:[chip(filled + ' / 33', 'ok')]},
              {lbl:{zh:'這一列', en:'this row'}, chips:row}],
      line:r === 0 ? 4 : 13,
      msg:r === 0
        ? {zh:'起點那一格是唯一手填的：dp[00001][TPE] = 0。剩下的每一格都只由比它小的整數決定——注意 mask | 1 << nxt 一定比 mask 大，所以由小到大跑一遍 range(1 << n) 本身就是一個合法的填表順序。Day 36 那張三角形的表要自己推順序，今天不用。',
           en:'The starting cell is the only one written by hand: dp[00001][TPE] = 0. Every other cell is decided by strictly smaller integers - mask | 1 << nxt is always larger than mask, so a plain ascending range(1 << n) already is a valid fill order. Yesterday the triangular table needed its order derived; today it comes for free.'}
        : {zh:'這一列填的是「走完 ' + setStr(vis) + '」的所有可能終點。' +
              (pcount(m) > 2
                ? '同一列裡的數字彼此不能比較，因為它們停在不同的城市，接下來要付的錢也不同；能比較的只有同一格的兩種到達方式。'
                : '每一格記的是到目前為止最便宜的走法，左邊的地圖畫的是這一列最便宜的那一條。'),
           en:'This row covers every possible finishing city for the set ' + setStr(vis) + '. ' +
              (pcount(m) > 2
                ? 'The numbers inside one row are not comparable with one another, because they end on different cities and therefore owe different amounts from here on; only two ways of reaching the same cell may be compared.'
                : 'Each cell keeps the cheapest prefix found so far, and the map on the left draws the cheapest one in this row.')}});
  });

  const tour = HK5.tour.concat([0]);
  const stAll = {}; for (let i = 0; i < 5; i++) stAll[i] = 'ok';
  F.push({shapes:cityNodes(P5, N5, stAll)
      .concat(pathEdges(P5, tour, 'ok', {arrow:true, labs:true, D:D5}))
      .concat(gridShapes(15, 31, -1))
      .concat([S.t(2.20, .78, {zh:'最佳環線 ' + fmt(HK5.best) + ' km',
                               en:'optimal tour ' + fmt(HK5.best) + ' km'}, {c:COL.tealL, fs:.36}),
               S.t(2.20, 5.90, tour.map(i => N5[i]).join('-'), {c:COL.pale, fs:.24}),
               S.t(7.05, .72, {zh:'加上回家的那一段', en:'plus the hop home'},
                   {c:COL.tealL, fs:.28})]),
    panels:[{lbl:{zh:'已填狀態', en:'states filled'}, chips:[chip('33 / 33', 'ok')]},
            {lbl:{zh:'答案', en:'answer'}, chips:[chip(fmt(HK5.best) + ' km', 'ok')]}],
    line:14,
    msg:{zh:'最後一列（走完全部五個城市）的每一格都還缺一段：回 TPE。把 D[j][TPE] 加回去取最小，就是 ' +
            fmt(HK5.best) + ' km，跟暴力枚舉 24 條路線的答案一模一樣——差別是這裡只碰了 33 個狀態，而且這個差距會隨城市數指數地拉開。',
         en:'Every cell of the last row, where all five cities are done, is still missing one hop: the drive back to TPE. Add D[j][TPE] and take the minimum to get ' +
            fmt(HK5.best) + ' km, exactly the answer brute force finds over all 24 routes - except that only 33 states were ever touched, and that gap widens exponentially with the number of cities.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 3 - two wrong answers that never raise: a tree, and a greedy tour
 * ======================================================================== */
const CODE_WRONG = [
  'def tsp_mask_only(D):            # the state forgets `last`',
  '    dp = [INF] * (1 << n); dp[1] = 0',
  '    for mask in range(1 << n):',
  '        for nxt in range(n):',
  '            if mask >> nxt & 1: continue',
  '            hop = min(D[i][nxt] for i in members(mask))',
  '            nm  = mask | 1 << nxt',
  '            dp[nm] = min(dp[nm], dp[mask] + hop)',
  '    return dp[(1 << n) - 1]      # = the MST weight, not a tour',
  '',
  'def nearest_neighbour(D):        # the state is only `here`',
  '    seen, cur, total = {0}, 0, 0',
  '    for _ in range(n - 1):',
  '        nxt = min((j for j in range(n) if j not in seen),',
  '                  key=lambda j: D[cur][j])',
  '        total += D[cur][nxt]; seen.add(nxt); cur = nxt',
  '    return total + D[cur][0]     # a real tour, just not the best one'
];

const P8 = mapper(XY8, 1.05, 1.30, 2.60, 4.30);

function maskOnlyRun(D){
  const n = D.length, full = (1 << n) - 1;
  const dp = new Array(1 << n).fill(INF), par = new Array(1 << n).fill(null);
  dp[1] = 0;
  for (let mask = 0; mask <= full; mask++){
    if (dp[mask] >= INF) continue;
    for (let nxt = 0; nxt < n; nxt++){
      if (mask >> nxt & 1) continue;
      let hop = INF, from = -1;
      bitsOf(mask, n).forEach(i => { if (D[i][nxt] < hop){ hop = D[i][nxt]; from = i; } });
      const nm = mask | 1 << nxt;
      if (dp[mask] + hop < dp[nm]){ dp[nm] = dp[mask] + hop; par[nm] = [mask, nxt, from, hop]; }
    }
  }
  const steps = []; let m = full;
  while (par[m]){ steps.push(par[m]); m = par[m][0]; }
  steps.reverse();
  return {total:dp[full], steps:steps};
}
function nnRun(D){
  const n = D.length, seen = [0], tour = [0];
  let cur = 0, total = 0;
  const hops = [];
  for (let s = 0; s < n - 1; s++){
    let best = INF, nxt = -1;
    for (let j = 0; j < n; j++) if (seen.indexOf(j) < 0 && D[cur][j] < best){ best = D[cur][j]; nxt = j; }
    hops.push([cur, nxt, best]); total += best; seen.push(nxt); tour.push(nxt); cur = nxt;
  }
  hops.push([cur, 0, D[cur][0]]); total += D[cur][0];
  return {total:total, hops:hops, tour:tour};
}
const MO5 = maskOnlyRun(D5);
const NN8 = nnRun(D8);
const HK8 = heldKarp(D8);

function wrongFrames(v){
  const F = new Frames();
  if (v === 0){
    F.push({shapes:cityNodes(P5, N5, {0:'hot'}).concat([
        S.t(2.20, .72, {zh:'只用 dp[mask]', en:'dp[mask] only'}, {c:COL.red, fs:.32}),
        S.t(2.20, 1.14, {zh:'接到最近的已走訪城市',
                         en:'join to the nearest visited city'}, {c:COL.grey, fs:.25}),
        S.t(6.8, 1.35, {zh:'少了 last，只寫得出',
                        en:'Without last, all one can write is'},
            {c:COL.pale, fs:.28}),
        S.t(6.8, 2.05, 'min(D[i][nxt] for i in mask)', {c:COL.red, fs:.32}),
        S.t(6.8, 2.80, {zh:'合法、會終止、會回傳數字',
                        en:'Legal. Terminates. Returns a number.'},
            {c:COL.pale, fs:.28}),
        S.t(6.8, 3.45, {zh:'只是在回答另一個問題',
                        en:'It just answers a different question.'}, {c:COL.orangeL, fs:.29})]),
      panels:[{lbl:{zh:'已走訪', en:'visited'}, chips:[chip('TPE', 'hot')]},
              {lbl:{zh:'累計', en:'running total'}, chips:[chip('0 km', 'dim')]}],
      line:1,
      msg:{zh:'把 last 從狀態裡拿掉之後，程式並沒有壞掉，只是被迫改問一個比較弱的問題：「把下一個城市接到目前已經走過的城市裡最近的那一個」。看著它跑四步，就知道它在蓋什麼。',
           en:'Take last out of the state and nothing breaks; the code is merely forced into a weaker question - attach the next city to whichever already-visited city is closest. Watch four steps of it and what it is building becomes obvious.'}});
    let mask = 1, total = 0;
    const edges = [];
    MO5.steps.forEach((s, k) => {
      const nxt = s[1], from = s[2], hop = s[3];
      mask |= 1 << nxt; total += hop; edges.push([from, nxt]);
      const st = {}; bitsOf(mask, 5).forEach(i => st[i] = 'done'); st[nxt] = 'bad'; st[from] = 'act';
      const esh = edges.map(e => S.e(P5[e[0]][0], P5[e[0]][1], P5[e[1]][0], P5[e[1]][1],
                                     {s:'bad', arrow:false, pad:.30, w:.07,
                                      lab:String(D5[e[0]][e[1]]), fs:.25}));
      F.push({shapes:cityNodes(P5, N5, st).concat(esh).concat([
          S.t(2.20, .78, N5[from] + ' → ' + N5[nxt] + '  +' + hop + ' km', {c:COL.red, fs:.32}),
          S.t(6.8, 1.35, 'dp[' + maskStr(mask, 5) + '] = ' + total, {c:COL.orangeL, fs:.36}),
          S.t(6.8, 2.25, {zh:'接到 ' + N5[from] + '，不是上一步的終點',
                          en:'attached to ' + N5[from] + ', not the last stop'},
              {c:COL.pale, fs:.28}),
          S.t(6.8, 2.90, {zh:'所以這根本不是一條線',
                          en:'so this is not a line at all'}, {c:COL.red, fs:.29})]),
        panels:[{lbl:{zh:'已走訪', en:'visited'},
                 chips:bitsOf(mask, 5).map(i => chip(N5[i], i === nxt ? 'bad' : 'dim'))},
                {lbl:{zh:'累計', en:'running total'}, chips:[chip(total + ' km', 'bad')]}],
        line:5,
        msg:{zh:'第 ' + (k + 1) + ' 步把 ' + N5[nxt] + ' 接進來，付的是它到已走訪集合的最短距離 ' + hop +
                ' km，來源是 ' + N5[from] + '。在真正的旅程裡這一段必須從上一步停的地方出發，但這個 dp 根本不知道上一步停在哪裡，所以它可以從集合裡任何一點長出新的邊。',
             en:'Step ' + (k + 1) + ' brings ' + N5[nxt] + ' in at ' + hop +
                ' km, its shortest distance to the visited set, growing out of ' + N5[from] +
                '. On a real trip that hop would have to start where the previous one ended, but this dp has no idea where that was, so it is free to sprout an edge from anywhere in the set.'}});
    });
    const stAll = {}; for (let i = 0; i < 5; i++) stAll[i] = 'bad';
    const esh = edges.map(e => S.e(P5[e[0]][0], P5[e[0]][1], P5[e[1]][0], P5[e[1]][1],
                                   {s:'bad', arrow:false, pad:.30, w:.07}));
    const tour = HK5.tour.concat([0]);
    const P5b = mapper(XY5, 5.75, 1.35, 3.30, 3.95);
    const stOk = {}; for (let i = 0; i < 5; i++) stOk[i] = 'ok';
    F.push({shapes:cityNodes(P5, N5, stAll).concat(esh)
        .concat(cityNodes(P5b, N5, stOk))
        .concat(pathEdges(P5b, tour, 'ok', {arrow:false}))
        .concat([
          S.t(2.20, .78, fmt(MO5.total) + ' km', {c:COL.red, fs:.42}),
          S.t(2.20, 5.85, {zh:'一棵樹：沒有人回家',
                           en:'a tree: nobody comes home'},
              {c:COL.red, fs:.26}),
          S.t(7.40, .78, fmt(HK5.best) + ' km', {c:COL.tealL, fs:.42}),
          S.t(7.40, 5.85, {zh:'一條環：每個城市兩個鄰居',
                           en:'a cycle: two neighbours each'},
              {c:COL.tealL, fs:.26}),
          S.t(4.83, 3.10, '<', {c:COL.orangeL, fs:.55})]),
      panels:[{lbl:{zh:'dp[mask] 的答案', en:'what dp[mask] returns'}, chips:[chip(fmt(MO5.total) + ' km', 'bad')]},
              {lbl:{zh:'真正的最佳解', en:'the true optimum'}, chips:[chip(fmt(HK5.best) + ' km', 'ok')]}],
      line:8,
      msg:{zh:'左邊這個東西有名字：最小生成樹。它是 TSP 有名的下界，所以永遠比最佳環線小——' +
              fmt(MO5.total) + ' 比 ' + fmt(HK5.best) +
              ' 小，單看數字毫無破綻。這就是為什麼今天的重點不是遞迴式，而是狀態：狀態少一個欄位，程式不會抗議，只會安靜地回答別的問題。',
           en:'The thing on the left has a name: the minimum spanning tree. It is a famous lower bound for the TSP, so it always comes out below the optimal tour - ' +
              fmt(MO5.total) + ' against ' + fmt(HK5.best) +
              ', and nothing about the number looks wrong. That is why today is about the state rather than the recurrence: drop one field and the program does not complain, it quietly answers something else.'}});
  } else {
    const stInit = {0:'hot'};
    F.push({shapes:cityNodes(P8, N8, stInit, .24).concat([
        S.t(2.35, .72, {zh:'最近鄰居法', en:'nearest neighbour'}, {c:COL.orangeL, fs:.32}),
        S.t(2.35, 1.12, {zh:'每步挑最近的沒走過的城市',
                         en:'hop to the closest unvisited city'}, {c:COL.grey, fs:.25}),
        S.t(6.7, 1.40, {zh:'狀態只有「現在站在哪」',
                        en:'the state is only "where am I now"'},
            {c:COL.pale, fs:.29}),
        S.t(6.7, 2.10, {zh:'跑得飛快，而且真的是合法環線',
                        en:'Instant, and it really is a legal tour.'},
            {c:COL.pale, fs:.28}),
        S.t(6.7, 3.00, {zh:'沒有例外，沒有警告，只是比較貴',
                        en:'No exception, no warning, just dearer.'},
            {c:COL.orangeL, fs:.29}),
        S.t(6.7, 3.90, {zh:'八個停靠點', en:'eight stops this time'},
            {c:COL.grey, fs:.27})]),
      panels:[{lbl:{zh:'目前位置', en:'standing at'}, chips:[chip('TPE', 'hot')]},
              {lbl:{zh:'累計', en:'running total'}, chips:[chip('0 km', 'dim')]}],
      line:11,
      msg:{zh:'另一個方向的錯誤：不是狀態少了欄位，而是乾脆不做 DP。最近鄰居法每一步都挑最近的城市，複雜度只有 O(n²)，而且結果是一條真的環線。問題在於它做的每一個決定都不能反悔。',
           en:'The other kind of wrong answer: not a missing field in the state, but no DP at all. Nearest neighbour hops to the closest city every time, costs only O(n^2), and produces a genuine tour. The catch is that every decision it makes is final.'}});
    let total = 0;
    const done = [0];
    NN8.hops.forEach((h, k) => {
      const [cur, nxt, d] = h;
      total += d; if (k < NN8.hops.length - 1) done.push(nxt);
      const st = {}; done.forEach(i => st[i] = 'done'); st[nxt] = 'hot'; st[cur] = 'act';
      const esh = [];
      for (let t = 0; t <= k; t++){
        const e = NN8.hops[t];
        esh.push(S.e(P8[e[0]][0], P8[e[0]][1], P8[e[1]][0], P8[e[1]][1],
                     {s:t === k ? 'hot' : 'done', arrow:false, pad:.28, w:.07,
                      lab:t === k ? String(e[2]) : null, fs:.25}));
      }
      const trap = (cur === 3 && nxt === 7);
      F.push({shapes:cityNodes(P8, N8, st, .24).concat(esh).concat([
          S.t(2.35, .78, N8[cur] + ' → ' + N8[nxt] + '  +' + d + ' km',
              {c:trap ? COL.red : COL.orangeL, fs:.32}),
          S.t(6.7, 1.40, {zh:'累計 ' + fmt(total) + ' km', en:'total so far ' + fmt(total) + ' km'},
              {c:COL.orangeL, fs:.34}),
          S.t(6.7, 2.40, trap
              ? {zh:'最近的是 HUN，南部全被丟下',
                 en:'closest is HUN - the south is stranded'}
              : {zh:'只看眼前最短的一段',
                 en:'only the shortest hop in front of it'},
              {c:trap ? COL.red : COL.pale, fs:.27})]),
        panels:[{lbl:{zh:'目前位置', en:'standing at'}, chips:[chip(N8[nxt], 'hot')]},
                {lbl:{zh:'累計', en:'running total'}, chips:[chip(fmt(total) + ' km', 'hot')]}],
        line:trap ? 13 : (k === NN8.hops.length - 1 ? 16 : 13),
        msg:trap
          ? {zh:'陷阱在這裡。站在日月潭，最近的沒走過的城市是花蓮（74 km），比往南的台南（116 km）近。走過去之後，整個南部（台南、高雄、台東）只能之後再繞回去，而那趟繞路的代價遠超過現在省下的 42 km。貪心看不到這件事，因為它的狀態裡沒有「還剩哪些城市」。',
             en:'Here is the trap. Standing at Sun Moon Lake, the closest unvisited city is Hualien at 74 km, nearer than Tainan at 116 km. Going there strands the whole south - Tainan, Kaohsiung, Taitung - to be picked up later, and that detour costs far more than the 42 km just saved. Greedy cannot see it, because its state has no record of which cities are left.'}
          : {zh:'從 ' + N8[cur] + ' 走到 ' + N8[nxt] + '，' + d +
                ' km，是目前所有沒走過的城市裡最近的一個。每一步都是當下的最佳選擇，這正是貪心法讓人放心的地方。',
             en:'From ' + N8[cur] + ' to ' + N8[nxt] + ' at ' + d +
                ' km, the closest of everything still unvisited. Every step is locally optimal, which is exactly what makes greedy feel safe.'}});
    });
    const P8b = mapper(XY8, 5.85, 1.30, 2.60, 4.30);
    const stA = {}, stB = {};
    for (let i = 0; i < 8; i++){ stA[i] = 'bad'; stB[i] = 'ok'; }
    const nnE = NN8.hops.map(e => S.e(P8[e[0]][0], P8[e[0]][1], P8[e[1]][0], P8[e[1]][1],
                                      {s:'bad', arrow:false, pad:.28, w:.07}));
    const hkT = HK8.tour.concat([0]);
    F.push({shapes:cityNodes(P8, N8, stA, .24).concat(nnE)
        .concat(cityNodes(P8b, N8, stB, .24))
        .concat(pathEdges(P8b, hkT, 'ok', {arrow:false}))
        .concat([
          S.t(2.35, .78, fmt(NN8.total) + ' km', {c:COL.red, fs:.42}),
          S.t(2.35, 5.95, {zh:'最近鄰居法', en:'nearest neighbour'}, {c:COL.red, fs:.30}),
          S.t(7.15, .78, fmt(HK8.best) + ' km', {c:COL.tealL, fs:.42}),
          S.t(7.15, 5.95, 'Held-Karp', {c:COL.tealL, fs:.30}),
          S.t(4.78, 2.70, '+' + (NN8.total - HK8.best) + ' km', {c:COL.orangeL, fs:.34}),
          S.t(4.78, 3.30, '+' + ((NN8.total - HK8.best) / HK8.best * 100).toFixed(1) + '%',
              {c:COL.orangeL, fs:.34}),
          S.t(4.78, 4.05, {zh:'沒有錯誤訊息', en:'no error raised'}, {c:COL.grey, fs:.26})]),
      panels:[{lbl:{zh:'貪心', en:'greedy'}, chips:[chip(fmt(NN8.total) + ' km', 'bad')]},
              {lbl:{zh:'精確解', en:'exact'}, chips:[chip(fmt(HK8.best) + ' km', 'ok')]}],
      line:16,
      msg:{zh:'貪心給的是一條真的環線，只是貴了 ' + (NN8.total - HK8.best) + ' km，多 ' +
              ((NN8.total - HK8.best) / HK8.best * 100).toFixed(1) +
              '%。這兩種錯法值得並排看：dp[mask] 給的數字太小（它其實是下界），貪心給的數字太大（它是一條可行解）。狀態壓縮 DP 的價值就在於它兩邊都不是——它給的是真正的最佳值，代價是 2ⁿ·n² 的表。',
           en:'Greedy hands back a genuine tour, merely ' + (NN8.total - HK8.best) + ' km dearer, ' +
              ((NN8.total - HK8.best) / HK8.best * 100).toFixed(1) +
              '% over. The two failures are worth seeing side by side: dp[mask] returns a number that is too small, because it is really a lower bound, and greedy returns one that is too large, because it is a feasible solution. State-compression DP is neither - it returns the optimum, and pays a 2^n * n^2 table for it.'}});
  }
  return F.list;
}

/* ======================================================================== *
 * Tab 4 - LeetCode 943: the same state, with overlap instead of distance
 * ======================================================================== */
const CODE_LC = [
  'def shortest_superstring(words):',
  '    ov = [[overlap(a, b) for b in words] for a in words]',
  '    dp  = [[-1] * n for _ in range(1 << n)]',
  '    par = [[-1] * n for _ in range(1 << n)]',
  '    for i in range(n):',
  '        dp[1 << i][i] = 0            # ANY word may start',
  '    for mask in range(1 << n):',
  '        for last in range(n):',
  '            if dp[mask][last] < 0: continue',
  '            for nxt in range(n):',
  '                if mask >> nxt & 1: continue',
  '                cand = dp[mask][last] + ov[last][nxt]',
  '                nm = mask | 1 << nxt',
  '                if cand > dp[nm][nxt]:',
  '                    dp[nm][nxt], par[nm][nxt] = cand, last',
  '    end = max(range(n), key=lambda i: dp[full][i])',
  '    return stitch(words, walk_back(par, full, end))'
];

const WORDS = ['catg', 'ctaagt', 'gcta', 'ttca'];
const NW = WORDS.length;
function overlap(a, b){
  const m = Math.min(a.length, b.length);
  for (let k = m; k > 0; k--) if (a.slice(a.length - k) === b.slice(0, k)) return k;
  return 0;
}
const OV = WORDS.map(a => WORDS.map((b, j) => a === b ? 0 : overlap(a, b)));
function superRun(fixedStart){
  const full = (1 << NW) - 1, dp = [], par = [];
  for (let m = 0; m <= full; m++){ dp.push(new Array(NW).fill(-1)); par.push(new Array(NW).fill(-1)); }
  if (fixedStart) dp[1][0] = 0;
  else for (let i = 0; i < NW; i++) dp[1 << i][i] = 0;
  for (let mask = 0; mask <= full; mask++)
    for (let last = 0; last < NW; last++){
      if (dp[mask][last] < 0) continue;
      for (let nxt = 0; nxt < NW; nxt++){
        if (mask >> nxt & 1) continue;
        const nm = mask | 1 << nxt, c = dp[mask][last] + OV[last][nxt];
        if (c > dp[nm][nxt]){ dp[nm][nxt] = c; par[nm][nxt] = last; }
      }
    }
  let end = -1, bestOv = -1;
  for (let i = 0; i < NW; i++) if (dp[full][i] > bestOv){ bestOv = dp[full][i]; end = i; }
  const order = []; let m = full, c = end;
  while (c !== -1){ order.push(c); const p = par[m][c]; m ^= 1 << c; c = p; }
  order.reverse();
  let out = WORDS[order[0]];
  const spans = [[0, WORDS[order[0]].length, order[0]]];
  for (let k = 1; k < order.length; k++){
    const o = OV[order[k - 1]][order[k]], start = out.length - o;
    out += WORDS[order[k]].slice(o);
    spans.push([start, start + WORDS[order[k]].length, order[k]]);
  }
  return {dp:dp, par:par, order:order, text:out, spans:spans, ov:bestOv};
}
const SUP_FREE = superRun(false), SUP_FIX = superRun(true);

const LX = 5.55, LW = 0.88, LY = 1.30, LH = 0.28;
const LMASKS = (function(){
  const a = [];
  for (let m = 1; m < 16; m++) a.push(m);
  a.sort((x, y) => pcount(x) - pcount(y) || x - y);
  return a;
})();
function lcGrid(run, upto, hotMask){
  const out = [S.t(LX - .18, LY - .18, 'mask', {c:COL.grey, fs:.24, anchor:'end'})];
  for (let j = 0; j < NW; j++)
    out.push(S.t(LX + j * LW + LW / 2 - .03, LY - .18, WORDS[j], {c:COL.tealL, fs:.22}));
  LMASKS.forEach((m, r) => {
    const y = LY + r * LH;
    out.push(S.t(LX - .18, y + LH * .70, maskStr(m, NW),
                 {c:m === hotMask ? COL.orangeL : COL.grey, fs:.22, anchor:'end'}));
    for (let j = 0; j < NW; j++){
      const v = run.dp[m][j], shown = r <= upto && v >= 0;
      const sty = shown ? (m === hotMask ? 'hot' : 'soft') : 'ghost';
      out.push(S.r(LX + j * LW, y, LW - .06, LH - .05, sty, shown ? String(v) : '', {fs:.21, rx:.05}));
    }
  });
  return out;
}
function ovMatrix(hotPair){
  const x0 = 0.95, y0 = 1.70, w = 0.72, h = 0.46;
  const out = [S.t(2.5, 1.32, {zh:'ov[a][b]：重疊的字母數',
                               en:'ov[a][b]: overlapping letters'},
                   {c:COL.grey, fs:.26})];
  for (let j = 0; j < NW; j++){
    out.push(S.t(x0 + j * w + w / 2 + .55, y0 - .14, WORDS[j], {c:COL.tealL, fs:.21}));
    out.push(S.t(x0 + .48, y0 + j * h + h * .68, WORDS[j], {c:COL.tealL, fs:.21, anchor:'end'}));
  }
  for (let i = 0; i < NW; i++) for (let j = 0; j < NW; j++){
    const hot = hotPair && hotPair[0] === i && hotPair[1] === j;
    out.push(S.r(x0 + .55 + j * w, y0 + i * h, w - .06, h - .06,
                 i === j ? 'ghost' : (hot ? 'hot' : 'soft'), i === j ? '' : String(OV[i][j]), {fs:.24, rx:.05}));
  }
  return out;
}
function stringShapes(run, y, sty){
  const n = run.text.length, cw = 0.44, x0 = (9.8 - n * cw) / 2, out = [];
  for (let i = 0; i < n; i++)
    out.push(S.r(x0 + i * cw, y, cw - .05, .52, sty, run.text[i], {fs:.30, rx:.05}));
  run.spans.forEach((s, k) => {
    const yy = y + .70 + k * .30;
    out.push(S.e(x0 + s[0] * cw + .04, yy, x0 + s[1] * cw - .09, yy,
                 {s:'act', arrow:false, w:.05}));
    out.push(S.t(x0 + s[1] * cw + .12, yy + .10, WORDS[s[2]], {c:COL.purpleL, fs:.24, anchor:'start'}));
  });
  return out;
}

function lcFrames(v){
  const run = v === 1 ? SUP_FIX : SUP_FREE;
  const F = new Frames();
  F.push({shapes:ovMatrix(null).concat(lcGrid(run, -1, -1)).concat([
      S.t(2.5, .72, {zh:'LC 943：接成最短的字串',
                     en:'LC 943: the shortest superstring'},
          {c:COL.tealL, fs:.28}),
      S.t(2.5, 4.20, WORDS.join('   '), {c:COL.pale, fs:.30}),
      S.t(2.5, 4.80, {zh:'總長 ' + WORDS.join('').length + ' 個字母',
                      en:WORDS.join('').length + ' letters in total'},
          {c:COL.grey, fs:.27}),
      S.t(2.5, 5.30, {zh:'重疊掉的都是省下來的長度',
                      en:'every overlap saves a letter'},
          {c:COL.grey, fs:.26}),
      S.t(2.5, 5.85, {zh:'狀態跟 TSP 一樣：(用過哪些, 最後是哪個)',
                      en:'same state as the TSP: (used, last)'},
          {c:COL.orangeL, fs:.24}),
      S.t(7.3, .72, 'dp[mask][last]', {c:COL.tealL, fs:.32})]),
    panels:[{lbl:{zh:'字串', en:'words'}, chips:WORDS.map(w => chip(w, 'dim'))},
            {lbl:{zh:'起點', en:'start'},
             chips:[chip(v === 1 ? WORDS[0] + ' (fixed)' : 'any', v === 1 ? 'bad' : 'ok')]}],
    line:1,
    msg:v === 1
      ? {zh:'同一題，只改一行：這一版把 dp[1 << 0][0] = 0 當成唯一的起點，其他三個字都不能當開頭。TSP 那邊這樣寫是對的，因為環線上任何城市都可以當起點；但這一題接出來的是一條線，不是環。',
         en:'Same problem, one line changed: this run seeds only dp[1 << 0][0] = 0, so the other three words may never start the string. On the TSP that shortcut was correct, because any city on a cycle can be called the start; here the answer is an open path, not a cycle.'}
      : {zh:'今天最後一個分頁換個題目，但狀態一字不改。TSP 是「最小化距離」，這題是「最大化重疊」；距離矩陣換成重疊矩陣，dp[mask][last] 原封不動。',
         en:'The last tab changes the problem and leaves the state untouched. The TSP minimised distance; this maximises overlap. Swap the distance matrix for an overlap matrix and dp[mask][last] carries over word for word.'}});

  LMASKS.forEach((m, r) => {
    if (pcount(m) === 1){
      F.push({shapes:ovMatrix(null).concat(lcGrid(run, r, m)).concat([
          S.t(2.5, .72, {zh:'起點：' + (v === 1 ? '只有 ' + WORDS[0] : '四個字都可以'),
                         en:'seeds: ' + (v === 1 ? 'only ' + WORDS[0] : 'all four words')},
              {c:v === 1 ? COL.red : COL.tealL, fs:.30}),
          S.t(2.5, 5.30, v === 1
              ? {zh:'只有 1000 那一列被種下',
                 en:'only the row 1000 is seeded'}
              : {zh:'每個字自己就是長度 1 的答案',
                 en:'each word alone is an answer'},
              {c:COL.pale, fs:.27}),
          S.t(7.3, .72, 'mask = ' + maskStr(m, NW), {c:COL.tealL, fs:.30})]),
        panels:[{lbl:{zh:'這一列', en:'this row'},
                 chips:bitsOf(m, NW).filter(j => run.dp[m][j] >= 0).map(j => chip(WORDS[j] + ' 0', 'hot'))},
                {lbl:{zh:'起點', en:'start'},
                 chips:[chip(v === 1 ? WORDS[0] : 'any', v === 1 ? 'bad' : 'ok')]}],
        line:5,
        msg:v === 1
          ? {zh:'種子只有一個。注意這一版還是會跑完、還是會回傳一個含有全部四個字的字串——它完全合法，只是比較長。這種錯誤在 LeetCode 上特別痛，因為小測資常常剛好讓兩種寫法答案相同。',
             en:'One seed only. Note that this version still finishes and still returns a string containing all four words - perfectly valid, merely longer. That is a nasty kind of wrong on LeetCode, because small test cases often let both versions agree.'}
          : {zh:'四個 1 位元的 mask 全部種成 0：任何一個字都可以當開頭。這一步就是 TSP 和這題唯一的結構差異——TSP 只種 dp[1][0]，因為環線可以從任何城市開始，起點不是決定；這裡起點是決定。',
             en:'All four single-bit masks are seeded with zero, so any word may open the string. This is the one structural difference from the TSP: there only dp[1][0] is seeded, because a cycle may be read from any city and the start is not a decision. Here it is.'}});
      return;
    }
    let bl = -1, bv = -1;
    for (let j = 0; j < NW; j++) if (run.dp[m][j] > bv){ bv = run.dp[m][j]; bl = j; }
    const chain = (function(){ const o = []; let mm = m, c = bl;
      while (c !== -1){ o.push(c); const p = run.par[mm][c]; mm ^= 1 << c; c = p; } return o.reverse(); })();
    const hp = chain.length > 1 ? [chain[chain.length - 2], bl] : null;
    F.push({shapes:ovMatrix(hp).concat(lcGrid(run, r, m)).concat([
        S.t(2.5, .72, 'mask = ' + maskStr(m, NW), {c:COL.orangeL, fs:.30}),
        S.t(2.5, 1.05, bitsOf(m, NW).map(i => WORDS[i]).join(' + '), {c:COL.grey, fs:.24}),
        S.t(2.5, 4.35, bv >= 0
            ? {zh:'最佳重疊 ' + bv, en:'best overlap ' + bv}
            : {zh:'這一列到不了', en:'this row is unreachable'},
            {c:bv >= 0 ? COL.pale : COL.red, fs:.29}),
        S.t(2.5, 4.90, bv >= 0 ? chain.map(i => WORDS[i]).join(' → ') : '',
            {c:COL.purpleL, fs:.25}),
        S.t(2.5, 5.55, bv >= 0
            ? {zh:'目前長度 ' + (bitsOf(m, NW).reduce((a, i) => a + WORDS[i].length, 0) - bv) + ' 個字母',
               en:'length so far ' + (bitsOf(m, NW).reduce((a, i) => a + WORDS[i].length, 0) - bv) + ' letters'}
            : {zh:'因為起點被鎖死了', en:'because the start was pinned'},
            {c:COL.grey, fs:.27}),
        S.t(7.3, .72, 'dp[' + maskStr(m, NW) + '][·]', {c:COL.tealL, fs:.32})]),
      panels:[{lbl:{zh:'這一列', en:'this row'},
               chips:bitsOf(m, NW).filter(j => run.dp[m][j] >= 0)
                 .map(j => chip(WORDS[j] + ' ' + run.dp[m][j], j === bl ? 'hot' : 'dim'))},
              {lbl:{zh:'最佳重疊', en:'best overlap'}, chips:[chip(bv >= 0 ? String(bv) : '—', bv >= 0 ? 'ok' : 'bad')]}],
      line:11,
      msg:bv < 0
        ? {zh:'整列空的：' + bitsOf(m, NW).map(i => WORDS[i]).join(' + ') +
              ' 這一組一定要從 ' + WORDS[bitsOf(m, NW)[0]] + ' 之外的字開頭才到得了，但起點被鎖在 ' + WORDS[0] + ' 上。',
           en:'The whole row is empty: the set ' + bitsOf(m, NW).map(i => WORDS[i]).join(' + ') +
              ' can only be reached by starting somewhere other than ' + WORDS[0] + ', and the start is pinned there.'}
        : {zh:'和 TSP 唯一不同的是把 min 換成 max、把距離換成重疊。' +
              '這一格記的是「用掉 ' + bitsOf(m, NW).map(i => WORDS[i]).join('、') + '、最後接的是 ' + WORDS[bl] +
              '」時能省下的最多字母數：' + bv + ' 個。省下越多，最後的字串越短。',
           en:'The only edits from the TSP are min becoming max and distance becoming overlap. ' +
              'This cell records the most letters that can be saved when ' + bitsOf(m, NW).map(i => WORDS[i]).join(', ') +
              ' are used and ' + WORDS[bl] + ' is the last one appended: ' + bv +
              '. The more saved, the shorter the final string.'}});
  });

  const other = v === 1 ? SUP_FREE : SUP_FIX;
  F.push({shapes:stringShapes(run, 1.65, v === 1 ? 'bad' : 'ok').concat([
      S.t(4.9, .80, run.text + '   (' + run.text.length + ')',
          {c:v === 1 ? COL.red : COL.tealL, fs:.40}),
      S.t(4.9, 4.30, {zh:'四個字總長 ' + WORDS.join('').length + '，重疊省下 ' + run.ov + '，剩 ' + run.text.length + '。',
                      en:WORDS.join('').length + ' letters of input, ' + run.ov + ' saved by overlap, ' + run.text.length + ' left.'},
          {c:COL.pale, fs:.31}),
      S.t(4.9, 5.05, v === 1
          ? {zh:'鎖死起點：' + SUP_FIX.text.length + ' 個字母，不是 ' + SUP_FREE.text.length +
                '——仍然合法，只是比較長',
             en:'Start pinned: ' + SUP_FIX.text.length + ' letters, not ' + SUP_FREE.text.length +
                ' - valid, just longer.'}
          : {zh:'把 min 換 max、公里換字母，同一張表',
             en:'min becomes max, km becomes letters - the same table.'},
          {c:v === 1 ? COL.red : COL.orangeL, fs:.30}),
      S.t(4.9, 5.80, {zh:'另一版的答案：' + other.text + '（' + other.text.length + '）',
                      en:'the other version answers ' + other.text + ' (' + other.text.length + ')'},
          {c:COL.grey, fs:.28})]),
    panels:[{lbl:{zh:'答案', en:'answer'}, chips:[chip(run.text, v === 1 ? 'bad' : 'ok')]},
            {lbl:{zh:'長度', en:'length'}, chips:[chip(String(run.text.length), v === 1 ? 'bad' : 'ok')]}],
    line:16,
    msg:v === 1
      ? {zh:'兩個版本都跑完、都回傳含有全部四個字的字串，差別只有 ' + (SUP_FIX.text.length - SUP_FREE.text.length) +
            ' 個字母。這是今天第三種無聲錯誤，而且它的來源正是把 TSP 的寫法照抄過來：那邊「起點隨便挑」是對的，因為環線沒有起點；這邊少了那個自由度，最短的接法就被排除在搜尋空間外面了。',
         en:'Both versions run to completion and both return a string containing all four words; they differ by ' + (SUP_FIX.text.length - SUP_FREE.text.length) +
            ' letters. This is the third silent failure of the day, and it comes from copying the TSP code across: there "pick any start" is correct because a cycle has no beginning, and here that freedom is exactly what the shortest stitching needed.'}
      : {zh:'接出來的字串是 ' + run.text + '，' + run.text.length +
            ' 個字母，每一個字都在裡面。把距離換成重疊、把 min 換成 max 之後，整個 Held-Karp 一行都不用改——這就是狀態壓縮 DP 值得認真學的原因：它不是一個演算法，是一種索引方式。',
         en:'The stitched string is ' + run.text + ' at ' + run.text.length +
            ' letters, with every word inside it. Once distance becomes overlap and min becomes max, not a line of Held-Karp needs changing - which is why state compression is worth learning properly: it is not an algorithm so much as a way of indexing.'}});
  return F.list;
}

/* ======================================================================== */
const DAY_META = {
  title:{zh:'Day 37 — 狀態壓縮 DP 與 Held-Karp：用一個整數當集合索引，精確解出 TSP',
         en:'Day 37 - State-compression DP and Held-Karp: indexing a table by a set to solve the TSP exactly'},
  sub:{zh:'表格的索引從「前 i 個」變成「哪些」。集合壓成整數之後填表順序是免費的，難的是狀態要記什麼。',
       en:'The table index stops being a prefix and becomes a set. Pack the set into an integer and the fill order comes for free; what is hard is deciding what the state must remember.'},
  tabs:[
    {
      id:'state', label:{zh:'狀態：(mask, last)', en:'the state: (mask, last)'},
      stage:{zh:'走過同一組城市的路線，可以合併到什麼程度',
             en:'how far two routes over the same set of cities may be merged'},
      view:VIEW,
      variants:[{zh:'(mask, last)（正確）', en:'(mask, last) - correct'},
                {zh:'只記 mask（錯誤）', en:'mask only - wrong'}],
      idea:{zh:'狀態壓縮 DP 的全部內容就是：把「哪些東西已經處理過」寫成一個整數的位元，拿它當表格索引。難的從來不是壓縮本身，而是壓縮之後還要補上什麼。TSP 的答案是 (mask, last)：光知道走過哪些城市不夠，還得知道現在站在哪，否則下一段路的長度根本寫不出來。合併之所以安全，是因為接下來要付的每一公里只看終點城市，跟前面怎麼繞無關——這正是最佳子結構在這題的具體樣子。24 條前綴壓成 12 個狀態，而這個比例會隨 n 指數地拉開。',
            en:'State compression is one idea: write "which things are already handled" into the bits of an integer and index the table with it. The compression is never the hard part; what has to be added back afterwards is. For the TSP the answer is (mask, last) - knowing which cities are done is not enough, because the length of the next hop cannot even be written without knowing where I am standing. The merge is safe precisely because every remaining kilometre depends only on the final city and not on the route that reached it, which is what optimal substructure looks like here. Twenty-four prefixes become twelve states, and that ratio widens exponentially in n.'},
      legend:[['#9d6bff', {zh:'同一組城市', en:'one set of cities'}],
              ['#3fe0dd', {zh:'留下來的狀態', en:'the state that is kept'}],
              ['#ff9736', {zh:'要比較的東西', en:'what is being compared'}],
              ['#ff5c5c', {zh:'丟掉 last 之後寫不出來的式子', en:'the expression that needs the discarded last'}]],
      code:CODE_STATE, build:stateFrames
    },
    {
      id:'tour', label:{zh:'Held-Karp 填表', en:'Held-Karp: filling the table'},
      stage:{zh:'一列一個子集合，一欄一個「現在站在哪」，由小到大就是合法順序',
             en:'a row per subset, a column per current city - and ascending order is already legal'},
      view:VIEW,
      idea:{zh:'Held-Karp 是 1962 年的老演算法，到今天仍然是 TSP 最好的精確解上界之一。表有 2ⁿ 列 n 欄，每格再掃 n 個下一步，所以是 O(2ⁿ·n²)；n = 20 大約是四億個單位，還算得動，n = 30 就不行了。這裡最值得注意的是 Day 36 花了整個分頁在講的「填表順序」今天完全消失了：mask | 1 << nxt 一定比 mask 大，所以 for mask in range(1 << n) 本身就是拓撲序。集合當索引時，位元的單調性順便把依賴關係排好了。',
            en:'Held-Karp dates from 1962 and is still one of the best exact bounds known for the TSP. The table has 2^n rows and n columns and each cell scans n successors, so O(2^n * n^2): around four hundred million units at n = 20, which is still runnable, and hopeless at n = 30. The striking part is that the fill order, which took a whole tab yesterday, has vanished - mask | 1 << nxt is always larger than mask, so for mask in range(1 << n) is already a topological order. When a set is the index, the monotonicity of the bits sorts the dependencies out for free.'},
      legend:[['#ff9736', {zh:'正在填的這一列', en:'the row being filled'}],
              ['#9d6bff', {zh:'同一列的其他終點', en:'other endings in the same row'}],
              ['#3fe0dd', {zh:'已完成 / 最佳環線', en:'done / the optimal tour'}],
              ['#123f4d', {zh:'不可能的狀態', en:'impossible states'}]],
      code:CODE_HK, build:hkFrames
    },
    {
      id:'wrong', label:{zh:'兩種不會報錯的錯', en:'two wrong answers that never raise'},
      stage:{zh:'一個太小（其實是最小生成樹），一個太大（其實是可行解）',
             en:'one too small - a spanning tree - and one too large - a feasible tour'},
      view:VIEW,
      variants:[{zh:'只記 mask → 最小生成樹', en:'mask only -> spanning tree'},
                {zh:'最近鄰居法 → 貴 15%', en:'nearest neighbour -> 15% dearer'}],
      idea:{zh:'兩種錯誤剛好夾住正確答案，值得一起記。第一種是狀態少了 last：唯一寫得出來的遞迴變成「接到最近的已走訪城市」，那長出來的是一棵樹不是一條環，回傳的數字正好是最小生成樹的重量，也就是 TSP 的經典下界，永遠比最佳解小。第二種是乾脆不做 DP：最近鄰居法每步挑最近的城市，答案是一條真的環線，所以永遠比最佳解大。兩者都不丟例外、都回傳看起來合理的數字——判斷一個 TSP 實作對不對，不能只看它有沒有跑完。',
            en:'The two failures bracket the right answer, which makes them worth remembering together. The first drops last from the state: the only recurrence still writable is "attach to the nearest visited city", which grows a tree rather than a cycle, and the number it returns is exactly the weight of the minimum spanning tree - the classic TSP lower bound, always below the optimum. The second abandons DP altogether: nearest neighbour takes the closest city every time and yields a genuine tour, so it always lands above the optimum. Neither raises anything, and both return plausible numbers - which is why "it ran" is not evidence that a TSP implementation is correct.'},
      legend:[['#ff5c5c', {zh:'錯誤版本產生的圖', en:'what the wrong version builds'}],
              ['#ff9736', {zh:'目前這一步', en:'the current step'}],
              ['#3fe0dd', {zh:'正確的最佳環線', en:'the correct optimal tour'}],
              ['#9d6bff', {zh:'這一段的來源', en:'where this hop grew from'}]],
      code:CODE_WRONG, build:wrongFrames
    },
    {
      id:'lc943', label:{zh:'LC 943 最短超級字串', en:'LC 943 shortest superstring'},
      stage:{zh:'同一個 (mask, last)，把距離換成重疊、把 min 換成 max',
             en:'the same (mask, last), with overlap for distance and max for min'},
      view:VIEW,
      variants:[{zh:'任何字都能當開頭（正確）', en:'any word may start - correct'},
                {zh:'鎖死第一個字（錯誤）', en:'first word pinned - wrong'}],
      idea:{zh:'把四段 DNA 接成最短的一條，本質就是 TSP：城市換成字串，距離換成「接上去之後還要多寫幾個字母」。把重疊量最大化跟把新增字母最小化是同一件事，所以整張 Held-Karp 的表原封不動搬過來就能用。唯一要小心的是起點：TSP 求的是環線，任何城市都可以叫做起點，所以只種 dp[1][0] 是免費的；這一題求的是一條線，第一個字是真的決定。把 TSP 的寫法照抄，會得到一個合法但比較長的答案，' + '而且小測資常常看不出來。',
            en:'Stitching four DNA reads into the shortest possible string is the TSP in disguise: cities become words and distance becomes "how many extra letters appending this costs". Maximising overlap and minimising new letters are the same objective, so the entire Held-Karp table carries over untouched. The one thing to watch is the start: the TSP asks for a cycle, any city may be called the beginning, and seeding only dp[1][0] is therefore free. This problem asks for an open path, so the first word is a real decision. Copy the TSP code across and the result is a valid but longer answer - and small test cases very often fail to show it.'},
      legend:[['#ff9736', {zh:'正在填的這一列', en:'the row being filled'}],
              ['#9d6bff', {zh:'每個字在答案裡的位置', en:'where each word sits in the answer'}],
              ['#3fe0dd', {zh:'正確答案', en:'the correct answer'}],
              ['#ff5c5c', {zh:'鎖死起點的版本', en:'the version with the start pinned'}]],
      code:CODE_LC, build:lcFrames
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
