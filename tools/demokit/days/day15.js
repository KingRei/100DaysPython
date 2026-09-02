// DAY: 15
// TITLE_ZH: 最小生成樹：Kruskal
// TITLE_EN: Minimum Spanning Tree - Kruskal
// SUB_ZH: 邊由便宜排到貴，一條一條收，只要不繞回自己就留下。判斷「會不會繞回自己」的那一步，就是昨天的 Union-Find。
// SUB_EN: Sort the edges cheapest first and keep each one unless it closes a cycle.
// FOLDER: day%2015%20-%20minimum%20spanning%20tree%20-%20kruskal
// MEDIUM: https://medium.com/100-days-of-python/day-15-%E6%9C%80%E5%B0%8F%E7%94%9F%E6%88%90%E6%A8%B9-kruskal-cae4865723fa

const NM = 'ABCDEFG';
const P = [[1.1, 3.3], [3.4, 2.4], [7.0, 3.0], [2.4, 4.4], [5.4, 4.4], [2.5, 6.2], [7.6, 5.8]];
const G = [[7, 0, 1], [5, 0, 3], [8, 1, 2], [9, 1, 3], [7, 1, 4], [5, 2, 4],
           [15, 3, 4], [6, 3, 5], [8, 4, 5], [9, 4, 6], [11, 5, 6]];
const RAD = .46;
const ekey = (u, v) => Math.min(u, v) + '-' + Math.max(u, v);

function UF(n){
  const p = [], rk = [];
  for (let i = 0; i < n; i++){ p.push(i); rk.push(0); }
  const f = x => { let r = x; while (p[r] !== r) r = p[r];
                   while (p[x] !== r){ const nx = p[x]; p[x] = r; x = nx; } return r; };
  return {p:p, find:f, count:n,
    union:function(a, b){ let ra = f(a), rb = f(b); if (ra === rb) return false;
      if (rk[ra] < rk[rb]){ const t = ra; ra = rb; rb = t; }
      p[rb] = ra; if (rk[ra] === rk[rb]) rk[ra]++; this.count--; return true; }};
}

// ---- drawing ---------------------------------------------------------------
function gShapes(edges, o){
  o = o || {}; const out = [];
  edges.forEach(([w, u, v]) => {
    const st = (o.est && o.est[ekey(u, v)]) || 'ghost';
    const [x1, y1] = P[u], [x2, y2] = P[v];
    const dx = x2 - x1, dy = y2 - y1, L = Math.sqrt(dx * dx + dy * dy), ux = dx / L, uy = dy / L;
    out.push(S.e(x1 + ux * RAD, y1 + uy * RAD, x2 - ux * RAD, y2 - uy * RAD,
      {s:st, w:st === 'ok' || st === 'done' ? .12 : .05, noArrow:true}));
    out.push(S.t((x1 + x2) / 2 + uy * .30, (y1 + y2) / 2 - ux * .30, String(w),
      {c:st === 'ghost' ? '#8fa3ac' : (st === 'bad' ? '#ff5c5c' : '#ffbe6b'), fs:.28}));
  });
  for (let i = 0; i < NM.length; i++)
    out.push(S.c(P[i][0], P[i][1], RAD, (o.st && o.st[i]) || 'idle', NM[i], {fs:.32}));
  return out;
}
function stripShapes(sorted, marks, cur){
  const out = [];
  sorted.forEach(([w, u, v], i) => {
    const x = .55 + i * .80;
    out.push(S.r(x, .30, .74, .70, i === cur ? 'act' : (marks[i] || 'ghost'),
      NM[u] + NM[v], {fs:.30, sub:String(w)}));
  });
  return out;
}
const panelState = (uf, chosen, total) => ({lbl:{zh:'演算法狀態', en:'algorithm state'},
  chips:[{t:(LANG === 'zh' ? '元件數 ' : 'components ') + uf.count, cls:uf.count === 1 ? 'ok' : ''},
         {t:(LANG === 'zh' ? '已選 ' : 'chosen ') + chosen + '/6', cls:''},
         {t:(LANG === 'zh' ? '總權重 ' : 'total ') + total, cls:'hot'}]});

const CODE = [
'def kruskal(n, edges):',
'    uf = UnionFind(n)',
'    chosen, total = [], 0',
'',
'    for w, u, v in sorted(edges):     # 便宜的先看',
'        if uf.union(u, v):            # 不同元件 -> 收下',
'            chosen.append((w, u, v))',
'            total += w',
'            if len(chosen) == n - 1:  # 生成樹剛好 n-1 條邊',
'                break',
'        # union 回傳 False = 兩端早就相連 = 這條邊會形成環',
'    return chosen, total'];

function buildScan(v){
  const F = new Frames();
  const edges = v === 0 ? G : G.filter(e => e[1] !== 6 && e[2] !== 6);
  const sorted = edges.slice().sort((a, b) => a[0] - b[0]);
  const uf = UF(7), marks = [], est = {};
  let total = 0, chosen = 0;
  F.push({shapes:stripShapes(sorted, marks, -1).concat(gShapes(edges, {})),
    panels:[panelState(uf, chosen, total)], line:0,
    msg:v === 0
      ? {zh:'目標：挑出<b>連通所有點、總權重最小</b>的一組邊。' +
            '這樣的一組邊剛好是一棵樹——<b>7 個點就是 6 條邊</b>，多一條必成環，少一條必斷開。' +
            'Kruskal 的策略樸素到不可思議：<b>把邊由便宜排到貴，一條一條看</b>。',
         en:'The goal: pick the set of edges that <b>connects every vertex at the lowest total weight</b>. Such a set is exactly a tree - <b>7 vertices means 6 edges</b>; one more forces a cycle, one fewer leaves it disconnected. Kruskal\'s strategy is almost absurdly plain: <b>sort the edges cheapest first and walk the list</b>.'}
      : {zh:'這次把 G 的兩條邊拿掉，圖<b>本來就斷成兩塊</b>。Kruskal 完全沒有假設圖是連通的，' +
            '接下來看它會停在哪裡。',
         en:'This time G\'s two edges are removed, so the graph is <b>already in two pieces</b>. Kruskal never assumed connectivity - watch where it stops.'}});
  let stop = false;
  sorted.forEach(([w, u, v2], i) => {
    if (stop) return;
    F.push({shapes:stripShapes(sorted, marks, i).concat(
        gShapes(edges, {est:Object.assign({}, est, {[ekey(u, v2)]:'hot'}), st:{[u]:'act', [v2]:'act'}})),
      panels:[panelState(uf, chosen, total)], line:4,
      msg:{zh:'看 <b>' + NM[u] + NM[v2] + '</b>（權重 ' + w + '）。' +
              '唯一要問的問題是：<b>' + NM[u] + ' 和 ' + NM[v2] + ' 已經連在一起了嗎？</b>',
           en:'Consider <b>' + NM[u] + NM[v2] + '</b> (weight ' + w + '). The only question is: <b>are ' +
              NM[u] + ' and ' + NM[v2] + ' already connected?</b>'}});
    if (uf.union(u, v2)){
      marks[i] = 'ok'; est[ekey(u, v2)] = 'ok'; total += w; chosen++;
      F.push({shapes:stripShapes(sorted, marks, -1).concat(
          gShapes(edges, {est:Object.assign({}, est), st:{[u]:'ok', [v2]:'ok'}})),
        panels:[panelState(uf, chosen, total)], line:5,
        msg:{zh:'不同元件 → <b>收下</b>。<code>union</code> 回傳 True，元件數少一個（剩 ' + uf.count + '）。' +
                '注意這一步<b>沒有做任何走訪</b>：昨天的 Union-Find 讓「會不會成環」變成兩次 find。',
             en:'Different components, so <b>take it</b>. <code>union</code> returns True and the component count drops to ' +
                uf.count + '. Note that <b>no traversal happened</b>: yesterday\'s Union-Find turned "would this close a cycle?" into two finds.'}});
      if (chosen === 6){
        F.push({shapes:stripShapes(sorted, marks, -1).concat(gShapes(edges, {est:est})),
          panels:[panelState(uf, chosen, total)], line:8,
          msg:{zh:'已經收到 <b>6 條邊</b>，7 個點全部連起來了，後面的邊<b>連看都不用看</b>。' +
                  '總權重 <b>' + total + '</b>——這就是最小生成樹。',
               en:'Six edges are in and all seven vertices are connected, so the rest of the list <b>never gets looked at</b>. Total weight <b>' +
                  total + '</b>: that is the minimum spanning tree.'}});
        stop = true;
        return;
      }
    } else {
      marks[i] = 'bad';
      F.push({shapes:stripShapes(sorted, marks, -1).concat(
          gShapes(edges, {est:Object.assign({}, est, {[ekey(u, v2)]:'bad'}), st:{[u]:'bad', [v2]:'bad'}})),
        panels:[panelState(uf, chosen, total)], line:10,
        msg:{zh:'兩端<b>早就在同一個元件</b>了 → 收下就會形成環 → 跳過。' +
                '而且它一定是那個環裡<b>最貴（至少不比較便宜）</b>的一條，因為便宜的都先看過了。',
             en:'Both ends are <b>already in the same component</b>, so taking this edge would close a cycle: skip it. And it is necessarily the <b>most expensive (or tied) edge in that cycle</b>, because everything cheaper was examined first.'}});
    }
  });
  if (v === 1)
    F.push({shapes:stripShapes(sorted, marks, -1).concat(gShapes(edges, {est:est, st:{6:'bad'}})),
      panels:[panelState(uf, chosen, total)], line:11,
      msg:{zh:'邊掃完了，卻只收到 <b>' + chosen + ' 條</b>（生成樹需要 6 條）。' +
              'G 沒有任何邊可以接上，於是得到的是<b>最小生成森林</b>：每個連通元件一棵樹。' +
              '<b>迴圈自己就處理好了這件事，一行特判都不用寫。</b>',
           en:'The list is exhausted with only <b>' + chosen +
              ' edges</b> (a spanning tree needs 6). G has no edge to join by, so the result is a <b>minimum spanning forest</b>: one tree per component. <b>The loop handled that on its own - not a single special case.</b>'}});
  return F.list;
}

function buildWhy(v){
  const F = new Frames();
  const CUT = [0, 1, 3, 5];                       // A B D F
  const inCut = i => CUT.indexOf(i) >= 0;
  const cross = G.filter(([w, u, v2]) => inCut(u) !== inCut(v2));
  if (v === 0){
    const region = S.r(.45, 1.75, 4.05, 5.0, 'ghost', '', {});
    const stCut = {}; CUT.forEach(i => stCut[i] = 'soft');
    F.push({shapes:[region].concat(gShapes(G, {st:stCut})),
      panels:[], line:5,
      msg:{zh:'貪心通常都有陷阱，Kruskal 為什麼不會？關鍵叫 <b>cut property</b>。' +
              '把 7 個點<b>任意</b>切成兩堆（這裡是 A B D F ｜ C E G），這叫一個 <b>cut</b>。',
           en:'Greedy usually has a catch - why not here? The key is the <b>cut property</b>. Split the seven vertices into two groups <b>any way you like</b> (here A B D F | C E G); that split is a <b>cut</b>.'}});
    const estC = {}; cross.forEach(([w, u, v2]) => estC[ekey(u, v2)] = 'hot');
    F.push({shapes:[region].concat(gShapes(G, {st:stCut, est:estC})),
      panels:[{lbl:{zh:'跨過 cut 的邊', en:'edges crossing the cut'},
        chips:cross.map(([w, u, v2]) => ({t:NM[u] + NM[v2] + ' ' + w, cls:w === 7 ? 'ok' : ''}))}],
      line:5,
      msg:{zh:'橫跨兩堆的邊有這幾條。<b>最後的生成樹至少要用掉其中一條</b>，' +
              '否則兩堆永遠連不起來——這是廢話，但正是證明的支點。',
           en:'These are the edges that cross. <b>Any spanning tree must use at least one of them</b>, otherwise the two groups are never joined. Obvious - and it is exactly the hinge of the proof.'}});
    F.push({shapes:[region].concat(gShapes(G, {st:stCut,
      est:Object.assign({}, estC, {[ekey(1, 4)]:'ok'})})),
      panels:[], line:5,
      msg:{zh:'其中最便宜的是 <b>BE = 7</b>。<b>cut property：這條邊一定屬於某棵最小生成樹。</b>' +
              '證明是一句話的<b>交換法</b>——假設某棵 MST 沒用它，把它加進去會形成一個環，' +
              '那個環一定還有另一條跨 cut 的邊（不然回不來），拿掉那條、換上 BE，' +
              '樹還是樹，總重量<b>只會更小或相等</b>。',
           en:'The cheapest of them is <b>BE = 7</b>. <b>Cut property: that edge belongs to some minimum spanning tree.</b> The proof is a one-line <b>exchange argument</b>: suppose an MST avoids it. Adding it creates a cycle, and that cycle must contain another edge crossing the same cut (otherwise it could not come back). Swap that one out for BE and you still have a tree, with total weight <b>no larger</b>.'}});
    F.push({shapes:[region].concat(gShapes(G, {st:stCut,
      est:Object.assign({}, estC, {[ekey(1, 4)]:'ok'})})),
      panels:[], line:5,
      msg:{zh:'關鍵在於這對<b>任何一種切法</b>都成立。Kruskal 每次收下一條邊時，' +
              '那條邊正好是「它兩端所在的元件 ｜ 其他所有點」這個 cut 上最便宜的跨越邊——' +
              '因為更便宜的都看過了，而且不是被收走就是被環擋掉。' +
              '<b>所以它每一步收的都是安全的邊，貪心不會後悔。</b>',
           en:'The point is that this holds for <b>every possible cut</b>. Each time Kruskal takes an edge, that edge is the cheapest crossing edge of the cut "the components at its two ends | everything else" - everything cheaper was already seen and was either taken or rejected as a cycle. <b>So every step takes a safe edge, and the greedy choice is never regretted.</b>'}});
    return F.list;
  }
  // cycle property
  const tree = {}; [[5, 0, 3], [5, 2, 4], [6, 3, 5], [7, 0, 1], [7, 1, 4], [9, 4, 6]]
    .forEach(([w, u, v2]) => tree[ekey(u, v2)] = 'done');
  F.push({shapes:gShapes(G, {est:Object.assign({}, tree)}), panels:[], line:10,
    msg:{zh:'反過來看被<b>跳過</b>的邊。Kruskal 掃到 <b>BD = 9</b> 時，B 和 D 已經相連了。' +
            '它憑什麼確定「以後也不會需要這條」？',
         en:'Now look at the edges that were <b>skipped</b>. When Kruskal reached <b>BD = 9</b>, B and D were already connected. How can it be sure that edge will never be needed?'}});
  F.push({shapes:gShapes(G, {est:Object.assign({}, tree, {[ekey(1, 3)]:'bad',
    [ekey(0, 1)]:'hot', [ekey(0, 3)]:'hot'})}), panels:[], line:10,
    msg:{zh:'因為 B 和 D 之間已經有一條路 <b>B–A–D（7 和 5）</b>，加上 BD 就形成一個環。' +
            '而環上的另外兩條邊<b>都比 9 便宜</b>——這不是巧合，是排序保證的：' +
            '<b>先被收走的一定比較便宜</b>。',
         en:'Because B and D are already joined by the path <b>B-A-D (7 and 5)</b>, adding BD closes a cycle. The other edges on that cycle are <b>both cheaper than 9</b> - not a coincidence but a guarantee of the sort order: <b>anything taken earlier is cheaper</b>.'}});
  F.push({shapes:gShapes(G, {est:Object.assign({}, tree, {[ekey(1, 3)]:'bad'})}),
    panels:[], line:10,
    msg:{zh:'這就是 <b>cycle property</b>：<b>一個環裡最貴的那條邊，一定不在任何最小生成樹裡</b>。' +
            '（真的用了它，只要拿掉它、換上環上任何一條，總重就變小。）' +
            '所以 union 回傳 False 時，直接丟掉是<b>安全的</b>，不需要回頭考慮。',
         en:'That is the <b>cycle property</b>: <b>the heaviest edge of a cycle is in no minimum spanning tree</b>. (If a tree used it, dropping it and reconnecting through the cycle would be strictly cheaper.) So when union returns False, discarding the edge is <b>safe</b> - it never needs reconsidering.'}});
  F.push({shapes:gShapes(G, {est:tree}), panels:[], line:8,
    msg:{zh:'兩條性質合起來就是 Kruskal 的正確性：<b>收下的是安全的（cut），丟掉的是沒用的（cycle）</b>。' +
            '成本則全押在排序上——<b>O(E log E)</b>；掃描本身只有 <b>O(E α(V))</b>，' +
            'α 是 Union-Find 的反阿克曼函數，實務上就是常數。',
         en:'Together the two properties give Kruskal its correctness: <b>what it takes is safe (cut), what it drops is useless (cycle)</b>. The cost sits almost entirely in the sort - <b>O(E log E)</b> - while the scan itself is <b>O(E a(V))</b>, where a is Union-Find\'s inverse Ackermann function, i.e. a constant in practice.'}});
  return F.list;
}

const PTS = [[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]];
const px = i => 1.0 + PTS[i][0] * .95, py = i => 6.3 - PTS[i][1] * .48;
function ptShapes(o){
  o = o || {}; const out = [];
  (o.edges || []).forEach(([w, u, v]) => {
    const st = (o.est && o.est[ekey(u, v)]) || 'ghost';
    out.push(S.e(px(u), py(u), px(v), py(v), {s:st, w:st === 'ok' ? .11 : .04, noArrow:true}));
    if (st !== 'ghost')
      out.push(S.t((px(u) + px(v)) / 2 + .18, (py(u) + py(v)) / 2 - .18, String(w),
        {c:st === 'bad' ? '#ff5c5c' : '#ffbe6b', fs:.28}));
  });
  for (let i = 0; i < PTS.length; i++)
    out.push(S.c(px(i), py(i), .40, (o.st && o.st[i]) || 'idle', String(i), {fs:.30,
      sub:'(' + PTS[i][0] + ',' + PTS[i][1] + ')', subfs:.24}));
  return out;
}
const CODE_LC = [
'def minCostConnectPoints(self, points):',
'    n = len(points)',
'    edges = []',
'    for i, j in combinations(range(n), 2):      # 完全圖：n(n-1)/2 條邊',
'        (x1, y1), (x2, y2) = points[i], points[j]',
'        edges.append((abs(x1 - x2) + abs(y1 - y2), i, j))',
'',
'    uf, total, taken = UnionFind(n), 0, 0',
'    for w, u, v in sorted(edges):               # 這一行就是 O(n^2 log n)',
'        if uf.union(u, v):',
'            total += w; taken += 1',
'            if taken == n - 1: break',
'    return total'];

function buildLC(){
  const F = new Frames(), n = PTS.length, edges = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++)
    edges.push([Math.abs(PTS[i][0] - PTS[j][0]) + Math.abs(PTS[i][1] - PTS[j][1]), i, j]);
  F.push({shapes:ptShapes({}), panels:[], line:0,
    msg:{zh:'<b>LeetCode 1584</b>：把所有點連起來，成本是曼哈頓距離，求最小總成本。' +
            '題目沒有給圖——<b>「連起來、最便宜」這幾個字就是最小生成樹</b>，看出來就解完一半。',
         en:'<b>LeetCode 1584</b>: connect all the points, cost is Manhattan distance, minimise the total. No graph is given - <b>the words "connect everything, cheapest" are the definition of a minimum spanning tree</b>, and spotting that is half the solution.'}});
  const all = {}; edges.forEach(([w, u, v]) => all[ekey(u, v)] = 'soft');
  F.push({shapes:ptShapes({edges:edges, est:all}), panels:[
    {lbl:{zh:'邊數', en:'edges'}, chips:[{t:'n = 5', cls:''}, {t:'n(n-1)/2 = 10', cls:'hot'}]}], line:3,
    msg:{zh:'圖要自己造：<b>任兩點之間都有一條邊</b>，所以這是一張<b>完全圖</b>，' +
            '5 個點就有 10 條邊，n = 1000 就有將近 50 萬條。',
         en:'You build the graph yourself: <b>every pair of points is an edge</b>, so this is a <b>complete graph</b> - 10 edges for 5 points, and nearly half a million when n = 1000.'}});
  const uf = UF(n), est = {}; let total = 0, taken = 0;
  const sorted = edges.slice().sort((a, b) => a[0] - b[0]);
  sorted.forEach(([w, u, v]) => {
    if (taken === n - 1) return;
    const ok = uf.union(u, v);
    est[ekey(u, v)] = ok ? 'ok' : 'bad';
    if (ok){ total += w; taken++; }
    F.push({shapes:ptShapes({edges:edges, est:Object.assign({}, all, est), st:{[u]:ok ? 'ok' : 'bad', [v]:ok ? 'ok' : 'bad'}}),
      panels:[{lbl:{zh:'狀態', en:'state'}, chips:[{t:(LANG === 'zh' ? '已選 ' : 'taken ') + taken + '/4', cls:''},
        {t:(LANG === 'zh' ? '總成本 ' : 'total ') + total, cls:'hot'}]}], line:ok ? 10 : 9,
      msg:ok ? {zh:'<b>' + u + '–' + v + '</b>（距離 ' + w + '）連的是兩個還沒相連的點 → 收下，總成本 ' + total + '。',
                en:'<b>' + u + '-' + v + '</b> (distance ' + w + ') joins two points that were not yet connected: take it, total ' + total + '.'}
             : {zh:'<b>' + u + '–' + v + '</b>（距離 ' + w + '）兩端已經連通 → 跳過。' +
                   '完全圖裡這種邊<b>特別多</b>，這也是為什麼「先排序再靠 Union-Find 過濾」才划算。',
                en:'<b>' + u + '-' + v + '</b> (distance ' + w + ') has both ends connected already: skip. Complete graphs produce <b>a lot</b> of these, which is exactly why "sort once, then filter with Union-Find" pays off.'}});
  });
  F.push({shapes:ptShapes({edges:edges, est:Object.assign({}, all, est)}),
    panels:[{lbl:{zh:'答案', en:'answer'}, chips:[{t:String(total), cls:'ok'}]}], line:11,
    msg:{zh:'答案 <b>' + total + '</b>。複雜度由排序決定：<b>O(n² log n)</b>，' +
            'n ≤ 1000 綽綽有餘。不過完全圖是<b>稠密圖</b>，' +
            '用 heap 逐點長出來的 Prim 是 O(n²)，理論上更適合這一題——明天就寫它。',
         en:'The answer is <b>' + total + '</b>. Complexity is set by the sort, <b>O(n^2 log n)</b>, comfortable at n <= 1000. But a complete graph is <b>dense</b>, and Prim - which grows the tree vertex by vertex with a heap - runs in O(n^2) here, which fits the problem better. That is tomorrow.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'最小生成樹：Kruskal', en:'Minimum Spanning Tree - Kruskal'},
  sub:{zh:'邊由便宜排到貴，一條一條收，只要不繞回自己就留下。判斷「會不會繞回自己」的那一步，就是昨天的 Union-Find。',
       en:'Sort the edges cheapest first and keep each one unless it closes a cycle. The cycle test is yesterday\'s Union-Find.'},
  tabs:[
    {id:'scan', label:{zh:'掃過排序後的邊', en:'scan the sorted edges'},
     stage:{zh:'便宜的先看，成環就跳過', en:'Cheapest first, skip anything that closes a cycle'},
     view:[10, 7.2],
     variants:[{zh:'連通圖 → 生成樹', en:'connected: a tree'},
               {zh:'斷開的圖 → 生成森林', en:'disconnected: a forest'}],
     idea:{zh:'Kruskal 只有兩件事：<b>sorted(edges)</b> 和 <b>uf.union(u, v)</b>。' +
              '前者決定貪心的順序，後者用 O(α) 回答「會不會成環」。' +
              '收滿 <b>V−1</b> 條就可以停，剩下的邊看都不用看。',
           en:'Kruskal is two things: <b>sorted(edges)</b> and <b>uf.union(u, v)</b>. The first fixes the greedy order, the second answers "does this close a cycle?" in O(a). Once <b>V-1</b> edges are in, the rest of the list is dead weight.'},
     legend:['act', 'ok', 'bad', 'ghost'], code:CODE, build:buildScan},
    {id:'why', label:{zh:'為什麼貪心是對的', en:'why greedy is correct'},
     stage:{zh:'cut property 與 cycle property', en:'The cut and cycle properties'},
     view:[10, 7.2],
     variants:[{zh:'收下的邊為何安全', en:'why a taken edge is safe'},
               {zh:'丟掉的邊為何沒用', en:'why a skipped edge is useless'}],
     idea:{zh:'貪心演算法的正確性都要靠一個「安全」定理。這裡是 <b>cut property</b>：' +
              '任何一種切法上<b>最便宜的跨越邊</b>都屬於某棵 MST；' +
              '以及 <b>cycle property</b>：任何一個環裡<b>最貴的邊</b>都不屬於任何 MST。' +
              '這兩句話一收一丟，剛好就是 Kruskal 迴圈裡的兩個分支。',
           en:'Every greedy algorithm needs a safety theorem. Here they are the <b>cut property</b> - the cheapest edge crossing any cut belongs to some MST - and the <b>cycle property</b> - the heaviest edge of any cycle belongs to none. One sentence for taking, one for dropping: exactly the two branches of Kruskal\'s loop.'},
     legend:['hot', 'ok', 'bad', 'soft'], code:CODE, build:buildWhy},
    {id:'lc', label:{zh:'LC 1584 連接所有點', en:'LC 1584 connect all points'},
     stage:{zh:'題目沒說是圖，但它就是 MST', en:'No graph in the statement - but it is an MST'},
     view:[10, 7.2],
     idea:{zh:'看到「把所有東西連起來，成本最小」就想到 MST。' +
              '這題的圖是<b>自己造的完全圖</b>，邊數 n(n−1)/2；' +
              '稀疏圖用 Kruskal（成本在排序），稠密圖用 Prim（成本在取最小）。',
           en:'"Connect everything at minimum cost" should read as MST. The graph here is a <b>complete graph you build yourself</b> with n(n-1)/2 edges. Sparse graphs favour Kruskal (cost is the sort); dense graphs favour Prim (cost is the repeated minimum).'},
     legend:['ok', 'bad', 'soft', 'idle'], code:CODE_LC, build:buildLC}
  ]
};
