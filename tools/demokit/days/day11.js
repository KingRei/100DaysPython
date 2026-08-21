// DAY: 11
// TITLE_ZH: 圖：把「關係」變成資料結構
// TITLE_EN: Graph - turning relationships into a data structure
// SUB_ZH: 點是角色、邊是他們一起出現的次數。用相鄰串列存起來，就能問「誰最重要」。
// SUB_EN: Vertices are characters, edges are how often they co-occur.
// FOLDER: day%2011%20-%20graph
// MEDIUM: https://medium.com/100-days-of-python/day-11-data-structure-graph-a4229c3dabaf

const NAMES = ['Arya', 'Jon', 'Robb', 'Bran', 'Sansa'];
const POS = {Arya:[2.0, 4.1], Jon:[3.5, 1.5], Robb:[5.2, 4.3], Bran:[6.9, 1.5], Sansa:[8.3, 4.1]};
const EDGES = [['Arya', 'Jon', 11], ['Arya', 'Robb', 15], ['Jon', 'Robb', 17],
               ['Robb', 'Bran', 24], ['Robb', 'Sansa', 47], ['Bran', 'Sansa', 8]];
const RAD = .58;

function graphShapes(o){
  o = o || {}; const out = [], vs = o.verts || NAMES, es = o.edges || [];
  es.forEach(([a, b, w]) => {
    const [x1, y1] = POS[a], [x2, y2] = POS[b];
    const st = (o.est && o.est[a + '-' + b]) || 'soft';
    const dx = x2 - x1, dy = y2 - y1, L = Math.sqrt(dx * dx + dy * dy);
    const ux = dx / L, uy = dy / L;
    out.push(S.e(x1 + ux * RAD, y1 + uy * RAD, x2 - ux * RAD, y2 - uy * RAD,
      {s:st, w:.05 + Math.min(w, 50) / 50 * .10, noArrow:!o.directed}));
    if (o.weights !== false)
      out.push(S.t((x1 + x2) / 2 + uy * .34, (y1 + y2) / 2 - ux * .34, String(w),
        {c:st === 'soft' ? '#8fa3ac' : '#ffbe6b', fs:.28}));
  });
  vs.forEach(n => {
    const [x, y] = POS[n];
    out.push(S.c(x, y, RAD, (o.st && o.st[n]) || 'idle', n, {fs:.30}));
  });
  return out;
}
function adjPanel(adj, hi){
  return {lbl:{zh:'_vertDict（相鄰串列）', en:'_vertDict (adjacency list)'},
    chips:NAMES.filter(n => adj[n]).map(n => ({
      t:n + ' -> {' + Object.keys(adj[n]).map(k => k + ':' + adj[n][k]).join(', ') + '}',
      cls:hi === n ? 'hot' : ''}))};
}

const CODE_BUILD = [
'class Vertex:',
'    def __init__(self, node):',
'        self._id = node',
'        self._adjacent = {}      # 鄰居 -> 權重',
'',
'class Graph:',
'    def __init__(self):',
'        self._vertDict = {}      # 名字 -> Vertex',
'',
'    def add_vertex(self, node):',
'        if node in self._vertDict: return self._vertDict[node]',
'        self._vertDict[node] = Vertex(node); return self._vertDict[node]',
'',
'    def add_edge(self, frm, to, w=0):',
'        self.add_vertex(frm); self.add_vertex(to)',
'        self._vertDict[frm].set_neighbor_weight(self._vertDict[to], w)'];
const CODE_REP = [
'# 相鄰串列 adjacency list：每個點記自己的鄰居',
'adj = {v: {} for v in V}          # 空間 O(V + E)',
'w = adj[u].get(v)                 # 查一條邊：O(deg(u))',
'for nb in adj[u]:                 # 走訪鄰居：O(deg(u))  <- 大部分演算法要的',
'',
'# 相鄰矩陣 adjacency matrix：V x V 的表',
'M = [[0] * len(V) for _ in V]     # 空間 O(V^2)',
'w = M[i][j]                       # 查一條邊：O(1)',
'for j in range(len(V)):           # 走訪鄰居：O(V)，就算只有兩個鄰居'];
const CODE_CENT = [
'degree_centrality = {}',
'weighted_degree_centrality = {}',
'for v in GoT:',
'    degree_centrality[v.get_id()] = v.get_degree_centrality()',
'    weighted_degree_centrality[v.get_id()] = v.get_weighted_degree_centrality()',
'',
'# get_degree_centrality  -> len(self._adjacent)',
'# get_weighted_degree_centrality -> sum(self._adjacent.values())',
'print(sorted(degree_centrality.items(), key=lambda d: d[1], reverse=True)[:5])'];

function buildBuild(){
  const F = new Frames(), adj = {}, done = [];
  F.push({shapes:[S.t(5, 3, {zh:'一張空的圖', en:'an empty graph'}, {c:'#8fa3ac', fs:.42})],
    panels:[{lbl:'_vertDict', chips:[{t:'{}', cls:''}]}], line:6,
    msg:{zh:'圖只有兩樣東西：<b>點（vertex）</b>和<b>邊（edge）</b>。' +
            '前面幾天的結構都在描述「順序」或「包含」，圖描述的是<b>任意兩個東西之間的關係</b>——' +
            '這也是為什麼它能同時表示地圖、社交網路、相依性和狀態機。',
         en:'A graph is just <b>vertices</b> and <b>edges</b>. Earlier structures described order or containment; a graph describes <b>an arbitrary relationship between any two things</b> - which is why the same structure models maps, social networks, dependencies and state machines.'}});
  EDGES.forEach(([a, b, w], k) => {
    if (!adj[a]) adj[a] = {};
    if (!adj[b]) adj[b] = {};
    const newV = [a, b].filter(n => done.indexOf(n) < 0);
    newV.forEach(n => done.push(n));
    F.push({shapes:graphShapes({verts:done, edges:EDGES.slice(0, k),
      st:Object.fromEntries(newV.map(n => [n, 'act']))}), panels:[adjPanel(adj)], line:9,
      msg:newV.length ? {zh:'<b>add_edge</b> 會先確保兩端的點存在：新增 ' +
                             newV.map(n => '<b>' + n + '</b>').join('、') +
                             '。點存在 dict 裡，所以「用名字找點」是 O(1)。',
                         en:'<b>add_edge</b> first makes sure both endpoints exist: adding ' +
                            newV.map(n => '<b>' + n + '</b>').join(', ') +
                            '. Vertices live in a dict, so looking one up by name is O(1).'}
                      : {zh:'<b>' + a + '</b> 和 <b>' + b + '</b> 都已經在圖裡了，不用重建。',
                         en:'Both <b>' + a + '</b> and <b>' + b + '</b> already exist, so nothing is recreated.'}});
    adj[a][b] = w;
    F.push({shapes:graphShapes({verts:done, edges:EDGES.slice(0, k + 1),
      est:{[a + '-' + b]:'ok'}, st:{[a]:'hot', [b]:'act'}}), panels:[adjPanel(adj, a)], line:14,
      msg:{zh:'把邊記在 <b>' + a + '._adjacent</b> 裡：<code>' + b + ' -> ' + w + '</code>。' +
              '權重就是他們在《冰與火之歌》裡一起出現的次數。' +
              '<b>邊記在點身上</b>，所以圖多大都不影響「拿到某個點的鄰居」有多快。',
           en:'The edge is stored inside <b>' + a + '._adjacent</b> as <code>' + b + ' -> ' + w +
              '</code>; the weight is how often they co-occur in the books. <b>Edges live on the vertex</b>, so the size of the graph never affects how fast you can list one vertex\'s neighbours.'}});
  });
  F.push({shapes:graphShapes({edges:EDGES}), panels:[adjPanel(adj),
    {lbl:{zh:'規模', en:'size'}, chips:[{t:'V = 5', cls:''}, {t:'E = 6', cls:''}]}], line:14,
    msg:{zh:'注意 <b>add_edge 只寫了一個方向</b>：' + EDGES[0][0] + ' 的鄰居有 ' + EDGES[0][1] +
            '，但反過來沒有。要無向圖就得兩邊都寫一次——' +
            '這個小細節等一下算中心性時會咬人。',
         en:'Note that <b>add_edge only writes one direction</b>: ' + EDGES[0][1] +
           ' appears among ' + EDGES[0][0] +
           '\'s neighbours but not the other way round. An undirected graph needs both writes - a detail that bites us in the centrality tab.'}});
  return F.list;
}

function buildRep(v){
  const F = new Frames();
  const adj = {}; EDGES.forEach(([a, b, w]) => { (adj[a] = adj[a] || {})[b] = w; (adj[b] = adj[b] || {})[a] = w; });
  if (v === 0){
    F.push({shapes:graphShapes({edges:EDGES}), panels:[adjPanel(adj)], line:1,
      msg:{zh:'<b>相鄰串列</b>：每個點只記自己的鄰居。空間是 <b>O(V + E)</b>，' +
              '圖愈稀疏愈省——真實世界的圖幾乎都很稀疏。',
           en:'<b>Adjacency list</b>: each vertex records only its own neighbours. Space is <b>O(V + E)</b>, and the sparser the graph the better - and real-world graphs are almost always sparse.'}});
    F.push({shapes:graphShapes({edges:EDGES, st:{Robb:'hot', Arya:'act', Jon:'act', Bran:'act', Sansa:'act'},
      est:{'Arya-Robb':'ok', 'Jon-Robb':'ok', 'Robb-Bran':'ok', 'Robb-Sansa':'ok'}}),
      panels:[adjPanel(adj, 'Robb')], line:3,
      msg:{zh:'「<b>Robb 的鄰居有誰？</b>」直接把他的 dict 讀出來，' +
              '成本是 <b>O(deg(u))</b>——只跟他自己的朋友數有關，跟整張圖多大無關。' +
              '走訪鄰居正是圖演算法最常做的事，所以這個代價才是重點。',
           en:'"<b>Who are Robb\'s neighbours?</b>" is a single dict read, costing <b>O(deg(u))</b> - proportional to his own friend count, not to the size of the graph. Iterating neighbours is what graph algorithms do all day, which is why this cost is the one that matters.'}});
    F.push({shapes:graphShapes({edges:EDGES, st:{Arya:'hot', Sansa:'bad'}}), panels:[adjPanel(adj)], line:2,
      msg:{zh:'但「<b>Arya 和 Sansa 之間有邊嗎？</b>」得掃過 Arya 的鄰居才知道，' +
              '最壞情況 O(deg(u))。如果你的演算法一直在問這種問題，就該換一種表示法。',
           en:'But "<b>is there an edge between Arya and Sansa?</b>" means scanning Arya\'s neighbours - O(deg(u)) in the worst case. If your algorithm keeps asking that question, you want the other representation.'}});
    return F.list;
  }
  const N = NAMES.length, cw = .95, x0 = 3.2, y0 = 1.6;
  const cell = (hi) => {
    const out = [];
    NAMES.forEach((n, j) => out.push(S.t(x0 + j * cw + cw / 2, y0 - .28, n, {c:'#c7a6ff', fs:.24})));
    NAMES.forEach((n, i) => {
      out.push(S.t(x0 - .18, y0 + i * cw + cw / 2 + .08, n, {c:'#c7a6ff', fs:.24, anchor:'end'}));
      NAMES.forEach((m, j) => {
        const w = adj[n] && adj[n][m] ? adj[n][m] : 0;
        const k = i + ',' + j;
        out.push(S.r(x0 + j * cw, y0 + i * cw, cw - .06, cw - .06,
          (hi && hi[k]) || (w ? 'idle' : 'ghost'), w ? String(w) : '0', {fs:.28}));
      });
    });
    return out;
  };
  F.push({shapes:cell(), panels:[], line:5,
    msg:{zh:'<b>相鄰矩陣</b>：一張 V x V 的表，第 i 列第 j 行放 i→j 的權重。' +
            '這裡 25 格只有 12 格有值——<b>其餘 13 格是為了「沒有這條邊」而付的錢</b>。',
         en:'<b>Adjacency matrix</b>: a V x V table where row i, column j holds the weight of i to j. Only 12 of these 25 cells carry data - <b>the other 13 are the price of recording absence</b>.'}});
  F.push({shapes:cell({'0,4':'ok'}), panels:[], line:6,
    msg:{zh:'好處很實在：「Arya 和 Sansa 之間有邊嗎？」= <code>M[0][4]</code>，' +
            '<b>O(1)</b>，而且是一次連續記憶體讀取。',
         en:'The upside is real: "is there an edge between Arya and Sansa?" is <code>M[0][4]</code> - <b>O(1)</b>, and a single contiguous memory read.'}});
  F.push({shapes:cell(Object.fromEntries(NAMES.map((_, j) => ['2,' + j, 'hot']))), panels:[], line:7,
    msg:{zh:'壞處也很實在：要列出 Robb 的鄰居得掃過<b>整整一列 O(V)</b>，' +
            '空間是 <b>O(V²)</b>。107 個角色只有 352 條邊，用矩陣就是 11449 格裝 352 個數字。' +
            '<b>稀疏圖用串列、稠密圖或需要大量「有沒有邊」查詢才用矩陣。</b>',
         en:'So is the downside: listing Robb\'s neighbours scans <b>a whole row, O(V)</b>, and space is <b>O(V^2)</b>. With 107 characters and 352 edges, a matrix means 11,449 cells holding 352 numbers. <b>Lists for sparse graphs; matrices for dense ones or when you constantly ask "is there an edge?"</b>'}});
  return F.list;
}

const TOP_D = [['Tyrion', 24], ['Robb', 23], ['Sansa', 23], ['Jon', 21], ['Tywin', 19]];
const TOP_W = [['Jon', 397], ['Sansa', 354], ['Robb', 286], ['Bran', 283], ['Tyrion', 264]];
function bars(rows, title){
  const out = [S.t(5, .9, title, {c:'#ffbe6b', fs:.36})], max = rows[0][1];
  rows.forEach(([n, v], i) => {
    const y = 1.6 + i * .95;
    out.push(S.t(2.2, y + .42, n, {c:'#dff2f5', fs:.32, anchor:'end'}));
    out.push(S.r(2.4, y, Math.max(.3, v / max * 5.6), .68, i === 0 ? 'act' : 'idle', '', {}));
    out.push(S.t(2.4 + Math.max(.3, v / max * 5.6) + .25, y + .44, String(v), {c:'#3fe0dd', fs:.30, anchor:'start'}));
  });
  return out;
}
function buildCent(v){
  const F = new Frames();
  const adj = {}; EDGES.forEach(([a, b, w]) => { (adj[a] = adj[a] || {})[b] = w; (adj[b] = adj[b] || {})[a] = w; });
  F.push({shapes:graphShapes({edges:EDGES, st:{Robb:'hot'}}), panels:[adjPanel(adj, 'Robb')], line:6,
    msg:{zh:'有了圖就能問問題。最簡單的一種是<b>度中心性（degree centrality）</b>：' +
            '一個點連了幾條邊，就是 <code>len(self._adjacent)</code>——' +
            '結構已經替你算好了，不需要任何額外走訪。',
         en:'Once the graph exists we can ask questions. The simplest is <b>degree centrality</b>: how many edges a vertex has, i.e. <code>len(self._adjacent)</code>. The structure already knows - no traversal needed.'}});
  if (v === 0){
    F.push({shapes:bars(TOP_D, {zh:'度中心性 top 5（107 個角色，352 條邊）',
      en:'degree centrality top 5 (107 vertices, 352 edges)'}), panels:[], line:3,
      msg:{zh:'把整份 <b>stormofswords.csv</b> 讀進來（107 個角色、352 條邊），' +
              'Tyrion 以 24 個鄰居居冠。<b>他認識的人最多</b>。',
           en:'Loading the whole <b>stormofswords.csv</b> (107 characters, 352 edges), Tyrion leads with 24 neighbours: <b>he knows the most people</b>.'}});
    F.push({shapes:bars(TOP_W, {zh:'加權度中心性 top 5', en:'weighted degree centrality top 5'}),
      panels:[], line:4,
      msg:{zh:'換成 <b>加權</b>版本（<code>sum(self._adjacent.values())</code>，也就是同場次數總和），' +
              '第一名變成 <b>Jon</b>。同一張圖、同一個「中心性」概念，' +
              '<b>只因為權重要不要算進去，答案就不一樣</b>。',
           en:'Switch to the <b>weighted</b> version (<code>sum(self._adjacent.values())</code>, total co-occurrences) and <b>Jon</b> takes the top spot. Same graph, same notion of centrality - <b>counting weights or not changes the answer</b>.'}});
    F.push({shapes:bars(TOP_D, {zh:'認識的人多 vs 相處的時間長', en:'many contacts vs deep contacts'}),
      panels:[], line:8,
      msg:{zh:'兩份榜單講的是不同的事：度數高＝<b>接觸面廣</b>（適合傳播、找八卦來源），' +
              '加權度數高＝<b>關係深</b>。選哪一個要看你想回答什麼問題，' +
              '而不是哪個數字比較好看。',
           en:'The two rankings mean different things: high degree is <b>reach</b> (good for diffusion), high weighted degree is <b>depth of ties</b>. Which one you want depends on the question you are asking, not on which number looks nicer.'}});
  } else {
    F.push({shapes:graphShapes({edges:EDGES, directed:true, st:{Robb:'hot'},
      est:{'Robb-Bran':'ok', 'Robb-Sansa':'ok'}}), panels:[adjPanel(adj, 'Robb')], line:6,
      msg:{zh:'但要小心一個陷阱：<code>add_edge(frm, to, w)</code> <b>只寫進 frm 的鄰居表</b>。' +
              'CSV 每列只讀一次，所以這其實是一張<b>有向圖</b>。',
           en:'One trap, though: <code>add_edge(frm, to, w)</code> <b>only writes into frm\'s neighbour dict</b>. Each CSV row is read once, so what we actually built is a <b>directed</b> graph.'}});
    F.push({shapes:bars([['Tyrion', 36], ['Jon', 26], ['Sansa', 26], ['Robb', 25], ['Jaime', 24]],
      {zh:'兩邊都寫（無向）時的度中心性', en:'degree centrality if both directions are stored'}),
      panels:[], line:3,
      msg:{zh:'如果補上反方向（<code>add_edge(b, a, w)</code>），數字就變成這樣——' +
              'Tyrion 從 24 變 36，名次也動了。' +
              '<b>「誰最重要」的答案，取決於你怎麼把資料塞進結構裡。</b>',
           en:'Add the reverse direction (<code>add_edge(b, a, w)</code>) and the numbers change: Tyrion goes from 24 to 36 and the order shifts. <b>The answer to "who matters most" depends on how you loaded the data.</b>'}});
    F.push({shapes:graphShapes({edges:EDGES}), panels:[], line:8,
      msg:{zh:'所以建圖時先問自己：<b>這個關係是對稱的嗎？</b>' +
              '一起出現、成為朋友 → 無向；追蹤、依賴、單行道、指向 → 有向。' +
              '搞錯了，後面所有的走訪、排序、最短路徑都會跟著錯。',
           en:'So when building a graph, ask first: <b>is this relationship symmetric?</b> Co-occurrence and friendship are undirected; following, dependency, one-way streets and references are directed. Get it wrong and every traversal, ordering and shortest path built on top inherits the mistake.'}});
  }
  return F.list;
}

const DAY_META = {
  title:{zh:'圖：把「關係」變成資料結構', en:'Graph - turning relationships into a data structure'},
  sub:{zh:'點是角色、邊是他們一起出現的次數。用相鄰串列存起來，就能問「誰最重要」。',
       en:'Vertices are characters, edges are how often they co-occur. Store it as an adjacency list and you can ask who matters most.'},
  tabs:[
    {id:'build', label:{zh:'建圖', en:'build the graph'},
     stage:{zh:'add_vertex 與 add_edge', en:'add_vertex and add_edge'}, view:[10, 6.0],
     idea:{zh:'圖 = 點 + 邊。每個點自己保管一本<b>鄰居名冊</b>（dict：鄰居 → 權重），' +
              '整張圖再用一本 dict 保管所有點。兩層 dict，就這樣。',
           en:'A graph is vertices plus edges. Each vertex keeps its own <b>neighbour book</b> (a dict of neighbour to weight) and the graph keeps a dict of all vertices. Two dicts - that is the whole structure.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_BUILD, build:buildBuild},
    {id:'rep', label:{zh:'串列 vs 矩陣', en:'list vs matrix'},
     stage:{zh:'同一張圖，兩種存法', en:'One graph, two representations'}, view:[10, 6.6],
     variants:[{zh:'相鄰串列', en:'adjacency list'}, {zh:'相鄰矩陣', en:'adjacency matrix'}],
     idea:{zh:'串列空間 O(V+E)、列鄰居 O(deg)、查一條邊 O(deg)；' +
              '矩陣空間 O(V²)、列鄰居 O(V)、查一條邊 O(1)。' +
              '<b>大部分圖演算法都在「列鄰居」，所以稀疏圖用串列。</b>',
           en:'A list costs O(V+E) space, O(deg) to list neighbours, O(deg) to test one edge. A matrix costs O(V^2) space, O(V) to list neighbours, O(1) to test one edge. <b>Most graph algorithms spend their time listing neighbours, so sparse graphs use lists.</b>'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_REP, build:buildRep},
    {id:'cent', label:{zh:'度中心性', en:'degree centrality'},
     stage:{zh:'誰是這個網路的中心？', en:'Who sits at the centre?'}, view:[10, 6.6],
     variants:[{zh:'加權 vs 不加權', en:'weighted vs unweighted'}, {zh:'有向的陷阱', en:'the directed trap'}],
     idea:{zh:'度中心性是最便宜的重要性指標：<b>不用走訪，結構本身就是答案</b>。' +
              '但它只看一步之遙——真正的「重要」還可以定義成' +
              '「多少最短路徑經過我」或「重要的人都指向我」，' +
              '那需要最短路徑與特徵向量的工具。',
           en:'Degree centrality is the cheapest importance measure: <b>no traversal at all, the structure is the answer</b>. But it only looks one hop out. Importance can also mean "how many shortest paths run through me" or "important people point at me" - those need shortest-path and eigenvector machinery.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_CENT, build:buildCent}
  ]
};
