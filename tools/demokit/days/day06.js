// DAY: 06
// TITLE_ZH: 二項堆積
// TITLE_EN: Binomial Heap
// SUB_ZH: 一堆二項樹排成一個二進位數：合併兩個堆積就是做一次加法，進位就是把兩棵樹連起來。
// SUB_EN: A forest that behaves like a binary number.
// FOLDER: day%2006%20-%20binomial%20heap
// MEDIUM: https://medium.com/100-days-of-python/day-06-binomial-heap-88ca2edb8255

const T0 = v => ({v:v, kids:[]});
const ORD = t => t.kids.length;
function link(a, b){                    // 兩棵同階的樹 -> 階數 +1，小的當根
  if (a.v <= b.v){ a.kids.unshift(b); return a; }
  b.kids.unshift(a); return b;
}
function clone(t){ return {v:t.v, kids:t.kids.map(clone)}; }
function countNodes(t){ return 1 + t.kids.reduce((s, k) => s + countNodes(k), 0); }
function leaves(t){ return t.kids.length ? t.kids.reduce((s, k) => s + leaves(k), 0) : 1; }

function treeShapes(t, x0, y0, unit, rowH, states, mark){
  const out = [], r = .28;
  let cursor = x0;
  (function place(n, d){
    if (!n.kids.length){ n._x = cursor + unit / 2; cursor += unit; }
    else { n.kids.forEach(k => place(k, d + 1));
           n._x = (n.kids[0]._x + n.kids[n.kids.length - 1]._x) / 2; }
    n._y = y0 + d * rowH;
  })(t, 0);
  (function draw(n){
    n.kids.forEach(k => { out.push(S.e(n._x, n._y, k._x, k._y,
      {pad:r + .03, arrow:false, s:(states && states[n.v]) === 'hot' ? 'hot' : 'idle'})); draw(k); });
  })(t);
  (function draw2(n){
    out.push(S.c(n._x, n._y, r, (states && states[n.v]) || 'idle', String(n.v), {fs:.32}));
    n.kids.forEach(draw2);
  })(t);
  if (mark) out.push(S.t((function root(n){ return n._x; })(t), y0 - .48, mark, {c:'#3fe0dd', fs:.30}));
  return out;
}
function forestShapes(trees, y0, states, labels){
  const out = []; let x = .55;
  const unit = .72, gapT = .85;
  trees.forEach((t, i) => {
    const w = leaves(t) * unit;
    out.push.apply(out, treeShapes(t, x, y0, unit, 1.05, states,
      labels === false ? null : 'B' + ORD(t)));
    x += w + gapT;
  });
  return out;
}
const bin = trees => { let s = 0; trees.forEach(t => s += Math.pow(2, ORD(t))); return s; };
const binStr = n => n.toString(2);
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));
const heapPanel = (trees, lbl) => ({lbl:lbl || {zh:'目前的森林', en:'current forest'},
  chips:trees.length ? trees.map(t => ({t:'B' + ORD(t) + ' (' + countNodes(t) + ')', cls:'act'})) : []});
const cntPanel = trees => ({lbl:{zh:'節點數 / 二進位', en:'node count / binary'},
  chips:[{t:bin(trees) + ' = 0b' + binStr(bin(trees)), cls:'ok'}]});

const CODE_LINK = [
'def link(a, b):                 # 兩棵「同階」的二項樹',
'    if a.key > b.key:',
'        a, b = b, a             # 小的當根，才維持 min-heap',
'    b.sibling = a.child',
'    a.child = b                 # b 整棵掛到 a 底下',
'    a.degree += 1               # 階數 +1，節點數變兩倍',
'    return a'];
const CODE_MERGE = [
'def merge(h1, h2):              # 就是二進位加法',
'    out, carry = [], None',
'    for d in range(max_degree + 1):',
'        bucket = [t for t in (h1, h2, [carry]) if t has degree d]',
'        if len(bucket) == 1:',
'            out.append(bucket[0])          # 這一位是 1',
'            carry = None',
'        elif len(bucket) >= 2:',
'            carry = link(bucket[0], bucket[1])   # 進位！',
'            if len(bucket) == 3: out.append(bucket[2])',
'    return out'];
const CODE_EXT = [
'def extract_min(self):',
'    t = min(self.trees, key=lambda t: t.key)   # 只看每棵樹的根',
'    self.trees.remove(t)',
'    kids = t.children                  # B(k-1), ..., B(1), B(0)',
'    self.trees = merge(self.trees, kids)       # 小孩自己就是一個合法的堆積',
'    return t.key'];

function buildLink(){
  const F = new Frames();
  let ts = [T0(5), T0(9), T0(3), T0(7), T0(6), T0(12), T0(4), T0(8)];
  F.push({shapes:forestShapes(ts, 2.4, {}), panels:[heapPanel(ts)], line:0,
    msg:{zh:'先看零件。<b>B₀</b> 就是一個節點。二項樹只有一種造法：<b>兩棵同階的樹接在一起</b>。',
         en:'Start with the parts. <b>B₀</b> is a single node. There is only one way to build binomial trees: <b>join two of the same order</b>.'}});
  let round = 0;
  while (ts.length > 1){
    const nxt = [];
    for (let i = 0; i < ts.length; i += 2){
      const a = ts[i], b = ts[i + 1];
      F.push({shapes:forestShapes(ts, 2.4, {[a.v]:'hot', [b.v]:'act'}), panels:[heapPanel(ts)], line:1,
        msg:{zh:'要接 <b>' + a.v + '</b> 和 <b>' + b.v + '</b> 這兩棵 B' + ORD(a) +
                '。誰當根？<b>比較小的那個</b>，否則就不是 min-heap 了。',
             en:'Joining <b>' + a.v + '</b> and <b>' + b.v + '</b>, both order ' + ORD(a) +
                '. Which becomes the root? <b>The smaller key</b>, or the min-heap property dies.'}});
      nxt.push(link(a, b));
    }
    ts = nxt; round++;
    F.push({shapes:forestShapes(ts, 2.4, {}), panels:[heapPanel(ts), cntPanel(ts)], line:5,
      msg:{zh:'得到 ' + ts.length + ' 棵 <b>B' + round + '</b>，每棵有 <b>' + Math.pow(2, round) +
              '</b> 個節點。規律：<b>Bk 有 2^k 個節點、樹根剛好有 k 個小孩、高度 k</b>。',
           en:'Now ' + ts.length + ' trees of order <b>' + round + '</b>, each with <b>' + Math.pow(2, round) +
              '</b> nodes. The pattern: <b>Bk holds 2^k nodes, its root has exactly k children, and its height is k</b>.'}});
  }
  F.push({shapes:forestShapes(ts, 2.0, {[ts[0].v]:'ok'}), panels:[heapPanel(ts), cntPanel(ts)], line:6,
    msg:{zh:'這就是為什麼它叫「二項」：<b>Bk 的第 d 層剛好有 C(k, d) 個節點</b>，' +
            '正是二項式係數。節點數永遠是 2 的次方——接下來整個結構都建立在這件事上。',
         en:'That is where the name comes from: <b>level d of Bk holds exactly C(k, d) nodes</b>, the binomial coefficients. The node count is always a power of two, and everything else is built on that fact.'}});
  return F.list;
}

function buildMerge(v){
  const F = new Frames();
  const mk = vals => vals.map(T0);
  let H1, H2, story;
  if (v === 0){
    H1 = [T0(7), link(T0(3), T0(9))];           // B0 + B1  -> 3 nodes
    H2 = [T0(5), link(T0(2), T0(8))];           // B0 + B1  -> 3 nodes
    story = {zh:'兩個各 3 個節點的堆積（二進位 <b>11</b>）要合併。3 + 3 = 6 = <b>110</b>，' +
                '所以答案一定是「一棵 B2 加一棵 B1」——形狀在動手之前就已經算出來了。',
             en:'Two heaps of 3 nodes each (binary <b>11</b>). 3 + 3 = 6 = <b>110</b>, so the answer must be one B2 plus one B1 - the shape is decided before we touch a single pointer.'};
  } else {
    H1 = [T0(4), link(T0(1), T0(6))];           // 3 nodes
    H2 = [T0(2)];                               // 1 node = insert
    story = {zh:'<b>插入</b>其實就是「跟一個只有 1 個節點的堆積合併」：3 + 1 = 4 = <b>100</b>，' +
                '會連續進位兩次，最後只剩一棵 B2。',
             en:'<b>Insertion</b> is just merging with a one-node heap: 3 + 1 = 4 = <b>100</b>, which carries twice and leaves a single B2.'};
  }
  const both = () => forestShapes(H1, 1.15, {}).concat(forestShapes(H2, 4.15, {}));
  F.push({shapes:both().concat([S.t(.3, .75, 'H1', {c:'#c7a6ff', fs:.34, anchor:'start'}),
                                S.t(.3, 3.75, 'H2', {c:'#ffbe6b', fs:.34, anchor:'start'})]),
    panels:[heapPanel(H1, {zh:'H1', en:'H1'}), heapPanel(H2, {zh:'H2', en:'H2'})], line:0,
    msg:story});
  // binary addition
  let out = [], carry = null, d = 0;
  const pool = H1.concat(H2);
  const left = () => pool.filter(t => !t._used);
  while (d <= 4 && (left().length || carry)){
    const bucket = pool.filter(t => !t._used && ORD(t) === d);
    bucket.forEach(t => t._used = true);
    const all = bucket.concat(carry ? [carry] : []);
    const shapes = () => forestShapes(out, 1.15, {}, false)
      .concat(forestShapes(all, 3.7, all.reduce((a, t) => (a[t.v] = 'hot', a), {})));
    F.push({shapes:shapes().concat([S.t(.3, .75, {zh:'已完成', en:'settled'}, {c:'#3fe0dd', fs:.30, anchor:'start'}),
      S.t(.3, 3.3, {zh:'這一位 (B' + d + ') 手上有 ' + all.length + ' 棵',
                    en:'digit B' + d + ': ' + all.length + ' tree(s) in hand'}, {c:'#ffbe6b', fs:.30, anchor:'start'})]),
      panels:[heapPanel(out, {zh:'結果', en:'result'})], line:3,
      msg:{zh:'看第 <b>' + d + '</b> 位（B' + d + '）：手上有 <b>' + all.length + '</b> 棵。' +
              (all.length === 0 ? '這一位是 0，跳過。' : ''),
           en:'Look at digit <b>' + d + '</b> (B' + d + '): <b>' + all.length + '</b> tree(s) in hand.' +
              (all.length === 0 ? ' The digit is 0, skip.' : '')}});
    if (all.length === 0){ carry = null; d++; continue; }
    if (all.length === 1){
      out.push(all[0]); carry = null;
      F.push({shapes:forestShapes(out, 1.15, {[all[0].v]:'ok'}), panels:[heapPanel(out, {zh:'結果', en:'result'})], line:5,
        msg:{zh:'只有一棵 → 這一位是 <b>1</b>，直接留下來，不用做任何事。',
             en:'Exactly one tree, so the digit is <b>1</b>: keep it and do nothing else.'}});
    } else {
      const a = all[0], b = all[1];
      F.push({shapes:forestShapes(out, 1.15, {}, false).concat(
        forestShapes([a, b], 3.7, {[a.v]:'hot', [b.v]:'act'})),
        panels:[heapPanel(out, {zh:'結果', en:'result'})], line:8,
        msg:{zh:'有兩棵同階 → <b>進位</b>！把 <b>' + a.v + '</b> 和 <b>' + b.v +
                '</b> 用 link 接成一棵 B' + (d + 1) + '，交給下一位。' +
                '這一步是 <b>O(1)</b>，只是改幾個指標。',
             en:'Two trees of the same order means <b>carry</b>: link <b>' + a.v + '</b> and <b>' + b.v +
                '</b> into one B' + (d + 1) + ' and hand it to the next digit. That step is <b>O(1)</b>, just a few pointers.'}});
      carry = link(a, b);
      if (all.length === 3) out.push(all[2]);
    }
    d++;
  }
  F.push({shapes:forestShapes(out, 1.6, out.reduce((a, t) => (a[t.v] = 'ok', a), {})),
    panels:[heapPanel(out, {zh:'合併結果', en:'merged heap'}), cntPanel(out)], line:9,
    msg:{zh:'合併完成，形狀是 <b>' + out.map(t => 'B' + ORD(t)).join(' + ') + '</b>，' +
            '共 ' + bin(out) + ' 個節點 = <b>0b' + binStr(bin(out)) + '</b>，跟一開始算的一樣。' +
            '整個過程只走過 <b>log n</b> 位數——這就是二項堆積的賣點：' +
            '<b>合併是 O(log n)</b>，而二元堆積要 O(n)。',
         en:'Done: the shape is <b>' + out.map(t => 'B' + ORD(t)).join(' + ') + '</b>, ' + bin(out) +
            ' nodes = <b>0b' + binStr(bin(out)) + '</b>, exactly as predicted. We only walked <b>log n</b> digits - and that is the whole point: <b>merging is O(log n)</b> here, versus O(n) for a binary heap.'}});
  return F.list;
}

function buildExtract(){
  const F = new Frames();
  let b2 = link(link(T0(3), T0(9)), link(T0(6), T0(11)));
  let heap = [T0(8), b2];
  F.push({shapes:forestShapes(heap, 2.0, {}), panels:[heapPanel(heap), cntPanel(heap)], line:0,
    msg:{zh:'堆積是 <b>B0 + B2</b>，5 個節點（0b101）。最小值一定在<b>某一棵樹的根</b>上，' +
            '所以只要看樹根，最多 log n 個。',
         en:'The heap is <b>B0 + B2</b>, 5 nodes (0b101). The minimum must be the root of one of the trees, so we only inspect roots - at most log n of them.'}});
  const minT = heap.reduce((m, t) => t.v < m.v ? t : m);
  F.push({shapes:forestShapes(heap, 2.0, {[minT.v]:'hot'}), panels:[heapPanel(heap)], line:1,
    msg:{zh:'比過所有樹根，最小的是 <b>' + minT.v + '</b>。',
         en:'Comparing the roots, the smallest is <b>' + minT.v + '</b>.'}});
  const kids = minT.kids.slice();
  heap = heap.filter(t => t !== minT);
  F.push({shapes:forestShapes(heap, 1.15, {}).concat(forestShapes(kids, 3.9,
    kids.reduce((a, t) => (a[t.v] = 'act', a), {}))).concat(
    [S.t(.3, 3.5, {zh:'被釋放的小孩', en:'orphaned children'}, {c:'#c7a6ff', fs:.30, anchor:'start'})]),
    panels:[heapPanel(heap, {zh:'剩下的森林', en:'remaining forest'}),
            heapPanel(kids, {zh:'小孩們', en:'children'})], line:3,
    msg:{zh:'拿掉根之後，它的小孩掉出來變成 <b>' + kids.map(t => 'B' + ORD(t)).join(' + ') + '</b>。' +
            '關鍵在於：<b>這幾棵剛好階數各不相同，本身就是一個合法的二項堆積</b>。',
         en:'Removing the root drops out its children as <b>' + kids.map(t => 'B' + ORD(t)).join(' + ') +
            '</b>. The key point: <b>their orders are all distinct, so they already form a valid binomial heap</b>.'}});
  // merge back
  let all = heap.concat(kids).sort((a, b) => ORD(a) - ORD(b));
  const out = []; let carry = null;
  for (let d = 0; d <= 3; d++){
    const bucket = all.filter(t => ORD(t) === d && !t._u); bucket.forEach(t => t._u = true);
    const grp = bucket.concat(carry ? [carry] : []);
    if (!grp.length){ carry = null; continue; }
    if (grp.length === 1){ out.push(grp[0]); carry = null; }
    else { carry = link(grp[0], grp[1]); if (grp.length === 3) out.push(grp[2]); }
  }
  if (carry) out.push(carry);
  F.push({shapes:forestShapes(out, 1.9, {}), panels:[heapPanel(out), cntPanel(out)], line:4,
    msg:{zh:'把小孩們跟剩下的森林<b>合併</b>——又是一次二進位加法。結果 <b>' +
            out.map(t => 'B' + ORD(t)).join(' + ') + '</b>，4 個節點（0b100）。',
         en:'Now <b>merge</b> the children back into the remaining forest - another binary addition. Result: <b>' +
            out.map(t => 'B' + ORD(t)).join(' + ') + '</b>, 4 nodes (0b100).'}});
  F.push({shapes:forestShapes(out, 1.9, out.reduce((a, t) => (a[t.v] = 'ok', a), {})),
    panels:[heapPanel(out), cntPanel(out)], line:5,
    msg:{zh:'取出 <b>' + minT.v + '</b> 完成，整趟只碰到樹根與一層小孩，<b>O(log n)</b>。' +
            '二項堆積用「森林 + 進位」換來便宜的合併，同樣的想法再往前推一步就是斐波那契堆積。',
         en:'Extracting <b>' + minT.v + '</b> touched only roots and one layer of children: <b>O(log n)</b>. A binomial heap trades a forest plus carries for cheap merging; push the same idea one step further and you get a Fibonacci heap.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'二項堆積', en:'Binomial Heap'},
  sub:{zh:'一堆二項樹排成一個二進位數：合併兩個堆積就是做一次加法，進位就是把兩棵樹連起來。',
       en:'A forest of binomial trees behaving like a binary number: merging two heaps is an addition, and a carry is a link.'},
  tabs:[
    {id:'link', label:{zh:'二項樹是什麼', en:'What is a binomial tree'},
     stage:{zh:'兩棵同階的樹接成一棵', en:'Two equal orders make the next one'}, view:[10, 6.4],
     idea:{zh:'<b>Bk 一定有 2^k 個節點</b>，而且只能由兩棵 Bk₋₁ 接成。' +
              '因為節點數是 2 的次方，一個「有 n 個節點的堆積」就只能拆成 n 的二進位那幾棵樹——' +
              '形狀完全由 n 決定。',
           en:'<b>Bk always holds 2^k nodes</b> and can only be formed from two copies of Bk-1. Because the sizes are powers of two, a heap of n nodes decomposes into exactly the trees in the binary expansion of n - the shape is fully determined by n.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_LINK, build:buildLink},
    {id:'merge', label:{zh:'合併 = 二進位加法', en:'merge = binary addition'},
     stage:{zh:'一位一位加，兩棵同階就進位', en:'Digit by digit; two equal orders carry'}, view:[10, 6.4],
     variants:[{zh:'合併 3 + 3', en:'merge 3 + 3'}, {zh:'插入 3 + 1', en:'insert 3 + 1'}],
     idea:{zh:'把堆積看成二進位數之後，合併就是加法、link 就是進位。' +
              '因為只有 <b>log n</b> 位，合併是 <b>O(log n)</b>；二元堆積要合併只能整個重建 O(n)。' +
              '插入不過是加 1 的特例。',
           en:'Once a heap is a binary number, merging is addition and linking is the carry. With only <b>log n</b> digits, merging is <b>O(log n)</b>, whereas merging two binary heaps means rebuilding in O(n). Insertion is simply adding one.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_MERGE, build:buildMerge},
    {id:'ext', label:{zh:'取出最小', en:'extract_min'},
     stage:{zh:'拔掉一個樹根，小孩自成一堆', en:'Pull one root; the children are a heap already'}, view:[10, 6.4],
     idea:{zh:'最小值只可能在樹根，所以先掃 <b>log n</b> 個根。拔掉之後，' +
              '它的小孩剛好是階數互不相同的一串樹——<b>本身就是合法的堆積</b>，' +
              '直接跟剩下的合併就好。',
           en:'The minimum can only be a root, so we scan <b>log n</b> roots. Once it is removed, its children happen to have pairwise distinct orders - <b>a valid heap on their own</b> - so we just merge them back in.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_EXT, build:buildExtract}
  ]
};
