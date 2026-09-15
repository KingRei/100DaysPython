// DAY: 31
// TITLE_ZH: Skip List 跳躍串列：用擲硬幣取代平衡的排序結構
// TITLE_EN: Skip list - an ordered map built out of coin flips
// SUB_ZH: 排序好的 linked list 沒辦法二分搜尋，因為它跳不到中間。Skip list 的辦法是在上面加幾條快速道路，而每個節點要蓋幾層，是出生時擲硬幣決定的，之後再也不動 —— 這就是 Redis sorted set。
// SUB_EN: A sorted linked list cannot be binary searched, because it cannot jump to the middle. A skip list bolts express lanes on top, and the height of each node is decided once, by a coin, when it is born - never rebalanced. This is the Redis sorted set.
// FOLDER: day%2031%20-%20skip%20list
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const ln = (code, frag) => { const i = code.findIndex(l => l.indexOf(frag) >= 0); return i < 0 ? 0 : i; };
function chip(t, cls){ return {t:t, cls:cls || ''}; }

/* ---- a real skip list, with the coin replaced by a fixed height table ---- */
function SkipList(maxL){
  this.maxL = maxL || 5;
  this.head = {key:null, fw:new Array(this.maxL).fill(null),
               span:new Array(this.maxL).fill(0)};
  this.level = 1;
  this.length = 0;
}
SkipList.prototype.descend = function(key){
  const update = new Array(this.maxL).fill(this.head);
  const rank = new Array(this.maxL).fill(0);
  let node = this.head, steps = 0;
  const trail = [];
  for (let i = this.level - 1; i >= 0; i--){
    rank[i] = (i === this.level - 1) ? 0 : rank[i + 1];
    trail.push({lv:i, key:node.key, drop:false});
    while (node.fw[i] && node.fw[i].key < key){
      rank[i] += node.span[i];
      node = node.fw[i];
      steps++;
      trail.push({lv:i, key:node.key, drop:false});
    }
    update[i] = node;
  }
  return {update:update, rank:rank, cand:node.fw[0], steps:steps, trail:trail};
};
SkipList.prototype.insert = function(key, h, skipBump){
  const d = this.descend(key), update = d.update, rank = d.rank;
  if (h > this.level){
    for (let i = this.level; i < h; i++){ update[i] = this.head; rank[i] = 0;
                                          this.head.span[i] = this.length; }
    this.level = h;
  }
  const node = {key:key, fw:new Array(h).fill(null), span:new Array(h).fill(0)};
  for (let i = 0; i < h; i++){
    node.fw[i] = update[i].fw[i];
    update[i].fw[i] = node;
    node.span[i] = update[i].span[i] - (rank[0] - rank[i]);
    update[i].span[i] = (rank[0] - rank[i]) + 1;
  }
  if (!skipBump)
    for (let i = h; i < this.level; i++) update[i].span[i] += 1;
  this.length++;
  return node;
};
SkipList.prototype.keys = function(){
  const out = []; let n = this.head.fw[0];
  while (n){ out.push(n.key); n = n.fw[0]; }
  return out;
};
SkipList.prototype.rows = function(){
  const out = [];
  for (let i = 0; i < this.level; i++){
    const row = []; let n = this.head.fw[i];
    while (n){ row.push(n.key); n = n.fw[i]; }
    out.push(row);
  }
  return out;
};
SkipList.prototype.spanOf = function(lv, key){   // span of the hop arriving at key
  let n = this.head;
  while (n.fw[lv]){
    if (n.fw[lv].key === key) return n.span[lv];
    n = n.fw[lv];
  }
  return null;
};
SkipList.prototype.rank = function(key){
  let n = this.head, r = 0;
  for (let i = this.level - 1; i >= 0; i--)
    while (n.fw[i] && n.fw[i].key <= key){ r += n.span[i]; n = n.fw[i]; }
  return (n !== this.head && n.key === key) ? r - 1 : -1;
};

/* the demo list: heights that a coin could plausibly have produced */
const HEIGHTS = {1:1, 2:2, 3:1, 4:3, 5:1, 6:1, 7:2, 9:4, 10:1, 11:2, 12:1};
function baseList(skipBump){
  const sl = new SkipList(5);
  [1,2,3,4,5,6,7,9,10,11,12].forEach(k => sl.insert(k, HEIGHTS[k], skipBump));
  return sl;
}

/* ---- drawing ------------------------------------------------------------ */
const XK = k => 1.15 + (k - 1) * 0.70;
const YL = lv => 0.95 + lv * 0.92;
const CW = 0.60, CH = 0.52;

function lanes(sl, opt){
  opt = opt || {};
  const hot = opt.hot || {};          // 'lv:key' -> style
  const hotE = opt.hotE || {};        // 'lv:from>to' -> style
  const spans = opt.spans;            // true = label every hop with its span
  const out = [];
  out.push(S.r(0.30, YL(0) - 0.02, 0.42, (sl.level - 1) * 0.92 + CH + 0.04,
               'soft', 'H', {fs:.28}));
  for (let i = 0; i < sl.level; i++){
    const y = YL(i);
    out.push(S.t(9.45, y + CH / 2, 'L' + i, {c:COL.grey, fs:.30}));
    let n = sl.head;
    while (n.fw[i]){
      const nx = n.fw[i];
      const from = (n === sl.head) ? 0.72 : XK(n.key) + CW;
      const ek = i + ':' + (n === sl.head ? 'H' : n.key) + '>' + nx.key;
      const est = hotE[ek];
      out.push(S.e(from, y + CH / 2, XK(nx.key), y + CH / 2,
                   {w:est ? .075 : .045, s:est || 'idle'}));
      if (spans && i > 0)
        out.push(S.t((from + XK(nx.key)) / 2, y + CH + .30, String(n.span[i]),
                     {c:est ? COL.orangeL : COL.grey, fs:.28}));
      out.push(S.r(XK(nx.key), y, CW, CH, hot[i + ':' + nx.key] || 'idle',
                   String(nx.key), {fs:.34}));
      n = nx;
    }
  }
  return out;
}

/* ================================================== 1. search: the staircase */
const CODE_SEARCH = [
  'def search(self, key):',
  '    node = self.head',
  '    for i in range(self.level - 1, -1, -1):   # top lane first',
  '        while node.forward[i] is not None \\',
  '              and node.forward[i].key < key:  # still short of the key?',
  '            node = node.forward[i]            # move right',
  '        # next key is >= target: drop a level',
  '    cand = node.forward[0]                    # level 0 is the whole list',
  '    return cand.val if cand and cand.key == key else None',
];

function searchFrames(v){
  const flat = (v === 1);
  const sl = new SkipList(5);
  [1,2,3,4,5,6,7,9,10,11,12].forEach(k => sl.insert(k, flat ? 1 : HEIGHTS[k]));
  const F = new Frames(), target = 11;
  const hot = {}, hotE = {};
  let node = sl.head, hops = 0;

  const panels = (lv, msgKey) => ([
    {lbl:{zh:'目前層', en:'level'}, chips:[chip(lv < 0 ? '-' : 'L' + lv, 'hot')]},
    {lbl:{zh:'站在哪個 key', en:'standing on'},
     chips:[chip(node === sl.head ? 'head' : String(node.key), 'act')]},
    {lbl:{zh:'往右走了幾步', en:'forward hops'}, chips:[chip(String(hops), 'ok')]},
    {lbl:{zh:'level 0 的全長', en:'keys on level 0'},
     chips:[chip(String(sl.keys().length), '')]}
  ]);

  F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), panels:panels(sl.level - 1),
          view:VIEW, line:1,
          msg:flat
            ? {zh:'只有 level 0：這就是一條排序好的 linked list。它知道順序，但跳不到中間，所以只能一步一步走。',
               en:'Level 0 only - a plain sorted linked list. It knows the order but cannot jump to the middle, so the only move available is one step right.'}
            : {zh:'要找 11。從最上面那條快速道路出發：層數越高節點越少，一步就能跨過一大段。',
               en:'Looking for 11. Start on the topmost express lane: the higher the level, the fewer the nodes, so one hop covers a lot of ground.'}});

  for (let i = sl.level - 1; i >= 0; i--){
    F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), panels:panels(i), view:VIEW,
            line:2,
            msg:{zh:'切到 L' + i + '。注意：我們不是重頭開始，是從剛才停下來的那個節點繼續 —— 上面幾層已經幫我們排除掉左邊那一段了。',
                 en:'Drop to L' + i + '. Note we do not restart: we continue from the node we stopped on, because the levels above have already ruled out everything to its left.'}});
    while (node.fw[i] && node.fw[i].key < target){
      const from = node === sl.head ? 'H' : node.key;
      hotE[i + ':' + from + '>' + node.fw[i].key] = 'hot';
      node = node.fw[i];
      hot[i + ':' + node.key] = 'hot';
      hops++;
      F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), panels:panels(i),
              view:VIEW, line:5,
              msg:{zh:node.key + ' 還小於 11，所以往右走是安全的：走過去不會錯過答案。',
                   en:node.key + ' is still below 11, so moving right is safe - we cannot step over the answer.'}});
    }
    if (node.fw[i])
      F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), panels:panels(i),
              view:VIEW, line:6,
              msg:{zh:'下一個是 ' + node.fw[i].key + '，已經不小於 11 了。往右會走過頭，所以改往下 —— 這一層的工作就結束了。',
                   en:'The next key on this level is ' + node.fw[i].key + ', which is no longer below 11. Going right would overshoot, so we go down instead; this level is done.'}});
  }
  const cand = node.fw[0];
  if (cand) hot['0:' + cand.key] = 'ok';
  F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), panels:panels(0), view:VIEW,
          line:7,
          msg:flat
            ? {zh:'找到 11，總共走了 ' + hops + ' 步。沒有快速道路，就只能把 1 到 10 全部走過一遍。',
               en:'Found 11 after ' + hops + ' hops. With no express lanes there is no way to avoid walking over every one of 1..10.'}
            : {zh:'找到 11，只走了 ' + hops + ' 步。快速道路把 1 到 9 那一整段直接跨過去了，而 level 0 仍然是完整的串列，所以答案不會漏掉。',
               en:'Found 11 in just ' + hops + ' hops. The express lanes flew over the whole run 1..9, while level 0 is still the complete list, so nothing can be missed.'}});
  return F.list;
}

/* ============================================ 2. insert: update[] and splice */
const CODE_INSERT = [
  'def insert(self, key, val):',
  '    update, rank, cand = self._descend(key)   # same walk as search',
  '    lvl = self.random_level()                 # flip coins: 1 + geometric',
  '    node = SkipNode(key, val, lvl)',
  '    for i in range(lvl):                      # splice in, level by level',
  '        node.forward[i] = update[i].forward[i]',
  '        update[i].forward[i] = node',
  '        node.span[i] = update[i].span[i] - (rank[0] - rank[i])',
  '        update[i].span[i] = (rank[0] - rank[i]) + 1',
  '    for i in range(lvl, self.level):          # levels the node cannot reach',
  '        update[i].span[i] += 1                # still jump over it now',
  '    self.length += 1',
];

function updChips(upd, level, mark){
  const out = [];
  for (let i = level - 1; i >= 0; i--)
    out.push(chip('L' + i + ':' + (upd[i] === null ? 'head'
                  : (upd[i].key === null ? 'head' : upd[i].key)),
                  mark === i ? 'hot' : 'act'));
  return out;
}

function insertFrames(){
  const sl = baseList(false);
  const F = new Frames(), key = 8, h = 3;
  const d = sl.descend(key);
  const hot = {}, hotE = {};

  F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), view:VIEW, line:1,
          panels:[{lbl:{zh:'要插入', en:'inserting'}, chips:[chip('8', 'hot')]},
                  {lbl:{zh:'update[]', en:'update[]'}, chips:[chip('-', '')]}],
          msg:{zh:'要插入 8。第一件事跟 search 一模一樣：走同一條樓梯。差別只在於這次要把「每一層最後停在哪個節點」記下來。',
               en:'Inserting 8. The first move is identical to a search: walk the same staircase. The only difference is that this time we write down where we stopped on each level.'}});

  // replay the descent, collecting update[]
  const upd = new Array(sl.level).fill(null);
  let node = sl.head;
  for (let i = sl.level - 1; i >= 0; i--){
    while (node.fw[i] && node.fw[i].key < key){
      const from = node === sl.head ? 'H' : node.key;
      hotE[i + ':' + from + '>' + node.fw[i].key] = 'hot';
      node = node.fw[i];
      hot[i + ':' + node.key] = 'act';
    }
    upd[i] = node;
    F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), view:VIEW, line:1,
            panels:[{lbl:{zh:'要插入', en:'inserting'}, chips:[chip('8', 'hot')]},
                    {lbl:{zh:'update[]', en:'update[]'},
                     chips:updChips(upd, sl.level, i)}],
            msg:{zh:'L' + i + ' 停在 ' + (node === sl.head ? 'head' : node.key) +
                   '，記進 update[' + i + ']。這一格的 forward 指標等一下可能要改寫，而且只有這一格 —— 其他節點完全不會被動到。',
                 en:'On L' + i + ' we stop at ' + (node === sl.head ? 'the head' : node.key) +
                    ', so that goes into update[' + i + ']. Its forward pointer is the one that may have to be rewritten - and it is the only one on this level that will be touched.'}});
  }

  F.push({shapes:lanes(sl, {hot:hot, hotE:hotE}), view:VIEW, line:2,
          panels:[{lbl:{zh:'擲硬幣', en:'coin flips'},
                   chips:[chip('H', 'ok'), chip('H', 'ok'), chip('T', 'bad')]},
                  {lbl:{zh:'得到高度', en:'height'}, chips:[chip('3', 'hot')]}],
          msg:{zh:'高度不是算出來的，是擲出來的：連續兩次正面才停，所以 8 拿到高度 3。沒有任何規則檢查它「應該」多高 —— 這就是 skip list 不需要旋轉的原因。',
               en:'The height is not computed, it is flipped: two heads then a tail, so 8 gets height 3. Nothing checks whether that height is "correct" - which is exactly why a skip list never needs a rotation.'}});

  // splice
  const sl2 = baseList(false);
  const d2 = sl2.descend(key);
  const upd2 = d2.update, rk = d2.rank;
  const node2 = {key:key, fw:new Array(h).fill(null), span:new Array(h).fill(0)};
  for (let i = 0; i < h; i++){
    node2.fw[i] = upd2[i].fw[i];
    upd2[i].fw[i] = node2;
    node2.span[i] = upd2[i].span[i] - (rk[0] - rk[i]);
    upd2[i].span[i] = (rk[0] - rk[i]) + 1;
    sl2.length = sl2.length;
    const hh = {}; hh[i + ':8'] = 'hot';
    for (let j = 0; j < i; j++) hh[j + ':8'] = 'ok';
    F.push({shapes:lanes(sl2, {hot:hh, spans:true}), view:VIEW, line:6,
            panels:[{lbl:{zh:'已接上的層', en:'levels linked'},
                     chips:[chip('L0..L' + i, 'ok')]},
                    {lbl:{zh:'新的 span', en:'new spans'},
                     chips:[chip('update -> 8 : ' + upd2[i].span[i], 'hot'),
                            chip('8 -> next : ' + node2.span[i], 'act')]}],
            msg:i === 0
              ? {zh:'L0 接上去，順序就已經正確了。後面所有的層都只是加速用的 —— 它們不影響答案，只影響走幾步。',
                 en:'Once L0 is linked the order is already correct. Every level above is pure acceleration: it changes how many hops a search takes, never what it finds.'}
              : {zh:'L' + i + ' 上原本一條指標跨過了 8，現在被切成兩段。兩段的 span 加起來要等於原本那一條加一 —— 這是 span 唯一的規則。',
                 en:'On L' + i + ' a single pointer used to fly over 8; it is now cut in two. The two new spans must add up to the old one plus one - that is the entire rule spans obey.'}});
  }

  // the bump above
  const hhh = {}; for (let j = 0; j < h; j++) hhh[j + ':8'] = 'ok';
  const bumpE = {};
  for (let i = h; i < sl2.level; i++){
    upd2[i].span[i] += 1;
    const from = upd2[i] === sl2.head ? 'H' : upd2[i].key;
    if (upd2[i].fw[i]) bumpE[i + ':' + from + '>' + upd2[i].fw[i].key] = 'hot';
  }
  sl2.length++;
  F.push({shapes:lanes(sl2, {hot:hhh, hotE:bumpE, spans:true}), view:VIEW,
          line:10,
          panels:[{lbl:{zh:'8 蓋到哪一層', en:'8 reaches up to'},
                   chips:[chip('L2', 'ok')]},
                  {lbl:{zh:'還要 +1 的層', en:'levels that still need +1'},
                   chips:[chip('L3', 'hot')]}],
          msg:{zh:'8 只蓋到 L2，但 L3 那條指標照樣從它頭上飛過去，所以它的 span 也要 +1。這一行跟指標無關、跟順序無關，只跟「數過幾個」有關 —— 也是最容易忘記的一行。',
               en:'8 only reaches L2, but the L3 pointer still flies over it, so that span has to grow by one too. This line touches no pointer and changes no order - it only fixes a count, which is why it is the line people forget.'}});

  const okAll = {}; for (let j = 0; j < h; j++) okAll[j + ':8'] = 'ok';
  F.push({shapes:lanes(sl2, {hot:okAll, spans:true}), view:VIEW, line:11,
          panels:[{lbl:{zh:'level 0', en:'level 0'},
                   chips:[chip(sl2.keys().join(' '), 'ok')]},
                  {lbl:{zh:'rank(8)', en:'rank(8)'},
                   chips:[chip(String(sl2.rank(8)), 'ok')]}],
          msg:{zh:'插入完成，總共只改寫了 3 個 forward 指標和 1 個 span。沒有旋轉、沒有重建、沒有任何節點搬家 —— 這是 skip list 相對於平衡樹最實際的好處。',
               en:'Done: three forward pointers rewritten and one span nudged. No rotation, no rebuild, nothing moved - which is the practical advantage a skip list has over a balanced tree.'}});
  return F.list;
}

/* ================================================== 3. spans become ZRANK */
const CODE_RANK = [
  'def rank(self, key):                          # Redis ZRANK',
  '    node, r = self.head, 0',
  '    for i in range(self.level - 1, -1, -1):',
  '        while node.forward[i] is not None \\',
  '              and node.forward[i].key <= key:  # note <=, not <',
  '            r += node.span[i]                 # count what we flew over',
  '            node = node.forward[i]',
  '    return r - 1 if node.key == key else -1   # 0-based',
];

function rankFrames(){
  const sl = baseList(false);
  const F = new Frames(), target = 11;
  const hot = {}, hotE = {};
  let node = sl.head, r = 0;
  const parts = [];

  const panels = () => ([
    {lbl:{zh:'累積的 span', en:'spans added so far'},
     chips:parts.length ? parts.map(p => chip(String(p), 'act')) : [chip('0', '')]},
    {lbl:{zh:'合計', en:'total'}, chips:[chip(String(r), 'hot')]},
    {lbl:{zh:'站在哪', en:'standing on'},
     chips:[chip(node === sl.head ? 'head' : String(node.key), 'ok')]}
  ]);

  F.push({shapes:lanes(sl, {hot:hot, hotE:hotE, spans:true}), panels:panels(),
          view:VIEW, line:0,
          msg:{zh:'每條指標上面那個數字是 span：它一口氣跨過了幾個 level 0 的節點。走訪的路徑不變，只是順手把這些數字加起來。',
               en:'The number on each pointer is its span: how many level-0 nodes that one hop flies over. The walk is exactly the search walk; we just add those numbers up on the way.'}});

  for (let i = sl.level - 1; i >= 0; i--){
    while (node.fw[i] && node.fw[i].key <= target){
      const from = node === sl.head ? 'H' : node.key;
      hotE[i + ':' + from + '>' + node.fw[i].key] = 'hot';
      r += node.span[i];
      parts.push(node.span[i]);
      node = node.fw[i];
      hot[i + ':' + node.key] = 'act';
      F.push({shapes:lanes(sl, {hot:hot, hotE:hotE, spans:true}),
              panels:panels(), view:VIEW, line:5,
              msg:{zh:'這一跳跨過 ' + parts[parts.length - 1] + ' 個節點，累計 ' + r +
                     '。重點是我們從來沒有真的走過那些節點 —— 數字是插入的時候就記好的。',
                   en:'This hop covers ' + parts[parts.length - 1] + ' nodes, running total ' + r +
                      '. The point is that we never actually visited them: the number was written down at insert time.'}});
    }
  }
  const hot2 = Object.assign({}, hot); hot2['0:' + target] = 'ok';
  F.push({shapes:lanes(sl, {hot:hot2, hotE:hotE, spans:true}), panels:panels(),
          view:VIEW, line:7,
          msg:{zh:parts.join(' + ') + ' = ' + r + '，減一得到 0-based 的 rank = ' + sl.rank(target) +
                 '。同一趟走訪同時回答了「在不在」和「排第幾」；反過來用就是 ZRANGE 依索引取值。',
               en:parts.join(' + ') + ' = ' + r + ', minus one gives the 0-based rank ' + sl.rank(target) +
                  '. One walk answers both "is it there" and "where is it"; run the same sums the other way and you get ZRANGE by index.'}});
  return F.list;
}

/* ============================================ 4. the span you forgot to bump */
const CODE_BUG = [
  '    for i in range(lvl):                      # the levels the node reaches',
  '        node.forward[i] = update[i].forward[i]',
  '        update[i].forward[i] = node',
  '        node.span[i] = update[i].span[i] - (rank[0] - rank[i])',
  '        update[i].span[i] = (rank[0] - rank[i]) + 1',
  '',
  '    # for i in range(lvl, self.level):        # <-- the missing loop',
  '    #     update[i].span[i] += 1',
  '',
  '    # keys(), search(), range() : still perfect',
  '    # rank()                    : silently wrong',
];

function bugFrames(v){
  const buggy = (v === 1);
  const sl = baseList(false);
  sl.insert(8, 3, buggy);
  const F = new Frames();
  const good = baseList(false); good.insert(8, 3, false);

  const bad = {};
  good.keys().forEach((k, idx) => { if (sl.rank(k) !== idx) bad[k] = idx; });
  const badKeys = Object.keys(bad).map(Number);

  const hotE = {};
  if (buggy){
    let n = sl.head;
    while (n.fw[3] && n.fw[3].key < 8) n = n.fw[3];
    if (n.fw[3]) hotE['3:' + (n === sl.head ? 'H' : n.key) + '>' + n.fw[3].key] = 'bad';
  }

  F.push({shapes:lanes(sl, {spans:true, hotE:hotE}), view:VIEW, line:buggy ? 6 : 0,
          panels:[{lbl:{zh:'level 0 的順序', en:'level 0 order'},
                   chips:[chip(sl.keys().join(' '), 'ok')]}],
          msg:buggy
            ? {zh:'把「幫上面幾層的 span 加一」那個迴圈註解掉，再插入一次 8。第一件要注意的事：順序完全正確。',
               en:'Comment out the loop that bumps the spans of the levels above, then insert 8 again. First thing to notice: the order is still exactly right.'}
            : {zh:'正確版本：8 蓋不到的 L3，span 也被 +1 了。接下來每個 key 的 rank 都會對得上。',
               en:'The correct version: L3, which 8 does not reach, had its span bumped anyway. Every rank below will line up.'}});

  F.push({shapes:lanes(sl, {spans:true, hotE:hotE}), view:VIEW, line:9,
          panels:[{lbl:{zh:'keys()', en:'keys()'}, chips:[chip('OK', 'ok')]},
                  {lbl:{zh:'search()', en:'search()'}, chips:[chip('OK', 'ok')]},
                  {lbl:{zh:'range()', en:'range()'}, chips:[chip('OK', 'ok')]}],
          msg:{zh:'走訪、查找、範圍查詢全部只看 forward 指標，而 forward 指標一根都沒錯。你為 ordered set 寫的測試，一條都不會紅。',
               en:'Iteration, lookup and range queries only ever follow forward pointers, and not one forward pointer is wrong. Every test you would write for an ordered set still passes.'}});

  const marks = {};
  badKeys.forEach(k => { marks['0:' + k] = 'bad'; });
  F.push({shapes:lanes(sl, {spans:true, hot:marks, hotE:hotE}), view:VIEW,
          line:10,
          panels:[{lbl:{zh:'rank 錯掉的 key', en:'keys with a wrong rank'},
                   chips:badKeys.length ? badKeys.map(k => chip(String(k), 'bad'))
                                        : [chip('none', 'ok')]},
                  {lbl:{zh:'例：rank(12)', en:'e.g. rank(12)'},
                   chips:[chip(String(sl.rank(12)), badKeys.length ? 'bad' : 'ok'),
                          chip('should be ' + good.rank(12), 'act')]}],
          msg:buggy
            ? {zh:'只有 rank 會抱怨，而且只在「有一條高層指標剛好跨過新節點」的時候才錯。ZRANGEBYSCORE 一切正常，ZRANK 卻在說謊 —— 這種 bug 可以在 production 活很久。',
               en:'Only rank complains, and only when a tall pointer happens to straddle the new node. ZRANGEBYSCORE keeps working while ZRANK quietly lies - the kind of bug that survives in production for a long time.'}
            : {zh:'正確版本裡沒有任何 rank 出錯。差別只有那兩行 —— 它們不接指標、不改順序，只修正「數過幾個」。',
               en:'In the correct version no rank is wrong. The difference is those two lines, which link nothing and reorder nothing: they only repair a count.'}});
  return F.list;
}

/* ================================================== 5. LeetCode 1206 */
const CODE_LC = [
  'def _update(self, target):                    # LC 1206, duplicates allowed',
  '    upd, node = [self.head] * self.MAXL, self.head',
  '    for i in range(self.level - 1, -1, -1):',
  '        while node.fw[i] and node.fw[i].val < target:   # strictly <',
  '            node = node.fw[i]',
  '        upd[i] = node',
  '    return upd, node.fw[0]        # the FIRST copy of target, if any',
  '',
  'def erase(self, num):',
  '    upd, cand = self._update(num)',
  '    if cand is None or cand.val != num: return False',
  '    for i in range(self.level):   # unlink one copy only',
  '        if upd[i].fw[i] is cand: upd[i].fw[i] = cand.fw[i]',
  '    return True',
];

const LC_VALS = [1, 2, 3, 3, 3, 5];
const LC_H    = [1, 2, 1, 2, 1, 2];

function lcShapes(mark, gone, lv){
  const out = [], y0 = 1.5, y1 = 2.6;
  out.push(S.r(0.35, y0, 0.45, 0.62, 'soft', 'H', {fs:.28}));
  out.push(S.t(9.35, y0 + .31, 'L0', {c:COL.grey, fs:.30}));
  out.push(S.t(9.35, y1 + .31, 'L1', {c:COL.grey, fs:.30}));
  let prev0 = 0.80, prev1 = 0.80;
  LC_VALS.forEach((v, i) => {
    if (gone === i) return;
    const x = 1.35 + i * 1.25;
    out.push(S.e(prev0, y0 + .31, x, y0 + .31, {w:.045}));
    out.push(S.r(x, y0, 0.95, 0.62, mark[i] || 'idle', String(v), {fs:.38}));
    out.push(S.t(x + .48, y0 - .28, '#' + i, {c:COL.grey, fs:.26}));
    prev0 = x + 0.95;
    if (LC_H[i] > 1){
      out.push(S.e(prev1, y1 + .31, x, y1 + .31, {w:.045}));
      out.push(S.r(x, y1, 0.95, 0.62, mark[i] || 'idle', String(v), {fs:.38}));
      prev1 = x + 0.95;
    }
  });
  if (lv != null)
    out.push(S.t(5.0, 3.85, lv, {c:COL.orangeL, fs:.34}));
  return out;
}

function lcFrames(v){
  const strict = (v === 0), target = 3;
  const F = new Frames();
  const cmp = (a) => strict ? (a < target) : (a <= target);
  const mark = {};
  let cur = -1;                    // -1 = head

  const panels = (landed) => ([
    {lbl:{zh:'比較方式', en:'comparison'},
     chips:[chip(strict ? 'fw.val < target' : 'fw.val <= target',
                 strict ? 'ok' : 'bad')]},
    {lbl:{zh:'停在', en:'stopped at'},
     chips:[chip(cur < 0 ? 'head' : '#' + cur + ' (' + LC_VALS[cur] + ')', 'act')]},
    {lbl:{zh:'候選 cand', en:'candidate'},
     chips:[chip(landed == null ? '-' : (landed >= LC_VALS.length ? 'None'
                 : '#' + landed + ' (' + LC_VALS[landed] + ')'),
                 landed != null && LC_VALS[landed] === target ? 'ok' : 'bad')]}
  ]);

  F.push({shapes:lcShapes(mark, null, null), panels:panels(null), view:VIEW,
          line:0,
          msg:{zh:'LC 1206 允許重複值。這裡有三個 3，erase(3) 只能刪掉其中一個 —— 所以「走到哪裡停下來」變成正確性問題，不只是效率問題。',
               en:'LC 1206 allows duplicates. There are three 3s here and erase(3) must remove exactly one, so where the walk stops becomes a correctness question, not just a speed one.'}});

  for (let i = 1; i >= 0; i--){
    let idx = cur;
    const nextOn = (from) => {
      for (let j = from + 1; j < LC_VALS.length; j++)
        if (i === 0 || LC_H[j] > 1) return j;
      return LC_VALS.length;
    };
    F.push({shapes:lcShapes(mark, null, 'L' + i), panels:panels(null),
            view:VIEW, line:2,
            msg:{zh:'切到 L' + i + '，從剛才停下來的地方繼續。',
                 en:'Drop to L' + i + ' and carry on from where we stopped.'}});
    let nx = nextOn(idx);
    while (nx < LC_VALS.length && cmp(LC_VALS[nx])){
      idx = nx; cur = idx;
      mark[idx] = strict ? 'act' : 'bad';
      F.push({shapes:lcShapes(mark, null, 'L' + i), panels:panels(null),
              view:VIEW, line:3,
              msg:strict
                ? {zh:'#' + idx + ' 的值是 ' + LC_VALS[idx] + '，小於 3，可以往右。第一個 3 一定會被留在前面。',
                   en:'#' + idx + ' holds ' + LC_VALS[idx] + ', which is below 3, so moving right is safe. The first 3 will be left in front of us.'}
                : {zh:'用 <= 的話，遇到 3 也會繼續往右走 —— 我們正在一個一個走過所有的 3。',
                   en:'With <=, a 3 is also a reason to keep going, so we are walking past the copies of 3 one by one.'}});
      nx = nextOn(idx);
    }
  }
  const landed = (() => { for (let j = cur + 1; j < LC_VALS.length; j++) return j;
                          return LC_VALS.length; })();
  const hit = landed < LC_VALS.length && LC_VALS[landed] === target;
  if (landed < LC_VALS.length) mark[landed] = hit ? 'ok' : 'bad';
  F.push({shapes:lcShapes(mark, null, null), panels:panels(landed), view:VIEW,
          line:6,
          msg:strict
            ? {zh:'停在第一個 3 的前面，cand 就是第一個 3。erase 只解開這一個節點，其他兩個 3 原封不動。',
               en:'We stop just before the first 3, so cand is the first 3. erase unlinks that single node and leaves the other two copies alone.'}
            : {zh:'用 <= 會一路走過全部的 3，停在 5 前面。cand 變成 5，erase(3) 直接回報「不存在」—— 但 3 明明在裡面。',
               en:'With <= the walk sails past every 3 and stops in front of 5. cand is now 5, so erase(3) reports that 3 is not there - while three copies of it sit in the list.'}});

  if (hit){
    const m2 = {}; m2[landed] = 'ok';
    F.push({shapes:lcShapes(m2, landed, null), panels:panels(landed),
            view:VIEW, line:12,
            msg:{zh:'刪掉一個 3，剩下兩個。重複值就是 LC 1206 的全部難度：一個 `<` 寫成 `<=`，結構還是排序好的，答案卻錯了。',
                 en:'One 3 removed, two left. Duplicates are the whole difficulty of LC 1206: turn one `<` into `<=` and the structure stays perfectly sorted while the answer goes wrong.'}});
  } else {
    F.push({shapes:lcShapes(mark, null, null), panels:panels(landed),
            view:VIEW, line:10,
            msg:{zh:'回傳 False。這種錯誤不會 crash、不會破壞順序，只會讓 erase 默默沒做事 —— 跟前一頁那個 span bug 是同一種味道。',
                 en:'It returns False. Nothing crashes and the order is untouched; erase just quietly does nothing - the same flavour of bug as the span on the previous tab.'}});
  }
  return F.list;
}

/* ================================================== DAY_META */
const DAY_META = {
  title: {zh:'Skip List 跳躍串列：用擲硬幣取代平衡的排序結構',
          en:'Skip list - an ordered map built out of coin flips'},
  sub:   {zh:'排序串列 + 快車道；Redis 的 sorted set 就是這樣做的',
          en:'A sorted linked list plus express lanes - exactly how a Redis sorted set is built'},
  tabs: [
    {
      id:'search', label:{zh:'搜尋', en:'search'},
      stage:{zh:'從最高層往右、走不動就往下', en:'go right on the top lane, drop when you cannot'},
      view:VIEW,
      variants:[{zh:'skip list', en:'skip list'}, {zh:'全高度 1（= 排序串列）', en:'all height 1 (= sorted list)'}],
      idea:{zh:'排序串列的問題不是「比較太多次」，而是「只能一格一格走」。跳躍串列不改順序，只是在上層加幾條快車道，讓你先用大步逼近，再下到細的那層收尾。',
            en:'The problem with a sorted linked list is not the number of comparisons, it is that you can only move one node at a time. A skip list keeps the order untouched and adds express lanes above it, so you approach with big strides and only pay for fine steps at the end.'},
      legend:[['#9d6bff', {zh:'目前節點', en:'current node'}], ['#ff9736', {zh:'剛走過的邊', en:'edge just followed'}], ['#3fe0dd', {zh:'找到', en:'found'}]],
      code:CODE_SEARCH, build:searchFrames
    },
    {
      id:'insert', label:{zh:'插入', en:'insert'},
      stage:{zh:'擲硬幣決定高度，再把 update[] 的指標接起來', en:'flip for a height, then rewire the pointers in update[]'},
      view:VIEW,
      idea:{zh:'AVL 和紅黑樹靠旋轉維持平衡；跳躍串列不旋轉，每個節點的高度在誕生時擲一次硬幣就決定，之後永遠不變。平衡是機率給的，不是搬出來的。',
            en:'AVL and red-black trees keep their shape by rotating. A skip list never rotates: a node flips for its height once when it is born and keeps it forever. The balance is handed to you by probability instead of being rearranged into place.'},
      legend:[['#9d6bff', {zh:'update[i]', en:'update[i]'}], ['#ff9736', {zh:'被改寫的指標', en:'pointer being rewritten'}], ['#3fe0dd', {zh:'新節點', en:'new node'}]],
      code:CODE_INSERT, build:insertFrames
    },
    {
      id:'rank', label:{zh:'rank / 排名', en:'rank'},
      stage:{zh:'把走過的 span 加起來就是名次', en:'add up the spans you flew over'},
      view:VIEW,
      idea:{zh:'每個指標記住自己跨過幾個第 0 層節點，這個數字叫 span。同一趟搜尋把走過的 span 加起來，就直接得到名次 —— 這就是 Redis ZRANK 與 ZRANGE 用索引取值的做法。',
            en:'Every pointer remembers how many level-0 nodes it flies over; that number is its span. Sum the spans along the same walk you already do for search and you get a rank for free - this is exactly how Redis answers ZRANK and index-based ZRANGE.'},
      legend:[['#ff9736', {zh:'走過的邊（計入 span）', en:'edge followed (span counted)'}], ['#9d6bff', {zh:'目前節點', en:'current node'}]],
      code:CODE_RANK, build:rankFrames
    },
    {
      id:'bug', label:{zh:'漏掉的那一行', en:'the missing line'},
      stage:{zh:'順序全對、搜尋全對、名次全錯', en:'order fine, search fine, ranks wrong'},
      view:VIEW,
      variants:[{zh:'正確版', en:'correct'}, {zh:'漏掉 span += 1', en:'missing span += 1'}],
      idea:{zh:'插入時，高於新節點的那幾層沒有任何指標需要改寫 —— 但它們跨過的節點多了一個，span 要 +1。漏掉這一行不會 crash、不會亂序、search 也照常，只有 rank 會默默給錯答案。',
            en:'When you insert, the levels above the new node need no pointer changes at all - but each of them now flies over one extra node, so its span must grow by one. Skip that line and nothing crashes, nothing is out of order, search still works, and only rank quietly lies to you.'},
      legend:[['#ff5c5c', {zh:'名次錯誤', en:'rank wrong'}], ['#3fe0dd', {zh:'名次正確', en:'rank correct'}]],
      code:CODE_BUG, build:bugFrames
    },
    {
      id:'lc1206', label:{zh:'LC 1206', en:'LC 1206'},
      stage:{zh:'Design Skiplist：重複值把 < 和 <= 分出勝負', en:'Design Skiplist: duplicates make < and <= different answers'},
      view:VIEW,
      variants:[{zh:'用 <（正確）', en:'using < (correct)'}, {zh:'用 <=（壞掉）', en:'using <= (broken)'}],
      idea:{zh:'LC 1206 允許重複值，erase 只能刪掉一個。搜尋時比較寫成嚴格小於，才會停在第一個相等的元素前面；寫成小於等於就會滑過所有的重複值，找不到本來就在裡面的東西。',
            en:'LC 1206 allows duplicates and erase must remove exactly one copy. Only a strict less-than stops the walk in front of the first equal element; less-than-or-equal slides past every duplicate and fails to find a value that is plainly there.'},
      legend:[['#9d6bff', {zh:'往右走過', en:'moved right past'}], ['#ff5c5c', {zh:'走過頭', en:'overshot'}], ['#3fe0dd', {zh:'候選正確', en:'candidate correct'}]],
      code:CODE_LC, build:lcFrames
    }
  ]
};
