// DAY: 30
// TITLE_ZH: LRU 與 LFU 快取：滿了的時候丟掉誰
// TITLE_EN: LRU and LFU caches - choosing what to throw away
// SUB_ZH: 快取 = 有上限的 map + 一條挑受害者的規則。map 那半是 Day 09 的 hash table，真正決定快或慢的是規則本身：最久沒用的？用最少次的？還是剛剛才用過的那個？
// SUB_EN: A cache is a bounded map plus a rule for picking a victim. The map half is Day 09's hash table; the rule is what actually decides whether the cache is fast: least recently used, least frequently used - or the one you just touched?
// FOLDER: day%2030%20-%20lru%20and%20lfu%20cache
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const ln = (code, frag) => { const i = code.findIndex(l => l.indexOf(frag) >= 0); return i < 0 ? 0 : i; };
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const pct = x => (100 * x).toFixed(1) + '%';

/* ---- shared stage pieces ------------------------------------------------ */
/* the script of calls across the top; `at` is the one being executed */
function opRow(ops, at, y){
  const w = 9.0 / ops.length, x0 = 0.4, out = [];
  ops.forEach((o, i) => {
    out.push(S.r(x0 + i * w, y, w - .12, .62,
                 i === at ? 'hot' : (i < at ? 'done' : 'idle'), o, {fs:.32}));
  });
  return out;
}
/* the doubly linked list, most recent on the left */
function listRow(keys, y, states, opt){
  opt = opt || {};
  const w = 1.25, gap = .28, n = keys.length;
  const x0 = opt.x0 == null ? 1.35 : opt.x0;
  const out = [];
  out.push(S.r(x0 - .95, y, .55, .95, 'soft', 'H', {fs:.3}));
  for (let i = 0; i < n; i++){
    const x = x0 + i * (w + gap);
    out.push(S.r(x, y, w, .95, (states && states[i]) || 'idle', String(keys[i]), {fs:.44}));
    const px = i === 0 ? x0 - .4 : x0 + (i - 1) * (w + gap) + w;
    out.push(S.e(px, y + .34, x, y + .34, {w:.045}));
    out.push(S.e(x, y + .66, px, y + .66, {w:.045}));
  }
  const lastX = x0 + (n - 1) * (w + gap) + w;
  out.push(S.r(Math.max(lastX + gap, x0 + gap), y, .55, .95, 'soft', 'T', {fs:.3}));
  out.push(S.e(n ? lastX : x0 - .4, y + .34, Math.max(lastX + gap, x0 + gap), y + .34, {w:.045}));
  out.push(S.t(x0 + w / 2, y + 1.38, {zh:'最近用過', en:'most recent'}, {c:COL.grey, fs:.28}));
  if (n) out.push(S.t(lastX - w / 2, y - .28, {zh:'下一個被丟掉的', en:'the next victim'},
                      {c:COL.orangeL, fs:.28}));
  return out;
}

/* ===================================================== 1. LRU, step by step */
const CODE_LRU = [
  'class LRUCache:                       # LeetCode 146',
  '    def get(self, key):',
  '        node = self.map.get(key)',
  '        if node is None: return -1            # a miss',
  '        self._unlink(node)                    # a read IS a use',
  '        self._push_front(node)',
  '        return node.val',
  '',
  '    def put(self, key, value):',
  '        if key in self.map:                   # already here',
  '            node = self.map[key]; node.val = value',
  '            self._unlink(node); self._push_front(node); return',
  '        if len(self.map) == self.capacity:',
  '            victim = self.tail.prev           # least recent',
  '            self._unlink(victim)',
  '            del self.map[victim.key]',
  '        node = Node(key, value)',
  '        self.map[key] = node',
  '        self._push_front(node)'
];

const SCRIPT = [['put', 'A'], ['put', 'B'], ['put', 'C'], ['get', 'A'],
                ['put', 'D'], ['get', 'B'], ['get', 'A']];
const CAP = 3;

/* one engine for both the LRU tab and the FIFO-bug tab: `refresh` is the
   single line that separates them */
function runLRU(refresh){
  const list = [], ev = [];
  SCRIPT.forEach((op, i) => {
    const [kind, key] = op;
    const at = list.indexOf(key);
    if (kind === 'get'){
      if (at < 0){
        ev.push({i:i, op:op, phase:'miss', list:list.slice()});
      } else {
        ev.push({i:i, op:op, phase:'found', list:list.slice(), mark:at});
        if (refresh){
          list.splice(at, 1); list.unshift(key);
          ev.push({i:i, op:op, phase:'refresh', list:list.slice(), mark:0});
        } else {
          ev.push({i:i, op:op, phase:'norefresh', list:list.slice(), mark:at});
        }
      }
      return;
    }
    if (at >= 0){
      list.splice(at, 1); list.unshift(key);
      ev.push({i:i, op:op, phase:'refresh', list:list.slice(), mark:0});
      return;
    }
    if (list.length === CAP){
      ev.push({i:i, op:op, phase:'full', list:list.slice(), mark:list.length - 1});
      const dead = list.pop();
      ev.push({i:i, op:op, phase:'evict', list:list.slice(), dead:dead});
    }
    list.unshift(key);
    ev.push({i:i, op:op, phase:'insert', list:list.slice(), mark:0});
  });
  return ev;
}

function lruPanels(e, refresh){
  return [
    {lbl:{zh:'hash map（鍵 → 節點）', en:'hash map (key → node)'},
     chips:e.list.slice().sort().map(k => chip(k))},
    {lbl:{zh:'串列：最近 → 最久', en:'list: recent → stale'},
     chips:e.list.map((k, i) => chip(k, i === 0 ? 'ok' : (i === e.list.length - 1 ? 'hot' : '')))},
    {lbl:{zh:'get 會不會移動節點', en:'does get move the node'},
     chips:[chip(refresh ? 'yes' : 'no', refresh ? 'ok' : 'bad')]}
  ];
}

function lruFrames(varIx){
  const refresh = varIx !== 1;
  const F = new Frames(), ev = runLRU(refresh);
  ev.forEach(e => {
    const st = {};
    if (e.mark != null) st[e.mark] = (e.phase === 'full' ? 'hot'
                                   : e.phase === 'norefresh' ? 'bad'
                                   : e.phase === 'found' ? 'act' : 'ok');
    const sh = opRow(SCRIPT.map(o => o[0] + ' ' + o[1]), e.i, .35)
      .concat(listRow(e.list, 2.55, st));
    if (e.dead != null){
      sh.push(S.r(7.9, 4.55, 1.25, .95, 'bad', e.dead, {fs:.44}));
      sh.push(S.t(8.52, 5.9, {zh:'丟掉', en:'evicted'}, {c:COL.red, fs:.3}));
    }
    let msg, line;
    const k = e.op[1];
    switch (e.phase){
      case 'miss':
        msg = {zh:'get ' + k + '：map 裡沒有，回傳 −1。快取只會變慢，不會變錯。',
               en:'get ' + k + ': not in the map, return -1. A miss costs time, never correctness.'};
        line = ln(CODE_LRU, 'return -1'); break;
      case 'found':
        msg = {zh:'get ' + k + '：map O(1) 找到節點。真正的重點在下一步 —— <b>讀也算用過</b>。',
               en:'get ' + k + ': the map finds the node in O(1). The interesting part is next - <b>a read is a use</b>.'};
        line = ln(CODE_LRU, 'node = self.map.get'); break;
      case 'refresh':
        msg = {zh:'把節點接到最前面。四個指標寫入，沒有搬資料，也沒有任何迴圈。',
               en:'Splice the node to the front: four pointer writes, no data movement, no loop.'};
        line = ln(CODE_LRU, '_push_front(node)'); break;
      case 'norefresh':
        msg = {zh:'這個版本的 get 什麼都不做 —— 節點留在原位，順序記錄的是<b>進來的時間</b>而不是用過的時間。',
               en:'This version of get does nothing - the node stays where it is, so the order now records <b>arrival</b>, not use.'};
        line = ln(CODE_LRU, 'return node.val'); break;
      case 'full':
        msg = {zh:'put ' + k + '：位子滿了。受害者永遠是 tail.prev —— 不用找，它就在那裡。',
               en:'put ' + k + ': the cache is full. The victim is always tail.prev - no search, it is simply there.'};
        line = ln(CODE_LRU, 'victim = self.tail.prev'); break;
      case 'evict':
        msg = {zh:'把 ' + e.dead + ' 從串列和 map 一起刪掉。兩邊都要刪，只刪一邊就會留下幽靈。',
               en:'Unlink ' + e.dead + ' and delete it from the map. Both halves, or you leave a ghost behind.'};
        line = ln(CODE_LRU, 'del self.map'); break;
      default:
        msg = {zh:'新節點放到最前面：剛用過的東西，離被丟掉最遠。',
               en:'The new node goes to the front: what was just used is furthest from eviction.'};
        line = ln(CODE_LRU, 'self._push_front(node)');
    }
    F.push({shapes:sh, panels:lruPanels(e, refresh), view:VIEW, line:line, msg:msg});
  });
  return F.list;
}

/* ============================================== 2. LFU: buckets + min_freq */
const CODE_LFU = [
  'class LFUCache:                       # LeetCode 460',
  '    def __init__(self, capacity):',
  '        self.vals = {}                        # key -> value',
  '        self.freq = {}                        # key -> use count',
  '        self.buckets = defaultdict(OrderedDict)   # count -> keys, LRU first',
  '        self.min_freq = 0',
  '',
  '    def _bump(self, key):',
  '        f = self.freq[key]',
  '        del self.buckets[f][key]',
  '        if not self.buckets[f]:               # the min bucket just emptied',
  '            del self.buckets[f]',
  '            if self.min_freq == f: self.min_freq = f + 1',
  '        self.freq[key] = f + 1',
  '        self.buckets[f + 1][key] = None       # newest at the back',
  '',
  '    def put(self, key, value):',
  '        if len(self.vals) == self.capacity:',
  '            victim, _ = self.buckets[self.min_freq].popitem(last=False)',
  '            del self.vals[victim]; del self.freq[victim]',
  '        self.vals[key] = value; self.freq[key] = 1',
  '        self.buckets[1][key] = None',
  '        self.min_freq = 1                     # the newcomer is the new minimum'
];

const LFU_SCRIPT = [['put', 'A'], ['put', 'B'], ['put', 'C'], ['get', 'C'],
                    ['get', 'B'], ['get', 'A'], ['put', 'D'], ['get', 'C']];
const LFU_CAP = 3;

/* buckets kept as {count: [keys, oldest-use first]} - the OrderedDict port */
function runLFU(lruTie){
  const freq = {}, buckets = {}, ins = {}, ev = [];
  let minf = 0, seq = 0;
  const put = (f, k) => { (buckets[f] = buckets[f] || []).push(k); };
  const snap = () => JSON.parse(JSON.stringify(buckets));
  LFU_SCRIPT.forEach((op, i) => {
    const [kind, k] = op;
    const known = freq[k] != null;
    if (kind === 'get' && !known){
      ev.push({i:i, op:op, phase:'miss', b:snap(), minf:minf, freq:Object.assign({}, freq)});
      return;
    }
    if (known){
      const f = freq[k];
      ev.push({i:i, op:op, phase:'hit', b:snap(), minf:minf, freq:Object.assign({}, freq), mark:[f, k]});
      buckets[f].splice(buckets[f].indexOf(k), 1);
      if (!buckets[f].length){ delete buckets[f]; if (minf === f) minf = f + 1; }
      freq[k] = f + 1; put(f + 1, k);
      ev.push({i:i, op:op, phase:'bump', b:snap(), minf:minf, freq:Object.assign({}, freq),
               mark:[f + 1, k], from:f});
      return;
    }
    if (Object.keys(freq).length === LFU_CAP){
      ev.push({i:i, op:op, phase:'full', b:snap(), minf:minf, freq:Object.assign({}, freq),
               markBucket:minf});
      const row = buckets[minf];
      const dead = lruTie ? row[0]
                          : row.slice().sort((p, q) => ins[p] - ins[q])[0];
      row.splice(row.indexOf(dead), 1);
      if (!row.length) delete buckets[minf];
      delete freq[dead];
      ev.push({i:i, op:op, phase:'evict', b:snap(), minf:minf, freq:Object.assign({}, freq), dead:dead});
    }
    freq[k] = 1; ins[k] = ++seq; put(1, k); minf = 1;
    ev.push({i:i, op:op, phase:'insert', b:snap(), minf:minf, freq:Object.assign({}, freq), mark:[1, k]});
  });
  return ev;
}

/* one row per frequency bucket, highest count on top */
function bucketShapes(b, minf, mark, markBucket){
  const counts = Object.keys(b).map(Number).sort((p, q) => q - p);
  const out = [], top = 1.55, rowH = 1.12;
  counts.forEach((f, r) => {
    const y = top + r * rowH;
    out.push(S.r(.45, y, 1.05, .86, f === minf ? 'act' : 'soft', 'f=' + f, {fs:.32}));
    b[f].forEach((k, j) => {
      const hit = mark && mark[0] === f && mark[1] === k;
      const st = hit ? 'hot' : (markBucket === f ? 'ok' : 'idle');
      out.push(S.r(1.95 + j * 1.0, y, .86, .86, st, k, {fs:.42}));
    });
    if (f === minf)
      out.push(S.t(.97, y - .22, 'min_freq', {c:COL.purpleL, fs:.28}));
    if (b[f].length > 1 && r === counts.length - 1)
      out.push(S.t(1.95 + (b[f].length * 1.0) / 2 - .07, y + 1.18,
                   {zh:'同一格內：舊 → 新', en:'inside a bucket: older → newer'},
                   {c:COL.grey, fs:.27}));
  });
  return out;
}

function lfuFrames(varIx){
  const lruTie = varIx !== 1;
  const F = new Frames(), ev = runLFU(lruTie);
  ev.forEach(e => {
    const sh = opRow(LFU_SCRIPT.map(o => o[0] + ' ' + o[1]), e.i, .35)
      .concat(bucketShapes(e.b, e.minf, e.mark, e.markBucket));
    if (e.dead != null){
      sh.push(S.r(7.9, 1.55, 1.25, .9, 'bad', e.dead, {fs:.44}));
      sh.push(S.t(8.52, 2.86, {zh:'丟掉', en:'evicted'}, {c:COL.red, fs:.3}));
    }
    const k = e.op[1];
    let msg, line;
    switch (e.phase){
      case 'miss':
        msg = {zh:'get ' + k + '：不在快取裡，回傳 −1，次數當然也不會增加。',
               en:'get ' + k + ': not cached, return -1 - and nothing gets its counter bumped.'};
        line = ln(CODE_LFU, 'self.vals = {}'); break;
      case 'hit':
        msg = {zh:'get ' + k + ' 命中。接下來要把它搬到「次數 +1」的那一格。',
               en:'get ' + k + ' hits. Now it has to move to the bucket one count higher.'};
        line = ln(CODE_LFU, 'f = self.freq[key]'); break;
      case 'bump':
        msg = e.from === e.minf - 1
          ? {zh:'搬完之後 f=' + e.from + ' 這一格空了，min_freq 只能往上 1 變成 ' + e.minf +
                '。<b>只有這一種情況</b>，所以永遠不必去找最小值。',
             en:'Moving it emptied bucket f=' + e.from + ', so min_freq can only step up by one to ' +
                e.minf + '. <b>That is the only case</b>, which is why it never has to search.'}
          : {zh:'搬到 f=' + (e.from + 1) + '，放在那一格的<b>最後面</b> —— 同次數時，舊的排前面先被丟。',
             en:'It lands at the back of bucket f=' + (e.from + 1) +
                ' - within one count, the older use sits at the front and dies first.'};
        line = ln(CODE_LFU, e.from === e.minf - 1 ? 'self.min_freq = f + 1' : 'buckets[f + 1][key]'); break;
      case 'full':
        msg = {zh:'put ' + k + '：滿了。受害者一定在 min_freq 這一格裡 —— 不掃描、不排序。',
               en:'put ' + k + ': full. The victim is inside the min_freq bucket - no scan, no sort.'};
        line = ln(CODE_LFU, 'popitem(last=False)'); break;
      case 'evict':
        msg = lruTie
          ? {zh:'丟掉 ' + e.dead + '：同次數的情況下取最舊用過的那個。LC 460 會考這個平手規則。',
             en:'Evict ' + e.dead + ': on a tie, take the least recently used one. LC 460 tests exactly this.'}
          : {zh:'這個版本改用「先插入的先丟」，丟掉 ' + e.dead +
                '。程式不會壞掉，只是答案不一樣 —— 而 LC 460 認的是上面那個。',
             en:'This version breaks the tie by insertion order and drops ' + e.dead +
                '. Nothing crashes; the answers just differ - and LC 460 wants the other one.'};
        line = ln(CODE_LFU, 'del self.vals[victim]'); break;
      default:
        msg = {zh:'新的 key 次數是 1，min_freq 直接重設成 1 —— 剛進來的東西是最沒有資格留下的。',
               en:'A new key starts at count 1, so min_freq resets to 1 - a newcomer has earned nothing yet.'};
        line = ln(CODE_LFU, 'self.min_freq = 1');
    }
    const panels = [
      {lbl:{zh:'次數（key: count）', en:'counts (key: count)'},
       chips:Object.keys(e.freq).sort().map(x => chip(x + ':' + e.freq[x]))},
      {lbl:{zh:'min_freq', en:'min_freq'}, chips:[chip(String(e.minf), 'act')]},
      {lbl:{zh:'平手時丟誰', en:'tie-break'},
       chips:[chip(lruTie ? 'least recently used' : 'first inserted', lruTie ? 'ok' : 'bad')]}
    ];
    F.push({shapes:sh, panels:panels, view:VIEW, line:line, msg:msg});
  });
  return F.list;
}

/* ======================================== 3. scan pollution: LRU's worst case */
const CODE_SCAN = [
  '# a loop of n+1 keys through a cache of n slots',
  'keys = ["A", "B", "C", "D", "E", "F"]      # 6 distinct keys',
  'cache = LRUCache(5)                        # 5 slots',
  '',
  'for round in range(60):                    # the demo plays the first 3',
  '    for k in keys:',
  '        if cache.get(k) == -1:             # always a miss',
  '            cache.put(k, work(k))          # evicts the one needed next',
  '',
  '# measured over 60 rounds:',
  '#   lru 0.0%   lfu 0.0%   fifo 0.0%   slru 0.0%   mru 78.9%',
  '# nothing is hit twice, so "frequently used" protects nobody',
  '#',
  '# sglang exposes this as a flag, not a rewrite:',
  '#   --radix-eviction-policy mru'
];

const SCAN_KEYS = ['A', 'B', 'C', 'D', 'E', 'F'];
const SCAN_CAP = 5;

function runScan(mru){
  const list = [], ev = [];   /* list[0] = most recent */
  let hits = 0, miss = 0, step = 0;
  for (let r = 0; r < 3; r++){
    for (let j = 0; j < SCAN_KEYS.length; j++){
      const k = SCAN_KEYS[j];
      const at = list.indexOf(k);
      if (at >= 0){
        hits++; list.splice(at, 1); list.unshift(k);
        ev.push({k:k, r:r, j:j, phase:'hit', list:list.slice(), hits:hits, miss:miss, step:++step});
        continue;
      }
      miss++;
      let dead = null;
      if (list.length === SCAN_CAP) dead = mru ? list.shift() : list.pop();
      list.unshift(k);
      ev.push({k:k, r:r, j:j, phase:'miss', list:list.slice(), hits:hits, miss:miss,
               dead:dead, step:++step});
    }
  }
  return ev;
}

function scanFrames(varIx){
  const mru = varIx === 1;
  const F = new Frames(), ev = runScan(mru);
  ev.forEach(e => {
    const st = {0:e.phase === 'hit' ? 'ok' : 'hot'};
    const sh = opRow(SCAN_KEYS.map(x => 'get ' + x), e.j, .35)
      .concat(listRow(e.list, 2.7, st, {x0:1.15}));
    sh.push(S.t(4.9, 5.35, {zh:'第 ' + (e.r + 1) + ' 圈', en:'round ' + (e.r + 1)},
                {c:COL.grey, fs:.3}));
    const rate = e.hits + e.miss ? (100 * e.hits / (e.hits + e.miss)).toFixed(1) : '0.0';
    sh.push(S.t(4.9, 5.85, {zh:'命中率 ' + rate + '%', en:'hit rate ' + rate + '%'},
                {c:e.hits ? COL.tealL : COL.red, fs:.36}));
    if (e.dead) sh.push(S.t(8.6, 2.45, {zh:'丟掉 ' + e.dead, en:'evicted ' + e.dead},
                            {c:COL.red, fs:.32}));
    let msg;
    if (e.phase === 'hit'){
      msg = {zh:'命中。MRU 保住了整條隊伍的尾巴，所以除了剛被丟掉的那一格以外，其他全都還在。',
             en:'A hit. MRU sacrifices the same slot over and over, so the rest of the working set survives.'};
    } else if (e.dead && !mru){
      msg = {zh:'又是 miss，而被丟掉的 ' + e.dead + ' 正好是<b>下一圈馬上要用的</b>。' +
                '快取一直很忙，但命中率是 0%。',
             en:'Another miss - and the key it just evicted, ' + e.dead +
                ', is <b>exactly the one needed next time round</b>. The cache works hard and hits nothing.'};
    } else if (e.dead && mru){
      msg = {zh:'MRU 丟掉的是剛剛才放進來的 ' + e.dead + '，看起來很蠢，但它讓其他 4 個 key 活下來。',
             en:'MRU throws away ' + e.dead +
                ', the key it just inserted. It looks absurd, and it keeps the other four alive.'};
    } else {
      msg = {zh:'還在暖機，位子還沒滿，先不用挑受害者。',
             en:'Still warming up - there are free slots, so nobody has to be evicted yet.'};
    }
    F.push({shapes:sh, panels:[
      {lbl:{zh:'命中 / 錯過', en:'hits / misses'},
       chips:[chip(String(e.hits), e.hits ? 'ok' : 'bad'), chip(String(e.miss), 'hot')]},
      {lbl:{zh:'快取內容（最近 → 最久）', en:'cache (recent → stale)'},
       chips:e.list.map(x => chip(x))},
      {lbl:{zh:'策略', en:'policy'}, chips:[chip(mru ? 'mru' : 'lru', mru ? 'ok' : 'bad')]}
    ], view:VIEW, line:ln(CODE_SCAN, e.phase === 'hit' ? 'mru 78.9%'
                                   : (e.dead ? 'cache.put(k' : 'if cache.get(k)')),
      msg:msg});
  });
  return F.list;
}

/* ============================================ 4. one heap, five key functions */
const CODE_POLICY = [
  '# sglang/srt/mem_cache/evict_policy.py  (shape, not a quote)',
  'class LRUStrategy:  get_priority = lambda n: n.last_access_time',
  'class LFUStrategy:  get_priority = lambda n: (n.hit_count, n.last_access_time)',
  'class FIFOStrategy: get_priority = lambda n: n.creation_time',
  'class MRUStrategy:  get_priority = lambda n: -n.last_access_time',
  'class SLRUStrategy: get_priority = lambda n: (n.hit_count >= 2,',
  '                                              n.last_access_time)',
  '',
  'victim = heapq.nsmallest(1, entries, key=strategy.get_priority)[0]',
  '',
  '#   --radix-eviction-policy {lru,lfu,fifo,mru,slru}',
  '# five policies, one heap: the only thing that changes is the key function'
];

/* four entries with hand-picked fields so every policy picks a different victim */
const ENTRIES = [
  {key:'A', created:5, last:9, hits:6},
  {key:'B', created:3, last:2, hits:4},
  {key:'C', created:1, last:4, hits:1},
  {key:'D', created:8, last:7, hits:0}
];
const POLICY_DEFS = [
  {id:'lru',  p:e => [e.last],                   why:{zh:'最久沒被用到的', en:'the stalest last access'}},
  {id:'lfu',  p:e => [e.hits, e.last],           why:{zh:'用最少次的（平手比最久沒用）', en:'fewest hits, ties by recency'}},
  {id:'fifo', p:e => [e.created],                why:{zh:'最早進來的，不管有沒有在用', en:'oldest arrival, use ignored'}},
  {id:'mru',  p:e => [-e.last],                  why:{zh:'剛剛才用過的', en:'the one just touched'}},
  {id:'slru', p:e => [e.hits >= 2 ? 1 : 0, e.last],
   why:{zh:'先丟沒被重複用過的（hit_count < 2）', en:'unprotected first (hit_count < 2)'}}
];
const cmpKey = (a, b) => { for (let i = 0; i < a.length; i++){ if (a[i] !== b[i]) return a[i] - b[i]; } return 0; };

function policyFrames(varIx){
  const F = new Frames(), P = POLICY_DEFS[varIx || 0];
  const keys = ENTRIES.map(e => ({e:e, k:P.p(e)}));
  const sorted = keys.slice().sort((x, y) => cmpKey(x.k, y.k));
  const victim = sorted[0].e.key;

  const entryShapes = (st) => {
    const out = [];
    ENTRIES.forEach((e, i) => {
      const x = .6 + i * 2.25;
      out.push(S.r(x, .9, 1.95, 1.85, st[e.key] || 'idle', '', {}));
      out.push(S.t(x + .97, 1.42, e.key, {c:COL.pale, fs:.5}));
      out.push(S.t(x + .97, 1.95, 'last=' + e.last, {c:COL.grey, fs:.28}));
      out.push(S.t(x + .97, 2.34, 'hits=' + e.hits + '  born=' + e.created, {c:COL.grey, fs:.28}));
    });
    return out;
  };

  F.push({shapes:entryShapes({}).concat([
      S.t(4.9, 3.5, {zh:'四個節點，欄位一模一樣', en:'four entries, identical fields'}, {c:COL.tealL, fs:.34}),
      S.t(4.9, 4.1, {zh:'差別只在下一步怎麼把它們壓成一個可比較的 key',
                     en:'all that differs is how we squeeze them into one comparable key'},
          {c:COL.grey, fs:.3})]),
    panels:[{lbl:{zh:'策略', en:'policy'}, chips:[chip(P.id, 'act')]}],
    view:VIEW, line:ln(CODE_POLICY, 'evict_policy.py'),
    msg:{zh:'每個節點都記著三件事：最後被用的時間、被命中幾次、什麼時候建立的。策略不改資料結構，只改<b>怎麼比大小</b>。',
         en:'Every entry records the same three things: last access, hit count, creation time. A policy changes none of that - only <b>how two entries compare</b>.'}});

  const kst = {}; ENTRIES.forEach(e => kst[e.key] = 'act');
  F.push({shapes:entryShapes(kst).concat(
      ENTRIES.map((e, i) => S.t(.6 + i * 2.25 + .97, 3.55, '(' + P.p(e).join(', ') + ')',
                                {c:COL.purpleL, fs:.34}))).concat([
      S.t(4.9, 4.35, P.why, {c:COL.tealL, fs:.34})]),
    panels:[{lbl:{zh:'get_priority 算出來的 key', en:'get_priority keys'},
             chips:ENTRIES.map(e => chip(e.key + ' → ' + P.p(e).join(','), 'act'))}],
    view:VIEW, line:ln(CODE_POLICY, 'class ' + P.id.toUpperCase() + 'Strategy'),
    msg:{zh:'套上 ' + P.id + ' 的 get_priority：四個節點各自得到一個 key。整個策略就是這一行 lambda。',
         en:'Apply the ' + P.id + ' key function: each entry collapses to one comparable key. The whole policy is that single lambda.'}});

  const vst = {}; ENTRIES.forEach(e => vst[e.key] = e.key === victim ? 'bad' : 'done');
  F.push({shapes:entryShapes(vst).concat([
      S.r(3.3, 3.55, 3.2, .95, 'hot', 'nsmallest(1, …)', {fs:.36}),
      S.t(4.9, 5.15, {zh:'受害者：' + victim, en:'victim: ' + victim}, {c:COL.red, fs:.42}),
      S.t(4.9, 5.75, {zh:'換一個策略，同一份資料會挑出不同的人',
                      en:'switch the policy and the same data yields a different victim'},
          {c:COL.grey, fs:.3})]),
    panels:[{lbl:{zh:'排序後（最先被丟 → 最後）', en:'sorted (first out → last)'},
             chips:sorted.map((s, i) => chip(s.e.key, i === 0 ? 'bad' : 'done'))},
            {lbl:{zh:'受害者', en:'victim'}, chips:[chip(victim, 'bad')]}],
    view:VIEW, line:ln(CODE_POLICY, 'nsmallest'),
    msg:{zh:'一顆 heap 取最小的那個就是受害者。五種策略共用同一段程式，使用者端只是 <b>--radix-eviction-policy</b> 換個字串。',
         en:'One heap, smallest key wins the axe. All five policies share this code - on the user side it is just a different string after <b>--radix-eviction-policy</b>.'}});
  return F.list;
}

/* ================================== 5. LeetCode 432 - the bucket trick alone */
const CODE_ALLONE = [
  'class AllOne:                          # LeetCode 432, everything O(1)',
  '    # a sorted doubly linked list of count buckets,',
  '    # each bucket holding a set of keys with that count',
  '',
  '    def inc(self, key):',
  '        cur = self.nodes.get(key)',
  '        c = cur.count if cur else 0',
  '        nxt = cur.next if cur else self.head.next',
  '        if nxt is self.tail or nxt.count != c + 1:',
  '            nxt = self._insert_after(cur or self.head, c + 1)',
  '        nxt.keys.add(key); self.nodes[key] = nxt',
  '        if cur: self._remove_key(cur, key)     # drop the empty bucket',
  '',
  '    def getMaxKey(self): return next(iter(self.tail.prev.keys), "")',
  '    def getMinKey(self): return next(iter(self.head.next.keys), "")',
  '',
  '# counts only ever move by 1, so the right bucket is always the',
  '# neighbour - which is why no sorting and no heap is needed'
];

const ALLONE_SCRIPT = [['inc', 'hello'], ['inc', 'hello'], ['inc', 'leet'],
                       ['inc', 'hello'], ['inc', 'leet'], ['inc', 'code'],
                       ['dec', 'code']];

function runAllOne(){
  const cnt = {}, ev = [];
  const chain = () => {
    const by = {};
    Object.keys(cnt).forEach(k => { (by[cnt[k]] = by[cnt[k]] || []).push(k); });
    return Object.keys(by).map(Number).sort((a, b) => a - b)
             .map(c => ({c:c, keys:by[c].sort()}));
  };
  ALLONE_SCRIPT.forEach((op, i) => {
    const [kind, k] = op, before = cnt[k] || 0;
    ev.push({i:i, op:op, phase:'look', chain:chain(), mark:k, before:before});
    if (kind === 'inc') cnt[k] = before + 1;
    else if (before <= 1) delete cnt[k]; else cnt[k] = before - 1;
    ev.push({i:i, op:op, phase:'move', chain:chain(), mark:cnt[k] != null ? k : null,
             before:before, after:cnt[k] == null ? 0 : cnt[k]});
  });
  const c = chain();
  ev.push({i:ALLONE_SCRIPT.length - 1, op:['getMax', ''], phase:'ends', chain:c,
           mx:c.length ? c[c.length - 1].keys[0] : '', mn:c.length ? c[0].keys[0] : ''});
  return ev;
}

function allOneFrames(){
  const F = new Frames(), ev = runAllOne();
  ev.forEach(e => {
    const sh = opRow(ALLONE_SCRIPT.map(o => o[0] + ' ' + o[1]), e.i, .35);
    const n = e.chain.length, w = 1.7, gap = .45;
    const x0 = 4.9 - (n * w + (n - 1) * gap) / 2;
    e.chain.forEach((b, j) => {
      const x = x0 + j * (w + gap);
      const hot = b.keys.indexOf(e.mark) >= 0;
      sh.push(S.r(x, 2.1, w, 1.5, hot ? 'hot' : 'idle', '', {}));
      sh.push(S.t(x + w / 2, 2.62, 'count ' + b.c, {c:COL.grey, fs:.3}));
      sh.push(S.t(x + w / 2, 3.22, b.keys.join(' '), {c:hot ? COL.orangeL : COL.pale, fs:.34}));
      if (j) sh.push(S.e(x - gap, 2.85, x, 2.85, {w:.05}));
    });
    if (n){
      sh.push(S.t(x0 + w / 2, 1.85, {zh:'getMinKey', en:'getMinKey'}, {c:COL.purpleL, fs:.3}));
      sh.push(S.t(x0 + (n - 1) * (w + gap) + w / 2, 1.85, 'getMaxKey', {c:COL.purpleL, fs:.3}));
    }
    let msg, line;
    if (e.phase === 'look'){
      msg = {zh:'先在 map 裡找到 ' + e.op[1] + ' 現在住在哪一格（count=' + e.before + '）。' +
                '這一步和 LFU 的 bucket 一模一樣。',
             en:'First the map says which bucket ' + e.op[1] + ' lives in (count=' + e.before +
                '). This is the LFU bucket trick, on its own.'};
      line = ln(CODE_ALLONE, 'cur = self.nodes.get');
    } else if (e.phase === 'move'){
      msg = e.after === 0
        ? {zh:'次數掉到 0，key 直接消失，空掉的格子也要一起拆掉 —— 不然 getMinKey 會指到一個空殼。',
           en:'The count hit zero, so the key disappears and the empty bucket must go with it - otherwise getMinKey points at an empty shell.'}
        : {zh:'次數 ' + e.before + ' → ' + e.after + '，只會移動到<b>隔壁</b>那一格。要找的位置永遠是鄰居，所以不用搜尋。',
           en:'The count goes ' + e.before + ' → ' + e.after +
              ', so the key moves to the <b>adjacent</b> bucket. The destination is always a neighbour, so nothing is searched.'};
      line = ln(CODE_ALLONE, e.after === 0 ? '_remove_key' : 'nxt.keys.add');
    } else {
      msg = {zh:'最大和最小就是鏈的兩端：getMaxKey = ' + e.mx + '，getMinKey = ' + e.mn +
                '。兩端只是陣列的頭尾指標，不需要 heap 也不需要排序。',
             en:'Max and min are just the two ends of the chain: getMaxKey = ' + e.mx +
                ', getMinKey = ' + e.mn + '. Both are just the ends of the chain - no heap, no sorting, no scan.'};
      line = ln(CODE_ALLONE, 'getMaxKey');
    }
    const counts = {};
    e.chain.forEach(b => b.keys.forEach(k => counts[k] = b.c));
    F.push({shapes:sh, panels:[
      {lbl:{zh:'count（key: 次數）', en:'counts (key: count)'},
       chips:Object.keys(counts).sort().map(k => chip(k + ':' + counts[k]))},
      {lbl:{zh:'bucket 鏈（小 → 大）', en:'bucket chain (low → high)'},
       chips:e.chain.map(b => chip(String(b.c)))}
    ], view:VIEW, line:line, msg:msg});
  });
  return F.list;
}

/* ======================================================================= meta */
const DAY_META = {
  title:{zh:'Day 30 — LRU 與 LFU 快取：滿了的時候丟掉誰',
         en:'Day 30 - LRU and LFU caches: choosing what to throw away'},
  sub:{zh:'快取 = 有上限的 map + 一條挑受害者的規則。真正決定快或慢的是那條規則。',
       en:'A cache is a bounded map plus a rule for picking a victim. The rule is what decides whether it is fast.'},
  tabs:[
    {id:'lru', label:{zh:'LRU（LC 146）', en:'LRU (LC 146)'},
     stage:{zh:'hash map + 雙向串列', en:'hash map + doubly linked list'},
     view:VIEW,
     variants:[{zh:'get 會刷新（正確的 LRU）', en:'get refreshes (real LRU)'},
               {zh:'get 不刷新（悄悄變成 FIFO）', en:'get does not refresh (silently FIFO)'}],
     idea:{zh:'map 負責 O(1) 找到節點，串列負責記住順序。把 get 裡那兩行刷新刪掉，程式照樣跑、照樣不會噴錯，但它記的已經不是「最久沒用」而是「最早進來」—— LRU 靜悄悄地變成了 FIFO。',
           en:'The map finds a node in O(1); the list remembers the order. Delete the two refresh lines inside get and the code still runs and still never raises - but the order now records arrival rather than use, and your LRU has quietly become a FIFO.'},
     legend:['hot', 'act', 'ok', 'bad', 'done'], code:CODE_LRU, build:lruFrames},

    {id:'lfu', label:{zh:'LFU（LC 460）', en:'LFU (LC 460)'},
     stage:{zh:'次數 bucket + min_freq', en:'frequency buckets + min_freq'},
     view:VIEW,
     variants:[{zh:'平手比最久沒用（LC 460 要的）', en:'tie-break by recency (what LC 460 wants)'},
               {zh:'平手比先插入的', en:'tie-break by insertion order'}],
     idea:{zh:'每個次數一個 OrderedDict，再加一個 min_freq。min_freq 只有兩種變法：命中把最小那格清空時變成 f+1，插入新 key 時變回 1。沒有第三種情況，所以它永遠不必去找最小值 —— 這就是 O(1) 的來源。',
           en:'One OrderedDict per frequency, plus min_freq. It only ever changes two ways: to f+1 when a hit empties the minimum bucket, and back to 1 when a new key arrives. There is no third case, so it never has to search - and that is where the O(1) comes from.'},
     legend:['hot', 'act', 'ok', 'bad', 'idle'], code:CODE_LFU, build:lfuFrames},

    {id:'scan', label:{zh:'掃描污染', en:'scan pollution'},
     stage:{zh:'6 個 key 繞 5 個位子', en:'6 keys looping through 5 slots'},
     view:VIEW,
     variants:[{zh:'LRU', en:'LRU'}, {zh:'MRU', en:'MRU'}],
     idea:{zh:'比快取大一點點的迴圈，會讓 LRU 每次都丟掉「下一個馬上要用的」，命中率剛好 0%。LFU 和 SLRU 一樣是 0%，因為沒有任何 key 被用過第二次，誰也累積不到次數。只有反其道而行的 MRU 活得下來。',
           en:'A loop slightly larger than the cache makes LRU evict exactly the key it is about to ask for, for a hit rate of precisely 0%. LFU and SLRU score 0% too - nothing is ever touched twice, so no counter ever rises. Only MRU, doing the opposite, survives.'},
     legend:['hot', 'ok', 'bad', 'idle'], code:CODE_SCAN, build:scanFrames},

    {id:'policy', label:{zh:'五種策略一顆 heap', en:'five policies, one heap'},
     stage:{zh:'entry → get_priority → 受害者', en:'entry → get_priority → victim'},
     view:VIEW,
     variants:[{zh:'lru', en:'lru'}, {zh:'lfu', en:'lfu'}, {zh:'fifo', en:'fifo'},
               {zh:'mru', en:'mru'}, {zh:'slru', en:'slru'}],
     idea:{zh:'沒有一種策略能贏過所有 workload，所以 sglang 不選邊站：五種策略就是五個 get_priority 的 key function，共用同一顆 heap，使用者用 --radix-eviction-policy 換字串。把策略寫成「怎麼比大小」，而不是寫成五份資料結構。',
           en:'No single policy wins every workload, so sglang refuses to pick one: the five strategies are five get_priority key functions over one shared heap, selected by a --radix-eviction-policy string. The policy is expressed as a comparison, not as five separate data structures.'},
     legend:['hot', 'act', 'bad', 'done', 'idle'], code:CODE_POLICY, build:policyFrames},

    {id:'allone', label:{zh:'LC 432 All O(1)', en:'LC 432 All O(1)'},
     stage:{zh:'次數 bucket 串成一條鏈', en:'count buckets on a linked chain'},
     view:VIEW,
     idea:{zh:'把 LFU 的 bucket 單獨拿出來就是 LC 432：次數一次只加減 1，所以目標格永遠是隔壁那一格，四個操作全部 O(1)。最大最小就是鏈的兩端，不用 heap 也不用排序。',
           en:'Strip the value cache away from LFU and what remains is LC 432: a count only ever moves by one, so the destination bucket is always the neighbour, and all four operations are O(1). Max and min are simply the two ends of the chain - no heap, no sorting.'},
     legend:['hot', 'act', 'idle'], code:CODE_ALLONE, build:allOneFrames}
  ]
};
