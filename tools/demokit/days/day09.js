// DAY: 09
// TITLE_ZH: 雜湊表：分離鏈結法
// TITLE_EN: Hash Table - Chaining
// SUB_ZH: 撞在一起怎麼辦？掛一條串列就好。負載因子、平均查詢長度、以及為什麼要重新雜湊。
// SUB_EN: When two keys collide, hang a list off the bucket.
// FOLDER: day%2009%20-%20hash%20table%20-%20chaining
// MEDIUM: https://medium.com/100-days-of-python/day-09-hash-table-chaining-ef74baa6732

function hsum(k){ let s = 0; for (let i = 0; i < k.length; i++) s += k.charCodeAt(i); return s; }
const BY = .95, BH = .74, BXX = 1.35, BW = 1.15, GAPY = .96, EW = 1.45;
function tableShapes(cap, table, o){
  o = o || {}; const out = [];
  for (let i = 0; i < cap; i++){
    const y = BY + i * GAPY;
    out.push(S.r(BXX, y, BW, BH, (o.slot && o.slot[i]) || 'soft', String(i), {fs:.36}));
    const chain = table[i] || [];
    if (!chain.length)
      out.push(S.t(BXX + BW + .62, y + BH * .68, 'None', {c:'#8fa3ac', fs:.28}));
    chain.forEach((e, j) => {
      const x = BXX + BW + .55 + j * (EW + .55);
      out.push(S.e(x - .50, y + BH / 2, x - .04, y + BH / 2,
        {s:(o.link && o.link[i] === j) ? 'hot' : 'idle', w:.045}));
      out.push(S.r(x, y, EW, BH, (o.ent && o.ent[e[0]]) || 'idle', e[0] + ':' + e[1], {fs:.34}));
    });
  }
  out.push(S.t(BXX + BW / 2, BY - .30, {zh:'桶子', en:'buckets'}, {c:'#3fe0dd', fs:.30}));
  return out;
}
const stats = (cap, table) => {
  let n = 0, longest = 0;
  for (let i = 0; i < cap; i++){ const c = (table[i] || []).length; n += c; longest = Math.max(longest, c); }
  return {n:n, alpha:(n / cap), longest:longest};
};
const statPanel = (cap, table) => {
  const s = stats(cap, table);
  return {lbl:{zh:'負載因子 α = n / 桶數', en:'load factor a = n / buckets'},
    chips:[{t:'n = ' + s.n, cls:''}, {t:'buckets = ' + cap, cls:''},
           {t:'a = ' + s.alpha.toFixed(2), cls:s.alpha > .75 ? 'bad' : 'ok'},
           {t:(LANG === 'zh' ? '最長鏈 ' : 'longest chain ') + s.longest, cls:s.longest > 2 ? 'bad' : ''}]};
};

const CODE_INS = [
'class HashTable:',
'    def __init__(self, cap=5):',
'        self.cap = cap',
'        self.slots = [[] for _ in range(cap)]   # 每格掛一條串列',
'',
'    def put(self, key, value):',
'        i = hash(key) % self.cap        # 1. 算出桶子',
'        for e in self.slots[i]:         # 2. 走這條鏈',
'            if e.key == key:',
'                e.value = value         #    已存在就更新',
'                return',
'        self.slots[i].append(Entry(key, value))   # 3. 掛到鏈尾'];
const CODE_GET = [
'def get(self, key):',
'    i = hash(key) % self.cap        # 一步跳到桶子',
'    for e in self.slots[i]:         # 只走這一條鏈',
'        if e.key == key:',
'            return e.value',
'    raise KeyError(key)',
'',
'def remove(self, key):',
'    i = hash(key) % self.cap',
'    self.slots[i] = [e for e in self.slots[i] if e.key != key]'];
const CODE_RES = [
'def _ensure_capacity(self):',
'    if self.n / self.cap <= 0.75:   # 負載因子還可以',
'        return',
'    old = self.slots',
'    self.cap *= 2                   # 桶子加倍',
'    self.slots = [[] for _ in range(self.cap)]',
'    for chain in old:',
'        for e in chain:',
'            i = hash(e.key) % self.cap   # 位置會變！必須重算',
'            self.slots[i].append(e)'];

const KEYS = [['ann', 31], ['bob', 24], ['cat', 28], ['dan', 40], ['eve', 19]];

function buildInsert(){
  const cap = 5, table = {}, F = new Frames();
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:3,
    msg:{zh:'分離鏈結法：<b>每個桶子不是放一筆資料，而是放一條串列</b>。' +
            '這樣不管撞幾次都塞得下——問題只剩下鏈會不會太長。',
         en:'Separate chaining: <b>each bucket holds a list, not a single entry</b>. Collisions can always be absorbed; the only question is how long the chains get.'}});
  KEYS.forEach(([k, v]) => {
    const i = hsum(k) % cap;
    F.push({shapes:tableShapes(cap, table, {slot:{[i]:'hot'}}).concat(
      [S.r(6.6, .95, 1.7, BH, 'act', '"' + k + '"', {fs:.36}),
       S.t(7.45, 2.15, 'hash = ' + hsum(k), {c:'#c7a6ff', fs:.30}),
       S.t(7.45, 2.75, hsum(k) + ' % ' + cap + ' = ' + i, {c:'#ffbe6b', fs:.34})]),
      panels:[statPanel(cap, table)], line:6,
      msg:{zh:'放 <b>' + k + '</b>：hash("' + k + '") = ' + hsum(k) + '，取 <b>% ' + cap +
              '</b> 得到桶子 <b>' + i + '</b>。<b>雜湊函式的工作是「把 key 平均地打散」</b>，' +
              '取餘數則負責把它壓進桶子的範圍。',
           en:'Insert <b>' + k + '</b>: hash("' + k + '") = ' + hsum(k) + ', and <b>% ' + cap +
              '</b> gives bucket <b>' + i + '</b>. The hash function spreads keys evenly; the modulo squeezes the result into the table.'}});
    const existing = (table[i] || []).length;
    if (existing)
      F.push({shapes:tableShapes(cap, table, {slot:{[i]:'hot'},
        ent:(table[i] || []).reduce((a, e) => (a[e[0]] = 'act', a), {})}),
        panels:[statPanel(cap, table)], line:7,
        msg:{zh:'桶子 ' + i + ' 已經有 <b>' + table[i].map(e => e[0]).join('、') + '</b>——<b>碰撞</b>了。' +
                '先走一遍這條鏈確認 key 沒重複，再掛到鏈尾。',
             en:'Bucket ' + i + ' already holds <b>' + table[i].map(e => e[0]).join(', ') +
                '</b> - a <b>collision</b>. Walk the chain first to check for a duplicate key, then append.'}});
    table[i] = (table[i] || []).concat([[k, v]]);
    F.push({shapes:tableShapes(cap, table, {ent:{[k]:'ok'}}), panels:[statPanel(cap, table)], line:11,
      msg:{zh:'掛好了。碰撞<b>不是錯誤</b>，是必然：只要 key 比桶子多，就一定會撞（鴿籠原理）。' +
              '重點是撞了以後付出的代價有多大。',
           en:'Done. A collision is <b>not a bug</b> but a certainty: more keys than buckets means collisions (pigeonhole). What matters is what a collision costs.'}});
  });
  const s = stats(cap, table);
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:6,
    msg:{zh:'現在 α = ' + s.alpha.toFixed(2) + '，最長的鏈有 ' + s.longest + ' 筆。' +
            '<b>平均查詢長度大約是 1 + α/2</b>——所以只要 α 控制在常數，查詢就是 O(1)。',
         en:'Now a = ' + s.alpha.toFixed(2) + ' with a longest chain of ' + s.longest +
            '. <b>The average probe length is about 1 + a/2</b>, so keeping a bounded keeps lookup O(1).'}});
  return F.list;
}

function buildGet(v){
  const cap = 5, table = {};
  KEYS.forEach(([k, val]) => { const i = hsum(k) % cap; table[i] = (table[i] || []).concat([[k, val]]); });
  const key = v === 0 ? 'eve' : (v === 1 ? 'zoe' : 'bob');
  const F = new Frames(), i = hsum(key) % cap;
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:0,
    msg:{zh:(v === 2 ? '要<b>刪掉</b>' : '要<b>查</b>') + ' <b>' + key + '</b>。' +
            '注意：不管表裡有幾筆資料，第一步永遠只是算一次雜湊。',
         en:(v === 2 ? 'We will <b>delete</b> ' : 'We will <b>look up</b> ') + '<b>' + key +
            '</b>. Note that the first step is one hash computation, no matter how big the table is.'}});
  F.push({shapes:tableShapes(cap, table, {slot:{[i]:'hot'}}), panels:[statPanel(cap, table)],
    line:v === 2 ? 8 : 1,
    msg:{zh:'hash("' + key + '") % ' + cap + ' = <b>' + i + '</b>，<b>其他 ' + (cap - 1) +
            ' 個桶子完全不用看</b>。',
         en:'hash("' + key + '") % ' + cap + ' = <b>' + i + '</b>, and <b>the other ' + (cap - 1) +
            ' buckets are never touched</b>.'}});
  const chain = table[i] || [];
  let found = false;
  for (let j = 0; j < chain.length; j++){
    const hit = chain[j][0] === key;
    F.push({shapes:tableShapes(cap, table, {slot:{[i]:'act'}, link:{[i]:j},
      ent:{[chain[j][0]]:hit ? 'ok' : 'hot'}}), panels:[statPanel(cap, table)], line:v === 2 ? 9 : 2,
      msg:{zh:'走到鏈上第 ' + (j + 1) + ' 個：<b>' + chain[j][0] + '</b>' +
              (hit ? ' —— 就是它。' : '，不是 <b>' + key + '</b>，往下一個。'),
           en:'Chain position ' + (j + 1) + ': <b>' + chain[j][0] + '</b>' +
              (hit ? ' - that is the one.' : ', not <b>' + key + '</b>, keep walking.')}});
    if (hit){ found = true;
      if (v === 2){
        table[i] = chain.filter(e => e[0] !== key);
        F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:9,
          msg:{zh:'從鏈上拿掉就好——<b>鏈結法的刪除很乾淨</b>，' +
                  '不會在表裡留下任何痕跡影響其他 key（下一天的開放定址就沒這麼好運）。',
               en:'Just unlink it. <b>Deletion under chaining is clean</b>: nothing is left behind to confuse other keys - unlike open addressing tomorrow.'}});
      }
      break; }
  }
  if (!found)
    F.push({shapes:tableShapes(cap, table, {slot:{[i]:'bad'}}), panels:[statPanel(cap, table)], line:5,
      msg:{zh:'整條鏈走完都沒有 → <b>KeyError</b>。查不到的成本也只是這條鏈的長度，' +
              '<b>跟表裡總共有多少筆資料無關</b>。',
           en:'The chain ran out, so <b>KeyError</b>. A miss costs the length of this one chain - <b>independent of how much the table holds</b>.'}});
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:v === 2 ? 9 : 4,
    msg:{zh:'成本 = <b>1 次雜湊 + 走一條鏈</b>。鏈的平均長度就是 α，' +
            '所以整張表的效能好壞，其實全部押在「α 別讓它變大」這件事上。',
         en:'Cost = <b>one hash plus one chain walk</b>. The average chain length is a, so the entire performance story reduces to keeping a small.'}});
  return F.list;
}

function buildResize(){
  const F = new Frames();
  let cap = 5, table = {};
  KEYS.forEach(([k, val]) => { const i = hsum(k) % cap; table[i] = (table[i] || []).concat([[k, val]]); });
  const extra = [['fay', 22], ['gus', 35]];
  extra.forEach(([k, val]) => { const i = hsum(k) % cap; table[i] = (table[i] || []).concat([[k, val]]); });
  const s0 = stats(cap, table);
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:1,
    msg:{zh:'塞了 ' + s0.n + ' 筆進 ' + cap + ' 個桶子，α = <b>' + s0.alpha.toFixed(2) +
            '</b>，最長的鏈已經 ' + s0.longest + ' 筆。' +
            '<b>再下去查詢就從 O(1) 慢慢變成走串列的 O(n)</b>。',
         en:'With ' + s0.n + ' entries in ' + cap + ' buckets, a = <b>' + s0.alpha.toFixed(2) +
            '</b> and the longest chain is ' + s0.longest + '. <b>Left alone, lookups drift from O(1) towards walking a list.</b>'}});
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:4,
    msg:{zh:'超過門檻（這裡用 0.75）就<b>把桶子數加倍</b>。門檻是個工程選擇：' +
            '太低浪費記憶體，太高鏈太長。',
         en:'Past the threshold (0.75 here) we <b>double the bucket count</b>. The threshold is an engineering dial: too low wastes memory, too high lengthens chains.'}});
  const old = table, oldCap = cap;
  cap = cap * 2; table = {};
  const all = [];
  for (let i = 0; i < oldCap; i++) (old[i] || []).forEach(e => all.push([i, e]));
  all.forEach(([from, e], idx) => {
    const to = hsum(e[0]) % cap;
    table[to] = (table[to] || []).concat([e]);
    F.push({shapes:tableShapes(cap, table, {ent:{[e[0]]:'hot'}, slot:{[to]:'hot'}}),
      panels:[statPanel(cap, table)], line:8,
      msg:{zh:'搬 <b>' + e[0] + '</b>：舊桶子 ' + from + ' → 新桶子 <b>' + to + '</b>（' +
              hsum(e[0]) + ' % ' + cap + '）。<b>不能直接複製過去</b>，' +
              '因為位置是 hash % 桶數算出來的，桶數變了位置就變了。',
           en:'Move <b>' + e[0] + '</b>: bucket ' + from + ' becomes <b>' + to + '</b> (' +
              hsum(e[0]) + ' % ' + cap + '). <b>Copying is not enough</b> - the slot is hash % capacity, and the capacity just changed.'}});
  });
  const s1 = stats(cap, table);
  F.push({shapes:tableShapes(cap, table), panels:[statPanel(cap, table)], line:9,
    msg:{zh:'重新雜湊完成：α 從 ' + s0.alpha.toFixed(2) + ' 降到 <b>' + s1.alpha.toFixed(2) +
            '</b>，最長鏈剩 ' + s1.longest + '。這一次 rehash 是 <b>O(n)</b> 的大工程，' +
            '但因為每次都是<b>加倍</b>，攤還下來每筆插入還是 <b>O(1)</b>——' +
            '跟 Python list 擴容是同一套算術。',
         en:'Rehash complete: a drops from ' + s0.alpha.toFixed(2) + ' to <b>' + s1.alpha.toFixed(2) +
            '</b> and the longest chain is ' + s1.longest + '. That rehash cost <b>O(n)</b>, but because we <b>double</b> each time it amortises to <b>O(1)</b> per insert - the same arithmetic that grows a Python list.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'雜湊表：分離鏈結法', en:'Hash Table - Chaining'},
  sub:{zh:'撞在一起怎麼辦？掛一條串列就好。負載因子、平均查詢長度、以及為什麼要重新雜湊。',
       en:'What happens when two keys collide? Hang a list off the bucket. Load factor, probe length, and why rehashing is unavoidable.'},
  tabs:[
    {id:'ins', label:{zh:'插入與碰撞', en:'insert & collisions'},
     stage:{zh:'key → hash → 桶子 → 掛上鏈', en:'key to hash to bucket to chain'}, view:[10, 6.4],
     idea:{zh:'雜湊表把「比較」換成「計算」：<b>位置是算出來的，不是找出來的</b>。' +
              '碰撞不可避免（key 比桶子多），鏈結法的回答是——就讓它們排隊。',
           en:'A hash table replaces comparison with arithmetic: <b>the slot is computed, not searched for</b>. Collisions are inevitable (more keys than buckets), and chaining answers them by letting colliders queue up.'},
     legend:['hot', 'act', 'ok', 'soft'], code:CODE_INS, build:buildInsert},
    {id:'get', label:{zh:'查詢與刪除', en:'lookup & delete'},
     stage:{zh:'一次雜湊，只走一條鏈', en:'One hash, one chain'}, view:[10, 6.4],
     variants:[{zh:'查得到 eve', en:'hit: eve'}, {zh:'查不到 zoe', en:'miss: zoe'},
               {zh:'刪除 bob', en:'delete bob'}],
     idea:{zh:'查詢成本 = 1 次雜湊 + 走一條鏈，平均長度是 α。' +
              '鏈結法還有一個實務上的好處：<b>刪除很乾淨</b>，直接從鏈上拿掉就好。',
           en:'Lookup costs one hash plus one chain walk of average length a. Chaining also has a practical bonus: <b>deletion is clean</b> - simply unlink the entry.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_GET, build:buildGet},
    {id:'resize', label:{zh:'負載因子與重新雜湊', en:'load factor & rehash'},
     stage:{zh:'桶子加倍，全部重算位置', en:'Double the buckets, recompute every slot'}, view:[10, 6.4],
     idea:{zh:'O(1) 是<b>有條件</b>的：條件就是 α 被壓住。超過門檻就把桶子加倍並<b>重算每一個位置</b>' +
              '（位置是 hash % 桶數，桶數變了就全變）。因為是倍增，攤還後插入仍是 O(1)。',
           en:'The O(1) comes with a condition: a must stay bounded. Past the threshold, double the buckets and <b>recompute every slot</b> (it is hash % capacity, and capacity changed). Because growth is multiplicative, inserts stay O(1) amortised.'},
     legend:['hot', 'ok', 'bad', 'soft'], code:CODE_RES, build:buildResize}
  ]
};
