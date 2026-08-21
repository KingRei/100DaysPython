// DAY: 07
// TITLE_ZH: list、tuple、dict 與 set
// TITLE_EN: list, tuple, dict and set
// SUB_ZH: 四個內建容器不是四種語法，是四種不同的成本結構——挑錯了，程式就會慢一個級數。
// SUB_EN: Four built-in containers, four cost models.
// FOLDER: day%2007%20-%20list%20tuple%20dict%20and%20set
// MEDIUM: https://medium.com/100-days-of-python/day-07-python-list-tuple-dict-and-set-78a164a5e207

const CW = 1.02, CH = .86;
const row = (vals, y, states, opt) => cellRow(vals, 5.0 - vals.length * CW / 2, y, CW, CH,
  Object.assign({states:states, fs:.42, ilift:.32}, opt || {}));
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));
const note = (y, s, c) => S.t(5.0, y, s, {c:c || '#8fa3ac', fs:.34});

const CODE_LIST = [
'nums = [10, 20, 30, 40]',
'',
'nums.append(50)      # 尾端加 -> 攤還 O(1)',
'nums.insert(0, 5)    # 頭部插 -> O(n)，後面全部要搬',
'nums.pop(0)          # 頭部刪 -> O(n)，同樣要搬',
'',
'30 in nums           # 成員檢查 -> O(n)，一格一格比'];
const CODE_TUPLE = [
't = (10, 20)         # 建立後就不能改',
't[0] = 99            # TypeError!',
'',
'seen = {}',
'seen[(3, 4)] = "座標"   # tuple 可以當 key（可雜湊）',
'seen[[3, 4]] = "座標"   # TypeError: unhashable type: list',
'',
'x, y = t             # 解包，函式回傳多個值就靠它'];
const CODE_DICT = [
'ages = {"ann": 31, "bob": 24}',
'',
'ages["cat"] = 28     # 插入 -> 平均 O(1)',
'ages["bob"]          # 查詢 -> 平均 O(1)，不用掃過其他 key',
'"bob" in ages        # 成員檢查 -> O(1)',
'',
'# 對照組：在 list 裡找同樣的東西是 O(n)'];
const CODE_SET = [
'a = {1, 2, 3, 4}',
'b = {3, 4, 5}',
'',
'a & b                # 交集 {3, 4}',
'a | b                # 聯集 {1, 2, 3, 4, 5}',
'a - b                # 差集 {1, 2}',
'',
'list(set(data))      # 去重複的一行寫法'];

function buildList(){
  const F = new Frames();
  let a = [10, 20, 30, 40];
  const P = () => [{lbl:{zh:'list', en:'list'}, chips:chipsOf(a, 'act')}];
  F.push({shapes:row(a, 2.6, {}).concat([note(4.5, {zh:'list 是一排連續的位置，索引 = 直接算地址',
    en:'a list is a contiguous run of slots; index = direct address'}, '#3fe0dd')]),
    panels:P(), line:0,
    msg:{zh:'list 的底層是一塊<b>連續</b>的空間。因為連續，<b>a[i] 永遠是 O(1)</b>——' +
            '直接算得出地址，不用走訪。所有的取捨都從「連續」這件事來。',
         en:'A list is one <b>contiguous</b> block. Because it is contiguous, <b>a[i] is always O(1)</b>: the address is computed, not walked. Every trade-off below follows from that.'}});
  F.push({shapes:row(a.concat([50]), 2.6, {4:'hot'}), panels:P(), line:2,
    msg:{zh:'<b>append</b>：尾端本來就留了備用空間，直接寫進去。偶爾空間用完要整批搬家，' +
            '但平均下來是<b>攤還 O(1)</b>。',
         en:'<b>append</b>: there is already spare room at the end, so it just writes. Occasionally the block is full and everything is copied, but averaged out it is <b>amortised O(1)</b>.'}});
  a = a.concat([50]);
  const shifted = {}; a.forEach((_, i) => shifted[i] = 'bad');
  F.push({shapes:row(a, 2.6, shifted).concat([note(4.5, {zh:'insert(0, 5)：每一格都要往右搬一格',
    en:'insert(0, 5): every element shifts one slot right'}, '#ff9736')]), panels:P(), line:3,
    msg:{zh:'<b>insert(0, 5)</b> 就完全不同了：索引 0 已經有人，' +
            '<b>後面 ' + a.length + ' 個元素全部要往右搬一格</b>。這是 <b>O(n)</b>。',
         en:'<b>insert(0, 5)</b> is a different story: slot 0 is occupied, so <b>all ' + a.length +
            ' elements shift one place right</b>. That is <b>O(n)</b>.'}});
  a = [5].concat(a);
  F.push({shapes:row(a, 2.6, {0:'ok'}), panels:P(), line:3,
    msg:{zh:'搬完才輪到新值就位。<b>pop(0)</b> 也一樣要把後面全部往回搬。' +
            '需要頻繁在頭部進出，就該換 <b>deque</b>。',
         en:'Only after the shuffle does the new value land. <b>pop(0)</b> pays the same price in reverse. If you push and pop at the front a lot, reach for a <b>deque</b>.'}});
  const key = 30;
  for (let i = 0; i < a.length; i++){
    const st = {}; for (let j = 0; j < i; j++) st[j] = 'done'; st[i] = a[i] === key ? 'ok' : 'hot';
    F.push({shapes:row(a, 2.6, st), panels:P(), line:6,
      msg:a[i] === key
        ? {zh:'第 ' + (i + 1) + ' 次比對才找到 <b>30</b>。<b>「x in list」是 O(n)</b>——' +
              '在迴圈裡這樣寫，複雜度會直接乘上去。',
           en:'Found <b>30</b> on comparison ' + (i + 1) + '. <b>"x in list" is O(n)</b>, and writing that inside a loop multiplies your complexity.'}
        : {zh:'<b>30 in nums</b>：list 不知道自己裝了什麼，只能一格一格比（現在是 ' + a[i] + '）。',
           en:'<b>30 in nums</b>: a list has no idea what it contains, so it compares slot by slot (currently ' + a[i] + ').'}});
    if (a[i] === key) break;
  }
  F.push({shapes:row(a, 2.6, {}).concat([note(4.5, {zh:'要頻繁查「有沒有」→ 換 set 或 dict',
    en:'checking membership a lot? use a set or dict'}, '#c7a6ff')]), panels:P(), line:6,
    msg:{zh:'記住這張成本表：<b>索引 O(1)、尾端 append 攤還 O(1)、頭部操作 O(n)、成員檢查 O(n)</b>。' +
            '最後一項是新手效能問題最常見的來源。',
         en:'Remember the cost table: <b>indexing O(1), append amortised O(1), front operations O(n), membership O(n)</b>. That last one is the most common performance bug of all.'}});
  return F.list;
}

function buildTuple(){
  const F = new Frames(), t = [10, 20];
  const P = extra => { const p = [{lbl:{zh:'tuple', en:'tuple'}, chips:chipsOf(t, 'ok')}];
    if (extra) p.push(extra); return p; };
  F.push({shapes:row(t, 2.6, {0:'ok', 1:'ok'}).concat([note(4.4, {zh:'tuple = 建好之後不能改',
    en:'tuple = fixed once created'}, '#3fe0dd')]), panels:P(), line:0,
    msg:{zh:'tuple 跟 list 長得很像，差別只有一個字：<b>不可變</b>。' +
            '但這個差別會一路影響到能不能當 key、能不能安全共用。',
         en:'A tuple looks like a list; the difference is one word: <b>immutable</b>. That single word decides whether it can be a dict key and whether it is safe to share.'}});
  F.push({shapes:row(t, 2.6, {0:'bad'}).concat([note(4.4, 'TypeError: object does not support item assignment', '#ff5c5c')]),
    panels:P(), line:1,
    msg:{zh:'<b>t[0] = 99</b> 直接丟 TypeError。不可變不是限制你，而是<b>給你保證</b>：' +
            '傳出去的 tuple 不會被別人偷偷改掉。',
         en:'<b>t[0] = 99</b> raises TypeError. Immutability is not a restriction so much as a <b>guarantee</b>: a tuple you hand out cannot be modified behind your back.'}});
  F.push({shapes:[S.r(1.4, 2.2, 2.4, CH, 'ok', '(3, 4)', {fs:.40}),
                  S.e(3.95, 2.2 + CH / 2, 5.6, 2.2 + CH / 2, {s:'ok'}),
                  S.t(4.8, 1.95, 'hash()', {c:'#3fe0dd', fs:.30}),
                  S.r(5.8, 2.2, 2.6, CH, 'ok', '#8a3f...', {fs:.36}),
                  S.r(1.4, 3.9, 2.4, CH, 'bad', '[3, 4]', {fs:.40}),
                  S.e(3.95, 3.9 + CH / 2, 5.6, 3.9 + CH / 2, {s:'bad'}),
                  S.t(4.8, 3.65, 'hash()', {c:'#ff5c5c', fs:.30}),
                  S.t(6.9, 4.42, 'TypeError: unhashable', {c:'#ff5c5c', fs:.34})],
    panels:P(), line:4,
    msg:{zh:'關鍵差別在這裡：<b>tuple 可以雜湊，list 不行</b>。' +
            '因為 dict 的 key 一旦算出雜湊值就會被放進某個位置，' +
            '如果 key 之後還能被改，那個位置就<b>永遠找不回來</b>了。',
         en:'Here is the crux: <b>a tuple is hashable, a list is not</b>. A dict places a key by its hash; if the key could change afterwards, the entry would be <b>lost forever</b> in the wrong slot.'}});
  F.push({shapes:[S.r(2.0, 2.4, 2.2, CH, 'act', '(3, 4)', {fs:.40, top:{zh:'座標當 key', en:'coordinate key'}}),
                  S.e(4.35, 2.4 + CH / 2, 5.3, 2.4 + CH / 2, {s:'act'}),
                  S.r(5.5, 2.4, 2.6, CH, 'ok', '"座標"', {fs:.36})],
    panels:P({lbl:{zh:'常見用途', en:'typical uses'},
      chips:[{t:'grid[(r, c)]', cls:'ok'}, {t:'return a, b', cls:'ok'}, {t:'@lru_cache', cls:'ok'}]}), line:5,
    msg:{zh:'所以「用座標當 key」「函式回傳多個值」「memo 快取的參數」都是 tuple 的地盤。' +
            '需要不停修改的資料才用 list——<b>選擇的依據是會不會變，不是喜好</b>。',
         en:'So grid coordinates as keys, multiple return values, and memo cache keys are all tuple territory. Use a list when the data must change - <b>the criterion is mutability, not taste</b>.'}});
  return F.list;
}

function buildDict(){
  const F = new Frames();
  const keys = ['ann', 'bob', 'cat'], vals = [31, 24, 28];
  const SL = 8, slotY = 3.9, sw = 1.0, sx = 5.0 - SL * sw / 2;
  const hash = k => ({ann:1, bob:5, cat:3})[k];
  const slots = new Array(SL).fill(null);
  const drawSlots = states => {
    const out = [];
    for (let i = 0; i < SL; i++)
      out.push(S.r(sx + i * sw, slotY, sw - .08, CH, states[i] || (slots[i] ? 'idle' : 'ghost'),
        slots[i] ? slots[i][0] : '', {fs:.34, sub:slots[i] ? String(slots[i][1]) : null}),
        S.t(sx + i * sw + (sw - .08) / 2, slotY - .22, String(i), {c:'#8fa3ac', fs:.24}));
    return out;
  };
  const P = () => [{lbl:{zh:'dict 內容', en:'dict contents'},
    chips:slots.filter(Boolean).map(e => ({t:e[0] + ': ' + e[1], cls:'ok'}))}];
  F.push({shapes:drawSlots({}).concat([note(1.4, {zh:'dict 背後是一排「桶子」',
    en:'a dict is backed by a row of buckets'}, '#3fe0dd')]), panels:P(), line:0,
    msg:{zh:'dict 不是一串鍵值對，而是<b>一排桶子 + 一個把 key 變成位置的函式</b>。' +
            '這一天先看行為，之後兩天會把這排桶子拆開來看。',
         en:'A dict is not a list of pairs: it is <b>a row of buckets plus a function turning a key into a position</b>. Today we look at behaviour; the next two days open the buckets up.'}});
  keys.forEach((k, i) => {
    const h = hash(k);
    F.push({shapes:drawSlots({[h]:'hot'}).concat([
      S.r(1.0, 1.5, 1.9, CH, 'act', '"' + k + '"', {fs:.36}),
      S.e(3.0, 1.5 + CH / 2, 4.4, 1.5 + CH / 2, {s:'act'}),
      S.t(3.7, 1.28, 'hash()', {c:'#c7a6ff', fs:.28}),
      S.r(4.6, 1.5, 2.3, CH, 'hot', '→ ' + h, {fs:.36})]), panels:P(), line:2,
      msg:{zh:'放 <b>' + k + '</b>：先算 <b>hash("' + k + '")</b>，取餘數得到位置 <b>' + h +
              '</b>，然後<b>直接跳過去</b>——不需要比較其他任何 key。',
           en:'Storing <b>' + k + '</b>: compute <b>hash("' + k + '")</b>, reduce it to slot <b>' + h +
              '</b> and <b>jump straight there</b> - no other key is ever compared.'}});
    slots[h] = [k, vals[i]];
    F.push({shapes:drawSlots({[h]:'ok'}), panels:P(), line:2,
      msg:{zh:'寫進桶子 ' + h + '。<b>字典的插入與查詢平均都是 O(1)</b>，' +
              '而且跟裡面有多少筆資料無關。',
           en:'Written into bucket ' + h + '. <b>Insert and lookup are both O(1) on average</b>, regardless of how much is already stored.'}});
  });
  F.push({shapes:drawSlots({5:'ok'}).concat([
    S.r(1.0, 1.5, 1.9, CH, 'act', '"bob"', {fs:.36}),
    S.e(3.0, 1.5 + CH / 2, 4.4, 1.5 + CH / 2, {s:'act'}),
    S.r(4.6, 1.5, 2.3, CH, 'ok', '→ 5', {fs:.36})]), panels:P(), line:3,
    msg:{zh:'查 <b>ages["bob"]</b>：同一個雜湊函式算出 <b>5</b>，一步到位。' +
            '對照 list 的成員檢查要一格一格比——這就是<b>用空間換時間</b>。',
         en:'Looking up <b>ages["bob"]</b>: the same hash gives <b>5</b>, one jump. Compare that with scanning a list slot by slot - this is <b>space traded for time</b>.'}});
  F.push({shapes:drawSlots({}).concat([note(1.4, {zh:'Python 3.7 起：走訪順序 = 插入順序',
    en:'since Python 3.7: iteration follows insertion order'}, '#c7a6ff')]), panels:P(), line:6,
    msg:{zh:'兩個常被誤會的點：<b>dict 從 Python 3.7 起保證走訪順序等於插入順序</b>（但它仍然不是排序），' +
            '而且 O(1) 是<b>平均</b>——碰撞很多時會退化，這正是接下來兩天要處理的問題。',
         en:'Two things people get wrong: <b>since Python 3.7 iteration order equals insertion order</b> (which is still not sorted), and O(1) is an <b>average</b> - heavy collisions degrade it, which is exactly what the next two days are about.'}});
  return F.list;
}

function buildSet(){
  const F = new Frames();
  const data = [3, 1, 4, 1, 5, 3, 4];
  const seen = [];
  const P = () => [{lbl:{zh:'set 內容', en:'set contents'}, chips:chipsOf(seen, 'ok')}];
  F.push({shapes:row(data, 1.6, {}).concat([note(3.0, {zh:'原始資料（有重複）',
    en:'raw data (with duplicates)'})]), panels:P(), line:0,
    msg:{zh:'set 就是<b>只有 key 沒有 value 的 dict</b>：同樣用雜湊，所以「有沒有出現過」是 O(1)。',
         en:'A set is <b>a dict with keys but no values</b>: same hashing, so "have I seen this?" costs O(1).'}});
  data.forEach((v, i) => {
    const dup = seen.indexOf(v) >= 0;
    const st = {}; for (let j = 0; j < i; j++) st[j] = 'done'; st[i] = dup ? 'bad' : 'hot';
    F.push({shapes:row(data, 1.6, st).concat(row(seen.concat(dup ? [] : [v]), 3.6,
      dup ? {} : {[seen.length]:'ok'}, {index:false}))
      .concat([note(5.2, {zh:'set（自動去重複）', en:'set (deduplicated)'})]),
      panels:P(), line:0,
      msg:dup ? {zh:'<b>' + v + '</b> 已經在裡面了 → 直接忽略。判斷「在不在」只花 O(1)。',
                 en:'<b>' + v + '</b> is already there, so it is ignored. The membership test itself was O(1).'}
              : {zh:'<b>' + v + '</b> 沒看過 → 加進去。',
                 en:'<b>' + v + '</b> is new, so it goes in.'}});
    if (!dup) seen.push(v);
  });
  const A = [1, 2, 3, 4], B = [3, 4, 5];
  const setBox = (x, vals, s, lbl) => {
    const out = [S.t(x + 1.1, 1.35, lbl, {c:'#3fe0dd', fs:.34})];
    vals.forEach((v, i) => out.push(S.c(x + .45 + i * .78, 2.1, .32, s, String(v), {fs:.34})));
    return out;
  };
  const ops = [
    ['&', [3, 4], {zh:'交集：兩邊都有的', en:'intersection: in both'},
     {zh:'找共同好友、找重複的檔案，都是這一行。', en:'mutual friends, duplicate files - all one line.'}],
    ['|', [1, 2, 3, 4, 5], {zh:'聯集：任一邊有的', en:'union: in either'},
     {zh:'合併兩批資料又不想有重複的時候用。', en:'merging two batches without duplicates.'}],
    ['-', [1, 2], {zh:'差集：只有 A 有的', en:'difference: only in A'},
     {zh:'「上次有、這次沒有」——比對前後差異的標準做法。',
      en:'"present before, gone now" - the standard way to diff two snapshots.'}]];
  ops.forEach(([op, res, name, why], k) => {
    F.push({shapes:setBox(.6, A, 'act', 'A = {1,2,3,4}').concat(setBox(5.8, B, 'hot', 'B = {3,4,5}'),
      [S.t(5.0, 3.9, 'A ' + op + ' B', {c:'#ffbe6b', fs:.46})],
      res.map((v, i) => S.c(3.4 + i * .8, 4.9, .32, 'ok', String(v), {fs:.34})),
      [S.t(5.0, 5.9, name, {c:'#8fa3ac', fs:.32})]),
      panels:[{lbl:{zh:'結果', en:'result'}, chips:chipsOf(res, 'ok')}], line:3 + k,
      msg:{zh:'<b>A ' + op + ' B = {' + res.join(', ') + '}</b>。' + tr(why) +
              '用 set 運算取代雙層迴圈，通常是從 O(n·m) 直接掉到 O(n+m)。',
           en:'<b>A ' + op + ' B = {' + res.join(', ') + '}</b>. ' + tr(why) +
              ' Replacing a nested loop with a set operation usually turns O(n·m) into O(n+m).'}});
  });
  F.push({shapes:[note(2.6, {zh:'挑容器的順序：要不要改？要不要順序？要不要查得快？',
    en:'choosing a container: mutable? ordered? fast lookup?'}, '#3fe0dd'),
    note(3.6, {zh:'list = 順序 / tuple = 固定 + 可當 key / dict = 對應 / set = 成員與去重',
      en:'list = order / tuple = fixed + hashable / dict = mapping / set = membership'}, '#c7a6ff')],
    panels:[], line:7,
    msg:{zh:'四個容器不是四種寫法，是四種成本結構。<b>在寫迴圈之前先問自己選對了沒有</b>——' +
            '很多「演算法優化」其實只是把 list 換成 set。',
         en:'These are four cost models, not four syntaxes. <b>Ask whether the container is right before optimising the loop</b> - a great deal of "algorithm tuning" is really just a list becoming a set.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'list、tuple、dict 與 set', en:'list, tuple, dict and set'},
  sub:{zh:'四個內建容器不是四種語法，是四種不同的成本結構——挑錯了，程式就會慢一個級數。',
       en:'Four built-in containers are four cost models, not four syntaxes - pick the wrong one and you lose a whole order of growth.'},
  tabs:[
    {id:'list', label:{zh:'list', en:'list'},
     stage:{zh:'連續空間帶來的好處與代價', en:'What contiguity buys and costs'},
     idea:{zh:'連續配置讓索引變成一次位址計算（O(1)），但也讓<b>頭部插入與刪除必須整排搬家</b>（O(n)）。' +
              '成員檢查同樣是 O(n)——這是實務上最常見的效能陷阱。',
           en:'Contiguity makes indexing a single address computation (O(1)) but forces <b>front insertions and deletions to shuffle the whole block</b> (O(n)). Membership is O(n) too - the most common performance trap in practice.'},
     legend:['hot', 'ok', 'bad', 'done'], code:CODE_LIST, build:buildList},
    {id:'tuple', label:{zh:'tuple', en:'tuple'},
     stage:{zh:'不可變換來的能力', en:'What immutability buys'},
     idea:{zh:'tuple 的重點不是「唯讀的 list」，而是<b>可雜湊</b>：能當 dict 的 key、能放進 set。' +
              '會變的東西不能當 key，因為位置一旦算好就不能再變。',
           en:'A tuple is not "a read-only list"; it is <b>hashable</b>, so it can be a dict key or a set member. Anything mutable cannot be a key, because its slot is decided once and must stay valid.'},
     legend:['ok', 'act', 'bad'], code:CODE_TUPLE, build:buildTuple},
    {id:'dict', label:{zh:'dict', en:'dict'},
     stage:{zh:'key → 位置，一步到位', en:'key to slot in one jump'},
     idea:{zh:'dict 用<b>雜湊函式把 key 直接換算成位置</b>，所以查詢跟資料量無關，平均 O(1)。' +
              '代價是額外的空間與「平均」兩個字——碰撞多的時候會退化。',
           en:'A dict <b>turns a key straight into a slot</b>, so lookup is independent of size: O(1) on average. The price is extra memory and that word "average" - collisions degrade it.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_DICT, build:buildDict},
    {id:'set', label:{zh:'set', en:'set'},
     stage:{zh:'去重複與集合運算', en:'Dedup and set algebra'},
     idea:{zh:'set 是沒有 value 的 dict。除了 O(1) 的成員檢查，' +
              '<b>交集、聯集、差集可以取代雙層迴圈</b>，常常直接把 O(n·m) 變成 O(n+m)。',
           en:'A set is a dict without values. Beyond O(1) membership, <b>intersection, union and difference replace nested loops</b>, often turning O(n·m) into O(n+m).'},
     legend:['hot', 'ok', 'bad', 'done'], code:CODE_SET, build:buildSet}
  ]
};
