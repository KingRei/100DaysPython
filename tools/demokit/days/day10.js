// DAY: 10
// TITLE_ZH: 雜湊表：開放定址法
// TITLE_EN: Hash Table - Open Addressing
// SUB_ZH: 不掛串列，撞到就往旁邊找空位。速度很快、快取很友善，代價是聚集與那個惱人的墓碑。
// SUB_EN: No chains - on collision, walk to the next free slot.
// FOLDER: day%2010%20-%20hash%20table%20-%20open%20addressing
// MEDIUM: https://medium.com/100-days-of-python/day-10-data-structure-hash-table-open-addressing-dc335fd3ae0

function hsum(k){ let s = 0; for (let i = 0; i < k.length; i++) s += k.charCodeAt(i); return s; }
const CAP = 7, SW = 1.24, SH = 1.0, SY = 2.35;
const sx = i => 5.0 - CAP * SW / 2 + i * SW;
const TOMB = '__T__';
function slotShapes(slots, o){
  o = o || {}; const out = [];
  for (let i = 0; i < CAP; i++){
    const e = slots[i];
    const style = (o.st && o.st[i]) || (e === TOMB ? 'bad' : (e ? 'idle' : 'ghost'));
    out.push(S.r(sx(i), SY, SW - .10, SH, style,
      e === TOMB ? '☠' : (e ? e[0] : ''), {fs:e === TOMB ? .50 : .34,
      sub:e && e !== TOMB ? String(e[1]) : null}));
    out.push(S.t(sx(i) + (SW - .10) / 2, SY + SH + .40, String(i), {c:'#8fa3ac', fs:.28}));
  }
  if (o.home != null)
    out.push(S.t(sx(o.home) + (SW - .10) / 2, SY - .38, {zh:'原本該在這', en:'home slot'},
      {c:'#c7a6ff', fs:.26}));
  if (o.probe != null)
    out.push(S.e(sx(o.probe) + (SW - .10) / 2, SY - 1.30, sx(o.probe) + (SW - .10) / 2, SY - .18,
      {s:'hot', w:.06}));
  if (o.key)
    out.push(S.r(.35, .55, 1.9, .78, 'act', '"' + o.key + '"', {fs:.34}),
             S.t(3.6, 1.05, o.note || '', {c:'#ffbe6b', fs:.32, anchor:'start'}));
  return out;
}
const filled = slots => slots.filter(e => e && e !== TOMB).length;
const statPanel = slots => ({lbl:{zh:'表格狀態', en:'table state'},
  chips:[{t:'n = ' + filled(slots), cls:''}, {t:'cap = ' + CAP, cls:''},
         {t:'a = ' + (filled(slots) / CAP).toFixed(2), cls:filled(slots) / CAP > .7 ? 'bad' : 'ok'}]});

const CODE_PUT = [
'def put(self, key, value):',
'    i = hash(key) % self.cap',
'    while self.slots[i] is not None:      # 這格有人',
'        if self.slots[i].key == key:',
'            self.slots[i].value = value   # 同一個 key -> 更新',
'            return',
'        i = (i + 1) % self.cap            # 線性探測：往右一格',
'    self.slots[i] = Entry(key, value)     # 找到空位'];
const CODE_GET = [
'def get(self, key):',
'    i = hash(key) % self.cap',
'    while self.slots[i] is not None:',
'        if self.slots[i].key == key:',
'            return self.slots[i].value',
'        i = (i + 1) % self.cap',
'    raise KeyError(key)      # 碰到「真正的空格」才能宣告找不到'];
const CODE_DEL = [
'def remove(self, key):',
'    i = self._find(key)',
'    self.slots[i] = DELETED   # 墓碑，不能直接設成 None！',
'    # 直接清空會切斷探測序列，',
'    # 後面的 key 就永遠找不到了',
'',
'# 查詢時：DELETED 要當作「有人」繼續往下探測',
'# 插入時：DELETED 可以覆蓋，當成空位用'];

const DATA = [['ann', 31], ['bob', 24], ['ida', 17], ['dan', 40]];

function buildPut(){
  const slots = new Array(CAP).fill(null), F = new Frames();
  F.push({shapes:slotShapes(slots), panels:[statPanel(slots)], line:0,
    msg:{zh:'開放定址法把資料<b>全部放在同一個陣列裡</b>，沒有額外的串列、沒有指標。' +
            '好處是記憶體連續、對 CPU 快取很友善；代價馬上就會看到。',
         en:'Open addressing keeps <b>everything inside one array</b>: no side lists, no pointers. The upside is contiguous memory and cache friendliness; the cost shows up immediately.'}});
  DATA.forEach(([k, v]) => {
    const home = hsum(k) % CAP;
    let i = home, hops = 0;
    F.push({shapes:slotShapes(slots, {home:home, probe:home, key:k,
      note:'hash % ' + CAP + ' = ' + home}), panels:[statPanel(slots)], line:1,
      msg:{zh:'放 <b>' + k + '</b>：算出它的家在 <b>' + home + '</b>。',
           en:'Insert <b>' + k + '</b>: its home slot is <b>' + home + '</b>.'}});
    while (slots[i]){
      hops++;
      F.push({shapes:slotShapes(slots, {home:home, probe:i, st:{[i]:'hot'}, key:k,
        note:(LANG === 'zh' ? '第 ' + hops + ' 次探測' : 'probe ' + hops)}),
        panels:[statPanel(slots)], line:6,
        msg:{zh:'第 ' + i + ' 格被 <b>' + slots[i][0] + '</b> 佔了 → <b>線性探測</b>：往右一格 (i+1) % ' +
                CAP + '。沒有串列可以掛，只能借用別人的位置。',
             en:'Slot ' + i + ' is taken by <b>' + slots[i][0] + '</b>, so <b>probe linearly</b>: (i+1) % ' +
                CAP + '. There is no chain to hang on - we must borrow someone else\'s slot.'}});
      i = (i + 1) % CAP;
    }
    slots[i] = [k, v];
    F.push({shapes:slotShapes(slots, {st:{[i]:'ok'}, home:home}), panels:[statPanel(slots)], line:7,
      msg:hops ? {zh:'<b>' + k + '</b> 最後住進第 ' + i + ' 格（探測了 ' + hops + ' 次）。' +
                     '<b>它不在自己的家，這件事之後查詢與刪除都得記得。</b>',
                  en:'<b>' + k + '</b> ends up in slot ' + i + ' after ' + hops +
                     ' probes. <b>It is not in its home slot - lookup and deletion must both account for that.</b>'}
                : {zh:'第 ' + i + ' 格是空的，直接住進去。',
                   en:'Slot ' + i + ' was free, so it moves straight in.'}});
  });
  F.push({shapes:slotShapes(slots, {st:{1:'hot', 2:'hot', 3:'hot'}}), panels:[statPanel(slots)], line:6,
    msg:{zh:'注意連在一起的那一段：<b>聚集（clustering）</b>。' +
            '一旦連成一片，任何雜湊到這段裡的 key 都得走完整段，' +
            '而且<b>聚集會自我加速</b>——愈大愈容易被撞進來。α 超過 0.7 之後就會很明顯。',
         en:'Look at the run of adjacent entries: <b>clustering</b>. Any key hashing into that run must walk it all, and <b>clusters grow on themselves</b> - the bigger they are, the more likely the next key lands in them. Past a = 0.7 it becomes very visible.'}});
  return F.list;
}

function buildGet(v){
  const slots = new Array(CAP).fill(null);
  DATA.forEach(([k, val]) => { let i = hsum(k) % CAP; while (slots[i]) i = (i + 1) % CAP; slots[i] = [k, val]; });
  const key = v === 0 ? 'dan' : 'zoe';
  const F = new Frames(), home = hsum(key) % CAP;
  F.push({shapes:slotShapes(slots, {home:home, key:key, note:'hash % ' + CAP + ' = ' + home}),
    panels:[statPanel(slots)], line:1,
    msg:{zh:'查 <b>' + key + '</b>：先跳到它的家 <b>' + home + '</b>。' +
            '但因為插入時可能被擠開，<b>沒看到不代表沒有</b>。',
         en:'Look up <b>' + key + '</b>: jump to its home slot <b>' + home +
            '</b>. But since inserts can displace, <b>not seeing it here does not mean it is absent</b>.'}});
  let i = home, hops = 0;
  while (slots[i]){
    hops++;
    const hit = slots[i][0] === key;
    F.push({shapes:slotShapes(slots, {probe:i, st:{[i]:hit ? 'ok' : 'hot'}, home:home, key:key}),
      panels:[statPanel(slots)], line:3,
      msg:{zh:'第 ' + i + ' 格是 <b>' + slots[i][0] + '</b>' +
              (hit ? ' —— 找到了（探測 ' + hops + ' 次）。' : '，不是我要的，往右一格繼續。'),
           en:'Slot ' + i + ' holds <b>' + slots[i][0] + '</b>' +
              (hit ? ' - found it after ' + hops + ' probes.' : ', not ours; step right and continue.')}});
    if (hit) return F.list;
    i = (i + 1) % CAP;
  }
  F.push({shapes:slotShapes(slots, {probe:i, st:{[i]:'act'}, home:home, key:key}),
    panels:[statPanel(slots)], line:6,
    msg:{zh:'碰到<b>真正的空格</b>（第 ' + i + ' 格）→ 可以確定 <b>' + key + '</b> 不在表裡。' +
            '這裡有個關鍵前提：<b>空格代表探測序列到此為止</b>。下一頁會看到刪除如何破壞這個前提。',
         en:'We hit a <b>genuinely empty slot</b> (' + i + '), so <b>' + key +
            '</b> is definitely absent. This relies on one assumption - <b>an empty slot terminates the probe sequence</b> - and deletion is about to break it.'}});
  return F.list;
}

function buildDel(v){
  const slots = new Array(CAP).fill(null);
  DATA.forEach(([k, val]) => { let i = hsum(k) % CAP; while (slots[i]) i = (i + 1) % CAP; slots[i] = [k, val]; });
  const F = new Frames();
  // find a displaced key: one that is not at its home
  let victimIdx = -1, displacedIdx = -1;
  for (let i = 0; i < CAP; i++)
    if (slots[i] && (hsum(slots[i][0]) % CAP) !== i){ displacedIdx = i; break; }
  if (displacedIdx >= 0) victimIdx = hsum(slots[displacedIdx][0]) % CAP;
  const victim = slots[victimIdx], displaced = slots[displacedIdx];
  F.push({shapes:slotShapes(slots, {st:{[victimIdx]:'act', [displacedIdx]:'act'}}),
    panels:[statPanel(slots)], line:0,
    msg:{zh:'<b>' + displaced[0] + '</b> 當初被 <b>' + victim[0] + '</b> 擠到第 ' + displacedIdx +
            ' 格，它的家其實是 ' + victimIdx + '。現在如果把 <b>' + victim[0] + '</b> 刪掉，會發生什麼事？',
         en:'<b>' + displaced[0] + '</b> was displaced by <b>' + victim[0] + '</b> into slot ' +
            displacedIdx + '; its home is really ' + victimIdx + '. So what happens if we delete <b>' +
            victim[0] + '</b>?'}});
  if (v === 0){
    const naive = slots.slice(); naive[victimIdx] = null;
    F.push({shapes:slotShapes(naive, {st:{[victimIdx]:'bad'}}), panels:[statPanel(naive)], line:2,
      msg:{zh:'<b>錯誤做法</b>：直接把第 ' + victimIdx + ' 格設成 None。看起來很合理。',
           en:'<b>The wrong way</b>: set slot ' + victimIdx + ' to None. Looks perfectly reasonable.'}});
    F.push({shapes:slotShapes(naive, {probe:victimIdx, st:{[victimIdx]:'bad'},
      key:displaced[0], note:'hash % ' + CAP + ' = ' + victimIdx}), panels:[statPanel(naive)], line:3,
      msg:{zh:'現在去查 <b>' + displaced[0] + '</b>：跳到家 ' + victimIdx +
              ' → <b>是空的</b> → 查詢直接宣告「不存在」。',
           en:'Now look up <b>' + displaced[0] + '</b>: jump to home ' + victimIdx +
              ' - <b>empty</b> - so the search declares it missing.'}});
    F.push({shapes:slotShapes(naive, {st:{[victimIdx]:'bad', [displacedIdx]:'bad'}}),
      panels:[statPanel(naive)], line:4,
      msg:{zh:'但 <b>' + displaced[0] + '</b> 明明還在第 ' + displacedIdx + ' 格！' +
              '<b>刪掉一個 key，弄丟了另一個 key</b>——探測序列被切斷了。' +
              '這是開放定址法最經典的坑。',
           en:'But <b>' + displaced[0] + '</b> is still sitting in slot ' + displacedIdx +
              '! <b>Deleting one key lost another</b>, because the probe sequence was cut in half. This is the classic open-addressing trap.'}});
  } else {
    const tomb = slots.slice(); tomb[victimIdx] = TOMB;
    F.push({shapes:slotShapes(tomb, {st:{[victimIdx]:'bad'}}), panels:[statPanel(tomb)], line:2,
      msg:{zh:'<b>正確做法</b>：放一個<b>墓碑</b>（DELETED）。它不是資料，也不是空格，' +
              '而是「這裡曾經有人，請繼續往下找」。',
           en:'<b>The right way</b>: leave a <b>tombstone</b> (DELETED). It is neither data nor emptiness; it means "someone lived here, keep probing".'}});
    F.push({shapes:slotShapes(tomb, {probe:victimIdx, st:{[victimIdx]:'act'}, key:displaced[0],
      note:'hash % ' + CAP + ' = ' + victimIdx}), panels:[statPanel(tomb)], line:6,
      msg:{zh:'查 <b>' + displaced[0] + '</b>：家是 ' + victimIdx + '，看到墓碑 → <b>當作有人，繼續往右</b>。',
           en:'Looking up <b>' + displaced[0] + '</b>: home is ' + victimIdx +
              ', we see a tombstone and <b>treat it as occupied, so we keep going</b>.'}});
    F.push({shapes:slotShapes(tomb, {probe:displacedIdx, st:{[displacedIdx]:'ok'}}),
      panels:[statPanel(tomb)], line:6,
      msg:{zh:'在第 ' + displacedIdx + ' 格找到 <b>' + displaced[0] + '</b>。探測序列保住了。',
           en:'Found <b>' + displaced[0] + '</b> in slot ' + displacedIdx + '. The probe chain survives.'}});
    const tomb2 = tomb.slice(); tomb2[victimIdx] = ['gus', 35];
    F.push({shapes:slotShapes(tomb2, {st:{[victimIdx]:'ok'}}), panels:[statPanel(tomb2)], line:7,
      msg:{zh:'插入時墓碑<b>可以被覆蓋</b>，所以空間不會浪費。' +
              '但墓碑對查詢來說仍然是「有人」，累積太多會讓探測愈來愈長——' +
              '所以刪除很多之後通常要整張表重建。',
           en:'On insertion a tombstone <b>can be overwritten</b>, so the space is not wasted. But to a lookup it still counts as occupied, so tombstones lengthen probes over time - which is why a delete-heavy table eventually gets rebuilt.'}});
  }
  return F.list;
}

const DAY_META = {
  title:{zh:'雜湊表：開放定址法', en:'Hash Table - Open Addressing'},
  sub:{zh:'不掛串列，撞到就往旁邊找空位。速度很快、快取很友善，代價是聚集與那個惱人的墓碑。',
       en:'No chains: on collision, walk to the next free slot. Fast and cache friendly - at the price of clustering and tombstones.'},
  tabs:[
    {id:'put', label:{zh:'插入與線性探測', en:'insert & linear probing'},
     stage:{zh:'撞到就往右找空位', en:'On collision, step right'}, view:[10, 6.4],
     idea:{zh:'所有資料都在同一個陣列裡，<b>沒有指標、沒有額外配置</b>，走訪時對 CPU 快取非常友善。' +
              '代價是被擠開的 key 不在自己的家，而且相鄰的佔用會<b>聚集</b>成一整段。',
           en:'Everything lives in one array with <b>no pointers and no extra allocation</b>, which is very kind to the CPU cache. The price is displaced keys and runs of occupied slots that <b>cluster</b>.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_PUT, build:buildPut},
    {id:'get', label:{zh:'查詢', en:'lookup'},
     stage:{zh:'沿著探測序列走到空格為止', en:'Follow the probe sequence to the first empty slot'},
     view:[10, 6.4],
     variants:[{zh:'查得到 dan', en:'hit: dan'}, {zh:'查不到 zoe', en:'miss: zoe'}],
     idea:{zh:'查詢必須走完<b>同一條探測序列</b>，而且只有碰到<b>真正的空格</b>才能宣告不存在。' +
              '這個「空格代表結束」的約定，正是刪除必須小心的原因。',
           en:'A lookup must follow the <b>same probe sequence</b>, and only a <b>genuinely empty slot</b> proves absence. That convention - empty means stop - is exactly what makes deletion delicate.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_GET, build:buildGet},
    {id:'del', label:{zh:'刪除與墓碑', en:'delete & tombstones'},
     stage:{zh:'直接清空會弄丟別的 key', en:'Clearing a slot loses other keys'}, view:[10, 6.4],
     variants:[{zh:'錯誤：直接設成 None', en:'wrong: set to None'}, {zh:'正確：放墓碑', en:'right: tombstone'}],
     idea:{zh:'刪除是開放定址法真正的難處：清空一格會<b>切斷探測序列</b>，把後面的 key 一起弄丟。' +
              '解法是墓碑——查詢時算「有人」、插入時算「空位」。' +
              '這也是為什麼 Python 的 dict（開放定址）在大量刪除後會重建整張表。',
           en:'Deletion is where open addressing really bites: clearing a slot <b>cuts the probe sequence</b> and strands the keys behind it. The fix is a tombstone: occupied to a lookup, free to an insert. It is also why Python\'s dict (open addressing) rebuilds itself after heavy deletion.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_DEL, build:buildDel}
  ]
};
