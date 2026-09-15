// DAY: 32
// TITLE_ZH: Day 32 — Bloom Filter、HyperLogLog、Count-Min Sketch：用固定記憶體回答集合問題
// TITLE_EN: Bloom filter, HyperLogLog and Count-Min sketch - answering set questions in kilobytes
// SUB_ZH: 三種機率型結構用固定大小的位元陣列回答「在不在」「有幾個不同的」「出現幾次」。它們都會錯，但錯的方向是設計好的：Bloom 只會多說 yes，HLL 兩邊都可能偏，CMS 只會多算 —— 知道誤差往哪邊偏，比知道誤差多大更重要。
// SUB_EN: Three sketches answer "is it in the set", "how many distinct" and "how often" out of a fixed-size array of bits. They are all wrong sometimes, but the direction of the error is designed in: a Bloom filter only ever says yes too often, HLL can miss either way, and a Count-Min sketch only ever over-counts. Knowing which way the error leans matters more than knowing how big it is.
// FOLDER: day%2032%20-%20probabilistic%20structures
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }

/* ---- one deterministic hash, used by all three sketches ----------------- */
function h32(s, seed){
  let h = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < s.length; i++){
    h = (h ^ s.charCodeAt(i)) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
  }
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
/* two independent hashes, then Kirsch-Mitzenmacher: g_i = h1 + i*h2 + i*i */
function hashPair(s){ return [h32(s, 1), (h32(s, 2) | 1) >>> 0]; }

/* ---- Bloom filter ------------------------------------------------------- */
function Bloom(m, k){ this.m = m; this.k = k; this.bits = new Array(m).fill(0); }
Bloom.prototype.indices = function(item){
  const hp = hashPair(item), out = [];
  for (let i = 0; i < this.k; i++)
    out.push(((hp[0] + i * hp[1] + i * i) >>> 0) % this.m);
  return out;
};
Bloom.prototype.add = function(item){ this.indices(item).forEach(i => this.bits[i] = 1); };
Bloom.prototype.has = function(item){ return this.indices(item).every(i => this.bits[i] === 1); };
Bloom.prototype.setCount = function(){ return this.bits.reduce((a, b) => a + b, 0); };

/* counting Bloom: the same geometry, 4-bit counters instead of single bits */
function Counting(m, k){ this.m = m; this.k = k; this.cnt = new Array(m).fill(0); }
Counting.prototype.indices = Bloom.prototype.indices;
Counting.prototype.add = function(item){
  this.indices(item).forEach(i => { if (this.cnt[i] < 15) this.cnt[i]++; }); };
Counting.prototype.del = function(item){
  this.indices(item).forEach(i => { if (this.cnt[i] > 0) this.cnt[i]--; }); };
Counting.prototype.has = function(item){ return this.indices(item).every(i => this.cnt[i] > 0); };

/* ---- HyperLogLog (tiny p, so the registers fit on screen) --------------- */
function HLL(p){ this.p = p; this.m = 1 << p; this.reg = new Array(this.m).fill(0); }
HLL.prototype.parts = function(item){
  const h = h32(item, 7) >>> 0;
  const idx = h & (this.m - 1);
  const rest = h >>> this.p;                  // the bits the index did not use
  let rho = 1;
  if (rest === 0) rho = 32 - this.p + 1;
  else { let w = rest; while ((w & 1) === 0){ rho++; w >>>= 1; } }
  return {h:h, idx:idx, rest:rest, rho:rho};
};
HLL.prototype.add = function(item){
  const p = this.parts(item);
  const grew = p.rho > this.reg[p.idx];
  if (grew) this.reg[p.idx] = p.rho;
  return Object.assign(p, {grew:grew});
};
HLL.prototype.count = function(){
  const m = this.m;
  const alpha = m === 16 ? 0.673 : m === 32 ? 0.697 : m === 64 ? 0.709
                                            : 0.7213 / (1 + 1.079 / m);
  let inv = 0, zeros = 0;
  for (let i = 0; i < m; i++){ inv += Math.pow(2, -this.reg[i]); if (this.reg[i] === 0) zeros++; }
  let est = alpha * m * m / inv;
  if (est <= 2.5 * m && zeros > 0) est = m * Math.log(m / zeros);   // linear counting
  return est;
};

/* ---- Count-Min sketch --------------------------------------------------- */
function CMS(d, w){
  this.d = d; this.w = w;
  this.rows = []; for (let i = 0; i < d; i++) this.rows.push(new Array(w).fill(0));
}
CMS.prototype.cols = function(item){
  const out = []; for (let i = 0; i < this.d; i++) out.push(h32(item, 31 + i * 17) % this.w);
  return out;
};
CMS.prototype.add = function(item, c){
  const cs = this.cols(item); cs.forEach((c2, i) => this.rows[i][c2] += (c == null ? 1 : c));
  return cs;
};
CMS.prototype.query = function(item){
  const cs = this.cols(item);
  return Math.min.apply(null, cs.map((c, i) => this.rows[i][c]));
};

/* ======================================================= 1. Bloom filter */
const BF_M = 23, BF_K = 3;
const MEMBERS = ['cat', 'dog', 'fox'];
const POOL = ('ant bat bee cow elk owl pig ram rat yak ape eel hen jay koi ' +
              'mole newt orca puma seal toad wolf crab deer frog').split(' ');

const CODE_BLOOM = [
  'def add(self, item):',
  '    h1, h2 = hash_pair(item)',
  '    for i in range(self.k):',
  '        idx = (h1 + i*h2 + i*i) % self.m   # double hashing',
  '        self.bits[idx] = 1',
  '',
  'def __contains__(self, item):',
  '    h1, h2 = hash_pair(item)',
  '    for i in range(self.k):',
  '        idx = (h1 + i*h2 + i*i) % self.m',
  '        if self.bits[idx] == 0:',
  '            return False   # a 0 proves it was never added',
  '    return True            # all ones - probably added',
];

const BX0 = 0.32, BCW = 0.40, BCY = 3.10, BCH = 0.55;
const bcx = i => BX0 + i * BCW + (BCW - .06) / 2;

function bitRow(bits, st){
  return cellRow(bits.map(b => b ? '1' : '0'), BX0, BCY, BCW, BCH,
                 {states:st, fs:.30, ifs:.24, ilift:.30});
}
function itemHead(word, idxs, hot, verdict){
  const out = [S.r(0.35, 0.55, 1.5, 0.70, hot || 'act', word, {fs:.40})];
  idxs.forEach((ix, i) => {
    const x = 2.55 + i * 1.75;
    const on = (hot === 'bad' && i === idxs.length - 1) ? 'bad' : (hot || 'act');
    out.push(S.r(x, 0.55, 1.45, 0.70, on, 'h' + i + ' = ' + ix, {fs:.32}));
    out.push(S.e(1.85, 0.90, x - .05, 0.90, {w:.04, s:'soft'}));
    out.push(S.e(x + .72, 1.25, bcx(ix), BCY - .06, {w:.055, s:on}));
  });
  if (verdict) out.push(S.t(4.9, 5.10, verdict.s, {c:verdict.c, fs:.42}));
  return out;
}

function bloomFrames(){
  const bf = new Bloom(BF_M, BF_K), F = new Frames();
  const done = {};
  const panels = (extra) => ([
    {lbl:{zh:'位元陣列 m', en:'bits m'}, chips:[chip(String(BF_M), '')]},
    {lbl:{zh:'雜湊個數 k', en:'hashes k'}, chips:[chip(String(BF_K), '')]},
    {lbl:{zh:'已設為 1 的位元', en:'bits set to 1'},
     chips:[chip(bf.setCount() + ' / ' + BF_M, 'ok')]},
    {lbl:{zh:'真正存進去的字', en:'words actually added'},
     chips:MEMBERS.map(w => chip(w, bf.has(w) ? 'ok' : ''))}
  ]).concat(extra || []);

  F.push({shapes:bitRow(bf.bits, {}), panels:panels(), view:VIEW, line:0,
          msg:{zh:'23 個位元，全是 0。Bloom filter 不存字串本身 —— 它只存「某幾個位元被打開了」這件事，所以不管塞多少字，記憶體都不會長大。',
               en:'Twenty-three bits, all zero. A Bloom filter never stores the strings themselves - it only records that certain bits got switched on, which is why the memory never grows no matter how much you add.'}});

  MEMBERS.forEach(w => {
    const ix = bf.indices(w);
    bf.add(w); ix.forEach(i => done[i] = 'ok');
    F.push({shapes:itemHead(w, ix, 'hot').concat(bitRow(bf.bits, Object.assign({}, done,
              Object.fromEntries(ix.map(i => [i, 'hot']))))),
            panels:panels(), view:VIEW, line:4,
            msg:{zh:'加入 "' + w + '"：k=3 個雜湊把它變成 3 個位置 ' + ix.join('、') +
                     '，把這三格設成 1。設過 1 的格子永遠不會變回 0，這就是為什麼 Bloom filter 不能刪東西。',
                 en:'Adding "' + w + '": the k=3 hashes turn it into the three positions ' +
                     ix.join(', ') + ', and those three bits go to 1. A bit that is 1 never goes back to 0 - that is exactly why a plain Bloom filter cannot delete.'}});
  });

  const hit = 'dog', hix = bf.indices(hit);
  F.push({shapes:itemHead(hit, hix, 'ok').concat(bitRow(bf.bits,
            Object.assign({}, done, Object.fromEntries(hix.map(i => [i, 'ok']))))),
          panels:panels(), view:VIEW, line:12,
          msg:{zh:'查詢 "' + hit + '"：同樣三個雜湊算出 ' + hix.join('、') +
                   '，三格都是 1，回答 yes。真的加過的東西一定會被找到 —— Bloom filter 沒有 false negative。',
               en:'Querying "' + hit + '": the same three hashes give ' + hix.join(', ') +
                   ', all three bits are 1, so the answer is yes. Anything genuinely added is always found - a Bloom filter has no false negatives.'}});

  const miss = POOL.filter(w => !bf.has(w) && new Set(bf.indices(w)).size === BF_K)[0];
  const mix = bf.indices(miss), zero = mix.filter(i => bf.bits[i] === 0)[0];
  F.push({shapes:itemHead(miss, mix, 'act').concat(bitRow(bf.bits,
            Object.assign({}, done, {[zero]:'bad'}))),
          panels:panels(), view:VIEW, line:11,
          msg:{zh:'查詢 "' + miss + '"：位置 ' + zero +
                   ' 是 0。只要有任何一格是 0，就代表這個字絕對沒被加過 —— no 是「證明」，可以放心相信。',
               en:'Querying "' + miss + '": position ' + zero +
                   ' is 0. A single zero is proof the word was never added - a no is a proof, and you can act on it without checking anything else.'}});

  const ghost = POOL.filter(w => bf.has(w) && new Set(bf.indices(w)).size === BF_K)[0];
  const gix = bf.indices(ghost);
  const owner = {};
  MEMBERS.forEach(w => bf.indices(w).forEach(i => { (owner[i] = owner[i] || []).push(w); }));
  F.push({shapes:itemHead(ghost, gix, 'bad').concat(bitRow(bf.bits,
            Object.assign({}, done, Object.fromEntries(gix.map(i => [i, 'bad']))))),
          panels:panels([{lbl:{zh:'這三格是誰打開的', en:'who set these bits'},
            chips:gix.map(i => chip(i + ' ← ' + owner[i].join('/'), 'bad'))}]),
          view:VIEW, line:12,
          msg:{zh:'"' + ghost + '" 從來沒被加過，但它的三格分別被 ' +
                   gix.map(i => owner[i][0]).join('、') +
                   ' 打開了，於是 Bloom filter 回答 yes。這就是 false positive：yes 只是「有可能」，要拿去真正的資料庫再確認一次。',
               en:'"' + ghost + '" was never added, yet its three bits were switched on by ' +
                   gix.map(i => owner[i][0]).join(', ') +
                   ' respectively, so the filter says yes. That is a false positive: a yes only means "maybe", and has to be confirmed against the real store.'}});

  F.push({shapes:bitRow(bf.bits, done), panels:panels(), view:VIEW, line:12,
          msg:{zh:'所以 Bloom filter 的誤差是有方向的：它只會多說 yes，永遠不會漏掉真的有的東西。當 no 很便宜、yes 只是「再查一次」時（快取前置過濾、擋掉不存在的 key），它就是對的工具。',
               en:'So the error has a direction: the filter only ever says yes too often, and never loses something that is really there. When a no is cheap and a yes just means "go check for real" - a cache pre-filter, a guard against keys that do not exist - this is exactly the right tool.'}});
  return F.list;
}

/* ================================================ 2. deleting is the trap */
const DEL_M = 20, DEL_K = 3;
const WORDS = 'alpha bravo charlie delta echo foxtrot golf hotel'.split(' ');
const VICTIM = 'hotel';

const CODE_DEL = [
  '# a plain Bloom filter has no delete',
  'def remove_wrong(self, item):',
  '    for idx in self.indices(item):',
  '        self.bits[idx] = 0    # also erases bits others rely on',
  '',
  '# counting Bloom filter: 4-bit counters, not single bits',
  'def add(self, item):',
  '    for idx in self.indices(item): self.cnt[idx] += 1',
  'def delete(self, item):',
  '    for idx in self.indices(item): self.cnt[idx] -= 1',
  'def __contains__(self, item):',
  '    return all(self.cnt[i] > 0 for i in self.indices(item))',
];

const DX0 = 0.45, DCW = 0.44, DCY = 1.70, DCH = 0.58;
const WX0 = 0.30, WCW = 1.18;

function delRow(vals, st, counting){
  return cellRow(vals, DX0, DCY, DCW, DCH, {states:st, fs:.30, ifs:.24, ilift:.30})
    .concat([S.t(4.85, DCY - .45, counting ? {zh:'counting Bloom：每格是一個 4-bit 計數器',
                                              en:'counting Bloom - every slot is a 4-bit counter'}
                                           : {zh:'Bloom filter：每格是一個位元',
                                              en:'plain Bloom filter - every slot is one bit'},
                 {c:COL.tealL, fs:.34})]);
}
function wordRow(states){
  const out = [];
  WORDS.forEach((w, i) => {
    out.push(S.r(WX0 + i * WCW, 4.35, WCW - .10, 0.66, states[w] || 'idle', w, {fs:.30}));
  });
  out.push(S.t(4.85, 4.05, {zh:'刪除之後，每個字還查得到嗎？',
                            en:'after the delete, can each word still be found?'},
               {c:COL.grey, fs:.32}));
  return out;
}

function deleteFrames(v){
  const counting = (v === 1);
  const F = new Frames();
  const bf = new Bloom(DEL_M, DEL_K), cb = new Counting(DEL_M, DEL_K);
  WORDS.forEach(w => { bf.add(w); cb.add(w); });
  const vals = () => counting ? cb.cnt.slice() : bf.bits.slice();
  const has = w => counting ? cb.has(w) : bf.has(w);
  const vix = bf.indices(VICTIM);
  const owner = {};
  WORDS.forEach(w => bf.indices(w).forEach(i => { (owner[i] = owner[i] || []).push(w); }));
  const shared = vix.filter(i => owner[i].length > 1);

  const panels = () => ([
    {lbl:{zh:'字數 n / 位元 m / 雜湊 k', en:'n / m / k'},
     chips:[chip(WORDS.length + ' / ' + DEL_M + ' / ' + DEL_K, '')]},
    {lbl:{zh:'要刪掉的字', en:'word being deleted'}, chips:[chip(VICTIM, 'hot')]},
    {lbl:{zh:'它的位置', en:'its slots'}, chips:vix.map(i => chip(String(i),
      owner[i].length > 1 ? 'bad' : ''))},
    {lbl:{zh:'查不到的字（false negative）', en:'words now missing (false negatives)'},
     chips:(() => { const bad = WORDS.filter(w => w !== VICTIM && !has(w));
                    return bad.length ? bad.map(w => chip(w, 'bad'))
                                      : [chip({zh:'（沒有）', en:'(none)'}, 'ok')]; })()}
  ]);

  const st0 = {};
  F.push({shapes:delRow(vals(), st0).concat(wordRow({})), panels:panels(), view:VIEW,
          line:counting ? 7 : 0,
          msg:counting
            ? {zh:'同樣八個字，但這次每一格記的是「有幾個字用到我」，不是「有沒有人用過我」。多花 4 倍記憶體，換回刪除的能力。',
               en:'The same eight words, but now each slot records how many items rely on it, not merely whether anyone ever did. Four times the memory buys the ability to delete.'}
            : {zh:'八個字進了 20 個位元。位元比字多，但一定會有幾格被兩個以上的字共用 —— 問題就出在這裡。',
               en:'Eight words in twenty bits. There are more bits than words, yet some slots inevitably end up shared by two or more items - and that is where the trouble starts.'}});

  const hotv = Object.fromEntries(vix.map(i => [i, owner[i].length > 1 ? 'bad' : 'hot']));
  F.push({shapes:delRow(vals(), hotv).concat(wordRow({[VICTIM]:'hot'})), panels:panels(),
          view:VIEW, line:counting ? 9 : 2,
          msg:{zh:'要刪掉 "' + VICTIM + '"，它佔用的是 ' + vix.join('、') + '。其中 ' +
                   shared.join('、') + ' 同時也被 ' +
                   shared.map(i => owner[i].filter(w => w !== VICTIM).join('/')).join('、') +
                   ' 用著 —— 這幾格不是 "' + VICTIM + '" 一個人的。',
               en:'To delete "' + VICTIM + '" we look at slots ' + vix.join(', ') + '. Slot ' +
                   shared.join(', ') + ' is also in use by ' +
                   shared.map(i => owner[i].filter(w => w !== VICTIM).join('/')).join(', ') +
                   ' - those slots do not belong to "' + VICTIM + '" alone.'}});

  if (counting) cb.del(VICTIM); else vix.forEach(i => bf.bits[i] = 0);
  const after = Object.fromEntries(vix.map(i => [i, counting ? 'ok' : 'bad']));
  F.push({shapes:delRow(vals(), after).concat(wordRow({[VICTIM]:'done'})), panels:panels(),
          view:VIEW, line:counting ? 9 : 3,
          msg:counting
            ? {zh:'計數器版本只是把這三格各減 1。共用的那一格從 2 掉到 1，還是大於 0，所以別人的資料沒有被破壞。',
               en:'The counting version merely subtracts one from each of the three slots. The shared slot drops from 2 to 1 - still above zero, so nobody else’s membership is destroyed.'}
            : {zh:'天真的刪法把這三格直接歸零。"' + VICTIM + '" 確實查不到了，但共用那格的鄰居也一起被清掉了。',
               en:'The naive delete just zeroes all three slots. "' + VICTIM + '" is indeed gone - and so is the evidence for whoever was sharing a slot with it.'}});

  const wst = {[VICTIM]:'done'};
  WORDS.filter(w => w !== VICTIM).forEach(w => {
    wst[w] = has(w) ? 'ok' : 'bad';
    F.push({shapes:delRow(vals(), after).concat(wordRow(Object.assign({}, wst))),
            panels:panels(), view:VIEW, line:counting ? 11 : 2,
            msg:has(w)
              ? {zh:'"' + w + '" 還在：它的三格都還是 ' + (counting ? '大於 0' : '1') + '。',
                 en:'"' + w + '" survives - all three of its slots are still ' +
                     (counting ? 'above zero' : 'ones') + '.'}
              : {zh:'"' + w + '" 不見了！它從頭到尾都在集合裡，卻被回答 no —— 這是 false negative，而 Bloom filter 本來保證不會發生這種事。一個「證明」變成了謊言。',
                 en:'"' + w + '" is gone. It was in the set the whole time and the filter now answers no - a false negative, the one thing a Bloom filter is supposed to make impossible. The proof has turned into a lie.'}});
  });

  F.push({shapes:delRow(vals(), after).concat(wordRow(wst)), panels:panels(), view:VIEW,
          line:counting ? 11 : 3,
          msg:counting
            ? {zh:'代價：4-bit 計數器等於 4 倍記憶體，而且計數器會飽和（到 15 就加不上去，之後再減就會錯）。要刪除就得放棄「只用 1 bit」這個最大的好處。',
               en:'The price: four-bit counters cost four times the memory, and counters saturate - once one hits 15 it stops growing, and every later decrement is wrong. Deletion means giving up the single best property, that a slot is one bit.'}
            : {zh:'結論不是「Bloom filter 有 bug」，而是「它從來就沒有 delete 這個操作」。要刪就重建整個 filter，或改用 counting 版本 —— 切到右邊的變體看看。',
               en:'The lesson is not that the Bloom filter is buggy - it is that delete was never one of its operations. If you need removal, rebuild the filter from scratch or switch to the counting variant on the right.'}});
  return F.list;
}

/* ================================================== 3. HyperLogLog */
const HP = 4, HM = 1 << HP;
const USERS = []; for (let i = 1; i <= 40; i++) USERS.push('user' + i);

const CODE_HLL = [
  'def add(self, item):',
  '    h = hash64(item)',
  '    idx  = h & (self.m - 1)        # low p bits pick a register',
  '    rest = h >> self.p             # the bits left over',
  '    rho  = trailing_zeros(rest) + 1',
  '    self.reg[idx] = max(self.reg[idx], rho)   # keep the record',
  '',
  'def count(self):',
  '    z = sum(2.0 ** -r for r in self.reg)',
  '    est = alpha_m * self.m ** 2 / z          # harmonic mean',
  '    if est <= 2.5 * self.m and 0 in self.reg:',
  '        est = self.m * log(self.m / self.reg.count(0))   # small range',
  '    return est',
];

const HBX = 1.05, HBW = 0.52, HBY = 0.85, HBH = 0.56;
const HRX = 0.55, HRW = 0.56, HRY = 3.35, HRH = 0.62;

function bitStrip(h, rho){
  const bits = [];
  for (let j = 15; j >= 0; j--) bits.push((h >>> j) & 1);
  const st = {};
  for (let j = 16 - HP; j < 16; j++) st[j] = 'hot';           // low p bits = index
  for (let t = 0; t < rho && (15 - HP - t) >= 0; t++) st[15 - HP - t] = 'act';
  const labels = []; for (let j = 15; j >= 0; j--) labels.push(String(j));
  return cellRow(bits, HBX, HBY, HBW, HBH, {states:st, labels:labels, fs:.30, ifs:.22,
                                            ilift:.28})
    .concat([
      S.t(9.45, HBY - .30, {zh:'低 p 位 = 暫存器編號', en:'low p bits = register index'},
          {c:COL.orangeL, fs:.28, anchor:'end'}),
      S.t(HBX, HBY - .30, {zh:'其餘位元：右端的 0 串長度 = rho',
                           en:'the rest: its run of zeros = rho'},
          {c:COL.purpleL, fs:.28, anchor:'start'})]);
}
function regRow(reg, st){
  return cellRow(reg, HRX, HRY, HRW, HRH, {states:st, fs:.34, ifs:.26, ilift:.32})
    .concat([S.t(4.9, HRY - .32, {zh:'16 個暫存器，每格只記看過最長的 0 串',
                                  en:'16 registers, each keeps its longest run of zeros'},
                 {c:COL.tealL, fs:.30})]);
}

function hllFrames(){
  const h = new HLL(HP), F = new Frames();
  const seen = new Set();
  const panels = (extra) => ([
    {lbl:{zh:'暫存器個數 m', en:'registers m'}, chips:[chip(String(HM), '')]},
    {lbl:{zh:'實際看過的不重複個數', en:'true distinct seen'},
     chips:[chip(String(seen.size), 'ok')]},
    {lbl:{zh:'估計值', en:'estimate'}, chips:[chip(h.count().toFixed(1), 'hot')]},
    {lbl:{zh:'記憶體', en:'memory'},
     chips:[chip(HM + ' B', 'ok'), chip({zh:'精確集合 ≈ ' + (seen.size * 48) + ' B',
                                          en:'exact set ~ ' + (seen.size * 48) + ' B'}, '')]}
  ]).concat(extra || []);

  F.push({shapes:regRow(h.reg, {}), panels:panels(), view:VIEW, line:0,
          msg:{zh:'要算「有幾個不重複的使用者」，精確做法是存一個 set —— 記憶體跟著資料長大。HyperLogLog 換個問題：既然雜湊值是均勻亂數，那「看過最長的一串 0」本身就透露了看過多少東西。',
               en:'Counting distinct users exactly means keeping a set, and the memory grows with the data. HyperLogLog asks a different question: if the hashes are uniformly random, then the longest run of zeros you have ever seen is itself evidence of how many things you have seen.'}});

  const shown = [];
  for (const u of USERS){
    if (shown.length >= 8) break;
    const p = h.parts(u);
    if (shown.length === 0 && p.rho < 2) continue;    // feature an interesting one first
    shown.push(u);
    const r = h.add(u); seen.add(u);
    const rst = {}; rst[r.idx] = r.grew ? 'hot' : 'done';
    F.push({shapes:bitStrip(r.h, r.rho).concat(regRow(h.reg, rst))
              .concat([S.r(0.30, HBY, 0.65, HBH, 'act', '', {fs:.28}),
                       S.t(0.62, HBY + HBH + .40, u, {c:COL.purpleL, fs:.30}),
                       S.t(4.9, 5.35,
                           {zh:u + ' → 編號 ' + r.idx + '、rho = ' + r.rho + '，' +
                               (r.grew ? '刷新紀錄' : '不比舊值大，忽略'),
                            en:u + ' -> register ' + r.idx + ', rho = ' + r.rho + ' - ' +
                               (r.grew ? 'a new record' : 'not a record, ignored')},
                           {c:r.grew ? COL.orangeL : COL.grey, fs:.36})]),
            panels:panels(), view:VIEW, line:r.grew ? 5 : 5,
            msg:{zh:'雜湊 ' + u + '：低 4 位挑出暫存器 ' + r.idx +
                     '，剩下的位元右端有 ' + (r.rho - 1) + ' 個 0，所以 rho = ' + r.rho + '。' +
                     (r.grew ? '比暫存器裡的舊值大，寫進去。'
                             : '不比舊值大，什麼都不做 —— 同一個字重複幾百次，結果也完全一樣，這就是為什麼它天生去重。'),
                 en:'Hash ' + u + ': the low four bits select register ' + r.idx +
                     ', the remaining bits end in ' + (r.rho - 1) + ' zeros, so rho = ' + r.rho + '. ' +
                     (r.grew ? 'That beats the value already stored, so it is written.'
                             : 'It does not beat the stored value, so nothing happens - feed the same item in a hundred times and the sketch is unchanged, which is why it deduplicates for free.')}});
  }

  USERS.forEach(u => { h.add(u); seen.add(u); });
  const est = h.count(), err = (est - seen.size) / seen.size * 100;
  F.push({shapes:regRow(h.reg, Object.fromEntries(h.reg.map((r, i) => [i, r ? 'ok' : 'bad']))),
          panels:panels(), view:VIEW, line:9,
          msg:{zh:'把 40 個使用者全部丟進去。估計值 ' + est.toFixed(1) + '，真值 40，誤差 ' +
                   err.toFixed(1) + '%。單一暫存器的雜訊很大，所以是取 16 個的調和平均 —— 調和平均會壓住那些偶然爆掉的大值。',
               en:'Feed in all forty users. The estimate is ' + est.toFixed(1) +
                   ' against a true 40, an error of ' + err.toFixed(1) +
                   '%. A single register is far too noisy, so the count is a harmonic mean over all sixteen - harmonic because it damps down the registers that got a freakishly long run.'}});

  const big = new HLL(10), n2 = 5000;
  for (let i = 0; i < n2; i++) big.add('u' + i);
  F.push({shapes:regRow(h.reg, {}), panels:panels([{lbl:{zh:'m 變大以後', en:'with a bigger m'},
            chips:[chip('m=1024 → ' + big.count().toFixed(0) + ' / 5000', 'ok')]}]),
          view:VIEW, line:9,
          msg:{zh:'誤差大約是 1.04/√m。這裡 m=16 所以是 26%，難看；把 m 拉到 1024，5000 個不重複值估成 ' +
                   big.count().toFixed(0) + '。實務上用 p=14（16384 個暫存器、16 KB），誤差 0.81% —— 而且不管資料是一百萬還是十億，都還是 16 KB。',
               en:'The error is about 1.04/sqrt(m). With m=16 that is 26%, which is ugly; raise m to 1024 and five thousand distinct values come out as ' +
                   big.count().toFixed(0) +
                   '. In practice everyone runs p=14 - 16384 registers, 16 KB - for 0.81% error, and it is still 16 KB whether the input is a million rows or a billion.'}});

  const a = new HLL(HP), b = new HLL(HP), whole = new HLL(HP);
  USERS.slice(0, 25).forEach(u => { a.add(u); whole.add(u); });
  USERS.slice(15).forEach(u => { b.add(u); whole.add(u); });
  const merged = new HLL(HP);
  for (let i = 0; i < HM; i++) merged.reg[i] = Math.max(a.reg[i], b.reg[i]);
  const same = merged.reg.every((r, i) => r === whole.reg[i]);
  F.push({shapes:regRow(merged.reg, Object.fromEntries(merged.reg.map((r, i) =>
            [i, r === whole.reg[i] ? 'ok' : 'bad']))),
          panels:panels([{lbl:{zh:'合併結果與單次掃描相同？', en:'merge equals one pass?'},
            chips:[chip(same ? 'yes' : 'no', same ? 'ok' : 'bad')]}]),
          view:VIEW, line:5,
          msg:{zh:'最實用的性質：兩台機器各自算自己的 HLL，合併時只要逐格取 max，結果跟「一台機器掃全部資料」一模一樣（' +
                   (same ? '這裡就是位元完全相同' : '不同') +
                   '）。合併不會累積誤差，所以每小時存一份、事後任意組合成任何區間都可以。',
               en:'The property that makes it worth shipping: two machines each keep their own sketch, and merging is a register-wise max whose result is bit-identical to one machine scanning everything (' +
                   (same ? 'exactly what happens here' : 'not here') +
                   '). Merging adds no error at all, so you can store one sketch per hour and later combine any range of them.'}});
  return F.list;
}

/* ================================================== 4. Count-Min sketch */
const VOCAB = ['the','of','and','to','in','cat','dog','fox','owl','yak',
               'elk','ram','bee','ant','pig','rat'];
const TRUE_COUNT = {}, STREAM = [];
VOCAB.forEach((w, i) => {
  const c = Math.max(1, Math.round(60 / (i + 1)));
  TRUE_COUNT[w] = c;
  for (let j = 0; j < c; j++) STREAM.push(w);
});

const CODE_CMS = [
  'def add(self, item, count=1):',
  '    for i, row in enumerate(self.rows):',
  '        row[hash(item, i) % self.w] += count',
  '',
  'def query(self, item):',
  '    return min(self.rows[i][hash(item, i) % self.w]',
  '               for i in range(self.d))',
  '',
  '# every row over-counts (collisions only ever add),',
  '# so the smallest row is the closest to the truth,',
  '# and the answer is never below the real count.',
];

const CD = 4, CW_ = 8;
const GX0 = 1.35, GCW = 0.95, GCH = 0.62, GY0 = 1.05, GDY = 0.92;

function grid(cms, hot, minRow){
  const out = [];
  for (let r = 0; r < cms.d; r++){
    const st = {};
    if (hot) st[hot[r]] = (r === minRow) ? 'ok' : 'hot';
    out.push.apply(out, cellRow(cms.rows[r], GX0, GY0 + r * GDY, GCW, GCH,
      {states:st, fs:.32, index:(r === cms.d - 1), ifs:.26, ilift:.32}));
    out.push(S.t(GX0 - .20, GY0 + r * GDY + GCH * .70, 'row ' + r,
                 {c:COL.grey, fs:.30, anchor:'end'}));
  }
  return out;
}

function cmsFrames(){
  const cms = new CMS(CD, CW_), F = new Frames();
  let n = 0;
  const panels = (extra) => ([
    {lbl:{zh:'d 列 × w 欄', en:'d rows x w columns'}, chips:[chip(CD + ' x ' + CW_, '')]},
    {lbl:{zh:'已讀入的事件數 N', en:'events read N'}, chips:[chip(String(n), '')]},
    {lbl:{zh:'不重複的字', en:'distinct words'}, chips:[chip(String(VOCAB.length), '')]},
    {lbl:{zh:'誤差上界 e/w · N', en:'error bound e/w * N'},
     chips:[chip('+' + (Math.E / CW_ * n).toFixed(0), 'hot')]}
  ]).concat(extra || []);

  F.push({shapes:grid(cms), panels:panels(), view:VIEW, line:0,
          msg:{zh:'Count-Min sketch 回答的是「這個東西出現幾次」。精確做法是一個 dict，key 越多越大；這裡永遠只有 4×8 = 32 個計數器，跟有幾種字完全無關。',
               en:'A Count-Min sketch answers "how often did this appear". The exact answer is a dict whose size follows the number of distinct keys; here it is always 4 x 8 = 32 counters, no matter how many distinct words show up.'}});

  ['the', 'of', 'the'].forEach(w => {
    const cs = cms.add(w); n++;
    F.push({shapes:grid(cms, cs), panels:panels([{lbl:{zh:'目前這個字', en:'current item'},
              chips:[chip(w, 'hot')]}]), view:VIEW, line:2,
            msg:{zh:'讀到 "' + w + '"：每一列用自己的雜湊挑一欄（' + cs.join('、') +
                     '），四格各加 1。每一列都是一份完整但會撞在一起的統計。',
                 en:'The word "' + w + '" arrives: each row picks its own column with its own hash (' +
                     cs.join(', ') + ') and adds one there. Every row is a complete tally of the stream, just one with collisions in it.'}});
  });
  STREAM.slice(3).forEach(w => { cms.add(w); n++; });

  const heavy = 'the', hc = cms.cols(heavy), hv = hc.map((c, i) => cms.rows[i][c]);
  const hmin = Math.min.apply(null, hv), hrow = hv.indexOf(hmin);
  F.push({shapes:grid(cms, hc, hrow), panels:panels([{lbl:{zh:'四列各自的讀數', en:'the four row readings'},
            chips:hv.map((v, i) => chip(String(v), i === hrow ? 'ok' : 'hot'))}]),
          view:VIEW, line:5,
          msg:{zh:'查 "' + heavy + '"（真實 ' + TRUE_COUNT[heavy] + ' 次）：四列讀到 ' +
                   hv.join('、') + '，取最小值 ' + hmin + '。碰撞只會讓數字變大，所以最小的那列受汙染最少 —— 而且答案永遠不會小於真值。',
               en:'Query "' + heavy + '" (true count ' + TRUE_COUNT[heavy] + '): the four rows read ' +
                   hv.join(', ') + ', and the answer is the minimum, ' + hmin +
                   '. Collisions can only push a counter up, so the smallest row is the least polluted one - and the answer can never fall below the truth.'}});

  const rare = VOCAB.slice().sort((a, b) =>
    (cms.query(b) - TRUE_COUNT[b]) - (cms.query(a) - TRUE_COUNT[a]))[0];
  const rc = cms.cols(rare), rv = rc.map((c, i) => cms.rows[i][c]);
  const rmin = Math.min.apply(null, rv), rrow = rv.indexOf(rmin);
  const over = rmin - TRUE_COUNT[rare];
  F.push({shapes:grid(cms, rc, rrow), panels:panels([{lbl:{zh:'真值 / 估計 / 多算', en:'true / estimate / over'},
            chips:[chip(String(TRUE_COUNT[rare]), ''), chip(String(rmin), 'hot'),
                   chip('+' + over, 'bad')]}]),
          view:VIEW, line:5,
          msg:{zh:'查 "' + rare + '"（真實只有 ' + TRUE_COUNT[rare] + ' 次）卻讀到 ' + rmin +
                   '，多算了 ' + over + '。注意兩次的絕對誤差差不多，都遠小於上界 +' +
                   (Math.E / CW_ * n).toFixed(0) + ' —— 但對一個只出現 ' + TRUE_COUNT[rare] +
                   ' 次的字來說，這是 ' + (over / TRUE_COUNT[rare] * 100).toFixed(0) + '% 的相對誤差。',
               en:'Query "' + rare + '", which truly occurred ' + TRUE_COUNT[rare] +
                   ' times, and the sketch reports ' + rmin + ' - an over-count of ' + over +
                   '. Both queries have a similar absolute error, comfortably inside the bound of +' +
                   (Math.E / CW_ * n).toFixed(0) + '. But on a word seen only ' + TRUE_COUNT[rare] +
                   ' times that same error is ' + (over / TRUE_COUNT[rare] * 100).toFixed(0) + '% of the answer.'}});

  F.push({shapes:grid(cms), panels:panels(), view:VIEW, line:10,
          msg:{zh:'這就是 CMS 的性格：誤差上界 e/w·N 是「絕對」的，跟這個字本身有多大無關。所以它對熱門的東西幾乎不會錯，對長尾則錯得離譜 —— 拿來找 heavy hitter 很好，拿來判斷「這個 key 是不是只出現一次」則完全不能用。',
               en:'That is the personality of the sketch: the bound e/w * N is absolute and has nothing to do with how big the item itself is. So it is nearly perfect on the popular items and wildly wrong on the tail - excellent for finding heavy hitters, useless for asking whether a key appeared exactly once.'}});
  return F.list;
}

/* ============================================ 5. LeetCode 347 on a stream */
const CODE_LC = [
  '# LC 347 - exact, when the whole array fits in memory',
  'def topKFrequent(nums, k):',
  '    cnt = Counter(nums)              # one entry per distinct key',
  '    return heapq.nlargest(k, cnt, key=cnt.get)',
  '',
  '# the streaming version - fixed memory, no full Counter',
  'def top_k_stream(stream, k, sketch):',
  '    best = {}',
  '    for x in stream:',
  '        sketch.add(x)',
  '        est = sketch.query(x)',
  '        if x in best or len(best) < k or est > min(best.values()):',
  '            best[x] = est',
  '            if len(best) > k:',
  '                best.pop(min(best, key=best.get))',
  '    return sorted(best, key=best.get, reverse=True)',
];

const LX0 = 0.15, LCW = 0.58, LCH = 0.56;
function freqRow(vals, y, states, title){
  return cellRow(vals, LX0, y, LCW, LCH,
                 {states:states, labels:VOCAB, fs:.26, ifs:.22, ilift:.28})
    .concat([S.t(4.85, y - .30, title, {c:COL.tealL, fs:.32})]);
}
function podium(y, items, cls, title){
  const out = [S.t(1.55, y + .42, title, {c:COL.grey, fs:.32, anchor:'end'})];
  items.forEach((it, i) => {
    out.push(S.r(1.75 + i * 2.35, y, 2.15, 0.66, cls[i] || 'ok',
                 '#' + (i + 1) + '  ' + it, {fs:.32}));
  });
  return out;
}

function lcFrames(v){
  const w = (v === 1) ? 16 : 8;
  const F = new Frames();
  const K = 3;
  const exact = VOCAB.slice().sort((a, b) => TRUE_COUNT[b] - TRUE_COUNT[a]).slice(0, K);
  const trueVals = VOCAB.map(x => TRUE_COUNT[x]);
  const cms = new CMS(CD, w);
  const best = {};
  for (const x of STREAM){
    cms.add(x);
    const est = cms.query(x), keys = Object.keys(best);
    if (x in best || keys.length < K ||
        est > Math.min.apply(null, keys.map(q => best[q]))){
      best[x] = est;
      if (Object.keys(best).length > K){
        let lo = null;
        for (const q in best) if (lo === null || best[q] < best[lo]) lo = q;
        delete best[lo];
      }
    }
  }
  const approx = Object.keys(best).sort((a, b) => best[b] - best[a]);
  const estVals = VOCAB.map(x => cms.query(x));
  const ok = approx.length === K && approx.every((x, i) => x === exact[i]);

  const exSt = {}; exact.forEach(x => exSt[VOCAB.indexOf(x)] = 'ok');
  const apSt = {};
  VOCAB.forEach((x, i) => {
    if (approx.indexOf(x) >= 0) apSt[i] = (exact.indexOf(x) >= 0) ? 'ok' : 'bad';
    else if (estVals[i] > TRUE_COUNT[x]) apSt[i] = 'hot';
  });
  const panels = (extra) => ([
    {lbl:{zh:'k', en:'k'}, chips:[chip(String(K), '')]},
    {lbl:{zh:'sketch 大小 d × w', en:'sketch size d x w'}, chips:[chip(CD + ' x ' + w, '')]},
    {lbl:{zh:'精確答案', en:'exact answer'}, chips:exact.map(x => chip(x, 'ok'))},
    {lbl:{zh:'sketch 的答案', en:'sketch answer'},
     chips:approx.map(x => chip(x + ':' + best[x], exact.indexOf(x) >= 0 ? 'ok' : 'bad'))},
    {lbl:{zh:'兩者相同？', en:'same top-k?'},
     chips:[chip(ok ? 'yes' : 'no', ok ? 'ok' : 'bad')]}
  ]).concat(extra || []);

  F.push({shapes:freqRow(trueVals, 1.05, {}, {zh:'真實次數（Counter 算出來的）',
                                              en:'true counts, as a Counter would give them'}),
          panels:panels(), view:VIEW, line:1,
          msg:{zh:'LeetCode 347 給你一個陣列，要出現次數前 k 名。標準解是 Counter 加一個 heap，O(n) 時間、O(不重複個數) 空間 —— 前提是所有 key 都放得進記憶體。',
               en:'LeetCode 347 hands you an array and asks for the k most frequent values. The textbook answer is a Counter plus a heap: O(n) time and memory proportional to the number of distinct keys - which assumes every key fits in memory.'}});

  F.push({shapes:freqRow(trueVals, 1.05, exSt, {zh:'真實次數（Counter 算出來的）',
                                                en:'true counts, as a Counter would give them'})
            .concat(podium(3.05, exact.map(x => x + '  ' + TRUE_COUNT[x]),
                           ['ok', 'ok', 'ok'], {zh:'精確', en:'exact'})),
          panels:panels(), view:VIEW, line:3,
          msg:{zh:'精確答案：' + exact.join('、') +
                   '。在面試裡到這裡就結束了，但在真實的流量統計裡，key 可能有幾億個，Counter 本身就是那個放不下的東西。',
               en:'The exact answer is ' + exact.join(', ') +
                   '. In an interview that is the end of it, but on real traffic there can be hundreds of millions of keys, and the Counter itself is the thing that does not fit.'}});

  F.push({shapes:freqRow(estVals, 1.05, apSt, {zh:'Count-Min sketch 的估計（' + CD + '×' + w + '）',
                                               en:'Count-Min estimates (' + CD + ' x ' + w + ')'}),
          panels:panels(), view:VIEW, line:10,
          msg:{zh:'改用固定大小的 sketch：每讀一個字就 add 再 query，並且只留 k 個候選。記憶體是 ' +
                   (CD * w) + ' 個計數器加 ' + K + ' 個候選，跟不重複個數無關。橘色的格子就是被多算的。',
               en:'Swap in the fixed-size sketch: add each word as it arrives, query it back, and keep only k candidates. The memory is ' +
                   (CD * w) + ' counters plus ' + K +
                   ' candidates, independent of how many distinct words exist. The orange cells are the ones being over-counted.'}});

  F.push({shapes:freqRow(estVals, 1.05, apSt, {zh:'Count-Min sketch 的估計（' + CD + '×' + w + '）',
                                               en:'Count-Min estimates (' + CD + ' x ' + w + ')'})
            .concat(podium(3.05, exact.map(x => x + '  ' + TRUE_COUNT[x]), ['ok','ok','ok'],
                           {zh:'精確', en:'exact'}))
            .concat(podium(4.15, approx.map(x => x + '  ' + best[x]),
                           approx.map(x => exact.indexOf(x) >= 0 ? 'ok' : 'bad'),
                           {zh:'sketch', en:'sketch'})),
          panels:panels(), view:VIEW, line:15,
          msg:ok
            ? {zh:'w = ' + w + ' 時兩邊完全一致。前幾名的真實次數遠大於 e/w·N 這個誤差，所以雜訊淹不掉它們 —— 這正是 heavy hitter 適合用 sketch 的原因。',
               en:'With w = ' + w + ' the two answers agree exactly. The leaders’ true counts are far larger than the e/w * N error, so the noise cannot drown them - which is precisely why heavy hitters are the right job for a sketch.'}
            : {zh:'w = ' + w + ' 太窄，第三名被 "' + approx[K - 1] +
                   '" 佔走了 —— 它真的只出現 ' + TRUE_COUNT[approx[K - 1]] + ' 次，卻被估成 ' +
                   best[approx[K - 1]] + '。誤差上界沒有被違反，是我們把「上界 e/w·N」開得比第三名的真實次數還大。',
               en:'With w = ' + w + ' the sketch is too narrow and third place is stolen by "' +
                   approx[K - 1] + '", a word that truly appears ' + TRUE_COUNT[approx[K - 1]] +
                   ' times but is estimated at ' + best[approx[K - 1]] +
                   '. No bound was violated: we simply let e/w * N grow larger than the true count of the item in third place.'}});

  F.push({shapes:freqRow(estVals, 1.05, apSt, {zh:'Count-Min sketch 的估計（' + CD + '×' + w + '）',
                                               en:'Count-Min estimates (' + CD + ' x ' + w + ')'})
            .concat(podium(3.05, exact.map(x => x + '  ' + TRUE_COUNT[x]), ['ok','ok','ok'],
                           {zh:'精確', en:'exact'}))
            .concat(podium(4.15, approx.map(x => x + '  ' + best[x]),
                           approx.map(x => exact.indexOf(x) >= 0 ? 'ok' : 'bad'),
                           {zh:'sketch', en:'sketch'})),
          panels:panels(), view:VIEW, line:11,
          msg:{zh:'所以 sketch 的大小不是隨便挑的：先問「我在意的第 k 名大概多大」，再讓 e/w·N 明顯小於它。三個結構都一樣 —— 先決定可以錯到什麼程度、錯的方向可不可以接受，再決定要花多少記憶體。',
               en:'So the size of a sketch is not a free parameter: ask how big the k-th item you care about is, then make e/w * N clearly smaller than that. All three structures work this way - decide how wrong you can afford to be and in which direction, then buy exactly that much memory.'}});
  return F.list;
}

/* ========================================================== the day meta */
const DAY_META = {
  title: {zh:'Bloom Filter、HyperLogLog、Count-Min Sketch：用固定記憶體回答集合問題',
          en:'Bloom filter, HyperLogLog and Count-Min sketch - answering set questions in kilobytes'},
  sub:   {zh:'三個會出錯的結構，但錯的方向是設計出來的',
          en:'Three structures that are allowed to be wrong - in a direction you choose'},
  tabs: [
    {
      id:'bloom', label:{zh:'Bloom 查詢', en:'Bloom'},
      stage:{zh:'k 個雜湊、一個位元陣列、只會多說 yes',
             en:'k hashes, one bit array, and an error that only ever says yes'},
      view:VIEW,
      idea:{zh:'Bloom filter 不存資料，只存「哪幾個位元被打開」。因為位元會被不同的字共用，它可能把沒加過的東西說成 yes，但絕對不會把加過的東西說成 no。這個不對稱就是它全部的價值：no 可以直接相信，yes 再去查真正的資料就好。',
            en:'A Bloom filter stores no data, only which bits got switched on. Because bits are shared between items it can say yes to something it never saw, but it can never say no to something it did. That asymmetry is the whole point: a no can be trusted outright, and a yes just means go and check the real store.'},
      legend:[['#ff9736', {zh:'正在寫入的位元', en:'bit being written'}],
              ['#3fe0dd', {zh:'已經是 1', en:'already 1'}],
              ['#ff5c5c', {zh:'誤判 / 證明不存在的 0', en:'false positive / the proving zero'}]],
      code:CODE_BLOOM, build:bloomFrames
    },
    {
      id:'delete', label:{zh:'刪除的陷阱', en:'the delete trap'},
      stage:{zh:'把位元清成 0，等於刪掉別人的資料',
             en:'zeroing a bit also erases somebody else'},
      view:VIEW,
      variants:[{zh:'天真的刪除', en:'naive delete'}, {zh:'counting Bloom', en:'counting Bloom'}],
      idea:{zh:'Bloom filter 沒有 delete，不是忘了實作，而是做不到：一個位元可能代表好幾個字，清成 0 就會讓別人也消失，於是「不會有 false negative」這個保證當場破功。要刪除就得把位元換成計數器，代價是 4 倍記憶體與飽和問題。',
            en:'A Bloom filter has no delete - not as an oversight but as a consequence. One bit can stand for several items, so clearing it makes the others vanish too and the no-false-negatives guarantee collapses on the spot. Deletion means replacing bits with counters, which costs four times the memory and brings saturation with it.'},
      legend:[['#ff9736', {zh:'要刪的字佔用的格子', en:'slots of the deleted item'}],
              ['#ff5c5c', {zh:'共用格 / 被害者', en:'shared slot / collateral damage'}],
              ['#3fe0dd', {zh:'仍然查得到', en:'still found'}]],
      code:CODE_DEL, build:deleteFrames
    },
    {
      id:'hll', label:{zh:'HyperLogLog', en:'HyperLogLog'},
      stage:{zh:'用「看過最長的 0 串」猜有幾個不重複的值',
             en:'guessing the distinct count from the longest run of zeros'},
      view:VIEW,
      idea:{zh:'如果雜湊值是均勻亂數，那麼「結尾有 k 個 0」的機率是 2^-k。看到很長的 0 串，代表你大概看過很多東西。HLL 把雜湊值切成兩段：低位選暫存器，高位量 0 串長度，最後對 m 個暫存器取調和平均。它天生去重，而且合併只要逐格取 max，完全不會累積誤差。',
            en:'If a hash is uniformly random, the chance of it ending in k zeros is 2^-k, so a long run of zeros is evidence that you have seen a lot of things. HLL splits the hash in two: the low bits choose a register, the high bits measure the run, and the count is a harmonic mean over m registers. It deduplicates by construction, and merging two sketches is a register-wise max that adds no error at all.'},
      legend:[['#ff9736', {zh:'低 p 位 = 暫存器編號', en:'low p bits = register index'}],
              ['#9d6bff', {zh:'0 串 → rho', en:'run of zeros -> rho'}],
              ['#3fe0dd', {zh:'已寫入的暫存器', en:'register holding a record'}]],
      code:CODE_HLL, build:hllFrames
    },
    {
      id:'cms', label:{zh:'Count-Min', en:'Count-Min'},
      stage:{zh:'d 列各自統計、取最小值，永遠不會少算',
             en:'d independent tallies, take the smallest, never under-count'},
      view:VIEW,
      idea:{zh:'每一列都是一份「會撞在一起」的完整統計，碰撞只會讓數字變大，所以取四列的最小值最接近真相，而且保證不小於真值。重點是誤差上界 e/w·N 是絕對值：對熱門的 key 幾乎無感，對長尾則可能錯好幾倍。',
            en:'Each row is a complete tally of the stream with collisions folded in, and a collision can only push a counter up - so the minimum across the rows is the closest to the truth and is guaranteed never to be below it. The catch is that the bound e/w * N is absolute: negligible for a popular key, and several times the answer for one in the tail.'},
      legend:[['#ff9736', {zh:'這個 key 的四個位置', en:'this key in each row'}],
              ['#3fe0dd', {zh:'最小值 = 回答', en:'the minimum = the answer'}],
              ['#ff5c5c', {zh:'被多算', en:'over-counted'}]],
      code:CODE_CMS, build:cmsFrames
    },
    {
      id:'lc347', label:{zh:'LC 347', en:'LC 347'},
      stage:{zh:'Top K Frequent：Counter 放得下就精確，放不下就用 sketch',
             en:'Top K Frequent: exact while the Counter fits, sketched when it does not'},
      view:VIEW,
      variants:[{zh:'w = 8（太窄）', en:'w = 8 (too narrow)'}, {zh:'w = 16', en:'w = 16'}],
      idea:{zh:'LC 347 的標準解是 Counter 加 heap，空間跟不重複 key 的數量成正比。換成 Count-Min sketch 加 k 個候選之後空間固定，但答案的正確性取決於「第 k 名的真實次數有沒有大過 e/w·N」。sketch 開得太小，長尾就會混進前幾名。',
            en:'The textbook LC 347 answer is a Counter plus a heap, and its memory tracks the number of distinct keys. Swap in a Count-Min sketch with k candidates and the memory becomes fixed, but correctness now depends on whether the true count of the k-th item exceeds e/w * N. Make the sketch too small and the tail climbs into the podium.'},
      legend:[['#3fe0dd', {zh:'正確的前 k 名', en:'correct top-k'}],
              ['#ff9736', {zh:'被多算的 key', en:'over-counted key'}],
              ['#ff5c5c', {zh:'錯誤地擠進前 k 名', en:'wrongly promoted into the top-k'}]],
      code:CODE_LC, build:lcFrames
    }
  ]
};
