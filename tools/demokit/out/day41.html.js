// DAY: 41
// TITLE_ZH: 模運算、快速冪、GCD 與質數篩：計算大到寫不下的數字
// TITLE_EN: Modular arithmetic, fast power, GCD and the prime sieve: computing with numbers too big to write down
// SUB_ZH: 3 的一百萬次方有 1,584,963 個 bit，先算出來再取餘數要 102 ms；pow(3, 1000000, MOD) 每一步都先取餘數，只要 2.3 µs。% 保得住加、減、乘，保不住除——這一條規則決定了今天每一個坑：nCk 用 // 從 n = 13 開始錯、Fermat 反元素在合數模數上默默算錯、LeetCode 372 把指數 mod 1336 錯一半。另外三樣工具：快速冪一個 bit 一次平方、Euclid 每一步至少把數字砍半、埃氏篩從 p² 開始劃。
// SUB_EN: 3 to the millionth power has 1,584,963 bits; computing it and then taking the remainder takes 102 ms, while pow(3, 1000000, MOD) reduces after every step and takes 2.3 µs. % survives addition, subtraction and multiplication but not division - and that one rule explains every trap today: nCk with // goes wrong from n = 13, Fermat's inverse quietly fails on a composite modulus, and LeetCode 372 with the exponent reduced mod 1336 is wrong half the time. The other three tools: fast power squares once per bit, Euclid at least halves the numbers every two steps, and the sieve of Eratosthenes starts crossing at p².
// FOLDER: day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const curV = () => (typeof varIx !== 'undefined' ? varIx : 0);

const isCJK = ch => { const c = ch.codePointAt(0);
  return (c >= 0x2e80 && c <= 0x9fff) || (c >= 0xff00 && c <= 0xff60) || (c >= 0x3000 && c <= 0x303f); };
function tw(str, fs){ let w = 0;
  for (const ch of String(str)) w += isCJK(ch) ? 1.0 : .55;
  return w * fs; }
function fitT(x, y, s, opt){
  opt = Object.assign({}, opt || {});
  const fs = opt.fs || .32, a = opt.anchor || 'middle', pad = .20;
  const room = a === 'start' ? VIEW[0] - pad - x
             : a === 'end'   ? x - pad
             : 2 * Math.min(x - pad, VIEW[0] - pad - x);
  const strs = (typeof s === 'string') ? [s] : [s.zh, s.en].filter(t => t != null);
  const w = Math.max.apply(null, strs.map(t => tw(t, fs)));
  if (w > room) opt.fs = fs * room / w;
  return S.t(x, y, s, opt);
}
function fitFs(lab, w, fs){
  const strs = (typeof lab === 'string') ? [lab] : [lab.zh, lab.en];
  const need = Math.max.apply(null, strs.map(t => tw(t, fs)));
  return need > w - .08 ? fs * (w - .08) / need : fs;
}
/* a box whose label always fits */
const box = (x, y, w, h, s, lab, fs) => S.r(x, y, w, h, s, lab, {fs:fitFs(String(lab), w, fs || h * .5)});
const pyList = a => '[' + a.join(', ') + ']';
const fmt = n => { const s = String(n), neg = s[0] === '-', d = neg ? s.slice(1) : s;
  return (neg ? '-' : '') + d.replace(/\B(?=(\d{3})+(?!\d))/g, ','); };
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}

/* ---------------------------------------------------------- the maths, as in number_theory.py */
const MOD = 1000000007n;
function sieveArr(n){                       /* is_prime[i] for 0 <= i < n */
  const ip = new Uint8Array(Math.max(n, 2)).fill(1); ip[0] = ip[1] = 0;
  for (let p = 2; p * p < n; p++) if (ip[p]) for (let j = p * p; j < n; j += p) ip[j] = 0;
  return Array.from(ip.slice(0, n));
}
const countBelow = n => n > 2 ? sieveArr(n).reduce((a, b) => a + b, 0) : 0;      /* LC 204 */
const countUpto  = n => n >= 2 ? sieveArr(n + 1).reduce((a, b) => a + b, 0) : 0; /* the bug */
const gcdN = (a, b) => { while (b) [a, b] = [b, a % b]; return Math.abs(a); };
function powModB(a, e, m){ a = BigInt(a) % m; e = BigInt(e); let r = 1n % m;
  while (e){ if (e & 1n) r = r * a % m; a = a * a % m; e >>= 1n; } return r; }
const powModN = (a, e, m) => Number(powModB(a, e, BigInt(m)));
const bitLen = e => BigInt(e).toString(2).length;
const popc = e => BigInt(e).toString(2).split('').filter(c => c === '1').length;
const mults = e => bitLen(e) - 1 + popc(e) - 1;          /* squarings + multiply-ins */

/* ================================================================ tab 1: sieve */
const CODE_SIEVE = [
  'def sieve(n):',
  '    is_prime = bytearray([1]) * max(n, 2)',
  '    is_prime[0] = is_prime[1] = 0',
  '    for p in range(2, isqrt(n - 1) + 1 if n > 1 else 2):',
  '        if is_prime[p]:',
  '            is_prime[p * p::p] = bytes(len(range(p * p, n, p)))',
  '    return is_prime[:n]'
];
const CODE_204 = [
  'class Solution:',
  '    def countPrimes(self, n):        # primes strictly below n',
  '        return sum(sieve(n)) if n > 2 else 0',
  '',
  'def count_primes_upto(n):           # the bug: primes <= n',
  '    return sum(sieve(n + 1)) if n >= 2 else 0'
];
const SN = 50;
/* grid of 0..49, 10 per row. st[i] in: none / open / prime / gone; overrides via o.hl */
function sieveStage(st, o){
  const sh = [], x0 = .9, w = .8, h = .68, y0 = .35, dy = .8;
  for (let i = 0; i < SN; i++){
    const x = x0 + (i % 10) * w, y = y0 + Math.floor(i / 10) * dy;
    let s = st[i] === 'none' ? 'ghost' : st[i] === 'prime' ? 'ok' : st[i] === 'gone' ? 'done' : 'idle';
    if (o.hl && o.hl[i]) s = o.hl[i];
    sh.push(S.r(x, y, w - .08, h, s, String(i), {fs:.30}));
  }
  if (o.big) sh.push(fitT(4.9, 4.78, o.big, {c:o.bigc || COL.orangeL, fs:.36}));
  return caps(sh, o);
}
function sieveFrames(v){
  return v === 1 ? lc204Frames() : eraFrames();
}
function eraFrames(){
  const F = new Frames(), st = [];
  for (let i = 0; i < SN; i++) st[i] = i < 2 ? 'none' : 'open';
  const primesNow = () => st.map((s, i) => s === 'prime' ? i : -1).filter(i => i >= 0);
  F.push({shapes:sieveStage(st, {cap:{zh:'n = 50：先假設 2 到 49 全部是質數', en:'n = 50: assume every number from 2 to 49 is prime'},
                                 cap2:{zh:'0 和 1 不是質數，一開始就劃掉', en:'0 and 1 are not prime and are crossed out from the start'}}),
    panels:[{lbl:'n', chips:[chip('50', 'act')]}], line:2,
    msg:{zh:'埃氏篩不去「檢查」任何一個數是不是質數，而是反過來：從最小的質數開始，把它的倍數全部劃掉。沒被任何人劃到的就是質數。這樣每個合數只會被它的質因數碰到，不必對每個數做一次試除。',
         en:'The sieve of Eratosthenes never tests whether a number is prime. It works the other way round: starting from the smallest prime, cross out all of its multiples. Whatever nobody crossed out is prime. Each composite is only touched by its own prime factors, instead of every number paying for a round of trial division.'}});
  const lim = Math.floor(Math.sqrt(SN - 1));
  for (let p = 2; p <= lim; p++){
    if (st[p] === 'gone'){
      const hl = {}; hl[p] = 'soft';
      F.push({shapes:sieveStage(st, {hl:hl, cap:{zh:'p = ' + p + ' 已經被劃掉：跳過', en:'p = ' + p + ' is already crossed out: skip it'},
                                     cap2:{zh:p + ' 的倍數也是 2 的倍數，早就劃好了', en:'multiples of ' + p + ' are multiples of 2 and are already gone'}}),
        panels:[{lbl:'p', chips:[chip(String(p), 'dim')]}, {lbl:{zh:'質數', en:'primes'}, chips:primesNow().map(q => chip(String(q), 'ok'))}], line:4,
        msg:{zh:p + ' 是合數，所以 if is_prime[p] 不成立，整個跳過。它的每一個倍數都是 2 的倍數（' + p + ' = 2 × ' + (p / 2) + '），p = 2 那一輪全部劃過了。篩法只拿質數去劃，這是它比「每個數都試」省的第一個地方。',
             en:p + ' is composite, so if is_prime[p] fails and the round is skipped. Every multiple of ' + p + ' is a multiple of 2 (' + p + ' = 2 x ' + (p / 2) + ') and was crossed out in the p = 2 round. The sieve only ever crosses with primes - the first place it saves work over testing every number.'}});
      continue;
    }
    st[p] = 'prime';
    const hl = {}; hl[p] = 'act';
    for (let j = 2 * p; j < p * p; j += p) hl[j] = 'soft';
    F.push({shapes:sieveStage(st, {hl:hl, big:{zh:'p = ' + p + '：從 p × p = ' + (p * p) + ' 開始劃', en:'p = ' + p + ': start crossing at p x p = ' + (p * p)},
                                   cap:{zh:p + ' 沒被任何人劃到，所以它是質數', en:p + ' was never crossed out, so it is prime'},
                                   cap2:p > 2 ? {zh:'深色框的 ' + (2 * p) + ' … ' + (p * p - p) + ' 已經被更小的質數劃過了', en:'the dark cells ' + (2 * p) + ' … ' + (p * p - p) + ' were crossed by smaller primes'}
                                              : {zh:'比 4 小的倍數只有 2 自己', en:'the only multiple below 4 is 2 itself'}}),
      panels:[{lbl:'p', chips:[chip(String(p), 'act')]}, {lbl:{zh:'質數', en:'primes'}, chips:primesNow().map(q => chip(String(q), 'ok'))}], line:4,
      msg:p === 2
        ? {zh:'2 是第一個沒被劃掉的數，所以是質數。為什麼從 p × p = 4 開始而不是 2 × 2？這裡剛好一樣；到了 p = 3 就看得出差別：3 × 2 = 6 已經是 2 的倍數。一般來說，k × p 只要 k < p，k 就有一個比 p 小的質因數，那一輪早就劃過了。',
           en:'2 is the first number nobody crossed out, so it is prime. Why start at p x p = 4? For p = 2 that is just 2 x 2; the difference shows at p = 3, where 3 x 2 = 6 is already a multiple of 2. In general k x p with k < p has a prime factor smaller than p, whose round already crossed it out.'}
        : {zh:p + ' 沒被劃掉，所以是質數。它的倍數 ' + pyList(Array.from({length:p - 2}, (_, i) => p * (i + 2))) + ' 都比 ' + (p * p) + ' 小，每一個都是 k × ' + p + '、k < ' + p + '，k 的質因數比 ' + p + ' 小，早就劃過了。所以第一個要劃的是 ' + p + ' × ' + p + ' = ' + (p * p) + '。',
           en:p + ' was never crossed out, so it is prime. Its multiples ' + pyList(Array.from({length:p - 2}, (_, i) => p * (i + 2))) + ' are all below ' + (p * p) + '; each is k x ' + p + ' with k < ' + p + ', and k has a prime factor below ' + p + ' that already crossed it. So the first new crossing is ' + p + ' x ' + p + ' = ' + (p * p) + '.'}});
    const hl2 = {}, fresh = [], again = [];
    hl2[p] = 'act';
    for (let j = p * p; j < SN; j += p){ if (st[j] === 'gone'){ again.push(j); hl2[j] = 'soft'; } else { fresh.push(j); hl2[j] = 'hot'; } st[j] = 'gone'; }
    F.push({shapes:sieveStage(st, {hl:hl2, big:{zh:'is_prime[' + (p * p) + '::' + p + '] = 0：新劃掉 ' + fresh.length + ' 個', en:'is_prime[' + (p * p) + '::' + p + '] = 0: ' + fresh.length + ' newly crossed out'},
                                   cap:{zh:'橘色：這一輪第一次被劃掉；深色：別人已經劃過', en:'orange: crossed out for the first time; dark: someone got there first'},
                                   cap2:{zh:'整段用一次 bytearray 切片指定完成，不是 Python 迴圈', en:'the whole stride is one bytearray slice assignment, not a Python loop'}}),
      panels:[{lbl:'p', chips:[chip(String(p), 'act')]}, {lbl:{zh:'新劃掉', en:'newly crossed'}, chips:fresh.map(q => chip(String(q), 'hot'))},
              {lbl:{zh:'重複劃到', en:'hit again'}, chips:again.length ? again.map(q => chip(String(q), 'dim')) : [chip('—', 'empty')]}], line:5,
      msg:{zh:'p = ' + p + ' 劃掉 ' + (fresh.length + again.length) + ' 格，其中 ' + fresh.length + ' 格是第一次被劃到' + (again.length ? '，' + again.length + ' 格（' + again.join('、') + '）是更小的質數已經劃過、又被碰一次' : '') + '。從 p² 開始只能省掉一部分重複；n = 1,000,000 時從 p² 開始共劃 2,122,046 次，從 2p 開始 2,197,837 次，連合數也拿來劃就是 2,775,208 次。',
           en:'p = ' + p + ' crosses ' + (fresh.length + again.length) + ' cells; ' + fresh.length + ' are crossed for the first time' + (again.length ? ' and ' + again.length + ' (' + again.join(', ') + ') were already gone and get hit again' : '') + '. Starting at p² only removes part of the repeats: for n = 1,000,000 the sieve crosses 2,122,046 times starting at p², 2,197,837 times starting at 2p, and 2,775,208 times if composites cross too.'}});
  }
  const p8 = lim + 1, hl = {};
  for (let i = 2; i < SN; i++) if (st[i] !== 'gone') hl[i] = 'ok';
  for (let i = 2; i < SN; i++) if (st[i] !== 'gone') st[i] = 'prime';
  F.push({shapes:sieveStage(st, {hl:hl, big:{zh:'p = ' + p8 + '：' + p8 + ' × ' + p8 + ' = ' + (p8 * p8) + ' ≥ 50，停', en:'p = ' + p8 + ': ' + p8 + ' x ' + p8 + ' = ' + (p8 * p8) + ' >= 50, stop'}, bigc:COL.tealL,
                                 cap:{zh:'剩下沒被劃掉的 ' + primesNow().length + ' 個，全部是質數', en:'the ' + primesNow().length + ' numbers left standing are all prime'},
                                 cap2:{zh:'50 以下的合數一定有一個 ≤ 7 的因數', en:'every composite below 50 has a factor <= 7'}}),
    panels:[{lbl:{zh:'質數', en:'primes'}, chips:primesNow().map(q => chip(String(q), 'ok'))}], line:6,
    msg:{zh:'迴圈只跑到 √49 = 7。合數 c = a × b 裡，比較小的那個因數一定 ≤ √c；所以 50 以下的合數都有一個 ≤ 7 的因數，前面幾輪已經把它們劃光了，剩下的 ' + primesNow().length + ' 個就是答案。速度：數 100 萬以下的 78,498 個質數，逐一試除 5.01 秒，篩法寫成 Python 迴圈 213 ms，用 bytearray 切片 18.0 ms。',
         en:'The loop only runs up to √49 = 7. In a composite c = a x b the smaller factor is at most √c, so every composite below 50 has a factor <= 7 and the earlier rounds have crossed all of them out; the ' + primesNow().length + ' left are the answer. Speed: counting the 78,498 primes below 1,000,000 takes 5.01 s by trial division, 213 ms with the sieve written as Python loops and 18.0 ms with bytearray slices.'}});
  return F.list;
}

/* LC 204: a row 0..13, shaded below n, cell n itself marked */
const ROWN = 14;
function lcStage(n, o){
  const sh = [], x0 = 1.05, w = .55, h = .6, y = 1.0, isP = sieveArr(ROWN);
  sh.push(fitT(x0 - .15, y + .4, 'i', {c:COL.tealL, fs:.28, anchor:'end'}));
  for (let i = 0; i < ROWN; i++){
    let s = 'ghost';
    if (n != null){
      if (i < n) s = isP[i] ? 'ok' : 'soft';
      else if (i === n) s = isP[i] ? 'bad' : 'idle';
    }
    sh.push(S.r(x0 + i * w, y, w - .07, h, s, String(i), {fs:.26}));
  }
  if (n != null && n <= ROWN){
    const xn = x0 + n * w - .035;
    sh.push(S.e(xn, y - .2, xn, y + h + .2, {s:'act', arrow:false, w:.05}));
    sh.push(fitT(xn, y - .3, 'n = ' + n, {c:COL.purpleL, fs:.24}));
  }
  if (o.rows){
    o.rows.forEach((r, k) => {
      const yy = 2.35 + k * .78;
      sh.push(fitT(4.1, yy + .38, r.lab, {c:COL.tealL, fs:.26, anchor:'end'}));
      sh.push(box(4.3, yy, 1.1, .56, r.s, r.v, .3));
      if (r.note) sh.push(fitT(5.6, yy + .38, r.note, {c:r.nc || COL.grey, fs:.24, anchor:'start'}));
    });
  }
  if (o.big) sh.push(fitT(4.9, 4.78, o.big, {c:o.bigc || COL.orangeL, fs:.34}));
  return caps(sh, o);
}
function lc204Frames(){
  const F = new Frames();
  F.push({shapes:lcStage(null, {cap:{zh:'LeetCode 204：算出「嚴格小於 n」的質數有幾個', en:'LeetCode 204: count the primes strictly below n'},
                                cap2:{zh:'範例是 n = 10、0、1，答案 4、0、0', en:'the examples are n = 10, 0, 1 with answers 4, 0, 0'}}),
    panels:[{lbl:{zh:'範例', en:'examples'}, chips:[chip('10 → 4'), chip('0 → 0'), chip('1 → 0')]}], line:1,
    msg:{zh:'題目要的是 n 以下的質數，n 本身不算。sieve(n) 的陣列長度剛好是 n、索引 0 到 n - 1，所以 sum(sieve(n)) 就是答案；寫成「n 以內」的人會開 sieve(n + 1)，多數到 n 自己。兩種寫法只差在一格，接下來看它什麼時候會被抓到。',
         en:'The problem wants the primes below n; n itself does not count. sieve(n) has exactly n entries, indices 0 to n - 1, so sum(sieve(n)) is the answer. Code written as "up to n" uses sieve(n + 1) and counts n as well. The two differ by a single cell - the question is when anyone notices.'}});
  [10, 0, 1, 2, 13].forEach(n => {
    const a = countBelow(n), b = countUpto(n), same = a === b, ex = [10, 0, 1].indexOf(n) >= 0;
    F.push({shapes:lcStage(n, {rows:[{lab:'countPrimes(' + n + ')', v:String(a), s:'ok', note:{zh:'嚴格小於 n', en:'strictly below n'}},
                                     {lab:'count_primes_upto(' + n + ')', v:String(b), s:same ? 'ok' : 'bad',
                                      note:same ? {zh:'一樣：n 不是質數', en:'same: n is not prime'} : {zh:'多數了 n = ' + n + ' 自己', en:'counted n = ' + n + ' itself'}, nc:same ? COL.grey : COL.red}],
                               cap:ex ? {zh:'範例 n = ' + n + '：兩個版本都答 ' + a + '，都過', en:'example n = ' + n + ': both versions answer ' + a + ' and pass'}
                                      : {zh:'n = ' + n + ' 是質數：答案 ' + a + '，錯的版本答 ' + b, en:'n = ' + n + ' is prime: the answer is ' + a + ', the wrong version says ' + b},
                               capc:same ? COL.pale : COL.red,
                               cap2:{zh:'綠色是 n 以下的質數；n 那一格' + (same ? '不是質數' : '是質數，被多數了'), en:'teal: primes below n; the cell at n ' + (same ? 'is not prime' : 'is prime and gets counted')}}),
      panels:[{lbl:{zh:'嚴格小於 n', en:'below n'}, chips:[chip(String(a), 'ok')]}, {lbl:{zh:'n 以內', en:'up to n'}, chips:[chip(String(b), same ? 'ok' : 'bad')]}],
      line:same ? 2 : 5,
      msg:same
        ? (ex ? {zh:'n = ' + n + ' 是題目給的範例。' + (n === 10 ? '10 不是質數，' : n + ' 根本沒有質數，') + '所以「小於 n」和「n 以內」都是 ' + a + '。三個範例剛好都落在 n 不是質數的地方，錯的版本全部通過——範例沒辦法替你檢查邊界。',
                    en:'n = ' + n + ' is one of the given examples. ' + (n === 10 ? '10 is not prime' : 'there are no primes this small') + ', so "below n" and "up to n" both give ' + a + '. All three examples sit where n is not prime, so the wrong version passes every one - the examples cannot check the boundary for you.'}
              : {zh:'n = ' + n + ' 不是質數，兩個版本一致。', en:'n = ' + n + ' is not prime, so both versions agree.'})
        : {zh:'n = ' + n + ' 本身是質數，「n 以內」的版本把它也數進去，答 ' + b + '；正確答案是 ' + a + '。這個錯只在 n 剛好是質數時出現，所以範例完全抓不到，送出後才在隱藏測資上 Wrong Answer。',
           en:'n = ' + n + ' is itself prime, so the "up to n" version counts it and answers ' + b + '; the right answer is ' + a + '. The mistake only appears when n happens to be prime, so the examples never trigger it - it surfaces as Wrong Answer on the hidden tests.'}});
  });
  let wrong = 0; const isP = sieveArr(5001); let primes = 0;
  for (let n = 0; n <= 5000; n++){ if (countBelowFast(n, isP) !== countUptoFast(n, isP)) wrong++; if (isP[n]) primes++; }
  F.push({shapes:lcStage(13, {big:{zh:'n = 0 … 5000：錯 ' + fmt(wrong) + ' 次 = 5000 以內的 ' + fmt(primes) + ' 個質數', en:'n = 0 … 5000: wrong ' + fmt(wrong) + ' times = the ' + fmt(primes) + ' primes up to 5000'},
                              cap:{zh:'每一個錯都發生在 n 是質數的地方，一個不多一個不少', en:'every failure is at a prime n - no more, no fewer'},
                              cap2:{zh:'題目的三個範例 10、0、1 沒有一個是質數', en:'none of the three examples 10, 0, 1 is prime'}}),
    panels:[{lbl:{zh:'錯幾次', en:'failures'}, chips:[chip(fmt(wrong), 'bad')]}, {lbl:{zh:'≤ 5000 的質數', en:'primes <= 5000'}, chips:[chip(fmt(primes), 'ok')]}], line:5,
    msg:{zh:'把 n 從 0 掃到 5000，「n 以內」的版本錯 ' + fmt(wrong) + ' 次，恰好等於 5000 以內的質數個數 ' + fmt(primes) + '。邊界題的通則：自己補幾個「答案剛好落在邊界上」的測資——這裡就是 n = 2、3、5、13 這種 n 本身是質數的情況。',
         en:'Sweeping n from 0 to 5000, the "up to n" version is wrong ' + fmt(wrong) + ' times, exactly the number of primes up to 5000, ' + fmt(primes) + '. The general rule for boundary problems: add your own tests where the answer sits on the boundary - here, n = 2, 3, 5, 13, where n itself is prime.'}});
  return F.list;
}
function countBelowFast(n, isP){ let c = 0; for (let i = 0; i < n; i++) c += isP[i]; return n > 2 ? c : 0; }
function countUptoFast(n, isP){ let c = 0; for (let i = 0; i <= n; i++) c += isP[i]; return n >= 2 ? c : 0; }

/* ================================================================ tab 2: fast power */
const CODE_POW = [
  'def power_mod(a, e, m):',
  '    result = 1 % m',
  '    a %= m',
  '    while e:',
  '        if e & 1:               # this bit is on: multiply its power in',
  '            result = result * a % m',
  '        a = a * a % m           # a, a^2, a^4, a^8 ... one squaring per bit',
  '        e >>= 1',
  '    return result'
];
const CODE_50 = [
  'def myPow(self, x, n):',
  '    if n < 0:',
  '        x, n = 1 / x, -n        # in Python -n never overflows',
  '    result = 1.0',
  '    while n:',
  '        if n & 1:',
  '            result *= x',
  '        x *= x',
  '        n >>= 1',
  '    return result'
];
/* run square-and-multiply once and record one row per bit */
function powTrace(a, e, m){
  const rows = []; let res = m == null ? 1.0 : 1 % m, x = m == null ? a : a % m, k = 0;
  while (e){
    const bit = e & 1;
    if (bit) res = m == null ? res * x : res * x % m;
    rows.push({k:k, bit:bit, apow:x, res:res});
    x = m == null ? x * x : x * x % m; e >>= 1; k++;
  }
  return rows;
}
const fl = v => (Number.isInteger(v) ? v.toFixed(1) : String(+v.toFixed(6)));
function powStage(e, rows, upto, o){
  const sh = [], nb = Math.max(1, e.toString(2).length), w = .7, bx0 = 4.9 - nb * w / 2;
  sh.push(fitT(bx0 - .2, .88, o.elab, {c:COL.tealL, fs:.28, anchor:'end'}));
  for (let k = nb - 1; k >= 0; k--){
    const b = (e >> k) & 1, x = bx0 + (nb - 1 - k) * w;
    const s = k === upto ? 'hot' : k < upto ? (b ? 'ok' : 'done') : (b ? 'soft' : 'ghost');
    sh.push(S.r(x, .5, w - .08, .58, s, String(b), {fs:.32}));
    sh.push(S.t(x + (w - .08) / 2, .38, 'bit ' + k, {c:COL.grey, fs:.18}));
  }
  const cx = [1.3, 2.6, 4.25, 6.25, 8.05], cw = [.9, .9, 1.7, 1.7, 1.4];
  const hd = ['k', 'bit', o.alab, 'result', ''];
  hd.forEach((t, i) => { if (t) sh.push(fitT(cx[i] + cw[i] / 2, 1.62, t, {c:COL.tealL, fs:.24})); });
  rows.forEach((r, i) => {
    if (i > upto) return;
    const y = 1.8 + i * .62, cur = i === upto;
    sh.push(box(cx[0], y, cw[0], .5, cur ? 'act' : 'soft', String(r.k), .26));
    sh.push(box(cx[1], y, cw[1], .5, r.bit ? (cur ? 'hot' : 'ok') : 'done', String(r.bit), .26));
    sh.push(box(cx[2], y, cw[2], .5, cur ? 'act' : 'soft', o.fv(r.apow), .26));
    sh.push(box(cx[3], y, cw[3], .5, r.bit ? (cur ? 'hot' : 'ok') : 'done', o.fv(r.res), .26));
    sh.push(fitT(cx[4], y + .34, r.bit ? {zh:'← 乘進來', en:'<- multiply in'} : {zh:'跳過', en:'skip'},
                 {c:r.bit ? COL.orangeL : COL.grey, fs:.22, anchor:'start'}));
  });
  if (o.big) sh.push(fitT(4.9, 4.82, o.big, {c:o.bigc || COL.orangeL, fs:.34}));
  return caps(sh, o);
}
function powFrames(v){
  return v === 1 ? myPowFrames() : pmFrames();
}
function pmFrames(){
  const F = new Frames(), A = 3, E = 13, M = 100, rows = powTrace(A, E, M);
  const o0 = {elab:'e = 13', alab:'a^(2^k) % 100', fv:String};
  F.push({shapes:powStage(E, rows, -1, Object.assign({}, o0, {
      cap:{zh:'3^13 mod 100：13 = 0b1101 = 8 + 4 + 1', en:'3^13 mod 100: 13 = 0b1101 = 8 + 4 + 1'},
      cap2:{zh:'所以 3^13 = 3^8 × 3^4 × 3^1，只需要 3 的 1、2、4、8 次方', en:'so 3^13 = 3^8 x 3^4 x 3^1 - only the 1st, 2nd, 4th and 8th powers of 3 are needed'}})),
    panels:[{lbl:'a', chips:[chip('3', 'act')]}, {lbl:'e', chips:[chip('13', 'act'), chip('0b1101')]}, {lbl:'m', chips:[chip('100')]}], line:1,
    msg:{zh:'照定義乘 12 次 3 也能算，但指數是 10 億就要乘 10 億次。快速冪的想法：把指數寫成二進位，13 = 8 + 4 + 1，3^13 就等於 3^8 × 3^4 × 3^1。3 的 1、2、4、8 次方每一個都是前一個平方，所以每個 bit 只要平方一次，遇到 1 再多乘一次。',
         en:'Multiplying 3 by itself 12 times works, but an exponent of a billion would need a billion multiplications. Fast power writes the exponent in binary instead: 13 = 8 + 4 + 1, so 3^13 = 3^8 x 3^4 x 3^1. Each of 3^1, 3^2, 3^4, 3^8 is the square of the one before, so every bit costs one squaring, plus one multiplication when the bit is 1.'}});
  rows.forEach((r, i) => {
    const prev = i ? rows[i - 1].res : 1;
    F.push({shapes:powStage(E, rows, i, Object.assign({}, o0, {
        big:r.bit ? 'result = ' + prev + ' x ' + r.apow + ' % 100 = ' + r.res : {zh:'bit ' + r.k + ' 是 0：result 不動，仍是 ' + r.res, en:'bit ' + r.k + ' is 0: result stays ' + r.res},
        bigc:r.bit ? COL.orangeL : COL.grey,
        cap:{zh:'第 ' + r.k + ' 位：a = 3^' + (1 << r.k) + ' mod 100 = ' + r.apow, en:'bit ' + r.k + ': a = 3^' + (1 << r.k) + ' mod 100 = ' + r.apow},
        cap2:i === 0 ? {zh:'每一步都先 % 100，數字永遠不超過兩位數', en:'everything is reduced % 100 at every step, so no number ever exceeds two digits'}
                     : {zh:'a 是上一列的平方：' + rows[i - 1].apow + '² = ' + rows[i - 1].apow * rows[i - 1].apow + '，mod 100 = ' + r.apow, en:'a is the previous row squared: ' + rows[i - 1].apow + '² = ' + rows[i - 1].apow * rows[i - 1].apow + ', mod 100 = ' + r.apow}})),
      panels:[{lbl:'a', chips:[chip(String(r.apow), 'act')]}, {lbl:'result', chips:[chip(String(r.res), r.bit ? 'hot' : '')]},
              {lbl:{zh:'剩下的 e', en:'e left'}, chips:[chip(String(E >> (r.k + 1)) + ' = 0b' + ((E >> (r.k + 1)).toString(2)), '')]}],
      line:r.bit ? 5 : 6,
      msg:r.bit
        ? {zh:'這一位是 1，代表 3^' + (1 << r.k) + ' 是 13 的組成之一，乘進 result：' + prev + ' × ' + r.apow + ' = ' + prev * r.apow + '，取餘數得 ' + r.res + '。可以先取餘數再乘，是因為 (x × y) % m = ((x % m) × (y % m)) % m——乘法和加減法都保得住 %，除法保不住，後面兩個分頁會看到。',
           en:'This bit is 1, so 3^' + (1 << r.k) + ' is part of 13 and gets multiplied in: ' + prev + ' x ' + r.apow + ' = ' + prev * r.apow + ', remainder ' + r.res + '. Reducing before multiplying is allowed because (x * y) % m = ((x % m) * (y % m)) % m - multiplication, addition and subtraction all survive %; division does not, as the later tabs show.'}
        : {zh:'這一位是 0：13 裡沒有 ' + (1 << r.k) + '，所以 3^' + (1 << r.k) + ' 不乘進來。但 a 照樣要平方，因為下一位需要 3^' + (2 << r.k) + '。每一位固定一次平方，這就是為什麼工作量跟著 bit 數走，而不是跟著指數的大小走。',
           en:'This bit is 0: 13 contains no ' + (1 << r.k) + ', so 3^' + (1 << r.k) + ' is not multiplied in. a is still squared, because the next bit needs 3^' + (2 << r.k) + '. One squaring per bit, always - which is why the work grows with the number of bits, not with the exponent itself.'}});
  });
  const last = rows[rows.length - 1].res, full = 3n ** 13n;
  F.push({shapes:powStage(E, rows, rows.length, Object.assign({}, o0, {
      big:'3**13 = ' + fmt(full) + ',  % 100 = ' + (full % 100n) + '  ✓', bigc:COL.tealL,
      cap:{zh:'乘法次數：13 是 ' + mults(13) + ' 次（直接乘 12 次）；10 億是 ' + mults(1000000000) + ' 次', en:'multiplications: ' + mults(13) + ' for 13 (12 the naive way); ' + mults(1000000000) + ' for a billion'},
      cap2:{zh:'10^18 只要 ' + mults(10n ** 18n) + ' 次——直接乘要一百京次', en:'10^18 takes only ' + mults(10n ** 18n) + ' - the naive way needs a billion billion'}})),
    panels:[{lbl:'result', chips:[chip(String(last), 'ok')]}, {lbl:{zh:'乘法次數', en:'multiplications'}, chips:[chip(String(mults(13)), 'ok'), chip('naive: 12', 'dim')]}], line:8,
    msg:{zh:'答案 ' + last + '，和 3**13 = 1,594,323 取餘數一樣。次數 = (位元數 − 1) 次平方 + (1 的個數 − 1) 次乘法：13 是 3 + 2 = 5 次，10 億是 ' + mults(1000000000) + ' 次，10^18 是 ' + mults(10n ** 18n) + ' 次。Python 內建的 pow(a, e, m) 就是這個演算法：3 的一百萬次方先算再取餘數要 102.54 ms（中間那個數有 1,584,963 個 bit），pow(3, 1000000, MOD) 只要 2.3 µs。',
         en:'The answer is ' + last + ', the same as 1,594,323 % 100. The count is (bits - 1) squarings plus (ones - 1) multiplications: 3 + 2 = 5 for 13, ' + mults(1000000000) + ' for a billion, ' + mults(10n ** 18n) + ' for 10^18. Python\'s built-in pow(a, e, m) is this algorithm: computing 3 to the millionth power and then taking the remainder costs 102.54 ms (the intermediate number has 1,584,963 bits), while pow(3, 1000000, MOD) takes 2.3 µs.'}});
  return F.list;
}
function myPowFrames(){
  const F = new Frames(), X = 2.0, N = -2, rows = powTrace(1 / X, -N, null);
  const o0 = {elab:'n = 2', alab:'x^(2^k)', fv:fl};
  F.push({shapes:powStage(-N, rows, -1, Object.assign({}, o0, {
      big:'myPow(2.0, -2)', bigc:COL.purpleL,
      cap:{zh:'LeetCode 50：n 是負數時先換成 (1/x)^(-n)', en:'LeetCode 50: for negative n, compute (1/x)^(-n) instead'},
      cap2:{zh:'x = 1 / 2.0 = 0.5，n = 2 = 0b10', en:'x = 1 / 2.0 = 0.5, n = 2 = 0b10'}})),
    panels:[{lbl:'x', chips:[chip('0.5', 'act')]}, {lbl:'n', chips:[chip('2', 'act')]}], line:2,
    msg:{zh:'LeetCode 50 要算 x 的 n 次方，n 的範圍是 -2³¹ 到 2³¹ - 1，所以必須用快速冪。負指數先翻成 (1/x)^(-n)。在 C++ 和 Java 裡 -n 對 n = -2³¹ 會溢位（2³¹ 放不進 int），要先轉成 long；Python 的整數沒有寬度，這一行直接就對。',
         en:'LeetCode 50 asks for x to the power n with n anywhere from -2^31 to 2^31 - 1, so fast power is required. A negative exponent becomes (1/x)^(-n). In C++ and Java, -n overflows for n = -2^31 (2^31 does not fit in an int) and has to go through a long first; Python ints have no width, so this line is simply correct.'}});
  rows.forEach((r, i) => {
    F.push({shapes:powStage(-N, rows, i, Object.assign({}, o0, {
        big:r.bit ? 'result = ' + fl(r.res) : {zh:'bit ' + r.k + ' 是 0：result 不動', en:'bit ' + r.k + ' is 0: result stays put'},
        bigc:r.bit ? COL.orangeL : COL.grey,
        cap:{zh:'第 ' + r.k + ' 位：x = ' + fl(r.apow), en:'bit ' + r.k + ': x = ' + fl(r.apow)},
        cap2:{zh:'沒有模數，這裡只是浮點數的平方與相乘', en:'no modulus here - just floating-point squaring and multiplying'}})),
      panels:[{lbl:'x', chips:[chip(fl(r.apow), 'act')]}, {lbl:'result', chips:[chip(fl(r.res), r.bit ? 'hot' : '')]}], line:r.bit ? 6 : 7,
      msg:r.bit ? {zh:'這一位是 1，乘進去：result = ' + fl(r.res) + '。2 的 -2 次方就是 1/4 = 0.25。', en:'This bit is 1, so it is multiplied in: result = ' + fl(r.res) + '. 2 to the -2 is 1/4 = 0.25.'}
                : {zh:'這一位是 0，result 不動；x 照樣平方成 ' + fl(r.apow * r.apow) + '，留給下一位用。', en:'This bit is 0, so result stays; x is still squared to ' + fl(r.apow * r.apow) + ' for the next bit.'}});
  });
  F.push({shapes:powStage(-N, rows, rows.length, Object.assign({}, o0, {
      big:'myPow(2.0, -2) = ' + fl(rows[rows.length - 1].res), bigc:COL.tealL,
      cap:{zh:'n = -2³¹：直接乘要 2,147,483,648 次，快速冪 ' + mults(2n ** 31n) + ' 次', en:'n = -2^31: 2,147,483,648 multiplications the naive way, ' + mults(2n ** 31n) + ' with fast power'},
      cap2:{zh:'迴圈寫法逐一相乘在 LeetCode 上一定超時', en:'the multiply-one-at-a-time loop always times out on LeetCode'}})),
    panels:[{lbl:'result', chips:[chip(fl(rows[rows.length - 1].res), 'ok')]}], line:9,
    msg:{zh:'答案 0.25。n = -2³¹ 時指數翻正是 2³¹，二進位是一個 1 後面 31 個 0：31 次平方就到了，直接乘要 2,147,483,648 次。這題和前一個變體是同一個迴圈，差別只在沒有 % m——快速冪對任何「可以結合的乘法」都成立：整數、浮點數、模數、矩陣（Fibonacci 的 F(10^18) mod 1e9+7 只要 83 次矩陣乘法）。',
         en:'The answer is 0.25. For n = -2^31 the flipped exponent is 2^31, a 1 followed by 31 zeros in binary: 31 squarings get there, versus 2,147,483,648 multiplications one at a time. This is the same loop as the other variant without the % m - fast power works for any associative multiplication: integers, floats, residues, matrices (Fibonacci F(10^18) mod 1e9+7 takes 83 matrix multiplications).'}});
  return F.list;
}

/* ================================================================ tab 3: GCD */
const CODE_GCD = [
  'def gcd(a, b):',
  '    while b:',
  '        a, b = b, a % b',
  '    return abs(a)'
];
const CODE_EXT = [
  'def ext_gcd(a, b):            # (g, x, y) with a*x + b*y == g',
  '    x0, y0, x1, y1 = 1, 0, 0, 1',
  '    while b:',
  '        q, a, b = a // b, b, a % b',
  '        x0, x1 = x1, x0 - q * x1',
  '        y0, y1 = y1, y0 - q * y1',
  '    return a, x0, y0'
];
const CODE_365 = [
  'class Solution:',
  '    def canMeasureWater(self, x, y, target):',
  '        return target <= x + y and target % math.gcd(x, y) == 0'
];
function euclidSteps(a, b){ const st = []; while (b){ st.push({a:a, b:b, q:Math.floor(a / b), r:a % b}); [a, b] = [b, a % b]; } return {g:a, st:st}; }
/* same count as gcd_subtract_steps: each a % b stands for a // b subtractions, minus the last one that hits 0 */
function subtractSteps(a, b){ let n = 0; while (b){ n += Math.floor(a / b); [a, b] = [b, a % b]; } return n - 1; }
function gcdStage(eu, upto, o){
  const sh = [], s = 7.4 / eu.st[0].a, x0 = 1.5;
  const cur = eu.st[Math.min(Math.max(upto, 0), eu.st.length - 1)];
  if (upto >= 0 && upto < eu.st.length){
    sh.push(fitT(x0 - .15, .92, 'a', {c:COL.tealL, fs:.28, anchor:'end'}));
    sh.push(box(x0, .55, cur.a * s, .55, 'idle', String(cur.a), .28));
    sh.push(fitT(x0 - .15, 1.67, 'b', {c:COL.tealL, fs:.28, anchor:'end'}));
    for (let i = 0; i < cur.q; i++) sh.push(box(x0 + i * cur.b * s, 1.3, cur.b * s - .03, .55, 'act', String(cur.b), .26));
    if (cur.r) sh.push(box(x0 + cur.q * cur.b * s, 1.3, cur.r * s - .03, .55, 'hot', String(cur.r), .26));
  }
  eu.st.forEach((t, i) => {
    if (i > upto) return;
    const y = 2.55 + i * .55;
    sh.push(fitT(4.9, y, fmt(t.a) + ' = ' + t.q + ' x ' + fmt(t.b) + ' + ' + fmt(t.r),
                 {c:i === upto ? COL.orangeL : COL.pale, fs:.3}));
  });
  if (o.big) sh.push(fitT(4.9, 4.82, o.big, {c:o.bigc || COL.orangeL, fs:.36}));
  return caps(sh, o);
}
function gcdFrames(v){
  return v === 1 ? extFrames() : v === 2 ? jugFrames() : euFrames();
}
function euFrames(){
  const F = new Frames(), A = 252, B = 105, eu = euclidSteps(A, B);
  F.push({shapes:gcdStage(eu, -1, {big:'gcd(252, 105) = ?', bigc:COL.purpleL,
                                   cap:{zh:'最大公因數：同時整除 a 和 b 的最大整數', en:'greatest common divisor: the largest integer dividing both a and b'},
                                   cap2:{zh:'不必分解質因數；Euclid 只用除法的餘數', en:'no factoring needed - Euclid only uses remainders'}}),
    panels:[{lbl:'a', chips:[chip('252', 'act')]}, {lbl:'b', chips:[chip('105', 'act')]}], line:0,
    msg:{zh:'分解質因數可以算 gcd，但大數分解非常難。Euclid 的演算法只用一個事實：任何同時整除 a 和 b 的數，也整除 a - q × b；反過來也成立。所以 gcd(a, b) = gcd(b, a % b)，一直換到餘數是 0。',
         en:'Factoring both numbers gives the gcd, but factoring large numbers is very hard. Euclid relies on one fact: anything that divides both a and b also divides a - q x b, and the other way round. So gcd(a, b) = gcd(b, a % b), repeated until the remainder is 0.'}});
  eu.st.forEach((t, i) => {
    F.push({shapes:gcdStage(eu, i, {cap:{zh:'用 ' + t.b + ' 去量 ' + t.a + '：放得下 ' + t.q + ' 段，剩 ' + t.r, en:'measure ' + t.a + ' with ' + t.b + ': it fits ' + t.q + ' times, ' + t.r + ' left over'},
                                     capc:t.r ? COL.pale : COL.tealL,
                                     cap2:t.r ? {zh:'下一輪換成 gcd(' + t.b + ', ' + t.r + ')', en:'the next round is gcd(' + t.b + ', ' + t.r + ')'}
                                              : {zh:'餘數 0：' + t.b + ' 剛好量完 ' + t.a, en:'remainder 0: ' + t.b + ' measures ' + t.a + ' exactly'}}),
      panels:[{lbl:'a', chips:[chip(String(t.a), 'act')]}, {lbl:'b', chips:[chip(String(t.b), 'act')]}, {lbl:'a % b', chips:[chip(String(t.r), t.r ? 'hot' : 'ok')]}], line:2,
      msg:t.r
        ? {zh:t.a + ' = ' + t.q + ' × ' + t.b + ' + ' + t.r + '。整除 ' + t.a + ' 和 ' + t.b + ' 的數，一定也整除 ' + t.a + ' - ' + t.q + ' × ' + t.b + ' = ' + t.r + '；整除 ' + t.b + ' 和 ' + t.r + ' 的數也一定整除 ' + t.a + '。兩對數的公因數一模一樣，所以可以把 (' + t.a + ', ' + t.b + ') 換成比較小的 (' + t.b + ', ' + t.r + ')。',
           en:t.a + ' = ' + t.q + ' x ' + t.b + ' + ' + t.r + '. Anything dividing ' + t.a + ' and ' + t.b + ' also divides ' + t.a + ' - ' + t.q + ' x ' + t.b + ' = ' + t.r + ', and anything dividing ' + t.b + ' and ' + t.r + ' also divides ' + t.a + '. The two pairs have exactly the same common divisors, so (' + t.a + ', ' + t.b + ') can be swapped for the smaller (' + t.b + ', ' + t.r + ').'}
        : {zh:t.b + ' 剛好整除 ' + t.a + '，所以 gcd(' + t.a + ', ' + t.b + ') = ' + t.b + '，也就是一開始 gcd(252, 105) 的答案。迴圈下一圈 b 是 0，回傳 a。',
           en:t.b + ' divides ' + t.a + ' exactly, so gcd(' + t.a + ', ' + t.b + ') = ' + t.b + ', which is also the answer for gcd(252, 105). On the next pass b is 0 and the loop returns a.'}});
  });
  const fib = euclidSteps(987, 610).st.length;
  F.push({shapes:gcdStage(eu, eu.st.length, {big:'gcd(252, 105) = ' + eu.g + '   (' + eu.st.length + ' steps)', bigc:COL.tealL,
                                             cap:{zh:'用減法要 ' + subtractSteps(A, B) + ' 次；gcd(1,000,000, 1) 用減法要 ' + fmt(subtractSteps(1000000, 1)) + ' 次，除法 1 次', en:'by subtraction: ' + subtractSteps(A, B) + ' steps; gcd(1,000,000, 1) by subtraction: ' + fmt(subtractSteps(1000000, 1)) + ', by division: 1'},
                                             cap2:{zh:'1000 以內最慢的是 gcd(987, 610)：' + fib + ' 步，兩個相鄰的 Fibonacci 數', en:'the slowest pair up to 1000 is gcd(987, 610): ' + fib + ' steps, two consecutive Fibonacci numbers'}}),
    panels:[{lbl:'gcd', chips:[chip(String(eu.g), 'ok')]}, {lbl:{zh:'步數', en:'steps'}, chips:[chip(String(eu.st.length), 'ok')]}], line:3,
    msg:{zh:'答案 ' + eu.g + '。為什麼快：每兩步 a 至少砍半（a % b 不是小於 b，就是小於 a 的一半），所以步數只跟位數成正比。最壞的情況是每次商都是 1，也就是相鄰的 Fibonacci 數：1000 以內最慢的是 gcd(987, 610)，' + fib + ' 步；兩個 1000 位數的隨機數也只要 1875 步。原始的「大減小」版本則可能要一百萬步。',
         en:'The answer is ' + eu.g + '. Why it is fast: every two steps a at least halves (a % b is below b and also below half of a), so the step count grows with the number of digits. The worst case is a quotient of 1 every time - consecutive Fibonacci numbers: the slowest pair up to 1000 is gcd(987, 610) at ' + fib + ' steps, and two random 1000-digit numbers take 1875. The original "subtract the smaller" version can need a million steps.'}});
  return F.list;
}

/* extended Euclid: the same loop, carrying x, y with a == A*x + B*y on every row */
function extTrace(a, b){
  const A = a, B = b, rows = [{a:a, b:b, q:null, x:1, y:0}];
  let x0 = 1, y0 = 0, x1 = 0, y1 = 1;
  while (b){
    const q = Math.floor(a / b); [a, b] = [b, a % b];
    [x0, x1] = [x1, x0 - q * x1]; [y0, y1] = [y1, y0 - q * y1];
    rows.push({a:a, b:b, q:q, x:x0, y:y0});
  }
  return {A:A, B:B, rows:rows};
}
function extStage(T, upto, o){
  const sh = [], cx = [.75, 1.85, 2.95, 4.05, 5.15, 6.4], cw = [1.0, 1.0, 1.0, 1.0, 1.0, 2.9];
  const hd = ['a', 'b', 'q', 'x', 'y', T.A + 'x + ' + T.B + 'y'];
  hd.forEach((t, i) => sh.push(fitT(cx[i] + cw[i] / 2, .62, t, {c:i >= 3 && i <= 4 ? COL.purpleL : COL.tealL, fs:.26})));
  T.rows.forEach((r, i) => {
    if (i > upto) return;
    const y = .8 + i * .6, cur = i === upto, last = r.b === 0;
    const vals = [r.a, r.b, r.q == null ? '' : r.q, r.x, r.y];
    vals.forEach((v, j) => sh.push(box(cx[j], y, cw[j] - .08, .5, cur ? (j >= 3 ? 'act' : 'hot') : (last && j === 0 ? 'ok' : 'soft'), String(v), .26)));
    const chk = T.A * r.x + T.B * r.y;
    sh.push(box(cx[5], y, cw[5], .5, chk === r.a ? (cur ? 'ok' : 'soft') : 'bad',
                T.A + '·' + (r.x < 0 ? '(' + r.x + ')' : r.x) + ' + ' + T.B + '·' + (r.y < 0 ? '(' + r.y + ')' : r.y) + ' = ' + chk, .24));
  });
  if (o.big) sh.push(fitT(4.9, 4.82, o.big, {c:o.bigc || COL.orangeL, fs:.36}));
  return caps(sh, o);
}
function extFrames(){
  const F = new Frames(), T = extTrace(240, 46), R = T.rows;
  R.forEach((r, i) => {
    const last = r.b === 0;
    F.push({shapes:extStage(T, i, {big:last ? 'gcd = ' + r.a + ' = 240 x (' + r.x + ') + 46 x ' + r.y : null, bigc:COL.tealL,
                                   cap:i === 0 ? {zh:'起點：240 = 240·1 + 46·0，46 = 240·0 + 46·1', en:'start: 240 = 240·1 + 46·0, and 46 = 240·0 + 46·1'}
                                     : last ? {zh:'b = 0：gcd 是 ' + r.a + '，那一列的 x、y 就是答案', en:'b = 0: the gcd is ' + r.a + ', and that row\'s x, y are the answer'}
                                     : {zh:'q = ' + r.q + '：新的一列 = 上上列 − ' + r.q + ' × 上一列', en:'q = ' + r.q + ': new row = the row two up - ' + r.q + ' x the previous row'},
                                   cap2:{zh:'右欄每一列都驗算：240x + 46y 一定等於那一列的 a', en:'the right column checks every row: 240x + 46y always equals that row\'s a'}}),
      panels:[{lbl:'a, b', chips:[chip(String(r.a), 'hot'), chip(String(r.b), '')]}, {lbl:'x, y', chips:[chip(String(r.x), 'act'), chip(String(r.y), 'act')]}],
      line:i === 0 ? 1 : last ? 6 : 4,
      msg:i === 0
        ? {zh:'擴展 Euclid 跑的是同一個迴圈，只是多記兩個係數：每一列都維持「a = 240x + 46y」。第一列 a 是 240 本身，所以 x = 1、y = 0；第二個數 46 則是 x = 0、y = 1。',
           en:'Extended Euclid runs the very same loop and carries two extra coefficients, keeping "a = 240x + 46y" true on every row. On the first row a is 240 itself, so x = 1, y = 0; the other number, 46, is x = 0, y = 1.'}
        : last
        ? {zh:'餘數變 0，gcd = ' + r.a + '，而這一列的係數給出 2 = 240 × (' + r.x + ') + 46 × ' + r.y + '。這就是 Bézout 等式：gcd 一定能寫成 a、b 的整數組合。反過來，a、b 的任何整數組合都是 gcd 的倍數——LeetCode 365 的倒水題整題就靠這一句。',
           en:'The remainder is 0, gcd = ' + r.a + ', and this row\'s coefficients give 2 = 240 x (' + r.x + ') + 46 x ' + r.y + '. That is Bézout\'s identity: the gcd can always be written as an integer combination of a and b. Conversely, every integer combination of a and b is a multiple of the gcd - LeetCode 365, the water jug problem, rests entirely on that sentence.'}
        : {zh:'新的 b 是 ' + R[i - 1].a + ' - ' + r.q + ' × ' + R[i - 1].b + '。因為上兩列各自都是 240x + 46y 的形式，相減之後也還是：x 跟著做 ' + R[i - 1].x + ' - ' + r.q + ' × (下一列的 x)，y 也一樣。右欄的驗算每一列都成立，這不是巧合，是這個迴圈的不變量。',
           en:'The new b is ' + R[i - 1].a + ' - ' + r.q + ' x ' + R[i - 1].b + '. Since the two rows above are both of the form 240x + 46y, so is their difference: x follows the same subtraction, and so does y. The check in the right column holds on every row - not by luck, it is the loop invariant.'}});
  });
  return F.list;
}

/* LC 365: BFS over (jug1, jug2) exactly like can_measure_bfs, keeping parents */
function jugPath(x, y, t){
  const key = s => s[0] + ',' + s[1], par = new Map([[key([0, 0]), null]]), q = [[0, 0]];
  const names = [{zh:'裝滿 ' + x + ' 升', en:'fill the ' + x}, {zh:'裝滿 ' + y + ' 升', en:'fill the ' + y},
                 {zh:'倒空 ' + x + ' 升', en:'empty the ' + x}, {zh:'倒空 ' + y + ' 升', en:'empty the ' + y},
                 {zh:x + ' 倒進 ' + y, en:'pour the ' + x + ' into the ' + y}, {zh:y + ' 倒進 ' + x, en:'pour the ' + y + ' into the ' + x}];
  while (q.length){
    const [a, b] = q.shift();
    if (a + b === t){
      const path = []; let k = key([a, b]);
      while (k){ const [ka, kb] = k.split(',').map(Number), p = par.get(k); path.unshift({s:[ka, kb], mv:p ? names[p.m] : null}); k = p ? p.k : null; }
      return path;
    }
    const ab = Math.min(a, y - b), ba = Math.min(b, x - a);
    [[x, b], [a, y], [0, b], [a, 0], [a - ab, b + ab], [a + ba, b - ba]].forEach((s, m) => {
      if (!par.has(key(s))){ par.set(key(s), {k:key([a, b]), m:m}); q.push(s); }
    });
  }
  return null;
}
function jugStage(x, y, lv, o){
  const sh = [], u = 2.6 / Math.max(x, y), base = 3.75, w = 1.1, xs = [1.3, 3.0], caps2 = [x, y];
  caps2.forEach((c, j) => {
    const h = c * u, X = xs[j];
    sh.push(S.r(X, base - h, w, h, 'ghost', ''));
    if (lv && lv[j]) sh.push(S.r(X + .05, base - lv[j] * u, w - .1, lv[j] * u, o.hot === j ? 'hot' : 'ok', String(lv[j]), {fs:.3}));
    sh.push(fitT(X + w / 2, base + .38, {zh:c + ' 升', en:c + ' L'}, {c:COL.tealL, fs:.26}));
  });
  (o.lines || []).forEach((ln, i) => sh.push(fitT(4.75, .7 + i * .5, ln.t, {c:ln.c || COL.pale, fs:.26, anchor:'start'})));
  if (o.big) sh.push(fitT(4.9, 4.82, o.big, {c:o.bigc || COL.orangeL, fs:.34}));
  return caps(sh, o);
}
function jugFrames(){
  const F = new Frames(), path = jugPath(3, 5, 4);
  F.push({shapes:jugStage(3, 5, [0, 0], {big:'canMeasureWater(3, 5, 4)', bigc:COL.purpleL,
                                         cap:{zh:'LeetCode 365：兩個水壺 3 升和 5 升，能不能剛好量出 4 升？', en:'LeetCode 365: jugs of 3 and 5 litres - can we end up with exactly 4?'},
                                         cap2:{zh:'動作只有三種：裝滿、倒空、一壺倒進另一壺', en:'only three moves: fill one, empty one, pour one into the other'}}),
    panels:[{lbl:'x, y', chips:[chip('3'), chip('5')]}, {lbl:'target', chips:[chip('4', 'act')]}], line:1,
    msg:{zh:'暴力解是對所有 (壺 1, 壺 2) 的狀態做 BFS。但每個動作都只是加上或減掉整壺的 3 或 5 升，所以任何時候兩壺的總量都是 3 和 5 的整數組合——由 Bézout，這些組合正好是 gcd(3, 5) = 1 的倍數。先用 BFS 找一條真的倒法，再看一行的公式。',
         en:'Brute force runs BFS over every (jug 1, jug 2) state. But each move adds or removes a whole 3 or a whole 5, so the total in the two jugs is always an integer combination of 3 and 5 - and by Bézout those combinations are exactly the multiples of gcd(3, 5) = 1. First a real pouring sequence from the BFS, then the one-line formula.'}});
  const lines = [{t:'(0, 0)', c:COL.grey}];
  path.slice(1).forEach((p, i) => {
    const prev = path[i].s, hot = p.s[0] !== prev[0] ? 0 : 1, done = p.s[0] + p.s[1] === 4;
    lines.forEach(l => l.c = COL.grey);
    lines.push({t:{zh:(i + 1) + '. ' + p.mv.zh + ' → (' + p.s.join(', ') + ')', en:(i + 1) + '. ' + p.mv.en + ' -> (' + p.s.join(', ') + ')'}, c:done ? COL.tealL : COL.orangeL});
    F.push({shapes:jugStage(3, 5, p.s, {hot:done ? null : hot, lines:lines.map(l => Object.assign({}, l)),
                                        cap:done ? {zh:'總共 ' + (p.s[0] + p.s[1]) + ' 升：量到了，用了 ' + (i + 1) + ' 步', en:(p.s[0] + p.s[1]) + ' litres in total: done in ' + (i + 1) + ' moves'}
                                                 : {zh:'第 ' + (i + 1) + ' 步：' + p.mv.zh, en:'move ' + (i + 1) + ': ' + p.mv.en},
                                        capc:done ? COL.tealL : COL.pale,
                                        cap2:{zh:'兩壺合計 ' + (p.s[0] + p.s[1]) + ' 升 = 3 和 5 的整數組合', en:'total ' + (p.s[0] + p.s[1]) + ' litres = an integer combination of 3 and 5'}}),
      panels:[{lbl:{zh:'狀態', en:'state'}, chips:[chip('(' + p.s.join(', ') + ')', done ? 'ok' : 'hot')]}, {lbl:{zh:'合計', en:'total'}, chips:[chip(String(p.s[0] + p.s[1]), done ? 'ok' : '')]}], line:1,
      msg:done
        ? {zh:'BFS 找到的最短倒法是 ' + (path.length - 1) + ' 步，最後兩壺合計 4 升。4 = 3 × 3 − 5：倒法換成算式，就是 Bézout 等式的一個解。BFS 要走過所有狀態，最多 (x + 1)(y + 1) 個；x、y 到 10⁶ 時就是一兆個狀態，所以需要公式。',
           en:'BFS finds a shortest sequence of ' + (path.length - 1) + ' moves, ending with 4 litres in total. 4 = 3 x 3 - 5: written as arithmetic, the pouring is one solution of Bézout\'s identity. BFS may visit every state, up to (x + 1)(y + 1) of them - a trillion when x and y reach a million - which is why the formula matters.'}
        : {zh:p.mv.zh + '，狀態變成 (' + p.s.join(', ') + ')。倒水不會產生 3 和 5 以外的「新單位」：每一步都只是加減整壺的量，所以總量永遠是 3a + 5b 的形式。',
           en:p.mv.en + ', giving (' + p.s.join(', ') + '). Pouring never creates a unit other than 3 and 5: every move adds or removes whole jugfuls, so the total is always of the form 3a + 5b.'}});
  });
  [[2, 6, 5], [3, 5, 9]].forEach(([x, y, t]) => {
    const g = gcdN(x, y), divides = t % g === 0, fits = t <= x + y, ans = divides && fits;
    const lines = [{t:'gcd(' + x + ', ' + y + ') = ' + g, c:COL.pale},
                   {t:t + ' % ' + g + ' = ' + (t % g) + (divides ? '  ✓' : '  ✗'), c:divides ? COL.tealL : COL.red},
                   {t:t + ' <= ' + x + ' + ' + y + ' = ' + (x + y) + (fits ? '  ✓' : '  ✗'), c:fits ? COL.tealL : COL.red},
                   {t:{zh:'BFS 結果：' + (jugPath(x, y, t) ? 'True' : 'False'), en:'BFS says: ' + (jugPath(x, y, t) ? 'True' : 'False')}, c:COL.grey}];
    if (!fits) lines.push({t:{zh:'少了 target <= x + y：True（錯）', en:'without target <= x + y: True (wrong)'}, c:COL.red});
    F.push({shapes:jugStage(x, y, [x, y], {lines:lines, big:'canMeasureWater(' + x + ', ' + y + ', ' + t + ') = ' + (ans ? 'True' : 'False'), bigc:ans ? COL.tealL : COL.red,
                                           cap:!divides ? {zh:'gcd 是 ' + g + '：能量出的量全是 ' + g + ' 的倍數，' + t + ' 不是', en:'the gcd is ' + g + ': every reachable amount is a multiple of ' + g + ', and ' + t + ' is not'}
                                                        : {zh:'gcd 整除 ' + t + '，但兩壺裝滿也只有 ' + (x + y) + ' 升', en:'the gcd divides ' + t + ', but both jugs full hold only ' + (x + y) + ' litres'},
                                           capc:COL.red,
                                           cap2:{zh:'公式兩個條件缺一不可', en:'the formula needs both conditions'}}),
      panels:[{lbl:'gcd', chips:[chip(String(g), '')]}, {lbl:{zh:'答案', en:'answer'}, chips:[chip(ans ? 'True' : 'False', ans ? 'ok' : 'bad')]}], line:2,
      msg:!divides
        ? {zh:'(2, 6, 5)：gcd(2, 6) = 2，兩壺裡的水量永遠是偶數（每一步都加減 2 或 6 升），5 是奇數，怎麼倒都到不了。這就是 target % gcd == 0 那半句。',
           en:'(2, 6, 5): gcd(2, 6) = 2, so the water in the jugs is always even (every move adds or removes 2 or 6 litres) and 5, being odd, is never reachable. That is the target % gcd == 0 half.'}
        : {zh:'(3, 5, 9)：gcd 是 1，1 整除 9，Bézout 也說 9 = 3 × 3 + 5 × 0。但水要裝在兩個壺裡，兩壺全滿也只有 8 升。只寫 target % gcd == 0 的版本在這裡回傳 True——Bézout 只管整數組合存不存在，不管容量，容量那一半要自己加上 target <= x + y。',
           en:'(3, 5, 9): the gcd is 1, 1 divides 9, and Bézout even offers 9 = 3 x 3 + 5 x 0. But the water has to sit in the two jugs, and both full hold 8 litres. The version that only checks target % gcd == 0 returns True here - Bézout only says the integer combination exists, not that it fits, so target <= x + y has to be added by hand.'}});
  });
  return F.list;
}

/* ================================================================ tab 4: Fermat on a composite modulus */
const CODE_INV = [
  'def inverse_fermat(a, m):',
  '    return pow(a, m - 2, m)      # correct only when m is prime - nothing checks',
  '',
  'def inverse(a, m):                # extended Euclid: any m',
  '    g, x, _ = ext_gcd(a % m, m)',
  '    if g != 1:',
  "        raise ValueError(f'{a} has no inverse mod {m}')",
  '    return x % m'
];
function invTrue(a, m){ const T = extTrace(a % m, m), r = T.rows[T.rows.length - 1]; return r.a === 1 ? ((r.x % m) + m) % m : null; }
function fermatStage(m, upto, o){
  const sh = [], n = m - 1, w = Math.min(.76, 7.5 / n), x0 = 1.95;
  const rl = ['a', 'gcd(a, ' + m + ')', 'a^' + (m - 2) + ' % ' + m, {zh:'a × 上一列 % ' + m, en:'a x that % ' + m}, 'pow(a, -1, ' + m + ')'];
  rl.forEach((t, j) => sh.push(fitT(x0 - .12, .5 + j * .72 + .33, t, {c:j === 2 ? COL.orangeL : j === 4 ? COL.purpleL : COL.tealL, fs:.22, anchor:'end'})));
  for (let a = 1; a <= n; a++){
    const x = x0 + (a - 1) * w, cur = a - 1 === upto, seen = a - 1 <= upto;
    const g = gcdN(a, m), f = powModN(a, m - 2, m), prod = a * f % m, t = invTrue(a, m);
    sh.push(box(x, .5, w - .07, .5, cur ? 'act' : 'soft', String(a), .26));
    if (!seen) continue;
    sh.push(box(x, 1.22, w - .07, .5, g === 1 ? 'soft' : 'done', String(g), .26));
    sh.push(box(x, 1.94, w - .07, .5, cur ? 'hot' : 'soft', String(f), .26));
    sh.push(box(x, 2.66, w - .07, .5, prod === 1 ? 'ok' : 'bad', String(prod), .26));
    sh.push(box(x, 3.38, w - .07, .5, t == null ? 'ghost' : 'act', t == null ? 'Error' : String(t), .24));
  }
  if (o.big) sh.push(fitT(4.9, 4.72, o.big, {c:o.bigc || COL.orangeL, fs:.34}));
  return caps(sh, o);
}
function fermatFrames(v){
  const F = new Frames(), m = v === 1 ? 10 : 11, n = m - 1, prime = v === 0;
  F.push({shapes:fermatStage(m, -1, {big:prime ? {zh:'m = 11 是質數', en:'m = 11 is prime'} : {zh:'m = 10 是合數', en:'m = 10 is composite'}, bigc:prime ? COL.tealL : COL.red,
                                    cap:{zh:'反元素：找 x 使 a × x % m == 1；Fermat 說 x = a^(m-2) % m', en:'the inverse: x with a * x % m == 1; Fermat says x = a^(m-2) % m'},
                                    cap2:{zh:'第四列是驗算：真的反元素乘回去一定是 1', en:'row four is the check: a real inverse multiplies back to 1'}}),
    panels:[{lbl:'m', chips:[chip(String(m), prime ? 'ok' : 'bad')]}], line:1,
    msg:prime
      ? {zh:'% 保不住除法，所以模運算裡的「除以 a」要改成「乘上 a 的反元素」：一個 x 使 a × x ≡ 1。Fermat 小定理：p 是質數、a 不是 p 的倍數時，a^(p-1) ≡ 1，所以 a^(p-2) 就是反元素。只要一次快速冪，這是競賽程式最常見的寫法。',
         en:'% does not survive division, so "divide by a" in modular arithmetic becomes "multiply by a\'s inverse": an x with a * x ≡ 1. Fermat\'s little theorem: when p is prime and a is not a multiple of p, a^(p-1) ≡ 1, so a^(p-2) is the inverse. One fast power - the most common recipe in competitive programming.'}
      : {zh:'同一行 pow(a, m - 2, m) 換到 m = 10。Fermat 小定理的前提是 m 為質數，可是函式裡沒有任何東西檢查這件事，它照樣會回傳一個數字。看看那些數字對不對。',
         en:'The same pow(a, m - 2, m), now with m = 10. Fermat\'s little theorem requires m to be prime, but nothing in the function checks that - it returns a number all the same. Let us see whether those numbers are right.'}});
  let right = 0, wrongInv = 0, noInv = 0;
  for (let a = 1; a <= n; a++){
    const g = gcdN(a, m), f = powModN(a, m - 2, m), prod = a * f % m, t = invTrue(a, m);
    if (prod === 1) right++; else if (t != null) wrongInv++; else noInv++;
    F.push({shapes:fermatStage(m, a - 1, {big:'pow(' + a + ', ' + (m - 2) + ', ' + m + ') = ' + f + ',  ' + a + ' x ' + f + ' % ' + m + ' = ' + prod, bigc:prod === 1 ? COL.tealL : COL.red,
                                          cap:prod === 1 ? {zh:a + ' 的反元素是 ' + f + '：Fermat 對了', en:'the inverse of ' + a + ' is ' + f + ': Fermat is right'}
                                              : t != null ? {zh:a + ' 有反元素 ' + t + '，Fermat 卻給了 ' + f, en:a + ' has the inverse ' + t + ', but Fermat answers ' + f}
                                              : {zh:'gcd(' + a + ', ' + m + ') = ' + g + '：' + a + ' 根本沒有反元素，Fermat 照樣回答 ' + f, en:'gcd(' + a + ', ' + m + ') = ' + g + ': ' + a + ' has no inverse at all, and Fermat still answers ' + f},
                                          capc:prod === 1 ? COL.pale : COL.red,
                                          cap2:t != null ? {zh:'擴展 Euclid（pow(a, -1, m)）給 ' + t, en:'extended Euclid (pow(a, -1, m)) gives ' + t}
                                                         : {zh:'pow(' + a + ', -1, ' + m + ') 直接丟 ValueError', en:'pow(' + a + ', -1, ' + m + ') raises ValueError'}}),
      panels:[{lbl:'a', chips:[chip(String(a), 'act')]}, {lbl:'Fermat', chips:[chip(String(f), prod === 1 ? 'ok' : 'bad')]},
              {lbl:{zh:'真的反元素', en:'true inverse'}, chips:[chip(t == null ? '—' : String(t), t == null ? 'empty' : 'act')]}],
      line:prod === 1 ? 1 : t != null ? 4 : 6,
      msg:prod === 1
        ? (prime ? {zh:a + ' × ' + f + ' = ' + a * f + ' = ' + Math.floor(a * f / m) + ' × 11 + 1。11 是質數，所以 1 到 10 每一個都和 11 互質，每一個都有反元素，Fermat 每一個都答對。',
                    en:a + ' x ' + f + ' = ' + a * f + ' = ' + Math.floor(a * f / m) + ' x 11 + 1. 11 is prime, so every number from 1 to 10 is coprime to it, every one has an inverse, and Fermat gets each of them right.'}
                 : {zh:'a = 1 對了，但只是因為 1 的任何次方都是 1，這不代表公式在 m = 10 上成立。', en:'a = 1 comes out right, but only because any power of 1 is 1 - it says nothing about the formula working for m = 10.'})
        : t != null
        ? {zh:a + ' 和 10 互質，真的反元素是 ' + t + '（' + a + ' × ' + t + ' = ' + a * t + '）。Fermat 給的 ' + f + ' 乘回去是 ' + prod + '，不是 1。原因：歐拉定理說的是 a^φ(m) ≡ 1，φ(10) = 4，不是 m - 1 = 9；只有 m 是質數時 φ(m) 才等於 m - 1。程式沒有丟錯誤，只是安靜地回傳錯的數字。',
           en:a + ' is coprime to 10 and its real inverse is ' + t + ' (' + a + ' x ' + t + ' = ' + a * t + '). Fermat\'s ' + f + ' multiplies back to ' + prod + ', not 1. The reason: Euler\'s theorem says a^φ(m) ≡ 1, and φ(10) = 4, not m - 1 = 9; φ(m) equals m - 1 only when m is prime. No error is raised - just a quietly wrong number.'}
        : {zh:'gcd(' + a + ', 10) = ' + g + '，' + a + ' × 任何數都是 ' + g + ' 的倍數，不可能 ≡ 1，所以根本沒有反元素。內建的 pow(' + a + ', -1, 10) 會丟 ValueError；Fermat 版本卻回傳 ' + f + '，假裝有答案。',
           en:'gcd(' + a + ', 10) = ' + g + ', so ' + a + ' times anything is a multiple of ' + g + ' and can never be ≡ 1 - there is no inverse at all. The built-in pow(' + a + ', -1, 10) raises ValueError; the Fermat version returns ' + f + ' and pretends there is an answer.'}});
  }
  F.push({shapes:fermatStage(m, n, {big:prime ? {zh:'10 個全對', en:'all 10 correct'} : {zh:'4 個有反元素的，Fermat 錯 ' + wrongInv + ' 個', en:'of the 4 invertible values, Fermat gets ' + wrongInv + ' wrong'},
                                    bigc:prime ? COL.tealL : COL.red,
                                    cap:prime ? {zh:'m 是質數：Fermat 和擴展 Euclid 一致', en:'m prime: Fermat and extended Euclid agree'}
                                              : {zh:'m = 1,000,000：400,000 個有反元素，Fermat 錯 399,999 個', en:'m = 1,000,000: 400,000 invertible values, Fermat wrong on 399,999'},
                                    capc:prime ? COL.pale : COL.red,
                                    cap2:{zh:'pow(a, -1, m) 用擴展 Euclid，任何 m 都對，沒有反元素時會報錯', en:'pow(a, -1, m) uses extended Euclid: right for any m, and it raises when there is no inverse'}}),
    panels:[{lbl:{zh:'對', en:'right'}, chips:[chip(String(right), 'ok')]}, {lbl:{zh:'錯', en:'wrong'}, chips:[chip(String(wrongInv), wrongInv ? 'bad' : 'dim')]},
            {lbl:{zh:'沒有反元素', en:'no inverse'}, chips:[chip(String(noInv), noInv ? 'bad' : 'dim')]}], line:prime ? 1 : 3,
    msg:prime
      ? {zh:'m = 11 時 10 個全對。1e9+7 也是質數，隨機抽 1,000 個 a，Fermat 錯 0 個——所以 nCk mod 1e9+7 用 Fermat 算反元素完全沒問題。問題出在把同一行程式搬到不是質數的模數上，下一個變體就是。',
         en:'With m = 11 all 10 are correct. 1e9+7 is prime too, and on 1,000 random a Fermat is wrong 0 times - so using Fermat for the inverses in nCk mod 1e9+7 is perfectly fine. The trouble starts when the same line moves to a modulus that is not prime, which is the other variant.'}
      : {zh:'m = 10 的 9 個 a 裡，Fermat 只有 a = 1 答對；有反元素的 3、7、9 全錯，沒有反元素的 2、4、5、6、8 照樣回傳數字。模數越大越糟：m = 1,337 時 1,140 個有反元素的錯 1,136 個，m = 1,000,000 時錯 399,999 個。LeetCode 372 的模數正是 1337 = 7 × 191，不是質數。修法：用 pow(a, -1, m)。',
         en:'Out of the 9 values of a for m = 10, Fermat only gets a = 1 right; the invertible 3, 7 and 9 are all wrong, and the non-invertible 2, 4, 5, 6, 8 still get a number back. It gets worse with size: for m = 1,337 Fermat is wrong on 1,136 of the 1,140 invertible values, for m = 1,000,000 on 399,999. LeetCode 372\'s modulus is exactly 1337 = 7 x 191, not a prime. The fix: pow(a, -1, m).'}});
  return F.list;
}

/* ================================================================ tab 5: nCk mod p */
const CODE_NCR = [
  'def factorials(n, p=MOD):',
  '    f = [1] * (n + 1)',
  '    for i in range(1, n + 1):',
  '        f[i] = f[i - 1] * i % p          # f[i] = i! % p',
  '    return f',
  '',
  'def ncr_floor_div(n, k, f, p=MOD):      # the bug',
  '    return f[n] // (f[k] * f[n - k] % p) % p'
];
const CODE_BINOM = [
  'class Binomial:',
  '    def __init__(self, n, p=MOD):',
  '        self.p = p',
  '        self.f = factorials(n, p)',
  '        self.inv_f = [1] * (n + 1)',
  '        self.inv_f[n] = pow(self.f[n], p - 2, p)   # p is prime: Fermat is fine',
  '        for i in range(n, 0, -1):                  # 1/(i-1)! = i * 1/i!',
  '            self.inv_f[i - 1] = self.inv_f[i] * i % p',
  '',
  '    def ncr(self, n, k):',
  '        if k < 0 or k > n:',
  '            return 0',
  '        return self.f[n] * self.inv_f[k] % self.p * self.inv_f[n - k] % self.p'
];
const NMAX = 100;
const FACT = (() => { const f = [1n]; for (let i = 1; i <= NMAX; i++) f.push(f[i - 1] * BigInt(i) % MOD); return f; })();
const INVF = (() => { const v = new Array(NMAX + 1).fill(1n); v[NMAX] = powModB(FACT[NMAX], MOD - 2n, MOD);
  for (let i = NMAX; i > 0; i--) v[i - 1] = v[i] * BigInt(i) % MOD; return v; })();
const fullFact = n => { let r = 1n; for (let i = 2n; i <= BigInt(n); i++) r *= i; return r; };
const ncrBug = (n, k) => FACT[n] / (FACT[k] * FACT[n - k] % MOD) % MOD;
const ncrGood = (n, k) => FACT[n] * INVF[k] % MOD * INVF[n - k] % MOD;
const ncrTrue = (n, k) => { let r = 1n; for (let i = 0; i < k; i++) r = r * BigInt(n - i) / BigInt(i + 1); return r; };
const NROWS = [10, 11, 12, 13, 14];
function ncrStage(v, upto, o){
  const sh = [], cx = [.65, 1.4, 3.85, 6.0, 7.75], cw = [.7, 2.4, 2.05, 1.65, 1.6];
  const hd = ['n', 'n!', 'f[n] = n! % p', v === 1 ? 'f[n]·inv_f[1]·inv_f[n-1]' : 'f[n] // (f[1]·f[n-1] % p)', {zh:'真正的 C(n, 1)', en:'true C(n, 1)'}];
  hd.forEach((t, i) => { const c = i === 3 ? COL.orangeL : COL.tealL;
    if (i === 3){ const two = v === 1 ? ['f[n]·inv_f[1]', '·inv_f[n-1]'] : ['f[n] //', '(f[1]·f[n-1] % p)']; sh.push(S.t(cx[i] + cw[i] / 2, .52, two[0], {c, fs:.2})); sh.push(S.t(cx[i] + cw[i] / 2, .76, two[1], {c, fs:.2})); }
    else sh.push(S.t(cx[i] + cw[i] / 2, .7, t, {c, fs:fitFs(t, cw[i] + .04, .22)})); });
  NROWS.forEach((n, i) => {
    if (i > upto) return;
    const y = .95 + i * .64, cur = i === upto, nf = fullFact(n), wrapped = nf >= MOD;
    const got = v === 1 ? ncrGood(n, 1) : ncrBug(n, 1), tru = ncrTrue(n, 1), ok = got === tru;
    sh.push(box(cx[0], y, cw[0] - .08, .52, cur ? 'act' : 'soft', String(n), .26));
    sh.push(box(cx[1], y, cw[1] - .08, .52, wrapped ? 'hot' : 'soft', fmt(nf), .26));
    sh.push(box(cx[2], y, cw[2] - .08, .52, wrapped ? 'hot' : 'soft', fmt(FACT[n]), .26));
    sh.push(box(cx[3], y, cw[3] - .08, .52, ok ? 'ok' : 'bad', fmt(got), .26));
    sh.push(box(cx[4], y, cw[4] - .08, .52, 'soft', fmt(tru), .26));
  });
  if (upto >= 3){             /* p sits between 12! and 13! */
    const yl = .95 + 3 * .64 - .07;
    sh.push(S.e(cx[1] - .05, yl, cx[2] + cw[2], yl, {s:'bad', arrow:false, w:.045}));
    sh.push(S.t(.4, yl + .08, 'p', {c:COL.red, fs:.24}));
  }
  if (o.big) sh.push(fitT(4.9, 4.72, o.big, {c:o.bigc || COL.orangeL, fs:.34}));
  return caps(sh, o);
}
function ncrFrames(v){
  const F = new Frames(), fix = v === 1;
  F.push({shapes:ncrStage(v, -1, {big:fix ? {zh:'修法：除法換成乘上反元素', en:'the fix: replace division by an inverse'} : 'C(n, k) = n! / (k! (n-k)!)', bigc:COL.purpleL,
                                  cap:{zh:'p = 1e9+7 = 1,000,000,007；表格裡存的是 n! % p，不是 n!', en:'p = 1e9+7 = 1,000,000,007; the table stores n! % p, not n!'},
                                  cap2:fix ? {zh:'inv_f[i] = (i!)⁻¹ mod p，p 是質數，用 Fermat 算一次再往回乘', en:'inv_f[i] = (i!)^-1 mod p; p is prime, so one Fermat call, then multiply backwards'}
                                           : {zh:'用 // 除以已經取過餘數的數，會發生什麼事？', en:'what happens when // divides numbers that were already reduced?'}}),
    panels:[{lbl:'p', chips:[chip('1,000,000,007')]}, {lbl:'k', chips:[chip('1', 'act')]}], line:fix ? 5 : 3,
    msg:fix
      ? {zh:'正確的做法：不要做除法。先存 f[i] = i! % p，再用 Fermat（p 是質數，這裡安全）算一次 inv_f[n] = (n!)⁻¹，然後利用 1/(i-1)! = i × 1/i! 往回推出整張 inv_f 表。之後 C(n, k) = f[n] × inv_f[k] × inv_f[n-k]，全部是乘法，乘法保得住 %。',
         en:'The right way: never divide. Store f[i] = i! % p, compute inv_f[n] = (n!)^-1 once with Fermat (p is prime, so that is safe), then walk backwards with 1/(i-1)! = i x 1/i! to fill the whole inv_f table. Now C(n, k) = f[n] x inv_f[k] x inv_f[n-k] - only multiplications, and multiplication survives %.'}
      : {zh:'nCk mod p 的常見寫法是先建 f[i] = i! % p 的表，每次查詢 O(1)。直覺版本照抄公式：f[n] // (f[k] × f[n-k] % p)。這裡固定 k = 1，正確答案就是 n，最容易看出哪裡開始壞。',
         en:'The usual way to answer many nCk mod p queries is a table f[i] = i! % p, then O(1) per query. The naive version copies the formula: f[n] // (f[k] x f[n-k] % p). Here k is fixed at 1, so the right answer is simply n - which makes it easy to see where it breaks.'}});
  NROWS.forEach((n, i) => {
    const nf = fullFact(n), wrapped = nf >= MOD, got = fix ? ncrGood(n, 1) : ncrBug(n, 1), tru = BigInt(n), ok = got === tru;
    F.push({shapes:ncrStage(v, i, {big:(fix ? 'f[' + n + ']·inv_f[1]·inv_f[' + (n - 1) + '] % p = ' : 'f[' + n + '] // f[' + (n - 1) + '] = ' + fmt(FACT[n]) + ' // ' + fmt(FACT[n - 1]) + ' = ') + fmt(got),
                                   bigc:ok ? COL.tealL : COL.red,
                                   cap:wrapped ? {zh:n + '! = ' + fmt(nf) + ' 比 p 大：f[' + n + '] 只是餘數', en:n + '! = ' + fmt(nf) + ' is larger than p: f[' + n + '] is just a remainder'}
                                               : {zh:n + '! = ' + fmt(nf) + ' 比 p 小：f[' + n + '] 就是 ' + n + '! 本身', en:n + '! = ' + fmt(nf) + ' is below p: f[' + n + '] is ' + n + '! itself'},
                                   capc:wrapped ? COL.orangeL : COL.pale,
                                   cap2:ok ? {zh:'C(' + n + ', 1) = ' + n + '，對', en:'C(' + n + ', 1) = ' + n + ', correct'} : {zh:'C(' + n + ', 1) 應該是 ' + n + '，得到 ' + fmt(got), en:'C(' + n + ', 1) should be ' + n + ', got ' + fmt(got)}}),
      panels:[{lbl:'n', chips:[chip(String(n), 'act')]}, {lbl:'f[n]', chips:[chip(fmt(FACT[n]), wrapped ? 'hot' : '')]}, {lbl:{zh:'答案', en:'answer'}, chips:[chip(fmt(got), ok ? 'ok' : 'bad')]}],
      line:fix ? 12 : 7,
      msg:fix
        ? (wrapped ? {zh:'f[' + n + '] = ' + fmt(FACT[n]) + ' 已經被取過餘數，不再是 ' + n + '!，但沒關係：乘上 inv_f 是在模 p 的世界裡「除」，(' + n + '! % p) × ((' + (n - 1) + '!)⁻¹ % p) ≡ ' + n + '。答案 ' + fmt(got) + '，對。',
                      en:'f[' + n + '] = ' + fmt(FACT[n]) + ' has already been reduced and is no longer ' + n + '!, and that is fine: multiplying by inv_f is division inside the mod-p world, so (' + n + '! % p) x ((' + (n - 1) + '!)^-1 % p) ≡ ' + n + '. The answer is ' + fmt(got) + ', correct.'}
                   : {zh:'n = ' + n + '：' + n + '! 還沒超過 p，兩種寫法都對。差別在後面幾列。', en:'n = ' + n + ': ' + n + '! has not passed p yet, so both versions are right. The difference shows in the rows below.'})
        : wrapped
        ? {zh:n === 13 ? '12! = 479,001,600 < p < 13! = 6,227,020,800。從 13 開始 f[n] 是 n! 除以 p 的餘數，比 f[n-1] 還小也不奇怪：' + fmt(FACT[n]) + ' // ' + fmt(FACT[n - 1]) + ' = ' + fmt(got) + '。% 保得住乘法，所以 f 表本身沒錯；錯的是拿餘數去做 //，因為 % 保不住除法。'
                        : 'n = ' + n + ' 一樣錯，得到 ' + fmt(got) + '。餘數之間做整數除法沒有任何意義，答案是什麼完全看運氣。',
           en:n === 13 ? '12! = 479,001,600 < p < 13! = 6,227,020,800. From 13 on f[n] is the remainder of n! divided by p, and it can even be smaller than f[n-1]: ' + fmt(FACT[n]) + ' // ' + fmt(FACT[n - 1]) + ' = ' + fmt(got) + '. % survives multiplication, so the f table itself is fine; the mistake is applying // to remainders, because % does not survive division.'
                       : 'n = ' + n + ' is wrong as well, giving ' + fmt(got) + '. Integer division between remainders means nothing, so the result is pure luck.'}
        : {zh:'n = ' + n + '：' + n + '! = ' + fmt(nf) + ' 還放得進 p 以內，f[n] 就是真正的階乘，// 剛好是真的除法，答案 ' + n + '。所有小測資都落在這個區間，所以這個 bug 在本機測試很難被發現。',
           en:'n = ' + n + ': ' + n + '! = ' + fmt(nf) + ' still fits below p, so f[n] is the real factorial and // is real division: the answer is ' + n + '. Every small test lives in this range, which is why this bug hides so well locally.'}});
  });
  let bad = 0, tot = 0;
  for (let n = 0; n <= 30; n++) for (let k = 0; k <= n; k++){ tot++; if (ncrBug(n, k) !== ncrTrue(n, k) % MOD) bad++; }
  const c100 = ncrGood(100, 50), c100bug = ncrBug(100, 50);
  F.push({shapes:ncrStage(v, NROWS.length, {big:fix ? 'C(100, 50) mod p = ' + fmt(c100) + '  ✓' : {zh:'n ≤ 30 的 ' + tot + ' 個答案錯 ' + bad + ' 個', en:bad + ' of the ' + tot + ' answers with n <= 30 are wrong'},
                                            bigc:fix ? COL.tealL : COL.red,
                                            cap:fix ? {zh:'建表 O(n) 一次，之後每次查詢 O(1)', en:'O(n) to build the table once, then O(1) per query'}
                                                    : {zh:'C(100, 50) 算出 ' + fmt(c100bug) + '，正確是 ' + fmt(c100), en:'C(100, 50) comes out as ' + fmt(c100bug) + '; the right answer is ' + fmt(c100)},
                                            capc:fix ? COL.pale : COL.red,
                                            cap2:fix ? {zh:'1,000 次查詢、n ≤ 10,000：查表不到 1 ms，math.comb 約 1.5 秒', en:'1,000 queries with n <= 10,000: table under 1 ms, math.comb about 1.5 s'}
                                                     : {zh:'n ≤ 12 全對，第一個錯的是 C(13, 1)', en:'every n <= 12 is right; the first failure is C(13, 1)'}}),
    panels:fix ? [{lbl:'C(100, 50)', chips:[chip(fmt(c100), 'ok')]}]
               : [{lbl:{zh:'錯', en:'wrong'}, chips:[chip(bad + ' / ' + tot, 'bad')]}, {lbl:'C(100, 50)', chips:[chip(fmt(c100bug), 'bad'), chip(fmt(c100), 'ok')]}],
    line:fix ? 12 : 7,
    msg:fix
      ? {zh:'C(100, 50) mod p = ' + fmt(c100) + '，和 number_theory.py 一樣；C(10⁶, 500,000) 也照樣是 O(1) 查一次。也可以每次呼叫 math.comb(n, k) % p，答案對，但它每次都先算出完整的大數：1,000 次查詢大約一秒半，查表不到一毫秒。這裡能放心用 Fermat，是因為 1e9+7 是質數——上一個分頁看過，換成合數模數就不行。',
         en:'C(100, 50) mod p = ' + fmt(c100) + ', matching number_theory.py, and C(10^6, 500,000) is still a single O(1) lookup. Calling math.comb(n, k) % p also gives correct answers, but it builds the full-size number every time: 1,000 queries take about a second and a half against under a millisecond for the table. Fermat is safe here only because 1e9+7 is prime - the previous tab showed what happens with a composite modulus.'}
      : {zh:'n ≤ 30 的 ' + tot + ' 個 C(n, k) 裡錯了 ' + bad + ' 個，C(100, 50) 得到 ' + fmt(c100bug) + '。這個 bug 從不丟錯誤：12 以內的測資全過，大一點就默默回傳錯的數字。規則只有一條：取過餘數以後，除法一律改成乘上反元素。',
         en:'Of the ' + tot + ' values C(n, k) with n <= 30, ' + bad + ' are wrong, and C(100, 50) comes out as ' + fmt(c100bug) + '. The bug never raises: every test up to 12 passes, and anything larger quietly returns the wrong number. One rule fixes it: once you have reduced mod p, every division becomes a multiplication by an inverse.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'Day 41 — 模運算、快速冪、GCD 與質數篩', en:'Day 41 - Modular arithmetic, fast power, GCD and the prime sieve'},
  sub:{zh:'% 保得住加、減、乘，保不住除——今天每一個坑都是這一條規則。', en:'% survives addition, subtraction and multiplication, but not division - every trap today is that one rule.'},
  tabs:[
    {
      id:'sieve', label:{zh:'埃氏篩', en:'the sieve'},
      stage:{zh:'0 到 49：每個質數劃掉它的倍數，沒被劃到的就是質數', en:'0 to 49: each prime crosses out its multiples; whatever survives is prime'},
      view:VIEW,
      variants:[{zh:'埃氏篩（n = 50）', en:'sieve of Eratosthenes - n = 50'},
                {zh:'LC 204：小於 n 還是 n 以內', en:'LC 204: below n or up to n'}],
      idea:{zh:'埃氏篩不檢查任何一個數，而是從最小的質數開始劃掉它的倍數：沒人劃到的就是質數。兩個省工的地方：每個質數 p 從 p² 開始劃（更小的倍數 k × p 已經被 k 的質因數劃過），而且 p 只要跑到 √n（合數一定有一個 ≤ √n 的因數）。在 Python 裡把「劃掉一整段」寫成 bytearray 切片指定：100 萬以下的質數，逐一試除 5.01 秒、篩法寫成迴圈 213 ms、切片 18.0 ms。LeetCode 204 要的是嚴格小於 n 的質數；寫成「n 以內」只在 n 本身是質數時錯，題目三個範例 10、0、1 全都抓不到。',
            en:'The sieve never tests a number; it starts from the smallest prime and crosses out its multiples, and whatever nobody crossed out is prime. Two savings: each prime p starts at p² (a smaller multiple k x p was already crossed by a prime factor of k), and p only runs up to √n (a composite always has a factor <= √n). In Python "cross out a whole stride" is one bytearray slice assignment: for the primes below 1,000,000, trial division takes 5.01 s, the sieve as Python loops 213 ms, and with slices 18.0 ms. LeetCode 204 wants the primes strictly below n; counting "up to n" is only wrong when n itself is prime, and the three examples 10, 0, 1 never catch it.'},
      legend:[['#ff9736', {zh:'這一輪新劃掉的', en:'crossed out this round'}],
              ['#9d6bff', {zh:'目前的質數 p / n 的位置', en:'the current prime p / position of n'}],
              ['#3fe0dd', {zh:'確定是質數', en:'known prime'}],
              ['#ff5c5c', {zh:'被多數的 n', en:'n counted by mistake'}]],
      get code(){ return curV() === 1 ? CODE_204 : CODE_SIEVE; },
      build:sieveFrames
    },
    {
      id:'pow', label:{zh:'快速冪', en:'fast power'},
      stage:{zh:'把指數寫成二進位：每個 bit 平方一次，遇到 1 再乘一次', en:'write the exponent in binary: one squaring per bit, one extra multiply per 1'},
      view:VIEW,
      variants:[{zh:'3^13 mod 100', en:'3^13 mod 100'},
                {zh:'LC 50：myPow(2.0, -2)', en:'LC 50: myPow(2.0, -2)'}],
      idea:{zh:'13 = 0b1101 = 8 + 4 + 1，所以 3^13 = 3^8 × 3^4 × 3^1。3 的 1、2、4、8 次方每個都是前一個的平方，於是乘法次數從「指數那麼多次」降到「位元數加上 1 的個數」：13 要 5 次、10 億要 41 次、10^18 要 82 次。每一步都先 % m，因為 % 保得住乘法，數字永遠不超過 m²：3 的一百萬次方先算再取餘數 102.54 ms，pow(3, 1000000, MOD) 2.3 µs。同一個迴圈不加 % 就是 LeetCode 50；負指數換成 (1/x)^(-n)，Python 的 -n 不會溢位。',
            en:'13 = 0b1101 = 8 + 4 + 1, so 3^13 = 3^8 x 3^4 x 3^1. Each of 3^1, 3^2, 3^4, 3^8 is the square of the previous one, so the multiplication count drops from "as many as the exponent" to "bits plus ones": 5 for 13, 41 for a billion, 82 for 10^18. Every step reduces % m first, which is allowed because % survives multiplication, so no number ever exceeds m²: 3 to the millionth then % takes 102.54 ms, pow(3, 1000000, MOD) 2.3 µs. The same loop without % is LeetCode 50; a negative exponent becomes (1/x)^(-n), and -n cannot overflow in Python.'},
      legend:[['#ff9736', {zh:'這一位 / 乘進來的結果', en:'this bit / the result it multiplies in'}],
              ['#9d6bff', {zh:'a 的 2^k 次方', en:'a to the 2^k'}],
              ['#3fe0dd', {zh:'已乘進來的位', en:'bits already multiplied in'}]],
      get code(){ return curV() === 1 ? CODE_50 : CODE_POW; },
      build:powFrames
    },
    {
      id:'gcd', label:{zh:'Euclid 與 GCD', en:'Euclid and GCD'},
      stage:{zh:'gcd(a, b) = gcd(b, a % b)：一直換成比較小的一對，直到餘數是 0', en:'gcd(a, b) = gcd(b, a % b): keep swapping for a smaller pair until the remainder is 0'},
      view:VIEW,
      variants:[{zh:'gcd(252, 105)', en:'gcd(252, 105)'},
                {zh:'擴展 Euclid (240, 46)', en:'extended Euclid (240, 46)'},
                {zh:'LC 365 倒水', en:'LC 365 water jugs'}],
      idea:{zh:'整除 a 和 b 的數一定整除 a % b，反過來也是，所以 gcd(a, b) = gcd(b, a % b)。每兩步 a 至少砍半，最壞情況是相鄰的 Fibonacci 數（1000 以內最多 14 步）。擴展 Euclid 多記兩個係數，每一列維持 a = 240x + 46y，最後得到 Bézout 等式 gcd = 240 × (-9) + 46 × 47。反過來說，a、b 的整數組合全是 gcd 的倍數——LeetCode 365 的兩個水壺只能量出 gcd(x, y) 的倍數，而且不能超過 x + y。少了後面那個條件，(3, 5, 9) 會錯答 True。',
            en:'Anything dividing a and b divides a % b and the other way round, so gcd(a, b) = gcd(b, a % b). Every two steps a at least halves; the worst case is consecutive Fibonacci numbers (at most 14 steps up to 1000). Extended Euclid carries two coefficients so that every row keeps a = 240x + 46y, ending with Bézout\'s identity gcd = 240 x (-9) + 46 x 47. Conversely, integer combinations of a and b are exactly the multiples of the gcd - the two jugs of LeetCode 365 can only measure multiples of gcd(x, y), and never more than x + y. Drop that last condition and (3, 5, 9) wrongly answers True.'},
      legend:[['#ff9736', {zh:'餘數 / 目前這一列', en:'the remainder / current row'}],
              ['#9d6bff', {zh:'b 的整段 / 係數 x、y', en:'whole copies of b / coefficients x, y'}],
              ['#3fe0dd', {zh:'水量 / 驗算成立', en:'water / the check holds'}],
              ['#ff5c5c', {zh:'不成立的條件', en:'a failed condition'}]],
      get code(){ const v = curV(); return v === 1 ? CODE_EXT : v === 2 ? CODE_365 : CODE_GCD; },
      build:gcdFrames
    },
    {
      id:'inverse', label:{zh:'反元素：Fermat 的前提', en:'inverses: Fermat\'s fine print'},
      stage:{zh:'pow(a, m - 2, m) 是不是 a 的反元素？只有 m 是質數時才是', en:'is pow(a, m - 2, m) the inverse of a? only when m is prime'},
      view:VIEW,
      variants:[{zh:'m = 11（質數）', en:'m = 11 - prime'},
                {zh:'m = 10（合數）', en:'m = 10 - composite'}],
      idea:{zh:'% 保不住除法，所以模運算裡的除以 a 要換成乘上 a 的反元素：a × x ≡ 1。Fermat 小定理給了一行寫法 pow(a, m - 2, m)，但前提是 m 為質數，而程式不會檢查。m = 11 時 10 個全對；m = 10 時，有反元素的 3、7、9 全錯，沒有反元素的 2、4、5、6、8 照樣回傳數字。m = 1,000,000 時 400,000 個可逆的數錯 399,999 個。pow(a, -1, m) 走擴展 Euclid，任何 m 都對，沒有反元素時直接 ValueError。',
            en:'% does not survive division, so dividing by a mod m becomes multiplying by a\'s inverse: a x x ≡ 1. Fermat\'s little theorem gives the one-liner pow(a, m - 2, m), but only for a prime m, and nothing in the code checks. With m = 11 all 10 are right; with m = 10 the invertible 3, 7, 9 all come out wrong, and the non-invertible 2, 4, 5, 6, 8 still get a number. For m = 1,000,000, 399,999 of the 400,000 invertible values are wrong. pow(a, -1, m) uses extended Euclid: correct for any m, and a ValueError when there is no inverse.'},
      legend:[['#ff9736', {zh:'Fermat 給的答案', en:'Fermat\'s answer'}],
              ['#3fe0dd', {zh:'乘回去等於 1', en:'multiplies back to 1'}],
              ['#ff5c5c', {zh:'乘回去不是 1', en:'does not multiply back to 1'}],
              ['#9d6bff', {zh:'擴展 Euclid 的反元素', en:'the inverse from extended Euclid'}]],
      get code(){ return CODE_INV; },
      build:fermatFrames
    },
    {
      id:'ncr', label:{zh:'nCk 的 // 撞牆', en:'nCk hits the // wall'},
      stage:{zh:'f[n] = n! % p：n! 超過 p 之後，// 就失去意義', en:'f[n] = n! % p: once n! passes p, // stops meaning anything'},
      view:VIEW,
      variants:[{zh:'// 除以取過餘數的數', en:'// on reduced numbers'},
                {zh:'乘上反元素階乘', en:'multiply by inverse factorials'}],
      idea:{zh:'nCk mod p 的標準做法是先建階乘表 f[i] = i! % p。直覺寫法 f[n] // (f[k] × f[n-k] % p) 在 n ≤ 12 全對，因為 12! = 479,001,600 還小於 p = 1e9+7，表裡存的就是真正的階乘；13! = 6,227,020,800 超過 p，f[13] 只是餘數，// 從這裡開始沒有意義：C(13, 1) 得到 0。n ≤ 30 的 496 個答案錯 369 個。修法是不做除法：用 Fermat（p 是質數）算一次 (n!)⁻¹，再用 1/(i-1)! = i × 1/i! 往回推出整張反元素表，C(n, k) = f[n] × inv_f[k] × inv_f[n-k]。',
            en:'The standard way to compute nCk mod p is a factorial table f[i] = i! % p. The naive f[n] // (f[k] x f[n-k] % p) is right for every n <= 12, because 12! = 479,001,600 is still below p = 1e9+7 and the table holds real factorials; 13! = 6,227,020,800 passes p, f[13] is only a remainder, and // means nothing from there on: C(13, 1) comes out as 0. Of the 496 answers with n <= 30, 369 are wrong. The fix never divides: compute (n!)^-1 once with Fermat (p is prime), walk back with 1/(i-1)! = i x 1/i! to fill an inverse table, and C(n, k) = f[n] x inv_f[k] x inv_f[n-k].'},
      legend:[['#ff9736', {zh:'已經超過 p、被取過餘數', en:'past p - already reduced'}],
              ['#3fe0dd', {zh:'答案正確', en:'correct answer'}],
              ['#ff5c5c', {zh:'答案錯誤 / p 的位置', en:'wrong answer / where p sits'}],
              ['#9d6bff', {zh:'目前這一列', en:'the current row'}]],
      get code(){ return curV() === 1 ? CODE_BINOM : CODE_NCR; },
      build:ncrFrames
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
