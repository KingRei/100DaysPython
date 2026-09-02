// DAY: 23
// TITLE_ZH: 二分搜尋的三種邊界寫法
// TITLE_EN: Binary search and its three boundary variants
// SUB_ZH: 二分搜尋很少是「找 x 在哪」。真正在用的版本是找邊界：第一個成立的位置。而邊界寫錯不會報錯，只會安靜地少一格。
// SUB_EN: Binary search is rarely "find x". The version people actually run finds a boundary - the first position where something becomes true. And a wrong boundary does not crash, it is quietly off by one.
// FOLDER: day%2023%20-%20binary%20search
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const mid2 = VIEW[0] / 2;
const rowX = (n, w) => (VIEW[0] - n * w) / 2;
const note = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});
const foot = (s, y) => S.t(mid2, y == null ? 6.18 : y, s, {c:COL.grey, fs:.27});

/* a lo/hi/mid marker under a cell row: idx may be n (past the end) */
function marks(x0, y, w, m){
  const out = [], used = [];
  for (const k in m){
    const i = m[k];
    if (i == null) continue;
    let x = x0 + i * w + (k === 'mid' ? (w - .06) / 2 : 0);
    while (used.some(u => Math.abs(u - x) < .30)) x += .32;
    used.push(x);
    out.push(S.t(x, y, k, {c:k === 'mid' ? COL.orangeL : COL.purpleL, fs:.30}));
    out.push(S.e(x, y + .16, x, y + .56, {s:k === 'mid' ? 'hot' : 'act'}));
  }
  return out;
}


/* the engines below build flat chip lists; the renderer wants panel groups */
function norm(list){
  list.forEach(f => {
    if (f.panels && f.panels.length && f.panels[0].chips === undefined)
      f.panels = [{lbl:{zh:'狀態', en:'state'}, chips:f.panels}];
  });
  return list;
}

/* ===================================================== 1. lower / upper */
const A1 = [2, 3, 3, 3, 5, 8, 8, 13];
const X1 = 3;
const CW = .82, CH = .74;

const CODE_B = [
'def bound(a, x, upper):',
'    lo, hi = 0, len(a)          # half open [lo, hi)',
'    while lo < hi:',
'        mid = lo + (hi - lo) // 2',
'        small = a[mid] <= x if upper else a[mid] < x',
'        if small:',
'            lo = mid + 1        # answer is strictly right of mid',
'        else:',
'            hi = mid            # mid itself may be the answer',
'    return lo                   # lo == hi is the boundary'
];

function boundFrames(upper){
  const F = new Frames();
  const a = A1, n = a.length, x = X1;
  const x0 = rowX(n, CW);
  const Y = 3.0;
  const name = upper ? 'upper_bound' : 'lower_bound';
  const rule = upper ? 'a[mid] <= x' : 'a[mid] < x';

  const shapes = (lo, hi, m) => {
    const st = {};
    for (let i = 0; i < n; i++) st[i] = i < lo ? 'done' : (i >= hi ? 'ghost' : 'idle');
    if (m != null) st[m] = 'hot';
    let s = cellRow(a, x0, Y, CW, CH, {states:st, index:true});
    s = s.concat(marks(x0, Y - 1.02, CW, {lo:lo, hi:hi, mid:m}));
    s.push(note(5.55, name + '(a, ' + x + ')   -   rule: ' + rule, COL.tealL, .36));
    s.push(note(4.95, 'left of lo: already known too small     right of hi: already known big enough',
                COL.grey, .28));
    return s;
  };
  const chips = (lo, hi, extra) => {
    const p = [{t:'lo ' + lo, cls:'act'}, {t:'hi ' + hi, cls:'act'}];
    return extra ? p.concat(extra) : p;
  };

  let lo = 0, hi = n;
  F.push({shapes:shapes(lo, hi, null), panels:chips(lo, hi), view:VIEW, line:1,
    msg:{zh:'開場就把整條陣列放進半開區間 [0, ' + n + ')。<b>hi 是長度不是最後一格</b> - 這樣「答案在最後面之後」才有位置可以表示。',
         en:'Start with the whole array as the half-open range [0, ' + n + '). <b>hi is the length, not the last index</b> - that is what gives "the answer is past the end" somewhere to live.'}});

  let guard = 0;
  while (lo < hi && guard++ < 40){
    const m = lo + ((hi - lo) >> 1);
    F.push({shapes:shapes(lo, hi, m), panels:chips(lo, hi, [{t:'mid ' + m + ' -> ' + a[m], cls:'hot'}]),
      view:VIEW, line:3,
      msg:{zh:'mid = lo + (hi - lo) // 2 = ' + m + '，a[' + m + '] = ' + a[m] + '。寫成 lo + (hi-lo)//2 而不是 (lo+hi)//2，是為了不讓兩個大索引相加溢位。',
           en:'mid = lo + (hi - lo) // 2 = ' + m + ', so a[' + m + '] = ' + a[m] + '. Written this way rather than (lo + hi) // 2 so two large indices can never overflow when added.'}});
    const small = upper ? a[m] <= x : a[m] < x;
    if (small){
      lo = m + 1;
      F.push({shapes:shapes(lo, hi, null), panels:chips(lo, hi, [{t:rule.replace('mid', String(m)) + ' true', cls:'ok'}]),
        view:VIEW, line:6,
        msg:{zh:'a[' + m + '] = ' + a[m] + ' 讓 <b>' + rule + '</b> 成立，代表 mid 本身也不可能是答案，所以 lo 可以跳到 mid + 1 = ' + lo + '。+1 是這裡不會無窮迴圈的唯一理由。',
             en:'a[' + m + '] = ' + a[m] + ' makes <b>' + rule + '</b> true, so mid itself cannot be the answer and lo jumps to mid + 1 = ' + lo + '. That +1 is the only reason this loop cannot spin forever.'}});
    } else {
      hi = m;
      F.push({shapes:shapes(lo, hi, null), panels:chips(lo, hi, [{t:rule.replace('mid', String(m)) + ' false', cls:'ok'}]),
        view:VIEW, line:8,
        msg:{zh:'<b>' + rule + '</b> 不成立，mid <b>有可能就是答案</b>，所以只能把 hi 收到 mid，不能收到 mid - 1。這一格差別就是「找到」跟「跳過答案」的差別。',
             en:'<b>' + rule + '</b> is false, so mid <b>might itself be the answer</b> and hi can only come down to mid, never mid - 1. That single index is the difference between finding the answer and stepping over it.'}});
    }
  }
  const other = upper ? 'lower_bound' : 'upper_bound';
  F.push({shapes:shapes(lo, hi, null), panels:chips(lo, hi, [{t:name + ' = ' + lo, cls:'ok'}]),
    view:VIEW, line:9,
    msg:{zh:'lo 和 hi 撞在一起 = ' + lo + '，區間空了，那個位置<b>就是</b>答案 - 迴圈裡沒有任何提早 return。' +
            (upper ? ' upper - lower = ' + lo + ' - 1 = 2 就是 x 出現的次數。'
                   : ' 這也正是 bisect_left；把 < 換成 <= 就變成 ' + other + '。'),
         en:'lo and hi meet at ' + lo + ', the range is empty, and that position <b>is</b> the answer - there is no early return anywhere in the loop.' +
            (upper ? ' upper - lower = ' + lo + ' - 1 = 2 is how many times x occurs.'
                   : ' This is exactly bisect_left; change < to <= and you get ' + other + '.')}});
  return F.list;
}

/* ================================================== 2. loops that look right */
const A2 = [1, 2, 3, 4, 5, 6, 7];
const X2 = 4;
const CAP = 9;

const CODE_X = [
'lo, hi = 0, len(a)        # (a) uses  0, len(a) - 1',
'while lo < hi:            # (a) uses  while lo <= hi',
'    mid = (lo + hi) // 2',
'    if a[mid] < x:',
'        lo = mid + 1      # (b) drops the + 1',
'    else:',
'        hi = mid          # (c) writes mid - 1',
'return lo'
];

function bugFrames(v){
  const F = new Frames();
  const a = A2, n = a.length, x = X2, x0 = rowX(n, CW), Y = 3.0;
  const closed = v === 0;
  const titles = [
    {zh:'(a) 閉區間 [lo, hi] 配 hi = mid', en:'(a) closed interval [lo, hi] with hi = mid'},
    {zh:'(b) lo = mid，少了 + 1', en:'(b) lo = mid, without the + 1'},
    {zh:'(c) hi = mid - 1', en:'(c) hi = mid - 1'}
  ];
  const shapes = (lo, hi, m, dead) => {
    const st = {};
    for (let i = 0; i < n; i++){
      const inRange = closed ? (i >= lo && i <= hi) : (i >= lo && i < hi);
      st[i] = inRange ? 'idle' : (i < lo ? 'done' : 'ghost');
    }
    if (m != null) st[m] = dead ? 'bad' : 'hot';
    let s = cellRow(a, x0, Y, CW, CH, {states:st, index:true});
    s = s.concat(marks(x0, Y - 1.02, CW, {lo:lo, hi:hi, mid:m}));
    s.push(note(5.55, titles[v], v === 2 ? COL.red : COL.orangeL, .36));
    s.push(note(5.02, {zh:'目標：找出第一個 a[i] >= 4 的位置，正確答案是 3',
                       en:'goal: first index with a[i] >= 4 - the correct answer is 3'},
                COL.grey, .28));
    return s;
  };

  let lo = 0, hi = closed ? n - 1 : n, steps = 0, hung = false;
  F.push({shapes:shapes(lo, hi, null), panels:[{t:'lo ' + lo, cls:'act'}, {t:'hi ' + hi, cls:'act'}],
    view:VIEW, line:0,
    msg:{zh:'三個迴圈都看起來像課本，三個都會跑，三個都錯。這一版是 ' + tr(titles[v]) + '。',
         en:'All three loops look like the textbook, all three run, all three are wrong. This one is ' + tr(titles[v]) + '.'}});

  while ((closed ? lo <= hi : lo < hi) && steps < CAP){
    steps++;
    const m = (lo + hi) >> 1;
    F.push({shapes:shapes(lo, hi, m), panels:[{t:'step ' + steps, cls:'hot'},
        {t:'lo ' + lo, cls:'act'}, {t:'hi ' + hi, cls:'act'}, {t:'mid ' + m, cls:'hot'}],
      view:VIEW, line:2,
      msg:{zh:'mid = (' + lo + ' + ' + hi + ') // 2 = ' + m + '，a[' + m + '] = ' + a[m] + '。',
           en:'mid = (' + lo + ' + ' + hi + ') // 2 = ' + m + ', so a[' + m + '] = ' + a[m] + '.'}});
    if (a[m] < x){
      const nlo = v === 1 ? m : m + 1;
      const stuck = nlo === lo;
      lo = nlo;
      F.push({shapes:shapes(lo, hi, stuck ? m : null, stuck), panels:[{t:'lo -> ' + lo, cls:stuck ? 'bad' : 'ok'}],
        view:VIEW, line:4,
        msg:stuck ? {zh:'lo = mid，但 mid 已經 <b>等於 lo</b>（hi - lo = 1 時 (lo+hi)//2 就是 lo），所以 lo 根本沒動。區間沒有變小，下一圈會做一模一樣的事。',
                     en:'lo = mid, but mid is <b>already lo</b> - when hi - lo is 1, (lo + hi) // 2 is lo - so lo does not move. The range did not shrink, and the next iteration will do exactly the same thing.'}
                  : {zh:'a[' + m + '] 太小，答案在 mid 右邊，lo 前進到 ' + lo + '。',
                     en:'a[' + m + '] is too small, so the answer is right of mid and lo advances to ' + lo + '.'}});
      if (stuck){ hung = true; break; }
    } else {
      const nhi = v === 2 ? m - 1 : m;
      const stuck = closed && nhi === hi && lo === hi;
      hi = nhi;
      F.push({shapes:shapes(lo, hi, stuck ? m : null, stuck || v === 2), panels:[{t:'hi -> ' + hi, cls:v === 2 ? 'bad' : 'ok'}],
        view:VIEW, line:6,
        msg:v === 2 ? {zh:'hi = mid - 1 把 <b>a[' + m + '] 自己丟掉了</b>。但 a[' + m + '] = ' + a[m] + ' 滿足 >= x，它<b>有可能就是答案</b> - 丟掉之後迴圈照樣結束，照樣回傳一個看起來合理的數字。',
                       en:'hi = mid - 1 throws away <b>a[' + m + '] itself</b>. But a[' + m + '] = ' + a[m] + ' satisfies >= x and <b>might be the answer</b> - once discarded the loop still terminates and still returns a perfectly plausible number.'}
                    : {zh:'a[' + m + '] >= x，mid 可能是答案，hi 收到 ' + hi + '。',
                       en:'a[' + m + '] >= x, so mid may be the answer and hi comes down to ' + hi + '.'}});
    }
    if (closed && lo === hi && a[(lo + hi) >> 1] >= x){ /* keep spinning to show it */ }
  }
  if (steps >= CAP) hung = true;
  const good = 3;
  F.push({shapes:shapes(lo, hi, null, hung),
    panels:[{t:hung ? 'no termination' : 'returns ' + lo, cls:'bad'}, {t:'correct 3', cls:'ok'}],
    view:VIEW, line:7,
    msg:hung ? {zh:'停在第 ' + steps + ' 步：區間不再變小，這個迴圈<b>不會結束</b>。掛住至少會被發現 - 真正危險的是 (c)，它會結束。',
                en:'Stopped after ' + steps + ' steps: the range stopped shrinking and this loop <b>never ends</b>. A hang at least gets noticed - the dangerous one is (c), which terminates.'}
             : {zh:'回傳 ' + lo + '，正確答案是 ' + good + '。<b>沒有例外、沒有當掉</b>，只是安靜地少一格 - 而且離開這個迴圈以後你再也看不出來。',
                en:'It returns ' + lo + ' where the answer is ' + good + '. <b>No exception, no crash</b>, just quietly off by one - and once you are outside this loop there is nothing left to see.'}});
  return F.list;
}

/* =============================================== 3. binary search on the answer */
const PILES = [30, 11, 23, 4, 20];
const HOURS_LIMIT = 6;
const SMAX = 30;
const koko = s => PILES.reduce((t, p) => t + Math.ceil(p / s), 0);

const CODE_A = [
'def search_first_true(lo, hi, ok):      # integer answer space',
'    while lo < hi:',
'        mid = lo + (hi - lo) // 2',
'        if ok(mid): hi = mid',
'        else:       lo = mid + 1',
'    return lo',
'',
'def bisect_float(lo, hi, ok, tol):      # the qps search',
'    while hi - lo > tol and rounds < budget:',
'        mid = round((lo + hi) / 2, 4)',
'        if mid <= lo or mid >= hi:',
'            break                       # midpoint collapsed',
'        if ok(mid): lo, best = mid, mid',
'        else:       hi = mid',
'    return best, lo, hi'
];

function kokoFrames(){
  const F = new Frames();
  const w = .30, h = .60, Y = 3.05, x0 = (VIEW[0] - SMAX * w) / 2;
  const known = {};
  const shapes = (lo, hi, m) => {
    const s = [];
    for (let i = 0; i < SMAX; i++){
      const sp = i + 1;
      let st = 'ghost', lab = '?';
      if (known[sp] != null){ st = known[sp] ? 'ok' : 'done'; lab = known[sp] ? 'T' : '.'; }
      if (sp === m){ st = 'hot'; lab = known[sp] == null ? '?' : (known[sp] ? 'T' : '.'); }
      s.push(S.r(x0 + i * w, Y, w - .05, h, st, lab, {fs:.34}));
      if (sp % 5 === 0 || sp === 1)
        s.push(S.t(x0 + i * w + w / 2, Y + h + .36, String(sp), {c:COL.grey, fs:.26}));
    }
    if (lo != null){
      s.push(S.e(x0 + (lo - 1) * w, Y - .30, x0 + (hi - 1) * w, Y - .30,
                 {s:'act', arrow:false, w:.07}));
      s.push(S.t((x0 + (lo - 1) * w + x0 + (hi - 1) * w) / 2, Y - .48,
                 {zh:'還沒排除的範圍', en:'range not yet ruled out'}, {c:COL.purpleL, fs:.28}));
    }
    s.push(note(1.55, 'ok(s) = hours(piles, s) <= ' + HOURS_LIMIT, COL.tealL, .38));
    s.push(note(2.10, {zh:'這一列從來沒有被建出來 - 每一格都要跑一次模擬才知道',
                       en:'this row is never built - each cell costs a simulation to learn'},
                COL.grey, .28));
    s.push(foot({zh:'候選速度（根香蕉 / 小時）', en:'candidate eating speed (bananas / hour)'}, Y + h + .95));
    return s;
  };

  let lo = 1, hi = SMAX + 1, n = 0;
  F.push({shapes:shapes(lo, hi, null), panels:[{t:'piles ' + PILES.join(' '), cls:'act'},
      {t:'h = ' + HOURS_LIMIT, cls:'act'}], view:VIEW, line:0,
    msg:{zh:'沒有陣列可以搜尋 - 要找的是<b>答案本身</b>：一個速度。範圍是 1 到 max(piles) = ' + SMAX + '，因為再快也沒有意義。',
         en:'There is no array to search. What is being searched is <b>the answer itself</b> - a speed. The range is 1 to max(piles) = ' + SMAX + ', because anything faster changes nothing.'}});

  while (lo < hi){
    const m = lo + ((hi - lo) >> 1);
    const hrs = koko(m);
    const good = hrs <= HOURS_LIMIT;
    known[m] = good;
    n++;
    F.push({shapes:shapes(lo, hi, m), panels:[{t:'probe ' + n, cls:'hot'},
        {t:'speed ' + m, cls:'hot'}, {t:hrs + ' hours', cls:good ? 'ok' : 'bad'}],
      view:VIEW, line:2,
      msg:{zh:'第 ' + n + ' 次探測：速度 ' + m + ' 要 ' + hrs + ' 小時，' +
              (good ? '<b>來得及</b>，所以答案不會比 ' + m + ' 大 - hi 收到 mid（mid 自己可能就是答案）。'
                    : '<b>來不及</b>，所以答案一定比 ' + m + ' 大 - lo 跳到 mid + 1。'),
           en:'Probe ' + n + ': speed ' + m + ' needs ' + hrs + ' hours, ' +
              (good ? 'which <b>fits</b>, so the answer is not above ' + m + ' - hi comes down to mid, because mid itself may be the answer.'
                    : 'which <b>does not fit</b>, so the answer is strictly above ' + m + ' and lo jumps to mid + 1.')}});
    if (good) hi = m; else lo = m + 1;
    F.push({shapes:shapes(lo, hi, null), panels:[{t:'lo ' + lo, cls:'act'}, {t:'hi ' + hi, cls:'act'}],
      view:VIEW, line:good ? 3 : 4,
      msg:{zh:'一次模擬換掉一半的候選速度。能這樣做的<b>唯一理由</b>是 hours(speed) 單調遞減 - 速度越快只會越早結束。這個論證才是這題真正的工作。',
           en:'One simulation eliminates half the candidate speeds. The <b>only</b> reason that is legal is that hours(speed) is non-increasing - going faster can never take longer. Making that argument is the actual work in this problem.'}});
  }
  F.push({shapes:shapes(null, null, lo), panels:[{t:'answer ' + lo, cls:'ok'}, {t:n + ' probes', cls:'ok'}],
    view:VIEW, line:5,
    msg:{zh:'答案 ' + lo + '，只探測了 ' + n + ' 次。注意 hours(' + lo + ') = ' + koko(lo) +
            ' 而 hours(' + (lo - 1) + ') = ' + koko(lo - 1) + ' - <b>第一個 True</b> 就是要找的邊界，跟第一段的 lower_bound 是同一個迴圈。',
         en:'The answer is ' + lo + ' after ' + n + ' probes. Note hours(' + lo + ') = ' + koko(lo) +
            ' while hours(' + (lo - 1) + ') = ' + koko(lo - 1) + ' - the <b>first True</b> is the boundary, and this is literally the lower_bound loop from the first tab.'}});
  return F.list;
}

function qpsFrames(){
  const F = new Frames();
  const CAPQ = 13.7, JIT = .6;
  let seed = 23;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const LX = .8, RX = 9.0, LO0 = 1, HI0 = 64;
  const px = v => LX + (v - LO0) / (HI0 - LO0) * (RX - LX);
  const Y = 3.2;
  const hist = [];

  const shapes = (lo, hi, mid, verdict) => {
    const s = [];
    s.push(S.e(LX, Y, RX, Y, {s:'soft', arrow:false, w:.04}));
    s.push(S.r(px(lo), Y - .22, Math.max(px(hi) - px(lo), .04), .44, 'act', ''));
    s.push(S.r(px(CAPQ) - .015, Y - .62, .03, 1.24, 'ok', ''));
    s.push(S.t(px(CAPQ), Y + 1.02, {zh:'真正的容量 13.7', en:'true capacity 13.7'}, {c:COL.tealL, fs:.28}));
    hist.forEach(p => s.push(S.c(px(p.q), Y, .07, p.ok ? 'ok' : 'bad', '')));
    if (mid != null){
      s.push(S.c(px(mid), Y, .16, verdict == null ? 'hot' : (verdict ? 'ok' : 'bad'), ''));
      s.push(S.t(px(mid), Y - .82, String(mid), {c:COL.orangeL, fs:.30}));
    }
    [1, 8, 16, 32, 64].forEach(v => s.push(S.t(px(v), Y + .62, String(v), {c:COL.grey, fs:.26})));
    s.push(note(1.45, {zh:'ok(qps) = 這一次 benchmark 有沒有守住 SLA', en:'ok(qps) = did this benchmark run meet the SLA'}, COL.tealL, .36));
    s.push(note(2.02, {zh:'每問一次都要跑一次完整 benchmark，而且量到的數字每次都不一樣',
                       en:'every question costs a full benchmark run, and the number comes back different each time'},
                COL.grey, .28));
    s.push(foot({zh:'請求速率 qps（連續的、不是整數）', en:'request rate in qps - continuous, not an integer'}, Y + 1.55));
    return s;
  };

  let lo = 1.0, hi = 64.0, best = null, rounds = 0, flips = 0;
  F.push({shapes:shapes(lo, hi, null), panels:[{t:'bracket 1 - 64', cls:'act'}], view:VIEW, line:7,
    msg:{zh:'同一個想法，但答案是浮點數，而且 ok() 是<b>一次量測</b>不是一次比較。兩件事因此改變：迴圈不能用 lo < hi 收尾，而且 ok() 有可能<b>說謊</b>。',
         en:'Same idea, but the answer is a float and ok() is <b>a measurement</b> rather than a comparison. Two things change: the loop cannot end on lo < hi, and ok() can <b>lie</b>.'}});

  while (hi - lo > .1 && rounds < 12){
    const m = Math.round((lo + hi) / 2 * 1e4) / 1e4;
    if (m <= lo || m >= hi) break;
    const measured = Math.round((CAPQ + (rnd() * 2 - 1) * JIT) * 1e3) / 1e3;
    const good = m <= measured;
    const truth = m <= CAPQ;
    const lied = good !== truth;
    if (lied) flips++;
    F.push({shapes:shapes(lo, hi, m, null), panels:[{t:'round ' + (rounds + 1), cls:'hot'},
        {t:'qps ' + m, cls:'hot'}], view:VIEW, line:9,
      msg:{zh:'中點 ' + m + ' qps，跑一次 benchmark。<b>四捨五入的中點</b>有可能剛好落在端點上 - 那就是浮點版的無窮迴圈，所以上面那行 break 一定要在。',
           en:'The midpoint is ' + m + ' qps, so run a benchmark. A <b>rounded midpoint</b> can land exactly on an endpoint - that is the float version of the infinite loop, which is why the break above has to be there.'}});
    hist.push({q:m, ok:good});
    F.push({shapes:shapes(good ? m : lo, good ? hi : m, m, good),
      panels:[{t:'measured ' + measured, cls:good ? 'ok' : 'bad'},
              {t:good ? 'PASS' : 'FAIL', cls:good ? 'ok' : 'bad'}].concat(lied ? [{t:'disagrees with truth', cls:'bad'}] : []),
      view:VIEW, line:good ? 12 : 13,
      msg:lied ? {zh:'量到 ' + measured + '，判定 ' + (good ? 'PASS' : 'FAIL') + ' - 但真正的容量是 13.7，<b>這一次判斷是錯的</b>。二分搜尋<b>不會回頭</b>：被丟掉的那一半永遠不會再看一眼，所以中段的一次雜訊就把答案定死了。',
                 en:'Measured ' + measured + ', verdict ' + (good ? 'PASS' : 'FAIL') + ' - but the true capacity is 13.7, so <b>this call was wrong</b>. Binary search <b>never goes back</b>: the discarded half is never looked at again, so one noisy answer in the middle fixes the result.'}
               : {zh:'量到 ' + measured + '，' + (good ? '通過 - 記下 best 並把下界推上去' : '沒過 - 上界收下來') + '。留住 <b>best</b> 而不是只留區間，是因為最後要回報的是「確實跑過而且過關」的那個數字。',
                  en:'Measured ' + measured + ', so it ' + (good ? 'passes - record best and push the lower bound up' : 'fails - pull the upper bound down') + '. Keeping <b>best</b> rather than just the bracket matters because the number reported has to be one that actually ran and actually passed.'}});
    if (good){ lo = m; best = m; } else hi = m;
    rounds++;
  }
  F.push({shapes:shapes(lo, hi, best, true),
    panels:[{t:'best ' + best, cls:'ok'}, {t:rounds + ' runs', cls:'ok'}, {t:flips + ' lies', cls:flips ? 'bad' : 'ok'}],
    view:VIEW, line:14,
    msg:{zh:'停在 ' + rounds + ' 回合、best = ' + best + '，其中 <b>' + flips + ' 次量測跟真值不一致</b>。這就是為什麼真的 autotuner 帶著 <b>round budget</b>、回報跑過的紀錄、而且容忍度要開得比雜訊大 - 收斂到小數點後三位只是把雜訊當成訊號。',
         en:'It stops after ' + rounds + ' rounds with best = ' + best + ', and <b>' + flips + ' of those measurements disagreed with the truth</b>. That is why a real autotuner carries a <b>round budget</b>, reports a record that actually ran, and sets a tolerance wider than the noise - converging to three decimals just means treating noise as signal.'}});
  return F.list;
}

/* ================================================= 4. galloping prefix match */
const GN = 32, GP = 21;
const CODE_G = [
'lo, step = 0, 1',
'while lo < n:',
'    hi = min(lo + step, n)',
'    if t0[lo:hi] != t1[lo:hi]:        # this window broke',
'        while hi - lo > 1:            # binary search inside it',
'            mid = (lo + hi) // 2',
'            if t0[lo:mid] == t1[lo:mid]: lo = mid',
'            else:                        hi = mid',
'        return lo',
'    lo, step = hi, step * 2'
];

function gallopFrames(){
  const F = new Frames();
  const w = .285, h = .58, Y = 3.1, x0 = (VIEW[0] - GN * w) / 2;
  let probes = 0, tokens = 0;

  const shapes = (known, lo, hi, phase) => {
    const s = [];
    for (let i = 0; i < GN; i++){
      let st = 'ghost';
      if (i < known) st = 'ok';
      else if (i >= lo && i < hi) st = 'hot';
      s.push(S.r(x0 + i * w, Y, w - .05, h, st, '', {}));
    }
    s.push(S.c(x0 + GP * w + (w - .05) / 2, Y + h + .52, .13, 'bad', ''));
    s.push(S.t(x0 + GP * w, Y + h + 1.00, {zh:'第一個不同的 token', en:'first differing token'},
               {c:COL.red, fs:.28}));
    if (hi > lo){
      s.push(S.e(x0 + lo * w, Y - .34, x0 + hi * w - .05, Y - .34, {s:'hot', arrow:false, w:.07}));
      s.push(S.t((x0 + lo * w + x0 + hi * w) / 2, Y - .54,
                 (phase === 'g' ? 'step ' : 'window ') + (hi - lo), {c:COL.orangeL, fs:.28}));
    }
    s.push(note(1.50, {zh:'兩條 token 序列共用多長的前綴？', en:'how long a prefix do these two token sequences share?'},
                COL.tealL, .36));
    s.push(note(2.05, {zh:'比較是整段一起送下去比，所以成本不是「幾次」而是「碰了幾個 token」',
                       en:'each compare hands a whole slice to C, so the cost is tokens touched, not probes made'},
                COL.grey, .28));
    s.push(foot({zh:'綠色 = 已經確定相同的前綴，灰色 = 還沒看過',
                 en:'green = prefix already known equal, grey = never looked at'}, Y + h + 1.55));
    return s;
  };

  F.push({shapes:shapes(0, 0, 0), panels:[{t:'n = ' + GN, cls:'act'}], view:VIEW, line:0,
    msg:{zh:'SGLang 的 radix cache 每一個進來的 request 都要問這件事。它<b>不</b>從頭一格一格比，也<b>不</b>直接對整條做二分搜尋。',
         en:'SGLang\'s radix cache asks this on every incoming request. It does <b>not</b> walk token by token, and it does <b>not</b> binary search the whole range either.'}});

  let lo = 0, step = 1, matched = GN;
  while (lo < GN){
    const hi = Math.min(lo + step, GN);
    probes++; tokens += hi - lo;
    const differs = GP < hi;
    F.push({shapes:shapes(lo, lo, hi, 'g'), panels:[{t:'probe ' + probes, cls:'hot'},
        {t:'window [' + lo + ', ' + hi + ')', cls:'hot'}, {t:'tokens ' + tokens, cls:'act'}],
      view:VIEW, line:3,
      msg:{zh:'比較長度 ' + (hi - lo) + ' 的<b>新視窗</b>，不是重比整段前綴 - lo 之前已經確定相同，沒有必要再碰。視窗每次加倍，' +
              (differs ? '而這一段<b>不一樣</b>了。' : '這一段一樣，lo 直接跳到 ' + hi + '。'),
           en:'Compare a <b>fresh window</b> of length ' + (hi - lo) + ' rather than the whole prefix again - everything before lo is already known equal. The window doubles each time, and ' +
              (differs ? 'this one <b>differs</b>.' : 'this one matches, so lo jumps straight to ' + hi + '.')}});
    if (differs){
      let a = lo, b = hi;
      F.push({shapes:shapes(a, a, b, 'b'), panels:[{t:'broken window [' + a + ', ' + b + ')', cls:'bad'}],
        view:VIEW, line:4,
        msg:{zh:'分歧點就在這個視窗裡面，<b>而且只在這裡</b>。現在才對這一段做二分搜尋 - 前面所有加倍的視窗長度加起來大約是 2p，所以整體成本跟<b>共用前綴的長度</b>成正比，跟陣列多長無關。',
             en:'The divergence is inside this window and <b>only</b> here. Only now does a binary search run, on this window alone. The doubling windows sum to about 2p, so the total cost tracks the <b>length of the shared prefix</b>, not the length of the arrays.'}});
      while (b - a > 1){
        const m = (a + b) >> 1;
        probes++; tokens += m - a;
        const eq = GP >= m;
        F.push({shapes:shapes(a, a, m, 'b'), panels:[{t:'mid ' + m, cls:'hot'},
            {t:eq ? 'equal' : 'differs', cls:eq ? 'ok' : 'bad'}, {t:'tokens ' + tokens, cls:'act'}],
          view:VIEW, line:5,
          msg:{zh:'比 [' + a + ', ' + m + ')：' + (eq ? '相同，lo 推到 ' + m + '。這裡寫 <b>lo = mid</b> 沒問題，因為迴圈條件是 hi - lo > 1，mid 一定嚴格大於 lo。'
                                                     : '不同，hi 收到 ' + m + '。') ,
               en:'Compare [' + a + ', ' + m + '): ' + (eq ? 'equal, so lo moves up to ' + m + '. Writing <b>lo = mid</b> is safe here because the loop condition is hi - lo > 1, which forces mid strictly above lo.'
                                                           : 'they differ, so hi comes down to ' + m + '.')}});
        if (eq) a = m; else b = m;
      }
      matched = a;
      lo = GN;
    } else { lo = hi; step *= 2; }
  }
  F.push({shapes:shapes(matched, 0, 0), panels:[{t:'matched ' + matched, cls:'ok'},
      {t:probes + ' probes', cls:'ok'}, {t:tokens + ' tokens', cls:'ok'}], view:VIEW, line:8,
    msg:{zh:'共用前綴 ' + matched + '，碰了 ' + tokens + ' 個 token。<b>直接二分搜尋整條</b>的 probe 次數更少（log n），但每個 probe 要比一整段前綴，token 數是 n log n 等級。galloping 不是全面比較好 - p 接近 n/2 時它的 probe 數大約是兩倍。它適合前綴快取，是因為那個 workload 是<b>兩極</b>的：新對話幾乎不共用，追問幾乎全共用。',
         en:'A shared prefix of ' + matched + ', touching ' + tokens + ' tokens. <b>Plain binary search</b> makes fewer probes (log n), but each probe compares an entire prefix, so its token count is n log n. Galloping is not uniformly better - near p = n/2 it makes about twice the probes. It is the right bet for a prefix cache because that workload is <b>bimodal</b>: a new conversation shares almost nothing, a follow-up shares almost everything.'}});
  return F.list;
}

/* ============================================ 5. undoing a prefix sum */
const SEQ = [3, 5, 2];
const CU = [0, 3, 8, 10];
const CODE_C2 = [
'cu = [0]                      # yesterday: the exclusive scan',
'for L in seq_lens:',
'    cu.append(cu[-1] + L)',
'',
'# every kernel that works one token at a time undoes it:',
'seq_of = searchsorted(cu, tok, right=True) - 1',
'#                              right=False is off by one'
];

function cuFrames(right){
  const F = new Frames();
  const N = 10, w = .78, h = .70;
  const YT = 4.0, YC = 1.9;
  const x0 = rowX(N, w);
  const cx0 = x0;
  const truth = [];
  SEQ.forEach((L, i) => { for (let k = 0; k < L; k++) truth.push(i); });
  const got = [];

  const shapes = (tok, res) => {
    const ts = {}, cs = {};
    for (let i = 0; i < N; i++)
      ts[i] = i < got.length ? (got[i] === truth[i] ? 'ok' : 'bad') : (i === tok ? 'hot' : 'idle');
    let s = [];
    CU.forEach((v, i) => {
      cs[i] = (res != null && i === res) ? 'act' : 'soft';
      s.push(S.r(cx0 + v * w, YC, w - .06, h, cs[i], String(v), {fs:.36}));
    });
    let r = cellRow([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], x0, YT, w, h, {states:ts, index:false});
    s.push(S.t(x0 - .22, YT + h * .62, {zh:'token', en:'token'}, {c:COL.tealL, fs:.30, anchor:'end'}));
    s.push(S.t(cx0 - .22, YC + h * .62, 'cu_seqlens', {c:COL.tealL, fs:.30, anchor:'end'}));
    s = s.concat(r);
    if (tok != null && res != null && res >= 0){
      const bx = x0 + tok * w + (w - .06) / 2;
      const tx = cx0 + CU[res] * w + (w - .06) / 2;
      s.push(S.e(bx, YT - .08, tx, YC + h + .10, {s:res === truth[tok] ? 'ok' : 'bad'}));
    }
    s.push(note(.95, right ? 'searchsorted(cu, tok, right=True) - 1'
                           : 'searchsorted(cu, tok, right=False) - 1',
                right ? COL.tealL : COL.red, .36));
    s.push(foot({zh:'seq_lens = 3, 5, 2 - 三個 request 打平成一條 token 陣列',
                 en:'seq_lens = 3, 5, 2 - three requests flattened into one token array'}, 6.1));
    return s;
  };

  F.push({shapes:shapes(null, null), panels:[{t:'seq_lens ' + SEQ.join(' '), cls:'act'},
      {t:'cu ' + CU.join(' '), cls:'act'}], view:VIEW, line:2,
    msg:{zh:'昨天的 counting sort 用 exclusive scan 把直方圖變成位址表。<b>今天要把它倒回去</b>：給一個打平的 token index，它屬於哪一個 request？倒過來的那個動作就是邊界搜尋。',
         en:'Yesterday counting sort turned a histogram into an offset table with an exclusive scan. <b>Today it gets undone</b>: given a flat token index, which request owns it? Undoing the scan is a boundary search.'}});

  for (let tok = 0; tok < N; tok++){
    let a = 0, b = CU.length;               // the real bisect, not a scan
    while (a < b){
      const m = a + ((b - a) >> 1);
      if (right ? CU[m] <= tok : CU[m] < tok) a = m + 1; else b = m;
    }
    const res = a - 1;
    got.push(res);
    const ok = res === truth[tok];
    const boundary = CU.indexOf(tok) >= 0 && tok < CU[CU.length - 1];
    F.push({shapes:shapes(tok, res), panels:[{t:'token ' + tok, cls:'hot'},
        {t:'seq ' + res, cls:ok ? 'ok' : 'bad'}, {t:'truth ' + truth[tok], cls:'act'}],
      view:VIEW, line:5,
      msg:ok ? {zh:'token ' + tok + ' -> request ' + res + '。<b>最後一個 <= tok 的起點</b>，這正是「我在哪一桶」的定義。' +
                    (boundary ? ' 而且這是一個 request 的<b>第一個 token</b> - 邊界寫法會不會錯全看這一格。' : ''),
                en:'token ' + tok + ' -> request ' + res + '. The <b>last start that is <= tok</b>, which is the definition of which bucket a value falls in.' +
                    (boundary ? ' And this is the <b>first token of a request</b> - the whole boundary question lives in exactly these cells.' : '')}
             : {zh:'token ' + tok + ' 拿到 request ' + res + '，正確答案是 ' + truth[tok] + '。right=False 問的是「第一個 >= tok 的起點」，而在邊界上那個起點<b>就是下一個 request 自己</b>。' +
                    (tok === 0 ? ' token 0 更慘：它拿到 -1，然後 -1 會去索引 cu_seqlens 的最後一格，讀到別人的狀態，<b>而且不會報錯</b>。' : ''),
                en:'token ' + tok + ' gets request ' + res + ' where the answer is ' + truth[tok] + '. right=False asks for the first start >= tok, and at a boundary that start <b>is the next request itself</b>.' +
                    (tok === 0 ? ' Token 0 is worse: it gets -1, which then indexes the last entry of cu_seqlens and reads another request\'s state, <b>raising nothing</b>.' : '')}});
  }
  const bad = [];
  for (let i = 0; i < N; i++) if (got[i] !== truth[i]) bad.push(i);
  F.push({shapes:shapes(null, null), panels:[{t:bad.length ? 'wrong at ' + bad.join(' ') : 'all correct',
      cls:bad.length ? 'bad' : 'ok'}], view:VIEW, line:5,
    msg:bad.length ? {zh:'錯的是 token ' + bad.join(', ') + ' - <b>每一個 request 的第一個 token</b>，一個不多一個不少。一個字元的差別，錯在每一次 batch 的每一個邊界上，而且從來不會拋例外。',
                      en:'The wrong ones are tokens ' + bad.join(', ') + ' - <b>the first token of every request</b>, no more and no less. One character of difference, wrong at every boundary of every batch, and it never raises.'}
                   : {zh:'全部正確。這條 searchsorted 出現在幾乎每個 batched kernel 裡，而 right=True <b>不是風格選擇</b> - 是對與不對的差別。',
                      en:'All correct. This searchsorted line sits in nearly every batched kernel, and right=True <b>is not a style choice</b> - it is the difference between right and wrong.'}});
  return F.list;
}

/* ================================================== 6. LC 34 */
const L34 = [5, 7, 7, 8, 8, 8, 10];
const CODE_L = [
'def search_range(nums, target):',
'    lo = lower_bound(nums, target)      # first index >= target',
'    hi = upper_bound(nums, target)      # first index >  target',
'    return [lo, hi - 1] if lo < hi else [-1, -1]'
];

function lcFrames(v){
  const target = v ? 6 : 8;
  const F = new Frames();
  const n = L34.length, x0 = rowX(n, CW), Y = 3.2;
  const shapes = (lo, hi, m, phase, hit) => {
    const st = {};
    for (let i = 0; i < n; i++) st[i] = i < lo ? 'done' : (i >= hi ? 'ghost' : 'idle');
    if (m != null) st[m] = 'hot';
    if (hit) for (let i = hit[0]; i <= hit[1]; i++) st[i] = 'ok';
    let s = cellRow(L34, x0, Y, CW, CH, {states:st, index:true});
    s = s.concat(marks(x0, Y - 1.02, CW, {lo:lo, hi:hi, mid:m}));
    s.push(note(5.60, 'search_range(nums, ' + target + ')', COL.tealL, .38));
    s.push(note(5.10, phase, COL.grey, .28));
    return s;
  };
  const run = (upper, lineNo) => {
    let lo = 0, hi = n;
    const phase = upper ? {zh:'第二次：upper_bound - 第一個 > target 的位置',
                           en:'pass two: upper_bound - the first index strictly greater than target'}
                        : {zh:'第一次：lower_bound - 第一個 >= target 的位置',
                           en:'pass one: lower_bound - the first index at or after target'};
    F.push({shapes:shapes(lo, hi, null, phase), panels:[{t:upper ? 'upper_bound' : 'lower_bound', cls:'act'}],
      view:VIEW, line:lineNo,
      msg:{zh:'同一個迴圈跑第二次，只有比較符號不同 - <b>a[mid] ' + (upper ? '<=' : '<') + ' target</b>。',
           en:'The same loop, run again with one comparison changed - <b>a[mid] ' + (upper ? '<=' : '<') + ' target</b>.'}});
    while (lo < hi){
      const m = lo + ((hi - lo) >> 1);
      const small = upper ? L34[m] <= target : L34[m] < target;
      F.push({shapes:shapes(lo, hi, m, phase), panels:[{t:'mid ' + m + ' -> ' + L34[m], cls:'hot'},
          {t:small ? 'go right' : 'keep mid', cls:'act'}], view:VIEW, line:lineNo,
        msg:{zh:'a[' + m + '] = ' + L34[m] + (small ? ' 還在 target 的左邊（含相等，因為這一趟用 <=），lo = mid + 1。'
                                                    : '，mid 有可能就是邊界，hi = mid。'),
             en:'a[' + m + '] = ' + L34[m] + (small ? ' is still on the left of the boundary' + (upper ? ' (ties count as left this pass, because of <=)' : '') + ', so lo = mid + 1.'
                                                    : ', and mid may itself be the boundary, so hi = mid.')}});
      if (small) lo = m + 1; else hi = m;
    }
    return lo;
  };
  const lo = run(false, 1);
  const hi = run(true, 2);
  const found = lo < hi;
  F.push({shapes:shapes(lo, hi, null, {zh:'', en:''}, found ? [lo, hi - 1] : null),
    panels:[{t:'lower ' + lo, cls:'ok'}, {t:'upper ' + hi, cls:'ok'},
            {t:found ? '[' + lo + ', ' + (hi - 1) + ']' : '[-1, -1]', cls:found ? 'ok' : 'bad'}],
    view:VIEW, line:3,
    msg:found ? {zh:'兩個邊界一夾就是答案 [' + lo + ', ' + (hi - 1) + ']，出現次數 = upper - lower = ' + (hi - lo) + '。整題的邏輯只有一行。',
                 en:'The two boundaries bracket the answer [' + lo + ', ' + (hi - 1) + '], and the count is upper - lower = ' + (hi - lo) + '. The entire problem is one line of logic.'}
              : {zh:'target 不在陣列裡，兩個邊界撞在同一格（' + lo + '），<b>lo == hi 自己就是 not found</b> - 不需要另外寫一個分支，也不需要 sentinel。這就是為什麼值得記住的是邊界版本，而不是「找 x」的版本。',
                 en:'The target is absent and both boundaries land on the same index (' + lo + '), so <b>lo == hi is the not-found case</b> - no extra branch, no sentinel. That is why the boundary version is the one worth memorising, not the find-x one.'}});
  return F.list;
}

/* ===================================================================== meta */
const DAY_META = {
  title:{zh:'Day 23 · 二分搜尋的三種邊界寫法', en:'Day 23 · Binary search and its three boundary variants'},
  sub:{zh:'二分搜尋很少是「找 x 在哪」。真正在用的版本是找邊界：第一個成立的位置。邊界寫錯不會報錯，只會安靜地少一格。',
       en:'Binary search is rarely "find x". The version people actually run finds a boundary - the first position where something becomes true. A wrong boundary does not crash, it is quietly off by one.'},
  tabs:[
    {id:'bounds', label:{zh:'lower / upper bound', en:'lower / upper bound'},
     stage:{zh:'半開區間 [lo, hi)：一個不變式，兩個版本，差一個字元',
            en:'the half-open range [lo, hi) - one invariant, two variants, one character apart'},
     view:VIEW,
     variants:[{zh:'lower_bound（a[mid] < x）', en:'lower_bound (a[mid] < x)'},
               {zh:'upper_bound（a[mid] <= x）', en:'upper_bound (a[mid] <= x)'}],
     idea:{zh:'整個迴圈只維護一句話：<b>a[:lo] 已知太小，a[hi:] 已知夠大</b>。中間那段還不知道。因為不變式從頭到尾成立，<b>lo == hi 的那一刻它就是答案</b> - 不需要提早 return，不需要「找不到」的特例。lower_bound 和 upper_bound 差在 <b>&lt;</b> 和 <b>&lt;=</b> 一個字元，也就是「相等要不要算在左邊」，而 upper - lower 就是出現次數。',
           en:'The loop maintains exactly one sentence: <b>a[:lo] is known too small, a[hi:] is known big enough</b>, and the middle is unknown. Because the invariant holds throughout, <b>the moment lo == hi that position is the answer</b> - no early return, no special not-found case. lower_bound and upper_bound differ by one character, <b>&lt;</b> versus <b>&lt;=</b>, which is just "do ties count as left", and upper - lower is the number of occurrences.'},
     legend:['hot', 'act', 'done', 'ghost'],
     code:CODE_B,
     build:(v) => norm(boundFrames(!!v))},
    {id:'bugs', label:{zh:'三個看起來對的迴圈', en:'three loops that look right'},
     stage:{zh:'兩個掛住、一個安靜地錯 - 危險的是會結束的那個',
            en:'two hang, one is quietly wrong - the dangerous one is the one that terminates'},
     view:VIEW,
     variants:[{zh:'(a) hi = mid 配閉區間', en:'(a) hi = mid on a closed range'},
               {zh:'(b) lo = mid', en:'(b) lo = mid'},
               {zh:'(c) hi = mid - 1', en:'(c) hi = mid - 1'}],
     idea:{zh:'二分搜尋只有兩種錯法：<b>區間沒有變小</b>（無窮迴圈），或<b>區間丟掉了答案</b>（安靜地錯）。(a) 和 (b) 屬於前者 - 至少會被發現。(c) 屬於後者：它<b>會結束</b>、不丟例外、回傳一個看起來很合理的數字，而 mid 那一格正好就是它該留下來的答案。順帶一提 <b>mid = lo + (hi - lo) // 2</b> 不是潔癖：在 int32 的 kernel 裡 (lo + hi) 會溢位成負數，那是 2006 年 JDK binarySearch 的那個 bug。',
           en:'There are only two ways to get binary search wrong: <b>the range fails to shrink</b> (an infinite loop) or <b>the range throws away the answer</b> (a silent error). (a) and (b) are the first kind and at least get noticed. (c) is the second: it <b>terminates</b>, raises nothing, returns a plausible number, and the cell it discarded was exactly the answer it was meant to keep. And <b>mid = lo + (hi - lo) // 2</b> is not fastidiousness: in an int32 kernel (lo + hi) overflows negative, which is the 2006 JDK binarySearch bug.'},
     legend:['hot', 'bad', 'done', 'ghost'],
     code:CODE_X,
     build:(v) => norm(bugFrames(v))},
    {id:'answer', label:{zh:'在答案上二分', en:'binary search on the answer'},
     stage:{zh:'沒有陣列，只有一個單調的 ok()',
            en:'no array anywhere - just a monotone ok()'},
     view:VIEW,
     variants:[{zh:'Koko 吃香蕉（整數）', en:'Koko eating bananas (integer)'},
               {zh:'最大 qps（浮點 + 雜訊）', en:'max sustainable qps (float, noisy)'}],
     idea:{zh:'二分搜尋真正需要的只有<b>一個單調的述詞</b>：ok(x) 在範圍上是 F F F ... T T T。排序好的陣列只是 ok(i) = a[i] >= x 這個特例。所以配方變成：猜答案是什麼的數字、寫出 ok(candidate)（通常是一次<b>模擬</b>而不是一次比較）、<b>論證它單調</b>、然後跑同一個 lower_bound 迴圈。第三步才是全部的工作 - ok 不單調時迴圈照樣結束、照樣回傳一個數字，只是那個數字沒有意義。浮點版還多兩個坑：終止條件變成容忍度加回合上限，而且四捨五入的中點會塌到端點上。',
           en:'All binary search actually needs is <b>a monotone predicate</b>: ok(x) is F F F ... T T T over the range. A sorted array is just the special case ok(i) = a[i] >= x. So the recipe becomes: guess what number the answer is, write ok(candidate) - usually a <b>simulation</b> rather than a comparison - <b>argue that it is monotone</b>, then run the same lower_bound loop. Step three is the entire job: if ok is not monotone the loop still terminates and still returns a number, and the number is meaningless. The float version adds two traps: termination becomes a tolerance plus a round budget, and a rounded midpoint can collapse onto an endpoint.'},
     legend:['hot', 'ok', 'bad', 'ghost'],
     code:CODE_A,
     build:(v) => norm(v ? qpsFrames() : kokoFrames())},
    {id:'gallop', label:{zh:'Galloping（radix cache）', en:'galloping (radix cache)'},
     stage:{zh:'成本跟著共用前綴走，不跟著陣列長度走',
            en:'cost follows the shared prefix, not the length of the arrays'},
     view:VIEW,
     idea:{zh:'SGLang 的 radix cache 每個 request 都要問「這兩條 token 共用多長的前綴」。它先用<b>加倍的視窗</b> 1、2、4、8... 往前跑，直到某個視窗不一樣，再<b>只對那個視窗</b>做二分搜尋。關鍵在成本模型：直接對整條二分搜尋的 probe 次數是 log n 很漂亮，但每個 probe 比的是一整段前綴，碰到的 token 數是 n log n 等級；galloping 的視窗長度加起來大約 2p，所以是 <b>O(p)</b>。它不是全面比較好 - p 接近 n/2 時 probe 數大約兩倍。會賭這一把，是因為前綴快取的 workload 本來就是兩極的。',
           en:'SGLang\'s radix cache asks "how long a prefix do these two token sequences share?" on every request. It runs <b>doubling windows</b> of 1, 2, 4, 8, ... until one differs, then binary searches <b>that window alone</b>. The point is the cost model: binary searching the whole range makes a beautiful log n probes, but each probe compares a whole prefix, so tokens touched grow like n log n. The doubling windows sum to about 2p, making galloping <b>O(p)</b>. It is not uniformly better - near p = n/2 it makes roughly twice the probes. The bet pays off because a prefix cache workload is bimodal.'},
     legend:['hot', 'ok', 'bad', 'ghost'],
     code:CODE_G,
     build:() => norm(gallopFrames())},
    {id:'cu', label:{zh:'把 prefix sum 倒回去', en:'undoing a prefix sum'},
     stage:{zh:'right=True 不是風格問題，是對不對的問題',
            en:'right=True is not a style choice - it is correctness'},
     view:VIEW,
     variants:[{zh:'right=True（正確）', en:'right=True (correct)'},
               {zh:'right=False（每個邊界都錯）', en:'right=False (wrong at every boundary)'}],
     idea:{zh:'昨天的 exclusive scan 把每個 request 的長度變成 cu_seqlens；今天每個一次處理一個 token 的 kernel 都要把它倒回去：<b>searchsorted(cu, tok, right=True) - 1</b>。這句話讀出來就是「最後一個 <= tok 的起點」，也就是「我在哪一桶」的定義 - 正是 upper_bound(x) - 1。換成 right=False 問的是「第一個 >= tok 的起點」，在邊界上那是<b>下一個</b> request，於是每個 request 的<b>第一個 token</b> 都算錯，而 token 0 會拿到 -1 去讀最後一格。全程不丟例外。',
           en:'Yesterday\'s exclusive scan turned per-request lengths into cu_seqlens; today every kernel that works one token at a time has to undo it: <b>searchsorted(cu, tok, right=True) - 1</b>. Read out loud that is "the last start that is <= tok", the definition of which bucket a value falls in - exactly upper_bound(x) - 1. Switch to right=False and it asks for the first start >= tok, which at a boundary is the <b>next</b> request, so the <b>first token of every request</b> is wrong and token 0 gets -1 and reads the last entry. Nothing raises anywhere.'},
     legend:['hot', 'ok', 'bad', 'act'],
     code:CODE_C2,
     build:(v) => norm(cuFrames(!v))},
    {id:'lc34', label:{zh:'LC 34 · First and Last Position', en:'LC 34 · First and Last Position'},
     stage:{zh:'兩個邊界一夾，連 not found 都不用寫',
            en:'two boundaries bracket the answer - even not-found writes itself'},
     view:VIEW,
     variants:[{zh:'target = 8（存在）', en:'target = 8 (present)'},
               {zh:'target = 6（不存在）', en:'target = 6 (absent)'}],
     idea:{zh:'這題是「為什麼要記邊界版本」的最好證據：有了 lower_bound 和 upper_bound，答案是 <b>[lo, hi - 1]</b>，出現次數是 <b>hi - lo</b>，而「找不到」自己從 <b>lo == hi</b> 掉出來 - 不需要額外的分支，也不需要 sentinel。相對地，用「找到 x 就 return」的版本開場，接下來就得往兩邊線性掃，最壞情況掉回 O(n)。',
           en:'This problem is the best argument for memorising the boundary version: with lower_bound and upper_bound the answer is <b>[lo, hi - 1]</b>, the count is <b>hi - lo</b>, and not-found falls out of <b>lo == hi</b> with no extra branch and no sentinel. Start instead from the "return as soon as a[mid] == x" version and you have to scan outwards from the hit, which is O(n) in the worst case.'},
     legend:['hot', 'ok', 'done', 'ghost'],
     code:CODE_L,
     build:(v) => norm(lcFrames(v))}
  ]
};
