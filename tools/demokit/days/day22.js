// DAY: 22
// TITLE_ZH: 不用比較的排序：counting、radix、bucket
// TITLE_EN: Sorting without comparing - counting, radix and bucket sort
// SUB_ZH: n log n 是「只能問 a < b?」的下限。不問這個問題，把 key 直接當成位址，下限就不適用了。
// SUB_EN: n log n bounds sorts that only ask "a < b?". Use the key as an address instead and the bound simply does not apply.
// FOLDER: day%2022%20-%20counting%20radix%20and%20bucket%20sort
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const CW = .8, CH = .72;
const rowX = n => (VIEW[0] - n * CW) / 2;
const note = (y, s, c) => S.t(VIEW[0] / 2, y, s, {c:c || COL.tealL, fs:.32});
const foot = s => S.t(VIEW[0] / 2, 6.18, s, {c:COL.grey, fs:.27});

/* ===================================================== 1. counting sort */
const CK = [4, 1, 3, 1, 0, 4, 1, 2];
const CP = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const K = 5;

const CODE_C = [
'def counting_sort(keys, k, payload):',
'    counts = [0] * k',
'    for v in keys:                 # count',
'        counts[v] += 1',
'    total, starts = 0, [0] * k     # scan',
'    for v in range(k):',
'        starts[v] = total',
'        total += counts[v]',
'    out = [None] * len(keys)       # scatter',
'    cursor = list(starts)',
'    for i, v in enumerate(keys):',
'        out[cursor[v]] = payload[i]',
'        cursor[v] += 1',
'    return out'
];

function countFrames(){
  const F = new Frames();
  const counts = new Array(K).fill(0);
  const starts = new Array(K).fill(0);
  const out = new Array(CK.length).fill(null);
  let cursor = null;

  const Y_IN = .75, Y_PAY = 1.55, Y_CNT = 2.9, Y_SCN = 3.95, Y_OUT = 5.15;
  const bx = rowX(K);

  const shapes = (o) => {
    o = o || {};
    let s = [];
    s = s.concat(cellRow(CK, rowX(8), Y_IN, CW, CH,
      {states:o.inSt || {}, title:{zh:'keys', en:'keys'}, index:false}));
    s = s.concat(cellRow(CP, rowX(8), Y_PAY, CW, CH,
      {states:o.paySt || {}, title:{zh:'payload', en:'payload'}, index:false}));
    s = s.concat(cellRow(counts, bx, Y_CNT, CW, CH,
      {states:o.cSt || {}, title:{zh:'count', en:'count'},
       labels:['key 0', 'key 1', 'key 2', 'key 3', 'key 4']}));
    if (o.showScan)
      s = s.concat(cellRow(o.cursorRow || starts, bx, Y_SCN, CW, CH,
        {states:o.sSt || {}, title:o.cursorRow ? {zh:'cursor', en:'cursor'} : {zh:'scan', en:'scan'},
         index:false}));
    s = s.concat(cellRow(out.map(v => v === null ? '' : v), rowX(8), Y_OUT, CW, CH,
      {states:o.oSt || {}, title:{zh:'output', en:'output'}, index:false}));
    if (o.note) s.push(note(2.45, o.note, o.noteC));
    s.push(foot({zh:'整張圖裡沒有任何一次 a < b 的比較',
                 en:'not one a < b comparison anywhere on this page'}));
    return s;
  };
  const panels = (extra) => [
    {lbl:{zh:'counts', en:'counts'},
     chips:counts.map((c, i) => ({t:i + ':' + c, cls:c ? 'ok' : 'dim'}))},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:{zh:'key 是 0..4 的小整數 - 這就是全部的前提',
                               en:'the keys are small integers in 0..4 - that is the entire precondition'}}),
          panels:panels(), view:VIEW, line:0,
          msg:{zh:'比較排序的下限來自「只能問 a &lt; b?」。counting sort 一次都不問：因為 key 本身就是 <b>0 到 k-1 的整數</b>，它可以直接把 key 當成陣列<b>位址</b>。下限講的是問題，不是演算法，所以這裡不是反例 - 是根本不在那個模型裡。',
               en:'The n log n bound comes from only ever asking "a &lt; b?". Counting sort never asks: because a key is an <b>integer in 0..k-1</b>, it can use the key itself as an <b>address</b>. The bound is about questions, not about algorithms, so this is not a counterexample - it is simply outside the model.'}});

  for (let i = 0; i < CK.length; i++){
    counts[CK[i]]++;
    const cSt = {}; cSt[CK[i]] = 'hot';
    const inSt = {}; inSt[i] = 'hot';
    F.push({shapes:shapes({inSt:inSt, cSt:cSt,
              note:{zh:'counts[' + CK[i] + '] += 1',
                    en:'counts[' + CK[i] + '] += 1'}}),
            panels:panels([{t:'key ' + CK[i], cls:'hot'}]), view:VIEW, line:3,
            msg:i === 0
              ? {zh:'第一階段只做一件事：<b>數</b>。每個元素只被摸一次，而且摸的是 counts[v] 這個位址，不是別的元素 - 所以這一趟可以每個 thread 一個元素、用 atomic 撞同一張表，完全平行。',
                 en:'Phase one does one thing: <b>count</b>. Each element is touched once, and what it touches is the address counts[v] - never another element. That is why this pass parallelises on sight: one thread per element, atomics into one shared table.'}
              : {zh:'繼續數。注意這裡完全沒有「順序」的概念 - 數完之後我們只知道每個 key 有幾個，不知道誰在誰前面。順序是下一步從位址生出來的。',
                 en:'Keep counting. Notice there is no notion of order here at all - after this pass we know how many of each key there are, not which came first. The order is manufactured from addresses in the next step.'}});
  }

  let total = 0;
  for (let v = 0; v < K; v++){ starts[v] = total; total += counts[v]; }
  F.push({shapes:shapes({showScan:true, sSt:{0:'ok', 1:'ok', 2:'ok', 3:'ok', 4:'ok'},
            note:{zh:'exclusive prefix sum：counts → starts',
                  en:'exclusive prefix sum: counts -> starts'}, noteC:COL.orangeL}),
          panels:panels([{t:'starts ' + starts.join(' '), cls:'ok'}]), view:VIEW, line:6,
          msg:{zh:'這一行是整個演算法的轉折點。scan 之後，第 v 格的意思<b>不再是「有幾個 v」</b>，而是「有幾個元素排在第一個 v 前面」- 也就是 v 這一桶的<b>起始位址</b>。直方圖變成了位址表。順帶一提，這一步是三個階段裡唯一不能直接平行的，另一個 tab 專門講它。',
               en:'This line is where the algorithm turns. After the scan, entry v no longer means <b>"how many v"</b> but "how many elements come before the first v" - the <b>starting address</b> of bucket v. The histogram has become an address table. It is also the only one of the three phases that does not parallelise for free; the scan tab is about exactly that.'}});

  cursor = starts.slice();
  for (let i = 0; i < CK.length; i++){
    const v = CK[i], pos = cursor[v];
    out[pos] = CP[i];
    cursor[v]++;
    const inSt = {}; inSt[i] = 'hot';
    const paySt = {}; paySt[i] = 'act';
    const sSt = {}; sSt[v] = 'act';
    const oSt = {}; oSt[pos] = 'ok';
    F.push({shapes:shapes({inSt:inSt, paySt:paySt, showScan:true, cursorRow:cursor.slice(),
              sSt:sSt, oSt:oSt,
              note:{zh:'out[' + pos + '] = ' + CP[i] + '　然後 cursor[' + v + '] 前進一格',
                    en:'out[' + pos + '] = ' + CP[i] + ', then cursor[' + v + '] moves on'}}),
            panels:panels([{t:'key ' + v + ' -> slot ' + pos, cls:'act'}]),
            view:VIEW, line:11,
            msg:i === 0
              ? {zh:'第三階段：<b>散</b>。每個元素直接寫到 cursor[key]，寫完 cursor 往前一格。payload（這裡是字母）跟著 key 一起被搬 - counting sort 本質上是個<b>排列</b>，任何跟 key 同索引的東西都能一起搬，MoE kernel 就是這樣把 routing weight 帶著走的。',
                 en:'Phase three: <b>scatter</b>. Each element is written straight to cursor[key], and the cursor moves on. The payload (the letters) travels with the key - counting sort is fundamentally a <b>permutation</b>, so anything indexed like the keys can ride along. That is how the MoE kernel carries routing weights next to token ids.'}
              : {zh:'因為我們從左往右走、cursor 也往右長，<b>相同 key 的元素會保持輸入順序</b> - 這就是穩定性。它現在看起來只是個好性質，到 radix sort 那一頁會變成正確性條件。',
                 en:'Because we walk the input left to right and the cursors also grow to the right, <b>equal keys keep their input order</b> - that is stability. Right now it looks like a nice property; on the radix tab it becomes a correctness condition.'}});
  }
  F.push({shapes:shapes({oSt:{0:'ok', 1:'ok', 2:'ok', 3:'ok', 4:'ok', 5:'ok', 6:'ok', 7:'ok'},
            note:{zh:'2n + k 次觸碰，沒有比較，穩定',
                  en:'2n + k touches, zero comparisons, stable'}, noteC:COL.tealL}),
          panels:panels([{t:'done', cls:'ok'}]), view:VIEW, line:13,
          msg:{zh:'成本是 n（數）+ k（scan）+ n（散）。<b>不是 O(n)，是 O(n + k)</b>：k 是 key 空間的大小，不是元素個數。k 小的時候這一項看不見，k 大的時候它就是全部的成本 - 用 counting sort 排 32-bit 整數要開 43 億個桶。radix sort 就是為了修這件事而存在的。',
               en:'The cost is n to count, k to scan, n to scatter. <b>Not O(n) but O(n + k)</b>, where k is the size of the key space, not the number of elements. The k term is invisible when k is small and is the entire cost when it is large - counting-sorting 32-bit integers means allocating four billion buckets. Radix sort exists to fix exactly that.'}});
  return F.list;
}

/* ====================================================== 2. LSD radix sort */
const RV = [0b1001, 0b0110, 0b1010, 0b0101, 0b0010, 0b1101];
const b4 = v => ('000' + v.toString(2)).slice(-4);

const CODE_R = [
'def radix_sort(values, bits, r, stable=True):',
'    passes = (bits + r - 1) // r',
'    for p in range(passes):',
'        digit = [(v >> (r * p)) & ((1 << r) - 1)',
'                 for v in values]',
'        # one counting sort per digit',
'        values = counting_sort(digit, 1 << r,',
'                               payload=values,',
'                               stable=stable)',
'    return values'
];

function radixFrames(stable){
  const F = new Frames();
  const R = 2, PASSES = 2;
  let cur = RV.slice();
  const Y = [1.05, 2.35, 3.65], YB = 4.95;

  const rowOf = (vals, y, st, title) =>
    cellRow(vals.map(b4), rowX(vals.length), y, 1.05, CH,
            {states:st || {}, title:title, index:false, fs:.30});

  const shapes = (o) => {
    o = o || {};
    let s = rowOf(RV, Y[0], o.st0, {zh:'input', en:'input'});
    if (o.rows) o.rows.forEach((r, i) => {
      s = s.concat(rowOf(r.vals, Y[i + 1], r.st, r.title));
    });
    if (o.note) s.push(note(5.85, o.note, o.noteC));
    s.push(foot({zh:'每一趟只看 2 個 bit，桶只有 4 個',
                 en:'each pass looks at 2 bits and needs only 4 buckets'}));
    return s;
  };
  const panels = (digits, extra) => [
    {lbl:{zh:'這一趟看的 digit', en:'digit this pass reads'},
     chips:(digits || []).map(d => ({t:String(d), cls:'hot'}))},
    {lbl:{zh:'狀態', en:'state'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:{zh:'4-bit 的值，一次排 2 個 bit',
                               en:'4-bit values, two bits at a time'}}),
          panels:panels(), view:VIEW, line:1,
          msg:{zh:'counting sort 怕 k 大，那就<b>把 key 切成小塊</b>。這裡把 4-bit 的值切成兩個 2-bit 的 digit，每個 digit 只需要 4 個桶。代價是要跑兩趟 - 用<b>趟數</b>換<b>桶數</b>，這是 radix sort 的全部交易內容。',
               en:'Counting sort chokes when k is large, so <b>cut the key into pieces</b>. Here a 4-bit value becomes two 2-bit digits, and each digit needs only four buckets. The price is two passes - trading <b>passes</b> for <b>buckets</b> is the whole deal radix sort offers.'}});

  const rowsAcc = [];
  for (let p = 0; p < PASSES; p++){
    const digits = cur.map(v => (v >> (R * p)) & 3);
    F.push({shapes:shapes({rows:rowsAcc.slice(),
              note:{zh:'pass ' + p + '：取 bit ' + (R * p) + '-' + (R * p + 1),
                    en:'pass ' + p + ': read bits ' + (R * p) + '-' + (R * p + 1)}}),
            panels:panels(digits), view:VIEW, line:3,
            msg:p === 0
              ? {zh:'LSD = least significant digit first，先排<b>最低位</b>。直覺上是反的（人排數字都先看最高位），但正是這個順序讓每一趟的結果可以被下一趟安全地保留。',
                 en:'LSD means least significant digit first, so we sort by the <b>lowest</b> bits first. It feels backwards - humans sort by the leading digit - but this order is exactly what lets each pass survive the next one.'}
              : {zh:'第二趟看高位。<b>關鍵在這裡</b>：高位相同的元素，這一趟不會動它們的相對順序，所以它們仍然照上一趟排好的低位順序排列。整個 radix sort 的正確性就掛在這句話上。',
                 en:'The second pass reads the high bits. <b>This is the load-bearing moment</b>: elements with the same high digit are not reordered by this pass, so they keep the low-bit order the previous pass established. The correctness of the whole algorithm hangs on that sentence.'}});

    const counts = [0, 0, 0, 0];
    digits.forEach(d => counts[d]++);
    const starts = [0, 0, 0, 0];
    let t = 0;
    for (let v = 0; v < 4; v++){ starts[v] = t; t += counts[v]; }
    const out = new Array(cur.length);
    if (stable){
      const c2 = starts.slice();
      for (let i = 0; i < cur.length; i++){ out[c2[digits[i]]++] = cur[i]; }
    } else {
      const c2 = starts.map((s, v) => s + counts[v] - 1);
      for (let i = 0; i < cur.length; i++){ out[c2[digits[i]]--] = cur[i]; }
    }
    cur = out;
    const st = {};
    cur.forEach((v, i) => { st[i] = 'ok'; });
    rowsAcc.push({vals:cur.slice(), st:st, title:{zh:'pass ' + p, en:'pass ' + p}});
    F.push({shapes:shapes({rows:rowsAcc.slice(),
              note:stable ? {zh:'穩定的 counting sort，相同 digit 保持原順序',
                             en:'a stable counting sort - equal digits keep their order'}
                          : {zh:'不穩定：每一桶被倒著填回去',
                             en:'unstable: each bucket was filled back to front'},
              noteC:stable ? COL.tealL : COL.red}),
            panels:panels(digits, [{t:'counts ' + counts.join(' '), cls:'ok'},
                                   {t:'starts ' + starts.join(' '), cls:'act'}]),
            view:VIEW, line:6,
            msg:stable
              ? {zh:'這一趟就是第一個 tab 那個 counting sort，只是 key 換成 digit、payload 換成整個值。桶只有 4 個，小到一個 thread 都可以自己留一份私有的 histogram，連 atomic 都省了 - 這就是實務上 digit 都取 4 到 8 bit 的原因。',
                 en:'This pass is literally the counting sort from the first tab, with the digit as the key and the whole value as the payload. Four buckets is small enough that a single thread can keep a private histogram and skip the atomics entirely - which is why real digits are 4 to 8 bits wide.'}
              : {zh:'同一個 counting sort，只是把每一桶<b>從後往前</b>填。單獨看這一趟，桶跟桶之間仍然是排好的 - 錯的地方只在<b>同一個 digit 內部順序被反轉了</b>，而這件事一趟之內看不出來。',
                 en:'The same counting sort, but each bucket is filled <b>back to front</b>. Looked at alone this pass still orders the buckets correctly - the only damage is that <b>elements sharing a digit come out reversed</b>, which a single pass cannot reveal.'}});
  }

  const sortedOK = cur.every((v, i) => i === 0 || cur[i - 1] <= v);
  const stF = {};
  if (!sortedOK){
    for (let i = 0; i + 1 < cur.length; i++) if (cur[i] > cur[i + 1]){ stF[i] = 'bad'; stF[i + 1] = 'bad'; }
  } else cur.forEach((v, i) => { stF[i] = 'ok'; });
  const rowsFinal = rowsAcc.slice(0, 1);
  rowsFinal.push({vals:cur.slice(), st:stF, title:{zh:'result', en:'result'}});
  F.push({shapes:shapes({rows:rowsFinal,
            note:sortedOK ? {zh:'排好了 - 而且從頭到尾沒有比較過任何兩個值',
                             en:'sorted - and no two values were ever compared'}
                          : {zh:'沒排好。不是慢，是錯',
                             en:'not sorted. Not slower - wrong'},
            noteC:sortedOK ? COL.tealL : COL.red}),
          panels:panels(null, [{t:sortedOK ? 'sorted' : 'NOT sorted', cls:sortedOK ? 'ok' : 'bad'}]),
          view:VIEW, line:9,
          msg:sortedOK
            ? {zh:'成本是 <b>ceil(bits / r) 趟</b>，每趟 O(n + 2^r)。32-bit 的 key、8-bit 的 digit 就是 4 趟 256 桶 - 跟開 43 億個桶比起來，這才是可以真的跑的東西。',
               en:'The cost is <b>ceil(bits / r) passes</b> of O(n + 2**r) each. A 32-bit key with an 8-bit digit is four passes over 256 buckets - which, unlike four billion buckets, is something you can actually run.'}
            : {zh:'第二趟把第一趟剛排好的低位順序<b>反轉</b>了，而且<b>沒有任何後面的趟可以補救</b> - 資訊已經丟掉了。所以「stable」在 radix sort 裡不是可有可無的好習慣，它是正確性條件本身。這也是為什麼 GPU 實作寧可多花力氣維持每個 thread 的私有 cursor，而不是讓 atomic 隨便決定誰拿到哪個位置。',
               en:'The second pass <b>reversed</b> the low-digit order the first pass had just established, and <b>no later pass can repair it</b> - the information is gone. So stability is not a nice-to-have in radix sort, it is the correctness condition. It is also why GPU implementations work hard to give each thread a private cursor instead of letting atomics decide who gets which slot.'}});
  return F.list;
}

/* ========================================================= 3. bucket sort */
const CODE_B = [
'def bucket_sort(values, m):',
'    lo, hi = min(values), max(values)',
'    buckets = [[] for _ in range(m)]',
'    for v in values:',
'        i = int((v - lo) / (hi - lo) * m)',
'        buckets[min(i, m - 1)].append(v)',
'    for b in buckets:',
'        insertion_sort(b)        # O(len(b)**2)',
'    return [v for b in buckets for v in b]'
];

function lcg(seed){ let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; }

function bucketFrames(clustered){
  const F = new Frames();
  const M = 12, N = 24;
  const rnd = lcg(clustered ? 99 : 7);
  const vals = [];
  for (let i = 0; i < N; i++){
    if (clustered){
      const v = .5 + (rnd() + rnd() + rnd() - 1.5) * .05;
      vals.push(Math.max(0, Math.min(.999, v)));
    } else vals.push(rnd());
  }
  vals[0] = 0; vals[1] = .999;
  const buckets = [];
  for (let i = 0; i < M; i++) buckets.push(0);
  const x0 = rowX(M), YB = 4.9, BAR = .26;

  const shapes = (o) => {
    o = o || {};
    let s = [];
    s.push(S.e(x0, 1.5, x0 + M * CW, 1.5, {c:COL.grey, w:.03}));
    vals.forEach((v, i) => {
      const placed = o.upto != null && i < o.upto;
      s.push(S.c(x0 + v * (M * CW - .1) + .05, 1.5, .13,
                 i === o.cur ? 'hot' : (placed ? 'ok' : 'ghost'), ''));
    });
    s.push(S.t(x0 - .25, 1.55, {zh:'資料', en:'values'}, {c:COL.tealL, fs:.32, anchor:'end'}));
    for (let b = 0; b < M; b++){
      const n = buckets[b];
      for (let j = 0; j < n; j++)
        s.push(S.r(x0 + b * CW + .07, YB - (j + 1) * BAR, CW - .2, BAR - .05,
                   (o.curB === b && j === n - 1) ? 'hot' : 'ok', '', {rx:.04}));
      s.push(S.r(x0 + b * CW + .07, YB + .06, CW - .2, .32, 'soft', String(n), {fs:.28, rx:.04}));
    }
    s.push(S.t(x0 - .25, YB + .26, {zh:'桶', en:'buckets'}, {c:COL.tealL, fs:.32, anchor:'end'}));
    if (o.note) s.push(note(5.7, o.note, o.noteC));
    s.push(foot({zh:'同樣的程式碼、同樣的 n、同樣的桶數',
                 en:'same code, same n, same number of buckets'}));
    return s;
  };
  const maxOcc = () => Math.max.apply(null, buckets);
  const panels = (extra) => [
    {lbl:{zh:'最滿的桶', en:'fullest bucket'}, chips:[{t:String(maxOcc()), cls:'hot'}]},
    {lbl:{zh:'空桶', en:'empty buckets'},
     chips:[{t:String(buckets.filter(c => c === 0).length), cls:'dim'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({upto:0, note:clustered
            ? {zh:'幾乎所有值都擠在 0.5 附近', en:'almost every value sits near 0.5'}
            : {zh:'值大致均勻分布在 [0, 1]', en:'the values are roughly uniform on [0, 1]'}}),
          panels:panels(), view:VIEW, line:0,
          msg:clustered
            ? {zh:'一模一樣的 bucket sort，只換掉輸入的<b>分布</b>。值全部擠在中間，但最小值和最大值沒變，所以桶寬也沒變。',
               en:'The very same bucket sort, with only the <b>distribution</b> of the input changed. The values cluster in the middle, but the min and max are unchanged, so the bucket width is unchanged too.'}
            : {zh:'bucket sort 不再把 key 當位址，而是當<b>估計值</b>：把值線性縮放進 m 個桶，桶內用任何排序法收尾，最後接起來。',
               en:'Bucket sort stops treating the key as an address and treats it as an <b>estimate</b>: scale each value into one of m buckets, sort inside each bucket however you like, concatenate.'}});

  for (let i = 0; i < N; i++){
    const b = Math.min(M - 1, Math.floor(vals[i] * M));
    buckets[b]++;
    if (i < 3 || i % 3 === 2 || i === N - 1)
      F.push({shapes:shapes({upto:i + 1, cur:i, curB:b,
                note:{zh:'第 ' + (i + 1) + ' 個值 → 桶 ' + b,
                      en:'value ' + (i + 1) + ' -> bucket ' + b}}),
              panels:panels([{t:'bucket ' + b, cls:'act'}]), view:VIEW, line:5,
              msg:i === 0
                ? {zh:'落到哪個桶只靠一次乘法和一次取整，沒有比較。<b>問題是桶的胖瘦不由我們決定</b> - 它由資料分布決定。',
                   en:'Which bucket a value lands in costs one multiply and one truncation, no comparison. <b>The catch is that we do not decide how fat a bucket gets</b> - the data distribution does.'}
                : {zh:'注意右邊那兩個數字：最滿的桶還有幾個桶是空的。bucket sort 的成本完全寫在這兩個數字上，跟程式碼一點關係也沒有。',
                   en:'Watch the two counters on the right: the fullest bucket and how many are empty. Bucket sort\'s cost is written entirely in those two numbers, and not at all in the code.'}});
  }
  const moves = buckets.reduce((a, c) => a + c * (c - 1) / 2, 0);
  F.push({shapes:shapes({upto:N,
            note:clustered ? {zh:'最滿的桶有 ' + maxOcc() + ' 個元素',
                              en:'the fullest bucket holds ' + maxOcc() + ' elements'}
                           : {zh:'每個桶都只有幾個元素',
                              en:'every bucket holds a handful'},
            noteC:clustered ? COL.red : COL.tealL}),
          panels:panels([{t:'inner-sort ~ ' + moves + ' moves', cls:clustered ? 'bad' : 'ok'}]),
          view:VIEW, line:7,
          msg:clustered
            ? {zh:'桶內 insertion sort 的成本是 O(len²)，最壞情況大約 ' + moves + ' 次搬移 - 幾乎所有元素都落在同一個桶裡，這個「線性」排序就退化成 n²。<b>bucket sort 沒有在排資料，它是在賭分布</b>；賭輸了不會報錯，只會安靜地變慢。',
               en:'The inner insertion sort costs O(len squared), so the worst case here is about ' + moves + ' moves - nearly everything landed in one bucket and the "linear" sort has degenerated to n squared. <b>Bucket sort does not sort the data, it bets on the distribution</b>, and losing the bet raises no exception - it just quietly gets slow.'}
            : {zh:'均勻輸入下每個桶平均只有 n/m 個元素，桶內排序總共大約 ' + moves + ' 次搬移，整體是<b>期望</b>線性 - 注意是期望，不是保證。同樣的假設在 GPU 上也成立：資料一歪，吃到胖桶的那些 thread block 就變成長尾。',
               en:'With uniform input each bucket holds about n/m elements and the inner sorts cost around ' + moves + ' moves in total, so the whole thing is linear <b>in expectation</b> - expectation, not guarantee. The same assumption governs a GPU: skew the data and the thread blocks that drew the fat buckets become the long tail.'}});
  return F.list;
}

/* =============================================================== 4. scan */
const CODE_S = [
'# the scan is the one phase that is sequential',
'starts[v] = starts[v - 1] + counts[v - 1]',
'',
'# Hillis-Steele: log n rounds, n log n adds',
'off = 1',
'while off < n:',
'    for i in range(off, n):        # in parallel',
'        buf[i] += buf[i - off]',
'    off *= 2',
'',
'# Blelloch: same depth class, 2(n-1) adds',
'up-sweep:    buf[bi] += buf[ai]',
'down-sweep:  buf[ai], buf[bi] = buf[bi], buf[bi] + buf[ai]'
];

const SH = [3, 1, 7, 0, 4, 1, 6, 3];

function scanFrames(blelloch){
  const F = new Frames();
  const n = SH.length;
  const rows = [];
  let work = 0, depth = 0;
  const x0 = rowX(n);

  const shapes = (o) => {
    o = o || {};
    let s = [];
    rows.forEach((r, i) => {
      s = s.concat(cellRow(r.vals, x0, .7 + i * .74, CW, .62,
        {states:r.st || {}, title:r.title, index:false, fs:.3}));
    });
    if (o.note) s.push(note(5.8, o.note, o.noteC));
    s.push(foot(blelloch
      ? {zh:'Blelloch：上掃建樹，下掃把部分和推回去', en:'Blelloch: a reduction tree up, partial sums back down'}
      : {zh:'Hillis-Steele：每一輪加上左邊 off 格', en:'Hillis-Steele: every element adds the one off to its left'}));
    return s;
  };
  const panels = (extra) => [
    {lbl:{zh:'加法次數 work', en:'additions (work)'}, chips:[{t:String(work), cls:'hot'}]},
    {lbl:{zh:'輪數 depth', en:'rounds (depth)'}, chips:[{t:String(depth), cls:'act'}]},
    {lbl:{zh:'序列版', en:'sequential'},
     chips:[{t:(n - 1) + ' adds, ' + (n - 1) + ' deep', cls:'dim'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  rows.push({vals:SH.slice(), title:{zh:'histogram', en:'histogram'}});
  F.push({shapes:shapes({note:{zh:'count 和 scatter 都能一個 thread 一個元素，scan 不行',
                               en:'count and scatter are one thread per element; the scan is not'}}),
          panels:panels(), view:VIEW, line:1,
          msg:{zh:'counting sort 的三個階段裡，數和散都可以直接一個 thread 一個元素。中間這個 scan 不行：<b>starts[v] 要讀上一格剛寫的結果</b>，這是序列的定義。整個平行排序的難點就縮在這一行。',
               en:'Of counting sort\'s three phases, count and scatter are one-thread-per-element on sight. The scan is not: <b>starts[v] reads what the previous iteration just wrote</b>, which is the definition of sequential. The entire difficulty of parallel sorting collapses into this one line.'}});

  if (!blelloch){
    let buf = SH.slice(), off = 1;
    while (off < n){
      const nxt = buf.slice(), st = {};
      for (let i = off; i < n; i++){ nxt[i] = buf[i] + buf[i - off]; work++; st[i] = 'hot'; }
      buf = nxt; depth++;
      rows.push({vals:buf.slice(), st:st, title:{zh:'off = ' + off, en:'off = ' + off}});
      F.push({shapes:shapes({note:{zh:'每個元素同時加上左邊第 ' + off + ' 格',
                                   en:'every element adds the one ' + off + ' to its left, all at once'}}),
              panels:panels([{t:'off = ' + off, cls:'hot'},
                             {t:(n - off) + ' adds this round', cls:'act'}]),
              view:VIEW, line:7,
              msg:off === 1
                ? {zh:'Hillis-Steele 的想法很暴力：每一輪把 stride 加倍，所有元素<b>同時</b>加上左邊 off 格的值。一輪之後每格含 2 個原始值，兩輪之後 4 個 - <b>log n 輪就吃完整個前綴和</b>。',
                   en:'Hillis-Steele is brutally simple: double the stride each round and have every element add the one off to its left, <b>all at the same time</b>. After one round each cell holds two original values, after two rounds four - <b>log n rounds and the prefix sum is done</b>.'}
                : {zh:'看 work 的累積速度：這一輪就做了 ' + (n - off) + ' 次加法。<b>總共 n log n 次加法，比它取代掉的序列迴圈還多</b>。它之所以划算，唯一的理由是那些 lane 本來就閒著。',
                   en:'Watch the work counter: this round alone did ' + (n - off) + ' additions. <b>The total is n log n additions - more than the sequential loop it replaces</b>. The only reason it is a good trade is that those lanes were idle anyway.'}});
      off <<= 1;
    }
    const excl = []; let t = 0;
    for (let i = 0; i < n; i++){ excl.push(t); t += SH[i]; }
    const st = {};
    for (let i = 0; i < n; i++) st[i] = 'ok';
    rows.push({vals:excl, st:st, title:{zh:'exclusive', en:'exclusive'}});
    F.push({shapes:shapes({note:{zh:'每格減掉自己就得到 exclusive scan',
                                 en:'subtract each element from its own total to get the exclusive scan'}}),
            panels:panels([{t:'work ' + work + ' / depth ' + depth, cls:'ok'}]),
            view:VIEW, line:8,
            msg:{zh:'depth 從 ' + (n - 1) + ' 掉到 ' + depth + '，work 從 ' + (n - 1) + ' 漲到 ' + work + '。這個取捨就是平行演算法的日常：<b>沒有變快，只是把一條長長的序列鏈換成寬而淺的樹</b>。另一個變體用一樣的深度做到線性 work。',
                 en:'Depth falls from ' + (n - 1) + ' to ' + depth + ' while work rises from ' + (n - 1) + ' to ' + work + '. That trade is everyday life in parallel algorithms: <b>nothing got faster, a long chain got replaced by a wide shallow tree</b>. The other variant reaches the same depth class with linear work.'}});
  } else {
    const buf = SH.slice();
    let off = 1, d = n >> 1;
    while (d > 0){
      const st = {};
      for (let i = 0; i < d; i++){
        const ai = off * (2 * i + 1) - 1, bi = off * (2 * i + 2) - 1;
        buf[bi] += buf[ai]; work++; st[bi] = 'hot'; st[ai] = 'soft';
      }
      depth++;
      rows.push({vals:buf.slice(), st:st, title:{zh:'up ' + depth, en:'up ' + depth}});
      F.push({shapes:shapes({note:{zh:'上掃第 ' + depth + ' 層：' + d + ' 次加法',
                                   en:'up-sweep level ' + depth + ': ' + d + ' additions'}}),
              panels:panels([{t:d + ' adds this round', cls:'hot'}]), view:VIEW, line:11,
              msg:depth === 1
                ? {zh:'上掃就是在原地蓋一棵<b>reduction tree</b>：每一層只有右邊那格吸收左邊那格，所以每層的加法次數<b>減半</b>。n/2 + n/4 + ... = n-1 次，這就是線性 work 的來源。',
                   en:'The up-sweep builds a <b>reduction tree</b> in place: at each level only the right cell absorbs the left one, so the number of additions <b>halves</b> every level. n/2 + n/4 + ... = n-1 additions, and that is where the linear work comes from.'}
                : {zh:'每一層都只碰上一層留下的那些「右端」格子。走到最後，最右邊那格就是全部的總和 - 樹根。',
                   en:'Each level touches only the right-hand cells the previous level wrote. At the top, the last cell holds the total for the whole array - the root of the tree.'}});
      off <<= 1; d >>= 1;
    }
    const total = buf[n - 1];
    buf[n - 1] = 0;
    d = 1;
    while (d < n){
      off >>= 1;
      const st = {};
      for (let i = 0; i < d; i++){
        const ai = off * (2 * i + 1) - 1, bi = off * (2 * i + 2) - 1;
        const tmp = buf[ai]; buf[ai] = buf[bi]; buf[bi] = buf[bi] + tmp;
        work++; st[ai] = 'act'; st[bi] = 'hot';
      }
      depth++; d <<= 1;
      rows.push({vals:buf.slice(), st:st, title:{zh:'down', en:'down'}});
      F.push({shapes:shapes({note:{zh:'下掃：交換再相加，把部分和推給左右子樹',
                                   en:'down-sweep: swap then add, pushing partial sums into both subtrees'}}),
              panels:panels([{t:'total = ' + total, cls:'ok'}]), view:VIEW, line:12,
              msg:{zh:'下掃前先把樹根清成 0 - 這一步就是「exclusive」的來源。接著每個節點把自己的值交給左子，左子原本的值加上去給右子，走完之後每格<b>剛好等於「左邊所有東西的總和」</b>。',
                   en:'Before the down-sweep the root is cleared to 0 - that single move is where the "exclusive" in exclusive scan comes from. Then each node hands its value to its left child and gives the sum to the right child, and when it finishes every cell holds <b>exactly the total of everything to its left</b>.'}});
    }
    const st = {};
    for (let i = 0; i < n; i++) st[i] = 'ok';
    rows[rows.length - 1].st = st;
    F.push({shapes:shapes({note:{zh:'work ' + work + '（= 2(n-1)），depth ' + depth,
                                 en:'work ' + work + ' (= 2(n-1)), depth ' + depth}, noteC:COL.tealL}),
            panels:panels([{t:'work ' + work + ' / depth ' + depth, cls:'ok'}]),
            view:VIEW, line:12,
            msg:{zh:'Blelloch 用 <b>2(n-1) 次加法</b>做到跟 Hillis-Steele 同一個 depth 等級。代價是它要求長度是 2 的冪 - 這就是為什麼 sglang 的 MoE kernel 在 HIP path 上會把 expert 直方圖<b>補零到 2 的冪</b>再做 scan。演算法的限制會一路長進 kernel 的參數裡。',
                 en:'Blelloch reaches the same depth class with <b>2(n-1) additions</b>. The price is that it wants a power-of-two length - which is exactly why sglang\'s MoE kernel <b>zero-pads the expert histogram up to a power of two</b> before scanning on the HIP path. An algorithmic constraint grows all the way out into a kernel parameter.'}});
  }
  return F.list;
}

/* ================================================= 5. moe_align_block_size */
const FLAT = [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3];
const NE = 5, BLOCK = 4, NUMEL = 12;

const CODE_M = [
'counts = histogram(expert of every routed token)',
'padded = [ceil(c / block) * block for c in counts]',
'prefix = exclusive_scan(padded)',
'sorted_token_ids = [numel] * prefix[-1]   # sentinel',
'cursor = list(prefix)',
'for i, e in enumerate(flat):',
'    sorted_token_ids[cursor[e]] = i',
'    cursor[e] += 1',
'for b in range(num_blocks):',
'    expert_ids[b] = bisect(prefix, b * block) - 1'
];

function moeFrames(){
  const F = new Frames();
  const counts = new Array(NE).fill(0);
  let padded = null, prefix = null, out = null, cursor = null;
  const expertIds = [];
  const CWs = .56;
  const xF = (VIEW[0] - FLAT.length * CWs) / 2;
  const xO = (VIEW[0] - 16 * CWs) / 2;
  const xC = rowX(NE);

  const shapes = (o) => {
    o = o || {};
    let s = [];
    s = s.concat(cellRow(FLAT, xF, .72, CWs, .56,
      {states:o.fSt || {}, title:{zh:'expert', en:'expert'}, fs:.28, index:false}));
    s = s.concat(cellRow(counts, xC, 1.95, CW, .56,
      {states:o.cSt || {}, title:{zh:'count', en:'count'}, index:false, fs:.3}));
    if (padded)
      s = s.concat(cellRow(padded, xC, 2.72, CW, .56,
        {states:o.pSt || {}, title:{zh:'padded', en:'padded'}, index:false, fs:.3}));
    if (prefix)
      s = s.concat(cellRow(prefix, xC - CW * .5, 3.49, CW, .56,
        {states:o.prSt || {}, title:{zh:'prefix', en:'prefix'}, index:false, fs:.3}));
    if (out){
      s = s.concat(cellRow(out.map(v => v === NUMEL ? '-' : v), xO, 4.5, CWs, .56,
        {states:o.oSt || {}, title:{zh:'sorted', en:'sorted'}, index:false, fs:.28}));
      for (let b = 0; b < 4; b++)
        s.push(S.t(xO + (b * BLOCK + BLOCK / 2) * CWs, 5.45,
          expertIds.length > b ? 'expert ' + expertIds[b] : 'block ' + b,
          {c:expertIds.length > b ? COL.orangeL : COL.grey, fs:.26}));
    }
    if (o.note) s.push(note(6.05, o.note, o.noteC));
    return s;
  };
  const panels = (extra) => [
    {lbl:{zh:'counts / expert', en:'counts per expert'},
     chips:counts.map((c, i) => ({t:i + ':' + c, cls:c ? 'ok' : 'dim'}))},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:{zh:'4 個 token、每個 top-3、5 個 expert、block = 4',
                               en:'4 tokens, top-3 each, 5 experts, block = 4'}}),
          panels:panels(), view:VIEW, line:0,
          msg:{zh:'這不是教科書題目，是每個 MoE 模型每一步 decode 都要跑的東西。router 給你「每個 token 該去哪些 expert」，而 GEMM 要求<b>同一個 expert 的 token 連續、而且以固定 block 對齊</b>。expert id 是小整數 - 所以這就是一個 counting sort。',
               en:'This is not a textbook exercise; it runs on every decode step of every MoE model. The router hands you which experts each token belongs to, and the GEMM demands that <b>each expert\'s tokens be contiguous and aligned to a fixed block</b>. Expert ids are small integers - so this is a counting sort.'}});

  for (let i = 0; i < FLAT.length; i++){
    counts[FLAT[i]]++;
    if (i === 0 || i % 4 === 3){
      const fSt = {}; fSt[i] = 'hot';
      const cSt = {}; cSt[FLAT[i]] = 'hot';
      F.push({shapes:shapes({fSt:fSt, cSt:cSt,
                note:{zh:'數 expert id，跟第一個 tab 一模一樣',
                      en:'histogram the expert ids - the same first phase as tab one'}}),
              panels:panels([{t:'expert ' + FLAT[i], cls:'hot'}]), view:VIEW, line:0,
              msg:i === 0
                ? {zh:'第一階段就是 histogram。真的 kernel 在這裡有兩種寫法：大 batch 用 <b>atomicAdd</b> 撞一張共享的表（快，但同 expert 內的順序不保證），小 batch 每個 thread 留一份私有 histogram（慢一點，但結果<b>可重現</b>）。切換條件寫死在 .cu 裡：<b>numel &lt; 1024 且 num_experts &le; 64</b>。',
                   en:'Phase one is a histogram. The real kernel has two ways to do it: large batches hammer one shared table with <b>atomicAdd</b> (fast, but the order within an expert is not guaranteed), small batches give every thread a private histogram (slower, but <b>reproducible</b>). The switch is hard-coded in the .cu: <b>numel &lt; 1024 and num_experts &le; 64</b>.'}
                : {zh:'注意這裡我們數的是「展平後的 token 位置」，每個 token 因為 top-k 會出現 k 次。counting sort 對這件事完全無感 - 它只看 key。',
                   en:'Note that we are counting flattened token slots: with top-k routing each token appears k times. Counting sort is entirely indifferent to that - it only ever looks at the key.'}});
    }
  }

  padded = counts.map(c => Math.ceil(c / BLOCK) * BLOCK);
  F.push({shapes:shapes({pSt:{1:'hot', 2:'hot', 3:'hot', 4:'hot'},
            note:{zh:'每一桶先無條件進位到 block_size，然後才 scan',
                  en:'round every bucket up to block_size FIRST, then scan'}, noteC:COL.orangeL}),
          panels:panels([{t:'block = ' + BLOCK, cls:'act'}]), view:VIEW, line:1,
          msg:{zh:'這一行是它跟教科書 counting sort 唯一的結構差異：<b>直方圖在 scan 之前先被進位</b>。每個 expert 有 3 個 token，進位成 4，於是每個 expert 的區段都從 block 邊界開始，GEMM 可以整塊讀、不用處理殘餘。代價是桶裡多出來的洞。',
               en:'This line is the only structural difference from the textbook version: <b>the histogram is rounded up before the prefix sum</b>. Each expert has 3 tokens, padded to 4, so every expert\'s region starts on a block boundary and the GEMM can read whole blocks with no ragged tail. The price is holes inside the buckets.'}});

  prefix = [0];
  for (let e = 0; e < NE; e++) prefix.push(prefix[e] + padded[e]);
  out = new Array(prefix[NE]).fill(NUMEL);
  cursor = prefix.slice(0, NE);
  F.push({shapes:shapes({prSt:{0:'ok', 1:'ok', 2:'ok', 3:'ok', 4:'ok', 5:'ok'},
            note:{zh:'洞先填上 sentinel = ' + NUMEL + '（一個不存在的 token）',
                  en:'the holes are pre-filled with the sentinel ' + NUMEL + ', a token that does not exist'}}),
          panels:panels([{t:'post_pad = ' + prefix[NE], cls:'ok'}]), view:VIEW, line:3,
          msg:{zh:'sentinel 是「最後一個 token 再加一」的索引，GEMM 讀到它就當作要 mask 掉的列。真的 kernel 更狠：<b>用一整個獨立的 thread block、以 int4 向量寫入</b>把 sentinel 先鋪滿，跟 histogram 同時進行 - 反正兩件事互不相干。',
               en:'The sentinel is the index one past the last real token; the GEMM is told to mask those rows out. The real kernel goes further: <b>a whole separate thread block pre-fills the sentinels with int4 vector stores</b> while the histogram is still running, because the two jobs never touch each other.'}});

  for (let i = 0; i < FLAT.length; i++){
    const e = FLAT[i], pos = cursor[e];
    out[pos] = i; cursor[e]++;
    if (i === 0 || i % 3 === 2){
      const fSt = {}; fSt[i] = 'hot';
      const oSt = {}; oSt[pos] = 'ok';
      const prSt = {}; prSt[e] = 'act';
      F.push({shapes:shapes({fSt:fSt, oSt:oSt, prSt:prSt,
                note:{zh:'token ' + i + ' → slot ' + pos + '（expert ' + e + '）',
                      en:'token ' + i + ' -> slot ' + pos + ' (expert ' + e + ')'}}),
              panels:panels([{t:'cursor[' + e + '] = ' + cursor[e], cls:'act'}]),
              view:VIEW, line:6,
              msg:i === 0
                ? {zh:'scatter 階段。在 .cu 裡這一行就是 <b>rank = atomicAdd(&amp;cumsum[e], 1)</b> - 每個 thread 用一次 atomic 搶一個位置。',
                   en:'The scatter. In the .cu this line is <b>rank = atomicAdd(&amp;cumsum[e], 1)</b> - every thread claims a slot with a single atomic.'}
                : {zh:'注意 prefix 陣列<b>同時扮演三個角色</b>：桶的起點、scatter 過程中會移動的 cursor、以及等一下二分搜尋的 key。一次 scan，三份工。',
                   en:'Notice the prefix array is doing <b>three jobs at once</b>: the bucket starts, the moving cursors during the scatter, and in a moment the key for a binary search. One scan, three uses.'}});
    }
  }

  for (let b = 0; b < prefix[NE] / BLOCK; b++){
    const start = b * BLOCK;
    let lo = 0, hi = NE + 1;
    while (lo < hi){ const mid = (lo + hi) >> 1; if (prefix[mid] <= start) lo = mid + 1; else hi = mid; }
    expertIds.push(lo - 1);
    const prSt = {}; prSt[lo - 1] = 'hot';
    const oSt = {};
    for (let j = start; j < start + BLOCK; j++) oSt[j] = 'act';
    F.push({shapes:shapes({prSt:prSt, oSt:oSt,
              note:{zh:'block ' + b + ' 從 slot ' + start + ' 開始 → expert ' + (lo - 1),
                    en:'block ' + b + ' starts at slot ' + start + ' -> expert ' + (lo - 1)}}),
            panels:panels([{t:'expert_ids ' + expertIds.join(' '), cls:'hot'}]),
            view:VIEW, line:9,
            msg:b === 0
              ? {zh:'GEMM 還需要知道「第 b 個 block 屬於哪個 expert」才能挑對應的權重。這件事不用另外算 - <b>拿 prefix 陣列做二分搜尋</b>就好。scan 的輸出被當成索引重複使用，這是 kernel 裡很典型的省法。',
                 en:'The GEMM also needs to know which expert owns block b, to pick the right weights. That answer does not need computing - <b>binary-search the prefix array</b>. The scan output gets reused as an index, which is a very kernel-shaped way to save work.'}
              : {zh:'每個 block 只要一次 log(num_experts) 的搜尋，num_experts = 256 也才 8 步。padding 造成的空 slot 在這裡也一併被歸給正確的 expert。',
                 en:'One log(num_experts) search per block - eight steps even at 256 experts. The padding slots get attributed to the right expert here too, for free.'}});
  }
  const oSt = {};
  out.forEach((v, i) => { oSt[i] = v === NUMEL ? 'soft' : 'ok'; });
  F.push({shapes:shapes({oSt:oSt,
            note:{zh:'sorted_token_ids 跟 sglang docstring 裡的答案完全一致',
                  en:'sorted_token_ids matches the worked example in sglang\'s docstring exactly'},
            noteC:COL.tealL}),
          panels:panels([{t:'padding ' + out.filter(v => v === NUMEL).length + ' / ' + out.length,
                          cls:'hot'}]), view:VIEW, line:9,
          msg:{zh:'整個 kernel 就是這樣：一樣的三個階段，多了「先進位再 scan」和「二分搜尋回推 expert」。真實尺寸下（4096 token、top-8、256 expert、block 64）padding 大約佔<b>兩成</b> - 那就是固定 block size 的價格，而 counting sort 的工作正是把資料排成讓這個價格付得下去的樣子。',
               en:'That is the whole kernel: the same three phases, plus "pad before the scan" and "binary-search the expert back out". At real sizes - 4096 tokens, top-8 of 256 experts, block 64 - the padding is around <b>twenty percent</b>. That is the price of a fixed block size, and arranging the data so the price is payable is exactly what the counting sort is for.'}});
  return F.list;
}

/* ============================================================= 6. LC 164 */
const GN = [15, 3, 8, 1, 24, 22];

const CODE_G = [
'def maximum_gap(nums):',
'    lo, hi, n = min(nums), max(nums), len(nums)',
'    w  = ceil((hi - lo) / (n - 1))    # n-1 buckets',
'    nb = (hi - lo) // w + 1',
'    bmin, bmax = [None] * nb, [None] * nb',
'    for v in nums:',
'        b = (v - lo) // w',
'        bmin[b] = min(bmin[b], v)',
'        bmax[b] = max(bmax[b], v)',
'    best, prev = 0, bmax[0]',
'    for b in range(1, nb):',
'        if bmin[b] is None:  continue    # empty',
'        best = max(best, bmin[b] - prev)',
'        prev = bmax[b]',
'    return best'
];

function gapFrames(){
  const F = new Frames();
  const n = GN.length, lo = Math.min.apply(null, GN), hi = Math.max.apply(null, GN);
  const w = Math.ceil((hi - lo) / (n - 1));
  const nb = Math.floor((hi - lo) / w) + 1;
  const bmin = new Array(nb).fill(null), bmax = new Array(nb).fill(null);
  const BW = Math.min(1.4, (VIEW[0] - 1.8) / nb);
  const x0 = (VIEW[0] - nb * BW) / 2;

  const shapes = (o) => {
    o = o || {};
    let s = [];
    s = s.concat(cellRow(GN, rowX(n), .75, CW, CH,
      {states:o.nSt || {}, title:{zh:'nums', en:'nums'}, index:false}));
    for (let b = 0; b < nb; b++){
      const empty = bmin[b] === null;
      s.push(S.r(x0 + b * BW + .06, 2.35, BW - .14, 1.5,
                 o.curB === b ? 'hot' : (empty ? 'ghost' : 'idle'), '', {rx:.08}));
      s.push(S.t(x0 + b * BW + BW / 2, 2.82,
                 empty ? {zh:'空', en:'empty'} : ('min ' + bmin[b]),
                 {c:empty ? COL.grey : COL.pale, fs:.28}));
      if (!empty)
        s.push(S.t(x0 + b * BW + BW / 2, 3.36, 'max ' + bmax[b], {c:COL.pale, fs:.28}));
      s.push(S.t(x0 + b * BW + BW / 2, 4.22,
                 '[' + (lo + b * w) + ', ' + (lo + (b + 1) * w - 1) + ']',
                 {c:COL.grey, fs:.24}));
    }
    if (o.gap) s.push(S.t(VIEW[0] / 2, 5.0, o.gap, {c:COL.orangeL, fs:.34}));
    if (o.note) s.push(note(5.7, o.note, o.noteC));
    s.push(foot({zh:'桶內的順序從頭到尾沒有算過',
                 en:'the order inside a bucket is never computed'}));
    return s;
  };
  const panels = (best) => [
    {lbl:{zh:'桶寬 w', en:'bucket width w'}, chips:[{t:String(w), cls:'act'}]},
    {lbl:{zh:'桶數', en:'buckets'}, chips:[{t:String(nb), cls:'act'}]},
    {lbl:{zh:'目前最大 gap', en:'best gap so far'},
     chips:[{t:String(best || 0), cls:best ? 'hot' : 'dim'}]}
  ];

  F.push({shapes:shapes({note:{zh:'w = ceil((max - min) / (n - 1)) = ' + w,
                               en:'w = ceil((max - min) / (n - 1)) = ' + w}}),
          panels:panels(0), view:VIEW, line:2,
          msg:{zh:'題目要「排序後相鄰兩數的最大差」，但限定 O(n) - 也就是<b>不准真的排序</b>。答案的形狀是被這個限制逼出來的：n 個數丟進 n-1 個桶，<b>至少有一個桶是空的</b>，所以最大 gap 至少有一個桶那麼寬，因此它<b>不可能出現在同一個桶裡面</b>。',
               en:'The problem asks for the largest gap between neighbours in the sorted array, in O(n) - which forbids actually sorting. That constraint forces the shape of the answer: put n numbers in n-1 buckets and <b>at least one bucket must be empty</b>, so the largest gap is at least one bucket wide and therefore <b>cannot live inside a bucket</b>.'}});

  for (let i = 0; i < n; i++){
    const b = Math.floor((GN[i] - lo) / w);
    bmin[b] = bmin[b] === null ? GN[i] : Math.min(bmin[b], GN[i]);
    bmax[b] = bmax[b] === null ? GN[i] : Math.max(bmax[b], GN[i]);
    const nSt = {}; nSt[i] = 'hot';
    F.push({shapes:shapes({nSt:nSt, curB:b,
              note:{zh:GN[i] + ' → 桶 ' + b + '，只更新這個桶的 min / max',
                    en:GN[i] + ' -> bucket ' + b + ', updating only its min and max'}}),
            panels:panels(0), view:VIEW, line:7,
            msg:i === 0
              ? {zh:'每個值只做一次除法就知道去哪個桶 - 這是 bucket sort 的分桶，不是排序。而且我們<b>只留 min 和 max</b>，桶裡其他值直接扔掉。',
                 en:'One division tells us which bucket a value belongs to - this is bucket sort\'s bucketing step, not a sort. And we keep <b>only the min and the max</b>; everything else in the bucket is thrown away.'}
              : {zh:'扔掉桶內順序是刻意的：<b>唯一可能構成答案的一對，是某個桶的 max 和下一個非空桶的 min</b>。同一個桶裡的兩個值差距一定小於 w，不可能是最大 gap，所以那部分的排序是可以省掉的工作。',
                 en:'Discarding the in-bucket order is the point: <b>the only pair that can form the answer is one bucket\'s max and the next non-empty bucket\'s min</b>. Two values inside a bucket differ by less than w and can never be the largest gap, so sorting them is work you are allowed to skip.'}});
  }

  let best = 0, prev = bmax[0];
  for (let b = 1; b < nb; b++){
    if (bmin[b] === null){
      F.push({shapes:shapes({curB:b,
                note:{zh:'空桶 - 跳過，但它的存在正是整個做法的保證',
                      en:'an empty bucket - skipped, and its existence is what makes the whole trick valid'},
                noteC:COL.orangeL}),
              panels:panels(best), view:VIEW, line:11,
              msg:{zh:'空桶不是意外，是<b>鴿籠原理保證一定會出現</b>的東西：n 個值、n-1 個桶。也正因為有空桶，某個跨桶的 gap 一定大於等於 w，而桶內任兩個值差距一定小於 w - 答案只可能落在桶跟桶的邊界上。',
                   en:'The empty bucket is not an accident, it is <b>guaranteed by pigeonhole</b>: n values into n-1 buckets. And because one exists, some cross-bucket gap is at least w, while any two values inside a bucket differ by less than w - so the answer can only sit on a boundary between buckets.'}});
      continue;
    }
    const cand = bmin[b] - prev;
    best = Math.max(best, cand);
    F.push({shapes:shapes({curB:b,
              gap:{zh:'gap = ' + bmin[b] + ' - ' + prev + ' = ' + cand,
                   en:'gap = ' + bmin[b] + ' - ' + prev + ' = ' + cand},
              note:{zh:'只比較「上一個非空桶的 max」和「這個桶的 min」',
                    en:'compare only the previous non-empty bucket\'s max with this bucket\'s min'}}),
            panels:panels(best), view:VIEW, line:12,
            msg:{zh:'整趟掃描只做每桶一次減法。總成本 O(n + nb) = O(n)，空間也是 O(n) - 跟 counting sort 完全一樣的形狀，因為它<b>就是</b> counting sort 的分桶階段，只是每桶留下的摘要從「數量」換成了「(min, max)」。',
                 en:'This sweep does one subtraction per bucket. The total is O(n + nb) = O(n), with O(n) space - the same shape as counting sort, because it <b>is</b> counting sort\'s bucketing phase with (min, max) kept as the per-bucket summary instead of a count.'}});
    prev = bmax[b];
  }
  F.push({shapes:shapes({gap:{zh:'答案 = ' + best, en:'answer = ' + best},
            note:{zh:'O(n) 時間、O(n) 空間，一次真正的排序都沒做',
                  en:'O(n) time, O(n) space, and not one real sort'}, noteC:COL.tealL}),
          panels:panels(best), view:VIEW, line:14,
          msg:{zh:'「排一排再看」是 O(n log n) 的答案，而且完全正確。線性解不是更聰明的排序，而是<b>看出哪一部分的排序可以不做</b>。這個念頭在系統程式裡出現的頻率遠高於面試 - 很多 kernel 的優化其實都是同一句話：先問清楚答案到底需要多少順序。',
               en:'"Sort it and look" is the O(n log n) answer, and it is perfectly correct. The linear answer is not a cleverer sort - it comes from <b>noticing which part of the sort you can skip</b>. That move shows up far more often in systems work than in interviews: a lot of kernel optimisation is the same sentence, asking how much order the answer actually needs.'}});
  return F.list;
}

/* ================================================================= meta */
const DAY_META = {
  title:{zh:'Day 22 · 不用比較的排序：counting、radix、bucket',
         en:'Day 22 · Sorting without comparing - counting, radix, bucket'},
  sub:{zh:'n log n 是「只能問 a &lt; b?」的下限。把 key 當成位址就跳出了那個模型 - 代價是 k、是穩定性、是分布假設，而中間那個 prefix sum 就是整個平行排序的難點所在。',
       en:'n log n bounds sorts that only ask "a &lt; b?". Using the key as an address leaves that model - and the price is k, stability, and a distribution assumption, with the prefix sum in the middle carrying all the parallel difficulty.'},
  tabs:[
    {id:'count', label:{zh:'Counting sort', en:'Counting sort'},
     stage:{zh:'數 → scan → 散，直方圖變成位址表',
            en:'count, scan, scatter - the histogram becomes an address table'},
     view:VIEW,
     idea:{zh:'三個階段：<b>count</b> 做直方圖、<b>scan</b> 把它變成每一桶的起始位址、<b>scatter</b> 把元素寫到 cursor[key]。從頭到尾沒有任何比較，所以 n log n 的下限根本不適用。成本是 <b>O(n + k)</b> - k 是 key 空間的大小，這一項才是後面所有故事的來源。',
           en:'Three phases: <b>count</b> builds a histogram, <b>scan</b> turns it into the starting address of each bucket, <b>scatter</b> writes each element to cursor[key]. No comparison happens anywhere, so the n log n bound simply does not apply. The cost is <b>O(n + k)</b>, and that k term is where every later problem comes from.'},
     legend:['hot', 'act', 'ok', 'soft'],
     code:CODE_C,
     build:() => countFrames()},
    {id:'radix', label:{zh:'LSD radix / 穩定性', en:'LSD radix / stability'},
     stage:{zh:'一次排幾個 bit，靠穩定性把前一趟的結果留住',
            en:'a few bits at a time, with stability preserving the previous pass'},
     view:VIEW,
     variants:[{zh:'穩定（正確）', en:'stable (correct)'},
               {zh:'不穩定（壞掉）', en:'unstable (broken)'}],
     idea:{zh:'radix sort 就是<b>一個 digit 一次的 counting sort</b>，從最低位開始。它正確的唯一理由是<b>每一趟都穩定</b>：高位相同的元素不會被重排，所以低位建立的順序活了下來。把穩定性拿掉不是變慢，是直接錯 - 而且沒有任何後續的 pass 能救回來。',
           en:'Radix sort is <b>counting sort one digit at a time</b>, least significant first. The only reason it is correct is that <b>every pass is stable</b>: elements sharing a high digit are not reordered, so the order the low digits established survives. Remove stability and it does not get slower, it gets wrong - and no later pass can repair it.'},
     legend:['hot', 'ok', 'bad', 'soft'],
     code:CODE_R,
     build:(v) => radixFrames(!v)},
    {id:'bucket', label:{zh:'Bucket sort 的賭注', en:'the bet bucket sort makes'},
     stage:{zh:'同樣的程式碼，換個分布就從 n 掉到 n²',
            en:'same code - change the distribution and n becomes n squared'},
     view:VIEW,
     variants:[{zh:'均勻分布', en:'uniform'}, {zh:'擠成一團', en:'clustered'}],
     idea:{zh:'bucket sort 把 key 當成<b>估計值</b>而不是位址：縮放進 m 個桶，桶內另外排，最後接起來。它是<b>期望</b>線性，前提是輸入接近均勻。分布不是註腳，是前置條件 - 資料一歪，某個桶就吃掉幾乎所有元素，而它<b>不會報錯</b>，只會安靜地變成 n²。',
           en:'Bucket sort treats the key as an <b>estimate</b> rather than an address: scale into m buckets, sort inside each, concatenate. It is linear <b>in expectation</b>, and only if the input is near-uniform. The distribution is not a footnote but a precondition - skew the data and one bucket swallows almost everything, and <b>nothing raises</b>: it just quietly becomes n squared.'},
     legend:['hot', 'ok', 'ghost', 'soft'],
     code:CODE_B,
     build:(v) => bucketFrames(!!v)},
    {id:'scan', label:{zh:'中間那個 scan', en:'the scan in the middle'},
     stage:{zh:'work 與 depth 的取捨，不是「變快」',
            en:'a trade between work and depth - nothing here is faster'},
     view:VIEW,
     variants:[{zh:'Hillis-Steele', en:'Hillis-Steele'}, {zh:'Blelloch', en:'Blelloch'}],
     idea:{zh:'count 和 scatter 都能一個 thread 一個元素，只有 scan 不行 - 它每一步都讀上一步剛寫的東西。<b>Hillis-Steele</b> 用 n log n 次加法把 depth 壓到 log n；<b>Blelloch</b> 用上掃 / 下掃兩趟樹，只花 2(n-1) 次加法拿到同一個 depth 等級。兩個都比原本的迴圈<b>做更多加法</b>，它們之所以划算只因為 lane 本來就閒著。',
           en:'Count and scatter are one thread per element; only the scan is not, because every step reads what the previous one just wrote. <b>Hillis-Steele</b> spends n log n additions to get depth down to log n. <b>Blelloch</b> uses an up-sweep and a down-sweep to reach the same depth class for 2(n-1) additions. Both do <b>more</b> additions than the loop they replace - they win only because the lanes were idle.'},
     legend:['hot', 'act', 'ok', 'soft'],
     code:CODE_S,
     build:(v) => scanFrames(!!v)},
    {id:'moe', label:{zh:'真的 kernel：MoE token sorting', en:'a real kernel: MoE token sorting'},
     stage:{zh:'moe_align_block_size 就是一個 counting sort',
            en:'moe_align_block_size is a counting sort'},
     view:VIEW,
     idea:{zh:'router 給每個 token 一組 expert id，GEMM 要求同一個 expert 的 token 連續、且對齊固定 block。expert id 是小整數 - 所以就是 counting sort，只加了兩件事：<b>每桶先進位到 block_size 再 scan</b>（洞用 sentinel 補），以及<b>用同一個 prefix 陣列二分搜尋回推每個 block 屬於哪個 expert</b>。一次 scan，三份工。',
           en:'The router gives each token a set of expert ids, and the GEMM wants each expert\'s tokens contiguous and block-aligned. Expert ids are small integers, so this is counting sort plus two things: <b>each bucket is rounded up to block_size before the scan</b> (holes filled with a sentinel), and <b>the same prefix array is binary-searched to recover which expert owns each block</b>. One scan, three uses.'},
     legend:['hot', 'act', 'ok', 'soft'],
     code:CODE_M,
     build:() => moeFrames()},
    {id:'lc164', label:{zh:'LC 164 · Maximum Gap', en:'LC 164 · Maximum Gap'},
     stage:{zh:'要 O(n)，就等於不准排序',
            en:'demanding O(n) is the same as forbidding a sort'},
     view:VIEW,
     idea:{zh:'把 n 個數丟進 n-1 個桶，鴿籠原理保證<b>至少一個桶是空的</b>，所以最大 gap 至少一個桶寬，<b>不可能落在桶內</b>。於是每個桶只需要留 min 和 max，桶內順序完全不用算 - 這就是它能線性的全部理由：不是排得比較快，是看出哪一段排序可以不做。',
           en:'Drop n numbers into n-1 buckets and pigeonhole guarantees <b>at least one is empty</b>, so the largest gap spans at least one bucket width and <b>cannot lie inside a bucket</b>. Each bucket therefore needs only a min and a max, and the in-bucket order is never computed. That is the entire reason it is linear: not a faster sort, but noticing which part of the sort can be skipped.'},
     legend:['hot', 'ghost', 'ok', 'act'],
     code:CODE_G,
     build:() => gapFrames()}
  ]
};
