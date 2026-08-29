// DAY: 21
// TITLE_ZH: 選擇問題與 Top-K：從 quickselect 到 GPU kernel
// TITLE_EN: Selection and top-k - from quickselect to the kernels inside an LLM
// SUB_ZH: 要前 k 名不必先排序；把這件事搬到 GPU 上，難的不是演算法，是讓幾百個 block 講好話。
// SUB_EN: You never had to sort to get the top k. Moving that onto a GPU, the hard part is not the algorithm - it is getting hundreds of blocks to agree.
// FOLDER: day%2021%20-%20selection%20and%20top-k
// MEDIUM: https://medium.com/100-days-of-python

/* ------------------------------------------------------------ layout bits */
const VIEW = [9.8, 6.4];
const CW = .62, CH = .62;
const rowX = n => (VIEW[0] - n * CW) / 2;

function arrRow(vals, y, st, opt){
  opt = opt || {};
  const x0 = opt.x0 == null ? rowX(vals.length) : opt.x0;
  const out = [];
  vals.forEach((v, i) => {
    out.push(S.r(x0 + i * CW, y, CW - .07, CH, (st && st[i]) || 'idle',
                 v == null ? '' : String(v), {fs:opt.fs || .30, rx:.05}));
  });
  if (opt.title)
    out.push(S.t(x0 - .24, y + CH * .58, opt.title, {c:COL.tealL, fs:.30, anchor:'end'}));
  (opt.marks || []).forEach(m => {
    out.push(S.t(x0 + m.i * CW + (CW - .07) / 2, m.below ? y + CH + .40 : y - .20,
                 m.t, {c:m.c || COL.purpleL, fs:.30}));
  });
  return out;
}
const note = (y, s, c, fs) => S.t(VIEW[0] / 2, y, s, {c:c || COL.tealL, fs:fs || .32});
const chips = (lbl, arr) => ({lbl:lbl, chips:arr});

/* a tiny seeded RNG so every replay records exactly the same frames */
function lcg(seed){
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/* ====================================================== 1. quickselect ==== */
const CODE_QS = [
'def quickselect(a, k):',
'    lo, hi = 0, len(a) - 1',
'    while lo <= hi:',
'        p = a[randint(lo, hi)]        # random pivot',
'        lt, i, gt = lo, lo, hi',
'        while i <= gt:                # Dutch national flag',
'            if a[i] > p:',
'                swap(lt, i); lt += 1; i += 1',
'            elif a[i] < p:',
'                swap(i, gt); gt -= 1  # i does NOT advance',
'            else:',
'                i += 1',
'        if k <= lt - lo:',
'            hi = lt - 1               # answer is all in the > side',
'        elif k <= gt + 1 - lo:',
'            break                     # k-th is one of the ties',
'        else:',
'            k -= gt + 1 - lo          # drop > and ==, keep hunting',
'            lo = gt + 1',
'    return a[:k]'
];

function qsFrames(cfg){
  const F = new Frames();
  const a = cfg.data.slice();
  const n = a.length;
  const Y = 3.1;
  let k = cfg.k, lo = 0, hi = n - 1, looked = 0, rounds = 0;
  const rnd = lcg(cfg.seed || 7);

  const draw = (o) => {
    o = o || {};
    const st = {};
    for (let i = 0; i < n; i++){
      st[i] = (i < lo || i > hi) ? 'done' : 'soft';
    }
    if (o.st) Object.keys(o.st).forEach(i => { st[i] = o.st[i]; });
    let sh = arrRow(a, Y, st, {title:{zh:'陣列', en:'array'}, marks:o.marks || []});
    sh.push(note(1.95, o.sub || '', COL.grey, .30));
    if (o.pivot != null)
      sh.push(note(5.35, {zh:'pivot = ' + o.pivot, en:'pivot = ' + o.pivot}, COL.orangeL, .36));
    sh.push(note(5.85, o.head || cfg.head, COL.tealL, .34));
    return sh;
  };
  const panels = () => [
    chips({zh:'還要找幾個', en:'still looking for'}, [{t:'k = ' + k, cls:'hot'}]),
    chips({zh:'還活著的區間', en:'live window'},
          [{t:'[' + lo + ', ' + hi + ']', cls:'act'},
           {t:(hi - lo + 1) + (' of ' + n), cls:''}]),
    chips({zh:'看過幾個元素', en:'elements examined'}, [{t:String(looked), cls:''}])
  ];

  F.push({shapes:draw({sub:{zh:'目標：前 ' + cfg.k + ' 大，而且不排序',
                            en:'goal: the top ' + cfg.k + ', without sorting'}}),
          panels:panels(), view:VIEW, line:0,
          msg:cfg.intro});

  while (lo <= hi && rounds < 8){
    rounds++;
    const pi = lo + Math.floor(rnd() * (hi - lo + 1));
    const p = a[pi];
    let lt = lo, i = lo, gt = hi;
    F.push({shapes:draw({pivot:p, st:{[pi]:'hot'}, marks:[{i:pi, t:'pivot'}],
                         sub:{zh:'隨機選 pivot：沒有哪一種輸入是特別的',
                              en:'a random pivot: no input is special'}}),
            panels:panels(), view:VIEW, line:3,
            msg:{zh:'從還活著的區間裡隨機挑一個 pivot = ' + p + '。隨機化的意義不是變快，是讓「最壞情況」不再由輸入決定，而是由骰子決定。',
                 en:'Pick a pivot at random from the live window: ' + p + '. Randomising does not make it faster; it moves the worst case from something the input controls to something the dice control.'}});

    while (i <= gt){
      looked++;
      const st = {};
      for (let j = lo; j < lt; j++) st[j] = 'ok';
      for (let j = gt + 1; j <= hi; j++) st[j] = 'done';
      st[i] = 'hot';
      const marks = [{i:lt, t:'lt'}, {i:i, t:'i', c:COL.orangeL}, {i:gt, t:'gt'}];
      let line, msg;
      if (a[i] > p){
        line = 7;
        msg = {zh:'a[' + i + '] = ' + a[i] + ' 大於 pivot，換到左邊那一區，lt 和 i 一起往前。',
               en:'a[' + i + '] = ' + a[i] + ' beats the pivot, so it swaps into the left region and both lt and i move on.'};
      } else if (a[i] < p){
        line = 9;
        msg = {zh:'a[' + i + '] = ' + a[i] + ' 小於 pivot，換到右邊。<b>i 不能前進</b> - 換過來的那個值還沒有人看過。這一行是三路 partition 最常寫錯的地方。',
               en:'a[' + i + '] = ' + a[i] + ' loses, so it swaps to the right end. <b>i must not advance</b> - the value swapped in has not been examined yet. This is the line people get wrong.'};
      } else {
        line = 11;
        msg = {zh:'a[' + i + '] 正好等於 pivot，留在中間這一區。有了這一區，一整排相同的值一次就處理完。',
               en:'a[' + i + '] ties with the pivot and stays in the middle region. That region is what lets a run of equal values retire in a single pass.'};
      }
      F.push({shapes:draw({pivot:p, st:st, marks:marks,
                           sub:{zh:'< p ｜ == p ｜ > p 三個區間', en:'three regions: > p | == p | < p'}}),
              panels:panels(), view:VIEW, line:line, msg:msg});
      if (a[i] > p){ const t = a[lt]; a[lt] = a[i]; a[i] = t; lt++; i++; }
      else if (a[i] < p){ const t = a[i]; a[i] = a[gt]; a[gt] = t; gt--; }
      else i++;
    }

    const big = lt - lo, eq = gt + 1 - lt;
    const st = {};
    for (let j = lo; j < lt; j++) st[j] = 'ok';
    for (let j = lt; j <= gt; j++) st[j] = 'act';
    if (k <= big){
      F.push({shapes:draw({pivot:p, st:st,
                           sub:{zh:big + ' 個大於 pivot，k = ' + k + ' 全落在裡面',
                                en:big + ' beat the pivot, and all ' + k + ' answers live in there'}}),
              panels:panels(), view:VIEW, line:13,
              msg:{zh:'前 k 大整個落在「大於 pivot」那一區裡，所以右邊全部丟掉 - 一次丟掉 ' + (hi - lt + 1) + ' 個元素，而且永遠不會再看它們一眼。',
                   en:'The whole answer lives in the "greater than pivot" region, so everything to the right is discarded - ' + (hi - lt + 1) + ' elements gone in one step, never to be looked at again.'}});
      hi = lt - 1;
    } else if (k <= big + eq){
      F.push({shapes:draw({pivot:p, st:st,
                           sub:{zh:'邊界落在「等於 pivot」那一區', en:'the boundary falls inside the ties'}}),
              panels:panels(), view:VIEW, line:15,
              msg:{zh:'第 k 大就是 pivot 本身：' + big + ' 個嚴格大於它，剩下的名額要從 ' + eq + ' 個平手的元素裡挑。<b>挑誰？演算法沒有規定</b> - 這正是後面 GPU kernel 要處理 tie_break 的原因。',
                   en:'The k-th largest is the pivot itself: ' + big + ' beat it, and the remaining slots come out of ' + eq + ' tied elements. <b>Which ones? The algorithm does not say</b> - which is exactly why the GPU kernels later need a tie_break.'}});
      lo = hi + 1;
      break;
    } else {
      F.push({shapes:draw({pivot:p, st:st,
                           sub:{zh:'大於和等於的都是贏家，繼續往右找', en:'the > and == regions are all winners; keep hunting right'}}),
              panels:panels(), view:VIEW, line:17,
              msg:{zh:'這 ' + (big + eq) + ' 個都進榜了，從 k 扣掉，然後只在右邊繼續找。注意每一輪都只往<b>一邊</b>遞迴 - n + n/2 + n/4 + … = 2n，這就是 O(n) 的來源。',
                   en:'All ' + (big + eq) + ' of them are in, so subtract them from k and keep hunting on the right only. Every round recurses into <b>one</b> side - n + n/2 + n/4 + ... = 2n, and that is where the O(n) comes from.'}});
      k -= big + eq;
      lo = gt + 1;
    }
  }

  const stF = {};
  const res = a.slice(0, cfg.k);
  for (let i = 0; i < n; i++) stF[i] = i < cfg.k ? 'ok' : 'done';
  F.push({shapes:draw({st:stF, head:cfg.head,
                       sub:{zh:'前 ' + cfg.k + ' 大：' + res.join(', '),
                            en:'the top ' + cfg.k + ': ' + res.join(', ')}}),
          panels:panels(), view:VIEW, line:19, msg:cfg.outro});
  return F.list;
}

/* ================================================== 2. radix select ======= */
const CODE_RX = [
'RADIX_BITS = 2                      # 4 bins; a real kernel uses 8 -> 256',
'',
'def radix_select(keys, k):',
'    prefix, live = 0, list(range(len(keys)))',
'    for shift in (6, 4, 2, 0):      # one byte at a time, high to low',
'        hist = [0] * 4',
'        for i in live:              # count - do not move the data',
'            hist[(keys[i] >> shift) & 3] += 1',
'        run = 0',
'        for b in (3, 2, 1, 0):      # walk bins from the top down',
'            if run + hist[b] >= k:',
'                break               # k-th lands inside bin b',
'            run += hist[b]',
'        k -= run                    # bins above b are all winners',
'        prefix |= b << shift',
'        live = [i for i in live if (keys[i] >> shift) & 3 == b]',
'    return prefix, live'
];

function rxFrames(cfg){
  const F = new Frames();
  const keys = cfg.keys.slice();
  const n = keys.length;
  const bits = v => ('00000000' + v.toString(2)).slice(-8);
  const Y = 4.05, HY = 1.15;
  let k = cfg.k, live = keys.map((_, i) => i), prefix = 0, decided = 0;

  const draw = (o) => {
    o = o || {};
    const sh = [];
    const st = {};
    for (let i = 0; i < n; i++) st[i] = live.indexOf(i) < 0 ? 'done' : 'soft';
    if (o.st) Object.keys(o.st).forEach(i => { st[i] = o.st[i]; });
    const x0 = rowX(n) - .5;
    keys.forEach((v, i) => {
      const b = bits(v);
      sh.push(S.r(x0 + i * CW, Y, CW - .07, .52, st[i], String(v), {fs:.26, rx:.05}));
      sh.push(S.t(x0 + i * CW + (CW - .07) / 2, Y - .18,
                  b.slice(0, decided) + '|' + b.slice(decided, decided + 2),
                  {c:st[i] === 'done' ? COL.grey : COL.orangeL, fs:.21}));
    });
    sh.push(S.t(x0 - .22, Y + .30, {zh:'值', en:'value'}, {c:COL.tealL, fs:.28, anchor:'end'}));
    // histogram
    if (o.hist){
      const top = Math.max.apply(null, o.hist) || 1;
      const bx = 2.7, bw = 1.0;
      for (let b = 0; b < 4; b++){
        const h = 1.5 * o.hist[b] / top;
        const col = b === o.bucket ? 'hot' : (o.above && b > o.bucket ? 'ok' : 'idle');
        sh.push(S.r(bx + b * (bw + .35), HY, bw, Math.max(h, .06), col, '', {rx:.04}));
        sh.push(S.t(bx + b * (bw + .35) + bw / 2, HY - .22,
                    'bin ' + ('00' + b.toString(2)).slice(-2), {c:COL.grey, fs:.26}));
        sh.push(S.t(bx + b * (bw + .35) + bw / 2, HY + Math.max(h, .06) + .18,
                    String(o.hist[b]), {c:b === o.bucket ? COL.orangeL : COL.tealL, fs:.28}));
      }
      if (o.bucket != null){
        const bxx = bx + o.bucket * (bw + .35);
        sh.push(S.r(bxx - .12, HY - .42, bw + .24, 2.34, 'hot', '', {rx:.06, o:.20, dash:'.14 .10'}));
      }
    }
    sh.push(note(3.32, o.sub || '', COL.grey, .30));
    sh.push(note(5.95, cfg.head, COL.tealL, .34));
    return sh;
  };
  const panels = () => [
    chips({zh:'還要找幾個', en:'still looking for'}, [{t:'k = ' + k, cls:'hot'}]),
    chips({zh:'已決定的前綴', en:'prefix decided'},
          [{t:('00000000' + prefix.toString(2)).slice(-8).slice(0, decided) || '(none)', cls:'ok'}]),
    chips({zh:'還活著的元素', en:'candidates alive'},
          [{t:live.length + ' / ' + n, cls:'act'}])
  ];

  F.push({shapes:draw({sub:{zh:'8-bit key，每輪處理 2 個 bit', en:'8-bit keys, two bits per round'}}),
          panels:panels(), view:VIEW, line:0, msg:cfg.intro});

  const shifts = [6, 4, 2, 0];
  for (let r = 0; r < 4; r++){
    const shift = shifts[r];
    const hist = [0, 0, 0, 0];
    live.forEach(i => { hist[(keys[i] >> shift) & 3]++; });
    F.push({shapes:draw({hist:hist, sub:{zh:'第 ' + (r + 1) + ' 輪：只數，不搬動任何資料',
                                          en:'round ' + (r + 1) + ': count only - nothing moves'}}),
            panels:panels(), view:VIEW, line:7,
            msg:{zh:'把還活著的 ' + live.length + ' 個元素照這 2 個 bit 丟進 4 個 bin 裡數一數。<b>資料完全沒有被搬動</b>，動的只有 4 個計數器 - 真正的 kernel 是 256 個計數器，不管那一列有 20 萬個元素還是 200 個。',
                 en:'Bin the ' + live.length + ' live elements by these two bits and count. <b>The data never moves</b>; only four counters do - in a real kernel, 256 counters, whether the row holds 200 values or 200,000.'}});

    let run = 0, b = 0;
    for (b = 3; b >= 0; b--){
      if (run + hist[b] >= k) break;
      F.push({shapes:draw({hist:hist, bucket:b, above:true,
                           sub:{zh:'bin ' + b + ' 只有 ' + hist[b] + ' 個，全部進榜，還差 ' + (k - run - hist[b]),
                                en:'bin ' + b + ' holds only ' + hist[b] + ' - all winners, ' + (k - run - hist[b]) + ' still needed'}}),
              panels:panels(), view:VIEW, line:11,
              msg:{zh:'從最高的 bin 往下累加：目前累積 ' + (run + hist[b]) + ' 個還不到 k = ' + k + '，所以這個 bin 裡的全部都穩穩進榜，繼續往下一個 bin。',
                   en:'Accumulate from the top bin downwards: ' + (run + hist[b]) + ' is still short of k = ' + k + ', so everything in this bin is safely in the answer. Move down.'}});
      run += hist[b];
    }
    if (b < 0) b = 0;
    F.push({shapes:draw({hist:hist, bucket:b, above:true,
                         sub:{zh:'累積 ' + (run + hist[b]) + ' >= k，第 k 大就在 bin ' + b + ' 裡',
                              en:'running total ' + (run + hist[b]) + ' >= k, so the k-th lives in bin ' + b}}),
            panels:panels(), view:VIEW, line:12,
            msg:{zh:'跨過 k 的那個 bin 就是<b>邊界 bin</b>。比它高的 bin 全是贏家，比它低的全部淘汰，只有它裡面的 ' + hist[b] + ' 個還沒定案 - 這一輪就靠 4 個數字把問題縮小了，完全沒有比較過任何兩個元素。',
                 en:'The bin that straddles k is the <b>boundary bin</b>. Everything above it is in, everything below it is out, and only the ' + hist[b] + ' inside are undecided. The round shrank the problem using four integers, without comparing a single pair of elements.'}});
    k -= run;
    prefix |= b << shift;
    decided += 2;
    live = live.filter(i => ((keys[i] >> shift) & 3) === b);
    F.push({shapes:draw({sub:{zh:'保留 bin ' + b + '，k 調整為 ' + k,
                              en:'keep bin ' + b + ', k becomes ' + k}}),
            panels:panels(), view:VIEW, line:15,
            msg:{zh:'k 扣掉上面那些贏家變成 ' + k + '，前綴多決定 2 個 bit，活著的剩 ' + live.length + ' 個。<b>k 會變</b>是 radix select 最容易寫錯的地方：每一輪都是在剩下的子集合裡問一個新的、比較小的問題。',
                 en:'Subtract the winners above, so k is now ' + k + '; two more bits of the prefix are pinned, and ' + live.length + ' candidates remain. <b>k changes</b> - the classic bug in radix select. Each round asks a new, smaller question of a smaller set.'}});
  }
  const stF = {}; live.forEach(i => { stF[i] = 'ok'; });
  F.push({shapes:draw({st:stF, sub:{zh:'第 ' + cfg.k + ' 大 = ' + prefix + '，四輪、256 個計數器，沒排序',
                                     en:'the ' + cfg.k + '-th largest = ' + prefix + ' - four rounds of counters, no sorting'}}),
          panels:panels(), view:VIEW, line:16, msg:cfg.outro});
  return F.list;
}

/* =========================================== 3. filtered top-k, one block = */
const CODE_FT = [
'def filtered_topk(row, k, capacity):',
'    hist = [0] * NBINS                      # pass 1: coarse key only',
'    for v in row:',
'        hist[coarse_bin(v)] += 1',
'    run = 0',
'    for b in range(NBINS - 1, -1, -1):      # find the straddling bin',
'        if run + hist[b] >= k: break',
'        run += hist[b]',
'    v_lo, v_hi = bin_edges(b)               # project the edges back to float',
'    winners, cand = [], []',
'    for v in row:                           # pass 2: two float compares',
'        if v >= v_hi:   winners.append(v)    #   certainly in',
'        elif v >= v_lo: cand.append(v)       #   undecided -> shared memory',
'    if len(cand) > capacity:',
'        return None                         # overflowed: use the grid kernel',
'    cand.sort(reverse=True)                 # refine on chip, no more passes',
'    return sorted(winners + cand[:k - len(winners)], reverse=True)'
];

function ftFrames(cfg){
  const F = new Frames();
  const row = cfg.row.slice(), n = row.length, k = cfg.k, cap = cfg.cap;
  const NB = 10, bin = v => Math.min(NB - 1, Math.floor(v / 10));
  const Y = 4.35, HY = 1.05;
  const hist = new Array(NB).fill(0);
  let seen = 0, bucket = null, vlo = null, vhi = null;
  const winners = [], cand = [], out = [];

  const drawRow = (st) => {
    const sh = [], x0 = rowX(n);
    row.forEach((v, i) => {
      sh.push(S.r(x0 + i * CW, Y, CW - .07, .50, (st && st[i]) || 'idle', String(v), {fs:.26, rx:.05}));
    });
    sh.push(S.t(x0 - .22, Y + .28, {zh:'一列分數', en:'one row of scores'},
                {c:COL.tealL, fs:.28, anchor:'end'}));
    return sh;
  };
  const drawHist = (mark) => {
    const sh = [], top = Math.max.apply(null, hist) || 1, bx = 1.15, bw = .70;
    for (let b = 0; b < NB; b++){
      const h = 1.6 * hist[b] / top;
      const cls = (mark != null && b === mark) ? 'hot' : (mark != null && b > mark ? 'ok' : 'idle');
      sh.push(S.r(bx + b * (bw + .18), HY, bw, Math.max(h, .05), cls, '', {rx:.04}));
      sh.push(S.t(bx + b * (bw + .18) + bw / 2, HY - .22, String(b * 10), {c:COL.grey, fs:.24}));
      sh.push(S.t(bx + b * (bw + .18) + bw / 2, HY + Math.max(h, .05) + .18, String(hist[b]),
                  {c:cls === 'hot' ? COL.orangeL : COL.tealL, fs:.26}));
    }
    return sh;
  };
  const panels = () => [
    chips({zh:'一定進榜 / 待定 / 淘汰', en:'winners / candidates / out'},
          [{t:String(winners.length), cls:'ok'}, {t:String(cand.length), cls:'hot'},
           {t:String(out.length), cls:''}]),
    chips({zh:'shared memory 容量', en:'shared-memory capacity'},
          [{t:cand.length + ' / ' + cap, cls:cand.length > cap ? 'bad' : 'act'}]),
    chips({zh:'邊界', en:'boundary'},
          [{t:vlo == null ? '-' : ('v_lo = ' + vlo), cls:''},
           {t:vhi == null ? '-' : ('v_hi = ' + vhi), cls:''}])
  ];

  F.push({shapes:drawRow({}).concat(drawHist(null),
            [note(3.55, {zh:'k = ' + k + '，shared memory 只裝得下 ' + cap + ' 個',
                         en:'k = ' + k + ', and shared memory holds only ' + cap},
                  COL.grey, .30), note(5.95, cfg.head, COL.tealL, .34)]),
          panels:panels(), view:VIEW, line:0, msg:cfg.intro});

  /* pass 1 - streaming histogram */
  for (let i = 0; i < n; i++){
    hist[bin(row[i])]++; seen++;
    if (i < 3 || i === n - 1){
      const st = {}; for (let j = 0; j <= i; j++) st[j] = j === i ? 'hot' : 'done';
      F.push({shapes:drawRow(st).concat(drawHist(null),
                [note(3.55, {zh:'串流第一趟：讀過就丟，只留 ' + NB + ' 個計數器',
                             en:'pass 1 streams: each value is read once and dropped, ' + NB + ' counters kept'},
                      COL.grey, .30), note(5.95, cfg.head, COL.tealL, .34)]),
              panels:panels(), view:VIEW, line:3,
              msg:i === n - 1
                ? {zh:'整列讀完，記憶體用量是常數。這是 filtered top-k 的第一個關鍵：<b>直方圖不需要把整列留下來</b>。',
                   en:'The whole row is consumed with constant memory. That is the first trick: <b>a histogram never needs the row kept around</b>.'}
                : {zh:'把 ' + row[i] + ' 丟進 bin ' + bin(row[i]) + '。實際的 kernel 用的是「粗 key」- 從浮點數取出高位元組。取多少位元決定了這張圖有多平均，後面就會看到差別。',
                   en:'Drop ' + row[i] + ' into bin ' + bin(row[i]) + '. A real kernel uses a <i>coarse key</i> - the top byte of the float. How many bits you take decides how evenly this histogram spreads, and that matters more than it looks.'}});
    }
  }

  /* find the straddling bin */
  let run = 0, b = NB - 1;
  for (; b >= 0; b--){
    if (run + hist[b] >= k) break;
    if (hist[b]){
      F.push({shapes:drawRow({}).concat(drawHist(b),
                [note(3.55, {zh:'bin ' + b + ' 的 ' + hist[b] + ' 個全部進榜',
                             en:'all ' + hist[b] + ' in bin ' + b + ' are winners'}, COL.grey, .30),
                 note(5.95, cfg.head, COL.tealL, .34)]),
              panels:panels(), view:VIEW, line:7,
              msg:{zh:'從最高的 bin 往下累加，還沒到 k。',
                   en:'Accumulating from the top bin down; still short of k.'}});
    }
    run += hist[b];
  }
  if (b < 0) b = 0;
  bucket = b; vlo = b * 10; vhi = (b + 1) * 10;
  F.push({shapes:drawRow({}).concat(drawHist(bucket),
            [note(3.55, {zh:'邊界 bin = ' + bucket + '，把它的兩個邊界投影回浮點數',
                         en:'boundary bin = ' + bucket + '; project its two edges back to float'}, COL.grey, .30),
             note(5.95, cfg.head, COL.tealL, .34)]),
          panels:panels(), view:VIEW, line:8,
          msg:{zh:'找到跨過 k 的 bin 之後，把它的上下界<b>各投影回浮點數一次</b>，得到 v_lo 和 v_hi。這兩個數字接下來整趟迴圈都不會變，所以編譯器可以把它們留在暫存器裡 - 第二趟就只是兩次浮點比較，沒有位元運算、沒有查表。',
               en:'Once the straddling bin is known, project its two edges back into float <b>once</b>, giving v_lo and v_hi. They are loop-invariant, so they sit in registers: the second pass becomes two float compares per element - no bit twiddling, no table lookups.'}});

  /* pass 2 - classify */
  const COLX = {win:1.4, cand:4.6, out:7.8};
  const drawCols = (hot) => {
    const sh = [];
    const col = (x, list, cls, ttl) => {
      sh.push(S.t(x + .55, .70, ttl, {c:COL.tealL, fs:.30}));
      list.slice(0, 8).forEach((v, j) => {
        sh.push(S.r(x, 1.05 + j * .40, 1.10, .34, cls, String(v), {fs:.24, rx:.04}));
      });
      if (list.length > 8)
        sh.push(S.t(x + .55, 1.05 + 8 * .40 + .22, '+' + (list.length - 8) + ' more',
                    {c:COL.grey, fs:.24}));
    };
    col(COLX.win, winners, 'ok', {zh:'一定進榜 (>= v_hi)', en:'certainly in (>= v_hi)'});
    col(COLX.cand, cand, 'hot', {zh:'待定 -> shared memory', en:'undecided -> shared memory'});
    col(COLX.out, out, 'done', {zh:'淘汰 (< v_lo)', en:'out (< v_lo)'});
    return sh;
  };
  for (let i = 0; i < n; i++){
    const v = row[i];
    let where, line;
    if (v >= vhi){ winners.push(v); where = 'winner'; line = 11; }
    else if (v >= vlo){ cand.push(v); where = 'cand'; line = 12; }
    else { out.push(v); where = 'out'; line = 12; }
    if (i < 4 || i === n - 1 || (where === 'cand' && cand.length <= 3)){
      const st = {}; for (let j = 0; j <= i; j++) st[j] = j === i ? 'hot' : 'done';
      F.push({shapes:drawRow(st).concat(drawCols(),
                [note(3.75, {zh:'v_lo = ' + vlo + '  v_hi = ' + vhi + '：兩次浮點比較就分完三類',
                             en:'v_lo = ' + vlo + ', v_hi = ' + vhi + ' - two float compares decide all three classes'},
                      COL.orangeL, .30), note(5.95, cfg.head, COL.tealL, .34)]),
              panels:panels(), view:VIEW, line:line,
              msg:where === 'winner'
                ? {zh:v + ' >= v_hi，穩穩進榜，連放進 shared memory 都不必。',
                   en:v + ' >= v_hi, so it is in for certain and never has to be stored on chip.'}
                : where === 'cand'
                ? {zh:v + ' 落在邊界 bin 裡，不看第二眼分不出勝負，寫進 shared memory 等第二階段。<b>會用掉晶片上空間的只有這一類</b>。',
                   en:v + ' lands inside the boundary bin, so it cannot be decided without a closer look - it goes into shared memory. <b>Only this class costs on-chip space</b>.'}
                : {zh:v + ' < v_lo，直接丟掉，不佔任何空間。',
                   en:v + ' < v_lo, dropped immediately, costing nothing.'}});
    }
  }

  const over = cand.length > cap;
  F.push({shapes:drawRow({}).concat(drawCols(),
            [note(3.75, over
                ? {zh:'待定 ' + cand.length + ' 個 > 容量 ' + cap + '：這一列不能用這顆 kernel',
                   en:cand.length + ' candidates exceed the ' + cap + '-slot budget - this row cannot use this kernel'}
                : {zh:'待定只剩 ' + cand.length + ' 個，佔整列的 ' + (100 * cand.length / n).toFixed(1) + '%',
                   en:'only ' + cand.length + ' candidates survive - ' + (100 * cand.length / n).toFixed(1) + '% of the row'},
              over ? COL.red : COL.orangeL, .30),
             note(5.95, cfg.head, COL.tealL, .34)]),
          panels:panels(), view:VIEW, line:over ? 14 : 16, msg:cfg.outro});
  return F.list;
}

/* ================================== 4-5. multi-CTA radix select + barrier = */
const CODE_CTA = [
'# one row, split across the whole grid - every CTA runs this same code',
'for r in range(NUM_ROUNDS):',
'    h = hist[r % 3]                     # triple buffered',
'    for i in range(cta, n, num_ctas):   # my slice of the row',
'        if alive(keys[i]):',
'            atomic_add(h[digit(keys[i], r)], 1)',
'    barrier(r)                          # everybody has finished counting',
'    if cta == 0:',
'        clear(hist[(r + 1) % 3])        # prepare the buffer two rounds out',
'    bucket, k = scan(h, k)              # every CTA scans the same numbers',
'    barrier(r)                          # nobody reads a buffer being cleared',
'',
'def barrier(phase):                     # grid-wide, counter never reset',
'    old = atomic_add(counter, 1)',
'    target = (phase + 1) * num_ctas',
'    while load_acquire(counter) < target:',
'        pass                            # >= , not == : a late CTA can overshoot'
];

function ctaFrames(cfg){
  const F = new Frames();
  const NC = cfg.ctas, keys = cfg.keys.slice(), n = keys.length;
  const NB = 4, mode = cfg.mode || 'ok';
  const slow = cfg.slow == null ? NC - 1 : cfg.slow;
  const laneY = c => 4.55 - c * .72;
  let counter = 0, k = cfg.k, live = keys.map((_, i) => i), stuck = null;
  const bufs = [0, 1, 2];
  let hist = [0, 0, 0, 0], curBuf = 0;

  const drawLanes = (o) => {
    o = o || {};
    const sh = [];
    for (let c = 0; c < NC; c++){
      const mine = [];
      for (let i = c; i < n; i += NC) mine.push(i);
      const cls = o.lane && o.lane[c] ? o.lane[c] : 'idle';
      sh.push(S.r(1.30, laneY(c), 4.60, .54, cls, '', {rx:.06, o:.55}));
      sh.push(S.t(1.18, laneY(c) + .36, 'CTA ' + c, {c:COL.tealL, fs:.28, anchor:'end'}));
      mine.forEach((idx, j) => {
        const dead = live.indexOf(idx) < 0;
        sh.push(S.t(1.55 + j * .58, laneY(c) + .36, String(keys[idx]),
                    {c:dead ? COL.grey : COL.pale, fs:.26}));
      });
      if (o.mark && o.mark[c])
        sh.push(S.t(6.05, laneY(c) + .36, o.mark[c], {c:o.markc && o.markc[c] || COL.orangeL,
                                                      fs:.27, anchor:'start'}));
    }
    return sh;
  };
  const drawHist = (bucket) => {
    const sh = [], top = Math.max.apply(null, hist) || 1, bx = 1.45, bw = .78;
    sh.push(S.t(1.30, 1.86, {zh:'共用直方圖 hist[' + curBuf + ']', en:'shared histogram hist[' + curBuf + ']'},
                {c:COL.tealL, fs:.28, anchor:'start'}));
    for (let b = 0; b < NB; b++){
      const h = 1.05 * hist[b] / top;
      const cls = bucket != null && b === bucket ? 'hot' : (bucket != null && b > bucket ? 'ok' : 'idle');
      sh.push(S.r(bx + b * (bw + .30), .55, bw, Math.max(h, .05), cls, '', {rx:.04}));
      sh.push(S.t(bx + b * (bw + .30) + bw / 2, .33, 'bin ' + b, {c:COL.grey, fs:.24}));
      sh.push(S.t(bx + b * (bw + .30) + bw / 2, .55 + Math.max(h, .05) + .18, String(hist[b]),
                  {c:cls === 'hot' ? COL.orangeL : COL.tealL, fs:.26}));
    }
    // triple-buffer strip
    for (let t = 0; t < 3; t++){
      const cls = t === curBuf ? 'hot' : (t === (curBuf + 1) % 3 ? 'act' : 'done');
      sh.push(S.r(6.55 + t * 1.02, .70, .92, .46, cls, 'hist[' + t + ']', {fs:.24, rx:.05}));
    }
    sh.push(S.t(7.60, 1.46, {zh:'累加中 ｜ 清空中 ｜ 上一輪', en:'filling | being cleared | last round'},
                {c:COL.grey, fs:.24}));
    return sh;
  };
  const panels = () => [
    chips({zh:'barrier 計數器', en:'barrier counter'},
          [{t:String(counter), cls:'act'}, {t:'target ' + cfg.target, cls:''}]),
    chips({zh:'還要找幾個', en:'still looking for'}, [{t:'k = ' + k, cls:'hot'}]),
    chips({zh:'還活著的元素', en:'candidates alive'}, [{t:live.length + ' / ' + n, cls:'ok'}])
  ];
  cfg.target = 0;

  const push = (o) => F.push({
    shapes:drawLanes(o).concat(drawHist(o.bucket),
      [note(5.35, o.sub || '', o.subc || COL.grey, .30), note(5.95, cfg.head, COL.tealL, .34)]),
    panels:panels(), view:VIEW, line:o.line, msg:o.msg});

  push({line:0, sub:{zh:NC + ' 個 CTA 共同處理同一列 - 沒有人看得到整列',
                     en:NC + ' CTAs share one row - none of them can see all of it'},
        msg:cfg.intro});

  const shifts = cfg.shifts;
  for (let r = 0; r < shifts.length; r++){
    curBuf = r % 3;
    hist = [0, 0, 0, 0];
    const order = [];
    for (let c = 0; c < NC; c++) if (c !== slow) order.push(c);
    order.push(slow);
    const arrived = {};
    for (let oi = 0; oi < order.length; oi++){
      const c = order[oi];
      let added = 0;
      for (let i = c; i < n; i += NC)
        if (live.indexOf(i) >= 0){ hist[(keys[i] >> shifts[r]) & 3]++; added++; }
      arrived[c] = true;
      counter++;
      cfg.target = (r + 1) * NC;
      const lane = {}, mark = {}, markc = {};
      Object.keys(arrived).forEach(x => { lane[x] = 'done'; mark[x] = 'waiting'; });
      lane[c] = 'hot'; mark[c] = '+' + added; markc[c] = COL.orangeL;
      push({lane:lane, mark:mark, markc:markc, line:5,
            sub:{zh:'CTA ' + c + ' 把自己那一份 atomic 加進共用直方圖',
                 en:'CTA ' + c + ' atomically folds its slice into the shared histogram'},
            msg:oi === 0
              ? {zh:'每個 CTA 只掃自己那些格子（grid-stride：i, i+' + NC + ', i+' + 2 * NC + ' …）。它算出來的是<b>部分</b>直方圖，對答案一點用都沒有 - 要等其他人都加完才有意義。',
                 en:'Each CTA walks only its own stride (i, i+' + NC + ', i+' + 2 * NC + ', ...). What it computes is a <b>partial</b> histogram, useless on its own - it only means something once everyone has folded theirs in.'}
              : c === slow
              ? {zh:'最後一個 CTA 終於排到 SM 上。前面那些早就到的只能空轉等它 - 這就是 grid barrier 真正的成本：<b>最慢的那一個決定所有人的節奏</b>。',
                 en:'The last CTA finally gets scheduled. The ones that arrived early have been spinning - that is the real cost of a grid barrier: <b>the slowest CTA sets everyone else’s pace</b>.'}
              : {zh:'計數器加到 ' + counter + '，離這一輪的目標 ' + ((r + 1) * NC) + ' 還差 ' + ((r + 1) * NC - counter) + ' 個。',
                 en:'The counter reaches ' + counter + '; this phase needs ' + ((r + 1) * NC) + ', so ' + ((r + 1) * NC - counter) + ' still to come.'}});
    }
    /* the barrier release */
    if (mode === 'broken' && r === cfg.breakRound){
      counter = 0; stuck = true;
      const lane = {}, mark = {}, markc = {};
      for (let c = 0; c < NC; c++){ lane[c] = c === 0 ? 'bad' : 'act'; }
      mark[0] = 'counter = 0';  markc[0] = COL.red;
      for (let c = 1; c < NC; c++){ mark[c] = 'stuck'; markc[c] = COL.red; }
      push({lane:lane, mark:mark, markc:markc, line:14, subc:COL.red,
            sub:{zh:'CTA 0 先出關，順手把計數器歸零 - 其他人永遠等不到 target',
                 en:'CTA 0 leaves first and helpfully zeroes the counter - nobody else will ever reach the target'},
            msg:{zh:'這就是 flashinfer issue #3610。<b>「誰來重設計數器」看起來是個雜務，其實是正確性問題</b>：CTA 0 先通過 barrier 就把 counter 歸零，還在自旋的 CTA 讀到 0 < ' + ((r + 1) * NC) + '，於是永遠迴圈下去。整個 kernel 掛在那裡，沒有錯誤訊息、沒有崩潰，只有一張永遠跑不完的 GPU。',
                 en:'This is flashinfer issue #3610. <b>"Who resets the counter" looks like housekeeping and is actually a correctness question</b>: CTA 0 clears it on the way out, the CTAs still spinning read 0 < ' + ((r + 1) * NC) + ', and they spin forever. The kernel hangs with no error, no crash - just a GPU that never finishes.'}});
      F.push({shapes:F.list[F.list.length - 1].shapes, panels:panels(), view:VIEW, line:15,
              msg:{zh:'修法有兩種，flashinfer 選的是第二種：<b>不要重設</b>。讓計數器單調遞增，每個 CTA 自己記住 phase，等的是 (phase+1) * num_ctas。沒有人需要清東西，也就沒有人可以清錯。另一種是把離開 barrier 這件事本身當成選舉 - 只有看到 old + 1 == target 的那一個 CTA 是最後一個，由它負責清。',
                   en:'There are two fixes, and flashinfer took the second: <b>never reset</b>. Let the counter climb monotonically, have each CTA remember its own phase, and wait for (phase + 1) * num_ctas. Nothing needs clearing, so nothing can be cleared wrongly. The alternative is to make leaving the barrier an election - exactly one CTA sees old + 1 == target, and that one is the last, so it does the cleanup.'}});
      return F.list;
    }
    const lane = {}; for (let c = 0; c < NC; c++) lane[c] = 'ok';
    push({lane:lane, line:15,
          sub:{zh:'counter = ' + counter + ' >= ' + ((r + 1) * NC) + '，全體通過',
               en:'counter = ' + counter + ' >= ' + ((r + 1) * NC) + ' - everybody passes'},
          msg:{zh:'注意這裡是 <b>>=</b> 而不是 ==。被 deschedule 的 CTA 醒來時，計數器可能已經被下一輪的人推過頭了；用 == 會直接錯過釋放條件，然後永遠等下去。而且計數器<b>從頭到尾不歸零</b> - 每個 CTA 自己記得現在是第幾個 phase，target 就是 (phase+1) x ' + NC + '。',
               en:'Note the <b>>=</b>, not ==. A descheduled CTA can wake up after the counter has already been pushed past its target by the next phase; == would miss the release and spin forever. And the counter is <b>never reset</b> - each CTA remembers its own phase, so the target is simply (phase + 1) x ' + NC + '.'}});

    let run = 0, b = NB - 1;
    for (; b >= 0; b--){ if (run + hist[b] >= k) break; run += hist[b]; }
    if (b < 0) b = 0;
    push({bucket:b, line:9,
          sub:{zh:'每個 CTA 各自掃同一份數字，得到同一個答案 - 不需要再溝通',
               en:'every CTA scans the same numbers and reaches the same answer - no further communication'},
          msg:{zh:'掃描沒有分工：' + NC + ' 個 CTA 都把這 4 個數字重算一遍。<b>重複計算比再做一次同步便宜得多</b>，這是寫 GPU kernel 反直覺但很常見的一招。',
               en:'The scan is not divided up: all ' + NC + ' CTAs redo the same four numbers. <b>Recomputing is far cheaper than synchronising again</b> - a counter-intuitive but very common move in kernel writing.'}});
    k -= run;
    live = live.filter(i => ((keys[i] >> shifts[r]) & 3) === b);
    curBuf = (r + 1) % 3;
    push({line:8,
          sub:{zh:'CTA 0 清空 hist[' + ((r + 2) % 3) + ']，其他人開始累加 hist[' + curBuf + ']',
               en:'CTA 0 clears hist[' + ((r + 2) % 3) + '] while the others start filling hist[' + curBuf + ']'},
          msg:{zh:'為什麼要三份？只有一份的話，得等所有人讀完才能清，清完才能寫，等於每輪要兩次同步。兩份的話，清空和累加會撞在一起。三份剛好讓「正在寫的」「正在清的」「上一輪的」互不重疊，一次 barrier 就同時保住兩個不變條件。',
               en:'Why three? With one buffer you must wait for every reader before clearing, and for the clear before writing - two syncs per round. With two, the clear races the accumulate. Three keeps "being filled", "being cleared" and "last round’s" disjoint, so a single barrier upholds both invariants at once.'}});
  }
  const lane = {}; for (let c = 0; c < NC; c++) lane[c] = 'ok';
  push({lane:lane, line:9, sub:{zh:'剩下 ' + live.length + ' 個，答案定案',
                                en:live.length + ' left - the answer is pinned'},
        msg:cfg.outro});
  return F.list;
}

/* ============================================================= 6. ties ==== */
const CODE_TIE = [
'pivot, above, equal = radix_select(values, k)',
'need = k - len(above)                # how many slots the ties must fill',
'',
'# CTA c owns elements c, c + num_ctas, c + 2 * num_ctas, ...',
'lanes = [[i for i in equal if i % num_ctas == c] for c in range(num_ctas)]',
'',
'if policy == "race":                 # whoever writes first wins',
'    pool = [i for c in shuffle(range(num_ctas)) for i in lanes[c]]',
'elif policy == "det":                # replay in CTA order: stable per launch',
'    pool = [i for c in range(num_ctas) for i in lanes[c]]',
'elif policy == "small":              # tie_break=Small: lowest row index wins',
'    pool = sorted(equal)',
'elif policy == "large":              # tie_break=Large: highest row index wins',
'    pool = sorted(equal, reverse=True)',
'',
'return sorted(above + pool[:need])'
];

function tieFrames(cfg){
  const F = new Frames();
  const vals = cfg.vals.slice(), n = vals.length, k = cfg.k, NC = cfg.ctas;
  const pol = cfg.policy;
  const pivot = cfg.pivot;
  const above = [], equal = [];
  vals.forEach((v, i) => { if (v > pivot) above.push(i); else if (v === pivot) equal.push(i); });
  const need = k - above.length;
  const Y = 4.55;
  const mkLanes = (c) => { const L = []; for (let x = 0; x < c; x++) L.push(equal.filter(i => i % c === x)); return L; };
  let curCtas = NC, lanes = mkLanes(NC);

  const draw = (o) => {
    o = o || {};
    const sh = [], x0 = rowX(n);
    vals.forEach((v, i) => {
      const st = o.st && o.st[i] ? o.st[i] : (above.indexOf(i) >= 0 ? 'ok'
                : (equal.indexOf(i) >= 0 ? 'idle' : 'done'));
      sh.push(S.r(x0 + i * CW, Y, CW - .07, .52, st, String(v), {fs:.26, rx:.05}));
      sh.push(S.t(x0 + i * CW + (CW - .07) / 2, Y + .74, String(i), {c:COL.grey, fs:.22}));
    });
    sh.push(S.t(x0 - .22, Y + .30, {zh:'分數', en:'scores'}, {c:COL.tealL, fs:.28, anchor:'end'}));
    for (let c = 0; c < curCtas; c++){
      const y = 3.15 - c * .55;
      sh.push(S.t(1.15, y + .32, 'CTA ' + c, {c:COL.tealL, fs:.26, anchor:'end'}));
      sh.push(S.r(1.28, y, 3.30, .50, o.lane && o.lane[c] || 'idle', '', {rx:.05, o:.45}));
      lanes[c].forEach((i, j) => {
        const cls = o.pick && o.pick.indexOf(i) >= 0 ? 'hot' : 'soft';
        sh.push(S.r(1.42 + j * .70, y + .05, .58, .40, cls, 'i=' + i, {fs:.22, rx:.04}));
      });
    }
    sh.push(S.t(5.05, 3.47, o.rlab || {zh:'誰先寫進去，誰就進榜', en:'first writer wins the slot'},
                {c:COL.grey, fs:.27, anchor:'start'}));
    (o.res || []).forEach((r, j) => {
      sh.push(S.r(5.05, 2.95 - j * .52, 3.40, .44, r.cls, r.t, {fs:.25, rx:.05}));
    });
    sh.push(note(.55, o.sub || '', o.subc || COL.grey, .30));
    sh.push(note(5.95, cfg.head, COL.tealL, .34));
    return sh;
  };
  const panels = (extra) => [
    chips({zh:'嚴格大於 pivot', en:'strictly above the pivot'},
          [{t:above.length + ' of ' + k, cls:'ok'}]),
    chips({zh:'和 pivot 平手', en:'tied with the pivot'},
          [{t:String(equal.length), cls:'hot'}, {t:'need ' + need, cls:'act'}]),
    chips({zh:'這一次的答案', en:'this run’s answer'}, extra || [{t:'-', cls:''}])
  ];

  F.push({shapes:draw({sub:{zh:'k = ' + k + '，pivot = ' + pivot + '：' + above.length +
                             ' 個穩穩進榜，還缺 ' + need + ' 個要從 ' + equal.length + ' 個平手的裡面挑',
                            en:'k = ' + k + ', pivot = ' + pivot + ': ' + above.length +
                               ' are safely in, and ' + need + ' more must come out of ' + equal.length + ' tied values'}}),
          panels:panels(), view:VIEW, line:1, msg:cfg.intro});

  F.push({shapes:draw({sub:{zh:'平手的元素照 grid-stride 分給不同 CTA - 誰拿到哪一個，取決於 grid 開多大',
                            en:'the tied elements are dealt out by grid stride - which CTA sees which depends on how wide the grid is'}}),
          panels:panels(), view:VIEW, line:4,
          msg:{zh:'注意這一步：平手的元素是照 <b>i % num_ctas</b> 分的，所以 grid 開 2 個 CTA 和開 5 個 CTA 時，同一個元素會落在不同人手上。這就是為什麼 batch size 一改、答案就跟著變。',
               en:'Look closely: the tied elements are dealt out by <b>i % num_ctas</b>, so the same element lands in a different CTA when the grid is two wide versus five wide. That is why the answer moves when the batch size changes.'}});

  const runs = [];
  const rnd = lcg(cfg.seed || 11);
  const nRuns = pol === 'race' ? 4 : 3;
  for (let t = 0; t < nRuns; t++){
    let pool;
    let order = [];
    for (let c = 0; c < NC; c++) order.push(c);
    let ctas = NC;
    if (pol === 'race'){
      for (let i = order.length - 1; i > 0; i--){
        const j = Math.floor(rnd() * (i + 1)); const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
      }
      pool = [].concat.apply([], order.map(c => mkLanes(NC)[c]));
    } else if (pol === 'det'){
      ctas = [2, 3, 5][t];
      const ln = [];
      for (let c = 0; c < ctas; c++) ln.push(equal.filter(i => i % ctas === c));
      pool = [].concat.apply([], ln);
    } else if (pol === 'small'){
      ctas = [2, 3, 5][t];
      pool = equal.slice().sort((a, b) => a - b);
    } else {
      ctas = [2, 3, 5][t];
      pool = equal.slice().sort((a, b) => b - a);
    }
    const pick = pool.slice(0, need);
    const ans = above.concat(pick).sort((a, b) => a - b);
    runs.push({pick:pick, ans:ans, order:order, ctas:ctas});
    curCtas = ctas; lanes = mkLanes(ctas);
    const lane = {};
    if (pol === 'race') order.forEach((c, j) => { lane[c] = j === 0 ? 'hot' : 'soft'; });
    const res = runs.map((r, j) => ({cls:j === runs.length - 1 ? 'hot' : 'done',
      t:(pol === 'race' ? ('run ' + (j + 1)) : (r.ctas + ' CTAs')) + ':  idx ' + r.ans.join(', ')}));
    F.push({shapes:draw({lane:lane, pick:pick, res:res,
              rlab:pol === 'race' ? {zh:'同一份輸入，跑四次', en:'same input, four launches'}
                                  : {zh:'同一份輸入，三種 grid 大小', en:'same input, three grid widths'},
              sub:pol === 'race'
                ? {zh:'CTA 抵達順序：' + order.join(' -> ') + '（由排程決定，不由你決定）',
                   en:'arrival order: ' + order.join(' -> ') + ' - decided by the scheduler, not by you'}
                : {zh:'grid = ' + ctas + ' 個 CTA', en:'grid = ' + ctas + ' CTAs'}}),
            panels:panels([{t:'idx ' + ans.join(', '), cls:'hot'}]), view:VIEW,
            line:pol === 'race' ? 7 : (pol === 'det' ? 9 : (pol === 'small' ? 11 : 13)),
            msg:cfg.stepMsg(runs.length, order, ctas, ans)});
  }
  const distinct = {};
  runs.forEach(r => { distinct[r.ans.join(',')] = 1; });
  const nd = Object.keys(distinct).length;
  F.push({shapes:draw({res:runs.map((r, j) => ({cls:'ok',
            t:(pol === 'race' ? ('run ' + (j + 1)) : (r.ctas + ' CTAs')) + ':  idx ' + r.ans.join(', ')})),
            rlab:{zh:'結果', en:'results'},
            subc:nd > 1 ? COL.red : COL.tealL,
            sub:nd > 1 ? {zh:nd + ' 種不同的答案 - 分數完全一樣，選出來的 token 卻不一樣',
                          en:nd + ' different answers - identical scores, different tokens chosen'}
                       : {zh:'每一種 grid 大小都給出同一個答案', en:'every grid width gives the same answer'}}),
          panels:panels(), view:VIEW, line:15, msg:cfg.outro});
  return F.list;
}

/* ============================================================ DAY_META ==== */
const LB = (zh, en) => ({zh:zh, en:en});


/* ============================================== 6. guess-verify-refine ==== */
const CODE_GVR = [
'def gvr_topk(row, k, prev_idx, target, cap):',
'    p = [row[i] for i in prev_idx]        # P1: k loads, never n',
'    lo, hi = min(p), max(p)',
'    thr = mean(p)',
'    cnt_lo, cnt_hi = k + k // 4, 1        # the counts are GUESSED',
'    for it in range(MAX_REFINE_ITERS):    # P2: secant on the count',
'        f = (cnt_lo - target) / (cnt_lo - cnt_hi)',
'        f = clamp(f, 0.05, 0.95)',
'        thr = lo + (hi - lo) * f',
'        c = count_ge(row, thr)            # the only full pass',
'        if k <= c <= cap:',
'            break                         # inside the accept window',
'        if c > cap: lo, cnt_lo = thr, c   # too many -> raise the floor',
'        else:       hi, cnt_hi = thr, c   # too few  -> lower the ceiling',
'    cand = [v for v in row if v >= thr]   # P3: collect once',
'    return exact_topk(cand, k)            # P4: histogram snap'
];

function gvrGauss(r){ const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function gvrRow(seed, n){
  const r = lcg(seed), a = [];
  for (let i = 0; i < n; i++) a.push(+(gvrGauss(r) + 3 * Math.pow(r(), 8)).toFixed(2));
  return a;
}
const gvrCount = (a, t) => { let c = 0; for (let i = 0; i < a.length; i++) if (a[i] >= t) c++; return c; };

function gvrFrames(cfg){
  const F = new Frames();
  const N = 120, K = 12, CAP = 40, MAXIT = 8;
  const stepA = gvrRow(11, N);
  const rr = lcg(511);
  const row = stepA.map(v => +(v + gvrGauss(rr) * 0.08).toFixed(2));
  const prev = stepA.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, K).map(p => p[1]);
  const target = cfg.target;

  /* ---- geometry ---------------------------------------------------- */
  const X0 = .80, X1 = 9.00, BY = 3.10, BH = 1.45, TY = 4.45, TH = .50;
  const VMIN = Math.min.apply(null, row) - .15, VMAX = Math.max.apply(null, row) + .15;
  const vx = v => X0 + (v - VMIN) / (VMAX - VMIN) * (X1 - X0);
  const cx = c => X0 + Math.min(c, N) / N * (X1 - X0);
  const NB = 40, hist = new Array(NB).fill(0);
  row.forEach(v => { hist[Math.min(NB - 1, Math.floor((v - VMIN) / (VMAX - VMIN) * NB))]++; });
  const hmax = Math.max.apply(null, hist);
  const bw = (X1 - X0) / NB;
  const f2 = x => (x >= 0 ? '+' : '') + x.toFixed(2);

  const draw = (o) => {
    const sh = [];
    sh.push(note(.46, cfg.head, COL.tealL, .34));
    sh.push(note(.88, o.sub || '', COL.grey, .28));
    /* bracket [lo, hi] */
    if (o.lo != null)
      sh.push(S.r(vx(o.lo), BY - BH - .18, Math.max(vx(o.hi) - vx(o.lo), .03), BH + .18,
                  'act', '', {rx:.04, o:.16}));
    /* histogram of the row */
    for (let b = 0; b < NB; b++){
      if (!hist[b]) continue;
      const h = Math.max(BH * hist[b] / hmax, .05);
      const mid = VMIN + (b + .5) / NB * (VMAX - VMIN);
      const st = (o.thr != null && mid >= o.thr) ? 'ok' : 'soft';
      sh.push(S.r(X0 + b * bw + .03, BY - h, bw - .06, h, st, '', {rx:.03, o:.9}));
    }
    sh.push(S.e(X0 - .10, BY, X1 + .16, BY, {s:'idle', arrow:false, o:.5, w:.03}));

    /* the threshold */
    if (o.thr != null){
      sh.push(S.e(vx(o.thr), BY - BH - .30, vx(o.thr), BY + .16, {s:'hot', arrow:false, w:.055}));
      sh.push(S.t(Math.max(1.1, Math.min(8.7, vx(o.thr))), BY - BH - .44,
                  't = ' + f2(o.thr), {c:COL.orangeL, fs:.30}));
    }
    if (o.lo != null){
      sh.push(S.t(Math.max(vx(o.lo) - .10, 1.35), BY + .42, 'val_lo ' + f2(o.lo),
                  {c:COL.purpleL, fs:.25, anchor:'end'}));
      sh.push(S.t(Math.min(vx(o.hi) + .10, X1 - 1.30), BY + .74, 'val_hi ' + f2(o.hi),
                  {c:COL.purpleL, fs:.25, anchor:'start'}));
    }
    /* the count track */
    sh.push(S.t(X1, TY - .12, 'count(v >= t)', {c:COL.tealL, fs:.28, anchor:'end'}));
    sh.push(S.r(X0, TY, X1 - X0, TH, 'soft', '', {rx:.05, o:.75}));
    sh.push(S.r(cx(K), TY, cx(CAP) - cx(K), TH, 'ok', '', {rx:.05, o:.55}));
    sh.push(S.t((cx(K) + cx(CAP)) / 2, TY + TH + .70,
                {zh:'accept window', en:'accept window'}, {c:COL.tealL, fs:.26}));
    if (o.cnt != null){
      sh.push(S.r(cx(o.cnt) - .04, TY - .10, .08, TH + .20, 'hot', '', {rx:.02}));
      sh.push(S.t(Math.max(1.0, Math.min(8.8, cx(o.cnt))), TY - .48,
                  String(o.cnt), {c:COL.orangeL, fs:.32}));
    }
    [[0, '0'], [K, 'k = 12'], [CAP, 'cap = 40'], [N, 'n = 120']].forEach(t => {
      sh.push(S.t(cx(t[0]), TY + TH + .32, t[1], {c:COL.grey, fs:.25}));
    });
    if (o.verdict) sh.push(note(6.05, o.verdict, o.vc || COL.orangeL, .32));
    return sh;
  };
  const panels = (st) => [
    chips({zh:'下界 val_lo / cnt_lo', en:'floor val_lo / cnt_lo'},
          [{t:f2(st.lo) + '  ->  ' + st.clo + (st.guessed ? ' (guessed)' : ''), cls:'act'}]),
    chips({zh:'上界 val_hi / cnt_hi', en:'ceiling val_hi / cnt_hi'},
          [{t:f2(st.hi) + '  ->  ' + st.chi + (st.guessed ? ' (guessed)' : ''), cls:'act'}]),
    chips({zh:'割線比例 f', en:'secant fraction f'},
          [{t:st.f == null ? '-' : st.f.toFixed(3) + (st.pinned ? '  (clamped)' : ''),
            cls:st.pinned ? 'bad' : 'hot'}]),
    chips({zh:'kFTarget', en:'kFTarget'},
          [{t:String(target) + (target < K ? '  < k' : '  = k'), cls:target < K ? 'bad' : 'ok'}]),
    chips({zh:'第幾次 P2 迭代', en:'P2 iterations'}, [{t:st.it + ' / ' + MAXIT, cls:'hot'}])
  ];

  /* ---- P1 ---------------------------------------------------------- */
  const p = prev.map(i => row[i]);
  let lo, hi, thr;
  if (cfg.cold){
    lo = Math.min.apply(null, row); hi = Math.max.apply(null, row); thr = (lo + hi) / 2;
  } else {
    lo = Math.min.apply(null, p); hi = Math.max.apply(null, p);
    thr = p.reduce((a, b) => a + b, 0) / p.length;
  }
  let clo = K + (K >> 2), chi = 1, done = 0, it = 0;

  F.push({shapes:draw({sub:{zh:'120 個 logits，要前 12 名', en:'120 logits, top 12 wanted'}}),
          panels:[chips({zh:'這一列', en:'this row'}, [{t:'n = ' + N + ',  k = ' + K, cls:'ok'}])],
          view:VIEW, line:0, msg:cfg.intro});

  F.push({shapes:draw({lo:lo, hi:hi, thr:thr, cnt:null,
            sub:cfg.p1sub}),
          panels:panels({lo:lo, hi:hi, clo:clo, chi:chi, f:null, it:0, guessed:true}),
          view:VIEW, line:4,
          msg:cfg.p1 && cfg.p1 || {zh:'P1 不掃這一列。它去讀<b>上一步這顆 kernel 自己吐出來的 12 個 index</b>，在今天的 logits 上取 min / max / mean - 12 次讀取，不是 120 次。這是一個賭注：token t 的前 k 名，位置跟 token t-1 差不多。<b>cnt_lo 和 cnt_hi 是憑空填的</b>（k + k/4 和 1），從來沒有數過。',
                        en:'P1 never scans the row. It reads <b>the 12 indices this same kernel emitted on the previous step</b> and takes min / max / mean of today’s logits there - 12 loads, not 120. That is a bet: the top k of token t sits roughly where the top k of token t-1 sat. <b>cnt_lo and cnt_hi are fabricated</b> (k + k/4 and 1) - neither was ever counted.'}});

  let c = gvrCount(row, thr);
  const seedVerdict = (c >= K && c <= CAP) ? {zh:'一次就中', en:'in window already'}
                    : (c > CAP ? {zh:'太多了 -> 把下界抬到這裡', en:'too many -> raise the floor to here'}
                               : {zh:'太少了 -> 把上界壓到這裡', en:'too few -> lower the ceiling to here'});
  F.push({shapes:draw({lo:lo, hi:hi, thr:thr, cnt:c, verdict:seedVerdict,
            sub:{zh:'數一次，看落在哪', en:'count once, see where it lands'}}),
          panels:panels({lo:lo, hi:hi, clo:clo, chi:chi, f:null, it:0, guessed:true}),
          view:VIEW, line:9,
          msg:cfg.seed(c)});
  if (c >= K && c <= CAP) done = 1;
  else if (c > CAP){ lo = thr; clo = c; }
  else { hi = thr; chi = c; }

  /* ---- P2 ---------------------------------------------------------- */
  while (it < MAXIT && done === 0){
    const rng = hi - lo;
    let f = null, pinned = false, nv;
    if (clo > chi && rng > 1e-10){
      f = (clo - target) / (clo - chi);
      const raw = f;
      f = Math.max(.05, Math.min(.95, f));
      if (it === 0) f = Math.min(f, .5);
      pinned = (raw !== f);
      nv = lo + rng * f;
    } else nv = (lo + hi) / 2;
    if (nv <= lo) nv = lo + rng * .05;
    if (nv >= hi) nv = hi - rng * .05;
    thr = nv; c = gvrCount(row, thr);
    let verdict, vc = COL.orangeL;
    if (c >= K && c <= CAP){ done = 1; verdict = {zh:'落在 accept window 裡 - 收工', en:'inside the accept window - stop'}; vc = COL.tealL; }
    else if (c > CAP){ verdict = {zh:'還是太多 -> 下界抬到 t', en:'still too many -> floor moves up to t'}; lo = thr; clo = c; }
    else { verdict = {zh:'太少了 -> 上界壓到 t', en:'too few -> ceiling moves down to t'}; hi = thr; chi = c; }
    it++;
    F.push({shapes:draw({lo:Math.min(lo, thr), hi:Math.max(hi, thr), thr:thr, cnt:c, verdict:verdict, vc:vc,
              sub:{zh:'P2 第 ' + it + ' 次：只數，不排序', en:'P2 iteration ' + it + ': counting only, no sorting'}}),
            panels:panels({lo:lo, hi:hi, clo:clo, chi:chi, f:f, pinned:pinned, it:it}),
            view:VIEW, line:6,
            msg:cfg.step(it, f, pinned, thr, c, done)});
  }

  const cand = gvrCount(row, done === 0 ? lo : thr);
  F.push({shapes:draw({lo:lo, hi:hi, thr:(done === 0 ? lo : thr), cnt:cand,
            verdict:done === 0 ? {zh:'放棄，退回 val_lo', en:'gave up, fell back to val_lo'}
                               : {zh:'P3 收集 ' + cand + ' 個候選 -> P4 精確取前 12', en:'P3 collects ' + cand + ' candidates -> P4 snaps to the exact 12'},
            vc:done === 0 ? COL.red : COL.tealL,
            sub:done === 0 ? {zh:'跑滿 ' + MAXIT + ' 次還沒進 window', en:MAXIT + ' iterations and never inside the window'}
                           : {zh:'共 ' + it + ' 次 P2 迭代', en:it + ' P2 iterations in total'}}),
          panels:panels({lo:lo, hi:hi, clo:clo, chi:chi, f:null, it:it}),
          view:VIEW, line:done === 0 ? 13 : 14,
          msg:cfg.outro(it, cand, done)});
  return F.list;
}

/* shared narration for the three GVR variants ---------------------------- */
const gvrSeedMsg = (c, cold) => c > 40
  ? {zh:'數出來 ' + c + ' 個，超過 cap = 40，所以這個 threshold <b>太低</b>，val_lo 抬到這裡 - 我們手上第一次有了「真的數過」的數字。',
     en:'The count is ' + c + ', above cap = 40, so this threshold is <b>too low</b> and val_lo moves up to here - the first number in hand that was actually measured.'}
  : (c < 12
    ? {zh:'只有 ' + c + ' 個，比 k = 12 還少，threshold <b>太高</b>，val_hi 壓下來。注意這一步做的事：<b>不排序、不搬資料，只數數</b> - 數數是可以完美平行的。',
       en:'Only ' + c + ', fewer than k = 12, so the threshold is <b>too high</b> and val_hi comes down. Note what this step does: <b>no sorting, no data movement, just counting</b> - and counting parallelises perfectly.'}
    : {zh:'' + c + ' 個，直接落在 [12, 40] 裡。' + (cold ? '中點猜中純粹是運氣。' : 'P1 的猜測一次就中，這一列連一次修正都不用。'),
       en:'' + c + ', already inside [12, 40]. ' + (cold ? 'The midpoint got lucky.' : 'P1’s guess landed first try, so this row needs no refinement at all.')});

function gvrStepMsg(it, f, pinned, c, done, target){
  if (done)
    return {zh:'第 ' + it + ' 次就進了 window，' + c + ' 個候選。P3 只要把這 ' + c + ' 個收進 shared memory，P4 在裡面做一次精確的 histogram 就結束 - 整列 120 個值只被完整掃過 ' + it + ' 次多一點，而且每一次都只是加法。',
            en:'Iteration ' + it + ' lands inside the window with ' + c + ' candidates. P3 collects just those ' + c + ' into shared memory, P4 runs one exact histogram over them, done - the full row was swept a little over ' + it + ' times, and every sweep was nothing but additions.'};
  if (pinned)
    return {zh:'f 被夾在 0.95 - 問題就在這裡。<b>kFTarget = ' + target + ' 比 k = 12 還小</b>，割線一直瞄向一個「window 一定會退貨」的數量；瞄不到的目標配上幾乎重合的兩個端點，每次只能把區間削掉 5%。第 ' + it + ' 次 count = ' + c + '，還是在外面。',
            en:'f is pinned at its 0.95 ceiling, and that is the whole bug. <b>kFTarget = ' + target + ' sits below k = 12</b>, so the secant keeps aiming at a count the accept window is required to reject; an unreachable target plus two nearly-coincident endpoints means each pass shaves only 5% off the bracket. Iteration ' + it + ': count = ' + c + ', still outside.'};
  return {zh:'f = ' + f.toFixed(3) + '：割線假設「數量隨 threshold 線性下降」，於是直接解出下一個猜測，而不是像 binary search 那樣永遠取中點。數出來 ' + c + ' 個 - ' + (c > 40 ? '還是太多，下界跟上來。' : '太少了，上界壓下來。') + '每一輪區間都變窄，而且是往正確的方向窄。',
          en:'f = ' + f.toFixed(3) + '. The secant assumes the count falls linearly with the threshold and solves for the next guess outright, instead of always halving like a binary search. The count comes back as ' + c + ' - ' + (c > 40 ? 'still too many, so the floor follows it up. ' : 'too few, so the ceiling comes down. ') + 'The bracket narrows every round, and narrows in the right direction.'};
}

function gvrOutroMsg(it, cand, done, target){
  if (!done)
    return {zh:'跑滿 8 次還是沒進 window，kernel 只能退回 val_lo 硬收 ' + cand + ' 個候選。（注意最後這個 count 其實落在 window 裡 - 迴圈從來沒有<b>測試</b>過 val_lo 這個端點，它只試區間內部的點）。upstream 的註解把這件事叫 <b>upper-clamp saturation</b>，修法只有一個常數：把 kFTarget 設成 k。真實 kernel 上這一改省下 1.5-2.2 倍的 P2 迭代，selection.py 的 200 列量測是 1.8 倍。<b>整段 kernel 最貴的東西，是一個瞄錯地方的目標值。</b>',
            en:'Eight iterations and never inside the window, so the kernel falls back to val_lo and swallows ' + cand + ' candidates. (note that this final count does land inside the window - the loop never <b>tested</b> the endpoint val_lo, only points strictly inside the bracket). Upstream’s own comment calls this <b>upper-clamp saturation</b>, and the fix is one constant: set kFTarget to k. On the real kernel that is worth 1.5-2.2x fewer P2 iterations; the 200-row sweep in selection.py measures 1.8x. <b>The most expensive thing in this kernel was a target aimed at the wrong number.</b>'};
  return {zh:'' + it + ' 次迭代、' + cand + ' 個候選就收工。值得停下來看一眼這件事有多奇怪：<b>我們從頭到尾沒有比較過任何兩個元素的大小</b>，也沒有砍過任何一個 bit。radix select 一輪一輪縮小值域，GVR 是直接解方程式猜出值域 - 賭的是「這一列的分布跟上一步差不多」，而在 decode 裡這個賭注幾乎每次都贏。',
          en:'Done in ' + it + ' iterations with ' + cand + ' candidates. It is worth pausing on how strange that is: <b>we never compared two elements against each other</b>, and never peeled off a single bit. Radix select shrinks the value range one digit at a time; GVR solves for it. The bet is that this row’s distribution looks like the last row’s - and inside decode, that bet almost always pays.'};
}

/* ================================================== 7. DSMEM cluster ===== */
const CODE_DSM = [
'# every CTA in the cluster counts its own slice',
'my = 0',
'for i in range(rank, n, cluster_size):',
'    if row[i] >= thr: my += 1',
'',
'smem[rank] = my                  # write to MY shared memory',
'cluster.sync()                   # barrier inside the cluster only',
'',
'total = 0',
'for peer in range(cluster_size):',
'    ptr = mapa(smem, peer)       # peer CTA’s shared address',
'    total += ld_shared_cluster(ptr)   # no global memory involved'
];

function dsmFrames(cfg){
  const F = new Frames();
  const CS = cfg.cs, N = 120;
  const stepA = gvrRow(11, N);
  const rr = lcg(511);
  const row = stepA.map(v => +(v + gvrGauss(rr) * 0.08).toFixed(2));
  const thr = row.slice().sort((a, b) => b - a)[33];
  const part = [];
  for (let r = 0; r < CS; r++){ let c = 0; for (let i = r; i < N; i += CS) if (row[i] >= thr) c++; part.push(c); }
  const TOTAL = part.reduce((a, b) => a + b, 0);

  const X0 = .80, X1 = 9.00, SY = 1.30, SH = .74, MY = 3.00, MH = .58, TY = 4.55, TH = .62;
  const SL = (CS === 1 ? 4 : CS);            /* keep the same slot width in every variant */
  const bw = (X1 - X0) / SL, cw = Math.min(bw - .16, 2.1);
  const bx = i => X0 + i * bw + (bw - cw) / 2;

  const draw = (o) => {
    const sh = [];
    sh.push(note(.44, cfg.head, COL.tealL, .34));
    sh.push(note(.84, o.sub || '', COL.grey, .28));
    for (let r = 0; r < CS; r++){
      sh.push(S.r(bx(r), SY, cw, SH, o.cta === r ? 'hot' : (o.counted > r ? 'ok' : 'idle'),
                  'CTA ' + r, {fs:.30, rx:.06}));
      sh.push(S.t(bx(r) + cw / 2, SY + SH + .34,
                  o.counted > r || o.cta === r
                    ? {zh:'我這片：' + part[r], en:'my slice: ' + part[r]}
                    : {zh:'每 ' + CS + ' 個取 1 個', en:'every ' + CS + 'th element'},
                  {c:o.cta === r ? COL.orangeL : COL.grey, fs:.26}));
      /* shared memory slot */
      if (o.smem){
        sh.push(S.r(bx(r) + cw / 2 - .55, MY, 1.10, MH, o.readPeer === r ? 'act' : 'ok',
                    String(part[r]), {fs:.30, rx:.06}));
        sh.push(S.t(bx(r) + cw / 2, MY - .18, 'smem[' + r + ']', {c:COL.purpleL, fs:.25}));
      }
      /* running total per CTA */
      if (o.tot){
        sh.push(S.r(bx(r) + cw / 2 - .55, TY, 1.10, TH,
                    o.tot[r] === TOTAL && o.doneAll ? 'ok' : 'hot',
                    String(o.tot[r]), {fs:.30, rx:.06}));
        sh.push(S.t(bx(r) + cw / 2, TY + TH + .34, 'total', {c:COL.grey, fs:.25}));
      }
      /* the mapa read arrows */
      if (o.readPeer != null && o.tot)
        sh.push(S.e(bx(o.readPeer) + cw / 2, MY + MH + .04, bx(r) + cw / 2, TY - .04,
                    {s:'act', w:.045, o:.85}));
    }
    if (CS === 1)
      for (let g = 1; g < 4; g++){
        sh.push(S.r(bx(g), SY, cw, SH, 'ghost', '', {rx:.06}));
        sh.push(S.t(bx(g) + cw / 2, SY + SH * .62, {zh:'沒有這個 CTA', en:'no such CTA'},
                    {c:COL.grey, fs:.26}));
      }
    if (o.bar)
      sh.push(S.r(X0, MY + MH + .40, X1 - X0, .16, 'act', '', {rx:.06, o:.45}));
    if (o.foot) sh.push(note(6.05, o.foot, o.fc || COL.tealL, .30));
    return sh;
  };
  const panels = (o) => [
    chips({zh:'cluster 大小', en:'cluster size'},
          [{t:String(CS) + (CS === 1 ? '  (no cluster)' : '  CTAs, same GPC'), cls:CS === 1 ? 'bad' : 'ok'}]),
    chips({zh:'每個 CTA 掃幾個元素', en:'elements scanned per CTA'},
          [{t:String(Math.ceil(N / CS)), cls:'hot'}]),
    chips({zh:'各自數到的數量', en:'partial counts'},
          part.map((v, i) => ({t:String(v), cls:o.counted > i ? 'ok' : 'empty'}))),
    chips({zh:'global memory atomics', en:'global memory atomics'}, [{t:'0', cls:'ok'}])
  ];

  F.push({shapes:draw({sub:{zh:'一列 120 個值，門檻已經定了', en:'one row of 120 values, threshold already chosen'},
            foot:{zh:'要回答的只有一個問題：有幾個 v >= t ?', en:'one question to answer: how many v >= t ?'}}),
          panels:panels({counted:0}), view:VIEW, line:0, msg:cfg.intro});

  for (let r = 0; r < CS; r++){
    F.push({shapes:draw({cta:r, counted:r,
              sub:{zh:'CTA ' + r + ' 用 stride ' + CS + ' 掃自己的那一份', en:'CTA ' + r + ' walks its share with stride ' + CS}}),
            panels:panels({counted:r + 1}), view:VIEW, line:3,
            msg:cfg.count(r, part[r], Math.ceil(N / CS))});
  }

  F.push({shapes:draw({counted:CS, smem:true,
            sub:{zh:'每個 CTA 把自己的數字寫進「自己的」shared memory', en:'each CTA writes its number into its OWN shared memory'}}),
          panels:panels({counted:CS}), view:VIEW, line:5, msg:cfg.write(part)});

  F.push({shapes:draw({counted:CS, smem:true, bar:true,
            sub:{zh:'cluster.sync()：只有這 ' + CS + ' 個 CTA 要等彼此', en:'cluster.sync(): only these ' + CS + ' CTAs wait for each other'}}),
          panels:panels({counted:CS}), view:VIEW, line:6, msg:cfg.sync(CS)});

  const tot = new Array(CS).fill(0);
  for (let p = 0; p < CS; p++){
    for (let r = 0; r < CS; r++) tot[r] += part[p];
    F.push({shapes:draw({counted:CS, smem:true, readPeer:p, tot:tot.slice(),
              doneAll:(p === CS - 1),
              sub:{zh:'每個 CTA 都去讀 smem[' + p + ']', en:'every CTA reads smem[' + p + ']'}}),
            panels:panels({counted:CS}), view:VIEW, line:11, msg:cfg.read(p, part[p], tot[0], CS)});
  }

  F.push({shapes:draw({counted:CS, smem:true, tot:tot.slice(), doneAll:true,
            sub:CS === 1 ? {zh:'一個 CTA、一個答案，沒有人要對帳', en:'one CTA, one answer, nobody to reconcile with'}
                         : {zh:'每個 CTA 手上都是同一個 ' + TOTAL, en:'every CTA now holds the same ' + TOTAL},
            foot:CS === 1 ? {zh:'count = ' + TOTAL + '，但這一列的延遲＝這一個 CTA 的延遲', en:'count = ' + TOTAL + ', but this row’s latency is this one CTA’s latency'}
                          : {zh:'count = ' + TOTAL + '，全程沒有碰過 global memory', en:'count = ' + TOTAL + ', and global memory was never touched'}}),
          panels:panels({counted:CS}), view:VIEW, line:11, msg:cfg.outro(TOTAL, CS)});
  return F.list;
}

const DAY_META = {
title:LB('選擇問題與 Top-K', 'Selection and top-k'),
sub:LB('要前 k 名，從來就不需要先排序。把這件事搬到 GPU 上，難的不是演算法，是讓幾百個 block 對同一個答案講好話。',
      'You never had to sort to get the top k. Moving it onto a GPU, the hard part is not the algorithm - it is getting hundreds of blocks to agree on one answer.'),
tabs:[
{
  id:'qs', label:LB('quickselect', 'quickselect'),
  stage:LB('三路 partition：每一輪只往一邊遞迴', 'three-way partition: recurse into one side only'),
  view:VIEW,
  variants:[LB('一般輸入', 'ordinary input'), LB('全部相同', 'all equal')],
  idea:LB('排序把 n 個元素之間的關係<b>全部</b>算出來，但你只問了「前 k 名是誰」。quickselect 每輪丟掉一半，n + n/2 + n/4 + … = 2n，是線性的。三路 partition 讓一整排相同的值一次退場。',
         'Sorting computes the full ordering of n elements when you only asked who the top k are. Quickselect throws half away each round: n + n/2 + n/4 + ... = 2n, which is linear. The three-way partition retires a whole run of equal values at once.'),
  legend:['hot', 'act', 'ok', 'soft', 'done'],
  code:CODE_QS,
  build:(v) => v === 1
    ? qsFrames({data:[5,5,5,5,5,5,5,5,5,5], k:4, seed:3,
        head:LB('全部相同的輸入', 'every element is identical'),
        intro:LB('十個一模一樣的值，k = 4。用兩路 partition 的話這是最壞情況，O(n²)；三路 partition 一輪就結束。',
                'Ten identical values, k = 4. For a two-way partition this is the quadratic worst case; the three-way version finishes in one round.'),
        outro:LB('一輪之內全部落進「等於 pivot」那一區，直接結束。但也請注意：答案是「任意四個 5」- <b>誰進榜完全沒有定義</b>。在 CPU 上沒人在意，在 GPU 上這件事會變成 bug 回報。',
                'Everything lands in the equal region and the search ends immediately. But notice what the answer is: "any four of the fives" - <b>which ones is simply undefined</b>. Nobody cares on a CPU. On a GPU it turns into a bug report.')})
    : qsFrames({data:[7,2,9,4,1,8,3,6,5,0,9,4], k:4, seed:7,
        head:LB('quickselect：找前 4 大', 'quickselect: the top 4'),
        intro:LB('十二個數字，要前四大。先問一個比較笨的問題：如果先排序，我們會算出所有 12 個數字之間的順序 - 但題目只問了四個名額。',
                'Twelve numbers, top four wanted. Start with the dumb question: sorting works out the order of all twelve, when the question only has four slots in it.'),
        outro:LB('前四大找出來了，而且陣列其他部分還是亂的 - 我們<b>只做了問題需要的工作</b>。實測在 20 萬個浮點數上：完整排序要 6,439,409 次比較，quickselect 只要 244,798 次。',
                'The top four are found and the rest of the array is still unsorted - we did <b>only the work the question needed</b>. Measured on 200,000 floats: a full sort costs 6,439,409 comparisons, quickselect 244,798.')})
},
{
  id:'radix', label:LB('radix select', 'radix select'),
  stage:LB('用計數器縮小範圍，完全不比較元素', 'narrowing with counters, never comparing elements'),
  view:VIEW,
  idea:LB('把浮點數轉成<b>保序的整數 key</b>之後，就可以一次處理幾個 bit：數一數每個 bin 有幾個，從最高的 bin 往下累加，跨過 k 的那個 bin 就是答案所在。每一輪搬動的是 256 個計數器，不是 n 個元素 - 這是它能上 GPU 的原因。',
         'Turn floats into an <b>order-preserving integer key</b> and you can decide several bits at a time: count how many fall in each bin, accumulate from the top, and the bin that straddles k is where the answer lives. A round moves 256 counters, not n elements - which is exactly why it belongs on a GPU.'),
  legend:['hot', 'ok', 'soft', 'done'],
  code:CODE_RX,
  build:() => rxFrames({keys:[201,17,88,240,63,199,142,7,88,255,120,33], k:3,
    head:LB('radix select：第 3 大是誰', 'radix select: who is third'),
    intro:LB('這裡把 key 縮成 8 bit、每輪 2 bit 好看清楚；真正的 kernel 是 32 bit 的 key、每輪 8 bit、256 個 bin。注意接下來<b>沒有任何兩個元素被拿來比較過</b>。',
            'Here the key is 8 bits and each round decides 2, so it fits on screen; a real kernel uses a 32-bit key, 8 bits a round, 256 bins. Watch for what does not happen: <b>no two elements are ever compared</b>.'),
    outro:LB('四輪、每輪 4 個計數器就定案了。這個結構之所以適合 GPU，是因為「數數」可以完全平行 - 每個 thread 數自己那一份，最後 atomic 加起來就好。',
            'Four rounds of four counters and it is decided. The reason this shape suits a GPU is that counting is embarrassingly parallel: every thread counts its own slice and folds the result in with an atomic add.')})
},
{
  id:'filter', label:LB('一個 block 一列', 'one block per row'),
  stage:LB('粗直方圖 + v_lo/v_hi：兩趟就把整列壓進 shared memory',
          'coarse histogram plus v_lo/v_hi: two passes squeeze a row onto chip'),
  view:VIEW,
  variants:[LB('分散的分數', 'well-spread scores'), LB('擠在一起的分數', 'scores piled up')],
  idea:LB('如果 k 不大，根本不必開整個 grid：一個 block 掃兩趟就好。第一趟用粗 key 做直方圖找出邊界 bin，把它的上下界投影回浮點數變成 v_lo / v_hi；第二趟只用兩次浮點比較，就把整列分成「一定進榜」「待定」「淘汰」三類，只有中間那一類需要佔用 shared memory。',
         'When k is small you do not need the whole grid: one block and two passes will do. Pass one histograms a coarse key to find the straddling bin, then projects its edges back into floats as v_lo / v_hi. Pass two needs just two float compares per element to split the row into certainly-in, undecided and out - and only the middle class costs shared memory.'),
  legend:['hot', 'ok', 'done', 'idle'],
  code:CODE_FT,
  build:(v) => v === 1
    ? ftFrames({row:[93,97,95,90,99,91,94,96,92,98,93,95,97,91], k:5, cap:4,
        head:LB('粗 key 太粗的時候', 'when the coarse key is too coarse'),
        intro:LB('同樣的 kernel，換一列分數：全部擠在同一個 bin 裡。這正是 attention logits 的日常 - fp32 的最高位元組是符號加 7 個指數 bit，同一個量級的分數會全部落進同一格。',
                'Same kernel, different row: every score falls in one bin. This is the everyday shape of attention logits - the top byte of an fp32 is the sign plus seven exponent bits, so scores of the same magnitude all pile into one slot.'),
        outro:LB('待定的元素比 shared memory 裝得下的還多，這顆 kernel 只能放棄，改用整個 grid 的版本。<b>解法不是加大 shared memory，是換一把更細的尺</b>：先把值收窄成 fp16 再取 key。實測同一批 20,000 個值，fp16 的粗 key 留下 98 個候選，bf16 留下 3,197 個 - 差了 32 倍，而差別只在尾數多了幾個 bit。',
                'More candidates survive than shared memory can hold, so this kernel bails out and the grid-wide version takes over. <b>The fix is not a bigger scratchpad, it is a finer ruler</b>: narrow the values to fp16 first, then take the key. Measured on the same 20,000 values, an fp16 coarse key leaves 98 candidates and a bf16 one leaves 3,197 - a factor of 32, decided by a handful of mantissa bits.')})
    : ftFrames({row:[93,12,44,97,7,61,95,33,86,21,4,72,38,66], k:5, cap:4,
        head:LB('filtered top-k：一個 block 就夠', 'filtered top-k: one block is enough'),
        intro:LB('十四個分數，要前五名，而且晶片上只有 4 個空位。聽起來不可能 - 但大部分元素根本不需要被記住。',
                'Fourteen scores, top five wanted, and only four slots of on-chip space. That sounds impossible - but most elements never need to be remembered at all.'),
        outro:LB('整列只有 1 個元素需要進 shared memory。這顆 kernel 不需要 grid barrier、不需要第二次 launch、CUDA graph 抓得住它 - 代價是 k 和候選數量都必須塞得進晶片上那 16 KB。',
                'Exactly one element of the row needed on-chip storage. This kernel needs no grid barrier, no second launch, and a CUDA graph can capture it - at the price that both k and the surviving candidates must fit in that 16 KB.')})
},
{
  id:'cta', label:LB('多個 CTA', 'many CTAs'),
  stage:LB('grid barrier：計數器不歸零，三份直方圖輪流用',
          'a grid barrier: the counter never resets, three histograms take turns'),
  view:VIEW,
  idea:LB('一列太長，一個 block 裝不下的時候，就得讓整個 grid 一起做同一列。每個 CTA 只看自己那一份、算出<b>部分</b>直方圖，然後所有人在 grid barrier 上會合。barrier 用一個從不歸零的計數器，每個 CTA 自己記住 phase，等 counter >= (phase+1) x num_ctas。',
         'When a row is too long for one block, the whole grid has to work on it together. Each CTA sees only its slice and produces a <b>partial</b> histogram, then everyone meets at a grid barrier. The barrier is one counter that is never reset: each CTA remembers its own phase and waits for counter >= (phase + 1) x num_ctas.'),
  legend:['hot', 'ok', 'act', 'done', 'idle'],
  code:CODE_CTA,
  build:() => ctaFrames({ctas:4, keys:[41,7,58,23,62,15,39,50,6,44,31,60], k:3,
    shifts:[4, 2, 0], mode:'ok', slow:3,
    head:LB('4 個 CTA，同一列', 'four CTAs, one row'),
    intro:LB('12 個 6-bit 的 key 分給 4 個 CTA，每輪決定 2 個 bit。從這裡開始，沒有任何一個 CTA 看得到整列 - 所有的正確性都靠同步撐著。',
            'Twelve 6-bit keys dealt across four CTAs, two bits decided per round. From here on no single CTA can see the whole row, and every correctness property rests on synchronisation.'),
    outro:LB('三輪、三次會合就定案了。真正貴的不是計算 - 是那幾次 barrier，以及每次 barrier 都要等最慢的那個 CTA。所以 kernel 只在「一個 block 真的裝不下」的時候才用這條路。',
            'Three rounds, three meetings, done. The expensive part is not the arithmetic - it is the barriers, and the fact that each one waits for the slowest CTA. Which is why this path is taken only when a single block genuinely cannot cope.')})
},
{
  id:'dead', label:LB('#3610 死鎖', 'the #3610 deadlock'),
  stage:LB('誰負責把計數器歸零？', 'who resets the counter?'),
  view:VIEW,
  variants:[LB('壞掉的版本', 'the broken version'), LB('修好的版本', 'the fix')],
  idea:LB('這是 flashinfer 真實的 issue #3610。barrier 通過之後總得有人把計數器清乾淨，好讓下一輪重新數 - 但如果「先出關的人」去清，還在自旋的 CTA 就會讀到 0，永遠等不到 target。修法是讓計數器單調遞增、每個 CTA 自己記 phase；或者把離開 barrier 當成一次選舉，只有最後一個看到 old+1 == target 的人負責善後。',
         'This is flashinfer issue #3610, for real. After a barrier releases, somebody has to clear the counter so the next round can start from zero - but if the <i>first</i> CTA out does the clearing, the CTAs still spinning read 0 and never reach their target. The fix is to let the counter climb monotonically with a per-CTA phase, or to treat leaving the barrier as an election in which exactly one CTA sees old + 1 == target and does the cleanup.'),
  legend:['hot', 'bad', 'ok', 'act', 'idle'],
  code:CODE_CTA,
  build:(v) => v === 1
    ? ctaFrames({ctas:4, keys:[41,7,58,23,62,15,39,50,6,44,31,60], k:3,
        shifts:[4, 2, 0], mode:'ok', slow:3,
        head:LB('修好之後：計數器只往上加', 'fixed: the counter only ever climbs'),
        intro:LB('同樣 4 個 CTA、同樣的資料，只改一件事：<b>不歸零</b>。第 r 輪等的是 (r+1) x 4，第 r+1 輪等的是 (r+2) x 4，沒有人需要清任何東西。',
                'Same four CTAs, same data, one change: <b>never reset</b>. Round r waits for (r+1) x 4, round r+1 for (r+2) x 4, and nothing needs clearing.'),
        outro:LB('沒有清空、就沒有清錯的機會。這類 bug 的教訓很一致：<b>能用單調遞增的狀態，就不要用需要重設的狀態</b> - 重設本身就是一個要同步的臨界區。',
                'Nothing is cleared, so nothing can be cleared at the wrong moment. The lesson generalises: <b>prefer monotonic state over state that must be reset</b> - the reset is itself a critical section that needs synchronising.')})
    : ctaFrames({ctas:4, keys:[41,7,58,23,62,15,39,50,6,44,31,60], k:3,
        shifts:[4, 2, 0], mode:'broken', breakRound:0, slow:3,
        head:LB('先出關的人順手把計數器歸零', 'the first CTA out clears the counter'),
        intro:LB('這個版本看起來完全合理：barrier 過了，總要有人把計數器重設，好讓下一輪重新數。問題只在於「誰」。',
                'This version looks entirely reasonable: the barrier released, so somebody should reset the counter for the next round. The only question is who.'),
        outro:LB('', '')})
},
{
  id:'ties', label:LB('平手', 'ties'),
  stage:LB('同樣的輸入，不同的答案', 'identical input, different answers'),
  view:VIEW,
  variants:[LB('競爭', 'race'), LB('deterministic', 'deterministic'), LB('tie_break=Small', 'tie_break=Small'), LB('tie_break=Large', 'tie_break=Large')],
  idea:LB('「前 k 大」在有平手的時候<b>不是唯一的</b>。kernel 提供兩個互相獨立的旋鈕：deterministic 管的是<b>輸出順序</b>，tie_break 管的是<b>誰進榜</b>。只開前者，同一次 launch 內結果穩定，但換個 batch size 就變了 - 因為 grid-stride 把 num_ctas 烙進了「誰看到哪個平手元素」。',
         'With ties, "the top k" is <b>not unique</b>. The kernels expose two orthogonal knobs: deterministic controls the <b>order of the output</b>, tie_break controls <b>membership</b>. Turn on only the first and results are stable within a launch shape but move when the batch size changes - because the grid-stride loop bakes num_ctas into which CTA sees which tied element.'),
  legend:['hot', 'ok', 'soft', 'done'],
  code:CODE_TIE,
  build:(v) => {
    const base = {vals:[9,4,7,4,2,4,8,4,1,4], k:5, pivot:4, ctas:3, seed:11};
    if (v === 1) return tieFrames(Object.assign({}, base, {policy:'det',
      head:LB('deterministic：同一次 launch 內穩定', 'deterministic: stable within one launch shape'),
      intro:LB('這次固定照 CTA 編號重播，不管誰先到。同一份輸入跑一百次都一樣 - 那還有什麼問題？',
              'Now we replay in CTA order regardless of who arrives first. A hundred runs give the same answer - so what is left to go wrong?'),
      stepMsg:(t, o, c, a) => LB('grid 開 ' + c + ' 個 CTA：平手元素被分成 ' + c + ' 條 lane，重播順序跟著改，答案是 idx ' + a.join(', ') + '。',
                                'With a grid of ' + c + ' CTAs the tied elements are dealt into ' + c + ' lanes, so the replay order changes with it. The answer is idx ' + a.join(', ') + '.'),
      outro:LB('三種 grid 大小、三個不同的穩定答案。<b>deterministic 保證的是「同一種 launch 形狀下可重現」，不是「同一份輸入下唯一」</b>。使用者改一下 batch size、輸出就變了，而這通常是最難查的那種 bug。',
              'Three grid widths, three different stable answers. <b>deterministic promises reproducibility for a fixed launch shape, not uniqueness for a fixed input</b>. A user changes the batch size, the output changes, and that is the hardest kind of bug to chase.')}));
    if (v === 2) return tieFrames(Object.assign({}, base, {policy:'small',
      head:LB('tie_break=Small：index 小的贏', 'tie_break=Small: the lowest index wins'),
      intro:LB('要真正唯一，規則必須跟<b>列本身</b>有關，不能跟 grid 有關。最簡單的規則：平手時 index 小的優先。',
              'For genuine uniqueness the rule has to be a property of <b>the row</b>, not of the grid. The simplest such rule: on a tie, the lower index wins.'),
      stepMsg:(t, o, c, a) => LB('grid 開 ' + c + ' 個 CTA，答案還是 idx ' + a.join(', ') + '。規則不看誰先到、也不看誰拿到哪一格。',
                                'Grid of ' + c + ' CTAs, and the answer is still idx ' + a.join(', ') + '. The rule does not care who arrived first, or who was dealt which slot.'),
      outro:LB('每一種 grid 大小都給同一個答案。代價是 CTA 不能只回報「我有幾個平手的」，還得回報 index，然後在 reduce 時比大小 - 這也是 filtered kernel 不支援 tie_break 的原因：它沒有整列的全域視角。',
              'Every grid width agrees. The price is that a CTA can no longer report just "I have three ties" - it has to report indices and compare them during the reduce. That is also why the filtered kernel does not offer tie_break: it has no row-global view to compare against.')}));
    if (v === 3) return tieFrames(Object.assign({}, base, {policy:'large',
      head:LB('tie_break=Large：index 大的贏', 'tie_break=Large: the highest index wins'),
      intro:LB('同一個機制，反過來的規則。放在一起看才看得出重點：<b>兩種都對</b>。',
              'Same mechanism, opposite rule. Seeing them side by side is the point: <b>both are correct</b>.'),
      stepMsg:(t, o, c, a) => LB('答案 idx ' + a.join(', ') + '，一樣不受 grid 大小影響。',
                                'The answer is idx ' + a.join(', ') + ', again independent of the grid.'),
      outro:LB('Small 和 Large 給出不同的答案，但兩個都是對的 - 因為「前 k 大」這個問題本身在平手時就沒有唯一解。你能要求的不是「正確」，而是<b>你和別人約好的那一個</b>。所以比對兩個實作的輸出之前，先確認它們的 tie_break 設定一樣。',
              'Small and Large disagree, and both are right - because "the top k" simply has no unique answer under ties. What you can ask for is not correctness but <b>the convention you agreed on</b>. So before diffing two implementations’ outputs, check they use the same tie_break.')}));
    return tieFrames(Object.assign({}, base, {policy:'race',
      head:LB('沒有規則的時候：誰先寫誰贏', 'with no rule: first writer wins'),
      intro:LB('10 個分數要前 5 名。9、7、8 穩穩進榜，剩下 2 個名額要從 5 個都是 4 的元素裡挑 - 演算法沒有規定挑誰。',
              'Ten scores, top five wanted. The 9, 7 and 8 are safe; the remaining two slots must come from five elements that all hold a 4 - and the algorithm does not say which.'),
      stepMsg:(t, o, c, a) => LB('第 ' + t + ' 次：CTA 抵達順序 ' + o.join(' -> ') + '，選到 idx ' + a.join(', ') + '。輸入一個 bit 都沒變。',
                                'Run ' + t + ': the CTAs arrived ' + o.join(' -> ') + ' and the answer is idx ' + a.join(', ') + '. Not one bit of the input changed.'),
      outro:LB('同一份輸入、同一顆 kernel，答案卻在跳。在 LLM 推論裡這代表：<b>同樣的 prompt、同樣的 seed、同樣的權重，取樣出來的 token 可能不一樣</b>。查到最後你會發現不是 sampling 的問題，是 top-k 在平手時沒有約定。',
              'Same input, same kernel, and the answer moves. Inside an LLM this reads as: <b>same prompt, same seed, same weights, different sampled token</b>. You will chase it through the sampler for a day before finding that top-k simply had no convention for ties.')}));
  }
},
{
  id:'gvr', label:LB('猜–驗–修', 'guess-verify-refine'),
  stage:LB('不砍 bit，直接把 threshold 解出來', 'stop peeling bits - solve for the threshold'),
  view:VIEW,
  variants:[LB('kFTarget = k（對的設定）', 'kFTarget = k (the right constant)'),
            LB('kFTarget = 0.75k（會卡死）', 'kFTarget = 0.75k (saturates)'),
            LB('冷啟動：沒有上一步', 'cold start: no previous step')],
  idea:LB('quickselect 和 radix select 都是「把資料切小」；GVR 反過來，<b>先猜一個 threshold，數一次，再用割線解出下一個猜測</b>。它敢這樣猜，是因為在 LLM decode 裡同一列的分布每一步只變一點點 - 上一步的前 k 名 index 就是這一步很好的起點。第二個變體示範 upstream 自己留下的坑：目標值瞄錯位置，迴圈就永遠收不進 accept window。',
         'Quickselect and radix select both shrink the data. GVR goes the other way: <b>guess a threshold, count once, then solve for the next guess with a secant</b>. It can afford to guess because inside LLM decode the same row barely moves between steps - last step’s top-k indices are an excellent starting point. The second variant reproduces a pathology upstream documents in its own tuning table: aim the target at the wrong number and the loop can never reach the accept window.'),
  legend:['hot', 'act', 'ok', 'bad', 'soft'],
  code:CODE_GVR,
  build:(v) => {
    if (v === 1) return gvrFrames({
      target:9, head:LB('kFTarget = 9，比 k 還小', 'kFTarget = 9, below k'),
      p1sub:LB('P1：只讀上一步吐出來的 12 個 index', 'P1: read only the 12 indices from the previous step'),
      intro:LB('同一列、同一個 P1 起點，只改一個常數：把割線瞄準的目標從 12 改成 9。看起來只是「保守一點」，實際上是災難。',
              'Same row, same P1 start, one constant changed: the secant now aims at 9 instead of 12. It looks like a harmless bit of conservatism. It is not.'),
      seed:(c) => gvrSeedMsg(c, false),
      step:(it, f, pinned, thr, c, done) => gvrStepMsg(it, f, pinned, c, done, 9),
      outro:(it, cand, done) => gvrOutroMsg(it, cand, done, 9)});
    if (v === 2) return gvrFrames({
      target:12, cold:true, head:LB('冷啟動：第一個 token', 'cold start: the very first token'),
      p1sub:LB('沒有上一步可以參考，只能取整列的 min / max', 'nothing to reuse - take min / max of the whole row'),
      intro:LB('prefill 的第一個 token 沒有「上一步」。這時候 P1 的賭注失效，只能老實掃一次整列拿 min / max，起點取中點。',
              'The first token of a prefill has no previous step. P1’s bet is unavailable, so it scans the row honestly for min / max and starts at the midpoint.'),
      p1:LB('沒有 pre_idx 可以讀，min / max 得從 120 個值裡掃出來 - 這一次 P1 真的付了整列的成本。中點是個沒有資訊的猜測，但區間是<b>真的</b>區間：真正的 threshold 一定在裡面，不像暖啟動時的 [min, max] 只涵蓋上一步的前 k 名。',
             'With no pre_idx to read, min / max have to come from all 120 values - this time P1 really does pay for a full pass. The midpoint is an uninformed guess, but the bracket is a <b>true</b> bracket: the real threshold is certainly inside it, unlike the warm start’s [min, max], which only spans last step’s top k.'),
      seed:(c) => gvrSeedMsg(c, true),
      step:(it, f, pinned, thr, c, done) => gvrStepMsg(it, f, pinned, c, done, 12),
      outro:(it, cand, done) => gvrOutroMsg(it, cand, done, 12)});
    return gvrFrames({
      target:12, head:LB('kFTarget = k = 12', 'kFTarget = k = 12'),
      p1sub:LB('P1：只讀上一步吐出來的 12 個 index', 'P1: read only the 12 indices from the previous step'),
      intro:LB('120 個 logits，要前 12 名。這一次不切資料，也不看 bit - 我們要<b>猜一個門檻值</b>，然後只用「數數」來修正它。',
              '120 logits, top 12 wanted. This time we do not partition and we do not look at bits - we <b>guess a cut-off value</b> and correct it using nothing but counting.'),
      seed:(c) => gvrSeedMsg(c, false),
      step:(it, f, pinned, thr, c, done) => gvrStepMsg(it, f, pinned, c, done, 12),
      outro:(it, cand, done) => gvrOutroMsg(it, cand, done, 12)});
  }
},
{
  id:'dsmem', label:LB('cluster 內對帳', 'agreeing inside a cluster'),
  stage:LB('多個 CTA 要湊出同一個 count，但不進 global memory', 'many CTAs, one count, no global memory'),
  view:VIEW,
  variants:[LB('4 個 CTA（DSMEM）', '4 CTAs (DSMEM)'),
            LB('2 個 CTA（DSMEM）', '2 CTAs (DSMEM)'),
            LB('1 個 CTA：不用對帳', '1 CTA: nobody to agree with')],
  idea:LB('前面「多個 CTA」那一節是靠 global memory 上的計數 barrier 對帳，一來一回就是幾百個 cycle。Hopper 之後多了 thread block cluster：同一個 GPC 裡的 CTA 可以<b>直接讀彼此的 shared memory</b>（mapa + ld.shared::cluster），於是每一輪的 count 聚合完全留在晶片上，global atomics 是 0。GVR 的 P2 每一次迭代都要聚合一次 count，這條路省下來的就是「迭代次數 x 一次 GMEM 往返」。這是 CUDA 專屬的；CDNA 上沒有對應機制，所以同一份 kernel 在 ROCm 上只能走 global 那條路。',
         'The earlier "many CTAs" section had the blocks agree through a counting barrier in global memory - a few hundred cycles each way. From Hopper on there are thread block clusters: CTAs in the same GPC can <b>read each other’s shared memory directly</b> (mapa + ld.shared::cluster), so the per-iteration count aggregation stays entirely on chip and global atomics drop to zero. GVR’s P2 aggregates a count on every iteration, so what this saves is "iterations x one global round trip". It is CUDA-only; CDNA has no equivalent, which is why the same kernel on ROCm has to take the global path.'),
  legend:['hot', 'act', 'ok', 'soft', 'idle'],
  code:CODE_DSM,
  build:(v) => {
    const cs = v === 1 ? 2 : (v === 2 ? 1 : 4);
    return dsmFrames({cs:cs,
      head:cs === 1 ? LB('cluster_size = 1', 'cluster_size = 1')
                    : LB('cluster_size = ' + cs + '：一列由 ' + cs + ' 個 CTA 分擔', 'cluster_size = ' + cs + ': one row shared by ' + cs + ' CTAs'),
      intro:cs === 1
        ? LB('先看沒有 cluster 的樣子：一個 CTA 扛整列 120 個值。不需要跟任何人對帳，程式碼裡所有 cluster 的部分都會被 const_expr 編掉 - 但這一列的延遲就是這一個 CTA 的延遲。',
            'Start without a cluster: one CTA carries all 120 values. There is nobody to agree with, and every cluster branch in the code is compiled away by const_expr - but the latency of this row is now the latency of this one CTA.')
        : LB('一列 120 個值，' + cs + ' 個 CTA 一起數「有幾個 v >= t」。難的不是數，是數完以後<b>四個人要拿到同一個總和</b>。',
            'One row of 120 values, ' + cs + ' CTAs counting how many satisfy v >= t. Counting is easy; the hard part is that afterwards <b>all of them must end up holding the same total</b>.'),
      count:(r, c, span) => cs === 1
        ? {zh:'唯一的 CTA 掃完 120 個值，數到 ' + c + ' 個。沒有聚合成本，但也沒有平行度 - 序列長度一長，這一列就是整個 kernel 的關鍵路徑。upstream 的 load-balance 版本就是在 launch 前先看 seq_len，長的列才分給多個 CTA。',
           en:'The only CTA sweeps all 120 values and counts ' + c + '. No aggregation cost, but no parallelism either - once the sequence gets long, this row becomes the kernel’s critical path. Upstream’s load-balanced variant classifies rows by seq_len before launch and only hands the long ones to a cluster.'}
        : {zh:'CTA ' + r + ' 用 stride ' + cs + ' 走過屬於它的 ' + span + ' 個值，數到 ' + c + ' 個。<b>用 stride 不用連續切塊</b>是為了合併存取：同一個 warp 的 32 條 thread 讀到的是連續位址。',
           en:'CTA ' + r + ' walks its ' + span + ' values with stride ' + cs + ' and counts ' + c + '. <b>Striding rather than slicing into contiguous chunks</b> is about coalescing: the 32 threads of a warp then touch consecutive addresses.'},
      write:(part) => cs === 1
        ? {zh:'把數字寫進自己的 shared memory。只有一個 CTA 的時候這一步等於什麼都沒做。',
           en:'Write the number into its own shared memory. With a single CTA this step does nothing at all.'}
        : {zh:'每個 CTA 把 partial count 寫進<b>自己的</b> shared memory - 注意沒有人寫進別人的空間，也沒有人碰 global memory。DSMEM 是「你可以讀我的」，不是「大家共用一塊」。',
           en:'Each CTA writes its partial count into <b>its own</b> shared memory - nobody writes into anyone else’s, and nobody touches global memory. DSMEM means "you may read mine", not "we share one buffer".'},
      sync:(n) => cs === 1
        ? {zh:'barrier 只涵蓋自己，直接通過。',
           en:'The barrier covers only itself and falls straight through.'}
        : {zh:'cluster.sync() 只要求這 ' + n + ' 個 CTA 互相等待，而它們被硬體保證同時 resident 在同一個 GPC 上 - 所以這個 barrier <b>不可能</b>死鎖。這正是前面 #3610 那個 grid-wide barrier 出事的地方：grid 大於一個 wave 時，還沒被排進去的 CTA 永遠等不到。',
           en:'cluster.sync() only asks these ' + n + ' CTAs to wait for each other, and the hardware guarantees they are co-resident in one GPC - so this barrier <b>cannot</b> deadlock. That is exactly where the grid-wide barrier of #3610 went wrong: when the grid exceeds one wave, the CTAs that were never scheduled can never arrive.'},
      read:(p, val, running, n) => cs === 1
        ? {zh:'讀自己的那一格，total = ' + running + '。整個聚合迴圈只有一圈。',
           en:'Read its own slot, total = ' + running + '. The whole aggregation loop is one trip.'}
        : {zh:'mapa 把「smem 的第 0 格」翻譯成 peer ' + p + ' 的實體 shared address，然後 ld.shared::cluster 直接讀過去，延遲跟讀自己的 shared memory 同一個量級。四個 CTA 同時做這件事，於是 total 一起長到 ' + running + '。<b>沒有 atomicAdd，沒有 L2，沒有往返</b>。',
           en:'mapa translates "slot 0 of smem" into peer ' + p + '’s physical shared address, and ld.shared::cluster reads it at roughly the latency of its own shared memory. All ' + n + ' CTAs do this at once, so every total climbs to ' + running + ' together. <b>No atomicAdd, no L2, no round trip.</b>'},
      outro:(total, n) => cs === 1
        ? {zh:'答案是 ' + total + '，只有一個 CTA 得知它。省下了通訊，代價是這一列沒有被平行化 - upstream 量到的斷點大約在 scan 長度 64K：短的列一個 CTA 就好，長的列才值得付 cluster 的錢。',
           en:'The answer is ' + total + ', known to a single CTA. Communication is free, but the row was never parallelised - upstream measures the break-even at a scan length of about 64K: short rows want one CTA, only long rows are worth the cluster.'}
        : {zh:'' + n + ' 個 CTA 手上都是 ' + total + '，可以各自獨立決定下一個 threshold，不需要再同步一次。GVR 的 P2 每一次迭代都要走一遍這個流程 - 把它從 global memory 搬到 GPC 內部，省的不是一次，是<b>迭代次數那麼多次</b>。這也是為什麼「少跑幾次迭代」（kFTarget 那個常數）和「每次迭代便宜一點」（cluster）是同一個問題的兩面。',
           en:'All ' + n + ' CTAs hold ' + total + ', so each can decide the next threshold on its own with no further synchronisation. GVR’s P2 runs this exchange on every iteration - moving it from global memory into the GPC saves not one round trip but <b>as many as there are iterations</b>. Which is why "run fewer iterations" (that kFTarget constant) and "make each iteration cheaper" (the cluster) are two faces of the same problem.'}});
  }
},
{
  id:'lc', label:LB('LeetCode 215', 'LeetCode 215'),
  stage:LB('Kth Largest Element in an Array', 'Kth Largest Element in an Array'),
  view:VIEW,
  variants:[LB('一般輸入', 'ordinary input'), LB('大量重複', 'heavy duplicates')],
  idea:LB('面試題就是 quickselect 本人。標準答案是「random pivot + 三路 partition」，而<b>面試官真正在等的是後者</b>：兩路 partition 遇到一整排相同的值會退化成 O(n²)，這正是第二個變體示範的情況。',
         'The interview problem is quickselect itself. The expected answer is "random pivot plus three-way partition", and <b>the second half is what the interviewer is waiting for</b>: a two-way partition degrades to O(n²) on a run of equal values, which is exactly what the second variant shows.'),
  legend:['hot', 'act', 'ok', 'soft', 'done'],
  code:CODE_QS,
  build:(v) => v === 1
    ? qsFrames({data:[3,2,3,1,2,4,5,5,6,4,3,5], k:4, seed:5,
        head:LB('LC 215：重複很多的輸入', 'LC 215: heavy duplicates'),
        intro:LB('nums = [3,2,3,1,2,4,5,5,6,4,3,5]，k = 4。第 4 大是 5 - 注意「第 4 大」數的是<b>位置</b>不是<b>相異值</b>，5 出現三次就佔三個名額。',
                'nums = [3,2,3,1,2,4,5,5,6,4,3,5], k = 4. The fourth largest is 5 - "fourth largest" counts <b>positions</b>, not <b>distinct values</b>, so three fives take three slots.'),
        outro:LB('三路 partition 讓那三個 5 一輪就退場。如果用兩路，pivot 每次都只能切下一個元素，n 個相同的值就是 n 輪 - O(n²)。這題只要提到這一點，就答完了。',
                'The three-way partition retires all three fives in one round. With a two-way split the pivot peels off one element at a time, so n equal values cost n rounds - O(n²). Say that in the interview and the question is answered.')})
    : qsFrames({data:[3,2,1,5,6,4], k:2, seed:9,
        head:LB('LC 215：第 2 大是誰', 'LC 215: the second largest'),
        intro:LB('nums = [3,2,1,5,6,4]，k = 2，答案是 5。用 sort 一行就寫完，但那是 O(n log n)；面試官問的是能不能做到 O(n)。',
                'nums = [3,2,1,5,6,4], k = 2, answer 5. A one-line sort solves it in O(n log n); the question is whether you can do it in O(n).'),
        outro:LB('答案是 a[k-1] = 5。三個補充值得說出口：pivot 一定要隨機（不然遞增輸入就是最壞情況）、用迴圈不要遞迴（避免 stack overflow）、如果資料是串流進來的就改用大小為 k 的 min-heap - O(n log k)、記憶體只要 O(k)，而且永遠不需要把整份資料留在手上。',
                'The answer is a[k-1] = 5. Three things worth saying out loud: the pivot must be random (otherwise a sorted input is the worst case), write it as a loop rather than recursion (stack depth), and if the data arrives as a stream, switch to a size-k min-heap - O(n log k) time, O(k) memory, and it never needs the whole input in hand.')})
}
]
};
