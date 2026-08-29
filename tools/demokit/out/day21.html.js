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
