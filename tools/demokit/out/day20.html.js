// DAY: 20
// TITLE_ZH: 分治排序：merge sort、quick sort，以及它們的平行版
// TITLE_EN: Divide and conquer, twice - merge sort, quick sort, and going parallel
// SUB_ZH: 兩個都是「切一半再遞迴」，差別只在工作放在切的時候還是合的時候。
// SUB_EN: Both cut the problem in half; the only difference is whether the work happens in the split or in the join.
// FOLDER: day%2020%20-%20quick%20sort%20and%20merge%20sort
// MEDIUM: https://medium.com/100-days-of-python

/* ------------------------------------------------------------ layout bits */
const VIEW = [9.4, 6.4];
const CW = .82, CH = .82;
const rowX = n => (VIEW[0] - n * CW) / 2;

function arrRow(vals, y, st, opt){
  opt = opt || {};
  const x0 = opt.x0 == null ? rowX(vals.length) : opt.x0;
  const out = [];
  vals.forEach((v, i) => {
    out.push(S.r(x0 + i * CW, y, CW - .08, CH, (st && st[i]) || 'idle',
                 v == null ? '' : String(v), {fs:.36, rx:.06}));
  });
  if (opt.title)
    out.push(S.t(x0 - .28, y + CH * .58, opt.title, {c:COL.tealL, fs:.32, anchor:'end'}));
  (opt.marks || []).forEach(m => {
    out.push(S.t(x0 + m.i * CW + (CW - .08) / 2, m.below ? y + CH + .46 : y - .26,
                 m.t, {c:m.c || COL.purpleL, fs:.34}));
  });
  return out;
}

const note = (y, s, c) => S.t(VIEW[0] / 2, y, s, {c:c || COL.tealL, fs:.34});
const n2 = n => n * (n - 1) / 2;

/* ------------------------------------------------------------- merge sort */
const MA = [5, 2, 9, 1, 7, 6, 3, 8];

const CODE_M = [
'def merge_sort(a):',
'    if len(a) <= 1:',
'        return a',
'    mid = len(a) // 2',
'    left  = merge_sort(a[:mid])',
'    right = merge_sort(a[mid:])',
'    return merge(left, right)',
'',
'def merge(left, right):',
'    out, i, j = [], 0, 0',
'    while i < len(left) and j < len(right):',
'        if left[i] <= right[j]:   # <= is what makes it stable',
'            out.append(left[i]); i += 1',
'        else:',
'            out.append(right[j]); j += 1',
'    return out + left[i:] + right[j:]'
];

function mergeFrames(){
  const F = new Frames();
  const a = MA.slice();
  const ROW = 2.9, OUT = 4.7;
  let cmp = 0, inv = 0, merges = 0;

  const shapes = (o) => {
    o = o || {};
    const st = {};
    for (let i = 0; i < a.length; i++) st[i] = (o.st && o.st[i]) ? o.st[i] : 'idle';
    let out = arrRow(a, ROW, st, {title:{zh:'陣列', en:'array'}, marks:o.marks || []});
    if (o.out){
      const os = {};
      o.out.forEach((v, k) => { os[k] = 'ok'; });
      out = out.concat(arrRow(o.out, OUT, os,
        {title:{zh:'合併結果', en:'merged'}, x0:rowX(a.length) + o.lo * CW}));
    }
    if (o.note) out.push(note(1.35, o.note, o.noteC));
    out.push(S.t(VIEW[0] / 2, 6.05,
      {zh:'切一半不用動腦，力氣全花在把兩段合起來',
       en:'splitting is free - all the effort goes into joining'}, {c:COL.grey, fs:.28}));
    return out;
  };
  const panels = (extra) => [
    {lbl:{zh:'比較次數', en:'comparisons'}, chips:[{t:String(cmp), cls:'ok'}]},
    {lbl:{zh:'已完成的 merge', en:'merges completed'}, chips:[{t:String(merges), cls:'act'}]},
    {lbl:{zh:'順便算到的 inversion', en:'inversions counted on the way'},
     chips:[{t:String(inv), cls:'hot'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:{zh:'先把陣列一路切到剩單一元素 - 這一半完全不用比較',
                               en:'cut all the way down to single elements - this half costs no comparisons at all'}}),
          panels:panels(), view:VIEW, line:3,
          msg:{zh:'merge sort 的遞迴長得很無聊：<b>mid = len(a) // 2</b>，然後左右各自遞迴。切的時候沒有任何比較，也不看資料內容，所以不管輸入長什麼樣子，樹的形狀都一樣 - 這就是它<b>沒有最壞情況</b>的原因。',
               en:'The recursion is deliberately boring: <b>mid = len(a) // 2</b>, then recurse on each half. The split does no comparisons and never looks at the data, so the tree has the same shape for every input - which is exactly why merge sort <b>has no bad case</b>.'}});

  function msort(lo, hi){
    if (hi - lo <= 1) return;
    const mid = (lo + hi) >> 1;
    msort(lo, mid); msort(mid, hi);
    const L = a.slice(lo, mid), R = a.slice(mid, hi);
    const stSeg = () => {
      const st = {};
      for (let i = lo; i < mid; i++) st[i] = 'act';
      for (let i = mid; i < hi; i++) st[i] = 'soft';
      return st;
    };
    F.push({shapes:shapes({st:stSeg(), lo:lo,
              note:{zh:'合併 [' + lo + ',' + mid + ') 與 [' + mid + ',' + hi + ')　兩段本身已經排好',
                    en:'merge [' + lo + ',' + mid + ') with [' + mid + ',' + hi + ') - both halves are already sorted'}}),
            panels:panels([{t:'left ' + L.join(' '), cls:'act'},
                           {t:'right ' + R.join(' '), cls:'dim'}]),
            view:VIEW, line:9,
            msg:{zh:'兩段都已經排序好，所以合併只要各拿一根手指從頭走：<b>每比一次就確定一個位置</b>，一趟 O(n)。這就是 merge sort 全部的成本來源。',
                 en:'Both halves are sorted, so merging is two fingers walking forward: <b>each comparison fixes one output slot</b>, one O(n) pass. This is where the entire cost of merge sort lives.'}});

    const out = []; let i = 0, j = 0;
    while (i < L.length && j < R.length){
      cmp++;
      const takeL = L[i] <= R[j];
      const gained = takeL ? 0 : (L.length - i);
      if (!takeL) inv += gained;
      const li = L[i], rj = R[j];
      out.push(takeL ? li : rj);
      const st = stSeg();
      st[lo + i] = 'hot'; st[mid + j] = 'hot';
      const msg = takeL
        ? {zh:'<b>' + li + ' &le; ' + rj + '</b>，拿左邊。相等時也拿左邊 - 這個 <b>&le;</b> 就是穩定性的全部祕密：相等的兩個元素，原本在左邊的還是先出去。',
           en:'<b>' + li + ' &le; ' + rj + '</b>, so take the left one. Ties go left too, and that <b>&le;</b> is the whole secret of stability: of two equal elements, the one that started on the left still leaves first.'}
        : {zh:'<b>' + rj + ' &lt; ' + li + '</b>，拿右邊。注意左邊還剩 <b>' + gained + '</b> 個沒出去，而它們全都比 ' + rj + ' 大 - 所以這一步同時算出 <b>' + gained + ' 個 inversion</b>，完全不用額外的迴圈。',
           en:'<b>' + rj + ' &lt; ' + li + '</b>, so take the right one. The left half still holds <b>' + gained + '</b> unemitted values and every one of them is bigger than ' + rj + ', so this single step also counts <b>' + gained + ' inversions</b> - no extra loop required.'};
      F.push({shapes:shapes({st:st, marks:[{i:lo + i, t:'i', c:COL.orangeL},
                                           {i:mid + j, t:'j', c:COL.purpleL}],
                             out:out.slice(), lo:lo}),
              panels:panels([{t:'left[i] = ' + li, cls:takeL ? 'hot' : 'dim'},
                             {t:'right[j] = ' + rj, cls:takeL ? 'dim' : 'hot'},
                             {t:takeL ? '+0 inv' : '+' + gained + ' inv', cls:takeL ? 'dim' : 'bad'}]),
              view:VIEW, line:takeL ? 12 : 14, msg:msg});
      if (takeL) i++; else j++;
    }
    while (i < L.length) out.push(L[i++]);
    while (j < R.length) out.push(R[j++]);
    for (let k = 0; k < out.length; k++) a[lo + k] = out[k];
    merges++;
    const stDone = {};
    for (let k = lo; k < hi; k++) stDone[k] = 'ok';
    F.push({shapes:shapes({st:stDone, lo:lo,
              note:{zh:'[' + lo + ',' + hi + ') 排好了，寫回陣列',
                    en:'[' + lo + ',' + hi + ') is sorted, written back'}}),
            panels:panels([{t:out.join(' '), cls:'ok'}]), view:VIEW, line:15,
            msg:{zh:'其中一邊先走完，剩下那一邊直接倒進去 - 它們本來就已經排好且全部比較大。<b>merge 需要一塊 O(n) 的暫存空間</b>，這是 merge sort 唯一輸給 quick sort 的地方，卻也正好是它能做外部排序的原因。',
                 en:'One side runs out first and the rest is poured in as-is - already sorted and already larger. <b>The merge needs O(n) scratch space</b>, the one place merge sort loses to quick sort, and also precisely why it can sort data that does not fit in memory.'}});
  }
  msort(0, a.length);

  const allOk = {};
  a.forEach((v, k) => { allOk[k] = 'ok'; });
  F.push({shapes:shapes({st:allOk,
            note:{zh:'比較 ' + cmp + ' 次　·　inversion ' + inv + ' 個　·　永遠是 O(n log n)',
                  en:cmp + ' comparisons  ·  ' + inv + ' inversions  ·  O(n log n), always'}}),
          panels:panels([{t:'sorted', cls:'ok'}]), view:VIEW, line:6,
          msg:{zh:'總共 <b>' + cmp + ' 次比較</b>。樹有 log n 層、每層總共走過 n 個元素，所以是 n log n，而且<b>與輸入無關</b>：排好的、倒序的、全部相同的，成本一模一樣。順帶把 <b>' + inv + ' 個 inversion</b> 也算出來了。',
               en:'<b>' + cmp + ' comparisons</b> in total. The tree has log n levels and each level walks n elements, so n log n - and <b>independent of the input</b>: sorted, reversed or all-equal data costs exactly the same. It also handed us <b>' + inv + ' inversions</b> for free.'}});
  return F.list;
}

/* ------------------------------------------------------------- quick sort */
const CODE_Q = [
'def quicksort(a, lo, hi):        # Lomuto, two-way',
'    if lo >= hi:',
'        return',
'    p = a[hi]                    # pivot = last element',
'    i = lo                       # invariant: a[lo:i] < p',
'    for j in range(lo, hi):',
'        if a[j] < p:',
'            a[i], a[j] = a[j], a[i]',
'            i += 1',
'    a[i], a[hi] = a[hi], a[i]    # pivot lands at i',
'    quicksort(a, lo, i - 1)      # and never moves again',
'    quicksort(a, i + 1, hi)'
];

function quickFrames(cfg){
  const F = new Frames();
  const a = cfg.arr.slice();
  const ROW = 3.1;
  let cmp = 0, maxDepth = 0;
  const fixed = {};

  const shapes = (o) => {
    o = o || {};
    const st = {};
    for (let i = 0; i < a.length; i++){
      st[i] = fixed[i] ? 'ok' : 'idle';
      if (o.seg && i >= o.seg[0] && i <= o.seg[1]) st[i] = 'soft';
      if (o.lt != null && o.seg && i >= o.seg[0] && i < o.lt) st[i] = 'act';
      if (o.piv === i) st[i] = 'hot';
    }
    const out = arrRow(a, ROW, st, {marks:o.marks || []});
    if (o.note) out.push(note(1.55, o.note, o.noteC));
    out.push(S.t(VIEW[0] / 2, 6.05, cfg.top, {c:COL.grey, fs:.28}));
    return out;
  };
  const panels = (extra) => [
    {lbl:{zh:'比較次數', en:'comparisons'}, chips:[{t:String(cmp), cls:cmp > 20 ? 'bad' : 'ok'}]},
    {lbl:{zh:'遞迴最深', en:'deepest recursion'},
     chips:[{t:String(maxDepth), cls:maxDepth > 4 ? 'bad' : 'act'}]},
    {lbl:{zh:'已定案的位置', en:'slots finalised'},
     chips:[{t:String(Object.keys(fixed).length), cls:'ok'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:cfg.opening.note}), panels:panels(), view:VIEW,
          line:3, msg:cfg.opening.msg});

  function qs(lo, hi, depth){
    if (lo >= hi){ if (lo === hi) fixed[lo] = 1; return; }
    maxDepth = Math.max(maxDepth, depth);
    const p = a[hi];
    F.push({shapes:shapes({seg:[lo, hi], piv:hi,
              note:{zh:'處理 a[' + lo + '..' + hi + ']，pivot = a[' + hi + '] = ' + p,
                    en:'working on a[' + lo + '..' + hi + '], pivot = a[' + hi + '] = ' + p}}),
            panels:panels([{t:'pivot ' + p, cls:'hot'}]), view:VIEW, line:3,
            msg:cfg.pick(lo, hi, p)});
    let i = lo;
    for (let j = lo; j < hi; j++){
      cmp++;
      const v = a[j], less = v < p;
      if (less){ const t = a[i]; a[i] = a[j]; a[j] = t; i++; }
      F.push({shapes:shapes({seg:[lo, hi], piv:hi, lt:i,
                marks:[{i:i, t:'i', c:COL.orangeL}, {i:j, t:'j', c:COL.purpleL}]}),
              panels:panels([{t:'a[j] = ' + v, cls:'act'},
                             {t:less ? '< pivot' : '>= pivot', cls:less ? 'hot' : 'dim'}]),
              view:VIEW, line:less ? 7 : 6,
              msg:less
                ? {zh:'<b>' + v + ' &lt; ' + p + '</b>，換到小的那一區並把 i 往前推。i 永遠指著「小區的下一個空位」，這是 Lomuto 唯一要記住的不變式。',
                   en:'<b>' + v + ' &lt; ' + p + '</b>, so it swaps into the small region and i advances. i always points at the next free slot of the small region - the only invariant Lomuto asks you to hold.'}
                : {zh:'<b>' + v + ' &ge; ' + p + '</b>，什麼都不用做，j 自己往前。大的元素會自然留在右半邊。',
                   en:'<b>' + v + ' &ge; ' + p + '</b>, so nothing happens and j moves on. Large values simply stay where they are, in the right half.'}});
    }
    const t = a[i]; a[i] = a[hi]; a[hi] = t;
    fixed[i] = 1;
    const lsz = i - lo, rsz = hi - i;
    F.push({shapes:shapes({seg:[lo, hi], piv:i,
              note:{zh:'pivot 落在 ' + i + '　·　左邊 ' + lsz + ' 個、右邊 ' + rsz + ' 個',
                    en:'pivot lands at ' + i + '  ·  ' + lsz + ' on the left, ' + rsz + ' on the right'},
              noteC:(lsz === 0 || rsz === 0) ? COL.red : COL.tealL}),
            panels:panels([{t:'split ' + lsz + ' / ' + rsz,
                            cls:(lsz === 0 || rsz === 0) ? 'bad' : 'ok'}]),
            view:VIEW, line:9, msg:cfg.after(lo, hi, i, lsz, rsz)});
    qs(lo, i - 1, depth + 1);
    qs(i + 1, hi, depth + 1);
  }
  qs(0, a.length - 1, 1);

  F.push({shapes:shapes({note:cfg.done.note(cmp, maxDepth),
                         noteC:cfg.bad ? COL.red : COL.tealL}),
          panels:panels([{t:'sorted', cls:cfg.bad ? 'bad' : 'ok'}]), view:VIEW,
          line:11, msg:cfg.done.msg(cmp, maxDepth)});
  return F.list;
}

const CFG_OK = {
  arr:[7, 2, 9, 4, 6, 1, 8, 3],
  top:{zh:'資料夠亂的時候，quick sort 是就地排序、常數小、快取友善',
       en:'on messy data quick sort is in-place, cache-friendly, and has a tiny constant'},
  opening:{note:{zh:'quick sort 沒有 merge 步驟 - pivot 一放好就永遠不動',
                 en:'no merge step at all - once the pivot is placed it never moves again'},
           msg:{zh:'quick sort 把力氣全部花在 <b>partition</b>：掃一遍，把小的丟左邊、大的留右邊，pivot 就落在它<b>最終的位置</b>。左右兩段排好之後不需要任何合併，因為它們早就在正確的區間裡了。',
                en:'Quick sort spends everything on the <b>partition</b>: one pass that pushes small values left and leaves large ones right, so the pivot lands in its <b>final</b> position. Once the two sides are sorted there is nothing to merge - they are already in the right places.'}},
  pick:(lo, hi, p) => ({zh:'取最後一個元素 <b>' + p + '</b> 當 pivot。挑哪一個決定了切得平不平均，而切得平不平均決定了整個演算法是 n log n 還是 n<sup>2</sup>。',
                        en:'Take the last element, <b>' + p + '</b>, as the pivot. Which element you pick decides how evenly the array splits, and that decides whether the whole algorithm is n log n or n squared.'}),
  after:(lo, hi, i, l, r) => ({zh:'一趟掃完，pivot 換到位置 <b>' + i + '</b>，左 ' + l + ' 右 ' + r + '。<b>這一格從此定案</b> - 後面的遞迴永遠不會再碰它，這就是 quick sort 不需要合併的原因。',
                               en:'One pass done: the pivot swaps to index <b>' + i + '</b>, leaving ' + l + ' on the left and ' + r + ' on the right. <b>That slot is now final</b> - no later call ever touches it, which is why quick sort needs no join step.'}),
  done:{note:(c, d) => ({zh:'比較 ' + c + ' 次　·　最深 ' + d + ' 層',
                         en:c + ' comparisons  ·  ' + d + ' levels deep'}),
        msg:(c, d) => ({zh:'共 <b>' + c + ' 次比較</b>、遞迴最深 <b>' + d + '</b> 層。切得夠平均時遞迴只有 log n 層，而且整趟都在原陣列上就地搬動 - 沒有暫存陣列、快取命中好，所以實務上常常比 merge sort 快，即使兩者都是 O(n log n)。',
                        en:'<b>' + c + ' comparisons</b> and <b>' + d + '</b> levels of recursion. When the splits are even the recursion is only log n deep, and every move happens inside the original array - no scratch buffer, excellent cache behaviour, which is why it often beats merge sort in practice even though both are O(n log n).'})}
};

const CFG_SORTED = {
  arr:[1, 2, 3, 4, 5, 6, 7, 8], bad:true,
  top:{zh:'輸入已經排好序 - 教科書版 quick sort 最怕的一種資料',
       en:'the input is already sorted - the one shape textbook quick sort cannot survive'},
  opening:{note:{zh:'已經排好的輸入，pivot 固定取最後一個',
                 en:'already sorted input, pivot fixed at the last element'},
           msg:{zh:'資料已經排好了，直覺會說「那不是更輕鬆嗎」。但 pivot 取<b>最後一個元素</b>時，它剛好是整段裡最大的 - 切出來是 <b>n-1 和 0</b>，一次只解決一個元素。',
                en:'The data is already sorted, which sounds like it should be easier. But with the pivot fixed at the <b>last element</b> it is always the largest value in the segment, so the split is <b>n-1 and 0</b>: one element retired per level.'}},
  pick:(lo, hi, p) => ({zh:'pivot = ' + p + '，又是這一段裡最大的一個。等一下你會看到 i 從頭到尾都不動。',
                        en:'pivot = ' + p + ', once again the largest value in this segment. Watch i refuse to move for the entire scan.'}),
  after:(lo, hi, i, l, r) => ({zh:'切成 <b>' + l + ' / ' + r + '</b> - 一邊是空的。遞迴每往下一層只少一個元素，於是層數是 n 而不是 log n，比較次數是 n<sup>2</sup>/2。<b>它不會報錯，只會慢到爆</b>，這在線上評測裡就是 TLE。',
                               en:'Split <b>' + l + ' / ' + r + '</b> - one side is empty. Each level removes a single element, so the depth is n instead of log n and the comparisons are n squared over 2. <b>Nothing raises an error; it just crawls</b>, which on an online judge reads as TLE.'}),
  done:{note:(c, d) => ({zh:'比較 ' + c + ' 次（= n(n-1)/2）　·　' + d + ' 層',
                         en:c + ' comparisons (= n(n-1)/2)  ·  ' + d + ' levels'}),
        msg:(c, d) => ({zh:'8 個元素就要 <b>' + c + ' 次比較</b>、<b>' + d + '</b> 層遞迴。放大到 200 個就是 <b>' + n2(200) + '</b> 次與 199 層 - CPython 預設遞迴上限 1000，資料再大一點會直接 RecursionError。修法只有一行：<b>pivot 隨機選</b>，讓輸入無法預測。',
                        en:'Eight elements already cost <b>' + c + ' comparisons</b> and <b>' + d + '</b> levels. At n = 200 that is <b>' + n2(200) + '</b> comparisons and 199 levels - and CPython gives up at 1000, so a slightly bigger array is a RecursionError. The fix is one line: <b>choose the pivot at random</b> so no input can predict it.'})}
};

const CFG_EQ = {
  arr:[7, 7, 7, 7, 7, 7, 7, 7], bad:true,
  top:{zh:'全部相同的元素 - 隨機 pivot 也救不了的一種資料',
       en:'every element equal - the case a random pivot does not fix'},
  opening:{note:{zh:'八個一模一樣的值，兩路 partition',
                 en:'eight identical values, two-way partition'},
           msg:{zh:'剛剛學到「隨機 pivot 就解決了」- 這一組資料是來打臉的。全部元素相同時，<b>不管挑哪一個當 pivot 都一樣</b>，因為 <b>a[j] &lt; p</b> 永遠是 False。',
                en:'The lesson so far was "random pivots fix it". This input exists to contradict it: when every element is equal it makes <b>no difference which one you pick</b>, because <b>a[j] &lt; p</b> is never true.'}},
  pick:(lo, hi, p) => ({zh:'pivot = ' + p + '。隨機選也是 7，選哪個都是 7 - 問題不在挑法，在於<b>兩路 partition 沒有「等於」這一區</b>。',
                        en:'pivot = ' + p + '. A random choice is also 7; every choice is 7. The problem is not the selection rule, it is that <b>a two-way partition has no "equal" region</b>.'}),
  after:(lo, hi, i, l, r) => ({zh:'又是 <b>' + l + ' / ' + r + '</b>。這些元素明明已經在正確位置了，卻被一層一層重新 partition - <b>做的全是白工</b>。',
                               en:'Split <b>' + l + ' / ' + r + '</b> again. These values are already in their correct places, yet every level re-partitions them - <b>pure wasted work</b>.'}),
  done:{note:(c, d) => ({zh:'比較 ' + c + ' 次　·　' + d + ' 層　·　隨機 pivot 完全沒幫上忙',
                         en:c + ' comparisons  ·  ' + d + ' levels  ·  the random pivot changed nothing'}),
        msg:(c, d) => ({zh:'<b>' + c + ' 次比較、' + d + ' 層</b>，跟排好序的輸入一模一樣糟。放大到 400 個相同的值就是 <b>' + n2(400) + '</b> 次比較。真正的修法是把 partition 從兩路改成<b>三路</b>：讓「等於 pivot」自成一區、直接退休 - 下一個分頁就是這件事。',
                        en:'<b>' + c + ' comparisons over ' + d + ' levels</b>, exactly as bad as the sorted input. Scale it to 400 equal values and it is <b>' + n2(400) + '</b> comparisons. The real fix is to make the partition <b>three-way</b>, giving "equal to the pivot" its own region that retires immediately - which is the next tab.'})}
};

/* -------------------------------- three-way partition / Dutch flag / LC 912 */
const CODE_3 = [
'def three_way_partition(a, lo, hi, p):',
'    lt, i, gt = lo, lo, hi   # a[lo:lt] < p',
'    while i <= gt:           # a[lt:i]  == p',
'        if a[i] < p:         # a[gt+1:] > p',
'            a[lt], a[i] = a[i], a[lt]',
'            lt += 1; i += 1',
'        elif a[i] > p:',
'            a[i], a[gt] = a[gt], a[i]',
'            gt -= 1          # do NOT advance i: the value',
'        else:                # swapped in from the right',
'            i += 1           # has not been examined yet',
'    return lt, gt            # a[lt:gt+1] is finished'
];

function threeFrames(cfg){
  const F = new Frames();
  const a = cfg.arr.slice();
  const ROW = 3.1;
  let cmp = 0, maxDepth = 0, retired = 0;
  const fixed = {};

  const shapes = (o) => {
    o = o || {};
    const st = {};
    for (let i = 0; i < a.length; i++){
      st[i] = fixed[i] ? 'ok' : 'idle';
      if (o.seg && i >= o.seg[0] && i <= o.seg[1]) st[i] = 'soft';
      if (o.lt != null && o.seg){
        if (i >= o.seg[0] && i < o.lt) st[i] = 'act';
        if (i >= o.lt && i < o.i) st[i] = 'hot';
      }
      if (o.eq && i >= o.eq[0] && i <= o.eq[1]) st[i] = 'hot';
    }
    const out = arrRow(a, ROW, st, {marks:o.marks || []});
    if (o.note) out.push(note(1.55, o.note, o.noteC));
    out.push(S.t(VIEW[0] / 2, 6.05, cfg.top, {c:COL.grey, fs:.28}));
    return out;
  };
  const panels = (extra) => [
    {lbl:{zh:'比較次數', en:'comparisons'}, chips:[{t:String(cmp), cls:'ok'}]},
    {lbl:{zh:'遞迴最深', en:'deepest recursion'}, chips:[{t:String(maxDepth), cls:'act'}]},
    {lbl:{zh:'被「等於區」直接退休的元素', en:'retired by the equal region'},
     chips:[{t:String(retired), cls:'ok'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  F.push({shapes:shapes({note:cfg.opening.note}), panels:panels(), view:VIEW,
          line:1, msg:cfg.opening.msg});

  function qs(lo, hi, depth){
    if (lo >= hi){ if (lo === hi) fixed[lo] = 1; return; }
    maxDepth = Math.max(maxDepth, depth);
    const p = a[(lo + hi) >> 1];
    let lt = lo, i = lo, gt = hi;
    F.push({shapes:shapes({seg:[lo, hi],
              note:{zh:'a[' + lo + '..' + hi + ']，pivot = ' + p,
                    en:'a[' + lo + '..' + hi + '], pivot = ' + p}}),
            panels:panels([{t:'pivot ' + p, cls:'hot'}]), view:VIEW, line:1,
            msg:cfg.pick(lo, hi, p)});
    while (i <= gt){
      cmp++;
      const marks = [{i:lt, t:'lt', c:COL.tealL}, {i:i, t:'i', c:COL.orangeL},
                     {i:gt, t:'gt', c:COL.purpleL}];
      const v = a[i];
      let msg, line;
      if (v < p){
        const t = a[lt]; a[lt] = a[i]; a[i] = t; lt++; i++;
        line = 4;
        msg = {zh:'<b>' + v + ' &lt; ' + p + '</b>，丟進左區，lt 與 i 一起前進。',
               en:'<b>' + v + ' &lt; ' + p + '</b>, so it goes to the left region and both lt and i advance.'};
      } else if (v > p){
        const t = a[i]; a[i] = a[gt]; a[gt] = t; gt--;
        line = 7;
        msg = {zh:'<b>' + v + ' &gt; ' + p + '</b>，跟右端交換、gt 退一格。注意 <b>i 不能前進</b>：從右邊換過來的那個值我們還沒看過，下一輪要重新判斷它。這是這段程式碼最容易寫錯的一行。',
               en:'<b>' + v + ' &gt; ' + p + '</b>, so swap it to the right end and step gt back. Note that <b>i must not advance</b>: the value swapped in from the right has never been examined, and the next iteration has to judge it. This is the single easiest line to get wrong.'};
      } else {
        i++; retired++;
        line = 10;
        msg = {zh:'<b>等於 pivot</b>：留在原地，i 前進。它已經在最終位置上了 - <b>這一格從此不再被任何遞迴碰到</b>，這正是全部相同的輸入從 n<sup>2</sup> 掉到 n 的原因。',
               en:'<b>Equal to the pivot</b>: leave it, advance i. It is already in its final place and <b>no recursive call will ever look at it again</b> - exactly why the all-equal input collapses from n squared to n.'};
      }
      F.push({shapes:shapes({seg:[lo, hi], lt:lt, i:i, marks:marks}),
              panels:panels([{t:'lt=' + lt + '  i=' + i + '  gt=' + gt, cls:'act'}]),
              view:VIEW, line:line, msg:msg});
    }
    for (let k = lt; k <= gt; k++) fixed[k] = 1;
    F.push({shapes:shapes({seg:[lo, hi], eq:[lt, gt],
              note:{zh:'&lt; ' + p + ' ｜ == ' + p + ' 共 ' + (gt - lt + 1) + ' 個 ｜ &gt; ' + p,
                    en:'< ' + p + '  |  == ' + p + ', ' + (gt - lt + 1) + ' of them  |  > ' + p}}),
            panels:panels([{t:'equal block ' + (gt - lt + 1), cls:'ok'}]),
            view:VIEW, line:11, msg:cfg.after(lt, gt, p)});
    qs(lo, lt - 1, depth + 1);
    qs(gt + 1, hi, depth + 1);
  }
  qs(0, a.length - 1, 1);

  F.push({shapes:shapes({note:cfg.done.note(cmp, maxDepth)}),
          panels:panels([{t:'sorted', cls:'ok'}]), view:VIEW, line:11,
          msg:cfg.done.msg(cmp, maxDepth)});
  return F.list;
}

const CFG_3EQ = {
  arr:[7, 7, 7, 7, 7, 7, 7, 7],
  top:{zh:'同一組全相同的資料，改用三路 partition',
       en:'the same all-equal input, now with a three-way partition'},
  opening:{note:{zh:'上一頁要 28 次比較、遞迴 7 層，這一頁看看要幾次',
                 en:'the previous tab needed 28 comparisons over 7 levels - watch this one'},
           msg:{zh:'三路 partition 維持三個區間：<b>&lt; p ｜ == p ｜ &gt; p</b>。關鍵在中間那一區：等於 pivot 的元素<b>已經在最終位置</b>，可以直接退休，不必再進遞迴。',
                en:'A three-way partition maintains three regions: <b>&lt; p | == p | &gt; p</b>. The middle one is the point: values equal to the pivot are <b>already in their final positions</b> and can retire immediately instead of entering the recursion.'}},
  pick:(lo, hi, p) => ({zh:'pivot = ' + p + '。這一次「等於」不再是沒人要的孤兒，而是自己一區。',
                        en:'pivot = ' + p + '. This time "equal" is not an orphan case - it gets a region of its own.'}),
  after:(lt, gt, p) => ({zh:'等於區有 <b>' + (gt - lt + 1) + '</b> 個元素，全部一次退休。兩邊遞迴收到的是空區間，整個排序就結束了。',
                         en:'The equal block holds <b>' + (gt - lt + 1) + '</b> elements and all of them retire at once. Both recursive calls receive empty ranges, and the sort is over.'}),
  done:{note:(c, d) => ({zh:'比較 ' + c + ' 次　·　' + d + ' 層　·　一趟掃完就排好',
                         en:c + ' comparisons  ·  ' + d + ' level  ·  sorted in a single pass'}),
        msg:(c, d) => ({zh:'<b>' + c + ' 次比較、' + d + ' 層</b>，對照兩路版的 28 次、7 層。放大到 400 個相同的值：兩路要 ' + n2(400) + ' 次，三路只要 400 次。<b>LeetCode 912 的測資裡就有大量重複值</b>，這正是它會不會 TLE 的分水嶺。',
                        en:'<b>' + c + ' comparisons at depth ' + d + '</b>, against 28 comparisons over 7 levels for the two-way version. At 400 equal values it is ' + n2(400) + ' versus 400. <b>LeetCode 912 ships exactly this kind of duplicate-heavy data</b>, and this is the line between accepted and TLE.'})}
};

const CFG_3MIX = {
  arr:[2, 3, 1, 3, 2, 3, 1, 2],
  top:{zh:'LeetCode 912：值域小、重複多 - 三路 partition 的主場',
       en:'LeetCode 912: few distinct values, many duplicates - three-way territory'},
  opening:{note:{zh:'只有三種值，但每種都出現好幾次',
                 en:'three distinct values, each repeated'},
           msg:{zh:'LeetCode 912 要你不用內建 sort 排一個陣列，測資特地放了<b>大量重複值與已排序片段</b>。三路 partition 對這種資料幾乎是線性的：每個相異值只會被當一次 pivot，然後它的所有副本一起退休。',
                en:'LeetCode 912 asks you to sort an array without the built-in sort, and the judge deliberately feeds you <b>heavy duplication and sorted stretches</b>. A three-way partition is nearly linear on that data: each distinct value becomes a pivot once, and then all of its copies retire together.'}},
  pick:(lo, hi, p) => ({zh:'pivot = ' + p + '（實作上要<b>隨機挑</b>，這裡取中間只是為了畫面固定）。',
                        en:'pivot = ' + p + '. A real implementation <b>picks at random</b>; the middle element is used here only to keep the animation reproducible.'}),
  after:(lt, gt, p) => ({zh:'所有的 <b>' + p + '</b> 一次就位（' + (gt - lt + 1) + ' 個）。相異值只有三種，所以整個排序最多三層。',
                         en:'Every <b>' + p + '</b> lands at once (' + (gt - lt + 1) + ' of them). With only three distinct values the whole sort is at most three levels deep.'}),
  done:{note:(c, d) => ({zh:'比較 ' + c + ' 次　·　' + d + ' 層　·　相異值只有 3 種',
                         en:c + ' comparisons  ·  ' + d + ' levels  ·  only 3 distinct values'}),
        msg:(c, d) => ({zh:'重複越多，三路 partition 越快 - 遞迴深度是由<b>相異值的個數</b>決定的，不是元素個數。真正送出去的答案還要再加兩件事：<b>pivot 隨機挑</b>（擋住排好序的測資），以及<b>只對較小的一邊遞迴、較大的一邊用迴圈</b>（把 stack 壓在 O(log n)）。',
                        en:'The more duplicates, the faster it goes: the recursion depth is set by the <b>number of distinct values</b>, not the number of elements. A submission needs two more things on top: <b>pick the pivot at random</b> to defeat sorted inputs, and <b>recurse on the smaller side while looping on the larger one</b> to hold the stack at O(log n).'})}
};

/* ------------------------------------------------------- parallel merging */
const CODE_P = [
'def merge_par(a, b):            # returns (out, work, depth)',
'    if len(a) < len(b): a, b = b, a',
'    if not b: return a, len(a), 1',
'    mid = len(a) // 2',
'    piv = a[mid]',
'    j = bisect_left(b, piv)     # the only sequential bit,',
'    #                             and it costs O(log n)',
'    left  = merge_par(a[:mid],     b[:j])   # these two are',
'    right = merge_par(a[mid+1:],   b[j:])   # fully independent',
'    return (left + [piv] + right,',
'            wl + wr + log2(len(b)),      # work: a sum',
'            max(dl, dr) + log2(len(b)))  # depth: a max'
];

const PA = [1, 4, 6, 9, 12, 15, 18, 21];
const PB = [2, 3, 7, 8, 14, 17, 20];

function bisectLeft(arr, lo, hi, v){
  while (lo < hi){ const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}
const bits = n => Math.max(1, 32 - Math.clz32(n));

/* work / depth of a whole merge sort - computed here, never typed in */
function mergeCost(arr, parallelMerge, parallelRec){
  const mergeSeq = (x, y) => {
    const out = []; let i = 0, j = 0;
    while (i < x.length && j < y.length) out.push(x[i] <= y[j] ? x[i++] : y[j++]);
    while (i < x.length) out.push(x[i++]);
    while (j < y.length) out.push(y[j++]);
    return [out, x.length + y.length, x.length + y.length];
  };
  const mergePar = (x, y) => {
    if (x.length < y.length){ const t = x; x = y; y = t; }
    if (!x.length) return [y.slice(), Math.max(y.length, 1), 1];
    if (!y.length) return [x.slice(), x.length, 1];
    if (x.length === 1) return [mergeSeq(x, y)[0], 2, 1];
    const mid = x.length >> 1, piv = x[mid], c = bits(y.length);
    const j = bisectLeft(y, 0, y.length, piv);
    const L = mergePar(x.slice(0, mid), y.slice(0, j));
    const R = mergePar(x.slice(mid + 1), y.slice(j));
    return [L[0].concat([piv], R[0]), L[1] + R[1] + c + 1, Math.max(L[2], R[2]) + c];
  };
  const go = (v) => {
    if (v.length <= 1) return [v.slice(), 0, 0];
    const m = v.length >> 1;
    const L = go(v.slice(0, m)), R = go(v.slice(m));
    const M = (parallelMerge ? mergePar : mergeSeq)(L[0], R[0]);
    return [M[0], L[1] + R[1] + M[1],
            (parallelRec ? Math.max(L[2], R[2]) : L[2] + R[2]) + M[2]];
  };
  return go(arr);
}

function parFrames(){
  const F = new Frames();
  const AY = 2.7, BY = 4.5;
  const x0 = rowX(PA.length);
  let work = 0, depth = 0;

  const shapes = (tasks, txt, txtC) => {
    const sa = {}, sb = {};
    tasks.forEach((t, k) => {
      const cls = k % 2 ? 'act' : 'soft';
      for (let i = t.a0; i < t.a1; i++) sa[i] = cls;
      for (let i = t.b0; i < t.b1; i++) sb[i] = cls;
      if (t.piv != null) sa[t.piv] = 'hot';
      if (t.pivB != null) sb[t.pivB] = 'hot';
    });
    let out = arrRow(PA, AY, sa, {title:{zh:'左半（已排序）', en:'left (sorted)'}, x0:x0});
    out = out.concat(arrRow(PB, BY, sb, {title:{zh:'右半（已排序）', en:'right (sorted)'}, x0:x0}));
    if (txt) out.push(note(1.35, txt, txtC));
    out.push(S.t(VIEW[0] / 2, 6.15,
      {zh:'同色的一組 = 一個獨立子問題，可以真的同時做',
       en:'each colour is one independent sub-problem - genuinely simultaneous'},
      {c:COL.grey, fs:.28}));
    return out;
  };
  const panels = (round, tasks, extra) => [
    {lbl:{zh:'第幾輪（= depth）', en:'round (= depth)'}, chips:[{t:String(round), cls:'hot'}]},
    {lbl:{zh:'目前獨立的子問題', en:'independent sub-problems'}, chips:[{t:String(tasks), cls:'ok'}]},
    {lbl:{zh:'累積 depth（二分搜尋的代價）', en:'accumulated depth (the binary searches)'},
     chips:[{t:String(depth), cls:'act'}]},
    {lbl:{zh:'目前這一步', en:'this step'}, chips:extra || [{t:'-', cls:'dim'}]}
  ];

  let tasks = [{a0:0, a1:PA.length, b0:0, b1:PB.length}];
  F.push({shapes:shapes(tasks, {zh:'兩段都已排序，要把它們合成一段 - 但不准用「從頭走到尾」的那個迴圈',
                                en:'two sorted halves to merge - but without the loop that walks from one end to the other'}),
          panels:panels(0, 1), view:VIEW, line:0,
          msg:{zh:'普通的 merge 是一個 <b>n 步的序列迴圈</b>：第 k 步要先知道第 k-1 步拿了誰。所以就算左右兩半是平行排好的，最後這一次合併仍然是 O(n) 的 depth - <b>核心再多也縮不短</b>。',
               en:'An ordinary merge is a <b>sequential loop of n steps</b>: step k needs to know what step k-1 emitted. So even if both halves were sorted in parallel, the final merge still has depth O(n) - <b>and no number of cores can shorten it</b>.'}});

  let round = 0;
  while (tasks.length && round < 6){
    round++;
    const next = [];
    let cost = 0, split = 0;
    tasks.forEach(t => {
      let A = [t.a0, t.a1], B = [t.b0, t.b1], onA = true;
      if (A[1] - A[0] < B[1] - B[0]){ const x = A; A = B; B = x; onA = false; }
      if (A[1] - A[0] <= 1) return;
      split++;
      const mid = A[0] + ((A[1] - A[0]) >> 1);
      const src = onA ? PA : PB, oth = onA ? PB : PA;
      const piv = src[mid];
      const c = bits(Math.max(1, B[1] - B[0]));
      cost = Math.max(cost, c); work += c + 1;
      const j = bisectLeft(oth, B[0], B[1], piv);
      const mk = (aa, bb) => onA ? {a0:aa[0], a1:aa[1], b0:bb[0], b1:bb[1]}
                                 : {a0:bb[0], a1:bb[1], b0:aa[0], b1:aa[1]};
      next.push(mk([A[0], mid], [B[0], j]));
      next.push(mk([mid + 1, A[1]], [j, B[1]]));
      if (onA) t.piv = mid; else t.pivB = mid;
    });
    if (!split) break;
    depth += cost;
    F.push({shapes:shapes(tasks, {zh:'第 ' + round + ' 輪：' + split + ' 個子問題各做一次二分搜尋（橘色 = 被選中的中位數）',
                                  en:'round ' + round + ': ' + split + ' sub-problems each do one binary search (orange = the chosen median)'}),
            panels:panels(round, tasks.length,
              [{t:'+' + cost + ' depth', cls:'hot'}, {t:'work ' + work, cls:'ok'}]),
            view:VIEW, line:5,
            msg:round === 1
              ? {zh:'取比較長那一段的<b>中位數</b>，在另一段裡<b>二分搜尋</b>它該插在哪。這條線一畫下去，左半對左半、右半對右半就<b>互不相干</b> - 兩邊可以同時合併。代價只有一次 log 的二分搜尋。',
                 en:'Take the <b>median</b> of the longer half and <b>binary-search</b> for where it belongs in the other. That single cut makes left-with-left and right-with-right <b>completely independent</b>, so both can be merged at the same time. The price is one logarithmic binary search.'}
              : {zh:'同一招再做一次，但這一輪的 <b>' + split + ' 個子問題是同時做的</b>：所以 depth 只加一次二分搜尋的代價（<b>取 max 而不是相加</b>），work 卻加了 ' + split + ' 份。這就是 work 與 depth 分家的地方。',
                 en:'The same move again, except this round’s <b>' + split + ' sub-problems all happen at once</b>: depth grows by a single binary search (<b>a max, not a sum</b>) while work grows by ' + split + ' of them. This is precisely where work and depth part company.'}});
    tasks = next.filter(t => (t.a1 - t.a0) + (t.b1 - t.b0) > 0);
  }

  const m = 64;
  const src = [];
  for (let i = 0; i < m; i++) src.push((i * 37) % m);
  const c1 = mergeCost(src, false, false);
  const c2 = mergeCost(src, false, true);
  const c3 = mergeCost(src, true, true);
  const rows = [[{zh:'一顆核心', en:'one core'}, c1],
                [{zh:'平行遞迴 + 普通 merge', en:'parallel halves, plain merge'}, c2],
                [{zh:'平行遞迴 + 平行 merge', en:'parallel halves, parallel merge'}, c3]];
  const table = () => {
    const out = [S.t(VIEW[0] / 2, 1.05,
      {zh:'整個 merge sort，n = ' + m + '（下面的數字由這一頁的程式現算）',
       en:'a whole merge sort, n = ' + m + ' (these numbers are computed on this page)'},
      {c:COL.tealL, fs:.32})];
    out.push(S.t(4.55, 1.85, {zh:'work', en:'work'}, {c:COL.grey, fs:.30}));
    out.push(S.t(6.05, 1.85, {zh:'depth', en:'depth'}, {c:COL.grey, fs:.30}));
    out.push(S.t(7.75, 1.85, {zh:'用得上幾顆核心', en:'usable cores'}, {c:COL.grey, fs:.30}));
    rows.forEach((r, k) => {
      const y = 2.15 + k * .95;
      out.push(S.r(.55, y, 3.35, .74, k === 2 ? 'ok' : 'soft', r[0], {fs:.30, rx:.08}));
      out.push(S.r(4.00, y, 1.10, .74, 'idle', String(r[1][1]), {fs:.32, rx:.08}));
      out.push(S.r(5.50, y, 1.10, .74, 'idle', String(r[1][2]), {fs:.32, rx:.08}));
      out.push(S.r(7.00, y, 1.50, .74, k === 2 ? 'hot' : 'idle',
                   (r[1][1] / r[1][2]).toFixed(1) + 'x', {fs:.32, rx:.08}));
    });
    return out;
  };
  F.push({shapes:table(), panels:panels(round, 0,
            [{t:'seq depth ' + c2[2], cls:'bad'}, {t:'par depth ' + c3[2], cls:'ok'}]),
          view:VIEW, line:11,
          msg:{zh:'把整個 merge sort 的兩筆帳算出來：<b>中間那一列才是陷阱</b> - 左右遞迴平行了，但每一次 merge 還是序列的，depth 只從 ' + c1[2] + ' 掉到 ' + c2[2] + '。換成平行 merge 之後 depth 是 <b>' + c3[2] + '</b>，用得上的核心數從 ' + (c2[1] / c2[2]).toFixed(1) + ' 變成 <b>' + (c3[1] / c3[2]).toFixed(1) + '</b>，代價是 work 多了 ' + Math.round(100 * (c3[1] - c2[1]) / c2[1]) + '%。',
               en:'Both bills for a whole merge sort. <b>The middle row is the trap</b>: the recursion is parallel but every merge is still sequential, so the depth only falls from ' + c1[2] + ' to ' + c2[2] + '. With a parallel merge the depth is <b>' + c3[2] + '</b> and the usable core count goes from ' + (c2[1] / c2[2]).toFixed(1) + ' to <b>' + (c3[1] / c3[2]).toFixed(1) + '</b>, paid for with ' + Math.round(100 * (c3[1] - c2[1]) / c2[1]) + '% more work.'}});

  F.push({shapes:table(), panels:panels(round, 0, [{t:'work / depth', cls:'act'}]),
          view:VIEW, line:11,
          msg:{zh:'記住這兩個字就好：<b>work</b> 是總共要做的事，<b>depth</b> 是最長的一條相依鏈。work 決定電費，depth 決定你就算買下整座機房也還是要等多久 - <b>work/depth 就是這個演算法最多用得上幾顆核心</b>。之後談 GPU kernel、談排程，用的都是同一把尺。',
               en:'Two words are enough: <b>work</b> is everything that must be done, <b>depth</b> is the longest chain of dependencies. Work sets the electricity bill; depth sets how long you wait even after buying the entire data centre - and <b>work/depth is the most cores this algorithm can ever use</b>. The same ruler comes back for GPU kernels and for schedulers later in the series.'}});
  return F.list;
}

/* ------------------------------------------------------------------ meta */
const DAY_META = {
  title:{zh:'Day 20 · Merge sort、Quick sort 與它們的平行版',
         en:'Day 20 · Merge sort, quick sort, and going parallel'},
  sub:{zh:'兩個都是分治，差別只有一句話：merge sort 把力氣花在「合」，quick sort 花在「切」。穩定性、最壞情況、外部排序、平行度，全部從這一句推出來。',
       en:'Two divide-and-conquer sorts. The only difference: merge sort puts the work in the join, quick sort puts it in the split. Stability, the worst case, external sorting and parallelism all fall out of that one line.'},
  tabs:[
    {id:'merge', label:{zh:'Merge sort', en:'Merge sort'},
     stage:{zh:'切一半不用動腦，力氣全在合併', en:'the split is free, the join does everything'},
     view:VIEW,
     idea:{zh:'merge 是兩根手指往前走：每比一次就確定一個輸出位置。因為切法完全不看資料，樹的形狀對任何輸入都一樣 - <b>沒有最壞情況</b>。比較用 <b>&le;</b> 而不是 <b>&lt;</b>，相等時先拿左邊，這就是穩定性；而「從右邊拿一個」的那一刻順手就能數出 inversion。',
           en:'The merge is two fingers walking forward: every comparison fixes one output slot. Because the split never looks at the data, the tree has the same shape for every input - <b>there is no bad case</b>. Comparing with <b>&le;</b> rather than <b>&lt;</b> sends ties left, and that is stability; and the moment you take from the right half you can count inversions for free.'},
     legend:['hot', 'act', 'soft', 'ok'],
     code:CODE_M,
     build:() => mergeFrames()},
    {id:'quick', label:{zh:'Quick sort', en:'Quick sort'},
     stage:{zh:'partition 做完，pivot 就永遠不動了',
            en:'once the partition is done the pivot never moves'},
     view:VIEW,
     idea:{zh:'Lomuto partition 維持一個不變式：<b>a[lo:i] 全都小於 pivot</b>。掃完一遍把 pivot 換到 i，它就落在最終位置，左右兩段再也不會互相影響 - 所以 quick sort 沒有合併步驟，也不需要暫存陣列。',
           en:'Lomuto’s partition holds one invariant: <b>everything in a[lo:i] is smaller than the pivot</b>. After the scan the pivot swaps into position i, which is its final home, and the two sides can never affect each other again - so quick sort has no join step and needs no scratch array.'},
     legend:['hot', 'act', 'soft', 'ok'],
     code:CODE_Q,
     build:() => quickFrames(CFG_OK)},
    {id:'worst', label:{zh:'兩種會炸掉的輸入', en:'the two inputs that break it'},
     stage:{zh:'不會報錯，只會從 n log n 掉到 n²',
            en:'nothing raises - it just falls from n log n to n squared'},
     view:VIEW,
     variants:[{zh:'已排序 + pivot 取最後一個', en:'sorted + last-element pivot'},
               {zh:'全部相同 + 隨機 pivot', en:'all equal + random pivot'}],
     idea:{zh:'快不快取決於切得平不平均。<b>已排序</b>的輸入配上固定 pivot 會切成 n-1 / 0，一行隨機化就能解決；但<b>全部相同</b>的輸入連隨機 pivot 都救不了，因為兩路 partition 根本沒有「等於」這一區。兩種情況都不會拋例外 - 它們只會安靜地變慢。',
           en:'Speed depends on the splits being even. A <b>sorted</b> input with a fixed pivot splits n-1 / 0, and one line of randomisation fixes it. An <b>all-equal</b> input defeats the random pivot too, because a two-way partition has no "equal" region at all. Neither case raises an exception - they just quietly get slow.'},
     legend:['hot', 'act', 'soft', 'bad'],
     code:CODE_Q,
     build:(v) => quickFrames(v ? CFG_EQ : CFG_SORTED)},
    {id:'dutch', label:{zh:'三路 partition / LC 912', en:'three-way / LC 912'},
     stage:{zh:'給「等於 pivot」一個自己的區間',
            en:'give "equal to the pivot" a region of its own'},
     view:VIEW,
     variants:[{zh:'全部相同', en:'all equal'}, {zh:'重複很多的資料', en:'duplicate-heavy data'}],
     idea:{zh:'荷蘭國旗問題：一次掃描維持 <b>&lt; p ｜ == p ｜ &gt; p</b> 三個區間。中間那一區直接退休，遞迴深度因此由<b>相異值的個數</b>決定而不是元素個數。最容易寫錯的一行是「跟右端交換之後 <b>i 不可以前進</b>」，因為換過來的值還沒被看過。',
           en:'The Dutch national flag problem: one scan maintaining <b>&lt; p | == p | &gt; p</b>. The middle region retires immediately, so the recursion depth is set by the <b>number of distinct values</b> rather than the number of elements. The easiest line to get wrong: after swapping with the right end, <b>i must not advance</b>, because the value swapped in has not been examined yet.'},
     legend:['hot', 'act', 'soft', 'ok'],
     code:CODE_3,
     build:(v) => threeFrames(v ? CFG_3MIX : CFG_3EQ)},
    {id:'par', label:{zh:'平行 merge：work 與 depth', en:'parallel merge: work vs depth'},
     stage:{zh:'把 merge 本身也做成分治', en:'make the merge itself divide and conquer'},
     view:VIEW,
     idea:{zh:'「左右各丟一顆核心」看起來就是平行版了，其實不是：最後那一次 n 個元素的 merge 是一個序列迴圈，depth 還是 O(n)。真正的做法是把 merge 也切開 - 取長邊的中位數、在短邊二分搜尋，兩側就完全獨立。<b>work 幾乎沒變，depth 從 n 掉到 log² n。</b>',
           en:'"Give each half a core" looks like the parallel version but is not: the final merge of n elements is a sequential loop, so the depth is still O(n). The real move is to split the merge as well - take the median of the longer side and binary-search it in the shorter one, and the two sides become independent. <b>Work barely changes; depth falls from n to log squared n.</b>'},
     legend:['hot', 'act', 'soft', 'ok'],
     code:CODE_P,
     build:() => parFrames()}
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
