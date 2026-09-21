// DAY: 35
// TITLE_ZH: Day 35 — LIS 與 LCS：最長遞增子序列與最長共同子序列
// TITLE_EN: LIS and LCS - the longest subsequence you keep, and the table you throw away
// SUB_ZH: 兩題常常被放在一起講，但只有一題有捷徑。LIS 可以把 O(n²) 的表整張刪掉，換成一個 tails 陣列加上 Day 23 的 lower_bound；LCS 沒有這種好事——除非兩邊互為排列，那時 LCS 其實就是 LIS。
// SUB_EN: Two problems that are always mentioned together, and only one of them has a shortcut. LIS lets you delete the quadratic table outright and replace it with a tails array plus the lower_bound from day 23. LCS has no such trick - unless the two inputs are permutations of one another, in which case LCS *is* LIS.
// FOLDER: day%2035%20-%20lis%20and%20lcs
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }

const A = [10, 9, 2, 5, 3, 7, 101, 18];
const TRAP = [2, 6, 8, 3, 4, 5, 1];

function lowerBound(arr, x){
  let lo = 0, hi = arr.length;
  while (lo < hi){ const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}
function upperBound(arr, x){
  let lo = 0, hi = arr.length;
  while (lo < hi){ const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

/* ======================================================================== *
 * Tab 1 - patience sorting, and the reconstruction it needs
 * ======================================================================== */
const CODE_LIS = [
  'def lis_patience(a):',
  '    tails = []          # tails[k] = smallest end of a length-k+1 run',
  '    parent, idx = [-1] * len(a), []',
  '    for i, x in enumerate(a):',
  '        k = bisect_left(tails, x)   # day 23 lower_bound',
  '        if k == len(tails):',
  '            tails.append(x); idx.append(i)   # a longer run',
  '        else:',
  '            tails[k] = x;    idx[k] = i      # a cheaper ending',
  '        parent[i] = idx[k - 1] if k else -1',
  '    return len(tails)   # the LENGTH is here',
  '',
  '# the subsequence itself comes from the parents, not from tails',
  'i = idx[-1]',
  'while i != -1: out.append(a[i]); i = parent[i]'
];

const AX0 = 1.35, AW = 0.86, AH = 0.62;

function lisShapes(a, i, tails, tst, extra, real){
  const ast = {};
  for (let j = 0; j < a.length; j++) ast[j] = j < i ? 'done' : (j === i ? 'hot' : 'idle');
  let out = cellRow(a, AX0, 5.05, AW, AH, {states:ast, index:false, fs:.32,
                                           title:'a'});
  out = out.concat(cellRow(tails.length ? tails : [''], AX0, 3.35, AW, AH,
    {states:tst || {}, index:false, fs:.32, title:'tails'}));
  if (real)
    out = out.concat(cellRow(real, AX0, 1.85, AW, AH,
      {states:real.map(() => 'ok').reduce((o, s, k) => (o[k] = s, o), {}),
       index:false, fs:.32, title:'LIS'}));
  return out.concat(extra || []);
}

function lisFrames(v){
  const a = v ? TRAP : A;
  const F = new Frames();
  const tails = [], idx = [], parent = new Array(a.length).fill(-1);

  F.push({shapes:lisShapes(a, -1, [], {}, [
      S.t(4.9, .70, {zh:'tails[k] ＝ 長度 k+1 的遞增子序列裡最小的結尾',
                     en:'tails[k] = the smallest possible ending of a length-k+1 run'},
          {c:COL.tealL, fs:.32})]),
    view:VIEW, line:1,
    panels:[{lbl:{zh:'輸入', en:'input'}, chips:[chip(a.join(', '), 'dim')]},
            {lbl:{zh:'tails', en:'tails'}, chips:[]}],
    msg:{zh:'tails 一開始是空的。它不是答案，是一張記分板：每一格記「要湊出這個長度，結尾最小可以小到多少」。因為「越小的結尾越容易被接下去」，所以每一格都只留最小的那個。',
         en:'tails starts empty. It is not the answer but a scoreboard: slot k records how small the ending of a length-k+1 run can be. A smaller ending is always easier to extend, so each slot only ever keeps the smallest one seen.'}});

  for (let i = 0; i < a.length; i++){
    const x = a[i], k = lowerBound(tails, x);
    const st = {};
    for (let j = 0; j < tails.length; j++) st[j] = 'idle';
    if (k < tails.length) st[k] = 'act';
    F.push({shapes:lisShapes(a, i, tails.slice(), st, [
        S.t(4.9, 2.55, (k < tails.length
            ? 'bisect_left(tails, ' + x + ') = ' + k + '  ->  tails[' + k + '] = ' + tails[k] + ' >= ' + x
            : 'bisect_left(tails, ' + x + ') = ' + k + '  ->  past the end'),
          {c:COL.purpleL, fs:.36})]),
      view:VIEW, line:4,
      panels:[{lbl:{zh:'這一步的值', en:'value being placed'}, chips:[chip(String(x), 'hot')]},
              {lbl:{zh:'tails', en:'tails'}, chips:tails.length ? tails.map(t => chip(String(t), 'dim')) : []}],
      msg:{zh:'tails 天生就是排序好的，所以找位置是一次二分搜尋——就是 Day 23 的 lower_bound：第一個「不小於 ' + x + '」的位置。' +
             (k < tails.length ? '答案是 ' + k + '，那一格現在放著 ' + tails[k] + '。' : '答案落在尾端後面，代表沒有任何一格擋得住它。'),
           en:'tails is sorted by construction, so finding the slot is one binary search - the lower_bound of day 23: the first position not smaller than ' + x + '. ' +
             (k < tails.length ? 'That is slot ' + k + ', which currently holds ' + tails[k] + '.' : 'It lands past the end, which means no slot can hold it back.')}});

    const appended = (k === tails.length);
    if (appended){ tails.push(x); idx.push(i); } else { tails[k] = x; idx[k] = i; }
    parent[i] = k ? idx[k - 1] : -1;
    const st2 = {};
    for (let j = 0; j < tails.length; j++) st2[j] = 'idle';
    st2[k] = appended ? 'ok' : 'hot';
    F.push({shapes:lisShapes(a, i, tails.slice(), st2, [
        S.t(4.9, 2.55, appended
            ? 'tails.append(' + x + ')   ->  length ' + tails.length
            : 'tails[' + k + '] = ' + x + '   ->  length still ' + tails.length,
          {c:appended ? COL.tealL : COL.orangeL, fs:.36})]),
      view:VIEW, line:appended ? 6 : 8,
      panels:[{lbl:{zh:'這一步的值', en:'value being placed'}, chips:[chip(String(x), 'hot')]},
              {lbl:{zh:'tails', en:'tails'}, chips:tails.map(t => chip(String(t), 'dim'))},
              {lbl:{zh:'目前長度', en:'length so far'}, chips:[chip(String(tails.length), 'ok')]}],
      msg:appended
        ? {zh:'沒有任何一格擋得住它，代表 ' + x + ' 可以接在目前最長的那條後面，長度加一。整個過程中，tails 只會在這種時候變長——所以最後的長度就是答案。',
           en:'Nothing could hold it back, so ' + x + ' extends the longest run seen so far and the length grows by one. tails only ever grows at moments like this, which is why its final length is the answer.'}
        : {zh:'長度沒有變，但第 ' + k + ' 格的結尾從 ' + '更大的值換成了 ' + x + '。這是整個演算法唯一在做的事：同樣長度的子序列，留結尾最小的那一個，因為它以後最好接。',
           en:'The length does not change, but slot ' + k + ' now ends at ' + x + ' instead of something larger. That swap is the only thing this algorithm ever does: among runs of equal length, keep the one with the smallest ending, because it is the easiest to extend later.'}});
  }

  /* reconstruct */
  const out = [];
  let p = idx.length ? idx[idx.length - 1] : -1;
  while (p !== -1){ out.push(a[p]); p = parent[p]; }
  out.reverse();
  const sameAsTails = out.join(',') === tails.join(',');
  F.push({shapes:lisShapes(a, a.length, tails.slice(),
      tails.map(() => 'soft').reduce((o, s, k) => (o[k] = s, o), {}), [
      S.t(4.9, 2.62, {zh:'tails 的長度永遠正確，內容通常不是答案',
                      en:'the length of tails is right, its contents usually are not'},
          {c:COL.orangeL, fs:.30}),
      S.t(4.9, .70, sameAsTails
        ? {zh:'這一組剛好一樣，但那是運氣；換一組就會不同', en:'they match here, but that is luck - try the other input'}
        : {zh:'tails 裡的 ' + tails[0] + ' 是 a 的最後一個元素，這串不是子序列',
           en:'tails[0] = ' + tails[0] + ' is the LAST element of a - not a subsequence'},
        {c:sameAsTails ? COL.grey : COL.red, fs:.32})], out),
    view:VIEW, line:14,
    panels:[{lbl:{zh:'長度', en:'length'}, chips:[chip(String(tails.length), 'ok')]},
            {lbl:{zh:'tails', en:'tails'}, chips:tails.map(t => chip(String(t), 'dim'))},
            {lbl:{zh:'真正的 LIS', en:'the real LIS'}, chips:out.map(t => chip(String(t), 'ok'))}],
    msg:{zh:'插入的時候順手記下「我接在誰後面」，最後從最長那條的結尾往回走，才拿得到真正的子序列。' +
           (sameAsTails ? '' : '這一組把陷阱放大了：tails 是 [' + tails.join(', ') + ']，但 ' + tails[0] + ' 在輸入裡是最後才出現的，' +
            '它不可能排在 ' + tails[1] + ' 前面——tails 只是每一格各自的最佳結尾，彼此之間沒有先後關係。'),
         en:'Recording "who did I land behind" at insert time, then walking back from the end of the longest run, is what actually produces the subsequence. ' +
           (sameAsTails ? '' : 'This input makes the trap obvious: tails is [' + tails.join(', ') + '], but ' + tails[0] + ' is the *last* element of the input and cannot possibly come before ' + tails[1] + '. tails holds the best ending for each length independently; those endings never had to coexist.')}});
  return F.list;
}

/* ======================================================================== *
 * Tab 2 - bisect_left vs bisect_right: one character, two problems
 * ======================================================================== */
const CODE_STRICT = [
  '# strictly increasing            # non-decreasing',
  'k = bisect_left(tails, x)        k = bisect_right(tails, x)',
  '',
  '# bisect_left  refuses to sit after an equal value',
  '#   -> an equal x overwrites the SAME slot, length unchanged',
  '# bisect_right steps past equals',
  '#   -> an equal x opens a NEW slot, length grows',
  'if k == len(tails): tails.append(x)',
  'else:               tails[k] = x'
];

const DUP = [1, 3, 3, 3, 5];

function strictFrames(v){
  const strict = (v === 0);
  const F = new Frames();
  const tails = [];
  F.push({shapes:lisShapes(DUP, -1, [], {}, [
      S.t(4.9, 2.55, strict ? 'bisect_left(tails, x)' : 'bisect_right(tails, x)',
          {c:COL.purpleL, fs:.42}),
      S.t(4.9, .70, strict
        ? {zh:'嚴格遞增：相等的值不算「變大」', en:'strictly increasing: equal is not an increase'}
        : {zh:'非遞減：相等的值也可以接下去', en:'non-decreasing: equal values may be appended'},
        {c:COL.tealL, fs:.32})]),
    view:VIEW, line:1,
    panels:[{lbl:{zh:'輸入', en:'input'}, chips:[chip(DUP.join(', '), 'dim')]}],
    msg:{zh:'同一個陣列 [1, 3, 3, 3, 5]，同一份程式碼，只把 bisect_left 換成 bisect_right。演算法的其他部分一個字都不用改，答案卻從 3 變成 5。',
         en:'The same array [1, 3, 3, 3, 5] and the same code, with bisect_left swapped for bisect_right. Not another line changes, and the answer moves from 3 to 5.'}});

  for (let i = 0; i < DUP.length; i++){
    const x = DUP[i];
    const k = strict ? lowerBound(tails, x) : upperBound(tails, x);
    const appended = (k === tails.length);
    const before = tails.slice();
    if (appended) tails.push(x); else tails[k] = x;
    const st = {};
    for (let j = 0; j < tails.length; j++) st[j] = 'idle';
    st[k] = appended ? 'ok' : 'hot';
    const equal = (k < before.length && before[k] === x);
    F.push({shapes:lisShapes(DUP, i, tails.slice(), st, [
        S.t(4.9, 2.55, (strict ? 'bisect_left' : 'bisect_right') + '(tails, ' + x + ') = ' + k +
            '  ->  ' + (appended ? 'append (length ' + tails.length + ')'
                                 : 'overwrite slot ' + k),
          {c:appended ? COL.tealL : COL.orangeL, fs:.34})]),
      view:VIEW, line:appended ? 7 : 8,
      panels:[{lbl:{zh:'這一步的值', en:'value'}, chips:[chip(String(x), 'hot')]},
              {lbl:{zh:'tails', en:'tails'}, chips:tails.map(t => chip(String(t), 'dim'))},
              {lbl:{zh:'長度', en:'length'}, chips:[chip(String(tails.length), appended ? 'ok' : 'dim')]}],
      msg:equal
        ? (strict
          ? {zh:'關鍵的一步：tails[' + k + '] 本來就是 ' + x + '，bisect_left 不肯站到相等值的後面，所以它回到同一格，把 ' + x + ' 蓋回 ' + x + '——什麼都沒發生，長度不動。相等不算遞增，這就是「嚴格」兩個字的全部實作。',
             en:'The decisive step: tails[' + k + '] already holds ' + x + ', and bisect_left refuses to sit after an equal value, so it returns that same slot and writes ' + x + ' over ' + x + '. Nothing happens and the length stays put. Equal is not an increase, and that is the entire implementation of the word "strictly".'}
          : {zh:'關鍵的一步：tails[' + (k - 1) + '] 也是 ' + x + '，但 bisect_right 會跨過相等值，落在後面一格，於是開出新的長度。相等的值被當成可以接下去，答案因此變長。',
             en:'The decisive step: tails[' + (k - 1) + '] is also ' + x + ', but bisect_right steps past equal values and lands one slot further along, opening a new length. Equal counts as extendable, so the answer grows.'})
        : {zh:(appended ? x + ' 比所有結尾都大，直接加長。' : x + ' 換掉第 ' + k + ' 格的結尾。') + '這一步兩種寫法的行為完全一樣——差別只在遇到相等值的時候。',
           en:(appended ? x + ' is larger than every ending, so it extends the run. ' : x + ' replaces the ending in slot ' + k + '. ') +
              'Both versions behave identically here; they only diverge on equal values.'}});
  }
  F.push({shapes:lisShapes(DUP, DUP.length, tails.slice(),
      tails.map(() => 'ok').reduce((o, s, k) => (o[k] = s, o), {}), [
      S.t(4.9, 2.55, (strict ? 'strictly increasing: 3' : 'non-decreasing: 5'),
          {c:COL.tealL, fs:.42}),
      S.t(4.9, .70, {zh:'面試時先問清楚：要嚴格遞增還是非遞減',
                     en:'ask first: strictly increasing, or non-decreasing?'},
          {c:COL.orangeL, fs:.32})]),
    view:VIEW, line:1,
    panels:[{lbl:{zh:'答案', en:'answer'}, chips:[chip(String(tails.length), 'ok')]}],
    msg:strict
      ? {zh:'三個 3 只佔一格，答案是 3（例如 1, 3, 5）。LeetCode 300 要的就是這個版本。',
         en:'The three 3s share one slot, so the answer is 3 - for instance 1, 3, 5. This is the version LeetCode 300 asks for.'}
      : {zh:'三個 3 各佔一格，答案是 5（1, 3, 3, 3, 5 整串）。同樣一個陣列、同樣一份程式碼，一個函式名之差。',
         en:'Each 3 gets its own slot and the answer is 5 - the whole array 1, 3, 3, 3, 5. Same array, same code, one function name apart.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 3 - LeetCode 354: the tie-break is the whole problem
 * ======================================================================== */
const CODE_354 = [
  'def max_envelopes(envelopes):',
  '    # width ascending, height DESCENDING inside equal widths',
  '    envelopes.sort(key=lambda e: (e[0], -e[1]))',
  '    return lis_patience([h for _, h in envelopes])',
  '',
  '# with (e[0], e[1]) instead, two envelopes of the same width',
  '# line up as an increasing pair of heights and get "nested"',
  '# - which is exactly what the problem forbids.'
];

const ENV = [[1, 1], [1, 2], [1, 3], [2, 4], [3, 5]];
const EX0 = 2.00, EWD = 1.28;

function envShapes(order, i, tails, tst, extra){
  const st = {};
  for (let j = 0; j < order.length; j++) st[j] = j < i ? 'done' : (j === i ? 'hot' : 'idle');
  let out = cellRow(order.map(e => e[0] + '×' + e[1]), EX0, 4.95, EWD, .66,
    {states:st, index:false, fs:.30, title:{zh:'排序後 w×h', en:'sorted w×h'}});
  out = out.concat(cellRow(order.map((e, j) => (j <= i ? e[1] : '')), EX0, 3.75, EWD, .60,
    {states:st, index:false, fs:.30, title:{zh:'高度', en:'heights'}}));
  out = out.concat(cellRow(tails.length ? tails : [''], EX0, 2.35, EWD, .60,
    {states:tst || {}, index:false, fs:.30, title:'tails'}));
  return out.concat(extra || []);
}

function envFrames(v){
  const desc = (v === 0);
  const order = ENV.slice().sort((a, b) => a[0] - b[0] || (desc ? b[1] - a[1] : a[1] - b[1]));
  const F = new Frames();
  const tails = [];

  F.push({shapes:envShapes(order, -1, [], {}, [
      S.t(4.9, 5.95, desc ? 'sort(key=lambda e: (e[0], -e[1]))'
                          : 'sort(key=lambda e: (e[0], e[1]))',
          {c:desc ? COL.tealL : COL.red, fs:.38}),
      S.t(4.9, .70, {zh:'原始資料：1×1, 1×2, 1×3, 2×4, 3×5（正確答案是 3）',
                     en:'envelopes 1x1, 1x2, 1x3, 2x4, 3x5 - the true answer is 3'},
          {c:COL.grey, fs:.30})]),
    view:VIEW, line:2,
    panels:[{lbl:{zh:'排序方式', en:'sort order'},
             chips:[chip(desc ? 'w asc, h desc' : 'w asc, h asc', desc ? 'ok' : 'bad')]},
            {lbl:{zh:'排序結果', en:'sorted'}, chips:order.map(e => chip(e[0] + 'x' + e[1], 'dim'))}],
    msg:desc
      ? {zh:'寬度一樣的信封不可能互相裝進去。把等寬的那一組高度「由大排到小」，就讓它們在高度序列裡永遠是遞減的，遞增子序列因此不可能同時挑到兩個等寬的信封——限制被編碼進排序，而不是寫成 if。',
         en:'Envelopes of equal width can never nest. Sorting the heights of each equal-width group in *descending* order makes them a decreasing run, so an increasing subsequence can never pick two of them. The constraint is encoded in the sort rather than written as an if.'}
      : {zh:'如果第二個 key 也用遞增排，等寬的 1×1、1×2、1×3 在高度序列裡就變成 1, 2, 3——一段漂亮的遞增。演算法會很開心地把它們當成三層俄羅斯娃娃。',
         en:'If the second key also sorts ascending, the equal-width 1x1, 1x2 and 1x3 become 1, 2, 3 in the height sequence - a perfectly good increasing run. The algorithm will happily nest all three.'}});

  for (let i = 0; i < order.length; i++){
    const h = order[i][1], k = lowerBound(tails, h);
    const appended = (k === tails.length);
    if (appended) tails.push(h); else tails[k] = h;
    const st = {};
    for (let j = 0; j < tails.length; j++) st[j] = 'idle';
    st[k] = appended ? 'ok' : 'hot';
    const sameW = (i > 0 && order[i - 1][0] === order[i][0]);
    F.push({shapes:envShapes(order, i, tails.slice(), st, [
        S.t(4.9, 1.55, 'h = ' + h + '  ->  ' + (appended ? 'tails.append  (length ' + tails.length + ')'
                                                          : 'tails[' + k + '] = ' + h + '  (length ' + tails.length + ')'),
          {c:appended ? COL.tealL : COL.orangeL, fs:.36})]),
      view:VIEW, line:3,
      panels:[{lbl:{zh:'目前信封', en:'envelope'}, chips:[chip(order[i][0] + 'x' + order[i][1], 'hot')]},
              {lbl:{zh:'tails（高度）', en:'tails (heights)'}, chips:tails.map(t => chip(String(t), 'dim'))},
              {lbl:{zh:'目前層數', en:'layers so far'}, chips:[chip(String(tails.length), appended ? 'ok' : 'dim')]}],
      msg:sameW
        ? (desc
          ? {zh:'這一個和前一個寬度一樣（都是 ' + order[i][0] + '）。因為高度是由大排到小，' + h + ' 比前一個矮，只會覆蓋掉某一格，不會讓長度變長——等寬的兩個永遠不會被同時選中。',
             en:'This envelope has the same width as the previous one (' + order[i][0] + '). Because heights descend inside a width group, ' + h + ' is shorter than the last one and can only overwrite a slot, never extend the run - so two equal-width envelopes can never both be chosen.'}
          : {zh:'這一個和前一個寬度一樣（都是 ' + order[i][0] + '），高度卻更高，於是長度被加一。錯誤就在這裡發生，而且沒有任何人抗議：寬度相等根本裝不進去。',
             en:'Same width as the previous envelope (' + order[i][0] + '), but a taller height, so the run grows by one. This is the bug, and nothing complains about it: equal widths cannot nest at all.'})
        : {zh:(appended ? '寬度變大了，高度 ' + h + ' 也比目前所有結尾都高，真的可以多包一層。'
                        : '高度 ' + h + ' 接不上更長的那條，只能把第 ' + k + ' 格換成比較矮的結尾。'),
           en:(appended ? 'The width has grown and the height ' + h + ' beats every current ending, so this really is one more layer. '
                        : 'Height ' + h + ' cannot extend the longest run, so it just makes slot ' + k + ' cheaper to extend later.')}});
  }

  F.push({shapes:envShapes(order, order.length, tails.slice(),
      tails.map(() => (desc ? 'ok' : 'bad')).reduce((o, s, k) => (o[k] = s, o), {}), [
      S.t(4.9, 1.55, (desc ? 'answer 3   (correct)' : 'answer ' + tails.length + '   (should be 3)'),
          {c:desc ? COL.tealL : COL.red, fs:.42}),
      S.t(4.9, .70, desc
        ? {zh:'1×1 ⊂ 2×4 ⊂ 3×5', en:'1x1 inside 2x4 inside 3x5'}
        : {zh:'它宣稱 1×1 ⊂ 1×2 ⊂ 1×3 —— 三個一樣寬的信封', en:'it claims 1x1 in 1x2 in 1x3 - three identical widths'},
        {c:desc ? COL.grey : COL.red, fs:.32})]),
    view:VIEW, line:desc ? 3 : 6,
    panels:[{lbl:{zh:'答案', en:'answer'}, chips:[chip(String(tails.length), desc ? 'ok' : 'bad')]}],
    msg:desc
      ? {zh:'三層，而且每一層寬高都嚴格變大。注意整段程式沒有一行在檢查「寬度是否相等」——那個條件完全靠排序的第二個 key 處理掉了。',
         en:'Three layers, each strictly larger in both dimensions. Notice that no line of code ever checks whether two widths are equal: that condition was handled entirely by the second sort key.'}
      : {zh:'答案變成 ' + tails.length + '，而且錯得很安靜——沒有例外、沒有警告，只是多了兩層不可能存在的娃娃。這類 bug 在面試裡最常見：排序的第二個 key 寫反了，測資小的時候還會通過。',
         en:'The answer becomes ' + tails.length + ', and it fails quietly: no exception, no warning, just two layers of dolls that cannot exist. This is the classic interview failure - the second sort key points the wrong way, and small test cases still pass.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 4 - the LCS table, filled and then walked backwards
 * ======================================================================== */
const CODE_LCS = [
  'def lcs(x, y):',
  '    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]',
  '    for i in range(1, len(x) + 1):',
  '        for j in range(1, len(y) + 1):',
  '            if x[i - 1] == y[j - 1]:',
  '                dp[i][j] = dp[i - 1][j - 1] + 1   # take both',
  '            else:',
  '                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])',
  '    return dp',
  '',
  '# walk back from the corner to recover the string itself',
  'if x[i - 1] == y[j - 1]: out.append(x[i - 1]); i -= 1; j -= 1',
  'elif dp[i - 1][j] >= dp[i][j - 1]: i -= 1',
  'else: j -= 1'
];

const X = 'AGGTAB', Y = 'GXTXAYB';
const GX0 = 2.05, GW = 0.74, GH = 0.60, GY0 = 1.05;

function gridShapes(dp, filled, st, extra){
  const out = [];
  for (let c = 0; c <= Y.length; c++)
    out.push(S.t(GX0 + c * GW + (GW - .06) / 2, GY0 + (X.length + 1) * GH + .34,
      c === 0 ? '-' : Y[c - 1], {c:COL.grey, fs:.32}));
  for (let r = 0; r <= X.length; r++){
    const y = GY0 + (X.length - r) * GH;
    out.push(S.t(GX0 - .26, y + GH * .62, r === 0 ? '-' : X[r - 1],
      {c:COL.tealL, fs:.32, anchor:'end'}));
    const vals = [], rst = {};
    for (let c = 0; c <= Y.length; c++){
      vals.push(filled[r + ':' + c] ? dp[r][c] : '');
      if (st[r + ':' + c]) rst[c] = st[r + ':' + c];
    }
    out.push.apply(out, cellRow(vals, GX0, y, GW, GH,
      {states:rst, index:false, fs:.30}));
  }
  return out.concat(extra || []);
}

function lcsFrames(){
  const F = new Frames();
  const dp = [], filled = {};
  for (let r = 0; r <= X.length; r++) dp.push(new Array(Y.length + 1).fill(0));
  for (let c = 0; c <= Y.length; c++) filled['0:' + c] = true;
  for (let r = 0; r <= X.length; r++) filled[r + ':0'] = true;

  F.push({shapes:gridShapes(dp, filled, {}, [
      S.t(4.9, 6.05, 'x = ' + X + '      y = ' + Y, {c:COL.tealL, fs:.40}),
      S.t(4.9, .48, {zh:'dp[i][j] ＝ x 前 i 個字與 y 前 j 個字的 LCS 長度',
                     en:'dp[i][j] = the LCS length of x[:i] and y[:j]'},
          {c:COL.grey, fs:.32})]),
    view:VIEW, line:1,
    panels:[{lbl:'x', chips:X.split('').map(c => chip(c, 'dim'))},
            {lbl:'y', chips:Y.split('').map(c => chip(c, 'dim'))}],
    msg:{zh:'第 0 列和第 0 行都是 0：其中一邊一個字都不拿，共同子序列只能是空字串。跟 Day 34 的背包一樣，邊界是白送的那一排。',
         en:'Row 0 and column 0 are all zeros: if one side contributes no characters, the only common subsequence is the empty one. As with day 34 knapsack, the border is the row you get for free.'}});

  for (let i = 1; i <= X.length; i++){
    for (let j = 1; j <= Y.length; j++){
      const match = (X[i - 1] === Y[j - 1]);
      dp[i][j] = match ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
      filled[i + ':' + j] = true;
      const st = {};
      st[i + ':' + j] = match ? 'ok' : 'act';
      if (match) st[(i - 1) + ':' + (j - 1)] = 'hot';
      else { st[(i - 1) + ':' + j] = 'hot'; st[i + ':' + (j - 1)] = 'hot'; }
      F.push({shapes:gridShapes(dp, filled, st, [
          S.t(4.9, 6.05, match
            ? 'x[' + (i - 1) + '] = y[' + (j - 1) + '] = ' + X[i - 1] + '   ->  1 + ' + dp[i - 1][j - 1] + ' = ' + dp[i][j]
            : X[i - 1] + ' != ' + Y[j - 1] + '   ->  max(' + dp[i - 1][j] + ', ' + dp[i][j - 1] + ') = ' + dp[i][j],
            {c:match ? COL.tealL : COL.purpleL, fs:.38})]),
        view:VIEW, line:match ? 5 : 7,
        panels:[{lbl:{zh:'比對中', en:'comparing'},
                 chips:[chip('x[' + (i - 1) + '] = ' + X[i - 1], 'hot'), chip('y[' + (j - 1) + '] = ' + Y[j - 1], 'hot')]},
                {lbl:{zh:'讀到的格子', en:'cells read'},
                 chips:match ? [chip('diagonal ' + dp[i - 1][j - 1], 'ok')]
                             : [chip('up ' + dp[i - 1][j], 'dim'), chip('left ' + dp[i][j - 1], 'dim')]}],
        msg:match
          ? {zh:'兩邊的最後一個字都是 ' + X[i - 1] + '。把它們配成一對永遠不吃虧——任何不用這一對的最佳解，都可以改成用它而不變短——所以直接取左上角加一，不必再比。',
             en:'Both sides end in ' + X[i - 1] + '. Pairing them off never costs anything: any optimal answer that avoids this pair can be rewritten to use it without getting shorter. So the cell is simply the diagonal plus one, with nothing to compare.'}
          : {zh:'兩個字不一樣（' + X[i - 1] + ' 與 ' + Y[j - 1] + '），代表至少有一個字用不到，但不知道是哪一個。那就兩個都試：丟掉 x 的最後一個字（上面那格）或丟掉 y 的（左邊那格），取比較大的。',
             en:'The characters differ (' + X[i - 1] + ' vs ' + Y[j - 1] + '), so at least one of them is useless - we just do not know which. Try both: drop x’s last character (the cell above) or y’s (the cell to the left), and keep the better one.'}});
    }
  }

  /* backtrack */
  let i = X.length, j = Y.length;
  const st = {}, out = [];
  while (i && j){
    st[i + ':' + j] = 'act';
    if (X[i - 1] === Y[j - 1]){
      st[i + ':' + j] = 'ok';
      out.unshift(X[i - 1]);
      F.push({shapes:gridShapes(dp, filled, Object.assign({}, st), [
          S.t(4.9, 6.05, {zh:'往回走：對角線代表這個字有被用到',
                          en:'walking back: a diagonal step = this character was used'},
              {c:COL.tealL, fs:.32}),
          S.t(4.9, .48, 'LCS so far: ' + out.join(''), {c:COL.tealL, fs:.40})]),
        view:VIEW, line:11,
        panels:[{lbl:{zh:'目前格子', en:'cell'}, chips:[chip('dp[' + i + '][' + j + '] = ' + dp[i][j], 'hot')]},
                {lbl:'LCS', chips:out.map(c => chip(c, 'ok'))}],
        msg:{zh:'兩個字相同，而且這一格正好是左上角加一——所以這個 ' + X[i - 1] + ' 真的被選進答案裡。往左上角斜著走一步。',
             en:'The characters match and this cell is exactly the diagonal plus one, so this ' + X[i - 1] + ' really is part of the answer. Step diagonally.'}});
      i--; j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) i--;
    else j--;
  }
  F.push({shapes:gridShapes(dp, filled, st, [
      S.t(4.9, 6.05, 'LCS = ' + out.join('') + '   (length ' + dp[X.length][Y.length] + ')',
          {c:COL.tealL, fs:.44}),
      S.t(4.9, .48, {zh:'只留兩列 → 長度還在，路徑不見了',
                     en:'keep two rows -> the length survives, the path does not'},
          {c:COL.orangeL, fs:.34})]),
    view:VIEW, line:10,
    panels:[{lbl:'LCS', chips:out.map(c => chip(c, 'ok'))},
            {lbl:{zh:'長度', en:'length'}, chips:[chip(String(dp[X.length][Y.length]), 'ok')]}],
    msg:{zh:'每一格只讀上一列和左邊一格，所以用兩列滾動就夠了，空間降到 O(min(n, m))。但回溯需要整張表——滾動版只能回答「多長」，不能回答「是哪一串」。這和 Day 34 一維背包丟掉「拿了哪些物品」是同一個取捨。',
         en:'Every cell reads only the row above and the cell to its left, so two rolling rows are enough and the space drops to O(min(n, m)). But backtracking needs the whole table: the rolling version can only answer how long, never which characters. It is the same trade the 1-D knapsack made on day 34 when it lost the list of items.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 5 - LCS of two permutations is an LIS in disguise
 * ======================================================================== */
const CODE_PERM = [
  'def lcs_of_permutations(p, q):',
  '    pos = {v: i for i, v in enumerate(p)}',
  '    mapped = [pos[v] for v in q]      # relabel q by its place in p',
  '    return lis_patience(mapped)       # O(n log n), not O(n*m)',
  '',
  '# a common subsequence of p and q',
  '#   = values in the same relative order in both',
  '#   = an increasing run of those labels',
  '# repeated values -> Hunt-Szymanski (match lists, same idea)'
];

const P = [1, 2, 3, 4, 5, 6];
const Q = [2, 4, 1, 5, 6, 3];
const PX0 = 2.05, PW = 0.98, PH = 0.62;

function permShapes(i, labels, tails, tst, keep, extra){
  const qst = {}, lst = {};
  for (let j = 0; j < Q.length; j++){
    qst[j] = keep ? (keep.indexOf(j) >= 0 ? 'ok' : 'done')
                  : (j < i ? 'done' : (j === i ? 'hot' : 'idle'));
    lst[j] = qst[j];
  }
  const pst = {};
  for (let j = 0; j < P.length; j++)
    pst[j] = (!keep && i >= 0 && P[j] === Q[i]) ? 'hot' : 'soft';
  let out = cellRow(P, PX0, 5.05, PW, PH, {states:pst, index:false, fs:.32, title:'p'});
  out = out.concat(cellRow(Q, PX0, 3.85, PW, PH, {states:qst, index:false, fs:.32, title:'q'}));
  out = out.concat(cellRow(labels, PX0, 2.65, PW, PH,
    {states:lst, index:false, fs:.32, title:{zh:'在 p 的位置', en:'index in p'}}));
  if (tails)
    out = out.concat(cellRow(tails.length ? tails : [''], PX0, 1.25, PW, PH,
      {states:tst || {}, index:false, fs:.32, title:'tails'}));
  return out.concat(extra || []);
}

function permFrames(){
  const F = new Frames();
  const labels = Q.map(() => '');
  F.push({shapes:permShapes(-1, labels.slice(), null, {}, null, [
      S.t(4.9, 6.05, {zh:'兩個排列的 LCS ＝ 一個 LIS', en:'the LCS of two permutations is an LIS'},
          {c:COL.tealL, fs:.40}),
      S.t(4.9, .60, {zh:'共同子序列＝在兩邊都保持同樣的先後順序',
                     en:'a common subsequence keeps the same order on both sides'},
          {c:COL.grey, fs:.32})]),
    view:VIEW, line:0,
    panels:[{lbl:'p', chips:P.map(v => chip(String(v), 'dim'))},
            {lbl:'q', chips:Q.map(v => chip(String(v), 'dim'))}],
    msg:{zh:'兩個序列的元素完全一樣，只是順序不同。一般的 LCS 要填 n×m 的表；但這個特例有捷徑，而且捷徑來自定義本身。',
         en:'The two sequences hold exactly the same values in different orders. The general LCS fills an n-by-m table, but this special case has a shortcut, and the shortcut falls straight out of the definition.'}});

  const posOf = {};
  P.forEach((v, i) => posOf[v] = i);
  for (let i = 0; i < Q.length; i++){
    labels[i] = posOf[Q[i]];
    F.push({shapes:permShapes(i, labels.slice(), null, {}, null, [
        S.t(4.9, .60, 'q[' + i + '] = ' + Q[i] + '  sits at index ' + posOf[Q[i]] + ' of p',
            {c:COL.purpleL, fs:.38})]),
      view:VIEW, line:2,
      panels:[{lbl:{zh:'改標籤', en:'relabelling'}, chips:[chip(Q[i] + ' -> ' + posOf[Q[i]], 'hot')]},
              {lbl:{zh:'標籤序列', en:'labels'}, chips:labels.filter(l => l !== '').map(l => chip(String(l), 'dim'))}],
      msg:{zh:'把 q 裡的每個值換成「它在 p 的第幾格」。順序資訊沒有遺失，只是換了一種寫法：p 本身的標籤剛好是 0,1,2,3,…，也就是遞增。',
           en:'Replace each value of q by the slot it occupies in p. No information is lost, only rewritten: p itself now reads 0, 1, 2, 3, ... which is simply "increasing".'}});
  }

  const tails = [], idx = [], parent = new Array(Q.length).fill(-1);
  for (let i = 0; i < Q.length; i++){
    const x = labels[i], k = lowerBound(tails, x);
    const appended = (k === tails.length);
    if (appended){ tails.push(x); idx.push(i); } else { tails[k] = x; idx[k] = i; }
    parent[i] = k ? idx[k - 1] : -1;
    const st = {};
    for (let j = 0; j < tails.length; j++) st[j] = 'idle';
    st[k] = appended ? 'ok' : 'hot';
    F.push({shapes:permShapes(i, labels.slice(), tails.slice(), st, null, [
        S.t(4.9, .60, 'label ' + x + '  ->  ' + (appended ? 'tails.append  (length ' + tails.length + ')'
                                                          : 'tails[' + k + '] = ' + x),
            {c:appended ? COL.tealL : COL.orangeL, fs:.36})]),
      view:VIEW, line:3,
      panels:[{lbl:{zh:'目前標籤', en:'label'}, chips:[chip(String(x), 'hot')]},
              {lbl:'tails', chips:tails.map(t => chip(String(t), 'dim'))},
              {lbl:{zh:'長度', en:'length'}, chips:[chip(String(tails.length), appended ? 'ok' : 'dim')]}],
      msg:{zh:'現在跑的是 Tab 1 一模一樣的 patience sorting，只是跑在標籤序列上。標籤遞增 ⇔ 這些值在 p 裡也是照順序出現 ⇔ 它們是一個共同子序列。',
           en:'This is the identical patience sorting from tab 1, now running on the labels. An increasing run of labels means those values appear in the same order in p as they do in q - which is precisely a common subsequence.'}});
  }
  const out = [];
  let p2 = idx[idx.length - 1];
  while (p2 !== -1){ out.unshift(p2); p2 = parent[p2]; }
  F.push({shapes:permShapes(Q.length, labels.slice(), tails.slice(),
      tails.map(() => 'ok').reduce((o, s, k) => (o[k] = s, o), {}), out, [
      S.t(4.9, 6.05, 'LCS = ' + out.map(k => Q[k]).join(', ') + '   (length ' + tails.length + ')',
          {c:COL.tealL, fs:.42}),
      S.t(4.9, .60, {zh:'n = 1200：O(n·m) 表 0.73 秒，LIS 版 0.0014 秒（約 500 倍）',
                     en:'n = 1200: the O(n*m) table 0.73s, the LIS version 0.0014s'},
          {c:COL.orangeL, fs:.30})]),
    view:VIEW, line:3,
    panels:[{lbl:'LCS', chips:out.map(k => chip(String(Q[k]), 'ok'))},
            {lbl:{zh:'長度', en:'length'}, chips:[chip(String(tails.length), 'ok')]}],
    msg:{zh:'同樣的答案，成本從 O(n·m) 掉到 O(n log n)。值有重複時不能直接查位置，但同一個想法還在：把每個值在另一邊出現的所有位置列出來、由大到小接進同一個 LIS——那就是 Hunt–Szymanski，也是 diff 工具在處理大檔案時走的路。',
         en:'Same answer, and the cost falls from O(n*m) to O(n log n). When values repeat you cannot look up a single position, but the idea survives: list every position where a value occurs on the other side, feed them into the same LIS in descending order, and you have Hunt-Szymanski - the route diff tools take on large files.'}});
  return F.list;
}

/* ======================================================================== */
const DAY_META = {
  title:{zh:'Day 35 — LIS 與 LCS：最長遞增子序列與最長共同子序列',
         en:'LIS and LCS - the longest subsequence you keep, and the table you throw away'},
  sub:{zh:'LIS 可以把 O(n²) 的表整張刪掉，換成一個 tails 陣列加上 Day 23 的 lower_bound；LCS 沒有這種好事——除非兩邊互為排列。',
       en:'LIS lets you delete the quadratic table outright and replace it with a tails array plus the lower_bound from day 23. LCS has no such trick - unless the two inputs are permutations of one another.'},
  tabs:[
    {
      id:'lis', label:{zh:'LIS：patience sorting', en:'LIS by patience sorting'},
      stage:{zh:'一個 tails 陣列，一次二分搜尋，O(n log n)',
             en:'one tails array, one binary search per element, O(n log n)'},
      view:VIEW,
      variants:[{zh:'[10, 9, 2, 5, 3, 7, 101, 18]', en:'[10, 9, 2, 5, 3, 7, 101, 18]'},
                {zh:'陷阱：tails 不是答案', en:'the trap: tails is not the answer'}],
      idea:{zh:'tails[k] 記的是「長度 k+1 的遞增子序列，結尾最小能到多少」。因為結尾越小越好接，所以每格只留最小值；因為 tails 天生排序好，找位置就是 Day 23 的 lower_bound。整個演算法只有兩個動作：接在尾端（長度加一）或覆蓋某一格（換一個更好接的結尾）。最後 tails 的長度就是答案——但 tails 的內容通常不是答案，換第二個輸入就看得到：那串裡的 1 是輸入的最後一個元素，根本不可能排在前面。真正的子序列要靠插入時記下的 parent 指標回溯。',
            en:'tails[k] records how small the ending of an increasing run of length k+1 can be. Smaller endings are easier to extend, so each slot keeps only the smallest; and because tails is sorted by construction, finding the slot is the lower_bound of day 23. The algorithm only ever does two things: append (the length grows) or overwrite a slot (a cheaper ending for the same length). The final length of tails is the answer - but its contents usually are not, as the second input shows: the 1 sitting in tails is the last element of the array and cannot precede anything. The actual subsequence comes from parent pointers taken at insert time.'},
      legend:[['#ff9736', {zh:'目前的值 / 被覆蓋的格子', en:'current value / overwritten slot'}],
              ['#9d6bff', {zh:'二分搜尋落點', en:'where the binary search lands'}],
              ['#3fe0dd', {zh:'長度增加 / 最終答案', en:'the length grows / the final answer'}],
              ['#2f5661', {zh:'已處理完', en:'already placed'}]],
      code:CODE_LIS, build:lisFrames
    },
    {
      id:'strict', label:{zh:'嚴格遞增 vs 非遞減', en:'strict vs non-decreasing'},
      stage:{zh:'bisect_left 換成 bisect_right：答案從 3 變 5',
             en:'swap bisect_left for bisect_right and the answer moves from 3 to 5'},
      view:VIEW,
      variants:[{zh:'bisect_left（嚴格遞增）', en:'bisect_left (strictly increasing)'},
                {zh:'bisect_right（非遞減）', en:'bisect_right (non-decreasing)'}],
      idea:{zh:'Day 23 分過 lower_bound 和 upper_bound，今天它們變成兩道不同的題目。bisect_left 不肯站到相等值的後面，所以重複的值會蓋回同一格，長度不動——這就是「嚴格遞增」；bisect_right 會跨過相等值，落在下一格，於是重複的值也能接下去——這就是「非遞減」。除了那個函式名以外，程式碼一個字都不用改，而錯的那個版本永遠不會報錯。所以面試時第一件事是問清楚題目要哪一種。',
            en:'Day 23 separated lower_bound from upper_bound; today they become two different problems. bisect_left refuses to sit after an equal value, so a repeat overwrites the same slot and the length does not move - that is "strictly increasing". bisect_right steps past equals into the next slot, so repeats extend the run - that is "non-decreasing". Not one other character changes, and the wrong choice never raises. Ask which one the question means before writing any code.'},
      legend:[['#ff9736', {zh:'覆蓋（長度不變）', en:'overwrite (length unchanged)'}],
              ['#3fe0dd', {zh:'加長', en:'append (length grows)'}],
              ['#8fa3ac', {zh:'還沒處理', en:'not reached yet'}],
              ['#2f5661', {zh:'已處理完', en:'already placed'}]],
      code:CODE_STRICT, build:strictFrames
    },
    {
      id:'lc354', label:{zh:'LC 354 俄羅斯娃娃信封', en:'LC 354 Russian doll envelopes'},
      stage:{zh:'排序的第二個 key 反了，答案就安靜地錯掉',
             en:'point the second sort key the wrong way and the answer is quietly wrong'},
      view:VIEW,
      variants:[{zh:'高度遞減（正確）', en:'height descending (correct)'},
                {zh:'高度遞增（錯誤）', en:'height ascending (wrong)'}],
      idea:{zh:'二維的問題先用排序壓掉一維：寬度排好之後，只剩高度要找最長遞增子序列。關鍵在等寬的那一組——它們彼此裝不進去，所以高度必須由大排到小，讓遞增子序列不可能同時選到兩個。把限制編碼進排序，比在迴圈裡寫 if 乾淨得多；但也因此，寫反的時候沒有任何地方會噴錯，只是多算了幾層不存在的娃娃。',
            en:'A two-dimensional problem is squashed to one by sorting: once the widths are in order, all that is left is a longest increasing subsequence of heights. The subtlety is the equal-width group - those cannot nest in each other, so their heights must descend, making it impossible for an increasing run to pick two of them. Encoding the constraint in the sort is far cleaner than an if inside the loop, but it also means that getting the key backwards raises nothing: it just counts layers of dolls that do not exist.'},
      legend:[['#ff9736', {zh:'目前信封 / 覆蓋', en:'current envelope / overwrite'}],
              ['#3fe0dd', {zh:'多一層', en:'one more layer'}],
              ['#ff5c5c', {zh:'錯誤的答案', en:'the wrong answer'}],
              ['#2f5661', {zh:'已處理完', en:'already placed'}]],
      code:CODE_354, build:envFrames
    },
    {
      id:'lcs', label:{zh:'LCS：二維表格', en:'LCS: the 2-D table'},
      stage:{zh:'相同就取對角線加一，不同就取上或左的較大者',
             en:'match: one plus the diagonal. Mismatch: the larger of up and left'},
      view:VIEW,
      idea:{zh:'LCS 沒有 LIS 那種捷徑（在一般情況下，比 O(n·m) 更快的演算法至今沒人做得出來，而且有理論證據說做不到），所以老老實實填表。只有兩種情況：末字相同就一定配成一對（交換論證保證不吃虧），取左上角加一；不同就代表其中一個字沒用，但不知道是哪個，於是兩邊都試、取大的。回溯要整張表，滾動兩列只能拿到長度——這正是 Day 34 一維背包丟掉「拿了哪些物品」的同一個取捨。而這張表的用途比題目本身重要得多：git diff 的「沒改到的行」就是 LCS，ROUGE-L 也是 LCS 長度算出來的 F 值。',
            en:'LCS has no shortcut like the LIS one - in the general case nobody has beaten O(n*m) by more than a log factor, and there is complexity-theoretic evidence that nobody will - so the table gets filled honestly. There are only two cases: equal last characters are always worth pairing (an exchange argument shows it costs nothing), giving one plus the diagonal; different last characters mean one of them is useless without telling you which, so try both and keep the larger. Backtracking needs the whole table, and two rolling rows can only return the length - the same trade the 1-D knapsack made on day 34. What the table is *for* matters more than the puzzle: the unchanged lines in a git diff are an LCS, and ROUGE-L is an F-measure over the LCS length.'},
      legend:[['#3fe0dd', {zh:'字元相同，取對角線加一', en:'characters match: 1 + diagonal'}],
              ['#9d6bff', {zh:'字元不同，取上/左較大者', en:'mismatch: max of up and left'}],
              ['#ff9736', {zh:'正在讀的格子', en:'the cells being read'}],
              ['#2f5661', {zh:'尚未填', en:'not filled yet'}]],
      code:CODE_LCS, build:lcsFrames
    },
    {
      id:'perm', label:{zh:'LCS 變回 LIS', en:'LCS collapses to LIS'},
      stage:{zh:'兩個排列的 LCS：把 q 換成「在 p 的第幾格」就是 LIS',
             en:'LCS of two permutations: relabel q by its index in p and run LIS'},
      view:VIEW,
      idea:{zh:'共同子序列的定義是「在兩邊都保持一樣的先後順序」。如果兩個序列互為排列，把 q 的每個值換成它在 p 的位置，p 自己就變成 0,1,2,3,…；於是「在 p 裡也照順序」這句話，剛好等於「這串標籤遞增」。LCS 就這樣變成 LIS，成本從 O(n·m) 掉到 O(n log n)，n=1200 時實測快了約 500 倍。值有重複時改用 match list、由大到小接進同一個 LIS，就是 Hunt–Szymanski——真實 diff 工具處理大檔案時走的路。',
            en:'A common subsequence is by definition a set of values that keep the same relative order on both sides. If the two sequences are permutations of each other, relabel every value of q by its index in p; then p itself reads 0, 1, 2, 3, ..., and "in the same order as p" becomes literally "these labels increase". LCS turns into LIS, the cost drops from O(n*m) to O(n log n), and at n = 1200 the measured speedup is about 500x. When values repeat, feeding each value’s match list into the same LIS in descending order gives Hunt-Szymanski, which is the route real diff tools take on large files.'},
      legend:[['#ff9736', {zh:'正在改標籤的值', en:'the value being relabelled'}],
              ['#3fe0dd', {zh:'最後選中的共同子序列', en:'the common subsequence found'}],
              ['#9d6bff', {zh:'p 裡對應的位置', en:'its position in p'}],
              ['#2f5661', {zh:'沒被選到', en:'not chosen'}]],
      code:CODE_PERM, build:permFrames
    }
  ]
};
