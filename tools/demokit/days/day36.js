// DAY: 36
// TITLE_ZH: Day 36 — 區間 DP 與樹形 DP：選分割點，而不是選下一個物品
// TITLE_EN: Interval DP and tree DP - choosing the split point instead of the next item
// SUB_ZH: Day 33 到 35 的表格都以「前 i 個」當索引；今天索引換了兩次形狀。區間 DP 以一段 [i, j] 當索引，每一格要決定的是從哪裡切；樹形 DP 以一棵子樹當索引，要決定的是每個小孩往上回報什麼。兩個經典的無聲錯誤都從這裡來：迴圈順序寫成 i-then-j 會讀到還沒算的格子，樹上每個點只回報一個數字會漏掉父節點唯一需要的那件事。
// SUB_EN: Days 33 to 35 all indexed the table by a prefix - "the best answer using the first i items". Today the index changes shape twice. Interval DP indexes by a segment [i, j] and the choice is where to split it; tree DP indexes by a subtree and the choice is what each child reports upward. Both classic silent bugs of the day live here: filling an interval table in the natural i-then-j order reads cells that do not exist yet, and giving a tree node one number instead of a pair throws away the one fact its parent needed.
// FOLDER: day%2036%20-%20interval%20and%20tree%20dp
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const fmt = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

/* ======================================================================== *
 * Tab 1 - matrix chain: the table is triangular and the loop order matters
 * ======================================================================== */
const DIMS = [40, 20, 30, 10, 30];
const NAMES = ['A', 'B', 'C', 'D'];
const NM = DIMS.length - 1;

const CODE_CHAIN = [
  'def matrix_chain(dims, length_first=True):',
  '    n = len(dims) - 1',
  '    dp = [[0] * n for _ in range(n)]',
  '    cells = (by_length(n) if length_first      # (0,1)(1,2)(2,3)(0,2)(1,3)(0,3)',
  '             else [(i, j) for i in range(n)    # (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)',
  '                          for j in range(i + 1, n)])',
  '    for i, j in cells:',
  '        dp[i][j] = min(',
  '            dp[i][k] + dp[k + 1][j]            # both must already be filled',
  '            + dims[i] * dims[k + 1] * dims[j + 1]',
  '            for k in range(i, j))',
  '    return dp[0][n - 1]'
];

const TX0 = 2.45, TW = 1.12, TY0 = 1.45, TH = 0.80;

function triShapes(dp, filled, states, note){
  const out = [];
  for (let j = 0; j < NM; j++)
    out.push(S.t(TX0 + j * TW + (TW - .06) / 2, TY0 - .28, 'j=' + j,
                 {c:'#8fa3ac', fs:.28}));
  for (let i = 0; i < NM; i++){
    out.push(S.t(TX0 - .22, TY0 + i * TH + TH * .46, 'i=' + i,
                 {c:'#8fa3ac', fs:.28, anchor:'end'}));
    for (let j = i; j < NM; j++){
      const st = (states && states[i + ',' + j]) || (filled[i + ',' + j] ? 'soft' : 'ghost');
      const lab = (i === j) ? '0' : (filled[i + ',' + j] ? fmt(dp[i][j]) : '');
      out.push(S.r(TX0 + j * TW, TY0 + i * TH, TW - .10, TH - .10, st, lab, {fs:.28}));
    }
  }
  for (let m = 0; m < NM; m++)
    out.push(S.t(7.05, TY0 + .38 + m * .52,
                 NAMES[m] + '  ' + DIMS[m] + 'x' + DIMS[m + 1],
                 {c:'#3fe0dd', fs:.30, anchor:'start'}));
  if (note) out.push(note);
  return out;
}

function chainFrames(v){
  const bad = (v === 1);
  const F = new Frames();
  const dp = [], filled = {};
  for (let i = 0; i < NM; i++){ dp.push(new Array(NM).fill(0)); filled[i + ',' + i] = 1; }

  const order = [];
  if (bad){
    for (let i = 0; i < NM; i++) for (let j = i + 1; j < NM; j++) order.push([i, j]);
  } else {
    for (let L = 2; L <= NM; L++) for (let i = 0; i + L - 1 < NM; i++) order.push([i, i + L - 1]);
  }

  F.push({shapes:triShapes(dp, filled, {},
      S.t(4.9, 5.25, {zh:'對角線是單一矩陣，不用乘，成本 0；答案在右上角那一格',
                      en:'the diagonal costs 0; the answer sits in the top-right cell'},
          {c:'#3fe0dd', fs:.28})),
    view:VIEW, line:2,
    panels:[{lbl:{zh:'填格順序', en:'fill order'},
             chips:order.map(c => chip('(' + c[0] + ',' + c[1] + ')', 'dim'))},
            {lbl:{zh:'dp[0][3]', en:'dp[0][3]'}, chips:[chip('?', 'dim')]}],
    msg:bad
      ? {zh:'這一版用最直覺的雙層迴圈：i 由上往下、j 由左往右。看起來每一格都會被填到，順序也很自然——問題在於「自然」跟「正確」是兩回事，而錯的時候不會有任何例外被丟出來。',
         en:'This run uses the instinctive double loop: i top to bottom, j left to right. Every cell does get filled and the order looks perfectly reasonable - the catch is that reasonable and correct are different things here, and being wrong raises nothing.'}
      : {zh:'dp[i][j] 是「把第 i 到第 j 個矩陣乘起來最少要幾次純量乘法」。表只有右上三角有意義，對角線先填 0。接下來每一格都會去讀比它短的區間，所以填表的順序必須以區間長度為外圈。',
         en:'dp[i][j] is the cheapest way to multiply matrices i..j. Only the upper triangle means anything, and the diagonal starts at 0. Every other cell reads strictly shorter intervals, which is exactly why the outer loop has to be over interval length.'}});

  for (const [i, j] of order){
    let best = Infinity, bestK = -1, reads = [];
    for (let k = i; k < j; k++){
      const okL = !!filled[i + ',' + k], okR = !!filled[(k + 1) + ',' + j];
      const cost = dp[i][k] + dp[k + 1][j] + DIMS[i] * DIMS[k + 1] * DIMS[j + 1];
      const st = {};
      st[i + ',' + j] = 'hot';
      st[i + ',' + k] = okL ? 'act' : 'bad';
      st[(k + 1) + ',' + j] = okR ? 'act' : 'bad';
      const broken = !okL || !okR;
      F.push({shapes:triShapes(dp, filled, st,
          S.t(4.9, 5.25,
              'k=' + k + ':  ' + fmt(dp[i][k]) + ' + ' + fmt(dp[k + 1][j]) + ' + ' +
              DIMS[i] + 'x' + DIMS[k + 1] + 'x' + DIMS[j + 1] + ' = ' + fmt(cost),
              {c:broken ? '#ff5c5c' : '#c7a6ff', fs:.33})),
        view:VIEW, line:8,
        panels:[{lbl:{zh:'正在填', en:'filling'}, chips:[chip('dp[' + i + '][' + j + ']', 'hot')]},
                {lbl:{zh:'讀到的兩格', en:'the two cells read'},
                 chips:[chip('dp[' + i + '][' + k + '] = ' + fmt(dp[i][k]), okL ? 'dim' : 'bad'),
                        chip('dp[' + (k + 1) + '][' + j + '] = ' + fmt(dp[k + 1][j]), okR ? 'dim' : 'bad')]}],
        msg:broken
          ? {zh:'就是這裡出事的：dp[' + (k + 1) + '][' + j + '] 還沒被算過，現在讀到的是初始值 0。0 是個完全合法的數字，min 照樣會挑它，程式一路跑完也不會抱怨——只是把一整段乘法的成本當成免費。',
             en:'This is the moment it breaks: dp[' + (k + 1) + '][' + j + '] has not been computed yet, so what gets read is the initial 0. Zero is a perfectly valid number, min happily takes it, and the program runs to completion without complaint - it has simply priced a whole block of multiplications at nothing.'}
          : {zh:'把 i..j 想成最上層那一次乘法：左邊是 i..k 的結果（' + DIMS[i] + 'x' + DIMS[k + 1] + '），右邊是 k+1..j 的結果（' + DIMS[k + 1] + 'x' + DIMS[j + 1] + '），這一次乘法本身要 ' + fmt(DIMS[i] * DIMS[k + 1] * DIMS[j + 1]) + ' 次。兩個子區間都比 i..j 短，所以剛剛已經算好了。',
             en:'Read i..j as its topmost multiplication: on the left the result of i..k (' + DIMS[i] + 'x' + DIMS[k + 1] + '), on the right the result of k+1..j (' + DIMS[k + 1] + 'x' + DIMS[j + 1] + '), and that final multiply alone costs ' + fmt(DIMS[i] * DIMS[k + 1] * DIMS[j + 1]) + '. Both sub-intervals are shorter than i..j, so both were finished earlier.'}});
      if (cost < best){ best = cost; bestK = k; }
      reads.push(k);
    }
    dp[i][j] = best; filled[i + ',' + j] = 1;
    const st2 = {}; st2[i + ',' + j] = (bad && (i !== 0 || j !== 1)) ? 'bad' : 'ok';
    F.push({shapes:triShapes(dp, filled, st2,
        S.t(4.9, 5.25, 'dp[' + i + '][' + j + '] = ' + fmt(best) + '   (split at k=' + bestK + ')',
            {c:st2[i + ',' + j] === 'bad' ? '#ff5c5c' : '#3fe0dd', fs:.33})),
      view:VIEW, line:10,
      panels:[{lbl:{zh:'剛填好', en:'just written'},
               chips:[chip('dp[' + i + '][' + j + '] = ' + fmt(best), 'ok')]},
              {lbl:{zh:'分割點', en:'split point'}, chips:[chip('k = ' + bestK, 'act')]}],
      msg:{zh:'這一格記下的是最小成本，另外一張表同時記下最佳的 k。最後要還原括號怎麼打，就是從右上角那一格順著 k 往下拆。',
           en:'The cell stores the minimum, and a parallel table stores the winning k. Reconstructing the bracketing at the end is just following those split points down from the top-right cell.'}});
  }

  const ans = dp[0][NM - 1];
  F.push({shapes:triShapes(dp, filled, {'0,3':bad ? 'bad' : 'ok'},
      S.t(4.9, 5.25, bad
        ? {zh:'答案 20,000 —— 比真正的最小值還小，因為有幾段乘法被當成免費',
           en:'reads 20,000 - below the true minimum; some products cost zero'}
        : {zh:'((A(BC))D) 只要 26,000 次；最糟的括號法要 69,000 次',
           en:'((A(BC))D) costs 26,000 multiplications; the worst costs 69,000'},
        {c:bad ? '#ff5c5c' : '#3fe0dd', fs:.28})),
    view:VIEW, line:11,
    panels:[{lbl:{zh:'dp[0][3]', en:'dp[0][3]'},
             chips:[chip(fmt(ans), bad ? 'bad' : 'ok')]},
            {lbl:{zh:'正確答案', en:'the true answer'}, chips:[chip('26,000', 'dim')]}],
    msg:bad
      ? {zh:'20,000 比 26,000 還小，但它不是更好的括號法——沒有任何一種乘法順序能做到 20,000。這個數字對應的是一個根本不存在的計畫，而整支程式從頭到尾沒有出錯訊息。區間 DP 的迴圈順序不是風格問題，它就是演算法本身。',
         en:'20,000 is smaller than 26,000, but it is not a better bracketing - no multiplication order achieves 20,000. The number describes a plan that does not exist, and nothing in the run ever complained. In interval DP the loop order is not a matter of style; it is the algorithm.'}
      : {zh:'四個矩陣就已經差了 2.65 倍，而括號法的數量是 Catalan 數，十五個矩陣就枚舉不完了。O(n^3) 的表把這件事壓成一個三角形：n^2 格，每格掃 n 個分割點。',
         en:'With only four matrices the gap is already 2.65x, and the number of bracketings is a Catalan number, so enumeration dies around fifteen matrices. The cubic DP squeezes all of that into a triangle: n^2 cells, each scanning n split points.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 2 - LeetCode 312: split on the LAST balloon, not the first
 * ======================================================================== */
const BALLOONS = [3, 1, 5, 8];
const PAD = [1].concat(BALLOONS, [1]);

const CODE_BURST = [
  'a = [1] + nums + [1]',
  'dp = [[0] * len(a) for _ in a]',
  '',
  '# k is the balloon popped LAST inside the open interval (i, j)',
  'for length in range(2, len(a)):',
  '    for i in range(len(a) - length):',
  '        j = i + length',
  '        dp[i][j] = max(dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]',
  '                       for k in range(i + 1, j))',
  '',
  '# the wrong one: k popped FIRST, with the neighbours it has right now',
  '#   a[k-1] * a[k] * a[k+1] + go(i, k-1) + go(k+1, j)   -> 98'
];

const BX0 = 2.05, BW = 0.94, BH = 0.72;

function padRow(y, states, title){
  return cellRow(PAD, BX0, y, BW, BH,
    {states:states || {}, index:false, fs:.34, title:title});
}

function burstLastDP(){
  const n = PAD.length;
  const dp = [], ch = [];
  for (let i = 0; i < n; i++){ dp.push(new Array(n).fill(0)); ch.push(new Array(n).fill(-1)); }
  for (let L = 2; L < n; L++)
    for (let i = 0; i + L < n; i++){
      const j = i + L;
      for (let k = i + 1; k < j; k++){
        const c = dp[i][k] + dp[k][j] + PAD[i] * PAD[k] * PAD[j];
        if (c > dp[i][j]){ dp[i][j] = c; ch[i][j] = k; }
      }
    }
  return {dp:dp, ch:ch};
}
function burstOrder(ch, i, j){
  const k = ch[i][j];
  if (k === -1) return [];
  return burstOrder(ch, i, k).concat(burstOrder(ch, k, j), [k]);
}
function burstFirstWrong(){
  const memo = {};
  function go(i, j){
    if (i > j) return 0;
    const key = i + ',' + j;
    if (memo[key] != null) return memo[key];
    let best = 0;
    for (let k = i; k <= j; k++)
      best = Math.max(best, PAD[k - 1] * PAD[k] * PAD[k + 1] + go(i, k - 1) + go(k + 1, j));
    return (memo[key] = best);
  }
  return go(1, BALLOONS.length);
}

function burstFrames(v){
  const F = new Frames();
  const wrong = (v === 1);
  const {dp, ch} = burstLastDP(), last = PAD.length - 1;

  const wallSt = {}; wallSt[0] = 'done'; wallSt[last] = 'done';
  F.push({shapes:padRow(1.75, wallSt, 'a').concat([
      S.t(4.9, 3.05, {zh:'兩端各補一顆 1，邊界就不用特別處理了',
                      en:'a 1 is padded at each end so the boundary needs no case'},
          {c:'#8fa3ac', fs:.30}),
      S.t(4.9, 3.80, wrong
        ? {zh:'問題：先戳哪一顆？', en:'the question: which balloon do you pop FIRST?'}
        : {zh:'問題：最後戳哪一顆？', en:'the question: which balloon is popped LAST?'},
        {c:wrong ? '#ff9736' : '#3fe0dd', fs:.38})]),
    view:VIEW, line:0,
    panels:[{lbl:{zh:'氣球', en:'balloons'}, chips:BALLOONS.map(b => chip(String(b), 'dim'))},
            {lbl:{zh:'目前最佳', en:'best so far'}, chips:[chip('0', 'dim')]}],
    msg:wrong
      ? {zh:'「先戳哪一顆」是所有人第一個想到的切法，因為戳氣球這個動作本身就是由前往後發生的。等一下會看到，這個切法算出來的是 98，而正確答案是 167——中間沒有任何一行會噴錯。',
         en:'"Which one do I pop first" is everyone\'s first split, because popping is something that happens front to back. It returns 98 where the true answer is 167, and not one line of it ever raises.'}
      : {zh:'反過來問：在這一段裡「最後」被戳掉的是誰？這個問法看起來很彆扭，但它是唯一能讓左右兩半互不干擾的問法。',
         en:'Turn the question around: inside this segment, which balloon is popped last? It reads awkwardly, and it is the only phrasing that makes the two halves independent.'}});

  if (wrong){
    /* pop 5 first, then watch the halves stop being independent */
    const k = 3;                             // value 5 in the padded array
    const st1 = {}; st1[k] = 'hot'; st1[k - 1] = 'act'; st1[k + 1] = 'act';
    F.push({shapes:padRow(1.75, st1, 'a').concat([
        S.t(4.9, 3.05, 'pop a[3] = 5 first  ->  1 * 5 * 8 = ' + (PAD[2] * PAD[3] * PAD[4]) + ' coins',
            {c:'#ffbe6b', fs:.34}),
        S.t(4.9, 3.80, {zh:'接著左半 (1, 2) 和右半 (4, 4) 各自遞迴',
                        en:'then recurse on left (1, 2) and right (4, 4) separately'},
            {c:'#8fa3ac', fs:.28})]),
      view:VIEW, line:11,
      panels:[{lbl:{zh:'先戳', en:'popped first'}, chips:[chip('5', 'hot')]},
              {lbl:{zh:'當下的鄰居', en:'neighbours right now'},
               chips:[chip('1', 'act'), chip('8', 'act')]}],
      msg:{zh:'到這裡都還是對的：5 目前的鄰居確實是 1 和 8，拿到 40 個金幣。問題出在下一步——遞迴左半的時候，它會假設左半的右邊界是牆。',
           en:'So far so good: 5 really does sit between 1 and 8 right now, and it earns 40 coins. The damage happens on the next line, when the left half is solved on the assumption that its right-hand boundary is a wall.'}});

    const st2 = {}; st2[k] = 'done'; st2[2] = 'bad'; st2[4] = 'bad';
    F.push({shapes:padRow(1.75, st2, 'a').concat([
        S.e(BX0 + 2 * BW + (BW - .06) / 2, 2.11, BX0 + 4 * BW + (BW - .06) / 2, 2.11,
            {s:'bad', pad:.10}),
        S.t(4.9, 3.05, {zh:'5 消失之後，1 和 8 變成鄰居了',
                        en:'once 5 is gone, 1 and 8 become neighbours'},
            {c:'#ff5c5c', fs:.34}),
        S.t(4.9, 3.80, {zh:'左半以後戳 1 時拿到的是 3*1*8，不是遞迴裡假設的 3*1*1',
                        en:'popping that 1 later earns 3*1*8, not the 3*1*1 assumed'},
            {c:'#ff5c5c', fs:.28})]),
      view:VIEW, line:11,
      panels:[{lbl:{zh:'左半假設的右鄰', en:'left half assumed'}, chips:[chip('wall = 1', 'bad')]},
              {lbl:{zh:'實際的右鄰', en:'actually'}, chips:[chip('8', 'bad')]}],
      msg:{zh:'這就是「先戳」不能用的原因：兩個子問題不獨立。左半能拿多少，取決於右半還剩下什麼，而遞迴式裡完全沒有地方可以表達這件事。DP 的前提是子問題可以單獨解，這個切法把前提弄壞了。',
           en:'This is why "pop first" cannot work: the subproblems are not independent. What the left half can earn depends on what survives in the right half, and the recurrence has nowhere to say so. DP assumes subproblems can be solved alone, and this split breaks that assumption.'}});

    F.push({shapes:padRow(1.75, {}, 'a').concat([
        S.t(4.9, 3.00, 'split on FIRST  ->  ' + burstFirstWrong(), {c:'#ff5c5c', fs:.44}),
        S.t(4.9, 3.80, 'split on LAST   ->  ' + dp[0][last], {c:'#3fe0dd', fs:.44}),
        S.t(4.9, 4.60, {zh:'兩個都跑得完，只有一個是對的',
                        en:'both run to completion; only one of them is right'},
            {c:'#8fa3ac', fs:.32})]),
      view:VIEW, line:11,
      panels:[{lbl:{zh:'先戳（錯）', en:'first (wrong)'}, chips:[chip(String(burstFirstWrong()), 'bad')]},
              {lbl:{zh:'最後戳（對）', en:'last (correct)'}, chips:[chip(String(dp[0][last]), 'ok')]}],
      msg:{zh:'98 和 167 差了將近一倍，但程式沒有當、沒有丟例外、甚至還很快。區間 DP 最常見的失敗長這樣：切法選錯，答案安靜地變小。判準只有一個——切完之後，兩邊的邊界是不是固定不動的。',
           en:'98 against 167 is nearly a factor of two, and yet nothing crashed, nothing raised, and it was fast. This is what a broken interval DP looks like: the wrong split, and a quietly smaller answer. The only test that matters is whether the two sides have boundaries that no longer move.'}});
    return F.list;
  }

  /* correct: scan the top-level choice of k, then replay the popping order */
  let best = 0, bestK = -1;
  for (let k = 1; k < last; k++){
    const c = dp[0][k] + dp[k][last] + PAD[0] * PAD[k] * PAD[last];
    const st = {}; st[k] = 'hot'; st[0] = 'act'; st[last] = 'act';
    for (let m = 1; m < k; m++) st[m] = 'soft';
    for (let m = k + 1; m < last; m++) st[m] = 'soft';
    F.push({shapes:padRow(1.75, st, 'a').concat([
        S.t(4.9, 3.05, 'k=' + k + ' last:  ' + dp[0][k] + ' + ' + dp[k][last] + ' + 1*'
            + PAD[k] + '*1  =  ' + c, {c:'#c7a6ff', fs:.34}),
        S.t(4.9, 3.80, {zh:'左半 (0, ' + k + ') 和右半 (' + k + ', ' + last + ') 的邊界都是不會動的牆',
                        en:'left (0, ' + k + ') and right (' + k + ', ' + last + ') sit between walls that never move'},
            {c:'#8fa3ac', fs:.28})]),
      view:VIEW, line:7,
      panels:[{lbl:{zh:'假設最後戳', en:'assume popped last'}, chips:[chip(String(PAD[k]), 'hot')]},
              {lbl:{zh:'這一刀的總分', en:'value of this split'}, chips:[chip(String(c), 'act')]},
              {lbl:{zh:'目前最佳', en:'best so far'}, chips:[chip(String(Math.max(best, c)), 'dim')]}],
      msg:{zh:'如果 ' + PAD[k] + ' 是整段裡最後一顆被戳掉的，那麼戳它的那一刻，中間的氣球全都不在了——它的鄰居必然就是兩端那兩面牆 1 和 1，跟兩邊怎麼戳完全無關。這就是子問題獨立的意思。',
           en:'If ' + PAD[k] + ' is the last one to go in this segment, then at the moment it pops everything between the walls is already gone - so its neighbours are exactly the two walls, 1 and 1, no matter how either side was cleared. That is what independence buys.'}});
    if (c > best){ best = c; bestK = k; }
  }

  const order = burstOrder(ch, 0, last);
  const alive = PAD.map(() => true);
  let coins = 0;
  const popped = [];
  for (const k of order){
    let L = k - 1; while (!alive[L]) L--;
    let R = k + 1; while (!alive[R]) R++;
    const gain = PAD[L] * PAD[k] * PAD[R];
    coins += gain; popped.push(PAD[k]);
    const st = {}; st[k] = 'hot'; st[L] = 'act'; st[R] = 'act';
    for (let m = 0; m < PAD.length; m++) if (!alive[m]) st[m] = 'done';
    F.push({shapes:padRow(1.75, st, 'a').concat([
        S.t(4.9, 3.05, PAD[L] + ' * ' + PAD[k] + ' * ' + PAD[R] + ' = ' + gain
            + '      total ' + coins, {c:'#ffbe6b', fs:.36}),
        S.t(4.9, 3.80, {zh:'順著 choice 表還原出來的實際戳法',
                        en:'the actual popping order, read back out of the choice table'},
            {c:'#8fa3ac', fs:.31})]),
      view:VIEW, line:8,
      panels:[{lbl:{zh:'戳掉', en:'popping'}, chips:[chip(String(PAD[k]), 'hot')]},
              {lbl:{zh:'已戳順序', en:'order so far'}, chips:popped.map(p => chip(String(p), 'dim'))},
              {lbl:{zh:'金幣', en:'coins'}, chips:[chip(String(coins), 'ok')]}],
      msg:{zh:'注意這個順序跟遞迴的順序是相反的：表裡記的是「誰最後戳」，還原出來卻要先戳最裡面的。dp 只保證分數，實際動作要從 choice 表倒著讀。',
           en:'Note that this order runs opposite to the recursion: the table records who goes last, but replaying it pops the innermost balloons first. dp guarantees the score; the moves come from reading the choice table backwards.'}});
    alive[k] = false;
  }

  F.push({shapes:padRow(1.75, PAD.map((_, i) => i === 0 || i === last ? 'done' : 'ok')
      .reduce((o, s, i) => (o[i] = s, o), {}), 'a').concat([
      S.t(4.9, 3.00, {zh:'最後戳 ＝ 邊界固定 ＝ 子問題獨立',
                      en:'popped last = fixed walls = independent subproblems'},
          {c:'#3fe0dd', fs:.33}),
      S.t(4.9, 3.80, 'total = ' + coins + '   (brute force over all 24 orders agrees)',
          {c:'#3fe0dd', fs:.30})]),
    view:VIEW, line:8,
    panels:[{lbl:{zh:'總分', en:'total'}, chips:[chip(String(coins), 'ok')]},
            {lbl:{zh:'戳法', en:'order'}, chips:popped.map(p => chip(String(p), 'ok'))},
            {lbl:{zh:'先戳版本', en:'the first-split version'}, chips:[chip(String(burstFirstWrong()), 'bad')]}],
    msg:{zh:'LeetCode 1547「切木棍」是同一題的鏡像：那邊反而是「第一刀」才切得開，因為第一刀把一段木頭切成兩段之後，兩段就再也不會互相影響。表格一樣、迴圈一樣，只有「哪一個選擇讓邊界固定下來」這件事不同。',
         en:'LeetCode 1547, cutting a stick, is the mirror image: there it is the *first* cut that separates, because once a piece is cut in two the halves never interact again. Same table, same loops - the only question that changes is which choice freezes the boundaries.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 3 - LeetCode 337: a tree node has to report a PAIR
 * ======================================================================== */
const TREE = [3, 2, 3, null, 3, null, 1];      // heap layout, null = missing
const POST = [4, 1, 6, 2, 0];                  // post-order over the present nodes

const CODE_ROB = [
  'def rob(node):                  # returns (take, skip)',
  '    if node is None:',
  '        return 0, 0',
  '    lt, ls = rob(node.left)',
  '    rt, rs = rob(node.right)',
  '    take = node.val + ls + rs   # I am robbed -> children must be skipped',
  '    skip = max(lt, ls) + max(rt, rs)   # I am not -> children are free',
  '    return take, skip',
  '',
  'answer = max(rob(root))',
  '',
  '# one number per node instead:  max(val + below, below)  -> 12, always all of them'
];

function kids(i){
  const out = [];
  if (TREE[2 * i + 1] != null) out.push(2 * i + 1);
  if (TREE[2 * i + 2] != null) out.push(2 * i + 2);
  return out;
}
function robPair(i){
  if (i >= TREE.length || TREE[i] == null) return [0, 0];
  const l = robPair(2 * i + 1), r = robPair(2 * i + 2);
  return [TREE[i] + l[1] + r[1], Math.max(l[0], l[1]) + Math.max(r[0], r[1])];
}
function robOne(i){
  if (i >= TREE.length || TREE[i] == null) return 0;
  const below = robOne(2 * i + 1) + robOne(2 * i + 2);
  return Math.max(TREE[i] + below, below);
}

const RY = 1.20, RROW = 1.16, RSPAN = 5.4;

function robShapes(states, labels, extra){
  let out = heapTreeShapes(TREE, 2.10, RY, RSPAN, RROW, states, {r:.42});
  const px = i => {
    const d = Math.floor(Math.log2(i + 1)), first = Math.pow(2, d) - 1;
    return 2.10 + RSPAN * ((i - first) + .5) / Math.pow(2, d);
  };
  const py = i => RY + Math.floor(Math.log2(i + 1)) * RROW;
  for (const k in labels)
    out.push(S.t(px(+k) + .58, py(+k) + .04, labels[k], {c:'#ffbe6b', fs:.32, anchor:'start'}));
  return out.concat(extra || []);
}

function robFrames(v){
  const one = (v === 1);
  const F = new Frames();
  const labels = {}, states = {};

  F.push({shapes:robShapes({}, {}, [
      S.t(4.9, 4.95, one
        ? {zh:'每個節點只回報一個數字：「這棵子樹最多能拿多少」',
           en:'each node reports one number: "the most this subtree can give"'}
        : {zh:'每個節點回報一對：（拿我，不拿我）',
           en:'each node reports a pair: (take me, skip me)'},
        {c:one ? '#ff9736' : '#3fe0dd', fs:.29}),
      S.t(4.9, 5.65, {zh:'規則：相鄰的兩層不能同時被拿',
                     en:'the rule: a node and its child may not both be taken'},
          {c:'#8fa3ac', fs:.30})]),
    view:VIEW, line:0,
    panels:[{lbl:{zh:'樹', en:'tree'}, chips:[chip('3 / 2 3 / _ 3 _ 1', 'dim')]},
            {lbl:{zh:'目前結果', en:'result so far'}, chips:[chip('—', 'dim')]}],
    msg:one
      ? {zh:'「這棵子樹最多能拿多少」聽起來就是題目要的東西，所以它常常是第一個被寫出來的狀態。它的問題不在算錯，而在資訊不夠：父節點要決定自己能不能拿，必須知道小孩到底有沒有被拿，而這個數字已經把那件事丟掉了。',
         en:'"The most this subtree can give" sounds exactly like what the problem asks for, so it is usually the first state anyone writes. The flaw is not arithmetic but missing information: to decide whether it may take itself, the parent has to know whether the child was taken, and that single number has already discarded it.'}
      : {zh:'由下往上算，每個節點回報兩個數字：自己被拿的情況下最多多少，以及自己沒被拿的情況下最多多少。兩個狀態一分開，父節點就永遠知道該配哪一個。',
         en:'Work bottom up and have every node report two numbers: the best with itself taken, and the best with itself skipped. Once those two states are separated, the parent always knows which one it is allowed to pair with.'}});

  for (const i of POST){
    const ch = kids(i);
    const st = Object.assign({}, states);
    st[i] = 'hot';
    for (const c of ch) st[c] = 'act';
    const lab = Object.assign({}, labels);
    if (one){
      const val = robOne(i), below = ch.reduce((s, c) => s + robOne(c), 0);
      lab[i] = String(val);
      F.push({shapes:robShapes(st, lab, [
          S.t(4.9, 4.95, 'max(' + TREE[i] + ' + ' + below + ', ' + below + ') = ' + val,
              {c:'#ffbe6b', fs:.34}),
          S.t(4.9, 5.65, {zh:'值都是正的，所以 val + below 永遠贏——每個點都會被拿',
                         en:'values are positive, so val + below always wins - all taken'},
              {c:'#ff5c5c', fs:.27})]),
        view:VIEW, line:11,
        panels:[{lbl:{zh:'節點', en:'node'}, chips:[chip('val = ' + TREE[i], 'hot')]},
                {lbl:{zh:'小孩加總', en:'children total'}, chips:[chip(String(below), 'act')]},
                {lbl:{zh:'回報', en:'reports'}, chips:[chip(String(val), 'bad')]}],
        msg:{zh:'這個 max 從來沒有真的在選：只要值是非負的，加上自己一定不會比較差，所以每次都走 val + below 那一邊。父節點看到的數字裡已經把小孩算進去了，卻沒有任何欄位告訴它「小孩被拿了，你不能拿」。',
             en:'That max never actually chooses: with non-negative values, adding yourself can never hurt, so it takes the val + below branch every time. The number the parent receives already includes the child, but carries no field to say "the child was taken, so you may not be".'}});
      labels[i] = String(val);
    } else {
      const [tk, sk] = robPair(i);
      const parts = ch.map(c => { const p = robPair(c); return p; });
      const skSum = parts.map(p => p[1]).reduce((a, b) => a + b, 0);
      const freeSum = parts.map(p => Math.max(p[0], p[1])).reduce((a, b) => a + b, 0);
      lab[i] = '(' + tk + ', ' + sk + ')';
      F.push({shapes:robShapes(st, lab, [
          S.t(4.9, 4.95, 'take = ' + TREE[i] + ' + ' + skSum + ' = ' + tk
              + '      skip = ' + freeSum, {c:'#ffbe6b', fs:.34}),
          S.t(4.9, 5.65, {zh:'take 只能配小孩的 skip；skip 可以配小孩的兩者取大',
                         en:'take pairs only with the children\'s skip; skip takes the better of the two'},
              {c:'#8fa3ac', fs:.24})]),
        view:VIEW, line:ch.length ? 5 : 2,
        panels:[{lbl:{zh:'節點', en:'node'}, chips:[chip('val = ' + TREE[i], 'hot')]},
                {lbl:{zh:'小孩回報', en:'children reported'},
                 chips:ch.length ? parts.map((p, n) => chip('(' + p[0] + ', ' + p[1] + ')', 'act'))
                                 : [chip('leaf', 'dim')]},
                {lbl:{zh:'本節點回報', en:'this node reports'},
                 chips:[chip('(' + tk + ', ' + sk + ')', 'ok')]}],
        msg:{zh:'兩條路分開算：要拿自己，兩個小孩就都只能用它們的 skip 值；不拿自己，兩個小孩各自取 max 就好。注意 skip 不代表小孩一定被拿，它只是「我這邊沒有限制」，所以兩種狀態都要留著往上傳。',
             en:'Two independent lines: to take this node, both children must contribute their skip value; to skip it, each child contributes whichever of its two numbers is larger. Note that skip does not mean the children were taken - it only means this node imposes no constraint, which is why both numbers keep travelling upward.'}});
      labels[i] = '(' + tk + ', ' + sk + ')';
    }
    states[i] = 'done';
  }

  const ans = one ? robOne(0) : Math.max(robPair(0)[0], robPair(0)[1]);
  const fin = {};
  if (one) for (let i = 0; i < TREE.length; i++){ if (TREE[i] != null) fin[i] = 'bad'; }
  else { fin[0] = 'ok'; fin[4] = 'ok'; fin[6] = 'ok'; fin[1] = 'done'; fin[2] = 'done'; }
  F.push({shapes:robShapes(fin, labels, [
      S.t(4.9, 4.95, one ? 'answer = ' + ans + '   (every node taken - illegal)'
                         : 'answer = max(take, skip) = ' + ans,
          {c:one ? '#ff5c5c' : '#3fe0dd', fs:.38}),
      S.t(4.9, 5.65, one
        ? {zh:'3 和它的小孩 2、3 同時被拿了，這在題目裡是違規的',
           en:'3 and its children 2 and 3 are taken at once, which is forbidden'}
        : {zh:'選中的是 3、3、1：沒有任何兩個相鄰',
           en:'the chosen nodes are 3, 3 and 1 - no two of them are adjacent'},
        {c:one ? '#ff5c5c' : '#8fa3ac', fs:.26})]),
    view:VIEW, line:9,
    panels:[{lbl:{zh:'一個數字的版本', en:'one number per node'}, chips:[chip(String(robOne(0)), 'bad')]},
            {lbl:{zh:'一對狀態的版本', en:'pair of states'},
             chips:[chip(String(Math.max(robPair(0)[0], robPair(0)[1])), 'ok')]}],
    msg:one
      ? {zh:'12 剛好就是全部節點的總和——這個「DP」其實只是把整棵樹加起來，穿了一件動態規劃的外衣。它不會報錯，而且在全正值的測資上看起來很合理，直到有人檢查被選中的點彼此相鄰為止。',
         en:'12 is precisely the sum of every node - this "DP" is just adding the whole tree up while wearing a dynamic-programming costume. It never raises, and on all-positive inputs it looks entirely plausible, right up until someone checks that the chosen nodes are adjacent.'}
      : {zh:'同一個「回報的東西不等於計分的東西」也撐起 LeetCode 124：節點往上回報的是「穿過我、最多只用一個小孩」的路徑，但它自己記分時可以在這裡轉彎、把兩個小孩都用上。樹的直徑也是同一招，只是把總和換成深度。',
         en:'The same "what a node reports is not what it scores" split drives LeetCode 124: a node returns the best downward path through it, using at most one child, yet the answer it contributes may bend at the node and use both. Tree diameter is the identical trick with depth instead of sum.'}});
  return F.list;
}

/* ======================================================================== *
 * Tab 4 - tree knapsack: the merge loop, and why it is not cubic
 * ======================================================================== */
const COURSES = {0:[1, 2], 1:[3, 4], 2:[5], 3:[], 4:[], 5:[]};
const CREDITS = [0, 2, 3, 4, 1, 5];
const BUDGET = 4;                                   // 3 real courses + the virtual root
const KPOST = [3, 4, 1, 5, 2, 0];
const KARR = [0, 1, 2, 3, 4, 5, null];              // heap layout of the same tree

const CODE_KNAP = [
  'def go(v):',
  '    dp = [0] + [value[v]] * budget      # taking v alone',
  '    size[v] = 1',
  '    for c in children[v]:',
  '        cdp, nd = go(c), list(dp)',
  '        for t in range(min(size[v], budget), 0, -1):',
  '            for u in range(1, min(size[c], budget - t) + 1):',
  '                nd[t + u] = max(nd[t + u], dp[t] + cdp[u])',
  '        dp, size[v] = nd, size[v] + size[c]',
  '    return dp',
  '',
  '# size(c) * size(merged so far) = the pairs whose LCA is v  ->  C(n, 2) in total'
];

const KX = 2.25, KSPAN = 5.2, KY = 1.15, KROW = 1.02;

function kPos(i){
  const d = Math.floor(Math.log2(i + 1)), first = Math.pow(2, d) - 1;
  return [KX + KSPAN * ((i - first) + .5) / Math.pow(2, d), KY + d * KROW];
}
function knapShapes(states, note, dpRow, dpLabel, charged){
  let out = heapTreeShapes(KARR, KX, KY, KSPAN, KROW, states, {r:.38});
  for (let i = 0; i < KARR.length; i++){
    if (KARR[i] == null) continue;
    const [x, y] = kPos(i);
    out.push(S.t(x, y + .62, CREDITS[KARR[i]] + (KARR[i] === 0 ? '' : ' cr'),
                 {c:'#8fa3ac', fs:.26}));
  }
  if (dpRow)
    out = out.concat(cellRow(dpRow, 3.05, 4.25, .86, .60,
      {states:{}, index:true, fs:.30, ifs:.24, ilift:.06, title:dpLabel || 'dp'}));
  if (charged != null)
    out.push(S.t(8.35, 4.62, 'pairs charged: ' + charged, {c:'#ffbe6b', fs:.30, anchor:'end'}));
  if (note) out.push(note);
  return out;
}

function knapFrames(){
  const F = new Frames();
  const dpOf = {}, size = {}, states = {};
  let charged = 0;

  F.push({shapes:knapShapes({}, S.t(4.9, 5.75,
      {zh:'選一門課就必須連它的先修課一起選，所以物品排成一棵樹',
       en:'a course can only be taken with its prerequisite: items form a tree'},
      {c:'#3fe0dd', fs:.26}), null, null, 0),
    view:VIEW, line:0,
    panels:[{lbl:{zh:'預算', en:'budget'}, chips:[chip(BUDGET + ' slots (root is free)', 'dim')]},
            {lbl:{zh:'學分', en:'credits'},
             chips:[1, 2, 3, 4, 5].map(v => chip(v + ':' + CREDITS[v], 'dim'))}],
    msg:{zh:'Day 34 的背包裡，物品彼此無關；這裡有依賴關係，拿下面的就必須先拿上面的。做法是讓每個節點各自維護一張「在我這棵子樹裡恰好選 t 門」的表，再把小孩的表一張一張併進來——一個小背包接著一個小背包。',
         en:'On day 34 the items were independent; here they have prerequisites, so choosing a node commits you to its ancestors. The fix is to give every node its own table - "exactly t courses taken inside my subtree" - and merge the children into it one at a time, a small knapsack per child.'}});

  for (const v of KPOST){
    let dp = new Array(BUDGET + 1).fill(0);
    for (let t = 1; t <= BUDGET; t++) dp[t] = CREDITS[v];
    size[v] = 1;
    const st0 = Object.assign({}, states); st0[v] = 'hot';
    F.push({shapes:knapShapes(st0, S.t(4.9, 5.75,
        {zh:'先放自己：子樹裡只選我一個的時候，就是我的學分',
         en:'start with the node itself: this course alone is worth its credits'},
        {c:'#8fa3ac', fs:.26}), dp.slice(), 'dp[' + v + ']', charged),
      view:VIEW, line:1,
      panels:[{lbl:{zh:'節點', en:'node'}, chips:[chip('course ' + v, 'hot')]},
              {lbl:{zh:'size', en:'size'}, chips:[chip('1', 'dim')]}],
      msg:{zh:'dp[' + v + '][t] 的意思是「在 ' + v + ' 這棵子樹裡、而且一定包含 ' + v + '、恰好選 t 門時的最高學分」。「一定包含自己」這個條件就是先修關係在表裡的樣子——沒有它，小孩可以在父親沒被選的情況下被選走。',
           en:'dp[' + v + '][t] means "the best credits from taking exactly t courses inside ' + v + '’s subtree, with ' + v + ' itself included". That "itself included" clause is what the prerequisite looks like inside the table - without it a child could be chosen while its parent was not.'}});

    for (const c of COURSES[v]){
      const cdp = dpOf[c], nd = dp.slice();
      const units = size[v] * size[c];
      for (let t = Math.min(size[v], BUDGET); t >= 1; t--)
        for (let u = 1; u <= Math.min(size[c], BUDGET - t); u++)
          nd[t + u] = Math.max(nd[t + u], dp[t] + cdp[u]);
      charged += units;
      const st = Object.assign({}, states); st[v] = 'hot';
      for (let i = 0; i < KARR.length; i++) if (KARR[i] === c) st[i] = 'act';
      F.push({shapes:knapShapes(st, S.t(4.9, 5.75,
          'size(' + c + ') x size(merged) = ' + size[c] + ' x ' + size[v] + ' = ' + units
          + ' units of inner-loop work',
          {c:'#ffbe6b', fs:.28}), nd.slice(), 'dp[' + v + ']', charged),
        view:VIEW, line:7,
        panels:[{lbl:{zh:'併入', en:'merging'}, chips:[chip('child ' + c, 'act')]},
                {lbl:{zh:'小孩的表', en:'child table'}, chips:cdp.map((x, i) => chip(i + ':' + x, 'dim'))},
                {lbl:{zh:'併完', en:'after merge'}, chips:nd.map((x, i) => chip(i + ':' + x, 'ok'))}],
        msg:{zh:'合併就是一個小背包：父親那邊已經選了 t 門，小孩這邊再選 u 門。迴圈的範圍不是隨便寫的——上限用的是子樹大小而不是預算，這正是整個複雜度論證的來源：這一步做了 size(' + c + ') x size(已併) 個單位的工。',
             en:'The merge is itself a tiny knapsack: t courses already chosen on the parent side, u more from this child. The loop bounds are not decoration - they are capped by subtree size rather than by budget, which is where the whole complexity argument comes from: this step does size(' + c + ') x size(merged) units of work.'}});
      dp = nd; size[v] += size[c];
    }
    dpOf[v] = dp;
    states[v] = 'done';
  }

  const best = dpOf[0][BUDGET];
  const n = 6, expected = n * (n - 1) / 2;
  F.push({shapes:knapShapes(states, S.t(4.9, 5.75,
      'every pair of nodes charged exactly once:  ' + charged + ' = C(6, 2) = ' + expected,
      {c:'#3fe0dd', fs:.29}), dpOf[0].slice(), 'dp[0]', charged),
    view:VIEW, line:11,
    panels:[{lbl:{zh:'最佳學分', en:'best credits'},
             chips:[chip(String(best) + ' with ' + (BUDGET - 1) + ' real courses', 'ok')]},
            {lbl:{zh:'內圈總工作量', en:'total inner-loop work'}, chips:[chip(String(charged), 'act')]},
            {lbl:{zh:'C(6, 2)', en:'C(6, 2)'}, chips:[chip(String(expected), 'dim')]}],
    msg:{zh:'看起來像三層迴圈，複雜度卻不是 O(n · budget²)。關鍵是一個計數論證：在節點 v 做的 size(child) x size(已併) 個單位，剛好就是「最近共同祖先是 v」的那些節點對 (u, w) 的數量。每一對節點在整棵樹裡只會有一個 LCA，所以全部加起來就是 C(n, 2) ——整個合併是 O(n²)，加上預算上限之後是 O(n · budget)。',
         en:'It looks like a triple loop, but it is not O(n * budget^2). The key is a counting argument: the size(child) x size(merged) units spent at node v are exactly the pairs (u, w) whose lowest common ancestor is v. Every pair of nodes has exactly one LCA in the tree, so summed over the whole tree each pair is charged once - C(n, 2) in total, making the merge O(n^2), and O(n * budget) once the bounds are capped.'}});
  return F.list;
}

/* ======================================================================== */
const DAY_META = {
  title:{zh:'Day 36 — 區間 DP 與樹形 DP', en:'Interval DP and tree DP'},
  sub:{zh:'表格的索引從「前 i 個」換成「一段 [i, j]」和「一棵子樹」，每一步要決定的也從「拿不拿下一個」換成「從哪裡切」和「小孩往上回報什麼」。',
       en:'The table stops being indexed by a prefix and starts being indexed by a segment [i, j] or a subtree - and the decision at each step stops being "take the next item" and becomes "where do I split" or "what does each child report upward".'},
  tabs:[
    {
      id:'chain', label:{zh:'矩陣鏈乘：迴圈順序', en:'matrix chain: the loop order'},
      stage:{zh:'三角形的表，外圈一定要是區間長度',
             en:'a triangular table whose outer loop has to be interval length'},
      view:VIEW,
      variants:[{zh:'長度優先（正確）', en:'length first (correct)'},
                {zh:'i 再 j（錯誤）', en:'i then j (wrong)'}],
      idea:{zh:'矩陣乘法有結合律，所以怎麼加括號都算得出同一個結果，但成本差很多：((A(BC))D) 要 26,000 次純量乘法，最糟的括號法要 69,000 次，而括號的種類是 Catalan 數，枚舉是沒有希望的。dp[i][j] 只問一件事：最上面那一次乘法切在哪裡。真正的陷阱是填表順序——dp[i][j] 會讀到 dp[k+1][j]，那是「更下面那一列」，用最直覺的 i-then-j 雙層迴圈時它還是 0。沒有例外、沒有警告，答案只是變成一個根本不存在的 20,000。',
            en:'Matrix multiplication is associative, so every bracketing computes the same matrix - at wildly different prices: ((A(BC))D) costs 26,000 scalar multiplications and the worst bracketing costs 69,000, while the number of bracketings is a Catalan number, so enumeration is hopeless. dp[i][j] asks one question: where does the topmost multiplication split? The real trap is the fill order - dp[i][j] reads dp[k+1][j], a row *below* it, which the instinctive i-then-j double loop has not filled yet. No exception, no warning, just an answer of 20,000 that corresponds to no bracketing at all.'},
      legend:[['#ff9736', {zh:'正在填的格子', en:'the cell being filled'}],
              ['#9d6bff', {zh:'讀到的兩格', en:'the two cells it reads'}],
              ['#3fe0dd', {zh:'已填好', en:'already filled'}],
              ['#ff5c5c', {zh:'還沒算就被讀走', en:'read before it was computed'}]],
      code:CODE_CHAIN, build:chainFrames
    },
    {
      id:'burst', label:{zh:'LC 312 戳氣球', en:'LC 312 burst balloons'},
      stage:{zh:'切在「最後戳的那一顆」，兩半才會互不影響',
             en:'split on the balloon popped LAST, and only then are the halves independent'},
      view:VIEW,
      variants:[{zh:'最後戳（正確）', en:'popped last (correct)'},
                {zh:'先戳（錯誤）', en:'popped first (wrong)'}],
      idea:{zh:'戳氣球的分數取決於當下的鄰居，所以「先戳哪一顆」這個最直覺的切法會壞掉：k 一消失，左半的右鄰居就變成右半的氣球，兩邊不再獨立。改問「哪一顆最後戳」就好了——如果 k 是這一段裡最後走的，那戳它的瞬間中間全空了，鄰居必然是兩端那兩面不會動的牆。同一張三角形的表、同一個長度優先迴圈，只有切的方向不同：98 和 167 的差別就在這裡，而且錯的版本一樣跑得完。LeetCode 1547 切木棍是它的鏡像，那邊是第一刀才切得開。',
            en:'The coins a balloon earns depend on its neighbours at that moment, so the instinctive "which one do I pop first" split breaks: the instant k is gone, the left half’s right-hand neighbour is a balloon from the right half and the two sides stop being independent. Ask which balloon goes *last* instead - if k is last inside the segment then everything between the walls is already gone when it pops, so its neighbours are exactly the two fixed walls. Same triangular table, same length-first loop, only the direction of the split changes, and that is the whole difference between 98 and 167 - with the wrong version running just as happily. LeetCode 1547, cutting a stick, is the mirror image, where the *first* cut is the separable one.'},
      legend:[['#ff9736', {zh:'這一步戳掉的', en:'popped at this step'}],
              ['#9d6bff', {zh:'它當下的鄰居', en:'its neighbours right now'}],
              ['#3fe0dd', {zh:'正確的答案', en:'the correct answer'}],
              ['#ff5c5c', {zh:'不獨立的子問題', en:'subproblems that are not independent'}]],
      code:CODE_BURST, build:burstFrames
    },
    {
      id:'rob', label:{zh:'LC 337 樹上打家劫舍', en:'LC 337 house robber III'},
      stage:{zh:'每個節點要回報一對狀態，一個數字是不夠的',
             en:'a node has to report a pair of states - one number is not enough'},
      view:VIEW,
      variants:[{zh:'一對（拿我／不拿我）', en:'a pair: (take me, skip me)'},
                {zh:'只回報一個數字（錯誤）', en:'one number per node (wrong)'}],
      idea:{zh:'狀態的形狀比遞迴本身難。「這棵子樹最多能拿多少」是最像題目的說法，但父節點要決定自己能不能拿，必須知道小孩到底有沒有被拿，而這個數字已經把那件事丟掉了；在全正值的測資上它甚至不會看起來怪，因為它算出來的就是所有節點的總和。改成回報一對數字，兩個狀態就重新變得獨立：要拿自己，小孩只能用 skip；不拿自己，小孩兩個值取大。同樣的「回報的東西不等於計分的東西」也撐起 LeetCode 124 和樹的直徑。',
            en:'The shape of the state is harder than the recursion. "The most this subtree can give" is the phrasing closest to the question, but the parent has to know whether the child was actually taken in order to decide about itself, and that single number has thrown the fact away - on all-positive inputs it does not even look odd, because what it computes is the sum of every node. Report a pair instead and the two states become independent again: to take this node the children may only contribute their skip value, and to skip it each child contributes the better of its two. The same "what a node returns is not what it scores" split drives LeetCode 124 and the tree diameter.'},
      legend:[['#ff9736', {zh:'正在處理的節點', en:'the node being processed'}],
              ['#9d6bff', {zh:'它的小孩', en:'its children'}],
              ['#3fe0dd', {zh:'最後選中的節點', en:'the nodes finally chosen'}],
              ['#ff5c5c', {zh:'違規：相鄰同時被選', en:'illegal: adjacent nodes both taken'}]],
      code:CODE_ROB, build:robFrames
    },
    {
      id:'knap', label:{zh:'樹上背包', en:'tree knapsack'},
      stage:{zh:'把小孩的表一張一張併進來，每一對節點只被算一次',
             en:'merge the children in one table at a time - and every pair of nodes is charged exactly once'},
      view:VIEW,
      idea:{zh:'Day 34 的背包物品彼此無關，這裡的物品有先修關係，於是排成一棵樹。每個節點維護「在我這棵子樹裡、包含我自己、恰好選 t 門」的表，小孩的表一張一張併進來，每一次合併都是一個小背包。最有意思的是複雜度：看起來是三層迴圈，實際上在節點 v 做的 size(child) x size(已併) 個單位，剛好就是「最近共同祖先是 v」的節點對數量；每一對節點只有一個 LCA，所以全樹加起來是 C(n, 2)，合併總共 O(n²)，再被預算上限壓成 O(n · budget)。迴圈的上限寫 min(size, budget) 不是優化，它就是這個論證本身。',
            en:'On day 34 the knapsack items were independent; here they have prerequisites, so they form a tree. Every node keeps a table of "exactly t courses taken inside my subtree, myself included", and the children are merged in one at a time, each merge a small knapsack of its own. The complexity is the interesting part: it looks like a triple loop, but the size(child) x size(merged) units spent at node v are precisely the pairs of nodes whose lowest common ancestor is v. Every pair has exactly one LCA, so over the whole tree each pair is charged once - C(n, 2) in total, O(n^2) for the merge, collapsing to O(n * budget) once the bounds are capped. Writing min(size, budget) as the loop bound is not an optimisation; it *is* the argument.'},
      legend:[['#ff9736', {zh:'正在合併的父節點', en:'the parent being merged into'}],
              ['#9d6bff', {zh:'併進來的小孩', en:'the child being merged'}],
              ['#3fe0dd', {zh:'完成的子樹', en:'a finished subtree'}],
              ['#8fa3ac', {zh:'還沒處理', en:'not reached yet'}]],
      code:CODE_KNAP, build:knapFrames
    }
  ]
};
