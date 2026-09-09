// DAY: 25
// TITLE_ZH: 字串匹配：KMP 與 Rabin-Karp
// TITLE_EN: String matching - KMP and Rabin-Karp
// SUB_ZH: KMP 靠 failure function 記住「已經比過的那一段」，永遠不回頭讀文字；Rabin-Karp 先比指紋再比字串。推論伺服器裡的 stop string 與 KV cache block id，用的就是這兩件事。
// SUB_EN: KMP remembers what it has already compared and never re-reads the text; Rabin-Karp compares fingerprints first and strings only when it must. Stop strings and KV cache block ids in an inference server are these two ideas.
// FOLDER: day%2025%20-%20string%20matching
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const mid2 = VIEW[0] / 2;
const ln = (code, frag) => { const i = code.findIndex(l => l.indexOf(frag) >= 0); return i < 0 ? 0 : i; };
const note = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});

function norm(list){
  list.forEach(f => {
    if (f.panels && f.panels.length && f.panels[0].chips === undefined)
      f.panels = [{lbl:{zh:'狀態', en:'state'}, chips:f.panels}];
  });
  return list;
}
function chip(t, cls){ return {t:t, cls:cls || ''}; }

/* the reference implementation, ported verbatim from string_matching.py */
function buildFailure(pat){
  const m = pat.length, fail = new Array(m).fill(0);
  let k = 0;
  for (let i = 1; i < m; i++){
    while (k > 0 && pat[i] !== pat[k]) k = fail[k - 1];
    if (pat[i] === pat[k]) k++;
    fail[i] = k;
  }
  return fail;
}

/* ================================================== 1. the naive matcher */
const CODE_N = [
'def naive_search(text, pat):',
'    n, m = len(text), len(pat)',
'    hits, cmps = [], 0',
'    for i in range(n - m + 1):        # every alignment',
'        j = 0',
'        while j < m:',
'            cmps += 1',
'            if text[i + j] != pat[j]:',
'                break                 # give up, slide by one',
'            j += 1',
'        if j == m:',
'            hits.append(i)',
'    return hits, cmps'
];

function naiveFrames(v){
  const text = v ? 'abcabcabd' : 'aaaaaaab';
  const pat  = v ? 'abcabd'    : 'aaab';
  const F = new Frames(), n = text.length, m = pat.length;
  const w = .68, x0 = (VIEW[0] - n * w) / 2, yT = 1.35, yP = 2.75;
  let cmps = 0, hits = [];

  const draw = (shift, j, st) => {
    const ts = {}, ps = {};
    for (let k = 0; k < shift; k++) ts[k] = 'done';
    if (j != null){
      for (let k = 0; k < j; k++){ ts[shift + k] = 'ok'; ps[k] = 'ok'; }
      ts[shift + j] = st; ps[j] = st;
    }
    const out = cellRow(text.split(''), x0, yT, w, .78, {states:ts, index:false});
    out.push(S.t(x0 - .25, yT + .48, 'text', {c:COL.tealL, fs:.32, anchor:'end'}));
    pat.split('').forEach((c, k) => {
      out.push(S.r(x0 + (shift + k) * w, yP, w - .06, .78, ps[k] || 'soft', c, {fs:.40}));
    });
    out.push(S.t(x0 - .25, yP + .48, 'pattern', {c:COL.tealL, fs:.32, anchor:'end'}));
    return out;
  };
  const panels = (shift, j) => [
    {lbl:{zh:'對齊位置 i', en:'alignment i'}, chips:[chip(String(shift), 'act')]},
    {lbl:{zh:'已比對 j', en:'matched j'}, chips:[chip(String(j), j ? 'ok' : '')]},
    {lbl:{zh:'字元比較次數', en:'character comparisons'}, chips:[chip(String(cmps), 'hot')]}
  ];

  F.push({shapes:draw(0, null), panels:panels(0, 0), view:VIEW, line:ln(CODE_N, 'for i in range'),
    msg:{zh:'天真版的想法只有一句：<b>每一個對齊位置都從頭比一次</b>。程式非常短，但要注意成本不是「對齊次數」，而是<b>字元比較次數</b> - 這兩個數字在最壞情況差了一個 m 倍。',
         en:'The naive matcher is one sentence: <b>try every alignment, and compare from scratch each time</b>. The code is tiny, but the cost is not the number of alignments - it is the number of <b>character comparisons</b>, and in the worst case those differ by a factor of m.'}});

  for (let i = 0; i + m <= n; i++){
    let j = 0;
    while (j < m){
      cmps++;
      const good = text[i + j] === pat[j];
      F.push({shapes:draw(i, j, good ? 'hot' : 'bad'), panels:panels(i, j), view:VIEW,
        line:ln(CODE_N, good ? '            j += 1' : 'if text[i + j] != pat[j]'),
        msg:good
          ? {zh:'text[' + (i + j) + '] 和 pat[' + j + '] 都是 <b>' + text[i + j] + '</b>，往右一格。這一次比較之後我們<b>知道了一件事</b>：文字的這一段就等於 pattern 的前 ' + (j + 1) + ' 個字元 - 記住這件事，KMP 就是靠它。',
             en:'text[' + (i + j) + '] and pat[' + j + '] are both <b>' + text[i + j] + '</b>, so step right. This comparison <b>told us something</b>: this stretch of the text equals the first ' + (j + 1) + ' characters of the pattern - remember that, it is the whole of KMP.'}
          : {zh:'不合：text[' + (i + j) + '] = <b>' + text[i + j] + '</b>，pat[' + j + '] = <b>' + pat[j] + '</b>。天真版在這裡<b>把已經比過的 ' + j + ' 個字元全部丟掉</b>，對齊位置往右一格，然後從 j = 0 重來 - 剛剛那 ' + j + ' 次比較的結果一點都沒有用到。',
             en:'Mismatch: text[' + (i + j) + '] = <b>' + text[i + j] + '</b> against pat[' + j + '] = <b>' + pat[j] + '</b>. The naive matcher now <b>throws away the ' + j + ' characters it just matched</b>, slides one place right and restarts at j = 0 - nothing learned from those ' + j + ' comparisons is used.'}});
      if (!good) break;
      j++;
    }
    if (j === m){
      hits.push(i);
      F.push({shapes:draw(i, m - 1, 'ok'), panels:panels(i, j), view:VIEW, line:ln(CODE_N, 'hits.append'),
        msg:{zh:'在位置 <b>' + i + '</b> 找到了。到目前為止用掉 <b>' + cmps + '</b> 次字元比較。',
             en:'A match at position <b>' + i + '</b>, after <b>' + cmps + '</b> character comparisons so far.'}});
    }
  }
  const kmpCmps = (function(){
    const fail = buildFailure(pat); let c = 0, j = 0;
    for (let i = 0; i < n; i++){
      while (j > 0 && text[i] !== pat[j]){ c++; j = fail[j - 1]; }
      c++;
      if (text[i] === pat[j]) j++;
      if (j === m) j = fail[j - 1];
    }
    return c;
  })();
  F.push({shapes:draw(0, null).concat([note(4.6, 'naive ' + cmps + ' comparisons   vs   KMP ' + kmpCmps, COL.orangeL, .40)]),
    panels:panels(0, 0), view:VIEW, line:ln(CODE_N, 'return hits'),
    msg:{zh:'總共 <b>' + cmps + '</b> 次比較，同一份輸入 KMP 只要 <b>' + kmpCmps + '</b> 次。差別完全來自「不合的時候要退回多遠」 - 天真版退回到<b>下一個對齊位置的開頭</b>，KMP 只退 pattern 的索引，<b>文字的索引一步都不退</b>。',
         en:'<b>' + cmps + '</b> comparisons in total, where KMP needs <b>' + kmpCmps + '</b> on the same input. The entire difference is how far you go back on a mismatch: the naive version returns to <b>the start of the next alignment</b>, while KMP moves only the pattern index and <b>never moves the text index backwards at all</b>.'}});
  return F.list;
}

/* ============================================== 2. failure function / KMP */
const CODE_K = [
'def build_failure(pat):',
'    fail = [0] * len(pat)',
'    k = 0                              # length of the current border',
'    for i in range(1, len(pat)):',
'        while k > 0 and pat[i] != pat[k]:',
'            k = fail[k - 1]            # fall back to a shorter border',
'        if pat[i] == pat[k]:',
'            k += 1',
'        fail[i] = k',
'    return fail',
'',
'def kmp_search(text, pat):',
'    fail, j, hits = build_failure(pat), 0, []',
'    for i, ch in enumerate(text):      # i never goes backwards',
'        while j > 0 and ch != pat[j]:',
'            j = fail[j - 1]',
'        if ch == pat[j]:',
'            j += 1',
'        if j == len(pat):',
'            hits.append(i - len(pat) + 1)',
'            j = fail[j - 1]',
'    return hits'
];

function kmpBuildFrames(){
  const pat = 'ababaca', m = pat.length, F = new Frames();
  const w = .82, x0 = (VIEW[0] - m * w) / 2, yP = 1.5, yF = 2.9;
  const fail = new Array(m).fill(null);
  let k = 0;

  const draw = (i, kk, sts) => {
    const out = cellRow(pat.split(''), x0, yP, w, .8, {states:sts || {}, index:false, fs:.42});
    out.push(S.t(x0 - .25, yP + .45, 'pat', {c:COL.tealL, fs:.32, anchor:'end'}));
    const fs = {};
    for (let q = 0; q < m; q++) if (fail[q] != null) fs[q] = (q === i ? 'ok' : 'done');
    out.push.apply(out, cellRow(fail.map(v => v == null ? '' : v), x0, yF, w, .7, {states:fs, index:false, fs:.38}));
    out.push(S.t(x0 - .25, yF + .40, 'fail', {c:COL.tealL, fs:.32, anchor:'end'}));
    if (i != null) out.push(S.t(x0 + i * w + w / 2 - .03, yP - .40, 'i', {c:COL.orangeL, fs:.34}));
    out.push(S.t(x0 + kk * w + w / 2 - .03, yP + 1.28, 'k=' + kk, {c:COL.purpleL, fs:.34}));
    return out;
  };
  const panels = (i, kk) => [
    {lbl:{zh:'i（正在填哪一格）', en:'i (cell being filled)'}, chips:[chip(String(i), 'hot')]},
    {lbl:{zh:'k（目前 border 長度）', en:'k (current border length)'}, chips:[chip(String(kk), 'act')]},
    {lbl:{zh:'fail', en:'fail'}, chips:fail.map(v => chip(v == null ? '·' : String(v), v == null ? '' : 'ok'))}
  ];

  F.push({shapes:draw(0, 0, {0:'ok'}), panels:panels(0, 0), view:VIEW, line:1,
    msg:{zh:'fail[i] 的定義只有一句話：<b>pat[0..i] 這一段裡，最長的「開頭也是結尾」的長度</b>（不能是整段）。這種開頭＝結尾的片段叫 <b>border</b>。fail[0] 一定是 0，因為單一字元不能拿自己當 border。',
         en:'fail[i] has a one-line definition: <b>the length of the longest prefix of pat[0..i] that is also its suffix</b> (the whole thing does not count). Such a piece is called a <b>border</b>. fail[0] is always 0 - a single character cannot borrow itself.'}});
  fail[0] = 0;

  for (let i = 1; i < m; i++){
    while (k > 0 && pat[i] !== pat[k]){
      F.push({shapes:draw(i, k, {[i]:'bad', [k]:'bad'}), panels:panels(i, k), view:VIEW, line:ln(CODE_K, 'k = fail[k - 1]'),
        msg:{zh:'pat[' + i + '] = <b>' + pat[i] + '</b> 接不上 pat[' + k + '] = <b>' + pat[k] + '</b>，所以長度 ' + k + ' 的 border 延伸不下去。但不必從 0 重來：<b>比它短的 border 一定是它的 border</b>，所以 k 直接跳到 fail[' + (k - 1) + '] = <b>' + fail[k - 1] + '</b>。這一行就是整個 KMP 的核心。',
             en:'pat[' + i + '] = <b>' + pat[i] + '</b> does not follow pat[' + k + '] = <b>' + pat[k] + '</b>, so the border of length ' + k + ' cannot be extended. There is no need to restart at 0: <b>a shorter border of the pattern is itself a border of that border</b>, so k jumps straight to fail[' + (k - 1) + '] = <b>' + fail[k - 1] + '</b>. This single line is the heart of KMP.'}});
      k = fail[k - 1];
    }
    const grow = pat[i] === pat[k];
    if (grow) k++;
    fail[i] = k;
    F.push({shapes:draw(i, Math.max(k - (grow ? 1 : 0), 0), {[i]:grow ? 'ok' : 'soft'}), panels:panels(i, k), view:VIEW,
      line:ln(CODE_K, grow ? '            k += 1' : '        fail[i] = k'),
      msg:grow
        ? {zh:'pat[' + i + '] = <b>' + pat[i] + '</b> 剛好等於 pat[' + (k - 1) + ']，border 長一格：fail[' + i + '] = <b>' + k + '</b>。也就是說 <b>' + pat.slice(0, k) + '</b> 同時是 <b>' + pat.slice(0, i + 1) + '</b> 的開頭和結尾。',
           en:'pat[' + i + '] = <b>' + pat[i] + '</b> matches pat[' + (k - 1) + '], so the border grows by one: fail[' + i + '] = <b>' + k + '</b>. In other words <b>' + pat.slice(0, k) + '</b> is both the head and the tail of <b>' + pat.slice(0, i + 1) + '</b>.'}
        : {zh:'一路退到 k = 0 還是接不上，fail[' + i + '] = <b>0</b>：<b>' + pat.slice(0, i + 1) + '</b> 沒有任何 border，比對到這裡失敗時只能整個重來。',
           en:'Even at k = 0 there is no match, so fail[' + i + '] = <b>0</b>: <b>' + pat.slice(0, i + 1) + '</b> has no border at all, and a mismatch here means starting the pattern over.'}});
  }
  const per = m - fail[m - 1];
  F.push({shapes:draw(m - 1, fail[m - 1], {}).concat([note(4.7, 'fail = [' + fail.join(', ') + ']', COL.tealL, .40),
      note(4.15, 'n - fail[n-1] = ' + m + ' - ' + fail[m - 1] + ' = ' + per + '  (the shortest period)', COL.orangeL, .34)]),
    panels:panels(m - 1, fail[m - 1]), view:VIEW, line:ln(CODE_K, '    return fail'),
    msg:{zh:'表建好了，成本是 <b>O(m)</b>：k 每次最多加一，而 while 迴圈只能把 k 往下拉，所以總下拉次數不會超過總上升次數。順手得到一個副產品：<b>n − fail[n−1] = ' + per + '</b> 就是這個字串的最短週期，LeetCode 459 整題就是這一行。',
         en:'The table is done in <b>O(m)</b>: k rises by at most one per step and the while loop only pulls it down, so the total number of pull-downs cannot exceed the total rises. A free by-product: <b>n − fail[n−1] = ' + per + '</b> is the shortest period of the string, which is the whole of LeetCode 459.'}});
  return F.list;
}

function kmpSearchFrames(){
  const text = 'abababacaba', pat = 'ababaca', n = text.length, m = pat.length;
  const fail = buildFailure(pat), F = new Frames();
  const w = .70, x0 = (VIEW[0] - n * w) / 2, yT = 1.35, yP = 2.75;
  let j = 0, hits = [];

  const draw = (i, jj, st) => {
    const ts = {}; for (let q = 0; q < i; q++) ts[q] = 'done';
    if (i != null) ts[i] = st;
    const out = cellRow(text.split(''), x0, yT, w, .76, {states:ts, index:false, fs:.40});
    out.push(S.t(x0 - .25, yT + .42, 'text', {c:COL.tealL, fs:.32, anchor:'end'}));
    const shift = i - jj, ps = {};
    for (let q = 0; q < jj; q++) ps[q] = 'ok';
    ps[jj] = st;
    pat.split('').forEach((c, q) => {
      out.push(S.r(x0 + (shift + q) * w, yP, w - .06, .76, ps[q] || 'soft', c, {fs:.40}));
    });
    out.push(S.t(x0 - .25, yP + .42, 'pattern', {c:COL.tealL, fs:.32, anchor:'end'}));
    out.push(S.t(x0 + i * w + w / 2 - .03, yT - .38, 'i=' + i, {c:COL.orangeL, fs:.32}));
    return out;
  };
  const panels = (i, jj) => [
    {lbl:{zh:'i（文字位置，只前進）', en:'i (text index, forward only)'}, chips:[chip(String(i), 'hot')]},
    {lbl:{zh:'j（已對上的長度）', en:'j (matched length)'}, chips:[chip(String(jj), 'act')]},
    {lbl:{zh:'fail', en:'fail'}, chips:fail.map(v => chip(String(v)))},
    {lbl:{zh:'命中位置', en:'hits'}, chips:hits.length ? hits.map(v => chip(String(v), 'ok')) : [chip('—')]}
  ];

  F.push({shapes:draw(0, 0, 'hot'), panels:panels(0, 0), view:VIEW, line:ln(CODE_K, 'def kmp_search'),
    msg:{zh:'有了 fail 表，搜尋就只剩一個 for 迴圈掃過文字。<b>i 只會往右</b>，永遠不回頭 - 這代表就算文字是一個讀不回去的 stream（socket、token 串流），這個演算法也能用。',
         en:'With the table in hand the search is one loop over the text. <b>i only moves right</b> and never rewinds - which means the algorithm still works when the text is a stream you cannot seek backwards in, like a socket or a token stream.'}});

  for (let i = 0; i < n; i++){
    while (j > 0 && text[i] !== pat[j]){
      const nj = fail[j - 1];
      F.push({shapes:draw(i, j, 'bad'), panels:panels(i, j), view:VIEW, line:ln(CODE_K, '            j = fail[j - 1]'),
        msg:{zh:'text[' + i + '] = <b>' + text[i] + '</b> 對不上 pat[' + j + ']。天真版會把對齊位置往右挪一格重來，KMP 不需要：剛剛那 ' + j + ' 個字元<b>就是 pattern 的前 ' + j + ' 個字元</b>，所以下一個還有機會的對齊位置，只能從 border 開始 - j 從 ' + j + ' 掉到 <b>' + nj + '</b>，<b>i 一步都不退</b>。',
             en:'text[' + i + '] = <b>' + text[i] + '</b> fails against pat[' + j + ']. The naive matcher would slide one place and restart; KMP does not have to. Those ' + j + ' characters <b>are</b> the first ' + j + ' characters of the pattern, so the only alignment still worth trying starts at the border: j drops from ' + j + ' to <b>' + nj + '</b> and <b>i does not move at all</b>.'}});
      j = nj;
    }
    const good = text[i] === pat[j];
    if (good) j++;
    let done = false;
    if (j === m){ hits.push(i - m + 1); done = true; }
    F.push({shapes:draw(i, Math.max(j - (good ? 1 : 0), 0), done ? 'ok' : (good ? 'hot' : 'bad')), panels:panels(i, j), view:VIEW,
      line:ln(CODE_K, done ? '            hits.append' : (good ? '            j += 1' : '        if ch == pat[j]')),
      msg:done
        ? {zh:'j 到了 ' + m + '，在位置 <b>' + (i - m + 1) + '</b> 找到完整的 pattern。找到之後 j 一樣退到 fail[' + (m - 1) + '] = <b>' + fail[m - 1] + '</b>，讓<b>重疊的下一個出現</b>也能被抓到。',
             en:'j reached ' + m + ': a full match ending here, starting at <b>' + (i - m + 1) + '</b>. After reporting it j drops to fail[' + (m - 1) + '] = <b>' + fail[m - 1] + '</b>, so an <b>overlapping</b> next occurrence is still found.'}
        : good
        ? {zh:'對上了，j = <b>' + j + '</b>。目前文字的結尾有 ' + j + ' 個字元和 pattern 的開頭一樣。',
           en:'Match, so j = <b>' + j + '</b>: the last ' + j + ' characters of the text seen so far equal the first ' + j + ' of the pattern.'}
        : {zh:'j 已經是 0，這個字元只能直接跳過。',
           en:'j is already 0, so this character is simply skipped.'}});
    if (done) j = fail[m - 1];
  }
  F.push({shapes:draw(n - 1, 0, 'done'), panels:panels(n - 1, j), view:VIEW, line:ln(CODE_K, '    return hits'),
    msg:{zh:'掃完了，命中 <b>[' + hits.join(', ') + ']</b>。全程 <b>O(n + m)</b>，而且記憶體只用了 fail 表 - 這也是為什麼推論伺服器可以用它一邊產生 token 一邊比對 stop string。',
         en:'Done, hits at <b>[' + hits.join(', ') + ']</b>, in <b>O(n + m)</b> with nothing but the fail table in memory - which is why an inference server can run it against a token stream as the tokens appear.'}});
  return F.list;
}

/* ================================== 3. a stop string that arrives in pieces */
const CODE_S = [
'class StreamStopMatcher:                    # one KMP state per stop string',
'    def feed(self, chunk):',
'        self.buf += chunk',
'        for i, ch in enumerate(chunk):',
'            for p, pat in enumerate(self.stops):',
'                j = self.state[p]',
'                while j > 0 and ch != pat[j]:',
'                    j = self.fails[p][j - 1]',
'                if ch == pat[j]:',
'                    j += 1',
'                self.state[p] = j           # survives the call',
'                if j == len(pat):',
'                    end = len(self.buf) - (len(chunk) - i - 1)',
'                    out = self.buf[: end - len(pat)]',
'                    self.buf = ""',
'                    return out, pat',
'        hold = max(self.state)              # might be the start of a stop',
'        cut = len(self.buf) - hold',
'        out, self.buf = self.buf[:cut], self.buf[cut:]',
'        return out, None'
];

function stopFrames(){
  const stops = ['</s>', '\n\nHuman:'];
  const fails = stops.map(buildFailure);
  const chunks = ['The answer is 4', '2.<', '/s', '> and then some more'];
  const F = new Frames();
  const shown = 'The answer is 42.</s>';          // everything consumed before the stop fires
  const N = shown.length, w = .42, x0 = (VIEW[0] - N * w) / 2, y = 2.2;
  const bounds = [[0, 15], [15, 18], [18, 20], [20, 21]];
  let state = [0, 0], emitted = 0, cur = 0, matched = null;

  const draw = (hi, hold) => {
    const st = {};
    for (let q = 0; q < N; q++){
      if (q >= cur + (hi == null ? 0 : 1)) st[q] = 'ghost';
      else if (q < emitted) st[q] = 'done';
      else st[q] = 'act';
    }
    if (hi != null) st[hi] = matched ? 'ok' : 'hot';
    const out = cellRow(shown.split(''), x0, y, w, .62, {states:st, index:false, fs:.30});
    bounds.forEach((b, k) => {
      if (b[0] > cur) return;
      const xa = x0 + b[0] * w + .04, xb = x0 + Math.min(b[1], cur + 1) * w - .10;
      const yy = y + .78 + (k % 2) * .34;
      out.push(S.e(xa, yy, xb, yy, {s:'soft', arrow:false, w:.05}));
      out.push(S.t((xa + xb) / 2, yy + .26, 'chunk ' + (k + 1), {c:COL.grey, fs:.26}));
    });
    if (emitted > 0) out.push(S.t(x0 + emitted * w / 2, y - .40, 'emitted', {c:COL.tealL, fs:.28}));
    if (hold) out.push(S.t(x0 + (emitted + hold / 2) * w, y - .40, 'held back (' + hold + ')', {c:COL.purpleL, fs:.28}));
    return out;
  };
  const panels = () => [
    {lbl:{zh:"'</s>' 對上幾個字", en:"matched length of '</s>'"}, chips:[chip(String(state[0]), state[0] ? 'act' : '')]},
    {lbl:{zh:"'\\n\\nHuman:' 對上幾個字", en:"matched length of '\\n\\nHuman:'"}, chips:[chip(String(state[1]), state[1] ? 'act' : '')]},
    {lbl:{zh:'已送出字元', en:'characters emitted'}, chips:[chip(String(emitted), 'ok')]},
    {lbl:{zh:'扣住不送', en:'held back'}, chips:[chip(String(cur - emitted), cur - emitted ? 'hot' : '')]}
  ];

  F.push({shapes:draw(null, 0), panels:panels(), view:VIEW, line:0,
    msg:{zh:'伺服器一邊生 token 一邊往外吐字，但 stop string 可能<b>被切在兩個 chunk 中間</b>。不能等全部生完再比對（那就不是 streaming 了），也不能每次都把整段輸出重掃一次。KMP 的狀態機剛好符合：<b>只留一個整數 j</b>，一個字元進來就更新一次。',
         en:'The server streams text out while it is still generating, but a stop string can be <b>split across two chunks</b>. Waiting for the end is not streaming, and rescanning the whole output every time is wasteful. A KMP state machine fits exactly: <b>keep one integer j</b> and update it per character.'}});

  for (let c = 0; c < chunks.length && !matched; c++){
    const chunk = chunks[c];
    for (let i = 0; i < chunk.length; i++){
      const ch = chunk[i];
      cur = bounds[c][0] + i;
      let fired = null;
      for (let p = 0; p < stops.length; p++){
        let j = state[p];
        while (j > 0 && ch !== stops[p][j]) j = fails[p][j - 1];
        if (ch === stops[p][j]) j++;
        state[p] = j;
        if (j === stops[p].length) fired = stops[p];
      }
      if (fired){ matched = fired; emitted = cur + 1 - fired.length; }
      const hold = matched ? 0 : Math.max(state[0], state[1]);
      const disp = ch === '\n' ? '\\n' : ch;
      F.push({shapes:draw(cur, hold), panels:panels(), view:VIEW,
        line:matched ? ln(CODE_S, 'return out, pat') : (state[0] ? ln(CODE_S, '                    j += 1') : ln(CODE_S, '                j = self.state[p]')),
        msg:matched
          ? {zh:"j 走到 4，<b>'&lt;/s&gt;' 完整出現了</b>。要送出去的只有 stop string <b>之前</b>的文字，stop 本身和後面全部丟掉 - 前面扣住的 3 個字元現在證明它們確實是 stop 的一部分，幸好沒有先送出去。",
             en:"j reached 4: <b>'&lt;/s&gt;' is complete</b>. Only the text <b>before</b> the stop string may be emitted; the stop itself and everything after it is dropped - the 3 characters held back turned out to be part of it, which is exactly why holding them was right."}
          : state[0] > 0
          ? {zh:"字元 <b>" + disp + "</b> 讓 '&lt;/s&gt;' 的比對長度變成 <b>" + state[0] + "</b>。這 " + state[0] + " 個字元<b>不能送出去</b>：它們可能是 stop string 的開頭，送出去就收不回來了。這就是 streaming 一定要有 hold-back 的原因。",
             en:"Character <b>" + disp + "</b> takes the match length for '&lt;/s&gt;' to <b>" + state[0] + "</b>. Those " + state[0] + " characters <b>cannot be emitted</b>: they might be the beginning of the stop string, and text already streamed cannot be taken back. This is why streaming needs a hold-back at all."}
          : {zh:'字元 <b>' + disp + '</b> 和任何 stop string 的開頭都不合，j 保持 0，可以安心送出。注意這裡<b>沒有回頭看過任何已送出的文字</b> - 狀態就是全部的記憶。',
             en:'Character <b>' + disp + '</b> does not start any stop string, j stays 0 and the character is safe to emit. Note that <b>no already-streamed text was re-read</b>: the state is the entire memory.'}});
      if (matched) break;
    }
    if (matched) break;
    const hold = Math.max(state[0], state[1]);
    emitted = bounds[c][1] - hold;
    F.push({shapes:draw(null, hold), panels:panels(), view:VIEW, line:ln(CODE_S, '        cut = len(self.buf) - hold'),
      msg:{zh:'chunk ' + (c + 1) + ' 處理完：送出 <b>' + emitted + '</b> 個字元，扣住 <b>' + hold + '</b> 個。扣住的長度就是<b>所有 stop string 目前最長的比對長度</b> - 少扣會漏掉跨 chunk 的 stop，多扣只是讓輸出多延遲一格。',
           en:'End of chunk ' + (c + 1) + ': <b>' + emitted + '</b> characters go out, <b>' + hold + '</b> are held. The amount held is <b>the longest partial match across all stop strings</b> - holding less would miss a stop split across chunks, holding more only adds latency.'}});
  }
  F.push({shapes:draw(null, 0).concat([note(4.9, 'emitted "The answer is 42." then STOP', COL.tealL, .38)]),
    panels:panels(), view:VIEW, line:ln(CODE_S, '                    return out, pat'),
    msg:{zh:'整趟下來，每個字元只被看過一次，記憶體只有兩個整數。sglang 在 <b>schedule_batch.py</b> 裡做的事情概念相同，只是它用 O(m²) 的暴力比對 - <b>因為它先把 tail 裁到 stop_str_max_len + 1 個字元</b>。把輸入的長度界住，是「換更好的演算法」以外一個同樣正當的解法。',
         en:'Every character was looked at once and the memory is two integers. sglang does the same thing conceptually in <b>schedule_batch.py</b>, except with an O(m²) brute-force comparison - <b>because it first clips the tail to stop_str_max_len + 1 characters</b>. Bounding the input is just as legitimate as reaching for a better algorithm, as long as you can prove the bound.'}});
  return F.list;
}

/* ============================= 4. tokens are not characters (the real bug) */
const CODE_T = [
'def stop_match_tail_len(stop_max_len, new_accepted_len, n_out):',
'    # sglang, schedule_batch.py',
'    return min(stop_max_len + 1 + max(new_accepted_len - 1, 0), n_out)',
'',
'# one token per step:      stop_max_len + 1 characters is enough',
'# speculative decoding:    several tokens land at once, and a stop',
'#                          string finishing *inside* that batch',
'#                          would fall out of a fixed-size window',
'',
'def first_detection(step, window_fn):',
'    t = 0',
'    while t < len(out_ids):',
'        t = min(t + step, len(out_ids))',
'        w = window_fn(step, t)',
'        if stop in tok.decode(out_ids[t - w:t]):',
'            return t',
'    return None'
];

function tokenFrames(v){
  const VOCAB = {1:'The', 2:' answer', 3:' is', 4:' 42]', 5:']', 6:' so', 7:' far', 8:' and', 9:' more'};
  const ids = [1, 2, 3, 4, 5, 6, 7, 8, 9], stop = ']]', step = v ? 4 : 1;
  const dec = a => a.map(i => VOCAB[i]).join('');
  const tailLen = (smax, acc, n) => Math.min(smax + 1 + Math.max(acc - 1, 0), n);
  const F = new Frames();
  const bw = 1.02, x0 = (VIEW[0] - ids.length * bw) / 2, y = 2.5;

  const draw = (t, wN, wS) => {
    const out = [];
    ids.forEach((id, k) => {
      let s = 'ghost';
      if (k < t) s = (k >= t - wS) ? 'hot' : 'done';
      if (k === 3 || k === 4) if (k < t) s = (k >= t - wS) ? 'hot' : 'act';
      out.push(S.r(x0 + k * bw, y, bw - .08, .78, s, VOCAB[id].replace(/ /g, '␣'), {fs:.30}));
      out.push(S.t(x0 + k * bw + (bw - .08) / 2, y - .34, 't' + (k + 1), {c:k < t ? COL.orangeL : COL.grey, fs:.26}));
    });
    if (t > 0){
      const xa = x0 + (t - wS) * bw + .04, xb = x0 + t * bw - .12;
      out.push(S.e(xa, y + 1.02, xb, y + 1.02, {s:'hot', arrow:false, w:.06}));
      out.push(S.t((xa + xb) / 2, y + 1.34, 'sglang window = ' + wS + ' tokens', {c:COL.orangeL, fs:.30}));
      const na = x0 + (t - wN) * bw + .04, nb = x0 + t * bw - .12;
      out.push(S.e(na, y - .78, nb, y - .78, {s:wN < wS ? 'bad' : 'ok', arrow:false, w:.06}));
      out.push(S.t((na + nb) / 2, y - 1.10, 'stop_max_len + 1 = ' + wN + ' tokens', {c:wN < wS ? COL.red : COL.tealL, fs:.30}));
    }
    out.push(S.t(mid2, y + 2.05, "stop string  ']]'  starts inside t4 and ends in t5", {c:COL.purpleL, fs:.32}));
    return out;
  };

  let hitN = null, hitS = null;
  F.push({shapes:draw(0, 0, 0), panels:[{lbl:{zh:'每步接受幾個 token', en:'tokens accepted per step'}, chips:[chip(String(step), 'hot')]}],
    view:VIEW, line:2,
    msg:{zh:'stop string 是<b>字元</b>，模型吐的是 <b>token</b>，而 token 的邊界跟人寫的字串一點關係都沒有：這裡 <b>&#39;]]&#39; 的第一個 ] 藏在 t4 裡面，第二個是 t5</b>。所以每一步都要把最後幾個 token 解碼回文字再找 - 問題是「最後幾個」要抓多少。',
         en:'A stop string is made of <b>characters</b>, the model emits <b>tokens</b>, and token boundaries have nothing to do with the string a user typed: here <b>the first ] of &#39;]]&#39; is buried inside t4 and the second one is t5</b>. So each step decodes the last few tokens back into text and searches - the question is how many "last few" must be.'}});

  let t = 0;
  while (t < ids.length){
    t = Math.min(t + step, ids.length);
    const wN = Math.min(stop.length + 1, t);
    const wS = tailLen(stop.length, step, t);
    const txtN = dec(ids.slice(t - wN, t)), txtS = dec(ids.slice(t - wS, t));
    const okN = txtN.indexOf(stop) >= 0, okS = txtS.indexOf(stop) >= 0;
    if (okN && hitN == null) hitN = t;
    if (okS && hitS == null) hitS = t;
    F.push({shapes:draw(t, wN, wS), view:VIEW, line:ln(CODE_T, 'if stop in tok.decode'),
      panels:[
        {lbl:{zh:'已產生 token 數 t', en:'tokens generated t'}, chips:[chip(String(t), 'hot')]},
        {lbl:{zh:'固定視窗解出的文字', en:'text seen by the fixed window'}, chips:[chip(JSON.stringify(txtN), okN ? 'ok' : 'bad')]},
        {lbl:{zh:'sglang 視窗解出的文字', en:'text seen by the sglang window'}, chips:[chip(JSON.stringify(txtS), okS ? 'ok' : 'act')]},
        {lbl:{zh:'偵測到 stop', en:'stop detected'}, chips:[chip(okS ? 'yes' : 'no', okS ? 'ok' : '')]}
      ],
      msg:okS && !okN
        ? {zh:'關鍵的一步。這一步一次接受了 <b>' + step + '</b> 個 token，stop string 在<b>這批的中間</b>就完成了。固定大小的視窗只看得到最後 ' + wN + ' 個 token，解出來是 ' + JSON.stringify(txtN) + ' - <b>整個跳過去了</b>。sglang 的視窗多加 <b>accepted − 1 = ' + (step - 1) + '</b>，才把 t4 拉回視野裡。',
             en:'This is the step that matters. It accepted <b>' + step + '</b> tokens at once and the stop string completed <b>in the middle of that batch</b>. A fixed window only sees the last ' + wN + ' tokens, decoding to ' + JSON.stringify(txtN) + ' - it <b>jumped clean over the stop</b>. sglang extends the window by <b>accepted − 1 = ' + (step - 1) + '</b>, which pulls t4 back into view.'}
        : okS
        ? {zh:'兩種視窗都看到 <b>' + JSON.stringify(stop) + '</b> 了。每步只接受一個 token 的時候，stop 一定結束在最後一個 token 上，所以 stop_max_len + 1 就夠 - <b>投機解碼一開，這個假設就破了</b>。',
             en:'Both windows can see <b>' + JSON.stringify(stop) + '</b>. When one token lands per step the stop string always finishes on the last token, so stop_max_len + 1 is enough - <b>turn speculative decoding on and that assumption breaks</b>.'}
        : {zh:'第 ' + t + ' 個 token 為止還沒看到 stop。視窗大小 = min(stop_max_len + 1 + (accepted − 1), t) = <b>' + wS + '</b>，用字元數去界 token 數是安全的，因為<b>一個 token 至少解出一個字元</b>。',
             en:'No stop yet at token ' + t + '. The window is min(stop_max_len + 1 + (accepted − 1), t) = <b>' + wS + '</b>. Bounding a token count with a character count is safe because <b>a token decodes to at least one character</b>.'}});
  }
  const fmt = x => x == null ? 'never' : 'token ' + x;
  F.push({shapes:draw(ids.length, Math.min(stop.length + 1, ids.length), tailLen(stop.length, step, ids.length)),
    view:VIEW, line:ln(CODE_T, '    return None'),
    panels:[
      {lbl:{zh:'固定視窗', en:'fixed window'}, chips:[chip(fmt(hitN), hitN == null ? 'bad' : 'ok')]},
      {lbl:{zh:'sglang 視窗', en:'sglang window'}, chips:[chip(fmt(hitS), 'ok')]}
    ],
    msg:step === 1
      ? {zh:'每步一個 token：兩種寫法都在 <b>' + fmt(hitS) + '</b> 抓到，看不出差別。這正是這個 bug 難發現的原因 - <b>它只在投機解碼打開時才會出現</b>。切到右邊的另一個模式看看。',
           en:'One token per step: both versions catch it at <b>' + fmt(hitS) + '</b> and look identical. That is exactly why this bug is hard to find - <b>it only shows up once speculative decoding is on</b>. Switch to the other mode.'}
      : {zh:'結論：固定視窗 <b>' + fmt(hitN) + '</b>，sglang 視窗 <b>' + fmt(hitS) + '</b>。抓到之後還有第二個問題 - stop 是在這批的第幾個 token 完成的？sglang 用 <b>_locate_str_stop_finished_len</b> 把 window 的前綴一段一段解碼回文字，直到 stop 出現，於是這裡回答「第 5 個 token 收工，保留 <b>The answer is 42</b>」。長度用 token 算，比對用字元算，兩邊只能靠解碼對齊。',
           en:'The verdict: fixed window <b>' + fmt(hitN) + '</b>, sglang window <b>' + fmt(hitS) + '</b>. Catching it raises a second question - which token inside the batch finished the stop? sglang answers with <b>_locate_str_stop_finished_len</b>, decoding growing prefixes of the window until the stop appears: finish after 5 tokens, keep <b>The answer is 42</b>. Lengths are counted in tokens, matches live in characters, and decoding is the only bridge between them.'}});
  return F.list;
}

/* ========================================= 5. Rabin-Karp: hash, then verify */
const CODE_R = [
'def rabin_karp(text, pat, base=256, mod=1_000_003):',
'    n, m = len(text), len(pat)',
'    high = pow(base, m - 1, mod)',
'    hp = ht = 0',
'    for i in range(m):                        # hash the pattern + first window',
'        hp = (hp * base + ord(pat[i])) % mod',
'        ht = (ht * base + ord(text[i])) % mod',
'    hits, verifications, false_pos = [], 0, 0',
'    for i in range(n - m + 1):',
'        if ht == hp:                          # a *maybe*, not a yes',
'            verifications += 1',
'            if text[i:i + m] == pat:',
'                hits.append(i)',
'            else:',
'                false_pos += 1                # a collision, caught here',
'        if i < n - m:                         # roll the window in O(1)',
'            ht = ((ht - ord(text[i]) * high) * base + ord(text[i + m])) % mod',
'    return hits, verifications, false_pos'
];

function rkFrames(v){
  const text = v ? 'xaahy' : 'ACGTACGTGACGTACGTTACGTACGTG';
  const pat  = v ? 'aca'   : 'ACGTACGTG';
  const mod  = v ? 101 : 1000003, base = 256;
  const n = text.length, m = pat.length, F = new Frames();
  const w = v ? .80 : .335, x0 = (VIEW[0] - n * w) / 2, y = 2.3;
  const md = x => ((x % mod) + mod) % mod;
  let high = 1; for (let i = 0; i < m - 1; i++) high = md(high * base);
  let hp = 0, ht = 0;
  for (let i = 0; i < m; i++){ hp = md(hp * base + pat.charCodeAt(i)); ht = md(ht * base + text.charCodeAt(i)); }
  let hits = [], ver = 0, fp = 0;

  const draw = (i, st) => {
    const sts = {};
    for (let q = 0; q < n; q++) sts[q] = (q >= i && q < i + m) ? st : (q < i ? 'done' : 'idle');
    const out = cellRow(text.split(''), x0, y, w, .66, {states:sts, index:false, fs:v ? .40 : .24});
    out.push(S.t(mid2, y + 1.35, 'pattern  ' + pat + '   hash = ' + hp, {c:COL.tealL, fs:.34}));
    out.push(S.t(x0 + (i + m / 2) * w, y - .48, 'window hash = ' + ht, {c:st === 'bad' ? COL.red : COL.orangeL, fs:.32}));
    return out;
  };
  const panels = i => [
    {lbl:{zh:'視窗位置 i', en:'window i'}, chips:[chip(String(i), 'hot')]},
    {lbl:{zh:'pattern 指紋', en:'pattern fingerprint'}, chips:[chip(String(hp), 'act')]},
    {lbl:{zh:'視窗指紋', en:'window fingerprint'}, chips:[chip(String(ht), ht === hp ? 'ok' : '')]},
    {lbl:{zh:'真的比字串幾次', en:'real comparisons'}, chips:[chip(String(ver), ver ? 'act' : '')]},
    {lbl:{zh:'誤報（被 verify 擋下）', en:'false positives caught'}, chips:[chip(String(fp), fp ? 'bad' : '')]},
    {lbl:{zh:'命中', en:'hits'}, chips:hits.length ? hits.map(h => chip(String(h), 'ok')) : [chip('—')]}
  ];

  F.push({shapes:draw(0, 'idle'), panels:panels(0), view:VIEW, line:ln(CODE_R, '        hp = (hp * base'),
    msg:{zh:'Rabin-Karp 換一個角度：<b>與其比字串，不如比指紋</b>。把長度 ' + m + ' 的視窗當成一個 base-' + base + ' 的數字取 mod ' + mod + '，pattern 的指紋是 <b>' + hp + '</b>。指紋不合就<b>一定</b>不是；指紋相合只是<b>可能</b>是。',
         en:'Rabin-Karp changes the question: <b>compare fingerprints instead of strings</b>. Read the ' + m + '-character window as a base-' + base + ' number mod ' + mod + '; the pattern fingerprint is <b>' + hp + '</b>. A different fingerprint means <b>definitely not</b>; an equal one only means <b>maybe</b>.'}});

  for (let i = 0; i + m <= n; i++){
    const eq = ht === hp;
    if (eq){
      ver++;
      const real = text.slice(i, i + m) === pat;
      if (real) hits.push(i); else fp++;
      F.push({shapes:draw(i, real ? 'ok' : 'bad'), panels:panels(i), view:VIEW,
        line:ln(CODE_R, real ? '                hits.append(i)' : '                false_pos += 1'),
        msg:real
          ? {zh:'指紋相同，<b>而且</b>逐字比對也真的相同 - 位置 <b>' + i + '</b> 命中。注意這一步的 O(m) 比對是<b>必要的</b>，不是保險。',
             en:'Fingerprints match <b>and</b> the character-by-character check agrees: a real hit at <b>' + i + '</b>. That O(m) comparison is <b>part of the algorithm</b>, not a belt-and-braces extra.'}
          : {zh:'指紋一樣，但字串是 <b>' + JSON.stringify(text.slice(i, i + m)) + '</b>，不是 <b>' + JSON.stringify(pat) + '</b> - <b>碰撞</b>。verify 那一行把它擋下來了。把 verify 拿掉，程式不會壞掉、不會噴錯，只會<b>安靜地回答錯的答案</b>，這是最難查的一種 bug。',
             en:'Same fingerprint, but the window is <b>' + JSON.stringify(text.slice(i, i + m)) + '</b>, not <b>' + JSON.stringify(pat) + '</b> - a <b>collision</b>, stopped by the verify line. Delete that line and nothing crashes; the function just <b>silently returns a wrong answer</b>, which is the worst kind of bug to chase.'}});
    } else {
      F.push({shapes:draw(i, 'hot'), panels:panels(i), view:VIEW, line:ln(CODE_R, '        if ht == hp:'),
        msg:{zh:'指紋 <b>' + ht + '</b> ≠ <b>' + hp + '</b>，這個視窗一個字元都不用比就可以跳過。整段掃描裡，<b>大部分位置的成本就只有一次乘加</b>。',
             en:'Fingerprint <b>' + ht + '</b> ≠ <b>' + hp + '</b>, so this window is dismissed without comparing a single character. For most positions in the scan, <b>the entire cost is one multiply-add</b>.'}});
    }
    if (i < n - m){
      ht = md((ht - text.charCodeAt(i) * high) * base + text.charCodeAt(i + m));
      F.push({shapes:draw(i + 1, 'act'), panels:panels(i + 1), view:VIEW, line:ln(CODE_R, '            ht = ((ht'),
        msg:{zh:'往右滾一格：<b>減掉離開的 ' + JSON.stringify(text[i]) + '，乘 base，加上進來的 ' + JSON.stringify(text[i + m]) + '</b>。重點在這裡 - 更新是 <b>O(1)</b>，不是 O(m)，所以整趟只要 O(n)。這也是 rolling hash 這個名字的由來。',
             en:'Roll one place right: <b>subtract the departing ' + JSON.stringify(text[i]) + ', multiply by base, add the arriving ' + JSON.stringify(text[i + m]) + '</b>. That is the whole trick - the update is <b>O(1)</b>, not O(m), so the scan is O(n). Hence "rolling" hash.'}});
    }
  }
  F.push({shapes:draw(n - m, 'done'), panels:panels(n - m), view:VIEW, line:ln(CODE_R, '    return hits'),
    msg:v
      ? {zh:'mod 只有 101 的時候，碰撞是家常便飯：<b>' + ver + '</b> 次指紋相同，其中 <b>' + fp + '</b> 次是假的，真正的命中 <b>' + (hits.length ? hits.join(', ') : '沒有') + '</b>。實務上 mod 取大質數讓碰撞機率低到可以忽略，但<b>「低到可以忽略」不等於零</b> - verify 那一行永遠不能省。',
           en:'With a modulus of only 101 collisions are routine: <b>' + ver + '</b> fingerprint matches, <b>' + fp + '</b> of them bogus, real hits <b>' + (hits.length ? hits.join(', ') : 'none') + '</b>. A large prime modulus makes collisions negligible in practice, but <b>negligible is not zero</b> - the verify line can never go.'}
      : {zh:'命中 <b>[' + hits.join(', ') + ']</b>，只做了 <b>' + ver + '</b> 次真正的字串比對、<b>' + fp + '</b> 次誤報。KMP 在這裡不會比較慢，Rabin-Karp 真正贏的場合是<b>一次找很多個同長度的 pattern</b>：把它們的指紋放進一個 set，一趟掃描就全部搞定。',
           en:'Hits at <b>[' + hits.join(', ') + ']</b> after only <b>' + ver + '</b> real string comparisons and <b>' + fp + '</b> false positives. KMP would not be slower here; where Rabin-Karp genuinely wins is <b>many patterns of the same length at once</b> - put their fingerprints in a set and one pass finds them all.'}});
  return F.list;
}

/* ============================ 6. the same hash idea, one page id at a time */
const CODE_P = [
'def page_hashes(token_ids, page_size):',
'    """sglang, mem_cache/utils.py - get_hash_str."""',
'    out, parent = [], None',
'    for i in range(0, len(token_ids) - page_size + 1, page_size):',
'        page = token_ids[i:i + page_size]',
'        h = hashlib.sha256()',
'        h.update(bytes(str(parent) + str(page), "utf-8"))   # fold in the parent',
'        parent = h.hexdigest()',
'        out.append(parent)',
'    return out',
'',
'# there is no verify step here, and there cannot be:',
'# the tokens of the cached page are not around to compare against.'
];

function pageFrames(){
  /* the digests are the ones string_matching.py prints */
  const A = ['e69d648c', 'ab130d42', '797558e5'];
  const B = ['e69d648c', 'ab130d42', '0c64f0b6'];
  const C = ['8099e991', '66d8068b'];
  const rows = [
    {name:'req A', pages:[['1000..1003'], ['1004..1007'], ['1, 2, 3, 4']], h:A},
    {name:'req B', pages:[['1000..1003'], ['1004..1007'], ['1, 2, 9, 9']], h:B},
    {name:'req C', pages:[['7, 7, 7, 7'], ['1, 2, 3, 4']], h:C}
  ];
  const F = new Frames(), bw = 2.05, x0 = 1.9, y0 = 4.3, dy = 1.55;
  const draw = (upto, marks) => {
    const out = [];
    rows.forEach((r, ri) => {
      const y = y0 - ri * dy;
      out.push(S.t(x0 - .35, y + .32, r.name, {c:COL.tealL, fs:.34, anchor:'end'}));
      r.pages.forEach((p, k) => {
        const idx = ri * 10 + k;
        const st = marks && marks[idx] ? marks[idx] : (upto == null || idx <= upto ? 'idle' : 'ghost');
        out.push(S.r(x0 + k * bw, y, bw - .18, .64, st, p[0], {fs:.28}));
        out.push(S.t(x0 + k * bw + (bw - .18) / 2, y - .28, r.h[k], {c:st === 'hot' ? COL.orangeL : (st === 'ok' ? COL.tealL : COL.grey), fs:.26}));
        if (k > 0) out.push(S.e(x0 + k * bw - .16, y + .32, x0 + k * bw - .02, y + .32, {s:'soft'}));
      });
    });
    return out;
  };
  F.push({shapes:draw(null, {}), view:VIEW, line:ln(CODE_P, 'h.update'),
    panels:[{lbl:{zh:'page 大小', en:'page size'}, chips:[chip('4 tokens', 'act')]}],
    msg:{zh:'同一個「先算指紋」的想法，在推論伺服器裡還有第二個用法：<b>KV cache 的 block id</b>。每一頁 token 算一個 SHA-256，而且<b>把上一頁的 digest 一起餵進去</b>，所以 id 代表的不是「這四個 token」，而是「<b>從開頭到這裡的一整串 token</b>」。',
         en:'The same fingerprint-first idea has a second use inside an inference server: <b>KV cache block ids</b>. Each page of tokens gets a SHA-256 that <b>folds in the previous page digest</b>, so an id does not stand for "these four tokens" but for "<b>every token from the start up to here</b>".'}});
  F.push({shapes:draw(null, {0:'ok', 1:'ok', 10:'ok', 11:'ok'}), view:VIEW, line:ln(CODE_P, '        parent = h.hexdigest()'),
    panels:[{lbl:{zh:'A 與 B 共用', en:'shared by A and B'}, chips:[chip('2 pages', 'ok')]}],
    msg:{zh:'A 和 B 的前兩頁 token 一樣，鏈式 hash 就給出<b>一模一樣的兩個 id</b>：<b>' + A[0] + '</b>、<b>' + A[1] + '</b>。這兩頁的 KV 直接共用，不用重算 - 上千個請求共用同一段 system prompt 的省法就是這個。',
         en:'A and B start with the same two pages, so the chained hash produces <b>the very same two ids</b>: <b>' + A[0] + '</b> and <b>' + A[1] + '</b>. Their KV pages are shared instead of recomputed, which is how a thousand requests carrying the same system prompt stop paying for it.'}});
  F.push({shapes:draw(null, {2:'hot', 12:'hot'}), view:VIEW, line:ln(CODE_P, '        parent = h.hexdigest()'),
    panels:[{lbl:{zh:'第三頁', en:'third page'}, chips:[chip(A[2], 'hot'), chip(B[2], 'hot')]}],
    msg:{zh:'第三頁一分岔，id 立刻不同（<b>' + A[2] + '</b> vs <b>' + B[2] + '</b>）。共用只到分岔為止，這正好是 prefix cache 想要的行為。',
         en:'The third page diverges and the ids part company immediately (<b>' + A[2] + '</b> vs <b>' + B[2] + '</b>). Sharing stops exactly at the fork, which is what a prefix cache wants.'}});
  F.push({shapes:draw(null, {2:'act', 21:'bad'}), view:VIEW, line:ln(CODE_P, '# there is no verify step here'),
    panels:[{lbl:{zh:'同樣的 4 個 token', en:'identical 4 tokens'}, chips:[chip(A[2], 'act'), chip(C[1], 'bad')]}],
    msg:{zh:'C 的最後一頁裝的 token 和 A 的最後一頁<b>一模一樣</b>，id 卻不同 - 因為前面的路徑不同。這是刻意的：attention 的 K/V 取決於<b>整段前綴</b>，內容一樣但位置不同的頁<b>不能</b>共用。',
         en:'C\'s last page holds <b>exactly the same four tokens</b> as A\'s last page and still gets a different id, because the path leading to it differs. That is deliberate: attention K/V depend on <b>the whole prefix</b>, so pages with equal content but different history <b>must not</b> be shared.'}});
  F.push({shapes:draw(null, {}).concat([note(.75, 'no verify step - and there cannot be one', COL.orangeL, .36)]),
    view:VIEW, line:ln(CODE_P, '# the tokens of the cached page'),
    panels:[{lbl:{zh:'和 Rabin-Karp 的差別', en:'unlike Rabin-Karp'}, chips:[chip('no verify', 'bad')]},
            {lbl:{zh:'靠什麼保證', en:'what carries the risk'}, chips:[chip('SHA-256', 'ok')]}],
    msg:{zh:'和 Rabin-Karp 最大的不同在這裡：<b>這裡沒有 verify 這一步，也做不到</b> - 要比對的那些 token 早就不在手上了，命中的當下只有 id。所以整個正確性壓在 <b>SHA-256 不會碰撞</b>這件事上；如果換成一個便宜的 hash，碰撞的後果不是慢，而是<b>一個請求讀到別人的 KV</b>。選 hash 的時候，「能不能事後驗證」比「快不快」重要得多。',
         en:'This is where it differs from Rabin-Karp most sharply: <b>there is no verify step, and there cannot be one</b> - the tokens you would compare against are long gone, all you hold at hit time is the id. Correctness therefore rests entirely on <b>SHA-256 not colliding</b>; swap in a cheap hash and a collision does not make things slow, it makes <b>one request read another request\'s KV</b>. When picking a hash, "can I verify afterwards?" matters far more than "is it fast?".'}});
  return F.list;
}

/* =============================================================== 7. LeetCode */
const CODE_L = [
'def repeated_substring_pattern(s):          # LC 459',
'    n = len(s)',
'    b = build_failure(s)[-1]                # longest border',
'    return b > 0 and n % (n - b) == 0       # n - b is the period',
'',
'def shortest_palindrome(s):                 # LC 214',
'    if not s:',
'        return s',
'    probe = s + "\\x00" + s[::-1]',
'    k = build_failure(probe)[-1]            # longest palindromic prefix',
'    return s[k:][::-1] + s'
];

function failScan(F, str, disp, y, x0, w, panelsExtra, why){
  const m = str.length, fail = new Array(m).fill(null);
  let k = 0;
  const draw = (i, sts) => {
    const out = cellRow(disp.split(''), x0, y, w, .68, {states:sts || {}, index:false, fs:.34});
    const fs = {};
    for (let q = 0; q < m; q++) if (fail[q] != null) fs[q] = 'done';
    if (i != null && fail[i] != null) fs[i] = 'ok';
    out.push.apply(out, cellRow(fail.map(v => v == null ? '' : v), x0, y - 1.0, w, .60, {states:fs, index:false, fs:.30}));
    out.push(S.t(x0 - .22, y + .38, 's', {c:COL.tealL, fs:.30, anchor:'end'}));
    out.push(S.t(x0 - .22, y - .66, 'fail', {c:COL.tealL, fs:.30, anchor:'end'}));
    return out;
  };
  fail[0] = 0;
  for (let i = 1; i < m; i++){
    while (k > 0 && str[i] !== str[k]) k = fail[k - 1];
    if (str[i] === str[k]) k++;
    fail[i] = k;
    F.push({shapes:draw(i, {[i]:'hot'}), view:VIEW, line:ln(CODE_L, 'build_failure'),
      panels:[{lbl:{zh:'i', en:'i'}, chips:[chip(String(i), 'hot')]},
              {lbl:{zh:'目前 border 長度', en:'current border length'}, chips:[chip(String(k), 'act')]}].concat(panelsExtra || []),
      msg:{zh:'fail[' + i + '] = <b>' + k + '</b>' + (k ? '：<b>' + disp.slice(0, k) + '</b> 同時是 <b>' + disp.slice(0, i + 1) + '</b> 的開頭與結尾。' : '：這一段沒有 border。') + (why || ''),
           en:'fail[' + i + '] = <b>' + k + '</b>' + (k ? ': <b>' + disp.slice(0, k) + '</b> is both the head and the tail of <b>' + disp.slice(0, i + 1) + '</b>.' : ': this prefix has no border at all.') + (why || '')}});
  }
  return {fail:fail, draw:draw};
}

function lcFrames(v){
  const F = new Frames();
  if (!v){
    const s = 'abcabcabcabc', n = s.length;
    const w = .70, x0 = (VIEW[0] - n * w) / 2;
    F.push({shapes:[], view:VIEW, line:0,
      msg:{zh:'<b>LeetCode 459 Repeated Substring Pattern</b>：s 能不能由它的某個子字串重複好幾次拼出來？直覺解法是試每個能整除 n 的長度。但這題其實是 failure function 的<b>一個副產品</b>。',
           en:'<b>LeetCode 459, Repeated Substring Pattern</b>: can s be built by repeating one of its substrings? The obvious solution tries every divisor length. But the answer is a <b>by-product of the failure function</b>.'},
      panels:[{lbl:{zh:'s', en:'s'}, chips:[chip(s, 'act')]}]});
    const r = failScan(F, s, s, 3.4, x0, w, [], '');
    const b = r.fail[n - 1], per = n - b;
    F.push({shapes:r.draw(n - 1, {}).concat([
        note(1.15, 'n - fail[n-1] = ' + n + ' - ' + b + ' = ' + per + '   and   ' + n + ' % ' + per + ' = ' + (n % per), COL.orangeL, .36)]),
      view:VIEW, line:ln(CODE_L, 'return b > 0'),
      panels:[{lbl:{zh:'最長 border', en:'longest border'}, chips:[chip(String(b), 'ok')]},
              {lbl:{zh:'週期 n − border', en:'period n − border'}, chips:[chip(String(per), 'act')]},
              {lbl:{zh:'答案', en:'answer'}, chips:[chip(String(n % per === 0 && b > 0), 'ok')]}],
      msg:{zh:'關鍵那一步：<b>如果 s 的開頭 b 個字元同時也是結尾</b>，那把 s 整個往右挪 n − b 格會和自己疊合，也就是 <b>n − b 是一個週期</b>。週期能整除 n（' + n + ' % ' + per + ' = 0），s 就真的是 <b>' + s.slice(0, per) + '</b> 重複 ' + (n / per) + ' 次。<b>b &gt; 0</b> 這個條件不能少，不然 "abc" 會被算成週期 3 重複一次。',
           en:'The step that matters: <b>if the first b characters of s are also its last b</b>, then sliding s right by n − b lands it on itself, i.e. <b>n − b is a period</b>. When that period divides n (' + n + ' % ' + per + ' = 0), s really is <b>' + s.slice(0, per) + '</b> repeated ' + (n / per) + ' times. The <b>b &gt; 0</b> guard is not optional - without it "abc" would count as its own repetition.'}});
  } else {
    const s = 'aacecaaa', rev = s.split('').reverse().join('');
    const probe = s + ' ' + rev, disp = s + '#' + rev, n = probe.length;
    const w = .52, x0 = (VIEW[0] - n * w) / 2;
    F.push({shapes:[], view:VIEW, line:ln(CODE_L, 'def shortest_palindrome'),
      panels:[{lbl:{zh:'s', en:'s'}, chips:[chip(s, 'act')]}],
      msg:{zh:'<b>LeetCode 214 Shortest Palindrome</b>：只能在<b>前面</b>加字元，把 s 變成回文，而且要最短。等價的問題是「s 最長的<b>回文前綴</b>有多長」 - 剩下的那一段反過來貼到前面就好。',
           en:'<b>LeetCode 214, Shortest Palindrome</b>: characters may only be added <b>in front</b>, and the result must be the shortest palindrome. Equivalently: how long is the longest <b>palindromic prefix</b> of s? Whatever is left over gets reversed onto the front.'}});
    F.push({shapes:[S.t(mid2, 3.9, 's        = ' + s, {c:COL.tealL, fs:.40}),
                    S.t(mid2, 3.3, 'reversed = ' + rev, {c:COL.purpleL, fs:.40}),
                    S.t(mid2, 2.5, 's + "#" + reversed = ' + disp, {c:COL.orangeL, fs:.36})],
      view:VIEW, line:ln(CODE_L, 'probe = s'),
      panels:[{lbl:{zh:'探針字串', en:'probe string'}, chips:[chip(disp, 'hot')]}],
      msg:{zh:'技巧在這裡：<b>回文前綴</b>就是「s 的前綴同時也是 reversed(s) 的後綴」。把兩段用一個<b>兩邊都不會出現的分隔字元</b>接起來，failure function 的最後一格就是這個長度 - border 不可能跨過分隔符，所以不會超過任何一段的長度。',
           en:'Here is the trick: a <b>palindromic prefix</b> is a prefix of s that is also a suffix of reversed(s). Glue the two together with a <b>separator that appears in neither</b>, and the last entry of the failure function is exactly that length - a border cannot straddle the separator, so it can never over-count.'}});
    const r = failScan(F, probe, disp, 3.6, x0, w, [], '');
    const k = r.fail[n - 1], ans = s.slice(k).split('').reverse().join('') + s;
    F.push({shapes:r.draw(n - 1, {}).concat([
        note(1.25, 'longest palindromic prefix = ' + s.slice(0, k) + '   ->   ' + ans, COL.orangeL, .36)]),
      view:VIEW, line:ln(CODE_L, 'return s[k:]'),
      panels:[{lbl:{zh:'回文前綴長度', en:'palindromic prefix length'}, chips:[chip(String(k), 'ok')]},
              {lbl:{zh:'要補在前面', en:'prepended'}, chips:[chip(JSON.stringify(s.slice(k).split('').reverse().join('')), 'act')]},
              {lbl:{zh:'答案', en:'answer'}, chips:[chip(ans, 'ok')]}],
      msg:{zh:'最後一格是 <b>' + k + '</b>，所以 <b>' + s.slice(0, k) + '</b> 是最長的回文前綴，把剩下的 <b>' + s.slice(k) + '</b> 反過來貼到最前面，得到 <b>' + ans + '</b>。整題 <b>O(n)</b>，而且從頭到尾只呼叫了一次 build_failure - 這就是為什麼值得把 border 當成一個獨立的工具記起來，而不是只當成 KMP 的內部細節。',
           en:'The final entry is <b>' + k + '</b>, so <b>' + s.slice(0, k) + '</b> is the longest palindromic prefix; reverse the leftover <b>' + s.slice(k) + '</b> onto the front and the answer is <b>' + ans + '</b>. The whole problem is <b>O(n)</b> with a single call to build_failure - which is why borders are worth remembering as a tool in their own right, not just as an implementation detail of KMP.'}});
  }
  return F.list;
}

/* ===================================================================== meta */
const DAY_META = {
  title:{zh:'Day 25 - 字串匹配：KMP 與 Rabin-Karp', en:'Day 25 - String matching: KMP and Rabin-Karp'},
  sub:{zh:'KMP 靠 failure function 記住已經比過的那一段，文字的指標永遠不回頭；Rabin-Karp 先比指紋、再比字串。推論伺服器的 stop string 和 KV cache block id，用的就是這兩件事。',
       en:'KMP remembers what it has already compared so the text index never rewinds; Rabin-Karp compares fingerprints first and strings only when it must. Stop strings and KV cache block ids in an inference server are these two ideas.'},
  tabs:[
    {id:'naive', label:{zh:'① 天真比對', en:'1. Naive'}, view:VIEW,
     stage:{zh:'每個對齊位置從頭比一次', en:'restart at every alignment'},
     variants:[{zh:'aaab in aaaaaaab', en:'aaab in aaaaaaab'}, {zh:'abcabd in abcabcabd', en:'abcabd in abcabcabd'}],
     idea:{zh:'天真版每次失敗都<b>把已經比過的字元丟掉</b>，回到下一個對齊位置從 j = 0 重來。但失敗的那一刻其實握有情報：文字的那一段<b>就等於 pattern 的前 j 個字元</b>。丟掉這個情報，就是 O(n·m) 的來源。',
           en:'On every mismatch the naive matcher <b>discards the characters it just compared</b> and restarts the next alignment at j = 0. Yet at that moment it knows something: that stretch of text <b>is</b> the first j characters of the pattern. Throwing that away is where O(n·m) comes from.'},
     legend:['hot', 'ok', 'bad', 'done'], code:CODE_N, build:naiveFrames},
    {id:'kmp', label:{zh:'② failure function', en:'2. Failure function'}, view:VIEW,
     stage:{zh:'border 表，以及用它搜尋', en:'the table of borders, and searching with it'},
     variants:[{zh:'建表 build_failure', en:'build the table'}, {zh:'用表搜尋 kmp_search', en:'search with it'}],
     idea:{zh:'fail[i] = pat[0..i] 最長的「開頭也是結尾」長度（border）。失敗時 j 退到 fail[j−1] 而不是 0，<b>i 一步都不退</b>；因為 k 每步最多加一，總下拉次數有界，所以是 O(n + m)。副產品：n − fail[n−1] 是最短週期。',
           en:'fail[i] is the longest border of pat[0..i] - the longest prefix that is also a suffix. On a mismatch j falls back to fail[j−1] instead of 0 and <b>i never moves backwards</b>; since k rises by at most one per step the total fall-backs are bounded, giving O(n + m). Free by-product: n − fail[n−1] is the shortest period.'},
     legend:['hot', 'act', 'ok', 'bad', 'done'], code:CODE_K,
     build:i => (i ? kmpSearchFrames() : kmpBuildFrames())},
    {id:'stream', label:{zh:'③ 串流中的 stop string', en:'3. Streaming stop string'}, view:VIEW,
     stage:{zh:'一個字元一個字元餵進來，不能回頭看', en:'fed one character at a time, no rewinding'},
     idea:{zh:'一個 KMP 狀態就能在<b>不保留任何已送出文字</b>的情況下比對 stop string。代價是必須<b>扣住</b>最長部分比對長度的那幾個字元 - 送出去就收不回來了。sglang 的 <code>check_match_stop_str_prefix</code> 做同一件事，用的是 O(m²) 暴力法，但先把 tail 裁到 stop_str_max_len + 1。',
           en:'A single KMP state matches stop strings while <b>keeping none of the text already streamed</b>. The price is <b>holding back</b> as many characters as the longest partial match - once emitted, text cannot be recalled. sglang\'s <code>check_match_stop_str_prefix</code> does the same job brute-force in O(m²), after clipping the tail to stop_str_max_len + 1.'},
     legend:['hot', 'act', 'ok', 'done', 'ghost'], code:CODE_S, build:stopFrames},
    {id:'spec', label:{zh:'④ token ≠ 字元（真實 bug）', en:'4. Tokens are not characters'}, view:VIEW,
     stage:{zh:'投機解碼一次接受多個 token', en:'speculative decoding accepts several tokens at once'},
     variants:[{zh:'每步 1 個 token', en:'1 token per step'}, {zh:'每步 4 個 token（投機解碼）', en:'4 tokens per step (speculation)'}],
     idea:{zh:'stop string 活在<b>字元</b>空間，輸出長度算在 <b>token</b> 空間。每步一個 token 時 stop_max_len + 1 個字元的視窗就夠；一旦投機解碼一次接受好幾個 token，stop 可能在<b>這一批的中間</b>完成而被整個跳過。sglang 的修法是把視窗多開 accepted − 1。',
           en:'A stop string lives in <b>character</b> space while output length is counted in <b>tokens</b>. With one token per step a window of stop_max_len + 1 characters suffices; once speculation accepts several tokens at once the stop can complete <b>inside the batch</b> and be jumped over entirely. sglang widens the window by accepted − 1.'},
     legend:['hot', 'act', 'ok', 'bad', 'ghost'], code:CODE_T, build:tokenFrames},
    {id:'rk', label:{zh:'⑤ Rabin-Karp', en:'5. Rabin-Karp'}, view:VIEW,
     stage:{zh:'先比指紋，相同才比字串', en:'compare fingerprints, then verify'},
     variants:[{zh:'DNA 搜尋（大質數 mod）', en:'DNA search (large prime mod)'}, {zh:'碰撞：mod = 101', en:'a collision: mod = 101'}],
     idea:{zh:'rolling hash 讓視窗右移只花 <b>O(1)</b>：減掉離開的字元、乘 base、加上進來的字元。指紋不同就<b>一定</b>不是，指紋相同只是<b>可能</b>是 - 所以 verify 那一行是演算法的一部分，拿掉它不會當掉，只會<b>安靜地答錯</b>。真正的主場是同長度的多 pattern 一次掃完。',
           en:'A rolling hash advances the window in <b>O(1)</b>: subtract the departing character, multiply by the base, add the arriving one. A different fingerprint means <b>definitely not</b>; an equal one only means <b>maybe</b> - so the verify line is part of the algorithm. Remove it and nothing crashes, the function just <b>answers wrongly in silence</b>. Its real home is many equal-length patterns in one pass.'},
     legend:['hot', 'act', 'ok', 'bad', 'done'], code:CODE_R, build:rkFrames},
    {id:'page', label:{zh:'⑥ KV cache block id', en:'6. KV cache block ids'}, view:VIEW,
     stage:{zh:'鏈式 hash，而且沒有 verify', en:'a chained hash, with no verify step'},
     idea:{zh:'sglang 的 page id 是把上一頁 digest 一起餵進去的 SHA-256，所以 id 代表<b>整段前綴</b>而不是這幾個 token - 內容相同但位置不同的頁<b>不能</b>共用，因為 attention 的 K/V 取決於前綴。和 Rabin-Karp 相反的是：<b>這裡沒有 verify，也做不到</b>，正確性完全押在 hash 不碰撞上。',
           en:'An sglang page id is a SHA-256 that folds in the previous page digest, so the id stands for <b>the whole prefix</b> rather than these few tokens - pages with identical content but different history <b>must not</b> be shared, because attention K/V depend on the prefix. Unlike Rabin-Karp, <b>there is no verify step here and there cannot be one</b>: correctness rests entirely on the hash not colliding.'},
     legend:['hot', 'act', 'ok', 'bad', 'ghost'], code:CODE_P, build:pageFrames},
    {id:'lc', label:{zh:'⑦ LeetCode', en:'7. LeetCode'}, view:VIEW,
     stage:{zh:'459 重複子字串 / 214 最短回文', en:'459 repeated substring / 214 shortest palindrome'},
     variants:[{zh:'459 週期', en:'459 period'}, {zh:'214 回文前綴', en:'214 palindromic prefix'}],
     idea:{zh:'兩題都不是「搜尋」，而是<b>直接用 border 這個量</b>。459：n − fail[n−1] 是最短週期，能整除 n 就成立（記得 border &gt; 0 的守門）。214：s + 分隔符 + reverse(s) 的最後一格，就是最長回文前綴的長度。',
           en:'Neither problem is a search - both <b>use the border quantity directly</b>. 459: n − fail[n−1] is the shortest period, and it works iff that period divides n (mind the border &gt; 0 guard). 214: the last entry of the failure function of s + separator + reverse(s) is the length of the longest palindromic prefix.'},
     legend:['hot', 'act', 'ok', 'done'], code:CODE_L, build:lcFrames}
  ]
};
