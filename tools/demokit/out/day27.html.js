// DAY: 27
// TITLE_ZH: 編輯距離與相似度：Levenshtein、Jaccard、MinHash
// TITLE_EN: Edit distance and similarity - Levenshtein, Jaccard, MinHash
// SUB_ZH: 「這兩個字串有多像」有兩種答案：逐字元的 Levenshtein 距離，精確但要 O(nm)；集合層級的 Jaccard，配上 MinHash 與 LSH，可以在幾千萬份文件裡找出近重複。
// SUB_EN: "How similar are these two strings" has two answers: Levenshtein distance, exact but O(nm); and set-level Jaccard, which with MinHash and LSH scales to tens of millions of documents.
// FOLDER: day%2027%20-%20edit%20distance%20and%20similarity
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const mid2 = VIEW[0] / 2;
const ln = (code, frag) => { const i = code.findIndex(l => l.indexOf(frag) >= 0); return i < 0 ? 0 : i; };
const note = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});
function chip(t, cls){ return {t:t, cls:cls || ''}; }
const f2 = x => x.toFixed(2);
const f3 = x => x.toFixed(3);

/* ---- verbatim ports of edit_similarity.py -------------------------------- */
function editTable(a, b){
  const n = a.length, m = b.length;
  const d = [];
  for (let i = 0; i <= n; i++){ d.push(new Array(m + 1).fill(0)); d[i][0] = i; }
  for (let j = 0; j <= m; j++) d[0][j] = j;
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++){
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
    }
  return d;
}
function editDistance(a, b){ return editTable(a, b)[a.length][b.length]; }
function editOps(a, b){
  const d = editTable(a, b), ops = [];
  let i = a.length, j = b.length;
  while (i > 0 || j > 0){
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && d[i][j] === d[i - 1][j - 1]){
      ops.push(['match', i - 1, a[i - 1], b[j - 1]]); i--; j--;
    } else if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + 1){
      ops.push(['substitute', i - 1, a[i - 1], b[j - 1]]); i--; j--;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1){
      ops.push(['delete', i - 1, a[i - 1], null]); i--;
    } else {
      ops.push(['insert', i, null, b[j - 1]]); j--;
    }
  }
  ops.reverse();
  return ops;
}
function pathCells(a, b){                       /* the cells the backtrace visits */
  const d = editTable(a, b), out = [];
  let i = a.length, j = b.length;
  out.push([i, j]);
  while (i > 0 || j > 0){
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && d[i][j] === d[i - 1][j - 1]){ i--; j--; }
    else if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + 1){ i--; j--; }
    else if (i > 0 && d[i][j] === d[i - 1][j] + 1){ i--; }
    else { j--; }
    out.push([i, j]);
  }
  out.reverse();
  return out;
}
function editRows(a, b){                        /* rolling row - records each row */
  const n = a.length, m = b.length, rows = [];
  let prev = [];
  for (let j = 0; j <= m; j++) prev.push(j);
  rows.push(prev.slice());
  for (let i = 1; i <= n; i++){
    const cur = new Array(m + 1).fill(0);
    cur[0] = i;
    for (let j = 1; j <= m; j++){
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = cur; rows.push(cur.slice());
  }
  return rows;
}
const INF = 1e9;
function editBounded(a, b, k){                  /* returns {d, cells, gate} */
  const n = a.length, m = b.length;
  if (Math.abs(n - m) > k) return {d:k + 1, cells:0, gate:true, band:[]};
  let prev = new Array(m + 1).fill(INF), cells = 0;
  const band = [];
  for (let j = 0; j <= Math.min(m, k); j++) prev[j] = j;
  band.push([Math.max(0, -k), Math.min(m, k)]);
  for (let i = 1; i <= n; i++){
    const cur = new Array(m + 1).fill(INF);
    const lo = Math.max(0, i - k), hi = Math.min(m, i + k);
    band.push([lo, hi]);
    if (lo === 0) cur[0] = i;
    for (let j = Math.max(1, lo); j <= hi; j++){
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      cells++;
    }
    let best = INF;
    for (let j = lo; j <= hi; j++) best = Math.min(best, cur[j]);
    if (best > k) return {d:k + 1, cells:cells, gate:false, band:band, quit:i};
    prev = cur;
  }
  return {d:prev[m], cells:cells, gate:false, band:band};
}
function lcsLength(a, b){
  const n = a.length, m = b.length;
  const d = [];
  for (let i = 0; i <= n; i++) d.push(new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      d[i][j] = a[i - 1] === b[j - 1] ? d[i - 1][j - 1] + 1 : Math.max(d[i - 1][j], d[i][j - 1]);
  return d[n][m];
}
function wordErrorRate(ref, hyp){
  const r = ref.toLowerCase().split(/\s+/), h = hyp.toLowerCase().split(/\s+/);
  const ops = editOps(r, h);
  const c = {match:0, substitute:0, delete:0, insert:0};
  ops.forEach(o => { c[o[0]]++; });
  return {ops:ops, S:c.substitute, D:c.delete, I:c.insert, N:r.length,
          wer:(c.substitute + c.delete + c.insert) / r.length, ref:r, hyp:h};
}
/* ---- set-level similarity ------------------------------------------------ */
function shingles(text, k){
  const w = text.toLowerCase().split(/\s+/).filter(x => x.length);
  if (w.length < k) return w.length ? [w.join(' ')] : [];
  const out = [];
  for (let i = 0; i + k <= w.length; i++) out.push(w.slice(i, i + k).join(' '));
  return Array.from(new Set(out));
}
function jaccard(a, b){
  const B = new Set(b), inter = a.filter(x => B.has(x)).length;
  const uni = new Set(a.concat(b)).size;
  return uni ? inter / uni : 0;
}
/* seeded FNV-1a: deterministic in every browser, unlike JS object hashing */
function hashSeed(item, seed){
  let h = (2166136261 ^ (seed * 2654435761)) >>> 0;
  for (let i = 0; i < item.length; i++){
    h ^= item.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
function minhashSignature(set, K){
  const sig = [];
  for (let s = 0; s < K; s++){
    let best = Infinity, arg = null;
    set.forEach(x => { const v = hashSeed(x, s); if (v < best){ best = v; arg = x; } });
    sig.push([best, arg]);
  }
  return sig;
}
function estimateJaccard(sa, sb){
  let eq = 0;
  for (let i = 0; i < sa.length; i++) if (sa[i][0] === sb[i][0]) eq++;
  return eq / sa.length;
}
function bandKey(sig, band, rows){
  return sig.slice(band * rows, band * rows + rows).map(x => x[0]).join(',');
}
function probCandidate(j, b, r){ return 1 - Math.pow(1 - Math.pow(j, r), b); }
function lshThreshold(b, r){ return Math.pow(1 / b, 1 / r); }

/* ===================================================== shared DP geometry */
const A = 'kitten', B = 'sitting';
const CW = .66, CH = .50, GX = .70, GY = .60, TX = 2.55, TY = 1.15;
const cx = j => TX + j * GX;
const cy = i => TY + i * GY;
const ctr = (i, j) => [cx(j) + CW / 2, cy(i) + CH / 2];

/* vals[i][j] = number or null (not computed yet); states keyed 'i,j' */
function tbl(vals, states, opt){
  opt = opt || {};
  const out = [], st = states || {};
  for (let j = 1; j <= B.length; j++)
    out.push(S.t(cx(j) + CW / 2, TY - .52, B[j - 1], {c:COL.purpleL, fs:.40}));
  for (let i = 1; i <= A.length; i++)
    out.push(S.t(TX - .38, cy(i) + CH * .78, A[i - 1], {c:COL.purpleL, fs:.40}));
  out.push(S.t(TX - 1.55, TY - .52, 'b = ' + B, {c:COL.purpleL, fs:.32}));
  out.push(S.t(TX - 1.55, cy(3) + CH / 2, 'a = ' + A, {c:COL.purpleL, fs:.32, anchor:'middle'}));
  for (let i = 0; i <= A.length; i++)
    for (let j = 0; j <= B.length; j++){
      const v = vals[i] == null ? null : vals[i][j];
      const s = st[i + ',' + j] || (v == null ? 'ghost' : 'idle');
      out.push(S.r(cx(j), cy(i), CW, CH, s, v == null ? '' : String(v), {fs:.34}));
    }
  return out;
}
const blank = () => { const v = []; for (let i = 0; i <= A.length; i++) v.push(new Array(B.length + 1).fill(null)); return v; };
/* a partially filled table: every cell up to (upto_i, upto_j) in row-major order */
function upto(d, i0, j0){
  const v = blank();
  for (let j = 0; j <= B.length; j++) v[0][j] = d[0][j];
  for (let i = 1; i <= A.length; i++){
    v[i][0] = d[i][0];
    for (let j = 1; j <= B.length; j++)
      if (i < i0 || (i === i0 && j <= j0)) v[i][j] = d[i][j];
  }
  return v;
}

/* ===================================================== 1. the table */
const CODE_DP = [
'def edit_table(a, b):',
'    n, m = len(a), len(b)',
'    d = [[0] * (m + 1) for _ in range(n + 1)]',
'    for i in range(n + 1): d[i][0] = i        # delete i characters',
'    for j in range(m + 1): d[0][j] = j        # insert j characters',
'    for i in range(1, n + 1):',
'        for j in range(1, m + 1):',
'            cost = 0 if a[i-1] == b[j-1] else 1',
'            d[i][j] = min(d[i-1][j] + 1,      # delete a[i-1]',
'                          d[i][j-1] + 1,      # insert b[j-1]',
'                          d[i-1][j-1] + cost) # substitute / keep',
'    return d'
];

function dpFrames(){
  const F = new Frames(), d = editTable(A, B), n = A.length, m = B.length;
  const pan = (i, j) => [
    {lbl:{zh:'a 的前綴', en:'prefix of a'}, chips:[chip(A.slice(0, i) || 'ε', 'act')]},
    {lbl:{zh:'b 的前綴', en:'prefix of b'}, chips:[chip(B.slice(0, j) || 'ε', 'act')]},
    {lbl:{zh:'d[i][j]', en:'d[i][j]'}, chips:[chip(String(d[i][j]), 'hot')]}
  ];

  const v0 = blank();
  for (let j = 0; j <= m; j++) v0[0][j] = j;
  for (let i = 0; i <= n; i++) v0[i][0] = i;
  const st0 = {};
  for (let j = 0; j <= m; j++) st0['0,' + j] = 'ok';
  for (let i = 0; i <= n; i++) st0[i + ',0'] = 'ok';
  F.push({shapes:tbl(v0, {}), panels:pan(0, 0), view:VIEW, line:ln(CODE_DP, 'def edit_table'),
    msg:{zh:'問題是：把 <b>kitten</b> 改成 <b>sitting</b> 最少要幾次「插入、刪除、替換」？暴力法是列出所有改法，數量是指數級的。DP 的作法是<b>換一個問題</b>：<code>d[i][j]</code> = 把 a 的前 i 個字元改成 b 的前 j 個字元的最小成本。',
         en:'The question: what is the fewest insert / delete / substitute operations that turn <b>kitten</b> into <b>sitting</b>? Enumerating edit sequences is exponential. DP <b>changes the question</b>: let <code>d[i][j]</code> be the cheapest way to turn the first i characters of a into the first j characters of b.'}});
  F.push({shapes:tbl(v0, st0), panels:pan(0, 0), view:VIEW, line:ln(CODE_DP, 'for j in range'),
    msg:{zh:'邊界不用想太久：把 <b>i 個字元變成空字串</b>只能刪 i 次，把<b>空字串變成 j 個字元</b>只能插 j 次。所以第 0 列是 0,1,2,…，第 0 行也是。整張表就從這兩條邊長出來。',
         en:'The border is forced: turning <b>i characters into the empty string</b> costs i deletions, and turning the <b>empty string into j characters</b> costs j insertions. So row 0 is 0,1,2,… and column 0 is too. The whole table grows out of these two edges.'}});

  const arrows = (i, j) => [
    S.e(ctr(i - 1, j - 1)[0], ctr(i - 1, j - 1)[1], ctr(i, j)[0], ctr(i, j)[1], {s:'hot', pad:.30, w:.05}),
    S.e(ctr(i - 1, j)[0], ctr(i - 1, j)[1], ctr(i, j)[0], ctr(i, j)[1], {s:'act', pad:.26, w:.05}),
    S.e(ctr(i, j - 1)[0], ctr(i, j - 1)[1], ctr(i, j)[0], ctr(i, j)[1], {s:'act', pad:.26, w:.05})
  ];
  const three = (i, j) => {
    const s = {};
    s[(i - 1) + ',' + (j - 1)] = 'act'; s[(i - 1) + ',' + j] = 'act'; s[i + ',' + (j - 1)] = 'act';
    s[i + ',' + j] = 'hot';
    return s;
  };
  const legendTxt = [
    S.t(8.35, 2.05, 'delete', {c:COL.purpleL, fs:.30, anchor:'start'}),
    S.t(8.35, 2.45, 'insert', {c:COL.purpleL, fs:.30, anchor:'start'}),
    S.t(8.35, 2.85, 'substitute', {c:COL.orangeL, fs:.30, anchor:'start'})
  ];

  F.push({shapes:tbl(upto(d, 1, 0), three(1, 1)).concat(arrows(1, 1), legendTxt,
      [note(6.05, 'cost = 1  because  a[0] = k  is not  b[0] = s', COL.orangeL, .34)]),
    panels:pan(1, 1), view:VIEW, line:ln(CODE_DP, 'cost = 0 if'),
    msg:{zh:'每一格<b>只有三個來源</b>，因為對齊 a[i-1] 和 b[j-1] 只有三種選擇：刪掉 a[i-1]（從上面來）、插入 b[j-1]（從左邊來）、或讓兩者對上（從左上來，相同就免費、不同就付 1）。三選一取 min，這就是整個演算法。',
         en:'A cell has <b>exactly three sources</b>, because aligning a[i-1] with b[j-1] leaves only three choices: delete a[i-1] (come from above), insert b[j-1] (come from the left), or pair them up (come from the diagonal - free if they match, 1 if they do not). Take the min of the three; that is the entire algorithm.'}});
  F.push({shapes:tbl(upto(d, 1, 1), three(1, 1)).concat(arrows(1, 1), legendTxt,
      [note(6.05, 'min(1 + 1,  1 + 1,  0 + 1) = 1', COL.tealL, .34)]),
    panels:pan(1, 1), view:VIEW, line:ln(CODE_DP, 'd[i][j] = min'),
    msg:{zh:'k → s：三條路都是 1，但意義不同 - 對角線的 1 是「替換」，另外兩條是「刪一次再插一次」，那要 2。min 幫我們選了替換。注意這裡沒有任何回溯或猜測，答案是<b>從已經算好的三格算出來的</b>。',
         en:'k → s: all three routes cost 1 here, but they mean different things - the diagonal 1 is a substitution, while going around costs a delete plus an insert, which is 2. The min picks the substitution. Note there is no search and no guessing: the answer is <b>computed from three cells that are already known</b>.'}});
  F.push({shapes:tbl(upto(d, 2, 2), three(2, 2)).concat(arrows(2, 2), legendTxt,
      [note(6.05, 'cost = 0  because  a[1] = i  equals  b[1] = i', COL.tealL, .34)]),
    panels:pan(2, 2), view:VIEW, line:ln(CODE_DP, 'cost = 0 if'),
    msg:{zh:'字元相同的時候，對角線的 cost 是 <b>0</b>：不用付任何代價就把兩個字元對上。整條對角線的 0 就是「兩個字串共用的部分」- 這也是為什麼待會兒把替換禁用，同一張表就變成 LCS 和 diff。',
         en:'When the characters agree the diagonal cost is <b>0</b>: pairing them up is free. A run of zero-cost diagonal moves is exactly the part the two strings share - which is why, once substitution is forbidden, this same table computes the LCS and drives <code>diff</code>.'}});

  [3, 4, 5].forEach(i => {
    F.push({shapes:tbl(upto(d, i, m), (() => { const s = {}; for (let j = 0; j <= m; j++) s[i + ',' + j] = 'ok'; return s; })()),
      panels:pan(i, m), view:VIEW, line:ln(CODE_DP, 'for j in range(1, m + 1)'),
      msg:{zh:'第 ' + i + ' 列填完。每格 O(1)，一共 (n+1)(m+1) 格，所以時間是 <b>O(nm)</b> - 對兩個 1000 字元的字串是一百萬格，還可以；對整個語料庫的兩兩比較就不行了，那是這一天後半段要解的問題。',
           en:'Row ' + i + ' is done. Each cell is O(1) and there are (n+1)(m+1) of them, so the time is <b>O(nm)</b> - a million cells for two 1000-character strings, which is fine; all-pairs over a corpus is not, and that is what the second half of today solves.'}});
  });
  const stEnd = {}; stEnd[n + ',' + m] = 'ok';
  F.push({shapes:tbl(d, stEnd).concat([note(6.05, 'edit_distance("kitten", "sitting") = ' + d[n][m], COL.tealL, .40)]),
    panels:pan(n, m), view:VIEW, line:ln(CODE_DP, 'return d'),
    msg:{zh:'右下角就是答案：<b>' + d[n][m] + '</b>。但請注意我們算出來的是<b>整張表</b>，不只是一個數字 - 表裡藏著「怎麼改」的完整資訊。下一個分頁就把它讀出來。',
         en:'The bottom-right corner is the answer: <b>' + d[n][m] + '</b>. But notice that we computed the <b>whole table</b>, not just one number - the table also holds the complete record of <em>how</em> to edit. The next tab reads it out.'}});
  return F.list;
}

/* ===================================================== 2. backtrace */
const CODE_OPS = [
'def edit_ops(a, b):',
'    d = edit_table(a, b)',
'    i, j, ops = len(a), len(b), []',
'    while i > 0 or j > 0:',
'        if i and j and a[i-1] == b[j-1] and d[i][j] == d[i-1][j-1]:',
'            ops.append(("match", i - 1, a[i-1], b[j-1])); i, j = i-1, j-1',
'        elif i and j and d[i][j] == d[i-1][j-1] + 1:',
'            ops.append(("substitute", i - 1, a[i-1], b[j-1])); i, j = i-1, j-1',
'        elif i and d[i][j] == d[i-1][j] + 1:',
'            ops.append(("delete", i - 1, a[i-1], None)); i -= 1',
'        else:',
'            ops.append(("insert", i, None, b[j-1])); j -= 1',
'    return ops[::-1]'
];
const OPLINE = {match:'a[i-1] == b[j-1] and d[i][j] == d[i-1][j-1]',
                substitute:'d[i][j] == d[i-1][j-1] + 1', delete:'d[i][j] == d[i-1][j] + 1',
                insert:'ops.append(("insert"'};
const OPCLS = {match:'ok', substitute:'hot', delete:'bad', insert:'act'};

function backFrames(){
  const F = new Frames(), d = editTable(A, B), n = A.length, m = B.length;
  const path = pathCells(A, B), ops = editOps(A, B);
  const opTxt = o => o[0] === 'match' ? 'keep ' + o[2]
                 : o[0] === 'substitute' ? o[2] + ' -> ' + o[3]
                 : o[0] === 'delete' ? 'del ' + o[2] : 'ins ' + o[3];

  const stFor = upToIdx => {          /* path cells already walked, from the end */
    const s = {};
    for (let t = path.length - 1; t >= upToIdx; t--) s[path[t][0] + ',' + path[t][1]] = 'ok';
    if (upToIdx >= 0) s[path[upToIdx][0] + ',' + path[upToIdx][1]] = 'hot';
    return s;
  };
  const panFor = k => [
    {lbl:{zh:'游標 (i, j)', en:'cursor (i, j)'},
     chips:[chip('(' + path[k][0] + ', ' + path[k][1] + ')', 'act')]},
    {lbl:{zh:'已還原的編輯（由後往前）', en:'script so far (built backwards)'},
     chips:ops.slice(k).map(o => chip(opTxt(o), OPCLS[o[0]]))}
  ];

  F.push({shapes:tbl(d, stFor(path.length - 1)),
    panels:panFor(ops.length), view:VIEW, line:ln(CODE_OPS, 'i, j, ops = len(a)'),
    msg:{zh:'距離只是一個數字，但拼字修正、<code>git diff</code>、語音辨識的錯誤分析都要知道<b>改了什麼</b>。作法是從右下角往回走：每一格問「我這個值是從哪一格算出來的」，走過的路就是編輯腳本。',
         en:'The distance is one number, but spelling correction, <code>git diff</code> and ASR error analysis all need to know <b>what changed</b>. So walk back from the bottom-right corner, asking at each cell "which neighbour produced my value?" - the route you walk is the edit script.'}});

  for (let k = ops.length - 1; k >= 0; k--){
    const o = ops[k], cell = path[k];
    const nxt = path[k + 1];
    const arr = S.e(ctr(nxt[0], nxt[1])[0], ctr(nxt[0], nxt[1])[1], ctr(cell[0], cell[1])[0], ctr(cell[0], cell[1])[1], {s:'hot', pad:.28, w:.055});
    const why = {
      match:{zh:'字元一樣而且 <b>d 沒有變大</b> - 這一步是免費的「保留」。免費的斜線走越多，兩個字串就越像。',
             en:'The characters agree and <b>d did not grow</b> - this step is a free "keep". The more free diagonal steps, the more alike the two strings are.'},
      substitute:{zh:'走斜線但 d 少了 1 - 對角線的成本是 1，所以這一步是<b>替換</b>：把 ' + o[2] + ' 換成 ' + o[3] + '。',
                  en:'A diagonal step where d drops by 1 - the diagonal cost was 1, so this is a <b>substitution</b>: ' + o[2] + ' becomes ' + o[3] + '.'},
      'delete':{zh:'從上面來，代表 <b>刪掉 a 的 ' + o[2] + '</b>：j 不動，i 退一格。',
                en:'Coming from above means <b>deleting a\'s ' + o[2] + '</b>: j stays, i steps back.'},
      insert:{zh:'從左邊來，代表 <b>插入 b 的 ' + o[3] + '</b>：i 不動，j 退一格。這裡 i 已經到底了（kitten 用完），剩下的只能插。',
              en:'Coming from the left means <b>inserting b\'s ' + o[3] + '</b>: i stays, j steps back. Here i is already exhausted (kitten ran out), so insertion is the only move left.'}
    }[o[0]];
    F.push({shapes:tbl(d, stFor(k)).concat([arr]),
      panels:panFor(k), view:VIEW, line:ln(CODE_OPS, OPLINE[o[0]]),
      msg:why});
  }
  const scriptTxt = ops.filter(o => o[0] !== 'match').map(opTxt).join('   |   ');
  F.push({shapes:tbl(d, stFor(0)).concat([
      note(5.95, scriptTxt, COL.orangeL, .36),
      note(6.3, 'three non-free steps = distance ' + d[n][m], COL.tealL, .32)]),
    panels:panFor(0), view:VIEW, line:ln(CODE_OPS, 'return ops[::-1]'),
    msg:{zh:'走回原點，把清單反轉就是<b>從左到右的編輯腳本</b>：k→s、e→i、插入 g。不是免費的步數剛好是 3，和右下角的數字一致 - 這就是「距離」和「怎麼改」是同一張表的兩種讀法。',
         en:'Back at the origin; reverse the list and you have the <b>left-to-right edit script</b>: k→s, e→i, insert g. The number of non-free steps is exactly 3, matching the corner - "how far" and "how to get there" are two readings of one table.'}});
  return F.list;
}

/* ===================================================== 3. space and the band */
const CODE_ROLL = [
'def edit_distance_rows(a, b):            # O(min(n, m)) space',
'    prev = list(range(len(b) + 1))',
'    for i, ca in enumerate(a, 1):',
'        cur = [i] + [0] * len(b)',
'        for j, cb in enumerate(b, 1):',
'            cur[j] = min(prev[j] + 1, cur[j-1] + 1,',
'                         prev[j-1] + (ca != cb))',
'        prev = cur                       # the old row is gone forever',
'    return prev[-1]',
'',
'def edit_distance_bounded(a, b, k):      # "is it within k?"',
'    if abs(len(a) - len(b)) > k:         # length gate: every extra',
'        return k + 1                     # character costs an indel',
'    for i in range(1, len(a) + 1):',
'        lo, hi = max(0, i - k), min(len(b), i + k)   # the band',
'        ...                              # only these cells are computed',
'        if min(cur[lo:hi+1]) > k:        # the whole band blew the budget',
'            return k + 1                 # -> stop, no need to finish'
];

function rollFrames(vi){
  const F = new Frames(), d = editTable(A, B), n = A.length, m = B.length;
  if (vi === 0){
    const rows = editRows(A, B);
    F.push({shapes:tbl(d, {}).concat([note(6.05, (n + 1) + ' x ' + (m + 1) + ' = ' + ((n + 1) * (m + 1)) + ' cells stored', COL.grey, .34)]),
      panels:[{lbl:{zh:'記憶體', en:'memory'}, chips:[chip('O(n m)', 'bad')]}],
      view:VIEW, line:ln(CODE_ROLL, 'def edit_distance_rows'),
      msg:{zh:'整張表要 O(nm) 的記憶體。兩個 10000 字元的字串就是一億格 - 大約 800 MB。但看一下遞迴式：<code>d[i][j]</code> 只用到 <b>d[i-1][*] 和 d[i][j-1]</b>，也就是<b>上一列和左邊一格</b>，再往上的列一次都沒被讀到。',
           en:'The full table costs O(nm) memory. Two 10000-character strings is a hundred million cells, roughly 800 MB. But look at the recurrence: <code>d[i][j]</code> only reads <b>d[i-1][*] and d[i][j-1]</b> - the previous row and the cell to its left. Rows above that are never touched again.'}});
    for (let i = 1; i <= n; i++){
      const v = blank();
      for (let j = 0; j <= m; j++){ v[i - 1][j] = rows[i - 1][j]; v[i][j] = rows[i][j]; }
      const s = {};
      for (let j = 0; j <= m; j++){ s[(i - 1) + ',' + j] = 'act'; s[i + ',' + j] = 'hot'; }
      F.push({shapes:tbl(v, s).concat([
          S.t(8.3, cy(i - 1) + CH * .72, 'prev', {c:COL.purpleL, fs:.32, anchor:'start'}),
          S.t(8.3, cy(i) + CH * .72, 'cur', {c:COL.orangeL, fs:.32, anchor:'start'})]),
        panels:[{lbl:{zh:'prev', en:'prev'}, chips:rows[i - 1].map(x => chip(String(x), 'act'))},
                {lbl:{zh:'cur', en:'cur'}, chips:rows[i].map(x => chip(String(x), 'hot'))}],
        view:VIEW, line:ln(CODE_ROLL, i === n ? 'prev = cur' : 'cur[j] = min'),
        msg:{zh:'所以只留<b>兩列</b>就夠了：算完 cur 就把它變成 prev，舊的直接丟掉。記憶體從 O(nm) 掉到 <b>O(min(n, m))</b>（把比較短的那個字串放在 b），時間一格都沒有多花。',
             en:'So <b>two rows</b> are enough: finish cur, promote it to prev, throw the old one away. Memory drops from O(nm) to <b>O(min(n, m))</b> - put the shorter string in b - and not a single cell of extra work.'}});
    }
    const vlast = blank();
    for (let j = 0; j <= m; j++) vlast[n][j] = rows[n][j];
    const slast = {}; slast[n + ',' + m] = 'ok';
    for (let j = 0; j < m; j++) slast[n + ',' + j] = 'hot';
    F.push({shapes:tbl(vlast, slast).concat([
        note(5.95, 'answer = ' + d[n][m] + ',  but the edit script is unrecoverable', COL.orangeL, .34)]),
      panels:[{lbl:{zh:'距離', en:'distance'}, chips:[chip(String(d[n][m]), 'ok')]},
              {lbl:{zh:'能還原腳本嗎', en:'can we backtrace?'}, chips:[chip('no', 'bad')]}],
      view:VIEW, line:ln(CODE_ROLL, 'return prev[-1]'),
      msg:{zh:'代價是<b>不能回溯</b>：路徑資訊隨著被丟掉的列一起消失了。所以要「距離」用滾動列，要「腳本」就得留整張表（或用 Hirschberg 的分治法，用 O(n) 空間換兩倍時間）。先問自己要哪一個，再決定寫哪一版。',
           en:'The price is that you <b>cannot backtrace</b>: the route vanished with the discarded rows. So use the rolling row when you want the distance, and keep the table when you want the script (or use Hirschberg\'s divide and conquer, which recovers it in O(n) space for twice the time). Decide which you need first, then pick the version.'}});
    return F.list;
  }
  /* ---- variant 1: bounded k ---- */
  const K = 2, res = editBounded(A, B, K);
  const bandCells = (upToRow) => {
    const v = blank(), s = {};
    for (let i = 0; i <= Math.min(upToRow, n); i++){
      const lo = Math.max(0, i - K), hi = Math.min(m, i + K);
      for (let j = lo; j <= hi; j++){ v[i][j] = d[i][j]; s[i + ',' + j] = i === upToRow ? 'hot' : 'ok'; }
    }
    return [v, s];
  };
  F.push({shapes:tbl(d, {}).concat([note(6.05, 'do we need every cell to answer "is the distance <= 2?"', COL.orangeL, .34)]),
    panels:[{lbl:{zh:'計算的格數', en:'cells computed'}, chips:[chip(String(n * m), 'bad')]}],
    view:VIEW, line:ln(CODE_ROLL, 'def edit_distance_bounded'),
    msg:{zh:'實務上很少有人真的需要「距離是多少」- 拼字修正只想知道<b>距離有沒有 ≤ k</b>，去重只想知道<b>夠不夠像</b>。一旦問題變成是非題，整張表就不必算完。',
         en:'In practice you rarely need the exact distance - a spell checker only asks <b>is it within k</b>, and dedup only asks <b>are these close enough</b>. Once the question is a yes/no, you no longer have to fill the table.'}});
  F.push({shapes:tbl(d, (() => { const s = {}; for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++)
        if (Math.abs(i - j) > K) s[i + ',' + j] = 'bad'; return s; })())
      .concat([note(6.05, 'any cell with |i - j| > k already costs more than k', COL.orangeL, .34)]),
    panels:[{lbl:{zh:'k', en:'k'}, chips:[chip(String(K), 'hot')]}],
    view:VIEW, line:ln(CODE_ROLL, 'lo, hi = max(0, i - k)'),
    msg:{zh:'關鍵觀察：從 (0,0) 走到 (i,j) 至少要 <b>|i - j|</b> 次插入或刪除，因為每一步斜線都讓 i 和 j 同時加一。所以 <b>|i - j| > k</b> 的格子，值一定已經大於 k - 算它是純粹浪費。',
         en:'The key observation: reaching (i, j) from (0, 0) costs at least <b>|i - j|</b> insertions or deletions, because only diagonal steps advance i and j together. So every cell with <b>|i - j| > k</b> is already worth more than k - computing it is pure waste.'}});
  for (let i = 0; i <= n; i++){
    const [v, s] = bandCells(i);
    const lo = Math.max(0, i - K), hi = Math.min(m, i + K);
    F.push({shapes:tbl(v, s).concat([
        S.t(8.3, cy(i) + CH * .72, 'lo=' + lo + ' hi=' + hi, {c:COL.orangeL, fs:.30, anchor:'start'})]),
      panels:[{lbl:{zh:'這一列算的格數', en:'cells in this row'}, chips:[chip(String(hi - lo + (i ? 1 : 1)), 'hot')]},
              {lbl:{zh:'這一列最小值', en:'row minimum'},
               chips:[chip(String(Math.min.apply(null, d[i].slice(lo, hi + 1))), Math.min.apply(null, d[i].slice(lo, hi + 1)) > K ? 'bad' : 'ok')]}],
      view:VIEW, line:ln(CODE_ROLL, 'lo, hi = max(0, i - k)'),
      msg:{zh:'只算 <b>lo = max(0, i-k)</b> 到 <b>hi = min(m, i+k)</b> 這一段，寬度固定是 2k+1。每列 O(k)，n 列就是 <b>O(nk)</b> - k 是小常數的時候，這比 O(nm) 少一個數量級。',
           en:'Compute only <b>lo = max(0, i-k)</b> through <b>hi = min(m, i+k)</b>, a band of fixed width 2k+1. That is O(k) per row and <b>O(nk)</b> overall - when k is a small constant this is an order of magnitude less than O(nm).'}});
  }
  F.push({shapes:bandCells(n)[0] && tbl(bandCells(n)[0], bandCells(n)[1]).concat([
      note(5.95, 'k = 2:  ' + editBounded(A, B, 2).cells + ' cells,   k = 1:  ' + editBounded(A, B, 1).cells + ' cells (and it quits early),   full table:  ' + (n * m), COL.tealL, .32),
      note(6.3, 'a 132-character pair with k = 3: 912 of 17424 cells = 5.2%', COL.orangeL, .32)]),
    panels:[{lbl:{zh:'k = 2 的答案', en:'answer at k = 2'}, chips:[chip(String(editBounded(A, B, 2).d), 'ok')]},
            {lbl:{zh:'k = 1 的答案', en:'answer at k = 1'}, chips:[chip('> 1', 'bad')]}],
    view:VIEW, line:ln(CODE_ROLL, 'if min(cur[lo:hi+1]) > k'),
    msg:{zh:'還有兩個提早結束的機會：<b>長度閘門</b>（<code>|n-m| > k</code> 直接回答，一格都不用算）和<b>整條 band 都超過 k</b> 就中止。k=1 時第 5 列整條就爆了，只算 14 格就收工。實測 132 字元、k=3 只碰 <b>5.2%</b> 的格子。',
         en:'Two more early exits: the <b>length gate</b> (<code>|n-m| > k</code> answers without touching a single cell) and quitting as soon as <b>the entire band exceeds k</b>. At k = 1 row 5 blows the budget, so only 14 cells are ever computed. On a real 132-character pair with k = 3 this touches <b>5.2%</b> of the table.'}});
  return F.list;
}

/* ===================================================== 4. words and lines */
const CODE_WORD = [
'def word_error_rate(reference, hypothesis):',
'    ref, hyp = reference.split(), hypothesis.split()',
'    ops = edit_ops(ref, hyp)          # same DP, alphabet = words',
'    S = sum(o == "substitute" for o, *_ in ops)',
'    D = sum(o == "delete"     for o, *_ in ops)',
'    I = sum(o == "insert"     for o, *_ in ops)',
'    return (S + D + I) / len(ref)     # WER',
'',
'def lcs_length(a, b):                 # edit distance WITHOUT substitution',
'    for i, j in cells:',
'        if a[i-1] == b[j-1]: d[i][j] = d[i-1][j-1] + 1',
'        else:                d[i][j] = max(d[i-1][j], d[i][j-1])',
'    return d[n][m]                    # this is what git diff computes'
];
const REF = 'the model streams tokens back to the client';
const HYP = 'the model streamed tokens to a client';

function wordFrames(vi){
  const F = new Frames();
  if (vi === 0){
    const r = wordErrorRate(REF, HYP), ops = r.ops;
    const CWW = 1.14, X0 = .38, YR = 2.0, YH = 3.5, BH = .62;
    const draw = (upto) => {
      const out = [];
      out.push(S.t(X0, YR - .45, 'reference', {c:COL.tealL, fs:.32, anchor:'start'}));
      out.push(S.t(X0, YH + BH + .48, 'hypothesis (what the model heard)', {c:COL.purpleL, fs:.32, anchor:'start'}));
      ops.forEach((o, t) => {
        const x = X0 + t * CWW, on = t <= upto;
        const cls = !on ? 'ghost' : OPCLS[o[0]];
        if (o[2] != null) out.push(S.r(x, YR, CWW - .08, BH, on ? (o[0] === 'match' ? 'ok' : cls) : 'ghost', o[2], {fs:.26}));
        if (o[3] != null) out.push(S.r(x, YH, CWW - .08, BH, on ? (o[0] === 'match' ? 'ok' : cls) : 'ghost', o[3], {fs:.26}));
        if (on && o[2] != null && o[3] != null)
          out.push(S.e(x + (CWW - .08) / 2, YR + BH, x + (CWW - .08) / 2, YH, {s:o[0] === 'match' ? 'ok' : 'hot', pad:.06, arrow:false, w:.05}));
        if (on && o[0] !== 'match')
          out.push(S.t(x + (CWW - .08) / 2, (YR + YH) / 2 + BH / 2 + .12, o[0].toUpperCase()[0], {c:o[0] === 'delete' ? COL.red : COL.orangeL, fs:.36}));
      });
      return out;
    };
    F.push({shapes:draw(-1), view:VIEW, line:ln(CODE_WORD, 'def word_error_rate'),
      panels:[{lbl:{zh:'reference 詞數 N', en:'reference words N'}, chips:[chip(String(r.N), 'act')]}],
      msg:{zh:'到目前為止「字元」只是<b>序列裡的元素</b>，換成別的東西演算法完全不用改。把元素換成<b>詞</b>，同一個 DP 就是語音辨識的標準指標 <b>WER</b>。這也是為什麼值得把 edit distance 寫成吃 list 而不是只吃 str。',
           en:'So far a "character" has just been <b>an element of a sequence</b>; swap in something else and the algorithm does not change. Make the elements <b>words</b> and the same DP becomes <b>WER</b>, the standard speech-recognition metric. That is why it pays to write edit distance over lists, not just strings.'}});
    ops.forEach((o, t) => {
      const why = {
        match:{zh:'對上了，不算錯。', en:'Aligned, not an error.'},
        substitute:{zh:'<b>substitution</b>：「' + o[2] + '」被聽成「' + o[3] + '」。這種錯通常是聲學或口音問題。',
                    en:'<b>Substitution</b>: "' + o[2] + '" came out as "' + o[3] + '". These are usually acoustic or accent errors.'},
        'delete':{zh:'<b>deletion</b>：「' + o[2] + '」整個掉了。掉字通常是端點偵測或收音的問題，和替換是完全不同的病。',
                  en:'<b>Deletion</b>: "' + o[2] + '" is missing entirely. Dropped words usually mean endpointing or audio problems - a different disease from substitutions.'},
        insert:{zh:'<b>insertion</b>：模型多生了「' + o[3] + '」。', en:'<b>Insertion</b>: the model produced an extra "' + o[3] + '".'}
      }[o[0]];
      const c = {match:0, substitute:0, 'delete':0, insert:0};
      ops.slice(0, t + 1).forEach(x => { c[x[0]]++; });
      F.push({shapes:draw(t), view:VIEW,
        line:ln(CODE_WORD, o[0] === 'substitute' ? 'S = sum' : o[0] === 'delete' ? 'D = sum' : o[0] === 'insert' ? 'I = sum' : 'ops = edit_ops'),
        panels:[{lbl:{zh:'S / D / I', en:'S / D / I'},
                 chips:[chip('S=' + c.substitute, 'hot'), chip('D=' + c['delete'], 'bad'), chip('I=' + c.insert, 'act')]},
                {lbl:{zh:'目前 WER', en:'WER so far'},
                 chips:[chip(f3((c.substitute + c['delete'] + c.insert) / r.N), 'ok')]}],
        msg:why});
    });
    F.push({shapes:draw(ops.length).concat([
        note(5.6, 'WER = (' + r.S + 'S + ' + r.D + 'D + ' + r.I + 'I) / ' + r.N + ' = ' + f3(r.wer), COL.tealL, .40),
        note(6.1, 'the number is the summary; the breakdown is the bug report', COL.orangeL, .32)]),
      view:VIEW, line:ln(CODE_WORD, 'return (S + D + I)'),
      panels:[{lbl:{zh:'WER', en:'WER'}, chips:[chip(f3(r.wer), 'ok')]}],
      msg:{zh:'WER = (S + D + I) / N = <b>' + f3(r.wer) + '</b>。兩個系統可以有一樣的 WER 卻完全不同：全是 substitution 表示聽錯字，全是 deletion 表示句子被切掉。<b>拆解比數字有用</b>，而拆解正是回溯給我們的。注意分母是 reference 的長度，所以 WER 可以超過 1。',
           en:'WER = (S + D + I) / N = <b>' + f3(r.wer) + '</b>. Two systems can share a WER and be nothing alike: all substitutions means mishearing, all deletions means the audio got cut. <b>The breakdown beats the number</b>, and the breakdown is exactly what the backtrace gives you. Note the denominator is the reference length, so WER can exceed 1.'}});
    return F.list;
  }
  /* ---- variant 1: LCS / diff ---- */
  const OLD = ['import os', 'def main():', '    run()', '    return 0'];
  const NEW = ['import os', 'import sys', 'def main():', '    return 0'];
  const ops = editOps(OLD, NEW).filter(o => o[0] !== 'substitute' || true);
  /* diff forbids substitution: rebuild as keep / delete / insert via LCS */
  const script = [];
  {
    const nA = OLD.length, mB = NEW.length, dd = [];
    for (let i = 0; i <= nA; i++) dd.push(new Array(mB + 1).fill(0));
    for (let i = 1; i <= nA; i++) for (let j = 1; j <= mB; j++)
      dd[i][j] = OLD[i - 1] === NEW[j - 1] ? dd[i - 1][j - 1] + 1 : Math.max(dd[i - 1][j], dd[i][j - 1]);
    let i = nA, j = mB;
    while (i > 0 || j > 0){
      if (i > 0 && j > 0 && OLD[i - 1] === NEW[j - 1]){ script.push(['keep', OLD[i - 1]]); i--; j--; }
      else if (j > 0 && (i === 0 || dd[i][j - 1] >= dd[i - 1][j])){ script.push(['insert', NEW[j - 1]]); j--; }
      else { script.push(['delete', OLD[i - 1]]); i--; }
    }
    script.reverse();
  }
  const LH = .58, XL = 2.2, YL = 1.3;
  const drawDiff = upto => {
    const out = [];
    out.push(S.t(XL - .3, YL - .35, 'diff', {c:COL.tealL, fs:.34, anchor:'start'}));
    script.forEach((s, t) => {
      const on = t <= upto;
      const cls = !on ? 'ghost' : (s[0] === 'keep' ? 'idle' : s[0] === 'delete' ? 'bad' : 'ok');
      out.push(S.r(XL, YL + t * (LH + .08), 5.4, LH, cls,
        (s[0] === 'keep' ? '  ' : s[0] === 'delete' ? '- ' : '+ ') + s[1], {fs:.32, dx:-2.0}));
    });
    return out;
  };
  F.push({shapes:drawDiff(-1).concat([
      note(5.7, 'lcs_length(old, new) = ' + lcsLength(OLD, NEW) + '  ->  ' +
        (OLD.length - lcsLength(OLD, NEW)) + ' deletions + ' + (NEW.length - lcsLength(OLD, NEW)) + ' insertions', COL.orangeL, .34)]),
    view:VIEW, line:ln(CODE_WORD, 'def lcs_length'),
    panels:[{lbl:{zh:'LCS 長度', en:'LCS length'}, chips:[chip(String(lcsLength(OLD, NEW)), 'ok')]}],
    msg:{zh:'把<b>替換禁掉</b>，同一張表就變成另一個演算法：只剩插入和刪除，最省的走法就是<b>保留最長的共同子序列</b>（LCS）。這就是 <code>git diff</code> 在做的事 - 它從來不說「這一行改成那一行」，只說刪掉哪幾行、加上哪幾行。',
         en:'<b>Forbid substitution</b> and the same table becomes a different algorithm: with only insert and delete left, the cheapest route is the one that <b>keeps the longest common subsequence</b>. That is what <code>git diff</code> computes - it never says "this line became that line", only which lines went and which arrived.'}});
  script.forEach((s, t) => {
    const why = {
      keep:{zh:'這一行兩邊都有，而且順序沒衝突 - 屬於 LCS，diff 不動它。',
            en:'This line exists on both sides in a compatible order - it is part of the LCS, so diff leaves it alone.'},
      'delete':{zh:'舊檔有、新檔沒有：印成 <b>-</b>。注意 diff 不會說它「變成」了下面那一行 - 沒有替換這個選項。',
                en:'In the old file but not the new: printed as <b>-</b>. Note diff does not claim it "became" the line below - substitution is not on the menu.'},
      insert:{zh:'新檔有、舊檔沒有：印成 <b>+</b>。',
              en:'In the new file but not the old: printed as <b>+</b>.'}
    }[s[0]];
    F.push({shapes:drawDiff(t), view:VIEW,
      line:ln(CODE_WORD, s[0] === 'keep' ? 'if a[i-1] == b[j-1]' : 'else:'),
      panels:[{lbl:{zh:'diff 腳本', en:'diff script'},
               chips:script.slice(0, t + 1).map(x => chip((x[0] === 'keep' ? '=' : x[0] === 'delete' ? '-' : '+') + ' ' + x[1].trim(),
                 x[0] === 'keep' ? '' : x[0] === 'delete' ? 'bad' : 'ok'))}],
      msg:why});
  });
  F.push({shapes:drawDiff(script.length).concat([
      note(5.7, 'edit distance answers "how far"; LCS answers "what is shared"', COL.tealL, .34),
      note(6.15, 'same table, one operation removed', COL.orangeL, .32)]),
    view:VIEW, line:ln(CODE_WORD, 'return d[n][m]'),
    panels:[{lbl:{zh:'保留 / 刪除 / 新增', en:'keep / delete / insert'},
             chips:[chip(String(script.filter(s => s[0] === 'keep').length)),
                    chip(String(script.filter(s => s[0] === 'delete').length), 'bad'),
                    chip(String(script.filter(s => s[0] === 'insert').length), 'ok')]}],
    msg:{zh:'兩個演算法的關係值得記住：<b>沒有替換的編輯距離 = n + m - 2·LCS</b>。差別只是准不准用替換這一個選項，而選項一改，工具的用途就完全不同 - 一個給拼字修正，一個給版本控制。',
         en:'The relationship is worth remembering: <b>edit distance without substitution = n + m - 2·LCS</b>. The only difference is whether one operation is allowed - and that single choice sends the same table to two different jobs, spell correction and version control.'}});
  return F.list;
}

/* ===================================================== 5. shingles + Jaccard */
const DOCS = ['the cat sat on the mat', 'the cat sat on a mat', 'a dog ran through the park'];
const SH = DOCS.map(d => shingles(d, 2));
const CODE_JAC = [
'def shingles(text, k=5):              # k consecutive words',
'    w = text.lower().split()',
'    return {" ".join(w[i:i+k]) for i in range(len(w) - k + 1)}',
'',
'def jaccard(a, b):                    # |A n B| / |A u B|',
'    return len(a & b) / len(a | b) if (a | b) else 0.0',
'',
'def all_pairs_jaccard(docs, k=5, threshold=0.5):',
'    sets = [shingles(d, k) for d in docs]',
'    for i in range(len(docs)):        # n(n-1)/2 comparisons',
'        for j in range(i + 1, len(docs)):',
'            if jaccard(sets[i], sets[j]) >= threshold:',
'                yield i, j            # 10M docs -> 5 * 10^13 pairs'
];

function jacFrames(){
  const F = new Frames();
  const words = DOCS[0].split(' ');
  const WX = 1.0, WY = 1.1, WW = 1.28, WH = .60;
  const wordRow = (hi, y, ws, x0) => {
    const out = [];
    (ws || words).forEach((w, i) => out.push(S.r((x0 == null ? WX : x0) + i * WW, y, WW - .1, WH,
      hi && i >= hi[0] && i <= hi[1] ? 'hot' : 'idle', w, {fs:.32})));
    return out;
  };
  const shBoxes = (list, y, states) => {
    const out = [];
    list.forEach((s, i) => out.push(S.r(.55 + i * 1.84, y, 1.72, .56, (states && states[i]) || 'ok', s, {fs:.28})));
    return out;
  };

  F.push({shapes:[note(1.6, 'two 5000-character documents  ->  25 000 000 DP cells', COL.orangeL, .40),
      note(2.6, '10 000 000 documents  ->  5 * 10^13 pairs', COL.red, .40),
      note(3.9, 'edit distance is exact, and that is exactly why it does not scale', COL.tealL, .34),
      note(4.9, 'so: stop comparing sequences, start comparing sets', COL.purpleL, .36)],
    view:VIEW, line:ln(CODE_JAC, 'def all_pairs_jaccard'),
    panels:[{lbl:{zh:'兩兩比較', en:'all-pairs cost'}, chips:[chip('n(n-1)/2', 'bad')]}],
    msg:{zh:'前半段的方法在「兩個字串」上很好用，但語料去重要處理的是<b>幾千萬份文件</b>。就算每對只花 1 微秒，5×10¹³ 對也要跑一年半。問題不在 DP 慢，而在<b>對數太多</b> - 所以要換一個更粗、但可以壓縮和索引的相似度。',
         en:'The first half works beautifully for two strings, but corpus dedup faces <b>tens of millions of documents</b>. Even at one microsecond per pair, 5x10^13 pairs is a year and a half. The problem is not that the DP is slow, it is that there are <b>too many pairs</b> - so we need a coarser similarity that can be compressed and indexed.'}});

  for (let i = 0; i + 2 <= words.length; i++){
    F.push({shapes:wordRow([i, i + 1], WY).concat(shBoxes(SH[0].slice(0, i + 1), 3.2),
        [S.t(WX, WY - .35, 'doc 0', {c:COL.purpleL, fs:.32, anchor:'start'}),
         S.t(.55, 2.85, 'shingles, k = 2', {c:COL.tealL, fs:.32, anchor:'start'})]),
      view:VIEW, line:ln(CODE_JAC, 'return {" ".join'),
      panels:[{lbl:{zh:'shingles', en:'shingles'}, chips:SH[0].slice(0, i + 1).map(s => chip(s, 'ok'))}],
      msg:{zh:'把文件切成<b>連續 k 個詞</b>的視窗（shingle），整份文件就變成一個<b>集合</b>。為什麼不直接用單字集合？因為單字丟掉了順序 - 「dog bites man」和「man bites dog」的單字集合一模一樣，但 2-shingle 完全不同。k 就是「要保留多少順序」的旋鈕。',
           en:'Slide a window of <b>k consecutive words</b> (a shingle) over the document and the document becomes a <b>set</b>. Why not just a bag of words? Because a bag throws order away - "dog bites man" and "man bites dog" have identical word sets but no 2-shingle in common. k is the knob for how much order you keep.'}});
  }
  const inter = SH[0].filter(x => SH[1].indexOf(x) >= 0);
  const stA = {}, stB = {};
  SH[0].forEach((s, i) => { stA[i] = inter.indexOf(s) >= 0 ? 'hot' : 'idle'; });
  SH[1].forEach((s, i) => { stB[i] = inter.indexOf(s) >= 0 ? 'hot' : 'idle'; });
  const uni = new Set(SH[0].concat(SH[1])).size;
  F.push({shapes:shBoxes(SH[0], 1.5, stA).concat(shBoxes(SH[1], 3.0, stB),
      [S.t(.55, 1.2, 'doc 0: ' + DOCS[0], {c:COL.purpleL, fs:.30, anchor:'start'}),
       S.t(.55, 2.7, 'doc 1: ' + DOCS[1], {c:COL.purpleL, fs:.30, anchor:'start'}),
       note(4.5, '|A n B| = ' + inter.length + '        |A u B| = ' + uni, COL.orangeL, .40),
       note(5.3, 'J(doc 0, doc 1) = ' + inter.length + ' / ' + uni + ' = ' + f3(jaccard(SH[0], SH[1])), COL.tealL, .44)]),
    view:VIEW, line:ln(CODE_JAC, 'def jaccard'),
    panels:[{lbl:{zh:'共同的 shingle', en:'shared shingles'}, chips:inter.map(s => chip(s, 'hot'))},
            {lbl:{zh:'Jaccard', en:'Jaccard'}, chips:[chip(f3(jaccard(SH[0], SH[1])), 'ok')]}],
    msg:{zh:'Jaccard 相似度就是<b>交集除以聯集</b>，值域 0 到 1。這兩句只差一個字，J = ' + f3(jaccard(SH[0], SH[1])) + ' - 看起來偏低，因為短文件的 shingle 很少，改一個詞會同時毀掉 k 個 shingle。真實文件有上千個 shingle，這個效應就小得多。',
         en:'Jaccard similarity is <b>the intersection over the union</b>, between 0 and 1. These two sentences differ by one word and score J = ' + f3(jaccard(SH[0], SH[1])) + ' - lower than intuition suggests, because a short document has few shingles and changing one word destroys k of them at once. Real documents have thousands, so the effect is much milder.'}});
  const st2 = {};
  SH[2].forEach((s, i) => { st2[i] = 'idle'; });
  F.push({shapes:shBoxes(SH[0], 1.5, {}).concat(shBoxes(SH[2], 3.0, st2),
      [S.t(.55, 1.2, 'doc 0: ' + DOCS[0], {c:COL.purpleL, fs:.30, anchor:'start'}),
       S.t(.55, 2.7, 'doc 2: ' + DOCS[2], {c:COL.purpleL, fs:.30, anchor:'start'}),
       note(4.6, 'J(doc 0, doc 2) = 0 / ' + (new Set(SH[0].concat(SH[2])).size) + ' = 0.000', COL.red, .44),
       note(5.5, 'both documents contain "the" - and it buys them nothing', COL.orangeL, .32)]),
    view:VIEW, line:ln(CODE_JAC, 'def jaccard'),
    panels:[{lbl:{zh:'共同的 shingle', en:'shared shingles'}, chips:[]},
            {lbl:{zh:'Jaccard', en:'Jaccard'}, chips:[chip('0.000', 'bad')]}],
    msg:{zh:'不相關的文件是乾淨的 0。注意兩篇都有 "the" 和 "a"，但 shingle 是<b>成對出現的詞</b>，所以停用詞不會像 bag-of-words 那樣製造假的相似度 - 這是 shingle 相對於單字集合的第二個好處。',
         en:'Unrelated documents score a clean 0. Both contain "the" and "a", but a shingle is <b>a pair of adjacent words</b>, so stop words do not manufacture false similarity the way a bag of words does - the second advantage of shingling.'}});
  F.push({shapes:[note(1.5, 'Jaccard is cheap for one pair', COL.tealL, .38),
      note(2.5, 'but there are still n(n-1)/2 pairs, and each set has thousands of shingles', COL.orangeL, .34),
      note(3.7, '8 documents  ->  28 comparisons', COL.tealL, .36),
      note(4.5, '10 000 000 documents  ->  5 * 10^13 comparisons', COL.red, .40),
      note(5.6, 'next: shrink every set to 128 integers, then stop looking at most pairs', COL.purpleL, .34)],
    view:VIEW, line:ln(CODE_JAC, 'for j in range(i + 1'),
    panels:[{lbl:{zh:'還沒解決的問題', en:'still unsolved'}, chips:[chip('n^2 pairs', 'bad'), chip('big sets', 'bad')]}],
    msg:{zh:'換成集合並沒有解決規模問題，只是換了個形狀：<b>比較還是 n²，而且每個集合都很大</b>。接下來兩個分頁分別打掉這兩件事 - MinHash 把集合壓成固定長度的簽章，LSH 讓大部分的對根本不用比。',
         en:'Switching to sets did not solve the scale problem, it just changed its shape: <b>still n² comparisons, and every set is large</b>. The next two tabs kill these one at a time - MinHash compresses each set to a fixed-length signature, and LSH makes most pairs never get compared at all.'}});
  return F.list;
}

/* ===================================================== 6. MinHash */
const CODE_MH = [
'def minhash_signature(s, num_hashes=128):',
'    # hashlib, seeded - NOT python\'s hash(), which is randomised per process',
'    return [min(_hash(x, seed) for x in s) for seed in range(num_hashes)]',
'',
'def estimate_jaccard(sig_a, sig_b):',
'    same = sum(x == y for x, y in zip(sig_a, sig_b))',
'    return same / len(sig_a)          # P(min equal) = J, exactly',
'',
'# error ~ sqrt(J (1 - J) / K):  4x the hashes buys 2x the accuracy'
];
const KMH = 8;
const SIGS = SH.map(s => minhashSignature(s, KMH));

function mhFrames(){
  const F = new Frames();
  const RW = 1.52, RX = 2.5, RY = 1.35, RH = .50, RG = .58;
  const sigTable = (upto, mark) => {
    const out = [];
    out.push(S.t(RX + RW / 2, RY - .42, 'doc 0 argmin', {c:COL.purpleL, fs:.28}));
    out.push(S.t(RX + RW * 1.5 + .2, RY - .42, 'doc 1 argmin', {c:COL.purpleL, fs:.28}));
    out.push(S.t(RX - .5, RY - .42, 'hash', {c:COL.tealL, fs:.28}));
    for (let s = 0; s < KMH; s++){
      const y = RY + s * RG;
      if (s > upto) continue;
      const eq = SIGS[0][s][0] === SIGS[1][s][0];
      out.push(S.t(RX - .5, y + RH * .75, 'h' + s, {c:COL.grey, fs:.28}));
      out.push(S.r(RX, y, RW, RH, mark && s === upto ? 'hot' : (eq ? 'ok' : 'idle'), SIGS[0][s][1], {fs:.26}));
      out.push(S.r(RX + RW + .2, y, RW, RH, mark && s === upto ? 'hot' : (eq ? 'ok' : 'idle'), SIGS[1][s][1], {fs:.26}));
      out.push(S.t(RX + 2 * RW + .55, y + RH * .75, eq ? '=' : 'x', {c:eq ? COL.tealL : COL.red, fs:.34}));
    }
    return out;
  };
  const trueJ = jaccard(SH[0], SH[1]);
  F.push({shapes:[note(1.4, 'imagine shuffling the union of the two sets at random', COL.tealL, .36),
      note(2.4, 'and keeping only the element that lands first', COL.tealL, .36),
      note(3.6, 'P(both sets pick the same element) = |A n B| / |A u B| = J', COL.orangeL, .42),
      note(4.8, 'a hash function is that shuffle, and min() is "lands first"', COL.purpleL, .34)],
    view:VIEW, line:ln(CODE_MH, 'def minhash_signature'),
    panels:[{lbl:{zh:'真實 J', en:'true J'}, chips:[chip(f3(trueJ), 'act')]}],
    msg:{zh:'MinHash 的核心是一個機率等式。想像把聯集裡的元素<b>隨機排列</b>，兩個集合各自看「我這邊排最前面的是誰」。只有當那個第一名<b>同時屬於兩邊</b>時答案才會一樣，而它落在交集的機率剛好就是 <b>|A∩B| / |A∪B| = J</b>。',
         en:'MinHash rests on one probabilistic identity. Imagine <b>randomly permuting</b> the union and asking each set "which of my elements comes first?". The two answers agree exactly when that first element <b>belongs to both sets</b> - and the probability of that is precisely <b>|A n B| / |A u B| = J</b>.'}});
  for (let s = 0; s < KMH; s++){
    const eq = SIGS[0][s][0] === SIGS[1][s][0];
    let same = 0;
    for (let t = 0; t <= s; t++) if (SIGS[0][t][0] === SIGS[1][t][0]) same++;
    F.push({shapes:sigTable(s, true), view:VIEW, line:ln(CODE_MH, 'return [min(_hash'),
      panels:[{lbl:{zh:'相同的位置 / 已用的 hash', en:'agreements / hashes used'},
               chips:[chip(same + ' / ' + (s + 1), 'hot')]},
              {lbl:{zh:'目前估計值', en:'estimate so far'}, chips:[chip(f3(same / (s + 1)), 'ok')]},
              {lbl:{zh:'真實 J', en:'true J'}, chips:[chip(f3(trueJ), 'act')]}],
      msg:eq ? {zh:'h' + s + ' 的最小值兩邊都是「' + SIGS[0][s][1] + '」- 這個 shingle 落在<b>交集</b>裡，所以兩邊同時選中它。每一個這樣的 <b>=</b> 就是一次成功的伯努利試驗。',
                en:'Under h' + s + ' both minima are "' + SIGS[0][s][1] + '" - that shingle lies in the <b>intersection</b>, so both sets picked it. Every such <b>=</b> is one successful Bernoulli trial.'}
             : {zh:'h' + s + ' 兩邊選到不同的 shingle：最小的那個只屬於其中一邊。<b>不相等不代表演算法錯了</b>，它就是那 (1 - J) 的機率 - 我們是在用投硬幣估計 J。',
                en:'Under h' + s + ' the minima differ: the smallest element belongs to only one of the sets. <b>A mismatch is not a bug</b>, it is the (1 - J) side of the coin - we are estimating J by flipping it.'}});
  }
  const est = estimateJaccard(SIGS[0], SIGS[1]);
  F.push({shapes:sigTable(KMH - 1, false).concat([
      S.t(RX + 2 * RW + 1.2, RY + 1.5, 'estimate = ' + f3(est), {c:COL.orangeL, fs:.34, anchor:'start'}),
      S.t(RX + 2 * RW + 1.2, RY + 2.1, 'true J   = ' + f3(trueJ), {c:COL.tealL, fs:.34, anchor:'start'}),
      S.t(RX + 2 * RW + 1.2, RY + 2.9, 'error ~ sqrt(J(1-J)/K)', {c:COL.purpleL, fs:.30, anchor:'start'}),
      S.t(RX + 2 * RW + 1.2, RY + 3.4, 'K = 12   -> ' + f3(Math.sqrt(trueJ * (1 - trueJ) / 12)), {c:COL.grey, fs:.28, anchor:'start'}),
      S.t(RX + 2 * RW + 1.2, RY + 3.8, 'K = 128  -> ' + f3(Math.sqrt(trueJ * (1 - trueJ) / 128)), {c:COL.grey, fs:.28, anchor:'start'}),
      S.t(RX + 2 * RW + 1.2, RY + 4.2, 'K = 1024 -> ' + f3(Math.sqrt(trueJ * (1 - trueJ) / 1024)), {c:COL.grey, fs:.28, anchor:'start'})]),
    view:VIEW, line:ln(CODE_MH, 'return same / len(sig_a)'),
    panels:[{lbl:{zh:'估計值', en:'estimate'}, chips:[chip(f3(est), 'hot')]},
            {lbl:{zh:'真實 J', en:'true J'}, chips:[chip(f3(trueJ), 'ok')]},
            {lbl:{zh:'簽章大小', en:'signature size'}, chips:[chip(KMH + ' ints', 'act')]}],
    msg:{zh:'用 ' + KMH + ' 個 hash 估到 <b>' + f3(est) + '</b>，真值 ' + f3(trueJ) + '。誤差是 <b>√(J(1−J)/K)</b>：要精確一倍就要 <b>四倍</b>的 hash。實務上 K = 128 已經夠用，重點是<b>簽章長度和文件長度無關</b> - 十萬字的文章一樣壓成 128 個整數。',
         en:'With ' + KMH + ' hashes the estimate is <b>' + f3(est) + '</b> against a true ' + f3(trueJ) + '. The error is <b>sqrt(J(1-J)/K)</b>, so halving it costs <b>four times</b> the hashes. K = 128 is the usual choice, and the real win is that the <b>signature length does not depend on the document length</b> - a hundred-thousand-word article also compresses to 128 integers.'}});
  F.push({shapes:[note(1.5, 'python\'s hash("abc") differs between processes (PYTHONHASHSEED)', COL.red, .34),
      note(2.4, 'so a signature written today would not match one read tomorrow', COL.orangeL, .34),
      note(3.6, 'use hashlib with an explicit seed:', COL.tealL, .34),
      note(4.4, 'blake2b(item, digest_size=8, key=seed.to_bytes(8, "little"))', COL.purpleL, .34),
      note(5.5, 'reproducibility is a correctness property, not a nicety', COL.tealL, .32)],
    view:VIEW, line:ln(CODE_MH, 'randomised per process'),
    panels:[{lbl:{zh:'常見地雷', en:'the classic trap'}, chips:[chip('hash()', 'bad'), chip('hashlib + seed', 'ok')]}],
    msg:{zh:'一個很容易踩的坑：Python 內建的 <code>hash()</code> 對字串會<b>每個 process 隨機加鹽</b>，所以今天算的簽章明天讀出來對不上，索引整個報廢。MinHash 一定要用<b>指定 seed 的 hashlib</b>；這是正確性問題，不是講究。',
         en:'One trap catches everyone: Python\'s built-in <code>hash()</code> is <b>randomly salted per process</b> for strings, so a signature written today will not match one read tomorrow and the index silently rots. MinHash must use <b>hashlib with an explicit seed</b>. This is a correctness issue, not a style preference.'}});
  return F.list;
}

/* ===================================================== 7. LSH banding */
const CODE_LSH = [
'def band_buckets(signatures, bands, rows):',
'    for doc, sig in enumerate(signatures):',
'        for b in range(bands):        # split the signature into b bands',
'            key = (b, tuple(sig[b*rows:(b+1)*rows]))',
'            buckets[key].append(doc)  # identical band -> same bucket',
'',
'def candidate_pairs(signatures, bands, rows):',
'    return {pair for bucket in buckets.values()',
'                 for pair in combinations(bucket, 2)}',
'',
'# P(candidate) = 1 - (1 - J**rows)**bands',
'# threshold    = (1 / bands) ** (1 / rows)',
'# LSH is a FILTER: verify every candidate exactly afterwards'
];
const NB = 4, NR = 2;

function lshFrames(){
  const F = new Frames();
  const BX = 2.2, BY = 1.30, BW = 1.30, BH = .48, BG = .52;
  const bandsView = (upto) => {
    const out = [];
    out.push(S.t(BX + BW / 2, BY - .40, 'doc 0', {c:COL.purpleL, fs:.30}));
    out.push(S.t(BX + BW + .25 + BW / 2, BY - .40, 'doc 1', {c:COL.purpleL, fs:.30}));
    for (let b = 0; b < NB; b++){
      const eq = bandKey(SIGS[0], b, NR) === bandKey(SIGS[1], b, NR);
      for (let r = 0; r < NR; r++){
        const idx = b * NR + r, y = BY + (b * NR + r) * BG + b * .12;
        const on = b <= upto;
        const s = !on ? 'ghost' : (eq ? 'ok' : 'idle');
        out.push(S.r(BX, y, BW, BH, s, on ? SIGS[0][idx][1] : '', {fs:.24}));
        out.push(S.r(BX + BW + .25, y, BW, BH, s, on ? SIGS[1][idx][1] : '', {fs:.24}));
        if (r === 0) out.push(S.t(BX - .45, y + BH * 1.4, 'band ' + b, {c:COL.grey, fs:.26}));
      }
      if (b <= upto)
        out.push(S.t(BX + 2 * BW + .95, BY + (b * NR) * BG + b * .12 + BH * 1.4,
          eq ? 'same bucket' : 'different', {c:eq ? COL.tealL : COL.grey, fs:.28, anchor:'start'}));
    }
    return out;
  };
  F.push({shapes:[note(1.4, 'MinHash shrank each document to ' + KMH + ' integers', COL.tealL, .36),
      note(2.3, 'but comparing every pair of signatures is still n(n-1)/2', COL.red, .36),
      note(3.5, 'idea: only compare documents that already agree somewhere', COL.orangeL, .38),
      note(4.6, 'cut the signature into b bands of r rows, and hash each band', COL.purpleL, .34),
      note(5.4, 'two documents are candidates if ANY band matches exactly', COL.purpleL, .34)],
    view:VIEW, line:ln(CODE_LSH, 'def band_buckets'),
    panels:[{lbl:{zh:'bands x rows', en:'bands x rows'}, chips:[chip(NB + ' x ' + NR, 'act')]}],
    msg:{zh:'簽章變小了，但<b>對數還是 n²</b>。LSH 的想法很直接：與其比較每一對，不如<b>用雜湊桶把可能相似的湊在一起</b>。把 ' + KMH + ' 個數字切成 ' + NB + ' 段、每段 ' + NR + ' 個，每一段當成一把鑰匙丟進雜湊表 - 只要<b>有任何一段完全相同</b>，這兩份文件就成為候選。',
         en:'The signatures are small now, but <b>the number of pairs is still n^2</b>. LSH is blunt about it: instead of comparing every pair, <b>hash likely-similar documents into the same bucket</b>. Cut the ' + KMH + ' numbers into ' + NB + ' bands of ' + NR + ', treat each band as a key, and two documents become candidates as soon as <b>any single band matches exactly</b>.'}});
  for (let b = 0; b < NB; b++){
    const eq = bandKey(SIGS[0], b, NR) === bandKey(SIGS[1], b, NR);
    let hit = false;
    for (let t = 0; t <= b; t++) if (bandKey(SIGS[0], t, NR) === bandKey(SIGS[1], t, NR)) hit = true;
    F.push({shapes:bandsView(b), view:VIEW, line:ln(CODE_LSH, 'key = (b, tuple'),
      panels:[{lbl:{zh:'目前的 band', en:'band'}, chips:[chip('b' + b, 'hot')]},
              {lbl:{zh:'已經是候選了嗎', en:'candidate yet?'}, chips:[chip(hit ? 'yes' : 'not yet', hit ? 'ok' : '')]}],
      msg:eq ? {zh:'band ' + b + ' 兩邊<b>一模一樣</b> - 這兩份文件被丟進同一個桶，成為候選。注意這裡沒有做任何相似度計算，只是一次雜湊表查詢，<b>O(1)</b>。',
                en:'Band ' + b + ' is <b>identical</b> on both sides, so the two documents land in one bucket and become candidates. No similarity was computed here - this is a single hash-table lookup, <b>O(1)</b>.'}
             : {zh:'band ' + b + ' 不同，這一段沒有把它們湊在一起。但只要<b>還有別的 band 命中</b>就夠了 - 這正是為什麼要切成很多段：band 越多，漏掉真正相似的機率越低。',
                en:'Band ' + b + ' differs, so this band does not pair them up. That is fine as long as <b>some other band hits</b> - which is exactly why we use many bands: more bands, less chance of missing a genuinely similar pair.'}});
  }
  const PX0 = 1.75, PX1 = 8.3, PY0 = 5.25, PY1 = 1.45;
  const px = j => PX0 + j * (PX1 - PX0);
  const py = p => PY0 - p * (PY0 - PY1);
  const curve = (b, r, col, wd) => {
    const out = [], N = 40;
    for (let t = 0; t < N; t++){
      const j0 = t / N, j1 = (t + 1) / N;
      out.push(S.e(px(j0), py(probCandidate(j0, b, r)), px(j1), py(probCandidate(j1, b, r)),
        {arrow:false, w:wd || .045, s:col}));
    }
    return out;
  };
  const axes = [
    S.e(PX0, PY0, PX1 + .15, PY0, {arrow:false, w:.035, s:'soft'}),
    S.e(PX0, PY0, PX0, PY1 - .2, {arrow:false, w:.035, s:'soft'}),
    S.t(px(.5), PY0 + .55, 'true Jaccard similarity  J', {c:COL.grey, fs:.30}),
    S.t(PX0 - .25, py(.5), 'P(candidate)', {c:COL.grey, fs:.30, anchor:'end'}),
    S.t(PX0 - .18, py(0) + .1, '0', {c:COL.grey, fs:.26, anchor:'end'}),
    S.t(PX0 - .18, py(1) + .1, '1', {c:COL.grey, fs:.26, anchor:'end'}),
    S.t(px(0), PY0 + .28, '0', {c:COL.grey, fs:.26}),
    S.t(px(.5), PY0 + .28, '0.5', {c:COL.grey, fs:.26}),
    S.t(px(1), PY0 + .28, '1', {c:COL.grey, fs:.26})
  ];
  const CFG = [[32, 4, 'hot'], [16, 8, 'act'], [8, 16, 'ok']];
  CFG.forEach(function(cfg, idx){
    const b = cfg[0], r = cfg[1];
    const sh = axes.slice();
    for (let t = 0; t <= idx; t++){
      const bb = CFG[t][0], rr = CFG[t][1], cc = CFG[t][2];
      sh.push.apply(sh, curve(bb, rr, cc, t === idx ? .06 : .04));
      sh.push(S.e(px(lshThreshold(bb, rr)), PY0, px(lshThreshold(bb, rr)), py(.5),
        {arrow:false, dash:'.12 .12', w:.03, s:cc}));
      sh.push(S.t(px(lshThreshold(bb, rr)), PY1 - .55 + t * .38,
        'b=' + bb + ', r=' + rr + '  ->  threshold ' + f2(lshThreshold(bb, rr)),
        {c:cc === 'hot' ? COL.orangeL : cc === 'act' ? COL.purpleL : COL.tealL, fs:.28,
         anchor:t === 0 ? 'start' : t === 1 ? 'middle' : 'end'}));
    }
    F.push({shapes:sh, view:VIEW, line:ln(CODE_LSH, 'threshold    = (1 / bands)'),
      panels:[{lbl:{zh:'設定', en:'configuration'}, chips:[chip('b = ' + b, 'hot'), chip('r = ' + r, 'act')]},
              {lbl:{zh:'門檻', en:'threshold'}, chips:[chip(f2(lshThreshold(b, r)), 'ok')]},
              {lbl:{zh:'J = 0.5 時被選中的機率', en:'P(candidate) at J = 0.5'},
               chips:[chip(f3(probCandidate(.5, b, r)), probCandidate(.5, b, r) > .5 ? 'ok' : 'bad')]}],
      msg:{zh:'機率是 <b>P = 1 − (1 − J^r)^b</b>：一個 band 全中的機率是 J^r，b 個 band 至少中一個就是這個式子。它的形狀是 <b>S 曲線</b>，陡峭的位置大約在 <b>(1/b)^(1/r) = ' + f2(lshThreshold(b, r)) + '</b>。b 和 r 不是效能參數，它們<b>就是</b>你要的相似度門檻。',
           en:'The probability is <b>P = 1 - (1 - J^r)^b</b>: one band matches with probability J^r, and we need at least one of b bands. The shape is an <b>S-curve</b> whose steep part sits near <b>(1/b)^(1/r) = ' + f2(lshThreshold(b, r)) + '</b>. b and r are not performance knobs - they <b>are</b> the similarity threshold you are asking for.'}});
  });
  F.push({shapes:[note(1.3, '8 documents,  28 exact comparisons', COL.grey, .36),
      note(2.2, 'LSH proposed 4 candidate pairs   (14.3%)', COL.orangeL, .40),
      note(3.1, 'verify all 4 with exact Jaccard  ->  same answer as brute force', COL.tealL, .36),
      note(4.3, 'false positives are cheap: the verify step removes them', COL.tealL, .32),
      note(5.0, 'false negatives are the real risk: tune b and r, not the verify', COL.orangeL, .32),
      note(5.9, 'same shape as an ANN index: cheap filter, exact rescoring', COL.purpleL, .32)],
    view:VIEW, line:ln(CODE_LSH, 'LSH is a FILTER'),
    panels:[{lbl:{zh:'候選 / 全部', en:'candidates / all pairs'}, chips:[chip('4 / 28', 'hot')]},
            {lbl:{zh:'和暴力法一致嗎', en:'matches brute force?'}, chips:[chip('yes', 'ok')]}],
    msg:{zh:'最後一件事，也是最容易搞錯的一件事：<b>LSH 是過濾器，不是答案</b>。它給出候選，你還是要對每個候選算一次精確的 Jaccard。誤報很便宜（驗證就篩掉了），<b>漏報才貴</b>，而漏報只能靠 b、r 調。向量檢索的 ANN 索引也是同一個形狀：便宜的粗篩加上精確的重排。',
         en:'The last point, and the one people get wrong: <b>LSH is a filter, not an answer</b>. It proposes candidates and you still verify each one with an exact Jaccard. False positives are cheap - the verify removes them - while <b>false negatives are expensive</b>, and only b and r control those. Vector search indexes have the same shape: a cheap filter followed by exact rescoring.'}});
  return F.list;
}

/* ===================================================== 8. LeetCode */
const CODE_LC = [
'# 72. Edit Distance',
'def minDistance(word1, word2):',
'    prev = list(range(len(word2) + 1))',
'    for i, c1 in enumerate(word1, 1):',
'        cur = [i] + [0] * len(word2)',
'        for j, c2 in enumerate(word2, 1):',
'            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (c1 != c2))',
'        prev = cur',
'    return prev[-1]',
'',
'# 1143. Longest Common Subsequence = edit distance without substitution',
'#   match: d[i][j] = d[i-1][j-1] + 1     else: max(d[i-1][j], d[i][j-1])',
'',
'# 161. One Edit Distance - k = 1, so never build the table at all',
'def isOneEditDistance(s, t):',
'    if len(s) > len(t): s, t = t, s',
'    if len(t) - len(s) > 1 or s == t: return False',
'    for i, (a, b) in enumerate(zip(s, t)):',
'        if a != b:',
'            return s[i+1:] == t[i+1:] if len(s) == len(t) else s[i:] == t[i+1:]',
'    return True                        # t has exactly one extra character'
];
function smallTbl(a, b, vals, states, X, Y, cw, ch, gx, gy){
  const out = [], st = states || {};
  for (let j = 1; j <= b.length; j++) out.push(S.t(X + j * gx + cw / 2, Y - .45, b[j - 1], {c:COL.purpleL, fs:.36}));
  for (let i = 1; i <= a.length; i++) out.push(S.t(X - .38, Y + i * gy + ch * .78, a[i - 1], {c:COL.purpleL, fs:.36}));
  for (let i = 0; i <= a.length; i++)
    for (let j = 0; j <= b.length; j++){
      const v = vals[i] == null ? null : vals[i][j];
      out.push(S.r(X + j * gx, Y + i * gy, cw, ch, st[i + ',' + j] || (v == null ? 'ghost' : 'idle'),
        v == null ? '' : String(v), {fs:.34}));
    }
  return out;
}

function lcFrames(vi){
  const F = new Frames();
  const X = 3.6, Y = 1.5, cw = .78, ch = .60, gx = .86, gy = .72;
  if (vi === 0){
    const a = 'horse', b = 'ros', d = editTable(a, b), rows = [];
    let prev = []; for (let j = 0; j <= b.length; j++) prev.push(j);
    rows.push(prev.slice());
    for (let i = 1; i <= a.length; i++){
      const cur = new Array(b.length + 1).fill(0); cur[0] = i;
      for (let j = 1; j <= b.length; j++)
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur; rows.push(cur.slice());
    }
    for (let i = 0; i <= a.length; i++){
      const v = [];
      for (let t = 0; t <= a.length; t++) v.push(t <= i ? rows[t].slice() : new Array(b.length + 1).fill(null));
      const s = {};
      for (let j = 0; j <= b.length; j++){ s[i + ',' + j] = 'hot'; if (i) s[(i - 1) + ',' + j] = 'act'; }
      F.push({shapes:smallTbl(a, b, v, s, X, Y, cw, ch, gx, gy).concat([
          S.t(X + 2 * gx, Y - 1.15, 'horse  ->  ros', {c:COL.tealL, fs:.40})]),
        view:VIEW, line:ln(CODE_LC, i === 0 ? 'prev = list(range' : 'cur[j] = min'),
        panels:[{lbl:{zh:'prev', en:'prev'}, chips:(i ? rows[i - 1] : rows[0]).map(x => chip(String(x), 'act'))},
                {lbl:{zh:'cur', en:'cur'}, chips:rows[i].map(x => chip(String(x), 'hot'))}],
        msg:i === 0 ? {zh:'LC 72 就是今天的主題本身。面試時值得直接寫<b>滾動列</b>版本：程式碼一樣短，空間卻從 O(nm) 掉到 O(m)，而且能順口解釋為什麼可以這樣做（永遠只讀上一列）。',
                       en:'LC 72 is today\'s topic itself. In an interview go straight for the <b>rolling row</b>: the code is no longer, the space drops from O(nm) to O(m), and you get to explain why it is safe - only the previous row is ever read.'}
                    : {zh:'第 ' + i + ' 列。每格還是那三個候選 - 面試官想聽的通常不是程式碼，而是<b>「為什麼只有三種來源」</b>：因為對齊 a[i-1] 和 b[j-1] 只剩刪、插、配對三種可能。',
                       en:'Row ' + i + '. Same three candidates per cell - what the interviewer usually wants is not the code but <b>why there are only three sources</b>: aligning a[i-1] with b[j-1] leaves delete, insert, or pair, and nothing else.'}});
    }
    const sEnd = {}; sEnd[a.length + ',' + b.length] = 'ok';
    const v = []; for (let t = 0; t <= a.length; t++) v.push(rows[t].slice());
    F.push({shapes:smallTbl(a, b, v, sEnd, X, Y, cw, ch, gx, gy).concat([
        note(5.9, 'minDistance("horse", "ros") = ' + d[a.length][b.length] +
          '        minDistance("intention", "execution") = ' + editDistance('intention', 'execution'), COL.tealL, .34)]),
      view:VIEW, line:ln(CODE_LC, 'return prev[-1]'),
      panels:[{lbl:{zh:'答案', en:'answer'}, chips:[chip(String(d[a.length][b.length]), 'ok')]}],
      msg:{zh:'答案 <b>' + d[a.length][b.length] + '</b>：horse → rorse → rose → ros。順帶一提，面試很常追問「如果只要知道距離有沒有 ≤ k」- 那就是分頁 3 的 band，<b>O(nk)</b>，而且大多數情況根本跑不到最後一列。',
           en:'The answer is <b>' + d[a.length][b.length] + '</b>: horse -> rorse -> rose -> ros. A very common follow-up is "what if I only need to know whether the distance is at most k" - that is the band from tab 3, <b>O(nk)</b>, and it usually quits long before the last row.'}});
    return F.list;
  }
  if (vi === 1){
    const a = 'abcde', b = 'ace', n = a.length, m = b.length, dd = [];
    for (let i = 0; i <= n; i++) dd.push(new Array(m + 1).fill(0));
    for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++)
      dd[i][j] = a[i - 1] === b[j - 1] ? dd[i - 1][j - 1] + 1 : Math.max(dd[i - 1][j], dd[i][j - 1]);
    for (let i = 0; i <= n; i++){
      const v = [];
      for (let t = 0; t <= n; t++) v.push(t <= i ? dd[t].slice() : new Array(m + 1).fill(null));
      const s = {};
      for (let j = 0; j <= m; j++) s[i + ',' + j] = 'hot';
      F.push({shapes:smallTbl(a, b, v, s, X, Y, cw, ch, gx, gy).concat([
          S.t(X + 2 * gx, Y - 1.15, 'LCS("abcde", "ace")', {c:COL.tealL, fs:.38})]),
        view:VIEW, line:ln(CODE_LC, '# 1143'),
        panels:[{lbl:{zh:'這一列', en:'this row'}, chips:dd[i].map(x => chip(String(x), 'hot'))}],
        msg:{zh:'LC 1143 的遞迴式和編輯距離只差一個地方：字元相同時 <b>+1</b>，不同時取 <b>max</b> 而不是 min - 因為這次是在<b>最大化保留</b>，不是最小化成本。看得出這一點，兩題就是同一題。',
             en:'The recurrence differs from edit distance in exactly one place: on a match it <b>adds 1</b>, and on a mismatch it takes the <b>max</b> rather than the min - because here we <b>maximise what is kept</b> instead of minimising what is spent. Once you see that, the two problems are one problem.'}});
    }
    const sE = {}; sE[n + ',' + m] = 'ok';
    F.push({shapes:smallTbl(a, b, dd, sE, X, Y, cw, ch, gx, gy).concat([
        note(5.9, 'LCS = ' + dd[n][m] + '  ("ace")        edit distance without substitution = n + m - 2 LCS = ' +
          (n + m - 2 * dd[n][m]), COL.tealL, .34)]),
      view:VIEW, line:ln(CODE_LC, '#   match:'),
      panels:[{lbl:{zh:'LCS', en:'LCS'}, chips:[chip(String(dd[n][m]), 'ok')]},
              {lbl:{zh:'n + m - 2·LCS', en:'n + m - 2 LCS'}, chips:[chip(String(n + m - 2 * dd[n][m]), 'act')]}],
      msg:{zh:'LCS = ' + dd[n][m] + '，而不用替換的編輯距離就是 <b>n + m − 2·LCS = ' + (n + m - 2 * dd[n][m]) + '</b>。這條式子在面試裡很值錢：它把「最少刪幾個字元讓兩個字串相同」（LC 583）直接變成一題 LCS。',
           en:'LCS = ' + dd[n][m] + ', and the substitution-free edit distance is <b>n + m - 2 LCS = ' + (n + m - 2 * dd[n][m]) + '</b>. That identity earns its keep in interviews: it turns "minimum deletions to make two strings equal" (LC 583) straight into an LCS.'}});
    return F.list;
  }
  const CASES = [['abc', 'abxc', true], ['abc', 'abx', true], ['abc', 'abc', false], ['abc', 'abxyc', false]];
  const BXX = 1.4, BYY = 2.0, BWW = .74, BHH = .74;
  CASES.forEach(function(c, ci){
    const s = c[0], t = c[1], ans = c[2];
    let i = 0;
    while (i < Math.min(s.length, t.length) && s[i] === t[i]) i++;
    const stS = {}, stT = {};
    for (let x = 0; x < i; x++){ stS[x] = 'ok'; stT[x] = 'ok'; }
    const gate = Math.abs(s.length - t.length) > 1 || s === t;
    if (!gate){ if (i < s.length) stS[i] = 'hot'; if (i < t.length) stT[i] = 'hot'; }
    const row = (str, y, st, x0) => str.split('').map((ch, x) =>
      S.r(x0 + x * (BWW + .12), y, BWW, BHH, gate ? 'bad' : (st[x] || 'idle'), ch, {fs:.40}));
    F.push({shapes:row(s, BYY, stS, BXX).concat(row(t, BYY + 1.5, stT, BXX), [
        S.t(BXX - .35, BYY + BHH * .7, 's', {c:COL.tealL, fs:.34, anchor:'end'}),
        S.t(BXX - .35, BYY + 1.5 + BHH * .7, 't', {c:COL.tealL, fs:.34, anchor:'end'}),
        S.t(6.1, BYY + .3, gate ? 'rejected by a gate' : 'first difference at i = ' + i,
          {c:gate ? COL.red : COL.orangeL, fs:.34, anchor:'start'}),
        S.t(6.1, BYY + 1.0, gate ? (s === t ? 'identical -> distance 0, not 1' : 'lengths differ by more than 1')
          : (s.length === t.length ? 'same length -> the rest must match' : 'one longer -> skip one char of t'),
          {c:COL.grey, fs:.30, anchor:'start'}),
        S.t(6.1, BYY + 1.7, 'isOneEditDistance = ' + ans, {c:ans ? COL.tealL : COL.red, fs:.36, anchor:'start'}),
        note(5.9, 'no table, no O(nm): one pass, O(1) extra space', COL.purpleL, .32)]),
      view:VIEW, line:ln(CODE_LC, gate ? 'if len(t) - len(s) > 1' : 'if a != b'),
      panels:[{lbl:{zh:'s / t', en:'s / t'}, chips:[chip(s, 'act'), chip(t, 'act')]},
              {lbl:{zh:'答案', en:'answer'}, chips:[chip(String(ans), ans ? 'ok' : 'bad')]}],
      msg:ci === 0 ? {zh:'LC 161 問「距離<b>剛好</b>是 1 嗎」。很多人反射性地建整張表 - 但 k 固定成 1 的時候根本不需要 DP：<b>掃到第一個不同的位置</b>，剩下只有兩種可能，比對後綴就結束了。O(n) 時間、O(1) 空間。',
                     en:'LC 161 asks whether the distance is <b>exactly</b> 1. The reflex is to build the table - but with k pinned to 1 there is no DP to do: <b>walk to the first difference</b>, and only two cases remain, both settled by comparing the suffixes. O(n) time, O(1) space.'}
              : gate ? {zh:'先用兩個閘門擋掉：長度差超過 1 就不可能是 1 次編輯（和分頁 3 的 length gate 是同一個道理），完全相同則距離是 0 不是 1。閘門擋掉的 case 一個字元都不用比。',
                        en:'Two gates come first: a length difference above 1 cannot be one edit - the same argument as the length gate in tab 3 - and identical strings are distance 0, not 1. Anything a gate rejects costs no character comparisons at all.'}
              : {zh:'長度' + (s.length === t.length ? '相同 → 這一步只能是<b>替換</b>，所以後面的後綴必須完全一樣。' : '差 1 → 這一步只能是<b>插入</b>，跳過 t 的一個字元後，後綴必須完全一樣。') + '兩種情況都只剩一次字串比較。',
                 en:'The lengths ' + (s.length === t.length ? 'match, so this step can only be a <b>substitution</b> and the suffixes after it must be identical.' : 'differ by one, so this step can only be an <b>insertion</b>: skip one character of t and the suffixes must be identical.') + ' Either way, one suffix comparison finishes the job.'}});
  });
  F.push({shapes:[note(1.6, '72:  the table itself', COL.tealL, .38),
      note(2.5, '1143:  the same table with substitution removed', COL.tealL, .38),
      note(3.4, '161:  k = 1, so no table at all', COL.orangeL, .38),
      note(4.7, 'the signal is knowing WHICH of the three the question needs', COL.purpleL, .34),
      note(5.6, 'and at corpus scale the answer is not a table either: MinHash + LSH', COL.tealL, .32)],
    view:VIEW, line:ln(CODE_LC, 'return True'),
    panels:[{lbl:{zh:'三題的關係', en:'the three problems'},
             chips:[chip('72 full DP', 'ok'), chip('1143 no substitute', 'ok'), chip('161 k=1, O(1) space', 'hot')]}],
    msg:{zh:'三題其實是同一個遞迴式的三種裁剪：要精確距離就填表，要共同部分就拿掉替換，只問「是不是 1」就連表都不建。<b>先問要回答什麼問題，再決定資料結構</b> - 到了語料規模，答案就變成 MinHash 加 LSH。',
         en:'The three problems are three trims of one recurrence: fill the table for the exact distance, drop substitution for the shared part, and skip the table entirely when the question is just "is it one?". <b>Decide what question you are answering before you pick the structure</b> - and at corpus scale the answer stops being a table at all: it is MinHash plus LSH.'}});
  return F.list;
}

/* ===================================================================== */
const DAY_META = {
  title:{zh:'Day 27 - 編輯距離與相似度：Levenshtein、Jaccard、MinHash',
         en:'Day 27 - Edit distance and similarity: Levenshtein, Jaccard, MinHash'},
  sub:{zh:'「這兩段文字有多像」有兩種答案：逐字元精確的 Levenshtein，和集合層級、可以壓成 128 個整數並且建索引的 Jaccard。',
       en:'"How similar are these two texts" has two answers: character-exact Levenshtein, and set-level Jaccard - which compresses to 128 integers and can be indexed.'},
  tabs:[
    {id:'dp', label:{zh:'1. Levenshtein 表', en:'1. the Levenshtein table'},
     stage:{zh:'kitten -> sitting，每格只有三個來源', en:'kitten -> sitting, three sources per cell'},
     view:VIEW, idea:{zh:'d[i][j] = 前 i 個字元變成前 j 個字元的最小成本；三選一取 min。',
                      en:'d[i][j] = cheapest way to turn the first i characters into the first j; take the min of three.'},
     legend:['hot', 'act', 'ok', 'ghost'], code:CODE_DP, build:dpFrames},
    {id:'back', label:{zh:'2. 回溯出編輯腳本', en:'2. backtrace the script'},
     stage:{zh:'從右下角往回走', en:'walking back from the corner'},
     view:VIEW, idea:{zh:'距離是一個數字，但拼字修正和 diff 要的是「改了什麼」。',
                      en:'The distance is a number, but spell check and diff need to know what changed.'},
     legend:['hot', 'ok', 'bad', 'act'], code:CODE_OPS, build:backFrames},
    {id:'roll', label:{zh:'3. 空間與 k-band', en:'3. space and the k-band'},
     stage:{zh:'只留兩列 / 只算 |i-j| <= k 的格子', en:'keep two rows / compute only |i-j| <= k'},
     view:VIEW, variants:[{zh:'滾動列：O(min(n,m)) 空間', en:'rolling row: O(min(n,m)) space'},
                          {zh:'bounded k：band 與提早結束', en:'bounded k: the band and the early exit'}],
     idea:{zh:'實務上要的是「距離有沒有 ≤ k」，而那個問題便宜得多。',
           en:'In practice the question is "is it within k", and that question is far cheaper.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_ROLL, build:rollFrames},
    {id:'word', label:{zh:'4. 換掉字母表：WER 與 diff', en:'4. change the alphabet: WER and diff'},
     stage:{zh:'元素從字元換成詞，再換成程式碼行', en:'elements become words, then lines of code'},
     view:VIEW, variants:[{zh:'WER：S / D / I 拆解', en:'WER: the S / D / I breakdown'},
                          {zh:'LCS 與 git diff', en:'LCS and git diff'}],
     idea:{zh:'同一個 DP，換一種元素就是另一個工具；拿掉替換就變成 diff。',
           en:'One DP; change the element type and it is a different tool. Remove substitution and it is diff.'},
     legend:['hot', 'ok', 'bad', 'act'], code:CODE_WORD, build:wordFrames},
    {id:'jac', label:{zh:'5. Shingle 與 Jaccard', en:'5. shingles and Jaccard'},
     stage:{zh:'文件變成集合', en:'a document becomes a set'},
     view:VIEW, idea:{zh:'精確的編輯距離撐不住語料規模：n² 對，每對 O(nm)。',
                      en:'Exact edit distance cannot survive corpus scale: n^2 pairs, each O(nm).'},
     legend:['hot', 'ok', 'idle', 'bad'], code:CODE_JAC, build:jacFrames},
    {id:'mh', label:{zh:'6. MinHash', en:'6. MinHash'},
     stage:{zh:'P(兩邊最小值相同) = J', en:'P(the two minima agree) = J'},
     view:VIEW, idea:{zh:'把任意大的集合壓成固定 K 個整數，誤差 ~ √(J(1−J)/K)。',
                      en:'Compress any set to a fixed K integers; the error is ~ sqrt(J(1-J)/K).'},
     legend:['hot', 'ok', 'idle', 'act'], code:CODE_MH, build:mhFrames},
    {id:'lsh', label:{zh:'7. LSH banding', en:'7. LSH banding'},
     stage:{zh:'切成 b 段，任一段相同就是候選', en:'b bands; any identical band makes a candidate'},
     view:VIEW, idea:{zh:'P = 1 − (1 − J^r)^b，門檻 ≈ (1/b)^(1/r)；LSH 是過濾器，不是答案。',
                      en:'P = 1 - (1 - J^r)^b, threshold ~ (1/b)^(1/r); LSH is a filter, not an answer.'},
     legend:['hot', 'ok', 'act', 'idle'], code:CODE_LSH, build:lshFrames},
    {id:'lc', label:{zh:'8. LeetCode', en:'8. LeetCode'},
     stage:{zh:'72 / 1143 / 161', en:'72 / 1143 / 161'},
     view:VIEW, variants:[{zh:'72 Edit Distance', en:'72 Edit Distance'},
                          {zh:'1143 Longest Common Subsequence', en:'1143 Longest Common Subsequence'},
                          {zh:'161 One Edit Distance', en:'161 One Edit Distance'}],
     idea:{zh:'同一個遞迴式的三種裁剪：填表、拿掉一個操作、或根本不建表。',
           en:'Three trims of one recurrence: fill it, drop an operation, or skip the table entirely.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_LC, build:lcFrames}
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
