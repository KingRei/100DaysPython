// DAY: 65
// TITLE_ZH: Embedding 與向量檢索：把一份文件變成一個方向，然後問「誰的方向最近」
// TITLE_EN: Embeddings and vector search: a document becomes a direction, and retrieval asks whose direction is closest
// SUB_ZH: 用一個「答案是種進去的」語料把每個決定都量出來：要不要正規化（差 0.617 vs 0.092）、留幾維（256 維反而輸 16 維）、每個數字留幾個 bit（int8 免費、1 bit 不免費）、一份文件要幾個向量（長文件切段是 5.8 倍召回），以及什麼時候第二個分數其實會害你。
// SUB_EN: A corpus with planted answers, so every decision can be measured: normalise or not (0.617 against 0.092), how many dimensions to keep (256 loses to 16), how few bits per number (int8 is free, one bit is not), one vector per document or several (sections are 5.8x the recall), and when a second score makes things worse.
// FOLDER: day%2065%20-%20embeddings%20and%20vector%20search
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
const box = (x, y, w, h, s, lab, fs) => S.r(x, y, w, h, s, lab, {fs:fitFs(lab, w, fs || h * .5)});
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
const side = (sh, x, y0, lines, dy) => (lines || []).forEach((l, k) =>
  sh.push(fitT(x, y0 + k * (dy || .5), l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));
const fmtN = x => x == null ? 'nan' : String(Math.round(x * 1e4) / 1e4);
const PURPLE = '#b45cff';

function barRow(sh, x0, base, bw, gap, vals, sts, labs, maxh, top){
  const hi = top || Math.max.apply(null, vals.map(Math.abs));
  vals.forEach((v, i) => {
    const hgt = Math.max(.05, maxh * Math.abs(v) / hi), x = x0 + i * (bw + gap);
    sh.push(S.r(x, base - hgt, bw, hgt, sts[i] || 'ok', null));
    if (labs && labs[i] != null) sh.push(fitT(x + bw / 2, base - hgt - .20, labs[i], {c:COL.pale, fs:.24}));
  });
  return sh;
}
// a 0..1 x 0..1 plot box; data y grows upward, SVG y grows downward
function Panel(x0, y0, w, h){ this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; }
Panel.prototype.px = function(p){
  return [this.x0 + this.w * p[0], this.y0 + this.h * (1 - p[1])];
};
Panel.prototype.frame = function(sh){
  sh.push(S.r(this.x0, this.y0, this.w, this.h, 'ghost', null));
  return sh;
};
function polyline(sh, panel, pts, st, w){
  for (let i = 1; i < pts.length; i++){
    const a = panel.px(pts[i - 1]), b = panel.px(pts[i]);
    sh.push(S.e(a[0], a[1], b[0], b[1], {s:st, arrow:false, w:w || .055}));
  }
  return sh;
}
const CFG = {"docs": 20000, "queries": 200, "targets": 10, "dims": 256, "vocab": 800};
const HEAD = {"base": 0.617, "oracle": 0.825, "lexical": 0.419, "nodot": 0.092};
const LEN = {"corr": 0.92, "median": 199, "rows": [{"name": "cosine (normalised)", "recall": 0.617, "len": 250}, {"name": "raw dot product", "recall": 0.092, "len": 1494}]};
const CURVE = [{"dims": 256, "recall": 0.617, "bytes": 1024, "gb": 0.95}, {"dims": 128, "recall": 0.678, "bytes": 512, "gb": 0.48}, {"dims": 64, "recall": 0.71, "bytes": 256, "gb": 0.24}, {"dims": 32, "recall": 0.73, "bytes": 128, "gb": 0.12}, {"dims": 16, "recall": 0.732, "bytes": 64, "gb": 0.06}, {"dims": 8, "recall": 0.672, "bytes": 32, "gb": 0.03}, {"dims": 4, "recall": 0.42, "bytes": 16, "gb": 0.01}];
const BEST = {"dims": 16, "recall": 0.732, "bytes": 64, "gb": 0.06};
const QUANT = {"full": [{"name": "float32", "bytes": 1024, "recall": 0.617}, {"name": "int8", "bytes": 256, "recall": 0.618}, {"name": "binary (1 bit)", "bytes": 32.0, "recall": 0.032}, {"name": "binary + rescore top 40", "bytes": 34.0, "recall": 0.086}], "trunc": [{"name": "float32", "bytes": 64, "recall": 0.732}, {"name": "int8", "bytes": 16, "recall": 0.732}, {"name": "binary (1 bit)", "bytes": 2.0, "recall": 0.429}, {"name": "binary + rescore top 40", "bytes": 2.1, "recall": 0.522}]};
const ASYM = [{"name": "both sides normalised (cosine)", "recall": 0.617}, {"name": "query left unnormalised", "recall": 0.617}, {"name": "documents left unnormalised", "recall": 0.092}, {"name": "query encoded without the idf weights", "recall": 0.629}, {"name": "query encoded with an idf of its own", "recall": 0.609}, {"name": "query with the idf applied twice", "recall": 0.588}];
const FUSE = [{"name": "vector score only", "recall": 0.617}, {"name": "lexical score only (on the same shortlist)", "recall": 0.25}, {"name": "reciprocal rank fusion (k=60)", "recall": 0.457}, {"name": "weighted 80% / 20%", "recall": 0.629}];
const CHUNK = {"cfg": {"docs": 4000, "sections": 5}, "rows": [{"name": "one vector for the whole document", "recall": 0.1, "vecs": 1, "bytes": 512}, {"name": "one vector per section (5), score = best section", "recall": 0.584, "vecs": 5, "bytes": 2560}]};
const SCAN = [{"docs": 10000, "ms": 0.6, "gb": 0.01, "qps": 1670}, {"docs": 100000, "ms": 6.71, "gb": 0.1, "qps": 149}, {"docs": 400000, "ms": 28.99, "gb": 0.38, "qps": 35}];
const DUP = {"copies": 3, "slots": 3};

/* ---------------------------------------------------------------- shared */
const R3 = x => x.toFixed(3);
function barRows(sh, rows, opt){
  // rows: [{label, value, max, right, sty}]
  const x0 = opt.x0 || 3.30, w = opt.w || 3.4, y0 = opt.y0 || 1.55, dy = opt.dy || 0.95;
  rows.forEach((r, i) => {
    const y = y0 + i * dy;
    sh.push(fitT(x0 - .15, y + .33, r.label, {anchor:'end', c:COL.tealL, fs:opt.fs || .27}));
    const bw = Math.max(.12, w * r.value / r.max);
    sh.push(S.r(x0, y, bw, .60, r.sty || 'ok', null));
    sh.push(fitT(x0 + bw + .14, y + .33, r.right, {anchor:'start',
            c:r.sty === 'bad' ? COL.red : COL.tealL, fs:opt.fs || .27}));
  });
  return sh;
}

/* ================================================================ tab 1: a direction */
const CODE_DIR = [
  '# an embedding is a direction; retrieval is "whose is closest"',
  'q = encode(query)              # one vector',
  'D = encode(documents)          # one vector each',
  '',
  'scores = unit(q) @ unit(D).T   # cosine = dot product of unit vectors',
  'top10  = argsort(-scores)[:10]'
];
function dirFrames(){
  const F = new Frames();
  const axes = (sh) => {
    sh.push(S.e(1.10, 4.70, 4.60, 4.70, {s:'soft', w:.05, arrow:false}));
    sh.push(S.e(1.10, 4.70, 1.10, 1.30, {s:'soft', w:.05, arrow:false}));
    return sh;
  };
  const vec = (sh, x, y, lab, st) => {
    sh.push(S.e(1.10, 4.70, x, y, {s:st, w:.07}));
    sh.push(fitT(x + .12, y - .12, lab, {anchor:'start', c:st === 'hot' ? COL.orangeL : COL.tealL, fs:.26}));
    return sh;
  };
  let sh = axes([]);
  vec(sh, 3.60, 2.10, 'document A', 'ok');
  vec(sh, 4.10, 3.30, 'document B (same topic, longer)', 'ok');
  vec(sh, 2.90, 1.70, 'query', 'hot');
  side(sh, 5.60, 1.70, [
    [{zh:'文件 A 與 query 的夾角最小', en:'document A sits at the smallest angle'}, COL.pale, .28],
    [{zh:'文件 B 的「長度」最大', en:'document B is simply the longest vector'}, COL.pale, .28],
    [{zh:'cosine 問角度', en:'cosine asks the angle'}, COL.grey, .26],
    [{zh:'dot 問角度 × 長度', en:'a dot product asks angle x length'}, COL.grey, .26]], .58);
  F.push({shapes:caps(sh, {cap:{zh:'一份文件＝空間裡的一個方向', en:'a document is a direction in space'}}),
          line:4, view:VIEW,
          panels:[{lbl:'documents', chips:[chip(CFG.docs.toLocaleString(), 'dim')]},
                  {lbl:'planted answers', chips:[chip(CFG.targets + ' per query', 'act')]}],
          msg:{zh:'這一天的語料是**自己種答案進去的**：' + CFG.docs.toLocaleString() + ' 份文件、' + CFG.queries + ' 個查詢，每個查詢有 ' + CFG.targets + ' 份被種進去的正確答案。所以 recall@10 是一個事實，不是感覺——後面每一個決定都可以直接量。',
               en:'The corpus for this day has its **answers planted in it**: ' + CFG.docs.toLocaleString() + ' documents, ' + CFG.queries + ' queries, and ' + CFG.targets + ' known-correct documents for each query. That makes recall@10 a fact rather than a judgement, so every decision below can simply be measured.'}});
  const sh2 = [];
  barRows(sh2, [
    {label:{zh:'字詞比對（tf-idf cosine）', en:'term matching (tf-idf cosine)'}, value:HEAD.lexical, max:1, right:R3(HEAD.lexical), sty:'idle'},
    {label:{zh:'我們的 embedding（' + CFG.dims + ' 維）', en:'our embedding (' + CFG.dims + ' dims)'}, value:HEAD.base, max:1, right:R3(HEAD.base), sty:'ok'},
    {label:{zh:'答案本人（天花板）', en:'the answer key (the ceiling)'}, value:HEAD.oracle, max:1, right:R3(HEAD.oracle), sty:'act'}],
    {y0:1.85, dy:1.15, x0:4.10, w:3.0, fs:.26});
  F.push({shapes:caps(sh2, {cap:{zh:'recall@10：三個數字先把期待值定住',
                                 en:'recall@10: three numbers to set expectations'}}),
          line:5, view:VIEW,
          panels:[{lbl:'embedding', chips:[chip(R3(HEAD.base), 'ok')]},
                  {lbl:'ceiling', chips:[chip(R3(HEAD.oracle), 'act')]}],
          msg:{zh:'**天花板不是 1.0 而是 ' + R3(HEAD.oracle) + '**：連「照著語料被產生的方式去排序」都找不回全部的答案，因為跟查詢同主題的背景文件本來就會擠進前十。所以看 recall 要先知道這個語料的上限在哪；一個 0.62 的系統離滿分很遠，但離可能的極限只差 0.21。',
               en:'**The ceiling is ' + R3(HEAD.oracle) + ', not 1.0**: even ranking by the exact recipe the corpus was generated from cannot recover every planted answer, because background documents on the same topic crowd into the top ten. Always find that ceiling first - a system at 0.62 looks far from perfect and is only 0.21 from the best anyone could do here.'}});
  return F.list;
}

/* ================================================================ tab 2: length */
const CODE_LEN = [
  'scores = q @ D.T              # raw dot product: angle x length',
  'scores = unit(q) @ unit(D).T  # cosine: angle only',
  '',
  '# sentence-transformers:  model.encode(..., normalize_embeddings=True)',
  '# faiss:                  faiss.normalize_L2(vectors)'
];
function lenFrames(){
  const F = new Frames();
  const mk = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'沒正規化的話，排序排的是「長度」',
                            en:'without normalisation the ranking is by length'},
                 {c:COL.tealL, fs:.31}));
    const top = Math.max.apply(null, LEN.rows.map(r => r.len));
    LEN.rows.forEach((r, i) => {
      if (i > upto) return;
      const y = 1.75 + i * 1.45;
      sh.push(fitT(3.05, y + .33, r.name, {anchor:'end', c:COL.tealL, fs:.27}));
      const w = Math.max(.2, 2.9 * r.len / top);
      sh.push(S.r(3.20, y, w, .62, i === 1 ? 'bad' : 'ok', null));
      sh.push(fitT(3.32 + w, y + .33, r.len + ' ' + (TRW), {anchor:'start',
              c:i === 1 ? COL.red : COL.tealL, fs:.26}));
      sh.push(fitT(9.5, y + .33, 'avg length of the top 10  |  recall@10 ' + R3(r.recall),
                   {anchor:'end', c:COL.grey, fs:.24}));
    });
    sh.push(fitT(.45, 4.95, {zh:'文件的中位數長度是 ' + LEN.median + ' 個字',
                             en:'the median document is ' + LEN.median + ' words long'},
                 {anchor:'start', c:COL.grey, fs:.26}));
    return sh;
  };
  const TRW = 'words';
  F.push({shapes:caps(mk(0), {cap:{zh:'cosine：只問角度', en:'cosine: the angle only'}}),
          line:1, view:VIEW,
          panels:[{lbl:'cosine', chips:[chip(R3(LEN.rows[0].recall), 'ok')]}],
          msg:{zh:'向量的長度跟文件有幾個字的相關係數是 **' + LEN.corr + '**——長文件的向量就是比較長。cosine 把這個長度除掉，只留方向，所以它問的是「講的是不是同一件事」。',
               en:'The length of a document vector correlates **' + LEN.corr + '** with how many words the document has - long documents simply produce long vectors. Cosine divides that out and keeps only the direction, so it asks whether two things are about the same subject.'}});
  F.push({shapes:caps(mk(1), {cap:{zh:'raw dot product：角度 × 長度',
                                   en:'a raw dot product: angle x length'}}),
          line:0, view:VIEW,
          panels:[{lbl:'cosine', chips:[chip(R3(LEN.rows[0].recall), 'ok')]},
                  {lbl:'raw dot', chips:[chip(R3(LEN.rows[1].recall), 'bad')]}],
          msg:{zh:'忘了正規化之後，前十名的平均長度從 ' + LEN.rows[0].len + ' 個字變成 ' + LEN.rows[1].len + ' 個字（中位數才 ' + LEN.median + '），recall 從 ' + R3(LEN.rows[0].recall) + ' 掉到 ' + R3(LEN.rows[1].recall) + '。**索引沒有壞掉，它只是改成照長度排名**——而且不會有任何錯誤訊息。這就是每個函式庫都有一個 normalize 旗標、而每個人都只會忘記一次的原因。',
               en:'Forget to normalise and the average hit goes from ' + LEN.rows[0].len + ' words to ' + LEN.rows[1].len + ' (the median document is ' + LEN.median + '), with recall falling from ' + R3(LEN.rows[0].recall) + ' to ' + R3(LEN.rows[1].recall) + '. **The index is not broken; it is now ranking by length** - and nothing anywhere reports an error. Hence the normalize flag in every library, and the fact that people forget it exactly once.'}});
  return F.list;
}

/* ================================================================ tab 3: build it */
const CODE_BUILD = [
  'counts = word counts, one row per document',
  'X      = log1p(counts) * idf        # down-weight common words',
  'U,S,Vt = svd(X - X.mean(0))         # the directions that matter',
  'vec    = X @ Vt[:dims].T            # project into `dims` numbers',
  '',
  '# a trained encoder swaps the SVD for a contrastive objective;',
  '# the shape of the result - one dense direction - is the same'
];
function buildFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    const steps = [[{zh:'字數矩陣', en:'word counts'}, {zh:'每份文件一列', en:'one row per document'}],
                   [{zh:'tf-idf', en:'tf-idf'}, {zh:'常見字降權', en:'common words count less'}],
                   [{zh:'SVD', en:'SVD'}, {zh:'找出重要的方向', en:'the directions that matter'}],
                   [{zh:'投影', en:'project'}, {zh:'每份 ' + CFG.dims + ' 個數字', en:CFG.dims + ' numbers each'}]];
    steps.forEach((st, i) => {
      if (i > upto) return;
      const x = .45 + i * 2.25;
      sh.push(S.r(x, 1.45, 1.95, .80, i === upto ? 'hot' : 'ok', null));
      sh.push(fitT(x + .97, 1.88, st[0], {c:COL.pale, fs:.30}));
      sh.push(fitT(x + .97, 2.55, st[1], {c:COL.grey, fs:.23}));
      if (i < 3 && i < upto) sh.push(S.e(x + 2.00, 1.85, x + 2.22, 1.85, {s:'ok', w:.05}));
    });
    barRows(sh, [
      {label:{zh:'tf-idf cosine（' + CFG.vocab + ' 維稀疏）', en:'tf-idf cosine (' + CFG.vocab + ' sparse dims)'},
       value:HEAD.lexical, max:1, right:R3(HEAD.lexical), sty:'idle'},
      {label:{zh:'投影到 ' + CFG.dims + ' 維', en:'projected to ' + CFG.dims + ' dims'},
       value:HEAD.base, max:1, right:R3(HEAD.base), sty:'ok'},
      {label:{zh:'答案本人', en:'the answer key'}, value:HEAD.oracle, max:1, right:R3(HEAD.oracle), sty:'act'}],
      {y0:3.30, dy:.85, x0:4.30, w:3.0});
    return sh;
  };
  [0, 2, 3].forEach((u, i) => {
    F.push({shapes:caps(stage(u), {cap:{zh:'二十行就有一個 embedding',
                                        en:'an embedding in twenty lines'}}),
            line:[0, 2, 3][i], view:VIEW,
            panels:[{lbl:'lexical', chips:[chip(R3(HEAD.lexical), 'dim')]},
                    {lbl:'embedding', chips:[chip(R3(HEAD.base), 'ok')]}],
            msg:[
              {zh:'先從最笨的做法開始：把每份文件當成一袋字，用 tf-idf 比較重疊。它拿到 ' + R3(HEAD.lexical) + '——會動，但只要查詢用的字跟文件不同就瞎了。',
               en:'Start with the dumb version: treat each document as a bag of words and compare overlap with tf-idf. That scores ' + R3(HEAD.lexical) + ' - it works, and it is blind the moment the query uses different words from the document.'},
              {zh:'SVD 做的事情是**找出「常常一起出現的字」構成的方向**，而不是把字一個一個比對。這一步就是「語意」的來源：不同的字只要出現在同樣的脈絡裡，就會被投影到附近。',
               en:'The SVD finds **directions made of words that keep appearing together**, instead of comparing words one by one. That step is where the "semantic" part comes from: different words that live in the same contexts end up projected close to each other.'},
              {zh:'投影到 ' + CFG.dims + ' 個數字之後 recall 從 ' + R3(HEAD.lexical) + ' 變成 ' + R3(HEAD.base) + '，而天花板是 ' + R3(HEAD.oracle) + '。**真正的 encoder 只是把 SVD 換成對比學習**（告訴模型哪兩段文字該靠近），產出的東西形狀一模一樣：每份文件一個稠密方向。所以後面關於維度、位元數、幾個向量的結論，在真模型上同樣成立。',
               en:'Projected into ' + CFG.dims + ' numbers, recall goes from ' + R3(HEAD.lexical) + ' to ' + R3(HEAD.base) + ', against a ceiling of ' + R3(HEAD.oracle) + '. **A real encoder only swaps the SVD for a contrastive objective** - telling the model which pairs belong together - and the output has exactly the same shape: one dense direction per document. Which is why everything below about dimensions, bits and vector counts carries over.'}
            ][i]});
  });
  return F.list;
}

/* ================================================================ tab 4: dimensions */
const CODE_DIMS = [
  '# the first d columns are already the best d-dimensional version',
  'small = vectors[:, :d]          # SVD gives nested dimensions for free',
  'small = unit(small)             # re-normalise after truncating',
  '',
  '# a trained model does this on purpose (Matryoshka):',
  '# SentenceTransformer(..., truncate_dim=256)'
];
function dimsFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'維度不是越多越好', en:'more dimensions is not more recall'},
                 {c:COL.tealL, fs:.31}));
    const lo = .30, hi = .80, x0 = 1.30, y0 = 4.80, w = 7.6, h = 3.1;
    for (let i = 0; i <= 4; i++){
      const v = lo + (hi - lo) * i / 4;
      const y = y0 - h * (v - lo) / (hi - lo);
      sh.push(S.e(x0, y, x0 + w, y, {s:'ghost', w:.03, arrow:false}));
      sh.push(fitT(x0 - .15, y + .08, v.toFixed(2), {anchor:'end', c:COL.grey, fs:.22}));
    }
    let prev = null;
    CURVE.forEach((r, i) => {
      const px = x0 + w * i / (CURVE.length - 1);
      const py = y0 - h * (r.recall - lo) / (hi - lo);
      if (prev && i <= upto) sh.push(S.e(prev[0], prev[1], px, py, {s:'ok', w:.055, arrow:false}));
      if (i <= upto){
        const hot = r.dims === BEST.dims;
        sh.push(S.c(px, py, hot ? .13 : .09, hot ? 'hot' : 'ok', null));
        if (hot) sh.push(fitT(px, py - .34, R3(r.recall), {c:COL.orangeL, fs:.26}));
      }
      sh.push(fitT(px, y0 + .34, String(r.dims), {c:COL.grey, fs:.24}));
      sh.push(fitT(px, y0 + .68, r.bytes + ' B', {c:COL.grey, fs:.22}));
      prev = [px, py];
    });
    sh.push(fitT(x0, 5.70, {zh:'維度（以及每份文件的位元組）',
                            en:'dimensions kept (and bytes per document)'},
                 {anchor:'start', c:COL.grey, fs:.24}));
    return sh;
  };
  [1, 4, CURVE.length - 1].forEach((u, i) => {
    F.push({shapes:caps(stage(u), {cap:{zh:'同一組向量，只是砍掉後面的維度',
                                        en:'the same vectors, with the tail cut off'}}),
            line:[1, 1, 2][i], view:VIEW,
            panels:[{lbl:CFG.dims + ' dims', chips:[chip(R3(CURVE[0].recall), 'dim')]},
                    {lbl:BEST.dims + ' dims', chips:[chip(R3(BEST.recall), 'ok'), chip(BEST.bytes + ' B', 'act')]}],
            msg:[
              {zh:'SVD 的欄位是**排好序的**：前 d 欄本來就是「最好的 d 維版本」，所以截斷不需要重新訓練，直接切就好（記得切完要重新正規化）。',
               en:'The columns of an SVD are **already ordered**: the first d of them are the best d-dimensional version of the same embedding, so truncation needs no retraining - just slice, and re-normalise afterwards.'},
              {zh:'砍到一半、四分之一，recall 不但沒掉還上升。原因是**尾巴那些方向是雜訊**，而雜訊也有投票權。',
               en:'Cut it in half, then to a quarter, and recall does not fall - it rises. The reason is that **the tail directions are noise**, and noise still gets a vote.'},
              {zh:'峰值在 ' + BEST.dims + ' 維：' + R3(BEST.recall) + ' vs ' + CFG.dims + ' 維的 ' + R3(CURVE[0].recall) + '，而記憶體是 ' + BEST.bytes + ' B vs ' + CURVE[0].bytes + ' B（每百萬份文件 ' + BEST.gb + ' GB vs ' + CURVE[0].gb + ' GB）。真正訓練過的 Matryoshka encoder 曲線是「先平再掉」而不是像這裡先升，但實務結論一樣：**維度是一個超參數，不是一種美德**，而且看到向量帳單先試截斷。',
               en:'The peak is at ' + BEST.dims + ' dimensions: ' + R3(BEST.recall) + ' against ' + R3(CURVE[0].recall) + ' at ' + CFG.dims + ', for ' + BEST.bytes + ' bytes per document instead of ' + CURVE[0].bytes + ' (' + BEST.gb + ' GB per million against ' + CURVE[0].gb + '). A trained Matryoshka encoder gives a curve that is flat and then falls rather than rising like this one, but the practical conclusion is identical: **the dimension count is a hyperparameter, not a virtue**, and truncation is the first thing to try on a vector bill.'}
            ][i]});
  });
  return F.list;
}

/* ================================================================ tab 5: bits */
const CODE_BITS = [
  'scale = abs(unit(all_docs)).max()      # ONE scale for the whole corpus',
  'q8    = round(unit(v) / scale * 127)   # int8: the dot product survives',
  '',
  'bits  = unit(v) - corpus_mean > 0      # 1 bit per dimension',
  'cand  = hamming_search(bits, k * 4)    # cheap, approximate',
  'top_k = rescore(cand, full_vectors)    # decide with the dear ones'
];
function bitsFrames(){
  const F = new Frames();
  const stage = (rows, title) => {
    const sh = [];
    sh.push(fitT(4.9, .92, title, {c:COL.tealL, fs:.31}));
    const topB = Math.max.apply(null, rows.map(r => r.bytes));
    rows.forEach((r, i) => {
      const y = 1.60 + i * 1.05;
      sh.push(fitT(3.05, y + .31, r.name, {anchor:'end', c:COL.tealL, fs:.25}));
      const wb = Math.max(.12, 1.9 * r.bytes / topB);
      sh.push(S.r(3.20, y, wb, .58, 'idle', null));
      sh.push(fitT(3.30 + wb, y + .31, r.bytes + ' B', {anchor:'start', c:COL.grey, fs:.24}));
      const good = r.recall > .8 * rows[0].recall;
      const wr = Math.max(.12, 2.6 * r.recall / rows[0].recall);
      sh.push(S.r(6.30, y, wr, .58, good ? 'ok' : 'bad', null));
      sh.push(fitT(6.40 + wr, y + .31, R3(r.recall), {anchor:'start',
              c:good ? COL.tealL : COL.red, fs:.25}));
    });
    return sh;
  };
  F.push({shapes:caps(stage(QUANT.full, {zh:'每個數字用幾個 bit（' + CFG.dims + ' 維）',
                                         en:'bits per number, at ' + CFG.dims + ' dimensions'}),
          {cap:{zh:'左：每份文件的位元組；右：recall@10', en:'left: bytes per document; right: recall@10'}}),
          line:1, view:VIEW,
          panels:[{lbl:'float32', chips:[chip(R3(QUANT.full[0].recall), 'ok')]},
                  {lbl:'int8', chips:[chip(R3(QUANT.full[1].recall), 'ok')]},
                  {lbl:'1 bit', chips:[chip(R3(QUANT.full[2].recall), 'bad')]}],
          msg:{zh:'**int8 是免費的**：用「整個語料共用一個 scale」去量化，內積只多了捨入誤差，recall ' + R3(QUANT.full[0].recall) + ' → ' + R3(QUANT.full[1].recall) + '，記憶體剩四分之一。**但每維一個 bit 在這裡是災難**（' + R3(QUANT.full[2].recall) + '）：這個 embedding 把資訊都塞在前幾維，而符號位元把所有維度當成一樣重要。',
               en:'**int8 is free**: quantize with one scale for the whole corpus and the dot product only picks up rounding error - recall ' + R3(QUANT.full[0].recall) + ' to ' + R3(QUANT.full[1].recall) + ' at a quarter of the memory. **One bit per dimension is a disaster here** (' + R3(QUANT.full[2].recall) + '), because this embedding keeps its information in the first few dimensions and a sign bit treats every dimension as equally important.'}});
  F.push({shapes:caps(stage(QUANT.trunc, {zh:'先截到 ' + BEST.dims + ' 維，再量化',
                                          en:'truncate to ' + BEST.dims + ' dimensions first, then quantize'}),
          {cap:{zh:'rescore＝用完整向量重排候選名單',
                en:'rescore: re-rank the shortlist with the full vectors'}}),
          line:5, view:VIEW,
          panels:[{lbl:'1 bit', chips:[chip(R3(QUANT.trunc[2].recall), 'bad')]},
                  {lbl:'+ rescore', chips:[chip(R3(QUANT.trunc[3].recall), 'ok')]},
                  {lbl:'bytes', chips:[chip(QUANT.trunc[2].bytes + ' B', 'act')]}],
          msg:{zh:'先把維度砍到資訊真的在的地方，1 bit 就從 ' + R3(QUANT.full[2].recall) + ' 變成 ' + R3(QUANT.trunc[2].recall) + '；再把 binary 找出的前 40 名用完整向量重排一次（rescore），回到 ' + R3(QUANT.trunc[3].recall) + '，而每份文件只要 ' + QUANT.trunc[2].bytes + ' 個位元組。**標準配方就是這個：用便宜的表示法搜尋，用貴的表示法決定。**',
               en:'Truncate first, down to where the information actually is, and one bit goes from ' + R3(QUANT.full[2].recall) + ' to ' + R3(QUANT.trunc[2].recall) + '. Then re-rank the binary shortlist of 40 with the full vectors and it reaches ' + R3(QUANT.trunc[3].recall) + ' while storing ' + QUANT.trunc[2].bytes + ' bytes per document. **That is the standard recipe: search in the cheap representation, decide in the dear one.**'}});
  return F.list;
}

/* ================================================================ tab 6: chunking */
const CODE_CHUNK = [
  '# one vector per document',
  'vec = encode(whole_document)         # the average of its sections',
  '',
  '# one vector per section, score = the best section',
  'vecs  = [encode(s) for s in sections(document)]',
  'score = max(unit(q) @ unit(v) for v in vecs)'
];
function chunkFrames(){
  const F = new Frames();
  const stage = (mode) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'一份講五件事的文件，沒有單一方向',
                            en:'a document about five things has no single direction'},
                 {c:COL.tealL, fs:.31}));
    const S5 = CHUNK.cfg.sections;
    for (let i = 0; i < S5; i++){
      const x = 1.30 + i * 1.45;
      sh.push(S.r(x, 1.45, 1.30, .72, i === 2 ? 'hot' : 'idle',
                  i === 2 ? 'answer' : 'other', {fs:.25}));
    }
    sh.push(fitT(4.9, 2.55, {zh:'一份文件的五個段落', en:'the five sections of one document'},
                 {c:COL.grey, fs:.25}));
    if (mode === 0){
      sh.push(S.r(3.95, 3.15, 2.0, .70, 'bad', 'one vector', {fs:.28}));
      sh.push(S.e(4.95, 2.30, 4.95, 3.10, {s:'bad', w:.05}));
      sh.push(fitT(4.9, 4.30, {zh:'平均之後，答案那一段被稀釋掉了',
                               en:'averaged, the section that answers is diluted away'},
                   {c:COL.red, fs:.28}));
    } else {
      for (let i = 0; i < S5; i++){
        const x = 1.30 + i * 1.45;
        sh.push(S.r(x + .25, 3.15, .80, .60, i === 2 ? 'ok' : 'idle', null));
        sh.push(S.e(x + .65, 2.30, x + .65, 3.10, {s:i === 2 ? 'ok' : 'ghost', w:.04}));
      }
      sh.push(fitT(4.9, 4.30, {zh:'每段一個向量，分數取最好的那一段',
                               en:'one vector per section, scored by the best one'},
                   {c:COL.tealL, fs:.28}));
    }
    const r = CHUNK.rows[mode];
    sh.push(fitT(4.9, 5.05, 'recall@10 ' + R3(r.recall) + '    ' + r.vecs +
                 ' vector(s)/doc    ' + r.bytes + ' B/doc',
                 {c:mode === 0 ? COL.red : COL.tealL, fs:.30}));
    return sh;
  };
  F.push({shapes:caps(stage(0), {cap:{zh:'整份文件一個向量', en:'one vector for the whole document'}}),
          line:1, view:VIEW,
          panels:[{lbl:'recall@10', chips:[chip(R3(CHUNK.rows[0].recall), 'bad')]}],
          msg:{zh:'' + CHUNK.cfg.docs.toLocaleString() + ' 份文件，每份由 ' + CHUNK.cfg.sections + ' 個不同主題的段落組成，答案只在其中一段；其他文件則是整份都「有點像」。整份文件一個向量的 recall 只有 ' + R3(CHUNK.rows[0].recall) + '——**五個方向的平均指向其中任何一個都不是**。',
               en:'' + CHUNK.cfg.docs.toLocaleString() + ' documents, each made of ' + CHUNK.cfg.sections + ' sections about different things, with the answer in exactly one of them; the other documents are vaguely on-topic all the way through. One vector for the whole document reaches ' + R3(CHUNK.rows[0].recall) + ' - **the average of five directions points at none of them**.'}});
  F.push({shapes:caps(stage(1), {cap:{zh:'每段一個向量', en:'one vector per section'}}),
          line:4, view:VIEW,
          panels:[{lbl:'whole doc', chips:[chip(R3(CHUNK.rows[0].recall), 'bad')]},
                  {lbl:'per section', chips:[chip(R3(CHUNK.rows[1].recall), 'ok')]},
                  {lbl:'cost', chips:[chip(CHUNK.rows[1].vecs + 'x vectors', 'act')]}],
          msg:{zh:'改成每段一個向量、分數取最好的那一段：recall ' + R3(CHUNK.rows[0].recall) + ' → ' + R3(CHUNK.rows[1].recall) + '（' + (CHUNK.rows[1].recall / CHUNK.rows[0].recall).toFixed(1) + ' 倍），代價是 ' + CHUNK.rows[1].vecs + ' 倍的向量數。**所以「一份文件一個向量」其實是一個假設：這份文件只講一件事。** chunking 就是在修這個假設，而 late-interaction 模型是把它推到極限（每個 span 甚至每個 token 一個向量）。',
               en:'Switch to one vector per section, scored by the best section, and recall goes ' + R3(CHUNK.rows[0].recall) + ' to ' + R3(CHUNK.rows[1].recall) + ' - ' + (CHUNK.rows[1].recall / CHUNK.rows[0].recall).toFixed(1) + 'x - for ' + CHUNK.rows[1].vecs + 'x the vectors. **So "one vector per document" is an assumption: that the document is about one thing.** Chunking is how everyone repairs it, and late-interaction models take the same idea to its limit, one span or even one token at a time.'}});
  return F.list;
}

/* ================================================================ tab 7: second score */
const CODE_FUSE = [
  '# reciprocal rank fusion: trust both lists equally',
  'score = 1/(60 + rank_vector) + 1/(60 + rank_lexical)',
  '',
  '# weighted: trust one of them more (measure the weight!)',
  'score = 0.8 * z(vector) + 0.2 * z(lexical)'
];
function fuseFrames(){
  const F = new Frames();
  const stage = (upto) => {
    const sh = [];
    sh.push(fitT(4.9, .92, {zh:'第二個分數不會自動讓結果變好',
                            en:'a second score is not automatically better'},
                 {c:COL.tealL, fs:.31}));
    const top = Math.max.apply(null, FUSE.map(r => r.recall));
    FUSE.forEach((r, i) => {
      if (i > upto) return;
      const y = 1.70 + i * 1.05;
      sh.push(fitT(4.05, y + .31, r.name.replace(' (on the same shortlist)', ''),
                   {anchor:'end', c:COL.tealL, fs:.25}));
      const w = Math.max(.12, 3.6 * r.recall / top);
      const bad = r.recall < FUSE[0].recall - .02;
      sh.push(S.r(4.20, y, w, .58, bad ? 'bad' : (i === 0 ? 'act' : 'ok'), null));
      sh.push(fitT(4.32 + w, y + .31, R3(r.recall), {anchor:'start',
              c:bad ? COL.red : COL.tealL, fs:.26}));
    });
    return sh;
  };
  F.push({shapes:caps(stage(1), {cap:{zh:'同一份候選名單，兩種分數',
                                      en:'the same shortlist, two different scores'}}),
          line:0, view:VIEW,
          panels:[{lbl:'vector', chips:[chip(R3(FUSE[0].recall), 'act')]},
                  {lbl:'lexical', chips:[chip(R3(FUSE[1].recall), 'bad')]}],
          msg:{zh:'在這個語料裡，查詢要的是「同時講兩個主題」的文件，而字詞重疊看不出這件事，所以純詞彙分數只有 ' + R3(FUSE[1].recall) + '，明顯比向量分數弱。',
               en:'In this corpus a query asks for documents about two topics at once, and word overlap cannot see that, so the purely lexical score reaches only ' + R3(FUSE[1].recall) + ' - clearly the weaker of the two.'}});
  F.push({shapes:caps(stage(3), {cap:{zh:'把弱的那個融進來會發生什麼事',
                                      en:'what happens when the weaker one is fused in'}}),
          line:1, view:VIEW,
          panels:[{lbl:'RRF', chips:[chip(R3(FUSE[2].recall), 'bad')]},
                  {lbl:'80/20', chips:[chip(R3(FUSE[3].recall), 'ok')]}],
          msg:{zh:'**RRF（倒數排名融合）預設兩張名單一樣可信**，於是 recall 從 ' + R3(FUSE[0].recall) + ' 掉到 ' + R3(FUSE[2].recall) + '；把權重壓到兩成才回到原點（' + R3(FUSE[3].recall) + '）。規則因此很清楚：**第二個訊號要在第一個錯的地方才有用，而融合權重是量出來的、不是預設值。**',
               en:'**Reciprocal rank fusion assumes both lists deserve equal trust**, and recall falls from ' + R3(FUSE[0].recall) + ' to ' + R3(FUSE[2].recall) + '; weighting the weaker score down to 20% merely gets back to where we started (' + R3(FUSE[3].recall) + '). The rule that leaves: **a second signal helps where the first is wrong, and the fusion weight is a measurement, not a default.**'}});
  return F.list;
}

/* ================================================================ tab 8: traps */
const CODE_TRAPS = [
  '# 1. the query side can be encoded differently by accident',
  'q = encode(query)                 # same model? same prefix? same idf?',
  '',
  '# 2. near-duplicates do not split the top-k, they take it',
  '# 3. a full scan is linear: fine at 10k, seconds at 10M',
  'scores = unit(q) @ unit(D).T      # one matmul over every document'
];
function trapFrames(){
  const F = new Frames();
  const sh = [];
  sh.push(fitT(4.9, .92, {zh:'兩邊不是對稱的', en:'the two sides are not symmetric'},
               {c:COL.tealL, fs:.31}));
  ASYM.forEach((r, i) => {
    const y = 1.45 + i * .78;
    sh.push(fitT(5.15, y + .26, r.name, {anchor:'end', c:COL.tealL, fs:.25}));
    const w = Math.max(.12, 2.8 * r.recall / ASYM[0].recall);
    const bad = r.recall < ASYM[0].recall - .05;
    sh.push(S.r(5.30, y, w, .50, bad ? 'bad' : (i === 0 ? 'act' : 'ok'), null));
    sh.push(fitT(5.42 + w, y + .26, R3(r.recall), {anchor:'start',
            c:bad ? COL.red : COL.tealL, fs:.24}));
  });
  F.push({shapes:caps(sh, {cap:{zh:'query 正規化不會改變排序，document 會',
                                en:'normalising the query cannot change the ranking; the documents can'}}),
          line:1, view:VIEW,
          panels:[{lbl:'query norm', chips:[chip('no effect', 'dim')]},
                  {lbl:'doc norm', chips:[chip(R3(ASYM[2].recall), 'bad')]}],
          msg:{zh:'第一件事是精確的：**把 query 正規化不可能改變排名**，因為同時縮放一邊只是把所有分數乘上同一個數。真正重要的是文件那一邊。後面三列則是「兩邊用不同方式編碼」——沒有 idf、自己的 idf、idf 用兩次——結果**上下都會跑**，而且完全沒有任何錯誤。真模型是用 query prefix 與 document prefix 訓練的，所以兩邊真的是不同的函式，弄反了就是這種無聲的位移。',
               en:'The first row is exact: **normalising the query cannot change the ranking**, because scaling one side multiplies every score by the same number. The document side is the one that matters. The last three rows encode the query differently from the documents - no idf, its own idf, the idf applied twice - and the answer moves **in both directions** with no error anywhere. Real encoders are trained with a query prefix and a document prefix, so the two sides genuinely are different functions, and swapping them is exactly this kind of silent shift.'}});

  const sh2 = [];
  sh2.push(fitT(4.9, .92, {zh:'另外兩個會安靜咬人的地方',
                           en:'two more things that bite quietly'}, {c:COL.tealL, fs:.31}));
  sh2.push(fitT(.45, 1.55, {zh:'near-duplicate：複製品不會分票，是直接佔位',
                            en:'near-duplicates do not split the vote, they take the seats'},
               {anchor:'start', c:COL.pale, fs:.28}));
  for (let i = 0; i < 10; i++){
    const dup = i < DUP.slots;
    sh2.push(S.r(.60 + i * .90, 1.85, .78, .62, dup ? 'bad' : 'ok',
                 dup ? 'copy' : 'doc', {fs:.24}));
  }
  sh2.push(fitT(.45, 2.85, {zh:'加 ' + DUP.copies + ' 份幾乎一樣的文件，就吃掉那個查詢前十名的 ' + DUP.slots + ' 格',
                            en:'adding ' + DUP.copies + ' near-identical copies takes ' + DUP.slots + ' of that query\'s ten slots'},
               {anchor:'start', c:COL.red, fs:.26}));
  sh2.push(fitT(.45, 3.70, {zh:'暴力掃描是線性的', en:'a full scan is linear'},
               {anchor:'start', c:COL.pale, fs:.28}));
  SCAN.forEach((r, i) => {
    const y = 4.05 + i * .62;
    sh2.push(fitT(2.95, y + .24, r.docs.toLocaleString() + ' docs', {anchor:'end', c:COL.tealL, fs:.25}));
    const w = Math.max(.12, 3.4 * r.ms / SCAN[SCAN.length - 1].ms);
    sh2.push(S.r(3.10, y, w, .46, i === SCAN.length - 1 ? 'hot' : 'ok', null));
    sh2.push(fitT(3.22 + w, y + .24, r.ms + ' ms/query  (' + r.qps + '/s)',
                  {anchor:'start', c:COL.grey, fs:.24}));
  });
  F.push({shapes:caps(sh2, {cap:{zh:'索引之前要去重；而掃描遲早會不夠用',
                                 en:'deduplicate before indexing, and the scan runs out eventually'}}),
          line:2, view:VIEW,
          panels:[{lbl:'duplicates', chips:[chip(DUP.slots + '/10 slots', 'bad')]},
                  {lbl:SCAN[SCAN.length - 1].docs.toLocaleString(), chips:[chip(SCAN[SCAN.length - 1].ms + ' ms', 'hot')]}],
          msg:{zh:'近重複文件是檢索裡最常見的「結果看起來很好、其實只有一個答案」：' + DUP.copies + ' 份複製品直接吃掉前十的 ' + DUP.slots + ' 格，**要在索引前去重，不是在取回後**。而暴力掃描只是一次矩陣乘法，一萬份文件 ' + SCAN[0].ms + ' ms 完全夠用；但它是嚴格線性的，' + SCAN[SCAN.length - 1].docs.toLocaleString() + ' 份就要 ' + SCAN[SCAN.length - 1].ms + ' ms，再大就得換成「拒絕看大部分資料」的索引結構。',
               en:'Near-duplicates are the classic "the results look great and they are all the same answer": ' + DUP.copies + ' copies take ' + DUP.slots + ' of the ten slots, so **deduplicate before indexing, not after retrieving**. And the full scan is just one matrix multiply - ' + SCAN[0].ms + ' ms at ten thousand documents, which is plenty - but it is strictly linear, ' + SCAN[SCAN.length - 1].ms + ' ms at ' + SCAN[SCAN.length - 1].docs.toLocaleString() + ', and past that you need an index that refuses to look at most of the data.'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'Embedding 與向量檢索', en:'Embeddings and vector search'},
  sub: {zh:'一份文件是一個方向，檢索就是問「誰的方向最近」——而每個決定都有價錢',
        en:'a document is a direction, retrieval asks whose is closest, and every decision has a price'},
  tabs: [
    {id:'dir', label:{zh:'一個方向', en:'a direction'},
     stage:{zh:'cosine 問角度，dot 問角度×長度', en:'cosine asks the angle, a dot product asks angle x length'},
     view:VIEW, code:CODE_DIR, build:dirFrames,
     idea:{zh:'語料的答案是**種進去的**（' + CFG.docs.toLocaleString() + ' 份文件、每個查詢 ' + CFG.targets + ' 份正解），所以 recall@10 是事實。三個基準：字詞比對 ' + R3(HEAD.lexical) + '、我們的 embedding ' + R3(HEAD.base) + '、**天花板只有 ' + R3(HEAD.oracle) + '**。',
           en:'The answers are **planted** in this corpus (' + CFG.docs.toLocaleString() + ' documents, ' + CFG.targets + ' correct ones per query), so recall@10 is a fact. Three baselines: term matching ' + R3(HEAD.lexical) + ', our embedding ' + R3(HEAD.base) + ', and **a ceiling of only ' + R3(HEAD.oracle) + '**.'},
     legend:[['#3fe0dd', {zh:'文件向量', en:'document vectors'}],
             ['#ff9736', {zh:'查詢向量', en:'the query vector'}]]},
    {id:'len', label:{zh:'長度偏誤', en:'length bias'},
     stage:{zh:'忘了正規化＝照長度排名', en:'forget to normalise and you rank by length'},
     view:VIEW, code:CODE_LEN, build:lenFrames,
     idea:{zh:'向量長度跟文件字數的相關係數 **' + LEN.corr + '**，所以 raw dot product 會把長文件推上去：前十名平均 ' + LEN.rows[1].len + ' 字（中位數 ' + LEN.median + '），recall ' + R3(LEN.rows[0].recall) + ' → ' + R3(LEN.rows[1].recall) + '，**而且不會報錯**。',
           en:'Vector length correlates **' + LEN.corr + '** with document length, so a raw dot product promotes long documents: the top ten averages ' + LEN.rows[1].len + ' words against a median of ' + LEN.median + ', and recall falls ' + R3(LEN.rows[0].recall) + ' to ' + R3(LEN.rows[1].recall) + ' **with no error anywhere**.'},
     legend:[['#3fe0dd', {zh:'cosine', en:'cosine'}],
             ['#ff5c5c', {zh:'raw dot product', en:'the raw dot product'}]]},
    {id:'build', label:{zh:'向量哪來的', en:'where the vectors come from'},
     stage:{zh:'counts → tf-idf → SVD → 投影', en:'counts, tf-idf, SVD, project'},
     view:VIEW, code:CODE_BUILD, build:buildFrames,
     idea:{zh:'二十行就能做出一個 embedding：tf-idf 之後做 SVD，**用「常常一起出現的字」組成方向**，recall 從 ' + R3(HEAD.lexical) + ' 升到 ' + R3(HEAD.base) + '。真 encoder 只是把 SVD 換成對比學習，輸出形狀一樣。',
           en:'Twenty lines make an embedding: tf-idf, then an SVD that **builds directions out of words that co-occur**, taking recall from ' + R3(HEAD.lexical) + ' to ' + R3(HEAD.base) + '. A real encoder only swaps the SVD for a contrastive objective; the output has the same shape.'},
     legend:[['#ff9736', {zh:'目前這一步', en:'the current step'}],
             ['#3fe0dd', {zh:'已完成的步驟', en:'steps already done'}]]},
    {id:'dims', label:{zh:'留幾維', en:'how many dimensions'},
     stage:{zh:'截斷曲線與記憶體', en:'the truncation curve and the memory'},
     view:VIEW, code:CODE_DIMS, build:dimsFrames,
     idea:{zh:'SVD 的欄位本來就排好序，**前 d 欄就是最好的 d 維版本**。峰值在 ' + BEST.dims + ' 維（' + R3(BEST.recall) + '）而不是 ' + CFG.dims + ' 維（' + R3(CURVE[0].recall) + '）：尾巴是雜訊，而雜訊也有投票權。每百萬份文件 ' + BEST.gb + ' GB vs ' + CURVE[0].gb + ' GB。',
           en:'SVD columns come pre-sorted, so **the first d of them are the best d-dimensional version**. The peak is at ' + BEST.dims + ' dims (' + R3(BEST.recall) + ') rather than ' + CFG.dims + ' (' + R3(CURVE[0].recall) + '): the tail is noise, and noise still votes. ' + BEST.gb + ' GB per million documents against ' + CURVE[0].gb + '.'},
     legend:[['#ff9736', {zh:'最佳維度', en:'the best dimension count'}],
             ['#3fe0dd', {zh:'recall 曲線', en:'the recall curve'}]]},
    {id:'bits', label:{zh:'每個數字幾個 bit', en:'bits per number'},
     stage:{zh:'int8、1 bit、rescore', en:'int8, one bit, and rescoring'},
     view:VIEW, code:CODE_BITS, build:bitsFrames,
     idea:{zh:'**int8 免費**（' + R3(QUANT.full[0].recall) + ' → ' + R3(QUANT.full[1].recall) + '，記憶體 1/4），因為整個語料共用一個 scale，內積只多捨入誤差。**1 bit 不免費**；先截到 ' + BEST.dims + ' 維再量化、並用完整向量 rescore 前 40 名，可以用 ' + QUANT.trunc[2].bytes + ' B/doc 拿到 ' + R3(QUANT.trunc[3].recall) + '。',
           en:'**int8 is free** (' + R3(QUANT.full[0].recall) + ' to ' + R3(QUANT.full[1].recall) + ' at a quarter of the memory) because one corpus-wide scale leaves the dot product intact. **One bit is not**; truncate to ' + BEST.dims + ' dims first and re-rank the top 40 with full vectors and ' + QUANT.trunc[2].bytes + ' bytes per document reaches ' + R3(QUANT.trunc[3].recall) + '.'},
     legend:[['#3fe0dd', {zh:'recall 保住了', en:'recall kept'}],
             ['#ff5c5c', {zh:'recall 掉了', en:'recall lost'}]]},
    {id:'chunk', label:{zh:'一份文件幾個向量', en:'how many vectors per document'},
     stage:{zh:'整份 vs 每段', en:'the whole document against its sections'},
     view:VIEW, code:CODE_CHUNK, build:chunkFrames,
     idea:{zh:'**「一份文件一個向量」是一個假設：這份文件只講一件事。** 五段不同主題、答案只在其中一段時，整份一個向量只有 ' + R3(CHUNK.rows[0].recall) + '，每段一個向量是 ' + R3(CHUNK.rows[1].recall) + '（' + (CHUNK.rows[1].recall / CHUNK.rows[0].recall).toFixed(1) + ' 倍，代價 ' + CHUNK.rows[1].vecs + ' 倍向量）。',
           en:'**One vector per document is an assumption: that the document is about one thing.** With five sections and the answer in one of them, the whole-document vector reaches ' + R3(CHUNK.rows[0].recall) + ' and one vector per section reaches ' + R3(CHUNK.rows[1].recall) + ' - ' + (CHUNK.rows[1].recall / CHUNK.rows[0].recall).toFixed(1) + 'x, for ' + CHUNK.rows[1].vecs + 'x the vectors.'},
     legend:[['#ff9736', {zh:'有答案的那一段', en:'the section with the answer'}],
             ['#ff5c5c', {zh:'被平均稀釋掉', en:'diluted by the average'}]]},
    {id:'fuse', label:{zh:'第二個分數', en:'a second score'},
     stage:{zh:'融合不是免費的', en:'fusion is not free'},
     view:VIEW, code:CODE_FUSE, build:fuseFrames,
     idea:{zh:'這個語料的查詢要「同時命中兩個主題」，詞彙分數看不到，只有 ' + R3(FUSE[1].recall) + '。**RRF 預設兩張名單一樣可信**，於是 ' + R3(FUSE[0].recall) + ' → ' + R3(FUSE[2].recall) + '；壓到兩成權重才回到原點。',
           en:'Queries here need two topics at once, which the lexical score cannot see - it reaches ' + R3(FUSE[1].recall) + '. **RRF assumes both lists deserve equal trust**, so ' + R3(FUSE[0].recall) + ' becomes ' + R3(FUSE[2].recall) + '; weighting the weaker one down to 20% merely gets back to the start.'},
     legend:[['#9d6bff', {zh:'只有向量分數', en:'the vector score alone'}],
             ['#ff5c5c', {zh:'融合之後變差', en:'worse after fusing'}]]},
    {id:'traps', label:{zh:'安靜的陷阱', en:'the quiet traps'},
     stage:{zh:'不對稱、重複、線性掃描', en:'asymmetry, duplicates, the linear scan'},
     view:VIEW, code:CODE_TRAPS, build:trapFrames,
     idea:{zh:'**正規化 query 不會改變排序**（縮放一邊只是把所有分數乘同一個數），但兩邊用不同方式編碼會讓答案上下跑、而且無聲。另外：' + DUP.copies + ' 份近重複直接吃掉前十的 ' + DUP.slots + ' 格（要在索引前去重），以及暴力掃描在 ' + SCAN[SCAN.length - 1].docs.toLocaleString() + ' 份文件要 ' + SCAN[SCAN.length - 1].ms + ' ms。',
           en:'**Normalising the query cannot change the ranking** - scaling one side multiplies every score equally - but encoding the two sides differently moves the answer in both directions, silently. Also: ' + DUP.copies + ' near-duplicates take ' + DUP.slots + ' of the ten slots (deduplicate before indexing), and the full scan costs ' + SCAN[SCAN.length - 1].ms + ' ms at ' + SCAN[SCAN.length - 1].docs.toLocaleString() + ' documents.'},
     legend:[['#9d6bff', {zh:'基準做法', en:'the baseline'}],
             ['#ff5c5c', {zh:'安靜地變差', en:'quietly worse'}],
             ['#ff9736', {zh:'掃描的上限', en:'where the scan runs out'}]]}
  ]
};
