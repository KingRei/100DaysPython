// DAY: 60
// TITLE_ZH: Beam search 與 speculative decoding：一個改變輸出去找更好的句子，一個完全不改輸出只求更快
// TITLE_EN: Beam search and speculative decoding: one changes the output to find a better sentence, the other changes nothing and just goes faster
// SUB_ZH: 前一天量到 greedy 產生的是 1,250 句裡的第 44 名——逐步取最大值不等於整句最好，那是搜尋問題不是取樣問題。beam search 是那個搜尋；speculative decoding 則是相反的點子：輸出的分佈一個位元都不改，只用一個便宜的模型先猜，讓昂貴的模型一次驗證多個 token。
// SUB_EN: The previous day measured greedy at rank 44 of 1,250: best token per step is not best sentence, and that is a search problem rather than a sampling one. Beam search is the search. Speculative decoding is the opposite idea - do not change the output distribution at all, just let a cheap model guess ahead so the expensive one verifies several tokens in one pass.
// FOLDER: day%2060%20-%20beam%20search%20and%20speculative%20decoding
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
const FRAG = {"reqs": 240, "rejC": 12, "rejFrag": 12, "rejP": 5, "utilC": 65.7, "utilP": 68.5, "capacity": 16384};
const GT = {"n": 1250, "best": "a small dog sat in the garden.", "bestLp": -6.244, "greedy": "the red cat sang in the garden.", "greedyLp": -6.717, "rank": 44, "worst": "a cat slept on the mat.", "worstLp": -9.173};
const BEAM = [{"w": 1, "rank": 44, "lp": -6.717, "text": "the red cat sang in the garden."}, {"w": 2, "rank": 44, "lp": -6.717, "text": "the red cat sang in the garden."}, {"w": 4, "rank": 10, "lp": -6.502, "text": "the quiet robot sang in the garden."}, {"w": 8, "rank": 5, "lp": -6.407, "text": "the old robot sang in the garden."}, {"w": 16, "rank": 5, "lp": -6.407, "text": "the old robot sang in the garden."}, {"w": 32, "rank": 5, "lp": -6.407, "text": "the old robot sang in the garden."}];
const MONO = {"rows": [{"w": 1, "lp": -6.188}, {"w": 2, "lp": -6.188}, {"w": 3, "lp": -6.422}, {"w": 4, "lp": -6.224}, {"w": 5, "lp": -6.224}, {"w": 6, "lp": -6.141}, {"w": 7, "lp": -5.947}, {"w": 8, "lp": -5.947}], "drops": 1, "dropAt": 3};
const LEN = {"corr": 0.061, "mean": 30.6, "topTotal": 28.8, "topChar": 36.5, "totalBest": "a small dog sat in the garden.", "charBest": "the small robot waited under the tree.", "short": 21, "long": 38, "top": 10};
const DIV = {"n": 16, "beam": ["the old robot sang in the garden.", "a small bird slept near the door.", "the red robot sang in the garden.", "the old robot sat by the river."], "sample": ["the robot sat on the mat.", "the red cat waited under the tree.", "a red robot waited on the mat.", "a small cat sarden."], "beamSubj": 8, "beamOv": 1.53, "sampSubj": 12.8, "sampOv": 1.34, "sets": 12};
const EX = [{"ctx": "the", "acc": 0.503, "err": 0.0, "trust": 0.131, "nocoin": 0.043}, {"ctx": "the small cat", "acc": 0.447, "err": 1.1102230246251565e-16, "trust": 0.367, "nocoin": 0.001}, {"ctx": "a red dog s", "acc": 0.753, "err": 1.1102230246251565e-16, "trust": 0.247, "nocoin": 0.0}];
const DRAFT = [{"order": 1, "acc": 0.173, "per": 1.69, "ok": 0.93, "distinct": 39}, {"order": 2, "acc": 0.605, "per": 3.42, "ok": 0.93, "distinct": 40}, {"order": 3, "acc": 0.837, "per": 4.35, "ok": 0.9, "distinct": 40}, {"order": 5, "acc": 0.929, "per": 4.72, "ok": 0.88, "distinct": 39}];
const GAM = [{"g": 1, "acc": 0.911, "per": 1.91, "theory": 1.91, "speedup": 1.74}, {"g": 2, "acc": 0.91, "per": 2.82, "theory": 2.74, "speedup": 2.35}, {"g": 4, "acc": 0.842, "per": 4.37, "theory": 3.65, "speedup": 3.12}, {"g": 8, "acc": 0.76, "per": 7.08, "theory": 3.82, "speedup": 3.94}];
const LOST = {"best": "a small dog sat in the garden.", "step": 9, "prefix": "a small d", "rank": 10, "cands": 64, "width": 8, "widths": [{"w": 32, "found": false}, {"w": 64, "found": false}, {"w": 128, "found": true}]};
const ONE = {"ctx": "the old robot s", "props": ["a", "n", "g", " "], "rows": [{"top": "a", "p": 0.691}, {"top": "n", "p": 0.499}, {"top": "g", "p": 0.996}, {"top": " ", "p": 0.996}, {"top": "i", "p": 0.296}]};
const TREE = [{"order": 1, "chain": 1.65, "tree": 1.96, "chainNode": 0.41, "treeNode": 0.68, "treeNodes": 2.88}, {"order": 2, "chain": 3.54, "tree": 3.32, "chainNode": 0.88, "treeNode": 0.91, "treeNodes": 3.65}, {"order": 3, "chain": 4.41, "tree": 3.72, "chainNode": 1.1, "treeNode": 0.97, "treeNodes": 3.86}];
const BATCH = {"perRound": 4.41, "rows": [{"b": 1, "base": 28.0, "spec": 28.0, "speedup": 4.41}, {"b": 8, "base": 28.0, "spec": 28.0, "speedup": 4.41}, {"b": 32, "base": 28.0, "spec": 28.0, "speedup": 4.41}, {"b": 128, "base": 28.0, "spec": 89.6, "speedup": 1.38}, {"b": 512, "base": 71.7, "spec": 358.4, "speedup": 0.88}]};

const PURPLE2 = '#c7a6ff';
function rowBars(sh, items, y0, dy, wmax, val, lab, st){
  items.forEach((it, i) => {
    const y = y0 - i * dy;
    sh.push(S.r(.70, y, Math.max(.06, wmax * val(it)), .40, st(it), null));
    sh.push(fitT(.60, y + .20, lab(it), {anchor:'end', c:COL.grey, fs:.26}));
  });
  return sh;
}

/* ================================================================ tab 1: the gap */
const CODE_GT = [
  '# the grammar admits 1,250 sentences, so score every one of them exactly',
  'def logprob(model, s):',
  '    ctx, tot = "\\n", 0.0',
  '    for ch in s + "\\n":',
  '        tot += log(model.probs(ctx)[model.stoi[ch]])',
  '        ctx += ch',
  '    return tot',
  '',
  'ranking = sorted((logprob(m, s), s) for s in all_sentences())',
  '# greedy takes the best CHARACTER at every step - not the best sentence'
];
function gtStage(o){
  const sh = [];
  const rows = [[{zh:'整句機率最高的', en:'the most likely sentence'}, GT.best, GT.bestLp, 'ok'],
                [{zh:'greedy 產生的', en:'what greedy produces'}, GT.greedy, GT.greedyLp, 'hot'],
                [{zh:'最低的那一句', en:'the least likely sentence'}, GT.worst, GT.worstLp, 'idle']];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.40 - i * 1.15;
    sh.push(fitT(.70, y + .60, r[0], {anchor:'start', c:COL.tealL, fs:.27}));
    sh.push(box(.70, y, 7.00, .56, r[3], r[1], .26));
    sh.push(fitT(9.20, y + .28, 'log p ' + r[2].toFixed(2), {anchor:'end', c:COL.pale, fs:.26}));
  });
  if (o.upto >= 1){
    const w = 8.50, frac = GT.rank / GT.n;
    sh.push(S.r(.70, .95, w, .30, 'ghost', null));
    sh.push(S.r(.70, .95, Math.max(.06, w * frac), .30, 'bad', null));
    sh.push(fitT(.70, .62, {zh:'greedy 的句子在 ' + GT.n.toLocaleString() + ' 句裡排第 ' + GT.rank,
                            en:'greedy ranks ' + GT.rank + ' of ' + GT.n.toLocaleString()},
                 {anchor:'start', c:COL.red, fs:.27}));
  }
  return caps(sh, o);
}
function gtFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:gtStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:8,
    cap:{zh:'先有標準答案：' + GT.n.toLocaleString() + ' 句全部精確算分',
         en:'ground truth first: all ' + GT.n.toLocaleString() + ' sentences, scored exactly'},
    panels:[{lbl:'sentences', chips:[chip(GT.n.toLocaleString(), 'dim')]},
            {lbl:'best log p', chips:[chip(GT.bestLp.toFixed(2), 'ok')]}],
    msg:{zh:'文法小到可以把每一句都列出來、每一句的機率都精確算出來（逐步 log 機率相加）。所以今天「有沒有找到最好的句子」是可以直接對答案的，不是感覺。',
         en:'The grammar is small enough to enumerate every sentence and score each one exactly by summing its per-step log probabilities. So today\'s question - did the search find the best sentence - can be checked against an answer key instead of eyeballed.'}});
  p({upto:1, line:9,
    cap:{zh:'而逐步取最大值落在第 ' + GT.rank + ' 名', en:'and taking the best character every step lands at rank ' + GT.rank},
    panels:[{lbl:'greedy rank', chips:[chip(GT.rank + ' / ' + GT.n.toLocaleString(), 'bad')]},
            {lbl:'less likely by', chips:[chip((Math.exp(GT.bestLp - GT.greedyLp)).toFixed(2) + 'x', 'bad')]}],
    msg:{zh:'greedy 每一步都對，整句卻只排第 ' + GT.rank + '。原因是它會走進「之後沒有好結尾」的前綴而且回不去——**逐步 argmax ≠ 序列 argmax**。這個落差調 temperature 或 top-p 都補不起來，因為它根本不是取樣問題，是搜尋問題。',
         en:'Greedy is right at every step and 44th overall. It walks into a prefix from which no good ending exists and cannot back out: **argmax per step is not argmax over sequences**. No temperature or top-p setting closes that gap, because it is not a sampling problem at all.'}});
  p({upto:2, line:9,
    cap:{zh:'整個值域大概長這樣', en:'and this is the whole range'},
    panels:[{lbl:'best', chips:[chip(GT.bestLp.toFixed(2), 'ok')]},
            {lbl:'greedy', chips:[chip(GT.greedyLp.toFixed(2), 'hot')]},
            {lbl:'worst', chips:[chip(GT.worstLp.toFixed(2), 'dim')]}],
    msg:{zh:'最好與最差之間差 ' + (GT.bestLp - GT.worstLp).toFixed(2) + ' 個 log 單位（約 ' + Math.round(Math.exp(GT.bestLp - GT.worstLp)) + ' 倍）。greedy 離最好的只差 ' + (GT.bestLp - GT.greedyLp).toFixed(2) + '——所以要注意：搜尋能買到的東西本來就有上限，下一個分頁就是在這個區間裡爬。',
         en:'Best to worst spans ' + (GT.bestLp - GT.worstLp).toFixed(2) + ' log units, about ' + Math.round(Math.exp(GT.bestLp - GT.worstLp)) + 'x in probability, and greedy sits only ' + (GT.bestLp - GT.greedyLp).toFixed(2) + ' below the top. Worth keeping in view: there is a ceiling on what search can buy, and the next tab climbs inside it.'}});
  return F.list;
}

/* ================================================================ tab 2: beam search */
const CODE_BEAM = [
  'beams = [(0.0, prefix)]',
  'for step in range(max_len):',
  '    cand = []',
  '    for lp, text in beams:',
  '        lg = log(model.probs(text))',
  '        for i in argsort(-lg, kind="stable")[:width]:   # ties are real',
  '            cand.append((lp + lg[i], text + vocab[i]))',
  '    beams = sorted(cand, reverse=True)[:width]          # keep w prefixes',
  '',
  '# width 1 IS greedy; width w costs w sequences of KV cache per step'
];
function beamStage(o){
  const sh = [];
  const top = Math.max.apply(null, BEAM.map(r => r.rank));
  BEAM.forEach((r, i) => {
    if (i > o.upto) return;
    const x = .95 + i * 1.50;
    const h = Math.max(.08, 2.60 * r.rank / top);
    sh.push(S.r(x, 1.60, .95, h, i === o.upto ? 'hot' : 'ok', null));
    sh.push(fitT(x + .47, 1.60 + h + .22, 'rank ' + r.rank, {c:COL.pale, fs:.24}));
    sh.push(fitT(x + .47, 1.32, 'w=' + r.w, {c:COL.grey, fs:.24}));
  });
  sh.push(fitT(4.9, .85, BEAM[Math.min(o.upto, BEAM.length - 1)].text,
               {c:COL.tealL, fs:.30}));
  sh.push(fitT(.70, 4.95, {zh:'柱子越矮越好（名次越前面）', en:'shorter is better: the rank of the sentence found'},
               {anchor:'start', c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function beamFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:beamStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'width 1 就是 greedy', en:'width 1 is greedy'},
    panels:[{lbl:'width 1', chips:[chip('rank ' + BEAM[0].rank, 'bad')]}],
    msg:{zh:'beam search 的 width=1 完全等於 greedy。順帶一提：這個模型有**真正的平手**（兩個字元機率一模一樣），所以排序必須是 stable 的，否則光是排序法不同就會讓 width=1 跟 greedy 給出不同的句子。',
         en:'A width-1 beam is exactly greedy. Worth noting: this model has genuine ties - two characters with identical probability - so the sort has to be stable, or the choice of sorting algorithm alone makes width-1 disagree with greedy.'}});
  p({upto:2, line:8,
    cap:{zh:'留 w 個前綴，不是留 1 個', en:'keep w prefixes alive instead of one'},
    panels:[{lbl:'width 2', chips:[chip('rank ' + BEAM[1].rank, 'bad')]},
            {lbl:'width 4', chips:[chip('rank ' + BEAM[2].rank, 'act')]}],
    msg:{zh:'每一步把所有存活前綴各自展開，再整體取分數最高的 w 個。於是「現在看起來第二好、但之後會贏」的前綴有機會活下來——width 4 就從第 ' + BEAM[0].rank + ' 名爬到第 ' + BEAM[2].rank + ' 名。',
         en:'At every step every surviving prefix is expanded and the best w of all the candidates are kept, so a prefix that looks second-best now but wins later has a chance to survive. Width 4 climbs from rank ' + BEAM[0].rank + ' to rank ' + BEAM[2].rank + '.'}});
  p({upto:BEAM.length - 1, line:9,
    cap:{zh:'但爬到第 ' + BEAM[BEAM.length - 1].rank + ' 名就停了', en:'but it stops at rank ' + BEAM[BEAM.length - 1].rank},
    panels:[{lbl:'width 8-32', chips:[chip('rank ' + BEAM[BEAM.length - 1].rank, 'act')]},
            {lbl:'optimum', chips:[chip('rank 1', 'ok')]},
            {lbl:'cost', chips:[chip('w sequences / step', 'bad')]}],
    msg:{zh:'再加寬也到不了第 1 名：最好的那一句的前綴在很早就被剪掉了。beam search 是**啟發式**不是最大化器——程式裡的 assert 只保證「不會超過列舉出來的最佳解」，不保證會找到它。成本則是線性的：width 8 等於每一步都在跑 8 條序列，8 份 KV cache、8 倍 attention。',
         en:'Widening further never reaches rank 1: the prefix of the best sentence was pruned early. Beam search is a heuristic, not a maximiser - the asserts in the file only guarantee that no beam beats the enumerated optimum, not that it finds it. The cost is linear: width 8 runs 8 sequences every step, 8 KV caches and 8x the attention work.'}});
  return F.list;
}

/* ================================================================ tab 3: the two bugs of beam search */
const CODE_CURSE = [
  '# 1. wider is not monotonically better',
  'for w in range(1, 17):',
  '    lp, text = beam_search(model, width=w)   # lp can go DOWN as w goes up',
  '',
  '# 2. every character multiplies the score by something < 1',
  'score = logp                      # alpha = 0: favours short outputs',
  'score = logp / len(text) ** alpha # the standard patch, a pure heuristic'
];
function curseStage(o){
  const sh = [];
  if (o.upto === 0){
    const lo = Math.min.apply(null, MONO.rows.map(r => r.lp));
    const hi = Math.max.apply(null, MONO.rows.map(r => r.lp));
    MONO.rows.forEach((r, i) => {
      const x = .85 + i * 1.15, h = .35 + 2.6 * (r.lp - lo) / (hi - lo);
      const bad = i > 0 && r.lp < MONO.rows[i - 1].lp - 1e-9;
      sh.push(S.r(x, 1.50, .80, h, bad ? 'bad' : 'ok', null));
      sh.push(fitT(x + .40, 1.50 + h + .22, r.lp.toFixed(2), {c:bad ? COL.red : COL.pale, fs:.23}));
      sh.push(fitT(x + .40, 1.22, 'w=' + r.w, {c:COL.grey, fs:.23}));
    });
    sh.push(fitT(.70, 4.95, {zh:'柱子越高越好（log p 越大）', en:'taller is better: higher log probability'},
                 {anchor:'start', c:COL.grey, fs:.26}));
  } else {
    const rows = [[{zh:'全部句子的平均長度', en:'mean length, all sentences'}, LEN.mean, 'idle'],
                  [{zh:'總分最高的 ' + LEN.top + ' 句', en:'top ' + LEN.top + ' by total log p'}, LEN.topTotal, 'hot'],
                  [{zh:'每字元平均最高的 ' + LEN.top + ' 句', en:'top ' + LEN.top + ' by log p per character'}, LEN.topChar, 'act']];
    rows.forEach((r, i) => {
      const y = 4.10 - i * .95;
      sh.push(fitT(.70, y + .20, r[0], {anchor:'start', c:COL.pale, fs:.27}));
      sh.push(S.r(4.60, y, Math.max(.06, 3.8 * r[1] / LEN.long), .40, r[2], null));
      sh.push(fitT(9.20, y + .20, r[1].toFixed(1) + ' chars', {anchor:'end', c:COL.grey, fs:.25}));
    });
    sh.push(fitT(4.9, 1.05, {zh:'長度與 log 機率的相關係數 ' + LEN.corr,
                             en:'correlation between length and log probability: ' + LEN.corr},
                 {c:COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function curseFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:curseStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'加寬 beam 不保證變好', en:'a wider beam is not always a better answer'},
    panels:[{lbl:'drops in 1-16', chips:[chip(String(MONO.drops), 'bad')]},
            {lbl:'first at width', chips:[chip(String(MONO.dropAt), 'bad')]}],
    msg:{zh:'換一個用比較小語料訓練的模型，一格一格掃 width：width ' + MONO.dropAt + ' 比 width ' + (MONO.dropAt - 1) + ' **更差**。這不是實作 bug——beam 變寬會讓更多前綴活著，而一個早期看起來普通的前綴可能活到後面把真正該贏的擠掉。翻譯文獻把這個現象在品質分數上的版本叫 beam search curse：width 超過 5～10 之後 BLEU 開始下降，而模型機率還在上升。',
         en:'On a model trained on a smaller corpus, scanning the width one step at a time: width ' + MONO.dropAt + ' is *worse* than width ' + (MONO.dropAt - 1) + '. Not an implementation bug - a wider beam keeps more prefixes alive, and a mediocre-looking early prefix can survive to crowd out the one that should have won. Translation papers call the quality-score version of this the beam search curse: past a width of 5 to 10, BLEU falls while model probability rises.'}});
  p({upto:1, line:6,
    cap:{zh:'而且總機率最高＝偏好短的', en:'and the highest total probability means the shortest'},
    panels:[{lbl:'by total', chips:[chip(LEN.topTotal.toFixed(1) + ' chars', 'hot')]},
            {lbl:'by per character', chips:[chip(LEN.topChar.toFixed(1) + ' chars', 'act')]},
            {lbl:'all', chips:[chip(LEN.mean.toFixed(1) + ' chars', 'dim')]}],
    msg:{zh:'每多一個字元就多乘一個小於 1 的數，所以「總機率最高」永遠偷偷夾帶「比較短」。在這裡效果很小（相關係數只有 ' + LEN.corr + '，句子長度只有 ' + LEN.short + '–' + LEN.long + ' 的差距），但方向看得到：總分前 ' + LEN.top + ' 名平均 ' + LEN.topTotal.toFixed(1) + ' 個字元，改用每字元平均之後變成 ' + LEN.topChar.toFixed(1) + '。翻譯那種長度差好幾十個 token 的場合，沒修正的 beam 會直接把句子砍短。修正法就是除以 length**alpha——一個沒有推導、只有慣例的 heuristic。',
         en:'Every extra character multiplies the score by something below 1, so "highest total probability" always smuggles in "shorter". Here the effect is small - the correlation is only ' + LEN.corr + ' and sentences span just ' + LEN.short + ' to ' + LEN.long + ' characters - but the direction is visible: the top ' + LEN.top + ' by total average ' + LEN.topTotal.toFixed(1) + ' characters against ' + LEN.topChar.toFixed(1) + ' when scored per character. In translation, where candidates differ by tens of tokens, an uncorrected beam truncates. The fix is dividing by length**alpha: a heuristic with a convention behind it and no derivation.'}});
  return F.list;
}

/* ================================================================ tab 4: diversity */
const CODE_DIV = [
  'beam_search(model, width=48, n_best=16)   # the 16 most probable sentences',
  'sample_lines(model, rng, n=16)            # 16 sentences drawn at t=1.0',
  '',
  '# a "subject" here is determiner + adjective + noun',
  'len({parse(s)[:3] for s in texts})'
];
function divStage(o){
  const sh = [];
  const cols = [[DIV.beam, {zh:'beam 的前 4 名', en:'the beam\'s top 4'}, 4.40, 'hot'],
                [DIV.sample, {zh:'抽樣抽到的 4 句', en:'4 sentences drawn at t=1.0'}, 2.35, 'ok']];
  cols.forEach((c, ci) => {
    if (ci > o.upto) return;
    sh.push(fitT(.70, c[2] + .62, c[1], {anchor:'start', c:COL.tealL, fs:.27}));
    c[0].forEach((t, i) => {
      sh.push(box(.70, c[2] - i * .50, 8.50, .42, c[3], t, .24));
    });
  });
  if (o.upto >= 1){
    sh.push(fitT(4.9, .75, {zh:'不同的主詞：beam ' + DIV.beamSubj + ' 個，抽樣 ' + DIV.sampSubj + ' 個（' + DIV.n + ' 句裡）',
                            en:'distinct subjects out of ' + DIV.n + ': beam ' + DIV.beamSubj + ', sampling ' + DIV.sampSubj},
                 {c:COL.pale, fs:.29}));
  }
  return caps(sh, o);
}
function divFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:divStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'跟 beam 要 n 個答案，它給你 n 個鄰居', en:'ask a beam for n answers and it returns n neighbours'},
    panels:[{lbl:'distinct sentences', chips:[chip(String(DIV.n), 'dim')]},
            {lbl:'distinct subjects', chips:[chip(String(DIV.beamSubj), 'bad')]}],
    msg:{zh:'' + DIV.n + ' 句全都不一樣，所以「有幾句不重複」看不出問題。看主詞就看得出來：' + DIV.n + ' 句裡只有 ' + DIV.beamSubj + ' 個不同的主詞，整組都是同一句話的變形。',
         en:'All ' + DIV.n + ' are different strings, so counting distinct sentences says nothing. Counting subjects does: ' + DIV.n + ' sentences share only ' + DIV.beamSubj + ' distinct subjects. The set is variations on one sentence.'}});
  p({upto:1, line:4,
    cap:{zh:'同一個模型，改用抽樣', en:'the same model, sampled instead'},
    panels:[{lbl:'beam subjects', chips:[chip(String(DIV.beamSubj), 'bad')]},
            {lbl:'sampled subjects', chips:[chip(String(DIV.sampSubj), 'ok'),
                                            chip('mean of ' + DIV.sets, 'dim')]}],
    msg:{zh:'同樣抽 ' + DIV.n + ' 句，抽樣給出 ' + DIV.sampSubj + ' 個不同主詞（' + DIV.sets + ' 組獨立抽樣的平均）。這不是 beam 寫壞了：beam 在找「最可能的輸出」，而最可能的那些輸出本來就長得很像。所以 beam search 適合**只有一個正確答案**的任務（翻譯、語音辨識、受限抽取），聊天模型則需要前一天的取樣。',
         en:'Drawing the same ' + DIV.n + ' sentences by sampling gives ' + DIV.sampSubj + ' distinct subjects, averaged over ' + DIV.sets + ' independent draws. This is not a broken beam: a beam searches for the most probable output, and the most probable outputs resemble each other by construction. Beam search belongs where one answer is correct - translation, speech recognition, constrained extraction - and chat models need the previous day\'s sampling.'}});
  return F.list;
}

/* ================================================================ tab 5: speculative round */
const CODE_SPEC = [
  '# draft gamma tokens with the cheap model',
  'for _ in range(gamma):',
  '    q = draft.probs(ctx); i = sample(q); ctx += vocab[i]',
  '',
  '# verify ALL of them in one pass of the expensive model',
  'for i, q in zip(proposals, qs):',
  '    p = target.probs(ctx)',
  '    if random() < min(1, p[i] / q[i]):      # accept',
  '        emit(i)',
  '    else:                                   # reject: repair and stop',
  '        emit(sample(norm(maximum(p - q, 0))))',
  '        break',
  'else:',
  '    emit(sample(p))                         # all accepted: a free bonus token'
];
function specStage(o){
  const sh = [], labs = ['i', 'n', ' ', 't'], acc = [true, true, true, false];
  labs.forEach((c, i) => {
    if (o.upto < 1 && i > o.upto) { /* draft appears progressively */ }
    const x = .85 + i * 1.55;
    sh.push(box(x, 3.55, 1.25, .62, 'act', "'" + c + "'", .30));
    if (o.upto >= 1){
      const ok = acc[i];
      sh.push(S.e(x + .62, 3.45, x + .62, 2.70, {s:ok ? 'ok' : 'bad', arrow:true, w:.05}));
      sh.push(box(x, 2.05, 1.25, .62, ok ? 'ok' : 'bad',
                  {zh:ok ? '接受' : '拒絕', en:ok ? 'keep' : 'reject'}, .26));
    }
  });
  sh.push(fitT(.70, 4.45, {zh:'draft 模型：' + labs.length + ' 次便宜的 forward',
                           en:'draft model: ' + labs.length + ' cheap passes'},
               {anchor:'start', c:COL.tealL, fs:.27}));
  if (o.upto >= 1){
    sh.push(fitT(.70, 1.65, {zh:'target 模型：一次 forward 就驗證完全部 ' + labs.length + ' 個位置',
                             en:'target model: ONE pass over all ' + labs.length + ' positions'},
                 {anchor:'start', c:COL.grey, fs:.26}));
  }
  if (o.upto >= 2){
    sh.push(box(7.05, 2.05, 2.20, .62, 'hot', {zh:'改抽 (p-q)+', en:'resample (p-q)+'}, .25));
    sh.push(fitT(4.9, 1.10, {zh:'這一輪吐出 3 個接受的 ＋ 1 個修補 ＝ 4 個 token，只用了 1 次 target forward',
                             en:'this round emits 3 accepted + 1 repair = 4 tokens for 1 target pass'},
                 {c:COL.pale, fs:.28}));
  }
  return caps(sh, o);
}
function specFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:specStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:3,
    cap:{zh:'先讓便宜的模型猜 4 個', en:'let the cheap model guess 4 tokens'},
    panels:[{lbl:'draft passes', chips:[chip('4 cheap', 'act')]}],
    msg:{zh:'draft 模型照常一個一個抽（它便宜，所以無所謂）。關鍵在下一步：這 4 個 token 可以被**一次** forward 驗證完，因為驗證是把 4 個位置一起送進去算，而產生它們本來需要 4 次 forward。decode 慢的原因是每一步都要把整個模型的權重讀一遍（前幾天的 bandwidth-bound），一次驗 4 個等於權重只讀一次。',
         en:'The draft samples one token at a time, which is fine because it is cheap. The point is the next step: those 4 tokens can be verified in ONE forward pass, because verification scores 4 positions at once while generating them would have taken 4 passes. Decode is slow because every step re-reads the whole model from memory (the bandwidth-bound picture of the earlier days), so verifying 4 at once reads the weights once.'}});
  p({upto:1, line:8,
    cap:{zh:'逐個擲硬幣：min(1, p/q)', en:'one coin per token: accept with min(1, p/q)'},
    panels:[{lbl:'accepted', chips:[chip('3', 'ok')]},
            {lbl:'rejected', chips:[chip('1', 'bad')]}],
    msg:{zh:'對每個提議擲一次硬幣：target 覺得它至少跟 draft 一樣可能（p ≥ q）就一定收；覺得比較不可能就以 p/q 的機率收。第一個被拒絕的地方就停——後面的提議都是建立在那個 token 上的，自然作廢。',
         en:'One coin per proposal: if the target likes the token at least as much as the draft did (p >= q) it is always kept; if the target likes it less, it is kept with probability p/q. The first rejection stops the round, because everything after it was conditioned on that token.'}});
  p({upto:2, line:11,
    cap:{zh:'被拒絕的那一格要補一個，不能直接丟', en:'a rejection is repaired, not just dropped'},
    panels:[{lbl:'emitted', chips:[chip('4 tokens', 'ok')]},
            {lbl:'target passes', chips:[chip('1', 'ok')]}],
    msg:{zh:'拒絕之後不是「重來」，而是從 (p − q) 的正部正規化後抽一個——這一步看起來像細節，其實是整個方法成立的關鍵（下一個分頁會證明）。而且全部被接受時還會白賺一個 token：那一輪的最後一次 target forward 本來就算出了下一格的分佈。',
         en:'A rejection is not a retry: the token is drawn from the normalised positive part of (p - q). That looks like a detail and is the reason the whole method is valid, which the next tab shows. And when every proposal is accepted the round emits a free bonus token, because the last verification pass already produced the distribution for the next position.'}});
  return F.list;
}

/* ================================================================ tab 6: exactness */
const CODE_EXACT = [
  'acc  = minimum(1, p / q)',
  'kept = q * acc                       # mass that survives the coin flip',
  'rest = 1 - kept.sum()                # probability of rejecting',
  'resid = maximum(p - q, 0); resid /= resid.sum()',
  'out = kept + rest * resid            # == p, to the last bit',
  '',
  '# two shortcuts that sound equivalent and are not:',
  'out = q                              # "the draft was good enough"',
  'out = where(p >= q, q, 0) + ...      # "same rule without the coin flip"'
];
function exactStage(o){
  const sh = [];
  EX.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.20 - i * 1.25;
    sh.push(fitT(.70, y + .60, "context '" + r.ctx + "'", {anchor:'start', c:COL.tealL, fs:.27}));
    const cells = [[{zh:'正確規則', en:'correct rule'}, r.err < 1e-9 ? '0' : r.err.toExponential(0), 'ok'],
                   [{zh:'直接信 draft', en:'trust the draft'}, r.trust.toFixed(3), 'bad'],
                   [{zh:'不擲硬幣', en:'no coin flip'}, r.nocoin.toFixed(3), r.nocoin > .01 ? 'bad' : 'idle']];
    cells.forEach((c, j) => {
      const x = .70 + j * 2.95;
      sh.push(box(x, y, 2.70, .58, c[2], c[0], .24));
      sh.push(fitT(x + 1.35, y - .24, c[1], {c:COL.pale, fs:.25}));
    });
  });
  sh.push(fitT(4.9, .70, {zh:'每一格是 max |輸出分佈 − target 分佈|',
                          en:'each cell is max |output distribution - target distribution|'},
               {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function exactFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:exactStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:5,
    cap:{zh:'輸出就是 target 的分佈，一個位元都不差',
         en:'the output is the target distribution, to the last bit'},
    panels:[{lbl:'accept rate', chips:[chip(EX[0].acc.toFixed(3), 'dim')]},
            {lbl:'max error', chips:[chip('1e-16', 'ok')]}],
    msg:{zh:'把兩條路加起來：q(x)·min(1, p/q) ＋ P(拒絕)·norm(p−q)+(x) ＝ p(x)。這不是「近似」——誤差是 1e-16，浮點數等級。所以 speculative decoding 是系統最佳化，不是品質取捨：你部署的模型就是回答你的模型。',
         en:'Add the two routes: q(x)*min(1, p/q) + P(reject)*norm(p - q)+(x) = p(x). Not approximately - the measured error is 1e-16, which is floating point. That is why speculative decoding is a systems optimisation rather than a quality trade-off: the model you deployed is the model that answers.'}});
  p({upto:EX.length - 1, line:8,
    cap:{zh:'兩個聽起來一樣的捷徑，都是錯的', en:'two shortcuts that sound equivalent, both wrong'},
    panels:[{lbl:'trust the draft', chips:[chip('up to ' + Math.max.apply(null, EX.map(r => r.trust)).toFixed(2), 'bad')]},
            {lbl:'no coin flip', chips:[chip('up to ' + Math.max.apply(null, EX.map(r => r.nocoin)).toFixed(2), 'bad')]}],
    msg:{zh:'「draft 夠好就直接收」寫出來就知道不對。但「把硬幣拿掉、只要 target 覺得不比 draft 差就收」看起來很合理——它仍然會把分佈搬動最多 ' + Math.max.apply(null, EX.map(r => r.nocoin)).toFixed(2) + '。隨機接受才讓每個 token 被接受的質量剛好是 min(p, q)，而殘差抽樣把剩下的補回來；兩半都是承重的。這種錯誤不會噴例外，只會讓模型默默變差。',
         en:'"The draft was good enough" is obviously wrong once written down. But "drop the coin and accept whenever the target likes the token at least as much" sounds fine - and still moves the distribution by up to ' + Math.max.apply(null, EX.map(r => r.nocoin)).toFixed(2) + '. The randomised accept is what makes the accepted mass come out as min(p, q) per token, and the residual draw puts the rest back. Both halves are load-bearing, and getting it wrong raises no error at all: the model just quietly gets worse.'}});
  return F.list;
}

/* ================================================================ tab 7: the knobs */
const CODE_KNOBS = [
  '# how good the draft is -> how often it is accepted',
  'draft = NGram(text, order=o)          # lower order = cheaper and worse',
  '',
  '# how far ahead to guess',
  'tokens_per_round = (1 - a**(gamma+1)) / (1 - a)     # if accepts were iid',
  'speedup = tokens_per_round / (1 + gamma * draft_cost)'
];
function knobsStage(o){
  const sh = [];
  if (o.upto === 0){
    DRAFT.forEach((r, i) => {
      const y = 4.10 - i * .85;
      sh.push(fitT(.70, y + .20, {zh:'draft order ' + r.order, en:'draft order ' + r.order},
                   {anchor:'start', c:COL.pale, fs:.27}));
      sh.push(S.r(3.00, y, Math.max(.06, 3.0 * r.acc), .40, 'hot', null));
      sh.push(fitT(6.20, y + .20, (r.acc * 100).toFixed(0) + '%', {anchor:'start', c:COL.tealL, fs:.25}));
      sh.push(fitT(9.20, y + .20, {zh:'合文法 ' + (r.ok * 100).toFixed(0) + '%',
                                   en:'grammatical ' + (r.ok * 100).toFixed(0) + '%'},
                   {anchor:'end', c:COL.grey, fs:.25}));
    });
    sh.push(fitT(4.9, .80, {zh:'接受率（橘）隨 draft 變好而上升，品質（右）完全不動',
                            en:'the accept rate moves with the draft; the quality column does not'},
                 {c:COL.grey, fs:.27}));
  } else {
    GAM.forEach((r, i) => {
      const y = 4.10 - i * .85, top = Math.max.apply(null, GAM.map(x => x.per));
      sh.push(fitT(.70, y + .20, 'gamma ' + r.g, {anchor:'start', c:COL.pale, fs:.27}));
      sh.push(S.r(2.40, y, Math.max(.06, 3.2 * r.per / top), .40, 'ok', null));
      sh.push(S.r(2.40, y - .22, Math.max(.06, 3.2 * r.theory / top), .18, 'ghost', null));
      sh.push(fitT(9.20, y + .20, r.per.toFixed(2) + {zh:' token/輪　理論 ', en:' tokens/round, theory '}[curV() === 0 ? 'zh' : 'en'] + r.theory.toFixed(2),
                   {anchor:'end', c:COL.grey, fs:.25}));
    });
    sh.push(fitT(4.9, .80, {zh:'實心＝實測，細條＝獨立假設下的理論值',
                            en:'solid: measured.  thin bar: what the independence assumption predicts'},
                 {c:COL.grey, fs:.27}));
  }
  return caps(sh, o);
}
function knobsFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:knobsStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'draft 有多好，只影響速度', en:'how good the draft is only changes the speed'},
    panels:[{lbl:'order 1', chips:[chip((DRAFT[0].acc * 100).toFixed(0) + '%', 'bad')]},
            {lbl:'order 5', chips:[chip((DRAFT[DRAFT.length - 1].acc * 100).toFixed(0) + '%', 'ok')]},
            {lbl:'grammatical', chips:[chip('unchanged', 'ok')]}],
    msg:{zh:'draft 用同一個 n-gram 但階數更低：越低越便宜也越爛。接受率從 ' + (DRAFT[0].acc * 100).toFixed(0) + '% 一路到 ' + (DRAFT[DRAFT.length - 1].acc * 100).toFixed(0) + '%，但右邊那欄（合文法率）幾乎不動——它**本來就不該動**，因為輸出分佈與 draft 無關。爛 draft 只會讓你慢，不會讓你錯。',
         en:'The draft is the same n-gram at a lower order: cheaper and worse. The accept rate runs from ' + (DRAFT[0].acc * 100).toFixed(0) + '% to ' + (DRAFT[DRAFT.length - 1].acc * 100).toFixed(0) + '% while the quality column barely moves - and it is not supposed to, because the output distribution does not depend on the draft at all. A bad draft costs speed, never correctness.'}});
  p({upto:1, line:5,
    cap:{zh:'猜多遠：實測比理論好，而且理由很有意思',
         en:'how far to guess - and why the measurement beats the formula'},
    panels:[{lbl:'gamma 8 measured', chips:[chip(GAM[GAM.length - 1].per.toFixed(2), 'ok')]},
            {lbl:'iid theory', chips:[chip(GAM[GAM.length - 1].theory.toFixed(2), 'dim')]},
            {lbl:'speedup', chips:[chip(GAM[GAM.length - 1].speedup.toFixed(2) + 'x', 'hot')]}],
    msg:{zh:'教科書公式是 (1 − a^(γ+1))/(1 − a)，假設每個提議**獨立**以 a 的機率被接受。gamma=8 時實測 ' + GAM[GAM.length - 1].per.toFixed(2) + ' 個 token/輪，理論只有 ' + GAM[GAM.length - 1].theory.toFixed(2) + '。公式沒錯，是假設錯了：接受是**相關**的——draft 在可預測的那一段（「…in the gar|den.」）會連續全中，只在分岔點出錯。所以實務上大家報的是「平均接受長度」而不是接受率。',
         en:'The textbook formula is (1 - a^(gamma+1))/(1 - a), assuming each proposal is accepted independently with probability a. At gamma=8 the measurement is ' + GAM[GAM.length - 1].per.toFixed(2) + ' tokens per round against a predicted ' + GAM[GAM.length - 1].theory.toFixed(2) + '. The formula is fine; its assumption is not. Acceptances are correlated: the draft is right through a whole predictable stretch ("...in the gar|den.") and wrong only at branch points. That is why deployments report accepted *length* rather than accept rate.'}});
  return F.list;
}

/* ================================================================ tabs 8-9: the servers */
function mapStage(o){
  const sh = [], rows = o.rows;
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.30 - i * .80;
    sh.push(box(.70, y, 3.30, .58, 'idle', r[0], .26));
    sh.push(S.e(4.15, y + .29, 4.75, y + .29, {s:'ghost', arrow:true, w:.05}));
    sh.push(box(4.90, y, 4.30, .58, i === o.upto ? 'hot' : 'ok', r[1], .26));
  });
  sh.push(fitT(2.35, 5.08, {zh:'這個檔案裡的東西', en:'what this file calls it'},
               {c:COL.grey, fs:.26}));
  sh.push(fitT(7.05, 5.08, o.title, {c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
const CODE_VLLM = [
  '# beam search is NOT a sampling parameter any more:',
  'llm.beam_search(prompts, BeamSearchParams(beam_width=4, max_tokens=64))',
  '',
  '# speculative decoding is server configuration:',
  'vllm serve meta-llama/Llama-3.1-70B-Instruct \\',
  '  --speculative-config \'{"model": "meta-llama/Llama-3.1-8B-Instruct",',
  '                         "num_speculative_tokens": 5}\'',
  '',
  '# or with no draft model at all - copy from the prompt:',
  '  --speculative-config \'{"method": "ngram", "num_speculative_tokens": 4,',
  '                         "prompt_lookup_max": 4}\''
];
const VLLM_ROWS = [
  [{zh:'beam_search(width)', en:'beam_search(width)'},
   {zh:'LLM.beam_search()（已移出 SamplingParams）', en:'LLM.beam_search(), moved out of SamplingParams'}],
  [{zh:'gamma', en:'gamma'}, {zh:'num_speculative_tokens', en:'num_speculative_tokens'}],
  [{zh:'draft 模型', en:'the draft model'}, {zh:'speculative model／ngram／EAGLE', en:'a draft model, ngram, or EAGLE'}],
  [{zh:'accept / reject 迴圈', en:'the accept / reject loop'},
   {zh:'rejection sampler（分佈不變）', en:'the rejection sampler (distribution preserved)'}]
];
function vllmFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:VLLM_ROWS, title:'vLLM'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'beam search 已經不是取樣參數了', en:'beam search is not a sampling parameter any more'},
    panels:[{lbl:'BeamSearchParams', chips:[chip('separate API', 'hot')]}],
    msg:{zh:'vLLM 把 beam search 從 `SamplingParams` 搬到獨立的 `LLM.beam_search()`，理由正是第二個分頁那件事：它是對序列的搜尋，要自己管 w 條序列的 KV cache（前幾天的 block table 要處理分叉與回收），跟「每個 token 一條規則」不是同一種東西。',
         en:'vLLM moved beam search out of `SamplingParams` into a separate `LLM.beam_search()` call, for exactly the reason in tab two: it is a search over sequences that has to manage w sequences of KV cache - forks and frees in the earlier days\' block tables - rather than a per-token rule.'}});
  p({upto:2, line:10,
    cap:{zh:'而最便宜的 draft 是「根本沒有模型」', en:'and the cheapest draft is no model at all'},
    panels:[{lbl:'method', chips:[chip('ngram', 'ok')]},
            {lbl:'prompt_lookup_max', chips:[chip('4', 'dim')]}],
    msg:{zh:'`method: "ngram"` 就是這個檔案裡的 draft 模型在生產環境的樣子：它不跑任何神經網路，直接從 prompt 裡找相同的字串並把後面那一段當作提議。摘要、改程式碼、RAG 這些「答案大量重複輸入」的場合效果出奇地好，而且成本是零。EAGLE/Medusa 則是另一端：用 target 模型自己的 hidden state 訓一個小頭，接受率更高而不需要第二個完整模型。',
         en:'`method: "ngram"` is this file\'s draft model in production: it runs no network at all, it finds a matching span in the prompt and proposes what followed it. On summarisation, code editing and RAG - where the answer repeats long spans of the input - it works startlingly well and costs nothing. EAGLE and Medusa are the other end: a small head trained on the target model\'s own hidden states, higher acceptance without a second full model.'}});
  p({upto:3, line:7,
    cap:{zh:'而且接受規則就是上一個分頁證明過的那個', en:'and the accept rule is the one proved two tabs ago'},
    panels:[{lbl:'output', chips:[chip('bit-identical', 'ok')]}],
    msg:{zh:'所以 speculative decoding 可以預設打開：它不是「比較快但稍微不一樣」，而是同一個分佈。唯一會變的是延遲，以及在大 batch 下可能變慢（被丟掉的提議仍然花了算力）。',
         en:'That is why speculative decoding can be turned on by default: it is not "faster but slightly different", it is the same distribution. The only things that change are latency - and, at large batch sizes, possibly for the worse, because rejected proposals still cost compute.'}});
  return F.list;
}
const CODE_SGLANG = [
  'python -m sglang.launch_server --model ... \\',
  '    --speculative-algorithm EAGLE3 \\',
  '    --speculative-draft-model-path <eagle head> \\',
  '    --speculative-num-steps 3 \\        # draft depth',
  '    --speculative-eagle-topk 4 \\       # branches kept per step',
  '    --speculative-num-draft-tokens 8   # nodes verified per round',
  '',
  '# models with their own extra prediction head (DeepSeek-V3, GLM):',
  '    --speculative-algorithm NEXTN'
];
const SGLANG_ROWS = [
  [{zh:'一條長度 gamma 的猜測', en:'one guess of length gamma'},
   {zh:'一棵樹：topk 個分支 × num-steps 層', en:'a tree: topk branches, num-steps deep'}],
  [{zh:'驗證 gamma 個位置', en:'verify gamma positions'},
   {zh:'--speculative-num-draft-tokens 個節點', en:'--speculative-num-draft-tokens nodes'}],
  [{zh:'draft 模型', en:'the draft model'}, {zh:'EAGLE head／NEXTN（模型自帶）', en:'an EAGLE head, or NEXTN built into the model'}],
  [{zh:'tokens/round', en:'tokens per round'}, {zh:'accepted length（要看 batch size）', en:'accepted length, read against batch size'}]
];
function sglangFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:SGLANG_ROWS, title:'SGLang'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:5,
    cap:{zh:'SGLang：猜的不是一條，是一棵樹', en:'SGLang: the draft is a tree, not a chain'},
    panels:[{lbl:'--speculative-eagle-topk', chips:[chip('4', 'hot')]},
            {lbl:'--speculative-num-steps', chips:[chip('3', 'dim')]}],
    msg:{zh:'`--speculative-eagle-topk` 是關鍵差別：draft 每一層保留好幾個分支，形成一棵樹，然後用一個遮罩讓每個節點只看得到自己的祖先，**一次 target forward 驗證整棵樹**。同樣一次昂貴的 forward，候選變多了——跟第七個分頁的取捨一樣，只是形狀更好。',
         en:'`--speculative-eagle-topk` is the real difference: the draft keeps several branches at each level, forming a tree, and one target pass verifies the whole tree using a mask that lets every node attend only to its own ancestors. The same single expensive pass, more candidates in it - the trade from tab seven with a better shape.'}});
  p({upto:3, line:8,
    cap:{zh:'而要看的數字是 accepted length 對 batch size',
         en:'and the number to watch is accepted length against batch size'},
    panels:[{lbl:'small batch', chips:[chip('wins', 'ok')]},
            {lbl:'large batch', chips:[chip('can lose', 'bad')]}],
    msg:{zh:'`NEXTN` 是模型自己帶 MTP head（DeepSeek-V3、GLM）時的同一套機制，draft 內建在 target 裡。而伺服器上真正要看的不是接受率而是**每輪接受長度 vs batch size**：speculative decoding 是拿算力換記憶體頻寬，小 batch（decode 是頻寬瓶頸）會贏，大 batch 下 GPU 本來就忙，被丟掉的提議反而比省下的 forward 更貴。',
         en:'`NEXTN` is the same mechanism when the model ships its own extra prediction head (DeepSeek-V3, GLM): the draft is built into the target. On a server the number that matters is not the accept rate but the accepted length per round against the batch size: speculation spends compute to save memory bandwidth, so it wins at small batches where decode is bandwidth-bound and can lose at large ones, where the GPU was already busy and the discarded proposals cost more than the saved passes.'}});
  return F.list;
}


/* ================================================================ extra tab: where the beam loses it */
const CODE_LOST = [
  'best = exact_ranking(model)[0][1]        # the answer key',
  'for t in range(len(best)):',
  '    target = prefix + best[:t+1]         # the winning prefix at this step',
  '    cand = expand(beams)                 # everything a width-w beam could keep',
  '    rank = index of target in sorted(cand)',
  '    if rank > width: break               # dropped, and never recoverable'
];
function lostStage(o){
  const sh = [];
  sh.push(fitT(4.9, 1.05, {zh:'最佳句：' + JSON.stringify(LOST.best),
                           en:'the best sentence: ' + JSON.stringify(LOST.best)},
               {c:COL.grey, fs:.27}));
  const chars = LOST.best.slice(0, LOST.step + 3).split('');
  chars.forEach((c, i) => {
    const x = .80 + i * .72;
    const state = i < LOST.step - 1 ? 'ok' : (i === LOST.step - 1 ? 'bad' : 'ghost');
    sh.push(box(x, 3.70, .62, .70, state, c === ' ' ? '_' : c, .32));
  });
  sh.push(fitT(.80, 4.65, {zh:'最佳句的前綴，一個字元一個字元跟著走',
                           en:'the best sentence\'s prefix, one character at a time'},
               {anchor:'start', c:COL.tealL, fs:.27}));
  if (o.upto >= 1){
    sh.push(fitT(.80, 3.10, {zh:'第 ' + LOST.step + ' 步、前綴 ' + JSON.stringify(LOST.prefix) + ' 的分數排在 ' + LOST.cands + ' 個候選裡的第 ' + LOST.rank + ' 名，而 beam 只留 ' + LOST.width + ' 個',
                             en:'at step ' + LOST.step + ' the prefix ' + JSON.stringify(LOST.prefix) + ' ranks ' + LOST.rank + ' of ' + LOST.cands + ' candidates, and the beam keeps ' + LOST.width},
                 {anchor:'start', c:COL.red, fs:.27}));
  }
  if (o.upto >= 2){
    LOST.widths.forEach((r, i) => {
      const y = 2.35 - i * .55;
      sh.push(box(.80, y, 2.40, .46, r.found ? 'ok' : 'idle', 'width ' + r.w, .25));
      sh.push(fitT(3.40, y + .23, r.found ? {zh:'找到了', en:'finds it'} : {zh:'還是找不到', en:'still misses it'},
                   {anchor:'start', c:r.found ? COL.tealL : COL.grey, fs:.26}));
    });
  }
  return caps(sh, o);
}
function lostFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:lostStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:3,
    cap:{zh:'跟著最佳句的前綴走一遍', en:'follow the winning prefix, step by step'},
    panels:[{lbl:'width', chips:[chip(String(LOST.width), 'dim')]}],
    msg:{zh:'既然知道標準答案，就可以直接問：最佳句是在**哪一步**被 beam 丟掉的？做法是每一步都把最佳句的前綴拿去跟所有候選比分數。',
         en:'Since the answer key exists, the question can be asked directly: at which step does the beam drop the best sentence? Score the winning prefix against every candidate at each step and watch.'}});
  p({upto:1, line:5,
    cap:{zh:'第 ' + LOST.step + ' 步就掉了', en:'it is gone at step ' + LOST.step},
    panels:[{lbl:'step', chips:[chip(String(LOST.step), 'bad')]},
            {lbl:'rank there', chips:[chip(LOST.rank + ' / ' + LOST.cands, 'bad')]}],
    msg:{zh:'掉在 ' + JSON.stringify(LOST.prefix) + '——當下它只排第 ' + LOST.rank + '，而 beam 只留 ' + LOST.width + ' 個。這就是 beam search 真正的失敗模式：不是最後比錯，而是**一個暫時落後、本來會追上來的前綴**被剪掉，之後所有以它開頭的句子一起消失。「a small d」在那個字還沒寫完之前，看起來就是比「the old r」差。',
         en:'It dies on ' + JSON.stringify(LOST.prefix) + ', ranked ' + LOST.rank + ' when the beam keeps ' + LOST.width + '. That is the real failure mode: not a wrong final comparison but a prefix that is temporarily behind and would have caught up. Once it is pruned every sentence starting with it is gone too. "a small d" simply looks worse than "the old r" until the word is finished.'}});
  p({upto:2, line:5,
    cap:{zh:'要多寬才找得到？', en:'so how wide does it have to be?'},
    panels:[{lbl:'needs width', chips:[chip(String(LOST.widths[LOST.widths.length - 1].w), 'ok')]}],
    msg:{zh:'要 width ' + LOST.widths[LOST.widths.length - 1].w + ' 才找得到最佳句——是那個「只差一名」的 width 8 的 16 倍。最後一點點品質要用掉大部分的算力，而每多一份 width 就是多一份 KV cache、多一份 attention。',
         en:'It takes width ' + LOST.widths[LOST.widths.length - 1].w + ' to find the best sentence - sixteen times the beam that got within one rank of it. The last bit of quality costs most of the compute, and every unit of width is another KV cache and another pass of attention.'}});
  return F.list;
}

/* ================================================================ extra tab: one pass */
const CODE_ONE = [
  '# a causal model scores position i from the tokens to its LEFT only',
  'seq = context + all gamma proposals',
  'P = model(seq)                 # one pass',
  '',
  '# P[i] is the distribution for position i, computed as if the proposals',
  '# after it did not exist - the mask hides them.  Same shape as training.',
  'assert P[i] == model(context + proposals[:i])'
];
function oneStage(o){
  const sh = [];
  const toks = ONE.ctx.split('').slice(-5).concat(ONE.props);
  toks.forEach((c, i) => {
    const x = .80 + i * .92, isProp = i >= 5;
    sh.push(box(x, 4.05, .80, .64, isProp ? 'act' : 'idle', c === ' ' ? '_' : c, .30));
  });
  sh.push(fitT(.80, 4.95, {zh:'context（灰）＋ draft 的 ' + ONE.props.length + ' 個提議（紫）',
                           en:'context, then the draft\'s ' + ONE.props.length + ' proposals'},
               {anchor:'start', c:COL.grey, fs:.26}));
  if (o.upto >= 1){
    ONE.rows.forEach((r, i) => {
      const y = 3.20 - i * .52;
      const bonus = i === ONE.rows.length - 1;
      sh.push(box(.80, y, 3.40, .44, bonus ? 'hot' : 'ok',
                  {zh:'位置 ' + i + '：最高是 ' + JSON.stringify(r.top) + ' ' + r.p.toFixed(3),
                   en:'position ' + i + ': top ' + JSON.stringify(r.top) + ' at ' + r.p.toFixed(3)}, .24));
      if (bonus)
        sh.push(fitT(4.45, y + .22, {zh:'← 白賺的那一個', en:'<- the free bonus token'},
                     {anchor:'start', c:COL.tealL, fs:.25}));
    });
    sh.push(fitT(9.20, .95, {zh:'全部來自同一次 forward', en:'all from a single forward pass'},
                 {anchor:'end', c:COL.pale, fs:.28}));
  }
  return caps(sh, o);
}
function oneFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:oneStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:2,
    cap:{zh:'為什麼驗證 gamma 個 token 只要一次 forward', en:'why verifying gamma tokens costs one pass, not gamma'},
    panels:[{lbl:'proposals', chips:[chip(String(ONE.props.length), 'act')]}],
    msg:{zh:'把 context 加上全部提議接成一條序列送進模型。關鍵在 causal mask：位置 i 的輸出只看得到它左邊的 token，所以「長度 i 的前綴」已經包含在這條序列裡了。',
         en:'Concatenate the context with all the proposals and run the model once. The causal mask is what makes this legal: the output at position i only sees tokens to its left, so the prefix of length i is already inside the longer sequence.'}});
  p({upto:1, line:6,
    cap:{zh:'一次 forward，gamma+1 個分佈', en:'one pass, gamma+1 distributions'},
    panels:[{lbl:'distributions', chips:[chip(String(ONE.rows.length), 'ok')]},
            {lbl:'passes', chips:[chip('1', 'ok')]}],
    msg:{zh:'這跟訓練時的一次 forward（teacher forcing）是**同一個計算**——模型本來就不需要自己的輸出才能替一個 token 打分，只需要它左邊的 token。而 decode 慢是慢在「每一步把整個模型的權重讀一遍」，所以一次讀完驗 ' + ONE.rows.length + ' 個位置，就是把那次讀攤給更多 token。這也是為什麼它是拿算力換頻寬（最後一個分頁會算到 batch size）。',
         en:'This is the same computation as one training step on that sequence - teacher forcing - because the model never needed its own output to score a token, only the tokens to its left. And decode is slow because each step re-reads the entire model, so verifying ' + ONE.rows.length + ' positions per read amortises it. That is why this trades compute for bandwidth, which the last tab turns into a batch size.'}});
  return F.list;
}

/* ================================================================ extra tab: tree drafting */
const CODE_TREE = [
  '# chain: gamma positions, all spent on depth',
  'for _ in range(gamma): propose one token, verify, accept or stop',
  '',
  '# tree: some positions spent on siblings instead',
  'for k in branches:                  # e.g. (2, 2, 1, 1)',
  '    for _ in range(k):              # k candidates for THIS position',
  '        i = sample(q)',
  '        if random() < min(1, resid[i] / q[i]): accept; break',
  '        resid = normalise(maximum(resid - q, 0))   # sibling spent',
  '# still exactly the target distribution, with more shots per position'
];
function treeStage(o){
  const sh = [];
  if (o.upto === 0){
    const chain = [0, 1, 2, 3];
    chain.forEach((i) => {
      const x = .95 + i * 1.35;
      sh.push(box(x, 3.80, 1.10, .62, 'ok', 't' + (i + 1), .28));
      if (i) sh.push(S.e(x - .22, 4.11, x - .03, 4.11, {s:'ghost', arrow:true, w:.05}));
    });
    sh.push(fitT(.95, 4.70, {zh:'chain：4 個位置全花在深度', en:'chain: four positions, all depth'},
                 {anchor:'start', c:COL.tealL, fs:.27}));
    const tree = [[0, 2], [1, 2], [2, 1], [3, 1]];
    tree.forEach(([i, k]) => {
      const x = .95 + i * 1.35;
      for (let j = 0; j < k; j++)
        sh.push(box(x, 1.95 - j * .72, 1.10, .62, j ? 'act' : 'hot', 't' + (i + 1) + (j ? "'" : ''), .26));
    });
    sh.push(fitT(.95, 2.85, {zh:'tree：前兩個位置各留一個備胎', en:'tree: the first two positions each keep a spare'},
                 {anchor:'start', c:COL.tealL, fs:.27}));
  } else {
    TREE.forEach((r, i) => {
      const y = 4.20 - i * 1.15, top = 5.0;
      sh.push(fitT(.70, y + .45, {zh:'draft order ' + r.order, en:'draft order ' + r.order},
                   {anchor:'start', c:COL.tealL, fs:.27}));
      sh.push(S.r(2.90, y + .30, Math.max(.06, 4.2 * r.chain / top), .34, 'ok', null));
      sh.push(S.r(2.90, y - .14, Math.max(.06, 4.2 * r.tree / top), .34, 'act', null));
      sh.push(fitT(9.20, y + .47, 'chain ' + r.chain.toFixed(2), {anchor:'end', c:COL.pale, fs:.25}));
      sh.push(fitT(9.20, y + .03, 'tree ' + r.tree.toFixed(2), {anchor:'end', c:'#c7a6ff', fs:.25}));
    });
    sh.push(fitT(4.9, .80, {zh:'每輪吐出的 token 數（越長越好）',
                            en:'tokens emitted per round - longer is better'},
                 {c:COL.grey, fs:.27}));
  }
  return caps(sh, o);
}
function treeFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:treeStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:5,
    cap:{zh:'同樣的位置數，兩種花法', en:'the same verified positions, spent two ways'},
    panels:[{lbl:'chain', chips:[chip('4 deep', 'ok')]},
            {lbl:'tree', chips:[chip('2+2+1+1', 'act')]}],
    msg:{zh:'一次 target forward 能驗幾個位置是固定的預算。chain 把預算全花在「猜更遠」，tree 則留一部分給「同一個位置的備胎」——第一個候選被拒時，直接試兄弟節點，而且因為殘差分佈有被更新，輸出仍然**完全等於** target 分佈。',
         en:'The number of positions one target pass can verify is a fixed budget. A chain spends all of it on guessing further ahead; a tree keeps some of it for a spare candidate at the same position, so a rejected first candidate can be followed by a sibling - and because the residual distribution is updated in between, the output is still exactly the target distribution.'}});
  p({upto:1, line:9,
    cap:{zh:'誰贏，完全看 draft 多準', en:'which one wins depends entirely on the draft'},
    panels:[{lbl:'draft order 1', chips:[chip('tree ' + TREE[0].tree.toFixed(2), 'ok'),
                                         chip('chain ' + TREE[0].chain.toFixed(2), 'bad')]},
            {lbl:'draft order 3', chips:[chip('chain ' + TREE[2].chain.toFixed(2), 'ok'),
                                         chip('tree ' + TREE[2].tree.toFixed(2), 'bad')]}],
    msg:{zh:'order 1 的 draft 很爛，第一個候選常常被拒，所以備胎常常救到整輪：tree 每輪 ' + TREE[0].tree.toFixed(2) + ' 個 token vs chain 的 ' + TREE[0].chain.toFixed(2) + '，而且用掉的位置還比較少（每個位置換到 ' + TREE[0].treeNode.toFixed(2) + ' vs ' + TREE[0].chainNode.toFixed(2) + ' 個 token）。order 3 的 draft 很準，第一個候選幾乎都中，備胎就是白花的位置——chain 反而贏。這就是 EAGLE 那種 tree draft 的全部論點，也是它為什麼給你一個 topk 旋鈕而不是寫死：正確的形狀取決於你實際拿到的接受率。',
         en:'With an order-1 draft the first candidate is usually rejected, so the spare rescues the round: the tree emits ' + TREE[0].tree.toFixed(2) + ' tokens per round against ' + TREE[0].chain.toFixed(2) + ' for the chain, and uses fewer verified positions doing it (' + TREE[0].treeNode.toFixed(2) + ' tokens per position against ' + TREE[0].chainNode.toFixed(2) + '). With an order-3 draft the first candidate is usually right and every sibling is a position that bought nothing, so the chain wins. That is the whole argument for EAGLE-style tree drafting - and the reason it ships as a topk knob rather than a default: the right shape depends on the acceptance rate you actually get.'}});
  return F.list;
}

/* ================================================================ extra tab: batch size */
const CODE_BATCH = [
  'read = weight_bytes / bandwidth          # 140 GB / 5 TB/s = 28 ms',
  '',
  'plain = max(read, batch * flops_per_token / flops)',
  'spec  = max(read, batch * (gamma + 1) * flops_per_token / flops)',
  'speedup = tokens_per_round * plain / spec',
  '',
  '# small batch: the read dominates, extra positions are free',
  '# large batch: arithmetic dominates, rejected drafts cost real time'
];
function batchStage(o){
  const sh = [];
  const top = Math.max.apply(null, BATCH.rows.map(r => r.speedup));
  BATCH.rows.forEach((r, i) => {
    if (i > o.upto) return;
    const x = .95 + i * 1.85, h = Math.max(.08, 2.9 * r.speedup / top);
    sh.push(S.r(x, 1.70, 1.15, h, r.speedup >= 1 ? 'ok' : 'bad', null));
    sh.push(fitT(x + .57, 1.70 + h + .22, r.speedup.toFixed(2) + 'x',
                 {c:r.speedup >= 1 ? COL.tealL : COL.red, fs:.26}));
    sh.push(fitT(x + .57, 1.42, 'batch ' + r.b, {c:COL.grey, fs:.24}));
  });
  sh.push(S.e(.80, 1.70 + 2.9 / top, 9.40, 1.70 + 2.9 / top, {s:'ghost', arrow:false, w:.04}));
  sh.push(fitT(9.40, 1.70 + 2.9 / top + .22, {zh:'1.0x（不做 speculation）', en:'1.0x: no speculation'},
               {anchor:'end', c:COL.grey, fs:.24}));
  sh.push(fitT(.80, 4.95, {zh:'70B 級：140 GB 權重 ÷ 5 TB/s ＝ 28 ms 的讀取',
                           en:'70B-class: 140 GB of weights at 5 TB/s is a 28 ms read'},
               {anchor:'start', c:COL.grey, fs:.27}));
  sh.push(fitT(.80, 5.25, {zh:'每個 token 140 GFLOP ÷ 1 PFLOP/s',
                           en:'140 GFLOP per token at 1 PFLOP/s'},
               {anchor:'start', c:COL.grey, fs:.27}));
  return caps(sh, o);
}
function batchFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:batchStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:2, line:3,
    cap:{zh:'小 batch：多出來的位置是免費的', en:'small batch: the extra positions are free'},
    panels:[{lbl:'batch 1-32', chips:[chip(BATCH.rows[0].speedup.toFixed(2) + 'x', 'ok')]},
            {lbl:'step time', chips:[chip(BATCH.rows[0].base.toFixed(0) + ' ms', 'dim')]}],
    msg:{zh:'batch 小的時候，一次 decode step 的時間就是「把整個模型讀一遍」，GPU 的算力在空轉。這時候多驗幾個位置不花額外時間，所以加速幾乎等於每輪吐出的 token 數（' + BATCH.perRound.toFixed(2) + '）。',
         en:'At a small batch a decode step costs exactly one read of the whole model and the arithmetic units idle. Verifying more positions then costs no extra time, so the speedup is essentially the tokens emitted per round (' + BATCH.perRound.toFixed(2) + ').'}});
  p({upto:BATCH.rows.length - 1, line:7,
    cap:{zh:'大 batch：反而比不做還慢', en:'large batch: slower than not speculating'},
    panels:[{lbl:'batch 128', chips:[chip(BATCH.rows[3].speedup.toFixed(2) + 'x', 'act')]},
            {lbl:'batch 512', chips:[chip(BATCH.rows[4].speedup.toFixed(2) + 'x', 'bad')]}],
    msg:{zh:'batch 夠大之後，算術本身變成瓶頸，而 speculative decoding 把「每條序列一個位置」變成 gamma+1 個——多出來的算力是真的要花時間的，被拒絕的提議就是純浪費。這個設定在 batch 512 時只有 ' + BATCH.rows[4].speedup.toFixed(2) + 'x，也就是**比不做還慢**。所以伺服器會依照當下的 batch size 決定要不要開 speculation，而不是全域打開——前面幾張表是必要條件，不是充分條件。',
         en:'Past a batch size where arithmetic fills the read, the extra positions cost real time, and every rejected proposal is pure waste. At batch 512 this configuration runs at ' + BATCH.rows[4].speedup.toFixed(2) + 'x - slower than not speculating at all. That is why servers gate speculation on the running batch size instead of enabling it globally, and why the earlier tables are necessary but not sufficient.'}});
  return F.list;
}

const DAY_META = {
  title: {zh:'Beam search 與 speculative decoding', en:'Beam search and speculative decoding'},
  sub: {zh:'一個改變輸出去找更好的句子，一個一個位元都不改只求更快',
        en:'one changes the output to find a better sentence; the other changes nothing and only goes faster'},
  tabs: [
    {id:'gap', label:{zh:'逐步最大 ≠ 整句最大', en:'best step vs best sentence'},
     stage:{zh:'先有標準答案', en:'ground truth first'},
     view:VIEW, code:CODE_GT, build:gtFrames,
     idea:{zh:'文法只有 ' + GT.n.toLocaleString() + ' 句，全部精確算分之後：greedy 的句子排第 ' + GT.rank + '，比最好的那句低 ' + Math.exp(GT.bestLp - GT.greedyLp).toFixed(2) + ' 倍。這個落差調 temperature 補不起來，因為它是搜尋問題不是取樣問題。',
           en:'Only ' + GT.n.toLocaleString() + ' sentences exist, so all of them can be scored exactly: greedy lands at rank ' + GT.rank + ', ' + Math.exp(GT.bestLp - GT.greedyLp).toFixed(2) + 'x less likely than the best. No temperature setting closes that gap, because it is a search problem rather than a sampling one.'},
     legend:[['#12b3b8', {zh:'整句最好的', en:'the best sentence'}],
             ['#ff9736', {zh:'greedy 給的', en:'what greedy gives'}],
             ['#ff5c5c', {zh:'它的名次', en:'where it ranks'}]]},
    {id:'beam', label:{zh:'beam search', en:'beam search'},
     stage:{zh:'留 w 個前綴', en:'keep w prefixes alive'},
     view:VIEW, code:CODE_BEAM, build:beamFrames,
     idea:{zh:'width 1 ＝ greedy（這個模型有真平手，排序必須 stable）。加寬到 ' + BEAM[2].w + ' 爬到第 ' + BEAM[2].rank + ' 名、到 ' + BEAM[3].w + ' 爬到第 ' + BEAM[3].rank + ' 名，然後停住——最佳句的前綴早就被剪掉了。beam search 是啟發式不是最大化器，而成本線性：width w ＝ 每步跑 w 條序列、w 份 KV cache。',
           en:'Width 1 is greedy (this model has real ties, so the sort must be stable). Widening to ' + BEAM[2].w + ' reaches rank ' + BEAM[2].rank + ' and to ' + BEAM[3].w + ' reaches rank ' + BEAM[3].rank + ', then stops: the best sentence\'s prefix was pruned early. A beam is a heuristic, not a maximiser, and it costs w sequences and w KV caches at every step.'},
     legend:[['#ff9736', {zh:'目前這個 width', en:'the width being shown'}],
             ['#12b3b8', {zh:'已經掃過的', en:'widths already shown'}]]},
    {id:'lost', label:{zh:'它在哪一步輸掉', en:'where it loses the winner'},
     stage:{zh:'被剪掉的是一個會追上來的前綴', en:'the pruned prefix was going to catch up'},
     view:VIEW, code:CODE_LOST, build:lostFrames,
     idea:{zh:'有標準答案就可以直接問「最佳句是在哪一步被丟掉的」：第 ' + LOST.step + ' 步、前綴 ' + JSON.stringify(LOST.prefix) + '，當下排第 ' + LOST.rank + ' 而 beam 只留 ' + LOST.width + ' 個。要 width ' + LOST.widths[LOST.widths.length - 1].w + ' 才找得到——最後一點品質要用掉大部分算力。',
           en:'With an answer key the question becomes exact: the best sentence is dropped at step ' + LOST.step + ' on the prefix ' + JSON.stringify(LOST.prefix) + ', ranked ' + LOST.rank + ' when the beam keeps ' + LOST.width + '. It takes width ' + LOST.widths[LOST.widths.length - 1].w + ' to find it - the last bit of quality costs most of the compute.'},
     legend:[['#12b3b8', {zh:'還活著的前綴', en:'prefix still alive'}],
             ['#ff5c5c', {zh:'被剪掉的那一步', en:'the step where it is pruned'}]]},
    {id:'curse', label:{zh:'beam 的兩個毛病', en:'two things beams get wrong'},
     stage:{zh:'更寬不一定更好，而且偏好短句', en:'wider is not always better, and shorter always wins'},
     view:VIEW, code:CODE_CURSE, build:curseFrames,
     idea:{zh:'(1) 在小語料的模型上，width ' + MONO.dropAt + ' 比 width ' + (MONO.dropAt - 1) + ' 更差——beam 不是 width 的單調函數（翻譯界的 beam search curse）。(2) 每多一個字元就乘一個小於 1 的數，所以「總機率最高」夾帶「比較短」：總分前 ' + LEN.top + ' 名平均 ' + LEN.topTotal + ' 字元 vs 每字元前 ' + LEN.top + ' 名的 ' + LEN.topChar + '。修正是除以 length**alpha，一個純 heuristic。',
           en:'(1) On a model trained on a smaller corpus, width ' + MONO.dropAt + ' is worse than width ' + (MONO.dropAt - 1) + ': a beam is not monotone in its width, the effect translation calls the beam search curse. (2) Every character multiplies the score by something below 1, so highest total probability smuggles in shortest: the top ' + LEN.top + ' by total average ' + LEN.topTotal + ' characters against ' + LEN.topChar + ' when scored per character. The patch is dividing by length**alpha, a pure heuristic.'},
     legend:[['#12b3b8', {zh:'比前一個寬度好', en:'better than the narrower beam'}],
             ['#ff5c5c', {zh:'反而更差', en:'worse than the narrower beam'}]]},
    {id:'div', label:{zh:'beam 不會給你選擇', en:'a beam gives neighbours, not options'},
     stage:{zh:'n 個最可能的答案長得像同一句', en:'the n most probable answers are one sentence, varied'},
     view:VIEW, code:CODE_DIV, build:divFrames,
     idea:{zh:DIV.n + ' 句全都不一樣，所以數「幾句不重複」沒意義；數主詞就看得出來：beam 只有 ' + DIV.beamSubj + ' 個不同主詞，抽樣有 ' + DIV.sampSubj + ' 個。beam 在找最可能的輸出，而最可能的輸出本來就彼此相似——所以它適合只有一個正確答案的任務（翻譯、語音辨識），聊天要用前一天的取樣。',
           en:'All ' + DIV.n + ' sentences differ, so counting distinct strings says nothing; counting subjects does: the beam has ' + DIV.beamSubj + ' distinct subjects against ' + DIV.sampSubj + ' for sampling. A beam searches for the most probable output and the most probable outputs resemble each other, which is why it belongs where one answer is correct - translation, speech recognition - and chat needs the previous day\'s sampling.'},
     legend:[['#ff9736', {zh:'beam 的答案', en:'the beam\'s answers'}],
             ['#12b3b8', {zh:'抽樣的答案', en:'sampled answers'}]]},
    {id:'spec', label:{zh:'speculative decoding', en:'speculative decoding'},
     stage:{zh:'猜 4 個，一次驗完', en:'guess four, verify in one pass'},
     view:VIEW, code:CODE_SPEC, build:specFrames,
     idea:{zh:'decode 慢是因為每一步都要把整個模型的權重讀一遍。所以讓便宜的 draft 先猜 gamma 個，再用**一次** target forward 同時驗證這 gamma 個位置：逐個以 min(1, p/q) 接受，第一個被拒的地方改從 (p−q) 的正部抽一個並結束這一輪；全中的話還白賺一個 token。',
           en:'Decode is slow because every step re-reads the whole model from memory. So let a cheap draft guess gamma tokens and verify all gamma positions in ONE target pass: accept each with probability min(1, p/q), replace the first rejection with a draw from the normalised positive part of (p - q), and collect a free bonus token when everything is accepted.'},
     legend:[['#9d6bff', {zh:'draft 的提議', en:'the draft\'s proposals'}],
             ['#12b3b8', {zh:'接受', en:'accepted'}],
             ['#ff5c5c', {zh:'拒絕並修補', en:'rejected and repaired'}]]},
    {id:'onepass', label:{zh:'為什麼只要一次 forward', en:'why one pass is enough'},
     stage:{zh:'causal mask 讓 gamma+1 個分佈一起出來', en:'the causal mask gives gamma+1 distributions at once'},
     view:VIEW, code:CODE_ONE, build:oneFrames,
     idea:{zh:'把 context 接上全部提議送進去一次：位置 i 的輸出只看得到左邊，所以「長度 i 的前綴」已經在這條序列裡了，' + ONE.rows.length + ' 個分佈一次算完。這跟訓練時的 teacher forcing 是同一個計算。decode 慢在每步重讀整個模型的權重，一次讀完驗 ' + ONE.rows.length + ' 個位置就是把那次讀攤掉。',
           en:'Feed the context plus every proposal through once: the output at position i only sees tokens to its left, so the prefix of length i is already inside the sequence and all ' + ONE.rows.length + ' distributions come out of the same pass. It is the same computation as teacher forcing in training. Decode is bound by re-reading the weights every step, so verifying ' + ONE.rows.length + ' positions per read is what amortises it.'},
     legend:[['#9d6bff', {zh:'draft 的提議', en:'the draft proposals'}],
             ['#12b3b8', {zh:'一次算出的分佈', en:'distributions from one pass'}],
             ['#ff9736', {zh:'白賺的那一個', en:'the free bonus token'}]]},
    {id:'exact', label:{zh:'為什麼這是免費的', en:'why this is free'},
     stage:{zh:'輸出分佈一個位元都沒變', en:'the output distribution does not move at all'},
     view:VIEW, code:CODE_EXACT, build:exactFrames,
     idea:{zh:'q(x)·min(1,p/q) ＋ P(拒絕)·norm(p−q)+(x) ＝ p(x)，實測誤差 1e-16（浮點等級）。而兩個聽起來等價的捷徑都會搬動分佈：「直接信 draft」最多 ' + Math.max.apply(null, EX.map(r => r.trust)).toFixed(2) + '、「不擲硬幣」最多 ' + Math.max.apply(null, EX.map(r => r.nocoin)).toFixed(2) + '——而且都不會噴錯，只會讓模型默默變差。',
           en:'q(x)*min(1, p/q) + P(reject)*norm(p - q)+(x) = p(x), with a measured error of 1e-16 - floating point. Two shortcuts that sound equivalent both move the distribution: trusting the draft by up to ' + Math.max.apply(null, EX.map(r => r.trust)).toFixed(2) + ' and dropping the coin flip by up to ' + Math.max.apply(null, EX.map(r => r.nocoin)).toFixed(2) + '. Neither raises an error; the model just quietly gets worse.'},
     legend:[['#12b3b8', {zh:'正確規則的誤差', en:'error of the correct rule'}],
             ['#ff5c5c', {zh:'捷徑的誤差', en:'error of the shortcuts'}]]},
    {id:'knobs', label:{zh:'兩個旋鈕', en:'the two knobs'},
     stage:{zh:'draft 多好，和猜多遠', en:'how good the draft is, and how far ahead'},
     view:VIEW, code:CODE_KNOBS, build:knobsFrames,
     idea:{zh:'draft 從 order 1 到 5，接受率 ' + (DRAFT[0].acc * 100).toFixed(0) + '% → ' + (DRAFT[DRAFT.length - 1].acc * 100).toFixed(0) + '%，而合文法率完全不動（這正是重點）。gamma=8 實測 ' + GAM[GAM.length - 1].per.toFixed(2) + ' token/輪，而「獨立接受」的理論只有 ' + GAM[GAM.length - 1].theory.toFixed(2) + '——因為接受是相關的：draft 在可預測的整段都對，只在分岔點錯。',
           en:'Moving the draft from order 1 to order 5 takes the accept rate from ' + (DRAFT[0].acc * 100).toFixed(0) + '% to ' + (DRAFT[DRAFT.length - 1].acc * 100).toFixed(0) + '% while the grammatical rate does not move at all, which is the whole point. At gamma=8 the measurement is ' + GAM[GAM.length - 1].per.toFixed(2) + ' tokens per round against ' + GAM[GAM.length - 1].theory.toFixed(2) + ' predicted by the independence assumption - acceptances are correlated: the draft is right through predictable stretches and wrong at branch points.'},
     legend:[['#ff9736', {zh:'接受率', en:'accept rate'}],
             ['#12b3b8', {zh:'每輪 token 數（實測）', en:'tokens per round, measured'}],
             ['#8fa3ac', {zh:'理論值', en:'what the formula predicts'}]]},
    {id:'tree', label:{zh:'猜一棵樹', en:'draft a tree'},
     stage:{zh:'同樣的位置預算，深度還是備胎', en:'the same budget: depth, or spares'},
     view:VIEW, code:CODE_TREE, build:treeFrames,
     idea:{zh:'一次 forward 能驗的位置是固定預算。chain 全花在深度，tree 留一部分給同一位置的備胎（殘差更新後仍然完全等於 target 分佈）。draft 爛的時候 tree 贏（order 1：' + TREE[0].tree.toFixed(2) + ' vs ' + TREE[0].chain.toFixed(2) + ' token/輪），draft 準的時候 chain 贏（order 3：' + TREE[2].chain.toFixed(2) + ' vs ' + TREE[2].tree.toFixed(2) + '）。這就是 EAGLE tree draft 要給 topk 旋鈕的原因。',
           en:'One pass verifies a fixed budget of positions. A chain spends all of it on depth; a tree keeps some for a spare at the same position, and the residual update keeps the output exactly on the target distribution. A bad draft makes the tree win (order 1: ' + TREE[0].tree.toFixed(2) + ' against ' + TREE[0].chain.toFixed(2) + ' tokens per round) and a good draft makes the chain win (order 3: ' + TREE[2].chain.toFixed(2) + ' against ' + TREE[2].tree.toFixed(2) + '). That is why EAGLE-style drafting ships a topk knob.'},
     legend:[['#12b3b8', {zh:'chain', en:'chain'}],
             ['#9d6bff', {zh:'tree（含備胎）', en:'tree, with spares'}]]},
    {id:'batch', label:{zh:'什麼時候反而變慢', en:'when it stops paying'},
     stage:{zh:'拿算力換頻寬，而算力會用完', en:'compute for bandwidth, until the compute runs out'},
     view:VIEW, code:CODE_BATCH, build:batchFrames,
     idea:{zh:'70B 級：140 GB 權重 ÷ 5 TB/s ＝ 28 ms 的讀取。小 batch 時 GPU 在等權重，多驗幾個位置免費，加速 ≈ 每輪 token 數 ' + BATCH.perRound.toFixed(2) + '；batch 128 剩 ' + BATCH.rows[3].speedup.toFixed(2) + 'x、batch 512 只有 ' + BATCH.rows[4].speedup.toFixed(2) + 'x——**比不做還慢**。所以 speculation 要依當下 batch size 開關。',
           en:'A 70B-class model reads 140 GB at 5 TB/s, a 28 ms step. At small batches the GPU waits for weights, so extra positions are free and the speedup is the tokens per round (' + BATCH.perRound.toFixed(2) + '). At batch 128 it is ' + BATCH.rows[3].speedup.toFixed(2) + 'x and at 512 only ' + BATCH.rows[4].speedup.toFixed(2) + 'x - slower than not speculating. Hence gating speculation on the running batch size.'},
     legend:[['#12b3b8', {zh:'比不做快', en:'faster than not speculating'}],
             ['#ff5c5c', {zh:'比不做慢', en:'slower than not speculating'}]]},
    {id:'sglang', label:{zh:'在 SGLang 裡', en:'in SGLang'},
     stage:{zh:'猜一棵樹，而不是一條鏈', en:'draft a tree instead of a chain'},
     view:VIEW, code:CODE_SGLANG, build:sglangFrames,
     idea:{zh:'`--speculative-eagle-topk` 讓 draft 變成一棵樹：每層留幾個分支，再用祖先遮罩一次驗證整棵樹——同一次昂貴 forward 裡塞更多候選。`NEXTN` 是模型自帶 MTP head（DeepSeek-V3、GLM）的版本。伺服器上要看的是 accepted length 對 batch size：拿算力換頻寬，小 batch 贏、大 batch 可能輸。',
           en:'`--speculative-eagle-topk` turns the draft into a tree: several branches per level, all verified in one pass with an ancestor mask, so a single expensive forward carries more candidates. `NEXTN` is the version for models that ship their own MTP head (DeepSeek-V3, GLM). On a server, read accepted length against batch size: this spends compute to save bandwidth, so it wins at small batches and can lose at large ones.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'SGLang 裡的名字', en:'what SGLang calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'vllm', label:{zh:'在 vLLM 裡', en:'in vLLM'},
     stage:{zh:'beam 搬家了，draft 可以沒有模型', en:'beam moved out, and the draft can be no model'},
     view:VIEW, code:CODE_VLLM, build:vllmFrames,
     idea:{zh:'beam search 已從 `SamplingParams` 移到獨立的 `LLM.beam_search()`——它要管 w 條序列的 KV cache，跟每 token 的規則不同類。speculative decoding 在伺服器端設定，而 `method: "ngram"` 根本不跑模型：直接從 prompt 裡抄後續，摘要／改碼／RAG 特別有效。',
           en:'Beam search moved from `SamplingParams` into a separate `LLM.beam_search()`, because it manages w sequences of KV cache rather than applying a per-token rule. Speculative decoding is server configuration, and `method: "ngram"` runs no model at all - it copies continuations out of the prompt, which is startlingly effective on summarisation, code editing and RAG.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'vLLM 裡的名字', en:'what vLLM calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
  ]
};
