// DAY: 51
// TITLE_ZH: BPE Tokenizer 從零實作：把一句話變成模型真正看到的那串整數
// TITLE_EN: Building a BPE tokenizer from scratch: turning a sentence into the integers a model actually sees
// SUB_ZH: 模型從來沒看過你的文字，它看到的是一串整數，而產生那串整數的東西替你做了一連串你沒做過的決定：詞在哪裡斷、開頭那個空格算不算詞的一部分、數字怎麼被切、以及你這句話要花掉幾個整數。訓練只有三行：數每一對相鄰符號、把最常出現的那一對合併、重複。而合併的順序就是 tokenizer 本身。
// SUB_EN: A model never sees your text. It sees a list of integers, and the thing that produced that list made a series of choices you did not: where words break, whether a leading space belongs to the next word, how a number is cut up, and how many integers your sentence costs. Training is three lines - count every adjacent pair, merge the most frequent one, repeat - and the order of those merges is the tokenizer.
// FOLDER: day%2051%20-%20bpe%20tokenizer%20from%20scratch
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
const TRACE = [{"n":1, "a":" ", "b":"t", "cnt":41, "piece":" t"}, {"n":2, "a":"e", "b":"n", "cnt":29, "piece":"en"}, {"n":3, "a":"e", "b":"r", "cnt":29, "piece":"er"}, {"n":4, "a":" ", "b":"i", "cnt":26, "piece":" i"}, {"n":5, "a":" t", "b":"h", "cnt":21, "piece":" th"}, {"n":6, "a":"c", "b":"o", "cnt":20, "piece":"co"}, {"n":7, "a":"o", "b":"k", "cnt":17, "piece":"ok"}, {"n":8, "a":"ok", "b":"en", "cnt":16, "piece":"oken"}];
const WORDS = [{"w":" tokens", "ids":[286], "pieces":[" tokens"]}, {"w":"tokens", "ids":[116, 263, 115], "pieces":["t", "oken", "s"]}, {"w":" the", "ids":[266], "pieces":[" the"]}, {"w":"the", "ids":[318], "pieces":["the"]}];
const SENT = {"text":"the cost of a model call is counted in tokens", "ids":[318, 377, 294, 264, 319, 384, 328, 269, 287, 289, 275, 286], "pieces":["the", " cost", " of", " a", " model", " ca", "ll", " is", " count", "ed", " in", " tokens"], "words":10, "bytes":45, "chars":45};
const WALK = {"word":" counted", "steps":[{"rank":null, "pieces":[" ", "c", "o", "u", "n", "t", "e", "d"]}, {"rank":6, "a":"c", "b":"o", "pieces":[" ", "co", "u", "n", "t", "e", "d"]}, {"rank":15, "a":" ", "b":"co", "pieces":[" co", "u", "n", "t", "e", "d"]}, {"rank":19, "a":"n", "b":"t", "pieces":[" co", "u", "nt", "e", "d"]}, {"rank":25, "a":"u", "b":"nt", "pieces":[" co", "unt", "e", "d"]}, {"rank":32, "a":" co", "b":"unt", "pieces":[" count", "e", "d"]}, {"rank":34, "a":"e", "b":"d", "pieces":[" count", "ed"]}]};
const TIES = {"steps":143, "ties":123, "step":2, "count":29, "examples":["en", "er"], "howmany":2};
const UTF8 = {"text":"模型看到的", "bytes":15, "n":13, "bad":12, "cells":[{"hex":"e6", "txt":null}, {"hex":"a8", "txt":null}, {"hex":"a1", "txt":null}, {"hex":"e5", "txt":null}, {"hex":"9e", "txt":null}, {"hex":"8b", "txt":null}, {"hex":"e7", "txt":null}, {"hex":"9c", "txt":null}, {"hex":"8b", "txt":null}, {"hex":"e5", "txt":null}, {"hex":"88", "txt":null}, {"hex":"b0", "txt":null}, {"hex":null, "txt":"的"}], "joined":"模型看到的", "perToken":"������������的"};
const PRETOK = {"no":{"merges":144, "tokens":502, "ratio":0.3934, "spaced":12, "examples":[" is not ", "t in", "t is ", " the tokeniz"]}, "yes":{"merges":143, "tokens":500, "ratio":0.3918, "spaced":0}, "bytes":1276};
const CURVE = {"rows":[{"vocab":256, "merges":0, "tokens":1276, "ratio":1.0}, {"vocab":300, "merges":44, "tokens":767, "ratio":0.6011}, {"vocab":350, "merges":94, "tokens":598, "ratio":0.4687}, {"vocab":399, "merges":143, "tokens":500, "ratio":0.3918}], "bytes":1276, "huff":4.6928};
const LC692 = {"ex1":["i", "love", "leetcode", "i", "love", "coding"], "ans1":["i", "love"], "ex2":["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], "ans2":["the", "is", "sunny", "day"]};
const LC394 = {"s":"3[a2[c]]", "steps":[{"ch":"3", "cur":"", "num":3, "stack":[]}, {"ch":"[", "cur":"", "num":0, "stack":[["", 3]]}, {"ch":"a", "cur":"a", "num":0, "stack":[["", 3]]}, {"ch":"2", "cur":"a", "num":2, "stack":[["", 3]]}, {"ch":"[", "cur":"", "num":0, "stack":[["", 3], ["a", 2]]}, {"ch":"c", "cur":"c", "num":0, "stack":[["", 3], ["a", 2]]}, {"ch":"]", "cur":"acc", "num":0, "stack":[["", 3]]}, {"ch":"]", "cur":"accaccacc", "num":0, "stack":[]}], "ans":"accaccacc"};

const SP = s => s.replace(/ /g, '␣');     // show a space inside a token box

function tokenStrip(sh, x0, y, pieces, sts, cw, ch, fs){
  cw = cw || .62; ch = ch || .56;
  let x = x0;
  pieces.forEach((t, i) => {
    const lab = SP(t), w = Math.max(cw, .26 + .30 * lab.length);
    sh.push(box(x, y, w, ch, (sts && sts[i]) || 'idle', lab, fs || .30));
    x += w + .09;
  });
  return x;
}
function stripWidth(pieces, cw){
  cw = cw || .62;
  return pieces.reduce((a, t) => a + Math.max(cw, .26 + .30 * SP(t).length) + .09, 0) - .09;
}

/* ================================================================ tab 1: the price of a space */
const CODE_WORDS = [
  "encode(' tokens')  ->  [286]",
  "encode('tokens')   ->  [116, 263, 115]",
  '',
  '# same five letters.  the version with a leading space is one',
  '# token because that is how the word appears in running text,',
  '# and the merge for it was learned.  without the space it was',
  '# never frequent enough to merge.'
];
function wordsStage(o){
  const sh = [];
  WORDS.slice(0, o.upto + 1).forEach((w, r) => {
    const y = 1.35 + r * 1.00;
    const st = w.ids.length === 1 ? 'ok' : 'bad';
    sh.push(fitT(2.25, y + .30, JSON.stringify(w.w), {anchor:'end', c:w.ids.length === 1 ? COL.tealL : COL.red, fs:.30}));
    const end = tokenStrip(sh, 2.45, y, w.pieces, w.pieces.map(() => st));
    sh.push(fitT(end + .25, y + .30,
      {zh:w.ids.length + ' 個 token', en:w.ids.length + (w.ids.length > 1 ? ' tokens' : ' token')},
      {anchor:'start', c:w.ids.length === 1 ? COL.tealL : COL.red, fs:.28}));
  });
  return caps(sh, o);
}
function wordsFrames(){
  const F = new Frames();
  const msgs = [
    {zh:"' tokens'（前面有一個空格）是一個 token，id 286。這個 tokenizer 在語料裡看過夠多次「空格＋tokens」，所以把它合併成了一個符號。",
     en:"' tokens', with a leading space, is a single token - id 286. The tokenizer saw that exact string often enough in the corpus to merge it into one symbol."},
    {zh:"拿掉那個空格，同樣五個字母變成 3 個 token。對 tokenizer 來說開頭的空格不是空白，它是詞的一部分，因為詞在真實文字裡就是長這樣出現的。",
     en:"Take the space away and the same five letters cost 3 tokens. A leading space is not whitespace to a tokenizer - it is part of the word, because that is how the word appears in running text."},
    {zh:"' the' 也是一個 token。",
     en:"' the' is one token too."},
    {zh:"'the' 也是一個 token——因為它常常出現在句首。哪些詞有「無空格版本」完全取決於語料，不是規則。這就是為什麼同一句話在不同模型會花掉不同數量的 token。",
     en:"'the' is also one token - because it turns up at the start of sentences often enough. Which words get a no-space version is decided by the corpus, not by a rule, and that is why the same sentence costs a different number of tokens in every model."}];
  WORDS.forEach((w, i) => {
    F.push({shapes:wordsStage({upto:i, cap:{zh:'同樣的字母，不同的價錢', en:'the same letters, a different price'}}),
      panels:[{lbl:'input', chips:[chip(JSON.stringify(w.w), 'hot')]},
              {lbl:'token ids', chips:w.ids.map(id => chip(String(id), w.ids.length === 1 ? 'ok' : 'bad'))}],
      line:i < 2 ? i : 3, msg:msgs[i], view:VIEW});
  });
  return F.list;
}

/* ================================================================ tab 2: one sentence */
const CODE_SENT = [
  'ids = encode(sentence, merges)',
  '',
  '# 10 words.  45 characters.  45 bytes.  12 tokens.',
  '# none of those four numbers can be computed from another,',
  '# and only the last one is what you are charged for.'
];
function sentStage(o){
  const sh = [];
  const n = o.upto == null ? SENT.pieces.length : o.upto + 1;
  const raw = SENT.pieces.map(t => Math.max(.46, .22 + .28 * SP(t).length));
  const total = raw.reduce((a, b) => a + b, 0) + .07 * (SENT.pieces.length - 1);
  const k = Math.min(1, 9.0 / total);
  let x = (9.8 - total * k) / 2;
  SENT.pieces.forEach((t, i) => {
    if (i >= n) return;
    const bw = raw[i] * k;
    const st = (t.trim() === 'ca' || t.trim() === 'll') ? 'hot' : 'ok';
    sh.push(box(x, 2.05, bw, .62, st, SP(t), .26));
    sh.push(fitT(x + bw / 2, 3.00, String(SENT.ids[i]), {c:COL.grey, fs:.22}));
    x += bw + .07 * k;
  });
  side(sh, .60, 4.05, [
    [{zh:SENT.words + ' 個詞', en:SENT.words + ' words'}, COL.grey, .28],
    [{zh:SENT.chars + ' 個字元', en:SENT.chars + ' characters'}, COL.grey, .28],
    [{zh:SENT.bytes + ' 個位元組', en:SENT.bytes + ' bytes'}, COL.grey, .28],
    [{zh:n + ' 個 token', en:n + ' tokens'}, COL.tealL, .32]], .46);
  return caps(sh, o);
}
function sentFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:sentStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:0,
    cap:{zh:'"' + SENT.text + '"', en:'"' + SENT.text + '"'},
    panels:[{lbl:'first token', chips:[chip(SENT.pieces[0], 'ok'), chip(String(SENT.ids[0]), 'dim')]}],
    msg:{zh:'一個 token 一個整數。模型收到的是下面那排數字，不是上面那排字。',
         en:'One token, one integer. What the model receives is the row of numbers underneath, not the row of letters above.'}});
  p({upto:5, line:2,
    cap:{zh:'橘色那兩個合起來是 "call"', en:'the two amber boxes are the word "call"'},
    panels:[{lbl:'call', chips:[chip(' ca', 'hot'), chip('ll', 'hot')]}],
    msg:{zh:'"call" 被切成 " ca" 和 "ll" 兩個 token。這不是 bug，只是這份語料剛好沒有讓 " call" 合併起來——模型永遠不會看到 call 這個詞，它看到的是兩個碎片。',
         en:'"call" comes apart into " ca" and "ll". Nothing is broken; this corpus simply never merged " call" into one symbol. The model will never see the word call - it sees two fragments.'}});
  p({line:2,
    cap:{zh:'四個數字，沒有一個算得出另一個', en:'four numbers, and none of them gives you another'},
    panels:[{lbl:'words / chars', chips:[chip(String(SENT.words), 'dim'), chip(String(SENT.chars), 'dim')]},
            {lbl:'bytes / tokens', chips:[chip(String(SENT.bytes), 'dim'), chip(String(SENT.ids.length), 'ok')]}],
    msg:{zh:SENT.words + ' 個詞、' + SENT.chars + ' 個字元、' + SENT.bytes + ' 個位元組、' + SENT.ids.length + ' 個 token。數詞不是數 token，數字元也不是。唯一能數 token 的東西是 tokenizer 本身——而計費與 context window 用的都是最後那個數字。',
         en:SENT.words + ' words, ' + SENT.chars + ' characters, ' + SENT.bytes + ' bytes, ' + SENT.ids.length + ' tokens. Counting words is not counting tokens and neither is counting characters. The only thing that counts tokens is the tokenizer - and the last number is the one you are billed for and the one the context window measures.'}});
  return F.list;
}

/* ================================================================ tab 3: training */
const CODE_TRAIN = [
  'while len(vocab) < vocab_size:',
  '    counts = pair_counts(words)',
  '    best = max(counts, key=...)      # most frequent pair',
  '    if counts[best] < 2: break       # nothing repeats any more',
  '    vocab[new_id] = vocab[a] + vocab[b]',
  '    words = {merge(w, best): f for w, f in words.items()}',
  '    merges.append((best, new_id))    # ORDER is the tokenizer'
];
function trainStage(o){
  const sh = [];
  const rows = TRACE.slice(0, o.upto + 1);
  rows.forEach((t, r) => {
    const y = 1.05 + r * .54;
    sh.push(fitT(1.30, y + .25, '#' + t.n, {anchor:'end', c:COL.grey, fs:.26}));
    sh.push(box(1.45, y, 1.05, .48, r === rows.length - 1 ? 'hot' : 'done', SP(t.a), .28));
    sh.push(fitT(2.62, y + .25, '+', {c:COL.grey, fs:.26}));
    sh.push(box(2.80, y, 1.05, .48, r === rows.length - 1 ? 'hot' : 'done', SP(t.b), .28));
    sh.push(fitT(4.05, y + .25, '->', {c:COL.grey, fs:.26}));
    sh.push(box(4.30, y, 1.45, .48, r === rows.length - 1 ? 'ok' : 'idle', SP(t.piece), .28));
    sh.push(fitT(6.05, y + .25, {zh:t.cnt + ' 次', en:t.cnt + ' times'},
                 {anchor:'start', c:COL.purpleL, fs:.26}));
  });
  return caps(sh, o);
}
function trainFrames(){
  const F = new Frames();
  const msgs = [
    {zh:'第一輪：最常出現的相鄰組合是「空格 + t」，在語料裡出現 41 次。把它合併成一個新符號，vocabulary 多一個 id。',
     en:'Round one: the most frequent adjacent pair is space followed by t, 41 times in the corpus. Merge it into one new symbol and the vocabulary gains an id.'},
    {zh:'第二輪是 "en"，29 次。注意 "er" 也是 29 次——這裡已經平手了，下一個分頁會講這件事。',
     en:'Round two is "en" at 29. Note that "er" is also 29 - there is already a tie here, which is the next tab.'},
    {zh:'"er" 29 次。', en:'"er", 29 times.'},
    {zh:'"' + TRACE[3].piece + '" ' + TRACE[3].cnt + ' 次。到這裡合併的都還是單一位元組。',
     en:'"' + TRACE[3].piece + '" ' + TRACE[3].cnt + ' times. Everything merged so far is still a pair of single bytes.'},
    {zh:'第五輪把「' + TRACE[4].a + '」和「h」合起來——而「' + TRACE[4].a + '」本身是第一輪的產物。合併可以疊在合併上面，這就是 token 怎麼長長的。',
     en:'Round five merges "' + TRACE[4].a + '" with "h" - and "' + TRACE[4].a + '" is itself the output of round one. Merges stack on merges, and that is how tokens get long.'},
    {zh:'"co" 20 次。', en:'"co", 20 times.'},
    {zh:'"ok" 17 次。', en:'"ok", 17 times.'},
    {zh:'第八輪把 "ok" 和 "en" 合成 "oken"——兩個合併出來的符號再合併一次。繼續跑下去就會得到 " token"、" tokens" 這種整個詞的 token。',
     en:'Round eight merges "ok" with "en" to make "oken" - two merged symbols merging again. Keep going and you get whole-word tokens like " token" and " tokens".'}];
  TRACE.forEach((t, i) => {
    F.push({shapes:trainStage({upto:i, cap:{zh:'數每一對相鄰符號，合併最常出現的那一對，重複',
                                            en:'count every adjacent pair, merge the most frequent, repeat'}}),
      panels:[{lbl:'merge', chips:[chip(SP(t.a), 'hot'), chip(SP(t.b), 'hot'), chip(SP(t.piece), 'ok')]},
              {lbl:'occurrences', chips:[chip(String(t.cnt), 'act')]}],
      line:i === 0 ? 1 : (i === 7 ? 6 : 2), msg:msgs[i], view:VIEW});
  });
  return F.list;
}

/* ================================================================ tab 4: ties */
const CODE_TIES = [
  'best = max(counts.items(),',
  '           key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]',
  '',
  '# the first field is the count.  the other two are the',
  '# tie-break, and without them the vocabulary depends on',
  '# dictionary insertion order.',
  '# picking the k most frequent with a written-down tie-break',
  '# is LeetCode 692.'
];
function tiesStage(o){
  const sh = [], base = 4.20, maxh = 2.30, bw = 1.30;
  [[{zh:'合併步數', en:'merge steps'}, TIES.steps, 'ok'],
   [{zh:'其中有平手', en:'steps with a tie'}, TIES.ties, 'bad']].forEach((d, i) => {
    if (i > o.upto) return;
    const x = 2.70 + i * 2.60, h = maxh * d[1] / TIES.steps;
    sh.push(S.r(x, base - h, bw, h, d[2], null));
    sh.push(fitT(x + bw / 2, base - h - .24, String(d[1]), {c:d[2] === 'ok' ? COL.tealL : COL.red, fs:.32}));
    sh.push(fitT(x + bw / 2, base + .32, d[0], {c:COL.grey, fs:.26}));
  });
  if (o.showFirst){
    sh.push(fitT(4.9, 5.00, {zh:'第一次平手在第 ' + TIES.step + ' 步：' + TIES.howmany + ' 對都出現 ' + TIES.count + ' 次',
                             en:'the first tie is at step ' + TIES.step + ': ' + TIES.howmany + ' pairs all occur ' + TIES.count + ' times'},
                 {c:COL.orange, fs:.30}));
    tokenStrip(sh, 4.00, 5.25, TIES.examples, TIES.examples.map(() => 'hot'), .80, .50, .28);
  }
  return caps(sh, o);
}
function tiesFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:tiesStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:0,
    cap:{zh:'這份語料一共跑了 ' + TIES.steps + ' 個合併步驟', en:'this corpus takes ' + TIES.steps + ' merge steps'},
    panels:[{lbl:'merge steps', chips:[chip(String(TIES.steps), 'ok')]}],
    msg:{zh:'每一步都要挑「最常出現的那一對」。問題是，最常出現的那一對通常不只一個。',
         en:'Every step picks the most frequent pair. The problem is that the most frequent pair is usually not unique.'}});
  p({upto:1, line:0,
    cap:{zh:TIES.ties + ' 步有平手，佔 ' + Math.round(100 * TIES.ties / TIES.steps) + '%',
         en:TIES.ties + ' of them had a tie, ' + Math.round(100 * TIES.ties / TIES.steps) + '%'},
    panels:[{lbl:'steps with a tie', chips:[chip(String(TIES.ties), 'bad')]},
            {lbl:'share', chips:[chip(Math.round(100 * TIES.ties / TIES.steps) + '%', 'bad')]}],
    msg:{zh:TIES.steps + ' 步裡有 ' + TIES.ties + ' 步（' + Math.round(100 * TIES.ties / TIES.steps) + '%）在最高票上是平手的。小語料特別明顯，但大語料一樣會發生，只是比例低一些。',
         en:TIES.ties + ' of the ' + TIES.steps + ' steps (' + Math.round(100 * TIES.ties / TIES.steps) + '%) had more than one pair tied for most frequent. A small corpus makes it obvious; a large one has the same problem at a lower rate.'}});
  p({upto:1, showFirst:true, line:1,
    cap:{zh:'平手就是分岔', en:'a tie is a fork'},
    panels:[{lbl:'tied at ' + TIES.count, chips:TIES.examples.map(e => chip(SP(e), 'hot'))}],
    msg:{zh:'第 ' + TIES.step + ' 步就平手了：' + TIES.examples.map(e => '"' + e + '"').join(' 和 ') + ' 都出現 ' + TIES.count + ' 次。挑哪一個都可以，但挑完之後那一對附近的計數會變，兩個 tokenizer 從這裡開始分家——換別的平手規則重跑是 141–143 次合併、500–504 個 token，壓縮只差幾個 token，而對「什麼是一個 token」有不同的答案。',
         en:'Step ' + TIES.step + ' is already tied: ' + TIES.examples.map(e => '"' + e + '"').join(' and ') + ' both occur ' + TIES.count + ' times. Either choice is defensible, but afterwards the counts around that pair change and the two tokenizers diverge - other tie-breaks give 141 to 143 merges and 500 to 504 tokens, so they compress to within a few tokens of each other and disagree about what a token is.'}});
  p({upto:1, showFirst:true, line:4,
    cap:{zh:'所以平手規則是規格的一部分', en:'so the tie-break is part of the specification'},
    panels:[{lbl:'rule', chips:[chip('(count, -id_a, -id_b)', 'ok')]}],
    msg:{zh:'這份程式的規則是「取 id 最小的那一對」，寫在 max 的 key 裡。有這條規則，同一份語料跑兩次會得到同一個 vocabulary；沒有的話，vocabulary 取決於 dict 的插入順序，換一個 Python 版本就可能變。挑最高票又要有明確平手規則——那就是 LeetCode 692。',
         en:'This file takes the lowest pair of ids, written into the key of `max`. With that rule two runs on the same corpus give the same vocabulary; without it the vocabulary depends on dictionary insertion order. Picking the most frequent thing with a written-down tie-break is exactly LeetCode 692.'}});
  return F.list;
}

/* ================================================================ tab 5: encoding */
const CODE_ENC = [
  'while len(ids) >= 2:',
  '    pairs = set(zip(ids, ids[1:]))',
  '    best = min(pairs & rank.keys(), key=rank.get)',
  '    ids = merge(ids, best)',
  '',
  '# the SMALLEST rank, not the longest match.  longest-match',
  '# over the same vocabulary is a different algorithm with',
  '# different output - which is why the merge list ships with',
  '# the model, not just the token strings.'
];
function encStage(o){
  const sh = [];
  WALK.steps.slice(0, o.upto + 1).forEach((st, r) => {
    const y = .95 + r * .62;
    const lab = st.rank == null ? {zh:'位元組', en:'bytes'} : {zh:'合併 #' + st.rank, en:'merge #' + st.rank};
    sh.push(fitT(2.05, y + .26, lab, {anchor:'end', c:st.rank == null ? COL.grey : COL.orange, fs:.26}));
    const last = (r === o.upto);
    tokenStrip(sh, 2.25, y, st.pieces, st.pieces.map(() => last ? 'ok' : 'done'), .54, .52, .26);
  });
  return caps(sh, o);
}
function encFrames(){
  const F = new Frames();
  WALK.steps.forEach((st, i) => {
    const msg = st.rank == null
      ? {zh:"' counted' 先變成 " + st.pieces.length + " 個位元組。接下來每一步都問同一個問題：現在還在的相鄰組合裡，哪一個是最早被學到的？",
         en:"' counted' starts as " + st.pieces.length + ' bytes. Every step then asks the same question: of the adjacent pairs still present, which one was learned earliest?'}
      : {zh:'套用第 ' + st.rank + ' 號合併（"' + st.a + '" + "' + st.b + '"），剩下 ' + st.pieces.length + ' 個 token。挑的是「rank 最小」而不是「最長的那個」。',
         en:'Apply merge #' + st.rank + ' ("' + st.a + '" + "' + st.b + '"), leaving ' + st.pieces.length + ' tokens. It picks the smallest rank, not the longest match.'};
    F.push({shapes:encStage({upto:i, cap:{zh:"編碼 ' counted'：照學到的順序重播",
                                          en:"encoding ' counted': replay the merges in the order they were learned"}}),
      panels:[{lbl:'tokens', chips:[chip(String(st.pieces.length), 'ok')]},
              {lbl:'merge rank', chips:[chip(st.rank == null ? '-' : String(st.rank), 'hot')]}],
      line:st.rank == null ? 0 : 2, msg:msg, view:VIEW});
  });
  const last = WALK.steps[WALK.steps.length - 1];
  F.push({shapes:encStage({upto:WALK.steps.length - 1,
                           cap:{zh:'8 個位元組變成 ' + last.pieces.length + ' 個 token',
                                en:'8 bytes became ' + last.pieces.length + ' tokens'}}),
    panels:[{lbl:'result', chips:last.pieces.map(p => chip(SP(p), 'ok'))}],
    line:6,
    msg:{zh:'重點是「照順序重播」這件事本身。同一組 token 字串，如果改用「最長匹配」去切，會得到不同的答案——所以 tokenizer 要發佈的是合併清單，不是只有 token 的字串表。',
         en:'The point is the replay itself. Given the same set of token strings, a longest-match tokenizer produces different output - which is why what ships with a model is the ordered merge list, not just the table of token strings.'}, view:VIEW});
  return F.list;
}

/* ================================================================ tab 6: utf-8 */
const CODE_UTF8 = [
  '# right:',
  "b''.join(vocab[i] for i in ids).decode('utf-8')",
  '',
  '# wrong, and it does not raise:',
  "''.join(vocab[i].decode('utf-8', errors='replace') for i in ids)",
  '',
  '# a three-byte character is routinely split across two',
  '# tokens.  join the bytes first, decode once.'
];
function utf8Stage(o){
  const sh = [];
  let x = .70;
  UTF8.cells.forEach((c, i) => {
    if (i > o.upto) return;
    const ok = c.txt != null, w = ok ? .78 : .58;
    sh.push(box(x, 1.55, w, .60, ok ? 'ok' : 'bad', ok ? c.txt : c.hex, .28));
    x += w + .08;
  });
  if (o.showResult){
    sh.push(fitT(4.55, 3.15, {zh:'先接起來，再解一次', en:'join the bytes, decode once'},
                 {anchor:'end', c:COL.grey, fs:.28}));
    sh.push(fitT(4.80, 3.15, UTF8.joined, {anchor:'start', c:COL.tealL, fs:.34}));
    sh.push(fitT(4.55, 3.75, {zh:'一個一個解，再接起來', en:'decode each token, then join'},
                 {anchor:'end', c:COL.grey, fs:.28}));
    sh.push(fitT(4.80, 3.75, UTF8.perToken, {anchor:'start', c:COL.red, fs:.34}));
  }
  return caps(sh, o);
}
function utf8Frames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:utf8Stage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:2, line:7,
    cap:{zh:UTF8.text + ' 是 ' + UTF8.bytes + ' 個位元組、' + UTF8.n + ' 個 token',
         en:UTF8.text + ' is ' + UTF8.bytes + ' bytes and ' + UTF8.n + ' tokens'},
    panels:[{lbl:'bytes / tokens', chips:[chip(String(UTF8.bytes), 'dim'), chip(String(UTF8.n), 'ok')]}],
    msg:{zh:'一個中文字是 3 個位元組。這些 token 大多是半個字——紅色的格子是「單獨拿出來不是合法 UTF-8」的位元組。',
         en:'One Chinese character is three bytes. Most of these tokens are part of a character - the red boxes are bytes that are not valid UTF-8 on their own.'}});
  p({upto:UTF8.cells.length - 1, line:7,
    cap:{zh:UTF8.n + ' 個 token 裡有 ' + UTF8.bad + ' 個單獨不是合法 UTF-8',
         en:UTF8.bad + ' of the ' + UTF8.n + ' tokens are not valid UTF-8 on their own'},
    panels:[{lbl:'not valid alone', chips:[chip(UTF8.bad + ' / ' + UTF8.n, 'bad')]}],
    msg:{zh:'三個位元組的字被切在 token 邊界上是常態，不是錯誤。錯誤發生在你怎麼把它接回來。',
         en:'A three-byte character being split across a token boundary is normal, not a bug. The bug is in how you put it back together.'}});
  p({upto:UTF8.cells.length - 1, showResult:true, line:1,
    cap:{zh:'同一串 token，兩種接法', en:'the same tokens, two ways of putting them back'},
    panels:[{lbl:'join then decode', chips:[chip(UTF8.joined, 'ok')]},
            {lbl:'decode then join', chips:[chip(UTF8.perToken, 'bad')]}],
    msg:{zh:'先把所有位元組接起來、再 decode 一次，就拿回原文。反過來一個一個 decode 再接起來，就得到一排替換字元——而 errors="replace" 正是它不丟例外的原因。逐 token 串流輸出最常撞到這個：正確的做法是把還沒構成完整字元的位元組先留著，而不是用 replace 蓋過去。',
         en:'Join all the bytes and decode once and you get the text back. Decode each token first and you get a row of replacement characters - and errors="replace" is exactly why it does not raise. Streaming output token by token is where this usually shows up; the fix is to hold back bytes that do not yet form a complete character, not to replace them.'}});
  return F.list;
}

/* ================================================================ tab 7: LeetCode */
const CODE_692 = [
  'count = Counter(words)',
  'return sorted(count, key=lambda w: (-count[w], w))[:k]',
  '',
  '# one key, both rules: falling on the count, rising on the',
  '# word.  the heap version has to invert BOTH fields at once,',
  '# which is why a plain tuple does not work.'
];
const CODE_394 = [
  'stack, cur, num = [], "", 0',
  'for ch in s:',
  '    if ch.isdigit():  num = num * 10 + int(ch)',
  '    elif ch == "[":   stack.append((cur, num)); cur, num = "", 0',
  '    elif ch == "]":   prev, times = stack.pop(); cur = prev + cur * times',
  '    else:             cur += ch',
  'return cur'
];
function lcStage(o){
  const sh = [];
  if (curV() === 0){
    const ex = o.second ? LC692.ex2 : LC692.ex1, ans = o.second ? LC692.ans2 : LC692.ans1;
    const counts = {};
    ex.forEach(w => { counts[w] = (counts[w] || 0) + 1; });
    const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || (a < b ? -1 : 1));
    sh.push(fitT(4.9, 1.00, {zh:'輸入：' + ex.join(', '), en:'input: ' + ex.join(', ')},
                 {c:COL.tealL, fs:.28}));
    keys.forEach((w, i) => {
      const y = 1.60 + i * .62, inAns = ans.indexOf(w) >= 0;
      sh.push(box(2.30, y, 2.00, .52, inAns ? 'ok' : 'idle', w, .28));
      sh.push(fitT(4.55, y + .26, {zh:counts[w] + ' 次', en:counts[w] + 'x'},
                   {anchor:'start', c:COL.purpleL, fs:.28}));
      if (inAns) sh.push(fitT(5.60, y + .26, {zh:'第 ' + (ans.indexOf(w) + 1) + ' 名', en:'#' + (ans.indexOf(w) + 1)},
                              {anchor:'start', c:COL.orange, fs:.28}));
    });
  } else {
    const st = LC394.steps[o.step];
    sh.push(fitT(4.9, 1.00, 's = "' + LC394.s + '"', {c:COL.tealL, fs:.32}));
    let x = 3.10;
    LC394.s.split('').forEach((ch, i) => {
      sh.push(box(x, 1.45, .56, .56, i === o.step ? 'hot' : (i < o.step ? 'done' : 'idle'), ch, .30));
      x += .62;
    });
    sh.push(fitT(2.60, 2.60, {zh:'cur', en:'cur'}, {anchor:'end', c:COL.grey, fs:.28}));
    sh.push(box(2.75, 2.35, 3.20, .54, 'ok', st.cur === '' ? '(empty)' : st.cur, .28));
    sh.push(fitT(2.60, 3.28, {zh:'num', en:'num'}, {anchor:'end', c:COL.grey, fs:.28}));
    sh.push(box(2.75, 3.03, 1.00, .54, st.num ? 'act' : 'idle', String(st.num), .28));
    sh.push(fitT(2.60, 3.96, {zh:'stack', en:'stack'}, {anchor:'end', c:COL.grey, fs:.28}));
    let sx = 2.75;
    st.stack.forEach(fr => {
      sh.push(box(sx, 3.71, 1.80, .54, 'hot',
                  '(' + (fr[0] === '' ? '""' : fr[0]) + ', ' + fr[1] + ')', .24));
      sx += 1.92;
    });
  }
  return caps(sh, o);
}
function lcFrames(){
  const F = new Frames();
  if (curV() === 0){
    F.push({shapes:lcStage({cap:{zh:'最高票優先，平手照字母排', en:'most frequent first, ties broken alphabetically'}}),
      panels:[{lbl:'answer', chips:LC692.ans1.map(w => chip(w, 'ok'))}], line:1,
      msg:{zh:'"i" 和 "love" 都出現兩次。答案把 "i" 排前面，不是因為它先出現，而是因為字母序比較小。兩條規則塞在同一個 key 裡：次數遞減、字串遞增。',
           en:'"i" and "love" both appear twice. The answer puts "i" first not because it was seen first but because it sorts first. Both rules live in one key: falling on the count, rising on the word.'}, view:VIEW});
    F.push({shapes:lcStage({second:true, cap:{zh:'四個名額，四種不同的次數', en:'four places, four different counts'}}),
      panels:[{lbl:'answer', chips:LC692.ans2.map(w => chip(w, 'ok'))}], line:1,
      msg:{zh:'這個例子裡 the 4 次、is 3 次、sunny 2 次、day 1 次，四個次數都不一樣，所以排序完全由次數決定，平手規則沒出場——真正考平手的是上一個例子。這題的全部難度還是那一句平手規則——而 BPE 的合併迴圈需要的正是同一句話：挑最常出現的那一對，平手時照一條寫下來的規則挑。',
           en:'Here the counts are 4, 3, 2 and 1 - all different - so the order comes entirely from the counts and the tie-break never fires; the tie is in the previous example. The entire difficulty of this problem is still that one tie-break clause - and it is the clause a BPE trainer needs: pick the most frequent pair, and when they tie, follow a rule you wrote down.'}, view:VIEW});
    F.push({shapes:lcStage({cap:{zh:'size-k 的 heap 要把兩個欄位同時反過來', en:'a size-k heap has to invert both fields at once'}}),
      panels:[{lbl:'heap', chips:[chip('(count, RevStr(word))', 'act')]}], line:4,
      msg:{zh:'vocabulary 很大的時候要用 size-k 的 min-heap，而它必須淘汰「最差」的候選：次數最少的，而同次數時是字母序最後的。所以次數要正著比、字串要反著比——一個普通的 tuple 做不到，要嘛包一層類別，要嘛把次數取負。',
           en:'With a large vocabulary you want a size-k min-heap, and it has to evict the worst candidate: the least frequent and, among equals, the alphabetically last. So the count compares forwards while the string compares backwards - a plain tuple cannot do that, you need a wrapper class or a negated count.'}, view:VIEW});
  } else {
    LC394.steps.forEach((st, i) => {
      const ch = st.ch;
      const msg = ch === '[' ? {zh:'碰到 [：把目前的 cur 和等著它的次數一起推進 stack，然後清空重新開始。',
                                en:'On [: push the text built so far together with the count waiting for it, then start a fresh buffer.'}
        : ch === ']' ? {zh:'碰到 ]：把 stack 頂端彈出來，用它的次數把剛才累積的那段重複，再接回前面那段。cur 現在是 "' + st.cur + '"。',
                        en:'On ]: pop the top of the stack, repeat the piece just built that many times, and append it to what came before. cur is now "' + st.cur + '".'}
        : /\d/.test(ch) ? {zh:'數字：num = num * 10 + ' + ch + '。這一行就是為什麼 12[a] 會重複 12 次而不是 2 次。',
                           en:'A digit: num = num * 10 + ' + ch + '. That line is why 12[a] repeats twelve times and not two.'}
        : {zh:'一般字元直接接到 cur 後面。', en:'An ordinary character just goes onto the end of cur.'};
      F.push({shapes:lcStage({step:i, cap:{zh:'一個 stack，每個 [ 推兩樣東西進去',
                                           en:'one stack, two things pushed at every opening bracket'}}),
        panels:[{lbl:'cur', chips:[chip(st.cur === '' ? '(empty)' : st.cur, 'ok')]},
                {lbl:'num / depth', chips:[chip(String(st.num), 'act'), chip(String(st.stack.length), 'hot')]}],
        line:/\d/.test(ch) ? 2 : (ch === '[' ? 3 : (ch === ']' ? 4 : 5)), msg:msg, view:VIEW});
    });
    F.push({shapes:lcStage({step:LC394.steps.length - 1,
                            cap:{zh:'答案 ' + LC394.ans, en:'the answer is ' + LC394.ans}}),
      panels:[{lbl:'result', chips:[chip(LC394.ans, 'ok')]}], line:6,
      msg:{zh:'在 ] 到達之前什麼都不展開，這就是巢狀情況不用遞迴也能work的原因。這是 decoder 的形狀：一邊是一串扁平的 id，另一邊是比較長的字串，而展開完全由一張表驅動——tokenizer 的 decode 就是沒有巢狀的同一招。',
           en:'Nothing is expanded until the closing bracket arrives, which is what makes the nested case work without recursion. This is the shape of a decoder: a flat sequence on one side, a longer string on the other, and the expansion driven entirely by a table - a tokenizer decode is the same move without the nesting.'}, view:VIEW});
  }
  return F.list;
}

/* ================================================================ metadata */
const DAY_META = {
  title:{zh:'Day 51 — BPE Tokenizer 從零實作', en:'Day 51 - A BPE tokenizer from scratch'},
  sub:{zh:'模型看到的是一串整數，而決定那串整數長什麼樣的，是一份照順序重播的合併清單。',
       en:'A model sees a list of integers, and what decides that list is an ordered list of merges, replayed.'},
  tabs:[
    {
      id:'words', label:{zh:'一個空格的價差', en:'the price of a space'},
      stage:{zh:'同樣五個字母，一個 token 還是三個', en:'the same five letters, one token or three'},
      view:VIEW,
      idea:{zh:"' tokens'（前面帶一個空格）是一個 token，id " + WORDS[0].ids[0] + "；拿掉空格變成 " + WORDS[1].ids.length + " 個。對 tokenizer 來說開頭的空格不是空白，而是詞的一部分，因為詞在真實文字裡就是長這樣出現的——所以訓練時「空格＋tokens」才會常到足以被合併成一個符號。哪些詞有「無空格版本」完全由語料決定（這裡 'the' 剛好也有，因為它常出現在句首），不是規則。這也是為什麼同一句話在不同模型會花掉不同數量的 token，以及為什麼在 prompt 裡多打或少打一個空格，真的會換掉送進模型的那串整數。",
            en:"' tokens' with a leading space is a single token, id " + WORDS[0].ids[0] + "; without the space the same five letters cost " + WORDS[1].ids.length + ". A leading space is not whitespace to a tokenizer - it is part of the word, because that is how the word appears in running text, which is why 'space followed by tokens' was frequent enough to merge into one symbol. Which words also get a no-space version is decided by the corpus, not by a rule ('the' happens to have one here because it starts sentences). That is why the same sentence costs a different number of tokens in every model, and why adding or dropping one space in a prompt genuinely changes the integers the model receives."},
      legend:[['#3fe0dd', {zh:'一個 token', en:'a single token'}],
              ['#ff5c5c', {zh:'被拆成好幾個', en:'split into several'}]],
      get code(){ return CODE_WORDS; },
      build:wordsFrames
    },
    {
      id:'sent', label:{zh:'一句話的帳單', en:'what a sentence costs'},
      stage:{zh:'10 個詞、45 個位元組、12 個 token', en:'10 words, 45 bytes, 12 tokens'},
      view:VIEW,
      idea:{zh:'同一句話有四個長度：' + SENT.words + ' 個詞、' + SENT.chars + ' 個字元、' + SENT.bytes + ' 個位元組、' + SENT.ids.length + ' 個 token，而沒有任何一個能從另一個算出來。計費與 context window 用的是最後那一個，而唯一能算出它的東西就是 tokenizer 本身。下面那排灰色數字才是模型真正收到的輸入。注意 "call" 被切成 " ca" 和 "ll"——這不是 bug，只是這份語料沒有讓 " call" 常到被合併，所以模型永遠不會看到 call 這個詞，它看到的是兩個碎片，而這兩個碎片在別的語境裡代表別的東西。',
            en:'One sentence has four lengths: ' + SENT.words + ' words, ' + SENT.chars + ' characters, ' + SENT.bytes + ' bytes and ' + SENT.ids.length + ' tokens, and none of them can be computed from another. Billing and the context window use the last one, and the only thing that can compute it is the tokenizer. The grey row underneath is the actual input to the model. Note "call" coming apart into " ca" and "ll" - not a bug, just a corpus in which " call" was never frequent enough to merge, so the model never sees the word call, it sees two fragments that mean other things in other contexts.'},
      legend:[['#3fe0dd', {zh:'一個 token', en:'one token'}],
              ['#ff9736', {zh:'被切開的 "call"', en:'the split word "call"'}],
              ['#8fa3ac', {zh:'模型收到的整數', en:'the integers the model receives'}]],
      get code(){ return CODE_SENT; },
      build:sentFrames
    },
    {
      id:'train', label:{zh:'訓練', en:'training'},
      stage:{zh:'數、合併、重複', en:'count, merge, repeat'},
      view:VIEW,
      idea:{zh:'訓練只有三行：數每一對相鄰符號、把最常出現的那一對合併成一個新符號、重複。前八輪是 ' + TRACE.map(t => '"' + t.piece + '"').join('、') + '。第五輪開始出現「合併疊在合併上」（"' + TRACE[4].a + '" 本身是第一輪的產物），第八輪更是兩個合併符號再合併，token 就是這樣越長越長的。它在 ' + TIES.steps + ' 次合併後停下來，因為已經沒有任何一對出現超過一次——合併只出現一次的東西會讓 vocabulary 變大而文字一點都沒變短。最重要的一點：合併是**照順序**存下來的，那個順序就是 tokenizer，編碼不過是把它重播一次。',
            en:'Training is three lines: count every adjacent pair, merge the most frequent one into a new symbol, repeat. The first eight rounds are ' + TRACE.map(t => '"' + t.piece + '"').join(', ') + '. From round five merges start stacking on merges ("' + TRACE[4].a + '" is itself the output of round one) and round eight merges two merged symbols together - that is how tokens get long. It stops after ' + TIES.steps + ' merges because no pair occurs twice any more; merging something that appears once grows the vocabulary and shortens nothing. The part that matters most: the merges are kept IN ORDER, that order is the tokenizer, and encoding is nothing but replaying it.'},
      legend:[['#ff9736', {zh:'這一輪合併的兩個', en:'the pair being merged'}],
              ['#3fe0dd', {zh:'新產生的 token', en:'the new token'}],
              ['#9d6bff', {zh:'在語料裡出現幾次', en:'occurrences in the corpus'}]],
      get code(){ return CODE_TRAIN; },
      build:trainFrames
    },
    {
      id:'ties', label:{zh:'平手', en:'ties'},
      stage:{zh:'最高票通常不只一個', en:'the most frequent pair is usually not unique'},
      view:VIEW,
      idea:{zh:'這份語料跑了 ' + TIES.steps + ' 個合併步驟，其中 ' + TIES.ties + ' 步（' + Math.round(100 * TIES.ties / TIES.steps) + '%）在最高票上是平手的。第一次就發生在第 ' + TIES.step + ' 步：' + TIES.examples.map(e => '"' + e + '"').join(' 和 ') + ' 都出現 ' + TIES.count + ' 次。挑哪一個都說得過去，但挑完之後那一對附近的計數會變，兩個 tokenizer 從這裡開始分家——換別的平手規則重跑是 141–143 次合併、500–504 個 token，壓縮只差幾個 token，而對「什麼是一個 token」給出不同的答案。所以平手規則不是實作細節而是規格的一部分：這份程式取 id 最小的那一對，寫在 max 的 key 裡；沒有這條規則，vocabulary 會跟著 dict 的插入順序跑。挑最高票又要有明確平手規則，就是 LeetCode 692。',
            en:'This corpus takes ' + TIES.steps + ' merge steps and ' + TIES.ties + ' of them (' + Math.round(100 * TIES.ties / TIES.steps) + '%) had more than one pair tied for most frequent. The first is step ' + TIES.step + ': ' + TIES.examples.map(e => '"' + e + '"').join(' and ') + ' both occur ' + TIES.count + ' times. Either choice is defensible, but afterwards the counts around that pair change and the two tokenizers diverge - other tie-breaks give 141 to 143 merges and 500 to 504 tokens, so they compress to within a few tokens of each other and give different answers to what a token is. So the tie-break is not an implementation detail but part of the specification: this file takes the lowest pair of ids, written into the key of max, and without it the vocabulary follows dictionary insertion order. Picking the most frequent thing with a written-down tie-break is LeetCode 692.'},
      legend:[['#3fe0dd', {zh:'合併步數', en:'merge steps'}],
              ['#ff5c5c', {zh:'其中有平手的', en:'steps with a tie'}],
              ['#ff9736', {zh:'第一次平手的兩個候選', en:'the first tied pair'}]],
      get code(){ return CODE_TIES; },
      build:tiesFrames
    },
    {
      id:'enc', label:{zh:'編碼', en:'encoding'},
      stage:{zh:'照 rank 重播，不是最長匹配', en:'replay by rank, not longest-match'},
      view:VIEW,
      idea:{zh:"編碼一個詞的時候，每一步都看「現在還在的相鄰組合裡，哪一個的 rank 最小」——也就是最早被學到的那一個——然後套用它。' counted' 從 8 個位元組開始，依序套用第 " + WALK.steps.slice(1).map(s => s.rank).join('、第 ') + " 號合併，最後剩下 " + WALK.steps[WALK.steps.length - 1].pieces.length + " 個 token。挑的是 rank 不是長度，這很重要：同一組 token 字串如果改用「最長匹配」去切，會得到不同的答案。所以一個 tokenizer 要發佈的是那份有順序的合併清單，而不是只有 token 的字串表——拿到字串表卻用別的規則去切，就是一個長得很像但其實不同的 tokenizer。",
            en:"Encoding a word asks the same question at every step: of the adjacent pairs still present, which has the smallest rank - the one learned earliest - and applies it. ' counted' starts as 8 bytes and applies merges #" + WALK.steps.slice(1).map(s => s.rank).join(', #') + ", ending at " + WALK.steps[WALK.steps.length - 1].pieces.length + " tokens. It picks by rank, not by length, and that matters: a longest-match tokenizer over the same set of token strings produces different output. So what ships with a model is the ordered merge list, not just the table of token strings - take the table and apply a different rule and you have a tokenizer that looks the same and is not."},
      legend:[['#3fe0dd', {zh:'目前這一步的結果', en:'the current state'}],
              ['#ff9736', {zh:'套用的合併編號', en:'the merge being applied'}]],
      get code(){ return CODE_ENC; },
      build:encFrames
    },
    {
      id:'utf8', label:{zh:'解碼與 UTF-8', en:'decoding and UTF-8'},
      stage:{zh:'先接起來，再解一次', en:'join the bytes, decode once'},
      view:VIEW,
      idea:{zh:'「' + UTF8.text + '」是 ' + UTF8.bytes + ' 個位元組、' + UTF8.n + ' 個 token，其中 ' + UTF8.bad + ' 個單獨拿出來不是合法的 UTF-8——三個位元組的字被切在 token 邊界上是常態，不是錯誤。錯在怎麼接回來：把所有位元組接起來再 decode 一次，拿回「' + UTF8.joined + '」；一個一個 decode 再接起來，拿到「' + UTF8.perToken + '」。後者不會丟例外，因為 errors="replace" 把每個壞掉的位元組換成替換字元了。逐 token 串流輸出最常撞到這個，而正確的修法是把還沒構成完整字元的位元組留在手上，不是用 replace 蓋過去。一個 tokenizer 最該測的性質就是 round trip：encode、decode、比對位元組。',
            en:'"' + UTF8.text + '" is ' + UTF8.bytes + ' bytes and ' + UTF8.n + ' tokens, of which ' + UTF8.bad + ' are not valid UTF-8 on their own - a three-byte character being split across a token boundary is normal, not a bug. The bug is in how you put it back: join all the bytes and decode once and you get "' + UTF8.joined + '"; decode each token and then join and you get "' + UTF8.perToken + '". The second does not raise, because errors="replace" turned every broken byte into a replacement character. Streaming output token by token is where this shows up, and the fix is to hold back bytes that do not yet form a complete character rather than replacing them. The property most worth testing is the round trip: encode, decode, compare bytes.'},
      legend:[['#3fe0dd', {zh:'單獨就是合法 UTF-8', en:'valid UTF-8 on its own'}],
              ['#ff5c5c', {zh:'只是半個字的位元組', en:'part of a character'}]],
      get code(){ return CODE_UTF8; },
      build:utf8Frames
    },
    {
      id:'lc', label:{zh:'LeetCode 692 / 394', en:'LeetCode 692 / 394'},
      stage:{zh:'平手規則，與展開', en:'the tie-break, and the expansion'},
      view:VIEW,
      variants:[{zh:'692 最高票', en:'692 top k frequent'}, {zh:'394 展開字串', en:'394 decode string'}],
      idea:{zh:'692 要回傳出現次數最多的 k 個詞，最多的排前面，平手時照字母序。那句平手規則就是這題的全部難度，也正是 BPE 合併迴圈需要的同一句話。一個 key 同時裝下兩條規則：次數遞減、字串遞增；而 size-k 的 min-heap 版本必須把兩個欄位同時反過來（淘汰的是「次數最少、同次數時字母序最後」的那一個），所以普通的 tuple 不夠用。394 要把 3[a2[c]] 展開成 accaccacc：一個 stack，每遇到 [ 就把「目前累積的字串」和「等著它的次數」一起推進去，遇到 ] 才彈出來展開。兩個細節：次數可能是多位數（num = num * 10 + d），以及 ] 之後的字元會繼續接在同一個 buffer 上。它就是 decoder 的形狀——一串扁平的 id 進去，一段比較長的字串出來，展開完全由一張表驅動，而 tokenizer 的 decode 就是沒有巢狀的同一招。',
            en:'692 asks for the k most frequent words, most frequent first, ties broken alphabetically. That tie-break clause is the entire difficulty of the problem and it is the same clause a BPE trainer needs. One key carries both rules - falling on the count, rising on the word - and the size-k min-heap version has to invert both fields at once, because it evicts the least frequent and, among equals, the alphabetically last, which a plain tuple cannot express. 394 expands 3[a2[c]] into accaccacc with one stack: at every opening bracket push the text built so far together with the count waiting for it, and expand only when the closing bracket arrives. Two details: counts can be multi-digit (num = num * 10 + d), and characters after a bracket closes keep accumulating onto the same buffer. It is the shape of a decoder - a flat sequence of ids in, a longer string out, the expansion driven entirely by a table - and a tokenizer decode is the same move without the nesting.'},
      legend:[['#3fe0dd', {zh:'答案裡的項目 / cur', en:'in the answer / cur'}],
              ['#ff9736', {zh:'目前這一步 / stack', en:'current step / stack'}],
              ['#9d6bff', {zh:'次數', en:'the count'}]],
      get code(){ return curV() === 0 ? CODE_692 : CODE_394; },
      build:lcFrames
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
