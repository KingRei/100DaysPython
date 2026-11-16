// DAY: 57
// TITLE_ZH: RadixAttention：讓開頭相同的請求共用同一份 KV cache
// TITLE_EN: RadixAttention: sharing one KV cache between requests that start the same way
// SUB_ZH: 在真的伺服器上，多數請求的開頭是一樣的——同一段 system prompt、同幾個範例、同一段對話的前幾輪。把快取放進一棵 radix tree（Day 24），查最長前綴、用 LRU（Day 30）淘汰但只淘汰葉子，這個工作量就不用做第二次。
// SUB_EN: On a real server most requests begin identically - the same system prompt, the same examples, the earlier turns of the same conversation. Put the cached tokens in a radix tree, look up the longest prefix already there, and evict with LRU but only from the leaves, and that work never has to happen twice.
// FOLDER: day%2057%20-%20radix%20attention
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
const WORK = {"reqs": 80, "convs": 20, "sys": 200, "shared": 400, "total": 47466, "reused": 40146, "hit": 84.6, "new": 7320, "slots": 7320, "plainSlots": 47466, "nodes": 83, "trie": 7216, "avgLen": 593.0};
const GROUPS = [{"name": "first turn", "n": 20, "len": 440.0, "match": 380.0, "share": 86.0}, {"name": "later turns", "n": 60, "len": 644.0, "match": 542.0, "share": 84.0}];
const SWEEP = [{"cap": 200, "frac": 3.0, "hit": 0.0, "evicted": 47466}, {"cap": 400, "frac": 5.0, "hit": 0.0, "evicted": 47466}, {"cap": 600, "frac": 8.0, "hit": 81.0, "evicted": 8499}, {"cap": 800, "frac": 11.0, "hit": 84.7, "evicted": 6482}, {"cap": 1200, "frac": 16.0, "hit": 84.7, "evicted": 6168}, {"cap": 2000, "frac": 27.0, "hit": 84.7, "evicted": 5264}, {"cap": 4000, "frac": 55.0, "hit": 84.7, "evicted": 3297}, {"cap": 7320, "frac": 100.0, "hit": 84.6, "evicted": 0}];
const TRAP = {"a": "abc", "b": "abd", "ta": [5], "tb": [4, 7], "textPrefix": 2, "tokPrefix": 0};
const EVICT = {"victim": 124, "children": 1, "orphaned": 379, "hit": 75.6};


/* ================================================================ tab 1: the waste */
const CODE_WORK = [
  '# a chat server\'s day, in token ids',
  'request = SYSTEM + EXAMPLES + conversation_so_far + new_user_turn',
  '',
  '#   SYSTEM             the same 200 tokens for everybody',
  '#   EXAMPLES           the same 200 tokens for everybody',
  '#   conversation_so_far  the same tokens as this user\'s previous turn',
  '#   new_user_turn      the only part that has never been seen'
];
function workStage(o){
  const sh = [], X0 = .80, W0 = 8.2;
  const parts = [{n:{zh:'system prompt', en:'system prompt'}, w:200, s:'hot'},
                 {n:{zh:'範例', en:'examples'}, w:200, s:'hot'},
                 {n:{zh:'這段對話前幾輪', en:'this conversation so far'}, w:204, s:'act'},
                 {n:{zh:'新的使用者輸入', en:'the new user turn'}, w:40, s:'ok'}];
  const tot = parts.reduce((a, p) => a + p.w, 0);
  let x = X0;
  parts.forEach((pt, i) => {
    if (i > o.upto) return;
    const w = W0 * pt.w / tot;
    sh.push(S.r(x, 3.40, w - .06, .76, pt.s, null));
    sh.push(fitT(x + w / 2, 3.08, pt.n, {c:COL.grey, fs:.24}));
    sh.push(fitT(x + w / 2, 4.32, pt.w + (i === 2 ? '+' : ''), {c:COL.pale, fs:.26}));
    x += w;
  });
  if (o.upto >= 3){
    sh.push(fitT(4.9, 2.40, {zh:'只有最右邊那一塊是新的', en:'only the rightmost piece is new'},
                 {c:COL.tealL, fs:.30}));
  }
  side(sh, .80, 1.70, o.notes || [], .46);
  return caps(sh, o);
}
function workFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:workStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:3,
    cap:{zh:'一個聊天伺服器的一天', en:'a chat server\'s day'},
    panels:[{lbl:'requests', chips:[chip(String(WORK.reqs), 'ok')]},
            {lbl:'shared head', chips:[chip(WORK.shared + ' tokens', 'hot')]}],
    msg:{zh:'這份模擬工作量是 ' + WORK.convs + ' 段對話、每段 4 輪，共 ' + WORK.reqs + ' 個請求。每個請求開頭都是同一段 ' + WORK.sys + ' 個 token 的 system prompt 加同樣兩個範例——' + WORK.shared + ' 個 token，一個字都沒變。',
         en:'The workload is ' + WORK.convs + ' conversations of four turns, ' + WORK.reqs + ' requests in all. Every one of them starts with the same ' + WORK.sys + '-token system prompt and the same two examples: ' + WORK.shared + ' tokens, identical every time.'}});
  p({upto:3, line:6,
    cap:{zh:'而第二輪開始，連對話本身也是重複的', en:'and from the second turn on, the conversation repeats too'},
    panels:[{lbl:'avg request', chips:[chip(WORK.avgLen + ' tokens', 'dim')]},
            {lbl:'new per request', chips:[chip(Math.round(WORK.new / WORK.reqs) + ' tokens', 'ok')]}],
    msg:{zh:'一個請求平均 ' + WORK.avgLen + ' 個 token，但其中真正沒被算過的平均只有 ' + Math.round(WORK.new / WORK.reqs) + ' 個。第二輪要把第一輪的問題和回答整個再送一次、第三輪送前兩輪——對話越長，重複的比例越高。',
         en:'An average request is ' + WORK.avgLen + ' tokens, of which only about ' + Math.round(WORK.new / WORK.reqs) + ' have never been computed before. The second turn resends the whole first exchange, the third resends two - the longer the conversation, the larger the repeated share.'}});
  p({upto:3, line:0,
    cap:{zh:'整份工作量加起來', en:'over the whole workload'},
    panels:[{lbl:'prompt tokens', chips:[chip(WORK.total.toLocaleString(), 'bad')]},
            {lbl:'actually new', chips:[chip(WORK.new.toLocaleString(), 'ok')]},
            {lbl:'reused', chips:[chip(WORK.hit.toFixed(1) + '%', 'hot')]}],
    msg:{zh:'全部 ' + WORK.total.toLocaleString() + ' 個 prompt token 裡，只有 ' + WORK.new.toLocaleString() + ' 個是新的——**' + WORK.hit.toFixed(1) + '% 是在重算已經算過的東西**。而 prefill 正是算力受限的那一半（昨天量過），所以「不用做」在這裡是最大的一筆節省。模型一個字都沒改，改的只是不要重算。',
         en:'Across all ' + WORK.total.toLocaleString() + ' prompt tokens only ' + WORK.new.toLocaleString() + ' are new: ' + WORK.hit.toFixed(1) + '% of the work is recomputing something already computed. Prefill is the compute-bound half (measured yesterday), so not doing it is the largest saving on the table. Nothing about the model changes - only that the work is not repeated.'}});
  return F.list;
}

/* ================================================================ tab 2: the tree */
const CODE_TREE = [
  'def match(self, tokens):          # longest cached prefix',
  '    node, i = self.root, 0',
  '    while i < len(tokens):',
  '        child = node.children.get(tokens[i])',
  '        if child is None: break',
  '        j = common_prefix_len(child.key, tokens[i:])',
  '        if j < len(child.key):    # diverges inside an edge',
  '            return i + j, child, j',
  '        i += len(child.key); node = child',
  '    return i, node, len(node.key)'
];
function treeStage(o){
  const sh = [];
  sh.push(box(.65, 3.45, 2.20, .60, 'hot', {zh:'system prompt 200', en:'system prompt 200'}, .26));
  sh.push(box(3.05, 3.45, 1.90, .60, 'hot', {zh:'範例 200', en:'examples 200'}, .26));
  sh.push(S.e(2.90, 3.75, 3.00, 3.75, {s:'ghost', arrow:true, w:.05}));
  [[4.60, {zh:'對話 1', en:'conversation 1'}], [3.45, {zh:'對話 2', en:'conversation 2'}],
   [2.30, {zh:'對話 20', en:'conversation 20'}]].forEach((c, i) => {
    if (i > o.upto) return;
    sh.push(box(5.55, c[0], 1.85, .52, 'act', c[1], .26));
    sh.push(S.e(5.00, 3.75, 5.50, c[0] + .26, {s:'ghost', arrow:true, w:.04}));
    [0, 1].forEach(t => {
      sh.push(box(7.65, c[0] - t * .58, 1.55, .48,
                  (o.upto === i && t === 1) ? 'ok' : 'idle',
                  {zh:'第 ' + (t + 1) + ' 輪', en:'turn ' + (t + 1)}, .24));
    });
  });
  side(sh, .65, 1.75, o.notes || [], .46);
  return caps(sh, o);
}
function treeFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:treeStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'一棵樹，每條邊上放一串 token', en:'one tree, with a run of tokens on each edge'},
    panels:[{lbl:'nodes', chips:[chip(WORK.nodes.toLocaleString(), 'ok')]},
            {lbl:'slots', chips:[chip(WORK.slots.toLocaleString(), 'ok')]}],
    msg:{zh:'第一個請求進來時整段變成一條邊。之後每個請求從 root 往下走，只要 token 還對得上就繼續——走到的長度就是「不用重算的前綴」。這棵樹總共 ' + WORK.nodes.toLocaleString() + ' 個節點、' + WORK.slots.toLocaleString() + ' 個 token slot。',
         en:'The first request arrives and becomes one long edge. Every request after it walks down from the root while the tokens still agree, and how far it gets is the prefix that does not need recomputing. The whole tree is ' + WORK.nodes.toLocaleString() + ' nodes holding ' + WORK.slots.toLocaleString() + ' token slots.'}});
  p({upto:1, line:6,
    cap:{zh:'走到一半分岔時，把那條邊切成兩段', en:'when a request diverges halfway, the edge is split in two'},
    panels:[{lbl:'split', chips:[chip('one edge -> two', 'hot')]}],
    msg:{zh:'第二段對話跟第一段共用 system prompt 和範例，但使用者說的話不一樣。於是那條長邊在分岔處被切開：上半段變成共用的父節點，兩邊各自掛一段尾巴。這就是 radix tree 跟一般 trie 的差別——**邊上放的是一整串 token，不是一個**。',
         en:'The second conversation shares the system prompt and the examples but not the user\'s words, so the long edge gets split at the point where they part: the head becomes a shared parent and each side keeps its own tail. That is what makes it a radix tree rather than a plain trie - an edge holds a run of tokens, not one.'}});
  p({upto:2, line:0,
    cap:{zh:'而那個差別不是裝飾', en:'and that difference is not cosmetic'},
    panels:[{lbl:'radix nodes', chips:[chip(WORK.nodes.toLocaleString(), 'ok')]},
            {lbl:'trie nodes', chips:[chip(WORK.trie.toLocaleString(), 'bad'),
                                      chip(Math.round(WORK.trie / WORK.nodes) + 'x', 'bad')]}],
    msg:{zh:'同樣這些序列，一個 token 一個節點的 trie 要 ' + WORK.trie.toLocaleString() + ' 個節點，radix tree 只要 ' + WORK.nodes.toLocaleString() + ' 個——差 ' + Math.round(WORK.trie / WORK.nodes) + ' 倍。至於記憶體：不共用的話要 ' + WORK.plainSlots.toLocaleString() + ' 個 slot，共用之後 ' + WORK.slots.toLocaleString() + '。',
         en:'Over the same sequences a one-token-per-node trie needs ' + WORK.trie.toLocaleString() + ' nodes against the radix tree\'s ' + WORK.nodes.toLocaleString() + ' - ' + Math.round(WORK.trie / WORK.nodes) + ' times more. And in memory: ' + WORK.plainSlots.toLocaleString() + ' slots without sharing, ' + WORK.slots.toLocaleString() + ' with it.'}});
  p({upto:2, line:0,
    cap:{zh:'誰分享到多少', en:'who gets how much'},
    panels:GROUPS.map(g => ({lbl:g.name, chips:[chip(g.match + ' / ' + g.len, 'ok'),
                                                chip(g.share + '%', 'hot')]})),
    msg:{zh:'一段對話的第一輪只分享得到 system prompt 和範例（平均 ' + GROUPS[0].match + ' / ' + GROUPS[0].len + ' ＝ ' + GROUPS[0].share + '%）；第二輪之後連自己前面講過的也在樹上（' + GROUPS[1].share + '%）。同一個結構同時處理這兩種情況，而且它完全不知道什麼是 system prompt、什麼是對話——它只知道前綴。',
         en:'The first turn of a conversation can only share the system prompt and the examples (' + GROUPS[0].match + ' of ' + GROUPS[0].len + ' tokens, ' + GROUPS[0].share + '%); from the second turn on, everything the conversation has already said is in the tree too (' + GROUPS[1].share + '%). One structure covers both, and it knows nothing about system prompts or conversations - only about prefixes.'}});
  return F.list;
}

/* ================================================================ tab 3: eviction */
const CODE_EVICT = [
  'def _evict_until_fits(self):',
  '    while self.size > self.capacity:',
  '        leaves = [n for n in tree if not n.children and n.refs == 0]',
  '        victim = min(leaves, key=lambda n: n.last_used)   # LRU, Day 30',
  '        del victim.parent.children[victim.key[0]]',
  '        self.size -= len(victim.key)',
  '',
  '# leaves only: an internal node is somebody else\'s prefix',
  '# refs > 0: a request is still decoding on this node'
];
function sweepStage(o){
  const sh = [], X0 = 1.55, Y0 = 1.70, WD = 6.9, HT = 2.60;
  sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  const xs = SWEEP.map((_, i) => X0 + WD * i / (SWEEP.length - 1));
  for (let i = 1; i < SWEEP.length; i++){
    if (i > o.upto) break;
    sh.push(S.e(xs[i - 1], Y0 + HT * SWEEP[i - 1].hit / 100,
                xs[i], Y0 + HT * SWEEP[i].hit / 100, {s:'ok', arrow:false, w:.06}));
  }
  SWEEP.forEach((r, i) => {
    if (i > o.upto) return;
    sh.push(S.c(xs[i], Y0 + HT * r.hit / 100, .11, r.hit < 50 ? 'bad' : 'ok', null));
    sh.push(fitT(xs[i], Y0 - .28, r.cap.toLocaleString(), {c:COL.grey, fs:.22}));
  });
  [[0, '0'], [50, '50%'], [100, '100%']].forEach(v => {
    sh.push(fitT(X0 - .18, Y0 + HT * v[0] / 100, v[1], {anchor:'end', c:COL.grey, fs:.24}));
  });
  sh.push(fitT(X0 + WD / 2, Y0 - .66, {zh:'池子裡有幾個 token slot', en:'token slots in the pool'},
               {c:COL.grey, fs:.26}));
  sh.push(fitT(X0 + .10, Y0 + HT + .30, {zh:'prompt token 有多少比例來自快取',
                                         en:'share of prompt tokens served from the cache'},
               {anchor:'start', c:COL.grey, fs:.25}));
  return caps(sh, o);
}
function evictFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:sweepStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:SWEEP.length - 1, line:3,
    cap:{zh:'池子是有限的，所以要淘汰', en:'the pool is finite, so something has to go'},
    panels:[{lbl:'full tree', chips:[chip(WORK.slots.toLocaleString() + ' slots', 'dim')]},
            {lbl:'hit rate', chips:[chip(WORK.hit.toFixed(1) + '%', 'ok')]}],
    msg:{zh:'整棵樹要 ' + WORK.slots.toLocaleString() + ' 個 slot。實際的伺服器沒有那麼多，所以要淘汰——用的就是 Day 30 的 LRU，只是多一條規則。',
         en:'The whole tree wants ' + WORK.slots.toLocaleString() + ' slots, and a real server does not have them, so entries have to be evicted. The policy is Day 30\'s LRU with one extra rule.'}});
  p({upto:SWEEP.length - 1, line:3,
    cap:{zh:'第一件事：小很多的池子一樣好用', en:'first: a much smaller pool works just as well'},
    panels:[{lbl:'pool ' + SWEEP[3].cap, chips:[chip(SWEEP[3].hit.toFixed(1) + '%', 'ok'),
                                                chip(SWEEP[3].frac + '% of the tree', 'dim')]},
            {lbl:'pool ' + SWEEP[SWEEP.length - 1].cap,
             chips:[chip(SWEEP[SWEEP.length - 1].hit.toFixed(1) + '%', 'ok')]}],
    msg:{zh:'只給 ' + SWEEP[3].cap.toLocaleString() + ' 個 slot（整棵樹的 ' + SWEEP[3].frac + '%）時命中率是 ' + SWEEP[3].hit.toFixed(1) + '%，給全部也只是 ' + SWEEP[SWEEP.length - 1].hit.toFixed(1) + '%。原因是**賺到命中的那一段是共用前綴，而只淘汰葉子的 LRU 永遠不會動到它**——它是內部節點，所有人都從它身上「經過」。',
         en:'With only ' + SWEEP[3].cap.toLocaleString() + ' slots - ' + SWEEP[3].frac + '% of the tree - the hit rate is ' + SWEEP[3].hit.toFixed(1) + '%, against ' + SWEEP[SWEEP.length - 1].hit.toFixed(1) + '% with all of it. The reason is that the part earning the hits is the shared prefix, and leaf-only LRU never touches it: it is an internal node that everybody passes through.'}});
  p({upto:SWEEP.length - 1, line:2,
    cap:{zh:'第二件事：再小下去不是退化，是崩掉', en:'second: below that it does not degrade, it collapses'},
    panels:[{lbl:'pool ' + SWEEP[1].cap, chips:[chip(SWEEP[1].hit.toFixed(1) + '%', 'bad')]},
            {lbl:'pool ' + SWEEP[2].cap, chips:[chip(SWEEP[2].hit.toFixed(1) + '%', 'ok')]}],
    msg:{zh:'' + SWEEP[1].cap + ' 個 slot 時命中率是 ' + SWEEP[1].hit.toFixed(1) + '%，' + SWEEP[2].cap + ' 個時是 ' + SWEEP[2].hit.toFixed(1) + '%。池子至少要裝得下「共用前綴（這裡 ' + WORK.shared + ' 個）＋ 一個請求活得下來的量」；低於那條線，每一次插入都把上一次剛寫進去的東西擠掉，等於沒有快取。',
         en:'At ' + SWEEP[1].cap + ' slots the hit rate is ' + SWEEP[1].hit.toFixed(1) + '% and at ' + SWEEP[2].cap + ' it is ' + SWEEP[2].hit.toFixed(1) + '%. The pool has to hold the shared prefix (' + WORK.shared + ' tokens here) plus enough of a request to survive insertion; under that line every insert evicts what the previous one just wrote, and the cache does nothing at all.'}});
  p({upto:SWEEP.length - 1, line:7,
    cap:{zh:'為什麼只能淘汰葉子', en:'why only leaves may be evicted'},
    panels:[{lbl:'victim', chips:[chip(EVICT.victim + ' tokens', 'bad')]},
            {lbl:'orphaned with it', chips:[chip(String(EVICT.orphaned), 'bad')]}],
    msg:{zh:'如果照一般 LRU 直接挑「最久沒用到」的節點，挑中的往往就是共用前綴——因為大家都是從它**經過**而不是停在它上面，它的時間戳最舊。實測挑中的那個節點自己只有 ' + EVICT.victim + ' 個 token，但底下掛著 ' + EVICT.children + ' 個子節點，一起消失的是 ' + EVICT.orphaned + ' 個 token。所以規則是「只淘汰葉子」，而且正在 decode 的請求所在的節點要用 refcount 釘住。',
         en:'Plain LRU picks the node with the oldest timestamp, and that is usually the shared prefix - because everyone matches *through* it rather than ending on it, nothing refreshes its timestamp. Measured here, the node it would pick holds ' + EVICT.victim + ' tokens itself but has ' + EVICT.children + ' children below it, and ' + EVICT.orphaned + ' tokens disappear with it. Hence the rule: leaves only, and nodes a running request is sitting on are pinned by a reference count.'}});
  return F.list;
}

/* ================================================================ tab 4: the trap */
const CODE_TRAP = [
  '# a toy tokeniser where "ab" and "abc" are both single tokens',
  'encode("abc") -> (5,)        # one token',
  'encode("abd") -> (4, 7)      # "ab" then "d"',
  '',
  '# the strings share two of three characters',
  '# the token sequences share nothing at all',
  '',
  '# so: match prefixes on token ids, after the tokeniser'
];
function trapStage(o){
  const sh = [], cw = 1.05;
  sh.push(fitT(1.10, 4.35, '"' + TRAP.a + '"', {anchor:'start', c:COL.pale, fs:.34}));
  TRAP.ta.forEach((t, j) => sh.push(box(2.60 + j * cw, 4.08, cw * .86, .60, 'ok', String(t), .30)));
  sh.push(fitT(1.10, 3.35, '"' + TRAP.b + '"', {anchor:'start', c:COL.pale, fs:.34}));
  TRAP.tb.forEach((t, j) => sh.push(box(2.60 + j * cw, 3.08, cw * .86, .60,
                                        o.upto >= 1 ? 'bad' : 'ok', String(t), .30)));
  if (o.upto >= 1){
    sh.push(fitT(4.9, 2.35, {zh:'共同的字元：' + TRAP.textPrefix + '　　共同的 token：' + TRAP.tokPrefix,
                             en:TRAP.textPrefix + ' characters in common, ' + TRAP.tokPrefix + ' tokens in common'},
                 {c:COL.red, fs:.30}));
  }
  side(sh, 1.10, 1.60, o.notes || [], .46);
  return caps(sh, o);
}
function trapFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:trapStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'前綴要比在 token 上，不是比在文字上', en:'prefixes match on tokens, not on text'},
    panels:[{lbl:'"' + TRAP.a + '"', chips:TRAP.ta.map(t => chip(String(t), 'ok'))}],
    msg:{zh:'用一個小 tokenizer：「ab」和「abc」都是單一個 token。「' + TRAP.a + '」被切成一個 token。',
         en:'With a tiny tokeniser where both "ab" and "abc" are single tokens, "' + TRAP.a + '" encodes to one token.'}});
  p({upto:1, line:5,
    cap:{zh:'只差最後一個字元，token 卻完全不一樣', en:'one character different, and nothing in common'},
    panels:[{lbl:'"' + TRAP.b + '"', chips:TRAP.tb.map(t => chip(String(t), 'bad'))},
            {lbl:'shared', chips:[chip(TRAP.textPrefix + ' chars', 'dim'),
                                  chip(TRAP.tokPrefix + ' tokens', 'bad')]}],
    msg:{zh:'「' + TRAP.b + '」被切成兩個 token，而且第一個就跟上面不一樣——tokenizer 跨過了那個邊界去合併。兩個字串有 ' + TRAP.textPrefix + ' 個字元相同，token 卻一個都不共用。如果快取是用「文字前綴」當 key，它會把一個根本不存在於第二個請求裡的 token 的 key/value 交出去——形狀對、數值有限、不會報錯，模型只是讀到了錯的東西。所以比對一定要在 tokenizer 之後、用 token id 做。',
         en:'"' + TRAP.b + '" encodes to two tokens, and the very first one already differs, because the tokeniser merged across that boundary. The strings share ' + TRAP.textPrefix + ' characters and the token sequences share none. A cache keyed on text would hand back keys and values for a token the second request never contained - right shapes, finite numbers, no error, and the model reading the wrong thing. Prefix matching belongs on token ids, after the tokeniser.'}});
  return F.list;
}

/* ================================================================ tab 5: PyTorch / JAX */
const CODE_TORCH = [
  '# transformers has no prefix sharing: you do it by hand, per request',
  'sys_ids = tok(SYSTEM_PROMPT, return_tensors="pt").to("cuda")',
  'base    = model(**sys_ids, use_cache=True).past_key_values   # computed once',
  '',
  'for user_turn in requests:',
  '    past = copy.deepcopy(base)          # a full COPY, per request',
  '    out  = model(input_ids=encode(user_turn), past_key_values=past,',
  '                 use_cache=True)',
  '',
  '# correct, and it costs one cache per request: nothing is shared,',
  '# only the prefill is skipped'
];
const CODE_JAX = [
  '# JAX: the cache is a padded array, so a prefix is a prefix of slots',
  'base_k, base_v, base_len = prefill(params, SYSTEM_IDS)   # jitted once',
  '',
  'def start_request(user_ids):',
  '    k = jax.lax.dynamic_update_slice(base_k, user_k, (0, base_len, 0))',
  '    ...',
  '',
  '# same story: the prefill is reused, the memory is not -',
  '# sharing one copy between live requests needs the server to own',
  '# the memory, which is the next tab'
];
function fwStage(o){
  const sh = [];
  const rows = o.which === 0
    ? [[{zh:'我們的 match()', en:'our match()'}, {zh:'沒有：自己留 base cache', en:'nothing: keep a base cache'}],
       [{zh:'共用節點', en:'a shared node'}, {zh:'deepcopy(base)：每個請求一份', en:'deepcopy(base), one per request'}],
       [{zh:'省下的 prefill', en:'prefill skipped'}, {zh:'有（這是主要好處）', en:'yes, and that is the win'}],
       [{zh:'省下的記憶體', en:'memory shared'}, {zh:'沒有', en:'no'}]]
    : [[{zh:'我們的 match()', en:'our match()'}, {zh:'prefill 一次、之後切片', en:'prefill once, then slice'}],
       [{zh:'共用節點', en:'a shared node'}, {zh:'dynamic_update_slice 疊上去', en:'dynamic_update_slice onto a copy'}],
       [{zh:'省下的 prefill', en:'prefill skipped'}, {zh:'有', en:'yes'}],
       [{zh:'省下的記憶體', en:'memory shared'}, {zh:'沒有（形狀必須固定）', en:'no: shapes must stay fixed'}]];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.05 - i * .92;
    sh.push(box(.70, y, 3.30, .62, 'idle', r[0], .26));
    sh.push(S.e(4.15, y + .31, 4.75, y + .31, {s:'ghost', arrow:true, w:.05}));
    sh.push(box(4.90, y, 4.30, .62, i === o.upto ? 'hot' : 'ok', r[1], .26));
  });
  sh.push(fitT(2.35, 5.02, {zh:'這個檔案裡的做法', en:'what this file does'}, {c:COL.grey, fs:.26}));
  sh.push(fitT(7.05, 5.02, o.which === 0 ? 'PyTorch / transformers' : 'JAX',
               {c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
function fwFrames(v){
  const F = new Frames();
  const p = (o) => F.push({shapes:fwStage(Object.assign({which:v}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  if (v === 0){
    p({upto:1, line:2,
      cap:{zh:'PyTorch：前綴共享要自己做', en:'PyTorch: prefix sharing is something you do yourself'},
      panels:[{lbl:'base cache', chips:[chip('computed once', 'ok')]},
              {lbl:'per request', chips:[chip('deepcopy', 'bad')]}],
      msg:{zh:'`transformers` 沒有跨請求的前綴共享。能做的是把 system prompt 先跑一次拿到 `past_key_values`，之後每個請求複製一份接著跑。prefill 確實省下來了，但**記憶體沒有共用**：每個在跑的請求都帶一份完整的副本。',
           en:'`transformers` has no cross-request prefix sharing. What you can do is run the system prompt once, keep its `past_key_values`, and deep-copy it for each request. The prefill really is saved - but the memory is not shared: every live request carries a full copy.'}});
    p({upto:3, line:8,
      cap:{zh:'所以它省時間不省空間', en:'so it saves time, not space'},
      panels:[{lbl:'prefill', chips:[chip('skipped', 'ok')]},
              {lbl:'memory', chips:[chip('one cache per request', 'bad')]}],
      msg:{zh:'這也是為什麼「共用」要等到有人**擁有整張卡的記憶體**才做得到：只要 cache 還是每個請求手上的一個物件，你最多只能省掉計算、省不掉空間。而今天量到的 ' + WORK.hit.toFixed(1) + '% 裡，省空間那一半（' + WORK.plainSlots.toLocaleString() + ' → ' + WORK.slots.toLocaleString() + ' slot）正是這種寫法拿不到的。',
           en:'This is also why real sharing has to wait for something that owns all of the GPU memory: as long as the cache is an object in each request\'s hands, you can skip the compute but not the storage. Of today\'s numbers, the memory half - ' + WORK.plainSlots.toLocaleString() + ' slots down to ' + WORK.slots.toLocaleString() + ' - is exactly what this style cannot get.'}});
    return F.list;
  }
  p({upto:1, line:1,
    cap:{zh:'JAX：cache 是固定形狀的陣列', en:'JAX: the cache is a fixed-shape array'},
    panels:[{lbl:'prefill', chips:[chip('jitted once', 'ok')]}],
    msg:{zh:'JAX 的 cache 是一個補滿的陣列，所以「前綴」就是「前面那幾個 slot」。把 system prompt 先 prefill 一次，之後每個請求從那個長度往後寫——同樣省掉 prefill。',
         en:'A JAX cache is a padded array, so a prefix is simply the first so many slots. Prefill the system prompt once, then have each request write from that length onwards: the prefill is saved the same way.'}});
  p({upto:3, line:7,
    cap:{zh:'但形狀固定代表還是一人一份', en:'but fixed shapes still mean one copy each'},
    panels:[{lbl:'memory', chips:[chip('one array per request', 'bad')]}],
    msg:{zh:'因為 jit 要靜態形狀，每個同時在跑的請求仍然要有自己的那塊陣列。跟 PyTorch 一樣：計算省得掉，空間省不掉。真正的共用需要一個「把整張卡的顯存切成 slot 再發出去」的東西——下一個分頁。',
         en:'Because jit needs static shapes, every concurrently running request still needs its own array. Same as PyTorch: the compute is saved, the space is not. Real sharing needs something that carves all of the GPU memory into slots and hands them out - which is the next tab.'}});
  return F.list;
}

/* ================================================================ tab 6: SGLang */
const CODE_SGLANG = [
  '# RadixAttention is where this idea comes from, and it is on by default',
  'python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B \\',
  '    --mem-fraction-static 0.9        # how big the token pool is',
  '    # --disable-radix-cache          # turn the sharing OFF, to compare',
  '',
  'curl localhost:30000/flush_cache     # drop the tree (between benchmarks)',
  'curl localhost:30000/get_server_info # "cache hit rate" lives here',
  '',
  '# vLLM has the same idea under another name and another structure:',
  '#   --enable-prefix-caching, which hashes fixed-size blocks'
];
function sgStage(o){
  const sh = [];
  const rows = [[{zh:'我們的 RadixCache', en:'our RadixCache'}, {zh:'RadixCache（真的叫這個）', en:'RadixCache, same name'}],
                [{zh:'node.key 上的 token', en:'tokens on node.key'}, {zh:'token_to_kv_pool 的 slot', en:'slots in token_to_kv_pool'}],
                [{zh:'走到哪裡 = 省下的 prefill', en:'how far you walked'}, {zh:'回報的 cache hit rate', en:'the reported cache hit rate'}],
                [{zh:'refs（正在 decode）', en:'refs, still decoding'}, {zh:'protected / lock_ref', en:'a protected node, lock_ref'}]];
  rows.forEach((r, i) => {
    if (i > o.upto) return;
    const y = 4.05 - i * .92;
    sh.push(box(.70, y, 3.30, .62, 'idle', r[0], .26));
    sh.push(S.e(4.15, y + .31, 4.75, y + .31, {s:'ghost', arrow:true, w:.05}));
    sh.push(box(4.90, y, 4.30, .62, i === o.upto ? 'hot' : 'ok', r[1], .26));
  });
  sh.push(fitT(2.35, 5.02, {zh:'這個檔案裡的名字', en:'what this file calls it'}, {c:COL.grey, fs:.26}));
  sh.push(fitT(7.05, 5.02, {zh:'SGLang 伺服器', en:'the SGLang server'}, {c:COL.tealL, fs:.28}));
  return caps(sh, o);
}
function sgFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:sgStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:0,
    cap:{zh:'接下來看 SGLang——今天這個東西就是從它來的', en:'now SGLang, which is where today\'s idea comes from'},
    panels:[{lbl:'default', chips:[chip('radix cache on', 'ok')]}],
    msg:{zh:'先說清楚層級：PyTorch 和 JAX 是你寫模型的函式庫，SGLang 是蓋在上面的**推論伺服器**——它載入一次權重、接很多使用者的請求、決定每一步誰跟誰同一批，並且掌管整張卡的顯存。正因為顯存歸它管，它才做得到前一個分頁做不到的事：**同一份快取真的被多個請求共用**。RadixAttention 就是 SGLang 的這個機制，而且預設就開著。',
         en:'First the levels: PyTorch and JAX are libraries you write a model in; SGLang is an inference server built on top - it loads the weights once, takes requests from many users, decides which of them share each batch, and owns all of the GPU memory. Because it owns the memory it can do what the previous tab could not: let several live requests share one copy of the cache. RadixAttention is that mechanism, and it is on by default.'}});
  p({upto:3, line:5,
    cap:{zh:'名字幾乎一樣，因為就是同一個結構', en:'the names line up, because it is the same structure'},
    panels:[{lbl:'flush_cache', chips:[chip('drops the tree', 'dim')]},
            {lbl:'hit rate', chips:[chip('reported per server', 'ok')]}],
    msg:{zh:'`RadixCache` 這個類別名在 SGLang 裡就是這個名字；我們放在 `node.key` 上的 token，在它那裡是 `token_to_kv_pool` 的 slot 編號；我們的 `refs` 是它的 protected 節點（正在跑的請求把節點鎖住，淘汰不準碰）。`/flush_cache` 把整棵樹丟掉——做 benchmark 時一定要先做這件事，不然第二次跑的數字是上一次的快取在幫忙。`--disable-radix-cache` 則可以關掉它來對照。',
         en:'The class really is called `RadixCache` in SGLang; the tokens we keep on `node.key` are slot indices into its `token_to_kv_pool`; our `refs` is its protected node, locked while a request is decoding so eviction cannot touch it. `/flush_cache` drops the tree - which is the first thing to do before a benchmark, or the second run is being helped by the first one\'s cache. `--disable-radix-cache` turns the sharing off so you can measure what it was worth.'}});
  p({upto:3, line:8,
    cap:{zh:'vLLM 做同一件事，但結構不一樣', en:'vLLM does the same thing with a different structure'},
    panels:[{lbl:'SGLang', chips:[chip('radix tree over tokens', 'ok')]},
            {lbl:'vLLM', chips:[chip('hash of fixed blocks', 'act')]}],
    msg:{zh:'vLLM 的 `--enable-prefix-caching` 目標一樣，但它不是走 radix tree，而是把 KV 切成固定大小的 block、對每個 block 的內容算雜湊，雜湊一樣就重用。兩種做法的差別正好是今天那棵樹的設計取捨：radix tree 可以在任意 token 位置分岔、但要維護樹；block hash 只能以 block 為單位對齊、實作簡單很多。兩邊都必須在 **token id** 上比對，理由就是上一個分頁那個例子。',
         en:'vLLM\'s `--enable-prefix-caching` has the same goal but not the same structure: it cuts the KV into fixed-size blocks, hashes each block\'s contents, and reuses a block when the hash matches. The difference is exactly today\'s design trade: a radix tree can branch at any token but has to be maintained, while block hashing only aligns at block boundaries and is far simpler to implement. Both have to compare token ids, for the reason in the previous tab.'}});
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'RadixAttention', en:'RadixAttention'},
  sub: {zh:'讓開頭相同的請求共用同一份 KV cache：radix tree 查前綴，LRU 只淘汰葉子',
        en:'sharing one KV cache between requests that start the same way: a radix tree for prefixes, LRU on leaves only'},
  tabs: [
    {id:'work', label:{zh:'重複的那一大塊', en:'the repeated part'},
     stage:{zh:'一個請求裡只有尾巴是新的', en:'only the tail of a request is new'},
     view:VIEW, code:CODE_WORK, build:workFrames,
     idea:{zh:'模擬 ' + WORK.convs + ' 段對話、' + WORK.reqs + ' 個請求：每個請求開頭都是同樣 ' + WORK.shared + ' 個 token，第二輪之後還會重送自己前面講過的。全部 ' + WORK.total.toLocaleString() + ' 個 prompt token 裡 **' + WORK.hit.toFixed(1) + '% 是重算**，而 prefill 正是算力受限的那一半。',
           en:WORK.convs + ' conversations, ' + WORK.reqs + ' requests: each begins with the same ' + WORK.shared + ' tokens, and from the second turn on it resends everything the conversation has already said. Of ' + WORK.total.toLocaleString() + ' prompt tokens, ' + WORK.hit.toFixed(1) + '% are a recomputation - and prefill is the compute-bound half.'},
     legend:[['#ff9736', {zh:'所有人共用的', en:'shared by everybody'}],
             ['#9d6bff', {zh:'這段對話自己的歷史', en:'this conversation\'s own history'}],
             ['#12b3b8', {zh:'真正新的', en:'genuinely new'}]]},
    {id:'tree', label:{zh:'radix tree', en:'the radix tree'},
     stage:{zh:'邊上放一串 token，不是一個', en:'a run of tokens per edge, not one'},
     view:VIEW, code:CODE_TREE, build:treeFrames,
     idea:{zh:'從 root 往下走到 token 不再相同為止，走過的長度就是不用重算的前綴；中途分岔就把那條邊切成兩段。同樣的序列，trie 要 ' + WORK.trie.toLocaleString() + ' 個節點，radix tree 只要 ' + WORK.nodes.toLocaleString() + '（' + Math.round(WORK.trie / WORK.nodes) + ' 倍差距），而 slot 從 ' + WORK.plainSlots.toLocaleString() + ' 降到 ' + WORK.slots.toLocaleString() + '。',
           en:'Walk down from the root while the tokens agree, and how far you get is the prefix that need not be recomputed; a request that diverges halfway splits the edge in two. Over the same sequences a trie needs ' + WORK.trie.toLocaleString() + ' nodes against the radix tree\'s ' + WORK.nodes.toLocaleString() + ' (' + Math.round(WORK.trie / WORK.nodes) + 'x), and the slots drop from ' + WORK.plainSlots.toLocaleString() + ' to ' + WORK.slots.toLocaleString() + '.'},
     legend:[['#ff9736', {zh:'共用的前綴', en:'the shared prefix'}],
             ['#9d6bff', {zh:'每段對話自己的分支', en:'each conversation\'s branch'}],
             ['#12b3b8', {zh:'這一輪新加的', en:'added by this turn'}]]},
    {id:'evict', label:{zh:'淘汰', en:'eviction'},
     stage:{zh:'LRU，但只能淘汰葉子', en:'LRU, but leaves only'},
     view:VIEW, code:CODE_EVICT, build:evictFrames,
     idea:{zh:'命中率在池子只有整棵樹 ' + SWEEP[3].frac + '% 時就已經是 ' + SWEEP[3].hit.toFixed(1) + '%（全部是 ' + SWEEP[SWEEP.length - 1].hit.toFixed(1) + '%），因為賺到命中的是共用前綴而它是內部節點、永遠不會被葉子淘汰碰到；但池子裝不下共用前綴時會直接崩到 ' + SWEEP[1].hit.toFixed(1) + '%。一般 LRU 會挑中的那個節點，底下掛著 ' + EVICT.orphaned + ' 個 token。',
           en:'The hit rate is already ' + SWEEP[3].hit.toFixed(1) + '% with a pool of ' + SWEEP[3].frac + '% of the tree (against ' + SWEEP[SWEEP.length - 1].hit.toFixed(1) + '% with all of it), because the part earning hits is the shared prefix and leaf-only LRU never reaches it. Below the size of that prefix it collapses to ' + SWEEP[1].hit.toFixed(1) + '%. The node a plain LRU would pick has ' + EVICT.orphaned + ' tokens hanging below it.'},
     legend:[['#12b3b8', {zh:'命中率', en:'hit rate'}],
             ['#ff5c5c', {zh:'池子太小，整個崩掉', en:'pool too small: it collapses'}]]},
    {id:'trap', label:{zh:'文字前綴 ≠ token 前綴', en:'text prefix is not token prefix'},
     stage:{zh:'差一個字元，token 完全不同', en:'one character apart, nothing in common'},
     view:VIEW, code:CODE_TRAP, build:trapFrames,
     idea:{zh:'tokenizer 會跨過邊界合併，所以兩個共用 ' + TRAP.textPrefix + ' 個字元的字串可以**一個 token 都不共用**。用文字當 key 的快取會交出一個第二個請求裡根本不存在的 token 的 K/V，而且不會報錯。比對一律在 token id 上做。',
           en:'A tokeniser merges across boundaries, so two strings sharing ' + TRAP.textPrefix + ' characters can share no tokens at all. A cache keyed on text would hand back the K and V of a token the second request never contained, and nothing would raise. Prefix matching belongs on token ids.'},
     legend:[['#12b3b8', {zh:'第一個請求的 token', en:'the first request\'s tokens'}],
             ['#ff5c5c', {zh:'第二個請求的 token', en:'the second request\'s tokens'}]]},
    {id:'fw', label:{zh:'PyTorch 與 JAX 裡', en:'in PyTorch and JAX'},
     stage:{zh:'省得掉計算，省不掉空間', en:'the compute is saved, the space is not'},
     view:VIEW, build:fwFrames,
     variants:[{zh:'PyTorch', en:'PyTorch'}, {zh:'JAX', en:'JAX'}],
     get code(){ return [CODE_TORCH, CODE_JAX][curV()]; },
     idea:{zh:'這兩個都是「你寫模型」的函式庫，cache 是你手上的物件。你可以把 system prompt 先算一次再複製給每個請求（PyTorch 的 deepcopy、JAX 的固定形狀陣列），prefill 確實省了——但每個在跑的請求還是各帶一份，所以 ' + WORK.plainSlots.toLocaleString() + ' → ' + WORK.slots.toLocaleString() + ' 那個省空間的部分拿不到。',
           en:'Both are libraries you write a model in, where the cache is an object in your hands. You can compute the system prompt once and copy it per request (deepcopy in PyTorch, a padded array in JAX) and the prefill really is saved - but every live request still carries its own copy, so the memory half of today\'s result (' + WORK.plainSlots.toLocaleString() + ' slots down to ' + WORK.slots.toLocaleString() + ') is out of reach.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'函式庫裡的做法', en:'what the library does'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'sglang', label:{zh:'再來看 SGLang', en:'now SGLang'},
     stage:{zh:'伺服器擁有顯存，所以共用才可能', en:'the server owns the memory, so sharing becomes possible'},
     view:VIEW, code:CODE_SGLANG, build:sgFrames,
     idea:{zh:'SGLang 不是函式庫而是推論伺服器，顯存歸它管，所以它能讓多個同時在跑的請求共用同一棵樹——RadixAttention 這個名字就是從這裡來的，而且預設開啟。名字幾乎一對一：`RadixCache`、`token_to_kv_pool` 的 slot、protected 節點（＝我們的 refs）。vLLM 的 `--enable-prefix-caching` 目標相同但改用固定大小 block 的雜湊。',
           en:'SGLang is not a library but an inference server, and because it owns the memory it can let several live requests share one tree - which is where the name RadixAttention comes from, and it is on by default. The names line up almost one to one: `RadixCache`, slots in `token_to_kv_pool`, protected nodes for our `refs`. vLLM\'s `--enable-prefix-caching` has the same goal using hashes of fixed-size blocks instead.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'伺服器裡的名字', en:'what the server calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]}
  ]
};
