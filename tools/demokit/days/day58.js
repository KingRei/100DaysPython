// DAY: 58
// TITLE_ZH: PagedAttention 與顯存管理：KV cache 的分頁配置器
// TITLE_EN: PagedAttention and GPU memory: the KV cache as a paging allocator
// SUB_ZH: kernel 那邊的改動很小——K 和 V 改成透過 block table 讀。真正的工作全在外面：固定大小的 block 怎麼發、池子滿了怎麼拒絕、同一個 prompt 的多個樣本怎麼共用、以及要從正在跑的請求手上把 block 收回來時怎麼辦。
// SUB_EN: The kernel change is small - read K and V through a block table instead of a flat array. The work is everything around it: handing out fixed-size blocks, refusing requests when the pool is full, sharing blocks between samples of one prompt, and taking blocks back from a request that is already running.
// FOLDER: day%2058%20-%20paged%20attention%20memory
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
const WASTE = [{"block": 1, "waste": 0.0, "tables": 201736, "slots": 201736}, {"block": 8, "waste": 0.4, "tables": 25318, "slots": 202544}, {"block": 16, "waste": 0.9, "tables": 12721, "slots": 203536}, {"block": 64, "waste": 3.5, "tables": 3266, "slots": 209024}, {"block": 256, "waste": 12.5, "tables": 901, "slots": 230656}];
const SAMP = {"prompt": 900, "samples": 4, "blocks": 57, "naive": 228, "shared": 60, "saved": 74.0, "cow": 3};
const PRE = {"nR": 54, "nS": 51, "recompute": 3840, "swap": 5696, "finished": 60, "stepsR": 104, "stepsS": 94};
const BUG = {"leakedBlocks": 617, "leakedTokens": 9872, "left": 55664, "capacity": 65536, "tokens": 70, "blocks": 5, "padded": 80, "garbage": 10, "share": 12.0};


/* ================================================================ tab 1: fragmentation */
const CODE_FRAG = [
  '# contiguous: one range per request, and it cannot move',
  'for (start, length) in self.free:        # best fit',
  '    if length >= need: take it',
  '',
  '# paged: any free block will do',
  'if need_blocks > len(self.free): refuse',
  'blocks = [self.free.pop() for _ in range(need_blocks)]',
  'self.table[rid].extend(blocks)          # the block table IS the allocation'
];
function fragStage(o){
  const sh = [];
  const segs = [[0, 1.6, 'hot'], [1.6, .5, null], [2.1, 2.2, 'ok'], [4.3, .6, null],
                [4.9, 1.9, 'act'], [6.8, .7, null], [7.5, 1.6, 'hot']];
  segs.forEach(g => {
    sh.push(S.r(.55 + g[0], 3.55, g[1] - .06, .62, g[2] || 'idle', null));
  });
  sh.push(fitT(.55, 4.40, {zh:'一個請求一段連續空間', en:'one contiguous range per request'},
               {anchor:'start', c:COL.grey, fs:.26}));
  [2.25, 5.15, 7.70].forEach(x => sh.push(fitT(x, 3.30, {zh:'空的', en:'free'}, {c:COL.grey, fs:.22})));
  if (o.upto >= 1){
    sh.push(fitT(4.9, 2.85, {zh:'需要 1.8 格的請求被拒絕——雖然空的總共有 1.8 格',
                             en:'a request needing 1.8 units is refused, although 1.8 are free'},
                 {c:COL.red, fs:.28}));
  }
  if (o.upto >= 2){
    const cols = ['hot', 'hot', 'ok', null, 'act', 'hot', null, 'ok', 'act', null,
                  'hot', 'ok', null, 'act', 'hot', null];
    cols.forEach((c, i) => sh.push(S.r(.55 + i * .57, 1.55, .50, .62, c || 'idle', null)));
    sh.push(fitT(.55, 2.35, {zh:'固定大小的 block', en:'fixed-size blocks'},
                 {anchor:'start', c:COL.grey, fs:.26}));
    sh.push(fitT(4.9, 1.10, {zh:'同樣四個空 block，誰都能用、順序也無所謂',
                             en:'the same four free blocks are usable by anyone, in any order'},
                 {c:COL.tealL, fs:.28}));
  }
  return caps(sh, o);
}
function fragFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:fragStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'連續配置：一個請求一段，而且搬不走', en:'contiguous: one range per request, and it cannot move'},
    panels:[{lbl:'requests', chips:[chip(String(FRAG.reqs), 'dim')]},
            {lbl:'pool', chips:[chip(FRAG.capacity.toLocaleString() + ' slots', 'dim')]}],
    msg:{zh:'先給連續配置最好的待遇：假設我們**事先知道**每個請求最後會多長，所以完全不多保留。即使如此，請求結束的時間不一樣，空出來的洞就會散在各處。',
         en:'Give the contiguous allocator the kindest possible case: pretend we know each request\'s final length in advance, so it never over-reserves. Even then, requests finish at different times and the holes they leave are scattered.'}});
  p({upto:1, line:2,
    cap:{zh:'於是會發生「有空間但拿不出來」', en:'and then: room exists, but not in one piece'},
    panels:[{lbl:'refused', chips:[chip(String(FRAG.rejC), 'bad')]},
            {lbl:'of those, with room free', chips:[chip(String(FRAG.rejFrag), 'bad')]}],
    msg:{zh:'實測：連續配置拒絕了 ' + FRAG.rejC + ' 個請求，而這 ' + FRAG.rejFrag + ' 個**全部**都是在「池子裡的空位加起來夠、只是不連續」的情況下被拒絕的。這就是外部碎片（external fragmentation），作業系統課本上那個。',
         en:'Measured: the contiguous pool refused ' + FRAG.rejC + ' requests, and all ' + FRAG.rejFrag + ' of them were refused while the pool had enough free space in total - just not in one piece. That is external fragmentation, straight out of the operating systems textbook.'}});
  p({upto:2, line:5,
    cap:{zh:'固定大小的 block 讓這個問題消失', en:'fixed-size blocks make the problem go away'},
    panels:[{lbl:'paged refused', chips:[chip(String(FRAG.rejP), 'ok')]},
            {lbl:'occupancy', chips:[chip(FRAG.utilC.toFixed(1) + '%', 'dim'),
                                     chip(FRAG.utilP.toFixed(1) + '%', 'ok')]}],
    msg:{zh:'改成固定大小的 block 之後，「空間的形狀」這件事就不存在了——任何一個空 block 都一樣好用，所以配置只剩「夠不夠」這一個問題。同一份流量只拒絕 ' + FRAG.rejP + ' 個，平均佔用率從 ' + FRAG.utilC.toFixed(1) + '% 升到 ' + FRAG.utilP.toFixed(1) + '%。代價在下一個分頁。',
         en:'With fixed-size blocks there is no such thing as the wrong shape of free memory: any free block is as good as any other, so allocation reduces to "is there enough". The same traffic now refuses only ' + FRAG.rejP + ' requests, and average occupancy rises from ' + FRAG.utilC.toFixed(1) + '% to ' + FRAG.utilP.toFixed(1) + '%. The price is the next tab.'}});
  return F.list;
}

/* ================================================================ tab 2: block size */
const CODE_BLOCK = [
  'blocks_for(tokens) = ceil(tokens / block)',
  '',
  '# the last block of every request is only partly used:',
  'wasted = blocks_for(n) * block - n        # internal fragmentation',
  '',
  '# and every block needs an entry in that request\'s block table',
  'table_entries = blocks_for(n)'
];
function blockStage(o){
  const sh = [], X0 = 1.55, Y0 = 1.75, WD = 6.9, HT = 2.55;
  sh.push(S.e(X0, Y0, X0 + WD, Y0, {s:'ghost', arrow:false, w:.04}));
  sh.push(S.e(X0, Y0, X0, Y0 + HT, {s:'ghost', arrow:false, w:.04}));
  const xs = WASTE.map((_, i) => X0 + WD * i / (WASTE.length - 1));
  const top = Math.max.apply(null, WASTE.map(r => r.waste)) * 1.12;
  const tmax = Math.max.apply(null, WASTE.map(r => r.tables));
  for (let i = 1; i < WASTE.length; i++){
    if (i > o.upto) break;
    sh.push(S.e(xs[i - 1], Y0 + HT * WASTE[i - 1].waste / top,
                xs[i], Y0 + HT * WASTE[i].waste / top, {s:'bad', arrow:false, w:.06}));
    sh.push(S.e(xs[i - 1], Y0 + HT * WASTE[i - 1].tables / tmax,
                xs[i], Y0 + HT * WASTE[i].tables / tmax, {s:'ok', arrow:false, w:.06}));
  }
  WASTE.forEach((r, i) => sh.push(fitT(xs[i], Y0 - .28, String(r.block), {c:COL.grey, fs:.24})));
  sh.push(fitT(X0 + WD / 2, Y0 - .66, {zh:'每個 block 幾個 token', en:'tokens per block'},
               {c:COL.grey, fs:.26}));
  sh.push(fitT(X0 + .12, Y0 + HT + .30, {zh:'紅：浪費掉的比例　　藍：block table 的項目數',
                                         en:'red: share of slots wasted    teal: block-table entries'},
               {anchor:'start', c:COL.grey, fs:.25}));
  return caps(sh, o);
}
function blockFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:blockStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:3,
    cap:{zh:'分頁的代價：每個請求最後一個 block 的尾巴', en:'what paging costs: the tail of each request\'s last block'},
    panels:[{lbl:'block 1', chips:[chip(WASTE[0].waste.toFixed(1) + '% wasted', 'ok'),
                                   chip(WASTE[0].tables.toLocaleString() + ' entries', 'bad')]}],
    msg:{zh:'一個 token 一個 block 的話什麼都不浪費，但每個 token 都要一筆 block table 的項目（這份流量是 ' + WASTE[0].tables.toLocaleString() + ' 筆）。這是內部碎片（internal fragmentation）與索引成本的取捨。',
         en:'One token per block wastes nothing, and needs one block-table entry per token - ' + WASTE[0].tables.toLocaleString() + ' of them for this traffic. That is the trade: internal fragmentation against the cost of indexing.'}});
  p({upto:WASTE.length - 1, line:4,
    cap:{zh:'block 越大，浪費越多、索引越少', en:'bigger blocks waste more and index less'},
    panels:WASTE.filter((_, i) => i % 2 === 0).map(r =>
      ({lbl:'block ' + r.block, chips:[chip(r.waste.toFixed(1) + '%', r.waste > 5 ? 'bad' : 'ok'),
                                       chip(r.tables.toLocaleString(), 'dim')]})),
    msg:{zh:'256 個 token 一個 block：索引少了 224 倍，但丟掉池子的 ' + WASTE[WASTE.length - 1].waste.toFixed(1) + '%。實務上大多落在 16 或 32——而且理由不只是浪費：**block 也是前綴共用能對齊的最小單位**（前一天那棵樹），block 越大、能共用的前綴就越粗。',
         en:'At 256 tokens per block the index is 224 times smaller and ' + WASTE[WASTE.length - 1].waste.toFixed(1) + '% of the pool is thrown away. Real servers sit at 16 or 32, and not only because of the waste: the block is also the granularity at which prefixes can be shared (the previous day\'s tree), so a bigger block means coarser sharing.'}});
  return F.list;
}

/* ================================================================ tab 3: sharing */
const CODE_SHARE = [
  'def share(self, src, dst, tokens):      # n samples of one prompt',
  '    shared = self.table[src][:blocks_for(tokens)]',
  '    for b in shared: self.refs[b] += 1  # a reference count keeps them alive',
  '    self.table[dst] = list(shared)',
  '',
  '# the LAST prompt block is only partly full, so the first write',
  '# by each sample copies that one block: copy-on-write, as in fork()'
];
function shareStage(o){
  const sh = [], cw = .46;
  const nb = 8;
  for (let s = 0; s < 4; s++){
    if (s > o.upto) break;
    const y = 4.35 - s * .78;
    sh.push(fitT(1.35, y + .26, {zh:'樣本 ' + s, en:'sample ' + s},
                 {anchor:'end', c:COL.grey, fs:.26}));
    for (let b = 0; b < nb; b++){
      const last = (b === nb - 1);
      const st = o.cow && last ? 'act' : (s === 0 || !o.shared ? 'hot' : 'ok');
      sh.push(S.r(1.55 + b * (cw + .06), y, cw, .52, st, null));
    }
    if (o.shared && s > 0){
      sh.push(fitT(1.55 + nb * (cw + .06) + .20, y + .26,
                   o.cow ? {zh:'共用前 7 個，最後一個自己複製', en:'shares 7, copies the last'}
                         : {zh:'指向同樣的 block', en:'points at the same blocks'},
                   {anchor:'start', c:COL.tealL, fs:.24}));
    }
  }
  sh.push(fitT(4.9, 5.05, {zh:'一個 prompt、四個樣本（圖中每格是一個 block）',
                           en:'one prompt, four samples (each cell is a block)'},
               {c:COL.grey, fs:.26}));
  return caps(sh, o);
}
function shareFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:shareStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:3, shared:false, line:0,
    cap:{zh:'同一個 prompt 要四個答案', en:'four answers to the same prompt'},
    panels:[{lbl:'naive', chips:[chip(SAMP.naive + ' blocks', 'bad')]}],
    msg:{zh:'最直覺的寫法是每個樣本自己一份 prompt 的 KV：' + SAMP.samples + ' × ' + SAMP.blocks + ' ＝ ' + SAMP.naive + ' 個 block。但這 ' + SAMP.samples + ' 份內容**完全一樣**，因為 prompt 一樣。',
         en:'The obvious implementation gives each sample its own copy of the prompt\'s KV: ' + SAMP.samples + ' x ' + SAMP.blocks + ' = ' + SAMP.naive + ' blocks. All ' + SAMP.samples + ' copies hold identical numbers, because the prompt is identical.'}});
  p({upto:3, shared:true, line:2,
    cap:{zh:'讓它們指向同一批 block，用 refcount 保活', en:'point them at the same blocks, kept alive by a reference count'},
    panels:[{lbl:'with sharing', chips:[chip(SAMP.shared + ' blocks', 'ok'),
                                        chip(SAMP.saved.toFixed(0) + '% less', 'ok')]}],
    msg:{zh:'block table 只是一串 block 編號，所以「共用」就是把同樣的編號抄進另一張表，再把那些 block 的 refcount 加一。' + SAMP.naive + ' → ' + SAMP.shared + ' 個 block，少了 ' + SAMP.saved.toFixed(0) + '%。',
         en:'A block table is just a list of block ids, so sharing means copying those ids into another table and incrementing each block\'s reference count. ' + SAMP.naive + ' blocks become ' + SAMP.shared + ', ' + SAMP.saved.toFixed(0) + '% fewer.'}});
  p({upto:3, shared:true, cow:true, line:6,
    cap:{zh:'只有最後那個「沒裝滿」的 block 要複製', en:'only the last, partly-full block has to be copied'},
    panels:[{lbl:'copy-on-write', chips:[chip(SAMP.cow + ' blocks', 'act')]}],
    msg:{zh:'prompt 的最後一個 block 沒裝滿，而每個樣本接下來要把**自己的**下一個 token 寫進那個 block——所以第一次寫入時各自複製一份，其餘的 block 繼續共用。複製一個 block 而不是整個 prompt，這就是 copy-on-write，跟 fork() 是同一招。',
         en:'The prompt\'s last block is only partly full, and each sample is about to write its own next token into it - so on that first write each sample copies that one block, while the rest stay shared. Copying one block instead of the whole prompt is copy-on-write, the same trick fork() uses.'}});
  return F.list;
}

/* ================================================================ tab 4: preemption */
const CODE_PREEMPT = [
  'if waiting and running and pool.free_tokens() < waiting[0].need:',
  '    victim = running.pop()           # newest first: least work invested',
  '    pool.release(victim.id)',
  '    if policy == "recompute":',
  '        victim.done = 0              # prefill it again later',
  '    else:',
  '        swap_out(victim)             # copy the blocks to host memory',
  '    waiting.append(victim)'
];
function preStage(o){
  const sh = [], base = 2.70, maxh = 1.90, bw = 1.70;
  const bars = [{v:PRE.recompute, t:{zh:'重算：再 prefill 一次', en:'recompute: prefilled again'}, st:'bad'},
                {v:PRE.swap, t:{zh:'換出：搬去主機再搬回', en:'swap: copied out and back'}, st:'act'}];
  const top = Math.max(PRE.recompute, PRE.swap) * 1.15;
  bars.forEach((b, i) => {
    if (i > o.upto) return;
    const x = 2.15 + i * 3.10, h = Math.max(.05, maxh * b.v / top);
    sh.push(S.r(x, base, bw, h, b.st, null));
    sh.push(fitT(x + bw / 2, base + h + .26, b.v.toLocaleString() + ' tokens',
                 {c:b.st === 'bad' ? COL.red : COL.hot, fs:.28}));
    sh.push(fitT(x + bw / 2, base - .32, b.t, {c:COL.grey, fs:.26}));
  });
  sh.push(fitT(4.9, 5.00, {zh:'池子滿了，必須從正在跑的請求手上收回 block',
                           en:'the pool is full: blocks have to be taken back from a running request'},
               {c:COL.grey, fs:.26}));
  side(sh, 1.10, 1.75, o.notes || [], .44);
  return caps(sh, o);
}
function preFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:preStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:0, line:1,
    cap:{zh:'先決定「犧牲誰」', en:'first, who gets interrupted'},
    panels:[{lbl:'preemptions', chips:[chip(String(PRE.nR), 'bad')]},
            {lbl:'victim rule', chips:[chip('newest first', 'dim')]}],
    msg:{zh:'60 個請求對上 4,096 個 slot 的池子，中途一定會滿。挑最新的那個下手，因為它身上投入的計算最少——vLLM 用的也是這條規則。發生了 ' + PRE.nR + ' 次。',
         en:'60 requests against a 4,096-slot pool will run out. The victim is the newest request, because it has the least work invested in it - which is also what vLLM does. It happened ' + PRE.nR + ' times here.'}});
  p({upto:1, line:4,
    cap:{zh:'再決定「用什麼付」', en:'then, which currency to pay in'},
    notes:[[{zh:'兩種政策都把全部 ' + PRE.finished + ' 個請求跑完了', en:'both policies finished all ' + PRE.finished + ' requests'}, COL.grey, .26],
           [{zh:'只是一個花算力（' + PRE.recompute.toLocaleString() + ' 個 token 重 prefill）',
             en:'one pays in compute (' + PRE.recompute.toLocaleString() + ' tokens re-prefilled)'}, COL.red, .26],
           [{zh:'一個花頻寬（' + PRE.swap.toLocaleString() + ' 個 token 搬出去再搬回來）',
             en:'the other in bandwidth (' + PRE.swap.toLocaleString() + ' tokens moved out and back)'}, COL.hot, .26]],
    panels:[{lbl:'recompute', chips:[chip(PRE.recompute.toLocaleString(), 'bad')]},
            {lbl:'swap', chips:[chip(PRE.swap.toLocaleString(), 'act')]}],
    msg:{zh:'收回來的 block 裡的東西要嘛丟掉、之後重算，要嘛先複製到主機記憶體、回來再複製回去。跟前一天的 HiCache 完全同一個取捨，只是換了一層：**算力 vs 頻寬**。前綴短的時候重算比較划算，長到一定程度就該用搬的。而最糟的第三種做法是什麼都不做——池子塞滿、每個請求都做到一半，然後大家一起卡死。',
         en:'What was in those blocks is either dropped and recomputed later, or copied to host memory and copied back. It is exactly the trade from the previous day\'s HiCache, one level up: compute against bandwidth. Recomputing wins for short prefixes, moving wins once the prefix is long enough. The third option - do nothing - means the pool fills with half-finished requests and everything deadlocks.'}});
  return F.list;
}

/* ================================================================ tab 5: the bugs */
const CODE_BUGS = [
  '# bug 1: a failure path that forgets to free',
  'try:    run(request)',
  'except: return error          # <- the blocks are still allocated',
  '',
  '# bug 2: the last block is partly full',
  'blocks = ceil(70 / 16) = 5    # 80 slots for 70 tokens',
  'attend(..., length=80)        # 10 slots of somebody else\'s old KV'
];
function bugStage(o){
  const sh = [];
  if (o.mode === 'leak'){
    const total = 40, lost = Math.round(total * BUG.leakedTokens / BUG.capacity);
    for (let i = 0; i < total; i++){
      sh.push(S.r(.70 + (i % 20) * .44, 3.60 - Math.floor(i / 20) * .70, .38, .58,
                  i < lost ? 'bad' : 'ok', null));
    }
    sh.push(fitT(4.9, 4.60, {zh:'池子裡的 slot（每格 ' + Math.round(BUG.capacity / total).toLocaleString() + ' 個）',
                             en:'the pool, ' + Math.round(BUG.capacity / total).toLocaleString() + ' slots per cell'},
                 {c:COL.grey, fs:.26}));
    sh.push(fitT(4.9, 2.35, {zh:'紅色的永遠回不來：' + BUG.leakedTokens.toLocaleString() + ' / ' + BUG.capacity.toLocaleString() + ' 個 slot',
                             en:'the red cells never come back: ' + BUG.leakedTokens.toLocaleString() + ' of ' + BUG.capacity.toLocaleString() + ' slots'},
                 {c:COL.red, fs:.30}));
  } else {
    const cw = .52;
    for (let i = 0; i < BUG.padded; i++){
      const st = i < BUG.tokens ? 'ok' : 'bad';
      sh.push(S.r(.70 + (i % 16) * cw, 3.70 - Math.floor(i / 16) * .70, cw - .06, .58,
                  st, null));
    }
    sh.push(fitT(4.9, 4.70, {zh:BUG.tokens + ' 個 token 佔 ' + BUG.blocks + ' 個 block ＝ ' + BUG.padded + ' 個 slot',
                             en:BUG.tokens + ' tokens occupy ' + BUG.blocks + ' blocks = ' + BUG.padded + ' slots'},
                 {c:COL.grey, fs:.28}));
    sh.push(fitT(4.9, 1.95, {zh:'最後 ' + BUG.garbage + ' 格是上一個用過這些 slot 的請求留下來的東西',
                             en:'the last ' + BUG.garbage + ' slots hold whatever the previous owner left there'},
                 {c:COL.red, fs:.28}));
  }
  return caps(sh, o);
}
function bugFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:bugStage(o), panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({mode:'leak', line:2,
    cap:{zh:'第一個：有一條路徑忘了還 block', en:'one: a path that forgets to give the blocks back'},
    panels:[{lbl:'leaked', chips:[chip(BUG.leakedBlocks + ' blocks', 'bad'),
                                  chip(BUG.leakedTokens.toLocaleString() + ' slots', 'bad')]},
            {lbl:'pool left', chips:[chip(BUG.left.toLocaleString(), 'dim')]}],
    msg:{zh:'200 個請求、十分之一在某條失敗路徑上結束而沒有釋放：漏掉 ' + BUG.leakedBlocks + ' 個 block ＝ ' + BUG.leakedTokens.toLocaleString() + ' 個 slot，池子從 ' + BUG.capacity.toLocaleString() + ' 剩下 ' + BUG.left.toLocaleString() + '。什麼都不會報錯，伺服器只是**越來越收不下請求**，回報上來的症狀是「跑幾個小時之後吞吐量就掉了」。抓法是一條不變式：閒置時「空的 block ＋ 所有 block table 裡的 block」必須等於池子大小。',
         en:'200 requests with one in ten ending on a failure path that forgets to free: ' + BUG.leakedBlocks + ' blocks, ' + BUG.leakedTokens.toLocaleString() + ' slots, and the pool falls from ' + BUG.capacity.toLocaleString() + ' to ' + BUG.left.toLocaleString() + '. Nothing raises; the server simply accepts fewer and fewer requests, and the bug is reported as "throughput degrades after a few hours". The test is an invariant checked when idle: free blocks plus blocks in block tables must equal the pool.'}});
  p({mode:'tail', line:5,
    cap:{zh:'第二個：沒告訴 kernel 最後一個 block 裝了多少', en:'two: not telling the kernel how full the last block is'},
    panels:[{lbl:'tokens / slots', chips:[chip(String(BUG.tokens), 'ok'), chip(String(BUG.padded), 'dim')]},
            {lbl:'garbage', chips:[chip(BUG.garbage + ' slots', 'bad'),
                                   chip(BUG.share.toFixed(0) + '% of the block', 'bad')]}],
    msg:{zh:BUG.tokens + ' 個 token 配 ' + BUG.blocks + ' 個 block ＝ ' + BUG.padded + ' 個 slot，最後 ' + BUG.garbage + ' 格裡是**別人一小時前用過的 KV**。kernel 需要兩樣東西：block table 和真正的序列長度，而 mask 必須用後者。用成補滿的長度也不會崩——softmax 照樣加總為 1，模型只是把陳年的 KV 也平均了進去。',
         en:BUG.tokens + ' tokens take ' + BUG.blocks + ' blocks = ' + BUG.padded + ' slots, and the last ' + BUG.garbage + ' hold KV that belonged to some other request an hour ago. The kernel needs both the block table and the true sequence length, and the mask has to use the second one. Use the padded length instead and nothing crashes: the softmax still sums to one, and the model quietly averages in stale KV.'}});
  return F.list;
}

/* ================================================================ tab 6: vLLM */
const CODE_VLLM = [
  'vllm serve meta-llama/Llama-3.1-8B \\',
  '    --block-size 16 \\                # tokens per block  (tab 2)',
  '    --gpu-memory-utilization 0.90 \\  # share of HBM the pool may take',
  '    --max-num-seqs 256 \\             # cap on concurrent sequences',
  '    --swap-space 4 \\                 # GiB of host memory for preemption',
  '    --enable-prefix-caching          # reuse blocks across requests',
  '',
  '# or, in Python:',
  'llm = LLM(model=..., block_size=16, gpu_memory_utilization=0.9)',
  'out = llm.generate(prompt, SamplingParams(n=4))   # n=4 shares the prompt blocks',
  '',
  '# watch for: "Sequence group ... is preempted by PreemptionMode.RECOMPUTE"'
];
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
const VLLM_ROWS = [
  [{zh:'PagedPool.block', en:'PagedPool.block'}, {zh:'--block-size（預設 16）', en:'--block-size (16 by default)'}],
  [{zh:'池子有幾個 block', en:'how many blocks in the pool'}, {zh:'--gpu-memory-utilization', en:'--gpu-memory-utilization'}],
  [{zh:'table[rid]', en:'table[rid]'}, {zh:'BlockTable / block_tables', en:'BlockTable / block_tables'}],
  [{zh:'refs[b]', en:'refs[b]'}, {zh:'ref_count ＋ copy-on-write', en:'ref_count and copy-on-write'}],
  [{zh:'preemption()', en:'preemption()'}, {zh:'RECOMPUTE 或 SWAP', en:'PreemptionMode RECOMPUTE / SWAP'}]
];
function vllmFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:VLLM_ROWS, title:'vLLM'}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:1,
    cap:{zh:'vLLM：PagedAttention 就是從這裡來的', en:'vLLM: this is where PagedAttention comes from'},
    panels:[{lbl:'--block-size', chips:[chip('16', 'hot')]},
            {lbl:'pool', chips:[chip('profiled at startup', 'ok')]}],
    msg:{zh:'vLLM 啟動時會先做一次 profiling：跑一輪測量權重和活化值佔掉多少，剩下的乘上 `--gpu-memory-utilization` 就是 KV 池，再除以「每個 block 幾個 byte」得到 block 數——跟前一天那個除法是同一個。`--block-size` 就是這個檔案裡的 `PagedPool.block`，預設 16。',
         en:'vLLM profiles at startup: it runs a pass to measure what the weights and activations take, multiplies what is left by `--gpu-memory-utilization` to get the KV pool, and divides by the bytes per block to get the block count - the same division as the previous day. `--block-size` is this file\'s `PagedPool.block`, 16 by default.'}});
  p({upto:3, line:9,
    cap:{zh:'n 個樣本共用 prompt，就是上面的 refcount', en:'n samples sharing a prompt is the refcount above'},
    panels:[{lbl:'SamplingParams(n=4)', chips:[chip('shares prompt blocks', 'ok')]},
            {lbl:'--enable-prefix-caching', chips:[chip('shares across requests', 'ok')]}],
    msg:{zh:'`SamplingParams(n=4)` 不會把 prompt 的 KV 複製四份，它讓四個序列指向同一批 block、refcount 加到 4，只有最後那個沒裝滿的 block 在寫入時複製。而 `--enable-prefix-caching` 把同一招用在**不同請求之間**：以 block 為單位算雜湊，一樣就重用——跟前一天 SGLang 的 radix tree 是同一個目標的兩種實作。',
         en:'`SamplingParams(n=4)` does not copy the prompt KV four times: the four sequences point at the same blocks with a reference count of 4, and only the partly-full last block is copied on write. `--enable-prefix-caching` applies the same idea across different requests by hashing whole blocks - the other implementation of the previous day\'s radix tree.'}});
  p({upto:4, line:11,
    cap:{zh:'而池子滿了的時候，它會告訴你', en:'and when the pool runs out, it tells you'},
    panels:[{lbl:'--swap-space', chips:[chip('4 GiB', 'dim')]},
            {lbl:'log line', chips:[chip('preempted by RECOMPUTE', 'bad')]}],
    msg:{zh:'日誌裡那行 `Sequence group ... is preempted by PreemptionMode.RECOMPUTE` 就是上一個分頁在做的事：有請求被收回 block、之後重算。偶爾出現沒關係，一直刷代表池子配得太小或 `--max-num-seqs` 開太大。`--swap-space` 給的是另一種付法——搬到主機記憶體再搬回來。',
         en:'The log line `Sequence group ... is preempted by PreemptionMode.RECOMPUTE` is the previous tab happening for real: a request had its blocks taken back and will be prefilled again. Occasional lines are fine; a stream of them means the pool is too small for the concurrency you asked for. `--swap-space` buys the other currency - host memory to move blocks into and back.'}});
  return F.list;
}

/* ================================================================ tab 7: SGLang */
const CODE_SGLANG = [
  'python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B \\',
  '    --mem-fraction-static 0.9 \\      # share of HBM for the KV pool',
  '    --page-size 1 \\                  # tokens per page (1 by default)',
  '    --max-running-requests 256 \\     # cap on the decode batch',
  '    --chunked-prefill-size 8192 \\    # split long prefills',
  '    --max-total-tokens 200000        # or set the pool size directly',
  '',
  '# the pool and its index tables:',
  '#   token_to_kv_pool   the slots themselves',
  '#   req_to_token_pool  one row per request: its token -> slot mapping',
  '',
  'curl localhost:30000/get_server_info   # pool size, #queued, #running'
];
const SG_ROWS = [
  [{zh:'PagedPool.block', en:'PagedPool.block'}, {zh:'--page-size（預設 1）', en:'--page-size (1 by default)'}],
  [{zh:'池子的大小', en:'the size of the pool'}, {zh:'--mem-fraction-static / --max-total-tokens', en:'--mem-fraction-static, --max-total-tokens'}],
  [{zh:'table[rid]', en:'table[rid]'}, {zh:'req_to_token_pool 的一列', en:'a row of req_to_token_pool'}],
  [{zh:'slot 本身', en:'the slots themselves'}, {zh:'token_to_kv_pool', en:'token_to_kv_pool'}],
  [{zh:'preemption()', en:'preemption()'}, {zh:'retract：收回並重算', en:'retract: take back and recompute'}]
];
function sglangFrames(){
  const F = new Frames();
  const p = (o) => F.push({shapes:mapStage(Object.assign({rows:SG_ROWS, title:{zh:'SGLang', en:'SGLang'}}, o)),
                           panels:o.panels, line:o.line, msg:o.msg, view:VIEW});
  p({upto:1, line:2,
    cap:{zh:'SGLang：同樣的零件，不同的預設', en:'SGLang: the same pieces, different defaults'},
    panels:[{lbl:'--page-size', chips:[chip('1', 'hot')]},
            {lbl:'--mem-fraction-static', chips:[chip('0.9', 'ok')]}],
    msg:{zh:'SGLang 的「page」預設是 **1 個 token**——也就是第 2 個分頁那張表最左邊那一欄：完全沒有內部碎片，代價是索引表很長。它付得起，因為它的索引是一張 `req_to_token_pool` 的大表（GPU 上的 2D tensor），不是每個請求一個 Python list。',
         en:'SGLang\'s page is one token by default - the leftmost column of the table in tab 2: zero internal fragmentation, at the cost of a very long index. It can afford that because the index is one big `req_to_token_pool` tensor on the GPU rather than a Python list per request.'}});
  p({upto:3, line:8,
    cap:{zh:'兩張表：slot 本身，和誰拿了哪些 slot', en:'two tables: the slots, and who holds which'},
    panels:[{lbl:'token_to_kv_pool', chips:[chip('the slots', 'ok')]},
            {lbl:'req_to_token_pool', chips:[chip('one row per request', 'ok')]}],
    msg:{zh:'`token_to_kv_pool` 是 slot 本身（每層的 K 和 V），`req_to_token_pool` 是「第 i 個請求的第 j 個 token 住在哪個 slot」。這兩張表合起來就是這個檔案裡的 `PagedPool.free` 加 `PagedPool.table`——只是住在 GPU 上，而且用 tensor 索引而不是 Python 迴圈。',
         en:'`token_to_kv_pool` holds the slots themselves (K and V per layer), and `req_to_token_pool` says where request i\'s token j lives. Together they are this file\'s `PagedPool.free` plus `PagedPool.table` - living on the GPU and indexed with tensors rather than Python loops.'}});
  p({upto:4, line:5,
    cap:{zh:'滿了的時候它選擇重算', en:'when it runs out, it chooses to recompute'},
    panels:[{lbl:'retract', chips:[chip('recompute', 'bad')]},
            {lbl:'why', chips:[chip('the radix tree keeps the prefix', 'ok')]}],
    msg:{zh:'SGLang 的做法叫 **retract**：把請求收回等待佇列、釋放它的 slot，之後重跑。它不預設換出到主機記憶體，理由正好是前一天那棵樹——被收回的請求的前綴多半還在 radix cache 裡，所以「重算」其實只重算沒命中的那一小段，比搬資料划算。`--chunked-prefill-size` 則是另一個防線：把長 prefill 切塊，讓它不會一次吃掉整個池子、也不會讓 decode 停太久。',
         en:'SGLang calls it **retract**: the request goes back to the waiting queue, its slots are freed, and it runs again later. It does not swap to host memory by default, and the reason is the previous day\'s tree - the retracted request\'s prefix is usually still in the radix cache, so "recompute" really means recomputing only the part that missed. `--chunked-prefill-size` is the other defence: slice a long prefill so it cannot take the whole pool at once, or stall decode for too long.'}});
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title: {zh:'PagedAttention 與顯存管理', en:'PagedAttention and GPU memory'},
  sub: {zh:'KV cache 的分頁配置器：block table、refcount、以及池子滿了要怎麼辦',
        en:'a paging allocator for the KV cache: block tables, reference counts, and what to do when the pool runs out'},
  tabs: [
    {id:'frag', label:{zh:'為什麼要分頁', en:'why pages at all'},
     stage:{zh:'有空間但形狀不對', en:'room exists, in the wrong shape'},
     view:VIEW, code:CODE_FRAG, build:fragFrames,
     idea:{zh:'就算事先知道每個請求的最終長度（最有利的假設），連續配置仍然會被外部碎片卡死：實測 ' + FRAG.reqs + ' 個請求下它拒絕了 ' + FRAG.rejC + ' 個，而**全部 ' + FRAG.rejFrag + ' 個都是在「空位夠但不連續」時被拒絕**。固定大小的 block 讓「形狀」這件事消失，同樣流量只拒絕 ' + FRAG.rejP + ' 個、佔用率從 ' + FRAG.utilC.toFixed(1) + '% 升到 ' + FRAG.utilP.toFixed(1) + '%。',
           en:'Even when told each request\'s exact final length - the kindest assumption - a contiguous pool is defeated by external fragmentation: it refused ' + FRAG.rejC + ' of ' + FRAG.reqs + ' requests and all ' + FRAG.rejFrag + ' of those refusals happened with enough free space, just not contiguous. Fixed-size blocks delete the notion of shape: the same traffic refuses ' + FRAG.rejP + ' and occupancy rises from ' + FRAG.utilC.toFixed(1) + '% to ' + FRAG.utilP.toFixed(1) + '%.'},
     legend:[['#ff9736', {zh:'請求 A 的空間', en:'request A\'s memory'}],
             ['#12b3b8', {zh:'請求 B', en:'request B'}],
             ['#9d6bff', {zh:'請求 C', en:'request C'}]]},
    {id:'block', label:{zh:'block 要多大', en:'how big is a block'},
     stage:{zh:'浪費 vs 索引大小', en:'waste against index size'},
     view:VIEW, code:CODE_BLOCK, build:blockFrames,
     idea:{zh:'分頁的代價是每個請求最後一個 block 的尾巴：1/8/16/64/256 個 token 一個 block 時浪費 ' + WASTE.map(r => r.waste.toFixed(1) + '%').join(' / ') + '，而 block table 的項目數從 ' + WASTE[0].tables.toLocaleString() + ' 降到 ' + WASTE[WASTE.length - 1].tables.toLocaleString() + '。16～32 是常見的折衷，而且 block 還是前綴共用能對齊的最小單位。',
           en:'The cost of paging is the tail of each request\'s last block: at 1, 8, 16, 64 and 256 tokens per block the waste is ' + WASTE.map(r => r.waste.toFixed(1) + '%').join(' / ') + ' while block-table entries fall from ' + WASTE[0].tables.toLocaleString() + ' to ' + WASTE[WASTE.length - 1].tables.toLocaleString() + '. 16 to 32 is the usual compromise - and the block is also the granularity at which prefixes can be shared.'},
     legend:[['#ff5c5c', {zh:'浪費掉的比例', en:'share of slots wasted'}],
             ['#12b3b8', {zh:'block table 的項目數', en:'block-table entries'}]]},
    {id:'share', label:{zh:'共用與 copy-on-write', en:'sharing and copy-on-write'},
     stage:{zh:'一個 prompt，四個答案', en:'one prompt, four answers'},
     view:VIEW, code:CODE_SHARE, build:shareFrames,
     idea:{zh:'block table 只是一串編號，所以同一個 prompt 的 ' + SAMP.samples + ' 個樣本可以指向同一批 block（refcount 保活）：' + SAMP.naive + ' → ' + SAMP.shared + ' 個 block，少 ' + SAMP.saved.toFixed(0) + '%。只有 prompt 最後那個沒裝滿的 block 需要在寫入時各自複製一份——copy-on-write，跟 fork() 同一招。',
           en:'A block table is just a list of ids, so ' + SAMP.samples + ' samples of one prompt can point at the same blocks with a reference count: ' + SAMP.naive + ' blocks become ' + SAMP.shared + ', ' + SAMP.saved.toFixed(0) + '% fewer. Only the prompt\'s partly-full last block has to be copied when each sample writes into it - copy-on-write, exactly as fork() does it.'},
     legend:[['#ff9736', {zh:'樣本 0 配置的 block', en:'blocks allocated by sample 0'}],
             ['#12b3b8', {zh:'其他樣本共用的', en:'shared by the other samples'}],
             ['#9d6bff', {zh:'寫入時複製的那一個', en:'the one copied on write'}]]},
    {id:'preempt', label:{zh:'池子滿了', en:'when the pool runs out'},
     stage:{zh:'重算還是搬出去', en:'recompute or move it out'},
     view:VIEW, code:CODE_PREEMPT, build:preFrames,
     idea:{zh:'60 個請求對 4,096 個 slot：發生 ' + PRE.nR + ' 次收回。重算政策要再 prefill ' + PRE.recompute.toLocaleString() + ' 個 token，換出政策要搬 ' + PRE.swap.toLocaleString() + ' 個 token 出去再搬回來，兩者都把 ' + PRE.finished + ' 個請求跑完。算力 vs 頻寬，跟前一天的分層是同一個取捨。',
           en:'60 requests against 4,096 slots cause ' + PRE.nR + ' preemptions. The recompute policy prefills ' + PRE.recompute.toLocaleString() + ' tokens again; the swap policy moves ' + PRE.swap.toLocaleString() + ' tokens out and back. Both finish all ' + PRE.finished + ' requests. Compute against bandwidth - the same trade as the previous day\'s tiers.'},
     legend:[['#ff5c5c', {zh:'重算（花算力）', en:'recompute: pay in compute'}],
             ['#ff9736', {zh:'換出（花頻寬）', en:'swap: pay in bandwidth'}]]},
    {id:'bugs', label:{zh:'兩個不會報錯的壞法', en:'two silent failures'},
     stage:{zh:'漏掉的 block，和沒裝滿的尾巴', en:'leaked blocks, and the partly-full tail'},
     view:VIEW, code:CODE_BUGS, build:bugFrames,
     idea:{zh:'(1) 失敗路徑忘了釋放：漏掉 ' + BUG.leakedTokens.toLocaleString() + ' / ' + BUG.capacity.toLocaleString() + ' 個 slot，伺服器只是越來越收不下請求；抓法是閒置時檢查「空的 ＋ 表裡的 ＝ 池子」。(2) 用補滿後的長度做 attention：' + BUG.tokens + ' 個 token 佔 ' + BUG.padded + ' 個 slot，最後 ' + BUG.garbage + ' 格是別人留下的舊 KV，softmax 照樣加總為 1。',
           en:'(1) A failure path that forgets to free leaks ' + BUG.leakedTokens.toLocaleString() + ' of ' + BUG.capacity.toLocaleString() + ' slots, and the server merely accepts fewer and fewer requests; the test is an idle-time invariant that free blocks plus tabled blocks equal the pool. (2) Attending over the padded length: ' + BUG.tokens + ' tokens occupy ' + BUG.padded + ' slots and the last ' + BUG.garbage + ' hold someone else\'s old KV, while the softmax still sums to one.'},
     legend:[['#12b3b8', {zh:'正常的 slot', en:'healthy slots'}],
             ['#ff5c5c', {zh:'漏掉的／是垃圾的', en:'leaked, or holding garbage'}]]},
    {id:'vllm', label:{zh:'在 vLLM 裡', en:'in vLLM'},
     stage:{zh:'PagedAttention 的出處', en:'where PagedAttention comes from'},
     view:VIEW, code:CODE_VLLM, build:vllmFrames,
     idea:{zh:'vLLM 的旗標跟這個檔案一一對應：`--block-size` 是 block 大小、`--gpu-memory-utilization` 決定池子多大（啟動時 profiling 之後相除）、`--max-num-seqs` 是併發上限、`--swap-space` 是換出用的主機記憶體、`--enable-prefix-caching` 是跨請求共用。`SamplingParams(n=4)` 就是這裡的 refcount＋copy-on-write，而日誌裡的 `preempted by PreemptionMode.RECOMPUTE` 就是第 4 個分頁。',
           en:'vLLM\'s flags map one to one onto this file: `--block-size` is the block size, `--gpu-memory-utilization` sets the pool (divided out after a startup profiling pass), `--max-num-seqs` caps concurrency, `--swap-space` is host memory for preemption, `--enable-prefix-caching` shares across requests. `SamplingParams(n=4)` is the reference count plus copy-on-write here, and the log line `preempted by PreemptionMode.RECOMPUTE` is tab four happening for real.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'vLLM 裡的名字', en:'what vLLM calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]},
    {id:'sglang', label:{zh:'在 SGLang 裡', en:'in SGLang'},
     stage:{zh:'同樣的零件，不同的預設', en:'the same pieces, different defaults'},
     view:VIEW, code:CODE_SGLANG, build:sglangFrames,
     idea:{zh:'SGLang 的 page 預設是 1 個 token（零內部碎片、索引很長，但索引是 GPU 上的 `req_to_token_pool` tensor 不是 Python list）；slot 本身在 `token_to_kv_pool`；池子大小用 `--mem-fraction-static` 或 `--max-total-tokens`；滿了用 **retract**（收回重算）而不是換出，因為前綴多半還在 radix tree 裡，重算只重算沒命中的那一段。',
           en:'SGLang\'s page is one token by default - zero internal fragmentation and a long index, affordable because the index is a `req_to_token_pool` tensor on the GPU rather than a Python list. The slots live in `token_to_kv_pool`, the pool size comes from `--mem-fraction-static` or `--max-total-tokens`, and when it runs out it **retracts** rather than swapping, because the prefix is usually still in the radix tree so recomputing only redoes the part that missed.'},
     legend:[['#ff9736', {zh:'目前這一列', en:'the row being shown'}],
             ['#12b3b8', {zh:'SGLang 裡的名字', en:'what SGLang calls it'}],
             ['#8fa3ac', {zh:'這個檔案裡的名字', en:'what this file calls it'}]]}
  ]
};
