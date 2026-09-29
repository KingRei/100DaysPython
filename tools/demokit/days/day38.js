// DAY: 38
// TITLE_ZH: Day 38 — 貪心演算法：區間排程與 Huffman 編碼
// TITLE_EN: Greedy: interval scheduling and Huffman coding
// SUB_ZH: 貪心法寫起來只有一個 sort 加一次掃描，難的從來不是寫，是證明你挑的排序鍵是對的。同一間會議室的八個申請，按結束時間排能排進 5 場，按開始時間排只剩 3 場，按時長排 4 場——三種都是合法、不重疊的時間表，程式不會報錯，也不會警告，錯的鍵只是默默少排了兩個人。今天用交換論證把「為什麼是結束時間」講清楚，再把同一套推理搬到 Huffman：每次合併兩棵最輕的子樹，而那個「每次」必須含重新排序，少了它，樹會塌成一條梯子。
// SUB_EN: A greedy algorithm is one sort plus one pass; the difficulty was never writing it but proving the sort key is the right one. Eight requests for one meeting room fit five meetings when sorted by finish time, three when sorted by start time, four when sorted by duration - and all three are legal, clash-free schedules, so nothing raises and nothing warns. Today the exchange argument explains why the finish time is the key that matters, and the same reasoning carries into Huffman coding, where "always merge the two lightest subtrees" quietly depends on a re-sort that is very easy to drop.
// FOLDER: day%2038%20-%20greedy%20and%20huffman%20coding
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }

/* text metrics, mirroring the harness model: a CJK glyph is one em wide, a
   Latin one about 0.55.  fitT shrinks a caption until both languages fit the
   view - the English sentence is always the wider of the two. */
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


/* ---------------------------------------------------------------- shared */
/* the eight meeting-room requests from greedy.py; slots of 15 min from 09:00 */
const MEET = [['A', 1, 9], ['B', 6, 7], ['C', 7, 8], ['D', 8, 12],
              ['E', 11, 13], ['F', 12, 18], ['G', 18, 23], ['H', 18, 25]];
const T0 = 9 * 60, UNIT = 15;
const clock = u => { const m = T0 + u * UNIT;
  return ('0' + Math.floor(m / 60)).slice(-2) + ':' + ('0' + (m % 60)).slice(-2); };
const spanStr = iv => clock(iv[1]) + '-' + clock(iv[2]);
const compat = (a, b) => a[2] <= b[1] || b[2] <= a[1];

/* timeline geometry: slot 0 (09:00) .. slot 26 (15:30) */
const X0 = 0.62, SCX = (9.05 - 0.62) / 26;
const sx = u => X0 + u * SCX;

function ruler(y, c){
  const out = [];
  for (let u = 0; u <= 24; u += 4){
    out.push(S.e(sx(u), y + .10, sx(u), y + .30, {s:'soft', arrow:false, w:.030, o:.8}));
    out.push(S.t(sx(u), y, clock(u), {c:c || COL.grey, fs:.24}));
  }
  out.push(S.e(sx(0), y + .30, sx(26), y + .30, {s:'soft', arrow:false, w:.025, o:.55}));
  return out;
}
/* one meeting drawn as a bar on row y */
function bar(m, y, h, st, opt){
  opt = opt || {};
  const x = sx(m[1]), w = sx(m[2]) - sx(m[1]);
  const o = [S.r(x, y, w, h, st, w > .45 ? m[0] : '', {fs:.27, rx:.07})];
  if (w <= .45) o.push(S.t(x + w / 2, y + h + .26, m[0], {c:COL.pale, fs:.24}));
  if (opt.left !== false)
    o.push(S.t(.52, y + h * .72, m[0], {c:opt.lc || COL.grey, fs:.28, anchor:'end'}));
  if (opt.span) o.push(S.t(sx(26) + .55, y + h * .72, spanStr(m), {c:COL.grey, fs:.22, anchor:'end'}));
  return o;
}

/* =======================================================================
 * Tab 1 - three sort keys, three legal schedules, three different sizes
 * ======================================================================= */
const CODE_SCHED = [
  'def select(intervals, key):               # greedy: one sort, one pass',
  '    out = []',
  '    for s, e in sorted(intervals, key=key):',
  '        if all(compatible((s, e), c) for c in out):',
  '            out.append((s, e))            # committed - never reconsidered',
  '    return out',
  '',
  '# sorted by end time the test collapses to  s >= last_end,',
  '# because nothing taken so far can finish later than the last one',
  '',
  'select(IV, key=lambda iv: iv[1])                 # by end      -> B C D F G  (5)',
  'select(IV, key=lambda iv: iv[0])                 # by start    -> A E G      (3)',
  'select(IV, key=lambda iv: (iv[1]-iv[0], iv[1]))  # by duration -> B C E G    (4)'
];

const KEYS = [
  {id:0, sortKey:(a, b) => a[2] - b[2],
   zh:'按結束時間排', en:'sorted by end time', line:10},
  {id:1, sortKey:(a, b) => a[1] - b[1],
   zh:'按開始時間排', en:'sorted by start time', line:11},
  {id:2, sortKey:(a, b) => (a[2] - a[1]) - (b[2] - b[1]) || a[2] - b[2],
   zh:'按時長排（短的先）', en:'sorted by duration, shortest first', line:12}
];
/* run the greedy for real and record what it did, step by step */
function runSelect(k){
  const order = MEET.slice().sort(KEYS[k].sortKey), out = [], log = [];
  order.forEach(m => {
    const clash = out.filter(c => !compat(m, c));
    if (!clash.length){ out.push(m); log.push({m:m, ok:true, clash:null}); }
    else log.push({m:m, ok:false, clash:clash[clash.length - 1]});
  });
  return {order:order, out:out, log:log};
}
const RES = [runSelect(0), runSelect(1), runSelect(2)];

const ROWY = 1.42, ROWH = .44, BARH = .32, ROOMY = 5.28;
function schedFrames(v){
  const F = new Frames(), R = RES[v], K = KEYS[v];
  const best = RES[0].out.length;
  /* row index of each meeting in this variant's sort order */
  const rowOf = {};
  R.order.forEach((m, i) => { rowOf[m[0]] = i; });
  const rowY = m => ROWY + rowOf[m[0]] * ROWH;

  function stage(upto, curName, states, extra){
    const sh = [fitT(4.9, .48, {zh:K.zh + '：貪心一路往下，只看「和已排的撞不撞」',
                               en:K.en + ' - one pass, the only test is "does it clash"'},
                    {c:COL.tealL, fs:.33})].concat(ruler(1.00));
    R.order.forEach(m => {
      sh.push.apply(sh, bar(m, rowY(m), BARH, states[m[0]] || 'idle',
                            {lc:m[0] === curName ? COL.orangeL : COL.grey}));
    });
    /* the room itself: what has actually been booked so far */
    sh.push(S.t(sx(0), ROOMY - .14, {zh:'會議室', en:'room'}, {c:COL.tealL, fs:.25, anchor:'start'}));
    sh.push(S.r(sx(0), ROOMY, sx(26) - sx(0), BARH, 'ghost', '', {rx:.07}));
    R.out.slice(0, upto).forEach(m => sh.push(S.r(sx(m[1]), ROOMY, sx(m[2]) - sx(m[1]),
                                                  BARH, 'ok', m[0], {fs:.26, rx:.07})));
    if (extra) sh.push.apply(sh, extra);
    return sh;
  }

  const states = {};
  F.push({shapes:stage(0, null, states, [
      fitT(4.9, 6.02, {zh:'八個人搶同一間會議室，排程只能挑不重疊的一組',
                      en:'eight requests, one room - the schedule must be clash-free'},
          {c:COL.pale, fs:.30})]),
    panels:[{lbl:{zh:'排序鍵', en:'sort key'},
             chips:[chip(v === 0 ? 'iv[1]  (end)' : v === 1 ? 'iv[0]  (start)' : 'iv[1]-iv[0]  (len)',
                         v === 0 ? 'ok' : 'bad')]},
            {lbl:{zh:'已排', en:'taken'}, chips:[]}],
    line:2,
    msg:v === 0
      ? {zh:'目標是把最多人排進同一間會議室，不是排最久的那個人。貪心的整份程式就是一個 sort 加一次掃描，所以全部的差別都落在那個排序鍵上——這裡先用「誰先把房間還出來」當鍵，因為房間一還出來，剩下的時間就完全乾淨，後面能塞什麼完全不受前面影響。',
         en:'The goal is to seat the most people in one room, not the busiest person. The entire greedy is a sort plus one pass, so every decision lives in the sort key. This run uses "who frees the room soonest", because the moment the room is free the remaining time is completely clean and what fits later does not depend on what came before.'}
      : v === 1
      ? {zh:'換一個聽起來同樣合理的鍵：誰先提申請誰先排。真實世界的排隊就是這樣運作的，而且它跑出來的時間表一定合法。問題在於「先開始」跟「先結束」是兩回事——一個九點多開始、開到十一點多的會，佔掉的是房間，不是禮貌。',
         en:'Here is a key that sounds every bit as reasonable: first come, first served. That is how a real queue works, and the schedule it produces is certainly legal. The trap is that starting early and finishing early are different things - a meeting that opens at 09:15 and runs to 11:15 occupies the room, not the moral high ground.'}
      : {zh:'第三個鍵：短的會先排，直覺是「短的比較省房間」。這個鍵對「總時數」來說確實省，但今天要最大化的是場次數，而一場十五分鐘的會如果卡在別人中間，省下來的時間根本沒有人用得到。',
         en:'A third key: shortest meetings first, on the instinct that short ones are cheap. That instinct optimises total minutes occupied, but the quantity being maximised today is the number of meetings - and fifteen minutes saved in the middle of someone else’s slot is fifteen minutes nobody can use.'}});

  let taken = 0;
  R.log.forEach((e, i) => {
    const m = e.m;
    states[m[0]] = 'hot';
    const lastEnd = (v < 2 && taken) ? R.out[taken - 1][2] : null;
    const guide = lastEnd == null ? [] :
      [S.e(sx(lastEnd), ROWY - .22, sx(lastEnd), ROOMY + BARH + .18,
           {s:'act', arrow:false, w:.045, dash:'.10 .10', o:.9}),
       S.t(sx(lastEnd), ROWY - .34, 'last_end ' + clock(lastEnd), {c:COL.purpleL, fs:.24})];
    F.push({shapes:stage(taken, m[0], states, guide.concat([
        fitT(4.9, 6.02, e.ok
            ? {zh:m[0] + ' ' + spanStr(m) + ' 排得進去，房間佔到 ' + clock(m[2]),
               en:m[0] + ' ' + spanStr(m) + ' fits; the room is now busy until ' + clock(m[2])}
            : {zh:m[0] + ' ' + spanStr(m) + ' 和已排的 ' + e.clash[0] + ' 撞了，跳過',
               en:m[0] + ' ' + spanStr(m) + ' clashes with ' + e.clash[0] + ' - skipped'},
            {c:e.ok ? COL.tealL : COL.red, fs:.30})])),
      panels:[{lbl:{zh:'目前考慮', en:'considering'}, chips:[chip(m[0] + ' ' + spanStr(m), 'hot')]},
              {lbl:{zh:'已排', en:'taken'},
               chips:R.out.slice(0, taken).map(x => chip(x[0], 'ok'))},
              {lbl:{zh:'last_end', en:'last_end'},
               chips:[chip(lastEnd == null ? '—' : clock(lastEnd), lastEnd == null ? 'empty' : 'dim')]}],
      line:3,
      msg:e.ok
        ? (taken === 0
          ? {zh:'第一場直接收下，沒有東西可以撞。值得停下來想的是：這一步就是整個證明的核心——' +
                (v === 0 ? '在所有申請裡 ' + m[0] + ' 最早把房間還出來，所以它留給後面的時間最長。'
                         : '這個鍵選中的是 ' + m[0] + '，它把房間佔到 ' + clock(m[2]) +
                           '，後面所有人只能從那之後開始。'),
             en:'The first one is taken unconditionally - there is nothing for it to clash with. It is worth pausing here, because this single step is what the proof is about: ' +
                (v === 0 ? m[0] + ' frees the room earlier than any other request, so it leaves the longest clean stretch behind it.'
                         : 'this key picks ' + m[0] + ', which holds the room until ' + clock(m[2]) + ', and everyone else must now start after that.')}
          : (v < 2
            ? {zh:m[0] + ' 在 ' + clock(m[1]) + ' 開始，不早於 last_end ' + clock(lastEnd) +
                  '，所以和前面每一場都不衝突——只比一個數字就夠了，因為已排的場次彼此不重疊又依序往後，' +
                  '結束最晚的必定是最後收下的那一場。',
               en:m[0] + ' starts at ' + clock(m[1]) + ', not before last_end ' + clock(lastEnd) +
                  ', so it clashes with none of the earlier picks. One number is enough to check: the taken meetings are disjoint and run forwards, so the latest finisher among them is always the most recent pick.'}
            : {zh:m[0] + ' ' + spanStr(m) + ' 和已排的每一場都不衝突，收下。按時長排的時候不能只看一個 last_end——' +
                  '已排的場次在時間軸上是散的，新的一場可能落在兩場中間，所以每次都要和全部已排的比對過。',
               en:m[0] + ' ' + spanStr(m) + ' clashes with nothing taken so far, so it is kept. Sorting by duration forbids the single last_end shortcut: the meetings taken so far are scattered along the day and a new one may land in a gap between two of them, so every pick must be checked against all of them.'}))
        : {zh:m[0] + ' ' + spanStr(m) + ' 蓋到了 ' + e.clash[0] + ' ' + spanStr(e.clash) +
              '，只能跳過。注意這裡沒有回頭路：貪心不會為了塞進 ' + m[0] +
              ' 而把 ' + e.clash[0] + ' 吐出來重排，這正是它快、也正是它需要被證明的原因。',
           en:m[0] + ' ' + spanStr(m) + ' overlaps ' + e.clash[0] + ' ' + spanStr(e.clash) +
              ', so it is skipped. Note there is no going back: the greedy will never give ' +
              e.clash[0] + ' up in order to fit ' + m[0] + ' in. That is why it is fast, and exactly why it needs a proof.'}});
    states[m[0]] = e.ok ? 'ok' : 'done';
    if (e.ok) taken++;
  });

  const namesOf = a => a.map(x => x[0]).join(' ');
  F.push({shapes:stage(R.out.length, null, states, [
      fitT(4.9, 5.92, {zh:'這一版排進 ' + R.out.length + ' 場：' + namesOf(R.out),
                      en:R.out.length + ' meetings: ' + namesOf(R.out)},
          {c:v === 0 ? COL.tealL : COL.red, fs:.34}),
      fitT(4.9, 6.24, v === 0
          ? {zh:'2⁸ 個子集合窮舉出來的最佳解也是 5 場，一場不多',
             en:'brute force over all 2^8 subsets also says 5 - not one more'}
          : {zh:'窮舉最多能排 ' + best + ' 場；這張表完全合法，只是少了 ' + (best - R.out.length) + ' 場',
             en:'brute force fits ' + best + '; this table is perfectly legal, it just seats ' +
                (best - R.out.length) + ' fewer'},
          {c:COL.grey, fs:.28})]),
    panels:[{lbl:{zh:'結果', en:'result'},
             chips:R.out.map(x => chip(x[0], v === 0 ? 'ok' : 'bad'))},
            {lbl:{zh:'場次', en:'count'},
             chips:[chip(R.out.length + ' / ' + best, v === 0 ? 'ok' : 'bad')]}],
    line:K.line,
    msg:v === 0
      ? {zh:'5 場，和窮舉 2⁸ 個子集合的答案一樣。這個鍵之所以對，一句話就講完：每次都挑最早把房間還出來的那一場，剩下能用的時間就是所有選擇裡最長的一段，而後面能排幾場只跟那段長度有關。下一個分頁把這句話變成可以驗證的實驗。',
         en:'Five meetings, matching the answer from brute-forcing all 2^8 subsets. Why this key works fits in one sentence: taking the request that frees the room earliest leaves the longest possible clean stretch, and how many meetings fit later depends on nothing but the length of that stretch. The next tab turns that sentence into an experiment.'}
      : {zh:'這張時間表完全合法：沒有任何兩場會重疊，程式沒有報錯，也沒有任何警告。它只是比最佳解少 ' +
            (best - R.out.length) + ' 場。這就是貪心法最危險的地方——錯的排序鍵不會壞掉，只會默默給你一個比較差的答案，' +
            '而「跑出來有結果」從來不是正確性的證據。',
         en:'This table is entirely legal: no two meetings overlap, nothing raised, nothing warned. It simply seats ' +
            (best - R.out.length) + ' fewer people than the optimum. That is what makes greedy dangerous - a wrong sort key does not break, it quietly returns a worse answer, and "it produced a result" is never evidence of correctness.'}});
  return F.list;
}

/* =======================================================================
 * Tab 2 - the exchange argument, run as an experiment
 * ======================================================================= */
const CODE_EX = [
  'def select_bruteforce(intervals):        # ground truth: try every subset',
  '    best = []',
  '    for mask in range(1 << len(intervals)):',
  '        pick = [intervals[i] for i in range(len(intervals)) if mask >> i & 1]',
  '        if len(pick) > len(best) and is_feasible(pick):',
  '            best = pick',
  '    return best',
  '',
  'first = min(intervals, key=lambda iv: iv[1])     # the earliest finisher',
  'unrestricted = len(select_bruteforce(intervals))',
  'rest   = [iv for iv in intervals if compatible(iv, first) and iv != first]',
  'forced = 1 + len(select_bruteforce(rest))        # forced to contain `first`',
  'assert unrestricted == forced                    # so taking it costs nothing'
];

function feasible(pick){
  for (let i = 0; i < pick.length; i++)
    for (let j = i + 1; j < pick.length; j++) if (!compat(pick[i], pick[j])) return false;
  return true;
}
/* brute force, recording every time the best-so-far grows */
function brute(list){
  let best = [];
  const log = [];
  for (let mask = 0; mask < (1 << list.length); mask++){
    const pick = [];
    for (let i = 0; i < list.length; i++) if (mask >> i & 1) pick.push(list[i]);
    if (pick.length > best.length && feasible(pick)){ best = pick; log.push({mask:mask, pick:pick.slice()}); }
  }
  return {best:best, log:log};
}
const FIRST = MEET.slice().sort((a, b) => a[2] - b[2])[0];          // B
const REST = MEET.filter(m => m !== FIRST && compat(m, FIRST));
const BRUTE_ALL = brute(MEET), BRUTE_REST = brute(REST);
const UNRESTRICTED = BRUTE_ALL.best.length;
const FORCED = 1 + BRUTE_REST.best.length;

function exFrames(){
  const F = new Frames();
  const rowY = i => ROWY + i * ROWH;
  function stage(states, cap, capc, cap2){
    const sh = [fitT(4.9, .48, {zh:'交換論證：最早結束的那一場，留著它會不會虧？',
                               en:'the exchange argument: does keeping the earliest finisher cost anything?'},
                    {c:COL.tealL, fs:.33})].concat(ruler(1.00));
    MEET.forEach((m, i) => sh.push.apply(sh, bar(m, rowY(i), BARH, states[m[0]] || 'soft',
                                                 {lc:states[m[0]] === 'hot' ? COL.orangeL : COL.grey})));
    if (cap) sh.push(fitT(4.9, 5.60, cap, {c:capc || COL.pale, fs:.31}));
    if (cap2) sh.push(fitT(4.9, 6.06, cap2, {c:COL.grey, fs:.27}));
    return sh;
  }
  const names = a => a.map(x => x[0]).join(' ');

  F.push({shapes:stage({B:'hot'},
      {zh:'B 10:30-10:45 是八個申請裡最早結束的一場',
       en:'B 10:30-10:45 finishes earlier than any other request'}, COL.orangeL,
      {zh:'問題不是「B 一定在最佳解裡嗎」，是「有沒有一個最佳解含 B」',
       en:'the claim is not that every optimum contains B, only that some optimum does'}),
    panels:[{lbl:{zh:'最早結束', en:'earliest finisher'}, chips:[chip('B ' + spanStr(FIRST), 'hot')]}],
    line:8,
    msg:{zh:'貪心法唯一需要證明的東西，是第一步。如果「先收下最早結束的那一場」不會讓答案變差，那剩下的問題和原問題長得一模一樣，同一句話可以再套一次，一路套到底。所以只要證一件事：存在一個最佳解含有 B。',
         en:'The only thing a greedy algorithm ever needs proving is its first step. If committing to the request that finishes earliest cannot make the answer worse, then what remains is the same problem in miniature and the same sentence applies again, all the way down. So there is exactly one claim to establish: some optimal schedule contains B.'}});

  BRUTE_ALL.log.forEach((e, i) => {
    const st = {}; e.pick.forEach(m => { st[m[0]] = 'ok'; });
    const last = i === BRUTE_ALL.log.length - 1;
    F.push({shapes:stage(st,
        {zh:'窮舉到目前為止最好的一組：' + e.pick.length + ' 場（' + names(e.pick) + '）',
         en:'best clash-free subset so far: ' + e.pick.length + ' (' + names(e.pick) + ')'},
        last ? COL.tealL : COL.pale,
        {zh:'mask = ' + ('00000000' + e.mask.toString(2)).slice(-8) + '，256 個子集合逐一檢查',
         en:'mask = ' + ('00000000' + e.mask.toString(2)).slice(-8) + ', all 256 subsets are checked'}),
      panels:[{lbl:{zh:'目前最佳', en:'best so far'}, chips:e.pick.map(m => chip(m[0], 'ok'))},
              {lbl:{zh:'大小', en:'size'}, chips:[chip(String(e.pick.length), last ? 'ok' : 'dim')]}],
      line:5,
      msg:last
        ? {zh:'窮舉停在 ' + UNRESTRICTED + ' 場。這是真正的上界，因為 256 個子集合一個都沒漏。' +
              '注意它挑出來的剛好含 B——但這還不算證明，只是這筆測資碰巧如此；真正要比的是下面這組數字。',
           en:'Brute force settles at ' + UNRESTRICTED + '. This is a genuine upper bound because not one of the 256 subsets was skipped. It happens to contain B, but that is an observation about this instance, not a proof; the number that matters is the one computed next.'}
        : {zh:'找到一組 ' + e.pick.length + ' 場不衝突的（' + names(e.pick) +
              '）。窮舉不聰明，它只是把每個子集合都攤開來檢查有沒有互相重疊，慢但不會錯——貪心的答案好不好，要跟它比才算數。',
           en:'A clash-free subset of size ' + e.pick.length + ' (' + names(e.pick) +
              '). Brute force is not clever; it lays out every subset and checks whether any two overlap. Slow, but never wrong - and it is the only thing worth measuring the greedy against.'}});
  });

  const stRest = {B:'ok'};
  MEET.forEach(m => { if (m !== FIRST && !compat(m, FIRST)) stRest[m[0]] = 'bad'; });
  const killed = MEET.filter(m => m !== FIRST && !compat(m, FIRST));
  F.push({shapes:stage(stRest,
      {zh:'強制收下 B：和它重疊的 ' + names(killed) + ' 就沒得排了',
       en:'force B into the schedule: ' + names(killed) + ' overlaps it and is out'}, COL.red,
      {zh:'B 只佔 15 分鐘，擋掉的東西是所有可能選法裡最少的',
       en:'B occupies 15 minutes - it blocks fewer requests than any other choice could'}),
    panels:[{lbl:{zh:'強制選入', en:'forced in'}, chips:[chip('B', 'ok')]},
            {lbl:{zh:'被它擋掉', en:'blocked by B'}, chips:killed.map(m => chip(m[0], 'bad'))}],
    line:10,
    msg:{zh:'把 B 釘在時間表裡，再看代價。和 B 重疊的只有 ' + names(killed) +
            ' 一場，因為 B 最早結束，它往右擋不到任何東西；換成任何別的申請當第一場，被擋掉的都只會更多，不會更少。這句話就是交換論證的全部內容。',
         en:'Pin B into the schedule and count the damage. Only ' + names(killed) +
            ' overlaps it, because B finishes before anything else does and therefore blocks nothing to its right. Choosing any other request as the first pick can only block more, never fewer - and that sentence is the whole exchange argument.'}});

  BRUTE_REST.log.forEach((e, i) => {
    const st = {B:'ok'};
    MEET.forEach(m => { if (m !== FIRST && !compat(m, FIRST)) st[m[0]] = 'bad'; });
    e.pick.forEach(m => { st[m[0]] = 'ok'; });
    const last = i === BRUTE_REST.log.length - 1;
    if (!last && e.pick.length < BRUTE_REST.best.length - 1) return;   // keep it short
    F.push({shapes:stage(st,
        {zh:'B 之外再窮舉剩下的：' + e.pick.length + ' 場（' + names(e.pick) + '）',
         en:'brute force over what is left of B: ' + e.pick.length + ' (' + names(e.pick) + ')'},
        last ? COL.tealL : COL.pale,
        {zh:'合起來 1 + ' + e.pick.length + ' = ' + (1 + e.pick.length) + ' 場',
         en:'together 1 + ' + e.pick.length + ' = ' + (1 + e.pick.length)}),
      panels:[{lbl:{zh:'含 B 的最佳', en:'best containing B'},
               chips:[chip('B', 'ok')].concat(e.pick.map(m => chip(m[0], 'ok')))},
              {lbl:{zh:'大小', en:'size'}, chips:[chip(String(1 + e.pick.length), last ? 'ok' : 'dim')]}],
      line:11,
      msg:last
        ? {zh:'含 B 的最佳解是 ' + FORCED + ' 場。',
           en:'The best schedule that contains B has ' + FORCED + ' meetings.'}
        : {zh:'把 B 和被它擋掉的拿掉之後，剩下的是一個更小的同類問題，再窮舉一次。目前 ' +
              e.pick.length + ' 場。',
           en:'With B and everything it blocks removed, what is left is a smaller copy of the same problem, brute-forced again. Currently ' + e.pick.length + '.'}});
  });

  F.push({shapes:stage((function(){ const s = {}; BRUTE_REST.best.concat([FIRST]).forEach(m => { s[m[0]] = 'ok'; }); return s; })(),
      {zh:'不限制：' + UNRESTRICTED + ' 場　　強制含 B：' + FORCED + ' 場',
       en:'unrestricted: ' + UNRESTRICTED + '     forced to contain B: ' + FORCED}, COL.tealL,
      {zh:'兩個數字一樣，所以收下 B 沒有虧到任何東西',
       en:'the two are equal, so committing to B gives nothing away'}),
    panels:[{lbl:{zh:'不限制', en:'unrestricted'}, chips:[chip(String(UNRESTRICTED), 'ok')]},
            {lbl:{zh:'強制含 B', en:'forced'}, chips:[chip(String(FORCED), 'ok')]},
            {lbl:{zh:'結論', en:'verdict'},
             chips:[chip(UNRESTRICTED === FORCED ? 'greedy is safe' : 'greedy loses',
                         UNRESTRICTED === FORCED ? 'ok' : 'bad')]}],
    line:12,
    msg:{zh:UNRESTRICTED + ' = ' + FORCED + '：不管最佳解原本長什麼樣，都可以把它的第一場換成 B 而不變短——' +
            '因為 B 結束得最早，換進去之後後面那些場次一場都不會被擠掉。這就是貪心可以放心往前衝的授權書。' +
            '而同一份授權書在「每場會有不同價值、要賺最多錢」的版本上就發不出來了：那時候 B 值 1 塊、A 值 8 塊，' +
            '換過去會虧，只好回去寫 DP。',
         en:UNRESTRICTED + ' = ' + FORCED + '. Whatever an optimal schedule looks like, its first meeting can be swapped for B without shortening it, because B finishes earliest and the swap cannot push any later meeting out. That is the licence the greedy runs on. The same licence cannot be issued for the weighted version, where meetings are worth money: there B is worth 1 and A is worth 8, the swap loses value, and the problem goes back to a DP table.'}});
  return F.list;
}

/* =======================================================================
 * Huffman - shared machinery for tab 3 and tab 4
 * ======================================================================= */
/* a 60-symbol message over a 6-letter alphabet; greedy.py runs the same code
   on a 53-character sentence, which is far too wide to draw a tree for */
const FREQ = [['E', 14], ['A', 13], ['T', 10], ['O', 9], ['N', 8], ['S', 6]];
const NSYM = FREQ.length;
const TOTAL = FREQ.reduce((a, f) => a + f[1], 0);
const FIXW = Math.max(1, (NSYM - 1).toString(2).length);      // ceil(log2(alphabet))
const FIXBITS = FIXW * TOTAL;
const wOf = {}; FREQ.forEach(f => { wOf[f[0]] = f[1]; });

const isLeaf = n => n.sym != null;
const symsOf = n => isLeaf(n) ? n.sym : symsOf(n.l) + symsOf(n.r);
function height(n){ return isLeaf(n) ? 0 : 1 + Math.max(height(n.l), height(n.r)); }
function codesOf(root){
  const out = {};
  (function walk(n, p){
    if (isLeaf(n)){ out[n.sym] = p || '0'; return; }
    walk(n.l, p + '0'); walk(n.r, p + '1');
  })(root, '');
  return out;
}
const leafOrder = root => { const o = [];
  (function w(n){ if (isLeaf(n)){ o.push(n.sym); return; } w(n.l); w(n.r); })(root); return o; };
const bitsOf = codes => FREQ.reduce((a, f) => a + f[1] * codes[f[0]].length, 0);
const prefixFree = codes => {
  const w = Object.keys(codes).map(k => codes[k]).sort();
  return w.every((c, i) => i === 0 || !c.startsWith(w[i - 1]));
};
const kraft = codes => Object.keys(codes).reduce((a, k) => a + Math.pow(2, -codes[k].length), 0);
function decode(bits, codes){
  const back = {}; Object.keys(codes).forEach(k => { back[codes[k]] = k; });
  let cur = '', out = '';
  for (const b of bits){ cur += b; if (back[cur] != null){ out += back[cur]; cur = ''; } }
  return cur === '' ? out : null;
}

/* the real thing: a heap, re-sorted after every push */
function huffmanRun(){
  let heap = FREQ.slice().sort((a, b) => a[0] < b[0] ? -1 : 1)
                 .map((f, i) => ({w:f[1], tie:i, n:{sym:f[0], w:f[1]}}));
  let tie = heap.length;
  const steps = [];
  const srt = h => h.slice().sort((a, b) => a.w - b.w || a.tie - b.tie);
  heap = srt(heap);
  while (heap.length > 1){
    const before = heap.slice();
    const a = heap.shift(), b = heap.shift();
    const node = {l:a.n, r:b.n, w:a.w + b.w};
    const entry = {w:node.w, tie:tie++, n:node};
    const rest = heap.slice();
    heap = srt(rest.concat([entry]));
    steps.push({before:before, a:a, b:b, entry:entry, rest:rest,
                after:heap.slice(), slot:heap.indexOf(entry)});
  }
  return {root:heap[0].n, steps:steps};
}
/* the bug: sort once, then merge left to right and never look at the queue again */
function ladderRun(){
  const q = FREQ.slice().sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1))
                .map(f => ({w:f[1], n:{sym:f[0], w:f[1]}}));
  let cur = q[0];
  const steps = [];
  for (let i = 1; i < q.length; i++){
    const node = {l:cur.n, r:q[i].n, w:cur.w + q[i].w};
    const merged = {w:node.w, n:node};
    /* how many still-unmerged items the merged weight should have sunk past */
    let past = 0;
    for (let j = i + 1; j < q.length; j++) if (q[j].w < merged.w) past++;
    steps.push({a:cur, b:q[i], merged:merged, i:i, past:past, queue:q});
    cur = merged;
  }
  return {root:cur.n, steps:steps, queue:q};
}
const HUF = huffmanRun(), LAD = ladderRun();
const HCODES = codesOf(HUF.root), LCODES = codesOf(LAD.root);
const HBITS = bitsOf(HCODES), LBITS = bitsOf(LCODES);
const MSG = (function(){ let s = ''; FREQ.forEach(f => { for (let i = 0; i < f[1]; i++) s += f[0]; });
  return s; })();

/* ------------------------------------------------------------ tree layout */
function treeShapes(roots, slotOf, opt){
  opt = opt || {};
  const out = [], baseY = opt.baseY, RH = opt.rh, R = opt.r || .30;
  const X = n => isLeaf(n) ? slotOf(n.sym) : (X(n.l) + X(n.r)) / 2;
  const Y = n => baseY - height(n) * RH;
  roots.forEach(root => {
    (function draw(n){
      if (!isLeaf(n)){
        [['l', '0'], ['r', '1']].forEach(([k, bit]) => {
          const c = n[k];
          out.push(S.e(X(n), Y(n), X(c), Y(c),
                       {s:(opt.states && opt.states[symsOf(n)]) || 'idle', arrow:false,
                        pad:R * .74, w:.045, lab:bit, fs:opt.bitfs || .23,
                        lx:bit === '0' ? -.17 : .17, ly:.02}));
          draw(c);
        });
      }
      const st = (opt.states && opt.states[symsOf(n)]) || (isLeaf(n) ? 'idle' : 'soft');
      if (isLeaf(n))
        out.push(S.c(X(n), Y(n), R, st, n.sym, {fs:R * .95, sub:String(n.w), subfs:opt.subfs || .24}));
      else
        out.push(S.c(X(n), Y(n), R * .82, st, String(n.w), {fs:R * .80}));
    })(root);
  });
  return out;
}

/* =======================================================================
 * Tab 3 - Huffman: merge the two rarest subtrees, over and over
 * ======================================================================= */
const CODE_HUF = [
  'heap = [(w, i, sym) for i, (sym, w) in enumerate(sorted(freqs.items()))]',
  'heapq.heapify(heap)                      # ordered by weight, ties by i',
  'merges = []',
  'while len(heap) > 1:',
  '    w1, _, n1 = heapq.heappop(heap)      # the rarest subtree',
  '    w2, _, n2 = heapq.heappop(heap)      # the second rarest',
  '    merges.append((n1, n2, w1 + w2))     # they become siblings, one level deeper',
  '    heapq.heappush(heap, (w1 + w2, tie, (n1, n2)))   # sinks back in by weight',
  '    tie += 1',
  '',
  'root = heap[0][2]                        # one tree left',
  'codes = {}; walk(root, "")               # left edge 0, right edge 1'
];

const SLOTW = 8.8 / NSYM;
const slotX = order => sym => .60 + (order.indexOf(sym) + .5) * SLOTW;
const QY = .92, QH = .54, QPITCH = 1.45, QW = 1.34, QX0 = .55;
const HORDER = leafOrder(HUF.root);

function queueShapes(entries, states){
  const out = [];
  entries.forEach((e, i) => {
    const lab = symsOf(e.n) + ' ' + e.w;
    const fs = Math.min(.28, .28 * (QW - .14) / Math.max(.01, tw(lab, .28)));
    out.push(S.r(QX0 + i * QPITCH, QY, QW, QH, states[i] || 'idle', lab, {fs:fs, rx:.08}));
  });
  return out;
}

function hufFrames(){
  const F = new Frames();
  const sx3 = slotX(HORDER);
  const baseY = 5.15, RH = .70;
  const roots = HUF.steps[0].before.map(e => e.n);
  const alive = roots.slice();
  const head = (t, c) => fitT(4.9, .46, t, {c:c || COL.tealL, fs:.33});

  F.push({shapes:[head({zh:'6 種符號、共 60 個字元：固定長度要 ' + FIXW + ' bits × ' + TOTAL + ' = ' + FIXBITS + ' bits',
                        en:NSYM + ' symbols, ' + TOTAL + ' characters: a fixed-length code costs ' +
                           FIXW + ' x ' + TOTAL + ' = ' + FIXBITS + ' bits'})]
        .concat(queueShapes(HUF.steps[0].before, {}))
        .concat(treeShapes(alive, sx3, {baseY:baseY, rh:RH}))
        .concat([fitT(4.9, 6.02, {zh:'常用符號要短碼：多一個 bit，都要乘上它出現的次數',
                                 en:'an extra bit costs one bit per occurrence of that symbol'},
                     {c:COL.pale, fs:.29})]),
    panels:[{lbl:{zh:'次數', en:'frequencies'}, chips:FREQ.map(f => chip(f[0] + ' ×' + f[1], 'dim'))},
            {lbl:{zh:'固定長度', en:'fixed-length'}, chips:[chip(FIXBITS + ' bits', 'bad')]}],
    line:0,
    msg:{zh:'六種符號用固定長度要 ' + FIXW + ' bits 一個，整段 ' + FIXBITS +
            ' bits。可是 E 出現 14 次、S 只有 6 次，給它們一樣長的碼等於把預算平均分給用不到的人。Huffman 的想法是把每個符號放在一棵二元樹的葉子上，碼長就是深度，於是問題變成：怎麼擺葉子，才能讓「深度 × 次數」的總和最小。',
         en:'Six symbols at a fixed width cost ' + FIXW + ' bits each, ' + FIXBITS +
            ' bits in total. But E appears 14 times and S only 6, so giving them equal-length codewords spends the budget on symbols that barely show up. Huffman puts every symbol on the leaf of a binary tree, where the codeword length is just the depth - so the question becomes how to place the leaves to minimise the sum of depth times frequency.'}});

  let bits = 0;
  HUF.steps.forEach((st, k) => {
    const qs = {}; qs[0] = 'hot'; qs[1] = 'hot';
    const tSt = {}; tSt[symsOf(st.a.n)] = 'hot'; tSt[symsOf(st.b.n)] = 'hot';
    F.push({shapes:[head({zh:'第 ' + (k + 1) + ' 次合併：拿走 queue 最前面兩個，' +
                              symsOf(st.a.n) + ' ' + st.a.w + ' 和 ' + symsOf(st.b.n) + ' ' + st.b.w,
                          en:'merge ' + (k + 1) + ': pop the two lightest, ' +
                             symsOf(st.a.n) + ' ' + st.a.w + ' and ' + symsOf(st.b.n) + ' ' + st.b.w},
                         COL.orangeL)]
          .concat(queueShapes(st.before, qs))
          .concat(treeShapes(alive, sx3, {baseY:baseY, rh:RH, states:tSt}))
          .concat([fitT(4.9, 6.02, {zh:'這兩棵之後每加一層，就要付 ' + (st.a.w + st.b.w) + ' 個 bit',
                                   en:'every extra level under these two costs ' + (st.a.w + st.b.w) + ' bits'},
                       {c:COL.orangeL, fs:.29})]),
      panels:[{lbl:{zh:'queue', en:'queue'},
               chips:st.before.map((e, i) => chip(symsOf(e.n) + ' ' + e.w, i < 2 ? 'hot' : 'dim'))},
              {lbl:{zh:'目前總 bits', en:'bits so far'}, chips:[chip(String(bits), 'dim')]}],
      line:5,
      msg:k === 0
        ? {zh:'先拿最輕的兩個：' + symsOf(st.a.n) + '(' + st.a.w + ') 和 ' + symsOf(st.b.n) + '(' + st.b.w +
              ')。為什麼一定是最輕的兩個？因為在最佳的樹裡，最深的那一層一定被出現次數最少的兩個符號佔著——' +
              '如果不是，把深處的常用符號和淺處的罕用符號對調，總 bits 一定變小。既然它們注定是兄弟，現在就把它們配成一對，不會有損失。',
           en:'Take the two lightest first: ' + symsOf(st.a.n) + '(' + st.a.w + ') and ' +
              symsOf(st.b.n) + '(' + st.b.w + '). Why must it be the two lightest? Because in an optimal tree the deepest level is occupied by the two rarest symbols - if it were not, swapping a frequent deep symbol with a rare shallow one would strictly lower the total. They are destined to be siblings, so pairing them now gives nothing away.'}
        : {zh:'同一句話再套一次，只是 queue 裡現在混著單一符號和已經合併的子樹。合併過的節點不再管裡面長什麼樣，只剩一個重量 ' +
              st.a.w + '——這就是為什麼這個演算法可以一路貪下去：子樹一旦成形，它對外只是一個數字。',
           en:'The same sentence applies again, except the queue now mixes single symbols with already-merged subtrees. A merged node forgets its own shape and is nothing but a weight, ' +
              st.a.w + ' - which is exactly why the greedy can keep going: once a subtree is formed it behaves like a single number.'}});

    /* apply the merge */
    const ai = alive.indexOf(st.a.n), bi = alive.indexOf(st.b.n);
    alive.splice(Math.max(ai, bi), 1); alive.splice(Math.min(ai, bi), 1);
    alive.push(st.entry.n);
    bits += st.entry.w;
    const qs2 = {}; qs2[st.slot] = 'act';
    const tS2 = {}; tS2[symsOf(st.entry.n)] = 'act';
    const lighter = st.after.slice(0, st.slot).map(e => symsOf(e.n) + ' ' + e.w).join('、') || '—';
    F.push({shapes:[head({zh:'合併成 ' + st.entry.w + '，放回 queue 第 ' + (st.slot + 1) + ' 位',
                          en:'merged into ' + st.entry.w + ', pushed back at position ' + (st.slot + 1)},
                         COL.purpleL)]
          .concat(queueShapes(st.after, qs2))
          .concat(treeShapes(alive, sx3, {baseY:baseY, rh:RH, states:tS2}))
          .concat([fitT(4.9, 6.02, {zh:'總 bits += ' + st.entry.w + '　→　' + bits,
                                   en:'total bits += ' + st.entry.w + '  ->  ' + bits},
                       {c:COL.tealL, fs:.30})]),
      panels:[{lbl:{zh:'queue', en:'queue'},
               chips:st.after.map((e, i) => chip(symsOf(e.n) + ' ' + e.w, i === st.slot ? 'act' : 'dim'))},
              {lbl:{zh:'目前總 bits', en:'bits so far'}, chips:[chip(String(bits), 'ok')]}],
      line:7,
      msg:st.slot > 0
        ? {zh:'新節點的重量是 ' + st.entry.w + '，比 ' + lighter + ' 都重，所以它要往後沉到第 ' + (st.slot + 1) +
              ' 位。這個「沉回去」不是實作細節，它就是演算法本身：合併過的子樹已經變重了，不該再被當成最輕的候選人。' +
              '順帶一提，總 bits 剛好等於所有合併重量的和——因為每合併一次，底下那 ' + st.entry.w +
              ' 個字元就各多揹一個 bit。',
           en:'The new node weighs ' + st.entry.w + ', more than ' + lighter +
              ', so it sinks back to position ' + (st.slot + 1) +
              '. That sink is not an implementation detail, it is the algorithm: a merged subtree has grown heavier and must stop being treated as a lightest candidate. Note too that the running bit total is simply the sum of the merge weights, because each merge adds one bit to every one of the ' + st.entry.w + ' characters underneath it.'}
        : {zh:'新節點重 ' + st.entry.w + '，但 queue 裡剩下的都更重，所以它留在最前面。' +
              '重點不是它排到第幾位，是每次都必須重新問一次「現在最輕的是誰」——問題本身已經被改掉了。' +
              '總 bits 累加 ' + st.entry.w + '，因為底下每個字元都多揹一個 bit。',
           en:'The new node weighs ' + st.entry.w + ', and everything left in the queue is heavier, so it stays at the front. What matters is not the position but that the question "which is lightest now" has to be asked again - the problem itself has changed. The bit total grows by ' + st.entry.w + ', one extra bit for every character below.'}});
  });

  const codeList = FREQ.map(f => f[0] + ' ×' + f[1] + '  →  ' + HCODES[f[0]]);
  F.push({shapes:[head({zh:'Huffman：' + HBITS + ' bits，比固定長度省 ' +
                            Math.round((FIXBITS - HBITS) / FIXBITS * 100) + '%',
                        en:'Huffman: ' + HBITS + ' bits, ' +
                           Math.round((FIXBITS - HBITS) / FIXBITS * 100) + '% below fixed length'})]
        .concat(treeShapes(alive, sx3, {baseY:baseY, rh:RH,
                  states:(function(){ const s = {}; FREQ.forEach(f => { s[f[0]] = 'ok'; }); return s; })()}))
        .concat(codeList.map((t, i) => S.t(.55 + (i % 3) * 3.2, 1.10 + Math.floor(i / 3) * .42, t,
                                           {c:COL.tealL, fs:.28, anchor:'start'})))
        .concat([fitT(4.9, 6.02, {zh:'5 次合併的重量加起來 = ' + HUF.steps.map(s => s.entry.w).join(' + ') + ' = ' + HBITS,
                                 en:'the five merge weights sum to ' + HUF.steps.map(s => s.entry.w).join(' + ') + ' = ' + HBITS},
                     {c:COL.pale, fs:.28})]),
    panels:[{lbl:{zh:'碼表', en:'codebook'}, chips:FREQ.map(f => chip(f[0] + ':' + HCODES[f[0]], 'ok'))},
            {lbl:{zh:'Huffman', en:'Huffman'}, chips:[chip(HBITS + ' bits', 'ok')]},
            {lbl:{zh:'固定長度', en:'fixed'}, chips:[chip(FIXBITS + ' bits', 'bad')]},
            {lbl:{zh:'prefix-free', en:'prefix-free'},
             chips:[chip(prefixFree(HCODES) ? 'yes' : 'no', 'ok')]}],
    line:11,
    msg:{zh:HBITS + ' bits，比固定長度的 ' + FIXBITS + ' bits 少 ' + (FIXBITS - HBITS) +
            '。兩個最常用的符號 E 和 A 拿到 2 bits，其他拿 3 bits——沒有任何符號的碼是另一個的開頭，所以解碼時不需要分隔符號，一路讀到有東西對上就切一刀。' +
            '和區間排程不同的是，這個貪心有完整的最佳性證明：沒有任何一種 prefix code 能把這 ' + TOTAL + ' 個字元壓到 ' + HBITS + ' bits 以下。',
         en:HBITS + ' bits against ' + FIXBITS + ' for the fixed-length code, a saving of ' + (FIXBITS - HBITS) +
            '. The two most frequent symbols, E and A, get 2 bits and the rest get 3. No codeword is a prefix of another, so decoding needs no separators at all: read bits until something matches, then cut. Unlike interval scheduling, this greedy comes with a full optimality proof - no prefix code can push these ' + TOTAL + ' characters below ' + HBITS + ' bits.'}});
  return F.list;
}

/* =======================================================================
 * Tab 4 - the silent failure: merge without re-sorting
 * ======================================================================= */
const CODE_LAD = [
  'items = sorted(freqs.items(), key=lambda kv: (kv[1], kv[0]))   # sorted once',
  'node, weight = items[0][0], items[0][1]',
  'for sym, w in items[1:]:                 # merge left to right, never re-sort',
  '    node, weight = (node, sym), weight + w',
  '',
  '# every merge still joins two subtrees, so this is still a legal prefix',
  '# code and still decodes perfectly - the tree is simply a ladder',
  'codes = {}; walk(node, "")',
  'assert is_prefix_free(codes)                        # True',
  'assert decode(encode(text, codes), codes) == text   # True'
];
const LORDER = leafOrder(LAD.root);
function slotFn(order, x0, w){ return sym => x0 + (order.indexOf(sym) + .5) * (w / order.length); }

function ladFrames(v){
  const F = new Frames();
  if (v === 0){
    const sx4 = slotFn(LORDER, .60, 8.8);
    const baseY = 5.20, RH = .62;
    const alive = LAD.queue.map(e => e.n);
    const head = (t, c) => fitT(4.9, .46, t, {c:c || COL.red, fs:.33});
    const qs = i => { const o = {}; if (i != null) o[i] = 'hot'; return o; };
    const qbox = (states, mergedIx) => LAD.queue.map((e, i) =>
      S.r(QX0 + i * QPITCH, QY, QW, QH, states[i] || (i < (mergedIx == null ? -1 : mergedIx) ? 'done' : 'idle'),
          e.n.sym + ' ' + e.w, {fs:.28, rx:.08}));

    F.push({shapes:[head({zh:'只排序一次，然後從左邊一路併過去',
                          en:'sort once, then merge left to right and never look back'})]
          .concat(qbox({}, null))
          .concat(treeShapes(alive, sx4, {baseY:baseY, rh:RH}))
          .concat([fitT(4.9, 6.02, {zh:'少掉的只有一行：把合併後的新節點放回去重新排序',
                                   en:'the only missing line is the one that puts the merged node back in order'},
                       {c:COL.red, fs:.30})]),
      panels:[{lbl:{zh:'排序後的 queue', en:'queue, sorted once'},
               chips:LAD.queue.map(e => chip(e.n.sym + ' ' + e.w, 'dim'))}],
      line:0,
      msg:{zh:'這個版本一開始完全正確：先照出現次數把六個符號排好，最輕的在最前面。接著它做的事聽起來也沒錯——「每次合併最輕的兩個」，從最左邊開始往右併。問題在於「最輕的兩個」這句話裡的「最輕」是會變的。',
           en:'This version starts out perfectly correct: sort the six symbols by frequency, lightest first. What it does next also sounds right - "merge the two lightest" - working left to right from the front. The trap is hidden in the word lightest, because which items are lightest keeps changing.'}});

    let bits = 0;
    LAD.steps.forEach((st, k) => {
      const merged = st.merged;
      bits += merged.w;
      const ai = alive.indexOf(st.a.n), bi = alive.indexOf(st.b.n);
      alive.splice(Math.max(ai, bi), 1); alive.splice(Math.min(ai, bi), 1);
      alive.push(merged.n);
      const states = {}; states[st.i] = 'hot'; if (k === 0) states[0] = 'hot';
      const tS = {}; tS[symsOf(merged.n)] = st.past ? 'bad' : 'act';
      const sinkX = QX0 + Math.min(NSYM - 1, st.i + st.past) * QPITCH + QW;
      const hint = st.past
        ? [S.e(QX0 + st.i * QPITCH + QW * .5, QY + QH + .30, sinkX - .10, QY + QH + .30,
               {s:'bad', w:.045, dash:'.10 .10'}),
           fitT(4.9, QY + QH + .62,
               {zh:'重新排序的話，' + merged.w + ' 會沉過 ' + st.past + ' 個更輕的',
                en:'after a re-sort, ' + merged.w + ' sinks past ' + st.past + ' lighter items'},
               {c:COL.red, fs:.26})]
        : [];
      F.push({shapes:[head({zh:'合併 ' + symsOf(st.a.n) + '(' + st.a.w + ') + ' + st.b.n.sym +
                                '(' + st.b.w + ') = ' + merged.w,
                            en:'merge ' + symsOf(st.a.n) + '(' + st.a.w + ') + ' + st.b.n.sym +
                               '(' + st.b.w + ') = ' + merged.w}, COL.orangeL)]
            .concat(qbox(states, st.i))
            .concat(hint)
            .concat(treeShapes(alive, sx4, {baseY:baseY, rh:RH, states:tS, r:.28}))
            .concat([fitT(4.9, 6.02, {zh:'總 bits += ' + merged.w + '　→　' + bits,
                                     en:'total bits += ' + merged.w + '  ->  ' + bits},
                         {c:st.past ? COL.red : COL.pale, fs:.29})]),
        panels:[{lbl:{zh:'手上的子樹', en:'the subtree in hand'},
                 chips:[chip(symsOf(merged.n) + ' ' + merged.w, st.past ? 'bad' : 'act')]},
                {lbl:{zh:'還沒併的', en:'not merged yet'},
                 chips:LAD.queue.slice(st.i + 1).map(e => chip(e.n.sym + ' ' + e.w,
                   e.w < merged.w ? 'hot' : 'dim'))},
                {lbl:{zh:'目前總 bits', en:'bits so far'}, chips:[chip(String(bits), st.past ? 'bad' : 'dim')]}],
        line:3,
        msg:st.past === 0
          ? {zh:'第一次合併和正確版本一模一樣：' + st.a.n.sym + ' 和 ' + st.b.n.sym +
                ' 本來就是最輕的兩個。錯誤還沒發生，但它已經被埋好了——合併出來的 ' + merged.w +
                ' 現在比後面好幾個都重，而這個版本不會再回頭看一眼。',
             en:'The first merge is identical to the correct version: ' + st.a.n.sym + ' and ' + st.b.n.sym +
                ' really are the two lightest. Nothing has gone wrong yet, but the fault is already planted - the merged node weighs ' + merged.w +
                ', heavier than several items still waiting, and this version will never look back at them.'}
          : {zh:'這裡就是無聲的錯誤：手上的子樹已經重 ' + merged.w + '，queue 裡還有 ' + st.past +
                ' 個比它輕的（' + LAD.queue.slice(st.i + 1).filter(e => e.w < merged.w).map(e => e.n.sym + ' ' + e.w).join('、') +
                '），正確的做法是把它沉回去、讓那些輕的先配對。少了重新排序，這棵子樹每被併一次就再深一層，' +
                '底下的 ' + merged.w + ' 個字元全部一起多揹一個 bit。',
             en:'This is where it goes wrong, silently. The subtree in hand already weighs ' + merged.w +
                ' while ' + st.past + ' lighter items are still waiting (' +
                LAD.queue.slice(st.i + 1).filter(e => e.w < merged.w).map(e => e.n.sym + ' ' + e.w).join(', ') +
                '); the correct move is to let it sink and pair those up first. Without the re-sort this same subtree is merged again and again, gaining a level each time, and every one of the ' + merged.w + ' characters underneath pays another bit.'}});
    });

    F.push({shapes:[head({zh:'長出來的是一條梯子：最深的碼 ' + Math.max.apply(null, Object.keys(LCODES).map(k => LCODES[k].length)) +
                              ' bits，Huffman 只要 ' + Math.max.apply(null, Object.keys(HCODES).map(k => HCODES[k].length)),
                          en:'what grew is a ladder: deepest codeword ' +
                             Math.max.apply(null, Object.keys(LCODES).map(k => LCODES[k].length)) +
                             ' bits against Huffman’s ' +
                             Math.max.apply(null, Object.keys(HCODES).map(k => HCODES[k].length))})]
          .concat(treeShapes(alive, sx4, {baseY:baseY, rh:RH, r:.28,
                    states:(function(){ const s = {}; FREQ.forEach(f => { s[f[0]] = 'bad'; }); return s; })()}))
          .concat(FREQ.map((f, i) => S.t(.55 + (i % 3) * 3.2, 1.10 + Math.floor(i / 3) * .42,
                                         f[0] + ' ×' + f[1] + '  →  ' + LCODES[f[0]],
                                         {c:COL.red, fs:.28, anchor:'start'})))
          .concat([fitT(4.9, 6.02, {zh:LBITS + ' bits：Huffman 省的 ' + (FIXBITS - HBITS) +
                                       ' bits 只剩 ' + (FIXBITS - LBITS),
                                   en:LBITS + ' bits: of Huffman\'s ' + (FIXBITS - HBITS) +
                                      ' saved bits only ' + (FIXBITS - LBITS) + ' survive'},
                       {c:COL.red, fs:.29})]),
      panels:[{lbl:{zh:'碼表', en:'codebook'}, chips:FREQ.map(f => chip(f[0] + ':' + LCODES[f[0]], 'bad'))},
              {lbl:{zh:'壞掉的版本', en:'broken'}, chips:[chip(LBITS + ' bits', 'bad')]},
              {lbl:{zh:'Huffman', en:'Huffman'}, chips:[chip(HBITS + ' bits', 'ok')]}],
      line:7,
      msg:{zh:'最後長成一條梯子：每個新符號都掛在樹的最外層，所以 S 的碼要 ' + LCODES['S'].length +
              ' bits。' + LBITS + ' bits 是合法的、可以解碼的、每個符號都有唯一的碼——它只是比較大。' +
              '固定長度要 ' + FIXBITS + ' bits，Huffman 要 ' + HBITS + '，這個版本 ' + LBITS +
              '：省下來的東西幾乎全部被那一行沒寫的重新排序吃掉了。',
           en:'The result is a ladder: every new symbol hangs off the outside of the tree, so S needs ' +
              LCODES['S'].length + ' bits. Those ' + LBITS + ' bits are legal, decodable and unambiguous - they are simply more. Fixed length costs ' +
              FIXBITS + ', Huffman ' + HBITS + ', and this version ' + LBITS +
              ': almost the entire saving was eaten by the one line of re-sorting that was never written.'}});
    return F.list;
  }

  /* variant 1 - the two trees side by side */
  const sxL = slotFn(LORDER, .30, 4.35), sxH = slotFn(HORDER, 5.10, 4.35);
  const baseY = 5.05, RH = .58;
  const okS = {}, badS = {};
  FREQ.forEach(f => { okS[f[0]] = 'ok'; badS[f[0]] = 'bad'; });
  const both = () => treeShapes([LAD.root], sxL, {baseY:baseY, rh:RH, r:.24, states:badS, bitfs:.20, subfs:.21})
    .concat(treeShapes([HUF.root], sxH, {baseY:baseY, rh:RH, r:.24, states:okS, bitfs:.20, subfs:.21}))
    .concat([fitT(2.45, .46, {zh:'沒有重新排序：' + LBITS + ' bits', en:'no re-sort: ' + LBITS + ' bits'},
                 {c:COL.red, fs:.32}),
             fitT(7.25, .46, {zh:'Huffman：' + HBITS + ' bits', en:'Huffman: ' + HBITS + ' bits'},
                 {c:COL.tealL, fs:.32}),
             S.e(4.90, .70, 4.90, 5.70, {s:'soft', arrow:false, w:.025, o:.45})]);
  F.push({shapes:both().concat([
      fitT(2.45, 5.92, {zh:'最深 ' + Math.max.apply(null, FREQ.map(f => LCODES[f[0]].length)) + ' bits',
                       en:'deepest ' + Math.max.apply(null, FREQ.map(f => LCODES[f[0]].length)) + ' bits'},
          {c:COL.red, fs:.28}),
      fitT(7.25, 5.92, {zh:'最深 ' + Math.max.apply(null, FREQ.map(f => HCODES[f[0]].length)) + ' bits',
                       en:'deepest ' + Math.max.apply(null, FREQ.map(f => HCODES[f[0]].length)) + ' bits'},
          {c:COL.tealL, fs:.28}),
      fitT(4.9, 6.28, {zh:'同樣六個葉子、同樣的重量，差別只在哪兩棵子樹被配成兄弟',
                      en:'the same six leaves with the same weights - only the pairing differs'},
          {c:COL.pale, fs:.28})]),
    panels:[{lbl:{zh:'壞掉的版本', en:'broken'}, chips:[chip(LBITS + ' bits', 'bad')]},
            {lbl:{zh:'Huffman', en:'Huffman'}, chips:[chip(HBITS + ' bits', 'ok')]},
            {lbl:{zh:'固定長度', en:'fixed'}, chips:[chip(FIXBITS + ' bits', 'dim')]}],
    line:6,
    msg:{zh:'左邊那棵樹不是壞掉的資料結構，它是一棵完全正常的二元樹，只是形狀爛。' + LBITS + ' bits 對 ' +
            HBITS + ' bits，多了 ' + (LBITS - HBITS) + ' bits，' +
            Math.round((LBITS - HBITS) / HBITS * 100) + '%。而且兩棵樹的葉子完全一樣，重量也完全一樣——' +
            '差別只在每一步把哪兩棵配成兄弟。',
         en:'The tree on the left is not a corrupted data structure; it is a perfectly ordinary binary tree with a bad shape. ' +
            LBITS + ' bits against ' + HBITS + ', which is ' + (LBITS - HBITS) + ' more, or ' +
            Math.round((LBITS - HBITS) / HBITS * 100) + '%. Both trees carry the same six leaves with the same weights - all that differs is which two were made siblings at each step.'}});

  const enc = s => Array.prototype.map.call(s, c => LCODES[c]).join('');
  const round = decode(enc(MSG), LCODES) === MSG;
  F.push({shapes:both().concat([
      fitT(4.9, 5.92, {zh:'prefix-free：兩邊都是　round-trip：兩邊都對　Kraft 和：兩邊都是 1.000',
                      en:'prefix-free: both.   round-trip: both.   Kraft sum: 1.000 on both'},
          {c:COL.orangeL, fs:.29}),
      fitT(4.9, 6.28, {zh:'沒有任何一個檢查會抓到它——它不是錯的答案，是比較貴的答案',
                      en:'no assertion catches this - it is not a wrong answer, it is an expensive one'},
          {c:COL.pale, fs:.28})]),
    panels:[{lbl:{zh:'prefix-free', en:'prefix-free'},
             chips:[chip('heap ' + (prefixFree(HCODES) ? 'yes' : 'no'), 'ok'),
                    chip('ladder ' + (prefixFree(LCODES) ? 'yes' : 'no'), 'bad')]},
            {lbl:{zh:'解回原文', en:'round-trip'},
             chips:[chip(round ? 'exact' : 'broken', round ? 'ok' : 'bad')]},
            {lbl:{zh:'Kraft', en:'Kraft'},
             chips:[chip(kraft(HCODES).toFixed(3), 'ok'), chip(kraft(LCODES).toFixed(3), 'bad')]}],
    line:9,
    msg:{zh:'把 ' + TOTAL + ' 個字元用梯子碼表編碼再解回來，字字相符；沒有任何碼是另一個的開頭；Kraft 和剛好 1.000，' +
            '代表這棵樹連一個分支都沒有浪費。所有你想得到的正確性檢查都會通過。這就是貪心錯誤的標準長相——' +
            '不是例外、不是亂碼，是一個合法但比較貴的答案，而唯一能發現它的方法是跟另一個實作比數字。',
         en:'Encode the ' + TOTAL + ' characters with the ladder codebook and decode them back: every character matches. No codeword prefixes another. The Kraft sum is exactly 1.000, meaning the tree wastes no branch at all. Every correctness check you can think of passes. This is what a greedy bug looks like - not an exception, not garbage, but a legal and more expensive answer, and the only way to notice is to compare the number against another implementation.'}});

  F.push({shapes:both().concat([
      fitT(2.45, 5.92, LAD.steps.map(s => s.merged.w).join(' + ') + ' = ' + LBITS, {c:COL.red, fs:.27}),
      fitT(7.25, 5.92, HUF.steps.map(s => s.entry.w).join(' + ') + ' = ' + HBITS, {c:COL.tealL, fs:.27}),
      fitT(4.9, 6.28, {zh:'總 bits 是合併重量的總和，重的越晚合併越好',
                      en:'total bits = sum of merge weights, so merge heavy nodes last'},
          {c:COL.pale, fs:.28})]),
    panels:[{lbl:{zh:'合併重量（壞）', en:'merge weights (broken)'},
             chips:LAD.steps.map(s => chip(String(s.merged.w), 'bad'))},
            {lbl:{zh:'合併重量（Huffman）', en:'merge weights (Huffman)'},
             chips:HUF.steps.map(s => chip(String(s.entry.w), 'ok'))}],
    line:3,
    msg:{zh:'把帳攤開最清楚：總 bits 等於每次合併後那個新節點重量的總和。梯子版把同一棵越來越重的子樹一路往上疊，' +
            '所以後面每一筆都很貴（' + LAD.steps.map(s => s.merged.w).join('、') + '）；Huffman 讓重的子樹盡量晚出場（' +
            HUF.steps.map(s => s.entry.w).join('、') + '）。' +
            '「每次取最輕的兩個」之所以最佳，就是因為這條加法帳，而 heap 的存在只是為了讓那個「最輕」每一步都是真的。',
         en:'The clearest way to see it is the bill: the bit total equals the sum of the weights created by each merge. The ladder keeps stacking the same ever-heavier subtree, so every later merge is expensive (' +
            LAD.steps.map(s => s.merged.w).join(', ') + '), while Huffman keeps the heavy subtrees out of the sum as long as it can (' +
            HUF.steps.map(s => s.entry.w).join(', ') + '). That addition is the whole reason "take the two lightest" is optimal - and the heap exists only to make sure that "lightest" is still true at every step.'}});
  return F.list;
}

/* =======================================================================
 * Tab 5 - LeetCode 435, the same greedy with the question inverted
 * ======================================================================= */
const CODE_435 = [
  'class Solution:',
  '    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:',
  '        intervals.sort(key=lambda iv: iv[1])   # earliest finishing first',
  '        kept, last_end = 0, float("-inf")',
  '        for s, e in intervals:',
  '            if s >= last_end:                  # fits after what we kept',
  '                kept += 1',
  '                last_end = e',
  '        return len(intervals) - kept           # removing == not keeping'
];
const LC435 = [
  {name:{zh:'LeetCode 的範例', en:'the LeetCode example'},
   iv:[['a', 1, 2], ['b', 2, 3], ['c', 3, 4], ['d', 1, 3]], fmt:u => String(u)},
  {name:{zh:'同一間會議室', en:'the meeting room'}, iv:MEET, fmt:clock}
];

function ivFrames(v){
  const F = new Frames(), C = LC435[v];
  const list = C.iv.slice(), n = list.length;
  const lo = Math.min.apply(null, list.map(m => m[1])), hi = Math.max.apply(null, list.map(m => m[2]));
  const px = u => .70 + (u - lo) / (hi - lo) * 8.2;
  const order = list.slice().sort((a, b) => a[2] - b[2]);
  const rowH = Math.min(.50, 4.0 / n), y0 = 1.45, bh = rowH * .66;
  const rowY = m => y0 + order.indexOf(m) * rowH;

  /* run it */
  const kept = [], log = [];
  let lastEnd = null;
  order.forEach(m => {
    if (lastEnd == null || m[1] >= lastEnd){ kept.push(m); lastEnd = m[2]; log.push({m:m, ok:true}); }
    else log.push({m:m, ok:false, blocker:kept[kept.length - 1]});
  });
  const removed = n - kept.length;

  const axis = () => {
    const out = [], ticks = 6;
    for (let i = 0; i <= ticks; i++){
      const u = lo + (hi - lo) * i / ticks;
      out.push(S.t(px(u), 1.02, C.fmt(Math.round(u)), {c:COL.grey, fs:.24}));
      out.push(S.e(px(u), 1.10, px(u), 1.28, {s:'soft', arrow:false, w:.028, o:.75}));
    }
    out.push(S.e(px(lo), 1.28, px(hi), 1.28, {s:'soft', arrow:false, w:.025, o:.5}));
    return out;
  };
  function stage(states, curName, le, cap, capc){
    const sh = [fitT(4.9, .46, {zh:'LeetCode 435：最少刪掉幾個區間，剩下的才不重疊',
                               en:'LeetCode 435 - the fewest intervals to delete so the rest do not overlap'},
                    {c:COL.tealL, fs:.32})].concat(axis());
    order.forEach(m => {
      const x = px(m[1]), w = Math.max(.14, px(m[2]) - px(m[1])), y = rowY(m);
      sh.push(S.r(x, y, w, bh, states[m[0]] || 'idle', w > .42 ? m[0] : '', {fs:.26, rx:.07}));
      if (w <= .42) sh.push(S.t(x + w / 2, y + bh + .24, m[0], {c:COL.pale, fs:.22}));
      sh.push(S.t(.56, y + bh * .74, m[0], {c:m[0] === curName ? COL.orangeL : COL.grey, fs:.26, anchor:'end'}));
      sh.push(S.t(9.62, y + bh * .74, '[' + C.fmt(m[1]) + ', ' + C.fmt(m[2]) + ']',
                  {c:COL.grey, fs:.21, anchor:'end'}));
    });
    if (le != null){
      sh.push(S.e(px(le), y0 - .20, px(le), y0 + n * rowH + .05, {s:'act', arrow:false, w:.042, dash:'.10 .10'}));
      sh.push(S.t(px(le), y0 - .32, 'last_end ' + C.fmt(le), {c:COL.purpleL, fs:.24}));
    }
    if (cap) sh.push(fitT(4.9, 6.10, cap, {c:capc || COL.pale, fs:.30}));
    return sh;
  }

  const states = {};
  F.push({shapes:stage(states, null, null,
      {zh:'刪最少 ＝ 留最多，所以真正要算的是「最多能留幾個」',
       en:'fewest deletions = most kept, so compute how many fit'},
      COL.orangeL),
    panels:[{lbl:{zh:'區間', en:'intervals'}, chips:list.map(m => chip(m[0], 'dim'))},
            {lbl:{zh:'留下', en:'kept'}, chips:[]}],
    line:1,
    msg:{zh:'這題看起來和排會議室不一樣，其實是同一題換句話問。要刪掉的數量 = 總數 − 留下來的數量，' +
            '而「留下來且互不重疊的最多有幾個」就是前面那個排程問題，一個字都不用改。' +
            '最小化和最大化能這樣互換，是因為刪和留是同一個決定的兩面。',
         en:'This looks like a different problem from booking the meeting room, but it is the same question phrased backwards. Deletions equal the total minus what is kept, and "the largest set that can be kept without overlapping" is precisely the scheduling problem from earlier, unchanged. The swap between minimising and maximising is free because deleting and keeping are two sides of one decision.'}});

  let le = null;
  log.forEach(e => {
    const m = e.m;
    states[m[0]] = 'hot';
    F.push({shapes:stage(states, m[0], le,
        e.ok ? {zh:m[0] + ' 留下，last_end 推到 ' + C.fmt(m[2]),
                en:m[0] + ' is kept, last_end moves to ' + C.fmt(m[2])}
             : {zh:m[0] + ' 和留下的 ' + e.blocker[0] + ' 重疊，刪掉',
                en:m[0] + ' overlaps the kept ' + e.blocker[0] + ' - delete it'},
        e.ok ? COL.tealL : COL.red),
      panels:[{lbl:{zh:'目前', en:'current'},
               chips:[chip(m[0] + ' [' + C.fmt(m[1]) + ',' + C.fmt(m[2]) + ']', 'hot')]},
              {lbl:{zh:'留下', en:'kept'}, chips:kept.filter(x => states[x[0]] === 'ok').map(x => chip(x[0], 'ok'))},
              {lbl:{zh:'last_end', en:'last_end'},
               chips:[chip(le == null ? '—' : C.fmt(le), le == null ? 'empty' : 'dim')]}],
      line:e.ok ? 5 : 4,
      msg:e.ok
        ? {zh:m[0] + ' 的開頭 ' + C.fmt(m[1]) + ' 不早於 last_end ' + (le == null ? '（還沒有）' : C.fmt(le)) +
              '，收下它並把 last_end 推到 ' + C.fmt(m[2]) + '。排序鍵是結束時間，所以 last_end 只會往後走，' +
              '一個變數就記完了所有已留區間的資訊。',
           en:m[0] + ' starts at ' + C.fmt(m[1]) + ', not before last_end ' + (le == null ? '(none yet)' : C.fmt(le)) +
              ', so keep it and push last_end to ' + C.fmt(m[2]) +
              '. Because the sort key is the end point, last_end only ever moves forward, and that single variable summarises everything kept so far.'}
        : {zh:m[0] + ' 蓋到 ' + e.blocker[0] + '，兩個只能留一個。留 ' + e.blocker[0] + ' 是因為它結束得早，' +
              '往後留下的空間比較大；刪掉的數量一樣是 1，但留下的那個對後面比較有利。',
           en:m[0] + ' overlaps ' + e.blocker[0] + ', so only one of the two can stay. Keeping ' + e.blocker[0] +
              ' is right because it ends sooner and leaves more room behind it; either choice deletes one interval, but this one is worth more to everything that follows.'}});
    states[m[0]] = e.ok ? 'ok' : 'bad';
    if (e.ok) le = m[2];
  });

  F.push({shapes:stage(states, null, le,
      {zh:'留下 ' + kept.length + ' 個，刪掉 ' + removed + ' 個',
       en:kept.length + ' kept, ' + removed + ' deleted'}, COL.tealL),
    panels:[{lbl:{zh:'留下', en:'kept'}, chips:kept.map(m => chip(m[0], 'ok'))},
            {lbl:{zh:'答案', en:'answer'}, chips:[chip(String(removed), 'ok')]}],
    line:8,
    msg:{zh:'答案是 ' + removed + '。同一份程式碼、同一個排序鍵，換一個 return 就從「排最多場會」變成「刪最少個區間」。' +
            'LeetCode 452 射氣球也是這張皮：把「留下」改成「一箭射爆一串」，判斷式從 s >= last_end 變成 s > last，' +
            '因為端點相接在那題算射中、在這題算不重疊。整個差別就是一個等號。',
         en:'The answer is ' + removed + '. Same code, same sort key; changing only the return turns "book the most meetings" into "delete the fewest intervals". LeetCode 452, bursting balloons with arrows, wears the same skin: keeping becomes shooting, and the test changes from s >= last_end to s > last, because touching endpoints count as a hit there and as non-overlapping here. The entire difference is one equals sign.'}});
  return F.list;
}

/* =======================================================================
 * Tab 6 - LeetCode 134, a greedy whose proof is about what it discards
 * ======================================================================= */
const CODE_134 = [
  'class Solution:',
  '    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:',
  '        if sum(gas) < sum(cost):',
  '            return -1              # not enough fuel in the whole loop',
  '        start, tank = 0, 0',
  '        for i in range(len(gas)):',
  '            tank += gas[i] - cost[i]',
  '            if tank < 0:           # i is not reachable from start',
  '                start, tank = i + 1, 0   # and nor is it from anywhere between',
  '        return start'
];
const GAS = [{gas:[1, 2, 3, 4, 5], cost:[3, 4, 5, 1, 2]},
             {gas:[2, 3, 4], cost:[3, 4, 3]}];

function gasFrames(v){
  const F = new Frames(), G = GAS[v], n = G.gas.length;
  const diff = G.gas.map((g, i) => g - G.cost[i]);
  const sumG = G.gas.reduce((a, b) => a + b, 0), sumC = G.cost.reduce((a, b) => a + b, 0);
  const feasible = sumG >= sumC;
  /* record the real run */
  const log = [];
  let start = 0, tank = 0;
  for (let i = 0; i < n; i++){
    tank += diff[i];
    const reset = tank < 0;
    log.push({i:i, tank:tank, start:start, reset:reset});
    if (reset){ start = i + 1; tank = 0; }
  }
  const answer = feasible ? start : -1;
  /* brute force ground truth */
  const brutes = [];
  for (let s = 0; s < n; s++){
    let t = 0, ok = true;
    for (let k = 0; k < n; k++){ t += diff[(s + k) % n]; if (t < 0){ ok = false; break; } }
    if (ok) brutes.push(s);
  }

  const PITCH = 8.3 / n, BX = i => .95 + i * PITCH, BW = Math.min(1.5, PITCH - .25);
  const BASE = 4.25, USC = .30;
  function stage(upto, cur, st, cap, capc){
    const sh = [fitT(4.9, .46, {zh:'LeetCode 134：一圈加油站，從哪裡出發才繞得完',
                               en:'LeetCode 134 - one ring of gas stations: where can the loop start'},
                    {c:COL.tealL, fs:.32})];
    for (let i = 0; i < n; i++){
      sh.push(S.r(BX(i), 1.20, BW, .74, st[i] || 'idle', String(i),
                  {fs:.30, top:'gas ' + G.gas[i], sub:'cost ' + G.cost[i], subfs:.24, topfs:.24}));
      const h = Math.abs(diff[i]) * USC;
      sh.push(S.r(BX(i) + BW * .22, diff[i] >= 0 ? BASE - h : BASE, BW * .56, Math.max(.06, h),
                  diff[i] >= 0 ? 'ok' : 'bad', '', {rx:.05}));
      sh.push(S.t(BX(i) + BW * .5, diff[i] >= 0 ? BASE - h - .16 : BASE + h + .32,
                  (diff[i] >= 0 ? '+' : '') + diff[i], {c:diff[i] >= 0 ? COL.tealL : COL.red, fs:.26}));
    }
    sh.push(S.e(.55, BASE, 9.35, BASE, {s:'soft', arrow:false, w:.028, o:.6}));
    sh.push(S.t(.10, BASE + .36, {zh:'油量差 gas − cost', en:'gas − cost'},
                {c:COL.grey, fs:.20, anchor:'start'}));
    for (let i = 0; i < upto; i++)
      sh.push(S.t(BX(i) + BW * .5, 5.30, 'tank ' + log[i].tank,
                  {c:log[i].reset ? COL.red : COL.pale, fs:.26}));
    if (cur != null){
      const s0 = log[cur].start;
      sh.push(S.e(BX(s0) + BW * .5, 5.68, BX(cur) + BW * .5, 5.68,
                  {s:log[cur].reset ? 'bad' : 'act', arrow:false, w:.05}));
      sh.push(S.t(BX(s0) + BW * .5, 5.98, {zh:'從 ' + s0 + ' 出發', en:'start ' + s0},
                  {c:COL.purpleL, fs:.25}));
    }
    if (cap) sh.push(fitT(4.9, 6.30, cap, {c:capc || COL.pale, fs:.29}));
    return sh;
  }

  F.push({shapes:stage(0, null, {},
      {zh:'總油量 ' + sumG + '，總花費 ' + sumC + '　→　' +
          (feasible ? '油夠，一定有解' : '油不夠，直接回傳 −1'),
       en:'total gas ' + sumG + ', total cost ' + sumC + '  ->  ' +
          (feasible ? 'enough fuel, a start must exist' : 'not enough fuel, return -1 immediately')},
      feasible ? COL.tealL : COL.red),
    panels:[{lbl:{zh:'gas', en:'gas'}, chips:G.gas.map(g => chip(String(g), 'dim'))},
            {lbl:{zh:'cost', en:'cost'}, chips:G.cost.map(c => chip(String(c), 'dim'))},
            {lbl:{zh:'總和', en:'totals'},
             chips:[chip(sumG + ' vs ' + sumC, feasible ? 'ok' : 'bad')]}],
    line:2,
    msg:feasible
      ? {zh:'先看總帳：全程能加的油 ' + sumG + '，全程要燒的 ' + sumC +
            '，油夠。這一比不只是省時間的捷徑，它本身就是一半的證明——只要總油量夠，就一定存在一個可行的起點，' +
            '剩下的工作只是把它指出來，而不是判斷有沒有。',
         en:'Start with the books: ' + sumG + ' units of fuel available over the whole loop against ' + sumC +
            ' burned, so there is enough. That comparison is not merely a shortcut, it is half the proof - whenever the total is enough some valid start must exist, and the remaining work is to point at it rather than to decide whether it exists.'}
      : {zh:'總油量 ' + sumG + ' 比總花費 ' + sumC + ' 少，不管從哪裡出發，繞完一圈的淨變化都是 ' +
            (sumG - sumC) + '，一定會在某處斷油。這種「先看總和」的判斷在貪心題裡很常見：' +
            '它把不可能的情況一次擋掉，貪心就只需要處理保證有解的情形。',
         en:'Total gas ' + sumG + ' is below total cost ' + sumC + '. Wherever the drive begins, one full lap changes the tank by ' +
            (sumG - sumC) + ', so it must run dry somewhere. Checking the total first is a common shape in greedy problems: it rules the impossible case out in one line, leaving the greedy to handle only inputs that are guaranteed solvable.'}});

  log.forEach((e, i) => {
    const st = {};
    for (let k = e.start; k <= i; k++) st[k] = e.reset ? 'bad' : 'act';
    st[i] = 'hot';
    F.push({shapes:stage(i + 1, i, st,
        e.reset ? {zh:'tank = ' + e.tank + ' < 0：從 ' + e.start + ' 出發到不了 ' + i + '，起點跳到 ' + (i + 1),
                   en:'tank = ' + e.tank + ' < 0: station ' + i + ' is unreachable from ' + e.start +
                      ', the start jumps to ' + (i + 1)}
                : {zh:'tank = ' + e.tank + '，還撐得住，繼續往前開',
                   en:'tank = ' + e.tank + ', still afloat - keep driving'},
        e.reset ? COL.red : COL.tealL),
      panels:[{lbl:{zh:'目前起點', en:'start'}, chips:[chip(String(e.start), 'act')]},
              {lbl:{zh:'油箱', en:'tank'}, chips:[chip(String(e.tank), e.reset ? 'bad' : 'ok')]},
              {lbl:{zh:'這一站', en:'station'},
               chips:[chip(i + ': ' + (diff[i] >= 0 ? '+' : '') + diff[i], 'hot')]}],
      line:e.reset ? 8 : 6,
      msg:e.reset
        ? {zh:'油箱在 ' + i + ' 站變成 ' + e.tank + '，所以 ' + e.start + ' 不是起點。真正值得記住的是下一句：' +
              e.start + ' 和 ' + i + ' 之間的每一站也都不是起點。因為從 ' + e.start +
              ' 一路開過來，每一站的油箱都還是非負的；如果改成從中間某站出發，到達 ' + i +
              ' 時手上的油只會更少，不會更多。一次掃描之所以夠，就是因為每次失敗都一口氣排除掉一整段，而不是一站。',
           en:'The tank hits ' + e.tank + ' at station ' + i + ', so ' + e.start +
              ' is not the answer. The sentence worth remembering is the next one: no station between ' + e.start +
              ' and ' + i + ' is the answer either. Driving from ' + e.start +
              ' the tank was non-negative at every stop along the way, so starting from one of those stops instead leaves strictly less fuel on arrival at ' + i +
              ', never more. One pass suffices precisely because each failure eliminates an entire stretch rather than a single station.'}
        : {zh:'這一站淨變化 ' + (diff[i] >= 0 ? '+' : '') + diff[i] + '，油箱 ' + e.tank +
              ' 仍然沒有見底，所以 ' + e.start + ' 到目前為止還活著。注意程式從頭到尾只維護兩個數字：' +
              '目前起點和從那裡累積到現在的油量——不需要知道中間任何一站的細節。',
           en:'This station nets ' + (diff[i] >= 0 ? '+' : '') + diff[i] + ' and the tank stands at ' + e.tank +
              ', still above empty, so ' + e.start + ' is alive so far. Note the program carries exactly two numbers the whole way: the current candidate start and the fuel accumulated since it - no detail about any station in between is ever needed.'}});
  });

  F.push({shapes:stage(n, null, (function(){ const s = {};
      if (feasible) for (let k = 0; k < n; k++) s[(answer + k) % n] = 'ok'; return s; })(),
      feasible ? {zh:'答案：從 ' + answer + ' 號站出發（窮舉每個起點得到 ' + brutes.join('、') + '）',
                  en:'answer: start at station ' + answer + ' (brute force over every start gives ' + brutes.join(', ') + ')'}
               : {zh:'答案：−1（窮舉每個起點也都繞不完）',
                  en:'answer: -1 (brute force agrees no start completes the loop)'},
      feasible ? COL.tealL : COL.red),
    panels:[{lbl:{zh:'答案', en:'answer'}, chips:[chip(String(answer), feasible ? 'ok' : 'bad')]},
            {lbl:{zh:'窮舉驗證', en:'brute force'},
             chips:brutes.length ? brutes.map(s => chip(String(s), 'ok')) : [chip('none', 'bad')]}],
    line:9,
    msg:feasible
      ? {zh:'答案 ' + answer + '，和逐一試每個起點跑一圈的窮舉結果一致。這題的貪心沒有排序鍵，' +
            '它的「貪」在於每次斷油就把整段候選人一次丟掉，而且丟掉的理由是可以證明的。' +
            '和今天前面兩個例子放在一起看：區間排程的證明在「換進去不會變差」，Huffman 的證明在「最輕的兩個注定是兄弟」，' +
            '這題的證明在「失敗的那一段整段都不用再試」——貪心演算法的難處永遠在這種句子，不在程式碼。',
         en:'The answer is ' + answer + ', agreeing with the brute force that drives a full lap from every station. This greedy has no sort key at all; what makes it greedy is that each dry tank throws away a whole stretch of candidates at once, and the throwing away is justified by a proof. Put it beside the other two examples: interval scheduling rests on "swapping it in never hurts", Huffman on "the two lightest are destined to be siblings", and this one on "everything in the failed stretch fails too". The hard part of a greedy algorithm always lives in a sentence like that, never in the code.'}
      : {zh:'−1，窮舉也同意。這個分頁的價值在於它示範了另一種貪心的證明形式：' +
            '前面的例子靠「交換不會變差」，這裡靠「失敗的那一段可以整段丟掉」。兩種都是在論證「被丟掉的東西不可能更好」，' +
            '只是丟的單位不同。',
         en:'-1, and the brute force agrees. What this tab is worth is showing a second shape of greedy proof: the earlier examples argued that an exchange never hurts, while this one argues that a failed stretch can be discarded wholesale. Both are arguments that the discarded options could not have been better; only the unit being discarded differs.'}});
  return F.list;
}

/* ======================================================================== */
const DAY_META = {
  title:{zh:'Day 38 — 貪心演算法：區間排程與 Huffman 編碼',
         en:'Day 38 - Greedy: interval scheduling and Huffman coding'},
  sub:{zh:'貪心法只有一個 sort 加一次掃描，難的不是寫，是證明那個排序鍵是對的——錯的鍵不會報錯，只會默默少排兩個人。',
       en:'A greedy algorithm is one sort and one pass. Writing it is easy; proving the sort key is the right one is not - and a wrong key never raises, it just quietly seats fewer people.'},
  tabs:[
    {
      id:'sched', label:{zh:'三個排序鍵', en:'three sort keys'},
      stage:{zh:'同樣八個申請、同一間會議室，換一個排序鍵就換一張時間表',
             en:'the same eight requests and one room - one sort key apart, three different timetables'},
      view:VIEW,
      variants:[{zh:'按結束時間（5 場）', en:'by end time - 5'},
                {zh:'按開始時間（3 場）', en:'by start time - 3'},
                {zh:'按時長（4 場）', en:'by duration - 4'}],
      idea:{zh:'區間排程的程式短到可以背：sort 一次，由前往後掃，撞到就跳過。所有的智慧都壓縮在那個排序鍵裡，而三個候選鍵看起來一樣合理——先結束的、先申請的、時間最短的。按結束時間排能排進 5 場，按開始時間 3 場，按時長 4 場，三張時間表全部合法、全部沒有重疊，程式一次都沒有報錯。這就是貪心法最該提防的地方：錯的鍵不會讓程式壞掉，只會讓答案變差，而「它跑出結果了」從來不是正確性的證據。為什麼是結束時間？因為房間一還出來，剩下的時間就完全乾淨，而最早還出來的那一場留下的乾淨時間最長——後面能排幾場只跟這段長度有關。',
            en:'The code for interval scheduling is short enough to memorise: sort once, sweep forward, skip anything that clashes. All of the intelligence is compressed into the sort key, and three candidate keys look equally sensible - finishes first, asked first, takes least time. By end time five meetings fit, by start time three, by duration four, and all three timetables are legal with no overlap anywhere; nothing raised once. That is the thing to fear about greedy algorithms: a wrong key does not break the program, it just degrades the answer, and "it produced a result" is never evidence of correctness. Why the end time? Because the moment the room is free the rest of the day is completely clean, the earliest finisher leaves the longest clean stretch, and how many meetings fit afterwards depends on nothing else.'},
      legend:[['#ff9736', {zh:'正在考慮的申請', en:'the request being considered'}],
              ['#3fe0dd', {zh:'排進去了', en:'booked'}],
              ['#9d6bff', {zh:'last_end：房間何時空出來', en:'last_end - when the room frees up'}],
              ['#2f5661', {zh:'被跳過', en:'skipped'}]],
      code:CODE_SCHED, build:schedFrames
    },
    {
      id:'exchange', label:{zh:'交換論證', en:'the exchange argument'},
      stage:{zh:'把「先收下最早結束的那一場不會虧」變成可以跑出來的數字',
             en:'turning "committing to the earliest finisher costs nothing" into a number you can run'},
      view:VIEW,
      idea:{zh:'貪心法唯一需要證明的是第一步，因為第一步之後剩下的問題和原問題同形，同一句話可以一路遞迴下去。這裡把證明當成實驗跑：窮舉 2⁸ 個子集合得到真正的最佳解 5 場，再把最早結束的 B 強制釘進時間表、對剩下的重跑一次窮舉，得到 1 + 4 = 5。兩個數字相等，代表「存在一個最佳解含 B」，所以貪心收下 B 沒有放棄任何東西。B 之所以安全，是因為它結束得最早、往右不擋任何人——換成別的申請當第一場只會擋掉更多。同一套推理在「每場會有不同價值」的版本上就失效了：那時 B 值 1 塊、A 值 8 塊，換進去會虧，只能回去寫 DP。',
            en:'The only step a greedy algorithm has to justify is its first one, because after that the remaining problem has the same shape and the same sentence recurses all the way down. Here the proof is run as an experiment: brute-forcing all 2^8 subsets gives the true optimum of 5, then B, the earliest finisher, is pinned into the schedule and the brute force is repeated over what is left, giving 1 + 4 = 5. The two numbers agree, which says some optimal schedule contains B, so committing to it gives nothing away. B is safe because it finishes before anything else and therefore blocks nothing to its right; any other first pick can only block more. The same reasoning collapses in the weighted version, where B is worth 1 and A is worth 8, the exchange loses money, and a DP table becomes unavoidable.'},
      legend:[['#3fe0dd', {zh:'目前最佳的一組', en:'the best subset so far'}],
              ['#ff9736', {zh:'最早結束的那一場', en:'the earliest finisher'}],
              ['#ff5c5c', {zh:'被強制選入的 B 擋掉', en:'blocked by forcing B in'}],
              ['#0a6b74', {zh:'沒被選到', en:'not in this subset'}]],
      code:CODE_EX, build:exFrames
    },
    {
      id:'huffman', label:{zh:'Huffman 編碼', en:'Huffman coding'},
      stage:{zh:'每次合併最輕的兩棵子樹，總 bits 就是所有合併重量的和',
             en:'merge the two lightest subtrees every time - the bit total is the sum of the merge weights'},
      view:VIEW,
      idea:{zh:'Huffman 是少數有完整最佳性證明的貪心法。把每個符號放在二元樹的葉子上，碼長就是深度，要最小化的是「深度 × 出現次數」的總和。關鍵觀察有兩個：第一，最深那一層一定被最罕見的兩個符號佔著，否則把深處的常用符號和淺處的罕用符號對調就會更省，所以現在就把它們配成兄弟不會有損失；第二，合併出來的子樹對外只剩一個重量，問題於是縮小成同一題再做一次。還有一個很好用的記帳方式：每合併一次，底下所有字元都多揹一個 bit，所以總 bits 剛好等於每次合併後新節點重量的總和。這也直接說明了為什麼重的子樹要盡量晚合併。',
            en:'Huffman is one of the few greedy algorithms that comes with a full optimality proof. Put each symbol on the leaf of a binary tree, where the codeword length is the depth, and the quantity to minimise is the sum of depth times frequency. Two observations carry it. First, the deepest level must be occupied by the two rarest symbols, since otherwise swapping a frequent deep symbol with a rare shallow one strictly improves the total - so pairing them now costs nothing. Second, a merged subtree presents itself to the world as a single weight, which shrinks the problem into another copy of itself. There is also a lovely way to keep the books: each merge adds one bit to every character beneath it, so the total equals the sum of the weights created by the merges - which says immediately that heavy subtrees should be merged as late as possible.'},
      legend:[['#ff9736', {zh:'被取出的兩棵最輕子樹', en:'the two lightest, just popped'}],
              ['#9d6bff', {zh:'合併後放回 queue 的新節點', en:'the merged node, pushed back'}],
              ['#3fe0dd', {zh:'完成的碼表', en:'the finished codebook'}],
              ['#0a6b74', {zh:'還在 queue 裡', en:'still queued'}]],
      code:CODE_HUF, build:hufFrames
    },
    {
      id:'ladder', label:{zh:'忘了重新排序', en:'the missing re-sort'},
      stage:{zh:'合併後沒有沉回 queue，樹就塌成一條梯子',
             en:'a merged node that never sinks back leaves the tree a ladder'},
      view:VIEW,
      variants:[{zh:'一路往右併', en:'merge left to right'},
                {zh:'兩棵樹並排比', en:'the two trees side by side'}],
      idea:{zh:'今天的無聲錯誤：先照次數排好一次，然後從最輕的往右一路合併，中間不再重新排序。少掉的只有一行，而它正好是演算法的核心——合併出來的節點已經變重了，應該沉回 queue 讓還沒配對的輕節點先出場，這裡卻永遠留在手上，於是同一棵子樹被反覆往上疊，樹塌成一條梯子。結果是 ' + LBITS + ' bits，Huffman 是 ' + HBITS + '，而完全不壓縮的固定長度也不過 ' + FIXBITS + ' bits：壓縮省下來的東西幾乎全部被吃掉。最可怕的是它每一項檢查都過——prefix-free 成立、編碼解碼完全還原、Kraft 和剛好 1.000。它不是錯的答案，是比較貴的答案，而唯一能發現它的方法是跟另一個實作比數字。',
            en:'Today’s silent failure: sort by frequency once, then merge from the lightest rightwards and never re-sort. Exactly one line is missing, and it happens to be the algorithm itself - a merged node has grown heavier and should sink back so that the still-unpaired light nodes go first, but here it stays in hand, the same subtree is stacked again and again, and the tree collapses into a ladder. The result is ' + LBITS + ' bits where Huffman needs ' + HBITS + ', and where a plain fixed-length code with no compression at all costs ' + FIXBITS + ': nearly the whole saving is gone. The alarming part is that every check passes - the code is prefix-free, encoding and decoding round-trip exactly, and the Kraft sum is precisely 1.000. It is not a wrong answer but an expensive one, and the only way to notice is to compare the number against another implementation.',
      },
      legend:[['#ff5c5c', {zh:'沒沉回去的節點 / 壞掉的樹', en:'the node that never sank / the broken tree'}],
              ['#ff9736', {zh:'目前這一步', en:'the current step'}],
              ['#3fe0dd', {zh:'正確的 Huffman 樹', en:'the correct Huffman tree'}],
              ['#2f5661', {zh:'已經併掉', en:'already merged'}]],
      code:CODE_LAD, build:ladFrames
    },
    {
      id:'lc435', label:{zh:'LC 435 不重疊區間', en:'LC 435 non-overlapping intervals'},
      stage:{zh:'「最少刪幾個」＝「最多留幾個」，同一個排序鍵，換一個 return',
             en:'fewest deletions equals most kept - same sort key, different return'},
      view:VIEW,
      variants:[{zh:'LeetCode 範例', en:'the LeetCode example'},
                {zh:'同一間會議室', en:'the meeting room'}],
      idea:{zh:'LeetCode 435 問最少要刪掉幾個區間才讓剩下的互不重疊。刪掉的數量等於總數減去留下的數量，所以真正要算的是「最多能留幾個互不重疊的」，那就是第一個分頁的排程問題，一個字都不用改。按結束時間排之後，判斷式塌成 s >= last_end，因為已留區間裡結束最晚的必定是最近收下的那一個，一個變數就記完了全部歷史。LeetCode 452 射氣球也是同一張皮，只差判斷式從 >= 變成 >：端點相接在那題算射中、在這題算不重疊，整個差別是一個等號。',
            en:'LeetCode 435 asks for the fewest intervals to delete so that the rest do not overlap. Deletions equal the total minus what is kept, so the quantity to compute is the largest non-overlapping set that can be kept - which is the scheduling problem from the first tab, unchanged. Once the list is ordered by end point the test collapses to s >= last_end, because the latest finisher among the kept intervals is always the most recent pick, and one variable therefore records the entire history. LeetCode 452, bursting balloons, wears the same skin with >= becoming >: touching endpoints count as a hit there and as non-overlapping here, and that equals sign is the whole difference.'},
      legend:[['#ff9736', {zh:'正在檢查的區間', en:'the interval being tested'}],
              ['#3fe0dd', {zh:'留下', en:'kept'}],
              ['#ff5c5c', {zh:'刪掉', en:'deleted'}],
              ['#9d6bff', {zh:'last_end', en:'last_end'}]],
      code:CODE_435, build:ivFrames
    },
    {
      id:'lc134', label:{zh:'LC 134 加油站', en:'LC 134 gas station'},
      stage:{zh:'斷油的時候，丟掉的不是一站，是一整段候選起點',
             en:'when the tank runs dry it is not one station that is discarded but a whole stretch'},
      view:VIEW,
      variants:[{zh:'油夠：答案 3', en:'enough fuel - answer 3'},
                {zh:'油不夠：−1', en:'not enough fuel - -1'}],
      idea:{zh:'加油站這題示範了貪心證明的另一種形狀。先比總和：總油量小於總花費就直接 −1，而只要總油量夠就一定存在可行起點，剩下的工作是指出它，不是判斷有沒有。接著一次掃描，油箱一旦變負就把起點跳到下一站——關鍵在於被跳過的那一整段也全部可以排除：從舊起點開過來時，中間每一站的油箱都還是非負的，所以改從中間出發，到達斷油那一站時手上的油只會更少。今天三個例子剛好是三種證法：區間排程靠「換進去不會變差」，Huffman 靠「最輕的兩個注定是兄弟」，這題靠「失敗的那一段整段不用再試」。貪心的難處永遠在這種句子，不在程式碼。',
            en:'The gas station problem shows a second shape of greedy proof. Compare the totals first: if the fuel available is below the fuel burned the answer is -1, and whenever the total is sufficient a valid start is guaranteed to exist, so the remaining job is to point at it rather than to decide whether it exists. Then one pass: the moment the tank goes negative the start jumps to the next station - and crucially the entire skipped stretch is eliminated with it, because driving from the old start the tank was non-negative at every stop, so starting from one of those stops instead arrives at the dry station with strictly less fuel. The three examples today are three different proofs: interval scheduling argues that an exchange never hurts, Huffman that the two lightest are destined to be siblings, and this one that a failed stretch can be discarded wholesale. The difficulty of a greedy algorithm always lives in a sentence like that, never in the code.'},
      legend:[['#ff9736', {zh:'目前這一站', en:'the station being driven'}],
              ['#9d6bff', {zh:'從目前起點開過的路段', en:'the stretch driven from the current start'}],
              ['#ff5c5c', {zh:'斷油：整段一起淘汰', en:'dry tank - the whole stretch is out'}],
              ['#3fe0dd', {zh:'可行的起點', en:'a start that completes the loop'}]],
      code:CODE_134, build:gasFrames
    }
  ]
};
