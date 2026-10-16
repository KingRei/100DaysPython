// DAY: 43
// TITLE_ZH: Python 的並行與平行：thread、process、asyncio 怎麼選，怎麼抓 race 與 deadlock，以及用 work 與 depth 量出能平行多少
// TITLE_EN: Concurrency and parallelism in Python: threads vs processes vs asyncio, catching races and deadlocks, and measuring parallelism with work and depth
// SUB_ZH: 在 CPython 裡，thread 能重疊的是「等待」不是「計算」：兩個吃 CPU 的工作給兩個 thread 跑跟一個接一個一樣慢，給兩個 process 才會快；二十個網路等待給二十個 thread 幾乎同時結束。並行的 bug 不會當掉：x += 1 會掉更新、五位哲學家在兩千萬分之一的交錯裡卡死、map-reduce 把同一個字分給兩個 reducer。把每一種交錯都數一遍，再用 work 與 depth 量出一個演算法最多能平行多少。
// SUB_EN: In CPython, threads overlap waiting, not computing: two CPU-heavy jobs on two threads are as slow as running them one after the other, on two processes they are faster, and twenty network waits on twenty threads finish almost together. Concurrency bugs do not crash: x += 1 loses updates, five philosophers freeze in one interleaving out of twenty million, map-reduce hands one word to two reducers. Count every interleaving, then measure with work and depth how parallel an algorithm can possibly be.
// FOLDER: day%2043%20-%20concurrency%20and%20parallelism
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
const fmt = n => { const s = String(n), neg = s[0] === '-', d = neg ? s.slice(1) : s;
  return (neg ? '-' : '') + d.replace(/\B(?=(\d{3})+(?!\d))/g, ','); };
function caps(sh, o){
  if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
  if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
  return sh;
}
const side = (sh, x, y0, lines, dy) => (lines || []).forEach((l, k) =>
  sh.push(fitT(x, y0 + k * (dy || .5), l[0], {anchor:'start', c:l[1] || COL.pale, fs:l[2] || .30})));

/* ================================================================ the interleaving checker
   A port of concurrency.py: each thread is a list of atomic steps
   ['read', var, reg] ['write', var, reg, add] ['acquire', L] ['release', L]
   ['wait', ev] ['set', ev] ['emit', text].  A state is plain data, so it can be a map key. */
const DEADLOCK = 'DEADLOCK';
function ckStart(threads, shared){
  return {pc:threads.map(() => 0), mem:Object.assign({}, shared), regs:threads.map(() => ({})),
          held:{}, ev:{}, out:''};
}
function ckEnabled(threads, st, i){
  if (st.pc[i] >= threads[i].length) return false;
  const op = threads[i][st.pc[i]];
  if (op[0] === 'acquire') return !(op[1] in st.held);
  if (op[0] === 'wait') return !!st.ev[op[1]];
  return true;
}
function ckStep(threads, st, i){
  const op = threads[i][st.pc[i]];
  const n = {pc:st.pc.slice(), mem:Object.assign({}, st.mem), regs:st.regs.map(r => Object.assign({}, r)),
             held:Object.assign({}, st.held), ev:Object.assign({}, st.ev), out:st.out};
  if (op[0] === 'read') n.regs[i][op[2]] = n.mem[op[1]];
  else if (op[0] === 'write') n.mem[op[1]] = n.regs[i][op[2]] + op[3];
  else if (op[0] === 'acquire') n.held[op[1]] = i;
  else if (op[0] === 'release') delete n.held[op[1]];
  else if (op[0] === 'set') n.ev[op[1]] = true;
  else if (op[0] === 'emit') n.out += op[1];
  n.pc[i]++;
  return n;
}
const ckKey = st => JSON.stringify([st.pc, Object.keys(st.mem).sort().map(k => [k, st.mem[k]]),
  st.regs.map(r => Object.keys(r).sort().map(k => [k, r[k]])), Object.keys(st.held).sort().map(k => [k, st.held[k]]),
  Object.keys(st.ev).sort(), st.out]);
function ckOutcome(threads, st){
  return threads.every((t, i) => st.pc[i] === t.length) ? JSON.stringify(st.mem) + '|' + st.out : DEADLOCK;
}
/* every interleaving, small cases: [{sched, final state, outcome}] */
function everySchedule(threads, shared){
  const res = [];
  (function go(st, sched){
    const nxt = threads.map((_, i) => i).filter(i => ckEnabled(threads, st, i));
    if (!nxt.length){ res.push({sched:sched.slice(), st:st, out:ckOutcome(threads, st)}); return; }
    nxt.forEach(i => { sched.push(i); go(ckStep(threads, st, i), sched); sched.pop(); });
  })(ckStart(threads, shared), []);
  return res;
}
/* outcome -> number of interleavings, memoised on the state */
function countSchedules(threads, shared){
  const memo = new Map();
  function go(st){
    const k = ckKey(st);
    if (memo.has(k)) return memo.get(k);
    const nxt = threads.map((_, i) => i).filter(i => ckEnabled(threads, st, i));
    let r;
    if (!nxt.length) r = {[ckOutcome(threads, st)]:1};
    else {
      r = {};
      nxt.forEach(i => { const sub = go(ckStep(threads, st, i));
        for (const o in sub) r[o] = (r[o] || 0) + sub[o]; });
    }
    memo.set(k, r);
    return r;
  }
  const r = go(ckStart(threads, shared));
  return {outcomes:r, states:memo.size, dead:Object.keys(r).includes(DEADLOCK) ? r[DEADLOCK] : 0};
}
function reachableStates(threads, shared){
  const seen = new Set(); let dead = 0;
  const stack = [ckStart(threads, shared)];
  while (stack.length){
    const st = stack.pop(), k = ckKey(st);
    if (seen.has(k)) continue;
    seen.add(k);
    const nxt = threads.map((_, i) => i).filter(i => ckEnabled(threads, st, i));
    if (!nxt.length && ckOutcome(threads, st) === DEADLOCK) dead++;
    nxt.forEach(i => stack.push(ckStep(threads, st, i)));
  }
  return [seen.size, dead];
}
const incrementer = (locked, times) => {
  let b = [['read', 'x', 'r'], ['write', 'x', 'r', 1]];
  if (locked) b = [['acquire', 'L']].concat(b, [['release', 'L']]);
  let out = []; for (let t = 0; t < (times || 1); t++) out = out.concat(b);
  return out;
};
const finalX = out => out === DEADLOCK ? DEADLOCK : JSON.parse(out.split('|')[0]).x;
function philosophers(n, ordered){
  const ts = [];
  for (let i = 0; i < n; i++){
    let a = i, b = (i + 1) % n;
    if (ordered && a > b){ const t = a; a = b; b = t; }
    ts.push([['acquire', 'f' + a], ['acquire', 'f' + b], ['emit', String(i)], ['release', 'f' + b], ['release', 'f' + a]]);
  }
  return ts;
}

/* ================================================================ tab 1: the GIL */
const CODE_GIL = [
  'def burn(n):                       # pure-Python CPU work',
  '    s = 0',
  '    for i in range(n):',
  '        s += i * i',
  '    return s',
  '',
  '# 2 threads: one GIL, the jobs take turns',
  'run_threads(burn, [(n,), (n,)])',
  '# 2 processes: two interpreters, two GILs',
  'pool.map(burn, [n, n])',
  '# 20 threads that mostly wait',
  'run_threads(time.sleep, [(0.05,)] * 20)'
];
/* jobs: list of segments [['cpu', k], ['wait', k], ...]; mode 'threads' shares one GIL */
function simJobs(jobs, mode){
  const segIx = jobs.map(() => 0), left = jobs.map(j => j[0][1]);
  const lanes = jobs.map(() => []);           // per tick: 'run' | 'gil' | 'io' | null
  let last = -1, t = 0, gilLog = [];
  const done = i => segIx[i] >= jobs[i].length;
  while (jobs.some((_, i) => !done(i)) && t < 60){
    const wantCpu = jobs.map((j, i) => !done(i) && j[segIx[i]][0] === 'cpu');
    let runners = [];
    if (mode === 'procs') runners = wantCpu.map((w, i) => w ? i : -1).filter(i => i >= 0);
    else {
      const cand = wantCpu.map((w, i) => w ? i : -1).filter(i => i >= 0);
      if (cand.length){ const other = cand.filter(i => i !== last); runners = [other.length ? other[0] : cand[0]]; }
    }
    gilLog.push(mode === 'procs' ? runners.slice() : (runners.length ? runners[0] : null));
    jobs.forEach((j, i) => {
      if (done(i)){ lanes[i].push(null); return; }
      const kind = j[segIx[i]][0];
      if (kind === 'cpu') lanes[i].push(runners.includes(i) ? 'run' : 'gil');
      else lanes[i].push('io');
      if (kind === 'wait' || runners.includes(i)){
        if (--left[i] === 0){ segIx[i]++; if (!done(i)) left[i] = j[segIx[i]][1]; }
      }
    });
    if (mode !== 'procs' && runners.length) last = runners[0];
    t++;
  }
  return {lanes:lanes, gil:gilLog, ticks:t};
}
const GIL_JOBS = [
  {mode:'threads', jobs:[[['cpu', 6]], [['cpu', 6]]]},
  {mode:'procs',   jobs:[[['cpu', 6]], [['cpu', 6]]]},
  {mode:'threads', jobs:[[['cpu', 1], ['wait', 6], ['cpu', 1]], [['cpu', 1], ['wait', 6], ['cpu', 1]]]}
];
const TX0 = 1.7, TW = .52;
function gilStage(sim, upto, o){
  const sh = [], names = ['A', 'B'], ys = [1.15, 2.35];
  ys.forEach((y, i) => {
    sh.push(S.t(.25, y + .4, (o.procs ? 'process ' : 'thread ') + names[i], {anchor:'start', c:COL.purpleL, fs:.28}));
    for (let t = 0; t < sim.ticks; t++){
      const v = t < upto ? sim.lanes[i][t] : null, x = TX0 + t * TW;
      if (t >= upto){ sh.push(S.r(x, y, TW - .05, .6, 'ghost', '')); continue; }
      if (v === 'run') sh.push(S.r(x, y, TW - .05, .6, t === upto - 1 ? 'hot' : 'ok', ''));
      else if (v === 'gil') sh.push(S.r(x, y, TW - .05, .6, t === upto - 1 ? 'bad' : 'soft', '', {dash:'.08 .06'}));
      else if (v === 'io') sh.push(S.e(x + .03, y + .3, x + TW - .08, y + .3, {s:'act', w:.06, arrow:false, dash:'.1 .07'}));
      else sh.push(S.r(x, y, TW - .05, .6, 'done', ''));
    }
  });
  for (let t = 0; t <= sim.ticks; t += 2){
    sh.push(S.e(TX0 + t * TW, 3.2, TX0 + t * TW, 3.3, {s:'idle', w:.03, arrow:false}));
    sh.push(S.t(TX0 + t * TW, 3.62, String(t), {c:COL.grey, fs:.24}));
  }
  sh.push(S.e(TX0, 3.2, TX0 + sim.ticks * TW, 3.2, {s:'idle', w:.03, arrow:false}));
  sh.push(S.t(TX0 - .15, 3.62, {zh:'時間', en:'tick'}, {c:COL.grey, fs:.24, anchor:'end'}));
  side(sh, .5, 4.3, o.side || [], .45);
  return caps(sh, o);
}
function gilFrames(v){
  const F = new Frames(), cfg = GIL_JOBS[v], sim = simJobs(cfg.jobs, cfg.mode), procs = cfg.mode === 'procs';
  const serial = cfg.jobs.reduce((s, j) => s + j.reduce((a, g) => a + g[1], 0), 0);
  const lineOf = [7, 9, 11][v];
  F.push({shapes:gilStage(sim, 0, {procs:procs,
      cap:[{zh:'兩個工作，各要 6 格 CPU', en:'two jobs, 6 ticks of CPU each'},
           {zh:'兩個工作，各要 6 格 CPU，但各自在自己的 process', en:'two jobs, 6 ticks of CPU each, each in its own process'},
           {zh:'兩個工作：送出請求（1 格）、等回應（6 格）、處理回應（1 格）', en:'two jobs: send a request (1 tick), wait for the reply (6), handle it (1)'}][v],
      cap2:{zh:'一個接一個跑要 ' + serial + ' 格', en:'one after the other: ' + serial + ' ticks'}}),
    panels:[{lbl:'tick', chips:[chip('0', 'empty')]}], line:lineOf,
    msg:[{zh:'CPython 一次只讓一個 thread 執行 Python bytecode：要執行就得拿著 GIL（全域直譯器鎖），預設每 5 毫秒交接一次。這裡每一格代表一段時間片，兩個 thread 都只想算東西。',
          en:'CPython lets one thread at a time run Python bytecode: to run you must hold the GIL (global interpreter lock), handed over every 5 ms by default. Each cell here is one time slice, and both threads only want to compute.'},
         {zh:'每個 process 有自己的直譯器、自己的 GIL。兩個 process 彼此不搶鎖，作業系統把它們放在兩顆核心上。代價是啟動 worker 和把參數送過去（pickle）的時間，這裡先不畫。',
          en:'Each process has its own interpreter and its own GIL. Two processes never compete for a lock, and the operating system puts them on two cores. The price is starting workers and sending them the arguments (pickling), left out of the picture here.'},
         {zh:'等網路、等硬碟、time.sleep 的時候，thread 會先把 GIL 放掉。所以只要工作大部分時間在等，兩個 thread 的等待可以完全重疊。',
          en:'A thread that waits on the network, the disk or time.sleep releases the GIL first. So when a job mostly waits, the waits of two threads overlap completely.'}][v]});
  for (let t = 1; t <= sim.ticks; t++){
    const g = sim.gil[t - 1];
    const holder = procs ? 'A + B' : (g == null ? {zh:'沒人', en:'nobody'} : 'thread ' + 'AB'[g]);
    const st = ['A', 'B'].map((n, i) => sim.lanes[i][t - 1]);
    const word = s => s === 'run' ? {zh:'執行', en:'runs'} : s === 'gil' ? {zh:'等 GIL', en:'waits for the GIL'} : s === 'io' ? {zh:'等 I/O（不用 GIL）', en:'waits on I/O (no GIL)'} : {zh:'完成', en:'done'};
    const busy = st.filter(s => s === 'run').length;
    const lines = [[{zh:'第 ' + t + ' 格：A ' + word(st[0]).zh + '，B ' + word(st[1]).zh, en:'tick ' + t + ': A ' + word(st[0]).en + ', B ' + word(st[1]).en}, COL.orangeL],
                   [{zh:'用到的核心：' + busy + ' / 2', en:'cores in use: ' + busy + ' / 2'}, busy === 2 ? COL.tealL : COL.grey]];
    let msg;
    if (v === 0) msg = t === 1 ? {zh:'A 拿到 GIL 先跑；B 雖然也準備好了，只能等。第二顆核心就這樣閒著。', en:'A gets the GIL and runs; B is ready too but can only wait. The second core sits idle.'}
      : t === 2 ? {zh:'時間片到了，GIL 交給 B，換 A 等。兩個 thread 輪流用「同一顆」核心，所以總時間跟一個接一個跑一樣——還要加上交接的成本。', en:'The slice is up and the GIL passes to B; now A waits. The two threads take turns on one core, so the total equals running them one after the other - plus the cost of the hand-offs.'}
      : t === sim.ticks ? {zh:'共 ' + sim.ticks + ' 格，跟循序的 ' + serial + ' 格一樣。實際量：兩個 thread 0.244 秒、一個接一個 0.207 秒（一次量測）——thread 在純計算上不會贏。', en:sim.ticks + ' ticks, the same as the serial ' + serial + '. Measured once on a 2-core VM: 0.244 s on two threads, 0.207 s one after the other - threads never win on pure computation.'}
      : {zh:'GIL 持續在 A、B 之間交接；任何時刻都只有一個 thread 在執行 Python。', en:'The GIL keeps passing between A and B; at any moment exactly one thread runs Python.'};
    else if (v === 1) msg = t === 1 ? {zh:'兩個 process 同時開始，各自拿著自己的 GIL，兩顆核心都在算。', en:'Both processes start at once, each holding its own GIL; both cores compute.'}
      : t === sim.ticks ? {zh:sim.ticks + ' 格就完成，大約一半。實際量到的加速是 1.3 到 2 倍之間（扣掉啟動 worker 和傳參數的成本），比 thread 好，但不是免費的。', en:'Done in ' + sim.ticks + ' ticks, about half. Measured speed-ups were between 1.3x and 2x (minus starting workers and sending arguments) - better than threads, but not free.'}
      : {zh:'沒有鎖要搶：每個 process 一直跑到自己做完。', en:'No lock to fight over: each process runs until it is done.'};
    else msg = st.includes('io') && st.includes('run') ? {zh:'一個在等回應（虛線，已放掉 GIL），另一個拿到 GIL 送出自己的請求。等待和執行重疊了。', en:'One waits for its reply (dashed, GIL released) while the other takes the GIL and sends its request. The wait and the work overlap.'}
      : st[0] === 'io' && st[1] === 'io' ? {zh:'兩個都在等。GIL 沒人用，但這段時間對兩個工作都算數——等待在同一段時間裡發生。', en:'Both are waiting. Nobody holds the GIL, yet the time counts for both jobs - the waits happen in the same stretch of time.'}
      : t === sim.ticks ? {zh:sim.ticks + ' 格對上循序的 ' + serial + ' 格。實際量：20 次各 50 毫秒的等待，迴圈 1.004 秒、20 個 thread 0.058 秒（一次量測）。等待越長、thread 越有用。', en:sim.ticks + ' ticks against ' + serial + ' one after the other. Measured once: twenty 50 ms waits took 1.004 s in a loop and 0.058 s on twenty threads. The longer the waits, the more threads help.'}
      : {zh:'處理回應又需要 GIL；兩個 thread 輪流拿，但這一段很短。', en:'Handling the reply needs the GIL again; the threads take turns, but this part is short.'};
    F.push({shapes:gilStage(sim, t, {procs:procs, side:lines,
        cap:{zh:'GIL：' + (typeof holder === 'string' ? holder : holder.zh), en:'GIL: ' + (typeof holder === 'string' ? holder : holder.en)},
        cap2:{zh:'實心：在執行；虛框：等 GIL；虛線：等 I/O', en:'solid: running; dashed box: waiting for the GIL; dashed line: waiting on I/O'}}),
      panels:[{lbl:'tick', chips:[chip(String(t), 'hot')]},
              {lbl:'GIL', chips:procs ? [chip('A', 'ok'), chip('B', 'ok')] : [chip(g == null ? '-' : 'AB'[g], g == null ? 'empty' : 'hot')]},
              {lbl:'cores', chips:[chip(busy + ' / 2', busy === 2 ? 'ok' : 'dim')]}],
      line:t === 1 ? lineOf : (v === 2 ? 11 : 3), msg:msg});
  }
  return F.list;
}

/* ================================================================ tab 2: races */
const CODE_RACE = [
  'def bump_with_call(box):',
  '    v = box.v          # read',
  '    _noop()            # any call: logging, validation, ...',
  '    box.v = v + 1      # write the value read before the call',
  '',
  'def work():',
  '    for _ in range(each):',
  '        with lock:     # the fix: read + write become one step',
  '            bump(box)'
];
const OPNAME = {read:{zh:'讀', en:'read'}, write:{zh:'寫', en:'write'}, acquire:{zh:'拿 lock', en:'acquire'}, release:{zh:'放 lock', en:'release'}};
const opLab = op => op[0] === 'read' ? 'r = x' : op[0] === 'write' ? 'x = r + 1' : op[0] === 'acquire' ? 'lock' : 'unlock';
/* stage: thread A ops left, B ops right, memory in the middle, schedule row at the bottom */
function raceStage(threads, st, sched, k, o){
  const sh = [], cols = [.5, 6.6];
  threads.forEach((ops, i) => {
    sh.push(S.t(cols[i] + 1.3, .55, 'thread ' + 'AB'[i], {c:COL.purpleL, fs:.32}));
    ops.forEach((op, j) => {
      const doneOp = j < st.pc[i], cur = o.cur && o.cur[0] === i && o.cur[1] === j;
      const blocked = o.blocked === i && j === st.pc[i];
      sh.push(box(cols[i], .8 + j * .62, 2.6, .5, cur ? 'hot' : blocked ? 'bad' : doneOp ? 'ok' : 'idle', opLab(op), .28));
    });
    const r = st.regs[i].r;
    sh.push(S.t(cols[i] + 1.3, .9 + ops.length * .62 + .25, {zh:'r = ' + (r == null ? '?' : r), en:'r = ' + (r == null ? '?' : r)}, {c:COL.orangeL, fs:.3}));
  });
  sh.push(S.t(4.9, .85, {zh:'共用記憶體', en:'shared memory'}, {c:COL.grey, fs:.26}));
  sh.push(box(4.25, 1.05, 1.3, .8, o.xbad ? 'bad' : o.xok ? 'ok' : 'act', 'x = ' + st.mem.x, .34));
  if (st.held && 'L' in st.held) sh.push(box(4.25, 2.1, 1.3, .5, 'hot', 'L: ' + 'AB'[st.held.L], .26));
  if (sched){
    const w = Math.min(1.0, 8.6 / sched.length), x0 = 4.9 - sched.length * w / 2;
    sched.forEach((i, j) => {
      const opIx = sched.slice(0, j).filter(t => t === i).length, op = threads[i][opIx];
      sh.push(box(x0 + j * w, 4.35, w - .06, .5, j < k ? (j === k - 1 ? 'hot' : 'done') : 'ghost', 'AB'[i] + '.' + (op[0] === 'acquire' ? 'lock' : op[0] === 'release' ? 'unlock' : op[0]), .24));
    });
    sh.push(S.t(4.9, 4.2, {zh:'這一種交錯的順序', en:'the order of this interleaving'}, {c:COL.grey, fs:.22}));
  }
  return caps(sh, o);
}
function replay(threads, sched){
  const sts = [ckStart(threads, {x:0})];
  sched.forEach(i => sts.push(ckStep(threads, sts[sts.length - 1], i)));
  return sts;
}
function raceFrames(v){
  const F = new Frames();
  if (v === 0){
    const th = [incrementer(), incrementer()], all = everySchedule(th, {x:0});
    let lost = 0, okc = 0;
    F.push({shapes:raceStage(th, ckStart(th, {x:0}), null, 0, {cap:{zh:'x = 0；兩個 thread 各做一次 x += 1，答案應該是 2', en:'x = 0; two threads each do x += 1, so the answer should be 2'},
        cap2:{zh:'x += 1 其實是兩步：讀到暫存 r，再把 r + 1 寫回去', en:'x += 1 is really two steps: read into a register r, then write r + 1 back'}}),
      panels:[{lbl:'interleavings', chips:[chip(String(all.length), 'act')]}], line:0,
      msg:{zh:'檢查器把每個 thread 寫成一串不可分割的步驟，然後跑過「每一種」交錯：每個 thread 自己的步驟維持原順序，彼此之間隨便穿插。兩個 thread、各兩步，總共 ' + all.length + ' 種。',
           en:'The checker writes each thread as a list of atomic steps and runs every interleaving: each thread keeps its own order, everything else may interleave. Two threads of two steps each give ' + all.length + ' interleavings.'}});
    all.forEach((run, ri) => {
      const sts = replay(th, run.sched), x = sts[sts.length - 1].mem.x, bad = x !== 2;
      if (bad) lost++; else okc++;
      const stepwise = ri === 0 || ri === 1;
      if (stepwise){
        run.sched.forEach((i, j) => {
          const before = sts[j], opIx = before.pc[i], op = th[i][opIx];
          const why = op[0] === 'read'
            ? {zh:'thread ' + 'AB'[i] + ' 讀 x，得到 ' + sts[j + 1].regs[i].r + '，放進自己的 r。' + (ri === 1 && j === 1 ? '注意：A 還沒寫回去，B 讀到的是「舊的」0。' : ''),
               en:'Thread ' + 'AB'[i] + ' reads x and gets ' + sts[j + 1].regs[i].r + ' into its own r.' + (ri === 1 && j === 1 ? ' Note: A has not written yet, so B reads the stale 0.' : '')}
            : {zh:'thread ' + 'AB'[i] + ' 寫回 r + 1 = ' + sts[j + 1].mem.x + '。' + (ri === 1 && j === 3 ? 'B 寫的是它之前讀到的 0 加 1——A 的更新被蓋掉了。' : ''),
               en:'Thread ' + 'AB'[i] + ' writes r + 1 = ' + sts[j + 1].mem.x + '.' + (ri === 1 && j === 3 ? ' B writes the 0 it read earlier, plus 1 - A\'s update is overwritten.' : '')};
          const last = j === run.sched.length - 1;
          F.push({shapes:raceStage(th, sts[j + 1], run.sched, j + 1, {cur:[i, opIx], xbad:last && bad, xok:last && !bad,
              cap:{zh:'交錯 ' + (ri + 1) + ' / ' + all.length + '，第 ' + (j + 1) + ' 步', en:'interleaving ' + (ri + 1) + ' / ' + all.length + ', step ' + (j + 1)},
              cap2:last ? (bad ? {zh:'結果 x = ' + x + '：掉了一次更新', en:'result x = ' + x + ': one update lost'} : {zh:'結果 x = 2：剛好一個接一個跑', en:'result x = 2: the threads happened to run one after the other'}) : null}),
            panels:[{lbl:'correct', chips:[chip(String(okc - (last ? 0 : (bad ? 0 : 1))), 'ok')]}, {lbl:'lost', chips:[chip(String(lost - (last ? 0 : (bad ? 1 : 0))), 'bad')]}],
            line:op[0] === 'read' ? 1 : 3, msg:why});
        });
      } else {
        F.push({shapes:raceStage(th, sts[sts.length - 1], run.sched, run.sched.length, {xbad:bad, xok:!bad,
            cap:{zh:'交錯 ' + (ri + 1) + ' / ' + all.length + '：x = ' + x, en:'interleaving ' + (ri + 1) + ' / ' + all.length + ': x = ' + x},
            cap2:bad ? {zh:'兩次讀都在兩次寫之前 → 兩個 thread 都寫 1', en:'both reads before both writes -> both threads write 1'} : {zh:'一個 thread 讀和寫都做完，另一個才開始', en:'one thread finished reading and writing before the other started'}}),
          panels:[{lbl:'correct', chips:[chip(String(okc), 'ok')]}, {lbl:'lost', chips:[chip(String(lost), 'bad')]}], line:bad ? 3 : 1,
          msg:bad ? {zh:'又一種會掉更新的交錯。規則很簡單：只要兩次「讀」都發生在兩次「寫」之前，兩個 thread 就都從 0 出發，各寫一次 1。',
                     en:'Another interleaving that loses an update. The rule is simple: whenever both reads happen before both writes, both threads start from 0 and each writes 1.'}
                  : {zh:'正確的交錯：thread 剛好一個接一個跑完。這種交錯只佔少數，而且 thread 越多越少。',
                     en:'A correct interleaving: the threads happened to run one after the other. These are the minority, and the more threads, the fewer.'}});
      }
    });
    const rows = [2, 3, 4].map(k => { const c = countSchedules(Array(k).fill(0).map(() => incrementer()), {x:0}).outcomes;
      let tot = 0, good = 0; for (const o in c){ tot += c[o]; if (finalX(o) === k) good += c[o]; } return [k, tot, good]; });
    F.push({shapes:raceStage(th, ckStart(th, {x:0}), null, 0, {
        cap:{zh:all.length + ' 種裡 ' + lost + ' 種掉更新；正確的只有 thread 剛好依序跑完的那 ' + okc + ' 種', en:lost + ' of ' + all.length + ' lose an update; only the ' + okc + ' where the threads ran one after the other are right'},
        cap2:{zh:rows.map(r => r[0] + ' 個 thread：' + r[2] + ' / ' + fmt(r[1])).join('，'), en:rows.map(r => r[0] + ' threads: ' + r[2] + ' of ' + fmt(r[1])).join(', ')}}),
      panels:[{lbl:'correct', chips:[chip(String(okc), 'ok')]}, {lbl:'lost', chips:[chip(String(lost), 'bad')]}], line:7,
      msg:{zh:'正確的交錯數就是「把 thread 排順序」的方法數：2、6、24；全部交錯則是 6、90、' + fmt(rows[2][1]) + '。真的 thread 跑一次只會看到其中一種——這就是為什麼 race 在測試時常常不出現。在 CPython 3.10 裡，四個 thread 各做 200,000 次 box.v += 1 每次都剛好 800,000；但在讀和寫之間加一個函式呼叫，我跑的幾次裡有 29% 到 64% 的更新消失。下一個選項加上 lock。',
           en:'The correct interleavings are the ways to order the threads: 2, 6, 24; all interleavings number 6, 90 and ' + fmt(rows[2][1]) + '. A real run sees only one of them - which is why races hide in testing. On CPython 3.10, four threads doing box.v += 1 200,000 times each always gave exactly 800,000; with one function call between the read and the write, 29 to 64 percent of the updates vanished across my runs. The next option adds a lock.'}});
  } else {
    const th = [incrementer(true), incrementer(true)];
    /* round-robin: alternate whenever the other thread can move */
    const sched = []; let st = ckStart(th, {x:0}), last = 1, steps = [];
    while (true){
      const nxt = [0, 1].filter(i => ckEnabled(th, st, i));
      if (!nxt.length) break;
      const pick = nxt.includes(1 - last) ? 1 - last : nxt[0];
      const blockedOther = [0, 1].find(i => i !== pick && st.pc[i] < th[i].length && !ckEnabled(th, st, i));
      steps.push({i:pick, op:th[pick][st.pc[pick]], blocked:blockedOther});
      sched.push(pick); st = ckStep(th, st, pick); last = pick;
    }
    const sts = replay(th, sched), cnt = countSchedules(th, {x:0}).outcomes, total = Object.values(cnt).reduce((a, b) => a + b, 0);
    F.push({shapes:raceStage(th, sts[0], sched, 0, {cap:{zh:'同樣兩個 thread，讀和寫被 lock 包起來', en:'the same two threads, read and write wrapped in a lock'},
        cap2:{zh:'排程器盡量輪流讓兩個 thread 執行', en:'the scheduler alternates between the threads whenever it can'}}),
      panels:[{lbl:'lock', chips:[chip('free', 'ok')]}], line:7,
      msg:{zh:'with lock: 讓「拿 lock → 讀 → 寫 → 放 lock」變成一整段：另一個 thread 想拿同一把 lock 時只能等。排程器還是一有機會就切換 thread——看看它切過去之後會發生什麼。',
           en:'with lock: turns "acquire, read, write, release" into one block: a thread that wants the same lock must wait. The scheduler still switches threads whenever it can - watch what happens when it does.'}});
    steps.forEach((s, j) => {
      const opIx = sts[j].pc[s.i], last = j === steps.length - 1, blk = s.blocked;
      const msg = s.op[0] === 'acquire' ? {zh:'thread ' + 'AB'[s.i] + ' 拿到 lock。', en:'Thread ' + 'AB'[s.i] + ' takes the lock.'}
        : s.op[0] === 'release' ? {zh:'thread ' + 'AB'[s.i] + ' 放掉 lock；另一個 thread 現在才能拿。', en:'Thread ' + 'AB'[s.i] + ' releases the lock; only now can the other thread take it.'}
        : s.op[0] === 'read' ? {zh:'thread ' + 'AB'[s.i] + ' 讀 x = ' + sts[j + 1].regs[s.i].r + '。' + (blk != null ? 'thread ' + 'AB'[blk] + ' 想拿 lock 但被擋住（紅框），排程器只能讓 ' + 'AB'[s.i] + ' 繼續。' : ''),
                                 en:'Thread ' + 'AB'[s.i] + ' reads x = ' + sts[j + 1].regs[s.i].r + '.' + (blk != null ? ' Thread ' + 'AB'[blk] + ' wants the lock but is blocked (red), so the scheduler has to let ' + 'AB'[s.i] + ' go on.' : '')}
        : {zh:'thread ' + 'AB'[s.i] + ' 寫 x = ' + sts[j + 1].mem.x + '。沒有人能在它讀和寫之間插進來。', en:'Thread ' + 'AB'[s.i] + ' writes x = ' + sts[j + 1].mem.x + '. Nobody could get between its read and its write.'};
      F.push({shapes:raceStage(th, sts[j + 1], sched, j + 1, {cur:[s.i, opIx], blocked:blk, xok:last,
          cap:{zh:'第 ' + (j + 1) + ' 步', en:'step ' + (j + 1)},
          cap2:last ? {zh:'x = ' + sts[j + 1].mem.x + '，正確', en:'x = ' + sts[j + 1].mem.x + ', correct'} : null}),
        panels:[{lbl:'lock', chips:[chip('L' in sts[j + 1].held ? 'AB'[sts[j + 1].held.L] : 'free', 'L' in sts[j + 1].held ? 'hot' : 'ok')]}],
        line:s.op[0] === 'acquire' || s.op[0] === 'release' ? 7 : s.op[0] === 'read' ? 1 : 3, msg:msg});
    });
    const c3 = countSchedules(Array(3).fill(0).map(() => incrementer(true)), {x:0}).outcomes;
    const t3 = Object.values(c3).reduce((a, b) => a + b, 0);
    F.push({shapes:raceStage(th, sts[sts.length - 1], sched, sched.length, {xok:true,
        cap:{zh:'加 lock 後只剩 ' + total + ' 種交錯可能發生，全部得到 2', en:'with the lock only ' + total + ' interleavings can happen, and all give 2'},
        cap2:{zh:'3 個 thread：' + t3 + ' 種，全部得到 3', en:'3 threads: ' + t3 + ', all give 3'}}),
      panels:[{lbl:'possible', chips:[chip(String(total), 'ok')]}], line:7,
      msg:{zh:'lock 不是讓壞的交錯「變少」，而是讓它們「不可能發生」：檢查器數出來只剩 ' + total + ' 種（兩個 thread 的先後順序），3 個 thread 是 ' + t3 + ' 種，每一種都正確。真的 thread 版本：四個 thread、各 200,000 次、包在 with lock: 裡，每次都是 800,000。',
           en:'A lock does not make bad interleavings rarer; it makes them impossible: the checker counts only ' + total + ' (the two orders of the threads), ' + t3 + ' for three threads, and every one is correct. With real threads: four threads, 200,000 increments each inside with lock:, always 800,000.'}});
  }
  return F.list;
}

/* ================================================================ tab 3: dining philosophers */
const CODE_PHIL = [
  'class DiningPhilosophers:                 # LeetCode 1226',
  '    def __init__(self, n=5):',
  '        self.n = n',
  '        self.forks = [threading.Lock() for _ in range(n)]',
  '',
  '    def wantsToEat(self, philosopher, pickLeftFork, pickRightFork,',
  '                   eat, putLeftFork, putRightFork):',
  '        left, right = philosopher, (philosopher + 1) % self.n',
  '        lo, hi = min(left, right), max(left, right)   # the fix',
  '        with self.forks[lo], self.forks[hi]:',
  '            pickLeftFork(); pickRightFork()',
  '            eat()',
  '            putLeftFork(); putRightFork()'
];
const PCX = 2.75, PCY = 2.7, PR = 2.05, FR = 1.2;
const pAng = i => -Math.PI / 2 + 2 * Math.PI * i / 5;
const fAng = f => -Math.PI / 2 + 2 * Math.PI * (f - .5) / 5;      /* fork f sits between P(f-1) and P(f) */
const pXY = i => [PCX + PR * Math.cos(pAng(i)), PCY + PR * Math.sin(pAng(i))];
const fXY = f => [PCX + FR * Math.cos(fAng(f)), PCY + FR * Math.sin(fAng(f))];
function philStage(th, st, o){
  const sh = [];
  for (let f = 0; f < 5; f++){
    const [x, y] = fXY(f), h = st.held['f' + f];
    if (h != null){ const [px, py] = pXY(h); sh.push(S.e(px, py, x, y, {s:'ok', w:.11, arrow:false, pad:.3})); }
  }
  (o.waits || []).forEach(([i, f]) => { const [px, py] = pXY(i), [x, y] = fXY(f);
    sh.push(S.e(px, py, x, y, {s:'bad', w:.05, dash:'.12 .08', pad:.42})); });
  for (let f = 0; f < 5; f++){
    const [x, y] = fXY(f), h = st.held['f' + f];
    sh.push(S.c(x, y, .27, h != null ? 'ok' : 'soft', 'f' + f, {fs:.24}));
  }
  for (let i = 0; i < 5; i++){
    const [x, y] = pXY(i), pc = st.pc[i], ate = st.out.includes(String(i));
    const s = o.cur === i ? 'hot' : o.stuck && o.stuck.includes(i) ? 'bad' : ate && pc >= th[i].length ? 'done' : pc === 3 || (pc > 2 && !ate) ? 'act' : 'idle';
    sh.push(S.c(x, y, .42, s, 'P' + i, {fs:.3}));
  }
  side(sh, 5.45, .75, o.side, .5);
  return caps(sh, o);
}
function philFrames(v){
  const F = new Frames(), ordered = v === 1, th = philosophers(5, ordered);
  const firstOf = i => th[i][0][1], secondOf = i => th[i][1][1];
  const rs = reachableStates(th, {}), cnt = countSchedules(th, {});
  const tot = Object.values(cnt.outcomes).reduce((a, b) => a + b, 0), dead = cnt.outcomes[DEADLOCK] || 0;
  let st = ckStart(th, {});
  F.push({shapes:philStage(th, st, {side:[[{zh:'Pi：左 fi、右 f(i+1)', en:'Pi: left fi, right f(i+1)'}, COL.grey],
        [ordered ? {zh:'規則：先拿編號小的叉子', en:'rule: lower-numbered fork first'} : {zh:'規則：先拿左邊的叉子', en:'rule: left fork first'}, ordered ? COL.tealL : COL.orangeL],
        [{zh:'P4 先拿 ' + firstOf(4) + '，再拿 ' + secondOf(4), en:'P4 takes ' + firstOf(4) + ' first, then ' + secondOf(4)}, COL.purpleL]],
      cap:{zh:'五位哲學家、五支叉子：吃飯要同時拿到左右兩支', en:'five philosophers, five forks: eating needs both neighbouring forks'},
      cap2:{zh:'每位哲學家是一個 thread，每支叉子是一把 lock', en:'each philosopher is a thread, each fork a lock'}}),
    panels:[{lbl:'states', chips:[chip(fmt(rs[0]), 'act')]}, {lbl:'deadlocked', chips:[chip(String(rs[1]), rs[1] ? 'bad' : 'ok')]}], line:ordered ? 8 : 7,
    msg:ordered ? {zh:'修正版：每個人都先拿兩支裡「編號小」的那支。對 P0 到 P3 來說就是左邊；只有 P4 不同：它的兩支是 f4 和 f0，所以先拿 f0。排程器跟上一個選項一樣，依序輪流讓 P0、P1、… 往前走一步。',
                   en:'The fixed version: everybody first takes the lower-numbered of their two forks. For P0 to P3 that is the left one; only P4 differs: its forks are f4 and f0, so it reaches for f0 first. The scheduler is the same as in the previous option: P0, P1, ... each take one step in turn.'}
                : {zh:'LeetCode 1226。最直覺的寫法：先拿左邊、再拿右邊、吃、放下。排程器依序輪流讓 P0、P1、P2、… 各往前走一步——這正是讓它卡死的那種時機。',
                   en:'LeetCode 1226. The obvious code: take the left fork, then the right, eat, put both down. The scheduler lets P0, P1, P2, ... take one step each in turn - exactly the timing that freezes it.'}});
  let last = -1, guard = 0, noted = false;
  while (guard++ < 80){
    const nxt = [0, 1, 2, 3, 4].filter(i => ckEnabled(th, st, i));
    const waiting = [0, 1, 2, 3, 4].filter(i => st.pc[i] < th[i].length && !ckEnabled(th, st, i))
      .map(i => [i, +th[i][st.pc[i]][1].slice(1)]);
    if (!nxt.length){
      const stuck = [0, 1, 2, 3, 4].filter(i => st.pc[i] < th[i].length);
      if (stuck.length){
        F.push({shapes:philStage(th, st, {waits:waiting, stuck:stuck,
            side:[[{zh:'各拿一支、等下一支', en:'each holds one, waits for next'}, COL.red],
                  [{zh:'可到達的狀態 ' + fmt(rs[0]) + ' 個，卡死的 ' + rs[1] + ' 個', en:fmt(rs[0]) + ' reachable states, ' + rs[1] + ' stuck'}, COL.pale],
                  [{zh:fmt(tot) + ' 種交錯裡', en:'of ' + fmt(tot) + ' interleavings'}, COL.orangeL],
                  [{zh:fmt(dead) + ' 種走到這裡', en:fmt(dead) + ' end here'}, COL.orangeL],
                  [{zh:'約每 ' + fmt(Math.floor(tot / dead)) + ' 種一次', en:'1 in ' + fmt(Math.floor(tot / dead))}, COL.orangeL]],
            cap:{zh:'Deadlock：沒有人能動，也沒有人吃完', en:'deadlock: nobody can move, nobody has finished'},
            cap2:{zh:'紅色虛線：在等的叉子', en:'dashed red: the fork being waited for'}}),
          panels:[{lbl:'states', chips:[chip(fmt(rs[0]), 'act')]}, {lbl:'deadlocked', chips:[chip(String(rs[1]), 'bad')]}], line:7,
          msg:{zh:'五個人都拿著左邊、等右邊，而右邊正被下一個人拿著：一個環。檢查器走過全部 ' + fmt(rs[0]) + ' 個可到達的狀態，卡死的只有這 1 個；在 ' + fmt(tot) + ' 種交錯裡只有 ' + fmt(dead) + ' 種會走到這裡（5 個人拿左叉的先後順序，5! = 120）。兩千萬分之一：測試永遠碰不到，正式環境總有一天會碰到。',
               en:'All five hold their left fork and wait for the right one, which the next philosopher holds: a cycle. The checker walks all ' + fmt(rs[0]) + ' reachable states and only this 1 is stuck; of ' + fmt(tot) + ' interleavings only ' + fmt(dead) + ' get here (the orders in which the five take their left forks, 5! = 120). One in twenty million: tests never hit it, production eventually will.'}});
      }
      break;
    }
    let i = nxt.find(k => k > last); if (i == null) i = nxt[0];
    const op = th[i][st.pc[i]];
    st = ckStep(th, st, i); last = i;
    const w2 = [0, 1, 2, 3, 4].filter(k => st.pc[k] < th[k].length && !ckEnabled(th, st, k)).map(k => [k, +th[k][st.pc[k]][1].slice(1)]);
    let msg;
    if (op[0] === 'acquire'){
      const f = op[1], first = st.pc[i] === 1;
      msg = first ? {zh:'P' + i + ' 拿起第一支叉子 ' + f + '。', en:'P' + i + ' picks up its first fork, ' + f + '.'}
                  : {zh:'P' + i + ' 拿到第二支叉子 ' + f + '，兩支都到手，可以吃了。', en:'P' + i + ' gets its second fork, ' + f + ': both in hand, it can eat.'};
      if (first && v === 0 && i === 4) msg = {zh:'P4 拿起左邊的 f4。現在五支叉子都被拿走了，每個人手上剛好一支。', en:'P4 picks up its left fork f4. Now all five forks are taken, one in each hand.'};
    } else if (op[0] === 'emit') msg = {zh:'P' + i + ' 吃飯。', en:'P' + i + ' eats.'};
    else msg = {zh:'P' + i + ' 放下 ' + op[1] + '。' + (w2.some(([k, f]) => 'f' + f === op[1]) ? '正在等它的人下一輪就能拿到。' : ''),
                en:'P' + i + ' puts down ' + op[1] + '.' + (w2.some(([k, f]) => 'f' + f === op[1]) ? ' Whoever was waiting for it can take it next.' : '')};
    if (ordered && op[0] === 'acquire' && st.pc[i] === 1 && i === 4) msg = {zh:'P4 先拿 f0（它兩支裡編號小的）。', en:'P4 takes f0 first (the lower of its two).'};
    if (ordered && !noted && w2.some(([k]) => k === 4) && st.pc[4] === 0 && (noted = true))
      msg = {zh:msg.zh + 'P4 要先拿 f0，但 f0 在 P0 手上，所以 P4 什麼都沒拿就在等——它不會擋住 f4，P3 因此可以拿到 f4。這就是環被打斷的地方。',
             en:msg.en + ' P4 must take f0 first, but P0 has it, so P4 waits holding nothing - leaving f4 free for P3. That is where the cycle breaks.'};
    F.push({shapes:philStage(th, st, {cur:i, waits:w2,
        side:[[{zh:'吃過：' + (st.out.split('').join(', ') || '—'), en:'have eaten: ' + (st.out.split('').join(', ') || '-')}, COL.tealL],
              [{zh:'在等：' + (w2.map(([k, f]) => 'P' + k + '→f' + f).join(' ') || '—'), en:'wait: ' + (w2.map(([k, f]) => 'P' + k + '→f' + f).join(' ') || '-')}, w2.length ? COL.red : COL.grey]],
        cap:{zh:'P' + i + '：' + (op[0] === 'acquire' ? '拿 ' + op[1] : op[0] === 'release' ? '放下 ' + op[1] : '吃'), en:'P' + i + ': ' + (op[0] === 'acquire' ? 'take ' + op[1] : op[0] === 'release' ? 'put down ' + op[1] : 'eat')},
        cap2:{zh:'實線：拿著的叉子；紅色虛線：在等的叉子', en:'solid: forks held; dashed red: forks waited for'}}),
      panels:[{lbl:'held', chips:[0, 1, 2, 3, 4].map(f => chip('f' + f + ':' + (st.held['f' + f] == null ? '-' : 'P' + st.held['f' + f]), st.held['f' + f] == null ? 'dim' : 'ok'))},
              {lbl:'eaten', chips:[chip(String(st.out.length), 'ok')]}],
      line:op[0] === 'emit' ? 11 : op[0] === 'acquire' ? 9 : 12, msg:msg});
  }
  if (ordered){
    F.push({shapes:philStage(th, st, {side:[[{zh:'五個人都吃到了', en:'all five have eaten'}, COL.tealL],
          [{zh:'可到達的狀態 ' + fmt(rs[0]) + ' 個', en:fmt(rs[0]) + ' reachable states'}, COL.pale],
          [{zh:'卡死的 ' + rs[1] + ' 個', en:rs[1] + ' stuck'}, COL.tealL]],
        cap:{zh:'先拿編號小的叉子：環永遠閉不起來', en:'lower-numbered fork first: the cycle can never close'},
        cap2:{zh:'檢查器走過每一個狀態，沒有一個卡死', en:'the checker visits every state and none is stuck'}}),
      panels:[{lbl:'states', chips:[chip(fmt(rs[0]), 'act')]}, {lbl:'deadlocked', chips:[chip(String(rs[1]), 'ok')]}], line:8,
      msg:{zh:'為什麼一定有效：deadlock 需要一個環，每個人都在等下一個人手上的叉子。沿著環走一圈，叉子編號不可能一直變大，總有一個人是「拿著大的、等小的」——而「先拿小的」這條規則讓這種人不存在。檢查器確認：' + fmt(rs[0]) + ' 個狀態、0 個卡死。這就是一般的鎖順序規則：所有 lock 都照同一個全域順序拿。',
           en:'Why it always works: a deadlock needs a cycle of philosophers each waiting for the fork the next one holds. Going round the cycle, fork numbers cannot keep increasing, so somebody must hold a higher fork while waiting for a lower one - and "lower first" means nobody ever does. The checker confirms: ' + fmt(rs[0]) + ' states, 0 stuck. This is the general lock-ordering rule: take all locks in one global order.'}});
  }
  return F.list;
}

/* ================================================================ tab 4: asyncio */
const CODE_ASYNC = [
  'async def fetch_await(i, delay, log, t0):',
  '    log.append((i, "start", now()))',
  '    await asyncio.sleep(delay)   # hands control back to the loop',
  '    log.append((i, "end", now()))',
  '',
  'async def fetch_blocking(i, delay, log, t0):',
  '    log.append((i, "start", now()))',
  '    time.sleep(delay)            # blocks the only thread there is',
  '    log.append((i, "end", now()))',
  '',
  'sem = asyncio.Semaphore(limit)',
  'async def one(i):',
  '    async with sem:',
  '        return await fetch(i, delay, log, t0)',
  'await asyncio.gather(*(one(i) for i in range(10)))'
];
/* a tiny event loop in virtual time: 10 tasks, each 1 unit (0.1 s) of waiting */
function simLoop(blocking, limit){
  const N = 10, start = Array(N).fill(null), end = Array(N).fill(null), events = [];
  let t = 0, running = 0;
  const ready = [...Array(N).keys()];
  const sleeping = [];
  while (ready.length || sleeping.length){
    while (ready.length && running < (limit || N)){
      const i = ready.shift();
      start[i] = t; running++;
      if (blocking){ events.push({t:t, i:i, kind:'block'}); t += 1; end[i] = t; running--; }
      else { sleeping.push([t + 1, i]); events.push({t:t, i:i, kind:'await'}); }
    }
    if (!sleeping.length) continue;
    sleeping.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const wake = sleeping[0][0];
    t = wake;
    const woke = sleeping.filter(s => s[0] === wake).map(s => s[1]);
    sleeping.splice(0, woke.length);
    woke.forEach(i => { end[i] = t; running--; });
    events.push({t:t, kind:'wake', ids:woke});
  }
  return {start:start, end:end, events:events, total:t};
}
const AX0 = 1.6, AU = .74;
function asyncStage(sim, upto, o){
  const sh = [];
  for (let i = 0; i < 10; i++){
    const y = .45 + i * .38;
    sh.push(S.t(AX0 - .15, y + .24, 'task ' + i, {anchor:'end', c:COL.purpleL, fs:.22}));
    const s0 = sim.start[i], e0 = sim.end[i];
    if (s0 != null && s0 <= upto){
      const fin = e0 != null && e0 <= upto, e = fin ? e0 : upto;
      if (e > s0) sh.push(S.r(AX0 + s0 * AU, y, (e - s0) * AU - .04, .28, fin ? (o.blocking ? 'bad' : 'ok') : 'hot', ''));
      else sh.push(S.r(AX0 + s0 * AU, y, .1, .28, 'hot', ''));
    }
  }
  sh.push(S.e(AX0, 4.32, AX0 + 10 * AU, 4.32, {s:'idle', w:.03, arrow:false}));
  for (let k = 0; k <= 10; k += 2){
    sh.push(S.e(AX0 + k * AU, 4.32, AX0 + k * AU, 4.42, {s:'idle', w:.03, arrow:false}));
    sh.push(S.t(AX0 + k * AU, 4.72, (k / 10).toFixed(1) + ' s', {c:COL.grey, fs:.22}));
  }
  sh.push(S.e(AX0 + upto * AU, .3, AX0 + upto * AU, 4.32, {s:'act', w:.04, arrow:false, dash:'.1 .08'}));
  return caps(sh, o);
}
function asyncFrames(v){
  const F = new Frames(), blocking = v === 1, limit = v === 2 ? 3 : null, sim = simLoop(blocking, limit);
  const lineOf = blocking ? 7 : v === 2 ? 12 : 2;
  F.push({shapes:asyncStage(sim, 0, {blocking:blocking,
      cap:{zh:'十個 task，每個要等 0.1 秒（例如等網路回應）', en:'ten tasks, each waiting 0.1 s (say, for a network reply)'},
      cap2:[{zh:'等待用 await asyncio.sleep', en:'the wait is await asyncio.sleep'}, {zh:'等待用 time.sleep——它還是寫在 async def 裡', en:'the wait is time.sleep - still inside an async def'}, {zh:'await，但同時最多 3 個（Semaphore(3)）', en:'await, but at most 3 at a time (Semaphore(3))'}][v]}),
    panels:[{lbl:'loop', chips:[chip('t = 0.0 s', 'act')]}], line:lineOf,
    msg:[{zh:'asyncio 只有一個 thread。event loop 一次執行一個 task，每個 task 一路跑到 await 某個慢東西時，就把控制權交回 loop，loop 去跑下一個。所以十個等待可以同時進行。',
          en:'asyncio has one thread. The event loop runs one task at a time; each task runs until it awaits something slow, then hands control back and the loop runs the next one. So ten waits can be in flight at once.'},
         {zh:'同樣十個 task，但等待寫成 time.sleep。它不是 await：task 不會把控制權交回 loop，而是把整個 thread 卡住 0.1 秒。程式照樣跑完、照樣回傳十個結果，沒有任何警告。',
          en:'The same ten tasks, but the wait is time.sleep. That is not an await: the task does not hand control back; it blocks the whole thread for 0.1 s. The program still finishes and still returns ten results, with no warning at all.'},
         {zh:'有時候「同時太多」也是問題：對方伺服器限流、連線數有上限。asyncio.Semaphore(3) 讓同時在等的 task 最多 3 個，其餘排隊。',
          en:'Sometimes "too many at once" is the problem: the server rate-limits, connections are capped. asyncio.Semaphore(3) lets at most 3 tasks wait at once; the rest queue up.'}][v]});
  sim.events.forEach(ev => {
    const upto = ev.t, active = sim.start.filter((s, i) => s != null && s <= upto && (sim.end[i] == null || sim.end[i] > upto)).length;
    let msg, line = lineOf;
    if (ev.kind === 'await'){
      if (ev.i > 0 && ev.i < 9 && !(limit && ev.i % 3 === 0)) return;
      msg = ev.i === 0 ? {zh:'task 0 開始，記下 start，然後 await asyncio.sleep(0.1)：把控制權交回 loop。loop 立刻去跑 task 1——時間還沒前進。', en:'task 0 starts, logs its start, then awaits asyncio.sleep(0.1): control goes back to the loop, which runs task 1 straight away - no time has passed.'}
        : {zh:'task ' + ev.i + ' 在 t = ' + (upto / 10).toFixed(1) + ' 秒開始並 await。現在有 ' + active + ' 個 task 同時在等。', en:'task ' + ev.i + ' starts at t = ' + (upto / 10).toFixed(1) + ' s and awaits. ' + active + ' tasks are now waiting at the same time.'};
      line = 2;
    } else if (ev.kind === 'block'){
      if (ev.i > 1 && ev.i < 9) return;
      msg = ev.i === 0 ? {zh:'task 0 開始，呼叫 time.sleep(0.1)。這一行不會交出控制權：loop 什麼都不能做，task 1 到 9 全部乾等。', en:'task 0 starts and calls time.sleep(0.1). This line does not give control back: the loop can do nothing, tasks 1 to 9 just wait.'}
        : {zh:'task ' + ev.i + ' 要等 task ' + (ev.i - 1) + ' 的 time.sleep 結束才輪得到。每個 task 的開始時間往後推 0.1 秒：0.0、0.1、0.2…', en:'task ' + ev.i + ' only gets its turn after task ' + (ev.i - 1) + '\'s time.sleep ends. Each start slips by 0.1 s: 0.0, 0.1, 0.2, ...'};
      line = 7;
    } else {
      msg = limit ? {zh:'t = ' + (upto / 10).toFixed(1) + ' 秒：task ' + ev.ids.join('、') + ' 醒來、做完，把 semaphore 的位子讓出來，排隊的下一批才開始。', en:'t = ' + (upto / 10).toFixed(1) + ' s: tasks ' + ev.ids.join(', ') + ' wake, finish and free their semaphore slots, so the next batch can start.'}
                  : {zh:'t = 0.1 秒：十個 sleep 同時到期，loop 依序喚醒每個 task，它們記下 end 就結束。總共 0.1 秒——等待完全重疊了。', en:'t = 0.1 s: all ten sleeps expire together; the loop wakes each task, which logs its end and returns. 0.1 s in total - the waits overlapped completely.'};
      line = 3;
    }
    F.push({shapes:asyncStage(sim, upto, {blocking:blocking,
        cap:{zh:'t = ' + (upto / 10).toFixed(1) + ' 秒；同時在等的 task：' + active, en:'t = ' + (upto / 10).toFixed(1) + ' s; tasks waiting right now: ' + active},
        cap2:{zh:'橘色：還在等；' + (blocking ? '紅色：做完（但是一個一個做）' : '青色：做完'), en:'orange: still waiting; ' + (blocking ? 'red: done (one at a time)' : 'teal: done')}}),
      panels:[{lbl:'loop', chips:[chip('t = ' + (upto / 10).toFixed(1) + ' s', 'act')]}, {lbl:'waiting', chips:[chip(String(active), active > 1 ? 'ok' : 'hot')]}],
      line:line, msg:msg});
  });
  F.push({shapes:asyncStage(sim, sim.total, {blocking:blocking,
      cap:{zh:'總共 ' + (sim.total / 10).toFixed(1) + ' 秒', en:'total ' + (sim.total / 10).toFixed(1) + ' s'},
      cap2:{zh:'實際量：await 0.10 秒、time.sleep 1.00 秒、Semaphore(3) 0.40 秒（一次量測）', en:'measured once: await 0.10 s, time.sleep 1.00 s, Semaphore(3) 0.40 s'}}),
    panels:[{lbl:'total', chips:[chip((sim.total / 10).toFixed(1) + ' s', blocking ? 'bad' : 'ok')]}], line:lineOf,
    msg:[{zh:'0.1 秒就全部完成。asyncio 適合大量等待：一千個在等的 task 也只是一千個小物件，不是一千個 thread。',
          en:'All done in 0.1 s. asyncio is built for lots of waiting: a thousand waiting tasks are a thousand small objects, not a thousand threads.'},
         {zh:'1.0 秒，慢十倍，而且沒有任何錯誤訊息——這是 asyncio 最常見的無聲失敗。任何會卡住的呼叫都一樣：requests.get、吃 CPU 的迴圈、同步的資料庫 driver。改用可以 await 的版本，或用 await asyncio.to_thread(blocking_call) 把它丟到 thread 上。',
          en:'1.0 s, ten times slower, and no error message - the most common silent failure in asyncio. Any blocking call does it: requests.get, a CPU-heavy loop, a synchronous database driver. Use an awaitable version, or await asyncio.to_thread(blocking_call) to push it onto a thread.'},
         {zh:'10 個 task、一次 3 個：分 4 輪，0.4 秒。這是刻意設下的限制，跟上一個選項的意外排隊是同一件事的兩面。',
          en:'10 tasks, 3 at a time: 4 rounds, 0.4 s. A deliberate limit - the planned version of the accidental queue in the previous option.'}][v]});
  return F.list;
}

/* ================================================================ tab 5: LeetCode 1242 */
const CODE_CRAWL = [
  'class Solution:                                  # LeetCode 1242',
  '    def crawl(self, startUrl, htmlParser, workers=16):',
  '        hostname = lambda url: url.split("/")[2]',
  '        host, seen = hostname(startUrl), {startUrl}',
  '        frontier = [startUrl]',
  '        with ThreadPoolExecutor(workers) as pool:',
  '            while frontier:',
  '                nxt = []',
  '                for urls in pool.map(htmlParser.getUrls, frontier):',
  '                    for v in urls:',
  '                        if hostname(v) == host and v not in seen:',
  '                            seen.add(v)',
  '                            nxt.append(v)',
  '                frontier = nxt',
  '        return list(seen)'
];
const YH = 'http://news.yahoo.com';
const WEB = {
  [YH + '/']:[YH + '/world', YH + '/sport', YH + '/tech', 'http://news.google.com/'],
  [YH + '/world']:[YH + '/world/asia', YH + '/world/europe', YH + '/'],
  [YH + '/sport']:[YH + '/sport/f1', YH + '/sport/tennis', 'http://yahoo.com/sport'],
  [YH + '/tech']:[YH + '/tech/ai', YH + '/tech/chips', YH + '/world'],
  [YH + '/world/asia']:[YH + '/world/asia/japan', YH + '/world/asia/taiwan'],
  [YH + '/world/europe']:[YH + '/world/europe/france'],
  [YH + '/sport/f1']:[YH + '/sport/f1/monaco'],
  [YH + '/sport/tennis']:[],
  [YH + '/tech/ai']:[YH + '/tech/ai/llm', 'http://news.yahoo.com.evil.example/ai'],
  [YH + '/tech/chips']:[YH + '/tech/ai'],
  [YH + '/world/asia/japan']:[], [YH + '/world/asia/taiwan']:[YH + '/'],
  [YH + '/world/europe/france']:[], [YH + '/sport/f1/monaco']:[],
  [YH + '/tech/ai/llm']:[YH + '/tech/ai/llm/agents'], [YH + '/tech/ai/llm/agents']:[],
  'http://news.google.com/':['http://news.google.com/a'], 'http://yahoo.com/sport':[],
  'http://news.yahoo.com.evil.example/ai':[]
};
const hostname = u => u.split('/')[2];
const shortName = u => { const h = hostname(u);
  if (h !== 'news.yahoo.com') return h === 'news.google.com' ? 'google' : h === 'yahoo.com' ? 'yahoo.com' : 'evil.example';
  const p = u.slice(YH.length); return p === '/' ? '/' : p.split('/').pop(); };
function crawlLayout(){
  const wave = {[YH + '/']:1}, q = [YH + '/'], cols = {1:[YH + '/']}, ext = [];
  for (let k = 0; k < q.length; k++){
    const u = q[k];
    WEB[u].forEach(v => {
      if (hostname(v) !== 'news.yahoo.com'){ if (!ext.includes(v)) ext.push(v); return; }
      if (!(v in wave)){ wave[v] = wave[u] + 1; (cols[wave[v]] = cols[wave[v]] || []).push(v); q.push(v); }
    });
  }
  const pos = {};
  Object.keys(cols).forEach(c => cols[c].forEach((u, j) => { pos[u] = [.25 + (c - 1) * 1.55, .55 + j * .66]; }));
  ext.forEach((u, j) => { pos[u] = [7.95, .85 + j * .9]; });
  return {pos:pos, wave:wave, waves:Object.keys(cols).length, ext:ext};
}
function crawlStage(L, o){
  const sh = [], W = 1.35, H = .44;
  Object.keys(WEB).forEach(u => WEB[u].forEach(v => {
    if (!(u in L.pos) || !(v in L.pos)) return;
    if (hostname(v) === 'news.yahoo.com' && L.wave[v] !== L.wave[u] + 1) return;
    if (L.ext.includes(v) && !(o.edgeHot && o.edgeHot.includes(v)) && !(o.bad && o.bad.includes(v))) return;
    const [x1, y1] = L.pos[u], [x2, y2] = L.pos[v];
    sh.push(S.e(x1 + W, y1 + H / 2, x2, y2 + H / 2, {s:(o.edgeHot && o.edgeHot.includes(v)) ? 'hot' : 'idle', w:.025, arrow:false, o:.6}));
  }));
  Object.keys(L.pos).forEach(u => {
    const [x, y] = L.pos[u], ext = L.ext.includes(u);
    const s = o.hot && o.hot.includes(u) ? 'hot' : o.bad && o.bad.includes(u) ? 'bad' : o.done && o.done.includes(u) ? 'ok'
            : o.seen && o.seen.includes(u) ? 'act' : ext ? 'ghost' : 'idle';
    sh.push(box(x, y, ext ? 1.6 : W, H, s, shortName(u), .24));
  });
  for (let c = 1; c <= L.waves; c++) sh.push(S.t(.25 + (c - 1) * 1.55 + W / 2, .35, {zh:'第 ' + c + ' 波', en:'wave ' + c}, {c:COL.grey, fs:.22}));
  sh.push(S.t(8.75, .62, {zh:'別的 host', en:'other hosts'}, {c:COL.grey, fs:.22}));
  side(sh, .3, 4.75, o.side, .42);
  return caps(sh, o);
}
function crawlFrames(v){
  const F = new Frames(), L = crawlLayout(), start = YH + '/', MS = 50;
  if (v === 0){
    const seen = [start], stack = [start], done = [];
    F.push({shapes:crawlStage(L, {seen:seen, side:[[{zh:'每次 getUrls 要 50 毫秒', en:'every getUrls call takes 50 ms'}, COL.grey]],
        cap:{zh:'依序爬：一次抓一頁', en:'serial crawl: one page at a time'}, cap2:{zh:'news.yahoo.com 上 16 頁', en:'16 pages on news.yahoo.com'}}),
      panels:[{lbl:'fetches', chips:[chip('0', 'empty')]}], line:3,
      msg:{zh:'LeetCode 1242：從 startUrl 出發，找出所有走得到、而且主機名稱相同的網頁。getUrls 是一次很慢的網路抓取。先看最直接的做法：一個 stack、一次抓一頁。',
           en:'LeetCode 1242: from startUrl, find every reachable page with the same host name. getUrls is a slow network fetch. First the direct way: a stack, one page at a time.'}});
    while (stack.length){
      const u = stack.pop(), got = WEB[u];
      got.forEach(x => { if (hostname(x) === hostname(start) && !seen.includes(x)){ seen.push(x); stack.push(x); } });
      done.push(u);
      F.push({shapes:crawlStage(L, {hot:[u], done:done.slice(0, -1), seen:seen, edgeHot:got,
          side:[[{zh:'已抓 ' + done.length + ' 頁，時間 ' + done.length * MS + ' 毫秒', en:done.length + ' fetches, ' + done.length * MS + ' ms'}, COL.orangeL]],
          cap:{zh:'抓 ' + shortName(u), en:'fetch ' + shortName(u)}, cap2:{zh:'紫色：已發現、還沒抓', en:'purple: found, not fetched yet'}}),
        panels:[{lbl:'fetches', chips:[chip(String(done.length), 'hot')]}, {lbl:'time', chips:[chip(done.length * MS + ' ms', 'act')]}], line:8,
        msg:done.length === 1 ? {zh:'抓起點，得到 4 個連結；google 那個主機名稱不同，不跟進去。其餘 3 個推進 stack。', en:'Fetch the start page: 4 links; the google one has a different host name and is skipped. The other 3 go on the stack.'}
          : !stack.length ? {zh:done.length + ' 頁、' + done.length + ' 次抓取、' + done.length * MS + ' 毫秒（實際量 0.80 秒）。每一次抓取都在等網路，CPU 幾乎沒事做——這正是 thread 幫得上忙的工作。', en:done.length + ' pages, ' + done.length + ' fetches, ' + done.length * MS + ' ms (measured: 0.80 s). Every fetch waits on the network while the CPU has nothing to do - exactly the kind of job threads help with.'}
          : {zh:'一次只等一個網頁：時間 = 頁數 × 50 毫秒。', en:'Waiting for one page at a time: time = pages x 50 ms.'}});
    }
  } else if (v === 1){
    const seen = [start], done = []; let frontier = [start], waves = 0;
    F.push({shapes:crawlStage(L, {seen:seen, cap:{zh:'一波一波：目前這一波的頁面同時抓', en:'in waves: every page of the current wave is fetched at once'},
        cap2:{zh:'pool.map 把 getUrls 丟給 thread pool', en:'pool.map hands getUrls to a thread pool'}}),
      panels:[{lbl:'waves', chips:[chip('0', 'empty')]}], line:5,
      msg:{zh:'改成廣度優先、一波一波來：frontier 裡的每一頁同時丟給 thread pool 抓。等網路時 thread 會放掉 GIL，所以 16 個 thread 一起等完全沒問題。seen 只有主 thread 會改，所以不需要 lock。',
           en:'Go breadth-first, in waves: every page in the frontier goes to the thread pool at once. Threads release the GIL while waiting on the network, so sixteen of them waiting together is fine. Only the main thread touches seen, so it needs no lock.'}});
    while (frontier.length){
      waves++;
      const nxt = [];
      frontier.forEach(u => WEB[u].forEach(x => { if (hostname(x) === hostname(start) && !seen.includes(x)){ seen.push(x); nxt.push(x); } }));
      F.push({shapes:crawlStage(L, {hot:frontier, done:done.slice(), seen:seen,
          side:[[{zh:'第 ' + waves + ' 波：同時抓 ' + frontier.length + ' 頁，時間 ' + waves * MS + ' 毫秒', en:'wave ' + waves + ': ' + frontier.length + ' pages at once, ' + waves * MS + ' ms'}, COL.orangeL]],
          cap:{zh:'第 ' + waves + ' 波', en:'wave ' + waves}, cap2:{zh:'這一波找到 ' + nxt.length + ' 個新網頁', en:nxt.length + ' new pages found in this wave'}}),
        panels:[{lbl:'waves', chips:[chip(String(waves), 'hot')]}, {lbl:'time', chips:[chip(waves * MS + ' ms', 'act')]}], line:8,
        msg:waves === 1 ? {zh:'第一波只有起點一頁，沒辦法更平行——其他網址還不知道。', en:'Wave 1 is just the start page; nothing else can run yet - the other URLs are still unknown.'}
          : nxt.length === 0 ? {zh:'最後一波：agents 這一頁要等 llm 那頁讀完才知道它存在。不管有幾個 thread，這條 ' + waves + ' 頁長的連結鏈都得走 ' + waves + ' 波。', en:'The last wave: the agents page only becomes known once llm has been read. However many threads you have, this ' + waves + '-page chain of links needs ' + waves + ' waves.'}
          : {zh:'這一波 ' + frontier.length + ' 頁同時抓，花的時間跟抓 1 頁一樣：50 毫秒。', en:frontier.length + ' pages fetched at once take as long as one: 50 ms.'}});
      frontier.forEach(u => done.push(u));
      frontier = nxt;
    }
    F.push({shapes:crawlStage(L, {done:done, side:[[{zh:'依序 ' + done.length + ' × 50 = ' + done.length * MS + ' 毫秒；分波 ' + waves + ' × 50 = ' + waves * MS + ' 毫秒', en:'serial ' + done.length + ' x 50 = ' + done.length * MS + ' ms; waves ' + waves + ' x 50 = ' + waves * MS + ' ms'}, COL.tealL]],
        cap:{zh:'時間 = 波數 × 一次抓取，不是頁數 × 一次抓取', en:'time = waves x one fetch, not pages x one fetch'}, cap2:{zh:'實際量：依序 0.80 秒、thread 0.26 秒', en:'measured: 0.80 s serial, 0.26 s with threads'}}),
      panels:[{lbl:'waves', chips:[chip(String(waves), 'ok')]}], line:14,
      msg:{zh:done.length + ' 頁是 work、' + waves + ' 波是 depth：就算有無限多個 thread，最少也要 ' + waves + ' 次抓取的時間。這跟「work 與 depth」分頁是同一個概念，只是加法換成了網頁。',
           en:done.length + ' pages are the work and ' + waves + ' waves the depth: even with unlimited threads you need the time of ' + waves + ' fetches. It is the same idea as the work and depth tab, with web pages instead of additions.'}});
  } else {
    const evil = 'http://news.yahoo.com.evil.example/ai', host = hostname(start);
    F.push({shapes:crawlStage(L, {hot:[YH + '/tech/ai'], side:[['host = "' + host + '"', COL.purpleL],
          [{zh:'tech/ai 連到 ' + evil.slice(7), en:'tech/ai links to ' + evil.slice(7)}, COL.orangeL]],
        cap:{zh:'同一個主機名稱，要怎麼比？', en:'same host name - compared how?'}, cap2:{zh:'題目要的是主機名稱完全相同', en:'the task asks for an identical host name'}}),
      panels:[{lbl:'host', chips:[chip(host, 'act')]}], line:10,
      msg:{zh:'tech/ai 這一頁連到 http://news.yahoo.com.evil.example/ai。它的主機名稱是 news.yahoo.com.evil.example，跟起點不同，不該爬。', en:'The tech/ai page links to http://news.yahoo.com.evil.example/ai. Its host name is news.yahoo.com.evil.example, not the start host, so it must not be crawled.'}});
    F.push({shapes:crawlStage(L, {bad:[evil], side:[['"' + host + '" in url  ->  ' + (evil.includes(host) ? 'True' : 'False'), COL.red],
          [{zh:'多爬了 1 頁：17 頁', en:'one page too many: 17'}, COL.red]],
        cap:{zh:'用 host in url：子字串比對', en:'host in url: a substring test'}, cap2:{zh:'news.yahoo.com 剛好是那個網址的一部分', en:'news.yahoo.com happens to be part of that URL'}}),
      panels:[{lbl:'pages', chips:[chip('17', 'bad')]}], line:10,
      msg:{zh:'寫成 host in url 的爬蟲會跟進去：字串 news.yahoo.com 確實出現在那個網址裡。範例裡不一定有這種網址，所以範例全過，換一組測資才錯。startswith 也救不了：http://news.yahoo.com.evil.example 一樣以 http://news.yahoo.com 開頭。',
           en:'A crawler written with host in url follows it: the string news.yahoo.com does appear in that URL. The examples may not contain such a URL, so they all pass and a different test fails. startswith does not save you either: http://news.yahoo.com.evil.example starts with http://news.yahoo.com too.'}});
    F.push({shapes:crawlStage(L, {done:Object.keys(L.wave), side:[['hostname(url) == host  ->  ' + (hostname(evil) === host ? 'True' : 'False'), COL.tealL],
          [{zh:'16 頁，正確', en:'16 pages, correct'}, COL.tealL]],
        cap:{zh:'先切出主機名稱，再用 == 比', en:'cut out the host name, then compare with =='}, cap2:'hostname(u) = u.split("/")[2]'}),
      panels:[{lbl:'pages', chips:[chip('16', 'ok')]}], line:10,
      msg:{zh:'把 http:// 和下一個 / 之間那段切出來，用 == 比對。news.yahoo.com.evil.example 不等於 news.yahoo.com，答案是正確的 16 頁。',
           en:'Cut out the part between http:// and the next /, and compare it with ==. news.yahoo.com.evil.example is not news.yahoo.com, and the answer is the correct 16 pages.'}});
  }
  return F.list;
}

/* ================================================================ tab 6: work and depth */
const CODE_WD = [
  'def sum_tree(n):                      # pairwise: depth log2(n)',
  '    level = [g.inp(i) for i in range(n)]',
  '    while len(level) > 1:',
  '        level = [g.add(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]',
  '',
  'def scan_hillis_steele(n):            # round d: add the value 2^d to the left',
  '    d = 1',
  '    while d < n:',
  '        cur = [cur[i] if i < d else g.add(cur[i - d], cur[i]) for i in range(n)]',
  '        d *= 2',
  '',
  'def scan_blelloch(n):                 # up-sweep, then down-sweep',
  '    ...  cur[i] = g.add(cur[i - d], cur[i])          # up',
  '    ...  cur[i - d], cur[i] = cur[i], g.add(left, cur[i])   # down',
  '',
  'def schedule(self, p):                # greedy: up to p ready additions per tick',
  '    run = [heappop(ready) for _ in range(min(p, len(ready)))]'
];
/* Dag: deps[k] = [a, b] with a/b = {i:input} or {k:node}; col[k] = drawing column */
function Dag(n){ this.n = n; this.deps = []; this.level = []; this.col = []; }
Dag.prototype.add = function(a, b, col){
  const lv = 1 + Math.max(a.k != null ? this.level[a.k] : 0, b.k != null ? this.level[b.k] : 0);
  this.deps.push([a, b]); this.level.push(lv); this.col.push(col);
  return {k:this.deps.length - 1, col:col};
};
Dag.prototype.depth = function(){ return this.level.length ? Math.max.apply(null, this.level) : 0; };
Dag.prototype.evaluate = function(vals, outs){
  const v = [], get = x => x == null ? 0 : x.i != null ? vals[x.i] : v[x.k];
  this.deps.forEach(([a, b]) => v.push(get(a) + get(b)));
  return outs.map(get);
};
Dag.prototype.schedule = function(p){
  const W = this.deps.length, waiting = this.deps.map(d => d.filter(x => x.k != null).length), users = this.deps.map(() => []);
  this.deps.forEach((d, k) => d.forEach(x => { if (x.k != null) users[x.k].push(k); }));
  const heap = [];
  const push = x => { heap.push(x); let i = heap.length - 1; while (i > 0){ const q = (i - 1) >> 1; if (heap[q] <= heap[i]) break; [heap[q], heap[i]] = [heap[i], heap[q]]; i = q; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length){ heap[0] = last; let i = 0;
    for (;;){ const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l] < heap[m]) m = l; if (r < heap.length && heap[r] < heap[m]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  for (let k = 0; k < W; k++) if (!waiting[k]) push(k);
  const ticks = [];
  while (heap.length){
    const run = []; while (run.length < p && heap.length) run.push(pop());
    ticks.push(run);
    run.forEach(k => users[k].forEach(u => { if (--waiting[u] === 0) push(u); }));
  }
  return ticks;
};
const inp = i => ({i:i, col:i});
function sumChain(n){ const g = new Dag(n); let acc = inp(0); for (let i = 1; i < n; i++) acc = g.add(acc, inp(i), i); return [g, [acc]]; }
function sumTree(n){ const g = new Dag(n); let lv = [...Array(n).keys()].map(inp);
  while (lv.length > 1){ const nx = []; for (let i = 0; i + 1 < lv.length; i += 2) nx.push(g.add(lv[i], lv[i + 1], (lv[i].col + lv[i + 1].col) / 2));
    if (lv.length % 2) nx.push(lv[lv.length - 1]); lv = nx; }
  return [g, lv]; }
function scanHS(n){ const g = new Dag(n); let cur = [...Array(n).keys()].map(inp);
  for (let d = 1; d < n; d *= 2) cur = cur.map((c, i) => i < d ? c : g.add(cur[i - d], c, i));
  return [g, cur]; }
function scanBlelloch(n){ const g = new Dag(n); const cur = [...Array(n).keys()].map(inp);
  for (let d = 1; d < n; d *= 2) for (let i = 2 * d - 1; i < n; i += 2 * d) cur[i] = g.add(cur[i - d], cur[i], i);
  cur[n - 1] = null;
  for (let d = n >> 1; d >= 1; d >>= 1) for (let i = 2 * d - 1; i < n; i += 2 * d){
    const left = cur[i - d]; cur[i - d] = cur[i]; cur[i] = cur[i] == null ? left : g.add(left, cur[i], i); }
  return [g, cur]; }
const WD_BUILD = [sumChain, sumTree, scanHS, scanBlelloch];
const WD_NAME = [{zh:'加總：迴圈', en:'sum: loop'}, {zh:'加總：樹', en:'sum: tree'}, {zh:'前綴和：Hillis-Steele', en:'scan: Hillis-Steele'}, {zh:'前綴和：Blelloch', en:'scan: Blelloch'}];
const WX = i => 1.05 + i * 1.0;
function wdStage(g, outs, done, hot, o){
  const sh = [], D = g.depth(), dy = Math.min(.95, 3.7 / D);
  const node = k => [WX(g.col[k]), .75 + g.level[k] * dy];
  const at = x => x.i != null ? [WX(x.i), .75] : node(x.k);
  g.deps.forEach((d, k) => { const [x, y] = node(k);
    d.forEach(src => { const [sx, sy] = at(src); sh.push(S.e(sx, sy + (src.i != null ? .22 : .17), x, y - .17, {s:hot.includes(k) ? 'hot' : 'idle', w:.025, arrow:false, o:.55})); }); });
  for (let i = 0; i < g.n; i++) sh.push(box(WX(i) - .24, .53, .48, .44, 'soft', String(i + 1), .24));
  g.deps.forEach((_, k) => { const [x, y] = node(k);
    sh.push(S.c(x, y, .17, hot.includes(k) ? 'hot' : done.has(k) ? 'ok' : 'idle', '+', {fs:.2})); });
  sh.push(S.t(.25, .82, {zh:'輸入', en:'input'}, {c:COL.grey, fs:.2, anchor:'start'}));
  return caps(sh, o);
}
function wdFrames(v){
  const F = new Frames(), P = 4, [g, outs] = WD_BUILD[v](8), W = g.deps.length, D = g.depth();
  const ticks = g.schedule(P), vals = [1, 2, 3, 4, 5, 6, 7, 8], res = g.evaluate(vals, outs);
  const lo = Math.max(Math.ceil(W / P), D), hi = Math.ceil(W / P) + D;
  const big = WD_BUILD[v](1024)[0], bW = big.deps.length, bD = big.depth();
  const bt = [4, 1024].map(p => big.schedule(p).length);
  const lineOf = [3, 3, 8, 12][v];
  let cross = null;
  if (v >= 2){ const hs = scanHS(1024)[0], bl = scanBlelloch(1024)[0]; let lo2 = 1, hi2 = 1024;
    while (lo2 < hi2){ const m = (lo2 + hi2) >> 1; if (hs.schedule(m).length < bl.schedule(m).length) hi2 = m; else lo2 = m + 1; }
    cross = lo2; }
  F.push({shapes:wdStage(g, outs, new Set(), [], {cap:{zh:WD_NAME[v].zh + '：work ' + W + '（加法次數）、depth ' + D + '（最長的依賴鏈）', en:WD_NAME[v].en + ': work ' + W + ' (additions), depth ' + D + ' (longest chain)'},
      cap2:{zh:'每個 + 是一次加法，線代表「需要那個結果」', en:'each + is one addition; a line means "needs that result"'}}),
    panels:[{lbl:'W', chips:[chip(String(W), 'act')]}, {lbl:'D', chips:[chip(String(D), 'act')]}, {lbl:'p', chips:[chip(String(P), 'hot')]}], line:lineOf,
    msg:[{zh:'用迴圈加總：每一次加法都要等上一次的總和，所以 7 次加法排成一條直線。work 7、depth 7——不管有幾個 worker，都得一步一步來。',
          en:'Summing with a loop: every addition needs the previous total, so the 7 additions form one straight chain. Work 7, depth 7 - however many workers you have, it goes one step at a time.'},
         {zh:'兩兩相加成一棵樹：還是 7 次加法（work 一樣），但最長的鏈只有 3 層。同一層的加法彼此獨立，可以同時做。',
          en:'Adding in pairs as a tree: still 7 additions (same work), but the longest chain is only 3 levels. Additions on the same level are independent and can run at once.'},
         {zh:'Hillis-Steele 前綴和（Day 22 見過）：第 d 輪，每個位置都加上左邊 2^d 格的值。只要 3 輪（depth 3），但每輪幾乎每個位置都要加：work ' + W + '，比迴圈的 7 多了一倍多。',
          en:'Hillis-Steele prefix sums (met on Day 22): in round d every position adds the value 2^d places to its left. Only 3 rounds (depth 3), but nearly every position adds in each round: work ' + W + ', more than twice the loop\'s 7.'},
         {zh:'Blelloch 前綴和：先沿著樹往上算部分和，再把「左邊全部的和」往下傳。work ' + W + '（大約兩個迴圈），depth ' + D + '。這是 exclusive scan：第 i 格是前 i 個數的和。',
          en:'Blelloch prefix sums: build partial sums up a tree, then pass "the sum of everything to my left" back down. Work ' + W + ' (about two loops), depth ' + D + '. It is an exclusive scan: slot i holds the sum of the first i numbers.'}][v]});
  const done = new Set();
  ticks.forEach((run, t) => {
    run.forEach(k => done.add(k));
    const idle = P - run.length;
    F.push({shapes:wdStage(g, outs, done, run, {cap:{zh:'第 ' + (t + 1) + ' 步：' + run.length + ' 個 worker 在做加法' + (idle ? '，' + idle + ' 個閒著' : ''), en:'tick ' + (t + 1) + ': ' + run.length + ' worker' + (run.length > 1 ? 's' : '') + ' adding' + (idle ? ', ' + idle + ' idle' : '')},
        cap2:{zh:'已完成 ' + done.size + ' / ' + W + ' 次加法', en:done.size + ' / ' + W + ' additions done'}}),
      panels:[{lbl:'tick', chips:[chip(String(t + 1), 'hot')]}, {lbl:'running', chips:run.map(k => chip('+' + k, 'hot'))}, {lbl:'idle', chips:[chip(String(idle), idle ? 'dim' : 'ok')]}],
      line:15, msg:t === 0 ? {zh:'貪婪排程：每一步，從「輸入都已經算好」的加法裡挑最多 ' + P + ' 個來做。' + (run.length < P ? '現在能做的只有 ' + run.length + ' 個，其他 worker 只能閒著——depth 在這裡卡住你。' : '一開始就有足夠的加法可以讓 ' + P + ' 個 worker 都忙。'),
                                    en:'Greedy scheduling: at every tick, run up to ' + P + ' additions whose inputs are ready.' + (run.length < P ? ' Only ' + run.length + ' can run now, so the other workers idle - depth is what holds you back here.' : ' At the start there is enough to keep all ' + P + ' workers busy.')}
        : {zh:run.length < P ? '只有 ' + run.length + ' 個加法的輸入準備好了，' + idle + ' 個 worker 閒著。' : P + ' 個 worker 全忙：這一段是 work 在決定速度。', en:run.length < P ? 'Only ' + run.length + ' addition' + (run.length > 1 ? 's have' : ' has') + ' its inputs ready; ' + idle + ' worker' + (idle > 1 ? 's' : '') + ' idle.' : 'All ' + P + ' workers busy: here the work sets the pace.'}});
  });
  F.push({shapes:wdStage(g, outs, done, [], {cap:{zh:P + ' 個 worker 花 ' + ticks.length + ' 步；Brent：' + lo + ' ≤ T_4 ≤ ' + hi, en:P + ' workers take ' + ticks.length + ' ticks; Brent: ' + lo + ' <= T_4 <= ' + hi},
      cap2:{zh:'1024 個數：W ' + fmt(bW) + '、D ' + bD + '；4 個 worker ' + fmt(bt[0]) + ' 步，1024 個 ' + bt[1] + ' 步', en:'1024 numbers: W ' + fmt(bW) + ', D ' + bD + '; ' + fmt(bt[0]) + ' ticks on 4 workers, ' + bt[1] + ' on 1024'}}),
    panels:[{lbl:'result', chips:res.map(x => chip(String(x), 'ok'))}, {lbl:'T_4', chips:[chip(String(ticks.length), 'ok')]}], line:15,
    msg:[{zh:'迴圈：' + ticks.length + ' 步，跟一個 worker 一樣。1024 個數一樣要 1,023 步，加 worker 完全沒用。最長的鏈（depth）就是速度的下限：T_p ≥ D。',
          en:'Loop: ' + ticks.length + ' ticks, the same as one worker. 1024 numbers still take 1,023 ticks; extra workers do nothing. The longest chain (the depth) is a floor: T_p >= D.'},
         {zh:'樹：' + ticks.length + ' 步。1024 個數時 depth 只有 10：4 個 worker 要 ' + fmt(bt[0]) + ' 步（被 work / p 限制），1024 個 worker 只要 ' + bt[1] + ' 步。Brent 定理：max(W/p, D) ≤ T_p ≤ W/p + D。',
          en:'Tree: ' + ticks.length + ' ticks. With 1024 numbers the depth is just 10: ' + fmt(bt[0]) + ' ticks on 4 workers (limited by W/p), ' + bt[1] + ' on 1024. Brent: max(W/p, D) <= T_p <= W/p + D.'},
         {zh:'Hillis-Steele：' + ticks.length + ' 步。1024 個數時 work 是 ' + fmt(bW) + '：4 個 worker 要 ' + fmt(bt[0]) + ' 步，比普通迴圈的 1,023 步還慢一倍多——「平行演算法」輸給了循序的。只有 worker 跟資料一樣多時（GPU 一個 block 裡）它才划算：1024 個 worker 只要 ' + bt[1] + ' 步。',
          en:'Hillis-Steele: ' + ticks.length + ' ticks. With 1024 numbers the work is ' + fmt(bW) + ': ' + fmt(bt[0]) + ' ticks on 4 workers, more than twice the plain loop\'s 1,023 - the "parallel algorithm" loses to the serial one. It only pays when workers match the data, as inside one GPU block: ' + bt[1] + ' ticks on 1024 workers.'},
         {zh:'Blelloch：' + ticks.length + ' 步，結果 ' + res.join(', ') + '。1024 個數時 work ' + fmt(bW) + '：4 個 worker ' + fmt(bt[0]) + ' 步，1024 個 ' + bt[1] + ' 步。worker 少於 ' + cross + ' 個時它都贏 Hillis-Steele——這就是它成為預設平行 scan 的原因。',
          en:'Blelloch: ' + ticks.length + ' ticks, result ' + res.join(', ') + '. With 1024 numbers the work is ' + fmt(bW) + ': ' + fmt(bt[0]) + ' ticks on 4 workers, ' + bt[1] + ' on 1024. It beats Hillis-Steele on anything below ' + cross + ' workers - which is why it is the default parallel scan.'}][v]});
  return F.list;
}

/* ================================================================ tab 7: map-reduce */
const CODE_MR = [
  'def map_doc(doc, reducers=4, salt=b""):',
  '    counts = Counter(doc)                       # map + combine',
  '    parts = [Counter() for _ in range(reducers)]',
  '    for w, c in counts.items():',
  '        parts[stable_bucket(w, reducers, salt)][w] += c',
  '    return parts',
  '',
  'def reduce_parts(all_parts, reducers=4):',
  '    out = {}',
  '    for r in range(reducers):                   # shuffle: reducer r gets part r',
  '        total = Counter()',
  '        for parts in all_parts:',
  '            total.update(parts[r])',
  '        out.update(total)                       # a word in two reducers: last wins',
  '    return out',
  '',
  'def mean_by_sum_count(chunks):                 # reduce (sum, count), divide once',
  '    s, n = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]),',
  '                  ((sum(c), len(c)) for c in chunks))',
  '    return s / n'
];
const MR_DOCS = ['the cat sat on the mat', 'the dog sat', 'a cat and a dog', 'the end'];
/* the demo's own bucket function: FNV-1a over salt + word (Python uses blake2b) */
function fnv(s){ let h = 0x811c9dc5; for (const ch of s){ h ^= ch.charCodeAt(0); h = Math.imul(h, 0x01000193) >>> 0; } return h; }
const bucket = (w, salt) => fnv(salt + w) % 2;
function mrSalts(disagree){
  if (!disagree) return ['', ''];
  for (let k = 1; k < 50; k++){ const s = 'm' + k;
    if (bucket('the', s) !== bucket('the', '')) return ['', s]; }
  return ['', 'm1'];
}
function mrStage(o){
  const sh = [];
  MR_DOCS.forEach((d, i) => {
    sh.push(box(.2, .45 + i * .78, 2.6, .5, o.docHot === i ? 'hot' : o.docDone && o.docDone.includes(i) ? 'done' : 'idle', d, .22));
    sh.push(S.t(.2, .4 + i * .78, (i % 2 ? 'mapper 1' : 'mapper 0') + ', doc ' + i, {anchor:'start', c:i % 2 ? COL.orangeL : COL.purpleL, fs:.18}));
  });
  (o.mapLines || []).forEach((l, k) => sh.push(fitT(3.05, .75 + k * .4, l[0], {anchor:'start', c:l[1], fs:.24})));
  [0, 1].forEach(r => {
    const y = .45 + r * 1.9;
    sh.push(box(6.55, y, 3.0, 1.6, o.redHot === r ? 'hot' : 'act', '', .3));
    sh.push(S.t(8.05, y + .32, 'reducer ' + r, {c:COL.purpleL, fs:.26}));
    (o.red && o.red[r] || []).forEach((l, k) => sh.push(fitT(6.7, y + .7 + k * .3, l[0], {anchor:'start', c:l[1] || COL.pale, fs:.2})));
  });
  (o.outLines || []).forEach((l, k) => sh.push(fitT(.3, 3.75 + k * .42, l[0], {anchor:'start', c:l[1] || COL.pale, fs:.26})));
  return caps(sh, o);
}
function counter(words){ const c = {}; words.forEach(w => c[w] = (c[w] || 0) + 1); return c; }
function mrFrames(v){
  const F = new Frames();
  if (v < 2){
    const salts = mrSalts(v === 1), truth = counter(MR_DOCS.join(' ').split(' '));
    const parts = MR_DOCS.map((d, i) => { const c = counter(d.split(' ')), p = [{}, {}];
      Object.keys(c).forEach(w => { p[bucket(w, salts[i % 2])][w] = c[w]; }); return p; });
    F.push({shapes:mrStage({cap:{zh:'4 份文件、2 個 mapper（偶數／奇數文件）、2 個 reducer', en:'4 documents, 2 mappers (even / odd documents), 2 reducers'},
        cap2:v ? {zh:'這次 mapper 1 用不同的方式分桶（像 spawn 出來、hash 種子不同的 process）', en:'this time mapper 1 buckets differently (like a spawned process with another hash seed)'}
               : {zh:'每個字送到 reducer bucket(word)，兩個 mapper 用同一個 bucket 函式', en:'each word goes to reducer bucket(word); both mappers use the same bucket function'}}),
      panels:[{lbl:'reducers', chips:[chip('2', 'act')]}], line:0,
      msg:v ? {zh:'同樣的工作，但 mapper 1 的 bucket 函式跟 mapper 0 不一樣。真實世界裡這就是 hash(word) % reducers 搭配 spawn 出來的 worker：Python 對字串的 hash() 每個 process 都用不同的隨機種子（Day 27 MinHash 學過同一課）。',
               en:'The same job, but mapper 1 buckets differently from mapper 0. In real life this is hash(word) % reducers with spawned workers: Python seeds the string hash() differently in every process (the same lesson as Day 27\'s MinHash).'}
            : {zh:'map-reduce：map 階段每份文件各自數字（這一步彼此獨立，depth 1）；shuffle 把每個字送到第 bucket(word) 號 reducer；reduce 把收到的次數加起來。',
               en:'Map-reduce: map counts each document on its own (independent work, depth 1); shuffle sends every word to reducer number bucket(word); reduce adds up the counts it receives.'}});
    MR_DOCS.forEach((d, i) => {
      const lines = [[{zh:'doc ' + i + '（mapper ' + (i % 2) + '）', en:'doc ' + i + ' (mapper ' + (i % 2) + ')'}, i % 2 ? COL.orangeL : COL.purpleL]];
      [0, 1].forEach(r => { const ws = Object.keys(parts[i][r]); if (ws.length) lines.push([ws.map(w => w + ':' + parts[i][r][w]).join(' ') + '  -> R' + r, COL.tealL]); });
      F.push({shapes:mrStage({docHot:i, docDone:[...Array(i).keys()], mapLines:lines,
          cap:{zh:'map doc ' + i + '：先在文件內數好，再依 bucket 分成兩份', en:'map doc ' + i + ': count inside the document, then split by bucket'},
          cap2:{zh:'先在本地合併（combiner）能讓 shuffle 少送很多資料', en:'combining locally first means far less data to shuffle'}}),
        panels:[{lbl:'mapper', chips:[chip(String(i % 2), i % 2 ? 'hot' : 'act')]}], line:4,
        msg:i === 0 ? {zh:'doc 0 有兩個 the：mapper 先在本地數成 the:2，再送出去，而不是送兩次 the:1。每個字依 bucket(word) 決定要送去哪個 reducer。', en:'doc 0 has "the" twice: the mapper counts it locally as the:2 before sending, instead of sending the:1 twice. bucket(word) decides which reducer each word goes to.'}
          : (v && i % 2 && bucket('the', salts[1]) !== bucket('the', '') && parts[i][bucket('the', salts[1])].the)
            ? {zh:'注意 the：mapper 1 把它送到 R' + bucket('the', salts[1]) + '，但 mapper 0 把它送到 R' + bucket('the', '') + '。同一個字被拆到兩個 reducer。', en:'Watch "the": mapper 1 sends it to R' + bucket('the', salts[1]) + ', but mapper 0 sent it to R' + bucket('the', '') + '. One word is split across two reducers.'}
            : {zh:'每份文件的 map 互不相干，可以同時在不同機器上做。', en:'The map of each document is independent of the others and can run on different machines at once.'}});
    });
    const red = [0, 1].map(r => { const t = {}; parts.forEach(p => Object.keys(p[r]).forEach(w => t[w] = (t[w] || 0) + p[r][w])); return t; });
    const split = Object.keys(red[0]).filter(w => w in red[1]);
    const redLines = red.map((t, r) => { const ws = Object.keys(t).sort(); const lines = [];
      for (let k = 0; k < ws.length; k += 4){ const chunk = ws.slice(k, k + 4);
        lines.push([chunk.map(w => w + ':' + t[w]).join('  '), chunk.some(w => split.includes(w)) ? COL.red : COL.pale]); }
      return lines; });
    F.push({shapes:mrStage({docDone:[0, 1, 2, 3], red:redLines,
        cap:{zh:'shuffle + reduce：reducer r 收齊每個 mapper 的第 r 份，加總', en:'shuffle + reduce: reducer r collects part r from every mapper and adds up'},
        cap2:{zh:'兩個 reducer 可以同時做', en:'the two reducers run at the same time'}}),
      panels:[{lbl:'R0', chips:[chip(String(Object.keys(red[0]).length) + ' words', 'act')]}, {lbl:'R1', chips:[chip(String(Object.keys(red[1]).length) + ' words', 'act')]}], line:12,
      msg:{zh:'每個 reducer 把收到的同一個字的次數加起來。只要「同一個字一定在同一個 reducer」，每個 reducer 的答案就是那些字的完整次數。', en:'Each reducer adds up the counts it received for each word. As long as every copy of a word lands in the same reducer, each reducer\'s answer is complete for its words.'}});
    const out = {}; red.forEach(t => Object.assign(out, t));
    const wrong = Object.keys(truth).filter(w => out[w] !== truth[w]);
    const both = Object.keys(red[0]).filter(w => w in red[1]);
    F.push({shapes:mrStage({docDone:[0, 1, 2, 3], red:redLines,
        outLines:[[{zh:'輸出（R0 再 R1，依序寫入）：', en:'output (R0 then R1, written in order):'}, COL.grey],
                  [Object.keys(out).sort().map(w => w + ':' + out[w]).join('  '), wrong.length ? COL.red : COL.tealL],
                  [wrong.length ? {zh:'錯 ' + wrong.length + ' 個字：' + wrong.map(w => w + ' 應為 ' + truth[w]).join('、'), en:wrong.length + ' word' + (wrong.length > 1 ? 's' : '') + ' wrong: ' + wrong.map(w => w + ' should be ' + truth[w]).join(', ')}
                                : {zh:'跟直接用 Counter 數的結果完全相同', en:'identical to counting everything with Counter'}, wrong.length ? COL.red : COL.tealL]],
        cap:wrong.length ? {zh:'同一個字出現在兩個 reducer：後寫的覆蓋先寫的', en:'a word in two reducers: the later one overwrites the earlier'} : {zh:'結果正確', en:'the result is correct'},
        cap2:wrong.length ? {zh:'Python 版：67 個字錯 51 個，例如 of 只剩 821（應為 1,640）', en:'the Python run: 51 of 67 words wrong, e.g. "of" 821 instead of 1,640'} : {zh:'Python 版：40 份文件、16,000 個字，2 個 process、4 個 reducer，跟 Counter 一樣', en:'the Python run: 40 documents, 16,000 words, 2 processes, 4 reducers - same as Counter'}}),
      panels:[{lbl:'wrong', chips:[chip(String(wrong.length), wrong.length ? 'bad' : 'ok')]}], line:13,
      msg:wrong.length ? {zh:both.join('、') + ' 被兩個 reducer 各數了一部分，輸出時後寫的蓋掉先寫的，留下的只是一半的次數。沒有例外、沒有警告。在 Linux 上用 fork 產生的 worker 繼承同一個 hash 種子，所以永遠不會出事；在 Windows 和 macOS 上預設是 spawn，就會出事。修法：用不依賴 process 的雜湊，例如 hashlib.blake2b。',
                                en:both.join(', ') + ' was counted partly by each reducer, and when the outputs are written the later one overwrites the earlier, leaving only part of the count. No exception, no warning. Workers made with fork (classic Linux) inherit one hash seed, so it never happens there; spawn, the default on Windows and macOS, triggers it. The fix: a hash that does not depend on the process, such as hashlib.blake2b.'}
                       : {zh:'每個字只在一個 reducer 裡，所以把各 reducer 的輸出接起來就是答案。這需要每個 mapper 對同一個字給出同一個 bucket。', en:'Every word lives in exactly one reducer, so concatenating the reducer outputs is the answer. That requires every mapper to give a word the same bucket.'}});
  } else {
    const chunks = [[120, 80, 100], [400], [90, 110, 100, 95, 105, 100]];
    const means = chunks.map(c => c.reduce((a, b) => a + b, 0) / c.length);
    const mom = means.reduce((a, b) => a + b, 0) / means.length;
    const S1 = chunks.map(c => c.reduce((a, b) => a + b, 0)), N1 = chunks.map(c => c.length);
    const tot = S1.reduce((a, b) => a + b, 0), n = N1.reduce((a, b) => a + b, 0);
    const stage = (o) => { const sh = [];
      chunks.forEach((c, i) => {
        sh.push(S.t(.3, .55 + i * 1.05, {zh:'區塊 ' + i + '（' + c.length + ' 筆，毫秒）', en:'chunk ' + i + ' (' + c.length + (c.length === 1 ? ' value' : ' values') + ', ms)'}, {anchor:'start', c:COL.purpleL, fs:.22}));
        c.forEach((x, j) => sh.push(box(.3 + j * .78, .7 + i * 1.05, .72, .46, o.hot === i ? 'hot' : 'idle', String(x), .22)));
        if (o.show && o.show[i]) sh.push(fitT(5.2, 1.0 + i * 1.05, o.show[i][0], {anchor:'start', c:o.show[i][1], fs:.26}));
      });
      (o.lines || []).forEach((l, k) => sh.push(fitT(.3, 4.0 + k * .45, l[0], {anchor:'start', c:l[1], fs:.28})));
      return caps(sh, o); };
    F.push({shapes:stage({cap:{zh:'三個 worker 各自算一個區塊的平均延遲', en:'three workers each average one chunk of latencies'}, cap2:{zh:'區塊大小不一樣：3、1、6 筆', en:'the chunks have different sizes: 3, 1 and 6 values'}}),
      panels:[{lbl:'values', chips:[chip(String(n), 'act')]}], line:16,
      msg:{zh:'reduce 這一步要能拆給很多 worker，合併的運算必須滿足結合律，而且每一份要帶著合併需要的全部資訊。平均數兩個條件都不滿足——看看會發生什麼。', en:'For a reduce to be split across workers, the combining operation must be associative and every piece must carry everything needed to combine it. A mean fails both - watch what happens.'}});
    const show = [];
    chunks.forEach((c, i) => { show.push([{zh:'平均 ' + means[i] + '；(總和, 筆數) = (' + S1[i] + ', ' + N1[i] + ')', en:'mean ' + means[i] + '; (sum, count) = (' + S1[i] + ', ' + N1[i] + ')'}, COL.tealL]);
      F.push({shapes:stage({hot:i, show:show.slice(), cap:{zh:'worker ' + i + '：平均 ' + means[i], en:'worker ' + i + ': mean ' + means[i]}, cap2:{zh:'同時記下總和與筆數', en:'also keep the sum and the count'}}),
        panels:[{lbl:'chunk', chips:[chip(String(i), 'hot')]}], line:18,
        msg:i === 1 ? {zh:'區塊 1 只有一筆：一個 400 毫秒的離群值。它的平均就是 400。', en:'Chunk 1 holds a single value: one 400 ms outlier. Its mean is 400.'}
          : {zh:'區塊 ' + i + '：' + c.length + ' 筆，平均 ' + means[i] + '。', en:'Chunk ' + i + ': ' + c.length + ' values, mean ' + means[i] + '.'}});
    });
    F.push({shapes:stage({show:show, lines:[[{zh:'平均的平均 = (' + means.join(' + ') + ') / 3 = ' + mom, en:'mean of means = (' + means.join(' + ') + ') / 3 = ' + mom}, COL.red]],
        cap:{zh:'平均的平均：' + mom, en:'mean of means: ' + mom}, cap2:{zh:'那一筆 400 被算成答案的三分之一', en:'the single 400 counts as a third of the answer'}}),
      panels:[{lbl:'answer', chips:[chip(String(mom), 'bad')]}], line:16,
      msg:{zh:'把三個平均再平均得到 ' + mom + '。錯在每個區塊被當成一樣重：只有一筆的區塊 1，跟有六筆的區塊 2 權重相同。區塊切法一變，答案就跟著變——平行度改了，結果也改了。', en:'Averaging the three means gives ' + mom + '. Every chunk is treated as equally heavy: chunk 1 with one value weighs as much as chunk 2 with six. Change how the data is cut and the answer changes - a different degree of parallelism gives a different result.'}});
    F.push({shapes:stage({show:show, lines:[[{zh:'平均的平均 = ' + mom, en:'mean of means = ' + mom}, COL.red],
          [{zh:'(總和, 筆數) 相加 = (' + tot + ', ' + n + ')，' + tot + ' / ' + n + ' = ' + tot / n, en:'add the (sum, count) pairs = (' + tot + ', ' + n + '), ' + tot + ' / ' + n + ' = ' + tot / n}, COL.tealL]],
        cap:{zh:'正確的平均：' + tot / n, en:'the true mean: ' + tot / n}, cap2:{zh:'(總和, 筆數) 的相加滿足結合律，最後再除一次', en:'adding (sum, count) pairs is associative; divide once at the end'}}),
      panels:[{lbl:'answer', chips:[chip(String(tot / n), 'ok')]}], line:19,
      msg:{zh:'reduce 的對象改成 (總和, 筆數)：相加滿足結合律，怎麼切、怎麼併都得到 (' + tot + ', ' + n + ')，最後除一次得 ' + tot / n + '。同一個道理：整數加法、max、min、集合聯集、Day 32 的 HyperLogLog 合併都能安全地平行 reduce；平均數不行，要求每個 bit 都一樣時浮點加法也不行（十個 0.1 用迴圈加是 0.9999999999999999，樹狀加是 1.0）。',
           en:'Reduce (sum, count) pairs instead: adding them is associative, so any split and any merge order gives (' + tot + ', ' + n + '), and one division at the end gives ' + tot / n + '. The same rule says integer +, max, min, set union and Day 32\'s HyperLogLog merge can be reduced in parallel safely; a mean cannot, and neither can floating-point + if you need identical bits (ten 0.1s add up to 0.9999999999999999 in a loop and 1.0 as a tree).'}});
  }
  return F.list;
}

/* ================================================================ meta */
const DAY_META = {
  title:{zh:'Day 43 — 並行與平行模型', en:'Day 43 - Concurrency and parallelism'},
  sub:{zh:'GIL、race、deadlock、asyncio、work 與 depth、map-reduce：在算還是在等，以及到底能平行多少。', en:'The GIL, races, deadlocks, asyncio, work and depth, map-reduce: computing or waiting, and how parallel it can be.'},
  tabs:[
    {
      id:'gil', label:{zh:'GIL', en:'the GIL'},
      stage:{zh:'兩個工作：thread 輪流拿 GIL，process 各有各的', en:'two jobs: threads take turns with one GIL, processes have one each'},
      view:VIEW,
      variants:[{zh:'2 thread，計算', en:'2 threads, computing'}, {zh:'2 process，計算', en:'2 processes, computing'}, {zh:'2 thread，等待', en:'2 threads, waiting'}],
      idea:{zh:'CPython 同一時間只讓一個 thread 執行 Python bytecode（拿著 GIL 的那個），預設每 5 毫秒交接一次。所以兩個吃 CPU 的 thread 是輪流用同一顆核心，總時間跟一個接一個跑一樣；兩個 process 各有直譯器和 GIL，真的同時在兩顆核心上跑。等網路、等硬碟、sleep 的時候 thread 會放掉 GIL，所以「等待」可以完全重疊。選工具只要問一句：這個工作在算，還是在等？這一頁的格子是示意；實際量測（雙核心 VM）：兩個計算工作 thread 0.244 秒、循序 0.207 秒、process 0.156 秒；20 次 50 毫秒等待，迴圈 1.004 秒、20 個 thread 0.058 秒。',
            en:'CPython lets one thread at a time run Python bytecode (the one holding the GIL), handing it over every 5 ms by default. So two CPU-bound threads take turns on one core and take as long as running them one after the other, while two processes each have their own interpreter and GIL and really run on two cores. A thread that waits on the network, the disk or sleep releases the GIL, so waits overlap completely. Picking a tool is one question: is the job computing or waiting? The cells on this page are schematic; measured on a 2-core VM: two CPU jobs took 0.244 s on threads, 0.207 s serially, 0.156 s on processes; twenty 50 ms waits took 1.004 s in a loop and 0.058 s on twenty threads.'},
      legend:[['#3fe0dd', {zh:'在執行（拿著 GIL）', en:'running (holds a GIL)'}],
              ['#ff9736', {zh:'這一格', en:'this tick'}],
              ['#9d6bff', {zh:'等 I/O，已放掉 GIL', en:'waiting on I/O, GIL released'}],
              ['#ff5c5c', {zh:'想執行但在等 GIL', en:'ready to run, waiting for the GIL'}]],
      get code(){ return CODE_GIL; },
      build:gilFrames
    },
    {
      id:'race', label:{zh:'race：每種交錯', en:'race: every interleaving'},
      stage:{zh:'兩個 thread 各做一次 x += 1：跑過每一種交錯', en:'two threads each do x += 1: run every interleaving'},
      view:VIEW,
      variants:[{zh:'沒有 lock', en:'no lock'}, {zh:'with lock', en:'with lock'}],
      idea:{zh:'x += 1 是「讀、加、寫」，另一個 thread 在讀和寫之間寫入，它的更新就被蓋掉。真的 thread 跑一次只看得到作業系統挑的一種交錯，所以這裡用一個檢查器把每一種交錯都跑一遍（跟 Day 42 把每一種亂數選擇都重播一遍是同一招）：兩個 thread 6 種裡 4 種掉更新，3 個 thread 90 種只有 6 種對，4 個 thread 2,520 種只有 24 種對——正確的剛好就是 thread 依序跑完的那些。lock 讓壞的交錯不可能發生。陷阱：在 CPython 3.10 裡 box.v += 1 剛好不會被切開，四個 thread 各 200,000 次每次都是 800,000；在讀和寫之間加一個函式呼叫，就有 29% 到 64% 的更新消失。',
            en:'x += 1 is read, add, write; if another thread writes between the read and the write, its update is overwritten. A real run sees only the one interleaving the OS happened to pick, so a checker runs every interleaving instead (the same trick as Day 42 replaying every random choice): with two threads 4 of 6 lose an update, with three only 6 of 90 are right, with four only 24 of 2,520 - exactly the ones where the threads ran one after the other. A lock makes the bad interleavings impossible. The trap: on CPython 3.10 box.v += 1 happens never to be split, and four threads doing it 200,000 times each always give 800,000; put one function call between the read and the write and 29 to 64 percent of the updates vanish.'},
      legend:[['#ff9736', {zh:'這一步', en:'this step'}],
              ['#3fe0dd', {zh:'做完的步驟 / 正確', en:'steps done / correct'}],
              ['#9d6bff', {zh:'共用的 x', en:'the shared x'}],
              ['#ff5c5c', {zh:'掉了更新 / 被 lock 擋住', en:'lost update / blocked by the lock'}]],
      get code(){ return CODE_RACE; },
      build:raceFrames
    },
    {
      id:'phil', label:{zh:'哲學家（LC 1226）', en:'philosophers (LC 1226)'},
      stage:{zh:'五支叉子、五把 lock：誰先拿哪一支', en:'five forks, five locks: who takes which first'},
      view:VIEW,
      variants:[{zh:'先拿左邊', en:'left fork first'}, {zh:'先拿編號小的', en:'lower fork first'}],
      idea:{zh:'每個人都先拿左邊叉子時，有可能每個人都拿著一支、等下一個人手上的那支：deadlock。檢查器走過全部 9,302 個可到達的狀態，卡死的只有 1 個；24 億 3 千萬種交錯裡只有 120 種走到那裡，約兩千萬分之一——測試碰不到，正式環境總有一天碰得到。修法是鎖順序：每個人都先拿編號小的叉子，P4 就會先伸手拿 f0，環閉不起來；8,447 個狀態、0 個卡死。LeetCode 1226 的解法就是這一行 min/max。',
            en:'When everybody takes the left fork first, everyone may end up holding one fork and waiting for the one the next person holds: deadlock. The checker walks all 9,302 reachable states and exactly 1 is stuck; 120 of 2,430,842,120 interleavings get there, about one in twenty million - tests never see it, production eventually will. The fix is lock ordering: everybody takes the lower-numbered fork first, so P4 reaches for f0 first and the cycle cannot close; 8,447 states, 0 stuck. The LeetCode 1226 solution is that one line of min/max.'},
      legend:[['#3fe0dd', {zh:'拿著的叉子', en:'a fork being held'}],
              ['#ff9736', {zh:'這一步的哲學家', en:'the philosopher moving now'}],
              ['#ff5c5c', {zh:'在等的叉子 / 卡死', en:'a fork waited for / stuck'}],
              ['#9d6bff', {zh:'兩支都到手', en:'has both forks'}]],
      get code(){ return CODE_PHIL; },
      build:philFrames
    },
    {
      id:'async', label:{zh:'asyncio', en:'asyncio'},
      stage:{zh:'一個 thread、一個 event loop、十個在等的 task', en:'one thread, one event loop, ten waiting tasks'},
      view:VIEW,
      variants:[{zh:'await', en:'await'}, {zh:'time.sleep（卡住）', en:'time.sleep (blocks)'}, {zh:'Semaphore(3)', en:'Semaphore(3)'}],
      idea:{zh:'asyncio 不用 thread 就能讓等待重疊：task 一路跑到 await，把控制權交回 event loop，loop 去跑別的 task。十個 0.1 秒的等待用 await asyncio.sleep，總共 0.1 秒。陷阱是 async def 裡的阻塞呼叫：time.sleep、requests.get、吃 CPU 的迴圈都不會交出控制權，十個 task 變成一個接一個跑、1.0 秒，而且沒有任何警告。改用可以 await 的版本，或 asyncio.to_thread。Semaphore(3) 是刻意的限制：同時最多三個，分 4 輪、0.4 秒。這一頁的時間軸是模擬；實際量到的是 0.10、1.00、0.40 秒。',
            en:'asyncio overlaps waits without threads: a task runs until an await, hands control back to the event loop, and the loop runs another task. Ten 0.1 s waits with await asyncio.sleep take 0.1 s in total. The trap is a blocking call inside an async def: time.sleep, requests.get or a CPU-heavy loop never hand control back, so the ten tasks run one after another, 1.0 s, with no warning. Use an awaitable version, or asyncio.to_thread. Semaphore(3) is the deliberate limit: at most three at once, 4 rounds, 0.4 s. The timeline here is simulated; measured runs gave 0.10, 1.00 and 0.40 s.'},
      legend:[['#ff9736', {zh:'還在等', en:'still waiting'}],
              ['#3fe0dd', {zh:'等完了', en:'done waiting'}],
              ['#ff5c5c', {zh:'被阻塞的呼叫拖成一個一個', en:'serialised by a blocking call'}],
              ['#9d6bff', {zh:'現在時間', en:'now'}]],
      get code(){ return CODE_ASYNC; },
      build:asyncFrames
    },
    {
      id:'crawl', label:{zh:'爬蟲（LC 1242）', en:'crawler (LC 1242)'},
      stage:{zh:'news.yahoo.com 上 16 頁，每次抓取 50 毫秒', en:'16 pages on news.yahoo.com, 50 ms per fetch'},
      view:VIEW,
      variants:[{zh:'依序', en:'serial'}, {zh:'一波一波', en:'in waves'}, {zh:'host in url', en:'host in url'}],
      idea:{zh:'LeetCode 1242：找出所有走得到、主機名稱相同的網頁，getUrls 很慢。抓網頁是在等，所以 thread 有用：廣度優先、每一波的網頁同時丟給 thread pool。依序要 16 次抓取的時間（0.80 秒），分波只要 5 次（0.26 秒），因為最長的連結鏈有 5 頁——16 是 work、5 是 depth，再多 thread 也打不破 5。陷阱：用 host in url 比對，news.yahoo.com.evil.example 也會混進來；要切出主機名稱再用 == 比。',
            en:'LeetCode 1242: find every reachable page with the same host name, where getUrls is slow. Fetching is waiting, so threads help: breadth-first, each wave fetched at once by a thread pool. Serial costs 16 fetches (0.80 s), waves cost 5 (0.26 s), because the longest chain of links is 5 pages - 16 is the work, 5 the depth, and no number of threads beats 5. The trap: comparing with host in url lets news.yahoo.com.evil.example in; cut out the host name and compare with ==.'},
      legend:[['#ff9736', {zh:'正在抓', en:'being fetched'}],
              ['#9d6bff', {zh:'已發現、還沒抓', en:'found, not fetched yet'}],
              ['#3fe0dd', {zh:'抓完了', en:'fetched'}],
              ['#ff5c5c', {zh:'不該爬的網頁', en:'should not be crawled'}]],
      get code(){ return CODE_CRAWL; },
      build:crawlFrames
    },
    {
      id:'wd', label:{zh:'work 與 depth', en:'work and depth'},
      stage:{zh:'8 個數、4 個 worker：每一步最多做 4 次加法', en:'8 numbers, 4 workers: at most 4 additions per tick'},
      view:VIEW,
      variants:[{zh:'加總：迴圈', en:'sum: loop'}, {zh:'加總：樹', en:'sum: tree'}, {zh:'Hillis-Steele', en:'Hillis-Steele'}, {zh:'Blelloch', en:'Blelloch'}],
      idea:{zh:'把計算畫成圖：work W 是運算總數（一個 worker 的時間），depth D 是最長的依賴鏈（無限多 worker 的時間）。貪婪排程在 p 個 worker 上需要 T_p 步，max(W/p, D) ≤ T_p ≤ W/p + D（Brent）。迴圈 D = W，加 worker 沒用；樹狀加總 D = log₂ n。前綴和有兩種平行版：Hillis-Steele depth 最小但 work 是九倍，1024 個數在 4 個 worker 上要 2,305 步，比迴圈的 1,023 步還慢；Blelloch 的 work 約 2n，worker 少於 545 個時都贏。Amdahl 定律是特例：5% 拆不開，64 核只有 15.4 倍。',
            en:'Draw the computation as a graph: work W is the number of operations (time on one worker), depth D the longest chain of dependencies (time on unlimited workers). A greedy scheduler on p workers needs T_p ticks with max(W/p, D) <= T_p <= W/p + D (Brent). A loop has D = W and extra workers do nothing; a tree sum has D = log2 n. Prefix sums have two parallel versions: Hillis-Steele has the smallest depth but nine times the work, so 1024 numbers take 2,305 ticks on 4 workers, slower than the loop\'s 1,023; Blelloch keeps the work near 2n and wins below 545 workers. Amdahl\'s law is the special case: with 5% that cannot be split, 64 cores give 15.4x.'},
      legend:[['#ff9736', {zh:'這一步在做的加法', en:'additions running this tick'}],
              ['#3fe0dd', {zh:'已完成', en:'done'}],
              ['#0a6b74', {zh:'還沒輪到', en:'not yet'}],
              ['#8fa3ac', {zh:'輸入', en:'inputs'}]],
      get code(){ return CODE_WD; },
      build:wdFrames
    },
    {
      id:'mr', label:{zh:'map-reduce', en:'map-reduce'},
      stage:{zh:'map 各自數、shuffle 依 bucket 分、reduce 加總', en:'map counts, shuffle splits by bucket, reduce adds up'},
      view:VIEW,
      variants:[{zh:'bucket 一致', en:'buckets agree'}, {zh:'bucket 不一致', en:'buckets disagree'}, {zh:'平均的平均', en:'mean of means'}],
      idea:{zh:'map-reduce 是給一台機器裝不下的資料用的 work-depth：map 每份文件各自做，shuffle 把每個字送到 bucket(word) 號 reducer，reduce 加總。兩個無聲的失敗：每個 mapper 必須把同一個字放進同一個桶子——hash(word) 在 spawn 出來的 process 裡種子不同（Windows、macOS 的預設），在 Windows、macOS 上幾乎每次都會把大部分的字數錯（Python 版模擬兩個 mapper 分桶不一致時，67 個字錯 51 個）；以及 reduce 必須滿足結合律、每份帶齊合併所需的資訊——平均的平均是 200，真正的平均是 130，要 reduce (總和, 筆數)。這一頁的語料和 bucket 函式是示範用的小例子。',
            en:'Map-reduce is work and depth for data that does not fit on one machine: map each document on its own, shuffle every word to reducer bucket(word), reduce by adding. Two silent failures: every mapper must put a word in the same bucket - hash(word) is seeded differently in spawned processes (the default on Windows and macOS), so most words come out wrong on almost every run there (simulating two mappers that disagree, the Python run got 51 of 67 words wrong); and the reduce must be associative with pieces that carry what the merge needs - the mean of means is 200 while the true mean is 130, so reduce (sum, count) pairs. The corpus and bucket function on this page are a small demo example.'},
      legend:[['#ff9736', {zh:'正在處理', en:'being processed'}],
              ['#9d6bff', {zh:'reducer', en:'reducer'}],
              ['#3fe0dd', {zh:'正確', en:'correct'}],
              ['#ff5c5c', {zh:'被拆開的字 / 錯的答案', en:'a split word / a wrong answer'}]],
      get code(){ return CODE_MR; },
      build:mrFrames
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
