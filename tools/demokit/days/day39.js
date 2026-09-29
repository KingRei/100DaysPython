// DAY: 39
// TITLE_ZH: 回溯與剪枝：N-Queens、數獨
// TITLE_EN: Backtracking and pruning: N-Queens and Sudoku
// SUB_ZH: 回溯法就是深度優先搜尋一棵從來沒有真的建出來的樹：放一顆、遞迴下去、拿回來、再試下一顆。choose、explore、unchoose 三行撐起今天的每一題。剪枝決定它跑不跑得完——部分解一違反規則就丟掉，底下整棵子樹跟著消失，8 皇后從 1,677 萬種擺法縮到 2,057 個節點；unchoose 決定它對不對——回溯的 bug 幾乎從不當機：少一行復原，程式會宣布 8 皇后無解；少一個 [:]，你會拿到 8 個空 list。數獨也一樣，先填選擇最少的格子，LeetCode 37 的範例從 4,209 個節點降到 52 個。
// SUB_EN: Backtracking is depth-first search over a tree of partial answers that is never actually built: place one piece, recurse, take it back, try the next. Three lines - choose, explore, unchoose - carry every problem today. Pruning decides whether it finishes: reject a partial answer the moment it breaks a rule and the whole subtree below it disappears, which takes 8 queens from 16.8 million placements to 2,057 nodes. The unchoose step decides whether it is right, and its bugs never crash: drop one undo and the program announces that 8 queens has no solution; drop one [:] and you get eight empty lists. Sudoku follows the same rule - branch on the cell with the fewest legal digits and LeetCode 37's example drops from 4,209 nodes to 52.
// FOLDER: day%2039%20-%20backtracking%20and%20pruning
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
function chip(t, cls){ return {t:t, cls:cls || ''}; }

/* text metrics, mirroring the harness model: a CJK glyph is one em wide, a
   Latin one about 0.55.  fitT shrinks a caption until both languages fit the
   view - the English sentence is usually the wider of the two. */
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
/* a label that must fit a box of width w: shrink the font instead of spilling */
function fitFs(lab, w, fs){
  const strs = (typeof lab === 'string') ? [lab] : [lab.zh, lab.en];
  const need = Math.max.apply(null, strs.map(t => tw(t, fs)));
  return need > w - .08 ? fs * (w - .08) / need : fs;
}
const pyList = a => '[' + a.join(', ') + ']';
const pySet = s => s.size ? '{' + Array.from(s).sort((a, b) => a - b).join(', ') + '}' : 'set()';
const fmt = n => n.toLocaleString('en-US');

/* tidy tree layout: leaves take consecutive slots in DFS order, a parent sits
   over the middle of its first and last child.  nodes = [{id, parent, kids}] */
function layoutTree(nodes, x0, x1, y0, dy){
  const byId = {}; nodes.forEach(n => { byId[n.id] = n; n.kids = []; });
  nodes.forEach(n => { if (n.parent != null) byId[n.parent].kids.push(n); });
  let slot = 0;
  const root = nodes[0];
  (function walk(n, d){
    n.depth = d;
    if (!n.kids.length){ n.slot = slot++; return; }
    n.kids.forEach(k => walk(k, d + 1));
    n.slot = (n.kids[0].slot + n.kids[n.kids.length - 1].slot) / 2;
  })(root, 0);
  const span = Math.max(1, slot);
  nodes.forEach(n => {
    n.x = x0 + (x1 - x0) * (n.slot + .5) / span;
    n.y = y0 + n.depth * dy;
  });
  return (x1 - x0) / span;
}

/* =======================================================================
 * Tab 1 - the template: choose, explore, unchoose on the subsets of [1,2,3]
 * ======================================================================= */
const CODE_SUB = [
  'def subsets(nums):',
  '    res = []',
  '    path = []',
  '',
  '    def go(start):',
  '        res.append(path[:])                  # record a COPY of the path',
  '        for i in range(start, len(nums)):',
  '            path.append(nums[i])             # choose',
  '            go(i + 1)                        # explore',
  '            path.pop()                       # unchoose',
  '',
  '    go(0)',
  '    return res'
];
const CODE_SUB_BAD = CODE_SUB.slice();
CODE_SUB_BAD[0] = 'def subsets_aliased(nums):';
CODE_SUB_BAD[5] = '        res.append(path)                     # BUG: stores the list object itself';

const NUMS = [1, 2, 3];
/* run subsets() for real and log every step it takes */
function recordSubsets(){
  const ev = [], path = [], res = [], nodes = [];
  function go(start, parent){
    const id = nodes.length;
    nodes.push({id:id, parent:parent, path:path.slice(), start:start});
    res.push(path.slice());
    ev.push({k:'rec', id:id, start:start, path:path.slice()});
    for (let i = start; i < NUMS.length; i++){
      path.push(NUMS[i]);
      ev.push({k:'choose', id:id, i:i, x:NUMS[i], path:path.slice(), child:nodes.length});
      go(i + 1, id);
      path.pop();
      ev.push({k:'pop', id:id, i:i, x:NUMS[i], path:path.slice(),
               next:(i + 1 < NUMS.length ? NUMS[i + 1] : null)});
    }
  }
  go(0, null);
  return {ev:ev, nodes:nodes, res:res};
}
const SUB = recordSubsets();
layoutTree(SUB.nodes, .35, 4.65, 1.30, 1.0);
const subLab = p => '[' + p.join(',') + ']';

function subFrames(v){
  const F = new Frames(), alias = v === 1, N = SUB.nodes;
  const TW = .98, TH = .42;
  let seen = 0;                    // how many tree nodes have been entered
  const stack = [];                // ids of the go() calls currently running
  const res = [];                  // correct: copies; alias: count of references

  function stage(o){
    const sh = [fitT(4.9, .45, alias
        ? {zh:'res.append(path)：8 格全是同一個 list', en:'res.append(path): all eight slots are one list'}
        : {zh:'每個節點進來先記一份 path 的副本', en:'every node records a copy of path on entry'},
        {c:alias ? COL.red : COL.tealL, fs:.32})];
    /* the recursion tree, drawn only as far as the search has reached */
    const onStack = new Set(stack);
    N.forEach(n => {
      if (n.parent == null || n.id >= seen) return;
      const p = N[n.parent];
      sh.push(S.e(p.x, p.y + TH / 2, n.x, n.y - TH / 2,
                  {s:onStack.has(n.id) ? 'act' : 'idle', arrow:false, w:.04,
                   lab:'+' + n.path[n.path.length - 1], fs:.22, lx:(n.x < p.x ? -.16 : .16), ly:0}));
    });
    if (o.pending != null){
      const c = N[o.pending], p = N[c.parent];
      sh.push(S.e(p.x, p.y + TH / 2, c.x, c.y - TH / 2,
                  {s:'hot', w:.05, dash:'.09 .07', lab:'+' + c.path[c.path.length - 1], fs:.22,
                   lx:(c.x < p.x ? -.16 : .16), ly:0}));
    }
    N.forEach(n => {
      if (n.id >= seen) return;
      const st = n.id === o.cur ? 'hot' : onStack.has(n.id) ? 'act' : 'ok';
      sh.push(S.r(n.x - TW / 2, n.y - TH / 2, TW, TH, st, subLab(n.path), {fs:.22, rx:.08}));
    });
    sh.push(S.t(.35, 5.02, {zh:'遞迴樹：節點 = 一次 go() 呼叫', en:'recursion tree: one node per go() call'},
                {c:COL.grey, fs:.24, anchor:'start'}));

    /* path: the one list every call shares */
    sh.push(S.t(5.75, 1.18, 'path', {c:COL.purpleL, fs:.28, anchor:'end'}));
    for (let i = 0; i < 3; i++){
      const has = i < o.path.length;
      sh.push(S.r(5.95 + i * .66, .82, .58, .5, has ? (i === o.path.length - 1 && o.hotPath ? 'hot' : 'act') : 'ghost',
                  has ? String(o.path[i]) : '', {fs:.28}));
    }
    sh.push(S.t(8.05, 1.18, 'len ' + o.path.length, {c:COL.grey, fs:.24, anchor:'start'}));

    if (!alias){
      sh.push(fitT(5.10, 1.92, {zh:'res：每格是當時 path 的副本', en:'res: each slot is a copy of that moment'},
                  {c:COL.tealL, fs:.24, anchor:'start'}));
      for (let i = 0; i < 8; i++){
        const col = i >> 2, row = i & 3, x = 5.40 + col * 2.2, y = 2.20 + row * .56;
        const has = i < res.length;
        sh.push(S.t(x - .12, y + .29, String(i), {c:COL.grey, fs:.22, anchor:'end'}));
        sh.push(S.r(x, y, 1.95, .42, has ? (i === res.length - 1 && o.newSlot ? 'hot' : 'ok') : 'ghost',
                    has ? pyList(res[i]) : '', {fs:.26}));
      }
    } else {
      sh.push(fitT(5.10, 1.92, {zh:'res：8 格都指向「同一個物件」',
                               en:'res: all slots point at one object'},
                  {c:COL.red, fs:.24, anchor:'start'}));
      const bx = 5.75, bw = 3.3, by = 3.85, bh = .56;
      for (let i = 0; i < 8; i++){
        const x = 5.30 + i * .53, has = i < res.length;
        sh.push(S.r(x, 2.25, .45, .40, has ? (i === res.length - 1 && o.newSlot ? 'hot' : 'act') : 'ghost',
                    String(i), {fs:.22}));
        if (has) sh.push(S.e(x + .225, 2.68, bx + bw * (i + .5) / 8, by - .04,
                             {s:i === res.length - 1 && o.newSlot ? 'hot' : 'act', w:.035}));
      }
      sh.push(S.r(bx, by, bw, bh, 'bad', pyList(o.path), {fs:.30}));
      sh.push(fitT(bx + bw / 2, by + bh + .38, {zh:'唯一的 list 物件（也就是 path）',
                                                 en:'the only list object - it IS path'},
                   {c:COL.red, fs:.24}));
    }
    if (o.cap) sh.push(fitT(4.9, 5.62, o.cap, {c:o.capc || COL.pale, fs:.30}));
    if (o.cap2) sh.push(fitT(4.9, 6.06, o.cap2, {c:COL.grey, fs:.26}));
    return sh;
  }
  const resChips = path => alias
    ? Array.from({length:res.length}, () => chip(pyList(path), 'bad'))
    : res.map(r => chip(pyList(r), 'ok'));
  const panels = (path, extra) => [
    {lbl:{zh:'呼叫堆疊', en:'call stack'},
     chips:stack.map(id => chip('go(' + N[id].start + ')', id === stack[stack.length - 1] ? 'hot' : 'act'))},
    {lbl:'path', chips:[chip(pyList(path), 'act')]},
    {lbl:alias ? {zh:'print(res) 此刻會印出', en:'print(res) right now'} : 'res',
     chips:resChips(path)}].concat(extra || []);

  F.push({shapes:stage({path:[], cap:{zh:'nums = [1, 2, 3]，要列出全部 2³ = 8 個子集合',
                                     en:'nums = [1, 2, 3] - list all 2^3 = 8 subsets'},
                        cap2:{zh:'只有一個 path 在整棵樹上共用：往下走 append，回來 pop',
                              en:'one path is shared by the whole tree: append going down, pop coming back'}}),
    panels:panels([]), line:11,
    msg:alias
      ? {zh:'和上一個版本只差一個地方：第 6 行把 path[:] 寫成了 path。程式照樣跑、照樣回傳 8 個東西，不會有任何錯誤訊息。要看清楚的是「res 裡到底存了什麼」——右邊把 res 的每一格畫成一個箭頭，箭頭指向哪個物件，那一格就是那個物件。',
         en:'One character difference from the other variant: line 6 says path instead of path[:]. The program still runs and still returns eight things, with no error anywhere. The thing to watch is what res actually stores - on the right each slot of res is drawn as an arrow, and a slot is whatever object its arrow points at.'}
      : {zh:'回溯法是深度優先搜尋一棵「部分答案」組成的樹，但這棵樹從來沒有真的建出來：程式只有一個 path，往下走的時候 append 一個數，回來的時候 pop 掉。每個節點都是一個合法的子集合，所以 go() 一進來就先記錄。',
         en:'Backtracking is depth-first search over a tree of partial answers, but the tree is never built: the program owns exactly one list, path, appends a number on the way down and pops it on the way back. Every node of this tree is itself a valid subset, so go() records one the moment it is entered.'}});

  SUB.ev.forEach(e => {
    const n = N[e.id];
    if (e.k === 'rec'){
      seen = Math.max(seen, e.id + 1);
      stack.push(e.id);
      res.push(e.path.slice());
      const leaf = e.start === NUMS.length;
      F.push({shapes:stage({cur:e.id, path:e.path, newSlot:true,
          cap:alias
            ? {zh:'res[' + (res.length - 1) + '] = path（不是副本），8 格裡已有 ' + res.length + ' 格指向它',
               en:'res[' + (res.length - 1) + '] = path, not a copy - ' + res.length + ' slots now point at it'}
            : {zh:'go(' + e.start + ') 記下 res[' + (res.length - 1) + '] = ' + pyList(e.path),
               en:'go(' + e.start + ') records res[' + (res.length - 1) + '] = ' + pyList(e.path)},
          capc:alias ? COL.red : COL.tealL,
          cap2:leaf ? {zh:'range(3, 3) 是空的：這個呼叫記完就返回', en:'range(3, 3) is empty: this call returns right after recording'}
                    : {zh:'接著從 nums[' + e.start + '] 開始往右挑', en:'next it picks from nums[' + e.start + '] rightwards'}}),
        panels:panels(e.path), line:5,
        msg:alias
          ? {zh:'go(' + e.start + ') 執行 res.append(path)。存進去的不是 ' + pyList(e.path) + ' 這個「值」，而是 path 這個 list 物件本身，所以 res 的 ' + res.length + ' 格此刻全部顯示 ' + pyList(e.path) + '——它們根本是同一個東西。之後 path 每 append 或 pop 一次，這 ' + res.length + ' 格就一起跟著變。' + (leaf ? '這是 [1,2,3] 這片葉子，range(3, 3) 是空的，所以這次呼叫馬上返回。' : ''), en:''}
             : {zh:'go(' + e.start + ') 進來第一件事是記錄 path 目前的樣子 ' + pyList(e.path) + '。每個節點本身就是一個合法的子集合，不用等到走到葉子才算數。寫 path[:] 是為了拷貝一份「此刻的樣子」：path 等一下還會被 append、pop 很多次，副本不會跟著變。' + (leaf ? '這裡 start = 3，range(3, 3) 是空的——[1,2,3] 後面已經沒有數字可以加，所以記完就返回。' : ''),
             en:''},
      });
      const last = F.list[F.list.length - 1];
      if (alias) last.msg.en = 'go(' + e.start + ') runs res.append(path). What goes in is not the value ' + pyList(e.path) + ' but the list object path itself, so all ' + res.length + ' slots of res show ' + pyList(e.path) + ' right now - they are literally the same thing. From here on, every append or pop on path changes all ' + res.length + ' of them at once.' + (leaf ? ' This is the leaf [1,2,3]: range(3, 3) is empty, so the call returns immediately.' : '');
      else last.msg.en = 'The first thing go(' + e.start + ') does is record what path looks like now, ' + pyList(e.path) + '. Every node is a valid subset in its own right; nothing has to wait for a leaf. path[:] takes a copy of this moment, because path is about to be appended to and popped many more times and the copy must not follow it.' + (leaf ? ' Here start = 3 and range(3, 3) is empty - there is nothing left to add after [1,2,3] - so the call returns right after recording.' : '');
    } else if (e.k === 'choose'){
      F.push({shapes:stage({cur:e.id, path:e.path, hotPath:true, pending:e.child,
          cap:{zh:'choose：path.append(' + e.x + ') → ' + pyList(e.path), en:'choose: path.append(' + e.x + ') -> ' + pyList(e.path)},
          capc:COL.orangeL,
          cap2:{zh:'explore：接著呼叫 go(' + (e.i + 1) + ')，下一層只能挑 ' + e.x + ' 右邊的數',
                en:'explore: go(' + (e.i + 1) + ') next - the level below may only pick numbers right of ' + e.x}}),
        panels:panels(e.path), line:7,
        msg:{zh:'choose：path.append(nums[' + e.i + '])，path 變成 ' + pyList(e.path) + '。接著 explore 呼叫 go(' + (e.i + 1) + ')，傳的是 i + 1 而不是 start + 1：下一層只能從 ' + e.x + ' 的右邊挑，所以 [2, 1] 這種順序顛倒的重複永遠不會出現，每個子集合剛好產生一次。' + (alias ? '注意 res 裡已經存好的每一格也同時變成了 ' + pyList(e.path) + '。' : ''),
             en:'choose: path.append(nums[' + e.i + ']), and path becomes ' + pyList(e.path) + '. Then explore calls go(' + (e.i + 1) + ') - i + 1, not start + 1 - so the level below can only pick from the right of ' + e.x + '. That is why a reordered duplicate such as [2, 1] can never appear: every subset is produced exactly once.' + (alias ? ' Notice that every slot already in res has just turned into ' + pyList(e.path) + ' as well.' : '')}});
    } else {
      stack.pop();
      const back = stack.length ? N[stack[stack.length - 1]] : null;
      F.push({shapes:stage({cur:e.id, path:e.path,
          cap:{zh:'unchoose：go(' + (e.i + 1) + ') 返回，path.pop() 拿掉 ' + e.x + ' → ' + pyList(e.path),
               en:'unchoose: go(' + (e.i + 1) + ') returned, path.pop() removes ' + e.x + ' -> ' + pyList(e.path)},
          capc:COL.purpleL,
          cap2:e.next != null
            ? {zh:'下一輪要試 ' + e.next + '：path 必須先回到 ' + pyList(e.path), en:'the loop tries ' + e.next + ' next, so path must be back to ' + pyList(e.path)}
            : {zh:'迴圈結束，go(' + n.start + ') 也返回', en:'the loop is done and go(' + n.start + ') returns too'}}),
        panels:panels(e.path), line:9,
        msg:e.next != null
          ? {zh:'unchoose：子呼叫回來了，path.pop() 把 ' + e.x + ' 拿掉，path 回到 ' + pyList(e.path) + '，和剛進入這個節點時一模一樣。這一行不能少：迴圈下一輪要放的是 ' + e.next + '，少了 pop，' + e.x + ' 會留在裡面，下一個子集合就成了 ' + pyList(e.path.concat([e.x, e.next])) + ' 而不是 ' + pyList(e.path.concat([e.next])) + '。' + (alias ? 'res 的每一格也跟著變回 ' + pyList(e.path) + '。' : ''),
             en:'unchoose: the child call is back, and path.pop() removes ' + e.x + ', leaving ' + pyList(e.path) + ' - exactly what this node saw on entry. The line cannot go: the next round of the loop places ' + e.next + ', and without the pop the ' + e.x + ' would still be there, making the next subset ' + pyList(e.path.concat([e.x, e.next])) + ' instead of ' + pyList(e.path.concat([e.next])) + '.' + (alias ? ' Every slot of res has changed back to ' + pyList(e.path) + ' with it.' : '')}
          : {zh:'unchoose：path.pop() 拿掉 ' + e.x + '，path 回到 ' + pyList(e.path) + '。這已經是迴圈最後一個數，go(' + n.start + ') 跟著返回' + (back ? '，交回給 go(' + back.start + ')' : '') + '。回溯的規矩是：任何一次呼叫返回時，共用的狀態必須和它進來時完全相同，上一層才能放心地接著試下一個。' + (alias ? '而 res 的 ' + res.length + ' 格此刻全都是 ' + pyList(e.path) + '。' : ''),
             en:'unchoose: path.pop() removes ' + e.x + ' and path is back to ' + pyList(e.path) + '. That was the last number in the loop, so go(' + n.start + ') returns as well' + (back ? ', handing control back to go(' + back.start + ')' : '') + '. The rule of backtracking: whenever a call returns, the shared state must be exactly as it was when the call began, so the level above can safely try its next option.' + (alias ? ' Meanwhile all ' + res.length + ' slots of res now read ' + pyList(e.path) + '.' : '')}});
    }
  });

  const printed = alias ? '[' + res.map(() => '[]').join(', ') + ']' : '[' + res.map(pyList).join(', ') + ']';
  F.push({shapes:stage({path:[],
      cap:alias ? {zh:'len(res) = 8 看起來對，內容卻是 8 個 []', en:'len(res) == 8 looks right, the contents are eight []'}
                : {zh:'8 個子集合，剛好 2³，每格是自己的副本', en:'8 subsets, exactly 2^3, each slot its own copy'},
      capc:alias ? COL.red : COL.tealL,
      cap2:alias ? {zh:'只差 [:] 三個字元；沒有例外，沒有警告', en:'three characters, [:], apart - no exception, no warning'}
                 : {zh:'樹上 8 個節點 = 8 次 go() 呼叫 = 8 個答案', en:'8 tree nodes = 8 calls to go() = 8 answers'}}),
    panels:panels([], [{lbl:{zh:'return res', en:'return res'}, chips:[chip(printed, alias ? 'bad' : 'ok')]}]), line:12,
    msg:alias
      ? {zh:'回傳 ' + printed + '。長度是 8，只檢查 len 的測試會通過；可是 8 格全都是 path 本身，而 path 在最後一次 pop 之後是空的，所以你拿到 8 個空 list。Python 的 list 是可變物件，append 存的是參照；要保存「這一刻的樣子」就一定要拷貝，path[:] 或 list(path) 都可以。',
         en:'It returns ' + printed + '. The length is 8, so a test that only checks len passes; but all eight slots are path itself, and path is empty after the final pop, so what comes back is eight empty lists. A Python list is mutable and append stores a reference; to keep "what it looks like right now" you must copy it, with path[:] or list(path).'}
      : {zh:'回傳 ' + printed + '，8 個子集合 = 2³。整個演算法只有三行在做事：choose（append）、explore（遞迴）、unchoose（pop）。今天後面每一題——N 皇后、數獨、組合總和、單字搜尋——都是這三行，差別只在「choose 之前先檢查能不能放」，也就是剪枝。',
         en:'It returns ' + printed + ': eight subsets, 2^3. The whole algorithm is three working lines - choose (append), explore (recurse), unchoose (pop). Every other problem today, N-Queens, Sudoku, Combination Sum and Word Search, is the same three lines; what changes is a test before choose that refuses to place something illegal, which is pruning.'}});
  return F.list;
}

/* =======================================================================
 * N-Queens - shared recorder and board drawing for tab 2 and tab 3
 * ======================================================================= */
const CODE_Q = [
  'def queens(n):',
  '    cols, diag, anti = set(), set(), set()   # c, r - c, r + c',
  '    board, sols = [], []',
  '',
  '    def go(r):',
  '        if r == n:',
  '            sols.append(board[:])',
  '            return',
  '        for c in range(n):',
  '            if c in cols or r - c in diag or r + c in anti:',
  '                continue                                  # prune',
  '            cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)',
  '            go(r + 1)',
  '            board.pop(); cols.remove(c); diag.remove(r - c); anti.remove(r + c)',
  '',
  '    go(0)',
  '    return sols'
];
const CODE_QBAD = CODE_Q.slice();
CODE_QBAD[0] = 'def queens_no_undo(n):';
CODE_QBAD[13] = '            board.pop()                          # BUG: sets never shrink';

/* run the search for real; every event carries a snapshot of the state.
   undo=false is queens_no_undo: board.pop() but no set.remove() */
function recordQueens(n, undo){
  const cols = new Set(), diag = new Set(), anti = new Set(), board = [], sols = [], ev = [];
  const ever = [];                               // every queen ever placed, in order
  let nodes = 0;
  const snap = o => Object.assign(o, {board:board.slice(), cols:Array.from(cols),
    diag:Array.from(diag), anti:Array.from(anti), nodes:nodes, nsol:sols.length});
  function attacker(r, c, why){
    const hit = q => why === 'cols' ? q[1] === c : why === 'diag' ? q[0] - q[1] === r - c : q[0] + q[1] === r + c;
    const live = board.map((cc, rr) => [rr, cc]).find(hit);
    if (live) return {q:live, ghost:false};
    return {q:ever.find(hit), ghost:true};
  }
  function go(r){
    nodes++;
    ev.push(snap({k:'enter', r:r}));
    if (r === n){ sols.push(board.slice()); ev.push(snap({k:'sol', r:r})); return; }
    for (let c = 0; c < n; c++){
      const why = cols.has(c) ? 'cols' : diag.has(r - c) ? 'diag' : anti.has(r + c) ? 'anti' : null;
      if (why){ ev.push(snap(Object.assign({k:'prune', r:r, c:c, why:why}, attacker(r, c, why)))); continue; }
      cols.add(c); diag.add(r - c); anti.add(r + c); board.push(c); ever.push([r, c]);
      ev.push(snap({k:'place', r:r, c:c}));
      go(r + 1);
      board.pop();
      if (undo){ cols.delete(c); diag.delete(r - c); anti.delete(r + c); }
      ev.push(snap({k:'pop', r:r, c:c}));
    }
    ev.push(snap({k:'ret', r:r}));
  }
  go(0);
  return {ev:ev, nodes:nodes, sols:sols};
}
const QRUN = {4:recordQueens(4, true), 5:recordQueens(5, true), 6:recordQueens(6, true),
              8:recordQueens(8, true)};
const QBAD = recordQueens(8, false);

const SYM = {cols:'|', diag:'\\', anti:'/'};
const WHY = {
  cols:{zh:'cols', en:'cols'}, diag:{zh:'diag', en:'diag'}, anti:{zh:'anti', en:'anti'}};
function whyText(p){
  const rc = '(' + p.r + ',' + p.c + ')', q = '(' + p.q[0] + ',' + p.q[1] + ')';
  const g = p.ghost ? {zh:'——但那顆皇后早就被拿走了', en:' - a queen that was taken back long ago'} : {zh:'', en:''};
  if (p.why === 'cols') return {zh:rc + ' 被 cols 擋：第 ' + p.c + ' 欄已被 ' + q + ' 佔了' + g.zh,
                                en:rc + ' fails cols: column ' + p.c + ' belongs to ' + q + g.en};
  if (p.why === 'diag') return {zh:rc + ' 被 diag 擋：r − c = ' + (p.r - p.c) + '，和 ' + q + ' 在同一條 \\ 對角線' + g.zh,
                                en:rc + ' fails diag: r - c = ' + (p.r - p.c) + ', the same "\\" diagonal as ' + q + g.en};
  return {zh:rc + ' 被 anti 擋：r + c = ' + (p.r + p.c) + '，和 ' + q + ' 在同一條 / 對角線' + g.zh,
          en:rc + ' fails anti: r + c = ' + (p.r + p.c) + ', the same "/" diagonal as ' + q + g.en};
}
const joinZ = a => a.join('；'), joinE = a => a.join('; ');

/* board + loop ladder.  st = {n, board, cols, diag, anti, ghosts, hot:[r,c],
   rejected:[prune events], loops:[[status per column] per row], depth} */
function queenStage(st){
  const n = st.n, cell = n <= 6 ? 3.5 / n : .48, bx = .62, by = 1.05, sh = [];
  const LX = 5.55;
  const cs = new Set(st.cols), ds = new Set(st.diag), as = new Set(st.anti);
  const onBoard = (r, c) => st.board[r] === c;
  const liveHit = (r, c) => st.board.some((cc, rr) => cc === c || rr - cc === r - c || rr + cc === r + c);
  const rej = {}; (st.rejected || []).forEach(p => { rej[p.r + ',' + p.c] = p; });
  sh.push(S.t(bx, .86, {zh:'棋盤（× = 被三個 set 擋住）', en:'board (x = blocked by the three sets)'},
              {c:COL.grey, fs:.22, anchor:'start'}));
  sh.push(fitT(LX, .86, {zh:'go(r) 的迴圈：每一欄的下場', en:'go(r): what each column got'},
              {c:COL.grey, fs:.22, anchor:'start'}));
  for (let r = 0; r < n; r++){
    const y = by + r * cell;
    sh.push(S.t(bx - .12, y + cell * .62, String(r), {c:r === st.depth ? COL.orangeL : COL.grey, fs:.22, anchor:'end'}));
    sh.push(S.t(LX - .12, y + cell * .62, 'r=' + r, {c:r === st.depth ? COL.orangeL : COL.grey, fs:.22, anchor:'end'}));
    for (let c = 0; c < n; c++){
      const x = bx + c * cell, key = r + ',' + c;
      const blocked = cs.has(c) || ds.has(r - c) || as.has(r + c);
      if (rej[key]){
        sh.push(S.r(x, y, cell - .05, cell - .05, 'bad', SYM[rej[key].why], {fs:cell * .5, rx:.05}));
      } else if (onBoard(r, c)){
        sh.push(S.r(x, y, cell - .05, cell - .05, (r + c) % 2 ? 'soft' : 'idle', '', {rx:.05}));
      } else if (blocked && !onBoard(r, c) && !(st.ghosts || []).some(g => g[0] === r && g[1] === c)){
        const ghostOnly = !liveHit(r, c);
        sh.push(S.r(x, y, cell - .05, cell - .05, ghostOnly ? 'bad' : 'done', '×',
                    {fs:cell * .42, rx:.05, o:ghostOnly ? .55 : 1}));
      } else {
        sh.push(S.r(x, y, cell - .05, cell - .05, (r + c) % 2 ? 'soft' : 'idle', '', {rx:.05}));
      }
      /* the ladder: same row, same column, but it shows the loop's history */
      const ls = (st.loops[r] || [])[c];
      const lx = LX + c * cell;
      if (!ls || r > st.depth) sh.push(S.r(lx, y, cell - .05, cell - .05, 'ghost', '', {rx:.05}));
      else if (ls === 'on') sh.push(S.r(lx, y, cell - .05, cell - .05, (st.hot && st.hot[0] === r) ? 'hot' : 'act', 'Q', {fs:cell * .42, rx:.05}));
      else if (ls === 'done') sh.push(S.r(lx, y, cell - .05, cell - .05, 'done', String(c), {fs:Math.max(.19, cell * .36), rx:.05}));
      else if (ls === 'sol') sh.push(S.r(lx, y, cell - .05, cell - .05, 'ok', 'Q', {fs:cell * .42, rx:.05}));
      else sh.push(S.r(lx, y, cell - .05, cell - .05, 'bad', SYM[ls], {fs:cell * .45, rx:.05}));
    }
  }
  for (let c = 0; c < n; c++){
    sh.push(S.t(bx + c * cell + (cell - .05) / 2, by + n * cell + .22, String(c), {c:COL.grey, fs:.2}));
    sh.push(S.t(LX + c * cell + (cell - .05) / 2, by + n * cell + .22, String(c), {c:COL.grey, fs:.2}));
  }
  const ctr = (r, c) => [bx + c * cell + (cell - .05) / 2, by + r * cell + (cell - .05) / 2];
  (st.ghosts || []).forEach(g => {
    const [x, y] = ctr(g[0], g[1]);
    sh.push(S.c(x, y, cell * .34, 'bad', 'Q', {fs:Math.max(.19, cell * .36), dash:'.07 .05', o:.9}));
  });
  st.board.forEach((c, r) => {
    const [x, y] = ctr(r, c);
    const hot = st.hot && st.hot[0] === r && st.hot[1] === c;
    sh.push(S.c(x, y, cell * .36, hot ? 'hot' : (st.solved ? 'ok' : 'act'), 'Q', {fs:cell * .38}));
  });
  (st.rejected || []).forEach(p => {
    if (!p.q || (p.q[0] === p.r && p.q[1] === p.c)) return;
    const [x1, y1] = ctr(p.q[0], p.q[1]), [x2, y2] = ctr(p.r, p.c);
    sh.push(S.e(x1, y1, x2, y2, {s:'bad', w:.035, dash:'.08 .06', pad:cell * .36}));
  });
  return sh;
}

/* replay a recorded queens run as frames.  bad = the no-undo variant */
function queenFrames(n, run, bad){
  const F = new Frames(), total = run.nodes, perRow = Math.pow(n, n);
  const loops = [], ghosts = [], trail = [];
  let pr = [], pops = [], place = null;

  const freeIn = (e, r) => { if (r >= n) return 0;
    const cs = new Set(e.cols), ds = new Set(e.diag), as = new Set(e.anti); let k = 0;
    for (let c = 0; c < n; c++) if (!cs.has(c) && !ds.has(r - c) && !as.has(r + c)) k++;
    return k; };
  function setChips(vals, e, kind){
    if (!bad) return vals.length ? [chip(pySet(new Set(vals)), 'act')] : [];
    const live = new Set(e.board.map((c, r) => kind === 'cols' ? c : kind === 'diag' ? r - c : r + c));
    return vals.slice().sort((a, b) => a - b).map(v => chip(String(v), live.has(v) ? 'act' : 'bad'));
  }
  function panels(e, solved){
    const p = [{lbl:'cols', chips:setChips(e.cols, e, 'cols')},
               {lbl:{zh:'diag（r − c，\\ 對角線）', en:'diag (r - c, "\\" diagonals)'}, chips:setChips(e.diag, e, 'diag')},
               {lbl:{zh:'anti（r + c，/ 對角線）', en:'anti (r + c, "/" diagonals)'}, chips:setChips(e.anti, e, 'anti')},
               {lbl:'board', chips:[chip(pyList(e.board), solved ? 'ok' : 'hot')]},
               {lbl:{zh:'已走節點（go 呼叫次數）', en:'nodes so far (calls to go)'},
                chips:[chip(e.nodes + ' / ' + total, 'dim')]}];
    if (bad) p.push({lbl:{zh:'進入過的部分棋盤', en:'partial boards entered'},
                     chips:trail.map((b, i) => chip(pyList(b), i === trail.length - 1 ? 'hot' : 'dim'))});
    else p.push({lbl:{zh:'找到的解', en:'solutions found'},
                 chips:run.sols.slice(0, e.nsol).map(s => chip(pyList(s), 'ok'))});
    return p;
  }
  function stage(e, o){
    const sh = [fitT(4.9, .45, bad
        ? {zh:'8×8，unchoose 只剩 board.pop()：三個 set 只增不減',
           en:'8x8 with only board.pop() - the three sets never shrink'}
        : {zh:n + '×' + n + ' 皇后：一列一顆，三個 set 做 O(1) 檢查',
           en:n + 'x' + n + ' queens: one per row, three sets for O(1) checks'},
        {c:bad ? COL.red : COL.tealL, fs:.32})];
    sh.push.apply(sh, queenStage({n:n, board:o.board || e.board, cols:e.cols, diag:e.diag, anti:e.anti,
      ghosts:bad ? ghosts : [], hot:o.hot, rejected:o.rej, loops:loops, depth:o.depth, solved:o.solved}));
    const capY = n === 8 ? 5.66 : 5.52;
    if (o.cap) sh.push(fitT(4.9, capY, o.cap, {c:o.capc || COL.pale, fs:.29}));
    if (o.cap2) sh.push(fitT(4.9, capY + .44, o.cap2, {c:COL.grey, fs:.25}));
    return sh;
  }
  const rc = (r, c) => '(' + r + ',' + c + ')';
  function popText(){
    if (!pops.length) return {zh:'', en:''};
    const list = pops.map(p => rc(p.r, p.c)).join('、'), listE = pops.map(p => rc(p.r, p.c)).join(', ');
    return bad
      ? {zh:'unchoose：' + list + ' 從 board 拿掉了，但三個 set 沒有 remove，它的欄和兩條對角線仍然標記為被佔——棋盤上看不見的幽靈皇后繼續在攻擊。',
         en:'unchoose: ' + listE + ' came off the board, but nothing was removed from the three sets, so its column and both diagonals are still marked - an invisible ghost queen keeps attacking. '}
      : {zh:'unchoose：取回 ' + list + '，三個 set 一起 remove，它佔的欄與兩條對角線重新開放。',
         en:'unchoose: ' + listE + ' taken back and removed from all three sets, freeing its column and both diagonals. '};
  }
  function pruneText(){
    if (!pr.length) return {zh:'', en:''};
    return {zh:joinZ(pr.map(p => whyText(p).zh)) + '。', en:joinE(pr.map(p => whyText(p).en)) + '. '};
  }

  run.ev.forEach(e => {
    if (e.k === 'enter'){
      trail.push(e.board.slice());
      if (e.r < n) loops[e.r] = new Array(n).fill(null);
      if (e.r === 0 && e.nodes === 1){
        F.push({shapes:stage(e, {depth:0,
            cap:bad ? {zh:'和上一頁同一個搜尋，只是 unchoose 那一行少了三個 remove', en:'the same search, with three remove() calls missing from unchoose'}
                    : {zh:'go(0)：第 0 列，從第 0 欄開始試', en:'go(0): row 0, starting from column 0'},
            capc:bad ? COL.red : COL.tealL,
            cap2:bad ? {zh:'紅色虛線的 Q = 已經拿走、卻還留在 set 裡的皇后', en:'a dashed red Q = a queen taken off the board but still in the sets'}
                     : {zh:'每列一顆的擺法有 ' + n + '^' + n + ' = ' + fmt(perRow) + ' 種；看剪枝後實際走幾個節點',
                        en:'one queen per row allows ' + n + '^' + n + ' = ' + fmt(perRow) + ' boards; count what the pruned search visits'}}),
          panels:panels(e), line:bad ? 13 : 15,
          msg:bad
            ? {zh:'這個版本和正確的搜尋只差第 14 行：board.pop() 還在，三個 set.remove() 被刪掉了。從棋盤上看，回溯一切正常，皇后會被拿走；但 cols、diag、anti 只會變大不會變小，被拿走的皇后仍然在「攻擊」它原本的欄和兩條對角線。程式不會出錯，只會算出錯的答案。',
               en:'This version differs from the correct search only in line 14: board.pop() is still there, the three set.remove() calls are gone. On the board everything looks like normal backtracking - queens do come off - but cols, diag and anti only ever grow, so a queen that was taken back still "attacks" its column and both diagonals. Nothing raises; the answer is simply wrong.'}
            : {zh:n + '×' + n + ' 的棋盤要放 ' + n + ' 顆互不攻擊的皇后（同欄、同列、同斜線都算攻擊）。每列恰好一顆，所以 go(r) 只需要決定第 r 列的皇后放哪一欄。三個 set 讓檢查只要 O(1)：cols 記已用的欄；同一條 \\ 對角線上每格的 r − c 都相同，所以 diag 記 r − c；同一條 / 對角線上 r + c 相同，anti 記 r + c。',
               en:'Place ' + n + ' queens on an ' + n + 'x' + n + ' board so that none attacks another - same column, same row or same diagonal all count. There is exactly one queen per row, so go(r) only has to choose a column for row r. Three sets make the safety test O(1): cols holds the columns in use; every square on one "\\" diagonal shares r - c, so diag holds r - c; every square on one "/" diagonal shares r + c, so anti holds r + c.'}});
      } else if (e.r < n){
        const r = place.r, c = place.c, k = freeIn(e, e.r), pz = popText(), qz = pruneText();
        F.push({shapes:stage(e, {depth:r, hot:[r, c], rej:pr,
            cap:{zh:(pr.length ? '拒絕 ' + pr.length + ' 格後，' : '') + '放下 ' + rc(r, c) + ' → go(' + e.r + ')',
                 en:(pr.length ? pr.length + ' rejected, then ' : '') + 'place ' + rc(r, c) + ' -> go(' + e.r + ')'},
            capc:COL.orangeL,
            cap2:k === 0 ? {zh:'第 ' + e.r + ' 列已經全被擋住：這一步注定是死路', en:'row ' + e.r + ' is already fully blocked: this branch is dead'}
                         : {zh:'第 ' + e.r + ' 列還剩 ' + k + ' 格沒被攻擊', en:'row ' + e.r + ' still has ' + k + ' unattacked square' + (k > 1 ? 's' : '')}}),
          panels:panels(e), line:11,
          msg:{zh:pz.zh + qz.zh + rc(r, c) + ' 的 c = ' + c + '、r − c = ' + (r - c) + '、r + c = ' + (r + c) + ' 三個 set 都查不到，代表它和目前的皇后不同欄、也不在同一條斜線上。choose：三個 set 各加一筆、board.append(' + c + ')；explore：進入 go(' + e.r + ')。每列只放一顆，所以同列衝突不用檢查。' +
                  (k === 0 ? '可是第 ' + e.r + ' 列已經一格都不剩了——程式還是得進去逐欄查過才知道，下一步就會看到。' : '第 ' + e.r + ' 列還剩 ' + k + ' 格沒被攻擊。'),
               en:pz.en + qz.en + rc(r, c) + ' has c = ' + c + ', r - c = ' + (r - c) + ', r + c = ' + (r + c) + ', and none of the three sets contains its value, so it shares no column and no diagonal with the queens already down. choose: one entry into each set and board.append(' + c + '); explore: go(' + e.r + '). One queen per row means a row clash is impossible by construction. ' +
                  (k === 0 ? 'But row ' + e.r + ' has no free square left - the program still has to walk in and test every column to find that out, which is the next step.' : 'Row ' + e.r + ' has ' + k + ' unattacked square' + (k > 1 ? 's' : '') + ' left.')}});
        pr = []; pops = [];
      }
    } else if (e.k === 'prune'){
      loops[e.r][e.c] = e.why; pr.push(e);
    } else if (e.k === 'place'){
      loops[e.r][e.c] = 'on'; place = e;
    } else if (e.k === 'sol'){
      const r = place.r, c = place.c, pz = popText(), qz = pruneText();
      loops[r][c] = 'sol';
      F.push({shapes:stage(e, {depth:r, hot:[r, c], rej:pr, solved:true,
          cap:{zh:'第 ' + e.nsol + ' 個解：' + pyList(e.board), en:'solution #' + e.nsol + ': ' + pyList(e.board)}, capc:COL.tealL,
          cap2:{zh:'sols.append(board[:])，然後照常 return、unchoose，繼續找', en:'sols.append(board[:]), then return and unchoose as usual - keep searching'}}),
        panels:panels(e, true), line:6,
        msg:{zh:pz.zh + qz.zh + '放下 ' + rc(r, c) + ' 之後進入 go(' + n + ')，r == n：' + n + ' 列各有一顆，而且每一顆放下時都通過了三個 set 的檢查，所以互不攻擊——這是第 ' + e.nsol + ' 個解。存的是 board[:] 副本，理由和第一個分頁一樣。題目要全部的解，所以找到了也不停：照常 return、unchoose，繼續試下一欄。',
             en:pz.en + qz.en + 'With ' + rc(r, c) + ' down, go(' + n + ') sees r == n: all ' + n + ' rows hold a queen, and each one passed the three-set test when it was placed, so none attacks another - solution #' + e.nsol + '. What gets stored is the copy board[:], for the same reason as in the first tab. The task wants every solution, so finding one is no reason to stop: return, unchoose, and try the next column.'}});
      loops[r][c] = 'on';
      pr = []; pops = [];
    } else if (e.k === 'pop'){
      loops[e.r][e.c] = 'done'; loops.length = e.r + 1;
      pops.push(e);
      if (bad) ghosts.push([e.r, e.c]);
    } else if (e.k === 'ret' && pr.length){
      const r = e.r, pz = popText(), qz = pruneText(), all = pr.length === n;
      const ghostOnly = pr.every(p => p.ghost);
      F.push({shapes:stage(e, {depth:r, rej:pr,
          cap:{zh:'第 ' + r + ' 列' + (all ? '每一欄' : '剩下的欄') + '都被擋：go(' + r + ') 返回，死路',
               en:'row ' + r + (all ? ': every column' : ': every remaining column') + ' blocked - go(' + r + ') returns, dead end'},
          capc:COL.red,
          cap2:r === 0 ? {zh:'棋盤上一顆皇后都沒有，擋住第 0 列的全是幽靈', en:'there is no queen on the board - row 0 is blocked by ghosts alone'}
               : {zh:'第 ' + r + ' 列以下整棵子樹被剪掉，回到第 ' + (r - 1) + ' 列換下一欄',
                  en:'everything below row ' + r + ' is pruned; back to row ' + (r - 1) + ' for its next column'}}),
        panels:panels(e), line:10,
        msg:r === 0
          ? {zh:pz.zh + qz.zh + '第 0 列剩下的欄全部被拒——但棋盤上現在一顆皇后都沒有。擋住它們的全是已經被拿走的幽靈，它們的欄和對角線從來沒從 set 裡刪掉。go(0) 返回，搜尋結束：0 個解。',
             en:pz.en + qz.en + 'Every remaining column of row 0 is rejected - yet there is not a single queen on the board. Everything blocking them is a ghost whose column and diagonals were never deleted from the sets. go(0) returns and the search is over: 0 solutions.'}
          : {zh:pz.zh + qz.zh + '第 ' + r + ' 列' + (all ? '的 ' + n + ' 欄全部' : '剩下的欄也都') + '被拒，go(' + r + ') 什麼都沒放就返回。剪枝就發生在這裡：上面 ' + r + ' 顆皇后已經讓這一列無處可放，所以第 ' + r + ' 列以下的整棵子樹一格都不用試，直接回到第 ' + (r - 1) + ' 列取回那顆皇后、換下一欄。' + (bad && ghostOnly ? '而且擋住它們的全是幽靈——那些格子其實根本沒有被攻擊。' : ''),
             en:pz.en + qz.en + (all ? 'All ' + n + ' columns' : 'Every remaining column') + ' of row ' + r + ' is rejected, so go(' + r + ') returns having placed nothing. This is where pruning pays: the ' + r + ' queens above leave this row no room, so the whole subtree below row ' + r + ' is never tried, and control goes straight back to row ' + (r - 1) + ' to take its queen back and move it along.' + (bad && ghostOnly ? ' And every one of those rejections came from a ghost - none of those squares is actually attacked.' : '')}});
      pr = []; pops = [];
    }
  });

  /* the final frame */
  const last = run.ev[run.ev.length - 1];
  if (bad){
    const good = QRUN[8];
    F.push({shapes:stage(last, {depth:-1,
        cap:{zh:'0 個解、' + run.nodes + ' 個節點：「8 皇后無解」', en:'0 solutions after ' + run.nodes + ' nodes: "8 queens is impossible"'}, capc:COL.red,
        cap2:{zh:'補回三個 remove：' + good.sols.length + ' 個解、' + fmt(good.nodes) + ' 個節點',
              en:'with the three remove() calls back: ' + good.sols.length + ' solutions, ' + fmt(good.nodes) + ' nodes'}}),
      panels:panels(last).concat([{lbl:{zh:'正確版本', en:'correct version'},
        chips:[chip(good.sols.length + ' solutions', 'ok'), chip(fmt(good.nodes) + ' nodes', 'ok')]}]), line:16,
      msg:{zh:'回傳 []，總共只進入 ' + run.nodes + ' 個部分棋盤。程式很有自信地宣布 8 皇后無解，沒有例外、沒有警告。把三個 remove 補回去，同一個搜尋會走 ' + fmt(good.nodes) + ' 個節點、找到 ' + good.sols.length + ' 個解。回溯的正確性完全建立在一個不變式上：每次呼叫返回時，共用狀態要和進來時一樣。board 做到了，三個 set 沒做到，而 set 才是負責判斷的那一方。',
           en:'It returns [] after entering only ' + run.nodes + ' partial boards, and announces with complete confidence that 8 queens has no solution - no exception, no warning. Put the three remove() calls back and the same search visits ' + fmt(good.nodes) + ' nodes and finds ' + good.sols.length + ' solutions. Backtracking is correct only because of one invariant: when a call returns, the shared state is exactly as it found it. board keeps that promise; the three sets do not, and the sets are what make the decisions.'}});
  } else {
    const sol = run.sols[run.sols.length - 1];
    F.push({shapes:stage(last, {depth:-1, board:sol, solved:true,
        cap:{zh:n + ' 皇后：' + run.sols.length + ' 個解、' + run.nodes + ' 個節點（含根）',
             en:n + ' queens: ' + run.sols.length + ' solutions, ' + run.nodes + ' nodes (root included)'}, capc:COL.tealL,
        cap2:{zh:'每列一顆的暴力擺法有 ' + fmt(perRow) + ' 種；剪枝後只走了 ' + run.nodes + ' 個節點',
              en:'one queen per row gives ' + fmt(perRow) + ' boards; the pruned search entered ' + run.nodes + ' nodes'}}),
      panels:panels(last, true), line:16,
      msg:{zh:'go(0) 的迴圈跑完，搜尋結束：' + run.sols.length + ' 個解，總共 ' + run.nodes + ' 次 go() 呼叫（含根）。如果先把每列一顆的 ' + fmt(perRow) + ' 種擺法全部產生出來再檢查，工作量會大好幾個數量級；回溯的差別在於一發現衝突就放棄，整棵子樹不再展開。棋盤上顯示的是最後找到的那個解。',
           en:'The loop in go(0) finishes and the search is over: ' + run.sols.length + ' solutions from ' + run.nodes + ' calls to go(), root included. Generating all ' + fmt(perRow) + ' one-queen-per-row boards first and checking them afterwards would cost orders of magnitude more; backtracking gives up the moment a clash appears, so the subtree behind it is never expanded. The board shows the last solution found.'}});
  }
  return F.list;
}

/* =======================================================================
 * Tab 4 - Sudoku: which cell you branch on
 * ======================================================================= */
const CODE_MRV = [
  'def go():                        # Solution37: MRV + bitmasks',
  '    if not empty:',
  '        return True',
  '    best, best_free, best_k = -1, 0, 10',
  '    for i, (r, c) in enumerate(empty):',
  '        free = ~(rows[r] | cols[c] | boxes[r // 3 * 3 + c // 3]) & 0x3FE',
  "        k = bin(free).count('1')",
  '        if k < best_k:',
  '            best, best_free, best_k = i, free, k',
  '            if k <= 1:',
  '                break            # cannot do better than forced',
  '    if best_k == 0:',
  '        return False             # a cell with no legal digit',
  '    r, c = empty[best]',
  '    empty[best] = empty[-1]; empty.pop()    # O(1) removal',
  '    b = r // 3 * 3 + c // 3',
  '    while best_free:',
  '        bit = best_free & -best_free',
  '        best_free ^= bit',
  '        rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit',
  '        board[r][c] = str(bit.bit_length() - 1)',
  '        if go():',
  '            return True',
  '        rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit',
  "    board[r][c] = '.'",
  '    empty.append((r, c))',
  '    empty[best], empty[-1] = empty[-1], empty[best]',
  '    return False'
];
const CODE_NAIVE = [
  'def go():                        # sudoku_naive: reading order, 1..9',
  '    nonlocal nodes',
  '    nodes += 1',
  '    for r in range(9):',
  '        for c in range(9):',
  '            if g[r][c] == 0:',
  '                for v in range(1, 10):',
  '                    if safe(r, c, v):            # scans row, column, box',
  '                        g[r][c] = v              # choose',
  '                        if go():                 # explore',
  '                            return True',
  '                        g[r][c] = 0              # unchoose',
  '                return False                     # no digit fits here',
  '    return True                                  # no empty cell left'
];
const LC37 = '53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79';
const parseSu = s => Array.from({length:9}, (_, r) =>
  Array.from({length:9}, (_, c) => s[r * 9 + c] === '.' ? 0 : +s[r * 9 + c]));
const boxOf = (r, c) => Math.floor(r / 3) * 3 + Math.floor(c / 3);
const popc = x => { let k = 0; while (x){ x &= x - 1; k++; } return k; };
const bitsOf = m => { const o = []; for (let v = 1; v <= 9; v++) if (m >> v & 1) o.push(v); return o; };
const GIVEN = parseSu(LC37);

/* sudoku_mrv, line for line: same scan order, same tie-break, same swap removal */
function recordMRV(){
  const g = parseSu(LC37), rows = new Array(9).fill(0), cols = new Array(9).fill(0), boxes = new Array(9).fill(0);
  const empty = [], ev = [];
  for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++){
    if (g[r][c]){ const bit = 1 << g[r][c]; rows[r] |= bit; cols[c] |= bit; boxes[boxOf(r, c)] |= bit; }
    else empty.push([r, c]);
  }
  let nodes = 0;
  const freeOf = (r, c) => ~(rows[r] | cols[c] | boxes[boxOf(r, c)]) & 0x3FE;
  function go(){
    nodes++;
    if (!empty.length){ ev.push({k:'done', nodes:nodes, g:g.map(x => x.slice())}); return true; }
    let best = -1, bestFree = 0, bestK = 10, scanned = empty.length;
    for (let i = 0; i < empty.length; i++){
      const [r, c] = empty[i], free = freeOf(r, c), k = popc(free);
      if (k < bestK){ best = i; bestFree = free; bestK = k;
        if (k <= 1){ scanned = i + 1; break; } }
    }
    const counts = {};
    empty.forEach(([r, c]) => { counts[r * 9 + c] = popc(freeOf(r, c)); });
    if (bestK === 0){ ev.push({k:'dead', nodes:nodes, g:g.map(x => x.slice()), counts:counts}); return false; }
    const [r, c] = empty[best];
    const nEmpty = empty.length;
    empty[best] = empty[empty.length - 1]; empty.pop();
    const b = boxOf(r, c);
    let free = bestFree;
    const masks = [rows[r], cols[c], boxes[b]];
    while (free){
      const bit = free & -free;
      free ^= bit;
      ev.push({k:'pick', nodes:nodes, r:r, c:c, kk:bestK, free:bestFree, v:Math.log2(bit), masks:masks,
               g:g.map(x => x.slice()), counts:counts, scanned:scanned, nEmpty:nEmpty});
      rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
      g[r][c] = Math.log2(bit);
      if (go()) return true;
      rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit;
      g[r][c] = 0;
    }
    empty.push([r, c]);
    const t = empty[best]; empty[best] = empty[empty.length - 1]; empty[empty.length - 1] = t;
    return false;
  }
  const ok = go();
  return {ev:ev, nodes:nodes, ok:ok};
}
/* sudoku_naive: first empty cell in reading order, digits 1..9, safe() scans.
   Events are kept for the first `lim` nodes only; the count runs to the end. */
function recordNaive(lim){
  const g = parseSu(LC37), ev = [];
  let nodes = 0, firstDead = null;
  function safe(r, c, v){
    for (let i = 0; i < 9; i++) if (g[r][i] === v || g[i][c] === v) return false;
    const br = r - r % 3, bc = c - c % 3;
    for (let i = br; i < br + 3; i++) for (let j = bc; j < bc + 3; j++) if (g[i][j] === v) return false;
    return true;
  }
  function go(){
    nodes++;
    const me = nodes;
    for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++){
      if (g[r][c] !== 0) continue;
      const rej = [], failed = [];
      for (let v = 1; v <= 9; v++){
        if (!safe(r, c, v)){ rej.push(v); continue; }
        if (nodes < lim) ev.push({k:'place', nodes:nodes + 1, r:r, c:c, v:v, rej:rej.slice(),
                                   failed:failed.slice(), g:g.map(x => x.slice())});
        g[r][c] = v;
        if (go()) return true;
        g[r][c] = 0;
        failed.push(v);
        if (nodes <= lim) ev.push({k:'undo', r:r, c:c, v:v});
      }
      if (firstDead == null) firstDead = {node:me, r:r, c:c};
      if (nodes <= lim) ev.push({k:'dead', nodes:nodes, node:me, r:r, c:c, rej:rej.slice(), failed:failed.slice(),
                                  g:g.map(x => x.slice())});
      return false;
    }
    return true;
  }
  const ok = go();
  return {ev:ev, nodes:nodes, ok:ok, firstDead:firstDead, g:g};
}
const MRV = recordMRV();
const NAIVE = recordNaive(60);

const SU = {x:.42, y:1.0, cell:.47};
function suGrid(g, o){
  const sh = [], {x, y, cell} = SU;
  for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++){
    const cx = x + c * cell, cy = y + r * cell, v = g[r][c], key = r * 9 + c;
    const hot = o.hot && o.hot[0] === r && o.hot[1] === c;
    if (hot) sh.push(S.r(cx, cy, cell - .04, cell - .04, o.hotSt || 'hot', o.hotLab == null ? '' : String(o.hotLab), {fs:.28, rx:.04}));
    else if (v && GIVEN[r][c]) sh.push(S.r(cx, cy, cell - .04, cell - .04, 'idle', String(v), {fs:.27, rx:.04}));
    else if (v) sh.push(S.r(cx, cy, cell - .04, cell - .04, o.solved ? 'ok' : 'act', String(v), {fs:.27, rx:.04}));
    else {
      sh.push(S.r(cx, cy, cell - .04, cell - .04, o.mark && o.mark[key] ? 'bad' : 'soft', '', {rx:.04, o:.85}));
      if (o.counts && o.counts[key] != null)
        sh.push(S.t(cx + (cell - .04) / 2, cy + cell * .62, String(o.counts[key]),
                    {c:o.counts[key] === 1 ? COL.purpleL : o.counts[key] === 0 ? COL.red : COL.grey, fs:.2}));
    }
  }
  for (let i = 0; i <= 3; i++){
    const p = i * 3 * cell - .02;
    sh.push(S.e(x + p, y - .02, x + p, y + 9 * cell - .02, {s:'ok', arrow:false, w:.03, o:.55}));
    sh.push(S.e(x - .02, y + p, x + 9 * cell - .02, y + p, {s:'ok', arrow:false, w:.03, o:.55}));
  }
  return sh;
}
/* nine little cells, one per digit */
function digitRow(y, title, stOf, tcol){
  const sh = [S.t(5.62, y + .29, title, {c:tcol || COL.grey, fs:.22, anchor:'end'})];
  for (let v = 1; v <= 9; v++){
    const st = stOf(v);
    sh.push(S.r(5.75 + (v - 1) * .41, y, .36, .38, st, String(v), {fs:.22, rx:.05}));
  }
  return sh;
}

function sudokuFrames(v){
  const F = new Frames(), naive = v === 1;
  const nEmpty0 = GIVEN.flat().filter(x => !x).length;
  const used = (g, r, c) => {
    const inRow = new Set(g[r].filter(Boolean)), inCol = new Set(g.map(row => row[c]).filter(Boolean)), inBox = new Set();
    const br = r - r % 3, bc = c - c % 3;
    for (let i = br; i < br + 3; i++) for (let j = bc; j < bc + 3; j++) if (g[i][j]) inBox.add(g[i][j]);
    return [inRow, inCol, inBox];
  };
  function maskRows(g, r, c, lastRow){
    const [a, b, d] = used(g, r, c), bx = boxOf(r, c);
    const sh = [fitT(5.12, .86, {zh:'(' + r + ',' + c + ') 的列、欄、宮已用掉的數字',
                                en:'digits used in the row, col, box of (' + r + ',' + c + ')'},
                    {c:COL.grey, fs:.22, anchor:'start'})];
    sh.push.apply(sh, digitRow(1.05, 'row ' + r, x => a.has(x) ? 'act' : 'ghost'));
    sh.push.apply(sh, digitRow(1.55, 'col ' + c, x => b.has(x) ? 'act' : 'ghost'));
    sh.push.apply(sh, digitRow(2.05, 'box ' + bx, x => d.has(x) ? 'act' : 'ghost'));
    sh.push.apply(sh, digitRow(2.72, lastRow.title, lastRow.st, lastRow.tcol));
    return sh;
  }
  function stage(g, o, side){
    const sh = [fitT(4.9, .45, naive
        ? {zh:'固定順序：第一個空格、數字 1 到 9', en:'fixed order: first empty cell, digits 1 to 9'}
        : {zh:'MRV：每次挑合法數字最少的空格', en:'MRV: always branch on the cell with the fewest legal digits'},
        {c:naive ? COL.orangeL : COL.tealL, fs:.32})];
    sh.push.apply(sh, suGrid(g, o));
    if (side) sh.push.apply(sh, side);
    (o.lines || []).forEach((t, i) => sh.push(fitT(5.12, 3.55 + i * .42, t, {c:i === 0 ? COL.pale : COL.grey, fs:.25, anchor:'start'})));
    if (o.cap) sh.push(fitT(4.9, 5.66, o.cap, {c:o.capc || COL.pale, fs:.29}));
    if (o.cap2) sh.push(fitT(4.9, 6.10, o.cap2, {c:COL.grey, fs:.25}));
    return sh;
  }
  const histo = counts => { const h = {}; Object.values(counts).forEach(k => { const key = k >= 3 ? '3+' : String(k); h[key] = (h[key] || 0) + 1; });
    return ['0', '1', '2', '3+'].filter(k => h[k]).map(k => chip('k=' + k + ': ' + h[k], k === '1' ? 'act' : k === '0' ? 'bad' : 'dim')); };

  if (!naive) return mrvFrames(F, stage, maskRows, histo, nEmpty0);
  return naiveFrames(F, stage, maskRows, used);
}

function mrvFrames(F, stage, maskRows, histo, nEmpty0){
  const picks = MRV.ev.filter(e => e.k === 'pick');
  const maxK = Math.max.apply(null, picks.map(e => e.kk));
  const first = picks[0];
  F.push({shapes:stage(GIVEN, {counts:first.counts,
      lines:[{zh:'小數字 = 合法數字個數 k', en:'small number = legal digits k'},
             {zh:'紫色 = 只剩一個數字能填（forced）', en:'purple = exactly one digit fits (forced)'}],
      cap:{zh:'LeetCode 37 範例：30 個提示，' + nEmpty0 + ' 個空格', en:'LeetCode 37 example: 30 givens, ' + nEmpty0 + ' empty cells'}, capc:COL.tealL,
      cap2:{zh:'rows / cols / boxes 各是 9 個 bitmask：bit v = 數字 v 已用', en:'rows / cols / boxes are 9 bitmasks each: bit v set = digit v used'}}),
    panels:[{lbl:{zh:'各空格的候選數分布', en:'candidate counts over the empty cells'}, chips:histo(first.counts)},
            {lbl:{zh:'已走節點', en:'nodes so far'}, chips:[chip('0 / ' + MRV.nodes, 'dim')]}],
    line:3,
    msg:{zh:'數獨的回溯和 N 皇后一樣：選一個空格、填一個合法數字、遞迴、不行就拿回來。差別在「選哪個空格」。MRV（minimum remaining values）每次都去掃所有空格，挑合法數字最少的那一個。rows[r] | cols[c] | boxes[b] 把三個 bitmask OR 起來就是已用的數字，取反之後剩下的 bit 就是能填的數字，k 是它們的個數。',
         en:'Sudoku backtracking works exactly like N-Queens: pick an empty cell, write a legal digit, recurse, take it back if it fails. What differs is which cell to pick. MRV - minimum remaining values - scans every empty cell and branches on the one with the fewest legal digits. OR-ing rows[r] | cols[c] | boxes[b] gives the digits already used, the complement gives the digits that still fit, and k counts them.'}});
  picks.forEach((e, i) => {
    const side = maskRows(e.g, e.r, e.c, {title:'free', tcol:COL.tealL,
      st:x => (e.free >> x & 1) ? (x === e.v ? 'hot' : 'ok') : 'ghost'});
    const forced = e.kk === 1;
    F.push({shapes:stage(e.g, {counts:e.counts, hot:[e.r, e.c], hotLab:e.v,
        lines:[{zh:'k = ' + e.kk + (forced ? '：只能填 ' + e.v : '：先試最小的 ' + e.v),
                en:'k = ' + e.kk + (forced ? ': only ' + e.v + ' fits' : ': try the smallest, ' + e.v)},
               {zh:'掃了 ' + e.scanned + '/' + e.nEmpty + ' 格' + (e.scanned < e.nEmpty ? '，k ≤ 1 就停' : ''),
                en:'scanned ' + e.scanned + '/' + e.nEmpty + ' cells' + (e.scanned < e.nEmpty ? ', stop at k <= 1' : '')},
               {zh:'節點 ' + e.nodes + ' / ' + MRV.nodes, en:'node ' + e.nodes + ' of ' + MRV.nodes}],
        cap:{zh:'節點 ' + e.nodes + '：(' + e.r + ',' + e.c + ') 填 ' + e.v + (forced ? '（唯一選擇）' : ''),
             en:'node ' + e.nodes + ': (' + e.r + ',' + e.c + ') gets ' + e.v + (forced ? ' (the only choice)' : '')},
        capc:COL.orangeL,
        cap2:{zh:'還剩 ' + (e.nEmpty - 1) + ' 個空格', en:(e.nEmpty - 1) + ' empty cells left'}}, side),
      panels:[{lbl:{zh:'各空格的候選數分布', en:'candidate counts over the empty cells'}, chips:histo(e.counts)},
              {lbl:{zh:'這一格', en:'this cell'}, chips:[chip('(' + e.r + ',' + e.c + ')', 'hot'), chip('free = ' + pyList(bitsOf(e.free)), 'ok')]},
              {lbl:{zh:'已走節點', en:'nodes so far'}, chips:[chip(e.nodes + ' / ' + MRV.nodes, 'dim')]}],
      line:20,
      msg:i === 0
        ? {zh:'第一次掃描：從 empty 串列的開頭往後看，第 ' + e.scanned + ' 格 (' + e.r + ',' + e.c + ') 就只剩 k = ' + e.kk + ' 個數字。k ≤ 1 不可能再更少，所以直接 break，不必掃完 ' + e.nEmpty + ' 格。它的列、欄、宮合起來已經用掉 ' + (9 - e.kk) + ' 個數字，只剩 ' + e.v + '——這一格根本沒有選擇，填下去不是猜，是推論。',
           en:'The first scan walks the empty list from the start, and cell number ' + e.scanned + ', (' + e.r + ',' + e.c + '), already has only k = ' + e.kk + ' legal digit. Nothing can beat k <= 1, so the loop breaks instead of scanning all ' + e.nEmpty + ' cells. Its row, column and box together use up ' + (9 - e.kk) + ' digits and leave only ' + e.v + ': the cell has no choice at all, so filling it is deduction, not guessing.'}
        : forced
        ? {zh:'(' + e.r + ',' + e.c + ') 的 row、col、box 三個 bitmask OR 起來只漏掉 ' + e.v + '，k = 1，填 ' + e.v + '。每填一個數字，同列、同欄、同宮的空格就各少一個候選——連鎖反應讓新的 k = 1 格子不斷出現，這正是 MRV 要搶先利用的：只要有 forced 的格子，就永遠不必猜。',
           en:'For (' + e.r + ',' + e.c + ') the OR of its row, column and box masks leaves only ' + e.v + ' uncovered: k = 1, so ' + e.v + ' goes in. Each digit written removes one candidate from every empty cell in the same row, column and box, and that chain reaction keeps producing new k = 1 cells. MRV exists to grab them first: as long as a forced cell exists, there is never a reason to guess.'}
        : {zh:'(' + e.r + ',' + e.c + ') 是目前最少的，但也有 ' + e.kk + ' 個選擇，只好從最小的 ' + e.v + ' 開始試；錯了會回來換下一個。',
           en:'(' + e.r + ',' + e.c + ') is the smallest choice available but still has ' + e.kk + ' options, so it tries the lowest, ' + e.v + ', and will come back for the next one if that fails.'}});
  });
  const done = MRV.ev.find(e => e.k === 'done');
  F.push({shapes:stage(done.g, {solved:true,
      lines:[{zh:'empty 空了 → return True', en:'empty is exhausted -> return True'},
             {zh:MRV.nodes + ' 個節點 = ' + picks.length + ' 次填數 + 最後一次呼叫', en:MRV.nodes + ' nodes = ' + picks.length + ' fills + the final call'},
             {zh:'最大 k = ' + maxK + (maxK === 1 ? '：一次都沒有猜' : ''), en:'largest k = ' + maxK + (maxK === 1 ? ': not a single guess' : '')}],
      cap:{zh:'解完：' + MRV.nodes + ' 個節點，沒有任何一次回頭', en:'solved in ' + MRV.nodes + ' nodes without a single step back'}, capc:COL.tealL,
      cap2:{zh:'固定順序的樸素版要 ' + fmt(NAIVE.nodes) + ' 個節點（切到另一個版本看）', en:'the naive fixed order needs ' + fmt(NAIVE.nodes) + ' nodes (see the other variant)'}}),
    panels:[{lbl:{zh:'節點數', en:'nodes'}, chips:[chip('MRV ' + MRV.nodes, 'ok'), chip('naive ' + fmt(NAIVE.nodes), 'bad')]},
            {lbl:{zh:'第 0 列', en:'row 0'}, chips:[chip(done.g[0].join(' '), 'ok')]}],
    line:2,
    msg:{zh:'第 ' + MRV.nodes + ' 次呼叫發現 empty 已經空了，回傳 True。' + picks.length + ' 個空格、' + MRV.nodes + ' 個節點' + (maxK === 1 ? '，每一步的 k 都是 1：MRV 在這個盤面上從頭到尾沒有猜過一次，也從來沒有回溯。' : '。') + '同一個盤面用「第一個空格、1 到 9」的固定順序要走 ' + fmt(NAIVE.nodes) + ' 個節點，是 ' + Math.round(NAIVE.nodes / MRV.nodes) + ' 倍。差別不在每個節點算得快不快，而在於先處理最受限的格子：錯誤在樹的淺處就曝光，而不是填了一大堆之後才發現。',
         en:'Call number ' + MRV.nodes + ' finds the empty list exhausted and returns True. ' + picks.length + ' empty cells, ' + MRV.nodes + ' nodes' + (maxK === 1 ? ', and k was 1 at every step: on this board MRV never guessed once and never backtracked.' : '.') + ' The same board in the fixed "first empty cell, digits 1 to 9" order needs ' + fmt(NAIVE.nodes) + ' nodes, ' + Math.round(NAIVE.nodes / MRV.nodes) + ' times as many. The difference is not how fast each node is, it is handling the most constrained cell first, so that a mistake shows up near the top of the tree instead of after a long run of fills.'}});
  return F.list;
}

function naiveFrames(F, stage, maskRows, used){
  let undos = [];
  const undoTxt = () => undos.length
    ? {zh:'（先 unchoose：' + undos.map(u => '(' + u.r + ',' + u.c + ') 的 ' + u.v).join('、') + ' 拿回來）',
       en:'(first unchoose: ' + undos.map(u => u.v + ' comes back out of (' + u.r + ',' + u.c + ')').join(', ') + ') '}
    : {zh:'', en:''};
  F.push({shapes:stage(GIVEN, {hot:[0, 2], hotSt:'act',
      lines:[{zh:'依閱讀順序找第一個空格', en:'take the first empty cell in reading order'},
             {zh:'數字 1..9 逐一用 safe() 檢查', en:'check digits 1..9 in turn with safe()'}],
      cap:{zh:'同一個盤面，改用教科書的固定順序', en:'the same board with the textbook fixed order'}, capc:COL.orangeL,
      cap2:{zh:'這裡只播前 60 個節點，最後一格是完整跑完的總數', en:'only the first 60 nodes are replayed; the last frame is the full count'}}),
    panels:[{lbl:{zh:'已走節點', en:'nodes so far'}, chips:[chip('0', 'dim')]}], line:3,
    msg:{zh:'教科書版的數獨求解器：每個節點從左上角往右下找第一個空格，數字從 1 試到 9，第一個 safe 的就填下去往下遞迴。它完全正確，只是不管哪一格有幾個選擇——一個只能填 1 種數字的格子，和一個能填 5 種的格子，會被一視同仁地按位置處理。',
         en:'The textbook Sudoku solver: at every node find the first empty cell scanning from the top left, try digits 1 to 9, and recurse on the first one that is safe. It is perfectly correct; it just pays no attention to how many options a cell has - a cell that admits one digit and a cell that admits five are treated alike, purely by position.'}});
  NAIVE.ev.forEach(e => {
    if (e.k === 'undo'){ undos.push(e); return; }
    const [a, b, d] = used(e.g, e.r, e.c);
    const legal = 9 - new Set([...a, ...b, ...d]).size;
    const u = undoTxt();
    if (e.k === 'place'){
      const side = maskRows(e.g, e.r, e.c, {title:{zh:'試過', en:'tried'}, tcol:COL.orangeL,
        st:x => x === e.v ? 'hot' : e.rej.indexOf(x) >= 0 ? 'bad' : e.failed.indexOf(x) >= 0 ? 'done' : 'ghost'});
      const rejTxt = e.rej.length ? pyList(e.rej) : '', rejC = '[' + e.rej.join(',') + ']';
      F.push({shapes:stage(e.g, {hot:[e.r, e.c], hotLab:e.v,
          lines:[{zh:'(' + e.r + ',' + e.c + ') 填 ' + e.v, en:'(' + e.r + ',' + e.c + ') gets ' + e.v},
                 e.rej.length ? {zh:'被擋掉：' + rejC, en:'blocked: ' + rejC} : {zh:'比它小的數字沒有被擋', en:'no smaller digit was blocked'},
                 e.failed.length ? {zh:'試過但失敗：' + pyList(e.failed), en:'tried and failed below: ' + pyList(e.failed)} : {zh:'節點 ' + e.nodes, en:'node ' + e.nodes}],
          cap:{zh:'節點 ' + e.nodes + '：(' + e.r + ',' + e.c + ') 填 ' + e.v, en:'node ' + e.nodes + ': (' + e.r + ',' + e.c + ') gets ' + e.v},
          capc:COL.orangeL,
          cap2:{zh:'這格其實有 ' + legal + ' 個合法數字，固定順序不管這個', en:'this cell has ' + legal + ' legal digit' + (legal === 1 ? '' : 's') + ' - the fixed order does not care'}}, side),
        panels:[{lbl:{zh:'已走節點', en:'nodes so far'}, chips:[chip(String(e.nodes), 'dim')]},
                {lbl:{zh:'這一格', en:'this cell'}, chips:[chip('(' + e.r + ',' + e.c + ') = ' + e.v, 'hot')]}],
        line:8,
        msg:{zh:u.zh + '(' + e.r + ',' + e.c + ') 是閱讀順序上第一個空格。' + (e.rej.length ? '比 ' + e.v + ' 小的 ' + rejTxt + ' 已經出現在同列、同欄或同宮，safe() 回傳 False。' : '') + (e.failed.length ? pyList(e.failed) + ' 雖然 safe，但填下去之後更深處無解，已經拿回來了。' : '') + e.v + ' 通過檢查就先填下去，進入下一個節點。這裡沒有任何「這一格有幾個選擇」的判斷，所以一個錯誤可能要等到填了好幾格之後才會曝光。',
             en:u.en + '(' + e.r + ',' + e.c + ') is the first empty cell in reading order. ' + (e.rej.length ? 'The smaller digits ' + rejTxt + ' already appear in its row, column or box, so safe() says no. ' : '') + (e.failed.length ? pyList(e.failed) + ' were safe, but led to a dead end further down and have been taken back. ' : '') + e.v + ' passes, so it goes in and the search moves to the next node. Nothing here asks how many options the cell has, so a wrong digit may only be exposed several fills later.'}});
    } else {
      const side = maskRows(e.g, e.r, e.c, {title:{zh:'試過', en:'tried'}, tcol:COL.red,
        st:x => e.rej.indexOf(x) >= 0 ? 'bad' : e.failed.indexOf(x) >= 0 ? 'done' : 'ghost'});
      F.push({shapes:stage(e.g, {hot:[e.r, e.c], hotSt:'bad',
          lines:[{zh:'(' + e.r + ',' + e.c + ')：1..9 沒有一個能用', en:'(' + e.r + ',' + e.c + '): none of 1..9 works'},
                 {zh:'被擋：' + pyList(e.rej), en:'blocked: ' + pyList(e.rej)},
                 {zh:'試過但失敗：' + (e.failed.length ? pyList(e.failed) : '—'), en:'tried and failed: ' + (e.failed.length ? pyList(e.failed) : '-')}],
          cap:{zh:'死路：(' + e.r + ',' + e.c + ') 無數字可填，return False', en:'dead end: nothing fits (' + e.r + ',' + e.c + '), return False'}, capc:COL.red,
          cap2:{zh:'錯的其實是更早填下去的某個數字', en:'the real mistake is a digit written earlier'}}, side),
        panels:[{lbl:{zh:'已走節點', en:'nodes so far'}, chips:[chip(String(e.nodes), 'dim')]},
                {lbl:{zh:'死路', en:'dead end'}, chips:[chip('(' + e.r + ',' + e.c + ')', 'bad')]}],
        line:12,
        msg:{zh:u.zh + '(' + e.r + ',' + e.c + ') 的 9 個數字全部用不了' + (e.failed.length ? '（' + pyList(e.failed) + ' 在更深處失敗，其餘已出現在同列、同欄或同宮）' : '——每一個都已經出現在同列、同欄或同宮') + '，go() 回傳 False。問題不在這一格，而在上面某個早就填下去的數字；可是固定順序得一層一層退回去、每一層把剩下的數字試完，才找得到它。MRV 會在一開始就注意到這一格只剩很少的選擇。',
             en:u.en + 'None of the nine digits fits (' + e.r + ',' + e.c + ')' + (e.failed.length ? ' (' + pyList(e.failed) + ' failed further down, the rest already appear in its row, column or box)' : ' - every one already appears in its row, column or box') + ', so go() returns False. The fault is not in this cell but in some digit written earlier, and the fixed order can only find it by retreating one level at a time and exhausting the remaining digits at each. MRV would have noticed early on that this cell had almost no options.'}});
    }
    undos = [];
  });
  F.push({shapes:stage(NAIVE.g, {solved:true,
      lines:[{zh:'完整跑完：' + fmt(NAIVE.nodes) + ' 個節點', en:'run to the end: ' + fmt(NAIVE.nodes) + ' nodes'},
             {zh:'MRV：' + MRV.nodes + ' 個節點，約 ' + Math.round(NAIVE.nodes / MRV.nodes) + ' 倍差距', en:'MRV: ' + MRV.nodes + ' nodes, about ' + Math.round(NAIVE.nodes / MRV.nodes) + 'x fewer'},
             {zh:'第一次死路出現在節點 ' + NAIVE.firstDead.node, en:'first dead end at node ' + NAIVE.firstDead.node}],
      cap:{zh:'同一個答案：固定順序 ' + fmt(NAIVE.nodes) + ' 個節點，MRV ' + MRV.nodes + ' 個', en:'same answer: fixed order ' + fmt(NAIVE.nodes) + ' nodes, MRV ' + MRV.nodes}, capc:COL.tealL,
      cap2:{zh:'這個數字是頁面載入時把樸素搜尋完整跑一遍算出來的', en:'this count comes from running the naive search to completion when the page loads'}}),
    panels:[{lbl:{zh:'節點數', en:'nodes'}, chips:[chip('naive ' + fmt(NAIVE.nodes), 'bad'), chip('MRV ' + MRV.nodes, 'ok')]},
            {lbl:{zh:'第 0 列', en:'row 0'}, chips:[chip(NAIVE.g[0].join(' '), 'ok')]}],
    line:13,
    msg:{zh:'前 60 個節點播到這裡；頁面在背後把同一個搜尋完整跑完：總共 ' + fmt(NAIVE.nodes) + ' 次 go() 呼叫才解出來，答案和 MRV 一模一樣。MRV 只要 ' + MRV.nodes + ' 次，差 ' + Math.round(NAIVE.nodes / MRV.nodes) + ' 倍。兩者的剪枝規則完全相同（同一個數字不能在同列、同欄、同宮出現兩次），差別只在分支的順序：先處理最受限的格子，死路就會在樹的淺處被發現。',
         en:'The replay stops after 60 nodes; behind the scenes the page ran the same search to the end: ' + fmt(NAIVE.nodes) + ' calls to go() to reach a solution identical to MRV\'s. MRV needed ' + MRV.nodes + ', a factor of ' + Math.round(NAIVE.nodes / MRV.nodes) + '. The pruning rule is the same in both - no digit twice in a row, column or box - and only the branching order differs: handle the most constrained cell first and dead ends are found near the top of the tree.'}});
  return F.list;
}

/* =======================================================================
 * Tab 5 - LeetCode 39: the start index is what keeps [2,2,3] from
 * reappearing as [2,3,2] and [3,2,2]
 * ======================================================================= */
const CODE_CS = [
  'def combination_sum_nodes(candidates, target, *, start_index=True):',
  '    cand = sorted(candidates)',
  '    res, path = [], []',
  '    nodes = 0',
  '',
  '    def go(start, remaining):',
  '        nonlocal nodes',
  '        nodes += 1',
  '        if remaining == 0:',
  '            res.append(path[:])',
  '            return',
  '        if remaining < 0:',
  '            return',
  '        for i in range(start if start_index else 0, len(cand)):',
  '            if cand[i] > remaining:',
  '                break                   # sorted: everything after is bigger',
  '            path.append(cand[i])        # choose',
  '            go(i, remaining - cand[i])  # explore (i, not i + 1: reuse allowed)',
  '            path.pop()                  # unchoose',
  '',
  '    go(0, target)',
  '    return res, nodes'
];
const CODE_CS_NOSTART = CODE_CS.slice();
CODE_CS_NOSTART[13] = '        for i in range(0, len(cand)):   # start_index=False: restart at 0';
const CS_CAND = [2, 3, 6, 7], CS_T = 7;

/* run the search for real; stubs are the break points - drawn in the tree
   but not counted, because break never calls go() */
function recordCS(si){
  const nodes = [], ev = [], path = [], res = [];
  let count = 0, pops = [];
  function go(start, rem, parent, x){
    count++;
    const id = nodes.length;
    nodes.push({id:id, parent:parent, rem:rem, val:x, path:path.slice(), start:start, num:count});
    let dup = false;
    if (rem === 0){
      const key = path.slice().sort((a, b) => a - b).join(',');
      dup = res.some(r => r.slice().sort((a, b) => a - b).join(',') === key);
      res.push(path.slice());
      nodes[id].sol = true; nodes[id].dup = dup;
    }
    ev.push({k:'enter', id:id, pops:pops, res:res.map(r => r.slice()), count:count, dup:dup});
    pops = [];
    if (rem <= 0) return;
    const lo = si ? start : 0;
    for (let i = lo; i < CS_CAND.length; i++){
      if (CS_CAND[i] > rem){
        const sid = nodes.length;
        nodes.push({id:sid, parent:id, stub:true, val:CS_CAND[i], rem:rem, path:path.slice()});
        ev.push({k:'brk', id:sid, par:id, i:i, pops:pops, res:res.map(r => r.slice()), count:count,
                 skipped:CS_CAND.slice(i + 1)});
        pops = [];
        break;
      }
      path.push(CS_CAND[i]);
      go(i, rem - CS_CAND[i], id, CS_CAND[i]);
      path.pop();
      pops.push(CS_CAND[i]);
    }
  }
  go(0, CS_T, null, null);
  layoutTree(nodes, .30, 6.15, 1.25, .95);
  return {nodes:nodes, ev:ev, res:res, count:count};
}
const CS = [recordCS(true), recordCS(false)];

function csFrames(v){
  const F = new Frames(), R = CS[v], N = R.nodes, other = CS[1 - v], nost = v === 1;
  const RAD = .21;
  function stage(seen, cur, stack, o){
    const sh = [fitT(4.9, .45, nost
        ? {zh:'每一層都從 cand[0] 重新開始：同一組數字換個順序又來一次', en:'every level restarts at cand[0]: the same numbers come back in another order'}
        : {zh:'下一層從 i 開始：只准用「不比自己小」的數', en:'the next level starts at i: only numbers no smaller than the last one'},
        {c:nost ? COL.red : COL.tealL, fs:.32})];
    const onStack = new Set(stack);
    N.forEach(n => {
      if (n.parent == null || !seen.has(n.id)) return;
      const p = N[n.parent];
      if (n.stub){
        sh.push(S.e(p.x, p.y + RAD, n.x, n.y - .12, {s:'bad', arrow:false, w:.03, dash:'.07 .06'}));
        return;
      }
      sh.push(S.e(p.x, p.y + RAD, n.x, n.y - RAD,
                  {s:onStack.has(n.id) ? 'act' : 'idle', arrow:false, w:.04,
                   lab:'+' + n.val, fs:.2, lx:(n.x < p.x ? -.15 : n.x > p.x ? .15 : .15), ly:0}));
    });
    N.forEach(n => {
      if (!seen.has(n.id)) return;
      if (n.stub){
        sh.push(S.t(n.x, n.y + .06, n.val + '>' + n.rem, {c:COL.red, fs:.19}));
        return;
      }
      const st = n.id === cur ? (n.dup ? 'bad' : 'hot') : n.sol ? (n.dup ? 'bad' : 'ok') : onStack.has(n.id) ? 'act' : 'done';
      sh.push(S.c(n.x, n.y, RAD, st, String(n.rem), {fs:.22}));
    });
    sh.push(S.t(.30, 5.20, {zh:'圈內 = remaining；紅字 = break 掉的數', en:'circle = remaining; red = the value that hit break'},
                {c:COL.grey, fs:.21, anchor:'start'}));
    /* right column: path, res, node counter */
    sh.push(S.t(6.55, 1.30, 'path', {c:COL.purpleL, fs:.26, anchor:'start'}));
    for (let i = 0; i < 3; i++){
      const has = i < o.path.length;
      sh.push(S.r(7.25 + i * .66, 1.0, .56, .46, has ? 'act' : 'ghost', has ? String(o.path[i]) : '', {fs:.26}));
    }
    sh.push(S.t(6.55, 2.05, 'res', {c:COL.tealL, fs:.26, anchor:'start'}));
    for (let j = 0; j < 4; j++){
      const has = j < o.res.length, r = has ? o.res[j] : null;
      const dup = has && o.res.slice(0, j).some(q => q.slice().sort().join() === r.slice().sort().join());
      sh.push(S.r(6.55, 2.25 + j * .56, 2.9, .44, has ? (dup ? 'bad' : 'ok') : 'ghost', has ? pyList(r) : '', {fs:.24}));
    }
    sh.push(S.t(6.55, 4.75, 'nodes', {c:COL.grey, fs:.24, anchor:'start'}));
    sh.push(S.t(7.55, 4.78, String(o.count), {c:COL.orangeL, fs:.36, anchor:'start'}));
    if (o.cap) sh.push(fitT(4.9, 5.66, o.cap, {c:o.capc || COL.pale, fs:.29}));
    if (o.cap2) sh.push(fitT(4.9, 6.10, o.cap2, {c:COL.grey, fs:.25}));
    return sh;
  }
  const panels = (stack, res, count) => [
    {lbl:{zh:'呼叫堆疊', en:'call stack'}, chips:stack.map((id, k) => chip('go(' + N[id].start + ', ' + N[id].rem + ')', k === stack.length - 1 ? 'hot' : 'act'))},
    {lbl:'res', chips:res.length ? res.map((r, j) => chip(pyList(r), res.slice(0, j).some(q => q.slice().sort().join() === r.slice().sort().join()) ? 'bad' : 'ok')) : [chip('[]', 'empty')]},
    {lbl:{zh:'節點數', en:'nodes'}, chips:[chip(String(count), 'dim')]}];
  const popTxt = pops => pops.length
    ? {zh:'（先 path.pop() 拿掉 ' + pops.join('、') + '）', en:'(first path.pop() removes ' + pops.join(', ') + ') '}
    : {zh:'', en:''};
  const seen = new Set();
  const stackOf = id => { const s = []; for (let n = N[id]; n; n = n.parent == null ? null : N[n.parent]) s.unshift(n.id); return s; };
  R.ev.forEach((e, ix) => {
    const n = N[e.id];
    if (e.k === 'brk'){
      const p = N[e.par];
      seen.add(e.id);
      const stack = stackOf(p.id), u = popTxt(e.pops);
      const skip = e.skipped.length ? pyList(e.skipped) : '';
      F.push({shapes:stage(seen, p.id, stack, {path:p.path, res:e.res, count:e.count,
          cap:{zh:'cand[' + e.i + '] = ' + n.val + ' > remaining ' + n.rem + ' → break', en:'cand[' + e.i + '] = ' + n.val + ' > remaining ' + n.rem + ' -> break'}, capc:COL.red,
          cap2:skip ? {zh:'後面的 ' + skip + ' 連看都不用看', en:'the rest, ' + skip + ', is never even looked at'}
                    : {zh:'這已經是最後一個候選', en:'this was the last candidate anyway'}}),
        panels:panels(stack, e.res, e.count), line:15,
        msg:{zh:u.zh + '還差 ' + n.rem + '，下一個候選是 ' + n.val + '，已經太大。cand 事先排過序，所以它後面' + (skip ? '的 ' + skip + ' 只會更大' : '也沒有別的數了') + '——用 break 直接結束這一層的迴圈，而不是 continue 一個一個試。這一步不呼叫 go()，所以不算節點；它是剪枝，把整排注定失敗的分支一次砍掉。',
             en:u.en + n.rem + ' is still missing and the next candidate, ' + n.val + ', is already too big. cand was sorted up front, so ' + (skip ? 'everything after it (' + skip + ') is bigger still' : 'there is nothing after it') + ' - break ends this level\'s loop outright instead of trying them one by one with continue. No go() call happens here, so it is not a node; it is pruning, cutting a whole row of doomed branches at once.'}});
      return;
    }
    seen.add(e.id);
    const stack = stackOf(e.id), u = popTxt(e.pops), path = n.path;
    let cap, cap2, msg, line;
    if (n.parent == null){
      line = 20;
      cap = {zh:'candidates = [2, 3, 6, 7]，target = 7，每個數可以重複用', en:'candidates = [2, 3, 6, 7], target = 7, each may be reused'};
      cap2 = nost ? {zh:'這一版拿掉 start index：每層都從 2 開始試', en:'this version drops the start index: every level tries from 2 again'}
                  : {zh:'答案是「組合」：[2,2,3] 和 [3,2,2] 算同一個', en:'answers are combinations: [2,2,3] and [3,2,2] are the same one'};
      msg = nost
        ? {zh:'同一個函式，把 start_index 關掉：for 迴圈每一層都從 0 開始，而不是從上一層選到的 i 開始。看起來只是少一個參數，程式也不會出錯；要看的是 res 最後會多出什麼。',
           en:'The same function with start_index switched off: every level\'s loop starts at 0 instead of at the i chosen one level up. It looks like one parameter fewer and nothing will raise; watch what ends up in res.'}
        : {zh:'LeetCode 39：從 [2, 3, 6, 7] 裡挑數字（可以重複挑）湊出 7。回溯樹的每個節點是一次 go(start, remaining) 呼叫，圈裡寫的是還差多少。兩個設計各管一件事：start 讓下一層只從 i 往後挑（i 而不是 i + 1，所以同一個數可以再用），這保證每個組合只會以「非遞減」的順序出現一次；排序加 break 則是剪枝。',
           en:'LeetCode 39: pick numbers from [2, 3, 6, 7], reusing any of them, so that they add up to 7. Every node of the tree is one go(start, remaining) call, and the circle shows how much is still missing. Two design choices each handle one job: start makes the next level pick only from index i onward (i, not i + 1, so a number can be reused), which guarantees each combination appears exactly once, in non-decreasing order; sorting plus break is the pruning.'};
    } else if (n.sol){
      line = 9;
      cap = n.dup ? {zh:pyList(path) + '：湊到 7，但這個組合已經在 res 裡了', en:pyList(path) + ': reaches 7, but this combination is already in res'}
                  : {zh:pyList(path) + '：剛好湊到 7 → res.append(path[:])', en:pyList(path) + ': exactly 7 -> res.append(path[:])'};
      cap2 = {zh:'第 ' + e.count + ' 個節點', en:'node ' + e.count};
      msg = n.dup
        ? {zh:u.zh + 'remaining 變成 0，所以 ' + pyList(path) + ' 被記進 res。可是它和前面的 ' + pyList(e.res.find(r => r.slice().sort().join() === path.slice().sort().join())) + ' 是同一包數字，只是順序不同。沒有 start index，樹會把同一個組合的每一種排列都走一遍——題目要的是組合，這就是重複答案；這些多出來的分支也白白增加了節點數。',
           en:u.en + 'remaining hits 0, so ' + pyList(path) + ' goes into res. But it is the same bag of numbers as ' + pyList(e.res.find(r => r.slice().sort().join() === path.slice().sort().join())) + ', just in another order. Without a start index the tree walks every ordering of the same combination - the problem asks for combinations, so this is a duplicate answer, and the extra branches cost extra nodes too.'}
        : {zh:u.zh + 'path 加起來剛好 7，remaining = 0，把 path 的副本 path[:] 放進 res 然後 return。用副本是因為 path 等一下還會被 pop 改掉（第一個分頁的 aliasing bug）。',
           en:u.en + 'path adds up to exactly 7, remaining = 0, so a copy, path[:], goes into res and the call returns. It must be a copy because path is about to be popped and changed (the aliasing bug from the first tab).'};
    } else {
      line = 17;
      const p = N[n.parent];
      cap = {zh:'path.append(' + n.val + ') → go(' + n.start + ', ' + n.rem + ')', en:'path.append(' + n.val + ') -> go(' + n.start + ', ' + n.rem + ')'};
      cap2 = nost ? {zh:'下一層又從 2 開始試', en:'the next level tries from 2 again'}
                  : {zh:'下一層只能從 ' + n.val + ' 往後挑', en:'the next level may only pick from ' + n.val + ' onward'};
      msg = nost
        ? {zh:u.zh + '選 ' + n.val + '，還差 ' + n.rem + '。因為沒有 start index，下一層的迴圈又從 cand[0] = 2 開始：' + (n.val > 2 ? '明明已經選過 ' + n.val + '，還可以回頭再選比它小的 2——這就是會產生 [' + n.val + ', 2, ...] 這種「倒著排」的組合的地方。' : '這一步和有 start index 的版本一樣，差別會在選到比 2 大的數之後出現。'),
           en:u.en + 'Take ' + n.val + ', ' + n.rem + ' still missing. With no start index the next loop starts again at cand[0] = 2: ' + (n.val > 2 ? 'having already taken ' + n.val + ', the search may go back and take the smaller 2 - this is exactly where orderings like [' + n.val + ', 2, ...] are born.' : 'so far this is identical to the start-index version; the difference appears once something bigger than 2 has been taken.')}
        : {zh:u.zh + '選 cand[' + n.start + '] = ' + n.val + '，還差 ' + n.rem + '。遞迴傳 i（' + n.start + '）下去，下一層只會從 ' + n.val + ' 往後挑：path 永遠是非遞減的，所以 [2,2,3] 只可能以這一種順序出現，不會再長出 [3,2,2]。',
           en:u.en + 'Take cand[' + n.start + '] = ' + n.val + ', ' + n.rem + ' still missing. The recursion passes i (' + n.start + ') down, so the next level only picks from ' + n.val + ' onward: path is always non-decreasing, and [2,2,3] can only ever appear in that one order - [3,2,2] never grows.'};
    }
    F.push({shapes:stage(seen, e.id, stack, {path:path, res:e.res, count:e.count, cap:cap, capc:n.sol ? (n.dup ? COL.red : COL.tealL) : COL.orangeL, cap2:cap2}),
      panels:panels(stack, e.res, e.count), line:line, msg:msg});
  });
  const dups = R.res.length - new Set(R.res.map(r => r.slice().sort().join())).size;
  F.push({shapes:stage(seen, -1, [], {path:[], res:R.res, count:R.count,
      cap:{zh:'回傳 ' + R.res.length + ' 個答案、' + R.count + ' 個節點', en:'returns ' + R.res.length + ' answers in ' + R.count + ' nodes'}, capc:nost ? COL.red : COL.tealL,
      cap2:nost ? {zh:'有 start index：' + other.res.length + ' 個答案、' + other.count + ' 個節點', en:'with the start index: ' + other.res.length + ' answers, ' + other.count + ' nodes'}
                : {zh:'拿掉 start index：' + other.res.length + ' 個答案、' + other.count + ' 個節點（另一個版本）', en:'without it: ' + other.res.length + ' answers, ' + other.count + ' nodes (the other variant)'}}),
    panels:panels([], R.res, R.count), line:21,
    msg:nost
      ? {zh:'回傳 ' + pyList(R.res.map(pyList)) + '，其中 ' + dups + ' 個是重複的排列，節點也從 ' + other.count + ' 個變成 ' + R.count + ' 個。程式一行錯誤都沒有，只是回答了另一個問題（排列而不是組合）。修法不是事後去重，而是把 start index 放回去：讓每個組合只有一條路走得到，重複的分支根本不會長出來。',
         en:'It returns ' + pyList(R.res.map(pyList)) + ', ' + dups + ' of which are reorderings, and the node count grows from ' + other.count + ' to ' + R.count + '. Nothing raised; the code simply answered a different question (orderings instead of combinations). The fix is not deduplicating afterwards but putting the start index back, so that each combination has exactly one path leading to it and the duplicate branches never grow.'}
      : {zh:'回傳 ' + pyList(R.res.map(pyList)) + '，' + R.count + ' 個節點。兩個機制各自的貢獻：start index 讓每個組合只有一條路（非遞減順序）；排序加 break 讓「太大了」的判斷一次砍掉整排候選。把 start index 拿掉（另一個版本）會得到 ' + other.res.length + ' 個答案、' + other.count + ' 個節點。',
         en:'It returns ' + pyList(R.res.map(pyList)) + ' after ' + R.count + ' nodes. Each mechanism does its own job: the start index gives every combination exactly one path (non-decreasing order), and sorting plus break lets one "too big" test cut a whole row of candidates. Drop the start index (the other variant) and you get ' + other.res.length + ' answers in ' + other.count + ' nodes.'}});
  return F.list;
}

/* =======================================================================
 * Tab 6 - LeetCode 79: the '#' mark must be taken back on the way out
 * ======================================================================= */
const CODE_79 = [
  'class Solution79:',
  '    def exist(self, board, word):',
  '        R, C = len(board), len(board[0])',
  '',
  '        def dfs(r, c, i):',
  '            if i == len(word):',
  '                return True',
  '            if not (0 <= r < R and 0 <= c < C) or board[r][c] != word[i]:',
  '                return False',
  "            saved, board[r][c] = board[r][c], '#'   # choose: mark as used",
  '            found = (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1)',
  '                     or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))',
  '            board[r][c] = saved                     # unchoose: ALWAYS',
  '            return found',
  '',
  '        return any(dfs(r, c, 0) for r in range(R) for c in range(C))'
];
const CODE_79BAD = [
  'def exist_no_restore(board, word):',
  '    g = [row[:] for row in board]',
  '    R, C = len(g), len(g[0])',
  '',
  '    def dfs(r, c, i):',
  '        if i == len(word):',
  '            return True',
  '        if not (0 <= r < R and 0 <= c < C) or g[r][c] != word[i]:',
  '            return False',
  "        g[r][c] = '#'                              # BUG: never put back",
  '        return (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1)',
  '                or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))',
  '',
  '    return any(dfs(r, c, 0) for r in range(R) for c in range(C))'
];
const WB = ['ABCE', 'SFCS', 'ADEE'];
/* run LC79 for real; restore=false is exist_no_restore */
function recordWS(word, restore, log){
  const g = WB.map(r => r.split('')), R = 3, C = 4, ev = [], stack = [];
  let calls = 0;
  const snap = () => g.map(r => r.slice());
  function dfs(r, c, i){
    calls++;
    if (i === word.length){
      if (log) ev.push({k:'done', r:r, c:c, i:i, g:snap(), stack:stack.slice(), calls:calls});
      return true;
    }
    const oob = !(r >= 0 && r < R && c >= 0 && c < C);
    if (oob || g[r][c] !== word[i]){
      if (log) ev.push({k:oob ? 'oob' : 'miss', r:r, c:c, i:i, ch:oob ? null : g[r][c], g:snap(), stack:stack.slice(), calls:calls});
      return false;
    }
    const saved = g[r][c];
    g[r][c] = '#';
    stack.push([r, c, i]);
    if (log) ev.push({k:'mark', r:r, c:c, i:i, ch:saved, g:snap(), stack:stack.slice(), calls:calls});
    const found = dfs(r + 1, c, i + 1) || dfs(r - 1, c, i + 1) || dfs(r, c + 1, i + 1) || dfs(r, c - 1, i + 1);
    if (restore){
      g[r][c] = saved;
      if (log) ev.push({k:'restore', r:r, c:c, i:i, ch:saved, found:found, g:snap(), stack:stack.slice(), calls:calls});
    } else if (log && !found) ev.push({k:'leave', r:r, c:c, i:i, ch:saved, g:snap(), stack:stack.slice(), calls:calls});
    stack.pop();
    return found;
  }
  let ok = false;
  for (let r = 0; r < R && !ok; r++) for (let c = 0; c < C && !ok; c++) if (dfs(r, c, 0)) ok = true;
  return {ok:ok, ev:ev, calls:calls, g:g};
}
const WS = [recordWS('CCB', true, true), recordWS('CCB', false, true)];
const WS_TABLE = ['ABCCED', 'SEE', 'ABCB', 'CCB'].map(w => ({w:w, good:recordWS(w, true).ok, bad:recordWS(w, false).ok}));
const pyB = b => b ? 'True' : 'False';

function wsFrames(v){
  const F = new Frames(), bad = v === 1, W = WS[v], word = 'CCB';
  const GX = .55, GY = 1.05, CELL = 1.0;
  function stage(e, o){
    const sh = [fitT(4.9, .45, bad
        ? {zh:"少了 board[r][c] = saved：'#' 永遠留在格子上", en:"no board[r][c] = saved: the '#' stays on the cell for good"}
        : {zh:"進入時標 '#'，離開時一定還原", en:"mark '#' on the way in, always restore it on the way out"},
        {c:bad ? COL.red : COL.tealL, fs:.32})];
    const onStack = new Set((e ? e.stack : []).map(q => q[0] * 4 + q[1]));
    const g = o.g;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++){
      const cur = e && e.r === r && e.c === c && e.k !== 'done' && e.k !== 'oob';
      const ch = g[r][c], marked = ch === '#';
      let st = marked ? (onStack.has(r * 4 + c) ? 'act' : 'bad') : 'idle';
      if (cur) st = e.k === 'miss' ? 'bad' : e.k === 'mark' ? 'hot' : e.k === 'restore' ? 'ok' : e.k === 'leave' ? 'bad' : st;
      sh.push(S.r(GX + c * CELL, GY + r * CELL, CELL - .1, CELL - .1, st, ch, {fs:.40, rx:.08}));
      if (marked) sh.push(S.t(GX + c * CELL + (CELL - .1) / 2, GY + r * CELL + CELL - .18, WB[r][c], {c:COL.grey, fs:.18}));
    }
    const st = e ? e.stack : [];
    for (let k = 1; k < st.length; k++){
      const a = st[k - 1], b = st[k];
      sh.push(S.e(GX + a[1] * CELL + .45, GY + a[0] * CELL + .45, GX + b[1] * CELL + .45, GY + b[0] * CELL + .45, {s:'act', w:.05}));
    }
    for (let c = 0; c < 4; c++) sh.push(S.t(GX + c * CELL + .45, GY - .14, String(c), {c:COL.grey, fs:.2}));
    for (let r = 0; r < 3; r++) sh.push(S.t(GX - .16, GY + r * CELL + .52, String(r), {c:COL.grey, fs:.2, anchor:'end'}));
    /* the word, with i marked */
    sh.push(S.t(5.35, 1.33, 'word', {c:COL.purpleL, fs:.26, anchor:'start'}));
    for (let i = 0; i < 3; i++){
      const done = e && i < (e.k === 'done' ? 3 : e.i), now = e && i === e.i && e.k !== 'done';
      sh.push(S.r(6.25 + i * .62, 1.02, .54, .5, now ? 'hot' : done ? 'ok' : 'ghost', word[i], {fs:.28}));
    }
    if (o.table){
      sh.push(S.t(5.35, 2.05, {zh:'其他字也這樣跑一次', en:'the same test on other words'}, {c:COL.grey, fs:.23, anchor:'start'}));
      sh.push(S.t(7.35, 2.45, {zh:'有還原', en:'restore'}, {c:COL.tealL, fs:.21}));
      sh.push(S.t(8.75, 2.45, {zh:'沒還原', en:'no restore'}, {c:COL.red, fs:.21}));
      WS_TABLE.forEach((t, j) => {
        const y = 2.62 + j * .5, diff = t.good !== t.bad;
        sh.push(S.t(5.35, y + .29, t.w, {c:diff ? COL.red : COL.pale, fs:.24, anchor:'start'}));
        sh.push(S.r(6.75, y, 1.2, .4, 'ok', pyB(t.good), {fs:.22}));
        sh.push(S.r(8.15, y, 1.2, .4, diff ? 'bad' : 'done', pyB(t.bad), {fs:.22}));
      });
    } else {
      sh.push(S.t(5.35, 2.05, {zh:'呼叫堆疊（已標 # 的格子）', en:'call stack (cells marked #)'}, {c:COL.grey, fs:.23, anchor:'start'}));
      const rows = st.map(q => ({lab:'dfs(' + q[0] + ', ' + q[1] + ', ' + q[2] + ')', s:'act'}));
      if (e && (e.k === 'miss' || e.k === 'oob' || e.k === 'done'))
        rows.push({lab:'dfs(' + e.r + ', ' + e.c + ', ' + e.i + ')', s:e.k === 'done' ? 'ok' : 'bad'});
      if (rows.length) rows[rows.length - 1].s = rows[rows.length - 1].s === 'act' ? 'hot' : rows[rows.length - 1].s;
      for (let k = 0; k < 4; k++){
        const rw = rows[k];
        sh.push(S.r(5.35, 2.30 + k * .52, 2.6, .42, rw ? rw.s : 'ghost', rw ? rw.lab : '', {fs:.22}));
      }
      sh.push(S.t(8.2, 2.62, {zh:'呼叫數', en:'calls'}, {c:COL.grey, fs:.21, anchor:'start'}));
      sh.push(S.t(8.2, 3.12, String(o.calls), {c:COL.orangeL, fs:.34, anchor:'start'}));
    }
    (o.lines || []).forEach((t, i) => sh.push(fitT(.55, 4.38 + i * .40, t, {c:i === 0 ? COL.pale : COL.grey, fs:.24, anchor:'start'})));
    if (o.cap) sh.push(fitT(4.9, 5.66, o.cap, {c:o.capc || COL.pale, fs:.29}));
    if (o.cap2) sh.push(fitT(4.9, 6.10, o.cap2, {c:COL.grey, fs:.25}));
    return sh;
  }
  const panels = e => [
    {lbl:{zh:'呼叫堆疊', en:'call stack'}, chips:e && e.stack.length ? e.stack.map((q, k) => chip('(' + q[0] + ',' + q[1] + ') i=' + q[2], k === e.stack.length - 1 ? 'hot' : 'act')) : [chip('—', 'empty')]},
    {lbl:{zh:"標成 '#' 的格子", en:"cells holding '#'"}, chips:(() => { const out = [];
      (e ? e.g : WB).forEach((row, r) => { for (let c = 0; c < 4; c++) if (row[c] === '#') out.push(chip('(' + r + ',' + c + ')', e && e.stack.some(q => q[0] === r && q[1] === c) ? 'act' : 'bad')); });
      return out.length ? out : [chip('—', 'empty')]; })()}];
  const at = (r, c) => '(' + r + ',' + c + ')';
  F.push({shapes:stage(null, {g:WB, calls:0,
      lines:[{zh:'從每一格出發試一次，沿上下左右走，一格不能用兩次', en:'start from every cell, step up/down/left/right, never reuse a cell'},
             {zh:'走過的格子暫時寫成 #，這就是「choose」', en:"a visited cell is temporarily overwritten with '#' - that is the choose step"}],
      cap:{zh:'LeetCode 79：board 裡找得到 "CCB" 嗎？', en:'LeetCode 79: can "CCB" be spelled on this board?'}, capc:COL.tealL,
      cap2:bad ? {zh:'這一版刪掉了還原那一行', en:'this version has the restore line deleted'}
               : {zh:'答案是 True：(1,2) → (0,2) → (0,1)', en:'the answer is True: (1,2) -> (0,2) -> (0,1)'}}),
    panels:panels(null), line:bad ? 13 : 15,
    msg:bad
      ? {zh:"同一題、同一個盤面，只少了一行：board[r][c] = saved。'#' 標記的用意是「這條路徑正在用這一格」，路徑退回去之後這個說法就不成立了——但沒人把它擦掉。看它會怎麼影響下一個起點。",
         en:"The same problem on the same board, missing one line: board[r][c] = saved. The '#' means \"the current path is using this cell\"; once the path backs out that statement is false - but nobody erases it. Watch what it does to the next starting cell."}
      : {zh:"Word Search 是格子上的回溯：從某一格出發，每一步往上下左右走，比對 word 的下一個字母。同一格不能重複用，所以進入一格時把它改成 '#'（choose），四個方向都試完之後改回原字母（unchoose）。'#' 只描述「目前這條路徑」，路徑一退就要跟著擦掉。",
         en:"Word Search is backtracking on a grid: start from a cell, step up, down, left or right, and match the next letter of the word. A cell cannot be used twice, so entering it overwrites it with '#' (choose), and once all four directions have been tried it gets its letter back (unchoose). The '#' describes only the current path, so it must be erased the moment the path backs out."}});
  W.ev.forEach(e => {
    let cap, cap2, msg, line, lines;
    const ch = e.ch;
    if (e.k === 'oob'){
      line = 8;
      cap = {zh:'dfs' + at(e.r, e.c) + ' 走出棋盤 → False', en:'dfs' + at(e.r, e.c) + ' walks off the board -> False'};
      lines = [{zh:'(' + e.r + ',' + e.c + ') 不在 3×4 的範圍內', en:at(e.r, e.c) + ' is outside the 3x4 board'}];
      msg = {zh:'往這個方向走會掉出棋盤，0 <= r < R and 0 <= c < C 不成立，直接回傳 False。這是最便宜的剪枝：不用看字母就知道不行。',
             en:'This direction leaves the board, 0 <= r < R and 0 <= c < C fails, and the call returns False at once - the cheapest pruning there is, no letter needs to be read.'};
    } else if (e.k === 'miss'){
      line = 8;
      const hash = ch === '#';
      cap = hash ? {zh:at(e.r, e.c) + " 是 '#'，不等於 '" + word[e.i] + "' → False", en:at(e.r, e.c) + " holds '#', not '" + word[e.i] + "' -> False"}
                 : {zh:at(e.r, e.c) + " 是 '" + ch + "'，要的是 '" + word[e.i] + "' → False", en:at(e.r, e.c) + " is '" + ch + "', need '" + word[e.i] + "' -> False"};
      const stale = hash && !e.stack.some(q => q[0] === e.r && q[1] === e.c);
      lines = [{zh:'第 ' + e.i + ' 個字母要 ' + word[e.i], en:'letter ' + e.i + ' must be ' + word[e.i]}];
      msg = stale
        ? {zh:"這就是 bug 發作的地方：" + at(e.r, e.c) + " 原本是 '" + WB[e.r][e.c] + "'，正好是需要的字母，但上一個起點走過它之後沒有還原，'#' 一直留著。現在沒有任何路徑在用它，它卻還是被當成「用過了」。真正的答案 (1,2) → (0,2) → (0,1) 就這樣被擋掉。",
           en:"This is where the bug bites: " + at(e.r, e.c) + " was '" + WB[e.r][e.c] + "', exactly the letter needed, but an earlier start walked over it and never restored it, so the '#' is still there. No path is using the cell now, yet it is still treated as taken - and the real answer (1,2) -> (0,2) -> (0,1) is blocked."}
        : hash
        ? {zh:"這一格是 '#'：它正在目前的路徑上。'#' 不等於任何字母，所以不必另外準備一個 visited 集合，比對字母那一步就順便擋掉了走回頭路。",
           en:"This cell holds '#': it is on the current path. '#' matches no letter, so no separate visited set is needed - the letter comparison already stops the path from doubling back."}
        : {zh:'字母不對，回傳 False。' + (e.i === 0 ? '這一格不能當起點，any() 換下一格。' : '這個方向走不通，換下一個方向。'),
           en:'Wrong letter, return False. ' + (e.i === 0 ? 'This cell cannot start the word, so any() moves to the next cell.' : 'This direction is a dead end; try the next one.')};
    } else if (e.k === 'mark'){
      line = 9;
      cap = {zh:at(e.r, e.c) + " = '" + ch + "' 對上第 " + e.i + " 個字母 → 標成 '#'", en:at(e.r, e.c) + " = '" + ch + "' matches letter " + e.i + " -> mark it '#'"};
      lines = [{zh:'接著依序試：下、上、右、左', en:'next, in order: down, up, right, left'}];
      msg = {zh:"'" + ch + "' 正是 word[" + e.i + "]，這一格加入路徑。先把它改成 '#' 再往四個方向遞迴，這樣更深的呼叫走回這一格時，比對會失敗，不會把同一格用兩次。",
             en:"'" + ch + "' is word[" + e.i + "], so the cell joins the path. It becomes '#' before the four recursive calls, so that a deeper call wandering back here fails the comparison instead of using the cell twice."};
    } else if (e.k === 'restore'){
      line = 12;
      cap = {zh:at(e.r, e.c) + " 還原成 '" + ch + "'，回傳 " + pyB(e.found), en:at(e.r, e.c) + " gets '" + ch + "' back, return " + pyB(e.found)};
      lines = [{zh:e.found ? '找到了，一路回傳 True' : '四個方向都不通', en:e.found ? 'found - True travels back up' : 'all four directions failed'}];
      msg = e.found
        ? {zh:"下面找到完整的字了。即使要回傳 True，還是先把 '" + ch + "' 寫回去：函式結束時 board 要和進來時一模一樣，呼叫者才不會看到被改過的盤面。",
           en:"The full word was found below. Even on the way out with True, '" + ch + "' is written back first: when the function returns the board must be exactly as it was on entry, so the caller never sees a modified board."}
        : {zh:"從 " + at(e.r, e.c) + " 出發的四個方向都失敗，所以這一格不在答案路徑上。把 '" + ch + "' 寫回去——之後別的路徑可能還需要經過這一格。",
           en:"All four directions from " + at(e.r, e.c) + " failed, so this cell is not on an answer path. '" + ch + "' is written back, because a later path may still need to pass through it."};
    } else if (e.k === 'leave'){
      line = 10;
      cap = {zh:at(e.r, e.c) + " 失敗離開，但 '#' 沒有擦掉", en:at(e.r, e.c) + " fails and leaves - the '#' stays"};
      lines = [{zh:"原本是 '" + ch + "'，現在永遠是 '#'", en:"it was '" + ch + "', now it is '#' for good"}];
      msg = {zh:"四個方向都不通，函式回傳 False，可是 " + at(e.r, e.c) + " 沒有改回 '" + ch + "'。紅色的格子就是這種「沒有路徑在用、卻還是被標記」的格子。程式不會出錯，它只是讓之後的搜尋在一張被弄髒的盤面上進行。",
             en:"All four directions fail and the call returns False, but " + at(e.r, e.c) + " never gets '" + ch + "' back. Red cells are exactly this: marked although no path uses them. Nothing raises; every later search just runs on a dirtied board."};
    } else {
      line = 6;
      cap = {zh:'i == len(word) → True', en:'i == len(word) -> True'};
      lines = [{zh:'三個字母全部對上', en:'all three letters matched'}];
      msg = {zh:'i 等於 3，代表 word 的每個字母都已經對上了：C (1,2) → C (0,2) → B (0,1)。True 沿著呼叫堆疊一路往上傳，any() 立刻停止。',
             en:'i equals 3, meaning every letter of the word has been matched: C (1,2) -> C (0,2) -> B (0,1). True travels up the call stack and any() stops at once.'};
    }
    F.push({shapes:stage(e, {g:e.g, calls:e.calls, lines:lines, cap:cap, cap2:{zh:'第 ' + e.calls + ' 次 dfs 呼叫', en:'dfs call ' + e.calls},
        capc:e.k === 'done' || e.k === 'restore' ? COL.tealL : e.k === 'mark' ? COL.orangeL : COL.red}),
      panels:panels(e), line:line, msg:msg});
  });
  const same = WS_TABLE.filter(t => t.good === t.bad).map(t => t.w);
  F.push({shapes:stage(null, {g:W.g, calls:W.calls, table:bad,
      lines:bad ? [{zh:'只有 CCB 露出 bug：它要回頭用前一個起點碰過的格子', en:'only CCB exposes the bug: it needs a cell an earlier start touched'}]
                : [{zh:'盤面和一開始一模一樣', en:'the board is exactly as it started'}],
      cap:bad ? {zh:'回傳 False，正確答案是 True', en:'returns False; the right answer is True'}
              : {zh:'回傳 True，共 ' + W.calls + ' 次 dfs 呼叫', en:'returns True after ' + W.calls + ' dfs calls'},
      capc:bad ? COL.red : COL.tealL,
      cap2:bad ? {zh:same.join('、') + ' 在沒還原的版本也答對', en:same.join(', ') + ' still come out right without the restore'}
               : {zh:'沒還原的版本：' + pyB(WS[1].ok), en:'the no-restore version: ' + pyB(WS[1].ok)}}),
    panels:panels(null), line:bad ? 13 : 15,
    msg:bad
      ? {zh:'回傳 False，但 C (1,2) → C (0,2) → B (0,1) 明明存在。第一個起點 (0,2) 把 (0,2) 和 (1,2) 都標成 # 然後失敗離開，沒有擦掉；輪到從 (1,2) 出發時，它已經是 #，連第一個字母都對不上。更危險的是右邊這張表：LeetCode 範例的 ' + same.join('、') + ' 在沒還原的版本都答對了，因為它們的答案路徑剛好不需要回頭用別人碰過的格子。範例全過不代表程式對——回溯的 bug 要專門構造「兩條路徑共用格子」的測資才抓得到。',
         en:'It returns False, yet C (1,2) -> C (0,2) -> B (0,1) plainly exists. The first start, (0,2), marked both (0,2) and (1,2) with # and left without erasing them; by the time (1,2) gets its turn as a start it is already #, and not even the first letter matches. The table on the right is the more dangerous part: the LeetCode examples ' + same.join(', ') + ' all come out right without the restore, because their answer paths happen never to reuse a cell someone else touched. Passing every example does not make the code right - a backtracking bug like this only shows up on a test built so that two paths share a cell.'}
      : {zh:'回傳 True，' + W.calls + ' 次 dfs 呼叫。注意結束時盤面上一個 # 都沒有：每一次 choose 都配了一次 unchoose，不管那條路成功還是失敗。把還原那一行刪掉（另一個版本），同一個盤面會回傳 False。',
         en:'It returns True after ' + W.calls + ' dfs calls. Note that not a single # is left on the board: every choose was paired with an unchoose, whether the path succeeded or not. Delete the restore line (the other variant) and the same board returns False.'}});
  return F.list;
}

/* ======================================================================== */
const curV = () => (typeof varIx !== 'undefined' ? varIx : 0);
const DAY_META = {
  title:{zh:'Day 39 — 回溯與剪枝：N-Queens、數獨',
         en:'Day 39 - Backtracking and pruning: N-Queens and Sudoku'},
  sub:{zh:'回溯就是 choose、explore、unchoose 三行：剪枝決定它跑不跑得完，unchoose 決定它對不對——而它的 bug 從來不當機。',
       en:'Backtracking is three lines - choose, explore, unchoose. Pruning decides whether it finishes, unchoose decides whether it is right, and its bugs never crash.'},
  tabs:[
    {
      id:'subsets', label:{zh:'模板與 path[:]', en:'the template and path[:]'},
      stage:{zh:'列出 [1, 2, 3] 的全部子集合：一個 path 在整棵樹上共用',
             en:'all subsets of [1, 2, 3]: one path shared by the whole tree'},
      view:VIEW,
      variants:[{zh:'path[:]（8 個子集合）', en:'path[:] - 8 subsets'},
                {zh:'path（aliasing bug）', en:'path - the aliasing bug'}],
      idea:{zh:'回溯的模板只有三行：path.append 是 choose、遞迴是 explore、path.pop 是 unchoose。整棵遞迴樹共用同一個 path，往下走的時候長一格，回來的時候縮一格，所以任何時刻 path 都剛好描述「從根走到目前節點」的那條路。這個共用是回溯省記憶體的原因，也是它最常見 bug 的來源：res.append(path) 存的不是當下的內容，而是 path 這個物件本身。8 次 append 放進去的是 8 個指向同一個 list 的參照，等搜尋結束 path 被 pop 回空的，print(res) 就印出 8 個 []。程式不會報錯，長度也對，只有內容錯。path[:] 在記錄的那一刻複製一份，之後 path 怎麼改都影響不到它。',
            en:'The backtracking template is three lines: path.append is choose, the recursive call is explore, path.pop is unchoose. The whole recursion tree shares one path, which grows by one on the way down and shrinks by one on the way back, so at any moment it describes exactly the route from the root to the current node. That sharing is why backtracking is cheap on memory, and it is also the source of its most common bug: res.append(path) stores not the current contents but the path object itself. Eight appends put eight references to one list into res, and once the search has popped path back to empty, print(res) shows eight empty lists. Nothing raises and the length is right; only the contents are wrong. path[:] copies the list at the moment it is recorded, so later changes to path cannot reach it.'},
      legend:[['#ff9736', {zh:'目前的呼叫 / 剛記下的一格', en:'the current call / the slot just recorded'}],
              ['#9d6bff', {zh:'還在呼叫堆疊上', en:'still on the call stack'}],
              ['#3fe0dd', {zh:'已記錄的副本', en:'a recorded copy'}],
              ['#ff5c5c', {zh:'被 8 個參照共用的同一個 list', en:'the one list all eight references share'}]],
      get code(){ return curV() === 1 ? CODE_SUB_BAD : CODE_SUB; },
      build:subFrames
    },
    {
      id:'queens', label:{zh:'N 皇后', en:'N-Queens'},
      stage:{zh:'一列放一顆，三個 set 讓「會不會被吃」變成 O(1) 的檢查',
             en:'one queen per row; three sets make "is it attacked?" an O(1) test'},
      view:VIEW,
      variants:[{zh:'n = 4（17 節點、2 解）', en:'n = 4 - 17 nodes, 2 solutions'},
                {zh:'n = 5（54 節點、10 解）', en:'n = 5 - 54 nodes, 10 solutions'},
                {zh:'n = 6（153 節點、4 解）', en:'n = 6 - 153 nodes, 4 solutions'}],
      idea:{zh:'N 皇后一列只放一顆，所以「同一列」的衝突從結構上就不存在，剩下欄、主對角線、副對角線三種。同一條主對角線上 r − c 都一樣，同一條副對角線上 r + c 都一樣，所以用三個 set 記住已被佔的 c、r − c、r + c，檢查一個格子是不是安全就是三次 set 查詢。剪枝的威力在於它發生得早：第 2 列的某個位置一被吃掉，底下所有「第 2 列放這裡」的擺法全部不用看。8 皇后如果每列隨便放要看 8⁸ ≈ 1,677 萬種，回溯只走 2,057 個節點就找到全部 92 個解。每放一顆要把三個 set 各加一筆，拿走時也要各刪一筆——少刪一筆，下一個分頁就是後果。',
            en:'N-Queens puts exactly one queen in each row, so a same-row conflict is impossible by construction and three kinds remain: column, diagonal and anti-diagonal. Every square on one diagonal shares r - c and every square on one anti-diagonal shares r + c, so three sets remember the occupied c, r - c and r + c, and testing a square is three set lookups. The power of pruning is that it happens early: the moment a square in row 2 is attacked, every placement that would have put a queen there is skipped wholesale. Placing 8 queens row by row without checks means 8^8, about 16.8 million boards; backtracking visits 2,057 nodes and finds all 92 solutions. Each placement adds one entry to each of the three sets and each removal must delete one from each - forget a delete and the next tab shows what happens.'},
      legend:[['#ff9736', {zh:'剛放下的皇后', en:'the queen just placed'}],
              ['#9d6bff', {zh:'目前路徑上的皇后', en:'queens on the current path'}],
              ['#ff5c5c', {zh:'被吃掉的格子與攻擊者', en:'attacked squares and their attacker'}],
              ['#3fe0dd', {zh:'找到的解', en:'a solution'}]],
      code:CODE_Q,
      build:v => queenFrames([4, 5, 6][v], QRUN[[4, 5, 6][v]], false)
    },
    {
      id:'noundo', label:{zh:'忘了 unchoose', en:'the missing unchoose'},
      stage:{zh:'少刪一次 set：拿走的皇后還在擋路',
             en:'one set removal missing: queens that were taken back still block squares'},
      view:VIEW,
      idea:{zh:'把 N 皇后的 unchoose 拿掉一半：board.pop() 還在，cols、diag、anti 三個 set 卻沒有對應的 remove。棋盤上看起來皇后已經拿走了，三個 set 卻還記得它，這些「幽靈皇后」繼續攻擊它們原本的欄和對角線。每走錯一次，能放的格子就少一些，很快每一列都被幽靈佔滿。8 × 8 的棋盤上，這個版本總共只走進 9 個盤面，然後宣布無解——正確答案是 92 個。它不會當機，不會丟例外，只是很有自信地回傳 0。回溯的正確性完全建立在一個對稱上：每一次 choose 改了什麼，unchoose 就要一樣不差地改回來。',
            en:'Remove half of the N-Queens unchoose: board.pop() stays, but the matching removals from cols, diag and anti are gone. On the board the queen looks gone, yet the three sets still remember it, and these ghost queens keep attacking their old column and diagonals. Every wrong turn leaves fewer usable squares, and soon every row is covered by ghosts. On an 8 x 8 board this version enters just 9 boards and then declares there is no solution - the right answer is 92. It does not crash and raises nothing; it confidently returns 0. The correctness of backtracking rests entirely on one symmetry: whatever choose changed, unchoose must change back, exactly.'},
      legend:[['#ff9736', {zh:'剛放下的皇后', en:'the queen just placed'}],
              ['#9d6bff', {zh:'目前路徑上的皇后', en:'queens on the current path'}],
              ['#ff5c5c', {zh:'幽靈皇后：已拿走卻還在 set 裡', en:'ghost queens: removed from the board, still in the sets'}],
              ['#2f5661', {zh:'被擋掉的格子', en:'blocked squares'}]],
      code:CODE_QBAD,
      build:() => queenFrames(8, QBAD, true)
    },
    {
      id:'sudoku', label:{zh:'數獨 MRV', en:'Sudoku and MRV'},
      stage:{zh:'LeetCode 37 範例：同樣的剪枝，只換「先填哪一格」',
             en:'the LeetCode 37 example: the same pruning, only the cell order changes'},
      view:VIEW,
      variants:[{zh:'MRV（52 節點）', en:'MRV - 52 nodes'},
                {zh:'固定順序（4,209 節點）', en:'fixed order - 4,209 nodes'}],
      idea:{zh:'數獨的回溯和 N 皇后同一個模板：選一個空格、填一個不衝突的數字、遞迴、失敗就拿回來。衝突檢查用三組 bitmask：rows[r]、cols[c]、boxes[b] 的第 v 個 bit 表示數字 v 已經用過，三個 OR 起來取反就是這一格還能填的數字。真正拉開差距的是分支順序。教科書版永遠填「閱讀順序上第一個空格」，LeetCode 37 的範例要走 4,209 個節點；MRV（minimum remaining values）每次挑合法數字最少的空格，同一個盤面只要 52 個節點，而且每一步都只有一個選擇——完全沒有猜。道理很直接：選擇越少的格子越可能暴露錯誤，先處理它，錯誤就在樹的淺處被發現，而不是在填了一大堆之後才發現。',
            en:'Sudoku backtracking uses the same template as N-Queens: pick an empty cell, write a digit that does not clash, recurse, take it back on failure. Clashes are tracked with three groups of bitmasks: bit v of rows[r], cols[c] or boxes[b] means digit v is already used, and the complement of the three OR-ed together is what the cell can still take. What actually separates the two versions is branching order. The textbook solver always fills the first empty cell in reading order and needs 4,209 nodes on the LeetCode 37 example; MRV - minimum remaining values - always picks the empty cell with the fewest legal digits and solves the same board in 52 nodes, with exactly one option at every step: no guessing at all. The reason is simple: the cell with the fewest options is the one most likely to expose a mistake, and handling it first exposes mistakes near the top of the tree rather than after a long run of fills.'},
      legend:[['#ff9736', {zh:'這一步填的格子', en:'the cell filled at this step'}],
              ['#9d6bff', {zh:'forced：只剩一個數字', en:'forced - one digit left'}],
              ['#3fe0dd', {zh:'還能填的數字 / 解完', en:'digits still allowed / solved'}],
              ['#ff5c5c', {zh:'死路 / 被擋掉的數字', en:'dead end / a blocked digit'}]],
      get code(){ return curV() === 1 ? CODE_NAIVE : CODE_MRV; },
      build:sudokuFrames
    },
    {
      id:'lc39', label:{zh:'LC 39 組合總和', en:'LC 39 combination sum'},
      stage:{zh:'candidates = [2, 3, 6, 7]、target = 7：start index 讓每個組合只出現一次',
             en:'candidates = [2, 3, 6, 7], target = 7: the start index lets each combination appear once'},
      view:VIEW,
      variants:[{zh:'有 start index（2 個答案）', en:'with start index - 2 answers'},
                {zh:'沒有 start index（4 個答案）', en:'no start index - 4 answers'}],
      idea:{zh:'Combination Sum 有兩個要點。第一個是 start index：遞迴時把目前的 i 傳下去，下一層只從 i 往後挑，於是 path 永遠是非遞減的，[2,2,3] 只會以這一個順序出現。拿掉它，每一層都從頭開始挑，[2,3,2]、[3,2,2] 也會長出來——題目要組合，你交出的卻是排列，節點數也從 10 變成 13。傳 i 而不是 i + 1，是因為同一個數字可以重複用。第二個是排序加 break：候選數由小到大排好，一旦 cand[i] 超過還差的數，後面的只會更大，用 break 整排跳過，而不是 continue 一個一個試。這兩個要點管的是不同的事：start index 管正確性（不重複），break 管效率（剪枝）。',
            en:'Combination Sum hinges on two things. The first is the start index: the current i is passed down, the next level only picks from i onward, so path is always non-decreasing and [2,2,3] can appear in that one order only. Remove it and every level picks from the beginning again, so [2,3,2] and [3,2,2] grow as well - the problem asks for combinations and you hand back orderings, while the node count rises from 10 to 13. It is i rather than i + 1 because a number may be reused. The second is sorting plus break: with the candidates in ascending order, once cand[i] exceeds what is still missing everything after it is bigger, so break skips the whole row instead of trying each with continue. The two handle different things: the start index is about correctness (no duplicates), break is about speed (pruning).'},
      legend:[['#ff9736', {zh:'目前的呼叫', en:'the current call'}],
              ['#9d6bff', {zh:'還在呼叫堆疊上 / path', en:'on the call stack / path'}],
              ['#3fe0dd', {zh:'湊到 target 的組合', en:'a combination that hits the target'}],
              ['#ff5c5c', {zh:'break 掉的候選 / 重複答案', en:'a candidate cut by break / a duplicate'}]],
      get code(){ return curV() === 1 ? CODE_CS_NOSTART : CODE_CS; },
      build:csFrames
    },
    {
      id:'lc79', label:{zh:'LC 79 單字搜尋', en:'LC 79 word search'},
      stage:{zh:"在 ABCE / SFCS / ADEE 裡找 \"CCB\"：'#' 用完一定要擦掉",
             en:"find \"CCB\" in ABCE / SFCS / ADEE: the '#' must be erased after use"},
      view:VIEW,
      variants:[{zh:'有還原（True）', en:'with restore - True'},
                {zh:'沒還原（False）', en:'no restore - False'}],
      idea:{zh:"Word Search 把回溯搬到格子上：path 就是走過的格子，choose 是把目前這格改成 '#'，unchoose 是把原來的字母寫回去。用 '#' 取代 visited 集合很省事，因為 '#' 不等於任何字母，比對字母那一步就順便擋掉了走回頭路。但 '#' 的意思只是「目前這條路徑正在用這一格」，路徑退回去之後就必須擦掉。少了這一行，LeetCode 的三個範例 ABCCED、SEE、ABCB 全都照樣答對，因為它們的答案路徑剛好不需要經過別人碰過的格子；\"CCB\" 就不行：第一個起點 (0,2) 把 (0,2)、(1,2) 都標成 # 後失敗離開，輪到從 (1,2) 出發時它已經是 #，真正的答案被擋掉，回傳 False。範例全過不代表回溯寫對了。",
            en:"Word Search moves backtracking onto a grid: the path is the cells walked so far, choose overwrites the current cell with '#', and unchoose writes the letter back. Using '#' instead of a visited set is convenient because '#' matches no letter, so the letter comparison already stops the path from doubling back. But '#' only means \"the current path is using this cell\" and has to be erased as soon as the path backs out. Without that line, the three LeetCode examples ABCCED, SEE and ABCB still all come out right, because their answer paths never need a cell somebody else touched. \"CCB\" is different: the first start, (0,2), marks (0,2) and (1,2) with # and leaves after failing; when (1,2) gets its turn as a start it is already #, the real answer is blocked and the result is False. Passing the examples does not mean the backtracking is right."},
      legend:[['#ff9736', {zh:'剛標成 # 的格子', en:'the cell just marked #'}],
              ['#9d6bff', {zh:'目前路徑上的格子', en:'cells on the current path'}],
              ['#3fe0dd', {zh:'還原 / 找到', en:'restored / found'}],
              ['#ff5c5c', {zh:'字母不符 / 沒擦掉的 #', en:'wrong letter / a # never erased'}]],
      get code(){ return curV() === 1 ? CODE_79BAD : CODE_79; },
      build:wsFrames
    }
  ]
};
