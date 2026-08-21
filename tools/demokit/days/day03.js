// DAY: 03
// TITLE_ZH: 樹
// TITLE_EN: Tree
// SUB_ZH: 前序、中序、後序其實只差一行的位置；還有為什麼樹的高度非得用遞迴不可。
// SUB_EN: Preorder, inorder, postorder.
// FOLDER: day%2003%20-%20tree
// MEDIUM: https://medium.com/100-days-of-python/day-03-tree-fe4bfcb4c8e7

const A = [8, 3, 10, 1, 6, null, 14, null, null, 4, 7, null, null, 13];
const L = i => (2 * i + 1 < A.length && A[2 * i + 1] != null) ? 2 * i + 1 : -1;
const R = i => (2 * i + 2 < A.length && A[2 * i + 2] != null) ? 2 * i + 2 : -1;
const tree = (states, opt) => heapTreeShapes(A, .35, 1.15, 9.3, 1.42, states, opt || {});
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));

const CODE_DFS = [
'def walk(node):',
'    if node is None:',
'        return',
'    # 前序 preorder: 這裡 visit(node)',
'    walk(node.left)',
'    # 中序 inorder: 這裡 visit(node)',
'    walk(node.right)',
'    # 後序 postorder: 這裡 visit(node)'];
const CODE_BFS = [
'from collections import deque',
'',
'def level_order(root):',
'    q, out = deque([root]), []',
'    while q:',
'        node = q.popleft()',
'        out.append(node.val)',
'        if node.left:  q.append(node.left)',
'        if node.right: q.append(node.right)',
'    return out'];
const CODE_H = [
'def height(node):',
'    if node is None:',
'        return 0',
'    lh = height(node.left)',
'    rh = height(node.right)',
'    return 1 + max(lh, rh)'];

function buildDFS(v){
  const kind = ['pre', 'in', 'post'][v];
  const name = [{zh:'前序 preorder', en:'preorder'}, {zh:'中序 inorder', en:'inorder'},
                {zh:'後序 postorder', en:'postorder'}][v];
  const line = [3, 5, 7][v];
  const F = new Frames(), out = [], stack = [], st = {};
  const P = () => [
    {lbl:{zh:'呼叫堆疊（遞迴到哪裡）', en:'call stack'}, chips:chipsOf(stack.map(i => A[i]), 'act')},
    {lbl:{zh:'走訪結果', en:'visit order'}, chips:chipsOf(out, 'ok')}];
  const snap = (cur, ln, msg) => {
    const s = Object.assign({}, st); if (cur >= 0) s[cur] = 'hot';
    stack.forEach(i => { if (s[i] !== 'hot') s[i] = 'act'; });
    F.push({shapes:tree(s), panels:P(), line:ln, msg:msg});
  };
  F.push({shapes:tree({}), panels:P(), line:0,
    msg:{zh:'同一棵樹、同一段遞迴，<b>' + tr(name) + '</b>只是把 visit 放在不同的位置。往下看那一行怎麼影響順序。',
         en:'Same tree, same recursion: <b>' + tr(name) + '</b> merely moves the visit line. Watch how that one line changes everything.'}});
  (function go(i){
    if (i < 0) return;
    stack.push(i);
    snap(i, 0, {zh:'進到節點 <b>' + A[i] + '</b>（遞迴深度 ' + stack.length + '）。',
                en:'Entering node <b>' + A[i] + '</b> (recursion depth ' + stack.length + ').'});
    if (kind === 'pre'){ out.push(A[i]); st[i] = 'ok';
      snap(i, line, {zh:'<b>前序</b>：一進來就先記下 <b>' + A[i] + '</b>，再去看小孩。' +
                         '所以父節點永遠排在自己的子樹前面。',
                     en:'<b>Preorder</b>: record <b>' + A[i] + '</b> on arrival, before touching the children. A parent always precedes its subtree.'}); }
    go(L(i));
    if (kind === 'in'){ out.push(A[i]); st[i] = 'ok';
      snap(i, line, {zh:'<b>中序</b>：左子樹整個走完才記下 <b>' + A[i] +
                         '</b>。在二元搜尋樹上，這個順序剛好由小到大。',
                     en:'<b>Inorder</b>: record <b>' + A[i] + '</b> only after the whole left subtree. On a binary search tree this order comes out sorted.'}); }
    go(R(i));
    if (kind === 'post'){ out.push(A[i]); st[i] = 'ok';
      snap(i, line, {zh:'<b>後序</b>：兩邊小孩都處理完才輪到 <b>' + A[i] +
                         '</b>。需要「先算完子樹再合併」的事情都用它。',
                     en:'<b>Postorder</b>: <b>' + A[i] + '</b> comes last, after both children. Use it whenever a node needs its subtrees resolved first.'}); }
    stack.pop();
    if (stack.length)
      snap(-1, 0, {zh:'<b>' + A[i] + '</b> 這一層結束，回到父節點 <b>' + A[stack[stack.length - 1]] + '</b>。',
                   en:'Done with <b>' + A[i] + '</b>, returning to parent <b>' + A[stack[stack.length - 1]] + '</b>.'});
  })(0);
  F.push({shapes:tree(st), panels:P(), line:line,
    msg:{zh:'<b>' + tr(name) + '</b>結果：<b>' + out.join(' → ') + '</b>。每個節點剛好進出一次，<b>O(n)</b>。',
         en:'<b>' + tr(name) + '</b> result: <b>' + out.join(' → ') + '</b>. Every node is entered and left exactly once, <b>O(n)</b>.'}});
  return F.list;
}

function buildBFS(){
  const F = new Frames(), q = [0], out = [], st = {};
  const P = () => [{lbl:{zh:'佇列 queue', en:'queue'}, chips:chipsOf(q.map(i => A[i]), 'act')},
                   {lbl:{zh:'走訪結果', en:'visit order'}, chips:chipsOf(out, 'ok')}];
  const snap = (cur, ln, msg) => {
    const s = Object.assign({}, st); q.forEach(i => s[i] = 'act'); if (cur >= 0) s[cur] = 'hot';
    F.push({shapes:tree(s), panels:P(), line:ln, msg:msg});
  };
  snap(-1, 3, {zh:'層序走訪換一個容器：把堆疊（遞迴）換成<b>佇列</b>，順序就從深度優先變成一層一層。',
               en:'Level order swaps the container: replace the stack (recursion) with a <b>queue</b> and depth-first becomes level-by-level.'});
  while (q.length){
    const i = q.shift();
    out.push(A[i]); st[i] = 'ok';
    snap(i, 5, {zh:'從佇列頭拿出 <b>' + A[i] + '</b> 並記下來。',
                en:'Pop <b>' + A[i] + '</b> from the front of the queue and record it.'});
    const kids = [L(i), R(i)].filter(k => k >= 0);
    kids.forEach(k => q.push(k));
    if (kids.length)
      snap(-1, 7, {zh:'把它的小孩 <b>' + kids.map(k => A[k]).join('、') +
                       '</b> 排到佇列尾巴。先進先出保證同一層的節點會排在一起。',
                   en:'Push its children <b>' + kids.map(k => A[k]).join(', ') +
                      '</b> to the rear. FIFO keeps each level together.'});
  }
  F.push({shapes:tree(st), panels:P(), line:9,
    msg:{zh:'層序結果 <b>' + out.join(' → ') + '</b>——正好是把陣列 A 從左讀到右。' +
            '這個「用佇列一層層擴散」的骨架，之後在圖上就叫 BFS。',
         en:'Level order gives <b>' + out.join(' → ') + '</b> - literally the array read left to right. This queue-driven spreading is exactly what BFS does on a graph.'}});
  return F.list;
}

function buildHeight(){
  const F = new Frames(), st = {}, stack = [], H = {};
  const P = () => [{lbl:{zh:'呼叫堆疊', en:'call stack'}, chips:chipsOf(stack.map(i => A[i]), 'act')},
                   {lbl:{zh:'已算出的高度', en:'heights resolved'},
                    chips:Object.keys(H).map(i => ({t:A[i] + ':' + H[i], cls:'ok'}))}];
  const snap = (cur, ln, msg, style) => {
    const s = Object.assign({}, st); stack.forEach(i => s[i] = 'act');
    if (cur >= 0) s[cur] = style || 'hot';
    F.push({shapes:tree(s), panels:P(), line:ln, msg:msg});
  };
  snap(-1, 0, {zh:'高度沒辦法用一個迴圈算出來，因為<b>一個節點的高度要先知道兩邊子樹的高度</b>。' +
                  '這種「先算小孩」的形狀就是後序。',
               en:'Height cannot be done with a single loop: <b>a node needs both subtree heights first</b>. That "children first" shape is postorder.'});
  const h = i => {
    if (i < 0) return 0;
    stack.push(i);
    snap(i, 0, {zh:'呼叫 <b>height(' + A[i] + ')</b>，先往下鑽。',
                en:'Call <b>height(' + A[i] + ')</b> and dive further down.'});
    const lh = h(L(i)), rh = h(R(i));
    const v = 1 + Math.max(lh, rh);
    H[i] = v; st[i] = 'ok'; stack.pop();
    F.push({shapes:tree(Object.assign({}, st, stack.reduce((a, k) => (a[k] = 'act', a), {}), {[i]:'hot'})),
      panels:P(), line:5,
      msg:{zh:'<b>' + A[i] + '</b> 的左邊高 ' + lh + '、右邊高 ' + rh + '，所以自己是 <b>1 + max = ' + v +
              '</b>。答案是從葉子往上「浮」回來的。',
           en:'<b>' + A[i] + '</b> has left height ' + lh + ' and right height ' + rh + ', so it is <b>1 + max = ' + v +
              '</b>. Answers bubble up from the leaves.'}});
    return v;
  };
  const total = h(0);
  F.push({shapes:tree(st), panels:P(), line:5,
    msg:{zh:'整棵樹高 <b>' + total + '</b>。同一個模板（後序 + 把子樹結果合併）可以算節點數、' +
            '判斷平衡、求直徑，之後在動態規劃和樹狀 DP 都會再看到。',
         en:'The tree is <b>' + total + '</b> tall. The same template - postorder plus combining subtree results - counts nodes, checks balance, finds the diameter, and reappears in tree DP.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'樹', en:'Tree'},
  sub:{zh:'前序、中序、後序其實只差一行的位置；還有為什麼樹的高度非得用遞迴不可。',
       en:'Preorder, inorder and postorder differ by one line; and why the height of a tree wants recursion.'},
  tabs:[
    {id:'dfs', label:{zh:'深度走訪', en:'DFS traversals'},
     stage:{zh:'同一段遞迴，visit 放在不同位置', en:'One recursion, three visit positions'},
     variants:[{zh:'前序', en:'preorder'}, {zh:'中序', en:'inorder'}, {zh:'後序', en:'postorder'}],
     idea:{zh:'三種走訪的差別只有 <b>visit 那一行擺在哪</b>：進來時、左邊走完時、兩邊都走完時。' +
              '記住它們各自的用途——前序複製結構、中序在 BST 上得到排序、後序負責由下往上彙整。',
           en:'The three traversals differ only in <b>where the visit line sits</b>: on arrival, after the left subtree, or after both. Preorder copies structure, inorder sorts a BST, postorder aggregates bottom-up.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_DFS, build:buildDFS},
    {id:'bfs', label:{zh:'層序走訪', en:'Level order'},
     stage:{zh:'把堆疊換成佇列', en:'Swap the stack for a queue'},
     idea:{zh:'遞迴其實就是借用<b>呼叫堆疊</b>。把容器換成佇列，走訪順序立刻變成一層一層——' +
              '這是深度優先與廣度優先唯一的結構差別。',
           en:'Recursion is really borrowing the <b>call stack</b>. Swap in a queue and the order becomes level by level - that container choice is the only structural difference between DFS and BFS.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_BFS, build:buildBFS},
    {id:'h', label:{zh:'樹高（後序的力量）', en:'height()'},
     stage:{zh:'答案從葉子往上浮', en:'Answers bubble up from the leaves'},
     idea:{zh:'<b>1 + max(左, 右)</b> 這行的前提是兩邊都已經算完。凡是「節點的答案由子樹的答案組成」' +
              '的問題，都可以套這個後序模板。',
           en:'<b>1 + max(left, right)</b> only works once both sides are resolved. Any problem where a node\'s answer is built from its subtrees\' answers fits this postorder template.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_H, build:buildHeight}
  ]
};
