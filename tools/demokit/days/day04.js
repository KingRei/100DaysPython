// DAY: 04
// TITLE_ZH: 二元搜尋樹
// TITLE_EN: Binary Search Tree
// SUB_ZH: 每個節點都是一次二分：插入、搜尋、以及最麻煩的「刪掉有兩個小孩的節點」。
// SUB_EN: Every node is one binary decision.
// FOLDER: day%2004%20-%20binary%20search%20tree
// MEDIUM: https://medium.com/100-days-of-python/day-04-data-structure-binary-search-tree-8aef2a8f61eb

function nd(v){ return {v:v, l:null, r:null}; }
function ins(root, v){
  if (!root) return nd(v);
  if (v < root.v) root.l = ins(root.l, v); else if (v > root.v) root.r = ins(root.r, v);
  return root;
}
function mkTree(vals){ let r = null; vals.forEach(v => r = ins(r, v)); return r; }
function inorder(n, out){ out = out || []; if (n){ inorder(n.l, out); out.push(n.v); inorder(n.r, out); } return out; }

function treeShapes(root, states, opt){
  opt = opt || {};
  const order = inorder(root), n = Math.max(order.length, 1);
  const span = 9.0 / n, out = [], pos = {};
  (function place(node, d){
    if (!node) return;
    place(node.l, d + 1);
    pos[node.v] = {x:.5 + (order.indexOf(node.v) + .5) * span, y:1.15 + d * 1.32};
    place(node.r, d + 1);
  })(root, 0);
  const Rr = .36;
  (function draw(node){
    if (!node) return;
    [node.l, node.r].forEach(k => {
      if (k) out.push(S.e(pos[node.v].x, pos[node.v].y, pos[k.v].x, pos[k.v].y,
        {pad:Rr + .04, s:(states && states[k.v] === 'hot') ? 'hot' : ((states && states[k.v]) === 'act' ? 'act' : 'idle')}));
      draw(k);
    });
  })(root);
  (function draw2(node){
    if (!node) return;
    out.push(S.c(pos[node.v].x, pos[node.v].y, Rr, (states && states[node.v]) || 'idle',
      String(node.v), {fs:.40, sub:opt.subs && opt.subs[node.v]}));
    draw2(node.l); draw2(node.r);
  })(root);
  if (opt.ghost)
    out.push(S.c(opt.ghost.x, opt.ghost.y, Rr, 'hot', String(opt.ghost.v), {fs:.40}));
  return out;
}
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));
const BASE = [8, 3, 10, 1, 6, 14, 4, 7, 13];

const CODE_INS = [
'def insert(node, v):',
'    if node is None:            # 空位就是它的家',
'        return Node(v)',
'    if v < node.v:',
'        node.left = insert(node.left, v)',
'    elif v > node.v:',
'        node.right = insert(node.right, v)',
'    return node                 # 相等就當作已存在'];
const CODE_SEARCH = [
'def search(node, k):',
'    while node is not None:',
'        if k == node.v:',
'            return node',
'        node = node.left if k < node.v else node.right',
'    return None'];
const CODE_DEL = [
'def delete(node, v):',
'    if node is None: return None',
'    if v < node.v:   node.left  = delete(node.left, v)',
'    elif v > node.v: node.right = delete(node.right, v)',
'    else:',
'        if node.left is None:  return node.right   # 0 或 1 個小孩',
'        if node.right is None: return node.left',
'        s = min_node(node.right)    # 中序後繼者',
'        node.v = s.v',
'        node.right = delete(node.right, s.v)',
'    return node'];

function panelsOf(root, extra){
  const p = [{lbl:{zh:'中序走訪（應該永遠是排序好的）', en:'inorder (must stay sorted)'},
              chips:chipsOf(inorder(root), 'ok')}];
  if (extra) p.push(extra);
  return p;
}

function buildInsert(){
  const F = new Frames(); let root = null;
  const vals = [8, 3, 10, 1, 6, 14, 4];
  F.push({shapes:[S.t(5, 3.0, {zh:'空的樹', en:'empty tree'}, {c:'#8fa3ac', fs:.44})],
    panels:panelsOf(null), line:1,
    msg:{zh:'二元搜尋樹的唯一規則：<b>左子樹全部比我小、右子樹全部比我大</b>。' +
            '插入就是照這條規則一路往下走到空位。',
         en:'A BST has one rule: <b>everything on the left is smaller, everything on the right is larger</b>. Inserting means following that rule down to an empty slot.'}});
  vals.forEach(v => {
    let cur = root, steps = 0;
    while (cur){
      steps++;
      const goLeft = v < cur.v;
      F.push({shapes:treeShapes(root, {[cur.v]:'hot'}), panels:panelsOf(root), line:goLeft ? 3 : 5,
        msg:{zh:'插入 <b>' + v + '</b>：跟 <b>' + cur.v + '</b> 比，' + v + (goLeft ? ' 比較小 → 往左' : ' 比較大 → 往右') +
                '。<b>每比一次就砍掉一半的樹</b>。',
             en:'Inserting <b>' + v + '</b>: compare with <b>' + cur.v + '</b>; ' + v +
                (goLeft ? ' is smaller, go left' : ' is larger, go right') + '. <b>Each comparison discards half the tree.</b>'}});
      cur = goLeft ? cur.l : cur.r;
    }
    root = ins(root, v);
    F.push({shapes:treeShapes(root, {[v]:'ok'}), panels:panelsOf(root), line:2,
      msg:{zh:'走到 <b>None</b>，這個空位就是 <b>' + v + '</b> 的家（比了 ' + steps + ' 次）。' +
              '新節點一定是<b>葉子</b>，不用搬動任何舊節點。',
           en:'We hit <b>None</b>, and that empty slot is where <b>' + v + '</b> belongs (' + steps +
              ' comparisons). A new node is always a <b>leaf</b>, so nothing else moves.'}});
  });
  F.push({shapes:treeShapes(root, inorder(root).reduce((a, v) => (a[v] = 'ok', a), {})),
    panels:panelsOf(root), line:0,
    msg:{zh:'中序走訪出來剛好是 <b>' + inorder(root).join(', ') + '</b>——排序好的。' +
            '樹本身沒排序，是那條規則讓中序自動排序。',
         en:'Inorder gives <b>' + inorder(root).join(', ') + '</b> - sorted. The tree is not stored in order; the rule makes inorder come out sorted for free.'}});
  return F.list;
}

function buildSearch(v){
  const root = mkTree(BASE), key = v === 0 ? 7 : 5, F = new Frames();
  let cur = root, steps = 0; const st = {};
  F.push({shapes:treeShapes(root, {}), panels:panelsOf(root), line:0,
    msg:{zh:'要找 <b>' + key + '</b>。跟串列一個一個走不同，這裡每一步都能丟掉一整個子樹。',
         en:'Looking for <b>' + key + '</b>. Unlike a linked list, every step here throws away an entire subtree.'}});
  while (cur){
    steps++;
    if (key === cur.v){
      st[cur.v] = 'ok';
      F.push({shapes:treeShapes(root, st), panels:panelsOf(root), line:3,
        msg:{zh:'找到 <b>' + key + '</b>，只比了 <b>' + steps + '</b> 次。樹裡有 ' + inorder(root).length +
                ' 個節點，卻只看了幾個。',
             en:'Found <b>' + key + '</b> in just <b>' + steps + '</b> comparisons - out of ' +
                inorder(root).length + ' nodes we looked at a handful.'}});
      F.push({shapes:treeShapes(root, st), panels:panelsOf(root), line:4,
        msg:{zh:'成本是<b>樹的高度</b>：平衡時 O(log n)，但如果照排序好的資料插入，樹會退化成一條線，' +
                '就變回 O(n)。這就是後來 AVL、紅黑樹要做旋轉的理由。',
             en:'The cost is the <b>height</b>: O(log n) when balanced, but inserting already-sorted data degenerates the tree into a list and you are back to O(n). That is why AVL and red-black trees rotate.'}});
      return F.list;
    }
    st[cur.v] = 'done';
    const goLeft = key < cur.v;
    F.push({shapes:treeShapes(root, Object.assign({}, st, {[cur.v]:'hot'})), panels:panelsOf(root), line:4,
      msg:{zh:'<b>' + key + '</b> ' + (goLeft ? '小於' : '大於') + ' <b>' + cur.v + '</b> → 往' +
              (goLeft ? '左' : '右') + '。另一半的子樹永遠不必看了。',
           en:'<b>' + key + '</b> is ' + (goLeft ? 'less' : 'greater') + ' than <b>' + cur.v + '</b>, so go ' +
              (goLeft ? 'left' : 'right') + '. The other subtree never needs to be examined.'}});
    cur = goLeft ? cur.l : cur.r;
  }
  F.push({shapes:treeShapes(root, st), panels:panelsOf(root), line:5,
    msg:{zh:'走到 <b>None</b>，代表 <b>' + key + '</b> 不在樹裡（比了 ' + steps + ' 次）。' +
            '值得注意的是：失敗的位置，正好就是它<b>該被插入</b>的位置。',
         en:'We hit <b>None</b>, so <b>' + key + '</b> is not in the tree (' + steps +
            ' comparisons). Note that where the search fails is exactly where the value <b>would be inserted</b>.'}});
  return F.list;
}

function buildDelete(v){
  const target = [1, 14, 3][v];
  const kind = [{zh:'葉子', en:'a leaf'}, {zh:'只有一個小孩', en:'one child'},
                {zh:'兩個小孩', en:'two children'}][v];
  const F = new Frames();
  let root = mkTree(BASE);
  const before = inorder(root).join(', ');
  F.push({shapes:treeShapes(root, {[target]:'act'}), panels:panelsOf(root), line:0,
    msg:{zh:'要刪掉 <b>' + target + '</b>，它是<b>' + tr(kind) + '</b>的情況。' +
            '刪除之所以麻煩，是因為<b>刪完還要維持排序規則</b>。',
         en:'Delete <b>' + target + '</b>, the case with <b>' + tr(kind) + '</b>. Deletion is the fiddly one because the ordering rule must survive it.'}});
  // walk down
  let cur = root, path = [];
  while (cur && cur.v !== target){ path.push(cur.v); cur = target < cur.v ? cur.l : cur.r; }
  if (path.length){
    const st = {}; path.forEach(p => st[p] = 'done'); st[target] = 'hot';
    F.push({shapes:treeShapes(root, st), panels:panelsOf(root), line:2,
      msg:{zh:'先跟搜尋一樣走下去找到它：' + path.join(' → ') + ' → <b>' + target + '</b>。',
           en:'First find it exactly like a search: ' + path.join(' → ') + ' → <b>' + target + '</b>.'}});
  }
  if (v === 0){
    F.push({shapes:treeShapes(root, {[target]:'bad'}), panels:panelsOf(root), line:5,
      msg:{zh:'葉子最簡單：<b>直接回傳 None</b>，父節點那一邊就變空。沒有人會受影響。',
           en:'A leaf is easy: <b>return None</b> and the parent side becomes empty. Nobody else is affected.'}});
  } else if (v === 1){
    F.push({shapes:treeShapes(root, {[target]:'bad', 13:'ok'}), panels:panelsOf(root), line:6,
      msg:{zh:'只有一個小孩：<b>讓小孩直接頂上來</b>。<b>13</b> 原本就在 <b>14</b> 的左邊、' +
              '也在 <b>10</b> 的右邊，接上去仍然合法。',
           en:'One child: <b>promote the child</b>. <b>13</b> was already left of <b>14</b> and right of <b>10</b>, so splicing it in keeps the rule.'}});
  } else {
    F.push({shapes:treeShapes(root, {3:'bad', 4:'act', 6:'act', 7:'act'}), panels:panelsOf(root), line:4,
      msg:{zh:'兩個小孩就不能直接拔掉——左右兩邊都得掛回去。<b>誰有資格接替 3？</b>',
           en:'With two children you cannot just remove it: both sides need a home. <b>Who is allowed to take over?</b>'}});
    F.push({shapes:treeShapes(root, {3:'bad', 6:'done', 4:'hot'},
      {subs:{4:{zh:'右子樹最小', en:'min of right'}}}), panels:panelsOf(root), line:7,
      msg:{zh:'答案是<b>中序後繼者</b>：右子樹裡最小的那個，也就是一直往左走到底 → <b>4</b>。' +
              '它比左子樹全部都大、又比右子樹其他人都小，是唯一能無痛頂替的值。',
           en:'The answer is the <b>inorder successor</b>: the smallest value in the right subtree, found by walking left as far as possible - <b>4</b>. It is larger than everything on the left and smaller than the rest of the right, the only painless substitute.'}});
    F.push({shapes:treeShapes(root, {3:'ok', 4:'hot'},
      {subs:{3:{zh:'值改成 4', en:'value becomes 4'}, 4:{zh:'待刪', en:'to be removed'}}}),
      panels:panelsOf(root), line:8,
      msg:{zh:'把 <b>4</b> <b>複製</b>到 3 的位置（只搬值，不搬節點）。',
           en:'<b>Copy</b> the value 4 into the node holding 3 - we move a value, not a node.'}});
    F.push({shapes:treeShapes(root, {3:'ok', 4:'bad'},
      {subs:{3:{zh:'現在裝的是 4', en:'now holds 4'}}}), panels:panelsOf(root), line:9,
      msg:{zh:'問題轉成「在右子樹裡刪掉 <b>4</b>」——而它最多只有一個小孩，回到簡單情況。' +
              '把難題化約成已解決的情況，是遞迴最常見的手法。',
           en:'The problem becomes "delete <b>4</b> from the right subtree", and that node has at most one child - back to an easy case. Reducing a hard case to a solved one is the classic recursive move.'}});
  }
  // apply deletion
  root = (function del(n, x){
    if (!n) return null;
    if (x < n.v) n.l = del(n.l, x);
    else if (x > n.v) n.r = del(n.r, x);
    else {
      if (!n.l) return n.r;
      if (!n.r) return n.l;
      let s = n.r; while (s.l) s = s.l;
      n.v = s.v; n.r = del(n.r, s.v);
    }
    return n;
  })(root, target);
  F.push({shapes:treeShapes(root, {}), panels:panelsOf(root), line:10,
    msg:{zh:'刪完了：中序從 <b>' + before + '</b> 變成 <b>' + inorder(root).join(', ') +
            '</b>——只少了 ' + target + '，順序完全沒亂。這就是驗收標準。',
         en:'Done: inorder went from <b>' + before + '</b> to <b>' + inorder(root).join(', ') +
            '</b> - only ' + target + ' is gone and the order is intact. That is the test that matters.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'二元搜尋樹', en:'Binary Search Tree'},
  sub:{zh:'每個節點都是一次二分：插入、搜尋、以及最麻煩的「刪掉有兩個小孩的節點」。',
       en:'Every node is one binary decision: insert, search, and the awkward case of deleting a node with two children.'},
  tabs:[
    {id:'ins', label:{zh:'插入', en:'insert'},
     stage:{zh:'一路比大小走到空位', en:'Compare your way down to an empty slot'},
     idea:{zh:'BST 的規則只有一條，卻讓「搜尋」變成一連串二分。插入永遠只加葉子，' +
              '所以<b>不會動到既有節點</b>。',
           en:'One rule turns lookup into a chain of halvings. Insertion only ever appends a leaf, so <b>existing nodes never move</b>.'},
     legend:['hot', 'ok', 'idle'], code:CODE_INS, build:buildInsert},
    {id:'search', label:{zh:'搜尋', en:'search'},
     stage:{zh:'每一步丟掉一整個子樹', en:'Each step discards a whole subtree'},
     variants:[{zh:'找得到 7', en:'hit 7'}, {zh:'找不到 5', en:'miss 5'}],
     idea:{zh:'搜尋成本等於<b>樹的高度</b>，不是節點數。平衡的樹 O(log n)；' +
              '照順序插入會退化成一條鏈，變回 O(n)——這就是自平衡樹存在的理由。',
           en:'The cost is the <b>height</b>, not the node count: O(log n) when balanced, but sorted input degenerates the tree into a chain and O(n) returns. Hence self-balancing trees.'},
     legend:['hot', 'ok', 'done', 'idle'], code:CODE_SEARCH, build:buildSearch},
    {id:'del', label:{zh:'刪除的三種情況', en:'delete: three cases'},
     stage:{zh:'刪完必須仍然是 BST', en:'It must still be a BST afterwards'},
     variants:[{zh:'葉子 (1)', en:'leaf (1)'}, {zh:'一個小孩 (14)', en:'one child (14)'},
               {zh:'兩個小孩 (3)', en:'two children (3)'}],
     idea:{zh:'難的只有第三種：用<b>中序後繼者</b>（右子樹最小值）頂替，再回頭刪掉那個後繼者。' +
              '驗收方式很簡單——刪完的中序走訪必須仍然由小到大。',
           en:'Only the third case is hard: substitute the <b>inorder successor</b> (the minimum of the right subtree), then delete that successor instead. The acceptance test is simple - inorder must still come out sorted.'},
     legend:['hot', 'act', 'bad', 'ok'], code:CODE_DEL, build:buildDelete}
  ]
};
