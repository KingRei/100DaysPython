// DAY: 01
// TITLE_ZH: 鏈結串列
// TITLE_EN: Linked List
// SUB_ZH: 一次一個 next 指標：加入、搜尋、刪除，還有雙向串列為什麼能 O(1) 拔掉中間的節點。
// SUB_EN: One next pointer at a time.
// FOLDER: day%2001%20-%20linked%20list
// MEDIUM: https://medium.com/100-days-of-python/day-01-linked-list-cf91b8937fc6

const NW = 1.12, NH = .92, GAP = 1.78, Y = 2.9;
const nodeX = i => 1.25 + i * GAP;

function listShapes(arr, o){
  o = o || {};
  const y = o.y == null ? Y : o.y, st = o.states || {}, out = [];
  const midy = y + NH / 2;
  arr.forEach((d, i) => {
    const s = st[i] || 'idle';
    out.push(S.r(nodeX(i), y, NW, NH, s, d, {fs:.46, dx:-NW * .18}));
    out.push(S.e(nodeX(i) + NW * .64, y, nodeX(i) + NW * .64, y + NH,
                 {arrow:false, s:'soft', w:.028, o:.8}));
    out.push(S.c(nodeX(i) + NW * .82, midy, .09, s === 'idle' ? 'soft' : s, ''));
    const from = nodeX(i) + NW * .82;
    if (i + 1 < arr.length)
      out.push(S.e(from, midy, nodeX(i + 1), midy, {s:o.linkStates && o.linkStates[i] || 'idle', w:.05}));
    else
      out.push(S.e(from, midy, nodeX(i) + NW + .70, midy, {s:'ghost', w:.045}),
               S.t(nodeX(i) + NW + 1.15, midy + .12, 'None', {c:'#8fa3ac', fs:.32}));
    if (o.doubly && i > 0)
      out.push(S.e(nodeX(i) - .02, y + NH + .30, nodeX(i - 1) + NW + .02, y + NH + .30,
                   {s:o.prevStates && o.prevStates[i] || 'soft', w:.042}));
  });
  if (o.doubly && arr.length)
    out.push(S.t(nodeX(arr.length - 1) + NW + .70, y + NH + .42, 'prev', {c:'#8fa3ac', fs:.26}));
  const hx = arr.length ? nodeX(0) + NW / 2 : 1.9;
  out.push(S.t(hx, y - 1.02, 'head', {c:'#c7a6ff', fs:.36}));
  if (arr.length)
    out.push(S.e(hx, y - .86, hx, y - .10, {s:'act', w:.05}));
  else
    out.push(S.t(hx, y - .50, 'None', {c:'#8fa3ac', fs:.32}));
  return out;
}
const chain = arr => arr.length ? arr.join(' -> ') + ' -> None' : 'None';
const panelOf = arr => [{lbl:{zh:'目前串列', en:'current list'},
  chips:arr.map(d => ({t:d, cls:'ok'}))}];

const CODE_ADD = [
'def add(self, data):            # 插到頭部 / push front',
'    node = Node(data)',
'    if self.head == None:',
'        self.head = node',
'    else:',
'        node.next = self.head',
'        self.head = node'];
const CODE_SEARCH = [
'def search(self, k):',
'    p = self.head',
'    while p != None:',
'        if p.data == k:',
'            return p',
'        p = p.next',
'    return None'];
const CODE_REMOVE = [
'def remove(self):               # 拔掉頭部 / pop front',
'    if self.head != None:',
'        tmp = self.head',
'        self.head = tmp.next'];
const CODE_DLL = [
'def remove(self, p):            # 雙向串列：拔掉任何一個 p',
'    if p == self.head:',
'        self.head = p.next',
'    else:',
'        tmp = p.prev',
'        tmp.next = p.next',
'        if tmp.next != None:',
'            tmp.next.prev = tmp'];

function buildAdd(){
  const F = new Frames(), arr = [];
  F.push({shapes:listShapes(arr), panels:panelOf(arr), line:0,
    msg:{zh:'空串列：<b>head</b> 指向 <b>None</b>。整條串列只靠這一個入口活著。',
         en:'Empty list: <b>head</b> points at <b>None</b>. That single entry point is the whole list.'}});
  ['a', 'b', 'c'].forEach(d => {
    const floatY = .85;
    const newNode = [S.r(nodeX(0), floatY, NW, NH, 'hot', d, {fs:.46, dx:-NW * .18}),
      S.c(nodeX(0) + NW * .82, floatY + NH / 2, .09, 'hot', ''),
      S.t(nodeX(0) + NW / 2, floatY - .30, 'node', {c:'#ffbe6b', fs:.30})];
    F.push({shapes:listShapes(arr).concat(newNode), panels:panelOf(arr), line:1,
      msg:{zh:'先做一個新節點 <b>' + d + '</b>。它的 <b>next</b> 還是空的，跟串列還沒有任何關係。',
           en:'Make a fresh node <b>' + d + '</b>. Its <b>next</b> is empty - it is not part of the list yet.'}});
    if (!arr.length){
      F.push({shapes:listShapes(arr).concat(newNode,
        [S.e(nodeX(0) + NW / 2, floatY + NH + .10, nodeX(0) + NW / 2, Y - .10, {s:'hot', w:.05})]),
        panels:panelOf(arr), line:3,
        msg:{zh:'串列本來是空的，所以 <b>head = node</b> 就結束了。',
             en:'The list was empty, so <b>head = node</b> is all it takes.'}});
    } else {
      F.push({shapes:listShapes(arr).concat(newNode,
        [S.e(nodeX(0) + NW * .82, floatY + NH / 2, nodeX(0) + NW * .35, Y - .06, {s:'hot', w:.05})]),
        panels:panelOf(arr), line:5,
        msg:{zh:'<b>順序很重要</b>：先讓新節點的 next 指到舊的 head（<b>' + arr[0] + '</b>），' +
                '再去動 head。反過來寫，舊串列就找不到了。',
             en:'<b>Order matters</b>: first point the new node at the old head (<b>' + arr[0] +
                '</b>), only then move head. Do it the other way round and the old list is lost.'}});
    }
    arr.unshift(d);
    F.push({shapes:listShapes(arr, {states:{0:'ok'}}), panels:panelOf(arr),
      line:arr.length === 1 ? 3 : 6,
      msg:{zh:'<b>head</b> 改指向 <b>' + d + '</b>。不管串列多長，插入頭部都是固定 <b>O(1)</b>。',
           en:'<b>head</b> now points at <b>' + d + '</b>. Push-front is <b>O(1)</b> no matter how long the list is.'}});
  });
  F.push({shapes:listShapes(arr, {states:{0:'ok', 1:'ok', 2:'ok'}}), panels:panelOf(arr), line:0,
    msg:{zh:'結果是 <b>' + chain(arr) + '</b>——加入順序 a、b、c，串列卻是倒過來的，因為每次都插在頭部。',
         en:'Result: <b>' + chain(arr) + '</b> - added a, b, c but the list reads backwards, because every add goes to the front.'}});
  return F.list;
}

function buildSearch(v){
  const arr = ['c', 'b', 'a'], key = v === 0 ? 'a' : 'z', F = new Frames();
  F.push({shapes:listShapes(arr), panels:panelOf(arr), line:0,
    msg:{zh:'要找 <b>' + key + '</b>。串列沒有索引，唯一的辦法是從 head 一個一個走。',
         en:'Looking for <b>' + key + '</b>. A list has no index - the only way in is to walk from head.'}});
  let hops = 0;
  for (let i = 0; i < arr.length; i++){
    hops++;
    const st = {}; for (let j = 0; j < i; j++) st[j] = 'done'; st[i] = 'act';
    const pMark = [S.t(nodeX(i) + NW / 2, Y + NH + .55, 'p', {c:'#c7a6ff', fs:.34})];
    F.push({shapes:listShapes(arr, {states:st}).concat(pMark), panels:panelOf(arr), line:3,
      msg:{zh:'第 ' + hops + ' 步：<b>p.data == ' + arr[i] + '</b>，' +
              (arr[i] === key ? '跟 <b>' + key + '</b> 相同——找到了。' : '不是 <b>' + key + '</b>，繼續往下。'),
           en:'Hop ' + hops + ': <b>p.data == ' + arr[i] + '</b>' +
              (arr[i] === key ? ' - that is <b>' + key + '</b>, found it.' : ', not <b>' + key + '</b>, keep going.')}});
    if (arr[i] === key){
      st[i] = 'ok';
      F.push({shapes:listShapes(arr, {states:st}).concat(pMark), panels:panelOf(arr), line:4,
        msg:{zh:'回傳這個節點。走了 <b>' + hops + '</b> 步——最壞情況就是整條串列 <b>O(n)</b>。',
             en:'Return that node. It took <b>' + hops + '</b> hops - worst case is the whole list, <b>O(n)</b>.'}});
      return F.list;
    }
    F.push({shapes:listShapes(arr, {states:st, linkStates:{[i]:'act'}}).concat(pMark),
      panels:panelOf(arr), line:5,
      msg:{zh:'<b>p = p.next</b>：沿著指標往右跳一格。',
           en:'<b>p = p.next</b>: follow the pointer one hop to the right.'}});
  }
  const st = {}; arr.forEach((_, i) => st[i] = 'done');
  F.push({shapes:listShapes(arr, {states:st}), panels:panelOf(arr), line:6,
    msg:{zh:'走到 <b>None</b> 還沒找到，回傳 None。整條串列都掃過了，這就是 <b>O(n)</b> 的代價。',
         en:'We hit <b>None</b> without a match, so return None. The whole list was scanned - that is the <b>O(n)</b> price.'}});
  return F.list;
}

function buildRemove(){
  const arr = ['c', 'b', 'a'], F = new Frames();
  F.push({shapes:listShapes(arr), panels:panelOf(arr), line:0,
    msg:{zh:'從 <b>' + chain(arr) + '</b> 開始，連拔兩次頭部。',
         en:'Starting from <b>' + chain(arr) + '</b>, pop the head twice.'}});
  for (let k = 0; k < 2; k++){
    const gone = arr[0];
    F.push({shapes:listShapes(arr, {states:{0:'hot'}}).concat(
      [S.t(nodeX(0) + NW / 2, Y + NH + .55, 'tmp', {c:'#ffbe6b', fs:.34})]),
      panels:panelOf(arr), line:2,
      msg:{zh:'<b>tmp = self.head</b>，先把要拔掉的節點 <b>' + gone + '</b> 記起來。',
           en:'<b>tmp = self.head</b> - remember the node we are about to drop, <b>' + gone + '</b>.'}});
    F.push({shapes:listShapes(arr, {states:{0:'bad', 1:'ok'}, linkStates:{0:'act'}}),
      panels:panelOf(arr), line:3,
      msg:{zh:'<b>head = tmp.next</b>：head 直接跳過 <b>' + gone + '</b>。沒有人指著它了，' +
              'Python 的垃圾回收會處理掉。',
           en:'<b>head = tmp.next</b>: head jumps over <b>' + gone + '</b>. Nothing references it any more, so Python collects it.'}});
    arr.shift();
    F.push({shapes:listShapes(arr, {states:{0:'ok'}}), panels:panelOf(arr), line:3,
      msg:{zh:'現在是 <b>' + chain(arr) + '</b>。沒有搬動任何資料，只改了一個指標——<b>O(1)</b>。',
           en:'Now <b>' + chain(arr) + '</b>. No data moved, one pointer changed - <b>O(1)</b>.'}});
  }
  F.push({shapes:listShapes(arr, {states:{0:'ok'}}), panels:panelOf(arr), line:1,
    msg:{zh:'陣列拔掉第一個元素要把後面全部往前搬（O(n)），串列只要動 head。這就是它存在的理由。',
         en:'Removing the first element of an array shifts everything left (O(n)); a linked list just moves head. That is the whole point.'}});
  return F.list;
}

function buildDLL(){
  const arr = ['c', 'b', 'a'], F = new Frames(), o = {doubly:true};
  F.push({shapes:listShapes(arr, o), panels:panelOf(arr), line:0,
    msg:{zh:'雙向串列每個節點多一個 <b>prev</b>（下方那排箭頭）。多花一點記憶體，換到往回走的能力。',
         en:'A doubly linked list adds a <b>prev</b> pointer (the lower row of arrows): a little more memory, bought back as the ability to walk backwards.'}});
  const i = 1;
  F.push({shapes:listShapes(arr, {doubly:true, states:{1:'act'}}).concat(
    [S.t(nodeX(1) + NW / 2, Y - .50, 'p', {c:'#c7a6ff', fs:.34})]),
    panels:panelOf(arr), line:1,
    msg:{zh:'要拔掉中間的 <b>b</b>。先問：它是不是 head？不是，走 else。',
         en:'We want to drop <b>b</b> in the middle. Is it the head? No, so take the else branch.'}});
  F.push({shapes:listShapes(arr, {doubly:true, states:{0:'hot', 1:'act'}, prevStates:{1:'hot'}}).concat(
    [S.t(nodeX(0) + NW / 2, Y - .50, 'tmp = p.prev', {c:'#ffbe6b', fs:.30})]),
    panels:panelOf(arr), line:4,
    msg:{zh:'<b>tmp = p.prev</b>——這就是關鍵：單向串列必須從頭走一遍才知道前一個是誰，' +
            '雙向串列一步就拿到。',
         en:'<b>tmp = p.prev</b> - this is the payoff: a singly linked list must walk from the head to find the predecessor; here it is one step.'}});
  F.push({shapes:listShapes(arr, {doubly:true, states:{0:'hot', 1:'bad', 2:'ok'}}).concat(
    [S.e(nodeX(0) + NW * .82, Y + NH / 2, nodeX(2), Y - .35, {s:'hot', w:.05}),
     S.e(nodeX(2), Y - .35, nodeX(2), Y + NH / 2 - .05, {s:'hot', w:.05, arrow:false})]),
    panels:panelOf(arr), line:5,
    msg:{zh:'<b>tmp.next = p.next</b>：讓 <b>a</b> 直接跨過 <b>b</b> 指到 <b>c</b>。',
         en:'<b>tmp.next = p.next</b>: let <b>a</b> reach over <b>b</b> straight to <b>c</b>.'}});
  F.push({shapes:listShapes(arr, {doubly:true, states:{0:'hot', 1:'bad', 2:'ok'}, prevStates:{2:'hot'}}),
    panels:panelOf(arr), line:7,
    msg:{zh:'別忘了另一邊：<b>tmp.next.prev = tmp</b>。雙向串列每次刪除都要修兩條指標，' +
            '少修一條就會在往回走的時候炸掉。',
         en:'Do not forget the other side: <b>tmp.next.prev = tmp</b>. Every deletion fixes two pointers; skip one and the backward walk breaks.'}});
  arr.splice(1, 1);
  F.push({shapes:listShapes(arr, {doubly:true, states:{0:'ok', 1:'ok'}}), panels:panelOf(arr), line:0,
    msg:{zh:'結果 <b>' + chain(arr) + '</b>。只要手上有節點，刪除就是 <b>O(1)</b>——' +
            '這正是 LRU 快取要用「雜湊表 + 雙向串列」的原因。',
         en:'Result <b>' + chain(arr) + '</b>. Given the node itself, deletion is <b>O(1)</b> - exactly why an LRU cache pairs a hash map with a doubly linked list.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'鏈結串列', en:'Linked List'},
  sub:{zh:'一次一個 next 指標：加入、搜尋、刪除，還有雙向串列為什麼能 O(1) 拔掉中間的節點。',
       en:'One next pointer at a time: add, search, remove, and why a doubly linked list can unlink a middle node in O(1).'},
  tabs:[
    {id:'add', label:{zh:'加入 add()', en:'add()'},
     stage:{zh:'把新節點插到頭部', en:'Pushing a node onto the front'},
     idea:{zh:'鏈結串列不是一塊連續記憶體，而是一串「資料 + 下一個的地址」。' +
              '<b>插入頭部只改兩個指標，跟長度無關</b>；代價是想拿第 k 個元素得從頭走 k 步。',
           en:'A linked list is not one contiguous block; it is a chain of "data + address of the next one". ' +
              '<b>Push-front touches two pointers regardless of length</b>; the price is that reaching element k costs k hops.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_ADD, build:buildAdd},
    {id:'search', label:{zh:'搜尋 search()', en:'search()'},
     stage:{zh:'從 head 一路走', en:'Walking from head'},
     variants:[{zh:'找得到 (a)', en:'hit (a)'}, {zh:'找不到 (z)', en:'miss (z)'}],
     idea:{zh:'串列<b>沒有索引</b>：查一個值只能從 head 沿著 next 走，平均掃一半、最壞掃全部，' +
              '都是 <b>O(n)</b>。想要 O(1) 查找就得換工具（雜湊表）。',
           en:'A list has <b>no index</b>: looking up a value means walking from head along next - half the list on average, all of it at worst, so <b>O(n)</b>. O(1) lookup needs a different tool (a hash table).'},
     legend:['act', 'ok', 'done', 'idle'], code:CODE_SEARCH, build:buildSearch},
    {id:'remove', label:{zh:'刪除 remove()', en:'remove()'},
     stage:{zh:'拔掉頭部', en:'Popping the head'},
     idea:{zh:'刪除的成本全在「找到前一個節點」。頭部的前一個就是 head 本身，所以 <b>O(1)</b>；' +
              '要刪中間的節點，單向串列就得先走過去。',
           en:'The cost of deletion is all in finding the predecessor. For the head there is none, so it is <b>O(1)</b>; deleting a middle node in a singly linked list means walking there first.'},
     legend:['hot', 'bad', 'ok', 'idle'], code:CODE_REMOVE, build:buildRemove},
    {id:'dll', label:{zh:'雙向串列', en:'doubly linked'},
     stage:{zh:'拔掉中間的節點', en:'Unlinking a middle node'},
     idea:{zh:'多存一個 <b>prev</b>，就能在拿到節點的當下直接把它拆掉，不用重新搜尋。' +
              '記憶體換時間的經典交易，之後做快取淘汰、圖的鄰接串列都會再看到。',
           en:'Storing one extra <b>prev</b> pointer lets you unlink a node the moment you hold it, with no second search. A classic memory-for-time trade you will meet again in cache eviction and adjacency lists.'},
     legend:['hot', 'act', 'bad', 'ok'], code:CODE_DLL, build:buildDLL}
  ]
};
