// DAY: 05
// TITLE_ZH: 二元堆積（優先佇列）
// TITLE_EN: Binary Heap (Priority Queue)
// SUB_ZH: 一個陣列假裝自己是一棵樹：上浮、下沉、O(n) 建堆，還有幾乎免費的堆積排序。
// SUB_EN: An array pretending to be a tree.
// FOLDER: day%2005%20-%20heap
// MEDIUM: https://medium.com/100-days-of-python/day-05-data-structure-priority-queue-binary-heap-6a35149c8c16

const AY = 5.55, AH = .82, AW = 1.0;
const ax0 = n => 5.0 - n * AW / 2;
function view(arr, states, msgRow){
  const out = heapTreeShapes(arr, .45, 1.05, 9.1, 1.32, states, {r:.35});
  return out.concat(cellRow(arr, ax0(arr.length), AY, AW, AH, {states:states, ilift:.30, fs:.42}),
    [S.t(ax0(arr.length) - .35, AY + AH * .62, {zh:'陣列', en:'array'}, {c:'#3fe0dd', fs:.30, anchor:'end'})]);
}
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));
const P = (arr, extra) => {
  const p = [{lbl:{zh:'堆積（陣列形式）', en:'heap as an array'}, chips:chipsOf(arr, 'act')}];
  if (extra) p.push(extra);
  return p;
};

const CODE_INS = [
'def insert(self, k):',
'    self.items.append(k)        # 先擺在最後一格',
'    self.percUp(len(self.items) - 1)',
'',
'def percUp(self, i):            # 上浮',
'    while i > 0:',
'        p = (i - 1) // 2        # 父節點',
'        if self.items[i] < self.items[p]:',
'            self.items[i], self.items[p] = self.items[p], self.items[i]',
'        else:',
'            break',
'        i = p'];
const CODE_DEL = [
'def del_min(self):',
'    top = self.items[0]         # 最小值一定在根',
'    self.items[0] = self.items[-1]   # 最後一個搬到根',
'    self.items.pop()',
'    self.percDown(0)',
'    return top',
'',
'def percDown(self, i):          # 下沉',
'    n = len(self.items)',
'    while 2 * i + 1 < n:',
'        c = 2 * i + 1',
'        if c + 1 < n and self.items[c + 1] < self.items[c]:',
'            c += 1              # 選比較小的那個小孩',
'        if self.items[c] < self.items[i]:',
'            self.items[i], self.items[c] = self.items[c], self.items[i]',
'            i = c',
'        else:',
'            break'];
const CODE_BUILD = [
'def build_heap(self, arr):',
'    self.items = arr[:]',
'    # 從最後一個「有小孩的」節點往前 percDown',
'    for i in range(len(arr) // 2 - 1, -1, -1):',
'        self.percDown(i)',
'    # 一個一個 insert 是 O(n log n)，這樣做是 O(n)'];
const CODE_SORT = [
'def heapsort(arr):',
'    h = BinaryHeap()',
'    h.build_heap(arr)           # O(n)',
'    return [h.del_min() for _ in range(len(arr))]',
'    # 每次拿走最小的 -> 出來就是排序好的 O(n log n)'];

function percUpFrames(F, arr, i, note){
  while (i > 0){
    const p = (i - 1) >> 1;
    if (arr[i] < arr[p]){
      F.push({shapes:view(arr, {[i]:'hot', [p]:'act'}), panels:P(arr), line:7,
        msg:{zh:'<b>' + arr[i] + '</b> 比父節點 <b>' + arr[p] + '</b> 小，違反規則 → 交換。' +
                (note || ''),
             en:'<b>' + arr[i] + '</b> is smaller than its parent <b>' + arr[p] + '</b>, which breaks the rule, so swap.'}});
      const t = arr[i]; arr[i] = arr[p]; arr[p] = t;
      F.push({shapes:view(arr, {[p]:'ok'}), panels:P(arr), line:11,
        msg:{zh:'往上升一層，繼續跟新的父節點比。<b>只沿著一條路往上，最多 log n 步</b>。',
             en:'Move up one level and compare with the new parent. <b>Only one path upward, at most log n steps.</b>'}});
      i = p;
    } else {
      F.push({shapes:view(arr, {[i]:'ok', [p]:'act'}), panels:P(arr), line:9,
        msg:{zh:'<b>' + arr[i] + '</b> 已經不小於父節點 <b>' + arr[p] + '</b>，停下來。' +
                '堆積<b>只要求父子關係</b>，不要求左右或整體有序。',
             en:'<b>' + arr[i] + '</b> is no longer smaller than its parent <b>' + arr[p] +
                '</b>, so stop. A heap only constrains <b>parent versus child</b> - not siblings, not global order.'}});
      return;
    }
  }
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr), line:5,
    msg:{zh:'一路浮到根：<b>' + arr[0] + '</b> 成了目前的最小值。',
         en:'It floated all the way to the root: <b>' + arr[0] + '</b> is now the minimum.'}});
}
function percDownFrames(F, arr, i, n, LM){
  n = n == null ? arr.length : n;
  LM = LM || {cmp:12, swap:14, stop:16, bottom:9};
  while (2 * i + 1 < n){
    let c = 2 * i + 1;
    if (c + 1 < n && arr[c + 1] < arr[c]) c++;
    F.push({shapes:view(arr, {[i]:'hot', [c]:'act'}), panels:P(arr), line:LM.cmp,
      msg:{zh:'跟<b>比較小的那個小孩</b> <b>' + arr[c] + '</b> 比。（一定要挑小的，' +
              '不然換上去的還是比另一個小孩大，又壞了。）',
           en:'Compare with the <b>smaller child</b>, <b>' + arr[c] + '</b>. It must be the smaller one - otherwise the promoted value would still exceed its sibling and break the rule again.'}});
    if (arr[c] < arr[i]){
      const t = arr[i]; arr[i] = arr[c]; arr[c] = t;
      F.push({shapes:view(arr, {[c]:'hot', [i]:'ok'}), panels:P(arr), line:LM.swap,
        msg:{zh:'交換後往下一層繼續。下沉的路徑也只有一條，<b>O(log n)</b>。',
             en:'Swap and continue one level down. The sinking path is a single line too, <b>O(log n)</b>.'}});
      i = c;
    } else {
      F.push({shapes:view(arr, {[i]:'ok'}), panels:P(arr), line:LM.stop,
        msg:{zh:'已經比兩個小孩都小，停。堆積條件恢復了。',
             en:'It is already smaller than both children, so stop. The heap property is restored.'}});
      return;
    }
  }
  F.push({shapes:view(arr, {}), panels:P(arr), line:LM.bottom,
    msg:{zh:'沉到底了（沒有小孩可以比），結束。',
         en:'It sank to the bottom - no children left to compare - so we are done.'}});
}

function buildInsert(){
  const F = new Frames(), arr = [];
  F.push({shapes:view(arr, {}), panels:P(arr), line:0,
    msg:{zh:'堆積的規則只有一條：<b>每個節點都不大於它的小孩</b>（最小堆）。' +
            '它<b>不是</b>排序好的樹，只保證最小值在根。而且整棵樹就存在一個陣列裡：' +
            '第 i 格的小孩是 2i+1 與 2i+2。',
         en:'A heap has one rule: <b>every node is no larger than its children</b> (a min-heap). It is <b>not</b> sorted - it only guarantees the minimum sits at the root. And the whole tree lives in one array: the children of i are 2i+1 and 2i+2.'}});
  [9, 6, 5, 2, 3].forEach(k => {
    arr.push(k);
    F.push({shapes:view(arr, {[arr.length - 1]:'hot'}), panels:P(arr), line:1,
      msg:{zh:'插入 <b>' + k + '</b>：先放在陣列最後一格，也就是樹的最後一個位置。' +
              '這樣<b>樹的形狀永遠是完整的</b>，不會有洞。',
           en:'Insert <b>' + k + '</b> at the end of the array, i.e. the last slot of the tree. This keeps the tree <b>complete</b> - no holes, ever.'}});
    if (arr.length > 1) percUpFrames(F, arr, arr.length - 1);
  });
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr), line:0,
    msg:{zh:'最後 <b>[' + arr.join(', ') + ']</b>。注意它<b>沒有排序</b>——' +
            '堆積只花力氣維持「根最小」，這正是它比排序便宜的原因。',
         en:'We end with <b>[' + arr.join(', ') + ']</b>. Notice it is <b>not sorted</b>: a heap only pays for "smallest at the root", which is exactly why it is cheaper than sorting.'}});
  return F.list;
}

function buildDel(){
  const arr = [2, 3, 5, 9, 6], F = new Frames(), out = [];
  const ex = () => ({lbl:{zh:'已取出', en:'popped'}, chips:chipsOf(out, 'ok')});
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr, ex()), line:1,
    msg:{zh:'最小值一定在根，所以 <b>peek 是 O(1)</b>。難的是拿走它以後怎麼補洞。',
         en:'The minimum is always at the root, so <b>peek is O(1)</b>. The hard part is filling the hole after you take it.'}});
  for (let k = 0; k < 2; k++){
    const top = arr[0];
    F.push({shapes:view(arr, {0:'hot', [arr.length - 1]:'act'}), panels:P(arr, ex()), line:2,
      msg:{zh:'取走 <b>' + top + '</b>，然後把<b>最後一格</b>的 <b>' + arr[arr.length - 1] +
              '</b> 搬到根。為什麼是最後一格？因為只有這樣樹才會保持完整。',
           en:'Take <b>' + top + '</b>, then move the <b>last slot</b> value <b>' + arr[arr.length - 1] +
              '</b> to the root. Why the last one? Because only that keeps the tree complete.'}});
    out.push(top);
    arr[0] = arr[arr.length - 1]; arr.pop();
    F.push({shapes:view(arr, {0:'hot'}), panels:P(arr, ex()), line:4,
      msg:{zh:'現在根上是個大數字，規則壞了 → 讓它<b>下沉</b>。',
           en:'Now a large value sits at the root and the rule is broken - let it <b>sink</b>.'}});
    percDownFrames(F, arr, 0);
  }
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr, ex()), line:5,
    msg:{zh:'連續取出 <b>' + out.join(', ') + '</b>——由小到大。這就是優先佇列：' +
            '插入與取出都是 <b>O(log n)</b>，之後在最短路徑與霍夫曼編碼裡都靠它。',
         en:'We popped <b>' + out.join(', ') + '</b> in increasing order. That is a priority queue: insert and pop are both <b>O(log n)</b>, and shortest-path and Huffman coding both lean on it.'}});
  return F.list;
}

function buildBuild(){
  const arr = [9, 4, 7, 1, 8, 2, 6], F = new Frames();
  F.push({shapes:view(arr, {}), panels:P(arr), line:1,
    msg:{zh:'給一個亂序陣列 <b>[' + arr.join(', ') + ']</b>，要把它變成堆積。' +
            '直覺做法是一個一個 insert，那是 <b>O(n log n)</b>。有更快的。',
         en:'Given the unsorted array <b>[' + arr.join(', ') + ']</b>, turn it into a heap. The obvious way is n inserts, <b>O(n log n)</b>. There is a faster way.'}});
  const start = (arr.length >> 1) - 1;
  const leafSt = {}; for (let i = start + 1; i < arr.length; i++) leafSt[i] = 'done';
  F.push({shapes:view(arr, leafSt), panels:P(arr), line:3,
    msg:{zh:'關鍵觀察：<b>一半以上的節點是葉子</b>（灰色那些），葉子本身就已經是合法的堆積，' +
            '完全不用處理。所以從索引 <b>' + start + '</b> 開始往前做就好。',
         en:'Key observation: <b>more than half the nodes are leaves</b> (greyed out) and a leaf is already a valid heap, so they need no work at all. Start at index <b>' + start + '</b> and walk backwards.'}});
  for (let i = start; i >= 0; i--){
    F.push({shapes:view(arr, Object.assign({}, leafSt, {[i]:'hot'})), panels:P(arr), line:4,
      msg:{zh:'處理索引 <b>' + i + '</b>（值 <b>' + arr[i] + '</b>）：它的兩棵子樹<b>已經</b>是堆積了，' +
              '所以只要讓它自己下沉就好。',
           en:'Handle index <b>' + i + '</b> (value <b>' + arr[i] + '</b>): both its subtrees are <b>already</b> heaps, so it only needs to sink into place.'}});
    percDownFrames(F, arr, i, null, {cmp:4, swap:4, stop:4, bottom:4});
  }
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr), line:5,
    msg:{zh:'完成：<b>[' + arr.join(', ') + ']</b>。看起來每個節點都可能沉 log n 層，但' +
            '<b>越深的節點越多、能沉的距離卻越短</b>，加起來收斂成 <b>O(n)</b>——比 n 次插入快一個級數。',
         en:'Done: <b>[' + arr.join(', ') + ']</b>. It looks like every node could sink log n levels, but <b>the deeper a node is, the more of them there are and the less far they can sink</b>; the sum collapses to <b>O(n)</b> - a whole factor better than n inserts.'}});
  return F.list;
}

function buildSort(){
  const arr = [9, 4, 7, 1, 8, 2, 6], F = new Frames(), out = [];
  const ex = () => ({lbl:{zh:'輸出（已排序）', en:'output (sorted)'}, chips:chipsOf(out, 'ok')});
  for (let i = (arr.length >> 1) - 1; i >= 0; i--){
    let j = i;
    while (2 * j + 1 < arr.length){
      let c = 2 * j + 1;
      if (c + 1 < arr.length && arr[c + 1] < arr[c]) c++;
      if (arr[c] < arr[j]){ const t = arr[j]; arr[j] = arr[c]; arr[c] = t; j = c; } else break;
    }
  }
  F.push({shapes:view(arr, {0:'ok'}), panels:P(arr, ex()), line:2,
    msg:{zh:'先用 <b>O(n)</b> 建堆，得到 <b>[' + arr.join(', ') + ']</b>。接下來只要一直拿走根就好。',
         en:'Build the heap in <b>O(n)</b> to get <b>[' + arr.join(', ') + ']</b>. From here we just keep taking the root.'}});
  while (arr.length){
    const top = arr[0];
    F.push({shapes:view(arr, {0:'hot'}), panels:P(arr, ex()), line:3,
      msg:{zh:'拿走根 <b>' + top + '</b>——它保證是<b>目前剩下最小的</b>。',
           en:'Take the root <b>' + top + '</b>, guaranteed to be the <b>smallest of what remains</b>.'}});
    out.push(top);
    arr[0] = arr[arr.length - 1]; arr.pop();
    if (arr.length){
      let j = 0;
      while (2 * j + 1 < arr.length){
        let c = 2 * j + 1;
        if (c + 1 < arr.length && arr[c + 1] < arr[c]) c++;
        if (arr[c] < arr[j]){ const t = arr[j]; arr[j] = arr[c]; arr[c] = t; j = c; } else break;
      }
      F.push({shapes:view(arr, {0:'act'}), panels:P(arr, ex()), line:3,
        msg:{zh:'補洞、下沉，新的最小值 <b>' + arr[0] + '</b> 又浮到根。' +
                '每次只花 <b>O(log n)</b>，總共 n 次。',
             en:'Fill the hole, sink, and the new minimum <b>' + arr[0] +
                '</b> surfaces at the root. Each pop costs <b>O(log n)</b>, n times over.'}});
    }
  }
  F.push({shapes:[S.t(5, 3.2, out.join('   '), {c:'#3fe0dd', fs:.62}),
                  S.t(5, 4.1, {zh:'已排序', en:'sorted'}, {c:'#8fa3ac', fs:.36})],
    panels:[{lbl:{zh:'輸出', en:'output'}, chips:chipsOf(out, 'ok')}], line:4,
    msg:{zh:'結果 <b>' + out.join(', ') + '</b>，總成本 <b>O(n log n)</b>，' +
            '而且<b>最壞情況也是 O(n log n)</b>（快速排序沒有這個保證）。' +
            '真正實作時還能就地做，不用額外空間。',
         en:'Result <b>' + out.join(', ') + '</b> at <b>O(n log n)</b> - and unlike quicksort, that is also the <b>worst case</b>. A real implementation can even do it in place, with no extra array.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'二元堆積（優先佇列）', en:'Binary Heap (Priority Queue)'},
  sub:{zh:'一個陣列假裝自己是一棵樹：上浮、下沉、O(n) 建堆，還有幾乎免費的堆積排序。',
       en:'An array pretending to be a tree: percolate up, percolate down, O(n) build, and a nearly free sort.'},
  tabs:[
    {id:'ins', label:{zh:'插入（上浮）', en:'insert / percUp'},
     stage:{zh:'放到最後一格，再往上浮', en:'Append at the end, then float up'}, view:[10, 7.0],
     idea:{zh:'堆積是<b>刻意做得不完全排序</b>的資料結構：只維持父子大小關係，' +
              '所以插入只要沿一條路往上比，<b>O(log n)</b>。完整樹的形狀讓它可以用陣列存，' +
              '沒有任何指標開銷。',
           en:'A heap is <b>deliberately only half-ordered</b>: it maintains the parent-child relation and nothing else, so an insert compares along a single upward path, <b>O(log n)</b>. Being a complete tree lets it live in a flat array with zero pointer overhead.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_INS, build:buildInsert},
    {id:'del', label:{zh:'取出最小（下沉）', en:'del_min / percDown'},
     stage:{zh:'最後一格補到根，再往下沉', en:'Last slot to the root, then sink'}, view:[10, 7.0],
     idea:{zh:'取出的難點是<b>補洞不能破壞樹的形狀</b>，所以拿最後一格來補，再讓它下沉。' +
              '下沉時一定要跟<b>比較小的那個小孩</b>換，否則規則還是壞的。',
           en:'The trick is that filling the hole must not break the shape, so the last slot fills it and then sinks. Always swap with the <b>smaller child</b>, or the rule stays broken.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_DEL, build:buildDel},
    {id:'build', label:{zh:'O(n) 建堆', en:'build_heap in O(n)'},
     stage:{zh:'從最後一個非葉節點往前下沉', en:'Sink from the last non-leaf backwards'}, view:[10, 7.0],
     idea:{zh:'一半的節點是葉子，本來就合法；剩下的由下往上處理時，<b>子樹已經是堆積</b>，' +
              '所以每個節點只需一次下沉。總和是 <b>O(n)</b> 而不是 O(n log n)——' +
              '這是「由下往上」比「由上往下」便宜的經典例子。',
           en:'Half the nodes are leaves and already valid; processing the rest bottom-up means <b>the subtrees are already heaps</b>, so each node sinks once. The total is <b>O(n)</b>, not O(n log n) - a classic case of bottom-up beating top-down.'},
     legend:['hot', 'act', 'ok', 'done'], code:CODE_BUILD, build:buildBuild},
    {id:'sort', label:{zh:'堆積排序', en:'heapsort'},
     stage:{zh:'一直拿走根', en:'Keep taking the root'}, view:[10, 7.0],
     idea:{zh:'排序只是「一直取最小」的副產品：<b>O(n)</b> 建堆 + <b>n</b> 次 <b>O(log n)</b> 取出。' +
              '它的賣點不是最快，而是<b>最壞情況仍然 O(n log n)</b>、而且可以就地完成。',
           en:'Sorting falls out of "keep taking the minimum": <b>O(n)</b> to build plus <b>n</b> pops at <b>O(log n)</b>. Its selling point is not raw speed but that <b>the worst case is still O(n log n)</b>, in place.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_SORT, build:buildSort}
  ]
};
