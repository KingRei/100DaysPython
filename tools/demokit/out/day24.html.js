// DAY: 24
// TITLE_ZH: Trie 與 Radix Tree：把共用的前綴只存一次
// TITLE_EN: Tries and radix trees - storing a shared prefix exactly once
// SUB_ZH: Trie 把「每個前綴都是一個節點」當成資料結構；radix tree 再把沒有分岔的那一串壓成一條邊。自動完成、IP 路由表、還有 LLM 伺服器的 KV cache，用的都是同一棵樹。
// SUB_EN: A trie makes "every prefix is a node" the data structure itself; a radix tree then collapses every chain without a branch into one edge. Autocomplete, IP routing tables and an LLM server's KV cache are all the same tree.
// FOLDER: day%2024%20-%20trie%20and%20radix%20tree
// MEDIUM: https://medium.com/100-days-of-python

const VIEW = [9.8, 6.4];
const mid2 = VIEW[0] / 2;
const note = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});
const foot = (s, y) => S.t(mid2, y == null ? 6.15 : y, s, {c:COL.grey, fs:.27});
const ln = (code, frag) => { const i = code.findIndex(l => l.indexOf(frag) >= 0); return i < 0 ? 0 : i; };

function norm(list){
  list.forEach(f => {
    if (f.panels && f.panels.length && f.panels[0].chips === undefined)
      f.panels = [{lbl:{zh:'狀態', en:'state'}, chips:f.panels}];
  });
  return list;
}

/* ----------------------------------------------------------------- trees
   one node shape for both structures:
     n.ch[firstChar] = {lab: <edge label>, node: <child>}
   a trie is the special case where every lab has length 1.            */
function TN(w){ return {ch:{}, w:!!w}; }

function layoutLR(root, x0, y0, dx, dy){
  const pos = new Map(); let row = 0;
  (function walk(n, d){
    const ks = Object.keys(n.ch);
    if (!ks.length){ pos.set(n, {x:x0 + d * dx, y:y0 + (row++) * dy}); return; }
    ks.forEach(k => walk(n.ch[k].node, d + 1));
    const ys = ks.map(k => pos.get(n.ch[k].node).y);
    pos.set(n, {x:x0 + d * dx, y:(Math.min(...ys) + Math.max(...ys)) / 2});
  })(root, 0);
  return pos;
}

/* st: Map(node -> style key), 'gone' hides a node and its incoming edge */
function treeShapes(root, pos, st, opt){
  opt = opt || {};
  const R = opt.r || .28, out = [], edges = [], nodes = [];
  (function walk(n, inLab){
    const p = pos.get(n), s = st.get(n) || 'idle';
    if (s !== 'gone'){
      if (n.w) nodes.push(S.c(p.x, p.y, R + .11, s === 'idle' ? 'ok' : s, ''));
      nodes.push(S.c(p.x, p.y, R, s, opt.inNode ? (inLab || '') : '',
                     {fs:.34, sub:opt.sub ? opt.sub(n) : null,
                      subc:s === 'hot' ? COL.orangeL : COL.tealL}));
    }
    Object.keys(n.ch).forEach(k => {
      const e = n.ch[k], c = e.node, cs = st.get(c) || 'idle';
      if (cs !== 'gone' && s !== 'gone'){
        const q = pos.get(c);
        edges.push(S.e(p.x, p.y, q.x, q.y, {pad:R + .06,
          s:(cs === 'hot' || cs === 'act' || cs === 'bad') ? cs : 'idle'}));
        if (!opt.inNode)
          edges.push(S.t((p.x + q.x) / 2, (p.y + q.y) / 2 - .18, e.lab,
            {c:cs === 'hot' ? COL.orangeL : (cs === 'bad' ? COL.red : COL.tealL), fs:.31}));
      }
      walk(c, e.lab);
    });
  })(root, '');
  return out.concat(edges, nodes);
}
function countNodes(root){
  let n = 0;
  (function walk(x){ n++; Object.keys(x.ch).forEach(k => walk(x.ch[k].node)); })(root);
  return n;
}

/* ==================================================== 1. the trie itself */
const WORDS1 = ['cat', 'car', 'dog'];
const CODE_T = [
'class TrieNode:',
'    __slots__ = ("children", "is_word")',
'    def __init__(self):',
'        self.children = {}          # one entry per character',
'        self.is_word  = False       # "a word ends here"',
'',
'def insert(root, word):',
'    node = root',
'    for ch in word:',
'        if ch not in node.children:',
'            node.children[ch] = TrieNode()',
'        node = node.children[ch]',
'    node.is_word = True',
'',
'def find(root, word):',
'    node = root',
'    for ch in word:',
'        node = node.children.get(ch)',
'        if node is None:',
'            return None             # no such prefix at all',
'    return node',
'',
'# search(w)      -> find(root, w) is not None and node.is_word',
'# startsWith(p)  -> find(root, p) is not None'
];

function trieOf(words){
  const root = TN();
  words.forEach(w => {
    let n = root;
    for (const c of w){
      if (!n.ch[c]) n.ch[c] = {lab:c, node:TN()};
      n = n.ch[c].node;
    }
    n.w = true;
  });
  return root;
}
function trieWalk(root, word){
  const out = [root]; let n = root;
  for (const c of word){
    const e = n.ch[c];
    if (!e){ out.push(null); return out; }
    n = e.node; out.push(n);
  }
  return out;
}

function trieFrames(variant){
  const F = new Frames();
  const full = trieOf(WORDS1.concat(['card']));
  const pos = layoutLR(full, 1.55, 1.60, 1.45, 1.30);
  const cardNode = trieWalk(full, 'card')[4];
  const base = () => {
    const st = new Map();
    if (variant === 0) st.set(cardNode, 'gone');
    return st;
  };
  const draw = (st, cap, capc) => {
    const s = treeShapes(full, pos, st, {inNode:true, r:.30,
      sub:n => n === trieWalk(full, 'cat')[3] ? 'cat'
             : n === trieWalk(full, 'car')[3] ? 'car'
             : n === cardNode ? 'card'
             : n === trieWalk(full, 'dog')[3] ? 'dog' : null});
    s.push(note(.78, cap, capc || COL.tealL, .34));
    return s;
  };

  if (variant === 0){
    const st0 = base();
    F.push({shapes:draw(st0, 'stored: cat, car, dog   -   ' + (countNodes(full) - 1) + ' nodes'),
      panels:[{t:'words 3', cls:'act'}], view:VIEW, line:ln(CODE_T, 'def insert'),
      msg:{zh:'三個單字，八個節點。注意樹上<b>沒有任何地方存著完整的字串</b> - 字母全在邊上，一個單字就是從根走下來的一條路徑。所以「所有以 ca 開頭的字」不用搜尋，它就是 ca 底下那棵子樹。',
           en:'Three words, eight nodes. Notice that <b>nowhere in the tree is a whole string stored</b> - the letters live on the edges and a word is just a path down from the root. That is why "every word starting with ca" needs no search at all: it is the subtree hanging under ca.'}});
    const path = trieWalk(full, 'card');
    for (let i = 1; i <= 4; i++){
      const st = base();
      for (let j = 1; j < i; j++) st.set(path[j], 'done');
      const n = path[i], isNew = (i === 4);
      if (isNew) st.delete(cardNode);
      st.set(n, isNew ? 'hot' : 'act');
      const ch = 'card'[i - 1];
      F.push({shapes:draw(st, isNew ? 'insert "card"  -  one new node' : 'insert "card"  -  reusing "' + 'card'.slice(0, i) + '"',
                          isNew ? COL.orangeL : COL.tealL),
        panels:[{t:'char ' + ch, cls:'act'}, {t:'depth ' + i, cls:''},
                {t:isNew ? 'new node' : 'already there', cls:isNew ? 'hot' : 'ok'}],
        view:VIEW, line:isNew ? ln(CODE_T, 'node.children[ch] = TrieNode()') : ln(CODE_T, 'node = node.children[ch]'),
        msg:isNew
          ? {zh:'走到 "car" 之後才發現沒有 d 這個小孩，於是<b>只</b>配置一個新節點。第四個單字只花了一個節點 - 因為 c、a、r 早就被前三個字建好了，插入的成本是<b>字串長度</b>，跟樹裡已經有多少字完全無關。',
             en:'Only after "car" is there no d child, so we allocate <b>one</b> new node. The fourth word cost a single node, because c, a and r were already built by the first three - insertion costs <b>the length of the word</b> and is completely independent of how many words the tree already holds.'}
          : {zh:'字元 "' + ch + '" 已經在 children 裡，直接往下走，什麼都不配置。這一步就是 trie 省空間的來源：<b>共用的前綴只存一次</b>。',
             en:'The character "' + ch + '" is already in children, so we walk down and allocate nothing. This step is where a trie\'s savings come from: <b>a shared prefix is stored exactly once</b>.'}});
    }
    const stF = new Map(); stF.set(cardNode, 'ok');
    F.push({shapes:draw(stF, 'stored: cat, car, card, dog   -   ' + countNodes(full) + ' nodes', COL.tealL),
      panels:[{t:'words 4', cls:'act'}, {t:'nodes 9', cls:'ok'}, {t:'+1 node', cls:'ok'}],
      view:VIEW, line:ln(CODE_T, 'node.is_word = True'),
      msg:{zh:'最後把 is_word 打開。這個旗標不是裝飾：<b>"car" 是一個字，但 "ca" 只是路過</b>，兩者在樹上長得一模一樣，差別只有這個布林值。這也是 LC 208 裡 search 和 startsWith 唯一的差別。',
           en:'Finally we set is_word. That flag is not decoration: <b>"car" is a word but "ca" is only passed through</b>, and on the tree they look identical - the boolean is the only difference. It is also the entire difference between search and startsWith in LC 208.'}});
    return F.list;
  }

  /* variant 1: search vs startsWith */
  const show = (word, upto, res) => {
    const st = new Map();
    const path = trieWalk(full, word);
    for (let j = 1; j <= upto; j++) st.set(path[j], j === upto ? 'hot' : 'done');
    return {st:st, node:path[upto]};
  };
  const q = ['c', 'a', 'r'];
  for (let i = 1; i <= 3; i++){
    const v = show('car', i);
    F.push({shapes:draw(v.st, 'find(root, "car")  -  step ' + i + ' of 3'),
      panels:[{t:'ch ' + q[i - 1], cls:'act'}, {t:'node found', cls:'ok'},
              {t:'is_word ' + (v.node.w ? 'True' : 'False'), cls:v.node.w ? 'ok' : ''}],
      view:VIEW, line:ln(CODE_T, 'node = node.children.get(ch)'),
      msg:{zh:'一次一個字元往下走，成本是 <b>O(len(word))</b>，不是 O(log n)，也不是 O(n)。樹裡有一千個字還是一百萬個字，走 "car" 都只是三步。',
           en:'One character at a time down the tree: the cost is <b>O(len(word))</b>, not O(log n) and not O(n). With a thousand words or a million in the tree, walking "car" is still three steps.'}});
  }
  const vr = show('car', 3);
  F.push({shapes:draw(vr.st, 'search("car") = True', COL.tealL),
    panels:[{t:'node exists', cls:'ok'}, {t:'is_word True', cls:'ok'}, {t:'search True', cls:'ok'}],
    view:VIEW, line:ln(CODE_T, '# search(w)'),
    msg:{zh:'節點在，而且 is_word 是 True，所以 "car" 真的是一個字。',
         en:'The node exists and is_word is True, so "car" really is a stored word.'}});
  const vc = show('ca', 2);
  F.push({shapes:draw(vc.st, 'search("ca") = False   but   startsWith("ca") = True', COL.orangeL),
    panels:[{t:'node exists', cls:'ok'}, {t:'is_word False', cls:'bad'},
            {t:'search False', cls:'bad'}, {t:'startsWith True', cls:'ok'}],
    view:VIEW, line:ln(CODE_T, '# startsWith'),
    msg:{zh:'同一個節點，兩個答案。<b>「這個前綴存在」和「這個字存在」是不同的問題</b>，而 trie 是少數能一次回答兩題的結構 - hash set 只能答第二題，答第一題得把所有前綴也一起塞進去。這就是自動完成為什麼用 trie。',
         en:'One node, two answers. <b>"this prefix exists" and "this word exists" are different questions</b>, and a trie is one of the few structures that answers both in one walk - a hash set can only answer the second unless you also insert every prefix of every word. That is why autocomplete is built on a trie.'}});
  const vx = show('cab', 2);
  vx.st.set(trieWalk(full, 'ca')[2], 'bad');
  F.push({shapes:draw(vx.st, 'find(root, "cab")  -  no "b" child', COL.red),
    panels:[{t:'ch b', cls:'bad'}, {t:'children a...', cls:''}, {t:'return None', cls:'bad'}],
    view:VIEW, line:ln(CODE_T, 'return None'),
    msg:{zh:'找不到的情況更便宜：走到沒有那個小孩就<b>立刻</b>結束，不必走完整個字。前綴不存在 = 子樹不存在，這是同一句話。',
         en:'Failure is even cheaper: the walk stops the <b>moment</b> a child is missing, without reading the rest of the word. "the prefix does not exist" and "the subtree does not exist" are the same sentence here.'}});
  return F.list;
}

/* ================================================== 2. radix / PATRICIA */
const CODE_R = [
'def insert(self, word):',
'    node, i = self.root, 0',
'    while i < len(word):',
'        first = word[i]',
'        edge = node.children.get(first)    # O(1) - keyed by first char',
'        if edge is None:',
'            node.children[first] = (word[i:], RadixNode(is_word=True))',
'            return                         # one edge carries the whole tail',
'        label, child = edge',
'        k = common_prefix_len(label, word[i:])',
'        if k == len(label):',
'            node, i = child, i + k         # whole edge matched: descend',
'            continue',
'        mid = RadixNode()                  # split at the divergence',
'        mid.children[label[k]] = (label[k:], child)',
'        node.children[first] = (label[:k], mid)',
'        i += k',
'        if i == len(word):',
'            mid.is_word = True',
'        else:',
'            mid.children[word[i]] = (word[i:], RadixNode(is_word=True))',
'        return',
'    node.is_word = True'
];

function radixFrames(words){
  const F = new Frames();
  const root = TN();
  const draw = (st, cap, capc, sub2) => {
    const pos = layoutLR(root, 1.35, 1.75, 1.95, 1.15);
    const s = treeShapes(root, pos, st, {r:.26, sub:n => n.wl || null});
    s.push(note(.80, cap, capc || COL.tealL, .34));
    if (sub2) s.push(note(5.95, sub2, COL.grey, .29));
    return s;
  };
  const trieN = countNodes(trieOf(words));

  words.forEach(word => {
    let node = root, i = 0, guard = 0, done = false;
    F.push({shapes:draw(new Map(), 'insert "' + word + '"', COL.orangeL),
      panels:[{t:'word ' + word, cls:'act'}, {t:'nodes ' + countNodes(root), cls:''}],
      view:VIEW, line:ln(CODE_R, 'node, i = self.root'),
      msg:{zh:'插入 "' + word + '"。radix tree 的節點<b>不是字元</b>，是分岔點；字串放在<b>邊</b>上，所以插入只有三種結局：接一條新邊、沿著邊走下去、或是把一條邊<b>剖開</b>。',
           en:'Inserting "' + word + '". A radix node is <b>not a character</b>, it is a branch point; the strings live on the <b>edges</b>, so an insert can only end three ways: hang a new edge, walk down an existing one, or <b>split</b> one.'}});

    while (i < word.length && guard++ < 20){
      const first = word[i], rest = word.slice(i), e = node.ch[first];
      if (!e){
        const nn = TN(true); nn.wl = word;
        node.ch[first] = {lab:rest, node:nn};
        const st = new Map(); st.set(nn, 'hot');
        F.push({shapes:draw(st, 'no child starts with "' + first + '"  ->  one new edge "' + rest + '"', COL.orangeL),
          panels:[{t:'first ' + first, cls:'act'}, {t:'new edge ' + rest, cls:'hot'},
                  {t:'nodes ' + countNodes(root), cls:''}],
          view:VIEW, line:ln(CODE_R, 'node.children[first] = (word[i:]'),
          msg:{zh:'沒有小孩是 "' + first + '" 開頭，所以剩下的 "' + rest + '" <b>整段</b>掛成一條邊，只用一個節點。children 用<b>第一個字元</b>當 key，是為了讓「該走哪條邊」保持 O(1) - 否則就得逐條比對。',
               en:'No child starts with "' + first + '", so the entire remaining "' + rest + '" hangs off as one edge and costs a single node. children is keyed by the <b>first character</b> precisely so that choosing an edge stays O(1) - otherwise you would have to compare every edge.'}});
        done = true; break;
      }
      const label = e.lab;
      let k = 0;
      while (k < label.length && k < rest.length && label[k] === rest[k]) k++;
      if (k === label.length){
        node = e.node; i += k;
        const st = new Map(); st.set(node, 'act');
        F.push({shapes:draw(st, 'edge "' + label + '" matched in full  ->  descend', COL.tealL),
          panels:[{t:'edge ' + label, cls:'ok'}, {t:'matched ' + k, cls:'ok'}, {t:'left ' + word.slice(i), cls:'act'}],
          view:VIEW, line:ln(CODE_R, 'node, i = child, i + k'),
          msg:{zh:'邊上的 "' + label + '" 整段都吻合，一步就吃掉 ' + k + ' 個字元。<b>trie 要走 ' + k + ' 個節點的地方，radix tree 只碰一個</b> - 這就是壓縮換到的東西。',
               en:'The whole label "' + label + '" matches, so one step consumes ' + k + ' characters. <b>Where a trie would visit ' + k + ' nodes, the radix tree touches one</b> - that is what the compression buys.'}});
        continue;
      }
      /* --- split --- */
      const st0 = new Map(); st0.set(e.node, 'bad');
      F.push({shapes:draw(st0, '"' + label + '" vs "' + rest + '"  -  they agree on "' + label.slice(0, k) + '" then diverge', COL.orangeL),
        panels:[{t:'label ' + label, cls:''}, {t:'rest ' + rest, cls:'act'},
                {t:'common ' + k, cls:'hot'}, {t:'split', cls:'bad'}],
        view:VIEW, line:ln(CODE_R, 'k = common_prefix_len'),
        msg:{zh:'這條邊只對了前 ' + k + ' 個字元（"' + label.slice(0, k) + '"），之後就分岔了。一條邊代表「這裡沒有分岔」，現在有了，所以這條邊<b>必須被剖開</b>。這是整個資料結構唯一麻煩的操作。',
             en:'The edge only agrees for the first ' + k + ' characters ("' + label.slice(0, k) + '") and then diverges. An edge means "no branch happens here", and now one does, so the edge <b>has to be split</b>. This is the only fiddly operation in the whole structure.'}});
      const mid = TN();
      mid.ch[label[k]] = {lab:label.slice(k), node:e.node};
      node.ch[first] = {lab:label.slice(0, k), node:mid};
      i += k;
      let tail = null;
      if (i === word.length){ mid.w = true; mid.wl = word; }
      else { tail = TN(true); tail.wl = word; mid.ch[word[i]] = {lab:word.slice(i), node:tail}; }
      const st1 = new Map(); st1.set(mid, 'hot'); if (tail) st1.set(tail, 'ok');
      F.push({shapes:draw(st1, 'split done: "' + label + '"  ->  "' + label.slice(0, k) + '" + "' + label.slice(k) + '"', COL.orangeL),
        panels:[{t:'new node', cls:'hot'}, {t:'"' + label.slice(0, k) + '"', cls:'ok'},
                {t:'"' + label.slice(k) + '"', cls:'ok'}, {t:tail ? '"' + word.slice(i) + '"' : 'is_word', cls:'ok'},
                {t:'nodes ' + countNodes(root), cls:''}],
        view:VIEW, line:ln(CODE_R, 'mid.children[label[k]]'),
        msg:{zh:'共用的頭 "' + label.slice(0, k) + '" 變成一個自己的節點，兩條尾巴掛在它下面。<b>沒有任何字串被複製</b> - 舊的子樹整棵原封不動地接過去，只是那條邊的界線往前移了。LLM 伺服器的前綴快取每次遇到「兩個 request 共用前面、後面分開」跑的就是這一段。',
             en:'The shared head "' + label.slice(0, k) + '" becomes a node of its own and the two tails hang under it. <b>Nothing is copied</b> - the old subtree is reparented untouched and only the boundary of that edge moved. An LLM server\'s prefix cache runs exactly this code every time two requests share a prompt and then say different things.'}});
      done = true; break;
    }
    if (!done && i >= word.length && !node.w){
      node.w = true; node.wl = word;
      const st = new Map(); st.set(node, 'ok');
      F.push({shapes:draw(st, '"' + word + '" is already a node on the path', COL.tealL),
        panels:[{t:'is_word = True', cls:'ok'}], view:VIEW, line:ln(CODE_R, 'node.is_word = True'),
        msg:{zh:'字走完了，剛好停在一個已經存在的節點上，只要把 is_word 打開。',
             en:'The word ran out exactly on an existing node, so only the is_word flag changes.'}});
    }
  });

  const rn = countNodes(root);
  F.push({shapes:draw(new Map(), words.join(', ') + '   -   ' + rn + ' radix nodes vs ' + trieN + ' trie nodes', COL.tealL,
                      'lookup still costs O(len(word)); what changed is how many nodes - and how many pointer hops - that walk touches'),
    panels:[{t:'radix ' + rn, cls:'ok'}, {t:'trie ' + trieN, cls:''},
            {t:Math.round(100 * (1 - rn / trieN)) + '% fewer', cls:'ok'}],
    view:VIEW, line:ln(CODE_R, 'def insert'),
    msg:{zh:'同樣的字，節點從 ' + trieN + ' 個變成 ' + rn + ' 個。省下來的不只是記憶體：每個節點都是一次<b>指標跳躍</b>加一個 dict，走一長串沒有分岔的字元卻要跳十幾次，是 trie 真正慢的地方。radix tree 把那一整串變成一次字串比較。',
         en:'Same words, ' + trieN + ' nodes become ' + rn + '. The saving is not only memory: every node is a <b>pointer hop</b> plus a dict, and hopping a dozen times through a chain that never branches is where a trie actually loses time. A radix tree turns that chain into one string comparison.'}});
  return F.list;
}

/* ============================================== 3. longest prefix match */
const CODE_P = [
'def lookup(root, ip):',
'    node = root',
'    best = root.rule                  # the default route, 0.0.0.0/0',
'    for part in octets(ip):           # walk the address, not the table',
'        edge = node.children.get(part)',
'        if edge is None:',
'            break                     # nothing more specific exists',
'        node = edge',
'        if node.rule is not None:',
'            best = node.rule          # deeper == more specific',
'    return best                       # the deepest marked node we passed'
];

function lpmTree(){
  const root = TN(true); root.wl = '/0 isp-default';
  const c = TN(true); c.wl = '/8 corp-core';
  const b = TN(true); b.wl = '/16 site-b';
  const r = TN(true); r.wl = '/24 rack-7';
  const l = TN(true); l.wl = '/16 lab';
  root.ch['10.'] = {lab:'10.', node:c};
  c.ch['20.'] = {lab:'20.', node:b};
  b.ch['30.'] = {lab:'30.', node:r};
  root.ch['192.'] = {lab:'192.168.', node:l};
  return {root:root, c:c, b:b, r:r, l:l};
}

function lpmFrames(variant){
  const F = new Frames();
  const T4 = lpmTree(), root = T4.root;
  const ip = variant ? '10.20.99.1' : '10.20.30.5';
  const parts = variant ? ['10.', '20.', '99.'] : ['10.', '20.', '30.'];
  const pos = layoutLR(root, 1.25, 2.05, 2.15, 1.45);
  const draw = (st, cap, capc, extra) => {
    const s = treeShapes(root, pos, st, {r:.28, sub:n => n.wl || null});
    s.push(note(.80, cap, capc || COL.tealL, .34));
    if (extra) s.push(note(5.85, extra, COL.grey, .29));
    return s;
  };
  let node = root, best = root, st = new Map();
  st.set(root, 'act');
  F.push({shapes:draw(st, 'lookup ' + ip + '   -   5 rules, none of them ordered', COL.tealL,
                      'a routing table is a set of prefixes; the answer is always the most specific one that matches'),
    panels:[{t:'ip ' + ip, cls:'act'}, {t:'best ' + best.wl, cls:'ok'}],
    view:VIEW, line:ln(CODE_P, 'best = root.rule'),
    msg:{zh:'路由表不是一串規則清單，而是一棵前綴樹。開場的答案是預設路由 <b>0.0.0.0/0</b>：它匹配所有位址，所以永遠有答案，不需要「找不到」的特例。',
         en:'A routing table is not a list of rules, it is a prefix tree. The starting answer is the default route <b>0.0.0.0/0</b>: it matches every address, so there is always an answer and no not-found case to write.'}});

  for (let i = 0; i < parts.length; i++){
    const key = parts[i], e = node.ch[key];
    if (!e){
      const st2 = new Map(); st2.set(node, 'bad'); st2.set(best, 'ok');
      F.push({shapes:draw(st2, 'no "' + key + '" child  ->  stop walking', COL.red,
                          'the walk is bounded by the address (32 bits), never by the size of the table'),
        panels:[{t:'part ' + key, cls:'bad'}, {t:'no child', cls:'bad'}, {t:'answer ' + best.wl, cls:'ok'}],
        view:VIEW, line:ln(CODE_P, 'break'),
        msg:{zh:'"' + key + '" 沒有小孩，代表<b>沒有更精確的規則</b>了，於是停下來，答案就是路上經過的<b>最深的有標記節點</b> - ' + best.wl + '。注意這裡沒有回溯：走過的最深標記早就記在 best 裡。',
             en:'There is no "' + key + '" child, which means <b>no more specific rule exists</b>, so the walk stops and the answer is the <b>deepest marked node it passed</b> - ' + best.wl + '. Note there is no backtracking: the deepest mark was recorded in best on the way down.'}});
      break;
    }
    node = e.node;
    const better = node.w;
    if (better) best = node;
    const st2 = new Map(); st2.set(node, 'hot'); if (best !== node) st2.set(best, 'ok');
    F.push({shapes:draw(st2, 'matched "' + key + '"  ->  best = ' + best.wl, better ? COL.orangeL : COL.tealL),
      panels:[{t:'part ' + key, cls:'act'}, {t:'depth ' + (i + 1), cls:''}, {t:'best ' + best.wl, cls:'ok'}],
      view:VIEW, line:ln(CODE_P, 'best = node.rule'),
      msg:{zh:'吃掉 "' + key + '"，落在 <b>' + node.wl + '</b> 上。它比之前的答案更深，也就更精確，所以 best 更新。<b>「最長前綴匹配」在樹上就是「走到最深的有標記節點」</b> - 規則的先後順序從來沒有進入這段程式碼。',
           en:'Consuming "' + key + '" lands on <b>' + node.wl + '</b>. It sits deeper than the previous answer, so it is more specific and best moves. <b>On a tree, "longest prefix match" is simply "the deepest marked node"</b> - the order the rules were added never enters this code.'}});
  }
  const stF = new Map(); stF.set(best, 'ok');
  F.push({shapes:draw(stF, ip + '  ->  ' + best.wl, COL.tealL,
                      'a linear scan would compare every rule and then still have to pick the longest match'),
    panels:[{t:ip, cls:'act'}, {t:best.wl, cls:'ok'}, {t:'hops <= 4', cls:'ok'}, {t:'rules scanned 0', cls:'ok'}],
    view:VIEW, line:ln(CODE_P, 'return best'),
    msg:{zh:'答案是 <b>' + best.wl + '</b>。重點在成本：這棵樹是<b>照著位址走</b>，不是照著表走，所以 4 個 octet（真實世界是 32 個 bit）就結束了 - <b>表裡有 5 條還是 90 萬條規則，時間一樣</b>。線性掃描則要比對每一條，而且比完還得自己挑出最長的那一條。',
         en:'The answer is <b>' + best.wl + '</b>. The cost is the point: the walk is driven by <b>the address, not the table</b>, so it finishes in 4 octets (32 bits in the real thing) - <b>five rules or 900,000 rules take the same time</b>. A linear scan has to compare every rule and then still work out which match was the longest.'}});
  return F.list;
}

/* ============================================ 4. the KV cache radix tree */
const CODE_K = [
'def insert(self, tokens):',
'    tokens = tokens[: len(tokens) // self.page * self.page]   # page aligned',
'    node, pos = self.root, 0',
'    while True:',
'        child = node.children.get(tokens[pos : pos + self.page])',
'        if child is None:',
'            break',
'        m = match_len(child.key, tokens[pos:])   # rounded down to a page',
'        if m < len(child.key):',
'            node = self.split(child, m)          # shared head becomes a node',
'            pos += m',
'            break',
'        node, pos = child, pos + m',
'    if pos < len(tokens):',
'        node.add_child(tokens[pos:], alloc_kv(tokens[pos:]))',
'    return pos            # tokens whose KV cache we did not have to compute'
];

function KV(page){
  const self = {page:page, alloc:0, splits:0, clock:0,
                root:{key:[], ch:{}, par:null, t:0, tag:null}};
  const ck = (arr, i) => arr.slice(i, i + page).join(',');
  const flo = m => Math.floor(m / page) * page;
  const common = (a, b) => { let i = 0; const n = Math.min(a.length, b.length);
    while (i < n && a[i] === b[i]) i++; return i; };
  self.rawMatch = (toks) => {
    let node = self.root, pos = 0, guard = 0;
    while (guard++ < 40){
      const c = node.ch[ck(toks, pos)];
      if (!c) return pos;
      const i = common(c.key, toks.slice(pos));
      pos += i;
      if (i < c.key.length) return pos;
      node = c;
    }
    return pos;
  };
  self.insert = (toks, tag) => {
    const aligned = toks.slice(0, Math.floor(toks.length / page) * page);
    self.clock++;
    let node = self.root, pos = 0, guard = 0;
    node.t = self.clock;
    while (guard++ < 40){
      const c = node.ch[ck(aligned, pos)];
      if (!c) break;
      const m = flo(common(c.key, aligned.slice(pos)));
      if (m === 0) break;
      if (m < c.key.length){
        const mid = {key:c.key.slice(0, m), ch:{}, par:node, t:self.clock, tag:'shared'};
        c.key = c.key.slice(m); c.par = mid;
        mid.ch[ck(c.key, 0)] = c;
        node.ch[ck(mid.key, 0)] = mid;
        self.splits++;
        node = mid; pos += m;
        break;
      }
      node = c; pos += m; node.t = self.clock;
    }
    let added = 0;
    if (pos < aligned.length){
      const tail = aligned.slice(pos);
      node.ch[ck(tail, 0)] = {key:tail, ch:{}, par:node, t:self.clock, tag:tag};
      added = tail.length;
      self.alloc += added;
    }
    return {reused:pos, alloc:added, aligned:aligned.length};
  };
  self.leaves = () => {
    const out = [];
    (function walk(n){
      const ks = Object.keys(n.ch);
      if (!ks.length && n !== self.root) out.push(n);
      ks.forEach(k => walk(n.ch[k]));
    })(self.root);
    return out;
  };
  self.remove = (n) => {
    for (const k in n.par.ch) if (n.par.ch[k] === n) delete n.par.ch[k];
    self.alloc -= n.key.length;
  };
  return self;
}
/* a KV tree drawn with the shared tree renderer */
function kvView(cache){
  const map = new Map();
  const conv = (n) => {
    const v = TN(false); map.set(n, v);
    Object.keys(n.ch).forEach(k => {
      const c = n.ch[k];
      v.ch[k] = {lab:c.key.length + ' tok', node:conv(c)};
    });
    return v;
  };
  const root = conv(cache.root);
  return {root:root, map:map};
}
function kvTreeShapes(cache, hot, y0, showT){
  const V = kvView(cache);
  const pos = layoutLR(V.root, 1.75, y0 == null ? 4.55 : y0, 2.05, .62);
  const st = new Map();
  (hot || []).forEach(([n, s]) => { const v = V.map.get(n); if (v) st.set(v, s); });
  const tagOf = new Map();
  (function walk(n){ const v = V.map.get(n);
    tagOf.set(v, n.tag ? (showT ? n.tag + '  ·  t' + n.t : n.tag) : null);
    Object.keys(n.ch).forEach(k => walk(n.ch[k])); })(cache.root);
  return treeShapes(V.root, pos, st, {r:.22, sub:n => tagOf.get(n) || null});
}

const SYS = [], TA = [], TB = [], TC = [];
for (let i = 1; i <= 10; i++) SYS.push(i);
for (let i = 21; i <= 26; i++) TA.push(i);
for (let i = 31; i <= 36; i++) TB.push(i);
for (let i = 41; i <= 44; i++) TC.push(i);
const REQ = [['req-A', SYS.concat(TA)], ['req-B', SYS.concat(TB)],
             ['req-C', SYS.concat(TA).concat(TC)]];

function kvFrames(page){
  const F = new Frames();
  const cache = KV(page);
  const W = .40, H = .50, X0 = .95, Y0 = 1.45, DY = .78;
  const rows = [];
  const bars = (cur) => {
    const out = [];
    rows.forEach((r, i) => {
      const y = Y0 + i * DY;
      out.push(S.t(X0 - .18, y + H * .70, r.name, {c:i === cur ? COL.orangeL : COL.grey, fs:.30, anchor:'end'}));
      r.toks.forEach((v, j) => {
        let s = 'idle';
        if (j < r.usable) s = 'ok';
        else if (j < r.raw) s = 'bad';
        else if (j < r.aligned) s = i === cur ? 'hot' : 'done';
        else s = 'ghost';
        out.push(S.r(X0 + j * W, y, W - .05, H, i === cur ? s : (s === 'hot' ? 'done' : s),
                     String(v), {fs:.24}));
      });
    });
    if (page > 1){
      const n = REQ[2][1].length;
      for (let p = 0; p * page <= n; p++)
        out.push(S.e(X0 + p * page * W - .025, Y0 - .22, X0 + p * page * W - .025,
                     Y0 + (rows.length - 1) * DY + H + .16,
                     {s:'ghost', dash:'.10 .12', arrow:false, w:.02, o:.5}));
    }
    return out;
  };
  const cap = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});

  F.push({shapes:[cap(.72, 'three requests, one shared 10-token system prompt', COL.tealL, .34),
                  cap(1.10, 'page size = ' + page + ' token' + (page > 1 ? 's' : ''), COL.grey, .29)]
            .concat(bars(-1), kvTreeShapes(cache, [], 4.6)),
    panels:[{t:'page ' + page, cls:'act'}, {t:'allocated 0', cls:''}],
    view:VIEW, line:ln(CODE_K, 'def insert'),
    msg:{zh:'一台 LLM 伺服器上，每個 request 的前綴幾乎都一樣：同一段 system prompt、同一份 few-shot 範例。<b>前綴相同，KV cache 就一定相同</b>（attention 只看左邊），所以那段可以直接借用而不是重算 - 這棵樹就是用來記住「誰已經算過什麼」。',
         en:'On an LLM server almost every request starts the same way: the same system prompt, the same few-shot examples. <b>Identical prefix means identical KV cache</b> - attention only looks left - so that part can be borrowed instead of recomputed, and this tree is what remembers who has already computed what.'}});

  REQ.forEach(([name, toks], i) => {
    const raw = cache.rawMatch(toks);
    const usable = Math.floor(raw / page) * page;
    const alignedLen = Math.floor(toks.length / page) * page;
    rows.push({name:name, toks:toks, raw:raw, usable:usable, aligned:alignedLen});
    const stranded = raw - usable;
    F.push({shapes:[cap(.72, name + ': ' + toks.length + ' tokens, ' + raw + ' of them already in the tree',
                        raw ? COL.tealL : COL.orangeL, .34),
                    cap(1.10, raw === 0 ? 'nothing matches yet - every token has to be computed'
                        : (stranded ? 'match rounded down to a whole page: ' + raw + ' -> ' + usable
                           : 'the match is already page aligned'),
                        stranded ? COL.red : COL.grey, .29)]
              .concat(bars(i), kvTreeShapes(cache, [], 4.6)),
      panels:[{t:name, cls:'act'}, {t:'raw match ' + raw, cls:raw ? 'ok' : ''},
              {t:'usable ' + usable, cls:'ok'},
              {t:stranded ? 'stranded ' + stranded : 'stranded 0', cls:stranded ? 'bad' : ''}],
      view:VIEW, line:ln(CODE_K, 'm = match_len'),
      msg:stranded
        ? {zh:'這裡有 <b>' + raw + '</b> 個 token 跟樹上完全一樣，但只有 <b>' + usable + '</b> 個能用。KV cache 是<b>一頁一頁</b>配置的，半頁不是一個可以共用的單位，所以匹配長度<b>無條件往下取整</b>到 page 的倍數 - 那 ' + stranded + ' 個一模一樣的 token 只好重算。頁越大，記帳越便宜，被卡住的 token 也越多。',
           en:'<b>' + raw + '</b> tokens here are identical to what the tree holds, but only <b>' + usable + '</b> can be reused. KV cache is allocated <b>a page at a time</b>, and half a page is not a shareable unit, so the match is <b>rounded down</b> to a multiple of the page - those ' + stranded + ' identical tokens get recomputed anyway. Bigger pages mean cheaper bookkeeping and more tokens stranded like this.'}
        : (raw === 0
          ? {zh:'第一個 request，樹是空的，' + toks.length + ' 個 token 全部都要算。它同時也把這條前綴<b>留給後面的人用</b> - 快取的價值是後面的 request 兌現的。',
             en:'The first request finds an empty tree, so all ' + toks.length + ' tokens must be computed. It also <b>leaves this prefix behind for everyone after it</b> - the value of the cache is cashed in by later requests.'}
          : {zh:'匹配長度 ' + raw + ' 本來就是 page 的倍數，沒有任何 token 被卡住，' + raw + ' 個 token 直接免費。',
             en:'The match of ' + raw + ' is already a multiple of the page, nothing is stranded, and those ' + raw + ' tokens are free.'})});

    const before = cache.splits;
    const r = cache.insert(toks, name + ' tail');
    const didSplit = cache.splits > before;
    F.push({shapes:[cap(.72, name + ': reused ' + r.reused + ', allocated ' + r.alloc,
                        COL.tealL, .34),
                    cap(1.10, didSplit ? 'the shared head was split into a node of its own'
                                       : 'the new tail hangs off as one edge', COL.grey, .29)]
              .concat(bars(i), kvTreeShapes(cache, [], 4.6)),
      panels:[{t:name, cls:'act'}, {t:'reused ' + r.reused, cls:'ok'},
              {t:'allocated ' + r.alloc, cls:'hot'},
              {t:'splits ' + cache.splits, cls:didSplit ? 'hot' : ''},
              {t:'tree ' + cache.alloc + ' tok', cls:''}],
      view:VIEW, line:didSplit ? ln(CODE_K, 'node = self.split') : ln(CODE_K, 'node.add_child'),
      msg:didSplit
        ? {zh:'兩個 request 共用的那一段被<b>剖成自己的節點</b>，兩條尾巴各自掛在它下面 - 跟前面插入 "car" 剖開 "cat" 是<b>同一段程式碼</b>，只是這裡的字元換成 token、節點上多掛了一塊 GPU 記憶體。分裂不搬動任何 KV，只是把邊的界線移了一下。',
           en:'The run the two requests share is <b>split into a node of its own</b> with the two tails hanging under it - literally <b>the same code</b> that split "cat" when we inserted "car", except the characters are tokens and each node also owns a slab of GPU memory. The split moves no KV data, only the boundary of an edge.'}
        : {zh:'剩下的 ' + r.alloc + ' 個 token 掛成一條新邊。<b>' + r.reused + ' 個 token 的 attention 完全沒有重算</b> - 對長 system prompt 來說，這就是 TTFT 的差別。',
           en:'The remaining ' + r.alloc + ' tokens hang off as one new edge. <b>The attention for ' + r.reused + ' tokens was never recomputed</b> - with a long system prompt that is the whole difference in time-to-first-token.'}});
  });

  const asked = REQ.reduce((s, r) => s + r[1].length, 0);
  F.push({shapes:[cap(.72, asked + ' tokens asked for, ' + cache.alloc + ' KV slots allocated ('
                      + Math.round(100 * (1 - cache.alloc / asked)) + '% saved)', COL.orangeL, .34),
                  cap(1.10, 'every node is a prefix somebody can continue from', COL.grey, .29)]
            .concat(bars(-1), kvTreeShapes(cache, [], 4.6)),
    panels:[{t:'asked ' + asked, cls:''}, {t:'allocated ' + cache.alloc, cls:'ok'},
            {t:Math.round(100 * (1 - cache.alloc / asked)) + '% saved', cls:'ok'},
            {t:'splits ' + cache.splits, cls:''}],
    view:VIEW, line:ln(CODE_K, 'return pos'),
    msg:{zh:'樹裡每一段不同的 token 序列<b>只存一次</b>，而且每個節點都是一個「別人可以從這裡接下去」的前綴。把 page 調成 1 可以看到另一種取捨：一個 token 都不會被卡住，但節點變多、記帳變貴。這就是 SGLang 的 RadixAttention 在做的事。',
         en:'Each distinct run of tokens is stored <b>exactly once</b>, and every node is a prefix somebody else can continue from. Switch the page size to 1 to see the other side of the trade: nothing is ever stranded, but there are more nodes and the bookkeeping costs more. This is what SGLang\'s RadixAttention does.'}});
  return F.list;
}

/* ================================================== 5. leaf-only eviction */
const CODE_E = [
'def evict(self, n_tokens):',
'    freed = 0',
'    while freed < n_tokens:',
'        leaves = [n for n in self.nodes if not n.children]',
'        if not leaves:',
'            break',
'        victim = min(leaves, key=lambda n: n.last_access)   # LRU',
'        del victim.parent.children[victim.child_key]',
'        free_kv(victim.value)',
'        freed += len(victim.key)',
'        # the parent may have just become a leaf itself',
'    return freed'
];

function evictFrames(){
  const F = new Frames();
  const cache = KV(4);
  REQ.forEach(([n, t]) => cache.insert(t, n + ' tail'));
  const target = 12;
  const cap = (y, s, c, fs) => S.t(mid2, y, s, {c:c || COL.tealL, fs:fs || .32});
  const view = (hot) => [cap(.95, 'the tree after three requests   -   ' + cache.alloc + ' tokens live', COL.tealL, .34)]
      .concat(kvTreeShapes(cache, hot, 2.35, true));

  const marks = () => cache.leaves().map(n => [n, 'ok']);
  F.push({shapes:view(marks()).concat([
      cap(5.10, 'ringed = a leaf, and only leaves are eviction candidates', COL.grey, .29),
      cap(5.55, 't = the last time that node was touched by a request', COL.grey, .29)]),
    panels:[{t:'live ' + cache.alloc + ' tok', cls:''},
            {t:'leaves ' + cache.leaves().length, cls:'ok'},
            {t:'need ' + target + ' tok', cls:'hot'}],
    view:VIEW, line:ln(CODE_E, 'leaves = ['),
    msg:{zh:'GPU 記憶體滿了，要騰出 ' + target + ' 個 token 的空間。候選人<b>只有葉子</b>：釋放一個中間節點會讓它底下整棵子樹變成孤兒，而那些子樹正是別人正在用的更長前綴。所以驅逐永遠是<b>從樹梢往內剝</b>。',
         en:'GPU memory is full and we need room for ' + target + ' tokens. The candidates are <b>leaves only</b>: freeing an interior node would orphan the whole subtree below it, and those subtrees are exactly the longer prefixes somebody else is using. So eviction always <b>peels inwards from the tips</b>.'}});

  let freed = 0, guard = 0;
  while (freed < target && guard++ < 6){
    const ls = cache.leaves();
    if (!ls.length) break;
    let victim = ls[0];
    ls.forEach(n => { if (n.t < victim.t) victim = n; });
    F.push({shapes:view(ls.map(n => [n, n === victim ? 'bad' : 'ok'])).concat([
        cap(5.10, 'oldest leaf: ' + victim.tag + ' (t' + victim.t + ')  ->  frees ' + victim.key.length + ' tokens', COL.red, .30)]),
      panels:[{t:'candidates ' + ls.length, cls:'ok'},
              {t:'victim ' + victim.tag, cls:'bad'},
              {t:'t' + victim.t, cls:''}, {t:'freed ' + freed + '/' + target, cls:'hot'}],
      view:VIEW, line:ln(CODE_E, 'victim = min'),
      msg:{zh:'在葉子裡挑<b>最久沒被碰過</b>的那一個。注意靠近根的共用前綴不會出現在這個名單上 - 它每來一個新 request 就被走過一次，時間戳一直更新，而且它有小孩根本不是葉子。<b>熱前綴自動活得比它的小孩久</b>，這不是特別寫的規則，是結構本身的結果。',
           en:'Among the leaves pick the one <b>least recently touched</b>. Notice the shared prefix near the root never appears on this list - every new request walks over it and refreshes its timestamp, and it has children so it is not a leaf anyway. <b>The hot prefix automatically outlives its own children</b>: nobody wrote that rule, it falls out of the structure.'}});
    const parent = victim.par;
    cache.remove(victim);
    freed += victim.key.length;
    const nowLeaf = Object.keys(parent.ch).length === 0 && parent !== cache.root;
    F.push({shapes:view((nowLeaf ? [[parent, 'hot']] : []).concat(cache.leaves().map(n => [n, 'ok'])))
        .concat([cap(5.10, nowLeaf ? 'its parent just became a leaf - now it is a candidate too'
                                   : 'freed ' + freed + ' of ' + target + ' tokens', nowLeaf ? COL.orangeL : COL.grey, .30)]),
      panels:[{t:'freed ' + freed + '/' + target, cls:freed >= target ? 'ok' : 'hot'},
              {t:'live ' + cache.alloc + ' tok', cls:''},
              {t:'leaves ' + cache.leaves().length, cls:'ok'}],
      view:VIEW, line:ln(CODE_E, 'freed += len'),
      msg:nowLeaf
        ? {zh:'節點消失後，它的父親<b>剛剛變成葉子</b>，下一輪才輪得到它。所以一條冷掉的分支是<b>一次剝一節</b>地往內縮，而不是整條一起丟 - 這也讓「別人只共用到一半」的情況仍然安全。',
           en:'With the node gone its parent <b>has just become a leaf</b> and only now becomes a candidate. A cold branch therefore retreats <b>one node at a time</b> rather than being dropped whole - which is also what keeps it safe when somebody else shares only part of it.'}
        : {zh:'釋放 ' + victim.key.length + ' 個 token 的 KV，樹上其他所有前綴都沒有受影響。',
           en:'That frees the KV for ' + victim.key.length + ' tokens and leaves every other prefix in the tree untouched.'}});
  }
  F.push({shapes:view(cache.leaves().map(n => [n, 'ok'])).concat([
      cap(5.10, 'freed ' + freed + ' tokens; the shared prefix is still there', COL.tealL, .30),
      cap(5.55, 'LRU here is a heap over leaves - the same policy, restricted to a legal set', COL.grey, .29)]),
    panels:[{t:'freed ' + freed, cls:'ok'}, {t:'live ' + cache.alloc + ' tok', cls:''},
            {t:'shared prefix kept', cls:'ok'}],
    view:VIEW, line:ln(CODE_E, 'return freed'),
    msg:{zh:'騰出了 ' + freed + ' 個 token，而每個 request 都會用到的那段共用前綴<b>一格都沒掉</b>。這就是 radix cache 的驅逐策略：<b>還是 LRU，只是候選集合被結構限制成「葉子」</b>。實作上是對葉子集合維護一個 heap，節點失去最後一個小孩時才被放進去。',
         en:'We freed ' + freed + ' tokens and the shared prefix every request needs <b>lost nothing at all</b>. That is the whole eviction policy of a radix cache: <b>still LRU, with the candidate set restricted by the structure to the leaves</b>. In practice it is a heap over the leaf set, and a node is only pushed into it once it loses its last child.'}});
  return F.list;
}

/* ================================================ 6. LC 211 - wildcards */
const CODE_W = [
'class WordDictionary:',
'    def addWord(self, word):',
'        insert(self.root, word)',
'',
'    def search(self, word):',
'        def dfs(node, i):',
'            if i == len(word):',
'                return node.is_word',
'            ch = word[i]',
'            if ch != ".":',
'                nxt = node.children.get(ch)      # one hash lookup',
'                return nxt is not None and dfs(nxt, i + 1)',
'            for nxt in node.children.values():   # "." has no key to look up',
'                if dfs(nxt, i + 1):',
'                    return True',
'            return False',
'        return dfs(self.root, 0)'
];

function wildFrames(variant){
  const F = new Frames();
  const full = trieOf(WORDS1.concat(['card']));
  const pos = layoutLR(full, 1.55, 1.60, 1.45, 1.30);
  const ends = {};
  ['cat', 'car', 'card', 'dog'].forEach(w => { ends[w] = trieWalk(full, w).slice(-1)[0]; });
  const subOf = (n) => {
    for (const w in ends) if (ends[w] === n) return w;
    return null;
  };
  const word = variant ? 'c.g' : '.og';
  const st = new Map();
  let visited = 0;
  const draw = (cap, capc, extra) => {
    const s = treeShapes(full, pos, st, {inNode:true, r:.30, sub:subOf});
    s.push(S.t(mid2, .78, cap, {c:capc || COL.tealL, fs:.34}));
    if (extra) s.push(S.t(mid2, 6.02, extra, {c:COL.grey, fs:.29}));
    return s;
  };
  const chips = (i, extra) => {
    const p = [{t:'search "' + word + '"', cls:'act'},
               {t:'at ' + (i < word.length ? 'char ' + i + ' = "' + word[i] + '"' : 'end of word'), cls:''},
               {t:'nodes visited ' + visited, cls:''}];
    return extra ? p.concat(extra) : p;
  };

  function dfs(node, i){
    if (i === word.length){
      const hit = node.w;
      st.set(node, hit ? 'ok' : 'bad');
      F.push({shapes:draw('end of "' + word + '"  ->  is_word = ' + (hit ? 'True' : 'False'),
                          hit ? COL.tealL : COL.red),
        panels:chips(i, [{t:hit ? 'match' : 'not a word', cls:hit ? 'ok' : 'bad'}]),
        view:VIEW, line:ln(CODE_W, 'return node.is_word'),
        msg:hit
          ? {zh:'字用完了，而且停在一個 is_word 的節點上 - 這條路徑就是一個真正的字，可以直接回 True，不用把其他分支走完。',
             en:'The pattern ran out on a node whose is_word is True - this path is a real word, so we can return True immediately without exploring the remaining branches.'}
          : {zh:'字用完了，但這個節點只是一個前綴，不是一個字。這是<b>失敗但不是錯誤</b>的情況，回 False 讓上一層去試下一個小孩。',
             en:'The pattern ran out but this node is only a prefix, not a word. This is a <b>failure that is not an error</b>: return False and let the level above try its next child.'}});
      return hit;
    }
    const ch = word[i];
    if (ch !== '.'){
      const e = node.ch[ch];
      if (!e){
        st.set(node, 'bad');
        F.push({shapes:draw('no "' + ch + '" child here  ->  back up', COL.red),
          panels:chips(i, [{t:'dead end', cls:'bad'}]),
          view:VIEW, line:ln(CODE_W, 'nxt = node.children.get(ch)'),
          msg:{zh:'一般字元只要<b>一次 dict 查詢</b>：沒有這個小孩就整條路死掉，立刻回上一層。這一步的成本跟樹裡有多少字無關。',
               en:'An ordinary character costs <b>one dict lookup</b>: no such child and the whole path dies, so we go straight back up. The cost of this step has nothing to do with how many words the tree holds.'}});
        return false;
      }
      visited++;
      st.set(e.node, 'hot');
      F.push({shapes:draw('"' + ch + '" is a real character  ->  one lookup, one child', COL.tealL),
        panels:chips(i, [{t:'child "' + ch + '"', cls:'ok'}]),
        view:VIEW, line:ln(CODE_W, 'return nxt is not None'),
        msg:{zh:'字元是確定的，所以只有<b>一個</b>小孩需要考慮 - 這裡跟一般的 trie 走訪一模一樣。',
             en:'The character is known, so there is exactly <b>one</b> child to consider - this step is identical to an ordinary trie walk.'}});
      const r = dfs(e.node, i + 1);
      if (!r) st.set(e.node, 'bad');
      return r;
    }
    const ks = Object.keys(node.ch);
    F.push({shapes:draw('"." at position ' + i + '  ->  no key to look up, try all ' + ks.length + ' children', COL.orangeL),
      panels:chips(i, [{t:'wildcard', cls:'hot'}, {t:ks.length + ' branches', cls:'hot'}]),
      view:VIEW, line:ln(CODE_W, 'for nxt in node.children.values()'),
      msg:{zh:'碰到 "."：<b>沒有 key 可以查</b>，所以這一步從「一次 dict 查詢」變成「把所有小孩都試一遍」。這就是為什麼這題不能用壓縮過的 hash map 做 - trie 之所以有用，正是因為它<b>還留著 children 這個可以逐一列舉的結構</b>。',
           en:'A ".": <b>there is no key to hash</b>, so this step turns from one dict lookup into trying every child. This is why the problem cannot be solved with a compressed hash map - the trie is useful here precisely because it <b>still keeps children as something you can enumerate</b>.'}});
    for (const k of ks){
      const c = node.ch[k].node;
      visited++;
      st.set(c, 'hot');
      F.push({shapes:draw('wildcard branch: trying "' + k + '"', COL.orangeL),
        panels:chips(i, [{t:'try "' + k + '"', cls:'hot'}]),
        view:VIEW, line:ln(CODE_W, 'if dfs(nxt, i + 1)'),
        msg:{zh:'試 "' + k + '" 這條分支。DFS 有深度上限 - <b>就是 pattern 的長度</b>，所以最壞情況是 O(26^(通配符個數))，但實際上每一層只會展開那個節點<b>真的有</b>的小孩，通常遠少於 26 個。',
             en:'Try the "' + k + '" branch. The DFS has a hard depth limit - <b>the length of the pattern</b> - so the worst case is O(26^(number of dots)), but each level only expands the children that node <b>actually has</b>, which is usually far fewer than 26.'}});
      if (dfs(c, i + 1)) return true;
      st.set(c, 'bad');
    }
    st.set(node, 'bad');
    F.push({shapes:draw('every branch failed  ->  return False', COL.red),
      panels:chips(i, [{t:'all ' + ks.length + ' failed', cls:'bad'}]),
      view:VIEW, line:ln(CODE_W, 'return False'),
      msg:{zh:'所有小孩都試過了還是不行，這一層宣告失敗。回溯本身沒有成本 - 樹沒有被改動過，只是遞迴退回去而已。',
           en:'Every child has been tried and none worked, so this level fails. The backtracking itself costs nothing - the tree was never modified, the recursion simply unwinds.'}});
    return false;
  }

  st.set(full, 'act');
  F.push({shapes:draw('search("' + word + '")  -  "." matches any single letter', COL.tealL,
                      'stored: cat, car, card, dog'),
    panels:chips(0), view:VIEW, line:ln(CODE_W, 'return dfs(self.root, 0)'),
    msg:{zh:'LC 211 只比 LC 208 多一個 "."，但它<b>問出了 trie 和 hash map 的差別</b>：hash map 需要完整的 key 才能查，而 trie 是一個字元一個字元往下走的，所以「這一個字元不確定」只是變成「這一步要分岔」，結構完全不用改。',
         en:'LC 211 adds a single "." to LC 208, and that dot <b>asks the question that separates a trie from a hash map</b>: a hash map needs a complete key before it can look anything up, while a trie descends one character at a time - so "this character is unknown" merely becomes "this step branches", and the structure does not change at all.'}});
  const ans = dfs(full, 0);
  F.push({shapes:draw('search("' + word + '") = ' + (ans ? 'True' : 'False'), ans ? COL.tealL : COL.red,
                      'nodes visited: ' + visited + '   -   bounded by the pattern, not by the dictionary'),
    panels:[{t:'search "' + word + '"', cls:'act'}, {t:ans ? 'True' : 'False', cls:ans ? 'ok' : 'bad'},
            {t:'visited ' + visited, cls:''}],
    view:VIEW, line:ln(CODE_W, 'def search'),
    msg:{zh:'答案是 <b>' + (ans ? 'True' : 'False') + '</b>，一共只碰了 ' + visited + ' 個節點。整段搜尋的規模<b>由 pattern 決定，不由字典大小決定</b> - 字典裡再多十萬個字，走訪的節點數也不會變，這正是把「前綴」變成「結構」換來的東西。',
         en:'The answer is <b>' + (ans ? 'True' : 'False') + '</b> after touching only ' + visited + ' nodes. The size of the search is <b>set by the pattern, not by the dictionary</b> - add another hundred thousand words and the number of nodes visited does not change. That is what you buy by turning "prefix" into "structure".'}});
  return F.list;
}

/* ===================================================================== meta */
const DAY_META = {
  title:{zh:'Day 24 · Trie 與 Radix Tree', en:'Day 24 · Tries and radix trees'},
  sub:{zh:'Trie 把「每個前綴都是一個節點」變成資料結構本身；radix tree 再把沒有分岔的那一串壓成一條邊。自動完成、IP 路由表、LLM 伺服器的 KV cache，用的都是同一棵樹。',
       en:'A trie makes "every prefix is a node" the data structure itself; a radix tree then collapses every chain that never branches into a single edge. Autocomplete, IP routing tables and an LLM server\'s KV cache are all the same tree.'},
  tabs:[
    {id:'trie', label:{zh:'Trie：每個前綴都是節點', en:'trie: every prefix is a node'},
     stage:{zh:'字母在邊上，節點是前綴 - 所以插入的成本是字長，不是字數',
            en:'letters on the edges, prefixes as nodes - so insertion costs the length of the word, not the size of the tree'},
     view:VIEW,
     variants:[{zh:'插入 "card"', en:'insert "card"'},
               {zh:'search 與 startsWith', en:'search vs startsWith'}],
     idea:{zh:'Trie 的想法只有一句：<b>把字串拆成路徑</b>。每一個節點代表「從根走到這裡的那個前綴」，所以「哪些字以 ca 開頭」不是一次搜尋，而是<b>一棵現成的子樹</b>。代價是 hash map 的 O(1) 變成 O(len(word))，換來的是 hash map 永遠給不了的東西：前綴查詢、字典序走訪、以及「不確定的字元」也能走。<b>is_word</b> 這個布林值是整個結構裡最容易忘記的一格 - 沒有它，「car 是一個字」和「ca 只是路過」在樹上長得完全一樣。',
           en:'A trie is one idea: <b>turn strings into paths</b>. Each node stands for the prefix that reaches it, so "which words start with ca" is not a search at all - it is <b>a subtree that already exists</b>. The price is that a hash map\'s O(1) becomes O(len(word)); what you buy is everything a hash map cannot do: prefix queries, sorted traversal, and walking with characters you do not know yet. <b>is_word</b> is the easiest cell in the structure to forget - without it, "car is a word" and "ca is merely passed through" look identical on the tree.'},
     legend:['hot', 'act', 'done', 'ok'],
     code:CODE_T,
     build:(v) => norm(trieFrames(v ? 1 : 0))},

    {id:'radix', label:{zh:'Radix tree：把沒分岔的邊壓掉', en:'radix tree: collapse the chains'},
     stage:{zh:'節點是分岔點，字串放在邊上 - 唯一麻煩的操作是「剖開」',
            en:'nodes are branch points and the strings live on the edges - the only hard operation is the split'},
     view:VIEW,
     variants:[{zh:'cat / car / card / dog', en:'cat / car / card / dog'},
               {zh:'romulus / romane / romanus', en:'romulus / romane / romanus'}],
     idea:{zh:'Trie 有一個明顯的浪費：一長串<b>沒有分岔</b>的字元也各佔一個節點，走過去就是一串指標跳躍。Radix（PATRICIA）tree 把那一串壓成<b>一條有標籤的邊</b>，於是節點數從「總字元數」掉到「分岔點數」。查詢的複雜度沒變，變的是碰到的節點數與 cache miss。代價全部集中在一個操作：新的字串走到一半跟邊上的標籤<b>分岔</b>時，必須把那條邊<b>剖開</b>，共用的頭變成一個新節點，兩條尾巴掛在它下面。這段程式碼看起來很像指標習題，但它就是 LLM 伺服器每次「兩個 request 共用前綴」時跑的那一段。',
           en:'A trie wastes something obvious: a long run of characters that <b>never branches</b> still costs one node per character, and walking it is a chain of pointer hops. A radix (PATRICIA) tree collapses that run into <b>one labelled edge</b>, so the node count drops from "number of characters" to "number of branch points". The complexity of a lookup does not change; the number of nodes touched and cache misses does. The entire cost lands on one operation: when a new string <b>diverges</b> halfway along an edge label, that edge has to be <b>split</b> - the shared head becomes a new node and the two tails hang under it. It reads like a pointer exercise, and it is exactly the code an LLM server runs whenever two requests share a prompt.'},
     legend:['hot', 'act', 'bad', 'ok'],
     code:CODE_R,
     build:(v) => norm(radixFrames(v ? ['romulus', 'romane', 'romanus'] : ['cat', 'car', 'card', 'dog']))},

    {id:'lpm', label:{zh:'最長前綴匹配（路由表）', en:'longest prefix match (routing)'},
     stage:{zh:'「最具體的規則獲勝」＝「最深的有標記節點」',
            en:'"the most specific rule wins" is literally "the deepest marked node"'},
     view:VIEW,
     variants:[{zh:'10.20.30.5（命中 /24）', en:'10.20.30.5 (hits the /24)'},
               {zh:'10.20.99.1（掉回 /16）', en:'10.20.99.1 (falls back to the /16)'}],
     idea:{zh:'路由表是一堆<b>會互相重疊</b>的規則：0.0.0.0/0、10.0.0.0/8、10.20.0.0/16、10.20.30.0/24 全都匹配 10.20.30.5，而正確答案是<b>最具體</b>的那一條。放進 trie 之後，「最具體」就等於「最深」，於是查詢變成：照著位址往下走，一路記住經過的最後一個有標記的節點。這件事有兩個結果值得記住 - 第一，<b>規則的先後順序完全不影響答案</b>，因為順序根本沒有進到資料結構裡；第二，成本由<b>位址的長度</b>（32 個 bit）決定，跟表裡有多少條規則無關，所以 90 萬條路由跟 5 條一樣快。真實的硬體用的是多 bit 一跳的壓縮 trie，但形狀就是這一棵。',
           en:'A routing table is a pile of <b>overlapping</b> rules: 0.0.0.0/0, 10.0.0.0/8, 10.20.0.0/16 and 10.20.30.0/24 all match 10.20.30.5, and the correct answer is the <b>most specific</b> one. Put them in a trie and "most specific" becomes "deepest", so a lookup is: walk down following the address, remembering the last marked node you passed. Two consequences are worth keeping - first, <b>the order the rules were added cannot affect the answer</b>, because order never enters the structure; second, the cost is set by <b>the length of the address</b> (32 bits) and not by the size of the table, so 900,000 routes cost what 5 routes cost. Real hardware uses a multi-bit compressed trie, but the shape is this one.'},
     legend:['hot', 'act', 'bad', 'ok'],
     code:CODE_P,
     build:(v) => norm(lpmFrames(v ? 1 : 0))},

    {id:'kv', label:{zh:'KV cache 的 radix tree', en:'the KV cache radix tree'},
     stage:{zh:'共用的前綴只算一次 - 而且是一頁一頁算的',
            en:'a shared prefix is computed once - and it is counted a page at a time'},
     view:VIEW,
     variants:[{zh:'page = 4（真實設定）', en:'page = 4 (how it really runs)'},
               {zh:'page = 1（token 級）', en:'page = 1 (token level)'}],
     idea:{zh:'Attention 只看左邊，所以<b>兩個 request 只要前綴一樣，那段的 KV cache 就一定一樣</b>。既然一樣，就沒有理由算第二次 - 需要的只是一個能回答「這串 token 我算到哪裡了」的結構，而那正是一棵 key 是 token 序列的 radix tree（SGLang 叫它 RadixAttention）。這裡有三個實作上的細節值得看：children 用<b>第一頁 token</b> 當 key，所以選邊是 O(1)；匹配長度會<b>無條件往下取整</b>到 page 的倍數，因為半頁不是可以共用的單位，於是會出現「明明一樣卻不能用」的 token；還有 request 分岔時的<b>剖開</b>，跟前面剖 "cat" 的是同一段程式碼。',
           en:'Attention only looks left, so <b>if two requests share a prefix, the KV cache for that prefix is identical</b>. If it is identical there is no reason to compute it twice - all you need is a structure that answers "how far into this token sequence have I already got", and that is a radix tree keyed by token sequences (SGLang calls it RadixAttention). Three implementation details are worth watching: children are keyed by the <b>first page of tokens</b>, so picking an edge stays O(1); the match length is <b>rounded down</b> to a multiple of the page, because half a page is not a shareable unit, which produces tokens that are identical yet unusable; and the <b>split</b> when two requests diverge is the same code that split "cat".'},
     legend:['hot', 'ok', 'bad', 'ghost'],
     code:CODE_K,
     build:(v) => norm(kvFrames(v ? 1 : 4))},

    {id:'evict', label:{zh:'驅逐只能從葉子開始', en:'eviction happens at the leaves'},
     stage:{zh:'LRU 沒有變，變的是候選集合被結構限制住了',
            en:'the policy is still LRU - the structure is what restricts the candidate set'},
     view:VIEW,
     idea:{zh:'快取滿了要丟東西，但這棵樹不能隨便丟：<b>釋放一個中間節點會讓它底下的子樹變成孤兒</b>，而那些子樹是別人正在用的、更長的前綴。所以候選人只有<b>葉子</b>，驅逐是從樹梢一節一節往內剝，一個節點要等到失去最後一個小孩才變成候選人。這個限制帶來一個很漂亮的副作用：<b>越靠近根、越多人共用的前綴，越不會被丟掉</b> - 一方面它每來一個 request 就被走過一次、時間戳一直更新，另一方面它有小孩就根本不是葉子。沒有人特別寫「保護熱前綴」這條規則，它是結構自己長出來的。',
           en:'When the cache fills up something has to go, but this tree cannot drop just anything: <b>freeing an interior node would orphan the subtree below it</b>, and those subtrees are longer prefixes somebody else is using. So only <b>leaves</b> are candidates, eviction peels inwards one node at a time, and a node becomes a candidate only after it loses its last child. That restriction has a lovely side effect: <b>the closer to the root and the more widely shared a prefix is, the harder it is to evict</b> - every request walks over it and refreshes its timestamp, and while it has children it is not a leaf at all. Nobody wrote a rule to protect the hot prefix; the structure grew one.'},
     legend:['hot', 'ok', 'bad', 'ghost'],
     code:CODE_E,
     build:() => norm(evictFrames())},

    {id:'lc211', label:{zh:'LC 211 · 帶萬用字元的字典', en:'LC 211 · Add and Search Word'},
     stage:{zh:'一個 "." 就問出了 trie 和 hash map 的差別',
            en:'a single "." asks the question that separates a trie from a hash map'},
     view:VIEW,
     variants:[{zh:'search(".og") = True', en:'search(".og") = True'},
               {zh:'search("c.g") = False', en:'search("c.g") = False'}],
     idea:{zh:'LC 208 用 hash set 也能勉強做（把每個前綴都塞進去），但 LC 211 不行：<b>"." 代表這個位置沒有 key 可以查</b>，而 hash map 一定要拿到完整的 key 才能動。Trie 是一個字元一個字元往下走的，所以「這個字元不確定」只是把一次 dict 查詢換成「把這個節點<b>現有的</b>小孩都試一遍」，結構完全不用改。實作上就是把走訪改成 DFS，深度上限是 pattern 的長度 - 最壞情況 O(26^通配符數)，但每一層只會展開真的存在的小孩，而且整趟走訪的規模<b>由 pattern 決定、不由字典大小決定</b>。',
           en:'LC 208 can be faked with a hash set (insert every prefix of every word), but LC 211 cannot: <b>a "." means there is no key to hash at that position</b>, and a hash map needs the complete key before it can do anything. A trie descends one character at a time, so "this character is unknown" merely swaps one dict lookup for "try the children this node <b>actually has</b>", and the structure is untouched. In code the walk becomes a DFS whose depth is bounded by the length of the pattern - worst case O(26^dots), but each level only expands children that exist, and the size of the whole search is <b>set by the pattern, not by the dictionary</b>.'},
     legend:['hot', 'ok', 'bad', 'act'],
     code:CODE_W,
     build:(v) => norm(wildFrames(v ? 1 : 0))}
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
