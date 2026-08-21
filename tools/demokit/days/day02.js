// DAY: 02
// TITLE_ZH: 堆疊與佇列
// TITLE_EN: Stack and Queue
// SUB_ZH: 後進先出 vs 先進先出：兩種「只准從特定一端動手」的容器，以及它們最經典的用途。
// SUB_EN: LIFO vs FIFO.
// FOLDER: day%2002%20-%20stack%20and%20queue
// MEDIUM: https://medium.com/100-days-of-python/day-02-stack-queue-1ab49a7b021d

const BW = 1.55, BH = .78, BX = 4.2, BOT = 5.35;
function stackShapes(st, o){
  o = o || {}; const out = [], n = st.length;
  out.push(S.e(BX - .18, BOT + BH + .05, BX - .18, BOT + BH - 4.2, {arrow:false, s:'soft', w:.05}),
           S.e(BX + BW + .18, BOT + BH + .05, BX + BW + .18, BOT + BH - 4.2, {arrow:false, s:'soft', w:.05}),
           S.e(BX - .22, BOT + BH + .05, BX + BW + .22, BOT + BH + .05, {arrow:false, s:'soft', w:.05}));
  st.forEach((v, i) => {
    const y = BOT - i * (BH + .12);
    out.push(S.r(BX, y, BW, BH, (o.states && o.states[i]) || 'idle', String(v), {fs:.44}));
  });
  const ty = BOT - (n - 1) * (BH + .12);
  if (n) out.push(S.t(BX + BW + .95, ty + BH * .62, 'top', {c:'#c7a6ff', fs:.34}),
                  S.e(BX + BW + .60, ty + BH / 2, BX + BW + .26, ty + BH / 2, {s:'act', w:.05}));
  else out.push(S.t(BX + BW / 2, BOT + BH * .55, {zh:'（空的）', en:'(empty)'}, {c:'#8fa3ac', fs:.34}));
  if (o.fly != null)
    out.push(S.r(BX, .75, BW, BH, o.flyStyle || 'hot', String(o.fly), {fs:.44}),
             S.e(BX + BW / 2, .75 + BH + .12, BX + BW / 2, BOT - n * (BH + .12) - .08,
                 {s:o.flyStyle || 'hot', w:.05}));
  out.push(S.t(BX + BW / 2, .45, o.cap || {zh:'只能從上面進出', en:'one open end'}, {c:'#8fa3ac', fs:.30}));
  return out;
}
const QW = 1.25, QH = .95, QY = 2.6, QX = 1.15;
function queueShapes(q, o){
  o = o || {}; const out = [];
  for (let i = 0; i < 6; i++)
    out.push(S.r(QX + i * (QW + .14), QY, QW, QH, 'ghost', ''));
  q.forEach((v, i) => out.push(S.r(QX + i * (QW + .14), QY, QW, QH,
    (o.states && o.states[i]) || 'idle', String(v), {fs:.44})));
  out.push(S.t(QX + QW / 2, QY - .55, {zh:'front 出口', en:'front (out)'}, {c:'#ffbe6b', fs:.30}),
           S.t(QX + (q.length ? q.length - 1 : 0) * (QW + .14) + QW / 2, QY + QH + .55,
               {zh:'rear 入口', en:'rear (in)'}, {c:'#c7a6ff', fs:.30}));
  if (o.fly != null)
    out.push(S.r(QX + 6.2 * (QW + .14), QY, QW, QH, o.flyStyle || 'hot', String(o.fly), {fs:.44}),
             S.e(QX + 6.2 * (QW + .14) - .10, QY + QH / 2,
                 QX + q.length * (QW + .14) + QW + .10, QY + QH / 2, {s:o.flyStyle || 'hot', w:.05}));
  if (o.out != null)
    out.push(S.r(QX - 1.85, QY, QW, QH, 'ok', String(o.out), {fs:.44}),
             S.e(QX - .12, QY + QH / 2, QX - .55, QY + QH / 2, {s:'ok', w:.05}));
  return out;
}
const chips = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));

const CODE_STACK = [
'class Stack:',
'    def __init__(self):',
'        self.items = []',
'',
'    def push(self, item):       # 推進去 O(1)',
'        self.items.append(item)',
'',
'    def pop(self):              # 拿最上面那個 O(1)',
'        return self.items.pop()',
'',
'    def peak(self):             # 只看不拿',
'        return self.items[-1]'];
const CODE_QUEUE = [
'from collections import deque',
'',
'class Queue:',
'    def __init__(self):',
'        self.items = deque()',
'',
'    def push(self, item):       # 從尾巴進 O(1)',
'        self.items.append(item)',
'',
'    def pop(self):              # 從頭出 O(1)',
'        return self.items.popleft()'];
const CODE_BR = [
'def is_valid(s):                # LeetCode 20',
'    pairs = {")": "(", "]": "[", "}": "{"}',
'    stack = []',
'    for ch in s:',
'        if ch in "([{":',
'            stack.append(ch)',
'        else:',
'            if not stack or stack.pop() != pairs[ch]:',
'                return False',
'    return not stack'];

function buildStack(){
  const F = new Frames(), st = [];
  const P = () => [{lbl:{zh:'堆疊內容（左＝底部）', en:'stack (left = bottom)'}, chips:chips(st, 'ok')}];
  F.push({shapes:stackShapes(st), panels:P(), line:2,
    msg:{zh:'堆疊就是一個「只開一個口」的盒子。底層其實只是 Python 的 list。',
         en:'A stack is a box with a single opening. Underneath it is just a Python list.'}});
  ['a', 'b', 'c'].forEach(v => {
    F.push({shapes:stackShapes(st, {fly:v}), panels:P(), line:4,
      msg:{zh:'<b>push(' + v + ')</b>：新元素只能從上面放進去。',
           en:'<b>push(' + v + ')</b>: new items can only go in from the top.'}});
    st.push(v);
    F.push({shapes:stackShapes(st, {states:{[st.length - 1]:'ok'}}), panels:P(), line:5,
      msg:{zh:'<b>append</b> 是在 list 尾端加一個，不用搬動任何舊資料——<b>O(1)</b>。' +
              '<b>top</b> 永遠跟著最後放進去的那個。',
           en:'<b>append</b> adds at the end of the list, moving nothing - <b>O(1)</b>. <b>top</b> always tracks the most recent item.'}});
  });
  for (let k = 0; k < 2; k++){
    const v = st[st.length - 1];
    F.push({shapes:stackShapes(st, {states:{[st.length - 1]:'hot'}}), panels:P(), line:8,
      msg:{zh:'<b>pop()</b> 拿走的一定是最上面的 <b>' + v + '</b>——最後進來的最先出去，這就是 <b>LIFO</b>。',
           en:'<b>pop()</b> always takes the top item <b>' + v + '</b> - last in, first out, i.e. <b>LIFO</b>.'}});
    st.pop();
    F.push({shapes:stackShapes(st, {fly:v, flyStyle:'ok', cap:{zh:'彈出 ' + v, en:'popped ' + v}}),
      panels:P(), line:8,
      msg:{zh:'底下的 <b>' + (st.length ? st[st.length - 1] : '（沒了）') + '</b> 這時才重見天日。' +
              '被壓住的元素永遠要等上面的先走。',
           en:'Only now does <b>' + (st.length ? st[st.length - 1] : '(nothing)') +
              '</b> become reachable. Buried items always wait for the ones above them.'}});
  }
  F.push({shapes:stackShapes(st, {states:{0:'ok'}}), panels:P(), line:10,
    msg:{zh:'<b>peak()</b> 只看不拿。函式呼叫、瀏覽器上一頁、編輯器的 undo，' +
            '凡是「最近做的先撤銷」都是堆疊。',
         en:'<b>peak()</b> looks without removing. Function calls, the browser back button, editor undo - anything shaped like "undo the most recent thing first" is a stack.'}});
  return F.list;
}

function buildQueue(){
  const F = new Frames(), q = [];
  const P = () => [{lbl:{zh:'佇列內容（左＝front）', en:'queue (left = front)'}, chips:chips(q, 'ok')}];
  F.push({shapes:queueShapes(q), panels:P(), line:4,
    msg:{zh:'佇列有<b>兩個口</b>：尾巴進、頭出。就是排隊買票。',
         en:'A queue has <b>two ends</b>: in at the rear, out at the front. It is a ticket line.'}});
  ['a', 'b', 'c'].forEach(v => {
    F.push({shapes:queueShapes(q, {fly:v}), panels:P(), line:6,
      msg:{zh:'<b>push(' + v + ')</b>：從 rear 排進去。',
           en:'<b>push(' + v + ')</b>: join at the rear.'}});
    q.push(v);
    F.push({shapes:queueShapes(q, {states:{[q.length - 1]:'ok'}}), panels:P(), line:7,
      msg:{zh:'目前隊伍：<b>' + q.join(', ') + '</b>。',
           en:'Queue now: <b>' + q.join(', ') + '</b>.'}});
  });
  for (let k = 0; k < 2; k++){
    const v = q[0];
    F.push({shapes:queueShapes(q, {states:{0:'hot'}}), panels:P(), line:9,
      msg:{zh:'<b>pop()</b> 拿走 front 的 <b>' + v + '</b>——最早排的最先走，<b>FIFO</b>。',
           en:'<b>pop()</b> takes <b>' + v + '</b> from the front - first in, first out, <b>FIFO</b>.'}});
    q.shift();
    F.push({shapes:queueShapes(q, {out:v, states:{0:'ok'}}), panels:P(), line:9,
      msg:{zh:'注意用的是 <b>deque.popleft()</b> 而不是 <b>list.pop(0)</b>：' +
              'list 拔掉第 0 格要把後面全部往前搬，是 <b>O(n)</b>；deque 兩端都是 <b>O(1)</b>。',
           en:'Note it is <b>deque.popleft()</b>, not <b>list.pop(0)</b>: popping index 0 of a list shifts everything left, <b>O(n)</b>, while a deque is <b>O(1)</b> at both ends.'}});
  }
  F.push({shapes:queueShapes(q, {states:{0:'ok'}}), panels:P(), line:0,
    msg:{zh:'公平排隊的地方就有佇列：工作排程、印表機、訊息佇列，還有一層一層擴散的圖走訪。',
         en:'Anywhere order must stay fair you find a queue: job schedulers, printers, message queues - and the ring-by-ring graph traversal you meet later.'}});
  return F.list;
}

function buildBrackets(v){
  const s = v === 0 ? '{[()]}' : '([)]';
  const F = new Frames(), pairs = {')':'(', ']':'[', '}':'{'}, st = [];
  const strShapes = (i, styleOf) => {
    const out = [], x0 = 2.55;
    for (let k = 0; k < s.length; k++)
      out.push(S.r(x0 + k * .78, .55, .68, .78, styleOf(k), s[k], {fs:.48}));
    return out;
  };
  const P = () => [{lbl:{zh:'堆疊', en:'stack'}, chips:chips(st, 'act')}];
  const view = () => stackShapes(st, {cap:{zh:'還沒配對到的左括號', en:'unmatched openers'}});
  F.push({shapes:strShapes(-1, () => 'idle').concat(view()), panels:P(), line:2,
    msg:{zh:'字串 <b>' + s + '</b>。括號配對的重點是：<b>最近打開的，必須最先關掉</b>——這正是堆疊。',
         en:'String <b>' + s + '</b>. Bracket matching says: <b>the most recently opened must close first</b> - exactly a stack.'}});
  for (let i = 0; i < s.length; i++){
    const ch = s[i];
    const styleOf = k => k < i ? 'done' : (k === i ? 'hot' : 'idle');
    if ('([{'.indexOf(ch) >= 0){
      st.push(ch);
      F.push({shapes:strShapes(i, styleOf).concat(stackShapes(st, {states:{[st.length - 1]:'hot'}})),
        panels:P(), line:5,
        msg:{zh:'左括號 <b>' + ch + '</b>：先記著，push 進堆疊。',
             en:'Opener <b>' + ch + '</b>: remember it, push onto the stack.'}});
    } else {
      const want = pairs[ch], got = st.length ? st[st.length - 1] : null;
      F.push({shapes:strShapes(i, styleOf).concat(
        stackShapes(st, {states:{[st.length - 1]:got === want ? 'ok' : 'bad'}})),
        panels:P(), line:7,
        msg:{zh:'右括號 <b>' + ch + '</b>，需要的搭檔是 <b>' + want + '</b>；堆疊頂端是 <b>' +
                (got || '（空的）') + '</b>——' + (got === want ? '對上了。' : '對不上。'),
             en:'Closer <b>' + ch + '</b> needs <b>' + want + '</b>; top of stack is <b>' +
                (got || '(empty)') + '</b> - ' + (got === want ? 'match.' : 'mismatch.')}});
      if (got !== want){
        F.push({shapes:strShapes(i, k => k === i ? 'bad' : (k < i ? 'done' : 'idle')).concat(
          stackShapes(st, {states:{[st.length - 1]:'bad'}})), panels:P(), line:8,
          msg:{zh:'直接回傳 <b>False</b>。<b>' + s + '</b> 的括號雖然數量一樣多，' +
                  '卻交叉了——這種錯誤只有堆疊看得出來。',
               en:'Return <b>False</b> immediately. <b>' + s + '</b> has the right counts but the brackets cross - only a stack notices.'}});
        return F.list;
      }
      st.pop();
      F.push({shapes:strShapes(i, k => k <= i ? 'done' : 'idle').concat(stackShapes(st)),
        panels:P(), line:7,
        msg:{zh:'配對成功，把 <b>' + want + '</b> 彈掉。堆疊少一層。',
             en:'Matched, so pop <b>' + want + '</b>. The stack gets one level shallower.'}});
    }
  }
  F.push({shapes:strShapes(s.length, () => 'ok').concat(stackShapes(st)), panels:P(), line:9,
    msg:{zh:'掃完了而且堆疊是空的，回傳 <b>True</b>。整個過程只掃一遍字串——<b>O(n)</b>。',
         en:'Scanned the whole string and the stack is empty, so return <b>True</b>. One pass, <b>O(n)</b>.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'堆疊與佇列', en:'Stack and Queue'},
  sub:{zh:'後進先出 vs 先進先出：兩種「只准從特定一端動手」的容器，以及它們最經典的用途。',
       en:'LIFO vs FIFO: two containers that deliberately restrict where you may reach in, and what that restriction buys you.'},
  tabs:[
    {id:'stack', label:{zh:'堆疊 Stack', en:'Stack'},
     stage:{zh:'push / pop 都在同一端', en:'push and pop share one end'},
     idea:{zh:'堆疊的價值來自<b>限制</b>：只能碰最上面那個。因為不用搬資料，push/pop/peak 全是 <b>O(1)</b>，' +
              '而且它天然記得「最近發生的事」。',
           en:'A stack earns its keep by <b>restriction</b>: you may only touch the top. Nothing is ever shifted, so push/pop/peak are all <b>O(1)</b>, and it naturally remembers "what happened most recently".'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_STACK, build:buildStack},
    {id:'queue', label:{zh:'佇列 Queue', en:'Queue'},
     stage:{zh:'尾巴進、頭出', en:'in at the rear, out at the front'},
     idea:{zh:'佇列保住的是<b>先來後到</b>。用 <b>deque</b> 而不是 list，是因為 list 的 pop(0) 要把整排往前搬；' +
              'deque 兩端都是 O(1)。',
           en:'A queue preserves <b>arrival order</b>. Use <b>deque</b> rather than a list: list.pop(0) shifts every element left, while a deque is O(1) at both ends.'},
     legend:['hot', 'act', 'ok', 'idle'], code:CODE_QUEUE, build:buildQueue},
    {id:'br', label:{zh:'應用：括號配對', en:'LeetCode 20'},
     stage:{zh:'用堆疊檢查括號是否合法', en:'Validating brackets with a stack'},
     variants:[{zh:'合法 {[()]}', en:'valid {[()]}'}, {zh:'交叉 ([)]', en:'crossing ([)]'}],
     idea:{zh:'只要問題長得像「最近打開的要最先關上」——括號、HTML 標籤、函式呼叫、遞迴——' +
              '堆疊就是答案。反過來，需要層層公平擴散的時候才換佇列。',
           en:'Whenever a problem reads "the most recently opened must close first" - brackets, HTML tags, function calls, recursion - a stack is the answer. Fair, level-by-level spreading is the queue case instead.'},
     legend:['hot', 'ok', 'bad', 'done'], code:CODE_BR, build:buildBrackets}
  ]
};
