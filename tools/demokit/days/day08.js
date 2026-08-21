// DAY: 08
// TITLE_ZH: Word Break
// TITLE_EN: Word Break
// SUB_ZH: 從指數級的暴力切法，到記憶化，再到一維 DP——同一題三種寫法看清楚重疊子問題。
// SUB_EN: From exponential brute force to memoisation to a one-line DP.
// FOLDER: day%2008%20-%20word%20break
// MEDIUM: https://medium.com/100-days-of-python/day-08-problem-word-break-15d8256cd0f9

const SS = 'catsanddog';
const DICT = ['cat', 'cats', 'and', 'sand', 'dog'];
const LW = .78, LH = .88, LY = 1.25, LX = 5.0 - SS.length * LW / 2;
function strShapesOf(str, states, cuts){
  const out = [], x0 = 5.0 - str.length * LW / 2;
  for (let i = 0; i < str.length; i++)
    out.push(S.r(x0 + i * LW, LY, LW - .06, LH, (states && states[i]) || 'idle', str[i], {fs:.44}));
  for (let i = 0; i <= str.length; i++)
    out.push(S.t(x0 + i * LW - .03, LY - .26, String(i), {c:'#8fa3ac', fs:.24}));
  (cuts || []).forEach(c => out.push(S.e(x0 + c * LW - .03, LY - .12, x0 + c * LW - .03, LY + LH + .16,
    {s:'act', arrow:false, w:.06})));
  return out;
}
const strShapes = (states, cuts) => strShapesOf(SS, states, cuts);
function dpShapes(dp, states){
  const out = [], y = 3.1;
  for (let i = 0; i < dp.length; i++){
    const v = dp[i];
    out.push(S.r(LX + i * LW - .38, y, LW - .06, LH,
      (states && states[i]) || (v === true ? 'ok' : (v === false ? 'done' : 'ghost')),
      v === null ? '?' : (v ? 'T' : 'F'), {fs:.40}));
    out.push(S.t(LX + i * LW - .38 + (LW - .06) / 2, y + LH + .32, String(i), {c:'#8fa3ac', fs:.24}));
  }
  out.push(S.t(LX - .70, y + LH * .62, 'dp', {c:'#3fe0dd', fs:.34, anchor:'end'}));
  return out;
}
const dictChips = hot => ({lbl:{zh:'字典', en:'word dict'},
  chips:DICT.map(w => ({t:w, cls:w === hot ? 'ok' : ''}))});
const chipsOf = (a, cls) => a.map(x => ({t:String(x), cls:cls || ''}));

const CODE_DP = [
'def word_break(s, words):       # LeetCode 139',
'    ws = set(words)             # O(1) 查字典',
'    dp = [False] * (len(s) + 1)',
'    dp[0] = True                # 空字串一定切得出來',
'    for i in range(1, len(s) + 1):',
'        for j in range(i):',
'            if dp[j] and s[j:i] in ws:',
'                dp[i] = True    # 前 i 個字切得開',
'                break',
'    return dp[len(s)]'];
const CODE_REC = [
'def word_break(s, words, memo=None):',
'    if memo is None: memo = {}',
'    if s in memo: return memo[s]        # 查快取',
'    if s == "": return True',
'    for w in words:',
'        if s.startswith(w):',
'            if word_break(s[len(w):], words, memo):',
'                memo[s] = True',
'                return True',
'    memo[s] = False                     # 記住「這段切不開」',
'    return False'];
const CODE_II = [
'def word_break_ii(s, words):    # LeetCode 140：回傳所有切法',
'    ws, memo = set(words), {}',
'    def go(rest):',
'        if rest in memo: return memo[rest]',
'        if rest == "": return [""]',
'        out = []',
'        for i in range(1, len(rest) + 1):',
'            w = rest[:i]',
'            if w in ws:',
'                for tail in go(rest[i:]):',
'                    out.append(w if tail == "" else w + " " + tail)',
'        memo[rest] = out',
'        return out',
'    return go(s)'];

function buildDP(){
  const F = new Frames(), n = SS.length;
  const dp = new Array(n + 1).fill(null); dp[0] = true;
  F.push({shapes:strShapes({}).concat(dpShapes(dp, {0:'ok'})),
    panels:[dictChips()], line:3,
    msg:{zh:'定義 <b>dp[i] = 前 i 個字元能不能剛好切成字典裡的詞</b>。' +
            '<b>dp[0] = True</b>：空字串當然切得出來（這個起手式決定了整個遞推能不能跑）。',
         en:'Define <b>dp[i] = can the first i characters be split exactly into dictionary words</b>. <b>dp[0] = True</b>: the empty string trivially can - and that seed is what makes the whole recurrence work.'}});
  for (let i = 1; i <= n; i++){
    dp[i] = false;
    for (let j = 0; j < i; j++){
      const w = SS.slice(j, i), ok = dp[j] && DICT.indexOf(w) >= 0;
      const st = {}; for (let k = j; k < i; k++) st[k] = dp[j] ? 'hot' : 'done';
      F.push({shapes:strShapes(st, [j, i]).concat(dpShapes(dp, {[j]:dp[j] ? 'ok' : 'done', [i]:'hot'})),
        panels:[dictChips(DICT.indexOf(w) >= 0 ? w : null)], line:6,
        msg:{zh:'試切點 <b>j = ' + j + '</b>：左邊 dp[' + j + '] = <b>' + (dp[j] ? 'True' : 'False') +
                '</b>，右邊那段是 <b>"' + w + '"</b>' +
                (dp[j] ? (DICT.indexOf(w) >= 0 ? '，在字典裡 → dp[' + i + '] = True。'
                                               : '，不在字典裡。')
                       : '。左邊本來就切不開，這個切點直接跳過。'),
             en:'Try cut <b>j = ' + j + '</b>: dp[' + j + '] is <b>' + (dp[j] ? 'True' : 'False') +
                '</b> and the piece is <b>"' + w + '"</b>' +
                (dp[j] ? (DICT.indexOf(w) >= 0 ? ', which is in the dictionary, so dp[' + i + '] = True.'
                                               : ', which is not in the dictionary.')
                       : '. The left part is unsplittable, so this cut is skipped.')}});
      if (ok){ dp[i] = true; break; }
    }
    if (!dp[i])
      F.push({shapes:strShapes({}).concat(dpShapes(dp, {[i]:'done'})), panels:[dictChips()], line:9,
        msg:{zh:'所有切點都試過了，<b>dp[' + i + '] = False</b>。前 ' + i + ' 個字（"' +
                SS.slice(0, i) + '"）切不開。',
             en:'Every cut failed, so <b>dp[' + i + '] = False</b>: the first ' + i + ' characters ("' +
                SS.slice(0, i) + '") cannot be split.'}});
  }
  F.push({shapes:strShapes({}, [3, 7]).concat(dpShapes(dp, {[n]:'ok'})), panels:[dictChips()], line:9,
    msg:{zh:'<b>dp[' + n + '] = True</b>，答案是可以（cats | and | dog）。' +
            '兩層迴圈是 <b>O(n²)</b> 次切點，每次比對字串 O(n)，' +
            '而且<b>每個 dp[i] 只算一次</b>——這就是動態規劃相對於暴力遞迴的全部差別。',
         en:'<b>dp[' + n + '] = True</b>: yes, it splits (cats | and | dog). Two loops give <b>O(n²)</b> cuts, and crucially <b>each dp[i] is computed exactly once</b> - that is the entire difference from brute force.'}});
  return F.list;
}

function buildRec(v){
  const STR = v === 1 ? 'catsandog' : SS;
  const F = new Frames(), memo = {};
  const stack = [];
  const P = () => [
    {lbl:{zh:'呼叫堆疊（剩下的字串）', en:'call stack (remaining suffix)'}, chips:chipsOf(stack, 'act')},
    {lbl:{zh:'memo 快取', en:'memo cache'},
     chips:Object.keys(memo).map(k => ({t:(k || '""') + '=' + (memo[k] ? 'T' : 'F'), cls:memo[k] ? 'ok' : 'bad'}))}];
  const showSuffix = (rest, st) => {
    const start = STR.length - rest.length, states = {};
    for (let i = 0; i < start; i++) states[i] = 'done';
    for (let i = start; i < STR.length; i++) states[i] = st || 'hot';
    return strShapesOf(STR, states, [start]);
  };
  let hits = 0;
  const go = rest => {
    stack.push(rest || '""');
    if (rest in memo){
      hits++;
      F.push({shapes:showSuffix(rest, 'ok'), panels:P(), line:2,
        msg:{zh:'<b>"' + rest + '"</b> 之前算過了 → 直接從 memo 拿 <b>' + memo[rest] +
                '</b>。這一步省下的是<b>一整棵子樹</b>的計算。',
             en:'<b>"' + rest + '"</b> was solved before, so read <b>' + memo[rest] +
                '</b> from memo. What that saves is <b>an entire subtree</b> of work.'}});
      stack.pop();
      return memo[rest];
    }
    F.push({shapes:showSuffix(rest), panels:P(), line:4,
      msg:{zh:'處理剩下的 <b>"' + (rest || '（空）') + '"</b>：試每一個開頭符合的字典詞。',
           en:'Handling the remaining <b>"' + (rest || '(empty)') + '"</b>: try every dictionary word that prefixes it.'}});
    if (rest === ''){
      F.push({shapes:showSuffix('', 'ok'), panels:P(), line:3,
        msg:{zh:'剩下空字串 → <b>True</b>。這條切法成功了。',
             en:'Nothing left, so <b>True</b> - this chain of cuts works.'}});
      stack.pop(); return true;
    }
    for (const w of DICT){
      if (rest.indexOf(w) === 0){
        F.push({shapes:showSuffix(rest, 'act'), panels:[dictChips(w), P()[1]], line:5,
          msg:{zh:'開頭配到 <b>"' + w + '"</b> → 遞迴處理剩下的 <b>"' + rest.slice(w.length) + '"</b>。',
               en:'It starts with <b>"' + w + '"</b>, so recurse on the rest: <b>"' + rest.slice(w.length) + '"</b>.'}});
        if (go(rest.slice(w.length))){ memo[rest] = true; stack.pop(); return true; }
      }
    }
    memo[rest] = false;
    F.push({shapes:showSuffix(rest, 'bad'), panels:P(), line:9,
      msg:{zh:'<b>"' + rest + '"</b> 怎麼切都不行 → 記進 memo。' +
              '<b>記住失敗比記住成功更值錢</b>：以後別的路徑走到同一段就不用再試一次。',
           en:'<b>"' + rest + '"</b> fails every way, so record it. <b>Remembering failures matters more than remembering successes</b>: any other path reaching this suffix stops immediately.'}});
    stack.pop(); return false;
  };
  F.push({shapes:strShapesOf(STR, {}), panels:P(), line:0,
    msg:{zh:'字串是 <b>"' + STR + '"</b>。換個角度：<b>切掉開頭的一個詞，剩下的還是同一個問題</b>。' +
            '直接遞迴會把同一段字串重算很多次，所以加一個 memo。',
         en:'Another angle: <b>strip one word off the front and the rest is the same problem</b>. Plain recursion re-solves the same suffix over and over, so we add a memo.'}});
  const ans = go(STR);
  F.push({shapes:strShapesOf(STR, {}, v === 1 ? [] : [3, 7]), panels:P(), line:2,
    msg:{zh:'答案 <b>' + ans + '</b>。過程中命中快取 <b>' + hits + '</b> 次。' +
            '不同的切法一定會走到相同的後綴——<b>重疊子問題</b>就是這樣長出來的，' +
            '有了 memo，狀態數就從指數變成「後綴的數量」也就是 O(n)。',
         en:'Answer <b>' + ans + '</b>, with <b>' + hits + '</b> cache hits. Different cut sequences inevitably land on the same suffix - that is what <b>overlapping subproblems</b> means - and with a memo the state count drops from exponential to "number of suffixes", i.e. O(n).'}});
  return F.list;
}

function buildII(){
  const F = new Frames(), memo = {};
  const showSuffix = (rest, st) => {
    const start = SS.length - rest.length, states = {};
    for (let i = 0; i < start; i++) states[i] = 'done';
    for (let i = start; i < SS.length; i++) states[i] = st || 'hot';
    return strShapes(states, [start]);
  };
  const P = () => [{lbl:{zh:'memo：每段後綴的所有切法', en:'memo: all splits per suffix'},
    chips:Object.keys(memo).map(k => ({t:(k || '""') + ' → ' + memo[k].length, cls:memo[k].length ? 'ok' : 'bad'}))}];
  F.push({shapes:strShapes({}), panels:P(), line:0,
    msg:{zh:'LeetCode 140 要的不是「能不能」，而是<b>所有切法</b>。' +
            '布林 dp 不夠用了——每個狀態要存的是<b>一串答案</b>。',
         en:'LeetCode 140 asks not "can it" but <b>for every split</b>. A boolean dp is no longer enough: each state must hold <b>a list of answers</b>.'}});
  const go = rest => {
    if (rest in memo) return memo[rest];
    if (rest === '') return [''];
    const out = [];
    for (let i = 1; i <= rest.length; i++){
      const w = rest.slice(0, i);
      if (DICT.indexOf(w) >= 0){
        F.push({shapes:showSuffix(rest, 'act'), panels:[dictChips(w)].concat(P()), line:8,
          msg:{zh:'在 <b>"' + rest + '"</b> 的開頭找到 <b>"' + w + '"</b>，' +
                  '把剩下的 <b>"' + rest.slice(i) + '"</b> 的每一種切法都接在後面。',
               en:'Found the prefix <b>"' + w + '"</b> of <b>"' + rest + '"</b>; every split of the remaining <b>"' +
                  rest.slice(i) + '"</b> gets it prepended.'}});
        go(rest.slice(i)).forEach(tail => out.push(tail === '' ? w : w + ' ' + tail));
      }
    }
    memo[rest] = out;
    F.push({shapes:showSuffix(rest, out.length ? 'ok' : 'bad'),
      panels:[{lbl:{zh:'"' + rest + '" 的切法', en:'splits of "' + rest + '"'},
               chips:out.length ? out.map(o => ({t:o, cls:'ok'})) : []}].concat(P()), line:11,
      msg:{zh:'<b>"' + rest + '"</b> 一共有 <b>' + out.length + '</b> 種切法，存進 memo。' +
              '注意存的是<b>這一段的答案</b>，跟前面怎麼切完全無關——這正是子問題可以共用的原因。',
           en:'<b>"' + rest + '"</b> has <b>' + out.length + '</b> splits; store them. Note the entry describes <b>this suffix alone</b>, independent of how we arrived - which is precisely why subproblems can be shared.'}});
    return out;
  };
  const all = go(SS);
  F.push({shapes:strShapes({}, [4, 8]),
    panels:[{lbl:{zh:'最終答案', en:'final answer'}, chips:all.map(a => ({t:a, cls:'ok'}))}], line:12,
    msg:{zh:'答案：<b>' + all.join('　/　') + '</b>。' +
            '兩題的骨架一模一樣，差別只在 memo 存布林還是存清單。' +
            '不過要小心：<b>切法的數量本身可能是指數級</b>，memo 能省重算，但省不掉輸出。',
         en:'Answer: <b>' + all.join('  /  ') + '</b>. Same skeleton as before; only the memo payload changed from a boolean to a list. One caveat: <b>the number of splits can itself be exponential</b> - memoisation removes recomputation, not output size.'}});
  return F.list;
}

const DAY_META = {
  title:{zh:'Word Break', en:'Word Break'},
  sub:{zh:'從指數級的暴力切法，到記憶化，再到一維 DP——同一題三種寫法看清楚重疊子問題。',
       en:'Brute force, memoisation, then a one-dimensional DP - one problem, three angles on overlapping subproblems.'},
  tabs:[
    {id:'dp', label:{zh:'一維 DP（LC 139）', en:'1-D DP (LC 139)'},
     stage:{zh:'dp[i] = 前 i 個字切得開嗎', en:'dp[i]: can the first i characters split?'},
     idea:{zh:'DP 的三件事：<b>狀態</b>（dp[i]）、<b>轉移</b>（找一個切點 j，左邊已知可切、右邊是個詞）、' +
              '<b>起始值</b>（dp[0] = True）。想清楚這三件事，程式只有五行。',
           en:'A DP needs three things: a <b>state</b> (dp[i]), a <b>transition</b> (find a cut j where the left is already splittable and the right is a word), and a <b>base case</b> (dp[0] = True). Get those right and the code is five lines.'},
     legend:['hot', 'ok', 'done', 'idle'], code:CODE_DP, build:buildDP},
    {id:'rec', label:{zh:'遞迴 + 記憶化', en:'recursion + memo'},
     stage:{zh:'同一段後綴只算一次', en:'Each suffix is solved once'},
     variants:[{zh:'成功 catsanddog', en:'success: catsanddog'}, {zh:'失敗 catsandog', en:'failure: catsandog'}],
     idea:{zh:'不同的切法會走到相同的後綴，這就是<b>重疊子問題</b>。' +
              '加一個 memo 之後，狀態數等於後綴數量，指數級就塌成線性。' +
              '記得<b>失敗也要記</b>，那才是省下最多時間的部分。',
           en:'Different cut sequences reach the same suffix - the definition of <b>overlapping subproblems</b>. With a memo the state count equals the number of suffixes and the exponential collapses. Remember to <b>cache failures too</b>; that is where most of the savings are.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_REC, build:buildRec},
    {id:'ii', label:{zh:'Word Break II（LC 140）', en:'Word Break II (LC 140)'},
     stage:{zh:'memo 從布林換成清單', en:'The memo now holds lists, not booleans'},
     idea:{zh:'「能不能」變成「有幾種」時，骨架不用改，只要把每個狀態的內容從 True/False 換成一串答案。' +
              '這個「狀態存什麼」的選擇，是 DP 題目之間唯一真正的差別。',
           en:'Turning "can it?" into "in how many ways?" leaves the skeleton alone: only the payload of each state changes from a boolean to a list. Choosing what a state stores is the one thing that really differs between DP problems.'},
     legend:['hot', 'act', 'ok', 'bad'], code:CODE_II, build:buildII}
  ]
};
