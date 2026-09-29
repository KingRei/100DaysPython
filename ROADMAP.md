# 100 Days of Python — Roadmap

已完成 Day 01–21（linked list → 選擇問題與 Top-K）。以下為 Day 22–100 的規劃。
原則：經典資料結構／演算法**壓到最精簡**（相似的主題同一天講完、順手帶到它的平行版本），
把省下來的篇幅全部灌進 LLM 內核與強化學習，最後 6 天做中型專案，把前面 94 天的積木組起來。

2026-08-20 改版重點：
- 經典段從原本的 Day 12–52 壓成 Day 12–43（省下 9 天）：MST、最短路、分治排序、
  平衡樹、機率型結構、字串等主題各自合併。
- **平行/並行不另闢章節**，融進各主題：排序那天講 parallel merge / sample sort，
  非比較排序那天講 scan (prefix-sum)，圖那天講平行 BFS 與 Δ-stepping，
  Day 43 收成系統性的 work-depth 模型，LLM 段接 continuous batching，RL 段接 fully async。
- 強化學習從 10 天擴成 **23 天（Day 72–94）**，而且重心放在演算法本身：
  Day 72–89 照 **Tianshou / CleanRL / TorchRL / SB3 / RLlib** 真正實作了什麼來排
  （DQN 家族與 distributional、TRPO/PPO、DDPG/TD3/SAC、HER/ICM/RND、offline RL、
  GAIL、MARL、model-based 與 MCTS）；Day 90–93 才轉到 LLM 後訓練與大規模 RL 基礎設施，
  對照 [radixark/miles](https://github.com/radixark/miles)（SGLang rollout + Megatron trainer，
  從 slime fork）看 rollout engine × trainer、TITO、fully async、MoE routing replay、
  低精度穩定性與容錯。Miles 只佔 4 天：它是把前面演算法放大到生產規模的案例，不是主角。

---

## Part 1 — 圖論（Day 12–19）
接續 Day 11 的圖表示法，把圖演算法一次走完。

| Day | 主題 | 配套題 / 備註 |
|---|---|---|
| 12 | BFS / DFS 走訪與應用 ✅ | LC 200 島嶼數量 |
| 13 | 拓樸排序（Kahn + DFS 兩種寫法）✅ | LC 207 課程表；DAG、循環偵測 |
| 14 | Union-Find（路徑壓縮 + 按秩合併）✅ | LC 547 省份數量 |
| 15 | 最小生成樹 — Kruskal ✅ | 接 Day 14；Prim 併到 Day 16 一起講 |
| 16 | Prim 的最小生成樹與 Dijkstra 的最短路徑，其實是同一支演算法 ✅ | LC 743；接 Day 05 heap。兩者只差 key 是「邊權」還是「距離和」 |
| 17 | 負權與全點對：Bellman-Ford / SPFA / Floyd-Warshall ✅ | LC 787；負權環偵測＝套利；Floyd 就是三層迴圈的 DP，k 必須在最外層 |
| 18 | Tarjan：強連通分量、橋、關節點 ✅ | LC 1192；disc/low 一次 DFS 三種讀法；跳過的是「邊」不是「父節點」；少了 on_stack 會把分量黏在一起 |
| 19 | A* 與平行圖搜尋 ✅ | 第一次碰 heuristic，AI 的橋樑；順帶 frontier 展開的平行 BFS、Δ-stepping |

## Part 2 — 排序與搜尋（Day 20–23）
四天講完，每天都帶一個「這東西怎麼平行做」。

| Day | 主題 | 備註 |
|---|---|---|
| 20 | 分治排序：Quick sort + Merge sort + 它們的平行版 ✅ | 三路 partition、最壞情況、外部排序；parallel merge、sample sort、work-depth 直覺 |
| 21 | 選擇問題：Heap sort、Top-K、Quickselect、中位數、Reservoir sampling ✅ | 接 Day 05；串流取樣；radix select 與 GPU top-k kernel（grid barrier、#3610 死鎖、ties） |
| 22 | 非比較排序：counting / radix / bucket ✅ | O(n) 的條件；平行 counting sort 的骨架就是 prefix-sum (scan) |
| 23 | 二分搜尋的三種邊界寫法 ✅ | 「在答案上二分」；LC 34/33/875/410；sglang 的五處邊界搜尋（graph bucket、cu_seqlens、radix cache gallop、qps autotune） |

## Part 3 — 字串演算法（Day 24–27）

| Day | 主題 | 備註 |
|---|---|---|
| 24 | Trie 與 Radix / Patricia Tree ✅ | 自動完成 + IP 路由表；longest prefix match；sglang RadixCache 的 split / page 對齊 / leaf-only LRU（RadixAttention 的地基） |
| 25 | 字串匹配：KMP + Rabin-Karp ✅ | failure function、rolling hash |
| 26 | Suffix Array + LCP 與 Aho-Corasick ✅ | 後綴結構與多模式匹配一起講 |
| 27 | 編輯距離與相似度：Levenshtein、Jaccard、MinHash ✅ | Levenshtein DP / backtrace / rolling row / bounded-k band、WER、LCS+diff；Jaccard shingles、MinHash、LSH banding；為 Day 66 檢索鋪路 |

## Part 4 — 進階樹、區間與機率型結構（Day 28–32）

| Day | 主題 | 備註 |
|---|---|---|
| 28 | 平衡樹一次講完：AVL、Red-Black、B-Tree ✅ | 接 Day 04；rotation 保中序、AVL 四種 case、紅黑五性質與 insert fixup（插入 ≤2 旋轉／刪除 ≤3）、B-tree 以 page 為節點（4KB→t=128，1e6 keys height 2）；LC 110/108/1382 + DSW |
| 29 | 區間結構：Segment Tree（lazy）+ Fenwick ✅ | 迭代 segment tree（2n 平坦陣列、每層最多 2 節點）、lazy 便條省 862 倍 node visits、Fenwick 靠 i & -i（n+1 格、快 2 倍）、兩個 BIT 做 range/range；min 沒有反元素所以 BIT 會靜默給錯；LC 307/315/370 |
| 30 | LRU / LFU Cache ✅ | LRU = map + 雙向串列（刪掉 get 的刷新就靜悄悄變 FIFO）、LFU 靠「min_freq 只有 +1 或重設 1 兩種變法」拿到 O(1)、平手比 recency（LC 460 會考）；兩種死法：scan pollution 讓 LRU/LFU/SLRU 全 0%（只有 MRU 96.1%）、熱點移動讓 LFU 剩 26.8%；sglang 五策略＝一顆 heap 五個 get_priority；LC 146/460/432；為 Day 57 KV cache 鋪路 |
| 31 | Skip List ✅ | 排序串列有順序卻不能二分（99 跳 vs 7 probe）、高度用擲硬幣決定一次不再改（完美快車道插一個 key 要改 16/16 節點）、`_descend` 的 update[]/rank[] 一趟供三種操作；span 讓同一趟走訪回答 ZRANK/ZRANGE，漏掉 `for i in range(lvl, self.level): span += 1` 會順序全對、search 全對、200 個 rank 錯 197；Redis p=0.25＝只留三分之一「額外」指標換 35% 跳數，MAXLEVEL 32／backward 指標／listpack 128；LC 1206 的 `<` 不能寫 `<=`、LC 315 對得上 Day 29 Fenwick；玩具 HNSW（幾何分布層數＋ef beam）16 倍資料只多 1.1 倍距離計算 |
| 32 | 機率型結構：Bloom Filter、HyperLogLog、Count-Min Sketch ✅ | 誤差的「方向」比大小重要：Bloom 只會多說 yes、CMS 只會多算、HLL 兩邊都偏（所以聯集免費、交集用排容 −16% 而聯集只差 −0.84%）；Bloom 10k@1% ＝ 9.59 bits/item、k=7、11982 B vs set 1103394 B（92 倍），超載是指數式崩潰（2x→16%、8x→97.5%），Kirsch–Mitzenmacher 兩個雜湊湊出 k 個；刪除不是沒實作而是不可能（m=64/k=3 清掉 alpha 順手弄丟 bravo 與 india），counting Bloom 用 4-bit 計數器換 4 倍空間；拿 filter 去重 20000 份文件會永久丟掉 22 份（0.16%）——差別在拿到 yes 之後做什麼；HLL p=14＝16384 B 固定不動，1e3→5e5 都在 1% 內（誤差 1.04/√m），merge 逐格取 max 且精確；CMS 5×5437 取 min 保證不少算，誤差是絕對值 ε·N＝前十名 +0.01%、後十名 +150%，conservative update 免費把平均誤差 0.3→0.1；LC 347 用 Counter+heap，stream 版 sketch 在 Zipf 上完全一致，但把分布壓平（頭 65 / 第 k 61）就漏掉 w-385 |

## Part 5 — 動態規劃、貪心與回溯（Day 33–39）

| Day | 主題 | 備註 |
|---|---|---|
| 33 | DP 入門：爬樓梯、硬幣找零 ✅ | DP 不是新演算法，是同一條遞推式換一個計算順序：naive f(30) 呼叫 4,356,617 次卻只有 31 個子問題（浪費 140,536 倍，f(0) 被算 1,346,269 次），n=28 時 naive 252.74 ms / memo 0.022 ms / rolling 0.0072 ms；遞推式是簡單的一半，「填表順序」才是會咬人的那一半（唯一規則：dp[i] 只依賴已填好的格子）；硬幣找零就是同一張表把 `+` 換成 `min`，coins=[1,5,6,9] amount=11 → `0 1 2 3 4 1 1 2 3 1 2 2`，答案 2 枚（6+5）而貪心給 3 枚（9+1+1）且不報錯（貪心只在 canonical 幣制成立），pick 陣列可回溯重建解，湊不出來自然留在 INF → -1；**本日核心 silent failure：兩個迴圈交換順序＝回答另一個問題** —— 硬幣在外＝組合（LC 518，[1,2,5]/5 → 4）、金額在外＝排列（LC 377，同題目 → 9），amount 30 時 58 vs 5,508,222，且 LC 377 名字叫「Combination Sum IV」卻要排列；漂亮的推論：stairs(n) ≡ coins {1,2} 的排列版（1,2,3,5,8,13… 完全重合，組合版只有 1,2,2,3,3,4）；一張表三個運算子：`+` 數方法、`min` 求最少、`or` 判可行（回顧 Day 08 word break：applepenapple → True 且僅 1 種切法，catsandog 沿路對得上 cat/cats/sand/and 仍卡在 og）；memo vs tabulation 的判準是狀態稀疏度：coins=[100,250] amount=10000 → 203 states vs 10,001 cells（49.3 倍），[1,2,5] amount=1000 → 1005 vs 1001（打平，且遞迴鏈剛好撞上預設 recursionlimit）；LC 70/322/518/377/139 |
| 34 | 背包問題（0/1、完全、多重） ✅ | Day 33 說「交換兩個迴圈＝回答另一個問題」，今天更小：**內層迴圈的「方向」就是「這個物品能不能重複拿」**——rope/book/pan/tent 重量 [3,4,5,6] 價值 [50,40,70,80] cap=10，同一段程式碼倒序＝130（rope+tent，重 9 **故意留一格空**，rope+pan 裝滿到 8 卻只值 120，所以「把空間用滿」的啟發式在四件物品上就輸了），正序＝150（3 條 rope，完全背包），差一個 `reversed()` 且 Python 一聲不吭；二維表最後一列 `0 0 0 50 50 70 80 90 120 130 130`，回溯靠「跟正上方那格不一樣＝拿了」還原 `(130, [0,3])`；多重背包 counts=[2,1,3,1] → 第三個答案 140，二進位拆分 `binary_split(13)=[1,2,4,6]`（0..13 每個數量都剛好是某個子集合，精確等價非近似，拆完跑的還是同一個倒序迴圈），weights=[3,4,5,6,7]/counts=[300,500,800,1200,2000]/cap=4000 → 同為 58,400，但攤開 4,800 物品 3,441 ms vs 拆分 50 物品 22 ms；貪心：ratio rope 16.67 > pan 14.00 > tent 13.33 > book 10.00，**可以切＝146.67 有證明（交換論證），不能切＝120 vs DP 130 且不報錯**；一列換運算子 `or` ＝ subset sum，LC 416 `[1,5,11,5]` True、`[2,2,3,5]` 偶數總和仍 False，整列布林塞進一個大整數 `bits |= bits << x`（位移讀快照，連方向都不用管），400 數字 target 9545：190 ms → 0.2 ms 約千倍；**pseudo-polynomial**：×1/10/100/1000 答案恆為 130，表格 44/404/4004/40004 格，輸入只有 15/30/47/64 bits → 對數值大小多項式、對輸入長度指數，這就是 0/1 背包仍是 NP-complete 的原因；LC 416/494/1049/474/279 |
| 35 | LIS 與 LCS ✅ | **本日核心：快的那個演算法，它的工作陣列不是答案**——patience sorting 九行 `O(n log n)`，但 `tails` 是記分板不是路線：`[2,6,8,3,4,5,1]` 跑完 `tails=[1,3,4,5]`（長度 4 正確），可是 `1` 是輸入的**最後一個**元素，這串根本不是子序列，真正的答案是 `[2,3,4,5]`；教科書的 `[10,9,2,5,3,7,101,18]` 剛好 `tails=[2,3,7,18]` 是合法 LIS，所以陷阱在範例上永遠不會爆；要拿回內容就得記 `idx[]`+`parent[]`（parent 在插入當下凍結，指向「那一刻」結束長度 k 的人，之後 `tails[k-1]` 被蓋掉也不影響）。`dp[i]` 必須定義成「**以 i 結尾**」，「前 i 個的 LIS」狀態不足（無法判斷能否接下去）；`bisect_left`／`bisect_right` 一個函式名＝嚴格遞增 vs 非遞減，`[1,3,3,3,5]` → 3 vs 5（day 23 的 lower_bound 回收）。LC 354 整題藏在排序 key 的一個負號：`(w, -h)` 同寬高度遞減＝遞減段最多貢獻一個元素，拿掉負號 `[[1,1],[1,2],[1,3]]` 回傳 3 且不報錯；LC 673 計數的 `=`（更長就取代）vs `+=`（等長就相加）寫反是標準死法，兩版都回傳合理整數，且 n≤2000 正是因為 patience 版沒地方放計數。LCS：`AGGTAB`/`GXTXAYB` → 4、回溯得 `GTAB`；滾動兩列拿得到數字但**拿不回路徑**（回 Day 34 同一個分裂），要路徑得用 Hirschberg。LC 1035「線不交叉」＝配對位置在兩排都遞增＝共同子序列，換套詞彙而已。**最漂亮的一段：兩個排列的 LCS 就是 LIS**（Hunt–Szymanski）——把 `q` 換成在 `p` 的索引再求 LIS，n=1200 隨機排列同為 67，`O(n·m)` 1.300s vs 0.0018s ≈ 700 倍，也是 `diff` 在真實檔案上快的原因；真實用途＝diff／git、ROUGE-L（"the cat sat on the mat" vs "the cat was sitting on the mat" = 0.769）、Needleman-Wunsch。LC 300/354/673/1035 |
| 36 | 區間 DP 與樹形 DP ✅ | **本日核心：子問題不再是前綴，而「哪個選擇讓兩邊獨立」每一題都要重問一次**——矩陣鏈乘 `dims=[40,20,30,10,30]`：`((A(BC))D)` 26,000、最糟括號法 69,000（2.65 倍），括號數是 Catalan 數所以枚舉無望；`dp[i][j]` 只問「最上層那一次乘法切在哪」，而**外層迴圈必須是區間長度**：直覺的 `for i: for j:` 會讀到 `dp[k+1][j]`（下面那一列，還是初始值 0），於是幾段乘法被當成免費，回傳 **20,000** —— 比真正的最小值小、對應不到任何一種括號法、Python 一聲不吭；`split[i][j]` 順手還原括號本身。LC 312 戳氣球 `[3,1,5,8]`：直覺的「**先**戳哪一顆」不可分解（k 戳掉後左半的右鄰居變成右半的球），回傳 **98** 而非 167 且不報錯；改問「開區間 (i,j) 裡**最後**戳哪一顆」——輪到它時兩側已空，鄰居正好是不會動的牆 i、j——167，和 24 種順序暴力一致，choice 表重播出真正順序 `[1,5,3,8]`。LC 1547 切木棍是**鏡像**：可分離的是**第一刀**（成本＝當下整段長度，切完兩段從此互不相干），n=7／cuts `[1,3,4,5]` 依序切 20、最佳 16；n≤10^6 但切點≤100 ＝ 表格必須以切點為索引。LC 516 character → 5（carac），區間 DP 四行，也等於 Day 35 的 `LCS(s, reversed s)`。LC 337：錯的不是遞迴式是**狀態**——每點只回報一個數字時，父節點無法知道「小孩自己有沒有被用掉」，值全正 ⇒ `max` 永遠選拿 ⇒ 父子一起拿，回傳 **12**（全部加總穿 DP 外衣），回報 `(robbed, skipped)` 一對才得 7；同一課也是 LC 124「節點**回報**的（往下最好的單臂）≠ 節點**計分**的（在它轉彎的路徑）」與樹直徑。樹上背包（選課＝先修樹）的合併雙層迴圈看似 `O(n^3)`：`min(size[v],budget)`／`min(size[c],budget-t)` 讓每個節點的工作量剛好是「LCA 是它的節點配對數」，全樹加總＝`C(n,2)`＝6 點 15，程式是數出來再 assert。深度陷阱：樹形 DP＝後序走訪，60,000 節點鏈狀樹遞迴直接 `RecursionError`（這個失敗至少大聲），stack 推出的順序倒著讀＝子先父後。LC 312/1547/516/337/124 |
| 37 | 狀態壓縮 DP（TSP） ✅ | **本日核心：表格索引從「前綴／容量／區間／子樹」變成一個集合，而集合就是一個整數**——五個操作 `has/add/remove/members/popcount` 撐起全天；Python 的 `&` 綁得比 `==` 緊（C 的經典 `mask & 1 == 0` 陷阱在這裡不存在），但 `~mask` 是負數、拿去當 list 索引會安靜地從另一頭讀，補集要寫 `mask ^ full`；子集枚舉 `sub = (sub-1) & mask` 必須是 do-while，寫成 `while sub:` 剛好漏掉放 base case 的空集合；「每個 mask 的每個子集」總量是 `3^n` 不是 `4^n`（n=10 → 59,049 vs 1,048,576）。**和 Day 36 正好相反：今天填表順序是免費的**——`mask | 1<<nxt` 數值必然大於 `mask`，所以一句 `for mask in range(1<<n)` 永遠讀不到空格子（程式直接數出「比自己超集還大的子集」有 0 個），難處全部搬到狀態本身。Held-Karp `dp[mask][last]`：台灣八個停靠點、5,040 種走法 → 689 km，五城市時 24 條長度 4 的前綴壓成 12 個狀態，`parent` 是把數字變回行程表的東西（遞迴不需要它，所以最常被省略、事後最痛），top-down `lru_cache` 版只碰到 449 個可達狀態而非 2,048。**本日 silent failure 有兩個，而且剛好把正確答案夾住**：(1) 狀態少掉 `last` 之後唯一寫得出來的成本是「到最近的已走訪城市」，那是**另一個問題的正確遞迴**——它長出一棵樹不是一條環，回傳 502 km，而同組資料的最小生成樹重量也正好 502（TSP 經典下界，永遠偏小、永遠合理、永遠不報錯）；(2) 最近鄰居法 `O(n²)` 給出貨真價實的環線 795 km（貴 106 km／15.4%），罪魁是內陸的日月潭：最近的是花蓮 74 km 而非台南 116 km，省下 42 km 卻把整個南部丟在後面（這組資料是設計過的——純環狀的台灣城市會讓貪心剛好找到最佳解）。兩個指數：n=15 時 `(n-1)!` 6,227,020,800 對上 `2^n·n²` 7,372,800；實測暴力 8.6/98.8/890 ms（n=8/9/10）、Held-Karp 1.2 ms（n=8）→ 614 ms（n=15），暴力外推到 n=15 約 42 小時。LC 943 是同一張表把距離換重疊、`min` 換 `max`，而它自己的陷阱正是 TSP 那個免費捷徑的鏡像——環線可以任選起點所以只種 `dp[1][0]` 沒差，這題是開放鏈、第一個字是真決定，只種一個起點仍回傳合法答案但 18 字元而非 16（小測資常常看不出來）；LC 526 則示範**有時候 `last` 真的可以從 mask 推回來**（`popcount(mask)` 就是位置），所以該問的不是「狀態夠小嗎」而是「下一步的成本只用狀態記得的東西寫得出來嗎」。LC 847/943/1125/526/698；超過約 20 個城市要改用啟發式 → Day 94 GA 解同一題（精確 vs 啟發式對照） |
| 38 | 貪心：區間排程、Huffman 編碼 ✅ | **本日核心：貪心選錯 key 不會 crash，它交回一個合法但更差的答案**——八場會議搶一間會議室，照結束時間排進 5 場（B C D F G）、照開始時間 3 場（A E G，A 一場兩小時的 design review 把底下三場短會一起卡死）、照會議長度 4 場（B C E G，兩場十五分鐘的看起來很便宜，結果 D 塞不下），**三份都是零衝突的合法時間表**，要暴力枚舉 2^8 個子集才看得出 5 是上限。結束時間是對的 key，理由是對後面所有會議來說唯一的限制就是「會議室幾點空出來」；證明是交換論證，而且可以直接跑成實驗——強制鎖住最早結束的 B 再暴力枚舉剩下的，5 = 5，代表這個貪心選擇不花代價。**題目稍微一改貪心就死**：每場會議值錢之後，同一個貪心拿 12（B C D F G）、DP 拿 17（A F H），而且照樣交回一份真的能用的時間表——加權區間排程要 Day 33 的 DP 配 Day 23 的 bisect，換 key 救不回來，因為「收下這場值不值得」取決於為它放棄了什麼。**Huffman 是可以被證明的貪心**：53 個字元、18 種符號，定長 5 bits = 265 bits，Huffman 204 bits（小 23%）；兩個細節在撐場面——`tie` 計數器（少了它權重相同時 Python 會掉下去比較 node payload，輸出取決於 tuple 巢狀 tuple 怎麼排）、以及 `heappush` 把合併後的節點**推回佇列**（合併節點比兩個小孩都重，必須以新權重重排，這是非用 heap 不可的唯一理由）。**本日 silent failure 就藏在那個 heappush**：排序一次、從左往右合併、永不重排 → 合併節點永遠是最輕的兩個之一，整棵樹垮成單邊梯子，最長編碼 6 bits → 17 bits，總共 316 bits（比完全不壓縮的 265 還大 19%）——但依然 prefix-free、依然一字不差還原、Kraft sum 依然剛好 1.000，連「編碼樹有沒有長出通往空處的分枝」這個結構檢查都攔不住。硬幣 1/3/4 付 6 給 [4,1,1] 三枚而不是兩枚 3，而 1/5/10/25 永遠正確，正是這個陷阱活這麼久的原因。LC 435（`s >= last_end`，端點相接不算重疊）對上 LC 452（氣球是閉區間，必須 `s > last`，把 435 原封不動搬過去就翻車，是最常見的追問翻車點）；LC 621 用 `max((peak-1)*(n+1)+tied, len(tasks))` 取代模擬（能這樣寫是因為字母只有 26 種）；LC 1046 是 Huffman 迴圈的倒影（取最重兩顆、推回差值，heapq 只有 min-heap 所以負號進去）；LC 134 是本日唯一不用交換論證的證明（tank 轉負則從 start 到 i 整段都不可能當起點，直接跳到 i+1，一趟掃完）。貪心不會大聲壞掉，它是以百分比壞掉——上線就把暴力解一起上線並在小輸入 assert 兩邊相等 → Day 52 BPE 是同一個頻率驅動合併迴圈的反面（合併最頻繁的相鄰對，而不是最罕見的兩個符號） |
| 39 | 回溯與剪枝：N-Queens、數獨 | 為 Day 92 MCTS 鋪路 |

## Part 6 — 數學與工程基本功（Day 40–43）

| Day | 主題 | 備註 |
|---|---|---|
| 40 | 位元運算技巧 | mask、popcount |
| 41 | 質數篩、快速冪、GCD、模運算 | |
| 42 | 隨機化演算法與蒙地卡羅 | 為 RL 鋪路 |
| 43 | 並行與平行模型：GIL、multiprocessing、asyncio、work-depth、map-reduce | 把前面各天散落的平行討論收成一套模型；為 Day 65 / 88 / 99 鋪路 |

---

## Part 7 — 機器學習底層（Day 44–51）
刻意壓縮：經典 ML 用兩天打包，把篇幅留給 LLM 與 RL。

| Day | 主題 | 備註 |
|---|---|---|
| 44 | numpy 向量化與自動微分直覺 | 手刻 mini autograd |
| 45 | 梯度下降家族：SGD、momentum、Adam | |
| 46 | 線性迴歸與邏輯迴歸 | 從零推導 |
| 47 | 反向傳播 + 純 numpy MLP | 全系列的分水嶺 |
| 48 | k-means 與 KNN | 兩個一起寫 |
| 49 | 決策樹 / 隨機森林 / Gradient Boosting | 三個一起寫 |
| 50 | PCA / SVD / t-SNE | 降維一次講完 |
| 51 | 評估與正則化：交叉驗證、overfitting、metrics | |

## Part 8 — LLM 內核（Day 52–65）
全部用 numpy / 少量 torch 從零實作。

| Day | 主題 | 備註 |
|---|---|---|
| 52 | BPE Tokenizer 從零實作 | 接 Day 38 Huffman |
| 53 | Attention 機制 | QKV、softmax 數值穩定 |
| 54 | Multi-Head Attention 與 Transformer Block | |
| 55 | 位置編碼：sinusoidal、RoPE、ALiBi | |
| 56 | 從零訓一個 char-level mini-GPT | 里程碑：能生成文字 |
| 57 | KV Cache | 接 Day 30 快取；prefill vs decode |
| 58 | RadixAttention：前綴共享的 KV cache | Day 24 radix tree + Day 30 LRU 驅逐；SGLang 的核心 |
| 59 | Paged Attention 與顯存管理 | 本質是分頁記憶體配置器 |
| 60 | 取樣策略：temperature、top-k、top-p | Day 72 bandit 的探索直覺 |
| 61 | Beam search 與 speculative decoding | |
| 62 | 量化：INT8 / FP8 / GPTQ 基礎 | 為 Day 90 低精度訓練鋪路 |
| 63 | MoE 與路由（top-k gating、load balance） | 為 Day 89 routing replay 鋪路 |
| 64 | FlashAttention 概念：tiling 與 online softmax | |
| 65 | Continuous batching 與推論排程器 | 接 Day 43 並行；Day 86 rollout engine 的地基 |

## Part 9 — 檢索、RAG 與記憶層（Day 66–71）

| Day | 主題 | 備註 |
|---|---|---|
| 66 | Embedding 與向量相似度檢索 | word2vec、cosine、暴力搜尋當 baseline |
| 67 | HNSW 近似最近鄰 | 接 Day 31 Skip List |
| 68 | IVF 與 Product Quantization | |
| 69 | BM25、hybrid search 與 rerank | 接 Day 25 rolling hash |
| 70 | Chunking 策略與 minimal RAG pipeline | 端到端可跑 |
| 71 | 記憶層：短期 / 長期 / episodic memory | Day 97 專案的地基 |

## Part 10 — 強化學習：演算法全覆蓋（Day 72–89）
這 18 天的選材直接照著幾套主流框架真正實作了什麼來排：覆蓋度參照 **Tianshou**，
每天的程式碼寫成 **CleanRL** 那種「一支檔案從 rollout 到 optimizer.step() 看得完」的風格，
架構名詞對照 **TorchRL** 的 primitives（TensorDict / collector / replay buffer / loss module），
數字跟 **Stable-Baselines3** 對照當 sanity check，分散式的部分看 **RLlib**（Ray actor-learner）。

| Day | 主題 | 備註 |
|---|---|---|
| 72 | 環境介面：Gymnasium API、wrapper、向量化環境與 collector | reset/step/terminated vs truncated；接 Day 43 並行，向量化就是 batch 化的 rollout |
| 73 | Multi-Armed Bandit：ε-greedy、UCB、Thompson | 接 Day 42 隨機化；對照 Day 60 取樣溫度 |
| 74 | MDP：Value Iteration 與 Policy Iteration | GridWorld；接 Part 5 的 DP |
| 75 | Monte Carlo 與 TD Learning：n-step、TD(λ) | bias-variance 的第一次現身 |
| 76 | Q-Learning 與 SARSA（表格式） | on-policy vs off-policy 的分水嶺 |
| 77 | DQN 家族：replay buffer、target network、Double、Dueling、PER | replay buffer 用 Day 21 的取樣結構，PER 用 Day 29 的 segment tree |
| 78 | Distributional RL：C51、QR-DQN、IQN → Rainbow | 學分布不只學期望值；分位數回歸 |
| 79 | REINFORCE 與 policy gradient | log-prob 技巧、baseline 降變異 |
| 80 | A2C 與 GAE | advantage estimation 的 λ 取捨 |
| 81 | NPG 與 TRPO：自然梯度與信賴域 | Fisher 矩陣、KL 約束，PPO 的前身 |
| 82 | PPO 與那些真正決定成敗的實作細節 | clipping、優勢正規化、value clipping、observation 正規化 |
| 83 | 連續控制：DDPG 與 TD3 | 決定性策略、twin critic、target policy smoothing、延遲更新 |
| 84 | SAC：最大熵 RL | 自動溫度調整；REDQ / CrossQ 的 sample efficiency 觀點 |
| 85 | 稀疏獎勵與探索：HER、ICM、RND | 事後重標記與好奇心；對照 Day 91 的可驗證獎勵 |
| 86 | Offline RL：BCQ、CQL、IQL、TD3+BC | distribution shift、保守價值估計；接 Day 76 off-policy |
| 87 | 模仿學習：BC、DAgger、GAIL | GAIL 的判別器就是學出來的 reward，直通 Day 90 reward model |
| 88 | Multi-Agent：self-play、MAPPO、QMIX | 值分解與 CTDE；呼應 Day 98 多代理協作 |
| 89 | Model-based 與規劃：Dyna、PSRL、MCTS、世界模型（Dreamer 概念） | 接 Day 39 回溯 + Day 19 A* + Day 73 UCB；MCTS 實作井字棋 / 連四 |

## Part 11 — LLM 後訓練與 RL 基礎設施（Day 90–93）
把前面 18 天的演算法搬到語言模型上，順便看清「大規模 RL 訓練框架長什麼樣」。
實作參照 [Miles](https://github.com/radixark/miles)（SGLang rollout + Megatron trainer，從 slime fork）與 SGLang 本身。

| Day | 主題 | 備註 |
|---|---|---|
| 90 | RLHF 全景：reward model 從零訓 + DPO / SimPO | Bradley-Terry loss；把 Day 56 的 mini-GPT 拿來 align |
| 91 | GRPO / GSPO / REINFORCE++ 與 RLVR | group-relative advantage：為什麼 LLM RL 丟掉 critic；可驗證獎勵與 reward hacking |
| 92 | RL 訓練框架的骨架：rollout engine × trainer | 接 Day 65 continuous batching：推論引擎當環境、TITO、權重同步、fully async 與 staleness |
| 93 | 大規模的魔鬼細節 | MoE routing replay（接 Day 63）、FP8 / MXFP8 / NVFP4 低精度穩定性（接 Day 62）、容錯與 observability |

## Part 12 — 無梯度優化（Day 94）

| Day | 主題 | 備註 |
|---|---|---|
| 94 | 演化與無梯度優化：GA、模擬退火、PSO、CMA-ES | 用 GA 解 Day 37 的 TSP；演化 vs 梯度，對照 Day 79 policy gradient |

## Part 13 — 中型專案：LLM Agent Harness（Day 95–100）
前面 94 天的零件，這裡組成一個能跑的東西。

| Day | 主題 | 備註 |
|---|---|---|
| 95 | Agent Harness I：tool calling 與 JSON action loop | 工具註冊表、schema 驗證、重試 |
| 96 | Agent Harness II：接上記憶層 | 用 Day 71 + Day 67 HNSW |
| 97 | Agent Harness III：ReAct / planner 與多步推理 | 錯誤恢復、預算控制 |
| 98 | Multi-Agent 協作與任務 DAG 排程 | 呼應 Day 13 拓樸排序 + Day 88 MARL |
| 99 | Eval Harness：LLM-as-judge、回歸測試、追蹤 | 也是 Day 91 verifier 的延伸 |
| 100 | 收官：100 天知識地圖 + 效能剖析 + 打包開源 | |

---

## 依賴關係（有意設計的伏筆）

- Day 05 heap → Day 16 Prim / Dijkstra、Day 21 heap sort、Day 67 HNSW 的候選集
- Day 13 拓樸排序 → Day 98 多代理任務 DAG
- Day 14 Union-Find → Day 15 Kruskal
- Day 22 prefix-sum (scan) → Day 43 平行模型 → Day 92 async rollout 排程
- Day 24 Trie / Radix Tree → Day 58 RadixAttention
- Day 29 Segment Tree → Day 77 prioritized replay buffer
- Day 30 LRU → Day 57 KV cache → Day 58 RadixAttention（radix tree + LRU 驅逐）→ Day 59 paged attention
- Day 31 Skip List → Day 67 HNSW
- Day 37 狀壓 TSP → Day 94 GA 解 TSP（精確 vs 啟發式對照）
- Day 38 Huffman → Day 52 BPE
- Day 39 回溯 → Day 89 MCTS
- Day 42 隨機化 → Day 73 bandit → Day 60 取樣策略（探索與溫度是同一件事）
- Day 43 asyncio / work-depth → Day 65 continuous batching → Day 72 向量化環境 → Day 92 fully async RL → Day 95 agent loop
- Day 56 mini-GPT → Day 90 RLHF / DPO → Day 91 GRPO → Day 95–100 專案
- Day 62 量化 → Day 93 低精度 RL 訓練
- Day 63 MoE 路由 → Day 93 rollout routing replay
- Day 65 推論排程器 → Day 92 rollout engine（同一個引擎，一個對外服務、一個當 RL 環境）
- Day 76 off-policy → Day 86 offline RL → Day 92 async 的 staleness 容忍度
- Day 80 GAE / Day 82 PPO → Day 91 GRPO（同一個 advantage，換成 group 內相對比較）
- Day 87 GAIL 判別器 → Day 90 reward model（都是學出來的獎勵）
- Day 91 verifier → Day 99 eval harness
