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
- 強化學習從 10 天擴成 **24 天（Day 71–94）**，而且重心放在演算法本身：
  Day 71–88 照 **Tianshou / CleanRL / TorchRL / SB3 / RLlib** 真正實作了什麼來排
  （DQN 家族與 distributional、TRPO/PPO、DDPG/TD3/SAC、HER/ICM/RND、offline RL、
  GAIL、MARL、model-based 與 MCTS）；Day 89–92 才轉到 LLM 後訓練與大規模 RL 基礎設施，
  對照 [radixark/miles](https://github.com/radixark/miles)（SGLang rollout + Megatron trainer，
  從 slime fork）看 rollout engine × trainer、TITO、fully async、MoE routing replay、
  低精度穩定性與容錯。Miles 只佔 4 天：它是把前面演算法放大到生產規模的案例，不是主角。

2026-10-14 調整：
- **砍掉原 Day 46（線性迴歸與邏輯迴歸）**：線性迴歸的 `2/n Xᵀ(Xw − y)` 在 Day 44 已經推過，
  剩下的 sigmoid / softmax / cross-entropy 併進反向傳播那天（新 Day 46）——MLP 的輸出層本來就要用。
- 省下的一天給**基因演算法**：原本一天要塞 GA + 模擬退火 + PSO + CMA-ES，現在 GA 自己一天（Day 93），
  其餘無梯度方法一天（Day 94）。
- 原 Day 47 之後全部往前移一天；Day 95–100 專案段不動，總數仍是 100 天。

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
| 27 | 編輯距離與相似度：Levenshtein、Jaccard、MinHash ✅ | Levenshtein DP / backtrace / rolling row / bounded-k band、WER、LCS+diff；Jaccard shingles、MinHash、LSH banding；為 Day 65 檢索鋪路 |

## Part 4 — 進階樹、區間與機率型結構（Day 28–32）

| Day | 主題 | 備註 |
|---|---|---|
| 28 | 平衡樹一次講完：AVL、Red-Black、B-Tree ✅ | 接 Day 04；rotation 保中序、AVL 四種 case、紅黑五性質與 insert fixup（插入 ≤2 旋轉／刪除 ≤3）、B-tree 以 page 為節點（4KB→t=128，1e6 keys height 2）；LC 110/108/1382 + DSW |
| 29 | 區間結構：Segment Tree（lazy）+ Fenwick ✅ | 迭代 segment tree（2n 平坦陣列、每層最多 2 節點）、lazy 便條省 862 倍 node visits、Fenwick 靠 i & -i（n+1 格、快 2 倍）、兩個 BIT 做 range/range；min 沒有反元素所以 BIT 會靜默給錯；LC 307/315/370 |
| 30 | LRU / LFU Cache ✅ | LRU = map + 雙向串列（刪掉 get 的刷新就靜悄悄變 FIFO）、LFU 靠「min_freq 只有 +1 或重設 1 兩種變法」拿到 O(1)、平手比 recency（LC 460 會考）；兩種死法：scan pollution 讓 LRU/LFU/SLRU 全 0%（只有 MRU 96.1%）、熱點移動讓 LFU 剩 26.8%；sglang 五策略＝一顆 heap 五個 get_priority；LC 146/460/432；為 Day 56 KV cache 鋪路 |
| 31 | Skip List ✅ | 排序串列有順序卻不能二分（99 跳 vs 7 probe）、高度用擲硬幣決定一次不再改（完美快車道插一個 key 要改 16/16 節點）、`_descend` 的 update[]/rank[] 一趟供三種操作；span 讓同一趟走訪回答 ZRANK/ZRANGE，漏掉 `for i in range(lvl, self.level): span += 1` 會順序全對、search 全對、200 個 rank 錯 197；Redis p=0.25＝只留三分之一「額外」指標換 35% 跳數，MAXLEVEL 32／backward 指標／listpack 128；LC 1206 的 `<` 不能寫 `<=`、LC 315 對得上 Day 29 Fenwick；玩具 HNSW（幾何分布層數＋ef beam）16 倍資料只多 1.1 倍距離計算 |
| 32 | 機率型結構：Bloom Filter、HyperLogLog、Count-Min Sketch ✅ | 誤差的「方向」比大小重要：Bloom 只會多說 yes、CMS 只會多算、HLL 兩邊都偏（所以聯集免費、交集用排容 −16% 而聯集只差 −0.84%）；Bloom 10k@1% ＝ 9.59 bits/item、k=7、11982 B vs set 1103394 B（92 倍），超載是指數式崩潰（2x→16%、8x→97.5%），Kirsch–Mitzenmacher 兩個雜湊湊出 k 個；刪除不是沒實作而是不可能（m=64/k=3 清掉 alpha 順手弄丟 bravo 與 india），counting Bloom 用 4-bit 計數器換 4 倍空間；拿 filter 去重 20000 份文件會永久丟掉 22 份（0.16%）——差別在拿到 yes 之後做什麼；HLL p=14＝16384 B 固定不動，1e3→5e5 都在 1% 內（誤差 1.04/√m），merge 逐格取 max 且精確；CMS 5×5437 取 min 保證不少算，誤差是絕對值 ε·N＝前十名 +0.01%、後十名 +150%，conservative update 免費把平均誤差 0.3→0.1；LC 347 用 Counter+heap，stream 版 sketch 在 Zipf 上完全一致，但把分布壓平（頭 65 / 第 k 61）就漏掉 w-385 |

## Part 5 — 動態規劃、貪心與回溯（Day 33–39）

| Day | 主題 | 備註 |
|---|---|---|
| 33 | DP 入門：爬樓梯、硬幣找零 ✅ | DP 不是新演算法，是同一條遞推式換一個計算順序：naive f(30) 呼叫 4,356,617 次卻只有 31 個子問題（浪費 140,536 倍，f(0) 被算 1,346,269 次），n=28 時 naive 252.74 ms / memo 0.022 ms / rolling 0.0072 ms；遞推式是簡單的一半，「填表順序」才是會咬人的那一半（唯一規則：dp[i] 只依賴已填好的格子）；硬幣找零就是同一張表把 `+` 換成 `min`，coins=[1,5,6,9] amount=11 → `0 1 2 3 4 1 1 2 3 1 2 2`，答案 2 枚（6+5）而貪心給 3 枚（9+1+1）且不報錯（貪心只在 canonical 幣制成立），pick 陣列可回溯重建解，湊不出來自然留在 INF → -1；**本日核心 silent failure：兩個迴圈交換順序＝回答另一個問題** —— 硬幣在外＝組合（LC 518，[1,2,5]/5 → 4）、金額在外＝排列（LC 377，同題目 → 9），amount 30 時 58 vs 5,508,222，且 LC 377 名字叫「Combination Sum IV」卻要排列；漂亮的推論：stairs(n) ≡ coins {1,2} 的排列版（1,2,3,5,8,13… 完全重合，組合版只有 1,2,2,3,3,4）；一張表三個運算子：`+` 數方法、`min` 求最少、`or` 判可行（回顧 Day 08 word break：applepenapple → True 且僅 1 種切法，catsandog 沿路對得上 cat/cats/sand/and 仍卡在 og）；memo vs tabulation 的判準是狀態稀疏度：coins=[100,250] amount=10000 → 203 states vs 10,001 cells（49.3 倍），[1,2,5] amount=1000 → 1005 vs 1001（打平，且遞迴鏈剛好撞上預設 recursionlimit）；LC 70/322/518/377/139 |
| 34 | 背包問題（0/1、完全、多重） ✅ | Day 33 說「交換兩個迴圈＝回答另一個問題」，今天更小：**內層迴圈的「方向」就是「這個物品能不能重複拿」**——rope/book/pan/tent 重量 [3,4,5,6] 價值 [50,40,70,80] cap=10，同一段程式碼倒序＝130（rope+tent，重 9 **故意留一格空**，rope+pan 裝滿到 8 卻只值 120，所以「把空間用滿」的啟發式在四件物品上就輸了），正序＝150（3 條 rope，完全背包），差一個 `reversed()` 且 Python 一聲不吭；二維表最後一列 `0 0 0 50 50 70 80 90 120 130 130`，回溯靠「跟正上方那格不一樣＝拿了」還原 `(130, [0,3])`；多重背包 counts=[2,1,3,1] → 第三個答案 140，二進位拆分 `binary_split(13)=[1,2,4,6]`（0..13 每個數量都剛好是某個子集合，精確等價非近似，拆完跑的還是同一個倒序迴圈），weights=[3,4,5,6,7]/counts=[300,500,800,1200,2000]/cap=4000 → 同為 58,400，但攤開 4,800 物品 3,441 ms vs 拆分 50 物品 22 ms；貪心：ratio rope 16.67 > pan 14.00 > tent 13.33 > book 10.00，**可以切＝146.67 有證明（交換論證），不能切＝120 vs DP 130 且不報錯**；一列換運算子 `or` ＝ subset sum，LC 416 `[1,5,11,5]` True、`[2,2,3,5]` 偶數總和仍 False，整列布林塞進一個大整數 `bits |= bits << x`（位移讀快照，連方向都不用管），400 數字 target 9545：190 ms → 0.2 ms 約千倍；**pseudo-polynomial**：×1/10/100/1000 答案恆為 130，表格 44/404/4004/40004 格，輸入只有 15/30/47/64 bits → 對數值大小多項式、對輸入長度指數，這就是 0/1 背包仍是 NP-complete 的原因；LC 416/494/1049/474/279 |
| 35 | LIS 與 LCS ✅ | **本日核心：快的那個演算法，它的工作陣列不是答案**——patience sorting 九行 `O(n log n)`，但 `tails` 是記分板不是路線：`[2,6,8,3,4,5,1]` 跑完 `tails=[1,3,4,5]`（長度 4 正確），可是 `1` 是輸入的**最後一個**元素，這串根本不是子序列，真正的答案是 `[2,3,4,5]`；教科書的 `[10,9,2,5,3,7,101,18]` 剛好 `tails=[2,3,7,18]` 是合法 LIS，所以陷阱在範例上永遠不會爆；要拿回內容就得記 `idx[]`+`parent[]`（parent 在插入當下凍結，指向「那一刻」結束長度 k 的人，之後 `tails[k-1]` 被蓋掉也不影響）。`dp[i]` 必須定義成「**以 i 結尾**」，「前 i 個的 LIS」狀態不足（無法判斷能否接下去）；`bisect_left`／`bisect_right` 一個函式名＝嚴格遞增 vs 非遞減，`[1,3,3,3,5]` → 3 vs 5（day 23 的 lower_bound 回收）。LC 354 整題藏在排序 key 的一個負號：`(w, -h)` 同寬高度遞減＝遞減段最多貢獻一個元素，拿掉負號 `[[1,1],[1,2],[1,3]]` 回傳 3 且不報錯；LC 673 計數的 `=`（更長就取代）vs `+=`（等長就相加）寫反是標準死法，兩版都回傳合理整數，且 n≤2000 正是因為 patience 版沒地方放計數。LCS：`AGGTAB`/`GXTXAYB` → 4、回溯得 `GTAB`；滾動兩列拿得到數字但**拿不回路徑**（回 Day 34 同一個分裂），要路徑得用 Hirschberg。LC 1035「線不交叉」＝配對位置在兩排都遞增＝共同子序列，換套詞彙而已。**最漂亮的一段：兩個排列的 LCS 就是 LIS**（Hunt–Szymanski）——把 `q` 換成在 `p` 的索引再求 LIS，n=1200 隨機排列同為 67，`O(n·m)` 1.300s vs 0.0018s ≈ 700 倍，也是 `diff` 在真實檔案上快的原因；真實用途＝diff／git、ROUGE-L（"the cat sat on the mat" vs "the cat was sitting on the mat" = 0.769）、Needleman-Wunsch。LC 300/354/673/1035 |
| 36 | 區間 DP 與樹形 DP ✅ | **本日核心：子問題不再是前綴，而「哪個選擇讓兩邊獨立」每一題都要重問一次**——矩陣鏈乘 `dims=[40,20,30,10,30]`：`((A(BC))D)` 26,000、最糟括號法 69,000（2.65 倍），括號數是 Catalan 數所以枚舉無望；`dp[i][j]` 只問「最上層那一次乘法切在哪」，而**外層迴圈必須是區間長度**：直覺的 `for i: for j:` 會讀到 `dp[k+1][j]`（下面那一列，還是初始值 0），於是幾段乘法被當成免費，回傳 **20,000** —— 比真正的最小值小、對應不到任何一種括號法、Python 一聲不吭；`split[i][j]` 順手還原括號本身。LC 312 戳氣球 `[3,1,5,8]`：直覺的「**先**戳哪一顆」不可分解（k 戳掉後左半的右鄰居變成右半的球），回傳 **98** 而非 167 且不報錯；改問「開區間 (i,j) 裡**最後**戳哪一顆」——輪到它時兩側已空，鄰居正好是不會動的牆 i、j——167，和 24 種順序暴力一致，choice 表重播出真正順序 `[1,5,3,8]`。LC 1547 切木棍是**鏡像**：可分離的是**第一刀**（成本＝當下整段長度，切完兩段從此互不相干），n=7／cuts `[1,3,4,5]` 依序切 20、最佳 16；n≤10^6 但切點≤100 ＝ 表格必須以切點為索引。LC 516 character → 5（carac），區間 DP 四行，也等於 Day 35 的 `LCS(s, reversed s)`。LC 337：錯的不是遞迴式是**狀態**——每點只回報一個數字時，父節點無法知道「小孩自己有沒有被用掉」，值全正 ⇒ `max` 永遠選拿 ⇒ 父子一起拿，回傳 **12**（全部加總穿 DP 外衣），回報 `(robbed, skipped)` 一對才得 7；同一課也是 LC 124「節點**回報**的（往下最好的單臂）≠ 節點**計分**的（在它轉彎的路徑）」與樹直徑。樹上背包（選課＝先修樹）的合併雙層迴圈看似 `O(n^3)`：`min(size[v],budget)`／`min(size[c],budget-t)` 讓每個節點的工作量剛好是「LCA 是它的節點配對數」，全樹加總＝`C(n,2)`＝6 點 15，程式是數出來再 assert。深度陷阱：樹形 DP＝後序走訪，60,000 節點鏈狀樹遞迴直接 `RecursionError`（這個失敗至少大聲），stack 推出的順序倒著讀＝子先父後。LC 312/1547/516/337/124 |
| 37 | 狀態壓縮 DP（TSP） ✅ | **本日核心：表格索引從「前綴／容量／區間／子樹」變成一個集合，而集合就是一個整數**——五個操作 `has/add/remove/members/popcount` 撐起全天；Python 的 `&` 綁得比 `==` 緊（C 的經典 `mask & 1 == 0` 陷阱在這裡不存在），但 `~mask` 是負數、拿去當 list 索引會安靜地從另一頭讀，補集要寫 `mask ^ full`；子集枚舉 `sub = (sub-1) & mask` 必須是 do-while，寫成 `while sub:` 剛好漏掉放 base case 的空集合；「每個 mask 的每個子集」總量是 `3^n` 不是 `4^n`（n=10 → 59,049 vs 1,048,576）。**和 Day 36 正好相反：今天填表順序是免費的**——`mask | 1<<nxt` 數值必然大於 `mask`，所以一句 `for mask in range(1<<n)` 永遠讀不到空格子（程式直接數出「比自己超集還大的子集」有 0 個），難處全部搬到狀態本身。Held-Karp `dp[mask][last]`：台灣八個停靠點、5,040 種走法 → 689 km，五城市時 24 條長度 4 的前綴壓成 12 個狀態，`parent` 是把數字變回行程表的東西（遞迴不需要它，所以最常被省略、事後最痛），top-down `lru_cache` 版只碰到 449 個可達狀態而非 2,048。**本日 silent failure 有兩個，而且剛好把正確答案夾住**：(1) 狀態少掉 `last` 之後唯一寫得出來的成本是「到最近的已走訪城市」，那是**另一個問題的正確遞迴**——它長出一棵樹不是一條環，回傳 502 km，而同組資料的最小生成樹重量也正好 502（TSP 經典下界，永遠偏小、永遠合理、永遠不報錯）；(2) 最近鄰居法 `O(n²)` 給出貨真價實的環線 795 km（貴 106 km／15.4%），罪魁是內陸的日月潭：最近的是花蓮 74 km 而非台南 116 km，省下 42 km 卻把整個南部丟在後面（這組資料是設計過的——純環狀的台灣城市會讓貪心剛好找到最佳解）。兩個指數：n=15 時 `(n-1)!` 6,227,020,800 對上 `2^n·n²` 7,372,800；實測暴力 8.6/98.8/890 ms（n=8/9/10）、Held-Karp 1.2 ms（n=8）→ 614 ms（n=15），暴力外推到 n=15 約 42 小時。LC 943 是同一張表把距離換重疊、`min` 換 `max`，而它自己的陷阱正是 TSP 那個免費捷徑的鏡像——環線可以任選起點所以只種 `dp[1][0]` 沒差，這題是開放鏈、第一個字是真決定，只種一個起點仍回傳合法答案但 18 字元而非 16（小測資常常看不出來）；LC 526 則示範**有時候 `last` 真的可以從 mask 推回來**（`popcount(mask)` 就是位置），所以該問的不是「狀態夠小嗎」而是「下一步的成本只用狀態記得的東西寫得出來嗎」。LC 847/943/1125/526/698；超過約 20 個城市要改用啟發式 → Day 93 GA 解同一題（精確 vs 啟發式對照） |
| 38 | 貪心：區間排程、Huffman 編碼 ✅ | **本日核心：貪心選錯 key 不會 crash，它交回一個合法但更差的答案**——八場會議搶一間會議室，照結束時間排進 5 場（B C D F G）、照開始時間 3 場（A E G，A 一場兩小時的 design review 把底下三場短會一起卡死）、照會議長度 4 場（B C E G，兩場十五分鐘的看起來很便宜，結果 D 塞不下），**三份都是零衝突的合法時間表**，要暴力枚舉 2^8 個子集才看得出 5 是上限。結束時間是對的 key，理由是對後面所有會議來說唯一的限制就是「會議室幾點空出來」；證明是交換論證，而且可以直接跑成實驗——強制鎖住最早結束的 B 再暴力枚舉剩下的，5 = 5，代表這個貪心選擇不花代價。**題目稍微一改貪心就死**：每場會議值錢之後，同一個貪心拿 12（B C D F G）、DP 拿 17（A F H），而且照樣交回一份真的能用的時間表——加權區間排程要 Day 33 的 DP 配 Day 23 的 bisect，換 key 救不回來，因為「收下這場值不值得」取決於為它放棄了什麼。**Huffman 是可以被證明的貪心**：53 個字元、18 種符號，定長 5 bits = 265 bits，Huffman 204 bits（小 23%）；兩個細節在撐場面——`tie` 計數器（少了它權重相同時 Python 會掉下去比較 node payload，輸出取決於 tuple 巢狀 tuple 怎麼排）、以及 `heappush` 把合併後的節點**推回佇列**（合併節點比兩個小孩都重，必須以新權重重排，這是非用 heap 不可的唯一理由）。**本日 silent failure 就藏在那個 heappush**：排序一次、從左往右合併、永不重排 → 合併節點永遠是最輕的兩個之一，整棵樹垮成單邊梯子，最長編碼 6 bits → 17 bits，總共 316 bits（比完全不壓縮的 265 還大 19%）——但依然 prefix-free、依然一字不差還原、Kraft sum 依然剛好 1.000，連「編碼樹有沒有長出通往空處的分枝」這個結構檢查都攔不住。硬幣 1/3/4 付 6 給 [4,1,1] 三枚而不是兩枚 3，而 1/5/10/25 永遠正確，正是這個陷阱活這麼久的原因。LC 435（`s >= last_end`，端點相接不算重疊）對上 LC 452（氣球是閉區間，必須 `s > last`，把 435 原封不動搬過去就翻車，是最常見的追問翻車點）；LC 621 用 `max((peak-1)*(n+1)+tied, len(tasks))` 取代模擬（能這樣寫是因為字母只有 26 種）；LC 1046 是 Huffman 迴圈的倒影（取最重兩顆、推回差值，heapq 只有 min-heap 所以負號進去）；LC 134 是本日唯一不用交換論證的證明（tank 轉負則從 start 到 i 整段都不可能當起點，直接跳到 i+1，一趟掃完）。貪心不會大聲壞掉，它是以百分比壞掉——上線就把暴力解一起上線並在小輸入 assert 兩邊相等 → Day 51 BPE 是同一個頻率驅動合併迴圈的反面（合併最頻繁的相鄰對，而不是最罕見的兩個符號） |
| 39 | 回溯與剪枝：N-Queens、數獨 ✅ | **本日核心：剪枝決定跑不跑得完，unchoose 決定對不對——回溯的 bug 不會 crash**。模板三行 choose → explore → unchoose，整個搜尋共用一個 `path`；**本日 silent failure #1：少了 copy**——`res.append(path)` 少了 `[:]` 回傳 8 個 `[]`（長度對、只有 1 個 list 物件，檢查 len 的測試照過）。**N-Queens** 一列一顆、三個 set（欄、r−c、r+c）：8 皇后 C(64,8)=4,426,165,368 → 8^8=16,777,216 → 8!=40,320 → 回溯只走 2,057 個節點（含根）找齊 92 組；公平對照是不剪斜線的欄排列樹 109,601 節點（53 倍）；4×4 整棵樹 17 節點、2 解、4 死路、44 格沒放就被淘汰。**#2：忘了 unchoose**——留 `board.pop()`、刪三個 `set.remove()`，幽靈皇后繼續佔欄佔斜線，9 個節點後宣布 8 皇后 **0 組解**；**剪枝必須健全：鏡像捷徑**（第一列只試左半再 ×2）偶數全對、有解的奇數全錯（5: 8≠10、7: 34≠40、9: 298≠352；n=3 對只因沒解可漏），中間欄是自己的鏡像、沒搜也沒算。Day 37 bitmask 版（`free & -free`、斜線每列位移一格）n=10 同樣 35,539 節點、快約 4 倍——檢查變快改的是每節點成本不是節點數。**數獨：挑哪一格分支比檢查多快重要**——MRV（合法數字最少的空格先分支，0 個就當場判死）：LC 37 範例 4,209 → 52（81×，每空格一個節點零猜錯）、Inkala 49,559 → 15,573（3×）、17 提示 anti-brute（第一列解是 987654321，左上角 7 個候選答案排最後）69,175,317 → 17,918（3,861×，約兩分鐘 → 四分之一秒，`--full` 重跑驗證）；老實話：MRV 每節點要掃全部空格，Inkala 上反而比閱讀順序 + bitmask 慢約一倍。LC 78（copy 就是全部技巧）；LC 39 start index 管正確（拿掉變 4 組含 [2,3,2]、[3,2,2]）、sort+break 管效率（[2,3,5,7,11]→30：1,239 → 575 節點）；LC 51/52；LC 37（MRV + bitmask 原地填）；**LC 79 拿掉還原 `#` 那行，LeetCode 三個範例全過**，但 "CCB" 回 False，1,108 個真拼得出的字串漏掉 201 個——範例全過不代表 unchoose 有寫 → Day 88 MCTS（同一棵走法樹、同一個 apply/undo 迴圈，改成抽樣而非窮舉）；SAT/約束求解的 unit propagation 就是 MRV 的「只剩一個選項直接填」 |

## Part 6 — 數學與工程基本功（Day 40–43）

| Day | 主題 | 備註 |
|---|---|---|
| 40 | 位元運算技巧：mask、popcount ✅ | **本日核心：Python 的 int 沒有寬度——逐欄的 `& \| ^ ~` 永遠安全，任何「假設有 32 位」的寫法都會安靜地壞掉，解法永遠是 `& 0xFFFFFFFF` 釘寬度、再把第 31 位讀成符號（s32）**。工具箱在 180=0b10110100 上：測第 2 位 1、設第 3 位 188、清第 4 位 164、翻第 7 位 52、`x & -x` 4、`x & (x-1)` 176、`x \| (x+1)` 181、Gray code 238；**優先序陷阱** `1 << n - 1` 是 8 不是 15（算術比位移優先；但比較比 `&` 低，所以 C 的 `x & 1 == 1` 陷阱在 Python 不存在）。**popcount**：20 萬個 32-bit 數，`bit_count()` 最快，`bin().count` 約 3×、SWAR 約 3×、Kernighan 約 11×、逐位位移約 27× 慢；Kernighan 圈數 3,199,046 vs 位移 6,200,129（1 的個數 vs 最高位）；百萬位整數 bit_count 比 bin().count 快約 70×。**本日 silent failure #1：負數的 popcount**——C 說 -5 有 31 個 1，`bin(-5).count("1")` 與 `(-5).bit_count()` 都回 2（數的是絕對值），Kernighan 在 -5 上永遠不停（第 40 圈 x = -2,199,023,255,552）。**整數當集合**：兩個 10 萬成員集合（0..999,999）交集 9,955，`(a & b).bit_count()` 比 set 快約 60×、記憶體 4,194,520 → 133,360 B（31×）；但逐個 `x \|= 1 << v` 每次都複製整個整數，要先用 bytearray 建（快 72×）。Gosper’s hack 直接列 k-子集：n=20,k=3 走 1,140 個而非過濾 1,048,576。LC 338 `dp[i] = dp[i >> 1] + (i & 1)`；LC 136 XOR 兩兩抵消；**#2：LC 137 逐欄計數 mod 3 忘了 s32**——[-2,-2,-3,-2] 回 4,294,967,293，兩個範例全過，1,000 組隨機測資錯 502 組＝剛好答案為負的那些（ones/twos 狀態機天生沒這問題）；**#3：LC 371 不用 + 算加法**——Python 裡 (-1)+1 的進位永遠往左走（第 40 圈 carry = 1,099,511,627,776），加上 mask 32 圈後掉出第 31 位得 0，但 (-1)+(-1) 沒讀符號回 4,294,967,294；LC 191；**LC 231 少了 `n > 0`，三個範例全過但 n=0 回 True**。→ Day 27 Jaccard／Day 32 Bloom filter 的 bit 陣列就是今天的整數集合；Day 34 `bits \|= bits << x` 子集和；Day 37/39 `free & -free`；Fenwick tree 的 lowbit 就是 `i & -i`；量化的 bit packing、hash 的 mix 函式都是同一套遮罩與位移 |
| 41 | 質數篩、快速冪、GCD、模運算 ✅ | **本日核心：`%` 保得住加、減、乘，保不住除——今天每一個陷阱都是這句話的後果**（`(12 // 4) % 5 = 3` 但 `(12 % 5) // (4 % 5) = 0`；Python 的 `-7 % 3 = 2` 跟著 m 的正負號，C/Java 給 -1）。**快速冪 square-and-multiply**：乘法次數 =（位元長 − 1）+（1 的個數 − 1），3^13 mod 100 = 23 只要 5 次（樸素 12）；e = 10^18 只要 82 次；`3**1_000_000 % MOD` 先造出 158 萬位元的大數約 66 ms，`pow(3, 1_000_000, MOD)` 約 2 µs（數萬倍）；LC 50 myPow 負指數與 n = -2^31（31 次 vs 21 億次）；矩陣快速冪 F(10^18) mod 1e9+7 = 209,783,453 只要 83 次矩陣乘法。**GCD**：gcd(252, 105) 三步得 21，減法版 gcd(10^6, 1) 要 999,999 次；1000 以內最差 14 步在 gcd(987, 610)＝相鄰 Fibonacci。**extended Euclid** 每列維持 a = 240x + 46y，得 2 = 240·(-9) + 46·47（Bézout）；LC 365 水壺＝target 是 gcd(x, y) 的倍數**且 target ≤ x + y**，少了後者 (3, 5, 9) 錯回 True，BFS 狀態搜尋四組全吻合。**本日 silent failure #1：合數模數上用 Fermat 求反元素**——m = 10 時 pow(3, 8, 10) = 1，3·1 % 10 = 3 不是 1；m = 10^6 的 400,000 個可逆 a 錯 399,999 個，1e9+7（質數）錯 0；`pow(a, -1, m)` 在不可逆時會丟例外，`pow(4, 8, 10) = 6` 只會安靜地回答。**#2：nCk 對取過餘的階乘做 `//`**——n ≤ 12 全對（12! < 1e9+7 < 13!），第一個錯是 C(13, 1) 回 0；n ≤ 30 的 496 個答案錯 369 個，C(100, 50) 回 0（正解 538,992,043）；改乘逆階乘表 → C(10^6, 500,000) = 996,692,777，1,000 次查詢查表不到 1 ms、`math.comb % p` 約 1.5 s。**埃拉托斯特尼篩法**：從 p² 開始劃（n = 10^6 劃 2,122,046 次，從 2p 開始 2,197,837，用所有數去劃 2,775,208）；一百萬以下 78,498 個質數，試除法約 4–5 s、迴圈篩約 0.2 s、bytearray 切片約 10 ms；一千萬以下 664,579 個約 0.3 s。**#3：LC 204 數的是「小於 n」**——寫成「≤ n」三個範例（10、0、1）全過，n = 0..5000 錯 669 個＝剛好那 669 個質數本身。LC 1922 好數字（5·4 交錯，n = 10^15 → 711,414,395）；**#4：LC 372 把指數 mod 1336**——2,000 組隨機錯 961 組，因為 1337 = 7 × 191 不是質數。→ Day 32 Bloom filter／Day 12 hash 的 mod 質數桶數；RSA 與 Diffie–Hellman 就是快速冪＋反元素；組合數 DP 的取模；rolling hash（Rabin–Karp）把整個字串當成 mod 質數下的多項式 |
| 42 | 隨機化演算法與蒙地卡羅 ✅ | **本日核心：隨機演算法的 bug 不會讓程式崩潰，只會讓機率歪掉——所以要把「每一種可能的隨機序列」都重播一次、用精確分數（Fraction）算出分佈，1/27 的偏差才藏不進雜訊裡**（odometer 列舉 randrange 的每個選擇）。**蒙地卡羅估 π**：100 / 1 萬 / 100 萬支飛鏢誤差 0.0216 / 0.0164 / 0.00077，多次重跑的 RMS 誤差貼著 1.642/√N（N=100 實測 0.163 vs 0.164）——飛鏢多 100 倍誤差才小 10 倍，6 位小數要約 10.8 兆支；同 seed 同結果。**Las Vegas vs Monte Carlo**：已排序的 2,000 筆用第一個當 pivot 1,999,000 次比較，隨機 pivot 200 次平均 24,795（理論 24,730）——答案永遠對、只有時間是隨機的；**Miller-Rabin**：Carmichael 數 561 = 3·11·17，Fermat 在 558 個底數中 318 個（57%）說謊，Miller-Rabin 只剩 8 個（1.4%，上界 1/4）；底數 2 的鏈 2^35 → 263 → 166 → 67 → 1，67 是 1 的非平凡平方根所以抓到；10 萬以下 78 個奇合數騙過底數 2 的 Fermat，單一隨機底數 MR 錯 38 次、5 個底數 0 次；2^61−1 用 12 個固定底數判定。**洗牌 silent failure #1：naive「跟任意位置交換」**——3 個元素 27 種執行落到 6 種排列，27 不是 6 的倍數所以不可能平均（acb/bac/bca 各 5/27、其餘 4/27）；4 個元素 256 / 24 最常見是最少見的 1.88 倍；6 萬次實洗 abc 8,927 vs bac 11,235。**#2：Fisher-Yates 寫成 `randrange(i)`**＝Sattolo 演算法，只產生循環排列（3 個只到 bca、cab；4 個只到 24 種中的 6 種，沒有任何元素留在原位）；正確版 6 種各 1/6。LC 384。**蓄水池抽樣**：第 i 個以 1/i 換入，連乘相消剛好每個 1/5；k=2 時 10 對各 1/10；**#3：i 從 0 數**——第一個元素機率 0、其餘 1/4，LC 382 三萬次裡 1 從沒出現（2: 15,042、3: 14,958）；LC 398 pick 是蓄水池的變形。**拒絕抽樣 LC 470 rand10**：兩次 rand7 得 49 格，只收前 40 格，期望呼叫 2·49/40 = 2.45（實測 2.456，回收被拒的餘數 2.193）；**#4：不拒絕直接 `% 10 + 1`**——數字 1 只有 4/49 其餘 5/49；10 萬次 min 9,882 / max 10,254。**LC 528 加權隨機**：prefix sums + `bisect_right(prefix, randrange(total))`；**#5：`randint(0, total)` 配 `bisect_left`**——多一個值又用錯 bisect，w=[1,3] 時 index 0 變 2/5（該 1/4），w=[1,99] 變 2/101（該 1/100）。LLM 取樣下一個 token 就是同一步驟套在 softmax(logits/T) 上：logits 2.0/1.0/0.1 在 T=0.5 → .864/.117/.019、T=1 → .659/.242/.099、T=2 → .502/.304/.194，T→0 即 greedy decoding。→ Day 20 quicksort 的隨機 pivot、Day 32 Bloom filter／Day 27 MinHash 的隨機雜湊、Day 41 的快速冪與模運算是 Miller-Rabin 的引擎；RL 的 policy 取樣、ε-greedy、Monte Carlo return、MCTS rollout、LLM 的 temperature / top-p 取樣、dropout 與 SGD 的 mini-batch 洗牌都是今天這一套 |
| 43 | 並行與平行模型：GIL、multiprocessing、asyncio、work-depth、map-reduce ✅ | **本日核心：CPython 的 thread 重疊的是「等待」不是「計算」——選工具只問「在算還是在等」；值不值得平行則看 work W 與 depth D（max(W/p, D) ≤ T_p ≤ W/p + D）**。**GIL**（雙核心 VM，計時都是約值）：兩個 CPU 工作 thread 約 0.24–0.37 s ≈ 循序、process 快約 1.3–2 倍；20 次 50 ms 等待迴圈約 1.0 s、20 個 thread 約 0.058 s；Pool.map chunksize=1 平方 5 萬個數要好幾秒（比迴圈慢上千倍）。**silent failure #1：race**——CPython 3.10 裡 4 thread × 200,000 次 `box.v += 1` 每次都剛好 800,000（只在迴圈跳轉與函式呼叫處切換，是版本巧合不是保證），讀寫之間加一個函式呼叫就掉 29%–64%；**交錯檢查器**（Day 42「重播每種亂數選擇」的同一招，按狀態 memo）：2/3/4 thread 各一次 x += 1 → 6/90/2,520 種交錯只有 2/6/24 種對（＝thread 依序跑完的那些），加 lock 只剩 2 種且全對。**#2：deadlock**——兩把 lock 反序 6 種交錯 2 種卡死；**LC 1226** 五位哲學家先拿左叉：9,302 個狀態只有 1 個卡死、2,430,842,120 種交錯裡 120 種走到（約兩千萬分之一，測試碰不到），真 thread 用 Barrier 逼出時五人全卡；先拿編號小的叉子 → 8,447 狀態 0 卡死。**LC 1114** 不同步 6 種輸出、兩個 Event 只剩 1 種。**#3：asyncio 裡的 time.sleep**——10 × 0.1 s：await 約 0.10 s、time.sleep 約 1.00 s 且無任何警告、Semaphore(3) 約 0.40 s。**LC 1242** 多執行緒爬蟲：16 頁依序 16 次抓取約 0.80 s，分波 5 波約 0.26 s（16 是 work、5 是 depth）；`host in url` 讓 news.yahoo.com.evil.example 混進來（17 頁）。**work-depth**：8 數迴圈 W7 D7、樹 W7 D3；1024 數 scan：迴圈 1023/1023、Hillis-Steele W9,217 D10、Blelloch W2,036 D18（Day 22 記 2,046/20 是含加 0 的那 10 次）；4 worker 時 HS 2,305 步比迴圈 1,023 還慢，HS 要到 545 個 worker 才贏 Blelloch；Amdahl 5% 循序 → 64 核 15.4 倍、上限 20 倍。**map-reduce**：40 文件 16,000 字 2 process 4 reducer ＝ Counter；**#4：mapper 分桶不一致**——67 字錯 51 個（of 821 vs 1,640）；`hash(word) % R` 在 spawn（Windows/macOS 預設）worker 實際錯約 50 個、fork 0 個（Day 27 hash 隨機化同一課）；**#5：平均的平均** 200 vs 真值 130 → reduce (sum, count)；十個 0.1 迴圈 0.9999999999999999、樹狀 1.0。→ Day 20 平行 merge sort（depth 2,046→235 換 84% work）／Day 22 scan／Day 32 HLL 合併；continuous batching 的 event loop、向量化環境與 fully async RL rollout、agent 等工具、資料平行訓練的 all-reduce（浮點順序不同 → 不同 GPU 數結果不逐 bit 相同） |

---

## Part 7 — 機器學習底層（Day 44–50）
刻意壓縮：經典 ML 用兩天打包，把篇幅留給 LLM 與 RL。

| Day | 主題 | 備註 |
|---|---|---|
| 44 | numpy 向量化與自動微分直覺 ✅ | **本日核心：兩半是同一課——把迴圈從直譯器搬進一次處理整個陣列的呼叫；而陣列與 autograd 的 bug 都不會報錯，只會給一個錯的數字**。**向量化**：一百萬個數平方和，for 迴圈約 41 ms、(a*a).sum() 約 1.2 ms、np.dot 約 0.1 ms（一次量測，數十到數百倍）；陣列＝緩衝區＋shape＋strides，`arange(12).reshape(3,4)` strides (32, 8)、`.T` (8, 32)、`[:, ::2]` (32, 16) 都是 view，`m.T.reshape(12)` 只能複製。**silent failure #1：broadcasting**——pred (5,1) − y (5,) 得 (5,5)，MSE 4.0100 而非 0.0100；**#2：uint8 繞回** 200+100=44、(a+b)//2 得 22（應 150，先轉 int16）；**#3：fancy index 是複製** `v[[0,2]][0]=99` 什麼都沒改。兩兩距離 300 點：迴圈約 205 ms、broadcasting 約 4.7 ms 但 (n,n,3) 暫存在 2 萬點時 9,600 MB、Gram 恆等式約 0.63 ms；**#4：Gram 不夾 0** 500 點有 63 個距離平方 < 0 → 63 個 NaN。**求導三法**：sin 在 1 的 central difference 誤差在 h=1e-5 最小（1.1e-11），h=1e-12 又爛回 1.2e-5（V 字形：截斷 vs 捨入）；dual number（forward mode）精確但 200 個輸入要 200 趟；reverse mode 1 趟 forward + 1 趟 backward。**mini autograd**（micrograd 式 Value，backward＝Day 13 拓樸順序的迭代 DFS）：L=(a*b+c)*f 在 a=2,b=−3,c=10,f=−2 → L=−8、dL/da=6、dL/db=−4、dL/dc=−2、dL/df=4；gradcheck 相對誤差 1.3e-10。**#5：`grad =` 取代 `+=`**——x*x 在 3 得 3（應 6）；**#6：不照拓樸順序的天真遞迴**——(2x)² 在 3 得 48（應 24），7 次走訪而非 4；**#7：不 zero_grad**——擬合 y=3x−0.5 60 步：歸零 w=3.020、loss 0.0083，不歸零 w=6.089、loss 3.0924（約 370 倍，一直擺盪）。純量圖 vs 矩陣公式：200×5 最小平方 4,408 個節點約 16 ms，`2/n Xᵀ(Xw−y)` 約 0.005 ms，差 5.3e-15。**LC 566** 第 k 個數 [k//n][k%n] → [k//c][k%c]（＝strides 算術），order="F" 給 [[1,5],[4,3],[2,6]]；**LC 150** 逆波蘭（Day 02 stack），**#8：`//` 往下取整** 官方長範例 22 變 12（6 // −132 = −1）；同一台 stack 機器換成 Value 就是 autograd 的 tape。→ 梯度下降家族（zero_grad＋step）、線性／邏輯迴歸（同一條 Xᵀ 公式）、numpy MLP 的反向傳播（三個 bug 都在，加上 loss 的 broadcasting）、PyTorch autograd tape／JAX 函式轉換、GPU 上一個 bias 加到整個 batch |
| 45 | 梯度下降家族：SGD、momentum、Adam ✅ | **本日核心：學習率在二次碗上有一個精確上限 2/最大曲率，越過它不會丟例外、loss 只是長到 inf 再變 nan；momentum 把步數對條件數的依賴從 kappa 變成 sqrt(kappa)；而 Adam 的每個細節寫錯都照樣回傳數字**。接 Day 44 的 zero_grad：同一個訓練迴圈只換掉 `w -= lr * grad` 那一行。**精確上限**：f = 0.5(w0² + 20w1²)，純 GD 每步把每座標乘 (1 − lr × 曲率)，故 lr < 2/20 = 0.1；0.0999 收斂、0.1001 發散，lr 0.11 跑 400 步後 w = [5.7e-21, 4.7e+31]（一個座標完美、一個飛走）。**momentum**：最佳 lr 0.0952 時速率 (k−1)/(k+1) = 0.9048、理論 138 步**實測 139**；heavy-ball 最佳 lr 0.1336 + beta 0.4026 速率 0.6345、理論 30 步實測 39（差距＝漸近速率忽略的暫態），仍少 3.6 倍。**#1：beta = 0.9 不是常數**——固定 lr 0.04 時 beta 0/0.4026/0.9 分別 338/190/245 步，抄來的 0.9 比調過的差。**真問題**（400×3 最小平方、一欄放大 8 倍、條件數 77，各自掃 lr）：SGD 400、momentum 181、Nesterov 163、AdaGrad 10、RMSProp 26、Adam 153 步；AdaGrad 贏是因為問題固定不變，同一個不斷累加的分母正是它在長訓練會卡死的原因。**#2：Adam 少了 bias correction 步子太「大」不是太小**——同一組 m、v 比較，未修正÷修正 = (1−0.9^t)/√(1−0.999^t)：t=1 是 3.16 倍、t=12 最糟 6.57 倍、t=1000 還有 1.26 倍（lr 0.05 實測前三步 0.274/0.363/0.404 對 0.087/0.086/0.086）。**#3：eps 放進根號**——eps=1e-8 的根號是 1e-4，v=1e-12 時兩個分母差 99 倍、v=1e-16 差 5000 倍，只在小梯度座標發作（PyTorch 用 √v + eps）。**#4：Adam + L2 ≠ AdamW**——L2 把 wd·w 塞進梯度後會被 √v 除掉，三個權重 √v = [0.41, 0.15, 4.06]（量自無衰減的 Adam，加懲罰項後最多差 2.5%），衰減強度（平均正規化為 1）變成 [0.77, 2.15, 0.08] 對 AdamW 的 [1, 1, 1]：梯度最大的權重被衰減少 10 倍。**#5：batch 越小 lr 上限越低**——全批上限 0.0140，最陡單列只有 0.00165（8.5 倍），lr 0.01 時 batch 1 爆掉而全批沒事；lr 0.002 下 batch 8 在 30 epoch 內多走三個數量級，然後停在 2.5e-3 的雜訊地板（全批繼續降）。**#6：shuffle 的真正用途**——資料本來就隨機時重洗量不出差別（1.6e-3 對 1.5e-3），依目標值排序過才差 12 倍。**LC 462** sum\|x−c\| 的最小值在中位數（導數是正負號＝下面幾個減上面幾個）；**平均數是最自然的錯答**，[1,2,3] 與 [1,10,2,9] 上兩者同值會過測，[1,2,3,100] 才分出 100 vs 146。對它跑梯度下降停在 2.354（偶數個時最小值點是區間），1/√k 排程是必要的——正負號梯度在最佳解附近不變小。**LC 852** arr[mid] < arr[mid+1] 就是斜率正負號 → 排序好的布林陣列，Day 23 二分搜尋；下坡時必須 hi = mid 不是 mid−1；4,001 個元素只看 24 個。三分搜尋是連續版本。梯度下降存在的理由是維度：二分／三分只統治一維單峰。→ optimizer 狀態是模型的兩倍（7B fp32：權重 28 GB、Adam 動量 56 GB），分散式訓練切分 optimizer 狀態處理的就是這個；AdamW 是 transformer 訓練預設；warmup 處理的是跟 bias correction 同一件事；RL 實作把 Adam eps 設成 1e-5 而非 1e-8 |
| 46 | 反向傳播 + 純 numpy MLP（含 sigmoid / softmax / cross-entropy） ✅ | **本日核心：一個線性層的反向傳播只有三行，每一行都只是前向轉置一次——`dW = x.T @ dz`、`db = dz.sum(axis=0)`、`dx = dz @ W.T`；而每一種把它們寫得稍微不對的方式，都會交出一個能跑完的訓練迴圈**。全系列的分水嶺。**原 Day 46「線性迴歸與邏輯迴歸」已刪**，分類的 loss 併進來：線性那半的 `2/n Xᵀ(Xw − y)` 在 Day 44 推過了。**softmax 穩定性**：logits [1000, 1001, 1002] 天真版三個都 nan、穩定版 [0.090031, 0.244728, 0.665241]，而正常 logit 上兩者只差 1.1e-16（所以測試永遠看不到）；cross entropy 先算機率再取 log，在 [−400, 0, 400] 上 `exp(−800)` 下溢成 0 → `log(0)` = inf，`log_softmax` 給 800.0（有限且可用的梯度）；`log(sigmoid(−800))` 同理（−inf vs −800）。**silent failure #1：softmax 用錯軸**——`axis=0` 時每個元素照樣在 [0, 1]、欄和照樣是 1，只有列和變成 1.475 / 1.525，連每列 argmax 都沒變（[0,1] vs [0,1]）：先壞掉的是 loss 不是預測。**p − y**：softmax 的 Jacobian 與 log 的倒數合成時互相消掉，梯度就是 `(p − y) / n`，對 central difference 差 2.47e-11——這就是框架提供融合版 `cross_entropy(logits, labels)`、且禁止傳機率進去的原因。**MLP**：三臂螺旋 300 點，一層 64 個 ReLU 隱藏單元訓練集準確率 1.000，拿掉隱藏層（softmax regression）只有 0.530——隱藏層不是把邊界弄彎，是把**空間**弄彎，輸出層仍在畫直線。gradcheck 7.85e-09。**#2：`db = dz.sum()`** 得到純量而純量會 broadcast（gradcheck 1.00）；**#3：忘了除以 n** 讓每步大 n 倍，看起來只像 lr 太高（0.846）；**#4：`dW = x @ dz` 少了轉置**——本該是形狀錯誤，但 **batch 剛好等於輸入寬度時就跑得過去**（n = d_in = 8 時 gradcheck 1.00，改 batch size 才變成 ValueError）；**#5：全零初始化**——64 個隱藏單元收到同樣梯度、256 個 activation 只有 1 個相異值，400 步後 0.333（三類亂猜）對隨機初始化的 1.000。**一個不是 bug 的失敗**：ReLU 在 0 沒有導數，backprop 用 `mask = x > 0` 回報 0、central difference 兩邊平均回報 0.5；餵一列全零（bias 又從 0 開始）讓 8 個 pre-activation 有 4 個落在角上，**正確的實作 gradcheck 拿到 1.00**，同一個網路換 tanh 是 3.4e-10。（這是實際踩到的：第一版 spiral 每條臂從 r=0 出發，三類共用原點。教訓＝梯度檢查用平滑激活函數、檢查失敗先看資料。）**LC 867** `out[j][i] = matrix[i][j]`，非方陣所以不能原地；對照 numpy 的 `a.T` 只把 strides (24,8) 對調成 (8,24) 不搬資料（Day 44），這就是 `x.T @ dz` 不比 `x @ dz` 貴、反向成本約等於前向的原因。**LC 48** 原地旋轉＝轉置 + 每列反轉；**內層迴圈寫成 `range(n)` 會讓每一對交換兩次**，矩陣原封不動、轉置悄悄沒做，輸出變成「每列反轉」的輸入。→ `torch.nn.Linear.backward` 就是這三行；融合版 cross_entropy 來自 p − y 的消去；為反向留下的 activation 正是 gradient checkpointing 拿去交換再重算的東西；He / Xavier 初始化都長在「一樣的權重產生一樣的單元」的下游 |
| 47 | k-means 與 KNN | 兩個一起寫 |
| 48 | 決策樹 / 隨機森林 / Gradient Boosting | 三個一起寫 |
| 49 | PCA / SVD / t-SNE | 降維一次講完 |
| 50 | 評估與正則化：交叉驗證、overfitting、metrics | |

## Part 8 — LLM 內核（Day 51–64）
全部用 numpy / 少量 torch 從零實作。

| Day | 主題 | 備註 |
|---|---|---|
| 51 | BPE Tokenizer 從零實作 | 接 Day 38 Huffman |
| 52 | Attention 機制 | QKV、softmax 數值穩定 |
| 53 | Multi-Head Attention 與 Transformer Block | |
| 54 | 位置編碼：sinusoidal、RoPE、ALiBi | |
| 55 | 從零訓一個 char-level mini-GPT | 里程碑：能生成文字 |
| 56 | KV Cache | 接 Day 30 快取；prefill vs decode |
| 57 | RadixAttention：前綴共享的 KV cache | Day 24 radix tree + Day 30 LRU 驅逐；SGLang 的核心 |
| 58 | Paged Attention 與顯存管理 | 本質是分頁記憶體配置器 |
| 59 | 取樣策略：temperature、top-k、top-p | Day 71 bandit 的探索直覺 |
| 60 | Beam search 與 speculative decoding | |
| 61 | 量化：INT8 / FP8 / GPTQ 基礎 | 為 Day 89 低精度訓練鋪路 |
| 62 | MoE 與路由（top-k gating、load balance） | 為 Day 88 routing replay 鋪路 |
| 63 | FlashAttention 概念：tiling 與 online softmax | |
| 64 | Continuous batching 與推論排程器 | 接 Day 43 並行；Day 85 rollout engine 的地基 |

## Part 9 — 檢索、RAG 與記憶層（Day 65–70）

| Day | 主題 | 備註 |
|---|---|---|
| 65 | Embedding 與向量相似度檢索 | word2vec、cosine、暴力搜尋當 baseline |
| 66 | HNSW 近似最近鄰 | 接 Day 31 Skip List |
| 67 | IVF 與 Product Quantization | |
| 68 | BM25、hybrid search 與 rerank | 接 Day 25 rolling hash |
| 69 | Chunking 策略與 minimal RAG pipeline | 端到端可跑 |
| 70 | 記憶層：短期 / 長期 / episodic memory | Day 97 專案的地基 |

## Part 10 — 強化學習：演算法全覆蓋（Day 71–88）
這 18 天的選材直接照著幾套主流框架真正實作了什麼來排：覆蓋度參照 **Tianshou**，
每天的程式碼寫成 **CleanRL** 那種「一支檔案從 rollout 到 optimizer.step() 看得完」的風格，
架構名詞對照 **TorchRL** 的 primitives（TensorDict / collector / replay buffer / loss module），
數字跟 **Stable-Baselines3** 對照當 sanity check，分散式的部分看 **RLlib**（Ray actor-learner）。

| Day | 主題 | 備註 |
|---|---|---|
| 71 | 環境介面：Gymnasium API、wrapper、向量化環境與 collector | reset/step/terminated vs truncated；接 Day 43 並行，向量化就是 batch 化的 rollout |
| 72 | Multi-Armed Bandit：ε-greedy、UCB、Thompson | 接 Day 42 隨機化；對照 Day 59 取樣溫度 |
| 73 | MDP：Value Iteration 與 Policy Iteration | GridWorld；接 Part 5 的 DP |
| 74 | Monte Carlo 與 TD Learning：n-step、TD(λ) | bias-variance 的第一次現身 |
| 75 | Q-Learning 與 SARSA（表格式） | on-policy vs off-policy 的分水嶺 |
| 76 | DQN 家族：replay buffer、target network、Double、Dueling、PER | replay buffer 用 Day 21 的取樣結構，PER 用 Day 29 的 segment tree |
| 77 | Distributional RL：C51、QR-DQN、IQN → Rainbow | 學分布不只學期望值；分位數回歸 |
| 78 | REINFORCE 與 policy gradient | log-prob 技巧、baseline 降變異 |
| 79 | A2C 與 GAE | advantage estimation 的 λ 取捨 |
| 80 | NPG 與 TRPO：自然梯度與信賴域 | Fisher 矩陣、KL 約束，PPO 的前身 |
| 81 | PPO 與那些真正決定成敗的實作細節 | clipping、優勢正規化、value clipping、observation 正規化 |
| 82 | 連續控制：DDPG 與 TD3 | 決定性策略、twin critic、target policy smoothing、延遲更新 |
| 83 | SAC：最大熵 RL | 自動溫度調整；REDQ / CrossQ 的 sample efficiency 觀點 |
| 84 | 稀疏獎勵與探索：HER、ICM、RND | 事後重標記與好奇心；對照 Day 90 的可驗證獎勵 |
| 85 | Offline RL：BCQ、CQL、IQL、TD3+BC | distribution shift、保守價值估計；接 Day 75 off-policy |
| 86 | 模仿學習：BC、DAgger、GAIL | GAIL 的判別器就是學出來的 reward，直通 Day 89 reward model |
| 87 | Multi-Agent：self-play、MAPPO、QMIX | 值分解與 CTDE；呼應 Day 98 多代理協作 |
| 88 | Model-based 與規劃：Dyna、PSRL、MCTS、世界模型（Dreamer 概念） | 接 Day 39 回溯 + Day 19 A* + Day 72 UCB；MCTS 實作井字棋 / 連四 |

## Part 11 — LLM 後訓練與 RL 基礎設施（Day 89–92）
把前面 18 天的演算法搬到語言模型上，順便看清「大規模 RL 訓練框架長什麼樣」。
實作參照 [Miles](https://github.com/radixark/miles)（SGLang rollout + Megatron trainer，從 slime fork）與 SGLang 本身。

| Day | 主題 | 備註 |
|---|---|---|
| 89 | RLHF 全景：reward model 從零訓 + DPO / SimPO | Bradley-Terry loss；把 Day 55 的 mini-GPT 拿來 align |
| 90 | GRPO / GSPO / REINFORCE++ 與 RLVR | group-relative advantage：為什麼 LLM RL 丟掉 critic；可驗證獎勵與 reward hacking |
| 91 | RL 訓練框架的骨架：rollout engine × trainer | 接 Day 64 continuous batching：推論引擎當環境、TITO、權重同步、fully async 與 staleness |
| 92 | 大規模的魔鬼細節 | MoE routing replay（接 Day 62）、FP8 / MXFP8 / NVFP4 低精度穩定性（接 Day 61）、容錯與 observability |

## Part 12 — 無梯度優化（Day 93–94）

| Day | 主題 | 備註 |
|---|---|---|
| 93 | 基因演算法：編碼、選擇、交配、突變 | 用 GA 解 Day 37 的狀壓 TSP，跟精確解對照（答案已知，可以量「離最佳解多遠」）。tournament / roulette 選擇、菁英保留、突變率與多樣性。無聲失敗：**排列編碼用單點交配會生出重複或缺城市的路徑**，看起來仍是一條「合法的」路線，要改 OX / PMX；fitness-proportionate 選擇在某個個體獨大時會提早收斂；沒有菁英保留會把目前最好的解弄丟 |
| 94 | 其他無梯度方法：模擬退火、PSO、CMA-ES | 退火的接受機率與降溫排程（降太快＝貪心、降太慢＝隨機漫步）、粒子群的慣性／認知／社會三項、CMA-ES 用共變異數矩陣學出搜尋方向。收尾：演化 vs 梯度，對照 Day 78 policy gradient——OpenAI ES 把整個 RL 當黑箱優化，用的就是這一套 |

## Part 13 — 中型專案：LLM Agent Harness（Day 95–100）
前面 94 天的零件，這裡組成一個能跑的東西。

| Day | 主題 | 備註 |
|---|---|---|
| 95 | Agent Harness I：tool calling 與 JSON action loop | 工具註冊表、schema 驗證、重試 |
| 96 | Agent Harness II：接上記憶層 | 用 Day 70 + Day 66 HNSW |
| 97 | Agent Harness III：ReAct / planner 與多步推理 | 錯誤恢復、預算控制 |
| 98 | Multi-Agent 協作與任務 DAG 排程 | 呼應 Day 13 拓樸排序 + Day 87 MARL |
| 99 | Eval Harness：LLM-as-judge、回歸測試、追蹤 | 也是 Day 90 verifier 的延伸 |
| 100 | 收官：100 天知識地圖 + 效能剖析 + 打包開源 | |

---

## 依賴關係（有意設計的伏筆）

- Day 05 heap → Day 16 Prim / Dijkstra、Day 21 heap sort、Day 66 HNSW 的候選集
- Day 13 拓樸排序 → Day 98 多代理任務 DAG
- Day 14 Union-Find → Day 15 Kruskal
- Day 22 prefix-sum (scan) → Day 43 平行模型 → Day 91 async rollout 排程
- Day 24 Trie / Radix Tree → Day 57 RadixAttention
- Day 29 Segment Tree → Day 76 prioritized replay buffer
- Day 30 LRU → Day 56 KV cache → Day 57 RadixAttention（radix tree + LRU 驅逐）→ Day 58 paged attention
- Day 31 Skip List → Day 66 HNSW
- Day 37 狀壓 TSP → Day 93 GA 解 TSP（精確 vs 啟發式對照）
- Day 38 Huffman → Day 51 BPE
- Day 39 回溯 → Day 88 MCTS
- Day 42 隨機化 → Day 72 bandit → Day 59 取樣策略（探索與溫度是同一件事）
- Day 43 asyncio / work-depth → Day 64 continuous batching → Day 71 向量化環境 → Day 91 fully async RL → Day 95 agent loop
- Day 55 mini-GPT → Day 89 RLHF / DPO → Day 90 GRPO → Day 95–100 專案
- Day 61 量化 → Day 92 低精度 RL 訓練
- Day 62 MoE 路由 → Day 92 rollout routing replay
- Day 64 推論排程器 → Day 91 rollout engine（同一個引擎，一個對外服務、一個當 RL 環境）
- Day 75 off-policy → Day 85 offline RL → Day 91 async 的 staleness 容忍度
- Day 79 GAE / Day 81 PPO → Day 90 GRPO（同一個 advantage，換成 group 內相對比較）
- Day 86 GAIL 判別器 → Day 89 reward model（都是學出來的獎勵）
- Day 90 verifier → Day 99 eval harness
