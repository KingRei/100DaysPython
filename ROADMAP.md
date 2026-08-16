# 100 Days of Python — Roadmap

已完成 Day 01–11（linked list → graph）。以下為 Day 12–100 的規劃。
原則：先把經典資料結構／演算法走完，中段轉進 AI 時代真正天天在用的東西，
最後 6 天做中型專案，把前面 94 天的積木組起來。

---

## Part 1 — 圖論（Day 12–20）
接續 Day 11 的圖表示法，把圖演算法一次走完。

| Day | 主題 | 配套題 / 備註 |
|---|---|---|
| 12 | BFS / DFS 走訪與應用 | LC 200 島嶼數量 |
| 13 | 拓樸排序（Kahn + DFS 兩種寫法） | LC 207 課程表；DAG、循環偵測 |
| 14 | Union-Find（路徑壓縮 + 按秩合併） | LC 547 省份數量 |
| 15 | 最小生成樹 — Kruskal | 接 Day 14 的 Union-Find |
| 16 | 最小生成樹 — Prim | 接 Day 05 的 heap |
| 17 | 最短路 — Dijkstra | LC 743 網路延遲 |
| 18 | 最短路 — Bellman-Ford / Floyd-Warshall | 負權、全點對 |
| 19 | Tarjan：強連通分量、橋、關節點 | |
| 20 | A* 啟發式搜尋 | 第一次碰 heuristic，AI 的橋樑 |

## Part 2 — 排序與搜尋（Day 21–26）

| Day | 主題 | 備註 |
|---|---|---|
| 21 | Quick sort 與 partition | 三路劃分、最壞情況 |
| 22 | Merge sort 與外部排序 | 大檔案排序 |
| 23 | Heap sort 與 Top-K | 接 Day 05 |
| 24 | 非比較排序：counting / radix / bucket | O(n) 的條件 |
| 25 | 二分搜尋的三種邊界寫法 | 「在答案上二分」 |
| 26 | Quickselect、中位數、Reservoir sampling | 串流取樣 |

## Part 3 — 字串演算法（Day 27–32）

| Day | 主題 | 備註 |
|---|---|---|
| 27 | Trie（前綴樹） | 自動完成 |
| 28 | Radix / Patricia Tree（壓縮 trie） | IP 路由表；Day 67 RadixAttention 的地基 |
| 29 | 字串匹配：KMP + Rabin-Karp | failure function、rolling hash |
| 30 | Suffix Array + LCP | |
| 31 | Aho-Corasick 多模式匹配 | 敏感詞過濾 |
| 32 | 編輯距離與相似度（Levenshtein、Jaccard、MinHash） | 為後面的檢索鋪路 |

## Part 4 — 進階樹與區間 / 機率型結構（Day 33–40）

| Day | 主題 | 備註 |
|---|---|---|
| 33 | AVL 樹 | 接 Day 04 |
| 34 | Red-Black Tree 與 B-Tree | 資料庫索引 |
| 35 | Segment Tree | 區間查詢 + lazy propagation |
| 36 | Fenwick Tree (BIT) | |
| 37 | LRU / LFU Cache | LC 146；為 Day 67 KV cache 鋪路 |
| 38 | Skip List | Redis 的 sorted set |
| 39 | Bloom Filter | |
| 40 | HyperLogLog 與 Count-Min Sketch | 串流估計 |

## Part 5 — 動態規劃、貪心與回溯（Day 41–48）

| Day | 主題 | 備註 |
|---|---|---|
| 41 | DP 入門：爬樓梯、硬幣找零 | 回顧 Day 08 word break |
| 42 | 背包問題（0/1、完全、多重） | |
| 43 | LIS 與 LCS | |
| 44 | 區間 DP（矩陣鏈乘、戳氣球） | |
| 45 | 樹形 DP | |
| 46 | 狀態壓縮 DP（TSP） | 對照 Day 92 用 GA 解 TSP |
| 47 | 貪心：區間排程、Huffman 編碼 | |
| 48 | 回溯與剪枝：N-Queens、數獨 | 為 Day 91 MCTS 鋪路 |

## Part 6 — 數學與工程基本功（Day 49–52）

| Day | 主題 | 備註 |
|---|---|---|
| 49 | 位元運算技巧 | mask、popcount |
| 50 | 質數篩、快速冪、GCD、模運算 | |
| 51 | 隨機化演算法與蒙地卡羅 | 為 RL 鋪路 |
| 52 | Python 並行：GIL、multiprocessing、asyncio | 為 Day 74 / 98 鋪路 |

---

## Part 7 — 機器學習底層（Day 53–60）
刻意壓縮：經典 ML 用兩天打包，把篇幅留給 LLM 與 RL。

| Day | 主題 | 備註 |
|---|---|---|
| 53 | numpy 向量化與自動微分直覺 | 手刻 mini autograd |
| 54 | 梯度下降家族：SGD、momentum、Adam | |
| 55 | 線性迴歸與邏輯迴歸 | 從零推導 |
| 56 | 反向傳播 + 純 numpy MLP | 全系列的分水嶺 |
| 57 | k-means 與 KNN | 兩個一起寫 |
| 58 | 決策樹 / 隨機森林 / Gradient Boosting | 三個一起寫 |
| 59 | PCA / SVD / t-SNE | 降維一次講完 |
| 60 | 評估與正則化：交叉驗證、overfitting、metrics | |

## Part 8 — LLM 內核（Day 61–74）
這一段是整個系列最有價值的部分，全部用 numpy / 少量 torch 從零實作。

| Day | 主題 | 備註 |
|---|---|---|
| 61 | BPE Tokenizer 從零實作 | 接 Day 47 Huffman 的編碼直覺 |
| 62 | Attention 機制 | QKV、softmax 數值穩定 |
| 63 | Multi-Head Attention 與 Transformer Block | |
| 64 | 位置編碼：sinusoidal、RoPE、ALiBi | |
| 65 | 從零訓一個 char-level mini-GPT | 里程碑：能生成文字 |
| 66 | KV Cache | 接 Day 37 快取；prefill vs decode |
| 67 | RadixAttention：前綴共享的 KV cache | Day 28 radix tree + Day 37 LRU 驅逐；SGLang 的核心 |
| 68 | Paged Attention 與顯存管理 | 本質是分頁記憶體配置器 |
| 69 | 取樣策略：temperature、top-k、top-p | |
| 70 | Beam search 與 speculative decoding | |
| 71 | 量化：INT8 / FP8 / GPTQ 基礎 | |
| 72 | MoE 與路由（top-k gating、load balance） | |
| 73 | FlashAttention 概念：tiling 與 online softmax | |
| 74 | Continuous batching 與推論排程器 | 接 Day 52 併發 |

## Part 9 — 檢索、RAG 與記憶層（Day 75–80）

| Day | 主題 | 備註 |
|---|---|---|
| 75 | Embedding 與向量相似度檢索 | word2vec、cosine、暴力搜尋當 baseline |
| 76 | HNSW 近似最近鄰 | 接 Day 38 Skip List |
| 77 | IVF 與 Product Quantization | |
| 78 | BM25、hybrid search 與 rerank | 接 Day 29 rolling hash |
| 79 | Chunking 策略與 minimal RAG pipeline | 端到端可跑 |
| 80 | 記憶層：短期 / 長期 / episodic memory | Day 96 專案的地基 |

## Part 10 — 強化學習（Day 81–90）
從 bandit 一路走到 DPO，剛好解釋現在的 LLM 是怎麼被 align 的。

| Day | 主題 | 備註 |
|---|---|---|
| 81 | Multi-Armed Bandit：ε-greedy、UCB、Thompson | 接 Day 51 隨機化 |
| 82 | MDP 與 Value Iteration | GridWorld |
| 83 | Policy Iteration 與 RL 中的 DP | 接 Part 5 |
| 84 | Monte Carlo 與 TD Learning | |
| 85 | Q-Learning（表格式） | |
| 86 | DQN：replay buffer、target network | |
| 87 | REINFORCE 與 policy gradient | |
| 88 | Actor-Critic / A2C | |
| 89 | PPO | |
| 90 | RLHF 與 DPO | 把 Day 66 的 mini-GPT 拿來 align |

## Part 11 — 搜尋與演化計算（Day 91–94）

| Day | 主題 | 備註 |
|---|---|---|
| 91 | MCTS（AlphaZero-lite，井字棋 / 連四） | 接 Day 48 回溯 + Day 20 A* |
| 92 | 基因演算法基礎：編碼、選擇、交配、突變 | 用 GA 解 Day 46 的 TSP |
| 93 | 模擬退火、粒子群與爬山法 | 無梯度優化橫向比較 |
| 94 | 演化策略 ES / CMA-ES 簡版與 NEAT | 演化 vs 梯度 |

## Part 12 — 中型專案：LLM Agent Harness（Day 95–100）
前面 94 天的零件，這裡組成一個能跑的東西。

| Day | 主題 | 備註 |
|---|---|---|
| 95 | Agent Harness I：tool calling 與 JSON action loop | 工具註冊表、schema 驗證、重試 |
| 96 | Agent Harness II：接上記憶層 | 用 Day 80 + Day 76 HNSW |
| 97 | Agent Harness III：ReAct / planner 與多步推理 | 錯誤恢復、預算控制 |
| 98 | Multi-Agent 協作與任務 DAG 排程 | 呼應 Day 13 拓樸排序 |
| 99 | Eval Harness：LLM-as-judge、回歸測試、追蹤 | |
| 100 | 收官：100 天知識地圖 + 效能剖析 + 打包開源 | |

---

## 依賴關係（有意設計的伏筆）

- Day 05 heap → Day 16 Prim、Day 17 Dijkstra、Day 23 heap sort
- Day 13 拓樸排序 → Day 98 多代理任務 DAG
- Day 14 Union-Find → Day 15 Kruskal
- Day 27 Trie → Day 28 Radix Tree → Day 67 RadixAttention
- Day 37 LRU → Day 66 KV cache → Day 67 RadixAttention（radix tree + LRU 驅逐）→ Day 68 paged attention
- Day 38 Skip List → Day 76 HNSW
- Day 46 狀壓 TSP → Day 92 GA 解 TSP（精確 vs 啟發式對照）
- Day 48 回溯 → Day 91 MCTS
- Day 52 asyncio → Day 74 continuous batching → Day 95 agent loop
- Day 65 mini-GPT → Day 90 RLHF/DPO → Day 95–100 專案
