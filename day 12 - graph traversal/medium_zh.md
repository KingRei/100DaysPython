# Day 12 — 圖的走訪：BFS 與 DFS

*100 Days of Python — 第 12 天*

昨天我們把圖建好了，今天要學怎麼走過它。畢竟一個走不進去的圖，說穿了只是一堆邊。

接下來這一週幾乎所有東西——拓樸排序、最小生成樹、最短路徑、連通分量、環偵測——都是下面這兩種走訪加上一點額外的記帳。所以今天內容不多，但後座力很長。

## 一個演算法，兩種容器

整個概念其實只有這樣：準備一個容器，裝「已經發現但還沒展開」的頂點。拿一個出來，看它的鄰居，把沒看過的丟進去。重複到容器空掉為止。

就這樣。你唯一要做的選擇是容器要用什麼：

- 用 **queue**（先進先出）就是 **BFS，breadth-first search**
- 用 **stack**（後進先出）就是 **DFS，depth-first search**

兩個都是 day 02 學過的資料結構，卻長出兩種完全不同的走訪方式。

## BFS：一層一層往外掃

BFS 每次取出最早被發現的頂點，所以它一定會把距離 1 的處理完，才碰距離 2 的。走訪結果就像以起點為中心一圈一圈擴散：

![BFS layers](imgs/day12_1.png)

```python
from collections import deque

def bfs(graph, start):
    """Return (visit order, distance from start, parent pointers)."""
    visited = {start}
    dist = {start: 0}
    parent = {start: None}
    order = []
    queue = deque([start])

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(graph.neighbors(u)):
            if v not in visited:
                visited.add(v)              # mark on push, not on pop
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return order, dist, parent
```

這段裡面只有一行容易寫錯，就是那行註解講的事。

**要在「推進 queue 的當下」標記 visited，不是在「取出來」的時候標記。** 如果你在取出時才標記，一個能被兩個不同父節點走到的頂點就會被推進去兩次，於是被處理兩次。在小圖上你不會發現；但在一百萬格的網格上你一定會發現，因為 queue 會悄悄爆掉。

順手維護 `dist` 和 `parent` 的好處是：BFS 在無權圖上不只是「走過每個點」，而是**依照距離遞增的順序**走過每個點。所以 `dist` 直接就是最短路徑的長度（以邊數計），而沿著 `parent` 往回走就能還原出路徑本身：

```python
def shortest_path(graph, start, goal):
    _, _, parent = bfs(graph, start)
    if goal not in parent:
        return None                     # unreachable
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]
```

迷宮解題、單字接龍（word ladder）、社群網路的「幾度分隔」，底層都是這一套，不需要更複雜的東西——前提是每條邊的成本都一樣。一旦邊有了不同的權重，這招就失效了，那時我們得把 queue 換成 heap，那就是 day 17 的 Dijkstra。

## DFS：一路走到底再回頭

把 queue 換成 stack，走訪就整個翻過來。DFS 認準一條路就走到底，走不動了才退回去：

![DFS order](imgs/day12_2.png)

```python
def dfs_recursive(graph, start, visited=None, order=None):
    if visited is None:
        visited, order = set(), []
    visited.add(start)
    order.append(start)
    for v in sorted(graph.neighbors(start)):
        if v not in visited:
            dfs_recursive(graph, v, visited, order)
    return order
```

遞迴版值得背起來，因為它讀起來就跟定義一樣：標記自己，然後往還沒標記的鄰居遞迴下去。這裡的 stack 就是呼叫堆疊。

當然也有迭代版，而且這不只是風格偏好——CPython 預設大約給你一千層 stack frame，一個又長又細的圖很容易就撞到 `RecursionError`：

```python
def dfs_iterative(graph, start):
    visited, order = set(), []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        for v in sorted(graph.neighbors(u), reverse=True):
            if v not in visited:
                stack.append(v)
    return order
```

注意迭代版在 pop 之後又檢查了一次 `visited`。這不是我上面警告的那個 bug——用 stack 的時候，同一個頂點確實可能同時躺在兩個位置，最省事的解法就是取出來時發現已經看過就跳過。

再看一眼上面那張圖的灰色邊。那些是 DFS **沒有走**的邊，因為兩端都已經拜訪過了。青色的邊構成 **DFS tree**。這個「樹邊 vs 其他邊」的分類不是為了好看：判斷環、找橋（bridge）與關節點、算強連通分量，靠的全是把非樹邊分類清楚。這筆帳我們會在 day 19 兌現。

## 那到底該用哪一個？

| | BFS | DFS |
|---|---|---|
| 容器 | queue | stack／遞迴 |
| 找得到 | 無權圖的最短路徑 | *某一條*路徑，不保證最短 |
| 記憶體 | O(寬度)，可能非常大 | O(深度) |
| 適合 | 層數、距離、「最近的」 | 環、連通分量、排序、回溯 |

兩者都是 O(V + E)：每個頂點進容器一次，每條邊從兩端各檢查一次。差別在記憶體。又寬又淺的圖，BFS 會把整個 frontier 塞在 queue 裡；又深又窄的圖，DFS 會把整條路徑堆在 stack 上。挑一個「你的圖不會踩到它最壞情況」的那個。

## 今日題目：LeetCode 200，島嶼數量

> 給定一個 `m x n` 的二維網格，`'1'` 是陸地、`'0'` 是水，回傳島嶼的數量。相鄰（上下左右）的陸地屬於同一座島。

這題值得花 15 分鐘的原因不在程式碼，而在於看穿它是一題**披著網格外皮的圖論題**。每個陸地格子是一個頂點，上下左右相鄰的兩個陸地格子之間有一條邊。數島嶼就是數連通分量——正是我們剛剛寫過的東西，只要對每個還沒走過的頂點各跑一次走訪。

![Number of islands](imgs/day12_3.png)

```python
def num_islands_bfs(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    grid = [list(row) for row in grid]      # work on a mutable copy
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != '1':
                continue
            count += 1                      # found a fresh island
            queue = deque([(r, c)])
            grid[r][c] = '0'                # sink it: the grid is our visited set
            while queue:
                i, j = queue.popleft()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == '1':
                        grid[ni][nj] = '0'
                        queue.append((ni, nj))
    return count
```

讓程式碼變短的關鍵，是走到哪就把陸地沉掉。你不需要另外開一個 `visited` 集合，因為算過的格子已經不是 `'1'` 了。面試時有兩件事要主動講出來：這個寫法會改到輸入（在意的話就先複製一份），還有 BFS 版的記憶體是 O(min(m, n))，而遞迴 flood fill 最壞會在呼叫堆疊上壓 m×n 層。在 1000×1000 的全陸地網格上，這就是「答得出來」和「`RecursionError`」的差別。

時間複雜度是 O(m × n)：每格看的次數是常數，而且最多被沉掉一次。

## 這些東西之後會用在哪

明天是**拓樸排序**，它其實就是 DFS 加一個後序時間戳——或者換個口味，用 BFS 跑 in-degree 也可以。再來 Kruskal 與 Prim 用同樣的方式往外長一棵樹，而 Dijkstra 說穿了就是把 BFS 的 queue 換成 day 05 的 heap。

今天如果只記一件事：**容器決定順序，順序決定這個走訪適合拿來做什麼。**

程式碼與 notebook：[day 12 on GitHub](https://github.com/KingRei/100DaysPython/tree/master/day%2012%20-%20graph%20traversal)

*下一篇：Day 13 — 拓樸排序，以及為什麼選課表其實是一張 DAG。*
