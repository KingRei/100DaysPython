# Backtracking and pruning: N-Queens and Sudoku

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that grows every subset of
`[1, 2, 3]` with one shared `path` and shows what `res.append(path)` without the copy
leaves behind, places queens row by row on 4, 5 and 6 boards while the three attack sets
light up the squares they rule out, reruns 8-queens with the three `set.remove()` calls
deleted and watches ghost queens block the board until it reports "no solution", solves
the LeetCode 37 board in 52 nodes by always branching on the cell with the fewest legal
digits (against 4,209 in reading order), and then replays LeetCode 39 and LeetCode 79
with and without the one line each of them depends on.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_1.png?raw=true)

Backtracking finds every 8-queens solution in 2,057 steps, where trying one queen per row
in every column would mean sixteen million boards. It is depth-first search over a tree of
partial answers that is never actually built: place one piece, recurse, take the piece
back, try the next one. Three lines carry every problem in this folder - **choose,
explore, unchoose**.

Two separate things decide whether it works. **Pruning decides whether it finishes.**
Rejecting a partial answer the moment it breaks a rule removes the whole subtree below it,
and the earlier the rejection, the bigger the subtree. **The unchoose step decides whether
it is right.** Backtracking bugs almost never crash: a missing undo reports that 8-queens
has no solution, a missing copy returns eight empty lists, and a symmetry shortcut is
exact for every even board size and wrong for every odd one that has a solution. Each of
those is below, with the number it gets wrong.

Day 38's greedy committed to one choice and never looked back. Backtracking is the
opposite: it commits to a choice only *tentatively* and is built around taking it back.

## The template: choose, explore, unchoose

```python
def subsets(nums):
    res, path = [], []

    def go(start):
        res.append(path[:])              # record a COPY of the path
        for i in range(start, len(nums)):
            path.append(nums[i])         # choose
            go(i + 1)                    # explore
            path.pop()                   # unchoose

    go(0)
    return res
```

One `path` list is shared by the whole search. It grows by one on the way down and shrinks
by one on the way back, so at every moment it describes exactly the route from the root to
the current node. `path.pop()` is what lets the loop try `[1, 3]` after `[1, 2]`: the 2 has
to come off first.

## Silent failure: storing the path instead of a copy

Write `res.append(path)` instead of `res.append(path[:])` and the function returns
`[[], [], [], [], [], [], [], []]`. The length is right - eight - so a test that only checks
`len` passes. But all eight slots hold the *same list object* (one distinct `id`), and by
the time the search returns, every `pop()` has emptied it. The copy freezes the path at
the moment it is recorded.

## LeetCode 78 - Subsets

**The task.** Given a list of distinct integers, produce every possible subset of it,
including the empty one and the whole list.

**Input / output.** A list `nums`. Returns a list of lists; each inner list is one subset,
and the order of the subsets (and of the numbers inside each one) does not matter, but no
subset may appear twice.

**Example.** `[1, 2, 3]` returns the 8 subsets `[]`, `[1]`, `[1,2]`, `[1,2,3]`, `[1,3]`,
`[2]`, `[2,3]`, `[3]` - every number is either in or out, so there are 2 x 2 x 2 of them.

**Constraints.** At most 10 numbers, each between -10 and 10, all different. Ten numbers
means the answer itself has 1,024 lists, so an exponential algorithm is not a failure here,
it is the size of the output. "All different" is the constraint that keeps the template
this short: with repeated numbers, `[1, 2]` could be built from two different 2s, and the
search would have to sort and skip equal neighbours.

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res, path = [], []

        def go(start):
            res.append(path[:])              # the copy is the whole trick
            for i in range(start, len(nums)):
                path.append(nums[i])
                go(i + 1)
                path.pop()

        go(0)
        return res
```

[leetcode.com/problems/subsets](https://leetcode.com/problems/subsets/)

## N-Queens: one queen per row, three sets for the attacks

Place n queens on an n x n board so that no two attack each other. Put exactly one queen
in each row and the only decision at row `r` is the column. A square `(r, c)` is attacked
along its column `c`, along its "\" diagonal - every square on it has the same `r - c` -
and along its "/" diagonal, where `r + c` is constant. Three sets make the safety check
three lookups:

```python
def queens(n):
    cols, diag, anti = set(), set(), set()
    board, sols = [], []

    def go(r):
        if r == n:
            sols.append(board[:])
            return
        for c in range(n):
            if c in cols or r - c in diag or r + c in anti:
                continue                                  # prune
            cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)
            go(r + 1)
            board.pop(); cols.remove(c); diag.remove(r - c); anti.remove(r + c)

    go(0)
    return sols
```

The figure at the top is the whole search on a 4 x 4 board: 17 nodes, 2 solutions, 4 dead
ends, and 44 squares turned down without a queen ever being placed on them.

## What the pruning buys

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_2.png?raw=true)

Every rule built into the shape of the search cuts the space on an 8 x 8 board:

| what is searched | size |
|---|---:|
| any 8 of the 64 squares | 4,426,165,368 |
| one queen per row | 16,777,216 |
| one per row and per column (8!) | 40,320 |
| backtracking with diagonal pruning (nodes entered) | 2,057 |

The fair comparison for the last row is the full tree of column permutations - the tree a
search without the diagonal check would walk - which has 109,601 nodes, 53 times more.
Checking the diagonals only once a board is full (generate all 40,320 permutations, test
each) finds the same 92 solutions; the pruned search just never builds the boards that
were already broken at row 2.

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solutions | 1 | 0 | 0 | 2 | 10 | 4 | 40 | 92 | 352 | 724 |
| nodes | 2 | 3 | 6 | 17 | 54 | 153 | 552 | 2,057 | 8,394 | 35,539 |

## Silent failure: forgetting the unchoose

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_3.png?raw=true)

Keep `board.pop()` and delete the three `set.remove()` calls. The board looks right - the
queen is gone - but the three sets still remember her, so every queen that is taken back
leaves a ghost that keeps attacking its column and both diagonals. On 8 x 8 the search
enters nine partial boards (`[]`, `[0]`, `[0,2]`, `[0,2,4]`, `[0,2,4,1]`, `[0,2,4,1,3]`,
`[0,2,4,1,7]`, `[0,2,6]`, `[5]`) and then returns **0 solutions**. The right answer is 92.
Nothing is raised; the function is simply, confidently wrong. The rule behind it is a
symmetry: whatever the choose step changed, the unchoose step has to change back, every
single piece of it.

## Pruning must never cut a real answer: the mirror shortcut

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_4.png?raw=true)

Flip any solution left to right and you get another solution, so a tempting speed-up is to
try only the left half of row 0 and double the count. That is exact on every even board.
On an odd board the middle column is its own mirror image: the shortcut never searches it
and never counts it, so it returns 8 instead of 10 for n = 5, 34 instead of 40 for n = 7,
298 instead of 352 for n = 9 (n = 3 comes out right only because it has no solutions to
lose). The fix is to search the middle column once and count it once. Pruning is only
allowed to remove subtrees that provably contain no answer - "these are probably the same"
is not a proof.

## The bitmask version: same tree, cheaper nodes

Day 37's bit tricks fit N-Queens exactly. Keep the attacked columns and both diagonals as
bit rows *as seen from the current row*: moving down one row shifts the "\" attacks one
bit and the "/" attacks one bit the other way, so there is no `r - c` bookkeeping at all,
and `free & -free` pulls out the lowest free column.

```python
def total_queens(n):
    full = (1 << n) - 1

    def go(cols, ld, rd):
        if cols == full:
            return 1
        total, free = 0, full & ~(cols | ld | rd)
        while free:
            bit = free & -free                   # lowest free column
            free ^= bit
            total += go(cols | bit, ((ld | bit) << 1) & full, (rd | bit) >> 1)
        return total

    return go(0, 0, 0)
```

On n = 10 both versions enter exactly 35,539 nodes; the bitmask one is about four times
faster. A better check changes what a node costs, not how many nodes there are.

## LeetCode 51 - N-Queens

**The task.** Find every way to place n queens on an n x n chessboard so that no two share
a row, a column or a diagonal.

**Input / output.** An integer `n`. Returns a list of boards; each board is a list of `n`
strings of length `n`, with `'Q'` for a queen and `'.'` for an empty square. Any order.

**Example.** `n = 4` returns two boards: `[".Q..", "...Q", "Q...", "..Q."]` and
`["..Q.", "Q...", "...Q", ".Q.."]`. Reading the columns row by row they are `[1, 3, 0, 2]`
and `[2, 0, 3, 1]`, mirror images of each other - the tree in the first figure finds
exactly these two.

**Constraints.** `1 <= n <= 9`. The upper bound is what makes plain backtracking the
intended solution: at n = 9 there are 352 boards and the pruned search enters 8,394
nodes. Note that n = 2 and n = 3 have **no** solution and must return an empty list,
while n = 1 has one.

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        cols, diag, anti = set(), set(), set()
        board, out = [], []

        def go(r):
            if r == n:
                out.append(['.' * c + 'Q' + '.' * (n - c - 1) for c in board])
                return
            for c in range(n):
                if c in cols or r - c in diag or r + c in anti:
                    continue
                cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)
                go(r + 1)
                board.pop(); cols.remove(c); diag.remove(r - c); anti.remove(r + c)

        go(0)
        return out
```

[leetcode.com/problems/n-queens](https://leetcode.com/problems/n-queens/)

## LeetCode 52 - N-Queens II

**The task.** The same puzzle, but only the number of solutions is wanted, not the boards.

**Input / output.** An integer `n`. Returns an integer.

**Example.** `n = 4` returns `2`, the two boards above. `n = 8` returns `92`.

**Constraints.** `1 <= n <= 9`. Because nothing has to be drawn, the board list and the
string building can go, which is what makes the bitmask search the natural answer.

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        full = (1 << n) - 1

        def go(cols, ld, rd):
            if cols == full:
                return 1
            total, free = 0, full & ~(cols | ld | rd)
            while free:
                bit = free & -free
                free ^= bit
                total += go(cols | bit, ((ld | bit) << 1) & full, (rd | bit) >> 1)
            return total

        return go(0, 0, 0)
```

[leetcode.com/problems/n-queens-ii](https://leetcode.com/problems/n-queens-ii/)

## Sudoku: which cell you branch on beats how fast you check it

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_5.png?raw=true)

The textbook Sudoku solver fills the first empty cell in reading order with 1 to 9, checks
the row, the column and the 3 x 3 box, and recurses. The better rule is called MRV,
*minimum remaining values*: always branch on the empty cell that has the **fewest** legal
digits left. A cell with one legal digit gets filled with no guessing at all. A cell with
zero legal digits proves the current path is dead right now, many levels before reading
order would run into it.

| puzzle | givens | reading order, nodes | MRV, nodes | ratio |
|---|---:|---:|---:|---:|
| LeetCode 37 example | 30 | 4,209 | 52 | 81x |
| Arto Inkala's "hardest sudoku" (2012) | 21 | 49,559 | 15,573 | 3x |
| anti-brute (17 givens) | 17 | 69,175,317 | 17,918 | 3,861x |

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2039%20-%20backtracking%20and%20pruning/imgs/day39_6.png?raw=true)

The third puzzle is built against reading order: its top row solves to `987654321`, the
exact reverse of the order in which the naive solver tries digits. The top-left cell has 7
legal digits and the answer is the last of them, 9, so the naive solver works through six
wrong digits there, each with a whole subtree behind it. It needs about two minutes of pure
Python for this board (run `python backtracking.py --full`). MRV instead starts on the third
cell of the second row, which has only 3 legal digits, and finishes in a quarter of a second.

Fewer nodes is not automatically less time, though. MRV scans every empty cell at every
node to find the tightest one. On Inkala, where it only saves a factor of three, the plain
reading-order solver with bitmask checks finishes in about half the time MRV takes. MRV
wins on the worst cases, which is where it matters.

## LeetCode 37 - Sudoku Solver

**The task.** Fill in a partially completed 9 x 9 Sudoku so that every row, every column
and every one of the nine 3 x 3 boxes contains each digit 1 to 9 exactly once.

**Input / output.** A 9 x 9 grid of single characters, `'1'` to `'9'` or `'.'` for an empty
cell. Returns nothing: the grid must be filled in place.

**Example.** The grid whose first row is `5 3 . . 7 . . . .` (30 digits given in total)
ends with first row `5 3 4 6 7 8 9 1 2`. Only one completion satisfies all 27 rules.

**Constraints.** The board is always 9 x 9, holds only digits and `'.'`, and is guaranteed
to have exactly one solution. The guarantee matters in two ways: the search can stop at
the first complete grid, and it never has to report failure. The in-place requirement means
the unchoose step must also write `'.'` back into the caller's board, not into a copy.

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9    # bit v = digit v used
        empty = []
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    empty.append((r, c))
                else:
                    bit = 1 << int(board[r][c])
                    rows[r] |= bit; cols[c] |= bit; boxes[r // 3 * 3 + c // 3] |= bit

        def go():
            if not empty:
                return True
            best, best_free, best_k = -1, 0, 10          # MRV: fewest legal digits
            for i, (r, c) in enumerate(empty):
                free = ~(rows[r] | cols[c] | boxes[r // 3 * 3 + c // 3]) & 0x3FE
                k = bin(free).count('1')
                if k < best_k:
                    best, best_free, best_k = i, free, k
                    if k <= 1:
                        break
            if best_k == 0:
                return False                             # dead end, found early
            r, c = empty[best]
            empty[best] = empty[-1]; empty.pop()
            b = r // 3 * 3 + c // 3
            while best_free:
                bit = best_free & -best_free
                best_free ^= bit
                rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
                board[r][c] = str(bit.bit_length() - 1)
                if go():
                    return True
                rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
            board[r][c] = '.'
            empty.append((r, c))
            empty[best], empty[-1] = empty[-1], empty[best]
            return False

        go()
```

[leetcode.com/problems/sudoku-solver](https://leetcode.com/problems/sudoku-solver/)

## LeetCode 39 - Combination Sum

**The task.** Given a set of distinct positive numbers and a target, list every
combination of those numbers that adds up to the target. A number may be used as many
times as you like.

**Input / output.** A list `candidates` and an integer `target`. Returns a list of lists.
Two combinations count as the same if they use the same numbers the same number of times,
so `[2, 2, 3]` and `[3, 2, 2]` are one answer, not two.

**Example.** `candidates = [2, 3, 6, 7]`, `target = 7` returns `[[2, 2, 3], [7]]`. 2 + 2 + 3
is 7, and 7 on its own is 7; nothing else works.

**Constraints.** Up to 30 candidates, each between 2 and 40, all different; the target is
between 1 and 40; fewer than 150 answers are guaranteed. The smallest candidate being 2
caps the depth of the search at 20. "All different" means a start index alone is enough to
avoid duplicates.

Two lines carry this problem, and they do different jobs. The **start index** - recurse
with `i`, never going back to earlier candidates - is about correctness: it forces every
path to be non-decreasing, so each combination appears in exactly one order. Without it
the search returns 4 answers, including `[2, 3, 2]` and `[3, 2, 2]`. Passing `i` rather
than `i + 1` is what allows reuse. **Sorting plus `break`** is the pruning: once a
candidate is bigger than what is left, every later one is too. On `[2, 3, 5, 7, 11]` with
target 30 that cuts the search from 1,239 nodes to 575 with the same 64 answers.

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res, path = [], []

        def go(start, remaining):
            if remaining == 0:
                res.append(path[:])
                return
            for i in range(start, len(candidates)):
                x = candidates[i]
                if x > remaining:
                    break                        # every later x is bigger too
                path.append(x)
                go(i, remaining - x)             # i, not i + 1: x may repeat
                path.pop()

        go(0, target)
        return res
```

[leetcode.com/problems/combination-sum](https://leetcode.com/problems/combination-sum/)

## LeetCode 79 - Word Search

**The task.** Decide whether a word can be spelled on a grid of letters by starting on any
cell and stepping up, down, left or right, one cell per letter, never using the same cell
twice in one spelling.

**Input / output.** A grid `board` of letters and a string `word`. Returns `True` or
`False`.

**Example.** On the board

```
A B C E
S F C S
A D E E
```

`"ABCCED"` is `True` (along the top row to the second C, down, then left along the
bottom), `"SEE"` is `True` (the S on the right edge, then the two Es below it), and
`"ABCB"` is `False`: after A, B, C the only B is the one already used.

**Constraints.** The grid is at most 6 x 6, the word is 1 to 15 letters, and letters can be
upper or lower case. The small grid is what makes a full depth-first search from every
cell acceptable; the "never reuse a cell" rule is what makes it backtracking rather than
plain DFS - a cell is only "used" while it is on the current path.

The standard trick marks a cell as used by overwriting it with `'#'`, which can never match
a letter. The unchoose is writing the letter back. Delete that line and all three examples
above still pass. But `"CCB"` - C at (1,2), up to C at (0,2), left to B - comes back
`False`: the first attempt, starting at (0,2), stamps both Cs with `'#'` and fails, so by
the time the search starts from (1,2) its answer is already blocked. Across every string
this board actually spells along paths of up to 7 cells (1,108 of them) the broken version
misses 201. Passing the examples does not mean the backtracking is right.

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        R, C = len(board), len(board[0])

        def dfs(r, c, i):
            if i == len(word):
                return True
            if not (0 <= r < R and 0 <= c < C) or board[r][c] != word[i]:
                return False
            saved, board[r][c] = board[r][c], '#'        # choose: mark as used
            found = (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1)
                     or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))
            board[r][c] = saved                          # unchoose: ALWAYS
            return found

        return any(dfs(r, c, 0) for r in range(R) for c in range(C))
```

[leetcode.com/problems/word-search](https://leetcode.com/problems/word-search/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| subsets (LeetCode 78) | O(n · 2^n) | O(n) besides the output |
| N-Queens, sets or bitmask (LeetCode 51 / 52) | O(n!) worst case, far fewer nodes in practice | O(n) |
| Sudoku, reading order | O(9^m) for m empty cells | O(m) |
| Sudoku, MRV | O(9^m) worst case, O(m) work per node to pick the cell | O(m) |
| Combination Sum (LeetCode 39) | exponential in target / min(candidates) | O(target / min) depth |
| Word Search (LeetCode 79) | O(R · C · 3^L) for a word of length L | O(L) recursion |

The worst-case bounds hardly move under pruning; what pruning changes is how much of the
tree is actually visited, which is why every row above is measured in nodes.

## References

- [Backtracking - Wikipedia](https://en.wikipedia.org/wiki/Backtracking)
- [Eight queens puzzle - Wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)
- [Sudoku solving algorithms - Wikipedia](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms)
- [Constraint satisfaction problem - Wikipedia](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem) - where "minimum remaining values" comes from
- Donald Knuth, "Dancing Links", 2000 - exact-cover backtracking for Sudoku and N-Queens
- Peter Norvig, [Solving Every Sudoku Puzzle](https://norvig.com/sudoku.html)
