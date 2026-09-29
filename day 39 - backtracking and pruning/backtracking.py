"""Day 39 - Backtracking and pruning: N-Queens and Sudoku.

Run me:  python3 backtracking.py            (a few seconds)
         python3 backtracking.py --full     (also counts the full 69 million
                                             node naive Sudoku search, ~2 min)

Backtracking is depth-first search over a tree of partial answers that is
never built: place one piece, recurse, take the piece back, try the next.
Three lines - choose, explore, unchoose - carry every problem today.

Two things decide whether it works.  Pruning decides whether it finishes:
reject a partial answer the moment it breaks a rule and the whole subtree
below it disappears.  And the unchoose step decides whether it is right:
backtracking bugs almost never crash.  A missing undo reports that 8-queens
has no solution, a missing copy returns 92 empty boards, a symmetry shortcut
is exact for every even n and wrong for every odd n that has a solution.
"""

from __future__ import annotations

import math
import sys
import time
from itertools import permutations

sys.setrecursionlimit(10_000)


def rule(title: str) -> None:
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


# ============================================================ the template
# choose -> explore -> unchoose, shown on the smallest possible problem:
# every subset of [1, 2, 3].

def subsets(nums: list[int], trace: list | None = None) -> list[list[int]]:
    res: list[list[int]] = []
    path: list[int] = []

    def go(start: int) -> None:
        res.append(path[:])                  # record a COPY of the path
        if trace is not None:
            trace.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])             # choose
            go(i + 1)                        # explore
            path.pop()                       # unchoose

    go(0)
    return res


def subsets_aliased(nums: list[int]) -> list[list[int]]:
    """The same function with `path[:]` written as `path`."""
    res: list[list[int]] = []
    path: list[int] = []

    def go(start: int) -> None:
        res.append(path)                     # BUG: stores the list object itself
        for i in range(start, len(nums)):
            path.append(nums[i])
            go(i + 1)
            path.pop()

    go(0)
    return res


# ================================================================ N-Queens
# One queen per row, so the only choice at row r is the column.  A square
# (r, c) is attacked through its column c, its "\" diagonal (every square on
# it has the same r - c) and its "/" diagonal (same r + c).  Three sets make
# the safety test O(1).

def queens(n: int, stats: dict | None = None) -> list[list[int]]:
    """All solutions; each is a list of column indices, one per row."""
    cols: set[int] = set()
    diag: set[int] = set()       # r - c
    anti: set[int] = set()       # r + c
    board: list[int] = []
    sols: list[list[int]] = []
    st = stats if stats is not None else {}
    st.setdefault('nodes', 0)
    st.setdefault('tested', 0)

    def go(r: int) -> None:
        st['nodes'] += 1
        if r == n:
            sols.append(board[:])
            return
        for c in range(n):
            st['tested'] += 1
            if c in cols or r - c in diag or r + c in anti:
                continue                                  # prune
            cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)
            go(r + 1)
            board.pop(); cols.remove(c); diag.remove(r - c); anti.remove(r + c)

    go(0)
    return sols


def queens_no_undo(n: int) -> tuple[list[list[int]], int, list[list[int]]]:
    """Same search, but the unchoose line only pops the board and forgets
    to release the column and the two diagonals.  Returns (solutions,
    nodes visited, every partial board it entered)."""
    cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()
    board: list[int] = []
    sols: list[list[int]] = []
    seen: list[list[int]] = []

    def go(r: int) -> None:
        seen.append(board[:])
        if r == n:
            sols.append(board[:])
            return
        for c in range(n):
            if c in cols or r - c in diag or r + c in anti:
                continue
            cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)
            go(r + 1)
            board.pop()                          # BUG: sets never shrink

    go(0)
    return sols, len(seen), seen


def queens_aliased(n: int) -> list[list[int]]:
    """Correct search, but stores `board` instead of `board[:]`."""
    cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()
    board: list[int] = []
    sols: list[list[int]] = []

    def go(r: int) -> None:
        if r == n:
            sols.append(board)                   # BUG: same list every time
            return
        for c in range(n):
            if c in cols or r - c in diag or r + c in anti:
                continue
            cols.add(c); diag.add(r - c); anti.add(r + c); board.append(c)
            go(r + 1)
            board.pop(); cols.remove(c); diag.remove(r - c); anti.remove(r + c)

    go(0)
    return sols


def queens_count_bits(n: int, stats: dict | None = None) -> int:
    """Day 37's bitmask trick applied to N-Queens (LeetCode 52).

    cols / ld / rd are bit rows of attacked columns as seen from the current
    row.  Moving down one row shifts the "\" attacks right and the "/"
    attacks left, so no r - c bookkeeping is needed at all."""
    full = (1 << n) - 1
    st = stats if stats is not None else {}
    st.setdefault('nodes', 0)

    def go(cols: int, ld: int, rd: int) -> int:
        st['nodes'] += 1
        if cols == full:
            return 1
        total = 0
        free = full & ~(cols | ld | rd)
        while free:
            bit = free & -free                  # lowest free column
            free ^= bit
            total += go(cols | bit, ((ld | bit) << 1) & full, (rd | bit) >> 1)
        return total

    return go(0, 0, 0)


def queens_count_half(n: int) -> int:
    """A tempting symmetry shortcut: only try the left half of row 0 and
    double the count, since the mirror image of a solution is a solution.
    Exact whenever n is even.  For odd n it never tries the middle column."""
    cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()
    count = 0

    def go(r: int) -> None:
        nonlocal count
        if r == n:
            count += 1
            return
        for c in (range(n // 2) if r == 0 else range(n)):
            if c in cols or r - c in diag or r + c in anti:
                continue
            cols.add(c); diag.add(r - c); anti.add(r + c)
            go(r + 1)
            cols.remove(c); diag.remove(r - c); anti.remove(r + c)

    go(0)
    return 2 * count


def queens_count_half_fixed(n: int) -> int:
    """The shortcut done right: the middle column of an odd board is its
    own mirror image, so it is searched once and counted once."""
    def from_first(cs) -> int:
        cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()
        count = 0

        def go(r: int) -> None:
            nonlocal count
            if r == n:
                count += 1
                return
            for c in (cs if r == 0 else range(n)):
                if c in cols or r - c in diag or r + c in anti:
                    continue
                cols.add(c); diag.add(r - c); anti.add(r + c)
                go(r + 1)
                cols.remove(c); diag.remove(r - c); anti.remove(r + c)

        go(0)
        return count

    total = 2 * from_first(range(n // 2))
    if n % 2:
        total += from_first([n // 2])
    return total


def queens_by_permutation(n: int) -> tuple[int, int]:
    """No pruning on diagonals: generate every column permutation and only
    check the diagonals once the board is full.  Returns (solutions, boards
    checked)."""
    found = checked = 0
    for p in permutations(range(n)):
        checked += 1
        if (len({r - c for r, c in enumerate(p)}) == n
                and len({r + c for r, c in enumerate(p)}) == n):
            found += 1
    return found, checked


def permutation_tree_nodes(n: int) -> int:
    """Nodes in the column-permutation tree (1 + n + n(n-1) + ... + n!)."""
    total, level = 1, 1
    for k in range(n):
        level *= n - k
        total += level
    return total


def queens_tree(n: int) -> dict:
    """The search tree the pruned N-Queens search walks, as nested dicts:
    {'path': [...], 'children': [...], 'pruned': [cols rejected], 'solution': bool}.
    Used by the figures; same order and same pruning as queens()."""
    cols: set[int] = set(); diag: set[int] = set(); anti: set[int] = set()

    def go(path: list[int]) -> dict:
        r = len(path)
        me = {'path': path[:], 'children': [], 'pruned': [], 'solution': r == n}
        if r == n:
            return me
        for c in range(n):
            if c in cols or r - c in diag or r + c in anti:
                me['pruned'].append(c)
                continue
            cols.add(c); diag.add(r - c); anti.add(r + c)
            me['children'].append(go(path + [c]))
            cols.remove(c); diag.remove(r - c); anti.remove(r + c)
        return me

    return go([])


def draw_board(cols: list[int]) -> list[str]:
    n = len(cols)
    return [' '.join('Q' if cols[r] == c else '.' for c in range(n)) for r in range(n)]


# ================================================================== Sudoku
# 81 characters, row by row, '.' for an empty cell.

PUZZLES: dict[str, str] = {
    # the example board from LeetCode 37: 30 givens, most cells are forced
    'LeetCode 37': '53..7....6..195....98....6.8...6...34..8.3..17...2...6.'
                   '6....28....419..5....8..79',
    # Arto Inkala's 2012 "world's hardest sudoku": 21 givens
    'Inkala': '8..........36......7..9.2...5...7.......457.....1...3...1'
              '....68..85...1..9....4..',
    # a 17-given puzzle whose top row solves to 987654321 - the exact
    # reverse of the order in which a naive solver tries the digits
    'anti-brute': '..............3.85..1.2.......5.7.....4...1...9.......5'
                  '......73..2.1........4...9',
}

# Nodes the naive row-major solver visits on 'anti-brute'.  Measured by
# sudoku_rowmajor_bits(PUZZLES['anti-brute'], cap=None) - about two minutes of
# pure Python, so main() only reruns it with --full.
ANTI_BRUTE_NAIVE_NODES = 69_175_317


def parse(s: str) -> list[list[int]]:
    return [[0 if ch == '.' else int(ch) for ch in s[r * 9:r * 9 + 9]] for r in range(9)]


def box_of(r: int, c: int) -> int:
    return r // 3 * 3 + c // 3


def is_valid_solution(g: list[list[int]], givens: str) -> bool:
    want = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != want or {g[r][i] for r in range(9)} != want:
            return False
    for b in range(9):
        if {g[r][c] for r in range(9) for c in range(9) if box_of(r, c) == b} != want:
            return False
    return all(ch == '.' or int(ch) == g[i // 9][i % 9] for i, ch in enumerate(givens))


class Cap(Exception):
    """Raised when a search exceeds its node budget."""


def sudoku_naive(s: str, cap: int | None = None):
    """The textbook solver: first empty cell in reading order, digits 1..9,
    and a safety test that scans the row, the column and the 3x3 box.
    Returns (grid or None, nodes visited); grid is None if the cap hit."""
    g = parse(s)
    nodes = 0

    def safe(r: int, c: int, v: int) -> bool:
        for i in range(9):
            if g[r][i] == v or g[i][c] == v:
                return False
        br, bc = r // 3 * 3, c // 3 * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if g[i][j] == v:
                    return False
        return True

    def go() -> bool:
        nonlocal nodes
        nodes += 1
        if cap is not None and nodes > cap:
            raise Cap
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    for v in range(1, 10):
                        if safe(r, c, v):
                            g[r][c] = v                  # choose
                            if go():                     # explore
                                return True
                            g[r][c] = 0                  # unchoose
                    return False                         # no digit fits here
        return True                                      # no empty cell left

    try:
        ok = go()
    except Cap:
        return None, nodes - 1
    return (g if ok else None), nodes


def sudoku_rowmajor_bits(s: str, cap: int | None = None):
    """Same order as sudoku_naive - so exactly the same tree and the same
    node count - but the row / column / box tests are three bitmasks
    (bit v set = digit v already used) instead of 27 cell reads."""
    g = parse(s)
    rows = [0] * 9; cols = [0] * 9; boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            if g[r][c]:
                bit = 1 << g[r][c]
                rows[r] |= bit; cols[c] |= bit; boxes[box_of(r, c)] |= bit
    empty = [(r, c) for r in range(9) for c in range(9) if g[r][c] == 0]
    nodes = 0

    def go(k: int) -> bool:
        nonlocal nodes
        nodes += 1
        if cap is not None and nodes > cap:
            raise Cap
        if k == len(empty):
            return True
        r, c = empty[k]
        b = box_of(r, c)
        free = ~(rows[r] | cols[c] | boxes[b]) & 0x3FE      # bits 1..9
        while free:
            bit = free & -free                              # smallest digit first
            free ^= bit
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            g[r][c] = bit.bit_length() - 1
            if go(k + 1):
                return True
            rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
            g[r][c] = 0
        return False

    try:
        ok = go(0)
    except Cap:
        return None, nodes - 1
    return (g if ok else None), nodes


def candidates(g: list[list[int]]) -> dict[tuple[int, int], list[int]]:
    """Legal digits for every empty cell of a grid."""
    out = {}
    for r in range(9):
        for c in range(9):
            if g[r][c] == 0:
                used = set(g[r]) | {g[i][c] for i in range(9)} | {
                    g[i][j] for i in range(r // 3 * 3, r // 3 * 3 + 3)
                    for j in range(c // 3 * 3, c // 3 * 3 + 3)}
                out[(r, c)] = [v for v in range(1, 10) if v not in used]
    return out


def sudoku_mrv(s: str, first_pick: list | None = None):
    """Minimum remaining values: branch on the empty cell that has the FEWEST
    legal digits.  A cell with one candidate is filled with no branching at
    all; a cell with zero candidates proves the current path is dead right
    now, many levels before the naive order would find out."""
    g = parse(s)
    rows = [0] * 9; cols = [0] * 9; boxes = [0] * 9
    empty: list[tuple[int, int]] = []
    for r in range(9):
        for c in range(9):
            if g[r][c]:
                bit = 1 << g[r][c]
                rows[r] |= bit; cols[c] |= bit; boxes[box_of(r, c)] |= bit
            else:
                empty.append((r, c))
    nodes = 0

    def go() -> bool:
        nonlocal nodes
        nodes += 1
        if not empty:
            return True
        best, best_free, best_k = -1, 0, 10
        for i, (r, c) in enumerate(empty):
            free = ~(rows[r] | cols[c] | boxes[box_of(r, c)]) & 0x3FE
            k = free.bit_count() if hasattr(int, 'bit_count') else bin(free).count('1')
            if k < best_k:
                best, best_free, best_k = i, free, k
                if k <= 1:
                    break                       # cannot do better than forced
        if best_k == 0:
            return False                        # a cell with no legal digit
        r, c = empty[best]
        if first_pick is not None and not first_pick:
            first_pick.append(((r, c), best_k))
        empty[best] = empty[-1]; empty.pop()    # O(1) removal
        b = box_of(r, c)
        free = best_free
        while free:
            bit = free & -free
            free ^= bit
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            g[r][c] = bit.bit_length() - 1
            if go():
                return True
            rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
            g[r][c] = 0
        empty.append((r, c))                    # put the cell back where it was
        empty[best], empty[-1] = empty[-1], empty[best]
        return False

    ok = go()
    return (g if ok else None), nodes


# =============================================================== LeetCode

class Solution78:
    """LeetCode 78 - Subsets."""

    def subsets(self, nums: list[int]) -> list[list[int]]:
        res, path = [], []

        def go(start: int) -> None:
            res.append(path[:])                  # the copy is the whole trick
            for i in range(start, len(nums)):
                path.append(nums[i])
                go(i + 1)
                path.pop()

        go(0)
        return res


class Solution39:
    """LeetCode 39 - Combination Sum."""

    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates = sorted(candidates)          # sorted, so we can stop early
        res, path = [], []

        def go(start: int, remaining: int) -> None:
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


def combination_sum_nodes(candidates: list[int], target: int, *, start_index=True,
                          sort_and_break=True) -> tuple[list[list[int]], int]:
    """Instrumented LeetCode 39 so the two knobs can be switched off:
    start_index=False restarts every level at candidate 0 (so [2,2,3] and
    [3,2,2] are both produced); sort_and_break=False keeps trying numbers
    that are already too big."""
    cand = sorted(candidates) if sort_and_break else list(candidates)
    res, path = [], []
    nodes = 0

    def go(start: int, remaining: int) -> None:
        nonlocal nodes
        nodes += 1
        if remaining == 0:
            res.append(path[:])
            return
        if remaining < 0:
            return
        for i in range(start if start_index else 0, len(cand)):
            if sort_and_break and cand[i] > remaining:
                break
            path.append(cand[i])
            go(i, remaining - cand[i])
            path.pop()

    go(0, target)
    return res, nodes


class Solution51:
    """LeetCode 51 - N-Queens."""

    def solveNQueens(self, n: int) -> list[list[str]]:
        cols, diag, anti = set(), set(), set()
        board: list[int] = []
        out: list[list[str]] = []

        def go(r: int) -> None:
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


class Solution52:
    """LeetCode 52 - N-Queens II (count only, bitmask)."""

    def totalNQueens(self, n: int) -> int:
        full = (1 << n) - 1

        def go(cols: int, ld: int, rd: int) -> int:
            if cols == full:
                return 1
            total = 0
            free = full & ~(cols | ld | rd)
            while free:
                bit = free & -free
                free ^= bit
                total += go(cols | bit, ((ld | bit) << 1) & full, (rd | bit) >> 1)
            return total

        return go(0, 0, 0)


class Solution37:
    """LeetCode 37 - Sudoku Solver (in place, MRV + bitmasks)."""

    def solveSudoku(self, board: list[list[str]]) -> None:
        rows = [0] * 9; cols = [0] * 9; boxes = [0] * 9
        empty = []
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    empty.append((r, c))
                else:
                    bit = 1 << int(board[r][c])
                    rows[r] |= bit; cols[c] |= bit; boxes[r // 3 * 3 + c // 3] |= bit

        def go() -> bool:
            if not empty:
                return True
            best, best_free, best_k = -1, 0, 10
            for i, (r, c) in enumerate(empty):
                free = ~(rows[r] | cols[c] | boxes[r // 3 * 3 + c // 3]) & 0x3FE
                k = bin(free).count('1')
                if k < best_k:
                    best, best_free, best_k = i, free, k
                    if k <= 1:
                        break
            if best_k == 0:
                return False
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


class Solution79:
    """LeetCode 79 - Word Search."""

    def exist(self, board: list[list[str]], word: str) -> bool:
        R, C = len(board), len(board[0])

        def dfs(r: int, c: int, i: int) -> bool:
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


def exist_no_restore(board: list[list[str]], word: str) -> bool:
    """LeetCode 79 with the unchoose line deleted (works on a copy)."""
    g = [row[:] for row in board]
    R, C = len(g), len(g[0])

    def dfs(r: int, c: int, i: int) -> bool:
        if i == len(word):
            return True
        if not (0 <= r < R and 0 <= c < C) or g[r][c] != word[i]:
            return False
        g[r][c] = '#'                                   # BUG: never put back
        return (dfs(r + 1, c, i + 1) or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1))

    return any(dfs(r, c, 0) for r in range(R) for c in range(C))


WORD_BOARD = [list('ABCE'), list('SFCS'), list('ADEE')]


def all_board_words(board: list[list[str]], max_len: int = 7) -> set[str]:
    """Every string spelled by a simple path of length <= max_len."""
    R, C = len(board), len(board[0])
    out: set[str] = set()

    def walk(r: int, c: int, seen: set, s: str) -> None:
        out.add(s)
        if len(s) == max_len:
            return
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and (rr, cc) not in seen:
                walk(rr, cc, seen | {(rr, cc)}, s + board[rr][cc])

    for r in range(R):
        for c in range(C):
            walk(r, c, {(r, c)}, board[r][c])
    return out


# ==================================================================== main

NAIVE_CAP = 200_000


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def main(full: bool = False) -> None:
    rule('1. the template: choose, explore, unchoose')
    trace: list = []
    subs = subsets([1, 2, 3], trace)
    print('  every call records the path it arrived with, then extends it:')
    for p in trace:
        print(f'    {"  " * len(p)}{p}')
    print(f'  -> {len(subs)} subsets, 2^3 = 8.  path.pop() is what lets the loop')
    print('     try [1, 3] after [1, 2]: the 2 has to come off first.')
    assert sorted(map(tuple, subs)) == sorted(
        tuple(x for x, keep in zip([1, 2, 3], bits) if keep)
        for bits in __import__('itertools').product([0, 1], repeat=3))

    rule('2. silent failure #1: storing the path instead of a copy')
    bad = subsets_aliased([1, 2, 3])
    print(f'  res.append(path)    -> {bad}')
    print(f'  len(res) = {len(bad)}  (correct!)   distinct list objects: {len({id(x) for x in bad})}')
    print('  all eight slots point at ONE list, and by the time the search')
    print('  returns, every pop() has emptied it.  A test that checks len passes.')
    assert len(bad) == 8 and all(x == [] for x in bad)

    rule('3. N-Queens: one queen per row, three sets for the attacks')
    st = {}
    sols = queens(8, st)
    print(f'  8 queens: {len(sols)} solutions.  The first one found:')
    for line in draw_board(sols[0]):
        print('    ' + line)
    print(f'  columns by row: {sols[0]}')
    print('  a square (r, c) is attacked along column c, the "\\" diagonal')
    print('  (same r - c) and the "/" diagonal (same r + c).')

    rule('4. what the pruning buys (8 x 8)')
    by_perm, checked = queens_by_permutation(8)
    ladder = [
        ('any 8 of the 64 squares', math.comb(64, 8)),
        ('one queen per row', 8 ** 8),
        ('one per row and per column (8!)', math.factorial(8)),
        ('backtracking with diagonal pruning', st['nodes']),
    ]
    for label, v in ladder:
        print(f'  {label:<36} {v:>15,}')
    print(f'  (the backtracking figure counts every node it enters, root included;')
    print(f'   it tested {st["tested"]:,} squares along the way.)')
    print(f'  generate-then-check over all 8! boards: {checked:,} boards for {by_perm} hits;')
    print(f'  the full permutation tree has {permutation_tree_nodes(8):,} nodes,')
    print(f'  {permutation_tree_nodes(8) / st["nodes"]:.0f}x more than the pruned search.')
    print('  n  solutions     nodes')
    for n in range(1, 11):
        s2 = {}
        k = len(queens(n, s2))
        print(f'  {n:>2} {k:>9} {s2["nodes"]:>9,}')
    assert by_perm == len(sols) == 92

    rule('5. silent failure #2: forgetting the unchoose')
    bad_sols, bad_nodes, seen = queens_no_undo(8)
    print(f'  board.pop() kept, the three set.remove() calls deleted')
    print(f'  -> {len(bad_sols)} solutions after {bad_nodes} nodes.  "8 queens is impossible."')
    print('  every partial board it entered:')
    for b in seen:
        print(f'    {b}')
    print('  each queen that is taken back leaves its column and diagonals')
    print('  marked, so the ghosts keep attacking squares nobody occupies.')
    assert bad_sols == [] and bad_nodes == 9

    rule('6. pruning has to be sound: the mirror-symmetry shortcut')
    print('  "try only the left half of row 0, then double the count"')
    print('   n   true   halved   verdict')
    for n in range(1, 11):
        t, h = len(queens(n)), queens_count_half(n)
        print(f'  {n:>2} {t:>6} {h:>8}   {"ok" if t == h else "WRONG"}')
    print('  exact on every even board, wrong on every odd board that has a')
    print('  solution (n = 3 has none to lose): the middle column is its own')
    print('  mirror image and never gets searched.')
    for n in range(1, 11):
        assert queens_count_half_fixed(n) == len(queens(n))

    rule('7. the bitmask version (Day 37): same tree, cheaper nodes')
    s_sets, s_bits = {}, {}
    (sol10, t_sets) = timed(queens, 10, s_sets)
    (cnt10, t_bits) = timed(queens_count_bits, 10, s_bits)
    print(f'  n = 10   sets: {len(sol10)} solutions, {s_sets["nodes"]:,} nodes, {t_sets * 1000:.0f} ms')
    print(f'           bits: {cnt10} solutions, {s_bits["nodes"]:,} nodes, {t_bits * 1000:.0f} ms')
    print('  identical node counts - the bitmask changes what a node costs,')
    print('  not how many there are.')
    assert s_sets['nodes'] == s_bits['nodes'] and cnt10 == len(sol10)

    rule('8. Sudoku: which cell you branch on beats how fast you check')
    print(f'  {"puzzle":<12}{"givens":>7}{"naive nodes":>14}{"MRV nodes":>11}{"ratio":>10}')
    rows_out = {}
    for name, s in PUZZLES.items():
        givens = sum(ch != '.' for ch in s)
        g1, n1 = sudoku_naive(s, cap=NAIVE_CAP)
        g2, n2 = sudoku_mrv(s)
        assert g2 is not None and is_valid_solution(g2, s)
        if g1 is None:
            n1_txt = f'>{NAIVE_CAP:,}'
            n1_full = ANTI_BRUTE_NAIVE_NODES
        else:
            assert g1 == g2
            n1_txt, n1_full = f'{n1:,}', n1
        rows_out[name] = (n1_full, n2)
        print(f'  {name:<12}{givens:>7}{n1_txt:>14}{n2:>11,}{n1_full / n2:>9,.0f}x')
    print(f'  anti-brute: the naive solver gave up at {NAIVE_CAP:,} nodes here;')
    print(f'  run to the end it visits {ANTI_BRUTE_NAIVE_NODES:,} nodes.')
    ab = sudoku_mrv(PUZZLES['anti-brute'])[0]
    print(f'  its top row is {"".join(map(str, ab[0]))} - naive tries 1 first in every cell.')
    pick: list = []
    sudoku_mrv(PUZZLES['anti-brute'], pick)
    (r, c), k = pick[0]
    print(f'  MRV\'s first branch: row {r}, column {c}, {k} candidate(s)')

    _, t_naive = timed(sudoku_naive, PUZZLES['Inkala'])
    _, t_bits = timed(sudoku_rowmajor_bits, PUZZLES['Inkala'])
    _, t_mrv = timed(sudoku_mrv, PUZZLES['Inkala'])
    print(f'  Inkala, wall clock: naive {t_naive * 1000:.0f} ms,'
          f' same order with bitmasks {t_bits * 1000:.0f} ms, MRV {t_mrv * 1000:.0f} ms')
    if full:
        g, n = sudoku_rowmajor_bits(PUZZLES['anti-brute'])
        print(f'  --full: naive order on anti-brute = {n:,} nodes')
        assert n == ANTI_BRUTE_NAIVE_NODES
    else:
        g, n = sudoku_rowmajor_bits(PUZZLES['Inkala'])
        assert n == rows_out['Inkala'][0]

    rule('9. LeetCode 78 - Subsets')
    print(f'  [1,2,3] -> {Solution78().subsets([1, 2, 3])}')

    rule('10. LeetCode 39 - Combination Sum')
    good, n_good = combination_sum_nodes([2, 3, 6, 7], 7)
    dup, n_dup = combination_sum_nodes([2, 3, 6, 7], 7, start_index=False)
    slow, n_slow = combination_sum_nodes([2, 3, 6, 7], 7, sort_and_break=False)
    print(f'  candidates [2,3,6,7], target 7 -> {Solution39().combinationSum([2, 3, 6, 7], 7)}')
    print(f'  with start index + sort/break : {len(good)} answers, {n_good} nodes')
    print(f'  no start index                : {len(dup)} answers {dup}, {n_dup} nodes')
    print(f'  no sort/break                 : {len(slow)} answers, {n_slow} nodes')
    big, n_big = combination_sum_nodes([2, 3, 5, 7, 11], 30)
    big2, n_big2 = combination_sum_nodes([2, 3, 5, 7, 11], 30, sort_and_break=False)
    print(f'  [2,3,5,7,11] -> 30: {len(big)} answers; {n_big:,} nodes with the break,'
          f' {n_big2:,} without')
    assert sorted(good) == [[2, 2, 3], [7]] and len(dup) == 4 and big == big2

    rule('11. LeetCode 51 / 52 - N-Queens')
    print(f'  n = 4 -> {Solution51().solveNQueens(4)}')
    print('  n : ' + '  '.join(f'{n}:{Solution52().totalNQueens(n)}' for n in range(1, 11)))

    rule('12. LeetCode 37 - Sudoku Solver')
    board = [list(PUZZLES['LeetCode 37'][r * 9:r * 9 + 9]) for r in range(9)]
    Solution37().solveSudoku(board)
    for row in board[:3]:
        print('   ' + ' '.join(row))
    print('    ...')
    assert is_valid_solution([[int(x) for x in row] for row in board], PUZZLES['LeetCode 37'])

    rule('13. LeetCode 79 - Word Search, and the missing restore')
    for w in ['ABCCED', 'SEE', 'ABCB', 'CCB']:
        print(f'  {w:<7} correct: {Solution79().exist([r[:] for r in WORD_BOARD], w)!s:<6}'
              f' no restore: {exist_no_restore(WORD_BOARD, w)}')
    words = all_board_words(WORD_BOARD)
    missed = sorted((w for w in words if not exist_no_restore(WORD_BOARD, w)),
                    key=lambda w: (len(w), w))
    print(f'  all three LeetCode examples still pass without the restore, but of')
    print(f'  the {len(words):,} strings this board really spells (paths up to 7 cells),')
    print(f'  the broken version misses {len(missed)}; the shortest: {missed[:4]}')
    assert all(Solution79().exist([r[:] for r in WORD_BOARD], w) for w in words)

    rule('all assertions passed')


if __name__ == '__main__':
    main(full='--full' in sys.argv)
