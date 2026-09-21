"""Day 36 - interval DP and tree DP: two shapes where the loop order is the algorithm.

Every DP so far filled a table left to right.  Today both families break that
habit:

  * interval DP fills a triangle **by interval length**, because dp[i][j]
    depends on shorter intervals that live to its left and below it - iterating
    i and j in the natural order reads cells that are still zero, and Python
    says nothing;
  * tree DP "fills" the table in post-order, because a node depends on its
    children - and on a 10^5-node path graph the recursive version dies of
    RecursionError, which is the one silent failure of the week that is
    actually loud.

Run me:  python interval_tree_dp.py
"""

import sys
from functools import lru_cache

# --------------------------------------------------------------------------
# the running examples
# --------------------------------------------------------------------------
DIMS = [40, 20, 30, 10, 30]      # 4 matrices: 40x20, 20x30, 30x10, 10x30
BALLOONS = [3, 1, 5, 8]
STICK_N, CUTS = 7, [1, 3, 4, 5]


def rule(title):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


# ==========================================================================
# 1. matrix chain multiplication - the canonical interval DP
# ==========================================================================
def chain_cost(dims, order):
    """Cost of one explicit parenthesisation, for checking the DP by hand.

    `order` is a nested tuple of matrix indices, e.g. ((0, 1), (2, 3)).
    Returns (rows, cols, scalar multiplications).
    """
    if isinstance(order, int):
        return dims[order], dims[order + 1], 0
    r1, c1, k1 = chain_cost(dims, order[0])
    r2, c2, k2 = chain_cost(dims, order[1])
    assert c1 == r2, 'shapes do not line up'
    return r1, c2, k1 + k2 + r1 * c1 * c2


def matrix_chain(dims):
    """dp[i][j] = cheapest way to multiply matrices i..j (inclusive).

    A product of matrices i..j is, at the very top, a product of exactly two
    things: i..k and k+1..j.  So the recurrence splits on that last
    multiplication:

        dp[i][j] = min over k of  dp[i][k] + dp[k+1][j] + rows(i)*cols(k)*cols(j)

    Both sub-intervals are strictly shorter than i..j, which is the whole
    reason the loop is over *length*.
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for length in range(2, n + 1):                 # <- the outer loop is length
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    return dp, split


def matrix_chain_naive_order(dims):
    """The same body with the natural i-then-j loops.  Wrong, and silent.

    dp[i][j] reads dp[k+1][j] with k+1 > i - a *later* row, which has not been
    filled yet and is still 0.  Nothing raises; the answer is simply too small.
    """
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1]


def parens(split, i, j, names=None):
    """Rebuild the parenthesisation from the split points."""
    if names is None:
        names = [chr(ord('A') + t) for t in range(len(split))]
    if i == j:
        return names[i]
    k = split[i][j]
    return '(' + parens(split, i, k, names) + parens(split, k + 1, j, names) + ')'


def chain_order(split, i, j):
    """The same reconstruction, as a nested tuple for chain_cost()."""
    if i == j:
        return i
    k = split[i][j]
    return (chain_order(split, i, k), chain_order(split, k + 1, j))


def matrix_chain_fill_order(n):
    """The (i, j) cells in the order the correct loop visits them."""
    out = []
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            out.append((i, i + length - 1))
    return out


# ==========================================================================
# 2. LeetCode 312 - burst balloons: split on the LAST balloon
# ==========================================================================
def burst_first_wrong(nums):
    """The instinctive recurrence: choose which balloon to pop FIRST.

    Pop k first, collect a[k-1] * a[k] * a[k+1] using the neighbours it has
    *right now*, then solve the two sides separately.  It does not decompose:
    once k is gone the left side's right-hand neighbour is a balloon from the
    right side, so the two halves are not independent and the coins they can
    still earn are not the ones this recurrence assumes.  It returns 98 on the
    example instead of 167 - no exception, just a smaller number.
    """
    a = [1] + list(nums) + [1]

    @lru_cache(maxsize=None)
    def go(i, j):                          # inclusive, in the padded array
        if i > j:
            return 0
        return max(a[k - 1] * a[k] * a[k + 1] + go(i, k - 1) + go(k + 1, j)
                   for k in range(i, j + 1))

    return go(1, len(nums))


def burst_last(nums):
    """The correct recurrence: choose which balloon is popped LAST in (i, j).

    If k is last inside the open interval (i, j), then by the time k pops
    everything strictly between i and j is gone, so its neighbours are exactly
    the fixed walls i and j.  The two sides never interact, and the
    subproblems are genuinely independent.
    """
    a = [1] + list(nums) + [1]
    n = len(a)
    dp = [[0] * n for _ in range(n)]
    choice = [[-1] * n for _ in range(n)]
    for length in range(2, n):                 # length = j - i, at least 2
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                cand = dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]
                if cand > dp[i][j]:
                    dp[i][j] = cand
                    choice[i][j] = k
    return dp[0][n - 1], dp, choice


def burst_order(choice, i, j):
    """The actual popping order, read out of the choice table."""
    k = choice[i][j]
    if k == -1:
        return []
    return burst_order(choice, i, k) + burst_order(choice, k, j) + [k]


def burst_brute(nums):
    """Every popping order, for small inputs.  The ground truth."""
    best = 0
    def go(cur, gained):
        nonlocal best
        if not cur:
            best = max(best, gained)
            return
        for i in range(len(cur)):
            left = cur[i - 1] if i else 1
            right = cur[i + 1] if i + 1 < len(cur) else 1
            go(cur[:i] + cur[i + 1:], gained + left * cur[i] * right)
    go(list(nums), 0)
    return best


# ==========================================================================
# 3. LeetCode 1547 - minimum cost to cut a stick: the same shape
# ==========================================================================
def cut_cost_in_order(n, cuts):
    """Cost of performing the cuts in the order they are given - the baseline."""
    pieces = [(0, n)]
    total = 0
    for c in cuts:
        for idx, (lo, hi) in enumerate(pieces):
            if lo < c < hi:
                total += hi - lo
                pieces[idx:idx + 1] = [(lo, c), (c, hi)]
                break
    return total


def min_cost_cuts(n, cuts):
    """Add the two ends as fake cuts, sort, and it is matrix chain again.

    dp[i][j] = cheapest way to make every cut strictly between points i and j.
    The *first* cut made inside a piece costs the whole length of that piece,
    and splits it into two pieces that never interact again - so here the
    natural "which one first" framing is the one that decomposes.
    """
    pts = sorted(cuts) + [0, n]
    pts.sort()
    m = len(pts)
    dp = [[0] * m for _ in range(m)]
    for length in range(2, m):
        for i in range(m - length):
            j = i + length
            dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i + 1, j)) + pts[j] - pts[i]
    return dp[0][m - 1]


# ==========================================================================
# 4. LeetCode 516 - longest palindromic subsequence, two ways
# ==========================================================================
def lps_interval(s):
    """Interval DP: dp[i][j] = the LPS of s[i..j]."""
    n = len(s)
    if n == 0:
        return 0, ''
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    # backtrack for the palindrome itself
    out, i, j = [], 0, n - 1
    while i < j:
        if s[i] == s[j]:
            out.append(s[i]); i += 1; j -= 1
        elif dp[i + 1][j] >= dp[i][j - 1]:
            i += 1
        else:
            j -= 1
    mid = [s[i]] if i == j else []
    return dp[0][n - 1], ''.join(out) + ''.join(mid) + ''.join(reversed(out))


def lps_via_lcs(s):
    """Day 35's LCS, run against the reversed string.  Same number.

    A palindromic subsequence of s is a subsequence that also appears in
    reversed(s), so LPS(s) = LCS(s, reversed(s)).  Identical O(n^2), and it is
    worth knowing because it turns a "new" problem into yesterday's.
    """
    x, y = s, s[::-1]
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(x)][len(y)]


# ==========================================================================
# 5. tree DP - a tiny binary tree, and the take/skip pair of states
# ==========================================================================
class Node:
    __slots__ = ('val', 'left', 'right')

    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def build_tree():
    """     3
          /   \\
         2     3
          \\     \\
           3     1        LeetCode 337's first example: the answer is 7.
    """
    return Node(3, Node(2, None, Node(3)), Node(3, None, Node(1)))


def rob_tree(root):
    """LeetCode 337.  Each node returns a PAIR: (robbed, skipped).

    One number per node is not enough state - "the best for this subtree"
    cannot tell the parent whether this node was used, and the parent's
    decision depends on exactly that.  Two numbers make the states independent
    again.  Same lesson as day 35's "dp[i] must end at i".
    """
    def go(node):
        if node is None:
            return 0, 0
        lr, ls = go(node.left)
        rr, rs = go(node.right)
        robbed = node.val + ls + rs           # take me: children must be skipped
        skipped = max(lr, ls) + max(rr, rs)   # skip me: children are free
        return robbed, skipped
    return max(go(root))


def rob_pair(node):
    """The (robbed, skipped) pair for one node - what `rob_tree` computes
    internally.  Exposed so the figures can print real numbers."""
    if node is None:
        return 0, 0
    lr, ls = rob_pair(node.left)
    rr, rs = rob_pair(node.right)
    return node.val + ls + rs, max(lr, ls) + max(rr, rs)


def rob_tree_one_state(root):
    """The wrong version: one number per node, "the best for this subtree".

    That single number cannot say whether the subtree's own root was used, so
    the parent has no way to know that adding itself is illegal - and with
    positive values `max` always prefers taking the node, so every node ends up
    taken, parents and children together.  It returns 12 instead of 7, which is
    just "the sum of everything" wearing a DP costume.
    """
    def go(node):
        if node is None:
            return 0
        below = go(node.left) + go(node.right)
        return max(node.val + below, below)
    return go(root)


def max_path_sum(root):
    """LeetCode 124.  The same split: what a node RETURNS is not what it SCORES.

    A node returns the best downward path through it (at most one child), but
    the answer it contributes may bend at the node and use both children.
    Mixing those two up is the entire difficulty of the problem.
    """
    best = float('-inf')

    def down(node):
        nonlocal best
        if node is None:
            return 0
        left = max(down(node.left), 0)         # a negative branch is worth skipping
        right = max(down(node.right), 0)
        best = max(best, node.val + left + right)   # bend here
        return node.val + max(left, right)          # go through here
    down(root)
    return best


def diameter(root):
    """The same pattern again: return depth, score depth(left)+depth(right)."""
    best = 0

    def depth(node):
        nonlocal best
        if node is None:
            return 0
        l, r = depth(node.left), depth(node.right)
        best = max(best, l + r)
        return 1 + max(l, r)
    depth(root)
    return best


# ==========================================================================
# 6. tree knapsack - a DP whose complexity argument is the interesting part
# ==========================================================================
COURSES = {
    0: [1, 2],          # 0 is a virtual root; every real course hangs off it
    1: [3, 4],
    2: [5],
    3: [], 4: [], 5: [],
}
CREDITS = [0, 2, 3, 4, 1, 5]      # credit of each course; course 0 is free


def tree_knapsack(children, value, root, budget):
    """Pick `budget` courses; a course may only be taken with its prerequisite.

    dp[v] is a list: dp[v][t] = best value from v's subtree taking exactly t
    courses, v included.  Merging a child is a small knapsack over the sizes
    already accumulated - and the loop bound `min(size[v], budget)` is not
    an optimisation, it is what makes the whole thing O(n * budget) instead of
    O(n * budget^2).
    """
    size = {}

    def go(v):
        dp = [0] * (budget + 1)
        for t in range(1, budget + 1):
            dp[t] = value[v]
        size[v] = 1
        for c in children[v]:
            cdp = go(c)
            nd = list(dp)
            for t in range(min(size[v], budget), 0, -1):
                for u in range(1, min(size[c], budget - t) + 1):
                    if dp[t] + cdp[u] > nd[t + u]:
                        nd[t + u] = dp[t] + cdp[u]
            dp = nd
            size[v] += size[c]
        return dp

    return go(root)[budget], size


def pair_count(children, root):
    """Why the naive-looking merge is O(n^2): every PAIR of nodes is counted once.

    The inner double loop at a node costs size(child) * size(already merged) -
    which is exactly the number of (u, v) pairs whose lowest common ancestor is
    this node.  Summed over the tree, every pair is charged exactly once, so
    the total is C(n, 2).
    """
    total = 0
    size = {}

    def go(v):
        acc = 1
        for c in children[v]:
            sc = go(c)
            nonlocal total
            total += acc * sc          # pairs whose LCA is v
            acc += sc
        size[v] = acc
        return acc

    n = go(root)
    return total, n * (n - 1) // 2


# ==========================================================================
# 7. the depth trap - recursion dies on a path graph
# ==========================================================================
def path_tree(n):
    """children lists for 0 -> 1 -> 2 -> ... -> n-1, the worst case for depth."""
    return {i: ([i + 1] if i + 1 < n else []) for i in range(n)}


def subtree_sums_recursive(children, value, root):
    def go(v):
        return value[v] + sum(go(c) for c in children[v])
    return go(root)


def subtree_sums_iterative(children, value, root):
    """Post-order without the call stack: push, then pop in reverse.

    The first pass produces a valid topological order of the tree (parents
    before children); reversing it gives children before parents, which is
    exactly what a tree DP needs.  No recursion limit, no C stack.
    """
    order, stack = [], [root]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(children[v])
    total = {}
    for v in reversed(order):
        total[v] = value[v] + sum(total[c] for c in children[v])
    return total[root]


# ==========================================================================
def main():
    rule('1. matrix chain multiplication - split on the LAST product')
    dp, split = matrix_chain(DIMS)
    n = len(DIMS) - 1
    print(f'  dims      = {DIMS}   ({n} matrices)')
    for t in range(n):
        print(f'    {chr(ord("A") + t)} : {DIMS[t]:>3d} x {DIMS[t + 1]:<3d}')
    best = dp[0][n - 1]
    order = chain_order(split, 0, n - 1)
    print(f'  best cost = {best:,}   {parens(split, 0, n - 1)}')
    worst_order = (0, (1, (2, 3)))
    print(f'  right-to-left  {chain_cost(DIMS, worst_order)[2]:,}   (A(B(CD)))')
    print(f'  left-to-right  {chain_cost(DIMS, (((0, 1), 2), 3))[2]:,}   (((AB)C)D)')
    assert chain_cost(DIMS, order)[2] == best

    rule('2. the loop order IS the algorithm')
    cells = matrix_chain_fill_order(n)
    print(f'  correct visiting order (by interval length):')
    print(f'    {cells}')
    print(f'  naive i-then-j order gives {matrix_chain_naive_order(DIMS):,} '
          f'instead of {best:,}')
    print('  because dp[i][j] reads dp[k+1][j] - a LOWER row, not yet filled.')
    assert matrix_chain_naive_order(DIMS) < best

    rule('3. LeetCode 312 - burst balloons')
    got, bdp, choice = burst_last(BALLOONS)
    brute = burst_brute(BALLOONS)
    order = burst_order(choice, 0, len(BALLOONS) + 1)
    print(f'  balloons          = {BALLOONS}')
    print(f'  brute force (4! orders) = {brute}')
    print(f'  "last one in the interval" DP = {got}   order {[BALLOONS[k - 1] for k in order]}')
    print(f'  "first one" recurrence        = {burst_first_wrong(BALLOONS)}  <- wrong, and silent')
    assert got == brute == 167
    assert burst_first_wrong(BALLOONS) == 98

    rule('4. LeetCode 1547 - minimum cost to cut a stick')
    print(f'  stick of length {STICK_N}, cuts at {CUTS}')
    print(f'  cutting in the given order = {cut_cost_in_order(STICK_N, CUTS)}')
    print(f'  interval DP (best order)   = {min_cost_cuts(STICK_N, CUTS)}')
    assert min_cost_cuts(STICK_N, CUTS) == 16
    assert min_cost_cuts(9, [5, 6, 1, 4, 2]) == 22

    rule('5. LeetCode 516 - longest palindromic subsequence')
    s = 'character'
    length, pal = lps_interval(s)
    print(f'  s        = {s!r}')
    print(f'  LPS      = {length}  ({pal!r})')
    print(f'  via day 35 LCS(s, reversed s) = {lps_via_lcs(s)}')
    assert lps_via_lcs(s) == length == 5
    assert lps_interval('bbbab')[0] == 4

    rule('6. tree DP - one number per node is not enough state')
    root = build_tree()
    print('        3')
    print('      /   \\')
    print('     2     3')
    print('      \\     \\')
    print('       3     1')
    print(f'  rob_tree (take/skip pair) = {rob_tree(root)}')
    print(f'  one number per node       = {rob_tree_one_state(root)}  <- takes a parent and its child')
    print(f'  max path sum (LC 124)     = {max_path_sum(root)}')
    print(f'  diameter in edges         = {diameter(root)}')
    assert rob_tree(root) == 7
    assert rob_tree_one_state(root) == 12   # = 3 + 2 + 3 + 3 + 1, everything
    assert max_path_sum(Node(-10, Node(9), Node(20, Node(15), Node(7)))) == 42

    rule('7. tree knapsack - and why the double loop is O(n^2), not O(n^3)')
    for m in range(1, 6):
        val, _ = tree_knapsack(COURSES, CREDITS, 0, m + 1)
        print(f'  take {m} real course(s): best credits = {val}')
    pairs, expected = pair_count(COURSES, 0)
    print(f'  merge work counted as pairs = {pairs} = C(n,2) = {expected}')
    assert pairs == expected
    assert tree_knapsack(COURSES, CREDITS, 0, 3)[0] == 8
    assert tree_knapsack(COURSES, CREDITS, 0, 5)[0] == 14

    rule('8. the depth trap')
    ch = path_tree(60_000)
    val = [1] * 60_000
    print(f'  a 60,000-node path graph, sys.getrecursionlimit() = {sys.getrecursionlimit()}')
    try:
        subtree_sums_recursive(ch, val, 0)
        print('  recursive: survived (unexpected)')
    except RecursionError:
        print('  recursive: RecursionError')
    print(f'  iterative post-order: {subtree_sums_iterative(ch, val, 0):,}')
    assert subtree_sums_iterative(ch, val, 0) == 60_000

    print()
    print('all checks passed.')


if __name__ == '__main__':
    main()
