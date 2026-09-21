"""Figures for Day 36 - interval DP and tree DP.

Every number on these figures is computed by importing interval_tree_dp.py.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from diagram_style import *            # noqa
import interval_tree_dp as T           # noqa

DIMS = T.DIMS
NAMES = ['A', 'B', 'C', 'D']


# --------------------------------------------------------------------------
def fig1():
    """The triangular table, and the order it must be filled in."""
    n = len(DIMS) - 1
    dp, split = T.matrix_chain(DIMS)
    order = T.matrix_chain_fill_order(n)
    rank = {cell: i + 1 for i, cell in enumerate(order)}
    k = split[0][n - 1]

    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    text(ax, 4.6, 5.26, 'dp[i][j] needs dp[i][k] and dp[k+1][j] - both shorter',
         color=TEAL_L, size=15, glowing=False)

    cw, ch = 1.05, 0.78
    x0, y0 = 2.55, 3.90
    reads = {(0, k), (k + 1, n - 1)}
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * cw, y0 - i * ch
            if j < i:
                continue
            if i == j:
                box(ax, x, y, cw, ch, label='0',
                    fill=GREY, edge=AMBER if (i, j) in reads else GREY_L,
                    lw=2.6 if (i, j) in reads else 1.3, size=13)
            else:
                lab = f'{rank[(i, j)]}   {dp[i][j]:,}'
                hot = (i, j) == (0, n - 1)
                box(ax, x, y, cw, ch, label=lab, fill=NAVY,
                    edge=AMBER if (hot or (i, j) in reads) else TEAL,
                    lw=2.6 if (hot or (i, j) in reads) else 1.5, size=11)
    for j in range(n):
        text(ax, x0 + j * cw + cw / 2, y0 + ch + 0.22, f'j={j}', color=TEAL,
             size=12, glowing=False)
    for i in range(n):
        text(ax, x0 - 0.22, y0 - i * ch + ch / 2, f'i={i}', color=TEAL,
             size=12, ha='right', glowing=False)
    text(ax, 0.20, 4.12, 'step  cost', color=TEAL, size=12, ha='left',
         glowing=False)

    text(ax, 4.6, 1.34,
         'the step number is the order the correct loop visits the cell',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 4.6, 0.92,
         f'amber: dp[0][{n - 1}] = {dp[0][n - 1]:,} is built from dp[0][{k}] and '
         f'dp[{k + 1}][{n - 1}], k = {k}',
         color=AMBER, size=13, glowing=False)
    text(ax, 4.6, 0.46,
         'looping i then j would fill step 4 before steps 2 and 3, and read zeros',
         color=RED, size=13, glowing=False)
    save(fig, 'day36_1.png')


# --------------------------------------------------------------------------
def span(order):
    """(lo, hi) matrix range covered by a nested order."""
    if isinstance(order, int):
        return order, order
    return span(order[0])[0], span(order[1])[1]


def steps(order):
    """Every multiplication of a parenthesisation, in the order it happens."""
    out = []

    def go(o):
        if isinstance(o, int):
            return
        go(o[0]); go(o[1])
        lo, hi = span(o)
        r1, c1, k1 = T.chain_cost(DIMS, o[0])
        r2, c2, k2 = T.chain_cost(DIMS, o[1])
        out.append((lo, hi, r1 * c1 * c2))

    go(order)
    return out


def fig2():
    """Two parenthesisations of the same chain, two very different bills."""
    good = T.chain_order(T.matrix_chain(DIMS)[1], 0, 3)
    bad = (0, (1, (2, 3)))
    gcost = T.chain_cost(DIMS, good)[2]
    bcost = T.chain_cost(DIMS, bad)[2]

    fig, ax = canvas(9.2, 6.8, xlim=(0, 9.2), ylim=(0, 6.8))
    text(ax, 4.6, 6.46, 'same four matrices, same result, different bill',
         color=TEAL_L, size=15, glowing=False)

    x0, cw = 1.70, 1.62
    for t, nm in enumerate(NAMES):
        box(ax, x0 + t * cw, 5.55, cw - 0.10, 0.62,
            label=f'{nm}  {DIMS[t]}x{DIMS[t + 1]}', fill=NAVY, edge=TEAL,
            lw=1.6, size=12)

    def plan(order, cost, top, color, title):
        text(ax, 0.20, top + 0.26, title, color=color, size=13, ha='left',
             glowing=False)
        y = top
        for lo, hi, c in steps(order):
            x1 = x0 + lo * cw
            x2 = x0 + hi * cw + cw - 0.10
            label = ''.join(NAMES[lo:hi + 1]) if hi > lo else NAMES[lo]
            box(ax, x1, y, x2 - x1, 0.52, label=f'{label}   {c:,}', fill=NAVY,
                edge=color, lw=2.2, size=12)
            y -= 0.68
        text(ax, 0.20, top - 0.16, f'= {cost:,}', color=color, size=14,
             ha='left', glowing=False)

    plan(good, gcost, 4.62, AMBER, '(A(BC))D')
    plan(bad, bcost, 2.28, RED, 'A(B(CD))')
    text(ax, 4.6, 0.30, f'the good order is {bcost / gcost:.2f}x cheaper, '
         f'and both orders compute the same 40x30 matrix',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day36_2.png')


# --------------------------------------------------------------------------
def fig3():
    """Burst balloons: first is not separable, last is."""
    nums = T.BALLOONS
    got, _, choice = T.burst_last(nums)
    wrong = T.burst_first_wrong(nums)
    fig, ax = canvas(9.2, 5.8, xlim=(0, 9.2), ylim=(0, 5.8))
    text(ax, 4.6, 5.46, 'which balloon do you split the interval on?',
         color=TEAL_L, size=15, glowing=False)

    cw = 0.86

    def strip(y, vals, hot=None, grey=()):
        cx = []
        x0 = 4.6 - len(vals) * cw / 2
        for i, v in enumerate(vals):
            eg = AMBER if i == hot else (GREY_L if i in grey else TEAL)
            box(ax, x0 + i * cw, y, cw, 0.60, label=str(v), fill=NAVY,
                edge=eg, lw=2.6 if i == hot else 1.5, size=13)
            cx.append(x0 + i * cw + cw / 2)
        return cx

    text(ax, 0.20, 4.62, 'pop it FIRST', color=RED, size=13, ha='left',
         glowing=False)
    cx = strip(4.32, [1] + nums + [1], hot=3, grey=(0, 5))
    text(ax, 4.6, 3.90,
         'after 5 goes, 1 and 8 become neighbours - the halves are not independent',
         color=RED, size=13, glowing=False)
    cross(ax, cx[3], 3.46, r=0.22)
    text(ax, 4.6, 3.02, f'the "first" recurrence returns {wrong}',
         color=RED, size=13, glowing=False)

    text(ax, 0.20, 2.32, 'pop it LAST', color=AMBER, size=13, ha='left',
         glowing=False)
    cx = strip(2.02, [1] + nums + [1], hot=3, grey=(0, 5))
    text(ax, 4.6, 1.60,
         'if 5 is last, everything between the walls is gone, so its neighbours',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 4.6, 1.22,
         'are exactly the walls 1 and 1 - the two sides never interact',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 4.6, 0.62,
         f'dp[i][j] = max over k of dp[i][k] + dp[k][j] + a[i]*a[k]*a[j]  ->  {got}',
         color=AMBER, size=13, glowing=False)
    save(fig, 'day36_3.png')


# --------------------------------------------------------------------------
def fig4():
    """Tree DP: one number per node is not enough state."""
    root = T.build_tree()
    good = T.rob_tree(root)
    bad = T.rob_tree_one_state(root)

    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    text(ax, 4.6, 5.26, 'every node reports TWO numbers, not one',
         color=TEAL_L, size=15, glowing=False)

    P = {'r': (4.6, 4.35), 'l': (2.9, 3.35), 'rr': (6.3, 3.35),
         'lr': (3.9, 2.35), 'rrr': (7.3, 2.35)}
    E = [('r', 'l'), ('r', 'rr'), ('l', 'lr'), ('rr', 'rrr')]
    for a, b in E:
        edge(ax, P[a], P[b], directed=False, r=0.40)
    nodes = {'r': root, 'l': root.left, 'rr': root.right,
             'lr': root.left.right, 'rrr': root.right.right}
    labels = {k: str(n.val) for k, n in nodes.items()}
    pairs = {k: '({}, {})'.format(*T.rob_pair(n)) for k, n in nodes.items()}
    for key, (x, y) in P.items():
        hot = key == 'r'
        node(ax, x, y, labels[key], r=0.40, edge=AMBER if hot else TEAL,
             lw=2.4 if hot else 1.8, size=14)
        text(ax, x + 0.56, y + 0.02, pairs[key], color=AMBER if hot else TEAL,
             size=12, ha='left', glowing=False)
    text(ax, 4.6, 1.52, '(take this node, skip this node)', color=AMBER,
         size=13, glowing=False)
    text(ax, 4.6, 1.10,
         'take = val + both children SKIPPED;  skip = best of each child, free',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 4.6, 0.62,
         f'answer = max(take, skip) = {good};  one number per node gives {bad}',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day36_4.png')


# --------------------------------------------------------------------------
def fig5():
    """Tree knapsack: the double loop costs one unit per pair of nodes."""
    pairs, expected = T.pair_count(T.COURSES, 0)
    fig, ax = canvas(9.2, 5.4, xlim=(0, 9.2), ylim=(0, 5.4))
    text(ax, 4.6, 5.06, 'the merge loop charges each PAIR of nodes exactly once',
         color=TEAL_L, size=15, glowing=False)

    P = {0: (6.0, 4.20), 1: (4.6, 3.20), 2: (7.8, 3.20),
         3: (3.8, 2.20), 4: (5.5, 2.20), 5: (7.8, 2.20)}
    for v, kids in T.COURSES.items():
        for c in kids:
            edge(ax, P[v], P[c], directed=False, r=0.38)
    for v, (x, y) in P.items():
        hot = v == 0
        node(ax, x, y, str(v), r=0.38, edge=AMBER if hot else TEAL,
             lw=2.4 if hot else 1.8, size=13)
    text(ax, 6.0, 4.72, 'root', color=AMBER, size=12, glowing=False)
    text(ax, 0.25, 3.20, 'merging child c costs', color=TEAL_L, size=12,
         ha='left', glowing=False)
    text(ax, 0.25, 2.80, 'size(c) x size(merged so far)', color=AMBER, size=12,
         ha='left', glowing=False)
    text(ax, 4.6, 1.50,
         'those products are the pairs (u, v) whose lowest common ancestor is this node',
         color=TEAL_L, size=13, glowing=False)
    text(ax, 4.6, 1.04,
         f'summed over the tree: {pairs} = C({len(T.COURSES)}, 2) = {expected}',
         color=AMBER, size=13, glowing=False)
    text(ax, 4.6, 0.56,
         'so the whole tree knapsack is O(n^2), not O(n * budget^2)',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day36_5.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures written')
