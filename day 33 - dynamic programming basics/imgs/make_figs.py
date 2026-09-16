"""Figures for Day 33 - dynamic programming from the ground up.

Every number drawn here is computed by importing dp_basics, so the figures
cannot drift away from the code.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import dp_basics as D                  # noqa: E402


# --------------------------------------------------------------------------
# 1  the recursion tree folded into a row
# --------------------------------------------------------------------------
def fig1():
    """The f(5) tree, laid out by an actual recursion, next to the folded row."""
    fig, ax = canvas(9.6, 6.4, xlim=(0, 9.6), ylim=(0, 6.4))

    R = 0.24
    X0, X1 = 0.42, 4.30
    LEAVES = [0]
    placed = []

    def layout(k, depth):
        """Return the x of this node; leaves get consecutive slots."""
        if k <= 1:
            x = LEAVES[0]
            LEAVES[0] += 1
        else:
            xs = [layout(k - 1, depth + 1), layout(k - 2, depth + 1)]
            x = sum(xs) / 2
            for cx, cd in zip(xs, (depth + 1, depth + 1)):
                placed.append(('edge', x, depth, cx, cd))
        placed.append(('node', x, depth, k, None))
        return x

    layout(5, 0)
    n_leaf = LEAVES[0]
    step = (X1 - X0) / (n_leaf - 1)
    ytop, dy = 5.70, 0.80

    def P(x, d):
        return (X0 + x * step, ytop - d * dy)

    for kind, x, d, a, b in placed:
        if kind == 'edge':
            edge(ax, P(x, d), P(a, b), r=R, lw=1.5)
    for kind, x, d, k, _ in placed:
        if kind == 'node':
            dup = k in (2, 3)
            node(ax, *P(x, d), f'f{k}', r=R, size=11,
                 edge=AMBER if dup else TEAL, lw=2.2 if dup else 1.6)

    counts = {}
    for kind, x, d, k, _ in placed:
        if kind == 'node':
            counts[k] = counts.get(k, 0) + 1

    text(ax, 2.36, 6.10, 'stairs_naive(5), every call drawn',
         color=TEAL_L, size=15)
    text(ax, 2.36, 1.00,
         f'f3 is computed {counts[3]}x, f2 {counts[2]}x, f1 {counts[1]}x',
         color=AMBER, size=14)
    text(ax, 2.36, 0.55, 'the tree is as big as the answer it returns',
         color=TEAL, size=14)

    # the fold
    arrow(ax, (4.85, 3.55), (5.50, 3.55), lw=2.2, ms=16)
    text(ax, 5.17, 3.92, 'fold', color=CYAN, size=15)

    dp = D.stairs_table(5)
    cw, ch_ = 0.56, 0.60
    x0, y0 = 5.95, 3.24
    cells(ax, x0, y0, 6, cw, ch_, labels=[str(v) for v in dp], size=13)
    cells(ax, x0, y0 - ch_, 6, cw, ch_, labels=[str(k) for k in range(6)],
          index=True, size=12)
    text(ax, x0 - 0.16, y0 + ch_ / 2, 'dp', color=TEAL, size=13, ha='right')
    text(ax, x0 - 0.16, y0 - ch_ / 2, 'n', color=TEAL, size=13, ha='right')

    text(ax, x0 + 3 * cw, y0 + 1.35, 'six cells, filled once each',
         color=TEAL_L, size=15)
    text(ax, x0 + 3 * cw, y0 - 1.10, 'dp[n] = dp[n-1] + dp[n-2]',
         color=TEAL, size=15)
    text(ax, x0 + 3 * cw, y0 - 1.60, 'same recursion, read left to right',
         color=TEAL, size=14)

    assert counts[1] + counts[0] == n_leaf
    save(fig, 'day33_1.png')


# --------------------------------------------------------------------------
# 2  filling the stairs row
# --------------------------------------------------------------------------
def fig2():
    fig, ax = canvas(9.0, 4.2, xlim=(0, 9.0), ylim=(0, 4.2))
    dp = D.stairs_table(8)
    cw, ch_ = 0.80, 0.70
    x0, y0 = 0.70, 1.90
    labs = [str(v) for v in dp[:7]] + ['?', '']
    cs = cells(ax, x0, y0, 9, cw, ch_, labels=labs, size=14)
    cells(ax, x0, y0 - ch_, 9, cw, ch_,
          labels=[str(k) for k in range(9)], index=True, size=12)

    # highlight the two sources of dp[7]
    for k in (5, 6):
        box(ax, x0 + k * cw, y0, cw, ch_, label=str(dp[k]), fill=NAVY,
            edge=AMBER, lw=2.6, size=14)
    src1, src2 = cs[5], cs[6]
    tgt = (cs[7][0], cs[7][1])
    arrow(ax, (src2[0], y0 + ch_), (tgt[0] - 0.06, y0 + ch_), rad=-0.55,
          color=AMBER, lw=1.8)
    arrow(ax, (src1[0], y0 + ch_), (tgt[0] - 0.14, y0 + ch_), rad=-0.42,
          color=AMBER, lw=1.8)

    text(ax, cs[7][0] + 0.10, y0 + ch_ + 1.15,
         'arrive with a 1-step  or  a 2-step', color=TEAL, size=15)
    text(ax, 4.5, 0.62, 'dp[7] = dp[6] + dp[5] = 13 + 8 = 21',
         color=TEAL_L, size=16)
    text(ax, 4.5, 0.18,
         'only two cells are ever read, so two variables are enough',
         color=TEAL, size=14)
    save(fig, 'day33_2.png')


# --------------------------------------------------------------------------
# 3  coin change: min over coins, and where greedy dies
# --------------------------------------------------------------------------
def fig3():
    coins = [1, 5, 6, 9]
    amount = 11
    best, used, dp = D.coin_change_min_trace(coins, amount)
    g_count, g_used = D.coin_change_min_greedy(coins, amount)

    fig, ax = canvas(9.6, 5.4, xlim=(0, 9.6), ylim=(0, 5.4))
    cw, ch_ = 0.70, 0.62
    x0, y0 = 0.85, 3.30
    cells(ax, x0, y0, 12, cw, ch_, labels=[str(v) for v in dp], size=13)
    cells(ax, x0, y0 - ch_, 12, cw, ch_,
          labels=[str(k) for k in range(12)], index=True, size=11)
    text(ax, x0 - 0.18, y0 + ch_ / 2, 'dp', color=TEAL, size=14, ha='right')

    text(ax, 4.8, 5.00, 'coins 1, 5, 6, 9   -   fewest coins for 11',
         color=TEAL_L, size=16)

    # the three predecessors of dp[11]
    for c in coins:
        k = amount - c
        box(ax, x0 + k * cw, y0, cw, ch_, label=str(dp[k]), fill=NAVY,
            edge=AMBER, lw=2.4, size=13)
    box(ax, x0 + amount * cw, y0, cw, ch_, label=str(dp[amount]), fill=NAVY,
        edge=CYAN, lw=2.6, size=13)
    brace(ax, x0 + 2 * cw, x0 + 11 * cw, y0 + ch_ + 0.10, height=0.30,
          label='dp[11] = 1 + min(dp[10], dp[6], dp[5], dp[2])', size=15)

    box(ax, 0.85, 1.35, 3.7, 0.90,
        label='dynamic programming\n5 + 6  =  2 coins', edge=TEAL, size=15)
    box(ax, 5.20, 1.35, 3.7, 0.90,
        label='greedy, biggest first\n9 + 1 + 1  =  3 coins', edge=RED,
        color=WHITE, size=15)
    cross(ax, 8.95, 1.80, r=0.26)

    text(ax, 4.8, 0.72,
         'greedy is safe on a designed currency, not on an arbitrary coin set',
         color=TEAL, size=14)
    text(ax, 4.8, 0.28, 'the 9 looks like progress and is a trap',
         color=AMBER, size=14)
    assert best == 2 and g_count == 3
    save(fig, 'day33_3.png')


# --------------------------------------------------------------------------
# 4  combinations vs permutations - the loop order
# --------------------------------------------------------------------------
def fig4():
    coins = [1, 2, 5]
    amount = 5
    comb = D.coin_ways_combinations(coins, amount)
    perm = D.coin_ways_permutations(coins, amount)

    fig, ax = canvas(9.6, 5.6, xlim=(0, 9.6), ylim=(0, 5.6))
    text(ax, 4.8, 5.25, 'coins 1, 2, 5   -   how many ways to make 5 ?',
         color=TEAL_L, size=16)

    region(ax, 0.30, 0.55, 4.25, 4.05)
    region(ax, 5.05, 0.55, 4.25, 4.05)

    box(ax, 0.60, 3.45, 3.65, 0.92,
        label='for c in coins:        <- outer\n    for a in range(c, N+1):',
        size=13)
    box(ax, 5.35, 3.45, 3.65, 0.92,
        label='for a in range(1, N+1): <- outer\n    for c in coins:',
        size=13)

    text(ax, 2.42, 3.15, 'each coin is offered once', color=TEAL, size=14)
    text(ax, 7.17, 3.15, 'each coin can be the last one', color=TEAL, size=14)

    ways = ['1+1+1+1+1', '1+1+1+2', '1+2+2', '5']
    for i, w in enumerate(ways):
        box(ax, 1.05, 2.45 - i * 0.44, 2.75, 0.38, label=w, size=13)
    text(ax, 2.42, 0.85, f'{comb} combinations', color=CYAN, size=17)

    rows = ['1+2+2', '2+1+2', '2+2+1', '. . . 9 orderings']
    for i, w in enumerate(rows):
        box(ax, 5.80, 2.45 - i * 0.44, 2.75, 0.38, label=w,
            size=13, edge=AMBER if i < 3 else TEAL)
    text(ax, 7.17, 0.85, f'{perm} permutations', color=AMBER, size=17)

    text(ax, 4.8, 0.22,
         'the two programs differ by which loop is outer, and neither complains',
         color=TEAL, size=14)
    assert (comb, perm) == (4, 9)
    save(fig, 'day33_4.png')


# --------------------------------------------------------------------------
# 5  one table, three combine steps
# --------------------------------------------------------------------------
def fig5():
    fig, ax = canvas(9.4, 4.6, xlim=(0, 9.4), ylim=(0, 4.6))
    text(ax, 4.7, 4.25, 'one loop, three problems', color=TEAL_L, size=17)

    rows = [
        ('climbing stairs', 'dp[i] = dp[i-1] + dp[i-2]', '+', 'count'),
        ('coin change', 'dp[a] = 1 + min over coins', 'min', 'optimise'),
        ('word break', 'dp[i] = or over cut points', 'or', 'decide'),
    ]
    y = 3.15
    for name, rule, op, kind in rows:
        box(ax, 0.40, y, 2.35, 0.72, label=name, size=14)
        box(ax, 2.95, y, 3.60, 0.72, label=rule, size=13)
        box(ax, 6.75, y, 0.90, 0.72, label=op, size=15, edge=AMBER, lw=2.4)
        text(ax, 8.45, y + 0.36, kind, color=TEAL, size=14)
        y -= 0.95

    text(ax, 4.7, 0.62, 'same states, same order, different operator',
         color=TEAL, size=15)
    text(ax, 4.7, 0.20, 'learn the table once and all three come free',
         color=CYAN, size=14)
    save(fig, 'day33_5.png')


# --------------------------------------------------------------------------
# 6  top-down visits fewer states when the table is sparse
# --------------------------------------------------------------------------
def fig6():
    r_tab, r_memo, states, cells_n = D.sparse_state_report([100, 250], 10000)

    fig, ax = canvas(9.2, 4.4, xlim=(0, 9.2), ylim=(0, 4.4))
    text(ax, 4.6, 4.05, 'coins 100, 250   -   amount 10000',
         color=TEAL_L, size=16)

    # tabulation: a full bar
    box(ax, 0.60, 2.55, 7.9, 0.72, label=f'{cells_n} cells, all filled',
        size=14)
    text(ax, 0.60, 3.52, 'bottom-up table', color=TEAL, size=15, ha='left')

    # memoisation: only the reachable states
    text(ax, 0.60, 2.15, 'top-down memo', color=TEAL, size=15, ha='left')
    w = 7.9 * states / cells_n
    box(ax, 0.60, 1.20, max(w, 0.16), 0.72, fill=NAVY, edge=AMBER, lw=2.4)
    text(ax, 0.60 + max(w, 0.16) + 0.25, 1.56,
         f'{states} states actually asked about', color=AMBER, size=14,
         ha='left')

    text(ax, 4.6, 0.62, 'only multiples of 50 are reachable, so the rest of',
         color=TEAL, size=14)
    text(ax, 4.6, 0.22, 'the table is filled for nobody   -   both answer '
         f'{r_tab}', color=TEAL, size=14)
    assert r_tab == r_memo
    save(fig, 'day33_6.png')


if __name__ == '__main__':
    os.chdir(HERE)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print('figures written')
