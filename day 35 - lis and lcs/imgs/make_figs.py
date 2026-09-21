"""Figures for Day 35 - LIS and LCS. Every number comes from lis_lcs.py."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from diagram_style import *            # noqa
import lis_lcs as L                    # noqa

A = L.A
TRAP = [2, 6, 8, 3, 4, 5, 1]


def row(ax, x0, y, vals, cw=0.78, ch=0.62, amber=(), grey=(), size=13):
    """A horizontal strip of cells; returns the centre of each."""
    pos = []
    for i, v in enumerate(vals):
        eg, lw = (AMBER, 2.6) if i in amber else (TEAL, 1.5)
        fill = GREY if i in grey else NAVY
        box(ax, x0 + i * cw, y, cw, ch, label=str(v), fill=fill, edge=eg,
            lw=lw, size=size)
        pos.append((x0 + i * cw + cw / 2, y + ch / 2))
    return pos


def fig1():
    """Patience sorting: deal the cards, count the piles."""
    piles = L.patience_piles(A)
    fig, ax = canvas(9.2, 6.0, xlim=(0, 9.2), ylim=(0, 6.0))
    text(ax, 4.6, 5.66, 'deal each card onto the leftmost pile whose top is >= it',
         color=TEAL_L, size=15, glowing=False)
    top = row(ax, 1.55, 4.55, A)
    text(ax, 1.37, 4.86, 'a', color=TEAL_L, size=14, ha='right', glowing=False)

    cw, ch, x0 = 0.90, 0.60, 2.00
    for k, pile in enumerate(piles):
        x = x0 + k * 1.45
        for j, v in enumerate(pile):
            last = (j == len(pile) - 1)
            box(ax, x, 3.05 - j * (ch + 0.08), cw, ch, label=str(v),
                fill=NAVY, edge=AMBER if last else GREY_L,
                lw=2.6 if last else 1.4, size=13)
        text(ax, x + cw / 2, 3.88, f'pile {k}', color=TEAL, size=12,
             glowing=False)
    text(ax, 6.60, 2.05, 'amber = the top card,', color=AMBER, size=13,
         ha='left', glowing=False)
    text(ax, 6.60, 1.70, 'i.e. tails[k]', color=AMBER, size=13, ha='left',
         glowing=False)
    text(ax, 4.6, 0.92,
         'the tops, read left to right, are the tails array [2, 3, 7, 18]',
         color=TEAL_L, size=14, glowing=False)
    text(ax, 4.6, 0.48,
         'four piles, so the longest increasing subsequence has length 4',
         color=AMBER, size=14, glowing=False)
    save(fig, 'day35_1.png')


def fig2():
    """tails has the right length and the wrong contents."""
    fig, ax = canvas(9.2, 5.4, xlim=(0, 9.2), ylim=(0, 5.4))
    text(ax, 4.6, 5.05, 'tails is a scoreboard, not an answer',
         color=TEAL_L, size=15, glowing=False)
    p = row(ax, 1.75, 3.95, TRAP, amber=(6,))
    text(ax, 1.57, 4.26, 'a', color=TEAL_L, size=14, ha='right', glowing=False)

    tails = L.lis_patience(TRAP)[1]
    real = L.lis_patience_reconstruct(TRAP)[1]
    row(ax, 1.75, 2.55, tails, amber=(0,))
    text(ax, 1.57, 2.86, 'tails', color=TEAL_L, size=14, ha='right', glowing=False)
    row(ax, 1.75, 1.40, real)
    text(ax, 1.57, 1.71, 'real LIS', color=TEAL_L, size=14, ha='right',
         glowing=False)

    arrow(ax, (p[6][0], 3.93), (2.14, 3.19), color=AMBER, lw=1.8, rad=-0.30)
    text(ax, 6.7, 2.86, 'the 1 arrives last, so [1, 3, 4, 5]', color=AMBER,
         size=13, ha='left', glowing=False)
    text(ax, 6.7, 2.50, 'is not even a subsequence of a', color=AMBER,
         size=13, ha='left', glowing=False)
    text(ax, 4.6, 0.62,
         'the length is always right; the path needs parent pointers taken at insert time',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day35_2.png')


def fig3():
    """The LCS table with the backtrack path."""
    x, y = 'AGGTAB', 'GXTXAYB'
    dp = L.lcs_table(x, y)
    fig, ax = canvas(9.2, 6.4, xlim=(0, 9.2), ylim=(0, 6.4))
    cw, ch, x0, y0 = 0.74, 0.58, 1.95, 1.25

    # backtrack path cells
    path, i, j = [], len(x), len(y)
    diag = []
    while i and j:
        path.append((i, j))
        if x[i - 1] == y[j - 1]:
            diag.append((i, j)); i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pos = {}
    for c in range(len(y) + 1):
        text(ax, x0 + c * cw + cw / 2, y0 + (len(x) + 1) * ch + 0.24,
             '-' if c == 0 else y[c - 1], color=TEAL, size=13, glowing=False)
    for r in range(len(x) + 1):
        yy = y0 + (len(x) - r) * ch
        text(ax, x0 - 0.20, yy + ch / 2, '-' if r == 0 else x[r - 1],
             color=TEAL, size=13, ha='right', glowing=False)
        for c in range(len(y) + 1):
            on = (r, c) in path
            box(ax, x0 + c * cw, yy, cw, ch, label=str(dp[r][c]),
                fill=GREY if (r, c) in diag else NAVY,
                edge=AMBER if on else TEAL, lw=2.6 if on else 1.3, size=12)
            pos[(r, c)] = (x0 + c * cw + cw / 2, yy + ch / 2)

    text(ax, 4.6, 5.92, 'match: 1 + diagonal      mismatch: max(up, left)',
         color=TEAL_L, size=15, glowing=False)
    text(ax, 4.6, 0.82, 'the four grey cells are the matches: G T A B',
         color=AMBER, size=14, glowing=False)
    text(ax, 4.6, 0.40,
         'keep two rows and the number survives, but the path does not',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day35_3.png')


def fig4():
    """LCS of two permutations is an LIS in disguise."""
    p = [1, 2, 3, 4, 5, 6]
    q = [2, 4, 1, 5, 6, 3]
    labels = [p.index(v) for v in q]
    _, sub = L.lcs_of_permutations(p, q)
    keep = {q.index(v) for v in sub}

    fig, ax = canvas(9.2, 5.6, xlim=(0, 9.2), ylim=(0, 5.6))
    text(ax, 4.6, 5.24, 'relabel q by "where does this value sit in p"',
         color=TEAL_L, size=15, glowing=False)
    pp = row(ax, 2.30, 4.15, p, cw=0.82)
    text(ax, 2.12, 4.46, 'p', color=TEAL_L, size=14, ha='right', glowing=False)
    qq = row(ax, 2.30, 2.75, q, cw=0.82, amber=tuple(keep))
    text(ax, 2.12, 3.06, 'q', color=TEAL_L, size=14, ha='right', glowing=False)
    ll = row(ax, 2.30, 1.55, labels, cw=0.82, amber=tuple(keep))
    text(ax, 2.12, 1.86, 'index in p', color=TEAL_L, size=14, ha='right',
         glowing=False)
    for k in (0, 3):
        arrow(ax, pp[p.index(q[k])], qq[k], color=TEAL, lw=1.4, rad=0.18)
    text(ax, 4.6, 0.92,
         'a common subsequence of p and q = an increasing run of those indices',
         color=AMBER, size=14, glowing=False)
    text(ax, 4.6, 0.48, 'so LCS collapses to LIS: O(n*m) becomes O(n log n)',
         color=TEAL, size=14, glowing=False)
    save(fig, 'day35_4.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('ok')
