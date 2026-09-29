"""Day 39 figures.  Every number is imported from backtracking.py - nothing typed by hand."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))
sys.dont_write_bytecode = True

from matplotlib.patches import Circle, Rectangle   # noqa: E402
from diagram_style import *          # noqa: F401,F403,E402
import backtracking as bt            # noqa: E402

os.chdir(HERE)


def W(h):
    """canvas(9, h) with a 0..10 x-range and equal aspect."""
    return canvas(9, h, xlim=(0, 10), ylim=(0, h * 10 / 9))


def mini_board(ax, cx, cy, n, path, size=0.84, edge_=TEAL, lw=1.4, ghosts=()):
    """A small n x n board centred at (cx, cy); row 0 on top."""
    s = size / n
    x0, y0 = cx - size / 2, cy + size / 2
    for r in range(n):
        for c in range(n):
            fill = NAVY if (r + c) % 2 == 0 else GREY
            ax.add_patch(Rectangle((x0 + c * s, y0 - (r + 1) * s), s, s,
                                   facecolor=fill, edgecolor='none', zorder=3))
    ax.add_patch(Rectangle((x0, y0 - size), size, size, facecolor='none',
                           edgecolor=edge_, linewidth=lw, zorder=4))
    for r, c in enumerate(path):
        ax.add_patch(Circle((x0 + (c + .5) * s, y0 - (r + .5) * s), s * 0.32,
                            facecolor=WHITE, edgecolor=WHITE, zorder=5))


# ---------------------------------------------------------------- figure 1
def fig1():
    """The whole 4-queens search tree: 17 nodes, 6 leaves, 2 solutions."""
    tree = bt.queens_tree(4)
    st = {}
    bt.queens(4, st)
    H = 6.9
    fig, ax = W(H)
    leaves = []

    def collect(t):
        if not t['children']:
            leaves.append(t)
        for ch in t['children']:
            collect(ch)
    collect(tree)
    xs = {id(t): 1.95 + i * (7.3 / (len(leaves) - 1)) for i, t in enumerate(leaves)}

    def place(t):
        for ch in t['children']:
            place(ch)
        if t['children']:
            xs[id(t)] = sum(xs[id(ch)] for ch in t['children']) / len(t['children'])
    place(tree)
    ytop, dy = 7.05, 1.42

    def y_of(t):
        return ytop - len(t['path']) * dy

    def draw(t):
        x, y = xs[id(t)], y_of(t)
        for ch in t['children']:
            ax.plot([x, xs[id(ch)]], [y - 0.46, y_of(ch) + 0.46], color=TEAL,
                    lw=1.5, zorder=2)
        if t['solution']:
            mini_board(ax, x, y, 4, t['path'], edge_=AMBER, lw=3.0)
            text(ax, x, y - 0.72, 'solution', color=AMBER, size=13)
        elif not t['children']:
            mini_board(ax, x, y, 4, t['path'], edge_=RED, lw=2.2)
            cross(ax, x, y - 0.68, r=0.14, lw=1.6)
        else:
            mini_board(ax, x, y, 4, t['path'])
        for ch in t['children']:
            draw(ch)
    draw(tree)
    for d in range(5):
        text(ax, 0.05, ytop - d * dy, f'{d} queen' + ('' if d == 1 else 's'),
             color=GREY_L, size=12, ha='left', glowing=False)
    sols = sum(1 for t in leaves if t['solution'])
    rejected = st['tested'] - (st['nodes'] - 1)
    text(ax, 5.0, 0.28,
         f'{st["nodes"]} nodes, {sols} solutions, {len(leaves) - sols} dead ends;'
         f' {rejected} squares rejected without ever placing a queen on them',
         color=TEAL, size=13, glowing=False)
    save(fig, 'day39_1.png')


# ---------------------------------------------------------------- figure 2
def fig2():
    """8 queens: how big the space is under each rule, log scale."""
    st = {}
    bt.queens(8, st)
    rows = [
        ('any 8 of the 64 squares', math.comb(64, 8), TEAL),
        ('one queen per row', 8 ** 8, TEAL),
        ('one per row and per column', math.factorial(8), TEAL),
        ('backtracking with pruning', st['nodes'], AMBER),
    ]
    H = 3.3
    fig, ax = W(H)
    top = max(v for _, v, _ in rows)
    x0, span = 4.1, 4.3
    for i, (lab, v, col) in enumerate(rows):
        y = 3.05 - i * 0.72
        w = span * math.log10(v) / math.log10(top)
        text(ax, x0 - 0.2, y, lab, color=col, size=14, ha='right', glowing=False)
        box(ax, x0, y - 0.2, w, 0.4, fill=NAVY, edge=col, lw=2.2 if col == AMBER else 1.6)
        text(ax, x0 + w + 0.15, y, f'{v:,}', color=col, size=14, ha='left', glowing=False)
    text(ax, x0 + span / 2, 0.3, 'bar length on a log scale: every step is a factor, not a difference',
         color=GREY_L, size=12, glowing=False)
    save(fig, 'day39_2.png')


# ---------------------------------------------------------------- figure 3
def fig3():
    """Forgotten unchoose: the moment the 8x8 search gives up."""
    sols, nodes, seen = bt.queens_no_undo(8)
    final = seen[-1]
    placed = {(r, c) for p in seen for r, c in enumerate(p)}
    real = {(r, c) for r, c in enumerate(final)}
    ghosts = placed - real
    cols = {c for _, c in placed}
    diag = {r - c for r, c in placed}
    anti = {r + c for r, c in placed}
    r_next = len(final)

    def hit(qs, r, c):
        return any(c == qc or r - c == qr - qc or r + c == qr + qc for qr, qc in qs)

    H = 5.0
    fig, ax = W(H)
    s, X0, Y0 = 0.52, 0.5, 5.05          # Y0 = top edge of the board
    for r in range(8):
        for c in range(8):
            fill = NAVY if (r + c) % 2 == 0 else GREY
            ax.add_patch(Rectangle((X0 + c * s, Y0 - (r + 1) * s), s, s, facecolor=fill,
                                   edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((X0, Y0 - 8 * s), 8 * s, 8 * s, facecolor='none',
                           edgecolor=TEAL, lw=1.6, zorder=3))
    for r in range(8):
        text(ax, X0 - 0.2, Y0 - (r + .5) * s, str(r), color=GREY_L, size=11, glowing=False)
    ghost_only = 0
    for c in range(8):
        cx, cy = X0 + (c + .5) * s, Y0 - (r_next + .5) * s
        blocked = c in cols or r_next - c in diag or r_next + c in anti
        assert blocked
        if hit(real, r_next, c):
            ax.add_patch(Rectangle((X0 + c * s, Y0 - (r_next + 1) * s), s, s,
                                   facecolor='none', edgecolor=TEAL_L, lw=2.4, zorder=4))
        else:
            ghost_only += 1
            cross(ax, cx, cy, r=0.15, lw=1.8)
    for r, c in ghosts:
        ax.add_patch(Circle((X0 + (c + .5) * s, Y0 - (r + .5) * s), s * 0.3,
                            facecolor='none', edgecolor=RED, lw=2.0, ls='--', zorder=5))
    for r, c in real:
        ax.add_patch(Circle((X0 + (c + .5) * s, Y0 - (r + .5) * s), s * 0.3,
                            facecolor=WHITE, edgecolor=WHITE, zorder=5))
    # right-hand side
    xR = 5.3
    legend(ax, xR, 4.75, [(WHITE, 'the one queen really on the board'),
                          (RED, f'{len(ghosts)} ghosts: taken off the board,'),
                          ], size=13)
    text(ax, xR + 0.42, 4.75 - 0.42 * 1.75, 'still marked in the three sets',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, xR, 3.35, f'row {r_next}: {8 - ghost_only} squares blocked by the real queen',
         color=TEAL_L, size=13, ha='left', glowing=False)
    text(ax, xR, 2.95, f'row {r_next}: {ghost_only} squares blocked only by ghosts',
         color=RED, size=13, ha='left', glowing=False)
    text(ax, xR, 2.1, f'{len(sols)} solutions', color=RED, size=20, ha='left')
    text(ax, xR, 1.55, f'after {nodes} nodes; the correct search', color=TEAL, size=13,
         ha='left', glowing=False)
    text(ax, xR, 1.15, f'finds {len(bt.queens(8))}. Nothing raised, nothing warned.',
         color=TEAL, size=13, ha='left', glowing=False)
    save(fig, 'day39_3.png')


# ---------------------------------------------------------------- figure 4
def fig4():
    """The mirror-symmetry shortcut misses the middle column of odd boards."""
    H = 3.9
    fig, ax = W(H)
    n = 5
    s, X0, Y = 0.72, 0.55, 2.3
    kinds = []
    for c in range(n):
        if c < n // 2:
            kinds.append(('tried', TEAL, NAVY))
        elif c == n // 2 and n % 2:
            kinds.append(('never', RED, NAVY))
        else:
            kinds.append(('x2', GREY_L, GREY))
    for c, (lab, col, fill) in enumerate(kinds):
        box(ax, X0 + c * s, Y, s, s, fill=fill, edge=col, lw=2.4 if col == RED else 1.6)
        text(ax, X0 + (c + .5) * s, Y + s / 2, str(c), color=WHITE, size=14, glowing=False)
    for a in range(n // 2):
        b = n - 1 - a
        arrow(ax, (X0 + (a + .5) * s, Y + s + 0.05), (X0 + (b + .5) * s, Y + s + 0.05),
              color=GREY_L, lw=1.3, rad=-0.45, style='<|-|>', ms=9)
    text(ax, X0 + n * s / 2, Y + s + 1.2, f'row 0 of a {n} x {n} board', color=TEAL, size=13,
         glowing=False)
    legend(ax, X0, Y - 0.45, [(NAVY, 'searched'),
                              (GREY, 'not searched, counted by the x 2'),
                              (RED, 'middle column: its own mirror image,'),
                              ], size=13)
    text(ax, X0 + 0.42, Y - 0.45 - 0.42 * 2.75, 'never searched and never counted',
         color=TEAL, size=13, ha='left', glowing=False)
    # table
    xT = 5.6
    text(ax, xT, 3.8, 'n', color=GREY_L, size=13, glowing=False)
    text(ax, xT + 1.3, 3.8, 'true', color=GREY_L, size=13, glowing=False)
    text(ax, xT + 2.8, 3.8, 'halved x 2', color=GREY_L, size=13, glowing=False)
    for i, m in enumerate(range(5, 11)):
        t, h = len(bt.queens(m)), bt.queens_count_half(m)
        col = TEAL if t == h else RED
        y = 3.35 - i * 0.48
        text(ax, xT, y, str(m), color=col, size=14, glowing=False)
        text(ax, xT + 1.3, y, str(t), color=col, size=14, glowing=False)
        text(ax, xT + 2.8, y, str(h), color=col, size=14, glowing=False)
        if t != h:
            text(ax, xT + 3.5, y, 'wrong', color=col, size=14, ha='left', glowing=False)
    save(fig, 'day39_4.png')


# ---------------------------------------------------------------- figure 5
def fig5():
    """Sudoku: nodes visited, naive reading order vs MRV, log scale."""
    data = []
    for name, s in bt.PUZZLES.items():
        g1, n1 = bt.sudoku_naive(s, cap=bt.NAIVE_CAP)
        n1 = n1 if g1 is not None else bt.ANTI_BRUTE_NAIVE_NODES
        _, n2 = bt.sudoku_mrv(s)
        data.append((name, sum(ch != '.' for ch in s), n1, n2))
    H = 4.1
    fig, ax = W(H)
    top = max(d[2] for d in data)
    x0, span = 2.6, 4.4
    for i, (name, givens, n1, n2) in enumerate(data):
        y = 3.95 - i * 1.25
        text(ax, x0 - 0.2, y - 0.2, name, color=TEAL_L, size=14, ha='right', glowing=False)
        text(ax, x0 - 0.2, y - 0.55, f'{givens} givens', color=GREY_L, size=11, ha='right',
             glowing=False)
        for j, (v, col, lab) in enumerate([(n1, GREY_L, 'naive'), (n2, AMBER, 'MRV')]):
            yy = y - j * 0.42
            w = span * math.log10(v) / math.log10(top)
            box(ax, x0, yy - 0.16, w, 0.32, fill=NAVY if j else GREY, edge=col, lw=1.6)
            text(ax, x0 + w + 0.12, yy, f'{lab} {v:,}', color=col, size=12, ha='left',
                 glowing=False)
        text(ax, 9.9, y - 0.2, f'{n1 / n2:,.0f}x', color=AMBER, size=15, ha='right')
    text(ax, x0 + span / 2, 0.12, 'nodes visited, log scale', color=GREY_L, size=12,
         glowing=False)
    save(fig, 'day39_5.png')


# ---------------------------------------------------------------- figure 6
def fig6():
    """Where each solver makes its first real choice on the anti-brute puzzle."""
    s = bt.PUZZLES['anti-brute']
    g = bt.parse(s)
    cand = bt.candidates(g)
    solved, _ = bt.sudoku_mrv(s)
    pick = []
    bt.sudoku_mrv(s, pick)
    (pr, pc), pk = pick[0]
    nr, nc = next((r, c) for r in range(9) for c in range(9) if g[r][c] == 0)
    H = 5.0
    fig, ax = W(H)
    cs, X0, Y0 = 0.5, 0.45, 5.1
    for r in range(9):
        for c in range(9):
            x, y = X0 + c * cs, Y0 - (r + 1) * cs
            if g[r][c]:
                box(ax, x, y, cs, cs, fill=NAVY, edge=TEAL, lw=0.8)
                text(ax, x + cs / 2, y + cs / 2, str(g[r][c]), color=WHITE, size=14,
                     glowing=False)
            else:
                ax.add_patch(Rectangle((x, y), cs, cs, facecolor='none', edgecolor=TEAL,
                                       lw=0.8, zorder=3))
                text(ax, x + cs / 2, y + cs / 2, str(len(cand[(r, c)])), color=GREY_L,
                     size=11, glowing=False)
    for k in range(4):
        ax.plot([X0 + k * 3 * cs] * 2, [Y0, Y0 - 9 * cs], color=TEAL_L, lw=2.2, zorder=4)
        ax.plot([X0, X0 + 9 * cs], [Y0 - k * 3 * cs] * 2, color=TEAL_L, lw=2.2, zorder=4)
    for (r, c), col in [((nr, nc), RED), ((pr, pc), AMBER)]:
        ax.add_patch(Rectangle((X0 + c * cs, Y0 - (r + 1) * cs), cs, cs, facecolor='none',
                               edgecolor=col, lw=3.0, zorder=6))
    xR = 5.3
    text(ax, xR, 4.75, 'grey digit = how many digits are still legal there',
         color=GREY_L, size=12, ha='left', glowing=False)
    text(ax, xR, 4.05, 'naive: first empty cell, top-left', color=RED, size=14, ha='left',
         glowing=False)
    text(ax, xR, 3.65, f'{len(cand[(nr, nc)])} legal digits, tried 1 upward;',
         color=TEAL, size=13, ha='left', glowing=False)
    top_row = ''.join(map(str, solved[0]))
    text(ax, xR, 3.3, f'the answer is {solved[nr][nc]} (top row {top_row})',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, xR, 2.95, f'{bt.ANTI_BRUTE_NAIVE_NODES:,} nodes', color=RED, size=13,
         ha='left', glowing=False)
    _, mrv_nodes = bt.sudoku_mrv(s)
    text(ax, xR, 2.2, 'MRV: the cell with the fewest', color=AMBER, size=14, ha='left',
         glowing=False)
    text(ax, xR, 1.85, f'legal digits ({pk}), wherever it is', color=AMBER, size=14,
         ha='left', glowing=False)
    text(ax, xR, 1.45, f'{mrv_nodes:,} nodes', color=AMBER, size=13, ha='left',
         glowing=False)
    save(fig, 'day39_6.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print('figures written')
