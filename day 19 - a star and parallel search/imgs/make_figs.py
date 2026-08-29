"""Figures for Day 19 - A* and parallel graph search.

Every number on these figures is computed by importing astar.py, never typed in
by hand.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: F401,F403
import astar as A                    # the day's actual implementation

OUT = HERE

# ---------------------------------------------------------------- the numbers
DIJ_PATH, DIJ_COST, DIJ_EXP = A.best_first(A.START, A.GOAL)
AST_PATH, AST_COST, AST_EXP = A.best_first(A.START, A.GOAL, A.manhattan)
W_PATH, W_COST, W_EXP = A.best_first(A.START, A.GOAL, A.manhattan, weight=5.0)
G_PATH, G_COST, G_EXP = A.best_first(A.START, A.GOAL, A.manhattan, greedy=True)
WADJ = A.weighted_adj()
GADJ = A.grid_adj()
LEVELS = A.parallel_bfs_levels(GADJ, A.START)


# =========================================================== fig 1: f = g + h
def fig1():
    fig, ax = canvas(9.6, 4.4, xlim=(0, 9.6), ylim=(0, 4.4))

    sx, sy = 0.9, 1.5
    ux, uy = 4.3, 2.9
    gx, gy = 8.6, 1.7

    # the walked route: real edges, real cost
    route = [(sx, sy), (1.9, 2.4), (3.0, 2.2), (ux, uy)]
    for a, b in zip(route, route[1:]):
        edge(ax, a, b, directed=False, r=0.34, lw=2.0)
    # the guess: one dashed jump, no idea what is really in the way
    arrow(ax, (ux + 0.34, uy - 0.1), (gx - 0.42, gy + 0.2),
          color=AMBER, ls='--', lw=2.0, rad=-0.10)

    node(ax, sx, sy, 'S', r=0.34)
    for px, py in route[1:-1]:
        node(ax, px, py, '', r=0.20)
    node(ax, ux, uy, 'u', r=0.34, edge=ORANGE if 'ORANGE' in dir() else TEAL_L)
    node(ax, gx, gy, 'G', r=0.34)

    text(ax, 2.3, 1.55, 'g(u)', color=TEAL_L, size=17)
    text(ax, 2.5, 1.05, 'what this route has already cost',
         color=TEAL, size=13, glowing=False)
    text(ax, 6.7, 3.30, 'h(u)', color=AMBER, size=17)
    text(ax, 6.7, 2.92, 'a guess at the rest', color=AMBER, size=13,
         glowing=False)

    text(ax, 4.8, 0.55, 'f(u) = g(u) + h(u)', color=TEAL_L, size=19)
    text(ax, 4.8, 0.15,
         'the frontier is ordered by f, and that is the whole difference',
         color=TEAL, size=13, glowing=False)

    text(ax, 0.2, 3.95, 'h = 0  ->  order by g alone  ->  Dijkstra',
         color=TEAL, size=13, ha='left', glowing=False)
    text(ax, 0.2, 3.60, 'h too large  ->  the guess wins the argument  ->  a '
                        'cheaper route is never tried',
         color=RED, size=13, ha='left', glowing=False)
    save(fig, os.path.join(OUT, 'day19_1.png'))


# ============================================ fig 2: who looks at how much map
def draw_grid(ax, x0, y0, cw, expanded, path, title, sub):
    exp, pth = set(expanded), set(path)
    for r in range(A.ROWS):
        for c in range(A.COLS):
            cell = (r, c)
            gx = x0 + c * cw
            gy = y0 - r * cw
            mud = A.COST[r][c] is not None and A.COST[r][c] > 1
            if cell in exp:
                fill = NAVY
            elif mud:
                fill = GREY
            else:
                fill = 'none'
            ec = AMBER if cell in pth else TEAL
            lw = 2.2 if cell in pth else 0.9
            ax.add_patch(Rectangle((gx, gy), cw, cw, facecolor=fill,
                                   edgecolor=ec, linewidth=lw, zorder=3))
            if mud:
                text(ax, gx + cw / 2, gy + cw / 2, '~', color=AMBER, size=11,
                     glowing=False, zorder=5)
            if cell == A.START:
                text(ax, gx + cw / 2, gy + cw / 2, 'S', color=WHITE, size=12,
                     glowing=False, zorder=5)
            if cell == A.GOAL:
                text(ax, gx + cw / 2, gy + cw / 2, 'G', color=WHITE, size=12,
                     glowing=False, zorder=5)
    w = A.COLS * cw
    text(ax, x0 + w / 2, y0 + cw + 0.42, title, color=TEAL_L, size=16,
         glowing=False)
    text(ax, x0 + w / 2, y0 + cw + 0.10, sub, color=TEAL, size=13,
         glowing=False)


def fig2():
    fig, ax = canvas(10.0, 5.6, xlim=(0, 10.0), ylim=(0, 5.6))
    cw = 0.46
    top = 4.55
    draw_grid(ax, 0.30, top, cw, DIJ_EXP, DIJ_PATH, 'Dijkstra  (h = 0)',
              'expanded %d cells, cost %d' % (len(DIJ_EXP), DIJ_COST))
    draw_grid(ax, 5.55, top, cw, AST_EXP, AST_PATH, 'A*  (Manhattan)',
              'expanded %d cells, cost %d' % (len(AST_EXP), AST_COST))
    legend(ax, 0.35, 1.60, [(NAVY, 'expanded'), (GREY, 'mud, costs 5'),
                            (AMBER, 'the path returned')], size=12)
    text(ax, 6.2, 1.42,
         'Same map, same answer, %d%% of the work.'
         % round(100.0 * len(AST_EXP) / len(DIJ_EXP)),
         color=TEAL_L, size=17)
    text(ax, 6.2, 0.98,
         'An admissible heuristic never changes the answer -',
         color=TEAL, size=13, glowing=False)
    text(ax, 6.2, 0.66,
         'only how much of the map you bother to look at.',
         color=TEAL, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day19_2.png'))


# ================================== fig 3: the table of fast and wrong answers
def fig3():
    rows = [
        ('h = 0  (Dijkstra)', DIJ_COST, len(DIJ_EXP), True),
        ('h = Manhattan  (A*)', AST_COST, len(AST_EXP), True),
        ('h x 5  (weighted A*)', W_COST, len(W_EXP), False),
        ('f = h  (greedy best-first)', G_COST, len(G_EXP), False),
    ]
    headers = ['priority', 'cost', 'expanded', 'optimal?']
    colw = [4.0, 1.3, 1.7, 1.5]
    x0, ytop, rh = 0.55, 3.45, 0.62

    fig, ax = canvas(9.6, 4.3, xlim=(0, 9.6), ylim=(0, 4.3))

    # header row
    cx = x0
    for w, hlab in zip(colw, headers):
        text(ax, cx + w / 2, ytop + rh * 0.5, hlab, color=TEAL_L, size=14)
        cx += w
    ax.plot([x0, x0 + sum(colw)], [ytop + 0.06, ytop + 0.06],
            color=TEAL, lw=1.4, zorder=3)

    for i, (name, cost, exp, good) in enumerate(rows):
        y = ytop - (i + 1) * rh
        vals = [name, str(cost), str(exp), 'yes' if good else 'NO']
        cx = x0
        for w, v, ha in zip(colw, vals, ['left', 'center', 'center', 'center']):
            col = TEAL if good else RED
            tx = cx + 0.16 if ha == 'left' else cx + w / 2
            text(ax, tx, y + rh / 2, v, color=col, size=14, ha=ha,
                 glowing=False)
            cx += w

    # the emphasis box: a single complete rectangle, all four sides identical
    by = ytop - 4 * rh
    bh = 2 * rh
    ax.add_patch(Rectangle((x0 - 0.18, by - 0.06), sum(colw) + 0.36, bh + 0.04,
                           facecolor='none', edgecolor=AMBER, linewidth=2.0,
                           linestyle='--', zorder=5))
    text(ax, x0 + sum(colw) + 0.55, by + bh / 2,
         'runs fine,\nreturns a\nworse path', color=AMBER, size=13, ha='left',
         glowing=False)

    text(ax, 4.8, 0.55,
         'Inflating the guess buys fewer expansions and sells the guarantee.',
         color=TEAL_L, size=16)
    text(ax, 4.8, 0.18,
         'Weighted A* at least promises "at most w times optimal"; greedy '
         'best-first promises nothing.',
         color=TEAL, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day19_3.png'))


# ================================================ fig 4: delta-stepping phases
def fig4():
    small, large = 1, 99
    _, ph_s, rl_s = A.delta_stepping(WADJ, 0, small)
    _, ph_l, rl_l = A.delta_stepping(WADJ, 0, large)

    fig, ax = canvas(9.8, 6.8, xlim=(0, 9.8), ylim=(0, 6.8))
    top, rh, sy = 5.55, 0.42, 1.20

    def column(x0, cx, phases, title, sub, relax):
        text(ax, cx, top + 0.86, title, color=TEAL_L, size=16, glowing=False)
        text(ax, cx, top + 0.48, sub, color=TEAL, size=13, glowing=False)
        for i, (b, verts) in enumerate(phases):
            y = top - i * rh
            text(ax, x0 - 0.22, y, 'phase %d' % (i + 1), color=TEAL, size=11,
                 ha='right', glowing=False)
            for j, v in enumerate(verts):
                box(ax, x0 + j * 0.44, y - 0.16, 0.36, 0.32, label=str(v),
                    size=10, lw=1.0)
        text(ax, cx, sy, '%d phases, %d edge relaxations' % (len(phases), relax),
             color=AMBER, size=14, glowing=False)

    column(1.05, 2.70, ph_s, 'delta = %d' % small,
           'one vertex per phase - Dijkstra', rl_s)
    column(6.10, 7.90, ph_l, 'delta = %d' % large,
           'fat buckets - Bellman-Ford', rl_l)

    ax.plot([5.55, 5.55], [0.95, 6.25], color=TEAL, lw=1.0, ls=':', zorder=2)

    text(ax, 4.9, 0.62,
         'A phase is a batch: every vertex in it can be relaxed at the same time.',
         color=TEAL_L, size=15)
    text(ax, 4.9, 0.24,
         'Fewer, fatter phases is what a parallel machine wants - until the '
         'redundant relaxations eat the win.',
         color=TEAL, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day19_4.png'))


# ================================================= fig 5: BFS as a frontier
def fig5():
    widths = [len(l) for l in LEVELS]
    fig, ax = canvas(9.8, 4.8, xlim=(0, 9.8), ylim=(0, 4.8))

    x0, y0, dx, dy, r = 0.75, 1.85, 0.70, 0.34, 0.13
    for i, w in enumerate(widths):
        x = x0 + i * dx
        for j in range(w):
            yy = y0 + (j - (w - 1) / 2.0) * dy
            ax.add_patch(Circle((x, yy), r, facecolor=NAVY, edgecolor=TEAL,
                                linewidth=1.2, zorder=4))
        text(ax, x, 0.62, str(i), color=TEAL, size=11, glowing=False)
    text(ax, x0 + (len(widths) - 1) * dx / 2, 0.22, 'level',
         color=TEAL, size=13, glowing=False)

    widest = max(widths)
    wi = widths.index(widest)
    brace(ax, x0 + wi * dx - 0.30, x0 + wi * dx + 0.30,
          y0 + ((widest - 1) / 2.0) * dy + 0.30,
          label='%d independent vertices' % widest, size=13)

    text(ax, 4.9, 4.45,
         'BFS in parallel is not a queue with locks - it is a frontier.',
         color=TEAL_L, size=16)
    text(ax, 4.9, 4.05,
         'work = O(V + E), unchanged;   depth = %d levels, and that is what '
         'the machine waits for.' % len(widths),
         color=TEAL, size=13, glowing=False)
    save(fig, os.path.join(OUT, 'day19_5.png'))


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print('figures written to', OUT)
