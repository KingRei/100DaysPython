"""Figures for day 12 - graph traversal (BFS / DFS). Run from this folder."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from diagram_style import *

POS = {1: (1.0, 3.3), 2: (3.0, 4.6), 3: (3.0, 2.0),
       4: (5.0, 5.4), 5: (5.0, 3.3), 6: (5.0, 1.2),
       7: (7.0, 4.4), 8: (7.0, 2.0)}
EDGES = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 5), (3, 6),
         (4, 7), (5, 7), (6, 8), (7, 8)]


def fig1_bfs():
    """BFS explores the graph one layer at a time."""
    fig, ax = canvas(9.6, 6.8, xlim=(0, 9.6), ylim=(0, 6.8))
    for x, lab in [(1.0, 'L0'), (3.0, 'L1'), (5.0, 'L2'), (7.0, 'L3')]:
        region(ax, x - 0.75, 0.45, 1.5, 5.5, fill=PALE, alpha=0.45)
        text(ax, x, 0.15, lab, color=TEAL_L, size=15)
    for a, b in EDGES:
        edge(ax, POS[a], POS[b], directed=False, lw=1.6)
    for n, (x, y) in POS.items():
        node(ax, x, y, n)
    text(ax, 1.0, 6.3, 'start', color=AMBER, size=15)
    arrow(ax, (1.0, 6.05), (1.0, 3.85), color=AMBER, lw=1.6)
    text(ax, 4.7, 6.3, 'BFS pops:   1  |  2  3  |  4  5  6  |  7  8',
         color=TEAL_L, size=16, ha='left')
    save(fig, 'day12_1.png')


def fig2_dfs():
    """DFS dives as deep as it can; dashed edges are the ones it skips."""
    order = {1: 1, 2: 2, 4: 3, 7: 4, 5: 5, 3: 6, 6: 7, 8: 8}
    tree = {(1, 2), (2, 4), (4, 7), (5, 7), (3, 5), (3, 6), (6, 8)}
    fig, ax = canvas(9.0, 7.0, xlim=(0, 9.0), ylim=(0, 7.0))
    for a, b in EDGES:
        is_tree = (a, b) in tree or (b, a) in tree
        edge(ax, POS[a], POS[b], directed=False, lw=2.0 if is_tree else 1.3,
             color=TEAL if is_tree else GREY_L)
        if not is_tree:
            pass
    cx = sum(p[0] for p in POS.values()) / len(POS)
    cy = sum(p[1] for p in POS.values()) / len(POS)
    for n, (x, y) in POS.items():
        node(ax, x, y, n)
        dx, dy = x - cx, y - cy
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        text(ax, x + dx / d * 0.85, y + dy / d * 0.85, f'#{order[n]}',
             color=AMBER, size=13)
    text(ax, 4.0, 6.8, 'visiting order 1 - 2 - 4 - 7 - 5 - 3 - 6 - 8',
         color=TEAL_L, size=16)
    legend(ax, 7.9, 1.3, [(TEAL, 'tree edge'), (GREY_L, 'already visited')])
    save(fig, 'day12_2.png')


def fig3_islands():
    """LeetCode 200 - number of islands."""
    grid = ['11000', '11000', '00100', '00011']
    fig, ax = canvas(8.8, 5.0, xlim=(0, 8.8), ylim=(0, 5.0))
    w = h = 0.85
    x0, y0 = 0.6, 4.0
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            land = ch == '1'
            box(ax, x0 + c * w, y0 - r * h, w, h, label=ch,
                fill=NAVY if land else GREY, size=14,
                color=WHITE if land else GREY_L,
                edge=AMBER if land else TEAL, lw=2.6 if land else 1.2)
    text(ax, 6.6, 3.7, '3 islands', color=AMBER, size=20)
    text(ax, 6.6, 2.9, 'every unvisited 1 starts\na flood fill; the fill sinks\nthe whole island',
         color=TEAL, size=14)
    save(fig, 'day12_3.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig1_bfs(); fig2_dfs(); fig3_islands()
    print('figures written')
