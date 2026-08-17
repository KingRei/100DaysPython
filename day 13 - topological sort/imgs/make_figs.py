"""Figures for day 13 - topological sort (Kahn + DFS). Run from this folder."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from diagram_style import *

# a small course-prerequisite DAG, laid out in its own topological layers
POS = {0: (1.2, 4.9), 1: (1.2, 2.1),
       2: (3.5, 5.9), 3: (3.5, 3.5), 4: (3.5, 1.1),
       5: (5.8, 4.7), 6: (5.8, 2.3),
       7: (8.1, 3.5)}
EDGES = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5),
         (3, 6), (4, 6), (5, 7), (6, 7)]
INDEG = {0: 0, 1: 0, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2}
ROUNDS = [[0, 1], [2, 3, 4], [5, 6], [7]]


def _draw_graph(ax, edge_color=None, lw=1.7):
    for a, b in EDGES:
        c = TEAL if edge_color is None else edge_color(a, b)
        edge(ax, POS[a], POS[b], color=c, lw=lw)
    for n, (x, y) in POS.items():
        node(ax, x, y, n)


def fig1_dag():
    """The DAG itself, with each node's in-degree."""
    fig, ax = canvas(9.8, 7.4, xlim=(0, 9.8), ylim=(0, 7.4))
    _draw_graph(ax)
    for n, (x, y) in POS.items():
        text(ax, x, y + 0.72, str(INDEG[n]), color=AMBER, size=14)
    text(ax, 4.9, 7.0, 'u -> v means "u must come before v";  amber = in-degree',
         color=TEAL_L, size=15)
    text(ax, 0.1, 0.25, 'no cycle anywhere, so at least one valid order exists',
         color=TEAL, size=14, ha='left')
    save(fig, 'day13_1.png')


def fig2_kahn():
    """Kahn's algorithm peels off the in-degree-zero nodes, round by round."""
    fig, ax = canvas(9.8, 7.4, xlim=(0, 9.8), ylim=(0, 7.4))
    for i, x in enumerate([1.2, 3.5, 5.8, 8.1]):
        region(ax, x - 0.80, 0.35, 1.60, 6.05, fill=PALE, alpha=0.45)
        text(ax, x, 0.05, 'round %d' % i, color=TEAL_L, size=14)
    _draw_graph(ax)
    text(ax, 4.9, 7.05,
         'each round takes every node whose in-degree has dropped to 0',
         color=TEAL_L, size=15)
    text(ax, 4.9, 6.62, 'order:   0  1   |   2  3  4   |   5  6   |   7',
         color=AMBER, size=16)
    save(fig, 'day13_2.png')


def _dfs_finish_order():
    """Run the same DFS the article uses, so the figure cannot drift from the code."""
    adj = {n: sorted(b for a, b in EDGES if a == n) for n in POS}
    seen, order = set(), []

    def go(u):
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                go(v)
        order.append(u)          # finished: every successor is already done

    for n in sorted(POS):
        if n not in seen:
            go(n)
    return order


def fig3_dfs():
    """DFS: a node is appended only after all of its successors are done."""
    order = _dfs_finish_order()
    finish = {n: i for i, n in enumerate(order)}
    fig, ax = canvas(9.8, 8.2, xlim=(0, 9.8), ylim=(-0.8, 7.4))
    _draw_graph(ax)
    cx = sum(p[0] for p in POS.values()) / len(POS)
    cy = sum(p[1] for p in POS.values()) / len(POS)
    for n, (x, y) in POS.items():
        dx, dy = x - cx, y - cy
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        text(ax, x + dx / d * 0.80, y + dy / d * 0.80, 'f%d' % finish[n],
             color=AMBER, size=13)
    text(ax, 4.9, 7.05, 'f = the order in which DFS finishes a node',
         color=TEAL_L, size=15)
    fmt = lambda seq: '   '.join(str(v) for v in seq)
    text(ax, 0.1, -0.30, 'finish order      ' + fmt(order),
         color=TEAL, size=15, ha='left')
    text(ax, 0.1, -0.70, 'reverse it  ->    ' + fmt(order[::-1]),
         color=AMBER, size=15, ha='left')
    save(fig, 'day13_3.png')


def fig4_cycle():
    """A cycle means no valid order - this is how both algorithms fail."""
    fig, ax = canvas(11.0, 4.6, xlim=(0, 11.0), ylim=(0, 4.6))
    P = {'A': (1.6, 3.2), 'B': (3.8, 3.2), 'C': (2.7, 1.4)}
    for a, b in [('A', 'B'), ('B', 'C')]:
        edge(ax, P[a], P[b], color=TEAL)
    edge(ax, P['C'], P['A'], color=RED, lw=2.0)
    for n, (x, y) in P.items():
        node(ax, x, y, n)
    cross(ax, 5.3, 2.4, r=0.34)
    text(ax, 6.0, 2.9, 'every node keeps in-degree >= 1,', color=TEAL,
         size=15, ha='left')
    text(ax, 6.0, 2.4, 'so Kahn empties the queue early:', color=TEAL,
         size=15, ha='left')
    text(ax, 6.0, 1.9, 'output shorter than n  =>  cycle', color=AMBER,
         size=15, ha='left')
    text(ax, 2.7, 0.35, 'A -> B -> C -> A', color=RED, size=16)
    save(fig, 'day13_4.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig1_dag(); fig2_kahn(); fig3_dfs(); fig4_cycle()
    print('figures written')
