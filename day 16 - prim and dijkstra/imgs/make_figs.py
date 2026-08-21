"""Figures for day 16 - Prim and Dijkstra, one heap-greedy skeleton."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from diagram_style import *   # noqa

POS = {'A': (0.9, 4.5), 'B': (3.7, 5.3), 'C': (6.8, 4.6),
       'D': (2.3, 2.7), 'E': (5.3, 2.7), 'F': (2.4, 0.7), 'G': (7.2, 1.3)}
EDGES = [('A', 'B', 7), ('A', 'D', 5), ('B', 'C', 8), ('B', 'D', 9), ('B', 'E', 7),
         ('C', 'E', 5), ('D', 'E', 15), ('D', 'F', 6), ('E', 'F', 8),
         ('E', 'G', 9), ('F', 'G', 11)]
MST = {('A', 'D'), ('C', 'E'), ('D', 'F'), ('A', 'B'), ('B', 'E'), ('E', 'G')}
SPT = {('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('D', 'F'), ('F', 'G')}
DIST = {'A': 0, 'B': 7, 'C': 15, 'D': 5, 'E': 14, 'F': 11, 'G': 22}


def draw_graph(ax, highlight=(), r=0.42, offset=(0.0, 0.0), pos=POS, weights=True):
    ox, oy = offset
    P = {k: (x + ox, y + oy) for k, (x, y) in pos.items()}
    for u, v, w in EDGES:
        hot = (u, v) in highlight or (v, u) in highlight
        edge(ax, P[u], P[v], color=AMBER if hot else GREY_L, lw=3.0 if hot else 1.2,
             directed=False, weight=w if weights else None,
             wcolor=AMBER if hot else GREY_L, r=r)
    for n, (x, y) in P.items():
        node(ax, x, y, n, r=r)
    return P


# ---------------------------------------------------------------- figure 1
# a snapshot in the middle of Prim: A, D, F are settled, and the frontier is a cut
fig, ax = canvas(11, 6.1, xlim=(0, 11.8), ylim=(0, 6.7))
region(ax, 0.15, 0.05, 3.05, 5.6)
BOUGHT = {('A', 'D'), ('D', 'F')}
CROSS = {('A', 'B'), ('B', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')}
for u, v, w in EDGES:
    key = (u, v)
    if key in BOUGHT:
        col, lw = AMBER, 3.0
    elif key in CROSS:
        col, lw = TEAL, 1.8
    else:
        col, lw = GREY_L, 1.1
    edge(ax, POS[u], POS[v], color=col, lw=lw, directed=False,
         weight=w, wcolor=col if col != GREY_L else GREY_L, r=0.42)
for n, (x, y) in POS.items():
    node(ax, x, y, n)
text(ax, 1.6, 5.95, 'settled: A D F', color=TEAL_L, size=15)
text(ax, 5.55, 6.45, 'the frontier is a cut, and Prim buys the cheapest edge across it', color=AMBER, size=16)
legend(ax, 8.55, 3.55, [(AMBER, 'already in the tree'),
                        (TEAL, 'crosses the frontier'),
                        (GREY_L, 'both ends outside')])
text(ax, 9.75, 1.95, 'next purchase: AB (7)', color=TEAL_L, size=15)
save(fig, 'day16_1.png')

# ---------------------------------------------------------------- figure 2
# same skeleton, one lambda apart
fig, ax = canvas(9, 4.6, xlim=(0, 11.0), ylim=(0, 4.4))
box(ax, 3.55, 2.55, 3.9, 1.25,
    label='pop the cheapest\nunsettled vertex', lw=1.8)
text(ax, 5.5, 4.15, 'one loop, one heap, O(E log V)', color=TEAL_L, size=16)
arrow(ax, (4.4, 2.5), (2.6, 1.75), color=TEAL, lw=1.7)
arrow(ax, (6.6, 2.5), (8.4, 1.75), color=TEAL, lw=1.7)
box(ax, 0.35, 0.45, 4.0, 1.25, label='key = w\nPrim', edge=AMBER, lw=2.2)
box(ax, 6.65, 0.45, 4.0, 1.25, label='key = d + w\nDijkstra', edge=AMBER, lw=2.2)
text(ax, 2.35, 0.12, 'cost of the edge', color=TEAL, size=14)
text(ax, 8.65, 0.12, 'cost of the whole path', color=TEAL, size=14)
save(fig, 'day16_2.png')

# ---------------------------------------------------------------- figure 3
# the two trees differ
fig, ax = canvas(11, 6.0, xlim=(0, 17.6), ylim=(0, 6.6))
draw_graph(ax, highlight=MST, r=0.38, offset=(-0.55, 0.15), weights=True)
draw_graph(ax, highlight=SPT, r=0.38, offset=(8.35, 0.15), weights=True)
text(ax, 3.4, 6.25, 'Prim: total weight 39', color=AMBER, size=16)
text(ax, 12.3, 6.25, 'Dijkstra: total weight 44', color=AMBER, size=16)
text(ax, 3.4, 0.18, 'A to G along this tree costs 23', color=TEAL, size=14)
text(ax, 12.3, 0.18, 'A to G along this tree costs 22', color=TEAL, size=14)
save(fig, 'day16_3.png')

# ---------------------------------------------------------------- figure 4
# a negative edge breaks Dijkstra
fig, ax = canvas(9, 3.9, xlim=(0, 9.2), ylim=(0, 3.7))
NP = {'S': (1.0, 1.9), 'A': (4.6, 3.0), 'B': (4.6, 0.8)}
for u, v, w, col in [('S', 'A', 5, TEAL), ('S', 'B', 2, TEAL), ('A', 'B', -10, RED)]:
    edge(ax, NP[u], NP[v], color=col, lw=2.0, directed=True,
         weight=w, wcolor=col, r=0.42)
for n, (x, y) in NP.items():
    node(ax, x, y, n)
text(ax, 7.3, 2.55, 'Dijkstra says  B = 2', color=RED, size=16)
text(ax, 7.3, 1.75, 'the truth is   B = -5', color=AMBER, size=16)
text(ax, 4.6, 0.05, 'B is settled before the -10 edge is ever relaxed', color=TEAL, size=14)
save(fig, 'day16_4.png')
